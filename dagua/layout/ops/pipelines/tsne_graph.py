"""sklearn-compatible t-SNE on graph geodesic distances."""

from __future__ import annotations

import math
from numbers import Integral
from typing import Optional, Union

import numpy as np
import torch

from dagua.layout.ops.graph_utils import layout_device
from dagua.layout.ops.pipelines import resolve_fidelity_dtype

_SKLEARN_EXPLORATION_MAX_ITER = 250
_SKLEARN_N_ITER_CHECK = 50
_SKLEARN_EARLY_EXAGGERATION = 12.0
_SKLEARN_MIN_GRAD_NORM = 1.0e-7
_SKLEARN_N_ITER_WITHOUT_PROGRESS = 300
_SKLEARN_MACHINE_EPSILON = np.finfo(np.double).eps
_PERPLEXITY_BINARY_SEARCH_STEPS = 100
_PERPLEXITY_TOLERANCE = 1.0e-5
_PERPLEXITY_ZERO_SUM_EPSILON = float(np.float32(1.0e-8))


def _check_random_state(seed: Union[int, np.random.RandomState, None]) -> np.random.RandomState:
    """Return a NumPy ``RandomState`` using sklearn seed semantics.

    Parameters
    ----------
    seed : int or numpy.random.RandomState or None
        Random seed, existing random state, or ``None``.

    Returns
    -------
    numpy.random.RandomState
        Random state compatible with ``sklearn.utils.check_random_state``.

    Raises
    ------
    ValueError
        If ``seed`` cannot initialize a ``RandomState``.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand  # type: ignore[attr-defined]
    if isinstance(seed, (Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(f"{seed!r} cannot be used to seed a numpy.random.RandomState instance.")


def _graph_geodesic_distances(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor],
) -> np.ndarray:
    """Compute the competitor adapter's dense graph geodesic matrix.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.

    Returns
    -------
    numpy.ndarray
        Dense finite distance matrix with shape ``[N, N]`` and dtype
        ``float32``. Disconnected pairs use the adapter's global
        ``max(2 * max_finite_distance, 1)`` fill.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    if num_nodes == 0:
        return np.zeros((0, 0), dtype=np.float32)

    edge_index_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    if edge_index_cpu.numel() == 0:
        rows = np.empty(0, dtype=np.int64)
        cols = np.empty(0, dtype=np.int64)
    else:
        edge_index_np = edge_index_cpu.numpy()
        rows = np.concatenate([edge_index_np[0], edge_index_np[1]])
        cols = np.concatenate([edge_index_np[1], edge_index_np[0]])

    if edge_weights is None:
        data = np.ones(rows.shape[0], dtype=np.float32)
    else:
        weights = edge_weights.detach().to(device="cpu", dtype=torch.float32).numpy()
        data = np.concatenate([weights, weights]).astype(np.float32, copy=False)

    adjacency = csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    distances = shortest_path(adjacency, directed=False)
    finite_mask = np.isfinite(distances)
    max_finite = float(np.max(distances[finite_mask])) if np.any(finite_mask) else 1.0
    fill_value = max(max_finite * 2.0, 1.0)
    dense = np.where(np.isinf(distances), fill_value, distances)
    return dense.astype(np.float32, copy=False)


def _binary_search_conditional_probabilities(
    squared_distances: np.ndarray,
    desired_perplexity: float,
) -> np.ndarray:
    """Compute sklearn-style conditional Gaussian affinities.

    Parameters
    ----------
    squared_distances : numpy.ndarray
        Dense squared distance matrix with shape ``[N, N]``. Values are cast
        to ``float32`` to match sklearn's exact precomputed path.
    desired_perplexity : float
        Target perplexity for every row's conditional distribution.

    Returns
    -------
    numpy.ndarray
        Conditional probability matrix with shape ``[N, N]`` and dtype
        ``float64``.
    """
    distances = np.asarray(squared_distances, dtype=np.float32)
    num_nodes = int(distances.shape[0])
    probabilities = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    desired_entropy = math.log(desired_perplexity)

    for node in range(num_nodes):
        beta_min = -np.inf
        beta_max = np.inf
        beta = 1.0

        for _ in range(_PERPLEXITY_BINARY_SEARCH_STEPS):
            probability_sum = 0.0
            for neighbor in range(num_nodes):
                if neighbor != node:
                    probability = math.exp(-float(distances[node, neighbor]) * beta)
                    probabilities[node, neighbor] = probability
                    probability_sum += probability

            if probability_sum == 0.0:
                probability_sum = _PERPLEXITY_ZERO_SUM_EPSILON

            weighted_distance_sum = 0.0
            for neighbor in range(num_nodes):
                probabilities[node, neighbor] /= probability_sum
                weighted_distance_sum += (
                    float(distances[node, neighbor]) * probabilities[node, neighbor]
                )

            entropy = math.log(probability_sum) + beta * weighted_distance_sum
            entropy_difference = entropy - desired_entropy
            if abs(entropy_difference) <= _PERPLEXITY_TOLERANCE:
                break

            if entropy_difference > 0.0:
                beta_min = beta
                beta = beta * 2.0 if math.isinf(beta_max) else (beta + beta_max) / 2.0
            else:
                beta_max = beta
                beta = beta / 2.0 if math.isinf(beta_min) else (beta + beta_min) / 2.0

    return probabilities


def _joint_probabilities(distances: np.ndarray, desired_perplexity: float) -> np.ndarray:
    """Compute sklearn exact condensed joint probabilities.

    Parameters
    ----------
    distances : numpy.ndarray
        Dense squared distance matrix with shape ``[N, N]``.
    desired_perplexity : float
        Target perplexity for the conditional Gaussian affinities.

    Returns
    -------
    numpy.ndarray
        Condensed joint probability vector with shape ``[N * (N - 1) / 2]``.
    """
    from scipy.spatial.distance import squareform

    conditional_probabilities = _binary_search_conditional_probabilities(
        distances,
        desired_perplexity,
    )
    symmetric_probabilities = conditional_probabilities + conditional_probabilities.T
    probability_sum = max(float(np.sum(symmetric_probabilities)), _SKLEARN_MACHINE_EPSILON)
    condensed = squareform(symmetric_probabilities, checks=False) / probability_sum
    return np.maximum(condensed, _SKLEARN_MACHINE_EPSILON)


def _kl_divergence_exact(
    params: np.ndarray,
    probabilities: np.ndarray,
    degrees_of_freedom: int,
    num_nodes: int,
    n_components: int,
    skip_num_points: int = 0,
    compute_error: bool = True,
) -> tuple[float, np.ndarray]:
    """Evaluate sklearn exact t-SNE KL divergence and gradient.

    Parameters
    ----------
    params : numpy.ndarray
        Flattened embedding parameters with shape ``[N * C]``.
    probabilities : numpy.ndarray
        Condensed joint probability vector with shape ``[N * (N - 1) / 2]``.
    degrees_of_freedom : int
        Student-t degrees of freedom.
    num_nodes : int
        Number of graph nodes.
    n_components : int
        Embedding dimension.
    skip_num_points : int, default=0
        Leading points to keep fixed when computing gradients.
    compute_error : bool, default=True
        Whether to compute and return KL divergence.

    Returns
    -------
    tuple[float, numpy.ndarray]
        KL divergence and flattened gradient with shape ``[N * C]``.
    """
    from scipy.spatial.distance import pdist, squareform

    embedded = params.reshape(num_nodes, n_components)
    distances = pdist(embedded, "sqeuclidean")
    distances /= degrees_of_freedom
    distances += 1.0
    distances **= (degrees_of_freedom + 1.0) / -2.0
    q_values = np.maximum(distances / (2.0 * np.sum(distances)), _SKLEARN_MACHINE_EPSILON)

    if compute_error:
        kl_divergence = 2.0 * np.dot(
            probabilities,
            np.log(np.maximum(probabilities, _SKLEARN_MACHINE_EPSILON) / q_values),
        )
    else:
        kl_divergence = np.nan

    grad = np.ndarray((num_nodes, n_components), dtype=params.dtype)
    probability_delta = squareform((probabilities - q_values) * distances)
    for node in range(skip_num_points, num_nodes):
        grad[node] = np.dot(
            np.ravel(probability_delta[node], order="K"),
            embedded[node] - embedded,
        )
    grad = grad.ravel()
    grad *= 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    return float(kl_divergence), grad


def _gradient_descent_exact(
    params: np.ndarray,
    probabilities: np.ndarray,
    degrees_of_freedom: int,
    num_nodes: int,
    n_components: int,
    start_iter: int,
    max_iter: int,
    momentum: float,
    learning_rate: float,
    n_iter_without_progress: int,
    skip_num_points: int = 0,
) -> tuple[np.ndarray, float, int]:
    """Run sklearn's gains-and-momentum gradient descent loop.

    Parameters
    ----------
    params : numpy.ndarray
        Initial flattened parameters with shape ``[N * C]``.
    probabilities : numpy.ndarray
        Condensed joint probability vector with shape ``[N * (N - 1) / 2]``.
    degrees_of_freedom : int
        Student-t degrees of freedom.
    num_nodes : int
        Number of graph nodes.
    n_components : int
        Embedding dimension.
    start_iter : int
        First iteration index.
    max_iter : int
        Exclusive maximum iteration index.
    momentum : float
        Momentum coefficient.
    learning_rate : float
        Optimizer learning rate.
    n_iter_without_progress : int
        Checked iterations allowed without KL improvement.
    skip_num_points : int, default=0
        Leading points to keep fixed when computing gradients.

    Returns
    -------
    tuple[numpy.ndarray, float, int]
        Final flattened parameters, last checked KL divergence, and last
        iteration index.
    """
    from scipy import linalg

    params = params.copy().ravel()
    update = np.zeros_like(params)
    gains = np.ones_like(params)
    error = np.finfo(float).max
    best_error = np.finfo(float).max
    best_iter = iteration = start_iter

    for iteration in range(start_iter, max_iter):
        check_convergence = (iteration + 1) % _SKLEARN_N_ITER_CHECK == 0
        compute_error = check_convergence or iteration == max_iter - 1
        error, grad = _kl_divergence_exact(
            params,
            probabilities,
            degrees_of_freedom,
            num_nodes,
            n_components,
            skip_num_points=skip_num_points,
            compute_error=compute_error,
        )

        increase = update * grad < 0.0
        decrease = np.invert(increase)
        gains[increase] += 0.2
        gains[decrease] *= 0.8
        np.clip(gains, 0.01, np.inf, out=gains)
        grad *= gains
        update = momentum * update - learning_rate * grad
        params += update

        if check_convergence:
            grad_norm = linalg.norm(grad)
            if error < best_error:
                best_error = error
                best_iter = iteration
            elif iteration - best_iter > n_iter_without_progress:
                break
            if grad_norm <= _SKLEARN_MIN_GRAD_NORM:
                break

    return params, float(error), int(iteration)


def _resolve_max_iter(steps: int, max_iter: Optional[int]) -> int:
    """Resolve public iteration aliases to sklearn's ``max_iter``.

    Parameters
    ----------
    steps : int
        Dagua engine step count. Positive values are used when ``max_iter`` is
        not explicitly supplied.
    max_iter : int, optional
        sklearn-compatible maximum iteration count.

    Returns
    -------
    int
        Effective maximum iteration count, clamped to sklearn's minimum of
        250.

    Raises
    ------
    ValueError
        If either iteration count is negative.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if max_iter is not None and max_iter < 0:
        raise ValueError("max_iter must be non-negative.")
    requested = int(max_iter) if max_iter is not None else (int(steps) if steps > 0 else 1000)
    return max(requested, _SKLEARN_EXPLORATION_MAX_ITER)


def _resolve_learning_rate(
    learning_rate: Union[float, str],
    num_nodes: int,
    early_exaggeration: float,
) -> float:
    """Resolve sklearn's t-SNE learning-rate setting.

    Parameters
    ----------
    learning_rate : float or {"auto"}
        Learning-rate setting.
    num_nodes : int
        Number of graph nodes.
    early_exaggeration : float
        Early exaggeration multiplier.

    Returns
    -------
    float
        Numeric learning rate.

    Raises
    ------
    ValueError
        If ``learning_rate`` is unsupported or non-positive.
    """
    if isinstance(learning_rate, str):
        if learning_rate != "auto":
            raise ValueError("learning_rate must be positive or 'auto'.")
        return float(max(num_nodes / early_exaggeration / 4.0, 50.0))
    resolved = float(learning_rate)
    if resolved <= 0.0:
        raise ValueError("learning_rate must be positive or 'auto'.")
    return resolved


def _fit_tsne_exact(
    distances: np.ndarray,
    perplexity: float,
    max_iter: int,
    seed: Union[int, np.random.RandomState, None],
    learning_rate: Union[float, str],
    early_exaggeration: float,
) -> np.ndarray:
    """Fit exact t-SNE from precomputed graph distances.

    Parameters
    ----------
    distances : numpy.ndarray
        Dense finite graph distance matrix with shape ``[N, N]``.
    perplexity : float
        Target t-SNE perplexity, already clamped below ``N``.
    max_iter : int
        sklearn-compatible maximum optimizer iteration count.
    seed : int or numpy.random.RandomState or None
        Seed for sklearn-compatible random initialization.
    learning_rate : float or {"auto"}
        Learning-rate setting.
    early_exaggeration : float
        Early exaggeration multiplier for the first 250 iterations.

    Returns
    -------
    numpy.ndarray
        Raw t-SNE embedding with shape ``[N, 2]`` and dtype ``float32``.
    """
    num_nodes = int(distances.shape[0])
    n_components = 2
    resolved_learning_rate = _resolve_learning_rate(
        learning_rate,
        num_nodes,
        early_exaggeration,
    )

    squared_distances = np.asarray(distances, dtype=np.float32, order="C").copy()
    squared_distances **= 2
    probabilities = _joint_probabilities(squared_distances, perplexity)
    random_state = _check_random_state(seed)
    embedded = 1.0e-4 * random_state.standard_normal(size=(num_nodes, n_components)).astype(
        np.float32
    )
    params = embedded.ravel()
    degrees_of_freedom = max(n_components - 1, 1)

    probabilities *= early_exaggeration
    params, _, iteration = _gradient_descent_exact(
        params,
        probabilities,
        degrees_of_freedom,
        num_nodes,
        n_components,
        start_iter=0,
        max_iter=_SKLEARN_EXPLORATION_MAX_ITER,
        momentum=0.5,
        learning_rate=resolved_learning_rate,
        n_iter_without_progress=_SKLEARN_EXPLORATION_MAX_ITER,
    )

    probabilities /= early_exaggeration
    remaining = max_iter - _SKLEARN_EXPLORATION_MAX_ITER
    if iteration < _SKLEARN_EXPLORATION_MAX_ITER or remaining > 0:
        params, _, _ = _gradient_descent_exact(
            params,
            probabilities,
            degrees_of_freedom,
            num_nodes,
            n_components,
            start_iter=iteration + 1,
            max_iter=max_iter,
            momentum=0.8,
            learning_rate=resolved_learning_rate,
            n_iter_without_progress=_SKLEARN_N_ITER_WITHOUT_PROGRESS,
        )

    return params.reshape(num_nodes, n_components)


def layout_tsne_graph_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    perplexity: float = 30.0,
    learning_rate: Union[float, str] = "auto",
    max_iter: Optional[int] = None,
    steps: int = 0,
    seed: Optional[int] = 42,
    edge_weights: Optional[torch.Tensor] = None,
    early_exaggeration: float = _SKLEARN_EARLY_EXAGGERATION,
    method: str = "exact",
    init: str = "random",
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run sklearn-compatible t-SNE on graph geodesic distances.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]`` used only to choose
        output device.
    perplexity : float, default=30.0
        Target t-SNE perplexity, capped to ``N - 1`` like the competitor.
    learning_rate : float or {"auto"}, default="auto"
        sklearn t-SNE learning rate.
    max_iter : int, optional
        sklearn-compatible maximum iteration count. Overrides ``steps`` when
        provided.
    steps : int, default=0
        Dagua engine iteration alias used when ``max_iter`` is omitted.
    seed : int, optional
        Random seed. ``None`` preserves the adapter's historical default of
        ``42``.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.
    early_exaggeration : float, default=12.0
        sklearn early exaggeration multiplier.
    method : {"exact"}, default="exact"
        t-SNE method. Only exact is implemented because the pinned competitor
        config forces exact for graph-distance fidelity.
    init : {"random"}, default="random"
        Initialization mode. ``"pca"`` is invalid for sklearn precomputed
        distances and is intentionally not implemented here.
    fidelity_dtype : torch.dtype, optional
        Output dtype selector forwarded by the engine.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If inputs or unsupported sklearn options are requested.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if perplexity <= 0.0:
        raise ValueError("perplexity must be positive.")
    if early_exaggeration <= 0.0:
        raise ValueError("early_exaggeration must be positive.")
    if method != "exact":
        raise ValueError("tsne graph pipeline currently supports method='exact' only.")
    if init != "random":
        raise ValueError("tsne graph pipeline currently supports init='random' only.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
            )

    resolved_dtype = resolve_fidelity_dtype(False, fidelity_dtype)
    device = layout_device(edge_index, node_sizes)
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=resolved_dtype, device=device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=resolved_dtype, device=device)

    effective_max_iter = _resolve_max_iter(steps, max_iter)
    distances = _graph_geodesic_distances(edge_index, num_nodes, edge_weights)
    coordinates = _fit_tsne_exact(
        distances=distances,
        perplexity=min(float(perplexity), float(num_nodes - 1)),
        max_iter=effective_max_iter,
        seed=seed if seed is not None else 42,
        learning_rate=learning_rate,
        early_exaggeration=early_exaggeration,
    )
    return torch.tensor(coordinates, dtype=resolved_dtype, device=device)


def layout_tsne_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    perplexity: float = 30.0,
    learning_rate: Union[float, str] = "auto",
    max_iter: Optional[int] = None,
    steps: int = 0,
    seed: Optional[int] = 42,
    edge_weights: Optional[torch.Tensor] = None,
    early_exaggeration: float = _SKLEARN_EARLY_EXAGGERATION,
    method: str = "exact",
    init: str = "random",
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Alias ``algorithm='tsne'`` to the graph-geodesic t-SNE pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]`` used only to choose
        output device.
    perplexity : float, default=30.0
        Target t-SNE perplexity.
    learning_rate : float or {"auto"}, default="auto"
        sklearn t-SNE learning rate.
    max_iter : int, optional
        sklearn-compatible maximum iteration count.
    steps : int, default=0
        Dagua engine iteration alias.
    seed : int, optional
        Random seed.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.
    early_exaggeration : float, default=12.0
        sklearn early exaggeration multiplier.
    method : {"exact"}, default="exact"
        t-SNE method.
    init : {"random"}, default="random"
        Initialization mode.
    fidelity_dtype : torch.dtype, optional
        Output dtype selector forwarded by the engine.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.
    """
    return layout_tsne_graph_pipeline(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=max_iter,
        steps=steps,
        seed=seed,
        edge_weights=edge_weights,
        early_exaggeration=early_exaggeration,
        method=method,
        init=init,
        fidelity_dtype=fidelity_dtype,
    )


__all__ = [
    "_binary_search_conditional_probabilities",
    "_graph_geodesic_distances",
    "_joint_probabilities",
    "_kl_divergence_exact",
    "layout_tsne_graph_pipeline",
    "layout_tsne_pipeline",
]
