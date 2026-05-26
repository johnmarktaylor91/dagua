"""tsNET layout pipeline."""

from __future__ import annotations

from numbers import Integral
from typing import Optional, Union

import numpy as np
import torch

from dagua.layout.ops.base import Pipeline, Repeat  # noqa: E402
from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig  # noqa: E402
from dagua.layout.ops.graph_utils import layout_device
from dagua.layout.ops.pipelines import resolve_fidelity_dtype
from dagua.layout.ops.state import (  # noqa: E402
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)
from dagua.layout.ops.tsnet import (  # noqa: E402
    TsnetFinalizePositions,
    TsnetGradientStep,
    TsnetInitializeOptimizer,
    TsnetInitializePositions,
    TsnetInitializePositionsConfig,
    TsnetPrepareState,
)

_SKLEARN_EXPLORATION_MAX_ITER = 250
_SKLEARN_N_ITER_CHECK = 50
_SKLEARN_EARLY_EXAGGERATION = 12.0
_SKLEARN_MIN_GRAD_NORM = 1.0e-7
_SKLEARN_N_ITER_WITHOUT_PROGRESS = 300
_SKLEARN_MACHINE_EPSILON = np.finfo(np.double).eps


def _uses_sklearn_exact_fidelity(fidelity_mode: Union[bool, str]) -> bool:
    """Return whether the public wrapper should run exact sklearn fidelity.

    Parameters
    ----------
    fidelity_mode : bool | str
        Fidelity selector. ``True``, ``"sklearn"``, and ``"exact"`` enable the
        local sklearn exact t-SNE port; ``False``, ``"native"``, and ``"torch"``
        use the native torch pipeline.

    Returns
    -------
    bool
        ``True`` when the sklearn exact fidelity path should run.

    Raises
    ------
    ValueError
        If ``fidelity_mode`` is an unsupported string value.
    """
    if isinstance(fidelity_mode, str):
        normalized = fidelity_mode.lower()
        if normalized in {"sklearn", "exact"}:
            return True
        if normalized in {"false", "native", "torch"}:
            return False
        raise ValueError(
            "fidelity_mode must be a bool or one of 'sklearn', 'exact', 'native', or 'torch'."
        )
    return bool(fidelity_mode)


def _check_random_state(seed: Union[int, np.random.RandomState, None]) -> np.random.RandomState:
    """Turn a seed into a NumPy ``RandomState`` using sklearn semantics.

    Parameters
    ----------
    seed : int | numpy.random.RandomState | None
        Random seed, existing ``RandomState``, or ``None``.

    Returns
    -------
    numpy.random.RandomState
        NumPy MT19937 random state matching ``sklearn.utils.check_random_state``.

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
        Flattened embedding parameter array with shape ``[N * C]``.
    probabilities : numpy.ndarray
        Condensed joint probability matrix with shape ``[N * (N - 1) / 2]``.
    degrees_of_freedom : int
        Degrees of freedom for the Student t distribution.
    num_nodes : int
        Number of nodes ``N``.
    n_components : int
        Embedding dimension ``C``.
    skip_num_points : int, default=0
        Number of leading points to keep fixed when computing gradients.
    compute_error : bool, default=True
        Whether to compute and return the KL objective value.

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
    """Run sklearn's batch gradient descent loop for exact t-SNE.

    Parameters
    ----------
    params : numpy.ndarray
        Initial flattened embedding parameters with shape ``[N * C]``.
    probabilities : numpy.ndarray
        Condensed joint probability matrix with shape ``[N * (N - 1) / 2]``.
    degrees_of_freedom : int
        Degrees of freedom for the Student t distribution.
    num_nodes : int
        Number of nodes ``N``.
    n_components : int
        Embedding dimension ``C``.
    start_iter : int
        First optimizer iteration index.
    max_iter : int
        Exclusive maximum optimizer iteration index.
    momentum : float
        Momentum coefficient for this optimizer phase.
    learning_rate : float
        Learning rate used by sklearn's ``learning_rate="auto"`` schedule.
    n_iter_without_progress : int
        Stop after this many checked iterations without KL improvement.
    skip_num_points : int, default=0
        Number of leading points to keep fixed when computing gradients.

    Returns
    -------
    tuple[numpy.ndarray, float, int]
        Final flattened parameters, last checked error, and last iteration.
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


def _fit_tsnet_exact_condensed(
    distances: np.ndarray,
    perplexity: float,
    steps: int,
    seed: int,
) -> np.ndarray:
    """Fit exact t-SNE from precomputed graph distances without calling TSNE.

    Parameters
    ----------
    distances : numpy.ndarray
        Dense graph-distance matrix with shape ``[N, N]``. Values are squared
        in-place to match sklearn's ``metric="precomputed"`` exact path.
    perplexity : float
        Target t-SNE perplexity, already clamped below ``N``.
    steps : int
        Requested max iteration count. Values below 250 are raised to 250 to
        preserve the existing dagua fidelity-wrapper behavior.
    seed : int
        Seed for sklearn-compatible NumPy ``RandomState`` initialization.

    Returns
    -------
    numpy.ndarray
        Raw t-SNE embedding with shape ``[N, 2]`` and dtype ``float32``.
    """
    from sklearn.manifold._t_sne import _joint_probabilities

    num_nodes = int(distances.shape[0])
    n_components = 2
    max_iter = max(int(steps), _SKLEARN_EXPLORATION_MAX_ITER)
    learning_rate = max(num_nodes / _SKLEARN_EARLY_EXAGGERATION / 4.0, 50.0)

    distances **= 2
    probabilities = _joint_probabilities(distances, perplexity, 0)
    random_state = _check_random_state(seed)
    embedded = 1.0e-4 * random_state.standard_normal(size=(num_nodes, n_components)).astype(
        np.float32
    )
    params = embedded.ravel()
    degrees_of_freedom = max(n_components - 1, 1)

    probabilities *= _SKLEARN_EARLY_EXAGGERATION
    params, _, iteration = _gradient_descent_exact(
        params,
        probabilities,
        degrees_of_freedom,
        num_nodes,
        n_components,
        start_iter=0,
        max_iter=_SKLEARN_EXPLORATION_MAX_ITER,
        momentum=0.5,
        learning_rate=learning_rate,
        n_iter_without_progress=_SKLEARN_EXPLORATION_MAX_ITER,
    )

    probabilities /= _SKLEARN_EARLY_EXAGGERATION
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
            learning_rate=learning_rate,
            n_iter_without_progress=_SKLEARN_N_ITER_WITHOUT_PROGRESS,
        )

    return params.reshape(num_nodes, n_components)


def _layout_tsnet_sklearn_reference(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor],
    perplexity: float,
    steps: int,
    seed: int,
    edge_weights: Optional[torch.Tensor],
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run a local sklearn exact t-SNE reference port for fidelity mode.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]`` used only to choose the
        output device.
    perplexity : float
        Target t-SNE perplexity.
    steps : int
        Maximum sklearn optimization iterations.
    seed : int
        Random seed forwarded to ``sklearn.manifold.TSNE``.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.
    fidelity_dtype : torch.dtype, default=torch.float32
        Internal dtype for the precomputed distance matrix.

    Returns
    -------
    torch.Tensor
        Reference coordinates with shape ``[N, 2]`` on the layout device.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    resolved_dtype = resolve_fidelity_dtype(True, fidelity_dtype)
    device = layout_device(edge_index, node_sizes)
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=resolved_dtype, device=device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=resolved_dtype, device=device)

    edge_index_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    if edge_index_cpu.numel() == 0:
        rows = np.empty(0, dtype=np.int64)
        cols = np.empty(0, dtype=np.int64)
    else:
        edge_index_np = edge_index_cpu.numpy()
        rows = np.concatenate([edge_index_np[0], edge_index_np[1]])
        cols = np.concatenate([edge_index_np[1], edge_index_np[0]])
    if edge_weights is None:
        np_dtype = np.float64 if resolved_dtype is torch.float64 else np.float32
        data = np.ones(rows.shape[0], dtype=np_dtype)
    else:
        torch_dtype = torch.float64 if resolved_dtype is torch.float64 else torch.float32
        weights = edge_weights.detach().to(device="cpu", dtype=torch_dtype).numpy()
        data = np.concatenate([weights, weights]).astype(
            np.float64 if resolved_dtype is torch.float64 else np.float32,
            copy=False,
        )

    adjacency = csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    distances = shortest_path(adjacency, directed=False)
    finite_mask = np.isfinite(distances)
    max_finite = float(np.max(distances[finite_mask])) if np.any(finite_mask) else 1.0
    fill_value = max(max_finite * 2.0, 1.0)
    dense_distances = np.where(np.isinf(distances), fill_value, distances).astype(
        np.float64 if resolved_dtype is torch.float64 else np.float32,
        copy=False,
    )

    coordinates = _fit_tsnet_exact_condensed(
        distances=dense_distances,
        perplexity=min(float(perplexity), float(num_nodes - 1)),
        steps=steps,
        seed=seed,
    )
    return torch.tensor(coordinates, dtype=resolved_dtype, device=device)


def build_tsnet_pipeline(
    steps: int = 1000,
    fidelity_mode: Union[bool, str] = False,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> Pipeline:
    """Build a tsNET layout pipeline.

    Reference fidelity
    ------------------
    Targets: scikit-learn 1.8.0 t-SNE graph adapter / van der Maaten and
        Hinton (2008), "Visualizing Data using t-SNE".
    Fidelity mode: ``fidelity_mode=True`` in the public wrapper routes through
        sklearn's exact t-SNE implementation; this builder still exposes the
        native torch composition for direct pipeline tests and diagnostics.
    Verified at: round_32 bounded subset median RMSD 0.398822; final
        100-seed report marks TSNET variants partial match at median RMSD
        0.151 to 0.276.
    Known divergences:
        - The native torch composition remains close but not bit-exact because
          sklearn uses SciPy/NumPy condensed-distance probability and gradient
          kernels plus its own two-call optimizer loop.
        - The Round 31 ``c=4`` gradient-scale hypothesis was reverted after
          direct gradient parity checks.

    Parameters
    ----------
    steps : int, default=1000
        Number of optimization updates.
    fidelity_mode : bool | str, default=False
        Preserve native sklearn-diagnostic settings when this builder is used
        directly.
    fidelity_dtype : torch.dtype, default=torch.float32
        Internal dtype used only when ``fidelity_mode`` is enabled.

    Returns
    -------
    Pipeline
        Pipeline implementing the tsNET algorithm. The pipeline produces final
        node coordinates by initializing positions, preparing t-SNE-style
        affinities, creating the optimizer state, applying repeated
        gains-and-momentum gradient steps, and finalizing the embedding.

    Raises
    ------
    ValueError
        If ``steps`` is negative.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    exact_fidelity = _uses_sklearn_exact_fidelity(fidelity_mode)
    resolved_dtype = resolve_fidelity_dtype(fidelity_mode, fidelity_dtype)
    dtype = resolved_dtype if exact_fidelity else torch.float32

    return Pipeline(
        [
            FixedSteps(FixedStepsConfig(n=steps)),
            TsnetInitializePositions(
                TsnetInitializePositionsConfig(fidelity_mode=exact_fidelity, dtype=dtype)
            ),
            TsnetPrepareState(),
            TsnetInitializeOptimizer(),
            Repeat(
                n=steps,
                ops=[
                    TsnetGradientStep(),
                ],
            ),
            TsnetFinalizePositions(),
        ],
        name="tsnet_pipeline",
    )


def layout_tsnet_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    perplexity: float = 30,
    steps: int = 1000,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    fidelity_mode: Union[bool, str] = False,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run the tsNET pipeline as a drop-in replacement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]`` used for final
        scaling.
    perplexity : float, default=30
        Target t-SNE perplexity. Currently only the default value of 30
        preserves bit-identity with classic; non-default values require
        extending ``TsnetPrepareState``.
    steps : int, default=1000
        Number of optimization updates.
    seed : int, default=42
        Random seed for the torch generator initialization.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.
    fidelity_mode : bool | str, default=False
        Route through the local sklearn exact t-SNE reference port when
        ``True``, ``"sklearn"``, or ``"exact"``.
    fidelity_dtype : torch.dtype, default=torch.float32
        Internal dtype used only when ``fidelity_mode`` is enabled.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``steps``, ``perplexity``, or ``edge_weights``
        are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if perplexity <= 0:
        raise ValueError("perplexity must be positive.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    resolved_dtype = resolve_fidelity_dtype(fidelity_mode, fidelity_dtype)
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
            )

    if _uses_sklearn_exact_fidelity(fidelity_mode):
        return _layout_tsnet_sklearn_reference(
            edge_index=edge_index,
            num_nodes=num_nodes,
            node_sizes=node_sizes,
            perplexity=perplexity,
            steps=steps,
            seed=seed,
            edge_weights=edge_weights,
            fidelity_dtype=resolved_dtype,
        )

    device = layout_device(edge_index, node_sizes)

    # Handle trivial cases exactly like classic.
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32, device=device)

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    state.extras["tsnet_perplexity"] = perplexity
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_tsnet_pipeline(
        steps=steps,
        fidelity_mode=fidelity_mode,
        fidelity_dtype=resolved_dtype,
    ).apply(
        problem,
        state,
        ctx,
    )
    if final_state.pos is None:
        raise RuntimeError("tsNET pipeline did not produce final positions.")
    return final_state.pos.to(dtype=torch.float32)


__all__ = ["build_tsnet_pipeline", "layout_tsnet_pipeline"]
