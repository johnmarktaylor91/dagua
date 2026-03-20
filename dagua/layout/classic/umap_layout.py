"""UMAP embedding for graph-theoretic distances."""

from __future__ import annotations

from collections import deque
from math import log2
from typing import Optional

import numpy as np
import torch
from scipy import optimize, sparse
from scipy.sparse import linalg as sparse_linalg

_EPSILON = 1.0e-9
_MIN_SPAN = 1.0e-6
_MIN_SIGMA_SCALE = 1.0e-3
_SMOOTH_K_TOLERANCE = 1.0e-5
_SMOOTH_K_BINARY_SEARCH_STEPS = 64
_SPECTRAL_SPARSE_THRESHOLD = 512
_GRADIENT_CLIP_VALUE = 4.0
_NEGATIVE_SAMPLE_RATE = 5


def _layout_device(
    edge_index: torch.Tensor,
    node_sizes: Optional[torch.Tensor],
) -> torch.device:
    """Choose the device used for the returned tensor.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor.

    Returns
    -------
    torch.device
        Output device for the final layout tensor.
    """
    if edge_index.numel() > 0:
        return edge_index.device
    if node_sizes is not None:
        return node_sizes.device
    return torch.device("cpu")


def _layout_extent(num_nodes: int, node_sizes: Optional[torch.Tensor]) -> float:
    """Estimate a stable output extent from graph size and node sizes.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor.

    Returns
    -------
    float
        Target half-width after normalization.
    """
    if node_sizes is None or node_sizes.numel() == 0:
        return max(float(max(num_nodes, 1)) ** 0.5 * 5.0, 1.0)

    max_size = float(node_sizes.to(dtype=torch.float32, device="cpu").max().item())
    return max(max_size * max(float(max(num_nodes, 1)) ** 0.5, 1.0) * 2.0, 1.0)


def _normalize_positions(positions: torch.Tensor, extent: float) -> torch.Tensor:
    """Center and scale coordinates into a deterministic bounding box.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    extent : float
        Target half-width.

    Returns
    -------
    torch.Tensor
        Centered and scaled coordinates.
    """
    if positions.shape[0] <= 1:
        return torch.zeros_like(positions)

    centered = positions - positions.mean(dim=0, keepdim=True)
    span = float(centered.abs().max().item())
    if span < _MIN_SPAN:
        centered = centered.clone()
        centered[:, 0] = torch.linspace(
            -1.0,
            1.0,
            steps=positions.shape[0],
            device=positions.device,
        )
        span = float(centered.abs().max().item())
    return centered * (extent / max(span, _MIN_SPAN))


def _validate_inputs(edge_index: torch.Tensor, num_nodes: int, n_neighbors: int) -> None:
    """Validate public UMAP layout arguments.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    n_neighbors : int
        Target neighborhood size for the fuzzy simplicial set.

    Returns
    -------
    None
        Raises ``ValueError`` when an input is invalid.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive.")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")
    if edge_index.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError("edge_index must use an integer dtype.")
    if edge_index.numel() == 0:
        return
    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    if int(edge_index_cpu.min().item()) < 0:
        raise ValueError("edge_index cannot contain negative node indices.")
    if int(edge_index_cpu.max().item()) >= num_nodes:
        raise ValueError("edge_index contains node indices outside num_nodes.")


def _build_undirected_adjacency(edge_index: torch.Tensor, num_nodes: int) -> list[list[int]]:
    """Build an undirected adjacency list from ``edge_index``.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    list[list[int]]
        Sorted undirected neighbor lists.
    """
    adjacency_sets = [set() for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return [[] for _ in range(num_nodes)]

    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    for source, target in zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist()):
        if source == target:
            continue
        adjacency_sets[source].add(target)
        adjacency_sets[target].add(source)

    return [sorted(neighbors) for neighbors in adjacency_sets]


def _bfs_distances(adjacency: list[list[int]], start: int) -> torch.Tensor:
    """Compute unweighted shortest-path distances from one source.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency list.
    start : int
        Source node index.

    Returns
    -------
    torch.Tensor
        Distance vector with shape ``[N]``. Unreachable nodes are left at
        ``inf`` so the caller can apply a single symmetric fill value.
    """
    num_nodes = len(adjacency)
    distances = torch.full((num_nodes,), float("inf"), dtype=torch.float32)
    distances[start] = 0.0
    queue: deque[int] = deque([start])

    while queue:
        node = queue.popleft()
        next_distance = float(distances[node].item() + 1.0)
        for neighbor in adjacency[node]:
            if bool(torch.isfinite(distances[neighbor]).item()):
                continue
            distances[neighbor] = next_distance
            queue.append(neighbor)
    return distances


def _all_pairs_shortest_paths(adjacency: list[list[int]]) -> torch.Tensor:
    """Compute all-pairs graph distances with repeated BFS.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency list.

    Returns
    -------
    torch.Tensor
        Distance matrix with shape ``[N, N]``.
    """
    if not adjacency:
        return torch.empty((0, 0), dtype=torch.float32)
    rows = [_bfs_distances(adjacency=adjacency, start=index) for index in range(len(adjacency))]
    distances = torch.stack(rows, dim=0)
    finite_mask = torch.isfinite(distances)
    max_finite = (
        float(distances[finite_mask].max().item()) if bool(finite_mask.any().item()) else 1.0
    )
    fill_value = max(max_finite * 2.0, 1.0)
    return torch.where(finite_mask, distances, torch.full_like(distances, fill_value))


def _knn_from_distances(
    distances: torch.Tensor,
    n_neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract k-nearest neighbors from a dense distance matrix.

    Parameters
    ----------
    distances : torch.Tensor
        Distance matrix with shape ``[N, N]``.
    n_neighbors : int
        Number of neighbors per row.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Neighbor indices and distances with shapes ``[N, K]``.
    """
    num_nodes = distances.shape[0]
    if num_nodes == 0:
        empty = torch.empty((0, 0), dtype=torch.long)
        return empty, empty.to(dtype=torch.float32)

    k = min(n_neighbors, max(num_nodes - 1, 1))
    adjusted = distances.clone()
    diagonal = torch.eye(num_nodes, dtype=torch.bool)
    adjusted = adjusted.masked_fill(diagonal, float("inf"))
    knn_distances, knn_indices = torch.topk(adjusted, k=k, largest=False, dim=1)
    return knn_indices.to(dtype=torch.long), knn_distances.to(dtype=torch.float32)


def _smooth_knn_dist(
    knn_distances: torch.Tensor,
    n_neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve the UMAP smooth-kNN bandwidth for every graph node.

    Parameters
    ----------
    knn_distances : torch.Tensor
        Neighbor distances with shape ``[N, K]``.
    n_neighbors : int
        Neighborhood size ``k``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``sigma`` and ``rho`` vectors with shape ``[N]``.
    """
    num_nodes = knn_distances.shape[0]
    if num_nodes == 0:
        empty = torch.empty((0,), dtype=torch.float32)
        return empty, empty

    sigmas = torch.empty((num_nodes,), dtype=torch.float32)
    rhos = torch.empty((num_nodes,), dtype=torch.float32)
    target = log2(float(max(n_neighbors, 2)))

    for index in range(num_nodes):
        distances = knn_distances[index]
        finite = distances[torch.isfinite(distances)]
        if finite.numel() == 0:
            sigmas[index] = 1.0
            rhos[index] = 0.0
            continue

        rho = float(finite[0].item())
        rhos[index] = rho
        mean_distance = max(float(finite.mean().item()), _MIN_SPAN)
        sigma_min = mean_distance * _MIN_SIGMA_SCALE
        lower = 0.0
        upper = 1.0

        def _membership_sum(sigma: float) -> float:
            if sigma <= 0.0:
                return float(finite.numel())
            shifted = torch.clamp(finite - rho, min=0.0)
            values = torch.exp(-shifted / sigma)
            return float(values.sum().item())

        while _membership_sum(upper) < target:
            upper *= 2.0
            if upper > 1.0e6:
                break

        sigma = upper
        for _ in range(_SMOOTH_K_BINARY_SEARCH_STEPS):
            sigma = 0.5 * (lower + upper)
            estimate = _membership_sum(max(sigma, sigma_min))
            if abs(estimate - target) <= _SMOOTH_K_TOLERANCE:
                break
            if estimate > target:
                upper = sigma
            else:
                lower = sigma

        sigmas[index] = max(sigma, sigma_min)

    return sigmas, rhos


def _symmetrized_fuzzy_graph(
    knn_indices: torch.Tensor,
    knn_distances: torch.Tensor,
    sigmas: torch.Tensor,
    rhos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the symmetrized fuzzy simplicial set used by graph UMAP.

    Parameters
    ----------
    knn_indices : torch.Tensor
        Neighbor indices with shape ``[N, K]``.
    knn_distances : torch.Tensor
        Neighbor distances with shape ``[N, K]``.
    sigmas : torch.Tensor
        Smooth-kNN bandwidths with shape ``[N]``.
    rhos : torch.Tensor
        Local connectivity radii with shape ``[N]``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Undirected fuzzy graph edges ``(head, tail, weight)``.
    """
    directed_weights: dict[tuple[int, int], float] = {}
    num_nodes, num_neighbors = knn_indices.shape

    for row in range(num_nodes):
        sigma = float(sigmas[row].item())
        rho = float(rhos[row].item())
        for column in range(num_neighbors):
            neighbor = int(knn_indices[row, column].item())
            distance = float(knn_distances[row, column].item())
            if not np.isfinite(distance):
                continue
            if distance <= rho or sigma <= 0.0:
                weight = 1.0
            else:
                weight = float(np.exp(-(distance - rho) / sigma))
            directed_weights[(row, neighbor)] = weight

    undirected: dict[tuple[int, int], float] = {}
    handled: set[tuple[int, int]] = set()
    for source, target in directed_weights:
        key = (min(source, target), max(source, target))
        if key in handled or source == target:
            continue
        handled.add(key)
        forward = directed_weights.get((source, target), 0.0)
        backward = directed_weights.get((target, source), 0.0)
        weight = forward + backward - (forward * backward)
        if weight > 0.0:
            undirected[key] = weight

    if not undirected:
        return (
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
        )

    pairs = list(undirected.keys())
    weights = list(undirected.values())
    head = torch.tensor([pair[0] for pair in pairs], dtype=torch.long)
    tail = torch.tensor([pair[1] for pair in pairs], dtype=torch.long)
    weight = torch.tensor(weights, dtype=torch.float32)
    return head, tail, weight


def _curve_function(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Evaluate UMAP's smooth low-dimensional membership curve.

    Parameters
    ----------
    x : numpy.ndarray
        Non-negative squared distances.
    a : float
        Fitted UMAP ``a`` parameter.
    b : float
        Fitted UMAP ``b`` parameter.

    Returns
    -------
    numpy.ndarray
        Curve values for the supplied distances.
    """
    return 1.0 / (1.0 + (a * np.power(x, 2.0 * b)))


def _fit_ab(min_dist: float, spread: float) -> tuple[float, float]:
    """Fit UMAP's ``a`` and ``b`` curve parameters from ``min_dist``.

    Parameters
    ----------
    min_dist : float
        Target minimum distance in the embedding.
    spread : float
        Embedding spread scale.

    Returns
    -------
    tuple[float, float]
        Fitted curve parameters.
    """
    xv = np.linspace(0.0, 3.0 * spread, 300)
    yv = np.where(xv < min_dist, 1.0, np.exp(-(xv - min_dist) / spread))
    try:
        params, _ = optimize.curve_fit(
            _curve_function,
            xv,
            yv,
            p0=(1.93, 0.79),
            maxfev=10_000,
        )
        return float(params[0]), float(params[1])
    except (RuntimeError, ValueError):
        return 1.93, 0.79


def _spectral_initialization(
    num_nodes: int,
    head: torch.Tensor,
    tail: torch.Tensor,
    weight: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    """Compute the normalized-Laplacian spectral initialization.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    head : torch.Tensor
        Edge heads with shape ``[E]``.
    tail : torch.Tensor
        Edge tails with shape ``[E]``.
    weight : torch.Tensor
        Symmetric edge weights with shape ``[E]``.
    seed : int
        Seed for the small initialization noise.

    Returns
    -------
    torch.Tensor
        Initial embedding with shape ``[N, 2]``.
    """
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32)
    if num_nodes == 2:
        return torch.tensor([[-10.0, 0.0], [10.0, 0.0]], dtype=torch.float32)

    if head.numel() == 0:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return (torch.rand((num_nodes, 2), generator=generator, dtype=torch.float32) - 0.5) * 20.0

    rows = torch.cat([head, tail]).numpy()
    cols = torch.cat([tail, head]).numpy()
    data = torch.cat([weight, weight]).numpy().astype(np.float64, copy=False)
    graph = sparse.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.float64)

    degree = np.asarray(graph.sum(axis=1)).reshape(-1)
    inv_sqrt_degree = np.zeros_like(degree)
    nonzero = degree > 0.0
    inv_sqrt_degree[nonzero] = 1.0 / np.sqrt(degree[nonzero])
    d_inv_sqrt = sparse.diags(inv_sqrt_degree)
    laplacian = sparse.identity(num_nodes, dtype=np.float64) - (d_inv_sqrt @ graph @ d_inv_sqrt)

    if num_nodes < _SPECTRAL_SPARSE_THRESHOLD:
        dense_laplacian = laplacian.toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(dense_laplacian)
    else:
        eigenvalues, eigenvectors = sparse_linalg.eigsh(laplacian, k=3, which="SM")

    order = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, order]
    coordinates = np.real(eigenvectors[:, 1:3])
    if coordinates.shape[1] == 1:
        coordinates = np.concatenate(
            [coordinates, np.zeros((num_nodes, 1), dtype=coordinates.dtype)],
            axis=1,
        )

    min_value = float(coordinates.min())
    max_value = float(coordinates.max())
    if max_value - min_value > _MIN_SPAN:
        coordinates = ((coordinates - min_value) / (max_value - min_value) * 20.0) - 10.0
    else:
        coordinates = np.zeros((num_nodes, 2), dtype=np.float32)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    noise = torch.randn((num_nodes, 2), generator=generator, dtype=torch.float32) * 1.0e-4
    return torch.from_numpy(coordinates.astype(np.float32, copy=False)) + noise


def _select_positive_edges(
    head: torch.Tensor,
    tail: torch.Tensor,
    weight: torch.Tensor,
    n_epochs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prune very weak edges and build epoch sampling intervals.

    Parameters
    ----------
    head : torch.Tensor
        Edge heads with shape ``[E]``.
    tail : torch.Tensor
        Edge tails with shape ``[E]``.
    weight : torch.Tensor
        Edge weights with shape ``[E]``.
    n_epochs : int
        Number of SGD epochs.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Pruned ``head``, ``tail``, and ``epochs_per_sample`` tensors.
    """
    if weight.numel() == 0:
        empty_long = torch.empty((0,), dtype=torch.long)
        empty_float = torch.empty((0,), dtype=torch.float32)
        return empty_long, empty_long, empty_float

    max_weight = float(weight.max().item())
    min_weight = max_weight / float(max(n_epochs, 1))
    keep = weight >= min_weight
    kept_head = head[keep]
    kept_tail = tail[keep]
    kept_weight = weight[keep]
    epochs_per_sample = max_weight / kept_weight
    return kept_head, kept_tail, epochs_per_sample.to(dtype=torch.float32)


def _positive_gradient(diff: torch.Tensor, distance_sq: float, a: float, b: float) -> torch.Tensor:
    """Compute the clipped attractive UMAP gradient for one positive edge.

    Parameters
    ----------
    diff : torch.Tensor
        Coordinate difference ``y_i - y_j`` with shape ``[2]``.
    distance_sq : float
        Squared Euclidean distance between ``y_i`` and ``y_j``.
    a : float
        UMAP curve parameter.
    b : float
        UMAP curve parameter.

    Returns
    -------
    torch.Tensor
        Clipped gradient contribution with shape ``[2]``.
    """
    if distance_sq <= 0.0:
        return torch.zeros_like(diff)
    grad_coeff = -2.0 * a * b * (distance_sq ** (b - 1.0)) / ((a * (distance_sq**b)) + 1.0)
    return torch.clamp(grad_coeff * diff, min=-_GRADIENT_CLIP_VALUE, max=_GRADIENT_CLIP_VALUE)


def _negative_gradient(
    diff: torch.Tensor,
    distance_sq: float,
    a: float,
    b: float,
    gamma: float,
) -> torch.Tensor:
    """Compute the clipped repulsive UMAP gradient for one negative sample.

    Parameters
    ----------
    diff : torch.Tensor
        Coordinate difference ``y_i - y_k`` with shape ``[2]``.
    distance_sq : float
        Squared Euclidean distance between ``y_i`` and ``y_k``.
    a : float
        UMAP curve parameter.
    b : float
        UMAP curve parameter.
    gamma : float
        Negative-sample repulsion strength.

    Returns
    -------
    torch.Tensor
        Clipped gradient contribution with shape ``[2]``.
    """
    if distance_sq <= 0.0:
        return torch.zeros_like(diff)
    grad_coeff = 2.0 * gamma * b / ((0.001 + distance_sq) * ((a * (distance_sq**b)) + 1.0))
    return torch.clamp(grad_coeff * diff, min=-_GRADIENT_CLIP_VALUE, max=_GRADIENT_CLIP_VALUE)


def _optimize_embedding(
    embedding: torch.Tensor,
    head: torch.Tensor,
    tail: torch.Tensor,
    epochs_per_sample: torch.Tensor,
    n_epochs: int,
    learning_rate: float,
    negative_sample_rate: int,
    gamma: float,
    a: float,
    b: float,
    seed: int,
) -> torch.Tensor:
    """Run the UMAP cross-entropy SGD with negative sampling.

    Parameters
    ----------
    embedding : torch.Tensor
        Initial embedding with shape ``[N, 2]``.
    head : torch.Tensor
        Positive edge heads with shape ``[E]``.
    tail : torch.Tensor
        Positive edge tails with shape ``[E]``.
    epochs_per_sample : torch.Tensor
        Edge sampling intervals with shape ``[E]``.
    n_epochs : int
        Number of SGD epochs.
    learning_rate : float
        Initial SGD learning rate.
    negative_sample_rate : int
        Number of negative samples per positive edge.
    gamma : float
        Repulsion strength for negative samples.
    a : float
        UMAP curve parameter.
    b : float
        UMAP curve parameter.
    seed : int
        Seed for deterministic negative sampling.

    Returns
    -------
    torch.Tensor
        Optimized embedding with shape ``[N, 2]``.
    """
    if head.numel() == 0 or n_epochs <= 0:
        return embedding

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    next_sample_epoch = torch.zeros_like(epochs_per_sample)
    epochs_per_negative_sample = epochs_per_sample / float(max(negative_sample_rate, 1))
    next_negative_epoch = torch.zeros_like(epochs_per_negative_sample)
    num_nodes = embedding.shape[0]

    for epoch in range(n_epochs):
        alpha = learning_rate * (1.0 - (float(epoch) / float(max(n_epochs, 1))))
        for edge_id in range(head.shape[0]):
            if float(next_sample_epoch[edge_id].item()) > float(epoch):
                continue

            source = int(head[edge_id].item())
            target = int(tail[edge_id].item())
            diff = embedding[source] - embedding[target]
            distance_sq = float(torch.dot(diff, diff).item())
            grad = _positive_gradient(diff=diff, distance_sq=distance_sq, a=a, b=b)
            embedding[source] = embedding[source] + (alpha * grad)
            embedding[target] = embedding[target] - (alpha * grad)
            next_sample_epoch[edge_id] = next_sample_epoch[edge_id] + epochs_per_sample[edge_id]

            if negative_sample_rate <= 0:
                continue

            negatives = 0
            while float(next_negative_epoch[edge_id].item()) <= float(epoch):
                negative = int(torch.randint(0, num_nodes, (1,), generator=generator).item())
                negative_diff = embedding[source] - embedding[negative]
                negative_distance_sq = float(torch.dot(negative_diff, negative_diff).item())
                negative_grad = _negative_gradient(
                    diff=negative_diff,
                    distance_sq=negative_distance_sq,
                    a=a,
                    b=b,
                    gamma=gamma,
                )
                embedding[source] = embedding[source] + (alpha * negative_grad)
                next_negative_epoch[edge_id] = (
                    next_negative_epoch[edge_id] + epochs_per_negative_sample[edge_id]
                )
                negatives += 1
                if negatives >= negative_sample_rate:
                    break

    return embedding


def layout_umap(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    spread: float = 1.0,
    n_epochs: Optional[int] = None,
    learning_rate: float = 1.0,
    negative_sample_rate: int = _NEGATIVE_SAMPLE_RATE,
    repulsion_strength: float = 1.0,
    seed: int = 42,
) -> torch.Tensor:
    """Lay out a graph with UMAP applied to shortest-path distances.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor used only for output scaling.
    n_neighbors : int, default=15
        Neighborhood size for the fuzzy simplicial set.
    min_dist : float, default=0.1
        Target minimum distance in the embedding.
    spread : float, default=1.0
        Target spread for the low-dimensional curve fit.
    n_epochs : int, optional
        Number of SGD epochs. Defaults to ``500`` when ``num_nodes <= 10000``
        and ``200`` otherwise.
    learning_rate : float, default=1.0
        Initial SGD learning rate.
    negative_sample_rate : int, default=5
        Negative samples per positive edge.
    repulsion_strength : float, default=1.0
        Repulsive weight ``gamma`` for negative samples.
    seed : int, default=42
        Random seed for spectral noise and negative sampling.

    Returns
    -------
    torch.Tensor
        Final positions with shape ``[N, 2]``.
    """
    _validate_inputs(edge_index=edge_index, num_nodes=num_nodes, n_neighbors=n_neighbors)
    if min_dist < 0.0:
        raise ValueError("min_dist must be non-negative.")
    if spread <= 0.0:
        raise ValueError("spread must be positive.")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if negative_sample_rate < 0:
        raise ValueError("negative_sample_rate must be non-negative.")
    if repulsion_strength < 0.0:
        raise ValueError("repulsion_strength must be non-negative.")
    if n_epochs is not None and n_epochs < 0:
        raise ValueError("n_epochs must be non-negative.")

    device = _layout_device(edge_index=edge_index, node_sizes=node_sizes)
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32, device=device)

    adjacency = _build_undirected_adjacency(edge_index=edge_index, num_nodes=num_nodes)
    distances = _all_pairs_shortest_paths(adjacency=adjacency)
    knn_indices, knn_distances = _knn_from_distances(distances=distances, n_neighbors=n_neighbors)
    sigmas, rhos = _smooth_knn_dist(knn_distances=knn_distances, n_neighbors=n_neighbors)
    head, tail, weight = _symmetrized_fuzzy_graph(
        knn_indices=knn_indices,
        knn_distances=knn_distances,
        sigmas=sigmas,
        rhos=rhos,
    )

    embedding = _spectral_initialization(
        num_nodes=num_nodes,
        head=head,
        tail=tail,
        weight=weight,
        seed=seed,
    )
    a, b = _fit_ab(min_dist=min_dist, spread=spread)

    epoch_count = n_epochs if n_epochs is not None else (500 if num_nodes <= 10_000 else 200)
    positive_head, positive_tail, epochs_per_sample = _select_positive_edges(
        head=head,
        tail=tail,
        weight=weight,
        n_epochs=epoch_count,
    )
    embedding = _optimize_embedding(
        embedding=embedding,
        head=positive_head,
        tail=positive_tail,
        epochs_per_sample=epochs_per_sample,
        n_epochs=epoch_count,
        learning_rate=learning_rate,
        negative_sample_rate=negative_sample_rate,
        gamma=repulsion_strength,
        a=a,
        b=b,
        seed=seed,
    )

    extent = _layout_extent(num_nodes=num_nodes, node_sizes=node_sizes)
    normalized = _normalize_positions(positions=embedding, extent=extent)
    return normalized.to(dtype=torch.float32, device=device)
