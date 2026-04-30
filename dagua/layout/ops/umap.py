"""Registered UMAP operations for composable layouts."""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass
from math import log2
from typing import ClassVar, Optional, Tuple, Union, cast

import numpy as np
import torch
from scipy import optimize, sparse
from scipy.sparse import linalg as sparse_linalg

from dagua.layout.ops.base import Op
from dagua.layout.ops.graph_utils import (
    layout_device,
    layout_extent,
)
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_EPSILON = 1.0e-9
_MIN_SPAN = 1.0e-6
_MIN_SIGMA_SCALE = 1.0e-3
_SMOOTH_K_TOLERANCE = 1.0e-5
_SMOOTH_K_BINARY_SEARCH_STEPS = 64
_SPECTRAL_SPARSE_THRESHOLD = 512
_GRADIENT_CLIP_VALUE = 4.0
_NEGATIVE_SAMPLE_RATE = 5

_KNN_INDICES_KEY = "umap_knn_indices"
_KNN_DISTANCES_KEY = "umap_knn_distances"
_SIGMAS_KEY = "umap_sigmas"
_RHOS_KEY = "umap_rhos"
_FUZZY_HEAD_KEY = "umap_fuzzy_head"
_FUZZY_TAIL_KEY = "umap_fuzzy_tail"
_FUZZY_WEIGHT_KEY = "umap_fuzzy_weight"
_CURVE_A_KEY = "umap_curve_a"
_CURVE_B_KEY = "umap_curve_b"
_POSITIVE_HEAD_KEY = "umap_positive_head"
_POSITIVE_TAIL_KEY = "umap_positive_tail"
_EPOCHS_PER_SAMPLE_KEY = "umap_epochs_per_sample"
_N_EPOCHS_KEY = "umap_n_epochs"
_N_NEIGHBORS_KEY = "umap_n_neighbors"
_LEARNING_RATE_KEY = "umap_learning_rate"
_NEGATIVE_SAMPLE_RATE_KEY = "umap_negative_sample_rate"
_GAMMA_KEY = "umap_gamma"
_MIN_DIST_KEY = "umap_min_dist"
_SPREAD_KEY = "umap_spread"


@dataclass(frozen=True)
class ValidateUMAPInputsConfig:
    """Configuration for :class:`ValidateUMAPInputs`.

    Parameters
    ----------
    n_neighbors : int, default=15
        Expected neighborhood size for downstream UMAP preprocessing.
    """

    n_neighbors: int = 15


@dataclass(frozen=True)
class StoreUMAPHyperparametersConfig:
    """Configuration for :class:`StoreUMAPHyperparameters`.

    Parameters
    ----------
    n_neighbors : int, default=15
        Number of nearest neighbors in the fuzzy simplicial graph.
    min_dist : float, default=0.1
        Target minimum low-dimensional distance.
    spread : float, default=1.0
        Effective radius of the embedding curve.
    n_epochs : int | None, default=None
        Optimization epoch count. ``None`` enables the small/large heuristic.
    learning_rate : float, default=1.0
        Initial SGD step size.
    negative_sample_rate : int, default=5
        Number of negative samples per positive edge update.
    repulsion_strength : float, default=1.0
        Gamma multiplier for negative samples.
    default_epochs_small : int, default=500
        Heuristic epoch count for graphs up to ``large_graph_threshold``.
    default_epochs_large : int, default=200
        Heuristic epoch count for graphs above ``large_graph_threshold``.
    large_graph_threshold : int, default=10_000
        Node-count boundary for the epoch heuristic.
    """

    n_neighbors: int = 15
    min_dist: float = 0.1
    spread: float = 1.0
    n_epochs: Optional[int] = None
    learning_rate: float = 1.0
    negative_sample_rate: int = 5
    repulsion_strength: float = 1.0
    default_epochs_small: int = 500
    default_epochs_large: int = 200
    large_graph_threshold: int = 10_000


def _build_undirected_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor] = None,
) -> list[list[tuple[int, float]]]:
    """Build an undirected adjacency list from ``edge_index``."""
    if edge_weights is None:
        adjacency_sets: list[set[int]] = [set() for _ in range(num_nodes)]
        if edge_index.numel() == 0:
            return [[] for _ in range(num_nodes)]

        edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
        for source, target in zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist()):
            if source == target:
                continue
            adjacency_sets[source].add(target)
            adjacency_sets[target].add(source)

        return [[(neighbor, 1.0) for neighbor in sorted(neighbors)] for neighbors in adjacency_sets]

    adjacency_maps: list[dict[int, float]] = [dict() for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return [sorted(neighbors.items()) for neighbors in adjacency_maps]

    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    weights_cpu = edge_weights.detach().to(device="cpu", dtype=torch.float32)
    for edge_id, (source, target) in enumerate(
        zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist())
    ):
        if source == target:
            continue
        cost = 1.0 / max(float(weights_cpu[edge_id].item()), _EPSILON)
        previous = adjacency_maps[source].get(target)
        adjacency_maps[source][target] = min(previous, cost) if previous is not None else cost
        previous = adjacency_maps[target].get(source)
        adjacency_maps[target][source] = min(previous, cost) if previous is not None else cost

    return [sorted(neighbors.items()) for neighbors in adjacency_maps]


def _undirected_edge_weight_lookup(
    edge_index: torch.Tensor,
    edge_weights: Optional[torch.Tensor],
) -> dict[tuple[int, int], float]:
    """Build an undirected edge-weight lookup table from the input graph."""
    if edge_weights is None or edge_index.numel() == 0:
        return {}

    lookup: dict[tuple[int, int], float] = {}
    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    weights_cpu = edge_weights.detach().to(device="cpu", dtype=torch.float32)
    for edge_id, (source, target) in enumerate(
        zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist())
    ):
        if source == target:
            continue
        pair = (min(source, target), max(source, target))
        lookup[pair] = lookup.get(pair, 0.0) + float(weights_cpu[edge_id].item())
    return lookup


def _bfs_distances(
    adjacency: list[list[int]],
    start: int,
) -> torch.Tensor:
    """Compute unweighted shortest-path distances from one source."""
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


def _dijkstra_distances(
    adjacency: list[list[tuple[int, float]]],
    start: int,
) -> torch.Tensor:
    """Compute weighted shortest-path distances from one source."""
    num_nodes = len(adjacency)
    distances = torch.full((num_nodes,), float("inf"), dtype=torch.float32)
    distances[start] = 0.0
    heap: list[tuple[float, int]] = [(0.0, start)]
    while heap:
        distance, node = heapq.heappop(heap)
        if distance > float(distances[node].item()):
            continue
        for neighbor, cost in adjacency[node]:
            candidate = distance + cost
            if candidate >= float(distances[neighbor].item()):
                continue
            distances[neighbor] = candidate
            heapq.heappush(heap, (candidate, neighbor))
    return distances


def _all_pairs_shortest_paths(
    adjacency: Union[list[list[int]], list[list[tuple[int, float]]]],
) -> torch.Tensor:
    """Compute all-pairs graph distances with repeated BFS or Dijkstra."""
    if not adjacency:
        return torch.empty((0, 0), dtype=torch.float32)
    is_weighted = any(neighbors and isinstance(neighbors[0], tuple) for neighbors in adjacency)
    if is_weighted:
        weighted_adjacency = cast(list[list[tuple[int, float]]], adjacency)
        rows = [
            _dijkstra_distances(adjacency=weighted_adjacency, start=index)
            for index in range(len(weighted_adjacency))
        ]
    else:
        unweighted_adjacency = cast(list[list[int]], adjacency)
        rows = [
            _bfs_distances(adjacency=unweighted_adjacency, start=index)
            for index in range(len(unweighted_adjacency))
        ]
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
        Dense shortest-path distance matrix with shape ``[N, N]``.
    n_neighbors : int
        Number of neighbors requested by UMAP, including the self neighbor for
        precomputed dense inputs.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Neighbor indices and distances, each with shape ``[N, K]``. Rows are
        sorted using stable distance order so tied graph distances follow the
        same index-order behavior as umap-learn's precomputed path.
    """
    num_nodes = distances.shape[0]
    if num_nodes == 0:
        empty = torch.empty((0, 0), dtype=torch.long)
        return empty, empty.to(dtype=torch.float32)

    k = min(n_neighbors, num_nodes)
    distances_np = distances.detach().to(device="cpu", dtype=torch.float32).numpy()
    # UMAP's dense precomputed path keeps the zero self-distance in the sorted
    # neighborhood and uses stable mergesort, which preserves index order for
    # the many tied distances found in unweighted graph shortest paths.
    knn_indices_np = np.argsort(distances_np, axis=1, kind="mergesort")[:, :k]
    row_indices = np.arange(num_nodes)[:, None]
    knn_distances_np = distances_np[row_indices, knn_indices_np]
    knn_indices = torch.from_numpy(knn_indices_np.copy()).to(dtype=torch.long)
    knn_distances = torch.from_numpy(knn_distances_np.copy()).to(dtype=torch.float32)
    return knn_indices.to(dtype=torch.long), knn_distances.to(dtype=torch.float32)


def _smooth_knn_dist(
    knn_distances: torch.Tensor,
    n_neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve the UMAP smooth-kNN bandwidth for every graph node."""
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

        positive = finite[finite > 0]
        rho = float(positive.min().item()) if positive.numel() > 0 else 0.0
        rhos[index] = rho
        mean_distance = max(float(finite.mean().item()), _MIN_SPAN)
        sigma_min = mean_distance * _MIN_SIGMA_SCALE
        lower = 0.0
        upper = 1.0

        def _membership_sum(sigma: float) -> float:
            if sigma <= 0.0:
                return float(finite[1:].numel())
            shifted = torch.clamp(finite[1:] - rho, min=0.0)
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
    """Build the symmetrized fuzzy simplicial set used by graph UMAP."""
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
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    return head, tail, weight_tensor


def _curve_function(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Evaluate UMAP's smooth low-dimensional membership curve."""
    return 1.0 / (1.0 + (a * np.power(x, 2.0 * b)))


def _fit_ab(min_dist: float, spread: float) -> tuple[float, float]:
    """Fit UMAP's ``a`` and ``b`` curve parameters from ``min_dist``."""
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
    """Compute the normalized-Laplacian spectral initialization."""
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
    """Prune weak edges and build epoch sampling intervals."""
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


def _positive_gradient(
    diff: torch.Tensor,
    distance_sq: float,
    a: float,
    b: float,
) -> torch.Tensor:
    """Compute the clipped attractive UMAP gradient for one positive edge."""
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
    """Compute the clipped repulsive UMAP gradient for one negative sample."""
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
    """Run the UMAP cross-entropy SGD with negative sampling."""
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


def _normalize_positions(positions: torch.Tensor, extent: float) -> torch.Tensor:
    """Center and scale coordinates into a deterministic bounding box."""
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


@register_op
class ValidateUMAPInputs(Op):
    """Validate the graph inputs required by the graph-UMAP pipeline."""

    config: ValidateUMAPInputsConfig

    name: ClassVar[str] = "validate_umap_inputs"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = ()
    access_pattern: ClassVar[str] = "global"

    def __init__(
        self,
        n_neighbors: int = 15,
        *,
        config: Optional[ValidateUMAPInputsConfig] = None,
    ) -> None:
        """Store graph-validation requirements.

        Parameters
        ----------
        n_neighbors : int, default=15
            Expected neighborhood size for downstream UMAP preprocessing.
        config : ValidateUMAPInputsConfig | None, optional
            Optional validation configuration. When provided, it takes
            precedence over ``n_neighbors``.
        """
        self.config = config or ValidateUMAPInputsConfig(n_neighbors=n_neighbors)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Validate edge index, node count, edge weights, and neighborhood size."""
        del ctx

        if problem.num_nodes < 0:
            raise ValueError("num_nodes must be non-negative.")
        if self.config.n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive.")
        if problem.edge_index.ndim != 2 or problem.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E].")
        if problem.edge_index.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ValueError("edge_index must use an integer dtype.")
        if problem.edge_weights is not None:
            if problem.edge_weights.ndim != 1:
                raise ValueError("edge_weights must have shape [E].")
            if problem.edge_weights.shape[0] != problem.edge_index.shape[1]:
                raise ValueError(
                    f"edge_weights length {problem.edge_weights.shape[0]} does not match "
                    f"edge count {problem.edge_index.shape[1]}"
                )
        if problem.edge_index.numel() == 0:
            return state
        edge_index_cpu = problem.edge_index.to(device="cpu", dtype=torch.long)
        if int(edge_index_cpu.min().item()) < 0:
            raise ValueError("edge_index cannot contain negative node indices.")
        if int(edge_index_cpu.max().item()) >= problem.num_nodes:
            raise ValueError("edge_index contains node indices outside num_nodes.")
        return state


@register_op
class StoreUMAPHyperparameters(Op):
    """Persist validated UMAP hyperparameters into ``state.extras``."""

    config: StoreUMAPHyperparametersConfig

    name: ClassVar[str] = "umap_store_hyperparameters"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    writes: ClassVar[Tuple[str, ...]] = (
        f"extras.{_N_NEIGHBORS_KEY}",
        f"extras.{_MIN_DIST_KEY}",
        f"extras.{_SPREAD_KEY}",
        f"extras.{_N_EPOCHS_KEY}",
        f"extras.{_LEARNING_RATE_KEY}",
        f"extras.{_NEGATIVE_SAMPLE_RATE_KEY}",
        f"extras.{_GAMMA_KEY}",
    )
    access_pattern: ClassVar[str] = "global"

    def __init__(
        self,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        spread: float = 1.0,
        n_epochs: Optional[int] = None,
        learning_rate: float = 1.0,
        negative_sample_rate: int = _NEGATIVE_SAMPLE_RATE,
        repulsion_strength: float = 1.0,
        default_epochs_small: int = 500,
        default_epochs_large: int = 200,
        large_graph_threshold: int = 10_000,
        *,
        config: Optional[StoreUMAPHyperparametersConfig] = None,
    ) -> None:
        """Store pipeline hyperparameters.

        Parameters
        ----------
        n_neighbors : int, default=15
            Number of nearest neighbors in the fuzzy simplicial graph.
        min_dist : float, default=0.1
            Target minimum low-dimensional distance.
        spread : float, default=1.0
            Effective radius of the embedding curve.
        n_epochs : int | None, optional
            Optimization epoch count. ``None`` enables the small/large heuristic.
        learning_rate : float, default=1.0
            Initial SGD step size.
        negative_sample_rate : int, default=5
            Number of negative samples per positive edge update.
        repulsion_strength : float, default=1.0
            Gamma multiplier for negative samples.
        default_epochs_small : int, default=500
            Heuristic epoch count for smaller graphs.
        default_epochs_large : int, default=200
            Heuristic epoch count for larger graphs.
        large_graph_threshold : int, default=10_000
            Node-count boundary for the epoch heuristic.
        config : StoreUMAPHyperparametersConfig | None, optional
            Optional hyperparameter configuration. When provided, it takes
            precedence over the scalar arguments.
        """
        self.config = config or StoreUMAPHyperparametersConfig(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            spread=spread,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            negative_sample_rate=negative_sample_rate,
            repulsion_strength=repulsion_strength,
            default_epochs_small=default_epochs_small,
            default_epochs_large=default_epochs_large,
            large_graph_threshold=large_graph_threshold,
        )

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Write all validated hyperparameters into solve-state extras."""
        del ctx

        if self.config.n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive.")
        if self.config.min_dist < 0.0:
            raise ValueError("min_dist must be non-negative.")
        if self.config.spread <= 0.0:
            raise ValueError("spread must be positive.")
        if self.config.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.config.negative_sample_rate < 0:
            raise ValueError("negative_sample_rate must be non-negative.")
        if self.config.repulsion_strength < 0.0:
            raise ValueError("repulsion_strength must be non-negative.")
        if self.config.n_epochs is not None and self.config.n_epochs < 0:
            raise ValueError("n_epochs must be non-negative.")

        n_epochs = self.config.n_epochs
        if n_epochs is None:
            n_epochs = (
                self.config.default_epochs_small
                if problem.num_nodes <= self.config.large_graph_threshold
                else self.config.default_epochs_large
            )

        state.extras[_N_NEIGHBORS_KEY] = self.config.n_neighbors
        state.extras[_MIN_DIST_KEY] = self.config.min_dist
        state.extras[_SPREAD_KEY] = self.config.spread
        state.extras[_N_EPOCHS_KEY] = n_epochs
        state.extras[_LEARNING_RATE_KEY] = self.config.learning_rate
        state.extras[_NEGATIVE_SAMPLE_RATE_KEY] = self.config.negative_sample_rate
        state.extras[_GAMMA_KEY] = self.config.repulsion_strength
        return state


@register_op
class BuildUMAPAdjacency(Op):
    """Build an undirected adjacency list from graph edges."""

    name: ClassVar[str] = "umap_build_adjacency"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = ("adjacency_weighted",)
    access_pattern: ClassVar[str] = "global"

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build an undirected weighted adjacency list and store it on the state."""
        del ctx

        state.adjacency_weighted = _build_undirected_adjacency(
            edge_index=problem.edge_index,
            num_nodes=problem.num_nodes,
            edge_weights=problem.edge_weights,
        )
        return state


@register_op
class ComputeAllPairsShortestPaths(Op):
    """Compute all-pairs graph distances via BFS or Dijkstra."""

    name: ClassVar[str] = "umap_all_pairs_shortest_paths"
    category: ClassVar[OpCategory] = OpCategory.DISTANCE
    reads: ClassVar[Tuple[str, ...]] = ("adjacency_weighted",)
    writes: ClassVar[Tuple[str, ...]] = ("distance_matrix",)
    requires: ClassVar[Tuple[str, ...]] = ("adjacency_weighted",)
    access_pattern: ClassVar[str] = "global"

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute and cache the dense shortest-path matrix."""
        del problem, ctx
        adjacency = state.adjacency_weighted
        if adjacency is None:
            raise ValueError("ComputeAllPairsShortestPaths requires state.adjacency_weighted.")
        state.distance_matrix = _all_pairs_shortest_paths(adjacency=adjacency)
        return state


@register_op
class ExtractKNN(Op):
    """Extract k-nearest neighbors from the dense distance matrix."""

    name: ClassVar[str] = "umap_extract_knn"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("distance_matrix", f"extras.{_N_NEIGHBORS_KEY}")
    writes: ClassVar[Tuple[str, ...]] = (
        f"extras.{_KNN_INDICES_KEY}",
        f"extras.{_KNN_DISTANCES_KEY}",
    )
    requires: ClassVar[Tuple[str, ...]] = ("distance_matrix", f"extras.{_N_NEIGHBORS_KEY}")
    access_pattern: ClassVar[str] = "global"

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Extract kNN indices and distances from cached graph distances."""
        del problem, ctx
        distances = state.distance_matrix
        if distances is None:
            raise ValueError("ExtractKNN requires state.distance_matrix.")
        n_neighbors = state.extras[_N_NEIGHBORS_KEY]
        knn_indices, knn_distances = _knn_from_distances(
            distances=distances,
            n_neighbors=n_neighbors,
        )
        state.extras[_KNN_INDICES_KEY] = knn_indices
        state.extras[_KNN_DISTANCES_KEY] = knn_distances
        return state


@register_op
class SmoothKNNDistances(Op):
    """Solve UMAP smooth-kNN bandwidth per node."""

    name: ClassVar[str] = "umap_smooth_knn_dist"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = (
        f"extras.{_KNN_DISTANCES_KEY}",
        f"extras.{_N_NEIGHBORS_KEY}",
    )
    writes: ClassVar[Tuple[str, ...]] = (
        f"extras.{_SIGMAS_KEY}",
        f"extras.{_RHOS_KEY}",
    )
    requires: ClassVar[Tuple[str, ...]] = (
        f"extras.{_KNN_DISTANCES_KEY}",
        f"extras.{_N_NEIGHBORS_KEY}",
    )
    access_pattern: ClassVar[str] = "global"

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Populate ``umap_sigmas`` and ``umap_rhos`` from kNN distance rows."""
        del problem, ctx
        knn_distances = state.extras[_KNN_DISTANCES_KEY]
        n_neighbors = state.extras[_N_NEIGHBORS_KEY]
        sigmas, rhos = _smooth_knn_dist(
            knn_distances=knn_distances,
            n_neighbors=n_neighbors,
        )
        state.extras[_SIGMAS_KEY] = sigmas
        state.extras[_RHOS_KEY] = rhos
        return state


@register_op
class BuildFuzzySimplicialSet(Op):
    """Build the symmetrized fuzzy simplicial-set graph."""

    name: ClassVar[str] = "umap_build_fuzzy_set"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = (
        f"extras.{_KNN_INDICES_KEY}",
        f"extras.{_KNN_DISTANCES_KEY}",
        f"extras.{_SIGMAS_KEY}",
        f"extras.{_RHOS_KEY}",
    )
    writes: ClassVar[Tuple[str, ...]] = (
        f"extras.{_FUZZY_HEAD_KEY}",
        f"extras.{_FUZZY_TAIL_KEY}",
        f"extras.{_FUZZY_WEIGHT_KEY}",
    )
    requires: ClassVar[Tuple[str, ...]] = (
        f"extras.{_KNN_INDICES_KEY}",
        f"extras.{_KNN_DISTANCES_KEY}",
        f"extras.{_SIGMAS_KEY}",
        f"extras.{_RHOS_KEY}",
    )
    access_pattern: ClassVar[str] = "global"

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build weighted fuzzy simplicial-set edges and scale by edge weights."""
        del ctx
        knn_indices = state.extras[_KNN_INDICES_KEY]
        knn_distances = state.extras[_KNN_DISTANCES_KEY]
        sigmas = state.extras[_SIGMAS_KEY]
        rhos = state.extras[_RHOS_KEY]

        head, tail, weight = _symmetrized_fuzzy_graph(
            knn_indices=knn_indices,
            knn_distances=knn_distances,
            sigmas=sigmas,
            rhos=rhos,
        )

        if problem.edge_weights is not None and weight.numel() > 0:
            edge_weight_lookup = _undirected_edge_weight_lookup(
                edge_index=problem.edge_index,
                edge_weights=problem.edge_weights,
            )
            scaled_weight = weight.clone()
            for index in range(weight.shape[0]):
                # Preserve the classic behavior where explicit graph weights
                # rescale the fuzzy-set membership after symmetrization.
                pair = (
                    min(int(head[index].item()), int(tail[index].item())),
                    max(int(head[index].item()), int(tail[index].item())),
                )
                scaled_weight[index] = scaled_weight[index] * edge_weight_lookup.get(pair, 1.0)
            weight = scaled_weight

        state.extras[_FUZZY_HEAD_KEY] = head
        state.extras[_FUZZY_TAIL_KEY] = tail
        state.extras[_FUZZY_WEIGHT_KEY] = weight
        return state


@register_op
class SpectralInitialization(Op):
    """Compute spectral initialization for the UMAP embedding."""

    name: ClassVar[str] = "umap_spectral_init"
    category: ClassVar[OpCategory] = OpCategory.INIT
    reads: ClassVar[Tuple[str, ...]] = (
        f"extras.{_FUZZY_HEAD_KEY}",
        f"extras.{_FUZZY_TAIL_KEY}",
        f"extras.{_FUZZY_WEIGHT_KEY}",
    )
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = (
        f"extras.{_FUZZY_HEAD_KEY}",
        f"extras.{_FUZZY_TAIL_KEY}",
        f"extras.{_FUZZY_WEIGHT_KEY}",
    )
    access_pattern: ClassVar[str] = "global"

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build initial positions from the fuzzy simplicial graph Laplacian."""
        del ctx
        head = state.extras[_FUZZY_HEAD_KEY]
        tail = state.extras[_FUZZY_TAIL_KEY]
        weight = state.extras[_FUZZY_WEIGHT_KEY]
        state.pos = _spectral_initialization(
            num_nodes=problem.num_nodes,
            head=head,
            tail=tail,
            weight=weight,
            seed=problem.seed,
        )
        return state


@register_op
class FitCurveParameters(Op):
    """Fit UMAP ``a`` and ``b`` curve parameters."""

    name: ClassVar[str] = "umap_fit_curve"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = (
        f"extras.{_MIN_DIST_KEY}",
        f"extras.{_SPREAD_KEY}",
    )
    writes: ClassVar[Tuple[str, ...]] = (
        f"extras.{_CURVE_A_KEY}",
        f"extras.{_CURVE_B_KEY}",
    )
    requires: ClassVar[Tuple[str, ...]] = (
        f"extras.{_MIN_DIST_KEY}",
        f"extras.{_SPREAD_KEY}",
    )
    access_pattern: ClassVar[str] = "global"

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Fit and persist ``umap_curve_a`` and ``umap_curve_b``."""
        del problem, ctx
        min_dist = state.extras[_MIN_DIST_KEY]
        spread = state.extras[_SPREAD_KEY]
        a, b = _fit_ab(min_dist=min_dist, spread=spread)
        state.extras[_CURVE_A_KEY] = a
        state.extras[_CURVE_B_KEY] = b
        return state


@register_op
class SelectPositiveEdges(Op):
    """Prune weak edges and build per-edge sample intervals."""

    name: ClassVar[str] = "umap_select_positive_edges"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = (
        f"extras.{_FUZZY_HEAD_KEY}",
        f"extras.{_FUZZY_TAIL_KEY}",
        f"extras.{_FUZZY_WEIGHT_KEY}",
        f"extras.{_N_EPOCHS_KEY}",
    )
    writes: ClassVar[Tuple[str, ...]] = (
        f"extras.{_POSITIVE_HEAD_KEY}",
        f"extras.{_POSITIVE_TAIL_KEY}",
        f"extras.{_EPOCHS_PER_SAMPLE_KEY}",
    )
    requires: ClassVar[Tuple[str, ...]] = (
        f"extras.{_FUZZY_HEAD_KEY}",
        f"extras.{_FUZZY_TAIL_KEY}",
        f"extras.{_FUZZY_WEIGHT_KEY}",
        f"extras.{_N_EPOCHS_KEY}",
    )
    access_pattern: ClassVar[str] = "global"

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Select positive edges and per-edge epoch intervals for optimization."""
        del problem, ctx
        head = state.extras[_FUZZY_HEAD_KEY]
        tail = state.extras[_FUZZY_TAIL_KEY]
        weight = state.extras[_FUZZY_WEIGHT_KEY]
        n_epochs = state.extras[_N_EPOCHS_KEY]
        positive_head, positive_tail, epochs_per_sample = _select_positive_edges(
            head=head,
            tail=tail,
            weight=weight,
            n_epochs=n_epochs,
        )
        state.extras[_POSITIVE_HEAD_KEY] = positive_head
        state.extras[_POSITIVE_TAIL_KEY] = positive_tail
        state.extras[_EPOCHS_PER_SAMPLE_KEY] = epochs_per_sample
        return state


@register_op
class OptimizeUMAPEmbedding(Op):
    """Run the UMAP cross-entropy optimizer."""

    name: ClassVar[str] = "umap_optimize_embedding"
    category: ClassVar[OpCategory] = OpCategory.OPTIMIZE
    reads: ClassVar[Tuple[str, ...]] = (
        "pos",
        f"extras.{_POSITIVE_HEAD_KEY}",
        f"extras.{_POSITIVE_TAIL_KEY}",
        f"extras.{_EPOCHS_PER_SAMPLE_KEY}",
        f"extras.{_N_EPOCHS_KEY}",
        f"extras.{_LEARNING_RATE_KEY}",
        f"extras.{_NEGATIVE_SAMPLE_RATE_KEY}",
        f"extras.{_GAMMA_KEY}",
        f"extras.{_CURVE_A_KEY}",
        f"extras.{_CURVE_B_KEY}",
    )
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)
    access_pattern: ClassVar[str] = "global"

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run deterministic UMAP SGD updates with optional negative sampling."""
        del ctx
        if state.pos is None:
            raise ValueError("OptimizeUMAPEmbedding requires state.pos to be set.")

        state.pos = _optimize_embedding(
            embedding=state.pos,
            head=state.extras[_POSITIVE_HEAD_KEY],
            tail=state.extras[_POSITIVE_TAIL_KEY],
            epochs_per_sample=state.extras[_EPOCHS_PER_SAMPLE_KEY],
            n_epochs=state.extras[_N_EPOCHS_KEY],
            learning_rate=state.extras[_LEARNING_RATE_KEY],
            negative_sample_rate=state.extras[_NEGATIVE_SAMPLE_RATE_KEY],
            gamma=state.extras[_GAMMA_KEY],
            a=state.extras[_CURVE_A_KEY],
            b=state.extras[_CURVE_B_KEY],
            seed=problem.seed,
        )
        return state


@register_op
class FinalizeUMAPPositions(Op):
    """Apply deterministic centering and scaling to final positions."""

    name: ClassVar[str] = "umap_finalize_positions"
    category: ClassVar[OpCategory] = OpCategory.POSTPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)
    access_pattern: ClassVar[str] = "global"

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Normalize positions and cast output device/dtype."""
        del ctx
        if state.pos is None:
            raise ValueError("FinalizeUMAPPositions requires state.pos to be set.")
        device = layout_device(edge_index=problem.edge_index, node_sizes=problem.node_sizes)
        extent = layout_extent(num_nodes=problem.num_nodes, node_sizes=problem.node_sizes)
        normalized = _normalize_positions(state.pos, extent=extent)
        state.pos = normalized.to(dtype=torch.float32, device=device)
        return state


__all__ = [
    "ValidateUMAPInputsConfig",
    "ValidateUMAPInputs",
    "StoreUMAPHyperparametersConfig",
    "StoreUMAPHyperparameters",
    "BuildUMAPAdjacency",
    "ComputeAllPairsShortestPaths",
    "ExtractKNN",
    "SmoothKNNDistances",
    "BuildFuzzySimplicialSet",
    "SpectralInitialization",
    "FitCurveParameters",
    "SelectPositiveEdges",
    "OptimizeUMAPEmbedding",
    "FinalizeUMAPPositions",
]
