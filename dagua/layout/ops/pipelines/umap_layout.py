"""UMAP graph layout expressed as a composable ops pipeline."""

from __future__ import annotations

import heapq
from collections import deque
from math import log2
from typing import ClassVar, Optional, Tuple, Union, cast

import numpy as np
import torch
from scipy import optimize, sparse
from scipy.sparse import linalg as sparse_linalg

from dagua.layout.ops.base import Op, Pipeline  # noqa: E402
from dagua.layout.ops.graph_utils import (
    layout_device as _layout_device,
)
from dagua.layout.ops.graph_utils import (
    layout_extent as _layout_extent,
)
from dagua.layout.ops.state import (  # noqa: E402
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)
from dagua.layout.ops.taxonomy import OpCategory  # noqa: E402

# -- extras key constants ---------------------------------------------------
_ADJACENCY_KEY = "umap_adjacency"
_DISTANCES_KEY = "umap_distances"
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
_NEGATIVE_SAMPLE_RATE = 5
_NEGATIVE_SAMPLE_RATE_KEY = "umap_negative_sample_rate"
_GAMMA_KEY = "umap_gamma"
_MIN_DIST_KEY = "umap_min_dist"
_SPREAD_KEY = "umap_spread"

# ---------------------------------------------------------------------------
# Constants and helper functions copied from dagua/layout/classic/umap_layout.py
# (bit-identical to their classic counterparts)
# ---------------------------------------------------------------------------

_EPSILON = 1.0e-9
_MIN_SPAN = 1.0e-6
_MIN_SIGMA_SCALE = 1.0e-3
_SMOOTH_K_TOLERANCE = 1.0e-5
_SMOOTH_K_BINARY_SEARCH_STEPS = 64
_SPECTRAL_SPARSE_THRESHOLD = 512
_GRADIENT_CLIP_VALUE = 4.0


def _build_undirected_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor] = None,
) -> Union[list[list[int]], list[list[tuple[int, float]]]]:
    """Build an undirected adjacency list from ``edge_index``."""
    if edge_weights is None:
        adjacency_sets: list[set[int]] = [set() for _ in range(num_nodes)]
        if edge_index.numel() == 0:
            return [sorted(neighbors) for neighbors in adjacency_sets]

        edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
        for source, target in zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist()):
            if source == target:
                continue
            adjacency_sets[source].add(target)
            adjacency_sets[target].add(source)

        return [sorted(neighbors) for neighbors in adjacency_sets]

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
    """Build undirected edge-weight lookup table from the input graph."""
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


def _bfs_distances(adjacency: list[list[int]], start: int) -> torch.Tensor:
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


def _dijkstra_distances(adjacency: list[list[tuple[int, float]]], start: int) -> torch.Tensor:
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
    """Extract k-nearest neighbors from a dense distance matrix."""
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
    """Prune very weak edges and build epoch sampling intervals."""
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


class _BuildUMAPAdjacency(Op):
    """Build undirected adjacency list from graph edges."""

    name: ClassVar[str] = "umap_build_adjacency"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build the undirected adjacency list exactly like classic UMAP.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with adjacency list stored in ``extras``.
        """
        del ctx

        adjacency = _build_undirected_adjacency(
            edge_index=problem.edge_index,
            num_nodes=problem.num_nodes,
            edge_weights=problem.edge_weights,
        )
        state.extras[_ADJACENCY_KEY] = adjacency
        return state


class _ComputeAllPairsShortestPaths(Op):
    """Compute all-pairs graph distances via BFS or Dijkstra."""

    name: ClassVar[str] = "umap_all_pairs_shortest_paths"
    category: ClassVar[OpCategory] = OpCategory.DISTANCE
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute dense shortest-path distance matrix.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state containing the adjacency list.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with distance matrix stored in ``extras``.
        """
        del problem, ctx

        adjacency = state.extras[_ADJACENCY_KEY]
        distances = _all_pairs_shortest_paths(adjacency=adjacency)
        state.extras[_DISTANCES_KEY] = distances
        return state


class _ExtractKNN(Op):
    """Extract k-nearest neighbors from the dense distance matrix."""

    name: ClassVar[str] = "umap_extract_knn"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Extract kNN indices and distances from the distance matrix.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state containing distance matrix and n_neighbors.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with kNN indices and distances stored in ``extras``.
        """
        del problem, ctx

        distances = state.extras[_DISTANCES_KEY]
        n_neighbors = state.extras[_N_NEIGHBORS_KEY]
        knn_indices, knn_distances = _knn_from_distances(
            distances=distances,
            n_neighbors=n_neighbors,
        )
        state.extras[_KNN_INDICES_KEY] = knn_indices
        state.extras[_KNN_DISTANCES_KEY] = knn_distances
        return state


class _SmoothKNNDist(Op):
    """Solve UMAP smooth-kNN bandwidth per node."""

    name: ClassVar[str] = "umap_smooth_knn_dist"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute per-node sigma and rho for the fuzzy simplicial set.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state containing kNN distances and n_neighbors.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with sigmas and rhos stored in ``extras``.
        """
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


class _BuildFuzzySimplicialSet(Op):
    """Build the symmetrized fuzzy simplicial set graph."""

    name: ClassVar[str] = "umap_build_fuzzy_set"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build the symmetrized fuzzy graph with optional edge-weight scaling.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs containing edge_index and edge_weights.
        state : SolveState
            Mutable solve state containing kNN data and sigma/rho.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with fuzzy graph head/tail/weight stored in ``extras``.
        """
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

        # Apply edge-weight scaling exactly like the classic implementation.
        if problem.edge_weights is not None and weight.numel() > 0:
            edge_weight_lookup = _undirected_edge_weight_lookup(
                edge_index=problem.edge_index,
                edge_weights=problem.edge_weights,
            )
            scaled_weight = weight.clone()
            for index in range(weight.shape[0]):
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


class _SpectralInitialization(Op):
    """Compute spectral initialization for the UMAP embedding."""

    name: ClassVar[str] = "umap_spectral_init"
    category: ClassVar[OpCategory] = OpCategory.INIT
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute spectral initialization from the fuzzy graph.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs containing num_nodes and seed.
        state : SolveState
            Mutable solve state containing the fuzzy graph.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with initial embedding in ``state.pos``.
        """
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


class _FitCurveParameters(Op):
    """Fit the UMAP a,b curve parameters from min_dist and spread."""

    name: ClassVar[str] = "umap_fit_curve"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Fit UMAP curve parameters for the low-dimensional membership function.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state containing min_dist and spread in extras.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with fitted curve_a and curve_b stored in ``extras``.
        """
        del problem, ctx

        min_dist = state.extras[_MIN_DIST_KEY]
        spread = state.extras[_SPREAD_KEY]
        a, b = _fit_ab(min_dist=min_dist, spread=spread)
        state.extras[_CURVE_A_KEY] = a
        state.extras[_CURVE_B_KEY] = b
        return state


class _SelectPositiveEdges(Op):
    """Prune weak edges and build epoch sampling intervals."""

    name: ClassVar[str] = "umap_select_positive_edges"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Select positive edges and compute per-edge epoch intervals.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state containing the fuzzy graph and epoch count.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with positive edges and epochs_per_sample in ``extras``.
        """
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


class _OptimizeUMAPEmbedding(Op):
    """Run the cross-entropy SGD optimization with negative sampling."""

    name: ClassVar[str] = "umap_optimize_embedding"
    category: ClassVar[OpCategory] = OpCategory.OPTIMIZE
    reads: ClassVar[Tuple[str, ...]] = ("pos", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run the UMAP SGD loop over the initial embedding.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs containing the seed.
        state : SolveState
            Mutable solve state containing the embedding and SGD parameters.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with optimized embedding in ``state.pos``.

        Raises
        ------
        ValueError
            If ``state.pos`` is missing.
        """
        del ctx

        if state.pos is None:
            raise ValueError("_OptimizeUMAPEmbedding requires state.pos to be set.")

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


class _FinalizeUMAPPositions(Op):
    """Apply classic UMAP's final normalization and device transfer."""

    name: ClassVar[str] = "umap_finalize_positions"
    category: ClassVar[OpCategory] = OpCategory.POSTPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Center, scale, and cast positions like classic UMAP.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state containing the optimized embedding.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with final output positions on the classic return device.

        Raises
        ------
        ValueError
            If ``state.pos`` is missing.
        """
        del ctx

        if state.pos is None:
            raise ValueError("_FinalizeUMAPPositions requires state.pos to be set.")

        device = _layout_device(
            edge_index=problem.edge_index,
            node_sizes=problem.node_sizes,
        )
        extent = _layout_extent(num_nodes=problem.num_nodes, node_sizes=problem.node_sizes)
        normalized = _normalize_positions(positions=state.pos, extent=extent)
        state.pos = normalized.to(dtype=torch.float32, device=device)
        return state


class _StoreUMAPHyperparameters(Op):
    """Store UMAP hyperparameters into extras for downstream ops."""

    name: ClassVar[str] = "umap_store_hyperparameters"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def __init__(
        self,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        spread: float = 1.0,
        n_epochs: Optional[int] = None,
        learning_rate: float = 1.0,
        negative_sample_rate: int = _NEGATIVE_SAMPLE_RATE,
        repulsion_strength: float = 1.0,
    ) -> None:
        """Store UMAP hyperparameters for the pipeline.

        Parameters
        ----------
        n_neighbors : int, default=15
            Neighborhood size for the fuzzy simplicial set.
        min_dist : float, default=0.1
            Target minimum distance in the embedding.
        spread : float, default=1.0
            Target spread for the low-dimensional curve fit.
        n_epochs : int, optional
            Number of SGD epochs. Resolved at apply time based on num_nodes.
        learning_rate : float, default=1.0
            Initial SGD learning rate.
        negative_sample_rate : int, default=5
            Negative samples per positive edge.
        repulsion_strength : float, default=1.0
            Repulsive weight gamma for negative samples.

        Returns
        -------
        None
            The op stores the supplied hyperparameters.
        """
        self._n_neighbors = n_neighbors
        self._min_dist = min_dist
        self._spread = spread
        self._n_epochs = n_epochs
        self._learning_rate = learning_rate
        self._negative_sample_rate = negative_sample_rate
        self._repulsion_strength = repulsion_strength

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Write all UMAP hyperparameters into the solve state extras.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Used to resolve default n_epochs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with UMAP hyperparameters stored in ``extras``.
        """
        del ctx

        n_epochs = self._n_epochs
        if n_epochs is None:
            n_epochs = 500 if problem.num_nodes <= 10_000 else 200

        state.extras[_N_NEIGHBORS_KEY] = self._n_neighbors
        state.extras[_MIN_DIST_KEY] = self._min_dist
        state.extras[_SPREAD_KEY] = self._spread
        state.extras[_N_EPOCHS_KEY] = n_epochs
        state.extras[_LEARNING_RATE_KEY] = self._learning_rate
        state.extras[_NEGATIVE_SAMPLE_RATE_KEY] = self._negative_sample_rate
        state.extras[_GAMMA_KEY] = self._repulsion_strength
        return state


def _validate_inputs(
    edge_index: torch.Tensor,
    num_nodes: int,
    n_neighbors: int,
    edge_weights: Optional[torch.Tensor],
) -> None:
    """Validate public UMAP layout arguments.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    n_neighbors : int
        Target neighborhood size for the fuzzy simplicial set.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.

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
    if edge_weights is not None and edge_weights.shape[0] != edge_index.shape[1]:
        raise ValueError(
            f"edge_weights length {edge_weights.shape[0]} does not match "
            f"edge count {edge_index.shape[1]}"
        )
    if edge_index.numel() == 0:
        return
    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    if int(edge_index_cpu.min().item()) < 0:
        raise ValueError("edge_index cannot contain negative node indices.")
    if int(edge_index_cpu.max().item()) >= num_nodes:
        raise ValueError("edge_index contains node indices outside num_nodes.")


def build_umap_layout_pipeline(
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    spread: float = 1.0,
    n_epochs: Optional[int] = None,
    learning_rate: float = 1.0,
    negative_sample_rate: int = _NEGATIVE_SAMPLE_RATE,
    repulsion_strength: float = 1.0,
) -> Pipeline:
    """Build a UMAP pipeline that is bit-identical to classic ``layout_umap``.

    Parameters
    ----------
    n_neighbors : int, default=15
        Neighborhood size for the fuzzy simplicial set.
    min_dist : float, default=0.1
        Target minimum distance in the embedding.
    spread : float, default=1.0
        Target spread for the low-dimensional curve fit.
    n_epochs : int, optional
        Number of SGD epochs. Defaults to 500 when num_nodes <= 10000
        and 200 otherwise.
    learning_rate : float, default=1.0
        Initial SGD learning rate.
    negative_sample_rate : int, default=5
        Negative samples per positive edge.
    repulsion_strength : float, default=1.0
        Repulsive weight gamma for negative samples.

    Returns
    -------
    Pipeline
        Pipeline that reproduces classic UMAP's fuzzy simplicial set
        construction, spectral initialization, cross-entropy SGD
        optimization, and final normalization.
    """
    return Pipeline(
        [
            _StoreUMAPHyperparameters(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                spread=spread,
                n_epochs=n_epochs,
                learning_rate=learning_rate,
                negative_sample_rate=negative_sample_rate,
                repulsion_strength=repulsion_strength,
            ),
            _BuildUMAPAdjacency(),
            _ComputeAllPairsShortestPaths(),
            _ExtractKNN(),
            _SmoothKNNDist(),
            _BuildFuzzySimplicialSet(),
            _SpectralInitialization(),
            _FitCurveParameters(),
            _SelectPositiveEdges(),
            _OptimizeUMAPEmbedding(),
            _FinalizeUMAPPositions(),
        ],
        name="umap_layout_pipeline",
    )


def layout_umap_layout_pipeline(
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
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the UMAP pipeline as a drop-in replacement for classic ``layout_umap``.

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
        Number of SGD epochs. Defaults to 500 when num_nodes <= 10000
        and 200 otherwise.
    learning_rate : float, default=1.0
        Initial SGD learning rate.
    negative_sample_rate : int, default=5
        Negative samples per positive edge.
    repulsion_strength : float, default=1.0
        Repulsive weight gamma for negative samples.
    seed : int, default=42
        Random seed for spectral noise and negative sampling.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Final position tensor with the same dtype, device, and values as
        classic ``layout_umap``.

    Raises
    ------
    ValueError
        If inputs are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    _validate_inputs(
        edge_index=edge_index,
        num_nodes=num_nodes,
        n_neighbors=n_neighbors,
        edge_weights=edge_weights,
    )
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

    # Early exits matching classic layout_umap exactly.
    device = _layout_device(edge_index=edge_index, node_sizes=node_sizes)
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
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    pipeline = build_umap_layout_pipeline(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        negative_sample_rate=negative_sample_rate,
        repulsion_strength=repulsion_strength,
    )
    final_state = pipeline.apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("UMAP pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_umap_layout_pipeline", "layout_umap_layout_pipeline"]
