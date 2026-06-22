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
from scipy.sparse import csgraph
from scipy.sparse import linalg as sparse_linalg

try:
    from numba import njit as _numba_njit
    from numba import prange as _numba_prange
    from numba import types as _numba_types
except ImportError:
    _numba_njit = None
    _numba_prange = range
    _numba_types = None

from dagua.layout.ops.base import Op
from dagua.layout.ops.graph_utils import (
    layout_device,
)
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_EPSILON = 1.0e-9
_MIN_SPAN = 1.0e-6
_MIN_SIGMA_SCALE = 1.0e-3
_SMOOTH_K_TOLERANCE = 1.0e-5
_SMOOTH_K_BINARY_SEARCH_STEPS = 64
_GRADIENT_CLIP_VALUE = 4.0
_NEGATIVE_SAMPLE_RATE = 5
_INT32_MIN = np.iinfo(np.int32).min + 1
_INT32_MAX = np.iinfo(np.int32).max - 1
_TAU_RAND_MASK = 0xFFFFFFFF
_SPECTRAL_DIMENSIONS = 2
_SPECTRAL_EIGENVECTOR_COUNT = _SPECTRAL_DIMENSIONS + 1
_SPECTRAL_NOISE_SCALE = 1.0e-4
_UMAP_INIT_MAX_COORD = 10.0

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
_OPTIMIZER_RNG_STATE_KEY = "umap_optimizer_rng_state"


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
    """Build an undirected weighted adjacency list from graph edges.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes in the graph.
    edge_weights : torch.Tensor | None, optional
        Optional edge-cost tensor with shape ``[E]``. The UMAP benchmark
        reference feeds these values into SciPy shortest paths as distances,
        so larger values lengthen graph paths here instead of strengthening
        attractive forces.

    Returns
    -------
    list[list[tuple[int, float]]]
        Sorted undirected adjacency list. Parallel edge costs are summed to
        mirror SciPy CSR duplicate coalescing in the reference adapter.
    """
    if edge_weights is None:
        adjacency_maps: list[dict[int, float]] = [dict() for _ in range(num_nodes)]
        if edge_index.numel() == 0:
            return [[] for _ in range(num_nodes)]

        edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
        for source, target in zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist()):
            if source == target:
                continue
            # SciPy CSR coalesces duplicate entries by summing them; count
            # parallel unweighted edges so benchmark distances see the same cost.
            previous = adjacency_maps[source].get(target)
            adjacency_maps[source][target] = previous + 1.0 if previous is not None else 1.0
            previous = adjacency_maps[target].get(source)
            adjacency_maps[target][source] = previous + 1.0 if previous is not None else 1.0

        return [sorted(neighbors.items()) for neighbors in adjacency_maps]

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
        cost = max(float(weights_cpu[edge_id].item()), _EPSILON)
        previous = adjacency_maps[source].get(target)
        adjacency_maps[source][target] = previous + cost if previous is not None else cost
        previous = adjacency_maps[target].get(source)
        adjacency_maps[target][source] = previous + cost if previous is not None else cost

    return [sorted(neighbors.items()) for neighbors in adjacency_maps]


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
    """Compute weighted shortest-path distances from one source.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Undirected adjacency list where each entry is ``(neighbor, cost)``.
    start : int
        Source node for the single-source shortest-path solve.

    Returns
    -------
    torch.Tensor
        Distance vector with shape ``[N]`` and dtype ``torch.float32``.
    """
    num_nodes = len(adjacency)
    distances = [float("inf")] * num_nodes
    distances[start] = 0.0
    heap: list[tuple[float, int]] = [(0.0, start)]
    while heap:
        distance, node = heapq.heappop(heap)
        if distance > distances[node]:
            continue
        for neighbor, cost in adjacency[node]:
            candidate = distance + cost
            if candidate >= distances[neighbor]:
                continue
            distances[neighbor] = candidate
            heapq.heappush(heap, (candidate, neighbor))
    return torch.tensor(distances, dtype=torch.float32)


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


def _effective_umap_n_neighbors(requested: int, num_nodes: int) -> int:
    """Clamp UMAP neighbor count to the valid graph range.

    Parameters
    ----------
    requested : int
        Requested UMAP ``n_neighbors`` value.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    int
        Effective neighbor count. For graphs with at least two nodes this is
        at most ``num_nodes - 1``, matching umap-learn's kNN precondition.
    """
    if num_nodes <= 1:
        return 1
    return min(requested, num_nodes - 1)


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
    """Solve the UMAP smooth-kNN bandwidth for every graph node.

    Parameters
    ----------
    knn_distances : torch.Tensor
        Neighbor distances with shape ``[N, K]``.
    n_neighbors : int
        Neighborhood size ``k`` used by the fuzzy simplicial set.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Smooth bandwidth ``sigma`` and local connectivity radius ``rho`` vectors
        with shape ``[N]``.
    """
    num_nodes = knn_distances.shape[0]
    if num_nodes == 0:
        empty = torch.empty((0,), dtype=torch.float32)
        return empty, empty

    sigmas = torch.empty((num_nodes,), dtype=torch.float32)
    rhos = torch.empty((num_nodes,), dtype=torch.float32)
    target = np.float32(log2(float(max(n_neighbors, 2))))
    distances_np = knn_distances.detach().to(device="cpu", dtype=torch.float32).numpy()
    mean_distances = np.float32(np.mean(distances_np)) if distances_np.size > 0 else np.float32(0.0)

    for index in range(num_nodes):
        finite = distances_np[index][np.isfinite(distances_np[index])]
        if finite.size == 0:
            sigmas[index] = 1.0
            rhos[index] = 0.0
            continue

        positive = finite[finite > 0.0]
        rho = np.float32(positive[0]) if positive.shape[0] > 0 else np.float32(0.0)
        rhos[index] = float(rho)

        lower = np.float32(0.0)
        upper = np.float32(np.inf)
        sigma = np.float32(1.0)
        for _ in range(_SMOOTH_K_BINARY_SEARCH_STEPS):
            estimate = np.float32(0.0)
            for distance in finite[1:]:
                shifted = np.float32(distance - rho)
                if shifted > 0.0:
                    estimate += np.float32(np.exp(np.float32(-(shifted / sigma))))
                else:
                    estimate += np.float32(1.0)
            if abs(estimate - target) <= _SMOOTH_K_TOLERANCE:
                break
            if estimate > target:
                upper = sigma
                sigma = np.float32((lower + upper) / np.float32(2.0))
            else:
                lower = sigma
                if upper == np.float32(np.inf):
                    sigma = np.float32(sigma * np.float32(2.0))
                else:
                    sigma = np.float32((lower + upper) / np.float32(2.0))

        if rho > 0.0:
            sigma_floor = np.float32(_MIN_SIGMA_SCALE) * np.float32(np.mean(finite))
        else:
            sigma_floor = np.float32(_MIN_SIGMA_SCALE) * mean_distances
        sigmas[index] = float(max(sigma, sigma_floor))

    return sigmas, rhos


def _native_compute_membership_strengths(
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
    sigmas: np.ndarray,
    rhos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct local fuzzy-set membership strengths.

    Parameters
    ----------
    knn_indices : numpy.ndarray
        Neighbor indices with shape ``[N, K]`` and dtype ``int64`` or
        ``int32``.
    knn_distances : numpy.ndarray
        Neighbor distances with shape ``[N, K]`` and dtype ``float32``.
    sigmas : numpy.ndarray
        Smooth-kNN bandwidths with shape ``[N]`` and dtype ``float32``.
    rhos : numpy.ndarray
        Local connectivity radii with shape ``[N]`` and dtype ``float32``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        COO rows, columns, and float32 membership values. The arithmetic
        mirrors ``umap-learn``'s numba ``compute_membership_strengths`` kernel.
    """
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]
    rows = np.zeros(knn_indices.size, dtype=np.int32)
    cols = np.zeros(knn_indices.size, dtype=np.int32)
    vals = np.zeros(knn_indices.size, dtype=np.float32)

    for row in range(n_samples):
        for column in range(n_neighbors):
            if knn_indices[row, column] == -1:
                continue
            if knn_indices[row, column] == row:
                val = 0.0
            elif knn_distances[row, column] - rhos[row] <= 0.0 or sigmas[row] == 0.0:
                val = 1.0
            else:
                val = np.exp(-((knn_distances[row, column] - rhos[row]) / sigmas[row]))

            offset = row * n_neighbors + column
            rows[offset] = row
            cols[offset] = knn_indices[row, column]
            vals[offset] = val

    return rows, cols, vals


_UMAP_COMPUTE_MEMBERSHIP = (
    _numba_njit(
        locals={"val": _numba_types.float32},
        parallel=True,
        fastmath=True,
    )(_native_compute_membership_strengths)
    if _numba_njit is not None and _numba_types is not None
    else _native_compute_membership_strengths
)


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
        Neighbor index tensor with shape ``[N, K]``.
    knn_distances : torch.Tensor
        Neighbor distance tensor with shape ``[N, K]``.
    sigmas : torch.Tensor
        Smooth kNN bandwidths with shape ``[N]``.
    rhos : torch.Tensor
        Local-connectivity radii with shape ``[N]``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        COO row indices, column indices, and fuzzy membership strengths. Both
        directions are retained because UMAP optimizes the sparse COO graph
        after fuzzy union, not a collapsed undirected edge list.
    """
    num_nodes, num_neighbors = knn_indices.shape
    indices_np = knn_indices.detach().to(device="cpu", dtype=torch.long).numpy()
    distances_np = knn_distances.detach().to(device="cpu", dtype=torch.float32).numpy()
    sigmas_np = sigmas.detach().to(device="cpu", dtype=torch.float32).numpy()
    rhos_np = rhos.detach().to(device="cpu", dtype=torch.float32).numpy()
    rows, cols, vals = _UMAP_COMPUTE_MEMBERSHIP(
        indices_np,
        distances_np,
        sigmas_np,
        rhos_np,
    )

    result = sparse.coo_matrix((vals, (rows, cols)), shape=(num_nodes, num_nodes))
    result.eliminate_zeros()
    transpose = result.transpose()
    product = result.multiply(transpose)
    result = 1.0 * (result + transpose - product) + 0.0 * product
    result.eliminate_zeros()
    coo = result.tocoo()

    if coo.nnz == 0:
        return (
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
        )

    head = torch.from_numpy(coo.row.astype(np.int64, copy=True))
    tail = torch.from_numpy(coo.col.astype(np.int64, copy=True))
    weight_tensor = torch.from_numpy(coo.data.astype(np.float32, copy=True))
    return head, tail, weight_tensor


def _curve_function(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Evaluate UMAP's smooth low-dimensional membership curve."""
    return 1.0 / (1.0 + a * x ** (2 * b))


def _fit_ab(min_dist: float, spread: float) -> tuple[float, float]:
    """Fit UMAP's ``a`` and ``b`` curve parameters from ``min_dist``."""
    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    try:
        params, _ = optimize.curve_fit(
            _curve_function,
            xv,
            yv,
        )
        return float(params[0]), float(params[1])
    except (RuntimeError, ValueError):
        return 1.93, 0.79


def _normalize_laplacian(graph: sparse.csr_matrix) -> sparse.csr_matrix:
    """Build UMAP's normalized Laplacian for a weighted fuzzy graph.

    Parameters
    ----------
    graph : scipy.sparse.csr_matrix
        Symmetric sparse fuzzy graph with shape ``[N, N]``.

    Returns
    -------
    scipy.sparse.csr_matrix
        Normalized graph Laplacian with shape ``[N, N]``.
    """
    sqrt_degree = np.sqrt(np.asarray(graph.sum(axis=0)).squeeze())
    inv_sqrt_degree = np.zeros_like(sqrt_degree)
    nonzero = sqrt_degree > 0.0
    inv_sqrt_degree[nonzero] = 1.0 / sqrt_degree[nonzero]
    d_inv_sqrt = sparse.spdiags(inv_sqrt_degree, 0, graph.shape[0], graph.shape[0])
    identity = sparse.identity(graph.shape[0], dtype=np.float64)
    return identity - d_inv_sqrt * graph * d_inv_sqrt


def _connected_spectral_embedding(
    graph: sparse.csr_matrix,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Compute raw connected-component spectral coordinates with ARPACK.

    Parameters
    ----------
    graph : scipy.sparse.csr_matrix
        Connected symmetric sparse fuzzy graph with shape ``[N, N]``.
    rng : numpy.random.RandomState, optional
        Random state advanced with UMAP's eigensolver initialization draw.

    Returns
    -------
    numpy.ndarray
        Raw non-trivial eigenvectors with shape ``[N, 2]``.
    """
    num_nodes = graph.shape[0]
    if num_nodes <= _SPECTRAL_DIMENSIONS + 1:
        return np.zeros((num_nodes, _SPECTRAL_DIMENSIONS), dtype=np.float32)

    laplacian = _normalize_laplacian(graph=graph)
    k = _SPECTRAL_EIGENVECTOR_COUNT
    ncv = max(7, int(np.sqrt(num_nodes)))
    if rng is not None:
        # UMAP draws this normal initialization even for the ARPACK path where
        # ``eigsh`` receives ``v0`` instead. Preserve that RNG advancement so
        # spectral noise and negative sampling receive the same base state.
        rng.normal(size=(num_nodes, k))
    eigenvalues, eigenvectors = sparse_linalg.eigsh(
        laplacian,
        k=k,
        which="SM",
        ncv=ncv,
        v0=np.ones(num_nodes, dtype=np.float64),
        tol=1.0e-4,
        maxiter=num_nodes * 5,
    )
    order = np.argsort(eigenvalues)[1:k]
    coordinates = np.real(eigenvectors[:, order])
    if coordinates.shape[1] < _SPECTRAL_DIMENSIONS:
        coordinates = np.pad(
            coordinates,
            ((0, 0), (0, _SPECTRAL_DIMENSIONS - coordinates.shape[1])),
            mode="constant",
        )
    return coordinates


def _component_meta_embedding(
    n_components: int,
    component_labels: np.ndarray,
    distance_matrix: Optional[np.ndarray],
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Place disconnected fuzzy-graph components before local embeddings.

    Parameters
    ----------
    n_components : int
        Number of connected components in the fuzzy graph.
    component_labels : numpy.ndarray
        Component label for every node with shape ``[N]``.
    distance_matrix : numpy.ndarray | None
        Precomputed graph-distance matrix with shape ``[N, N]``. When
        available, average inter-component distances drive the meta layout.
    rng : numpy.random.RandomState, optional
        Random state advanced by larger component-level spectral layouts.

    Returns
    -------
    numpy.ndarray
        Component-level coordinates with shape ``[C, 2]``.
    """
    if n_components <= 2 * _SPECTRAL_DIMENSIONS:
        half_components = int(np.ceil(n_components / 2.0))
        base = np.hstack(
            [
                np.eye(half_components),
                np.zeros((half_components, _SPECTRAL_DIMENSIONS - half_components)),
            ]
        )
        return np.vstack([base, -base])[:n_components].astype(np.float32, copy=False)

    if distance_matrix is None:
        fallback = np.arange(n_components * _SPECTRAL_DIMENSIONS, dtype=np.float32).reshape(
            n_components,
            _SPECTRAL_DIMENSIONS,
        )
        return fallback / max(float(fallback.max()), 1.0)

    component_distances = np.zeros((n_components, n_components), dtype=np.float64)
    for source_label in range(n_components):
        source_rows = distance_matrix[component_labels == source_label]
        for target_label in range(source_label + 1, n_components):
            between = source_rows[:, component_labels == target_label]
            distance = float(np.mean(between)) if between.size > 0 else 0.0
            component_distances[source_label, target_label] = distance
            component_distances[target_label, source_label] = distance

    affinity = np.exp(-(component_distances**2))
    np.fill_diagonal(affinity, 0.0)
    meta_graph = sparse.csr_matrix(affinity, dtype=np.float64)
    meta_embedding = _connected_spectral_embedding(graph=meta_graph, rng=rng)
    max_abs = float(np.max(np.abs(meta_embedding))) if meta_embedding.size > 0 else 0.0
    if max_abs > _MIN_SPAN:
        meta_embedding = meta_embedding / max_abs
    return meta_embedding.astype(np.float32, copy=False)


def _multi_component_spectral_embedding(
    graph: sparse.csr_matrix,
    n_components: int,
    component_labels: np.ndarray,
    rng: np.random.RandomState,
    distance_matrix: Optional[np.ndarray],
) -> np.ndarray:
    """Compute UMAP-style initialization for disconnected fuzzy graphs.

    Parameters
    ----------
    graph : scipy.sparse.csr_matrix
        Symmetric fuzzy graph with shape ``[N, N]``.
    n_components : int
        Number of connected components.
    component_labels : numpy.ndarray
        Component labels with shape ``[N]``.
    rng : numpy.random.RandomState
        Random state used for small-component placement and eigensolver
        initialization draws.
    distance_matrix : numpy.ndarray | None
        Dense precomputed graph distances with shape ``[N, N]``.

    Returns
    -------
    numpy.ndarray
        Raw multi-component spectral coordinates with shape ``[N, 2]``.
    """
    result = np.empty((graph.shape[0], _SPECTRAL_DIMENSIONS), dtype=np.float32)
    meta_embedding = _component_meta_embedding(
        n_components=n_components,
        component_labels=component_labels,
        distance_matrix=distance_matrix,
        rng=rng,
    )

    for label in range(n_components):
        component_mask = component_labels == label
        component_graph = graph.tocsr()[component_mask, :].tocsc()[:, component_mask].tocsr()
        distances = np.linalg.norm(meta_embedding - meta_embedding[label], axis=1)
        positive_distances = distances[distances > 0.0]
        data_range = float(positive_distances.min() / 2.0) if positive_distances.size > 0 else 1.0

        if (
            component_graph.shape[0] < 2 * _SPECTRAL_DIMENSIONS
            or component_graph.shape[0] <= _SPECTRAL_DIMENSIONS + 1
        ):
            component_embedding = rng.uniform(
                low=-data_range,
                high=data_range,
                size=(component_graph.shape[0], _SPECTRAL_DIMENSIONS),
            )
        else:
            component_embedding = _connected_spectral_embedding(graph=component_graph, rng=rng)
            max_abs = (
                float(np.max(np.abs(component_embedding))) if component_embedding.size > 0 else 0.0
            )
            if max_abs > _MIN_SPAN:
                component_embedding = component_embedding * (data_range / max_abs)

        result[component_mask] = component_embedding + meta_embedding[label]

    return result


def _scale_umap_initial_coordinates(
    coordinates: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Apply UMAP's noisy max-abs scale before the final embedding rescale.

    Parameters
    ----------
    coordinates : numpy.ndarray
        Raw initialization coordinates with shape ``[N, 2]``.
    rng : numpy.random.RandomState
        Random state used for the small noise added after max-abs scaling.

    Returns
    -------
    numpy.ndarray
        Noisy max-abs-scaled embedding with shape ``[N, 2]`` and dtype
        ``float32``. ``umap-learn`` draws the optimizer RNG state before the
        later per-axis ``[0, 10]`` rescale, so that final rescale happens in
        :class:`SpectralInitialization`.
    """
    max_abs = float(np.max(np.abs(coordinates))) if coordinates.size > 0 else 0.0
    if max_abs > _MIN_SPAN:
        coords = (coordinates * (_UMAP_INIT_MAX_COORD / max_abs)).astype(np.float32)
    else:
        coords = coordinates.astype(np.float32, copy=True)

    noise = rng.normal(scale=_SPECTRAL_NOISE_SCALE, size=coords.shape).astype(np.float32)
    return (coords + noise).astype(np.float32, copy=False)


def _rescale_umap_embedding(coordinates: np.ndarray) -> torch.Tensor:
    """Apply UMAP's unconditional per-axis ``[0, 10]`` embedding rescale.

    Parameters
    ----------
    coordinates : numpy.ndarray
        Initial coordinates with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Rescaled initial embedding with shape ``[N, 2]``.
    """
    coords = coordinates.astype(np.float32, copy=True)
    if coords.shape[0] == 0:
        return torch.empty((0, 2), dtype=torch.float32)

    axis_min = coords.min(axis=0)
    axis_span = coords.max(axis=0) - axis_min
    safe_span = np.where(axis_span > 0.0, axis_span, _MIN_SPAN)
    coords = _UMAP_INIT_MAX_COORD * (coords - axis_min) / safe_span
    return torch.from_numpy(coords.astype(np.float32, copy=False))


def _raw_spectral_initialization(
    num_nodes: int,
    head: torch.Tensor,
    tail: torch.Tensor,
    weight: torch.Tensor,
    seed: int,
    distance_matrix: Optional[torch.Tensor] = None,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Compute UMAP initialization before the final affine embedding rescale.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    head : torch.Tensor
        Fuzzy-graph edge heads with shape ``[E]``.
    tail : torch.Tensor
        Fuzzy-graph edge tails with shape ``[E]``.
    weight : torch.Tensor
        Fuzzy-graph edge weights with shape ``[E]``.
    seed : int
        Seed for initialization noise and small-component placement.
    distance_matrix : torch.Tensor | None, optional
        Dense graph-distance matrix with shape ``[N, N]`` used to place
        disconnected fuzzy-graph components.
    rng : numpy.random.RandomState, optional
        Mutable random state. When provided, it is advanced where umap-learn
        advances its ``random_state`` during initialization.

    Returns
    -------
    numpy.ndarray
        Initial embedding with shape ``[N, 2]`` before UMAP's final affine
        embedding rescale.
    """
    if num_nodes == 0:
        return np.empty((0, 2), dtype=np.float32)
    if num_nodes == 1:
        return np.zeros((1, 2), dtype=np.float32)
    if num_nodes == 2:
        return np.array([[-10.0, 0.0], [10.0, 0.0]], dtype=np.float32)

    random_state = rng if rng is not None else np.random.RandomState(seed)
    if num_nodes < 10:
        coordinates = random_state.uniform(
            low=-_UMAP_INIT_MAX_COORD,
            high=_UMAP_INIT_MAX_COORD,
            size=(num_nodes, _SPECTRAL_DIMENSIONS),
        ).astype(np.float32)
        return coordinates

    rows = head.detach().to(device="cpu", dtype=torch.long).numpy()
    cols = tail.detach().to(device="cpu", dtype=torch.long).numpy()
    data = weight.detach().to(device="cpu", dtype=torch.float32).numpy()
    graph = sparse.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    graph.sum_duplicates()
    graph.eliminate_zeros()

    n_components, component_labels = csgraph.connected_components(graph)
    if n_components > 1:
        distances_np = (
            distance_matrix.detach().to(device="cpu", dtype=torch.float32).numpy()
            if distance_matrix is not None
            else None
        )
        coordinates = _multi_component_spectral_embedding(
            graph=graph,
            n_components=int(n_components),
            component_labels=component_labels,
            rng=random_state,
            distance_matrix=distances_np,
        )
    else:
        coordinates = _connected_spectral_embedding(graph=graph, rng=random_state)

    return _scale_umap_initial_coordinates(coordinates=coordinates, rng=random_state)


def _spectral_initialization(
    num_nodes: int,
    head: torch.Tensor,
    tail: torch.Tensor,
    weight: torch.Tensor,
    seed: int,
    distance_matrix: Optional[torch.Tensor] = None,
    rng: Optional[np.random.RandomState] = None,
) -> torch.Tensor:
    """Compute UMAP spectral initialization in its public ``[0, 10]`` frame.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    head : torch.Tensor
        Fuzzy-graph edge heads with shape ``[E]``.
    tail : torch.Tensor
        Fuzzy-graph edge tails with shape ``[E]``.
    weight : torch.Tensor
        Fuzzy-graph edge weights with shape ``[E]``.
    seed : int
        Seed for initialization noise and small-component placement.
    distance_matrix : torch.Tensor | None, optional
        Dense graph-distance matrix with shape ``[N, N]`` used to place
        disconnected fuzzy-graph components.
    rng : numpy.random.RandomState, optional
        Mutable random state used by initialization.

    Returns
    -------
    torch.Tensor
        Rescaled initial embedding with shape ``[N, 2]`` and dtype
        ``torch.float32``.
    """
    raw_embedding = _raw_spectral_initialization(
        num_nodes=num_nodes,
        head=head,
        tail=tail,
        weight=weight,
        seed=seed,
        distance_matrix=distance_matrix,
        rng=rng,
    )
    return _rescale_umap_embedding(coordinates=raw_embedding)


def _select_positive_edges(
    head: torch.Tensor,
    tail: torch.Tensor,
    weight: torch.Tensor,
    n_epochs: int,
    default_epochs: int = 500,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prune weak edges and build epoch sampling intervals.

    Parameters
    ----------
    head : torch.Tensor
        Fuzzy-graph source indices with shape ``[E]``.
    tail : torch.Tensor
        Fuzzy-graph target indices with shape ``[E]``.
    weight : torch.Tensor
        Fuzzy membership strengths with shape ``[E]``.
    n_epochs : int
        Requested optimization epoch count.
    default_epochs : int, default=500
        UMAP's automatic epoch count, used for pruning when ``n_epochs <= 10``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Pruned source indices, target indices, pruned fuzzy weights, and
        positive sample intervals.
    """
    if weight.numel() == 0:
        empty_long = torch.empty((0,), dtype=torch.long)
        empty_float = torch.empty((0,), dtype=torch.float64)
        return empty_long, empty_long, empty_float.to(dtype=torch.float32), empty_float

    max_weight = float(weight.max().item())
    prune_epochs = n_epochs if n_epochs > 10 else default_epochs
    min_weight = max_weight / float(max(prune_epochs, 1))
    keep = weight >= min_weight
    kept_head = head[keep]
    kept_tail = tail[keep]
    kept_weight = weight[keep]
    epochs_per_sample = max_weight / kept_weight.to(dtype=torch.float64)
    return kept_head, kept_tail, kept_weight, epochs_per_sample


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


def _clip_gradient(value: float) -> np.float32:
    """Clamp a scalar UMAP SGD gradient component.

    Parameters
    ----------
    value : float
        Raw gradient component.

    Returns
    -------
    numpy.float32
        Component clipped to UMAP's ``[-4, 4]`` update range.
    """
    return np.float32(min(max(value, -_GRADIENT_CLIP_VALUE), _GRADIENT_CLIP_VALUE))


def _to_signed_int32(value: int) -> int:
    """Convert an unsigned 32-bit integer to signed int32 range.

    Parameters
    ----------
    value : int
        Integer value carrying the low 32 bits of a Tausworthe draw.

    Returns
    -------
    int
        Signed int32 interpretation of ``value``.
    """
    masked = value & _TAU_RAND_MASK
    if masked >= (1 << 31):
        return masked - (1 << 32)
    return masked


def _make_tau_state(
    embedding: torch.Tensor,
    seed: int,
    rng_state: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Create umap-learn-compatible per-source Tausworthe states.

    Parameters
    ----------
    embedding : torch.Tensor
        Initial low-dimensional coordinates with shape ``[N, 2]``.
    seed : int
        Random seed used to draw UMAP's base three-int RNG state.
    rng_state : numpy.ndarray, optional
        Pre-advanced base UMAP RNG state with shape ``[3]``. When supplied,
        it is used instead of drawing from a fresh ``seed`` state.

    Returns
    -------
    numpy.ndarray
        Mutable Tausworthe state array with shape ``[N, 3]`` and dtype
        ``int64``. Each source row is offset by the first embedding coordinate
        cast to float64 and reinterpreted as int64, matching umap-learn.
    """
    if rng_state is None:
        random_state = np.random.RandomState(seed)
        base_state = random_state.randint(_INT32_MIN, _INT32_MAX, 3).astype(np.int64)
    else:
        base_state = rng_state.astype(np.int64, copy=True)
    first_coordinate = embedding.detach().to(device="cpu", dtype=torch.float32)[:, 0].numpy()
    coordinate_state = first_coordinate.astype(np.float64).view(np.int64).reshape(-1, 1)
    return np.full((embedding.shape[0], base_state.shape[0]), base_state, dtype=np.int64) + (
        coordinate_state
    )


def _tau_rand_int(state: np.ndarray) -> int:
    """Draw a signed int32 from UMAP's combined Tausworthe generator.

    Parameters
    ----------
    state : numpy.ndarray
        Mutable per-source RNG state with shape ``[3]`` and dtype ``int64``.

    Returns
    -------
    int
        Signed int32 random integer matching ``umap.utils.tau_rand_int``.
    """
    state_0 = int(state[0])
    state_1 = int(state[1])
    state_2 = int(state[2])
    state_0 = (((state_0 & 4294967294) << 12) & _TAU_RAND_MASK) ^ (
        (((state_0 << 13) & _TAU_RAND_MASK) ^ state_0) >> 19
    )
    state_1 = (((state_1 & 4294967288) << 4) & _TAU_RAND_MASK) ^ (
        (((state_1 << 2) & _TAU_RAND_MASK) ^ state_1) >> 25
    )
    state_2 = (((state_2 & 4294967280) << 17) & _TAU_RAND_MASK) ^ (
        (((state_2 << 3) & _TAU_RAND_MASK) ^ state_2) >> 11
    )
    state[0] = state_0
    state[1] = state_1
    state[2] = state_2
    return _to_signed_int32(state_0 ^ state_1 ^ state_2)


def _tau_rand_int_numba(state: np.ndarray) -> int:
    """Draw one Tausworthe int using numba's int32 return cast.

    Parameters
    ----------
    state : numpy.ndarray
        Mutable per-source RNG state with shape ``[3]`` and dtype ``int64``.

    Returns
    -------
    int
        Signed int32 random integer after numba applies the declared return
        type, matching ``umap.utils.tau_rand_int``.
    """
    state[0] = (((state[0] & 4294967294) << 12) & _TAU_RAND_MASK) ^ (
        (((state[0] << 13) & _TAU_RAND_MASK) ^ state[0]) >> 19
    )
    state[1] = (((state[1] & 4294967288) << 4) & _TAU_RAND_MASK) ^ (
        (((state[1] << 2) & _TAU_RAND_MASK) ^ state[1]) >> 25
    )
    state[2] = (((state[2] & 4294967280) << 17) & _TAU_RAND_MASK) ^ (
        (((state[2] << 3) & _TAU_RAND_MASK) ^ state[2]) >> 11
    )
    return state[0] ^ state[1] ^ state[2]


_UMAP_TAU_RAND_INT = (
    _numba_njit("i4(i8[:])")(_tau_rand_int_numba) if _numba_njit is not None else _tau_rand_int
)


def _squared_euclidean_distance(
    current: np.ndarray,
    other: np.ndarray,
) -> np.float32:
    """Compute UMAP's reduced Euclidean distance for two embedding rows.

    Parameters
    ----------
    current : numpy.ndarray
        First coordinate row with shape ``[D]`` and dtype ``float32``.
    other : numpy.ndarray
        Second coordinate row with shape ``[D]`` and dtype ``float32``.

    Returns
    -------
    numpy.float32
        Squared Euclidean distance between the two rows.
    """
    result = np.float32(0.0)
    for dimension in range(current.shape[0]):
        diff = current[dimension] - other[dimension]
        result += diff * diff
    return result


def _clip_umap_gradient(value: float) -> float:
    """Clamp one UMAP SGD gradient component.

    Parameters
    ----------
    value : float
        Raw gradient component.

    Returns
    -------
    float
        Component restricted to UMAP's ``[-4, 4]`` clipping range.
    """
    if value > _GRADIENT_CLIP_VALUE:
        return _GRADIENT_CLIP_VALUE
    if value < -_GRADIENT_CLIP_VALUE:
        return -_GRADIENT_CLIP_VALUE
    return value


_UMAP_RDIST = (
    _numba_njit(
        "f4(f4[::1],f4[::1])",
        fastmath=True,
        locals={
            "result": _numba_types.float32,
            "diff": _numba_types.float32,
            "dimension": _numba_types.intp,
        },
    )(_squared_euclidean_distance)
    if _numba_njit is not None
    else _squared_euclidean_distance
)
_UMAP_CLIP = _numba_njit(_clip_umap_gradient) if _numba_njit is not None else _clip_umap_gradient


def _native_umap_single_epoch(
    head_embedding: np.ndarray,
    tail_embedding: np.ndarray,
    head: np.ndarray,
    tail: np.ndarray,
    n_vertices: int,
    epochs_per_sample: np.ndarray,
    a: float,
    b: float,
    rng_state_per_sample: np.ndarray,
    gamma: float,
    dim: int,
    move_other: bool,
    alpha: float,
    epochs_per_negative_sample: np.ndarray,
    epoch_of_next_negative_sample: np.ndarray,
    epoch_of_next_sample: np.ndarray,
    epoch: int,
) -> None:
    """Run one native UMAP Euclidean SGD epoch.

    Parameters
    ----------
    embedding : numpy.ndarray
        Mutable embedding array with shape ``[N, 2]`` and dtype ``float32``.
    head : numpy.ndarray
        Positive-edge source indices with shape ``[E]``.
    tail : numpy.ndarray
        Positive-edge target indices with shape ``[E]``.
    n_vertices : int
        Number of graph vertices.
    epochs_per_sample : numpy.ndarray
        Positive-edge sampling intervals with shape ``[E]``.
    a : float
        UMAP low-dimensional curve parameter.
    b : float
        UMAP low-dimensional curve parameter.
    rng_state_per_sample : numpy.ndarray
        Mutable Tausworthe states with shape ``[N, 3]`` and dtype ``int64``.
    gamma : float
        Repulsion multiplier for negative samples.
    dim : int
        Embedding dimensionality.
    alpha : float
        Current learning rate.
    epochs_per_negative_sample : numpy.ndarray
        Negative-sample intervals with shape ``[E]``.
    epoch_of_next_negative_sample : numpy.ndarray
        Next negative-sample epoch per edge with shape ``[E]``.
    epoch_of_next_sample : numpy.ndarray
        Next positive-sample epoch per edge with shape ``[E]``.
    epoch : int
        Current epoch number.

    Returns
    -------
    None
        The embedding and epoch counters are mutated in place.
    """
    for edge_id in _numba_prange(epochs_per_sample.shape[0]):
        if epoch_of_next_sample[edge_id] <= epoch:
            source = head[edge_id]
            target = tail[edge_id]
            current = head_embedding[source]
            other = tail_embedding[target]

            distance_sq = _UMAP_RDIST(current, other)

            if distance_sq > 0.0:
                grad_coeff = -2.0 * a * b * pow(distance_sq, b - 1.0)
                grad_coeff /= (a * pow(distance_sq, b)) + 1.0
            else:
                grad_coeff = 0.0

            for dimension in range(dim):
                grad_d = _UMAP_CLIP(grad_coeff * (current[dimension] - other[dimension]))
                current[dimension] += grad_d * alpha
                if move_other:
                    other[dimension] += -grad_d * alpha

            epoch_of_next_sample[edge_id] += epochs_per_sample[edge_id]
            n_neg_samples = int(
                (epoch - epoch_of_next_negative_sample[edge_id])
                / epochs_per_negative_sample[edge_id]
            )

            for _ in range(n_neg_samples):
                negative = _UMAP_TAU_RAND_INT(rng_state_per_sample[source]) % n_vertices
                other = tail_embedding[negative]

                distance_sq = _UMAP_RDIST(current, other)

                if distance_sq > 0.0:
                    grad_coeff = 2.0 * gamma * b
                    grad_coeff /= (0.001 + distance_sq) * ((a * pow(distance_sq, b)) + 1.0)
                elif source == negative:
                    continue
                else:
                    grad_coeff = 0.0

                for dimension in range(dim):
                    if grad_coeff > 0.0:
                        grad_d = _UMAP_CLIP(grad_coeff * (current[dimension] - other[dimension]))
                    else:
                        grad_d = 0.0
                    current[dimension] += grad_d * alpha

            epoch_of_next_negative_sample[edge_id] += (
                n_neg_samples * epochs_per_negative_sample[edge_id]
            )


_UMAP_SINGLE_EPOCH = (
    _numba_njit(_native_umap_single_epoch, fastmath=True)
    if _numba_njit is not None
    else _native_umap_single_epoch
)


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
    rng_state: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """Run the UMAP cross-entropy SGD with negative sampling.

    Parameters
    ----------
    embedding : torch.Tensor
        Initial low-dimensional coordinates with shape ``[N, 2]``.
    head : torch.Tensor
        Source vertex indices for positive fuzzy-graph edges with shape ``[E]``.
    tail : torch.Tensor
        Target vertex indices for positive fuzzy-graph edges with shape ``[E]``.
    epochs_per_sample : torch.Tensor
        Positive-edge sampling intervals with shape ``[E]``.
    n_epochs : int
        Number of optimization epochs.
    learning_rate : float
        Initial SGD learning rate.
    negative_sample_rate : int
        Negative samples scheduled per positive edge sample.
    gamma : float
        Repulsion multiplier for negative samples.
    a : float
        UMAP low-dimensional curve parameter.
    b : float
        UMAP low-dimensional curve parameter.
    seed : int
        Random seed for negative-sample Tausworthe state initialization.
    rng_state : numpy.ndarray, optional
        Pre-advanced base UMAP RNG state with shape ``[3]``.

    Returns
    -------
    torch.Tensor
        Optimized low-dimensional coordinates with shape ``[N, 2]``.
    """
    if head.numel() == 0 or n_epochs <= 0:
        return embedding

    embedding_np = embedding.detach().to(device="cpu", dtype=torch.float32).numpy().copy()
    head_np = head.detach().to(device="cpu", dtype=torch.long).numpy()
    tail_np = tail.detach().to(device="cpu", dtype=torch.long).numpy()
    epochs_per_sample_np = (
        epochs_per_sample.detach()
        .to(
            device="cpu",
            dtype=torch.float64,
        )
        .numpy()
    )
    if negative_sample_rate > 0:
        epochs_per_negative_sample = epochs_per_sample_np / float(negative_sample_rate)
    else:
        epochs_per_negative_sample = np.full_like(epochs_per_sample_np, 1.0e12)
    next_sample_epoch = epochs_per_sample_np.copy()
    next_negative_epoch = epochs_per_negative_sample.copy()
    num_nodes = embedding.shape[0]
    rng_state_per_source = _make_tau_state(
        embedding=torch.from_numpy(embedding_np),
        seed=seed,
        rng_state=rng_state,
    )
    alpha = float(learning_rate)

    for epoch in range(n_epochs):
        _UMAP_SINGLE_EPOCH(
            embedding_np,
            embedding_np,
            head_np,
            tail_np,
            num_nodes,
            epochs_per_sample_np,
            a,
            b,
            rng_state_per_source,
            gamma,
            embedding_np.shape[1],
            True,
            alpha,
            epochs_per_negative_sample,
            next_negative_epoch,
            next_sample_epoch,
            epoch,
        )
        alpha = float(learning_rate) * (1.0 - (float(epoch) / float(max(n_epochs, 1))))

    return torch.from_numpy(embedding_np).to(device=embedding.device, dtype=torch.float32)


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
        """Write all validated hyperparameters into solve-state extras.

        Parameters
        ----------
        problem : LayoutProblem
            Layout problem carrying graph size and random seed metadata.
        state : SolveState
            Mutable solve state that receives UMAP hyperparameter extras.
        ctx : RuntimeContext
            Runtime context supplied by the pipeline executor.

        Returns
        -------
        SolveState
            Updated solve state with validated UMAP hyperparameters.
        """
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

        state.extras[_N_NEIGHBORS_KEY] = _effective_umap_n_neighbors(
            requested=self.config.n_neighbors,
            num_nodes=problem.num_nodes,
        )
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
        "distance_matrix",
        f"extras.{_FUZZY_HEAD_KEY}",
        f"extras.{_FUZZY_TAIL_KEY}",
        f"extras.{_FUZZY_WEIGHT_KEY}",
    )
    writes: ClassVar[Tuple[str, ...]] = ("pos", f"extras.{_OPTIMIZER_RNG_STATE_KEY}")
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
        rng = np.random.RandomState(problem.seed)
        initial_embedding = _raw_spectral_initialization(
            num_nodes=problem.num_nodes,
            head=head,
            tail=tail,
            weight=weight,
            seed=problem.seed,
            distance_matrix=state.distance_matrix,
            rng=rng,
        )
        state.extras[_OPTIMIZER_RNG_STATE_KEY] = rng.randint(
            _INT32_MIN,
            _INT32_MAX,
            3,
        ).astype(np.int64)
        state.pos = _rescale_umap_embedding(coordinates=initial_embedding)
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
        f"extras.{_FUZZY_HEAD_KEY}",
        f"extras.{_FUZZY_TAIL_KEY}",
        f"extras.{_FUZZY_WEIGHT_KEY}",
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
        del ctx
        head = state.extras[_FUZZY_HEAD_KEY]
        tail = state.extras[_FUZZY_TAIL_KEY]
        weight = state.extras[_FUZZY_WEIGHT_KEY]
        n_epochs = state.extras[_N_EPOCHS_KEY]
        default_epochs = 500 if problem.num_nodes <= 10_000 else 200
        positive_head, positive_tail, positive_weight, epochs_per_sample = _select_positive_edges(
            head=head,
            tail=tail,
            weight=weight,
            n_epochs=n_epochs,
            default_epochs=default_epochs,
        )
        state.extras[_FUZZY_HEAD_KEY] = positive_head
        state.extras[_FUZZY_TAIL_KEY] = positive_tail
        state.extras[_FUZZY_WEIGHT_KEY] = positive_weight
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
        f"extras.{_OPTIMIZER_RNG_STATE_KEY}",
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
            rng_state=state.extras.get(_OPTIMIZER_RNG_STATE_KEY),
        )
        return state


@register_op
class FinalizeUMAPPositions(Op):
    """Cast final UMAP positions to the requested output device and dtype."""

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
        """Cast output positions without changing UMAP's final coordinate frame."""
        del ctx
        if state.pos is None:
            raise ValueError("FinalizeUMAPPositions requires state.pos to be set.")
        device = layout_device(edge_index=problem.edge_index, node_sizes=problem.node_sizes)
        state.pos = state.pos.to(dtype=torch.float32, device=device)
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
