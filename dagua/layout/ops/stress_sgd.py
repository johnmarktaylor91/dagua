"""Stress-SGD registered operations.

This module contains the algorithmic primitives needed by the composable
pipeline implementation. The operations are intentionally registered so they can
be composed and reused by any stress-SGD variant that requires the same
building blocks.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass
from typing import Any, ClassVar, Iterator, Optional, Tuple, Union

import numpy as np
import torch

from dagua.layout.ops.base import Op
from dagua.layout.ops.graph_utils import bfs_distances, dijkstra_distances
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_AUTO_FULL_EPOCH_THRESHOLD = 1_000
_AUTO_SAMPLE_THRESHOLD = 1_000
_DEFAULT_EPS = 0.01
_DEFAULT_MAX_EXACT_NODES = 10_000
_MAX_PIVOTS = 200
_UNREACHED = -1
_DISCONNECTED_FALLBACK_SCALE = 10.0

_STRESS_SGD_ADJ_KEY = "stress_sgd_adjacency"
_STRESS_SGD_CONNECTED_KEY = "stress_sgd_connected"
_STRESS_SGD_DEVICE_KEY = "stress_sgd_device"
_STRESS_SGD_EXACT_KEY = "stress_sgd_exact_mode"
_STRESS_SGD_NUM_NODES_KEY = "stress_sgd_num_nodes"
_STRESS_SGD_RNG_KEY = "stress_sgd_rng"
_STRESS_SGD_TRACE_KEY = "stress_sgd_traces"
_STRESS_SGD_WEIGHTED_KEY = "stress_sgd_weighted"


@dataclass(frozen=True)
class InitializeStressSGDStateConfig:
    """Configuration for :class:`InitializeStressSGDState`.

    Parameters
    ----------
    trace_every : int, default=0
        Trace interval passed through from the public adapter.
    disconnected_fallback_scale : float, default=10.0
        Coordinate scale used by the disconnected-graph fallback layout.
    reference_disconnected_policy : bool, default=False
        Whether to mirror ``s_gd2`` adapter behavior for edge-case graphs.
    independent_shuffle_rng : bool, default=False
        Whether pair-order shuffling should use a fresh seed stream independent
        from the Python initialization draws.
    """

    trace_every: int = 0
    disconnected_fallback_scale: float = _DISCONNECTED_FALLBACK_SCALE
    reference_disconnected_policy: bool = False
    independent_shuffle_rng: bool = False


@dataclass(frozen=True)
class PrepareStressSGDTermsConfig:
    """Configuration for :class:`PrepareStressSGDTerms`.

    Parameters
    ----------
    max_exact_nodes : int, default=10000
        Maximum node count for materializing all-pairs exact terms.
    max_pivots : int, default=200
        Upper bound on the approximate-mode pivot count.
    exact_float64_terms : bool, default=False
        Whether exact distances and weights should be stored as ``float64``.
    reference_term_order : bool, default=False
        Whether to materialize exact terms in native ``s_gd2`` traversal order.
    """

    max_exact_nodes: int = _DEFAULT_MAX_EXACT_NODES
    max_pivots: int = _MAX_PIVOTS
    exact_float64_terms: bool = False
    reference_term_order: bool = False


@dataclass(frozen=True)
class RunStressSGDExactScheduleConfig:
    """Configuration for :class:`RunStressSGDExactSchedule`.

    Parameters
    ----------
    steps : int, default=30
        Number of SGD epochs.
    eps : float, default=0.01
        Final schedule shrinkage factor.
    trace_every : int, default=0
        Snapshot interval, zero disables traces.
    """

    steps: int = 30
    eps: float = _DEFAULT_EPS
    trace_every: int = 0


@dataclass(frozen=True)
class RunStressSGDApproximateScheduleConfig:
    """Configuration for :class:`RunStressSGDApproximateSchedule`.

    Parameters
    ----------
    steps : int, default=30
        Number of optimization epochs.
    eps : float, default=0.01
        Final schedule shrinkage factor.
    sample_size : int | str, default="auto"
        Pivot sampler budget (``"auto"`` or explicit positive integer).
    trace_every : int, default=0
        Snapshot interval, zero disables traces.
    auto_full_epoch_threshold : int, default=1000
        Node threshold below which ``sample_size="auto"`` expands to a full epoch.
    auto_sample_threshold : int, default=1000
        Fixed sampling budget used by ``sample_size="auto"`` on larger graphs.
    """

    steps: int = 30
    eps: float = _DEFAULT_EPS
    sample_size: Union[int, str] = "auto"
    trace_every: int = 0
    auto_full_epoch_threshold: int = _AUTO_FULL_EPOCH_THRESHOLD
    auto_sample_threshold: int = _AUTO_SAMPLE_THRESHOLD


def _bfs_graph_distances(
    adjacency: list[list[tuple[int, float]]],
    source: int,
) -> np.ndarray:
    """Compute unweighted graph distances from one source.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Undirected adjacency list as ``(neighbor, weight)`` tuples.
    source : int
        Source node index.

    Returns
    -------
    np.ndarray
        Integer distances with shape ``[N]``. Unreachable nodes are ``-1``.
    """
    return bfs_distances(adjacency, source)


def _dijkstra_graph_distances(
    adjacency: list[list[tuple[int, float]]],
    source: int,
) -> np.ndarray:
    """Compute weighted graph distances from one source.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Undirected adjacency list as ``(neighbor, weight)`` tuples.
    source : int
        Source node index.

    Returns
    -------
    np.ndarray
        Float distances with shape ``[N]``. Unreachable nodes are ``inf``.
    """
    return dijkstra_distances(adjacency, source)


def _graph_distances(
    adjacency: list[list[tuple[int, float]]],
    source: int,
    weighted: bool,
) -> np.ndarray:
    """Compute graph distances from a source node.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Undirected adjacency list as ``(neighbor, weight)`` tuples.
    source : int
        Source node index.
    weighted : bool
        Whether to use Dijkstra (``True``) or BFS (``False``).

    Returns
    -------
    np.ndarray
        Distances with unreachable entries set to ``-1`` for BFS and ``inf`` for
        Dijkstra.
    """
    if not weighted:
        return _bfs_graph_distances(adjacency, source)

    return np.where(
        np.isinf(_dijkstra_graph_distances(adjacency, source)),
        float(_UNREACHED),
        _dijkstra_graph_distances(adjacency, source),
    )


def _is_connected(
    adjacency: list[list[tuple[int, float]]],
) -> bool:
    """Return whether all nodes are reachable from node zero.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Undirected adjacency list as ``(neighbor, weight)`` tuples.

    Returns
    -------
    bool
        ``True`` when all nodes are connected to index zero.
    """
    if len(adjacency) <= 1:
        return True
    return bool(np.all(_bfs_graph_distances(adjacency, 0) >= 0))


def _schedule_bounds(distance_data: np.ndarray) -> tuple[float, float]:
    """Compute minimum and maximum positive graph distances.

    Parameters
    ----------
    distance_data : np.ndarray
        A matrix or vector of shortest-path distances.

    Returns
    -------
    tuple[float, float]
        ``(d_min, d_max)`` where each value is at least ``1.0``.
    """
    positive_distances = distance_data[distance_data > 0]
    if positive_distances.size == 0:
        return 1.0, 1.0

    d_min = float(positive_distances.min())
    d_max = float(positive_distances.max())
    return max(d_min, 1.0), max(d_max, d_min, 1.0)


def _learning_rate(
    step_index: int,
    steps: int,
    d_min: float,
    d_max: float,
    eps: float = _DEFAULT_EPS,
) -> float:
    """Evaluate the classic exponential SGD schedule from graph distances.

    Parameters
    ----------
    step_index : int
        Zero-based step index.
    steps : int
        Total number of steps.
    d_min : float
        Minimum positive graph distance.
    d_max : float
        Maximum positive graph distance.
    eps : float
        Final schedule shrinkage factor.

    Returns
    -------
    float
        Step size ``eta_t``.
    """
    if d_min <= 0.0 or d_max <= 0.0:
        raise ValueError("Distance bounds must be positive.")

    eta_max = d_max * d_max
    eta_min = eps * d_min * d_min
    if steps <= 1:
        return eta_max

    decay = math.log(eta_max / eta_min) / float(steps - 1)
    return eta_max * math.exp(-decay * float(step_index))


def _schedule_from_weights(
    steps: int,
    w_min: float,
    w_max: float,
    eps: float,
) -> np.ndarray:
    """Compute exact-schedule values from pair weights.

    Parameters
    ----------
    steps : int
        Number of optimization epochs.
    w_min : float
        Minimum pair weight.
    w_max : float
        Maximum pair weight.
    eps : float
        Final schedule shrinkage factor.

    Returns
    -------
    np.ndarray
        Schedule tensor with shape ``[steps]``.
    """
    if steps <= 0:
        return np.empty((0,), dtype=np.float64)

    safe_min = max(w_min, np.finfo(np.float64).tiny)
    safe_max = max(w_max, np.finfo(np.float64).tiny)
    eta_max = 1.0 / safe_min
    eta_min = eps / safe_max
    if steps == 1:
        return np.asarray([eta_max], dtype=np.float64)

    decay = math.log(eta_max / eta_min) / float(steps - 1)
    return np.asarray(
        [eta_max * math.exp(-decay * float(step_index)) for step_index in range(steps)],
        dtype=np.float64,
    )


def _resolve_sample_size(
    num_nodes: int,
    sample_size: Union[int, str],
    auto_full_epoch_threshold: int,
    auto_sample_threshold: int,
) -> int:
    """Resolve the large-graph sample budget.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    sample_size : int | str
        ``"auto"`` or a positive sample count.
    auto_full_epoch_threshold : int
        Node threshold below which ``"auto"`` expands to a full epoch.
    auto_sample_threshold : int
        Fixed sampling budget used by ``"auto"`` on larger graphs.

    Returns
    -------
    int
        The resolved sample count.

    Raises
    ------
    ValueError
        When ``sample_size`` is not valid.
    """
    if isinstance(sample_size, str):
        if sample_size != "auto":
            raise ValueError("sample_size must be a positive integer or 'auto'.")
        if num_nodes <= auto_full_epoch_threshold:
            return num_nodes
        return auto_sample_threshold

    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    return sample_size


def _choose_pivots(
    num_nodes: int,
    max_pivots: int,
    rng: Any,
) -> np.ndarray:
    """Choose pivot nodes for approximate distance queries.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    max_pivots : int
        Upper bound for the pivot count.
    rng : numpy RandomState
        Random state to consume for pivot sampling.

    Returns
    -------
    np.ndarray
        Pivot indices with shape ``[P]`` and dtype ``int32``.
    """
    if num_nodes <= max_pivots:
        return np.arange(num_nodes, dtype=np.int32)

    return rng.choice(num_nodes, size=max_pivots, replace=False).astype(np.int32, copy=False)


def _compute_pivot_distances(
    adjacency: list[list[tuple[int, float]]],
    pivots: np.ndarray,
    weighted: bool,
) -> np.ndarray:
    """Compute pivot-to-node shortest-path distances.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Undirected adjacency list.
    pivots : np.ndarray
        Pivot IDs.
    weighted : bool
        Whether to use weighted distances.

    Returns
    -------
    np.ndarray
        Pivot distances with shape ``[P, N]`` and dtype ``float32``.
    """
    num_nodes = len(adjacency)
    if pivots.size == 0:
        return np.empty((0, num_nodes), dtype=np.float32)

    distances = np.empty((int(pivots.size), num_nodes), dtype=np.float32)
    for pivot_index, pivot_node in enumerate(pivots.tolist()):
        distances[pivot_index] = _graph_distances(
            adjacency,
            int(pivot_node),
            weighted=weighted,
        ).astype(np.float32)
    return distances


def _approx_distance(
    source_index: int,
    target_index: int,
    pivot_dist: np.ndarray,
) -> float:
    """Approximate one shortest-path distance from pivot data.

    Parameters
    ----------
    source_index : int
        Source node index.
    target_index : int
        Target node index.
    pivot_dist : np.ndarray
        Pivot distances with shape ``[P, N]``.

    Returns
    -------
    float
        A symmetric approximation of graph distance.
    """
    if pivot_dist.size == 0:
        return 1.0

    pivot_i = pivot_dist[:, source_index]
    pivot_j = pivot_dist[:, target_index]
    best_lower = float(np.max(np.abs(pivot_i - pivot_j)))
    best_upper = float(np.min(pivot_i + pivot_j))
    if math.isfinite(best_upper):
        return max((best_lower + best_upper) * 0.5, 1.0)
    return max(best_lower, 1.0)


def _pair_count(num_nodes: int) -> int:
    """Count unordered node pairs.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.

    Returns
    -------
    int
        Number of pairs in ``triu`` order.
    """
    return (num_nodes * (num_nodes - 1)) // 2


def _build_exact_terms(
    adjacency: list[list[tuple[int, float]]],
    weighted: bool,
    exact_float64_terms: bool = False,
    reference_term_order: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build full upper-triangle stress-SGD term data.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Undirected adjacency list.
    weighted : bool
        Whether to use weighted shortest-path distances.
    exact_float64_terms : bool, default=False
        Store exact distance and weight parameters as ``float64`` to match the
        native ``s_gd2`` term representation.
    reference_term_order : bool, default=False
        Build terms in native ``s_gd2`` BFS/Dijkstra discovery order instead of
        target-index order.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(sources, targets, distances, weights)`` with matching shapes.
    """
    if reference_term_order:
        return _build_exact_terms_reference_order(
            adjacency=adjacency,
            weighted=weighted,
            exact_float64_terms=exact_float64_terms,
        )

    num_nodes = len(adjacency)
    num_terms = _pair_count(num_nodes)
    sources = np.empty((num_terms,), dtype=np.int32)
    targets = np.empty((num_terms,), dtype=np.int32)
    float_dtype = np.float64 if exact_float64_terms else np.float32
    distances = np.empty((num_terms,), dtype=float_dtype)
    weights = np.empty((num_terms,), dtype=float_dtype)

    write_index = 0
    for source_index in range(num_nodes - 1):
        source_distances = _graph_distances(adjacency, source_index, weighted=weighted)
        for target_index in range(source_index + 1, num_nodes):
            graph_distance = float(source_distances[target_index])
            if graph_distance <= 0:
                raise ValueError("Stress-SGD requires a connected graph.")

            sources[write_index] = source_index
            targets[write_index] = target_index
            distances[write_index] = graph_distance
            weights[write_index] = 1.0 / (graph_distance * graph_distance)
            write_index += 1

    return sources, targets, distances, weights


def _build_exact_terms_reference_order(
    adjacency: list[list[tuple[int, float]]],
    weighted: bool,
    exact_float64_terms: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build exact stress terms in native ``s_gd2`` traversal order.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Undirected adjacency list.
    weighted : bool
        Whether to use weighted Dijkstra traversal.
    exact_float64_terms : bool, default=False
        Store exact distance and weight parameters as ``float64``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(sources, targets, distances, weights)`` with matching shapes.
    """
    num_nodes = len(adjacency)
    num_terms = _pair_count(num_nodes)
    sources = np.empty((num_terms,), dtype=np.int32)
    targets = np.empty((num_terms,), dtype=np.int32)
    float_dtype = np.float64 if exact_float64_terms else np.float32
    distances = np.empty((num_terms,), dtype=float_dtype)
    weights = np.empty((num_terms,), dtype=float_dtype)

    write_index = 0
    terms_size_goal = 0
    for source_index in range(num_nodes - 1):
        terms_size_goal += num_nodes - source_index - 1
        stream = (
            _weighted_reference_term_stream(adjacency, source_index)
            if weighted
            else _unweighted_reference_term_stream(adjacency, source_index)
        )
        for target_index, graph_distance in stream:
            sources[write_index] = source_index
            targets[write_index] = target_index
            distances[write_index] = graph_distance
            weights[write_index] = 1.0 / (graph_distance * graph_distance)
            write_index += 1
            if write_index == terms_size_goal:
                break

        if write_index != terms_size_goal:
            raise ValueError("Stress-SGD requires a connected graph.")

    return sources, targets, distances, weights


def _unweighted_reference_term_stream(
    adjacency: list[list[tuple[int, float]]],
    source: int,
) -> Iterator[tuple[int, float]]:
    """Yield unweighted terms in native ``s_gd2`` BFS discovery order.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Undirected adjacency list.
    source : int
        Source node for the BFS traversal.

    Yields
    ------
    tuple[int, float]
        ``(target, graph_distance)`` for pairs with ``source < target``.
    """
    distances = [_UNREACHED] * len(adjacency)
    distances[source] = 0
    queue: deque[int] = deque([source])

    while queue:
        current = queue.popleft()
        next_distance = distances[current] + 1
        for neighbor, _ in adjacency[current]:
            if distances[neighbor] != _UNREACHED:
                continue
            distances[neighbor] = next_distance
            queue.append(neighbor)
            if source < neighbor:
                yield neighbor, float(next_distance)


def _weighted_reference_term_stream(
    adjacency: list[list[tuple[int, float]]],
    source: int,
) -> Iterator[tuple[int, float]]:
    """Yield weighted terms in native ``s_gd2`` Dijkstra pop order.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Undirected weighted adjacency list.
    source : int
        Source node for the Dijkstra traversal.

    Yields
    ------
    tuple[int, float]
        ``(target, graph_distance)`` for pairs with ``source < target``.
    """
    distances = [math.inf] * len(adjacency)
    visited = [False] * len(adjacency)
    distances[source] = 0.0
    heap: list[tuple[float, int]] = [(0.0, source)]

    while heap:
        graph_distance, current = heapq.heappop(heap)
        if visited[current]:
            continue
        visited[current] = True
        if source < current:
            yield current, graph_distance

        for neighbor, weight in adjacency[current]:
            new_distance = graph_distance + weight
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                heapq.heappush(heap, (new_distance, neighbor))


def _sample_pairs(
    num_nodes: int,
    sample_size: int,
    rng: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample node pairs from ``[0, num_nodes)``.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    sample_size : int
        Number of pair attempts.
    rng : numpy RandomState
        Random state used for sampling.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Source and target index arrays.
    """
    if num_nodes <= 1:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    sources = rng.randint(0, num_nodes, size=sample_size).astype(np.int64, copy=False)
    targets = rng.randint(0, num_nodes, size=sample_size).astype(np.int64, copy=False)
    valid_pairs = sources != targets
    return sources[valid_pairs], targets[valid_pairs]


def _trace_snapshot(
    traces: list[torch.Tensor],
    positions: np.ndarray,
    device: torch.device,
) -> None:
    """Append one trace snapshot.

    Parameters
    ----------
    traces : list[torch.Tensor]
        Trace destination list.
    positions : np.ndarray
        Current coordinates with shape ``[N, 2]``.
    device : torch.device
        Destination device for traced positions.
    """
    traces.append(torch.from_numpy(positions.astype(np.float32, copy=True)).to(device=device))


def _apply_pair_update(
    positions: np.ndarray,
    source_index: int,
    target_index: int,
    target_distance: float,
    weight: float,
    eta: float,
) -> None:
    """Apply one sequential Stress-SGD pair update.

    Parameters
    ----------
    positions : np.ndarray
        Positions with shape ``[N, 2]``.
    source_index : int
        Source node index.
    target_index : int
        Target node index.
    target_distance : float
        Desired shortest-path distance.
    weight : float
        Stress weight ``1 / d^2``.
    eta : float
        Current step size.
    """
    mu = min(eta * weight, 1.0)
    dx = float(positions[source_index, 0] - positions[target_index, 0])
    dy = float(positions[source_index, 1] - positions[target_index, 1])
    magnitude = math.hypot(dx, dy)
    if magnitude <= 0.0:
        return

    ratio = mu * (magnitude - target_distance) / (2.0 * magnitude)
    positions[source_index, 0] -= ratio * dx
    positions[target_index, 0] += ratio * dx
    positions[source_index, 1] -= ratio * dy
    positions[target_index, 1] += ratio * dy


def _disconnected_fallback_layout(
    num_nodes: int,
    seed: int,
    device: torch.device,
    scale: float,
) -> torch.Tensor:
    """Return deterministic disconnected fallback coordinates.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    seed : int
        Seed for ``torch.Generator``.
    device : torch.device
        Output device.
    scale : float
        Coordinate scale factor applied to the fallback Gaussian samples.

    Returns
    -------
    torch.Tensor
        Fallback coordinates with shape ``[N, 2]``.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    positions = torch.randn((num_nodes, 2), generator=generator, dtype=torch.float32) * scale
    return positions.to(device=device)


def _initial_positions_from_state(state: SolveState, num_nodes: int) -> np.ndarray:
    """Resolve exact-mode initial positions.

    Parameters
    ----------
    state : SolveState
        Current solve state, optionally carrying caller-provided warm-start
        positions in ``state.pos``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    np.ndarray
        Initial coordinates with shape ``[N, 2]`` and dtype ``float64``.
    """
    if state.pos is None:
        return np.random.rand(num_nodes, 2)
    if state.pos.ndim != 2 or tuple(state.pos.shape) != (num_nodes, 2):
        raise ValueError("Stress-SGD init_pos must be shape [N, 2].")
    return state.pos.detach().cpu().numpy().astype(np.float64, copy=True)


def _has_usable_edges(adjacency: list[list[tuple[int, float]]]) -> bool:
    """Return whether an adjacency list contains any non-self graph edge.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Undirected adjacency list after self-loop filtering.

    Returns
    -------
    bool
        ``True`` when at least one adjacency entry remains.
    """
    return any(len(neighbors) > 0 for neighbors in adjacency)


@register_op
class InitializeStressSGDState(Op):
    """Prepare graph-wide Stress-SGD state and disconnected fallback handling.

    This op resolves graph connectivity and connected-component fallback behavior,
    then captures the exact flags and RNG state needed by the schedule builders.
    """

    name: ClassVar[str] = "stress_sgd_initialize_state"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("adjacency",)
    writes: ClassVar[Tuple[str, ...]] = ("extras", "pos", "converged")
    requires: ClassVar[Tuple[str, ...]] = ("adjacency",)

    def __init__(
        self,
        trace_every: int = 0,
        reference_disconnected_policy: bool = False,
        independent_shuffle_rng: bool = False,
    ) -> None:
        """Create an initializer with classic fallback and trace settings.

        Parameters
        ----------
        trace_every : int
            Trace interval passed through from the public adapter.
        reference_disconnected_policy : bool, default=False
            Return zeros for edgeless graphs and raise for disconnected graphs,
            matching the reference adapter/native split.
        independent_shuffle_rng : bool, default=False
            Use a fresh NumPy RNG stream for pair-order shuffling. Native
            ``s_gd2`` initializes coordinates in Python, then seeds the C++
            shuffle stream independently with the same seed.
        """
        self.config = InitializeStressSGDStateConfig(
            trace_every=trace_every,
            reference_disconnected_policy=reference_disconnected_policy,
            independent_shuffle_rng=independent_shuffle_rng,
        )

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Prepare shared state and disconnected fallback output.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout problem inputs.
        state : SolveState
            Mutable state cache.
        ctx : RuntimeContext
            Runtime context (unused).

        Returns
        -------
        SolveState
            Updated with deterministic state metadata and fallback output when
            disconnected.
        """
        del ctx

        num_nodes = int(problem.num_nodes)
        device = problem.edge_index.device
        adjacency = state.adjacency

        state.extras[_STRESS_SGD_NUM_NODES_KEY] = num_nodes
        state.extras[_STRESS_SGD_DEVICE_KEY] = device
        state.extras[_STRESS_SGD_ADJ_KEY] = adjacency
        state.extras[_STRESS_SGD_WEIGHTED_KEY] = problem.edge_weights is not None

        if num_nodes <= 1:
            state.pos = torch.zeros((num_nodes, 2), dtype=torch.float32, device=device)
            state.extras[_STRESS_SGD_CONNECTED_KEY] = True
            state.converged = True
            if self.config.trace_every > 0:
                state.extras[_STRESS_SGD_TRACE_KEY] = []
            return state

        if not isinstance(adjacency, list):
            raise ValueError("InitializeStressSGDState requires state.adjacency to be a list.")
        if self.config.reference_disconnected_policy and not _has_usable_edges(adjacency):
            fallback_position = torch.zeros((num_nodes, 2), dtype=torch.float32, device=device)
            state.pos = fallback_position
            state.extras[_STRESS_SGD_CONNECTED_KEY] = False
            state.converged = True
            if self.config.trace_every > 0:
                state.extras[_STRESS_SGD_TRACE_KEY] = [fallback_position.clone()]
            return state

        connected = _is_connected(adjacency)
        state.extras[_STRESS_SGD_CONNECTED_KEY] = bool(connected)
        if not connected:
            if self.config.reference_disconnected_policy:
                raise ValueError("Stress-SGD reference fidelity mode requires a connected graph.")
            fallback_position = _disconnected_fallback_layout(
                num_nodes=num_nodes,
                seed=problem.seed,
                device=device,
                scale=self.config.disconnected_fallback_scale,
            )
            state.pos = fallback_position
            state.converged = True
            if self.config.trace_every > 0:
                state.extras[_STRESS_SGD_TRACE_KEY] = [fallback_position.clone()]
            else:
                state.extras.pop(_STRESS_SGD_TRACE_KEY, None)
            return state

        # ``s_gd2`` draws initial coordinates in Python, then its native kernel
        # seeds pair-order shuffling independently. Legacy mode keeps the
        # archive-compatible shared stream; fidelity mode separates the streams.
        np.random.seed(problem.seed)
        state.extras[_STRESS_SGD_RNG_KEY] = (
            np.random.RandomState(problem.seed)
            if self.config.independent_shuffle_rng
            else np.random
        )
        state.extras.pop(_STRESS_SGD_CONNECTED_KEY, None)
        return state


@register_op
class PrepareStressSGDTerms(Op):
    """Prepare exact pair tensors or pivot approximations for Stress-SGD."""

    name: ClassVar[str] = "stress_sgd_prepare_terms"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("adjacency", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("distance_matrix", "extras")
    requires: ClassVar[Tuple[str, ...]] = ("adjacency", "extras")

    def __init__(
        self,
        max_exact_nodes: int = _DEFAULT_MAX_EXACT_NODES,
        exact_float64_terms: bool = False,
        reference_term_order: bool = False,
    ) -> None:
        """Create a configured term builder.

        Parameters
        ----------
        max_exact_nodes : int
            Maximum number of nodes for full all-pairs tensor construction.
        exact_float64_terms : bool, default=False
            Store exact distances and weights as ``float64`` for reference
            fidelity.
        reference_term_order : bool, default=False
            Build exact terms in native ``s_gd2`` traversal order.
        """
        self.config = PrepareStressSGDTermsConfig(
            max_exact_nodes=max_exact_nodes,
            exact_float64_terms=exact_float64_terms,
            reference_term_order=reference_term_order,
        )

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Populate exact pairs or pivot approximations.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable state cache.
        ctx : RuntimeContext
            Runtime context (unused).

        Returns
        -------
        SolveState
            Updated with term tensors in ``state.extras``.
        """
        del ctx

        if state.converged:
            return state

        adjacency = state.extras.get(_STRESS_SGD_ADJ_KEY)
        if not isinstance(adjacency, list):
            raise ValueError("PrepareStressSGDTerms requires state.adjacency list data.")

        num_nodes = state.extras[_STRESS_SGD_NUM_NODES_KEY]
        weighted = bool(state.extras.get(_STRESS_SGD_WEIGHTED_KEY, False))
        rng = state.extras.get(_STRESS_SGD_RNG_KEY)

        if num_nodes <= self.config.max_exact_nodes:
            state.extras[_STRESS_SGD_EXACT_KEY] = True
            sources, targets, distances, weights = _build_exact_terms(
                adjacency=adjacency,
                weighted=weighted,
                exact_float64_terms=self.config.exact_float64_terms,
                reference_term_order=self.config.reference_term_order,
            )
            state.extras["stress_sgd_sources"] = sources
            state.extras["stress_sgd_targets"] = targets
            state.extras["stress_sgd_distances"] = distances
            state.extras["stress_sgd_weights"] = weights
            return state

        state.extras[_STRESS_SGD_EXACT_KEY] = False
        pivots = _choose_pivots(
            num_nodes=num_nodes,
            max_pivots=min(num_nodes, self.config.max_pivots),
            rng=rng,
        )
        pivot_dist = _compute_pivot_distances(
            adjacency=adjacency,
            pivots=pivots,
            weighted=weighted,
        )
        # Large graphs keep only pivot distances to avoid the quadratic memory
        # blow-up of explicit all-pairs terms.
        state.distance_matrix = torch.from_numpy(pivot_dist)
        return state


@register_op
class RunStressSGDExactSchedule(Op):
    """Run the exact upper-triangle Stress-SGD schedule to completion."""

    name: ClassVar[str] = "stress_sgd_run_exact_schedule"
    category: ClassVar[OpCategory] = OpCategory.OPTIMIZE
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("pos", "extras", "converged")
    requires: ClassVar[Tuple[str, ...]] = ("extras",)

    def __init__(self, steps: int = 30, eps: float = _DEFAULT_EPS, trace_every: int = 0) -> None:
        """Create a configured exact-schedule runner.

        Parameters
        ----------
        steps : int
            Number of SGD epochs.
        eps : float
            Final schedule shrinkage factor.
        trace_every : int
            Optional trace interval, zero disables snapshots.
        """
        self.config = RunStressSGDExactScheduleConfig(
            steps=steps,
            eps=eps,
            trace_every=trace_every,
        )

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run the sequential exact pairwise Stress-SGD kernel.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs (unused in this step).
        state : SolveState
            Mutable layout state.
        ctx : RuntimeContext
            Runtime context (unused).

        Returns
        -------
        SolveState
            State with final positions and optional trace snapshots.
        """
        del problem, ctx

        if state.converged:
            return state
        if not state.extras.get(_STRESS_SGD_EXACT_KEY):
            return state

        adjacency = state.extras.get(_STRESS_SGD_ADJ_KEY)
        if not isinstance(adjacency, list):
            raise ValueError("RunStressSGDExactSchedule requires adjacency data in state.extras.")

        num_nodes = state.extras[_STRESS_SGD_NUM_NODES_KEY]
        device = state.extras[_STRESS_SGD_DEVICE_KEY]
        rng = state.extras[_STRESS_SGD_RNG_KEY]
        sources = state.extras["stress_sgd_sources"]
        targets = state.extras["stress_sgd_targets"]
        distances = state.extras["stress_sgd_distances"]
        weights = state.extras["stress_sgd_weights"]
        config = self.config

        # Default positions use the shared NumPy RNG to match s_gd2's draw
        # order; caller-provided positions enable direct parity probes.
        positions = _initial_positions_from_state(state, num_nodes)
        traces: list[torch.Tensor] = []
        schedule = _schedule_from_weights(
            steps=config.steps,
            w_min=float(weights.min()),
            w_max=float(weights.max()),
            eps=config.eps,
        )
        order = np.arange(sources.shape[0], dtype=np.int64)

        if config.trace_every > 0 and config.steps == 0:
            _trace_snapshot(traces, positions, device)

        for step_index, eta in enumerate(schedule):
            # The classic implementation reshuffles pair order every epoch so
            # sequential updates do not bias toward a fixed traversal.
            rng.shuffle(order)
            for term_index in order:
                source_index = int(sources[term_index])
                target_index = int(targets[term_index])
                _apply_pair_update(
                    positions=positions,
                    source_index=source_index,
                    target_index=target_index,
                    target_distance=float(distances[term_index]),
                    weight=float(weights[term_index]),
                    eta=float(eta),
                )

            if config.trace_every > 0 and (step_index + 1) % config.trace_every == 0:
                _trace_snapshot(traces, positions, device)

        state.pos = torch.from_numpy(positions.astype(np.float32, copy=False)).to(device=device)
        state.extras[_STRESS_SGD_TRACE_KEY] = traces
        state.converged = True
        return state


@register_op
class RunStressSGDApproximateSchedule(Op):
    """Run the pivot-approximate large-graph Stress-SGD schedule."""

    name: ClassVar[str] = "stress_sgd_run_approximate_schedule"
    category: ClassVar[OpCategory] = OpCategory.OPTIMIZE
    reads: ClassVar[Tuple[str, ...]] = ("distance_matrix", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("pos", "extras", "converged")
    requires: ClassVar[Tuple[str, ...]] = ("distance_matrix", "extras")

    def __init__(
        self,
        steps: int = 30,
        eps: float = _DEFAULT_EPS,
        sample_size: Union[int, str] = "auto",
        trace_every: int = 0,
    ) -> None:
        """Create a configured approximate schedule runner.

        Parameters
        ----------
        steps : int
            Number of optimization epochs.
        eps : float
            Final schedule shrinkage factor.
        sample_size : int | str
            Pivot sampler budget (``"auto"`` or explicit positive integer).
        trace_every : int
            Snapshot interval, zero disables traces.
        """
        self.config = RunStressSGDApproximateScheduleConfig(
            steps=steps,
            eps=eps,
            sample_size=sample_size,
            trace_every=trace_every,
        )

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run the pivot approximation variant.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs (unused in this step).
        state : SolveState
            Mutable layout state.
        ctx : RuntimeContext
            Runtime context (unused).

        Returns
        -------
        SolveState
            State with final positions and optional trace snapshots.
        """
        del problem, ctx

        if state.converged:
            return state
        if state.extras.get(_STRESS_SGD_EXACT_KEY, True):
            return state

        num_nodes = state.extras[_STRESS_SGD_NUM_NODES_KEY]
        device = state.extras[_STRESS_SGD_DEVICE_KEY]
        rng = state.extras[_STRESS_SGD_RNG_KEY]
        pivot_distances = state.distance_matrix
        config = self.config
        if not isinstance(pivot_distances, torch.Tensor):
            raise ValueError(
                "RunStressSGDApproximateSchedule requires pivot distances in state.distance_matrix."
            )
        pivot_dist = pivot_distances.to(dtype=torch.float64, device="cpu").numpy()
        effective_sample_size = _resolve_sample_size(
            num_nodes,
            config.sample_size,
            auto_full_epoch_threshold=config.auto_full_epoch_threshold,
            auto_sample_threshold=config.auto_sample_threshold,
        )

        positions = np.random.rand(num_nodes, 2)
        traces: list[torch.Tensor] = []
        d_min, d_max = _schedule_bounds(pivot_dist)
        use_full_epoch = (
            effective_sample_size >= num_nodes and num_nodes <= config.auto_full_epoch_threshold
        )

        full_sources: Optional[np.ndarray] = None
        full_targets: Optional[np.ndarray] = None
        full_order: Optional[np.ndarray] = None
        if use_full_epoch:
            full_sources, full_targets = np.triu_indices(num_nodes, k=1)
            full_order = np.arange(full_sources.shape[0], dtype=np.int64)

        if config.trace_every > 0 and config.steps == 0:
            _trace_snapshot(traces, positions, device)

        for step_index in range(config.steps):
            eta = _learning_rate(step_index, config.steps, d_min, d_max, eps=config.eps)
            if use_full_epoch:
                assert full_sources is not None
                assert full_targets is not None
                assert full_order is not None
                # Small "approximate" problems still expand to the full pair set
                # when ``sample_size="auto"`` so the classic branch is preserved.
                rng.shuffle(full_order)
                for pair_index in full_order:
                    source_index = int(full_sources[pair_index])
                    target_index = int(full_targets[pair_index])
                    target_distance = _approx_distance(source_index, target_index, pivot_dist)
                    weight = 1.0 / float(target_distance * target_distance)
                    _apply_pair_update(
                        positions=positions,
                        source_index=source_index,
                        target_index=target_index,
                        target_distance=target_distance,
                        weight=weight,
                        eta=eta,
                    )
            else:
                sampled_sources, sampled_targets = _sample_pairs(
                    num_nodes=num_nodes,
                    sample_size=effective_sample_size,
                    rng=rng,
                )
                for source_index, target_index in zip(
                    sampled_sources.tolist(),
                    sampled_targets.tolist(),
                ):
                    target_distance = _approx_distance(
                        source_index=source_index,
                        target_index=target_index,
                        pivot_dist=pivot_dist,
                    )
                    weight = 1.0 / float(target_distance * target_distance)
                    _apply_pair_update(
                        positions=positions,
                        source_index=source_index,
                        target_index=target_index,
                        target_distance=target_distance,
                        weight=weight,
                        eta=eta,
                    )

            if config.trace_every > 0 and (step_index + 1) % config.trace_every == 0:
                _trace_snapshot(traces, positions, device)

        state.pos = torch.from_numpy(positions.astype(np.float32, copy=False)).to(device=device)
        state.extras[_STRESS_SGD_TRACE_KEY] = traces
        state.converged = True
        return state
