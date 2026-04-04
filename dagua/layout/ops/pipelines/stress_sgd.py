"""Stress-SGD expressed as a composable ops pipeline.

Produces BIT-IDENTICAL output to the classic ``layout_stress_sgd`` for both
the exact (small-graph) and approximate (large-graph pivot) code paths.  All
numeric work is delegated to the classic module's helper functions -- the ops
only orchestrate the call sequence.
"""

from __future__ import annotations

import math
from collections import deque
from typing import ClassVar, Optional, Tuple, Union

import numpy as np
import torch

from dagua.layout.ops.graph_utils import (
    build_undirected_adjacency as _build_undirected_adjacency,
)

# ---------------------------------------------------------------------------
# Algorithm-specific constants and functions copied from
# dagua/layout/classic/stress_sgd.py (bit-identical)
# ---------------------------------------------------------------------------

_AUTO_FULL_EPOCH_THRESHOLD = 1_000
_AUTO_SAMPLE_THRESHOLD = 1_000
_DEFAULT_EPS = 0.01
_DEFAULT_MAX_EXACT_NODES = 10_000
_MAX_PIVOTS = 200
_UNREACHED = -1
_DISCONNECTED_FALLBACK_SCALE = 10.0
_UNREACHED_FLOAT = 1e6


def _bfs_distances(
    adjacency: list[list[tuple[int, float]]],
    source: int,
) -> np.ndarray:
    """Compute unweighted shortest-path distances from one source via BFS.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Adjacency list from ``build_undirected_adjacency``.
    source : int
        Source node index.

    Returns
    -------
    np.ndarray
        Integer distances with shape ``[N]``. Unreachable nodes have value ``-1``.
    """
    num_nodes = len(adjacency)
    distances = np.full(num_nodes, _UNREACHED, dtype=np.int32)
    distances[source] = 0
    frontier: deque[int] = deque([source])

    while frontier:
        node = frontier.popleft()
        next_distance = int(distances[node]) + 1
        for neighbor, _ in adjacency[node]:
            if int(distances[neighbor]) != _UNREACHED:
                continue
            distances[neighbor] = next_distance
            frontier.append(neighbor)

    return distances


def _dijkstra_distances(
    adjacency: list[list[tuple[int, float]]],
    source: int,
) -> np.ndarray:
    """Compute weighted shortest-path distances from one source via Dijkstra.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Weighted adjacency list from ``build_undirected_adjacency``.
    source : int
        Source node index.

    Returns
    -------
    np.ndarray
        Float distances with shape ``[N]``. Unreachable nodes have value inf.
    """
    num_nodes = len(adjacency)
    distances = np.full(num_nodes, np.inf, dtype=np.float64)
    distances[source] = 0.0
    visited = np.zeros(num_nodes, dtype=np.bool_)

    import heapq

    frontier: list[tuple[float, int]] = [(0.0, source)]

    while frontier:
        distance, node = heapq.heappop(frontier)
        if visited[node]:
            continue
        visited[node] = True
        for neighbor, weight in adjacency[node]:
            candidate = distance + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                heapq.heappush(frontier, (candidate, neighbor))

    return distances


def _graph_distances(
    adjacency: list[list[tuple[int, float]]],
    source: int,
    weighted: bool,
) -> np.ndarray:
    """Compute graph distances from one source node.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Undirected adjacency list.
    source : int
        Shortest-path source node index.
    weighted : bool
        Whether to use Dijkstra instead of BFS.

    Returns
    -------
    np.ndarray
        Distances with shape ``[N]`` and ``-1`` for unreachable nodes.
    """
    if not weighted:
        return _bfs_distances(adjacency, source)

    distances = _dijkstra_distances(adjacency, source)
    return np.where(np.isinf(distances), float(_UNREACHED), distances)


def _is_connected(adjacency: list[list[tuple[int, float]]]) -> bool:
    """Report whether the undirected graph is connected.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Undirected adjacency list.

    Returns
    -------
    bool
        ``True`` when every node is reachable from node zero.
    """
    if len(adjacency) <= 1:
        return True
    return bool(np.all(_bfs_distances(adjacency, 0) >= 0))


def _schedule_bounds(distance_data: np.ndarray) -> tuple[float, float]:
    """Compute positive graph-distance bounds for the SGD schedule.

    Parameters
    ----------
    distance_data : np.ndarray
        Exact or approximate graph distances.

    Returns
    -------
    tuple[float, float]
        Minimum and maximum positive graph distances.
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
    """Evaluate the exact ``s_gd2`` exponential schedule from distance bounds.

    Parameters
    ----------
    step_index : int
        Zero-based schedule index.
    steps : int
        Total number of epochs.
    d_min : float
        Minimum positive graph distance.
    d_max : float
        Maximum positive graph distance.
    eps : float, default=0.01
        Final schedule scale factor.

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
    lambd = math.log(eta_max / eta_min) / float(steps - 1)
    return eta_max * math.exp(-lambd * float(step_index))


def _schedule_from_weights(
    steps: int,
    w_min: float,
    w_max: float,
    eps: float,
) -> np.ndarray:
    """Compute the exact ``s_gd2`` learning-rate schedule.

    Parameters
    ----------
    steps : int
        Number of epochs.
    w_min : float
        Minimum pair weight.
    w_max : float
        Maximum pair weight.
    eps : float
        Final schedule scale factor.

    Returns
    -------
    np.ndarray
        Schedule values with shape ``[steps]``.
    """
    if steps <= 0:
        return np.empty((0,), dtype=np.float64)

    safe_min = max(w_min, np.finfo(np.float64).tiny)
    safe_max = max(w_max, np.finfo(np.float64).tiny)
    eta_max = 1.0 / safe_min
    eta_min = eps / safe_max
    if steps == 1:
        return np.asarray([eta_max], dtype=np.float64)

    lambd = math.log(eta_max / eta_min) / float(steps - 1)
    return np.asarray(
        [eta_max * math.exp(-lambd * float(step_index)) for step_index in range(steps)],
        dtype=np.float64,
    )


def _resolve_sample_size(num_nodes: int, sample_size: Union[int, str]) -> int:
    """Resolve the effective large-graph sampling budget.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    sample_size : int or str
        User-provided sampling budget. ``"auto"`` uses a full epoch for small
        fallback graphs and a fixed-size sample otherwise.

    Returns
    -------
    int
        Effective per-epoch pair budget.

    Raises
    ------
    ValueError
        If ``sample_size`` is invalid.
    """
    if isinstance(sample_size, str):
        if sample_size != "auto":
            raise ValueError("sample_size must be a positive integer or 'auto'.")
        if num_nodes <= _AUTO_FULL_EPOCH_THRESHOLD:
            return num_nodes
        return _AUTO_SAMPLE_THRESHOLD

    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    return sample_size


def _choose_pivots(num_nodes: int, max_pivots: int, rng: np.random.RandomState) -> np.ndarray:
    """Choose pivot nodes for approximate shortest-path queries.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    max_pivots : int
        Maximum number of pivots.
    rng : np.random.RandomState
        Random state used by the large-graph fallback.

    Returns
    -------
    np.ndarray
        Pivot indices with shape ``[P]``.
    """
    if num_nodes <= max_pivots:
        return np.arange(num_nodes, dtype=np.int32)
    return rng.choice(num_nodes, size=max_pivots, replace=False).astype(np.int32, copy=False)


def _disconnected_fallback_layout(
    num_nodes: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a deterministic random layout for disconnected inputs.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    seed : int
        Seed used to stabilize the fallback coordinates.
    device : torch.device
        Device for the returned tensor.

    Returns
    -------
    torch.Tensor
        Fallback positions with shape ``[N, 2]``.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    positions = (
        torch.randn((num_nodes, 2), generator=generator, dtype=torch.float32)
        * _DISCONNECTED_FALLBACK_SCALE
    )
    return positions.to(device=device)


def _compute_pivot_distances(
    adjacency: list[list[tuple[int, float]]],
    pivots: np.ndarray,
    weighted: bool,
) -> np.ndarray:
    """Run shortest-path queries from each pivot node.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Undirected adjacency list.
    pivots : np.ndarray
        Pivot indices with shape ``[P]``.
    weighted : bool
        Whether to use Dijkstra instead of BFS.

    Returns
    -------
    np.ndarray
        Pivot-to-node distances with shape ``[P, N]``.
    """
    num_nodes = len(adjacency)
    if pivots.size == 0:
        return np.empty((0, num_nodes), dtype=np.float32)

    distances = np.empty((int(pivots.size), num_nodes), dtype=np.float32)
    for pivot_index, pivot in enumerate(pivots.tolist()):
        distances[pivot_index] = _graph_distances(
            adjacency,
            int(pivot),
            weighted=weighted,
        ).astype(np.float32)
    return distances


def _approx_distance(source_index: int, target_index: int, pivot_dist: np.ndarray) -> float:
    """Approximate one graph distance from pivot BFS data.

    Parameters
    ----------
    source_index : int
        Source node index.
    target_index : int
        Target node index.
    pivot_dist : np.ndarray
        Pivot-to-node distances with shape ``[P, N]``.

    Returns
    -------
    float
        Approximate shortest-path distance.
    """
    if pivot_dist.size == 0:
        return 1.0

    pivot_i = pivot_dist[:, source_index]
    pivot_j = pivot_dist[:, target_index]
    lower = np.abs(pivot_i - pivot_j)
    upper = pivot_i + pivot_j
    best_lower = float(np.max(lower))
    best_upper = float(np.min(upper))
    if math.isfinite(best_upper):
        return max((best_lower + best_upper) * 0.5, 1.0)
    return max(best_lower, 1.0)


def _pair_count(num_nodes: int) -> int:
    """Compute the number of unordered node pairs.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.

    Returns
    -------
    int
        Count of pairs in the upper triangle.
    """
    return (num_nodes * (num_nodes - 1)) // 2


def _build_exact_terms(
    adjacency: list[list[tuple[int, float]]],
    weighted: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the exact upper-triangle ``s_gd2`` term arrays.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Undirected adjacency list.
    weighted : bool, default=False
        Whether to use Dijkstra instead of BFS.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(sources, targets, distances, weights)`` with matching shape ``[M]``.
    """
    num_nodes = len(adjacency)
    num_terms = _pair_count(num_nodes)
    sources = np.empty((num_terms,), dtype=np.int32)
    targets = np.empty((num_terms,), dtype=np.int32)
    distances = np.empty((num_terms,), dtype=np.float32)
    weights = np.empty((num_terms,), dtype=np.float32)

    write_index = 0
    for source_index in range(num_nodes - 1):
        source_distances = _graph_distances(adjacency, source_index, weighted=weighted)
        for target_index in range(source_index + 1, num_nodes):
            graph_distance = float(source_distances[target_index])
            if graph_distance <= 0:
                raise ValueError("Stress-SGD requires a connected graph.")

            weight = 1.0 / (graph_distance * graph_distance)
            sources[write_index] = source_index
            targets[write_index] = target_index
            distances[write_index] = graph_distance
            weights[write_index] = weight
            write_index += 1

    return sources, targets, distances, weights


def _sample_pairs(
    num_nodes: int,
    sample_size: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a large-graph pair batch for one SGD epoch.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    sample_size : int
        Per-epoch pair budget.
    rng : np.random.RandomState
        Random state used by the large-graph fallback.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Source and target indices with matching shape ``[S]``.
    """
    if num_nodes <= 1:
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty

    sources = rng.randint(0, num_nodes, size=sample_size).astype(np.int64, copy=False)
    targets = rng.randint(0, num_nodes, size=sample_size).astype(np.int64, copy=False)
    valid_pairs = sources != targets
    return sources[valid_pairs], targets[valid_pairs]


def _trace_snapshot(
    traces: list[torch.Tensor],
    positions: np.ndarray,
    device: torch.device,
) -> None:
    """Append a detached position snapshot to the trace list.

    Parameters
    ----------
    traces : list[torch.Tensor]
        Mutable trace buffer.
    positions : np.ndarray
        Current positions with shape ``[N, 2]``.
    device : torch.device
        Target torch device.

    Returns
    -------
    None
        The trace list is mutated in place.
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
    """Apply one sequential Stress-SGD pair update in place.

    Parameters
    ----------
    positions : np.ndarray
        Current positions with shape ``[N, 2]``.
    source_index : int
        Source node index.
    target_index : int
        Target node index.
    target_distance : float
        Desired graph distance for the pair.
    weight : float
        Pair weight ``1 / d^2``.
    eta : float
        Current step size.

    Returns
    -------
    None
        Positions are updated in place.
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


def _run_exact_schedule(
    num_nodes: int,
    sources: np.ndarray,
    targets: np.ndarray,
    distances: np.ndarray,
    weights: np.ndarray,
    steps: int,
    eps: float,
    trace_every: int,
    device: torch.device,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, list[torch.Tensor]]:
    """Run the exact ``s_gd2`` schedule with sequential numpy updates.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    sources : np.ndarray
        Term source indices with shape ``[M]``.
    targets : np.ndarray
        Term target indices with shape ``[M]``.
    distances : np.ndarray
        Exact graph distances with shape ``[M]``.
    weights : np.ndarray
        Exact graph weights with shape ``[M]``.
    steps : int
        Number of epochs.
    eps : float
        Final schedule scale factor.
    trace_every : int
        Trace interval in epochs.
    device : torch.device
        Target torch device for trace snapshots.
    rng : np.random.RandomState
        Random state used by the translated C++ kernel.

    Returns
    -------
    tuple[np.ndarray, list[torch.Tensor]]
        Final positions and optional trace snapshots.
    """
    positions = np.random.rand(num_nodes, 2)
    traces: list[torch.Tensor] = []
    schedule = _schedule_from_weights(
        steps=steps,
        w_min=float(weights.min()),
        w_max=float(weights.max()),
        eps=eps,
    )
    order = np.arange(sources.shape[0], dtype=np.int64)

    if trace_every > 0 and steps == 0:
        _trace_snapshot(traces, positions, device)

    for step_index, eta in enumerate(schedule):
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

        if trace_every > 0 and (step_index + 1) % trace_every == 0:
            _trace_snapshot(traces, positions, device)

    return positions, traces


def _run_approximate_schedule(
    num_nodes: int,
    pivot_dist: np.ndarray,
    steps: int,
    eps: float,
    sample_size: int,
    trace_every: int,
    device: torch.device,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, list[torch.Tensor]]:
    """Run the legacy pivot-based approximation for large graphs.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    pivot_dist : np.ndarray
        Pivot-to-node distances with shape ``[P, N]``.
    steps : int
        Number of epochs.
    eps : float
        Final schedule scale factor.
    sample_size : int
        Per-epoch pair budget when sampling is enabled.
    trace_every : int
        Trace interval in epochs.
    device : torch.device
        Target torch device for trace snapshots.
    rng : np.random.RandomState
        Random state used by the large-graph fallback.

    Returns
    -------
    tuple[np.ndarray, list[torch.Tensor]]
        Final positions and optional trace snapshots.
    """
    positions = np.random.rand(num_nodes, 2)
    traces: list[torch.Tensor] = []
    d_min, d_max = _schedule_bounds(pivot_dist)
    use_full_epoch = sample_size >= num_nodes and num_nodes <= _AUTO_FULL_EPOCH_THRESHOLD

    full_sources: Optional[np.ndarray] = None
    full_targets: Optional[np.ndarray] = None
    full_order: Optional[np.ndarray] = None
    if use_full_epoch:
        full_sources, full_targets = np.triu_indices(num_nodes, k=1)
        full_order = np.arange(full_sources.shape[0], dtype=np.int64)

    if trace_every > 0 and steps == 0:
        _trace_snapshot(traces, positions, device)

    for step_index in range(steps):
        eta = _learning_rate(step_index, steps, d_min, d_max, eps=eps)
        if use_full_epoch:
            assert full_sources is not None
            assert full_targets is not None
            assert full_order is not None
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
            sampled_sources, sampled_targets = _sample_pairs(num_nodes, sample_size, rng)
            sampled_pairs = zip(sampled_sources.tolist(), sampled_targets.tolist())
            for source_index, target_index in sampled_pairs:
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

        if trace_every > 0 and (step_index + 1) % trace_every == 0:
            _trace_snapshot(traces, positions, device)

    return positions, traces


from dagua.layout.ops.base import Op, Pipeline  # noqa: E402
from dagua.layout.ops.state import (  # noqa: E402
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)
from dagua.layout.ops.taxonomy import OpCategory  # noqa: E402

# ---------------------------------------------------------------------------
# Pipeline-local ops
# ---------------------------------------------------------------------------


class _InitStressSGDPositions(Op):
    """Seed numpy global RNG and initialize positions exactly like classic."""

    name: ClassVar[str] = "stress_sgd_init_positions"
    category: ClassVar[OpCategory] = OpCategory.INIT
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Seed numpy global RNG and build undirected adjacency.

        This op seeds ``np.random.seed(seed)`` exactly as the classic
        implementation does, so all downstream ``np.random`` calls
        produce the same sequence.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure. Unused.

        Returns
        -------
        SolveState
            State with adjacency and connectivity stored in ``extras``.
        """
        del ctx

        device = problem.edge_index.device
        num_nodes = problem.num_nodes
        edge_weights = problem.edge_weights

        # Store device for later ops
        state.extras["stress_sgd_device"] = device
        state.extras["stress_sgd_num_nodes"] = num_nodes

        # Handle trivial graphs
        if num_nodes <= 1:
            state.pos = torch.zeros((num_nodes, 2), dtype=torch.float32, device=device)
            state.converged = True
            return state

        # Build adjacency
        adjacency = _build_undirected_adjacency(
            problem.edge_index, num_nodes, edge_weights=edge_weights
        )
        state.extras["stress_sgd_adjacency"] = adjacency

        # Connectivity check
        if not _is_connected(adjacency):
            state.pos = _disconnected_fallback_layout(
                num_nodes=num_nodes,
                seed=problem.seed,
                device=device,
            )
            state.converged = True
            return state

        # Seed global numpy RNG -- this MUST happen before np.random.rand()
        # in the schedule runner, matching the classic implementation exactly.
        np.random.seed(problem.seed)
        state.extras["stress_sgd_rng"] = np.random  # global RNG handle
        state.extras["stress_sgd_weighted"] = edge_weights is not None

        return state


class _PrepareExactTerms(Op):
    """Build exact upper-triangle term arrays for small graphs."""

    name: ClassVar[str] = "stress_sgd_prepare_exact_terms"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def __init__(self, max_exact_nodes: int = _DEFAULT_MAX_EXACT_NODES) -> None:
        """Store the exact-node threshold.

        Parameters
        ----------
        max_exact_nodes : int
            Maximum node count for the exact code path.
        """
        self.max_exact_nodes = max_exact_nodes

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build exact terms or prepare approximate pivot data.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure. Unused.

        Returns
        -------
        SolveState
            State with term arrays or pivot data in ``extras``.
        """
        del ctx

        # Skip if already converged (trivial or disconnected)
        if state.converged:
            return state

        num_nodes = state.extras["stress_sgd_num_nodes"]
        adjacency = state.extras["stress_sgd_adjacency"]
        weighted = state.extras["stress_sgd_weighted"]
        rng = state.extras["stress_sgd_rng"]

        if num_nodes <= self.max_exact_nodes:
            state.extras["stress_sgd_mode"] = "exact"
            sources, targets, distances, weights = _build_exact_terms(adjacency, weighted=weighted)
            state.extras["stress_sgd_sources"] = sources
            state.extras["stress_sgd_targets"] = targets
            state.extras["stress_sgd_distances"] = distances
            state.extras["stress_sgd_weights"] = weights
        else:
            state.extras["stress_sgd_mode"] = "approximate"
            pivots = _choose_pivots(num_nodes, min(num_nodes, _MAX_PIVOTS), rng)
            pivot_dist = _compute_pivot_distances(adjacency, pivots, weighted=weighted)
            state.extras["stress_sgd_pivot_dist"] = pivot_dist

        return state


class _RunExactSchedule(Op):
    """Run the full exact s_gd2 schedule as a single op.

    This op encapsulates the entire epoch loop for the exact code path,
    matching the classic ``_run_exact_schedule`` function call-for-call.
    The sequential Gauss-Seidel updates with shuffled pair order cannot
    be split across separate ops without breaking the RNG sequence.
    """

    name: ClassVar[str] = "stress_sgd_run_exact_schedule"
    category: ClassVar[OpCategory] = OpCategory.OPTIMIZE
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("pos", "extras")

    def __init__(
        self,
        steps: int = 30,
        eps: float = _DEFAULT_EPS,
        trace_every: int = 0,
    ) -> None:
        """Store schedule parameters.

        Parameters
        ----------
        steps : int
            Number of SGD epochs.
        eps : float
            Final schedule scale factor.
        trace_every : int
            Trace snapshot interval.
        """
        self.steps = steps
        self.eps = eps
        self.trace_every = trace_every

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Execute the exact schedule, producing final positions.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure. Unused.

        Returns
        -------
        SolveState
            State with final positions from the exact schedule.
        """
        del ctx

        if state.converged:
            return state
        if state.extras.get("stress_sgd_mode") != "exact":
            return state

        num_nodes = state.extras["stress_sgd_num_nodes"]
        device = state.extras["stress_sgd_device"]
        rng = state.extras["stress_sgd_rng"]

        positions_np, traces = _run_exact_schedule(
            num_nodes=num_nodes,
            sources=state.extras["stress_sgd_sources"],
            targets=state.extras["stress_sgd_targets"],
            distances=state.extras["stress_sgd_distances"],
            weights=state.extras["stress_sgd_weights"],
            steps=self.steps,
            eps=self.eps,
            trace_every=self.trace_every,
            device=device,
            rng=rng,
        )

        state.pos = torch.from_numpy(positions_np.astype(np.float32, copy=False)).to(device=device)
        if traces:
            state.extras["stress_sgd_traces"] = traces
        state.converged = True
        return state


class _RunApproximateSchedule(Op):
    """Run the pivot-based approximate schedule as a single op.

    This op encapsulates the entire epoch loop for the large-graph
    approximate code path, matching ``_run_approximate_schedule`` exactly.
    """

    name: ClassVar[str] = "stress_sgd_run_approximate_schedule"
    category: ClassVar[OpCategory] = OpCategory.OPTIMIZE
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("pos", "extras")

    def __init__(
        self,
        steps: int = 30,
        eps: float = _DEFAULT_EPS,
        sample_size: Union[int, str] = "auto",
        trace_every: int = 0,
    ) -> None:
        """Store schedule parameters.

        Parameters
        ----------
        steps : int
            Number of SGD epochs.
        eps : float
            Final schedule scale factor.
        sample_size : int or str
            Large-graph sampling budget.
        trace_every : int
            Trace snapshot interval.
        """
        self.steps = steps
        self.eps = eps
        self.sample_size = sample_size
        self.trace_every = trace_every

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Execute the approximate schedule, producing final positions.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure. Unused.

        Returns
        -------
        SolveState
            State with final positions from the approximate schedule.
        """
        del ctx

        if state.converged:
            return state
        if state.extras.get("stress_sgd_mode") != "approximate":
            return state

        num_nodes = state.extras["stress_sgd_num_nodes"]
        device = state.extras["stress_sgd_device"]
        rng = state.extras["stress_sgd_rng"]
        pivot_dist = state.extras["stress_sgd_pivot_dist"]

        effective_sample_size = _resolve_sample_size(num_nodes, self.sample_size)

        positions_np, traces = _run_approximate_schedule(
            num_nodes=num_nodes,
            pivot_dist=pivot_dist,
            steps=self.steps,
            eps=self.eps,
            sample_size=effective_sample_size,
            trace_every=self.trace_every,
            device=device,
            rng=rng,
        )

        state.pos = torch.from_numpy(positions_np.astype(np.float32, copy=False)).to(device=device)
        if traces:
            state.extras["stress_sgd_traces"] = traces
        state.converged = True
        return state


# ---------------------------------------------------------------------------
# Pipeline builder and drop-in adapter
# ---------------------------------------------------------------------------


def build_stress_sgd_pipeline(
    steps: int = 30,
    eps: float = _DEFAULT_EPS,
    max_exact_nodes: int = _DEFAULT_MAX_EXACT_NODES,
    sample_size: Union[int, str] = "auto",
    trace_every: int = 0,
) -> Pipeline:
    """Build a stress-SGD pipeline that is bit-identical to classic ``layout_stress_sgd``.

    Parameters
    ----------
    steps : int, default=30
        Number of SGD epochs.
    eps : float, default=0.01
        Final schedule scale factor.
    max_exact_nodes : int, default=10000
        Node threshold for exact vs approximate code path.
    sample_size : int or str, default="auto"
        Large-graph sampling budget for the approximate path.
    trace_every : int, default=0
        If positive, record position snapshots every this many epochs.

    Returns
    -------
    Pipeline
        Pipeline that reproduces classic stress-SGD's initialization,
        distance computation, schedule execution, and output conversion.

    Raises
    ------
    ValueError
        If ``steps`` is negative.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative.")

    return Pipeline(
        [
            _InitStressSGDPositions(),
            _PrepareExactTerms(max_exact_nodes=max_exact_nodes),
            _RunExactSchedule(
                steps=steps,
                eps=eps,
                trace_every=trace_every,
            ),
            _RunApproximateSchedule(
                steps=steps,
                eps=eps,
                sample_size=sample_size,
                trace_every=trace_every,
            ),
        ],
        name="stress_sgd_pipeline",
    )


def layout_stress_sgd_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 30,
    seed: int = 42,
    sample_size: Union[int, str] = "auto",
    trace_every: int = 0,
    eps: float = _DEFAULT_EPS,
    max_exact_nodes: int = _DEFAULT_MAX_EXACT_NODES,
    edge_weights: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, "tuple[torch.Tensor, list[torch.Tensor]]"]:
    """Run the stress-SGD pipeline as a drop-in replacement for classic ``layout_stress_sgd``.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Unused, accepted for interface compatibility.
    steps : int, default=30
        Number of SGD epochs.
    seed : int, default=42
        Random seed for numpy initialization.
    sample_size : int or str, default="auto"
        Large-graph sampling budget.
    trace_every : int, default=0
        If positive, record position snapshots.
    eps : float, default=0.01
        Final schedule scale factor.
    max_exact_nodes : int, default=10000
        Node threshold for exact vs approximate code path.
    edge_weights : torch.Tensor, optional
        Optional per-edge weights with shape ``[E]``.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, list[torch.Tensor]]
        Final positions, or ``(positions, traces)`` if tracing is enabled.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``steps``, ``trace_every``, ``eps``, or
        ``max_exact_nodes`` are invalid.
    RuntimeError
        If the pipeline fails to produce positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if trace_every < 0:
        raise ValueError("trace_every must be non-negative")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if max_exact_nodes < 0:
        raise ValueError("max_exact_nodes must be non-negative")

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_stress_sgd_pipeline(
        steps=steps,
        eps=eps,
        max_exact_nodes=max_exact_nodes,
        sample_size=sample_size,
        trace_every=trace_every,
    ).apply(problem, state, ctx)

    if final_state.pos is None:
        raise RuntimeError("Stress-SGD pipeline did not produce final positions.")

    traces = final_state.extras.get("stress_sgd_traces", [])
    if trace_every > 0:
        return final_state.pos, traces
    return final_state.pos


__all__ = ["build_stress_sgd_pipeline", "layout_stress_sgd_pipeline"]
