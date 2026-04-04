"""Stress-SGD registered operations.

This module contains the algorithmic primitives needed by the composable
pipeline implementation. The operations are intentionally registered so they can
be composed and reused by any stress-SGD variant that requires the same
building blocks.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar, Optional, Tuple, Union

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


def _resolve_sample_size(num_nodes: int, sample_size: Union[int, str]) -> int:
    """Resolve the large-graph sample budget.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    sample_size : int | str
        ``"auto"`` or a positive sample count.

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
        if num_nodes <= _AUTO_FULL_EPOCH_THRESHOLD:
            return num_nodes
        return _AUTO_SAMPLE_THRESHOLD

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build full upper-triangle stress-SGD term data.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Undirected adjacency list.
    weighted : bool
        Whether to use weighted shortest-path distances.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(sources, targets, distances, weights)`` with matching shapes.
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

            sources[write_index] = source_index
            targets[write_index] = target_index
            distances[write_index] = graph_distance
            weights[write_index] = 1.0 / (graph_distance * graph_distance)
            write_index += 1

    return sources, targets, distances, weights


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

    Returns
    -------
    torch.Tensor
        Fallback coordinates with shape ``[N, 2]``.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    positions = (
        torch.randn((num_nodes, 2), generator=generator, dtype=torch.float32)
        * _DISCONNECTED_FALLBACK_SCALE
    )
    return positions.to(device=device)


@register_op
class InitializeStressSGDState(Op):
    """Prepare Stress-SGD state and deterministic RNG setup.

    This op resolves graph connectivity and connected-component fallback behavior,
    then captures all problem flags needed by the schedule builders.
    """

    name: ClassVar[str] = "stress_sgd_initialize_state"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = ("extras", "pos")

    def __init__(self, trace_every: int = 0) -> None:
        """Create an initializer with trace cadence metadata.

        Parameters
        ----------
        trace_every : int
            Trace interval passed through from the public adapter.
        """
        self.trace_every = trace_every

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
            if self.trace_every > 0:
                state.extras[_STRESS_SGD_TRACE_KEY] = []
            return state

        if not isinstance(adjacency, list):
            raise ValueError("InitializeStressSGDState requires state.adjacency to be a list.")
        connected = _is_connected(adjacency)
        state.extras[_STRESS_SGD_CONNECTED_KEY] = bool(connected)
        if not connected:
            fallback_position = _disconnected_fallback_layout(
                num_nodes=num_nodes,
                seed=problem.seed,
                device=device,
            )
            state.pos = fallback_position
            state.converged = True
            if self.trace_every > 0:
                state.extras[_STRESS_SGD_TRACE_KEY] = [fallback_position.clone()]
            else:
                state.extras.pop(_STRESS_SGD_TRACE_KEY, None)
            return state

        np.random.seed(problem.seed)
        state.extras[_STRESS_SGD_RNG_KEY] = np.random
        state.extras.pop(_STRESS_SGD_CONNECTED_KEY, None)
        return state


@register_op
class PrepareStressSGDTerms(Op):
    """Prepare exact or approximate pair data for the Stress-SGD kernels."""

    name: ClassVar[str] = "stress_sgd_prepare_terms"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("adjacency", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("extras",)
    requires: ClassVar[Tuple[str, ...]] = ("adjacency",)

    def __init__(self, max_exact_nodes: int = _DEFAULT_MAX_EXACT_NODES) -> None:
        """Create a configured term builder.

        Parameters
        ----------
        max_exact_nodes : int
            Maximum number of nodes for full all-pairs tensor construction.
        """
        self.max_exact_nodes = max_exact_nodes

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

        if num_nodes <= self.max_exact_nodes:
            state.extras[_STRESS_SGD_EXACT_KEY] = True
            sources, targets, distances, weights = _build_exact_terms(
                adjacency=adjacency,
                weighted=weighted,
            )
            state.extras["stress_sgd_sources"] = sources
            state.extras["stress_sgd_targets"] = targets
            state.extras["stress_sgd_distances"] = distances
            state.extras["stress_sgd_weights"] = weights
            return state

        state.extras[_STRESS_SGD_EXACT_KEY] = False
        pivots = _choose_pivots(
            num_nodes=num_nodes,
            max_pivots=min(num_nodes, _MAX_PIVOTS),
            rng=rng,
        )
        pivot_dist = _compute_pivot_distances(
            adjacency=adjacency,
            pivots=pivots,
            weighted=weighted,
        )
        state.distance_matrix = torch.from_numpy(pivot_dist)
        return state


@register_op
class RunStressSGDExactSchedule(Op):
    """Run the full exact upper-triangle Stress-SGD schedule."""

    name: ClassVar[str] = "stress_sgd_run_exact_schedule"
    category: ClassVar[OpCategory] = OpCategory.OPTIMIZE
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("pos", "extras")
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
        self.steps = steps
        self.eps = eps
        self.trace_every = trace_every

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

        positions = np.random.rand(num_nodes, 2)
        traces: list[torch.Tensor] = []
        schedule = _schedule_from_weights(
            steps=self.steps,
            w_min=float(weights.min()),
            w_max=float(weights.max()),
            eps=self.eps,
        )
        order = np.arange(sources.shape[0], dtype=np.int64)

        if self.trace_every > 0 and self.steps == 0:
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

            if self.trace_every > 0 and (step_index + 1) % self.trace_every == 0:
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
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("pos", "extras")
    requires: ClassVar[Tuple[str, ...]] = ("extras",)

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
        if not isinstance(pivot_distances, torch.Tensor):
            raise ValueError(
                "RunStressSGDApproximateSchedule requires pivot distances in state.distance_matrix."
            )
        pivot_dist = pivot_distances.to(dtype=torch.float64, device="cpu").numpy()
        effective_sample_size = _resolve_sample_size(num_nodes, self.sample_size)

        positions = np.random.rand(num_nodes, 2)
        traces: list[torch.Tensor] = []
        d_min, d_max = _schedule_bounds(pivot_dist)
        use_full_epoch = (
            effective_sample_size >= num_nodes and num_nodes <= _AUTO_FULL_EPOCH_THRESHOLD
        )

        full_sources: Optional[np.ndarray] = None
        full_targets: Optional[np.ndarray] = None
        full_order: Optional[np.ndarray] = None
        if use_full_epoch:
            full_sources, full_targets = np.triu_indices(num_nodes, k=1)
            full_order = np.arange(full_sources.shape[0], dtype=np.int64)

        if self.trace_every > 0 and self.steps == 0:
            _trace_snapshot(traces, positions, device)

        for step_index in range(self.steps):
            eta = _learning_rate(step_index, self.steps, d_min, d_max, eps=self.eps)
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

            if self.trace_every > 0 and (step_index + 1) % self.trace_every == 0:
                _trace_snapshot(traces, positions, device)

        state.pos = torch.from_numpy(positions.astype(np.float32, copy=False)).to(device=device)
        state.extras[_STRESS_SGD_TRACE_KEY] = traces
        state.converged = True
        return state
