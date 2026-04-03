"""Stress-SGD expressed as a composable ops pipeline.

Produces BIT-IDENTICAL output to the classic ``layout_stress_sgd`` for both
the exact (small-graph) and approximate (large-graph pivot) code paths.  All
numeric work is delegated to the classic module's helper functions -- the ops
only orchestrate the call sequence.
"""

from __future__ import annotations

from typing import ClassVar, Optional, Tuple, Union

import numpy as np
import torch

from dagua.layout.classic.stress_sgd import (
    _DEFAULT_EPS,
    _DEFAULT_MAX_EXACT_NODES,
    _MAX_PIVOTS,
    _build_exact_terms,
    _build_undirected_adjacency,
    _choose_pivots,
    _compute_pivot_distances,
    _disconnected_fallback_layout,
    _is_connected,
    _resolve_sample_size,
)
from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory

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

        from dagua.layout.classic.stress_sgd import _run_exact_schedule

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

        from dagua.layout.classic.stress_sgd import _run_approximate_schedule

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
