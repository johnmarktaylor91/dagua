"""GEM (Graph Embedder) expressed as a composable ops pipeline.

The GEM algorithm uses a Gauss-Seidel per-node update schedule with per-node
temperatures, oscillation/rotation detection, and weighted gravity.  For small
graphs (N <= 5000) the sequential update order is essential for correctness --
each node sees already-updated positions from earlier nodes in the same round.
For larger graphs a batched fallback with vectorized forces is used.

This pipeline delegates to the classic helpers so that output is bit-identical
to ``layout_gem``.
"""

from __future__ import annotations

from typing import ClassVar, Optional, Tuple

import torch

from dagua.layout.classic.gem import (
    _DESIRED_LENGTH,
    _FULL_REPULSION_LIMIT,
    _INITIAL_TEMPERATURE,
    _MIN_DISTANCE,
    _MINIMAL_TEMPERATURE,
    _NUMBER_OF_ROUNDS,
    _SEQUENTIAL_NODE_LIMIT,
    _attractive_force_batch,
    _build_undirected_adjacency,
    _compute_impulse_sequential,
    _degree_weights,
    _gravity_force,
    _ideal_distance,
    _initialize_positions,
    _layout_device,
    _layout_extent,
    _node_diagonals,
    _normalize_positions,
    _repulsive_force_batch,
    _repulsive_force_sampled,
    _update_node_sequential,
    _update_temperatures,
)
from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory


class _InitializeGEMPositions(Op):
    """Initialize positions exactly like classic GEM."""

    name: ClassVar[str] = "gem_initialize_positions"
    category: ClassVar[OpCategory] = OpCategory.INIT
    writes: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Seed ``state.pos`` from torch randn, matching classic GEM init.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs containing ``num_nodes`` and ``seed``.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with ``state.pos`` populated as a CPU ``float32`` tensor,
            or early-exit positions for trivial graphs.
        """
        del ctx

        device = _layout_device(problem.edge_index, problem.node_sizes)
        if problem.num_nodes == 0:
            state.pos = torch.empty((0, 2), dtype=torch.float32, device=device)
            state.converged = True
            return state
        if problem.num_nodes == 1:
            state.pos = torch.zeros((1, 2), dtype=torch.float32, device=device)
            state.converged = True
            return state

        state.pos = _initialize_positions(problem.num_nodes, device, problem.seed)
        return state


class _PrepareGEMState(Op):
    """Populate GEM-specific cached state required by the solve step."""

    name: ClassVar[str] = "gem_prepare_state"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build GEM-specific auxiliary data structures.

        Stores degree weights, node desired lengths, adjacency lists,
        initial temperatures, extent, and capped iteration count into
        ``state.extras`` for downstream ops.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state receiving GEM extras.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with GEM auxiliary data stored in ``extras``.
        """
        del ctx

        if state.converged:
            return state

        device = _layout_device(problem.edge_index, problem.node_sizes)
        extent = _layout_extent(problem.num_nodes, problem.node_sizes)
        capped_iters = min(state.total_steps, _NUMBER_OF_ROUNDS)

        state.extras["gem_extent"] = extent
        state.extras["gem_capped_iters"] = capped_iters
        state.extras["gem_device"] = device
        state.extras["gem_is_sequential"] = problem.num_nodes <= _SEQUENTIAL_NODE_LIMIT
        return state


class _GEMSequentialSolve(Op):
    """Run the OGDF-like sequential Gauss-Seidel GEM loop.

    This op wraps the full sequential iteration because the Gauss-Seidel
    update order is not decomposable into separate force/apply ops -- each
    node sees already-updated positions of previously processed nodes.
    """

    name: ClassVar[str] = "gem_sequential_solve"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("pos", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Execute the sequential GEM loop, identical to classic.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with initialized positions.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with positions updated by the sequential GEM solver.
        """
        del ctx

        if state.converged or not state.extras.get("gem_is_sequential", False):
            return state

        num_nodes = problem.num_nodes
        capped_iters = state.extras["gem_capped_iters"]

        positions = state.pos.to(device="cpu", dtype=torch.float64)
        degree_wts = _degree_weights(
            problem.edge_index,
            num_nodes,
            torch.device("cpu"),
        ).to(dtype=torch.float64)
        node_desired_lengths = (
            _node_diagonals(num_nodes, problem.node_sizes).to(dtype=torch.float64) + _DESIRED_LENGTH
        )
        adjacency = _build_undirected_adjacency(
            problem.edge_index, num_nodes, edge_weights=problem.edge_weights
        )
        local_temperatures = torch.full((num_nodes,), _INITIAL_TEMPERATURE, dtype=torch.float64)
        previous_impulse = torch.zeros((num_nodes, 2), dtype=torch.float64)
        skew_gauge = torch.zeros((num_nodes,), dtype=torch.float64)
        barycenter = (positions * degree_wts.unsqueeze(1)).sum(dim=0)
        global_temperature = _INITIAL_TEMPERATURE

        generator = torch.Generator(device="cpu")
        generator.manual_seed(problem.seed)
        permutation: list[int] = []
        rounds_remaining = capped_iters

        while global_temperature > _MINIMAL_TEMPERATURE and rounds_remaining > 0:
            if not permutation:
                permutation = torch.randperm(num_nodes, generator=generator).tolist()
            node_index = permutation.pop()
            impulse = _compute_impulse_sequential(
                node_index=node_index,
                positions=positions,
                barycenter=barycenter,
                adjacency=adjacency,
                node_desired_lengths=node_desired_lengths,
                degree_weights=degree_wts,
            )
            global_temperature = _update_node_sequential(
                node_index=node_index,
                positions=positions,
                impulse=impulse,
                previous_impulse=previous_impulse,
                local_temperatures=local_temperatures,
                skew_gauge=skew_gauge,
                degree_weights=degree_wts,
                barycenter=barycenter,
                global_temperature=global_temperature,
            )
            rounds_remaining -= 1

        state.pos = positions.to(dtype=torch.float32)
        return state


class _GEMBatchedSolve(Op):
    """Run the vectorized batched GEM loop for large graphs.

    For graphs with N > 5000 the batched fallback uses vectorized
    repulsion, attraction, and gravity forces with per-node temperature
    adaptation identical to the classic batched path.
    """

    name: ClassVar[str] = "gem_batched_solve"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("pos", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Execute the batched GEM loop, identical to classic.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with initialized positions.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with positions updated by the batched GEM solver.
        """
        del ctx

        if state.converged or state.extras.get("gem_is_sequential", True):
            return state

        num_nodes = problem.num_nodes
        device = state.extras["gem_device"]
        extent = state.extras["gem_extent"]
        capped_iters = state.extras["gem_capped_iters"]

        positions = state.pos
        degree_wts = _degree_weights(problem.edge_index, num_nodes, device)
        node_desired_lengths = (
            _node_diagonals(num_nodes, problem.node_sizes) + _DESIRED_LENGTH
        ).to(device=device)
        temperatures = torch.full(
            (num_nodes,),
            fill_value=_INITIAL_TEMPERATURE,
            dtype=torch.float32,
            device=device,
        )
        previous_impulse = torch.zeros_like(positions)
        skew_gauge = torch.zeros((num_nodes,), dtype=torch.float32, device=device)
        sampled_ideal_distance = _ideal_distance(num_nodes, extent)

        for step in range(capped_iters):
            if float(temperatures.mean().item()) < _MINIMAL_TEMPERATURE:
                break

            if num_nodes > _FULL_REPULSION_LIMIT:
                repulsive = _repulsive_force_sampled(
                    positions, problem.seed, step, sampled_ideal_distance
                )
            else:
                repulsive = _repulsive_force_batch(positions, node_desired_lengths)
            attractive = _attractive_force_batch(
                positions=positions,
                edge_index=problem.edge_index,
                node_desired_lengths=node_desired_lengths,
                degree_weights=degree_wts,
                edge_weights=problem.edge_weights,
            )
            gravity = _gravity_force(positions, degree_wts)
            impulse = repulsive + attractive + gravity
            norm = torch.linalg.norm(impulse, dim=1, keepdim=True)
            safe_norm = norm.clamp(min=_MIN_DISTANCE)
            movement = torch.where(
                norm > 0,
                impulse * (temperatures.unsqueeze(1) / safe_norm),
                torch.zeros_like(impulse),
            )

            temperatures, skew_gauge = _update_temperatures(
                temperatures=temperatures,
                current_impulse=movement,
                previous_impulse=previous_impulse,
                skew_gauge=skew_gauge,
                initial_temperature=_INITIAL_TEMPERATURE,
            )

            positions = positions + movement
            previous_impulse = movement

        state.pos = positions
        return state


class _FinalizeGEMPositions(Op):
    """Apply classic GEM's final normalize, rescale, and cast."""

    name: ClassVar[str] = "gem_finalize_positions"
    category: ClassVar[OpCategory] = OpCategory.POSTPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("pos", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Center, normalize, scale, and cast positions like classic GEM.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state containing final positions.
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
            raise ValueError("_FinalizeGEMPositions requires state.pos to be set.")

        if state.converged:
            return state

        device = state.extras["gem_device"]
        extent = state.extras["gem_extent"]

        state.pos = _normalize_positions(state.pos, extent).to(
            dtype=torch.float32,
            device=device,
        )
        return state


def build_gem_pipeline(max_iters: int = 500) -> Pipeline:
    """Build a GEM pipeline that is bit-identical to classic ``layout_gem``.

    Parameters
    ----------
    max_iters : int, default=500
        Maximum number of OGDF node updates.

    Returns
    -------
    Pipeline
        Pipeline that reproduces classic GEM's initialization, sequential
        or batched solve, and postprocessing.

    Raises
    ------
    ValueError
        If ``max_iters`` is negative.
    """
    if max_iters < 0:
        raise ValueError("max_iters must be non-negative.")

    from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig

    return Pipeline(
        [
            FixedSteps(FixedStepsConfig(n=max_iters)),
            _InitializeGEMPositions(),
            _PrepareGEMState(),
            _GEMSequentialSolve(),
            _GEMBatchedSolve(),
            _FinalizeGEMPositions(),
        ],
        name="gem_pipeline",
    )


def layout_gem_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    max_iters: int = 500,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the GEM pipeline as a drop-in replacement for classic ``layout_gem``.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor used for scaling.
    max_iters : int, default=500
        Maximum number of OGDF node updates.
    seed : int, default=42
        Random seed for initialization and node permutations.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Final position tensor with the same dtype, device, and values as
        classic ``layout_gem``.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``max_iters``, or ``edge_weights`` are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if max_iters < 0:
        raise ValueError("max_iters must be non-negative.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
            )

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_gem_pipeline(max_iters=max_iters).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("GEM pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_gem_pipeline", "layout_gem_pipeline"]
