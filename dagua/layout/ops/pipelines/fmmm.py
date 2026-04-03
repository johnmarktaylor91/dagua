"""FM^3 multilevel force-directed layout expressed as a composable ops pipeline.

FM^3 uses solar-system-style coarsening, a coarse FR initialization, Barnes-Hut
repulsion during refinement, and OGDF-style lambda interpolation for prolongation.
This pipeline delegates to the classic helpers so that output is bit-identical to
``layout_fmmm``.
"""

from __future__ import annotations

import random
from typing import ClassVar, Optional, Tuple

import torch

from dagua.layout.classic.fmmm import (
    _build_hierarchy,
    _layout_device,
    _layout_extent,
    _normalize_positions,
    _prolong_positions,
    _refine_level,
    _refine_level_with_edge_lengths,
)
from dagua.layout.classic.fr import layout_fr
from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory


class _InitializeFMMMState(Op):
    """Build the FM^3 multilevel hierarchy and store it in extras.

    Computes the solar-system coarsening hierarchy, extent, refinement area,
    and level budget, storing everything the downstream ops need.
    """

    name: ClassVar[str] = "fmmm_initialize_state"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    _steps: int

    def __init__(self, steps: int = 100) -> None:
        """Store the total refinement budget.

        Parameters
        ----------
        steps : int, default=100
            Total refinement budget across hierarchy levels.

        Returns
        -------
        None
            Configuration stored for use by ``apply()``.
        """
        self._steps = steps

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build the hierarchy and store FM^3 metadata in ``state.extras``.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state receiving FM^3 extras.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with FM^3 hierarchy data stored in ``extras``.
        """
        del ctx

        extent = _layout_extent(problem.num_nodes, problem.node_sizes)
        refinement_area = (2.0 * extent) ** 2
        levels, hierarchy_steps = _build_hierarchy(
            problem.edge_index,
            problem.num_nodes,
            seed=problem.seed,
            edge_weights=problem.edge_weights,
        )

        state.extras["fmmm_levels"] = levels
        state.extras["fmmm_hierarchy_steps"] = hierarchy_steps
        state.extras["fmmm_extent"] = extent
        state.extras["fmmm_refinement_area"] = refinement_area
        state.extras["fmmm_level_budget"] = max(10, self._steps // max(len(levels), 1))
        state.extras["fmmm_steps"] = self._steps
        return state


class _InitializeCoarsestLevel(Op):
    """Initialize the coarsest hierarchy level using classic FR.

    Runs ``layout_fr`` on the coarsest level graph exactly as the classic
    implementation does.
    """

    name: ClassVar[str] = "fmmm_initialize_coarsest"
    category: ClassVar[OpCategory] = OpCategory.INIT
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Seed positions from FR on the coarsest hierarchy level.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs containing ``seed``.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with ``state.pos`` populated from coarse FR layout.
        """
        del ctx

        levels = state.extras["fmmm_levels"]
        steps = state.extras["fmmm_steps"]
        coarsest_level = levels[-1]
        coarsest_nodes = coarsest_level.num_nodes

        coarse_positions = layout_fr(
            coarsest_level.edge_index,
            coarsest_nodes,
            node_sizes=None,
            steps=max(50, steps),
            seed=problem.seed,
            edge_weights=coarsest_level.edge_weights,
        )
        if isinstance(coarse_positions, tuple):
            state.pos = coarse_positions[0].to(dtype=torch.float32)
        else:
            state.pos = coarse_positions.to(dtype=torch.float32)
        return state


class _RefineCoarsestLevel(Op):
    """Refine the coarsest level if the hierarchy has prolongation steps.

    Uses per-edge desired lengths for attraction forces with Barnes-Hut
    repulsion, exactly matching the classic implementation's coarsest-level
    refinement pass.
    """

    name: ClassVar[str] = "fmmm_refine_coarsest"
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
        """Refine the coarsest level with edge-length-aware forces.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with coarse positions.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with refined coarsest-level positions.
        """
        del ctx

        hierarchy_steps = state.extras["fmmm_hierarchy_steps"]
        if not hierarchy_steps:
            return state

        levels = state.extras["fmmm_levels"]
        coarsest_level = levels[-1]
        level_budget = state.extras["fmmm_level_budget"]
        refinement_area = state.extras["fmmm_refinement_area"]

        state.pos = _refine_level_with_edge_lengths(
            state.pos,
            coarsest_level.edge_index,
            coarsest_level.edge_lengths,
            level_budget,
            theta=1.0,
            area=refinement_area,
            edge_weights=coarsest_level.edge_weights,
        )
        return state


class _UncoarsenLoop(Op):
    """Prolong and refine through the hierarchy from coarse to fine.

    Iterates from the coarsest prolongation step back to the finest level,
    prolonging positions using OGDF-style lambda interpolation and then
    refining each level with Barnes-Hut forces.
    """

    name: ClassVar[str] = "fmmm_uncoarsen_loop"
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
        """Prolong and refine through the hierarchy levels.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs containing ``seed``.
        state : SolveState
            Mutable solve state with coarsest-level positions.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with fine-level positions after uncoarsening.
        """
        del ctx

        hierarchy_steps = state.extras["fmmm_hierarchy_steps"]
        if not hierarchy_steps:
            return state

        levels = state.extras["fmmm_levels"]
        level_budget = state.extras["fmmm_level_budget"]
        refinement_area = state.extras["fmmm_refinement_area"]
        rng = random.Random(problem.seed)

        positions = state.pos
        for level in range(len(hierarchy_steps) - 1, -1, -1):
            fine_positions = _prolong_positions(positions, hierarchy_steps[level], rng)
            positions = _refine_level_with_edge_lengths(
                fine_positions,
                levels[level].edge_index,
                levels[level].edge_lengths,
                level_budget,
                theta=1.0,
                area=refinement_area,
                edge_weights=levels[level].edge_weights,
            )

        state.pos = positions
        return state


class _SingleLevelFallback(Op):
    """Refine the single level when no coarsening hierarchy was built.

    When the graph is already small enough that coarsening produces no
    hierarchy steps, this op runs a standard refinement pass on the
    original level using uniform ideal edge length.
    """

    name: ClassVar[str] = "fmmm_single_level_fallback"
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
        """Refine with uniform-length forces if no hierarchy exists.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with FR-initialized positions.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with refined positions.
        """
        del ctx

        hierarchy_steps = state.extras["fmmm_hierarchy_steps"]
        if hierarchy_steps:
            return state

        levels = state.extras["fmmm_levels"]
        level_budget = state.extras["fmmm_level_budget"]
        refinement_area = state.extras["fmmm_refinement_area"]

        state.pos = _refine_level(
            state.pos,
            levels[0].edge_index,
            level_budget,
            theta=1.0,
            area=refinement_area,
            edge_weights=levels[0].edge_weights,
        )
        return state


class _FinalizeFMMMPositions(Op):
    """Apply classic FM^3's final normalization, centering, and scaling."""

    name: ClassVar[str] = "fmmm_finalize_positions"
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
        """Center, normalize, and cast positions like classic FM^3.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state containing refined positions.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with final output positions.

        Raises
        ------
        ValueError
            If ``state.pos`` is missing.
        """
        del ctx

        if state.pos is None:
            raise ValueError("_FinalizeFMMMPositions requires state.pos to be set.")

        device = _layout_device(problem.edge_index, problem.node_sizes)
        extent = state.extras["fmmm_extent"]
        state.pos = _normalize_positions(state.pos.to(device), extent).to(
            dtype=torch.float32, device=device
        )
        return state


def build_fmmm_pipeline(steps: int = 100) -> Pipeline:
    """Build an FM^3 pipeline that is bit-identical to classic ``layout_fmmm``.

    Parameters
    ----------
    steps : int, default=100
        Total refinement budget across hierarchy levels.

    Returns
    -------
    Pipeline
        Pipeline that reproduces classic FM^3's hierarchy construction,
        coarse FR initialization, per-level refinement, prolongation,
        and final normalization.

    Raises
    ------
    ValueError
        If ``steps`` is negative.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative.")

    return Pipeline(
        [
            _InitializeFMMMState(steps=steps),
            _InitializeCoarsestLevel(),
            _RefineCoarsestLevel(),
            _UncoarsenLoop(),
            _SingleLevelFallback(),
            _FinalizeFMMMPositions(),
        ],
        name="fmmm_pipeline",
    )


def layout_fmmm_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 100,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the FM^3 pipeline as a drop-in replacement for classic ``layout_fmmm``.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor used for extent calculation and output device.
    steps : int, default=100
        Total refinement budget across hierarchy levels.
    seed : int, default=42
        Random seed for coarsening, FR initialization, and prolongation jitter.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Final position tensor with the same dtype, device, and values as
        classic ``layout_fmmm``.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``steps``, or ``edge_weights`` are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
            )

    device = _layout_device(edge_index, node_sizes)
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
    final_state = build_fmmm_pipeline(steps=steps).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("FM^3 pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_fmmm_pipeline", "layout_fmmm_pipeline"]
