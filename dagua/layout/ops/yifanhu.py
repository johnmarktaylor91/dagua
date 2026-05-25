"""Composable Yifan Hu multilevel force-directed operations.

The implementation is intentionally dagua-only: no installable Python
reference package was available during Round 33, so these ops provide a native
multilevel force-directed layout rather than a bit-paired port.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Tuple

from dagua.layout.ops.base import Op
from dagua.layout.ops.sfdp import (
    BuildSFDPGraph,
    BuildSFDPHierarchy,
    InitSFDPCoarsestPositions,
    SFDPFinalizePositions,
    SFDPProlongateAndRefineLevels,
    SFDPRefineCoarsestLevel,
)
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_DEFAULT_THETA = 1.2
_DEFAULT_REPULSIVE_EXPONENT = 0.0
_DEFAULT_FINAL_TUNING_FRACTION = 0.2
_SFDP_GRAPH_KEY = "sfdp_graphs"


@register_op
@dataclass(frozen=True)
class BuildYifanHuGraph(Op):
    """Build the undirected weighted graph representation for YifanHu."""

    name: ClassVar[str] = "yifanhu_build_graph"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = ("extras",)
    requires: ClassVar[Tuple[str, ...]] = ()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build the weighted graph used by subsequent multilevel ops.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs containing topology and optional edge
            weights.
        state : SolveState
            Mutable solve state receiving the graph representation.
        ctx : RuntimeContext
            Execution infrastructure forwarded to the shared graph builder.

        Returns
        -------
        SolveState
            State with graph metadata stored in ``extras``.
        """
        return BuildSFDPGraph().apply(problem, state, ctx)


@register_op
@dataclass(frozen=True)
class BuildYifanHuHierarchy(Op):
    """Build a heavy-edge matching hierarchy for YifanHu refinement."""

    name: ClassVar[str] = "yifanhu_build_hierarchy"
    category: ClassVar[OpCategory] = OpCategory.COARSEN
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)
    requires: ClassVar[Tuple[str, ...]] = ("extras.sfdp_base_graph",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Coarsen the graph until matching no longer gives useful reduction.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs containing the random seed.
        state : SolveState
            Mutable solve state with the base graph in ``extras``.
        ctx : RuntimeContext
            Execution infrastructure forwarded to the shared coarsener.

        Returns
        -------
        SolveState
            State with hierarchy graphs, mappings, and deterministic generator
            metadata populated.
        """
        return BuildSFDPHierarchy().apply(problem, state, ctx)


@register_op
@dataclass(frozen=True)
class InitYifanHuCoarsestPositions(Op):
    """Initialize the coarsest hierarchy level for YifanHu."""

    name: ClassVar[str] = "yifanhu_init_coarsest_positions"
    category: ClassVar[OpCategory] = OpCategory.INIT
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("pos", "ideal_length")
    requires: ClassVar[Tuple[str, ...]] = ("extras.sfdp_graphs", "extras.sfdp_generator")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Seed coarsest-level positions and ideal edge length.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. The shared initializer reads seed-derived
            generator state from ``state.extras``.
        state : SolveState
            Mutable solve state receiving initial coordinates.
        ctx : RuntimeContext
            Execution infrastructure forwarded to the shared initializer.

        Returns
        -------
        SolveState
            State with ``pos`` and ``ideal_length`` set for the coarsest graph.
        """
        return InitSFDPCoarsestPositions().apply(problem, state, ctx)


@register_op
@dataclass(frozen=True)
class YifanHuRefineCoarsestLevel(Op):
    """Relax the coarsest YifanHu level with Barnes-Hut force steps.

    Parameters
    ----------
    steps : int, default=500
        Maximum number of force-directed iterations.
    theta : float, default=1.2
        Barnes-Hut opening angle. Gephi's YifanHu exposes a permissive default,
        so this pipeline uses ``1.2`` rather than SFDP's stricter ``0.6``.
    repulsive_exponent : float, default=0.0
        Inverse-distance repulsion exponent used for Hu-style updates.
    """

    steps: int = 500
    theta: float = _DEFAULT_THETA
    repulsive_exponent: float = _DEFAULT_REPULSIVE_EXPONENT

    name: ClassVar[str] = "yifanhu_refine_coarsest_level"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("pos", "ideal_length", "extras", "converged")
    writes: ClassVar[Tuple[str, ...]] = ("pos", "extras", "converged")
    requires: ClassVar[Tuple[str, ...]] = ("pos", "ideal_length", "extras.sfdp_graphs")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run coarsest-level YifanHu force refinement.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with coarsest positions.
        ctx : RuntimeContext
            Execution infrastructure used by repeated force ops.

        Returns
        -------
        SolveState
            State with refined coarsest-level positions.
        """
        return SFDPRefineCoarsestLevel(
            steps=self.steps,
            theta=self.theta,
            repulsive_exponent=self.repulsive_exponent,
        ).apply(problem, state, ctx)


@register_op
@dataclass(frozen=True)
class YifanHuProlongateAndRefineLevels(Op):
    """Prolongate through finer levels and refine each YifanHu level.

    Parameters
    ----------
    steps : int, default=500
        Maximum number of force-directed iterations per hierarchy level.
    theta : float, default=1.2
        Barnes-Hut opening angle threshold.
    repulsive_exponent : float, default=0.0
        Inverse-distance repulsion exponent used for Hu-style updates.
    """

    steps: int = 500
    theta: float = _DEFAULT_THETA
    repulsive_exponent: float = _DEFAULT_REPULSIVE_EXPONENT

    name: ClassVar[str] = "yifanhu_prolongate_and_refine_levels"
    category: ClassVar[OpCategory] = OpCategory.PROLONG
    reads: ClassVar[Tuple[str, ...]] = ("pos", "ideal_length", "extras", "converged")
    writes: ClassVar[Tuple[str, ...]] = ("pos", "ideal_length", "extras", "converged")
    requires: ClassVar[Tuple[str, ...]] = (
        "pos",
        "ideal_length",
        "extras.sfdp_graphs",
        "extras.sfdp_mappings",
        "extras.sfdp_generator",
    )

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Interpolate positions onto each finer graph and relax them.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state positioned at the current hierarchy level.
        ctx : RuntimeContext
            Execution infrastructure used by repeated force ops.

        Returns
        -------
        SolveState
            State with positions on the finest graph.
        """
        return SFDPProlongateAndRefineLevels(
            steps=self.steps,
            theta=self.theta,
            repulsive_exponent=self.repulsive_exponent,
        ).apply(problem, state, ctx)


@register_op
@dataclass(frozen=True)
class YifanHuFinalTuning(Op):
    """Run a short final force pass on the finest graph.

    Parameters
    ----------
    steps : int, default=100
        Number of final tuning iterations on the original graph.
    theta : float, default=1.2
        Barnes-Hut opening angle threshold.
    repulsive_exponent : float, default=0.0
        Inverse-distance repulsion exponent used for Hu-style updates.
    """

    steps: int = 100
    theta: float = _DEFAULT_THETA
    repulsive_exponent: float = _DEFAULT_REPULSIVE_EXPONENT

    name: ClassVar[str] = "yifanhu_final_tuning"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("pos", "ideal_length", "extras", "converged")
    writes: ClassVar[Tuple[str, ...]] = ("pos", "extras", "converged")
    requires: ClassVar[Tuple[str, ...]] = ("pos", "ideal_length", "extras.sfdp_graphs")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Refine the final graph after all prolongation is complete.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state containing finest-level positions.
        ctx : RuntimeContext
            Execution infrastructure used by repeated force ops.

        Returns
        -------
        SolveState
            State with final-tuned finest-level positions.
        """
        if self.steps <= 0:
            return state

        graphs = state.extras.get(_SFDP_GRAPH_KEY)
        if not graphs:
            raise ValueError("YifanHuFinalTuning requires hierarchy graph metadata.")

        original_graphs = graphs
        state.extras[_SFDP_GRAPH_KEY] = [graphs[0]]
        try:
            return SFDPRefineCoarsestLevel(
                steps=self.steps,
                theta=self.theta,
                repulsive_exponent=self.repulsive_exponent,
            ).apply(problem, state, ctx)
        finally:
            state.extras[_SFDP_GRAPH_KEY] = original_graphs


@register_op
@dataclass(frozen=True)
class YifanHuFinalizePositions(Op):
    """Normalize, orient, and cast final YifanHu coordinates."""

    name: ClassVar[str] = "yifanhu_finalize_positions"
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
        """Center, direction-orient, scale, and move output positions to device.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs used for output extent and device.
        state : SolveState
            Mutable solve state containing final positions.
        ctx : RuntimeContext
            Execution infrastructure forwarded to the shared finalizer.

        Returns
        -------
        SolveState
            State with final ``float32`` coordinates shaped ``[N, 2]``.
        """
        return SFDPFinalizePositions().apply(problem, state, ctx)


def final_tuning_steps(steps: int) -> int:
    """Compute the YifanHu final-tuning iteration budget.

    Parameters
    ----------
    steps : int
        Main per-level iteration budget.

    Returns
    -------
    int
        Short final pass iteration count. A positive main budget gets at least
        one final tuning step.
    """
    if steps <= 0:
        return 0
    return max(1, int(round(float(steps) * _DEFAULT_FINAL_TUNING_FRACTION)))


__all__ = [
    "BuildYifanHuGraph",
    "BuildYifanHuHierarchy",
    "InitYifanHuCoarsestPositions",
    "YifanHuRefineCoarsestLevel",
    "YifanHuProlongateAndRefineLevels",
    "YifanHuFinalTuning",
    "YifanHuFinalizePositions",
    "final_tuning_steps",
]
