"""Generic multilevel V-cycle control-flow ops.

Sprint 2: the dagua_native default uses these ops to layout large graphs via
coarsen -> init on coarsest -> prolong + refine per level. Agnostic to the
specific loss set + init op; those are supplied at construction time.

Design:
- ``VCycleRefine`` is the high-level op. It reads ``state.hierarchy`` (built
  by e.g. ``HeavyEdgeMatching``) and iterates from coarsest to finest,
  constructing a per-level ``LayoutProblem`` view and invoking the supplied
  ``refine_pipeline`` (typically ``build_gradient_core`` from
  ``dagua/layout/ops/pipelines/dagua_native.py``).
- ``coarse_init_op`` is applied once on the coarsest level before refinement.
- Prolongation uses ``DirectMapping``-style ``state.pos = coarse_pos[fine_to_coarse]``
  with optional gaussian jitter to break near-degenerate coarse positions.
- If the hierarchy is empty (graph too small to coarsen) this op is a no-op;
  callers must handle the single-level case.

Budget distribution: caller passes ``coarse_steps`` (applied at the coarsest
level) and ``fine_steps_fn`` that returns per-level step counts. Default is
geometric decay (SFDP-style).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, ClassVar, Tuple

import torch

from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.state import HierarchyLevel, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op


def _level_problem(
    problem: LayoutProblem,
    level: HierarchyLevel,
) -> LayoutProblem:
    """Build a per-level ``LayoutProblem`` view.

    Parameters
    ----------
    problem : LayoutProblem
        Top-level immutable problem (the finest graph).
    level : HierarchyLevel
        Coarsened level to build a problem for.

    Returns
    -------
    LayoutProblem
        A new ``LayoutProblem`` whose ``edge_index`` and ``node_sizes`` come
        from ``level``; direction / seed / flex inherit from the finest
        problem. Clusters and flex do NOT propagate to coarse levels in the
        MVP (Sprint 5 handles flex propagation).
    """
    if level.edge_index is None or level.fine_to_coarse is None:
        raise ValueError("Hierarchy level is offloaded; reload before use.")
    return LayoutProblem(
        edge_index=level.edge_index,
        num_nodes=level.num_nodes,
        node_sizes=level.node_sizes,
        direction=problem.direction,
        clusters=None,
        cluster_parents=None,
        flex=None,
        edge_weights=None,
        seed=problem.seed,
    )


def _default_per_level_steps(
    level_index: int,
    num_levels: int,
    coarse_steps: int,
    finest_steps: int,
) -> int:
    """Geometric-decay step schedule coarse->fine.

    Coarsest level gets ``coarse_steps``. Each finer level gets linearly
    fewer steps, bottoming out at ``finest_steps`` at the finest level.

    Sprint 2 bug-fix (2026-04-22): clamp level_index to [0, num_levels-1].
    Caller used to pass level_index=-1 for the finest pass, which produced
    NEGATIVE step counts -> Repeat ran ZERO iterations -> finest level
    was never refined (the V-cycle's primary quality regression).
    """
    if num_levels <= 1:
        return max(finest_steps, 1)
    level_index = max(0, min(level_index, num_levels - 1))
    frac = level_index / max(num_levels - 1, 1)
    steps = int(finest_steps + (coarse_steps - finest_steps) * frac)
    return max(steps, 1)


@dataclass(frozen=True)
class VCycleRefineConfig:
    """Configuration for :class:`VCycleRefine`.

    Parameters
    ----------
    coarse_steps : int, default=300
        Refinement steps at the coarsest level (gets the most structure-
        forming work since positions are otherwise random).
    finest_steps : int, default=60
        Refinement steps at the finest (user-facing) level.
    jitter_scale : float, default=0.05
        Gaussian jitter added at prolongation so coarse-level ties don't
        collapse fine nodes onto identical positions.
    min_hierarchy_levels : int, default=1
        V-cycle requires at least this many hierarchy levels. Below this,
        VCycleRefine is a no-op and caller must handle single-level layout.
    """

    coarse_steps: int = 300
    finest_steps: int = 60
    jitter_scale: float = 0.05
    min_hierarchy_levels: int = 1


@register_op
@dataclass
class VCycleRefine(Op):
    """Coarsest-init + prolong+refine through all hierarchy levels.

    Sprint 2 core op. Reads ``state.hierarchy`` (built upstream by a
    coarsen op such as ``HeavyEdgeMatching``), initializes positions on the
    coarsest level via ``coarse_init_pipeline``, then iterates from coarsest
    to finest invoking ``refine_pipeline_factory(level_steps)`` per level.

    The init + refine pipelines are supplied as FACTORIES returning
    Pipeline objects so the caller can close over its own loss set
    (e.g. the dagua_native loss list) without this op knowing about them.
    """

    name: ClassVar[str] = "vcycle_refine"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("hierarchy",)
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("hierarchy",)

    coarse_init_pipeline: Callable[[LayoutProblem], Pipeline] = field(default=None)  # type: ignore
    refine_pipeline_factory: Callable[[int], Pipeline] = field(default=None)  # type: ignore
    config: VCycleRefineConfig = field(default_factory=VCycleRefineConfig)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run coarsest-init + per-level prolong + refine.

        Parameters
        ----------
        problem : LayoutProblem
            Finest-level (user-facing) layout problem.
        state : SolveState
            Must have ``state.hierarchy`` populated by an upstream coarsen op.
        ctx : RuntimeContext
            Execution context forwarded to sub-pipelines.

        Returns
        -------
        SolveState
            State with ``state.pos`` populated at the FINEST level.
        """
        hierarchy = state.hierarchy or []
        if len(hierarchy) < self.config.min_hierarchy_levels:
            # Not enough levels -- caller runs single-level pipeline instead.
            return state

        if self.coarse_init_pipeline is None or self.refine_pipeline_factory is None:
            raise ValueError(
                "VCycleRefine requires coarse_init_pipeline and refine_pipeline_factory."
            )

        # 1. Init on coarsest level.
        coarsest_level = hierarchy[-1]
        coarsest_problem = _level_problem(problem, coarsest_level)
        init_pipeline = self.coarse_init_pipeline(coarsest_problem)
        state.pos = None  # force coarse init (NativeEngineInit reseeds when pos is None)
        state = init_pipeline.apply(coarsest_problem, state, ctx)

        # 2. Refine coarsest level.
        coarse_steps = _default_per_level_steps(
            level_index=len(hierarchy) - 1,
            num_levels=len(hierarchy),
            coarse_steps=self.config.coarse_steps,
            finest_steps=self.config.finest_steps,
        )
        state = self.refine_pipeline_factory(coarse_steps).apply(coarsest_problem, state, ctx)

        # 3. Walk coarse -> fine. hierarchy[level_idx].fine_to_coarse maps
        # [num_fine] -> coarse index. The "fine side" of this level is the
        # previous coarsening (hierarchy[level_idx-1]) or the original
        # problem when level_idx == 0.
        for level_idx in range(len(hierarchy) - 1, -1, -1):
            coarse_level = hierarchy[level_idx]
            fine_to_coarse = coarse_level.fine_to_coarse
            if fine_to_coarse is None:
                raise ValueError(f"Hierarchy level {level_idx} has no fine_to_coarse mapping.")

            # Prolong: fine_pos[i] = coarse_pos[fine_to_coarse[i]]
            coarse_pos = state.pos
            if coarse_pos is None:
                raise ValueError("VCycleRefine: state.pos missing before prolong.")
            fine_to_coarse_device = fine_to_coarse.to(device=coarse_pos.device)
            fine_pos = coarse_pos[fine_to_coarse_device].detach().clone()
            if self.config.jitter_scale != 0.0 and coarse_level.num_fine > 0:
                # Use a deterministic generator per level so reproducibility
                # is preserved across runs with the same seed.
                gen = torch.Generator(device="cpu").manual_seed(
                    int(problem.seed) + 1000 * (level_idx + 1)
                )
                jitter = torch.randn(
                    (coarse_level.num_fine, 2),
                    generator=gen,
                    dtype=coarse_pos.dtype,
                ).to(device=coarse_pos.device)
                fine_pos = fine_pos + jitter * self.config.jitter_scale
            fine_pos.requires_grad_(True)
            state.pos = fine_pos

            # Build the problem at the fine side of this coarsening step.
            if level_idx == 0:
                fine_problem = problem  # finest == user's graph
            else:
                fine_problem = _level_problem(problem, hierarchy[level_idx - 1])

            # Sprint 2 bug-fix: reset per-level solve bookkeeping so each
            # level behaves like an independent solve. Carrying step,
            # total_steps, stall_count, and loss history across levels
            # caused premature StallCount break + stale anneal weights.
            state.step = 0
            state.total_steps = 0
            state.stall_count = 0
            state.converged = False
            state.prev_loss = None
            state.extras.pop("stall_last_loss", None)

            # Refine at this level.
            level_steps = _default_per_level_steps(
                level_index=level_idx - 1,
                num_levels=len(hierarchy),
                coarse_steps=self.config.coarse_steps,
                finest_steps=self.config.finest_steps,
            )
            state = self.refine_pipeline_factory(level_steps).apply(fine_problem, state, ctx)

        return state


__all__ = ["VCycleRefine", "VCycleRefineConfig"]
