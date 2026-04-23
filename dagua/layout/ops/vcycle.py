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


def _compose_finest_to_coarse(
    hierarchy: list, level_idx: int, device: torch.device
) -> torch.Tensor:
    """Return tensor mapping finest (original) node id -> coarse id at
    ``hierarchy[level_idx]``.

    ``hierarchy[0].fine_to_coarse`` maps the original problem's node ids to
    level-0 coarse ids. Each subsequent level chains on top of the previous,
    so the composed mapping is just nested indexing.
    """
    composed = hierarchy[0].fine_to_coarse.to(device=device, dtype=torch.long)
    for k in range(1, level_idx + 1):
        next_map = hierarchy[k].fine_to_coarse.to(device=device, dtype=torch.long)
        composed = next_map[composed]
    return composed


def _propagate_flex_to_coarse(
    flex,
    composed_mapping: torch.Tensor,
    num_coarse: int,
):
    """Propagate a finest-level ``FlexConstraints`` to a coarse level.

    Pins: each finest pin is moved onto the coarse node that contains its
    fine target. When multiple fine pins collapse into the same coarse
    node, the first one (lowest fine index) wins. This preserves the
    user-specified target on the coarse-level representative so coarse
    refinement doesn't wander, and prolongation seeds the finer levels
    with a position already close to the user's intent.

    Align groups: fine member indices are mapped through the composed
    coarsening. A group collapses if fewer than 2 distinct coarse members
    remain (nothing to align). Group weight and axis are preserved.

    Flex spacing (flex_node_sep / flex_node_sep_weight) propagates as-is
    -- the target spacing is a user-facing constant that doesn't depend
    on how many coarse representatives we ended up with.
    """
    if flex is None:
        return None

    # `FlexConstraints` is defined in state.py; import lazily to avoid cycles.
    from dagua.layout.ops.state import FlexConstraints  # noqa: WPS433

    device = composed_mapping.device
    coarse_pin_indices = None
    coarse_pin_targets = None
    coarse_pin_weights = None
    coarse_soft_mask = None
    coarse_hard_mask = None
    if flex.pin_indices is not None and flex.pin_indices.numel() > 0:
        fine_idx_cpu = flex.pin_indices.to(device=device, dtype=torch.long)
        coarse_ids = composed_mapping[fine_idx_cpu]  # [P]
        # Dedup priority: a HARD pin on any axis outranks a soft pin on
        # the same coarse node. Among equally-ranked pins (both hard or
        # both soft), the lowest fine index wins -- stable w.r.t. the
        # fine pin order. Without this rule a later soft pin that
        # happens to precede a hard pin in the user's insertion order
        # would silently down-rank the hard pin at the coarse level.
        hard_mask_per_fine = None
        if flex.hard_pin_mask is not None:
            hard_mask_per_fine = flex.hard_pin_mask.any(dim=1).tolist()
        best_per_coarse: dict[int, tuple[int, int]] = {}
        for p, c in enumerate(coarse_ids.tolist()):
            rank = 1 if (hard_mask_per_fine is not None and hard_mask_per_fine[p]) else 0
            prev = best_per_coarse.get(c)
            if prev is None or rank > prev[0]:
                best_per_coarse[c] = (rank, p)
        keep_fine_positions = sorted(pos for _, pos in best_per_coarse.values())
        if keep_fine_positions:
            keep_tensor = torch.tensor(keep_fine_positions, dtype=torch.long, device=device)
            coarse_pin_indices = coarse_ids[keep_tensor]
            coarse_pin_targets = flex.pin_targets.to(device=device)[keep_tensor]
            coarse_pin_weights = flex.pin_weights.to(device=device)[keep_tensor]
            if flex.soft_pin_mask is not None:
                coarse_soft_mask = flex.soft_pin_mask.to(device=device)[keep_tensor]
            if flex.hard_pin_mask is not None:
                coarse_hard_mask = flex.hard_pin_mask.to(device=device)[keep_tensor]

    coarse_align_groups = None
    if flex.align_groups:
        new_groups = []
        for group in flex.align_groups:
            indices, weight, axis = group[0], group[1], group[2]
            fine_indices = indices.to(device=device, dtype=torch.long)
            # Only keep fine indices inside the finest node-id range (the
            # composed_mapping doesn't know about out-of-range ids).
            valid = (fine_indices >= 0) & (fine_indices < composed_mapping.numel())
            if not valid.any():
                continue
            coarse_ids = composed_mapping[fine_indices[valid]]
            unique_coarse = torch.unique(coarse_ids)
            if unique_coarse.numel() >= 2:
                new_groups.append((unique_coarse, weight, axis))
        if new_groups:
            coarse_align_groups = new_groups

    if coarse_pin_indices is None and coarse_align_groups is None and flex.flex_node_sep is None:
        return None

    return FlexConstraints(
        pin_indices=coarse_pin_indices,
        pin_targets=coarse_pin_targets,
        pin_weights=coarse_pin_weights,
        soft_pin_mask=coarse_soft_mask,
        hard_pin_mask=coarse_hard_mask,
        align_groups=coarse_align_groups,
        flex_node_sep=flex.flex_node_sep,
        flex_node_sep_weight=flex.flex_node_sep_weight,
    )


def _level_problem(
    problem: LayoutProblem,
    level: HierarchyLevel,
    flex=None,
) -> LayoutProblem:
    """Build a per-level ``LayoutProblem`` view.

    Parameters
    ----------
    problem : LayoutProblem
        Top-level immutable problem (the finest graph).
    level : HierarchyLevel
        Coarsened level to build a problem for.
    flex : FlexConstraints | None, default=None
        Propagated coarse flex (from ``_propagate_flex_to_coarse``). When
        the level represents a coarsening of the FINEST problem and a
        FlexConstraints view has been propagated, pass it here so coarse
        refinement honours pins and alignment groups; otherwise ``None``
        leaves the coarse level unconstrained.

    Returns
    -------
    LayoutProblem
        A new ``LayoutProblem`` whose ``edge_index`` and ``node_sizes`` come
        from ``level``; direction / seed inherit from the finest problem.
        Clusters do NOT propagate to coarse levels in the Sprint 5 scope
        (cluster-centroid pinning is tracked separately).
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
        flex=flex,
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

        # Precompute flex propagation per level BEFORE init so coarse
        # refinement honours user pins and alignment groups (Sprint 5).
        level_flex: list = [None] * len(hierarchy)
        if problem.flex is not None:
            # Anchor the composed mapping on the device the hierarchy uses.
            ref_tensor = hierarchy[0].fine_to_coarse
            fine_device = ref_tensor.device if ref_tensor is not None else torch.device("cpu")
            for k in range(len(hierarchy)):
                composed = _compose_finest_to_coarse(hierarchy, k, device=fine_device)
                level_flex[k] = _propagate_flex_to_coarse(
                    problem.flex, composed, hierarchy[k].num_nodes
                )

        # 1. Init on coarsest level.
        coarsest_level = hierarchy[-1]
        coarsest_problem = _level_problem(problem, coarsest_level, flex=level_flex[-1])
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
                fine_problem = _level_problem(
                    problem, hierarchy[level_idx - 1], flex=level_flex[level_idx - 1]
                )

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
