"""Deprecated legacy monolith for dagua's native tensor layout engine.

Config-time helpers live in ``dagua.layout.resolve``. The pipeline body here
is pure composed ops; no inline helpers, no imports from
``dagua.layout.engine``.

This module preserves the sprint-20d all-in-one native path for the
``force_pipeline="legacy_monolith"`` escape hatch. New topology dispatch lives
in ``dagua.layout.ops.pipelines.dagua_native`` and the ``native_*`` modules.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Optional, cast

import numpy as np
import torch

from dagua.config import LayoutConfig
from dagua.layout.graph_classify import GraphStructure, classify_graph
from dagua.layout.ops.anneal import (
    InitAnnealingSchedule,
    InitAnnealingScheduleConfig,
    WeightAnnealing,
)
from dagua.layout.ops.barycenter import BarycenterReorder, BarycenterReorderConfig
from dagua.layout.ops.base import EarlyBreak, LossGroup, Pipeline, Repeat
from dagua.layout.ops.cluster_arrange import (
    ClusterGridArrange,
    ClusterGridArrangeConfig,
)
from dagua.layout.ops.coarsen import HeavyEdgeMatching
from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig, StallCount, StallCountConfig
from dagua.layout.ops.coordinate import (
    BrandesKoepfHorizontalRefine,
    BrandesKoepfHorizontalRefineConfig,
    ClusterAwareXCompaction,
    ClusterAwareXCompactionConfig,
    RankRowSnap,
    RankRowSnapConfig,
)
from dagua.layout.ops.distance import (
    PivotDistanceQueries,
    PivotSelection,
    PivotSelectionConfig,
)
from dagua.layout.ops.force_2d_init import Force2DInitIfFlat, Force2DInitIfFlatConfig
from dagua.layout.ops.init import (
    NativeEngineInit,
    NativeEngineInitConfig,
)
from dagua.layout.ops.layering import ActivateExpandedGraphState, InsertDummyNodes
from dagua.layout.ops.optimize import (
    ClipGradNorm,
    ClipGradNormConfig,
    CreateOptimizer,
    CreateOptimizerConfig,
    OptimizerStep,
    OptimizerZeroGrad,
)
from dagua.layout.ops.ordering import (
    ClusterContiguousOrder,
    ClusterContiguousOrderConfig,
    MedianSweep,
    MedianSweepConfig,
    TransposeHeuristic,
    TransposeHeuristicConfig,
)
from dagua.layout.ops.postprocess import AspectRatioFit, AspectRatioFitConfig, StripDummyNodes
from dagua.layout.ops.preprocess import BuildAdjacency, BuildAdjacencyConfig, DetectComponents
from dagua.layout.ops.project import (
    HardPinProjection,
    OverlapProjection,
    OverlapProjectionConfig,
    PeriodicOverlapProjection,
    PeriodicOverlapProjectionConfig,
)
from dagua.layout.ops.state import (
    ExecutionPlan,
    FlexConstraints,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)
from dagua.layout.ops.vcycle import VCycleRefine, VCycleRefineConfig
from dagua.layout.resolve import (
    build_flex_constraints,
    build_loss_ops,
    normalize_node_sizes,
    prepare_pipeline_config,
    resolve_quality_budgets,
)
from dagua.utils import longest_path_layering

_COMPONENT_TILE_PAD_FACTOR = 2.0
_COMPONENT_PACK_TARGET_ASPECT = 1.0
_COMPONENT_PACK_AREA_WEIGHT = 0.05
_COMPONENT_DOMINANCE_SKIP_FRACTION = 0.85
# Tiny hand-authored DAGs rarely have enough long-edge mass for dummy nodes to
# help, and one skip edge can dominate the post-strip geometry.
_DUMMY_NODE_MIN_NODES = 20
_SMALL_DAG_MEDIAN_TRANSPOSE_MAX_NODES = 30
PackedComponent = tuple[torch.Tensor, torch.Tensor, float, float]


def _resolve_native_layer_assignments(
    edge_index: torch.Tensor,
    num_nodes: int,
    layer_assignments: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Return CPU layer assignments for native dummy-node gating.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity with shape ``[2, E]``.
    num_nodes : int
        Number of nodes in the graph.
    layer_assignments : torch.Tensor | None
        Optional caller-provided layer assignments with shape ``[N]``.

    Returns
    -------
    torch.Tensor | None
        CPU long layer assignments when they can be resolved.
    """
    if layer_assignments is not None:
        return layer_assignments.detach().to(device="cpu", dtype=torch.long)
    if num_nodes == 0 or edge_index.numel() == 0:
        return None
    resolved = longest_path_layering(edge_index.detach().to(device="cpu"), num_nodes, device="cpu")
    if isinstance(resolved, torch.Tensor):
        return resolved.to(device="cpu", dtype=torch.long)
    return torch.tensor(resolved, dtype=torch.long)


def _has_long_layer_edges(
    edge_index: torch.Tensor,
    layer_assignments: Optional[torch.Tensor],
) -> bool:
    """Return whether any edge spans at least two layers.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity with shape ``[2, E]``.
    layer_assignments : torch.Tensor | None
        Layer assignments with shape ``[N]``.

    Returns
    -------
    bool
        ``True`` when at least one edge has target-source layer span ``>= 2``.
    """
    if layer_assignments is None or edge_index.numel() == 0:
        return False
    edges_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    spans = layer_assignments[edges_cpu[1]] - layer_assignments[edges_cpu[0]]
    return bool((spans >= 2).any().item())


def _max_layer_width_from_layers(layer_assignments: Optional[torch.Tensor]) -> Optional[int]:
    """Return the widest layer implied by layer assignments.

    Parameters
    ----------
    layer_assignments : torch.Tensor | None
        Layer assignments with shape ``[N]``.

    Returns
    -------
    int | None
        Maximum number of nodes in any layer, or ``None`` when no layer
        assignment is available.
    """
    if layer_assignments is None or layer_assignments.numel() == 0:
        return None
    layers_cpu = layer_assignments.detach().to(device="cpu", dtype=torch.long)
    normalized_layers = layers_cpu - int(layers_cpu.min().item())
    return int(torch.bincount(normalized_layers).max().item())


def _resolved_max_layer_width(
    structure: Optional[GraphStructure],
    layer_assignments: Optional[torch.Tensor],
) -> Optional[int]:
    """Return classified or computed max layer width for native gates.

    Parameters
    ----------
    structure : GraphStructure | None
        Optional graph classification metadata.
    layer_assignments : torch.Tensor | None
        Layer assignments with shape ``[N]``.

    Returns
    -------
    int | None
        Maximum layer width when it can be resolved.
    """
    if structure is not None:
        classified_width = int(getattr(structure, "max_layer_width", 0))
        if classified_width > 0:
            return classified_width
    return _max_layer_width_from_layers(layer_assignments)


def _should_use_native_dummy_nodes(
    config: LayoutConfig,
    structure: Optional[GraphStructure],
    edge_index: torch.Tensor,
    layer_assignments: Optional[torch.Tensor],
) -> bool:
    """Return whether native should expand long DAG edges with dummies.

    Parameters
    ----------
    config : LayoutConfig
        Prepared layout configuration.
    structure : GraphStructure | None
        Classified topology for the current connected component.
    edge_index : torch.Tensor
        Graph connectivity with shape ``[2, E]``.
    layer_assignments : torch.Tensor | None
        Resolved layer assignments with shape ``[N]``.

    Returns
    -------
    bool
        ``True`` only for connected, non-flat DAGs with at least one long edge.
    """
    if not bool(getattr(config, "insert_dummy_nodes", True)):
        return False
    if structure is None:
        return False
    if not bool(getattr(structure, "is_directed_acyclic", getattr(structure, "is_acyclic", True))):
        return False
    if int(getattr(structure, "num_components", 1)) != 1:
        return False
    if int(getattr(structure, "num_layers", 0)) <= 1:
        return False
    if layer_assignments is None:
        return False
    num_nodes = int(layer_assignments.shape[0])
    has_long_layer_edges = _has_long_layer_edges(
        edge_index=edge_index,
        layer_assignments=layer_assignments,
    )
    max_layer_width = _resolved_max_layer_width(
        structure=structure,
        layer_assignments=layer_assignments,
    )
    if num_nodes < _DUMMY_NODE_MIN_NODES and not (
        has_long_layer_edges and max_layer_width is not None and max_layer_width >= 1
    ):
        return False
    if "dense_dag" in getattr(structure, "topology_tags", ()):
        return False
    return has_long_layer_edges


def _should_apply_brandes_koepf_refine(
    config: LayoutConfig,
    structure: Optional[GraphStructure],
    layer_assignments: Optional[torch.Tensor],
) -> bool:
    """Return whether the native pipeline should enable BK refinement.

    Parameters
    ----------
    config : LayoutConfig
        Prepared layout configuration.
    structure : GraphStructure | None
        Classified topology for the current problem.
    layer_assignments : torch.Tensor | None
        Layer assignments with shape ``[N]``.

    Returns
    -------
    bool
        ``True`` unless the user disabled BK.
    """
    if not bool(getattr(config, "brandes_koepf_refine", True)):
        return False
    return True


def _should_use_native_median_transpose(
    config: LayoutConfig,
    is_acyclic: bool,
) -> bool:
    """Return whether native median/transpose crossing reduction should run.

    Parameters
    ----------
    config : LayoutConfig
        Prepared layout configuration.
    is_acyclic : bool
        Whether the active graph is directed acyclic.

    Returns
    -------
    bool
        ``True`` for acyclic graphs larger than the tiny-DAG cutoff when the
        user-facing feature flag is enabled.
    """
    if not bool(getattr(config, "use_native_median_transpose", True)):
        return False
    if not is_acyclic:
        return False
    num_nodes = getattr(config, "_dagua_native_num_nodes", None)
    if num_nodes is not None and int(num_nodes) <= _SMALL_DAG_MEDIAN_TRANSPOSE_MAX_NODES:
        return False
    return True


def _stress_pivot_prep(config: LayoutConfig) -> list:
    """Return pivot-prep ops for Sprint 15 stress loss when w_stress > 0.

    Ops run once at pipeline entry (after NativeEngineInit, before the
    optimizer loop) so every gradient step can read cached
    state.pivot_indices + state.pivot_distances without rebuilding them.
    """
    if getattr(config, "w_stress", 0.0) <= 0.0:
        return []
    n_pivots = int(getattr(config, "w_stress_n_pivots", 50))
    weighted = False  # pivot BFS uses unweighted graph-theoretic distance
    return [
        BuildAdjacency(BuildAdjacencyConfig(weighted=weighted)),
        PivotSelection(PivotSelectionConfig(n_pivots=n_pivots)),
        PivotDistanceQueries(),
    ]


def build_gradient_core(
    losses: list,
    steps: int,
    overlap_interval: int,
    stall_limit: int,
    rel_threshold: float,
    time_budget_s: Optional[float] = None,
) -> Pipeline:
    """Build the inner differentiable optimizer as a named sub-pipeline.

    Sprint 1 exit criterion: "the 'inner differentiable optimizer' is its
    own named sub-pipeline (`GradientCore`), pluggable between initializers."
    Packages the Repeat loop body + PeriodicOverlapProjection +
    StallCount + EarlyBreak as a standalone Pipeline called "gradient_core"
    that composes into the top-level pipeline, letting Sprint 2+ swap
    initializers without touching the optimization loop.

    Parameters
    ----------
    losses : list
        Weighted loss operators to evaluate each step.
    steps : int
        Maximum optimization steps.
    overlap_interval : int
        Periodic overlap projection interval.
    stall_limit : int
        Consecutive-no-improve steps before early break.
    rel_threshold : float
        Relative-loss threshold for stall detection.
    time_budget_s : float, optional
        Wall-clock budget in seconds. When exceeded after an optimization
        step, the loop exits and downstream final polish still runs.

    Returns
    -------
    Pipeline
        Named ``gradient_core`` sub-pipeline.
    """
    return Pipeline(
        [
            Repeat(
                n=steps,
                ops=[
                    WeightAnnealing(),
                    OptimizerZeroGrad(),
                    # Sprint 1 memory port: per_loss backward frees each
                    # loss term's autograd graph immediately, cutting peak
                    # RSS 3-4x vs combined.
                    LossGroup(losses=losses, backward_mode="per_loss"),
                    ClipGradNorm(ClipGradNormConfig(max_norm=100.0)),
                    OptimizerStep(),
                    HardPinProjection(),
                    PeriodicOverlapProjection(
                        PeriodicOverlapProjectionConfig(
                            interval=overlap_interval,
                            padding=2.0,
                            iterations=None,
                            run_on_last_step=True,
                        ),
                    ),
                    StallCount(
                        StallCountConfig(
                            limit=stall_limit,
                            rel_threshold=rel_threshold,
                            time_budget_s=time_budget_s,
                        ),
                    ),
                    EarlyBreak(lambda problem, state, ctx: state.converged),
                ],
            ),
        ],
        name="gradient_core",
    )


def _build_refine_pipeline_factory(
    losses: list,
    overlap_interval: int,
    stall_limit: int,
    rel_threshold: float,
    resolved_node_sep: float,
    resolved_rank_sep: float,
    resolved_device: str,
    resolved_verbose: bool,
    weight_config: InitAnnealingScheduleConfig,
    optimizer_type: str,
    lr: float,
    time_budget_s: Optional[float] = None,
):
    """Return a factory that builds a refine pipeline with a given step count.

    The V-cycle calls this per level to get a Pipeline sized to that level's
    budget. Each invocation re-runs NativeEngineInit (which recomputes
    layer_index + layers for the current level's edge_index while
    preserving state.pos if already set, so prolonged positions survive)
    + anneal + optimizer creation + gradient_core.
    """

    def _factory(steps: int) -> Pipeline:
        return Pipeline(
            [
                # Recompute layers/layer_index for the current level's
                # edge_index. NativeEngineInit preserves state.pos if set
                # (so prolonged coarse positions pass through untouched).
                NativeEngineInit(
                    NativeEngineInitConfig(
                        node_sep=resolved_node_sep,
                        rank_sep=resolved_rank_sep,
                        device=resolved_device,
                        verbose=resolved_verbose,
                    ),
                ),
                InitAnnealingSchedule(weight_config),
                CreateOptimizer(
                    CreateOptimizerConfig(
                        optimizer_type=optimizer_type,
                        lr=lr,
                        target="pos",
                        key="default",
                    ),
                ),
                build_gradient_core(
                    losses=losses,
                    steps=steps,
                    overlap_interval=overlap_interval,
                    stall_limit=stall_limit,
                    rel_threshold=rel_threshold,
                    time_budget_s=time_budget_s,
                ),
            ],
            name=f"vcycle_refine_level_{steps}",
        )

    return _factory


def _build_coarse_init_pipeline_factory(
    resolved_node_sep: float,
    resolved_rank_sep: float,
    resolved_device: str,
    resolved_verbose: bool,
    optimizer_type: str,
    lr: float,
    weight_config: InitAnnealingScheduleConfig,
):
    """Return a factory that builds the coarse-level init pipeline for a level."""

    def _factory(level_problem: LayoutProblem) -> Pipeline:
        del level_problem  # problem is passed at apply-time; no per-level config yet
        return Pipeline(
            [
                NativeEngineInit(
                    NativeEngineInitConfig(
                        node_sep=resolved_node_sep,
                        rank_sep=resolved_rank_sep,
                        device=resolved_device,
                        verbose=resolved_verbose,
                    ),
                ),
                InitAnnealingSchedule(weight_config),
                CreateOptimizer(
                    CreateOptimizerConfig(
                        optimizer_type=optimizer_type,
                        lr=lr,
                        target="pos",
                        key="default",
                    ),
                ),
            ],
            name="vcycle_coarse_init",
        )

    return _factory


def _prepare_native_config(
    config: LayoutConfig,
    num_nodes: int,
    edge_index: torch.Tensor,
    device: str,
    optimizer_type: str,
    layer_assignments: Optional[torch.Tensor] = None,
    prebuilt_layer_index: Optional[Any] = None,
    graph_structure: Optional[GraphStructure] = None,
    skip_classification: bool = False,
) -> LayoutConfig:
    """Resolve one native-pipeline config for a specific problem instance.

    Parameters
    ----------
    config : LayoutConfig
        User-facing base configuration.
    num_nodes : int
        Number of nodes in the current problem.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    device : str
        Target execution device name.
    optimizer_type : str
        Optimizer name forwarded into private native config attrs.
    layer_assignments : torch.Tensor, optional
        Optional layer assignments with shape ``[N]`` for the current problem.
    prebuilt_layer_index : Any, optional
        Optional prebuilt layer index for the current problem.
    graph_structure : GraphStructure, optional
        Optional pre-classified structure for the current problem.
    skip_classification : bool, default=False
        Whether to skip graph classification during config preparation.

    Returns
    -------
    LayoutConfig
        Shallow config copy annotated with resolved private native-pipeline
        metadata for this problem instance.
    """
    prepared_config = prepare_pipeline_config(
        config=config,
        num_nodes=num_nodes,
        edge_index=edge_index,
        device=device,
        layer_assignments=layer_assignments,
        prebuilt_layer_index=prebuilt_layer_index,
        graph_structure=graph_structure,
        skip_classification=skip_classification,
    )
    setattr(prepared_config, "_dagua_native_optimizer_type", optimizer_type)
    structure = getattr(prepared_config, "_dagua_native_structure", None)
    resolved_layers = _resolve_native_layer_assignments(
        edge_index=edge_index,
        num_nodes=num_nodes,
        layer_assignments=layer_assignments,
    )
    if resolved_layers is not None:
        setattr(prepared_config, "_dagua_native_layer_assignments", resolved_layers)
    setattr(
        prepared_config,
        "_dagua_native_use_dummy_nodes",
        _should_use_native_dummy_nodes(
            config=prepared_config,
            structure=structure,
            edge_index=edge_index,
            layer_assignments=resolved_layers,
        ),
    )
    return prepared_config


def _run_native_problem(
    problem: LayoutProblem,
    state: SolveState,
    ctx: RuntimeContext,
    config: LayoutConfig,
) -> torch.Tensor:
    """Run the native pipeline for one already-prepared problem.

    Parameters
    ----------
    problem : LayoutProblem
        Prepared layout problem for one connected component or the full graph.
    state : SolveState
        Mutable state for the solve.
    ctx : RuntimeContext
        Execution context shared across the solve.
    config : LayoutConfig
        Resolved native config for ``problem``.

    Returns
    -------
    torch.Tensor
        Final positions with shape ``[N, 2]``.

    Raises
    ------
    RuntimeError
        If the pipeline completes without producing positions.
    """
    from dagua.layout.graph_classify import GraphFamily
    from dagua.layout.ops.coordinate import (
        ReingoldTilfordTree,
        ReingoldTilfordTreeConfig,
    )

    structure: Any = problem.structure
    if structure is None:
        structure = getattr(config, "structure", None)
    if structure is None:
        structure = classify_graph(problem.edge_index, problem.num_nodes)
        problem.structure = cast(Any, structure)

    if (
        getattr(structure, "family", None) == GraphFamily.TREE
        and getattr(config, "use_tree_fast_path", True)
        and problem.num_nodes > 0
    ):
        rt_state = ReingoldTilfordTree(ReingoldTilfordTreeConfig()).apply(problem, state, ctx)
        if rt_state.pos is not None:
            return rt_state.pos.detach()

    final_state = build_dagua_pipeline(config).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("dagua_native pipeline did not produce final positions.")
    result = final_state.pos.detach()
    if result.shape[0] > problem.num_nodes:
        result = result[: problem.num_nodes]
    return result


def _has_pins(flex: Optional[FlexConstraints]) -> bool:
    """Return whether prepared flex constraints contain pinned nodes.

    Parameters
    ----------
    flex : FlexConstraints, optional
        Prepared flex constraints for the current problem.

    Returns
    -------
    bool
        ``True`` when at least one pin is present.
    """
    if flex is None or flex.pin_indices is None:
        return False
    return int(flex.pin_indices.numel()) > 0


def _has_cross_component_flex(
    flex: Optional[FlexConstraints],
    component_ids: Optional[torch.Tensor],
) -> bool:
    """Return whether any alignment group spans multiple weak components.

    Parameters
    ----------
    flex : FlexConstraints, optional
        Prepared flex constraints for the parent problem.
    component_ids : torch.Tensor, optional
        Weak-component labels with shape ``[N]``.

    Returns
    -------
    bool
        ``True`` when an alignment group references nodes from multiple
        components. Global spacing flex is allowed because it is re-applied
        inside each child solve and does not bind specific components
        together.
    """
    if flex is None or component_ids is None or not flex.align_groups:
        return False

    labels = component_ids.to(dtype=torch.long)
    for group_indices, _, _ in flex.align_groups:
        members = group_indices.to(device=labels.device, dtype=torch.long)
        if members.numel() < 2:
            continue
        if torch.unique(labels[members], sorted=False).numel() > 1:
            return True
    return False


def _should_decompose_components(
    problem: LayoutProblem,
    config: LayoutConfig,
    component_ids: Optional[torch.Tensor],
) -> bool:
    """Return whether adapter-level component decomposition should run.

    Parameters
    ----------
    problem : LayoutProblem
        Prepared parent problem.
    config : LayoutConfig
        Resolved native config for the parent problem.
    component_ids : torch.Tensor, optional
        Weak-component labels with shape ``[N]``.

    Returns
    -------
    bool
        ``True`` when the parent graph is safe and useful to split into
        independent weak components.
    """
    if not getattr(config, "decompose_components", True):
        return False
    if problem.num_nodes < 2:
        return False
    if problem.clusters:
        return False
    if _has_pins(problem.flex):
        return False

    structure = problem.structure
    if structure is not None and int(getattr(structure, "num_components", 1)) <= 1:
        return False

    if component_ids is None or component_ids.numel() == 0:
        return False
    if int(component_ids.max().item()) <= 0:
        return False
    component_sizes = torch.bincount(component_ids.to(dtype=torch.long))
    if component_sizes.numel() > 0:
        largest_component = int(component_sizes.max().item())
        if largest_component / max(problem.num_nodes, 1) >= _COMPONENT_DOMINANCE_SKIP_FRACTION:
            return False
        sorted_sizes, _ = torch.sort(component_sizes, descending=True)
        second_largest = int(sorted_sizes[1].item()) if sorted_sizes.numel() > 1 else 0
        singleton_components = int((component_sizes == 1).sum().item())
        if singleton_components >= 3 and largest_component >= (10 * max(second_largest, 1)):
            return False
    if _has_cross_component_flex(problem.flex, component_ids):
        return False
    return True


def _subset_flex(
    flex: Optional[FlexConstraints],
    local_index: torch.Tensor,
) -> Optional[FlexConstraints]:
    """Project parent flex constraints into one component-local node space.

    Parameters
    ----------
    flex : FlexConstraints, optional
        Parent flex constraints.
    local_index : torch.Tensor
        Mapping from parent node id to child-local id with shape ``[N_parent]``.
        Nodes outside the child component contain ``-1``.

    Returns
    -------
    FlexConstraints | None
        Child-local flex constraints. Alignment groups that shrink below two
        in-component members are dropped.
    """
    if flex is None:
        return None

    device = local_index.device
    pin_indices = None
    pin_targets = None
    pin_weights = None
    soft_pin_mask = None
    hard_pin_mask = None
    if flex.pin_indices is not None and flex.pin_indices.numel() > 0:
        parent_pins = flex.pin_indices.to(device=device, dtype=torch.long)
        pin_mask = local_index[parent_pins] >= 0
        if pin_mask.any():
            pin_indices = local_index[parent_pins[pin_mask]]
            if flex.pin_targets is not None:
                pin_targets = flex.pin_targets.to(device=device, dtype=torch.float32)[pin_mask]
            if flex.pin_weights is not None:
                pin_weights = flex.pin_weights.to(device=device, dtype=torch.float32)[pin_mask]
            if flex.soft_pin_mask is not None:
                soft_pin_mask = flex.soft_pin_mask.to(device=device, dtype=torch.bool)[pin_mask]
            if flex.hard_pin_mask is not None:
                hard_pin_mask = flex.hard_pin_mask.to(device=device, dtype=torch.bool)[pin_mask]

    align_groups = None
    if flex.align_groups:
        projected_groups: list[tuple[torch.Tensor, float, int]] = []
        for indices, weight, axis in flex.align_groups:
            members = indices.to(device=device, dtype=torch.long)
            valid_members = local_index[members]
            local_members = valid_members[valid_members >= 0]
            unique_members = torch.unique(local_members, sorted=True)
            if unique_members.numel() >= 2:
                projected_groups.append((unique_members, float(weight), int(axis)))
        if projected_groups:
            align_groups = projected_groups

    if pin_indices is None and align_groups is None and flex.flex_node_sep is None:
        return None

    return FlexConstraints(
        pin_indices=pin_indices,
        pin_targets=pin_targets,
        pin_weights=pin_weights,
        soft_pin_mask=soft_pin_mask,
        hard_pin_mask=hard_pin_mask,
        align_groups=align_groups,
        flex_node_sep=flex.flex_node_sep,
        flex_node_sep_weight=flex.flex_node_sep_weight,
    )


def _extract_component_problem(
    parent_problem: LayoutProblem,
    parent_state: SolveState,
    component_nodes: torch.Tensor,
    layer_assignments: Optional[torch.Tensor] = None,
) -> tuple[LayoutProblem, SolveState, torch.Tensor, Optional[torch.Tensor]]:
    """Build one relabeled child problem for a weak component.

    Parameters
    ----------
    parent_problem : LayoutProblem
        Prepared parent problem.
    parent_state : SolveState
        Parent solve state. Only ``pos`` is read for optional init seeding.
    component_nodes : torch.Tensor
        Parent node indices belonging to one weak component with shape ``[K]``.
    layer_assignments : torch.Tensor, optional
        Optional parent layer assignments with shape ``[N_parent]``.

    Returns
    -------
    tuple[LayoutProblem, SolveState, torch.Tensor, torch.Tensor | None]
        Child problem, child state, parent index map with shape ``[K]``, and
        optional child layer assignments with shape ``[K]``.
    """
    device = parent_problem.edge_index.device
    parent_indices = component_nodes.to(device=device, dtype=torch.long)
    local_index = torch.full((parent_problem.num_nodes,), -1, dtype=torch.long, device=device)
    local_index[parent_indices] = torch.arange(
        parent_indices.numel(),
        device=device,
        dtype=torch.long,
    )

    membership = torch.zeros(parent_problem.num_nodes, dtype=torch.bool, device=device)
    membership[parent_indices] = True
    edge_index = parent_problem.edge_index
    if edge_index.numel() == 0:
        edge_mask = torch.zeros((0,), dtype=torch.bool, device=device)
    else:
        edge_mask = membership[edge_index[0]] & membership[edge_index[1]]

    sub_edge_index = local_index[edge_index[:, edge_mask]]
    sub_node_sizes = (
        None if parent_problem.node_sizes is None else parent_problem.node_sizes[parent_indices]
    )
    sub_edge_weights = (
        None if parent_problem.edge_weights is None else parent_problem.edge_weights[edge_mask]
    )

    sub_init_pos = None
    if parent_state.pos is not None:
        sub_init_pos = parent_state.pos[parent_indices].clone()
        if sub_init_pos.numel() > 0:
            sub_init_pos -= sub_init_pos.mean(dim=0, keepdim=True)

    sub_layer_assignments = None
    if layer_assignments is not None:
        sub_layer_assignments = layer_assignments.to(device=device, dtype=torch.long)[
            parent_indices
        ].clone()

    # Propagate an UNDIRECTED parent verdict to component children:
    # GraphStructure already carries ``is_semantically_directed`` and
    # classify_graph's ``graph=`` override reads that same attribute via
    # getattr, so passing the parent structure keeps an undirected
    # declaration (or inference) from being silently re-inferred per weak
    # component. Restricted to the undirected case so directed graphs keep
    # their exact prior per-component classification (bit-identical
    # default-path guarantee).
    parent_structure = parent_problem.structure
    direction_override = (
        parent_structure
        if getattr(parent_structure, "is_semantically_directed", None) is False
        else None
    )
    classified_structure = classify_graph(
        sub_edge_index.detach().to(device="cpu", dtype=torch.long),
        int(parent_indices.numel()),
        layer_assignments=(
            None
            if sub_layer_assignments is None
            else sub_layer_assignments.detach().to(device="cpu", dtype=torch.long)
        ),
        graph=direction_override,
    )
    child_problem = LayoutProblem(
        edge_index=sub_edge_index,
        num_nodes=int(parent_indices.numel()),
        node_sizes=sub_node_sizes,
        direction=parent_problem.direction,
        clusters=None,
        cluster_parents=None,
        structure=cast(Any, classified_structure),
        flex=_subset_flex(parent_problem.flex, local_index),
        edge_weights=sub_edge_weights,
        seed=parent_problem.seed,
    )
    return child_problem, SolveState(pos=sub_init_pos), parent_indices, sub_layer_assignments


def _grid_dimensions(
    components: list[PackedComponent],
    cols: int,
    gap: float,
) -> tuple[float, float]:
    """Measure the outer bounding box for one row-major packing choice.

    Parameters
    ----------
    components : list[PackedComponent]
        Packed component descriptors ``(parent_indices, local_pos, width, height)``.
    cols : int
        Number of columns to place per row.
    gap : float
        Gap inserted between adjacent component boxes.

    Returns
    -------
    tuple[float, float]
        ``(bbox_width, bbox_height)`` for the candidate packing.
    """
    if not components:
        return 0.0, 0.0

    max_width = 0.0
    total_height = 0.0
    row_width = 0.0
    row_height = 0.0
    for index, (_, _, width, height) in enumerate(components):
        if index > 0 and index % cols == 0:
            max_width = max(max_width, max(row_width - gap, 0.0))
            total_height += row_height + gap
            row_width = 0.0
            row_height = 0.0
        row_width += width + gap
        row_height = max(row_height, height)

    max_width = max(max_width, max(row_width - gap, 0.0))
    total_height += row_height
    return max_width, total_height


def _choose_component_grid(
    components: list[PackedComponent],
    gap: float,
) -> int:
    """Choose the row-major column count that best approximates square packing.

    Parameters
    ----------
    components : list[PackedComponent]
        Packed component descriptors ``(parent_indices, local_pos, width, height)``.
    gap : float
        Gap inserted between adjacent component boxes.

    Returns
    -------
    int
        Selected column count in ``[1, len(components)]``.
    """
    if not components:
        return 1

    base_area = sum(max(width, 1.0) * max(height, 1.0) for _, _, width, height in components)
    best_cols = 1
    best_score = float("inf")
    for cols in range(1, len(components) + 1):
        bbox_width, bbox_height = _grid_dimensions(components, cols=cols, gap=gap)
        if bbox_width <= 0.0 or bbox_height <= 0.0:
            return cols
        aspect = bbox_width / max(bbox_height, 1.0e-6)
        area_penalty = (bbox_width * bbox_height) / max(base_area, 1.0)
        score = abs(math.log(aspect / _COMPONENT_PACK_TARGET_ASPECT)) + (
            _COMPONENT_PACK_AREA_WEIGHT * area_penalty
        )
        if score < best_score or (math.isclose(score, best_score) and cols < best_cols):
            best_score = score
            best_cols = cols
    return best_cols


def _row_major_offsets(
    components: list[PackedComponent],
    cols: int,
    gap: float,
) -> list[tuple[float, float]]:
    """Compute row-major tile offsets for the chosen packing grid.

    Parameters
    ----------
    components : list[PackedComponent]
        Packed component descriptors ``(parent_indices, local_pos, width, height)``.
    cols : int
        Number of columns to place per row.
    gap : float
        Gap inserted between adjacent component boxes.

    Returns
    -------
    list[tuple[float, float]]
        Per-component ``(x_offset, y_offset)`` translations.
    """
    offsets: list[tuple[float, float]] = []
    x_cursor = 0.0
    y_cursor = 0.0
    row_height = 0.0
    for index, (_, _, width, height) in enumerate(components):
        if index > 0 and index % cols == 0:
            x_cursor = 0.0
            y_cursor += row_height + gap
            row_height = 0.0
        offsets.append((x_cursor, y_cursor))
        x_cursor += width + gap
        row_height = max(row_height, height)
    return offsets


def _tile_component_positions(
    component_results: list[tuple[torch.Tensor, torch.Tensor]],
    node_sep: float,
) -> torch.Tensor:
    """Tile independently solved component layouts back into parent space.

    Parameters
    ----------
    component_results : list[tuple[torch.Tensor, torch.Tensor]]
        Parent node indices and local positions for each solved component.
    node_sep : float
        Resolved node separation used to size component padding.

    Returns
    -------
    torch.Tensor
        Tiled parent position tensor with shape ``[N, 2]``.
    """
    if not component_results:
        return torch.zeros((0, 2), dtype=torch.float32)

    gap = max(float(node_sep) * _COMPONENT_TILE_PAD_FACTOR, 1.0)
    packed: list[PackedComponent] = []
    for parent_indices, pos in component_results:
        x_min = float(pos[:, 0].min().item())
        x_max = float(pos[:, 0].max().item())
        y_min = float(pos[:, 1].min().item())
        y_max = float(pos[:, 1].max().item())
        local = pos.clone()
        local[:, 0] -= x_min
        local[:, 1] -= y_min
        width = max(x_max - x_min, float(node_sep))
        height = max(y_max - y_min, float(node_sep))
        packed.append((parent_indices, local, width, height))

    packed.sort(key=lambda item: (-int(item[0].numel()), -(item[2] * item[3])))
    cols = _choose_component_grid(packed, gap=gap)
    offsets = _row_major_offsets(packed, cols=cols, gap=gap)

    total_nodes = sum(int(parent_indices.numel()) for parent_indices, _ in component_results)
    out = torch.zeros((total_nodes, 2), dtype=packed[0][1].dtype, device=packed[0][1].device)
    for (parent_indices, local, _, _), (offset_x, offset_y) in zip(packed, offsets):
        out[parent_indices, 0] = local[:, 0] + offset_x
        out[parent_indices, 1] = local[:, 1] + offset_y

    out -= out.mean(dim=0, keepdim=True)
    return out


def _score_native_result(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    *,
    is_semantically_directed: bool,
    declared_hierarchical: bool,
    all_pairs_dist: Optional[np.ndarray] = None,
) -> float:
    """Return the composite metric score for one native layout candidate.

    Parameters
    ----------
    pos : torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Graph connectivity with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.
    is_semantically_directed : bool
        Whether edge direction has domain meaning.
    declared_hierarchical : bool
        Whether the graph is both semantically directed and acyclic.
    all_pairs_dist : Optional[numpy.ndarray], optional
        Cached unweighted shortest paths with shape ``[N, N]``.

    Returns
    -------
    float
        Higher-is-better composite score for the candidate layout.

    Notes
    -----
    Seeds the torch global RNG before scoring because ``full()`` uses
    stochastic sampling (edge crossings, overlaps, stress samples). Without
    a deterministic seed, the internal best-of-k selection and any external
    re-score can disagree on candidate ranking; see the sprint-20e
    ``test_multi_start_k_three_scores_at_least_single`` regression for the
    motivating case.
    """
    from dagua.metrics import composite_auto, full

    torch.manual_seed(0)
    numeric = full(
        pos,
        edge_index,
        node_sizes=node_sizes,
        all_pairs_dist=all_pairs_dist,
    )
    numeric["declared_hierarchical"] = declared_hierarchical
    return float(composite_auto(numeric, is_semantically_directed))


def build_dagua_pipeline(config: LayoutConfig) -> Pipeline:
    """Build the composable native-engine pipeline from a resolved config.

    Parameters
    ----------
    config : LayoutConfig
        Layout configuration annotated with resolved private pipeline
        metadata (``_dagua_native_*`` attrs set by
        :func:`dagua.layout.resolve.prepare_pipeline_config`).

    Returns
    -------
    Pipeline
        Native-engine pipeline composed purely from registered ops.
    """
    resolved_steps = int(
        getattr(config, "_dagua_native_steps", config.steps if config.steps > 0 else 0),
    )
    resolved_node_sep = float(getattr(config, "_dagua_native_node_sep", config.node_sep))
    resolved_rank_sep = float(getattr(config, "_dagua_native_rank_sep", config.rank_sep))
    resolved_device = str(getattr(config, "_dagua_native_device", config.device))
    resolved_verbose = bool(getattr(config, "_dagua_native_verbose", config.verbose))
    resolved_use_dummy_nodes = bool(getattr(config, "_dagua_native_use_dummy_nodes", False))
    overlap_interval = int(getattr(config, "_dagua_native_overlap_interval", 5))
    final_projection_iterations = int(
        getattr(config, "_dagua_native_final_projection_iterations", 10),
    )
    stall_limit = int(getattr(config, "_dagua_native_stall_limit", 5))
    rel_threshold = float(getattr(config, "_dagua_native_rel_threshold", 1.0e-4))
    time_budget_s = getattr(config, "_dagua_native_time_budget_s", None)
    optimizer_type = str(getattr(config, "_dagua_native_optimizer_type", "adam"))
    losses = build_loss_ops(
        config=config,
        node_sep=resolved_node_sep,
        rank_sep=resolved_rank_sep,
    )
    weight_config = InitAnnealingScheduleConfig(
        w_dag=config.w_dag,
        w_attract=config.w_attract,
        w_repel=config.w_repel,
        w_overlap=config.w_overlap,
        w_cluster=config.w_cluster,
        w_cluster_contain=config.w_cluster_contain,
        w_crossing=config.w_crossing,
        w_straightness=config.w_straightness,
        w_length_variance=config.w_length_variance,
        w_spacing=config.w_spacing,
        w_fanout=config.w_fanout,
        w_back_edge=config.w_back_edge,
        w_stress=getattr(config, "w_stress", 0.0),
    )
    structure = getattr(config, "_dagua_native_structure", None) or getattr(
        config,
        "structure",
        None,
    )
    is_acyclic = (
        bool(getattr(structure, "is_directed_acyclic", getattr(structure, "is_acyclic", True)))
        if structure is not None
        else True
    )
    layer_assignments = getattr(config, "_dagua_native_layer_assignments", None)
    enable_native_median_transpose = _should_use_native_median_transpose(
        config=config,
        is_acyclic=is_acyclic,
    )
    native_median_passes = int(getattr(config, "native_median_passes", 4))
    native_transpose_passes = int(getattr(config, "native_transpose_passes", 8))
    enable_brandes_koepf_refine = _should_apply_brandes_koepf_refine(
        config=config,
        structure=structure,
        layer_assignments=layer_assignments,
    )
    crossing_reduction_ops: list[Any] = [
        BarycenterReorder(BarycenterReorderConfig()),
    ]
    if enable_native_median_transpose:
        crossing_reduction_ops.extend(
            [
                MedianSweep(MedianSweepConfig(passes=native_median_passes)),
                TransposeHeuristic(TransposeHeuristicConfig(passes=native_transpose_passes)),
            ]
        )
    crossing_reduction_ops.append(
        ClusterContiguousOrder(
            ClusterContiguousOrderConfig(
                enabled=bool(getattr(config, "cluster_aware_x_compaction", True))
            )
        )
    )
    crossing_reduction_ops.append(
        BrandesKoepfHorizontalRefine(
            BrandesKoepfHorizontalRefineConfig(
                node_sep=resolved_node_sep,
                enabled=enable_brandes_koepf_refine,
                structure=structure,
            )
        )
    )
    crossing_reduction_ops.extend(
        [
            ClusterAwareXCompaction(
                ClusterAwareXCompactionConfig(
                    enabled=bool(getattr(config, "cluster_aware_x_compaction", True)),
                    node_sep=resolved_node_sep,
                    cluster_gap_multiplier=0.75,
                    min_clusters=1,
                    min_long_edge_fraction=0.25,
                )
            ),
            RankRowSnap(
                RankRowSnapConfig(
                    enabled=bool(getattr(config, "layered_rank_row_snap", True)),
                    is_acyclic=is_acyclic,
                    min_layers=10,
                )
            ),
        ]
    )
    if time_budget_s is not None:
        crossing_reduction_ops = []
    # Sprint 2: branch on N. V-cycle above threshold; flat below.
    use_vcycle = bool(getattr(config, "_dagua_native_use_vcycle", False))
    if use_vcycle:
        refine_factory = _build_refine_pipeline_factory(
            losses=losses,
            overlap_interval=overlap_interval,
            stall_limit=stall_limit,
            rel_threshold=rel_threshold,
            resolved_node_sep=resolved_node_sep,
            resolved_rank_sep=resolved_rank_sep,
            resolved_device=resolved_device,
            resolved_verbose=resolved_verbose,
            weight_config=weight_config,
            optimizer_type=optimizer_type,
            lr=config.lr,
            time_budget_s=time_budget_s,
        )
        coarse_init_factory = _build_coarse_init_pipeline_factory(
            resolved_node_sep=resolved_node_sep,
            resolved_rank_sep=resolved_rank_sep,
            resolved_device=resolved_device,
            resolved_verbose=resolved_verbose,
            optimizer_type=optimizer_type,
            lr=config.lr,
            weight_config=weight_config,
        )
        return Pipeline(
            [
                FixedSteps(FixedStepsConfig(n=resolved_steps)),
                # Sprint 2 V-cycle path: coarsen via HeavyEdgeMatching, then
                # init on coarsest level and prolong+refine through hierarchy.
                # Triggered when num_nodes >= config.multilevel_threshold.
                HeavyEdgeMatching(),
                VCycleRefine(
                    coarse_init_pipeline=coarse_init_factory,
                    refine_pipeline_factory=refine_factory,
                    config=VCycleRefineConfig(
                        coarse_steps=max(resolved_steps, 200),
                        finest_steps=max(resolved_steps // 3, 40),
                        jitter_scale=0.05,
                        min_hierarchy_levels=1,
                    ),
                ),
                OverlapProjection(
                    OverlapProjectionConfig(
                        padding=2.0,
                        iterations=final_projection_iterations,
                    ),
                ),
            ],
            name="dagua_native_pipeline_vcycle",
        )

    return Pipeline(
        [
            FixedSteps(FixedStepsConfig(n=resolved_steps)),
            # Sprint 1 attempted FamilyConditionalInit with spectral fallback
            # for flat graphs; iteration on the held-out 30-graph suite showed
            # widespread regressions (sparse_layered 64->30, grid_square 89->45,
            # directed_dag_medium 50->29, bipartite 45->34) because the
            # layer-ratio heuristic fires too aggressively. Spectral init
            # works for large undirected graphs but not for the mixed DAG-ish
            # suite we have. Reverted to plain NativeEngineInit; the
            # FamilyConditionalInit op is registered but unused by default.
            # Revisit in Sprint 1 follow-up with a stricter family predicate.
            NativeEngineInit(
                NativeEngineInitConfig(
                    node_sep=resolved_node_sep,
                    rank_sep=resolved_rank_sep,
                    device=resolved_device,
                    verbose=resolved_verbose,
                    layer_assignments=getattr(config, "_dagua_native_layer_assignments", None),
                    prebuilt_layer_index=getattr(
                        config,
                        "_dagua_native_prebuilt_layer_index",
                        None,
                    ),
                ),
            ),
            # Sprint 17: cyclic-graph 2D init fallback. NativeEngineInit
            # uses longest-path layering, which collapses cyclic graphs
            # (small_world, social-net) to a single layer (all y=0). The
            # downstream gradient pipeline can't recover -- spring +
            # repulsion losses operate on a 1D-collapsed initial state.
            # Force2DInitIfFlat detects num_layers <= 1 and randomizes y
            # to give the optimizer 2D space to work with from step 1.
            # Acyclic graphs are unaffected (num_layers >= 2 is the
            # common case).
            Force2DInitIfFlat(Force2DInitIfFlatConfig()),
            *(
                [
                    InsertDummyNodes(),
                    ActivateExpandedGraphState(),
                ]
                if resolved_use_dummy_nodes
                else []
            ),
            # Sprint 15: pivot-stress pre-prep. When w_stress > 0, build
            # adjacency + select pivots + query BFS distances so the
            # PivotApproxStressLoss (added to losses by
            # build_loss_ops) has state.pivot_indices +
            # state.pivot_distances populated. Happens once per layout
            # call; the pivot cache is reused across every gradient step.
            *_stress_pivot_prep(config),
            InitAnnealingSchedule(weight_config),
            CreateOptimizer(
                CreateOptimizerConfig(
                    optimizer_type=optimizer_type,
                    lr=config.lr,
                    target="pos",
                    key="default",
                ),
            ),
            # Sprint 1: named gradient_core sub-pipeline extracted so
            # Sprint 2+ can swap initializers without touching the loop.
            build_gradient_core(
                losses=losses,
                steps=resolved_steps,
                overlap_interval=overlap_interval,
                stall_limit=stall_limit,
                rel_threshold=rel_threshold,
                time_budget_s=time_budget_s,
            ),
            # Sprint 10: barycenter crossing-minimization polish.
            # Runs after gradient_core / V-cycle so continuous layout
            # sets y-coordinates + a first-pass x ordering, then this
            # op reorders within-layer x-positions by barycenter of
            # adjacent-layer neighbours. Preserves y (DAG direction)
            # and overlap (permutes the same set of x's).
            *crossing_reduction_ops,
            OverlapProjection(
                OverlapProjectionConfig(
                    padding=2.0,
                    iterations=final_projection_iterations,
                ),
            ),
            StripDummyNodes(),
            # Sprint 13: rescale bbox to a sane aspect ratio. Dagua
            # loses heavily on aspect_ratio_deviation because the
            # gradient has no explicit AR term; layouts grow wherever
            # repulsion pushes. This op is a uniform-per-axis rescale
            # (preserves overlap-free property).
            AspectRatioFit(
                AspectRatioFitConfig(
                    target_aspect=getattr(config, "_dagua_native_target_aspect", None),
                ),
            ),
            # Sprint 18: cluster-centroid grid arrangement. Fires
            # only when problem.clusters is populated AND the
            # current aspect is degenerate (cluster columns stacked
            # vertically). Re-positions cluster centroids on a
            # roughly-square grid; intra-cluster geometry preserved.
            # Addresses clustered_deep 77.50 ceiling where 6
            # clusters of 16 nodes ended up at x=0.
            ClusterGridArrange(ClusterGridArrangeConfig()),
        ],
        name="dagua_native_pipeline",
    )


def layout_dagua_native_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: torch.Tensor,
    config: Optional[LayoutConfig] = None,
    device: Optional[str] = None,
    optimizer_type: str = "adam",
    init_pos: Optional[torch.Tensor] = None,
    clusters: Optional[dict[str, Any]] = None,
    cluster_parents: Optional[dict[str, str]] = None,
    layer_assignments: Optional[torch.Tensor] = None,
    prebuilt_layer_index: Optional[Any] = None,
    graph_structure: Optional[GraphStructure] = None,
    skip_classification: bool = False,
    seed: Optional[int] = None,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the native-engine pipeline as a drop-in tensor layout entrypoint.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity with shape ``[2, E]``.
    num_nodes : int
        Number of nodes in the graph.
    node_sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.
    config : LayoutConfig, optional
        User-facing layout configuration.
    device : str, optional
        Target execution device override.
    optimizer_type : str, default="adam"
        Optimizer implementation used by the native gradient core.
    init_pos : torch.Tensor, optional
        Optional initial positions with shape ``[N, 2]``.
    clusters : dict[str, Any], optional
        Cluster membership metadata.
    cluster_parents : dict[str, str], optional
        Parent mapping for nested clusters.
    layer_assignments : torch.Tensor, optional
        Optional layer assignments with shape ``[N]``.
    prebuilt_layer_index : Any, optional
        Pre-computed layer index for the current graph.
    graph_structure : GraphStructure, optional
        Pre-classified graph metadata.
    skip_classification : bool, default=False
        Whether to skip graph-family classification during config prep.
    seed : int, optional
        Seed override forwarded from the layout dispatcher.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Detached position tensor with shape ``[N, 2]``.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")

    effective_config = copy.copy(config) if config is not None else LayoutConfig()
    quality_budgets = resolve_quality_budgets(
        float(getattr(effective_config, "quality", 0.5)),
        num_nodes=num_nodes,
    )
    if (
        int(getattr(effective_config, "multi_start_k", 1)) == 1
        and not bool(getattr(effective_config, "_dagua_native_multi_start_resolved", False))
        and getattr(effective_config, "time_budget_s", None) is None
    ):
        effective_config.multi_start_k = quality_budgets.multi_start_k
        setattr(effective_config, "_dagua_native_multi_start_resolved", True)

    # Sprint-20d: stress route for degenerate-layering cyclic graphs.
    # Small-world / dense-cyclic graphs have no acyclic skeleton, so the
    # layered pipeline collapses to one-per-layer and wrecks every metric
    # that assumes a layer axis. Detect: original-graph has cycles +
    # post-cycle-reversal layering is fully degenerate. Route to
    # stress-majorization, then post-scale to real node-separation.
    if (
        getattr(effective_config, "route_flat_to_stress", True)
        and getattr(effective_config, "algorithm", None) in (None, "dagua_native")
        and num_nodes >= 20
        and edge_index is not None
        and edge_index.numel() > 0
    ):
        try:
            from dagua.layout.cycle import detect_back_edges, make_acyclic_robust
            from dagua.utils import longest_path_layering

            if bool(detect_back_edges(edge_index, num_nodes).any().item()):
                self_loop_mask = edge_index[0] != edge_index[1]
                filtered = edge_index[:, self_loop_mask]
                if filtered.shape[1] > 0:
                    acyclic_edges, _ = make_acyclic_robust(filtered, num_nodes)
                    layers = longest_path_layering(acyclic_edges, num_nodes)
                    layer_seq = layers if isinstance(layers, list) else layers.tolist()
                    unique = set(layer_seq)
                    if len(unique) == num_nodes and max(layer_seq.count(v) for v in unique) == 1:
                        from dagua.layout.ops.pipelines.stress_sgd import (
                            layout_stress_sgd_pipeline,
                        )

                        stress_seed = seed if seed is not None else effective_config.seed
                        if stress_seed is None:
                            stress_seed = 42
                        stress_pos = cast(
                            torch.Tensor,
                            layout_stress_sgd_pipeline(
                                edge_index=edge_index,
                                num_nodes=num_nodes,
                                node_sizes=node_sizes,
                                seed=int(stress_seed),
                            ),
                        )
                        if stress_pos.shape[0] > 1:
                            mean_w = (
                                float(node_sizes[:, 0].mean().item())
                                if node_sizes is not None
                                else 60.0
                            )
                            target = max(mean_w * 1.3, 1.0)
                            centered = stress_pos - stress_pos.mean(dim=0, keepdim=True)
                            diffs = centered.unsqueeze(0) - centered.unsqueeze(1)
                            dists = diffs.pow(2).sum(-1).sqrt()
                            n = centered.shape[0]
                            mask = ~torch.eye(n, dtype=torch.bool, device=dists.device)
                            if mask.any():
                                current_min = float(dists[mask].min().item())
                                if current_min > 1e-6:
                                    stress_pos = centered * (target / current_min)
                        return stress_pos
        except Exception:
            # Stress route is best-effort; fall through to the layered path.
            pass

    multi_start_k = int(getattr(effective_config, "multi_start_k", 1))
    if multi_start_k > 1:
        contest_structure = graph_structure or classify_graph(edge_index, num_nodes)
        is_semantically_directed = bool(
            getattr(contest_structure, "is_semantically_directed", True)
        )
        declared_hierarchical = is_semantically_directed and bool(
            getattr(
                contest_structure,
                "is_directed_acyclic",
                getattr(contest_structure, "is_acyclic", True),
            )
        )
        seed_base = seed if seed is not None else effective_config.seed
        if seed_base is None:
            seed_base = 42
        best_pos: Optional[torch.Tensor] = None
        best_score = float("-inf")
        for seed_offset in range(multi_start_k):
            candidate_seed = int(seed_base) + seed_offset
            candidate_config = copy.copy(effective_config)
            candidate_config.seed = candidate_seed
            candidate_config.multi_start_k = 1
            setattr(candidate_config, "_dagua_native_multi_start_resolved", True)
            candidate_pos = layout_dagua_native_pipeline(
                edge_index=edge_index,
                num_nodes=num_nodes,
                node_sizes=node_sizes,
                config=candidate_config,
                device=device,
                optimizer_type=optimizer_type,
                init_pos=init_pos,
                clusters=clusters,
                cluster_parents=cluster_parents,
                layer_assignments=layer_assignments,
                prebuilt_layer_index=prebuilt_layer_index,
                graph_structure=graph_structure,
                skip_classification=skip_classification,
                seed=candidate_seed,
                edge_weights=edge_weights,
            )
            candidate_score = _score_native_result(
                pos=candidate_pos,
                edge_index=edge_index,
                node_sizes=node_sizes,
                is_semantically_directed=is_semantically_directed,
                declared_hierarchical=declared_hierarchical,
            )
            if candidate_score > best_score:
                best_score = candidate_score
                best_pos = candidate_pos
        if best_pos is None:
            raise RuntimeError("dagua_native multi-start did not produce candidate positions.")
        return best_pos

    requested_device = device or effective_config.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        requested_device = "cpu"
    target_device = torch.device(requested_device)
    if num_nodes == 0:
        return torch.zeros((0, 2), dtype=torch.float32, device=target_device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32, device=target_device)

    normalized_node_sizes = normalize_node_sizes(node_sizes=node_sizes, device=target_device)
    prepared_edge_index = edge_index.to(device=target_device, dtype=torch.long)
    prepared_init_pos = (
        init_pos.to(device=target_device, dtype=torch.float32) if init_pos is not None else None
    )
    prepared_edge_weights = (
        edge_weights.to(device=target_device, dtype=torch.float32)
        if edge_weights is not None
        else None
    )
    prepared_layer_assignments = (
        layer_assignments.to(device=target_device, dtype=torch.long)
        if layer_assignments is not None
        else None
    )
    resolved_seed = seed if seed is not None else effective_config.seed
    if resolved_seed is not None:
        torch.manual_seed(int(resolved_seed))
        if target_device.type == "cuda":
            torch.cuda.manual_seed(int(resolved_seed))

    prepared_config = _prepare_native_config(
        config=effective_config,
        num_nodes=num_nodes,
        edge_index=prepared_edge_index,
        device=str(target_device),
        optimizer_type=optimizer_type,
        layer_assignments=prepared_layer_assignments,
        prebuilt_layer_index=prebuilt_layer_index,
        graph_structure=graph_structure,
        skip_classification=skip_classification,
    )
    flex_constraints = build_flex_constraints(
        config=prepared_config,
        num_nodes=num_nodes,
        device=target_device,
    )
    problem = LayoutProblem(
        edge_index=prepared_edge_index,
        num_nodes=num_nodes,
        node_sizes=normalized_node_sizes,
        direction=prepared_config.direction,
        clusters=clusters,
        cluster_parents=cluster_parents,
        structure=getattr(prepared_config, "_dagua_native_structure", None),
        flex=flex_constraints,
        edge_weights=prepared_edge_weights,
        seed=int(resolved_seed if resolved_seed is not None else 42),
    )
    state = SolveState(pos=prepared_init_pos)
    ctx = RuntimeContext(
        plan=ExecutionPlan(
            device=str(target_device),
            optimizer_type=optimizer_type,
        ),
    )
    component_ids: Optional[torch.Tensor] = None
    if (
        getattr(prepared_config, "decompose_components", True)
        and num_nodes >= 2
        and not problem.clusters
        and not _has_pins(problem.flex)
    ):
        component_state = DetectComponents().apply(problem, SolveState(), ctx)
        component_ids = component_state.component_ids

    if _should_decompose_components(problem, prepared_config, component_ids):
        component_results: list[tuple[torch.Tensor, torch.Tensor]] = []
        parent_layers = getattr(prepared_config, "_dagua_native_layer_assignments", None)
        assert component_ids is not None
        for component_id in torch.unique(component_ids, sorted=True).tolist():
            component_nodes = torch.nonzero(
                component_ids == component_id,
                as_tuple=False,
            ).squeeze(1)
            child_problem, child_state, parent_indices, child_layers = _extract_component_problem(
                problem,
                state,
                component_nodes,
                layer_assignments=parent_layers,
            )
            if child_problem.num_nodes <= 1:
                child_pos = torch.zeros(
                    (child_problem.num_nodes, 2),
                    dtype=torch.float32,
                    device=target_device,
                )
            else:
                child_config = _prepare_native_config(
                    config=effective_config,
                    num_nodes=child_problem.num_nodes,
                    edge_index=child_problem.edge_index,
                    device=str(target_device),
                    optimizer_type=optimizer_type,
                    layer_assignments=child_layers,
                    prebuilt_layer_index=None,
                    graph_structure=cast(Optional[GraphStructure], child_problem.structure),
                    skip_classification=False,
                )
                child_pos = _run_native_problem(child_problem, child_state, ctx, child_config)
            component_results.append((parent_indices, child_pos))

        tiled_positions = _tile_component_positions(
            component_results,
            node_sep=float(
                getattr(prepared_config, "_dagua_native_node_sep", prepared_config.node_sep)
            ),
        )
        outer_state = AspectRatioFit(AspectRatioFitConfig()).apply(
            problem,
            SolveState(pos=tiled_positions),
            ctx,
        )
        if outer_state.pos is None:
            raise RuntimeError("dagua_native component tiling did not produce positions.")
        return outer_state.pos.detach()

    return _run_native_problem(problem, state, ctx, prepared_config)


__all__ = ["build_dagua_pipeline", "build_gradient_core", "layout_dagua_native_pipeline"]
