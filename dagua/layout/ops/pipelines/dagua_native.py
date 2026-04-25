"""Topology-dispatched adapter for dagua's native tensor layout engine."""

from __future__ import annotations

import copy
from typing import Any, Optional

import torch

from dagua.config import LayoutConfig
from dagua.layout.graph_classify import GraphFamily, GraphStructure, classify_graph
from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.pipelines import dagua_native_legacy
from dagua.layout.ops.pipelines._native_shared import (
    _prepare_native_config,
    _should_apply_brandes_koepf_refine,
    _should_decompose_components,
    _should_use_native_dummy_nodes,
    _should_use_native_median_transpose,
    _tile_component_positions,
    build_gradient_core,
)
from dagua.layout.ops.pipelines.native_force_directed import (
    build_native_force_directed_pipeline,
    layout_native_force_directed_pipeline,
)
from dagua.layout.ops.pipelines.native_hybrid import build_native_hybrid_pipeline
from dagua.layout.ops.pipelines.native_layered_dag import build_native_layered_dag_pipeline
from dagua.layout.ops.pipelines.native_planar import (
    PlanarityFailure,
    build_native_planar_pipeline,
)
from dagua.layout.ops.pipelines.native_tree import build_native_tree_pipeline
from dagua.layout.ops.postprocess import AspectRatioFit, AspectRatioFitConfig
from dagua.layout.ops.preprocess import DetectComponents
from dagua.layout.ops.state import (
    ExecutionPlan,
    FlexConstraints,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)
from dagua.layout.resolve import build_flex_constraints, normalize_node_sizes

_COMPONENT_DOMINANCE_SKIP_FRACTION = 0.85


def _selected_force_pipeline(config: LayoutConfig) -> Optional[str]:
    """Return the user-selected native sub-pipeline override.

    Parameters
    ----------
    config : LayoutConfig
        Layout configuration.

    Returns
    -------
    str | None
        Normalized force-pipeline value.
    """
    value = getattr(config, "force_pipeline", None)
    if value is None:
        return None
    return str(value).lower()


def _choose_native_pipeline(structure: Optional[GraphStructure], config: LayoutConfig) -> str:
    """Choose a native sub-pipeline for one prepared problem.

    Parameters
    ----------
    structure : GraphStructure, optional
        Classified graph topology.
    config : LayoutConfig
        Prepared layout configuration.

    Returns
    -------
    str
        One of ``"tree"``, ``"layered_dag"``, ``"force_directed"``,
        ``"hybrid"``, or ``"legacy_monolith"``.
    """
    forced = _selected_force_pipeline(config)
    if forced in {"tree", "layered_dag", "force_directed", "hybrid", "planar", "legacy_monolith"}:
        return forced
    if structure is None:
        return "layered_dag"

    family = structure.family
    num_nodes = int(getattr(config, "_dagua_native_num_nodes", 0))
    small_tree_cutoff = int(getattr(config, "small_n_tree_cutoff", 64))
    if num_nodes <= small_tree_cutoff and family in {GraphFamily.TREE, GraphFamily.CHAIN}:
        return "tree"
    # Sprint-20g: planar dispatch when the classifier confirms exact
    # planarity AND the user has explicitly opted in via try_planar_first.
    # Default is False because the current Schnyder-init + flat-stress
    # planar pipeline drops the dag_consistency / depth_spearman bonus
    # that layered_dag earns on planar DAGs (loses 3-35 composite points
    # vs layered_dag on every benchmark candidate).
    if getattr(config, "try_planar_first", False) and bool(getattr(structure, "is_planar", False)):
        return "planar"
    cyclicity_ratio = float(getattr(structure, "cyclicity_ratio", 0.0))
    # Sprint-20g: removed auto-route to force_directed. Empirically the
    # PivotMDS+Stress force pipeline loses to layered_dag/hybrid on every
    # cyclic benchmark candidate today (2026-04-24 measurement). Users can
    # still opt in via force_pipeline="force_directed".
    if family == GraphFamily.FORCE_DIRECTED and cyclicity_ratio > 0.5:
        return "force_directed"
    if family == GraphFamily.HYBRID or cyclicity_ratio > 0.05:
        return "hybrid"
    return "layered_dag"


def build_dagua_pipeline(config: LayoutConfig) -> Pipeline:
    """Build the topology-selected native pipeline.

    Parameters
    ----------
    config : LayoutConfig
        Prepared native configuration with ``_dagua_native_*`` metadata.

    Returns
    -------
    Pipeline
        Selected native sub-pipeline.
    """
    structure = getattr(config, "_dagua_native_structure", None) or getattr(
        config,
        "structure",
        None,
    )
    selected = _choose_native_pipeline(structure=structure, config=config)
    if selected == "legacy_monolith":
        return dagua_native_legacy.build_dagua_pipeline(config)
    if selected == "tree":
        return build_native_tree_pipeline(config)
    if selected == "planar":
        return build_native_planar_pipeline(config)
    if selected == "force_directed":
        return build_native_force_directed_pipeline(config)
    if selected == "hybrid":
        return build_native_hybrid_pipeline(config)
    return build_native_layered_dag_pipeline(config)


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
        Prepared flex constraints.
    component_ids : torch.Tensor, optional
        Weak-component labels with shape ``[N]``.

    Returns
    -------
    bool
        ``True`` when an alignment group references multiple components.
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


def _should_decompose_native_components(
    problem: LayoutProblem,
    config: LayoutConfig,
    component_ids: Optional[torch.Tensor],
) -> bool:
    """Return whether native dispatch should solve weak components separately.

    Parameters
    ----------
    problem : LayoutProblem
        Prepared parent layout problem.
    config : LayoutConfig
        Prepared native configuration.
    component_ids : torch.Tensor, optional
        Weak-component labels with shape ``[N]``.

    Returns
    -------
    bool
        ``True`` when independent component solving is safe and useful.
    """
    forced = _selected_force_pipeline(config)
    if forced == "legacy_monolith":
        return _should_decompose_components(problem, config, component_ids)
    if not getattr(config, "decompose_components", True):
        return False
    if problem.num_nodes < 2 or problem.clusters or _has_pins(problem.flex):
        return False

    structure = problem.structure
    if structure is not None:
        if int(getattr(structure, "num_components", 1)) <= 1:
            return False
        if bool(getattr(structure, "has_dominant_component", False)):
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
    if _has_cross_component_flex(problem.flex, component_ids):
        return False
    return True


def _subset_flex(
    flex: Optional[FlexConstraints],
    local_index: torch.Tensor,
) -> Optional[FlexConstraints]:
    """Project parent flex constraints into component-local node ids.

    Parameters
    ----------
    flex : FlexConstraints, optional
        Parent flex constraints.
    local_index : torch.Tensor
        Parent-to-local node map with shape ``[N_parent]``.

    Returns
    -------
    FlexConstraints | None
        Child-local flex constraints.
    """
    return dagua_native_legacy._subset_flex(flex, local_index)


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
        Parent solve state.
    component_nodes : torch.Tensor
        Parent node ids in this component with shape ``[K]``.
    layer_assignments : torch.Tensor, optional
        Optional parent layer assignments with shape ``[N_parent]``.

    Returns
    -------
    tuple[LayoutProblem, SolveState, torch.Tensor, torch.Tensor | None]
        Child problem, child state, parent indices, and child layer assignments.
    """
    return dagua_native_legacy._extract_component_problem(
        parent_problem,
        parent_state,
        component_nodes,
        layer_assignments,
    )


def _run_native_problem(
    problem: LayoutProblem,
    state: SolveState,
    ctx: RuntimeContext,
    config: LayoutConfig,
) -> torch.Tensor:
    """Run the selected native sub-pipeline for one prepared problem.

    Parameters
    ----------
    problem : LayoutProblem
        Prepared layout problem for one component or full graph.
    state : SolveState
        Mutable solve state.
    ctx : RuntimeContext
        Runtime execution context.
    config : LayoutConfig
        Prepared native configuration.

    Returns
    -------
    torch.Tensor
        Final positions with shape ``[N, 2]``.
    """
    structure = problem.structure or getattr(config, "_dagua_native_structure", None)
    if structure is None:
        structure = classify_graph(problem.edge_index, problem.num_nodes)
        problem.structure = structure

    selected = _choose_native_pipeline(structure=structure, config=config)
    if selected == "legacy_monolith":
        return dagua_native_legacy._run_native_problem(problem, state, ctx, config)
    if selected == "force_directed":
        return layout_native_force_directed_pipeline(
            edge_index=problem.edge_index,
            num_nodes=problem.num_nodes,
            node_sizes=problem.node_sizes,
            config=config,
            seed=problem.seed,
            edge_weights=problem.edge_weights,
        )

    try:
        final_state = build_dagua_pipeline(config).apply(problem, state, ctx)
    except PlanarityFailure:
        if _selected_force_pipeline(config) == "planar":
            raise
        # Auto-routed to planar but validation failed at runtime (e.g.
        # disconnected components). Fall back to the standard topology
        # selection without trying planar again.
        fallback_config = copy.copy(config)
        fallback_config.try_planar_first = False
        final_state = build_dagua_pipeline(fallback_config).apply(problem, state, ctx)
        selected = _choose_native_pipeline(structure=structure, config=fallback_config)
    if final_state.pos is None:
        raise RuntimeError(f"native {selected} pipeline did not produce final positions.")
    result = final_state.pos.detach()
    if result.shape[0] > problem.num_nodes:
        result = result[: problem.num_nodes]
    return result


def _score_native_result(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
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

    Returns
    -------
    float
        Higher-is-better composite score.
    """
    return dagua_native_legacy._score_native_result(pos, edge_index, node_sizes)


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
    """Run the topology-dispatched native pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.
    config : LayoutConfig, optional
        Layout configuration.
    device : str, optional
        Target execution device.
    optimizer_type : str, default="adam"
        Optimizer implementation for gradient sub-pipelines.
    init_pos : torch.Tensor, optional
        Optional initial positions with shape ``[N, 2]``.
    clusters : dict[str, Any], optional
        Cluster membership metadata.
    cluster_parents : dict[str, str], optional
        Nested-cluster parent metadata.
    layer_assignments : torch.Tensor, optional
        Optional layer assignments with shape ``[N]``.
    prebuilt_layer_index : Any, optional
        Optional pre-built layer index.
    graph_structure : GraphStructure, optional
        Optional pre-classified topology.
    skip_classification : bool, default=False
        Whether to skip classification during config preparation.
    seed : int, optional
        RNG seed override.
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
    if _selected_force_pipeline(effective_config) == "legacy_monolith":
        return dagua_native_legacy.layout_dagua_native_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            node_sizes=node_sizes,
            config=effective_config,
            device=device,
            optimizer_type=optimizer_type,
            init_pos=init_pos,
            clusters=clusters,
            cluster_parents=cluster_parents,
            layer_assignments=layer_assignments,
            prebuilt_layer_index=prebuilt_layer_index,
            graph_structure=graph_structure,
            skip_classification=skip_classification,
            seed=seed,
            edge_weights=edge_weights,
        )

    multi_start_k = int(getattr(effective_config, "multi_start_k", 1))
    if multi_start_k > 1:
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
            candidate_score = _score_native_result(candidate_pos, edge_index, node_sizes)
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

    if _should_decompose_native_components(problem, prepared_config, component_ids):
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
                    graph_structure=child_problem.structure,
                    skip_classification=False,
                )
                # Sprint-19d component packing is a protected win; preserve
                # the child-solve behavior inside the wrapper while the new
                # flat/hybrid buckets are tuned independently.
                if _selected_force_pipeline(child_config) is None:
                    child_config.force_pipeline = "legacy_monolith"
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


__all__ = [
    "_choose_native_pipeline",
    "_prepare_native_config",
    "_run_native_problem",
    "_should_apply_brandes_koepf_refine",
    "_should_use_native_dummy_nodes",
    "_should_use_native_median_transpose",
    "build_dagua_pipeline",
    "build_gradient_core",
    "layout_dagua_native_pipeline",
]
