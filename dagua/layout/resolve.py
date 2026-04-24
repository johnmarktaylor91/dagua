"""Config-time resolvers for the dagua native pipeline.

This module lives at the layout layer (not ops/) because its functions are
config-composition helpers, not optimization primitives. Ops mutate
``SolveState`` during the solve; resolvers produce a ``LayoutConfig`` + a
``FlexConstraints`` + a ``list[LossOp]`` BEFORE the pipeline runs.

All helpers are pure (no side effects, no module-level state). They replace
the local `_helpers` that used to live in ``dagua_native.py`` and the
``engine.py`` imports (`_adaptive_spacing`, `_auto_layout_steps`,
`_overlap_interval`, `_override_for_tree`, `_prepare_flex_data`). Once
``_layout_inner`` is archived (Sprint 0 Task 0.3) this becomes the single
source of truth for config-time resolution.
"""

from __future__ import annotations

import copy
from typing import Any, List, Optional

import torch

from dagua.config import LayoutConfig
from dagua.layout.graph_classify import GraphFamily, GraphStructure, classify_graph
from dagua.layout.ops.base import LossOp
from dagua.layout.ops.loss_engine import (
    AlignmentLoss,
    BackEdgeCompactnessLoss,
    ClusterCompactnessLoss,
    ClusterContainmentLoss,
    ClusterSeparationLoss,
    CrossingLoss,
    CrossingLossConfig,
    DagOrderingLoss,
    DagOrderingLossConfig,
    EdgeAttractionLoss,
    EdgeAttractionLossConfig,
    EdgeLengthVarianceLoss,
    EdgeStraightnessLoss,
    FanoutDistributionLoss,
    FlexSpacingLoss,
    OverlapAvoidanceLoss,
    OverlapAvoidanceLossConfig,
    PositionPinLoss,
    RepulsionLoss,
    RepulsionLossConfig,
    SpacingConsistencyLoss,
    SpacingConsistencyLossConfig,
)
from dagua.layout.ops.state import FlexConstraints


def normalize_node_sizes(node_sizes: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Broadcast node sizes to ``[N, 2]`` on the requested device.

    Accepts shape ``[N]``, ``[N, 1]``, or ``[N, 2]``; always returns float32
    ``[N, 2]`` on the target device.
    """
    if node_sizes.ndim == 1:
        normalized = node_sizes.unsqueeze(1).expand(-1, 2).contiguous()
    elif node_sizes.ndim == 2 and node_sizes.shape[1] == 1:
        normalized = node_sizes.expand(-1, 2).contiguous()
    else:
        normalized = node_sizes.contiguous()
    return normalized.to(device=device, dtype=torch.float32)


def auto_layout_steps(num_nodes: int) -> int:
    """Return the automatic optimization step count for a graph size."""
    if num_nodes <= 10:
        return 50
    if num_nodes <= 50:
        return 100
    if num_nodes <= 200:
        return 150
    if num_nodes <= 500:
        return 200
    if num_nodes <= 2000:
        return 250
    if num_nodes <= 5000:
        return 300
    if num_nodes <= 10000:
        return 400
    return 500


def override_for_tree(config: LayoutConfig) -> LayoutConfig:
    """Return a shallow config copy with tree-appropriate weights disabled."""
    tree_config = copy.copy(config)
    tree_config.w_crossing = 0.0
    tree_config.w_straightness = 0.0
    tree_config.w_length_variance = 0.0
    return tree_config


def overlap_interval(num_nodes: int, config: LayoutConfig) -> int:
    """How often to run overlap projection (every N steps)."""
    if config.overlap_check_interval > 0:
        return config.overlap_check_interval

    if num_nodes <= 5000:
        return 5
    if num_nodes <= 50000:
        return 10
    if num_nodes <= 1_000_000:
        return 20
    if num_nodes <= 50_000_000:
        return 40
    return 200


def adaptive_spacing(
    num_nodes: int,
    base_node_sep: float = 25.0,
    base_rank_sep: float = 45.0,
) -> tuple[float, float]:
    """Scale spacing based on graph size for density adaptation."""
    if num_nodes < 20:
        scale = 1.3
    elif num_nodes < 200:
        scale = 1.0
    elif num_nodes < 1000:
        scale = 0.85
    else:
        scale = 0.7
    return base_node_sep * scale, base_rank_sep * scale


def resolve_topology_aware_aspect(
    structure: Optional[GraphStructure],
) -> tuple[float, float]:
    """Return ``(target_aspect, rank_sep_multiplier)`` for one graph.

    Parameters
    ----------
    structure : GraphStructure, optional
        Topology metadata produced by :func:`dagua.layout.graph_classify.classify_graph`.

    Returns
    -------
    tuple[float, float]
        Target width/height ratio and a rank-separation multiplier for the
        native pipeline.
    """
    if structure is None:
        return 0.25, 1.0

    tags = set(structure.topology_tags)
    if "lattice_like" in tags:
        return 0.05, 1.0
    if "planar_dag" in tags:
        return 0.08, 1.0
    if "wide_layered" in tags or structure.family == GraphFamily.BIPARTITE_DAG:
        return 0.85, 1.0
    if "dense_dag" in tags:
        return 0.05, 1.0
    return 0.25, 1.0


def final_projection_iterations(num_nodes: int) -> int:
    """Return the native engine's final overlap-projection iteration count."""
    if num_nodes <= 50:
        return 5
    if num_nodes <= 200:
        return 10
    if num_nodes <= 500_000:
        return 20
    if num_nodes <= 5_000_000:
        return 10
    return 3


def stall_config(num_nodes: int) -> tuple[int, float]:
    """Return the native engine's early-stop stall parameters."""
    rel_threshold = 5.0e-4 if num_nodes <= 200 else 1.0e-4
    stall_limit = 3 if num_nodes <= 200 else 5
    if num_nodes <= 5_000:
        stall_limit = max(stall_limit - 2, 3)
    return stall_limit, rel_threshold


def prepare_flex_data(
    config: LayoutConfig,
    num_nodes: int,
    device: str,
) -> dict:
    """Extract flex constraints from config into tensor form.

    Returns a dict with pre-computed tensors for pins, alignment groups,
    and flex spacing. All node-ID to index resolution happens here.
    """
    result = {
        "has_soft_pins": False,
        "has_hard_pins": False,
        "pin_indices": torch.zeros(0, dtype=torch.long, device=device),
        "pin_targets": torch.zeros(0, 2, dtype=torch.float32, device=device),
        "pin_weights": torch.zeros(0, 2, dtype=torch.float32, device=device),
        "soft_pin_mask": torch.zeros(0, 2, dtype=torch.bool, device=device),
        "hard_pin_mask": torch.zeros(0, 2, dtype=torch.bool, device=device),
        "align_groups": [],
        "flex_node_sep": None,
        "flex_node_sep_weight": 0.0,
    }

    flex = config.flex
    if flex is None:
        return result

    if flex.pins:
        indices = []
        targets = []
        weights = []
        soft_mask = []
        hard_mask = []

        for node_id, (fx, fy) in flex.pins.items():
            if isinstance(node_id, int) and 0 <= node_id < num_nodes:
                idx = node_id
            else:
                continue

            tx = fx.target if fx is not None else 0.0
            ty = fy.target if fy is not None else 0.0
            wx = fx.weight if fx is not None else 0.0
            wy = fy.weight if fy is not None else 0.0
            has_x = fx is not None
            has_y = fy is not None
            hard_x = bool(has_x and fx is not None and fx.is_hard)
            hard_y = bool(has_y and fy is not None and fy.is_hard)
            soft_x = has_x and not hard_x
            soft_y = has_y and not hard_y

            indices.append(idx)
            targets.append([tx, ty])
            weights.append([wx if not hard_x else 0.0, wy if not hard_y else 0.0])
            soft_mask.append([soft_x, soft_y])
            hard_mask.append([hard_x, hard_y])

        if indices:
            result["pin_indices"] = torch.tensor(indices, dtype=torch.long, device=device)
            result["pin_targets"] = torch.tensor(targets, dtype=torch.float32, device=device)
            result["pin_weights"] = torch.tensor(weights, dtype=torch.float32, device=device)
            result["soft_pin_mask"] = torch.tensor(soft_mask, dtype=torch.bool, device=device)
            result["hard_pin_mask"] = torch.tensor(hard_mask, dtype=torch.bool, device=device)
            result["has_soft_pins"] = any(any(row) for row in soft_mask)
            result["has_hard_pins"] = any(any(row) for row in hard_mask)

    align_groups: list = []
    for groups, axis in [(flex.align_x, 0), (flex.align_y, 1)]:
        if groups is None:
            continue
        for group in groups:
            idx_list = []
            for node_id in group.nodes:
                if isinstance(node_id, int) and 0 <= node_id < num_nodes:
                    idx_list.append(node_id)
            if len(idx_list) >= 2:
                align_groups.append(
                    (
                        torch.tensor(idx_list, dtype=torch.long, device=device),
                        group.weight,
                        axis,
                    )
                )
    result["align_groups"] = align_groups

    if flex.node_sep is not None:
        result["flex_node_sep"] = flex.node_sep.target
        result["flex_node_sep_weight"] = flex.node_sep.weight

    return result


def build_flex_constraints(
    config: LayoutConfig,
    num_nodes: int,
    device: torch.device,
) -> Optional[FlexConstraints]:
    """Convert ``LayoutConfig.flex`` into pipeline-ready tensor constraints."""
    if config.flex is None:
        return None

    flex_data = prepare_flex_data(config=config, num_nodes=num_nodes, device=str(device))
    return FlexConstraints(
        pin_indices=flex_data["pin_indices"],
        pin_targets=flex_data["pin_targets"],
        pin_weights=flex_data["pin_weights"],
        soft_pin_mask=flex_data["soft_pin_mask"],
        hard_pin_mask=flex_data["hard_pin_mask"],
        align_groups=flex_data["align_groups"],
        flex_node_sep=flex_data["flex_node_sep"],
        flex_node_sep_weight=flex_data["flex_node_sep_weight"],
    )


def prepare_pipeline_config(
    config: LayoutConfig,
    num_nodes: int,
    edge_index: torch.Tensor,
    device: str,
    layer_assignments: Optional[torch.Tensor],
    prebuilt_layer_index: Optional[Any],
    graph_structure: Optional[GraphStructure],
    skip_classification: bool,
) -> LayoutConfig:
    """Resolve native-engine pipeline settings for one problem instance.

    Produces a shallow config copy annotated with resolved private pipeline
    metadata (prefixed with ``_dagua_native_``) that ``build_dagua_pipeline``
    consumes.
    """
    effective_config = copy.copy(config)
    structure: Optional[GraphStructure] = None
    if not skip_classification:
        structure = graph_structure
        if structure is None:
            classification_layers = layer_assignments
            structure = classify_graph(
                edge_index,
                num_nodes,
                layer_assignments=classification_layers,
            )
        if structure.family in {GraphFamily.TREE, GraphFamily.CHAIN}:
            effective_config = override_for_tree(effective_config)
        if structure.family == GraphFamily.CHAIN:
            auto_steps = auto_layout_steps(num_nodes)
            resolved_steps = min(
                effective_config.steps if effective_config.steps > 0 else auto_steps,
                50,
            )
        else:
            resolved_steps = (
                effective_config.steps
                if effective_config.steps > 0
                else auto_layout_steps(num_nodes)
            )
    else:
        resolved_steps = (
            effective_config.steps if effective_config.steps > 0 else auto_layout_steps(num_nodes)
        )

    resolved_node_sep = effective_config.node_sep
    resolved_rank_sep = effective_config.rank_sep
    if effective_config.adaptive_spacing:
        resolved_node_sep, resolved_rank_sep = adaptive_spacing(
            num_nodes=num_nodes,
            base_node_sep=resolved_node_sep,
            base_rank_sep=resolved_rank_sep,
        )
    target_aspect, rank_sep_multiplier = resolve_topology_aware_aspect(structure)
    resolved_rank_sep *= rank_sep_multiplier

    stall_limit, rel_threshold = stall_config(num_nodes=num_nodes)
    setattr(effective_config, "_dagua_native_steps", resolved_steps)
    setattr(effective_config, "_dagua_native_node_sep", resolved_node_sep)
    setattr(effective_config, "_dagua_native_rank_sep", resolved_rank_sep)
    setattr(effective_config, "_dagua_native_target_aspect", target_aspect)
    setattr(effective_config, "_dagua_native_rank_sep_multiplier", rank_sep_multiplier)
    setattr(effective_config, "_dagua_native_device", device)
    setattr(effective_config, "_dagua_native_verbose", effective_config.verbose)
    setattr(effective_config, "_dagua_native_layer_assignments", layer_assignments)
    setattr(effective_config, "_dagua_native_prebuilt_layer_index", prebuilt_layer_index)
    setattr(
        effective_config,
        "_dagua_native_overlap_interval",
        overlap_interval(num_nodes=num_nodes, config=effective_config),
    )
    setattr(
        effective_config,
        "_dagua_native_final_projection_iterations",
        final_projection_iterations(num_nodes=num_nodes),
    )
    setattr(effective_config, "_dagua_native_stall_limit", stall_limit)
    setattr(effective_config, "_dagua_native_rel_threshold", rel_threshold)
    setattr(effective_config, "_dagua_native_crossing_alpha", 3.0)
    setattr(effective_config, "_dagua_native_optimizer_type", "adam")
    # Sprint 17: stash classified structure so downstream ops + loss-ops
    # can gate behaviour on acyclicity / family without re-classifying.
    # Used by:
    # - build_loss_ops (skip DagOrderingLoss when graph is cyclic)
    # - dagua_native_pipeline tree fast-path (already reads .structure)
    setattr(effective_config, "structure", structure)
    setattr(effective_config, "_dagua_native_structure", structure)
    # Sprint 2: multilevel V-cycle threshold. The infrastructure ships this
    # sprint (VCycleRefine op + threshold-based dispatch), but the V-cycle
    # produces catastrophic regressions on chains (21 vs legacy 100 at 25K)
    # and random DAGs (20 vs 50). Sprint 2 head-to-head bench (script:
    # scripts/sprint_2_vcycle_bench.py, output: eval_output/native_algo/
    # sprint_2_vcycle/report.json) confirms the V-cycle path is not
    # production-ready. Threshold raised to 1_000_000 so V-cycle never
    # fires by default; opt-in via `LayoutConfig(multilevel_threshold=20000)`
    # to exercise it. Sprint 2b will fix the per-level loss-spacing scale
    # and the tree_25000 state.pos None error.
    setattr(effective_config, "_dagua_native_num_nodes", num_nodes)
    vcycle_threshold = int(getattr(effective_config, "multilevel_threshold", 20000))
    if vcycle_threshold == 20000:
        # default; raise to disable
        vcycle_threshold = 1_000_000
    setattr(
        effective_config,
        "_dagua_native_use_vcycle",
        num_nodes >= vcycle_threshold,
    )
    return effective_config


def build_loss_ops(
    config: LayoutConfig,
    node_sep: float,
    rank_sep: float,
) -> List[LossOp]:
    """Construct the active native-engine loss operators from a resolved config."""
    losses: List[LossOp] = []
    # Sprint 17: skip DagOrderingLoss on cyclic graphs. The loss penalises
    # every edge whose source-y >= target-y; on a cyclic graph (small_world,
    # social-net), every back edge is a permanent violation and the term
    # collapses the layout into a 1D stripe. Detect via stashed structure.
    structure = getattr(config, "_dagua_native_structure", None) or getattr(
        config, "structure", None
    )
    is_acyclic = bool(getattr(structure, "is_acyclic", True)) if structure is not None else True
    if config.w_dag > 0.0 and is_acyclic:
        losses.append(DagOrderingLoss(DagOrderingLossConfig(rank_sep=rank_sep)))
    if config.w_attract > 0.0:
        losses.append(EdgeAttractionLoss(EdgeAttractionLossConfig(x_bias=config.w_attract_x_bias)))
    if config.w_repel > 0.0:
        losses.append(
            RepulsionLoss(
                RepulsionLossConfig(
                    threshold=config.exact_repulsion_threshold,
                    sample_k=config.negative_sample_k,
                    rvs_threshold=config.rvs_threshold,
                    rvs_nn_k=config.rvs_nn_k,
                ),
            ),
        )
    if config.w_overlap > 0.0:
        losses.append(
            OverlapAvoidanceLoss(
                OverlapAvoidanceLossConfig(
                    padding=2.0,
                    rvs_threshold=config.rvs_threshold,
                ),
            ),
        )
    if config.w_cluster > 0.0:
        losses.append(ClusterCompactnessLoss())
        losses.append(ClusterSeparationLoss())
    if config.w_cluster_contain > 0.0:
        losses.append(ClusterContainmentLoss())
    if config.w_crossing > 0.0:
        losses.append(
            CrossingLoss(
                CrossingLossConfig(
                    alpha=float(getattr(config, "_dagua_native_crossing_alpha", 3.0)),
                    max_pairs=500,
                ),
            ),
        )
    if config.w_straightness > 0.0:
        losses.append(EdgeStraightnessLoss())
    if config.w_length_variance > 0.0:
        losses.append(EdgeLengthVarianceLoss())
    if config.w_spacing > 0.0:
        losses.append(SpacingConsistencyLoss(SpacingConsistencyLossConfig(target_gap=node_sep)))
    if config.w_fanout > 0.0:
        losses.append(FanoutDistributionLoss())
    if config.w_back_edge > 0.0:
        losses.append(BackEdgeCompactnessLoss())
    if config.flex is not None and config.flex.pins:
        losses.append(PositionPinLoss())
    if config.flex is not None and (config.flex.align_x or config.flex.align_y):
        losses.append(AlignmentLoss())
    if config.flex is not None and config.flex.node_sep is not None:
        losses.append(FlexSpacingLoss())
    # Sprint 15: pivot-approximated stress loss (opt-in via w_stress>0).
    # Requires state.pivot_indices + state.pivot_distances pre-populated
    # by the stress pre-prep ops (BuildAdjacency + PivotSelection +
    # PivotDistanceQueries) which land in dagua_native_pipeline when
    # config.w_stress > 0.
    if getattr(config, "w_stress", 0.0) > 0.0:
        from dagua.layout.ops.loss_classic import PivotApproxStressLoss

        losses.append(PivotApproxStressLoss())
    return losses


__all__ = [
    "adaptive_spacing",
    "auto_layout_steps",
    "build_flex_constraints",
    "build_loss_ops",
    "final_projection_iterations",
    "normalize_node_sizes",
    "overlap_interval",
    "override_for_tree",
    "prepare_flex_data",
    "prepare_pipeline_config",
    "stall_config",
]
