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
import math
from dataclasses import dataclass
from typing import Any, List, Literal, Optional

import torch

from dagua.config import LayoutConfig
from dagua.layout.aesthetics import apply_loss_multipliers, resolve_aesthetic_profile
from dagua.layout.graph_classify import GraphFamily, GraphStructure, classify_graph
from dagua.layout.ops.base import LossOp
from dagua.layout.ops.loss_engine import (
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
    RepulsionLoss,
    RepulsionLossConfig,
    SpacingConsistencyLoss,
    SpacingConsistencyLossConfig,
)
from dagua.layout.ops.state import FlexConstraints


@dataclass(frozen=True)
class QualityBudgets:
    """Resolved time-vs-quality budgets for native layout pipelines.

    Parameters
    ----------
    step_multiplier : float
        Multiplier applied to automatic step counts.
    multi_start_k : int
        Number of deterministic seeds to score in best-of-k native layout.
    stress_n_pivots : int
        Pivot count for native stress initialization, capped by graph size.
    smacof_iters : int
        Dense SMACOF polish iterations for native stress when under its node
        cutoff.
    polish_battery : {"off", "default", "full"}
        Post-pipeline polish candidate budget.
    ml_refine_multiplier : float
        Multiplier for native-stress multilevel per-level refinement rounds.
    barnes_hut_theta : float
        Barnes-Hut opening angle budget for pipelines that consume it.
    sampling_rate : float
        Relative sampling budget for approximate large-graph refinements.

    Notes
    -----
    Dagua does not currently track constructor-provided fields separately
    from dataclass defaults. Callers therefore apply these budgets only when
    a field still equals its default sentinel, such as ``steps == 0`` or an
    omitted algorithm parameter.
    """

    step_multiplier: float
    multi_start_k: int
    stress_n_pivots: int
    smacof_iters: int
    polish_battery: Literal["off", "default", "full"]
    ml_refine_multiplier: float
    barnes_hut_theta: float
    sampling_rate: float


def _interp_piecewise(value: float, points: list[tuple[float, float]]) -> float:
    """Linearly interpolate a scalar over sorted control points.

    Parameters
    ----------
    value : float
        Input value.
    points : list[tuple[float, float]]
        Sorted ``(x, y)`` control points.

    Returns
    -------
    float
        Interpolated output, clamped to the endpoint values.
    """
    if value <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if value <= x1:
            ratio = (value - x0) / (x1 - x0)
            return y0 + (y1 - y0) * ratio
    return points[-1][1]


def resolve_quality_budgets(quality: float, num_nodes: int = 0) -> QualityBudgets:
    """Resolve public quality into concrete native layout budgets.

    Parameters
    ----------
    quality : float
        Normalized quality value in ``[0, 1]``.
    num_nodes : int, default=0
        Optional graph size used to cap pivot budgets. ``0`` means no graph
        cap is available yet.

    Returns
    -------
    QualityBudgets
        Frozen budget bundle. Values are monotonic with quality; balanced
        ``0.5`` preserves current default layered budgets.

    Raises
    ------
    ValueError
        If ``quality`` lies outside ``[0, 1]``.
    """
    q = float(quality)
    if q < 0.0 or q > 1.0:
        raise ValueError("quality must be in [0, 1].")

    log_multiplier = _interp_piecewise(
        q,
        [
            (0.0, math.log(0.4)),
            (0.5, math.log(1.0)),
            (0.75, math.log(2.0)),
            (1.0, math.log(4.0)),
        ],
    )
    pivot_target = int(
        round(
            _interp_piecewise(
                q,
                [
                    (0.25, 32.0),
                    (0.5, 64.0),
                    (0.75, 128.0),
                    (1.0, 256.0),
                ],
            )
        )
    )
    if num_nodes > 0:
        pivot_target = min(max(int(num_nodes), 1), pivot_target)

    smacof_iters = int(
        round(
            _interp_piecewise(
                q,
                [
                    (0.25, 0.0),
                    (0.5, 4.0),
                    (0.75, 24.0),
                    (1.0, 50.0),
                ],
            )
        )
    )
    if q < 0.35:
        polish_battery: Literal["off", "default", "full"] = "off"
    elif q >= 0.75:
        polish_battery = "full"
    else:
        polish_battery = "default"
    ml_refine_multiplier = _interp_piecewise(
        q,
        [
            (0.25, 0.5),
            (0.5, 1.0),
            (0.75, 2.0),
            (1.0, 3.0),
        ],
    )
    return QualityBudgets(
        step_multiplier=math.exp(log_multiplier),
        multi_start_k=1 if q < 0.7 else (3 if q < 0.9 else 5),
        stress_n_pivots=max(1, pivot_target),
        smacof_iters=smacof_iters,
        polish_battery=polish_battery,
        ml_refine_multiplier=ml_refine_multiplier,
        barnes_hut_theta=_interp_piecewise(q, [(0.0, 1.4), (0.5, 1.0), (1.0, 0.6)]),
        sampling_rate=_interp_piecewise(q, [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0)]),
    )


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
        return 0.45, 1.0
    if "wide_layered" in tags or structure.family == GraphFamily.BIPARTITE_DAG:
        return 0.85, 1.0
    if "dense_dag" in tags:
        return 0.05, 1.0
    return 0.25, 1.0


def duplicate_edge_multiplicity(edge_index: torch.Tensor) -> int:
    """Return the maximum duplicate multiplicity of non-self directed edges.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.

    Returns
    -------
    int
        Maximum count for a repeated ``(source, target)`` pair. Empty graphs
        and graphs without duplicates return ``1``.
    """
    if edge_index.numel() == 0:
        return 1
    counts: dict[tuple[int, int], int] = {}
    for source, target in edge_index.detach().to(device="cpu", dtype=torch.long).t().tolist():
        if source == target:
            continue
        key = (int(source), int(target))
        counts[key] = counts.get(key, 0) + 1
    return max(counts.values(), default=1)


def cap_tiny_multiedge_rank_sep(
    config: LayoutConfig,
    edge_index: torch.Tensor,
    num_nodes: int,
    resolved_rank_sep: float,
) -> float:
    """Cap rank spacing for tiny DAGs with duplicate edges.

    Parameters
    ----------
    config : LayoutConfig
        Effective layout configuration.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    resolved_rank_sep : float
        Rank separation after adaptive and topology-aware scaling.

    Returns
    -------
    float
        Possibly capped rank separation.
    """
    if not bool(getattr(config, "tiny_multiedge_rank_sep_cap", True)):
        return resolved_rank_sep
    if bool(getattr(config, "_dagua_native_has_clusters", False)):
        return resolved_rank_sep
    if num_nodes > 10 or duplicate_edge_multiplicity(edge_index) <= 1:
        return resolved_rank_sep
    cap = float(getattr(config, "tiny_multiedge_rank_sep_max", 240.0))
    return min(resolved_rank_sep, cap)


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
    """Extract surviving scalar flex data for pipeline adapters.

    Parameters
    ----------
    config : LayoutConfig
        Layout configuration.
    num_nodes : int
        Number of graph nodes, retained for the stable helper signature.
    device : str
        Tensor device string.

    Returns
    -------
    dict
        Empty pin/alignment tensors plus scalar flex spacing. R9 constraints
        are lowered by the engine-side constraint path, not this adapter.
    """
    _ = num_nodes
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
    # r80-S8 aesthetic-priority knob: resolve ONCE per problem instance so
    # every downstream consumer (the loss-weight multipliers below AND the
    # undirected-portfolio contest's candidate scorer, which reads the
    # stashed profile back off this same prepared config) uses the
    # identical profile object -- required for contest fairness. `None`
    # (the default, unset path) is a true no-op: no config copy churn beyond
    # the one already happening here, no wrapper call on the scoring path.
    aesthetic_profile = resolve_aesthetic_profile(effective_config)
    if aesthetic_profile is not None:
        effective_config = apply_loss_multipliers(effective_config, aesthetic_profile)
    setattr(effective_config, "_dagua_native_aesthetic_profile", aesthetic_profile)
    quality_budgets = resolve_quality_budgets(
        float(getattr(effective_config, "quality", 0.5)),
        num_nodes=num_nodes,
    )
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
            if effective_config.steps > 0:
                resolved_steps = effective_config.steps
            else:
                resolved_steps = min(int(round(auto_steps * quality_budgets.step_multiplier)), 50)
        else:
            resolved_steps = (
                effective_config.steps
                if effective_config.steps > 0
                else int(round(auto_layout_steps(num_nodes) * quality_budgets.step_multiplier))
            )
    else:
        resolved_steps = (
            effective_config.steps
            if effective_config.steps > 0
            else int(round(auto_layout_steps(num_nodes) * quality_budgets.step_multiplier))
        )
    resolved_steps = max(resolved_steps, 1)

    if (
        int(getattr(effective_config, "multi_start_k", 1)) == 1
        and not bool(getattr(effective_config, "_dagua_native_multi_start_resolved", False))
        and effective_config.time_budget_s is None
    ):
        effective_config.multi_start_k = quality_budgets.multi_start_k
        setattr(effective_config, "_dagua_native_multi_start_resolved", True)

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
    resolved_rank_sep = cap_tiny_multiedge_rank_sep(
        config=effective_config,
        edge_index=edge_index,
        num_nodes=num_nodes,
        resolved_rank_sep=resolved_rank_sep,
    )

    stall_limit, rel_threshold = stall_config(num_nodes=num_nodes)
    setattr(effective_config, "_dagua_native_steps", resolved_steps)
    setattr(effective_config, "_dagua_native_quality_budgets", quality_budgets)
    setattr(effective_config, "_dagua_native_polish_battery", quality_budgets.polish_battery)
    setattr(effective_config, "_dagua_native_time_budget_s", effective_config.time_budget_s)
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
    resolved_final_projection_iterations = final_projection_iterations(num_nodes=num_nodes)
    if effective_config.time_budget_s is not None:
        resolved_final_projection_iterations = min(resolved_final_projection_iterations, 1)
    setattr(
        effective_config,
        "_dagua_native_final_projection_iterations",
        resolved_final_projection_iterations,
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
                    exact=config.exact_repulsion,
                ),
            ),
        )
    if config.w_overlap > 0.0:
        losses.append(
            OverlapAvoidanceLoss(
                OverlapAvoidanceLossConfig(
                    padding=2.0,
                    rvs_threshold=config.rvs_threshold,
                    exact=config.exact_repulsion,
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
    "QualityBudgets",
    "resolve_quality_budgets",
    "stall_config",
]
