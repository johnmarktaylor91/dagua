"""Input-gated conditional groups for the scoring-only V3 ruler.

The registry in this module is intentionally separate from ``ruler_v3.py`` so
new conditional groups can be added without changing the CORE scorer. Gates
read only declared graph metadata, never the drawing, following DOC-3 in the
joint V3 ruler design.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from dagua.eval.distributional_fidelity import pairwise_procrustes_matrix
from dagua.eval.drawing import native_route_coverage, routes_to_curves
from dagua.metrics import (
    _deterministic_sample_indices,
    _ensure_cpu,
    bend_count,
    cluster_edge_intrusion_score,
    cluster_exclusion_score,
    cluster_label_occlusion_score,
    cluster_nesting_fidelity_score,
    cluster_sibling_overlap_score,
    composite_drawing,
    directed_flow_score,
    port_angular_resolution,
    routed_crossing_rate,
)

G5_QUALITY_BAND_DELTA = 0.05
G5_CHANGE_EPSILON = 1e-9
G5_ZERO_CHANGE_SCALE = 0.05
G7_SEVERE_PORT_CAP = 0.5
G7_PORT_SIDE_COSINE_THRESHOLD = math.sqrt(0.5)
G7_ROUTE_SAMPLE_CANONICAL_DISTANCE = 1.0
G7_TERMINAL_SEPARATION_FRACTION = 0.5
G2_SLOT_A_TOTAL = 5.0
G2_SLOT_B_TOTAL = 3.0
G2_SLOT_A_DIAGNOSTIC_FACETS = frozenset({"G2_cluster_exclusion"})
G4_SLOT_A_TOTAL = 5.0
G4_SLOT_B_TOTAL = 5.0
COMPACTNESS_RATIO_PLATEAU_HI = 8.0
COMPACTNESS_LOG_DECAY = 1.0
TREE_RATIO_PLATEAU_LO = 0.5
TREE_RATIO_PLATEAU_HI = 2.0
TREE_RATIO_LOG_DECAY = 1.0

TIER_1_WEIGHT = 4.0
TIER_2_WEIGHT = 2.0
TIER_3_WEIGHT = 1.0
WEIGHT_CV_SATURATION = 1.0
NONDEGENERATE_WEIGHT_CV = 1e-9
LOCAL_WEIGHT_MONOTONICITY_NODE_BUDGET = 512
G6_WEIGHTED_STRESS_SOURCE_BUDGET = 200
G6_WEIGHTED_STRESS_TARGET_BUDGET = 1000

# Frozen intrinsic-ruler reference. Provenance: DaguaGraph.compute_node_sizes()
# with the corpus-default NodeStyle (font_size=9.0, padding=(11.0, 9.0),
# roundrect, shrink_text) yields a default short-label node box [44.0, 34.0].
CANONICAL_NODE_HEIGHT_REF = 34.0
INTRINSIC_RULER_FLAG = "DEGENERATE_SCALE"


@dataclass(frozen=True)
class GroupSlot:
    """Frozen tier slot occupied by one conditional-group facet.

    Parameters
    ----------
    facet_name : str
        Published facet code for the slot.
    tier : int
        Evidence tier number, where 1 is strongest and 3 is weakest.
    weight_within_slot : float
        Split weight inside the slot. Batch-1 groups all use a single check per
        slot, so this value is ``1.0``.
    """

    facet_name: str
    tier: int
    weight_within_slot: float


@dataclass(frozen=True)
class GroupFacetScore:
    """Publication-ready score from an applicable conditional-group facet.

    Parameters
    ----------
    code : str
        Stable facet code.
    name : str
        Human-readable facet name.
    tier : int
        Evidence tier number.
    score : Optional[float]
        Normalized value in ``[0, 1]`` or ``None`` when absent under DOC-4.
    base_weight : float
        Frozen tier-slot weight before graded-gate multipliers.
    effective_weight : float
        Weight after graded-gate multipliers such as acyclicity fraction or
        weight-CV saturation.
    applicable : bool
        Whether this facet contributes to the macro-average.
    applicability_reason : str
        Input-only gate reason for publication records.
    metadata : Mapping[str, Any]
        Normalization formula, sample counts, invariance notes, and diagnostics.
    replaces_core : Optional[str]
        CORE facet code replaced by this group facet, used for G6's C1 target
        swap to avoid stress double-counting.
    """

    code: str
    name: str
    tier: int
    score: Optional[float]
    base_weight: float
    effective_weight: float
    applicable: bool
    applicability_reason: str
    metadata: Mapping[str, Any]
    replaces_core: Optional[str] = None


@dataclass(frozen=True)
class GroupEvaluation:
    """Result of evaluating one conditional group.

    Parameters
    ----------
    key : str
        Registry key for the group.
    applicable : bool
        Whether the input-only gate fired.
    applicability_reason : str
        Published gate reason.
    facets : Mapping[str, GroupFacetScore]
        Applicable facet records keyed by facet code.
    metadata : Mapping[str, Any]
        Group-level diagnostics and probe metadata.
    """

    key: str
    applicable: bool
    applicability_reason: str
    facets: Mapping[str, GroupFacetScore]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class IntrinsicRulerCanonicalization:
    """Per-drawing canonical-frame scale for reused V2 machinery.

    Parameters
    ----------
    unit_ruler : float
        Median node-box height in the drawing's input units.
    reference_height : float
        Frozen canonical node-box height used by V3.
    scale : float
        ``unit_ruler / reference_height`` or ``1.0`` for degenerate drawings.
    degenerate : bool
        Whether the non-positive ruler guard fired.
    flags : Tuple[str, ...]
        Row flags contributed by this canonicalization.
    """

    unit_ruler: float
    reference_height: float
    scale: float
    degenerate: bool
    flags: Tuple[str, ...]


@dataclass(frozen=True)
class ConditionalGroup:
    """Registry entry for one V3 conditional group.

    Parameters
    ----------
    key : str
        Stable group key.
    applicability_gate : Callable[[Mapping[str, Any]], bool]
        Input-only gate over declared graph metadata.
    tier_slots : Tuple[GroupSlot, ...]
        Frozen DOC-6 tier slots occupied by this group.
    score_fn : Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, Mapping[str, Any]], GroupEvaluation
    ]
        Function that scores the group when its gate fires.
    """

    key: str
    applicability_gate: Callable[[Mapping[str, Any]], bool]
    tier_slots: Tuple[GroupSlot, ...]
    score_fn: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, Mapping[str, Any]], GroupEvaluation
    ]


@dataclass(frozen=True)
class TreeForest:
    """Normalized directed rooted forest used by G4.

    Parameters
    ----------
    children : Mapping[int, Tuple[int, ...]]
        Children keyed by parent node id.
    parents : Mapping[int, Optional[int]]
        Parent keyed by node id; roots map to ``None``.
    roots : Tuple[int, ...]
        Declared or inferred roots.
    depths : Mapping[int, int]
        Rooted depth per node.
    components : Tuple[Tuple[int, ...], ...]
        Forest components as node-id tuples.
    subtree_nodes : Mapping[int, Tuple[int, ...]]
        Descendant node ids including the key node.
    subtree_leaves : Mapping[int, int]
        Leaf count per subtree.
    subtree_hashes : Mapping[int, str]
        Canonical ordered subtree hash per node.
    """

    children: Mapping[int, Tuple[int, ...]]
    parents: Mapping[int, Optional[int]]
    roots: Tuple[int, ...]
    depths: Mapping[int, int]
    components: Tuple[Tuple[int, ...], ...]
    subtree_nodes: Mapping[int, Tuple[int, ...]]
    subtree_leaves: Mapping[int, int]
    subtree_hashes: Mapping[int, str]


def evaluate_conditional_groups(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    graph_meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, GroupEvaluation]:
    """Evaluate all registered Batch-1 V3 conditional groups.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Directed or undirected edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node box sizes with shape ``[N, 2]``. Batch-1 groups do not use sizes
        directly, but the registry signature leaves the seam for G2/G7.
    graph_meta : Optional[Mapping[str, Any]], optional
        Declared input metadata used by gates. Positions are never consulted for
        applicability.

    Returns
    -------
    Dict[str, GroupEvaluation]
        Group evaluations keyed by registry key.
    """
    meta: Mapping[str, Any] = {} if graph_meta is None else graph_meta
    results: Dict[str, GroupEvaluation] = {}
    for key, group in GROUP_REGISTRY.items():
        if not group.applicability_gate(meta):
            reason = _inapplicable_reason(key, meta)
            results[key] = GroupEvaluation(
                key=key,
                applicable=False,
                applicability_reason=reason,
                facets={},
                metadata={"gate": "input_only", "tier_slots": _slot_metadata(group.tier_slots)},
            )
            continue
        results[key] = group.score_fn(pos, edge_index, node_sizes, meta)
    return results


def _slot_metadata(slots: Sequence[GroupSlot]) -> Tuple[Dict[str, Any], ...]:
    """Serialize tier-slot declarations for publication metadata.

    Parameters
    ----------
    slots : Sequence[GroupSlot]
        Frozen group slots.

    Returns
    -------
    Tuple[Dict[str, Any], ...]
        JSON-friendly slot records.
    """
    return tuple(
        {
            "facet_name": slot.facet_name,
            "tier": slot.tier,
            "weight_within_slot": slot.weight_within_slot,
        }
        for slot in slots
    )


def _intrinsic_ruler_canonicalization(sizes: torch.Tensor) -> IntrinsicRulerCanonicalization:
    """Return the single V3 intrinsic-ruler scale for a drawing.

    Parameters
    ----------
    sizes : torch.Tensor
        Node box sizes with shape ``[N, 2]`` in drawing units.

    Returns
    -------
    IntrinsicRulerCanonicalization
        Frozen median-height ruler and the canonical-frame divisor.
    """
    size_tensor = _ensure_cpu(sizes).to(dtype=torch.float64)
    if size_tensor.ndim != 2 or int(size_tensor.shape[1]) != 2 or int(size_tensor.shape[0]) == 0:
        unit_ruler = 0.0
    else:
        unit_ruler = float(torch.median(size_tensor[:, 1]).item())
    degenerate = not math.isfinite(unit_ruler) or unit_ruler <= 0.0
    scale = 1.0 if degenerate else unit_ruler / CANONICAL_NODE_HEIGHT_REF
    return IntrinsicRulerCanonicalization(
        unit_ruler=unit_ruler,
        reference_height=CANONICAL_NODE_HEIGHT_REF,
        scale=scale,
        degenerate=degenerate,
        flags=(INTRINSIC_RULER_FLAG,) if degenerate else tuple(),
    )


def _canonicalization_metadata(
    canonicalization: IntrinsicRulerCanonicalization,
) -> Dict[str, Any]:
    """Serialize intrinsic-ruler canonicalization metadata.

    Parameters
    ----------
    canonicalization : IntrinsicRulerCanonicalization
        Per-drawing canonicalization record.

    Returns
    -------
    Dict[str, Any]
        Publication metadata for canonicalized wrapper facets.
    """
    return {
        "intrinsic_ruler": "median_node_box_height",
        "intrinsic_ruler_value": canonicalization.unit_ruler,
        "intrinsic_ruler_reference_height": canonicalization.reference_height,
        "intrinsic_ruler_scale": canonicalization.scale,
        "intrinsic_ruler_degenerate": canonicalization.degenerate,
        "intrinsic_ruler_guard": "s:=1_and_DEGENERATE_SCALE_when_median_node_height_non_positive",
        "intrinsic_ruler_reference_provenance": (
            "DaguaGraph.compute_node_sizes default NodeStyle short-label node height"
        ),
    }


def _canonicalize_tensor(
    tensor: torch.Tensor,
    canonicalization: IntrinsicRulerCanonicalization,
) -> torch.Tensor:
    """Divide tensor coordinates or extents into the canonical frame.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor expressed in drawing units.
    canonicalization : IntrinsicRulerCanonicalization
        Per-drawing canonicalization record.

    Returns
    -------
    torch.Tensor
        Tensor expressed in canonical node-height units.
    """
    return tensor / canonicalization.scale


def _canonicalize_point(
    point: Tuple[float, float],
    canonicalization: IntrinsicRulerCanonicalization,
) -> Tuple[float, float]:
    """Divide one point into the canonical frame.

    Parameters
    ----------
    point : Tuple[float, float]
        Point in drawing units.
    canonicalization : IntrinsicRulerCanonicalization
        Per-drawing canonicalization record.

    Returns
    -------
    Tuple[float, float]
        Point in canonical node-height units.
    """
    return (float(point[0]) / canonicalization.scale, float(point[1]) / canonicalization.scale)


def _canonicalize_curves(
    curves: Sequence[Any],
    canonicalization: IntrinsicRulerCanonicalization,
) -> Tuple[Any, ...]:
    """Divide routed curves into the canonical frame.

    Parameters
    ----------
    curves : Sequence[Any]
        BezierCurve-compatible routed curves.
    canonicalization : IntrinsicRulerCanonicalization
        Per-drawing canonicalization record.

    Returns
    -------
    Tuple[Any, ...]
        Curves in canonical node-height units.
    """
    from dagua.edges import BezierCurve

    canonical_curves = []
    for curve in curves:
        waypoints = getattr(curve, "waypoints", None)
        canonical_waypoints = (
            tuple(_canonicalize_point(point, canonicalization) for point in waypoints)
            if waypoints is not None
            else None
        )
        canonical_curves.append(
            BezierCurve(
                _canonicalize_point(curve.p0, canonicalization),
                _canonicalize_point(curve.cp1, canonicalization),
                _canonicalize_point(curve.cp2, canonicalization),
                _canonicalize_point(curve.p1, canonicalization),
                waypoints=canonical_waypoints,
                routing=getattr(curve, "routing", "bezier"),
                direction=getattr(curve, "direction", "TB"),
                step_fraction=getattr(curve, "step_fraction", None),
            )
        )
    return tuple(canonical_curves)


def _canonicalize_port_records(
    ports: Sequence[Mapping[str, Any]],
    canonicalization: IntrinsicRulerCanonicalization,
) -> Tuple[Dict[str, Any], ...]:
    """Divide explicit port coordinates into the canonical frame.

    Parameters
    ----------
    ports : Sequence[Mapping[str, Any]]
        Declared port endpoint records.
    canonicalization : IntrinsicRulerCanonicalization
        Per-drawing canonicalization record.

    Returns
    -------
    Tuple[Dict[str, Any], ...]
        Port records with explicit ``position`` coordinates canonicalized.
    """
    canonical_ports = []
    for port in ports:
        record = dict(port)
        point = record.get("position")
        if point is not None:
            record["position"] = _canonicalize_point(
                (float(point[0]), float(point[1])),
                canonicalization,
            )
        canonical_ports.append(record)
    return tuple(canonical_ports)


def _canonicalize_label_positions(
    label_positions: Any,
    canonicalization: IntrinsicRulerCanonicalization,
) -> Any:
    """Divide optional edge-label positions into the canonical frame.

    Parameters
    ----------
    label_positions : Any
        Optional sequence of label positions or ``None`` entries.
    canonicalization : IntrinsicRulerCanonicalization
        Per-drawing canonicalization record.

    Returns
    -------
    Any
        Label positions in canonical node-height units.
    """
    if label_positions is None:
        return None
    if isinstance(label_positions, torch.Tensor):
        return _canonicalize_tensor(label_positions, canonicalization)
    return [
        None if point is None else _canonicalize_point(point, canonicalization)
        for point in label_positions
    ]


def _merge_flags(*flag_groups: Sequence[str]) -> Tuple[str, ...]:
    """Return unique row flags preserving first-seen order.

    Parameters
    ----------
    *flag_groups : Sequence[str]
        Flag sequences to merge.

    Returns
    -------
    Tuple[str, ...]
        Unique flags.
    """
    flags: List[str] = []
    for group in flag_groups:
        for flag in group:
            if flag not in flags:
                flags.append(flag)
    return tuple(flags)


def _tier_weight(tier: int) -> float:
    """Return the frozen V3 multiplier for a tier.

    Parameters
    ----------
    tier : int
        Evidence tier number.

    Returns
    -------
    float
        Frozen multiplier.
    """
    if tier == 1:
        return TIER_1_WEIGHT
    if tier == 2:
        return TIER_2_WEIGHT
    if tier == 3:
        return TIER_3_WEIGHT
    raise ValueError(f"unsupported V3 tier: {tier}")


def _declared_hierarchy_gate(meta: Mapping[str, Any]) -> bool:
    """Gate G1 from declared hierarchy metadata only.

    Parameters
    ----------
    meta : Mapping[str, Any]
        Declared graph metadata.

    Returns
    -------
    bool
        ``True`` when the input row explicitly declares hierarchy/ranks.
    """
    return bool(
        meta.get("declared_hierarchical", False)
        or meta.get("has_dag_metadata", False)
        or meta.get("has_rank_metadata", False)
        or meta.get("declared_hierarchy", False)
    )


def _planted_partition_gate(meta: Mapping[str, Any]) -> bool:
    """Gate G3 from pre-registered planted partition metadata only.

    Parameters
    ----------
    meta : Mapping[str, Any]
        Declared graph metadata.

    Returns
    -------
    bool
        ``True`` only when planted community ground truth is declared.
    """
    return _metadata_sequence(meta, ("planted_partition", "planted_partitions")) is not None


def _declared_cluster_gate(meta: Mapping[str, Any]) -> bool:
    """Gate G2 from declared cluster metadata only.

    Parameters
    ----------
    meta : Mapping[str, Any]
        Declared graph metadata.

    Returns
    -------
    bool
        ``True`` when the corpus row declares clusters or cluster parents.
    """
    return meta.get("clusters") is not None or meta.get("cluster_parents") is not None


def _declared_tree_gate(meta: Mapping[str, Any]) -> bool:
    """Gate G4 from declared rooted-tree convention metadata only.

    Parameters
    ----------
    meta : Mapping[str, Any]
        Declared graph metadata.

    Returns
    -------
    bool
        ``True`` when the row declares a rooted tree/forest and a tree drawing
        convention. The scorer validates the edge input against the frozen tree
        predicate because the registry gate is metadata-only.
    """
    convention = str(meta.get("tree_convention", meta.get("tree_layout", ""))).lower()
    has_convention = convention in {"layered", "radial"}
    return bool(
        has_convention
        and (
            meta.get("declared_tree", False)
            or meta.get("rooted_tree", False)
            or meta.get("tree_roots") is not None
            or meta.get("root") is not None
        )
    )


def _weighted_gate(meta: Mapping[str, Any]) -> bool:
    """Gate G6 from declared non-degenerate weighted-distance semantics.

    Parameters
    ----------
    meta : Mapping[str, Any]
        Declared graph metadata.

    Returns
    -------
    bool
        ``True`` when weights and a geometric weight mode are declared.
    """
    if str(meta.get("weight_mode", "distance")).lower() == "thickness-only":
        return False
    weights = _edge_weights_from_meta(meta)
    if weights is None or weights.size == 0:
        return False
    return bool(_weight_cv(weights) > NONDEGENERATE_WEIGHT_CV)


def _declared_temporal_gate(meta: Mapping[str, Any]) -> bool:
    """Gate G5 from declared frame history and node identity only.

    Parameters
    ----------
    meta : Mapping[str, Any]
        Declared graph metadata.

    Returns
    -------
    bool
        ``True`` when previous frame geometry and node identity metadata are
        declared by the corpus row.
    """
    has_frames = any(
        meta.get(key) is not None for key in ("previous", "previous_positions", "frames")
    )
    has_identity = any(
        meta.get(key) is not None
        for key in ("node_identity", "node_identity_map", "node_ids", "frame_node_ids")
    )
    has_declared_change = all(
        _temporal_frame_has_graph_change(frame, meta) for frame in _temporal_frame_records(meta)
    )
    return bool(has_frames and has_identity and has_declared_change)


def _declared_port_gate(meta: Mapping[str, Any]) -> bool:
    """Gate G7 from declared ports or row-level routing declarations only.

    Parameters
    ----------
    meta : Mapping[str, Any]
        Declared graph metadata.

    Returns
    -------
    bool
        ``True`` when the input row declares port constraints or the row-level
        routed bundle. Engine-captured route geometry alone cannot change
        applicability.
    """
    return bool(
        meta.get("ports") is not None
        or meta.get("port_sides") is not None
        or meta.get("port_order") is not None
        or meta.get("routing_declared", False)
    )


def _inapplicable_reason(key: str, meta: Mapping[str, Any]) -> str:
    """Return a concise input-gate failure reason.

    Parameters
    ----------
    key : str
        Group key.
    meta : Mapping[str, Any]
        Declared graph metadata.

    Returns
    -------
    str
        Published applicability reason.
    """
    if key == "G1":
        return "inapplicable:no_declared_hierarchy_metadata"
    if key == "G3":
        return "inapplicable:no_pre_registered_planted_partition"
    if key == "G2":
        return "inapplicable:no_declared_cluster_metadata"
    if key == "G4":
        return "inapplicable:no_declared_rooted_tree_convention"
    if key == "G6" and str(meta.get("weight_mode", "distance")).lower() == "thickness-only":
        return "inapplicable:thickness_only_weights_make_no_geometric_claim"
    if key == "G6":
        return "inapplicable:no_declared_non_degenerate_weight_semantics"
    if key == "G5":
        return "inapplicable:no_declared_previous_frames_with_node_identity_and_graph_change"
    if key == "G7":
        return "inapplicable:no_declared_ports_or_row_level_routing"
    return "inapplicable:input_gate_false"


def _score_g1(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    meta: Mapping[str, Any],
) -> GroupEvaluation:
    """Score G1 directed-flow facets on declared-hierarchy rows.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Directed edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node box sizes with shape ``[N, 2]``; unused by G1.
    meta : Mapping[str, Any]
        Declared input metadata.

    Returns
    -------
    GroupEvaluation
        G1 flow and depth-order facet records.
    """
    del node_sizes
    positions = _ensure_cpu(pos).to(dtype=torch.float64)
    edges = _normalize_edge_index(edge_index)
    direction = str(meta.get("flow_direction", meta.get("direction", "TB")))
    back_edge_mask = _optional_bool_tensor(meta.get("back_edge_mask"), int(edges.shape[1]))
    hierarchy_edges = _drop_masked_edges(edges, back_edge_mask)
    frac_acyclic, depth = _acyclic_fraction_and_depth(
        hierarchy_edges,
        int(positions.shape[0]),
        meta,
    )
    flow = directed_flow_score(
        positions,
        edges,
        direction=direction,
        back_edge_mask=back_edge_mask,
    )["directed_flow_score"]
    depth_score, depth_meta = _depth_spearman_score(positions, depth, direction)
    reason = "applicable:declared_hierarchy_metadata_x_acyclicity_fraction"
    common_meta = {
        "gate": "declared_hierarchy_x_frac_acyclic",
        "frac_acyclic": frac_acyclic,
        "declared_axis_direction": direction,
        "invariance": "axis_anchored_after_declared_axis_transform;not_rotation_invariant",
        "normalization": "directed_flow_score reused from dagua.metrics; depth=(spearman+1)/2",
    }
    facets = {
        "G1_directed_flow": GroupFacetScore(
            code="G1_directed_flow",
            name="directed_flow_score",
            tier=2,
            score=float(flow),
            base_weight=_tier_weight(2),
            effective_weight=_tier_weight(2) * frac_acyclic,
            applicable=True,
            applicability_reason=reason,
            metadata={**common_meta, "sample_count": int(hierarchy_edges.shape[1])},
        ),
        "G1_depth_order": GroupFacetScore(
            code="G1_depth_order",
            name="depth_spearman",
            tier=3,
            score=depth_score,
            base_weight=_tier_weight(3),
            effective_weight=0.0,
            applicable=False,
            applicability_reason="diagnostic_only:redundant_with_G1_directed_flow_and_G4_depth",
            metadata={
                **common_meta,
                **depth_meta,
                "diagnostic_only": True,
                "audit_demotion": "sec_2_3_redundant_T3_scale_invariant_facet",
            },
        ),
    }
    return GroupEvaluation(
        key="G1",
        applicable=True,
        applicability_reason=reason,
        facets=facets,
        metadata={"tier_slots": _slot_metadata(GROUP_REGISTRY["G1"].tier_slots), **common_meta},
    )


def _score_g3(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    meta: Mapping[str, Any],
) -> GroupEvaluation:
    """Score G3 planted-community geometric clustering fidelity.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``; unused by G3.
    node_sizes : torch.Tensor
        Node box sizes with shape ``[N, 2]``; unused by G3.
    meta : Mapping[str, Any]
        Declared input metadata containing planted partitions.

    Returns
    -------
    GroupEvaluation
        G3 ARI facet record.
    """
    del edge_index, node_sizes
    positions = _ensure_cpu(pos).to(dtype=torch.float64)
    planted = _metadata_sequence(meta, ("planted_partition", "planted_partitions"))
    if planted is None:
        raise ValueError("G3 score called without planted partition metadata")
    planted_labels = np.asarray(list(planted), dtype=np.int64)
    if planted_labels.shape[0] != int(positions.shape[0]):
        raise ValueError("planted_partition must have length N")
    cluster_count = int(np.unique(planted_labels).size)
    predicted = _hac_labels(positions.numpy(), cluster_count)
    ari = adjusted_rand_index(planted_labels, predicted)
    reason = "applicable:pre_registered_planted_partition"
    metadata = {
        "gate": "pre_registered_planted_partition_only",
        "normalization": "adjusted_rand_index remapped as max(0, ARI)",
        "raw_adjusted_rand_index": ari,
        "sample_count": int(positions.shape[0]),
        "cluster_count": cluster_count,
        "hac_linkage": "average",
        "hac_metric": "euclidean",
        "invariance": "translation_rotation_reflection_position_scale_and_unit_scale_invariant",
        "deformation_monotonicity_probe": "implemented_by_tests/test_ruler_v3_groups.py",
    }
    facets = {
        "G3_community_ari": GroupFacetScore(
            code="G3_community_ari",
            name="planted_partition_hac_ari",
            tier=2,
            score=max(0.0, min(1.0, ari)),
            base_weight=_tier_weight(2),
            effective_weight=_tier_weight(2),
            applicable=True,
            applicability_reason=reason,
            metadata=metadata,
        )
    }
    return GroupEvaluation(
        key="G3",
        applicable=True,
        applicability_reason=reason,
        facets=facets,
        metadata={"tier_slots": _slot_metadata(GROUP_REGISTRY["G3"].tier_slots), **metadata},
    )


def _score_g2(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    meta: Mapping[str, Any],
) -> GroupEvaluation:
    """Score G2 declared-cluster containment and faithfulness facets.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node box sizes with shape ``[N, 2]``.
    meta : Mapping[str, Any]
        Declared cluster metadata.

    Returns
    -------
    GroupEvaluation
        G2 facet records split across two T2 tier slots.
    """
    positions = _ensure_cpu(pos).to(dtype=torch.float64)
    edges = _normalize_edge_index(edge_index)
    sizes = _ensure_cpu(node_sizes).to(dtype=torch.float64)
    clusters = _declared_clusters(meta, int(positions.shape[0]))
    cluster_parents = _declared_cluster_parents(meta)
    cluster_labels = _declared_cluster_labels(meta)
    declared_labels = _cluster_labels_from_clusters(clusters, int(positions.shape[0]))
    cluster_count = int(np.unique(declared_labels).size)
    has_ari_membership = _has_cluster_with_min_members(clusters, min_members=2)
    ari_raw: Optional[float] = None
    if has_ari_membership:
        predicted = _hac_labels(positions.numpy(), cluster_count)
        ari_raw = adjusted_rand_index(declared_labels, predicted)
    compactness_score, compactness_meta = _cluster_log_ratio_compactness(
        positions,
        sizes,
        clusters,
    )
    reason = "applicable:declared_cluster_metadata"
    slot_a_canonicalization = _intrinsic_ruler_canonicalization(sizes)
    common = {
        "gate": "declared_clusters_or_cluster_parents_only",
        "cluster_count": cluster_count,
        "flags": slot_a_canonicalization.flags,
        "invariance_slot_a": (
            "translation_invariant;rotation_reflection_axis_anchored_aabb;"
            "position_only_scale_sensitive_by_design;unit_scale_invariant_exact_via_"
            "intrinsic_ruler_canonicalization;anisotropic_not_invariant;"
            "node_relabel_invariant;canonical_unit_f3_honesty"
        ),
        "invariance_slot_b_ari": "translation_rotation_reflection_position_scale_invariant",
        "invariance_slot_b_compactness": "size_anchored_position_scale_sensitive_by_design",
        "deformation_monotonicity_probe": "implemented_by_tests/test_ruler_v3_groups.py",
    }
    facets = {}
    slot_a_scores = _g2_slot_a_scores(
        positions,
        edges,
        sizes,
        clusters,
        cluster_parents,
        cluster_labels,
        canonicalization=slot_a_canonicalization,
    )
    active_slot_a_total = sum(
        slot_weight
        for _code, _name, raw_score, slot_weight, _metadata in slot_a_scores
        if raw_score is not None and _code not in G2_SLOT_A_DIAGNOSTIC_FACETS
    )
    for code, name, raw_score, slot_weight, metadata in slot_a_scores:
        diagnostic_only = code in G2_SLOT_A_DIAGNOSTIC_FACETS
        effective_weight = (
            _tier_weight(2) * slot_weight / active_slot_a_total
            if active_slot_a_total > 0.0 and not diagnostic_only
            else 0.0
        )
        facets[code] = GroupFacetScore(
            code=code,
            name=name,
            tier=2,
            score=raw_score,
            base_weight=effective_weight,
            effective_weight=effective_weight if raw_score is not None else 0.0,
            applicable=raw_score is not None and not diagnostic_only,
            applicability_reason=(
                "diagnostic_only:sec_2_3_redundant_with_C4" if diagnostic_only else reason
            ),
            metadata={
                **common,
                **metadata,
                "slot": "G2_A_containment_hygiene",
                "slot_internal_weight": slot_weight,
                **(
                    {
                        "diagnostic_only": True,
                        "audit_demotion": "sec_2_3_redundant_with_C4_clean_units_field",
                    }
                    if diagnostic_only
                    else {}
                ),
                "normalization": (
                    "reused frozen dagua.metrics AABB cluster scorer in canonical node-height frame"
                ),
                "invariance": (
                    "translation_invariant;rotation_reflection_axis_anchored_aabb;"
                    "position_only_scale_sensitive_by_design;unit_scale_invariant_exact_via_"
                    "intrinsic_ruler_canonicalization;anisotropic_not_invariant;"
                    "node_relabel_invariant"
                ),
                "f3_note": (
                    "G2 Slot A compared projection-off; canonicalization delta published per "
                    "the unit-invariance ruling"
                ),
            },
        )
    ari_score = None if ari_raw is None else max(0.0, min(1.0, ari_raw))
    slot_b_scores = (
        (
            "G2_cluster_hac_ari",
            "declared_cluster_hac_ari",
            ari_score,
            2.0,
            {
                "raw_adjusted_rand_index": ari_raw,
                "hac_linkage": "average",
                "hac_metric": "euclidean",
                "normalization": "adjusted_rand_index remapped as max(0, ARI)",
                "invariance": "translation_rotation_reflection_position_scale_invariant",
                "diagnostic_capable": True,
                "membership_gate": "requires_at_least_one_declared_cluster_with_at_least_2_members",
            },
        ),
        (
            "G2_cluster_compactness_log_band",
            "cluster_compactness_log_ratio_band",
            compactness_score,
            1.0,
            {
                **compactness_meta,
                "normalization": "leaf cluster AABB/member-area log-ratio band",
                "invariance": "size_anchored_position_scale_sensitive_by_design",
                "monotonicity_note": (
                    "intra_cluster_diagnostic_within_ARI_primary_slot_b; "
                    "whole-cluster placement monotonicity is carried by G2_cluster_hac_ari"
                ),
            },
        ),
    )
    active_slot_b_total = sum(
        slot_weight
        for _code, _name, score, slot_weight, _metadata in slot_b_scores
        if score is not None
    )
    for code, name, score, slot_weight, metadata in slot_b_scores:
        effective_weight = (
            _tier_weight(2) * slot_weight / active_slot_b_total
            if active_slot_b_total > 0.0
            else 0.0
        )
        facets[code] = GroupFacetScore(
            code=code,
            name=name,
            tier=2,
            score=score,
            base_weight=effective_weight,
            effective_weight=effective_weight if score is not None else 0.0,
            applicable=score is not None,
            applicability_reason=reason,
            metadata={
                **common,
                **metadata,
                "slot": "G2_B_cluster_faithfulness",
                "slot_internal_weight": slot_weight,
            },
        )
    return GroupEvaluation(
        key="G2",
        applicable=True,
        applicability_reason=reason,
        facets=facets,
        metadata={"tier_slots": _slot_metadata(GROUP_REGISTRY["G2"].tier_slots), **common},
    )


def _score_g4(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    meta: Mapping[str, Any],
) -> GroupEvaluation:
    """Score G4 rooted-tree layered or radial facets.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Directed parent-child edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node box sizes with shape ``[N, 2]``.
    meta : Mapping[str, Any]
        Declared tree roots and convention metadata.

    Returns
    -------
    GroupEvaluation
        G4 facet records split across two T2 tier slots.
    """
    positions = _ensure_cpu(pos).to(dtype=torch.float64)
    edges = _normalize_edge_index(edge_index)
    sizes = _ensure_cpu(node_sizes).to(dtype=torch.float64)
    convention = str(meta.get("tree_convention", meta.get("tree_layout", "layered"))).lower()
    reason = f"applicable:declared_rooted_tree_{convention}_convention"
    try:
        forest = _tree_forest(edges, int(positions.shape[0]), meta)
    except ValueError as exc:
        return GroupEvaluation(
            key="G4",
            applicable=False,
            applicability_reason="inapplicable:malformed_declared_tree",
            facets={},
            metadata={
                "tier_slots": _slot_metadata(GROUP_REGISTRY["G4"].tier_slots),
                "gate": "declared_rooted_tree_forest_x_declared_convention",
                "tree_convention": convention,
                "malformed_reason": str(exc),
            },
        )
    common = {
        "gate": "declared_rooted_tree_forest_x_declared_convention",
        "tree_convention": convention,
        "root_count": len(forest.roots),
        "invariance": "mixed;see_facet_metadata",
        "normalization": "continuous ratios/orders/ranks with frozen BJL/radial reference extents",
        "deformation_monotonicity_probe": "implemented_by_tests/test_ruler_v3_groups.py",
    }
    if convention == "radial":
        scored = _g4_radial_scores(positions, sizes, forest)
    else:
        scored = _g4_layered_scores(positions, sizes, forest)
    facets = {}
    for code, name, score, slot, slot_weight, slot_total, metadata in scored:
        effective_weight = _tier_weight(2) * slot_weight / slot_total
        facets[code] = GroupFacetScore(
            code=code,
            name=name,
            tier=2,
            score=score,
            base_weight=effective_weight,
            effective_weight=effective_weight if score is not None else 0.0,
            applicable=score is not None,
            applicability_reason=reason,
            metadata={**common, **metadata, "slot": slot, "slot_internal_weight": slot_weight},
        )
    return GroupEvaluation(
        key="G4",
        applicable=True,
        applicability_reason=reason,
        facets=facets,
        metadata={"tier_slots": _slot_metadata(GROUP_REGISTRY["G4"].tier_slots), **common},
    )


def _score_g6(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    meta: Mapping[str, Any],
) -> GroupEvaluation:
    """Score G6 weighted-distance fidelity and local monotonicity.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node box sizes with shape ``[N, 2]``; unused by G6.
    meta : Mapping[str, Any]
        Declared weight metadata.

    Returns
    -------
    GroupEvaluation
        G6 weighted KSM replacement facet and local monotonicity facet.
    """
    del node_sizes
    positions = _ensure_cpu(pos).to(dtype=torch.float64)
    edges = _normalize_edge_index(edge_index)
    weights = _edge_weights_from_meta(meta)
    if weights is None:
        raise ValueError("G6 score called without declared edge weights")
    if weights.shape[0] != int(edges.shape[1]):
        raise ValueError("edge_weights must have length E")
    mode = str(meta.get("weight_mode", "distance")).lower()
    distances = _weights_to_distances(weights, mode)
    cv = _weight_cv(weights)
    cv_gate = min(1.0, cv / WEIGHT_CV_SATURATION)
    weighted_dist = _weighted_all_pairs_distances(edges, int(positions.shape[0]), distances)
    ksm = weighted_isotonic_ksm(
        positions,
        weighted_dist,
        n_sources=G6_WEIGHTED_STRESS_SOURCE_BUDGET,
        n_targets=G6_WEIGHTED_STRESS_TARGET_BUDGET,
    )
    local = local_weight_monotonicity_score(positions, edges, weights, mode)
    reason = "applicable:declared_non_degenerate_weight_semantics"
    common = {
        "gate": "declared_non_degenerate_weights_x_weight_cv_saturation",
        "weight_mode": mode,
        "weight_cv": cv,
        "weight_cv_gate": cv_gate,
        "invariance": "translation_rotation_reflection_position_scale_and_unit_scale_invariant",
    }
    facets = {
        "G6_weighted_ksm": GroupFacetScore(
            code="G6_weighted_ksm",
            name="weighted_shortest_path_isotonic_ksm",
            tier=1,
            score=float(ksm["weighted_ksm_score"]),
            base_weight=_tier_weight(1),
            effective_weight=_tier_weight(1) * cv_gate,
            applicable=True,
            applicability_reason=reason,
            metadata={
                **common,
                **ksm,
                "normalization": "aspect-preserving isotonic KSM on weighted APSP distances",
                "replaces_core": "C1",
            },
            replaces_core="C1",
        ),
        "G6_local_weight_monotonicity": GroupFacetScore(
            code="G6_local_weight_monotonicity",
            name="local_weight_monotonicity",
            tier=3,
            score=float(local["local_weight_monotonicity_score"]),
            base_weight=_tier_weight(3),
            effective_weight=_tier_weight(3) * cv_gate,
            applicable=True,
            applicability_reason=reason,
            metadata={
                **common,
                **local,
                "normalization": "mean incident-edge Spearman remapped as (rho+1)/2",
            },
        ),
    }
    return GroupEvaluation(
        key="G6",
        applicable=True,
        applicability_reason=reason,
        facets=facets,
        metadata={"tier_slots": _slot_metadata(GROUP_REGISTRY["G6"].tier_slots), **common},
    )


def _score_g5(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    meta: Mapping[str, Any],
) -> GroupEvaluation:
    """Score G5 temporal stability under a static-quality band.

    Parameters
    ----------
    pos : torch.Tensor
        Current-frame node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``; unused except for the registry
        signature.
    node_sizes : torch.Tensor
        Node box sizes with shape ``[N, 2]``; unused by G5.
    meta : Mapping[str, Any]
        Declared temporal metadata containing previous frame positions, node
        identity, quality-band records, and graph-change magnitude.

    Returns
    -------
    GroupEvaluation
        G5 stability facet record.
    """
    del edge_index, node_sizes
    positions = _ensure_cpu(pos).to(dtype=torch.float64)
    current_ids = _temporal_current_node_ids(meta, int(positions.shape[0]))
    frame_records = _temporal_frame_records(meta)
    if not frame_records:
        raise ValueError("G5 score called without declared previous frame metadata")

    frame_scores: list[float] = []
    frame_metadata: list[Dict[str, Any]] = []
    band_decays: list[float] = []
    stability_band_flag = False
    for frame in frame_records:
        aligned_prev, curr_aligned, matched_count = _aligned_temporal_pair(
            frame,
            positions,
            current_ids,
        )
        observed = _normalized_mean_displacement(aligned_prev, curr_aligned)
        if not _temporal_frame_has_graph_change(frame, meta):
            raise ValueError(
                "G5 requires declared graph_change; observed displacement is not a default"
            )
        graph_change = max(
            0.0,
            float(frame.get("graph_change", frame.get("change", meta.get("graph_change")))),
        )
        dcq = _change_proportionality_score(observed, graph_change)
        decay, below_band = _temporal_quality_decay(frame, meta)
        stability_band_flag = stability_band_flag or below_band
        band_decays.append(decay)
        frame_scores.append(dcq * decay)
        procrustes = pairwise_procrustes_matrix(
            [aligned_prev.numpy(), curr_aligned.numpy()],
            free_aspect=False,
        )
        frame_metadata.append(
            {
                "matched_nodes": matched_count,
                "observed_procrustes_aligned_displacement": observed,
                "ground_truth_graph_change": graph_change,
                "dcq_change_proportionality": dcq,
                "quality_band_decay": decay,
                "below_quality_band": below_band,
                "pairwise_procrustes_distance": float(procrustes[0, 1]),
            }
        )

    score = float(np.mean(frame_scores)) if frame_scores else None
    reason = "applicable:declared_previous_frames_with_node_identity_and_graph_change"
    metadata = {
        "gate": "previous_frames_x_node_identity_x_declared_graph_change",
        "normalization": (
            "mean graded DCQ proportionality after Procrustes alignment x quality-band decay"
        ),
        "quality_band_delta": G5_QUALITY_BAND_DELTA,
        "frame_count": len(frame_scores),
        "quality_band_decays": tuple(float(value) for value in band_decays),
        "frame_records": tuple(frame_metadata),
        "flags": ("STABILITY_BAND",) if stability_band_flag else tuple(),
        "invariance": (
            "translation_rotation_reflection_position_scale_invariant_after_procrustes_alignment"
        ),
        "honesty_note": (
            "Temporal stability aids orientation/navigation across frames; it is not "
            "a general readability substitute."
        ),
    }
    facets = {
        "G5_temporal_stability": GroupFacetScore(
            code="G5_temporal_stability",
            name="temporal_dcq_stability_band",
            tier=2,
            score=score,
            base_weight=_tier_weight(2),
            effective_weight=_tier_weight(2) if score is not None else 0.0,
            applicable=score is not None,
            applicability_reason=reason,
            metadata=metadata,
        )
    }
    return GroupEvaluation(
        key="G5",
        applicable=True,
        applicability_reason=reason,
        facets=facets,
        metadata={"tier_slots": _slot_metadata(GROUP_REGISTRY["G5"].tier_slots), **metadata},
    )


def _score_g7(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    meta: Mapping[str, Any],
) -> GroupEvaluation:
    """Score G7 port compliance and routed drawing facets.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node box sizes with shape ``[N, 2]``.
    meta : Mapping[str, Any]
        Declared port and optional route metadata.

    Returns
    -------
    GroupEvaluation
        G7 placement compliance plus routed-bundle facet records when routes
        are declared.
    """
    positions = _ensure_cpu(pos).to(dtype=torch.float64)
    edges = _normalize_edge_index(edge_index)
    sizes = _ensure_cpu(node_sizes).to(dtype=torch.float64)
    ports = _declared_port_records(meta, edges)
    routes = meta.get("route_paths", meta.get("routes"))
    routing_declared = bool(meta.get("routing_declared", False))
    route_input = routes if routes is not None else [None] * int(edges.shape[1])
    curves = routes_to_curves(route_input, positions, edges) if routing_declared else None
    routed_canonicalization = _intrinsic_ruler_canonicalization(sizes)
    compliance_positions = _canonicalize_tensor(positions, routed_canonicalization)
    compliance_curves = (
        _canonicalize_curves(curves, routed_canonicalization) if curves is not None else None
    )
    compliance_ports = _canonicalize_port_records(ports, routed_canonicalization)
    compliance = _port_compliance_score(
        compliance_positions,
        edges,
        compliance_ports,
        compliance_curves,
    )
    severe = bool(compliance["severe_side_violation_count"] > 0)
    group_cap = G7_SEVERE_PORT_CAP if severe else 1.0
    flags = _merge_flags(
        ("PORT_VIOLATION",) if compliance["hard_compliance"] < 1.0 else tuple(),
        routed_canonicalization.flags if curves is not None else tuple(),
    )
    reason = "applicable:declared_ports_or_routing"
    common = {
        "gate": "declared_ports_or_routing_only",
        "flags": flags,
        "group_cap": group_cap,
        "normalization": "hard discrete compliance plus decomposed frozen routed drawing terms",
        "invariance_compliance": "discrete_side_and_order_satisfaction_position_scale_invariant",
        "invariance_approach_angle": "rotation_anchored_to_declared_port_side_by_design",
        "invariance_clearance_congestion": (
            "translation_invariant;rotation_reflection_axis_anchored_aabb;"
            "position_only_scale_sensitive_by_design;unit_scale_invariant_exact_via_"
            "intrinsic_ruler_canonicalization;anisotropic_not_invariant;"
            "node_relabel_invariant;canonical_unit_f3_honesty"
        ),
        "deformation_monotonicity_probe": "implemented_by_tests/test_ruler_v3_groups.py",
    }
    compliance_score = min(float(compliance["hard_compliance"]), group_cap)
    facets = {}
    if int(compliance["compliance_checks"]) > 0:
        facets["G7_port_hard_compliance"] = GroupFacetScore(
            code="G7_port_hard_compliance",
            name="port_side_order_position_hard_compliance",
            tier=2,
            score=compliance_score,
            base_weight=_tier_weight(2),
            effective_weight=_tier_weight(2),
            applicable=True,
            applicability_reason=reason,
            metadata={**common, **compliance, "slot": "G7_port_compliance"},
        )
    if curves is not None:
        routed = _g7_routed_bundle_scores(
            positions,
            edges,
            sizes,
            curves,
            ports,
            meta,
            group_cap,
            canonicalization=routed_canonicalization,
        )
        for code, name, score, metadata in routed:
            facets[code] = GroupFacetScore(
                code=code,
                name=name,
                tier=2,
                score=score,
                base_weight=_tier_weight(2),
                effective_weight=_tier_weight(2),
                applicable=True,
                applicability_reason=reason,
                metadata={**common, **metadata},
            )
    return GroupEvaluation(
        key="G7",
        applicable=True,
        applicability_reason=reason,
        facets=facets,
        metadata={
            "tier_slots": _slot_metadata(GROUP_REGISTRY["G7"].tier_slots),
            **common,
            "native_route_coverage": native_route_coverage(routes, int(edges.shape[1])),
            "routed_bundle_applicable": curves is not None,
            "routing_declared": routing_declared,
        },
    )


def _normalize_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    """Return a CPU long edge tensor with shape ``[2, E]``.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor to normalize.

    Returns
    -------
    torch.Tensor
        CPU ``int64`` edge tensor.
    """
    edges = _ensure_cpu(edge_index).to(dtype=torch.long)
    if edges.ndim != 2 or int(edges.shape[0]) != 2:
        raise ValueError("edge_index must have shape [2, E]")
    return edges


def _temporal_frame_records(meta: Mapping[str, Any]) -> Tuple[Mapping[str, Any], ...]:
    """Normalize declared temporal frame metadata.

    Parameters
    ----------
    meta : Mapping[str, Any]
        Declared graph metadata.

    Returns
    -------
    Tuple[Mapping[str, Any], ...]
        Previous-frame records. Each record contains at least ``positions``.
    """
    frames = meta.get("previous", meta.get("frames"))
    if isinstance(frames, Mapping):
        return (frames,)
    if frames is not None:
        records = []
        for frame in frames:
            if isinstance(frame, Mapping):
                records.append(frame)
            else:
                records.append({"positions": frame})
        return tuple(records)
    previous_positions = meta.get("previous_positions")
    if previous_positions is None:
        return tuple()
    return (
        {
            "positions": previous_positions,
            "node_ids": meta.get("previous_node_ids", meta.get("frame_node_ids")),
            "quality": meta.get("previous_v3_core_score", meta.get("frame_v3_core_score")),
            "graph_change": meta.get("graph_change"),
        },
    )


def _temporal_frame_has_graph_change(
    frame: Mapping[str, Any],
    meta: Mapping[str, Any],
) -> bool:
    """Return whether a temporal frame declares ground-truth graph change.

    Parameters
    ----------
    frame : Mapping[str, Any]
        Previous-frame metadata.
    meta : Mapping[str, Any]
        Current row metadata.

    Returns
    -------
    bool
        ``True`` when a frame- or row-level change magnitude is declared.
    """
    return frame.get("graph_change", frame.get("change", meta.get("graph_change"))) is not None


def _temporal_current_node_ids(meta: Mapping[str, Any], num_nodes: int) -> Tuple[Any, ...]:
    """Return current-frame node identities.

    Parameters
    ----------
    meta : Mapping[str, Any]
        Declared graph metadata.
    num_nodes : int
        Current-frame node count.

    Returns
    -------
    Tuple[Any, ...]
        Stable node identities aligned to current positions.
    """
    ids = meta.get("node_ids", meta.get("node_identity"))
    if ids is None and isinstance(meta.get("node_identity_map"), Mapping):
        mapping = meta["node_identity_map"]
        ids = [mapping.get(index, mapping.get(str(index), index)) for index in range(num_nodes)]
    if ids is None:
        ids = list(range(num_nodes))
    values = tuple(ids)
    if len(values) != num_nodes:
        raise ValueError("current temporal node identity metadata must have length N")
    return values


def _aligned_temporal_pair(
    frame: Mapping[str, Any],
    current_positions: torch.Tensor,
    current_ids: Sequence[Any],
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Return Procrustes-aligned previous/current positions on matched nodes.

    Parameters
    ----------
    frame : Mapping[str, Any]
        Previous-frame metadata.
    current_positions : torch.Tensor
        Current positions with shape ``[N, 2]``.
    current_ids : Sequence[Any]
        Current node identities aligned to ``current_positions``.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, int]
        Aligned previous positions, matched current positions, and match count.
    """
    previous = torch.as_tensor(frame.get("positions"), dtype=torch.float64)
    if previous.ndim != 2 or int(previous.shape[1]) != 2:
        raise ValueError("temporal frame positions must have shape [N, 2]")
    previous_ids = tuple(frame.get("node_ids", frame.get("node_identity", current_ids)))
    if len(previous_ids) != int(previous.shape[0]):
        raise ValueError("previous temporal node identity metadata must match frame positions")
    previous_lookup = {identity: index for index, identity in enumerate(previous_ids)}
    previous_rows: list[int] = []
    current_rows: list[int] = []
    for current_index, identity in enumerate(current_ids):
        if identity in previous_lookup:
            previous_rows.append(previous_lookup[identity])
            current_rows.append(current_index)
    if len(previous_rows) < 2:
        raise ValueError("G5 requires at least two node identities shared with the previous frame")
    prev = previous[previous_rows]
    curr = current_positions[current_rows]
    return _procrustes_align(prev, curr), curr, len(previous_rows)


def _procrustes_align(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Align a point set to a target by similarity Procrustes transform.

    Parameters
    ----------
    source : torch.Tensor
        Source positions with shape ``[N, 2]``.
    target : torch.Tensor
        Target positions with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Source positions aligned into target coordinates.
    """
    src_centered = source - source.mean(dim=0, keepdim=True)
    tgt_centered = target - target.mean(dim=0, keepdim=True)
    src_norm = torch.linalg.vector_norm(src_centered)
    tgt_norm = torch.linalg.vector_norm(tgt_centered)
    if float(src_norm.item()) <= 1e-12 or float(tgt_norm.item()) <= 1e-12:
        return source - source.mean(dim=0, keepdim=True) + target.mean(dim=0, keepdim=True)
    src_unit = src_centered / src_norm
    tgt_unit = tgt_centered / tgt_norm
    u, _singular, vh = torch.linalg.svd(src_unit.T @ tgt_unit)
    rotation = u @ vh
    scale = tgt_norm / src_norm
    return (src_centered @ rotation) * scale + target.mean(dim=0, keepdim=True)


def _normalized_mean_displacement(aligned_previous: torch.Tensor, current: torch.Tensor) -> float:
    """Measure mean displacement after Procrustes alignment.

    Parameters
    ----------
    aligned_previous : torch.Tensor
        Previous positions aligned to current coordinates with shape ``[N, 2]``.
    current : torch.Tensor
        Current positions with shape ``[N, 2]``.

    Returns
    -------
    float
        Mean node displacement divided by current RMS radius.
    """
    displacement = torch.linalg.vector_norm(current - aligned_previous, dim=1).mean()
    centered = current - current.mean(dim=0, keepdim=True)
    scale = torch.linalg.vector_norm(centered, dim=1).mean()
    return float((displacement / torch.clamp(scale, min=1e-12)).item())


def _change_proportionality_score(observed: float, graph_change: float) -> float:
    """Score proportionality between drawn displacement and graph change.

    Parameters
    ----------
    observed : float
        Normalized Procrustes-aligned displacement.
    graph_change : float
        Declared ground-truth graph-change magnitude.

    Returns
    -------
    float
        Score in ``[0, 1]`` penalizing thrash and false stability.
    """
    if graph_change <= G5_CHANGE_EPSILON:
        return math.exp(-observed / G5_ZERO_CHANGE_SCALE)
    ratio = (observed + G5_CHANGE_EPSILON) / (graph_change + G5_CHANGE_EPSILON)
    return math.exp(-abs(math.log(ratio)))


def _temporal_quality_decay(
    frame: Mapping[str, Any],
    meta: Mapping[str, Any],
) -> Tuple[float, bool]:
    """Return graded quality-band decay for one temporal frame.

    Parameters
    ----------
    frame : Mapping[str, Any]
        Previous-frame metadata.
    meta : Mapping[str, Any]
        Current row metadata with the optional quality delta.

    Returns
    -------
    Tuple[float, bool]
        Decay in ``[0, 1]`` and whether the band was violated.
    """
    quality = frame.get("quality", frame.get("v3_core_score", meta.get("frame_v3_core_score", 1.0)))
    best = 1.0
    threshold = (1.0 - float(meta.get("temporal_quality_delta", G5_QUALITY_BAND_DELTA))) * (
        float(best)
    )
    if threshold <= 0.0:
        return 1.0, False
    decay = max(0.0, min(1.0, float(quality) / threshold))
    return decay, decay < 1.0


def _declared_port_records(
    meta: Mapping[str, Any],
    edge_index: torch.Tensor,
) -> Tuple[Dict[str, Any], ...]:
    """Normalize declared port metadata to endpoint records.

    Parameters
    ----------
    meta : Mapping[str, Any]
        Declared graph metadata.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.

    Returns
    -------
    Tuple[Dict[str, Any], ...]
        Port endpoint records with ``edge``, ``endpoint``, ``node``, and
        optional ``side``/``order``/``position`` keys.
    """
    ports = meta.get("ports")
    if ports is not None:
        return tuple(dict(port) for port in ports)
    records: list[Dict[str, Any]] = []
    side_data = meta.get("port_sides")
    if isinstance(side_data, Mapping):
        for raw_edge, value in side_data.items():
            edge = int(raw_edge)
            if isinstance(value, Mapping):
                for endpoint in ("source", "target"):
                    if endpoint in value:
                        records.append(
                            {"edge": edge, "endpoint": endpoint, "side": value[endpoint]}
                        )
            else:
                source_side, target_side = value
                records.append({"edge": edge, "endpoint": "source", "side": source_side})
                records.append({"edge": edge, "endpoint": "target", "side": target_side})
    order_data = meta.get("port_order")
    if isinstance(order_data, Mapping):
        records = [_complete_port_record(record, edge_index) for record in records]
        for record in records:
            key = (record.get("node"), record.get("side"))
            record["order"] = order_data.get(key, order_data.get(str(key), record.get("order")))
    return tuple(records)


def _port_compliance_score(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    ports: Sequence[Mapping[str, Any]],
    curves: Optional[Sequence[Any]],
) -> Dict[str, Any]:
    """Score hard port side/order/position compliance.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    ports : Sequence[Mapping[str, Any]]
        Declared port endpoint records.
    curves : Optional[Sequence[Any]]
        Optional routed curves aligned to ``edge_index``.

    Returns
    -------
    Dict[str, Any]
        Compliance score and violation diagnostics.
    """
    checks = 0
    satisfied = 0
    side_violations = 0
    severe_side_violations = 0
    normalized_ports = [_complete_port_record(port, edge_index) for port in ports]
    for port in normalized_ports:
        side = port.get("side")
        if side is not None:
            checks += 1
            score = _port_side_alignment_score(positions, edge_index, port, curves)
            ok = score >= G7_PORT_SIDE_COSINE_THRESHOLD
            satisfied += int(ok)
            if not ok:
                side_violations += 1
                if score <= 0.0:
                    severe_side_violations += 1
        if port.get("position") is not None:
            checks += 1
            if _port_position_satisfied(positions, port):
                satisfied += 1
    order_checks, order_satisfied = _port_order_satisfaction(
        positions,
        edge_index,
        normalized_ports,
        curves,
    )
    checks += order_checks
    satisfied += order_satisfied
    hard = 1.0 if checks == 0 else satisfied / checks
    return {
        "hard_compliance": float(hard),
        "compliance_checks": checks,
        "compliance_satisfied": satisfied,
        "side_violation_count": side_violations,
        "severe_side_violation_count": severe_side_violations,
        "order_checks": order_checks,
        "order_satisfied": order_satisfied,
    }


def _complete_port_record(port: Mapping[str, Any], edge_index: torch.Tensor) -> Dict[str, Any]:
    """Fill endpoint and node fields for a declared port record.

    Parameters
    ----------
    port : Mapping[str, Any]
        Raw port metadata.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.

    Returns
    -------
    Dict[str, Any]
        Completed port record.
    """
    record = dict(port)
    edge = int(record.get("edge", record.get("edge_id", 0)))
    record["edge"] = edge
    endpoint = str(record.get("endpoint", "source"))
    if "node" not in record and 0 <= edge < int(edge_index.shape[1]):
        endpoint = "target" if endpoint in {"target", "t"} else "source"
        row = 1 if endpoint == "target" else 0
        record["node"] = int(edge_index[row, edge].item())
        record["endpoint"] = endpoint
    return record


def _port_side_alignment_score(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    port: Mapping[str, Any],
    curves: Optional[Sequence[Any]],
) -> float:
    """Return cosine alignment between route approach and declared side.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    port : Mapping[str, Any]
        Completed port record.
    curves : Optional[Sequence[Any]]
        Optional routed curves.

    Returns
    -------
    float
        Cosine-like score where ``1`` is exactly orthogonal to the declared
        side and ``-1`` exits through the opposite side.
    """
    node = int(port["node"])
    edge = int(port["edge"])
    side_normal = _port_side_normal(str(port["side"]))
    vector = _port_endpoint_vector(
        positions,
        edge_index,
        node,
        edge,
        str(port.get("endpoint")),
        curves,
    )
    length = math.hypot(vector[0], vector[1])
    if length <= 1e-12:
        return -1.0
    return (vector[0] * side_normal[0] + vector[1] * side_normal[1]) / length


def _port_endpoint_vector(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    node: int,
    edge: int,
    endpoint: str,
    curves: Optional[Sequence[Any]],
) -> Tuple[float, float]:
    """Return the vector from a node center toward the edge's port approach.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node : int
        Port node id.
    edge : int
        Edge id.
    endpoint : str
        ``source`` or ``target``.
    curves : Optional[Sequence[Any]]
        Optional routed curves.

    Returns
    -------
    Tuple[float, float]
        Endpoint approach vector.
    """
    if curves is not None and 0 <= edge < len(curves):
        points = _curve_points(curves[edge])
        origin = (float(positions[node, 0].item()), float(positions[node, 1].item()))
        if endpoint == "target":
            points = tuple(reversed(points))
        sample = _route_sample_at_arc_length(origin, points)
        return (sample[0] - origin[0], sample[1] - origin[1])
    other_row = 1 if endpoint != "target" else 0
    other = int(edge_index[other_row, edge].item())
    return (
        float(positions[other, 0].item() - positions[node, 0].item()),
        float(positions[other, 1].item() - positions[node, 1].item()),
    )


def _route_sample_at_arc_length(
    origin: Tuple[float, float],
    points: Sequence[Tuple[float, float]],
) -> Tuple[float, float]:
    """Return a route point sampled after a canonical arc length from a node.

    Parameters
    ----------
    origin : Tuple[float, float]
        Node center.
    points : Sequence[Tuple[float, float]]
        Route points ordered away from the endpoint being tested.

    Returns
    -------
    Tuple[float, float]
        Sampled route point in the same frame as ``origin`` and ``points``.
    """
    if not points:
        return origin
    target_distance = G7_ROUTE_SAMPLE_CANONICAL_DISTANCE
    previous = (float(points[0][0]), float(points[0][1]))
    travelled = math.hypot(previous[0] - origin[0], previous[1] - origin[1])
    for point in points[1:]:
        current = (float(point[0]), float(point[1]))
        segment = math.hypot(current[0] - previous[0], current[1] - previous[1])
        if segment <= 1e-12:
            previous = current
            continue
        if travelled + segment >= target_distance:
            fraction = max(0.0, min(1.0, (target_distance - travelled) / segment))
            sample = (
                previous[0] + fraction * (current[0] - previous[0]),
                previous[1] + fraction * (current[1] - previous[1]),
            )
            return sample
        travelled += segment
        previous = current
    return previous


def _port_side_normal(side: str) -> Tuple[float, float]:
    """Return the outward normal vector for a port side.

    Parameters
    ----------
    side : str
        Side code such as ``E``, ``W``, ``N``, or ``S``.

    Returns
    -------
    Tuple[float, float]
        Unit side normal.
    """
    key = side.strip().lower()
    if key in {"e", "east", "right"}:
        return (1.0, 0.0)
    if key in {"w", "west", "left"}:
        return (-1.0, 0.0)
    if key in {"n", "north", "top"}:
        return (0.0, 1.0)
    if key in {"s", "south", "bottom"}:
        return (0.0, -1.0)
    raise ValueError(f"unsupported port side: {side}")


def _port_position_satisfied(positions: torch.Tensor, port: Mapping[str, Any]) -> bool:
    """Return whether an explicit port point lies on its declared side.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    port : Mapping[str, Any]
        Completed port record.

    Returns
    -------
    bool
        ``True`` when the declared point is on the declared side half-plane.
    """
    side = port.get("side")
    if side is None:
        return True
    node = int(port["node"])
    point = port["position"]
    normal = _port_side_normal(str(side))
    delta = (
        float(point[0]) - float(positions[node, 0].item()),
        float(point[1]) - float(positions[node, 1].item()),
    )
    return delta[0] * normal[0] + delta[1] * normal[1] >= 0.0


def _port_order_satisfaction(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    ports: Sequence[Mapping[str, Any]],
    curves: Optional[Sequence[Any]],
) -> Tuple[int, int]:
    """Score declared order along each node side.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    ports : Sequence[Mapping[str, Any]]
        Completed port records.
    curves : Optional[Sequence[Any]]
        Optional routed curves aligned to ``edge_index``.

    Returns
    -------
    Tuple[int, int]
        Number of order checks and satisfied checks.
    """
    buckets: Dict[Tuple[int, str], list[Tuple[float, int]]] = {}
    for port in ports:
        if port.get("order") is None or port.get("side") is None:
            continue
        node = int(port["node"])
        side = str(port["side"])
        coordinate = _port_order_coordinate(positions, edge_index, port, curves)
        if coordinate is None:
            continue
        buckets.setdefault((node, side), []).append((coordinate, int(port["order"])))
    checks = 0
    satisfied = 0
    for values in buckets.values():
        if len(values) < 2:
            continue
        if (
            max(coordinate for coordinate, _order in values)
            - min(coordinate for coordinate, _order in values)
            <= 1e-12
        ):
            continue
        checks += len(values)
        drawn = [order for _coord, order in sorted(values)]
        satisfied += len(values) if drawn == sorted(drawn) else 0
    return checks, satisfied


def _port_order_coordinate(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    port: Mapping[str, Any],
    curves: Optional[Sequence[Any]],
) -> Optional[float]:
    """Return a side tangent coordinate for port ordering.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    port : Mapping[str, Any]
        Completed port record.
    curves : Optional[Sequence[Any]]
        Optional routed curves aligned to ``edge_index``.

    Returns
    -------
    Optional[float]
        Tangent coordinate along the declared side, or ``None`` when no real
        port geometry is resolvable.
    """
    point = port.get("position")
    if point is not None:
        x_value, y_value = float(point[0]), float(point[1])
    elif curves is not None:
        edge = int(port["edge"])
        if edge < 0 or edge >= len(curves):
            return None
        node = int(port["node"])
        points = _curve_points(curves[edge])
        if str(port.get("endpoint")) == "target":
            points = tuple(reversed(points))
        origin = (float(positions[node, 0].item()), float(positions[node, 1].item()))
        x_value, y_value = _route_sample_at_arc_length(origin, points)
    else:
        return None
    side = str(port.get("side", "E")).lower()
    return x_value if side in {"n", "north", "top", "s", "south", "bottom"} else y_value


def _g7_routed_bundle_scores(
    positions: torch.Tensor,
    edges: torch.Tensor,
    sizes: torch.Tensor,
    curves: Sequence[Any],
    ports: Sequence[Mapping[str, Any]],
    meta: Mapping[str, Any],
    group_cap: float,
    *,
    canonicalization: IntrinsicRulerCanonicalization,
) -> Tuple[Tuple[str, str, float, Dict[str, Any]], ...]:
    """Return G7 routed bundle facet tuples.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edges : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.
    curves : Sequence[Any]
        Routed curves aligned to edges.
    ports : Sequence[Mapping[str, Any]]
        Declared port records.
    meta : Mapping[str, Any]
        Declared graph metadata.
    group_cap : float
        Severe-violation cap applied to G7 facets.
    canonicalization : IntrinsicRulerCanonicalization
        Shared per-drawing ruler used to canonicalize the frozen V2 composite seam.

    Returns
    -------
    Tuple[Tuple[str, str, float, Dict[str, Any]], ...]
        Routed facet tuples.
    """
    labels = meta.get("edge_labels", meta.get("routed_labels"))
    label_positions = meta.get("label_positions")
    canonical_positions = _canonicalize_tensor(positions, canonicalization)
    canonical_sizes = _canonicalize_tensor(sizes, canonicalization)
    canonical_curves = _canonicalize_curves(curves, canonicalization)
    canonical_label_positions = _canonicalize_label_positions(label_positions, canonicalization)
    drawing = composite_drawing(
        canonical_positions,
        edges,
        canonical_sizes,
        canonical_curves,
        label_positions=canonical_label_positions,
        edge_labels=labels,
        seed=0,
    )
    routed_crossings = routed_crossing_rate(curves, edges, seed=0)
    bends = bend_count(curves)
    port_ar = port_angular_resolution(canonical_curves, edges)
    side_approach = _port_approach_angle_score(canonical_positions, edges, ports, canonical_curves)
    terminal = _terminal_congestion_score(
        canonical_positions,
        canonical_sizes,
        edges,
        ports,
        canonical_curves,
    )
    label_terms = [
        drawing.get("drawing_term_label_node"),
        drawing.get("drawing_term_label_label"),
    ]
    quality_terms = [
        side_approach,
        drawing.get("drawing_term_crossing"),
        drawing.get("drawing_term_edge_node"),
        *[term for term in label_terms if term is not None],
        terminal,
    ]
    economy_terms = [
        drawing.get("drawing_term_bend"),
        _flow_consistency_for_routed_row(positions, edges, meta),
    ]
    quality = min(group_cap, _mean_score_terms(quality_terms))
    economy = min(group_cap, _mean_score_terms(economy_terms))
    return (
        (
            "G7_routed_curve_quality",
            "routed_crossings_edge_node_labels_terminal_quality",
            quality,
            {
                "slot": "G7_routed_quality",
                "drawing_term_crossing": drawing.get("drawing_term_crossing"),
                "drawing_term_edge_node": drawing.get("drawing_term_edge_node"),
                "drawing_term_label_node": drawing.get("drawing_term_label_node"),
                "drawing_term_label_label": drawing.get("drawing_term_label_label"),
                "port_side_approach_score": side_approach,
                "port_angular_resolution_score": port_ar.get("port_angular_resolution_score"),
                "terminal_congestion_score": terminal,
                "routed_crossing_rate": routed_crossings["routed_crossing_rate"],
                "composite_drawing_reuse": drawing,
                "normalization": ("reused frozen composite_drawing in canonical node-height frame"),
                "invariance": (
                    "translation_invariant;rotation_reflection_axis_anchored_aabb;"
                    "position_only_scale_sensitive_by_design;unit_scale_invariant_exact_via_"
                    "intrinsic_ruler_canonicalization;anisotropic_not_invariant;"
                    "node_relabel_invariant"
                ),
                "f3_note": (
                    "G7 routed-quality compared projection-off; canonicalization delta "
                    "published per the unit-invariance ruling"
                ),
                **_canonicalization_metadata(canonicalization),
            },
        ),
        (
            "G7_routed_bend_terminal_economy",
            "routed_bend_dogleg_terminal_flow_economy",
            economy,
            {
                "slot": "G7_routed_economy",
                "drawing_term_bend": drawing.get("drawing_term_bend"),
                "bend_mean_per_edge": bends["bend_mean_per_edge"],
                "terminal_congestion_score": terminal,
                "flow_consistency_score": _flow_consistency_for_routed_row(positions, edges, meta),
                "composite_drawing_reuse": drawing,
            },
        ),
    )


def _port_approach_angle_score(
    positions: torch.Tensor,
    edges: torch.Tensor,
    ports: Sequence[Mapping[str, Any]],
    curves: Sequence[Any],
) -> float:
    """Score routed approach angles against declared port sides.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edges : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    ports : Sequence[Mapping[str, Any]]
        Declared port records.
    curves : Sequence[Any]
        Routed curves aligned to edges.

    Returns
    -------
    float
        Mean clipped side-normal alignment in ``[0, 1]``.
    """
    scores = []
    for port in (_complete_port_record(port, edges) for port in ports):
        if port.get("side") is None:
            continue
        scores.append(max(0.0, _port_side_alignment_score(positions, edges, port, curves)))
    return float(np.mean(scores)) if scores else 1.0


def _terminal_congestion_score(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    edges: torch.Tensor,
    ports: Sequence[Mapping[str, Any]],
    curves: Sequence[Any],
) -> float:
    """Score same-side terminal separation with node-size anchoring.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.
    edges : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    ports : Sequence[Mapping[str, Any]]
        Declared port records.
    curves : Sequence[Any]
        Routed curves aligned to edges.

    Returns
    -------
    float
        Mean terminal separation score in ``[0, 1]``.
    """
    buckets: Dict[Tuple[int, str], list[float]] = {}
    for port in (_complete_port_record(port, edges) for port in ports):
        if port.get("side") is None:
            continue
        node = int(port["node"])
        side = str(port["side"])
        vector = _port_endpoint_vector(
            positions,
            edges,
            node,
            int(port["edge"]),
            str(port["endpoint"]),
            curves,
        )
        tangent_coord = (
            vector[1] if side.lower() in {"e", "east", "right", "w", "west", "left"} else vector[0]
        )
        buckets.setdefault((node, side), []).append(float(tangent_coord))
    scores = []
    for node_side, coords in buckets.items():
        if len(coords) < 2:
            continue
        node = node_side[0]
        coords = sorted(coords)
        min_sep = min(abs(after - before) for before, after in zip(coords, coords[1:]))
        node_diag = float(torch.linalg.vector_norm(sizes[node]).item())
        target = max(1e-12, G7_TERMINAL_SEPARATION_FRACTION * node_diag)
        scores.append(min(1.0, min_sep / target))
    return float(np.mean(scores)) if scores else 1.0


def _flow_consistency_for_routed_row(
    positions: torch.Tensor,
    edges: torch.Tensor,
    meta: Mapping[str, Any],
) -> float:
    """Return directed-flow consistency for routed data-flow rows.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edges : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    meta : Mapping[str, Any]
        Declared graph metadata.

    Returns
    -------
    float
        Directed-flow score when a flow direction is declared, otherwise 1.0.
    """
    if meta.get("flow_direction") is None and meta.get("direction") is None:
        return 1.0
    return float(
        directed_flow_score(
            positions,
            edges,
            direction=str(meta.get("flow_direction", meta.get("direction", "TB"))),
        )["directed_flow_score"]
    )


def _mean_score_terms(values: Sequence[Any]) -> float:
    """Average finite normalized score terms.

    Parameters
    ----------
    values : Sequence[Any]
        Optional score-like values.

    Returns
    -------
    float
        Mean finite value in ``[0, 1]`` or ``1.0`` when no terms apply.
    """
    finite = [max(0.0, min(1.0, float(value))) for value in values if value is not None]
    return float(np.mean(finite)) if finite else 1.0


def _curve_points(curve: Any) -> Tuple[Tuple[float, float], ...]:
    """Return representative points from a routed curve.

    Parameters
    ----------
    curve : Any
        BezierCurve-compatible routed curve.

    Returns
    -------
    Tuple[Tuple[float, float], ...]
        Waypoints or endpoints.
    """
    waypoints = getattr(curve, "waypoints", None)
    if waypoints is not None:
        return tuple((float(x), float(y)) for x, y in waypoints)
    return (tuple(curve.p0), tuple(curve.p1))


def _optional_bool_tensor(value: Any, expected_len: int) -> Optional[torch.Tensor]:
    """Normalize an optional boolean mask from metadata.

    Parameters
    ----------
    value : Any
        Optional sequence-like mask.
    expected_len : int
        Required mask length.

    Returns
    -------
    Optional[torch.Tensor]
        Boolean tensor or ``None``.
    """
    if value is None:
        return None
    mask = torch.as_tensor(value, dtype=torch.bool)
    if int(mask.numel()) != expected_len:
        raise ValueError("back_edge_mask must have length E")
    return mask.reshape(expected_len)


def _drop_masked_edges(edge_index: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Drop edges marked as declared feedback/back edges.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    mask : Optional[torch.Tensor]
        Boolean mask with shape ``[E]`` where ``True`` means excluded from DAG
        hierarchy computations.

    Returns
    -------
    torch.Tensor
        Edge tensor with masked columns removed.
    """
    if mask is None:
        return edge_index
    return edge_index[:, ~mask]


def _metadata_sequence(
    meta: Mapping[str, Any],
    keys: Sequence[str],
) -> Optional[Sequence[Any]]:
    """Return the first sequence metadata value found under candidate keys.

    Parameters
    ----------
    meta : Mapping[str, Any]
        Declared graph metadata.
    keys : Sequence[str]
        Candidate metadata keys.

    Returns
    -------
    Optional[Sequence[Any]]
        Sequence value when present.
    """
    for key in keys:
        value = meta.get(key)
        if value is not None:
            return value
    return None


def _declared_clusters(meta: Mapping[str, Any], num_nodes: int) -> Dict[str, Tuple[int, ...]]:
    """Normalize declared cluster metadata to a membership mapping.

    Parameters
    ----------
    meta : Mapping[str, Any]
        Declared graph metadata.
    num_nodes : int
        Number of nodes in the drawing.

    Returns
    -------
    Dict[str, Tuple[int, ...]]
        Cluster members keyed by stable cluster name.
    """
    value = meta.get("clusters")
    if isinstance(value, Mapping):
        return {
            str(name): tuple(sorted({int(index) for index in members}))
            for name, members in value.items()
        }
    if value is not None:
        labels = list(value)
        if len(labels) != num_nodes:
            raise ValueError("cluster label sequence must have length N")
        clusters: Dict[str, list[int]] = {}
        for index, label in enumerate(labels):
            clusters.setdefault(str(label), []).append(index)
        return {name: tuple(indices) for name, indices in clusters.items()}
    parents = meta.get("cluster_parents")
    if isinstance(parents, Mapping):
        return {str(name): tuple() for name in parents}
    raise ValueError("G2 score called without declared clusters")


def _declared_cluster_parents(meta: Mapping[str, Any]) -> Dict[str, Optional[str]]:
    """Normalize optional cluster-parent metadata.

    Parameters
    ----------
    meta : Mapping[str, Any]
        Declared graph metadata.

    Returns
    -------
    Dict[str, Optional[str]]
        Parent lookup keyed by cluster name.
    """
    parents = meta.get("cluster_parents")
    if not isinstance(parents, Mapping):
        return {}
    return {str(name): None if parent is None else str(parent) for name, parent in parents.items()}


def _declared_cluster_labels(meta: Mapping[str, Any]) -> Dict[str, str]:
    """Normalize optional explicit cluster labels.

    Parameters
    ----------
    meta : Mapping[str, Any]
        Declared graph metadata.

    Returns
    -------
    Dict[str, str]
        Label text keyed by cluster name.
    """
    labels = meta.get("cluster_labels")
    if not isinstance(labels, Mapping):
        return {}
    return {str(name): str(label) for name, label in labels.items()}


def _has_cluster_with_min_members(
    clusters: Mapping[str, Sequence[int]],
    *,
    min_members: int,
) -> bool:
    """Return whether any declared cluster has enough explicit members.

    Parameters
    ----------
    clusters : Mapping[str, Sequence[int]]
        Declared cluster membership.
    min_members : int
        Minimum required member count.

    Returns
    -------
    bool
        ``True`` when at least one cluster has ``min_members`` unique nodes.
    """
    return any(
        len({int(index) for index in members}) >= min_members for members in clusters.values()
    )


def _cluster_labels_from_clusters(
    clusters: Mapping[str, Sequence[int]],
    num_nodes: int,
) -> np.ndarray:
    """Return one declared cluster label per node for CQ-HAC ARI.

    Parameters
    ----------
    clusters : Mapping[str, Sequence[int]]
        Declared cluster membership.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    numpy.ndarray
        Integer labels with shape ``[N]``.
    """
    labels = np.full(num_nodes, -1, dtype=np.int64)
    for cluster_id, name in enumerate(sorted(clusters)):
        for index in clusters[name]:
            node = int(index)
            if 0 <= node < num_nodes and labels[node] < 0:
                labels[node] = cluster_id
    next_label = len(clusters)
    for index in range(num_nodes):
        if labels[index] < 0:
            labels[index] = next_label
            next_label += 1
    return labels


def _g2_slot_a_scores(
    positions: torch.Tensor,
    edges: torch.Tensor,
    sizes: torch.Tensor,
    clusters: Mapping[str, Sequence[int]],
    cluster_parents: Mapping[str, Optional[str]],
    cluster_labels: Mapping[str, str],
    *,
    canonicalization: IntrinsicRulerCanonicalization,
) -> Tuple[Tuple[str, str, Optional[float], float, Dict[str, Any]], ...]:
    """Return reused G2 Slot A AABB hygiene scores.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edges : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    sizes : torch.Tensor
        Node box sizes with shape ``[N, 2]``.
    clusters : Mapping[str, Sequence[int]]
        Declared cluster membership.
    cluster_parents : Mapping[str, Optional[str]]
        Declared cluster nesting parents.
    cluster_labels : Mapping[str, str]
        Explicit cluster labels.
    canonicalization : IntrinsicRulerCanonicalization
        Shared per-drawing ruler used to canonicalize the frozen V2 cluster seam.

    Returns
    -------
    Tuple[Tuple[str, str, Optional[float], float, Dict[str, Any]], ...]
        Slot A facet tuples.
    """
    canonical_positions = _canonicalize_tensor(positions, canonicalization)
    canonical_sizes = _canonicalize_tensor(sizes, canonicalization)
    shared_metadata = _canonicalization_metadata(canonicalization)
    exclusion = cluster_exclusion_score(
        canonical_positions,
        canonical_sizes,
        clusters,
        cluster_parents,
        cluster_labels,
    )
    sibling = cluster_sibling_overlap_score(
        canonical_positions, canonical_sizes, clusters, cluster_parents, cluster_labels
    )
    nesting = cluster_nesting_fidelity_score(
        canonical_positions, canonical_sizes, clusters, cluster_parents, cluster_labels
    )
    intrusion = cluster_edge_intrusion_score(
        canonical_positions, edges, canonical_sizes, clusters, cluster_parents, cluster_labels
    )
    labels = cluster_label_occlusion_score(
        canonical_positions,
        canonical_sizes,
        clusters,
        cluster_parents,
        cluster_labels,
    )
    return (
        (
            "G2_cluster_exclusion",
            "cluster_exclusion",
            _optional_score(exclusion["cluster_exclusion_score"]),
            2.0,
            {**exclusion, **shared_metadata},
        ),
        (
            "G2_cluster_sibling_overlap",
            "cluster_sibling_overlap",
            _optional_score(sibling["cluster_sibling_overlap_score"]),
            2.0,
            {**sibling, **shared_metadata},
        ),
        (
            "G2_cluster_nesting_fidelity",
            "cluster_nesting_fidelity",
            _optional_score(nesting["cluster_nesting_fidelity_score"]),
            1.0,
            {**nesting, **shared_metadata},
        ),
        (
            "G2_cluster_edge_intrusion",
            "cluster_edge_intrusion",
            _optional_score(intrusion["cluster_edge_intrusion_score"]),
            1.0,
            {**intrusion, **shared_metadata},
        ),
        (
            "G2_cluster_label_occlusion",
            "cluster_label_occlusion",
            _optional_score(labels["cluster_label_occlusion_score"]),
            1.0,
            {**labels, **shared_metadata},
        ),
    )


def _optional_score(value: Any) -> Optional[float]:
    """Clamp an optional score to ``[0, 1]``.

    Parameters
    ----------
    value : Any
        Raw score value or ``None``.

    Returns
    -------
    Optional[float]
        Clamped score or ``None``.
    """
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def _cluster_log_ratio_compactness(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    clusters: Mapping[str, Sequence[int]],
) -> Tuple[Optional[float], Dict[str, Any]]:
    """Score cluster compactness with a log-ratio sprawl band.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    sizes : torch.Tensor
        Node box sizes with shape ``[N, 2]``.
    clusters : Mapping[str, Sequence[int]]
        Declared cluster membership.

    Returns
    -------
    Tuple[Optional[float], Dict[str, Any]]
        Compactness score and ratio diagnostics.
    """
    boxes_min = positions - sizes / 2.0
    boxes_max = positions + sizes / 2.0
    ratios: list[float] = []
    for members in clusters.values():
        valid = [int(index) for index in members if 0 <= int(index) < int(positions.shape[0])]
        if len(valid) < 2:
            continue
        lower = boxes_min[valid].min(dim=0).values
        upper = boxes_max[valid].max(dim=0).values
        cluster_area = float(torch.prod(torch.clamp(upper - lower, min=1e-12)).item())
        member_area = float(torch.prod(torch.clamp(sizes[valid], min=1e-12), dim=1).sum().item())
        if member_area > 1e-12:
            ratios.append(cluster_area / member_area)
    if not ratios:
        return None, {"cluster_compactness_log_mean_ratio": None, "sample_count": 0}
    # Scores intentionally floor at 0 beyond the 100x log-band cutoff; 100x is maximally bad.
    scores = [
        1.0
        if ratio <= COMPACTNESS_RATIO_PLATEAU_HI
        else max(
            0.0,
            1.0
            - math.log(ratio / COMPACTNESS_RATIO_PLATEAU_HI)
            / math.log(COMPACTNESS_LOG_DECAY * 100.0 / COMPACTNESS_RATIO_PLATEAU_HI),
        )
        for ratio in ratios
    ]
    return float(np.mean(scores)), {
        "cluster_compactness_log_mean_ratio": float(np.mean(ratios)),
        "cluster_compactness_log_ratios": tuple(float(ratio) for ratio in ratios),
        "sample_count": len(ratios),
        "plateau_hi": COMPACTNESS_RATIO_PLATEAU_HI,
    }


def _tree_forest(edge_index: torch.Tensor, num_nodes: int, meta: Mapping[str, Any]) -> TreeForest:
    """Normalize and validate a rooted tree or forest.

    Parameters
    ----------
    edge_index : torch.Tensor
        Directed parent-child edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    meta : Mapping[str, Any]
        Declared tree metadata.

    Returns
    -------
    TreeForest
        Normalized forest structure.
    """
    raw_edges = [(int(source), int(target)) for source, target in edge_index.t().tolist()]
    children: Dict[int, list[int]] = {node: [] for node in range(num_nodes)}
    parents: Dict[int, Optional[int]] = {node: None for node in range(num_nodes)}
    for source, target in raw_edges:
        children[source].append(target)
        if parents[target] is not None:
            raise ValueError("G4 requires each tree node to have at most one parent")
        parents[target] = source
    roots = _declared_tree_roots(meta, parents)
    for root in roots:
        parents[root] = None
    components: list[Tuple[int, ...]] = []
    depths: Dict[int, int] = {}
    for root in roots:
        queue = [root]
        depths[root] = 0
        component: list[int] = []
        visited: set[int] = set()
        while queue:
            node = queue.pop(0)
            if node in visited:
                raise ValueError("G4 declared tree contains a cycle reachable from a root")
            visited.add(node)
            component.append(node)
            for child in children[node]:
                depths[child] = depths[node] + 1
                queue.append(child)
        components.append(tuple(component))
    if len(depths) != num_nodes:
        missing = sorted(set(range(num_nodes)) - set(depths))
        raise ValueError(f"G4 declared roots do not reach all nodes: {missing}")
    for component_nodes in components:
        component_edges = [
            edge
            for edge in raw_edges
            if edge[0] in set(component_nodes) and edge[1] in set(component_nodes)
        ]
        if not _frozen_is_tree_component(component_edges, len(component_nodes)):
            raise ValueError("G4 requires every declared component to be an undirected tree")
    subtree_nodes: Dict[int, Tuple[int, ...]] = {}
    subtree_leaves: Dict[int, int] = {}
    subtree_hashes: Dict[int, str] = {}
    for root in reversed(roots):
        _fill_subtree(root, children, subtree_nodes, subtree_leaves, subtree_hashes)
    return TreeForest(
        children={node: tuple(values) for node, values in children.items()},
        parents=parents,
        roots=roots,
        depths=depths,
        components=tuple(components),
        subtree_nodes=subtree_nodes,
        subtree_leaves=subtree_leaves,
        subtree_hashes=subtree_hashes,
    )


def _declared_tree_roots(
    meta: Mapping[str, Any],
    parents: Mapping[int, Optional[int]],
) -> Tuple[int, ...]:
    """Return declared or uniquely inferred tree roots.

    Parameters
    ----------
    meta : Mapping[str, Any]
        Declared tree metadata.
    parents : Mapping[int, Optional[int]]
        Parent lookup.

    Returns
    -------
    Tuple[int, ...]
        Root node ids.
    """
    roots_value = meta.get("tree_roots")
    if roots_value is not None:
        return tuple(sorted(int(root) for root in roots_value))
    root_value = meta.get("root")
    if root_value is not None:
        return (int(root_value),)
    return tuple(sorted(node for node, parent in parents.items() if parent is None))


def _frozen_is_tree_component(edges: Sequence[Tuple[int, int]], node_count: int) -> bool:
    """Call the frozen V3 tree-component predicate.

    Parameters
    ----------
    edges : Sequence[Tuple[int, int]]
        Component edges.
    node_count : int
        Component node count.

    Returns
    -------
    bool
        ``True`` when the component is a tree.
    """
    from dagua.eval.ruler_v3 import _is_tree_component

    return _is_tree_component(edges, node_count)


def _fill_subtree(
    node: int,
    children: Mapping[int, Sequence[int]],
    subtree_nodes: Dict[int, Tuple[int, ...]],
    subtree_leaves: Dict[int, int],
    subtree_hashes: Dict[int, str],
) -> None:
    """Fill recursive subtree metadata for one node.

    Parameters
    ----------
    node : int
        Root of the subtree.
    children : Mapping[int, Sequence[int]]
        Children keyed by parent.
    subtree_nodes : Dict[int, Tuple[int, ...]]
        Mutable subtree-node output.
    subtree_leaves : Dict[int, int]
        Mutable leaf-count output.
    subtree_hashes : Dict[int, str]
        Mutable canonical-hash output.

    Returns
    -------
    None
    """
    stack: list[Tuple[int, bool]] = [(node, False)]
    visiting: set[int] = set()
    while stack:
        current, expanded = stack.pop()
        if expanded:
            visiting.discard(current)
            child_values = tuple(children[current])
            if not child_values:
                subtree_nodes[current] = (current,)
                subtree_leaves[current] = 1
                subtree_hashes[current] = "()"
                continue
            nodes = [current]
            leaves = 0
            child_hashes: list[str] = []
            for child in child_values:
                nodes.extend(subtree_nodes[child])
                leaves += subtree_leaves[child]
                child_hashes.append(subtree_hashes[child])
            subtree_nodes[current] = tuple(sorted(nodes))
            subtree_leaves[current] = leaves
            subtree_hashes[current] = "(" + "".join(sorted(child_hashes)) + ")"
            continue
        if current in subtree_nodes:
            continue
        if current in visiting:
            raise ValueError("G4 declared tree contains a subtree cycle")
        visiting.add(current)
        stack.append((current, True))
        for child in reversed(tuple(children[current])):
            stack.append((child, False))


def _g4_layered_scores(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    forest: TreeForest,
) -> Tuple[Tuple[str, str, Optional[float], str, float, float, Dict[str, Any]], ...]:
    """Return layered G4 facet tuples.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.
    forest : TreeForest
        Normalized rooted forest.

    Returns
    -------
    Tuple[Tuple[str, str, Optional[float], str, float, float, Dict[str, Any]], ...]
        Layered facet tuples.
    """
    depth = np.asarray([forest.depths[index] for index in range(int(positions.shape[0]))])
    depth_rho = _spearman(positions[:, 1].numpy(), depth)
    depth_score = None if depth_rho is None else (depth_rho + 1.0) / 2.0
    return (
        (
            "G4_layered_depth_bands",
            "rt_depth_layering",
            depth_score,
            "G4_A_RT_axioms",
            2.0,
            G4_SLOT_A_TOTAL,
            {
                "raw_spearman": depth_rho,
                "invariance": "axis_anchored;not_rotation_or_position_scale_invariant",
            },
        ),
        (
            "G4_layered_parent_centering",
            "rt_parent_centering",
            _parent_centering_score(positions, sizes, forest),
            "G4_A_RT_axioms",
            2.0,
            G4_SLOT_A_TOTAL,
            {"invariance": "axis_anchored;not_rotation_or_position_scale_invariant"},
        ),
        (
            "G4_layered_sibling_order",
            "rt_sibling_order_preservation",
            _sibling_order_score(positions[:, 0].numpy(), forest),
            "G4_A_RT_axioms",
            1.0,
            G4_SLOT_A_TOTAL,
            {"invariance": "axis_anchored_order;not_rotation_invariant"},
        ),
        (
            "G4_layered_subtree_congruence",
            "rt_subtree_congruence",
            _subtree_congruence_score(positions, forest),
            "G4_B_tidiness",
            2.0,
            G4_SLOT_B_TOTAL,
            {},
        ),
        (
            "G4_layered_small_subtree_distribution",
            "rt_small_subtree_distribution",
            _small_subtree_distribution_score(positions, forest),
            "G4_B_tidiness",
            1.0,
            G4_SLOT_B_TOTAL,
            {"invariance": "axis_anchored;not_rotation_or_position_scale_invariant"},
        ),
        (
            "G4_layered_contour_separation",
            "rt_subtree_contour_separation",
            _subtree_contour_separation_score(positions, sizes, forest),
            "G4_B_tidiness",
            1.0,
            G4_SLOT_B_TOTAL,
            {},
        ),
        (
            "G4_layered_width_ratio",
            "rt_width_vs_bjl_reference_ratio",
            _tree_extent_ratio_score(positions, sizes, forest, radial=False),
            "G4_B_tidiness",
            1.0,
            G4_SLOT_B_TOTAL,
            {"frozen_normalizer": "dagua.eval.ruler_v3._tree_reference_extent"},
        ),
    )


def _g4_radial_scores(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    forest: TreeForest,
) -> Tuple[Tuple[str, str, Optional[float], str, float, float, Dict[str, Any]], ...]:
    """Return radial G4 facet tuples.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.
    forest : TreeForest
        Normalized rooted forest.

    Returns
    -------
    Tuple[Tuple[str, str, Optional[float], str, float, float, Dict[str, Any]], ...]
        Radial facet tuples.
    """
    radii = _node_radii(positions, forest)
    depth = np.asarray([forest.depths[index] for index in range(int(positions.shape[0]))])
    radius_rho = _spearman(radii, depth)
    radius_score = None if radius_rho is None else (radius_rho + 1.0) / 2.0
    angles = _node_angles(positions, forest)
    return (
        (
            "G4_radial_depth_monotonicity",
            "radial_depth_monotonicity",
            radius_score,
            "G4_A_radial_axioms",
            2.0,
            G4_SLOT_A_TOTAL,
            {"raw_spearman": radius_rho},
        ),
        (
            "G4_radial_angular_allocation",
            "radial_angular_sector_leaf_allocation",
            _angular_allocation_score(angles, forest),
            "G4_A_radial_axioms",
            2.0,
            G4_SLOT_A_TOTAL,
            {},
        ),
        (
            "G4_radial_circular_order",
            "radial_circular_order_preservation",
            _sibling_circular_order_score(angles, forest),
            "G4_A_radial_axioms",
            1.0,
            G4_SLOT_A_TOTAL,
            {},
        ),
        (
            "G4_radial_angular_overlap",
            "radial_no_angular_overlap",
            _angular_overlap_score(angles, forest),
            "G4_B_radial_tidiness",
            2.0,
            G4_SLOT_B_TOTAL,
            {
                "fair_share_floor_note": (
                    "full-circle row-fair angular floor can ceiling perfect multi-level radial "
                    "drawings below 1.0 by design"
                )
            },
        ),
        (
            "G4_radial_subtree_congruence",
            "radial_subtree_congruence",
            _subtree_congruence_score(positions, forest),
            "G4_B_radial_tidiness",
            1.0,
            G4_SLOT_B_TOTAL,
            {},
        ),
        (
            "G4_radial_small_subtree_distribution",
            "radial_small_subtree_distribution",
            _small_subtree_distribution_score(positions, forest),
            "G4_B_radial_tidiness",
            1.0,
            G4_SLOT_B_TOTAL,
            {},
        ),
        (
            "G4_radial_area_ratio",
            "radial_bounded_area_vs_reference_ratio",
            _tree_extent_ratio_score(positions, sizes, forest, radial=True),
            "G4_B_radial_tidiness",
            1.0,
            G4_SLOT_B_TOTAL,
            {"frozen_normalizer": "dagua.eval.ruler_v3._radial_reference_extent"},
        ),
    )


def _parent_centering_score(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    forest: TreeForest,
) -> float:
    """Score parent x-coordinate centering over child centroids.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    sizes : torch.Tensor
        Node box sizes with shape ``[N, 2]``.
    forest : TreeForest
        Normalized rooted forest.

    Returns
    -------
    float
        Continuous score in ``[0, 1]``.
    """
    penalties: list[float] = []
    for parent, children in forest.children.items():
        if not children:
            continue
        child_center = float(positions[list(children), 0].mean().item())
        child_values = positions[list(children), 0].numpy()
        family_span = (
            _position_span(child_values)
            if len(children) > 1
            else _intrinsic_ruler_canonicalization(sizes).scale
        )
        penalties.append(
            min(1.0, abs(float(positions[parent, 0].item()) - child_center) / family_span)
        )
    return 1.0 - float(np.mean(penalties)) if penalties else 1.0


def _sibling_order_score(axis_values: np.ndarray, forest: TreeForest) -> float:
    """Score canonical sibling order against a drawn scalar order.

    Parameters
    ----------
    axis_values : numpy.ndarray
        Drawn scalar coordinate with shape ``[N]``.
    forest : TreeForest
        Normalized rooted forest.

    Returns
    -------
    float
        One minus the mean normalized inversion count.
    """
    penalties: list[float] = []
    for children in forest.children.values():
        count = len(children)
        if count < 2:
            continue
        inversions = 0
        total = count * (count - 1) // 2
        for left_index, left in enumerate(children):
            for right in children[left_index + 1 :]:
                if axis_values[int(left)] > axis_values[int(right)]:
                    inversions += 1
        penalties.append(inversions / max(1, total))
    return 1.0 - float(np.mean(penalties)) if penalties else 1.0


def _sibling_circular_order_score(angles: np.ndarray, forest: TreeForest) -> float:
    """Score sibling circular order while allowing the 0/2pi cut to move.

    Parameters
    ----------
    angles : numpy.ndarray
        Node polar angles with shape ``[N]``.
    forest : TreeForest
        Normalized rooted forest.

    Returns
    -------
    float
        One minus the minimum normalized inversion count over circular cuts.
    """
    penalties: list[float] = []
    for children in forest.children.values():
        count = len(children)
        if count < 2:
            continue
        drawn = [
            child for _angle, child in sorted((angles[int(child)], child) for child in children)
        ]
        declared = list(children)
        order_rank = {child: index for index, child in enumerate(declared)}
        best = 1.0
        total = count * (count - 1) // 2
        for offset in range(count):
            rotated = drawn[offset:] + drawn[:offset]
            inversions = 0
            ranks = [order_rank[child] for child in rotated]
            for left_index, left_rank in enumerate(ranks):
                for right_rank in ranks[left_index + 1 :]:
                    if left_rank > right_rank:
                        inversions += 1
            best = min(best, inversions / max(1, total))
        penalties.append(best)
    return 1.0 - float(np.mean(penalties)) if penalties else 1.0


def _subtree_congruence_score(positions: torch.Tensor, forest: TreeForest) -> float:
    """Score congruent placement for identical canonical subtree classes.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    forest : TreeForest
        Normalized rooted forest.

    Returns
    -------
    float
        Shape-congruence score in ``[0, 1]``.
    """
    classes: Dict[str, list[int]] = {}
    for node, hash_value in forest.subtree_hashes.items():
        if len(forest.subtree_nodes[node]) > 1:
            classes.setdefault(hash_value, []).append(node)
    penalties: list[float] = []
    for nodes in classes.values():
        if len(nodes) < 2:
            continue
        signatures = [_subtree_signature(positions, forest, node) for node in nodes]
        reference = signatures[0]
        for signature in signatures[1:]:
            if reference.shape != signature.shape:
                continue
            residual = min(
                _normalized_shape_residual(reference, signature),
                _normalized_shape_residual(reference, signature * np.array([-1.0, 1.0])),
            )
            penalties.append(min(1.0, residual))
    return 1.0 - float(np.mean(penalties)) if penalties else 1.0


def _subtree_signature(positions: torch.Tensor, forest: TreeForest, node: int) -> np.ndarray:
    """Return a scale-normalized ordered subtree point signature.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    forest : TreeForest
        Normalized rooted forest.
    node : int
        Subtree root.

    Returns
    -------
    numpy.ndarray
        Relative points with shape ``[K, 2]``.
    """
    nodes = list(forest.subtree_nodes[node])
    points = positions[nodes].numpy() - positions[node].numpy()
    scale = max(float(np.linalg.norm(points, axis=1).max(initial=0.0)), 1e-12)
    return points / scale


def _normalized_shape_residual(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Return Procrustes-like residual after optimal rotation.

    Parameters
    ----------
    reference : numpy.ndarray
        Reference point signature with shape ``[K, 2]``.
    candidate : numpy.ndarray
        Candidate point signature with shape ``[K, 2]``.

    Returns
    -------
    float
        Root-mean-square residual normalized to the reference energy.
    """
    matrix = candidate.T @ reference
    try:
        left, _singular, right_t = np.linalg.svd(matrix)
    except np.linalg.LinAlgError:
        return 1.0
    rotated = candidate @ (left @ right_t)
    denom = max(float(np.mean(np.sum(reference * reference, axis=1))), 1e-12)
    return math.sqrt(float(np.mean(np.sum((reference - rotated) ** 2, axis=1))) / denom)


def _small_subtree_distribution_score(positions: torch.Tensor, forest: TreeForest) -> float:
    """Score sibling spacing proportional to subtree leaf counts.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    forest : TreeForest
        Normalized rooted forest.

    Returns
    -------
    float
        Distribution score in ``[0, 1]``.
    """
    penalties: list[float] = []
    for children in forest.children.values():
        if len(children) < 2:
            continue
        centers = np.asarray(
            [
                float(positions[list(forest.subtree_nodes[child]), 0].mean().item())
                for child in children
            ],
            dtype=np.float64,
        )
        drawn = np.diff(np.sort(centers))
        if drawn.size == 0 or float(drawn.sum()) <= 1e-12:
            penalties.append(1.0)
            continue
        drawn_share = drawn / drawn.sum()
        leaf_weights = np.asarray([forest.subtree_leaves[child] for child in children], dtype=float)
        target = (leaf_weights[:-1] + leaf_weights[1:]) / 2.0
        target_share = target / max(float(target.sum()), 1e-12)
        penalties.append(min(1.0, float(np.mean(np.abs(drawn_share - target_share)))))
    return 1.0 - float(np.mean(penalties)) if penalties else 1.0


def _subtree_contour_separation_score(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    forest: TreeForest,
) -> float:
    """Score overlap avoidance between sibling subtree AABBs.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``; accepted for the shared G4 signature.
    forest : TreeForest
        Normalized rooted forest.

    Returns
    -------
    float
        Contour-separation score in ``[0, 1]``.
    """
    penalties: list[float] = []
    for children in forest.children.values():
        for left_index, left in enumerate(children):
            left_box = _subtree_bounds(positions, sizes, forest.subtree_nodes[left])
            for right in children[left_index + 1 :]:
                right_box = _subtree_bounds(positions, sizes, forest.subtree_nodes[right])
                overlap = _box_intersection_area(left_box, right_box)
                denom = max(1e-12, min(_box_area(left_box), _box_area(right_box)))
                penalties.append(min(1.0, overlap / denom))
    return 1.0 - float(np.mean(penalties)) if penalties else 1.0


def _tree_extent_ratio_score(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    forest: TreeForest,
    *,
    radial: bool,
) -> Optional[float]:
    """Score drawn tree extent against a frozen reference aspect ratio.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.
    forest : TreeForest
        Normalized rooted forest.
    radial : bool
        Whether to use the frozen radial reference extent.

    Returns
    -------
    Optional[float]
        Log-band ratio score.
    """
    from dagua.eval.ruler_v3 import _radial_reference_extent, _tree_reference_extent

    scores: list[float] = []
    edge_list = [
        (parent, child) for parent, children in forest.children.items() for child in children
    ]
    for component in forest.components:
        bounds = _subtree_position_bounds(positions, component)
        drawn_width = max(bounds[2] - bounds[0], 1e-12)
        drawn_height = max(bounds[3] - bounds[1], 1e-12)
        ref_width, ref_height = (
            _radial_reference_extent(sizes, edge_list, component)
            if radial
            else _tree_reference_extent(sizes, edge_list, component)
        )
        drawn_aspect = drawn_width / drawn_height
        ref_aspect = max(ref_width, 1e-12) / max(ref_height, 1e-12)
        scores.append(_log_band_score(drawn_aspect / max(ref_aspect, 1e-12)))
    return float(np.mean(scores)) if scores else None


def _node_radii(positions: torch.Tensor, forest: TreeForest) -> np.ndarray:
    """Return node radii from their component roots.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    forest : TreeForest
        Normalized rooted forest.

    Returns
    -------
    numpy.ndarray
        Radius per node with shape ``[N]``.
    """
    radii = np.zeros(int(positions.shape[0]), dtype=np.float64)
    for root, component in zip(forest.roots, forest.components):
        root_point = positions[root].numpy()
        radii[list(component)] = np.linalg.norm(
            positions[list(component)].numpy() - root_point,
            axis=1,
        )
    return radii


def _node_angles(positions: torch.Tensor, forest: TreeForest) -> np.ndarray:
    """Return node polar angles around component roots.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    forest : TreeForest
        Normalized rooted forest.

    Returns
    -------
    numpy.ndarray
        Angles in radians with shape ``[N]``.
    """
    angles = np.zeros(int(positions.shape[0]), dtype=np.float64)
    for root, component in zip(forest.roots, forest.components):
        delta = positions[list(component)].numpy() - positions[root].numpy()
        angles[list(component)] = np.mod(np.arctan2(delta[:, 1], delta[:, 0]), 2.0 * math.pi)
    return angles


def _angular_allocation_score(angles: np.ndarray, forest: TreeForest) -> float:
    """Score child angular sectors against subtree leaf shares.

    Parameters
    ----------
    angles : numpy.ndarray
        Node angles with shape ``[N]``.
    forest : TreeForest
        Normalized rooted forest.

    Returns
    -------
    float
        Angular allocation score in ``[0, 1]``.
    """
    penalties: list[float] = []
    for children in forest.children.values():
        if len(children) < 2:
            continue
        spans = np.asarray(
            [_angular_span(angles[list(forest.subtree_nodes[child])]) for child in children]
        )
        if float(spans.sum()) <= 1e-12:
            child_angles = np.asarray([angles[int(child)] for child in children], dtype=np.float64)
            drawn_share = _circular_voronoi_shares(child_angles)
        else:
            drawn_share = spans / spans.sum()
        leaves = np.asarray([forest.subtree_leaves[child] for child in children], dtype=np.float64)
        target_share = leaves / max(float(leaves.sum()), 1e-12)
        penalties.append(min(1.0, float(np.mean(np.abs(drawn_share - target_share)))))
    return 1.0 - float(np.mean(penalties)) if penalties else 1.0


def _angular_overlap_score(angles: np.ndarray, forest: TreeForest) -> float:
    """Score absence of angular overlap between sibling subtrees.

    Parameters
    ----------
    angles : numpy.ndarray
        Node angles with shape ``[N]``.
    forest : TreeForest
        Normalized rooted forest.

    Returns
    -------
    float
        No-overlap score in ``[0, 1]``.
    """
    penalties: list[float] = []
    for children in forest.children.values():
        leaves = np.asarray([forest.subtree_leaves[child] for child in children], dtype=np.float64)
        target_shares = leaves / max(float(leaves.sum()), 1e-12)
        sectors = []
        for child, target_share in zip(children, target_shares):
            values = angles[list(forest.subtree_nodes[child])]
            center = _circular_mean(values)
            half_width = max(_angular_span(values) / 2.0, math.pi * float(target_share))
            sectors.append(_circular_interval_segments(center, half_width))
        for left_index, left in enumerate(sectors):
            for right in sectors[left_index + 1 :]:
                overlap = _circular_segments_overlap(left, right)
                left_width = sum(segment[1] - segment[0] for segment in left)
                right_width = sum(segment[1] - segment[0] for segment in right)
                denom = max(1e-12, min(left_width, right_width))
                penalties.append(min(1.0, overlap / denom))
    return 1.0 - float(np.mean(penalties)) if penalties else 1.0


def _circular_voronoi_shares(values: np.ndarray) -> np.ndarray:
    """Return circular sector shares induced by child root angles.

    Parameters
    ----------
    values : numpy.ndarray
        Child root angles in radians.

    Returns
    -------
    numpy.ndarray
        Sector shares summing to one, aligned to ``values``.
    """
    count = int(values.size)
    if count == 0:
        return np.asarray([], dtype=np.float64)
    if count == 1:
        return np.asarray([1.0], dtype=np.float64)
    ordered_indices = np.argsort(np.mod(values, 2.0 * math.pi))
    ordered = np.mod(values[ordered_indices], 2.0 * math.pi)
    prev_gap = np.empty(count, dtype=np.float64)
    next_gap = np.empty(count, dtype=np.float64)
    for index in range(count):
        prev_value = ordered[index - 1] if index > 0 else ordered[-1] - 2.0 * math.pi
        next_value = ordered[(index + 1) % count] + (2.0 * math.pi if index == count - 1 else 0.0)
        prev_gap[index] = ordered[index] - prev_value
        next_gap[index] = next_value - ordered[index]
    shares_ordered = (prev_gap + next_gap) / (4.0 * math.pi)
    shares = np.empty(count, dtype=np.float64)
    shares[ordered_indices] = shares_ordered
    return shares / max(float(shares.sum()), 1e-12)


def _circular_mean(values: np.ndarray) -> float:
    """Return a circular mean angle.

    Parameters
    ----------
    values : numpy.ndarray
        Angles in radians.

    Returns
    -------
    float
        Mean angle in ``[0, 2pi)``.
    """
    if values.size == 0:
        return 0.0
    sin_mean = float(np.sin(values).mean())
    cos_mean = float(np.cos(values).mean())
    return float(np.mod(math.atan2(sin_mean, cos_mean), 2.0 * math.pi))


def _circular_interval_segments(
    center: float,
    half_width: float,
) -> Tuple[Tuple[float, float], ...]:
    """Return non-wrapping segments for one circular interval.

    Parameters
    ----------
    center : float
        Interval center angle in radians.
    half_width : float
        Half interval width in radians.

    Returns
    -------
    Tuple[Tuple[float, float], ...]
        One or two segments in ``[0, 2pi]``.
    """
    full = 2.0 * math.pi
    if half_width >= math.pi:
        return ((0.0, full),)
    start = center - half_width
    end = center + half_width
    if start < 0.0:
        return ((start + full, full), (0.0, end))
    if end > full:
        return ((start, full), (0.0, end - full))
    return ((start, end),)


def _circular_segments_overlap(
    left: Sequence[Tuple[float, float]],
    right: Sequence[Tuple[float, float]],
) -> float:
    """Return overlap length between circular interval segment sets.

    Parameters
    ----------
    left : Sequence[Tuple[float, float]]
        First segment set.
    right : Sequence[Tuple[float, float]]
        Second segment set.

    Returns
    -------
    float
        Total overlap length in radians.
    """
    total = 0.0
    for left_start, left_end in left:
        for right_start, right_end in right:
            total += max(0.0, min(left_end, right_end) - max(left_start, right_start))
    return total


def _angular_span(values: np.ndarray) -> float:
    """Return the shortest circular span containing angle values.

    Parameters
    ----------
    values : numpy.ndarray
        Angles in radians.

    Returns
    -------
    float
        Span in radians.
    """
    if values.size < 2:
        return 0.0
    ordered = np.sort(np.mod(values, 2.0 * math.pi))
    gaps = np.diff(np.concatenate([ordered, ordered[:1] + 2.0 * math.pi]))
    return float(2.0 * math.pi - gaps.max())


def _angular_interval(values: np.ndarray) -> Tuple[float, float]:
    """Return a deterministic unwrapped interval for angle values.

    Parameters
    ----------
    values : numpy.ndarray
        Angles in radians.

    Returns
    -------
    Tuple[float, float]
        Interval endpoints in radians.
    """
    if values.size == 0:
        return (0.0, 0.0)
    ordered = np.sort(np.mod(values, 2.0 * math.pi))
    if ordered.size == 1:
        return (float(ordered[0]), float(ordered[0]))
    gaps = np.diff(np.concatenate([ordered, ordered[:1] + 2.0 * math.pi]))
    start = int(np.argmax(gaps) + 1) % ordered.size
    unwrapped = np.concatenate([ordered[start:], ordered[:start] + 2.0 * math.pi])
    return float(unwrapped[0]), float(unwrapped[-1])


def _subtree_bounds(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    nodes: Sequence[int],
) -> Tuple[float, float, float, float]:
    """Return the visual AABB for a subtree.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.
    nodes : Sequence[int]
        Node ids to bound.

    Returns
    -------
    Tuple[float, float, float, float]
        Bounds as ``(min_x, min_y, max_x, max_y)``.
    """
    node_list = [int(node) for node in nodes]
    lower = (positions[node_list] - sizes[node_list] / 2.0).min(dim=0).values
    upper = (positions[node_list] + sizes[node_list] / 2.0).max(dim=0).values
    return (
        float(lower[0].item()),
        float(lower[1].item()),
        float(upper[0].item()),
        float(upper[1].item()),
    )


def _subtree_position_bounds(
    positions: torch.Tensor,
    nodes: Sequence[int],
) -> Tuple[float, float, float, float]:
    """Return the position-only AABB for a subtree.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    nodes : Sequence[int]
        Node ids to bound.

    Returns
    -------
    Tuple[float, float, float, float]
        Bounds as ``(min_x, min_y, max_x, max_y)``.
    """
    node_list = [int(node) for node in nodes]
    lower = positions[node_list].min(dim=0).values
    upper = positions[node_list].max(dim=0).values
    return (
        float(lower[0].item()),
        float(lower[1].item()),
        float(upper[0].item()),
        float(upper[1].item()),
    )


def _box_area(bounds: Tuple[float, float, float, float]) -> float:
    """Return AABB area.

    Parameters
    ----------
    bounds : Tuple[float, float, float, float]
        Box bounds.

    Returns
    -------
    float
        Positive area.
    """
    return max(0.0, bounds[2] - bounds[0]) * max(0.0, bounds[3] - bounds[1])


def _box_intersection_area(
    left: Tuple[float, float, float, float],
    right: Tuple[float, float, float, float],
) -> float:
    """Return AABB intersection area.

    Parameters
    ----------
    left : Tuple[float, float, float, float]
        First box.
    right : Tuple[float, float, float, float]
        Second box.

    Returns
    -------
    float
        Intersection area.
    """
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    return width * height


def _position_span(values: np.ndarray) -> float:
    """Return a stable scalar span.

    Parameters
    ----------
    values : numpy.ndarray
        Coordinate values.

    Returns
    -------
    float
        Positive span.
    """
    return max(float(np.max(values) - np.min(values)), 1e-12)


def _log_band_score(ratio: float) -> float:
    """Score a positive ratio against a symmetric log band.

    Parameters
    ----------
    ratio : float
        Positive measured/reference ratio.

    Returns
    -------
    float
        Score in ``[0, 1]``.
    """
    if TREE_RATIO_PLATEAU_LO <= ratio <= TREE_RATIO_PLATEAU_HI:
        return 1.0
    if ratio < TREE_RATIO_PLATEAU_LO:
        return max(
            0.0,
            1.0 - math.log(TREE_RATIO_PLATEAU_LO / max(ratio, 1e-12)) / math.log(10.0),
        )
    return max(
        0.0,
        1.0 - math.log(ratio / TREE_RATIO_PLATEAU_HI) / math.log(10.0 * TREE_RATIO_LOG_DECAY),
    )


def _acyclic_fraction_and_depth(
    edge_index: torch.Tensor,
    num_nodes: int,
    meta: Mapping[str, Any],
) -> Tuple[float, np.ndarray]:
    """Compute G1's graded acyclicity multiplier and topological depth.

    Parameters
    ----------
    edge_index : torch.Tensor
        Directed edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    meta : Mapping[str, Any]
        Declared metadata that may include frozen topological depths.

    Returns
    -------
    Tuple[float, numpy.ndarray]
        Fraction of edges consistent with deterministic topological depths and
        depth labels with shape ``[N]``.
    """
    declared = _metadata_sequence(meta, ("topological_depth", "topological_depths", "ranks"))
    if declared is not None:
        depth = np.asarray(list(declared), dtype=np.float64)
        if depth.shape[0] != num_nodes:
            raise ValueError("declared topological depths must have length N")
    else:
        depth = _longest_dag_depth(edge_index, num_nodes)
    if int(edge_index.shape[1]) == 0:
        return 1.0, depth
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    forward = depth[dst] > depth[src]
    if not bool(np.any(forward)) and declared is None:
        return 0.0, depth
    return float(np.mean(forward.astype(np.float64))), depth


def _longest_dag_depth(edge_index: torch.Tensor, num_nodes: int) -> np.ndarray:
    """Return deterministic longest-path depths from Kahn's algorithm.

    Parameters
    ----------
    edge_index : torch.Tensor
        Directed edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    numpy.ndarray
        Depth labels with shape ``[N]``. Nodes left in cycles retain the best
        depth inferred before the cycle blocks traversal.
    """
    adjacency: Tuple[list[int], ...] = tuple([] for _index in range(num_nodes))
    mutable = [list(neighbors) for neighbors in adjacency]
    indegree = np.zeros(num_nodes, dtype=np.int64)
    for source, target in edge_index.t().tolist():
        mutable[int(source)].append(int(target))
        indegree[int(target)] += 1
    queue = [index for index in range(num_nodes) if indegree[index] == 0]
    depth = np.zeros(num_nodes, dtype=np.float64)
    cursor = 0
    while cursor < len(queue):
        node = queue[cursor]
        cursor += 1
        for target in sorted(mutable[node]):
            depth[target] = max(depth[target], depth[node] + 1.0)
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    return depth


def _depth_spearman_score(
    pos: torch.Tensor,
    depth: np.ndarray,
    direction: str,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """Score rank agreement between drawn and topological depth.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    depth : numpy.ndarray
        Topological depth values with shape ``[N]``.
    direction : str
        Declared flow direction.

    Returns
    -------
    Tuple[Optional[float], Dict[str, Any]]
        Normalized Spearman score and publication metadata.
    """
    if depth.size < 2 or np.unique(depth).size < 2:
        return None, {"sample_count": int(depth.size), "depth_spearman_raw": None}
    axis_values = _axis_values(pos, direction)
    rho = _spearman(axis_values, depth)
    if rho is None:
        return None, {"sample_count": int(depth.size), "depth_spearman_raw": None}
    return (rho + 1.0) / 2.0, {
        "sample_count": int(depth.size),
        "depth_spearman_raw": rho,
    }


def _axis_values(pos: torch.Tensor, direction: str) -> np.ndarray:
    """Project positions onto the declared flow axis.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    direction : str
        Declared flow direction.

    Returns
    -------
    numpy.ndarray
        Signed axis coordinates with shape ``[N]``.
    """
    positions = pos.numpy()
    if direction == "BT":
        return -positions[:, 1]
    if direction == "LR":
        return positions[:, 0]
    if direction == "RL":
        return -positions[:, 0]
    return positions[:, 1]


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Compute average ranks with deterministic tie handling.

    Parameters
    ----------
    values : numpy.ndarray
        Values to rank.

    Returns
    -------
    numpy.ndarray
        Average ranks with shape matching ``values``.
    """
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Compute Spearman rank correlation.

    Parameters
    ----------
    x : numpy.ndarray
        First values with shape ``[N]``.
    y : numpy.ndarray
        Second values with shape ``[N]``.

    Returns
    -------
    Optional[float]
        Correlation in ``[-1, 1]`` or ``None`` for constant inputs.
    """
    if x.shape[0] != y.shape[0] or x.shape[0] < 2:
        return None
    rx = _rankdata(x.astype(np.float64))
    ry = _rankdata(y.astype(np.float64))
    rx -= rx.mean()
    ry -= ry.mean()
    denom = math.sqrt(float(np.dot(rx, rx) * np.dot(ry, ry)))
    if denom <= 1e-24:
        return None
    return max(-1.0, min(1.0, float(np.dot(rx, ry) / denom)))


def _hac_labels(points: np.ndarray, cluster_count: int) -> np.ndarray:
    """Cluster drawing coordinates with frozen average-linkage HAC.

    Parameters
    ----------
    points : numpy.ndarray
        Position array with shape ``[N, 2]``.
    cluster_count : int
        Frozen cluster count from the planted partition.

    Returns
    -------
    numpy.ndarray
        Predicted integer labels with shape ``[N]``.
    """
    if points.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    if cluster_count <= 1:
        return np.zeros(points.shape[0], dtype=np.int64)
    if cluster_count >= points.shape[0]:
        return np.arange(points.shape[0], dtype=np.int64)
    from scipy.cluster.hierarchy import fcluster, linkage

    linkage_matrix = linkage(points, method="average", metric="euclidean", optimal_ordering=False)
    return fcluster(linkage_matrix, cluster_count, criterion="maxclust").astype(np.int64)


def adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Compute the adjusted Rand index without a scikit-learn dependency.

    Parameters
    ----------
    labels_true : numpy.ndarray
        Ground-truth labels with shape ``[N]``.
    labels_pred : numpy.ndarray
        Predicted labels with shape ``[N]``.

    Returns
    -------
    float
        Adjusted Rand index in ``[-0.5, 1]``.
    """
    if labels_true.shape[0] != labels_pred.shape[0]:
        raise ValueError("ARI inputs must have the same length")
    n = int(labels_true.shape[0])
    if n < 2:
        return 1.0
    true_ids, true_inverse = np.unique(labels_true, return_inverse=True)
    pred_ids, pred_inverse = np.unique(labels_pred, return_inverse=True)
    contingency = np.zeros((true_ids.size, pred_ids.size), dtype=np.int64)
    np.add.at(contingency, (true_inverse, pred_inverse), 1)
    sum_comb = float(sum(_comb2(value) for value in contingency.ravel()))
    true_comb = float(sum(_comb2(value) for value in contingency.sum(axis=1)))
    pred_comb = float(sum(_comb2(value) for value in contingency.sum(axis=0)))
    total_comb = float(_comb2(n))
    if total_comb <= 0.0:
        return 1.0
    expected = true_comb * pred_comb / total_comb
    maximum = 0.5 * (true_comb + pred_comb)
    denominator = maximum - expected
    if abs(denominator) <= 1e-24:
        return 1.0 if sum_comb == maximum else 0.0
    return float((sum_comb - expected) / denominator)


def _comb2(value: int) -> int:
    """Return ``value choose 2`` for non-negative counts.

    Parameters
    ----------
    value : int
        Count.

    Returns
    -------
    int
        Number of unordered pairs.
    """
    return int(value) * (int(value) - 1) // 2


def _edge_weights_from_meta(meta: Mapping[str, Any]) -> Optional[np.ndarray]:
    """Normalize declared edge weights from metadata.

    Parameters
    ----------
    meta : Mapping[str, Any]
        Declared graph metadata.

    Returns
    -------
    Optional[numpy.ndarray]
        Positive weights with shape ``[E]`` when declared.
    """
    value = meta.get("edge_weights", meta.get("weights"))
    if value is None:
        return None
    weights = np.asarray(list(value), dtype=np.float64)
    if weights.ndim != 1:
        raise ValueError("edge_weights must be one-dimensional")
    if bool(np.any(~np.isfinite(weights))) or bool(np.any(weights <= 0.0)):
        raise ValueError("edge_weights must be positive finite values")
    return weights


def _weight_cv(weights: np.ndarray) -> float:
    """Return the coefficient of variation for declared weights.

    Parameters
    ----------
    weights : numpy.ndarray
        Positive weights with shape ``[E]``.

    Returns
    -------
    float
        Coefficient of variation.
    """
    mean = float(weights.mean()) if weights.size else 0.0
    if mean <= 1e-24:
        return 0.0
    return float(weights.std() / mean)


def _weights_to_distances(weights: np.ndarray, mode: str) -> np.ndarray:
    """Transform declared weights to geometric distances.

    Parameters
    ----------
    weights : numpy.ndarray
        Positive declared weights with shape ``[E]``.
    mode : str
        Declared semantics: ``distance``/``cost`` or
        ``strength``/``similarity``.

    Returns
    -------
    numpy.ndarray
        Positive edge distances with shape ``[E]``.
    """
    if mode in {"strength", "similarity"}:
        return 1.0 / np.maximum(weights, 1e-12)
    if mode in {"distance", "cost"}:
        return weights.astype(np.float64)
    raise ValueError(f"unsupported weight_mode for G6: {mode}")


def _weighted_all_pairs_distances(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_distances: np.ndarray,
) -> np.ndarray:
    """Compute frozen weighted shortest-path distances.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    edge_distances : numpy.ndarray
        Positive edge distances with shape ``[E]``.

    Returns
    -------
    numpy.ndarray
        Weighted APSP matrix with ``inf`` for unreachable pairs.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import shortest_path

    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    data = np.concatenate([edge_distances, edge_distances])
    adjacency = coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()
    return shortest_path(adjacency, directed=False, unweighted=False)


def weighted_isotonic_ksm(
    pos: torch.Tensor,
    weighted_distances: np.ndarray,
    *,
    n_sources: int = 200,
    n_targets: int = 1000,
) -> Dict[str, float]:
    """Compute isotonic KSM against weighted shortest-path distances.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    weighted_distances : numpy.ndarray
        Weighted APSP matrix with shape ``[N, N]``.
    n_sources : int, optional
        Deterministic source-row budget.
    n_targets : int, optional
        Deterministic target budget per source.

    Returns
    -------
    Dict[str, float]
        Weighted KSM score and sample counts.
    """
    positions = _ensure_cpu(pos).to(dtype=torch.float64).numpy()
    sources, targets, target_distances = _weighted_distance_pairs(
        weighted_distances,
        n_sources=n_sources,
        n_targets=n_targets,
    )
    if sources.size == 0:
        return {"weighted_ksm_score": 0.0, "weighted_ksm_n_pairs": 0, "weighted_ksm_n_sources": 0}
    geometric = np.linalg.norm(positions[sources] - positions[targets], axis=1)
    denominator = float(np.dot(geometric, geometric))
    if denominator <= 1e-24:
        return {
            "weighted_ksm_score": 0.0,
            "weighted_ksm_n_pairs": int(sources.size),
            "weighted_ksm_n_sources": int(np.unique(sources).size),
        }
    fitted = _pav_fitted_values(target_distances, geometric)
    residual = geometric - fitted
    stress = math.sqrt(float(np.mean(residual**2) / np.mean(geometric**2)))
    return {
        "weighted_ksm_score": max(0.0, min(1.0, 1.0 - stress)),
        "weighted_ksm_n_pairs": int(sources.size),
        "weighted_ksm_n_sources": int(np.unique(sources).size),
    }


def _weighted_distance_pairs(
    weighted_distances: np.ndarray,
    *,
    n_sources: int,
    n_targets: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample finite weighted-distance pairs deterministically.

    Parameters
    ----------
    weighted_distances : numpy.ndarray
        Weighted APSP matrix with shape ``[N, N]``.
    n_sources : int
        Source-row budget.
    n_targets : int
        Per-source target budget.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Source indices, target indices, and weighted distances.
    """
    n = int(weighted_distances.shape[0])
    source_indices = _deterministic_sample_indices(n, min(n, n_sources))
    pair_sources: list[int] = []
    pair_targets: list[int] = []
    pair_distances: list[float] = []
    for source in source_indices:
        row = weighted_distances[int(source)]
        reachable = np.flatnonzero(np.isfinite(row) & (row > 0.0))
        if reachable.size == 0:
            continue
        selected = reachable
        if reachable.size > n_targets:
            selected = reachable[_deterministic_sample_indices(reachable.size, n_targets)]
        pair_sources.extend([int(source)] * int(selected.size))
        pair_targets.extend(int(target) for target in selected.tolist())
        pair_distances.extend(float(row[int(target)]) for target in selected.tolist())
    return (
        np.asarray(pair_sources, dtype=np.int64),
        np.asarray(pair_targets, dtype=np.int64),
        np.asarray(pair_distances, dtype=np.float64),
    )


def _pav_fitted_values(target: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Fit nondecreasing isotonic values with the pair-adjacent-violators algorithm.

    Parameters
    ----------
    target : numpy.ndarray
        Target distances that define the monotone order.
    observed : numpy.ndarray
        Observed geometric distances.

    Returns
    -------
    numpy.ndarray
        Fitted values in original pair order.
    """
    order = np.argsort(target, kind="mergesort")
    y = observed[order].astype(np.float64)
    blocks: list[tuple[int, int, float, int]] = []
    for index, value in enumerate(y):
        blocks.append((index, index + 1, float(value), 1))
        while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2]:
            left = blocks.pop()
            right = blocks.pop()
            count = left[3] + right[3]
            mean = (left[2] * left[3] + right[2] * right[3]) / count
            blocks.append((right[0], left[1], mean, count))
    fitted_sorted = np.empty_like(y)
    for start, end, mean, _count in blocks:
        fitted_sorted[start:end] = mean
    fitted = np.empty_like(fitted_sorted)
    fitted[order] = fitted_sorted
    return fitted


def local_weight_monotonicity_score(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    weights: np.ndarray,
    mode: str,
) -> Dict[str, float]:
    """Score local agreement between incident weight and drawn length order.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    weights : numpy.ndarray
        Declared edge weights with shape ``[E]``.
    mode : str
        Declared weight semantics.

    Returns
    -------
    Dict[str, float]
        Local monotonicity score and sample count.
    """
    positions = _ensure_cpu(pos).to(dtype=torch.float64)
    edges = _normalize_edge_index(edge_index)
    edge_lengths = torch.linalg.vector_norm(
        positions[edges[0]] - positions[edges[1]],
        dim=1,
    ).numpy()
    target_distances = _weights_to_distances(weights, mode)
    incident = _incident_edges(edges, int(positions.shape[0]))
    nodes = _deterministic_sample_indices(
        int(positions.shape[0]),
        min(int(positions.shape[0]), LOCAL_WEIGHT_MONOTONICITY_NODE_BUDGET),
    )
    scores: list[float] = []
    for node in nodes:
        edge_ids = incident[int(node)]
        if len(edge_ids) < 2:
            continue
        rho = _spearman(target_distances[edge_ids], edge_lengths[edge_ids])
        if rho is not None:
            scores.append((rho + 1.0) / 2.0)
    return {
        "local_weight_monotonicity_score": float(np.mean(scores)) if scores else 1.0,
        "local_weight_monotonicity_n_neighborhoods": int(len(scores)),
    }


def _incident_edges(edge_index: torch.Tensor, num_nodes: int) -> Tuple[np.ndarray, ...]:
    """Return incident edge ids for each node.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    Tuple[numpy.ndarray, ...]
        Incident edge-id arrays for each node.
    """
    buckets: list[list[int]] = [[] for _index in range(num_nodes)]
    for edge_id, (source, target) in enumerate(edge_index.t().tolist()):
        buckets[int(source)].append(edge_id)
        buckets[int(target)].append(edge_id)
    return tuple(np.asarray(bucket, dtype=np.int64) for bucket in buckets)


GROUP_REGISTRY: Dict[str, ConditionalGroup] = {
    "G1": ConditionalGroup(
        key="G1",
        applicability_gate=_declared_hierarchy_gate,
        tier_slots=(GroupSlot("G1_directed_flow", 2, 1.0),),
        score_fn=_score_g1,
    ),
    "G2": ConditionalGroup(
        key="G2",
        applicability_gate=_declared_cluster_gate,
        tier_slots=(
            GroupSlot("G2_cluster_sibling_overlap", 2, 2.0 / G2_SLOT_A_TOTAL),
            GroupSlot("G2_cluster_nesting_fidelity", 2, 1.0 / G2_SLOT_A_TOTAL),
            GroupSlot("G2_cluster_edge_intrusion", 2, 1.0 / G2_SLOT_A_TOTAL),
            GroupSlot("G2_cluster_label_occlusion", 2, 1.0 / G2_SLOT_A_TOTAL),
            GroupSlot("G2_cluster_hac_ari", 2, 2.0 / G2_SLOT_B_TOTAL),
            GroupSlot("G2_cluster_compactness_log_band", 2, 1.0 / G2_SLOT_B_TOTAL),
        ),
        score_fn=_score_g2,
    ),
    "G3": ConditionalGroup(
        key="G3",
        applicability_gate=_planted_partition_gate,
        tier_slots=(GroupSlot("G3_community_ari", 2, 1.0),),
        score_fn=_score_g3,
    ),
    "G4": ConditionalGroup(
        key="G4",
        applicability_gate=_declared_tree_gate,
        tier_slots=(
            GroupSlot("G4_layered_or_radial_axis_depth", 2, 2.0 / G4_SLOT_A_TOTAL),
            GroupSlot("G4_layered_or_radial_centering_allocation", 2, 2.0 / G4_SLOT_A_TOTAL),
            GroupSlot("G4_layered_or_radial_order", 2, 1.0 / G4_SLOT_A_TOTAL),
            GroupSlot("G4_layered_or_radial_congruence_or_overlap", 2, 2.0 / G4_SLOT_B_TOTAL),
            GroupSlot("G4_layered_or_radial_distribution", 2, 1.0 / G4_SLOT_B_TOTAL),
            GroupSlot("G4_layered_or_radial_separation", 2, 1.0 / G4_SLOT_B_TOTAL),
            GroupSlot("G4_layered_or_radial_extent_ratio", 2, 1.0 / G4_SLOT_B_TOTAL),
        ),
        score_fn=_score_g4,
    ),
    "G5": ConditionalGroup(
        key="G5",
        applicability_gate=_declared_temporal_gate,
        tier_slots=(GroupSlot("G5_temporal_stability", 2, 1.0),),
        score_fn=_score_g5,
    ),
    "G6": ConditionalGroup(
        key="G6",
        applicability_gate=_weighted_gate,
        tier_slots=(
            GroupSlot("G6_weighted_ksm", 1, 1.0),
            GroupSlot("G6_local_weight_monotonicity", 3, 1.0),
        ),
        score_fn=_score_g6,
    ),
    "G7": ConditionalGroup(
        key="G7",
        applicability_gate=_declared_port_gate,
        tier_slots=(
            GroupSlot("G7_port_hard_compliance", 2, 1.0),
            GroupSlot("G7_routed_curve_quality", 2, 1.0),
            GroupSlot("G7_routed_bend_terminal_economy", 2, 1.0),
        ),
        score_fn=_score_g7,
    ),
}


__all__ = [
    "ConditionalGroup",
    "GROUP_REGISTRY",
    "GroupEvaluation",
    "GroupFacetScore",
    "GroupSlot",
    "adjusted_rand_index",
    "evaluate_conditional_groups",
    "local_weight_monotonicity_score",
    "weighted_isotonic_ksm",
]
