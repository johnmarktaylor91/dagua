"""Input-gated conditional groups for the scoring-only V3 ruler.

The registry in this module is intentionally separate from ``ruler_v3.py`` so
new conditional groups can be added without changing the CORE scorer. Gates
read only declared graph metadata, never the drawing, following DOC-3 in the
joint V3 ruler design.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from dagua.metrics import (
    _deterministic_sample_indices,
    _ensure_cpu,
    cluster_edge_intrusion_score,
    cluster_exclusion_score,
    cluster_label_occlusion_score,
    cluster_nesting_fidelity_score,
    cluster_sibling_overlap_score,
    directed_flow_score,
)

G2_SLOT_A_TOTAL = 7.0
G2_SLOT_B_TOTAL = 3.0
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
    frac_acyclic, depth = _acyclic_fraction_and_depth(edges, int(positions.shape[0]), meta)
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
            metadata={**common_meta, "sample_count": int(edges.shape[1])},
        ),
        "G1_depth_order": GroupFacetScore(
            code="G1_depth_order",
            name="depth_spearman",
            tier=3,
            score=depth_score,
            base_weight=_tier_weight(3),
            effective_weight=_tier_weight(3) * frac_acyclic,
            applicable=depth_score is not None,
            applicability_reason=reason,
            metadata={**common_meta, **depth_meta},
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
    predicted = _hac_labels(positions.numpy(), cluster_count)
    ari_raw = adjusted_rand_index(declared_labels, predicted)
    compactness_score, compactness_meta = _cluster_log_ratio_compactness(
        positions,
        sizes,
        clusters,
    )
    reason = "applicable:declared_cluster_metadata"
    common = {
        "gate": "declared_clusters_or_cluster_parents_only",
        "cluster_count": cluster_count,
        "invariance_slot_a": "size_anchored_position_scale_sensitive_by_design",
        "invariance_slot_b_ari": "translation_rotation_reflection_position_scale_invariant",
        "invariance_slot_b_compactness": "size_anchored_position_scale_sensitive_by_design",
        "deformation_monotonicity_probe": "implemented_by_tests/test_ruler_v3_groups.py",
    }
    facets = {}
    for code, name, raw_score, slot_weight, metadata in _g2_slot_a_scores(
        positions,
        edges,
        sizes,
        clusters,
        cluster_parents,
        cluster_labels,
    ):
        effective_weight = _tier_weight(2) * slot_weight / G2_SLOT_A_TOTAL
        facets[code] = GroupFacetScore(
            code=code,
            name=name,
            tier=2,
            score=raw_score,
            base_weight=effective_weight,
            effective_weight=effective_weight if raw_score is not None else 0.0,
            applicable=raw_score is not None,
            applicability_reason=reason,
            metadata={
                **common,
                **metadata,
                "slot": "G2_A_containment_hygiene",
                "slot_internal_weight": slot_weight,
                "normalization": "reused frozen dagua.metrics AABB cluster scorer",
                "invariance": "size_anchored_position_scale_sensitive_by_design",
            },
        )
    ari_score = max(0.0, min(1.0, ari_raw))
    for code, name, score, slot_weight, metadata in (
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
            },
        ),
    ):
        effective_weight = _tier_weight(2) * slot_weight / G2_SLOT_B_TOTAL
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
    forest = _tree_forest(edges, int(positions.shape[0]), meta)
    convention = str(meta.get("tree_convention", meta.get("tree_layout", "layered"))).lower()
    reason = f"applicable:declared_rooted_tree_{convention}_convention"
    common = {
        "gate": "declared_rooted_tree_forest_x_declared_convention",
        "tree_convention": convention,
        "root_count": len(forest.roots),
        "invariance": "translation_rotation_reflection_position_scale_invariant",
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
        n_sources=int(meta.get("weighted_stress_sources", 200)),
        n_targets=int(meta.get("weighted_stress_targets", 1000)),
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

    Returns
    -------
    Tuple[Tuple[str, str, Optional[float], float, Dict[str, Any]], ...]
        Slot A facet tuples.
    """
    exclusion = cluster_exclusion_score(positions, sizes, clusters, cluster_parents, cluster_labels)
    sibling = cluster_sibling_overlap_score(
        positions, sizes, clusters, cluster_parents, cluster_labels
    )
    nesting = cluster_nesting_fidelity_score(
        positions, sizes, clusters, cluster_parents, cluster_labels
    )
    intrusion = cluster_edge_intrusion_score(
        positions, edges, sizes, clusters, cluster_parents, cluster_labels
    )
    labels = cluster_label_occlusion_score(
        positions,
        sizes,
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
            exclusion,
        ),
        (
            "G2_cluster_sibling_overlap",
            "cluster_sibling_overlap",
            _optional_score(sibling["cluster_sibling_overlap_score"]),
            2.0,
            sibling,
        ),
        (
            "G2_cluster_nesting_fidelity",
            "cluster_nesting_fidelity",
            _optional_score(nesting["cluster_nesting_fidelity_score"]),
            1.0,
            nesting,
        ),
        (
            "G2_cluster_edge_intrusion",
            "cluster_edge_intrusion",
            _optional_score(intrusion["cluster_edge_intrusion_score"]),
            1.0,
            intrusion,
        ),
        (
            "G2_cluster_label_occlusion",
            "cluster_label_occlusion",
            _optional_score(labels["cluster_label_occlusion_score"]),
            1.0,
            labels,
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
    for node in children:
        children[node].sort()
    components: list[Tuple[int, ...]] = []
    depths: Dict[int, int] = {}
    for root in roots:
        queue = [root]
        depths[root] = 0
        component: list[int] = []
        while queue:
            node = queue.pop(0)
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
    child_values = tuple(children[node])
    if not child_values:
        subtree_nodes[node] = (node,)
        subtree_leaves[node] = 1
        subtree_hashes[node] = "()"
        return
    nodes = [node]
    leaves = 0
    child_hashes: list[str] = []
    for child in child_values:
        _fill_subtree(child, children, subtree_nodes, subtree_leaves, subtree_hashes)
        nodes.extend(subtree_nodes[child])
        leaves += subtree_leaves[child]
        child_hashes.append(subtree_hashes[child])
    subtree_nodes[node] = tuple(sorted(nodes))
    subtree_leaves[node] = leaves
    subtree_hashes[node] = "(" + "".join(sorted(child_hashes)) + ")"


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
            {"raw_spearman": depth_rho},
        ),
        (
            "G4_layered_parent_centering",
            "rt_parent_centering",
            _parent_centering_score(positions, forest),
            "G4_A_RT_axioms",
            2.0,
            G4_SLOT_A_TOTAL,
            {},
        ),
        (
            "G4_layered_sibling_order",
            "rt_sibling_order_preservation",
            _sibling_order_score(positions[:, 0].numpy(), forest),
            "G4_A_RT_axioms",
            1.0,
            G4_SLOT_A_TOTAL,
            {},
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
            {},
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
            _sibling_order_score(angles, forest),
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
            {},
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


def _parent_centering_score(positions: torch.Tensor, forest: TreeForest) -> float:
    """Score parent x-coordinate centering over child centroids.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    forest : TreeForest
        Normalized rooted forest.

    Returns
    -------
    float
        Continuous score in ``[0, 1]``.
    """
    span = _position_span(positions[:, 0].numpy())
    penalties: list[float] = []
    for parent, children in forest.children.items():
        if not children:
            continue
        child_center = float(positions[list(children), 0].mean().item())
        penalties.append(min(1.0, abs(float(positions[parent, 0].item()) - child_center) / span))
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
    del sizes
    penalties: list[float] = []
    for children in forest.children.values():
        for left_index, left in enumerate(children):
            left_box = _subtree_position_bounds(positions, forest.subtree_nodes[left])
            for right in children[left_index + 1 :]:
                right_box = _subtree_position_bounds(positions, forest.subtree_nodes[right])
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
            penalties.append(1.0)
            continue
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
        intervals = [
            _angular_interval(angles[list(forest.subtree_nodes[child])]) for child in children
        ]
        for left_index, left in enumerate(intervals):
            for right in intervals[left_index + 1 :]:
                overlap = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
                denom = max(1e-12, min(left[1] - left[0], right[1] - right[0]))
                penalties.append(min(1.0, overlap / denom))
    return 1.0 - float(np.mean(penalties)) if penalties else 1.0


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
        tier_slots=(
            GroupSlot("G1_directed_flow", 2, 1.0),
            GroupSlot("G1_depth_order", 3, 1.0),
        ),
        score_fn=_score_g1,
    ),
    "G2": ConditionalGroup(
        key="G2",
        applicability_gate=_declared_cluster_gate,
        tier_slots=(
            GroupSlot("G2_cluster_exclusion", 2, 2.0 / G2_SLOT_A_TOTAL),
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
    "G6": ConditionalGroup(
        key="G6",
        applicability_gate=_weighted_gate,
        tier_slots=(
            GroupSlot("G6_weighted_ksm", 1, 1.0),
            GroupSlot("G6_local_weight_monotonicity", 3, 1.0),
        ),
        score_fn=_score_g6,
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
