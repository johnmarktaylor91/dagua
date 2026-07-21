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

from dagua.metrics import _deterministic_sample_indices, _ensure_cpu, directed_flow_score

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
    "G3": ConditionalGroup(
        key="G3",
        applicability_gate=_planted_partition_gate,
        tier_slots=(GroupSlot("G3_community_ari", 2, 1.0),),
        score_fn=_score_g3,
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
