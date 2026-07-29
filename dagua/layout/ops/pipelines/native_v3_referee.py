"""Runtime-restricted V3 referee for native finalist selection."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch

from dagua.eval.ruler_v3 import (
    C4_CLEARANCE_BAND_NODE_DIAGONALS,
    FAMILY_SOFTMIN_TAU,
    FROZEN_CONSTANTS,
    RulerV3Facet,
    RulerV3Result,
    _apply_headline_degeneracy_fold,
    _boxes_to_centers_sizes,
    _build_facets,
    _coincident_collapse_fraction,
    _component_weighted_ksm,
    _conditional_group_flags,
    _crossing_pair_geometry,
    _ensure_cpu,
    _headline_degeneracy_fold,
    _is_degenerate_scale,
    _is_sprawl_collapse,
    _mean_node_diagonal,
    _merge_conditional_group_facets,
    _node_visual_boxes,
    _normalize_edge_index,
    _normalize_label_geometry,
    _normalize_node_sizes,
    _optional_float,
    _row_flags,
    _score_family,
    _severe_g6_breach_from_facets,
    _triple_view_scores,
    angle_weighted_crossing_score,
    angular_resolution_score,
    crossing_angle_90_score,
    crossing_weight_multiplier,
    edge_length_cv,
    edge_length_deviation_score,
    evaluate_conditional_groups,
    gabriel_score,
    multi_radius_neighborhood_preservation,
    path_continuity_score,
    referee_eligibility_key,
    whitespace_sprawl_score,
)
from dagua.layout.ops.state import LayoutProblem
from dagua.metrics import node_occlusion_score

V3_REFEREE_DWU_FRACTION = 0.25
_FROZEN_STRESS_SOURCES = 200
_FROZEN_STRESS_TARGETS = 1000
_FROZEN_CROSSING_SAMPLES = 1_000_000
_FROZEN_NEIGHBORHOOD_SAMPLES = 5000


def fast_smooth_clearance_occlusion_score(
    centers: torch.Tensor,
    sizes: torch.Tensor,
    *,
    label_inclusive: bool,
    seed: Optional[int],
) -> Dict[str, float]:
    """Score C4 with bit-exact vectorized pair arithmetic.

    Parameters
    ----------
    centers : torch.Tensor
        Visual box centers with shape ``[N, 2]``.
    sizes : torch.Tensor
        Visual box sizes with shape ``[N, 2]``.
    label_inclusive : bool
        Whether the visual boxes include labels.
    seed : Optional[int]
        Frozen seed forwarded to the legacy overlap metadata.

    Returns
    -------
    dict[str, float]
        Frozen C4 diagnostics with the same values as the V3 loop scorer.
    """
    legacy = node_occlusion_score(centers, sizes, seed=seed)
    count = int(centers.shape[0])
    if count < 2:
        return {
            **legacy,
            "node_occlusion_score": 1.0,
            "clearance_penalty": 0.0,
            "clearance_contact_pairs": 0,
            "clearance_band_node_diagonals": C4_CLEARANCE_BAND_NODE_DIAGONALS,
            "overlap_area_severity": 0.0,
            "packed_seam_severity": 0.0,
            "clearance_abut_count": 0,
            "visual_packing_fill": 1.0 if count == 1 else 0.0,
            "label_inclusive": label_inclusive,
        }
    mean_diag = _mean_node_diagonal(sizes)
    band = max(1.0e-12, C4_CLEARANCE_BAND_NODE_DIAGONALS * mean_diag)
    centers_cpu = _ensure_cpu(centers).to(dtype=torch.float64)
    sizes_cpu = _ensure_cpu(sizes).to(dtype=torch.float64)
    if not bool(torch.isfinite(centers_cpu).all().item()):
        raise ValueError("visual box centers must be finite")
    if not bool(torch.isfinite(sizes_cpu).all().item()):
        raise ValueError("visual box sizes must be finite")
    if bool((sizes_cpu <= 0.0).any().item()):
        raise ValueError("visual box sizes must be positive")
    box_areas = torch.prod(sizes_cpu, dim=1)
    if not bool(torch.isfinite(box_areas).all().item()) or bool((box_areas <= 0.0).any().item()):
        raise ValueError("visual box areas must be finite and positive")
    area_sum = float(box_areas.sum().item())
    mins = centers_cpu - sizes_cpu / 2.0
    maxes = centers_cpu + sizes_cpu / 2.0
    bbox_min = torch.min(mins, dim=0).values
    bbox_max = torch.max(maxes, dim=0).values
    bbox_size = bbox_max - bbox_min
    bbox_area = float((bbox_size[0] * bbox_size[1]).item())
    if not math.isfinite(bbox_area) or bbox_area <= 0.0:
        raise ValueError("visual union bounding-box area must be finite and positive")

    left, right = torch.triu_indices(count, count, offset=1)
    delta = torch.abs(centers_cpu[left] - centers_cpu[right])
    gap_xy = delta - (sizes_cpu[left] + sizes_cpu[right]) / 2.0
    any_positive = (gap_xy > 0.0).any(dim=1)
    all_negative = (gap_xy < 0.0).all(dim=1)
    positive_gap = torch.clamp(gap_xy, min=0.0)
    clearance = torch.linalg.vector_norm(positive_gap, dim=1)
    seam_mask = any_positive & (clearance < band)
    seam_values = (1.0 - clearance / band) ** 2
    contact_mask = seam_mask | (~any_positive)
    penalty_values = torch.where(~any_positive, torch.ones_like(seam_values), seam_values)
    strict_overlap_mask = (~any_positive) & all_negative
    inter_x = torch.minimum(torch.minimum(-gap_xy[:, 0], sizes_cpu[left, 0]), sizes_cpu[right, 0])
    inter_y = torch.minimum(torch.minimum(-gap_xy[:, 1], sizes_cpu[left, 1]), sizes_cpu[right, 1])
    intersections = inter_x * inter_y
    min_areas = torch.minimum(box_areas[left], box_areas[right])
    area_severity_values = (intersections / min_areas)[strict_overlap_mask]

    seam_severity_sum = 0.0
    for value in seam_values[seam_mask].tolist():
        seam_severity_sum += value
    clearance_penalty = 0.0
    for value in penalty_values[contact_mask].tolist():
        clearance_penalty += value
    area_severity_sum = 0.0
    for value in area_severity_values.tolist():
        area_severity_sum += value

    strict_overlap_count = int(strict_overlap_mask.sum().item())
    n_abut = int(((~any_positive) & ~all_negative).sum().item())
    clearance_contact_pairs = int(contact_mask.sum().item())
    packed_seam_severity = seam_severity_sum / max(1, count)
    score = 1.0 / (1.0 + 2.0 * clearance_penalty / max(1, count))
    return {
        **legacy,
        "overlap_count": strict_overlap_count,
        "node_occlusion_score": max(0.0, min(1.0, score)),
        "legacy_node_occlusion_score": float(legacy["node_occlusion_score"]),
        "clearance_penalty": clearance_penalty,
        "clearance_contact_pairs": clearance_contact_pairs,
        "clearance_band_node_diagonals": C4_CLEARANCE_BAND_NODE_DIAGONALS,
        "overlap_area_severity": area_severity_sum / max(1, count),
        "packed_seam_severity": packed_seam_severity,
        "clearance_abut_count": n_abut,
        "visual_packing_fill": area_sum / bbox_area,
        "label_inclusive": label_inclusive,
    }


def _fast_visual_occlusion_score(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    label_sizes: Optional[torch.Tensor],
    label_offsets: Optional[torch.Tensor],
    *,
    seed: Optional[int],
) -> Dict[str, float]:
    """Return C4 on node visual extents using the runtime fast C4 kernel.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node box sizes with shape ``[N, 2]``.
    label_sizes : Optional[torch.Tensor]
        Optional label extents with shape ``[N, 2]``.
    label_offsets : Optional[torch.Tensor]
        Optional label center offsets with shape ``[N, 2]``.
    seed : Optional[int]
        Frozen seed forwarded to the overlap counter.

    Returns
    -------
    dict[str, float]
        C4 score and diagnostics.
    """
    if label_sizes is None or label_offsets is None:
        centers, sizes = pos, node_sizes
        label_inclusive = False
    else:
        boxes = _node_visual_boxes(pos, node_sizes, label_sizes, label_offsets)
        centers, sizes = _boxes_to_centers_sizes(boxes)
        label_inclusive = True
    return fast_smooth_clearance_occlusion_score(
        centers,
        sizes,
        label_inclusive=label_inclusive,
        seed=seed,
    )


def _runtime_v3_graph_meta(problem: LayoutProblem) -> Dict[str, Any]:
    """Return runtime-visible metadata for the restricted V3 referee.

    Parameters
    ----------
    problem : LayoutProblem
        Native layout problem carrying user-visible graph declarations.

    Returns
    -------
    dict[str, Any]
        Input metadata for V3 groups. Corpus-only answer-key fields are
        deliberately excluded.
    """
    meta: Dict[str, Any] = {}
    structure = getattr(problem, "structure", None)
    declared_hierarchical = bool(
        getattr(structure, "is_semantically_directed", False)
        and getattr(structure, "is_directed_acyclic", getattr(structure, "is_acyclic", False))
    )
    if declared_hierarchical:
        meta["declared_hierarchical"] = True
        meta["flow_direction"] = str(problem.direction)
    if problem.clusters:
        meta["clusters"] = problem.clusters
    if problem.cluster_parents:
        meta["cluster_parents"] = problem.cluster_parents
    if problem.edge_weights is not None:
        edge_weights = problem.edge_weights.detach().to(device="cpu", dtype=torch.float64).flatten()
        edge_count = int(problem.edge_index.shape[1]) if problem.edge_index.ndim == 2 else 0
        if (
            int(edge_weights.numel()) == edge_count
            and edge_count > 0
            and bool(torch.isfinite(edge_weights).all().item())
            and bool((edge_weights > 0.0).all().item())
        ):
            meta["edge_weights"] = edge_weights.tolist()
            meta["weight_mode"] = "distance"
    return meta


def score_v3_runtime(
    pos: torch.Tensor,
    problem: LayoutProblem,
    *,
    all_pairs_dist: Optional[np.ndarray] = None,
) -> Tuple[Tuple[int, float], float, Mapping[str, RulerV3Facet]]:
    """Score one finalist with the runtime-visible frozen V3 referee.

    Parameters
    ----------
    pos : torch.Tensor
        Candidate node positions with shape ``[N, 2]``.
    problem : LayoutProblem
        Native layout problem providing topology and runtime-visible metadata.
    all_pairs_dist : numpy.ndarray, optional
        Cached unweighted shortest-path distances with shape ``[N, N]``.

    Returns
    -------
    tuple[tuple[int, float], float, Mapping[str, RulerV3Facet]]
        Severe-G6 eligibility key, tiered V3 headline score, and facet records.
    """
    result = score_v3_runtime_result(pos, problem, all_pairs_dist=all_pairs_dist)
    return referee_eligibility_key(result), float(result.scores["tiered"]), result.facets


def score_v3_runtime_result(
    pos: torch.Tensor,
    problem: LayoutProblem,
    *,
    all_pairs_dist: Optional[np.ndarray] = None,
) -> RulerV3Result:
    """Return the full restricted V3 result for one native finalist.

    Parameters
    ----------
    pos : torch.Tensor
        Candidate node positions with shape ``[N, 2]``.
    problem : LayoutProblem
        Native layout problem providing topology and runtime-visible metadata.
    all_pairs_dist : numpy.ndarray, optional
        Cached unweighted shortest-path distances with shape ``[N, N]``.

    Returns
    -------
    RulerV3Result
        Frozen V3 result, restricted only by runtime-visible graph metadata.
    """
    graph_meta = _runtime_v3_graph_meta(problem)
    positions = _ensure_cpu(pos).to(dtype=torch.float64)
    edges = _normalize_edge_index(problem.edge_index)
    sizes, used_default_sizes = _normalize_node_sizes(problem.node_sizes, int(positions.shape[0]))
    labels, offsets = _normalize_label_geometry(None, None, int(positions.shape[0]))
    num_nodes = int(positions.shape[0])

    crossing_geometry = _crossing_pair_geometry(
        positions,
        edges,
        n_samples=_FROZEN_CROSSING_SAMPLES,
        seed=0,
    )
    c1 = _component_weighted_ksm(
        positions,
        edges,
        sizes,
        stress_sources=_FROZEN_STRESS_SOURCES,
        stress_targets=_FROZEN_STRESS_TARGETS,
        all_pairs_dist=all_pairs_dist,
    )
    c2 = angle_weighted_crossing_score(
        positions,
        edges,
        n_samples=_FROZEN_CROSSING_SAMPLES,
        seed=0,
        _geometry=crossing_geometry,
    )
    c3 = multi_radius_neighborhood_preservation(
        positions,
        edges,
        num_nodes=num_nodes,
        radii=(1, 2, 3),
        n_samples=_FROZEN_NEIGHBORHOOD_SAMPLES,
        all_pairs_dist=all_pairs_dist,
    )
    c4 = _fast_visual_occlusion_score(positions, sizes, labels, offsets, seed=0)
    c5 = whitespace_sprawl_score(
        positions,
        edges,
        sizes,
        label_sizes=labels,
        label_offsets=offsets,
    )
    c6 = crossing_angle_90_score(
        positions,
        edges,
        n_samples=_FROZEN_CROSSING_SAMPLES,
        seed=0,
        _geometry=crossing_geometry,
    )
    c7 = gabriel_score(positions, edges)
    c8 = path_continuity_score(positions, edges)
    c9 = angular_resolution_score(positions, edges)
    c10 = edge_length_deviation_score(positions, edges, declared_targets=None)
    length_stats = edge_length_cv(positions, edges)

    node_diag_mean = _mean_node_diagonal(sizes)
    whitespace_ratio = float(c5["whitespace_ratio"])
    coincident_collapse_fraction = _coincident_collapse_fraction(positions, sizes)
    degenerate_scale = _is_degenerate_scale(
        float(length_stats["edge_length_mean"]),
        node_diag_mean,
    )
    raw_scores: Dict[str, Optional[float]] = {
        "C1": None if edges.shape[1] == 0 else float(c1["ksm_score"]),
        "C2": float(c2["edge_crossing_score"]),
        "C3": _optional_float(c3["multi_radius_neighborhood_preservation_score"]),
        "C4": float(c4["node_occlusion_score"]),
        "C5": float(c5["whitespace_sprawl_score"]),
        "C6": float(c6["crossing_angle_score"]),
        "C7": float(c7["gabriel_score"]),
        "C8": _optional_float(c8["path_continuity_score"]),
        "C9": _optional_float(c9["angular_resolution_score"]),
        "C10": None if edges.shape[1] == 0 else float(c10["edge_length_deviation_score"]),
    }
    if degenerate_scale:
        raw_scores["C5"] = 0.0
    facets = _build_facets(
        raw_scores,
        {
            "C1": c1,
            "C2": c2,
            "C3": c3,
            "C4": {
                **c4,
                "default_node_sizes": used_default_sizes,
                "label_inclusive": labels is not None,
            },
            "C5": c5,
            "C6": c6,
            "C7": c7,
            "C8": c8,
            "C9": c9,
            "C10": c10,
        },
        num_nodes=num_nodes,
    )
    group_results = evaluate_conditional_groups(positions, edges, sizes, graph_meta)
    facets = _merge_conditional_group_facets(facets, group_results)
    scores = _triple_view_scores(facets, graph_meta)
    sprawl_collapse = _is_sprawl_collapse(
        occlusion_score=raw_scores["C4"],
        whitespace_ratio=whitespace_ratio,
    )
    scores = _apply_headline_degeneracy_fold(
        scores,
        edge_length_mean=float(length_stats["edge_length_mean"]),
        node_diag_mean=node_diag_mean,
        degenerate_scale=degenerate_scale,
        sprawl_collapse=sprawl_collapse,
        occlusion_score=raw_scores["C4"],
        whitespace_ratio=whitespace_ratio,
        overlap_count=int(c4["overlap_count"]),
        overlap_area_severity=float(c4["overlap_area_severity"]),
        clearance_penalty=float(c4["clearance_penalty"]),
        clearance_contact_pairs=int(c4["clearance_contact_pairs"]),
        visual_packing_fill=float(c4["visual_packing_fill"]),
        num_nodes=num_nodes,
    )
    flags = _row_flags(
        degenerate_scale=degenerate_scale,
        occlusion_score=raw_scores["C4"],
        whitespace_ratio=whitespace_ratio,
        sprawl_collapse=sprawl_collapse,
        coincident_collapse_fraction=coincident_collapse_fraction,
        conditional_group_flags=_conditional_group_flags(group_results),
        severe_g6_breach_flag=_severe_g6_breach_from_facets(facets),
    )
    return RulerV3Result(
        facets=facets,
        scores=scores,
        flags=flags,
        applicability={code: facet.applicable for code, facet in facets.items()},
        coverage={
            "applicable_facets": sum(1 for facet in facets.values() if facet.applicable),
            "total_facets": len(facets),
            "tier1_applicable_facets": sum(
                1 for facet in facets.values() if facet.applicable and facet.tier == 1
            ),
            "applicable_groups": sum(1 for group in group_results.values() if group.applicable),
        },
        metadata={
            "num_nodes": num_nodes,
            "num_edges": int(edges.shape[1]) if edges.numel() else 0,
            "frozen_constants_manifest": "dagua.eval.ruler_v3_frozen.FROZEN_CONSTANTS",
            "frozen_constant_count": len(FROZEN_CONSTANTS),
            "softmin_family": _score_family(facets, graph_meta),
            "softmin_tau": FAMILY_SOFTMIN_TAU,
            "default_node_sizes": used_default_sizes,
            "node_diag_mean": node_diag_mean,
            "edge_length_mean": float(length_stats["edge_length_mean"]),
            "coincident_collapse_fraction": coincident_collapse_fraction,
            "headline_degeneracy_fold": _headline_degeneracy_fold(
                edge_length_mean=float(length_stats["edge_length_mean"]),
                node_diag_mean=node_diag_mean,
                degenerate_scale=degenerate_scale,
                sprawl_collapse=sprawl_collapse,
                occlusion_score=raw_scores["C4"],
                whitespace_ratio=whitespace_ratio,
                overlap_count=int(c4["overlap_count"]),
                overlap_area_severity=float(c4["overlap_area_severity"]),
                clearance_penalty=float(c4["clearance_penalty"]),
                clearance_contact_pairs=int(c4["clearance_contact_pairs"]),
                visual_packing_fill=float(c4["visual_packing_fill"]),
                num_nodes=num_nodes,
            ),
            "crossing_weight_multiplier": crossing_weight_multiplier(num_nodes),
            "conditional_groups": group_results,
        },
    )


def v3_proxy_fold(quick_metrics: Mapping[str, Any], num_nodes: int) -> float:
    """Return a cheap V3-shaped proxy for shortlist ordering only.

    Parameters
    ----------
    quick_metrics : Mapping[str, Any]
        Metrics payload from ``dagua.metrics.quick``.
    num_nodes : int
        Number of nodes, used for the frozen crossing multiplier shape.

    Returns
    -------
    float
        Tier-renormalized proxy score on a 0--100 scale.
    """
    crossing_weight = (4.0 + 2.0) * crossing_weight_multiplier(num_nodes)
    weighted: list[tuple[float, float]] = []
    for key, weight in (
        ("ksm_score", 4.0),
        ("edge_crossing_score", crossing_weight),
        ("neighborhood_preservation_score", 4.0),
        ("node_occlusion_score", 2.0),
        ("whitespace_sprawl_score", 2.0),
        ("gabriel_score", 2.0),
        ("angular_resolution_score", 1.0),
        ("edge_length_deviation_score", 1.0),
    ):
        value = quick_metrics.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            weighted.append((float(value), weight))
    total_weight = sum(weight for _, weight in weighted)
    if total_weight <= 0.0:
        return 0.0
    return 100.0 * sum(score * weight for score, weight in weighted) / total_weight


__all__ = [
    "V3_REFEREE_DWU_FRACTION",
    "_runtime_v3_graph_meta",
    "fast_smooth_clearance_occlusion_score",
    "score_v3_runtime",
    "score_v3_runtime_result",
    "v3_proxy_fold",
]
