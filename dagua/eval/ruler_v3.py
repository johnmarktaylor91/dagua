"""Scoring-only V3 quality ruler core facets.

This module is intentionally isolated from layout generation and the frozen V2
metric module. It computes the Phase-1 V3 CORE ruler facets, applicability
metadata, tiered/equal/Tier-1-only composites, and hard row flags without
changing any optimizer or selector behavior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from dagua.metrics import (
    _all_pairs_unweighted,
    _build_csr,
    _crossing_pair_geometry,
    _deterministic_sample_indices,
    _ensure_cpu,
    angular_resolution_score,
    edge_crossing_score,
    edge_length_cv,
    edge_length_deviation_score,
    gabriel_score,
    isotonic_stress,
    node_occlusion_score,
    path_continuity_score,
)

TIER_1_WEIGHT = 4.0
TIER_2_WEIGHT = 2.0
TIER_3_WEIGHT = 1.0

CORE_TIERS: Dict[str, int] = {
    "C1": 1,
    "C2": 1,
    "C3": 1,
    "C4": 2,
    "C5": 2,
    "C6": 2,
    "C7": 2,
    "C8": 2,
    "C9": 3,
    "C10": 3,
}
CORE_NAMES: Dict[str, str] = {
    "C1": "ksm_stress",
    "C2": "degree_corrected_crossings",
    "C3": "multi_radius_neighborhood_preservation",
    "C4": "node_occlusion",
    "C5": "whitespace_sprawl_band",
    "C6": "crossing_angle",
    "C7": "gabriel_edge_node_occlusion",
    "C8": "path_continuity",
    "C9": "angular_resolution",
    "C10": "edge_length_uniformity",
}
PURE_GEOMETRY_FACETS = ("C1", "C2", "C3", "C6", "C7", "C8", "C9", "C10")

CROSSING_DECAY_N_MID = 700.0
CROSSING_DECAY_S = 0.2
DEFAULT_NODE_WIDTH = 1.0
DEFAULT_NODE_HEIGHT = 1.0
DEGENERATE_SCALE_RATIO = 0.25
OCCLUSION_FLOOR_THRESHOLD = 0.75
WHITESPACE_CONTENT_MULTIPLIER = 2.0
WHITESPACE_RATIO_LO = 1.0
WHITESPACE_RATIO_HI = 16.0
WHITESPACE_CROWDING_DECAY = 0.5
WHITESPACE_SPRAWL_DECAY = 1.25
SPRAWL_RATIO_FACTOR = 4.0


@dataclass(frozen=True)
class RulerV3Facet:
    """Single V3 ruler facet publication record."""

    code: str
    name: str
    tier: int
    score: Optional[float]
    base_weight: float
    effective_weight: float
    applicable: bool
    applicability_reason: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class RulerV3Result:
    """Composite V3 ruler output for one layout."""

    facets: Mapping[str, RulerV3Facet]
    scores: Mapping[str, float]
    flags: Tuple[str, ...]
    applicability: Mapping[str, bool]
    coverage: Mapping[str, int]
    metadata: Mapping[str, Any]


def tier_weight(tier: int) -> float:
    """Return the frozen V3 multiplier for an evidence tier.

    Parameters
    ----------
    tier : int
        Evidence tier number, where 1 is strongest and 3 is weakest.

    Returns
    -------
    float
        Frozen tier multiplier.
    """
    if tier == 1:
        return TIER_1_WEIGHT
    if tier == 2:
        return TIER_2_WEIGHT
    if tier == 3:
        return TIER_3_WEIGHT
    raise ValueError(f"unsupported V3 tier: {tier}")


def crossing_weight_multiplier(num_nodes: int) -> float:
    """Compute the frozen log-size crossing weight multiplier.

    Parameters
    ----------
    num_nodes : int
        Input graph node count.

    Returns
    -------
    float
        Multiplier in ``[0.5, 1.0]`` applied to C2's aggregate weight only.
    """
    n = max(1.0, float(num_nodes))
    exponent = (math.log10(CROSSING_DECAY_N_MID) - math.log10(n)) / CROSSING_DECAY_S
    sigmoid = 1.0 / (1.0 + math.exp(-exponent))
    return 0.5 + 0.5 * sigmoid


def renormalized_score(
    values: Mapping[str, Optional[float]],
    weights: Mapping[str, float],
) -> float:
    """Aggregate applicable score terms without imputing missing facets.

    Parameters
    ----------
    values : Mapping[str, Optional[float]]
        Score values in ``[0, 1]``. Missing, ``None``, and non-finite values are
        excluded from both numerator and denominator.
    weights : Mapping[str, float]
        Frozen non-negative weights keyed by score name.

    Returns
    -------
    float
        Renormalized composite score on a 0--100 scale.
    """
    applicable: List[Tuple[float, float]] = []
    for name, weight in weights.items():
        value = values.get(name)
        if value is None:
            continue
        finite_value = float(value)
        if not math.isfinite(finite_value):
            continue
        applicable.append((float(weight), max(0.0, min(1.0, finite_value))))
    total_weight = sum(weight for weight, _ in applicable)
    if total_weight <= 0.0:
        return 0.0
    return 100.0 * sum(weight * value for weight, value in applicable) / total_weight


def score_core_v3(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: Optional[torch.Tensor] = None,
    *,
    label_sizes: Optional[torch.Tensor] = None,
    label_offsets: Optional[torch.Tensor] = None,
    edge_length_targets: Optional[torch.Tensor] = None,
    stress_sources: int = 200,
    stress_targets: int = 1000,
    crossing_samples: int = 1_000_000,
    neighborhood_samples: int = 5000,
    all_pairs_dist: Optional[np.ndarray] = None,
    seed: int = 0,
) -> RulerV3Result:
    """Score the V3 CORE ruler facets for a straight-line layout.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : Optional[torch.Tensor], optional
        Node box sizes with shape ``[N, 2]``. A frozen unit box is used when
        omitted so the CORE size-anchored facets remain applicable.
    label_sizes : Optional[torch.Tensor], optional
        Optional node-label extents with shape ``[N, 2]`` for C5.
    label_offsets : Optional[torch.Tensor], optional
        Optional label center offsets from each node center with shape ``[N, 2]``.
    edge_length_targets : Optional[torch.Tensor], optional
        Optional declared edge-length targets with shape ``[E]`` for C10's
        common-scale target fit.
    stress_sources : int, optional
        Deterministic source budget for C1 KSM.
    stress_targets : int, optional
        Deterministic per-source target budget for C1 KSM.
    crossing_samples : int, optional
        Frozen edge-pair sampling budget for C2 and C6.
    neighborhood_samples : int, optional
        Deterministic node-row budget for C3.
    all_pairs_dist : Optional[numpy.ndarray], optional
        Optional precomputed graph-distance matrix with shape ``[N, N]``.
    seed : int, optional
        Frozen seed for sampled V2-compatible primitives.

    Returns
    -------
    RulerV3Result
        Per-facet records, triple-view composites, row flags, and metadata.
    """
    positions = _ensure_cpu(pos).to(dtype=torch.float64)
    edges = _normalize_edge_index(edge_index)
    sizes, used_default_sizes = _normalize_node_sizes(node_sizes, int(positions.shape[0]))
    labels, offsets = _normalize_label_geometry(label_sizes, label_offsets, int(positions.shape[0]))
    num_nodes = int(positions.shape[0])

    crossing_geometry = _crossing_pair_geometry(
        positions, edges, n_samples=crossing_samples, seed=seed
    )
    c1 = _component_weighted_ksm(
        positions,
        edges,
        sizes,
        stress_sources=stress_sources,
        stress_targets=stress_targets,
        all_pairs_dist=all_pairs_dist,
    )
    c2 = edge_crossing_score(
        positions,
        edges,
        n_samples=crossing_samples,
        seed=seed,
        _geometry=crossing_geometry,
    )
    c3 = multi_radius_neighborhood_preservation(
        positions,
        edges,
        num_nodes=num_nodes,
        radii=(1, 2, 3),
        n_samples=neighborhood_samples,
        all_pairs_dist=all_pairs_dist,
    )
    c4 = node_occlusion_score(positions, sizes, seed=seed)
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
        n_samples=crossing_samples,
        seed=seed,
        _geometry=crossing_geometry,
    )
    c7 = gabriel_score(positions, edges)
    c8 = path_continuity_score(positions, edges)
    c9 = angular_resolution_score(positions, edges)
    c10 = edge_length_deviation_score(positions, edges, declared_targets=edge_length_targets)
    length_stats = edge_length_cv(positions, edges)

    node_diag_mean = _mean_node_diagonal(sizes)
    degenerate_scale = _is_degenerate_scale(
        float(length_stats["edge_length_mean"]),
        node_diag_mean,
    )
    raw_scores: Dict[str, Optional[float]] = {
        "C1": float(c1["ksm_score"]),
        "C2": float(c2["edge_crossing_score"]),
        "C3": float(c3["multi_radius_neighborhood_preservation_score"]),
        "C4": float(c4["node_occlusion_score"]),
        "C5": float(c5["whitespace_sprawl_score"]),
        "C6": float(c6["crossing_angle_score"]),
        "C7": float(c7["gabriel_score"]),
        "C8": _optional_float(c8["path_continuity_score"]),
        "C9": _optional_float(c9["angular_resolution_score"]),
        "C10": float(c10["edge_length_deviation_score"]),
    }
    if degenerate_scale:
        raw_scores["C5"] = 0.0

    facets = _build_facets(
        raw_scores,
        {
            "C1": c1,
            "C2": c2,
            "C3": c3,
            "C4": {**c4, "default_node_sizes": used_default_sizes},
            "C5": c5,
            "C6": c6,
            "C7": c7,
            "C8": c8,
            "C9": c9,
            "C10": c10,
        },
        num_nodes=num_nodes,
    )
    scores = _triple_view_scores(facets)
    flags = _row_flags(
        degenerate_scale=degenerate_scale,
        occlusion_score=raw_scores["C4"],
        whitespace_ratio=float(c5["whitespace_ratio"]),
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
        },
        metadata={
            "num_nodes": num_nodes,
            "num_edges": int(edges.shape[1]) if edges.numel() else 0,
            "default_node_sizes": used_default_sizes,
            "node_diag_mean": node_diag_mean,
            "edge_length_mean": float(length_stats["edge_length_mean"]),
            "crossing_weight_multiplier": crossing_weight_multiplier(num_nodes),
        },
    )


def composite_v3(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> float:
    """Return the tiered V3 CORE headline score for one layout.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : Optional[torch.Tensor], optional
        Node box sizes with shape ``[N, 2]``.
    **kwargs : Any
        Additional keyword arguments forwarded to :func:`score_core_v3`.

    Returns
    -------
    float
        Tiered V3 CORE headline score on a 0--100 scale.
    """
    return float(score_core_v3(pos, edge_index, node_sizes, **kwargs).scores["tiered"])


def multi_radius_neighborhood_preservation(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    num_nodes: Optional[int] = None,
    radii: Sequence[int] = (1, 2, 3),
    n_samples: int = 5000,
    all_pairs_dist: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Score graph-distance neighborhood preservation over multiple radii.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : Optional[int], optional
        Node count. Defaults to the first dimension of ``pos``.
    radii : Sequence[int], optional
        Inclusive graph-distance radii to compare against layout k-nearest
        neighborhoods.
    n_samples : int, optional
        Maximum number of deterministic query rows.
    all_pairs_dist : Optional[numpy.ndarray], optional
        Optional precomputed graph-distance matrix with shape ``[N, N]``.

    Returns
    -------
    Dict[str, Any]
        Mean multi-radius Jaccard score and per-radius publication metadata.
    """
    positions = _ensure_cpu(pos).to(dtype=torch.float64)
    edges = _normalize_edge_index(edge_index)
    n = int(positions.shape[0]) if num_nodes is None else int(num_nodes)
    if n <= 1:
        return {
            "multi_radius_neighborhood_preservation_score": 1.0,
            "neighborhood_radius_scores": {},
            "neighborhood_n_rows": 0,
        }
    distances = _graph_distances(edges, n, max(radii), all_pairs_dist)
    rows_np = _deterministic_sample_indices(n, min(n, n_samples))
    layout_distances = torch.cdist(positions[torch.from_numpy(rows_np)], positions).numpy()
    radius_scores: Dict[int, float] = {}
    for radius in radii:
        radius_scores[int(radius)] = _radius_jaccard_score(
            distances,
            layout_distances,
            rows_np,
            int(radius),
        )
    values = list(radius_scores.values())
    return {
        "multi_radius_neighborhood_preservation_score": float(np.mean(values)) if values else 0.0,
        "neighborhood_radius_scores": radius_scores,
        "neighborhood_n_rows": int(len(rows_np)),
    }


def crossing_angle_90_score(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    n_samples: int = 1_000_000,
    *,
    seed: int = 0,
    _geometry: Optional[Any] = None,
) -> Dict[str, float]:
    """Score crossing angles against the V3 90-degree ideal.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    n_samples : int, optional
        Frozen edge-pair sampling budget.
    seed : int, optional
        Frozen sampling seed.
    _geometry : Optional[Any], optional
        Optional crossing geometry from the frozen crossing enumerator.

    Returns
    -------
    Dict[str, float]
        Crossing-angle score, crossing count, and sampled pair count.
    """
    geometry = _geometry or _crossing_pair_geometry(pos, edge_index, n_samples=n_samples, seed=seed)
    p1, p2, p3, p4, crossing = geometry[:5]
    if not bool(crossing.any().item()):
        return {
            "crossing_angle_score": 1.0,
            "crossing_angle_count": 0,
            "crossing_angle_n_pairs": int(crossing.numel()),
            "crossing_angle_ideal_degrees": 90.0,
        }
    first = p2[crossing] - p1[crossing]
    second = p4[crossing] - p3[crossing]
    norms = torch.linalg.vector_norm(first, dim=1) * torch.linalg.vector_norm(second, dim=1)
    cosine = torch.where(
        norms > 1e-12,
        (first * second).sum(dim=1).abs() / norms.clamp(min=1e-12),
        torch.ones_like(norms),
    ).clamp(0.0, 1.0)
    angles = torch.acos(cosine)
    score = torch.clamp(angles / (math.pi / 2.0), max=1.0).mean().item()
    return {
        "crossing_angle_score": float(max(0.0, min(1.0, score))),
        "crossing_angle_count": int(crossing.sum()),
        "crossing_angle_n_pairs": int(crossing.numel()),
        "crossing_angle_ideal_degrees": 90.0,
    }


def whitespace_sprawl_score(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    *,
    label_sizes: Optional[torch.Tensor] = None,
    label_offsets: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Score label-inclusive visual area against a frozen content floor.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]`` used for degenerate-scale metadata.
    node_sizes : torch.Tensor
        Node box sizes with shape ``[N, 2]``.
    label_sizes : Optional[torch.Tensor], optional
        Optional node-label extents with shape ``[N, 2]``.
    label_offsets : Optional[torch.Tensor], optional
        Optional label center offsets from node centers with shape ``[N, 2]``.

    Returns
    -------
    Dict[str, float]
        Whitespace band score, visual area, floor, and ratio metadata.
    """
    positions = _ensure_cpu(pos).to(dtype=torch.float64)
    edges = _normalize_edge_index(edge_index)
    sizes = _ensure_cpu(node_sizes).to(dtype=torch.float64)
    labels, offsets = _normalize_label_geometry(label_sizes, label_offsets, int(positions.shape[0]))
    node_boxes = _centered_boxes(positions, sizes)
    boxes = list(node_boxes)
    if labels is not None and offsets is not None:
        boxes.extend(_centered_boxes(positions + offsets, labels))
    visual_area = _union_bbox_area(boxes)
    content_area = _content_area(sizes, labels)
    structure_floor = _generic_structure_area_floor(sizes, edges)
    area_floor = max(WHITESPACE_CONTENT_MULTIPLIER * content_area, structure_floor, 1e-12)
    ratio = visual_area / area_floor
    return {
        "whitespace_sprawl_score": _log_band_score(ratio),
        "whitespace_visual_area": visual_area,
        "whitespace_area_floor": area_floor,
        "whitespace_ratio": ratio,
        "whitespace_structure_area_floor": structure_floor,
        "whitespace_content_area": content_area,
    }


def _normalize_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    """Return a CPU long edge tensor with shape ``[2, E]``.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor to validate and normalize.

    Returns
    -------
    torch.Tensor
        CPU ``int64`` edge tensor with shape ``[2, E]``.
    """
    edges = _ensure_cpu(edge_index).to(dtype=torch.long)
    if edges.ndim != 2 or int(edges.shape[0]) != 2:
        raise ValueError("edge_index must have shape [2, E]")
    return edges


def _normalize_node_sizes(
    node_sizes: Optional[torch.Tensor],
    num_nodes: int,
) -> Tuple[torch.Tensor, bool]:
    """Return node sizes, using a frozen unit-box default when absent.

    Parameters
    ----------
    node_sizes : Optional[torch.Tensor]
        Optional node box sizes with shape ``[N, 2]``.
    num_nodes : int
        Number of nodes in the layout.

    Returns
    -------
    Tuple[torch.Tensor, bool]
        Normalized sizes and whether the default was used.
    """
    if node_sizes is None:
        default_size = torch.tensor(
            [DEFAULT_NODE_WIDTH, DEFAULT_NODE_HEIGHT],
            dtype=torch.float64,
        )
        return (
            default_size.repeat(num_nodes, 1),
            True,
        )
    sizes = _ensure_cpu(node_sizes).to(dtype=torch.float64)
    if sizes.shape != (num_nodes, 2):
        raise ValueError("node_sizes must have shape [N, 2]")
    if bool((sizes <= 0).any().item()):
        raise ValueError("node_sizes must be positive")
    return sizes, False


def _normalize_label_geometry(
    label_sizes: Optional[torch.Tensor],
    label_offsets: Optional[torch.Tensor],
    num_nodes: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Validate optional label extents and offsets.

    Parameters
    ----------
    label_sizes : Optional[torch.Tensor]
        Optional label box sizes with shape ``[N, 2]``.
    label_offsets : Optional[torch.Tensor]
        Optional label center offsets with shape ``[N, 2]``.
    num_nodes : int
        Number of nodes in the layout.

    Returns
    -------
    Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]
        Normalized label sizes and offsets, or ``(None, None)``.
    """
    if label_sizes is None:
        return None, None
    labels = _ensure_cpu(label_sizes).to(dtype=torch.float64)
    if labels.shape != (num_nodes, 2):
        raise ValueError("label_sizes must have shape [N, 2]")
    if bool((labels < 0).any().item()):
        raise ValueError("label_sizes must be non-negative")
    if label_offsets is None:
        offsets = torch.zeros((num_nodes, 2), dtype=torch.float64)
    else:
        offsets = _ensure_cpu(label_offsets).to(dtype=torch.float64)
        if offsets.shape != (num_nodes, 2):
            raise ValueError("label_offsets must have shape [N, 2]")
    return labels, offsets


def _graph_distances(
    edge_index: torch.Tensor,
    num_nodes: int,
    max_dist: int,
    all_pairs_dist: Optional[np.ndarray],
) -> np.ndarray:
    """Return graph distances with unreachable entries encoded as ``-1``.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    max_dist : int
        Maximum graph distance needed by the caller.
    all_pairs_dist : Optional[numpy.ndarray]
        Optional precomputed distance matrix with shape ``[N, N]``.

    Returns
    -------
    numpy.ndarray
        Integer distance matrix with shape ``[N, N]``.
    """
    if all_pairs_dist is not None:
        return all_pairs_dist
    offsets, targets = _build_csr(edge_index, num_nodes)
    return _all_pairs_unweighted(offsets, targets, num_nodes, max_dist=max_dist)


def _radius_jaccard_score(
    graph_distances: np.ndarray,
    layout_distances: np.ndarray,
    rows: np.ndarray,
    radius: int,
) -> float:
    """Score one graph-distance radius against layout nearest neighbors.

    Parameters
    ----------
    graph_distances : numpy.ndarray
        Graph-distance matrix with shape ``[N, N]`` and ``-1`` for unreachable
        pairs.
    layout_distances : numpy.ndarray
        Layout distance rows with shape ``[Q, N]`` corresponding to ``rows``.
    rows : numpy.ndarray
        Query node indices with shape ``[Q]``.
    radius : int
        Inclusive graph-distance radius.

    Returns
    -------
    float
        Mean Jaccard score over rows with a non-empty graph neighborhood.
    """
    row_scores: List[float] = []
    for row_offset, node in enumerate(rows.tolist()):
        graph_neighbors = set(
            int(index)
            for index in np.flatnonzero(
                (graph_distances[node] > 0) & (graph_distances[node] <= radius)
            ).tolist()
        )
        if not graph_neighbors:
            continue
        layout_row = layout_distances[row_offset].copy()
        layout_row[int(node)] = float("inf")
        nearest = np.argsort(layout_row, kind="mergesort")[: len(graph_neighbors)]
        layout_neighbors = set(int(index) for index in nearest.tolist())
        union = graph_neighbors | layout_neighbors
        row_scores.append(len(graph_neighbors & layout_neighbors) / len(union) if union else 1.0)
    return float(np.mean(row_scores)) if row_scores else 0.0


def _component_weighted_ksm(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    *,
    stress_sources: int,
    stress_targets: int,
    all_pairs_dist: Optional[np.ndarray],
) -> Dict[str, float]:
    """Compute KSM per component and combine by visual hull weights.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node box sizes with shape ``[N, 2]`` used only as a zero-area fallback.
    stress_sources : int
        Deterministic source budget for each component.
    stress_targets : int
        Deterministic target budget for each component.
    all_pairs_dist : Optional[numpy.ndarray]
        Optional graph-distance matrix for the full graph.

    Returns
    -------
    Dict[str, float]
        Component-weighted ``ksm_score`` and sample metadata.
    """
    components = _connected_components(edge_index, int(pos.shape[0]))
    nontrivial = [component for component in components if len(component) > 1]
    if len(nontrivial) <= 1:
        return isotonic_stress(
            pos,
            edge_index,
            int(pos.shape[0]),
            n_sources=stress_sources,
            n_targets=stress_targets,
            all_pairs_dist=all_pairs_dist,
        )
    weights = [_component_visual_weight(pos, node_sizes, component) for component in nontrivial]
    if sum(weights) <= 1e-12:
        weights = [float(len(component) * (len(component) - 1)) for component in nontrivial]
    weighted_score = 0.0
    weighted_se = 0.0
    total_pairs = 0
    total_sources = 0
    total_weight = sum(weights)
    for component, weight in zip(nontrivial, weights):
        component_edges = _induced_component_edges(edge_index, component)
        component_pos = pos[component]
        component_dist = None
        if all_pairs_dist is not None:
            component_dist = all_pairs_dist[np.ix_(component, component)]
        result = isotonic_stress(
            component_pos,
            component_edges,
            len(component),
            n_sources=min(stress_sources, len(component)),
            n_targets=stress_targets,
            all_pairs_dist=component_dist,
        )
        share = weight / total_weight
        weighted_score += share * float(result["ksm_score"])
        weighted_se += share * float(result["ksm_se"])
        total_pairs += int(result["ksm_n_pairs"])
        total_sources += int(result["ksm_n_sources"])
    return {
        "ksm_score": max(0.0, min(1.0, weighted_score)),
        "ksm_se": weighted_se,
        "ksm_n_pairs": total_pairs,
        "ksm_n_sources": total_sources,
        "ksm_n_components": len(nontrivial),
    }


def _connected_components(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
    """Find undirected graph components.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    List[List[int]]
        Components as sorted node-index lists.
    """
    adjacency: List[List[int]] = [[] for _ in range(num_nodes)]
    for source, target in edge_index.t().tolist() if edge_index.numel() else []:
        s = int(source)
        t = int(target)
        if s == t:
            continue
        adjacency[s].append(t)
        adjacency[t].append(s)
    seen = [False] * num_nodes
    components: List[List[int]] = []
    for start in range(num_nodes):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        component: List[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def _induced_component_edges(edge_index: torch.Tensor, component: Sequence[int]) -> torch.Tensor:
    """Remap edges induced by a component to local node ids.

    Parameters
    ----------
    edge_index : torch.Tensor
        Full edge tensor with shape ``[2, E]``.
    component : Sequence[int]
        Sorted global node ids in one component.

    Returns
    -------
    torch.Tensor
        Local edge tensor with shape ``[2, E_c]``.
    """
    local = {node: index for index, node in enumerate(component)}
    pairs: List[Tuple[int, int]] = []
    for source, target in edge_index.t().tolist() if edge_index.numel() else []:
        s = int(source)
        t = int(target)
        if s in local and t in local:
            pairs.append((local[s], local[t]))
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def _component_visual_weight(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    component: Sequence[int],
) -> float:
    """Return the hull-area visual weight for one component.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node box sizes with shape ``[N, 2]``.
    component : Sequence[int]
        Node ids in one component.

    Returns
    -------
    float
        Convex-hull area, with zero-area components falling back to node-box
        content area.
    """
    points = pos[list(component)].numpy()
    hull_area = _convex_hull_area(points)
    if hull_area > 1e-12:
        return hull_area
    return float((node_sizes[list(component), 0] * node_sizes[list(component), 1]).sum().item())


def _convex_hull_area(points: np.ndarray) -> float:
    """Compute the area of a two-dimensional convex hull.

    Parameters
    ----------
    points : numpy.ndarray
        Point coordinates with shape ``[N, 2]``.

    Returns
    -------
    float
        Hull polygon area, or ``0`` for fewer than three unique points.
    """
    unique = sorted(set((float(x), float(y)) for x, y in points.tolist()))
    if len(unique) < 3:
        return 0.0

    def cross(
        origin: Tuple[float, float],
        first: Tuple[float, float],
        second: Tuple[float, float],
    ) -> float:
        """Return the signed turn area for three hull points.

        Parameters
        ----------
        origin : Tuple[float, float]
            Shared origin point.
        first : Tuple[float, float]
            First endpoint.
        second : Tuple[float, float]
            Second endpoint.

        Returns
        -------
        float
            Signed cross product.
        """
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (
            second[0] - origin[0]
        )

    lower: List[Tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: List[Tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    area = 0.0
    for index, point in enumerate(hull):
        next_point = hull[(index + 1) % len(hull)]
        area += point[0] * next_point[1] - next_point[0] * point[1]
    return abs(area) * 0.5


def _centered_boxes(
    centers: torch.Tensor,
    sizes: torch.Tensor,
) -> List[Tuple[float, float, float, float]]:
    """Build axis-aligned boxes from centers and sizes.

    Parameters
    ----------
    centers : torch.Tensor
        Box centers with shape ``[N, 2]``.
    sizes : torch.Tensor
        Box sizes with shape ``[N, 2]``.

    Returns
    -------
    List[Tuple[float, float, float, float]]
        Boxes as ``(min_x, min_y, max_x, max_y)`` tuples.
    """
    half = sizes / 2.0
    mins = centers - half
    maxes = centers + half
    return [
        (
            float(mins[index, 0]),
            float(mins[index, 1]),
            float(maxes[index, 0]),
            float(maxes[index, 1]),
        )
        for index in range(int(centers.shape[0]))
    ]


def _union_bbox_area(boxes: Sequence[Tuple[float, float, float, float]]) -> float:
    """Return the area of the bounding box enclosing input boxes.

    Parameters
    ----------
    boxes : Sequence[Tuple[float, float, float, float]]
        Axis-aligned boxes as ``(min_x, min_y, max_x, max_y)`` tuples.

    Returns
    -------
    float
        Enclosing bounding-box area.
    """
    if not boxes:
        return 0.0
    min_x = min(box[0] for box in boxes)
    min_y = min(box[1] for box in boxes)
    max_x = max(box[2] for box in boxes)
    max_y = max(box[3] for box in boxes)
    return max(0.0, max_x - min_x) * max(0.0, max_y - min_y)


def _content_area(node_sizes: torch.Tensor, label_sizes: Optional[torch.Tensor]) -> float:
    """Return total node and label content area.

    Parameters
    ----------
    node_sizes : torch.Tensor
        Node box sizes with shape ``[N, 2]``.
    label_sizes : Optional[torch.Tensor]
        Optional label box sizes with shape ``[N, 2]``.

    Returns
    -------
    float
        Sum of node and label rectangle areas.
    """
    area = float((node_sizes[:, 0] * node_sizes[:, 1]).sum().item())
    if label_sizes is not None:
        area += float((label_sizes[:, 0] * label_sizes[:, 1]).sum().item())
    return area


def _generic_structure_area_floor(node_sizes: torch.Tensor, edge_index: torch.Tensor) -> float:
    """Return the frozen generic structure area floor for C5.

    Parameters
    ----------
    node_sizes : torch.Tensor
        Node box sizes with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.

    Returns
    -------
    float
        Input-only generic floor based on content and graph size.
    """
    del edge_index
    content = _content_area(node_sizes, None)
    return WHITESPACE_CONTENT_MULTIPLIER * content


def _log_band_score(ratio: float) -> float:
    """Score an area ratio with flat plateau and asymmetric log decay.

    Parameters
    ----------
    ratio : float
        ``visual_area / area_floor``.

    Returns
    -------
    float
        Band score in ``[0, 1]``.
    """
    safe_ratio = max(float(ratio), 1e-12)
    if WHITESPACE_RATIO_LO <= safe_ratio <= WHITESPACE_RATIO_HI:
        return 1.0
    if safe_ratio < WHITESPACE_RATIO_LO:
        distance = math.log(WHITESPACE_RATIO_LO / safe_ratio)
        return math.exp(-WHITESPACE_CROWDING_DECAY * distance * distance)
    distance = math.log(safe_ratio / WHITESPACE_RATIO_HI)
    return math.exp(-WHITESPACE_SPRAWL_DECAY * distance * distance)


def _mean_node_diagonal(node_sizes: torch.Tensor) -> float:
    """Return the mean node-box diagonal.

    Parameters
    ----------
    node_sizes : torch.Tensor
        Node box sizes with shape ``[N, 2]``.

    Returns
    -------
    float
        Mean Euclidean box diagonal.
    """
    if int(node_sizes.shape[0]) == 0:
        return 0.0
    diagonal = torch.sqrt(node_sizes[:, 0] ** 2 + node_sizes[:, 1] ** 2)
    return float(diagonal.mean().item())


def _is_degenerate_scale(edge_length_mean: float, node_diag_mean: float) -> bool:
    """Detect point-collapse scale relative to node geometry.

    Parameters
    ----------
    edge_length_mean : float
        Mean edge length in drawing units.
    node_diag_mean : float
        Mean node-box diagonal in drawing units.

    Returns
    -------
    bool
        ``True`` when edges are too short to be legible relative to node boxes.
    """
    if node_diag_mean <= 1e-8:
        return False
    return edge_length_mean < DEGENERATE_SCALE_RATIO * node_diag_mean


def _optional_float(value: Any) -> Optional[float]:
    """Convert a possibly missing score value to ``Optional[float]``.

    Parameters
    ----------
    value : Any
        Score-like value.

    Returns
    -------
    Optional[float]
        Float value, or ``None`` when the input is missing or non-finite.
    """
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _build_facets(
    scores: Mapping[str, Optional[float]],
    metadata: Mapping[str, Mapping[str, Any]],
    *,
    num_nodes: int,
) -> Dict[str, RulerV3Facet]:
    """Build per-facet publication records.

    Parameters
    ----------
    scores : Mapping[str, Optional[float]]
        Raw facet scores keyed by V3 facet code.
    metadata : Mapping[str, Mapping[str, Any]]
        Per-facet metric metadata.
    num_nodes : int
        Input node count for size-decayed crossing weight.

    Returns
    -------
    Dict[str, RulerV3Facet]
        Facet records keyed by V3 facet code.
    """
    facets: Dict[str, RulerV3Facet] = {}
    for code in CORE_TIERS:
        score = scores.get(code)
        tier = CORE_TIERS[code]
        base_weight = tier_weight(tier)
        effective_weight = base_weight
        if code == "C2":
            effective_weight *= crossing_weight_multiplier(num_nodes)
        facets[code] = RulerV3Facet(
            code=code,
            name=CORE_NAMES[code],
            tier=tier,
            score=score,
            base_weight=base_weight,
            effective_weight=effective_weight,
            applicable=score is not None,
            applicability_reason="core_facet_input_applicable",
            metadata=metadata.get(code, {}),
        )
    return facets


def _triple_view_scores(facets: Mapping[str, RulerV3Facet]) -> Dict[str, float]:
    """Compute equal, tiered, and Tier-1-only composite views.

    Parameters
    ----------
    facets : Mapping[str, RulerV3Facet]
        Facet publication records keyed by V3 facet code.

    Returns
    -------
    Dict[str, float]
        Composite views on a 0--100 scale.
    """
    values = {code: facet.score for code, facet in facets.items()}
    tiered_weights = {code: facet.effective_weight for code, facet in facets.items()}
    equal_weights = {code: 1.0 for code in facets}
    tier1_weights = {
        code: facet.effective_weight for code, facet in facets.items() if facet.tier == 1
    }
    return {
        "tiered": renormalized_score(values, tiered_weights),
        "equal": renormalized_score(values, equal_weights),
        "tier1_only": renormalized_score(values, tier1_weights),
    }


def _row_flags(
    *,
    degenerate_scale: bool,
    occlusion_score: Optional[float],
    whitespace_ratio: float,
) -> Tuple[str, ...]:
    """Return hard row flags for floor violations.

    Parameters
    ----------
    degenerate_scale : bool
        Whether the degenerate-scale guard fired.
    occlusion_score : Optional[float]
        C4 node-occlusion score.
    whitespace_ratio : float
        C5 ``visual_area / area_floor`` ratio.

    Returns
    -------
    Tuple[str, ...]
        Frozen row flag names.
    """
    flags: List[str] = []
    if degenerate_scale:
        flags.append("DEGENERATE_SCALE")
    if occlusion_score is not None and occlusion_score < OCCLUSION_FLOOR_THRESHOLD:
        flags.append("OCCLUSION_FLOOR")
    if whitespace_ratio > WHITESPACE_RATIO_HI * SPRAWL_RATIO_FACTOR:
        flags.append("SPRAWL")
    return tuple(flags)


__all__ = [
    "CORE_NAMES",
    "CORE_TIERS",
    "OCCLUSION_FLOOR_THRESHOLD",
    "PURE_GEOMETRY_FACETS",
    "RulerV3Facet",
    "RulerV3Result",
    "SPRAWL_RATIO_FACTOR",
    "WHITESPACE_RATIO_HI",
    "WHITESPACE_RATIO_LO",
    "composite_v3",
    "crossing_angle_90_score",
    "crossing_weight_multiplier",
    "multi_radius_neighborhood_preservation",
    "renormalized_score",
    "score_core_v3",
    "tier_weight",
    "whitespace_sprawl_score",
]
