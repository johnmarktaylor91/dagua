"""Declared cluster cohesion, separation, containment, hierarchy, and label facets."""

# Contract identity docstrings intentionally keep each title and digest on one line.
# ruff: noqa: E501

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

from dagua.eval.ruler_v4._util import aabb_pair, bounded, mean_result
from dagua.eval.ruler_v4.frames import RobustFrame, robust_frame
from dagua.eval.ruler_v4.scene import BoxGeometry, FacetResult, Scene, na_result, value_result


def _clusters(scene: Scene) -> Dict[str, Tuple[int, ...]]:
    """Return nonempty, canonical declared cluster memberships.

    Parameters
    ----------
    scene : Scene
        Validated clustered scene.

    Returns
    -------
    dict[str, tuple[int, ...]]
        Sorted unique member indices per cluster.
    """

    return {
        name: tuple(sorted(set(members)))
        for name, members in sorted(scene.graph.clusters.items())
        if members
    }


def _regions(scene: Scene) -> Dict[str, RobustFrame]:
    """Derive continuous robust rectangular cluster regions.

    Parameters
    ----------
    scene : Scene
        Validated clustered scene.

    Returns
    -------
    dict[str, RobustFrame]
        Region frame per declared cluster.
    """

    return {
        name: robust_frame(scene.positions[list(members)], scene.intrinsic_unit)
        for name, members in _clusters(scene).items()
    }


def _frame_box(frame: RobustFrame, owner: int = -1) -> BoxGeometry:
    """Convert a robust region frame into box geometry.

    Parameters
    ----------
    frame : RobustFrame
        Robust region.
    owner : int
        Optional synthetic owner id.

    Returns
    -------
    BoxGeometry
        Axis-aligned region box.
    """

    return BoxGeometry(frame.center, frame.half_extents, owner)


def U25(scene: Scene) -> FacetResult:
    """Cluster cohesion / compactness. Frozen SHA-256: 1abaa433699c51d9c1d2ed8c7bde4643df3d8eb46fe9133d465b434e40f15299."""

    clusters = _clusters(scene)
    if not clusters:
        return na_result("no_declared_clusters")
    defects = []
    for members in clusters.values():
        if len(members) < 2:
            continue
        distances = torch.pdist(scene.positions[list(members)]) / scene.intrinsic_unit
        defects.append(bounded(float(torch.mean(distances))))
    if not defects:
        return na_result("clusters_too_small")
    defect = sum(defects) / len(defects)
    return value_result(defect, {"U25.headline": defect}, {"cluster_count": len(defects)})


def U26(scene: Scene) -> FacetResult:
    """Cluster separation (+ community faithfulness). Frozen SHA-256: bba49ea4d943942ee7510d7e16651470a63a488a990394d5b3ec8e1b09311508."""

    clusters = _clusters(scene)
    if len(clusters) < 2:
        return na_result("too_few_declared_clusters")
    names = list(clusters)
    separation = []
    overlap = []
    cross_edge = []
    regions = _regions(scene)
    edge_set = {tuple(sorted(edge)) for edge in scene.graph.edges if edge[0] != edge[1]}
    for left_index, left_name in enumerate(names):
        left_members = set(clusters[left_name])
        for right_name in names[left_index + 1 :]:
            if (
                scene.graph.cluster_parents.get(left_name) == right_name
                or scene.graph.cluster_parents.get(right_name) == left_name
            ):
                continue
            right_members = set(clusters[right_name])
            if left_members & right_members:
                continue
            center_distance = float(
                torch.linalg.vector_norm(regions[left_name].center - regions[right_name].center)
            )
            separation.append(math.exp(-center_distance / scene.intrinsic_unit))
            _, fraction = aabb_pair(_frame_box(regions[left_name]), _frame_box(regions[right_name]))
            overlap.append(fraction)
            possible = max(1, len(left_members) * len(right_members))
            actual = sum(
                tuple(sorted((left, right))) in edge_set
                for left in left_members
                for right in right_members
            )
            cross_edge.append(actual / possible)
    if not separation:
        return na_result("no_disjoint_cluster_pairs")
    values = {
        "U26.i": sum(separation) / len(separation),
        "U26.ii": sum(overlap) / len(overlap),
        "U26.iii": sum(cross_edge) / len(cross_edge),
    }
    return mean_result("U26", values, {"cluster_pair_count": len(separation)})


def U27(scene: Scene) -> FacetResult:
    """Containment / non-intrusion. Frozen SHA-256: fee484e87c819fbe1457b7519fb0586ef1ca5f27f8fed0e7eeaa5fc7ba997f5c."""

    clusters = _clusters(scene)
    if not clusters:
        return na_result("no_declared_clusters")
    regions = _regions(scene)
    node_intrusions = []
    edge_intrusions = []
    boundary_debt = []
    for name, members in clusters.items():
        member_set = set(members)
        region = _frame_box(regions[name])
        for node in scene.node_boxes:
            if node.owner in member_set:
                continue
            _, overlap = aabb_pair(region, node)
            node_intrusions.append(overlap)
        for route in scene.routes:
            source, target = scene.graph.edges[route.edge_index]
            if source in member_set and target in member_set:
                continue
            local = torch.abs(route.points - region.center) - region.half_extents
            inside = torch.all(local <= 0.0, dim=1)
            edge_intrusions.append(float(torch.mean(inside.to(torch.float64))))
        member_points = scene.positions[list(members)]
        normalized = torch.abs(member_points - region.center) / region.half_extents
        boundary_debt.append(
            float(torch.mean(torch.clamp(torch.max(normalized, dim=1).values - 0.8, min=0.0)))
        )
    values = {
        "U27.i": sum(node_intrusions) / len(node_intrusions) if node_intrusions else 0.0,
        "U27.ii": sum(edge_intrusions) / len(edge_intrusions) if edge_intrusions else 0.0,
        "U27.iii": min(1.0, sum(boundary_debt) / len(boundary_debt)),
    }
    return mean_result("U27", values)


def _containment_debt(inner: RobustFrame, outer: RobustFrame) -> float:
    """Return normalized region containment overflow.

    Parameters
    ----------
    inner, outer : RobustFrame
        Child and parent regions.

    Returns
    -------
    float
        Maximum normalized excess beyond the parent frame.
    """

    excess = torch.abs(inner.center - outer.center) + inner.half_extents - outer.half_extents
    normalized = torch.clamp(excess / outer.half_extents, min=0.0)
    return min(1.0, float(torch.max(normalized)))


def U28(scene: Scene) -> FacetResult:
    """Hierarchy nesting fidelity. Frozen SHA-256: 5d48376583595a1fad3dee50676d440eb2f0d5bf9bdd6acb03b93535a421e649."""

    if not scene.graph.cluster_parents:
        return na_result("no_declared_cluster_hierarchy")
    regions = _regions(scene)
    containment = []
    sibling_overlap = []
    ordering = []
    children_by_parent: Dict[str, list[str]] = {}
    for child, parent in scene.graph.cluster_parents.items():
        if child not in regions or parent not in regions:
            continue
        containment.append(_containment_debt(regions[child], regions[parent]))
        area_child = 4.0 * float(torch.prod(regions[child].half_extents))
        area_parent = 4.0 * float(torch.prod(regions[parent].half_extents))
        ordering.append(max(0.0, 1.0 - area_parent / max(area_child, 1e-12)))
        children_by_parent.setdefault(parent, []).append(child)
    for children in children_by_parent.values():
        for index, left in enumerate(children):
            for right in children[index + 1 :]:
                _, overlap = aabb_pair(_frame_box(regions[left]), _frame_box(regions[right]))
                sibling_overlap.append(overlap)
    if not containment:
        return na_result("invalid_cluster_hierarchy")
    values = {
        "U28.i": sum(containment) / len(containment),
        "U28.ii": sum(sibling_overlap) / len(sibling_overlap) if sibling_overlap else 0.0,
        "U28.iii": sum(ordering) / len(ordering),
    }
    return mean_result("U28", values)


def U29(scene: Scene) -> FacetResult:
    """Cluster shape coherence. Frozen SHA-256: 658cf935c25388c5959b5c4af9880840c59b28fe1b6284d0dfe1dfa0d9e7a819."""

    regions = _regions(scene)
    if not regions:
        return na_result("no_declared_clusters")
    defects = []
    for region in regions.values():
        ratio = float(torch.max(region.half_extents) / torch.min(region.half_extents))
        defects.append(bounded(abs(math.log(ratio))))
    defect = sum(defects) / len(defects)
    return value_result(defect, {"U29.headline": defect}, {"cluster_count": len(defects)})


def U30(scene: Scene) -> FacetResult:
    """Cluster labels. Frozen SHA-256: 64ce190db377b294922a09b4b0e045224c02d611ff451b33f3fb5d07d1ca69c4."""

    regions = _regions(scene)
    if not regions or "cluster_labels" not in scene.profile.visible_channels:
        return na_result("no_declared_cluster_labels")
    placement = []
    primitive_overlap = []
    region_ambiguity = []
    for index, (name, region) in enumerate(regions.items()):
        width = max(1, len(name)) * scene.style.average_character_width * scene.style.font_size
        height = scene.style.line_height * scene.style.font_size
        center = region.center + torch.tensor([0.0, region.half_extents[1] + height / 2.0])
        label = BoxGeometry(center, torch.tensor([width / 2.0, height / 2.0]), index)
        placement.append(
            bounded(float(torch.linalg.vector_norm(center - region.center)) / scene.intrinsic_unit)
        )
        overlaps = [aabb_pair(label, node)[1] for node in scene.node_boxes]
        primitive_overlap.append(max(overlaps, default=0.0))
        foreign = [
            float(torch.linalg.vector_norm(center - other.center)) / scene.intrinsic_unit
            for other_name, other in regions.items()
            if other_name != name
        ]
        region_ambiguity.append(math.exp(-min(foreign)) if foreign else 0.0)
    values = {
        "U30.i": sum(placement) / len(placement),
        "U30.ii": sum(primitive_overlap) / len(primitive_overlap),
        "U30.iii": sum(region_ambiguity) / len(region_ambiguity),
    }
    return mean_result("U30", values, {"label_count": len(regions)})
