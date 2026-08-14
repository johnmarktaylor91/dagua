"""Primitive clearance, label, resolution, scale, and frame-economy facets."""

# Contract identity docstrings intentionally keep each title and digest on one line.
# ruff: noqa: E501

from __future__ import annotations

import math
from typing import List

import torch

from dagua.eval.ruler_v4._util import aabb_pair, bounded, mean_result, soft_pos
from dagua.eval.ruler_v4.frames import overflow_defect, robust_frame
from dagua.eval.ruler_v4.scene import FacetResult, Scene, na_result, value_result


def U17(scene: Scene) -> FacetResult:
    """Node-node occlusion / clearance. Frozen SHA-256: c4b70fd5614c1c66e1e96cb62e41fbe87803319f5b50f96b4bf8bbc2cc74a322."""

    if scene.node_count < 2:
        return na_result("too_few_nodes")
    defects = []
    overlap_count = 0
    for left in range(scene.node_count):
        for right in range(left + 1, scene.node_count):
            signed, overlap = aabb_pair(scene.node_boxes[left], scene.node_boxes[right])
            overlap_count += int(overlap > 0.0)
            clearance = max(0.0, 0.5 - signed / scene.intrinsic_unit)
            defects.append(min(1.0, overlap + bounded(clearance**2)))
    defect = sum(defects) / len(defects)
    return value_result(
        defect,
        {"U17.1": defect},
        {"pair_count": len(defects), "overlap_count": overlap_count},
    )


def U18(scene: Scene) -> FacetResult:
    """Label legibility (node labels). Frozen SHA-256: 68e32c4e9edfac024c8d2e651538d5e41e0781134960f75ca06b131cb40b346d."""

    labels = scene.node_label_boxes
    if not labels:
        return na_result("no_declared_node_labels")
    label_label: List[float] = []
    label_node: List[float] = []
    label_edge: List[float] = []
    for index, label in enumerate(labels):
        for other in labels[index + 1 :]:
            _, overlap = aabb_pair(label, other)
            label_label.append(overlap)
        for node in scene.node_boxes:
            if node.owner == label.owner:
                continue
            _, overlap = aabb_pair(label, node)
            label_node.append(overlap)
        for route in scene.routes:
            if (
                route.edge_index < scene.edge_count
                and label.owner in scene.graph.edges[route.edge_index]
            ):
                continue
            distances = torch.linalg.vector_norm(route.points - label.center, dim=1)
            label_edge.append(math.exp(-float(torch.min(distances)) / scene.intrinsic_unit))
    values = {
        "U18.ll": sum(label_label) / len(label_label) if label_label else 0.0,
        "U18.ln": sum(label_node) / len(label_node) if label_node else 0.0,
        "U18.le": sum(label_edge) / len(label_edge) if label_edge else 0.0,
    }
    return mean_result("U18", values, {"label_count": len(labels)})


def U19(scene: Scene) -> FacetResult:
    """Text legibility feasibility. Frozen SHA-256: b7166c90068923a4edb1bcd4dad3371ebb7dd9134d7e8f657afa9b2ae2a7e2da."""

    if scene.profile.viewport is None:
        return na_result("no_physical_viewport")
    if not scene.node_label_boxes and not scene.edge_label_boxes:
        return na_result("no_declared_labels")
    viewport_height = scene.profile.viewport[1]
    heights = [float(2.0 * box.half_extents[1]) for box in scene.node_label_boxes]
    heights.extend(float(2.0 * box.half_extents[1]) for box in scene.edge_label_boxes)
    minimum_fraction = min(heights) / viewport_height
    defect = bounded(soft_pos(0.012 - minimum_fraction, 0.002) / 0.012)
    return value_result(
        defect, {"U19.headline": defect}, {"minimum_height_fraction": minimum_fraction}
    )


def U20a(scene: Scene) -> FacetResult:
    """Resolution-limit degeneracy. Frozen SHA-256: 65e179ba53966fecc9fb77c4c219e05086bee5dac536540ab46bab785313b530."""

    if scene.node_count < 2:
        return na_result("too_few_nodes")
    pair_distances = torch.pdist(scene.positions) / scene.intrinsic_unit
    spacing = float(torch.mean(torch.exp(-pair_distances)).item())
    centered = scene.positions - torch.mean(scene.positions, dim=0)
    singular = torch.linalg.svdvals(centered)
    rank_collapse = 1.0
    if singular.numel() >= 2 and float(singular[0]) > 0.0:
        rank_collapse = max(0.0, 1.0 - float(singular[1] / singular[0]))
    if scene.graph.ranks is None:
        layer_collapse = 0.0
    else:
        ranks = torch.tensor(scene.graph.ranks, dtype=torch.long)
        same_rank = []
        for left in range(scene.node_count):
            for right in range(left + 1, scene.node_count):
                if ranks[left] == ranks[right]:
                    same_rank.append(
                        math.exp(
                            -float(
                                torch.linalg.vector_norm(
                                    scene.positions[left] - scene.positions[right]
                                )
                            )
                            / scene.intrinsic_unit
                        )
                    )
        layer_collapse = sum(same_rank) / len(same_rank) if same_rank else 0.0
    values = {"U20a.i": spacing, "U20a.ii": rank_collapse, "U20a.iii": layer_collapse}
    return mean_result("U20a", values)


def U20b(scene: Scene) -> FacetResult:
    """Scale-legibility plateau. Frozen SHA-256: 8037cbb7308319db2434687b3b149b5ee6797a53f78eda9e81727ebe39531947."""

    if scene.profile.scale_normalized:
        return na_result("scale_normalized_profile")
    if scene.edge_count == 0:
        return na_result("no_edges")
    edges = torch.tensor(scene.graph.edges, dtype=torch.long)
    lengths = torch.linalg.vector_norm(
        scene.positions[edges[:, 0]] - scene.positions[edges[:, 1]], dim=1
    )
    normalized = lengths / scene.intrinsic_unit
    short = torch.exp(-normalized)
    long = 1.0 - torch.exp(-torch.clamp(normalized - 8.0, min=0.0))
    defect = float(torch.mean(torch.maximum(short, long)).item())
    return value_result(
        defect, {"U20b.headline": defect}, {"median_edge_length_u": float(torch.median(normalized))}
    )


def U21(scene: Scene) -> FacetResult:
    """Frame economy / anti-sprawl. Frozen SHA-256: c662b51acf5fa3863ad367627168b280cdc6eb23194f7577203d9b0bd899d4de."""

    frame = robust_frame(scene.positions, scene.intrinsic_unit)
    primitive_area = sum(float(4.0 * torch.prod(box.half_extents)) for box in scene.node_boxes)
    component_count = _component_count(scene)
    area_reference = primitive_area / 0.10 * (1.0 + 0.5 * (component_count - 1))
    sparse_raw = soft_pos(math.log(frame.area / (4.0 * area_reference)))
    sparse = sparse_raw / (1.0 + sparse_raw)
    mass_out, anchor, overflow = overflow_defect(scene, frame)
    defect = 1.0 - (1.0 - sparse) * (1.0 - overflow)
    return value_result(
        defect,
        {"U21.d_sparse_n": sparse, "U21.d_overflow": overflow},
        {
            "frame_area": frame.area,
            "frame_regime": frame.regime,
            "trim_count": frame.trim_count,
            "mass_out": mass_out,
            "overflow_anchor": anchor,
            "degenerate_frame_coincident": all(frame.floor_bound),
        },
    )


def _component_count(scene: Scene) -> int:
    """Count connected components without importing another facet family.

    Parameters
    ----------
    scene : Scene
        Validated graph scene.

    Returns
    -------
    int
        Number of simple-support components.
    """

    remaining = set(range(scene.node_count))
    neighbors = [set() for _ in range(scene.node_count)]
    for source, target in scene.graph.edges:
        neighbors[source].add(target)
        neighbors[target].add(source)
    count = 0
    while remaining:
        count += 1
        frontier = [remaining.pop()]
        while frontier:
            node = frontier.pop()
            for neighbor in neighbors[node] & remaining:
                remaining.remove(neighbor)
                frontier.append(neighbor)
    return count
