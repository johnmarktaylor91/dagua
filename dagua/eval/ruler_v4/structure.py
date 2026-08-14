"""Structure, neighborhood, scale-neutral shape, and frame facet family."""

# Contract identity docstrings intentionally keep each title and digest on one line.
# ruff: noqa: E501

from __future__ import annotations

import math
from typing import Dict, List, Set

import torch

from dagua.eval.ruler_v4._util import (
    adjacency,
    bounded,
    correlation_defect,
    graph_distances,
    isotonic_stress,
    mean_result,
    midranks,
    node_degrees,
    pair_values,
    soft_pos,
)
from dagua.eval.ruler_v4.frames import robust_frame
from dagua.eval.ruler_v4.scene import FacetResult, Scene, na_result, value_result


def U01(scene: Scene) -> FacetResult:
    """Distance/stress fidelity. Frozen SHA-256: 0db6143734cebf7f28d5a3f231cc7accb4010b8fed86c78ab470927e93651641."""

    graph_order, layout = pair_values(scene, graph_distances(scene))
    if graph_order.numel() < 3 or torch.unique(graph_order).numel() < 2:
        return na_result("too_few_distance_pairs")
    defect = isotonic_stress(graph_order, layout)
    return value_result(defect, {"U01.headline": defect}, {"pair_count": graph_order.numel()})


def U01b(scene: Scene) -> FacetResult:
    """Distance strata/bands. Frozen SHA-256: 30b3e2ee99109b898bf5d301e7d0b1b1fd96065cccd875ecf423757683399e66."""

    graph_order, layout = pair_values(scene, graph_distances(scene))
    if graph_order.numel() < 3 or torch.unique(graph_order).numel() < 2:
        return na_result("too_few_distance_pairs")
    diameter = float(torch.max(graph_order))
    local_mask = graph_order <= min(2.0, diameter)
    long_mask = graph_order > max(2.0, diameter / 2.0)
    if not bool(local_mask.any()) or not bool(long_mask.any()):
        return na_result("underpopulated_distance_bands")
    values = {
        "U01b.local": isotonic_stress(graph_order[local_mask], layout[local_mask]),
        "U01b.long": isotonic_stress(graph_order[long_mask], layout[long_mask]),
    }
    return mean_result(
        "U01b", values, {"local_pairs": int(local_mask.sum()), "long_pairs": int(long_mask.sum())}
    )


def U02(scene: Scene) -> FacetResult:
    """Shepard rank fidelity. Frozen SHA-256: d4a6ff0c6a7699832bfe925d38feee236309ca858a371b36ba4fb5169e294235."""

    graph_order, layout = pair_values(scene, graph_distances(scene))
    if graph_order.numel() < 6:
        return na_result("too_few_distance_pairs")
    defect = correlation_defect(midranks(graph_order), midranks(layout))
    return value_result(defect, {"U02.headline": defect}, {"pair_count": graph_order.numel()})


def _radius_neighbors(distances: torch.Tensor, node: int, radius: int) -> Set[int]:
    """Return graph neighbors within a closed hop radius.

    Parameters
    ----------
    distances : torch.Tensor
        Graph distance matrix ``[N, N]``.
    node : int
        Center node.
    radius : int
        Positive hop radius.

    Returns
    -------
    set[int]
        Neighbor indices excluding the center.
    """

    return {
        index
        for index, distance in enumerate(distances[node].tolist())
        if index != node and math.isfinite(distance) and distance <= radius
    }


def U03(scene: Scene) -> FacetResult:
    """Neighborhood preservation, multi-radius, SOFT. Frozen SHA-256: 48c17bf5152255db40a9e35db6ad09fedb7aca9c57c6bf33d7901c43288d7741."""

    if scene.node_count < 4:
        return na_result("too_few_nodes")
    graph_order = graph_distances(scene)
    layout_order = torch.cdist(scene.positions, scene.positions)
    values: Dict[str, float] = {}
    saturated = True
    for radius in (1, 2, 4):
        center_defects: List[float] = []
        for node in range(scene.node_count):
            expected = _radius_neighbors(graph_order, node, radius)
            if not expected or len(expected) == scene.node_count - 1:
                continue
            saturated = False
            nearest = torch.argsort(layout_order[node]).tolist()[1 : len(expected) + 1]
            observed = set(nearest)
            center_defects.append(1.0 - len(expected & observed) / len(expected | observed))
        values[f"U03.r_{radius}"] = (
            sum(center_defects) / len(center_defects) if center_defects else 0.0
        )
    if saturated:
        return na_result("neighborhoods_saturated")
    return mean_result("U03", values)


def _density_defect(scene: Scene, bandwidth: float) -> float:
    """Compare local graph mass and primitive occupancy kernels.

    Parameters
    ----------
    scene : Scene
        Validated scene.
    bandwidth : float
        Gaussian bandwidth in scene coordinates.

    Returns
    -------
    float
        Normalized Jensen-Shannon divergence.
    """

    distances = torch.cdist(scene.positions, scene.positions)
    kernel = torch.exp(-(distances**2) / (2.0 * bandwidth**2))
    demand_mass = 1.0 + node_degrees(scene) / 2.0
    areas = torch.tensor(
        [float(4.0 * torch.prod(box.half_extents)) for box in scene.node_boxes], dtype=torch.float64
    )
    demand = kernel @ demand_mass
    occupancy = kernel @ areas
    demand = demand / torch.sum(demand)
    occupancy = occupancy / torch.sum(occupancy)
    mixture = (demand + occupancy) / 2.0
    left = torch.sum(torch.where(demand > 0.0, demand * torch.log(demand / mixture), 0.0))
    right = torch.sum(torch.where(occupancy > 0.0, occupancy * torch.log(occupancy / mixture), 0.0))
    return min(1.0, float((left + right) / (2.0 * math.log(2.0))))


def U04a(scene: Scene) -> FacetResult:
    """Density-map fidelity (structure). Frozen SHA-256: 59a7eeda57c5735b2ced3d050af16b57181e127735012a0deb20ac7fff2d08db."""

    if scene.node_count < 10:
        return na_result("too_few_nodes")
    values = {
        "U04a.2u": _density_defect(scene, 2.0 * scene.intrinsic_unit),
        "U04a.8u": _density_defect(scene, 8.0 * scene.intrinsic_unit),
    }
    return mean_result("U04a", values)


def U04b(scene: Scene) -> FacetResult:
    """Crowding / whitespace legibility. Frozen SHA-256: 2005ce3d0597b90fefc5f0cc5197f44b64b2c24911eea27e858afe2f2e3c3dd4."""

    if scene.node_count < 2:
        return na_result("too_few_nodes")
    distances = torch.cdist(scene.positions, scene.positions)
    distances.fill_diagonal_(float("inf"))
    nearest = torch.min(distances, dim=1).values / scene.intrinsic_unit
    crowding = float(torch.mean(torch.exp(-nearest)).item())
    frame = robust_frame(scene.positions, scene.intrinsic_unit)
    primitive_area = sum(float(4.0 * torch.prod(box.half_extents)) for box in scene.node_boxes)
    whitespace = bounded(max(0.0, frame.area / max(primitive_area, 1e-12) - 10.0) / 10.0)
    return mean_result("U04b", {"U04b.part_1": crowding, "U04b.part_2": whitespace})


def U05(scene: Scene) -> FacetResult:
    """Graph-geometric shape fidelity. Frozen SHA-256: 47a4b416c09f564831aaf1d9fb66a80bef138bce6bb8daf9a02c3d3b1fa0cfa3."""

    graph_order, layout = pair_values(scene, torch.sqrt(graph_distances(scene)))
    if graph_order.numel() < 3:
        return na_result("too_few_resistance_pairs")
    defect = isotonic_stress(graph_order, layout)
    return value_result(defect, {"U05.headline": defect})


def U06(scene: Scene) -> FacetResult:
    """Symmetry display. Frozen SHA-256: 9f6f3aa20625f493f3193bb2c8e377b79d2801257ef494d71cd433421497dabe."""

    if scene.node_count < 4:
        return na_result("no_nontrivial_automorphism")
    frame = robust_frame(scene.positions, scene.intrinsic_unit)
    centered = scene.positions - frame.center
    reflected = centered.clone()
    reflected[:, 0] *= -1.0
    distances = torch.cdist(reflected, centered)
    nearest = torch.min(distances, dim=1).values / scene.intrinsic_unit
    defect = bounded(float(torch.mean(nearest).item()))
    return value_result(defect, {"U06.headline": defect})


def U09(scene: Scene) -> FacetResult:
    """Edge-length coherence, stratified/conditional. Frozen SHA-256: deda54396b4f2de53842b94593bb0d0cc1fdd2084cc0874cec819c0de15abad7."""

    if scene.edge_count < 2:
        return na_result("too_few_edges")
    edges = torch.tensor(scene.graph.edges, dtype=torch.long)
    lengths = torch.linalg.vector_norm(
        scene.positions[edges[:, 0]] - scene.positions[edges[:, 1]], dim=1
    )
    if scene.graph.edge_weights is not None and scene.graph.weight_semantics == "distance_cost":
        targets = torch.tensor(scene.graph.edge_weights, dtype=torch.float64)
        ratios = lengths / targets
    else:
        ratios = lengths
    median = torch.median(ratios)
    mad = torch.median(torch.abs(ratios - median))
    defect = bounded(float(mad / torch.clamp(median, min=1e-12)))
    return value_result(
        defect, {"U09.headline": defect}, {"median": float(median), "mad": float(mad)}
    )


def U14(scene: Scene) -> FacetResult:
    """False adjacency (node-pair proximity ambiguity). Frozen SHA-256: 20a4536ac81681828042a6d06ad76113d0b5c08305e08887799956799a339d65."""

    neighbors = adjacency(scene)
    burdens = []
    for left in range(scene.node_count):
        for right in range(left + 1, scene.node_count):
            if right in neighbors[left]:
                continue
            distance = float(
                torch.linalg.vector_norm(scene.positions[left] - scene.positions[right])
            )
            burdens.append(math.exp(-distance / scene.intrinsic_unit))
    if not burdens:
        return na_result("no_nonadjacent_pairs")
    defect = sum(burdens) / len(burdens)
    return value_result(defect, {"U14.headline": defect}, {"pair_count": len(burdens)})


def U22(scene: Scene) -> FacetResult:
    """Aspect ratio / shape. Frozen SHA-256: d141e6fef540d3115fe342b275133278b550ee5d064663779b19c58b61b6304e."""

    if scene.node_count < 2:
        return na_result("too_few_nodes")
    frame = robust_frame(scene.positions, scene.intrinsic_unit)
    ratio = float(torch.max(frame.half_extents) / torch.min(frame.half_extents))
    defect = bounded(abs(math.log(ratio)))
    return value_result(
        defect, {"U22.headline": defect}, {"aspect_ratio": ratio, "regime": frame.regime}
    )


def U23(scene: Scene) -> FacetResult:
    """Visual balance. Frozen SHA-256: c14a86d9e00099c5db2c6d571e1af057896146222e96737678253a76ffdd5972."""

    frame = robust_frame(scene.positions, scene.intrinsic_unit)
    degrees = 1.0 + node_degrees(scene)
    centroid = torch.sum(scene.positions * degrees[:, None], dim=0) / torch.sum(degrees)
    normalized = torch.linalg.vector_norm((centroid - frame.center) / frame.half_extents)
    defect = bounded(float(normalized))
    return value_result(defect, {"U23.headline": defect}, {"centroid_offset": float(normalized)})


def U24(scene: Scene) -> FacetResult:
    """Total ink economy. Frozen SHA-256: 5857b951b89bbebff02131437db2b8070d84dc03460746adc7f7e65ac1d03573."""

    frame = robust_frame(scene.positions, scene.intrinsic_unit)
    node_ink = sum(float(4.0 * torch.prod(box.half_extents)) for box in scene.node_boxes)
    route_ink = 0.0
    for route in scene.routes:
        route_ink += float(
            torch.sum(torch.linalg.vector_norm(route.points[1:] - route.points[:-1], dim=1))
        )
    route_ink *= scene.style.route_stroke_width * scene.style.coordinate_scale
    ratio = (node_ink + route_ink) / frame.area
    defect = bounded(soft_pos(ratio / 0.10 - 1.0))
    return value_result(defect, {"U24.headline": defect}, {"ink_ratio": ratio})
