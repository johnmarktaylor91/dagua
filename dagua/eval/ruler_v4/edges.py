"""Crossing, routing, angular, ambiguity, and edge-label facet family."""

# Contract identity docstrings intentionally keep each title and digest on one line.
# ruff: noqa: E501

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import DefaultDict, List

import torch

from dagua.eval.ruler_v4._util import (
    bounded,
    mean_result,
    proper_intersection,
    route_segments,
    smoothstep,
)
from dagua.eval.ruler_v4.scene import BoxGeometry, FacetResult, Scene, na_result, value_result


def _segment_angle(first: torch.Tensor, second: torch.Tensor) -> float:
    """Return the acute unoriented angle between two vectors.

    Parameters
    ----------
    first, second : torch.Tensor
        Two-dimensional vectors.

    Returns
    -------
    float
        Angle in radians in ``[0, pi/2]``.
    """

    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    if float(denominator) == 0.0:
        return 0.0
    cosine = min(1.0, max(-1.0, abs(float(torch.dot(first, second) / denominator))))
    return math.acos(cosine)


def _crossing_severities(scene: Scene) -> List[float]:
    """Compute smooth crossing event severities.

    Parameters
    ----------
    scene : Scene
        Validated route scene.

    Returns
    -------
    list[float]
        One normalized severity per proper nonincident crossing.
    """

    segments = route_segments(scene)
    values = []
    for index, (route_a, _, start_a, end_a) in enumerate(segments):
        edge_a = scene.graph.edges[scene.routes[route_a].edge_index]
        for route_b, _, start_b, end_b in segments[index + 1 :]:
            if route_a == route_b:
                continue
            edge_b = scene.graph.edges[scene.routes[route_b].edge_index]
            if set(edge_a) & set(edge_b):
                continue
            if proper_intersection(start_a, end_a, start_b, end_b):
                angle = _segment_angle(end_a - start_a, end_b - start_b)
                values.append(1.0 - math.sin(angle))
    return values


def U07(scene: Scene) -> FacetResult:
    """NORMATIVE CONTRACT: Crossing slot. Frozen SHA-256: 63526aa6824dfa5180234ba97086725621e8b9e8aa3c40713c85a7c4a86caf2b."""

    if scene.edge_count < 2 or len(scene.routes) < 2:
        return mean_result(
            "U07",
            {"U7.base": 0.0, "U7.tail": 0.0},
            {"crossing_count": 0, "eligible_pairs": 0},
        )
    severities = _crossing_severities(scene)
    eligible = 0
    for left in range(scene.edge_count):
        for right in range(left + 1, scene.edge_count):
            if not set(scene.graph.edges[left]) & set(scene.graph.edges[right]):
                eligible += 1
    if eligible == 0:
        return mean_result(
            "U07",
            {"U7.base": 0.0, "U7.tail": 0.0},
            {"crossing_count": 0, "eligible_pairs": 0},
        )
    base_raw = sum(1.0 + severity for severity in severities) / eligible
    base = bounded(base_raw)
    if severities:
        ordered = sorted(severities, reverse=True)
        tail_count = max(1, math.ceil(0.10 * len(ordered)))
        tail = bounded(sum(ordered[:tail_count]) / tail_count)
    else:
        tail = 0.0
    return mean_result(
        "U07",
        {"U7.base": base, "U7.tail": tail},
        {"crossing_count": len(severities), "eligible_pairs": eligible},
    )


def U08(scene: Scene) -> FacetResult:
    """Angular resolution at nodes. Frozen SHA-256: bcc8037fd0f620ce8e43fa7ba68acd711601e12355dd48a2741469dc997a4ebf."""

    directions: DefaultDict[int, List[torch.Tensor]] = defaultdict(list)
    for source, target in scene.graph.edges:
        delta = scene.positions[target] - scene.positions[source]
        if float(torch.linalg.vector_norm(delta)) == 0.0:
            continue
        directions[source].append(delta)
        directions[target].append(-delta)
    defects = []
    for vectors in directions.values():
        if len(vectors) < 2:
            continue
        angles = sorted(
            math.atan2(float(vector[1]), float(vector[0])) % (2.0 * math.pi) for vector in vectors
        )
        gaps = [
            (angles[(index + 1) % len(angles)] - angles[index]) % (2.0 * math.pi)
            for index in range(len(angles))
        ]
        minimum = min(gaps)
        ideal = 2.0 * math.pi / len(vectors)
        defect = max(0.0, 1.0 - minimum / ideal)
        defects.append(0.0 if defect < 1e-14 else defect)
    if not defects:
        return na_result("no_degree_two_nodes")
    defect = sum(defects) / len(defects)
    return value_result(defect, {"U08.headline": defect}, {"node_count": len(defects)})


def _point_box_signed(point: torch.Tensor, box: BoxGeometry) -> float:
    """Return signed point clearance from an axis-aligned box.

    Parameters
    ----------
    point : torch.Tensor
        Point with shape ``[2]``.
    box : BoxGeometry
        Derived obstacle box.

    Returns
    -------
    float
        Positive outside clearance and negative penetration depth.
    """

    excess = torch.abs(point - box.center) - box.half_extents
    outside = float(torch.linalg.vector_norm(torch.clamp(excess, min=0.0)))
    inside = min(max(float(excess[0]), float(excess[1])), 0.0)
    return outside + inside


def U10(scene: Scene) -> FacetResult:
    """Edge-node occlusion / false attachment. Frozen SHA-256: 91f7929473c6d2f9fb9afc0b0fded75bdc899fe521ca942794b6042739aaf850."""

    if not scene.routes:
        return na_result("no_routes")
    burdens = []
    penetration_count = 0
    for route in scene.routes:
        incident = set(scene.graph.edges[route.edge_index])
        samples = torch.cat(((route.points[:-1] + route.points[1:]) / 2.0, route.points), dim=0)
        for box in scene.node_boxes:
            if box.owner in incident:
                continue
            signed = min(_point_box_signed(sample, box) for sample in samples)
            if signed < 0.0:
                penetration_count += 1
            burdens.append(bounded(max(0.0, 0.5 - signed / scene.intrinsic_unit) ** 2))
    if not burdens:
        return na_result("no_nonincident_route_obstacles")
    defect = sum(burdens) / len(burdens)
    return value_result(
        defect,
        {"U10.headline": defect},
        {"pair_count": len(burdens), "penetration_count": penetration_count},
    )


def _route_bends(points: torch.Tensor) -> List[float]:
    """Return normalized bend burdens for one route.

    Parameters
    ----------
    points : torch.Tensor
        Route vertices ``[P, 2]``.

    Returns
    -------
    list[float]
        Zero for straight continuation, one for reversal.
    """

    result = []
    for index in range(1, points.shape[0] - 1):
        incoming = points[index] - points[index - 1]
        outgoing = points[index + 1] - points[index]
        denominator = torch.linalg.vector_norm(incoming) * torch.linalg.vector_norm(outgoing)
        if float(denominator) == 0.0:
            result.append(1.0)
            continue
        cosine = min(1.0, max(-1.0, float(torch.dot(incoming, outgoing) / denominator)))
        result.append(math.acos(cosine) / math.pi)
    return result


def U11(scene: Scene) -> FacetResult:
    """Routed-edge quality: NORMATIVE CONTRACT (A1). Frozen SHA-256: 411dab9b787a8465d2f2591f3048bb6d11d1e57075071532d37c647e74e71dfd."""

    if not scene.routes:
        return na_result("no_routes")
    self_crossings = 0
    bend_values = []
    tortuosities = []
    endpoint_skews = []
    segment_imbalance = []
    for route in scene.routes:
        points = route.points
        for left in range(points.shape[0] - 1):
            for right in range(left + 2, points.shape[0] - 1):
                if proper_intersection(
                    points[left], points[left + 1], points[right], points[right + 1]
                ):
                    self_crossings += 1
        bends = _route_bends(points)
        bend_values.extend(bends)
        segment_lengths = torch.linalg.vector_norm(points[1:] - points[:-1], dim=1)
        chord = float(torch.linalg.vector_norm(points[-1] - points[0]))
        arc = float(torch.sum(segment_lengths))
        tortuosities.append(bounded(max(0.0, arc / max(chord, 1e-12) - 1.0)))
        source, target = scene.graph.edges[route.edge_index]
        source_direction = points[1] - points[0]
        target_direction = points[-1] - points[-2]
        baseline = scene.positions[target] - scene.positions[source]
        endpoint_skews.append(
            (
                _segment_angle(source_direction, baseline)
                + _segment_angle(target_direction, baseline)
            )
            / math.pi
        )
        if segment_lengths.numel() > 1:
            segment_imbalance.append(
                bounded(
                    float(
                        torch.std(segment_lengths)
                        / torch.clamp(torch.mean(segment_lengths), min=1e-12)
                    )
                )
            )
        else:
            segment_imbalance.append(0.0)
    values = {
        "U11.i": bounded(self_crossings / len(scene.routes)),
        "U11.ii": sum(bend_values) / len(bend_values) if bend_values else 0.0,
        "U11.iii": sum(tortuosities) / len(tortuosities),
        "U11.iv": sum(endpoint_skews) / len(endpoint_skews),
        "U11.v": sum(segment_imbalance) / len(segment_imbalance),
    }
    return mean_result("U11", values, {"self_intersection_count": self_crossings})


def U12(scene: Scene) -> FacetResult:
    """Path/edge continuity. Frozen SHA-256: 0170de3c481a98b8d063ff505c039de969221b1e43c3bef596a64c7e0c083d47."""

    if not scene.routes:
        return na_result("no_routes")
    bends = [value for route in scene.routes for value in _route_bends(route.points)]
    if not bends:
        defect = 0.0
    else:
        defect = sum(bends) / len(bends)
    return value_result(defect, {"U12.headline": defect}, {"bend_count": len(bends)})


def U13(scene: Scene) -> FacetResult:
    """Near-parallel edge ambiguity / bundle confusion. Frozen SHA-256: 359205189c5fa413aae039d4033602af0a665c9e93e583c274b57d1341d1ca43."""

    segments = route_segments(scene)
    parallel = []
    close = []
    for index, (route_a, _, start_a, end_a) in enumerate(segments):
        for route_b, _, start_b, end_b in segments[index + 1 :]:
            if route_a == route_b:
                continue
            angle = _segment_angle(end_a - start_a, end_b - start_b)
            midpoint_distance = float(
                torch.linalg.vector_norm((start_a + end_a - start_b - end_b) / 2.0)
            )
            parallel.append(max(0.0, 1.0 - angle / (math.pi / 12.0)))
            normalized_distance = midpoint_distance / scene.intrinsic_unit
            close.append(max(0.0, 1.0 - normalized_distance / 3.5) ** 2)
    if not parallel:
        return na_result("too_few_route_segments")
    values = {
        "U13.i": sum(left * right for left, right in zip(parallel, close)) / len(parallel),
        "U13.ii": sum(close) / len(close),
    }
    return mean_result("U13", values)


def U15(scene: Scene) -> FacetResult:
    """Multi-edge / self-loop legibility. Frozen SHA-256: ba372b7b0425c2a202e8d76de7a16b88f9bb18ebb2a6220df0ecfc4089ad83c7."""

    multiplicities = Counter(
        tuple(sorted(edge)) for edge in scene.graph.edges if edge[0] != edge[1]
    )
    parallel_groups = [edge for edge, count in multiplicities.items() if count > 1]
    loops = [index for index, edge in enumerate(scene.graph.edges) if edge[0] == edge[1]]
    if not parallel_groups and not loops:
        return na_result("no_multiedges_or_self_loops")
    route_by_edge = {route.edge_index: route for route in scene.routes}
    pair_defects = []
    for edge in parallel_groups:
        indices = [
            index
            for index, candidate in enumerate(scene.graph.edges)
            if tuple(sorted(candidate)) == edge
        ]
        for left_position, left_index in enumerate(indices):
            for right_index in indices[left_position + 1 :]:
                left = route_by_edge.get(left_index)
                right = route_by_edge.get(right_index)
                if left is None or right is None:
                    continue
                left_length = float(
                    torch.sum(torch.linalg.vector_norm(left.points[1:] - left.points[:-1], dim=1))
                )
                right_length = float(
                    torch.sum(torch.linalg.vector_norm(right.points[1:] - right.points[:-1], dim=1))
                )
                maximum_length = max(left_length, right_length)
                shared_length = min(left_length, right_length)
                if maximum_length == 0.0:
                    pair_defects.append(0.0)
                    continue
                left_samples = _sample_polyline(left.points, 33)[4:-4]
                right_samples = _sample_polyline(right.points, 33)[4:-4]
                separation = float(torch.min(torch.cdist(left_samples, right_samples)))
                relative = separation / shared_length if shared_length > 0.0 else 0.0
                defect = max(0.0, 1.0 - relative / 0.05) ** 2
                fade_shared = float(
                    smoothstep(torch.tensor(shared_length / (0.1 * maximum_length)))
                )
                fade_absolute = float(
                    smoothstep(torch.tensor(maximum_length / (0.1 * scene.intrinsic_unit)))
                )
                pair_defects.append(fade_shared * fade_absolute * defect)
    multi = sum(pair_defects) / len(pair_defects) if pair_defects else 0.0
    loop_defects = []
    for loop_index in loops:
        route = route_by_edge.get(loop_index)
        if route is None:
            continue
        owner = scene.graph.edges[loop_index][0]
        box = scene.node_boxes[owner]
        signed = (
            min(_point_box_signed(point, box) for point in route.points[1:-1])
            if route.points.shape[0] > 2
            else 0.0
        )
        loop_defects.append(bounded(max(0.0, -signed) / scene.intrinsic_unit))
    loop = sum(loop_defects) / len(loop_defects) if loop_defects else 0.0
    return mean_result(
        "U15",
        {"U15.i": multi, "U15.ii": loop},
        {"parallel_pair_count": len(pair_defects), "loop_count": len(loop_defects)},
    )


def _sample_polyline(points: torch.Tensor, count: int) -> torch.Tensor:
    """Sample a polyline at equally spaced arc-length fractions.

    Parameters
    ----------
    points : torch.Tensor
        Polyline vertices with shape ``[P, 2]``.
    count : int
        Number of samples, at least two.

    Returns
    -------
    torch.Tensor
        Sample points with shape ``[count, 2]``.
    """

    lengths = torch.linalg.vector_norm(points[1:] - points[:-1], dim=1)
    cumulative = torch.cat((torch.zeros(1, dtype=torch.float64), torch.cumsum(lengths, dim=0)))
    total = cumulative[-1]
    if float(total) == 0.0:
        return points[0].repeat(count, 1)
    targets = torch.linspace(0.0, float(total), count, dtype=torch.float64)
    segment = torch.searchsorted(cumulative[1:], targets, right=False)
    segment = torch.clamp(segment, max=points.shape[0] - 2)
    local = (targets - cumulative[segment]) / torch.clamp(lengths[segment], min=1e-12)
    return points[segment] + local[:, None] * (points[segment + 1] - points[segment])


def U16(scene: Scene) -> FacetResult:
    """Edge-label placement. Frozen SHA-256: 10e732d51189a4e9bf607044ce70d96042a3f3474fa172917467b043e49145a7."""

    if not scene.edge_label_boxes:
        return na_result("no_declared_edge_labels")
    overlaps = []
    ownership = []
    route_by_edge = {route.edge_index: route for route in scene.routes}
    for index, label in enumerate(scene.edge_label_boxes):
        for other in scene.edge_label_boxes[index + 1 :]:
            delta = torch.abs(label.center - other.center) - (
                label.half_extents + other.half_extents
            )
            overlaps.append(
                float(torch.prod(torch.clamp(-delta, min=0.0)))
                / max(1e-12, float(4.0 * torch.prod(label.half_extents)))
            )
        route = route_by_edge.get(label.owner)
        if route is not None:
            sample_distances = torch.linalg.vector_norm(route.points - label.center, dim=1)
            ownership.append(bounded(float(torch.min(sample_distances)) / scene.intrinsic_unit))
    values = {
        "U16.i": min(1.0, sum(overlaps) / len(overlaps)) if overlaps else 0.0,
        "U16.ii": sum(ownership) / len(ownership) if ownership else 1.0,
    }
    return mean_result("U16", values, {"label_count": len(scene.edge_label_boxes)})
