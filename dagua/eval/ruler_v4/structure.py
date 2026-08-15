"""Structure, neighborhood, scale-neutral shape, and frame facet family."""

# Contract identity docstrings intentionally keep each title and digest on one line.
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Sequence, Set, Tuple

import torch

from dagua.eval.ruler_v4._util import (
    aabb_pair,
    adjacency,
    components,
    correlation_defect,
    declared_axis,
    global_blend,
    graph_distances,
    isotonic_stress,
    mean_result,
    midranks,
    node_degrees,
    primary_isotonic_fit,
    resolved_routes,
    smoothstep,
    snap_unit,
    soft_pos,
)
from dagua.eval.ruler_v4.frames import robust_frame, robust_projection
from dagua.eval.ruler_v4.scene import (
    BoxGeometry,
    FacetResult,
    Scene,
    invalid_result,
    na_result,
    value_result,
)


def U01(scene: Scene) -> FacetResult:
    """Distance/stress fidelity. Frozen SHA-256: 0db6143734cebf7f28d5a3f231cc7accb4010b8fed86c78ab470927e93651641."""

    strata = _distance_strata(scene)
    if not strata:
        return na_result("too_few_distance_pairs")
    order = torch.cat([item[2] for item in strata])
    layout = torch.cat([item[3] for item in strata])
    fitted = primary_isotonic_fit(order, layout)
    values: List[float] = []
    weights: List[float] = []
    published: Dict[str, float] = {}
    cursor = 0
    for component_index, band, local_order, local_layout in strata:
        count = local_order.numel()
        stress = _stress_from_fit(local_order, local_layout, fitted[cursor : cursor + count])
        published[f"component_{component_index}.{band}"] = stress
        values.append(stress)
        weights.append(float(count))
        cursor += count
    defect = global_blend(values, weights)
    return value_result(
        defect,
        {"U01.headline": defect},
        {"pair_count": order.numel(), "stratum_stress": published},
    )


def U01b(scene: Scene) -> FacetResult:
    """Distance strata/bands. Frozen SHA-256: 30b3e2ee99109b898bf5d301e7d0b1b1fd96065cccd875ecf423757683399e66."""

    strata = _distance_strata(scene)
    if not strata:
        return na_result("too_few_distance_pairs")
    order = torch.cat([item[2] for item in strata])
    layout = torch.cat([item[3] for item in strata])
    fitted = primary_isotonic_fit(order, layout)
    by_band: Dict[str, List[float]] = {"local": [], "long": []}
    band_weights: Dict[str, List[float]] = {"local": [], "long": []}
    band_counts = {"local": 0, "long": 0}
    component_band_counts: Dict[str, List[int]] = {"local": [], "long": []}
    cursor = 0
    for _, band, local_order, local_layout in strata:
        count = local_order.numel()
        if band in by_band:
            by_band[band].append(
                _stress_from_fit(local_order, local_layout, fitted[cursor : cursor + count])
            )
            band_weights[band].append(float(count))
            band_counts[band] += count
            component_band_counts[band].append(count)
        cursor += count
    values: Dict[str, float] = {}
    dropped = []
    for band in ("local", "long"):
        if component_band_counts[band] and max(component_band_counts[band]) >= 30:
            values[f"U01b.{band}"] = global_blend(by_band[band], band_weights[band])
        else:
            dropped.append(f"U01b.{band}")
    if not values:
        return na_result("band_underpopulated", {"band_counts": band_counts})
    return mean_result(
        "U01b",
        values,
        {
            "local_pairs": band_counts["local"],
            "long_pairs": band_counts["long"],
            "dropped_subterms": tuple(dropped),
        },
        renormalize_missing=False,
    )


def _distance_strata(
    scene: Scene, order_matrix: torch.Tensor | None = None
) -> List[Tuple[int, str, torch.Tensor, torch.Tensor]]:
    """Build exhaustive per-component diameter-quantile distance strata.

    Parameters
    ----------
    scene : Scene
        Validated graph scene.
    order_matrix : torch.Tensor or None
        Optional graph-side coordinates ``[N, N]``. Hop distance is the default.

    Returns
    -------
    list[tuple[int, str, torch.Tensor, torch.Tensor]]
        Component index, band name, order coordinates, and layout distances.
    """

    hops = graph_distances(scene)
    coordinates = hops if order_matrix is None else order_matrix
    result: List[Tuple[int, str, torch.Tensor, torch.Tensor]] = []
    for component_index, members in enumerate(components(scene)):
        if len(members) < 3:
            continue
        local_indices = torch.tensor(members, dtype=torch.long)
        rows, cols = torch.triu_indices(len(members), len(members), offset=1)
        source = local_indices[rows]
        target = local_indices[cols]
        hop_values = hops[source, target]
        order_values = coordinates[source, target]
        layout_values = torch.linalg.vector_norm(
            scene.positions[source] - scene.positions[target], dim=1
        )
        diameter = int(torch.max(hop_values).item())
        if diameter <= 2:
            result.append((component_index, "local", order_values, layout_values))
            continue
        local_limit = math.ceil(diameter / 3.0)
        long_start = math.ceil(2.0 * diameter / 3.0)
        masks = {
            "local": hop_values <= local_limit,
            "mid": (hop_values > local_limit) & (hop_values <= long_start),
            "long": hop_values > long_start,
        }
        for band, mask in masks.items():
            if bool(mask.any()):
                result.append((component_index, band, order_values[mask], layout_values[mask]))
    return result


def _stress_from_fit(order: torch.Tensor, layout: torch.Tensor, fitted: torch.Tensor) -> float:
    """Read one stratum's Kruskal stress from a shared isotonic fit.

    Parameters
    ----------
    order, layout, fitted : torch.Tensor
        Equal-length graph order, layout distance, and fitted disparity vectors.

    Returns
    -------
    float
        Stress-1 with the frozen zero-layout convention.
    """

    denominator = float(torch.sum(layout * layout).item())
    if denominator == 0.0:
        return 1.0 if torch.unique(order).numel() > 1 else 0.0
    residual = float(torch.sum((layout - fitted) ** 2).item())
    return min(1.0, math.sqrt(residual / denominator))


def U02(scene: Scene) -> FacetResult:
    """Shepard rank fidelity. Frozen SHA-256: d4a6ff0c6a7699832bfe925d38feee236309ca858a371b36ba4fb5169e294235."""

    distances = graph_distances(scene)
    component_defects: List[float] = []
    component_weights: List[float] = []
    graph_values: List[float] = []
    degenerate_count = 0
    for members in components(scene):
        if len(members) < 3:
            continue
        indices = torch.tensor(members, dtype=torch.long)
        rows, cols = torch.triu_indices(len(members), len(members), offset=1)
        source = indices[rows]
        target = indices[cols]
        graph_order = distances[source, target]
        layout = torch.linalg.vector_norm(scene.positions[source] - scene.positions[target], dim=1)
        graph_values.extend(float(value) for value in graph_order)
        if torch.unique(graph_order).numel() == 1 or torch.unique(layout).numel() == 1:
            defect = 0.5
            degenerate_count += 1
        else:
            defect = correlation_defect(midranks(graph_order), midranks(layout))
        component_defects.append(defect)
        component_weights.append(float(graph_order.numel()))
    if not component_defects:
        return na_result("too_few_distance_pairs")
    if len(set(graph_values)) == 1:
        return na_result("constant_graph_distance")
    defect = global_blend(component_defects, component_weights)
    return value_result(
        defect,
        {"U02.headline": defect},
        {
            "pair_count": int(sum(component_weights)),
            "degenerate_layout_ranks": degenerate_count,
        },
    )


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

    graph_order = graph_distances(scene)
    layout_order = torch.cdist(scene.positions, scene.positions)
    values: Dict[str, float] = {}
    eligibility: Dict[str, bool] = {}
    degree_terciles: Dict[str, Tuple[Tuple[int, ...], ...]] = {}
    degree_stratum_defects: Dict[str, Tuple[float, ...]] = {}
    center_panels: Dict[str, Tuple[int, ...]] = {}
    graph_adjacency = adjacency(scene)
    for radius in (1, 2, 4):
        component_values: List[float] = []
        component_weights: List[float] = []
        radius_eligible = False
        for component_index, members in enumerate(components(scene)):
            if len(members) <= 2:
                continue
            member_set = set(members)
            component_distances = graph_order[members][:, members]
            component_diameter = int(torch.max(component_distances).item())
            if radius > max(1, math.floor(component_diameter / 2.0)):
                continue
            sampled_centers = sorted(
                members,
                key=lambda node: (
                    hashlib.sha256(
                        f"{scene.graph_hash}:U03-ctr:{component_index}:0:{node}".encode()
                    ).digest(),
                    node,
                ),
            )[: min(len(members), 256)]
            panel_key = f"component_{component_index}.panel_0"
            center_panels[panel_key] = tuple(sampled_centers)
            if len(members) > len(sampled_centers):
                sampled_set = set(sampled_centers)
                second_panel = sorted(
                    (node for node in members if node not in sampled_set),
                    key=lambda node: (
                        hashlib.sha256(
                            f"{scene.graph_hash}:U03-ctr:{component_index}:1:{node}".encode()
                        ).digest(),
                        node,
                    ),
                )[: min(len(members) - len(sampled_centers), 256)]
                center_panels[f"component_{component_index}.panel_1"] = tuple(second_panel)
            coverages = [
                len(_radius_neighbors(graph_order, node, radius) & member_set) / (len(members) - 1)
                for node in sampled_centers
            ]
            coverage_median = float(torch.median(torch.tensor(coverages, dtype=torch.float64)))
            if coverage_median > 0.9:
                continue
            radius_eligible = True
            degree_order = sorted(members, key=lambda node: (len(graph_adjacency[node]), node))
            sampled_set = set(sampled_centers)
            terciles = [
                [
                    node
                    for node in degree_order[
                        (index * len(degree_order)) // 3 : ((index + 1) * len(degree_order)) // 3
                    ]
                    if node in sampled_set
                ]
                for index in range(3)
            ]
            statistic_key = f"r_{radius}.component_{component_index}"
            degree_terciles[statistic_key] = tuple(tuple(centers) for centers in terciles)
            tercile_values: List[float] = []
            for centers in terciles:
                center_defects: List[float] = []
                for node in centers:
                    expected = _radius_neighbors(graph_order, node, radius) & member_set
                    if not expected:
                        continue
                    candidate_nodes = [candidate for candidate in members if candidate != node]
                    candidate_distances = layout_order[node, candidate_nodes]
                    k_value = len(expected)
                    rho = float(torch.kthvalue(candidate_distances, k_value).values)
                    if rho == 0.0:
                        center_defects.append(0.0)
                        continue
                    credits = []
                    for neighbor in expected:
                        argument = (rho - float(layout_order[node, neighbor])) / (0.25 * rho)
                        argument = max(-60.0, min(60.0, argument))
                        credits.append(1.0 / (1.0 + math.exp(-argument)))
                    raw_defect = 1.0 - sum(credits) / k_value
                    gate = float(
                        smoothstep(
                            torch.tensor(rho / (0.1 * scene.intrinsic_unit), dtype=torch.float64)
                        )
                    )
                    center_defects.append(gate * raw_defect)
                if center_defects:
                    tercile_values.append(global_blend(center_defects))
            degree_stratum_defects[statistic_key] = tuple(tercile_values)
            if tercile_values:
                component_values.append(snap_unit(sum(tercile_values) / len(tercile_values)))
                # Section 4 weights block: "Components pooled by node-count
                # input mass". Section 7's "components by node mass" summary
                # conflicts; the dedicated weights block governs (docketed).
                component_weights.append(float(len(members)))
        eligibility[f"r_{radius}"] = radius_eligible
        if component_values:
            values[f"U03.r_{radius}"] = snap_unit(
                sum(weight * value for weight, value in zip(component_weights, component_values))
                / sum(component_weights)
            )
    if not values:
        return na_result("neighborhoods_saturated")
    return mean_result(
        "U03",
        values,
        {
            "radius_eligibility": eligibility,
            "degree_terciles": degree_terciles,
            "degree_stratum_defects": degree_stratum_defects,
            "center_panels": center_panels,
        },
        renormalize_missing=False,
    )


def _grid_coarsening(node_count: int) -> float:
    """Return U04's input-only power-of-two coarsening factor.

    Parameters
    ----------
    node_count : int
        Positive graph node count.

    Returns
    -------
    float
        ``2^ceil(max(0, log4(N/4096)))``.
    """

    exponent = math.ceil(max(0.0, math.log(max(node_count / 4096.0, 1.0), 4.0)))
    return float(2**exponent)


def _rotation_matrix(angle: float) -> torch.Tensor:
    """Return a float64 planar rotation matrix.

    Parameters
    ----------
    angle : float
        Counter-clockwise angle in radians.

    Returns
    -------
    torch.Tensor
        Rotation matrix with shape ``[2, 2]``.
    """

    cosine = math.cos(angle)
    sine = math.sin(angle)
    return torch.tensor([[cosine, -sine], [sine, cosine]], dtype=torch.float64)


def _box_polygon(box: BoxGeometry) -> torch.Tensor:
    """Return one axis-aligned BoxGeometry as a counter-clockwise polygon.

    Parameters
    ----------
    box : BoxGeometry
        Axis-aligned box with center and half-extents tensors.

    Returns
    -------
    torch.Tensor
        Four vertices with shape ``[4, 2]``.
    """

    center = box.center
    half = box.half_extents
    return center + torch.tensor(
        [
            [-float(half[0]), -float(half[1])],
            [float(half[0]), -float(half[1])],
            [float(half[0]), float(half[1])],
            [-float(half[0]), float(half[1])],
        ],
        dtype=torch.float64,
    )


def _capsule_polygon(
    start: torch.Tensor, end: torch.Tensor, width: float, tolerance: float
) -> torch.Tensor:
    """Flatten one round-capped segment at the style-owned tolerance.

    Parameters
    ----------
    start, end : torch.Tensor
        Distinct segment endpoints with shape ``[2]``.
    width : float
        Positive full stroke width.
    tolerance : float
        Positive maximum radial sagitta of the flattened round caps.

    Returns
    -------
    torch.Tensor
        Counter-clockwise convex capsule polygon.
    """

    direction = end - start
    angle = math.atan2(float(direction[1]), float(direction[0]))
    radius = width / 2.0
    ratio = max(0.0, 1.0 - tolerance / radius)
    maximum_step = 2.0 * math.acos(ratio)
    semicircle_segments = max(2, math.ceil(math.pi / maximum_step))
    start_angles = torch.linspace(
        angle + math.pi / 2.0,
        angle + 3.0 * math.pi / 2.0,
        semicircle_segments + 1,
        dtype=torch.float64,
    )
    end_angles = torch.linspace(
        angle - math.pi / 2.0,
        angle + math.pi / 2.0,
        semicircle_segments + 1,
        dtype=torch.float64,
    )
    start_cap = start + radius * torch.stack(
        (torch.cos(start_angles), torch.sin(start_angles)), dim=1
    )
    end_cap = end + radius * torch.stack((torch.cos(end_angles), torch.sin(end_angles)), dim=1)
    return torch.cat((start_cap, end_cap), dim=0)


def _route_stroke_polygon_groups(scene: Scene) -> List[List[torch.Tensor]]:
    """Return one flattened capsule-polygon group per semantic route.

    Parameters
    ----------
    scene : Scene
        Validated scene.

    Returns
    -------
    list[list[torch.Tensor]]
        One list of convex segment capsules per route.
    """

    groups: List[List[torch.Tensor]] = []
    for route in resolved_routes(scene):
        width = (
            scene.style.edge_stroke_widths[route.edge_index]
            if scene.style.edge_stroke_widths
            else scene.style.route_stroke_width * scene.style.coordinate_scale
        )
        capsules = [
            _capsule_polygon(start, end, width, scene.style.flattening_tolerance)
            for start, end in zip(route.points[:-1], route.points[1:])
            if float(torch.linalg.vector_norm(end - start)) > 0.0
        ]
        if capsules:
            groups.append(capsules)
    return groups


def _opaque_polygon_groups(
    scene: Scene, angle: float, center: torch.Tensor
) -> List[List[torch.Tensor]]:
    """Build all flattened opaque primitive groups in one rotated grid frame.

    Parameters
    ----------
    scene : Scene
        Validated scene.
    angle : float
        Grid-frame rotation angle.
    center : torch.Tensor
        Rotation center with shape ``[2]``.

    Returns
    -------
    list[list[torch.Tensor]]
        Rotated primitive groups. Each box is a singleton; each route contains its
        flattened segment capsules.
    """

    groups = [[_box_polygon(box)] for box in scene.node_boxes]
    groups.extend([_box_polygon(box)] for box in scene.node_label_boxes)
    groups.extend([_box_polygon(box)] for box in scene.edge_label_boxes)
    groups.extend(_route_stroke_polygon_groups(scene))
    rotation = _rotation_matrix(-angle)
    return [[(polygon - center) @ rotation.T + center for polygon in group] for group in groups]


def _polygon_area(polygon: torch.Tensor) -> float:
    """Return the unsigned shoelace area of one polygon.

    Parameters
    ----------
    polygon : torch.Tensor
        Vertices with shape ``[P, 2]``.

    Returns
    -------
    float
        Nonnegative area.
    """

    if polygon.shape[0] < 3:
        return 0.0
    following = torch.roll(polygon, shifts=-1, dims=0)
    return (
        abs(float(torch.sum(polygon[:, 0] * following[:, 1] - polygon[:, 1] * following[:, 0])))
        / 2.0
    )


def _clip_polygon_axis(
    polygon: torch.Tensor, axis: int, boundary: float, keep_greater: bool
) -> torch.Tensor:
    """Clip a polygon against one axis-aligned half-plane.

    Parameters
    ----------
    polygon : torch.Tensor
        Vertices with shape ``[P, 2]``.
    axis : int
        Coordinate axis, zero or one.
    boundary : float
        Half-plane boundary coordinate.
    keep_greater : bool
        Keep coordinates greater than the boundary when true.

    Returns
    -------
    torch.Tensor
        Clipped polygon vertices.
    """

    if polygon.shape[0] == 0:
        return polygon
    output: List[torch.Tensor] = []

    def inside(point: torch.Tensor) -> bool:
        """Test one vertex against the active half-plane.

        Parameters
        ----------
        point : torch.Tensor
            Vertex with shape ``[2]``.

        Returns
        -------
        bool
            Whether the vertex is retained.
        """

        value = float(point[axis])
        return value >= boundary if keep_greater else value <= boundary

    for start, end in zip(torch.roll(polygon, shifts=1, dims=0), polygon):
        start_inside = inside(start)
        end_inside = inside(end)
        if start_inside != end_inside:
            delta = float(end[axis] - start[axis])
            parameter = (boundary - float(start[axis])) / delta
            output.append(start + parameter * (end - start))
        if end_inside:
            output.append(end)
    return torch.stack(output) if output else torch.empty((0, 2), dtype=torch.float64)


def _clip_polygon_cell(
    polygon: torch.Tensor, left: float, right: float, bottom: float, top: float
) -> torch.Tensor:
    """Clip one polygon to an axis-aligned grid cell.

    Parameters
    ----------
    polygon : torch.Tensor
        Vertices with shape ``[P, 2]``.
    left, right, bottom, top : float
        Cell bounds.

    Returns
    -------
    torch.Tensor
        Clipped vertices.
    """

    result = _clip_polygon_axis(polygon, 0, left, True)
    result = _clip_polygon_axis(result, 0, right, False)
    result = _clip_polygon_axis(result, 1, bottom, True)
    return _clip_polygon_axis(result, 1, top, False)


def _segment_intersection_x(
    start_a: torch.Tensor,
    end_a: torch.Tensor,
    start_b: torch.Tensor,
    end_b: torch.Tensor,
) -> float | None:
    """Return the x-coordinate of one proper segment intersection.

    Parameters
    ----------
    start_a, end_a, start_b, end_b : torch.Tensor
        Segment endpoints with shape ``[2]``.

    Returns
    -------
    float or None
        Proper-intersection x coordinate.
    """

    direction_a = end_a - start_a
    direction_b = end_b - start_b
    denominator = float(direction_a[0] * direction_b[1] - direction_a[1] * direction_b[0])
    if denominator == 0.0:
        return None
    offset = start_b - start_a
    parameter_a = float(offset[0] * direction_b[1] - offset[1] * direction_b[0]) / denominator
    parameter_b = float(offset[0] * direction_a[1] - offset[1] * direction_a[0]) / denominator
    if 0.0 < parameter_a < 1.0 and 0.0 < parameter_b < 1.0:
        return float((start_a + parameter_a * direction_a)[0])
    return None


def _vertical_polygon_interval(polygon: torch.Tensor, x_value: float) -> Tuple[float, float] | None:
    """Intersect a convex polygon with one open vertical line.

    Parameters
    ----------
    polygon : torch.Tensor
        Convex vertices with shape ``[P, 2]``.
    x_value : float
        Interior slab coordinate, never a polygon vertex x.

    Returns
    -------
    tuple[float, float] or None
        Closed y interval.
    """

    intersections: List[float] = []
    for start, end in zip(polygon, torch.roll(polygon, shifts=-1, dims=0)):
        start_x = float(start[0])
        end_x = float(end[0])
        if (start_x < x_value < end_x) or (end_x < x_value < start_x):
            parameter = (x_value - start_x) / (end_x - start_x)
            intersections.append(float(start[1] + parameter * (end[1] - start[1])))
    if len(intersections) < 2:
        return None
    return min(intersections), max(intersections)


def _union_vertical_length(polygons: Sequence[torch.Tensor], x_value: float) -> float:
    """Return union length of polygon vertical sections.

    Parameters
    ----------
    polygons : sequence[torch.Tensor]
        Convex polygons.
    x_value : float
        Open-slab x coordinate.

    Returns
    -------
    float
        Nonnegative union length.
    """

    intervals = sorted(
        interval
        for polygon in polygons
        if (interval := _vertical_polygon_interval(polygon, x_value)) is not None
    )
    if not intervals:
        return 0.0
    total = 0.0
    lower, upper = intervals[0]
    for next_lower, next_upper in intervals[1:]:
        if next_lower <= upper:
            upper = max(upper, next_upper)
        else:
            total += upper - lower
            lower, upper = next_lower, next_upper
    return total + upper - lower


def _polygon_union_area(polygons: Sequence[torch.Tensor]) -> float:
    """Return exact union area of convex polygons by vertical decomposition.

    Parameters
    ----------
    polygons : sequence[torch.Tensor]
        Convex clipped polygons.

    Returns
    -------
    float
        Nonnegative union area.
    """

    valid = [polygon for polygon in polygons if polygon.shape[0] >= 3]
    if not valid:
        return 0.0
    x_values = {float(point[0]) for polygon in valid for point in polygon}
    edges = [
        (start, end)
        for polygon in valid
        for start, end in zip(polygon, torch.roll(polygon, shifts=-1, dims=0))
    ]
    for index, (start_a, end_a) in enumerate(edges):
        for start_b, end_b in edges[index + 1 :]:
            coordinate = _segment_intersection_x(start_a, end_a, start_b, end_b)
            if coordinate is not None:
                x_values.add(coordinate)
    ordered = sorted(x_values)
    area = 0.0
    for left, right in zip(ordered[:-1], ordered[1:]):
        if right <= left:
            continue
        first = left + (right - left) / 3.0
        second = left + 2.0 * (right - left) / 3.0
        area += (
            (right - left)
            * (_union_vertical_length(valid, first) + _union_vertical_length(valid, second))
            / 2.0
        )
    return area


def _grid_geometry(
    scene: Scene, cell_size: float, angle: float
) -> Tuple[torch.Tensor, float, float, int, int, List[List[torch.Tensor]]]:
    """Build one U04 oriented measurement grid and rotated primitive list.

    Parameters
    ----------
    scene : Scene
        Validated scene.
    cell_size : float
        Input-only cell side length.
    angle : float
        Grid orientation.

    Returns
    -------
    tuple
        Frame center, lower x/y, grid width/height, and rotated primitive groups.
    """

    frame = robust_frame(scene.positions, scene.intrinsic_unit)
    clamp = 16.0 * math.sqrt(scene.node_count) * scene.intrinsic_unit
    half = torch.minimum(frame.half_extents, torch.full((2,), clamp, dtype=torch.float64))
    corners = frame.center + torch.tensor(
        [
            [-float(half[0]), -float(half[1])],
            [float(half[0]), -float(half[1])],
            [float(half[0]), float(half[1])],
            [-float(half[0]), float(half[1])],
        ],
        dtype=torch.float64,
    )
    rotation = _rotation_matrix(-angle)
    rotated = (corners - frame.center) @ rotation.T + frame.center
    minimum_x = min(float(point[0]) for point in rotated) - cell_size
    maximum_x = max(float(point[0]) for point in rotated) + cell_size
    minimum_y = min(float(point[1]) for point in rotated) - cell_size
    maximum_y = max(float(point[1]) for point in rotated) + cell_size
    width = max(1, math.ceil((maximum_x - minimum_x) / cell_size))
    height = max(1, math.ceil((maximum_y - minimum_y) / cell_size))
    return (
        frame.center,
        minimum_x,
        minimum_y,
        width,
        height,
        _opaque_polygon_groups(scene, angle, frame.center),
    )


def _cell_polygon_masses(
    polygon_groups: Sequence[Sequence[torch.Tensor]],
    minimum_x: float,
    minimum_y: float,
    width: int,
    height: int,
    cell_size: float,
) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], List[torch.Tensor]]]:
    """Clip opaque polygons into grid cells and accumulate summed ink.

    Parameters
    ----------
    polygon_groups : sequence[sequence[torch.Tensor]]
        Rotated convex polygons grouped by opaque primitive.
    minimum_x, minimum_y : float
        Grid lower corner.
    width, height : int
        Grid dimensions.
    cell_size : float
        Cell side length.

    Returns
    -------
    tuple[dict, dict]
        Summed ink area and clipped polygons per occupied cell.
    """

    masses: Dict[Tuple[int, int], float] = {}
    clipped_by_cell: Dict[Tuple[int, int], List[torch.Tensor]] = {}
    for group in polygon_groups:
        polygon_min_x = min(float(point[0]) for polygon in group for point in polygon)
        polygon_max_x = max(float(point[0]) for polygon in group for point in polygon)
        polygon_min_y = min(float(point[1]) for polygon in group for point in polygon)
        polygon_max_y = max(float(point[1]) for polygon in group for point in polygon)
        first_x = max(0, math.floor((polygon_min_x - minimum_x) / cell_size))
        last_x = min(width - 1, math.floor((polygon_max_x - minimum_x) / cell_size))
        first_y = max(0, math.floor((polygon_min_y - minimum_y) / cell_size))
        last_y = min(height - 1, math.floor((polygon_max_y - minimum_y) / cell_size))
        for column in range(first_x, last_x + 1):
            left = minimum_x + column * cell_size
            right = left + cell_size
            for row in range(first_y, last_y + 1):
                bottom = minimum_y + row * cell_size
                top = bottom + cell_size
                clipped: List[torch.Tensor] = []
                for polygon in group:
                    clipped_polygon = _clip_polygon_cell(polygon, left, right, bottom, top)
                    if clipped_polygon.shape[0] >= 3:
                        clipped.append(clipped_polygon)
                area = _polygon_union_area(clipped)
                if area == 0.0:
                    continue
                key = (column, row)
                masses[key] = masses.get(key, 0.0) + area
                clipped_by_cell.setdefault(key, []).extend(clipped)
    return masses, clipped_by_cell


def _gaussian_cell_mass(
    center_x: float,
    center_y: float,
    left: float,
    right: float,
    bottom: float,
    top: float,
    bandwidth: float,
) -> float:
    """Integrate a normalized isotropic Gaussian over one cell.

    Parameters
    ----------
    center_x, center_y : float
        Gaussian center.
    left, right, bottom, top : float
        Cell bounds.
    bandwidth : float
        Positive isotropic standard deviation.

    Returns
    -------
    float
        Probability mass in the cell.
    """

    scale = math.sqrt(2.0) * bandwidth
    x_mass = 0.5 * (math.erf((right - center_x) / scale) - math.erf((left - center_x) / scale))
    y_mass = 0.5 * (math.erf((top - center_y) / scale) - math.erf((bottom - center_y) / scale))
    return max(0.0, x_mass * y_mass)


def _density_grid_defect(scene: Scene, cell_size: float, angle: float) -> Tuple[float, float]:
    """Evaluate one exact polygon/Gaussian U04a oriented grid.

    Parameters
    ----------
    scene : Scene
        Validated scene.
    cell_size : float
        Input-only cell size.
    angle : float
        Grid orientation.

    Returns
    -------
    tuple[float, float]
        Normalized JSD and escaped-ink fraction.
    """

    center, minimum_x, minimum_y, width, height, groups = _grid_geometry(scene, cell_size, angle)
    occupancy, _ = _cell_polygon_masses(groups, minimum_x, minimum_y, width, height, cell_size)
    total_ink = sum(_polygon_union_area(group) for group in groups)
    inside_ink = sum(occupancy.values())
    outside_ink = max(0.0, total_ink - inside_ink)
    rotation = _rotation_matrix(-angle)
    positions = (scene.positions - center) @ rotation.T + center
    degrees = node_degrees(scene)
    node_masses = 1.0 + degrees / 2.0
    total_demand = float(torch.sum(node_masses))
    demand: Dict[Tuple[int, int], float] = {}
    bandwidth = 2.0 * scene.intrinsic_unit
    for column in range(width):
        left = minimum_x + column * cell_size
        right = left + cell_size
        for row in range(height):
            bottom = minimum_y + row * cell_size
            top = bottom + cell_size
            mass = sum(
                float(node_masses[node])
                * _gaussian_cell_mass(
                    float(positions[node, 0]),
                    float(positions[node, 1]),
                    left,
                    right,
                    bottom,
                    top,
                    bandwidth,
                )
                for node in range(scene.node_count)
            )
            if mass > 0.0:
                demand[(column, row)] = mass
    outside_demand = max(0.0, total_demand - sum(demand.values()))
    keys = set(occupancy) | set(demand)
    left_divergence = 0.0
    right_divergence = 0.0
    for key in keys | {(-1, -1)}:
        occupancy_mass = outside_ink if key == (-1, -1) else occupancy.get(key, 0.0)
        demand_mass = outside_demand if key == (-1, -1) else demand.get(key, 0.0)
        occupancy_probability = occupancy_mass / total_ink
        demand_probability = demand_mass / total_demand
        mixture = (occupancy_probability + demand_probability) / 2.0
        if occupancy_probability > 0.0:
            left_divergence += occupancy_probability * math.log(occupancy_probability / mixture)
        if demand_probability > 0.0:
            right_divergence += demand_probability * math.log(demand_probability / mixture)
    normalized = (left_divergence + right_divergence) / (2.0 * math.log(2.0))
    # JSD over 2 log 2 is analytically in [0, 1]; only float dust is shed
    # (snap_unit), so a real range violation still reaches the guards.
    return snap_unit(normalized), outside_ink / total_ink


def _crowding_grid_defect(scene: Scene, cell_size: float, angle: float) -> float:
    """Evaluate one U04b ink-mass CVaR over exact polygon union coverage.

    Parameters
    ----------
    scene : Scene
        Validated scene.
    cell_size : float
        Input-only cell side.
    angle : float
        Grid orientation.

    Returns
    -------
    float
        Worst-10%-ink coverage severity.
    """

    _, minimum_x, minimum_y, width, height, groups = _grid_geometry(scene, cell_size, angle)
    ink, clipped = _cell_polygon_masses(groups, minimum_x, minimum_y, width, height, cell_size)
    records = []
    for key, mass in ink.items():
        coverage = min(1.0, _polygon_union_area(clipped[key]) / (cell_size * cell_size))
        severity = float(smoothstep(torch.tensor((coverage - 0.5) / 0.4, dtype=torch.float64)))
        records.append((severity, mass, key))
    if not records:
        return 0.0
    records.sort(key=lambda item: (-item[0], item[2]))
    tail_mass = 0.10 * sum(item[1] for item in records)
    remaining = tail_mass
    burden = 0.0
    for severity, mass, _ in records:
        selected = min(mass, remaining)
        burden += selected * severity
        remaining -= selected
        if remaining <= 0.0:
            break
    return burden / tail_mass


def U04a(scene: Scene) -> FacetResult:
    """Density-map fidelity (structure). Frozen SHA-256: 59a7eeda57c5735b2ced3d050af16b57181e127735012a0deb20ac7fff2d08db."""

    if scene.node_count < 10:
        return na_result("no_declared_geometry")
    coarsening = _grid_coarsening(scene.node_count)
    values: Dict[str, float] = {}
    escaped: Dict[str, Tuple[float, ...]] = {}
    for label, base in (("U04a.2u", 2.0), ("U04a.8u", 8.0)):
        oriented = [
            _density_grid_defect(
                scene,
                base * scene.intrinsic_unit * coarsening,
                index * (math.pi / 2.0) / 8.0,
            )
            for index in range(8)
        ]
        values[label] = sum(item[0] for item in oriented) / 8.0
        escaped[label] = tuple(item[1] for item in oriented)
    return mean_result(
        "U04a",
        values,
        {
            "kappa": coarsening,
            "orientation_count": 8,
            "escaped_ink_fractions": escaped,
        },
    )


def U04b(scene: Scene) -> FacetResult:
    """Crowding / whitespace legibility. Frozen SHA-256: 2005ce3d0597b90fefc5f0cc5197f44b64b2c24911eea27e858afe2f2e3c3dd4."""

    if scene.node_count < 10:
        return na_result("no_declared_geometry")
    coarsening = _grid_coarsening(scene.node_count)
    values: Dict[str, float] = {}
    for label, base in (("U04b.part_1", 1.0), ("U04b.part_2", 4.0)):
        oriented = [
            _crowding_grid_defect(
                scene,
                base * scene.intrinsic_unit * coarsening,
                index * (math.pi / 2.0) / 8.0,
            )
            for index in range(8)
        ]
        values[label] = sum(oriented) / 8.0
    return mean_result(
        "U04b",
        values,
        {"kappa": coarsening, "orientation_count": 8},
    )


def U05(scene: Scene) -> FacetResult:
    """Graph-geometric shape fidelity. Frozen SHA-256: 47a4b416c09f564831aaf1d9fb66a80bef138bce6bb8daf9a02c3d3b1fa0cfa3."""

    resistance = _effective_resistance_metric(scene)
    defects: List[float] = []
    weights: List[float] = []
    for members in components(scene):
        if len(members) < 3:
            continue
        indices = torch.tensor(members, dtype=torch.long)
        rows, cols = torch.triu_indices(len(members), len(members), offset=1)
        source = indices[rows]
        target = indices[cols]
        order = resistance[source, target]
        layout = torch.linalg.vector_norm(scene.positions[source] - scene.positions[target], dim=1)
        defects.append(isotonic_stress(order, layout))
        weights.append(float(order.numel()))
    if not defects:
        return na_result("too_few_resistance_pairs")
    defect = global_blend(defects, weights)
    return value_result(defect, {"U05.headline": defect}, {"component_count": len(defects)})


def _effective_resistance_metric(scene: Scene) -> torch.Tensor:
    """Compute the exact square-root effective-resistance metric.

    Parameters
    ----------
    scene : Scene
        Validated graph scene.

    Returns
    -------
    torch.Tensor
        Pairwise ``sqrt(R_eff)`` matrix with shape ``[N, N]``.
    """

    count = scene.node_count
    result = torch.full((count, count), float("inf"), dtype=torch.float64)
    for members in components(scene):
        size = len(members)
        if size == 1:
            result[members[0], members[0]] = 0.0
            continue
        local = {node: index for index, node in enumerate(members)}
        laplacian = torch.zeros((size, size), dtype=torch.float64)
        for source, target in scene.graph.edges:
            if source == target or source not in local or target not in local:
                continue
            left = local[source]
            right = local[target]
            laplacian[left, left] += 1.0
            laplacian[right, right] += 1.0
            laplacian[left, right] -= 1.0
            laplacian[right, left] -= 1.0
        inverse = torch.linalg.pinv(laplacian, hermitian=True)
        diagonal = torch.diagonal(inverse)
        resistance = torch.clamp(diagonal[:, None] + diagonal[None, :] - 2.0 * inverse, min=0.0)
        indices = torch.tensor(members, dtype=torch.long)
        result[indices[:, None], indices[None, :]] = torch.sqrt(resistance)
    return result


def U06(scene: Scene) -> FacetResult:
    """Symmetry display. Frozen SHA-256: 9f6f3aa20625f493f3193bb2c8e377b79d2801257ef494d71cd433421497dabe."""

    if not scene.graph.symmetry_generators:
        return na_result("no_certified_symmetry")
    residuals: List[float] = []
    for permutation in scene.graph.symmetry_generators:
        moved = [index for index, target in enumerate(permutation) if index != target]
        if not moved:
            continue
        source = scene.positions[moved]
        target = scene.positions[[permutation[index] for index in moved]]
        source = source - torch.mean(source, dim=0)
        target = target - torch.mean(target, dim=0)
        left, _, right = torch.linalg.svd(source.T @ target)
        rotation = left @ right
        denominator = float(torch.sum(source * source).item())
        scale = (
            float(torch.sum(torch.linalg.svdvals(source.T @ target)).item()) / denominator
            if denominator > 0.0
            else 0.0
        )
        center = torch.median(source, dim=0).values
        spread = float(torch.median(torch.linalg.vector_norm(source - center, dim=1)).item())
        if spread == 0.0:
            residuals.append(1.0)
            continue
        aligned = scale * source @ rotation
        residual = torch.sqrt(torch.mean(torch.sum((aligned - target) ** 2, dim=1)))
        ratio = float(residual) / spread
        raw = ratio * ratio / (1.0 + ratio * ratio)
        gate = float(
            smoothstep(torch.tensor(spread / (0.1 * scene.intrinsic_unit), dtype=torch.float64))
        )
        residuals.append(gate * raw + (1.0 - gate))
    if not residuals:
        return na_result("no_certified_symmetry")
    defect = global_blend(residuals)
    return value_result(
        defect,
        {"U06.headline": defect},
        {"certified_generator_count": len(residuals)},
    )


def U09(scene: Scene) -> FacetResult:
    """Edge-length coherence, stratified/conditional. Frozen SHA-256: deda54396b4f2de53842b94593bb0d0cc1fdd2084cc0874cec819c0de15abad7."""

    if scene.edge_count < 5:
        return na_result("too_few_edges")
    lengths = torch.tensor(
        [
            float(torch.sum(torch.linalg.vector_norm(route.points[1:] - route.points[:-1], dim=1)))
            for route in resolved_routes(scene)
        ],
        dtype=torch.float64,
    )
    log_lengths = torch.log(lengths + 1e-12 * scene.intrinsic_unit)
    if scene.graph.edge_weights is not None and scene.graph.weight_semantics == "target_length":
        targets = torch.tensor(scene.graph.edge_weights, dtype=torch.float64)
        log_lengths = log_lengths - torch.log(targets)
    component_index: Dict[int, int] = {}
    for index, members in enumerate(components(scene)):
        for node in members:
            component_index[node] = index
    strata: Dict[int, List[int]] = {}
    for edge_index, (source, _) in enumerate(scene.graph.edges):
        strata.setdefault(component_index[source], []).append(edge_index)
    large_strata = [indices for indices in strata.values() if len(indices) >= 5]
    merged = [index for indices in strata.values() if len(indices) < 5 for index in indices]
    if merged:
        large_strata.append(merged)
    defects: List[float] = []
    weights: List[float] = []
    dispersions: List[float] = []
    for indices in large_strata:
        local_values = log_lengths[indices]
        median = torch.median(local_values)
        dispersion = 1.4826 * float(torch.median(torch.abs(local_values - median)).item())
        defects.append(dispersion / (dispersion + math.log(2.0)))
        weights.append(float(len(indices)))
        dispersions.append(dispersion)
    defect = global_blend(defects, weights)
    return value_result(
        defect,
        {"U09.headline": defect},
        {"dispersions": tuple(dispersions), "merged_edge_count": len(merged)},
    )


def U14(scene: Scene) -> FacetResult:
    """False adjacency (node-pair proximity ambiguity). Frozen SHA-256: 20a4536ac81681828042a6d06ad76113d0b5c08305e08887799956799a339d65."""

    neighbors = adjacency(scene)
    component_ids: Dict[int, int] = {}
    for component_index, members in enumerate(components(scene)):
        for member in members:
            component_ids[member] = component_index
    pair_population = [
        (left, right)
        for left in range(scene.node_count)
        for right in range(left + 1, scene.node_count)
        if component_ids[left] == component_ids[right] and right not in neighbors[left]
    ]
    if len(pair_population) < 20:
        return na_result("too_few_same_component_nonedges", {"pair_count": len(pair_population)})

    diagonals = [
        float(2.0 * torch.linalg.vector_norm(box.half_extents)) for box in scene.node_boxes
    ]
    budgets: List[float] = []
    for node, local_neighbors in enumerate(neighbors):
        degree = len(local_neighbors)
        maximum_neighbor = max((diagonals[item] for item in local_neighbors), default=0.0)
        radius = 1.5 * (diagonals[node] / 2.0 + maximum_neighbor / 2.0)
        demand = sum(diagonals[item] for item in local_neighbors)
        raw = (2.0 * math.pi * radius - demand) / max(degree, 1)
        if degree == 0:
            raw = 1.5 * diagonals[node] / 2.0
        budgets.append(max(raw, 0.125 * scene.intrinsic_unit))
    survival = [1.0] * scene.node_count
    raw_pairs: List[Dict[str, float]] = []
    floor = 0.25 * scene.intrinsic_unit
    for left, right in pair_population:
        signed_gap, _ = aabb_pair(scene.node_boxes[left], scene.node_boxes[right])
        gap = max(0.0, signed_gap)
        normalized = gap / scene.intrinsic_unit
        absolute = (1.0 - normalized * normalized) ** 2 if normalized < 1.0 else 0.0
        achievable = min(budgets[left], budgets[right])
        achievement = 1.0 - float(smoothstep(torch.tensor(gap / achievable, dtype=torch.float64)))
        argument = (achievable - floor) / (0.5 * floor)
        blend = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, argument))))
        pair_defect = snap_unit((1.0 - blend) * absolute + blend * achievement)
        survival[left] *= 1.0 - pair_defect
        survival[right] *= 1.0 - pair_defect
        raw_pairs.append(
            {
                "left": float(left),
                "right": float(right),
                "gap_over_u": normalized,
                "budget_over_u": achievable / scene.intrinsic_unit,
                "lambda": blend,
                "d_abs": absolute,
                "d_ach": achievement,
                "defect": pair_defect,
            }
        )
    node_defects = [1.0 - value for value in survival]
    node_masses = (
        list(scene.graph.node_masses)
        if scene.graph.node_masses is not None
        else [1.0] * scene.node_count
    )
    defect = global_blend(node_defects, node_masses)
    return value_result(
        defect,
        {"U14.headline": defect},
        {
            "pair_count": len(pair_population),
            "node_defects": tuple(node_defects),
            "top_pairs": tuple(sorted(raw_pairs, key=lambda row: row["defect"], reverse=True)[:20]),
        },
    )


def U22(scene: Scene) -> FacetResult:
    """Aspect ratio / shape. Frozen SHA-256: d141e6fef540d3115fe342b275133278b550ee5d064663779b19c58b61b6304e."""

    if scene.node_count < 4:
        return na_result("insufficient_node_population")
    ranks = scene.graph.ranks
    axis = declared_axis(scene)
    floor_bound = False
    if axis is not None:
        cross = torch.tensor([-axis[1], axis[0]], dtype=torch.float64)
        axis_extent = robust_projection(scene.positions @ axis, scene.intrinsic_unit)
        cross_extent = robust_projection(scene.positions @ cross, scene.intrinsic_unit)
        # A_obs is breadth over depth: the sec 6 target max_layer_width /
        # n_layers pairs "40 nodes in 12 layers" with "SHOULD draw wide"
        # (> 1 means wider than deep), so the observed ratio must put the
        # cross-axis (breadth) extent in the numerator.
        observed = cross_extent.half_extent / axis_extent.half_extent
        floor_bound = axis_extent.floor_bound or cross_extent.floor_bound
        if ranks is not None:
            rank_tensor = torch.tensor(ranks, dtype=torch.long)
            counts = torch.stack(
                [(rank_tensor == rank).sum() for rank in torch.unique(rank_tensor)]
            )
            target = float(torch.max(counts)) / counts.numel()
        else:
            target = 1.0
        measurement = "declared_axis"
    else:
        extents = []
        for index in range(180):
            angle = math.pi * index / 180.0
            direction = torch.tensor([math.cos(angle), math.sin(angle)], dtype=torch.float64)
            extent = robust_projection(scene.positions @ direction, scene.intrinsic_unit)
            extents.append(extent.half_extent)
            floor_bound = floor_bound or extent.floor_bound
        observed = max(extents) / min(extents)
        if ranks is not None:
            # Section 6's layer-profile target is computed from declared ranks
            # alone; no direction is needed for the TARGET even though the
            # frame stays direction-free. The rotation-scan aspect is >= 1 by
            # construction, so the orientation-less target is folded onto the
            # same side of unity.
            rank_tensor = torch.tensor(ranks, dtype=torch.long)
            counts = torch.stack(
                [(rank_tensor == rank).sum() for rank in torch.unique(rank_tensor)]
            )
            profile = float(torch.max(counts)) / counts.numel()
            target = max(profile, 1.0 / profile)
        else:
            target = 1.0
        measurement = "frozen_direction_set"
    declared_class = scene.graph.declared_graph_class
    if declared_class is not None and declared_class not in {
        "path",
        "chain",
        "tree",
        "lattice",
        "grid",
        "cycle",
        "ring",
    }:
        # Section 13 case (c): a class outside the frozen exemption table's
        # domain is typed INVALID, conditioned on the class string alone --
        # no rank exception. A silent kappa_class = 1 fallback (or a
        # ranks-declaring graph silently keeping the layer-profile target)
        # would be the undocumented score-visible branch the contract
        # pre-bans.
        return invalid_result("unknown_declared_class")
    if declared_class is not None and ranks is None:
        # Section 6's kappa_class constants are elongation magnitudes
        # (direction-free convention, A_obs >= 1). In the declared-axis
        # branch `observed` is signed breadth/depth, so each constant maps
        # through its class's elongation direction: a path/chain elongates
        # ALONG the flow axis (depth), so kappa_class = 8 is a
        # breadth/depth target of 1/8; a tree's max(1, b/d) and a grid's
        # declared width/height are already breadth-over-depth quantities
        # and pass through unchanged. In the direction-free branch
        # (axis is None) observed >= 1 folds every target onto the same
        # side of unity, so the reciprocal is a no-op there. See
        # DISCREPANCIES.md entry 30.
        if declared_class in {"path", "chain"}:
            target *= 8.0 if axis is None else 1.0 / 8.0
        elif declared_class == "tree":
            if scene.graph.tree_depths is None:
                return invalid_result("unknown_declared_class")
            depth = max(scene.graph.tree_depths)
            breadth = max(
                scene.graph.tree_depths.count(level) for level in set(scene.graph.tree_depths)
            )
            target *= max(1.0, breadth / max(depth, 1))
        elif declared_class in {"lattice", "grid"}:
            if scene.graph.lattice_dimensions is None:
                return invalid_result("unknown_declared_class")
            width, height = scene.graph.lattice_dimensions
            aspect = width / height
            # Direction-free observed is >= 1 by construction, so the
            # declared aspect folds onto the same side of unity there
            # (the DISCREPANCIES.md entry 26 fold); the signed frame
            # takes it as declared.
            target *= aspect if axis is not None else max(aspect, 1.0 / aspect)
    excess = soft_pos(abs(math.log(observed / target)) - math.log(3.0))
    defect = excess / (1.0 + excess)
    return value_result(
        defect,
        {"U22.headline": defect},
        {
            "aspect_ratio": observed,
            "target": target,
            "measurement": measurement,
            "degenerate_extent_floor": floor_bound,
        },
    )


def U23(scene: Scene) -> FacetResult:
    """Visual balance. Frozen SHA-256: c14a86d9e00099c5db2c6d571e1af057896146222e96737678253a76ffdd5972."""

    defects: List[float] = []
    weights: List[float] = []
    normalized_values: List[float] = []
    measurement = "rotation_averaged"
    node_masses = (
        list(scene.graph.node_masses)
        if scene.graph.node_masses is not None
        else [1.0] * scene.node_count
    )
    for members in components(scene):
        if len(members) < 4:
            continue
        points = scene.positions[members]
        frame = robust_frame(points, scene.intrinsic_unit)
        local_masses = torch.tensor([node_masses[node] for node in members], dtype=torch.float64)
        centroid = torch.sum(points * local_masses[:, None], dim=0) / torch.sum(local_masses)
        offset = centroid - frame.center
        axis = declared_axis(scene)
        if axis is not None:
            cross = torch.tensor([-axis[1], axis[0]], dtype=torch.float64)
            extent = robust_projection(points @ cross, scene.intrinsic_unit)
            normalized = abs(float(torch.dot(offset, cross))) / extent.half_extent
            measurement = "declared_cross_axis"
        else:
            extents = []
            for index in range(180):
                angle = math.pi * index / 180.0
                direction = torch.tensor([math.cos(angle), math.sin(angle)], dtype=torch.float64)
                extents.append(
                    robust_projection(points @ direction, scene.intrinsic_unit).half_extent
                )
            normalized = float(torch.linalg.vector_norm(offset)) / (sum(extents) / len(extents))
        defects.append(float(smoothstep(torch.tensor(normalized / 0.5, dtype=torch.float64))))
        weights.append(float(torch.sum(local_masses)))
        normalized_values.append(normalized)
    if not defects:
        return na_result("component_too_small")
    defect = global_blend(defects, weights)
    return value_result(
        defect,
        {"U23.headline": defect},
        {"centroid_offsets": tuple(normalized_values), "measurement": measurement},
    )


def U24(scene: Scene) -> FacetResult:
    """Total ink economy. Frozen SHA-256: 5857b951b89bbebff02131437db2b8070d84dc03460746adc7f7e65ac1d03573."""

    if scene.edge_count == 0:
        return na_result("no_edges")
    if scene.node_count < 4:
        return na_result("insufficient_node_population")
    primitive_area = sum(float(4.0 * torch.prod(box.half_extents)) for box in scene.node_boxes)
    component_count = len(components(scene))
    area_reference = primitive_area / 0.10 * (1.0 + 0.5 * (component_count - 1))
    ideal_spacing = math.sqrt(area_reference / scene.node_count)
    route_length = 0.0
    for route in resolved_routes(scene):
        route_length += float(
            torch.sum(torch.linalg.vector_norm(route.points[1:] - route.points[:-1], dim=1))
        )
    ink_ratio = route_length / (scene.edge_count * ideal_spacing)
    excess = soft_pos(math.log(ink_ratio / 3.0)) if ink_ratio > 0.0 else 0.0
    defect = excess / (1.0 + excess)
    return value_result(
        defect,
        {"U24.headline": defect},
        {"ink_ratio": ink_ratio, "ideal_spacing": ideal_spacing},
    )
