"""Multi-component packing, planar-face, and channel-contrast facets."""

# Contract identity docstrings intentionally keep each title and digest on one line.
# ruff: noqa: E501

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

import torch

from dagua.eval.ruler_v4._util import (
    components,
    global_blend,
    mean_result,
    proper_intersection,
    resolved_routes,
    route_segments,
    smoothstep,
    snap_unit,
)
from dagua.eval.ruler_v4.frames import RobustFrame, robust_frame
from dagua.eval.ruler_v4.legibility import _box_segment_clearance, _segment_segment_distance
from dagua.eval.ruler_v4.scene import (
    BoxGeometry,
    FacetResult,
    Route,
    Scene,
    invalid_result,
    na_result,
)
from dagua.eval.ruler_v4.structure import _box_polygon, _capsule_polygon, _polygon_union_area


def _component_frames(scene: Scene) -> List[Tuple[List[int], RobustFrame]]:
    """Build robust frames for simple-support components.

    Parameters
    ----------
    scene : Scene
        Validated graph scene.

    Returns
    -------
    list[tuple[list[int], RobustFrame]]
        Canonical members and their robust frames.
    """

    return [
        (members, robust_frame(scene.positions[members], scene.intrinsic_unit))
        for members in components(scene)
    ]


def U38(scene: Scene) -> FacetResult:
    """Multi-component packing. Frozen SHA-256: 1a0023e6c0515714dbda83f97a6a57e1c4ecbf7e27f15a6650c674ca9f0f68b1."""

    frames = _component_frames(scene)
    if len(frames) < 2:
        return na_result("single_component")
    member_to_component = {
        member: component_index
        for component_index, (members, _) in enumerate(frames)
        for member in members
    }
    routes_by_component: Dict[int, List[Route]] = {index: [] for index in range(len(frames))}
    for route in resolved_routes(scene):
        source, _ = scene.graph.edges[route.edge_index]
        routes_by_component[member_to_component[source]].append(route)
    boxes_by_component = {
        index: [scene.node_boxes[node] for node in members]
        for index, (members, _) in enumerate(frames)
    }
    pair_losses: Dict[Tuple[int, int], float] = {}
    for left_index in range(len(frames)):
        for right_index in range(left_index + 1, len(frames)):
            clearance = _visible_component_clearance(
                boxes_by_component[left_index],
                routes_by_component[left_index],
                boxes_by_component[right_index],
                routes_by_component[right_index],
                scene,
            )
            argument = (0.50 * scene.intrinsic_unit - clearance) / (0.10 * scene.intrinsic_unit)
            pair_losses[(left_index, right_index)] = _stable_sigmoid(argument)
    component_masses = [len(members) for members, _ in frames]
    component_losses: List[float] = []
    for component_index in range(len(frames)):
        numerator = 0.0
        denominator = 0.0
        for other_index, mass in enumerate(component_masses):
            if other_index == component_index:
                continue
            key = tuple(sorted((component_index, other_index)))
            numerator += mass * pair_losses[key]
            denominator += mass
        component_losses.append(numerator / denominator)
    mass_mean = sum(mass * loss for mass, loss in zip(component_masses, component_losses)) / sum(
        component_masses
    )
    equal_mean = sum(component_losses) / len(component_losses)
    tail = _equal_cvar(component_losses, 0.20)
    clear = snap_unit(0.50 * mass_mean + 0.25 * equal_mean + 0.25 * tail)

    global_frame = robust_frame(scene.positions, scene.intrinsic_unit)
    component_areas = _raster_component_areas(
        boxes_by_component, routes_by_component, global_frame, scene
    )
    occupied = sum(component_areas) / global_frame.area
    pack = _stable_sigmoid((math.log(0.20) - math.log(occupied + 2.0**-40)) / 0.25)
    total_area = sum(component_areas)
    total_mass = sum(component_masses)
    area_shares = [area / total_area for area in component_areas]
    mass_shares = [mass / total_mass for mass in component_masses]
    midpoint = [(area + mass) / 2.0 for area, mass in zip(area_shares, mass_shares)]
    jsd = 0.0
    for area, mass, middle in zip(area_shares, mass_shares, midpoint):
        if area > 0.0:
            jsd += 0.5 * area * math.log(area / middle)
        if mass > 0.0:
            jsd += 0.5 * mass * math.log(mass / middle)
    # The Jensen-Shannon divergence is analytically >= 0, but the signed log
    # sum returns ~-8e-17 when the two share vectors agree to within dust.
    proportionality = snap_unit(jsd / math.log(2.0))
    values = {
        "U38.L_clear": clear,
        "U38.L_pack": pack,
        "U38.L_prop": proportionality,
    }
    return mean_result(
        "U38",
        values,
        {
            "component_count": len(frames),
            "component_losses": tuple(component_losses),
            "component_areas": tuple(component_areas),
            "occupied_fraction": occupied,
            "pair_losses": pair_losses,
        },
    )


def _stable_sigmoid(value: float) -> float:
    """Evaluate a numerically stable scalar logistic.

    Parameters
    ----------
    value : float
        Logistic argument.

    Returns
    -------
    float
        Value in ``[0, 1]``.
    """

    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-min(value, 60.0)))
    exponential = math.exp(max(value, -60.0))
    return exponential / (1.0 + exponential)


def _equal_cvar(values: Sequence[float], tail_fraction: float) -> float:
    """Return an exact equal-mass upper-tail mean.

    Parameters
    ----------
    values : sequence[float]
        Nonempty loss population.
    tail_fraction : float
        Tail mass fraction in ``(0, 1]``.

    Returns
    -------
    float
        Fractional-boundary worst-tail mean.
    """

    ordered = sorted(values, reverse=True)
    target = tail_fraction * len(ordered)
    if target <= 1.0:
        return ordered[0]
    whole = int(math.floor(target))
    fraction = target - whole
    burden = sum(ordered[:whole])
    if fraction > 0.0 and whole < len(ordered):
        burden += fraction * ordered[whole]
    return burden / target


def _visible_component_clearance(
    left_boxes: Sequence[BoxGeometry],
    left_routes: Sequence[Route],
    right_boxes: Sequence[BoxGeometry],
    right_routes: Sequence[Route],
    scene: Scene,
) -> float:
    """Return signed clearance between two components' visible geometry.

    Parameters
    ----------
    left_boxes, right_boxes : sequence[BoxGeometry]
        Visible node primitives by component.
    left_routes, right_routes : sequence[Route]
        Visible route centerlines by component.
    scene : Scene
        Validated scene supplying route stroke widths.

    Returns
    -------
    float
        Minimum signed primitive clearance.
    """

    candidates: List[float] = []
    for left in left_boxes:
        for right in right_boxes:
            delta = torch.abs(left.center - right.center) - (left.half_extents + right.half_extents)
            outside = float(torch.linalg.vector_norm(torch.clamp(delta, min=0.0)))
            inside = min(max(float(delta[0]), float(delta[1])), 0.0)
            candidates.append(outside + inside)
    half_width = scene.style.route_stroke_width * scene.style.coordinate_scale / 2.0
    for box in left_boxes:
        for route in right_routes:
            for index in range(route.points.shape[0] - 1):
                candidates.append(
                    _box_segment_clearance(box, route.points[index], route.points[index + 1])
                    - half_width
                )
    for box in right_boxes:
        for route in left_routes:
            for index in range(route.points.shape[0] - 1):
                candidates.append(
                    _box_segment_clearance(box, route.points[index], route.points[index + 1])
                    - half_width
                )
    for left_route in left_routes:
        for right_route in right_routes:
            for left_index in range(left_route.points.shape[0] - 1):
                for right_index in range(right_route.points.shape[0] - 1):
                    candidates.append(
                        _segment_segment_distance(
                            left_route.points[left_index],
                            left_route.points[left_index + 1],
                            right_route.points[right_index],
                            right_route.points[right_index + 1],
                        )
                        - 2.0 * half_width
                    )
    return min(candidates)


def _raster_component_areas(
    boxes_by_component: Dict[int, List[BoxGeometry]],
    routes_by_component: Dict[int, List[Route]],
    frame: RobustFrame,
    scene: Scene,
) -> List[float]:
    """Rasterize component primitive unions on the frozen 512x512, 4x grid.

    Parameters
    ----------
    boxes_by_component : dict[int, list[BoxGeometry]]
        Node primitives by component.
    routes_by_component : dict[int, list[Route]]
        Route primitives by component.
    frame : RobustFrame
        Shared U21 robust measurement frame.
    scene : Scene
        Validated scene supplying stroke width.

    Returns
    -------
    list[float]
        Occupied primitive-union area per component.
    """

    supersampled = 512 * 4
    lower = frame.center - frame.half_extents
    upper = frame.center + frame.half_extents
    x = torch.linspace(float(lower[0]), float(upper[0]), supersampled + 1, dtype=torch.float64)
    y = torch.linspace(float(lower[1]), float(upper[1]), supersampled + 1, dtype=torch.float64)
    x_mid = (x[:-1] + x[1:]) / 2.0
    y_mid = (y[:-1] + y[1:]) / 2.0
    cell_area = float((x[1] - x[0]) * (y[1] - y[0]))
    occupied_counts = [0] * len(boxes_by_component)
    half_width = scene.style.route_stroke_width * scene.style.coordinate_scale / 2.0
    row_chunk = 16
    for row_start in range(0, supersampled, row_chunk):
        local_y = y_mid[row_start : row_start + row_chunk]
        grid_y, grid_x = torch.meshgrid(local_y, x_mid, indexing="ij")
        points = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1)
        for component_index in range(len(boxes_by_component)):
            occupied = torch.zeros(points.shape[0], dtype=torch.bool)
            for box in boxes_by_component[component_index]:
                occupied |= torch.all(torch.abs(points - box.center) <= box.half_extents, dim=1)
            for route in routes_by_component[component_index]:
                for index in range(route.points.shape[0] - 1):
                    start = route.points[index]
                    end = route.points[index + 1]
                    delta = end - start
                    denominator = float(torch.dot(delta, delta))
                    if denominator == 0.0:
                        distance = torch.linalg.vector_norm(points - start, dim=1)
                    else:
                        parameter = torch.clamp(
                            torch.sum((points - start) * delta, dim=1) / denominator,
                            0.0,
                            1.0,
                        )
                        projection = start + parameter[:, None] * delta
                        distance = torch.linalg.vector_norm(points - projection, dim=1)
                    occupied |= distance <= half_width
            occupied_counts[component_index] += int(torch.sum(occupied))
    return [count * cell_area for count in occupied_counts]


def _crossing_count(scene: Scene) -> int:
    """Count proper nonincident route crossings.

    Parameters
    ----------
    scene : Scene
        Validated routed scene.

    Returns
    -------
    int
        Exact segment-pair crossing count.
    """

    segments = route_segments(scene)
    count = 0
    for index, (route_a, _, start_a, end_a) in enumerate(segments):
        edge_a = scene.graph.edges[route_a]
        for route_b, _, start_b, end_b in segments[index + 1 :]:
            if route_a == route_b:
                continue
            edge_b = scene.graph.edges[route_b]
            if set(edge_a) & set(edge_b):
                continue
            count += int(proper_intersection(start_a, end_a, start_b, end_b))
    return count


def U41(scene: Scene) -> FacetResult:
    """Planarity and face quality. Frozen SHA-256: b26cdb05f3d09cda123b5fd6becbc8d7ebdaca931b89d0f2d2ee1e2d0f0f518a."""

    certificate = scene.graph.planarity_certificate
    if certificate is None:
        return na_result("NOT_INPUT_CERTIFIED_PLANAR")
    if not bool(certificate.get("planar", False)):
        return na_result("NOT_INPUT_CERTIFIED_PLANAR")
    face_opportunity = scene.edge_count - scene.node_count + len(components(scene))
    if face_opportunity <= 0:
        return na_result("NO_BOUNDED_FACE_OPPORTUNITY")
    arrangement = _arrangement_faces(scene)
    if arrangement is None:
        return invalid_result("collinear_route_overlap")
    faces, crossings = arrangement
    primitive_area = sum(float(4.0 * torch.prod(box.half_extents)) for box in scene.node_boxes)
    area_reference = (
        primitive_area / 0.10 * (1.0 + 0.5 * (len(components(scene)) - 1)) / face_opportunity
    )
    convex_sum = 0.0
    area_sum = 0.0
    raw_faces: List[Dict[str, float]] = []
    for face in faces:
        signed_area = _polygon_signed_area(face)
        area = abs(signed_area)
        hull_area = abs(_polygon_signed_area(_convex_hull(face)))
        convexity_defect = 1.0 - area / hull_area if hull_area > 0.0 else 1.0
        reflex_terms: List[float] = []
        for index, point in enumerate(face):
            incoming = point - face[index - 1]
            outgoing = face[(index + 1) % len(face)] - point
            denominator = float(
                torch.linalg.vector_norm(incoming) * torch.linalg.vector_norm(outgoing)
            )
            if denominator == 0.0:
                continue
            sine = float(incoming[0] * outgoing[1] - incoming[1] * outgoing[0]) / denominator
            reflex_terms.append(_stable_sigmoid(-sine / 0.03))
        reflex = sum(reflex_terms) / len(reflex_terms) if reflex_terms else 1.0
        combined = 0.70 * convexity_defect + 0.30 * reflex
        balance = abs(area - area_reference) / (area + area_reference)
        convex_sum += combined / face_opportunity
        area_sum += balance / face_opportunity
        raw_faces.append(
            {
                "area": area,
                "hull_area": hull_area,
                "convexity_defect": convexity_defect,
                "reflex_burden": reflex,
                "balance_burden": balance,
            }
        )
    convexity = 1.0 - math.exp(-convex_sum)
    balance = 1.0 - math.exp(-area_sum)
    values = {"U41.L_conv": convexity, "U41.L_area": balance}
    return mean_result(
        "U41",
        values,
        {
            "F0": face_opportunity,
            "crossing_pairs_k": crossings,
            "certificate_verified": True,
            "arrangement_face_count": len(faces),
            "a_ref": area_reference,
            "faces": tuple(raw_faces),
        },
    )


def _segment_intersection_parameters(
    start_a: torch.Tensor,
    end_a: torch.Tensor,
    start_b: torch.Tensor,
    end_b: torch.Tensor,
) -> Tuple[float, float] | None:
    """Return proper intersection parameters for two segments.

    Parameters
    ----------
    start_a, end_a, start_b, end_b : torch.Tensor
        Segment endpoints with shape ``[2]``.

    Returns
    -------
    tuple[float, float] or None
        Interior parameters on A and B, if the segments cross properly.
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
        return parameter_a, parameter_b
    return None


def _collinear_positive_overlap(
    start_a: torch.Tensor,
    end_a: torch.Tensor,
    start_b: torch.Tensor,
    end_b: torch.Tensor,
) -> bool:
    """Test whether collinear segments share positive interior length.

    Parameters
    ----------
    start_a, end_a, start_b, end_b : torch.Tensor
        Segment endpoints with shape ``[2]``.

    Returns
    -------
    bool
        Whether the collinear projections overlap by positive length.
    """

    direction = end_a - start_a
    cross_start = float(
        direction[0] * (start_b - start_a)[1] - direction[1] * (start_b - start_a)[0]
    )
    cross_end = float(direction[0] * (end_b - start_a)[1] - direction[1] * (end_b - start_a)[0])
    if cross_start != 0.0 or cross_end != 0.0:
        return False
    axis = 0 if abs(float(direction[0])) >= abs(float(direction[1])) else 1
    left_a, right_a = sorted((float(start_a[axis]), float(end_a[axis])))
    left_b, right_b = sorted((float(start_b[axis]), float(end_b[axis])))
    return min(right_a, right_b) > max(left_a, left_b)


def _arrangement_faces(scene: Scene) -> Tuple[List[List[torch.Tensor]], int] | None:
    """Construct bounded DCEL faces of the visible routed arrangement.

    Parameters
    ----------
    scene : Scene
        Input-certified planar graph drawing.

    Returns
    -------
    tuple[list[list[torch.Tensor]], int] or None
        Bounded face polygons and proper crossing count, or ``None`` for forbidden
        positive-length collinear route overlap.
    """

    segments = route_segments(scene)
    split_parameters: List[List[float]] = [[0.0, 1.0] for _ in segments]
    crossing_count = 0
    for left_index, (edge_left, _, start_left, end_left) in enumerate(segments):
        left_terminals = set(scene.graph.edges[edge_left])
        for right_index in range(left_index + 1, len(segments)):
            edge_right, _, start_right, end_right = segments[right_index]
            if edge_left == edge_right:
                continue
            if _collinear_positive_overlap(start_left, end_left, start_right, end_right):
                return None
            if left_terminals & set(scene.graph.edges[edge_right]):
                continue
            parameters = _segment_intersection_parameters(
                start_left, end_left, start_right, end_right
            )
            if parameters is None:
                continue
            split_parameters[left_index].append(parameters[0])
            split_parameters[right_index].append(parameters[1])
            crossing_count += 1

    point_ids: Dict[Tuple[float, float], int] = {}
    point_values: List[torch.Tensor] = []
    undirected_edges: Set[Tuple[int, int]] = set()

    def point_id(point: torch.Tensor) -> int:
        """Intern one exact arrangement coordinate.

        Parameters
        ----------
        point : torch.Tensor
            Float64 point with shape ``[2]``.

        Returns
        -------
        int
            Stable arrangement vertex id.
        """

        key = (float(point[0]), float(point[1]))
        if key not in point_ids:
            point_ids[key] = len(point_values)
            point_values.append(point)
        return point_ids[key]

    for segment, parameters in zip(segments, split_parameters):
        _, _, start, end = segment
        ordered = sorted(set(parameters))
        vertices = [point_id(start + parameter * (end - start)) for parameter in ordered]
        for left, right in zip(vertices, vertices[1:]):
            if left != right:
                undirected_edges.add(tuple(sorted((left, right))))
    adjacency: Dict[int, List[int]] = {index: [] for index in range(len(point_values))}
    for left, right in undirected_edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    for vertex, neighbors in adjacency.items():
        origin = point_values[vertex]
        neighbors.sort(
            key=lambda target: math.atan2(
                float(point_values[target][1] - origin[1]),
                float(point_values[target][0] - origin[0]),
            )
        )
    visited: Set[Tuple[int, int]] = set()
    cycles: List[List[torch.Tensor]] = []
    for left, right in sorted(
        (oriented for edge in undirected_edges for oriented in (edge, (edge[1], edge[0])))
    ):
        if (left, right) in visited:
            continue
        cycle_ids: List[int] = []
        start = (left, right)
        current = start
        while current not in visited:
            visited.add(current)
            previous, vertex = current
            cycle_ids.append(previous)
            neighbors = adjacency[vertex]
            reverse_index = neighbors.index(previous)
            next_vertex = neighbors[(reverse_index - 1) % len(neighbors)]
            current = (vertex, next_vertex)
        if current == start and len(cycle_ids) >= 3:
            polygon = [point_values[index] for index in cycle_ids]
            if _polygon_signed_area(polygon) > 0.0:
                cycles.append(polygon)
    return cycles, crossing_count


def _polygon_signed_area(points: Sequence[torch.Tensor]) -> float:
    """Return a polygon's signed shoelace area.

    Parameters
    ----------
    points : sequence[torch.Tensor]
        Ordered polygon vertices.

    Returns
    -------
    float
        Positive area for counter-clockwise order.
    """

    return 0.5 * sum(
        float(left[0] * right[1] - left[1] * right[0])
        for left, right in zip(points, tuple(points[1:]) + (points[0],))
    )


def _convex_hull(points: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """Return the canonical monotone-chain convex hull.

    Parameters
    ----------
    points : sequence[torch.Tensor]
        Polygon boundary vertices.

    Returns
    -------
    list[torch.Tensor]
        Counter-clockwise hull vertices.
    """

    unique = sorted({(float(point[0]), float(point[1])) for point in points})
    if len(unique) <= 2:
        return [torch.tensor(point, dtype=torch.float64) for point in unique]

    def cross(
        origin: Tuple[float, float], left: Tuple[float, float], right: Tuple[float, float]
    ) -> float:
        """Return the scalar turn of three tuple points.

        Parameters
        ----------
        origin, left, right : tuple[float, float]
            Ordered planar coordinates.

        Returns
        -------
        float
            Positive for a counter-clockwise turn.
        """

        return (left[0] - origin[0]) * (right[1] - origin[1]) - (left[1] - origin[1]) * (
            right[0] - origin[0]
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
    return [torch.tensor(point, dtype=torch.float64) for point in lower[:-1] + upper[:-1]]


@dataclass(frozen=True)
class _ChannelPrimitive:
    """One visible primitive carrying a declared categorical colour channel.

    Parameters
    ----------
    identifier : str
        Canonical primitive identifier.
    kind : str
        ``node`` or ``edge``.
    owner : int
        Canonical node or edge index.
    category : str
        Declared categorical value.
    color : tuple[float, float, float]
        Opaque sRGB colour in ``[0, 1]``.
    box : BoxGeometry or None
        Node geometry.
    route : Route or None
        Edge geometry.
    mass : float
        Positive input-owned object mass.
    rank : int
        Back-to-front z-order rank.
    """

    identifier: str
    kind: str
    owner: int
    category: str
    color: Tuple[float, float, float]
    box: BoxGeometry | None
    route: Route | None
    mass: float
    rank: int


def _srgb_luminance(color: Tuple[float, float, float]) -> float:
    """Return WCAG relative luminance for one sRGB triple.

    Parameters
    ----------
    color : tuple[float, float, float]
        Canonical sRGB channels in ``[0, 1]``.

    Returns
    -------
    float
        Relative luminance in ``[0, 1]``.
    """

    linear = [
        channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
        for channel in color
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _srgb_to_lab(color: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Convert canonical sRGB through D65 XYZ to CIELAB.

    Parameters
    ----------
    color : tuple[float, float, float]
        sRGB channels in ``[0, 1]``.

    Returns
    -------
    tuple[float, float, float]
        CIE ``L*, a*, b*`` coordinates.
    """

    linear = [
        channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
        for channel in color
    ]
    x = 0.4124564 * linear[0] + 0.3575761 * linear[1] + 0.1804375 * linear[2]
    y = 0.2126729 * linear[0] + 0.7151522 * linear[1] + 0.0721750 * linear[2]
    z = 0.0193339 * linear[0] + 0.1191920 * linear[1] + 0.9503041 * linear[2]

    def transform(value: float) -> float:
        """Apply the CIELAB reference-white transfer function.

        Parameters
        ----------
        value : float
            XYZ coordinate divided by its D65 reference value.

        Returns
        -------
        float
            Transformed coordinate.
        """

        delta = 6.0 / 29.0
        return value ** (1.0 / 3.0) if value > delta**3 else value / (3.0 * delta**2) + 4.0 / 29.0

    fx = transform(x / 0.95047)
    fy = transform(y)
    fz = transform(z / 1.08883)
    return 116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)


def _ciede2000(left: Tuple[float, float, float], right: Tuple[float, float, float]) -> float:
    """Return the CIEDE2000 colour difference for two Lab triples.

    Parameters
    ----------
    left, right : tuple[float, float, float]
        CIELAB coordinates under the same D65 reference white.

    Returns
    -------
    float
        Nonnegative standard CIEDE2000 difference with unit parametric factors.
    """

    light_left, a_left, b_left = left
    light_right, a_right, b_right = right
    chroma_left = math.hypot(a_left, b_left)
    chroma_right = math.hypot(a_right, b_right)
    chroma_bar = (chroma_left + chroma_right) / 2.0
    power = chroma_bar**7
    g_value = 0.5 * (1.0 - math.sqrt(power / (power + 25.0**7)))
    adjusted_a_left = (1.0 + g_value) * a_left
    adjusted_a_right = (1.0 + g_value) * a_right
    adjusted_chroma_left = math.hypot(adjusted_a_left, b_left)
    adjusted_chroma_right = math.hypot(adjusted_a_right, b_right)

    def hue(adjusted_a: float, b_value: float) -> float:
        """Return a canonical CIE hue angle in degrees.

        Parameters
        ----------
        adjusted_a, b_value : float
            Adjusted CIELAB chromatic coordinates.

        Returns
        -------
        float
            Hue angle in ``[0, 360)``.
        """

        return math.degrees(math.atan2(b_value, adjusted_a)) % 360.0

    hue_left = hue(adjusted_a_left, b_left)
    hue_right = hue(adjusted_a_right, b_right)
    delta_light = light_right - light_left
    delta_chroma = adjusted_chroma_right - adjusted_chroma_left
    if adjusted_chroma_left * adjusted_chroma_right == 0.0:
        delta_hue = 0.0
    else:
        raw_hue = hue_right - hue_left
        if raw_hue > 180.0:
            raw_hue -= 360.0
        elif raw_hue < -180.0:
            raw_hue += 360.0
        delta_hue = raw_hue
    delta_h = (
        2.0
        * math.sqrt(adjusted_chroma_left * adjusted_chroma_right)
        * math.sin(math.radians(delta_hue / 2.0))
    )
    mean_light = (light_left + light_right) / 2.0
    mean_chroma = (adjusted_chroma_left + adjusted_chroma_right) / 2.0
    if adjusted_chroma_left * adjusted_chroma_right == 0.0:
        mean_hue = hue_left + hue_right
    elif abs(hue_left - hue_right) <= 180.0:
        mean_hue = (hue_left + hue_right) / 2.0
    elif hue_left + hue_right < 360.0:
        mean_hue = (hue_left + hue_right + 360.0) / 2.0
    else:
        mean_hue = (hue_left + hue_right - 360.0) / 2.0
    t_value = (
        1.0
        - 0.17 * math.cos(math.radians(mean_hue - 30.0))
        + 0.24 * math.cos(math.radians(2.0 * mean_hue))
        + 0.32 * math.cos(math.radians(3.0 * mean_hue + 6.0))
        - 0.20 * math.cos(math.radians(4.0 * mean_hue - 63.0))
    )
    light_scale = 1.0 + 0.015 * (mean_light - 50.0) ** 2 / math.sqrt(
        20.0 + (mean_light - 50.0) ** 2
    )
    chroma_scale = 1.0 + 0.045 * mean_chroma
    hue_scale = 1.0 + 0.015 * mean_chroma * t_value
    rotation_angle = 30.0 * math.exp(-(((mean_hue - 275.0) / 25.0) ** 2))
    chroma_power = mean_chroma**7
    rotation = (
        -2.0
        * math.sqrt(chroma_power / (chroma_power + 25.0**7))
        * math.sin(math.radians(2.0 * rotation_angle))
    )
    light_term = delta_light / light_scale
    chroma_term = delta_chroma / chroma_scale
    hue_term = delta_h / hue_scale
    return math.sqrt(
        max(
            0.0,
            light_term**2 + chroma_term**2 + hue_term**2 + rotation * chroma_term * hue_term,
        )
    )


def _channel_rank(scene: Scene, identifier: str, group: str, owner: int) -> int:
    """Resolve an individual or group z-order declaration deterministically.

    Parameters
    ----------
    scene : Scene
        Validated scene.
    identifier : str
        Canonical primitive id such as ``node:3``.
    group : str
        Group token, ``nodes`` or ``routes``.
    owner : int
        Canonical owner index used as a stable within-group tie-break.

    Returns
    -------
    int
        Back-to-front integer rank.
    """

    if identifier in scene.z_order:
        return 2 * len(scene.z_order) * scene.z_order.index(identifier) + owner
    if group in scene.z_order:
        return 2 * len(scene.z_order) * scene.z_order.index(group) + owner
    return owner


def _channel_primitives(scene: Scene) -> List[_ChannelPrimitive]:
    """Derive all primitives carrying applicable declared colour channels.

    Parameters
    ----------
    scene : Scene
        Validated channel-declaring scene.

    Returns
    -------
    list[_ChannelPrimitive]
        Canonically ordered primitive population.
    """

    result: List[_ChannelPrimitive] = []
    routes = {route.edge_index: route for route in resolved_routes(scene)}
    node_masses = scene.graph.node_masses or tuple(1.0 for _ in range(scene.node_count))
    for declaration in scene.style.channel_set:
        attributes = (
            scene.graph.node_attributes
            if declaration.primitive_kind == "node"
            else scene.graph.edge_attributes
        )
        values = attributes.get(declaration.attribute)
        if values is None:
            continue
        for owner, category in enumerate(values):
            color = declaration.value_map[category]
            if declaration.primitive_kind == "node":
                identifier = f"node:{owner}"
                result.append(
                    _ChannelPrimitive(
                        identifier,
                        "node",
                        owner,
                        category,
                        color,
                        scene.node_boxes[owner],
                        None,
                        float(node_masses[owner]),
                        _channel_rank(scene, identifier, "nodes", owner),
                    )
                )
            else:
                identifier = f"edge:{owner}"
                result.append(
                    _ChannelPrimitive(
                        identifier,
                        "edge",
                        owner,
                        category,
                        color,
                        None,
                        routes[owner],
                        1.0,
                        _channel_rank(scene, identifier, "routes", owner),
                    )
                )
    return result


def _primitive_stroke_width(primitive: _ChannelPrimitive, scene: Scene) -> float:
    """Return the corpus-owned visible stroke width of one primitive.

    Parameters
    ----------
    primitive : _ChannelPrimitive
        Channel-carrying node or route.
    scene : Scene
        Validated scene supplying per-edge and default widths.

    Returns
    -------
    float
        Positive scene-coordinate stroke width, or zero for a node fill.
    """

    if primitive.route is None:
        return 0.0
    if scene.style.edge_stroke_widths:
        return scene.style.edge_stroke_widths[primitive.owner]
    return scene.style.route_stroke_width * scene.style.coordinate_scale


def _primitive_polygons(primitive: _ChannelPrimitive, scene: Scene) -> List[torch.Tensor]:
    """Return convex polygons whose union is one opaque primitive.

    Parameters
    ----------
    primitive : _ChannelPrimitive
        Channel-carrying node or route.
    scene : Scene
        Validated scene supplying route flattening parameters.

    Returns
    -------
    list[torch.Tensor]
        One box polygon or the route's flattened segment capsules.
    """

    if primitive.box is not None:
        return [_box_polygon(primitive.box)]
    assert primitive.route is not None
    width = _primitive_stroke_width(primitive, scene)
    return [
        _capsule_polygon(start, end, width, scene.style.flattening_tolerance)
        for start, end in zip(primitive.route.points[:-1], primitive.route.points[1:])
        if float(torch.linalg.vector_norm(end - start)) > 0.0
    ]


def _cross_2d(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return the scalar planar cross product.

    Parameters
    ----------
    left, right : torch.Tensor
        Vectors with shape ``[2]``.

    Returns
    -------
    float
        Signed scalar cross product.
    """

    return float(left[0] * right[1] - left[1] * right[0])


def _convex_polygon_intersection(subject: torch.Tensor, clip: torch.Tensor) -> torch.Tensor:
    """Intersect two counter-clockwise convex polygons exactly after flattening.

    Parameters
    ----------
    subject, clip : torch.Tensor
        Convex polygon vertices with shapes ``[P, 2]`` and ``[Q, 2]``.

    Returns
    -------
    torch.Tensor
        Intersection polygon, possibly empty.
    """

    output = [point for point in subject]
    for clip_start, clip_end in zip(clip, torch.roll(clip, shifts=-1, dims=0)):
        if not output:
            break
        input_points = output
        output = []
        for start, end in zip(input_points[-1:] + input_points[:-1], input_points):
            start_side = _cross_2d(clip_end - clip_start, start - clip_start)
            end_side = _cross_2d(clip_end - clip_start, end - clip_start)
            start_inside = start_side >= 0.0
            end_inside = end_side >= 0.0
            if start_inside != end_inside:
                subject_direction = end - start
                clip_direction = clip_end - clip_start
                denominator = _cross_2d(subject_direction, clip_direction)
                if denominator != 0.0:
                    parameter = _cross_2d(clip_start - start, clip_direction) / denominator
                    output.append(start + parameter * subject_direction)
            if end_inside:
                output.append(end)
    return torch.stack(output) if len(output) >= 3 else torch.empty((0, 2), dtype=torch.float64)


def _polygon_group_intersections(
    left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]
) -> List[torch.Tensor]:
    """Return all nonempty convex pieces in a polygon-union intersection.

    Parameters
    ----------
    left, right : sequence[torch.Tensor]
        Convex polygon groups.

    Returns
    -------
    list[torch.Tensor]
        Convex intersection pieces whose union is the group intersection.
    """

    pieces: List[torch.Tensor] = []
    for left_polygon in left:
        for right_polygon in right:
            piece = _convex_polygon_intersection(left_polygon, right_polygon)
            if piece.shape[0] >= 3:
                pieces.append(piece)
    return pieces


def _box_boundary_band(box: BoxGeometry, width: float) -> List[torch.Tensor]:
    """Return four rectangles partitioning a box's internal boundary band.

    Parameters
    ----------
    box : BoxGeometry
        Foreground node box.
    width : float
        Positive inward band width.

    Returns
    -------
    list[torch.Tensor]
        Non-overlapping boundary-band rectangles.
    """

    low = box.center - box.half_extents
    high = box.center + box.half_extents
    band = min(width, float(torch.min(box.half_extents)))

    def rectangle(left: float, right: float, bottom: float, top: float) -> torch.Tensor:
        """Build one counter-clockwise rectangle.

        Parameters
        ----------
        left, right, bottom, top : float
            Rectangle bounds.

        Returns
        -------
        torch.Tensor
            Four vertices with shape ``[4, 2]``.
        """

        return torch.tensor(
            ((left, bottom), (right, bottom), (right, top), (left, top)),
            dtype=torch.float64,
        )

    left = float(low[0])
    right = float(high[0])
    bottom = float(low[1])
    top = float(high[1])
    return [
        rectangle(left, right, bottom, bottom + band),
        rectangle(left, right, top - band, top),
        rectangle(left, left + band, bottom + band, top - band),
        rectangle(right - band, right, bottom + band, top - band),
    ]


def _primitive_boundary_band(primitive: _ChannelPrimitive, scene: Scene) -> List[torch.Tensor]:
    """Return the exact flattened boundary-band polygon group for a primitive.

    Parameters
    ----------
    primitive : _ChannelPrimitive
        Foreground primitive.
    scene : Scene
        Validated scene supplying widths and intrinsic unit.

    Returns
    -------
    list[torch.Tensor]
        Boundary-band polygons.
    """

    width = max(_primitive_stroke_width(primitive, scene), 0.05 * scene.intrinsic_unit)
    if primitive.box is not None:
        return _box_boundary_band(primitive.box, width)
    return _primitive_polygons(primitive, scene)


def _primitive_clearance(left: _ChannelPrimitive, right: _ChannelPrimitive, scene: Scene) -> float:
    """Return signed visible-geometry clearance between two channel primitives.

    Parameters
    ----------
    left, right : _ChannelPrimitive
        Primitive pair.
    scene : Scene
        Validated scene supplying route stroke width.

    Returns
    -------
    float
        Signed clearance in scene coordinates.
    """

    if left.box is not None and right.box is not None:
        delta = torch.abs(left.box.center - right.box.center) - (
            left.box.half_extents + right.box.half_extents
        )
        return float(torch.linalg.vector_norm(torch.clamp(delta, min=0.0))) + min(
            max(float(delta[0]), float(delta[1])), 0.0
        )
    if left.box is not None and right.route is not None:
        return (
            min(
                _box_segment_clearance(left.box, start, end)
                for start, end in zip(right.route.points[:-1], right.route.points[1:])
            )
            - _primitive_stroke_width(right, scene) / 2.0
        )
    if right.box is not None and left.route is not None:
        return (
            min(
                _box_segment_clearance(right.box, start, end)
                for start, end in zip(left.route.points[:-1], left.route.points[1:])
            )
            - _primitive_stroke_width(left, scene) / 2.0
        )
    assert left.route is not None and right.route is not None
    return (
        min(
            _segment_segment_distance(left_start, left_end, right_start, right_end)
            for left_start, left_end in zip(left.route.points[:-1], left.route.points[1:])
            for right_start, right_end in zip(right.route.points[:-1], right.route.points[1:])
        )
        - (_primitive_stroke_width(left, scene) + _primitive_stroke_width(right, scene)) / 2.0
    )


def _primitive_visibility(
    primitive: _ChannelPrimitive,
    population: Sequence[_ChannelPrimitive],
    scene: Scene,
) -> float:
    """Return exact opaque visible-area fraction after z-order compositing.

    Parameters
    ----------
    primitive : _ChannelPrimitive
        Primitive being measured.
    population : sequence[_ChannelPrimitive]
        Full declared channel population.
    scene : Scene
        Validated scene supplying flattened primitive geometry.

    Returns
    -------
    float
        Visible fraction in ``[0, 1]``.
    """

    geometry = _primitive_polygons(primitive, scene)
    area = _polygon_union_area(geometry)
    occluded: List[torch.Tensor] = []
    for other in population:
        if other.rank > primitive.rank:
            occluded.extend(
                _polygon_group_intersections(geometry, _primitive_polygons(other, scene))
            )
    occluded_area = _polygon_union_area(occluded)
    return max(0.0, min(1.0, 1.0 - occluded_area / area))


def _effective_backdrop(
    primitive: _ChannelPrimitive,
    population: Sequence[_ChannelPrimitive],
    scene: Scene,
) -> Tuple[float, float, float]:
    """Return the boundary-band area-weighted opaque backdrop colour.

    Parameters
    ----------
    primitive : _ChannelPrimitive
        Foreground primitive.
    population : sequence[_ChannelPrimitive]
        Full declared primitive population.
    scene : Scene
        Scene supplying geometry and canvas colour.

    Returns
    -------
    tuple[float, float, float]
        Effective opaque sRGB backdrop.
    """

    band = _primitive_boundary_band(primitive, scene)
    band_area = _polygon_union_area(band)
    covered: List[torch.Tensor] = []
    weighted = [0.0, 0.0, 0.0]
    covered_area = 0.0
    candidates = sorted(
        (other for other in population if other.rank < primitive.rank),
        key=lambda item: item.rank,
        reverse=True,
    )
    for other in candidates:
        pieces = _polygon_group_intersections(band, _primitive_polygons(other, scene))
        if not pieces:
            continue
        new_area = _polygon_union_area(covered + pieces)
        contribution = max(0.0, new_area - covered_area)
        for channel in range(3):
            weighted[channel] += contribution * other.color[channel]
        covered.extend(pieces)
        covered_area = new_area
        if covered_area >= band_area:
            break
    canvas_area = max(0.0, band_area - covered_area)
    for channel in range(3):
        weighted[channel] += canvas_area * scene.style.canvas_background[channel]
    return tuple(value / band_area for value in weighted)


_CVD_MATRICES: Tuple[Tuple[Tuple[float, float, float], ...], ...] = (
    (
        (0.152286, 1.052583, -0.204868),
        (0.114503, 0.786281, 0.099216),
        (-0.003882, -0.048116, 1.051998),
    ),
    (
        (0.367322, 0.860646, -0.227968),
        (0.280085, 0.672501, 0.047413),
        (-0.011820, 0.042940, 0.968881),
    ),
    (
        (1.255528, -0.076749, -0.178779),
        (-0.078411, 0.930809, 0.147602),
        (0.004733, 0.691367, 0.303900),
    ),
)


def _simulate_cvd(
    color: Tuple[float, float, float],
    matrix: Tuple[Tuple[float, float, float], ...],
) -> Tuple[float, float, float]:
    """Apply one frozen dichromacy matrix to an sRGB triple.

    Parameters
    ----------
    color : tuple[float, float, float]
        Input sRGB colour.
    matrix : tuple[tuple[float, float, float], ...]
        Frozen 3-by-3 simulation matrix.

    Returns
    -------
    tuple[float, float, float]
        Clamped simulated sRGB colour.
    """

    return tuple(
        min(1.0, max(0.0, sum(coefficient * channel for coefficient, channel in zip(row, color))))
        for row in matrix
    )


def _soft_min(values: Sequence[float], temperature: float) -> float:
    """Return the stable mean-form soft minimum.

    Parameters
    ----------
    values : sequence[float]
        Nonempty scalar population.
    temperature : float
        Positive temperature.

    Returns
    -------
    float
        Smooth minimum with singleton identity.
    """

    minimum = min(values)
    return minimum - temperature * math.log(
        sum(math.exp(-(value - minimum) / temperature) for value in values) / len(values)
    )


def U42(scene: Scene) -> FacetResult:
    """Encoding fidelity & contrast. Frozen SHA-256: 7ee672a532e6032cab87ca1ffe35156192c1a19edf384a4db5e33803d389c94e."""

    if not scene.style.channel_set:
        return na_result("no_declared_channels")
    population = _channel_primitives(scene)
    if not population:
        return na_result("channel_not_declared")
    visibility = {
        primitive.identifier: _primitive_visibility(primitive, population, scene)
        for primitive in population
    }
    contrast_losses: List[float] = []
    contrast_weights: List[float] = []
    zero_contrast = 0
    for primitive in population:
        foreground = _srgb_luminance(primitive.color)
        backdrop = _srgb_luminance(_effective_backdrop(primitive, population, scene))
        ratio = (max(foreground, backdrop) + 0.05) / (min(foreground, backdrop) + 0.05)
        if ratio == 1.0:
            zero_contrast += 1
        target = 4.5
        if primitive.box is not None:
            area = float(4.0 * torch.prod(primitive.box.half_extents))
            if area > 24.0 * scene.intrinsic_unit * scene.intrinsic_unit:
                target = 3.0
        loss = 1.0 - float(
            smoothstep(torch.tensor((ratio - 1.0) / (target - 1.0), dtype=torch.float64))
        )
        contrast_losses.append(visibility[primitive.identifier] * loss)
        contrast_weights.append(primitive.mass)
    distinguishability: List[float] = []
    robustness: List[float] = []
    pair_weights: List[float] = []
    proximate_count = 0
    for left_index, left in enumerate(population):
        for right in population[left_index + 1 :]:
            if left.category == right.category:
                continue
            clearance = max(0.0, _primitive_clearance(left, right, scene))
            proximity = 1.0 - float(
                smoothstep(
                    torch.tensor(clearance / (2.0 * scene.intrinsic_unit), dtype=torch.float64)
                )
            )
            if proximity <= 0.0:
                continue
            proximate_count += 1
            visible = proximity * min(visibility[left.identifier], visibility[right.identifier])
            delta = _ciede2000(_srgb_to_lab(left.color), _srgb_to_lab(right.color))
            distinguishability.append(
                visible * (1.0 - float(smoothstep(torch.tensor(delta / 20.0, dtype=torch.float64))))
            )
            simulated = [
                _ciede2000(
                    _srgb_to_lab(_simulate_cvd(left.color, matrix)),
                    _srgb_to_lab(_simulate_cvd(right.color, matrix)),
                )
                for matrix in _CVD_MATRICES
            ]
            soft_delta = _soft_min(simulated, 0.05)
            robustness.append(
                visible
                * (1.0 - float(smoothstep(torch.tensor(soft_delta / 20.0, dtype=torch.float64))))
            )
            pair_weights.append((left.mass + right.mass) / 2.0)
    values: Dict[str, float] = {
        "U42.i": global_blend(contrast_losses, contrast_weights),
    }
    if distinguishability:
        values["U42.ii"] = global_blend(distinguishability, pair_weights)
        values["U42.iv"] = global_blend(robustness, pair_weights)
    return mean_result(
        "U42",
        values,
        {
            "primitive_count": len(population),
            "proximate_different_category_pairs": proximate_count,
            "zero_contrast_count": zero_contrast,
            "fully_occluded_count": sum(value == 0.0 for value in visibility.values()),
            "visibility": visibility,
        },
    )
