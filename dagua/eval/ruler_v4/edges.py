"""Crossing, routing, angular, ambiguity, and edge-label facet family."""

# Contract identity docstrings intentionally keep each title and digest on one line.
# ruff: noqa: E501

from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Optional, Sequence, Tuple

import torch

from dagua.eval.ruler_v4._util import (
    aabb_pair,
    blend_with_weights,
    global_blend,
    mean_result,
    proper_intersection,
    resolved_ranks,
    resolved_routes,
    route_segments,
    smoothstep,
    snap_unit,
)
from dagua.eval.ruler_v4.scene import (
    BoxGeometry,
    FacetResult,
    ResultState,
    Route,
    Scene,
    na_result,
    value_result,
)

_ANGULAR_ZERO_ENVELOPE = 1e-12

_U07_WORKED_EXAMPLE_GAMMA = 1.0
_U07_WORKED_EXAMPLE_LAMBDA_T = 0.5
_U11_TERMINAL_DISK_SIDES = 16
_U11_TERMINAL_CLEAR_RADIUS = 0.5


@dataclass(frozen=True)
class _CrossingEvent:
    """One exact transversal or collinear U07 crossing event.

    Parameters
    ----------
    edge_a, edge_b : int
        Declared edge indices.
    point : torch.Tensor
        Event point with shape ``[2]``.
    angle : float
        Acute crossing angle in radians.
    proximity : float
        Nearest graph-terminal distance in intrinsic-unit multiples.
    pair_multiplicity : int
        Total event count for the unordered edge pair.
    density : float
        Local crossing-density argument.
    severity : float
        Frozen four-component severity.
    """

    edge_a: int
    edge_b: int
    point: torch.Tensor
    angle: float
    proximity: float
    pair_multiplicity: int
    density: float
    severity: float


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


def _oriented_angle(first: torch.Tensor, second: torch.Tensor) -> float:
    """Return the oriented angle between two direction vectors.

    Unlike :func:`_segment_angle`, no absolute value is taken, so
    anti-parallel vectors read ``pi`` rather than ``0``. U11 sec 5 (v)
    compares terminal tangents that point away from their shared node:
    coincident tangents (angle 0) are the merge-identity limit while
    opposite tangents (angle pi) are maximally distinguishable.

    Parameters
    ----------
    first, second : torch.Tensor
        Two-dimensional vectors.

    Returns
    -------
    float
        Angle in radians in ``[0, pi]``.
    """

    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    if float(denominator) == 0.0:
        return 0.0
    cosine = min(1.0, max(-1.0, float(torch.dot(first, second) / denominator)))
    return math.acos(cosine)


def _segment_event_point(
    start_a: torch.Tensor,
    end_a: torch.Tensor,
    start_b: torch.Tensor,
    end_b: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, float]]:
    """Return a proper crossing point and acute angle when one event exists.

    Parameters
    ----------
    start_a, end_a, start_b, end_b : torch.Tensor
        Segment endpoints with shape ``[2]``.

    Returns
    -------
    tuple[torch.Tensor, float] or None
        Event point and acute angle. Positive-length collinear overlap produces
        its midpoint with angle zero.
    """

    direction_a = end_a - start_a
    direction_b = end_b - start_b
    cross = float(direction_a[0] * direction_b[1] - direction_a[1] * direction_b[0])
    if cross != 0.0:
        offset = start_b - start_a
        parameter_a = float(offset[0] * direction_b[1] - offset[1] * direction_b[0]) / cross
        parameter_b = float(offset[0] * direction_a[1] - offset[1] * direction_a[0]) / cross
        if 0.0 < parameter_a < 1.0 and 0.0 < parameter_b < 1.0:
            return start_a + parameter_a * direction_a, _segment_angle(direction_a, direction_b)
        return None
    if (
        float(direction_a[0] * (start_b - start_a)[1] - direction_a[1] * (start_b - start_a)[0])
        != 0.0
    ):
        return None
    length_squared = float(torch.dot(direction_a, direction_a))
    if length_squared == 0.0:
        return None
    left = float(torch.dot(start_b - start_a, direction_a)) / length_squared
    right = float(torch.dot(end_b - start_a, direction_a)) / length_squared
    overlap_start = max(0.0, min(left, right))
    overlap_end = min(1.0, max(left, right))
    if overlap_end <= overlap_start:
        return None
    midpoint = start_a + ((overlap_start + overlap_end) / 2.0) * direction_a
    return midpoint, 0.0


def _crossing_events(scene: Scene, gamma: float) -> List[_CrossingEvent]:
    """Compute U07 crossing events and frozen four-component severities.

    Parameters
    ----------
    scene : Scene
        Validated route scene.
    gamma : float
        Positive fitted crossing-severity scale.

    Returns
    -------
    list[_CrossingEvent]
        One event per inter-edge crossing, including adjacent edge pairs.
    """

    segments = route_segments(scene)
    provisional: List[Tuple[int, int, torch.Tensor, float, float]] = []
    for index, (edge_a, _, start_a, end_a) in enumerate(segments):
        terminals_a = scene.graph.edges[edge_a]
        for edge_b, _, start_b, end_b in segments[index + 1 :]:
            if edge_a == edge_b:
                continue
            event = _segment_event_point(start_a, end_a, start_b, end_b)
            if event is None:
                continue
            point, angle = event
            terminals_b = scene.graph.edges[edge_b]
            terminal_points = scene.positions[list((*terminals_a, *terminals_b))]
            proximity = float(torch.min(torch.linalg.vector_norm(terminal_points - point, dim=1)))
            provisional.append((min(edge_a, edge_b), max(edge_a, edge_b), point, angle, proximity))
    multiplicities = Counter((left, right) for left, right, _, _, _ in provisional)
    results = []
    for event_index, (edge_a, edge_b, point, angle, proximity) in enumerate(provisional):
        sine_squared = math.sin(angle) ** 2
        density = 0.0
        for other_index, (_, _, other_point, other_angle, _) in enumerate(provisional):
            if event_index == other_index:
                continue
            normalized_squared = (
                float(torch.sum((point - other_point) ** 2).item())
                / (6.0 * scene.intrinsic_unit) ** 2
            )
            density += math.sin(other_angle) ** 2 * max(0.0, 1.0 - normalized_squared) ** 2
        multiplicity = multiplicities[(edge_a, edge_b)]
        angle_term = math.cos(angle) ** 2
        proximity_ratio = proximity / scene.intrinsic_unit
        proximity_term = max(0.0, 1.0 - proximity_ratio**2 / 16.0) ** 2
        repeat_term = (multiplicity - 1.0) / multiplicity
        density_term = density / (density + 3.0)
        severity = gamma * (
            0.50 * angle_term
            + sine_squared * (0.20 * proximity_term + 0.15 * density_term)
            + 0.15 * repeat_term
        )
        results.append(
            _CrossingEvent(
                edge_a,
                edge_b,
                point,
                angle,
                proximity_ratio,
                multiplicity,
                density,
                severity,
            )
        )
    return results


def U07(
    scene: Scene,
    gamma: float = _U07_WORKED_EXAMPLE_GAMMA,
    lambda_T: float = _U07_WORKED_EXAMPLE_LAMBDA_T,
) -> FacetResult:
    """NORMATIVE CONTRACT: Crossing slot. Frozen SHA-256: 63526aa6824dfa5180234ba97086725621e8b9e8aa3c40713c85a7c4a86caf2b.

    Parameters
    ----------
    scene : Scene
        Validated canonical scene.
    gamma : float
        Fitted severity scale in the contract range ``(0, 3]``. The default is
        the phase-4 worked-example value and is not a P5 fitted selection.
    lambda_T : float
        Fitted excess-severity weight in ``[0, 1]``. The default is the phase-4
        worked-example value and is not a P5 fitted selection.

    Returns
    -------
    FacetResult
        Crossing defect and its contract rows.

    Raises
    ------
    ValueError
        If a fitted parameter lies outside its frozen contract range.
    """

    if not math.isfinite(gamma) or not 0.0 < gamma <= 3.0:
        raise ValueError("U07 gamma must be finite and lie in (0, 3]")
    if not math.isfinite(lambda_T) or not 0.0 <= lambda_T <= 1.0:
        raise ValueError("U07 lambda_T must be finite and lie in [0, 1]")

    events = _crossing_events(scene, gamma)
    eligible = 0
    for left in range(scene.edge_count):
        for right in range(left + 1, scene.edge_count):
            if not set(scene.graph.edges[left]) & set(scene.graph.edges[right]):
                eligible += 1
    base_raw = sum(1.0 + event.severity for event in events)
    threshold = 0.5 * gamma
    tail_raw = sum(max(0.0, event.severity - threshold) for event in events)
    opportunity = eligible + 1
    normalized = (base_raw + lambda_T * tail_raw) / opportunity
    defect = normalized / (normalized + 0.25) if normalized > 0.0 else 0.0
    base_x = base_raw / opportunity
    tail_x = lambda_T * tail_raw / opportunity
    base = base_x / (base_x + 0.25) if base_x > 0.0 else 0.0
    tail = tail_x / (tail_x + 0.25) if tail_x > 0.0 else 0.0
    return value_result(
        defect,
        {"U7.base": base, "U7.tail": tail},
        {
            "crossing_count": len(events),
            "eligible_pairs": eligible,
            "opportunity_guarded": opportunity,
            "base_raw": base_raw,
            "tail_raw": tail_raw,
            "gamma": gamma,
            "lambda_T": lambda_T,
            "events": tuple(
                {
                    "edge_pair": (event.edge_a, event.edge_b),
                    "point": event.point.tolist(),
                    "angle": event.angle,
                    "proximity": event.proximity,
                    "multiplicity": event.pair_multiplicity,
                    "density": event.density,
                    "severity": event.severity,
                }
                for event in events
            ),
        },
    )


def U08(scene: Scene) -> FacetResult:
    """Angular resolution at nodes. Frozen SHA-256: bcc8037fd0f620ce8e43fa7ba68acd711601e12355dd48a2741469dc997a4ebf."""

    directions = _incident_secants(scene)
    input_degrees = [0] * scene.node_count
    for source, target in scene.graph.edges:
        if source == target:
            input_degrees[source] += 2
        else:
            input_degrees[source] += 1
            input_degrees[target] += 1
    defects: List[float] = []
    for node, degree in enumerate(input_degrees):
        if degree < 3:
            continue
        records = sorted(directions[node], key=lambda item: (item[0], item[1]))
        effective_count = sum(confidence for _, _, confidence in records)
        fade = float(smoothstep(torch.tensor(effective_count - 2.0, dtype=torch.float64)))
        if fade == 0.0:
            defects.append(0.0)
            continue
        fair_share = 2.0 * math.pi / effective_count
        weighted_exponentials = 0.0
        total_weight = 0.0
        count = len(records)
        for left in range(count):
            for step in range(1, count):
                right = (left + step) % count
                between = [(left + offset) % count for offset in range(1, step)]
                weight = records[left][2] * records[right][2]
                for middle in between:
                    weight *= 1.0 - records[middle][2]
                if weight == 0.0:
                    continue
                delta = (records[right][0] - records[left][0]) % (2.0 * math.pi)
                pair_defect = max(0.0, 1.0 - delta / fair_share)
                if pair_defect <= _ANGULAR_ZERO_ENVELOPE:
                    pair_defect = 0.0
                weighted_exponentials += weight * math.exp(pair_defect / 0.1)
                total_weight += weight
        if total_weight == 0.0:
            defects.append(0.0)
            continue
        node_defect = 0.1 * math.log(weighted_exponentials / total_weight)
        # The log-sum-exp mean is analytically in [0, 1] for pair defects in
        # [0, 1]; the normalized mean can carry one ULP of dust either side.
        defects.append(snap_unit(fade * node_defect))
    if not defects:
        return na_result("no_high_degree_nodes")
    defect = global_blend(defects)
    return value_result(defect, {"U08.headline": defect}, {"node_count": len(defects)})


def _point_at_arc_fraction(points: torch.Tensor, fraction: float) -> torch.Tensor:
    """Interpolate a polyline at one fraction of total arc length.

    Parameters
    ----------
    points : torch.Tensor
        Polyline vertices with shape ``[P, 2]``.
    fraction : float
        Arc-length fraction in ``[0, 1]``.

    Returns
    -------
    torch.Tensor
        Interpolated point with shape ``[2]``.
    """

    lengths = torch.linalg.vector_norm(points[1:] - points[:-1], dim=1)
    total = float(torch.sum(lengths).item())
    if total == 0.0:
        return points[0]
    target = min(1.0, max(0.0, fraction)) * total
    cumulative = 0.0
    for index, length_tensor in enumerate(lengths):
        length = float(length_tensor)
        if cumulative + length >= target:
            local = (target - cumulative) / length if length > 0.0 else 0.0
            return points[index] + local * (points[index + 1] - points[index])
        cumulative += length
    return points[-1]


def _incident_secants(scene: Scene) -> DefaultDict[int, List[Tuple[float, int, float]]]:
    """Extract U08 departure angles and tangent confidences at every endpoint.

    Parameters
    ----------
    scene : Scene
        Validated graph scene.

    Returns
    -------
    defaultdict[int, list[tuple[float, int, float]]]
        Node to ``(angle, canonical endpoint key, confidence)`` records.
    """

    records: DefaultDict[int, List[Tuple[float, int, float]]] = defaultdict(list)
    for route in resolved_routes(scene):
        source, target = scene.graph.edges[route.edge_index]
        lengths = torch.linalg.vector_norm(route.points[1:] - route.points[:-1], dim=1)
        total = float(torch.sum(lengths).item())
        confidence = float(
            smoothstep(torch.tensor(total / (0.1 * scene.intrinsic_unit), dtype=torch.float64))
        )
        if confidence == 0.0:
            if source == target:
                records[source].extend(
                    ((0.0, 2 * route.edge_index, 0.0), (0.0, 2 * route.edge_index + 1, 0.0))
                )
            else:
                records[source].append((0.0, 2 * route.edge_index, 0.0))
                records[target].append((0.0, 2 * route.edge_index + 1, 0.0))
            continue
        source_vector = _point_at_arc_fraction(route.points, 0.10) - scene.positions[source]
        target_vector = _point_at_arc_fraction(route.points, 0.90) - scene.positions[target]
        source_angle = math.atan2(float(source_vector[1]), float(source_vector[0])) % (
            2.0 * math.pi
        )
        target_angle = math.atan2(float(target_vector[1]), float(target_vector[0])) % (
            2.0 * math.pi
        )
        records[source].append((source_angle, 2 * route.edge_index, confidence))
        records[target].append((target_angle, 2 * route.edge_index + 1, confidence))
    return records


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


def _unit_interval_root(constant: float, slope: float) -> List[float]:
    """Return an affine root when it lies strictly inside the unit interval.

    Parameters
    ----------
    constant, slope : float
        Coefficients of ``constant + slope*t``.

    Returns
    -------
    list[float]
        Empty or one-element root list.
    """

    if slope == 0.0:
        return []
    root = -constant / slope
    return [root] if 0.0 < root < 1.0 else []


def _unit_interval_quadratic_roots(a: float, b: float, c: float) -> List[float]:
    """Return real quadratic roots strictly inside the unit interval.

    Parameters
    ----------
    a, b, c : float
        Coefficients of ``a*t^2 + b*t + c``.

    Returns
    -------
    list[float]
        Canonically sorted in-range roots.
    """

    if a == 0.0:
        return _unit_interval_root(c, b)
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return []
    square_root = math.sqrt(max(0.0, discriminant))
    roots = [(-b - square_root) / (2.0 * a), (-b + square_root) / (2.0 * a)]
    return sorted({root for root in roots if 0.0 < root < 1.0})


def _sqrt_quadratic_primitive(a: float, b: float, c: float, value: float) -> float:
    """Evaluate U10 section 5a's primitive of ``sqrt(a*t^2+b*t+c)``.

    Parameters
    ----------
    a, b, c : float
        Quadratic coefficients with a globally nonnegative quadratic.
    value : float
        Evaluation coordinate.

    Returns
    -------
    float
        Closed-form primitive value.
    """

    quadratic = max(0.0, a * value * value + b * value + c)
    if a == 0.0:
        return math.sqrt(max(0.0, c)) * value
    delta = max(0.0, 4.0 * a * c - b * b)
    root = math.sqrt(quadratic)
    if delta == 0.0:
        center = -b / (2.0 * a)
        sign = -1.0 if value < center else 1.0
        return math.sqrt(a) * sign * (value - center) ** 2 / 2.0
    return (2.0 * a * value + b) * root / (4.0 * a) + delta / (8.0 * a**1.5) * math.asinh(
        (2.0 * a * value + b) / math.sqrt(delta)
    )


def _segment_box_deficit_integral(
    start: torch.Tensor,
    end: torch.Tensor,
    box: BoxGeometry,
    stroke_half_width: float,
    intrinsic_unit: float,
) -> Tuple[float, bool]:
    """Integrate U10's exact clearance-deficit kernel on one segment.

    Parameters
    ----------
    start, end : torch.Tensor
        Flattened route-segment endpoints with shape ``[2]``.
    box : BoxGeometry
        Axis-aligned specialization of the contract OBB obstacle.
    stroke_half_width : float
        Nonnegative route stroke half-width.
    intrinsic_unit : float
        Positive scene intrinsic unit.

    Returns
    -------
    tuple[float, bool]
        Dimensionless exact integral and whether the segment penetrates the box.
    """

    local_start = start - box.center
    direction = end - start
    length = float(torch.linalg.vector_norm(direction))
    if length == 0.0:
        return 0.0, False
    half = box.half_extents
    clearance_band = 0.5 * intrinsic_unit
    radius = float(torch.linalg.vector_norm(half))
    band_radius = clearance_band + stroke_half_width
    clamp_radius = stroke_half_width - radius
    breaks = [0.0, 1.0]
    for axis in range(2):
        origin = float(local_start[axis])
        delta = float(direction[axis])
        breaks.extend(_unit_interval_root(origin, delta))
        for sign in (-1.0, 1.0):
            breaks.extend(_unit_interval_root(origin - sign * float(half[axis]), delta))
            breaks.extend(
                _unit_interval_root(
                    sign * origin - float(half[axis]) - band_radius,
                    sign * delta,
                )
            )
            breaks.extend(
                _unit_interval_root(
                    sign * origin - float(half[axis]) - clamp_radius,
                    sign * delta,
                )
            )
    for sign_x in (-1.0, 1.0):
        for sign_y in (-1.0, 1.0):
            constant_x = sign_x * float(local_start[0]) - float(half[0])
            constant_y = sign_y * float(local_start[1]) - float(half[1])
            slope_x = sign_x * float(direction[0])
            slope_y = sign_y * float(direction[1])
            breaks.extend(_unit_interval_root(constant_x - constant_y, slope_x - slope_y))
            quadratic_a = slope_x * slope_x + slope_y * slope_y
            quadratic_b = 2.0 * (constant_x * slope_x + constant_y * slope_y)
            quadratic_c = constant_x * constant_x + constant_y * constant_y
            breaks.extend(
                _unit_interval_quadratic_roots(
                    quadratic_a,
                    quadratic_b,
                    quadratic_c - band_radius * band_radius,
                )
            )
            breaks.extend(
                _unit_interval_quadratic_roots(
                    quadratic_a,
                    quadratic_b,
                    quadratic_c - clamp_radius * clamp_radius,
                )
            )
    ordered = sorted(set(breaks))
    total = 0.0
    penetrates = False
    maximum = (1.0 + radius / clearance_band) ** 2

    def signed_distance(parameter: float) -> float:
        """Return exact signed point-to-box distance in the box frame.

        Parameters
        ----------
        parameter : float
            Segment parameter in ``[0, 1]``.

        Returns
        -------
        float
            Positive exterior distance or negative interior depth.
        """

        point = local_start + parameter * direction
        excess = torch.abs(point) - half
        if bool((excess > 0.0).any()):
            return float(torch.linalg.vector_norm(torch.clamp(excess, min=0.0)))
        return max(float(excess[0]), float(excess[1]))

    for lower, upper in zip(ordered[:-1], ordered[1:]):
        if upper <= lower:
            continue
        midpoint = (lower + upper) / 2.0
        point = local_start + midpoint * direction
        excess = torch.abs(point) - half
        distance = signed_distance(midpoint)
        clearance = distance - stroke_half_width
        penetrates = penetrates or distance < 0.0
        if clearance >= clearance_band:
            continue
        if clearance <= -radius:
            total += maximum * (upper - lower)
            continue
        positive_axes = excess > 0.0
        if bool(positive_axes.all()):
            signs = torch.where(point >= 0.0, 1.0, -1.0)
            constants = signs * local_start - half
            slopes = signs * direction
            a = float(torch.dot(slopes, slopes))
            b = 2.0 * float(torch.dot(constants, slopes))
            c = float(torch.dot(constants, constants))
            polynomial = (
                band_radius * band_radius * (upper - lower)
                + a * (upper**3 - lower**3) / 3.0
                + b * (upper**2 - lower**2) / 2.0
                + c * (upper - lower)
            )
            square_root = _sqrt_quadratic_primitive(a, b, c, upper) - (
                _sqrt_quadratic_primitive(a, b, c, lower)
            )
            total += (polynomial - 2.0 * band_radius * square_root) / clearance_band**2
            continue
        distance_lower = signed_distance(lower)
        distance_upper = signed_distance(upper)
        slope = (distance_upper - distance_lower) / (upper - lower)
        intercept = distance_lower - slope * lower - stroke_half_width
        if slope == 0.0:
            total += ((clearance_band - intercept) / clearance_band) ** 2 * (upper - lower)
        else:
            primitive_upper = -((clearance_band - slope * upper - intercept) ** 3) / (
                3.0 * slope * clearance_band**2
            )
            primitive_lower = -((clearance_band - slope * lower - intercept) ** 3) / (
                3.0 * slope * clearance_band**2
            )
            total += primitive_upper - primitive_lower
    return max(0.0, length * total / intrinsic_unit), penetrates


def _raw_u10_blend(values: List[float], opportunity: int) -> float:
    """Apply U10's raw-component aggregation and common saturation map.

    Parameters
    ----------
    values : list[float]
        Full input-only population of nonnegative per-pair integrals.
    opportunity : int
        Analytic scored-pair count, equal to ``len(values)``.

    Returns
    -------
    float
        Contract U10 defect in ``[0, 1)``.
    """

    ordered = sorted(values)
    mean = sum(ordered) / opportunity
    tail_mass = 0.10 * opportunity
    remaining = tail_mass
    tail_sum = 0.0
    for value in reversed(ordered):
        mass = min(1.0, remaining)
        tail_sum += mass * value
        remaining -= mass
        if remaining <= 0.0:
            break
    cvar = tail_sum / tail_mass
    maximum = ordered[-1]
    smooth_maximum = maximum + 0.05 * math.log(
        sum(math.exp((value - maximum) / 0.05) for value in ordered) / opportunity
    )

    def saturate(value: float) -> float:
        """Map one nonnegative component through U10's frozen x0 curve.

        Parameters
        ----------
        value : float
            Nonnegative raw component.

        Returns
        -------
        float
            Saturated component.
        """

        return value / (value + 0.01) if value > 0.0 else 0.0

    return 0.65 * saturate(mean) + 0.25 * saturate(cvar) + 0.10 * saturate(smooth_maximum)


def U10(scene: Scene) -> FacetResult:
    """Edge-node occlusion / false attachment. Frozen SHA-256: 91f7929473c6d2f9fb9afc0b0fded75bdc899fe521ca942794b6042739aaf850."""

    if scene.edge_count < 1 or scene.node_count < 3:
        return na_result("too_few_objects")
    burdens: List[float] = []
    penetration_count = 0
    for route in resolved_routes(scene):
        incident = set(scene.graph.edges[route.edge_index])
        obstacles = [box for box in scene.node_boxes if box.owner not in incident]
        obstacles.extend(box for box in scene.node_label_boxes if box.owner not in incident)
        width = (
            scene.style.edge_stroke_widths[route.edge_index]
            if scene.style.edge_stroke_widths
            else scene.style.route_stroke_width * scene.style.coordinate_scale
        )
        for box in obstacles:
            burden = 0.0
            penetrates = False
            for start, end in zip(route.points[:-1], route.points[1:]):
                segment_burden, segment_penetrates = _segment_box_deficit_integral(
                    start,
                    end,
                    box,
                    width / 2.0,
                    scene.intrinsic_unit,
                )
                burden += segment_burden
                penetrates = penetrates or segment_penetrates
            burdens.append(burden)
            penetration_count += int(penetrates)
    if not burdens:
        return na_result("too_few_objects")
    defect = _raw_u10_blend(burdens, len(burdens))
    return value_result(
        defect,
        {"U10.headline": defect},
        {
            "pair_count": len(burdens),
            "penetration_count": penetration_count,
            "pair_integrals": tuple(burdens),
            "sum_D_eb": sum(burdens),
        },
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

    self_crossings = 0
    row_one: List[float] = []
    row_two: List[float] = []
    row_three: List[float] = []
    row_four: List[float] = []
    terminal_records: DefaultDict[int, List[Tuple[torch.Tensor, Optional[torch.Tensor]]]] = (
        defaultdict(list)
    )
    raw_edges: List[Dict[str, float]] = []
    routes = resolved_routes(scene)
    ranks = resolved_ranks(scene)
    for route in routes:
        points = route.points
        vectors = points[1:] - points[:-1]
        lengths = torch.linalg.vector_norm(vectors, dim=1)
        arc = float(torch.sum(lengths))
        source, target = scene.graph.edges[route.edge_index]
        source_tangent = _first_nonzero_tangent(points, False)
        target_tangent = _first_nonzero_tangent(points, True)
        terminal_records[source].append((points[0], source_tangent))
        terminal_records[target].append((points[-1], target_tangent))
        if arc == 0.0:
            continue
        chord_vector = points[-1] - points[0]
        chord = float(torch.linalg.vector_norm(chord_vector))
        chord_direction = (
            chord_vector / chord if chord > 0.0 else torch.zeros(2, dtype=torch.float64)
        )
        event_severity = 0.0
        for left in range(points.shape[0] - 1):
            for right in range(left + 2, points.shape[0] - 1):
                if proper_intersection(
                    points[left], points[left + 1], points[right], points[right + 1]
                ):
                    self_crossings += 1
                    angle = _segment_angle(vectors[left], vectors[right])
                    event_severity += 1.0 + math.cos(angle) ** 2
        self_defect = 1.0 - math.exp(-math.log(2.0) * event_severity)
        route_diameter = float(torch.max(torch.cdist(points, points)))
        backtracking = sum(
            max(0.0, -float(torch.dot(chord_direction, vector / length))) * float(length)
            for vector, length in zip(vectors, lengths)
            if float(length) > 0.0
        )
        backtracking_defect = 1.0 - math.exp(-backtracking / max(chord, route_diameter, 1e-300))
        row_one.append(1.0 - (1.0 - self_defect) * (1.0 - backtracking_defect))

        turns = _signed_route_turns(points)
        total_turn = sum(abs(turn) for turn in turns)
        wiggle = total_turn - abs(sum(turns))
        baseline_length, baseline_turn = _route_baseline(scene, route)
        if scene.graph.edge_styles is not None and source != target:
            style = scene.graph.edge_styles[route.edge_index]
            excess_turn = _zero_hinge(total_turn - baseline_turn, 0.05)
            if style == "straight":
                bend_defect = 1.0 - math.exp(-total_turn / (math.pi / 2.0))
            elif style == "orthogonal":
                off_axis = (
                    sum(
                        float(length)
                        * (1.0 - math.cos(4.0 * _nearest_axis_deviation(vector)))
                        / 2.0
                        for vector, length in zip(vectors, lengths)
                        if float(length) > 0.0
                    )
                    / arc
                )
                bend_defect = 0.5 * (1.0 - math.exp(-off_axis / 0.15)) + 0.5 * (
                    1.0 - math.exp(-excess_turn / (math.pi / 2.0))
                )
            else:
                turn_weight, wiggle_weight = (0.7, 0.3) if style == "polyline" else (0.6, 0.4)
                bend_defect = turn_weight * (
                    1.0 - math.exp(-excess_turn / (math.pi / 2.0))
                ) + wiggle_weight * (1.0 - math.exp(-wiggle / (math.pi / 2.0)))
            row_two.append(bend_defect)
        if source != target:
            log_ratio = math.log(arc / baseline_length)
            row_three.append(1.0 - math.exp(-_zero_hinge(log_ratio, 0.05) / math.log(2.0)))
        feedback = scene.graph.feedback is not None and scene.graph.feedback[route.edge_index]
        same_rank = ranks is not None and ranks[source] == ranks[target]
        if (
            scene.graph.directed
            and scene.graph.flow_axis is not None
            and source != target
            and not feedback
            and not same_rank
        ):
            axis = torch.tensor(scene.graph.flow_axis, dtype=torch.float64)
            counterflow = sum(
                max(0.0, -float(torch.dot(axis, vector / length))) * float(length)
                for vector, length in zip(vectors, lengths)
                if float(length) > 0.0
            )
            row_four.append(1.0 - math.exp(-(counterflow / arc) / 0.25))
        raw_edges.append(
            {
                "edge": float(route.edge_index),
                "arc_length": arc,
                "baseline_length": baseline_length,
                "total_turn": total_turn,
                "baseline_turn": baseline_turn,
                "backtracking": backtracking,
                "self_event_severity": event_severity,
            }
        )
    values: Dict[str, float] = {}
    if row_one:
        values["U11.i"] = global_blend(row_one)
    if row_two:
        values["U11.ii"] = global_blend(row_two)
    if row_three:
        values["U11.iii"] = global_blend(row_three)
    if row_four:
        values["U11.iv"] = global_blend(row_four)
    if not scene.graph.ports:
        node_defects: List[float] = []
        node_weights: List[float] = []
        for records in terminal_records.values():
            if len(records) < 2:
                continue
            confusability = []
            for left_index, left in enumerate(records):
                for right in records[left_index + 1 :]:
                    gap_factor = math.exp(
                        -(
                            (
                                float(torch.linalg.vector_norm(left[0] - right[0]))
                                / (scene.intrinsic_unit / 4.0)
                            )
                            ** 2
                        )
                    )
                    if left[1] is None or right[1] is None:
                        # The gap factor of the closed form stays well defined
                        # when a zero-arc route has no initial tangent; only the
                        # angle factor is undefined and takes its supremum, so a
                        # coincident tangent-less pair is the coincidence limit
                        # while a distant one still earns its separation.
                        confusability.append(gap_factor)
                        continue
                    confusability.append(
                        gap_factor
                        * math.exp(
                            -((_oriented_angle(left[1], right[1]) / math.radians(15.0)) ** 2)
                        )
                    )
            node_defects.append(snap_unit(sum(confusability) / len(confusability)))
            node_weights.append(float(len(records)))
        if node_defects:
            values["U11.v"] = global_blend(node_defects, node_weights)
    dropped_subterms = []
    if scene.graph.edge_styles is None:
        dropped_subterms.append("U11.ii:no_declared_style")
    feedback_count = sum(scene.graph.feedback or ())
    return FacetResult(
        ResultState.VALUE,
        None,
        None,
        values,
        {
            "self_intersection_count": self_crossings,
            "edges": tuple(raw_edges),
            "dropped_subterms": tuple(dropped_subterms),
            "feedback_edge_count": feedback_count,
            "feedback_edge_coverage": feedback_count / scene.edge_count
            if scene.edge_count
            else 0.0,
        },
    )


def _signed_route_turns(points: torch.Tensor) -> List[float]:
    """Return signed exterior turns of a flattened polyline.

    Parameters
    ----------
    points : torch.Tensor
        Route vertices with shape ``[P, 2]``.

    Returns
    -------
    list[float]
        Signed turns in radians.
    """

    turns: List[float] = []
    for index in range(1, points.shape[0] - 1):
        incoming = points[index] - points[index - 1]
        outgoing = points[index + 1] - points[index]
        if (
            float(torch.linalg.vector_norm(incoming)) == 0.0
            or float(torch.linalg.vector_norm(outgoing)) == 0.0
        ):
            continue
        cross = float(incoming[0] * outgoing[1] - incoming[1] * outgoing[0])
        dot = float(torch.dot(incoming, outgoing))
        turns.append(math.atan2(cross, dot))
    return turns


def _zero_hinge(value: float, width: float) -> float:
    """Evaluate U11's zero-anchored C1 excess hinge.

    Parameters
    ----------
    value : float
        Signed excess.
    width : float
        Positive quadratic transition width.

    Returns
    -------
    float
        Zero for nonpositive input and asymptotically linear excess.
    """

    if value <= 0.0:
        return 0.0
    if value <= width:
        return value * value / (2.0 * width)
    return value - width / 2.0


def _nearest_axis_deviation(vector: torch.Tensor) -> float:
    """Return acute direction deviation from the nearest page axis.

    Parameters
    ----------
    vector : torch.Tensor
        Nonzero segment vector with shape ``[2]``.

    Returns
    -------
    float
        Deviation in ``[0, pi/4]``.
    """

    angle = math.atan2(float(vector[1]), float(vector[0])) % (math.pi / 2.0)
    return min(angle, math.pi / 2.0 - angle)


def _first_nonzero_tangent(points: torch.Tensor, reverse: bool) -> Optional[torch.Tensor]:
    """Return a unit terminal tangent pointing away from its node.

    Parameters
    ----------
    points : torch.Tensor
        Route vertices with shape ``[P, 2]``.
    reverse : bool
        Read from the target terminal when true.

    Returns
    -------
    torch.Tensor or None
        Unit tangent, or none for an all-zero route.
    """

    ordered = torch.flip(points, dims=(0,)) if reverse else points
    for point in ordered[1:]:
        vector = point - ordered[0]
        length = float(torch.linalg.vector_norm(vector))
        if length > 0.0:
            return vector / length
    return None


def _regular_polygon(center: torch.Tensor, radius: float, sides: int) -> torch.Tensor:
    """Build a counter-clockwise regular polygon.

    Parameters
    ----------
    center : torch.Tensor
        Polygon center with shape ``[2]``.
    radius : float
        Positive circumradius.
    sides : int
        Number of polygon sides, at least three.

    Returns
    -------
    torch.Tensor
        Polygon vertices with shape ``[sides, 2]``.
    """

    angles = torch.arange(sides, dtype=torch.float64) * (2.0 * math.pi / sides)
    offsets = radius * torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    return center + offsets


def _box_boundary_segments(box: BoxGeometry) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Return the four canonical boundary segments of one axis-aligned box.

    Parameters
    ----------
    box : BoxGeometry
        Axis-aligned obstacle box.

    Returns
    -------
    list[tuple[torch.Tensor, torch.Tensor]]
        Four boundary segments in counter-clockwise order.
    """

    lower = box.center - box.half_extents
    upper = box.center + box.half_extents
    corners = [
        torch.tensor([lower[0], lower[1]], dtype=torch.float64),
        torch.tensor([upper[0], lower[1]], dtype=torch.float64),
        torch.tensor([upper[0], upper[1]], dtype=torch.float64),
        torch.tensor([lower[0], upper[1]], dtype=torch.float64),
    ]
    return list(zip(corners, corners[1:] + corners[:1]))


def _polygon_boundary_segments(polygon: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Return consecutive boundary segments of one polygon.

    Parameters
    ----------
    polygon : torch.Tensor
        Polygon vertices with shape ``[P, 2]``.

    Returns
    -------
    list[tuple[torch.Tensor, torch.Tensor]]
        Closed polygon boundary segments.
    """

    return [
        (polygon[index], polygon[(index + 1) % polygon.shape[0]])
        for index in range(polygon.shape[0])
    ]


def _segment_boundary_parameter(
    start: torch.Tensor,
    end: torch.Tensor,
    boundary_start: torch.Tensor,
    boundary_end: torch.Tensor,
) -> Optional[Tuple[float, torch.Tensor]]:
    """Return one transversal segment-boundary intersection.

    Parameters
    ----------
    start, end : torch.Tensor
        Query segment endpoints with shape ``[2]``.
    boundary_start, boundary_end : torch.Tensor
        Boundary segment endpoints with shape ``[2]``.

    Returns
    -------
    tuple[float, torch.Tensor] or None
        Query parameter and intersection point, including endpoints.
    """

    query = end - start
    boundary = boundary_end - boundary_start
    denominator = float(query[0] * boundary[1] - query[1] * boundary[0])
    if denominator == 0.0:
        return None
    offset = boundary_start - start
    query_parameter = float(offset[0] * boundary[1] - offset[1] * boundary[0]) / denominator
    boundary_parameter = float(offset[0] * query[1] - offset[1] * query[0]) / denominator
    if 0.0 <= query_parameter <= 1.0 and 0.0 <= boundary_parameter <= 1.0:
        return query_parameter, start + query_parameter * query
    return None


def _point_in_convex_polygon(point: torch.Tensor, polygon: torch.Tensor) -> bool:
    """Test closed membership in a counter-clockwise convex polygon.

    Parameters
    ----------
    point : torch.Tensor
        Query point with shape ``[2]``.
    polygon : torch.Tensor
        Counter-clockwise convex polygon with shape ``[P, 2]``.

    Returns
    -------
    bool
        Whether the point lies inside or on the polygon.
    """

    for start, end in _polygon_boundary_segments(polygon):
        edge = end - start
        offset = point - start
        if float(edge[0] * offset[1] - edge[1] * offset[0]) < -1e-12:
            return False
    return True


def _cleared_obstacle_contains(
    point: torch.Tensor, box: BoxGeometry, terminal_polygons: Sequence[torch.Tensor]
) -> bool:
    """Test strict membership in a terminal-cleared U11 obstacle.

    Parameters
    ----------
    point : torch.Tensor
        Query point with shape ``[2]``.
    box : BoxGeometry
        Uninflated node obstacle.
    terminal_polygons : sequence[torch.Tensor]
        Two terminal-centered regular 16-gons removed from the obstacle.

    Returns
    -------
    bool
        Whether the point lies in the residual obstacle interior.
    """

    if not bool(torch.all(torch.abs(point - box.center) < box.half_extents)):
        return False
    return not any(_point_in_convex_polygon(point, polygon) for polygon in terminal_polygons)


def _segment_cleared_obstacle_interior_intersection(
    start: torch.Tensor,
    end: torch.Tensor,
    box: BoxGeometry,
    terminal_polygons: Sequence[torch.Tensor],
) -> bool:
    """Test whether a segment crosses a terminal-cleared obstacle interior.

    Parameters
    ----------
    start, end : torch.Tensor
        Segment endpoints with shape ``[2]``.
    box : BoxGeometry
        Uninflated node obstacle.
    terminal_polygons : sequence[torch.Tensor]
        Polygonized terminal disks removed from the box.

    Returns
    -------
    bool
        Whether any positive-length segment interval lies in the residual obstacle.
    """

    parameters = [0.0, 1.0]
    boundaries = _box_boundary_segments(box)
    for polygon in terminal_polygons:
        boundaries.extend(_polygon_boundary_segments(polygon))
    for boundary_start, boundary_end in boundaries:
        intersection = _segment_boundary_parameter(start, end, boundary_start, boundary_end)
        if intersection is not None:
            parameters.append(intersection[0])
    ordered = sorted(set(parameters))
    direction = end - start
    return any(
        _cleared_obstacle_contains(
            start + ((lower + upper) / 2.0) * direction,
            box,
            terminal_polygons,
        )
        for lower, upper in zip(ordered[:-1], ordered[1:])
        if upper > lower
    )


def _cleared_obstacle_vertices(
    box: BoxGeometry, terminal_polygons: Sequence[torch.Tensor]
) -> List[torch.Tensor]:
    """Return visibility vertices of a terminal-cleared box.

    Parameters
    ----------
    box : BoxGeometry
        Uninflated node obstacle.
    terminal_polygons : sequence[torch.Tensor]
        Polygonized terminal disks removed from the box.

    Returns
    -------
    list[torch.Tensor]
        Residual box corners, disk vertices inside the box, and boundary intersections.
    """

    box_segments = _box_boundary_segments(box)
    candidates = [start for start, _ in box_segments]
    for polygon in terminal_polygons:
        for point in polygon:
            if bool(torch.all(torch.abs(point - box.center) <= box.half_extents)):
                candidates.append(point)
        for box_start, box_end in box_segments:
            for polygon_start, polygon_end in _polygon_boundary_segments(polygon):
                intersection = _segment_boundary_parameter(
                    box_start,
                    box_end,
                    polygon_start,
                    polygon_end,
                )
                if intersection is not None:
                    candidates.append(intersection[1])
    retained = [
        point
        for point in candidates
        if not _cleared_obstacle_contains(point, box, terminal_polygons)
    ]
    unique: Dict[Tuple[float, float], torch.Tensor] = {}
    for point in retained:
        unique[(float(point[0]), float(point[1]))] = point
    return [unique[key] for key in sorted(unique)]


def _route_baseline(scene: Scene, route: Route) -> Tuple[float, float]:
    """Compute U11's obstacle-aware capped visibility-path baseline.

    Parameters
    ----------
    scene : Scene
        Validated scene with derived node boxes.
    route : Route
        Visible route whose endpoint obstacles are exempt.

    Returns
    -------
    tuple[float, float]
        Capped baseline length and its total absolute turning.
    """

    source, target = scene.graph.edges[route.edge_index]
    start = route.points[0]
    end = route.points[-1]
    chord = float(torch.linalg.vector_norm(end - start))
    if chord == 0.0:
        return max(scene.intrinsic_unit, 1e-300), 0.0
    cap = 4.0 * chord
    clear_radius = _U11_TERMINAL_CLEAR_RADIUS * scene.intrinsic_unit
    terminal_polygons = (
        _regular_polygon(start, clear_radius, _U11_TERMINAL_DISK_SIDES),
        _regular_polygon(end, clear_radius, _U11_TERMINAL_DISK_SIDES),
    )
    obstacles = [
        box
        for box in scene.node_boxes
        if box.owner not in {source, target}
        and float(torch.linalg.vector_norm(box.center - start))
        + float(torch.linalg.vector_norm(box.center - end))
        <= cap
    ]
    if not any(
        _segment_cleared_obstacle_interior_intersection(
            start,
            end,
            box,
            terminal_polygons,
        )
        for box in obstacles
    ):
        return chord, 0.0
    obstacle_vertices: List[torch.Tensor] = []
    for box in obstacles:
        obstacle_vertices.extend(_cleared_obstacle_vertices(box, terminal_polygons))
    unique_vertices: Dict[Tuple[float, float], torch.Tensor] = {}
    for point in obstacle_vertices:
        unique_vertices[(float(point[0]), float(point[1]))] = point
    vertices = [start, end] + [unique_vertices[key] for key in sorted(unique_vertices)]
    neighbors: List[List[Tuple[int, float]]] = [[] for _ in vertices]
    for left_index, left in enumerate(vertices):
        for right_index in range(left_index + 1, len(vertices)):
            right = vertices[right_index]
            if any(
                _segment_cleared_obstacle_interior_intersection(
                    left,
                    right,
                    box,
                    terminal_polygons,
                )
                for box in obstacles
            ):
                continue
            distance = float(torch.linalg.vector_norm(right - left))
            neighbors[left_index].append((right_index, distance))
            neighbors[right_index].append((left_index, distance))
    distances = [math.inf] * len(vertices)
    paths: List[Tuple[int, ...]] = [tuple() for _ in vertices]
    distances[0] = 0.0
    paths[0] = (0,)
    queue: List[Tuple[float, Tuple[int, ...], int]] = [(0.0, (0,), 0)]
    while queue:
        distance, path, node = heapq.heappop(queue)
        if distance != distances[node] or path != paths[node]:
            continue
        for neighbor, edge_length in sorted(neighbors[node]):
            candidate = distance + edge_length
            candidate_path = path + (neighbor,)
            if candidate < distances[neighbor] or (
                candidate == distances[neighbor] and candidate_path < paths[neighbor]
            ):
                distances[neighbor] = candidate
                paths[neighbor] = candidate_path
                heapq.heappush(queue, (candidate, candidate_path, neighbor))
    visible = distances[1]
    temperature = 0.1 * chord
    if math.isfinite(visible):
        minimum = min(visible, cap)
        soft_min = minimum - temperature * math.log(
            math.exp(-(visible - minimum) / temperature) + math.exp(-(cap - minimum) / temperature)
        )
        baseline = max(chord, soft_min)
        path_points = torch.stack([vertices[index] for index in paths[1]])
        turning = sum(abs(value) for value in _signed_route_turns(path_points))
    else:
        baseline = cap
        turning = 0.0
    return baseline, turning


def U12(scene: Scene) -> FacetResult:
    """Path/edge continuity. Frozen SHA-256: 0170de3c481a98b8d063ff505c039de969221b1e43c3bef596a64c7e0c083d47."""

    directions = _incident_secants(scene)
    degrees = [0] * scene.node_count
    loop_nodes = set()
    for source, target in scene.graph.edges:
        if source == target:
            loop_nodes.add(source)
            degrees[source] += 2
        else:
            degrees[source] += 1
            degrees[target] += 1
    nodes = [node for node, degree in enumerate(degrees) if degree == 2 and node not in loop_nodes]
    if len(nodes) < 5:
        return na_result("no_degree2_chains")
    deviations = []
    for node in nodes:
        records = directions[node]
        if len(records) != 2:
            deviations.append(0.0)
            continue
        left_angle, _, left_confidence = records[0]
        right_angle, _, right_confidence = records[1]
        left = torch.tensor([math.cos(left_angle), math.sin(left_angle)], dtype=torch.float64)
        right = torch.tensor([math.cos(right_angle), math.sin(right_angle)], dtype=torch.float64)
        cosine = min(1.0, max(-1.0, float(torch.dot(-left, right))))
        deviation = math.acos(cosine) / math.pi
        deviations.append(left_confidence * right_confidence * deviation)
    defect = global_blend(deviations)
    return value_result(
        defect,
        {"U12.headline": defect},
        {"degree2_node_count": len(nodes)},
    )


def U13(scene: Scene) -> FacetResult:
    """Near-parallel edge ambiguity / bundle confusion. Frozen SHA-256: 359205189c5fa413aae039d4033602af0a665c9e93e583c274b57d1341d1ca43."""

    if scene.edge_count < 2:
        return na_result("too_few_edges")
    routes = resolved_routes(scene)
    bundles = scene.graph.edge_bundles or tuple(None for _ in range(scene.edge_count))
    contributions: List[float] = []
    pair_records: List[Tuple[int, int, float]] = []
    for left_index, left in enumerate(routes):
        for right in routes[left_index + 1 :]:
            if (
                bundles[left.edge_index] is not None
                and bundles[left.edge_index] == bundles[right.edge_index]
            ):
                continue
            shared_nodes = set(scene.graph.edges[left.edge_index]) & set(
                scene.graph.edges[right.edge_index]
            )
            left_points = left.points
            right_points = right.points
            if shared_nodes:
                left_edge = scene.graph.edges[left.edge_index]
                right_edge = scene.graph.edges[right.edge_index]
                left_points = _trim_polyline(
                    left_points,
                    2.0 * scene.intrinsic_unit if left_edge[0] in shared_nodes else 0.0,
                    2.0 * scene.intrinsic_unit if left_edge[1] in shared_nodes else 0.0,
                )
                right_points = _trim_polyline(
                    right_points,
                    2.0 * scene.intrinsic_unit if right_edge[0] in shared_nodes else 0.0,
                    2.0 * scene.intrinsic_unit if right_edge[1] in shared_nodes else 0.0,
                )
            contribution = _parallel_route_integral(left_points, right_points, scene.intrinsic_unit)
            contributions.append(contribution)
            pair_records.append((left.edge_index, right.edge_index, contribution))
    degrees = [0] * scene.node_count
    for source, target in scene.graph.edges:
        degrees[source] += 1
        degrees[target] += 1
    opportunity = int(scene.edge_count + sum(degree * (degree - 1) / 2.0 for degree in degrees))
    raw_sum = sum(contributions)
    values: Dict[str, float] = {
        "U13.i": _raw_u13_blend(contributions, opportunity),
    }
    bundle_defects: List[float] = []
    if scene.graph.edge_bundles is not None:
        grouped: DefaultDict[str, List[Route]] = defaultdict(list)
        for route, bundle in zip(routes, bundles):
            if bundle is not None:
                grouped[bundle].append(route)
        for members in grouped.values():
            for end_fraction in (0.10, 0.90):
                departure_points = [
                    _point_at_arc_fraction(route.points, end_fraction) for route in members
                ]
                minimum = min(
                    float(torch.linalg.vector_norm(left - right))
                    for left_index, left in enumerate(departure_points)
                    for right in departure_points[left_index + 1 :]
                )
                bundle_defects.append(max(0.0, 1.0 - minimum / (0.5 * scene.intrinsic_unit)) ** 2)
        if bundle_defects:
            values["U13.ii"] = blend_with_weights(
                bundle_defects,
                None,
                (0.65 * 0.4 / 0.9, 0.25 * 0.4 / 0.9, 0.6),
            )
    return mean_result(
        "U13",
        values,
        {
            "opportunity": opportunity,
            "sum_C_ef": raw_sum,
            "pair_contributions": tuple(pair_records),
            "bundle_end_defects": tuple(bundle_defects),
        },
    )


def _raw_u13_blend(values: List[float], opportunity: int) -> float:
    """Aggregate U13 pair integrals through its mean/tail/smoothmax blend.

    Parameters
    ----------
    values : list[float]
        Nonnegative admitted pair integrals.
    opportunity : int
        Frozen input-only normalizer for the mean component. Tail components use
        the contract's full route-pair population, including measured zeros.

    Returns
    -------
    float
        Contract-blended ambiguity defect.
    """

    if opportunity <= 0:
        return 0.0
    if not values:
        return 0.0
    population = sorted(values)
    mean = sum(population) / opportunity
    tail_mass = 0.10 * len(population)
    remaining = tail_mass
    tail_sum = 0.0
    for value in reversed(population):
        selected = min(1.0, remaining)
        tail_sum += selected * value
        remaining -= selected
        if remaining <= 0.0:
            break
    cvar = tail_sum / tail_mass
    maximum = population[-1]
    smooth_maximum = maximum + 0.05 * math.log(
        sum(math.exp((value - maximum) / 0.05) for value in population) / len(population)
    )

    def saturate(value: float) -> float:
        """Apply U13's common ``x/(x+0.05)`` saturation.

        Parameters
        ----------
        value : float
            Nonnegative raw aggregate.

        Returns
        -------
        float
            Bounded component burden.
        """

        return value / (value + 0.05) if value > 0.0 else 0.0

    return 0.65 * saturate(mean) + 0.25 * saturate(cvar) + 0.10 * saturate(smooth_maximum)


def _trim_polyline(points: torch.Tensor, start_trim: float, end_trim: float) -> torch.Tensor:
    """Trim fixed arc-length windows from a flattened polyline's ends.

    Parameters
    ----------
    points : torch.Tensor
        Polyline vertices with shape ``[P, 2]``.
    start_trim, end_trim : float
        Nonnegative arc lengths removed from the source and target ends.

    Returns
    -------
    torch.Tensor
        Remaining polyline, or a repeated midpoint when the windows consume it.
    """

    lengths = torch.linalg.vector_norm(points[1:] - points[:-1], dim=1)
    total = float(torch.sum(lengths))
    if start_trim + end_trim >= total:
        midpoint = _point_at_arc_fraction(points, 0.5)
        return torch.stack((midpoint, midpoint))
    cumulative = torch.cat((torch.zeros(1, dtype=torch.float64), torch.cumsum(lengths, dim=0)))

    def at_distance(distance: float) -> torch.Tensor:
        """Interpolate one point at an absolute arc distance.

        Parameters
        ----------
        distance : float
            Distance from the source terminal.

        Returns
        -------
        torch.Tensor
            Interpolated point with shape ``[2]``.
        """

        segment = int(
            torch.searchsorted(
                cumulative[1:], torch.tensor(distance, dtype=torch.float64), right=False
            )
        )
        local = (distance - float(cumulative[segment])) / max(float(lengths[segment]), 1e-300)
        return points[segment] + local * (points[segment + 1] - points[segment])

    start_distance = start_trim
    end_distance = total - end_trim
    retained = [at_distance(start_distance)]
    retained.extend(
        points[index]
        for index in range(1, points.shape[0] - 1)
        if start_distance < float(cumulative[index]) < end_distance
    )
    retained.append(at_distance(end_distance))
    return torch.stack(retained)


def _parallel_route_integral(
    left: torch.Tensor, right: torch.Tensor, intrinsic_unit: float
) -> float:
    """Integrate U13's kernel against the nearest point on the other route.

    Parameters
    ----------
    left, right : torch.Tensor
        Flattened polylines with shape ``[P, 2]``.
    intrinsic_unit : float
        Positive scene intrinsic unit.

    Returns
    -------
    float
        Dimensionless exact piecewise-polynomial route-pair integral.
    """

    total = 0.0
    radius = 3.0 * intrinsic_unit
    for start_left, end_left in zip(left[:-1], left[1:]):
        vector_left = end_left - start_left
        length_left = float(torch.linalg.vector_norm(vector_left))
        if length_left == 0.0:
            continue
        unit_left = vector_left / length_left
        right_segments = [
            (start_right, end_right, end_right - start_right)
            for start_right, end_right in zip(right[:-1], right[1:])
            if float(torch.linalg.vector_norm(end_right - start_right)) > 0.0
        ]
        if not right_segments:
            continue
        base_breaks = [0.0, 1.0]
        for start_right, _, vector_right in right_segments:
            denominator = float(torch.dot(vector_right, vector_right))
            offset = start_left - start_right
            projection_constant = float(torch.dot(offset, vector_right)) / denominator
            projection_slope = float(torch.dot(vector_left, vector_right)) / denominator
            base_breaks.extend(_unit_interval_root(projection_constant, projection_slope))
            base_breaks.extend(_unit_interval_root(projection_constant - 1.0, projection_slope))
        ordered_base = sorted(set(base_breaks))
        for lower, upper in zip(ordered_base[:-1], ordered_base[1:]):
            if upper <= lower:
                continue
            midpoint = (lower + upper) / 2.0
            candidates: List[Tuple[float, float, float, float]] = []
            for start_right, end_right, vector_right in right_segments:
                coefficients = _segment_distance_polynomial(
                    start_left,
                    vector_left,
                    start_right,
                    end_right,
                    vector_right,
                    midpoint,
                )
                unit_right = vector_right / torch.linalg.vector_norm(vector_right)
                cosine = abs(float(torch.dot(unit_left, unit_right)))
                candidates.append((*coefficients, cosine**4))
            envelope_breaks = [lower, upper]
            for left_index, left_candidate in enumerate(candidates):
                envelope_breaks.extend(
                    _quadratic_roots_in_interval(
                        left_candidate[0],
                        left_candidate[1],
                        left_candidate[2] - radius * radius,
                        lower,
                        upper,
                    )
                )
                for right_candidate in candidates[left_index + 1 :]:
                    envelope_breaks.extend(
                        _quadratic_roots_in_interval(
                            left_candidate[0] - right_candidate[0],
                            left_candidate[1] - right_candidate[1],
                            left_candidate[2] - right_candidate[2],
                            lower,
                            upper,
                        )
                    )
            ordered = sorted(set(envelope_breaks))
            for interval_left, interval_right in zip(ordered[:-1], ordered[1:]):
                if interval_right <= interval_left:
                    continue
                sample = (interval_left + interval_right) / 2.0
                candidate = min(
                    candidates,
                    key=lambda item: item[0] * sample * sample + item[1] * sample + item[2],
                )
                squared_distance = (
                    candidate[0] * sample * sample + candidate[1] * sample + candidate[2]
                )
                if squared_distance >= radius * radius:
                    continue
                total += (
                    candidate[3]
                    * length_left
                    * _quartic_distance_kernel_integral(
                        candidate[0],
                        candidate[1],
                        candidate[2],
                        radius,
                        interval_left,
                        interval_right,
                    )
                    / intrinsic_unit
                )
    return total


def _segment_distance_polynomial(
    query_start: torch.Tensor,
    query_vector: torch.Tensor,
    segment_start: torch.Tensor,
    segment_end: torch.Tensor,
    segment_vector: torch.Tensor,
    sample_parameter: float,
) -> Tuple[float, float, float]:
    """Return one point-to-segment squared-distance polynomial branch.

    Parameters
    ----------
    query_start, query_vector : torch.Tensor
        Query point path ``query_start + t*query_vector``.
    segment_start, segment_end, segment_vector : torch.Tensor
        Nonzero candidate segment geometry.
    sample_parameter : float
        Parameter selecting the constant, interior, or terminal projection branch.

    Returns
    -------
    tuple[float, float, float]
        Coefficients ``(a, b, c)`` of squared distance ``a*t^2+b*t+c``.
    """

    denominator = float(torch.dot(segment_vector, segment_vector))
    offset = query_start - segment_start
    projection = (
        float(torch.dot(offset, segment_vector))
        + sample_parameter * float(torch.dot(query_vector, segment_vector))
    ) / denominator
    if projection <= 0.0:
        difference = query_start - segment_start
        return (
            float(torch.dot(query_vector, query_vector)),
            2.0 * float(torch.dot(difference, query_vector)),
            float(torch.dot(difference, difference)),
        )
    if projection >= 1.0:
        difference = query_start - segment_end
        return (
            float(torch.dot(query_vector, query_vector)),
            2.0 * float(torch.dot(difference, query_vector)),
            float(torch.dot(difference, difference)),
        )
    offset_projection = float(torch.dot(offset, segment_vector))
    vector_projection = float(torch.dot(query_vector, segment_vector))
    return (
        float(torch.dot(query_vector, query_vector)) - vector_projection**2 / denominator,
        2.0
        * (
            float(torch.dot(offset, query_vector))
            - offset_projection * vector_projection / denominator
        ),
        float(torch.dot(offset, offset)) - offset_projection**2 / denominator,
    )


def _quadratic_roots_in_interval(
    a: float, b: float, c: float, lower: float, upper: float
) -> List[float]:
    """Return real roots strictly inside one parameter interval.

    Parameters
    ----------
    a, b, c : float
        Quadratic coefficients.
    lower, upper : float
        Open interval bounds.

    Returns
    -------
    list[float]
        Sorted unique roots inside ``(lower, upper)``.
    """

    if a == 0.0:
        if b == 0.0:
            return []
        root = -c / b
        return [root] if lower < root < upper else []
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return []
    square_root = math.sqrt(max(0.0, discriminant))
    roots = ((-b - square_root) / (2.0 * a), (-b + square_root) / (2.0 * a))
    return sorted({root for root in roots if lower < root < upper})


def _quartic_distance_kernel_integral(
    a: float,
    b: float,
    c: float,
    radius: float,
    lower: float,
    upper: float,
) -> float:
    """Integrate ``(1-(a*t^2+b*t+c)/radius^2)^2`` exactly.

    Parameters
    ----------
    a, b, c : float
        Squared-distance polynomial coefficients.
    radius : float
        Positive compact-support distance.
    lower, upper : float
        Parameter interval lying inside the support.

    Returns
    -------
    float
        Exact nonnegative kernel integral.
    """

    radius_squared = radius * radius
    radius_fourth = radius_squared * radius_squared
    coefficients = (
        1.0 - 2.0 * c / radius_squared + c * c / radius_fourth,
        -2.0 * b / radius_squared + 2.0 * b * c / radius_fourth,
        -2.0 * a / radius_squared + (b * b + 2.0 * a * c) / radius_fourth,
        2.0 * a * b / radius_fourth,
        a * a / radius_fourth,
    )

    def primitive(value: float) -> float:
        """Evaluate the quartic antiderivative.

        Parameters
        ----------
        value : float
            Parameter value.

        Returns
        -------
        float
            Antiderivative value.
        """

        return sum(
            coefficient * value ** (degree + 1) / (degree + 1)
            for degree, coefficient in enumerate(coefficients)
        )

    return max(0.0, primitive(upper) - primitive(lower))


def U15(scene: Scene) -> FacetResult:
    """Multi-edge / self-loop legibility. Frozen SHA-256: ba372b7b0425c2a202e8d76de7a16b88f9bb18ebb2a6220df0ecfc4089ad83c7."""

    multiplicities = Counter(
        tuple(sorted(edge)) for edge in scene.graph.edges if edge[0] != edge[1]
    )
    parallel_groups = [edge for edge, count in multiplicities.items() if count > 1]
    loops = [index for index, edge in enumerate(scene.graph.edges) if edge[0] == edge[1]]
    if not parallel_groups and not loops:
        return na_result("no_multiedges_or_selfloops")
    route_by_edge = {route.edge_index: route for route in resolved_routes(scene)}
    class_defects: List[float] = []
    class_pair_defects: Dict[Tuple[int, int], Tuple[float, ...]] = {}
    for edge in parallel_groups:
        indices = [
            index
            for index, candidate in enumerate(scene.graph.edges)
            if tuple(sorted(candidate)) == edge
        ]
        pair_defects: List[float] = []
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
                left_measured = _trim_polyline(
                    left.points, 0.1 * shared_length, 0.1 * shared_length
                )
                right_measured = _trim_polyline(
                    right.points, 0.1 * shared_length, 0.1 * shared_length
                )
                separation = min(
                    _segment_segment_distance(left_start, left_end, right_start, right_end)
                    for left_start, left_end in zip(left_measured[:-1], left_measured[1:])
                    for right_start, right_end in zip(right_measured[:-1], right_measured[1:])
                )
                relative = separation / shared_length if shared_length > 0.0 else 0.0
                defect = max(0.0, 1.0 - relative / 0.05) ** 2
                fade_shared = float(
                    smoothstep(
                        torch.tensor(shared_length / (0.1 * maximum_length), dtype=torch.float64)
                    )
                )
                fade_absolute = float(
                    smoothstep(
                        torch.tensor(
                            maximum_length / (0.1 * scene.intrinsic_unit), dtype=torch.float64
                        )
                    )
                )
                pair_defects.append(fade_shared * fade_absolute * defect)
        if pair_defects:
            class_defects.append(global_blend(pair_defects))
            class_pair_defects[edge] = tuple(pair_defects)
    values: Dict[str, float] = {}
    if class_defects:
        values["U15.i"] = global_blend(class_defects)
    loop_defects: List[float] = []
    for loop_index in loops:
        route = route_by_edge.get(loop_index)
        if route is None:
            continue
        owner = scene.graph.edges[loop_index][0]
        obstacles = [box for box in scene.node_label_boxes if box.owner == owner]
        obstacles.extend(box for box in scene.node_boxes if box.owner != owner)
        obstacles.extend(box for box in scene.node_label_boxes if box.owner != owner)
        width = (
            scene.style.edge_stroke_widths[loop_index]
            if scene.style.edge_stroke_widths
            else scene.style.route_stroke_width * scene.style.coordinate_scale
        )
        integral = 0.0
        for obstacle in obstacles:
            for start, end in zip(route.points[:-1], route.points[1:]):
                contribution, _ = _segment_box_deficit_integral(
                    start,
                    end,
                    obstacle,
                    width / 2.0,
                    scene.intrinsic_unit,
                )
                integral += contribution
        loop_defects.append(integral / (integral + 0.05) if integral > 0.0 else 0.0)
    if loop_defects:
        values["U15.ii"] = global_blend(loop_defects)
    return mean_result(
        "U15",
        values,
        {
            "parallel_classes": class_pair_defects,
            "loop_defects": tuple(loop_defects),
        },
    )


def _segment_segment_distance(
    start_left: torch.Tensor,
    end_left: torch.Tensor,
    start_right: torch.Tensor,
    end_right: torch.Tensor,
) -> float:
    """Return the exact minimum distance between two planar line segments.

    Parameters
    ----------
    start_left, end_left, start_right, end_right : torch.Tensor
        Segment endpoints with shape ``[2]``.

    Returns
    -------
    float
        Nonnegative Euclidean distance.
    """

    if proper_intersection(start_left, end_left, start_right, end_right):
        return 0.0
    return min(
        _point_segment_distance(start_left, start_right, end_right),
        _point_segment_distance(end_left, start_right, end_right),
        _point_segment_distance(start_right, start_left, end_left),
        _point_segment_distance(end_right, start_left, end_left),
    )


def _point_segment_distance(point: torch.Tensor, start: torch.Tensor, end: torch.Tensor) -> float:
    """Return exact Euclidean distance from a point to a line segment.

    Parameters
    ----------
    point, start, end : torch.Tensor
        Planar points with shape ``[2]``.

    Returns
    -------
    float
        Nonnegative distance.
    """

    direction = end - start
    denominator = float(torch.dot(direction, direction))
    if denominator == 0.0:
        return float(torch.linalg.vector_norm(point - start))
    parameter = min(1.0, max(0.0, float(torch.dot(point - start, direction)) / denominator))
    return float(torch.linalg.vector_norm(point - (start + parameter * direction)))


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
    overlap_defects: List[float] = []
    ownership_defects: List[float] = []
    overlap_sums: List[float] = []
    ambiguity_values: List[float] = []
    anchoring_values: List[float] = []
    route_by_edge = {route.edge_index: route for route in resolved_routes(scene)}
    obstacle_boxes = (
        list(scene.edge_label_boxes) + list(scene.node_label_boxes) + list(scene.node_boxes)
    )
    for label in scene.edge_label_boxes:
        label_area = float(4.0 * torch.prod(label.half_extents))
        overlap_sum = 0.0
        for obstacle in obstacle_boxes:
            if obstacle is label:
                continue
            _, overlap_fraction = aabb_pair(label, obstacle)
            obstacle_area = float(4.0 * torch.prod(obstacle.half_extents))
            overlap_area = overlap_fraction * min(label_area, obstacle_area)
            overlap_sum += overlap_area / min(label_area, obstacle_area)
        for edge_index, route in route_by_edge.items():
            width = (
                scene.style.edge_stroke_widths[edge_index]
                if scene.style.edge_stroke_widths
                else scene.style.route_stroke_width * scene.style.coordinate_scale
            )
            if edge_index == label.owner:
                label_height = 2.0 * float(label.half_extents[1])
                route_area = _route_box_ink_area(
                    route.points,
                    label,
                    width,
                    excluded_center=label.center,
                    excluded_radius=label_height,
                )
            else:
                route_area = _route_box_ink_area(route.points, label, width)
            overlap_sum += route_area / label_area
        overlap_sums.append(overlap_sum)
        overlap_defects.append(overlap_sum / (overlap_sum + 0.25))

        own_route = route_by_edge[label.owner]
        own_distance = _point_polyline_distance(label.center, own_route.points)
        foreign_distance = min(
            (
                _point_polyline_distance(label.center, route.points)
                for edge_index, route in route_by_edge.items()
                if edge_index != label.owner
            ),
            default=math.inf,
        )
        regularizer = 0.25 * scene.intrinsic_unit
        ratio = (
            (own_distance + regularizer) / (foreign_distance + regularizer)
            if math.isfinite(foreign_distance)
            else 0.0
        )
        ambiguity = ratio * ratio / (1.0 + ratio * ratio)
        label_height = 2.0 * float(label.half_extents[1])
        anchoring_ratio = max(0.0, own_distance - label_height) / (2.0 * scene.intrinsic_unit)
        anchoring = anchoring_ratio * anchoring_ratio / (1.0 + anchoring_ratio * anchoring_ratio)
        ownership = 1.0 - (1.0 - ambiguity) * (1.0 - anchoring)
        ambiguity_values.append(ambiguity)
        anchoring_values.append(anchoring)
        ownership_defects.append(ownership)
    values = {
        "U16.i": global_blend(overlap_defects),
        "U16.ii": global_blend(ownership_defects),
    }
    return mean_result(
        "U16",
        values,
        {
            "label_count": len(scene.edge_label_boxes),
            "overlap_sums": tuple(overlap_sums),
            "ambiguity": tuple(ambiguity_values),
            "anchoring": tuple(anchoring_values),
        },
    )


def _point_polyline_distance(point: torch.Tensor, points: torch.Tensor) -> float:
    """Return exact point-to-flattened-polyline distance.

    Parameters
    ----------
    point : torch.Tensor
        Query point with shape ``[2]``.
    points : torch.Tensor
        Polyline vertices with shape ``[P, 2]``.

    Returns
    -------
    float
        Minimum Euclidean distance.
    """

    distance, _ = _point_polyline_projection(point, points)
    return distance


def _point_polyline_projection(
    point: torch.Tensor, points: torch.Tensor
) -> Tuple[float, torch.Tensor]:
    """Return distance and nearest point on a flattened polyline.

    Parameters
    ----------
    point : torch.Tensor
        Query point with shape ``[2]``.
    points : torch.Tensor
        Polyline vertices with shape ``[P, 2]``.

    Returns
    -------
    tuple[float, torch.Tensor]
        Minimum distance and its canonical first nearest projection.
    """

    candidates: List[Tuple[float, torch.Tensor]] = []
    for start, end in zip(points[:-1], points[1:]):
        direction = end - start
        denominator = float(torch.dot(direction, direction))
        if denominator == 0.0:
            candidates.append((float(torch.linalg.vector_norm(point - start)), start))
            continue
        parameter = min(
            1.0,
            max(0.0, float(torch.dot(point - start, direction)) / denominator),
        )
        projection = start + parameter * direction
        candidates.append((float(torch.linalg.vector_norm(point - projection)), projection))
    return min(candidates, key=lambda item: item[0])


def _route_box_ink_area(
    points: torch.Tensor,
    box: BoxGeometry,
    width: float,
    *,
    excluded_center: Optional[torch.Tensor] = None,
    excluded_radius: float = 0.0,
) -> float:
    """Return flattened route-ribbon area whose centerline lies inside a box.

    Parameters
    ----------
    points : torch.Tensor
        Polyline vertices with shape ``[P, 2]``.
    box : BoxGeometry
        Axis-aligned label obstacle.
    width : float
        Positive ribbon width.
    excluded_center : torch.Tensor or None
        Optional anchor center whose local exemption disk is removed.
    excluded_radius : float
        Radius of the local anchor exemption disk.

    Returns
    -------
    float
        Centerline coverage times width, capped by the obstacle area.
    """

    lower = box.center - box.half_extents
    upper = box.center + box.half_extents
    length = 0.0
    for start, end in zip(points[:-1], points[1:]):
        direction = end - start
        entry = 0.0
        exit_ = 1.0
        for axis in range(2):
            delta = float(direction[axis])
            if delta == 0.0:
                if float(start[axis]) < float(lower[axis]) or float(start[axis]) > float(
                    upper[axis]
                ):
                    entry = 1.0
                    exit_ = 0.0
                    break
                continue
            first = (float(lower[axis]) - float(start[axis])) / delta
            second = (float(upper[axis]) - float(start[axis])) / delta
            entry = max(entry, min(first, second))
            exit_ = min(exit_, max(first, second))
            if entry > exit_:
                break
        if entry <= exit_:
            admitted = exit_ - entry
            if excluded_center is not None and excluded_radius > 0.0:
                offset = start - excluded_center
                quadratic_a = float(torch.dot(direction, direction))
                quadratic_b = 2.0 * float(torch.dot(offset, direction))
                quadratic_c = float(torch.dot(offset, offset)) - excluded_radius**2
                discriminant = quadratic_b**2 - 4.0 * quadratic_a * quadratic_c
                if quadratic_a > 0.0 and discriminant >= 0.0:
                    root = math.sqrt(discriminant)
                    circle_entry = (-quadratic_b - root) / (2.0 * quadratic_a)
                    circle_exit = (-quadratic_b + root) / (2.0 * quadratic_a)
                    admitted -= max(
                        0.0,
                        min(exit_, circle_exit) - max(entry, circle_entry),
                    )
            length += max(0.0, admitted) * float(torch.linalg.vector_norm(direction))
    return min(length * width, float(4.0 * torch.prod(box.half_extents)))
