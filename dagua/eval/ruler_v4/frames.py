"""Robust trimmed-core frames and intrinsic-unit frame measurements."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import torch

from dagua.eval.ruler_v4._tracing import Scalar, as_float, keep, tracing_active
from dagua.eval.ruler_v4._util import resolved_routes
from dagua.eval.ruler_v4.scene import Scene

N_SMALL = 50
N_FLOOR = 4
TRIM_RATE = 0.02
MIN_TRIM = 2
MIN_CORE = 4
CORE_FRACTION = 0.5
MAD_MULTIPLIER = 3.0
HALF_EXTENT_FLOOR = 0.5


@dataclass(frozen=True)
class RobustFrame:
    """Robust axis-aligned scene frame.

    Parameters
    ----------
    center : torch.Tensor
        Frame center with shape ``[2]``.
    half_extents : torch.Tensor
        Floored robust half extents with shape ``[2]``.
    trim_count : int
        Explicit per-side trim count.
    regime : int
        U21 frame regime, one through three.
    floor_bound : tuple[bool, bool]
        Whether the universal floor bound on each axis.
    """

    center: torch.Tensor
    half_extents: torch.Tensor
    trim_count: int
    regime: int
    floor_bound: Tuple[bool, bool]

    @property
    def area(self) -> Scalar:
        """Return full frame area.

        Returns
        -------
        float or torch.Tensor
            ``4 * half_width * half_height``; the historical float off a
            trace (bit-identical), the live scalar tensor inside one.
        """

        return keep(4.0 * torch.prod(self.half_extents))


@dataclass(frozen=True)
class RobustProjection:
    """One direction-resolved robust center and half-width.

    Parameters
    ----------
    center : float
        Robust projected center.
    half_extent : float
        Floored robust projected half-width.
    floor_bound : bool
        Whether the universal half-extent floor binds.
    """

    center: float
    half_extent: float
    floor_bound: bool


def minimum_core_count(node_count: int) -> int:
    """Return the normative minimum retained core population.

    Parameters
    ----------
    node_count : int
        Number of primitive centers.

    Returns
    -------
    int
        At least half the population and at least four objects.
    """

    return max(MIN_CORE, math.ceil(CORE_FRACTION * node_count))


def trim_count(node_count: int) -> int:
    """Return the explicit U21 trim count per side and direction.

    Parameters
    ----------
    node_count : int
        Number of primitive centers.

    Returns
    -------
    int
        Clamped trim count. Small-N regimes report the same input-only count even
        though their spread estimator does not discard objects.
    """

    if node_count < 0:
        raise ValueError("node_count must be nonnegative")
    maximum = max(MIN_TRIM, math.floor((node_count - minimum_core_count(node_count)) / 2.0))
    return min(max(math.ceil(TRIM_RATE * node_count), MIN_TRIM), maximum)


def _median(values: torch.Tensor) -> torch.Tensor:
    """Compute the conventional midpoint median along axis zero.

    Parameters
    ----------
    values : torch.Tensor
        Float64 tensor with shape ``[N, D]``.

    Returns
    -------
    torch.Tensor
        Median vector with shape ``[D]``.
    """

    ordered, _ = torch.sort(values, dim=0)
    count = ordered.shape[0]
    if count % 2:
        return ordered[count // 2]
    return (ordered[count // 2 - 1] + ordered[count // 2]) / 2.0


def robust_frame(positions: torch.Tensor, unit: float) -> RobustFrame:
    """Construct the U21 robust frame without any bounding-box extrema.

    Parameters
    ----------
    positions : torch.Tensor
        Finite node centers with shape ``[N, 2]``.
    unit : float
        Positive median primitive diagonal.

    Returns
    -------
    RobustFrame
        Exact input-count-selected trimmed-core or median/MAD frame.

    Raises
    ------
    ValueError
        If geometry is empty, malformed, non-finite, or has a nonpositive unit.
    """

    if positions.ndim != 2 or positions.shape[1] != 2 or positions.shape[0] == 0:
        raise ValueError("positions must have shape [N, 2] with N >= 1")
    # Inside a trace the frame rides the positions' autograd graph (the trim
    # selection is order statistics: piecewise-linear, a.e. differentiable);
    # the exact path keeps the historical detach byte-for-byte. Values are
    # identical either way -- detach never changes them.
    if tracing_active():
        points = positions.to(device="cpu", dtype=torch.float64)
    else:
        points = positions.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(points).all()) or unit <= 0.0 or not math.isfinite(unit):
        raise ValueError("frame inputs must be finite and unit must be positive")
    count = points.shape[0]
    trim = trim_count(count)
    if count >= N_SMALL:
        ordered, _ = torch.sort(points, dim=0)
        lower = ordered[trim]
        upper = ordered[count - trim - 1]
        center = (lower + upper) / 2.0
        raw_half = (upper - lower) / 2.0
        regime = 1
    else:
        center = _median(points)
        deviation = torch.abs(points - center)
        raw_half = MAD_MULTIPLIER * _median(deviation)
        regime = 2 if count >= N_FLOOR else 3
    floor = torch.full((2,), HALF_EXTENT_FLOOR * unit, dtype=torch.float64)
    bound = raw_half <= floor
    half_extents = torch.maximum(raw_half, floor)
    return RobustFrame(
        center=center.clone(),
        half_extents=half_extents.clone(),
        trim_count=trim,
        regime=regime,
        floor_bound=(bool(bound[0]), bool(bound[1])),
    )


def robust_core_positions(positions: torch.Tensor) -> torch.Tensor:
    """Return the actual point set retained by the robust-frame trim rule.

    Parameters
    ----------
    positions : torch.Tensor
        Finite node centers with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Retained centers. The small-population MAD regimes retain every point;
        the order-statistic regime discards points outside either trimmed axis.
    """

    points = positions.detach().to(device="cpu", dtype=torch.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError("positions must have shape [N, 2] with N >= 1")
    if not bool(torch.isfinite(points).all()):
        raise ValueError("positions must be finite")
    return points[robust_core_mask(points)].clone()


def robust_core_mask(positions: torch.Tensor) -> torch.Tensor:
    """Return the membership mask of the U21 retained point core.

    Parameters
    ----------
    positions : torch.Tensor
        Finite node centers with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Boolean retained-core mask with shape ``[N]``.
    """

    points = positions.detach().to(device="cpu", dtype=torch.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError("positions must have shape [N, 2] with N >= 1")
    if not bool(torch.isfinite(points).all()):
        raise ValueError("positions must be finite")
    count = points.shape[0]
    if count < N_SMALL:
        return torch.ones(count, dtype=torch.bool)
    trim = trim_count(count)
    ordered, _ = torch.sort(points, dim=0)
    lower = ordered[trim]
    upper = ordered[count - trim - 1]
    return torch.all((points >= lower) & (points <= upper), dim=1)


def robust_projection(projections: torch.Tensor, unit: float) -> RobustProjection:
    """Apply the U21 frame rule in one fixed input-owned direction.

    Parameters
    ----------
    projections : torch.Tensor
        Finite scalar projections with shape ``[N]``.
    unit : float
        Positive intrinsic unit.

    Returns
    -------
    RobustProjection
        Robust projected center and universally floored half-width.
    """

    if projections.ndim != 1 or projections.numel() == 0:
        raise ValueError("projections must have shape [N] with N >= 1")
    values = projections.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(values).all()) or unit <= 0.0 or not math.isfinite(unit):
        raise ValueError("projection inputs must be finite and unit must be positive")
    count = values.numel()
    if count >= N_SMALL:
        ordered, _ = torch.sort(values)
        trim = trim_count(count)
        lower = float(ordered[trim])
        upper = float(ordered[count - trim - 1])
        center = (lower + upper) / 2.0
        raw_half = (upper - lower) / 2.0
    else:
        center_tensor = _median(values[:, None])[0]
        center = float(center_tensor)
        deviations = torch.abs(values - center_tensor)[:, None]
        raw_half = MAD_MULTIPLIER * float(_median(deviations)[0])
    floor = HALF_EXTENT_FLOOR * unit
    return RobustProjection(center, max(raw_half, floor), raw_half <= floor)


def box_outside_area(
    box_center: torch.Tensor,
    box_half_extents: torch.Tensor,
    frame: RobustFrame,
) -> float:
    """Compute exact axis-aligned primitive area outside a robust frame.

    Parameters
    ----------
    box_center : torch.Tensor
        Primitive center with shape ``[2]``.
    box_half_extents : torch.Tensor
        Primitive half extents with shape ``[2]``.
    frame : RobustFrame
        Input robust frame.

    Returns
    -------
    float
        Nonnegative escaped primitive area.
    """

    box_low = box_center - box_half_extents
    box_high = box_center + box_half_extents
    frame_low = frame.center - frame.half_extents
    frame_high = frame.center + frame.half_extents
    overlap_extent = torch.clamp(
        torch.minimum(box_high, frame_high) - torch.maximum(box_low, frame_low), min=0.0
    )
    total = 4.0 * torch.prod(box_half_extents)
    inside = torch.prod(overlap_extent)
    return float(torch.clamp(total - inside, min=0.0).item())


def overflow_anchor(scene: Scene, frame: RobustFrame) -> float:
    """Compute the input-anchored achievable minimum overflow from U21.

    Parameters
    ----------
    scene : Scene
        Validated scene and StyleContract-derived primitives.
    frame : RobustFrame
        Robust frame whose explicit trim count selects the anchor.

    Returns
    -------
    float
        Closed-form escaped-mass anchor normalized by reference core area.
    """

    count = scene.node_count
    grid_side = math.ceil(math.sqrt(count))
    trim = frame.trim_count
    surviving_side = max(0, grid_side - 2 * trim)
    inside_count = min(count, surviving_side**2)
    outside_count = count - inside_count
    primitive_area = sum(
        float(4.0 * torch.prod(box.half_extents).item()) for box in scene.node_boxes
    )
    mean_area = primitive_area / count
    max_diagonal = max(
        float(2.0 * torch.linalg.vector_norm(box.half_extents).item()) for box in scene.node_boxes
    )
    core_steps = grid_side - 1 - 2 * trim
    if core_steps <= 0:
        # U21 defines the below-floor value by continuity at N_small.  Reusing the
        # N_small grid geometry keeps the anchor input-only and avoids a zero area.
        grid_side = math.ceil(math.sqrt(N_SMALL))
        reference_trim = trim_count(N_SMALL)
        surviving_side = max(0, grid_side - 2 * reference_trim)
        inside_count = min(N_SMALL, surviving_side**2)
        outside_count = N_SMALL - inside_count
        core_steps = grid_side - 1 - 2 * reference_trim
    core_area = (core_steps * max_diagonal) ** 2
    return float((outside_count * mean_area) / core_area)


def overflow_defect(scene: Scene, frame: RobustFrame) -> Tuple[float, float, float]:
    """Compute U21 escaped mass and its input-anchored smooth defect.

    Parameters
    ----------
    scene : Scene
        Validated scene.
    frame : RobustFrame
        Robust frame.

    Returns
    -------
    tuple[float, float, float]
        Raw escaped mass per frame area, anchor, and bounded defect.
    """

    node_masses = scene.graph.node_masses or tuple(1.0 for _ in range(scene.node_count))
    escaped_area = sum(
        float(node_masses[box.owner]) * box_outside_area(box.center, box.half_extents, frame)
        for box in scene.node_boxes
    )
    for route in resolved_routes(scene):
        stroke_width = (
            scene.style.edge_stroke_widths[route.edge_index]
            if scene.style.edge_stroke_widths
            else scene.style.route_stroke_width * scene.style.coordinate_scale
        )
        for start, end in zip(route.points[:-1], route.points[1:]):
            segment_length = float(torch.linalg.vector_norm(end - start).item())
            inside_length = _segment_length_inside_frame(start, end, frame)
            escaped_area += max(0.0, segment_length - inside_length) * stroke_width
    # This defect pipeline is float-valued this seam version (escaped areas
    # accumulate as floats), so the frame area is a detached read here.
    mass_out = escaped_area / as_float(frame.area)
    anchor = overflow_anchor(scene, frame)
    excess = max(0.0, mass_out - anchor)
    smooth = 0.0 if excess <= 0.0 else excess**2 / (excess + 0.05)
    return mass_out, anchor, smooth / (1.0 + smooth)


def declared_content_area(scene: Scene) -> float:
    """Return the port's declared-primitive content-area approximation.

    Parameters
    ----------
    scene : Scene
        Validated scene.

    Returns
    -------
    float
        Sum of node areas and flattened route-ribbon areas.

    Notes
    -----
    Exact overlapping-primitive union certification remains recorded in
    ``DISCREPANCIES.md``; this published statistic uses the same approximation
    as escaped mass.
    """

    area = sum(float(4.0 * torch.prod(box.half_extents)) for box in scene.node_boxes)
    for route in resolved_routes(scene):
        stroke_width = (
            scene.style.edge_stroke_widths[route.edge_index]
            if scene.style.edge_stroke_widths
            else scene.style.route_stroke_width * scene.style.coordinate_scale
        )
        area += sum(
            float(torch.linalg.vector_norm(end - start)) * stroke_width
            for start, end in zip(route.points[:-1], route.points[1:])
        )
    return area


def point_hull_area(points: torch.Tensor) -> float:
    """Return the exact convex-hull area of two-dimensional points.

    Parameters
    ----------
    points : torch.Tensor
        Point coordinates with shape ``[N, 2]``.

    Returns
    -------
    float
        Nonnegative hull area, zero for fewer than three distinct points.
    """

    unique = sorted(set((float(point[0]), float(point[1])) for point in points))
    if len(unique) < 3:
        return 0.0

    def cross(
        origin: Tuple[float, float], left: Tuple[float, float], right: Tuple[float, float]
    ) -> float:
        """Return the signed turn of three points.

        Parameters
        ----------
        origin, left, right : tuple[float, float]
            Two-dimensional coordinates.

        Returns
        -------
        float
            Signed twice-triangle area.
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
    hull = lower[:-1] + upper[:-1]
    twice_area = sum(
        left[0] * right[1] - left[1] * right[0] for left, right in zip(hull, hull[1:] + hull[:1])
    )
    return abs(twice_area) / 2.0


def _segment_length_inside_frame(
    start: torch.Tensor, end: torch.Tensor, frame: RobustFrame
) -> float:
    """Clip a segment to a robust frame and return its retained length.

    Parameters
    ----------
    start, end : torch.Tensor
        Segment endpoints with shape ``[2]``.
    frame : RobustFrame
        Axis-aligned clipping frame.

    Returns
    -------
    float
        Segment length lying inside the frame rectangle.
    """

    direction = end - start
    lower = 0.0
    upper = 1.0
    frame_low = frame.center - frame.half_extents
    frame_high = frame.center + frame.half_extents
    for axis in range(2):
        delta = float(direction[axis])
        origin = float(start[axis])
        low = float(frame_low[axis])
        high = float(frame_high[axis])
        if delta == 0.0:
            if origin < low or origin > high:
                return 0.0
            continue
        first = (low - origin) / delta
        second = (high - origin) / delta
        lower = max(lower, min(first, second))
        upper = min(upper, max(first, second))
        if lower >= upper:
            return 0.0
    return (upper - lower) * float(torch.linalg.vector_norm(direction).item())
