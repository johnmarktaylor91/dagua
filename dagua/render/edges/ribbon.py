"""Ribbon generation for data-coordinate edge bodies."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np
from matplotlib.path import Path
from numpy.typing import NDArray

from dagua.render.edges.geometry import (
    FLOAT_EPSILON,
    CubicBezier,
    Point,
    adaptive_subdivide,
    as_point,
    perpendicular,
    polyline_from_samples,
    unit_vector,
)

DEFAULT_CAP_SEGMENTS = 8
DEFAULT_MITER_LIMIT = 4.0
DEFAULT_RIBBON_FLATNESS = 0.12


def line_intersection(
    point_a: Point,
    direction_a: Point,
    point_b: Point,
    direction_b: Point,
) -> Optional[Point]:
    """Intersect two infinite 2D lines.

    Parameters
    ----------
    point_a : numpy.ndarray
        Point on the first line with shape ``[2]``.
    direction_a : numpy.ndarray
        Direction of the first line with shape ``[2]``.
    point_b : numpy.ndarray
        Point on the second line with shape ``[2]``.
    direction_b : numpy.ndarray
        Direction of the second line with shape ``[2]``.

    Returns
    -------
    numpy.ndarray | None
        Intersection point, or ``None`` when the lines are nearly parallel.
    """
    denominator = float(np.cross(direction_a, direction_b))
    if abs(denominator) <= FLOAT_EPSILON:
        return None
    delta = point_b - point_a
    t = float(np.cross(delta, direction_b) / denominator)
    return point_a + direction_a * t


def _polyline_directions(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute unit segment directions for a polyline.

    Parameters
    ----------
    points : numpy.ndarray
        Polyline points with shape ``[N, 2]``.

    Returns
    -------
    numpy.ndarray
        Segment directions with shape ``[N - 1, 2]``.
    """
    if points.shape[0] < 2:
        raise ValueError("At least two points are required to build a ribbon.")
    deltas = np.diff(points, axis=0)
    directions = np.zeros_like(deltas)
    fallback = np.array([1.0, 0.0], dtype=np.float64)
    for index, delta in enumerate(deltas):
        directions[index] = unit_vector(delta, fallback=fallback)
        fallback = directions[index]
    return directions


def _offset_join_points(
    points: NDArray[np.float64],
    half_width: float,
    side: float,
    join_style: str,
    miter_limit: float,
) -> List[Point]:
    """Build one outline of the ribbon with miter or bevel joins.

    Parameters
    ----------
    points : numpy.ndarray
        Polyline vertices with shape ``[N, 2]``.
    half_width : float
        Half of the ribbon width in data units.
    side : float
        ``+1`` for the left outline and ``-1`` for the right outline.
    join_style : str
        Either ``"miter"`` or ``"bevel"``.
    miter_limit : float
        Maximum allowed miter length relative to ``half_width``.

    Returns
    -------
    list[numpy.ndarray]
        Ordered outline points.
    """
    directions = _polyline_directions(points)
    normals = np.vstack([perpendicular(direction) for direction in directions]) * side
    outline: List[Point] = [points[0] + normals[0] * half_width]

    for index in range(1, points.shape[0] - 1):
        point = points[index]
        prev_direction = directions[index - 1]
        next_direction = directions[index]
        prev_normal = normals[index - 1]
        next_normal = normals[index]

        prev_offset = point + prev_normal * half_width
        next_offset = point + next_normal * half_width
        intersection = line_intersection(prev_offset, prev_direction, next_offset, next_direction)
        if intersection is None:
            outline.extend([prev_offset, next_offset])
            continue

        miter_length = float(np.linalg.norm(intersection - point))
        if join_style == "miter" and miter_length <= half_width * max(miter_limit, 1.0):
            outline.append(intersection)
        else:
            outline.extend([prev_offset, next_offset])

    outline.append(points[-1] + normals[-1] * half_width)
    return outline


def _round_cap(
    center: Point,
    tangent: Point,
    half_width: float,
    at_end: bool,
    n_segments: int = DEFAULT_CAP_SEGMENTS,
) -> List[Point]:
    """Approximate a semicircular cap with line segments.

    Parameters
    ----------
    center : numpy.ndarray
        Cap center with shape ``[2]``.
    tangent : numpy.ndarray
        Tangent direction at the end point with shape ``[2]``.
    half_width : float
        Cap radius in data units.
    at_end : bool
        Whether the cap is for the end of the ribbon.
    n_segments : int, default=8
        Number of linear segments used for the arc.

    Returns
    -------
    list[numpy.ndarray]
        Points from left edge to right edge along the cap.
    """
    if half_width <= FLOAT_EPSILON:
        return [center]
    unit_tangent = unit_vector(tangent)
    normal = perpendicular(unit_tangent)
    points: List[Point] = []
    angles = np.linspace(0.0, math.pi, n_segments + 1, dtype=np.float64)
    for angle in angles:
        if at_end:
            point = (
                center
                + normal * (math.cos(angle) * half_width)
                + unit_tangent * (math.sin(angle) * half_width)
            )
        else:
            point = (
                center
                + normal * (math.cos(angle) * half_width)
                - unit_tangent * (math.sin(angle) * half_width)
            )
        points.append(point)
    return points


def polyline_ribbon_path(
    points: NDArray[np.float64],
    width: float,
    cap_start: str = "butt",
    cap_end: str = "butt",
    join_style: str = "miter",
    miter_limit: float = DEFAULT_MITER_LIMIT,
) -> Path:
    """Convert a centerline polyline into a closed ribbon path.

    Parameters
    ----------
    points : numpy.ndarray
        Centerline vertices with shape ``[N, 2]``.
    width : float
        Ribbon width in data units.
    cap_start : str, default="butt"
        Start-cap style: ``"butt"`` or ``"round"``.
    cap_end : str, default="butt"
        End-cap style: ``"butt"`` or ``"round"``.
    join_style : str, default="miter"
        Corner join style: ``"miter"`` or ``"bevel"``.
    miter_limit : float, default=4.0
        Miter limit expressed as a multiple of half-width.

    Returns
    -------
    matplotlib.path.Path
        Closed body polygon.
    """
    if points.shape[0] < 2:
        raise ValueError("At least two points are required to build a ribbon.")
    half_width = max(float(width) * 0.5, FLOAT_EPSILON)
    directions = _polyline_directions(points)
    left_outline = _offset_join_points(points, half_width, 1.0, join_style, miter_limit)
    right_outline = _offset_join_points(points, half_width, -1.0, join_style, miter_limit)

    polygon: List[Point] = list(left_outline)
    if cap_end == "round":
        polygon.extend(_round_cap(points[-1], directions[-1], half_width, at_end=True)[1:])
    else:
        polygon.append(right_outline[-1])

    polygon.extend(reversed(right_outline[:-1]))
    if cap_start == "round":
        start_cap = _round_cap(points[0], directions[0], half_width, at_end=False)
        polygon.extend(reversed(start_cap[:-1]))

    vertices = np.vstack([*polygon, polygon[0]])
    codes = [Path.MOVETO] + [Path.LINETO] * (vertices.shape[0] - 2) + [Path.CLOSEPOLY]
    return Path(vertices, codes)


def curve_ribbon_path(
    curve: CubicBezier,
    width: float,
    flatness: float = DEFAULT_RIBBON_FLATNESS,
    cap_start: str = "butt",
    cap_end: str = "butt",
    join_style: str = "miter",
    miter_limit: float = DEFAULT_MITER_LIMIT,
) -> Path:
    """Build a ribbon path from a cubic bezier centerline.

    Parameters
    ----------
    curve : CubicBezier
        Curve centerline.
    width : float
        Ribbon width in data units.
    flatness : float, default=0.12
        Maximum adaptive-subdivision flatness in data units.
    cap_start : str, default="butt"
        Start-cap style.
    cap_end : str, default="butt"
        End-cap style.
    join_style : str, default="miter"
        Join style used at subdivision corners.
    miter_limit : float, default=4.0
        Miter limit relative to half-width.

    Returns
    -------
    matplotlib.path.Path
        Closed ribbon path.
    """
    samples = adaptive_subdivide(curve, flatness=flatness)
    points = polyline_from_samples(samples)
    return polyline_ribbon_path(
        points=points,
        width=width,
        cap_start=cap_start,
        cap_end=cap_end,
        join_style=join_style,
        miter_limit=miter_limit,
    )


def simple_quad_ribbon(curve: CubicBezier, width: float) -> Path:
    """Build a simplified two-segment ribbon for high edge counts.

    Parameters
    ----------
    curve : CubicBezier
        Curve centerline. Only the endpoints are used.
    width : float
        Ribbon width in data units.

    Returns
    -------
    matplotlib.path.Path
        Closed quadrilateral ribbon.
    """
    points = np.vstack([curve.p0, curve.p1])
    return polyline_ribbon_path(points, width=width, cap_start="butt", cap_end="butt")


def contour_endpoints(path: Path) -> NDArray[np.float64]:
    """Extract the first two vertices of a trim contour.

    Parameters
    ----------
    path : matplotlib.path.Path
        Trim contour path.

    Returns
    -------
    numpy.ndarray
        First two contour points with shape ``[2, 2]``.
    """
    vertices = np.asarray(path.vertices, dtype=np.float64)
    if vertices.shape[0] < 2:
        raise ValueError("Trim contour must contain at least two vertices.")
    return vertices[:2]


def trim_polyline_end(points: NDArray[np.float64], contour: Optional[Path]) -> NDArray[np.float64]:
    """Replace the final centerline point using a trim contour midpoint.

    Parameters
    ----------
    points : numpy.ndarray
        Centerline points with shape ``[N, 2]``.
    contour : matplotlib.path.Path | None
        Optional body/head join contour.

    Returns
    -------
    numpy.ndarray
        Updated centerline points.
    """
    if contour is None or points.shape[0] == 0:
        return points
    endpoints = contour_endpoints(contour)
    trimmed = np.array(points, copy=True)
    trimmed[-1] = endpoints.mean(axis=0)
    return trimmed


def points_to_path(points: Sequence[Sequence[float]], closed: bool = False) -> Path:
    """Build a simple polyline path.

    Parameters
    ----------
    points : Sequence[Sequence[float]]
        Polyline vertices.
    closed : bool, default=False
        Whether to close the shape.

    Returns
    -------
    matplotlib.path.Path
        Linear path.
    """
    array = np.vstack([as_point(point) for point in points])
    if closed:
        vertices = np.vstack([array, array[0]])
        codes = [Path.MOVETO] + [Path.LINETO] * (vertices.shape[0] - 2) + [Path.CLOSEPOLY]
        return Path(vertices, codes)
    codes = [Path.MOVETO] + [Path.LINETO] * (array.shape[0] - 1)
    return Path(array, codes)
