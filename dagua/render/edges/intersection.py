"""Ray-shape intersection helpers for node boundary clipping."""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence

import numpy as np

from dagua.render.edges.geometry import FLOAT_EPSILON, Point, as_point


def _smallest_positive(values: Iterable[float]) -> Optional[float]:
    """Return the smallest non-negative finite value.

    Parameters
    ----------
    values : Iterable[float]
        Candidate values.

    Returns
    -------
    float | None
        Minimum valid value, or ``None`` when no valid candidate exists.
    """
    candidates = [
        float(value) for value in values if np.isfinite(value) and value >= -FLOAT_EPSILON
    ]
    if not candidates:
        return None
    return min(candidates)


def ray_rect_intersection(
    center: Sequence[float],
    half_size: Sequence[float],
    ray_origin: Sequence[float],
    ray_direction: Sequence[float],
) -> Point:
    """Intersect a ray with an axis-aligned rectangle.

    Parameters
    ----------
    center : Sequence[float]
        Rectangle center.
    half_size : Sequence[float]
        Half-width and half-height.
    ray_origin : Sequence[float]
        Ray origin.
    ray_direction : Sequence[float]
        Ray direction.

    Returns
    -------
    numpy.ndarray
        Boundary intersection point.
    """
    center_point = as_point(center)
    half = as_point(half_size)
    origin = as_point(ray_origin) - center_point
    direction = as_point(ray_direction)
    candidates: List[float] = []

    if abs(direction[0]) > FLOAT_EPSILON:
        for x in (-half[0], half[0]):
            t = (x - origin[0]) / direction[0]
            y = origin[1] + t * direction[1]
            if t >= -FLOAT_EPSILON and -half[1] - FLOAT_EPSILON <= y <= half[1] + FLOAT_EPSILON:
                candidates.append(float(t))
    if abs(direction[1]) > FLOAT_EPSILON:
        for y in (-half[1], half[1]):
            t = (y - origin[1]) / direction[1]
            x = origin[0] + t * direction[0]
            if t >= -FLOAT_EPSILON and -half[0] - FLOAT_EPSILON <= x <= half[0] + FLOAT_EPSILON:
                candidates.append(float(t))

    t_hit = _smallest_positive(candidates)
    if t_hit is None:
        return center_point
    return center_point + origin + direction * t_hit


def ray_ellipse_intersection(
    center: Sequence[float],
    half_size: Sequence[float],
    ray_origin: Sequence[float],
    ray_direction: Sequence[float],
) -> Point:
    """Intersect a ray with an axis-aligned ellipse.

    Parameters
    ----------
    center : Sequence[float]
        Ellipse center.
    half_size : Sequence[float]
        Semi-major and semi-minor axes.
    ray_origin : Sequence[float]
        Ray origin.
    ray_direction : Sequence[float]
        Ray direction.

    Returns
    -------
    numpy.ndarray
        Boundary intersection point.
    """
    center_point = as_point(center)
    half = as_point(half_size)
    origin = as_point(ray_origin) - center_point
    direction = as_point(ray_direction)
    if half[0] <= FLOAT_EPSILON or half[1] <= FLOAT_EPSILON:
        return center_point

    a = (direction[0] / half[0]) ** 2 + (direction[1] / half[1]) ** 2
    b = 2.0 * (
        (origin[0] * direction[0]) / (half[0] ** 2) + (origin[1] * direction[1]) / (half[1] ** 2)
    )
    c = (origin[0] / half[0]) ** 2 + (origin[1] / half[1]) ** 2 - 1.0
    roots = np.roots([a, b, c])
    t_hit = _smallest_positive(float(root.real) for root in roots if abs(root.imag) <= 1e-8)
    if t_hit is None:
        return center_point
    return center_point + origin + direction * t_hit


def _polygon_vertices(shape: str, half_size: Point) -> np.ndarray:
    """Return polygon vertices for supported polygonal node shapes.

    All vertices are in center-relative coordinates (center at origin).

    Parameters
    ----------
    shape : str
        Shape name.
    half_size : numpy.ndarray
        Half-width and half-height.

    Returns
    -------
    numpy.ndarray
        Vertices with shape ``[N, 2]``, wound counter-clockwise.
    """
    hw = float(half_size[0])
    hh = float(half_size[1])
    if shape == "diamond":
        return np.array(
            [[0.0, hh], [hw, 0.0], [0.0, -hh], [-hw, 0.0]],
            dtype=np.float64,
        )
    if shape == "triangle":
        # Upward-pointing triangle: apex at top, base at bottom.
        return np.array(
            [[0.0, hh], [hw, -hh], [-hw, -hh]],
            dtype=np.float64,
        )
    if shape == "hexagon":
        # Regular hexagon inscribed in bounding box.
        qw = hw * 0.5  # quarter width for hexagon facets
        return np.array(
            [[qw, hh], [hw, 0.0], [qw, -hh], [-qw, -hh], [-hw, 0.0], [-qw, hh]],
            dtype=np.float64,
        )
    if shape == "pentagon":
        # Regular pentagon inscribed in bounding box.
        angles = [math.pi / 2 + i * 2 * math.pi / 5 for i in range(5)]
        return np.array(
            [[hw * math.cos(a), hh * math.sin(a)] for a in angles],
            dtype=np.float64,
        )
    if shape == "octagon":
        # Clipped-corner rectangle.
        clip = min(hw, hh) * 0.3
        return np.array(
            [
                [hw - clip, hh],
                [hw, hh - clip],
                [hw, -(hh - clip)],
                [hw - clip, -hh],
                [-(hw - clip), -hh],
                [-hw, -(hh - clip)],
                [-hw, hh - clip],
                [-(hw - clip), hh],
            ],
            dtype=np.float64,
        )
    if shape == "parallelogram":
        skew = hw * 0.25
        return np.array(
            [[-hw + skew, hh], [hw + skew, hh], [hw - skew, -hh], [-hw - skew, -hh]],
            dtype=np.float64,
        )
    if shape == "trapezoid":
        inset = hw * 0.25
        return np.array(
            [[-hw + inset, hh], [hw - inset, hh], [hw, -hh], [-hw, -hh]],
            dtype=np.float64,
        )
    if shape == "star":
        # 5-pointed star: alternating outer and inner vertices.
        inner_ratio = 0.32
        verts = []
        for i in range(10):
            angle = math.pi / 2 + i * math.pi / 5
            r = 1.0 if i % 2 == 0 else inner_ratio
            verts.append([hw * r * math.cos(angle), hh * r * math.sin(angle)])
        return np.array(verts, dtype=np.float64)
    raise ValueError(f"Unsupported polygon shape: {shape!r}.")


def _ray_segment_intersection(
    origin: Point, direction: Point, a: Point, b: Point
) -> Optional[float]:
    """Intersect a ray with a finite line segment.

    Parameters
    ----------
    origin : numpy.ndarray
        Ray origin.
    direction : numpy.ndarray
        Ray direction.
    a : numpy.ndarray
        Segment start.
    b : numpy.ndarray
        Segment end.

    Returns
    -------
    float | None
        Ray parameter at the intersection, or ``None``.
    """
    segment = b - a
    denominator = float(np.cross(direction, segment))
    if abs(denominator) <= FLOAT_EPSILON:
        return None
    delta = a - origin
    t = float(np.cross(delta, segment) / denominator)
    u = float(np.cross(delta, direction) / denominator)
    if t < -FLOAT_EPSILON or u < -FLOAT_EPSILON or u > 1.0 + FLOAT_EPSILON:
        return None
    return t


def ray_polygon_intersection(
    center: Sequence[float],
    half_size: Sequence[float],
    shape: str,
    ray_origin: Sequence[float],
    ray_direction: Sequence[float],
) -> Point:
    """Intersect a ray with a polygonal node shape.

    Works for any shape supported by :func:`_polygon_vertices`:
    diamond, triangle, hexagon, pentagon, octagon, parallelogram,
    trapezoid, and star.

    Parameters
    ----------
    center : Sequence[float]
        Node center.
    half_size : Sequence[float]
        Half-width and half-height.
    shape : str
        Polygon shape name.
    ray_origin : Sequence[float]
        Ray origin.
    ray_direction : Sequence[float]
        Ray direction.

    Returns
    -------
    numpy.ndarray
        Boundary intersection point.
    """
    center_point = as_point(center)
    origin = as_point(ray_origin) - center_point
    direction = as_point(ray_direction)
    vertices = _polygon_vertices(shape, as_point(half_size))
    candidates: List[float] = []
    for start, end in zip(vertices, np.vstack([vertices[1:], vertices[:1]])):
        hit = _ray_segment_intersection(origin, direction, start, end)
        if hit is not None:
            candidates.append(hit)
    t_hit = _smallest_positive(candidates)
    if t_hit is None:
        return center_point
    return center_point + origin + direction * t_hit


def ray_diamond_intersection(
    center: Sequence[float],
    half_size: Sequence[float],
    ray_origin: Sequence[float],
    ray_direction: Sequence[float],
) -> Point:
    """Intersect a ray with a diamond node (backward-compatible wrapper)."""
    return ray_polygon_intersection(center, half_size, "diamond", ray_origin, ray_direction)


def ray_roundrect_intersection(
    center: Sequence[float],
    half_size: Sequence[float],
    corner_radius: float,
    ray_origin: Sequence[float],
    ray_direction: Sequence[float],
) -> Point:
    """Intersect a ray with an axis-aligned rounded rectangle.

    Parameters
    ----------
    center : Sequence[float]
        Rounded-rectangle center.
    half_size : Sequence[float]
        Half-width and half-height.
    corner_radius : float
        Corner radius in data units.
    ray_origin : Sequence[float]
        Ray origin.
    ray_direction : Sequence[float]
        Ray direction.

    Returns
    -------
    numpy.ndarray
        Boundary intersection point.
    """
    center_point = as_point(center)
    half = as_point(half_size)
    radius = float(max(min(corner_radius, half[0], half[1]), 0.0))
    if radius <= FLOAT_EPSILON:
        return ray_rect_intersection(center, half_size, ray_origin, ray_direction)

    origin = as_point(ray_origin) - center_point
    direction = as_point(ray_direction)
    candidates: List[float] = []

    inner_x = half[0] - radius
    inner_y = half[1] - radius

    if abs(direction[0]) > FLOAT_EPSILON:
        for x in (-half[0], half[0]):
            t = (x - origin[0]) / direction[0]
            y = origin[1] + t * direction[1]
            if t >= -FLOAT_EPSILON and -inner_y - FLOAT_EPSILON <= y <= inner_y + FLOAT_EPSILON:
                candidates.append(float(t))
    if abs(direction[1]) > FLOAT_EPSILON:
        for y in (-half[1], half[1]):
            t = (y - origin[1]) / direction[1]
            x = origin[0] + t * direction[0]
            if t >= -FLOAT_EPSILON and -inner_x - FLOAT_EPSILON <= x <= inner_x + FLOAT_EPSILON:
                candidates.append(float(t))

    for corner_x in (-inner_x, inner_x):
        for corner_y in (-inner_y, inner_y):
            relative_origin = origin - np.array([corner_x, corner_y], dtype=np.float64)
            a = float(np.dot(direction, direction))
            b = 2.0 * float(np.dot(relative_origin, direction))
            c = float(np.dot(relative_origin, relative_origin) - radius**2)
            for root in np.roots([a, b, c]):
                if abs(root.imag) > 1e-8:
                    continue
                t = float(root.real)
                if t < -FLOAT_EPSILON:
                    continue
                point = origin + direction * t
                if (
                    abs(point[0]) >= inner_x - FLOAT_EPSILON
                    and abs(point[1]) >= inner_y - FLOAT_EPSILON
                    and np.sign(point[0]) == np.sign(corner_x)
                    and np.sign(point[1]) == np.sign(corner_y)
                ):
                    candidates.append(t)

    t_hit = _smallest_positive(candidates)
    if t_hit is None:
        return center_point
    return center_point + origin + direction * t_hit


def intersect_node_boundary(
    center: Sequence[float],
    half_size: Sequence[float],
    shape: str,
    corner_radius: float,
    ray_origin: Sequence[float],
    ray_direction: Sequence[float],
) -> Point:
    """Intersect a ray with a supported node boundary.

    Parameters
    ----------
    center : Sequence[float]
        Node center.
    half_size : Sequence[float]
        Half-width and half-height.
    shape : str
        Node shape.
    corner_radius : float
        Round-rectangle corner radius.
    ray_origin : Sequence[float]
        Ray origin.
    ray_direction : Sequence[float]
        Ray direction.

    Returns
    -------
    numpy.ndarray
        Boundary intersection point.
    """
    if shape == "rect":
        return ray_rect_intersection(center, half_size, ray_origin, ray_direction)
    if shape == "roundrect":
        return ray_roundrect_intersection(
            center, half_size, corner_radius, ray_origin, ray_direction
        )
    if shape in {"ellipse", "circle", "double_circle"}:
        return ray_ellipse_intersection(center, half_size, ray_origin, ray_direction)
    # All polygon-based shapes use the generic polygon intersection.
    _POLYGON_SHAPES = {
        "diamond",
        "triangle",
        "hexagon",
        "pentagon",
        "octagon",
        "parallelogram",
        "trapezoid",
        "star",
    }
    if shape in _POLYGON_SHAPES:
        return ray_polygon_intersection(center, half_size, shape, ray_origin, ray_direction)
    return ray_rect_intersection(center, half_size, ray_origin, ray_direction)
