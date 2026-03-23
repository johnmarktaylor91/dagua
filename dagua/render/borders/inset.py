"""Inset helpers for data-coordinate node and cluster borders."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from matplotlib.path import Path
from numpy.typing import NDArray

from dagua.render.borders.shapes import (
    ShapeSpec,
    build_shape_path,
    closed_path_from_vertices,
    polygon_vertices,
)

FloatArray = NDArray[np.float64]
MAX_BORDER_FRACTION = 0.4
DEFAULT_MITER_LIMIT = 4.0
MIN_INNER_SCALE = 0.1
FLOAT_EPSILON = 1e-9


def clamp_border_width(
    border_width: float,
    width: float,
    height: float,
    max_fraction: float = MAX_BORDER_FRACTION,
) -> float:
    """Clamp a border width so the inset shape always retains interior area.

    Parameters
    ----------
    border_width : float
        Requested border width in data units.
    width : float
        Outer width in data units.
    height : float
        Outer height in data units.
    max_fraction : float, default=0.4
        Maximum fraction of the smaller dimension that one side of the border
        may consume.

    Returns
    -------
    float
        Clamped border width in data units.
    """

    maximum = min(float(width), float(height)) * max_fraction / 2.0
    return max(0.0, min(float(border_width), maximum))


def polygon_signed_area(vertices: FloatArray) -> float:
    """Return the signed area of a polygon.

    Parameters
    ----------
    vertices : numpy.ndarray
        Polygon vertices with shape ``[N, 2]``.

    Returns
    -------
    float
        Positive for counter-clockwise polygons and negative for clockwise.
    """

    rolled = np.roll(vertices, -1, axis=0)
    return 0.5 * float(np.sum(vertices[:, 0] * rolled[:, 1] - vertices[:, 1] * rolled[:, 0]))


def _line_intersection(
    point_a: FloatArray,
    direction_a: FloatArray,
    point_b: FloatArray,
    direction_b: FloatArray,
) -> FloatArray | None:
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
        Intersection point, or ``None`` for nearly parallel lines.
    """

    denominator = float(np.cross(direction_a, direction_b))
    if abs(denominator) <= FLOAT_EPSILON:
        return None
    delta = point_b - point_a
    scale = float(np.cross(delta, direction_b) / denominator)
    return point_a + direction_a * scale


def _unit(vector: FloatArray, fallback: Tuple[float, float] = (1.0, 0.0)) -> FloatArray:
    """Return a normalized 2D vector.

    Parameters
    ----------
    vector : numpy.ndarray
        Input vector with shape ``[2]``.
    fallback : tuple[float, float], default=(1.0, 0.0)
        Replacement direction when the norm is tiny.

    Returns
    -------
    numpy.ndarray
        Unit-length vector with shape ``[2]``.
    """

    norm = float(np.linalg.norm(vector))
    if norm <= FLOAT_EPSILON:
        return np.asarray(fallback, dtype=np.float64)
    return vector / norm


def inset_convex_polygon(
    vertices: FloatArray,
    border_width: float,
    miter_limit: float = DEFAULT_MITER_LIMIT,
) -> FloatArray:
    """Inset a convex polygon by offsetting each edge inward.

    Parameters
    ----------
    vertices : numpy.ndarray
        Polygon vertices with shape ``[N, 2]``.
    border_width : float
        Inset distance in data units.
    miter_limit : float, default=4.0
        Maximum allowed miter length as a multiple of the border width.

    Returns
    -------
    numpy.ndarray
        Inset polygon vertices with shape ``[N, 2]``.
    """

    points = np.asarray(vertices, dtype=np.float64)
    if polygon_signed_area(points) < 0.0:
        points = points[::-1].copy()

    edge_vectors = np.roll(points, -1, axis=0) - points
    edge_directions = np.vstack([_unit(edge) for edge in edge_vectors])
    inward_normals = np.column_stack([-edge_directions[:, 1], edge_directions[:, 0]])
    inset_points: list[FloatArray] = []

    for index in range(points.shape[0]):
        point = points[index]
        prev_index = (index - 1) % points.shape[0]
        next_index = index
        prev_direction = edge_directions[prev_index]
        next_direction = edge_directions[next_index]
        prev_normal = inward_normals[prev_index]
        next_normal = inward_normals[next_index]
        prev_offset = point + prev_normal * border_width
        next_offset = point + next_normal * border_width
        intersection = _line_intersection(prev_offset, prev_direction, next_offset, next_direction)
        if intersection is None:
            inset_points.append((prev_offset + next_offset) / 2.0)
            continue
        miter_length = float(np.linalg.norm(intersection - point))
        if miter_length > max(border_width * max(miter_limit, 1.0), border_width + FLOAT_EPSILON):
            inset_points.append((prev_offset + next_offset) / 2.0)
            continue
        inset_points.append(intersection)

    return np.vstack(inset_points)


def inset_star(vertices: FloatArray, border_width: float) -> FloatArray:
    """Inset the star shape by uniform centroid scaling.

    Parameters
    ----------
    vertices : numpy.ndarray
        Star vertices with shape ``[10, 2]``.
    border_width : float
        Inset distance in data units.

    Returns
    -------
    numpy.ndarray
        Scaled star vertices with shape ``[10, 2]``.
    """

    centroid = np.mean(vertices, axis=0)
    directions = vertices - centroid
    distances = np.linalg.norm(directions, axis=1, keepdims=True)
    safe_distances = np.where(distances > FLOAT_EPSILON, distances, 1.0)
    inner_distances = np.maximum(distances - border_width, distances * MIN_INNER_SCALE)
    return centroid + directions * (inner_distances / safe_distances)


def inset_shape_path(spec: ShapeSpec, border_width: float) -> Path:
    """Return the inset outline path for one shape.

    Parameters
    ----------
    spec : ShapeSpec
        Outer shape description.
    border_width : float
        Inset distance in data units.

    Returns
    -------
    matplotlib.path.Path
        Inset shape path in data coordinates.
    """

    inset = clamp_border_width(border_width, spec.width, spec.height)
    if inset <= FLOAT_EPSILON:
        return build_shape_path(spec)

    inner_width = max(spec.width - 2.0 * inset, spec.width * MIN_INNER_SCALE)
    inner_height = max(spec.height - 2.0 * inset, spec.height * MIN_INNER_SCALE)
    if spec.shape in {"roundrect", "rect"}:
        inner_spec = ShapeSpec(
            center_x=spec.center_x,
            center_y=spec.center_y,
            width=inner_width,
            height=inner_height,
            shape=spec.shape,
            corner_radius=max(spec.corner_radius - inset, 0.0),
        )
        return build_shape_path(inner_spec)
    if spec.shape in {"ellipse", "circle", "cylinder"}:
        inner_spec = ShapeSpec(
            center_x=spec.center_x,
            center_y=spec.center_y,
            width=inner_width,
            height=inner_height,
            shape=spec.shape,
            corner_radius=spec.corner_radius,
        )
        return build_shape_path(inner_spec)
    try:
        vertices = polygon_vertices(spec)
    except ValueError:
        # Curved and compound shapes do not expose polygon vertices, so inset
        # them by rebuilding the same shape against the smaller inner bounds.
        inner_spec = ShapeSpec(
            center_x=spec.center_x,
            center_y=spec.center_y,
            width=inner_width,
            height=inner_height,
            shape=spec.shape,
            corner_radius=spec.corner_radius,
        )
        return build_shape_path(inner_spec)
    if spec.shape == "star":
        return closed_path_from_vertices(inset_star(vertices, inset))
    return closed_path_from_vertices(inset_convex_polygon(vertices, inset))
