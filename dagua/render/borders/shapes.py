"""Shape-path helpers for data-coordinate node and cluster borders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch
from matplotlib.path import Path
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ShapeSpec:
    """Describe one node or cluster outline in data coordinates.

    Parameters
    ----------
    center_x : float
        Shape center x-coordinate.
    center_y : float
        Shape center y-coordinate.
    width : float
        Outer width in data units.
    height : float
        Outer height in data units.
    shape : str
        Dagua shape name.
    corner_radius : float, default=0.0
        Rounded-rectangle corner radius in data units.
    """

    center_x: float
    center_y: float
    width: float
    height: float
    shape: str
    corner_radius: float = 0.0


def triangle_vertices(center_x: float, center_y: float, width: float, height: float) -> FloatArray:
    """Return the wide Graphviz-like triangle vertices.

    Parameters
    ----------
    center_x : float
        Shape center x-coordinate.
    center_y : float
        Shape center y-coordinate.
    width : float
        Bounding-box width in data units.
    height : float
        Bounding-box height in data units.

    Returns
    -------
    numpy.ndarray
        Vertices with shape ``[3, 2]``.
    """

    half_width = width / 2.0
    half_height = height / 2.0
    return np.array(
        [
            [center_x, center_y + half_height],
            [center_x + half_width, center_y - half_height],
            [center_x - half_width, center_y - half_height],
        ],
        dtype=np.float64,
    )


def regular_polygon_vertices(
    num_vertices: int,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    rotation: float = np.pi / 2.0,
) -> FloatArray:
    """Return vertices for a regular polygon inscribed in a bounding box.

    Parameters
    ----------
    num_vertices : int
        Number of polygon corners.
    center_x : float
        Shape center x-coordinate.
    center_y : float
        Shape center y-coordinate.
    width : float
        Bounding-box width in data units.
    height : float
        Bounding-box height in data units.
    rotation : float, default=pi/2
        Initial rotation in radians.

    Returns
    -------
    numpy.ndarray
        Vertices with shape ``[num_vertices, 2]``.
    """

    angles = rotation + (2.0 * np.pi * np.arange(num_vertices, dtype=np.float64) / num_vertices)
    return np.column_stack(
        [
            center_x + (width / 2.0) * np.cos(angles),
            center_y + (height / 2.0) * np.sin(angles),
        ]
    ).astype(np.float64)


def star_vertices(center_x: float, center_y: float, width: float, height: float) -> FloatArray:
    """Return the five-point star vertices used by the matplotlib renderer.

    Parameters
    ----------
    center_x : float
        Shape center x-coordinate.
    center_y : float
        Shape center y-coordinate.
    width : float
        Bounding-box width in data units.
    height : float
        Bounding-box height in data units.

    Returns
    -------
    numpy.ndarray
        Vertices with shape ``[10, 2]``.
    """

    points: List[List[float]] = []
    outer_rx = width / 2.0
    outer_ry = height / 2.0
    inner_rx = outer_rx * 0.32
    inner_ry = outer_ry * 0.32
    for index in range(10):
        angle = np.pi / 2.0 + index * np.pi / 5.0
        radius_x = outer_rx if index % 2 == 0 else inner_rx
        radius_y = outer_ry if index % 2 == 0 else inner_ry
        points.append(
            [
                center_x + radius_x * np.cos(angle),
                center_y + radius_y * np.sin(angle),
            ]
        )
    return np.asarray(points, dtype=np.float64)


def polygon_vertices(spec: ShapeSpec) -> FloatArray:
    """Return polygon vertices for polygonal node shapes.

    Parameters
    ----------
    spec : ShapeSpec
        Shape description.

    Returns
    -------
    numpy.ndarray
        Vertices with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If ``spec.shape`` is not polygonal.
    """

    x = float(spec.center_x)
    y = float(spec.center_y)
    width = float(spec.width)
    height = float(spec.height)
    if spec.shape == "diamond":
        return np.array(
            [
                [x, y + height / 2.0],
                [x + width / 2.0, y],
                [x, y - height / 2.0],
                [x - width / 2.0, y],
            ],
            dtype=np.float64,
        )
    if spec.shape == "triangle":
        return triangle_vertices(x, y, width, height)
    if spec.shape == "hexagon":
        return regular_polygon_vertices(6, x, y, width, height)
    if spec.shape == "pentagon":
        return regular_polygon_vertices(5, x, y, width, height)
    if spec.shape == "octagon":
        return regular_polygon_vertices(8, x, y, width, height, rotation=np.pi / 8.0)
    if spec.shape == "star":
        return star_vertices(x, y, width, height)
    if spec.shape == "parallelogram":
        skew = width * 0.28
        return np.array(
            [
                [x - width / 2.0 + skew, y + height / 2.0],
                [x + width / 2.0, y + height / 2.0],
                [x + width / 2.0 - skew, y - height / 2.0],
                [x - width / 2.0, y - height / 2.0],
            ],
            dtype=np.float64,
        )
    if spec.shape == "trapezoid":
        inset = width * 0.28
        return np.array(
            [
                [x - width / 2.0 + inset, y + height / 2.0],
                [x + width / 2.0 - inset, y + height / 2.0],
                [x + width / 2.0, y - height / 2.0],
                [x - width / 2.0, y - height / 2.0],
            ],
            dtype=np.float64,
        )
    raise ValueError(f"Shape {spec.shape!r} does not have polygon vertices.")


def closed_path_from_vertices(vertices: FloatArray) -> Path:
    """Build a closed linear path from polygon vertices.

    Parameters
    ----------
    vertices : numpy.ndarray
        Polygon vertices with shape ``[N, 2]``.

    Returns
    -------
    matplotlib.path.Path
        Closed path with one subpath.
    """

    if vertices.shape[0] < 3:
        raise ValueError("A closed polygon requires at least three vertices.")
    closed_vertices = np.vstack([vertices, vertices[0:1]])
    codes = [Path.MOVETO] + [Path.LINETO] * (closed_vertices.shape[0] - 2) + [Path.CLOSEPOLY]
    return Path(closed_vertices, codes)


def cylinder_path(spec: ShapeSpec) -> Path:
    """Return the native cylinder outline path.

    Parameters
    ----------
    spec : ShapeSpec
        Shape description.

    Returns
    -------
    matplotlib.path.Path
        Closed cylinder path with cubic caps.
    """

    cap_height = max(spec.height * 0.16, 1.0)
    top_center_y = spec.center_y + spec.height / 2.0 - cap_height
    bottom_center_y = spec.center_y - spec.height / 2.0 + cap_height
    vertices = np.array(
        [
            [spec.center_x - spec.width / 2.0, top_center_y],
            [spec.center_x - spec.width / 2.0, top_center_y + cap_height],
            [spec.center_x + spec.width / 2.0, top_center_y + cap_height],
            [spec.center_x + spec.width / 2.0, top_center_y],
            [spec.center_x + spec.width / 2.0, bottom_center_y],
            [spec.center_x + spec.width / 2.0, bottom_center_y - cap_height],
            [spec.center_x - spec.width / 2.0, bottom_center_y - cap_height],
            [spec.center_x - spec.width / 2.0, bottom_center_y],
            [spec.center_x - spec.width / 2.0, top_center_y],
        ],
        dtype=np.float64,
    )
    codes = [
        Path.MOVETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CLOSEPOLY,
    ]
    return Path(vertices, codes)


def extract_patch_path(patch: object) -> Path:
    """Extract a patch outline path in data coordinates.

    Parameters
    ----------
    patch : object
        Matplotlib patch instance.

    Returns
    -------
    matplotlib.path.Path
        Path expressed in data coordinates.
    """

    if isinstance(patch, (Ellipse, Circle)):
        return patch.get_path().transformed(patch.get_patch_transform())
    return patch.get_path()


def build_shape_path(spec: ShapeSpec) -> Path:
    """Return the native matplotlib outline path for one supported shape.

    Parameters
    ----------
    spec : ShapeSpec
        Shape description.

    Returns
    -------
    matplotlib.path.Path
        Closed path in data coordinates.
    """

    shape = spec.shape
    if shape in {"roundrect", "rect"}:
        corner_radius = float(spec.corner_radius) if shape == "roundrect" else 0.0
        boxstyle = (
            f"round,pad=0,rounding_size={corner_radius}" if corner_radius > 0.0 else "square,pad=0"
        )
        patch = FancyBboxPatch(
            (spec.center_x - spec.width / 2.0, spec.center_y - spec.height / 2.0),
            spec.width,
            spec.height,
            boxstyle=boxstyle,
        )
        return extract_patch_path(patch)
    if shape == "ellipse":
        return extract_patch_path(Ellipse((spec.center_x, spec.center_y), spec.width, spec.height))
    if shape == "circle":
        diameter = max(spec.width, spec.height)
        return extract_patch_path(Circle((spec.center_x, spec.center_y), diameter / 2.0))
    if shape == "cylinder":
        return cylinder_path(spec)
    if shape in {
        "diamond",
        "triangle",
        "hexagon",
        "pentagon",
        "octagon",
        "star",
        "parallelogram",
        "trapezoid",
    }:
        return closed_path_from_vertices(polygon_vertices(spec))
    fallback = ShapeSpec(
        center_x=spec.center_x,
        center_y=spec.center_y,
        width=spec.width,
        height=spec.height,
        shape="roundrect",
        corner_radius=spec.corner_radius,
    )
    return build_shape_path(fallback)


def path_to_closed_vertices(path: Path) -> FloatArray:
    """Approximate a closed path as a polygon for dash walking and offsets.

    Parameters
    ----------
    path : matplotlib.path.Path
        Source path.

    Returns
    -------
    numpy.ndarray
        Closed polygon with shape ``[N, 2]`` and ``points[0] == points[-1]``.
    """

    polygons = path.to_polygons(closed_only=True)
    if not polygons:
        raise ValueError("Expected a closed path with at least one polygon.")
    vertices = np.asarray(polygons[0], dtype=np.float64)
    if vertices.shape[0] < 3:
        raise ValueError("Closed path approximation must contain at least three vertices.")
    if not np.allclose(vertices[0], vertices[-1]):
        vertices = np.vstack([vertices, vertices[0:1]])
    return vertices
