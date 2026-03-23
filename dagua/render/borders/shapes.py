"""Shape-path helpers for data-coordinate node and cluster borders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch
from matplotlib.path import Path
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ELLIPSE_KAPPA = 0.5522847498


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


def open_path_from_vertices(vertices: FloatArray) -> Path:
    """Build an open linear path from ordered vertices.

    Parameters
    ----------
    vertices : numpy.ndarray
        Polyline vertices with shape ``[N, 2]``.

    Returns
    -------
    matplotlib.path.Path
        Open path with one subpath.

    Raises
    ------
    ValueError
        If fewer than two vertices are provided.
    """

    if vertices.shape[0] < 2:
        raise ValueError("An open path requires at least two vertices.")
    codes = [Path.MOVETO] + [Path.LINETO] * (vertices.shape[0] - 1)
    return Path(vertices, codes)


def _normalize_vector(vector: FloatArray) -> FloatArray:
    """Return a unit-length copy of one 2D vector.

    Parameters
    ----------
    vector : numpy.ndarray
        Vector with shape ``[2]``.

    Returns
    -------
    numpy.ndarray
        Unit-length vector with shape ``[2]``. Degenerate vectors return
        ``[0.0, 0.0]``.
    """

    magnitude = float(np.linalg.norm(vector))
    if magnitude <= np.finfo(np.float64).eps:
        return np.zeros(2, dtype=np.float64)
    return (vector / magnitude).astype(np.float64)


def _ellipse_anchor(
    center_x: float, center_y: float, radius_x: float, radius_y: float, angle: float
) -> FloatArray:
    """Return one point on an axis-aligned ellipse.

    Parameters
    ----------
    center_x : float
        Ellipse center x-coordinate.
    center_y : float
        Ellipse center y-coordinate.
    radius_x : float
        Horizontal radius.
    radius_y : float
        Vertical radius.
    angle : float
        Parametric angle in radians.

    Returns
    -------
    numpy.ndarray
        Anchor point with shape ``[2]``.
    """

    return np.array(
        [
            center_x + radius_x * np.cos(angle),
            center_y + radius_y * np.sin(angle),
        ],
        dtype=np.float64,
    )


def _ellipse_outward_normal(radius_x: float, radius_y: float, angle: float) -> FloatArray:
    """Return the outward unit normal on an axis-aligned ellipse.

    Parameters
    ----------
    radius_x : float
        Horizontal radius.
    radius_y : float
        Vertical radius.
    angle : float
        Parametric angle in radians.

    Returns
    -------
    numpy.ndarray
        Outward-facing unit vector with shape ``[2]``.
    """

    normal = np.array(
        [
            np.cos(angle) / max(radius_x, np.finfo(np.float64).eps),
            np.sin(angle) / max(radius_y, np.finfo(np.float64).eps),
        ],
        dtype=np.float64,
    )
    return _normalize_vector(normal)


def ellipse_cubic_path(center_x: float, center_y: float, radius_x: float, radius_y: float) -> Path:
    """Approximate one ellipse using four cubic Bezier arcs.

    Parameters
    ----------
    center_x : float
        Ellipse center x-coordinate.
    center_y : float
        Ellipse center y-coordinate.
    radius_x : float
        Horizontal radius.
    radius_y : float
        Vertical radius.

    Returns
    -------
    matplotlib.path.Path
        Closed ellipse path.
    """

    handle_x = ELLIPSE_KAPPA * radius_x
    handle_y = ELLIPSE_KAPPA * radius_y
    vertices = np.array(
        [
            [center_x + radius_x, center_y],
            [center_x + radius_x, center_y + handle_y],
            [center_x + handle_x, center_y + radius_y],
            [center_x, center_y + radius_y],
            [center_x - handle_x, center_y + radius_y],
            [center_x - radius_x, center_y + handle_y],
            [center_x - radius_x, center_y],
            [center_x - radius_x, center_y - handle_y],
            [center_x - handle_x, center_y - radius_y],
            [center_x, center_y - radius_y],
            [center_x + handle_x, center_y - radius_y],
            [center_x + radius_x, center_y - handle_y],
            [center_x + radius_x, center_y],
            [center_x + radius_x, center_y],
        ],
        dtype=np.float64,
    )
    codes = [Path.MOVETO] + [Path.CURVE4] * 12 + [Path.CLOSEPOLY]
    return Path(vertices, codes)


def _ellipse_cubic_path_cw(
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
) -> Path:
    """Approximate one ellipse using four cubic Bezier arcs, winding clockwise.

    Identical to :func:`ellipse_cubic_path` but winds in the opposite
    direction (right -> bottom -> left -> top -> right).  Used as the
    inner subpath of double-circle shapes so that the nonzero fill rule
    leaves a visible ring.

    Parameters
    ----------
    center_x : float
        Ellipse center x-coordinate.
    center_y : float
        Ellipse center y-coordinate.
    radius_x : float
        Horizontal radius.
    radius_y : float
        Vertical radius.

    Returns
    -------
    matplotlib.path.Path
        Closed clockwise ellipse path.
    """

    handle_x = ELLIPSE_KAPPA * radius_x
    handle_y = ELLIPSE_KAPPA * radius_y
    # Same start as CCW but traverse right -> bottom -> left -> top -> right.
    vertices = np.array(
        [
            [center_x + radius_x, center_y],
            [center_x + radius_x, center_y - handle_y],
            [center_x + handle_x, center_y - radius_y],
            [center_x, center_y - radius_y],
            [center_x - handle_x, center_y - radius_y],
            [center_x - radius_x, center_y - handle_y],
            [center_x - radius_x, center_y],
            [center_x - radius_x, center_y + handle_y],
            [center_x - handle_x, center_y + radius_y],
            [center_x, center_y + radius_y],
            [center_x + handle_x, center_y + radius_y],
            [center_x + radius_x, center_y + handle_y],
            [center_x + radius_x, center_y],
            [center_x + radius_x, center_y],
        ],
        dtype=np.float64,
    )
    codes = [Path.MOVETO] + [Path.CURVE4] * 12 + [Path.CLOSEPOLY]
    return Path(vertices, codes)


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


def double_circle_path(spec: ShapeSpec) -> Path:
    """Return a compound path containing two concentric ellipses.

    The inner ellipse winds clockwise (opposite to the outer) so that
    matplotlib's nonzero fill rule leaves the gap between the two rings
    unfilled, making the inner circle visually distinct.

    Parameters
    ----------
    spec : ShapeSpec
        Shape description.

    Returns
    -------
    matplotlib.path.Path
        Compound path with outer CCW and inner CW ellipse subpaths.
    """

    radius_x = spec.width / 2.0
    radius_y = spec.height / 2.0
    # Return the outer ellipse only.  The inner ring is drawn as a
    # separate stroke-only element by the renderer (see _draw_nodes)
    # because matplotlib PatchCollection compound paths do not reliably
    # preserve fill-rule ring behaviour.
    return ellipse_cubic_path(spec.center_x, spec.center_y, radius_x, radius_y)


def cloud_path(spec: ShapeSpec) -> Path:
    """Return an organic cloud outline built from outward-bulging cubic arcs.

    Parameters
    ----------
    spec : ShapeSpec
        Shape description.

    Returns
    -------
    matplotlib.path.Path
        Closed cloud path.
    """

    radius_x = spec.width / 2.0
    radius_y = spec.height / 2.0
    anchor_angles = np.linspace(np.pi / 6.0, 2.0 * np.pi + np.pi / 6.0, 7, dtype=np.float64)
    bulge_factors = np.array([0.21, 0.17, 0.24, 0.18, 0.23, 0.19], dtype=np.float64)
    start = _ellipse_anchor(
        spec.center_x, spec.center_y, radius_x, radius_y, float(anchor_angles[0])
    )
    vertices: List[List[float]] = [start.tolist()]
    codes: List[int] = [Path.MOVETO]

    for index in range(6):
        start_angle = float(anchor_angles[index])
        end_angle = float(anchor_angles[index + 1])
        start_anchor = _ellipse_anchor(
            spec.center_x, spec.center_y, radius_x, radius_y, start_angle
        )
        end_anchor = _ellipse_anchor(spec.center_x, spec.center_y, radius_x, radius_y, end_angle)
        mid_angle = (start_angle + end_angle) / 2.0
        chord = end_anchor - start_anchor
        control_distance = float(np.linalg.norm(chord)) * 0.32
        tangent_start = _normalize_vector(
            np.array(
                [-radius_x * np.sin(start_angle), radius_y * np.cos(start_angle)],
                dtype=np.float64,
            )
        )
        tangent_end = _normalize_vector(
            np.array(
                [-radius_x * np.sin(end_angle), radius_y * np.cos(end_angle)],
                dtype=np.float64,
            )
        )
        outward = _ellipse_outward_normal(radius_x, radius_y, mid_angle)
        bulge = radius_x * float(bulge_factors[index])
        control_1 = start_anchor + tangent_start * control_distance + outward * bulge
        control_2 = end_anchor - tangent_end * control_distance + outward * bulge
        vertices.extend([control_1.tolist(), control_2.tolist(), end_anchor.tolist()])
        codes.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])

    vertices.append(start.tolist())
    codes.append(Path.CLOSEPOLY)
    return Path(np.asarray(vertices, dtype=np.float64), codes)


def stadium_path(spec: ShapeSpec) -> Path:
    """Return a capsule outline with semicircular left and right ends.

    Parameters
    ----------
    spec : ShapeSpec
        Shape description.

    Returns
    -------
    matplotlib.path.Path
        Closed stadium path.
    """

    center_x = spec.center_x
    center_y = spec.center_y
    radius = spec.height / 2.0
    straight_half_width = max(spec.width / 2.0 - radius, 0.0)
    left_center_x = center_x - straight_half_width
    right_center_x = center_x + straight_half_width
    handle = ELLIPSE_KAPPA * radius
    vertices = np.array(
        [
            [left_center_x, center_y + radius],
            [right_center_x, center_y + radius],
            [right_center_x + handle, center_y + radius],
            [right_center_x + radius, center_y + handle],
            [right_center_x + radius, center_y],
            [right_center_x + radius, center_y - handle],
            [right_center_x + handle, center_y - radius],
            [right_center_x, center_y - radius],
            [left_center_x, center_y - radius],
            [left_center_x - handle, center_y - radius],
            [left_center_x - radius, center_y - handle],
            [left_center_x - radius, center_y],
            [left_center_x - radius, center_y + handle],
            [left_center_x - handle, center_y + radius],
            [left_center_x, center_y + radius],
            [left_center_x, center_y + radius],
        ],
        dtype=np.float64,
    )
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CLOSEPOLY,
    ]
    return Path(vertices, codes)


def tab_path(spec: ShapeSpec) -> Path:
    """Return a rectangle with a small top-left tab.

    Parameters
    ----------
    spec : ShapeSpec
        Shape description.

    Returns
    -------
    matplotlib.path.Path
        Closed tab path.
    """

    half_width = spec.width / 2.0
    half_height = spec.height / 2.0
    left = spec.center_x - half_width
    right = spec.center_x + half_width
    bottom = spec.center_y - half_height
    top = spec.center_y + half_height
    tab_width = spec.width * 0.30
    tab_height = spec.height * 0.20
    vertices = np.array(
        [
            [left, bottom],
            [right, bottom],
            [right, top],
            [left + tab_width, top],
            [left + tab_width, top + tab_height],
            [left, top + tab_height],
        ],
        dtype=np.float64,
    )
    return closed_path_from_vertices(vertices)


def note_path(spec: ShapeSpec) -> Path:
    """Return a note outline with a folded top-right corner.

    Parameters
    ----------
    spec : ShapeSpec
        Shape description.

    Returns
    -------
    matplotlib.path.Path
        Compound path with an outer outline and inner fold line.
    """

    half_width = spec.width / 2.0
    half_height = spec.height / 2.0
    left = spec.center_x - half_width
    right = spec.center_x + half_width
    bottom = spec.center_y - half_height
    top = spec.center_y + half_height
    fold = min(half_width, half_height) * 0.30
    outer = closed_path_from_vertices(
        np.array(
            [
                [left, bottom],
                [right, bottom],
                [right, top - fold],
                [right - fold, top],
                [left, top],
            ],
            dtype=np.float64,
        )
    )
    fold_line = open_path_from_vertices(
        np.array(
            [
                [right - fold, top],
                [right - fold, top - fold],
                [right, top - fold],
            ],
            dtype=np.float64,
        )
    )
    return Path.make_compound_path(outer, fold_line)


def document_path(spec: ShapeSpec) -> Path:
    """Return a document outline with a single wavy bottom edge.

    Parameters
    ----------
    spec : ShapeSpec
        Shape description.

    Returns
    -------
    matplotlib.path.Path
        Closed document path.
    """

    half_width = spec.width / 2.0
    half_height = spec.height / 2.0
    left = spec.center_x - half_width
    right = spec.center_x + half_width
    bottom = spec.center_y - half_height
    top = spec.center_y + half_height
    mid_x = spec.center_x
    amplitude = spec.height * 0.12
    vertices = np.array(
        [
            [left, top],
            [right, top],
            [right, bottom],
            [right - spec.width * 0.18, bottom - amplitude * 0.20],
            [mid_x + spec.width * 0.16, bottom - amplitude * 1.10],
            [mid_x, bottom - amplitude],
            [mid_x - spec.width * 0.16, bottom - amplitude * 0.90],
            [left + spec.width * 0.18, bottom + amplitude * 0.20],
            [left, bottom],
            [left, top],
            [left, top],
        ],
        dtype=np.float64,
    )
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]
    return Path(vertices, codes)


def box3d_path(spec: ShapeSpec) -> Path:
    """Return an isometric box outline with top and right face dividers.

    Parameters
    ----------
    spec : ShapeSpec
        Shape description.

    Returns
    -------
    matplotlib.path.Path
        Compound path containing the outer box silhouette and interior divider
        lines for the front, top, and right faces.
    """

    half_width = spec.width / 2.0
    half_height = spec.height / 2.0
    left = spec.center_x - half_width
    right = spec.center_x + half_width
    bottom = spec.center_y - half_height
    top = spec.center_y + half_height
    depth = min(half_width, half_height) * 0.25
    offset_x = depth
    offset_y = depth * 0.70
    front_right = right - offset_x
    front_top = top - offset_y
    back_left = left + offset_x
    back_bottom = bottom + offset_y

    silhouette = closed_path_from_vertices(
        np.array(
            [
                [left, bottom],
                [front_right, bottom],
                [right, back_bottom],
                [right, top],
                [back_left, top],
                [left, front_top],
            ],
            dtype=np.float64,
        )
    )
    top_divider = open_path_from_vertices(
        np.array(
            [
                [left, front_top],
                [front_right, front_top],
            ],
            dtype=np.float64,
        )
    )
    right_divider = open_path_from_vertices(
        np.array(
            [
                [front_right, bottom],
                [front_right, front_top],
            ],
            dtype=np.float64,
        )
    )
    return Path.make_compound_path(silhouette, top_divider, right_divider)


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
    if shape == "double_circle":
        return double_circle_path(spec)
    if shape == "cloud":
        return cloud_path(spec)
    if shape == "stadium":
        return stadium_path(spec)
    if shape == "tab":
        return tab_path(spec)
    if shape == "note":
        return note_path(spec)
    if shape == "document":
        return document_path(spec)
    if shape == "box3d":
        return box3d_path(spec)
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
