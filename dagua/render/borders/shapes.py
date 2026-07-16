"""Shape-path helpers for data-coordinate node and cluster borders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch
from matplotlib.path import Path
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
CornerRadius = Union[float, Tuple[float, float, float, float]]
ELLIPSE_KAPPA = 0.5522847498
GRAPHVIZ_BIO_DETAIL = 6.0
GRAPHVIZ_BIO_ARROW_LENGTH = 12.0
GRAPHVIZ_BIO_BLOCK_ARROW_LENGTH = 18.0
GRAPHVIZ_BIO_PROMOTER_STEM_FRACTION = 0.25
GRAPHVIZ_BIO_PROMOTER_ARROW_FRACTION = 0.625
GRAPHVIZ_INVTRAPEZIUM_INSET_RATIO = 29.95 / 144.0
GRAPHVIZ_COMPONENT_DETAIL = 4.0
GRAPHVIZ_FOLDER_TAB_HEIGHT = 4.0
GRAPHVIZ_FOLDER_TAB_WIDTH = 27.0
GRAPHVIZ_M_CIRCLE_CHORD_RATIO = 0.75
GRAPHVIZ_M_CORNER_MARK_LENGTH = 12.0
GRAPHVIZ_NOTE_FOLD_SIZE = 6.0
GRAPHVIZ_OCTAGON_PERIPHERY_GAP = 4.0
GRAPHVIZ_TAB_HEIGHT = 4.0
GRAPHVIZ_TAB_WIDTH = 12.0
SEMICIRCLE_DEFAULT_CURVATURE = 1.0
SEMICIRCLE_MIN_CURVATURE = 1e-6
ROUNDED_POLYGON_RADIUS_FRACTION = 0.08


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
    corner_radius : float | tuple[float, float, float, float], default=0.0
        Rounded-rectangle corner radius in data units. Tuples are ordered as
        ``(top_left, top_right, bottom_right, bottom_left)``.
    aspect_ratio : float | None, default=None
        Optional shape-specific aspect ratio hint. Semicircle variants use
        this to tune their dome curvature while keeping the flat edge fixed.
    polygon_points : list[tuple[float, float]] | None, default=None
        Custom polygon vertices in Cytoscape's centered unit coordinates,
        where each axis spans ``-1`` to ``1`` across the node box.
    """

    center_x: float
    center_y: float
    width: float
    height: float
    shape: str
    corner_radius: CornerRadius = 0.0
    aspect_ratio: Optional[float] = None
    polygon_points: Optional[List[Tuple[float, float]]] = None


def scale_corner_radius(corner_radius: CornerRadius, scale: float) -> CornerRadius:
    """Scale one corner-radius specification.

    Parameters
    ----------
    corner_radius : float | tuple[float, float, float, float]
        Scalar or per-corner radius specification.
    scale : float
        Multiplicative scale factor.

    Returns
    -------
    float | tuple[float, float, float, float]
        Scaled radius specification with negative values clamped to ``0.0``.
    """

    safe_scale = max(float(scale), 0.0)
    if isinstance(corner_radius, tuple):
        return tuple(max(float(value), 0.0) * safe_scale for value in corner_radius)
    return max(float(corner_radius), 0.0) * safe_scale


def add_corner_radius(corner_radius: CornerRadius, delta: float) -> CornerRadius:
    """Offset one corner-radius specification by a constant amount.

    Parameters
    ----------
    corner_radius : float | tuple[float, float, float, float]
        Scalar or per-corner radius specification.
    delta : float
        Additive offset in data units.

    Returns
    -------
    float | tuple[float, float, float, float]
        Offset radius specification with negative values clamped to ``0.0``.
    """

    if isinstance(corner_radius, tuple):
        return tuple(max(float(value) + float(delta), 0.0) for value in corner_radius)
    return max(float(corner_radius) + float(delta), 0.0)


def normalize_corner_radii(
    corner_radius: CornerRadius,
    width: float,
    height: float,
) -> Tuple[float, float, float, float]:
    """Return clamped per-corner radii for a rounded rectangle.

    Parameters
    ----------
    corner_radius : float | tuple[float, float, float, float]
        Scalar or per-corner radius specification.
    width : float
        Rounded-rectangle width in data units.
    height : float
        Rounded-rectangle height in data units.

    Returns
    -------
    tuple[float, float, float, float]
        Per-corner radii ordered as
        ``(top_left, top_right, bottom_right, bottom_left)``.
    """

    if isinstance(corner_radius, tuple):
        top_left, top_right, bottom_right, bottom_left = (
            max(float(value), 0.0) for value in corner_radius
        )
    else:
        radius = max(float(corner_radius), 0.0)
        top_left = top_right = bottom_right = bottom_left = radius

    safe_width = max(float(width), 0.0)
    safe_height = max(float(height), 0.0)
    if safe_width <= 0.0 or safe_height <= 0.0:
        return (0.0, 0.0, 0.0, 0.0)

    max_radius = min(safe_width / 2.0, safe_height / 2.0)
    top_left = min(top_left, max_radius)
    top_right = min(top_right, max_radius)
    bottom_right = min(bottom_right, max_radius)
    bottom_left = min(bottom_left, max_radius)

    scale_factor = min(
        1.0,
        safe_width
        / max(top_left + top_right, bottom_left + bottom_right, np.finfo(np.float64).eps),
        safe_height
        / max(top_left + bottom_left, top_right + bottom_right, np.finfo(np.float64).eps),
    )
    return (
        top_left * scale_factor,
        top_right * scale_factor,
        bottom_right * scale_factor,
        bottom_left * scale_factor,
    )


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
    # Tuned down from ``0.32`` to produce slimmer inner valleys that feel more
    # Graphviz-like and keep short labels away from the inward star points.
    inner_rx = outer_rx * 0.25
    inner_ry = outer_ry * 0.25
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


def graphviz_octagon_vertices(spec: ShapeSpec, offset: float = 0.0) -> FloatArray:
    """Return Graphviz octagon vertices with an optional outward offset.

    Parameters
    ----------
    spec : ShapeSpec
        Base octagon bounds in data units.
    offset : float, default=0.0
        Perpendicular outward distance from every base edge.

    Returns
    -------
    numpy.ndarray
        Clockwise vertices with shape ``[8, 2]``.
    """

    x = float(spec.center_x)
    y = float(spec.center_y)
    width = float(spec.width)
    height = float(spec.height)
    corner_ratio = 1.0 - 1.0 / np.sqrt(2.0)
    corner_x = width * corner_ratio
    corner_y = height * corner_ratio
    left = x - width / 2.0
    right = x + width / 2.0
    bottom = y - height / 2.0
    top = y + height / 2.0
    vertices = np.array(
        [
            [left + corner_x, top],
            [right - corner_x, top],
            [right, top - corner_y],
            [right, bottom + corner_y],
            [right - corner_x, bottom],
            [left + corner_x, bottom],
            [left, bottom + corner_y],
            [left, top - corner_y],
        ],
        dtype=np.float64,
    )
    if offset <= 0.0:
        return vertices

    offset_vertices: List[FloatArray] = []
    for index, point in enumerate(vertices):
        previous_point = vertices[(index - 1) % len(vertices)]
        next_point = vertices[(index + 1) % len(vertices)]
        incoming = point - previous_point
        outgoing = next_point - point
        incoming_normal = _normalize_vector(np.array([-incoming[1], incoming[0]]))
        outgoing_normal = _normalize_vector(np.array([-outgoing[1], outgoing[0]]))
        shifted_incoming = point + incoming_normal * offset
        shifted_outgoing = point + outgoing_normal * offset
        matrix = np.column_stack([incoming, -outgoing])
        if abs(float(np.linalg.det(matrix))) <= np.finfo(np.float64).eps:
            offset_vertices.append((shifted_incoming + shifted_outgoing) / 2.0)
            continue
        parameters = np.linalg.solve(matrix, shifted_outgoing - shifted_incoming)
        offset_vertices.append(shifted_incoming + incoming * parameters[0])
    return np.vstack(offset_vertices)


def graphviz_octagon_path(spec: ShapeSpec, offset: float = 0.0) -> Path:
    """Return a Graphviz octagon path at one periphery offset.

    Parameters
    ----------
    spec : ShapeSpec
        Base octagon bounds in data units.
    offset : float, default=0.0
        Perpendicular outward distance from every base edge.

    Returns
    -------
    matplotlib.path.Path
        Closed octagon path.
    """

    return closed_path_from_vertices(graphviz_octagon_vertices(spec, offset))


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
    if spec.shape == "polygon":
        if spec.polygon_points is None or len(spec.polygon_points) < 3:
            raise ValueError("Custom polygon shapes require at least three polygon_points.")
        unit_points = np.asarray(spec.polygon_points, dtype=np.float64)
        if unit_points.ndim != 2 or unit_points.shape[1] != 2:
            raise ValueError("polygon_points must contain two-coordinate vertices.")
        return np.column_stack(
            (
                x + unit_points[:, 0] * width / 2.0,
                y + unit_points[:, 1] * height / 2.0,
            )
        )
    if spec.shape in {"diamond", "Mdiamond"}:
        return np.array(
            [
                [x, y + height / 2.0],
                [x + width / 2.0, y],
                [x, y - height / 2.0],
                [x - width / 2.0, y],
            ],
            dtype=np.float64,
        )
    if spec.shape == "Msquare":
        side = max(width, height)
        half_side = side / 2.0
        return np.array(
            [
                [x - half_side, y + half_side],
                [x + half_side, y + half_side],
                [x + half_side, y - half_side],
                [x - half_side, y - half_side],
            ],
            dtype=np.float64,
        )
    if spec.shape == "triangle":
        return triangle_vertices(x, y, width, height)
    if spec.shape == "hexagon":
        return regular_polygon_vertices(6, x, y, width, height)
    if spec.shape == "pentagon":
        return regular_polygon_vertices(5, x, y, width, height)
    if spec.shape in {"house", "invhouse"}:
        # Graphviz normalizes the regular pentagon horizontally to the node
        # box while retaining its regular-polygon vertical coordinates.
        shoulder_y = height * 0.1545084972
        base_y = -height * 0.4045084972
        vertices = np.array(
            [
                [x, y + height / 2.0],
                [x + width / 2.0, y + shoulder_y],
                [x + width / 2.0, y + base_y],
                [x - width / 2.0, y + base_y],
                [x - width / 2.0, y + shoulder_y],
            ],
            dtype=np.float64,
        )
        if spec.shape == "invhouse":
            vertices[:, 1] = 2.0 * y - vertices[:, 1]
        return vertices
    if spec.shape == "octagon":
        return regular_polygon_vertices(8, x, y, width, height, rotation=np.pi / 8.0)
    if spec.shape in {"doubleoctagon", "tripleoctagon"}:
        periphery_count = 2 if spec.shape == "doubleoctagon" else 3
        return graphviz_octagon_vertices(
            spec,
            GRAPHVIZ_OCTAGON_PERIPHERY_GAP * (periphery_count - 1),
        )
    if spec.shape == "star":
        return star_vertices(x, y, width, height)
    if spec.shape == "parallelogram":
        # Tuned from ``0.30`` so the slant remains obvious without making the
        # top edge look detached from the node body.
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
        # Tuned from ``0.20`` so the top edge reads clearly narrower while
        # still leaving enough interior width for labels.
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
    if spec.shape == "invtrapezium":
        # Graphviz 7.0.5 normalizes its regular four-sided polygon to this
        # horizontal inset (29.95 points for a 144-point-wide node).
        inset = width * GRAPHVIZ_INVTRAPEZIUM_INSET_RATIO
        return np.array(
            [
                [x - width / 2.0 + inset, y - height / 2.0],
                [x + width / 2.0 - inset, y - height / 2.0],
                [x + width / 2.0, y + height / 2.0],
                [x - width / 2.0, y + height / 2.0],
            ],
            dtype=np.float64,
        )
    if spec.shape == "promoter":
        detail = GRAPHVIZ_BIO_DETAIL
        arrow_base_x = x - width / 2.0 + width * GRAPHVIZ_BIO_PROMOTER_ARROW_FRACTION
        stem_x = x - width / 2.0 + width * GRAPHVIZ_BIO_PROMOTER_STEM_FRACTION
        return np.array(
            [
                [arrow_base_x, y + 3.0 * detail],
                [stem_x, y + 3.0 * detail],
                [stem_x, y],
                [stem_x + detail, y],
                [stem_x + detail, y + 2.0 * detail],
                [arrow_base_x, y + 2.0 * detail],
                [arrow_base_x, y + 1.5 * detail],
                [arrow_base_x + GRAPHVIZ_BIO_ARROW_LENGTH, y + 2.5 * detail],
                [arrow_base_x, y + 3.5 * detail],
            ],
            dtype=np.float64,
        )
    if spec.shape == "cds":
        detail = GRAPHVIZ_BIO_DETAIL
        left = x - width / 2.0
        right = x + width / 2.0
        bottom = y - height / 2.0
        top = y + height / 2.0
        return np.array(
            [
                [right - GRAPHVIZ_BIO_ARROW_LENGTH, top - detail],
                [left, top - detail],
                [left, bottom + detail],
                [right - GRAPHVIZ_BIO_ARROW_LENGTH, bottom + detail],
                [right, y],
            ],
            dtype=np.float64,
        )
    if spec.shape == "terminator":
        detail = GRAPHVIZ_BIO_DETAIL
        half_detail = detail / 2.0
        return np.array(
            [
                [x + half_detail, y],
                [x + half_detail, y + detail],
                [x + 1.5 * detail, y + detail],
                [x + 1.5 * detail, y + 2.0 * detail],
                [x - 1.5 * detail, y + 2.0 * detail],
                [x - 1.5 * detail, y + detail],
                [x - half_detail, y + detail],
                [x - half_detail, y],
            ],
            dtype=np.float64,
        )
    if spec.shape in {"ribosite", "proteasesite"}:
        half_detail = GRAPHVIZ_BIO_DETAIL / 2.0
        quarter_detail = GRAPHVIZ_BIO_DETAIL / 4.0
        base_y = y + GRAPHVIZ_BIO_DETAIL
        return np.array(
            [
                [x + half_detail, base_y],
                [x + half_detail, base_y + quarter_detail],
                [x + quarter_detail, base_y + half_detail],
                [x + half_detail, base_y + 3.0 * quarter_detail],
                [x + half_detail, base_y + GRAPHVIZ_BIO_DETAIL],
                [x + quarter_detail, base_y + GRAPHVIZ_BIO_DETAIL],
                [x, base_y + 3.0 * quarter_detail],
                [x - quarter_detail, base_y + GRAPHVIZ_BIO_DETAIL],
                [x - half_detail, base_y + GRAPHVIZ_BIO_DETAIL],
                [x - half_detail, base_y + 3.0 * quarter_detail],
                [x - quarter_detail, base_y + half_detail],
                [x - half_detail, base_y + quarter_detail],
                [x - half_detail, base_y],
                [x - quarter_detail, base_y],
                [x, base_y + quarter_detail],
                [x + quarter_detail, base_y],
            ],
            dtype=np.float64,
        )
    if spec.shape in {"rpromoter", "rarrow"}:
        detail = GRAPHVIZ_BIO_DETAIL
        left = x - width / 2.0
        right = x + width / 2.0
        bottom = y - height / 2.0
        top = y + height / 2.0
        arrow_base_x = right - GRAPHVIZ_BIO_BLOCK_ARROW_LENGTH
        if spec.shape == "rpromoter":
            vertices = [
                [arrow_base_x, top - detail],
                [left, top - detail],
                [left, bottom],
                [left + GRAPHVIZ_BIO_BLOCK_ARROW_LENGTH, bottom],
                [left + GRAPHVIZ_BIO_BLOCK_ARROW_LENGTH, bottom + detail],
                [arrow_base_x, bottom + detail],
                [arrow_base_x, bottom],
                [right, y],
                [arrow_base_x, top],
            ]
        else:
            vertices = [
                [arrow_base_x, top - detail],
                [left, top - detail],
                [left, bottom + detail],
                [arrow_base_x, bottom + detail],
                [arrow_base_x, bottom],
                [right, y],
                [arrow_base_x, top],
            ]
        return np.asarray(vertices, dtype=np.float64)
    if spec.shape == "larrow":
        detail = GRAPHVIZ_BIO_DETAIL
        left = x - width / 2.0
        right = x + width / 2.0
        bottom = y - height / 2.0
        top = y + height / 2.0
        arrow_base_x = left + GRAPHVIZ_BIO_BLOCK_ARROW_LENGTH
        return np.array(
            [
                [right, top - detail],
                [arrow_base_x, top - detail],
                [arrow_base_x, top],
                [left, y],
                [arrow_base_x, bottom],
                [arrow_base_x, bottom + detail],
                [right, bottom + detail],
            ],
            dtype=np.float64,
        )
    if spec.shape == "insulator":
        half_size = GRAPHVIZ_BIO_DETAIL
        return np.array(
            [
                [x + half_size, y + half_size],
                [x + half_size, y - half_size],
                [x - half_size, y - half_size],
                [x - half_size, y + half_size],
            ],
            dtype=np.float64,
        )
    if spec.shape == "signature":
        detail = GRAPHVIZ_BIO_DETAIL
        left = x - width / 2.0
        right = x + width / 2.0
        bottom = y - height / 2.0
        top = y + height / 2.0
        return np.array(
            [
                [right, top - detail],
                [left, top - detail],
                [left, bottom + detail],
                [right, bottom + detail],
            ],
            dtype=np.float64,
        )
    if spec.shape == "arrow":
        half_width = width / 2.0
        half_height = height / 2.0
        return np.array(
            [
                [x - half_width, y + half_height],
                [x + half_width * 0.5, y + half_height],
                [x + half_width, y],
                [x + half_width * 0.5, y - half_height],
                [x - half_width, y - half_height],
                [x - half_width * 0.3, y],
            ],
            dtype=np.float64,
        )
    raise ValueError(f"Shape {spec.shape!r} does not have polygon vertices.")


def arrow_path(spec: ShapeSpec) -> Path:
    """Return the rightward arrow/chevron outline path.

    Parameters
    ----------
    spec : ShapeSpec
        Shape description.

    Returns
    -------
    matplotlib.path.Path
        Closed arrow path.
    """

    return closed_path_from_vertices(polygon_vertices(spec))


def assembly_path(spec: ShapeSpec) -> Path:
    """Return Graphviz's two-bar synthetic-biology assembly symbol.

    Parameters
    ----------
    spec : ShapeSpec
        Shape description.

    Returns
    -------
    matplotlib.path.Path
        Compound path containing the two closed rectangular bars.
    """

    half_width = 2.0 * GRAPHVIZ_BIO_DETAIL
    half_height = GRAPHVIZ_BIO_DETAIL / 2.0
    center_offset = 0.75 * GRAPHVIZ_BIO_DETAIL
    upper = closed_path_from_vertices(
        np.array(
            [
                [spec.center_x - half_width, spec.center_y + center_offset - half_height],
                [spec.center_x + half_width, spec.center_y + center_offset - half_height],
                [spec.center_x + half_width, spec.center_y + center_offset + half_height],
                [spec.center_x - half_width, spec.center_y + center_offset + half_height],
            ],
            dtype=np.float64,
        )
    )
    lower = closed_path_from_vertices(
        np.array(
            [
                [spec.center_x - half_width, spec.center_y - center_offset - half_height],
                [spec.center_x + half_width, spec.center_y - center_offset - half_height],
                [spec.center_x + half_width, spec.center_y - center_offset + half_height],
                [spec.center_x - half_width, spec.center_y - center_offset + half_height],
            ],
            dtype=np.float64,
        )
    )
    return Path.make_compound_path(upper, lower)


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


def rounded_polygon_path(vertices: FloatArray, radius: float) -> Path:
    """Build a closed polygon path with quadratic curves at every corner.

    Parameters
    ----------
    vertices : numpy.ndarray
        Polygon corners with shape ``[N, 2]``.
    radius : float
        Distance from each corner to the curve endpoints in data units.

    Returns
    -------
    matplotlib.path.Path
        Closed path containing one quadratic curve per polygon corner.
    """

    if vertices.shape[0] < 3:
        raise ValueError("A rounded polygon requires at least three vertices.")

    safe_radius = max(float(radius), 0.0)
    incoming_points: List[FloatArray] = []
    outgoing_points: List[FloatArray] = []
    for index, vertex in enumerate(vertices):
        previous_vertex = vertices[(index - 1) % vertices.shape[0]]
        next_vertex = vertices[(index + 1) % vertices.shape[0]]
        incoming = vertex - previous_vertex
        outgoing = next_vertex - vertex
        incoming_length = float(np.linalg.norm(incoming))
        outgoing_length = float(np.linalg.norm(outgoing))
        corner_radius = min(safe_radius, incoming_length / 2.0, outgoing_length / 2.0)
        incoming_unit = incoming / max(incoming_length, np.finfo(np.float64).eps)
        outgoing_unit = outgoing / max(outgoing_length, np.finfo(np.float64).eps)
        incoming_points.append(vertex - incoming_unit * corner_radius)
        outgoing_points.append(vertex + outgoing_unit * corner_radius)

    path_vertices: List[FloatArray] = [incoming_points[0]]
    codes: List[int] = [Path.MOVETO]
    for index, vertex in enumerate(vertices):
        path_vertices.extend([vertex, outgoing_points[index]])
        codes.extend([Path.CURVE3, Path.CURVE3])
        next_index = (index + 1) % vertices.shape[0]
        if next_index != 0:
            path_vertices.append(incoming_points[next_index])
            codes.append(Path.LINETO)
    path_vertices.append(incoming_points[0])
    codes.append(Path.CLOSEPOLY)
    return Path(np.asarray(path_vertices, dtype=np.float64), codes)


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


def roundrect_path(spec: ShapeSpec) -> Path:
    """Return a rounded-rectangle outline with optional per-corner radii.

    Parameters
    ----------
    spec : ShapeSpec
        Shape description.

    Returns
    -------
    matplotlib.path.Path
        Closed rounded-rectangle path.
    """

    left = float(spec.center_x) - float(spec.width) / 2.0
    right = float(spec.center_x) + float(spec.width) / 2.0
    bottom = float(spec.center_y) - float(spec.height) / 2.0
    top = float(spec.center_y) + float(spec.height) / 2.0
    top_left, top_right, bottom_right, bottom_left = normalize_corner_radii(
        spec.corner_radius,
        spec.width,
        spec.height,
    )
    handle_factor = ELLIPSE_KAPPA

    vertices: List[List[float]] = [[left + top_left, top]]
    codes: List[int] = [int(Path.MOVETO)]

    vertices.append([right - top_right, top])
    codes.append(int(Path.LINETO))
    if top_right > 0.0:
        vertices.extend(
            [
                [right - top_right + handle_factor * top_right, top],
                [right, top - top_right + handle_factor * top_right],
                [right, top - top_right],
            ]
        )
        codes.extend([int(Path.CURVE4), int(Path.CURVE4), int(Path.CURVE4)])
    else:
        vertices.append([right, top])
        codes.append(int(Path.LINETO))

    vertices.append([right, bottom + bottom_right])
    codes.append(int(Path.LINETO))
    if bottom_right > 0.0:
        vertices.extend(
            [
                [right, bottom + bottom_right - handle_factor * bottom_right],
                [right - bottom_right + handle_factor * bottom_right, bottom],
                [right - bottom_right, bottom],
            ]
        )
        codes.extend([int(Path.CURVE4), int(Path.CURVE4), int(Path.CURVE4)])
    else:
        vertices.append([right, bottom])
        codes.append(int(Path.LINETO))

    vertices.append([left + bottom_left, bottom])
    codes.append(int(Path.LINETO))
    if bottom_left > 0.0:
        vertices.extend(
            [
                [left + bottom_left - handle_factor * bottom_left, bottom],
                [left, bottom + bottom_left - handle_factor * bottom_left],
                [left, bottom + bottom_left],
            ]
        )
        codes.extend([int(Path.CURVE4), int(Path.CURVE4), int(Path.CURVE4)])
    else:
        vertices.append([left, bottom])
        codes.append(int(Path.LINETO))

    vertices.append([left, top - top_left])
    codes.append(int(Path.LINETO))
    if top_left > 0.0:
        vertices.extend(
            [
                [left, top - top_left + handle_factor * top_left],
                [left + top_left - handle_factor * top_left, top],
                [left + top_left, top],
            ]
        )
        codes.extend([int(Path.CURVE4), int(Path.CURVE4), int(Path.CURVE4)])
    else:
        vertices.append([left, top])
        codes.append(int(Path.LINETO))

    vertices.append([left + top_left, top])
    codes.append(int(Path.CLOSEPOLY))
    return Path(np.asarray(vertices, dtype=np.float64), codes)


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
    codes: List[int] = [int(Path.MOVETO)]

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
        codes.extend([int(Path.CURVE4), int(Path.CURVE4), int(Path.CURVE4)])

    vertices.append(start.tolist())
    codes.append(int(Path.CLOSEPOLY))
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


def _semicircle_curvature_ratio(spec: ShapeSpec) -> float:
    """Return the curvature ratio used by semicircle variants.

    Parameters
    ----------
    spec : ShapeSpec
        Shape description.

    Returns
    -------
    float
        Positive curvature ratio. ``1.0`` preserves the default half-ellipse
        defined by the node bounds, larger values flatten the dome, and
        smaller values deepen it when the requested bounds allow it.
    """

    if spec.aspect_ratio is None or spec.aspect_ratio <= SEMICIRCLE_MIN_CURVATURE:
        return SEMICIRCLE_DEFAULT_CURVATURE
    return float(spec.aspect_ratio)


def semicircle_path(spec: ShapeSpec, orientation: str = "up") -> Path:
    """Build a semicircle or semi-ellipse outline.

    Parameters
    ----------
    spec : ShapeSpec
        Shape description.
    orientation : str, default="up"
        Dome orientation. Supported values are ``"up"``, ``"down"``,
        ``"left"``, and ``"right"``.

    Returns
    -------
    matplotlib.path.Path
        Closed semicircle path in data coordinates.
    """

    cx = float(spec.center_x)
    cy = float(spec.center_y)
    width = float(spec.width)
    height = float(spec.height)
    half_width = width / 2.0
    half_height = height / 2.0
    curvature_ratio = _semicircle_curvature_ratio(spec)

    if orientation in {"up", "down"}:
        radius_x = half_width
        # Semicircle height is the ellipse semi-axis. When aspect_ratio is
        # provided, interpret it as the ellipse axis ratio (rx / ry) so callers
        # can flatten or deepen the dome independently from the node width.
        radius_y = min(height, half_width / curvature_ratio)
        radius_y = max(radius_y, SEMICIRCLE_MIN_CURVATURE)
        handle_x = ELLIPSE_KAPPA * radius_x
        handle_y = ELLIPSE_KAPPA * radius_y
        left = cx - half_width
        right = cx + half_width
        if orientation == "up":
            flat_y = cy - half_height
            top_y = flat_y + radius_y
            vertices = np.array(
                [
                    [left, flat_y],
                    [right, flat_y],
                    [right, flat_y + handle_y],
                    [cx + handle_x, top_y],
                    [cx, top_y],
                    [cx - handle_x, top_y],
                    [left, flat_y + handle_y],
                    [left, flat_y],
                    [left, flat_y],
                ],
                dtype=np.float64,
            )
        else:
            flat_y = cy + half_height
            bottom_y = flat_y - radius_y
            vertices = np.array(
                [
                    [left, flat_y],
                    [right, flat_y],
                    [right, flat_y - handle_y],
                    [cx + handle_x, bottom_y],
                    [cx, bottom_y],
                    [cx - handle_x, bottom_y],
                    [left, flat_y - handle_y],
                    [left, flat_y],
                    [left, flat_y],
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
            Path.CLOSEPOLY,
        ]
        return Path(vertices, codes)

    radius_y = half_height
    radius_x = min(width, half_height * curvature_ratio)
    radius_x = max(radius_x, SEMICIRCLE_MIN_CURVATURE)
    handle_x = ELLIPSE_KAPPA * radius_x
    handle_y = ELLIPSE_KAPPA * radius_y
    top = cy + half_height
    bottom = cy - half_height
    if orientation == "left":
        flat_x = cx + half_width
        left_x = flat_x - radius_x
        vertices = np.array(
            [
                [flat_x, top],
                [flat_x, bottom],
                [flat_x - handle_x, bottom],
                [left_x, cy - handle_y],
                [left_x, cy],
                [left_x, cy + handle_y],
                [flat_x - handle_x, top],
                [flat_x, top],
                [flat_x, top],
            ],
            dtype=np.float64,
        )
    else:
        flat_x = cx - half_width
        right_x = flat_x + radius_x
        vertices = np.array(
            [
                [flat_x, top],
                [flat_x, bottom],
                [flat_x + handle_x, bottom],
                [right_x, cy - handle_y],
                [right_x, cy],
                [right_x, cy + handle_y],
                [flat_x + handle_x, top],
                [flat_x, top],
                [flat_x, top],
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
    tab_width = min(GRAPHVIZ_TAB_WIDTH, spec.width)
    tab_height = min(GRAPHVIZ_TAB_HEIGHT, spec.height / 2.0)
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
        Closed note silhouette. The renderer draws the interior fold lines.
    """

    half_width = spec.width / 2.0
    half_height = spec.height / 2.0
    left = spec.center_x - half_width
    right = spec.center_x + half_width
    bottom = spec.center_y - half_height
    top = spec.center_y + half_height
    fold = min(GRAPHVIZ_NOTE_FOLD_SIZE, half_width, half_height)
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
    return outer


def folder_path(spec: ShapeSpec) -> Path:
    """Return Graphviz's rectangular folder with a raised right tab.

    Parameters
    ----------
    spec : ShapeSpec
        Shape description.

    Returns
    -------
    matplotlib.path.Path
        Closed folder silhouette.
    """

    left = spec.center_x - spec.width / 2.0
    right = spec.center_x + spec.width / 2.0
    bottom = spec.center_y - spec.height / 2.0
    top = spec.center_y + spec.height / 2.0
    tab_width = min(GRAPHVIZ_FOLDER_TAB_WIDTH, spec.width)
    tab_height = min(GRAPHVIZ_FOLDER_TAB_HEIGHT, spec.height / 2.0)
    slope_width = min(3.0, tab_width / 2.0)
    return closed_path_from_vertices(
        np.array(
            [
                [left, bottom],
                [right, bottom],
                [right, top],
                [right - slope_width, top + tab_height],
                [right - tab_width + slope_width, top + tab_height],
                [right - tab_width, top],
                [left, top],
            ],
            dtype=np.float64,
        )
    )


def component_path(spec: ShapeSpec) -> Path:
    """Return Graphviz's component box with two left-side lugs.

    Parameters
    ----------
    spec : ShapeSpec
        Shape description.

    Returns
    -------
    matplotlib.path.Path
        Closed component silhouette.
    """

    left = spec.center_x - spec.width / 2.0
    right = spec.center_x + spec.width / 2.0
    bottom = spec.center_y - spec.height / 2.0
    top = spec.center_y + spec.height / 2.0
    detail = min(GRAPHVIZ_COMPONENT_DETAIL, spec.height / 8.0, spec.width / 4.0)
    return closed_path_from_vertices(
        np.array(
            [
                [left, top],
                [right, top],
                [right, bottom],
                [left, bottom],
                [left, bottom + detail],
                [left - detail, bottom + detail],
                [left - detail, bottom + 2.0 * detail],
                [left, bottom + 2.0 * detail],
                [left, top - 2.0 * detail],
                [left - detail, top - 2.0 * detail],
                [left - detail, top - detail],
                [left, top - detail],
            ],
            dtype=np.float64,
        )
    )


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


def extract_patch_path(patch: Ellipse | Circle | FancyBboxPatch) -> Path:
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
    rounded_polygon_bases = {
        "round_triangle": "triangle",
        "round_diamond": "diamond",
        "round_pentagon": "pentagon",
        "round_hexagon": "hexagon",
        "round_octagon": "octagon",
    }
    if shape in rounded_polygon_bases:
        base_spec = ShapeSpec(
            center_x=spec.center_x,
            center_y=spec.center_y,
            width=spec.width,
            height=spec.height,
            shape=rounded_polygon_bases[shape],
            corner_radius=0.0,
            aspect_ratio=spec.aspect_ratio,
            polygon_points=spec.polygon_points,
        )
        radius = min(max(float(spec.width), 0.0), max(float(spec.height), 0.0))
        radius *= ROUNDED_POLYGON_RADIUS_FRACTION
        return rounded_polygon_path(polygon_vertices(base_spec), radius)
    if shape == "roundrect":
        return roundrect_path(spec)
    if shape == "rect":
        patch = FancyBboxPatch(
            (spec.center_x - spec.width / 2.0, spec.center_y - spec.height / 2.0),
            spec.width,
            spec.height,
            boxstyle="square,pad=0",
        )
        return extract_patch_path(patch)
    if shape == "ellipse":
        return extract_patch_path(Ellipse((spec.center_x, spec.center_y), spec.width, spec.height))
    if shape == "circle":
        diameter = max(spec.width, spec.height)
        return extract_patch_path(Circle((spec.center_x, spec.center_y), diameter / 2.0))
    if shape == "Mcircle":
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
    if shape in {"semicircle", "semicircle_up"}:
        return semicircle_path(spec, "up")
    if shape == "semicircle_down":
        return semicircle_path(spec, "down")
    if shape == "semicircle_left":
        return semicircle_path(spec, "left")
    if shape == "semicircle_right":
        return semicircle_path(spec, "right")
    if shape == "tab":
        return tab_path(spec)
    if shape == "folder":
        return folder_path(spec)
    if shape == "component":
        return component_path(spec)
    if shape == "note":
        return note_path(spec)
    if shape == "document":
        return document_path(spec)
    if shape == "box3d":
        return box3d_path(spec)
    if shape == "arrow":
        return arrow_path(spec)
    if shape == "polygon":
        return closed_path_from_vertices(polygon_vertices(spec))
    if shape == "assembly":
        return assembly_path(spec)
    if shape == "promoter":
        return closed_path_from_vertices(polygon_vertices(spec))
    if shape == "cds":
        return closed_path_from_vertices(polygon_vertices(spec))
    if shape == "terminator":
        return closed_path_from_vertices(polygon_vertices(spec))
    if shape == "ribosite":
        return closed_path_from_vertices(polygon_vertices(spec))
    if shape == "proteasesite":
        return closed_path_from_vertices(polygon_vertices(spec))
    if shape == "rpromoter":
        return closed_path_from_vertices(polygon_vertices(spec))
    if shape == "rarrow":
        return closed_path_from_vertices(polygon_vertices(spec))
    if shape == "larrow":
        return closed_path_from_vertices(polygon_vertices(spec))
    if shape == "insulator":
        return closed_path_from_vertices(polygon_vertices(spec))
    if shape == "signature":
        return closed_path_from_vertices(polygon_vertices(spec))
    if shape == "invtrapezium":
        return closed_path_from_vertices(polygon_vertices(spec))
    if shape in {
        "diamond",
        "Mdiamond",
        "Msquare",
        "triangle",
        "house",
        "invhouse",
        "hexagon",
        "pentagon",
        "octagon",
        "doubleoctagon",
        "tripleoctagon",
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
        aspect_ratio=spec.aspect_ratio,
        polygon_points=spec.polygon_points,
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
