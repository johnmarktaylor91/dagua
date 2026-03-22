"""Arrowhead geometry for custom data-coordinate edge rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from matplotlib.path import Path

from dagua.render.edges.geometry import FLOAT_EPSILON, Point, as_point, perpendicular, unit_vector

ArrowBuilder = Callable[[float, float, float], "ArrowheadResult"]

LOCAL_CIRCLE_SAMPLES = 24
JOIN_OVERLAP_RATIO = 0.18
JOIN_OVERLAP_LENGTH_FLOOR = 0.35
JOIN_SHOULDER_RATIO = 0.72
OPEN_HEAD_STROKE_SCALE = 1.2
TEE_HEAD_STROKE_SCALE = 1.35
HOLLOW_HEAD_STROKE_SCALE = 1.35
OPEN_HEAD_WIDTH_GAIN = 1.12
HOLLOW_HEAD_DIMENSION_SCALE = 1.3
COMPOUND_HEAD_GAP_RATIO = 0.42
COMPOUND_HEAD_GAP_FLOOR = 0.6


@dataclass(frozen=True)
class ArrowheadResult:
    """Arrowhead geometry split by paint mode.

    Parameters
    ----------
    filled_paths : list[matplotlib.path.Path]
        Closed paths drawn with fill.
    stroked_paths : list[matplotlib.path.Path]
        Paths drawn as outlines only.
    trim_contour : matplotlib.path.Path
        Contour where the body should terminate.
    trim_t : float, default=1.0
        Centerline parameter where trimming occurs. The renderer fills this.
    stroke_width_scale : float, default=1.0
        Multiplier applied when the arrowhead is rendered with strokes.
    """

    filled_paths: List[Path]
    stroked_paths: List[Path]
    trim_contour: Path
    trim_t: float = 1.0
    stroke_width_scale: float = 1.0

    @staticmethod
    def compose(results: Sequence["ArrowheadResult"]) -> "ArrowheadResult":
        """Compose sequential arrowhead primitives into one result.

        Parameters
        ----------
        results : Sequence[ArrowheadResult]
            Primitive results ordered from the tip toward the body.

        Returns
        -------
        ArrowheadResult
            Concatenated geometry whose trim contour comes from the last
            primitive in the chain.
        """
        if not results:
            raise ValueError("At least one arrowhead result is required for composition.")
        filled_paths: List[Path] = []
        stroked_paths: List[Path] = []
        for result in results:
            filled_paths.extend(result.filled_paths)
            stroked_paths.extend(result.stroked_paths)
        return ArrowheadResult(
            filled_paths=filled_paths,
            stroked_paths=stroked_paths,
            trim_contour=results[-1].trim_contour,
            trim_t=results[-1].trim_t,
            stroke_width_scale=max(result.stroke_width_scale for result in results),
        )


@dataclass(frozen=True)
class PrimitiveSpec:
    """Registered arrowhead primitive.

    Parameters
    ----------
    name : str
        Registry key.
    builder : Callable[[float, float], ArrowheadResult]
        Builder in local coordinates.
    stroke_only : bool, default=False
        Whether the primitive is always outline-only.
    """

    name: str
    builder: ArrowBuilder
    stroke_only: bool = False


def _local_path(points: Sequence[Sequence[float]], closed: bool = True) -> Path:
    """Build a local linear path.

    Parameters
    ----------
    points : Sequence[Sequence[float]]
        Path vertices in local coordinates.
    closed : bool, default=True
        Whether to close the path.

    Returns
    -------
    matplotlib.path.Path
        Linear path.
    """
    vertices = np.vstack([as_point(point) for point in points])
    if closed:
        vertices = np.vstack([vertices, vertices[0]])
        codes = [Path.MOVETO] + [Path.LINETO] * (vertices.shape[0] - 2) + [Path.CLOSEPOLY]
        return Path(vertices, codes)
    codes = [Path.MOVETO] + [Path.LINETO] * (vertices.shape[0] - 1)
    return Path(vertices, codes)


def _local_trim_contour(x: float, width: float) -> Path:
    """Build the local trim contour for a primitive.

    Parameters
    ----------
    x : float
        Body-side location of the contour.
    width : float
        Full contour width.

    Returns
    -------
    matplotlib.path.Path
        Straight trim segment.
    """
    half_width = width * 0.5
    return _local_path([(x, half_width), (x, -half_width)], closed=False)


def _local_circle(center_x: float, radius: float) -> Path:
    """Approximate a local circle with line segments.

    Parameters
    ----------
    center_x : float
        Circle center on the local body axis.
    radius : float
        Circle radius.

    Returns
    -------
    matplotlib.path.Path
        Closed polygonal circle.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, LOCAL_CIRCLE_SAMPLES, endpoint=False)
    points = [(center_x + radius * np.cos(angle), radius * np.sin(angle)) for angle in angles]
    return _local_path(points, closed=True)


def _clip_local_path(path: Path, side: str) -> Path:
    """Clip a local path to its left or right half.

    Parameters
    ----------
    path : matplotlib.path.Path
        Local path.
    side : str
        ``"left"``, ``"right"``, or ``"both"``.

    Returns
    -------
    matplotlib.path.Path
        Clipped path.
    """
    if side == "both":
        return path
    vertices = np.array(path.vertices, copy=True)
    if side == "left":
        vertices[:, 1] = np.maximum(vertices[:, 1], 0.0)
    else:
        vertices[:, 1] = np.minimum(vertices[:, 1], 0.0)
    return Path(vertices, path.codes)


def _transform_path(path: Path, tip: Point, body_direction: Point) -> Path:
    """Transform a local arrow path into world coordinates.

    Parameters
    ----------
    path : matplotlib.path.Path
        Local path.
    tip : numpy.ndarray
        Tip position in world coordinates.
    body_direction : numpy.ndarray
        Unit vector pointing from the tip back into the edge body.

    Returns
    -------
    matplotlib.path.Path
        World-coordinate path.
    """
    left = perpendicular(body_direction)
    local = np.asarray(path.vertices, dtype=np.float64)
    world = tip + (local[:, 0:1] * body_direction[None, :]) + (local[:, 1:2] * left[None, :])
    return Path(world, path.codes)


def _resolve_fill(
    result: ArrowheadResult,
    fill_mode: str,
    stroke_only: bool,
) -> ArrowheadResult:
    """Apply fill-mode overrides to a primitive result.

    Parameters
    ----------
    result : ArrowheadResult
        Base primitive geometry.
    fill_mode : str
        Either ``"filled"`` or ``"hollow"``.
    stroke_only : bool
        Whether the primitive must remain outline-only.

    Returns
    -------
    ArrowheadResult
        Paint-adjusted result.
    """
    if stroke_only or fill_mode == "filled":
        return result
    return ArrowheadResult(
        filled_paths=[],
        stroked_paths=[*result.stroked_paths, *result.filled_paths],
        trim_contour=result.trim_contour,
        trim_t=result.trim_t,
        stroke_width_scale=max(result.stroke_width_scale, HOLLOW_HEAD_STROKE_SCALE),
    )


def _join_overlap(length: float, body_width: float) -> float:
    """Return the overlap distance used to seat a head into the ribbon body.

    Parameters
    ----------
    length : float
        Arrowhead length in local coordinates.
    body_width : float
        Ribbon body width in local coordinates.

    Returns
    -------
    float
        Overlap distance measured along the ribbon axis.
    """
    return min(length * JOIN_OVERLAP_RATIO, max(body_width * 0.75, JOIN_OVERLAP_LENGTH_FLOOR))


def _open_head_stroke_scale(body_width: float, base_scale: float = OPEN_HEAD_STROKE_SCALE) -> float:
    """Return a stroke scale that grows with thicker ribbon bodies.

    Parameters
    ----------
    body_width : float
        Ribbon body width in local coordinates.
    base_scale : float, default=1.2
        Baseline multiplier for light edges.

    Returns
    -------
    float
        Stroke multiplier used for outline-only heads.
    """
    proportional_boost = min(max(body_width, 0.0), 6.0) * 0.08
    return min(base_scale + proportional_boost, 2.0)


def _ornament_half_width(width: float, body_width: float) -> float:
    """Return the outer half-width used by ornamental open heads.

    Parameters
    ----------
    width : float
        Requested head width in local coordinates.
    body_width : float
        Ribbon body width in local coordinates.

    Returns
    -------
    float
        Ornament half-width. The body width is a hard lower bound so stroked
        heads start from the shaft edges before they flare outward.
    """
    return max(body_width * 0.5, width * 0.5 * OPEN_HEAD_WIDTH_GAIN)


def _joined_polygon_points(
    length: float,
    body_width: float,
    shoulder_x: float,
    upper_points: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Build a symmetric polygon with a ribbon-width neck.

    Parameters
    ----------
    length : float
        Arrowhead length in local coordinates.
    body_width : float
        Ribbon body width in local coordinates.
    shoulder_x : float
        Local x position of the last full-width shoulder.
    upper_points : Sequence[tuple[float, float]]
        Upper-half polygon vertices from tip toward the body.

    Returns
    -------
    list[tuple[float, float]]
        Polygon vertices, excluding the implicit closing point.
    """
    overlap = _join_overlap(length, body_width)
    join_x = max(length - overlap, shoulder_x + FLOAT_EPSILON)
    neck_half_width = max(body_width * 0.5, FLOAT_EPSILON)
    points = list(upper_points)
    points.extend([(join_x, neck_half_width), (length, neck_half_width)])
    points.extend([(length, -neck_half_width), (join_x, -neck_half_width)])
    for x, y in reversed(upper_points[1:]):
        points.append((x, -y))
    return points


def _triangle(length: float, width: float, body_width: float) -> ArrowheadResult:
    """Build a standard triangular head with a ribbon-width neck."""
    shoulder_x = max(length * JOIN_SHOULDER_RATIO, body_width * 0.75)
    return ArrowheadResult(
        filled_paths=[
            _local_path(
                _joined_polygon_points(
                    length,
                    body_width,
                    shoulder_x=shoulder_x,
                    upper_points=[(0.0, 0.0), (shoulder_x, width * 0.5)],
                )
            )
        ],
        stroked_paths=[],
        trim_contour=_local_trim_contour(length - _join_overlap(length, body_width), body_width),
    )


def _inverted_triangle(length: float, width: float, body_width: float) -> ArrowheadResult:
    """Build an inverted triangular head with a ribbon-width neck."""
    overlap = _join_overlap(length, body_width)
    join_x = max(length - overlap, length * 0.3)
    neck_half_width = max(body_width * 0.5, FLOAT_EPSILON)
    return ArrowheadResult(
        filled_paths=[
            _local_path(
                [
                    (0.0, width * 0.5),
                    (join_x, neck_half_width),
                    (length, neck_half_width),
                    (length, -neck_half_width),
                    (join_x, -neck_half_width),
                    (0.0, -width * 0.5),
                ]
            )
        ],
        stroked_paths=[],
        trim_contour=_local_trim_contour(join_x, body_width),
    )


def _diamond(length: float, width: float, body_width: float) -> ArrowheadResult:
    """Build a diamond head with a tightened neck transition."""
    shoulder_x = max(length * 0.48, body_width)
    neck_x = max(length * 0.78, shoulder_x + FLOAT_EPSILON)
    return ArrowheadResult(
        filled_paths=[
            _local_path(
                _joined_polygon_points(
                    length,
                    body_width,
                    shoulder_x=neck_x,
                    upper_points=[
                        (0.0, 0.0),
                        (shoulder_x, width * 0.5),
                        (neck_x, width * 0.18),
                    ],
                )
            )
        ],
        stroked_paths=[],
        trim_contour=_local_trim_contour(length - _join_overlap(length, body_width), body_width),
    )


def _box(length: float, width: float, body_width: float) -> ArrowheadResult:
    """Build a rectangular head with a small body overlap."""
    overlap = _join_overlap(length, body_width)
    join_x = max(length - overlap, length * 0.55)
    neck_half_width = max(body_width * 0.5, FLOAT_EPSILON)
    return ArrowheadResult(
        filled_paths=[
            _local_path(
                [
                    (0.0, width * 0.5),
                    (join_x, width * 0.5),
                    (length, neck_half_width),
                    (length, -neck_half_width),
                    (join_x, -width * 0.5),
                    (0.0, -width * 0.5),
                ]
            )
        ],
        stroked_paths=[],
        trim_contour=_local_trim_contour(join_x, body_width),
    )


def _dot(length: float, width: float, body_width: float) -> ArrowheadResult:
    """Build a filled circular head."""
    radius = min(length, width) * 0.5
    diameter = radius * 2.0
    join_x = max(diameter - _join_overlap(max(length, diameter), body_width), radius)
    return ArrowheadResult(
        filled_paths=[_local_circle(radius, radius)],
        stroked_paths=[],
        trim_contour=_local_trim_contour(join_x, max(body_width, FLOAT_EPSILON)),
    )


def _tee(length: float, width: float, body_width: float) -> ArrowheadResult:
    """Build a tee/bar head as a weighted stroked mark."""
    overlap = _join_overlap(length, body_width)
    bar_x = max(length - overlap, length * 0.35)
    neck_half_width = max(body_width * 0.5, FLOAT_EPSILON)
    path = _local_path([(bar_x, neck_half_width), (bar_x, -neck_half_width)], closed=False)
    return ArrowheadResult(
        filled_paths=[],
        stroked_paths=[path],
        trim_contour=_local_trim_contour(bar_x, max(body_width, FLOAT_EPSILON)),
        stroke_width_scale=_open_head_stroke_scale(body_width, base_scale=TEE_HEAD_STROKE_SCALE),
    )


def _vee(length: float, width: float, body_width: float) -> ArrowheadResult:
    """Build an open vee head with two explicit chevron arms."""
    overlap = _join_overlap(length, body_width)
    join_x = max(length - overlap, length * 0.58)
    outer_half_width = _ornament_half_width(width, body_width)
    return ArrowheadResult(
        filled_paths=[],
        stroked_paths=[
            _local_path([(join_x, outer_half_width), (0.0, 0.0)], closed=False),
            _local_path([(join_x, -outer_half_width), (0.0, 0.0)], closed=False),
        ],
        trim_contour=_local_trim_contour(join_x, body_width),
        stroke_width_scale=_open_head_stroke_scale(body_width),
    )


def _crow(length: float, width: float, body_width: float) -> ArrowheadResult:
    """Build a crow-foot head whose tines emerge from the shaft edges."""
    neck_half_width = max(body_width * 0.5, FLOAT_EPSILON)
    tine_half_width = _ornament_half_width(width * 0.62, body_width)
    center = _local_path([(length, 0.0), (0.0, 0.0)], closed=False)
    left = _local_path(
        [
            (length, neck_half_width),
            (0.0, tine_half_width),
        ],
        closed=False,
    )
    right = _local_path(
        [
            (length, -neck_half_width),
            (0.0, -tine_half_width),
        ],
        closed=False,
    )
    overlap = _join_overlap(length, body_width)
    return ArrowheadResult(
        filled_paths=[],
        stroked_paths=[center, left, right],
        trim_contour=_local_trim_contour(length - overlap, body_width),
        stroke_width_scale=_open_head_stroke_scale(
            body_width,
            base_scale=OPEN_HEAD_STROKE_SCALE + 0.05,
        ),
    )


def _curve(length: float, width: float, body_width: float, invert: bool = False) -> ArrowheadResult:
    """Build a curved outline head that starts from one ribbon edge."""
    sign = -1.0 if invert else 1.0
    overlap = _join_overlap(length, body_width)
    neck_half_width = max(body_width * 0.5, FLOAT_EPSILON)
    outer_half_width = _ornament_half_width(width, body_width)
    vertices = np.array(
        [
            [length, sign * neck_half_width],
            [length * 0.72, sign * neck_half_width],
            [length * 0.26, sign * outer_half_width],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    return ArrowheadResult(
        filled_paths=[],
        stroked_paths=[Path(vertices, codes)],
        trim_contour=_local_trim_contour(length - overlap, body_width),
        stroke_width_scale=_open_head_stroke_scale(
            body_width,
            base_scale=OPEN_HEAD_STROKE_SCALE + 0.1,
        ),
    )


def _simple(length: float, width: float, body_width: float) -> ArrowheadResult:
    """Build a simple filled head with a tighter neck transition."""
    shoulder_x = max(length * 0.55, body_width)
    neck_x = max(length * 0.82, shoulder_x + FLOAT_EPSILON)
    return ArrowheadResult(
        filled_paths=[
            _local_path(
                _joined_polygon_points(
                    length,
                    body_width,
                    shoulder_x=neck_x,
                    upper_points=[
                        (0.0, 0.0),
                        (shoulder_x, width * 0.55),
                        (neck_x, width * 0.32),
                    ],
                )
            )
        ],
        stroked_paths=[],
        trim_contour=_local_trim_contour(length - _join_overlap(length, body_width), body_width),
    )


def _fancy(length: float, width: float, body_width: float) -> ArrowheadResult:
    """Build a stockier filled head with an overlapped neck."""
    shoulder_x = max(length * 0.4, body_width)
    neck_x = max(length * 0.85, shoulder_x + FLOAT_EPSILON)
    return ArrowheadResult(
        filled_paths=[
            _local_path(
                _joined_polygon_points(
                    length,
                    body_width,
                    shoulder_x=neck_x,
                    upper_points=[
                        (0.0, 0.0),
                        (shoulder_x, width * 0.6),
                        (neck_x, width * 0.42),
                    ],
                )
            )
        ],
        stroked_paths=[],
        trim_contour=_local_trim_contour(length - _join_overlap(length, body_width), body_width),
    )


def _wedge(length: float, width: float, body_width: float) -> ArrowheadResult:
    """Build a wedge head with the same neck model as the normal head."""
    return ArrowheadResult(
        filled_paths=[
            _local_path(
                _joined_polygon_points(
                    length,
                    body_width,
                    shoulder_x=max(length * JOIN_SHOULDER_RATIO, body_width * 0.75),
                    upper_points=[
                        (0.0, 0.0),
                        (max(length * JOIN_SHOULDER_RATIO, body_width * 0.75), width * 0.5),
                    ],
                )
            )
        ],
        stroked_paths=[],
        trim_contour=_local_trim_contour(length - _join_overlap(length, body_width), body_width),
    )


def _bracket(length: float, width: float, body_width: float) -> ArrowheadResult:
    """Build a bracket head whose arms begin on the ribbon edges."""
    overlap = _join_overlap(length, body_width)
    bracket_x = max(length - overlap * 0.6, length * 0.56)
    neck_half_width = max(body_width * 0.5, FLOAT_EPSILON)
    bracket_half_width = _ornament_half_width(width, body_width)
    path = _local_path(
        [
            (length, neck_half_width),
            (bracket_x, neck_half_width),
            (bracket_x, bracket_half_width),
            (bracket_x, -bracket_half_width),
            (bracket_x, -neck_half_width),
            (length, -neck_half_width),
        ],
        closed=False,
    )
    return ArrowheadResult(
        filled_paths=[],
        stroked_paths=[path],
        trim_contour=_local_trim_contour(length - overlap, body_width),
        stroke_width_scale=_open_head_stroke_scale(body_width),
    )


def _none(length: float, width: float, body_width: float) -> ArrowheadResult:
    """Build an empty head that only reserves trim space."""
    return ArrowheadResult(
        filled_paths=[],
        stroked_paths=[],
        trim_contour=_local_trim_contour(length * 0.5, max(body_width, FLOAT_EPSILON)),
    )


ARROWHEAD_REGISTRY: Dict[str, PrimitiveSpec] = {
    "normal": PrimitiveSpec("normal", _triangle),
    "inv": PrimitiveSpec("inv", _inverted_triangle),
    "dot": PrimitiveSpec("dot", _dot),
    "diamond": PrimitiveSpec("diamond", _diamond),
    "box": PrimitiveSpec("box", _box),
    "tee": PrimitiveSpec("tee", _tee),
    "bar": PrimitiveSpec("bar", _tee),
    "vee": PrimitiveSpec("vee", _vee, stroke_only=True),
    "crow": PrimitiveSpec("crow", _crow, stroke_only=True),
    "curve": PrimitiveSpec(
        "curve",
        lambda length, width, body_width: _curve(length, width, body_width, invert=False),
        stroke_only=True,
    ),
    "icurve": PrimitiveSpec(
        "icurve",
        lambda length, width, body_width: _curve(length, width, body_width, invert=True),
        stroke_only=True,
    ),
    "simple": PrimitiveSpec("simple", _simple),
    "fancy": PrimitiveSpec("fancy", _fancy),
    "wedge": PrimitiveSpec("wedge", _wedge),
    "bracket": PrimitiveSpec("bracket", _bracket, stroke_only=True),
    "none": PrimitiveSpec("none", _none),
}

ARROWHEAD_ALIASES: Dict[str, str] = {
    "circle": "odot",
    "open": "onormal",
    "odot": "odot",
    "obox": "obox",
    "odiamond": "odiamond",
}

PRIMITIVE_NAMES = sorted(ARROWHEAD_REGISTRY.keys(), key=len, reverse=True)


@dataclass(frozen=True)
class ParsedPrimitive:
    """One parsed arrowhead primitive with modifiers."""

    shape: str
    open_fill: bool
    side: str


def _parse_one(spec: str, start: int) -> Tuple[ParsedPrimitive, int]:
    """Parse one Graphviz-style primitive from a compound spec.

    Parameters
    ----------
    spec : str
        Compound arrow specification.
    start : int
        Parse offset.

    Returns
    -------
    tuple[ParsedPrimitive, int]
        Parsed primitive and next offset.
    """
    open_fill = False
    side = "both"
    index = start
    if index < len(spec) and spec[index] == "o":
        open_fill = True
        index += 1
    if index < len(spec) and spec[index] in {"l", "r"}:
        side = "left" if spec[index] == "l" else "right"
        index += 1
    for name in PRIMITIVE_NAMES:
        if spec.startswith(name, index):
            return ParsedPrimitive(shape=name, open_fill=open_fill, side=side), index + len(name)
    raise ValueError(f"Unknown arrowhead spec near {spec[start:]!r}.")


def parse_arrowhead_spec(spec: str) -> List[ParsedPrimitive]:
    """Parse a Graphviz-style arrowhead specification.

    Parameters
    ----------
    spec : str
        Compound arrowhead string.

    Returns
    -------
    list[ParsedPrimitive]
        Parsed primitives from tip to body.
    """
    normalized = ARROWHEAD_ALIASES.get(spec, spec)
    if normalized in {"odot", "obox", "odiamond"}:
        return [ParsedPrimitive(shape=normalized[1:], open_fill=True, side="both")]
    if normalized == "none":
        return [ParsedPrimitive(shape="none", open_fill=False, side="both")]

    index = 0
    primitives: List[ParsedPrimitive] = []
    while index < len(normalized) and len(primitives) < 4:
        primitive, index = _parse_one(normalized, index)
        primitives.append(primitive)
    if index != len(normalized):
        raise ValueError(f"Could not fully parse arrowhead spec {spec!r}.")
    return primitives


def _translated_path(path: Path, dx: float) -> Path:
    """Translate a local path along the body axis.

    Parameters
    ----------
    path : matplotlib.path.Path
        Local path.
    dx : float
        Translation amount.

    Returns
    -------
    matplotlib.path.Path
        Shifted path.
    """
    vertices = np.array(path.vertices, copy=True)
    vertices[:, 0] += dx
    return Path(vertices, path.codes)


def _local_extent_x(result: ArrowheadResult) -> float:
    """Return the body-axis extent of a local arrowhead result.

    Parameters
    ----------
    result : ArrowheadResult
        Local-space arrowhead geometry.

    Returns
    -------
    float
        Maximum x extent across the result geometry.
    """
    max_x = float(np.max(result.trim_contour.vertices[:, 0]))
    for path in [*result.filled_paths, *result.stroked_paths]:
        max_x = max(max_x, float(np.max(path.vertices[:, 0])))
    return max_x


def _compound_gap(length: float, width: float, body_width: float) -> float:
    """Return the tangent spacing inserted between compound primitives.

    Parameters
    ----------
    length : float
        Primitive length in local coordinates.
    width : float
        Primitive width in local coordinates.
    body_width : float
        Ribbon body width in local coordinates.

    Returns
    -------
    float
        Tangent gap between adjacent compound markers.
    """
    return max(
        COMPOUND_HEAD_GAP_FLOOR,
        body_width * COMPOUND_HEAD_GAP_RATIO,
        min(length, width) * 0.18,
    )


def build_arrowhead(
    spec: str,
    tip: Sequence[float],
    tangent: Sequence[float],
    length: float,
    width: float,
    body_width: float | None = None,
    fill_mode: str = "filled",
) -> ArrowheadResult:
    """Build a world-coordinate arrowhead result.

    Parameters
    ----------
    spec : str
        Arrowhead name or compound Graphviz spec.
    tip : Sequence[float]
        Tip position in world coordinates.
    tangent : Sequence[float]
        Vector pointing from the tip back into the body.
    length : float
        Base arrowhead length in data units.
    width : float
        Base arrowhead width in data units.
    body_width : float | None, default=None
        Ribbon body width at the arrowhead junction in data units.
    fill_mode : str, default="filled"
        Either ``"filled"`` or ``"hollow"``.

    Returns
    -------
    ArrowheadResult
        Arrowhead geometry in world coordinates.
    """
    tip_point = as_point(tip)
    body_direction = unit_vector(as_point(tangent))
    resolved_body_width = float(width if body_width is None else body_width)
    parsed = parse_arrowhead_spec(spec)

    local_results: List[ArrowheadResult] = []
    offset = 0.0
    for index, primitive in enumerate(parsed):
        registry_key = primitive.shape
        if registry_key not in ARROWHEAD_REGISTRY:
            raise ValueError(f"Unsupported arrowhead primitive: {registry_key!r}.")
        spec_entry = ARROWHEAD_REGISTRY[registry_key]
        primitive_fill_mode = "hollow" if primitive.open_fill else fill_mode
        primitive_length = float(length)
        primitive_width = float(width)
        if primitive_fill_mode == "hollow" and not spec_entry.stroke_only:
            primitive_length *= HOLLOW_HEAD_DIMENSION_SCALE
            primitive_width *= HOLLOW_HEAD_DIMENSION_SCALE
        base_result = spec_entry.builder(primitive_length, primitive_width, resolved_body_width)
        clipped_result = ArrowheadResult(
            filled_paths=[
                _clip_local_path(path, primitive.side) for path in base_result.filled_paths
            ],
            stroked_paths=[
                _clip_local_path(path, primitive.side) for path in base_result.stroked_paths
            ],
            trim_contour=_clip_local_path(base_result.trim_contour, primitive.side),
            trim_t=base_result.trim_t,
            stroke_width_scale=base_result.stroke_width_scale,
        )
        resolved_result = _resolve_fill(
            clipped_result,
            fill_mode=primitive_fill_mode,
            stroke_only=spec_entry.stroke_only,
        )
        translated_result = ArrowheadResult(
            filled_paths=[_translated_path(path, offset) for path in resolved_result.filled_paths],
            stroked_paths=[
                _translated_path(path, offset) for path in resolved_result.stroked_paths
            ],
            trim_contour=_translated_path(resolved_result.trim_contour, offset),
            trim_t=resolved_result.trim_t,
            stroke_width_scale=resolved_result.stroke_width_scale,
        )
        local_results.append(translated_result)
        next_offset = _local_extent_x(translated_result)
        if index < len(parsed) - 1:
            next_offset += _compound_gap(primitive_length, primitive_width, resolved_body_width)
        offset = next_offset

    composed = ArrowheadResult.compose(local_results)
    return ArrowheadResult(
        filled_paths=[
            _transform_path(path, tip_point, body_direction) for path in composed.filled_paths
        ],
        stroked_paths=[
            _transform_path(path, tip_point, body_direction) for path in composed.stroked_paths
        ],
        trim_contour=_transform_path(composed.trim_contour, tip_point, body_direction),
        trim_t=composed.trim_t,
        stroke_width_scale=composed.stroke_width_scale,
    )


def arrowhead_back_point(result: ArrowheadResult) -> Point:
    """Return the midpoint of the trim contour.

    Parameters
    ----------
    result : ArrowheadResult
        Arrowhead geometry.

    Returns
    -------
    numpy.ndarray
        Midpoint of the trim contour.
    """
    vertices = np.asarray(result.trim_contour.vertices, dtype=np.float64)
    if vertices.shape[0] < 2:
        raise ValueError("Trim contour must contain at least two vertices.")
    return vertices[:2].mean(axis=0)


def available_arrowheads() -> List[str]:
    """Return the supported built-in arrowhead names.

    Returns
    -------
    list[str]
        Sorted built-in names, including aliases.
    """
    names = set(ARROWHEAD_REGISTRY)
    names.update({"circle", "open", "odot", "obox", "odiamond"})
    return sorted(names)
