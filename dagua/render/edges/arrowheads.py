"""Arrowhead geometry for custom data-coordinate edge rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from matplotlib.path import Path

from dagua.render.edges.geometry import FLOAT_EPSILON, Point, as_point, perpendicular, unit_vector

ArrowBuilder = Callable[[float, float], "ArrowheadResult"]

LOCAL_CIRCLE_SAMPLES = 24


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
    """

    filled_paths: List[Path]
    stroked_paths: List[Path]
    trim_contour: Path
    trim_t: float = 1.0

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
    )


def _triangle(length: float, width: float) -> ArrowheadResult:
    """Build a standard triangular head."""
    return ArrowheadResult(
        filled_paths=[_local_path([(0.0, 0.0), (length, width * 0.5), (length, -width * 0.5)])],
        stroked_paths=[],
        trim_contour=_local_trim_contour(length, width),
    )


def _inverted_triangle(length: float, width: float) -> ArrowheadResult:
    """Build an inverted triangular head."""
    return ArrowheadResult(
        filled_paths=[_local_path([(0.0, width * 0.5), (0.0, -width * 0.5), (length, 0.0)])],
        stroked_paths=[],
        trim_contour=_local_trim_contour(length, width),
    )


def _diamond(length: float, width: float) -> ArrowheadResult:
    """Build a diamond head."""
    return ArrowheadResult(
        filled_paths=[
            _local_path(
                [
                    (0.0, 0.0),
                    (length * 0.5, width * 0.5),
                    (length, 0.0),
                    (length * 0.5, -width * 0.5),
                ]
            )
        ],
        stroked_paths=[],
        trim_contour=_local_trim_contour(length, width),
    )


def _box(length: float, width: float) -> ArrowheadResult:
    """Build a rectangular head."""
    return ArrowheadResult(
        filled_paths=[
            _local_path(
                [
                    (0.0, width * 0.5),
                    (length, width * 0.5),
                    (length, -width * 0.5),
                    (0.0, -width * 0.5),
                ]
            )
        ],
        stroked_paths=[],
        trim_contour=_local_trim_contour(length, width),
    )


def _dot(length: float, width: float) -> ArrowheadResult:
    """Build a filled circular head."""
    radius = min(length, width) * 0.5
    diameter = radius * 2.0
    return ArrowheadResult(
        filled_paths=[_local_circle(radius, radius)],
        stroked_paths=[],
        trim_contour=_local_trim_contour(diameter, diameter),
    )


def _tee(length: float, width: float) -> ArrowheadResult:
    """Build a tee/bar head."""
    bar_center = max(length * 0.35, width * 0.2)
    half_thickness = max(width * 0.12, length * 0.08)
    path = _local_path(
        [
            (bar_center - half_thickness, width * 0.5),
            (bar_center + half_thickness, width * 0.5),
            (bar_center + half_thickness, -width * 0.5),
            (bar_center - half_thickness, -width * 0.5),
        ]
    )
    return ArrowheadResult(
        filled_paths=[path],
        stroked_paths=[],
        trim_contour=_local_trim_contour(bar_center + half_thickness, width),
    )


def _vee(length: float, width: float) -> ArrowheadResult:
    """Build an open vee head."""
    path = _local_path([(length, width * 0.5), (0.0, 0.0), (length, -width * 0.5)], closed=False)
    return ArrowheadResult(
        filled_paths=[],
        stroked_paths=[path],
        trim_contour=_local_trim_contour(length, width),
    )


def _crow(length: float, width: float) -> ArrowheadResult:
    """Build a crow-foot head."""
    center = _local_path([(0.0, 0.0), (length, 0.0)], closed=False)
    left = _local_path([(0.0, 0.0), (length, width * 0.5)], closed=False)
    right = _local_path([(0.0, 0.0), (length, -width * 0.5)], closed=False)
    return ArrowheadResult(
        filled_paths=[],
        stroked_paths=[center, left, right],
        trim_contour=_local_trim_contour(length * 0.7, width),
    )


def _curve(length: float, width: float, invert: bool = False) -> ArrowheadResult:
    """Build a curved outline head."""
    sign = -1.0 if invert else 1.0
    vertices = np.array(
        [
            [0.0, 0.0],
            [length * 0.3, sign * width * 0.55],
            [length * 0.7, sign * width * 0.55],
            [length, 0.0],
        ],
        dtype=np.float64,
    )
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    return ArrowheadResult(
        filled_paths=[],
        stroked_paths=[Path(vertices, codes)],
        trim_contour=_local_trim_contour(length, width),
    )


def _simple(length: float, width: float) -> ArrowheadResult:
    """Build a simple filled head."""
    return ArrowheadResult(
        filled_paths=[
            _local_path(
                [
                    (0.0, 0.0),
                    (length * 0.55, width * 0.55),
                    (length, width * 0.32),
                    (length * 0.8, 0.0),
                    (length, -width * 0.32),
                    (length * 0.55, -width * 0.55),
                ]
            )
        ],
        stroked_paths=[],
        trim_contour=_local_trim_contour(length, width),
    )


def _fancy(length: float, width: float) -> ArrowheadResult:
    """Build a stockier filled head."""
    return ArrowheadResult(
        filled_paths=[
            _local_path(
                [
                    (0.0, 0.0),
                    (length * 0.4, width * 0.6),
                    (length * 0.85, width * 0.42),
                    (length, 0.0),
                    (length * 0.85, -width * 0.42),
                    (length * 0.4, -width * 0.6),
                ]
            )
        ],
        stroked_paths=[],
        trim_contour=_local_trim_contour(length, width),
    )


def _wedge(length: float, width: float) -> ArrowheadResult:
    """Build a wedge head."""
    return ArrowheadResult(
        filled_paths=[_local_path([(0.0, 0.0), (length, width * 0.5), (length, -width * 0.5)])],
        stroked_paths=[],
        trim_contour=_local_trim_contour(length, width),
    )


def _bracket(length: float, width: float) -> ArrowheadResult:
    """Build a bracket head."""
    path = _local_path(
        [
            (length, width * 0.5),
            (length * 0.2, width * 0.5),
            (length * 0.2, -width * 0.5),
            (length, -width * 0.5),
        ],
        closed=False,
    )
    return ArrowheadResult(
        filled_paths=[],
        stroked_paths=[path],
        trim_contour=_local_trim_contour(length * 0.2, width),
    )


def _none(length: float, width: float) -> ArrowheadResult:
    """Build an empty head that only reserves trim space."""
    return ArrowheadResult(
        filled_paths=[],
        stroked_paths=[],
        trim_contour=_local_trim_contour(length * 0.5, max(width, FLOAT_EPSILON)),
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
        "curve", lambda length, width: _curve(length, width, invert=False), stroke_only=True
    ),
    "icurve": PrimitiveSpec(
        "icurve", lambda length, width: _curve(length, width, invert=True), stroke_only=True
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


def build_arrowhead(
    spec: str,
    tip: Sequence[float],
    tangent: Sequence[float],
    length: float,
    width: float,
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
    fill_mode : str, default="filled"
        Either ``"filled"`` or ``"hollow"``.

    Returns
    -------
    ArrowheadResult
        Arrowhead geometry in world coordinates.
    """
    tip_point = as_point(tip)
    body_direction = unit_vector(as_point(tangent))
    parsed = parse_arrowhead_spec(spec)

    local_results: List[ArrowheadResult] = []
    offset = 0.0
    for primitive in parsed:
        registry_key = primitive.shape
        if registry_key not in ARROWHEAD_REGISTRY:
            raise ValueError(f"Unsupported arrowhead primitive: {registry_key!r}.")
        spec_entry = ARROWHEAD_REGISTRY[registry_key]
        base_result = spec_entry.builder(length, width)
        clipped_result = ArrowheadResult(
            filled_paths=[
                _clip_local_path(path, primitive.side) for path in base_result.filled_paths
            ],
            stroked_paths=[
                _clip_local_path(path, primitive.side) for path in base_result.stroked_paths
            ],
            trim_contour=_clip_local_path(base_result.trim_contour, primitive.side),
            trim_t=base_result.trim_t,
        )
        resolved_result = _resolve_fill(
            clipped_result,
            fill_mode="hollow" if primitive.open_fill else fill_mode,
            stroke_only=spec_entry.stroke_only,
        )
        translated_result = ArrowheadResult(
            filled_paths=[_translated_path(path, offset) for path in resolved_result.filled_paths],
            stroked_paths=[
                _translated_path(path, offset) for path in resolved_result.stroked_paths
            ],
            trim_contour=_translated_path(resolved_result.trim_contour, offset),
            trim_t=resolved_result.trim_t,
        )
        local_results.append(translated_result)
        offset = float(np.max(translated_result.trim_contour.vertices[:, 0]))

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
