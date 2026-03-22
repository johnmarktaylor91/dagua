"""Batched custom edge rendering for matplotlib."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from dagua.render.edges.arrowheads import ArrowheadResult, arrowhead_back_point, build_arrowhead
from dagua.render.edges.dashes import DashPattern, DashSegment, dash_curve
from dagua.render.edges.geometry import (
    FLOAT_EPSILON,
    CubicBezier,
    build_arc_length_table,
    mean_curve_width,
    offset_cubic_control_points,
    sample_curve,
    subcurve,
    t_at_arc_length,
    validate_lane_separation,
    vector_norm,
)
from dagua.render.edges.labels import EdgeLabelPlacement, place_edge_label
from dagua.render.edges.ribbon import curve_ribbon_path, simple_quad_ribbon

RenderTier = str
DEFAULT_BODY_COLOR = "#8C8C8C"
DEFAULT_ALPHA = 0.7
DEFAULT_STROKE_WIDTH = 0.75
MAX_SEPARATION_RETRIES = 6
HEAD_DENSITY_ANGLE_DEGREES = 15.0
HEAD_DENSITY_FALLBACK_COUNT = 10
HEAD_DENSITY_HIDE_COUNT = 14
MIN_DENSE_HEAD_SCALE = 0.45
MIN_ARROW_LENGTH_FACTOR = 1.6
MIN_ARROW_WIDTH_FACTOR = 1.15
THICK_STROKED_HEAD_GAIN = 0.05
THICK_STROKED_HEAD_CAP = 1.4
MIN_RENDER_WIDTH = 0.5
ARROW_LENGTH_WIDTH_FLOOR = 3.0
ARROW_WIDTH_WIDTH_FLOOR = 2.5
SHORT_EDGE_HEAD_FRACTION = 0.72
SHORT_EDGE_HEAD_FRACTION_BOTH_TERMINALS = 0.44
THICK_DASH_THRESHOLD = 4.0
THICK_DASH_CONNECTOR_WIDTH_RATIO = 0.22
THICK_DASH_CONNECTOR_ALPHA = 0.24
THICK_DASH_SEGMENT_WIDTH_RATIO = 0.9
THICK_DOTTED_SEGMENT_WIDTH_RATIO = 0.74
ZERO_LENGTH_LOOP_SCALE = 1.75
ZERO_LENGTH_LOOP_FLOOR = 6.0
LABEL_TERMINAL_MARGIN_FRACTION = 0.55
LABEL_TERMINAL_MARGIN_FLOOR = 0.08
LABEL_TERMINAL_MARGIN_CAP = 0.32
LABEL_CLEARANCE_WIDTH_RATIO = 1.6
LABEL_CLEARANCE_FLOOR = 4.0


def _points_to_data_units(ax: Any, points: float, axis: str) -> float:
    """Convert a display-point distance into data units.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    points : float
        Distance in typographic points.
    axis : str
        Axis selector, either ``"x"`` or ``"y"``.

    Returns
    -------
    float
        Distance in data coordinates.
    """
    pixels = points * ax.figure.dpi / 72.0
    transformed = ax.transData.transform([(0.0, 0.0), (1.0, 1.0)])
    scale = (
        abs(transformed[1][0] - transformed[0][0])
        if axis == "x"
        else abs(transformed[1][1] - transformed[0][1])
    )
    if scale <= 1e-9:
        return 0.0
    return pixels / scale


def _compute_display_scale(ax: Any) -> float:
    """Compute the point-to-data conversion factor for edge-label geometry.

    Parameters
    ----------
    ax : Any
        Matplotlib axes with established limits and aspect ratio.

    Returns
    -------
    float
        Multiplicative factor such that ``data_units = points * scale``.
    """
    scale_x = _points_to_data_units(ax, 1.0, "x")
    scale_y = _points_to_data_units(ax, 1.0, "y")
    scale = min(scale_x, scale_y)
    return scale if scale > 1e-9 else 1.0


@dataclass
class DaguaEdge:
    """Edge geometry and styling for the custom renderer.

    Parameters
    ----------
    curve : CubicBezier
        Edge centerline in data coordinates.
    width : float, default=0.75
        Body width in data units.
    color : str, default="#8C8C8C"
        Body fill color.
    alpha : float, default=0.7
        Body alpha.
    linestyle : str | Sequence[float], default="solid"
        Body dash pattern.
    arrowhead : str, default="normal"
        Head arrow spec.
    tail_arrow : str, default="none"
        Tail arrow spec.
    arrowhead_length : float | None, default=None
        Head arrow length in data units.
    arrowhead_width : float | None, default=None
        Head arrow width in data units.
    tail_arrow_length : float | None, default=None
        Tail arrow length in data units.
    tail_arrow_width : float | None, default=None
        Tail arrow width in data units.
    arrow_fill : str, default="filled"
        ``"filled"`` or ``"hollow"``.
    arrow_color : str | None, default=None
        Arrow color override.
    stroke_width : float, default=0.75
        Stroke width in display points for line tiers and outline-only heads.
    label : str | None, default=None
        Optional edge label.
    label_position : float, default=0.5
        Arc-length fraction for label placement.
    label_offset : float, default=3.0
        Perpendicular label offset in data units.
    label_rotate : bool, default=False
        Whether labels follow the local tangent.
    label_side : str, default="auto"
        Label side hint.
    label_font_size : float, default=7.0
        Font size in points.
    label_font_color : str, default="#111111"
        Label text color.
    label_background : str, default="#FAFAFA"
        Label background color.
    label_font_family : str, default=""
        Font family override.
    label_font_weight : str, default="regular"
        Font weight override.
    group_key : tuple[int, int] | None, default=None
        Parallel-edge grouping key.
    source_node : int | None, default=None
        Source node index when available.
    target_node : int | None, default=None
        Target node index when available.
    """

    curve: CubicBezier
    width: float = 0.75
    color: str = DEFAULT_BODY_COLOR
    alpha: float = DEFAULT_ALPHA
    linestyle: DashPattern = "solid"
    arrowhead: str = "normal"
    tail_arrow: str = "none"
    arrowhead_length: Optional[float] = None
    arrowhead_width: Optional[float] = None
    tail_arrow_length: Optional[float] = None
    tail_arrow_width: Optional[float] = None
    arrow_fill: str = "filled"
    arrow_color: Optional[str] = None
    stroke_width: float = DEFAULT_STROKE_WIDTH
    label: Optional[str] = None
    label_position: float = 0.5
    label_offset: float = 3.0
    label_rotate: bool = False
    label_side: str = "auto"
    label_font_size: float = 7.0
    label_font_color: str = "#111111"
    label_background: str = "#FAFAFA"
    label_font_family: str = ""
    label_font_weight: str = "regular"
    group_key: Optional[Tuple[int, int]] = None
    source_node: Optional[int] = None
    target_node: Optional[int] = None

    def resolved_arrow_length(self) -> float:
        """Return the effective arrowhead length.

        Returns
        -------
        float
            Arrowhead length in data units.
        """
        base_length = (
            float(self.arrowhead_length)
            if self.arrowhead_length is not None
            else max(self.width * 4.0, self.width)
        )
        return max(base_length, self.width * ARROW_LENGTH_WIDTH_FLOOR)

    def resolved_arrow_width(self) -> float:
        """Return the effective arrowhead width.

        Returns
        -------
        float
            Arrowhead width in data units.
        """
        base_width = (
            float(self.arrowhead_width)
            if self.arrowhead_width is not None
            else max(self.width * 3.0, self.width)
        )
        return max(base_width, self.width * ARROW_WIDTH_WIDTH_FLOOR)

    def resolved_tail_arrow_length(self) -> float:
        """Return the effective tail arrow length.

        Returns
        -------
        float
            Tail-arrow length in data units.
        """
        if self.tail_arrow_length is not None:
            return max(float(self.tail_arrow_length), self.width * ARROW_LENGTH_WIDTH_FLOOR)
        return self.resolved_arrow_length()

    def resolved_tail_arrow_width(self) -> float:
        """Return the effective tail arrow width.

        Returns
        -------
        float
            Tail-arrow width in data units.
        """
        if self.tail_arrow_width is not None:
            return max(float(self.tail_arrow_width), self.width * ARROW_WIDTH_WIDTH_FLOOR)
        return self.resolved_arrow_width()


@dataclass(frozen=True)
class PreparedEdge:
    """Prepared edge geometry ready for body/head rendering."""

    edge: DaguaEdge
    lane_curve: CubicBezier
    body_curve: Optional[CubicBezier]
    head_result: Optional[ArrowheadResult]
    tail_result: Optional[ArrowheadResult]


def choose_rendering_tier(num_edges: int) -> RenderTier:
    """Choose the rendering tier from edge count.

    Parameters
    ----------
    num_edges : int
        Number of visible edges.

    Returns
    -------
    str
        One of ``"full"``, ``"simplified"``, ``"lines"``, or ``"bundled"``.
    """
    if num_edges <= 1000:
        return "full"
    if num_edges <= 10000:
        return "simplified"
    if num_edges <= 100000:
        return "lines"
    return "bundled"


def _render_width(width: float) -> float:
    """Return the visible body width used for rasterized ribbon rendering.

    Parameters
    ----------
    width : float
        Requested body width in data units.

    Returns
    -------
    float
        Width clamped to a minimum visible floor.
    """
    return max(float(width), MIN_RENDER_WIDTH)


def _curve_length(curve: CubicBezier) -> float:
    """Return the sampled arc length of a cubic curve.

    Parameters
    ----------
    curve : CubicBezier
        Curve to measure.

    Returns
    -------
    float
        Total arc length in data units.
    """
    return build_arc_length_table(curve).total_length


def _is_degenerate_curve(curve: CubicBezier) -> bool:
    """Return whether a curve has no visible span.

    Parameters
    ----------
    curve : CubicBezier
        Curve to inspect.

    Returns
    -------
    bool
        ``True`` when the curve collapses to a coincident endpoint.
    """
    return _curve_length(curve) <= FLOAT_EPSILON


def _coincident_endpoint_loop(edge: DaguaEdge) -> CubicBezier:
    """Build a visible micro-loop for coincident endpoints.

    Parameters
    ----------
    edge : DaguaEdge
        Degenerate edge whose endpoints coincide.

    Returns
    -------
    CubicBezier
        Loop curve centered on the degenerate endpoint.
    """
    center = np.asarray(edge.curve.p0, dtype=np.float64)
    terminal_extent = max(
        edge.resolved_arrow_length() if edge.arrowhead != "none" else 0.0,
        edge.resolved_tail_arrow_length() if edge.tail_arrow != "none" else 0.0,
    )
    loop_radius = max(
        ZERO_LENGTH_LOOP_FLOOR,
        _render_width(edge.width) * 3.0,
        terminal_extent * ZERO_LENGTH_LOOP_SCALE,
    )
    return CubicBezier.from_points(
        center,
        center + np.array([loop_radius, loop_radius * 1.2], dtype=np.float64),
        center + np.array([-loop_radius, loop_radius * 1.2], dtype=np.float64),
        center,
    )


def _normalize_edge_curve(edge: DaguaEdge) -> DaguaEdge:
    """Replace degenerate centerlines with a visible fallback loop.

    Parameters
    ----------
    edge : DaguaEdge
        Edge to normalize.

    Returns
    -------
    DaguaEdge
        Edge with a renderable centerline.
    """
    if not _is_degenerate_curve(edge.curve):
        return edge
    return replace(edge, curve=_coincident_endpoint_loop(edge))


def _segment_render_width(edge: DaguaEdge) -> float:
    """Return the ribbon width used for one visible dash segment.

    Parameters
    ----------
    edge : DaguaEdge
        Edge being rendered.

    Returns
    -------
    float
        Width used for filled dash ribbons.
    """
    render_width = _render_width(edge.width)
    if not isinstance(edge.linestyle, str):
        return render_width
    if render_width < THICK_DASH_THRESHOLD:
        return render_width
    if edge.linestyle == "dotted":
        return render_width * THICK_DOTTED_SEGMENT_WIDTH_RATIO
    if edge.linestyle in {"dashed", "dashdot"}:
        return render_width * THICK_DASH_SEGMENT_WIDTH_RATIO
    return render_width


def _needs_dash_connector(edge: DaguaEdge) -> bool:
    """Return whether a thick dashed edge should get a continuous under-stroke.

    Parameters
    ----------
    edge : DaguaEdge
        Edge being rendered.

    Returns
    -------
    bool
        ``True`` when a subtle connector stroke should be painted.
    """
    return (
        isinstance(edge.linestyle, str)
        and edge.linestyle in {"dashed", "dashdot"}
        and _render_width(edge.width) >= THICK_DASH_THRESHOLD
    )


def _terminal_dimensions(
    edge: DaguaEdge,
    curve_length: float,
    terminal: str,
    has_both_terminals: bool,
) -> Tuple[float, float]:
    """Return head dimensions clamped to the available visible span.

    Parameters
    ----------
    edge : DaguaEdge
        Edge owning the terminal marker.
    curve_length : float
        Available arc length before terminal trimming.
    terminal : str
        ``"head"`` or ``"tail"``.
    has_both_terminals : bool
        Whether both ends render arrow markers.

    Returns
    -------
    tuple[float, float]
        Terminal ``(length, width)`` in data units.
    """
    if terminal == "head":
        base_length = edge.resolved_arrow_length()
        base_width = edge.resolved_arrow_width()
    else:
        base_length = edge.resolved_tail_arrow_length()
        base_width = edge.resolved_tail_arrow_width()
    if curve_length <= FLOAT_EPSILON:
        return base_length, base_width
    max_fraction = (
        SHORT_EDGE_HEAD_FRACTION_BOTH_TERMINALS if has_both_terminals else SHORT_EDGE_HEAD_FRACTION
    )
    capped_length = min(base_length, curve_length * max_fraction)
    min_length = _render_width(edge.width) * MIN_ARROW_LENGTH_FACTOR
    resolved_length = max(capped_length, min_length)
    max_width = max(_render_width(edge.width) * MIN_ARROW_WIDTH_FACTOR, resolved_length * 0.9)
    resolved_width = min(base_width, max_width)
    resolved_width = max(resolved_width, _render_width(edge.width) * MIN_ARROW_WIDTH_FACTOR)
    return resolved_length, resolved_width


def _label_margin_fraction(edge: DaguaEdge, curve_length: float) -> float:
    """Return the inward label clamp used to keep text clear of terminals.

    Parameters
    ----------
    edge : DaguaEdge
        Edge whose label is being placed.
    curve_length : float
        Reference curve length in data units.

    Returns
    -------
    float
        Fractional margin applied at both curve ends.
    """
    if curve_length <= FLOAT_EPSILON:
        return LABEL_TERMINAL_MARGIN_FLOOR
    terminal_extent = 0.0
    if edge.arrowhead != "none":
        terminal_extent = max(terminal_extent, edge.resolved_arrow_length())
    if edge.tail_arrow != "none":
        terminal_extent = max(terminal_extent, edge.resolved_tail_arrow_length())
    if terminal_extent <= FLOAT_EPSILON:
        return LABEL_TERMINAL_MARGIN_FLOOR
    margin = (terminal_extent * LABEL_TERMINAL_MARGIN_FRACTION) / curve_length
    return min(max(margin, LABEL_TERMINAL_MARGIN_FLOOR), LABEL_TERMINAL_MARGIN_CAP)


def _label_offset(edge: DaguaEdge) -> float:
    """Return the visible label clearance away from the edge body.

    Parameters
    ----------
    edge : DaguaEdge
        Edge whose label is being placed.

    Returns
    -------
    float
        Perpendicular label offset in data units.
    """
    return max(
        edge.label_offset,
        _render_width(edge.width) * LABEL_CLEARANCE_WIDTH_RATIO,
        LABEL_CLEARANCE_FLOOR,
    )


def _tail_body_direction(curve: CubicBezier) -> np.ndarray:
    """Return the direction from the tail tip into the edge body."""
    tangent = curve.cp1 - curve.p0
    if vector_norm(tangent) <= FLOAT_EPSILON:
        tangent = curve.p1 - curve.p0
    return tangent


def _head_body_direction(curve: CubicBezier) -> np.ndarray:
    """Return the direction from the head tip into the edge body."""
    tangent = curve.cp2 - curve.p1
    if vector_norm(tangent) <= FLOAT_EPSILON:
        tangent = curve.p0 - curve.p1
    return tangent


def _terminal_face(direction: np.ndarray) -> str:
    """Bucket a terminal tangent into a coarse node-face label.

    Parameters
    ----------
    direction : numpy.ndarray
        Vector pointing from the terminal tip back into the edge body.

    Returns
    -------
    str
        One of ``"east"``, ``"west"``, ``"north"``, or ``"south"``.
    """
    if abs(float(direction[0])) >= abs(float(direction[1])):
        return "east" if float(direction[0]) >= 0.0 else "west"
    return "north" if float(direction[1]) >= 0.0 else "south"


def _fallback_terminal_key(tip: np.ndarray, direction: np.ndarray) -> Tuple[str, int, int, str]:
    """Build a coarse terminal-grouping key when node ids are unavailable.

    Parameters
    ----------
    tip : numpy.ndarray
        Terminal tip position in data coordinates.
    direction : numpy.ndarray
        Vector pointing from the terminal tip back into the edge body.

    Returns
    -------
    tuple[str, int, int, str]
        Fallback grouping key.
    """
    return (
        "tip",
        int(round(float(tip[0]) * 10.0)),
        int(round(float(tip[1]) * 10.0)),
        _terminal_face(direction),
    )


def _terminal_angle(direction: np.ndarray) -> float:
    """Return the terminal angle in degrees.

    Parameters
    ----------
    direction : numpy.ndarray
        Vector pointing from the terminal tip back into the edge body.

    Returns
    -------
    float
        Terminal angle in degrees on ``[0, 360)``.
    """
    angle = float(np.degrees(np.arctan2(float(direction[1]), float(direction[0]))))
    return angle % 360.0


def _nearest_angular_separation(angles: Sequence[float], index: int) -> float:
    """Return the nearest angular separation for one terminal direction.

    Parameters
    ----------
    angles : Sequence[float]
        Terminal directions in degrees.
    index : int
        Angle index to evaluate.

    Returns
    -------
    float
        Smallest pairwise separation in degrees. Singletons return ``360.0``.
    """
    if len(angles) <= 1:
        return 360.0
    anchor = angles[index]
    separations = []
    for offset, other in enumerate(angles):
        if offset == index:
            continue
        delta = abs(anchor - other)
        separations.append(min(delta, 360.0 - delta))
    return min(separations)


def _scaled_head_size(edge: DaguaEdge, scale: float, terminal: str) -> Tuple[float, float]:
    """Return density-adjusted head dimensions for one terminal.

    Parameters
    ----------
    edge : DaguaEdge
        Edge style to scale.
    scale : float
        Multiplicative head scale.
    terminal : str
        ``"head"`` or ``"tail"``.

    Returns
    -------
    tuple[float, float]
        Scaled ``(length, width)`` in data units.
    """
    if terminal == "head":
        base_length = edge.resolved_arrow_length()
        base_width = edge.resolved_arrow_width()
    else:
        base_length = edge.resolved_tail_arrow_length()
        base_width = edge.resolved_tail_arrow_width()
    scaled_length = max(_render_width(edge.width) * MIN_ARROW_LENGTH_FACTOR, base_length * scale)
    scaled_width = max(_render_width(edge.width) * MIN_ARROW_WIDTH_FACTOR, base_width * scale)
    return scaled_length, scaled_width


def _stroked_head_linewidth(edge: DaguaEdge, result: ArrowheadResult) -> float:
    """Return the display stroke width for one outline-style arrowhead.

    Parameters
    ----------
    edge : DaguaEdge
        Edge owning the arrowhead.
    result : ArrowheadResult
        Prepared arrowhead geometry.

    Returns
    -------
    float
        Stroke width in display points.
    """
    base_width = float(edge.stroke_width) * result.stroke_width_scale
    proportional_boost = 1.0 + (
        min(max(float(edge.stroke_width), 0.0), 8.0) * THICK_STROKED_HEAD_GAIN
    )
    return base_width * min(proportional_boost, THICK_STROKED_HEAD_CAP)


def _apply_density_rule(
    edge: DaguaEdge,
    terminal: str,
    min_angle: float,
    count: int,
) -> DaguaEdge:
    """Return an edge with density-aware head adjustments applied.

    Parameters
    ----------
    edge : DaguaEdge
        Edge to adjust.
    terminal : str
        ``"head"`` or ``"tail"``.
    min_angle : float
        Smallest angular separation in the local terminal group, in degrees.
    count : int
        Number of edges using the same node face.

    Returns
    -------
    DaguaEdge
        Edge with adjusted arrowhead style and size.
    """
    scale = 1.0
    if min_angle < HEAD_DENSITY_ANGLE_DEGREES:
        fraction = max(min_angle, 0.0) / HEAD_DENSITY_ANGLE_DEGREES
        scaled_fraction = MIN_DENSE_HEAD_SCALE + (1.0 - MIN_DENSE_HEAD_SCALE) * fraction
        scale = max(MIN_DENSE_HEAD_SCALE, scaled_fraction)

    spec = edge.arrowhead if terminal == "head" else edge.tail_arrow
    if count > HEAD_DENSITY_FALLBACK_COUNT:
        spec = "none" if count >= HEAD_DENSITY_HIDE_COUNT and min_angle < 8.0 else "tee"
        scale = min(scale, 0.7)

    if terminal == "head":
        length, width = _scaled_head_size(edge, scale, terminal="head")
        return replace(edge, arrowhead=spec, arrowhead_length=length, arrowhead_width=width)
    length, width = _scaled_head_size(edge, scale, terminal="tail")
    return replace(edge, tail_arrow=spec, tail_arrow_length=length, tail_arrow_width=width)


def _apply_terminal_density_rules(edges: Sequence[DaguaEdge]) -> List[DaguaEdge]:
    """Shrink or simplify crowded terminal markers.

    Parameters
    ----------
    edges : Sequence[DaguaEdge]
        Lane-adjusted edges.

    Returns
    -------
    list[DaguaEdge]
        Density-adjusted edges.
    """
    updated_edges = list(edges)
    terminal_specs = (
        ("head", "arrowhead", "target_node", "p1", _head_body_direction),
        ("tail", "tail_arrow", "source_node", "p0", _tail_body_direction),
    )
    for terminal, arrow_attr, node_attr, tip_attr, direction_fn in terminal_specs:
        groups: Dict[Tuple[object, str], List[Tuple[int, float]]] = {}
        for index, edge in enumerate(updated_edges):
            spec = getattr(edge, arrow_attr)
            if spec == "none":
                continue
            tip = np.asarray(getattr(edge.curve, tip_attr), dtype=np.float64)
            direction = direction_fn(edge.curve)
            node_key = getattr(edge, node_attr)
            resolved_key = (
                node_key if node_key is not None else _fallback_terminal_key(tip, direction),
                _terminal_face(direction),
            )
            groups.setdefault(resolved_key, []).append((index, _terminal_angle(direction)))

        for members in groups.values():
            angles = [angle for _, angle in members]
            count = len(members)
            for offset, (edge_index, _) in enumerate(members):
                min_angle = _nearest_angular_separation(angles, offset)
                updated_edges[edge_index] = _apply_density_rule(
                    updated_edges[edge_index],
                    terminal=terminal,
                    min_angle=min_angle,
                    count=count,
                )
    return updated_edges


def _trimmed_body_curve(
    edge: DaguaEdge, curve: CubicBezier
) -> Tuple[Optional[CubicBezier], Optional[ArrowheadResult], Optional[ArrowheadResult]]:
    """Trim a centerline to leave room for arrowheads.

    Parameters
    ----------
    edge : DaguaEdge
        Edge style and arrow configuration.
    curve : CubicBezier
        Lane-adjusted curve.

    Returns
    -------
    tuple[CubicBezier | None, ArrowheadResult | None, ArrowheadResult | None]
        Trimmed body curve, head result, and tail result.
    """
    head_result: Optional[ArrowheadResult] = None
    tail_result: Optional[ArrowheadResult] = None
    table = build_arc_length_table(curve)
    has_head = edge.arrowhead != "none"
    has_tail = edge.tail_arrow != "none"
    has_both_terminals = has_head and has_tail
    render_width = _render_width(edge.width)

    if has_head:
        head_length, head_width = _terminal_dimensions(
            edge,
            curve_length=table.total_length,
            terminal="head",
            has_both_terminals=has_both_terminals,
        )
        head_result = build_arrowhead(
            edge.arrowhead,
            tip=curve.p1,
            tangent=_head_body_direction(curve),
            length=head_length,
            width=head_width,
            body_width=render_width,
            fill_mode=edge.arrow_fill,
        )
    if has_tail:
        tail_length, tail_width = _terminal_dimensions(
            edge,
            curve_length=table.total_length,
            terminal="tail",
            has_both_terminals=has_both_terminals,
        )
        tail_result = build_arrowhead(
            edge.tail_arrow,
            tip=curve.p0,
            tangent=_tail_body_direction(curve),
            length=tail_length,
            width=tail_width,
            body_width=render_width,
            fill_mode=edge.arrow_fill,
        )

    start_trim = 0.0
    end_trim = table.total_length
    if tail_result is not None:
        start_trim = min(
            vector_norm(arrowhead_back_point(tail_result) - curve.p0), table.total_length
        )
    if head_result is not None:
        end_trim = max(
            table.total_length - vector_norm(arrowhead_back_point(head_result) - curve.p1),
            start_trim,
        )

    if end_trim - start_trim <= render_width:
        return None, head_result, tail_result

    start_t = t_at_arc_length(table, start_trim)
    end_t = t_at_arc_length(table, end_trim)
    trimmed = subcurve(curve, start_t, end_t)

    if head_result is not None:
        head_result = ArrowheadResult(
            filled_paths=head_result.filled_paths,
            stroked_paths=head_result.stroked_paths,
            trim_contour=head_result.trim_contour,
            trim_t=end_t,
            stroke_width_scale=head_result.stroke_width_scale,
        )
    if tail_result is not None:
        tail_result = ArrowheadResult(
            filled_paths=tail_result.filled_paths,
            stroked_paths=tail_result.stroked_paths,
            trim_contour=tail_result.trim_contour,
            trim_t=start_t,
            stroke_width_scale=tail_result.stroke_width_scale,
        )
    return trimmed, head_result, tail_result


def _group_edges(edges: Sequence[DaguaEdge]) -> Dict[Tuple[int, int], List[DaguaEdge]]:
    """Group edges for parallel-lane separation.

    Parameters
    ----------
    edges : Sequence[DaguaEdge]
        Edges to group.

    Returns
    -------
    dict[tuple[int, int], list[DaguaEdge]]
        Grouped edges.
    """
    groups: Dict[Tuple[int, int], List[DaguaEdge]] = {}
    for index, edge in enumerate(edges):
        if edge.group_key is not None:
            key = edge.group_key
        else:
            key = (index, index)
        groups.setdefault(key, []).append(edge)
    return groups


def _lane_offsets(count: int, separation: float) -> List[float]:
    """Return symmetric lane offsets for a group.

    Parameters
    ----------
    count : int
        Number of lanes.
    separation : float
        Lane spacing in data units.

    Returns
    -------
    list[float]
        Signed offsets.
    """
    center = (count - 1) * 0.5
    return [(index - center) * separation for index in range(count)]


def _apply_lane_offsets(edges: Sequence[DaguaEdge]) -> List[DaguaEdge]:
    """Apply validated parallel-lane offsets to grouped edges.

    Parameters
    ----------
    edges : Sequence[DaguaEdge]
        Source edges.

    Returns
    -------
    list[DaguaEdge]
        Edges with updated centerlines.
    """
    updated_edges: List[DaguaEdge] = []
    for grouped_edges in _group_edges(edges).values():
        if len(grouped_edges) == 1:
            updated_edges.append(grouped_edges[0])
            continue

        max_width = max(edge.width for edge in grouped_edges)
        separation = max(max_width * 1.5, 1.0)
        offsets = _lane_offsets(len(grouped_edges), separation)
        centerlines = [
            offset_cubic_control_points(edge.curve, offset)
            for edge, offset in zip(grouped_edges, offsets)
        ]
        retries = 0
        while retries < MAX_SEPARATION_RETRIES:
            valid, _ = validate_lane_separation(centerlines, min_gap=max_width * 1.05, n_samples=50)
            if valid:
                break
            separation *= 1.25
            offsets = _lane_offsets(len(grouped_edges), separation)
            centerlines = [
                offset_cubic_control_points(edge.curve, offset)
                for edge, offset in zip(grouped_edges, offsets)
            ]
            retries += 1

        for edge, lane_curve in zip(grouped_edges, centerlines):
            updated_edges.append(replace(edge, curve=lane_curve))
    return updated_edges


class DaguaEdgeCollection:
    """Batch renderer for custom edges."""

    def __init__(self, edges: Sequence[DaguaEdge], tier: Optional[RenderTier] = None) -> None:
        """Initialize a batched edge collection.

        Parameters
        ----------
        edges : Sequence[DaguaEdge]
            Edges to render.
        tier : str | None, default=None
            Optional tier override.
        """
        normalized_edges = [_normalize_edge_curve(edge) for edge in edges]
        lane_edges = _apply_lane_offsets(normalized_edges)
        self.edges = _apply_terminal_density_rules(lane_edges)
        self.tier = tier or choose_rendering_tier(len(self.edges))
        self.prepared_edges = [
            PreparedEdge(
                edge=edge,
                lane_curve=edge.curve,
                body_curve=body_curve,
                head_result=head_result,
                tail_result=tail_result,
            )
            for edge in self.edges
            for body_curve, head_result, tail_result in [_trimmed_body_curve(edge, edge.curve)]
        ]

    def render(
        self,
        ax: Any,
        label_background_alpha: float = 0.85,
        svg_hover_map: Optional[Dict[str, str]] = None,
    ) -> List[Any]:
        """Render bodies, heads, and labels in the required pass order.

        Parameters
        ----------
        ax : Any
            Matplotlib axes.
        label_background_alpha : float, default=0.85
            Label background opacity.
        svg_hover_map : dict[str, str] | None, default=None
            Optional SVG hover-text accumulator.

        Returns
        -------
        list[Any]
            Created matplotlib artists.
        """
        artists: List[Any] = []
        artists.extend(self.render_bodies(ax))
        artists.extend(self.render_heads(ax))
        artists.extend(
            self.render_labels(
                ax,
                display_scale=_compute_display_scale(ax),
                label_background_alpha=label_background_alpha,
                svg_hover_map=svg_hover_map,
            )
        )
        return artists

    def render_bodies(self, ax: Any) -> List[Any]:
        """Render the body pass for all edges.

        Parameters
        ----------
        ax : Any
            Matplotlib axes.

        Returns
        -------
        list[Any]
            Added body artists.
        """
        from matplotlib.collections import LineCollection, PatchCollection
        from matplotlib.colors import to_rgba
        from matplotlib.patches import PathPatch

        body_paths: List[PathPatch] = []
        body_colors: List[Tuple[float, float, float, float]] = []
        line_segments: List[np.ndarray] = []
        line_widths: List[float] = []
        line_colors: List[Tuple[float, float, float, float]] = []

        for prepared in self.prepared_edges:
            if prepared.body_curve is None:
                continue
            edge = prepared.edge
            render_width = _render_width(edge.width)
            if self.tier in {"lines", "bundled"}:
                dash_segments = dash_curve(
                    prepared.body_curve,
                    edge.linestyle,
                    render_width,
                    align_to_end=prepared.head_result is not None,
                )
                for segment in dash_segments:
                    line_segments.append(sample_curve(segment.curve, 18))
                    line_widths.append(edge.stroke_width)
                    line_colors.append(to_rgba(edge.color, edge.alpha))
                continue

            dash_segments = dash_curve(
                prepared.body_curve,
                edge.linestyle,
                render_width,
                align_to_end=prepared.head_result is not None,
            )
            if _needs_dash_connector(edge):
                connector_width = max(
                    MIN_RENDER_WIDTH,
                    render_width * THICK_DASH_CONNECTOR_WIDTH_RATIO,
                )
                connector_path = (
                    simple_quad_ribbon(prepared.body_curve, connector_width)
                    if self.tier == "simplified"
                    else curve_ribbon_path(prepared.body_curve, width=connector_width)
                )
                body_paths.append(PathPatch(connector_path))
                body_colors.append(to_rgba(edge.color, min(edge.alpha, THICK_DASH_CONNECTOR_ALPHA)))
            if not dash_segments:
                dash_segments = [DashSegment(prepared.body_curve, cap_start="butt", cap_end="butt")]
            segment_width = _segment_render_width(edge)
            for segment in dash_segments:
                if self.tier == "simplified":
                    path = simple_quad_ribbon(segment.curve, segment_width)
                else:
                    path = curve_ribbon_path(
                        segment.curve,
                        width=segment_width,
                        cap_start=segment.cap_start,
                        cap_end=segment.cap_end,
                    )
                body_paths.append(PathPatch(path))
                body_colors.append(to_rgba(edge.color, edge.alpha))

        artists: List[Any] = []
        if body_paths:
            collection = PatchCollection(
                body_paths,
                match_original=False,
                facecolors=body_colors,
                edgecolors="none",
                linewidths=0.0,
                zorder=1,
            )
            ax.add_collection(collection)
            artists.append(collection)
        if line_segments:
            collection = LineCollection(
                line_segments,
                colors=line_colors,
                linewidths=line_widths,
                capstyle="round",
                joinstyle="round",
                zorder=1,
            )
            ax.add_collection(collection)
            artists.append(collection)
        return artists

    def render_heads(self, ax: Any) -> List[Any]:
        """Render the arrowhead pass for all edges.

        Parameters
        ----------
        ax : Any
            Matplotlib axes.

        Returns
        -------
        list[Any]
            Added arrowhead artists.
        """
        from matplotlib.collections import PatchCollection
        from matplotlib.colors import to_rgba
        from matplotlib.patches import PathPatch

        filled_patches: List[PathPatch] = []
        filled_colors: List[Tuple[float, float, float, float]] = []
        stroked_patches: List[PathPatch] = []
        stroked_colors: List[Tuple[float, float, float, float]] = []
        stroked_widths: List[float] = []

        for prepared in self.prepared_edges:
            edge = prepared.edge
            arrow_color = edge.arrow_color or edge.color
            for result in (prepared.head_result, prepared.tail_result):
                if result is None:
                    continue
                for path in result.filled_paths:
                    filled_patches.append(PathPatch(path))
                    filled_colors.append(to_rgba(arrow_color, edge.alpha))
                for path in result.stroked_paths:
                    stroked_patches.append(PathPatch(path))
                    stroked_colors.append(to_rgba(arrow_color, edge.alpha))
                    stroked_widths.append(_stroked_head_linewidth(edge, result))

        artists: List[Any] = []
        if filled_patches:
            filled = PatchCollection(
                filled_patches,
                match_original=False,
                facecolors=filled_colors,
                edgecolors="none",
                linewidths=0.0,
                zorder=2,
            )
            ax.add_collection(filled)
            artists.append(filled)
        if stroked_patches:
            stroked = PatchCollection(
                stroked_patches,
                match_original=False,
                facecolors="none",
                edgecolors=stroked_colors,
                linewidths=stroked_widths,
                capstyle="round",
                joinstyle="round",
                zorder=2,
            )
            ax.add_collection(stroked)
            artists.append(stroked)
        return artists

    def label_placements(self) -> List[Optional[EdgeLabelPlacement]]:
        """Resolve label placements for all edges.

        Returns
        -------
        list[EdgeLabelPlacement | None]
            One placement per prepared edge.
        """
        placements: List[Optional[EdgeLabelPlacement]] = []
        for prepared in self.prepared_edges:
            edge = prepared.edge
            if not edge.label:
                placements.append(None)
                continue
            reference_curve = prepared.body_curve or prepared.lane_curve
            curve_length = _curve_length(reference_curve)
            label_margin = _label_margin_fraction(edge, curve_length)
            label_position = min(max(edge.label_position, label_margin), 1.0 - label_margin)
            placements.append(
                place_edge_label(
                    reference_curve,
                    label_position=label_position,
                    label_offset=_label_offset(edge),
                    label_rotate=edge.label_rotate,
                    label_side=edge.label_side,
                )
            )
        return placements

    def render_labels(
        self,
        ax: Any,
        display_scale: float,
        label_background_alpha: float = 0.85,
        svg_hover_map: Optional[Dict[str, str]] = None,
    ) -> List[Any]:
        """Render labels on top of bodies and heads with data-coordinate paths.

        Parameters
        ----------
        ax : Any
            Matplotlib axes.
        display_scale : float
            Points-to-data conversion factor.
        label_background_alpha : float, default=0.85
            Label background opacity.
        svg_hover_map : dict[str, str] | None, default=None
            Optional SVG hover-text accumulator.

        Returns
        -------
        list[Any]
            Added patch artists.
        """
        from dagua.render.text import DaguaText, render_text

        specs: List[DaguaText] = []
        for edge_index, (prepared, placement) in enumerate(
            zip(self.prepared_edges, self.label_placements())
        ):
            if placement is None or not prepared.edge.label:
                continue
            edge = prepared.edge
            label_text = edge.label
            assert label_text is not None
            specs.append(
                DaguaText(
                    x=placement.x,
                    y=placement.y,
                    text=label_text,
                    font_size=edge.label_font_size,
                    font_family=edge.label_font_family,
                    font_weight=edge.label_font_weight,
                    font_color=edge.label_font_color,
                    ha="center",
                    va="center",
                    rotation=placement.angle_degrees if edge.label_rotate else 0.0,
                    background=edge.label_background,
                    background_alpha=label_background_alpha,
                    background_padding=(
                        edge.label_font_size * 0.15,
                        edge.label_font_size * 0.15,
                    ),
                    background_corner_radius=edge.label_font_size * 0.15,
                    clip_on=False,
                    zorder=4.0,
                    gid=f"dagua-edge-label-{edge_index}",
                )
            )
            if svg_hover_map is not None:
                hover_text = label_text
                svg_hover_map.setdefault(f"dagua-edge-label-{edge_index}", hover_text)
                svg_hover_map.setdefault(
                    f"dagua-edge-label-{edge_index}-background",
                    hover_text,
                )
        return render_text(ax, specs, display_scale, svg_hover_map)


def render_edges(
    ax: Any,
    edges: Sequence[DaguaEdge],
    tier: Optional[RenderTier] = None,
    label_background_alpha: float = 0.85,
) -> List[Any]:
    """Render custom edges with one convenience call.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    edges : Sequence[DaguaEdge]
        Edges to render.
    tier : str | None, default=None
        Optional tier override.
    label_background_alpha : float, default=0.85
        Label background opacity.

    Returns
    -------
    list[Any]
        Created artists.
    """
    collection = DaguaEdgeCollection(edges=edges, tier=tier)
    return collection.render(ax=ax, label_background_alpha=label_background_alpha)


def average_width(edges: Sequence[DaguaEdge]) -> float:
    """Return the average body width of an edge set.

    Parameters
    ----------
    edges : Sequence[DaguaEdge]
        Edge set.

    Returns
    -------
    float
        Mean width in data units.
    """
    return mean_curve_width(edge.width for edge in edges)
