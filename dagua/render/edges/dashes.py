"""Arc-length dash placement for cubic edge bodies."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from typing import List, Sequence, Tuple, Union

from dagua.render.edges.geometry import (
    FLOAT_EPSILON,
    ArcLengthTable,
    CubicBezier,
    build_arc_length_table,
    subcurve,
)

DashPattern = Union[str, Sequence[float]]
MIN_BODY_LENGTH = 0.5
DOTTED_ON_RATIO = 0.18
DOTTED_OFF_RATIO = 1.85
TERMINAL_DASH_MIN_RATIO = 0.65


@dataclass(frozen=True)
class DashSegment:
    """Visible dash segment extracted from a cubic curve.

    Parameters
    ----------
    curve : CubicBezier
        Visible sub-curve.
    cap_start : str
        Cap style at the dash start.
    cap_end : str
        Cap style at the dash end.
    """

    curve: CubicBezier
    cap_start: str
    cap_end: str


def parse_dash_pattern(pattern: DashPattern, width: float) -> Tuple[float, ...]:
    """Normalize a dash pattern into data-coordinate on/off lengths.

    Parameters
    ----------
    pattern : str | Sequence[float]
        Dash description.
    width : float
        Edge width in data units. Built-in string styles scale from this value.

    Returns
    -------
    tuple[float, ...]
        Alternating on/off distances.
    """
    scaled_width = max(float(width), 0.25)
    if isinstance(pattern, str):
        if pattern == "solid":
            return ()
        if pattern == "dashed":
            return (4.0 * scaled_width, 2.75 * scaled_width)
        if pattern == "dotted":
            return (DOTTED_ON_RATIO * scaled_width, DOTTED_OFF_RATIO * scaled_width)
        if pattern == "dashdot":
            return (
                4.0 * scaled_width,
                2.2 * scaled_width,
                DOTTED_ON_RATIO * scaled_width,
                2.2 * scaled_width,
            )
        raise ValueError(f"Unsupported dash pattern: {pattern!r}.")

    values = tuple(float(value) for value in pattern)
    if not values:
        return ()
    if any(value <= 0.0 for value in values):
        raise ValueError("Dash pattern values must be positive.")
    if len(values) % 2 == 1:
        return values + values
    return values


def _segment_bounds(table: ArcLengthTable, start: float, stop: float) -> Tuple[float, float]:
    """Convert arc-length bounds into a cubic parameter interval.

    Parameters
    ----------
    table : ArcLengthTable
        Curve lookup table.
    start : float
        Start arc length.
    stop : float
        End arc length.

    Returns
    -------
    tuple[float, float]
        Parametric interval.
    """
    start_t = (
        0.0
        if start <= FLOAT_EPSILON
        else float(
            (table.ts[(table.lengths >= start).argmax()]) if start <= table.total_length else 1.0
        )
    )
    if stop >= table.total_length:
        end_t = 1.0
    else:
        end_t = float(table.ts[(table.lengths >= stop).argmax()])
    if stop > start:
        from dagua.render.edges.geometry import t_at_arc_length

        start_t = t_at_arc_length(table, start)
        end_t = t_at_arc_length(table, stop)
    return start_t, end_t


def dash_curve(
    curve: CubicBezier,
    pattern: DashPattern,
    width: float,
    min_body_length: float = MIN_BODY_LENGTH,
) -> List[DashSegment]:
    """Cut visible dash segments from a cubic centerline.

    Parameters
    ----------
    curve : CubicBezier
        Curve to split.
    pattern : str | Sequence[float]
        Dash description.
    width : float
        Edge width in data units.
    min_body_length : float, default=0.5
        Minimum visible body length.

    Returns
    -------
    list[DashSegment]
        Visible dash sub-curves.
    """
    dash_pattern = parse_dash_pattern(pattern, width)
    if not dash_pattern:
        return [DashSegment(curve=curve, cap_start="butt", cap_end="butt")]

    table = build_arc_length_table(curve)
    total_length = table.total_length
    if total_length <= min_body_length:
        return []
    if total_length <= dash_pattern[0]:
        return [DashSegment(curve=curve, cap_start="round", cap_end="round")]

    visible_segments: List[Tuple[DashSegment, float, float]] = []
    current_length = 0.0
    draw_segment = True

    for part_length in cycle(dash_pattern):
        if current_length >= total_length - FLOAT_EPSILON:
            break
        next_length = min(current_length + part_length, total_length)
        segment_length = next_length - current_length
        minimum_length = min(min_body_length, max(part_length * 0.8, FLOAT_EPSILON))
        if draw_segment and segment_length >= minimum_length:
            start_t, end_t = _segment_bounds(table, current_length, next_length)
            visible_segments.append(
                (
                    DashSegment(
                        curve=subcurve(curve, start_t, end_t),
                        cap_start="round",
                        cap_end="round",
                    ),
                    part_length,
                    next_length,
                )
            )
        current_length = next_length
        draw_segment = not draw_segment

    if visible_segments:
        last_segment, nominal_length, stop_length = visible_segments[-1]
        actual_length = build_arc_length_table(last_segment.curve).total_length
        truncated_tail = (
            stop_length >= total_length - FLOAT_EPSILON
            and actual_length < nominal_length * TERMINAL_DASH_MIN_RATIO
            and len(visible_segments) > 1
        )
        if truncated_tail:
            visible_segments.pop()

    return [segment for segment, _, _ in visible_segments]


def visible_dash_length(pattern: DashPattern, width: float) -> float:
    """Return the first visible dash length for a pattern.

    Parameters
    ----------
    pattern : str | Sequence[float]
        Dash description.
    width : float
        Edge width in data units.

    Returns
    -------
    float
        Leading visible segment length, or ``0.0`` for solid edges.
    """
    parsed = parse_dash_pattern(pattern, width)
    if not parsed:
        return 0.0
    return float(parsed[0])
