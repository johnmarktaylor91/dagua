"""Arc-length dash placement for cubic edge bodies."""

from __future__ import annotations

import math
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
# Round caps add one full line width to the visible dot diameter, so the
# centerline span stays near zero for dotted marks that should read as circles.
DOTTED_ON_RATIO = 0.01
DOTTED_OFF_RATIO = 2.0
DASHED_ON_RATIO = 4.0
DASHED_OFF_RATIO = 2.75
DASHDOT_ON_RATIO = 5.0
DASHDOT_OFF_AFTER_DASH_RATIO = 3.0
DASHDOT_DOT_RATIO = 0.01
DASHDOT_OFF_AFTER_DOT_RATIO = 3.0
TERMINAL_DASH_MIN_RATIO = 0.65
THICK_PATTERN_THRESHOLD = 3.25
THICK_PATTERN_GAIN = 0.16
THICK_DASH_ON_SHRINK = 0.9


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
        thick_gain = (
            1.0
            if scaled_width <= THICK_PATTERN_THRESHOLD
            else 1.0 + min((scaled_width - THICK_PATTERN_THRESHOLD) * THICK_PATTERN_GAIN, 0.9)
        )
        if pattern == "solid":
            return ()
        if pattern == "dashed":
            on_length = DASHED_ON_RATIO * scaled_width
            off_length = DASHED_OFF_RATIO * scaled_width
            if scaled_width > THICK_PATTERN_THRESHOLD:
                on_length *= THICK_DASH_ON_SHRINK
                off_length *= thick_gain
            return (on_length, off_length)
        if pattern == "dotted":
            on_length = DOTTED_ON_RATIO * scaled_width
            off_length = DOTTED_OFF_RATIO * scaled_width
            return (on_length, off_length)
        if pattern == "dashdot":
            dash_length = DASHDOT_ON_RATIO * scaled_width
            gap_after_dash = DASHDOT_OFF_AFTER_DASH_RATIO * scaled_width
            dot_length = DASHDOT_DOT_RATIO * scaled_width
            gap_after_dot = DASHDOT_OFF_AFTER_DOT_RATIO * scaled_width
            return (
                dash_length,
                gap_after_dash,
                dot_length,
                gap_after_dot,
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


def _segment_caps(pattern: DashPattern, part_index: int) -> Tuple[str, str]:
    """Return the cap style for one visible pattern segment.

    Parameters
    ----------
    pattern : str | Sequence[float]
        Dash description.
    part_index : int
        Index of the current visible segment inside one full pattern cycle.

    Returns
    -------
    tuple[str, str]
        Start and end cap style for the segment.
    """
    if pattern == "dotted":
        return "round", "round"
    if pattern == "dashdot" and part_index % 4 == 2:
        return "round", "round"
    if pattern in {"dashed", "dashdot"}:
        return "butt", "butt"
    return "round", "round"


def _aligned_start_length(total_length: float, dash_pattern: Sequence[float]) -> float:
    """Return a negative phase shift that ends on a full visible dash.

    Parameters
    ----------
    total_length : float
        Total arc length of the trimmed body.
    dash_pattern : Sequence[float]
        Alternating on/off distances. The first distance is the primary dash.

    Returns
    -------
    float
        Starting arc length. Negative values indicate a clipped initial segment.
    """
    cycle_length = float(sum(dash_pattern))
    if cycle_length <= FLOAT_EPSILON:
        return 0.0
    phase = math.fmod(total_length - float(dash_pattern[0]), cycle_length)
    if phase < 0.0:
        phase += cycle_length
    if phase <= FLOAT_EPSILON:
        return 0.0
    return phase - cycle_length


def dash_curve(
    curve: CubicBezier,
    pattern: DashPattern,
    width: float,
    min_body_length: float = MIN_BODY_LENGTH,
    align_to_end: bool = False,
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
    align_to_end : bool, default=False
        Whether to phase-shift the pattern so a full visible dash reaches the
        end of the curve. This keeps dashed bodies flowing directly into a
        terminal arrowhead instead of leaving a final gap.

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
        cap_start, cap_end = _segment_caps(pattern, 0)
        return [DashSegment(curve=curve, cap_start=cap_start, cap_end=cap_end)]

    visible_segments: List[Tuple[DashSegment, float, float]] = []
    current_length = _aligned_start_length(total_length, dash_pattern) if align_to_end else 0.0
    draw_segment = True

    for part_index in cycle(range(len(dash_pattern))):
        part_length = dash_pattern[part_index]
        if current_length >= total_length - FLOAT_EPSILON:
            break
        next_length = min(current_length + part_length, total_length)
        visible_start = max(current_length, 0.0)
        visible_stop = max(min(next_length, total_length), visible_start)
        segment_length = visible_stop - visible_start
        minimum_length = min(min_body_length, max(part_length * 0.8, FLOAT_EPSILON))
        if draw_segment and segment_length >= minimum_length:
            start_t, end_t = _segment_bounds(table, visible_start, visible_stop)
            cap_start, cap_end = _segment_caps(pattern, part_index)
            visible_segments.append(
                (
                    DashSegment(
                        curve=subcurve(curve, start_t, end_t),
                        cap_start=cap_start,
                        cap_end=cap_end,
                    ),
                    part_length,
                    visible_stop,
                )
            )
        current_length = next_length
        draw_segment = not draw_segment

    if visible_segments:
        last_segment, nominal_length, stop_length = visible_segments[-1]
        actual_length = build_arc_length_table(last_segment.curve).total_length
        truncated_tail = (
            not align_to_end
            and stop_length >= total_length - FLOAT_EPSILON
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
