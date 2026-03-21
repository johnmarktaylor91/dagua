"""Ribbon dash helpers for data-coordinate node and cluster borders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from matplotlib.path import Path
from numpy.typing import NDArray

from dagua.render.borders.shapes import path_to_closed_vertices
from dagua.render.edges.dashes import DashPattern, parse_dash_pattern
from dagua.render.edges.ribbon import polyline_ribbon_path

FloatArray = NDArray[np.float64]
FLOAT_EPSILON = 1e-9


@dataclass(frozen=True)
class PolylineDashSegment:
    """Visible dash segment extracted from a closed perimeter polyline.

    Parameters
    ----------
    points : numpy.ndarray
        Segment centerline vertices with shape ``[N, 2]``.
    cap_start : str
        Start cap style passed to the ribbon builder.
    cap_end : str
        End cap style passed to the ribbon builder.
    """

    points: FloatArray
    cap_start: str
    cap_end: str


def _segment_caps(pattern: DashPattern) -> tuple[str, str]:
    """Return the cap styles for one visible perimeter segment.

    Parameters
    ----------
    pattern : str | Sequence[float]
        Dash pattern description.

    Returns
    -------
    tuple[str, str]
        Start and end cap styles.
    """

    if pattern == "dotted":
        return "round", "round"
    if pattern == "dashed":
        return "butt", "butt"
    if isinstance(pattern, str):
        return "butt", "butt"
    return "butt", "butt"


def _arc_length_table(points: FloatArray) -> FloatArray:
    """Return cumulative arc lengths for a closed polyline.

    Parameters
    ----------
    points : numpy.ndarray
        Closed polyline with shape ``[N, 2]``.

    Returns
    -------
    numpy.ndarray
        Cumulative lengths with shape ``[N]``.
    """

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(segment_lengths)])


def _interpolate_point(points: FloatArray, lengths: FloatArray, distance: float) -> FloatArray:
    """Interpolate one point at a cumulative arc length.

    Parameters
    ----------
    points : numpy.ndarray
        Closed polyline with shape ``[N, 2]``.
    lengths : numpy.ndarray
        Cumulative lengths with shape ``[N]``.
    distance : float
        Arc length along the polyline.

    Returns
    -------
    numpy.ndarray
        Interpolated point with shape ``[2]``.
    """

    clamped = min(max(distance, 0.0), float(lengths[-1]))
    if clamped <= FLOAT_EPSILON:
        return points[0].copy()
    if lengths[-1] - clamped <= FLOAT_EPSILON:
        return points[-1].copy()
    segment_index = int(np.searchsorted(lengths, clamped, side="right") - 1)
    segment_index = max(0, min(segment_index, points.shape[0] - 2))
    start = points[segment_index]
    stop = points[segment_index + 1]
    start_length = float(lengths[segment_index])
    stop_length = float(lengths[segment_index + 1])
    if stop_length - start_length <= FLOAT_EPSILON:
        return start.copy()
    ratio = (clamped - start_length) / (stop_length - start_length)
    return start + (stop - start) * ratio


def _slice_polyline(
    points: FloatArray,
    lengths: FloatArray,
    start: float,
    stop: float,
) -> FloatArray:
    """Return the closed-polyline segment between two arc-length positions.

    Parameters
    ----------
    points : numpy.ndarray
        Closed polyline with shape ``[N, 2]``.
    lengths : numpy.ndarray
        Cumulative lengths with shape ``[N]``.
    start : float
        Start arc length.
    stop : float
        Stop arc length.

    Returns
    -------
    numpy.ndarray
        Segment points with shape ``[M, 2]``.
    """

    start_point = _interpolate_point(points, lengths, start)
    stop_point = _interpolate_point(points, lengths, stop)
    start_index = int(np.searchsorted(lengths, start, side="right") - 1)
    stop_index = int(np.searchsorted(lengths, stop, side="left"))
    segment_points: List[FloatArray] = [start_point]
    for index in range(start_index + 1, max(stop_index, start_index + 1)):
        point = points[index]
        if not np.allclose(point, segment_points[-1]):
            segment_points.append(point.copy())
    if not np.allclose(stop_point, segment_points[-1]):
        segment_points.append(stop_point)
    if len(segment_points) == 1:
        segment_points.append(stop_point.copy())
    return np.vstack(segment_points)


def dash_segments(
    centerline_path: Path,
    pattern: DashPattern,
    width: float,
) -> List[PolylineDashSegment]:
    """Cut visible dash segments from a closed perimeter centerline.

    Parameters
    ----------
    centerline_path : matplotlib.path.Path
        Closed centerline path for the border body.
    pattern : str | Sequence[float]
        Dash description.
    width : float
        Border width in data units.

    Returns
    -------
    list[PolylineDashSegment]
        Visible dash segments.
    """

    normalized_pattern = parse_dash_pattern(pattern, width)
    if not normalized_pattern:
        return []
    points = path_to_closed_vertices(centerline_path)
    lengths = _arc_length_table(points)
    total_length = float(lengths[-1])
    if total_length <= FLOAT_EPSILON:
        return []

    visible_segments: List[PolylineDashSegment] = []
    current_length = 0.0
    draw_segment = True
    part_index = 0
    while current_length < total_length - FLOAT_EPSILON:
        part_length = float(normalized_pattern[part_index % len(normalized_pattern)])
        next_length = min(current_length + part_length, total_length)
        visible_length = next_length - current_length
        if draw_segment and visible_length > FLOAT_EPSILON:
            segment_points = _slice_polyline(points, lengths, current_length, next_length)
            cap_start, cap_end = _segment_caps(pattern)
            visible_segments.append(
                PolylineDashSegment(
                    points=segment_points,
                    cap_start=cap_start,
                    cap_end=cap_end,
                )
            )
        current_length = next_length
        draw_segment = not draw_segment
        part_index += 1
    return visible_segments


def dash_ribbon_paths(centerline_path: Path, pattern: DashPattern, width: float) -> List[Path]:
    """Build filled ribbon paths for each visible perimeter dash.

    Parameters
    ----------
    centerline_path : matplotlib.path.Path
        Closed centerline path for the border body.
    pattern : str | Sequence[float]
        Dash description.
    width : float
        Border width in data units.

    Returns
    -------
    list[matplotlib.path.Path]
        Closed filled ribbon paths.
    """

    paths: List[Path] = []
    for segment in dash_segments(centerline_path, pattern, width):
        paths.append(
            polyline_ribbon_path(
                segment.points,
                width=width,
                cap_start=segment.cap_start,
                cap_end=segment.cap_end,
            )
        )
    return paths
