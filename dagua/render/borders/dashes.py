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
# Higher values shorten visible dashes more aggressively on tight bends so the
# dash cadence does not visually bunch up around corners.
_CURVATURE_DASH_SENSITIVITY = 8.0
# Floor for curvature-based shortening. Even on sharp corners the visible dash
# should keep at least 40% of its nominal length to avoid turning into noise.
_MIN_CURVATURE_SCALE = 0.4


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


def _estimate_curvatures(points: FloatArray) -> FloatArray:
    """Estimate unsigned curvature at each vertex of a closed polyline.

    Parameters
    ----------
    points : numpy.ndarray
        Closed polyline with shape ``[N, 2]``.

    Returns
    -------
    numpy.ndarray
        Unsigned curvature samples with shape ``[N]``.

    Notes
    -----
    The estimator uses the turning angle implied by adjacent polyline segments
    and normalizes by the local triangle area proxy. Endpoints reuse their
    nearest interior sample because closed outlines are represented with the
    first vertex duplicated at the end.
    """

    point_count = points.shape[0]
    curvatures = np.zeros(point_count, dtype=np.float64)
    for index in range(1, point_count - 1):
        edge_in = points[index] - points[index - 1]
        edge_out = points[index + 1] - points[index]
        cross_product = abs(edge_in[0] * edge_out[1] - edge_in[1] * edge_out[0])
        edge_in_length = float(np.linalg.norm(edge_in))
        edge_out_length = float(np.linalg.norm(edge_out))
        chord_length = float(np.linalg.norm(edge_in + edge_out))
        denominator = edge_in_length * edge_out_length * chord_length
        if denominator > FLOAT_EPSILON:
            curvatures[index] = 2.0 * cross_product / denominator
    if point_count > 1:
        curvatures[0] = curvatures[1]
        curvatures[-1] = curvatures[-2]
    return curvatures


def _curvature_scale(curvature: float) -> float:
    """Map one curvature sample to a visible dash-length multiplier.

    Parameters
    ----------
    curvature : float
        Unsigned local curvature in inverse data units.

    Returns
    -------
    float
        Multiplicative scale for the visible dash length.

    Notes
    -----
    Straight runs stay near ``1.0``. As curvature rises the scale decays
    hyperbolically until it hits :data:`_MIN_CURVATURE_SCALE`, which keeps the
    cadence adaptive without erasing dash segments on compact shapes.
    """

    return max(
        1.0 / (1.0 + float(curvature) * _CURVATURE_DASH_SENSITIVITY),
        _MIN_CURVATURE_SCALE,
    )


def _curvature_at_arc_length(
    lengths: FloatArray,
    curvatures: FloatArray,
    target_length: float,
) -> float:
    """Return the sampled curvature nearest to one arc-length position.

    Parameters
    ----------
    lengths : numpy.ndarray
        Cumulative arc lengths with shape ``[N]``.
    curvatures : numpy.ndarray
        Vertex curvature samples with shape ``[N]``.
    target_length : float
        Arc-length position along the polyline.

    Returns
    -------
    float
        Curvature sample associated with the active segment start.
    """

    index = int(np.searchsorted(lengths, target_length, side="right") - 1)
    index = max(0, min(index, curvatures.shape[0] - 1))
    return float(curvatures[index])


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
    curvatures = _estimate_curvatures(points)
    total_length = float(lengths[-1])
    if total_length <= FLOAT_EPSILON:
        return []

    visible_segments: List[PolylineDashSegment] = []
    current_length = 0.0
    draw_segment = True
    part_index = 0
    while current_length < total_length - FLOAT_EPSILON:
        base_part_length = float(normalized_pattern[part_index % len(normalized_pattern)])
        part_length = (
            base_part_length
            if not draw_segment
            else base_part_length
            # Only shorten painted spans. Leaving the gaps untouched preserves
            # the overall rhythm while preventing visible dashes from crowding
            # into high-curvature corners.
            * _curvature_scale(_curvature_at_arc_length(lengths, curvatures, current_length))
        )
        next_length = min(current_length + part_length, total_length)
        visible_length = next_length - current_length
        if draw_segment and visible_length > FLOAT_EPSILON:
            segment_points = _slice_polyline(points, lengths, current_length, next_length)
            if segment_points.shape[0] >= 2:
                # Recompute the polyline slice for each visible span so the
                # ribbon builder sees the local bend geometry rather than a
                # pre-flattened dash approximation.
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
