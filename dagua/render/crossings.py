"""Edge crossing detection for jump-style rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from dagua.edges import BezierCurve, evaluate_bezier

_PARALLEL_EPSILON = 1e-10
_CROSSING_DEDUP_TOLERANCE = 1e-6
_CURVE_SAMPLE_SEGMENTS = 16
_ENDPOINT_T_EPSILON = 0.05


@dataclass(frozen=True)
class EdgeCrossing:
    """One intersection point between two routed edges."""

    edge_a: int
    edge_b: int
    x: float
    y: float
    t_a: float
    t_b: float
    angle: float = 0.0  # crossing angle in radians (0 = parallel, pi/2 = perpendicular)


def _segment_intersection(
    ax1: float,
    ay1: float,
    ax2: float,
    ay2: float,
    bx1: float,
    by1: float,
    bx2: float,
    by2: float,
) -> Optional[Tuple[float, float, float, float]]:
    """Return the intersection between two line segments.

    Parameters
    ----------
    ax1 : float
        First endpoint x-coordinate of segment A.
    ay1 : float
        First endpoint y-coordinate of segment A.
    ax2 : float
        Second endpoint x-coordinate of segment A.
    ay2 : float
        Second endpoint y-coordinate of segment A.
    bx1 : float
        First endpoint x-coordinate of segment B.
    by1 : float
        First endpoint y-coordinate of segment B.
    bx2 : float
        Second endpoint x-coordinate of segment B.
    by2 : float
        Second endpoint y-coordinate of segment B.

    Returns
    -------
    Optional[tuple[float, float, float, float]]
        ``(x, y, t_a, t_b)`` when the segments intersect, otherwise ``None``.
    """
    dx_a = ax2 - ax1
    dy_a = ay2 - ay1
    dx_b = bx2 - bx1
    dy_b = by2 - by1

    denom = dx_a * dy_b - dy_a * dx_b
    if abs(denom) < _PARALLEL_EPSILON:
        return None

    t_a = ((bx1 - ax1) * dy_b - (by1 - ay1) * dx_b) / denom
    t_b = ((bx1 - ax1) * dy_a - (by1 - ay1) * dx_a) / denom
    if 0.0 <= t_a <= 1.0 and 0.0 <= t_b <= 1.0:
        x = ax1 + t_a * dx_a
        y = ay1 + t_a * dy_a
        return (x, y, t_a, t_b)
    return None


def _sample_curve(curve: BezierCurve, num_segments: int = _CURVE_SAMPLE_SEGMENTS) -> np.ndarray:
    """Sample a bezier curve into straight-line points.

    Parameters
    ----------
    curve : BezierCurve
        Cubic bezier curve to approximate.
    num_segments : int, default=16
        Number of straight segments in the approximation.

    Returns
    -------
    numpy.ndarray
        Sample points with shape ``[num_segments + 1, 2]``.
    """
    sample_count = max(int(num_segments), 1) + 1
    parameters = np.linspace(0.0, 1.0, sample_count, dtype=float)
    return np.array([evaluate_bezier(curve, float(t)) for t in parameters], dtype=float)


def _bbox_overlaps(points_a: np.ndarray, points_b: np.ndarray) -> bool:
    """Return whether two sampled polylines have overlapping bounding boxes.

    Parameters
    ----------
    points_a : numpy.ndarray
        First polyline with shape ``[N, 2]``.
    points_b : numpy.ndarray
        Second polyline with shape ``[M, 2]``.

    Returns
    -------
    bool
        ``True`` when the axis-aligned bounding boxes overlap.
    """
    a_min_x = float(np.min(points_a[:, 0]))
    a_max_x = float(np.max(points_a[:, 0]))
    a_min_y = float(np.min(points_a[:, 1]))
    a_max_y = float(np.max(points_a[:, 1]))
    b_min_x = float(np.min(points_b[:, 0]))
    b_max_x = float(np.max(points_b[:, 0]))
    b_min_y = float(np.min(points_b[:, 1]))
    b_max_y = float(np.max(points_b[:, 1]))
    return not (a_max_x < b_min_x or b_max_x < a_min_x or a_max_y < b_min_y or b_max_y < a_min_y)


def _far_from_endpoints(curve: BezierCurve, x: float, y: float, min_distance: float) -> bool:
    """Return whether a crossing is far enough from a curve's endpoints.

    Parameters
    ----------
    curve : BezierCurve
        Routed edge curve being tested.
    x : float
        Crossing x-coordinate.
    y : float
        Crossing y-coordinate.
    min_distance : float
        Minimum allowed distance from either endpoint.

    Returns
    -------
    bool
        ``True`` when the crossing is not an endpoint artifact.
    """
    if min_distance <= 0.0:
        return True
    start_distance = float(np.hypot(x - curve.p0[0], y - curve.p0[1]))
    end_distance = float(np.hypot(x - curve.p1[0], y - curve.p1[1]))
    return start_distance >= min_distance and end_distance >= min_distance


def _is_duplicate_crossing(
    crossings: Sequence[EdgeCrossing],
    edge_a: int,
    edge_b: int,
    x: float,
    y: float,
) -> bool:
    """Return whether a crossing duplicates one already recorded.

    Parameters
    ----------
    crossings : Sequence[EdgeCrossing]
        Crossings accumulated so far.
    edge_a : int
        First edge index.
    edge_b : int
        Second edge index.
    x : float
        Crossing x-coordinate.
    y : float
        Crossing y-coordinate.

    Returns
    -------
    bool
        ``True`` when an equivalent crossing is already present.
    """
    for crossing in crossings:
        if crossing.edge_a != edge_a or crossing.edge_b != edge_b:
            continue
        if (
            abs(crossing.x - x) <= _CROSSING_DEDUP_TOLERANCE
            and abs(crossing.y - y) <= _CROSSING_DEDUP_TOLERANCE
        ):
            return True
    return False


def detect_crossings(
    curves: Sequence[Optional[BezierCurve]],
    edge_count: int,
    min_distance: float = 2.0,
) -> List[EdgeCrossing]:
    """Detect pairwise crossings between routed edges.

    Parameters
    ----------
    curves : Sequence[Optional[BezierCurve]]
        Routed edge curves. ``None`` entries are skipped.
    edge_count : int
        Number of edges to consider from ``curves``.
    min_distance : float, default=2.0
        Minimum distance from either edge endpoint required to report a
        crossing. This avoids reporting shared-node joins as crossings.

    Returns
    -------
    list[EdgeCrossing]
        Crossing records sorted by edge index and parameter order.
    """
    curve_count = min(max(int(edge_count), 0), len(curves))
    if curve_count < 2:
        return []

    sampled: List[Optional[np.ndarray]] = []
    for edge_index in range(curve_count):
        curve = curves[edge_index]
        sampled.append(None if curve is None else _sample_curve(curve))

    crossings: List[EdgeCrossing] = []
    for edge_a in range(curve_count):
        curve_a = curves[edge_a]
        points_a = sampled[edge_a]
        if curve_a is None or points_a is None:
            continue
        # Skip zero-length edges (same source/target position)
        if points_a.shape[0] >= 2:
            edge_span = float(np.linalg.norm(points_a[-1] - points_a[0]))
            if edge_span < 1e-6:
                continue
        # Skip self-loops (source == target, detected by nearly-closed curve)
        if points_a.shape[0] >= 2 and edge_span < min_distance:
            continue

        segment_count_a = points_a.shape[0] - 1
        for edge_b in range(edge_a + 1, curve_count):
            curve_b = curves[edge_b]
            points_b = sampled[edge_b]
            if curve_b is None or points_b is None or not _bbox_overlaps(points_a, points_b):
                continue
            # Skip zero-length edges
            if points_b.shape[0] >= 2 and float(np.linalg.norm(points_b[-1] - points_b[0])) < 1e-6:
                continue

            segment_count_b = points_b.shape[0] - 1
            for segment_a in range(segment_count_a):
                for segment_b in range(segment_count_b):
                    result = _segment_intersection(
                        float(points_a[segment_a, 0]),
                        float(points_a[segment_a, 1]),
                        float(points_a[segment_a + 1, 0]),
                        float(points_a[segment_a + 1, 1]),
                        float(points_b[segment_b, 0]),
                        float(points_b[segment_b, 1]),
                        float(points_b[segment_b + 1, 0]),
                        float(points_b[segment_b + 1, 1]),
                    )
                    if result is None:
                        continue

                    x, y, local_t_a, local_t_b = result
                    global_t_a = (segment_a + local_t_a) / max(segment_count_a, 1)
                    global_t_b = (segment_b + local_t_b) / max(segment_count_b, 1)
                    if (
                        global_t_a < _ENDPOINT_T_EPSILON
                        or global_t_a > 1.0 - _ENDPOINT_T_EPSILON
                        or global_t_b < _ENDPOINT_T_EPSILON
                        or global_t_b > 1.0 - _ENDPOINT_T_EPSILON
                    ):
                        continue
                    if not _far_from_endpoints(curve_a, x, y, min_distance):
                        continue
                    if not _far_from_endpoints(curve_b, x, y, min_distance):
                        continue
                    if _is_duplicate_crossing(crossings, edge_a, edge_b, x, y):
                        continue

                    # Compute crossing angle from segment directions
                    dx_a = float(points_a[segment_a + 1, 0] - points_a[segment_a, 0])
                    dy_a = float(points_a[segment_a + 1, 1] - points_a[segment_a, 1])
                    dx_b = float(points_b[segment_b + 1, 0] - points_b[segment_b, 0])
                    dy_b = float(points_b[segment_b + 1, 1] - points_b[segment_b, 1])
                    len_a = max((dx_a**2 + dy_a**2) ** 0.5, 1e-10)
                    len_b = max((dx_b**2 + dy_b**2) ** 0.5, 1e-10)
                    cos_angle = abs((dx_a * dx_b + dy_a * dy_b) / (len_a * len_b))
                    crossing_angle = float(np.arccos(np.clip(cos_angle, 0.0, 1.0)))

                    crossings.append(
                        EdgeCrossing(
                            edge_a=edge_a,
                            edge_b=edge_b,
                            x=x,
                            y=y,
                            t_a=global_t_a,
                            t_b=global_t_b,
                            angle=crossing_angle,
                        )
                    )

    return sorted(
        crossings,
        key=lambda crossing: (crossing.edge_a, crossing.t_a, crossing.edge_b, crossing.t_b),
    )
