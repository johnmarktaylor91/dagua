"""Bezier geometry helpers for custom edge rendering.

The custom edge renderer operates entirely in data coordinates. This module
provides the cubic-bezier math used by ribbon generation, dash placement,
arrowhead trimming, and label placement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

FLOAT_EPSILON = 1e-9
MAX_SUBDIVISION_DEPTH = 16

Point = NDArray[np.float64]


def as_point(value: Sequence[float]) -> Point:
    """Convert a sequence into a 2D float64 point array.

    Parameters
    ----------
    value : Sequence[float]
        Two numeric coordinates.

    Returns
    -------
    numpy.ndarray
        Point with shape ``[2]`` and dtype ``float64``.
    """
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (2,):
        raise ValueError(f"Expected point with shape [2], got {point.shape}.")
    return point


def vector_norm(vector: Point) -> float:
    """Return the Euclidean norm of a 2D vector.

    Parameters
    ----------
    vector : numpy.ndarray
        Vector with shape ``[2]``.

    Returns
    -------
    float
        Euclidean length.
    """
    return float(np.hypot(vector[0], vector[1]))


def unit_vector(vector: Point, fallback: Optional[Point] = None) -> Point:
    """Normalize a vector with an optional fallback.

    Parameters
    ----------
    vector : numpy.ndarray
        Vector with shape ``[2]``.
    fallback : numpy.ndarray | None, default=None
        Replacement vector used when ``vector`` is too small to normalize.

    Returns
    -------
    numpy.ndarray
        Unit vector with shape ``[2]``.
    """
    norm = vector_norm(vector)
    if norm > FLOAT_EPSILON:
        return vector / norm
    if fallback is None:
        return np.array([1.0, 0.0], dtype=np.float64)
    fallback_norm = vector_norm(fallback)
    if fallback_norm > FLOAT_EPSILON:
        return fallback / fallback_norm
    return np.array([1.0, 0.0], dtype=np.float64)


def perpendicular(vector: Point) -> Point:
    """Return the left-hand perpendicular of a 2D vector.

    Parameters
    ----------
    vector : numpy.ndarray
        Vector with shape ``[2]``.

    Returns
    -------
    numpy.ndarray
        Perpendicular vector with shape ``[2]``.
    """
    return np.array([-vector[1], vector[0]], dtype=np.float64)


@dataclass(frozen=True)
class CubicBezier:
    """Cubic bezier in data coordinates.

    Parameters
    ----------
    p0 : numpy.ndarray
        Start point with shape ``[2]``.
    cp1 : numpy.ndarray
        First control point with shape ``[2]``.
    cp2 : numpy.ndarray
        Second control point with shape ``[2]``.
    p1 : numpy.ndarray
        End point with shape ``[2]``.
    """

    p0: Point
    cp1: Point
    cp2: Point
    p1: Point

    @classmethod
    def from_points(
        cls,
        p0: Sequence[float],
        cp1: Sequence[float],
        cp2: Sequence[float],
        p1: Sequence[float],
    ) -> "CubicBezier":
        """Build a curve from Python sequences.

        Parameters
        ----------
        p0 : Sequence[float]
            Start point.
        cp1 : Sequence[float]
            First control point.
        cp2 : Sequence[float]
            Second control point.
        p1 : Sequence[float]
            End point.

        Returns
        -------
        CubicBezier
            Normalized cubic curve.
        """
        return cls(as_point(p0), as_point(cp1), as_point(cp2), as_point(p1))

    def control_polygon(self) -> NDArray[np.float64]:
        """Return the four control points as an array.

        Returns
        -------
        numpy.ndarray
            Array with shape ``[4, 2]``.
        """
        return np.vstack([self.p0, self.cp1, self.cp2, self.p1])


@dataclass(frozen=True)
class SamplePoint:
    """Sampled point on a cubic curve.

    Parameters
    ----------
    t : float
        Parametric position in ``[0, 1]``.
    point : numpy.ndarray
        Sample position with shape ``[2]``.
    tangent : numpy.ndarray
        Tangent vector with shape ``[2]``.
    """

    t: float
    point: Point
    tangent: Point


@dataclass(frozen=True)
class ArcLengthTable:
    """Lookup table mapping cubic parameter to accumulated arc length.

    Parameters
    ----------
    ts : numpy.ndarray
        Monotone parametric samples with shape ``[N]``.
    lengths : numpy.ndarray
        Accumulated arc lengths with shape ``[N]``.
    total_length : float
        Final curve length.
    """

    ts: NDArray[np.float64]
    lengths: NDArray[np.float64]
    total_length: float


def evaluate_cubic(curve: CubicBezier, t: float) -> Point:
    """Evaluate a cubic bezier at parameter ``t``.

    Parameters
    ----------
    curve : CubicBezier
        Curve to evaluate.
    t : float
        Parametric position in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        Position with shape ``[2]``.
    """
    clamped_t = float(min(max(t, 0.0), 1.0))
    u = 1.0 - clamped_t
    return (
        (u**3) * curve.p0
        + (3.0 * u * u * clamped_t) * curve.cp1
        + (3.0 * u * clamped_t * clamped_t) * curve.cp2
        + (clamped_t**3) * curve.p1
    )


def tangent_cubic(curve: CubicBezier, t: float) -> Point:
    """Evaluate the first derivative of a cubic bezier.

    Parameters
    ----------
    curve : CubicBezier
        Curve to differentiate.
    t : float
        Parametric position in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        Tangent vector with shape ``[2]``.
    """
    clamped_t = float(min(max(t, 0.0), 1.0))
    u = 1.0 - clamped_t
    tangent = (
        (3.0 * u * u) * (curve.cp1 - curve.p0)
        + (6.0 * u * clamped_t) * (curve.cp2 - curve.cp1)
        + (3.0 * clamped_t * clamped_t) * (curve.p1 - curve.cp2)
    )
    if vector_norm(tangent) > FLOAT_EPSILON:
        return tangent
    return curve.p1 - curve.p0


def second_derivative_cubic(curve: CubicBezier, t: float) -> Point:
    """Evaluate the second derivative of a cubic bezier.

    Parameters
    ----------
    curve : CubicBezier
        Curve to differentiate.
    t : float
        Parametric position in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        Second derivative vector with shape ``[2]``.
    """
    clamped_t = float(min(max(t, 0.0), 1.0))
    u = 1.0 - clamped_t
    return (6.0 * u) * (curve.cp2 - 2.0 * curve.cp1 + curve.p0) + (6.0 * clamped_t) * (
        curve.p1 - 2.0 * curve.cp2 + curve.cp1
    )


def cubic_flatness(curve: CubicBezier) -> float:
    """Measure cubic flatness by control-point deviation from the chord.

    Parameters
    ----------
    curve : CubicBezier
        Curve to measure.

    Returns
    -------
    float
        Maximum distance from the interior control points to the chord.
    """
    chord = curve.p1 - curve.p0
    chord_length = vector_norm(chord)
    if chord_length <= FLOAT_EPSILON:
        return max(vector_norm(curve.cp1 - curve.p0), vector_norm(curve.cp2 - curve.p0))
    distance_1 = abs(np.cross(chord, curve.cp1 - curve.p0)) / chord_length
    distance_2 = abs(np.cross(chord, curve.cp2 - curve.p0)) / chord_length
    return float(max(distance_1, distance_2))


def split_cubic(curve: CubicBezier, t: float = 0.5) -> Tuple[CubicBezier, CubicBezier]:
    """Split a cubic bezier using De Casteljau subdivision.

    Parameters
    ----------
    curve : CubicBezier
        Curve to split.
    t : float, default=0.5
        Split parameter in ``[0, 1]``.

    Returns
    -------
    tuple[CubicBezier, CubicBezier]
        Left and right sub-curves.
    """
    clamped_t = float(min(max(t, 0.0), 1.0))
    p01 = (1.0 - clamped_t) * curve.p0 + clamped_t * curve.cp1
    p12 = (1.0 - clamped_t) * curve.cp1 + clamped_t * curve.cp2
    p23 = (1.0 - clamped_t) * curve.cp2 + clamped_t * curve.p1
    p012 = (1.0 - clamped_t) * p01 + clamped_t * p12
    p123 = (1.0 - clamped_t) * p12 + clamped_t * p23
    p0123 = (1.0 - clamped_t) * p012 + clamped_t * p123
    return (
        CubicBezier(curve.p0, p01, p012, p0123),
        CubicBezier(p0123, p123, p23, curve.p1),
    )


def subcurve(curve: CubicBezier, t0: float, t1: float) -> CubicBezier:
    """Extract the sub-curve spanning ``[t0, t1]``.

    Parameters
    ----------
    curve : CubicBezier
        Source curve.
    t0 : float
        Start parameter.
    t1 : float
        End parameter.

    Returns
    -------
    CubicBezier
        Subdivided curve.
    """
    start = float(min(max(t0, 0.0), 1.0))
    end = float(min(max(t1, 0.0), 1.0))
    if end < start:
        start, end = end, start
    if start <= 0.0 and end >= 1.0:
        return curve
    left, _ = split_cubic(curve, end)
    if start <= 0.0:
        return left
    relative_t = start / end if end > FLOAT_EPSILON else 0.0
    _, sliced = split_cubic(left, relative_t)
    return sliced


def _adaptive_subdivide(
    curve: CubicBezier,
    t0: float,
    t1: float,
    flatness: float,
    depth: int,
    results: List[SamplePoint],
) -> None:
    """Recursively subdivide a cubic curve until it is visually flat.

    Parameters
    ----------
    curve : CubicBezier
        Curve segment under test.
    t0 : float
        Parametric start of the segment in the parent curve.
    t1 : float
        Parametric end of the segment in the parent curve.
    flatness : float
        Maximum allowed flatness deviation in data units.
    depth : int
        Remaining subdivision depth.
    results : list[SamplePoint]
        Output accumulator.

    Returns
    -------
    None
        Appends end samples into ``results``.
    """
    if depth <= 0 or cubic_flatness(curve) <= flatness:
        tangent = tangent_cubic(curve, 1.0)
        results.append(SamplePoint(t1, curve.p1, tangent))
        return

    left, right = split_cubic(curve, 0.5)
    mid = (t0 + t1) * 0.5
    _adaptive_subdivide(left, t0, mid, flatness, depth - 1, results)
    _adaptive_subdivide(right, mid, t1, flatness, depth - 1, results)


def adaptive_subdivide(curve: CubicBezier, flatness: float = 0.75) -> List[SamplePoint]:
    """Sample a cubic bezier using flatness-driven subdivision.

    Parameters
    ----------
    curve : CubicBezier
        Curve to sample.
    flatness : float, default=0.75
        Maximum chord deviation per accepted segment in data units.

    Returns
    -------
    list[SamplePoint]
        Ordered samples including both endpoints.
    """
    start_tangent = tangent_cubic(curve, 0.0)
    results = [SamplePoint(0.0, curve.p0, start_tangent)]
    _adaptive_subdivide(curve, 0.0, 1.0, flatness, MAX_SUBDIVISION_DEPTH, results)
    return results


def polyline_length(points: NDArray[np.float64]) -> float:
    """Return the length of a point polyline.

    Parameters
    ----------
    points : numpy.ndarray
        Polyline vertices with shape ``[N, 2]``.

    Returns
    -------
    float
        Sum of segment lengths.
    """
    if points.shape[0] < 2:
        return 0.0
    deltas = np.diff(points, axis=0)
    return float(np.linalg.norm(deltas, axis=1).sum())


def build_arc_length_table(curve: CubicBezier, n_samples: int = 128) -> ArcLengthTable:
    """Build an arc-length lookup table for a cubic curve.

    Parameters
    ----------
    curve : CubicBezier
        Curve to sample.
    n_samples : int, default=128
        Number of parametric samples.

    Returns
    -------
    ArcLengthTable
        Monotone lookup table.
    """
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2.")
    ts = np.linspace(0.0, 1.0, n_samples, dtype=np.float64)
    points = np.vstack([evaluate_cubic(curve, float(t)) for t in ts])
    lengths = np.zeros(n_samples, dtype=np.float64)
    if n_samples > 1:
        lengths[1:] = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    return ArcLengthTable(ts=ts, lengths=lengths, total_length=float(lengths[-1]))


def t_at_arc_length(table: ArcLengthTable, distance: float) -> float:
    """Invert the arc-length table.

    Parameters
    ----------
    table : ArcLengthTable
        Precomputed lookup table.
    distance : float
        Requested arc length from the start of the curve.

    Returns
    -------
    float
        Approximated parametric ``t`` value.
    """
    if table.total_length <= FLOAT_EPSILON:
        return 0.0
    clamped_distance = float(min(max(distance, 0.0), table.total_length))
    index = int(np.searchsorted(table.lengths, clamped_distance, side="right"))
    if index <= 0:
        return float(table.ts[0])
    if index >= table.lengths.shape[0]:
        return float(table.ts[-1])
    left_length = table.lengths[index - 1]
    right_length = table.lengths[index]
    if right_length - left_length <= FLOAT_EPSILON:
        return float(table.ts[index])
    alpha = (clamped_distance - left_length) / (right_length - left_length)
    return float((1.0 - alpha) * table.ts[index - 1] + alpha * table.ts[index])


def t_at_fraction(table: ArcLengthTable, fraction: float) -> float:
    """Find the parameter at a normalized arc-length fraction.

    Parameters
    ----------
    table : ArcLengthTable
        Precomputed lookup table.
    fraction : float
        Fraction in ``[0, 1]``.

    Returns
    -------
    float
        Approximated parametric ``t`` value.
    """
    clamped_fraction = float(min(max(fraction, 0.0), 1.0))
    return t_at_arc_length(table, table.total_length * clamped_fraction)


def sample_curve(curve: CubicBezier, n_samples: int) -> NDArray[np.float64]:
    """Sample positions along a cubic curve.

    Parameters
    ----------
    curve : CubicBezier
        Curve to sample.
    n_samples : int
        Number of samples.

    Returns
    -------
    numpy.ndarray
        Sampled points with shape ``[N, 2]``.
    """
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2.")
    ts = np.linspace(0.0, 1.0, n_samples, dtype=np.float64)
    return np.vstack([evaluate_cubic(curve, float(t)) for t in ts])


def offset_cubic_control_points(curve: CubicBezier, offset: float) -> CubicBezier:
    """Approximate a parallel centerline by offsetting control points locally.

    Parameters
    ----------
    curve : CubicBezier
        Base curve.
    offset : float
        Signed offset distance in data units.

    Returns
    -------
    CubicBezier
        Offset approximation suitable for parallel-edge lanes.
    """
    if abs(offset) <= FLOAT_EPSILON:
        return curve
    control_points = curve.control_polygon()
    tangent_sources = np.vstack(
        [
            curve.cp1 - curve.p0,
            curve.cp2 - curve.p0,
            curve.p1 - curve.cp1,
            curve.p1 - curve.cp2,
        ]
    )
    chord = curve.p1 - curve.p0
    offset_points: List[Point] = []
    for point, tangent in zip(control_points, tangent_sources):
        normal = perpendicular(unit_vector(tangent, fallback=chord))
        offset_points.append(point + normal * offset)
    return CubicBezier(*offset_points)


def dense_cross_min_distance(
    curve_a: CubicBezier,
    curve_b: CubicBezier,
    n_samples: int = 50,
) -> float:
    """Compute the sampled minimum distance between two curves.

    Parameters
    ----------
    curve_a : CubicBezier
        First curve.
    curve_b : CubicBezier
        Second curve.
    n_samples : int, default=50
        Samples per curve.

    Returns
    -------
    float
        Minimum sampled pairwise distance.
    """
    samples_a = sample_curve(curve_a, n_samples)
    samples_b = sample_curve(curve_b, n_samples)
    distances = np.linalg.norm(samples_a[:, None, :] - samples_b[None, :, :], axis=2)
    return float(np.min(distances))


def validate_lane_separation(
    centerlines: Sequence[CubicBezier],
    min_gap: float,
    n_samples: int = 50,
) -> Tuple[bool, Optional[float]]:
    """Validate adjacent parallel-edge lanes using dense cross-sampling.

    Parameters
    ----------
    centerlines : Sequence[CubicBezier]
        Lane centerlines ordered from one side to the other.
    min_gap : float
        Required minimum distance.
    n_samples : int, default=50
        Samples per lane used for the pairwise check.

    Returns
    -------
    tuple[bool, float | None]
        Success flag and the smallest observed distance when validation fails.
    """
    if len(centerlines) < 2:
        return True, None
    for first, second in zip(centerlines[:-1], centerlines[1:]):
        observed_gap = dense_cross_min_distance(first, second, n_samples=n_samples)
        if observed_gap < min_gap:
            return False, observed_gap
    return True, None


def point_tangent_at_fraction(curve: CubicBezier, fraction: float) -> Tuple[Point, Point, float]:
    """Return the point, tangent, and parameter at an arc-length fraction.

    Parameters
    ----------
    curve : CubicBezier
        Curve to inspect.
    fraction : float
        Arc-length fraction in ``[0, 1]``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, float]
        Position, tangent, and resolved parameter ``t``.
    """
    table = build_arc_length_table(curve)
    t = t_at_fraction(table, fraction)
    return evaluate_cubic(curve, t), tangent_cubic(curve, t), t


def trim_curve_from_end(curve: CubicBezier, trim_distance: float) -> Tuple[CubicBezier, float]:
    """Trim a cubic curve backward from its end by arc length.

    Parameters
    ----------
    curve : CubicBezier
        Source curve.
    trim_distance : float
        Distance removed from the end of the curve.

    Returns
    -------
    tuple[CubicBezier, float]
        Trimmed curve and the retained end parameter in the source curve.
    """
    if trim_distance <= FLOAT_EPSILON:
        return curve, 1.0
    table = build_arc_length_table(curve)
    remaining_length = max(table.total_length - trim_distance, 0.0)
    trim_t = t_at_arc_length(table, remaining_length)
    return subcurve(curve, 0.0, trim_t), trim_t


def curve_turn_angles(samples: Sequence[SamplePoint]) -> List[float]:
    """Compute polyline turn angles between adjacent sampled tangents.

    Parameters
    ----------
    samples : Sequence[SamplePoint]
        Ordered curve samples.

    Returns
    -------
    list[float]
        Turn angles in radians between consecutive unit tangents.
    """
    turns: List[float] = []
    if len(samples) < 3:
        return turns
    previous = unit_vector(samples[0].tangent)
    for sample in samples[1:]:
        current = unit_vector(sample.tangent, fallback=previous)
        dot = float(np.clip(np.dot(previous, current), -1.0, 1.0))
        turns.append(math.acos(dot))
        previous = current
    return turns


def polyline_from_samples(samples: Sequence[SamplePoint]) -> NDArray[np.float64]:
    """Convert sample points into a position array.

    Parameters
    ----------
    samples : Sequence[SamplePoint]
        Ordered samples.

    Returns
    -------
    numpy.ndarray
        Points with shape ``[N, 2]``.
    """
    if not samples:
        return np.zeros((0, 2), dtype=np.float64)
    return np.vstack([sample.point for sample in samples])


def mean_curve_width(widths: Iterable[float]) -> float:
    """Return a stable mean width for tier selection.

    Parameters
    ----------
    widths : Iterable[float]
        Width values in data units.

    Returns
    -------
    float
        Arithmetic mean, or ``0.0`` when empty.
    """
    width_list = [float(width) for width in widths]
    if not width_list:
        return 0.0
    return float(sum(width_list) / len(width_list))
