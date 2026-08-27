"""Crossing, routing, angular, ambiguity, and edge-label facet family."""

# Contract identity docstrings intentionally keep each title and digest on one line.
# ruff: noqa: E501

from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from dagua.eval.ruler_v4._tracing import (
    Scalar,
    as_float,
    keep,
    p_abs,
    p_exp,
    p_log,
    p_max,
    p_min,
    p_sqrt,
    p_sum,
    record_subterm,
    tracing_active,
)
from dagua.eval.ruler_v4._util import (
    aabb_pair,
    blend_with_weights,
    global_blend,
    mean_result,
    proper_intersection,
    resolved_ranks,
    resolved_routes,
    route_segments,
    smoothstep,
    snap_unit,
)
from dagua.eval.ruler_v4.scene import (
    BoxGeometry,
    FacetResult,
    ResultState,
    Route,
    Scene,
    na_result,
    value_result,
)

_ANGULAR_ZERO_ENVELOPE = 1e-12


def _p_sin(value: Scalar) -> Scalar:
    """Sine; ``math.sin`` on floats, ``torch.sin`` on tensors."""

    if isinstance(value, torch.Tensor):
        return torch.sin(value)
    return math.sin(value)


def _p_cos(value: Scalar) -> Scalar:
    """Cosine; ``math.cos`` on floats, ``torch.cos`` on tensors."""

    if isinstance(value, torch.Tensor):
        return torch.cos(value)
    return math.cos(value)


def _p_acos(value: Scalar) -> Scalar:
    """Arccosine; ``math.acos`` on floats, ``torch.acos`` on tensors."""

    if isinstance(value, torch.Tensor):
        return torch.acos(value)
    return math.acos(value)


def _p_asinh(value: Scalar) -> Scalar:
    """Inverse hyperbolic sine; ``math.asinh`` on floats, ``torch.asinh`` on tensors."""

    if isinstance(value, torch.Tensor):
        return torch.asinh(value)
    return math.asinh(value)


def _p_atan2(numerator: Scalar, denominator: Scalar) -> Scalar:
    """Two-argument arctangent; ``math.atan2`` on floats, ``torch.atan2`` on tensors."""

    if isinstance(numerator, torch.Tensor) or isinstance(denominator, torch.Tensor):
        return torch.atan2(_scalar_tensor(numerator), _scalar_tensor(denominator))
    return math.atan2(numerator, denominator)


def _p_mod(value: Scalar, modulus: float) -> Scalar:
    """Modulo; Python ``%`` on floats, ``torch.remainder`` on tensors."""

    if isinstance(value, torch.Tensor):
        return torch.remainder(value, modulus)
    return value % modulus


def _scalar_tensor(value: Scalar) -> torch.Tensor:
    """Promote one scalar to a float64 tensor, preserving any graph."""

    if isinstance(value, torch.Tensor):
        return value
    return torch.tensor(float(value), dtype=torch.float64)


def _smooth_fade(value: Scalar) -> Scalar:
    """Evaluate the shared quintic-smoothstep fade on one scalar.

    Parameters
    ----------
    value : float or torch.Tensor
        Fade argument in the gate's own units.

    Returns
    -------
    float or torch.Tensor
        ``float(smoothstep(tensor(value)))`` on the exact path (the historical
        expression, bit-identical), or the live smoothstep tensor in a trace.
    """

    if isinstance(value, torch.Tensor):
        return smoothstep(value)
    return float(smoothstep(torch.tensor(value, dtype=torch.float64)))


def _norm_or_zero(vector: torch.Tensor) -> Scalar:
    """Keep a Euclidean norm, detaching only the exact-zero boundary.

    ``vector_norm`` has an undefined (NaN) gradient at the zero vector; every
    consumer reads the distance through an even or hinged kernel whose slope at
    exact contact is zero, so the detached 0.0 is the exact subgradient there.

    Parameters
    ----------
    vector : torch.Tensor
        Difference vector with shape ``[2]``.

    Returns
    -------
    float or torch.Tensor
        ``keep(norm)`` off the boundary, the float 0.0 exactly on it.
    """

    norm = keep(torch.linalg.vector_norm(vector))
    if as_float(norm) == 0.0:
        return 0.0
    return norm


def _raw(value: Scalar) -> object:
    """Cast one raw diagnostic to its published float, preserving float types.

    Parameters
    ----------
    value : float or torch.Tensor
        Scalar leaving the score-visible chain into a raw= mapping.

    Returns
    -------
    object
        ``float(value.detach().item())`` for tensors, the value unchanged
        otherwise (so historical int/float raw entries keep their exact type).
    """

    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return value


_U07_WORKED_EXAMPLE_GAMMA = 1.0
_U07_WORKED_EXAMPLE_LAMBDA_T = 0.5
_U11_TERMINAL_DISK_SIDES = 16
_U11_TERMINAL_CLEAR_RADIUS = 0.5
VECTORIZED_EXACT_SCORERS = True
_U07_PAIR_BLOCK_SIZE = 262_144
_U07_DENSITY_BLOCK_ELEMENTS = 4_194_304
_U07_DENSITY_GRID_MIN_EVENTS = 4_096
_U07_DENSITY_GRID_CELL_MARGIN = 1e-6
_U07_DENSITY_GRID_MAX_CELL_MAGNITUDE = float(2**30)
_U11_PAIR_OBSTACLE_BUDGET = 262_144
_U11_BROAD_ROW_BLOCK = 8_388_608
_U11_OBSTACLE_SELECTION_BATCH_MIN = 128


@dataclass(frozen=True)
class _CrossingEvent:
    """One exact transversal or collinear U07 crossing event.

    Parameters
    ----------
    edge_a, edge_b : int
        Declared edge indices.
    point : torch.Tensor
        Event point with shape ``[2]``.
    angle : float or torch.Tensor
        Acute crossing angle in radians (live tensor inside a trace).
    proximity : float or torch.Tensor
        Nearest graph-terminal distance in intrinsic-unit multiples.
    pair_multiplicity : int
        Total event count for the unordered edge pair.
    density : float or torch.Tensor
        Local crossing-density argument.
    severity : float or torch.Tensor
        Frozen four-component severity.
    """

    edge_a: int
    edge_b: int
    point: torch.Tensor
    angle: Scalar
    proximity: Scalar
    pair_multiplicity: int
    density: Scalar
    severity: Scalar


def _segment_angle(first: torch.Tensor, second: torch.Tensor) -> Scalar:
    """Return the acute unoriented angle between two vectors.

    Parameters
    ----------
    first, second : torch.Tensor
        Two-dimensional vectors.

    Returns
    -------
    float or torch.Tensor
        Angle in radians in ``[0, pi/2]``; a live tensor inside a trace.
    """

    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    if float(denominator) == 0.0:
        return 0.0
    cosine = p_min(1.0, p_max(-1.0, p_abs(keep(torch.dot(first, second) / denominator))))
    if as_float(cosine) >= 1.0:
        # acos is non-differentiable at the clamp boundary (infinite slope);
        # the historical float value there is exactly acos(1) = 0.
        return math.acos(1.0)
    return _p_acos(cosine)


def _oriented_angle(first: torch.Tensor, second: torch.Tensor) -> Scalar:
    """Return the oriented angle between two direction vectors.

    Unlike :func:`_segment_angle`, no absolute value is taken, so
    anti-parallel vectors read ``pi`` rather than ``0``. U11 sec 5 (v)
    compares terminal tangents that point away from their shared node:
    coincident tangents (angle 0) are the merge-identity limit while
    opposite tangents (angle pi) are maximally distinguishable.

    Parameters
    ----------
    first, second : torch.Tensor
        Two-dimensional vectors.

    Returns
    -------
    float or torch.Tensor
        Angle in radians in ``[0, pi]``; a live tensor inside a trace.
    """

    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    if float(denominator) == 0.0:
        return 0.0
    cosine = p_min(1.0, p_max(-1.0, keep(torch.dot(first, second) / denominator)))
    if as_float(cosine) >= 1.0:
        return math.acos(1.0)
    if as_float(cosine) <= -1.0:
        return math.acos(-1.0)
    return _p_acos(cosine)


def _segment_event_point(
    start_a: torch.Tensor,
    end_a: torch.Tensor,
    start_b: torch.Tensor,
    end_b: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, Scalar]]:
    """Return a proper crossing point and acute angle when one event exists.

    Parameters
    ----------
    start_a, end_a, start_b, end_b : torch.Tensor
        Segment endpoints with shape ``[2]``.

    Returns
    -------
    tuple[torch.Tensor, float or torch.Tensor] or None
        Event point and acute angle. Positive-length collinear overlap produces
        its midpoint with angle zero.
    """

    direction_a = end_a - start_a
    direction_b = end_b - start_b
    cross = keep(direction_a[0] * direction_b[1] - direction_a[1] * direction_b[0])
    if as_float(cross) != 0.0:
        offset = start_b - start_a
        parameter_a = keep(offset[0] * direction_b[1] - offset[1] * direction_b[0]) / cross
        parameter_b = keep(offset[0] * direction_a[1] - offset[1] * direction_a[0]) / cross
        if 0.0 < as_float(parameter_a) < 1.0 and 0.0 < as_float(parameter_b) < 1.0:
            return start_a + parameter_a * direction_a, _segment_angle(direction_a, direction_b)
        return None
    if (
        as_float(direction_a[0] * (start_b - start_a)[1] - direction_a[1] * (start_b - start_a)[0])
        != 0.0
    ):
        return None
    length_squared = keep(torch.dot(direction_a, direction_a))
    if as_float(length_squared) == 0.0:
        return None
    left = keep(torch.dot(start_b - start_a, direction_a)) / length_squared
    right = keep(torch.dot(end_b - start_a, direction_a)) / length_squared
    overlap_start = p_max(0.0, p_min(left, right))
    overlap_end = p_min(1.0, p_max(left, right))
    if as_float(overlap_end) <= as_float(overlap_start):
        return None
    midpoint = start_a + ((overlap_start + overlap_end) / 2.0) * direction_a
    return midpoint, 0.0


def _crossing_events(scene: Scene, gamma: float) -> List[_CrossingEvent]:
    """Compute U07 crossing events and frozen four-component severities.

    Parameters
    ----------
    scene : Scene
        Validated route scene.
    gamma : float
        Positive fitted crossing-severity scale.

    Returns
    -------
    list[_CrossingEvent]
        One event per inter-edge crossing, including adjacent edge pairs.
    """

    segments = route_segments(scene)
    provisional: List[Tuple[int, int, torch.Tensor, Scalar, Scalar]] = []
    for index, (edge_a, _, start_a, end_a) in enumerate(segments):
        terminals_a = scene.graph.edges[edge_a]
        for edge_b, _, start_b, end_b in segments[index + 1 :]:
            if edge_a == edge_b:
                continue
            event = _segment_event_point(start_a, end_a, start_b, end_b)
            if event is None:
                continue
            point, angle = event
            terminals_b = scene.graph.edges[edge_b]
            terminal_points = scene.positions[list((*terminals_a, *terminals_b))]
            proximity = keep(torch.min(torch.linalg.vector_norm(terminal_points - point, dim=1)))
            if as_float(proximity) == 0.0:
                # vector_norm has a NaN gradient exactly at zero; the severity
                # reads proximity only through rho^2, whose slope there is 0.
                proximity = 0.0
            provisional.append((min(edge_a, edge_b), max(edge_a, edge_b), point, angle, proximity))
    multiplicities = Counter((left, right) for left, right, _, _, _ in provisional)
    results = []
    for event_index, (edge_a, edge_b, point, angle, proximity) in enumerate(provisional):
        sine_squared = _p_sin(angle) ** 2
        density: Scalar = 0.0
        for other_index, (_, _, other_point, other_angle, _) in enumerate(provisional):
            if event_index == other_index:
                continue
            normalized_squared = (
                keep(torch.sum((point - other_point) ** 2)) / (6.0 * scene.intrinsic_unit) ** 2
            )
            density += _p_sin(other_angle) ** 2 * p_max(0.0, 1.0 - normalized_squared) ** 2
        multiplicity = multiplicities[(edge_a, edge_b)]
        angle_term = _p_cos(angle) ** 2
        proximity_ratio = proximity / scene.intrinsic_unit
        proximity_term = p_max(0.0, 1.0 - proximity_ratio**2 / 16.0) ** 2
        repeat_term = (multiplicity - 1.0) / multiplicity
        density_term = density / (density + 3.0)
        severity = gamma * (
            0.50 * angle_term
            + sine_squared * (0.20 * proximity_term + 0.15 * density_term)
            + 0.15 * repeat_term
        )
        results.append(
            _CrossingEvent(
                edge_a,
                edge_b,
                point,
                angle,
                proximity_ratio,
                multiplicity,
                density,
                severity,
            )
        )
    return results


def _crossing_candidate_pairs_vectorized(
    segments: Sequence[Tuple[int, int, torch.Tensor, torch.Tensor]],
) -> List[Tuple[int, int]]:
    """Find U07 event-bearing segment pairs with batched float64 decisions.

    Parameters
    ----------
    segments : sequence[tuple[int, int, torch.Tensor, torch.Tensor]]
        Flattened routes in the shipped scorer's canonical segment order.

    Returns
    -------
    list[tuple[int, int]]
        Segment-index pairs in the exact order of the scalar nested sweep.

    Notes
    -----
    Only the event-presence decision is batched. The score-visible event point,
    angle, proximity, density, and severity are rebuilt by the shipped scalar
    arithmetic after this filter. Each batched predicate uses the same float64
    operations in the same per-element order as :func:`_segment_event_point`.
    """

    if len(segments) < 2:
        return []
    edge_indices = torch.tensor([segment[0] for segment in segments], dtype=torch.int64)
    starts = torch.stack([segment[2].detach().cpu() for segment in segments])
    ends = torch.stack([segment[3].detach().cpu() for segment in segments])
    pair_indices = torch.triu_indices(len(segments), len(segments), offset=1)
    candidates: List[Tuple[int, int]] = []
    for block_start in range(0, pair_indices.shape[1], _U07_PAIR_BLOCK_SIZE):
        block = pair_indices[:, block_start : block_start + _U07_PAIR_BLOCK_SIZE]
        left = block[0]
        right = block[1]
        direction_a = ends[left] - starts[left]
        direction_b = ends[right] - starts[right]
        offset = starts[right] - starts[left]
        cross = direction_a[:, 0] * direction_b[:, 1] - direction_a[:, 1] * direction_b[:, 0]
        nonparallel = cross != 0.0
        safe_cross = torch.where(nonparallel, cross, torch.ones_like(cross))
        parameter_a = (
            offset[:, 0] * direction_b[:, 1] - offset[:, 1] * direction_b[:, 0]
        ) / safe_cross
        parameter_b = (
            offset[:, 0] * direction_a[:, 1] - offset[:, 1] * direction_a[:, 0]
        ) / safe_cross
        transversal = (
            nonparallel
            & (parameter_a > 0.0)
            & (parameter_a < 1.0)
            & (parameter_b > 0.0)
            & (parameter_b < 1.0)
        )

        collinear = ~nonparallel & (
            direction_a[:, 0] * offset[:, 1] - direction_a[:, 1] * offset[:, 0] == 0.0
        )
        length_squared = (direction_a * direction_a).sum(dim=1)
        safe_length_squared = torch.where(
            length_squared != 0.0,
            length_squared,
            torch.ones_like(length_squared),
        )
        left_parameter = (offset * direction_a).sum(dim=1) / safe_length_squared
        right_parameter = ((ends[right] - starts[left]) * direction_a).sum(
            dim=1
        ) / safe_length_squared
        overlap_start = torch.maximum(
            torch.zeros_like(left_parameter), torch.minimum(left_parameter, right_parameter)
        )
        overlap_end = torch.minimum(
            torch.ones_like(left_parameter), torch.maximum(left_parameter, right_parameter)
        )
        overlap = collinear & (length_squared != 0.0) & (overlap_end > overlap_start)
        selected = ((edge_indices[left] != edge_indices[right]) & (transversal | overlap)).nonzero(
            as_tuple=False
        )
        candidates.extend(
            (int(left[index]), int(right[index])) for index in selected[:, 0].tolist()
        )
    return candidates


def _event_densities_exact(
    provisional: Sequence[Tuple[int, int, torch.Tensor, Scalar, Scalar]],
    intrinsic_unit: float,
) -> Optional[List[float]]:
    """Accumulate every U07 event density on the exact path, bit-identically.

    The historical per-pair loop computes, for event ``i`` over events
    ``j != i`` in ascending order,

        density_i += sin(angle_j)^2 * max(0, 1 - |p_i - p_j|^2 / (6 iu)^2)^2

    with each operation a single IEEE-754 double op (``keep`` casts the torch
    squared distance to float before the divide). This helper computes the
    same terms as float64 tensor blocks -- each elementwise op is the same
    IEEE double op on the same operands -- and accumulates columns ``j`` in
    ascending order with elementwise tensor adds, so every event's addition
    sequence is the loop's. The skipped ``j == i`` term is zeroed before
    accumulation; all terms are non-negative, so adding ``+0.0`` at that slot
    leaves the accumulator bit-identical.

    Parameters
    ----------
    provisional : sequence[tuple]
        Provisional U07 events ``(edge_a, edge_b, point, angle, proximity)``.
    intrinsic_unit : float
        Scene intrinsic unit.

    Returns
    -------
    list[float] or None
        Per-event densities, or ``None`` when the inputs are not the float64
        exact-path geometry (the caller then runs the historical loop).
    """

    points = [point for _, _, point, _, _ in provisional]
    stacked = torch.stack([point.detach() for point in points])
    if stacked.dtype != torch.float64:
        return None
    weights = torch.tensor(
        [float(_p_sin(angle) ** 2) for _, _, _, angle, _ in provisional],
        dtype=torch.float64,
    )
    denominator = (6.0 * intrinsic_unit) ** 2
    count = len(points)
    accumulator = _event_density_accumulator_grid(stacked, weights, denominator, count)
    if accumulator is None:
        accumulator = _event_density_accumulator_dense(stacked, weights, denominator, count)
    return [float(value) for value in accumulator.tolist()]


def _event_density_accumulator_dense(
    stacked: torch.Tensor,
    weights: torch.Tensor,
    denominator: float,
    count: int,
) -> torch.Tensor:
    """Accumulate every density over all ``count**2`` terms in column blocks.

    This is the historical block accumulation: every event adds every column
    ``j`` in ascending order (the ``j == i`` slot zeroed first), one IEEE
    double add per term.

    Parameters
    ----------
    stacked : torch.Tensor
        ``[count, 2]`` float64 event points.
    weights : torch.Tensor
        ``[count]`` float64 per-event ``sin(angle)^2`` weights.
    denominator : float
        Kernel support scale ``(6 * intrinsic_unit)**2``.
    count : int
        Number of events.

    Returns
    -------
    torch.Tensor
        ``[count]`` float64 accumulated densities.
    """

    column_block = max(1, _U07_DENSITY_BLOCK_ELEMENTS // count)
    accumulator = torch.zeros(count, dtype=torch.float64)
    with torch.no_grad():
        for column_start in range(0, count, column_block):
            column_end = min(count, column_start + column_block)
            difference = stacked[:, None, :] - stacked[None, column_start:column_end, :]
            squared = (difference**2).sum(dim=2)
            normalized = squared / denominator
            terms = (
                torch.clamp(1.0 - normalized, min=0.0) ** 2 * weights[None, column_start:column_end]
            )
            diagonal = torch.arange(column_start, column_end)
            terms[diagonal, diagonal - column_start] = 0.0
            for column_offset in range(column_end - column_start):
                accumulator += terms[:, column_offset]
    return accumulator


def _event_density_accumulator_grid(
    stacked: torch.Tensor,
    weights: torch.Tensor,
    denominator: float,
    count: int,
) -> Optional[torch.Tensor]:
    """Accumulate densities skipping provably ``+0.0`` far-pair terms.

    The kernel is compactly supported: ``term(i, j)`` is EXACTLY ``+0.0``
    whenever ``squared >= denominator`` (``clamp(1 - normalized, min=0)``
    collapses to positive zero and ``+0.0 * w`` stays ``+0.0`` for every
    finite ``w >= +0.0``, which the finiteness guard below ensures). Every
    accumulator starts at ``+0.0`` and only ever adds non-negative terms, so
    it is never ``-0.0`` and eliding a ``+0.0`` add is bit-inert.

    Events are binned into a uniform grid of cell size
    ``sqrt(denominator) * (1 + _U07_DENSITY_GRID_CELL_MARGIN)``. A nonzero
    term forces ``fl(dx^2) <= squared * (1 + eps)^2 < denominator *
    (1 + eps)^2``, i.e. ``|dx| <= sqrt(denominator) * (1 + 3eps)``, strictly
    below one cell (the ``1e-6`` margin over-covers the accumulated float
    error by ~9 orders; the magnitude guard bounds the division rounding so
    scaled coordinates within one cell differ by < 1 before ``floor``). So
    all nonzero terms of event ``i`` lie inside its 3x3 cell neighborhood,
    and the ascending-index union ``J_C`` (shared by every event of cell
    ``C``) is a superset of them. Running the SAME column-block accumulation
    as the dense path over the ``[|C|, |J_C|]`` submatrix therefore performs,
    per event, the identical single-IEEE-op term chain on identical operands
    and the identical ascending-``j`` add sequence, minus only ``+0.0``
    adds -- including the historical zeroed ``j == i`` slot, which is zeroed
    here the same way.

    Parameters
    ----------
    stacked : torch.Tensor
        ``[count, 2]`` float64 event points.
    weights : torch.Tensor
        ``[count]`` float64 per-event ``sin(angle)^2`` weights.
    denominator : float
        Kernel support scale ``(6 * intrinsic_unit)**2``.
    count : int
        Number of events.

    Returns
    -------
    torch.Tensor or None
        ``[count]`` float64 accumulated densities, or ``None`` when a guard
        fails (small inputs, non-finite geometry, degenerate kernel scale, or
        oversized cell coordinates) and the dense path must run instead.
    """

    if count < _U07_DENSITY_GRID_MIN_EVENTS:
        return None
    if not (math.isfinite(denominator) and denominator > 0.0):
        return None
    if not bool(torch.isfinite(stacked).all()) or not bool(torch.isfinite(weights).all()):
        return None
    cell = math.sqrt(denominator) * (1.0 + _U07_DENSITY_GRID_CELL_MARGIN)
    if not (math.isfinite(cell) and cell > 0.0):
        return None
    scaled = stacked.numpy() / cell
    if float(np.abs(scaled).max()) >= _U07_DENSITY_GRID_MAX_CELL_MAGNITUDE:
        return None
    cell_coordinates = np.floor(scaled).astype(np.int64)
    cells, inverse, cell_counts = np.unique(
        cell_coordinates, axis=0, return_inverse=True, return_counts=True
    )
    # Stable argsort keeps original event order within a cell, so every
    # member list below is ascending in global event index.
    member_order = np.argsort(inverse.ravel(), kind="stable")
    group_ends = np.cumsum(cell_counts)
    group_starts = group_ends - cell_counts
    group_of_cell = {(int(x), int(y)): g for g, (x, y) in enumerate(cells)}
    accumulator = torch.zeros(count, dtype=torch.float64)
    with torch.no_grad():
        for group, (cell_x, cell_y) in enumerate(cells):
            row_indices = member_order[group_starts[group] : group_ends[group]]
            neighbor_members = [
                member_order[group_starts[g] : group_ends[g]]
                for g in (
                    group_of_cell.get((cell_x + dx, cell_y + dy))
                    for dx in (-1, 0, 1)
                    for dy in (-1, 0, 1)
                )
                if g is not None
            ]
            column_indices = np.sort(np.concatenate(neighbor_members))
            rows_t = torch.from_numpy(row_indices)
            row_points = stacked[rows_t]
            column_points = stacked[torch.from_numpy(column_indices)]
            column_weights = weights[torch.from_numpy(column_indices)]
            # Own cell is always inside the 3x3 union, so each row's j == i
            # column exists exactly once in the ascending column list.
            diagonal_columns = np.searchsorted(column_indices, row_indices)
            column_block = max(1, _U07_DENSITY_BLOCK_ELEMENTS // max(1, len(row_indices)))
            sub_accumulator = torch.zeros(len(row_indices), dtype=torch.float64)
            for column_start in range(0, len(column_indices), column_block):
                column_end = min(len(column_indices), column_start + column_block)
                difference = (
                    row_points[:, None, :] - column_points[None, column_start:column_end, :]
                )
                squared = (difference**2).sum(dim=2)
                normalized = squared / denominator
                terms = (
                    torch.clamp(1.0 - normalized, min=0.0) ** 2
                    * column_weights[None, column_start:column_end]
                )
                in_block = np.nonzero(
                    (diagonal_columns >= column_start) & (diagonal_columns < column_end)
                )[0]
                if in_block.size:
                    terms[
                        torch.from_numpy(in_block),
                        torch.from_numpy(diagonal_columns[in_block] - column_start),
                    ] = 0.0
                for column_offset in range(column_end - column_start):
                    sub_accumulator += terms[:, column_offset]
            accumulator[rows_t] = sub_accumulator
    return accumulator


def _crossing_events_vectorized(scene: Scene, gamma: float) -> List[_CrossingEvent]:
    """Compute U07 events after a vectorized, decision-only crossing sweep.

    Parameters
    ----------
    scene : Scene
        Validated route scene.
    gamma : float
        Positive fitted crossing-severity scale.

    Returns
    -------
    list[_CrossingEvent]
        Events with score-visible values rebuilt by the scalar exact formulas.
    """

    segments = route_segments(scene)
    provisional: List[Tuple[int, int, torch.Tensor, Scalar, Scalar]] = []
    for left_index, right_index in _crossing_candidate_pairs_vectorized(segments):
        edge_a, _, start_a, end_a = segments[left_index]
        edge_b, _, start_b, end_b = segments[right_index]
        event = _segment_event_point(start_a, end_a, start_b, end_b)
        if event is None:
            # This guard is deliberately retained: the vectorized sweep is a
            # decision accelerator, while the shipped scalar predicate remains
            # the final authority for every published event.
            continue
        point, angle = event
        terminals_a = scene.graph.edges[edge_a]
        terminals_b = scene.graph.edges[edge_b]
        terminal_points = scene.positions[list((*terminals_a, *terminals_b))]
        proximity = keep(torch.min(torch.linalg.vector_norm(terminal_points - point, dim=1)))
        if as_float(proximity) == 0.0:
            proximity = 0.0
        provisional.append((min(edge_a, edge_b), max(edge_a, edge_b), point, angle, proximity))

    multiplicities = Counter((left, right) for left, right, _, _, _ in provisional)
    # The traced path keeps the historical per-pair loop (density gradients
    # flow through it); the exact path accumulates the same terms in the same
    # order from float64 tensor blocks -- see _event_densities_exact.
    densities = (
        _event_densities_exact(provisional, scene.intrinsic_unit)
        if provisional and not tracing_active()
        else None
    )
    results: List[_CrossingEvent] = []
    for event_index, (edge_a, edge_b, point, angle, proximity) in enumerate(provisional):
        sine_squared = _p_sin(angle) ** 2
        if densities is not None:
            density: Scalar = densities[event_index]
        else:
            density = 0.0
            for other_index, (_, _, other_point, other_angle, _) in enumerate(provisional):
                if event_index == other_index:
                    continue
                normalized_squared = (
                    keep(torch.sum((point - other_point) ** 2)) / (6.0 * scene.intrinsic_unit) ** 2
                )
                density += _p_sin(other_angle) ** 2 * p_max(0.0, 1.0 - normalized_squared) ** 2
        multiplicity = multiplicities[(edge_a, edge_b)]
        angle_term = _p_cos(angle) ** 2
        proximity_ratio = proximity / scene.intrinsic_unit
        proximity_term = p_max(0.0, 1.0 - proximity_ratio**2 / 16.0) ** 2
        repeat_term = (multiplicity - 1.0) / multiplicity
        density_term = density / (density + 3.0)
        severity = gamma * (
            0.50 * angle_term
            + sine_squared * (0.20 * proximity_term + 0.15 * density_term)
            + 0.15 * repeat_term
        )
        results.append(
            _CrossingEvent(
                edge_a,
                edge_b,
                point,
                angle,
                proximity_ratio,
                multiplicity,
                density,
                severity,
            )
        )
    return results


def U07(
    scene: Scene,
    gamma: float = _U07_WORKED_EXAMPLE_GAMMA,
    lambda_T: float = _U07_WORKED_EXAMPLE_LAMBDA_T,
) -> FacetResult:
    """NORMATIVE CONTRACT: Crossing slot. Frozen SHA-256: 63526aa6824dfa5180234ba97086725621e8b9e8aa3c40713c85a7c4a86caf2b.

    Parameters
    ----------
    scene : Scene
        Validated canonical scene.
    gamma : float
        Fitted severity scale in the contract range ``(0, 3]``. The default is
        the phase-4 worked-example value and is not a P5 fitted selection.
    lambda_T : float
        Fitted excess-severity weight in ``[0, 1]``. The default is the phase-4
        worked-example value and is not a P5 fitted selection.

    Returns
    -------
    FacetResult
        Crossing defect and its contract rows.

    Raises
    ------
    ValueError
        If a fitted parameter lies outside its frozen contract range.
    """

    if not math.isfinite(gamma) or not 0.0 < gamma <= 3.0:
        raise ValueError("U07 gamma must be finite and lie in (0, 3]")
    if not math.isfinite(lambda_T) or not 0.0 <= lambda_T <= 1.0:
        raise ValueError("U07 lambda_T must be finite and lie in [0, 1]")

    events = (
        _crossing_events_vectorized(scene, gamma)
        if VECTORIZED_EXACT_SCORERS
        else _crossing_events(scene, gamma)
    )
    eligible = 0
    for left in range(scene.edge_count):
        for right in range(left + 1, scene.edge_count):
            if not set(scene.graph.edges[left]) & set(scene.graph.edges[right]):
                eligible += 1
    base_raw = p_sum([1.0 + event.severity for event in events])
    threshold = 0.5 * gamma
    tail_raw = p_sum([p_max(0.0, event.severity - threshold) for event in events])
    opportunity = eligible + 1
    normalized = (base_raw + lambda_T * tail_raw) / opportunity
    defect = normalized / (normalized + 0.25) if as_float(normalized) > 0.0 else 0.0
    base_x = base_raw / opportunity
    tail_x = lambda_T * tail_raw / opportunity
    base = base_x / (base_x + 0.25) if as_float(base_x) > 0.0 else 0.0
    tail = tail_x / (tail_x + 0.25) if as_float(tail_x) > 0.0 else 0.0
    return value_result(
        defect,
        {"U7.base": base, "U7.tail": tail},
        {
            "crossing_count": len(events),
            "eligible_pairs": eligible,
            "opportunity_guarded": opportunity,
            "base_raw": _raw(base_raw),
            "tail_raw": _raw(tail_raw),
            "gamma": gamma,
            "lambda_T": lambda_T,
            "events": tuple(
                {
                    "edge_pair": (event.edge_a, event.edge_b),
                    "point": event.point.tolist(),
                    "angle": _raw(event.angle),
                    "proximity": _raw(event.proximity),
                    "multiplicity": event.pair_multiplicity,
                    "density": _raw(event.density),
                    "severity": _raw(event.severity),
                }
                for event in events
            ),
        },
    )


def U08(scene: Scene) -> FacetResult:
    """Angular resolution at nodes. Frozen SHA-256: bcc8037fd0f620ce8e43fa7ba68acd711601e12355dd48a2741469dc997a4ebf."""

    directions = _incident_secants(scene)
    input_degrees = [0] * scene.node_count
    for source, target in scene.graph.edges:
        if source == target:
            input_degrees[source] += 2
        else:
            input_degrees[source] += 1
            input_degrees[target] += 1
    defects: List[Scalar] = []
    for node, degree in enumerate(input_degrees):
        if degree < 3:
            continue
        records = sorted(directions[node], key=lambda item: (as_float(item[0]), item[1]))
        effective_count = p_sum([confidence for _, _, confidence in records])
        fade = _smooth_fade(effective_count - 2.0)
        if as_float(fade) == 0.0:
            defects.append(0.0)
            continue
        fair_share = 2.0 * math.pi / effective_count
        weighted_exponentials: Scalar = 0.0
        total_weight: Scalar = 0.0
        count = len(records)
        for left in range(count):
            for step in range(1, count):
                right = (left + step) % count
                between = [(left + offset) % count for offset in range(1, step)]
                weight = records[left][2] * records[right][2]
                for middle in between:
                    weight *= 1.0 - records[middle][2]
                if as_float(weight) == 0.0:
                    continue
                delta = _p_mod(records[right][0] - records[left][0], 2.0 * math.pi)
                pair_defect = p_max(0.0, 1.0 - delta / fair_share)
                if as_float(pair_defect) <= _ANGULAR_ZERO_ENVELOPE:
                    pair_defect = 0.0
                weighted_exponentials += weight * p_exp(pair_defect / 0.1)
                total_weight += weight
        if as_float(total_weight) == 0.0:
            defects.append(0.0)
            continue
        node_defect = 0.1 * p_log(weighted_exponentials / total_weight)
        # The log-sum-exp mean is analytically in [0, 1] for pair defects in
        # [0, 1]; the normalized mean can carry one ULP of dust either side.
        defects.append(snap_unit(fade * node_defect))
    if not defects:
        return na_result("no_high_degree_nodes")
    defect = global_blend(defects)
    return value_result(defect, {"U08.headline": defect}, {"node_count": len(defects)})


def _point_at_arc_fraction(points: torch.Tensor, fraction: float) -> torch.Tensor:
    """Interpolate a polyline at one fraction of total arc length.

    Parameters
    ----------
    points : torch.Tensor
        Polyline vertices with shape ``[P, 2]``.
    fraction : float
        Arc-length fraction in ``[0, 1]``.

    Returns
    -------
    torch.Tensor
        Interpolated point with shape ``[2]``.
    """

    lengths = torch.linalg.vector_norm(points[1:] - points[:-1], dim=1)
    total = keep(torch.sum(lengths))
    if as_float(total) == 0.0:
        return points[0]
    target = min(1.0, max(0.0, fraction)) * total
    cumulative: Scalar = 0.0
    for index, length_tensor in enumerate(lengths):
        length = keep(length_tensor)
        if as_float(cumulative + length) >= as_float(target):
            local = (target - cumulative) / length if as_float(length) > 0.0 else 0.0
            return points[index] + local * (points[index + 1] - points[index])
        cumulative += length
    return points[-1]


def _incident_secants(scene: Scene) -> DefaultDict[int, List[Tuple[Scalar, int, Scalar]]]:
    """Extract U08 departure angles and tangent confidences at every endpoint.

    Parameters
    ----------
    scene : Scene
        Validated graph scene.

    Returns
    -------
    defaultdict[int, list[tuple[float or torch.Tensor, int, float or torch.Tensor]]]
        Node to ``(angle, canonical endpoint key, confidence)`` records.
    """

    records: DefaultDict[int, List[Tuple[Scalar, int, Scalar]]] = defaultdict(list)
    for route in resolved_routes(scene):
        source, target = scene.graph.edges[route.edge_index]
        lengths = torch.linalg.vector_norm(route.points[1:] - route.points[:-1], dim=1)
        total = keep(torch.sum(lengths))
        confidence = _smooth_fade(total / (0.1 * scene.intrinsic_unit))
        if as_float(confidence) == 0.0:
            if source == target:
                records[source].extend(
                    ((0.0, 2 * route.edge_index, 0.0), (0.0, 2 * route.edge_index + 1, 0.0))
                )
            else:
                records[source].append((0.0, 2 * route.edge_index, 0.0))
                records[target].append((0.0, 2 * route.edge_index + 1, 0.0))
            continue
        source_vector = _point_at_arc_fraction(route.points, 0.10) - scene.positions[source]
        target_vector = _point_at_arc_fraction(route.points, 0.90) - scene.positions[target]
        source_angle = _p_mod(
            _p_atan2(keep(source_vector[1]), keep(source_vector[0])), 2.0 * math.pi
        )
        target_angle = _p_mod(
            _p_atan2(keep(target_vector[1]), keep(target_vector[0])), 2.0 * math.pi
        )
        records[source].append((source_angle, 2 * route.edge_index, confidence))
        records[target].append((target_angle, 2 * route.edge_index + 1, confidence))
    return records


def _point_box_signed(point: torch.Tensor, box: BoxGeometry) -> float:
    """Return signed point clearance from an axis-aligned box.

    Parameters
    ----------
    point : torch.Tensor
        Point with shape ``[2]``.
    box : BoxGeometry
        Derived obstacle box.

    Returns
    -------
    float
        Positive outside clearance and negative penetration depth.
    """

    excess = torch.abs(point - box.center) - box.half_extents
    outside = float(torch.linalg.vector_norm(torch.clamp(excess, min=0.0)))
    inside = min(max(float(excess[0]), float(excess[1])), 0.0)
    return outside + inside


def _unit_interval_root(constant: float, slope: float) -> List[float]:
    """Return an affine root when it lies strictly inside the unit interval.

    Parameters
    ----------
    constant, slope : float
        Coefficients of ``constant + slope*t``.

    Returns
    -------
    list[float]
        Empty or one-element root list.
    """

    if slope == 0.0:
        return []
    root = -constant / slope
    return [root] if 0.0 < root < 1.0 else []


def _unit_interval_quadratic_roots(a: float, b: float, c: float) -> List[float]:
    """Return real quadratic roots strictly inside the unit interval.

    Parameters
    ----------
    a, b, c : float
        Coefficients of ``a*t^2 + b*t + c``.

    Returns
    -------
    list[float]
        Canonically sorted in-range roots.
    """

    if a == 0.0:
        return _unit_interval_root(c, b)
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return []
    square_root = math.sqrt(max(0.0, discriminant))
    roots = [(-b - square_root) / (2.0 * a), (-b + square_root) / (2.0 * a)]
    return sorted({root for root in roots if 0.0 < root < 1.0})


def _sqrt_quadratic_primitive(a: Scalar, b: Scalar, c: Scalar, value: float) -> Scalar:
    """Evaluate U10 section 5a's primitive of ``sqrt(a*t^2+b*t+c)``.

    Parameters
    ----------
    a, b, c : float or torch.Tensor
        Quadratic coefficients with a globally nonnegative quadratic.
    value : float
        Evaluation coordinate.

    Returns
    -------
    float or torch.Tensor
        Closed-form primitive value; a live tensor inside a trace.
    """

    quadratic = p_max(0.0, a * value * value + b * value + c)
    if as_float(a) == 0.0:
        return p_sqrt(p_max(0.0, c)) * value
    delta = p_max(0.0, 4.0 * a * c - b * b)
    if as_float(quadratic) == 0.0:
        # sqrt has an infinite slope at exactly zero; the historical float
        # value there is exactly 0 (a breakpoint sitting on the tangency).
        root: Scalar = 0.0
    else:
        root = p_sqrt(quadratic)
    if as_float(delta) == 0.0:
        center = -b / (2.0 * a)
        sign = -1.0 if value < as_float(center) else 1.0
        return p_sqrt(a) * sign * (value - center) ** 2 / 2.0
    return (2.0 * a * value + b) * root / (4.0 * a) + delta / (8.0 * a**1.5) * _p_asinh(
        (2.0 * a * value + b) / p_sqrt(delta)
    )


def _segment_box_deficit_integral(
    start: torch.Tensor,
    end: torch.Tensor,
    box: BoxGeometry,
    stroke_half_width: float,
    intrinsic_unit: float,
) -> Tuple[Scalar, bool]:
    """Integrate U10's exact clearance-deficit kernel on one segment.

    Parameters
    ----------
    start, end : torch.Tensor
        Flattened route-segment endpoints with shape ``[2]``.
    box : BoxGeometry
        Axis-aligned specialization of the contract OBB obstacle.
    stroke_half_width : float
        Nonnegative route stroke half-width.
    intrinsic_unit : float
        Positive scene intrinsic unit.

    Returns
    -------
    tuple[float or torch.Tensor, bool]
        Dimensionless exact integral and whether the segment penetrates the box.
        The piece partition is decided on detached values; the integrand is
        continuous across every internal breakpoint, so boundary-motion terms
        cancel and the detached-partition gradient is the a.e.-exact one.
    """

    local_start = start - box.center
    direction = end - start
    length = keep(torch.linalg.vector_norm(direction))
    if as_float(length) == 0.0:
        return 0.0, False
    half = box.half_extents
    clearance_band = 0.5 * intrinsic_unit
    radius = float(torch.linalg.vector_norm(half))
    band_radius = clearance_band + stroke_half_width
    clamp_radius = stroke_half_width - radius
    breaks = [0.0, 1.0]
    for axis in range(2):
        origin = float(local_start[axis])
        delta = float(direction[axis])
        breaks.extend(_unit_interval_root(origin, delta))
        for sign in (-1.0, 1.0):
            breaks.extend(_unit_interval_root(origin - sign * float(half[axis]), delta))
            breaks.extend(
                _unit_interval_root(
                    sign * origin - float(half[axis]) - band_radius,
                    sign * delta,
                )
            )
            breaks.extend(
                _unit_interval_root(
                    sign * origin - float(half[axis]) - clamp_radius,
                    sign * delta,
                )
            )
    for sign_x in (-1.0, 1.0):
        for sign_y in (-1.0, 1.0):
            constant_x = sign_x * float(local_start[0]) - float(half[0])
            constant_y = sign_y * float(local_start[1]) - float(half[1])
            slope_x = sign_x * float(direction[0])
            slope_y = sign_y * float(direction[1])
            breaks.extend(_unit_interval_root(constant_x - constant_y, slope_x - slope_y))
            quadratic_a = slope_x * slope_x + slope_y * slope_y
            quadratic_b = 2.0 * (constant_x * slope_x + constant_y * slope_y)
            quadratic_c = constant_x * constant_x + constant_y * constant_y
            breaks.extend(
                _unit_interval_quadratic_roots(
                    quadratic_a,
                    quadratic_b,
                    quadratic_c - band_radius * band_radius,
                )
            )
            breaks.extend(
                _unit_interval_quadratic_roots(
                    quadratic_a,
                    quadratic_b,
                    quadratic_c - clamp_radius * clamp_radius,
                )
            )
    ordered = sorted(set(breaks))
    total: Scalar = 0.0
    penetrates = False
    maximum = (1.0 + radius / clearance_band) ** 2

    def signed_distance(parameter: float) -> Scalar:
        """Return exact signed point-to-box distance in the box frame.

        Parameters
        ----------
        parameter : float
            Segment parameter in ``[0, 1]``.

        Returns
        -------
        float or torch.Tensor
            Positive exterior distance or negative interior depth.
        """

        point = local_start + parameter * direction
        excess = torch.abs(point) - half
        if bool((excess > 0.0).any()):
            return keep(torch.linalg.vector_norm(torch.clamp(excess, min=0.0)))
        return p_max(keep(excess[0]), keep(excess[1]))

    for lower, upper in zip(ordered[:-1], ordered[1:]):
        if upper <= lower:
            continue
        midpoint = (lower + upper) / 2.0
        point = local_start + midpoint * direction
        excess = torch.abs(point) - half
        distance = signed_distance(midpoint)
        clearance = distance - stroke_half_width
        penetrates = penetrates or as_float(distance) < 0.0
        if as_float(clearance) >= clearance_band:
            continue
        if as_float(clearance) <= -radius:
            total += maximum * (upper - lower)
            continue
        positive_axes = excess > 0.0
        if bool(positive_axes.all()):
            signs = torch.where(point >= 0.0, 1.0, -1.0)
            constants = signs * local_start - half
            slopes = signs * direction
            a = keep(torch.dot(slopes, slopes))
            b = 2.0 * keep(torch.dot(constants, slopes))
            c = keep(torch.dot(constants, constants))
            polynomial = (
                band_radius * band_radius * (upper - lower)
                + a * (upper**3 - lower**3) / 3.0
                + b * (upper**2 - lower**2) / 2.0
                + c * (upper - lower)
            )
            square_root = _sqrt_quadratic_primitive(a, b, c, upper) - (
                _sqrt_quadratic_primitive(a, b, c, lower)
            )
            total += (polynomial - 2.0 * band_radius * square_root) / clearance_band**2
            continue
        distance_lower = signed_distance(lower)
        distance_upper = signed_distance(upper)
        slope = (distance_upper - distance_lower) / (upper - lower)
        intercept = distance_lower - slope * lower - stroke_half_width
        if as_float(slope) == 0.0:
            total += ((clearance_band - intercept) / clearance_band) ** 2 * (upper - lower)
        else:
            primitive_upper = -((clearance_band - slope * upper - intercept) ** 3) / (
                3.0 * slope * clearance_band**2
            )
            primitive_lower = -((clearance_band - slope * lower - intercept) ** 3) / (
                3.0 * slope * clearance_band**2
            )
            total += primitive_upper - primitive_lower
    return p_max(0.0, length * total / intrinsic_unit), penetrates


def _raw_u10_blend(values: List[Scalar], opportunity: int) -> Scalar:
    """Apply U10's raw-component aggregation and common saturation map.

    Parameters
    ----------
    values : list[float or torch.Tensor]
        Full input-only population of nonnegative per-pair integrals.
    opportunity : int
        Analytic scored-pair count, equal to ``len(values)``.

    Returns
    -------
    float or torch.Tensor
        Contract U10 defect in ``[0, 1)``; a live tensor inside a trace.
    """

    ordered = sorted(values, key=as_float)
    mean = p_sum(ordered) / opportunity
    tail_mass = 0.10 * opportunity
    remaining = tail_mass
    tail_sum: Scalar = 0.0
    for value in reversed(ordered):
        mass = min(1.0, remaining)
        tail_sum += mass * value
        remaining -= mass
        if remaining <= 0.0:
            break
    cvar = tail_sum / tail_mass
    maximum = ordered[-1]
    smooth_maximum = maximum + 0.05 * p_log(
        p_sum([p_exp((value - maximum) / 0.05) for value in ordered]) / opportunity
    )

    def saturate(value: Scalar) -> Scalar:
        """Map one nonnegative component through U10's frozen x0 curve.

        Parameters
        ----------
        value : float or torch.Tensor
            Nonnegative raw component.

        Returns
        -------
        float or torch.Tensor
            Saturated component.
        """

        return value / (value + 0.01) if as_float(value) > 0.0 else 0.0

    return 0.65 * saturate(mean) + 0.25 * saturate(cvar) + 0.10 * saturate(smooth_maximum)


def U10(scene: Scene) -> FacetResult:
    """Edge-node occlusion / false attachment. Frozen SHA-256: 91f7929473c6d2f9fb9afc0b0fded75bdc899fe521ca942794b6042739aaf850."""

    if scene.edge_count < 1 or scene.node_count < 3:
        return na_result("too_few_objects")
    burdens: List[Scalar] = []
    penetration_count = 0
    for route in resolved_routes(scene):
        incident = set(scene.graph.edges[route.edge_index])
        obstacles = [box for box in scene.node_boxes if box.owner not in incident]
        obstacles.extend(box for box in scene.node_label_boxes if box.owner not in incident)
        width = (
            scene.style.edge_stroke_widths[route.edge_index]
            if scene.style.edge_stroke_widths
            else scene.style.route_stroke_width * scene.style.coordinate_scale
        )
        for box in obstacles:
            burden: Scalar = 0.0
            penetrates = False
            for start, end in zip(route.points[:-1], route.points[1:]):
                segment_burden, segment_penetrates = _segment_box_deficit_integral(
                    start,
                    end,
                    box,
                    width / 2.0,
                    scene.intrinsic_unit,
                )
                burden += segment_burden
                penetrates = penetrates or segment_penetrates
            burdens.append(burden)
            penetration_count += int(penetrates)
    if not burdens:
        return na_result("too_few_objects")
    defect = _raw_u10_blend(burdens, len(burdens))
    return value_result(
        defect,
        {"U10.headline": defect},
        {
            "pair_count": len(burdens),
            "penetration_count": penetration_count,
            "pair_integrals": tuple(_raw(burden) for burden in burdens),
            "sum_D_eb": _raw(p_sum(burdens)),
        },
    )


def _route_bends(points: torch.Tensor) -> List[float]:
    """Return normalized bend burdens for one route.

    Parameters
    ----------
    points : torch.Tensor
        Route vertices ``[P, 2]``.

    Returns
    -------
    list[float]
        Zero for straight continuation, one for reversal.
    """

    result = []
    for index in range(1, points.shape[0] - 1):
        incoming = points[index] - points[index - 1]
        outgoing = points[index + 1] - points[index]
        denominator = torch.linalg.vector_norm(incoming) * torch.linalg.vector_norm(outgoing)
        if float(denominator) == 0.0:
            result.append(1.0)
            continue
        cosine = min(1.0, max(-1.0, float(torch.dot(incoming, outgoing) / denominator)))
        result.append(math.acos(cosine) / math.pi)
    return result


def U11(scene: Scene) -> FacetResult:
    """Routed-edge quality: NORMATIVE CONTRACT (A1). Frozen SHA-256: 411dab9b787a8465d2f2591f3048bb6d11d1e57075071532d37c647e74e71dfd."""

    self_crossings = 0
    row_one: List[Scalar] = []
    row_two: List[Scalar] = []
    row_three: List[Scalar] = []
    row_four: List[Scalar] = []
    terminal_records: DefaultDict[int, List[Tuple[torch.Tensor, Optional[torch.Tensor]]]] = (
        defaultdict(list)
    )
    raw_edges: List[Dict[str, float]] = []
    routes = resolved_routes(scene)
    ranks = resolved_ranks(scene)
    for route in routes:
        points = route.points
        vectors = points[1:] - points[:-1]
        lengths = torch.linalg.vector_norm(vectors, dim=1)
        arc = keep(torch.sum(lengths))
        source, target = scene.graph.edges[route.edge_index]
        source_tangent = _first_nonzero_tangent(points, False)
        target_tangent = _first_nonzero_tangent(points, True)
        terminal_records[source].append((points[0], source_tangent))
        terminal_records[target].append((points[-1], target_tangent))
        if as_float(arc) == 0.0:
            continue
        chord_vector = points[-1] - points[0]
        chord = keep(torch.linalg.vector_norm(chord_vector))
        chord_direction = (
            chord_vector / chord if as_float(chord) > 0.0 else torch.zeros(2, dtype=torch.float64)
        )
        event_severity: Scalar = 0.0
        for left in range(points.shape[0] - 1):
            for right in range(left + 2, points.shape[0] - 1):
                if proper_intersection(
                    points[left], points[left + 1], points[right], points[right + 1]
                ):
                    self_crossings += 1
                    angle = _segment_angle(vectors[left], vectors[right])
                    event_severity += 1.0 + _p_cos(angle) ** 2
        self_defect = 1.0 - p_exp(-math.log(2.0) * event_severity)
        if tracing_active():
            # torch.cdist's backward is NaN at the (always-present) zero
            # diagonal; max over squared distances then sqrt is the same value
            # with a well-defined gradient (arc > 0 guarantees a positive max).
            pair_differences = points[:, None, :] - points[None, :, :]
            route_diameter: Scalar = torch.sqrt((pair_differences**2).sum(dim=-1).max())
        else:
            route_diameter = keep(torch.max(torch.cdist(points, points)))
        backtracking = p_sum(
            [
                p_max(0.0, -keep(torch.dot(chord_direction, vector / length))) * keep(length)
                for vector, length in zip(vectors, lengths)
                if float(length) > 0.0
            ]
        )
        backtracking_defect = 1.0 - p_exp(
            -backtracking / p_max(p_max(chord, route_diameter), 1e-300)
        )
        row_one.append(1.0 - (1.0 - self_defect) * (1.0 - backtracking_defect))

        turns = _signed_route_turns(points)
        total_turn = p_sum([p_abs(turn) for turn in turns])
        wiggle = total_turn - p_abs(p_sum(turns))
        baseline_length, baseline_turn = _route_baseline(
            scene, route, vectorized=VECTORIZED_EXACT_SCORERS
        )
        if scene.graph.edge_styles is not None and source != target:
            style = scene.graph.edge_styles[route.edge_index]
            excess_turn = _zero_hinge(total_turn - baseline_turn, 0.05)
            if style == "straight":
                bend_defect = 1.0 - p_exp(-total_turn / (math.pi / 2.0))
            elif style == "orthogonal":
                off_axis = (
                    p_sum(
                        [
                            keep(length)
                            * (1.0 - _p_cos(4.0 * _nearest_axis_deviation(vector)))
                            / 2.0
                            for vector, length in zip(vectors, lengths)
                            if float(length) > 0.0
                        ]
                    )
                    / arc
                )
                bend_defect = 0.5 * (1.0 - p_exp(-off_axis / 0.15)) + 0.5 * (
                    1.0 - p_exp(-excess_turn / (math.pi / 2.0))
                )
            else:
                turn_weight, wiggle_weight = (0.7, 0.3) if style == "polyline" else (0.6, 0.4)
                bend_defect = turn_weight * (
                    1.0 - p_exp(-excess_turn / (math.pi / 2.0))
                ) + wiggle_weight * (1.0 - p_exp(-wiggle / (math.pi / 2.0)))
            row_two.append(bend_defect)
        if source != target:
            log_ratio = p_log(arc / baseline_length)
            row_three.append(1.0 - p_exp(-_zero_hinge(log_ratio, 0.05) / math.log(2.0)))
        feedback = scene.graph.feedback is not None and scene.graph.feedback[route.edge_index]
        same_rank = ranks is not None and ranks[source] == ranks[target]
        if (
            scene.graph.directed
            and scene.graph.flow_axis is not None
            and source != target
            and not feedback
            and not same_rank
        ):
            axis = torch.tensor(scene.graph.flow_axis, dtype=torch.float64)
            counterflow = p_sum(
                [
                    p_max(0.0, -keep(torch.dot(axis, vector / length))) * keep(length)
                    for vector, length in zip(vectors, lengths)
                    if float(length) > 0.0
                ]
            )
            row_four.append(1.0 - p_exp(-(counterflow / arc) / 0.25))
        raw_edges.append(
            {
                "edge": float(route.edge_index),
                "arc_length": _raw(arc),
                "baseline_length": _raw(baseline_length),
                "total_turn": _raw(total_turn),
                "baseline_turn": _raw(baseline_turn),
                "backtracking": _raw(backtracking),
                "self_event_severity": _raw(event_severity),
            }
        )
    values: Dict[str, Scalar] = {}
    if row_one:
        values["U11.i"] = global_blend(row_one)
    if row_two:
        values["U11.ii"] = global_blend(row_two)
    if row_three:
        values["U11.iii"] = global_blend(row_three)
    if row_four:
        values["U11.iv"] = global_blend(row_four)
    if not scene.graph.ports:
        node_defects: List[Scalar] = []
        node_weights: List[float] = []
        for records in terminal_records.values():
            if len(records) < 2:
                continue
            confusability: List[Scalar] = []
            for left_index, left in enumerate(records):
                for right in records[left_index + 1 :]:
                    gap_norm = keep(torch.linalg.vector_norm(left[0] - right[0]))
                    if as_float(gap_norm) == 0.0:
                        # vector_norm has an undefined (NaN) gradient at the
                        # exact-coincidence point, but the Gaussian kernel
                        # exp(-(|d|/s)^2) is smooth there with slope exactly 0:
                        # the detached constant IS the exact gradient.
                        gap_norm = 0.0
                    gap_factor = p_exp(-((gap_norm / (scene.intrinsic_unit / 4.0)) ** 2))
                    if left[1] is None or right[1] is None:
                        # The gap factor of the closed form stays well defined
                        # when a zero-arc route has no initial tangent; only the
                        # angle factor is undefined and takes its supremum, so a
                        # coincident tangent-less pair is the coincidence limit
                        # while a distant one still earns its separation.
                        confusability.append(gap_factor)
                        continue
                    confusability.append(
                        gap_factor
                        * p_exp(-((_oriented_angle(left[1], right[1]) / math.radians(15.0)) ** 2))
                    )
            node_defects.append(snap_unit(p_sum(confusability) / len(confusability)))
            node_weights.append(float(len(records)))
        if node_defects:
            values["U11.v"] = global_blend(node_defects, node_weights)
    published: Dict[str, float] = {}
    for key, item in values.items():
        if isinstance(item, torch.Tensor):
            # Mirror of scene.value_result's traced seam: U11 builds its
            # FacetResult directly (headline None), so the live subterm tensors
            # are recorded here and published as the same detached floats.
            record_subterm(key, item)
            published[key] = float(item.detach().item())
        else:
            published[key] = item
    dropped_subterms = []
    if scene.graph.edge_styles is None:
        dropped_subterms.append("U11.ii:no_declared_style")
    feedback_count = sum(scene.graph.feedback or ())
    return FacetResult(
        ResultState.VALUE,
        None,
        None,
        published,
        {
            "self_intersection_count": self_crossings,
            "edges": tuple(raw_edges),
            "dropped_subterms": tuple(dropped_subterms),
            "feedback_edge_count": feedback_count,
            "feedback_edge_coverage": feedback_count / scene.edge_count
            if scene.edge_count
            else 0.0,
        },
    )


def _signed_route_turns(points: torch.Tensor) -> List[Scalar]:
    """Return signed exterior turns of a flattened polyline.

    Parameters
    ----------
    points : torch.Tensor
        Route vertices with shape ``[P, 2]``.

    Returns
    -------
    list[float or torch.Tensor]
        Signed turns in radians; live tensors inside a trace.
    """

    turns: List[Scalar] = []
    for index in range(1, points.shape[0] - 1):
        incoming = points[index] - points[index - 1]
        outgoing = points[index + 1] - points[index]
        if (
            float(torch.linalg.vector_norm(incoming)) == 0.0
            or float(torch.linalg.vector_norm(outgoing)) == 0.0
        ):
            continue
        cross = keep(incoming[0] * outgoing[1] - incoming[1] * outgoing[0])
        dot = keep(torch.dot(incoming, outgoing))
        turns.append(_p_atan2(cross, dot))
    return turns


def _zero_hinge(value: Scalar, width: float) -> Scalar:
    """Evaluate U11's zero-anchored C1 excess hinge.

    Parameters
    ----------
    value : float or torch.Tensor
        Signed excess.
    width : float
        Positive quadratic transition width.

    Returns
    -------
    float or torch.Tensor
        Zero for nonpositive input and asymptotically linear excess. The
        branch is decided on the detached value; the nonpositive branch is the
        hinge's exact constant-zero arm (zero value and zero slope at onset).
    """

    if as_float(value) <= 0.0:
        return 0.0
    if as_float(value) <= width:
        return value * value / (2.0 * width)
    return value - width / 2.0


def _nearest_axis_deviation(vector: torch.Tensor) -> Scalar:
    """Return acute direction deviation from the nearest page axis.

    Parameters
    ----------
    vector : torch.Tensor
        Nonzero segment vector with shape ``[2]``.

    Returns
    -------
    float or torch.Tensor
        Deviation in ``[0, pi/4]``.
    """

    angle = _p_mod(_p_atan2(keep(vector[1]), keep(vector[0])), math.pi / 2.0)
    return p_min(angle, math.pi / 2.0 - angle)


def _first_nonzero_tangent(points: torch.Tensor, reverse: bool) -> Optional[torch.Tensor]:
    """Return a unit terminal tangent pointing away from its node.

    Parameters
    ----------
    points : torch.Tensor
        Route vertices with shape ``[P, 2]``.
    reverse : bool
        Read from the target terminal when true.

    Returns
    -------
    torch.Tensor or None
        Unit tangent, or none for an all-zero route.
    """

    ordered = torch.flip(points, dims=(0,)) if reverse else points
    for point in ordered[1:]:
        vector = point - ordered[0]
        length = keep(torch.linalg.vector_norm(vector))
        if as_float(length) > 0.0:
            return vector / length
    return None


def _regular_polygon(center: torch.Tensor, radius: float, sides: int) -> torch.Tensor:
    """Build a counter-clockwise regular polygon.

    Parameters
    ----------
    center : torch.Tensor
        Polygon center with shape ``[2]``.
    radius : float
        Positive circumradius.
    sides : int
        Number of polygon sides, at least three.

    Returns
    -------
    torch.Tensor
        Polygon vertices with shape ``[sides, 2]``.
    """

    angles = torch.arange(sides, dtype=torch.float64) * (2.0 * math.pi / sides)
    offsets = radius * torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    return center + offsets


def _box_boundary_segments(box: BoxGeometry) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Return the four canonical boundary segments of one axis-aligned box.

    Parameters
    ----------
    box : BoxGeometry
        Axis-aligned obstacle box.

    Returns
    -------
    list[tuple[torch.Tensor, torch.Tensor]]
        Four boundary segments in counter-clockwise order.
    """

    lower = box.center - box.half_extents
    upper = box.center + box.half_extents
    if tracing_active():
        # Graph-preserving construction: torch.tensor([...]) would detach the
        # live corner coordinates; stacking yields bit-identical values.
        corners = [
            torch.stack((lower[0], lower[1])),
            torch.stack((upper[0], lower[1])),
            torch.stack((upper[0], upper[1])),
            torch.stack((lower[0], upper[1])),
        ]
    else:
        corners = [
            torch.tensor([lower[0], lower[1]], dtype=torch.float64),
            torch.tensor([upper[0], lower[1]], dtype=torch.float64),
            torch.tensor([upper[0], upper[1]], dtype=torch.float64),
            torch.tensor([lower[0], upper[1]], dtype=torch.float64),
        ]
    return list(zip(corners, corners[1:] + corners[:1]))


def _polygon_boundary_segments(polygon: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Return consecutive boundary segments of one polygon.

    Parameters
    ----------
    polygon : torch.Tensor
        Polygon vertices with shape ``[P, 2]``.

    Returns
    -------
    list[tuple[torch.Tensor, torch.Tensor]]
        Closed polygon boundary segments.
    """

    return [
        (polygon[index], polygon[(index + 1) % polygon.shape[0]])
        for index in range(polygon.shape[0])
    ]


def _segment_boundary_parameter(
    start: torch.Tensor,
    end: torch.Tensor,
    boundary_start: torch.Tensor,
    boundary_end: torch.Tensor,
) -> Optional[Tuple[Scalar, torch.Tensor]]:
    """Return one transversal segment-boundary intersection.

    Parameters
    ----------
    start, end : torch.Tensor
        Query segment endpoints with shape ``[2]``.
    boundary_start, boundary_end : torch.Tensor
        Boundary segment endpoints with shape ``[2]``.

    Returns
    -------
    tuple[float or torch.Tensor, torch.Tensor] or None
        Query parameter and intersection point, including endpoints.
    """

    query = end - start
    boundary = boundary_end - boundary_start
    denominator = keep(query[0] * boundary[1] - query[1] * boundary[0])
    if as_float(denominator) == 0.0:
        return None
    offset = boundary_start - start
    query_parameter = keep(offset[0] * boundary[1] - offset[1] * boundary[0]) / denominator
    boundary_parameter = keep(offset[0] * query[1] - offset[1] * query[0]) / denominator
    if 0.0 <= as_float(query_parameter) <= 1.0 and 0.0 <= as_float(boundary_parameter) <= 1.0:
        return query_parameter, start + query_parameter * query
    return None


_FloatPoint = Tuple[float, float]
_FloatSegment = Tuple[float, float, float, float]


@dataclass(frozen=True)
class _ObstacleSnapshot:
    """Detached float geometry of one terminal-cleared U11 obstacle.

    U11's visibility DECISIONS (which segments are blocked, which candidate
    vertices survive) are pure control flow: the exact scorer always read
    them through ``as_float``/``bool`` casts, so no gradient ever flowed
    through them. Evaluating the predicates on plain Python floats performs
    the same IEEE-754 double operations the per-element tensor arithmetic
    performed (multiply, subtract, divide, and compare are each correctly
    rounded in both), so every decision is bit-identical while avoiding a
    per-operation tensor dispatch that made medium scenes take hours, and
    avoiding autograd graph growth on the traced path. Score-visible VALUES
    (chord, the chosen path's vertex coordinates) never come from here.

    Parameters
    ----------
    center : tuple[float, float]
        Box center.
    half_extents : tuple[float, float]
        Box half extents.
    box_boundaries : tuple[tuple[float, float, float, float], ...]
        Four counter-clockwise box boundary segments as (sx, sy, ex, ey).
    polygons : tuple[tuple[tuple[float, float], ...], ...]
        The two terminal-centered regular 16-gons removed from the obstacle.
    """

    center: _FloatPoint
    half_extents: _FloatPoint
    box_boundaries: Tuple[_FloatSegment, ...]
    polygons: Tuple[Tuple[_FloatPoint, ...], ...]


def _polygon_floats(
    terminal_polygons: Sequence[torch.Tensor],
) -> Tuple[Tuple[_FloatPoint, ...], ...]:
    """Read terminal polygon vertices as detached floats.

    Parameters
    ----------
    terminal_polygons : sequence[torch.Tensor]
        Polygonized terminal disks with shape ``[P, 2]`` each.

    Returns
    -------
    tuple[tuple[tuple[float, float], ...], ...]
        Vertex coordinates per polygon.
    """

    return tuple(
        tuple((float(vertex[0]), float(vertex[1])) for vertex in polygon)
        for polygon in terminal_polygons
    )


def _polygon_boundary_floats(
    polygons: Tuple[Tuple[_FloatPoint, ...], ...],
) -> Tuple[_FloatSegment, ...]:
    """Return consecutive float boundary segments of the terminal polygons.

    Parameters
    ----------
    polygons : tuple[tuple[tuple[float, float], ...], ...]
        Terminal polygon vertices shared by every obstacle of one route.

    Returns
    -------
    tuple[tuple[float, float, float, float], ...]
        Closed boundary segments as (sx, sy, ex, ey).
    """

    boundaries: List[_FloatSegment] = []
    for polygon in polygons:
        count = len(polygon)
        boundaries.extend(polygon[index] + polygon[(index + 1) % count] for index in range(count))
    return tuple(boundaries)


def _obstacle_snapshot(
    box: BoxGeometry, polygons: Tuple[Tuple[_FloatPoint, ...], ...]
) -> _ObstacleSnapshot:
    """Build the detached float decision geometry of one obstacle.

    Parameters
    ----------
    box : BoxGeometry
        Uninflated node obstacle.
    polygons : tuple[tuple[tuple[float, float], ...], ...]
        Detached terminal polygon vertices shared across the route.

    Returns
    -------
    _ObstacleSnapshot
        Float twin of the obstacle used for visibility decisions.
    """

    center_x, center_y = float(box.center[0]), float(box.center[1])
    half_x, half_y = float(box.half_extents[0]), float(box.half_extents[1])
    lower_x, lower_y = center_x - half_x, center_y - half_y
    upper_x, upper_y = center_x + half_x, center_y + half_y
    corners: Tuple[_FloatPoint, ...] = (
        (lower_x, lower_y),
        (upper_x, lower_y),
        (upper_x, upper_y),
        (lower_x, upper_y),
    )
    return _ObstacleSnapshot(
        center=(center_x, center_y),
        half_extents=(half_x, half_y),
        box_boundaries=tuple(corners[index] + corners[(index + 1) % 4] for index in range(4)),
        polygons=polygons,
    )


def _segment_parameter(
    start: _FloatPoint, end: _FloatPoint, boundary: _FloatSegment
) -> Optional[float]:
    """Return one transversal query parameter on detached floats.

    Float twin of :func:`_segment_boundary_parameter` for decisions only:
    identical operations in identical order, so the returned parameter is
    bit-identical to ``as_float`` of the tensor computation.

    Parameters
    ----------
    start, end : tuple[float, float]
        Query segment endpoints.
    boundary : tuple[float, float, float, float]
        Boundary segment endpoints as (sx, sy, ex, ey).

    Returns
    -------
    float or None
        Query parameter of the intersection, including endpoints.
    """

    boundary_start_x, boundary_start_y, boundary_end_x, boundary_end_y = boundary
    query_x = end[0] - start[0]
    query_y = end[1] - start[1]
    boundary_x = boundary_end_x - boundary_start_x
    boundary_y = boundary_end_y - boundary_start_y
    denominator = query_x * boundary_y - query_y * boundary_x
    if denominator == 0.0:
        return None
    offset_x = boundary_start_x - start[0]
    offset_y = boundary_start_y - start[1]
    query_parameter = (offset_x * boundary_y - offset_y * boundary_x) / denominator
    boundary_parameter = (offset_x * query_y - offset_y * query_x) / denominator
    if 0.0 <= query_parameter <= 1.0 and 0.0 <= boundary_parameter <= 1.0:
        return query_parameter
    return None


def _segment_parameters(
    start: _FloatPoint, end: _FloatPoint, boundaries: Sequence[_FloatSegment]
) -> List[float]:
    """Collect valid query parameters against a boundary collection.

    Parameters
    ----------
    start, end : tuple[float, float]
        Query segment endpoints.
    boundaries : sequence[tuple[float, float, float, float]]
        Boundary segments to intersect.

    Returns
    -------
    list[float]
        Valid query parameters in boundary order.
    """

    parameters = []
    for boundary in boundaries:
        parameter = _segment_parameter(start, end, boundary)
        if parameter is not None:
            parameters.append(parameter)
    return parameters


def _point_in_convex_polygon(
    point_x: float, point_y: float, polygon: Tuple[_FloatPoint, ...]
) -> bool:
    """Test closed membership in a counter-clockwise convex polygon.

    Parameters
    ----------
    point_x, point_y : float
        Query point.
    polygon : tuple[tuple[float, float], ...]
        Counter-clockwise convex polygon vertices.

    Returns
    -------
    bool
        Whether the point lies inside or on the polygon.
    """

    count = len(polygon)
    for index in range(count):
        start_x, start_y = polygon[index]
        end_x, end_y = polygon[(index + 1) % count]
        if (end_x - start_x) * (point_y - start_y) - (end_y - start_y) * (
            point_x - start_x
        ) < -1e-12:
            return False
    return True


def _cleared_obstacle_contains(point_x: float, point_y: float, obstacle: _ObstacleSnapshot) -> bool:
    """Test strict membership in a terminal-cleared U11 obstacle.

    Parameters
    ----------
    point_x, point_y : float
        Query point.
    obstacle : _ObstacleSnapshot
        Detached decision geometry of the obstacle.

    Returns
    -------
    bool
        Whether the point lies in the residual obstacle interior.
    """

    if not (
        abs(point_x - obstacle.center[0]) < obstacle.half_extents[0]
        and abs(point_y - obstacle.center[1]) < obstacle.half_extents[1]
    ):
        return False
    return not any(
        _point_in_convex_polygon(point_x, point_y, polygon) for polygon in obstacle.polygons
    )


def _segment_cleared_obstacle_interior_intersection(
    start: _FloatPoint,
    end: _FloatPoint,
    obstacle: _ObstacleSnapshot,
    polygon_parameters: Sequence[float],
) -> bool:
    """Test whether a segment crosses a terminal-cleared obstacle interior.

    Parameters
    ----------
    start, end : tuple[float, float]
        Segment endpoints as detached floats.
    obstacle : _ObstacleSnapshot
        Detached decision geometry of the obstacle.
    polygon_parameters : sequence[float]
        Precomputed valid query parameters of this segment against the two
        terminal polygons, shared across every obstacle of the route.

    Returns
    -------
    bool
        Whether any positive-length segment interval lies in the residual obstacle.
    """

    parameters = [0.0, 1.0]
    parameters.extend(_segment_parameters(start, end, obstacle.box_boundaries))
    parameters.extend(polygon_parameters)
    ordered = sorted(set(parameters))
    direction_x = end[0] - start[0]
    direction_y = end[1] - start[1]
    for lower, upper in zip(ordered[:-1], ordered[1:]):
        if upper > lower:
            midpoint = (lower + upper) / 2.0
            if _cleared_obstacle_contains(
                start[0] + midpoint * direction_x,
                start[1] + midpoint * direction_y,
                obstacle,
            ):
                return True
    return False


def _batched_boundary_parameters(
    starts: np.ndarray,
    ends: np.ndarray,
    boundaries: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate segment-boundary parameters in one float64 batch.

    Parameters
    ----------
    starts, ends : numpy.ndarray
        Detached segment endpoints with shape ``[S, 2]``.
    boundaries : numpy.ndarray
        Boundary endpoints with shape ``[B, 4]`` or ``[O, B, 4]``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Query parameters and their validity mask, shaped ``[S, B]`` or
        ``[S, O, B]`` respectively.
    """

    query = ends - starts
    boundary = boundaries[..., 2:] - boundaries[..., :2]
    if boundaries.ndim == 2:
        query_x = query[:, None, 0]
        query_y = query[:, None, 1]
        boundary_x = boundary[None, :, 0]
        boundary_y = boundary[None, :, 1]
        offset = boundaries[None, :, :2] - starts[:, None, :]
    else:
        query_x = query[:, None, None, 0]
        query_y = query[:, None, None, 1]
        boundary_x = boundary[None, :, :, 0]
        boundary_y = boundary[None, :, :, 1]
        offset = boundaries[None, :, :, :2] - starts[:, None, None, :]
    denominator = query_x * boundary_y - query_y * boundary_x
    nonparallel = denominator != 0.0
    safe_denominator = np.where(nonparallel, denominator, np.ones_like(denominator))
    query_parameter = (offset[..., 0] * boundary_y - offset[..., 1] * boundary_x) / safe_denominator
    boundary_parameter = (offset[..., 0] * query_y - offset[..., 1] * query_x) / safe_denominator
    valid = (
        nonparallel
        & (query_parameter >= 0.0)
        & (query_parameter <= 1.0)
        & (boundary_parameter >= 0.0)
        & (boundary_parameter <= 1.0)
    )
    return query_parameter, valid


def _matched_boundary_parameters(
    starts: np.ndarray,
    ends: np.ndarray,
    boundaries: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate one boundary collection per corresponding query segment.

    Parameters
    ----------
    starts, ends : numpy.ndarray
        Detached segment endpoints with shape ``[S, 2]``.
    boundaries : numpy.ndarray
        Corresponding boundary collections with shape ``[S, B, 4]``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Query parameters and validity masks, both shaped ``[S, B]``.
    """

    query = ends - starts
    boundary = boundaries[..., 2:] - boundaries[..., :2]
    query_x = query[:, None, 0]
    query_y = query[:, None, 1]
    boundary_x = boundary[:, :, 0]
    boundary_y = boundary[:, :, 1]
    offset = boundaries[..., :2] - starts[:, None, :]
    denominator = query_x * boundary_y - query_y * boundary_x
    nonparallel = denominator != 0.0
    safe_denominator = np.where(nonparallel, denominator, np.ones_like(denominator))
    query_parameter = (offset[..., 0] * boundary_y - offset[..., 1] * boundary_x) / safe_denominator
    boundary_parameter = (offset[..., 0] * query_y - offset[..., 1] * query_x) / safe_denominator
    valid = (
        nonparallel
        & (query_parameter >= 0.0)
        & (query_parameter <= 1.0)
        & (boundary_parameter >= 0.0)
        & (boundary_parameter <= 1.0)
    )
    return query_parameter, valid


def _snapshot_block_arrays(
    snapshots: Sequence[_ObstacleSnapshot],
    polygon_boundaries: Sequence[_FloatSegment],
) -> Tuple[np.ndarray, ...]:
    """Convert one route's obstacle snapshots to the batched decision arrays.

    Pure per-snapshot conversions of already-detached floats -- building them
    once per route and reusing across every visibility-row call yields the
    same arrays as rebuilding per call (the historical behaviour), element
    for element.
    """

    polygon_array = np.asarray(polygon_boundaries, dtype=np.float64)
    polygon_points = np.asarray(snapshots[0].polygons, dtype=np.float64)
    polygon_starts = polygon_points
    polygon_edges = np.roll(polygon_points, shift=-1, axis=1) - polygon_points
    box_boundaries = np.asarray(
        [snapshot.box_boundaries for snapshot in snapshots], dtype=np.float64
    )
    centers = np.asarray([snapshot.center for snapshot in snapshots], dtype=np.float64)
    half_extents = np.asarray([snapshot.half_extents for snapshot in snapshots], dtype=np.float64)
    # One ULP keeps the broad phase conservative when extreme coordinates
    # round a boundary-adjacent segment onto the reconstructed AABB edge.
    box_minimum = np.nextafter(centers - half_extents, -np.inf)
    box_maximum = np.nextafter(centers + half_extents, np.inf)
    return (
        polygon_array,
        polygon_starts,
        polygon_edges,
        box_boundaries,
        centers,
        half_extents,
        box_minimum,
        box_maximum,
    )


def _segments_blocked_vectorized(
    starts: Sequence[_FloatPoint],
    ends: Sequence[_FloatPoint],
    snapshots: Sequence[_ObstacleSnapshot],
    polygon_boundaries: Sequence[_FloatSegment],
    arrays: Optional[Tuple[np.ndarray, ...]] = None,
) -> List[bool]:
    """Test many U11 visibility edges against all cleared obstacles.

    Parameters
    ----------
    starts, ends : sequence[tuple[float, float]]
        Detached query segment endpoints in matching order.
    snapshots : sequence[_ObstacleSnapshot]
        Detached obstacle geometry for one route.
    polygon_boundaries : sequence[tuple[float, float, float, float]]
        The two terminal-polygon boundaries shared by every obstacle.
    arrays : tuple[numpy.ndarray, ...], optional
        Precomputed ``_snapshot_block_arrays`` for this route; None rebuilds
        them from ``snapshots`` (identical values either way).

    Returns
    -------
    list[bool]
        Blocked decisions in input segment order.

    Notes
    -----
    Invalid intersections are represented by duplicate zero parameters before
    sorting. The scalar path removes duplicates with ``set``; duplicates create
    only zero-width intervals, which that path also skips. All score-visible
    coordinates and shortest-path lengths remain on the shipped scalar path.
    """

    if len(starts) == 0:
        return []
    if not snapshots:
        return [False] * len(starts)
    if arrays is None:
        arrays = _snapshot_block_arrays(snapshots, polygon_boundaries)
    (
        polygon_array,
        polygon_starts,
        polygon_edges,
        box_boundaries,
        centers,
        half_extents,
        box_minimum,
        box_maximum,
    ) = arrays
    blocked = [False] * len(starts)
    all_starts = np.asarray(starts, dtype=np.float64)
    all_ends = np.asarray(ends, dtype=np.float64)
    # Polygon parameters are filled lazily per block for exactly the segments
    # that reach the narrow phase (their rows are the only ones ever read);
    # each filled row is the same row the historical all-segments batch
    # produced, because `_batched_boundary_parameters` has no cross-row term.
    all_polygon_parameters = np.zeros((all_starts.shape[0], polygon_array.shape[0]))
    segment_minimum = np.minimum(all_starts, all_ends)
    segment_maximum = np.maximum(all_starts, all_ends)
    blocked_mask = np.zeros(len(starts), dtype=bool)
    # The broad phase is row-blocked: a dense scene's full [segments,
    # obstacles] candidate matrix reaches tens of GiB (measured 21.9 GiB at
    # 10.3M x 1136), so blocks bound memory. Candidate verdicts are decided
    # row-independently in the narrow phase, so the candidate stream's exact
    # membership and order are value-inert: every extra conservative keep
    # narrow-evaluates to False and every skip is of an already-decided OR.
    total_segments = all_starts.shape[0]
    obstacle_count = centers.shape[0]
    row_block = max(1, _U11_BROAD_ROW_BLOCK // max(1, obstacle_count))
    slack_scale = 16.0 * np.finfo(np.float64).eps
    for row_start in range(0, total_segments, row_block):
        row_end = min(total_segments, row_start + row_block)
        alive = np.nonzero(~blocked_mask[row_start:row_end])[0] + row_start
        if alive.size == 0:
            continue
        # AABB overlap, axis-split to avoid [rows, obstacles, 2] temporaries
        # (identical boolean per pair: reordered AND of the same comparisons).
        block_candidates = (segment_maximum[alive, None, 0] > box_minimum[None, :, 0]) & (
            segment_minimum[alive, None, 0] < box_maximum[None, :, 0]
        )
        block_candidates &= (segment_maximum[alive, None, 1] > box_minimum[None, :, 1]) & (
            segment_minimum[alive, None, 1] < box_maximum[None, :, 1]
        )
        # Separating-axis prune: if the segment's LINE has the whole box
        # strictly on one side (exact |cross(d, c-a)| > exact |d_y|h_x +
        # |d_x|h_y, the box's projection onto the segment normal), no point
        # of the segment lies strictly inside the box, so the narrow phase's
        # every interval midpoint fails |p-c| < h and the pair's verdict is
        # False -- dropping it is decision-inert. The AABB phase alone keeps
        # ~200 candidates per long diagonal segment where only ~2-5 truly
        # cross (the boxes are small node markers), which is what made the
        # dense tail superlinear-hostile. FP conservatism: each side is
        # computed with <= 3 rounding steps (relative error bound
        # (1+u)^3 - 1 < 4u each, u = eps/2), and every rounding error is
        # bounded in ABSOLUTE terms by 4u times the cancellation-free
        # magnitude of that side (mag for the cross, rhs for the extents
        # side). A pair is only dropped when fl|cross| exceeds fl(rhs) plus
        # 16*eps*(mag + rhs) -- an over-cover of the summed error bounds by
        # more than 2x, itself computed from nonnegative terms so its own
        # rounding is second-order and absorbed by that margin. Pairs at or
        # inside the slack stay and the narrow phase decides them unchanged.
        directions = all_ends[alive] - all_starts[alive]
        abs_dx = np.abs(directions[:, 0])[:, None]
        abs_dy = np.abs(directions[:, 1])[:, None]
        offsets_x = centers[None, :, 0] - all_starts[alive, None, 0]
        offsets_y = centers[None, :, 1] - all_starts[alive, None, 1]
        cross = directions[:, 0][:, None] * offsets_y - directions[:, 1][:, None] * offsets_x
        rhs = abs_dy * half_extents[None, :, 0] + abs_dx * half_extents[None, :, 1]
        magnitude = abs_dx * np.abs(offsets_y) + abs_dy * np.abs(offsets_x)
        block_candidates &= np.abs(cross) <= rhs + slack_scale * (magnitude + rhs)
        local_indices = np.argwhere(block_candidates)
        if local_indices.size == 0:
            continue
        candidate_indices = np.column_stack((alive[local_indices[:, 0]], local_indices[:, 1]))
        needed = np.unique(candidate_indices[:, 0])
        needed_parameters, needed_valid = _batched_boundary_parameters(
            all_starts[needed], all_ends[needed], polygon_array
        )
        all_polygon_parameters[needed] = np.where(
            needed_valid, needed_parameters, np.zeros_like(needed_parameters)
        )
        _blocked_batches(
            candidate_indices,
            all_starts,
            all_ends,
            all_polygon_parameters,
            box_boundaries,
            centers,
            half_extents,
            polygon_starts,
            polygon_edges,
            blocked,
            blocked_mask,
        )
    return blocked


def _blocked_batches(
    candidate_indices: np.ndarray,
    all_starts: np.ndarray,
    all_ends: np.ndarray,
    all_polygon_parameters: np.ndarray,
    box_boundaries: np.ndarray,
    centers: np.ndarray,
    half_extents: np.ndarray,
    polygon_starts: np.ndarray,
    polygon_edges: np.ndarray,
    blocked: List[bool],
    blocked_mask: np.ndarray,
) -> None:
    """Run one broad-phase block's candidates through the U11 narrow phase.

    Mutates ``blocked``/``blocked_mask`` in place; every surviving row's
    arithmetic is the historical batch pipeline unchanged.

    Candidates are consumed per segment in argwhere order with geometrically
    doubling chunks (the batched twin of the scalar path's per-segment
    ``any()``): a segment stops contributing rows once its OR is decided, so
    dense fields pay for the few obstacles up to the first blocker instead of
    every overlapping obstacle (measured 72-206 candidates per segment with
    ~75% of segments blocked). Each evaluated pair's verdict is decided by
    that pair alone; skipping rows of already-blocked segments cannot change
    any segment's final OR, so the emitted flags are byte-identical to the
    exhaustive pipeline.
    """

    total = candidate_indices.shape[0]
    if total == 0:
        return
    segments = candidate_indices[:, 0]
    group_starts = np.flatnonzero(np.r_[True, segments[1:] != segments[:-1]])
    group_ends = np.r_[group_starts[1:], total]
    group_segments = segments[group_starts]
    cursor = group_starts.copy()
    chunk = np.ones(group_starts.shape[0], dtype=np.int64)
    while True:
        alive = (cursor < group_ends) & ~blocked_mask[group_segments]
        if not alive.any():
            return
        counts = np.minimum(chunk[alive], group_ends[alive] - cursor[alive])
        row_starts = np.repeat(cursor[alive], counts)
        row_offsets = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        round_rows = candidate_indices[row_starts + row_offsets]
        cursor[alive] += counts
        chunk[alive] *= 2
        for batch_start in range(0, round_rows.shape[0], _U11_PAIR_OBSTACLE_BUDGET):
            batch = round_rows[batch_start : batch_start + _U11_PAIR_OBSTACLE_BUDGET]
            batch = batch[~blocked_mask[batch[:, 0]]]
            if batch.shape[0] == 0:
                continue
            _blocked_candidate_batch(
                batch,
                all_starts,
                all_ends,
                all_polygon_parameters,
                box_boundaries,
                centers,
                half_extents,
                polygon_starts,
                polygon_edges,
                blocked,
                blocked_mask,
            )


def _blocked_candidate_batch(
    batch: np.ndarray,
    all_starts: np.ndarray,
    all_ends: np.ndarray,
    all_polygon_parameters: np.ndarray,
    box_boundaries: np.ndarray,
    centers: np.ndarray,
    half_extents: np.ndarray,
    polygon_starts: np.ndarray,
    polygon_edges: np.ndarray,
    blocked: List[bool],
    blocked_mask: np.ndarray,
) -> None:
    """Decide one budget-bounded batch of (segment, obstacle) pairs.

    The pair arithmetic below is the historical batch pipeline unchanged;
    every verdict depends only on its own row.
    """

    segment_indices = batch[:, 0]
    obstacle_indices = batch[:, 1]
    start_tensor = all_starts[segment_indices]
    end_tensor = all_ends[segment_indices]
    polygon_parameters = all_polygon_parameters[segment_indices]
    box_parameters, box_valid = _matched_boundary_parameters(
        start_tensor, end_tensor, box_boundaries[obstacle_indices]
    )
    box_parameters = np.where(box_valid, box_parameters, np.zeros_like(box_parameters))
    parameters = np.sort(
        np.concatenate(
            (
                np.zeros((batch.shape[0], 1), dtype=np.float64),
                np.ones((batch.shape[0], 1), dtype=np.float64),
                box_parameters,
                polygon_parameters,
            ),
            axis=1,
        ),
        axis=1,
    )
    lower = parameters[:, :-1]
    upper = parameters[:, 1:]
    midpoints = (lower + upper) / 2.0
    direction = end_tensor - start_tensor
    points = start_tensor[:, None, :] + midpoints[..., None] * direction[:, None, :]
    active = (upper > lower) & (
        np.abs(points - centers[obstacle_indices, None, :])
        < half_extents[obstacle_indices, None, :]
    ).all(axis=2)
    active_indices = np.argwhere(active)
    if active_indices.size == 0:
        return
    active_points = points[active]
    offsets = active_points[:, None, None, :] - polygon_starts[None, :, :, :]
    crosses = (
        polygon_edges[None, :, :, 0] * offsets[..., 1]
        - polygon_edges[None, :, :, 1] * offsets[..., 0]
    )
    inside_terminal_polygon = (crosses >= -1e-12).all(axis=2).any(axis=1)
    residual_candidates = np.unique(active_indices[~inside_terminal_polygon, 0])
    residual_segments = np.unique(segment_indices[residual_candidates])
    blocked_mask[residual_segments] = True
    for index in residual_segments:
        blocked[int(index)] = True


def _route_obstacles_vectorized(
    scene: Scene,
    route: Route,
    cap: Scalar,
) -> List[BoxGeometry]:
    """Select U11 route obstacles with a batched detached distance test.

    Parameters
    ----------
    scene : Scene
        Validated scene with derived node boxes.
    route : Route
        Route whose terminal-owned boxes are exempt.
    cap : float or torch.Tensor
        Four-chord obstacle-search cap.

    Returns
    -------
    list[BoxGeometry]
        Eligible obstacles in canonical node-box order.
    """

    if not scene.node_boxes:
        return []
    source, target = scene.graph.edges[route.edge_index]
    centers = torch.stack([box.center.detach().cpu() for box in scene.node_boxes])
    start = route.points[0].detach().cpu()
    end = route.points[-1].detach().cpu()
    within_cap = torch.linalg.vector_norm(centers - start, dim=1) + torch.linalg.vector_norm(
        centers - end, dim=1
    ) <= as_float(cap)
    return [
        box
        for index, box in enumerate(scene.node_boxes)
        if box.owner not in {source, target} and bool(within_cap[index])
    ]


def _cleared_obstacle_vertices(
    box: BoxGeometry,
    terminal_polygons: Sequence[torch.Tensor],
    snapshot: _ObstacleSnapshot,
) -> List[torch.Tensor]:
    """Return visibility vertices of a terminal-cleared box.

    Membership decisions run on the detached float snapshot; the candidate
    COORDINATES stay tensors (live inside a trace) because the chosen
    visibility path's length is rebuilt from them.

    Parameters
    ----------
    box : BoxGeometry
        Uninflated node obstacle.
    terminal_polygons : sequence[torch.Tensor]
        Polygonized terminal disks removed from the box.
    snapshot : _ObstacleSnapshot
        Detached decision geometry of the same obstacle.

    Returns
    -------
    list[torch.Tensor]
        Residual box corners, disk vertices inside the box, and boundary intersections.
    """

    box_segments = _box_boundary_segments(box)
    candidates = [start for start, _ in box_segments]
    center_x, center_y = snapshot.center
    half_x, half_y = snapshot.half_extents
    for polygon_index, polygon in enumerate(terminal_polygons):
        for point_index, point in enumerate(polygon):
            point_x, point_y = snapshot.polygons[polygon_index][point_index]
            if abs(point_x - center_x) <= half_x and abs(point_y - center_y) <= half_y:
                candidates.append(point)
        for box_start, box_end in box_segments:
            for polygon_start, polygon_end in _polygon_boundary_segments(polygon):
                intersection = _segment_boundary_parameter(
                    box_start,
                    box_end,
                    polygon_start,
                    polygon_end,
                )
                if intersection is not None:
                    candidates.append(intersection[1])
    retained = [
        point
        for point in candidates
        if not _cleared_obstacle_contains(float(point[0]), float(point[1]), snapshot)
    ]
    unique: Dict[Tuple[float, float], torch.Tensor] = {}
    for point in retained:
        unique[(float(point[0]), float(point[1]))] = point
    return [unique[key] for key in sorted(unique)]


def _cleared_obstacle_vertices_vectorized(
    box: BoxGeometry,
    terminal_polygons: Sequence[torch.Tensor],
    snapshot: _ObstacleSnapshot,
) -> List[torch.Tensor]:
    """Return U11 visibility vertices after batched intersection decisions.

    Parameters
    ----------
    box : BoxGeometry
        Uninflated node obstacle.
    terminal_polygons : sequence[torch.Tensor]
        Polygonized terminal disks removed from the box.
    snapshot : _ObstacleSnapshot
        Detached decision geometry of the same obstacle.

    Returns
    -------
    list[torch.Tensor]
        The scalar scorer's exact tensor coordinates in canonical sorted order.

    Notes
    -----
    The batch identifies which of the 128 box/polygon boundary pairs intersect.
    Every surviving intersection point is then rebuilt with
    :func:`_segment_boundary_parameter`, preserving score-visible coordinates
    and their traced autograd graph exactly.
    """

    box_segments = _box_boundary_segments(box)
    candidates = [start for start, _ in box_segments]
    center_x, center_y = snapshot.center
    half_x, half_y = snapshot.half_extents
    box_boundaries = np.asarray(snapshot.box_boundaries, dtype=np.float64)
    box_starts = box_boundaries[:, :2]
    box_directions = box_boundaries[:, 2:] - box_starts
    for polygon_index, polygon in enumerate(terminal_polygons):
        for point_index, point in enumerate(polygon):
            point_x, point_y = snapshot.polygons[polygon_index][point_index]
            if abs(point_x - center_x) <= half_x and abs(point_y - center_y) <= half_y:
                candidates.append(point)
        polygon_segments = _polygon_boundary_segments(polygon)
        polygon_boundaries = np.asarray(
            [
                snapshot.polygons[polygon_index][segment_index]
                + snapshot.polygons[polygon_index][(segment_index + 1) % len(polygon)]
                for segment_index in range(len(polygon))
            ],
            dtype=np.float64,
        )
        polygon_starts = polygon_boundaries[:, :2]
        polygon_directions = polygon_boundaries[:, 2:] - polygon_starts
        denominator = (
            box_directions[:, None, 0] * polygon_directions[None, :, 1]
            - box_directions[:, None, 1] * polygon_directions[None, :, 0]
        )
        nonparallel = denominator != 0.0
        safe_denominator = np.where(nonparallel, denominator, np.ones_like(denominator))
        offset = polygon_starts[None, :, :] - box_starts[:, None, :]
        box_parameter = (
            offset[..., 0] * polygon_directions[None, :, 1]
            - offset[..., 1] * polygon_directions[None, :, 0]
        ) / safe_denominator
        polygon_parameter = (
            offset[..., 0] * box_directions[:, None, 1]
            - offset[..., 1] * box_directions[:, None, 0]
        ) / safe_denominator
        intersects = (
            nonparallel
            & (box_parameter >= 0.0)
            & (box_parameter <= 1.0)
            & (polygon_parameter >= 0.0)
            & (polygon_parameter <= 1.0)
        )
        for box_index, polygon_segment_index in np.argwhere(intersects):
            intersection = _segment_boundary_parameter(
                box_segments[int(box_index)][0],
                box_segments[int(box_index)][1],
                polygon_segments[int(polygon_segment_index)][0],
                polygon_segments[int(polygon_segment_index)][1],
            )
            if intersection is not None:
                candidates.append(intersection[1])
    retained = [
        point
        for point in candidates
        if not _cleared_obstacle_contains(float(point[0]), float(point[1]), snapshot)
    ]
    unique: Dict[Tuple[float, float], torch.Tensor] = {}
    for point in retained:
        unique[(float(point[0]), float(point[1]))] = point
    return [unique[key] for key in sorted(unique)]


def _route_baseline(
    scene: Scene, route: Route, *, vectorized: bool = False
) -> Tuple[Scalar, Scalar]:
    """Compute U11's obstacle-aware capped visibility-path baseline.

    Parameters
    ----------
    scene : Scene
        Validated scene with derived node boxes.
    route : Route
        Visible route whose endpoint obstacles are exempt.
    vectorized : bool
        Whether review-gated detached visibility decisions use batched float64
        arithmetic. False retains the shipped scalar implementation.

    Returns
    -------
    tuple[float or torch.Tensor, float or torch.Tensor]
        Capped baseline length and its total absolute turning. ALL visibility
        decisions -- blocked segments, retained vertices, and the shortest
        PATH -- run on detached float geometry (each is a decision, and the
        float predicates are bit-identical twins of the historical tensor
        reads); the chosen path's length is then rebuilt live from the vertex
        geometry inside a trace, and the contract-named ``softmin`` with
        ``t_soft = 0.1 * chord_e`` (U11.md sec 7 step 4) flows tensors.
    """

    source, target = scene.graph.edges[route.edge_index]
    start = route.points[0]
    end = route.points[-1]
    chord = keep(torch.linalg.vector_norm(end - start))
    if as_float(chord) == 0.0:
        return max(scene.intrinsic_unit, 1e-300), 0.0
    cap = 4.0 * chord
    clear_radius = _U11_TERMINAL_CLEAR_RADIUS * scene.intrinsic_unit
    terminal_polygons = (
        _regular_polygon(start, clear_radius, _U11_TERMINAL_DISK_SIDES),
        _regular_polygon(end, clear_radius, _U11_TERMINAL_DISK_SIDES),
    )
    if vectorized and len(scene.node_boxes) >= _U11_OBSTACLE_SELECTION_BATCH_MIN:
        obstacles = _route_obstacles_vectorized(scene, route, cap)
    else:
        obstacles = [
            box
            for box in scene.node_boxes
            if box.owner not in {source, target}
            and float(torch.linalg.vector_norm(box.center - start))
            + float(torch.linalg.vector_norm(box.center - end))
            <= as_float(cap)
        ]
    polygons_f = _polygon_floats(terminal_polygons)
    polygon_boundaries = _polygon_boundary_floats(polygons_f)
    snapshots = [_obstacle_snapshot(box, polygons_f) for box in obstacles]
    start_f = (float(start[0]), float(start[1]))
    end_f = (float(end[0]), float(end[1]))
    chord_polygon_parameters = _segment_parameters(start_f, end_f, polygon_boundaries)
    # One chord cannot amortize tensor construction; the vectorized path begins
    # at the quadratic visibility graph after this identical scalar fast path.
    chord_blocked = any(
        _segment_cleared_obstacle_interior_intersection(
            start_f,
            end_f,
            snapshot,
            chord_polygon_parameters,
        )
        for snapshot in snapshots
    )
    if not chord_blocked:
        return chord, 0.0
    obstacle_vertices: List[torch.Tensor] = []
    for box, snapshot in zip(obstacles, snapshots):
        if vectorized:
            obstacle_vertices.extend(
                _cleared_obstacle_vertices_vectorized(box, terminal_polygons, snapshot)
            )
        else:
            obstacle_vertices.extend(_cleared_obstacle_vertices(box, terminal_polygons, snapshot))
    unique_vertices: Dict[Tuple[float, float], torch.Tensor] = {}
    for point in obstacle_vertices:
        unique_vertices[(float(point[0]), float(point[1]))] = point
    vertices = [start, end] + [unique_vertices[key] for key in sorted(unique_vertices)]
    vertices_f: List[_FloatPoint] = [(float(point[0]), float(point[1])) for point in vertices]
    if vectorized:
        # LAZY visibility rows: the emitted values read ONLY distances[1] and
        # paths[1], so the full [V^2/2, obstacles] blocked matrix (10.3M x
        # 1136 measured on the dense tail, >200s for ONE route) is replaced by
        # per-vertex rows computed at first pop. Every per-pair decision is the
        # SAME `_segments_blocked_vectorized` on the SAME (low, high)-oriented
        # segment -- its row arithmetic has no cross-row term (the narrow
        # phase's blocked-row skip only elides rows whose OR is already
        # decided), so each boolean is bit-identical to the full-matrix call.
        neighbor_rows: Dict[int, List[Tuple[int, float]]] = {}
        block_arrays = _snapshot_block_arrays(snapshots, polygon_boundaries) if snapshots else None
        vertex_array = np.asarray(vertices_f, dtype=np.float64)
        # Lower bounds on the remaining distance to the target vertex, used
        # only by the corridor guard below (never emitted).
        target_lower = np.linalg.norm(vertex_array - vertex_array[1], axis=1)

        def _expand(node: int) -> List[Tuple[int, float]]:
            row = neighbor_rows.get(node)
            if row is not None:
                return row
            others = [other for other in range(len(vertices)) if other != node]
            ordered = [(node, other) if node < other else (other, node) for other in others]
            # Fancy-indexed views of the SAME float64 coordinates the per-call
            # list construction would produce -- np.asarray inside the callee
            # passes ndarrays through untouched.
            blocked_row = _segments_blocked_vectorized(
                vertex_array[np.asarray([low for low, _ in ordered], dtype=np.intp)],
                vertex_array[np.asarray([high for _, high in ordered], dtype=np.intp)],
                snapshots,
                polygon_boundaries,
                arrays=block_arrays,
            )
            row = []
            for (low, high), other, other_blocked in zip(ordered, others, blocked_row):
                if other_blocked:
                    continue
                row.append((other, float(torch.linalg.vector_norm(vertices[high] - vertices[low]))))
            neighbor_rows[node] = row
            return row

    else:
        neighbors: List[List[Tuple[int, float]]] = [[] for _ in vertices]
        for left_index, left in enumerate(vertices):
            left_f = vertices_f[left_index]
            for right_index in range(left_index + 1, len(vertices)):
                right = vertices[right_index]
                right_f = vertices_f[right_index]
                pair_polygon_parameters = _segment_parameters(left_f, right_f, polygon_boundaries)
                if any(
                    _segment_cleared_obstacle_interior_intersection(
                        left_f,
                        right_f,
                        snapshot,
                        pair_polygon_parameters,
                    )
                    for snapshot in snapshots
                ):
                    continue
                distance = float(torch.linalg.vector_norm(right - left))
                neighbors[left_index].append((right_index, distance))
                neighbors[right_index].append((left_index, distance))

        def _expand(node: int) -> List[Tuple[int, float]]:
            return sorted(neighbors[node])

    distances = [math.inf] * len(vertices)
    paths: List[Tuple[int, ...]] = [tuple() for _ in vertices]
    distances[0] = 0.0
    paths[0] = (0,)
    queue: List[Tuple[float, Tuple[int, ...], int]] = [(0.0, (0,), 0)]
    while queue:
        distance, path, node = heapq.heappop(queue)
        if distance != distances[node] or path != paths[node]:
            continue
        if node == 1:
            # Settled-at-first-valid-pop: every edge length is strictly
            # positive (vertices are unique), so any equal-distance relaxation
            # into node 1 comes from a predecessor at STRICTLY smaller
            # distance, which popped earlier -- (distances[1], paths[1]) is
            # already the (distance, lexicographic-path) minimum here and no
            # later pop can change it. Skipped pops feed nothing else.
            break
        if (
            vectorized
            and distances[1] != math.inf
            and (distance + float(target_lower[node])) * (1.0 - 1e-9) > distances[1]
        ):
            # Corridor guard: edge lengths are Euclidean, so every completion
            # of this node's path into node 1 has EXACT length >= exact(d_u) +
            # euclid(u, 1), and its float candidate at node 1 deviates from
            # that by < (2V+5) rounding units (one norm per leg, <= V
            # accumulating additions) -- < 1e-11 relative for any V here,
            # over-covered 100x by the 1e-9 margin. The guard therefore
            # certifies every such candidate exceeds the recorded distances[1]
            # strictly (no improvement, and no equal-distance lexicographic
            # tie, which requires exact float equality). Intermediate nodes
            # this expansion could have updated only matter through
            # completions into node 1, which the same bound dominates, so
            # skipping the expansion leaves (distances[1], paths[1]) -- the
            # only values the emitted baseline and turning read -- unchanged.
            # Emitted values never read this guard's arithmetic.
            continue
        for neighbor, edge_length in _expand(node):
            candidate = distance + edge_length
            candidate_path = path + (neighbor,)
            if candidate < distances[neighbor] or (
                candidate == distances[neighbor] and candidate_path < paths[neighbor]
            ):
                distances[neighbor] = candidate
                paths[neighbor] = candidate_path
                heapq.heappush(queue, (candidate, candidate_path, neighbor))
    visible: Scalar = distances[1]
    temperature = 0.1 * chord
    if math.isfinite(distances[1]):
        if tracing_active():
            # The float Dijkstra above decided WHICH path is shortest; the
            # chosen path's length is rebuilt live from the vertex geometry
            # (accumulation order may differ from the float search by ULPs,
            # the measured traced-vs-exact envelope, never the decision).
            path_indices = paths[1]
            visible = torch.stack(
                [
                    torch.linalg.vector_norm(vertices[second] - vertices[first])
                    for first, second in zip(path_indices[:-1], path_indices[1:])
                ]
            ).sum()
        minimum = p_min(visible, cap)
        soft_min = minimum - temperature * p_log(
            p_exp(-(visible - minimum) / temperature) + p_exp(-(cap - minimum) / temperature)
        )
        baseline = p_max(chord, soft_min)
        path_points = torch.stack([vertices[index] for index in paths[1]])
        turning = p_sum([p_abs(value) for value in _signed_route_turns(path_points)])
    else:
        baseline = cap
        turning = 0.0
    return baseline, turning


def U12(scene: Scene) -> FacetResult:
    """Path/edge continuity. Frozen SHA-256: 0170de3c481a98b8d063ff505c039de969221b1e43c3bef596a64c7e0c083d47."""

    directions = _incident_secants(scene)
    degrees = [0] * scene.node_count
    loop_nodes = set()
    for source, target in scene.graph.edges:
        if source == target:
            loop_nodes.add(source)
            degrees[source] += 2
        else:
            degrees[source] += 1
            degrees[target] += 1
    nodes = [node for node, degree in enumerate(degrees) if degree == 2 and node not in loop_nodes]
    if len(nodes) < 5:
        return na_result("no_degree2_chains")
    deviations: List[Scalar] = []
    for node in nodes:
        records = directions[node]
        if len(records) != 2:
            deviations.append(0.0)
            continue
        left_angle, _, left_confidence = records[0]
        right_angle, _, right_confidence = records[1]
        if isinstance(left_angle, torch.Tensor) or isinstance(right_angle, torch.Tensor):
            left = torch.stack(
                (_p_cos(_scalar_tensor(left_angle)), _p_sin(_scalar_tensor(left_angle)))
            )
            right = torch.stack(
                (_p_cos(_scalar_tensor(right_angle)), _p_sin(_scalar_tensor(right_angle)))
            )
        else:
            left = torch.tensor([math.cos(left_angle), math.sin(left_angle)], dtype=torch.float64)
            right = torch.tensor(
                [math.cos(right_angle), math.sin(right_angle)], dtype=torch.float64
            )
        cosine = p_min(1.0, p_max(-1.0, keep(torch.dot(-left, right))))
        if as_float(cosine) >= 1.0:
            # acos has infinite slope at the clamp boundary; the historical
            # float value there is exactly acos(1)/pi = 0 (perfect continuation).
            deviation: Scalar = math.acos(1.0) / math.pi
        elif as_float(cosine) <= -1.0:
            deviation = math.acos(-1.0) / math.pi
        else:
            deviation = _p_acos(cosine) / math.pi
        deviations.append(left_confidence * right_confidence * deviation)
    defect = global_blend(deviations)
    return value_result(
        defect,
        {"U12.headline": defect},
        {"degree2_node_count": len(nodes)},
    )


def U13(scene: Scene) -> FacetResult:
    """Near-parallel edge ambiguity / bundle confusion. Frozen SHA-256: 359205189c5fa413aae039d4033602af0a665c9e93e583c274b57d1341d1ca43."""

    if scene.edge_count < 2:
        return na_result("too_few_edges")
    routes = resolved_routes(scene)
    bundles = scene.graph.edge_bundles or tuple(None for _ in range(scene.edge_count))
    contributions: List[Scalar] = []
    pair_records: List[Tuple[int, int, float]] = []
    for left_index, left in enumerate(routes):
        for right in routes[left_index + 1 :]:
            if (
                bundles[left.edge_index] is not None
                and bundles[left.edge_index] == bundles[right.edge_index]
            ):
                continue
            shared_nodes = set(scene.graph.edges[left.edge_index]) & set(
                scene.graph.edges[right.edge_index]
            )
            left_points = left.points
            right_points = right.points
            if shared_nodes:
                left_edge = scene.graph.edges[left.edge_index]
                right_edge = scene.graph.edges[right.edge_index]
                left_points = _trim_polyline(
                    left_points,
                    2.0 * scene.intrinsic_unit if left_edge[0] in shared_nodes else 0.0,
                    2.0 * scene.intrinsic_unit if left_edge[1] in shared_nodes else 0.0,
                )
                right_points = _trim_polyline(
                    right_points,
                    2.0 * scene.intrinsic_unit if right_edge[0] in shared_nodes else 0.0,
                    2.0 * scene.intrinsic_unit if right_edge[1] in shared_nodes else 0.0,
                )
            contribution = _parallel_route_integral(left_points, right_points, scene.intrinsic_unit)
            contributions.append(contribution)
            pair_records.append((left.edge_index, right.edge_index, _raw(contribution)))
    degrees = [0] * scene.node_count
    for source, target in scene.graph.edges:
        degrees[source] += 1
        degrees[target] += 1
    opportunity = int(scene.edge_count + sum(degree * (degree - 1) / 2.0 for degree in degrees))
    raw_sum = p_sum(contributions)
    values: Dict[str, Scalar] = {
        "U13.i": _raw_u13_blend(contributions, opportunity),
    }
    bundle_defects: List[Scalar] = []
    if scene.graph.edge_bundles is not None:
        grouped: DefaultDict[str, List[Route]] = defaultdict(list)
        for route, bundle in zip(routes, bundles):
            if bundle is not None:
                grouped[bundle].append(route)
        for members in grouped.values():
            for end_fraction in (0.10, 0.90):
                departure_points = [
                    _point_at_arc_fraction(route.points, end_fraction) for route in members
                ]
                minimum = min(
                    [
                        keep(torch.linalg.vector_norm(left - right))
                        for left_index, left in enumerate(departure_points)
                        for right in departure_points[left_index + 1 :]
                    ],
                    key=as_float,
                )
                bundle_defects.append(p_max(0.0, 1.0 - minimum / (0.5 * scene.intrinsic_unit)) ** 2)
        if bundle_defects:
            values["U13.ii"] = blend_with_weights(
                bundle_defects,
                None,
                (0.65 * 0.4 / 0.9, 0.25 * 0.4 / 0.9, 0.6),
            )
    return mean_result(
        "U13",
        values,
        {
            "opportunity": opportunity,
            "sum_C_ef": _raw(raw_sum),
            "pair_contributions": tuple(pair_records),
            "bundle_end_defects": tuple(_raw(defect) for defect in bundle_defects),
        },
    )


def _raw_u13_blend(values: List[Scalar], opportunity: int) -> Scalar:
    """Aggregate U13 pair integrals through its mean/tail/smoothmax blend.

    Parameters
    ----------
    values : list[float or torch.Tensor]
        Nonnegative admitted pair integrals.
    opportunity : int
        Frozen input-only normalizer for the mean component. Tail components use
        the contract's full route-pair population, including measured zeros.

    Returns
    -------
    float or torch.Tensor
        Contract-blended ambiguity defect; a live tensor inside a trace.
    """

    if opportunity <= 0:
        return 0.0
    if not values:
        return 0.0
    population = sorted(values, key=as_float)
    mean = p_sum(population) / opportunity
    tail_mass = 0.10 * len(population)
    remaining = tail_mass
    tail_sum: Scalar = 0.0
    for value in reversed(population):
        selected = min(1.0, remaining)
        tail_sum += selected * value
        remaining -= selected
        if remaining <= 0.0:
            break
    cvar = tail_sum / tail_mass
    maximum = population[-1]
    smooth_maximum = maximum + 0.05 * p_log(
        p_sum([p_exp((value - maximum) / 0.05) for value in population]) / len(population)
    )

    def saturate(value: Scalar) -> Scalar:
        """Apply U13's common ``x/(x+0.05)`` saturation.

        Parameters
        ----------
        value : float or torch.Tensor
            Nonnegative raw aggregate.

        Returns
        -------
        float or torch.Tensor
            Bounded component burden.
        """

        return value / (value + 0.05) if as_float(value) > 0.0 else 0.0

    return 0.65 * saturate(mean) + 0.25 * saturate(cvar) + 0.10 * saturate(smooth_maximum)


def _trim_polyline(points: torch.Tensor, start_trim: Scalar, end_trim: Scalar) -> torch.Tensor:
    """Trim fixed arc-length windows from a flattened polyline's ends.

    Parameters
    ----------
    points : torch.Tensor
        Polyline vertices with shape ``[P, 2]``.
    start_trim, end_trim : float or torch.Tensor
        Nonnegative arc lengths removed from the source and target ends.

    Returns
    -------
    torch.Tensor
        Remaining polyline, or a repeated midpoint when the windows consume it.
    """

    lengths = torch.linalg.vector_norm(points[1:] - points[:-1], dim=1)
    total = keep(torch.sum(lengths))
    if as_float(start_trim + end_trim) >= as_float(total):
        midpoint = _point_at_arc_fraction(points, 0.5)
        return torch.stack((midpoint, midpoint))
    cumulative = torch.cat((torch.zeros(1, dtype=torch.float64), torch.cumsum(lengths, dim=0)))

    def at_distance(distance: Scalar) -> torch.Tensor:
        """Interpolate one point at an absolute arc distance.

        Parameters
        ----------
        distance : float or torch.Tensor
            Distance from the source terminal.

        Returns
        -------
        torch.Tensor
            Interpolated point with shape ``[2]``.
        """

        segment = int(
            torch.searchsorted(
                cumulative[1:], torch.tensor(as_float(distance), dtype=torch.float64), right=False
            )
        )
        local = (distance - keep(cumulative[segment])) / p_max(keep(lengths[segment]), 1e-300)
        return points[segment] + local * (points[segment + 1] - points[segment])

    start_distance = start_trim
    end_distance = total - end_trim
    retained = [at_distance(start_distance)]
    retained.extend(
        points[index]
        for index in range(1, points.shape[0] - 1)
        if as_float(start_distance) < float(cumulative[index]) < as_float(end_distance)
    )
    retained.append(at_distance(end_distance))
    return torch.stack(retained)


def _parallel_route_integral(
    left: torch.Tensor, right: torch.Tensor, intrinsic_unit: float
) -> Scalar:
    """Integrate U13's kernel against the nearest point on the other route.

    Parameters
    ----------
    left, right : torch.Tensor
        Flattened polylines with shape ``[P, 2]``.
    intrinsic_unit : float
        Positive scene intrinsic unit.

    Returns
    -------
    float or torch.Tensor
        Dimensionless exact piecewise-polynomial route-pair integral. The
        breakpoint partition and nearest-branch selection are decided on
        detached values; the piecewise integrand is continuous across every
        internal breakpoint, so the detached-partition gradient is a.e. exact.
    """

    total: Scalar = 0.0
    radius = 3.0 * intrinsic_unit
    for start_left, end_left in zip(left[:-1], left[1:]):
        vector_left = end_left - start_left
        length_left = keep(torch.linalg.vector_norm(vector_left))
        if as_float(length_left) == 0.0:
            continue
        unit_left = vector_left / length_left
        right_segments = [
            (start_right, end_right, end_right - start_right)
            for start_right, end_right in zip(right[:-1], right[1:])
            if float(torch.linalg.vector_norm(end_right - start_right)) > 0.0
        ]
        if not right_segments:
            continue
        base_breaks = [0.0, 1.0]
        for start_right, _, vector_right in right_segments:
            denominator = float(torch.dot(vector_right, vector_right))
            offset = start_left - start_right
            projection_constant = float(torch.dot(offset, vector_right)) / denominator
            projection_slope = float(torch.dot(vector_left, vector_right)) / denominator
            base_breaks.extend(_unit_interval_root(projection_constant, projection_slope))
            base_breaks.extend(_unit_interval_root(projection_constant - 1.0, projection_slope))
        ordered_base = sorted(set(base_breaks))
        for lower, upper in zip(ordered_base[:-1], ordered_base[1:]):
            if upper <= lower:
                continue
            midpoint = (lower + upper) / 2.0
            candidates: List[Tuple[Scalar, Scalar, Scalar, Scalar]] = []
            for start_right, end_right, vector_right in right_segments:
                coefficients = _segment_distance_polynomial(
                    start_left,
                    vector_left,
                    start_right,
                    end_right,
                    vector_right,
                    midpoint,
                )
                unit_right = vector_right / torch.linalg.vector_norm(vector_right)
                cosine = p_abs(keep(torch.dot(unit_left, unit_right)))
                candidates.append((*coefficients, cosine**4))
            envelope_breaks = [lower, upper]
            for left_index, left_candidate in enumerate(candidates):
                envelope_breaks.extend(
                    _quadratic_roots_in_interval(
                        left_candidate[0],
                        left_candidate[1],
                        left_candidate[2] - radius * radius,
                        lower,
                        upper,
                    )
                )
                for right_candidate in candidates[left_index + 1 :]:
                    envelope_breaks.extend(
                        _quadratic_roots_in_interval(
                            left_candidate[0] - right_candidate[0],
                            left_candidate[1] - right_candidate[1],
                            left_candidate[2] - right_candidate[2],
                            lower,
                            upper,
                        )
                    )
            ordered = sorted(set(envelope_breaks))
            for interval_left, interval_right in zip(ordered[:-1], ordered[1:]):
                if interval_right <= interval_left:
                    continue
                sample = (interval_left + interval_right) / 2.0
                candidate = min(
                    candidates,
                    key=lambda item: as_float(
                        item[0] * sample * sample + item[1] * sample + item[2]
                    ),
                )
                squared_distance = (
                    candidate[0] * sample * sample + candidate[1] * sample + candidate[2]
                )
                if as_float(squared_distance) >= radius * radius:
                    continue
                total += (
                    candidate[3]
                    * length_left
                    * _quartic_distance_kernel_integral(
                        candidate[0],
                        candidate[1],
                        candidate[2],
                        radius,
                        interval_left,
                        interval_right,
                    )
                    / intrinsic_unit
                )
    return total


def _segment_distance_polynomial(
    query_start: torch.Tensor,
    query_vector: torch.Tensor,
    segment_start: torch.Tensor,
    segment_end: torch.Tensor,
    segment_vector: torch.Tensor,
    sample_parameter: float,
) -> Tuple[Scalar, Scalar, Scalar]:
    """Return one point-to-segment squared-distance polynomial branch.

    Parameters
    ----------
    query_start, query_vector : torch.Tensor
        Query point path ``query_start + t*query_vector``.
    segment_start, segment_end, segment_vector : torch.Tensor
        Nonzero candidate segment geometry.
    sample_parameter : float
        Parameter selecting the constant, interior, or terminal projection branch.

    Returns
    -------
    tuple[float or torch.Tensor, ...]
        Coefficients ``(a, b, c)`` of squared distance ``a*t^2+b*t+c``.
    """

    denominator = keep(torch.dot(segment_vector, segment_vector))
    offset = query_start - segment_start
    projection = (
        keep(torch.dot(offset, segment_vector))
        + sample_parameter * keep(torch.dot(query_vector, segment_vector))
    ) / denominator
    if as_float(projection) <= 0.0:
        difference = query_start - segment_start
        return (
            keep(torch.dot(query_vector, query_vector)),
            2.0 * keep(torch.dot(difference, query_vector)),
            keep(torch.dot(difference, difference)),
        )
    if as_float(projection) >= 1.0:
        difference = query_start - segment_end
        return (
            keep(torch.dot(query_vector, query_vector)),
            2.0 * keep(torch.dot(difference, query_vector)),
            keep(torch.dot(difference, difference)),
        )
    offset_projection = keep(torch.dot(offset, segment_vector))
    vector_projection = keep(torch.dot(query_vector, segment_vector))
    return (
        keep(torch.dot(query_vector, query_vector)) - vector_projection**2 / denominator,
        2.0
        * (
            keep(torch.dot(offset, query_vector))
            - offset_projection * vector_projection / denominator
        ),
        keep(torch.dot(offset, offset)) - offset_projection**2 / denominator,
    )


def _quadratic_roots_in_interval(
    a: Scalar, b: Scalar, c: Scalar, lower: float, upper: float
) -> List[float]:
    """Return real roots strictly inside one parameter interval.

    Parameters
    ----------
    a, b, c : float or torch.Tensor
        Quadratic coefficients; breakpoints are decided on detached values.
    lower, upper : float
        Open interval bounds.

    Returns
    -------
    list[float]
        Sorted unique roots inside ``(lower, upper)``.
    """

    a = as_float(a)
    b = as_float(b)
    c = as_float(c)
    if a == 0.0:
        if b == 0.0:
            return []
        root = -c / b
        return [root] if lower < root < upper else []
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return []
    square_root = math.sqrt(max(0.0, discriminant))
    roots = ((-b - square_root) / (2.0 * a), (-b + square_root) / (2.0 * a))
    return sorted({root for root in roots if lower < root < upper})


def _quartic_distance_kernel_integral(
    a: Scalar,
    b: Scalar,
    c: Scalar,
    radius: float,
    lower: float,
    upper: float,
) -> Scalar:
    """Integrate ``(1-(a*t^2+b*t+c)/radius^2)^2`` exactly.

    Parameters
    ----------
    a, b, c : float or torch.Tensor
        Squared-distance polynomial coefficients.
    radius : float
        Positive compact-support distance.
    lower, upper : float
        Parameter interval lying inside the support.

    Returns
    -------
    float or torch.Tensor
        Exact nonnegative kernel integral; a live tensor inside a trace.
    """

    radius_squared = radius * radius
    radius_fourth = radius_squared * radius_squared
    coefficients = (
        1.0 - 2.0 * c / radius_squared + c * c / radius_fourth,
        -2.0 * b / radius_squared + 2.0 * b * c / radius_fourth,
        -2.0 * a / radius_squared + (b * b + 2.0 * a * c) / radius_fourth,
        2.0 * a * b / radius_fourth,
        a * a / radius_fourth,
    )

    def primitive(value: float) -> Scalar:
        """Evaluate the quartic antiderivative.

        Parameters
        ----------
        value : float
            Parameter value.

        Returns
        -------
        float or torch.Tensor
            Antiderivative value.
        """

        return p_sum(
            [
                coefficient * value ** (degree + 1) / (degree + 1)
                for degree, coefficient in enumerate(coefficients)
            ]
        )

    return p_max(0.0, primitive(upper) - primitive(lower))


def U15(scene: Scene) -> FacetResult:
    """Multi-edge / self-loop legibility. Frozen SHA-256: ba372b7b0425c2a202e8d76de7a16b88f9bb18ebb2a6220df0ecfc4089ad83c7."""

    multiplicities = Counter(
        tuple(sorted(edge)) for edge in scene.graph.edges if edge[0] != edge[1]
    )
    parallel_groups = [edge for edge, count in multiplicities.items() if count > 1]
    loops = [index for index, edge in enumerate(scene.graph.edges) if edge[0] == edge[1]]
    if not parallel_groups and not loops:
        return na_result("no_multiedges_or_selfloops")
    route_by_edge = {route.edge_index: route for route in resolved_routes(scene)}
    class_defects: List[Scalar] = []
    class_pair_defects: Dict[Tuple[int, int], Tuple[float, ...]] = {}
    for edge in parallel_groups:
        indices = [
            index
            for index, candidate in enumerate(scene.graph.edges)
            if tuple(sorted(candidate)) == edge
        ]
        pair_defects: List[Scalar] = []
        for left_position, left_index in enumerate(indices):
            for right_index in indices[left_position + 1 :]:
                left = route_by_edge.get(left_index)
                right = route_by_edge.get(right_index)
                if left is None or right is None:
                    continue
                left_length = keep(
                    torch.sum(torch.linalg.vector_norm(left.points[1:] - left.points[:-1], dim=1))
                )
                right_length = keep(
                    torch.sum(torch.linalg.vector_norm(right.points[1:] - right.points[:-1], dim=1))
                )
                maximum_length = p_max(left_length, right_length)
                shared_length = p_min(left_length, right_length)
                if as_float(maximum_length) == 0.0:
                    pair_defects.append(0.0)
                    continue
                left_measured = _trim_polyline(
                    left.points, 0.1 * shared_length, 0.1 * shared_length
                )
                right_measured = _trim_polyline(
                    right.points, 0.1 * shared_length, 0.1 * shared_length
                )
                separation = min(
                    [
                        _segment_segment_distance(left_start, left_end, right_start, right_end)
                        for left_start, left_end in zip(left_measured[:-1], left_measured[1:])
                        for right_start, right_end in zip(right_measured[:-1], right_measured[1:])
                    ],
                    key=as_float,
                )
                relative = separation / shared_length if as_float(shared_length) > 0.0 else 0.0
                defect = p_max(0.0, 1.0 - relative / 0.05) ** 2
                fade_shared = _smooth_fade(shared_length / (0.1 * maximum_length))
                fade_absolute = _smooth_fade(maximum_length / (0.1 * scene.intrinsic_unit))
                pair_defects.append(fade_shared * fade_absolute * defect)
        if pair_defects:
            class_defects.append(global_blend(pair_defects))
            class_pair_defects[edge] = tuple(_raw(defect) for defect in pair_defects)
    values: Dict[str, Scalar] = {}
    if class_defects:
        values["U15.i"] = global_blend(class_defects)
    loop_defects: List[Scalar] = []
    for loop_index in loops:
        route = route_by_edge.get(loop_index)
        if route is None:
            continue
        owner = scene.graph.edges[loop_index][0]
        obstacles = [box for box in scene.node_label_boxes if box.owner == owner]
        obstacles.extend(box for box in scene.node_boxes if box.owner != owner)
        obstacles.extend(box for box in scene.node_label_boxes if box.owner != owner)
        width = (
            scene.style.edge_stroke_widths[loop_index]
            if scene.style.edge_stroke_widths
            else scene.style.route_stroke_width * scene.style.coordinate_scale
        )
        integral: Scalar = 0.0
        for obstacle in obstacles:
            for start, end in zip(route.points[:-1], route.points[1:]):
                contribution, _ = _segment_box_deficit_integral(
                    start,
                    end,
                    obstacle,
                    width / 2.0,
                    scene.intrinsic_unit,
                )
                integral += contribution
        loop_defects.append(integral / (integral + 0.05) if as_float(integral) > 0.0 else 0.0)
    if loop_defects:
        values["U15.ii"] = global_blend(loop_defects)
    return mean_result(
        "U15",
        values,
        {
            "parallel_classes": class_pair_defects,
            "loop_defects": tuple(_raw(defect) for defect in loop_defects),
        },
    )


def _segment_segment_distance(
    start_left: torch.Tensor,
    end_left: torch.Tensor,
    start_right: torch.Tensor,
    end_right: torch.Tensor,
) -> Scalar:
    """Return the exact minimum distance between two planar line segments.

    Parameters
    ----------
    start_left, end_left, start_right, end_right : torch.Tensor
        Segment endpoints with shape ``[2]``.

    Returns
    -------
    float or torch.Tensor
        Nonnegative Euclidean distance; a live tensor inside a trace.
    """

    if proper_intersection(start_left, end_left, start_right, end_right):
        return 0.0
    return min(
        [
            _point_segment_distance(start_left, start_right, end_right),
            _point_segment_distance(end_left, start_right, end_right),
            _point_segment_distance(start_right, start_left, end_left),
            _point_segment_distance(end_right, start_left, end_left),
        ],
        key=as_float,
    )


def _point_segment_distance(point: torch.Tensor, start: torch.Tensor, end: torch.Tensor) -> Scalar:
    """Return exact Euclidean distance from a point to a line segment.

    Parameters
    ----------
    point, start, end : torch.Tensor
        Planar points with shape ``[2]``.

    Returns
    -------
    float or torch.Tensor
        Nonnegative distance.
    """

    direction = end - start
    denominator = keep(torch.dot(direction, direction))
    if as_float(denominator) == 0.0:
        return _norm_or_zero(point - start)
    parameter = p_min(1.0, p_max(0.0, keep(torch.dot(point - start, direction)) / denominator))
    return _norm_or_zero(point - (start + parameter * direction))


def _sample_polyline(points: torch.Tensor, count: int) -> torch.Tensor:
    """Sample a polyline at equally spaced arc-length fractions.

    Parameters
    ----------
    points : torch.Tensor
        Polyline vertices with shape ``[P, 2]``.
    count : int
        Number of samples, at least two.

    Returns
    -------
    torch.Tensor
        Sample points with shape ``[count, 2]``.
    """

    lengths = torch.linalg.vector_norm(points[1:] - points[:-1], dim=1)
    cumulative = torch.cat((torch.zeros(1, dtype=torch.float64), torch.cumsum(lengths, dim=0)))
    total = cumulative[-1]
    if float(total) == 0.0:
        return points[0].repeat(count, 1)
    targets = torch.linspace(0.0, float(total), count, dtype=torch.float64)
    segment = torch.searchsorted(cumulative[1:], targets, right=False)
    segment = torch.clamp(segment, max=points.shape[0] - 2)
    local = (targets - cumulative[segment]) / torch.clamp(lengths[segment], min=1e-12)
    return points[segment] + local[:, None] * (points[segment + 1] - points[segment])


def U16(scene: Scene) -> FacetResult:
    """Edge-label placement. Frozen SHA-256: 10e732d51189a4e9bf607044ce70d96042a3f3474fa172917467b043e49145a7."""

    if not scene.edge_label_boxes:
        return na_result("no_declared_edge_labels")
    overlap_defects: List[Scalar] = []
    ownership_defects: List[Scalar] = []
    overlap_sums: List[Scalar] = []
    ambiguity_values: List[Scalar] = []
    anchoring_values: List[Scalar] = []
    route_by_edge = {route.edge_index: route for route in resolved_routes(scene)}
    obstacle_boxes = (
        list(scene.edge_label_boxes) + list(scene.node_label_boxes) + list(scene.node_boxes)
    )
    for label in scene.edge_label_boxes:
        label_area = float(4.0 * torch.prod(label.half_extents))
        overlap_sum: Scalar = 0.0
        for obstacle in obstacle_boxes:
            if obstacle is label:
                continue
            _, overlap_fraction = aabb_pair(label, obstacle)
            obstacle_area = float(4.0 * torch.prod(obstacle.half_extents.detach()))
            overlap_area = overlap_fraction * min(label_area, obstacle_area)
            overlap_sum += overlap_area / min(label_area, obstacle_area)
        for edge_index, route in route_by_edge.items():
            width = (
                scene.style.edge_stroke_widths[edge_index]
                if scene.style.edge_stroke_widths
                else scene.style.route_stroke_width * scene.style.coordinate_scale
            )
            if edge_index == label.owner:
                label_height = 2.0 * float(label.half_extents[1])
                route_area = _route_box_ink_area(
                    route.points,
                    label,
                    width,
                    excluded_center=label.center,
                    excluded_radius=label_height,
                )
            else:
                route_area = _route_box_ink_area(route.points, label, width)
            overlap_sum += route_area / label_area
        overlap_sums.append(overlap_sum)
        overlap_defects.append(overlap_sum / (overlap_sum + 0.25))

        own_route = route_by_edge[label.owner]
        own_distance = _point_polyline_distance(label.center, own_route.points)
        foreign_candidates = [
            _point_polyline_distance(label.center, route.points)
            for edge_index, route in route_by_edge.items()
            if edge_index != label.owner
        ]
        foreign_distance: Scalar = (
            min(foreign_candidates, key=as_float) if foreign_candidates else math.inf
        )
        regularizer = 0.25 * scene.intrinsic_unit
        ratio = (
            (own_distance + regularizer) / (foreign_distance + regularizer)
            if math.isfinite(as_float(foreign_distance))
            else 0.0
        )
        ambiguity = ratio * ratio / (1.0 + ratio * ratio)
        label_height = 2.0 * float(label.half_extents[1])
        anchoring_ratio = p_max(0.0, own_distance - label_height) / (2.0 * scene.intrinsic_unit)
        anchoring = anchoring_ratio * anchoring_ratio / (1.0 + anchoring_ratio * anchoring_ratio)
        ownership = 1.0 - (1.0 - ambiguity) * (1.0 - anchoring)
        ambiguity_values.append(ambiguity)
        anchoring_values.append(anchoring)
        ownership_defects.append(ownership)
    values: Dict[str, Scalar] = {
        "U16.i": global_blend(overlap_defects),
        "U16.ii": global_blend(ownership_defects),
    }
    return mean_result(
        "U16",
        values,
        {
            "label_count": len(scene.edge_label_boxes),
            "overlap_sums": tuple(_raw(value) for value in overlap_sums),
            "ambiguity": tuple(_raw(value) for value in ambiguity_values),
            "anchoring": tuple(_raw(value) for value in anchoring_values),
        },
    )


def _point_polyline_distance(point: torch.Tensor, points: torch.Tensor) -> Scalar:
    """Return exact point-to-flattened-polyline distance.

    Parameters
    ----------
    point : torch.Tensor
        Query point with shape ``[2]``.
    points : torch.Tensor
        Polyline vertices with shape ``[P, 2]``.

    Returns
    -------
    float or torch.Tensor
        Minimum Euclidean distance; a live tensor inside a trace.
    """

    distance, _ = _point_polyline_projection(point, points)
    return distance


def _point_polyline_projection(
    point: torch.Tensor, points: torch.Tensor
) -> Tuple[Scalar, torch.Tensor]:
    """Return distance and nearest point on a flattened polyline.

    Parameters
    ----------
    point : torch.Tensor
        Query point with shape ``[2]``.
    points : torch.Tensor
        Polyline vertices with shape ``[P, 2]``.

    Returns
    -------
    tuple[float or torch.Tensor, torch.Tensor]
        Minimum distance and its canonical first nearest projection.
    """

    candidates: List[Tuple[Scalar, torch.Tensor]] = []
    for start, end in zip(points[:-1], points[1:]):
        direction = end - start
        denominator = keep(torch.dot(direction, direction))
        if as_float(denominator) == 0.0:
            candidates.append((_norm_or_zero(point - start), start))
            continue
        parameter = p_min(
            1.0,
            p_max(0.0, keep(torch.dot(point - start, direction)) / denominator),
        )
        projection = start + parameter * direction
        candidates.append((_norm_or_zero(point - projection), projection))
    return min(candidates, key=lambda item: as_float(item[0]))


def _route_box_ink_area(
    points: torch.Tensor,
    box: BoxGeometry,
    width: float,
    *,
    excluded_center: Optional[torch.Tensor] = None,
    excluded_radius: float = 0.0,
) -> Scalar:
    """Return flattened route-ribbon area whose centerline lies inside a box.

    Parameters
    ----------
    points : torch.Tensor
        Polyline vertices with shape ``[P, 2]``.
    box : BoxGeometry
        Axis-aligned label obstacle.
    width : float
        Positive ribbon width.
    excluded_center : torch.Tensor or None
        Optional anchor center whose local exemption disk is removed.
    excluded_radius : float
        Radius of the local anchor exemption disk.

    Returns
    -------
    float or torch.Tensor
        Centerline coverage times width, capped by the obstacle area. The
        clip window parameters flow live (they carry real boundary gradient);
        only degeneracy/emptiness branches read detached values.
    """

    lower = box.center - box.half_extents
    upper = box.center + box.half_extents
    length: Scalar = 0.0
    for start, end in zip(points[:-1], points[1:]):
        direction = end - start
        entry: Scalar = 0.0
        exit_: Scalar = 1.0
        for axis in range(2):
            delta = keep(direction[axis])
            if as_float(delta) == 0.0:
                if float(start[axis]) < float(lower[axis]) or float(start[axis]) > float(
                    upper[axis]
                ):
                    entry = 1.0
                    exit_ = 0.0
                    break
                continue
            first = (keep(lower[axis]) - keep(start[axis])) / delta
            second = (keep(upper[axis]) - keep(start[axis])) / delta
            entry = p_max(entry, p_min(first, second))
            exit_ = p_min(exit_, p_max(first, second))
            if as_float(entry) > as_float(exit_):
                break
        if as_float(entry) <= as_float(exit_):
            admitted = exit_ - entry
            if excluded_center is not None and excluded_radius > 0.0:
                offset = start - excluded_center
                quadratic_a = keep(torch.dot(direction, direction))
                quadratic_b = 2.0 * keep(torch.dot(offset, direction))
                quadratic_c = keep(torch.dot(offset, offset)) - excluded_radius**2
                discriminant = quadratic_b**2 - 4.0 * quadratic_a * quadratic_c
                if as_float(quadratic_a) > 0.0 and as_float(discriminant) >= 0.0:
                    root = 0.0 if as_float(discriminant) == 0.0 else p_sqrt(discriminant)
                    circle_entry = (-quadratic_b - root) / (2.0 * quadratic_a)
                    circle_exit = (-quadratic_b + root) / (2.0 * quadratic_a)
                    admitted -= p_max(
                        0.0,
                        p_min(exit_, circle_exit) - p_max(entry, circle_entry),
                    )
            length += p_max(0.0, admitted) * keep(torch.linalg.vector_norm(direction))
    return p_min(length * width, float(4.0 * torch.prod(box.half_extents.detach())))
