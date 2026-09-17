"""Declared cluster cohesion, separation, containment, hierarchy, and label facets."""

# Contract identity docstrings intentionally keep each title and digest on one line.
# ruff: noqa: E501

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

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
    tracing_active,
)
from dagua.eval.ruler_v4._util import (
    ALPHA_GRID,
    aabb_pair,
    blend_with_weights,
    compose_facet_rows,
    global_blend,
    mean_result,
    resolved_routes,
    selected_alpha_grid_offset,
    smoothstep,
    snap_unit,
    soft_pos,
)
from dagua.eval.ruler_v4.frames import RobustFrame, robust_frame
from dagua.eval.ruler_v4.legibility import _box_segment_clearance, _route_box_overlap_area
from dagua.eval.ruler_v4.scene import (
    BoxGeometry,
    FacetResult,
    Scene,
    invalid_result,
    na_result,
    value_result,
)

_U30_PAD_TARGET_U = 0.50


def _scalar_tensor(value: Scalar) -> torch.Tensor:
    """Promote one scalar to a float64 tensor, preserving any graph."""

    if isinstance(value, torch.Tensor):
        return value
    return torch.tensor(float(value), dtype=torch.float64)


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


def _p_prod(values: Sequence[Scalar]) -> Scalar:
    """Product; ``math.prod`` on floats, a left-to-right product with tensors."""

    items = list(values)
    if any(isinstance(item, torch.Tensor) for item in items):
        result: Scalar = 1.0
        for item in items:
            result = result * item
        return result
    return math.prod(items)


def _smooth_fade(value: Scalar) -> Scalar:
    """Evaluate the shared quintic-smoothstep fade on one float64 scalar.

    Parameters
    ----------
    value : float or torch.Tensor
        Fade argument in the gate's own units.

    Returns
    -------
    float or torch.Tensor
        ``float(smoothstep(tensor(value, float64)))`` on the exact path (the
        historical expression, bit-identical), or the live smoothstep tensor
        inside a trace.
    """

    if isinstance(value, torch.Tensor):
        return smoothstep(value)
    return float(smoothstep(torch.tensor(value, dtype=torch.float64)))


def _smooth_fade32(value: Scalar) -> Scalar:
    """Evaluate the quintic fade through the historical float32 construction.

    The U25/U26 fade sites build ``torch.tensor(x)`` without a dtype, so the
    historical value carries a float32 round-trip. The tensor branch mirrors
    that quantization differentiably (cast down, fade, cast back up), which
    preserves the float path's value bit-for-bit.

    Parameters
    ----------
    value : float or torch.Tensor
        Fade argument in the gate's own units.

    Returns
    -------
    float or torch.Tensor
        Fade with the float32 round-trip preserved on both branches.
    """

    if isinstance(value, torch.Tensor):
        return smoothstep(value.to(torch.float32)).to(torch.float64)
    return float(smoothstep(torch.tensor(value)))


def _norm_or_zero(vector: torch.Tensor) -> Scalar:
    """Keep a Euclidean norm, detaching only the exact-zero boundary.

    ``vector_norm`` has an undefined (NaN) gradient at the zero vector; every
    consumer reads the distance through an even or hinged kernel whose slope
    at exact coincidence is zero, so the detached 0.0 is the exact
    subgradient there.

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


def _row_norms_or_zero(matrix: torch.Tensor) -> torch.Tensor:
    """Return per-row Euclidean norms with the exact-zero rows detached.

    Parameters
    ----------
    matrix : torch.Tensor
        Difference vectors with shape ``[K, 2]``.

    Returns
    -------
    torch.Tensor
        Row norms with shape ``[K]``; exact-zero rows carry the constant
        zero (the exact subgradient of every even downstream kernel) so the
        live rows' backward stays NaN-free.
    """

    zero_rows = torch.linalg.vector_norm(matrix.detach(), dim=1) == 0.0
    if not bool(zero_rows.any()):
        return torch.linalg.vector_norm(matrix, dim=1)
    safe = torch.where(zero_rows.unsqueeze(1), torch.ones_like(matrix), matrix)
    return torch.where(
        zero_rows,
        torch.zeros((), dtype=matrix.dtype),
        torch.linalg.vector_norm(safe, dim=1),
    )


def _pow_or_zero(base: Scalar, exponent: float) -> Scalar:
    """Return ``base ** exponent`` with the saturated constant at zero.

    Parameters
    ----------
    base : float or torch.Tensor
        Nonnegative power base.
    exponent : float
        Positive frozen exponent below one.

    Returns
    -------
    float or torch.Tensor
        ``base ** exponent``. A traced base at exactly zero returns the
        constant zero, avoiding ``0 ** w``'s infinite backward; the value is
        identical.
    """

    if isinstance(base, torch.Tensor) and float(base.detach().item()) <= 0.0:
        return torch.zeros((), dtype=torch.float64)
    return base**exponent


def _interim_cluster_severity(
    defect: Scalar,
    overlap_area: Scalar,
    subject_area: float,
    *,
    subject_is_lower: bool,
) -> Scalar:
    """Apply contract-bounded z-order severity to an unresolved cluster pair.

    Parameters
    ----------
    defect : float or torch.Tensor
        Unadjusted intersecting-pair defect.
    overlap_area : float or torch.Tensor
        Pair overlap area in scene units squared.
    subject_area : float
        Area of the scored cluster label.
    subject_is_lower : bool
        Whether the scored label is below the obstacle in stored order.

    Returns
    -------
    float or torch.Tensor
        Effective burden between one-half and the unadjusted defect.

    Notes
    -----
    TODO(scheduler-owner): replace the node-label/cluster-label relative-order
    branch when the primitive-id grammar is frozen. The default class order and
    same-kind canonical order are already frozen.
    """

    if as_float(overlap_area) <= 0.0:
        return defect
    occluded = p_min(1.0, overlap_area / subject_area) if subject_is_lower else 0.0
    return defect * (0.5 + 0.5 * occluded)


def _clusters(scene: Scene, minimum_size: int = 1) -> Dict[str, Tuple[int, ...]]:
    """Return nonempty, canonical declared cluster memberships.

    Parameters
    ----------
    scene : Scene
        Validated clustered scene.
    minimum_size : int
        Smallest canonical member-set size to retain.

    Returns
    -------
    dict[str, tuple[int, ...]]
        Sorted unique member indices per cluster.
    """

    canonical: Dict[str, Tuple[int, ...]] = {}
    seen: Set[Tuple[int, ...]] = set()
    for name, members in sorted(scene.graph.clusters.items()):
        member_set = tuple(sorted(set(members)))
        if len(member_set) < minimum_size or member_set in seen:
            continue
        canonical[name] = member_set
        seen.add(member_set)
    return canonical


def _node_masses(scene: Scene) -> List[float]:
    """Return positive input-owned node masses.

    Parameters
    ----------
    scene : Scene
        Validated scene.

    Returns
    -------
    list[float]
        One mass per canonical node, defaulting to one.
    """

    if scene.graph.node_masses is None:
        return [1.0] * scene.node_count
    return [float(value) for value in scene.graph.node_masses]


def _induced_diameter(scene: Scene, members: Sequence[int]) -> int:
    """Return the maximum finite hop distance in an induced cluster graph.

    Parameters
    ----------
    scene : Scene
        Validated scene.
    members : sequence[int]
        Canonical cluster members.

    Returns
    -------
    int
        Maximum shortest-path distance, or zero when no induced edge exists.
    """

    member_set = set(members)
    neighbors: Dict[int, List[int]] = {node: [] for node in members}
    for source, target in scene.graph.edges:
        if source in member_set and target in member_set and source != target:
            neighbors[source].append(target)
            neighbors[target].append(source)
    diameter = 0
    for source in members:
        distance = {source: 0}
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for target in neighbors[node]:
                if target not in distance:
                    distance[target] = distance[node] + 1
                    queue.append(target)
        diameter = max(diameter, max(distance.values(), default=0))
    return diameter


def _lower_quantile(values: Sequence[Scalar], quantile: float) -> Scalar:
    """Return a deterministic lower empirical quantile.

    Parameters
    ----------
    values : sequence[float or torch.Tensor]
        Nonempty finite sample.
    quantile : float
        Quantile level in ``[0, 1]``.

    Returns
    -------
    float or torch.Tensor
        Lower-order-statistic quantile; a live tensor when any input is
        (the selection is an order statistic, a.e. differentiable).
    """

    items = list(values)
    if any(isinstance(item, torch.Tensor) for item in items):
        stacked = torch.stack([_scalar_tensor(item) for item in items])
        return torch.quantile(stacked, quantile, interpolation="lower")
    tensor = torch.tensor(items, dtype=torch.float64)
    return float(torch.quantile(tensor, quantile, interpolation="lower"))


def _degree_vector(scene: Scene) -> List[int]:
    """Return declared undirected multigraph degrees.

    Parameters
    ----------
    scene : Scene
        Validated scene.

    Returns
    -------
    list[int]
        Degree per canonical node, with self-loops counted twice.
    """

    degrees = [0] * scene.node_count
    for source, target in scene.graph.edges:
        degrees[source] += 1
        degrees[target] += 1
    return degrees


def _value_decile_buckets(values: Sequence[int]) -> List[int]:
    """Assign value-based decile buckets without splitting ties.

    Parameters
    ----------
    values : sequence[int]
        Input-owned scalar values.

    Returns
    -------
    list[int]
        Bucket ids in ``[1, 10]``.
    """

    cuts = [_lower_quantile(values, level / 10.0) for level in range(1, 10)]
    return [1 + sum(value > cut for cut in cuts) for value in values]


@dataclass(frozen=True)
class _ClusterRegion:
    """Derived union of member OBBs inflated by one spacing radius.

    Parameters
    ----------
    boxes : tuple[BoxGeometry, ...]
        Fixed member primitive boxes.
    radius : float or torch.Tensor
        Median local-spacing inflation radius; a live tensor inside a trace.
    bounds : BoxGeometry
        Axis-aligned broad-phase bounds of the union.
    """

    boxes: Tuple[BoxGeometry, ...]
    radius: Scalar
    bounds: BoxGeometry


def _cluster_spacing(scene: Scene, members: Sequence[int]) -> Scalar:
    """Return the cluster region's median k-nearest-member spacing.

    Parameters
    ----------
    scene : Scene
        Validated scene.
    members : sequence[int]
        Canonical member indices.

    Returns
    -------
    float or torch.Tensor
        Lower median of the ``ceil(sqrt(n))``-th neighbour distances.
    """

    neighbor_rank = min(len(members) - 1, math.ceil(math.sqrt(len(members))))
    scales: List[Scalar] = []
    for member in members:
        distances = sorted(
            (
                _norm_or_zero(scene.positions[member] - scene.positions[other])
                for other in members
                if other != member
            ),
            key=as_float,
        )
        scales.append(distances[max(0, neighbor_rank - 1)])
    return _lower_quantile(scales, 0.5)


def _regions(scene: Scene) -> Dict[str, _ClusterRegion]:
    """Derive the normative offset-OBB union for each canonical cluster.

    Parameters
    ----------
    scene : Scene
        Validated clustered scene.

    Returns
    -------
    dict[str, _ClusterRegion]
        Offset-union region per declared cluster.
    """

    regions: Dict[str, _ClusterRegion] = {}
    for name, members in _clusters(scene).items():
        radius = _cluster_spacing(scene, members) if len(members) >= 2 else 0.0
        boxes = tuple(scene.node_boxes[member] for member in members)
        lower = torch.stack([box.center - box.half_extents - radius for box in boxes]).amin(dim=0)
        upper = torch.stack([box.center + box.half_extents + radius for box in boxes]).amax(dim=0)
        bounds = BoxGeometry((lower + upper) / 2.0, (upper - lower) / 2.0, -1)
        regions[name] = _ClusterRegion(boxes, radius, bounds)
    return regions


def _cluster_robust_core_center(scene: Scene, members: Sequence[int]) -> torch.Tensor:
    """Return the U21 robust-core center for one declared cluster.

    Parameters
    ----------
    scene : Scene
        Validated clustered scene.
    members : sequence[int]
        Canonical cluster member indices.

    Returns
    -------
    torch.Tensor
        Robust center with shape ``[2]``.
    """

    return robust_frame(scene.positions[list(members)], scene.intrinsic_unit).center


def _frame_box(frame: RobustFrame, owner: int = -1) -> BoxGeometry:
    """Convert a robust region frame into box geometry.

    Parameters
    ----------
    frame : RobustFrame
        Robust region.
    owner : int
        Optional synthetic owner id.

    Returns
    -------
    BoxGeometry
        Axis-aligned region box.
    """

    return BoxGeometry(frame.center, frame.half_extents, owner)


def _signed_box_to_inflated_box(left: BoxGeometry, right: BoxGeometry, radius: Scalar) -> Scalar:
    """Return signed clearance from one AABB to a rounded inflated AABB.

    Parameters
    ----------
    left, right : BoxGeometry
        Query and region-member boxes.
    radius : float or torch.Tensor
        Minkowski-disc inflation radius of ``right``.

    Returns
    -------
    float or torch.Tensor
        Positive clearance, zero contact, or negative penetration depth.
    """

    delta = torch.abs(left.center - right.center) - (left.half_extents + right.half_extents)
    outside = _norm_or_zero(torch.clamp(delta, min=0.0))
    inside = p_min(p_max(keep(delta[0]), keep(delta[1])), 0.0)
    return outside + inside - radius


def _signed_box_region(box: BoxGeometry, region: _ClusterRegion) -> Scalar:
    """Return signed clearance from an AABB to an offset-union region.

    Parameters
    ----------
    box : BoxGeometry
        Query primitive box.
    region : _ClusterRegion
        Cluster offset-union.

    Returns
    -------
    float or torch.Tensor
        Minimum signed clearance to any union member.
    """

    return min(
        (_signed_box_to_inflated_box(box, member, region.radius) for member in region.boxes),
        key=as_float,
    )


def _segment_box_interval(
    start: torch.Tensor, end: torch.Tensor, center: torch.Tensor, half_extents: torch.Tensor
) -> Tuple[Scalar, Scalar] | None:
    """Clip a segment to an axis-aligned rectangle in parameter space.

    Parameters
    ----------
    start, end : torch.Tensor
        Segment endpoints with shape ``[2]``.
    center, half_extents : torch.Tensor
        Rectangle center and positive half extents.

    Returns
    -------
    tuple[float, float] or None
        Closed parameter interval within ``[0, 1]``, if nonempty; live
        endpoints inside a trace.
    """

    direction = end - start
    lower = center - half_extents
    upper = center + half_extents
    entry: Scalar = 0.0
    exit_: Scalar = 1.0
    for axis in range(2):
        delta = keep(direction[axis])
        if as_float(delta) == 0.0:
            if as_float(start[axis]) < as_float(lower[axis]) or as_float(start[axis]) > as_float(
                upper[axis]
            ):
                return None
            continue
        first = (keep(lower[axis]) - keep(start[axis])) / delta
        second = (keep(upper[axis]) - keep(start[axis])) / delta
        entry = p_max(entry, p_min(first, second))
        exit_ = p_min(exit_, p_max(first, second))
        if as_float(entry) > as_float(exit_):
            return None
    return entry, exit_


def _segment_circle_interval(
    start: torch.Tensor, end: torch.Tensor, center: torch.Tensor, radius: Scalar
) -> Tuple[Scalar, Scalar] | None:
    """Clip a segment to a circle in parameter space.

    Parameters
    ----------
    start, end : torch.Tensor
        Segment endpoints with shape ``[2]``.
    center : torch.Tensor
        Circle center.
    radius : float or torch.Tensor
        Nonnegative circle radius.

    Returns
    -------
    tuple[float, float] or None
        Closed parameter interval within ``[0, 1]``, if nonempty; live
        endpoints inside a trace.
    """

    direction = end - start
    offset = start - center
    a = keep(torch.dot(direction, direction))
    if as_float(a) == 0.0:
        return (0.0, 1.0) if as_float(torch.dot(offset, offset)) <= as_float(radius) ** 2 else None
    b = 2.0 * keep(torch.dot(offset, direction))
    c = keep(torch.dot(offset, offset)) - radius * radius
    discriminant = b * b - 4.0 * a * c
    if as_float(discriminant) < 0.0:
        return None
    if as_float(discriminant) == 0.0:
        # sqrt backward at exact tangency is infinite; the historical value
        # there is exactly zero, whose downstream slope is what both interval
        # endpoints share, so the detached constant is the honest arm.
        root: Scalar = 0.0
    else:
        root = p_sqrt(p_max(0.0, discriminant))
    entry = p_max(0.0, (-b - root) / (2.0 * a))
    exit_ = p_min(1.0, (-b + root) / (2.0 * a))
    return (entry, exit_) if as_float(entry) <= as_float(exit_) else None


def _merge_interval_measure(intervals: Sequence[Tuple[Scalar, Scalar]]) -> Scalar:
    """Return the measure of a union of intervals in ``[0, 1]``.

    Parameters
    ----------
    intervals : sequence[tuple[float, float]]
        Closed parameter intervals.

    Returns
    -------
    float or torch.Tensor
        Union length in parameter units.
    """

    if not intervals:
        return 0.0
    ordered = sorted(intervals, key=lambda item: (as_float(item[0]), as_float(item[1])))
    total: Scalar = 0.0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if as_float(next_start) <= as_float(end):
            end = p_max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _inflated_half_extents(half_extents: torch.Tensor, radius: Scalar, axis: int) -> torch.Tensor:
    """Inflate one box's half extents by the region radius along one axis.

    Parameters
    ----------
    half_extents : torch.Tensor
        Positive half extents with shape ``[2]``.
    radius : float or torch.Tensor
        Offset-union inflation radius.
    axis : int
        Inflated axis, zero or one.

    Returns
    -------
    torch.Tensor
        Inflated half extents; the float branch keeps the historical
        constant-tensor construction, the tensor branch stacks the live
        radius into the same values.
    """

    if isinstance(radius, torch.Tensor):
        offset = (radius, torch.zeros((), dtype=torch.float64))
        return half_extents + torch.stack(offset if axis == 0 else offset[::-1])
    values = [radius, 0.0] if axis == 0 else [0.0, radius]
    return half_extents + torch.tensor(values, dtype=torch.float64)


def _segment_region_fraction(
    start: torch.Tensor, end: torch.Tensor, region: _ClusterRegion
) -> Scalar:
    """Return the exact arc fraction of one segment inside an offset-OBB union.

    Parameters
    ----------
    start, end : torch.Tensor
        Segment endpoints with shape ``[2]``.
    region : _ClusterRegion
        Offset-union cluster region.

    Returns
    -------
    float or torch.Tensor
        Fraction of the segment covered by the union.
    """

    intervals: List[Tuple[Scalar, Scalar]] = []
    radius = region.radius
    for box in region.boxes:
        horizontal = _segment_box_interval(
            start,
            end,
            box.center,
            _inflated_half_extents(box.half_extents, radius, 0),
        )
        vertical = _segment_box_interval(
            start,
            end,
            box.center,
            _inflated_half_extents(box.half_extents, radius, 1),
        )
        if horizontal is not None:
            intervals.append(horizontal)
        if vertical is not None:
            intervals.append(vertical)
        for x_sign in (-1.0, 1.0):
            for y_sign in (-1.0, 1.0):
                corner = box.center + box.half_extents * torch.tensor(
                    [x_sign, y_sign], dtype=torch.float64
                )
                interval = _segment_circle_interval(start, end, corner, radius)
                if interval is not None:
                    intervals.append(interval)
    return _merge_interval_measure(intervals)


def _merge_angular_intervals(
    intervals: Sequence[Tuple[Scalar, Scalar]], full_turn: float
) -> List[Tuple[Scalar, Scalar]]:
    """Merge angular intervals already split onto one canonical turn.

    Parameters
    ----------
    intervals : sequence[tuple[float, float]]
        Intervals contained in ``[0, full_turn]``.
    full_turn : float
        Canonical turn length, normally ``2*pi``.

    Returns
    -------
    list[tuple[float, float]]
        Sorted disjoint covered intervals; live endpoints inside a trace.
    """

    if not intervals:
        return []
    ordered = sorted(
        ((p_max(0.0, left), p_min(full_turn, right)) for left, right in intervals),
        key=lambda item: (as_float(item[0]), as_float(item[1])),
    )
    merged = [ordered[0]]
    for left, right in ordered[1:]:
        previous_left, previous_right = merged[-1]
        if as_float(left) <= as_float(previous_right):
            merged[-1] = (previous_left, p_max(previous_right, right))
        else:
            merged.append((left, right))
    return merged


def _equal_disc_union_area_perimeter(
    centers: torch.Tensor, radius: Scalar
) -> Tuple[Scalar, Scalar]:
    """Compute exact area and perimeter of an equal-disc union from exposed arcs.

    Parameters
    ----------
    centers : torch.Tensor
        Disc centers with shape ``[K, 2]`` and float64 coordinates.
    radius : float or torch.Tensor
        Shared positive disc radius.

    Returns
    -------
    tuple[float, float]
        Analytic union area and perimeter; live tensors inside a trace.
    """

    full_turn = 2.0 * math.pi
    area_integral: Scalar = 0.0
    perimeter: Scalar = 0.0
    for index, center in enumerate(centers):
        covered: List[Tuple[Scalar, Scalar]] = []
        duplicate_covered = False
        for other_index, other in enumerate(centers):
            if other_index == index:
                continue
            delta = other - center
            distance = _norm_or_zero(delta)
            if as_float(distance) == 0.0:
                if other_index < index:
                    duplicate_covered = True
                    break
                continue
            if as_float(distance) >= 2.0 * as_float(radius):
                continue
            angle = _p_mod(_p_atan2(keep(delta[1]), keep(delta[0])), full_turn)
            cosine = p_min(1.0, distance / (2.0 * radius))
            if as_float(cosine) >= 1.0:
                # acos has an infinite slope at the clamp boundary; the
                # historical float value there is exactly acos(1) = 0.
                half_width: Scalar = math.acos(1.0)
            else:
                half_width = _p_acos(cosine)
            left = angle - half_width
            right = angle + half_width
            if as_float(left) < 0.0:
                covered.extend(((left + full_turn, full_turn), (0.0, right)))
            elif as_float(right) > full_turn:
                covered.extend(((left, full_turn), (0.0, right - full_turn)))
            else:
                covered.append((left, right))
        if duplicate_covered:
            continue
        merged = _merge_angular_intervals(covered, full_turn)
        exposed: List[Tuple[Scalar, Scalar]] = []
        cursor: Scalar = 0.0
        for left, right in merged:
            if as_float(cursor) < as_float(left):
                exposed.append((cursor, left))
            cursor = p_max(cursor, right)
        if as_float(cursor) < full_turn:
            exposed.append((cursor, full_turn))
        center_x = keep(center[0])
        center_y = keep(center[1])
        for left, right in exposed:
            width = right - left
            perimeter += radius * width
            area_integral += 0.5 * (
                radius * radius * width
                + radius
                * (
                    center_x * (_p_sin(right) - _p_sin(left))
                    - center_y * (_p_cos(right) - _p_cos(left))
                )
            )
    return p_abs(area_integral), perimeter


def _region_contains_points(region: _ClusterRegion, points: torch.Tensor) -> torch.Tensor:
    """Test points against an offset-OBB union.

    Parameters
    ----------
    region : _ClusterRegion
        Cluster offset-union.
    points : torch.Tensor
        Query points with shape ``[P, 2]``.

    Returns
    -------
    torch.Tensor
        Boolean membership mask with shape ``[P]``.
    """

    contained = torch.zeros(points.shape[0], dtype=torch.bool)
    for box in region.boxes:
        delta = torch.abs(points - box.center) - box.half_extents
        signed = torch.linalg.vector_norm(torch.clamp(delta, min=0.0), dim=1) + torch.minimum(
            torch.maximum(delta[:, 0], delta[:, 1]), torch.zeros_like(delta[:, 0])
        )
        contained |= signed <= region.radius
    return contained


def _region_vertical_intervals(
    region: _ClusterRegion, x_value: float
) -> List[Tuple[Scalar, Scalar]]:
    """Return merged vertical sections of one rounded-box union.

    Parameters
    ----------
    region : _ClusterRegion
        Union of equally offset axis-aligned boxes.
    x_value : float
        Vertical slice coordinate.

    Returns
    -------
    list[tuple[float, float]]
        Disjoint closed y intervals in increasing order; live endpoints
        inside a trace.
    """

    intervals: List[Tuple[Scalar, Scalar]] = []
    radius = region.radius
    for box in region.boxes:
        horizontal_excess = p_max(
            p_abs(x_value - keep(box.center[0])) - float(box.half_extents[0]),
            0.0,
        )
        if as_float(horizontal_excess) > as_float(radius):
            continue
        squared = p_max(0.0, radius * radius - horizontal_excess**2)
        if as_float(squared) == 0.0:
            # sqrt backward at exact tangency is infinite; the historical
            # float value there is exactly zero.
            extension: Scalar = 0.0
        else:
            extension = p_sqrt(squared)
        intervals.append(
            (
                keep(box.center[1] - box.half_extents[1]) - extension,
                keep(box.center[1] + box.half_extents[1]) + extension,
            )
        )
    intervals.sort(key=lambda item: (as_float(item[0]), as_float(item[1])))
    merged: List[Tuple[Scalar, Scalar]] = []
    for lower, upper in intervals:
        if not merged or as_float(lower) > as_float(merged[-1][1]):
            merged.append((lower, upper))
        else:
            previous_lower, previous_upper = merged[-1]
            merged[-1] = (previous_lower, p_max(previous_upper, upper))
    return merged


def _region_top_at_x(region: _ClusterRegion, x_value: float) -> Scalar:
    """Return the upper boundary of a rounded-box union at one x-coordinate.

    Parameters
    ----------
    region : _ClusterRegion
        Cluster offset-union.
    x_value : float
        Horizontal coordinate of the placement anchor.

    Returns
    -------
    float or torch.Tensor
        Highest region-boundary ordinate at the requested coordinate. When
        the vertical line misses the union (a cluster drawn as separated
        lumps whose robust-core center falls in the gap), the boundary at
        the nearest covered x is used, else the top of the region bounds.
        Both fallbacks are input-only: they read the same derived region
        every caller already holds, never the drawing being judged.
    """

    intervals = _region_vertical_intervals(region, x_value)
    if not intervals and region.boxes:
        candidates = []
        for box in region.boxes:
            lower = as_float(box.center[0] - box.half_extents[0]) - as_float(region.radius)
            upper = as_float(box.center[0] + box.half_extents[0]) + as_float(region.radius)
            candidates.append(min(max(x_value, lower), upper))
        nearest = min(candidates, key=lambda value: abs(value - x_value))
        intervals = _region_vertical_intervals(region, nearest)
    if not intervals:
        return keep(region.bounds.center[1] + region.bounds.half_extents[1])
    return max((upper for _, upper in intervals), key=as_float)


def _signed_top_padding(box: BoxGeometry, region: _ClusterRegion) -> Scalar:
    """Measure signed inward padding from a label's near edge to the region top.

    Parameters
    ----------
    box : BoxGeometry
        Derived cluster-label box.
    region : _ClusterRegion
        Owning cluster region.

    Returns
    -------
    float or torch.Tensor
        Positive inset inside the boundary, zero at contact, and negative outside.
    """

    boundary = _region_top_at_x(region, float(box.center[0]))
    near_edge = float(box.center[1] + box.half_extents[1])
    return boundary - near_edge


def _interval_measure(intervals: Sequence[Tuple[Scalar, Scalar]]) -> Scalar:
    """Return total length of disjoint intervals.

    Parameters
    ----------
    intervals : sequence[tuple[float, float]]
        Disjoint intervals.

    Returns
    -------
    float or torch.Tensor
        Nonnegative total length.
    """

    return p_sum([upper - lower for lower, upper in intervals])


def _interval_intersection_measure(
    left: Sequence[Tuple[Scalar, Scalar]], right: Sequence[Tuple[Scalar, Scalar]]
) -> Scalar:
    """Return the length of the intersection of two interval unions.

    Parameters
    ----------
    left, right : sequence[tuple[float, float]]
        Increasing disjoint interval lists.

    Returns
    -------
    float or torch.Tensor
        Nonnegative intersection length.
    """

    left_index = 0
    right_index = 0
    total: Scalar = 0.0
    while left_index < len(left) and right_index < len(right):
        lower = p_max(left[left_index][0], right[right_index][0])
        upper = p_min(left[left_index][1], right[right_index][1])
        total += p_max(0.0, upper - lower)
        if as_float(left[left_index][1]) < as_float(right[right_index][1]):
            left_index += 1
        else:
            right_index += 1
    return total


def _adaptive_simpson(
    function: Callable[[float], Scalar],
    left: float,
    right: float,
    tolerance: float,
    depth: int = 20,
) -> Scalar:
    """Integrate one continuous scalar function by deterministic adaptive Simpson.

    The subdivision schedule (panel bounds and refinement decisions) is
    decided on detached values, exactly as the float path decides it; the
    integrand values flow live, so the quadrature-weighted combination
    carries the a.e.-exact gradient of the same closed form.

    Parameters
    ----------
    function : callable[[float], float or torch.Tensor]
        Continuous integrand.
    left, right : float
        Finite integration bounds.
    tolerance : float
        Positive absolute error target for this interval.
    depth : int
        Maximum subdivision depth.

    Returns
    -------
    float or torch.Tensor
        Deterministic integral estimate.
    """

    midpoint = (left + right) / 2.0
    left_value = function(left)
    midpoint_value = function(midpoint)
    right_value = function(right)
    whole = (right - left) * (left_value + 4.0 * midpoint_value + right_value) / 6.0

    def refine(
        lower: float,
        upper: float,
        lower_value: Scalar,
        center_value: Scalar,
        upper_value: Scalar,
        estimate: Scalar,
        local_tolerance: float,
        remaining_depth: int,
    ) -> Scalar:
        """Recursively refine one Simpson panel.

        Parameters
        ----------
        lower, upper : float
            Panel bounds.
        lower_value, center_value, upper_value : float or torch.Tensor
            Cached integrand values.
        estimate : float or torch.Tensor
            Parent Simpson estimate.
        local_tolerance : float
            Panel absolute error target.
        remaining_depth : int
            Remaining subdivision budget.

        Returns
        -------
        float or torch.Tensor
            Refined panel integral.
        """

        center = (lower + upper) / 2.0
        left_center = (lower + center) / 2.0
        right_center = (center + upper) / 2.0
        left_center_value = function(left_center)
        right_center_value = function(right_center)
        left_estimate = (
            (center - lower) * (lower_value + 4.0 * left_center_value + center_value) / 6.0
        )
        right_estimate = (
            (upper - center) * (center_value + 4.0 * right_center_value + upper_value) / 6.0
        )
        combined = left_estimate + right_estimate
        if remaining_depth == 0 or as_float(p_abs(combined - estimate)) <= 15.0 * local_tolerance:
            return combined + (combined - estimate) / 15.0
        return refine(
            lower,
            center,
            lower_value,
            left_center_value,
            center_value,
            left_estimate,
            local_tolerance / 2.0,
            remaining_depth - 1,
        ) + refine(
            center,
            upper,
            center_value,
            right_center_value,
            upper_value,
            right_estimate,
            local_tolerance / 2.0,
            remaining_depth - 1,
        )

    return refine(
        left,
        right,
        left_value,
        midpoint_value,
        right_value,
        whole,
        tolerance,
        depth,
    )


def _region_relation_area(
    left: _ClusterRegion, right: _ClusterRegion | None = None
) -> Tuple[Scalar, Scalar]:
    """Measure rounded-box union area and optional intersection continuously.

    Parameters
    ----------
    left : _ClusterRegion
        Primary rounded-box union.
    right : _ClusterRegion or None
        Optional second rounded-box union.

    Returns
    -------
    tuple[float, float]
        Primary area and intersection area; the second value is zero when absent.

    Notes
    -----
    Exact rounded-box vertical sections are integrated between every arc/flat transition.
    Adaptive Simpson refinement targets relative area error below ``1e-10`` without a
    score-visible raster grid. Breakpoints and refinement are detached decisions;
    the integrated section lengths flow live.
    """

    regions = (left,) if right is None else (left, right)
    breakpoints = sorted(
        {
            float(box.center[0]) + offset
            for region in regions
            for box in region.boxes
            for offset in (
                -float(box.half_extents[0]) - as_float(region.radius),
                -float(box.half_extents[0]),
                float(box.half_extents[0]),
                float(box.half_extents[0]) + as_float(region.radius),
            )
        }
    )
    scale = max(
        1.0,
        float(4.0 * torch.prod(left.bounds.half_extents)),
    )
    interval_tolerance = 1e-11 * scale / max(1, len(breakpoints) - 1)

    def left_length(x_value: float) -> Scalar:
        """Return the primary region's vertical union length.

        Parameters
        ----------
        x_value : float
            Slice coordinate.

        Returns
        -------
        float or torch.Tensor
            Vertical union length.
        """

        return _interval_measure(_region_vertical_intervals(left, x_value))

    area = p_sum(
        [
            _adaptive_simpson(left_length, lower, upper, interval_tolerance)
            for lower, upper in zip(breakpoints[:-1], breakpoints[1:])
            if upper > lower
        ]
    )
    if right is None:
        return area, 0.0

    def overlap_length(x_value: float) -> Scalar:
        """Return the two regions' vertical intersection length.

        Parameters
        ----------
        x_value : float
            Slice coordinate.

        Returns
        -------
        float or torch.Tensor
            Vertical intersection length.
        """

        return _interval_intersection_measure(
            _region_vertical_intervals(left, x_value),
            _region_vertical_intervals(right, x_value),
        )

    overlap = p_sum(
        [
            _adaptive_simpson(overlap_length, lower, upper, interval_tolerance)
            for lower, upper in zip(breakpoints[:-1], breakpoints[1:])
            if upper > lower
        ]
    )
    return area, p_min(area, p_max(0.0, overlap))


def _region_depth_outside(child: _ClusterRegion, parent: _ClusterRegion) -> Scalar:
    """Estimate one-sided Hausdorff depth of a child region outside its parent.

    Parameters
    ----------
    child, parent : _ClusterRegion
        Child and parent offset-union regions.

    Returns
    -------
    float or torch.Tensor
        Maximum positive signed distance among exact rounded-box extremal candidates.
    """

    candidates: List[torch.Tensor] = []
    directions = torch.tensor(
        ((-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)), dtype=torch.float64
    )
    corners = torch.tensor(
        ((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)), dtype=torch.float64
    )
    for box in child.boxes:
        candidates.extend(box.center + corners * box.half_extents)
        candidates.extend(box.center + directions * (box.half_extents + child.radius))
    maximum: Scalar = 0.0
    for point in candidates:
        query = BoxGeometry(point, torch.zeros(2, dtype=torch.float64), -1)
        maximum = p_max(maximum, _signed_box_region(query, parent))
    return p_max(0.0, maximum)


def _cluster_budget(scene: Scene, members: Sequence[int]) -> float:
    """Return U27's input-only cluster achievability budget.

    Parameters
    ----------
    scene : Scene
        Validated scene.
    members : sequence[int]
        Cluster members.

    Returns
    -------
    float
        Positive clearance budget in scene units.
    """

    diagonals = [
        float(2.0 * torch.linalg.vector_norm(scene.node_boxes[member].half_extents))
        for member in members
    ]
    input_radius = max(diagonals)
    raw = (2.0 * math.pi * 1.5 * (input_radius + max(diagonals) / 2.0) - sum(diagonals)) / len(
        members
    )
    return max(raw, 0.125 * scene.intrinsic_unit)


def _alpha_grid_for_budget(scene: Scene, budget: float) -> Tuple[float, ...]:
    """Evaluate the twelve shared U17 absolute-term weights for a budget.

    Parameters
    ----------
    scene : Scene
        Validated scene providing the intrinsic unit.
    budget : float
        Positive local achievability budget.

    Returns
    -------
    tuple[float, ...]
        Twelve grid-row weights in normative order.
    """

    floor = 0.25 * scene.intrinsic_unit
    argument = (budget - floor) / (0.5 * floor)
    blend = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, argument))))
    # No snap here: with alpha_clear in (0, 0.5] and alpha_high in [0, 1]
    # the blend lies strictly inside (0, 0.5], and the grid weights are
    # not a guarded [0, 1] quantity in the first place.
    return tuple(
        alpha_clear * (1.0 - blend) + alpha_clear * blend * alpha_high
        for _, alpha_clear, alpha_high in ALPHA_GRID
    )


def U25(scene: Scene) -> FacetResult:
    """Cluster cohesion / compactness. Frozen SHA-256: 1abaa433699c51d9c1d2ed8c7bde4643df3d8eb46fe9133d465b434e40f15299."""

    if not scene.graph.clusters:
        return na_result("no_declared_clusters")
    clusters = _clusters(scene, 3)
    if not clusters:
        return na_result("clusters_too_small", {"declared_cluster_count": len(_clusters(scene))})

    frame = robust_frame(scene.positions, scene.intrinsic_unit)
    core_diagonal = 2.0 * keep(torch.linalg.vector_norm(frame.half_extents))
    masses = _node_masses(scene)
    defects: List[Scalar] = []
    cluster_masses: List[float] = []
    statistics: Dict[str, object] = {"D_core": as_float(core_diagonal), "clusters": {}}
    for name, members in clusters.items():
        points = scene.positions[list(members)]
        center = torch.median(points, dim=0).values
        radii = _row_norms_or_zero(points - center)
        radius = keep(torch.median(radii))
        radius_q90 = float(torch.quantile(radii, 0.9, interpolation="lower"))
        fraction = len(members) / scene.node_count
        diameter = _induced_diameter(scene, members)
        elongation = 1.0 + 0.5 * max(0.0, diameter / math.sqrt(len(members)) - 1.0)
        reference = 0.5 * math.sqrt(fraction) * elongation
        ratio = radius / p_max(core_diagonal, torch.finfo(torch.float64).tiny)
        if as_float(ratio) <= 0.0:
            median_defect: Scalar = 0.0
        else:
            excess = soft_pos(p_log(ratio / (2.0 * reference)))
            median_defect = excess / (1.0 + excess)

        member_defects: List[Scalar] = []
        member_masses: List[float] = []
        denominator = 2.0 * reference * core_diagonal
        member_radii = list(radii) if tracing_active() else radii.tolist()
        for member, member_radius in zip(members, member_radii):
            if as_float(member_radius) <= 0.0:
                member_excess: Scalar = 0.0
            else:
                member_excess = soft_pos(p_log(member_radius / denominator))
            member_defects.append(1.0 - _smooth_fade32(1.0 / (1.0 + member_excess)))
            member_masses.append(masses[member])
        tail = global_blend(member_defects, member_masses)
        full_defect = 1.0 - _pow_or_zero(1.0 - median_defect, 0.7) * _pow_or_zero(1.0 - tail, 0.3)
        defects.append(full_defect)
        cluster_masses.append(sum(masses[member] for member in members))
        statistics["clusters"][name] = {
            "median_center": center.tolist(),
            "R_c": as_float(radius),
            "R_c_q90": radius_q90,
            "rho_c": as_float(ratio),
            "f_c": fraction,
            "diameter": diameter,
            "rho_ref": reference,
            "member_tail": as_float(tail),
            "defect": as_float(full_defect),
        }
    defect = global_blend(defects, cluster_masses)
    statistics["cluster_count"] = len(defects)
    return value_result(defect, {"U25.headline": defect}, statistics)


def U26(scene: Scene) -> FacetResult:
    """Cluster separation (+ community faithfulness). Frozen SHA-256: bba49ea4d943942ee7510d7e16651470a63a488a990394d5b3ec8e1b09311508."""

    if not scene.graph.clusters:
        return na_result("no_declared_clusters")
    clusters = _clusters(scene, 3)
    if len(clusters) < 2:
        return na_result("insufficient_declared_clusters")

    frame = robust_frame(scene.positions, scene.intrinsic_unit)
    core_diagonal = 2.0 * keep(torch.linalg.vector_norm(frame.half_extents))
    masses = _node_masses(scene)
    incidence_defects: List[Scalar] = []
    incidence_masses: List[float] = []
    incidence_by_node_cluster: Dict[Tuple[int, str], Scalar] = {}
    memberships: Dict[int, Set[str]] = {node: set() for node in range(scene.node_count)}
    for name, members in clusters.items():
        member_set = set(members)
        fraction = len(members) / scene.node_count
        target = 0.15 * math.sqrt(fraction)
        foreign = [node for node in range(scene.node_count) if node not in member_set]
        for node in members:
            memberships[node].add(name)
            own_distances = [
                _norm_or_zero(scene.positions[node] - scene.positions[other])
                for other in members
                if other != node
            ]
            own_q = _lower_quantile(own_distances, 0.75)
            foreign_distance = min(
                (
                    _norm_or_zero(scene.positions[node] - scene.positions[other])
                    for other in foreign
                ),
                default=core_diagonal,
                key=as_float,
            )
            margin = (foreign_distance - own_q) / p_max(core_diagonal, 1e-12)
            defect = 1.0 - _smooth_fade32((margin + target) / (2.0 * target))
            incidence_defects.append(defect)
            incidence_masses.append(masses[node])
            incidence_by_node_cluster[(node, name)] = defect
    values: Dict[str, Scalar] = {
        "U26.i": global_blend(incidence_defects, incidence_masses),
    }

    degrees = _degree_vector(scene)
    degree_buckets = _value_decile_buckets(degrees)
    cluster_names = list(clusters)
    size_buckets_list = _value_decile_buckets([len(clusters[name]) for name in cluster_names])
    size_buckets = dict(zip(cluster_names, size_buckets_list))

    def governing(node: int) -> str:
        """Return the smallest eligible governing cluster for one node.

        Parameters
        ----------
        node : int
            Canonical node index.

        Returns
        -------
        str
            Canonical governing cluster id.
        """

        return min(memberships[node], key=lambda name: (len(clusters[name]), name))

    strata: Dict[Tuple[Tuple[int, int], Tuple[int, int]], Dict[str, List[Scalar]]] = {}
    clustered = [node for node, names in memberships.items() if names]
    for offset, left in enumerate(clustered):
        for right in clustered[offset + 1 :]:
            common_eligible = memberships[left] & memberships[right]
            common_any = {
                name
                for name, members in scene.graph.clusters.items()
                if left in members and right in members
            }
            if common_eligible:
                common_name = min(common_eligible, key=lambda name: (len(clusters[name]), name))
                size_pair = (size_buckets[common_name], size_buckets[common_name])
                arm = "same"
                fractions = (len(clusters[common_name]) / scene.node_count,) * 2
            elif common_any:
                continue
            else:
                left_name = governing(left)
                right_name = governing(right)
                size_pair = tuple(sorted((size_buckets[left_name], size_buckets[right_name])))
                arm = "cross"
                fractions = (
                    len(clusters[left_name]) / scene.node_count,
                    len(clusters[right_name]) / scene.node_count,
                )
            key = (tuple(sorted((degree_buckets[left], degree_buckets[right]))), size_pair)
            record = strata.setdefault(key, {"same": [], "cross": [], "fractions": []})
            distance = _norm_or_zero(scene.positions[left] - scene.positions[right]) / p_max(
                core_diagonal, 1e-12
            )
            record[arm].append(distance)
            record["fractions"].append(sum(fractions) / 2.0)
    control_defects: List[Scalar] = []
    control_weights: List[float] = []
    for record in strata.values():
        same = record["same"]
        cross = record["cross"]
        if len(same) < 20 or len(cross) < 20:
            continue
        contrast = _lower_quantile(cross, 0.5) - _lower_quantile(same, 0.5)
        fraction_bar = sum(as_float(item) for item in record["fractions"]) / len(
            record["fractions"]
        )
        control_defects.append(
            1.0 - _smooth_fade32(contrast / (0.10 * math.sqrt(max(fraction_bar, 1e-12))))
        )
        control_weights.append(float(len(same) + len(cross)))
    if control_defects:
        weight_total = sum(control_weights)
        values["U26.ii"] = p_sum(
            [
                defect * weight / weight_total
                for defect, weight in zip(control_defects, control_weights)
            ]
        )

    boundary_defects: List[Scalar] = []
    boundary_masses: List[float] = []
    for node, name in incidence_by_node_cluster:
        member_set = set(clusters[name])
        if any(
            (source == node and target not in member_set)
            or (target == node and source not in member_set)
            for source, target in scene.graph.edges
        ):
            boundary_defects.append(incidence_by_node_cluster[(node, name)])
            boundary_masses.append(masses[node])
    if boundary_defects:
        values["U26.iii"] = blend_with_weights(
            boundary_defects,
            boundary_masses,
            (0.0, 0.25 / 0.35, 0.10 / 0.35),
        )

    return mean_result(
        "U26",
        values,
        {
            "D_core": as_float(core_diagonal),
            "incidence_count": len(incidence_defects),
            "matched_stratum_count": len(control_defects),
            "boundary_incidence_count": len(boundary_defects),
            "dropped_subterms": ("U26.ii:no_matched_control_stratum",)
            if not control_defects
            else (),
        },
    )


def U27(scene: Scene, alpha_grid_index: Optional[int]) -> FacetResult:
    """Containment / non-intrusion. Frozen SHA-256: fee484e87c819fbe1457b7519fb0586ef1ca5f27f8fed0e7eeaa5fc7ba997f5c.

    Parameters
    ----------
    scene : Scene
        Validated canonical scene.
    alpha_grid_index : int or None
        Required shared-grid state. A manifest row index selects a value;
        ``None`` publishes the preregistered unselected envelope.

    Returns
    -------
    FacetResult
        Selected scalar value or typed unselected-grid envelope.
    """

    selected_offset = selected_alpha_grid_offset(alpha_grid_index)
    if not scene.graph.clusters:
        return na_result("no_declared_clusters")
    clusters = _clusters(scene, 3)
    member_clusters = _clusters(scene, 2)
    if not member_clusters:
        return na_result("cluster_too_small_for_region")
    regions = _regions(scene)
    node_intrusions_by_grid: List[List[Scalar]] = [[] for _ in range(12)]
    node_intrusion_masses: List[float] = []
    member_outside: List[Scalar] = []
    member_masses: List[float] = []
    route_intrusions: List[Scalar] = []
    masses = _node_masses(scene)
    cluster_masses = {
        name: sum(masses[member] for member in members) for name, members in member_clusters.items()
    }
    grid = tuple((alpha_clear, alpha_high) for _, alpha_clear, alpha_high in ALPHA_GRID)
    for name, members in clusters.items():
        member_set = set(members)
        region = regions[name]
        diagonals = [
            float(2.0 * torch.linalg.vector_norm(scene.node_boxes[member].half_extents))
            for member in members
        ]
        input_spacing = max(diagonals)
        input_radius = input_spacing
        raw_budget = (
            2.0 * math.pi * 1.5 * (input_radius + max(diagonals) / 2.0) - sum(diagonals)
        ) / len(members)
        budget = max(raw_budget, 0.125 * scene.intrinsic_unit)
        floor = 0.25 * scene.intrinsic_unit
        logistic_argument = (budget - floor) / (0.5 * floor)
        floor_blend = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, logistic_argument))))
        for node in scene.node_boxes:
            if node.owner in member_set:
                continue
            penetration = p_max(0.0, -_signed_box_region(node, region))
            absolute = 1.0 - p_exp(-penetration / (0.25 * scene.intrinsic_unit))
            excess = 1.0 - _smooth_fade(1.0 - penetration / budget)
            for grid_index, (alpha_clear, alpha_high) in enumerate(grid):
                alpha = alpha_clear * (1.0 - floor_blend) + (alpha_clear * floor_blend * alpha_high)
                node_intrusions_by_grid[grid_index].append(
                    _interim_cluster_severity(
                        snap_unit(alpha * absolute + (1.0 - alpha) * excess),
                        1.0,
                        1.0,
                        subject_is_lower=False,
                    )
                )
            node_intrusion_masses.append(masses[node.owner] * cluster_masses[name])
        for route in resolved_routes(scene):
            source, target = scene.graph.edges[route.edge_index]
            if source in member_set or target in member_set:
                continue
            lengths = _row_norms_or_zero(route.points[1:] - route.points[:-1])
            total_length = keep(torch.sum(lengths))
            if as_float(total_length) == 0.0:
                continue
            inside_length: Scalar = 0.0
            segment_lengths = list(lengths) if tracing_active() else lengths.tolist()
            for index, segment_length in enumerate(segment_lengths):
                inside_length += segment_length * _segment_region_fraction(
                    route.points[index], route.points[index + 1], region
                )
            ratio = inside_length / total_length
            route_intrusions.append(1.0 - _smooth_fade(1.0 - ratio / 0.25))

    for name, members in member_clusters.items():
        region = regions[name]
        for member in members:
            others = tuple(box for box in region.boxes if box.owner != member)
            if not others:
                continue
            leave_one_out = _ClusterRegion(others, region.radius, region.bounds)
            outside = p_max(0.0, _signed_box_region(scene.node_boxes[member], leave_one_out))
            member_outside.append(1.0 - _smooth_fade(1.0 - outside / (2.0 * scene.intrinsic_unit)))
            member_masses.append(masses[member] * cluster_masses[name])

    member_value = global_blend(member_outside, member_masses) if member_outside else None
    route_weights = [
        cluster_masses[name]
        for name, members in clusters.items()
        for route in resolved_routes(scene)
        if not set(scene.graph.edges[route.edge_index]) & set(members)
        and float(torch.sum(torch.linalg.vector_norm(route.points[1:] - route.points[:-1], dim=1)))
        > 0.0
    ]
    route_value = global_blend(route_intrusions, route_weights) if route_intrusions else None
    grid_values: List[float] = []
    grid_rows: List[Tuple[Scalar, Dict[str, Scalar]]] = []
    for intrusion_values in node_intrusions_by_grid:
        values: Dict[str, Scalar] = {}
        if intrusion_values:
            values["U27.i"] = global_blend(intrusion_values, node_intrusion_masses)
        if member_value is not None:
            values["U27.ii"] = member_value
        if route_value is not None:
            values["U27.iii"] = route_value
        # The row composition runs unpublished per grid row; only the
        # selected row reaches value_result (and, in a trace, the buffer).
        row_value = compose_facet_rows("U27", values)
        if row_value is not None:
            grid_values.append(as_float(row_value))
            grid_rows.append((row_value, values))
    if not grid_values:
        return na_result("no_foreign_nodes")
    upper_index = max(range(len(grid_values)), key=grid_values.__getitem__)
    raw = {
        "grid_envelope": (min(grid_values), max(grid_values)),
        "grid_values": tuple(grid_values),
        "upper_subterms": {key: as_float(item) for key, item in grid_rows[upper_index][1].items()},
        "foreign_incidence_count": len(node_intrusion_masses),
        "member_incidence_count": len(member_outside),
        "route_incidence_count": len(route_intrusions),
    }
    if selected_offset is None:
        return na_result("alpha_grid_unselected", raw)
    raw.update(
        {
            "alpha_grid_index": selected_offset + 1,
            "alpha_grid_name": ALPHA_GRID[selected_offset][0],
        }
    )
    selected_value, selected_subterms = grid_rows[selected_offset]
    return value_result(selected_value, selected_subterms, raw)


def U28(scene: Scene, alpha_grid_index: Optional[int]) -> FacetResult:
    """Hierarchy nesting fidelity. Frozen SHA-256: 5d48376583595a1fad3dee50676d440eb2f0d5bf9bdd6acb03b93535a421e649.

    Parameters
    ----------
    scene : Scene
        Validated canonical scene.
    alpha_grid_index : int or None
        Required shared-grid state. A manifest row index selects a value;
        ``None`` publishes the preregistered unselected envelope.

    Returns
    -------
    FacetResult
        Selected scalar value or typed unselected-grid envelope.
    """

    selected_offset = selected_alpha_grid_offset(alpha_grid_index)
    if not scene.graph.cluster_parents:
        return na_result("no_declared_hierarchy")
    canonical = _clusters(scene)
    for child, parent in scene.graph.cluster_parents.items():
        if child not in canonical or parent not in canonical:
            return invalid_result("hierarchy_membership_mismatch")
        if not set(canonical[child]).issubset(canonical[parent]):
            return invalid_result("hierarchy_membership_mismatch")
    for start in scene.graph.cluster_parents:
        seen: Set[str] = set()
        current = start
        while current in scene.graph.cluster_parents:
            if current in seen:
                return invalid_result("hierarchy_not_acyclic")
            seen.add(current)
            current = scene.graph.cluster_parents[current]

    regions = _regions(scene)
    masses = _node_masses(scene)
    area_cache = {name: _region_relation_area(region)[0] for name, region in regions.items()}
    containment_by_grid: List[List[Scalar]] = [[] for _ in range(12)]
    containment_masses: List[float] = []
    sibling_overlap: List[Scalar] = []
    sibling_masses: List[float] = []
    size_relation: List[Scalar] = []
    size_masses: List[float] = []
    children_by_parent: Dict[str, list[str]] = {}
    for child, parent in scene.graph.cluster_parents.items():
        if child not in regions or parent not in regions or canonical[child] == canonical[parent]:
            continue
        if len(canonical[child]) < 3 or len(canonical[parent]) < 3:
            continue
        area_child = area_cache[child]
        area_parent = area_cache[parent]
        _, overlap_with_parent = _region_relation_area(regions[child], regions[parent])
        escaped = p_max(0.0, 1.0 - overlap_with_parent / p_max(area_child, 1e-12))
        depth = _region_depth_outside(regions[child], regions[parent])
        absolute = 1.0 - p_exp(-depth / (0.25 * scene.intrinsic_unit))
        excess = 1.0 - _smooth_fade(1.0 - escaped / 0.10)
        for index, alpha in enumerate(
            _alpha_grid_for_budget(scene, _cluster_budget(scene, canonical[child]))
        ):
            containment_by_grid[index].append(alpha * absolute + (1.0 - alpha) * excess)
        child_mass = sum(masses[node] for node in canonical[child])
        parent_mass = sum(masses[node] for node in canonical[parent])
        containment_masses.append(child_mass)
        residual = p_log(area_child / area_parent) - math.log(child_mass / parent_mass)
        size_relation.append(1.0 - _smooth_fade(1.0 - p_abs(residual) / math.log(3.0)))
        size_masses.append(child_mass)
        children_by_parent.setdefault(parent, []).append(child)
    for children in children_by_parent.values():
        for index, left in enumerate(children):
            for right in children[index + 1 :]:
                left_area = area_cache[left]
                right_area = area_cache[right]
                _, overlap_area = _region_relation_area(regions[left], regions[right])
                fraction = overlap_area / p_max(p_min(left_area, right_area), 1e-12)
                sibling_overlap.append(1.0 - _smooth_fade(1.0 - fraction / 0.15))
                sibling_masses.append(
                    min(
                        sum(masses[node] for node in canonical[left]),
                        sum(masses[node] for node in canonical[right]),
                    )
                )
    if not any(containment_by_grid) and not sibling_overlap:
        return na_result("no_eligible_hierarchy_relations")
    sibling_value = global_blend(sibling_overlap, sibling_masses) if sibling_overlap else None
    size_value = global_blend(size_relation, size_masses) if size_relation else None
    grid_values: List[float] = []
    grid_rows: List[Tuple[Scalar, Dict[str, Scalar]]] = []
    for containment in containment_by_grid:
        values: Dict[str, Scalar] = {}
        if containment:
            values["U28.i"] = global_blend(containment, containment_masses)
        if sibling_value is not None:
            values["U28.ii"] = sibling_value
        if size_value is not None:
            values["U28.iii"] = size_value
        # The row composition runs unpublished per grid row; only the
        # selected row reaches value_result (and, in a trace, the buffer).
        row_value = compose_facet_rows("U28", values)
        if row_value is not None:
            grid_values.append(as_float(row_value))
            grid_rows.append((row_value, values))
    upper_index = max(range(len(grid_values)), key=grid_values.__getitem__)
    raw = {
        "grid_envelope": (min(grid_values), max(grid_values)),
        "grid_values": tuple(grid_values),
        "upper_subterms": {key: as_float(item) for key, item in grid_rows[upper_index][1].items()},
        "parent_child_count": len(size_relation),
        "sibling_pair_count": len(sibling_overlap),
        "region_areas": {name: as_float(area) for name, area in area_cache.items()},
    }
    if selected_offset is None:
        return na_result("alpha_grid_unselected", raw)
    raw.update(
        {
            "alpha_grid_index": selected_offset + 1,
            "alpha_grid_name": ALPHA_GRID[selected_offset][0],
        }
    )
    selected_value, selected_subterms = grid_rows[selected_offset]
    return value_result(selected_value, selected_subterms, raw)


def U29(scene: Scene) -> FacetResult:
    """Cluster shape coherence. Frozen SHA-256: 658cf935c25388c5959b5c4af9880840c59b28fe1b6284d0dfe1dfa0d9e7a819."""

    if not scene.graph.clusters:
        return na_result("no_declared_clusters")
    clusters = _clusters(scene, 5)
    if not clusters:
        return na_result(
            "clusters_too_small_for_shape", {"declared_cluster_count": len(_clusters(scene))}
        )
    masses = _node_masses(scene)
    defects: List[Scalar] = []
    cluster_masses: List[float] = []
    statistics: Dict[str, object] = {"clusters": {}}
    for name, members in clusters.items():
        spacing = _cluster_spacing(scene, members)
        radius = p_max(spacing, 0.05 * scene.intrinsic_unit)
        centers = scene.positions[list(members)]
        area, perimeter = _equal_disc_union_area_perimeter(centers, radius)
        quotient = perimeter * perimeter / (4.0 * math.pi * p_max(area, 1e-300))
        diameter = _induced_diameter(scene, members)
        reference = 1.0 + diameter / math.sqrt(len(members))
        excess = soft_pos(p_log(quotient / (1.5 * reference)))
        defect = excess / (1.0 + excess)
        defects.append(defect)
        cluster_masses.append(sum(masses[member] for member in members))
        statistics["clusters"][name] = {
            "area": as_float(area),
            "perimeter": as_float(perimeter),
            "Q": as_float(quotient),
            "Q_ref": reference,
            "diameter": diameter,
            "radius": as_float(radius),
            "defect": as_float(defect),
        }
    value = global_blend(defects, cluster_masses)
    statistics["cluster_count"] = len(defects)
    return value_result(value, {"U29.headline": value}, statistics)


def U30(scene: Scene, alpha_grid_index: Optional[int]) -> FacetResult:
    """Cluster labels. Frozen SHA-256: 64ce190db377b294922a09b4b0e045224c02d611ff451b33f3fb5d07d1ca69c4.

    Parameters
    ----------
    scene : Scene
        Validated canonical scene.
    alpha_grid_index : int or None
        Required shared-grid state. A manifest row index selects a value;
        ``None`` publishes the preregistered unselected envelope.

    Returns
    -------
    FacetResult
        Selected scalar value or typed unselected-grid envelope.
    """

    selected_offset = selected_alpha_grid_offset(alpha_grid_index)
    if not scene.graph.clusters:
        return na_result("no_declared_clusters")
    clusters = _clusters(scene, 3)
    if not clusters or "cluster_labels" not in scene.profile.visible_channels:
        return na_result("no_declared_cluster_labels")
    regions = {name: region for name, region in _regions(scene).items() if name in clusters}
    masses = _node_masses(scene)
    labels = {name: box for name, box in scene.cluster_label_boxes.items() if name in regions}
    if not labels:
        return invalid_result("required_cluster_label_missing")
    label_masses: Dict[str, float] = {}
    for name in labels:
        label_masses[name] = sum(masses[node] for node in clusters[name])

    association: List[Scalar] = []
    association_masses: List[float] = []
    padding_by_grid: List[List[Scalar]] = [[] for _ in range(12)]
    occlusion_by_grid: List[List[Scalar]] = [[] for _ in range(12)]
    all_masses: List[float] = []
    for name, label in labels.items():
        region = regions[name]
        budget = _cluster_budget(scene, clusters[name])
        alpha_grid = _alpha_grid_for_budget(scene, budget)
        own_clearance = _signed_box_region(label, region)
        foreign_clearances = [
            _signed_box_region(label, other)
            for other_name, other in regions.items()
            if other_name != name
        ]
        if foreign_clearances:
            margin = (min(foreign_clearances, key=as_float) - own_clearance) / budget
            association.append(1.0 - _smooth_fade((margin + 1.0) / 2.0))
            association_masses.append(label_masses[name])
        pad = _signed_top_padding(label, region)
        pad_target = _U30_PAD_TARGET_U * scene.intrinsic_unit
        deviation = p_abs(pad - pad_target)
        absolute = 1.0 - p_exp(-p_max(0.0, own_clearance) / (0.25 * scene.intrinsic_unit))
        excess = 1.0 - _smooth_fade(1.0 - deviation / pad_target)
        obstacle_defects_by_grid: List[List[Scalar]] = [[] for _ in range(12)]
        label_area = float(4.0 * torch.prod(label.half_extents))
        obstacles = [(box, False) for box in scene.node_boxes]
        obstacles.extend((box, False) for box in scene.node_label_boxes)
        obstacles.extend(
            (other, name < other_name) for other_name, other in labels.items() if other_name != name
        )
        for obstacle, subject_is_lower in obstacles:
            signed, overlap = aabb_pair(label, obstacle)
            overlap_area = overlap * min(
                float(4.0 * torch.prod(label.half_extents)),
                float(4.0 * torch.prod(obstacle.half_extents)),
            )
            absolute_occlusion = 1.0 - p_exp(
                -overlap_area / (0.05 * scene.intrinsic_unit * scene.intrinsic_unit)
            )
            excess_occlusion: Scalar = (
                1.0 if as_float(signed) <= 0.0 else 1.0 - _smooth_fade(signed / budget)
            )
            for grid_index, alpha in enumerate(alpha_grid):
                pair_defect = alpha * absolute_occlusion + (1.0 - alpha) * excess_occlusion
                obstacle_defects_by_grid[grid_index].append(
                    _interim_cluster_severity(
                        pair_defect,
                        overlap_area,
                        label_area,
                        subject_is_lower=subject_is_lower,
                    )
                )
        for route in resolved_routes(scene):
            width = (
                scene.style.edge_stroke_widths[route.edge_index]
                if scene.style.edge_stroke_widths
                else scene.style.route_stroke_width * scene.style.coordinate_scale
            )
            centerline_clearance = min(
                (
                    _box_segment_clearance(label, start, end)
                    for start, end in zip(route.points[:-1], route.points[1:])
                ),
                key=as_float,
            )
            signed = centerline_clearance - width / 2.0
            overlap_area = _route_box_overlap_area(route.points, label, width)
            absolute_occlusion = 1.0 - p_exp(
                -overlap_area / (0.05 * scene.intrinsic_unit * scene.intrinsic_unit)
            )
            excess_occlusion = (
                1.0 if as_float(signed) <= 0.0 else 1.0 - _smooth_fade(signed / budget)
            )
            for grid_index, alpha in enumerate(alpha_grid):
                pair_defect = alpha * absolute_occlusion + (1.0 - alpha) * excess_occlusion
                obstacle_defects_by_grid[grid_index].append(
                    _interim_cluster_severity(
                        pair_defect,
                        overlap_area,
                        label_area,
                        subject_is_lower=False,
                    )
                )
        for grid_index, alpha in enumerate(alpha_grid):
            padding_by_grid[grid_index].append(alpha * absolute + (1.0 - alpha) * excess)
            survival = _p_prod([1.0 - defect for defect in obstacle_defects_by_grid[grid_index]])
            occlusion_by_grid[grid_index].append(1.0 - survival)
        all_masses.append(label_masses[name])

    association_value = global_blend(association, association_masses) if association else None
    grid_values: List[float] = []
    grid_rows: List[Tuple[Scalar, Dict[str, Scalar]]] = []
    for padding, occlusion in zip(padding_by_grid, occlusion_by_grid):
        values: Dict[str, Scalar] = {}
        if association_value is not None:
            values["U30.i"] = association_value
        values["U30.ii"] = global_blend(padding, all_masses)
        values["U30.iii"] = global_blend(occlusion, all_masses)
        # The row composition runs unpublished per grid row; only the
        # selected row reaches value_result (and, in a trace, the buffer).
        row_value = compose_facet_rows("U30", values)
        if row_value is not None:
            grid_values.append(as_float(row_value))
            grid_rows.append((row_value, values))
    upper_index = max(range(len(grid_values)), key=grid_values.__getitem__)
    raw = {
        "grid_envelope": (min(grid_values), max(grid_values)),
        "grid_values": tuple(grid_values),
        "upper_subterms": {key: as_float(item) for key, item in grid_rows[upper_index][1].items()},
        "label_count": len(labels),
        "derived_label_boxes": labels,
    }
    if selected_offset is None:
        return na_result("alpha_grid_unselected", raw)
    raw.update(
        {
            "alpha_grid_index": selected_offset + 1,
            "alpha_grid_name": ALPHA_GRID[selected_offset][0],
        }
    )
    selected_value, selected_subterms = grid_rows[selected_offset]
    return value_result(selected_value, selected_subterms, raw)
