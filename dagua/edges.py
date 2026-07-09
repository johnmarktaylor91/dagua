"""Heuristic bezier edge routing — compute control points after layout.

For each edge, computes cubic bezier control points (p0, cp1, cp2, p1)
based on the geometry of the source and target positions.
Supports per-edge routing modes (bezier, straight, ortho, taxi) and
shape-aware port positioning (ellipse, diamond, rectangle).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import torch


@dataclass
class BezierCurve:
    """A routed edge centerline.

    Parameters
    ----------
    p0 : tuple[float, float]
        Start point.
    cp1 : tuple[float, float]
        First cubic control point or first bend anchor for waypoint routes.
    cp2 : tuple[float, float]
        Second cubic control point or last bend anchor for waypoint routes.
    p1 : tuple[float, float]
        End point.
    waypoints : tuple[tuple[float, float], ...] | None, default=None
        Optional polyline vertices for routing modes such as ``"ortho"`` and
        ``"taxi"`` that need hard bends. When present, evaluation and tangents
        follow the waypoint polyline instead of the cubic control points.
    routing : str, default="bezier"
        Routing mode used to produce the curve.
    direction : str, default="TB"
        Layout direction used when constructing rectilinear routes.
    step_fraction : float or torch.Tensor or None, default=None
        Taxi-route step fraction. The optimizer may carry this as a tensor
        before converting the final route back to floats for rendering.
    """

    p0: Tuple[float, float]
    cp1: Tuple[float, float]
    cp2: Tuple[float, float]
    p1: Tuple[float, float]
    waypoints: Optional[Tuple[Tuple[float, float], ...]] = None
    routing: str = "bezier"
    direction: str = "TB"
    step_fraction: Optional[Union[float, torch.Tensor]] = None


Rect = Tuple[float, float, float, float]


def _polyline_curve(
    points: Sequence[Tuple[float, float]],
    routing: str = "bezier",
    direction: str = "TB",
    step_fraction: Optional[Union[float, torch.Tensor]] = None,
) -> BezierCurve:
    """Build a routed curve backed by explicit polyline waypoints.

    Parameters
    ----------
    points : sequence[tuple[float, float]]
        Polyline vertices in draw order.
    routing : str, default="bezier"
        Routing mode represented by the polyline.
    direction : str, default="TB"
        Layout direction used to construct the polyline.
    step_fraction : float or torch.Tensor or None, default=None
        Taxi step fraction associated with the polyline, if any.

    Returns
    -------
    BezierCurve
        Curve whose endpoints and bend anchors mirror the supplied polyline.
    """
    deduped: List[Tuple[float, float]] = []
    for point in points:
        normalized_point = (float(point[0]), float(point[1]))
        same_x = deduped and math.isclose(deduped[-1][0], normalized_point[0], abs_tol=1e-9)
        same_y = deduped and math.isclose(deduped[-1][1], normalized_point[1], abs_tol=1e-9)
        if same_x and same_y:
            continue
        deduped.append(normalized_point)

    if not deduped:
        raise ValueError("Polyline routes require at least one point.")
    if len(deduped) == 1:
        point = deduped[0]
        return BezierCurve(
            point,
            point,
            point,
            point,
            waypoints=(point,),
            routing=routing,
            direction=direction,
            step_fraction=step_fraction,
        )

    first_bend = deduped[1] if len(deduped) > 2 else deduped[0]
    last_bend = deduped[-2] if len(deduped) > 2 else deduped[-1]
    return BezierCurve(
        deduped[0],
        first_bend,
        last_bend,
        deduped[-1],
        waypoints=tuple(deduped),
        routing=routing,
        direction=direction,
        step_fraction=step_fraction,
    )


def _polyline_point_at(points: Sequence[Tuple[float, float]], t: float) -> Tuple[float, float]:
    """Evaluate a waypoint polyline at normalized arc-length ``t``.

    Parameters
    ----------
    points : sequence[tuple[float, float]]
        Ordered polyline vertices.
    t : float
        Normalized distance in ``[0, 1]``.

    Returns
    -------
    tuple[float, float]
        Interpolated point on the polyline.
    """
    if len(points) <= 1:
        point = points[0]
        return float(point[0]), float(point[1])

    clamped_t = min(max(float(t), 0.0), 1.0)
    segment_lengths: List[float] = []
    total_length = 0.0
    for start, stop in zip(points, points[1:]):
        segment_length = math.hypot(stop[0] - start[0], stop[1] - start[1])
        segment_lengths.append(segment_length)
        total_length += segment_length
    if total_length <= 1e-9:
        point = points[0]
        return float(point[0]), float(point[1])

    target_length = clamped_t * total_length
    traversed = 0.0
    for index, segment_length in enumerate(segment_lengths):
        if segment_length <= 1e-9:
            continue
        next_traversed = traversed + segment_length
        if target_length <= next_traversed or index == len(segment_lengths) - 1:
            local_t = (target_length - traversed) / segment_length
            start = points[index]
            stop = points[index + 1]
            return (
                float(start[0] + (stop[0] - start[0]) * local_t),
                float(start[1] + (stop[1] - start[1]) * local_t),
            )
        traversed = next_traversed

    point = points[-1]
    return float(point[0]), float(point[1])


def _rect_bounds(rect: Any) -> Rect:
    """Return normalized rectangle bounds from a tuple or placement box.

    Parameters
    ----------
    rect : Any
        Rectangle as ``(x_min, y_min, x_max, y_max)`` or a
        ``ClusterPlacementBox``-like object.

    Returns
    -------
    tuple[float, float, float, float]
        Normalized rectangle bounds.
    """
    if hasattr(rect, "inner_bbox") and hasattr(rect, "label_band_y_extent"):
        inner_x_min, _, inner_x_max, _ = rect.inner_bbox
        y_max = float(rect.label_band_y_extent[0])
        y_min = y_max - float(rect.height)
        center_x = (float(inner_x_min) + float(inner_x_max)) / 2.0
        x_min = center_x - float(rect.width) / 2.0
        x_max = center_x + float(rect.width) / 2.0
    else:
        x_min, y_min, x_max, y_max = rect
    return (
        min(float(x_min), float(x_max)),
        min(float(y_min), float(y_max)),
        max(float(x_min), float(x_max)),
        max(float(y_min), float(y_max)),
    )


def _point_in_rect(point: Tuple[float, float], rect: Rect) -> bool:
    """Return whether a point lies inside or on a rectangle.

    Parameters
    ----------
    point : tuple[float, float]
        Point to test.
    rect : tuple[float, float, float, float]
        Rectangle bounds as ``(x_min, y_min, x_max, y_max)``.

    Returns
    -------
    bool
        ``True`` when the point is inside the rectangle.
    """
    x_min, y_min, x_max, y_max = rect
    return x_min <= point[0] <= x_max and y_min <= point[1] <= y_max


def _segment_rect_intersection(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    rect: Rect,
) -> Optional[Tuple[float, Tuple[float, float]]]:
    """Return the first segment intersection with a rectangle perimeter.

    Parameters
    ----------
    p0 : tuple[float, float]
        Segment start.
    p1 : tuple[float, float]
        Segment end.
    rect : tuple[float, float, float, float]
        Rectangle bounds as ``(x_min, y_min, x_max, y_max)``.

    Returns
    -------
    tuple[float, tuple[float, float]] | None
        Segment parameter and point for the first intersection, or ``None``.
    """
    x_min, y_min, x_max, y_max = rect
    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    dy = y1 - y0
    candidates: List[Tuple[float, Tuple[float, float]]] = []

    if abs(dx) > 1e-12:
        for x_edge in (x_min, x_max):
            t = (x_edge - x0) / dx
            if 0.0 <= t <= 1.0:
                y = y0 + t * dy
                if y_min - 1e-9 <= y <= y_max + 1e-9:
                    candidates.append((float(t), (float(x_edge), float(y))))
    if abs(dy) > 1e-12:
        for y_edge in (y_min, y_max):
            t = (y_edge - y0) / dy
            if 0.0 <= t <= 1.0:
                x = x0 + t * dx
                if x_min - 1e-9 <= x <= x_max + 1e-9:
                    candidates.append((float(t), (float(x), float(y_edge))))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0]


def polyline_intersect_rect(
    polyline: Sequence[Tuple[float, float]],
    rect: Any,
) -> Optional[Tuple[int, Tuple[float, float]]]:
    """Find the first intersection of a polyline with a rectangle perimeter.

    Parameters
    ----------
    polyline : sequence[tuple[float, float]]
        Ordered edge-body points.
    rect : Any
        Rectangle as ``(x_min, y_min, x_max, y_max)`` or a placement box.

    Returns
    -------
    tuple[int, tuple[float, float]] | None
        Segment index and intersection point, or ``None`` when the polyline
        never crosses the perimeter.
    """
    if len(polyline) < 2:
        return None
    bounds = _rect_bounds(rect)
    for index in range(len(polyline) - 1):
        p0 = (float(polyline[index][0]), float(polyline[index][1]))
        p1 = (float(polyline[index + 1][0]), float(polyline[index + 1][1]))
        if _point_in_rect(p0, bounds) == _point_in_rect(p1, bounds):
            continue
        intersection = _segment_rect_intersection(p0, p1, bounds)
        if intersection is not None:
            _, point = intersection
            return index, point
    return None


def clip_edge_at_cluster_boundaries(
    edge_polyline: List[Tuple[float, float]],
    src_idx: int,
    tgt_idx: int,
    cluster_membership: Dict[int, List[str]],
    cluster_bboxes: Dict[str, Any],
    skip_inner_cluster: bool = True,
) -> List[Tuple[float, float]]:
    """Clip an edge polyline at cluster perimeters.

    Parameters
    ----------
    edge_polyline : list[tuple[float, float]]
        Ordered points along the visible edge body from source to target.
    src_idx : int
        Source node index.
    tgt_idx : int
        Target node index.
    cluster_membership : dict[int, list[str]]
        Node index to cluster names containing that node.
    cluster_bboxes : dict[str, Any]
        Cluster name to rectangle bounds or ``ClusterPlacementBox``.
    skip_inner_cluster : bool, default=True
        When ``True``, clusters containing both endpoints are ignored.

    Returns
    -------
    list[tuple[float, float]]
        A possibly shorter polyline with the endpoint-side portion inside the
        outermost crossed cluster removed.
    """
    if src_idx == tgt_idx or len(edge_polyline) < 2:
        return edge_polyline

    src_clusters = set(cluster_membership.get(int(src_idx), []))
    tgt_clusters = set(cluster_membership.get(int(tgt_idx), []))
    crossed_clusters = [
        name
        for name in cluster_bboxes
        if (name in src_clusters) != (name in tgt_clusters)
        and (not skip_inner_cluster or name not in src_clusters.intersection(tgt_clusters))
    ]
    if not crossed_clusters:
        return edge_polyline

    clipped = list(edge_polyline)
    target_side_clusters = [name for name in crossed_clusters if name in tgt_clusters]
    source_side_clusters = [name for name in crossed_clusters if name in src_clusters]

    if target_side_clusters:
        best: Optional[Tuple[int, Tuple[float, float]]] = None
        for name in target_side_clusters:
            hit = polyline_intersect_rect(clipped, cluster_bboxes[name])
            if hit is not None and (best is None or hit[0] < best[0]):
                best = hit
        if best is not None:
            segment_index, point = best
            clipped = [*clipped[: segment_index + 1], point]

    if source_side_clusters:
        reversed_polyline = list(reversed(clipped))
        best_reversed: Optional[Tuple[int, Tuple[float, float]]] = None
        for name in source_side_clusters:
            hit = polyline_intersect_rect(reversed_polyline, cluster_bboxes[name])
            if hit is not None and (best_reversed is None or hit[0] < best_reversed[0]):
                best_reversed = hit
        if best_reversed is not None:
            segment_index, point = best_reversed
            retained_reversed = [*reversed_polyline[: segment_index + 1], point]
            clipped = list(reversed(retained_reversed))

    return clipped


def _polyline_tangent_at(points: Sequence[Tuple[float, float]], t: float) -> Tuple[float, float]:
    """Return the local tangent vector for a waypoint polyline.

    Parameters
    ----------
    points : sequence[tuple[float, float]]
        Ordered polyline vertices.
    t : float
        Normalized distance in ``[0, 1]``.

    Returns
    -------
    tuple[float, float]
        Tangent vector along the active polyline segment.
    """
    if len(points) <= 1:
        return (0.0, 0.0)

    clamped_t = min(max(float(t), 0.0), 1.0)
    segment_lengths: List[float] = []
    total_length = 0.0
    for start, stop in zip(points, points[1:]):
        segment_length = math.hypot(stop[0] - start[0], stop[1] - start[1])
        segment_lengths.append(segment_length)
        total_length += segment_length
    if total_length <= 1e-9:
        return (0.0, 0.0)

    target_length = clamped_t * total_length
    traversed = 0.0
    for index, segment_length in enumerate(segment_lengths):
        if segment_length <= 1e-9:
            continue
        next_traversed = traversed + segment_length
        if target_length <= next_traversed or index == len(segment_lengths) - 1:
            start = points[index]
            stop = points[index + 1]
            return (float(stop[0] - start[0]), float(stop[1] - start[1]))
        traversed = next_traversed

    last_start = points[-2]
    last_stop = points[-1]
    return (float(last_stop[0] - last_start[0]), float(last_stop[1] - last_start[1]))


def _compute_self_loop_curve(
    sx: float,
    sy: float,
    sw: float,
    sh: float,
    direction: str,
) -> BezierCurve:
    """Compute a direction-aware self-loop curve.

    Creates a wide semicircular arc that exits and re-enters the node at
    two distinct points on the outward-facing edge, matching the visual
    style of Graphviz and matplotlib self-loops.

    Parameters
    ----------
    sx : float
        X coordinate of the node center.
    sy : float
        Y coordinate of the node center.
    sw : float
        Node width.
    sh : float
        Node height.
    direction : str
        Layout direction. Supported values are ``"TB"``, ``"BT"``, ``"LR"``,
        and ``"RL"``.

    Returns
    -------
    BezierCurve
        Wide semicircular loop on the outward-facing side.
    """
    # Loop exits at two separate points on the node edge, spread apart,
    # with control points creating a wide circular arc.
    spread = max(sw, sh) * 0.35  # how far apart the exit/entry points are
    arc_height = max(sw, sh) * 1.1  # tuned down from 1.6 to keep loops compact

    # Cubic bezier approximation of a semicircular arc:
    # For a semicircle of radius r, the control point offset is ~r * 1.33
    cp_factor = 1.33

    if direction == "BT":
        edge_y = sy - sh / 2  # bottom edge
        return BezierCurve(
            p0=(sx - spread, edge_y),
            cp1=(sx - spread * cp_factor, edge_y - arc_height),
            cp2=(sx + spread * cp_factor, edge_y - arc_height),
            p1=(sx + spread, edge_y),
        )

    if direction == "LR":
        edge_x = sx - sw / 2  # left edge
        return BezierCurve(
            p0=(edge_x, sy - spread),
            cp1=(edge_x - arc_height, sy - spread * cp_factor),
            cp2=(edge_x - arc_height, sy + spread * cp_factor),
            p1=(edge_x, sy + spread),
        )

    if direction == "RL":
        edge_x = sx + sw / 2  # right edge
        return BezierCurve(
            p0=(edge_x, sy + spread),
            cp1=(edge_x + arc_height, sy + spread * cp_factor),
            cp2=(edge_x + arc_height, sy - spread * cp_factor),
            p1=(edge_x, sy - spread),
        )

    # TB (default): loop above the node
    edge_y = sy + sh / 2  # top edge
    return BezierCurve(
        p0=(sx + spread, edge_y),
        cp1=(sx + spread * cp_factor, edge_y + arc_height),
        cp2=(sx - spread * cp_factor, edge_y + arc_height),
        p1=(sx - spread, edge_y),
    )


def _compute_directional_ports(
    sx: float,
    sy: float,
    sw: float,
    sh: float,
    tx: float,
    ty: float,
    tw: float,
    th: float,
    direction: str,
    port_style: str,
    out_rank: int,
    out_total: int,
    in_rank: int,
    in_total: int,
) -> Tuple[float, float, float, float]:
    """Compute direction-aware source and target port positions.

    Parameters
    ----------
    sx : float
        Source center X coordinate.
    sy : float
        Source center Y coordinate.
    sw : float
        Source node width.
    sh : float
        Source node height.
    tx : float
        Target center X coordinate.
    ty : float
        Target center Y coordinate.
    tw : float
        Target node width.
    th : float
        Target node height.
    direction : str
        Layout direction.
    port_style : str
        Either ``"distributed"`` or ``"center"``.
    out_rank : int
        Source-side port rank for this edge.
    out_total : int
        Total number of outgoing ports on the source.
    in_rank : int
        Target-side port rank for this edge.
    in_total : int
        Total number of incoming ports on the target.

    Returns
    -------
    tuple[float, float, float, float]
        Source port X/Y followed by target port X/Y.
    """
    if direction in ("TB", "BT"):
        if direction == "TB":
            src_port_y = sy - sh / 2
            tgt_port_y = ty + th / 2
        else:
            src_port_y = sy + sh / 2
            tgt_port_y = ty - th / 2

        if port_style == "center":
            src_port_x = sx
            tgt_port_x = tx
        else:
            src_port_x = sx - sw / 2 + sw * (out_rank + 0.5) / out_total
            tgt_port_x = tx - tw / 2 + tw * (in_rank + 0.5) / in_total

        return src_port_x, src_port_y, tgt_port_x, tgt_port_y

    if direction in ("LR", "RL"):
        if direction == "LR":
            src_port_x = sx + sw / 2
            tgt_port_x = tx - tw / 2
        else:
            src_port_x = sx - sw / 2
            tgt_port_x = tx + tw / 2

        if port_style == "center":
            src_port_y = sy
            tgt_port_y = ty
        else:
            src_port_y = sy - sh / 2 + sh * (out_rank + 0.5) / out_total
            tgt_port_y = ty - th / 2 + th * (in_rank + 0.5) / in_total

        return src_port_x, src_port_y, tgt_port_x, tgt_port_y

    src_port_x = sx
    src_port_y = sy - sh / 2 if ty < sy else sy + sh / 2
    tgt_port_x = tx
    tgt_port_y = ty + th / 2 if sy > ty else ty - th / 2
    return src_port_x, src_port_y, tgt_port_x, tgt_port_y


def _reverse_back_edge_ports(
    sx: float,
    sy: float,
    sw: float,
    sh: float,
    tx: float,
    ty: float,
    tw: float,
    th: float,
    direction: str,
    src_port_x: float,
    src_port_y: float,
    tgt_port_x: float,
    tgt_port_y: float,
) -> Tuple[float, float, float, float]:
    """Reverse port sides when an edge runs against the layout direction.

    Parameters
    ----------
    sx : float
        Source center X coordinate.
    sy : float
        Source center Y coordinate.
    sw : float
        Source node width.
    sh : float
        Source node height.
    tx : float
        Target center X coordinate.
    ty : float
        Target center Y coordinate.
    tw : float
        Target node width.
    th : float
        Target node height.
    direction : str
        Layout direction.
    src_port_x : float
        Current source port X coordinate.
    src_port_y : float
        Current source port Y coordinate.
    tgt_port_x : float
        Current target port X coordinate.
    tgt_port_y : float
        Current target port Y coordinate.

    Returns
    -------
    tuple[float, float, float, float]
        Possibly adjusted source and target port coordinates.
    """
    if direction == "TB" and tgt_port_y > src_port_y:
        return src_port_x, sy + sh / 2, tgt_port_x, ty - th / 2
    if direction == "BT" and tgt_port_y < src_port_y:
        return src_port_x, sy - sh / 2, tgt_port_x, ty + th / 2
    if direction == "LR" and tgt_port_x < src_port_x:
        return sx - sw / 2, src_port_y, tx + tw / 2, tgt_port_y
    if direction == "RL" and tgt_port_x > src_port_x:
        return sx + sw / 2, src_port_y, tx - tw / 2, tgt_port_y

    return src_port_x, src_port_y, tgt_port_x, tgt_port_y


def route_edges(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    direction: str = "TB",
    graph: Optional[Any] = None,
) -> List[BezierCurve]:
    """Compute routed edge curves for a laid-out graph.

    Parameters
    ----------
    positions : torch.Tensor
        Node centers with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Source and target indices with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node widths and heights with shape ``[N, 2]``.
    direction : str, default="TB"
        Layout direction used to choose the node side for edge ports.
    graph : Any | None, default=None
        Optional graph object providing per-edge styles and per-node shapes.

    Returns
    -------
    list[BezierCurve]
        One routed curve per edge.
    """
    if edge_index.numel() == 0:
        return []

    pos = positions.detach().cpu()
    sizes = node_sizes.detach().cpu()
    num_edges = edge_index.shape[1]
    src_indices = edge_index[0].tolist()
    tgt_indices = edge_index[1].tolist()
    pair_counts: Dict[Tuple[int, int], int] = {}
    pair_ranks: Dict[int, int] = {}
    for e_idx, (source, target) in enumerate(zip(src_indices, tgt_indices)):
        pair_key = (int(source), int(target))
        rank = pair_counts.get(pair_key, 0)
        pair_ranks[e_idx] = rank
        pair_counts[pair_key] = rank + 1
    x_coords = pos[:, 0].tolist()
    y_coords = pos[:, 1].tolist()
    widths = sizes[:, 0].tolist()
    heights = sizes[:, 1].tolist()

    out_order: Dict[int, Tuple[int, int]] = {}
    in_order: Dict[int, Tuple[int, int]] = {}

    # Track port assignment order (sort by target/source x position)
    out_edges: Dict[int, List[Tuple[int, float]]] = {}
    in_edges: Dict[int, List[Tuple[int, float]]] = {}
    for e_idx in range(num_edges):
        s, t = src_indices[e_idx], tgt_indices[e_idx]
        out_edges.setdefault(s, []).append((e_idx, x_coords[t]))
        in_edges.setdefault(t, []).append((e_idx, x_coords[s]))

    # Sort ports by connected node x-position to reduce crossings
    for node in out_edges:
        out_edges[node].sort(key=lambda x: x[1])
        for rank, (e_idx, _) in enumerate(out_edges[node]):
            out_order[e_idx] = (rank, len(out_edges[node]))

    for node in in_edges:
        in_edges[node].sort(key=lambda x: x[1])
        for rank, (e_idx, _) in enumerate(in_edges[node]):
            in_order[e_idx] = (rank, len(in_edges[node]))

    # Pre-compute cluster bboxes and node membership for cluster-aware routing
    cluster_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    node_cluster_set: Dict[int, Set[str]] = {}
    node_shapes = None
    edge_styles = None
    if graph is not None and hasattr(graph, "clusters") and graph.clusters:
        from dagua.utils import collect_cluster_leaves

        for cname, cmembers in graph.clusters.items():
            if isinstance(cmembers, dict):
                cmembers = collect_cluster_leaves(cmembers)
            if not cmembers:
                continue
            # Get cluster style padding
            cstyle = graph.get_style_for_cluster(cname)
            cpad = cstyle.padding
            # Compute bbox
            valid_members = [m for m in cmembers if 0 <= m < pos.shape[0]]
            if valid_members:
                idx = torch.tensor(valid_members, dtype=torch.long)
                member_pos = pos[idx]
                member_sizes = sizes[idx] / 2
                bx_min = (member_pos[:, 0] - member_sizes[:, 0]).min().item() - cpad
                bx_max = (member_pos[:, 0] + member_sizes[:, 0]).max().item() + cpad
                by_min = (member_pos[:, 1] - member_sizes[:, 1]).min().item() - cpad
                by_max = (member_pos[:, 1] + member_sizes[:, 1]).max().item() + cpad + 14
                cluster_bboxes[cname] = (bx_min, by_min, bx_max, by_max)
            for m in valid_members:
                if m not in node_cluster_set:
                    node_cluster_set[m] = set()
                node_cluster_set[m].add(cname)

    if graph is not None:
        node_shapes = [graph.get_style_for_node(i).shape for i in range(pos.shape[0])]
        edge_styles = [_get_route_edge_style(graph, i) for i in range(num_edges)]

    # Spatial grid for node-avoidance deflection (deliverable r80-S7#1). Cell
    # size tracks the mean node diagonal so each cell holds O(1) nodes on
    # typical graphs; falls back to a constant for degenerate all-zero sizes.
    num_nodes = pos.shape[0]
    node_grid: Dict[Tuple[int, int], List[int]] = {}
    grid_cell_size = 40.0
    spread_scales: List[float] = []
    if num_nodes > 0:
        mean_diag = sum(math.hypot(w, h) for w, h in zip(widths, heights)) / num_nodes
        if mean_diag > 1e-6:
            grid_cell_size = mean_diag
        node_grid = _build_node_grid(x_coords, y_coords, grid_cell_size)
        # r80-S7b#3: per-node density scale for the port-spread budget.
        spread_scales = _local_density_spread_scales(node_grid, grid_cell_size, x_coords, y_coords)

    # r80-S7b#2: store of already-accepted routes for the crossing-aware
    # acceptance referee (greedy monotone: each edge's S7 modifications are
    # kept only if they do not create net new crossings against edges
    # routed before it).
    routed_polylines: List[List[Tuple[float, float]]] = []
    routed_bboxes: List[Rect] = []
    _REFEREE_SAMPLES = 12

    curves = []
    for e_idx in range(num_edges):
        s, t = src_indices[e_idx], tgt_indices[e_idx]
        sx, sy = x_coords[s], y_coords[s]
        tx, ty = x_coords[t], y_coords[t]
        sw, sh = widths[s], heights[s]
        tw, th = widths[t], heights[t]

        # Self-loops stay on the outward-facing side for the layout direction.
        if s == t:
            loop_curve = _compute_self_loop_curve(sx, sy, sw, sh, direction)
            loop_poly = _curve_polyline_samples(loop_curve, sample_count=_REFEREE_SAMPLES)
            routed_polylines.append(loop_poly)
            routed_bboxes.append(_poly_bbox(loop_poly))
            curves.append(loop_curve)
            continue

        # Per-edge style
        edge_style = None
        if edge_styles is not None:
            edge_style = edge_styles[e_idx]

        # Port positions
        out_rank, out_total = out_order.get(e_idx, (0, 1))
        in_rank, in_total = in_order.get(e_idx, (0, 1))

        port_style = edge_style.port_style if edge_style is not None else "distributed"
        src_port_x, src_port_y, tgt_port_x, tgt_port_y = _compute_directional_ports(
            sx,
            sy,
            sw,
            sh,
            tx,
            ty,
            tw,
            th,
            direction,
            port_style,
            out_rank,
            out_total,
            in_rank,
            in_total,
        )
        src_port_x, src_port_y, tgt_port_x, tgt_port_y = _reverse_back_edge_ports(
            sx,
            sy,
            sw,
            sh,
            tx,
            ty,
            tw,
            th,
            direction,
            src_port_x,
            src_port_y,
            tgt_port_x,
            tgt_port_y,
        )

        # Shape-aware port adjustment
        if node_shapes is not None:
            src_port_x, src_port_y = _adjust_port_for_shape(
                node_shapes[s], sx, sy, sw, sh, src_port_x, src_port_y, is_source=True
            )
            tgt_port_x, tgt_port_y = _adjust_port_for_shape(
                node_shapes[t], tx, ty, tw, th, tgt_port_x, tgt_port_y, is_source=False
            )

        # Per-edge routing and curvature
        routing = edge_style.routing if edge_style is not None else "bezier"
        curvature = edge_style.curvature if edge_style is not None else 0.4
        pair_key = (int(s), int(t))
        if pair_counts.get(pair_key, 0) > 1 and pair_ranks.get(e_idx, 0) % 2 == 1:
            # Parallel edges should fan across both sides of the chord, matching
            # Graphviz's alternating spline lanes instead of stacking one-sided arcs.
            curvature = -curvature

        # Port angular spread (r80-S7#2): bias the initial/final tangent by
        # each port's rank among its peers on the same node face, so
        # adjacent edges separate visually instead of leaving as a
        # near-parallel bundle. Sign convention matches the existing
        # neighbor-position sort (out_order/in_order) -- this only adds a
        # secondary angular nudge on top of that primary ordering.
        # r80-S7b#3: the 46-deg budget is scaled down per node in dense
        # neighborhoods, where a wide fan buys port-angle score but pays
        # more in edge-edge crossings.
        src_bias_deg = _port_spread_bias_deg(
            out_rank, out_total, max_spread_deg=46.0 * spread_scales[s]
        )
        tgt_bias_deg = _port_spread_bias_deg(
            in_rank, in_total, max_spread_deg=46.0 * spread_scales[t]
        )

        curve = _compute_curve(
            src_port_x,
            src_port_y,
            tgt_port_x,
            tgt_port_y,
            direction,
            routing,
            curvature,
            src_bias_deg,
            tgt_bias_deg,
        )

        # Cluster-aware deflection: if the curve crosses a foreign cluster bbox,
        # push control points to route around it
        if graph is not None and cluster_bboxes:
            curve = _deflect_around_clusters(
                curve,
                s,
                t,
                node_cluster_set,
                cluster_bboxes,
                direction,
            )

        # Baseline for the crossing-aware referee: the pre-S7 route (zero
        # tangent bias, no node avoidance) with the same pre-existing
        # cluster deflection applied. Only rebuilt when the tangent bias
        # actually changed this edge; otherwise the pre-avoidance curve
        # IS the baseline.
        base_curve = curve
        if routing == "bezier" and (src_bias_deg != 0.0 or tgt_bias_deg != 0.0):
            base_curve = _compute_curve(
                src_port_x,
                src_port_y,
                tgt_port_x,
                tgt_port_y,
                direction,
                routing,
                curvature,
            )
            if graph is not None and cluster_bboxes:
                base_curve = _deflect_around_clusters(
                    base_curve,
                    s,
                    t,
                    node_cluster_set,
                    cluster_bboxes,
                    direction,
                )

        # Node-bbox avoidance: deflect bezier control points around any
        # non-endpoint node the curve passes through. Bezier-only (r80-S7#1);
        # ON by default, per-edge opt-out via EdgeStyle.avoid_nodes.
        avoid_nodes = edge_style.avoid_nodes if edge_style is not None else True
        if avoid_nodes and routing == "bezier" and num_nodes > 2:
            curve = _deflect_around_nodes(
                curve,
                s,
                t,
                node_grid,
                grid_cell_size,
                x_coords,
                y_coords,
                widths,
                heights,
            )

        # r80-S7b#2: crossing-aware acceptance (greedy monotone referee).
        # If the S7 modifications (tangent bias and/or node deflection)
        # changed this edge, keep them only when they do not create net
        # new edge-edge crossings against the edges already routed --
        # same referee philosophy as the placement portfolio, applied per
        # edge. Deterministic; edges are judged in index order.
        if routing == "bezier" and (curve.cp1 != base_curve.cp1 or curve.cp2 != base_curve.cp2):
            cand_poly = _curve_polyline_samples(curve, sample_count=_REFEREE_SAMPLES)
            cand_bbox = _poly_bbox(cand_poly)
            base_poly = _curve_polyline_samples(base_curve, sample_count=_REFEREE_SAMPLES)
            base_bbox = _poly_bbox(base_poly)
            base_crossings = _count_route_crossings(
                base_poly, base_bbox, routed_polylines, routed_bboxes
            )
            cand_crossings = _count_route_crossings(
                cand_poly, cand_bbox, routed_polylines, routed_bboxes, stop_above=base_crossings
            )
            if cand_crossings > base_crossings:
                curve = base_curve
                accepted_poly, accepted_bbox = base_poly, base_bbox
            else:
                accepted_poly, accepted_bbox = cand_poly, cand_bbox
        else:
            accepted_poly = _curve_polyline_samples(curve, sample_count=_REFEREE_SAMPLES)
            accepted_bbox = _poly_bbox(accepted_poly)

        routed_polylines.append(accepted_poly)
        routed_bboxes.append(accepted_bbox)
        curves.append(curve)

    return curves


def _get_route_edge_style(graph: Any, edge_index: int) -> Any:
    """Return an edge style for routing, accepting dict overrides.

    Parameters
    ----------
    graph : Any
        Graph object with ``edge_styles`` and ``get_style_for_edge``.
    edge_index : int
        Edge index to resolve.

    Returns
    -------
    Any
        EdgeStyle-like object with routing, curvature, and port fields.
    """
    raw_styles = getattr(graph, "edge_styles", [])
    if edge_index < len(raw_styles) and isinstance(raw_styles[edge_index], dict):
        from dagua.styles import EdgeStyle

        style = EdgeStyle(**raw_styles[edge_index])
        raw_styles[edge_index] = style
        return style
    return graph.get_style_for_edge(edge_index)


def _adjust_port_for_shape(
    shape: str,
    cx: float,
    cy: float,
    w: float,
    h: float,
    port_x: float,
    port_y: float,
    is_source: bool,
) -> Tuple[float, float]:
    """Adjust port position to lie on the shape boundary.

    For rectangles/roundrects, ports are already on the bounding box edge — no adjustment.
    For ellipses/circles and semicircle variants, project onto the curved boundary.
    For diamonds, project onto the diamond edge.
    """
    if shape in ("rect", "roundrect"):
        return port_x, port_y

    if shape in ("ellipse", "circle"):
        # Project port onto ellipse boundary
        a = w / 2  # semi-major (horizontal)
        b = h / 2  # semi-minor (vertical)
        if a < 1e-6 or b < 1e-6:
            return port_x, port_y

        # Direction from center to port
        dx = port_x - cx
        dy = port_y - cy
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1e-6:
            # Default: use top/bottom center
            return cx, cy + (b if is_source else -b)

        # Parametric angle
        angle = math.atan2(dy / b, dx / a)
        return cx + a * math.cos(angle), cy + b * math.sin(angle)

    if shape in {
        "semicircle",
        "semicircle_up",
        "semicircle_down",
        "semicircle_left",
        "semicircle_right",
    }:
        from dagua.render.edges.intersection import ray_semicircle_intersection

        dx = port_x - cx
        dy = port_y - cy
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return cx, cy + (h / 2 if is_source else -h / 2)

        orientation = "up"
        if shape == "semicircle_down":
            orientation = "down"
        elif shape == "semicircle_left":
            orientation = "left"
        elif shape == "semicircle_right":
            orientation = "right"

        hit = ray_semicircle_intersection(
            center=[cx, cy],
            half_size=[w / 2, h / 2],
            orientation=orientation,
            ray_origin=[cx, cy],
            ray_direction=[dx, dy],
        )
        return float(hit[0]), float(hit[1])

    if shape == "diamond":
        # Diamond edges: 4 sides connecting top/right/bottom/left
        # Project port onto nearest diamond edge
        dx = port_x - cx
        dy = port_y - cy

        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return cx, cy + (h / 2 if is_source else -h / 2)

        # Normalize to diamond coordinates
        # Diamond boundary: |dx|/(w/2) + |dy|/(h/2) = 1
        hw, hh = w / 2, h / 2
        scale = abs(dx) / hw + abs(dy) / hh
        if scale < 1e-6:
            return port_x, port_y

        return cx + dx / scale, cy + dy / scale

    # For all other polygon shapes (triangle, hexagon, pentagon, octagon,
    # star, parallelogram, trapezoid), use ray-polygon intersection to
    # project the port onto the actual shape boundary.
    _POLYGON_SHAPES = {
        "triangle",
        "hexagon",
        "pentagon",
        "octagon",
        "star",
        "parallelogram",
        "trapezoid",
    }
    # Non-convex shapes where arrowheads can enter concavities.
    _CONCAVE_SHAPES = {"star"}
    if shape in _POLYGON_SHAPES:
        from dagua.render.edges.intersection import ray_polygon_intersection

        dx = port_x - cx
        dy = port_y - cy
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return cx, cy + (h / 2 if is_source else -h / 2)

        hit = ray_polygon_intersection(
            center=[cx, cy],
            half_size=[w / 2, h / 2],
            shape=shape,
            ray_origin=[cx, cy],
            ray_direction=[dx, dy],
        )
        hx, hy = float(hit[0]), float(hit[1])
        # For concave shapes, push the port slightly outward so arrowheads
        # don't extend into interior concavities.
        if shape in _CONCAVE_SHAPES:
            bdx, bdy = hx - cx, hy - cy
            dist = math.sqrt(bdx * bdx + bdy * bdy)
            if dist > 1e-6:
                outward = min(w, h) * 0.06
                hx += bdx / dist * outward
                hy += bdy / dist * outward
        return hx, hy

    return port_x, port_y


def _compute_curve(
    sx: float,
    sy: float,
    tx: float,
    ty: float,
    direction: str = "TB",
    routing: str = "bezier",
    curvature: float = 0.4,
    src_tangent_bias_deg: float = 0.0,
    tgt_tangent_bias_deg: float = 0.0,
) -> BezierCurve:
    """Compute an edge curve for the requested routing mode.

    Parameters
    ----------
    sx : float
        Source x-coordinate.
    sy : float
        Source y-coordinate.
    tx : float
        Target x-coordinate.
    ty : float
        Target y-coordinate.
    direction : str, default="TB"
        Layout flow direction. Supported values are ``"TB"``, ``"BT"``,
        ``"LR"``, and ``"RL"``.
    routing : str, default="bezier"
        Routing mode. Supported values are ``"bezier"``, ``"straight"``,
        ``"ortho"``, and ``"taxi"``.
    curvature : float, default=0.4
        Curvature factor for bezier routing.
    src_tangent_bias_deg : float, default=0.0
        Extra rotation (degrees) applied to the initial control point around
        the source port, used to fan out adjacent edges leaving a shared
        port face (port angular spread, r80-S7#2). Bezier routing only.
    tgt_tangent_bias_deg : float, default=0.0
        Same rotation applied to the final control point around the target
        port.

    Returns
    -------
    BezierCurve
        Cubic bezier control points representing the routed edge.
    """
    if routing == "straight":
        return _compute_straight(sx, sy, tx, ty)
    if routing == "ortho":
        return _compute_ortho(sx, sy, tx, ty, direction)
    if routing == "taxi":
        return _compute_taxi(sx, sy, tx, ty, direction)
    return _compute_bezier(
        sx, sy, tx, ty, direction, curvature, src_tangent_bias_deg, tgt_tangent_bias_deg
    )


def _compute_straight(
    sx: float,
    sy: float,
    tx: float,
    ty: float,
) -> BezierCurve:
    """Straight line: control points = endpoints (degenerate bezier)."""
    return BezierCurve((sx, sy), (sx, sy), (tx, ty), (tx, ty), routing="straight")


def _compute_ortho(
    sx: float,
    sy: float,
    tx: float,
    ty: float,
    direction: str = "TB",
) -> BezierCurve:
    """Route an edge through one Manhattan elbow corridor."""
    if direction in ("TB", "BT"):
        mid_y = (sy + ty) / 2
        return _polyline_curve(
            [(sx, sy), (sx, mid_y), (tx, mid_y), (tx, ty)],
            routing="ortho",
            direction=direction,
        )

    mid_x = (sx + tx) / 2
    return _polyline_curve(
        [(sx, sy), (mid_x, sy), (mid_x, ty), (tx, ty)],
        routing="ortho",
        direction=direction,
    )


def _compute_taxi(
    sx: float,
    sy: float,
    tx: float,
    ty: float,
    direction: str = "TB",
) -> BezierCurve:
    """Compute a Manhattan-style taxi curve between two endpoints.

    Parameters
    ----------
    sx : float
        Source x-coordinate.
    sy : float
        Source y-coordinate.
    tx : float
        Target x-coordinate.
    ty : float
        Target y-coordinate.
    direction : str, default="TB"
        Layout flow direction. Vertical flows route vertical-then-horizontal;
        horizontal flows route horizontal-then-vertical.

    Returns
    -------
    BezierCurve
        Degenerate cubic bezier whose control points form an L-shaped or
        Z-shaped Manhattan route while preserving the existing bezier
        renderer interface.
    """
    if math.isclose(sx, tx, abs_tol=1e-9) and math.isclose(sy, ty, abs_tol=1e-9):
        return _polyline_curve(
            [(sx, sy)],
            routing="taxi",
            direction=direction,
            step_fraction=0.35,
        )

    step_fraction = 0.35
    if direction in ("TB", "BT"):
        first_y = sy + (ty - sy) * step_fraction
        second_y = sy + (ty - sy) * (1.0 - step_fraction)
        mid_x = (sx + tx) / 2.0
        return _polyline_curve(
            [
                (sx, sy),
                (sx, first_y),
                (mid_x, first_y),
                (mid_x, second_y),
                (tx, second_y),
                (tx, ty),
            ],
            routing="taxi",
            direction=direction,
            step_fraction=step_fraction,
        )

    first_x = sx + (tx - sx) * step_fraction
    second_x = sx + (tx - sx) * (1.0 - step_fraction)
    mid_y = (sy + ty) / 2.0
    return _polyline_curve(
        [
            (sx, sy),
            (first_x, sy),
            (first_x, mid_y),
            (second_x, mid_y),
            (second_x, ty),
            (tx, ty),
        ],
        routing="taxi",
        direction=direction,
        step_fraction=step_fraction,
    )


def _compute_bezier(
    sx: float,
    sy: float,
    tx: float,
    ty: float,
    direction: str = "TB",
    curvature: float = 0.4,
    src_tangent_bias_deg: float = 0.0,
    tgt_tangent_bias_deg: float = 0.0,
) -> BezierCurve:
    """Compute cubic bezier control points based on edge geometry.

    curvature controls the offset factor: 0=straight, 1=maximum curve.
    src_tangent_bias_deg/tgt_tangent_bias_deg rotate the initial/final
    control points around their respective ports to fan adjacent edges
    apart (port angular spread, r80-S7#2); 0 reproduces prior behavior.
    """
    dx = tx - sx
    dy = ty - sy
    dist = (dx**2 + dy**2) ** 0.5

    curvature_sign = -1.0 if curvature < 0.0 else 1.0
    curvature_magnitude = abs(float(curvature))

    if dist < 1e-6 or curvature_magnitude < 1e-6:
        return BezierCurve((sx, sy), (sx, sy), (tx, ty), (tx, ty), routing="bezier")

    abs_dx = abs(dx)
    abs_dy = abs(dy)

    # Vertical flow (TB/BT): control points offset in y
    if direction in ("TB", "BT"):
        if abs_dx < abs_dy * 0.3:
            # Nearly vertical: gentle S-curve
            offset = abs_dy * curvature_magnitude * 0.75
            cp1 = (sx, sy + offset)
            cp2 = (tx, ty - offset)
        elif dy > 0:
            # Normal downward edge: smooth bezier
            offset_y = abs_dy * curvature_magnitude
            cp1 = (sx, sy + offset_y)
            cp2 = (tx, ty - offset_y)
        else:
            # Back edge (upward): arc perpendicular to the chord so
            # the curve is visibly bowed even at high curvature values.
            perp_x = -dy / dist  # perpendicular unit vector
            perp_y = dx / dist
            # Choose the side that arcs away from the main flow.
            side = 1.0 if perp_x >= 0 else -1.0
            offset = dist * min(curvature_magnitude, 2.0) * 0.45 + 30.0
            cp1 = (sx + side * perp_x * offset, sy + side * perp_y * offset)
            cp2 = (tx + side * perp_x * offset, ty + side * perp_y * offset)
    else:
        # Horizontal flow (LR/RL)
        if abs_dy < abs_dx * 0.3:
            offset = abs_dx * curvature_magnitude * 0.75
            cp1 = (sx + offset, sy)
            cp2 = (tx - offset, ty)
        else:
            offset_x = abs_dx * curvature_magnitude
            cp1 = (sx + offset_x, sy)
            cp2 = (tx - offset_x, ty)

    if curvature_sign < 0.0:
        cp1 = _reflect_point_across_line(cp1, (sx, sy), (tx, ty))
        cp2 = _reflect_point_across_line(cp2, (sx, sy), (tx, ty))

    # r80-S7b: the rank-based bias assumes "positive rank offset = tilt
    # toward larger neighbor x". Whether that means a CW or CCW rotation
    # depends on which way the local tangent points, so derive the sign
    # from the tangent itself (frame-independent). The original S7 code
    # applied a fixed sign, which INVERTED the fan on up-going tangents:
    # adjacent edges rotated toward each other and crossed right after
    # leaving the node -- a direct source of the S7 dgrX regression.
    if src_tangent_bias_deg:
        sign = _spread_rotation_sign((cp1[0] - sx, cp1[1] - sy))
        if sign != 0.0:
            cp1 = _rotate_point_around(cp1, (sx, sy), src_tangent_bias_deg * sign)
    if tgt_tangent_bias_deg:
        sign = _spread_rotation_sign((cp2[0] - tx, cp2[1] - ty))
        if sign != 0.0:
            cp2 = _rotate_point_around(cp2, (tx, ty), tgt_tangent_bias_deg * sign)

    return BezierCurve((sx, sy), cp1, cp2, (tx, ty), routing="bezier", direction=direction)


def _spread_rotation_sign(tangent: Tuple[float, float]) -> float:
    """Return the rotation-sign multiplier that makes a POSITIVE rank bias
    tilt a tangent toward LARGER neighbor coordinate.

    For a CCW rotation by ``theta``, the first-order displacement of a
    tangent ``(vx, vy)`` is ``(-vy, vx) * theta``. Ports are ranked by
    neighbor x (and distributed low-to-high along the face), so:

    - vertical-ish tangents (|vy| >= |vx|): rank spread acts on x; moving
      x positive under positive bias needs ``theta * (-vy) > 0`` -> sign
      is ``-sign(vy)``.
    - horizontal-ish tangents: rank spread acts on y (ports run bottom to
      top along the vertical face); moving y positive needs
      ``theta * vx > 0`` -> sign is ``sign(vx)``.

    Parameters
    ----------
    tangent : tuple[float, float]
        Direction from the port toward its adjacent control point.

    Returns
    -------
    float
        ``+1.0``, ``-1.0``, or ``0.0`` for a degenerate tangent.
    """
    vx, vy = tangent
    if abs(vy) >= abs(vx):
        if vy > 0:
            return -1.0
        return 1.0 if vy < 0 else 0.0
    return 1.0 if vx > 0 else -1.0


def _rotate_point_around(
    point: Tuple[float, float],
    pivot: Tuple[float, float],
    degrees: float,
) -> Tuple[float, float]:
    """Rotate ``point`` around ``pivot`` by ``degrees`` (counter-clockwise).

    Parameters
    ----------
    point : tuple[float, float]
        Point to rotate.
    pivot : tuple[float, float]
        Center of rotation.
    degrees : float
        Rotation angle in degrees.

    Returns
    -------
    tuple[float, float]
        Rotated point.
    """
    if abs(degrees) < 1e-9:
        return point
    rad = math.radians(degrees)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    dx = point[0] - pivot[0]
    dy = point[1] - pivot[1]
    return (
        pivot[0] + dx * cos_a - dy * sin_a,
        pivot[1] + dx * sin_a + dy * cos_a,
    )


def _port_spread_bias_deg(
    rank: int,
    total: int,
    max_spread_deg: float = 46.0,
) -> float:
    """Return a deterministic tangent-rotation bias for one port among peers.

    Ports sharing a node face are ranked (see the neighbor-position sort in
    :func:`route_edges`); this spreads their initial tangent angles evenly
    around 0 so adjacent edges separate visually (port angular spread,
    r80-S7#2) while preserving the existing rank order (crossing-reduction
    property untouched -- this only adds a secondary angular nudge).

    Parameters
    ----------
    rank : int
        This port's rank among its ``total`` peers (0-indexed).
    total : int
        Number of ports sharing the same node face.
    max_spread_deg : float, default=46.0
        Total angular spread budget across all ranked ports; matches the
        upper end of dot's observed 10-46 deg port angular resolution.

    Returns
    -------
    float
        Bias angle in degrees, 0.0 when there is nothing to spread
        (``total <= 1``).
    """
    if total <= 1:
        return 0.0
    frac = (rank - (total - 1) / 2.0) / (total - 1)
    return frac * max_spread_deg


def _reflect_point_across_line(
    point: Tuple[float, float],
    line_start: Tuple[float, float],
    line_end: Tuple[float, float],
) -> Tuple[float, float]:
    """Reflect a point across a line segment's infinite supporting line.

    Parameters
    ----------
    point : tuple[float, float]
        Point to mirror.
    line_start : tuple[float, float]
        First point on the mirror line.
    line_end : tuple[float, float]
        Second point on the mirror line.

    Returns
    -------
    tuple[float, float]
        Reflected point. Degenerate mirror lines return ``point`` unchanged.
    """
    px, py = point
    ax, ay = line_start
    bx, by = line_end
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return point

    t = ((px - ax) * dx + (py - ay) * dy) / denom
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return (2.0 * proj_x - px, 2.0 * proj_y - py)


def evaluate_bezier(curve: BezierCurve, t: float) -> Tuple[float, float]:
    """Evaluate a routed curve at parameter ``t`` in ``[0, 1]``."""
    if curve.waypoints is not None:
        return _polyline_point_at(curve.waypoints, t)
    p0, p1, p2, p3 = curve.p0, curve.cp1, curve.cp2, curve.p1
    u = 1 - t
    x = u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0]
    y = u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1]
    return (x, y)


def bezier_tangent(curve: BezierCurve, t: float) -> Tuple[float, float]:
    """Compute the local tangent vector at parameter ``t``."""
    if curve.waypoints is not None:
        return _polyline_tangent_at(curve.waypoints, t)
    p0, p1, p2, p3 = curve.p0, curve.cp1, curve.cp2, curve.p1
    u = 1 - t
    dx = 3 * u**2 * (p1[0] - p0[0]) + 6 * u * t * (p2[0] - p1[0]) + 3 * t**2 * (p3[0] - p2[0])
    dy = 3 * u**2 * (p1[1] - p0[1]) + 6 * u * t * (p2[1] - p1[1]) + 3 * t**2 * (p3[1] - p2[1])
    return (dx, dy)


def _deflect_around_clusters(
    curve: BezierCurve,
    src_idx: int,
    tgt_idx: int,
    node_cluster_set: dict,
    cluster_bboxes: dict,
    direction: str,
    margin: float = 12.0,
) -> BezierCurve:
    """Deflect bezier control points to avoid crossing foreign cluster bboxes.

    A cluster is "foreign" if neither the source nor target node belongs to it.
    For each foreign cluster whose bbox the straight-line path would cross,
    push the control points to route around the nearest side of the bbox.
    """
    if curve.waypoints is not None:
        return curve

    src_clusters = node_cluster_set.get(src_idx, set())
    tgt_clusters = node_cluster_set.get(tgt_idx, set())
    own_clusters = src_clusters | tgt_clusters

    # Sample the curve at several points to detect crossings
    p0, p1 = curve.p0, curve.p1
    cp1, cp2 = list(curve.cp1), list(curve.cp2)
    modified = False

    for cname, (bx_min, by_min, bx_max, by_max) in cluster_bboxes.items():
        if cname in own_clusters:
            continue  # skip clusters the edge belongs to

        # Check if the midpoint or quarter-points of the curve fall inside this bbox
        crossings = []
        for t in [0.25, 0.5, 0.75]:
            pt = evaluate_bezier(curve, t)
            if bx_min <= pt[0] <= bx_max and by_min <= pt[1] <= by_max:
                crossings.append(t)

        if not crossings:
            continue

        # Determine which side to route around (closest edge of bbox to the midpoint)
        mid = evaluate_bezier(curve, 0.5)
        cx_mid = (bx_min + bx_max) / 2
        cy_mid = (by_min + by_max) / 2

        # Calculate distance to each side from the line connecting src to tgt
        if direction in ("TB", "BT"):
            # Prefer routing around left or right side
            if mid[0] < cx_mid:
                # Route around left side
                deflect_x = bx_min - margin
            else:
                # Route around right side
                deflect_x = bx_max + margin
            cp1[0] = deflect_x
            cp2[0] = deflect_x
        else:
            # Horizontal layout: route around top or bottom
            if mid[1] < cy_mid:
                deflect_y = by_min - margin
            else:
                deflect_y = by_max + margin
            cp1[1] = deflect_y
            cp2[1] = deflect_y

        modified = True

    if modified:
        return BezierCurve(p0, (cp1[0], cp1[1]), (cp2[0], cp2[1]), p1)
    return curve


def _curve_samples_hit_rect(curve: BezierCurve, rect: Rect, sample_count: int = 9) -> bool:
    """Return whether interior curve samples fall inside a rectangle.

    Endpoints (t=0, t=1) are intentionally excluded since they legitimately
    sit on the source/target node boundary.
    """
    for i in range(1, sample_count):
        t = i / sample_count
        pt = evaluate_bezier(curve, t)
        if _point_in_rect(pt, rect):
            return True
    return False


def _build_node_grid(
    x_coords: Sequence[float],
    y_coords: Sequence[float],
    cell_size: float,
) -> Dict[Tuple[int, int], List[int]]:
    """Bucket node indices into a uniform grid for fast neighborhood queries.

    Parameters
    ----------
    x_coords : sequence[float]
        Node center X coordinates.
    y_coords : sequence[float]
        Node center Y coordinates.
    cell_size : float
        Grid cell edge length in data coordinates.

    Returns
    -------
    dict[tuple[int, int], list[int]]
        Mapping from grid cell to the node indices whose center falls in it.
    """
    grid: Dict[Tuple[int, int], List[int]] = {}
    for idx, (x, y) in enumerate(zip(x_coords, y_coords)):
        cell = (int(math.floor(x / cell_size)), int(math.floor(y / cell_size)))
        grid.setdefault(cell, []).append(idx)
    return grid


def _grid_candidates(
    bbox: Rect,
    grid: Dict[Tuple[int, int], List[int]],
    cell_size: float,
) -> List[int]:
    """Return deduplicated node indices whose grid cell overlaps ``bbox``.

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        Query bounds as ``(x_min, y_min, x_max, y_max)``.
    grid : dict[tuple[int, int], list[int]]
        Spatial grid built by :func:`_build_node_grid`.
    cell_size : float
        Grid cell edge length matching the grid's construction.

    Returns
    -------
    list[int]
        Candidate node indices near the query box.
    """
    x_min, y_min, x_max, y_max = bbox
    cx_min = int(math.floor(x_min / cell_size))
    cx_max = int(math.floor(x_max / cell_size))
    cy_min = int(math.floor(y_min / cell_size))
    cy_max = int(math.floor(y_max / cell_size))
    seen: Set[int] = set()
    for cx in range(cx_min, cx_max + 1):
        for cy in range(cy_min, cy_max + 1):
            for idx in grid.get((cx, cy), ()):
                seen.add(idx)
    return list(seen)


def _local_density_spread_scales(
    node_grid: Dict[Tuple[int, int], List[int]],
    cell_size: float,
    x_coords: Sequence[float],
    y_coords: Sequence[float],
    sparse_count: float = 4.0,
    floor: float = 0.3,
) -> List[float]:
    """Per-node scale factors that shrink the port-spread budget in dense areas.

    r80-S7b#3: the full 46-deg fan is safe in roomy layouts (external
    dot/elk positions) but creates edge-edge crossings in dagua's compact
    corridors. Scale each node's spread budget by local crowding: count
    neighbors in the 3x3 grid-cell block around the node (cell size is the
    mean node diagonal, so this is roughly a 1.5-diagonal radius) and
    shrink as ``sqrt(sparse_count / n_local)`` below a sparsity threshold
    of ``sparse_count`` neighbors, with a hard floor so the fan never
    fully collapses.

    Parameters
    ----------
    node_grid : dict[tuple[int, int], list[int]]
        Spatial grid over node centers (see :func:`_build_node_grid`).
    cell_size : float
        Grid cell edge length used to build ``node_grid``.
    x_coords, y_coords : sequence[float]
        Node center coordinates, indexed by node id.
    sparse_count : float, default=4.0
        Neighbor count at or below which the full budget applies.
    floor : float, default=0.3
        Minimum scale in the densest neighborhoods.

    Returns
    -------
    list[float]
        Scale factor in ``[floor, 1.0]`` per node.
    """
    scales: List[float] = []
    for x, y in zip(x_coords, y_coords):
        cx = int(math.floor(x / cell_size))
        cy = int(math.floor(y / cell_size))
        n_local = -1  # exclude the node itself
        for gx in range(cx - 1, cx + 2):
            for gy in range(cy - 1, cy + 2):
                n_local += len(node_grid.get((gx, gy), ()))
        if n_local <= sparse_count:
            scales.append(1.0)
        else:
            scales.append(max(floor, (sparse_count / n_local) ** 0.5))
    return scales


def _deflect_around_nodes(
    curve: BezierCurve,
    src_idx: int,
    tgt_idx: int,
    node_grid: Dict[Tuple[int, int], List[int]],
    grid_cell_size: float,
    x_coords: Sequence[float],
    y_coords: Sequence[float],
    widths: Sequence[float],
    heights: Sequence[float],
    margin: float = 4.0,
    max_attempts: int = 4,
) -> BezierCurve:
    """Deflect bezier control points around non-endpoint node bboxes.

    Generalizes :func:`_deflect_around_clusters` to arbitrary chord
    directions: pushes both control points perpendicular to the src->tgt
    chord, away from whichever node the curve is passing through, growing
    the offset over a bounded number of attempts. Deterministic (no RNG).
    In dense neighborhoods where no bounded attempt clears the box, the
    curve is left as-is for that node rather than looping forever.

    Parameters
    ----------
    curve : BezierCurve
        Routed curve to adjust. Non-bezier (waypoint) curves pass through
        unmodified -- ortho/taxi routing is out of scope for this pass.
    src_idx : int
        Source node index (excluded from avoidance).
    tgt_idx : int
        Target node index (excluded from avoidance).
    node_grid : dict[tuple[int, int], list[int]]
        Spatial grid over all node centers (see :func:`_build_node_grid`).
    grid_cell_size : float
        Grid cell size used to build ``node_grid``.
    x_coords, y_coords, widths, heights : sequence[float]
        Per-node geometry, indexed by node id.
    margin : float, default=4.0
        Inflation added around each node bbox before intersection testing.
    max_attempts : int, default=4
        Number of growing-offset deflection attempts per blocking node
        before giving up on that node (dense-neighborhood fallback).

    Returns
    -------
    BezierCurve
        Curve with control points deflected around blocking node bboxes,
        or the original curve when no clearing deflection was found/needed.
    """
    if curve.waypoints is not None:
        return curve

    p0, p1 = curve.p0, curve.p1
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    chord_len = math.hypot(dx, dy)
    if chord_len < 1e-6:
        return curve

    cp1 = list(curve.cp1)
    cp2 = list(curve.cp2)

    # Candidate nodes: grid cells overlapping the control polygon's bbox,
    # inflated by margin plus a generous max-node-radius pad.
    poly_xs = [p0[0], cp1[0], cp2[0], p1[0]]
    poly_ys = [p0[1], cp1[1], cp2[1], p1[1]]
    pad = grid_cell_size + margin
    query_bbox = (
        min(poly_xs) - pad,
        min(poly_ys) - pad,
        max(poly_xs) + pad,
        max(poly_ys) + pad,
    )
    candidates = _grid_candidates(query_bbox, node_grid, grid_cell_size)

    perp_x, perp_y = -dy / chord_len, dx / chord_len
    modified = False

    for node_idx in candidates:
        if node_idx == src_idx or node_idx == tgt_idx:
            continue
        w, h = widths[node_idx], heights[node_idx]
        if w <= 0.0 and h <= 0.0:
            continue
        cx, cy = x_coords[node_idx], y_coords[node_idx]
        rect = (cx - w / 2 - margin, cy - h / 2 - margin, cx + w / 2 + margin, cy + h / 2 + margin)

        current = BezierCurve(p0, (cp1[0], cp1[1]), (cp2[0], cp2[1]), p1)
        if not _curve_samples_hit_rect(current, rect):
            continue

        side = (cx - p0[0]) * perp_x + (cy - p0[1]) * perp_y
        push_sign = -1.0 if side >= 0.0 else 1.0
        base_offset = max(w, h) / 2.0 + margin
        # A uniform two-control-point push only displaces the curve by
        # 3t(1-t)*offset at parameter t, which shrinks toward the chord
        # near the curve's endpoints -- so obstacles that sit close to
        # either endpoint need much larger offsets to actually clear.
        # r80-S7b#1: chord-length-scaled cap. The offset may never exceed
        # a fixed fraction of the chord: an offset comparable to or larger
        # than the chord makes the curve loop back on itself (the lasso
        # curls seen on short cluster-boundary edges in the S7 render
        # review). Short edges therefore get proportionally small nudges;
        # if the capped ladder cannot clear the box, the fallback below
        # leaves the edge unchanged (bounded attempts, never loops).
        growth = (2.0, 4.5, 9.0, 16.0)
        max_offset = chord_len * 0.6

        last_offset = None
        for attempt in range(max_attempts):
            factor = growth[min(attempt, len(growth) - 1)]
            offset = min(base_offset * factor, max_offset)
            if offset == last_offset:
                break  # ladder saturated at the chord cap; retrying is futile
            last_offset = offset
            trial_cp1 = (cp1[0] + push_sign * perp_x * offset, cp1[1] + push_sign * perp_y * offset)
            trial_cp2 = (cp2[0] + push_sign * perp_x * offset, cp2[1] + push_sign * perp_y * offset)
            trial = BezierCurve(p0, trial_cp1, trial_cp2, p1)
            if not _curve_samples_hit_rect(trial, rect):
                cp1, cp2 = list(trial_cp1), list(trial_cp2)
                modified = True
                break
        # else: dense neighborhood -- no clearing offset found within the
        # attempt budget; leave the edge as-is for this node and move on
        # (never loop forever).

    if modified:
        return BezierCurve(
            p0,
            (cp1[0], cp1[1]),
            (cp2[0], cp2[1]),
            p1,
            routing=curve.routing,
            direction=curve.direction,
        )
    return curve


def _segments_cross(
    a0: Tuple[float, float],
    a1: Tuple[float, float],
    b0: Tuple[float, float],
    b1: Tuple[float, float],
    eps: float = 1e-9,
) -> bool:
    """Return whether two segments intersect strictly in their interiors.

    Endpoint contacts (t or u at 0/1, e.g. two edges sharing a port) and
    parallel overlaps are NOT counted -- this mirrors how a human reads a
    drawing: touching at a shared node is not a crossing.

    Parameters
    ----------
    a0, a1 : tuple[float, float]
        First segment endpoints.
    b0, b1 : tuple[float, float]
        Second segment endpoints.
    eps : float, default=1e-9
        Interior-strictness margin on both parameters.

    Returns
    -------
    bool
        ``True`` when the segments properly cross.
    """
    d1x = a1[0] - a0[0]
    d1y = a1[1] - a0[1]
    d2x = b1[0] - b0[0]
    d2y = b1[1] - b0[1]
    denom = d1x * d2y - d1y * d2x
    if abs(denom) < eps:
        return False
    dx = b0[0] - a0[0]
    dy = b0[1] - a0[1]
    t = (dx * d2y - dy * d2x) / denom
    u = (dx * d1y - dy * d1x) / denom
    return eps < t < 1.0 - eps and eps < u < 1.0 - eps


def _poly_bbox(poly: Sequence[Tuple[float, float]]) -> Rect:
    """Return the axis-aligned bounding box of a polyline.

    Parameters
    ----------
    poly : sequence[tuple[float, float]]
        Polyline points.

    Returns
    -------
    tuple[float, float, float, float]
        Bounds as ``(x_min, y_min, x_max, y_max)``.
    """
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (min(xs), min(ys), max(xs), max(ys))


def _count_route_crossings(
    poly: Sequence[Tuple[float, float]],
    poly_bbox: Rect,
    routed_polylines: Sequence[Sequence[Tuple[float, float]]],
    routed_bboxes: Sequence[Rect],
    stop_above: Optional[int] = None,
) -> int:
    """Count segment crossings between one polyline and already-routed edges.

    Used by the r80-S7b#2 crossing-aware acceptance referee in
    :func:`route_edges`. AABB reject per routed edge keeps the common case
    cheap; ``stop_above`` allows early exit as soon as the count exceeds
    the competing variant's count (the comparison outcome is then decided).

    Parameters
    ----------
    poly : sequence[tuple[float, float]]
        Candidate polyline.
    poly_bbox : tuple[float, float, float, float]
        Pre-computed bbox of ``poly``.
    routed_polylines : sequence[sequence[tuple[float, float]]]
        Polylines of already-accepted edges.
    routed_bboxes : sequence[tuple[float, float, float, float]]
        Their bboxes, index-aligned.
    stop_above : int, optional
        Early-exit threshold: return as soon as the count exceeds it.

    Returns
    -------
    int
        Number of properly-crossing segment pairs found (possibly truncated
        at ``stop_above + 1`` when early exit triggers).
    """
    count = 0
    lx0, ly0, lx1, ly1 = poly_bbox
    n_seg = len(poly) - 1
    for other_poly, (bx0, by0, bx1, by1) in zip(routed_polylines, routed_bboxes):
        if bx1 < lx0 or bx0 > lx1 or by1 < ly0 or by0 > ly1:
            continue
        n_other = len(other_poly) - 1
        for i in range(n_seg):
            for j in range(n_other):
                if _segments_cross(poly[i], poly[i + 1], other_poly[j], other_poly[j + 1]):
                    count += 1
                    if stop_above is not None and count > stop_above:
                        return count
    return count


def _curve_polyline_samples(
    curve: BezierCurve, sample_count: int = 10
) -> List[Tuple[float, float]]:
    """Return an explicit polyline approximation of a routed curve.

    Parameters
    ----------
    curve : BezierCurve
        Curve to sample.
    sample_count : int, default=10
        Number of samples for bezier curves. Waypoint (ortho/taxi) curves
        return their exact vertices instead of resampling.

    Returns
    -------
    list[tuple[float, float]]
        Ordered points approximating the curve.
    """
    if curve.waypoints is not None:
        return [(float(p[0]), float(p[1])) for p in curve.waypoints]
    return [evaluate_bezier(curve, i / (sample_count - 1)) for i in range(sample_count)]


def _label_path_crossings(
    label_bbox: Rect,
    owner_edge_idx: int,
    curve_polylines: Sequence[Sequence[Tuple[float, float]]],
    curve_path_bboxes: Sequence[Rect],
) -> int:
    """Count how many OTHER edges' routed paths cut through a label box.

    The label's own edge is excluded -- a label sitting on its own curve is
    expected, not a collision.

    Parameters
    ----------
    label_bbox : tuple[float, float, float, float]
        Candidate label bounds as ``(x_min, y_min, x_max, y_max)``.
    owner_edge_idx : int
        Index of the edge this label belongs to (skipped).
    curve_polylines : sequence[sequence[tuple[float, float]]]
        Pre-sampled polyline per edge (see :func:`_curve_polyline_samples`).
    curve_path_bboxes : sequence[tuple[float, float, float, float]]
        Pre-computed bbox per polyline, for a cheap AABB reject.

    Returns
    -------
    int
        Number of other edges whose path crosses ``label_bbox``.
    """
    lx0, ly0, lx1, ly1 = label_bbox
    crossings = 0
    for other_idx, poly in enumerate(curve_polylines):
        if other_idx == owner_edge_idx:
            continue
        bx0, by0, bx1, by1 = curve_path_bboxes[other_idx]
        if bx1 < lx0 or bx0 > lx1 or by1 < ly0 or by0 > ly1:
            continue
        if polyline_intersect_rect(poly, label_bbox) is not None:
            crossings += 1
    return crossings


def place_edge_labels(
    curves: List[BezierCurve],
    positions: torch.Tensor,
    node_sizes: torch.Tensor,
    edge_labels: List[Optional[str]],
    graph: Optional[Any] = None,
) -> List[Optional[Tuple[float, float]]]:
    """Compute collision-avoiding positions for edge labels.

    Algorithm (greedy):
    1. For each labeled edge, evaluate bezier at style.label_position
    2. Compute label bbox and offset perpendicular to curve tangent
    3. Check collisions with node bboxes and previously placed labels
    4. If collision, try alternate t values then larger perpendicular offsets
    5. Pick position with minimum overlap

    Returns list of (x, y) per edge, or None for unlabeled edges.
    """
    from dagua.utils import measure_text_fallback

    result: List[Optional[Tuple[float, float]]] = [None] * len(curves)

    if not any(edge_labels):
        return result

    pos = positions.detach().cpu()
    sizes = node_sizes.detach().cpu()
    n = pos.shape[0]

    # Pre-compute node bboxes: (x_min, y_min, x_max, y_max)
    node_bboxes = []
    for i in range(n):
        hw, hh = sizes[i, 0].item() / 2, sizes[i, 1].item() / 2
        cx, cy = pos[i, 0].item(), pos[i, 1].item()
        node_bboxes.append((cx - hw, cy - hh, cx + hw, cy + hh))

    # r80-S7#4: pre-sample every edge's path once so label candidates can be
    # scored against label-vs-edge-path overlap too (previously only
    # label-vs-node and label-vs-label were scored). Coarse per-curve bbox
    # enables a cheap AABB reject before the exact polyline check.
    curve_polylines: List[List[Tuple[float, float]]] = []
    curve_path_bboxes: List[Rect] = []
    for curve in curves:
        # 20 samples (vs the 10-point default) so a label-sized box can't
        # fall entirely between two consecutive samples on a long, nearly
        # straight edge and go undetected.
        poly = _curve_polyline_samples(curve, sample_count=20)
        curve_polylines.append(poly)
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        curve_path_bboxes.append((min(xs), min(ys), max(xs), max(ys)))

    placed_bboxes: List[Tuple[float, float, float, float]] = []

    for e_idx, curve in enumerate(curves):
        if e_idx >= len(edge_labels) or not edge_labels[e_idx]:
            continue

        label_text = edge_labels[e_idx]
        assert label_text is not None
        style = graph.get_style_for_edge(e_idx) if graph is not None else None
        label_t = style.label_position if style is not None else 0.5
        font_size = style.label_font_size if style is not None else 7.0
        label_offset = style.label_offset if style is not None else 8.0
        label_side = style.label_side if style is not None else "auto"
        label_avoidance = style.label_avoidance if style is not None else True

        # Measure label
        lw, lh = measure_text_fallback(label_text, font_size)
        lw += 4.0  # padding
        lh += 2.0

        best_pos = None
        best_overlap = float("inf")

        # r80-S7#4: widened t-offset ladder (was 5 candidates: 0, +-0.1,
        # +-0.2) so labels have more positions along the curve to try
        # before settling for a collision.
        t_offsets = (
            [0.0, 0.08, -0.08, 0.16, -0.16, 0.28, -0.28, 0.4, -0.4] if label_avoidance else [0.0]
        )
        perp_scales = _label_offset_candidates(label_offset, allow_search=label_avoidance)
        side_signs = _label_side_candidates(label_side, allow_search=label_avoidance)

        # Try candidate positions
        for t_offset in t_offsets:
            t = max(0.05, min(0.95, label_t + t_offset))
            mx, my = evaluate_bezier(curve, t)

            # Perpendicular offset from tangent
            tdx, tdy = bezier_tangent(curve, t)
            tmag = (tdx**2 + tdy**2) ** 0.5
            if tmag < 1e-6:
                perp_x, perp_y = 0.0, 1.0
            else:
                perp_x, perp_y = -tdy / tmag, tdx / tmag

            for side_sign in side_signs:
                for perp_scale in perp_scales:
                    cx = mx + perp_x * perp_scale * side_sign
                    cy = my + perp_y * perp_scale * side_sign

                    # Label bbox
                    lx0 = cx - lw / 2
                    ly0 = cy - lh / 2
                    lx1 = cx + lw / 2
                    ly1 = cy + lh / 2

                    # Count overlap with node bboxes
                    overlap = 0.0
                    for nb in node_bboxes:
                        ox = max(0.0, min(lx1, nb[2]) - max(lx0, nb[0]))
                        oy = max(0.0, min(ly1, nb[3]) - max(ly0, nb[1]))
                        overlap += ox * oy

                    # Count overlap with previously placed labels
                    for pb in placed_bboxes:
                        ox = max(0.0, min(lx1, pb[2]) - max(lx0, pb[0]))
                        oy = max(0.0, min(ly1, pb[3]) - max(ly0, pb[1]))
                        overlap += ox * oy

                    # r80-S7#4: penalize label-vs-edge-path overlap (other
                    # edges' routed curves cutting through this label),
                    # scaled to the label's own area so one path crossing
                    # costs roughly as much as a full label-vs-node overlap.
                    label_bbox = (lx0, ly0, lx1, ly1)
                    path_crossings = _label_path_crossings(
                        label_bbox, e_idx, curve_polylines, curve_path_bboxes
                    )
                    overlap += path_crossings * lw * lh

                    if overlap < best_overlap:
                        best_overlap = overlap
                        best_pos = (cx, cy)
                        best_bbox = (lx0, ly0, lx1, ly1)

                    if overlap == 0.0:
                        break
                if best_overlap == 0.0:
                    break
            if best_overlap == 0.0:
                break

        if best_pos is not None:
            result[e_idx] = best_pos
            placed_bboxes.append(best_bbox)

    return result


def preferred_edge_label_position(
    curve: BezierCurve,
    label_position: float = 0.5,
    label_offset: float = 8.0,
    label_side: str = "auto",
) -> Tuple[float, float]:
    """Return the preferred label anchor before collision-avoidance search."""
    t = max(0.05, min(0.95, label_position))
    mx, my = evaluate_bezier(curve, t)
    tdx, tdy = bezier_tangent(curve, t)
    tmag = (tdx**2 + tdy**2) ** 0.5
    if tmag < 1e-6:
        perp_x, perp_y = 0.0, 1.0
    else:
        perp_x, perp_y = -tdy / tmag, tdx / tmag
    side_sign = _label_side_candidates(label_side, allow_search=False)[0]
    return mx + perp_x * label_offset * side_sign, my + perp_y * label_offset * side_sign


def edge_endpoint_label_position(
    curve: BezierCurve,
    endpoint: str,
    label_offset: float = 5.0,
) -> Tuple[float, float]:
    """Return a label anchor near the source or target end of a bezier edge.

    Parameters
    ----------
    curve : BezierCurve
        Cubic bezier describing the routed edge centerline.
    endpoint : str
        Endpoint selector. Supported values are ``"head"`` and ``"tail"``.
    label_offset : float, default=5.0
        Distance in data coordinates to move away from the node along the local
        tangent direction so the label does not sit directly on the boundary.

    Returns
    -------
    tuple[float, float]
        Label anchor in data coordinates.

    Raises
    ------
    ValueError
        If ``endpoint`` is not ``"head"`` or ``"tail"``.
    """
    if endpoint not in {"head", "tail"}:
        raise ValueError(f"Unsupported edge endpoint label target: {endpoint!r}")

    t = 0.9 if endpoint == "head" else 0.1
    x, y = evaluate_bezier(curve, t)
    tdx, tdy = bezier_tangent(curve, t)
    magnitude = (tdx**2 + tdy**2) ** 0.5
    if magnitude < 1e-6:
        return x, y

    direction_x = tdx / magnitude
    direction_y = tdy / magnitude
    if endpoint == "head":
        return x - direction_x * label_offset, y - direction_y * label_offset
    return x + direction_x * label_offset, y + direction_y * label_offset


def _label_side_candidates(label_side: str, allow_search: bool) -> List[float]:
    if label_side == "left":
        return [1.0]
    if label_side == "right":
        return [-1.0]
    return [1.0, -1.0] if allow_search else [1.0]


def _label_offset_candidates(label_offset: float, allow_search: bool) -> List[float]:
    base = max(1.0, float(label_offset))
    if not allow_search:
        return [base]
    # r80-S7#4: widened perpendicular-nudge ladder (was 3 candidates) so
    # dense graphs have more room to dodge nodes/labels/edge paths before
    # falling back to the highest-overlap candidate.
    return [
        base,
        max(4.0, base * 1.5),
        max(2.0, base * 0.5),
        max(6.0, base * 2.25),
        max(1.0, base * 0.25),
    ]
