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

    curves = []
    for e_idx in range(num_edges):
        s, t = src_indices[e_idx], tgt_indices[e_idx]
        sx, sy = x_coords[s], y_coords[s]
        tx, ty = x_coords[t], y_coords[t]
        sw, sh = widths[s], heights[s]
        tw, th = widths[t], heights[t]

        # Self-loops stay on the outward-facing side for the layout direction.
        if s == t:
            curves.append(_compute_self_loop_curve(sx, sy, sw, sh, direction))
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

        curve = _compute_curve(
            src_port_x,
            src_port_y,
            tgt_port_x,
            tgt_port_y,
            direction,
            routing,
            curvature,
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
    return _compute_bezier(sx, sy, tx, ty, direction, curvature)


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
) -> BezierCurve:
    """Compute cubic bezier control points based on edge geometry.

    curvature controls the offset factor: 0=straight, 1=maximum curve.
    """
    dx = tx - sx
    dy = ty - sy
    dist = (dx**2 + dy**2) ** 0.5

    if dist < 1e-6 or curvature < 1e-6:
        return BezierCurve((sx, sy), (sx, sy), (tx, ty), (tx, ty), routing="bezier")

    abs_dx = abs(dx)
    abs_dy = abs(dy)

    # Vertical flow (TB/BT): control points offset in y
    if direction in ("TB", "BT"):
        if abs_dx < abs_dy * 0.3:
            # Nearly vertical: gentle S-curve
            offset = abs_dy * curvature * 0.75
            cp1 = (sx, sy + offset)
            cp2 = (tx, ty - offset)
        elif dy > 0:
            # Normal downward edge: smooth bezier
            offset_y = abs_dy * curvature
            cp1 = (sx, sy + offset_y)
            cp2 = (tx, ty - offset_y)
        else:
            # Back edge (upward): arc perpendicular to the chord so
            # the curve is visibly bowed even at high curvature values.
            perp_x = -dy / dist  # perpendicular unit vector
            perp_y = dx / dist
            # Choose the side that arcs away from the main flow.
            side = 1.0 if perp_x >= 0 else -1.0
            offset = dist * min(curvature, 2.0) * 0.45 + 30.0
            cp1 = (sx + side * perp_x * offset, sy + side * perp_y * offset)
            cp2 = (tx + side * perp_x * offset, ty + side * perp_y * offset)
    else:
        # Horizontal flow (LR/RL)
        if abs_dy < abs_dx * 0.3:
            offset = abs_dx * curvature * 0.75
            cp1 = (sx + offset, sy)
            cp2 = (tx - offset, ty)
        else:
            offset_x = abs_dx * curvature
            cp1 = (sx + offset_x, sy)
            cp2 = (tx - offset_x, ty)

    return BezierCurve((sx, sy), cp1, cp2, (tx, ty), routing="bezier", direction=direction)


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

        t_offsets = [0.0, 0.1, -0.1, 0.2, -0.2] if label_avoidance else [0.0]
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
    return [base, max(4.0, base * 1.5), max(2.0, base * 0.5)]
