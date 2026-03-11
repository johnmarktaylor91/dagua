"""Heuristic bezier edge routing — compute control points after layout.

For each edge, computes cubic bezier control points (p0, cp1, cp2, p1)
based on the geometry of the source and target positions.
Supports per-edge routing modes (bezier, straight, ortho) and
shape-aware port positioning (ellipse, diamond, rectangle).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass
class BezierCurve:
    """A cubic bezier curve defined by 4 control points."""

    p0: Tuple[float, float]
    cp1: Tuple[float, float]
    cp2: Tuple[float, float]
    p1: Tuple[float, float]


def route_edges(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    direction: str = "TB",
    graph: Optional[object] = None,
) -> List[BezierCurve]:
    """Compute bezier control points for all edges.

    Args:
        positions: [N, 2] node positions
        edge_index: [2, E] source-target pairs
        node_sizes: [N, 2] node widths and heights
        direction: layout direction for port placement
        graph: optional DaguaGraph for per-edge routing and per-node shape

    Returns:
        List of BezierCurve objects, one per edge.
    """
    if edge_index.numel() == 0:
        return []

    pos = positions.detach().cpu()
    sizes = node_sizes.detach().cpu()
    num_edges = edge_index.shape[1]
    src_indices = edge_index[0].tolist()
    tgt_indices = edge_index[1].tolist()

    # Compute port positions for each node
    # Count outgoing and incoming edges per node
    out_count = {}
    in_count = {}
    out_order = {}
    in_order = {}

    for e_idx in range(num_edges):
        s, t = src_indices[e_idx], tgt_indices[e_idx]
        out_count[s] = out_count.get(s, 0) + 1
        in_count[t] = in_count.get(t, 0) + 1

    # Track port assignment order (sort by target/source x position)
    out_edges = {}  # node -> [(edge_idx, target_x)]
    in_edges = {}  # node -> [(edge_idx, source_x)]
    for e_idx in range(num_edges):
        s, t = src_indices[e_idx], tgt_indices[e_idx]
        out_edges.setdefault(s, []).append((e_idx, pos[t, 0].item()))
        in_edges.setdefault(t, []).append((e_idx, pos[s, 0].item()))

    # Sort ports by connected node x-position to reduce crossings
    for node in out_edges:
        out_edges[node].sort(key=lambda x: x[1])
        for rank, (e_idx, _) in enumerate(out_edges[node]):
            out_order[e_idx] = (rank, len(out_edges[node]))

    for node in in_edges:
        in_edges[node].sort(key=lambda x: x[1])
        for rank, (e_idx, _) in enumerate(in_edges[node]):
            in_order[e_idx] = (rank, len(in_edges[node]))

    curves = []
    for e_idx in range(num_edges):
        s, t = src_indices[e_idx], tgt_indices[e_idx]
        sx, sy = pos[s, 0].item(), pos[s, 1].item()
        tx, ty = pos[t, 0].item(), pos[t, 1].item()
        sw, sh = sizes[s, 0].item(), sizes[s, 1].item()
        tw, th = sizes[t, 0].item(), sizes[t, 1].item()

        # Per-edge style
        edge_style = None
        if graph is not None:
            edge_style = graph.get_style_for_edge(e_idx)

        # Port positions
        out_rank, out_total = out_order.get(e_idx, (0, 1))
        in_rank, in_total = in_order.get(e_idx, (0, 1))

        # Port style: "center" puts all ports at node center
        port_style = edge_style.port_style if edge_style is not None else "distributed"
        if port_style == "center":
            src_port_x = sx
            src_port_y = sy + sh / 2
            tgt_port_x = tx
            tgt_port_y = ty - th / 2
        else:
            # Distributed: spread across node edge
            src_port_x = sx - sw / 2 + sw * (out_rank + 0.5) / out_total
            src_port_y = sy + sh / 2  # bottom
            tgt_port_x = tx - tw / 2 + tw * (in_rank + 0.5) / in_total
            tgt_port_y = ty - th / 2  # top

        # Shape-aware port adjustment
        if graph is not None:
            src_port_x, src_port_y = _adjust_port_for_shape(
                graph, s, sx, sy, sw, sh, src_port_x, src_port_y, is_source=True
            )
            tgt_port_x, tgt_port_y = _adjust_port_for_shape(
                graph, t, tx, ty, tw, th, tgt_port_x, tgt_port_y, is_source=False
            )

        # Per-edge routing and curvature
        routing = edge_style.routing if edge_style is not None else "bezier"
        curvature = edge_style.curvature if edge_style is not None else 0.4

        curve = _compute_curve(src_port_x, src_port_y, tgt_port_x, tgt_port_y, direction, routing, curvature)
        curves.append(curve)

    return curves


def _adjust_port_for_shape(
    graph, node_idx: int,
    cx: float, cy: float, w: float, h: float,
    port_x: float, port_y: float,
    is_source: bool,
) -> Tuple[float, float]:
    """Adjust port position to lie on the shape boundary.

    For rectangles/roundrects, ports are already on the bounding box edge — no adjustment.
    For ellipses/circles, project onto the ellipse curve.
    For diamonds, project onto the diamond edge.
    """
    style = graph.get_style_for_node(node_idx)
    shape = style.shape

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

    return port_x, port_y


def _compute_curve(
    sx: float, sy: float,
    tx: float, ty: float,
    direction: str = "TB",
    routing: str = "bezier",
    curvature: float = 0.4,
) -> BezierCurve:
    """Compute curve based on routing mode."""
    if routing == "straight":
        return _compute_straight(sx, sy, tx, ty)
    elif routing == "ortho":
        return _compute_ortho(sx, sy, tx, ty, direction)
    else:
        return _compute_bezier(sx, sy, tx, ty, direction, curvature)


def _compute_straight(
    sx: float, sy: float,
    tx: float, ty: float,
) -> BezierCurve:
    """Straight line: control points = endpoints (degenerate bezier)."""
    return BezierCurve((sx, sy), (sx, sy), (tx, ty), (tx, ty))


def _compute_ortho(
    sx: float, sy: float,
    tx: float, ty: float,
    direction: str = "TB",
) -> BezierCurve:
    """Right-angle routing via midpoint control points."""
    if direction in ("TB", "BT"):
        mid_y = (sy + ty) / 2
        return BezierCurve((sx, sy), (sx, mid_y), (tx, mid_y), (tx, ty))
    else:
        mid_x = (sx + tx) / 2
        return BezierCurve((sx, sy), (mid_x, sy), (mid_x, ty), (tx, ty))


def _compute_bezier(
    sx: float, sy: float,
    tx: float, ty: float,
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
        return BezierCurve((sx, sy), (sx, sy), (tx, ty), (tx, ty))

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
            # Back edge (upward): wide arc to the side
            arc_width = abs_dy * curvature * 1.25 + abs_dx * 0.3 + 30
            side = 1 if dx >= 0 else -1
            mid_y = (sy + ty) / 2
            cp1 = (sx + side * arc_width, mid_y)
            cp2 = (tx + side * arc_width, mid_y)
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

    return BezierCurve((sx, sy), cp1, cp2, (tx, ty))


def evaluate_bezier(curve: BezierCurve, t: float) -> Tuple[float, float]:
    """Evaluate cubic bezier at parameter t in [0, 1]."""
    p0, p1, p2, p3 = curve.p0, curve.cp1, curve.cp2, curve.p1
    u = 1 - t
    x = u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0]
    y = u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1]
    return (x, y)


def bezier_tangent(curve: BezierCurve, t: float) -> Tuple[float, float]:
    """Compute tangent vector at parameter t."""
    p0, p1, p2, p3 = curve.p0, curve.cp1, curve.cp2, curve.p1
    u = 1 - t
    dx = 3 * u**2 * (p1[0] - p0[0]) + 6 * u * t * (p2[0] - p1[0]) + 3 * t**2 * (p3[0] - p2[0])
    dy = 3 * u**2 * (p1[1] - p0[1]) + 6 * u * t * (p2[1] - p1[1]) + 3 * t**2 * (p3[1] - p2[1])
    return (dx, dy)


def place_edge_labels(
    curves: List[BezierCurve],
    positions: torch.Tensor,
    node_sizes: torch.Tensor,
    edge_labels: List[Optional[str]],
    graph: Optional[object] = None,
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
        style = graph.get_style_for_edge(e_idx) if graph is not None else None
        label_t = style.label_position if style is not None else 0.5
        font_size = style.label_font_size if style is not None else 7.0

        # Measure label
        lw, lh = measure_text_fallback(label_text, font_size)
        lw += 4.0  # padding
        lh += 2.0

        best_pos = None
        best_overlap = float("inf")

        # Try candidate positions
        for t_offset in [0.0, 0.1, -0.1, 0.2, -0.2]:
            t = max(0.05, min(0.95, label_t + t_offset))
            mx, my = evaluate_bezier(curve, t)

            # Perpendicular offset from tangent
            tdx, tdy = bezier_tangent(curve, t)
            tmag = (tdx**2 + tdy**2) ** 0.5
            if tmag < 1e-6:
                perp_x, perp_y = 0.0, 1.0
            else:
                perp_x, perp_y = -tdy / tmag, tdx / tmag

            for perp_scale in [4.0, 8.0, 12.0]:
                cx = mx + perp_x * perp_scale
                cy = my + perp_y * perp_scale

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

        if best_pos is not None:
            result[e_idx] = best_pos
            placed_bboxes.append(best_bbox)

    return result
