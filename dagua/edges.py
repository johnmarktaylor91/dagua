"""Heuristic bezier edge routing — compute control points after layout.

For each edge, computes cubic bezier control points (p0, cp1, cp2, p1)
based on the geometry of the source and target positions.
"""

from __future__ import annotations

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
) -> List[BezierCurve]:
    """Compute bezier control points for all edges.

    Args:
        positions: [N, 2] node positions
        edge_index: [2, E] source-target pairs
        node_sizes: [N, 2] node widths and heights
        direction: layout direction for port placement

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

        # Port positions (distributed along node edge)
        out_rank, out_total = out_order.get(e_idx, (0, 1))
        in_rank, in_total = in_order.get(e_idx, (0, 1))

        # Source port: bottom of node (outgoing)
        src_port_x = sx - sw / 2 + sw * (out_rank + 0.5) / out_total
        src_port_y = sy + sh / 2  # bottom

        # Target port: top of node (incoming)
        tgt_port_x = tx - tw / 2 + tw * (in_rank + 0.5) / in_total
        tgt_port_y = ty - th / 2  # top

        curve = _compute_bezier(src_port_x, src_port_y, tgt_port_x, tgt_port_y, direction)
        curves.append(curve)

    return curves


def _compute_bezier(
    sx: float, sy: float,
    tx: float, ty: float,
    direction: str = "TB",
) -> BezierCurve:
    """Compute cubic bezier control points based on edge geometry."""
    dx = tx - sx
    dy = ty - sy
    dist = (dx**2 + dy**2) ** 0.5

    if dist < 1e-6:
        return BezierCurve((sx, sy), (sx, sy), (tx, ty), (tx, ty))

    abs_dx = abs(dx)
    abs_dy = abs(dy)

    # Vertical flow (TB/BT): control points offset in y
    if direction in ("TB", "BT"):
        if abs_dx < abs_dy * 0.3:
            # Nearly vertical: gentle S-curve
            offset = abs_dy * 0.3
            cp1 = (sx, sy + offset)
            cp2 = (tx, ty - offset)
        elif dy > 0:
            # Normal downward edge: smooth bezier
            offset_y = abs_dy * 0.4
            cp1 = (sx, sy + offset_y)
            cp2 = (tx, ty - offset_y)
        else:
            # Back edge (upward): wide arc to the side
            arc_width = abs_dy * 0.5 + abs_dx * 0.3 + 30
            side = 1 if dx >= 0 else -1
            mid_y = (sy + ty) / 2
            cp1 = (sx + side * arc_width, mid_y)
            cp2 = (tx + side * arc_width, mid_y)
    else:
        # Horizontal flow (LR/RL)
        if abs_dy < abs_dx * 0.3:
            offset = abs_dx * 0.3
            cp1 = (sx + offset, sy)
            cp2 = (tx - offset, ty)
        else:
            offset_x = abs_dx * 0.4
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
