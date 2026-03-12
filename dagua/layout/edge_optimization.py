"""Differentiable edge optimization — gradient descent on bezier control points.

Optimizes edge routing by minimizing a weighted combination of:
1. Edge-edge crossings (sigmoid-relaxed proxy)
2. Edge-node crossings (proximity penalty to node bboxes)
3. Port angular resolution (penalize small angles between incident edges)
4. Curvature consistency (variance of κ across edges)
5. Curvature penalty (prefer straighter paths)

Data representation:
- Control points: [E, 2, 2] tensor (cp1 and cp2 per edge, requires_grad)
- Endpoints: [E, 2, 2] tensor (p0 and p1, fixed from port positions)
- Bezier evaluation: [E, T, 2] for T sample points per curve
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch

from dagua.edges import BezierCurve


def _build_cluster_data_for_edge_opt(graph, positions, node_sizes):
    """Pre-compute cluster bboxes and node membership for edge optimization.

    Returns:
        cluster_bboxes: [C, 4] tensor (x_min, y_min, x_max, y_max) per cluster
        node_cluster_mask: [N, C] bool tensor — True if node i is in cluster c
        None, None if no clusters
    """
    if graph is None or not hasattr(graph, 'clusters') or not graph.clusters:
        return None, None

    from dagua.utils import collect_cluster_leaves

    pos = positions.detach().cpu()
    sizes = node_sizes.detach().cpu()
    N = pos.shape[0]
    cluster_names = sorted(graph.clusters.keys())
    C = len(cluster_names)
    if C == 0:
        return None, None

    bboxes = torch.zeros(C, 4)  # [C, 4]
    node_mask = torch.zeros(N, C, dtype=torch.bool)

    for ci, cname in enumerate(cluster_names):
        members = graph.clusters[cname]
        if isinstance(members, dict):
            members = collect_cluster_leaves(members)
        if not members:
            continue
        valid = [m for m in members if 0 <= m < N]
        if not valid:
            continue

        idx = torch.tensor(valid, dtype=torch.long)
        cstyle = graph.get_style_for_cluster(cname)
        pad = cstyle.padding

        member_pos = pos[idx]
        member_sizes = sizes[idx]
        half_sizes = member_sizes / 2

        bboxes[ci, 0] = (member_pos[:, 0] - half_sizes[:, 0]).min() - pad
        bboxes[ci, 1] = (member_pos[:, 1] - half_sizes[:, 1]).min() - pad
        bboxes[ci, 2] = (member_pos[:, 0] + half_sizes[:, 0]).max() + pad
        bboxes[ci, 3] = (member_pos[:, 1] + half_sizes[:, 1]).max() + pad

        node_mask[idx, ci] = True

    return bboxes, node_mask


def optimize_edges(
    curves: List[BezierCurve],
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    config,
    graph: Optional[object] = None,
) -> List[BezierCurve]:
    """Optimize bezier control points via gradient descent.

    Args:
        curves: initial BezierCurve list from heuristic routing
        positions: [N, 2] node positions
        edge_index: [2, E] edge pairs
        node_sizes: [N, 2] node widths and heights
        config: LayoutConfig with edge_opt_* fields
        graph: optional DaguaGraph

    Returns:
        Optimized list of BezierCurve objects.
    """
    E = len(curves)
    if E == 0:
        return curves

    steps = getattr(config, "edge_opt_steps", 0)
    if steps < 0:
        # Explicit skip (-1)
        return curves
    if steps == 0:
        # Auto-scale based on edge count
        steps = min(100, max(20, E * 2))

    lr = getattr(config, "edge_opt_lr", 0.1)

    # Extract endpoints and control points from curves
    endpoints = torch.zeros(E, 2, 2)  # [E, 2(p0/p1), 2(xy)]
    cp = torch.zeros(E, 2, 2)  # [E, 2(cp1/cp2), 2(xy)]

    for i, c in enumerate(curves):
        endpoints[i, 0, 0] = c.p0[0]
        endpoints[i, 0, 1] = c.p0[1]
        endpoints[i, 1, 0] = c.p1[0]
        endpoints[i, 1, 1] = c.p1[1]
        cp[i, 0, 0] = c.cp1[0]
        cp[i, 0, 1] = c.cp1[1]
        cp[i, 1, 0] = c.cp2[0]
        cp[i, 1, 1] = c.cp2[1]

    cp = cp.requires_grad_(True)
    pos = positions.detach().cpu().float()
    sizes = node_sizes.detach().cpu().float()

    # Loss weights from config
    w_crossing = getattr(config, "w_edge_crossing", 5.0)
    w_node_crossing = getattr(config, "w_edge_node_crossing", 10.0)
    w_angular = getattr(config, "w_edge_angular_res", 2.0)
    w_curv_consistency = getattr(config, "w_edge_curvature_consistency", 1.0)
    w_curv_penalty = getattr(config, "w_edge_curvature_penalty", 0.5)
    w_cluster_crossing = getattr(config, "w_edge_cluster_crossing", 8.0)

    # Pre-compute cluster data for edge-cluster crossing loss
    cluster_bboxes, node_cluster_mask = _build_cluster_data_for_edge_opt(
        graph, positions, node_sizes
    )

    # Pre-compute incident edge indices per node for angular resolution
    ei = edge_index.detach().cpu()
    src_list = ei[0].tolist()
    tgt_list = ei[1].tolist()

    optimizer = torch.optim.Adam([cp], lr=lr)

    # Pre-compute t samples for bezier evaluation
    T = 10
    t_samples = torch.linspace(0.0, 1.0, T).unsqueeze(0)  # [1, T]

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)

        # Evaluate bezier curves at T points: [E, T, 2]
        points = _evaluate_bezier_batch(endpoints, cp, t_samples)

        total_loss = torch.tensor(0.0)

        # 1. Edge crossing loss
        if w_crossing > 0 and E > 1:
            total_loss = total_loss + w_crossing * _edge_crossing_loss(points, E)

        # 2. Edge-node crossing loss
        if w_node_crossing > 0:
            total_loss = total_loss + w_node_crossing * _edge_node_crossing_loss(
                points, pos, sizes, src_list, tgt_list, E
            )

        # 3. Port angular resolution loss
        if w_angular > 0:
            total_loss = total_loss + w_angular * _port_angular_resolution_loss(
                endpoints, cp, src_list, tgt_list
            )

        # 4. Curvature consistency loss
        if w_curv_consistency > 0:
            total_loss = total_loss + w_curv_consistency * _curvature_consistency_loss(
                endpoints, cp, t_samples
            )

        # 5. Curvature penalty loss
        if w_curv_penalty > 0:
            total_loss = total_loss + w_curv_penalty * _curvature_penalty_loss(
                endpoints, cp, t_samples
            )

        # 6. Edge-cluster crossing loss
        if w_cluster_crossing > 0 and cluster_bboxes is not None:
            total_loss = total_loss + w_cluster_crossing * _edge_cluster_crossing_loss(
                points, cluster_bboxes, node_cluster_mask, src_list, tgt_list, E
            )

        if total_loss.requires_grad:
            total_loss.backward()
            # Replace NaN/Inf gradients with zero before stepping
            if cp.grad is not None and not torch.isfinite(cp.grad).all():
                cp.grad = torch.where(torch.isfinite(cp.grad), cp.grad, torch.zeros_like(cp.grad))
            torch.nn.utils.clip_grad_norm_([cp], max_norm=50.0)
            optimizer.step()
            # Clamp control points to stay within reasonable range of endpoints
            with torch.no_grad():
                if not torch.isfinite(cp).all():
                    # Revert NaN/Inf control points to linear interpolation
                    fallback_cp1 = endpoints[:, 0, :] * 0.667 + endpoints[:, 1, :] * 0.333
                    fallback_cp2 = endpoints[:, 0, :] * 0.333 + endpoints[:, 1, :] * 0.667
                    mask = ~torch.isfinite(cp[:, 0, :])
                    cp[:, 0, :] = torch.where(mask, fallback_cp1, cp[:, 0, :])
                    mask = ~torch.isfinite(cp[:, 1, :])
                    cp[:, 1, :] = torch.where(mask, fallback_cp2, cp[:, 1, :])

    # Convert back to BezierCurve list (with final NaN safety check)
    cp_final = cp.detach()
    if not torch.isfinite(cp_final).all():
        # Optimization diverged — return original curves unchanged
        return curves

    result = []
    for i in range(E):
        result.append(BezierCurve(
            p0=(endpoints[i, 0, 0].item(), endpoints[i, 0, 1].item()),
            cp1=(cp_final[i, 0, 0].item(), cp_final[i, 0, 1].item()),
            cp2=(cp_final[i, 1, 0].item(), cp_final[i, 1, 1].item()),
            p1=(endpoints[i, 1, 0].item(), endpoints[i, 1, 1].item()),
        ))

    return result


def _evaluate_bezier_batch(
    endpoints: torch.Tensor,
    cp: torch.Tensor,
    t_samples: torch.Tensor,
) -> torch.Tensor:
    """Evaluate cubic bezier curves at sample points.

    B(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3

    Args:
        endpoints: [E, 2, 2] — P0 and P3
        cp: [E, 2, 2] — P1 (cp1) and P2 (cp2)
        t_samples: [1, T] — parameter values

    Returns:
        [E, T, 2] — evaluated points
    """
    t = t_samples.unsqueeze(-1)  # [1, T, 1]
    u = 1.0 - t

    p0 = endpoints[:, 0:1, :]  # [E, 1, 2]
    p3 = endpoints[:, 1:2, :]  # [E, 1, 2]
    p1 = cp[:, 0:1, :]  # [E, 1, 2]
    p2 = cp[:, 1:2, :]  # [E, 1, 2]

    points = (
        u**3 * p0 +
        3 * u**2 * t * p1 +
        3 * u * t**2 * p2 +
        t**3 * p3
    )
    return points  # [E, T, 2]


def _edge_crossing_loss(points: torch.Tensor, E: int) -> torch.Tensor:
    """Sigmoid-relaxed edge crossing proxy.

    Samples T points per curve, creates piecewise linear segments,
    checks pairs for crossings via soft intersection test.
    For E > 5K, sample edge pairs.
    """
    # Segments: [E, T-1, 2] start and end
    T = points.shape[1]
    seg_start = points[:, :-1, :]  # [E, T-1, 2]
    seg_end = points[:, 1:, :]  # [E, T-1, 2]

    S = T - 1  # segments per edge

    # For large E, sample pairs
    max_pairs = 5000
    if E * (E - 1) // 2 > max_pairs:
        idx1 = torch.randint(0, E, (max_pairs,))
        idx2 = torch.randint(0, E, (max_pairs,))
        valid = idx1 < idx2
        idx1 = idx1[valid]
        idx2 = idx2[valid]
    else:
        # All pairs
        idx = torch.triu_indices(E, E, offset=1)
        idx1, idx2 = idx[0], idx[1]

    if idx1.numel() == 0:
        return torch.tensor(0.0)

    # For each pair of edges, check one representative segment pair (midpoint segments)
    mid_s = S // 2
    p1 = seg_start[idx1, mid_s]  # [P, 2]
    p2 = seg_end[idx1, mid_s]  # [P, 2]
    p3 = seg_start[idx2, mid_s]  # [P, 2]
    p4 = seg_end[idx2, mid_s]  # [P, 2]

    # Soft crossing via signed area test
    d1 = p2 - p1
    d2 = p4 - p3
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]

    d3 = p3 - p1
    # Clamp both sides to avoid division by near-zero (cross can be negative)
    safe_cross = cross.sign() * cross.abs().clamp(min=1e-6)
    t = (d3[:, 0] * d2[:, 1] - d3[:, 1] * d2[:, 0]) / safe_cross
    u = (d3[:, 0] * d1[:, 1] - d3[:, 1] * d1[:, 0]) / safe_cross

    # Sigmoid relaxation: crossing when both t,u in (0,1)
    sharpness = 10.0
    t_in = torch.sigmoid(sharpness * t) * torch.sigmoid(sharpness * (1 - t))
    u_in = torch.sigmoid(sharpness * u) * torch.sigmoid(sharpness * (1 - u))
    soft_crossing = t_in * u_in

    return soft_crossing.mean()


def _edge_node_crossing_loss(
    points: torch.Tensor,
    pos: torch.Tensor,
    sizes: torch.Tensor,
    src_list: list,
    tgt_list: list,
    E: int,
) -> torch.Tensor:
    """Penalty for edge sample points close to unrelated node bboxes.

    Uses relu(safety_margin - manhattan_dist_to_bbox)^2.
    """
    N = pos.shape[0]
    T = points.shape[1]
    safety_margin = 3.0

    # For efficiency, only check nearby nodes via spatial proximity
    # Flatten all sample points: [E*T, 2]
    flat_points = points.reshape(-1, 2)  # [E*T, 2]

    # Node bbox half-sizes
    half_w = sizes[:, 0] / 2  # [N]
    half_h = sizes[:, 1] / 2

    # For each sample point, distance to each node bbox
    # This is O(E*T*N) — use spatial hashing for large N
    if N > 500:
        # Sample subset of nodes
        sample_n = min(500, N)
        node_idx = torch.randperm(N)[:sample_n]
    else:
        node_idx = torch.arange(N)
        sample_n = N

    # Build source/target lookup for each edge's sample points
    edge_of_point = torch.arange(E).unsqueeze(1).expand(-1, T).reshape(-1)  # [E*T]

    # [E*T, sample_n]
    px = flat_points[:, 0:1]  # [E*T, 1]
    py = flat_points[:, 1:2]

    nx = pos[node_idx, 0:1].t()  # [1, sample_n]
    ny = pos[node_idx, 1:2].t()
    nhw = half_w[node_idx].unsqueeze(0)  # [1, sample_n]
    nhh = half_h[node_idx].unsqueeze(0)

    # Manhattan distance from point to bbox boundary (negative = inside)
    dx = (px - nx).abs() - nhw  # [E*T, sample_n]
    dy = (py - ny).abs() - nhh

    # Distance to bbox: max(dx, 0) + max(dy, 0) for outside;
    # for points inside, both dx and dy are negative
    dist_to_bbox = torch.relu(dx) + torch.relu(dy)

    # Mask out edges' own source/target nodes
    src_nodes = torch.tensor(src_list)  # [E]
    tgt_nodes = torch.tensor(tgt_list)  # [E]

    # Build mask: for each point, mask its edge's src and tgt
    edge_src = src_nodes[edge_of_point]  # [E*T]
    edge_tgt = tgt_nodes[edge_of_point]

    # Check if each node in sample is the src/tgt of each point's edge
    is_own_src = (edge_src.unsqueeze(1) == node_idx.unsqueeze(0))  # [E*T, sample_n]
    is_own_tgt = (edge_tgt.unsqueeze(1) == node_idx.unsqueeze(0))
    is_own = is_own_src | is_own_tgt

    # Penalty: relu(safety - dist)^2, zeroed for own nodes
    penalty = torch.relu(safety_margin - dist_to_bbox) ** 2
    penalty = penalty * (~is_own).float()

    return penalty.mean()


def _port_angular_resolution_loss(
    endpoints: torch.Tensor,
    cp: torch.Tensor,
    src_list: list,
    tgt_list: list,
) -> torch.Tensor:
    """Penalize small angles between incident edges at each node.

    Tangent at source: B'(0) = 3(P1 - P0)
    Tangent at target: B'(1) = 3(P3 - P2)
    """
    E = endpoints.shape[0]
    if E < 2:
        return torch.tensor(0.0)

    min_angle_target = math.pi / 8  # 22.5 degrees

    # Source tangents: 3 * (cp1 - p0) → direction leaving source
    src_tangent = cp[:, 0, :] - endpoints[:, 0, :]  # [E, 2]
    # Target tangents: 3 * (p3 - cp2) → direction arriving at target
    tgt_tangent = endpoints[:, 1, :] - cp[:, 1, :]  # [E, 2]

    # Group tangents by node
    src_nodes = torch.tensor(src_list)
    tgt_nodes = torch.tensor(tgt_list)
    all_nodes = torch.cat([src_nodes, tgt_nodes])
    all_tangents = torch.cat([src_tangent, tgt_tangent], dim=0)  # [2E, 2]

    unique_nodes = all_nodes.unique()
    total_loss = torch.tensor(0.0)
    count = 0

    for node in unique_nodes:
        mask = all_nodes == node
        node_tangents = all_tangents[mask]  # [K, 2]
        K = node_tangents.shape[0]
        if K < 2:
            continue

        # Normalize tangents
        norms = node_tangents.norm(dim=1, keepdim=True).clamp(min=1e-6)
        unit = node_tangents / norms

        # Angles between all pairs
        dots = (unit @ unit.t()).clamp(-1.0, 1.0)  # [K, K]
        angles = torch.acos(dots)

        # Only upper triangle (unique pairs)
        idx = torch.triu_indices(K, K, offset=1)
        pair_angles = angles[idx[0], idx[1]]

        # Penalty: relu(target - angle)^2
        penalty = torch.relu(min_angle_target - pair_angles) ** 2
        total_loss = total_loss + penalty.sum()
        count += pair_angles.numel()

    if count == 0:
        return torch.tensor(0.0)
    return total_loss / count


def _bezier_derivatives_batch(
    endpoints: torch.Tensor,
    cp: torch.Tensor,
    t_samples: torch.Tensor,
):
    """Compute first and second derivatives of bezier curves.

    B'(t) = 3[(1-t)^2(P1-P0) + 2(1-t)t(P2-P1) + t^2(P3-P2)]
    B''(t) = 6[(1-t)(P2-2P1+P0) + t(P3-2P2+P1)]

    Returns:
        d1: [E, T, 2] — first derivative
        d2: [E, T, 2] — second derivative
    """
    t = t_samples.unsqueeze(-1)  # [1, T, 1]
    u = 1.0 - t

    p0 = endpoints[:, 0:1, :]
    p3 = endpoints[:, 1:2, :]
    p1 = cp[:, 0:1, :]
    p2 = cp[:, 1:2, :]

    # First derivative
    d1 = 3 * (u**2 * (p1 - p0) + 2 * u * t * (p2 - p1) + t**2 * (p3 - p2))

    # Second derivative
    d2 = 6 * (u * (p2 - 2 * p1 + p0) + t * (p3 - 2 * p2 + p1))

    return d1, d2


def _curvature_consistency_loss(
    endpoints: torch.Tensor,
    cp: torch.Tensor,
    t_samples: torch.Tensor,
) -> torch.Tensor:
    """Penalize variance of curvature κ across all edges.

    κ = |B' × B''| / |B'|³
    """
    d1, d2 = _bezier_derivatives_batch(endpoints, cp, t_samples)

    # Cross product in 2D: d1_x * d2_y - d1_y * d2_x
    cross = d1[:, :, 0] * d2[:, :, 1] - d1[:, :, 1] * d2[:, :, 0]  # [E, T]
    d1_norm = d1.norm(dim=2).clamp(min=1e-6)  # [E, T]

    kappa = cross.abs() / d1_norm.clamp(min=1.0)**3  # [E, T]  clamp avoids blowup on short edges

    # Mean curvature per edge, then variance across edges
    mean_kappa = kappa.mean(dim=1)  # [E]
    if mean_kappa.numel() < 2:
        return torch.tensor(0.0)

    return mean_kappa.var()


def _curvature_penalty_loss(
    endpoints: torch.Tensor,
    cp: torch.Tensor,
    t_samples: torch.Tensor,
) -> torch.Tensor:
    """Penalize total curvature — prefer straighter paths.

    mean(κ²) across all edges and sample points.
    """
    d1, d2 = _bezier_derivatives_batch(endpoints, cp, t_samples)

    cross = d1[:, :, 0] * d2[:, :, 1] - d1[:, :, 1] * d2[:, :, 0]
    d1_norm = d1.norm(dim=2).clamp(min=1e-6)

    kappa = cross.abs() / d1_norm.clamp(min=1.0)**3  # clamp avoids blowup on short edges
    return (kappa**2).mean()


def _edge_cluster_crossing_loss(
    points: torch.Tensor,
    cluster_bboxes: torch.Tensor,
    node_cluster_mask: torch.Tensor,
    src_list: list,
    tgt_list: list,
    E: int,
) -> torch.Tensor:
    """Penalize edge sample points inside foreign cluster bboxes.

    A cluster is "foreign" to an edge if neither the source nor target node
    belongs to that cluster. Uses soft proximity: sigmoid penetration depth.

    Args:
        points: [E, T, 2] sampled bezier points
        cluster_bboxes: [C, 4] (x_min, y_min, x_max, y_max) per cluster
        node_cluster_mask: [N, C] bool — True if node i is in cluster c
        src_list: source node indices per edge
        tgt_list: target node indices per edge
        E: number of edges
    """
    C = cluster_bboxes.shape[0]
    T = points.shape[1]

    # Build foreign mask: [E, C] — True if neither src nor tgt belongs to cluster c
    src_t = torch.tensor(src_list[:E], dtype=torch.long)
    tgt_t = torch.tensor(tgt_list[:E], dtype=torch.long)
    src_in = node_cluster_mask[src_t]  # [E, C]
    tgt_in = node_cluster_mask[tgt_t]  # [E, C]
    foreign = ~(src_in | tgt_in)  # [E, C]

    if not foreign.any():
        return torch.tensor(0.0)

    # For each sample point, compute penetration into each cluster bbox
    # points: [E, T, 2], cluster_bboxes: [C, 4]
    px = points[:, :, 0]  # [E, T]
    py = points[:, :, 1]  # [E, T]

    # Expand for broadcasting: [E, T, C]
    bx_min = cluster_bboxes[:, 0].unsqueeze(0).unsqueeze(0)  # [1, 1, C]
    by_min = cluster_bboxes[:, 1].unsqueeze(0).unsqueeze(0)
    bx_max = cluster_bboxes[:, 2].unsqueeze(0).unsqueeze(0)
    by_max = cluster_bboxes[:, 3].unsqueeze(0).unsqueeze(0)

    px_exp = px.unsqueeze(2)  # [E, T, 1]
    py_exp = py.unsqueeze(2)  # [E, T, 1]

    # Penetration depth per axis (positive = inside bbox)
    pen_x = torch.relu(px_exp - bx_min) * torch.relu(bx_max - px_exp)  # [E, T, C]
    pen_y = torch.relu(py_exp - by_min) * torch.relu(by_max - py_exp)

    # Inside bbox when both pen_x > 0 and pen_y > 0
    # Use soft product as penetration measure
    penetration = pen_x * pen_y  # [E, T, C]

    # Mask by foreign: only penalize foreign clusters
    foreign_exp = foreign.unsqueeze(1).expand(-1, T, -1)  # [E, T, C]
    masked = penetration * foreign_exp.float()

    return masked.mean()
