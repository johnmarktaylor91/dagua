"""Aesthetic quality metrics for layout evaluation.

All metrics are computable from positions + graph topology (no rendering needed
for placement metrics). Edge curvature metrics need bezier control points.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import numpy as np


def compute_all_metrics(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    clusters: Optional[Dict] = None,
    direction: str = "TB",
) -> Dict[str, float]:
    """Compute all aesthetic metrics for a layout.

    Args:
        direction: Layout direction ("TB", "BT", "LR", "RL"). Affects which
            axis is used for DAG fraction and edge straightness metrics.
    """
    pos = positions.detach().cpu()
    ei = edge_index.detach().cpu()
    ns = node_sizes.detach().cpu()
    n = pos.shape[0]

    metrics = {}
    metrics["num_nodes"] = n
    metrics["num_edges"] = ei.shape[1] if ei.numel() > 0 else 0
    metrics["edge_crossings"] = count_crossings(pos, ei)
    metrics["dag_fraction"] = compute_dag_fraction(pos, ei, direction=direction)
    metrics["node_overlaps"] = count_overlaps(pos, ns)
    metrics["mean_edge_length"] = compute_mean_edge_length(pos, ei)
    metrics["edge_length_variance"] = compute_edge_length_variance(pos, ei)
    metrics["edge_straightness"] = compute_edge_straightness(pos, ei, direction=direction)
    metrics["x_alignment"] = compute_x_alignment(pos, ei, direction=direction)
    metrics["total_area"] = compute_total_area(pos, ns)
    metrics["aspect_ratio"] = compute_aspect_ratio(pos, ns)
    metrics["min_node_gap"] = compute_min_node_gap(pos, ns)
    metrics["overall_quality"] = overall_quality(metrics)
    return metrics


def count_crossings(pos: torch.Tensor, edge_index: torch.Tensor) -> int:
    """Count the number of edge pair intersections.

    For graphs with <= 500 edges: exact O(E^2) count.
    For larger graphs: vectorized random sample of edge pairs, scaled to estimate.
    """
    if edge_index.numel() == 0 or edge_index.shape[1] < 2:
        return 0

    n_edges = edge_index.shape[1]
    src = edge_index[0]
    tgt = edge_index[1]

    if n_edges <= 500:
        # Exact count
        crossings = 0
        for i in range(n_edges):
            for j in range(i + 1, n_edges):
                a, b = pos[src[i]].numpy(), pos[tgt[i]].numpy()
                c, d = pos[src[j]].numpy(), pos[tgt[j]].numpy()
                if _segments_intersect(a, b, c, d):
                    crossings += 1
        return crossings
    else:
        # Vectorized random sampling
        max_pairs = min(125000, n_edges * (n_edges - 1) // 2)
        total_pairs = n_edges * (n_edges - 1) // 2

        torch.manual_seed(42)
        idx_i = torch.randint(0, n_edges, (max_pairs,))
        idx_j = torch.randint(0, n_edges, (max_pairs,))
        valid = idx_i != idx_j
        idx_i = idx_i[valid]
        idx_j = idx_j[valid]

        # Vectorized crossing test using cross products
        a = pos[src[idx_i]]  # [P, 2]
        b = pos[tgt[idx_i]]  # [P, 2]
        c = pos[src[idx_j]]  # [P, 2]
        d = pos[tgt[idx_j]]  # [P, 2]

        # Cross product based intersection test
        d1 = _cross_2d(c, d, a)  # [P]
        d2 = _cross_2d(c, d, b)  # [P]
        d3 = _cross_2d(a, b, c)  # [P]
        d4 = _cross_2d(a, b, d)  # [P]

        # Proper intersection: signs differ on both tests
        intersect = ((d1 > 0) != (d2 > 0)) & ((d3 > 0) != (d4 > 0))

        # Exclude shared endpoints
        shared = (torch.norm(a - c, dim=1) < 1e-6) | (torch.norm(a - d, dim=1) < 1e-6) | \
                 (torch.norm(b - c, dim=1) < 1e-6) | (torch.norm(b - d, dim=1) < 1e-6)
        intersect = intersect & ~shared

        checked = idx_i.shape[0]
        crossings = int(intersect.sum().item())
        if checked == 0:
            return 0
        return int(crossings * total_pairs / checked)


def _cross_2d(o: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Vectorized 2D cross product: (a-o) x (b-o)."""
    return (a[:, 0] - o[:, 0]) * (b[:, 1] - o[:, 1]) - (a[:, 1] - o[:, 1]) * (b[:, 0] - o[:, 0])


def _segments_intersect(a, b, c, d) -> bool:
    """Check if line segments AB and CD intersect (excluding shared endpoints)."""
    if np.allclose(a, c) or np.allclose(a, d) or np.allclose(b, c) or np.allclose(b, d):
        return False

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1 = cross(c, d, a)
    d2 = cross(c, d, b)
    d3 = cross(a, b, c)
    d4 = cross(a, b, d)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def compute_dag_fraction(pos: torch.Tensor, edge_index: torch.Tensor, direction: str = "TB") -> float:
    """Fraction of edges pointing in the layout direction.

    TB: target.y > source.y (downward)
    BT: target.y < source.y (upward)
    LR: target.x > source.x (rightward)
    RL: target.x < source.x (leftward)
    """
    if edge_index.numel() == 0:
        return 1.0

    src, tgt = edge_index[0], edge_index[1]
    if direction == "TB":
        correct = pos[tgt, 1] > pos[src, 1]
    elif direction == "BT":
        correct = pos[tgt, 1] < pos[src, 1]
    elif direction == "LR":
        correct = pos[tgt, 0] > pos[src, 0]
    elif direction == "RL":
        correct = pos[tgt, 0] < pos[src, 0]
    else:
        correct = pos[tgt, 1] > pos[src, 1]
    return correct.float().mean().item()


def count_overlaps(pos: torch.Tensor, node_sizes: torch.Tensor) -> int:
    """Count overlapping node bounding box pairs.

    Vectorized for N <= 2000, sampling-based for larger graphs.
    """
    n = pos.shape[0]
    if n <= 1:
        return 0

    if n <= 2000:
        # Vectorized: all pairs
        dx = (pos[:, 0].unsqueeze(0) - pos[:, 0].unsqueeze(1)).abs()
        dy = (pos[:, 1].unsqueeze(0) - pos[:, 1].unsqueeze(1)).abs()
        min_dx = (node_sizes[:, 0].unsqueeze(0) + node_sizes[:, 0].unsqueeze(1)) / 2
        min_dy = (node_sizes[:, 1].unsqueeze(0) + node_sizes[:, 1].unsqueeze(1)) / 2
        overlapping = (dx < min_dx) & (dy < min_dy)
        overlapping.fill_diagonal_(False)
        # Upper triangle only
        return int(overlapping.triu(diagonal=1).sum().item())
    else:
        # Vectorized random sampling for large graphs
        max_pairs = min(500000, n * (n - 1) // 2)
        total_pairs = n * (n - 1) // 2

        torch.manual_seed(42)
        idx_i = torch.randint(0, n, (max_pairs,))
        idx_j = torch.randint(0, n, (max_pairs,))
        valid = idx_i != idx_j
        idx_i = idx_i[valid]
        idx_j = idx_j[valid]

        dx = (pos[idx_i, 0] - pos[idx_j, 0]).abs()
        dy = (pos[idx_i, 1] - pos[idx_j, 1]).abs()
        min_dx = (node_sizes[idx_i, 0] + node_sizes[idx_j, 0]) / 2
        min_dy = (node_sizes[idx_i, 1] + node_sizes[idx_j, 1]) / 2
        overlapping = (dx < min_dx) & (dy < min_dy)

        checked = idx_i.shape[0]
        overlap_count = int(overlapping.sum().item())
        if checked == 0:
            return 0
        return int(overlap_count * total_pairs / checked)


def compute_mean_edge_length(pos: torch.Tensor, edge_index: torch.Tensor) -> float:
    """Average Euclidean distance between connected nodes."""
    if edge_index.numel() == 0:
        return 0.0
    src, tgt = edge_index[0], edge_index[1]
    lengths = ((pos[src] - pos[tgt]) ** 2).sum(dim=1).sqrt()
    return lengths.mean().item()


def compute_edge_length_variance(pos: torch.Tensor, edge_index: torch.Tensor) -> float:
    """Variance of edge lengths."""
    if edge_index.numel() == 0 or edge_index.shape[1] <= 1:
        return 0.0
    src, tgt = edge_index[0], edge_index[1]
    lengths = ((pos[src] - pos[tgt]) ** 2).sum(dim=1).sqrt()
    return lengths.var().item()


def compute_edge_straightness(pos: torch.Tensor, edge_index: torch.Tensor, direction: str = "TB") -> float:
    """Average angular deviation from the primary axis (in degrees).

    For TB/BT: deviation from vertical (atan2(dx, dy)).
    For LR/RL: deviation from horizontal (atan2(dy, dx)).
    """
    if edge_index.numel() == 0:
        return 0.0
    src, tgt = edge_index[0], edge_index[1]
    dx = (pos[tgt, 0] - pos[src, 0]).abs()
    dy = (pos[tgt, 1] - pos[src, 1]).abs().clamp(min=1e-6)
    if direction in ("LR", "RL"):
        dx = dx.clamp(min=1e-6)
        angles = torch.atan2(dy, dx) * 180 / torch.pi
    else:
        angles = torch.atan2(dx, dy) * 180 / torch.pi
    return angles.mean().item()


def compute_x_alignment(pos: torch.Tensor, edge_index: torch.Tensor, direction: str = "TB") -> float:
    """Mean absolute displacement along the cross-axis between connected nodes.

    For TB/BT: x-displacement (edges should be vertically aligned).
    For LR/RL: y-displacement (edges should be horizontally aligned).
    """
    if edge_index.numel() == 0:
        return 0.0
    src, tgt = edge_index[0], edge_index[1]
    if direction in ("LR", "RL"):
        return (pos[src, 1] - pos[tgt, 1]).abs().mean().item()
    return (pos[src, 0] - pos[tgt, 0]).abs().mean().item()


def compute_total_area(pos: torch.Tensor, node_sizes: torch.Tensor) -> float:
    """Bounding box area of the layout."""
    if pos.shape[0] == 0:
        return 0.0
    x_min = (pos[:, 0] - node_sizes[:, 0] / 2).min().item()
    x_max = (pos[:, 0] + node_sizes[:, 0] / 2).max().item()
    y_min = (pos[:, 1] - node_sizes[:, 1] / 2).min().item()
    y_max = (pos[:, 1] + node_sizes[:, 1] / 2).max().item()
    return (x_max - x_min) * (y_max - y_min)


def compute_aspect_ratio(pos: torch.Tensor, node_sizes: torch.Tensor) -> float:
    """Width / height of bounding box."""
    if pos.shape[0] == 0:
        return 1.0
    x_min = (pos[:, 0] - node_sizes[:, 0] / 2).min().item()
    x_max = (pos[:, 0] + node_sizes[:, 0] / 2).max().item()
    y_min = (pos[:, 1] - node_sizes[:, 1] / 2).min().item()
    y_max = (pos[:, 1] + node_sizes[:, 1] / 2).max().item()
    w = x_max - x_min
    h = y_max - y_min
    return w / max(h, 1e-6)


def compute_min_node_gap(pos: torch.Tensor, node_sizes: torch.Tensor) -> float:
    """Minimum distance between any two non-overlapping node bboxes."""
    n = pos.shape[0]
    if n <= 1:
        return float("inf")

    min_gap = float("inf")
    for i in range(min(n, 200)):
        for j in range(i + 1, min(n, 200)):
            dx = abs(pos[i, 0] - pos[j, 0]).item()
            dy = abs(pos[i, 1] - pos[j, 1]).item()
            min_dx = (node_sizes[i, 0] + node_sizes[j, 0]).item() / 2
            min_dy = (node_sizes[i, 1] + node_sizes[j, 1]).item() / 2
            gap_x = dx - min_dx
            gap_y = dy - min_dy
            if gap_x > 0 or gap_y > 0:
                gap = max(gap_x, 0) + max(gap_y, 0)
                min_gap = min(min_gap, gap)
    return min_gap


def overall_quality(metrics: Dict[str, float]) -> float:
    """Single scalar summarizing layout quality. Higher is better."""
    score = 0.0
    score -= 100 * metrics.get("node_overlaps", 0)
    score -= 10 * metrics.get("edge_crossings", 0)
    score += 50 * metrics.get("dag_fraction", 1.0)
    score -= 1 * metrics.get("edge_length_variance", 0)
    score -= 2 * metrics.get("x_alignment", 0)
    n = max(metrics.get("num_nodes", 1), 1)
    score -= 0.5 * metrics.get("total_area", 0) / n
    return score


def graphviz_delta(dagua_metrics: Dict, graphviz_metrics: Dict) -> Dict[str, float]:
    """How much worse/better is Dagua vs Graphviz for each metric."""
    delta = {}
    for key in dagua_metrics:
        if key in graphviz_metrics and isinstance(dagua_metrics[key], (int, float)):
            delta[key] = dagua_metrics[key] - graphviz_metrics[key]
    return delta
