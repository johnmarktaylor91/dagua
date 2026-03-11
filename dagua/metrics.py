"""Layout quality metrics suite for dagua.

Three-tier system designed for computational feasibility at 50M+ nodes:
- Tier 1: O(N) or O(|E|), always compute
- Tier 2: Sampled, O(n_samples * BFS_cost), compute when feasible
- Tier 3: DAG/hierarchy-specific, O(N) given metadata

API:
- quick(pos, edge_index, ...) -> dict   — Tier-1 only, runs in seconds at any scale
- full(pos, edge_index, ...) -> dict    — All tiers including sampled metrics
- compare(pos_a, pos_b) -> dict         — Procrustes comparison of two layouts
- composite(metrics) -> float           — Single scalar 0-100

All metric functions are also importable individually.
"""

from __future__ import annotations

import math
import time as _time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().cpu() if t.device.type != "cpu" else t.detach()


def segments_intersect(p1, p2, p3, p4):
    """Vectorized segment intersection test.  All inputs [N, 2].

    Returns [N] bool tensor.  Complexity: O(N).
    """
    d1 = p2 - p1  # [N, 2]
    d2 = p4 - p3
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]  # [N]

    parallel = cross.abs() < 1e-10

    d3 = p3 - p1
    t = (d3[:, 0] * d2[:, 1] - d3[:, 1] * d2[:, 0]) / cross.clamp(min=1e-10)
    u = (d3[:, 0] * d1[:, 1] - d3[:, 1] * d1[:, 0]) / cross.clamp(min=1e-10)

    return (~parallel) & (t > 0) & (t < 1) & (u > 0) & (u < 1)


def _build_csr(edge_index: torch.Tensor, num_nodes: int):
    """Build CSR adjacency from edge_index.  Returns (offsets, targets) numpy arrays."""
    src, tgt = edge_index[0], edge_index[1]
    # Make undirected for BFS
    all_src = torch.cat([src, tgt])
    all_tgt = torch.cat([tgt, src])

    order = all_src.argsort()
    csr_tgt = all_tgt[order].numpy()

    degree = torch.zeros(num_nodes, dtype=torch.long)
    degree.scatter_add_(0, all_src, torch.ones(all_src.shape[0], dtype=torch.long))
    offsets = torch.zeros(num_nodes + 1, dtype=torch.long)
    offsets[1:] = degree.cumsum(0)
    return offsets.numpy(), csr_tgt


def _bfs_distances(csr_offsets, csr_targets, source: int, max_dist: int = 20):
    """BFS from source using CSR adjacency.  Returns numpy array of distances (-1 = unreached)."""
    N = len(csr_offsets) - 1
    dist = np.full(N, -1, dtype=np.int64)
    dist[source] = 0
    queue = deque([source])

    while queue:
        node = queue.popleft()
        d = dist[node]
        if d >= max_dist:
            continue
        for j in range(csr_offsets[node], csr_offsets[node + 1]):
            child = int(csr_targets[j])
            if dist[child] == -1:
                dist[child] = d + 1
                queue.append(child)
    return dist


# ---------------------------------------------------------------------------
# Tier 1: Always compute — O(N) or O(|E|)
# ---------------------------------------------------------------------------

def edge_length_cv(pos: torch.Tensor, edge_index: torch.Tensor) -> Dict[str, float]:
    """Edge length coefficient of variation + distribution stats.

    Complexity: O(|E|).
    Target: CV < 0.5 good, < 0.3 excellent, > 1.0 problem.
    """
    if edge_index.numel() == 0:
        return {"edge_length_cv": 0.0, "edge_length_mean": 0.0, "edge_length_std": 0.0,
                "edge_length_min": 0.0, "edge_length_max": 0.0,
                "edge_length_p01": 0.0, "edge_length_p99": 0.0}

    src, tgt = edge_index[0], edge_index[1]
    lengths = torch.norm(pos[src] - pos[tgt], dim=1)
    mean_len = lengths.mean()
    std_len = lengths.std() if lengths.shape[0] > 1 else torch.tensor(0.0)
    cv = (std_len / mean_len).item() if mean_len > 1e-8 else 0.0

    return {
        "edge_length_cv": cv,
        "edge_length_mean": mean_len.item(),
        "edge_length_std": std_len.item(),
        "edge_length_min": lengths.min().item(),
        "edge_length_max": lengths.max().item(),
        "edge_length_p01": lengths.quantile(0.01).item() if len(lengths) > 1 else lengths[0].item(),
        "edge_length_p99": lengths.quantile(0.99).item() if len(lengths) > 1 else lengths[0].item(),
    }


def dag_consistency(pos: torch.Tensor, edge_index: torch.Tensor,
                    direction: str = "TB",
                    back_edge_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """Fraction of edges pointing in the correct DAG direction + violation details.

    Complexity: O(|E|).
    Target: 1.0 (exact).

    Args:
        back_edge_mask: if provided, back edges are excluded from consistency
            checks and reported separately as back_edge_count / back_edge_fraction.
    """
    if edge_index.numel() == 0:
        return {"dag_consistency": 1.0, "dag_num_violations": 0,
                "dag_mean_violation_mag": 0.0, "dag_max_violation_mag": 0.0,
                "back_edge_count": 0, "back_edge_fraction": 0.0}

    total_edges = edge_index.shape[1]

    # Filter to forward edges only when back_edge_mask is provided
    if back_edge_mask is not None and back_edge_mask.any():
        forward_mask = ~back_edge_mask
        forward_ei = edge_index[:, forward_mask]
        n_back = back_edge_mask.sum().item()
    else:
        forward_ei = edge_index
        n_back = 0

    result: Dict[str, float] = {}

    if forward_ei.numel() == 0:
        result.update({
            "dag_consistency": 1.0, "dag_num_violations": 0,
            "dag_mean_violation_mag": 0.0, "dag_max_violation_mag": 0.0,
        })
    else:
        src, tgt = forward_ei[0], forward_ei[1]
        if direction == "TB":
            y_src, y_tgt = pos[src, 1], pos[tgt, 1]
            correct = y_tgt > y_src
            violation_mag = torch.clamp(y_src[~correct] - y_tgt[~correct], min=0)
        elif direction == "BT":
            y_src, y_tgt = pos[src, 1], pos[tgt, 1]
            correct = y_tgt < y_src
            violation_mag = torch.clamp(y_tgt[~correct] - y_src[~correct], min=0)
        elif direction == "LR":
            x_src, x_tgt = pos[src, 0], pos[tgt, 0]
            correct = x_tgt > x_src
            violation_mag = torch.clamp(x_src[~correct] - x_tgt[~correct], min=0)
        elif direction == "RL":
            x_src, x_tgt = pos[src, 0], pos[tgt, 0]
            correct = x_tgt < x_src
            violation_mag = torch.clamp(x_tgt[~correct] - x_src[~correct], min=0)
        else:
            y_src, y_tgt = pos[src, 1], pos[tgt, 1]
            correct = y_tgt > y_src
            violation_mag = torch.clamp(y_src[~correct] - y_tgt[~correct], min=0)

        n_violations = (~correct).sum().item()
        result.update({
            "dag_consistency": correct.float().mean().item(),
            "dag_num_violations": int(n_violations),
            "dag_mean_violation_mag": violation_mag.mean().item() if n_violations > 0 else 0.0,
            "dag_max_violation_mag": violation_mag.max().item() if n_violations > 0 else 0.0,
        })

    result["back_edge_count"] = int(n_back)
    result["back_edge_fraction"] = n_back / total_edges if total_edges > 0 else 0.0
    return result


def depth_position_correlation(pos: torch.Tensor, topo_depth: torch.Tensor) -> Dict[str, float]:
    """Spearman rank correlation between topological depth and y-coordinate.

    Complexity: O(N log N) for the sort.
    Target: > 0.95 good, > 0.99 excellent.
    """
    from scipy.stats import spearmanr

    depth_np = topo_depth.cpu().numpy() if isinstance(topo_depth, torch.Tensor) else np.array(topo_depth)
    y_np = pos[:, 1].cpu().numpy()
    rho, pval = spearmanr(depth_np, y_np)
    return {"depth_spearman_rho": float(rho), "depth_spearman_pval": float(pval)}


def count_overlaps_detailed(pos: torch.Tensor, node_sizes: torch.Tensor) -> Dict[str, int]:
    """Count overlapping node bounding box pairs using spatial hashing.

    Complexity: O(N) expected with proper cell sizing.
    Target: 0 (any nonzero = projection bug).
    """
    n = pos.shape[0]
    if n <= 1:
        return {"overlap_count": 0}

    half_w = node_sizes[:, 0] / 2
    half_h = node_sizes[:, 1] / 2

    # For small graphs, use exact pairwise
    if n <= 2000:
        dx = (pos[:, 0].unsqueeze(0) - pos[:, 0].unsqueeze(1)).abs()
        dy = (pos[:, 1].unsqueeze(0) - pos[:, 1].unsqueeze(1)).abs()
        min_dx = half_w.unsqueeze(0) + half_w.unsqueeze(1)
        min_dy = half_h.unsqueeze(0) + half_h.unsqueeze(1)
        overlapping = (dx < min_dx) & (dy < min_dy)
        overlapping.fill_diagonal_(False)
        return {"overlap_count": int(overlapping.triu(diagonal=1).sum().item())}

    # Spatial hash: O(N) expected
    cell_size = max(node_sizes[:, 0].max().item(), node_sizes[:, 1].max().item()) + 1.0
    cx = torch.floor(pos[:, 0] / cell_size).long()
    cy = torch.floor(pos[:, 1] / cell_size).long()
    cx_min, cy_min = cx.min(), cy.min()
    cx_rel = cx - cx_min
    cy_rel = cy - cy_min
    cy_range = max(int(cy_rel.max().item()) + 1, 1)
    cell_hash = cx_rel * cy_range + cy_rel

    sorted_idx = cell_hash.argsort()
    sorted_hash = cell_hash[sorted_idx]

    changes = torch.where(sorted_hash[1:] != sorted_hash[:-1])[0] + 1
    starts = torch.cat([torch.zeros(1, dtype=torch.long), changes])
    ends = torch.cat([changes, torch.tensor([n], dtype=torch.long)])
    cell_sizes_arr = ends - starts

    # Only check cells with 2+ nodes (cap per-cell to avoid O(N²) degenerate case)
    multi_mask = cell_sizes_arr >= 2
    multi_starts = starts[multi_mask]
    multi_ends = ends[multi_mask]

    overlap_count = 0
    n_cells = multi_starts.shape[0]

    # Vectorized check per cell, capped at 200 nodes per cell
    for i in range(min(int(n_cells), 100000)):
        s = multi_starts[i].item()
        e = multi_ends[i].item()
        cell_nodes = sorted_idx[s:e]
        m = cell_nodes.shape[0]
        if m > 200:
            cell_nodes = cell_nodes[torch.randperm(m)[:200]]
            m = 200

        p = pos[cell_nodes]
        hw_c = half_w[cell_nodes]
        hh_c = half_h[cell_nodes]

        dx = (p[:, 0].unsqueeze(1) - p[:, 0].unsqueeze(0)).abs()
        dy = (p[:, 1].unsqueeze(1) - p[:, 1].unsqueeze(0)).abs()
        min_dx = hw_c.unsqueeze(1) + hw_c.unsqueeze(0)
        min_dy = hh_c.unsqueeze(1) + hh_c.unsqueeze(0)
        overlapping = (dx < min_dx) & (dy < min_dy)
        overlapping.fill_diagonal_(False)
        overlap_count += int(overlapping.triu(diagonal=1).sum().item())

    return {"overlap_count": overlap_count}


def aspect_ratio(pos: torch.Tensor, node_sizes: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """Bounding box width / height.

    Complexity: O(N).
    """
    if pos.shape[0] == 0:
        return {"aspect_ratio": 1.0, "bbox_width": 0.0, "bbox_height": 0.0}

    if node_sizes is not None:
        x_min = (pos[:, 0] - node_sizes[:, 0] / 2).min().item()
        x_max = (pos[:, 0] + node_sizes[:, 0] / 2).max().item()
        y_min = (pos[:, 1] - node_sizes[:, 1] / 2).min().item()
        y_max = (pos[:, 1] + node_sizes[:, 1] / 2).max().item()
    else:
        x_min, x_max = pos[:, 0].min().item(), pos[:, 0].max().item()
        y_min, y_max = pos[:, 1].min().item(), pos[:, 1].max().item()

    w = x_max - x_min
    h = y_max - y_min
    return {
        "aspect_ratio": w / max(h, 1e-6),
        "bbox_width": w,
        "bbox_height": h,
    }


def edge_direction_straightness(pos: torch.Tensor, edge_index: torch.Tensor,
                                direction: str = "TB") -> Dict[str, float]:
    """Mean angular deviation from primary axis (degrees).

    For straight-line edges (no bezier).  For TB/BT: deviation from vertical.
    For LR/RL: deviation from horizontal.

    Complexity: O(|E|).
    Target: mean < 15° is good.
    """
    if edge_index.numel() == 0:
        return {"edge_straightness_mean_deg": 0.0, "edge_straightness_below_15": 1.0}

    src, tgt = edge_index[0], edge_index[1]
    dx = (pos[tgt, 0] - pos[src, 0]).abs()
    dy = (pos[tgt, 1] - pos[src, 1]).abs().clamp(min=1e-6)

    if direction in ("LR", "RL"):
        dx = dx.clamp(min=1e-6)
        angles = torch.atan2(dy, dx) * 180 / torch.pi
    else:
        angles = torch.atan2(dx, dy) * 180 / torch.pi

    return {
        "edge_straightness_mean_deg": angles.mean().item(),
        "edge_straightness_below_15": (angles < 15).float().mean().item(),
    }


# ---------------------------------------------------------------------------
# Tier 2: Sampled metrics
# ---------------------------------------------------------------------------

def sampled_stress(pos: torch.Tensor, edge_index: torch.Tensor, num_nodes: int,
                   n_sources: int = 200, n_targets: int = 1000,
                   max_dist: int = 20) -> Dict[str, float]:
    """Sampled graph-theoretic stress.

    Measures how well Euclidean distances preserve graph distances.
    Complexity: O(n_sources * (V + E)).
    """
    pos_np = pos.cpu().numpy()
    ei = _ensure_cpu(edge_index)
    N = num_nodes

    if ei.numel() == 0 or N < 2:
        return {"sampled_stress": 0.0, "stress_n_pairs": 0,
                "stress_n_sources": 0, "stress_n_targets": 0}

    csr_off, csr_tgt = _build_csr(ei, N)
    sources = torch.randperm(N)[:min(n_sources, N)].numpy()

    total_stress = 0.0
    count = 0

    for src_node in sources:
        dist = _bfs_distances(csr_off, csr_tgt, int(src_node), max_dist=max_dist)

        reachable = np.where(dist > 0)[0]
        if len(reachable) == 0:
            continue

        targets = reachable[np.random.permutation(len(reachable))[:n_targets]]
        d_graph = dist[targets].astype(np.float64)
        d_euclidean = np.linalg.norm(pos_np[targets] - pos_np[src_node], axis=1)

        w = 1.0 / (d_graph ** 2)
        stress = (w * (d_graph - d_euclidean) ** 2).sum()
        total_stress += stress
        count += len(targets)

    return {
        "sampled_stress": total_stress / max(count, 1),
        "stress_n_pairs": count,
        "stress_n_sources": len(sources),
        "stress_n_targets": n_targets,
    }


def sampled_crossing_rate(pos: torch.Tensor, edge_index: torch.Tensor,
                          n_samples: int = 1_000_000) -> Dict[str, float]:
    """Estimated crossing rate via random edge-pair sampling.

    Complexity: O(n_samples).
    """
    if edge_index.numel() == 0 or edge_index.shape[1] < 2:
        return {"crossing_rate": 0.0, "crossing_se": 0.0,
                "crossing_estimated_total": 0, "crossing_n_samples": 0}

    E = edge_index.shape[1]
    src, tgt = edge_index[0], edge_index[1]
    actual_samples = min(n_samples, E * (E - 1) // 2)

    idx1 = torch.randint(0, E, (actual_samples,))
    idx2 = torch.randint(0, E, (actual_samples,))

    # Exclude pairs sharing a node
    e1s, e1t = src[idx1], tgt[idx1]
    e2s, e2t = src[idx2], tgt[idx2]
    shares_node = (e1s == e2s) | (e1s == e2t) | (e1t == e2s) | (e1t == e2t)
    same_edge = idx1 == idx2
    valid = ~shares_node & ~same_edge

    if valid.sum() == 0:
        return {"crossing_rate": 0.0, "crossing_se": 0.0,
                "crossing_estimated_total": 0, "crossing_n_samples": 0}

    p1 = pos[e1s[valid]]
    p2 = pos[e1t[valid]]
    p3 = pos[e2s[valid]]
    p4 = pos[e2t[valid]]

    crossings = segments_intersect(p1, p2, p3, p4)
    n_valid = int(valid.sum().item())
    rate = crossings.float().mean().item()
    se = (rate * (1 - rate) / n_valid) ** 0.5 if n_valid > 0 else 0.0

    return {
        "crossing_rate": rate,
        "crossing_se": se,
        "crossing_estimated_total": int(rate * E * (E - 1) / 2),
        "crossing_n_samples": n_valid,
    }


def neighborhood_preservation(pos: torch.Tensor, edge_index: torch.Tensor,
                              num_nodes: int, n_samples: int = 5000,
                              k: int = 10) -> Dict[str, float]:
    """Fraction of k-nearest graph neighbors that are also k-nearest Euclidean neighbors.

    Complexity: O(n_samples * (BFS_cost + N)) — use spatial index for large N.
    Target: > 0.5 good, > 0.7 excellent.
    """
    pos_np = pos.cpu().numpy()
    ei = _ensure_cpu(edge_index)
    N = num_nodes

    if ei.numel() == 0 or N < k + 1:
        return {"neighborhood_mean": 0.0, "neighborhood_median": 0.0,
                "neighborhood_std": 0.0}

    csr_off, csr_tgt = _build_csr(ei, N)
    samples = torch.randperm(N)[:min(n_samples, N)].numpy()

    scores = []
    for node in samples:
        # k nearest graph neighbors via BFS
        dist = _bfs_distances(csr_off, csr_tgt, int(node), max_dist=k)
        reachable = np.where((dist > 0) & (dist <= k))[0]
        if len(reachable) < k:
            graph_neighbors = set(reachable.tolist())
        else:
            # Take the k closest by graph distance
            order = np.argsort(dist[reachable])[:k]
            graph_neighbors = set(reachable[order].tolist())

        if len(graph_neighbors) == 0:
            continue

        # k nearest Euclidean neighbors
        dists_eucl = np.linalg.norm(pos_np - pos_np[node], axis=1)
        dists_eucl[node] = np.inf
        eucl_neighbors = set(np.argpartition(dists_eucl, k)[:k].tolist())

        overlap = len(graph_neighbors & eucl_neighbors)
        scores.append(overlap / min(k, len(graph_neighbors)))

    if not scores:
        return {"neighborhood_mean": 0.0, "neighborhood_median": 0.0,
                "neighborhood_std": 0.0}

    scores_arr = np.array(scores)
    return {
        "neighborhood_mean": float(scores_arr.mean()),
        "neighborhood_median": float(np.median(scores_arr)),
        "neighborhood_std": float(scores_arr.std()),
    }


def angular_resolution(pos: torch.Tensor, edge_index: torch.Tensor,
                       n_samples: int = 10000) -> Dict[str, float]:
    """Minimum angle between incident edges at each node (sampled).

    Complexity: O(n_samples * avg_degree * log(avg_degree)).
    Target: mean > 20° decent, < 10% below 10°.
    """
    if edge_index.numel() == 0:
        return {"angular_res_mean_deg": 360.0, "angular_res_median_deg": 360.0,
                "angular_res_below_10deg": 0.0}

    src, tgt = edge_index[0], edge_index[1]
    N = pos.shape[0]
    PI = torch.pi

    # Build undirected edge angles per node
    all_src = torch.cat([src, tgt])
    all_tgt = torch.cat([tgt, src])
    dx = pos[all_tgt, 0] - pos[all_src, 0]
    dy = pos[all_tgt, 1] - pos[all_src, 1]
    angles = torch.atan2(dy, dx)  # [-pi, pi]

    # Count degree per node
    degree = torch.zeros(N, dtype=torch.long)
    degree.scatter_add_(0, all_src, torch.ones(all_src.shape[0], dtype=torch.long))

    # Only sample nodes with degree >= 2
    candidates = torch.where(degree >= 2)[0]
    if candidates.numel() == 0:
        return {"angular_res_mean_deg": 360.0, "angular_res_median_deg": 360.0,
                "angular_res_below_10deg": 0.0}

    sample = candidates[torch.randperm(candidates.numel())[:min(n_samples, candidates.numel())]]

    min_angles = []
    for node in sample:
        mask = all_src == node
        node_angles = angles[mask].sort().values
        if node_angles.numel() < 2:
            continue
        diffs = node_angles[1:] - node_angles[:-1]
        wrap = 2 * PI - (node_angles[-1] - node_angles[0])
        all_diffs = torch.cat([diffs, wrap.unsqueeze(0)])
        min_angles.append(all_diffs.min().item())

    if not min_angles:
        return {"angular_res_mean_deg": 360.0, "angular_res_median_deg": 360.0,
                "angular_res_below_10deg": 0.0}

    min_angles_t = torch.tensor(min_angles)
    deg = min_angles_t * 180 / PI

    return {
        "angular_res_mean_deg": deg.mean().item(),
        "angular_res_median_deg": deg.median().item(),
        "angular_res_below_10deg": (deg < 10).float().mean().item(),
    }


# ---------------------------------------------------------------------------
# Tier 3: DAG/hierarchy-specific
# ---------------------------------------------------------------------------

def cluster_separation(pos: torch.Tensor, cluster_ids: torch.Tensor) -> Dict[str, float]:
    """Ratio of inter-cluster distance to intra-cluster spread.

    Complexity: O(N + C²) where C = number of clusters.
    Target: mean ratio > 3.0, < 5% overlapping.
    """
    unique_clusters = cluster_ids.unique()
    n_clusters = unique_clusters.numel()

    if n_clusters <= 1:
        return {"cluster_mean_sep_ratio": 0.0, "cluster_min_sep_ratio": 0.0,
                "cluster_frac_overlapping": 0.0}

    centroids = {}
    spreads = {}
    for c in unique_clusters:
        mask = cluster_ids == c
        cp = pos[mask]
        centroids[c.item()] = cp.mean(dim=0)
        spreads[c.item()] = cp.std(dim=0).norm().item() if cp.shape[0] > 1 else 1.0

    # Sample cluster pairs if too many
    cluster_list = list(centroids.keys())
    if n_clusters > 200:
        # Random sample of pairs
        n_pairs = min(20000, n_clusters * (n_clusters - 1) // 2)
        i_idx = np.random.randint(0, n_clusters, n_pairs)
        j_idx = np.random.randint(0, n_clusters, n_pairs)
        pairs = [(cluster_list[i], cluster_list[j]) for i, j in zip(i_idx, j_idx) if i < j]
    else:
        pairs = [(cluster_list[i], cluster_list[j])
                 for i in range(n_clusters) for j in range(i + 1, n_clusters)]

    separations = []
    for ci, cj in pairs:
        dist = torch.norm(centroids[ci] - centroids[cj]).item()
        avg_spread = (spreads[ci] + spreads[cj]) / 2
        if avg_spread > 1e-6:
            separations.append(dist / avg_spread)

    if not separations:
        return {"cluster_mean_sep_ratio": 0.0, "cluster_min_sep_ratio": 0.0,
                "cluster_frac_overlapping": 0.0}

    sep = torch.tensor(separations)
    return {
        "cluster_mean_sep_ratio": sep.mean().item(),
        "cluster_min_sep_ratio": sep.min().item(),
        "cluster_frac_overlapping": (sep < 2.0).float().mean().item(),
    }


def layer_uniformity(pos: torch.Tensor, topo_depth) -> Dict[str, float]:
    """Uniformity of vertical spacing between adjacent topological layers.

    Complexity: O(N).
    Target: layer_spacing_cv < 0.3.
    """
    if isinstance(topo_depth, torch.Tensor):
        depth_t = topo_depth
    else:
        depth_t = torch.tensor(topo_depth, dtype=torch.long)

    unique_depths = depth_t.unique().sort().values
    if unique_depths.numel() < 2:
        return {"layer_spacing_cv": 0.0, "layer_spacing_mean": 0.0,
                "layer_spacing_min": 0.0, "layer_spacing_max": 0.0, "n_layers": 1}

    medians = []
    for d in unique_depths:
        mask = depth_t == d
        medians.append(pos[mask, 1].median().item())

    medians_t = torch.tensor(medians)
    spacings = (medians_t[1:] - medians_t[:-1]).abs()

    if spacings.numel() < 2 or spacings.mean() < 1e-8:
        return {"layer_spacing_cv": 0.0, "layer_spacing_mean": spacings.mean().item(),
                "layer_spacing_min": spacings.min().item(), "layer_spacing_max": spacings.max().item(),
                "n_layers": int(unique_depths.numel())}

    return {
        "layer_spacing_cv": (spacings.std() / spacings.mean()).item(),
        "layer_spacing_mean": spacings.mean().item(),
        "layer_spacing_min": spacings.min().item(),
        "layer_spacing_max": spacings.max().item(),
        "n_layers": int(unique_depths.numel()),
    }


def edge_node_crossing_count(
    curves,
    positions: torch.Tensor,
    node_sizes: torch.Tensor,
    edge_index: torch.Tensor,
    T: int = 10,
) -> Dict[str, Any]:
    """Count edge-node bbox crossings by sampling T points per bezier.

    Tier 2 metric. Returns count and rate.
    """
    from dagua.edges import evaluate_bezier

    if not curves or positions.shape[0] == 0:
        return {"edge_node_crossings": 0, "edge_node_crossing_rate": 0.0}

    pos = _ensure_cpu(positions)
    sizes = _ensure_cpu(node_sizes)
    ei = _ensure_cpu(edge_index)
    N = pos.shape[0]
    E = len(curves)

    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    src_list = ei[0].tolist() if ei.numel() > 0 else []
    tgt_list = ei[1].tolist() if ei.numel() > 0 else []

    crossings = 0
    for e_idx, curve in enumerate(curves):
        src_node = src_list[e_idx] if e_idx < len(src_list) else -1
        tgt_node = tgt_list[e_idx] if e_idx < len(tgt_list) else -1

        for t_step in range(1, T):
            t = t_step / T
            px, py = evaluate_bezier(curve, t)
            for n_idx in range(N):
                if n_idx == src_node or n_idx == tgt_node:
                    continue
                nx, ny = pos[n_idx, 0].item(), pos[n_idx, 1].item()
                hw, hh = half_w[n_idx].item(), half_h[n_idx].item()
                if abs(px - nx) < hw and abs(py - ny) < hh:
                    crossings += 1
                    break  # count at most one crossing per sample point

    rate = crossings / max(E * (T - 1), 1)
    return {"edge_node_crossings": crossings, "edge_node_crossing_rate": rate}


def label_overlap_count(
    label_positions: List[Optional[Tuple[float, float]]],
    edge_labels: list,
    positions: torch.Tensor,
    node_sizes: torch.Tensor,
    label_font_size: float = 7.0,
) -> Dict[str, int]:
    """Count overlaps between edge labels and node bboxes, and between labels.

    Tier 1 metric. Uses bbox approximation for label sizes.
    """
    from dagua.utils import measure_text_fallback

    pos = _ensure_cpu(positions)
    sizes = _ensure_cpu(node_sizes)
    N = pos.shape[0]

    # Build label bboxes
    label_bboxes = []
    for i, lp in enumerate(label_positions):
        if lp is None or i >= len(edge_labels) or not edge_labels[i]:
            continue
        lw, lh = measure_text_fallback(edge_labels[i], label_font_size)
        lw += 4.0
        lh += 2.0
        label_bboxes.append((lp[0] - lw / 2, lp[1] - lh / 2, lp[0] + lw / 2, lp[1] + lh / 2))

    # Node bboxes
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2

    # Label-node overlaps
    label_node_overlaps = 0
    for lb in label_bboxes:
        for n_idx in range(N):
            nx, ny = pos[n_idx, 0].item(), pos[n_idx, 1].item()
            hw, hh = half_w[n_idx].item(), half_h[n_idx].item()
            nb = (nx - hw, ny - hh, nx + hw, ny + hh)
            if lb[0] < nb[2] and lb[2] > nb[0] and lb[1] < nb[3] and lb[3] > nb[1]:
                label_node_overlaps += 1

    # Label-label overlaps
    label_overlaps = 0
    for i in range(len(label_bboxes)):
        for j in range(i + 1, len(label_bboxes)):
            a, b = label_bboxes[i], label_bboxes[j]
            if a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]:
                label_overlaps += 1

    return {"label_overlaps": label_overlaps, "label_node_overlaps": label_node_overlaps}


def edge_curvature_consistency(curves) -> Dict[str, float]:
    """Compute curvature statistics across all edges.

    κ at 5 sample points per edge.
    Tier 1 metric.
    """
    from dagua.edges import evaluate_bezier, bezier_tangent

    if not curves:
        return {"edge_curvature_cv": 0.0, "edge_curvature_mean": 0.0}

    import math

    mean_curvatures = []
    for curve in curves:
        kappas = []
        for t_step in range(5):
            t = (t_step + 0.5) / 5
            # First derivative
            dx1, dy1 = bezier_tangent(curve, t)
            # Second derivative (finite difference approximation)
            dt = 0.01
            dx2a, dy2a = bezier_tangent(curve, min(t + dt, 1.0))
            dx2b, dy2b = bezier_tangent(curve, max(t - dt, 0.0))
            ddx = (dx2a - dx2b) / (2 * dt)
            ddy = (dy2a - dy2b) / (2 * dt)

            cross = abs(dx1 * ddy - dy1 * ddx)
            norm = (dx1**2 + dy1**2) ** 1.5
            if norm > 1e-8:
                kappas.append(cross / norm)

        if kappas:
            mean_curvatures.append(sum(kappas) / len(kappas))

    if not mean_curvatures:
        return {"edge_curvature_cv": 0.0, "edge_curvature_mean": 0.0}

    mean_k = sum(mean_curvatures) / len(mean_curvatures)
    if len(mean_curvatures) > 1 and mean_k > 1e-8:
        var_k = sum((k - mean_k) ** 2 for k in mean_curvatures) / (len(mean_curvatures) - 1)
        cv = var_k ** 0.5 / mean_k
    else:
        cv = 0.0

    return {"edge_curvature_cv": cv, "edge_curvature_mean": mean_k}


def port_angular_resolution(
    curves,
    edge_index: torch.Tensor,
) -> Dict[str, float]:
    """Minimum angle between incident edge tangents at actual ports.

    Tier 2 metric.
    """
    from dagua.edges import bezier_tangent

    if not curves or edge_index.numel() == 0:
        return {"port_angular_res_mean_deg": 360.0}

    ei = _ensure_cpu(edge_index)
    E = len(curves)
    src_list = ei[0].tolist()
    tgt_list = ei[1].tolist()

    # Collect tangent vectors per node
    node_tangents: Dict[int, List[Tuple[float, float]]] = {}

    for e_idx, curve in enumerate(curves):
        if e_idx >= len(src_list):
            break
        s, t = src_list[e_idx], tgt_list[e_idx]

        # Source tangent: B'(0)
        sdx, sdy = bezier_tangent(curve, 0.0)
        node_tangents.setdefault(s, []).append((sdx, sdy))

        # Target tangent: B'(1)
        tdx, tdy = bezier_tangent(curve, 1.0)
        node_tangents.setdefault(t, []).append((tdx, tdy))

    min_angles = []
    for node, tangents in node_tangents.items():
        if len(tangents) < 2:
            continue

        # Compute angles
        angles = []
        for dx, dy in tangents:
            angles.append(math.atan2(dy, dx))
        angles.sort()

        # Min angle between adjacent
        diffs = [angles[i + 1] - angles[i] for i in range(len(angles) - 1)]
        diffs.append(2 * math.pi - (angles[-1] - angles[0]))
        min_angles.append(min(diffs))

    if not min_angles:
        return {"port_angular_res_mean_deg": 360.0}

    mean_deg = sum(a * 180 / math.pi for a in min_angles) / len(min_angles)
    return {"port_angular_res_mean_deg": mean_deg}


def within_layer_compactness(pos: torch.Tensor, topo_depth) -> Dict[str, float]:
    """Fraction of each layer's x-range that is occupied by nodes.

    Complexity: O(N).
    """
    if isinstance(topo_depth, torch.Tensor):
        depth_t = topo_depth
    else:
        depth_t = torch.tensor(topo_depth, dtype=torch.long)

    unique_depths = depth_t.unique()
    scores = []

    for d in unique_depths:
        layer_mask = depth_t == d
        layer_nodes = torch.where(layer_mask)[0]
        if layer_nodes.numel() < 2:
            continue

        layer_x = pos[layer_nodes, 0]
        total_width = (layer_x.max() - layer_x.min()).item()
        if total_width < 1e-6:
            scores.append(1.0)
            continue

        n_bins = min(100, layer_nodes.numel())
        hist = torch.histc(layer_x, bins=n_bins)
        occupied = (hist > 0).float().sum() / n_bins
        scores.append(occupied.item())

    if not scores:
        return {"layer_compactness_mean": 1.0, "layer_compactness_min": 1.0}

    scores_t = torch.tensor(scores)
    return {
        "layer_compactness_mean": scores_t.mean().item(),
        "layer_compactness_min": scores_t.min().item(),
    }


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

def composite(metrics: Dict[str, float]) -> float:
    """Weighted combination of normalized metrics.  Higher = better.  Range 0-100.

    Weights:
    - DAG consistency: 25
    - Edge length uniformity (1 - CV): 20
    - Depth correlation: 15
    - No overlaps (binary): 10
    - Edge straightness: 10
    - Crossings (inverted): 10
    - Angular resolution: 5
    - Cluster separation: 5
    """
    score = 0.0

    # DAG consistency (25) — most critical
    score += 25 * metrics.get("dag_consistency", 0.0)

    # Edge length uniformity (20) — invert CV, cap at 1.0
    score += 20 * max(0.0, 1.0 - metrics.get("edge_length_cv", 1.0))

    # Depth correlation (15)
    score += 15 * max(0.0, metrics.get("depth_spearman_rho", 0.0))

    # No overlaps (10) — binary
    score += 10 * (1.0 if metrics.get("overlap_count", 1) == 0 else 0.0)

    # Edge straightness (10) — lower deviation = better
    straight_deg = metrics.get("edge_straightness_mean_deg", 45.0)
    score += 10 * max(0.0, 1.0 - straight_deg / 45.0)

    # Crossing density (10) — lower is better
    crossing_score = max(0.0, 1.0 - metrics.get("crossing_rate", 0.5) * 10)
    score += 10 * crossing_score

    # Angular resolution (5)
    angle_score = min(1.0, metrics.get("angular_res_mean_deg", 20.0) / 40.0)
    score += 5 * angle_score

    # Cluster separation (5)
    if "cluster_mean_sep_ratio" in metrics:
        sep_score = min(1.0, metrics["cluster_mean_sep_ratio"] / 5.0)
        score += 5 * sep_score
    else:
        score += 5 * 0.5  # neutral if no clusters

    # Edge-node crossings (3) — from existing margins, lower is better
    if "edge_node_crossing_rate" in metrics:
        enc_score = max(0.0, 1.0 - metrics["edge_node_crossing_rate"] * 5)
        score += 3 * enc_score

    # Label overlap (2) — fewer is better
    if "label_overlaps" in metrics or "label_node_overlaps" in metrics:
        total_label_overlaps = metrics.get("label_overlaps", 0) + metrics.get("label_node_overlaps", 0)
        lo_score = 1.0 if total_label_overlaps == 0 else max(0.0, 1.0 - total_label_overlaps * 0.1)
        score += 2 * lo_score

    return score


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def quick(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    topo_depth=None,
    node_sizes: Optional[torch.Tensor] = None,
    direction: str = "TB",
    back_edge_mask: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Tier-1 metrics only.  O(N + E), runs in seconds at any scale.

    Args:
        pos: [N, 2] positions.
        edge_index: [2, E] edge tensor.
        topo_depth: [N] int tensor/list of topological depths (computed if None).
        node_sizes: [N, 2] for overlap counting (skipped if None).
        direction: layout direction for DAG consistency.
        back_edge_mask: [E] bool mask of back edges (excluded from dag_consistency).

    Returns:
        Flat dict of metric name -> value.
    """
    t0 = _time.perf_counter()
    pos = _ensure_cpu(pos)
    ei = _ensure_cpu(edge_index)
    N = pos.shape[0]
    E = ei.shape[1] if ei.numel() > 0 else 0

    bem = None
    if back_edge_mask is not None:
        bem = back_edge_mask.cpu() if back_edge_mask.device.type != "cpu" else back_edge_mask

    result: Dict[str, Any] = {
        "_graph_n_nodes": N,
        "_graph_n_edges": E,
    }

    # Edge length CV
    result.update(edge_length_cv(pos, ei))

    # DAG consistency
    result.update(dag_consistency(pos, ei, direction=direction, back_edge_mask=bem))

    # Depth-position correlation
    if topo_depth is None and ei.numel() > 0:
        from dagua.utils import longest_path_layering
        topo_depth = longest_path_layering(ei, N)

    if topo_depth is not None:
        if not isinstance(topo_depth, torch.Tensor):
            topo_depth = torch.tensor(topo_depth, dtype=torch.long)
        result.update(depth_position_correlation(pos, topo_depth))

    # Overlap count
    if node_sizes is not None:
        ns = _ensure_cpu(node_sizes)
        result.update(count_overlaps_detailed(pos, ns))

    # Aspect ratio
    ns_arg = _ensure_cpu(node_sizes) if node_sizes is not None else None
    result.update(aspect_ratio(pos, ns_arg))

    # Edge straightness
    result.update(edge_direction_straightness(pos, ei, direction=direction))

    result["_compute_time_seconds"] = _time.perf_counter() - t0
    return result


def full(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    topo_depth=None,
    node_sizes: Optional[torch.Tensor] = None,
    cluster_ids: Optional[torch.Tensor] = None,
    direction: str = "TB",
    stress_sources: int = 200,
    stress_targets: int = 1000,
    crossing_samples: int = 1_000_000,
    neighborhood_samples: int = 5000,
    curves=None,
    label_positions=None,
    edge_labels=None,
    back_edge_mask: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """All metrics including sampled Tier-2 and DAG-specific Tier-3.

    Args:
        pos: [N, 2] positions.
        edge_index: [2, E] edge tensor.
        topo_depth: [N] topological depths (computed if None).
        node_sizes: [N, 2] for overlap/cluster metrics.
        cluster_ids: [N] cluster assignments for cluster metrics.
        direction: layout direction.
        stress_sources: number of BFS sources for stress sampling.
        stress_targets: number of targets per source.
        crossing_samples: number of edge pairs to sample.
        neighborhood_samples: number of nodes to sample.
        curves: optional BezierCurve list for edge-aware metrics.
        label_positions: optional pre-computed label positions.
        edge_labels: optional edge label list for label overlap metrics.
        back_edge_mask: [E] bool mask of back edges (excluded from dag_consistency).

    Returns:
        Flat dict of all metrics.
    """
    t0 = _time.perf_counter()
    pos = _ensure_cpu(pos)
    ei = _ensure_cpu(edge_index)
    N = pos.shape[0]
    E = ei.shape[1] if ei.numel() > 0 else 0

    # Compute topo_depth if needed
    if topo_depth is None and ei.numel() > 0:
        from dagua.utils import longest_path_layering
        topo_depth = longest_path_layering(ei, N)

    if topo_depth is not None and not isinstance(topo_depth, torch.Tensor):
        topo_depth = torch.tensor(topo_depth, dtype=torch.long)

    # Tier 1
    result = quick(pos, ei, topo_depth=topo_depth, node_sizes=node_sizes,
                   direction=direction, back_edge_mask=back_edge_mask)

    # Tier 2: Sampled metrics
    result.update(sampled_stress(pos, ei, N,
                                 n_sources=stress_sources, n_targets=stress_targets))
    result.update(sampled_crossing_rate(pos, ei, n_samples=crossing_samples))
    result.update(neighborhood_preservation(pos, ei, N, n_samples=neighborhood_samples))
    result.update(angular_resolution(pos, ei))

    # Edge-aware metrics (when curves are provided)
    if curves is not None:
        result.update(edge_curvature_consistency(curves))
        result.update(port_angular_resolution(curves, ei))
        if node_sizes is not None:
            ns = _ensure_cpu(node_sizes)
            result.update(edge_node_crossing_count(curves, pos, ns, ei))
        if label_positions is not None and edge_labels is not None and node_sizes is not None:
            ns = _ensure_cpu(node_sizes)
            result.update(label_overlap_count(label_positions, edge_labels, pos, ns))

    # Tier 3: DAG-specific
    if cluster_ids is not None:
        cluster_ids = _ensure_cpu(cluster_ids)
        result.update(cluster_separation(pos, cluster_ids))

    if topo_depth is not None:
        result.update(layer_uniformity(pos, topo_depth))
        result.update(within_layer_compactness(pos, topo_depth))

    # Composite score
    result["composite_score"] = composite(result)

    result["_compute_time_seconds"] = _time.perf_counter() - t0
    result["_sample_sizes"] = {
        "stress_sources": stress_sources,
        "stress_targets": stress_targets,
        "crossing_samples": crossing_samples,
        "neighborhood_samples": neighborhood_samples,
    }

    return result


def compare(pos_dagua: torch.Tensor, pos_reference: torch.Tensor) -> Dict[str, float]:
    """Procrustes comparison of two layouts of the same graph.

    Complexity: O(N).
    Target: disparity < 0.3 = structurally similar.
    """
    from scipy.spatial import procrustes

    mtx1 = pos_reference.cpu().numpy().astype(np.float64)
    mtx2 = pos_dagua.cpu().numpy().astype(np.float64)

    try:
        _, _, disparity = procrustes(mtx1, mtx2)
        return {"procrustes_disparity": float(disparity)}
    except Exception:
        return {"procrustes_disparity": 1.0}


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

def compute_all_metrics(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    clusters: Optional[Dict] = None,
    direction: str = "TB",
) -> Dict[str, float]:
    """Legacy API — wraps quick() for backward compatibility.

    Returns the same keys as the old implementation plus new ones.
    """
    result = quick(positions, edge_index, node_sizes=node_sizes, direction=direction)

    pos_cpu = _ensure_cpu(positions)
    ei_cpu = _ensure_cpu(edge_index)

    # Map new keys to old keys for backward compat
    compat = {
        "num_nodes": result.get("_graph_n_nodes", 0),
        "num_edges": result.get("_graph_n_edges", 0),
        "edge_crossings": count_crossings(pos_cpu, ei_cpu),
        "dag_fraction": result.get("dag_consistency", 1.0),
        "node_overlaps": result.get("overlap_count", 0),
        "mean_edge_length": result.get("edge_length_mean", 0.0),
        "edge_length_variance": result.get("edge_length_std", 0.0) ** 2,
        "edge_straightness": result.get("edge_straightness_mean_deg", 0.0),
        "x_alignment": compute_x_alignment(pos_cpu, ei_cpu, direction=direction),
        "total_area": result.get("bbox_width", 0.0) * result.get("bbox_height", 0.0),
        "aspect_ratio": result.get("aspect_ratio", 1.0),
    }
    compat["overall_quality"] = overall_quality(compat)
    compat.update(result)
    return compat


def overall_quality(metrics: Dict[str, float]) -> float:
    """Legacy single scalar summarizing layout quality.  Higher is better."""
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


# Legacy function-name aliases (old API returned scalars)

def count_crossings(pos: torch.Tensor, edge_index: torch.Tensor) -> int:
    """Legacy: count edge crossings.  Exact for E<=500, sampled for larger."""
    pos = _ensure_cpu(pos)
    ei = _ensure_cpu(edge_index)
    if ei.numel() == 0 or ei.shape[1] < 2:
        return 0
    E = ei.shape[1]
    src, tgt = ei[0], ei[1]

    if E <= 500:
        crossings = 0
        pos_np = pos.numpy()
        for i in range(E):
            for j in range(i + 1, E):
                a, b = pos_np[src[i]], pos_np[tgt[i]]
                c, d = pos_np[src[j]], pos_np[tgt[j]]
                if _segments_intersect_scalar(a, b, c, d):
                    crossings += 1
        return crossings
    else:
        result = sampled_crossing_rate(pos, ei, n_samples=125000)
        return result["crossing_estimated_total"]


def _segments_intersect_scalar(a, b, c, d) -> bool:
    """Scalar segment intersection test for legacy count_crossings."""
    if np.allclose(a, c) or np.allclose(a, d) or np.allclose(b, c) or np.allclose(b, d):
        return False
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1, d2 = cross(c, d, a), cross(c, d, b)
    d3, d4 = cross(a, b, c), cross(a, b, d)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def compute_dag_fraction(pos: torch.Tensor, edge_index: torch.Tensor,
                         direction: str = "TB") -> float:
    """Legacy: fraction of edges in correct DAG direction."""
    return dag_consistency(_ensure_cpu(pos), _ensure_cpu(edge_index), direction)["dag_consistency"]


def compute_edge_straightness(pos: torch.Tensor, edge_index: torch.Tensor,
                              direction: str = "TB") -> float:
    """Legacy: mean angular deviation from primary axis (degrees)."""
    return edge_direction_straightness(
        _ensure_cpu(pos), _ensure_cpu(edge_index), direction
    )["edge_straightness_mean_deg"]


def compute_x_alignment(pos: torch.Tensor, edge_index: torch.Tensor,
                        direction: str = "TB") -> float:
    """Legacy: mean cross-axis displacement between connected nodes."""
    pos = _ensure_cpu(pos)
    ei = _ensure_cpu(edge_index)
    if ei.numel() == 0:
        return 0.0
    src, tgt = ei[0], ei[1]
    if direction in ("LR", "RL"):
        return (pos[src, 1] - pos[tgt, 1]).abs().mean().item()
    return (pos[src, 0] - pos[tgt, 0]).abs().mean().item()


def count_overlaps(pos: torch.Tensor, node_sizes: torch.Tensor) -> int:
    """Legacy: count overlapping node pairs (returns int)."""
    return count_overlaps_detailed(_ensure_cpu(pos), _ensure_cpu(node_sizes))["overlap_count"]


def compute_mean_edge_length(pos: torch.Tensor, edge_index: torch.Tensor) -> float:
    """Legacy: average Euclidean edge length."""
    return edge_length_cv(_ensure_cpu(pos), _ensure_cpu(edge_index))["edge_length_mean"]


def compute_edge_length_variance(pos: torch.Tensor, edge_index: torch.Tensor) -> float:
    """Legacy: variance of edge lengths."""
    r = edge_length_cv(_ensure_cpu(pos), _ensure_cpu(edge_index))
    return r["edge_length_std"] ** 2


def compute_total_area(pos: torch.Tensor, node_sizes: torch.Tensor) -> float:
    """Legacy: bounding box area."""
    r = aspect_ratio(_ensure_cpu(pos), _ensure_cpu(node_sizes))
    return r["bbox_width"] * r["bbox_height"]


def compute_aspect_ratio(pos: torch.Tensor, node_sizes: torch.Tensor) -> float:
    """Legacy: width / height."""
    return aspect_ratio(_ensure_cpu(pos), _ensure_cpu(node_sizes))["aspect_ratio"]


def compute_min_node_gap(pos: torch.Tensor, node_sizes: torch.Tensor) -> float:
    """Legacy: minimum gap between non-overlapping node bboxes (sampled)."""
    pos = _ensure_cpu(pos)
    ns = _ensure_cpu(node_sizes)
    n = pos.shape[0]
    if n <= 1:
        return float("inf")
    min_gap = float("inf")
    for i in range(min(n, 200)):
        for j in range(i + 1, min(n, 200)):
            dx = abs(pos[i, 0] - pos[j, 0]).item()
            dy = abs(pos[i, 1] - pos[j, 1]).item()
            min_dx = (ns[i, 0] + ns[j, 0]).item() / 2
            min_dy = (ns[i, 1] + ns[j, 1]).item() / 2
            gap_x = dx - min_dx
            gap_y = dy - min_dy
            if gap_x > 0 or gap_y > 0:
                gap = max(gap_x, 0) + max(gap_y, 0)
                min_gap = min(min_gap, gap)
    return min_gap
