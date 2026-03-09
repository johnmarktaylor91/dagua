"""Composable constraint loss functions.

Each function takes pos [N, 2] and relevant graph data, returns scalar loss tensor.
All losses are differentiable through PyTorch autograd.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from dagua.utils import longest_path_layering


def dag_ordering_loss(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    rank_sep: float = 50.0,
) -> torch.Tensor:
    """Targets must be below sources in y-coordinate.

    Margin accounts for node heights + rank separation.
    Competes with: crossing_loss (strict layers vs flexible ordering).
    Cooperates with: edge_straightness_loss.
    """
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=pos.device)

    src, tgt = edge_index[0], edge_index[1]
    margin = (node_sizes[src, 1] + node_sizes[tgt, 1]) / 2 + rank_sep * 0.5
    violation = F.relu(pos[src, 1] - pos[tgt, 1] + margin)
    return violation.mean()


def edge_attraction_loss(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    x_bias: float = 4.0,
) -> torch.Tensor:
    """Connected nodes pull together. x-bias encourages vertical edges.

    Competes with: repulsion_loss.
    Cooperates with: edge_straightness_loss.
    """
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=pos.device)

    src, tgt = edge_index[0], edge_index[1]
    dx = pos[src, 0] - pos[tgt, 0]
    dy = pos[src, 1] - pos[tgt, 1]
    return x_bias * (dx**2).mean() + (dy**2).mean()


def repulsion_loss(
    pos: torch.Tensor,
    num_nodes: int,
    threshold: int = 2000,
    sample_k: int = 128,
) -> torch.Tensor:
    """All nodes repel each other. Exact for small graphs, sampled for large.

    For N <= 2000: exact O(N^2).
    For N > 2000: negative sampling with self-index exclusion.

    Competes with: edge_attraction_loss, cluster_compactness_loss.
    Cooperates with: overlap_avoidance_loss.
    """
    if num_nodes <= 1:
        return torch.tensor(0.0, device=pos.device)

    if num_nodes <= threshold:
        # Exact O(N^2)
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # [N, N, 2]
        dist_sq = (diff**2).sum(dim=2) + 1e-4
        # Zero out self-repulsion
        mask = ~torch.eye(num_nodes, dtype=torch.bool, device=pos.device)
        return (1.0 / dist_sq[mask]).mean()
    else:
        # Negative sampling with self-index exclusion
        k = min(sample_k, num_nodes - 1)
        # Generate random indices and exclude self
        arange = torch.arange(num_nodes, device=pos.device)
        # Sample from [0, num_nodes-1) and shift indices >= self up by 1
        raw_idx = torch.randint(0, num_nodes - 1, (num_nodes, k), device=pos.device)
        self_idx = arange.unsqueeze(1).expand(-1, k)  # [N, k]
        idx = raw_idx + (raw_idx >= self_idx).long()  # shift to exclude self
        diff = pos.unsqueeze(1) - pos[idx]  # [N, k, 2]
        dist_sq = (diff**2).sum(dim=2) + 1e-4
        return (1.0 / dist_sq).mean()


def overlap_avoidance_loss(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    padding: float = 2.0,
) -> torch.Tensor:
    """Soft penalty on bounding box intersection. Only when overlapping on BOTH axes.

    For N <= 500: exact O(N^2).
    For N > 500: grid-based spatial hashing (expected O(N) for sparse layouts).

    Cooperates with: repulsion_loss.
    """
    n = pos.shape[0]
    if n <= 1:
        return torch.tensor(0.0, device=pos.device)

    if n > 500:
        return _overlap_loss_grid(pos, node_sizes, padding)

    # All pairs (exact, for small graphs)
    dx_abs = torch.abs(pos.unsqueeze(0)[:, :, 0] - pos.unsqueeze(1)[:, :, 0])
    dy_abs = torch.abs(pos.unsqueeze(0)[:, :, 1] - pos.unsqueeze(1)[:, :, 1])

    min_dx = (node_sizes.unsqueeze(0)[:, :, 0] + node_sizes.unsqueeze(1)[:, :, 0]) / 2 + padding
    min_dy = (node_sizes.unsqueeze(0)[:, :, 1] + node_sizes.unsqueeze(1)[:, :, 1]) / 2 + padding

    overlap = F.relu(min_dx - dx_abs) * F.relu(min_dy - dy_abs)

    # Zero out diagonal
    mask = ~torch.eye(n, dtype=torch.bool, device=pos.device)
    return overlap[mask].mean()


def _overlap_loss_grid(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    padding: float,
) -> torch.Tensor:
    """Grid-based overlap loss for large graphs. Expected O(N) for sparse layouts.

    Divides space into cells of size max(node_width, node_height) + padding.
    Only checks pairs in same or adjacent (3x3 neighborhood) cells.
    """
    n = pos.shape[0]
    device = pos.device

    # Determine cell size from max node dimension
    max_w = node_sizes[:, 0].max().item()
    max_h = node_sizes[:, 1].max().item()
    cell_size = max(max_w, max_h) + padding
    if cell_size < 1.0:
        cell_size = 1.0

    # Detach positions for grid assignment (no grad through hashing)
    pos_det = pos.detach()

    # Compute cell indices for each node
    cx = torch.floor(pos_det[:, 0] / cell_size).long()
    cy = torch.floor(pos_det[:, 1] / cell_size).long()

    # Build grid -> node list mapping using Python dict
    # (torch operations for the actual loss computation)
    grid: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    cx_list = cx.tolist()
    cy_list = cy.tolist()
    for i in range(n):
        grid[(cx_list[i], cy_list[i])].append(i)

    # Collect candidate pairs from same + adjacent cells
    pair_i_list = []
    pair_j_list = []
    seen_cells = set()

    for (gx, gy), nodes_in_cell in grid.items():
        if not nodes_in_cell:
            continue
        # Check 3x3 neighborhood (only cells we haven't paired with this cell yet)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                neighbor_key = (gx + dx, gy + dy)
                if neighbor_key not in grid:
                    continue
                neighbor_nodes = grid[neighbor_key]

                if neighbor_key == (gx, gy):
                    # Same cell: upper-triangle pairs only
                    for a_idx in range(len(nodes_in_cell)):
                        for b_idx in range(a_idx + 1, len(nodes_in_cell)):
                            pair_i_list.append(nodes_in_cell[a_idx])
                            pair_j_list.append(nodes_in_cell[b_idx])
                elif neighbor_key > (gx, gy):
                    # Cross-cell: only process each cell pair once (ordered)
                    for a in nodes_in_cell:
                        for b in neighbor_nodes:
                            pair_i_list.append(a)
                            pair_j_list.append(b)

    if len(pair_i_list) == 0:
        return torch.tensor(0.0, device=device)

    # Cap the number of pairs to prevent memory issues in pathological cases
    max_pairs = min(len(pair_i_list), n * 256)
    if len(pair_i_list) > max_pairs:
        indices = torch.randperm(len(pair_i_list))[:max_pairs].tolist()
        pair_i_list = [pair_i_list[k] for k in indices]
        pair_j_list = [pair_j_list[k] for k in indices]

    pi = torch.tensor(pair_i_list, dtype=torch.long, device=device)
    pj = torch.tensor(pair_j_list, dtype=torch.long, device=device)

    # Compute overlap for candidate pairs (differentiable through pos)
    dx_abs = torch.abs(pos[pi, 0] - pos[pj, 0])
    dy_abs = torch.abs(pos[pi, 1] - pos[pj, 1])

    min_dx = (node_sizes[pi, 0] + node_sizes[pj, 0]) / 2 + padding
    min_dy = (node_sizes[pi, 1] + node_sizes[pj, 1]) / 2 + padding

    overlap = F.relu(min_dx - dx_abs) * F.relu(min_dy - dy_abs)

    if overlap.numel() == 0:
        return torch.tensor(0.0, device=device)

    return overlap.mean()


def cluster_compactness_loss(
    pos: torch.Tensor,
    clusters: dict,
    device: torch.device,
) -> torch.Tensor:
    """Nodes in same cluster attract their cluster centroid.

    Competes with: repulsion_loss.
    """
    total = torch.tensor(0.0, device=device)
    count = 0
    for name, members in clusters.items():
        if isinstance(members, list) and len(members) > 1:
            idx = torch.tensor(members, device=device, dtype=torch.long)
            centroid = pos[idx].mean(dim=0, keepdim=True)
            total = total + ((pos[idx] - centroid) ** 2).sum(dim=1).mean()
            count += 1

    if count == 0:
        return torch.tensor(0.0, device=device)
    return total / count


def cluster_separation_loss(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    clusters: dict,
    padding: float = 10.0,
    device: torch.device = None,
) -> torch.Tensor:
    """Sibling cluster bounding boxes repel. Differentiable through min/max.

    For > 50 clusters, samples 50 random pairs to keep cost bounded.
    """
    if device is None:
        device = pos.device

    cluster_list = [
        (name, torch.tensor(members, device=device, dtype=torch.long))
        for name, members in clusters.items()
        if isinstance(members, list) and len(members) > 0
    ]

    if len(cluster_list) < 2:
        return torch.tensor(0.0, device=device)

    # Build list of all cluster pairs
    num_clusters = len(cluster_list)
    if num_clusters > 50:
        # Sample 50 random pairs
        all_pairs = []
        max_sample = min(50, num_clusters * (num_clusters - 1) // 2)
        sampled = set()
        attempts = 0
        while len(sampled) < max_sample and attempts < max_sample * 10:
            i = random.randint(0, num_clusters - 1)
            j = random.randint(0, num_clusters - 1)
            if i != j:
                pair = (min(i, j), max(i, j))
                sampled.add(pair)
            attempts += 1
        all_pairs = list(sampled)
    else:
        all_pairs = [
            (i, j)
            for i in range(num_clusters)
            for j in range(i + 1, num_clusters)
        ]

    total = torch.tensor(0.0, device=device)
    for i, j in all_pairs:
        idx_i = cluster_list[i][1]
        idx_j = cluster_list[j][1]

        # Compute bboxes from member positions
        bbox_i_min = pos[idx_i].min(dim=0).values - node_sizes[idx_i].max(dim=0).values / 2 - padding
        bbox_i_max = pos[idx_i].max(dim=0).values + node_sizes[idx_i].max(dim=0).values / 2 + padding
        bbox_j_min = pos[idx_j].min(dim=0).values - node_sizes[idx_j].max(dim=0).values / 2 - padding
        bbox_j_max = pos[idx_j].max(dim=0).values + node_sizes[idx_j].max(dim=0).values / 2 + padding

        overlap_x = F.relu(torch.min(bbox_i_max[0], bbox_j_max[0]) - torch.max(bbox_i_min[0], bbox_j_min[0]))
        overlap_y = F.relu(torch.min(bbox_i_max[1], bbox_j_max[1]) - torch.max(bbox_i_min[1], bbox_j_min[1]))
        total = total + overlap_x * overlap_y

    return total


def crossing_loss(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    alpha: float = 5.0,
    max_pairs: int = 2000,
    layer_assignments: Optional[List[int]] = None,
) -> torch.Tensor:
    """Differentiable crossing proxy using adjacent-layer sigmoid relaxation.

    For each pair of edges between the same two adjacent layers, a crossing
    occurs when the x-ordering of sources is inverted relative to the
    x-ordering of targets: sigmoid(-alpha * (x_u1 - x_u2) * (x_v1 - x_v2)).

    For non-adjacent edges (spanning multiple layers), they are decomposed
    into segments by computing virtual x-positions at intermediate layers
    via linear interpolation.

    If layer_assignments is None, falls back to the original all-pairs proxy.

    Competes with: dag_ordering_loss (strict layers vs flexible ordering).
    """
    num_edges = edge_index.shape[1]
    if num_edges < 2:
        return torch.tensor(0.0, device=pos.device)

    if layer_assignments is None:
        return _crossing_loss_fallback(pos, edge_index, alpha, max_pairs)

    return _crossing_loss_layered(pos, edge_index, alpha, max_pairs, layer_assignments)


def _crossing_loss_fallback(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    alpha: float,
    max_pairs: int,
) -> torch.Tensor:
    """Original crossing proxy for when layer info is unavailable.

    Only used as fallback — the layered version is preferred.
    """
    num_edges = edge_index.shape[1]

    # Sample edge pairs if too many
    if num_edges * (num_edges - 1) // 2 > max_pairs:
        n_sample = min(max_pairs, num_edges)
        perm = torch.randperm(num_edges, device=pos.device)[:n_sample]
        ei = edge_index[:, perm]
    else:
        ei = edge_index
        n_sample = ei.shape[1]

    # Vectorized: compute source and target x-positions
    src_x = pos[ei[0], 0]
    tgt_x = pos[ei[1], 0]

    # Check pairs
    n = ei.shape[1]
    if n > 200:
        n_pairs = min(max_pairs, n * (n - 1) // 2)
        i_idx = torch.randint(0, n, (n_pairs,), device=pos.device)
        j_idx = torch.randint(0, n, (n_pairs,), device=pos.device)
        mask = i_idx != j_idx
        i_idx, j_idx = i_idx[mask], j_idx[mask]
    else:
        i_idx, j_idx = torch.triu_indices(n, n, offset=1, device=pos.device)

    if i_idx.numel() == 0:
        return torch.tensor(0.0, device=pos.device)

    dx_src = src_x[i_idx] - src_x[j_idx]
    dx_tgt = tgt_x[i_idx] - tgt_x[j_idx]

    crossing_proxy = torch.sigmoid(-alpha * dx_src * dx_tgt)
    return crossing_proxy.sum()


def _crossing_loss_layered(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    alpha: float,
    max_pairs: int,
    layer_assignments: List[int],
) -> torch.Tensor:
    """Adjacent-layer crossing loss with virtual node decomposition (vectorized).

    For edges spanning multiple layers, creates virtual x-positions at each
    intermediate layer via linear interpolation. Then for each pair of adjacent
    layers, counts soft inversions among the edge segments crossing those layers.

    Fully vectorized: no per-edge Python loops. Uses repeat_interleave to
    expand multi-span edges into segments.
    """
    device = pos.device
    num_edges = edge_index.shape[1]

    layers_t = torch.tensor(layer_assignments, dtype=torch.long, device=device)

    src = edge_index[0]
    tgt = edge_index[1]
    src_layer = layers_t[src]
    tgt_layer = layers_t[tgt]

    # Ensure src is in the lower layer (smaller layer index)
    needs_swap = src_layer > tgt_layer
    actual_src = torch.where(needs_swap, tgt, src)
    actual_tgt = torch.where(needs_swap, src, tgt)
    actual_src_layer = torch.where(needs_swap, tgt_layer, src_layer)
    actual_tgt_layer = torch.where(needs_swap, src_layer, tgt_layer)

    # Compute spans; skip zero-span edges (same layer)
    span = actual_tgt_layer - actual_src_layer  # [E]
    valid = span > 0
    if not valid.any():
        return torch.tensor(0.0, device=device)

    actual_src_layer = actual_src_layer[valid]
    actual_tgt_layer = actual_tgt_layer[valid]
    actual_src_v = actual_src[valid]
    actual_tgt_v = actual_tgt[valid]
    span_v = span[valid]  # [E_valid], all > 0

    src_x = pos[actual_src_v, 0]  # [E_valid]
    tgt_x = pos[actual_tgt_v, 0]  # [E_valid]
    span_f = span_v.float()  # [E_valid]

    # Cap total segments to prevent memory explosion
    max_total_segments = max(num_edges * 4, 50000)
    total_segments = span_v.sum().item()

    if total_segments > max_total_segments:
        # Sample a subset of edges to keep segment count bounded
        n_edges_valid = span_v.shape[0]
        # Estimate: keep edges proportionally so total segments ~ max_total_segments
        # Prefer short-span edges (they're more informative per segment)
        sample_n = min(n_edges_valid, max(max_total_segments // max(span_v.float().mean().long().item(), 1), 100))
        perm = torch.randperm(n_edges_valid, device=device)[:sample_n]
        actual_src_layer = actual_src_layer[perm]
        actual_tgt_layer = actual_tgt_layer[perm]
        src_x = src_x[perm]
        tgt_x = tgt_x[perm]
        span_v = span_v[perm]
        span_f = span_v.float()

    # Vectorized segment expansion using repeat_interleave
    # For each edge with span s, create s segments (layers k=0..s-1)
    # seg_edge_idx: which edge each segment belongs to
    seg_edge_idx = torch.repeat_interleave(
        torch.arange(span_v.shape[0], device=device), span_v
    )

    # For each segment, compute its offset k within the edge's span
    # cumsum of spans gives ending positions; offsets are positions within each group
    offsets = torch.arange(seg_edge_idx.shape[0], device=device)
    cum_spans = torch.zeros(span_v.shape[0] + 1, dtype=torch.long, device=device)
    cum_spans[1:] = span_v.cumsum(0)
    seg_k = offsets - cum_spans[seg_edge_idx]  # k = 0, 1, ..., span-1 for each edge

    # Compute segment layer, x_from, x_to via vectorized interpolation
    seg_layers = actual_src_layer[seg_edge_idx] + seg_k

    seg_frac_from = seg_k.float() / span_f[seg_edge_idx]
    seg_frac_to = (seg_k.float() + 1) / span_f[seg_edge_idx]

    seg_src_x = src_x[seg_edge_idx]
    seg_tgt_x = tgt_x[seg_edge_idx]
    seg_x_from = seg_src_x + (seg_tgt_x - seg_src_x) * seg_frac_from
    seg_x_to = seg_src_x + (seg_tgt_x - seg_src_x) * seg_frac_to

    # Group segments by layer and compute pairwise crossing proxy.
    # For efficiency, use a sampling strategy rather than iterating all layers.
    # Sort segments by layer, then sample pairs within same-layer groups.
    n_segs_total = seg_layers.shape[0]

    if n_segs_total < 2:
        return torch.tensor(0.0, device=device)

    # Strategy: sample random pairs from the full segment array, but only
    # count pairs that share the same layer. This avoids per-layer Python loops.
    n_sample_attempts = min(max_pairs * 4, n_segs_total * 20)

    # Sort by layer for efficient same-layer pair generation
    sort_idx = seg_layers.argsort()
    sorted_layers = seg_layers[sort_idx]
    sorted_x_from = seg_x_from[sort_idx]
    sorted_x_to = seg_x_to[sort_idx]

    # Find layer boundaries using unique_consecutive
    unique_layers, counts = sorted_layers.unique_consecutive(return_counts=True)
    # Only consider layers with 2+ segments
    multi_mask = counts >= 2
    if not multi_mask.any():
        return torch.tensor(0.0, device=device)

    multi_counts = counts[multi_mask]
    # Compute offsets into sorted array for each layer group
    offsets = torch.zeros(counts.shape[0] + 1, dtype=torch.long, device=device)
    offsets[1:] = counts.cumsum(0)
    multi_offsets = offsets[:-1][multi_mask]

    # For layers with few segments, we can do exact; for many, sample
    # Build pairs by sampling within each group
    pair_i_list = []
    pair_j_list = []

    # Fast path: if total segments is small, do all-pairs per layer
    total_possible_pairs = ((multi_counts * (multi_counts - 1)) // 2).sum().item()

    if total_possible_pairs <= max_pairs:
        # Exact: enumerate all pairs per layer
        for k in range(multi_counts.shape[0]):
            off = multi_offsets[k].item()
            cnt = multi_counts[k].item()
            idx_i, idx_j = torch.triu_indices(cnt, cnt, offset=1, device=device)
            pair_i_list.append(idx_i + off)
            pair_j_list.append(idx_j + off)
    else:
        # Sampling: proportional to number of pairs per layer, capped at max_pairs total
        # Weight layers by their number of pairs
        pairs_per_layer = (multi_counts * (multi_counts - 1)) // 2
        total_possible = pairs_per_layer.sum().float()
        samples_per_layer = (pairs_per_layer.float() / total_possible * max_pairs).long()
        samples_per_layer = samples_per_layer.clamp(min=1)

        for k in range(multi_counts.shape[0]):
            off = multi_offsets[k].item()
            cnt = multi_counts[k].item()
            n_samp = min(samples_per_layer[k].item(), cnt * (cnt - 1) // 2)

            if cnt <= 200 and cnt * (cnt - 1) // 2 <= n_samp:
                idx_i, idx_j = torch.triu_indices(cnt, cnt, offset=1, device=device)
            else:
                idx_i = torch.randint(0, cnt, (n_samp,), device=device)
                idx_j = torch.randint(0, cnt, (n_samp,), device=device)
                valid_mask = idx_i != idx_j
                idx_i, idx_j = idx_i[valid_mask], idx_j[valid_mask]

            pair_i_list.append(idx_i + off)
            pair_j_list.append(idx_j + off)

    if not pair_i_list:
        return torch.tensor(0.0, device=device)

    all_i = torch.cat(pair_i_list)
    all_j = torch.cat(pair_j_list)

    if all_i.numel() == 0:
        return torch.tensor(0.0, device=device)

    dx_from = sorted_x_from[all_i] - sorted_x_from[all_j]
    dx_to = sorted_x_to[all_i] - sorted_x_to[all_j]

    crossing_proxy = torch.sigmoid(-alpha * dx_from * dx_to)
    # Use sum (not mean) so loss scales with number of crossings,
    # giving sufficient gradient magnitude to compete with attraction.
    return crossing_proxy.sum()


def edge_straightness_loss(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Penalize horizontal displacement between connected nodes.

    Cooperates with: edge_attraction_loss (x_bias), dag_ordering_loss.
    """
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=pos.device)

    src, tgt = edge_index[0], edge_index[1]
    dx = pos[src, 0] - pos[tgt, 0]
    return (dx**2).mean()


def edge_length_variance_loss(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Penalize variance of edge lengths. Uniform > minimum.

    Cooperates with: edge_attraction_loss.
    """
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=pos.device)

    src, tgt = edge_index[0], edge_index[1]
    lengths = ((pos[src] - pos[tgt]) ** 2).sum(dim=1).sqrt()
    if lengths.numel() <= 1:
        return torch.tensor(0.0, device=pos.device)
    return lengths.var()
