"""Composable constraint loss functions.

Each function takes pos [N, 2] and relevant graph data, returns scalar loss tensor.
All losses are differentiable through PyTorch autograd.

Scaling strategy (Sprint 3 — fully vectorized):
- ALL operations use scatter/segment tensor ops — ZERO per-layer Python loops
- Repulsion: sample K neighbors from same/adjacent layers via layer_offsets indexing
- Overlap: same sampling approach with bounding-box intersection
- Size-aware repulsion from AMD GPU layout patterns
- torch.where everywhere (no CPU-GPU sync from .any() checks)
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from dagua.layout.layers import LayerIndex
from dagua.utils import longest_path_layering


# ─── Edge-based losses (O(E), trivially parallelizable) ─────────────────────


def _non_self_edges(edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return source/target indices with self-loops removed."""
    src, tgt = edge_index[0], edge_index[1]
    keep = src != tgt
    return src[keep], tgt[keep]


def dag_ordering_loss(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    rank_sep: float = 50.0,
) -> torch.Tensor:
    """Targets must be below sources in y-coordinate."""
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=pos.device)

    src, tgt = _non_self_edges(edge_index)
    if src.numel() == 0:
        return torch.tensor(0.0, device=pos.device)
    margin = (node_sizes[src, 1] + node_sizes[tgt, 1]) / 2 + rank_sep * 0.5
    violation = F.relu(pos[src, 1] - pos[tgt, 1] + margin)
    return violation.mean()


def edge_attraction_loss(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    x_bias: float = 4.0,
) -> torch.Tensor:
    """Connected nodes pull together. x-bias encourages vertical edges.

    AMD insight: cap attraction at 1/3 distance to prevent overshoot.
    """
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=pos.device)

    src, tgt = _non_self_edges(edge_index)
    if src.numel() == 0:
        return torch.tensor(0.0, device=pos.device)
    diff = pos[src] - pos[tgt]  # [E, 2]
    dist_sq = (diff ** 2).sum(dim=1)  # [E]

    # Cap: attraction force proportional to dist_sq, but capped at 1/3 of distance
    # This prevents nodes from overshooting past their targets
    dist = dist_sq.sqrt()
    max_force = dist / 3.0  # 1/3 of distance
    force = dist.clamp(max=1.0)  # normalized force magnitude
    cap = torch.where(force > max_force, max_force / (force + 1e-8), torch.ones_like(force))

    dx = diff[:, 0]
    dy = diff[:, 1]
    return x_bias * (dx ** 2 * cap).mean() + (dy ** 2 * cap).mean()


def edge_straightness_loss(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Penalize horizontal displacement between connected nodes."""
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=pos.device)

    src, tgt = _non_self_edges(edge_index)
    if src.numel() == 0:
        return torch.tensor(0.0, device=pos.device)
    dx = pos[src, 0] - pos[tgt, 0]
    return (dx**2).mean()


def edge_length_variance_loss(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Penalize variance of edge lengths."""
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=pos.device)

    src, tgt = _non_self_edges(edge_index)
    if src.numel() <= 1:
        return torch.tensor(0.0, device=pos.device)
    lengths = ((pos[src] - pos[tgt]) ** 2).sum(dim=1).add(1e-8).sqrt()
    if lengths.numel() <= 1:
        return torch.tensor(0.0, device=pos.device)
    return lengths.var()


# ─── Repulsion (fully vectorized — no per-layer Python loops) ────────────────


def repulsion_loss(
    pos: torch.Tensor,
    num_nodes: int,
    threshold: int = 2000,
    sample_k: int = 128,
    layer_index: Optional[LayerIndex] = None,
    node_sizes: Optional[torch.Tensor] = None,
    rvs_threshold: int = 5000,
    rvs_nn_k: int = 20,
) -> torch.Tensor:
    """All nodes repel each other.

    Tiered strategy:
    - N <= threshold: exact O(N^2)
    - threshold < N <= rvs_threshold with layer_index: layer-local scatter sampling
    - N > rvs_threshold: RVS (Random Vertex Sampling) — O(N^(3/4) * N^(1/4) + N*K_nn)
    - fallback: global negative sampling
    """
    if num_nodes <= 1:
        return torch.tensor(0.0, device=pos.device)

    if num_nodes <= threshold:
        return _repulsion_exact(pos, num_nodes, node_sizes)

    # RVS for large graphs (>5K nodes)
    if num_nodes > rvs_threshold and layer_index is not None:
        return _repulsion_rvs(pos, layer_index, sample_k, rvs_nn_k, node_sizes)

    if layer_index is not None:
        return _repulsion_scatter(pos, layer_index, sample_k, node_sizes)

    return _repulsion_sampled(pos, num_nodes, sample_k)


def _repulsion_exact(
    pos: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Exact O(N^2) repulsion with size-aware scaling (AMD pattern)."""
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # [N, N, 2]
    dist_sq = (diff**2).sum(dim=2) + 1e-4
    mask = ~torch.eye(num_nodes, dtype=torch.bool, device=pos.device)

    if node_sizes is not None:
        # Size-aware: larger nodes repel harder (AMD pattern)
        # Scale by combined bounding box area proxy
        combined_w = node_sizes[:, 0].unsqueeze(0) + node_sizes[:, 0].unsqueeze(1)
        combined_h = node_sizes[:, 1].unsqueeze(0) + node_sizes[:, 1].unsqueeze(1)
        size_factor = (combined_w * combined_h) / (combined_w * combined_h).mean()
        return (size_factor[mask] / dist_sq[mask]).mean()

    return (1.0 / dist_sq[mask]).mean()


def _repulsion_sampled(pos: torch.Tensor, num_nodes: int, sample_k: int) -> torch.Tensor:
    """Global negative sampling with self-index exclusion."""
    k = min(sample_k, num_nodes - 1)
    arange = torch.arange(num_nodes, device=pos.device)
    raw_idx = torch.randint(0, num_nodes - 1, (num_nodes, k), device=pos.device)
    self_idx = arange.unsqueeze(1).expand(-1, k)
    idx = raw_idx + (raw_idx >= self_idx).long()
    diff = pos.unsqueeze(1) - pos[idx]  # [N, k, 2]
    dist_sq = (diff**2).sum(dim=2) + 1e-4
    return (1.0 / dist_sq).mean()


def _repulsion_scatter(
    pos: torch.Tensor,
    layer_index: LayerIndex,
    sample_k: int,
    node_sizes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fully vectorized layer-local repulsion — ZERO per-layer Python loops.

    For each node, samples K neighbors from the same and adjacent layers
    using the pre-computed layer_offsets. All operations are batched tensor ops.

    AMD insights applied:
    - Size-aware repulsion: scale by (w1+w2)*(h1+h2)
    - torch.where everywhere (no CPU-GPU sync)
    """
    device = pos.device
    N = pos.shape[0]
    K = min(sample_k, N - 1)
    if K <= 0:
        return torch.tensor(0.0, device=device)

    layers = layer_index.node_to_layer  # [N]
    offsets = layer_index.layer_offsets  # [L+1]
    sorted_nodes = layer_index.sorted_nodes  # [N]
    num_layers = layer_index.num_layers

    # For each node, compute the sampling range: nodes in [layer-1, layer+1]
    adj_layer_lo = (layers - 1).clamp(min=0)  # [N]
    adj_layer_hi = (layers + 2).clamp(max=num_layers)  # [N]

    adj_start = offsets[adj_layer_lo]  # [N] — start index in sorted_nodes
    adj_end = offsets[adj_layer_hi]  # [N] — end index in sorted_nodes
    range_size = (adj_end - adj_start).float()  # [N]

    # Sample K indices within each node's [adj_start, adj_end) range
    rand = torch.rand(N, K, device=device)  # [N, K] in [0, 1)
    sample_offsets = adj_start.unsqueeze(1) + (rand * range_size.unsqueeze(1)).long()  # [N, K]
    sample_offsets = sample_offsets.clamp(max=N - 1)

    # Map to actual node indices
    sampled = sorted_nodes[sample_offsets]  # [N, K]

    # Exclude self-pairs (unconditional — no .any() check)
    self_idx = torch.arange(N, device=device).unsqueeze(1)  # [N, 1]
    not_self = sampled != self_idx  # [N, K] bool

    # Compute repulsion
    diff = pos.unsqueeze(1) - pos[sampled]  # [N, K, 2]
    dist_sq = (diff ** 2).sum(dim=2) + 1e-4  # [N, K]

    if node_sizes is not None:
        # Size-aware repulsion (AMD pattern): scale by combined size
        src_w = node_sizes[:, 0].unsqueeze(1).expand(-1, K)  # [N, K]
        src_h = node_sizes[:, 1].unsqueeze(1).expand(-1, K)  # [N, K]
        tgt_w = node_sizes[sampled, 0]  # [N, K]
        tgt_h = node_sizes[sampled, 1]  # [N, K]
        combined_size = (src_w + tgt_w) * (src_h + tgt_h)
        mean_size = combined_size.mean()
        size_factor = combined_size / (mean_size + 1e-8)
        repulsion = size_factor / dist_sq
    else:
        repulsion = 1.0 / dist_sq

    # Mask out self-pairs (unconditional torch.where — no CPU sync)
    repulsion = torch.where(not_self, repulsion, torch.zeros_like(repulsion))

    valid_count = not_self.sum().float()
    return torch.where(
        valid_count > 0,
        repulsion.sum() / valid_count,
        torch.tensor(0.0, device=device),
    )


def _repulsion_rvs(
    pos: torch.Tensor,
    layer_index: LayerIndex,
    sample_k: int,
    nn_k: int,
    node_sizes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Random Vertex Sampling (RVS) repulsion for very large graphs.

    Key idea from scaling memo: select N^(3/4) active nodes to update,
    each gets N^(1/4) random samples + K nearest neighbors within adjacent layers.

    O(N^(3/4) * (N^(1/4) + K_nn)) per step — near-linear for practical K_nn.

    AMD insights applied:
    - Size-aware repulsion: scale by (w1+w2)*(h1+h2)
    - torch.where everywhere (no CPU-GPU sync)
    """
    device = pos.device
    N = pos.shape[0]

    # Determine active set size and random sample count
    # Cap at 1M for N > 100M to avoid multi-GB intermediate tensors
    n_active = min(max(int(N ** 0.75), min(N, 256)), 1_000_000)
    n_random = max(int(N ** 0.25), 4)
    K_nn = min(nn_k, N - 1)
    K_total = n_random + K_nn

    layers = layer_index.node_to_layer
    offsets = layer_index.layer_offsets
    sorted_nodes = layer_index.sorted_nodes
    num_layers = layer_index.num_layers

    # Select active nodes uniformly at random (randint avoids [N] allocation)
    active_idx = torch.randint(0, N, (n_active,), device=device)  # [A]
    A = active_idx.shape[0]

    # For each active node, compute its adjacent-layer range
    active_layers = layers[active_idx]  # [A]
    adj_lo = (active_layers - 1).clamp(min=0)
    adj_hi = (active_layers + 2).clamp(max=num_layers)
    adj_start = offsets[adj_lo]  # [A]
    adj_end = offsets[adj_hi]  # [A]
    range_size = (adj_end - adj_start).float()  # [A]

    # Part 1: Random samples from adjacent layers
    rand = torch.rand(A, n_random, device=device)
    rand_offsets = adj_start.unsqueeze(1) + (rand * range_size.unsqueeze(1)).long()
    rand_offsets = rand_offsets.clamp(max=N - 1)
    rand_sampled = sorted_nodes[rand_offsets]  # [A, n_random]

    # Part 2: Approximate nearest neighbors within same layer
    # Sort by x-position within same layer, take K_nn nearest in sort order
    # This is O(N log N) once, then O(K_nn) per active node
    same_start = offsets[active_layers]  # [A]
    same_end = offsets[active_layers + 1]  # [A]
    same_range = (same_end - same_start).float()  # [A]

    if K_nn > 0:
        # Pure random sampling within same-layer bounds — simpler and faster
        # than the offset-based "nearest" approach, with equivalent quality
        # at large N where random samples are dense enough.
        rand_nn = torch.rand(A, K_nn, device=device)
        nn_indices = (same_start.unsqueeze(1) + (rand_nn * same_range.unsqueeze(1)).long()).clamp(min=0, max=N - 1)
        nn_sampled = sorted_nodes[nn_indices]  # [A, K_nn]
    else:
        nn_sampled = torch.zeros(A, 0, dtype=torch.long, device=device)

    # Combine random + nearest-neighbor samples
    all_sampled = torch.cat([rand_sampled, nn_sampled], dim=1)  # [A, K_total]
    K = all_sampled.shape[1]

    # Exclude self-pairs
    self_idx = active_idx.unsqueeze(1)  # [A, 1]
    not_self = all_sampled != self_idx  # [A, K]

    # Compute repulsion
    active_pos = pos[active_idx]  # [A, 2]
    sample_pos = pos[all_sampled]  # [A, K, 2]
    diff = active_pos.unsqueeze(1) - sample_pos  # [A, K, 2]
    dist_sq = (diff ** 2).sum(dim=2) + 1e-4  # [A, K]

    if node_sizes is not None:
        src_w = node_sizes[active_idx, 0].unsqueeze(1).expand(-1, K)
        src_h = node_sizes[active_idx, 1].unsqueeze(1).expand(-1, K)
        tgt_w = node_sizes[all_sampled, 0]
        tgt_h = node_sizes[all_sampled, 1]
        combined_size = (src_w + tgt_w) * (src_h + tgt_h)
        mean_size = combined_size.mean()
        size_factor = combined_size / (mean_size + 1e-8)
        repulsion = size_factor / dist_sq
    else:
        repulsion = 1.0 / dist_sq

    repulsion = torch.where(not_self, repulsion, torch.zeros_like(repulsion))

    valid_count = not_self.sum().float()
    return torch.where(
        valid_count > 0,
        repulsion.sum() / valid_count,
        torch.tensor(0.0, device=device),
    )


# ─── Overlap avoidance (fully vectorized — no per-layer Python loops) ────────


def overlap_avoidance_loss(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    padding: float = 2.0,
    layer_index: Optional[LayerIndex] = None,
    rvs_threshold: int = 100000,
) -> torch.Tensor:
    """Soft penalty on bounding box intersection.

    For N <= 500: exact O(N^2).
    For N > 500 with layer_index: vectorized same-layer sampling (ZERO Python loops).
    For N > 500 without layer_index: vectorized grid-based.
    """
    n = pos.shape[0]
    if n <= 1:
        return torch.tensor(0.0, device=pos.device)

    if n <= 500:
        return _overlap_exact(pos, node_sizes, padding)

    if n > rvs_threshold and layer_index is not None:
        return _overlap_active_subset(pos, node_sizes, padding, layer_index)

    if layer_index is not None:
        return _overlap_scatter(pos, node_sizes, padding, layer_index)

    return _overlap_grid_vectorized(pos, node_sizes, padding)


def _overlap_exact(pos: torch.Tensor, node_sizes: torch.Tensor, padding: float) -> torch.Tensor:
    """All-pairs overlap for small graphs."""
    n = pos.shape[0]
    dx_abs = torch.abs(pos.unsqueeze(0)[:, :, 0] - pos.unsqueeze(1)[:, :, 0])
    dy_abs = torch.abs(pos.unsqueeze(0)[:, :, 1] - pos.unsqueeze(1)[:, :, 1])
    min_dx = (node_sizes.unsqueeze(0)[:, :, 0] + node_sizes.unsqueeze(1)[:, :, 0]) / 2 + padding
    min_dy = (node_sizes.unsqueeze(0)[:, :, 1] + node_sizes.unsqueeze(1)[:, :, 1]) / 2 + padding
    overlap = F.relu(min_dx - dx_abs) * F.relu(min_dy - dy_abs)
    mask = ~torch.eye(n, dtype=torch.bool, device=pos.device)
    return overlap[mask].mean()


def _overlap_scatter(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    padding: float,
    layer_index: LayerIndex,
) -> torch.Tensor:
    """Fully vectorized same-layer overlap — ZERO per-layer Python loops.

    Samples K neighbors from the same layer for each node.
    Cross-layer overlaps are impossible (rank_sep separates them).
    """
    device = pos.device
    N = pos.shape[0]
    K = min(128, N - 1)
    if K <= 0:
        return torch.tensor(0.0, device=device)

    layers = layer_index.node_to_layer  # [N]
    offsets = layer_index.layer_offsets  # [L+1]
    sorted_nodes = layer_index.sorted_nodes  # [N]

    # For each node, sample from same layer only (cross-layer separated by rank_sep)
    layer_start = offsets[layers]  # [N]
    layer_end = offsets[layers + 1]  # [N]
    range_size = (layer_end - layer_start).float()  # [N]

    # Sample K indices within same layer
    rand = torch.rand(N, K, device=device)
    sample_offsets = layer_start.unsqueeze(1) + (rand * range_size.unsqueeze(1)).long()
    sample_offsets = sample_offsets.clamp(max=N - 1)
    sampled = sorted_nodes[sample_offsets]  # [N, K]

    # Exclude self-pairs
    self_idx = torch.arange(N, device=device).unsqueeze(1)
    not_self = sampled != self_idx  # [N, K]

    # Compute bounding box overlap
    half_w_src = node_sizes[:, 0].unsqueeze(1).expand(-1, K) / 2  # [N, K]
    half_h_src = node_sizes[:, 1].unsqueeze(1).expand(-1, K) / 2
    half_w_tgt = node_sizes[sampled, 0] / 2  # [N, K]
    half_h_tgt = node_sizes[sampled, 1] / 2

    dx_abs = torch.abs(pos[:, 0].unsqueeze(1) - pos[sampled, 0])  # [N, K]
    dy_abs = torch.abs(pos[:, 1].unsqueeze(1) - pos[sampled, 1])

    min_dx = half_w_src + half_w_tgt + padding
    min_dy = half_h_src + half_h_tgt + padding

    overlap = F.relu(min_dx - dx_abs) * F.relu(min_dy - dy_abs)  # [N, K]

    # Mask out self-pairs (unconditional torch.where)
    overlap = torch.where(not_self, overlap, torch.zeros_like(overlap))

    valid_count = not_self.sum().float()
    return torch.where(
        valid_count > 0,
        overlap.sum() / valid_count,
        torch.tensor(0.0, device=device),
    )


def _overlap_active_subset(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    padding: float,
    layer_index: LayerIndex,
) -> torch.Tensor:
    """RVS-style overlap for very large graphs (N > 100K).

    Instead of computing overlap for ALL N nodes with K=128 neighbors
    (creating [N, 128] tensors = GB at millions of nodes), select an
    active subset of N^(3/4) nodes and sample K neighbors for each.

    At 5M nodes: active=88K, K=64 → [88K, 64] ≈ 22MB vs [5M, 128] ≈ 2.5GB.
    """
    device = pos.device
    N = pos.shape[0]

    # Cap at 1M for N > 100M to avoid multi-GB intermediate tensors
    n_active = min(max(int(N ** 0.75), min(N, 256)), 1_000_000)
    K = min(64, N - 1)
    if K <= 0:
        return torch.tensor(0.0, device=device)

    active_idx = torch.randint(0, N, (n_active,), device=device)
    A = active_idx.shape[0]

    layers = layer_index.node_to_layer
    offsets = layer_index.layer_offsets
    sorted_nodes = layer_index.sorted_nodes

    # Sample K neighbors from same layer for each active node
    active_layers = layers[active_idx]
    layer_start = offsets[active_layers]
    layer_end = offsets[active_layers + 1]
    range_size = (layer_end - layer_start).float()

    rand = torch.rand(A, K, device=device)
    sample_offsets = layer_start.unsqueeze(1) + (rand * range_size.unsqueeze(1)).long()
    sample_offsets = sample_offsets.clamp(max=N - 1)
    sampled = sorted_nodes[sample_offsets]  # [A, K]

    # Exclude self-pairs
    self_idx = active_idx.unsqueeze(1)
    not_self = sampled != self_idx  # [A, K]

    # Compute bounding box overlap
    half_w_src = node_sizes[active_idx, 0].unsqueeze(1).expand(-1, K) / 2
    half_h_src = node_sizes[active_idx, 1].unsqueeze(1).expand(-1, K) / 2
    half_w_tgt = node_sizes[sampled, 0] / 2
    half_h_tgt = node_sizes[sampled, 1] / 2

    dx_abs = torch.abs(pos[active_idx, 0].unsqueeze(1) - pos[sampled, 0])
    dy_abs = torch.abs(pos[active_idx, 1].unsqueeze(1) - pos[sampled, 1])

    min_dx = half_w_src + half_w_tgt + padding
    min_dy = half_h_src + half_h_tgt + padding

    overlap = F.relu(min_dx - dx_abs) * F.relu(min_dy - dy_abs)

    overlap = torch.where(not_self, overlap, torch.zeros_like(overlap))

    valid_count = not_self.sum().float()
    return torch.where(
        valid_count > 0,
        overlap.sum() / valid_count,
        torch.tensor(0.0, device=device),
    )


def _overlap_grid_vectorized(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    padding: float,
) -> torch.Tensor:
    """Vectorized grid-based overlap for large graphs without layer info.

    Grid construction and pair-finding use pure tensor ops — no Python loops.
    """
    n = pos.shape[0]
    device = pos.device

    max_w = node_sizes[:, 0].max().item()
    max_h = node_sizes[:, 1].max().item()
    cell_size = max(max_w, max_h) + padding
    if cell_size < 1.0:
        cell_size = 1.0

    pos_det = pos.detach()

    # Assign cells
    cx = torch.floor(pos_det[:, 0] / cell_size).long()
    cy = torch.floor(pos_det[:, 1] / cell_size).long()

    # Encode cell as single key for sorting
    cx_min = cx.min()
    cy_min = cy.min()
    cx_rel = cx - cx_min
    cy_rel = cy - cy_min
    cy_range = cy_rel.max().item() + 1
    cell_keys = cx_rel * cy_range + cy_rel

    # Sort nodes by cell
    sort_idx = cell_keys.argsort()
    sorted_keys = cell_keys[sort_idx]

    # Find cell boundaries
    changes = torch.where(sorted_keys[1:] != sorted_keys[:-1])[0] + 1
    starts = torch.cat([torch.zeros(1, dtype=torch.long, device=device), changes])
    ends = torch.cat([changes, torch.tensor([n], dtype=torch.long, device=device)])

    cell_sizes_arr = ends - starts
    multi_mask = cell_sizes_arr >= 2
    multi_starts = starts[multi_mask]
    multi_ends = ends[multi_mask]

    n_multi = multi_starts.shape[0]
    if n_multi == 0:
        return torch.tensor(0.0, device=device)

    # Cap cells processed
    max_cells = 1000
    if n_multi > max_cells:
        perm = torch.randperm(n_multi, device=device)[:max_cells]
        multi_starts = multi_starts[perm]
        multi_ends = multi_ends[perm]
        n_multi = max_cells

    multi_counts = multi_ends - multi_starts

    # Pre-fetch cell boundaries to CPU once to avoid per-iteration GPU sync
    starts_cpu = multi_starts.cpu().tolist()
    ends_cpu = multi_ends.cpu().tolist()

    # Batch small cells together for vectorized processing.
    # Cells with <= max_cell nodes are padded into a single [B, max_cell] batch.
    max_cell = 64
    small_mask = multi_counts <= max_cell
    n_small = small_mask.sum().item()

    total = torch.tensor(0.0, device=device)
    count = 0

    if n_small > 0:
        # Gather small cells into a padded batch
        small_indices = torch.where(small_mask)[0]
        B = small_indices.shape[0]
        batch_nodes = torch.zeros(B, max_cell, dtype=torch.long, device=device)
        batch_valid = torch.zeros(B, max_cell, dtype=torch.bool, device=device)
        small_idx_cpu = small_indices.cpu().tolist()
        for bi, si in enumerate(small_idx_cpu):
            s, e = starts_cpu[si], ends_cpu[si]
            m = e - s
            batch_nodes[bi, :m] = sort_idx[s:e]
            batch_valid[bi, :m] = True

        # Vectorized all-pairs overlap for the batch
        bp = pos[batch_nodes]  # [B, M, 2]
        bsz = node_sizes[batch_nodes]  # [B, M, 2]
        dx_abs = torch.abs(bp[:, :, 0].unsqueeze(2) - bp[:, :, 0].unsqueeze(1))  # [B, M, M]
        dy_abs = torch.abs(bp[:, :, 1].unsqueeze(2) - bp[:, :, 1].unsqueeze(1))
        min_dx = (bsz[:, :, 0].unsqueeze(2) + bsz[:, :, 0].unsqueeze(1)) / 2 + padding
        min_dy = (bsz[:, :, 1].unsqueeze(2) + bsz[:, :, 1].unsqueeze(1)) / 2 + padding
        overlap = F.relu(min_dx - dx_abs) * F.relu(min_dy - dy_abs)
        # Mask: valid pairs only, exclude self and padding
        pair_valid = batch_valid.unsqueeze(2) & batch_valid.unsqueeze(1)  # [B, M, M]
        diag_mask = ~torch.eye(max_cell, dtype=torch.bool, device=device).unsqueeze(0)
        pair_valid = pair_valid & diag_mask
        masked_overlap = overlap * pair_valid.float()
        total = total + masked_overlap.sum()
        count += int(pair_valid.sum().item())

    # Process remaining large cells individually
    large_mask = ~small_mask
    if large_mask.any():
        large_indices = torch.where(large_mask)[0].cpu().tolist()
        for li in large_indices:
            s, e = starts_cpu[li], ends_cpu[li]
            cell_nodes = sort_idx[s:e]
            m = cell_nodes.shape[0]
            if m > 200:
                perm2 = torch.randperm(m, device=device)[:200]
                cell_nodes = cell_nodes[perm2]
                m = 200

            p = pos[cell_nodes]
            sz = node_sizes[cell_nodes]
            dx_abs = torch.abs(p[:, 0].unsqueeze(0) - p[:, 0].unsqueeze(1))
            dy_abs = torch.abs(p[:, 1].unsqueeze(0) - p[:, 1].unsqueeze(1))
            min_dx = (sz[:, 0].unsqueeze(0) + sz[:, 0].unsqueeze(1)) / 2 + padding
            min_dy = (sz[:, 1].unsqueeze(0) + sz[:, 1].unsqueeze(1)) / 2 + padding
            overlap = F.relu(min_dx - dx_abs) * F.relu(min_dy - dy_abs)
            mask = ~torch.eye(m, dtype=torch.bool, device=device)
            cell_overlap = overlap[mask]
            if cell_overlap.numel() > 0:
                total = total + cell_overlap.sum()
                count += cell_overlap.numel()

    if count == 0:
        return torch.tensor(0.0, device=device)
    return total / count


# ─── Crossing loss ──────────────────────────────────────────────────────────


def crossing_loss(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    alpha: float = 5.0,
    max_pairs: int = 2000,
    layer_assignments: Optional[Union[List[int], torch.Tensor]] = None,
) -> torch.Tensor:
    """Differentiable crossing proxy using adjacent-layer sigmoid relaxation."""
    num_edges = edge_index.shape[1]
    if num_edges < 2:
        return torch.tensor(0.0, device=pos.device)

    # For small edge counts, use the simpler fallback (no virtual node overhead)
    if layer_assignments is None or num_edges < 20:
        return _crossing_loss_fallback(pos, edge_index, alpha, max_pairs)

    return _crossing_loss_layered(pos, edge_index, alpha, max_pairs, layer_assignments)


def _crossing_loss_fallback(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    alpha: float,
    max_pairs: int,
) -> torch.Tensor:
    """Original crossing proxy for when layer info is unavailable."""
    num_edges = edge_index.shape[1]

    if num_edges * (num_edges - 1) // 2 > max_pairs:
        n_sample = min(max_pairs, num_edges)
        perm = torch.randperm(num_edges, device=pos.device)[:n_sample]
        ei = edge_index[:, perm]
    else:
        ei = edge_index
        n_sample = ei.shape[1]

    src_x = pos[ei[0], 0]
    tgt_x = pos[ei[1], 0]

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
    layer_assignments: Union[List[int], torch.Tensor],
) -> torch.Tensor:
    """Adjacent-layer crossing loss with virtual node decomposition (vectorized)."""
    device = pos.device
    num_edges = edge_index.shape[1]

    if isinstance(layer_assignments, torch.Tensor):
        layers_t = layer_assignments.to(device=device)
    else:
        layers_t = torch.tensor(layer_assignments, dtype=torch.long, device=device)

    src = edge_index[0]
    tgt = edge_index[1]
    src_layer = layers_t[src]
    tgt_layer = layers_t[tgt]

    needs_swap = src_layer > tgt_layer
    actual_src = torch.where(needs_swap, tgt, src)
    actual_tgt = torch.where(needs_swap, src, tgt)
    actual_src_layer = torch.where(needs_swap, tgt_layer, src_layer)
    actual_tgt_layer = torch.where(needs_swap, src_layer, tgt_layer)

    span = actual_tgt_layer - actual_src_layer
    valid = span > 0
    if not valid.any():
        return torch.tensor(0.0, device=device)

    actual_src_layer = actual_src_layer[valid]
    actual_tgt_layer = actual_tgt_layer[valid]
    actual_src_v = actual_src[valid]
    actual_tgt_v = actual_tgt[valid]
    span_v = span[valid]

    src_x = pos[actual_src_v, 0]
    tgt_x = pos[actual_tgt_v, 0]
    span_f = span_v.float()

    # Cap total segments
    max_total_segments = max(num_edges * 4, 50000)
    total_segments = span_v.sum().item()

    if total_segments > max_total_segments:
        n_edges_valid = span_v.shape[0]
        avg_span = max(span_f.mean().long().item(), 1)
        sample_n = int(min(n_edges_valid, max(max_total_segments // avg_span, 100)))
        perm = torch.randperm(n_edges_valid, device=device)[:sample_n]
        actual_src_layer = actual_src_layer[perm]
        actual_tgt_layer = actual_tgt_layer[perm]
        src_x = src_x[perm]
        tgt_x = tgt_x[perm]
        span_v = span_v[perm]
        span_f = span_v.float()

    seg_edge_idx = torch.repeat_interleave(
        torch.arange(span_v.shape[0], device=device), span_v
    )

    offsets = torch.arange(seg_edge_idx.shape[0], device=device)
    cum_spans = torch.zeros(span_v.shape[0] + 1, dtype=torch.long, device=device)
    cum_spans[1:] = span_v.cumsum(0)
    seg_k = offsets - cum_spans[seg_edge_idx]

    seg_layers = actual_src_layer[seg_edge_idx] + seg_k

    seg_frac_from = seg_k.float() / span_f[seg_edge_idx]
    seg_frac_to = (seg_k.float() + 1) / span_f[seg_edge_idx]

    seg_src_x = src_x[seg_edge_idx]
    seg_tgt_x = tgt_x[seg_edge_idx]
    seg_x_from = seg_src_x + (seg_tgt_x - seg_src_x) * seg_frac_from
    seg_x_to = seg_src_x + (seg_tgt_x - seg_src_x) * seg_frac_to

    n_segs_total = seg_layers.shape[0]
    if n_segs_total < 2:
        return torch.tensor(0.0, device=device)

    sort_idx = seg_layers.argsort()
    sorted_layers = seg_layers[sort_idx]
    sorted_x_from = seg_x_from[sort_idx]
    sorted_x_to = seg_x_to[sort_idx]

    unique_layers, counts = sorted_layers.unique_consecutive(return_counts=True)
    multi_mask = counts >= 2
    if not multi_mask.any():
        return torch.tensor(0.0, device=device)

    multi_counts = counts[multi_mask]
    offsets_arr = torch.zeros(counts.shape[0] + 1, dtype=torch.long, device=device)
    offsets_arr[1:] = counts.cumsum(0)
    multi_offsets = offsets_arr[:-1][multi_mask]

    pair_i_list = []
    pair_j_list = []

    total_possible_pairs = ((multi_counts * (multi_counts - 1)) // 2).sum().item()

    # Pre-fetch to CPU to avoid GPU sync stalls in the loop (.item() on GPU tensors)
    multi_offsets_cpu = multi_offsets.tolist()
    multi_counts_cpu = multi_counts.tolist()

    if total_possible_pairs <= max_pairs:
        for k in range(len(multi_counts_cpu)):
            off = multi_offsets_cpu[k]
            cnt = multi_counts_cpu[k]
            idx_i, idx_j = torch.triu_indices(cnt, cnt, offset=1, device=device)
            pair_i_list.append(idx_i + off)
            pair_j_list.append(idx_j + off)
    else:
        pairs_per_layer = (multi_counts * (multi_counts - 1)) // 2
        total_possible = pairs_per_layer.sum().float()
        samples_per_layer = (pairs_per_layer.float() / total_possible * max_pairs).long()
        samples_per_layer = samples_per_layer.clamp(min=1)
        samples_per_layer_cpu = samples_per_layer.tolist()

        for k in range(len(multi_counts_cpu)):
            off = multi_offsets_cpu[k]
            cnt = multi_counts_cpu[k]
            n_samp = min(samples_per_layer_cpu[k], cnt * (cnt - 1) // 2)

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
    return crossing_proxy.sum()


# ─── Cluster losses ─────────────────────────────────────────────────────────


def _resolve_cluster_members(members, device):
    """Resolve cluster members to a flat list of indices, handling nested dicts."""
    from dagua.utils import collect_cluster_leaves
    if isinstance(members, dict):
        members = collect_cluster_leaves(members)
    if isinstance(members, list) and len(members) > 0:
        return torch.tensor(members, device=device, dtype=torch.long)
    return None


def cluster_compactness_loss(
    pos: torch.Tensor,
    clusters: dict,
    device: torch.device,
) -> torch.Tensor:
    """Nodes in same cluster attract their cluster centroid."""
    total = torch.tensor(0.0, device=device)
    count = 0
    for name, members in clusters.items():
        idx = _resolve_cluster_members(members, device)
        if idx is not None and idx.shape[0] > 1:
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
    device: Optional[torch.device] = None,
    cluster_parents: Optional[Dict[str, Optional[str]]] = None,
) -> torch.Tensor:
    """Sibling cluster bounding boxes repel.

    When cluster_parents is provided, only repels clusters at the same
    hierarchy level (same parent or both root-level). Parent vs child
    should NOT repel — containment loss handles that.
    """
    if device is None:
        device = pos.device

    cluster_list = []
    for name, members in clusters.items():
        idx = _resolve_cluster_members(members, device)
        if idx is not None and idx.shape[0] > 0:
            parent = cluster_parents.get(name) if cluster_parents else None
            cluster_list.append((name, idx, parent))

    if len(cluster_list) < 2:
        return torch.tensor(0.0, device=device)

    # Build sibling pairs: only repel clusters with the same parent
    num_clusters = len(cluster_list)
    if cluster_parents:
        all_pairs = []
        for i in range(num_clusters):
            for j in range(i + 1, num_clusters):
                if cluster_list[i][2] == cluster_list[j][2]:  # same parent (or both None)
                    all_pairs.append((i, j))
    elif num_clusters > 50:
        max_sample = int(min(50, num_clusters * (num_clusters - 1) // 2))
        sampled: set[tuple[int, int]] = set()
        attempts = 0
        while len(sampled) < max_sample and attempts < max_sample * 10:
            i = random.randint(0, num_clusters - 1)
            j = random.randint(0, num_clusters - 1)
            if i != j:
                sampled.add((min(i, j), max(i, j)))
            attempts += 1
        all_pairs = list(sampled)
    else:
        all_pairs = [
            (i, j)
            for i in range(num_clusters)
            for j in range(i + 1, num_clusters)
        ]

    if not all_pairs:
        return torch.tensor(0.0, device=device)

    total = torch.tensor(0.0, device=device)
    for i, j in all_pairs:
        idx_i = cluster_list[i][1]
        idx_j = cluster_list[j][1]

        bbox_i_min = pos[idx_i].min(dim=0).values - node_sizes[idx_i].max(dim=0).values / 2 - padding
        bbox_i_max = pos[idx_i].max(dim=0).values + node_sizes[idx_i].max(dim=0).values / 2 + padding
        bbox_j_min = pos[idx_j].min(dim=0).values - node_sizes[idx_j].max(dim=0).values / 2 - padding
        bbox_j_max = pos[idx_j].max(dim=0).values + node_sizes[idx_j].max(dim=0).values / 2 + padding

        overlap_x = F.relu(torch.min(bbox_i_max[0], bbox_j_max[0]) - torch.max(bbox_i_min[0], bbox_j_min[0]))
        overlap_y = F.relu(torch.min(bbox_i_max[1], bbox_j_max[1]) - torch.max(bbox_i_min[1], bbox_j_min[1]))
        total = total + overlap_x * overlap_y

    return total


def cluster_containment_loss(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    clusters: dict,
    cluster_parents: Dict[str, Optional[str]],
    padding: float = 18.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Child cluster bboxes must stay inside parent cluster bboxes.

    For each (child, parent) pair in cluster_parents:
    - Compute child bbox from its leaf members
    - Compute parent bbox from its leaf members
    - Penalize child bbox edges extending outside parent bbox:
      ReLU(parent_min - child_min)² + ReLU(child_max - parent_max)² per axis
    """
    if device is None:
        device = pos.device

    total = torch.tensor(0.0, device=device)
    count = 0

    for child_name, parent_name in cluster_parents.items():
        if parent_name is None:
            continue
        if child_name not in clusters or parent_name not in clusters:
            continue

        child_idx = _resolve_cluster_members(clusters[child_name], device)
        parent_idx = _resolve_cluster_members(clusters[parent_name], device)
        if child_idx is None or parent_idx is None:
            continue

        # Child bbox
        child_min = pos[child_idx].min(dim=0).values - node_sizes[child_idx].max(dim=0).values / 2
        child_max = pos[child_idx].max(dim=0).values + node_sizes[child_idx].max(dim=0).values / 2

        # Parent bbox (with padding — parent should be larger)
        parent_min = pos[parent_idx].min(dim=0).values - node_sizes[parent_idx].max(dim=0).values / 2 - padding
        parent_max = pos[parent_idx].max(dim=0).values + node_sizes[parent_idx].max(dim=0).values / 2 + padding

        # Penalize child extending outside parent
        violation = (
            F.relu(parent_min - child_min) ** 2 +
            F.relu(child_max - parent_max) ** 2
        ).sum()
        total = total + violation
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=device)
    return total / count


# ─── Spacing consistency ──────────────────────────────────────────────────────


# ─── Flex constraints (pins, alignment, flex spacing) ────────────────────────


def position_pin_loss(
    pos: torch.Tensor,
    pin_indices: torch.Tensor,
    pin_targets: torch.Tensor,
    pin_weights: torch.Tensor,
    pin_mask: torch.Tensor,
) -> torch.Tensor:
    """Soft penalty pulling pinned nodes toward targets.

    Args:
        pos: [N, 2] node positions.
        pin_indices: [P] indices of pinned nodes.
        pin_targets: [P, 2] target (x, y) for each pin.
        pin_weights: [P, 2] weight for each pin axis.
        pin_mask: [P, 2] bool — True where the axis is constrained.

    Hard pins (weight=inf) are handled via post-step projection, not here.
    This function only computes loss for finite-weight pins.
    """
    if pin_indices.numel() == 0:
        return torch.tensor(0.0, device=pos.device)

    pinned_pos = pos[pin_indices]  # [P, 2]
    diff = (pinned_pos - pin_targets) ** 2  # [P, 2]
    weighted = diff * pin_weights * pin_mask.float()  # [P, 2]
    return weighted.sum() / max(pin_mask.sum().item(), 1.0)


def alignment_loss(
    pos: torch.Tensor,
    align_groups: List[Tuple[torch.Tensor, float, int]],
) -> torch.Tensor:
    """Penalize positional variance within alignment groups.

    Args:
        pos: [N, 2] node positions.
        align_groups: list of (indices_tensor, weight, axis) where axis is 0=x, 1=y.
    """
    if not align_groups:
        return torch.tensor(0.0, device=pos.device)

    total = torch.tensor(0.0, device=pos.device)
    count = 0
    for indices, weight, axis in align_groups:
        if indices.numel() < 2:
            continue
        coords = pos[indices, axis]  # [G]
        mean = coords.mean()
        total = total + weight * ((coords - mean) ** 2).mean()
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=pos.device)
    return total / count


def flex_spacing_loss(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    layer_index: Optional[LayerIndex],
    target_sep: float,
    weight: float,
) -> torch.Tensor:
    """Penalize deviation from flex spacing targets.

    Similar to spacing_consistency_loss but specifically weighted
    for the flex system. Uses the flex weight to scale the loss.
    """
    if layer_index is None or weight <= 0:
        return torch.tensor(0.0, device=pos.device)

    # Delegate to spacing_consistency_loss and re-weight
    base_loss = spacing_consistency_loss(pos, node_sizes, layer_index, target_gap=target_sep)
    return weight * base_loss


def project_hard_pins(
    pos: torch.Tensor,
    pin_indices: torch.Tensor,
    pin_targets: torch.Tensor,
    pin_mask: torch.Tensor,
) -> None:
    """Project hard-pinned nodes to their target positions (in-place).

    Called after optimizer.step() to enforce weight=inf pins.

    Args:
        pos: [N, 2] node positions (modified in-place).
        pin_indices: [P] indices of hard-pinned nodes.
        pin_targets: [P, 2] target positions.
        pin_mask: [P, 2] bool — True where axis is hard-pinned.
    """
    if pin_indices.numel() == 0:
        return

    with torch.no_grad():
        current = pos[pin_indices]  # [P, 2]
        projected = torch.where(pin_mask, pin_targets, current)
        pos.data[pin_indices] = projected


def spacing_consistency_loss(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    layer_index: Optional[LayerIndex],
    target_gap: float = 25.0,
) -> torch.Tensor:
    """Penalize deviation from target horizontal spacing within layers.

    For each layer, sort nodes by x, measure consecutive gaps, and penalize
    variance. This produces the even "visual rhythm" the style guide describes.

    Uses vectorized approach: composite sort key → consecutive pairs.
    """
    if layer_index is None:
        return torch.tensor(0.0, device=pos.device)

    device = pos.device
    N = pos.shape[0]
    if N < 2:
        return torch.tensor(0.0, device=device)

    # Skip for very large graphs — global argsort on N nodes creates ~50 GB
    # of intermediates at 1B nodes. Repulsion/overlap already handle spacing
    # at this scale via RVS sampling.
    if N > 100_000_000:
        return torch.tensor(0.0, device=device)

    layers = layer_index.node_to_layer
    offsets = layer_index.layer_offsets
    num_layers = layer_index.num_layers

    # Sort all nodes by (layer, x_position) — one global sort, O(N log N)
    sort_key = layers.float() * 1e8 + pos[:, 0].detach()
    sorted_idx = sort_key.argsort()

    sorted_layers = layers[sorted_idx]
    sorted_x = pos[sorted_idx, 0]
    sorted_w = node_sizes[sorted_idx, 0]

    # Consecutive pairs within same layer
    same_layer = sorted_layers[:-1] == sorted_layers[1:]
    if not same_layer.any():
        return torch.tensor(0.0, device=device)

    # Gap = center-to-center distance minus half-widths
    dx = sorted_x[1:] - sorted_x[:-1]
    half_w = (sorted_w[:-1] + sorted_w[1:]) / 2.0
    gap = dx - half_w  # actual gap between edges

    # Only consider same-layer pairs
    gap_in_layer = gap[same_layer]

    if gap_in_layer.numel() == 0:
        return torch.tensor(0.0, device=device)

    # Penalize deviation from target gap (squared)
    deviation = gap_in_layer - target_gap
    return (deviation ** 2).mean()


# ─── Fan-out distribution loss ────────────────────────────────────────────────


def fanout_distribution_loss(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    degree_threshold: int = 5,
) -> torch.Tensor:
    """Penalize uneven angular distribution of children for high-degree nodes.

    For hub nodes (out_degree >= degree_threshold), computes the angles from
    hub to each child, sorts them, and penalizes variance in the angular gaps.
    This prevents the optimizer from collapsing fan-out children into a tight cluster.

    O(E) + O(K log K) per hub.
    """
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=pos.device)

    device = pos.device
    N = pos.shape[0]
    src, tgt = edge_index[0], edge_index[1]

    # Compute out-degree per node
    out_degree = torch.zeros(N, dtype=torch.long, device=device)
    out_degree.scatter_add_(0, src, torch.ones(src.shape[0], dtype=out_degree.dtype, device=device))

    # Find hub nodes
    hub_mask = out_degree >= degree_threshold
    hub_nodes = torch.where(hub_mask)[0]

    if hub_nodes.numel() == 0:
        return torch.tensor(0.0, device=pos.device)

    edge_order = src.argsort()
    sorted_src = src[edge_order]
    sorted_tgt = tgt[edge_order]
    hub_starts = torch.searchsorted(sorted_src, hub_nodes)
    hub_degrees = out_degree[hub_nodes]

    total_loss = torch.tensor(0.0, device=device)
    count = 0

    for hub_idx, hub in enumerate(hub_nodes.tolist()):
        start = int(hub_starts[hub_idx].item())
        k = int(hub_degrees[hub_idx].item())
        if k < 2:
            continue
        children = sorted_tgt[start : start + k]

        # Compute angles from hub to each child
        dx = pos[children, 0] - pos[hub, 0]
        dy = pos[children, 1] - pos[hub, 1]
        angles = torch.atan2(dy, dx)  # [-pi, pi]

        # Sort angles and compute gaps
        sorted_angles, _ = angles.sort()
        gaps = sorted_angles[1:] - sorted_angles[:-1]
        # Wrap-around gap
        wrap_gap = (2 * 3.141592653589793) - (sorted_angles[-1] - sorted_angles[0])
        all_gaps = torch.cat([gaps, wrap_gap.unsqueeze(0)])

        # Ideal gap = 2*pi / k
        ideal_gap = (2 * 3.141592653589793) / k
        # Penalize variance from ideal
        total_loss = total_loss + ((all_gaps - ideal_gap) ** 2).mean()
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=device)
    return total_loss / count


# ─── Back-edge compactness loss ───────────────────────────────────────────────


def back_edge_compactness_loss(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Penalize horizontal distance in back-edge pairs (target above source).

    Back edges (where target y < source y) should route compactly. This loss
    penalizes the squared horizontal distance between back-edge endpoints,
    encouraging tighter back-edge routing.

    O(E), trivially vectorized.
    """
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=pos.device)

    src, tgt = edge_index[0], edge_index[1]
    src_y = pos[src, 1]
    tgt_y = pos[tgt, 1]

    # Back edges: target is above source (lower y = higher on screen)
    back_mask = tgt_y < src_y

    if not back_mask.any():
        return torch.tensor(0.0, device=pos.device)

    # Horizontal distance for back edges
    dx = pos[src[back_mask], 0] - pos[tgt[back_mask], 0]
    return (dx ** 2).mean()
