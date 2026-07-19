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
import warnings
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import torch
import torch.nn.functional as F

from dagua.layout.layers import LayerIndex

_DEFAULT_W_CLUSTER_CONTAIN = 2.0


def warn_legacy_cluster_loss_config(
    cluster_aware: bool,
    w_cluster_containment: float,
) -> None:
    """Warn when legacy cluster loss knobs are set under cluster-aware layout.

    Parameters
    ----------
    cluster_aware : bool
        Whether recursive cluster-aware placement is enabled.
    w_cluster_containment : float
        Configured legacy containment loss weight.

    Returns
    -------
    None
        Emits a ``DeprecationWarning`` only when a legacy knob is non-default.
    """
    if cluster_aware and w_cluster_containment != _DEFAULT_W_CLUSTER_CONTAIN:
        warnings.warn(
            "w_cluster_contain is ignored when cluster_aware=True; set "
            "cluster_aware=False to use legacy cluster containment loss.",
            DeprecationWarning,
            stacklevel=2,
        )


# ─── Edge-based losses (O(E), trivially parallelizable) ─────────────────────


class EdgeBatchLike(Protocol):
    """Protocol for pre-computed edge batch tensors."""

    src: torch.Tensor
    tgt: torch.Tensor
    dx: torch.Tensor
    dy: torch.Tensor
    dist_sq: torch.Tensor


class SampledNodeLike(Protocol):
    """Protocol for shared sampled-node state."""

    active_idx: torch.Tensor
    sampled: torch.Tensor


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
    edge_ctx: Optional[EdgeBatchLike] = None,
) -> torch.Tensor:
    """Penalize edges whose targets drift above their sources.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge indices with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.
    rank_sep : float, default=50.0
        Desired vertical separation between successive layers.
    edge_ctx : EdgeBatchLike | None, default=None
        Optional shared edge batch context with self-loops already removed.

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    if edge_ctx is not None:
        src, tgt = edge_ctx.src, edge_ctx.tgt
    elif edge_index.numel() == 0:
        return torch.tensor(0.0, device=pos.device)
    else:
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
    edge_ctx: Optional[EdgeBatchLike] = None,
) -> torch.Tensor:
    """Connected nodes pull together. x-bias encourages vertical edges.

    AMD insight: cap attraction at 1/3 distance to prevent overshoot.
    """
    if edge_ctx is not None:
        dx = edge_ctx.dx
        dy = edge_ctx.dy
        dist_sq = edge_ctx.dist_sq
    elif edge_index.numel() == 0:
        return torch.tensor(0.0, device=pos.device)
    else:
        src, tgt = _non_self_edges(edge_index)
        if src.numel() == 0:
            return torch.tensor(0.0, device=pos.device)
        dx = pos[src, 0] - pos[tgt, 0]
        dy = pos[src, 1] - pos[tgt, 1]
        dist_sq = dx.square() + dy.square()
    if dist_sq.numel() == 0:
        return torch.tensor(0.0, device=pos.device)

    # Cap: attraction force proportional to dist_sq, but capped at 1/3 of distance
    # This prevents nodes from overshooting past their targets
    cap = torch.ones_like(dist_sq)
    near_mask = dist_sq < 9.0
    if near_mask.any():
        dist = dist_sq[near_mask].sqrt()
        max_force = dist / 3.0
        force = dist.clamp(max=1.0)
        cap[near_mask] = torch.where(
            force > max_force,
            max_force / (force + 1e-8),
            torch.ones_like(force),
        )

    return x_bias * (dx.square() * cap).mean() + (dy.square() * cap).mean()


def edge_straightness_loss(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    edge_ctx: Optional[EdgeBatchLike] = None,
) -> torch.Tensor:
    """Penalize horizontal displacement between connected nodes.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge indices with shape ``[2, E]``.
    edge_ctx : EdgeBatchLike | None, default=None
        Optional shared edge batch context with pre-computed horizontal deltas.

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    if edge_ctx is not None:
        dx = edge_ctx.dx
    elif edge_index.numel() == 0:
        return torch.tensor(0.0, device=pos.device)
    else:
        src, tgt = _non_self_edges(edge_index)
        if src.numel() == 0:
            return torch.tensor(0.0, device=pos.device)
        dx = pos[src, 0] - pos[tgt, 0]
    if dx.numel() == 0:
        return torch.tensor(0.0, device=pos.device)
    return dx.square().mean()


def edge_length_variance_loss(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    edge_ctx: Optional[EdgeBatchLike] = None,
) -> torch.Tensor:
    """Penalize relative (scale-invariant) variation in edge lengths.

    Sprint 11: the legacy formulation returned raw ``lengths.var()``,
    whose magnitude scales with (mean_length)^2. On layouts with
    typical edge lengths 50-200, the variance was in the thousands
    and the config default weight 0.7 made this loss's gradient
    either dominate or be swamped depending on graph scale -- and
    Dagua wins 0/17 holdout graphs on ``edge_length_cv``.

    New formulation: coefficient-of-variation squared =
    ``var(lengths) / mean(lengths)^2``. Scale-invariant, bounded
    roughly in [0, ~1], directly targets the metric we're evaluated on.
    Combined with a proportionally-larger default weight (tuned in
    :mod:`dagua.config`), this makes edge uniformity an active
    constraint during gradient descent instead of background noise.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge indices with shape ``[2, E]``.
    edge_ctx : EdgeBatchLike | None, default=None
        Optional shared edge batch context with pre-computed squared distances.

    Returns
    -------
    torch.Tensor
        Scalar loss value (approximately CV^2 of edge lengths).
    """
    if edge_ctx is not None:
        dist_sq = edge_ctx.dist_sq
    elif edge_index.numel() == 0:
        return torch.tensor(0.0, device=pos.device)
    else:
        src, tgt = _non_self_edges(edge_index)
        if src.numel() <= 1:
            return torch.tensor(0.0, device=pos.device)
        dist_sq = (pos[src] - pos[tgt]).square().sum(dim=1)
    if dist_sq.numel() <= 1:
        return torch.tensor(0.0, device=pos.device)
    lengths = dist_sq.add(1e-8).sqrt()
    if lengths.numel() <= 1:
        return torch.tensor(0.0, device=pos.device)
    mean_len = lengths.mean().clamp(min=1e-6)
    return lengths.var() / (mean_len * mean_len)


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
    sampled_ctx: Optional[SampledNodeLike] = None,
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

    if sampled_ctx is not None:
        return _repulsion_rvs_from_context(pos, sampled_ctx, rvs_nn_k, node_sizes)

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
    layers_long = layers if layers.dtype == torch.long else layers.to(dtype=torch.long)
    offsets = layer_index.layer_offsets  # [L+1]
    sorted_nodes = layer_index.sorted_nodes  # [N]
    num_layers = layer_index.num_layers

    # For each node, compute the sampling range: nodes in [layer-1, layer+1]
    adj_layer_lo = (layers_long - 1).clamp(min=0)  # [N]
    adj_layer_hi = (layers_long + 2).clamp(max=num_layers)  # [N]

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
    dist_sq = (diff**2).sum(dim=2) + 1e-4  # [N, K]

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
    n_active = min(max(int(N**0.75), min(N, 256)), 1_000_000)
    n_random = max(int(N**0.25), 4)
    K_nn = min(nn_k, N - 1)
    layers = layer_index.node_to_layer
    layers_long = layers if layers.dtype == torch.long else layers.to(dtype=torch.long)
    offsets = layer_index.layer_offsets
    sorted_nodes = layer_index.sorted_nodes
    num_layers = layer_index.num_layers

    # Select active nodes uniformly at random (randint avoids [N] allocation)
    active_idx = torch.randint(0, N, (n_active,), device=device)  # [A]
    A = active_idx.shape[0]

    # For each active node, compute its adjacent-layer range
    active_layers = layers_long[active_idx]  # [A]
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
        nn_indices = (same_start.unsqueeze(1) + (rand_nn * same_range.unsqueeze(1)).long()).clamp(
            min=0, max=N - 1
        )
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
    dist_sq = (diff**2).sum(dim=2) + 1e-4  # [A, K]

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


def _repulsion_rvs_from_context(
    pos: torch.Tensor,
    sampled_ctx: SampledNodeLike,
    nn_k: int,
    node_sizes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Evaluate RVS repulsion from a shared sampled-node context."""
    device = pos.device
    N = pos.shape[0]
    active_idx = sampled_ctx.active_idx
    sampled = sampled_ctx.sampled
    if active_idx.numel() == 0 or sampled.numel() == 0:
        return torch.tensor(0.0, device=device)

    k_nn = min(nn_k, N - 1)
    n_random = max(int(N**0.25), 4)
    k_same = min(max(64, k_nn), sampled.shape[1])
    same_sampled = sampled[:, :k_same]
    same_for_repel = same_sampled[:, : min(k_nn, same_sampled.shape[1])]
    random_start = min(k_same, sampled.shape[1])
    random_end = min(random_start + n_random, sampled.shape[1])
    random_sampled = sampled[:, random_start:random_end]

    if same_for_repel.shape[1] > 0 and random_sampled.shape[1] > 0:
        all_sampled = torch.cat([same_for_repel, random_sampled], dim=1)
    elif same_for_repel.shape[1] > 0:
        all_sampled = same_for_repel
    else:
        all_sampled = random_sampled
    if all_sampled.shape[1] == 0:
        return torch.tensor(0.0, device=device)

    self_idx = active_idx.unsqueeze(1)
    not_self = all_sampled != self_idx

    active_pos = pos[active_idx]
    sample_pos = pos[all_sampled]
    diff = active_pos.unsqueeze(1) - sample_pos
    dist_sq = diff.square().sum(dim=2) + 1e-4

    if node_sizes is not None:
        src_w = node_sizes[active_idx, 0].unsqueeze(1).expand(-1, all_sampled.shape[1])
        src_h = node_sizes[active_idx, 1].unsqueeze(1).expand(-1, all_sampled.shape[1])
        tgt_w = node_sizes[all_sampled, 0]
        tgt_h = node_sizes[all_sampled, 1]
        combined_size = (src_w + tgt_w) * (src_h + tgt_h)
        mean_size = combined_size.mean()
        repulsion = (combined_size / (mean_size + 1e-8)) / dist_sq
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
    sampled_ctx: Optional[SampledNodeLike] = None,
    debug_callback: Optional[Callable[[str], None]] = None,
) -> torch.Tensor:
    """Return the overlap-avoidance penalty for the active layout state.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.
    padding : float, default=2.0
        Extra separation margin applied around every node.
    layer_index : LayerIndex | None, default=None
        Optional per-layer index used for layer-local overlap sampling.
    rvs_threshold : int, default=100000
        Node-count threshold where the large-graph active-subset path replaces
        the exact or same-layer scatter paths.
    sampled_ctx : SampledNodeLike | None, default=None
        Optional shared sampled-node context. When present with at least one
        active row, this path takes precedence over the size-based overlap
        branches so subset-GPU execution can reuse the same sampled rows as
        repulsion.
    debug_callback : Callable[[str], None] | None, default=None
        Optional observer notified when the shared sampled-node path is chosen.

    Returns
    -------
    torch.Tensor
        Scalar overlap penalty.
    """
    n = pos.shape[0]
    if n <= 1:
        return torch.tensor(0.0, device=pos.device)

    has_active_sampled_ctx = sampled_ctx is not None and sampled_ctx.active_idx.numel() > 0
    if has_active_sampled_ctx:
        assert sampled_ctx is not None
        if debug_callback is not None:
            debug_callback("Overlap path: sampled_ctx")
        return _overlap_active_subset_from_context(pos, node_sizes, padding, sampled_ctx)

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
    layers_long = layers if layers.dtype == torch.long else layers.to(dtype=torch.long)
    offsets = layer_index.layer_offsets  # [L+1]
    sorted_nodes = layer_index.sorted_nodes  # [N]

    # For each node, sample from same layer only (cross-layer separated by rank_sep)
    layer_start = offsets[layers_long]  # [N]
    layer_end = offsets[layers_long + 1]  # [N]
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
    n_active = min(max(int(N**0.75), min(N, 256)), 1_000_000)
    K = min(64, N - 1)
    if K <= 0:
        return torch.tensor(0.0, device=device)

    active_idx = torch.randint(0, N, (n_active,), device=device)
    A = active_idx.shape[0]

    layers = layer_index.node_to_layer
    layers_long = layers if layers.dtype == torch.long else layers.to(dtype=torch.long)
    offsets = layer_index.layer_offsets
    sorted_nodes = layer_index.sorted_nodes

    # Sample K neighbors from same layer for each active node
    active_layers = layers_long[active_idx]
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


def _overlap_active_subset_from_context(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    padding: float,
    sampled_ctx: SampledNodeLike,
) -> torch.Tensor:
    """Evaluate sampled overlap avoidance from a shared sampled-node context."""
    device = pos.device
    active_idx = sampled_ctx.active_idx
    sampled = sampled_ctx.sampled[:, : min(64, sampled_ctx.sampled.shape[1])]
    if active_idx.numel() == 0 or sampled.numel() == 0:
        return torch.tensor(0.0, device=device)

    self_idx = active_idx.unsqueeze(1)
    not_self = sampled != self_idx

    half_w_src = node_sizes[active_idx, 0].unsqueeze(1).expand(-1, sampled.shape[1]) / 2
    half_h_src = node_sizes[active_idx, 1].unsqueeze(1).expand(-1, sampled.shape[1]) / 2
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

    seg_edge_idx = torch.repeat_interleave(torch.arange(span_v.shape[0], device=device), span_v)

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

    total_possible_pairs = ((multi_counts * (multi_counts - 1)) // 2).sum().item()
    pairs_per_layer = (multi_counts * (multi_counts - 1)) // 2

    if total_possible_pairs <= max_pairs:
        total_pairs = int(pairs_per_layer.sum().item())
        if total_pairs == 0:
            return torch.tensor(0.0, device=device)

        rows_per_group = (multi_counts - 1).clamp(min=0)
        group_for_row = torch.repeat_interleave(
            torch.arange(multi_counts.shape[0], device=device), rows_per_group
        )
        count_for_row = multi_counts[group_for_row]
        row_in_group = torch.arange(group_for_row.shape[0], device=device)
        group_row_offsets = torch.zeros(multi_counts.shape[0] + 1, dtype=torch.long, device=device)
        group_row_offsets[1:] = rows_per_group.cumsum(0)
        row_in_group = row_in_group - group_row_offsets[group_for_row]

        pairs_this_row = count_for_row - 1 - row_in_group
        pair_row_idx = torch.repeat_interleave(
            torch.arange(group_for_row.shape[0], device=device), pairs_this_row
        )
        pair_row_in_group = row_in_group[pair_row_idx]
        pair_group = group_for_row[pair_row_idx]

        pair_seq = torch.arange(total_pairs, device=device)
        row_pair_offsets = torch.zeros(group_for_row.shape[0] + 1, dtype=torch.long, device=device)
        row_pair_offsets[1:] = pairs_this_row.cumsum(0)
        col_in_row = pair_seq - row_pair_offsets[pair_row_idx]

        local_i = pair_row_in_group
        local_j = pair_row_in_group + 1 + col_in_row
        all_i = local_i + multi_offsets[pair_group]
        all_j = local_j + multi_offsets[pair_group]
    else:
        total_possible = pairs_per_layer.sum().float()
        samples_per_layer = (
            (pairs_per_layer.float() / total_possible * max_pairs).long().clamp(min=1)
        )
        total_samples = int(samples_per_layer.sum().item())
        if total_samples == 0:
            return torch.tensor(0.0, device=device)

        group_id = torch.repeat_interleave(
            torch.arange(multi_counts.shape[0], device=device), samples_per_layer
        )
        c_expanded = multi_counts[group_id]
        off_expanded = multi_offsets[group_id]

        rand_i = (torch.rand(total_samples, device=device) * c_expanded.float()).long()
        rand_j = (torch.rand(total_samples, device=device) * c_expanded.float()).long()
        valid = rand_i != rand_j
        all_i = rand_i[valid] + off_expanded[valid]
        all_j = rand_j[valid] + off_expanded[valid]

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


class _ClusterCache:
    """Per-pipeline-call cache of vectorized cluster index tensors.

    All cluster losses (compactness, separation, containment) iterate over the
    SAME cluster dict every step. Building flat (cluster_id, node_idx) tensors
    and sibling/containment pair tensors ONCE here turns each step from O(C +
    P) Python work into a handful of scatter ops. node_sizes is constant across
    steps so per-cluster max sizes are precomputed under no_grad.
    """

    __slots__ = (
        "device",
        "num_clusters",
        "cluster_idx_flat",
        "node_idx_flat",
        "membership_count",
        "compactness_active_mask",
        "compactness_active_count",
        "sibling_left",
        "sibling_right",
        "size_max_per_cluster",
        "containment_child_idx",
        "containment_parent_idx",
        "containment_count",
    )

    def __init__(
        self,
        clusters: dict,
        cluster_parents: Optional[Dict[str, Optional[str]]],
        node_sizes: Optional[torch.Tensor],
        device: torch.device,
    ) -> None:
        self.device = device

        names: List[str] = []
        flat_cluster_ids: List[int] = []
        flat_node_idx: List[int] = []
        membership_count_py: List[int] = []
        for name, members in clusters.items():
            idx = _resolve_cluster_members(members, device)
            if idx is None:
                continue
            cid = len(names)
            names.append(name)
            n_members = int(idx.shape[0])
            membership_count_py.append(n_members)
            flat_cluster_ids.extend([cid] * n_members)
            flat_node_idx.extend(idx.tolist())

        self.num_clusters = len(names)
        if self.num_clusters == 0:
            self.cluster_idx_flat = torch.empty(0, dtype=torch.long, device=device)
            self.node_idx_flat = torch.empty(0, dtype=torch.long, device=device)
            self.membership_count = torch.empty(0, dtype=torch.float32, device=device)
            self.compactness_active_mask = torch.empty(0, dtype=torch.bool, device=device)
            self.compactness_active_count = 0
            self.sibling_left = torch.empty(0, dtype=torch.long, device=device)
            self.sibling_right = torch.empty(0, dtype=torch.long, device=device)
            self.size_max_per_cluster = torch.empty((0, 2), dtype=torch.float32, device=device)
            self.containment_child_idx = torch.empty(0, dtype=torch.long, device=device)
            self.containment_parent_idx = torch.empty(0, dtype=torch.long, device=device)
            self.containment_count = 0
            return

        self.cluster_idx_flat = torch.tensor(flat_cluster_ids, dtype=torch.long, device=device)
        self.node_idx_flat = torch.tensor(flat_node_idx, dtype=torch.long, device=device)
        membership_count = torch.tensor(membership_count_py, dtype=torch.float32, device=device)
        self.membership_count = membership_count
        active = membership_count > 1
        self.compactness_active_mask = active
        self.compactness_active_count = int(active.sum().item())

        # Sibling pairs: same parent (or both root-level when parents provided),
        # or all-pairs when no hierarchy. Sample if many clusters and no
        # hierarchy is given (matches legacy behaviour).
        sibling_pairs: List[Tuple[int, int]] = []
        if cluster_parents:
            parents_per_cluster = [cluster_parents.get(n) for n in names]
            for i in range(self.num_clusters):
                for j in range(i + 1, self.num_clusters):
                    if parents_per_cluster[i] == parents_per_cluster[j]:
                        sibling_pairs.append((i, j))
        elif self.num_clusters > 50:
            max_sample = int(min(50, self.num_clusters * (self.num_clusters - 1) // 2))
            sampled: set[tuple[int, int]] = set()
            attempts = 0
            while len(sampled) < max_sample and attempts < max_sample * 10:
                i = random.randint(0, self.num_clusters - 1)
                j = random.randint(0, self.num_clusters - 1)
                if i != j:
                    sampled.add((min(i, j), max(i, j)))
                attempts += 1
            sibling_pairs = list(sampled)
        else:
            sibling_pairs = [
                (i, j) for i in range(self.num_clusters) for j in range(i + 1, self.num_clusters)
            ]

        if sibling_pairs:
            pair_tensor = torch.tensor(sibling_pairs, dtype=torch.long, device=device)
            self.sibling_left = pair_tensor[:, 0].contiguous()
            self.sibling_right = pair_tensor[:, 1].contiguous()
        else:
            self.sibling_left = torch.empty(0, dtype=torch.long, device=device)
            self.sibling_right = torch.empty(0, dtype=torch.long, device=device)

        # Per-cluster max node size (constant across steps -- no autograd needed).
        if node_sizes is not None:
            with torch.no_grad():
                node_sizes_d = node_sizes.to(device=device, dtype=torch.float32)
                size_per_row = node_sizes_d[self.node_idx_flat]
                size_max = torch.full(
                    (self.num_clusters, 2), float("-inf"), device=device, dtype=torch.float32
                )
                size_max.scatter_reduce_(
                    0,
                    self.cluster_idx_flat.unsqueeze(1).expand_as(size_per_row),
                    size_per_row,
                    reduce="amax",
                    include_self=True,
                )
                # Singletons should still get a real size (scatter_reduce hit them once).
                self.size_max_per_cluster = size_max
        else:
            self.size_max_per_cluster = torch.zeros(
                (self.num_clusters, 2), dtype=torch.float32, device=device
            )

        # Containment pairs: (child_id, parent_id) for every parent-having
        # cluster present in clusters AND parents.
        if cluster_parents:
            name_to_id = {n: i for i, n in enumerate(names)}
            child_ids: List[int] = []
            parent_ids: List[int] = []
            for child_name, parent_name in cluster_parents.items():
                if parent_name is None:
                    continue
                if child_name not in name_to_id or parent_name not in name_to_id:
                    continue
                child_ids.append(name_to_id[child_name])
                parent_ids.append(name_to_id[parent_name])
            if child_ids:
                self.containment_child_idx = torch.tensor(
                    child_ids, dtype=torch.long, device=device
                )
                self.containment_parent_idx = torch.tensor(
                    parent_ids, dtype=torch.long, device=device
                )
            else:
                self.containment_child_idx = torch.empty(0, dtype=torch.long, device=device)
                self.containment_parent_idx = torch.empty(0, dtype=torch.long, device=device)
            self.containment_count = len(child_ids)
        else:
            self.containment_child_idx = torch.empty(0, dtype=torch.long, device=device)
            self.containment_parent_idx = torch.empty(0, dtype=torch.long, device=device)
            self.containment_count = 0


def _segment_arg_first_match(
    values_per_row: torch.Tensor,
    cluster_idx_flat: torch.Tensor,
    num_clusters: int,
    ref_per_row: torch.Tensor,
) -> torch.Tensor:
    """Return per-cluster row id of the FIRST row matching the cluster's reference.

    Used as segment-argmin / segment-argmax with first-occurrence tie-break,
    matching ``torch.min(dim=0).indices`` / ``torch.max(dim=0).indices``
    semantics exactly. Caller must compute ``ref_per_row`` as the per-cluster
    extremum (under no_grad).
    """
    R = values_per_row.shape[0]
    if R == 0:
        return torch.empty(num_clusters, dtype=torch.long, device=values_per_row.device)
    match = values_per_row == ref_per_row
    row_ids = torch.arange(R, device=values_per_row.device, dtype=torch.long)
    sentinel = torch.full_like(row_ids, R)
    candidate = torch.where(match, row_ids, sentinel)
    first = torch.full((num_clusters,), R, dtype=torch.long, device=values_per_row.device)
    first.scatter_reduce_(0, cluster_idx_flat, candidate, reduce="amin", include_self=True)
    return first


def _bbox_min_max_per_cluster(
    pos: torch.Tensor,
    cache: _ClusterCache,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (bbox_min, bbox_max) tensors of shape [C, 2].

    Gradient routing matches the legacy ``pos[idx].min(dim=0).values`` /
    ``.max(dim=0).values`` bit-for-bit, including the tie-breaking rule
    (lowest row id wins). Implementation:

    1. Compute per-cluster min/max VALUES under ``no_grad`` via
       ``scatter_reduce(amin/amax)`` -- reference floats only, no autograd.
    2. For each axis, find the first row in each cluster matching that
       reference (segment-argmin / -argmax with first-occurrence tie-break).
    3. Re-index the LIVE ``pos`` tensor at those rows. Backprop now flows
       through indexed gather to exactly ONE argmin / argmax row per
       cluster per axis -- identical to legacy ``min(dim=0)`` semantics
       even when multiple cluster members tie at the boundary.
    """
    pos_per_row = pos[cache.node_idx_flat]  # [R, 2]
    cluster_idx_2d = cache.cluster_idx_flat.unsqueeze(1).expand_as(pos_per_row)

    with torch.no_grad():
        ref_min = torch.full(
            (cache.num_clusters, 2), float("inf"), device=pos.device, dtype=pos.dtype
        )
        ref_min.scatter_reduce_(0, cluster_idx_2d, pos_per_row, reduce="amin", include_self=True)
        ref_max = torch.full(
            (cache.num_clusters, 2), float("-inf"), device=pos.device, dtype=pos.dtype
        )
        ref_max.scatter_reduce_(0, cluster_idx_2d, pos_per_row, reduce="amax", include_self=True)

        argmin_x = _segment_arg_first_match(
            pos_per_row[:, 0],
            cache.cluster_idx_flat,
            cache.num_clusters,
            ref_min[cache.cluster_idx_flat, 0],
        )
        argmin_y = _segment_arg_first_match(
            pos_per_row[:, 1],
            cache.cluster_idx_flat,
            cache.num_clusters,
            ref_min[cache.cluster_idx_flat, 1],
        )
        argmax_x = _segment_arg_first_match(
            pos_per_row[:, 0],
            cache.cluster_idx_flat,
            cache.num_clusters,
            ref_max[cache.cluster_idx_flat, 0],
        )
        argmax_y = _segment_arg_first_match(
            pos_per_row[:, 1],
            cache.cluster_idx_flat,
            cache.num_clusters,
            ref_max[cache.cluster_idx_flat, 1],
        )

    bbox_min = torch.stack((pos_per_row[argmin_x, 0], pos_per_row[argmin_y, 1]), dim=1)
    bbox_max = torch.stack((pos_per_row[argmax_x, 0], pos_per_row[argmax_y, 1]), dim=1)
    return bbox_min, bbox_max


def cluster_compactness_loss(
    pos: torch.Tensor,
    clusters: dict,
    device: torch.device,
    cache: Optional[_ClusterCache] = None,
) -> torch.Tensor:
    """Nodes in same cluster attract their cluster centroid (vectorized)."""
    if cache is None:
        cache = _ClusterCache(clusters, None, None, device)
    if cache.compactness_active_count == 0:
        return torch.tensor(0.0, device=device)

    pos_per_row = pos[cache.node_idx_flat]  # [R, 2]
    cluster_idx_2d = cache.cluster_idx_flat.unsqueeze(1).expand_as(pos_per_row)
    centroid_sum = torch.zeros((cache.num_clusters, 2), device=pos.device, dtype=pos.dtype)
    centroid_sum.scatter_add_(0, cluster_idx_2d, pos_per_row)
    centroid = centroid_sum / cache.membership_count.unsqueeze(1).to(pos.dtype)

    centroid_per_row = centroid[cache.cluster_idx_flat]  # [R, 2]
    sq_dist = ((pos_per_row - centroid_per_row) ** 2).sum(dim=1)  # [R]
    sq_sum = torch.zeros(cache.num_clusters, device=pos.device, dtype=pos.dtype)
    sq_sum.scatter_add_(0, cache.cluster_idx_flat, sq_dist)
    per_cluster_mean = sq_sum / cache.membership_count.to(pos.dtype)

    active = cache.compactness_active_mask
    return per_cluster_mean[active].sum() / float(cache.compactness_active_count)


def cluster_separation_loss(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    clusters: dict,
    padding: float = 10.0,
    device: Optional[torch.device] = None,
    cluster_parents: Optional[Dict[str, Optional[str]]] = None,
    cache: Optional[_ClusterCache] = None,
) -> torch.Tensor:
    """Sibling cluster bounding boxes repel (vectorized).

    When cluster_parents is provided, only repels clusters at the same
    hierarchy level (same parent or both root-level). Parent vs child
    should NOT repel -- containment loss handles that.
    """
    if device is None:
        device = pos.device
    if cache is None:
        cache = _ClusterCache(clusters, cluster_parents, node_sizes, device)
    if cache.num_clusters < 2 or cache.sibling_left.numel() == 0:
        return torch.tensor(0.0, device=device)

    bbox_min, bbox_max = _bbox_min_max_per_cluster(pos, cache)
    half_size = cache.size_max_per_cluster.to(pos.dtype) / 2 + padding
    bbox_min = bbox_min - half_size
    bbox_max = bbox_max + half_size

    left = cache.sibling_left
    right = cache.sibling_right
    overlap = F.relu(
        torch.minimum(bbox_max[left], bbox_max[right])
        - torch.maximum(bbox_min[left], bbox_min[right])
    )  # [P, 2]
    return (overlap[:, 0] * overlap[:, 1]).sum()


def cluster_containment_loss(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    clusters: dict,
    cluster_parents: Dict[str, Optional[str]],
    padding: float = 18.0,
    device: Optional[torch.device] = None,
    cache: Optional[_ClusterCache] = None,
) -> torch.Tensor:
    """Child cluster bboxes must stay inside parent cluster bboxes (vectorized).

    For each (child, parent) pair in cluster_parents:
    - Compute child bbox from its leaf members
    - Compute parent bbox from its leaf members (with padding)
    - Penalize child bbox edges extending outside parent bbox.
    """
    if device is None:
        device = pos.device
    if cache is None:
        cache = _ClusterCache(clusters, cluster_parents, node_sizes, device)
    if cache.containment_count == 0:
        return torch.tensor(0.0, device=device)

    bbox_min, bbox_max = _bbox_min_max_per_cluster(pos, cache)
    half_size = cache.size_max_per_cluster.to(pos.dtype) / 2

    child = cache.containment_child_idx
    parent = cache.containment_parent_idx

    child_min = bbox_min[child] - half_size[child]
    child_max = bbox_max[child] + half_size[child]
    parent_min = bbox_min[parent] - half_size[parent] - padding
    parent_max = bbox_max[parent] + half_size[parent] + padding

    violation = F.relu(parent_min - child_min) ** 2 + F.relu(child_max - parent_max) ** 2
    return violation.sum() / float(cache.containment_count)


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
    # Smooth-L1 (Huber) around the target instead of plain squared-L2:
    # quadratic inside a unit half-beta, linear outside. This bounds the
    # per-axis gradient magnitude to ``weight`` regardless of how far the
    # node is from target, so a distant soft pin no longer blows past the
    # engine's ClipGradNorm(max_norm=100) guard and gets shredded into a
    # near-zero step. Legacy squared-L2 gave gradient = 2 * weight * |dist|,
    # which at dist=500 & weight=50 is 50000 per axis -- two orders of
    # magnitude above the clip ceiling, reducing the effective pull per
    # step to ~1/500 of the intended strength.
    delta = pinned_pos - pin_targets  # [P, 2]
    beta = 1.0
    abs_delta = delta.abs()
    quad = 0.5 * delta.square() / beta
    lin = abs_delta - 0.5 * beta
    huber = torch.where(abs_delta < beta, quad, lin)  # [P, 2]
    weighted = huber * pin_weights * pin_mask.float()  # [P, 2]
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


def _spacing_consistency_loss_layerlocal(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    layers: torch.Tensor,
    offsets: torch.Tensor,
    num_layers: int,
    target_gap: float,
    device: torch.device,
    layer_index: Optional[LayerIndex] = None,
) -> torch.Tensor:
    """Layer-local spacing consistency for very large graphs (N > 100M).

    Iterates over layers and sorts within each, avoiding an O(N)-sized
    global argsort. Peak memory is O(max_layer_width) instead of O(N).

    Mathematically equivalent to the global sort path since the composite
    key already groups by layer.

    Parameters
    ----------
    pos : torch.Tensor
        [N, 2] node positions.
    node_sizes : torch.Tensor
        [N, 2] node widths and heights.
    layers : torch.Tensor
        [N] layer assignment per node.
    offsets : torch.Tensor
        [L+1] cumulative layer offsets.
    num_layers : int
        Number of layers.
    target_gap : float
        Target horizontal gap between adjacent nodes.
    device : torch.device
        Compute device.
    layer_index : LayerIndex, optional
        Layer-index structure that may already include nodes sorted by layer.

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    total_deviation_sq = torch.tensor(0.0, device=device)
    total_pairs = 0

    sorted_by_layer = layer_index.sorted_nodes if layer_index is not None else layers.argsort()

    for layer_idx in range(num_layers):
        start = int(offsets[layer_idx].item())
        end = int(offsets[layer_idx + 1].item())
        n_layer = end - start
        if n_layer < 2:
            continue

        layer_nodes = sorted_by_layer[start:end]
        layer_x = pos[layer_nodes, 0]
        local_order = layer_x.detach().argsort()
        sorted_nodes = layer_nodes[local_order]

        sorted_x = pos[sorted_nodes, 0]
        sorted_w = node_sizes[sorted_nodes, 0]

        dx = sorted_x[1:] - sorted_x[:-1]
        half_w = (sorted_w[:-1] + sorted_w[1:]) / 2.0
        gap = dx - half_w

        deviation = gap - target_gap
        total_deviation_sq = total_deviation_sq + (deviation**2).sum()
        total_pairs += n_layer - 1

    if total_pairs == 0:
        return torch.tensor(0.0, device=device)

    return total_deviation_sq / total_pairs


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

    layers = layer_index.node_to_layer
    offsets = layer_index.layer_offsets
    num_layers = layer_index.num_layers

    # For very large graphs, use layer-local sorting to avoid O(N)-sized intermediates.
    # Peak memory: O(max_layer_width) instead of O(N).
    # Results are mathematically identical since the composite key groups by layer.
    if N > 100_000_000:
        return _spacing_consistency_loss_layerlocal(
            pos,
            node_sizes,
            layers,
            offsets,
            num_layers,
            target_gap,
            device,
            layer_index=layer_index,
        )

    # Standard path: global sort (one kernel launch, fast for moderate N)
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
    return (deviation**2).mean()


# ─── Fan-out distribution loss ────────────────────────────────────────────────


def fanout_distribution_loss(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    degree_threshold: int = 5,
    edge_ctx: Optional[EdgeBatchLike] = None,
    step: int = 0,
    edge_is_sampled: bool = False,
) -> torch.Tensor:
    """Penalize uneven angular distribution of children for high-degree nodes.

    For hub nodes, compute the angular gaps between outgoing edges and penalize
    deviations from the ideal equal-spacing target. Large full-graph scans are
    amortized, but sampled edge batches still evaluate every step because they
    are already bounded by the engine's edge batching.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge indices with shape ``[2, E]``.
    degree_threshold : int, default=5
        Minimum out-degree required for a node to be treated as a fan-out hub.
    edge_ctx : EdgeBatchLike | None, default=None
        Optional sampled edge context for the current step. When present, hub
        degrees are estimated from this sampled edge set instead of the full
        graph, which keeps huge-graph solves aligned with edge batching.
    step : int, default=0
        Zero-based optimizer step. Only used for large full-edge scans.
    edge_is_sampled : bool, default=False
        Whether ``edge_index`` already represents a sampled subset of edges.

    Returns
    -------
    torch.Tensor
        Scalar fan-out regularization term.
    """
    if edge_ctx is None and not edge_is_sampled and pos.shape[0] > 1_000_000 and step % 5 != 0:
        return torch.tensor(0.0, device=pos.device, dtype=pos.dtype, requires_grad=True)

    device = pos.device
    if edge_ctx is not None:
        src, tgt = edge_ctx.src, edge_ctx.tgt
    elif edge_index.numel() == 0:
        return torch.tensor(0.0, device=pos.device)
    else:
        src, tgt = _non_self_edges(edge_index)

    if src.numel() == 0:
        return torch.tensor(0.0, device=device)

    edge_order = src.argsort()
    sorted_src = src[edge_order]
    sorted_tgt = tgt[edge_order]
    hub_nodes, hub_degrees = sorted_src.unique_consecutive(return_counts=True)

    if hub_nodes.numel() == 0:
        return torch.tensor(0.0, device=device)

    hub_mask = hub_degrees >= degree_threshold
    if not hub_mask.any():
        return torch.tensor(0.0, device=device)

    hub_offsets = torch.zeros(hub_degrees.shape[0] + 1, dtype=torch.long, device=device)
    hub_offsets[1:] = hub_degrees.cumsum(0)
    hub_starts = hub_offsets[:-1][hub_mask]

    valid_hub_mask = hub_degrees[hub_mask] >= 2
    if not valid_hub_mask.any():
        return torch.tensor(0.0, device=device)

    hub_nodes_v = hub_nodes[hub_mask][valid_hub_mask]
    hub_starts_v = hub_starts[valid_hub_mask]
    hub_degrees_v = hub_degrees[hub_mask][valid_hub_mask]
    num_hubs = hub_nodes_v.shape[0]

    child_flat_idx = torch.repeat_interleave(torch.arange(num_hubs, device=device), hub_degrees_v)
    total_children = int(hub_degrees_v.sum().item())
    child_seq = torch.arange(total_children, device=device)
    hub_child_offsets = torch.zeros(num_hubs + 1, dtype=torch.long, device=device)
    hub_child_offsets[1:] = hub_degrees_v.cumsum(0)
    local_offset = child_seq - hub_child_offsets[child_flat_idx]

    global_child_pos = hub_starts_v[child_flat_idx] + local_offset
    children_all = sorted_tgt[global_child_pos]

    hub_expanded = hub_nodes_v[child_flat_idx]
    dx = pos[children_all, 0] - pos[hub_expanded, 0]
    dy = pos[children_all, 1] - pos[hub_expanded, 1]
    angles = torch.atan2(dy, dx)

    two_pi = 2.0 * 3.141592653589793
    angles_positive = angles % two_pi
    # Use float64 for the sort key to avoid precision loss at large hub counts.
    # float32 loses integer precision above 2^24 = 16M, causing hub ID
    # collisions in the sort key and interleaved hub IDs after sorting.
    big = two_pi + 1.0
    sort_key = child_flat_idx.double() * big + angles_positive.double()
    sorted_order = sort_key.argsort()
    sorted_angles = angles_positive[sorted_order]
    sorted_hub_id = child_flat_idx[sorted_order]

    same_hub = sorted_hub_id[:-1] == sorted_hub_id[1:]
    consecutive_gaps = sorted_angles[1:] - sorted_angles[:-1]

    _, boundary_counts = sorted_hub_id.unique_consecutive(return_counts=True)
    boundary_offsets = torch.zeros(boundary_counts.shape[0] + 1, dtype=torch.long, device=device)
    boundary_offsets[1:] = boundary_counts.cumsum(0)

    first_angles = sorted_angles[boundary_offsets[:-1]]
    last_angles = sorted_angles[boundary_offsets[1:] - 1]
    wrap_gaps = two_pi - (last_angles - first_angles)

    ideal_gaps = two_pi / hub_degrees_v.float()
    ideal_expanded_consecutive = ideal_gaps[sorted_hub_id[:-1][same_hub]]
    gap_deviation_consecutive = (consecutive_gaps[same_hub] - ideal_expanded_consecutive) ** 2
    # Defensive: if unique_consecutive found fewer groups than expected
    # (rare edge case from float precision or degenerate angles), align sizes
    if wrap_gaps.shape[0] != ideal_gaps.shape[0]:
        min_len = min(wrap_gaps.shape[0], ideal_gaps.shape[0])
        wrap_gaps = wrap_gaps[:min_len]
        ideal_gaps = ideal_gaps[:min_len]
    gap_deviation_wrap = (wrap_gaps - ideal_gaps) ** 2

    per_hub_gap_loss = torch.zeros(num_hubs, device=device)
    per_hub_gap_loss.scatter_add_(0, sorted_hub_id[:-1][same_hub], gap_deviation_consecutive)
    # Align gap_deviation_wrap with per_hub_gap_loss if sizes differ
    if gap_deviation_wrap.shape[0] < num_hubs:
        padded = torch.zeros(num_hubs, device=device)
        padded[: gap_deviation_wrap.shape[0]] = gap_deviation_wrap
        gap_deviation_wrap = padded
    elif gap_deviation_wrap.shape[0] > num_hubs:
        gap_deviation_wrap = gap_deviation_wrap[:num_hubs]
    per_hub_gap_loss = per_hub_gap_loss + gap_deviation_wrap
    per_hub_mean = per_hub_gap_loss / hub_degrees_v.float()
    return per_hub_mean.mean()


# ─── Back-edge compactness loss ───────────────────────────────────────────────


def back_edge_compactness_loss(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    edge_ctx: Optional[EdgeBatchLike] = None,
) -> torch.Tensor:
    """Penalize horizontal distance in back-edge pairs (target above source).

    Back edges (where target y < source y) should route compactly. This loss
    penalizes the squared horizontal distance between back-edge endpoints,
    encouraging tighter back-edge routing.

    O(E), trivially vectorized.
    """
    if edge_ctx is not None:
        src = edge_ctx.src
        tgt = edge_ctx.tgt
        dx = edge_ctx.dx
        dy = edge_ctx.dy
    elif edge_index.numel() == 0:
        return torch.tensor(0.0, device=pos.device)
    else:
        src, tgt = edge_index[0], edge_index[1]
        dx = pos[src, 0] - pos[tgt, 0]
        dy = pos[src, 1] - pos[tgt, 1]

    # Back edges: target is above source (lower y = higher on screen)
    back_mask = dy > 0

    if not back_mask.any():
        return torch.tensor(0.0, device=pos.device)

    # Horizontal distance for back edges
    del src, tgt
    return dx[back_mask].square().mean()


def constraint_order_loss(
    pos: torch.Tensor,
    pairs: List[Tuple[torch.Tensor, torch.Tensor, int, float, float]],
) -> torch.Tensor:
    """Penalize directed ordering violations for user constraints.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    pairs : list[tuple[torch.Tensor, torch.Tensor, int, float, float]]
        Order constraints as left/right index tensors, axis, gap, and weight.

    Returns
    -------
    torch.Tensor
        Scalar normalized hinge loss.
    """
    terms = []
    for left, right, axis, gap, weight in pairs:
        if left.numel() == 0 or right.numel() == 0:
            continue
        left_coord = pos[left, axis].mean()
        right_coord = pos[right, axis].mean()
        terms.append(weight * F.relu(left_coord + gap - right_coord).square())
    if not terms:
        return torch.zeros((), device=pos.device, dtype=pos.dtype)
    return torch.stack(terms).mean()


def constraint_separate_loss(
    pos: torch.Tensor,
    pairs: List[Tuple[torch.Tensor, torch.Tensor, Optional[int], float, float]],
) -> torch.Tensor:
    """Penalize insufficient distance between selected units.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    pairs : list[tuple[torch.Tensor, torch.Tensor, int | None, float, float]]
        Separate constraints as selections, optional axis, gap, and weight.

    Returns
    -------
    torch.Tensor
        Scalar normalized separation loss.
    """
    terms = []
    for left, right, axis, gap, weight in pairs:
        if left.numel() == 0 or right.numel() == 0:
            continue
        a = pos[left].mean(dim=0)
        b = pos[right].mean(dim=0)
        if axis is None:
            dist = (a - b).square().sum().sqrt().clamp(min=1e-6)
        else:
            dist = (a[axis] - b[axis]).abs()
        terms.append(weight * F.relu(gap - dist).square())
    if not terms:
        return torch.zeros((), device=pos.device, dtype=pos.dtype)
    return torch.stack(terms).mean()


def constraint_group_loss(
    pos: torch.Tensor,
    groups: List[Tuple[torch.Tensor, float, float]],
) -> torch.Tensor:
    """Penalize spread inside user-defined groups.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    groups : list[tuple[torch.Tensor, float, float]]
        Group index tensors with padding and weight.

    Returns
    -------
    torch.Tensor
        Scalar compactness loss.
    """
    terms = []
    for indices, padding, weight in groups:
        if indices.numel() <= 1:
            continue
        group_pos = pos[indices]
        centroid = group_pos.mean(dim=0, keepdim=True)
        terms.append(weight * (group_pos - centroid).square().sum(dim=1).mean() / (padding + 1.0))
    if not terms:
        return torch.zeros((), device=pos.device, dtype=pos.dtype)
    return torch.stack(terms).mean()


def constraint_anchor_loss(
    pos: torch.Tensor,
    anchors: List[Tuple[torch.Tensor, torch.Tensor, float]],
) -> torch.Tensor:
    """Penalize deviation from fixed external anchor coordinates.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    anchors : list[tuple[torch.Tensor, torch.Tensor, float]]
        Index tensors, target tensors, and weights.

    Returns
    -------
    torch.Tensor
        Scalar anchor loss.
    """
    terms = []
    for indices, targets, weight in anchors:
        if indices.numel() == 0:
            continue
        terms.append(weight * (pos[indices] - targets).square().mean())
    if not terms:
        return torch.zeros((), device=pos.device, dtype=pos.dtype)
    return torch.stack(terms).mean()


def constraint_emphasize_loss(
    pos: torch.Tensor,
    paths: List[Tuple[torch.Tensor, float]],
) -> torch.Tensor:
    """Encourage emphasized paths to be shorter and more collinear.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    paths : list[tuple[torch.Tensor, float]]
        Ordered path indices and weights.

    Returns
    -------
    torch.Tensor
        Scalar path emphasis loss.
    """
    terms = []
    for indices, weight in paths:
        if indices.numel() <= 2:
            continue
        points = pos[indices]
        deltas = points[1:] - points[:-1]
        length_term = deltas.square().sum(dim=1).mean()
        chord = points[-1] - points[0]
        norm = chord.square().sum().sqrt().clamp(min=1e-6)
        direction = chord / norm
        centered = points - points[0]
        projected = centered @ direction
        nearest = points[0] + projected.unsqueeze(1) * direction
        terms.append(weight * (0.01 * length_term + (points - nearest).square().mean()))
    if not terms:
        return torch.zeros((), device=pos.device, dtype=pos.dtype)
    return torch.stack(terms).mean()


def constraint_focus_loss(
    pos: torch.Tensor,
    focuses: List[Tuple[torch.Tensor, torch.Tensor, float, float]],
) -> torch.Tensor:
    """Pull focused selections toward target points.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    focuses : list[tuple[torch.Tensor, torch.Tensor, float, float]]
        Selected indices, target point, zoom, and weight.

    Returns
    -------
    torch.Tensor
        Scalar focus loss.
    """
    terms = []
    for indices, target, zoom, weight in focuses:
        if indices.numel() == 0:
            continue
        target = target.to(device=pos.device, dtype=pos.dtype)
        scale = max(float(zoom), 1e-6)
        terms.append(weight * (pos[indices] - target).square().mean() / scale)
    if not terms:
        return torch.zeros((), device=pos.device, dtype=pos.dtype)
    return torch.stack(terms).mean()


def constraint_contain_loss(
    pos: torch.Tensor,
    contains: List[Tuple[torch.Tensor, Any, float, float]],
) -> torch.Tensor:
    """Penalize selected points outside their container.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    contains : list[tuple[torch.Tensor, Any, float, float]]
        Containment payloads as selected indices, container selector, padding,
        and finite weight.

    Returns
    -------
    torch.Tensor
        Scalar normalized outside-distance loss.
    """
    terms = []
    for indices, within, padding, weight in contains:
        if indices.numel() == 0:
            continue
        xmin, ymin, xmax, ymax = _contain_bounds(pos, within, padding)
        points = pos[indices]
        dx = F.relu(xmin - points[:, 0]) + F.relu(points[:, 0] - xmax)
        dy = F.relu(ymin - points[:, 1]) + F.relu(points[:, 1] - ymax)
        terms.append(weight * (dx.square() + dy.square()).mean())
    if not terms:
        return torch.zeros((), device=pos.device, dtype=pos.dtype)
    return torch.stack(terms).mean()


def project_hard_contains(
    pos: torch.Tensor,
    contains: List[Tuple[torch.Tensor, Any, float, float]],
) -> None:
    """Clamp hard containment selections into their derived container.

    Parameters
    ----------
    pos : torch.Tensor
        Mutable position tensor with shape ``[N, 2]``.
    contains : list[tuple[torch.Tensor, Any, float, float]]
        Hard containment payloads.

    Returns
    -------
    None
        Mutates ``pos`` in place.
    """
    if not contains:
        return
    with torch.no_grad():
        for indices, within, padding, _weight in contains:
            if indices.numel() == 0:
                continue
            xmin, ymin, xmax, ymax = _contain_bounds(pos, within, padding)
            pos[indices, 0].clamp_(xmin, xmax)
            pos[indices, 1].clamp_(ymin, ymax)


def _contain_bounds(
    pos: torch.Tensor,
    within: Any,
    padding: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return containment bounds in layout coordinates.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    within : Any
        Canvas selector or tensor of container node indices.
    padding : float
        Interior padding.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ``xmin, ymin, xmax, ymax`` tensors on ``pos.device``.
    """
    from dagua.constraints import CanvasSelector, resolve_canvas_point

    pad = torch.as_tensor(float(padding), dtype=pos.dtype, device=pos.device)
    if isinstance(within, CanvasSelector):
        left = resolve_canvas_point(within.edge("left"), pos)[0]
        top = resolve_canvas_point(within.edge("top"), pos)[1]
        right = resolve_canvas_point(within.edge("right"), pos)[0]
        bottom = resolve_canvas_point(within.edge("bottom"), pos)[1]
        if left is None or top is None or right is None or bottom is None:
            raise ValueError("Canvas containment edges must resolve to concrete bounds.")
        return (
            torch.as_tensor(float(left), dtype=pos.dtype, device=pos.device) + pad,
            torch.as_tensor(float(top), dtype=pos.dtype, device=pos.device) + pad,
            torch.as_tensor(float(right), dtype=pos.dtype, device=pos.device) - pad,
            torch.as_tensor(float(bottom), dtype=pos.dtype, device=pos.device) - pad,
        )
    if isinstance(within, torch.Tensor) and within.numel() > 0:
        container = pos[within]
        return (
            container[:, 0].min() + pad,
            container[:, 1].min() + pad,
            container[:, 0].max() - pad,
            container[:, 1].max() - pad,
        )
    xmin = pos[:, 0].min() + pad
    ymin = pos[:, 1].min() + pad
    xmax = pos[:, 0].max() - pad
    ymax = pos[:, 1].max() - pad
    return xmin, ymin, xmax, ymax
