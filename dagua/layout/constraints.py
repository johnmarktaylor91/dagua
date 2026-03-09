"""Composable constraint loss functions.

Each function takes pos [N, 2] and relevant graph data, returns scalar loss tensor.
All losses are differentiable through PyTorch autograd.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


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
    threshold: int = 5000,
    sample_k: int = 128,
) -> torch.Tensor:
    """All nodes repel each other. Exact for small graphs, sampled for large.

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
        # Negative sampling
        k = min(sample_k, num_nodes - 1)
        idx = torch.randint(0, num_nodes, (num_nodes, k), device=pos.device)
        diff = pos.unsqueeze(1) - pos[idx]  # [N, k, 2]
        dist_sq = (diff**2).sum(dim=2) + 1e-4
        return (1.0 / dist_sq).mean()


def overlap_avoidance_loss(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    padding: float = 2.0,
) -> torch.Tensor:
    """Soft penalty on bounding box intersection. Only when overlapping on BOTH axes.

    Cooperates with: repulsion_loss.
    """
    n = pos.shape[0]
    if n <= 1:
        return torch.tensor(0.0, device=pos.device)

    # For efficiency, only check nearby pairs for large graphs
    if n > 500:
        return _overlap_loss_sampled(pos, node_sizes, padding, k=256)

    # All pairs
    dx_abs = torch.abs(pos.unsqueeze(0)[:, :, 0] - pos.unsqueeze(1)[:, :, 0])
    dy_abs = torch.abs(pos.unsqueeze(0)[:, :, 1] - pos.unsqueeze(1)[:, :, 1])

    min_dx = (node_sizes.unsqueeze(0)[:, :, 0] + node_sizes.unsqueeze(1)[:, :, 0]) / 2 + padding
    min_dy = (node_sizes.unsqueeze(0)[:, :, 1] + node_sizes.unsqueeze(1)[:, :, 1]) / 2 + padding

    overlap = F.relu(min_dx - dx_abs) * F.relu(min_dy - dy_abs)

    # Zero out diagonal
    mask = ~torch.eye(n, dtype=torch.bool, device=pos.device)
    return overlap[mask].mean()


def _overlap_loss_sampled(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    padding: float,
    k: int = 256,
) -> torch.Tensor:
    """Sampled overlap loss for larger graphs."""
    n = pos.shape[0]
    k = min(k, n - 1)
    idx = torch.randint(0, n, (n, k), device=pos.device)

    dx_abs = torch.abs(pos[:, 0].unsqueeze(1) - pos[idx, 0])
    dy_abs = torch.abs(pos[:, 1].unsqueeze(1) - pos[idx, 1])

    min_dx = (node_sizes[:, 0].unsqueeze(1) + node_sizes[idx, 0]) / 2 + padding
    min_dy = (node_sizes[:, 1].unsqueeze(1) + node_sizes[idx, 1]) / 2 + padding

    overlap = F.relu(min_dx - dx_abs) * F.relu(min_dy - dy_abs)
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
    """Sibling cluster bounding boxes repel. Differentiable through min/max."""
    if device is None:
        device = pos.device

    cluster_list = [
        (name, torch.tensor(members, device=device, dtype=torch.long))
        for name, members in clusters.items()
        if isinstance(members, list) and len(members) > 0
    ]

    if len(cluster_list) < 2:
        return torch.tensor(0.0, device=device)

    total = torch.tensor(0.0, device=device)
    for i in range(len(cluster_list)):
        for j in range(i + 1, len(cluster_list)):
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
) -> torch.Tensor:
    """Differentiable crossing proxy using sigmoid relaxation.

    Competes with: dag_ordering_loss (strict layers vs flexible ordering).
    """
    num_edges = edge_index.shape[1]
    if num_edges < 2:
        return torch.tensor(0.0, device=pos.device)

    # Sample edge pairs if too many
    if num_edges * (num_edges - 1) // 2 > max_pairs:
        n_sample = min(max_pairs, num_edges)
        perm = torch.randperm(num_edges, device=pos.device)[:n_sample]
        ei = edge_index[:, perm]
    else:
        ei = edge_index
        n_sample = ei.shape[1]

    # For each pair of edges, check if they cross
    total = torch.tensor(0.0, device=pos.device)
    count = 0

    # Vectorized: compute source and target x-positions
    src_x = pos[ei[0], 0]
    tgt_x = pos[ei[1], 0]

    # Check pairs
    n = ei.shape[1]
    if n > 200:
        # Sample pairs
        n_pairs = min(max_pairs, n * (n - 1) // 2)
        i_idx = torch.randint(0, n, (n_pairs,), device=pos.device)
        j_idx = torch.randint(0, n, (n_pairs,), device=pos.device)
        # Avoid self-pairs
        mask = i_idx != j_idx
        i_idx, j_idx = i_idx[mask], j_idx[mask]
    else:
        # All pairs
        i_idx, j_idx = torch.triu_indices(n, n, offset=1, device=pos.device)

    if i_idx.numel() == 0:
        return torch.tensor(0.0, device=pos.device)

    dx_src = src_x[i_idx] - src_x[j_idx]
    dx_tgt = tgt_x[i_idx] - tgt_x[j_idx]

    # Crossing happens when source order flips vs target order
    crossing_proxy = torch.sigmoid(-alpha * dx_src * dx_tgt)
    return crossing_proxy.mean()


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
