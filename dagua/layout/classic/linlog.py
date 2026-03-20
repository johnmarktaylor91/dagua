"""LinLog energy graph layout.

This module implements a LinLog-style energy model with linear attraction on
edges and logarithmic repulsion over sampled node pairs. The optimizer is a
small Adam loop over node coordinates.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

_FULL_REPULSION_LIMIT = 2_000
_SAMPLED_REPULSION_NEIGHBORS = 128
_MIN_DISTANCE = 1.0e-3


def _layout_device(
    edge_index: torch.Tensor,
    node_sizes: Optional[torch.Tensor],
) -> torch.device:
    """Choose the output device for the layout.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor.

    Returns
    -------
    torch.device
        Output device.
    """
    if edge_index.numel() > 0:
        return edge_index.device
    if node_sizes is not None:
        return node_sizes.device
    return torch.device("cpu")


def _layout_extent(num_nodes: int, node_sizes: Optional[torch.Tensor]) -> float:
    """Estimate a drawing extent for final normalization.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Optional node sizes.

    Returns
    -------
    float
        Target half-width for the drawing.
    """
    if node_sizes is None or node_sizes.numel() == 0:
        return max(float(max(num_nodes, 1)) ** 0.5 * 5.0, 1.0)

    max_size = float(node_sizes.to(dtype=torch.float32, device="cpu").max().item())
    return max(max_size * max(float(max(num_nodes, 1)) ** 0.5, 1.0) * 2.0, 1.0)


def _initialize_positions(num_nodes: int, device: torch.device, seed: int) -> torch.Tensor:
    """Create deterministic random initial coordinates.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    device : torch.device
        Device for the position tensor.
    seed : int
        Random seed.

    Returns
    -------
    torch.Tensor
        Initial positions with shape ``[N, 2]``.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    positions = torch.randn((num_nodes, 2), generator=generator, dtype=torch.float32)
    return positions.to(device)


def _normalize_positions(positions: torch.Tensor, extent: float) -> torch.Tensor:
    """Center and scale coordinates to a stable output range.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    extent : float
        Target half-width.

    Returns
    -------
    torch.Tensor
        Normalized coordinates.
    """
    if positions.shape[0] <= 1:
        return torch.zeros_like(positions)

    centered = positions - positions.mean(dim=0, keepdim=True)
    span = centered.abs().max().clamp(min=1.0)
    return centered * (extent / span)


def _unique_undirected_edge_ids(
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    """Pack unique undirected edges into sortable integer ids.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    num_nodes : int
        Number of nodes in the graph.
    device : torch.device
        Device for the returned tensor.

    Returns
    -------
    torch.Tensor
        Unique packed edge ids with shape ``[E_unique]``.
    """
    if edge_index.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=device)

    src = edge_index[0].to(device=device, dtype=torch.long)
    dst = edge_index[1].to(device=device, dtype=torch.long)
    ordered_src = torch.minimum(src, dst)
    ordered_dst = torch.maximum(src, dst)
    mask = ordered_src != ordered_dst
    if not bool(mask.any()):
        return torch.empty((0,), dtype=torch.long, device=device)
    packed_ids = ordered_src[mask] * num_nodes + ordered_dst[mask]
    return torch.unique(packed_ids)


def _full_non_edge_pairs(
    num_nodes: int,
    edge_ids: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Enumerate all unordered non-edge pairs for exact repulsion.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    edge_ids : torch.Tensor
        Packed unique undirected edge ids.
    device : torch.device
        Device for the returned tensors.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Source and target indices for all unordered non-edge pairs.
    """
    src, dst = torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)
    if edge_ids.numel() == 0:
        return src, dst

    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
    edge_src = edge_ids // num_nodes
    edge_dst = edge_ids % num_nodes
    adjacency[edge_src, edge_dst] = True
    adjacency[edge_dst, edge_src] = True
    mask = ~adjacency[src, dst]
    return src[mask], dst[mask]


def _sample_non_edge_pairs(
    num_nodes: int,
    edge_ids: torch.Tensor,
    device: torch.device,
    step: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Draw deterministic non-edge pairs for sampled repulsion.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    edge_ids : torch.Tensor
        Packed unique undirected edge ids.
    device : torch.device
        Device for the sampled tensors.
    step : int
        Optimization step.
    seed : int
        Base random seed.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, int]
        Source indices, target indices, and the total non-edge pair count.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + step + 1)
    total_pairs = max(num_nodes * (num_nodes - 1) // 2, 1)
    total_non_edge_pairs = max(total_pairs - int(edge_ids.numel()), 0)
    sample_size = min(
        total_non_edge_pairs,
        max(num_nodes, num_nodes * _SAMPLED_REPULSION_NEIGHBORS // 2),
    )
    if sample_size == 0:
        empty = torch.empty((0,), dtype=torch.long, device=device)
        return empty, empty, total_non_edge_pairs

    edge_ids_cpu = edge_ids.to(device="cpu", dtype=torch.long)
    sampled_src: list[torch.Tensor] = []
    sampled_dst: list[torch.Tensor] = []
    collected = 0

    for _ in range(8):
        if collected >= sample_size:
            break

        draw_size = max((sample_size - collected) * 3, num_nodes)
        src = torch.randint(0, num_nodes, (draw_size,), generator=generator, dtype=torch.long)
        dst = torch.randint(0, num_nodes, (draw_size,), generator=generator, dtype=torch.long)
        distinct_mask = src != dst
        if not bool(distinct_mask.any()):
            continue

        ordered_src = torch.minimum(src[distinct_mask], dst[distinct_mask])
        ordered_dst = torch.maximum(src[distinct_mask], dst[distinct_mask])
        if edge_ids_cpu.numel() > 0:
            pair_ids = ordered_src * num_nodes + ordered_dst
            non_edge_mask = ~torch.isin(pair_ids, edge_ids_cpu)
            ordered_src = ordered_src[non_edge_mask]
            ordered_dst = ordered_dst[non_edge_mask]

        if ordered_src.numel() == 0:
            continue

        take_count = min(int(ordered_src.numel()), sample_size - collected)
        sampled_src.append(ordered_src[:take_count])
        sampled_dst.append(ordered_dst[:take_count])
        collected += take_count

    if collected == 0:
        empty = torch.empty((0,), dtype=torch.long, device=device)
        return empty, empty, total_non_edge_pairs

    return (
        torch.cat(sampled_src).to(device=device),
        torch.cat(sampled_dst).to(device=device),
        total_non_edge_pairs,
    )


def _linlog_loss(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    seed: int,
    step: int,
    a: float = 1.0,
    r: float = 0.0,
) -> torch.Tensor:
    """Evaluate the sampled LinLog objective.

    Parameters
    ----------
    positions : torch.Tensor
        Current coordinates with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    seed : int
        Base random seed.
    step : int
        Optimization step.
    a : float, default=1.0
        Attraction exponent in ``|p_i - p_j|^a``.
    r : float, default=0.0
        Repulsion exponent in ``-|p_i - p_j|^r``. ``0`` recovers the standard
        logarithmic LinLog repulsion.

    Returns
    -------
    torch.Tensor
        Scalar objective value.
    """
    attraction = torch.tensor(0.0, dtype=positions.dtype, device=positions.device)
    edge_ids = _unique_undirected_edge_ids(
        edge_index=edge_index,
        num_nodes=int(positions.shape[0]),
        device=positions.device,
    )
    if edge_index.numel() > 0:
        src = edge_index[0].to(device=positions.device, dtype=torch.long)
        dst = edge_index[1].to(device=positions.device, dtype=torch.long)
        edge_lengths = torch.linalg.norm(positions[src] - positions[dst], dim=1).clamp(
            min=_MIN_DISTANCE
        )
        attraction = edge_lengths.pow(a).sum()

    num_nodes = int(positions.shape[0])
    if num_nodes <= _FULL_REPULSION_LIMIT:
        non_edge_src, non_edge_dst = _full_non_edge_pairs(
            num_nodes=num_nodes,
            edge_ids=edge_ids,
            device=positions.device,
        )
        if int(non_edge_src.numel()) == 0:
            repulsion = torch.tensor(0.0, dtype=positions.dtype, device=positions.device)
        else:
            pairwise_distances = torch.linalg.norm(
                positions[non_edge_src] - positions[non_edge_dst],
                dim=1,
            ).clamp(min=_MIN_DISTANCE)
            if r == 0.0:
                repulsion = -torch.log(pairwise_distances).sum()
            else:
                repulsion = -pairwise_distances.pow(r).sum()
    else:
        src, dst, total_non_edge_pairs = _sample_non_edge_pairs(
            num_nodes=num_nodes,
            edge_ids=edge_ids,
            device=positions.device,
            step=step,
            seed=seed,
        )
        sampled_lengths = torch.linalg.norm(
            positions[src] - positions[dst],
            dim=1,
        ).clamp(min=_MIN_DISTANCE)
        if int(sampled_lengths.numel()) == 0:
            repulsion = torch.tensor(0.0, dtype=positions.dtype, device=positions.device)
        elif r == 0.0:
            repulsion = -torch.log(sampled_lengths).mean() * float(total_non_edge_pairs)
        else:
            repulsion = -sampled_lengths.pow(r).mean() * float(total_non_edge_pairs)

    return attraction + repulsion


def layout_linlog(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 300,
    seed: int = 42,
    a: float = 1.0,
    r: float = 0.0,
) -> torch.Tensor:
    """Lay out a graph with a LinLog energy model.

    Reference
    ---------
    Noack, "An Energy Model for Visual Graph Clustering" (2009).

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Optional node sizes used only for scaling the final layout.
    steps : int, default=300
        Number of Adam updates.
    seed : int, default=42
        Random seed for initialization and repulsion sampling.
    a : float, default=1.0
        Attraction exponent in ``|p_i - p_j|^a``.
    r : float, default=0.0
        Repulsion exponent in ``-|p_i - p_j|^r``. ``0`` recovers the standard
        logarithmic LinLog energy.

    Returns
    -------
    torch.Tensor
        Node positions with shape ``[N, 2]``.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if a < 0:
        raise ValueError("a must be non-negative.")
    if r < 0:
        raise ValueError("r must be non-negative.")

    device = _layout_device(edge_index, node_sizes)
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32, device=device)

    positions = _initialize_positions(num_nodes, device, seed).requires_grad_(True)
    initial_lr = min(0.05, 0.8 / float(max(num_nodes, 1)))
    final_lr = max(initial_lr * 0.1, initial_lr / math.sqrt(float(max(steps, 1))))
    optimizer = torch.optim.Adam([positions], lr=initial_lr)

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = _linlog_loss(positions, edge_index, seed, step, a=a, r=r)
        loss.backward()
        optimizer.step()

        fraction = float(step + 1) / float(max(steps, 1))
        decay = initial_lr + (final_lr - initial_lr) * fraction
        optimizer.param_groups[0]["lr"] = decay

    extent = _layout_extent(num_nodes, node_sizes)
    return _normalize_positions(positions.detach(), extent).to(dtype=torch.float32, device=device)
