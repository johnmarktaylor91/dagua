"""LinLog energy graph layout.

This module implements a LinLog-style energy model with linear attraction on
edges and logarithmic repulsion over sampled node pairs. The optimizer is a
small Adam loop over node coordinates.
"""

from __future__ import annotations

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


def _sample_repulsion_pairs(
    num_nodes: int,
    device: torch.device,
    step: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw deterministic random node pairs for repulsion sampling.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    device : torch.device
        Device for the sampled tensors.
    step : int
        Optimization step.
    seed : int
        Base random seed.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Source and target index tensors of equal length.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + step + 1)
    sample_size = min(num_nodes, _SAMPLED_REPULSION_NEIGHBORS)
    src = torch.arange(num_nodes, dtype=torch.long).repeat_interleave(sample_size)
    dst = torch.randint(
        0,
        num_nodes,
        (num_nodes * sample_size,),
        generator=generator,
        dtype=torch.long,
    )
    mask = src != dst
    return src[mask].to(device), dst[mask].to(device)


def _linlog_loss(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    seed: int,
    step: int,
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

    Returns
    -------
    torch.Tensor
        Scalar objective value.
    """
    attraction = torch.tensor(0.0, dtype=positions.dtype, device=positions.device)
    if edge_index.numel() > 0:
        src = edge_index[0].to(device=positions.device, dtype=torch.long)
        dst = edge_index[1].to(device=positions.device, dtype=torch.long)
        edge_lengths = torch.linalg.norm(positions[src] - positions[dst], dim=1).clamp(
            min=_MIN_DISTANCE
        )
        attraction = edge_lengths.mean()

    num_nodes = int(positions.shape[0])
    if num_nodes <= _FULL_REPULSION_LIMIT:
        delta = positions.unsqueeze(1) - positions.unsqueeze(0)
        distances = torch.linalg.norm(delta, dim=2).clamp(min=_MIN_DISTANCE)
        mask = ~torch.eye(num_nodes, dtype=torch.bool, device=positions.device)
        repulsion = -torch.log(distances[mask]).mean()
    else:
        src, dst = _sample_repulsion_pairs(num_nodes, positions.device, step, seed)
        sampled_lengths = torch.linalg.norm(positions[src] - positions[dst], dim=1).clamp(
            min=_MIN_DISTANCE
        )
        repulsion = -torch.log(sampled_lengths).mean()

    gravity = 0.01 * positions.square().mean()
    return attraction + 1.5 * repulsion + gravity


def layout_linlog(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 300,
    seed: int = 42,
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

    Returns
    -------
    torch.Tensor
        Node positions with shape ``[N, 2]``.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")

    device = _layout_device(edge_index, node_sizes)
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32, device=device)

    positions = _initialize_positions(num_nodes, device, seed).requires_grad_(True)
    optimizer = torch.optim.Adam([positions], lr=0.08)

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = _linlog_loss(positions, edge_index, seed, step)
        loss.backward()
        optimizer.step()

        decay = 0.08 * (1.0 - float(step + 1) / float(max(steps, 1))) + 0.01
        optimizer.param_groups[0]["lr"] = decay

    extent = _layout_extent(num_nodes, node_sizes)
    return _normalize_positions(positions.detach(), extent).to(dtype=torch.float32, device=device)
