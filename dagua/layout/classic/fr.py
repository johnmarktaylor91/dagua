"""Fruchterman-Reingold force-directed layout.

Classic spring-electrical model: all nodes repel (Coulomb), connected nodes
attract (Hooke). Positions update by direct force displacement with linear
cooling. No gradients, no optimizer — pure physics simulation.

Reference: Fruchterman & Reingold, "Graph Drawing by Force-directed Placement"
(1991), Software — Practice and Experience.
"""

from __future__ import annotations

from math import sqrt
from typing import Optional, Union

import torch

MIN_DISTANCE = 0.01
SAMPLED_REPULSION_LIMIT = 10_000
SAMPLED_NEIGHBORS = 1_000
CONVERGENCE_THRESHOLD = 1.0e-4
OUTPUT_SCALE_FACTOR = 50.0


def _layout_device(
    edge_index: torch.Tensor,
    node_sizes: Optional[torch.Tensor],
) -> torch.device:
    """Choose the compute device for the layout.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor.

    Returns
    -------
    torch.device
        Device used for the simulation tensors.
    """
    if edge_index.numel() > 0:
        return edge_index.device
    if node_sizes is not None:
        return node_sizes.device
    return torch.device("cpu")


def _initialize_positions(
    num_nodes: int,
    area: float,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Create seeded random initial positions within the layout square.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    area : float
        Bounding area of the layout square.
    device : torch.device
        Device for the position tensor.
    seed : int
        Random seed for initialization.

    Returns
    -------
    torch.Tensor
        Initial positions with shape ``[N, 2]``.
    """
    side = area**0.5
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    pos = torch.rand((num_nodes, 2), generator=generator, dtype=torch.float32)
    return pos.mul(side).to(device)


def _repulsive_displacement_full(pos: torch.Tensor, k: float) -> torch.Tensor:
    """Compute all-pairs repulsive displacement with full broadcasting.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    k : float
        Ideal spring length.

    Returns
    -------
    torch.Tensor
        Repulsive displacement with shape ``[N, 2]``.
    """
    delta = pos.unsqueeze(1) - pos.unsqueeze(0)
    dist = torch.linalg.norm(delta, dim=2).clamp(min=MIN_DISTANCE)
    force = (k * k) / dist
    return (delta / dist.unsqueeze(2) * force.unsqueeze(2)).sum(dim=1)


def _repulsive_displacement_sampled(
    pos: torch.Tensor,
    k: float,
    seed: int,
    step: int,
) -> torch.Tensor:
    """Approximate repulsive displacement by random sampling for large graphs.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    k : float
        Ideal spring length.
    seed : int
        Base random seed.
    step : int
        Current simulation step.

    Returns
    -------
    torch.Tensor
        Approximate repulsive displacement with shape ``[N, 2]``.

    Notes
    -----
    NetworkX evaluates the full ``O(N^2)`` repulsion term. Dagua keeps this
    sampled approximation only above ``SAMPLED_REPULSION_LIMIT`` as a
    graph-size-specific acceleration; for smaller graphs the force update
    matches the reference implementation.
    """
    num_nodes = pos.shape[0]
    sample_size = min(num_nodes, SAMPLED_NEIGHBORS)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + step + 1)
    sampled = torch.randint(
        0,
        num_nodes,
        (num_nodes, sample_size),
        generator=generator,
        dtype=torch.long,
    ).to(pos.device)
    neighbors = pos[sampled]
    delta = pos.unsqueeze(1) - neighbors
    dist = torch.linalg.norm(delta, dim=2).clamp(min=MIN_DISTANCE)
    force = (k * k) / dist
    return (delta / dist.unsqueeze(2) * force.unsqueeze(2)).sum(dim=1)


def _repulsive_displacement(
    pos: torch.Tensor,
    k: float,
    seed: int,
    step: int,
) -> torch.Tensor:
    """Dispatch between full and sampled repulsion modes.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    k : float
        Ideal spring length.
    seed : int
        Base random seed.
    step : int
        Current simulation step.

    Returns
    -------
    torch.Tensor
        Repulsive displacement with shape ``[N, 2]``.
    """
    if pos.shape[0] > SAMPLED_REPULSION_LIMIT:
        return _repulsive_displacement_sampled(pos, k, seed, step)
    return _repulsive_displacement_full(pos, k)


def _adjacency_matrix(
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    """Build the dense symmetric adjacency used by the exact FR update.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes in the graph.
    device : torch.device
        Device for the adjacency matrix.

    Returns
    -------
    torch.Tensor
        Dense adjacency matrix with shape ``[N, N]``.
    """
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    if edge_index.numel() == 0:
        return adjacency

    src = edge_index[0].to(dtype=torch.long, device=device)
    dst = edge_index[1].to(dtype=torch.long, device=device)
    adjacency[src, dst] = 1.0
    adjacency[dst, src] = 1.0
    return adjacency


def _exact_displacement(pos: torch.Tensor, adjacency: torch.Tensor, k: float) -> torch.Tensor:
    """Compute the exact NetworkX FR displacement for smaller graphs.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    adjacency : torch.Tensor
        Dense adjacency matrix with shape ``[N, N]``.
    k : float
        Ideal spring length.

    Returns
    -------
    torch.Tensor
        Displacement tensor with shape ``[N, 2]``.
    """
    delta = pos.unsqueeze(1) - pos.unsqueeze(0)
    distance = torch.linalg.norm(delta, dim=-1).clamp(min=MIN_DISTANCE)
    return torch.einsum(
        "ijk,ij->ik",
        delta,
        ((k * k) / distance.square()) - (adjacency * distance / k),
    )


def _attractive_displacement(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    k: float,
) -> torch.Tensor:
    """Compute edge-based attractive displacement.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    k : float
        Ideal spring length.

    Returns
    -------
    torch.Tensor
        Attractive displacement with shape ``[N, 2]``.
    """
    disp = torch.zeros_like(pos)
    if edge_index.numel() == 0:
        return disp

    src = edge_index[0].to(dtype=torch.long, device=pos.device)
    dst = edge_index[1].to(dtype=torch.long, device=pos.device)

    delta = pos[dst] - pos[src]
    dist = torch.linalg.norm(delta, dim=1).clamp(min=MIN_DISTANCE)
    force = (dist * dist) / k
    edge_disp = (delta / dist.unsqueeze(1)) * force.unsqueeze(1)

    disp.index_add_(0, src, edge_disp)
    disp.index_add_(0, dst, -edge_disp)
    return disp


def _initial_temperature(pos: torch.Tensor) -> float:
    """Compute the NetworkX initial temperature from the current extent.

    Parameters
    ----------
    pos : torch.Tensor
        Current positions with shape ``[N, 2]``.

    Returns
    -------
    float
        Initial simulation temperature.
    """
    if pos.shape[0] == 0:
        return 0.0

    x_extent = float((pos[:, 0].max() - pos[:, 0].min()).item())
    y_extent = float((pos[:, 1].max() - pos[:, 1].min()).item())
    return max(x_extent, y_extent) * 0.1


def _rescale_layout(positions: torch.Tensor, scale: float) -> torch.Tensor:
    """Center and rescale positions like ``networkx.rescale_layout``.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    scale : float
        Target half-width after rescaling.

    Returns
    -------
    torch.Tensor
        Rescaled positions with shape ``[N, 2]``.
    """
    centered = positions - positions.mean(dim=0, keepdim=True)
    limit = float(centered.abs().max().item())
    if limit > 0.0:
        centered = centered * (scale / limit)
    return centered


def layout_fr(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 50,
    seed: int = 42,
    area: Optional[float] = None,
    trace_every: int = 0,
) -> Union[torch.Tensor, tuple[torch.Tensor, list[torch.Tensor]]]:
    """Run Fruchterman-Reingold layout.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list, shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Node sizes ``[N, 2]``. Unused by FR but accepted for interface compat.
    steps : int
        Number of simulation steps.
    seed : int
        Random seed for initial placement.
    area : float, optional
        Bounding area for the layout. Default ``num_nodes * 100``.
    trace_every : int
        If > 0, record position snapshots every this many steps.
        When active, returns ``(final_pos, [snapshots])``.

    Returns
    -------
    torch.Tensor or tuple
        Final positions ``[N, 2]``, or ``(positions, traces)`` if tracing.
    """
    _ = node_sizes

    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if trace_every < 0:
        raise ValueError("trace_every must be non-negative")

    device = _layout_device(edge_index, node_sizes)
    empty = torch.empty((0, 2), dtype=torch.float32, device=device)
    traces: list[torch.Tensor] = []
    if num_nodes == 0:
        return (empty, traces) if trace_every > 0 else empty

    layout_area = float(num_nodes * 100 if area is None else area)
    if layout_area <= 0.0:
        raise ValueError("area must be positive")

    pos = _initialize_positions(num_nodes, layout_area, device, seed)
    adjacency = (
        _adjacency_matrix(edge_index=edge_index, num_nodes=num_nodes, device=device)
        if num_nodes <= SAMPLED_REPULSION_LIMIT
        else None
    )
    if steps == 0:
        scaled = _rescale_layout(pos, scale=sqrt(max(num_nodes, 1)) * OUTPUT_SCALE_FACTOR)
        return (scaled, traces) if trace_every > 0 else scaled

    k = (layout_area / num_nodes) ** 0.5
    temp = _initial_temperature(pos)
    cooling = temp / (steps + 1)

    for step in range(steps):
        if adjacency is not None:
            disp = _exact_displacement(pos, adjacency, k)
        else:
            disp = _repulsive_displacement(pos, k, seed, step)
            disp = disp + _attractive_displacement(pos, edge_index, k)

        disp_norm = torch.linalg.norm(disp, dim=1, keepdim=True).clamp(min=MIN_DISTANCE)
        delta_pos = disp * (temp / disp_norm)
        pos = pos + delta_pos

        temp -= cooling

        if trace_every > 0 and step % trace_every == 0:
            traces.append(pos.clone())

        if (torch.linalg.norm(delta_pos) / num_nodes) < CONVERGENCE_THRESHOLD:
            break

    scaled = _rescale_layout(pos, scale=sqrt(max(num_nodes, 1)) * OUTPUT_SCALE_FACTOR)

    if trace_every > 0:
        return scaled, traces
    return scaled
