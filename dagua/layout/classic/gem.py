"""GEM-style adaptive force-directed layout.

This is a simplified Graph Embedder Method implementation with per-node
temperatures, adaptive impulse damping, sampled repulsion for larger graphs,
and a weak gravity term toward the barycenter.
"""

from __future__ import annotations

from typing import Optional

import torch

_MIN_DISTANCE = 1.0e-3
_FULL_REPULSION_LIMIT = 2_000
_SAMPLED_REPULSION_NEIGHBORS = 96
_PERTURBATION_MAX_ANGLE = 1.64
_TEMPERATURE_GROWTH_FACTOR = 3.0
_TEMPERATURE_SHRINK_FACTOR = 1.0 / 3.0


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
    """Estimate a stable drawing extent.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor.

    Returns
    -------
    float
        Target half-width.
    """
    if node_sizes is None or node_sizes.numel() == 0:
        return max(float(max(num_nodes, 1)) ** 0.5 * 5.0, 1.0)

    max_size = float(node_sizes.to(dtype=torch.float32, device="cpu").max().item())
    return max(max_size * max(float(max(num_nodes, 1)) ** 0.5, 1.0) * 2.0, 1.0)


def _initialize_positions(num_nodes: int, device: torch.device, seed: int) -> torch.Tensor:
    """Create deterministic random coordinates.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    device : torch.device
        Device for the result.
    seed : int
        Random seed.

    Returns
    -------
    torch.Tensor
        Initial positions with shape ``[N, 2]``.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn((num_nodes, 2), generator=generator, dtype=torch.float32).to(device)


def _normalize_positions(positions: torch.Tensor, extent: float) -> torch.Tensor:
    """Center and scale coordinates into a bounded drawing box.

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


def _ideal_distance(num_nodes: int, extent: float) -> float:
    """Estimate the FR-style ideal distance used by GEM repulsion.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    extent : float
        Target half-width of the final drawing box.

    Returns
    -------
    float
        Ideal pairwise spacing constant ``k``.
    """
    return max(extent / max(float(max(num_nodes, 1)) ** 0.5, 1.0), _MIN_DISTANCE)


def _repulsive_force_full(positions: torch.Tensor, ideal_distance: float) -> torch.Tensor:
    """Compute exact all-pairs repulsion.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    ideal_distance : float
        FR-style ideal distance constant ``k``.

    Returns
    -------
    torch.Tensor
        Repulsive force per node.
    """
    delta = positions.unsqueeze(1) - positions.unsqueeze(0)
    distances = torch.linalg.norm(delta, dim=2).clamp(min=_MIN_DISTANCE)
    force = (ideal_distance * ideal_distance) / distances
    return (delta / distances.unsqueeze(2) * force.unsqueeze(2)).sum(dim=1)


def _repulsive_force_sampled(
    positions: torch.Tensor,
    seed: int,
    step: int,
    ideal_distance: float,
) -> torch.Tensor:
    """Approximate repulsion by sampled random neighbors.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    seed : int
        Base random seed.
    step : int
        Iteration index.
    ideal_distance : float
        FR-style ideal distance constant ``k``.

    Returns
    -------
    torch.Tensor
        Approximate repulsive force per node.
    """
    num_nodes = int(positions.shape[0])
    sample_size = min(num_nodes, _SAMPLED_REPULSION_NEIGHBORS)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + step + 1)
    sampled = torch.randint(
        0,
        num_nodes,
        (num_nodes, sample_size),
        generator=generator,
        dtype=torch.long,
    ).to(positions.device)
    neighbors = positions[sampled]
    delta = positions.unsqueeze(1) - neighbors
    distances = torch.linalg.norm(delta, dim=2).clamp(min=_MIN_DISTANCE)
    force = (ideal_distance * ideal_distance) / distances
    return (delta / distances.unsqueeze(2) * force.unsqueeze(2)).sum(dim=1)


def _repulsive_force(
    positions: torch.Tensor,
    seed: int,
    step: int,
    ideal_distance: float,
) -> torch.Tensor:
    """Dispatch between exact and sampled repulsion.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    seed : int
        Base random seed.
    step : int
        Iteration index.
    ideal_distance : float
        FR-style ideal distance constant ``k``.

    Returns
    -------
    torch.Tensor
        Repulsive force per node.
    """
    if positions.shape[0] > _FULL_REPULSION_LIMIT:
        return _repulsive_force_sampled(positions, seed, step, ideal_distance)
    return _repulsive_force_full(positions, ideal_distance)


def _attractive_force(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    ideal_distance: float,
) -> torch.Tensor:
    """Compute exact spring attraction along graph edges.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    ideal_distance : float
        GEM target edge length constant ``k``.

    Returns
    -------
    torch.Tensor
        Attractive force per node.
    """
    forces = torch.zeros_like(positions)
    if edge_index.numel() == 0:
        return forces

    src = edge_index[0].to(device=positions.device, dtype=torch.long)
    dst = edge_index[1].to(device=positions.device, dtype=torch.long)
    delta = positions[dst] - positions[src]
    distances = torch.linalg.norm(delta, dim=1).clamp(min=_MIN_DISTANCE)
    edge_force = delta * (distances / max(ideal_distance, _MIN_DISTANCE)).unsqueeze(1)
    forces.index_add_(0, src, edge_force)
    forces.index_add_(0, dst, -edge_force)
    return forces


def _rotate_impulse(
    impulse: torch.Tensor,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Apply GEM's symmetry-breaking random rotation to each impulse vector.

    Parameters
    ----------
    impulse : torch.Tensor
        Raw impulse tensor with shape ``[N, 2]``.
    generator : torch.Generator
        Deterministic generator seeded from the public layout seed.
    device : torch.device
        Target device for the rotated tensor.

    Returns
    -------
    torch.Tensor
        Rotated impulse tensor with shape ``[N, 2]``.
    """
    angles = (
        torch.rand((impulse.shape[0],), generator=generator, dtype=torch.float32) * 2.0 - 1.0
    ) * _PERTURBATION_MAX_ANGLE
    angles = angles.to(device=device)
    cos_angle = torch.cos(angles)
    sin_angle = torch.sin(angles)
    x_coord = impulse[:, 0]
    y_coord = impulse[:, 1]
    return torch.stack(
        [
            x_coord * cos_angle - y_coord * sin_angle,
            x_coord * sin_angle + y_coord * cos_angle,
        ],
        dim=1,
    )


def _update_temperatures(
    temperatures: torch.Tensor,
    direction: torch.Tensor,
    previous_impulse: torch.Tensor,
    extent: float,
) -> torch.Tensor:
    """Adjust per-node temperatures from impulse alignment.

    Parameters
    ----------
    temperatures : torch.Tensor
        Per-node temperature tensor with shape ``[N]``.
    direction : torch.Tensor
        Current normalized impulse directions, shape ``[N, 2]``.
    previous_impulse : torch.Tensor
        Previous unnormalized impulses, shape ``[N, 2]``.
    extent : float
        Drawing half-width used to cap temperature growth.

    Returns
    -------
    torch.Tensor
        Updated temperature tensor with shape ``[N]``.

    Notes
    -----
    The paper updates temperature multiplicatively only for strong alignment
    or reversal. Neutral moves keep the current temperature instead of
    applying additional damping.
    """
    previous_norm = torch.linalg.norm(previous_impulse, dim=1, keepdim=True).clamp(
        min=_MIN_DISTANCE
    )
    cosine = (direction * (previous_impulse / previous_norm)).sum(dim=1)

    same_direction = cosine > 0.5
    opposite_direction = cosine < -0.2
    temperatures = torch.where(
        same_direction,
        temperatures * _TEMPERATURE_GROWTH_FACTOR,
        temperatures,
    )
    temperatures = torch.where(
        opposite_direction,
        temperatures * _TEMPERATURE_SHRINK_FACTOR,
        temperatures,
    )
    temperatures = temperatures.clamp(min=0.01, max=extent * 0.25)
    return temperatures


def layout_gem(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    max_iters: int = 500,
    seed: int = 42,
) -> torch.Tensor:
    """Lay out a graph with a GEM-style adaptive force simulation.

    Reference
    ---------
    Frick, Ludwig, and Mehldau, "A Fast Adaptive Layout Algorithm for
    Undirected Graphs" (1995).

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Optional node sizes used only for final scaling.
    max_iters : int, default=500
        Number of simulation iterations.
    seed : int, default=42
        Random seed for initialization and perturbations.

    Returns
    -------
    torch.Tensor
        Node positions with shape ``[N, 2]``.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if max_iters < 0:
        raise ValueError("max_iters must be non-negative.")

    device = _layout_device(edge_index, node_sizes)
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32, device=device)

    extent = _layout_extent(num_nodes, node_sizes)
    ideal_distance = _ideal_distance(num_nodes, extent)
    positions = _initialize_positions(num_nodes, device, seed)
    temperatures = torch.full(
        (num_nodes,),
        fill_value=max(extent / max(num_nodes, 1) ** 0.5, 0.05),
        dtype=torch.float32,
        device=device,
    )
    previous_impulse = torch.zeros_like(positions)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    for step in range(max_iters):
        repulsive = 0.08 * _repulsive_force(positions, seed, step, ideal_distance)
        attractive = -0.02 * _attractive_force(positions, edge_index, ideal_distance)
        barycenter = positions.mean(dim=0, keepdim=True)
        gravity = -0.01 * (positions - barycenter)
        impulse = _rotate_impulse(
            repulsive + attractive + gravity,
            generator=generator,
            device=device,
        )
        norm = torch.linalg.norm(impulse, dim=1, keepdim=True).clamp(min=_MIN_DISTANCE)
        direction = impulse / norm

        temperatures = _update_temperatures(
            temperatures=temperatures,
            direction=direction,
            previous_impulse=previous_impulse,
            extent=extent,
        )

        positions = positions + direction * temperatures.unsqueeze(1)
        previous_impulse = impulse

    return _normalize_positions(positions, extent).to(dtype=torch.float32, device=device)
