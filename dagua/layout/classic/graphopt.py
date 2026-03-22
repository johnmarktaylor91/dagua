"""igraph GraphOpt translated into a small pure-PyTorch solver.

The implementation follows the igraph C routine closely: synchronous Coulomb
repulsion across all nearby pairs, Hooke-style spring forces on edges, and a
hard per-axis movement clamp instead of temperature cooling.
"""

from __future__ import annotations

import random
from typing import Optional

import torch

_COULOMBS_CONSTANT = 8_987_500_000.0
_MIN_DISTANCE = 1.0e-12
_MAX_REPULSION_DISTANCE = 500.0


def _layout_device(
    edge_index: torch.Tensor,
    node_sizes: Optional[torch.Tensor],
) -> torch.device:
    """Choose the device used for the returned tensor.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor.

    Returns
    -------
    torch.device
        Output device for the final layout tensor.
    """
    if edge_index.numel() > 0:
        return edge_index.device
    if node_sizes is not None:
        return node_sizes.device
    return torch.device("cpu")


def _validate_inputs(
    edge_index: torch.Tensor,
    num_nodes: int,
    niter: int,
    node_mass: float,
    max_sa_movement: float,
    edge_weights: Optional[torch.Tensor],
) -> None:
    """Validate the public GraphOpt arguments.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    niter : int
        Number of layout iterations.
    node_mass : float
        Shared node mass used to scale movement.
    max_sa_movement : float
        Maximum movement per axis per iteration.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.

    Returns
    -------
    None
        Raises ``ValueError`` when an input is invalid.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if niter < 0:
        raise ValueError("niter must be non-negative.")
    if node_mass <= 0.0:
        raise ValueError("node_mass must be positive.")
    if max_sa_movement < 0.0:
        raise ValueError("max_sa_movement must be non-negative.")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
            )

    if edge_index.numel() == 0:
        return

    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    min_index = int(edge_index_cpu.min().item())
    max_index = int(edge_index_cpu.max().item())
    if min_index < 0:
        raise ValueError("edge_index cannot contain negative node indices.")
    if max_index >= num_nodes:
        raise ValueError("edge_index contains node indices outside [0, num_nodes).")


def _initialize_positions(num_nodes: int, seed: int) -> torch.Tensor:
    """Create GraphOpt's random starting coordinates.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    seed : int
        Random seed for the initial placement.

    Returns
    -------
    torch.Tensor
        Initial positions with shape ``[N, 2]`` and dtype ``float64``.
    """
    rng = random.Random(seed)
    data = [[rng.random(), rng.random()] for _ in range(num_nodes)]
    return torch.tensor(data, dtype=torch.float64)


def _spring_edges(
    edge_index: torch.Tensor,
    edge_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return the non-self-loop springs exactly as encoded in ``edge_index``.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.

    Returns
    -------
    tuple[torch.Tensor, Optional[torch.Tensor]]
        Spring edge tensor with shape ``[2, E_clean]`` and the matching
        filtered edge weights when provided.

    Notes
    -----
    igraph GraphOpt treats every edge occurrence as its own spring. Parallel
    edges and reciprocal directed edges therefore contribute multiple spring
    forces and must not be deduplicated here.
    """
    if edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long), None

    edges = edge_index.to(device="cpu", dtype=torch.long)
    non_self = edges[0] != edges[1]
    if not bool(non_self.any().item()):
        return torch.empty((2, 0), dtype=torch.long), None

    filtered_edges = edges[:, non_self].contiguous()
    filtered_weights = None
    if edge_weights is not None:
        filtered_weights = edge_weights.detach().to(device="cpu", dtype=torch.float64)[non_self]
    return filtered_edges, filtered_weights.contiguous() if filtered_weights is not None else None


def layout_graphopt(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: int = 42,
    niter: int = 500,
    node_charge: float = 0.001,
    node_mass: float = 30.0,
    spring_length: float = 0.0,
    spring_constant: float = 1.0,
    max_sa_movement: float = 5.0,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Lay out a graph with the igraph GraphOpt algorithm.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Unused placeholder kept for interface compatibility.
    seed : int, default=42
        Random seed for the initial random layout.
    niter : int, default=500
        Number of GraphOpt iterations to execute.
    node_charge : float, default=0.001
        Charge used by the Coulomb repulsion term.
    node_mass : float, default=30.0
        Shared mass used to convert force into motion.
    spring_length : float, default=0.0
        Rest length of edge springs.
    spring_constant : float, default=1.0
        Hooke-law spring constant.
    max_sa_movement : float, default=5.0
        Maximum per-axis movement allowed in one iteration.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]`` that scales spring
        magnitudes for matching edge occurrences.

    Returns
    -------
    torch.Tensor
        Final positions with shape ``[N, 2]`` and dtype ``float32``.
    """
    _validate_inputs(
        edge_index=edge_index,
        num_nodes=num_nodes,
        niter=niter,
        node_mass=node_mass,
        max_sa_movement=max_sa_movement,
        edge_weights=edge_weights,
    )
    device = _layout_device(edge_index=edge_index, node_sizes=node_sizes)
    del node_sizes
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)

    positions = _initialize_positions(num_nodes=num_nodes, seed=seed)
    spring_edges, spring_weights = _spring_edges(
        edge_index=edge_index,
        edge_weights=edge_weights,
    )
    pair_source, pair_target = torch.triu_indices(num_nodes, num_nodes, offset=1)
    max_repulsion_distance_sq = _MAX_REPULSION_DISTANCE * _MAX_REPULSION_DISTANCE

    for _ in range(niter):
        forces = torch.zeros((num_nodes, 2), dtype=torch.float64)

        if node_charge != 0.0 and num_nodes > 1:
            delta = positions[pair_source] - positions[pair_target]
            distance_sq = delta.square().sum(dim=1)
            mask = (distance_sq > _MIN_DISTANCE) & (distance_sq < max_repulsion_distance_sq)
            if bool(mask.any().item()):
                pair_delta = delta[mask]
                pair_distance_sq = distance_sq[mask]
                pair_distance = torch.sqrt(pair_distance_sq)
                direction = pair_delta / pair_distance.unsqueeze(1)
                magnitude = _COULOMBS_CONSTANT * (node_charge * node_charge) / pair_distance_sq
                contribution = direction * magnitude.unsqueeze(1)
                forces.index_add_(0, pair_source[mask], contribution)
                forces.index_add_(0, pair_target[mask], -contribution)

        if spring_edges.numel() > 0:
            source = spring_edges[0]
            target = spring_edges[1]
            delta = positions[source] - positions[target]
            distance = torch.linalg.norm(delta, dim=1)
            mask = distance > _MIN_DISTANCE
            if bool(mask.any().item()):
                masked_distance = distance[mask]
                direction = delta[mask] / masked_distance.unsqueeze(1)
                stretch = (masked_distance - spring_length).abs()
                magnitude = 0.5 * spring_constant * stretch
                if spring_weights is not None:
                    magnitude = magnitude * spring_weights[mask].to(dtype=magnitude.dtype)
                source_sign = torch.where(
                    masked_distance > spring_length,
                    torch.full_like(masked_distance, -1.0),
                    torch.full_like(masked_distance, 1.0),
                )
                contribution = direction * (magnitude * source_sign).unsqueeze(1)
                forces.index_add_(0, source[mask], contribution)
                forces.index_add_(0, target[mask], -contribution)

        movement = torch.clamp(forces / node_mass, min=-max_sa_movement, max=max_sa_movement)
        positions = positions + movement

    return positions.to(dtype=torch.float32, device=device)
