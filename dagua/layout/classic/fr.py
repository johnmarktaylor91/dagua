"""Fruchterman-Reingold layout translated from NetworkX's dense solver.

The implementation mirrors ``networkx.drawing.layout._fruchterman_reingold``
line by line, while preserving Dagua's existing public signature and optional
trace output.
"""

from __future__ import annotations

from math import sqrt
from typing import Optional, Union

import numpy as np
import torch

MIN_DISTANCE = 0.01
CONVERGENCE_THRESHOLD = 1.0e-4
OUTPUT_SCALE_FACTOR = 50.0


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
        Output device for the final position tensor.
    """
    if edge_index.numel() > 0:
        return edge_index.device
    if node_sizes is not None:
        return node_sizes.device
    return torch.device("cpu")


def _initialize_positions(num_nodes: int, seed: int) -> torch.Tensor:
    """Create the exact NetworkX random initialization in the unit square.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    seed : int
        Random seed for the initial placement.

    Returns
    -------
    torch.Tensor
        Initial positions with shape ``[N, 2]`` and dtype ``float64``.
    """
    random_state = np.random.RandomState(seed)
    positions = np.asarray(random_state.rand(num_nodes, 2), dtype=np.float64)
    return torch.from_numpy(positions)


def _adjacency_matrix(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build the directed dense adjacency matrix used by NetworkX FR.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Dense adjacency matrix with shape ``[N, N]`` and dtype ``float64``.

    Raises
    ------
    ValueError
        If ``edge_index`` has an invalid shape or contains out-of-range nodes.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")

    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float64)
    if edge_index.numel() == 0:
        return adjacency

    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    sources = edge_index_cpu[0]
    targets = edge_index_cpu[1]
    if (
        torch.any(sources < 0)
        or torch.any(sources >= num_nodes)
        or torch.any(targets < 0)
        or torch.any(targets >= num_nodes)
    ):
        raise ValueError("edge_index contains a node index outside [0, num_nodes).")

    if edge_weights is not None:
        weights = edge_weights.detach().to(device="cpu", dtype=torch.float64)
        adjacency[sources, targets] = weights
    else:
        adjacency[sources, targets] = 1.0
    return adjacency


def _initial_temperature(positions: torch.Tensor) -> float:
    """Compute the initial temperature used by NetworkX FR.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    float
        Initial temperature.
    """
    if positions.shape[0] == 0:
        return 0.0

    x_extent = float((positions[:, 0].max() - positions[:, 0].min()).item())
    y_extent = float((positions[:, 1].max() - positions[:, 1].min()).item())
    return max(x_extent, y_extent) * 0.1


def _rescale_layout(positions: torch.Tensor, scale: float) -> torch.Tensor:
    """Center and scale coordinates like ``networkx.rescale_layout``.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    scale : float
        Target half-width after rescaling.

    Returns
    -------
    torch.Tensor
        Rescaled position tensor with shape ``[N, 2]``.
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
    pos: Optional[torch.Tensor] = None,
    edge_weights: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, list[torch.Tensor]]]:
    """Lay out a graph with the NetworkX Fruchterman-Reingold update.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Unused, accepted for interface compatibility.
    steps : int, default=50
        Maximum number of FR iterations.
    seed : int, default=42
        Random seed used for the unit-square initialization.
    area : float, optional
        Accepted for interface compatibility. The NetworkX reference path uses
        a unit domain, so this value does not affect the translated solver.
    trace_every : int, default=0
        If greater than zero, record snapshots every ``trace_every`` updates.
    pos : torch.Tensor, optional
        Initial positions with shape ``[N, 2]``. When provided, replaces the
        random unit-square initialization. Must have dtype ``float64``.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``. When provided, weights
        scale the FR attraction term through the dense adjacency matrix.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, list[torch.Tensor]]
        Final positions with shape ``[N, 2]``. When tracing is enabled,
        returns ``(positions, traces)``.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``steps``, ``trace_every``, or ``area`` are invalid.
    """
    _ = node_sizes

    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if trace_every < 0:
        raise ValueError("trace_every must be non-negative.")
    if area is not None and area <= 0.0:
        raise ValueError("area must be positive.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
            )

    device = _layout_device(edge_index=edge_index, node_sizes=node_sizes)
    traces: list[torch.Tensor] = []
    if num_nodes == 0:
        empty = torch.empty((0, 2), dtype=torch.float32, device=device)
        return (empty, traces) if trace_every > 0 else empty

    if pos is not None:
        if pos.shape != (num_nodes, 2):
            raise ValueError(f"pos must have shape ({num_nodes}, 2), got {tuple(pos.shape)}")
        positions = pos.to(dtype=torch.float64, device="cpu").clone()
    else:
        positions = _initialize_positions(num_nodes=num_nodes, seed=seed)
    adjacency = _adjacency_matrix(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weights=edge_weights,
    )
    optimal_distance = sqrt(1.0 / float(max(num_nodes, 1)))
    temperature = _initial_temperature(positions)
    cooling_step = temperature / float(steps + 1) if steps > 0 else 0.0

    for iteration in range(steps):
        delta = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        distance = torch.linalg.norm(delta, dim=-1)
        distance = torch.clamp(distance, min=MIN_DISTANCE)
        displacement = torch.einsum(
            "ijk,ij->ik",
            delta,
            (optimal_distance * optimal_distance / distance.square())
            - (adjacency * distance / optimal_distance),
        )
        length = torch.linalg.norm(displacement, dim=-1)
        length = torch.clamp(length, min=MIN_DISTANCE)
        delta_pos = torch.einsum("ij,i->ij", displacement, temperature / length)
        positions = positions + delta_pos

        if trace_every > 0 and iteration % trace_every == 0:
            traces.append(positions.to(dtype=torch.float32, device=device).clone())

        temperature -= cooling_step
        if (torch.linalg.norm(delta_pos) / num_nodes) < CONVERGENCE_THRESHOLD:
            break

    scaled = _rescale_layout(positions, scale=1.0)
    scaled = scaled * (sqrt(float(max(num_nodes, 1))) * OUTPUT_SCALE_FACTOR)
    final_positions = scaled.to(dtype=torch.float32, device=device)

    if trace_every > 0:
        return final_positions, traces
    return final_positions
