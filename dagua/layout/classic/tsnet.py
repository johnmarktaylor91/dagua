"""tsNET graph layout based on t-SNE over graph distances.

This implementation computes graph shortest-path distances, converts them into
high-dimensional affinities with per-node perplexity matching, and optimizes a
two-dimensional Student-t embedding with early exaggeration.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from dagua.layout.classic._graph_distances import (
    all_pairs_shortest_paths as _shared_all_pairs_shortest_paths,
)
from dagua.layout.classic._graph_distances import (
    build_undirected_adjacency as _shared_build_undirected_adjacency,
)
from dagua.layout.classic.pivot_mds import layout_pivot_mds

_MIN_DISTANCE = 1.0e-12


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
        Optional node sizes.

    Returns
    -------
    float
        Target half-width.
    """
    if node_sizes is None or node_sizes.numel() == 0:
        return max(float(max(num_nodes, 1)) ** 0.5 * 5.0, 1.0)

    max_size = float(node_sizes.to(dtype=torch.float32, device="cpu").max().item())
    return max(max_size * max(float(max(num_nodes, 1)) ** 0.5, 1.0) * 2.0, 1.0)


def _normalize_positions(positions: torch.Tensor, extent: float) -> torch.Tensor:
    """Center and scale coordinates into a stable drawing box.

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


def _build_undirected_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor] = None,
) -> list[list[tuple[int, float]]]:
    """Build an undirected adjacency list from an edge tensor.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    edge_weights : torch.Tensor, optional
        Optional per-edge weights with shape ``[E]``.

    Returns
    -------
    list[list[tuple[int, float]]]
        One sorted neighbor list per node.
    """
    return _shared_build_undirected_adjacency(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weights=edge_weights,
    )


def _all_pairs_shortest_paths(
    adjacency: list[list[tuple[int, float]]],
    weighted: bool,
) -> torch.Tensor:
    """Compute the full shortest-path matrix.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Undirected adjacency list.
    weighted : bool
        Whether to use Dijkstra instead of BFS.

    Returns
    -------
    torch.Tensor
        Distance matrix with shape ``[N, N]``.
    """
    raw_distances = _shared_all_pairs_shortest_paths(adjacency, weighted=weighted)
    if raw_distances.size == 0:
        return torch.empty((0, 0), dtype=torch.float32)

    cleaned = raw_distances.astype(np.float64, copy=True)
    for node in range(cleaned.shape[0]):
        row = cleaned[node]
        finite_mask = np.isfinite(row) if weighted else row >= 0
        max_distance = float(row[finite_mask].max()) if bool(finite_mask.any()) else 0.0
        fill_value = max_distance + 1.0 if cleaned.shape[0] > 1 else 0.0
        row[~finite_mask] = fill_value
    return torch.tensor(cleaned, dtype=torch.float32)


def _row_probabilities(distances: torch.Tensor, perplexity: float) -> torch.Tensor:
    """Match one row's Gaussian bandwidth to a target perplexity.

    Parameters
    ----------
    distances : torch.Tensor
        Row of graph distances with shape ``[N]``.
    perplexity : float
        Target perplexity.

    Returns
    -------
    torch.Tensor
        Conditional probability row with shape ``[N]``.
    """
    num_nodes = int(distances.shape[0])
    if num_nodes <= 1:
        return torch.zeros_like(distances)

    mask = torch.ones(num_nodes, dtype=torch.bool)
    mask[int(torch.argmin(distances).item())] = False
    squared = distances.square()

    beta = torch.tensor(1.0, dtype=torch.float32)
    beta_min = torch.tensor(float("-inf"), dtype=torch.float32)
    beta_max = torch.tensor(float("inf"), dtype=torch.float32)
    target_entropy = torch.log(torch.tensor(perplexity, dtype=torch.float32))

    probabilities = torch.zeros_like(distances)
    for _ in range(50):
        weights = torch.exp(-squared * beta) * mask.to(dtype=torch.float32)
        weights_sum = weights.sum().clamp(min=_MIN_DISTANCE)
        probabilities = weights / weights_sum
        entropy = -(probabilities[mask] * probabilities[mask].clamp(min=_MIN_DISTANCE).log()).sum()
        error = entropy - target_entropy
        if torch.abs(error) < 1.0e-4:
            break
        if error > 0:
            beta_min = beta
            beta = beta * 2.0 if torch.isinf(beta_max) else (beta + beta_max) * 0.5
        else:
            beta_max = beta
            beta = beta * 0.5 if torch.isinf(beta_min) else (beta + beta_min) * 0.5

    probabilities[int(torch.argmin(distances).item())] = 0.0
    return probabilities


def _high_dimensional_affinities(distance_matrix: torch.Tensor, perplexity: float) -> torch.Tensor:
    """Build the symmetric t-SNE input affinity matrix.

    Parameters
    ----------
    distance_matrix : torch.Tensor
        Shortest-path matrix with shape ``[N, N]``.
    perplexity : float
        Target perplexity.

    Returns
    -------
    torch.Tensor
        Symmetric probability matrix ``P`` with shape ``[N, N]``.
    """
    rows = [
        _row_probabilities(distance_matrix[node], perplexity)
        for node in range(distance_matrix.shape[0])
    ]
    conditional = torch.stack(rows, dim=0)
    symmetrized = (conditional + conditional.transpose(0, 1)) / (
        2.0 * max(distance_matrix.shape[0], 1)
    )
    return symmetrized.clamp(min=_MIN_DISTANCE)


def _tsne_loss(positions: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
    """Evaluate the exact t-SNE KL objective.

    Parameters
    ----------
    positions : torch.Tensor
        Current embedding with shape ``[N, 2]``.
    probabilities : torch.Tensor
        Symmetric input affinity matrix ``P``.

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    delta = positions.unsqueeze(1) - positions.unsqueeze(0)
    squared_distances = delta.square().sum(dim=2)
    numerators = (1.0 + squared_distances).reciprocal()
    diagonal_mask = ~torch.eye(
        positions.shape[0],
        dtype=torch.bool,
        device=positions.device,
    )
    numerators = numerators * diagonal_mask.to(dtype=numerators.dtype)
    q = numerators / numerators.sum().clamp(min=_MIN_DISTANCE)
    return (probabilities * (probabilities.log() - q.clamp(min=_MIN_DISTANCE).log())).sum()


def _gradient_descent_step(
    positions: torch.Tensor,
    grad: torch.Tensor,
    update: torch.Tensor,
    gains: torch.Tensor,
    learning_rate: float,
    momentum: float,
    min_gain: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one exact t-SNE gains-plus-momentum update.

    Parameters
    ----------
    positions : torch.Tensor
        Current embedding with shape ``[N, 2]``.
    grad : torch.Tensor
        Gradient tensor with shape ``[N, 2]``.
    update : torch.Tensor
        Velocity tensor with shape ``[N, 2]``.
    gains : torch.Tensor
        Per-parameter gain tensor with shape ``[N, 2]``.
    learning_rate : float
        Step size for the current optimization phase.
    momentum : float
        Momentum coefficient for the current optimization phase.
    min_gain : float
        Floor applied to every gain entry.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Updated ``(positions, update, gains)`` tensors.
    """
    inc = (update * grad) < 0.0
    dec = ~inc
    gains[inc] += 0.2
    gains[dec] *= 0.8
    gains.clamp_(min=min_gain)
    grad = grad * gains

    update = momentum * update - learning_rate * grad
    positions.add_(update)
    return positions, update, gains


def layout_tsnet(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    perplexity: float = 30,
    steps: int = 1000,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Lay out a graph with a tsNET-style t-SNE objective over graph distances.

    Reference
    ---------
    Kruiger et al., "Graph Layout by t-SNE" (2017). The optimization loop
    follows the original t-SNE gains-plus-momentum update rule rather than an
    adaptive optimizer.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Optional node sizes used only for final scaling.
    perplexity : float, default=30
        Target t-SNE perplexity.
    steps : int, default=1000
        Number of optimization updates.
    seed : int, default=42
        Random seed.
    edge_weights : torch.Tensor, optional
        Optional per-edge weights with shape ``[E]``. When provided, graph
        distances are computed with Dijkstra instead of BFS.

    Returns
    -------
    torch.Tensor
        Node positions with shape ``[N, 2]``.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if perplexity <= 0:
        raise ValueError("perplexity must be positive.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")

    device = _layout_device(edge_index, node_sizes)
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32, device=device)

    adjacency = _build_undirected_adjacency(edge_index, num_nodes, edge_weights=edge_weights)
    distances = _all_pairs_shortest_paths(adjacency, weighted=edge_weights is not None)
    probabilities = _high_dimensional_affinities(
        distances,
        min(perplexity, float(max(num_nodes - 1, 1))),
    ).to(device)

    initial_positions = layout_pivot_mds(
        edge_index,
        num_nodes,
        node_sizes=node_sizes,
        n_pivots=min(32, num_nodes),
        seed=seed,
        edge_weights=edge_weights,
    ).to(device)
    positions = initial_positions.clone().requires_grad_(True)
    early_exaggeration = 12.0
    early_exaggeration_steps = min(250, steps)
    min_gain = 0.01
    early_learning_rate = max(float(max(num_nodes, 1)) / 48.0, 200.0)
    late_learning_rate = early_learning_rate * 4.0
    update = torch.zeros_like(positions)
    gains = torch.ones_like(positions)

    for step in range(steps):
        exaggeration = early_exaggeration if step < early_exaggeration_steps else 1.0
        loss = _tsne_loss(positions, probabilities * exaggeration)
        loss.backward()

        grad = positions.grad.detach().clone()
        momentum = 0.5 if step < early_exaggeration_steps else 0.8
        learning_rate = (
            early_learning_rate if step < early_exaggeration_steps else late_learning_rate
        )
        with torch.no_grad():
            positions, update, gains = _gradient_descent_step(
                positions=positions,
                grad=grad,
                update=update,
                gains=gains,
                learning_rate=learning_rate,
                momentum=momentum,
                min_gain=min_gain,
            )
            positions.grad.zero_()

    extent = _layout_extent(num_nodes, node_sizes)
    return _normalize_positions(positions.detach(), extent).to(dtype=torch.float32, device=device)
