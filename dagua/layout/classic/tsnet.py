"""tsNET graph layout based on t-SNE over graph distances.

This implementation computes graph shortest-path distances, converts them into
high-dimensional affinities with per-node perplexity matching, and optimizes a
two-dimensional Student-t embedding with early exaggeration.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import torch

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


def _build_undirected_adjacency(edge_index: torch.Tensor, num_nodes: int) -> list[list[int]]:
    """Build an undirected adjacency list from an edge tensor.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    list[list[int]]
        One sorted neighbor list per node.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")

    adjacency_sets = [set() for _ in range(num_nodes)]
    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    for source, target in zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist()):
        if source < 0 or source >= num_nodes or target < 0 or target >= num_nodes:
            raise ValueError("edge_index contains a node index outside [0, num_nodes).")
        if source == target:
            continue
        adjacency_sets[source].add(target)
        adjacency_sets[target].add(source)

    return [sorted(neighbors) for neighbors in adjacency_sets]


def _bfs_distances(adjacency: list[list[int]], start: int) -> torch.Tensor:
    """Compute unweighted shortest-path distances from one source.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency list.
    start : int
        Source node index.

    Returns
    -------
    torch.Tensor
        Float distance vector with unreachable nodes set to ``diameter + 1``.
    """
    num_nodes = len(adjacency)
    distances = [-1] * num_nodes
    distances[start] = 0
    queue: deque[int] = deque([start])
    diameter = 0

    while queue:
        node = queue.popleft()
        next_distance = distances[node] + 1
        for neighbor in adjacency[node]:
            if distances[neighbor] == -1:
                distances[neighbor] = next_distance
                diameter = max(diameter, next_distance)
                queue.append(neighbor)

    fill_value = float(diameter + 1 if num_nodes > 1 else 0.0)
    return torch.tensor(
        [fill_value if distance < 0 else float(distance) for distance in distances],
        dtype=torch.float32,
    )


def _all_pairs_shortest_paths(adjacency: list[list[int]]) -> torch.Tensor:
    """Compute the full shortest-path matrix with repeated BFS.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency list.

    Returns
    -------
    torch.Tensor
        Distance matrix with shape ``[N, N]``.
    """
    rows = [_bfs_distances(adjacency, node) for node in range(len(adjacency))]
    return torch.stack(rows, dim=0)


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


def layout_tsnet(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    perplexity: float = 30,
    steps: int = 1000,
    seed: int = 42,
) -> torch.Tensor:
    """Lay out a graph with a t-SNE objective over shortest-path distances.

    Reference
    ---------
    Maaten and Hinton, "Visualizing Data using t-SNE" (2008); adapted here to
    graph shortest-path distances as in tsNET-style graph drawing.

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
        Number of Adam updates.
    seed : int, default=42
        Random seed.

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

    adjacency = _build_undirected_adjacency(edge_index, num_nodes)
    distances = _all_pairs_shortest_paths(adjacency)
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
    ).to(device)
    positions = initial_positions.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([positions], lr=0.25)

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        exaggeration = 4.0 if step < min(250, steps // 2) else 1.0
        loss = _tsne_loss(positions, probabilities * exaggeration)
        loss.backward()
        optimizer.step()
        optimizer.param_groups[0]["lr"] = 0.25 * (0.98 ** (step / 50.0))

    extent = _layout_extent(num_nodes, node_sizes)
    return _normalize_positions(positions.detach(), extent).to(dtype=torch.float32, device=device)
