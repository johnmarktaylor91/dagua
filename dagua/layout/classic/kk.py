"""Kamada-Kawai spring layout based on graph-theoretic distances.

Every pair of nodes is connected by a spring whose ideal length is proportional
to their shortest-path distance. The layout minimizes total stress — the squared
difference between geometric and graph-theoretic distances.

Reference: Kamada & Kawai, "An Algorithm for Drawing General Undirected Graphs"
(1989), Information Processing Letters.
"""

from __future__ import annotations

from collections import deque
from typing import Optional, Union

import torch

_PIVOT_THRESHOLD = 5000
_PIVOT_COUNT = 200


def _build_undirected_adjacency(edge_index: torch.Tensor, num_nodes: int) -> list[list[int]]:
    """Build an undirected adjacency list from a ``[2, E]`` edge tensor.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    list[list[int]]
        Adjacency list with one neighbor list per node.

    Raises
    ------
    ValueError
        If ``edge_index`` has an invalid shape or references an out-of-range node.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")

    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return adjacency

    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    sources = edge_index_cpu[0].tolist()
    targets = edge_index_cpu[1].tolist()

    for source, target in zip(sources, targets):
        if source < 0 or source >= num_nodes or target < 0 or target >= num_nodes:
            raise ValueError("edge_index contains a node index outside [0, num_nodes).")
        adjacency[source].append(target)
        if source != target:
            adjacency[target].append(source)

    return adjacency


def _bfs_distances(adjacency: list[list[int]], start: int) -> list[int]:
    """Compute unweighted shortest-path distances from one source node.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency list.
    start : int
        Source node index.

    Returns
    -------
    list[int]
        Distances from ``start`` to every node, with ``-1`` for unreachable nodes.
    """
    num_nodes = len(adjacency)
    distances = [-1] * num_nodes
    distances[start] = 0
    frontier: deque[int] = deque([start])

    while frontier:
        node = frontier.popleft()
        next_distance = distances[node] + 1
        for neighbor in adjacency[node]:
            if distances[neighbor] == -1:
                distances[neighbor] = next_distance
                frontier.append(neighbor)

    return distances


def _fill_unreachable_distances(distances: torch.Tensor, diameter: int) -> torch.Tensor:
    """Replace unreachable entries with ``diameter + 1``.

    Parameters
    ----------
    distances : torch.Tensor
        Distance tensor containing ``-1`` for unreachable entries.
    diameter : int
        Maximum finite shortest-path distance observed so far.

    Returns
    -------
    torch.Tensor
        Distance tensor with all entries finite.
    """
    fill_value = float(diameter + 1 if distances.shape[0] > 1 else 0)
    return torch.where(distances >= 0, distances, torch.full_like(distances, fill_value))


def _compute_all_pairs_shortest_paths(adjacency: list[list[int]]) -> torch.Tensor:
    """Compute the full all-pairs shortest-path matrix with BFS.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency list.

    Returns
    -------
    torch.Tensor
        Shortest-path distances with shape ``[N, N]`` and dtype ``float32``.
    """
    num_nodes = len(adjacency)
    rows: list[list[int]] = []
    diameter = 0

    for node in range(num_nodes):
        distances = _bfs_distances(adjacency, node)
        finite_distances = [distance for distance in distances if distance >= 0]
        if finite_distances:
            diameter = max(diameter, max(finite_distances))
        rows.append(distances)

    distance_matrix = torch.tensor(rows, dtype=torch.float32)
    return _fill_unreachable_distances(distance_matrix, diameter)


def _sample_pivot_distances(
    adjacency: list[list[int]],
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate all-pairs distances with BFS from sampled pivot nodes.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency list.
    seed : int
        Random seed for deterministic pivot sampling.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Pivot indices with shape ``[P]`` and pivot distances with shape ``[N, P]``.
    """
    num_nodes = len(adjacency)
    pivot_count = min(_PIVOT_COUNT, num_nodes)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    pivot_indices = torch.randperm(num_nodes, generator=generator)[:pivot_count]

    rows: list[list[int]] = []
    diameter = 0
    for pivot in pivot_indices.tolist():
        distances = _bfs_distances(adjacency, pivot)
        finite_distances = [distance for distance in distances if distance >= 0]
        if finite_distances:
            diameter = max(diameter, max(finite_distances))
        rows.append(distances)

    pivot_distances = torch.tensor(rows, dtype=torch.float32).transpose(0, 1).contiguous()
    return pivot_indices.to(dtype=torch.long), _fill_unreachable_distances(
        pivot_distances, diameter
    )


def _ideal_edge_length(num_nodes: int) -> float:
    """Compute the base target length for graph-theoretic distance one.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    float
        Ideal length multiplier ``L0``.
    """
    area = float(max(num_nodes, 1))
    return float((area / max(num_nodes, 1)) ** 0.5)


def _target_lengths_and_strengths(distances: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert graph distances into KK target lengths and spring strengths.

    Parameters
    ----------
    distances : torch.Tensor
        Shortest-path distances with shape ``[N, N]`` or ``[N, P]``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Target lengths ``L`` and spring strengths ``K``.
    """
    lengths = distances * _ideal_edge_length(int(distances.shape[0]))
    safe_distances = torch.where(distances > 0, distances, torch.ones_like(distances))
    strengths = torch.where(
        distances > 0,
        1.0 / safe_distances.square(),
        torch.zeros_like(distances),
    )
    return lengths, strengths


def _initialize_positions(num_nodes: int, seed: int) -> torch.Tensor:
    """Create deterministic random initial positions.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    seed : int
        Random seed for the initial placement.

    Returns
    -------
    torch.Tensor
        Initial positions with shape ``[N, 2]``.
    """
    if num_nodes <= 1:
        return torch.zeros((num_nodes, 2), dtype=torch.float32)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn((num_nodes, 2), generator=generator, dtype=torch.float32)


def layout_kk(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 500,
    seed: int = 42,
    trace_every: int = 0,
) -> Union[torch.Tensor, tuple[torch.Tensor, list[torch.Tensor]]]:
    """Run Kamada-Kawai stress minimization layout.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list, shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Unused, accepted for interface compatibility.
    steps : int
        Number of optimization steps.
    seed : int
        Random seed for initial placement.
    trace_every : int
        If > 0, record snapshots every N steps.

    Returns
    -------
    torch.Tensor or tuple
        Final positions ``[N, 2]``, or ``(positions, traces)`` if tracing.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``steps``, or ``trace_every`` are invalid.
    """
    del node_sizes

    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if trace_every < 0:
        raise ValueError("trace_every must be non-negative.")

    if num_nodes == 0:
        empty_positions = torch.empty((0, 2), dtype=torch.float32)
        return (empty_positions, []) if trace_every > 0 else empty_positions

    adjacency = _build_undirected_adjacency(edge_index=edge_index, num_nodes=num_nodes)
    use_pivots = num_nodes > _PIVOT_THRESHOLD

    if use_pivots:
        pivot_indices, pivot_distances = _sample_pivot_distances(adjacency=adjacency, seed=seed)
        target_lengths, spring_strengths = _target_lengths_and_strengths(pivot_distances)
    else:
        pivot_indices = torch.empty(0, dtype=torch.long)
        distance_matrix = _compute_all_pairs_shortest_paths(adjacency)
        target_lengths, spring_strengths = _target_lengths_and_strengths(distance_matrix)

    positions = torch.nn.Parameter(_initialize_positions(num_nodes=num_nodes, seed=seed))
    optimizer = torch.optim.Adam([positions], lr=1.0)
    traces: list[torch.Tensor] = []

    for step in range(steps):
        optimizer.zero_grad()
        if use_pivots:
            pivot_positions = positions[pivot_indices]
            pairwise_distances = torch.cdist(positions, pivot_positions)
            stress = (spring_strengths * (pairwise_distances - target_lengths).square()).sum()
        else:
            pairwise_distances = torch.cdist(positions, positions)
            weighted_error = spring_strengths * (pairwise_distances - target_lengths).square()
            stress = torch.triu(weighted_error, diagonal=1).sum()
        stress.backward()
        optimizer.step()

        if trace_every > 0 and (step + 1) % trace_every == 0:
            traces.append(positions.detach().clone())

    final_positions = positions.detach()
    return (final_positions, traces) if trace_every > 0 else final_positions
