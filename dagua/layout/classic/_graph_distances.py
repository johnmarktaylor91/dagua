"""Shared graph-distance utilities for classic layout algorithms.

Provides BFS (unweighted) and Dijkstra (weighted) shortest-path computations
used by distance-based layout methods: Kamada-Kawai, stress-SGD, maxent-stress,
pivot-MDS, tsNET, and (SGD)^2.
"""

from __future__ import annotations

import heapq
from collections import deque
from typing import Optional

import numpy as np
import torch

UNREACHABLE: int = -1
UNREACHABLE_FLOAT: float = 1e6


def build_undirected_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor] = None,
) -> list[list[tuple[int, float]]]:
    """Build an undirected weighted adjacency list from edge_index.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    edge_weights : torch.Tensor, optional
        Per-edge weights with shape ``[E]``. Defaults to 1.0 for all edges.

    Returns
    -------
    list[list[tuple[int, float]]]
        Adjacency list where each entry is ``(neighbor, weight)``.
        Neighbors are sorted by index for determinism.
    """
    adjacency: list[dict[int, float]] = [{} for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return [[] for _ in range(num_nodes)]

    ei_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    sources = ei_cpu[0].tolist()
    targets = ei_cpu[1].tolist()

    if edge_weights is not None:
        weights = edge_weights.detach().cpu().float().tolist()
    else:
        weights = [1.0] * len(sources)

    for src, tgt, weight in zip(sources, targets, weights):
        if src == tgt:
            continue
        if tgt not in adjacency[src] or weight < adjacency[src][tgt]:
            adjacency[src][tgt] = weight
        if src not in adjacency[tgt] or weight < adjacency[tgt][src]:
            adjacency[tgt][src] = weight

    return [sorted(neighbors.items()) for neighbors in adjacency]


def build_directed_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor] = None,
) -> list[list[tuple[int, float]]]:
    """Build a directed weighted adjacency list from edge_index.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    edge_weights : torch.Tensor, optional
        Per-edge weights with shape ``[E]``. Defaults to 1.0 for all edges.

    Returns
    -------
    list[list[tuple[int, float]]]
        Directed adjacency list where each entry is ``(neighbor, weight)``.
    """
    adjacency: list[dict[int, float]] = [{} for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return [[] for _ in range(num_nodes)]

    ei_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    sources = ei_cpu[0].tolist()
    targets = ei_cpu[1].tolist()

    if edge_weights is not None:
        weights = edge_weights.detach().cpu().float().tolist()
    else:
        weights = [1.0] * len(sources)

    for src, tgt, weight in zip(sources, targets, weights):
        if tgt not in adjacency[src] or weight < adjacency[src][tgt]:
            adjacency[src][tgt] = weight

    return [sorted(neighbors.items()) for neighbors in adjacency]


def bfs_distances(
    adjacency: list[list[tuple[int, float]]],
    source: int,
) -> np.ndarray:
    """Compute unweighted shortest-path distances from one source via BFS.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Adjacency list from ``build_undirected_adjacency``.
    source : int
        Source node index.

    Returns
    -------
    np.ndarray
        Integer distances with shape ``[N]``. Unreachable nodes have value -1.
    """
    num_nodes = len(adjacency)
    distances = np.full(num_nodes, UNREACHABLE, dtype=np.int32)
    distances[source] = 0
    frontier: deque[int] = deque([source])

    while frontier:
        node = frontier.popleft()
        next_dist = int(distances[node]) + 1
        for neighbor, _ in adjacency[node]:
            if int(distances[neighbor]) != UNREACHABLE:
                continue
            distances[neighbor] = next_dist
            frontier.append(neighbor)

    return distances


def dijkstra_distances(
    adjacency: list[list[tuple[int, float]]],
    source: int,
) -> np.ndarray:
    """Compute weighted shortest-path distances from one source via Dijkstra.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Weighted adjacency list from ``build_undirected_adjacency``.
    source : int
        Source node index.

    Returns
    -------
    np.ndarray
        Float distances with shape ``[N]``. Unreachable nodes have value inf.
    """
    num_nodes = len(adjacency)
    distances = np.full(num_nodes, np.inf, dtype=np.float64)
    distances[source] = 0.0
    visited = np.zeros(num_nodes, dtype=np.bool_)
    heap: list[tuple[float, int]] = [(0.0, source)]

    while heap:
        dist, node = heapq.heappop(heap)
        if visited[node]:
            continue
        visited[node] = True
        for neighbor, weight in adjacency[node]:
            new_dist = dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(heap, (new_dist, neighbor))

    return distances


def all_pairs_shortest_paths(
    adjacency: list[list[tuple[int, float]]],
    weighted: bool = False,
) -> np.ndarray:
    """Compute all-pairs shortest-path distance matrix.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Adjacency list from ``build_undirected_adjacency``.
    weighted : bool, default=False
        Use Dijkstra (True) or BFS (False).

    Returns
    -------
    np.ndarray
        Distance matrix with shape ``[N, N]``.
        Unreachable pairs: -1 (unweighted) or inf (weighted).
    """
    num_nodes = len(adjacency)
    if weighted:
        distances = np.full((num_nodes, num_nodes), np.inf, dtype=np.float64)
        for source in range(num_nodes):
            distances[source] = dijkstra_distances(adjacency, source)
    else:
        distances = np.full((num_nodes, num_nodes), UNREACHABLE, dtype=np.int32)
        for source in range(num_nodes):
            distances[source] = bfs_distances(adjacency, source)
    return distances


def is_connected(adjacency: list[list[tuple[int, float]]]) -> bool:
    """Check whether the undirected graph is connected.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Undirected adjacency list.

    Returns
    -------
    bool
        True when every node is reachable from node 0.
    """
    if len(adjacency) <= 1:
        return True
    return bool(np.all(bfs_distances(adjacency, 0) >= 0))
