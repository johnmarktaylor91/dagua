"""Pivot MDS graph layout based on landmark shortest-path distances.

The implementation follows the classical pivot MDS workflow: select landmark
nodes with a max-min heuristic, run BFS from each pivot, double-center the
squared distance matrix, and recover a rank-2 embedding with SVD.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import torch

_MIN_SPAN = 1.0e-6


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
        Device for the returned position tensor.
    """
    if edge_index.numel() > 0:
        return edge_index.device
    if node_sizes is not None:
        return node_sizes.device
    return torch.device("cpu")


def _layout_extent(num_nodes: int, node_sizes: Optional[torch.Tensor]) -> float:
    """Estimate a drawing extent from the graph size and node sizes.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor.

    Returns
    -------
    float
        Target half-width after normalization.
    """
    if node_sizes is None or node_sizes.numel() == 0:
        return max(float(max(num_nodes, 1)) ** 0.5 * 5.0, 1.0)

    max_size = float(node_sizes.to(dtype=torch.float32, device="cpu").max().item())
    return max(max_size * max(float(max(num_nodes, 1)) ** 0.5, 1.0) * 2.0, 1.0)


def _normalize_positions(positions: torch.Tensor, extent: float) -> torch.Tensor:
    """Center and scale coordinates into a stable bounding box.

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
    span = float(centered.abs().max().item())
    if span < _MIN_SPAN:
        centered = centered.clone()
        centered[:, 0] = torch.linspace(
            -1.0, 1.0, steps=positions.shape[0], device=positions.device
        )
        span = float(centered.abs().max().item())
    return centered * (extent / max(span, _MIN_SPAN))


def _build_undirected_adjacency(edge_index: torch.Tensor, num_nodes: int) -> list[list[int]]:
    """Build an undirected adjacency list.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    list[list[int]]
        One neighbor list per node.

    Raises
    ------
    ValueError
        If the edge tensor shape or node ids are invalid.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")

    adjacency_sets = [set() for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return [[] for _ in range(num_nodes)]

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
        Float distances with unreachable nodes replaced by ``diameter + 1``.
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


def _select_pivots(
    adjacency: list[list[int]],
    n_pivots: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select pivots with a deterministic max-min heuristic.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency list.
    n_pivots : int
        Number of pivots to choose.
    seed : int
        Random seed for the first pivot.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Pivot indices ``[P]`` and pivot-to-node distance matrix ``[P, N]``.
    """
    num_nodes = len(adjacency)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    selected = torch.zeros(num_nodes, dtype=torch.bool)
    pivot_indices: list[int] = []
    pivot_distances: list[torch.Tensor] = []

    first_pivot = int(torch.randint(0, num_nodes, (1,), generator=generator).item())
    pivot_indices.append(first_pivot)
    selected[first_pivot] = True

    min_distances = _bfs_distances(adjacency, first_pivot)
    pivot_distances.append(min_distances)

    while len(pivot_indices) < n_pivots:
        candidate_scores = min_distances.masked_fill(selected, -1.0)
        next_pivot = int(torch.argmax(candidate_scores).item())
        if selected[next_pivot]:
            break
        selected[next_pivot] = True
        pivot_indices.append(next_pivot)
        distances = _bfs_distances(adjacency, next_pivot)
        pivot_distances.append(distances)
        min_distances = torch.minimum(min_distances, distances)

    return torch.tensor(pivot_indices, dtype=torch.long), torch.stack(pivot_distances, dim=0)


def _pivot_mds_coordinates(distance_matrix: torch.Tensor) -> torch.Tensor:
    """Recover a 2D embedding from pivot distances with SVD.

    Parameters
    ----------
    distance_matrix : torch.Tensor
        Pivot-to-node distance matrix with shape ``[P, N]``.

    Returns
    -------
    torch.Tensor
        Coordinate matrix with shape ``[N, 2]``.
    """
    squared = distance_matrix.square()
    row_means = squared.mean(dim=1, keepdim=True)
    col_means = squared.mean(dim=0, keepdim=True)
    grand_mean = squared.mean()
    centered = -0.5 * (squared - row_means - col_means + grand_mean)

    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    coord_dims = min(2, int(singular_values.shape[0]))
    if coord_dims == 0:
        return torch.zeros((distance_matrix.shape[1], 2), dtype=torch.float32)

    scales = torch.sqrt(singular_values[:coord_dims].clamp_min(0.0))
    coords = vh[:coord_dims].transpose(0, 1) * scales.unsqueeze(0)
    if coord_dims == 1:
        zeros = torch.zeros((coords.shape[0], 1), dtype=coords.dtype)
        coords = torch.cat((coords, zeros), dim=1)
    return coords.to(dtype=torch.float32)


def layout_pivot_mds(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    n_pivots: int = 50,
    seed: int = 42,
) -> torch.Tensor:
    """Lay out a graph with pivot multidimensional scaling.

    Reference
    ---------
    Brandes and Pich, "Eigensolver Methods for Progressive Multidimensional
    Scaling of Large Data" (2007); adapted to graph shortest-path landmarks.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Optional node sizes used only to scale the drawing.
    n_pivots : int, default=50
        Maximum number of pivots.
    seed : int, default=42
        Random seed for the first pivot.

    Returns
    -------
    torch.Tensor
        Node positions with shape ``[N, 2]``.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if n_pivots <= 0:
        raise ValueError("n_pivots must be positive.")

    device = _layout_device(edge_index, node_sizes)
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32, device=device)

    adjacency = _build_undirected_adjacency(edge_index, num_nodes)
    _, pivot_distances = _select_pivots(adjacency, min(n_pivots, num_nodes), seed)
    raw_positions = _pivot_mds_coordinates(pivot_distances)
    extent = _layout_extent(num_nodes, node_sizes)
    normalized = _normalize_positions(raw_positions.to(device), extent)
    return normalized.to(dtype=torch.float32, device=device)
