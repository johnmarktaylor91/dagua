"""Deterministic tidy tree layout via a Reingold-Tilford style traversal."""

from __future__ import annotations

import sys
from collections import deque
from typing import Optional

import torch

from dagua.layout.classic._graph_distances import (
    build_undirected_adjacency as _shared_build_undirected_adjacency,
)


def _layout_device(
    edge_index: torch.Tensor,
    node_sizes: Optional[torch.Tensor],
) -> torch.device:
    """Choose the device used for the returned layout tensor.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``.

    Returns
    -------
    torch.device
        Device for the final output tensor.
    """
    if edge_index.numel() > 0:
        return edge_index.device
    if node_sizes is not None:
        return node_sizes.device
    return torch.device("cpu")


def _node_spacing(
    node_sizes: Optional[torch.Tensor],
    axis: int,
    default: float,
) -> float:
    """Estimate spacing along one axis from node sizes.

    Parameters
    ----------
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``.
    axis : int
        Axis index inside ``node_sizes``.
    default : float
        Fallback spacing when node sizes are unavailable.

    Returns
    -------
    float
        Spacing multiplier for the requested axis.
    """
    if node_sizes is None or node_sizes.numel() == 0:
        return default
    max_size = float(node_sizes.to(dtype=torch.float32, device="cpu")[:, axis].max().item())
    return max(max_size * 1.5, default)


def _root_candidates(edge_index: torch.Tensor, num_nodes: int) -> list[int]:
    """Rank root candidates with zero-indegree nodes first.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    list[int]
        Node indices in deterministic root preference order.
    """
    indegree = [0] * num_nodes
    if edge_index.numel() > 0:
        targets = edge_index.detach().to(device="cpu", dtype=torch.long)[1].tolist()
        for target in targets:
            indegree[target] += 1
    return sorted(
        range(num_nodes),
        key=lambda node_idx: (indegree[node_idx] != 0, indegree[node_idx], node_idx),
    )


def _bfs_forest(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> tuple[list[int], list[list[int]], list[int]]:
    """Extract a deterministic BFS forest from an arbitrary graph.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    tuple[list[int], list[list[int]], list[int]]
        Forest roots, child lists per node, and node depths.
    """
    adjacency = _shared_build_undirected_adjacency(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weights=None,
    )
    children: list[list[int]] = [[] for _ in range(num_nodes)]
    depths = [0] * num_nodes
    visited = [False] * num_nodes
    roots: list[int] = []

    for root in _root_candidates(edge_index=edge_index, num_nodes=num_nodes):
        if visited[root]:
            continue
        roots.append(root)
        visited[root] = True
        queue: deque[int] = deque([root])
        while queue:
            node = queue.popleft()
            for neighbor, _ in adjacency[node]:
                if visited[neighbor]:
                    continue
                visited[neighbor] = True
                depths[neighbor] = depths[node] + 1
                children[node].append(neighbor)
                queue.append(neighbor)

    return roots, children, depths


def _assign_preliminary_x(
    root_idx: int,
    children: list[list[int]],
    preliminary_x: list[float],
    next_leaf_x: list[float],
) -> tuple[float, float]:
    """Assign tidy x-coordinates using an iterative post-order traversal.

    Parameters
    ----------
    root_idx : int
        Root node of the subtree being laid out.
    children : list[list[int]]
        BFS-forest child lists.
    preliminary_x : list[float]
        Mutable x-coordinate buffer for all nodes.
    next_leaf_x : list[float]
        Single-item mutable counter tracking the next free leaf coordinate.

    Returns
    -------
    tuple[float, float]
        Inclusive ``(min_x, max_x)`` span of the root subtree.
    """
    subtree_spans: dict[int, tuple[float, float]] = {}
    for node_idx in _postorder_iterative(root_idx, children):
        if not children[node_idx]:
            preliminary_x[node_idx] = next_leaf_x[0]
            next_leaf_x[0] += 1.0
            subtree_spans[node_idx] = (preliminary_x[node_idx], preliminary_x[node_idx])
            continue

        first_child = children[node_idx][0]
        last_child = children[node_idx][-1]
        preliminary_x[node_idx] = 0.5 * (preliminary_x[first_child] + preliminary_x[last_child])
        subtree_spans[node_idx] = (
            subtree_spans[first_child][0],
            subtree_spans[last_child][1],
        )
    return subtree_spans[root_idx]


def _postorder_iterative(root: int, children: list[list[int]]) -> list[int]:
    """Return an iterative post-order traversal for one rooted subtree.

    Parameters
    ----------
    root : int
        Root node of the subtree.
    children : list[list[int]]
        Child lists indexed by node id.

    Returns
    -------
    list[int]
        Node ids in post-order.
    """
    stack: list[tuple[int, bool]] = [(root, False)]
    traversal: list[int] = []
    while stack:
        node_idx, processed = stack.pop()
        if processed:
            traversal.append(node_idx)
            continue
        stack.append((node_idx, True))
        for child_idx in reversed(children[node_idx]):
            stack.append((child_idx, False))
    return traversal


def layout_reingold_tilford(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: int = 42,
    horizontal: bool = False,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Lay out a graph as a tidy tree or BFS forest.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]`` used to choose stable
        layer and sibling spacing.
    seed : int, default=42
        Accepted for interface compatibility. This layout is deterministic.
    horizontal : bool, default=False
        If ``True``, rotate the final layout so depth grows along the x-axis.
    edge_weights : torch.Tensor, optional
        Accepted for interface compatibility and validated when provided.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If ``num_nodes`` is negative or ``edge_weights`` has the wrong shape.
    """
    _ = seed

    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
            )

    device = _layout_device(edge_index=edge_index, node_sizes=node_sizes)
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)
    sys.setrecursionlimit(max(sys.getrecursionlimit(), num_nodes * 2))

    roots, children, depths = _bfs_forest(edge_index=edge_index, num_nodes=num_nodes)
    sibling_spacing = _node_spacing(node_sizes=node_sizes, axis=0, default=1.0)
    layer_spacing = _node_spacing(node_sizes=node_sizes, axis=1, default=1.5)
    component_gap = sibling_spacing * 2.0

    preliminary_x = [0.0] * num_nodes
    next_leaf_x = [0.0]
    for root in roots:
        _assign_preliminary_x(root, children, preliminary_x, next_leaf_x)
        next_leaf_x[0] += component_gap

    positions = torch.zeros((num_nodes, 2), dtype=torch.float32)
    for node_idx in range(num_nodes):
        positions[node_idx, 0] = float(preliminary_x[node_idx]) * sibling_spacing
        positions[node_idx, 1] = float(depths[node_idx]) * layer_spacing

    positions -= positions.mean(dim=0, keepdim=True)
    if horizontal:
        positions = positions[:, [1, 0]]
    return positions.to(device=device)
