"""Cycle detection and acyclic transformation for graph layout.

Provides DFS-based back-edge detection and edge reversal so the layout
engine always operates on a DAG. After layout, original directions are
restored — the edge router already handles upward edges with wide arcs.
"""

from __future__ import annotations

import torch


def detect_back_edges(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Detect back edges via iterative DFS with WHITE/GRAY/BLACK coloring.

    Args:
        edge_index: [2, E] LongTensor of (src, tgt) pairs.
        num_nodes: total number of nodes.

    Returns:
        BoolTensor of shape [E] — True for back edges (including self-loops).
    """
    num_edges = edge_index.shape[1] if edge_index.numel() > 0 else 0
    if num_edges == 0:
        return torch.zeros(0, dtype=torch.bool)

    src = edge_index[0].tolist()
    tgt = edge_index[1].tolist()

    # Build adjacency: node -> list of (target_node, edge_index)
    adj: list[list[tuple[int, int]]] = [[] for _ in range(num_nodes)]
    for ei in range(num_edges):
        adj[src[ei]].append((tgt[ei], ei))

    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * num_nodes
    is_back = [False] * num_edges

    # Compute in-degrees to start from sources first
    in_degree = [0] * num_nodes
    for ei in range(num_edges):
        in_degree[tgt[ei]] += 1

    # Visit order: sources (in-degree 0) first, then remaining
    visit_order = [n for n in range(num_nodes) if in_degree[n] == 0]
    visit_order.extend(n for n in range(num_nodes) if in_degree[n] > 0)

    for start in visit_order:
        if color[start] != WHITE:
            continue

        # Iterative DFS: stack holds (node, adj_iterator_index)
        stack: list[tuple[int, int]] = [(start, 0)]
        color[start] = GRAY

        while stack:
            node, idx = stack[-1]
            if idx < len(adj[node]):
                stack[-1] = (node, idx + 1)
                child, ei = adj[node][idx]
                if color[child] == GRAY:
                    is_back[ei] = True
                elif color[child] == WHITE:
                    color[child] = GRAY
                    stack.append((child, 0))
            else:
                color[node] = BLACK
                stack.pop()

    return torch.tensor(is_back, dtype=torch.bool)


def make_acyclic(edge_index: torch.Tensor, back_edge_mask: torch.Tensor) -> torch.Tensor:
    """Return a new edge_index with back edges reversed (src/tgt swapped).

    Args:
        edge_index: [2, E] original edges.
        back_edge_mask: [E] bool mask of back edges.

    Returns:
        [2, E] tensor with back edges reversed.
    """
    result = edge_index.clone()
    if back_edge_mask.any():
        result[0, back_edge_mask], result[1, back_edge_mask] = (
            edge_index[1, back_edge_mask],
            edge_index[0, back_edge_mask],
        )
    return result
