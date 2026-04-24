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


def _is_acyclic(edge_index: torch.Tensor, num_nodes: int) -> bool:
    """Return whether ``edge_index`` admits a topological ordering.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    bool
        ``True`` when Kahn's algorithm can process every node.
    """
    if num_nodes <= 1 or edge_index.numel() == 0:
        return True

    children: list[list[int]] = [[] for _ in range(num_nodes)]
    in_degree = [0] * num_nodes
    for source, target in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        children[source].append(target)
        in_degree[target] += 1

    ready = [node_idx for node_idx, degree in enumerate(in_degree) if degree == 0]
    processed = 0
    head = 0
    while head < len(ready):
        node_idx = ready[head]
        head += 1
        processed += 1
        for child_idx in children[node_idx]:
            in_degree[child_idx] -= 1
            if in_degree[child_idx] == 0:
                ready.append(child_idx)
    return processed == num_nodes


def _greedy_fas(edge_index: torch.Tensor, num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Reverse a greedy feedback-arc set to force an acyclic orientation.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(acyclic_edges, reversed_mask)`` where ``reversed_mask`` marks the
        input edges flipped to align with the greedy node order.
    """
    num_edges = int(edge_index.shape[1])
    if num_nodes <= 1 or num_edges == 0:
        return edge_index.clone(), torch.zeros((num_edges,), dtype=torch.bool)

    outgoing_edges: list[list[int]] = [[] for _ in range(num_nodes)]
    incoming_edges: list[list[int]] = [[] for _ in range(num_nodes)]
    sources = edge_index[0].tolist()
    targets = edge_index[1].tolist()
    for edge_idx, (source, target) in enumerate(zip(sources, targets)):
        outgoing_edges[source].append(edge_idx)
        incoming_edges[target].append(edge_idx)

    active_nodes = [True] * num_nodes
    active_edges = [True] * num_edges
    order: list[int] = []

    for _ in range(num_nodes):
        best_node = -1
        best_score: tuple[int, int] | None = None
        for node_idx in range(num_nodes):
            if not active_nodes[node_idx]:
                continue
            in_degree = sum(1 for edge_idx in incoming_edges[node_idx] if active_edges[edge_idx])
            out_degree = sum(1 for edge_idx in outgoing_edges[node_idx] if active_edges[edge_idx])
            # Eades-Lin-Smyth: pick source-like nodes (max out - in) first so
            # the final order goes source -> sink. Reversed mask then flips
            # only the edges that actually go against that order (the FAS).
            score = (out_degree - in_degree, -node_idx)
            if best_score is None or score > best_score:
                best_node = node_idx
                best_score = score

        if best_node < 0:
            break

        order.append(best_node)
        active_nodes[best_node] = False
        for edge_idx in incoming_edges[best_node]:
            active_edges[edge_idx] = False
        for edge_idx in outgoing_edges[best_node]:
            active_edges[edge_idx] = False

    order_index = {node_idx: order_pos for order_pos, node_idx in enumerate(order)}
    reversed_mask = torch.tensor(
        [order_index[source] > order_index[target] for source, target in zip(sources, targets)],
        dtype=torch.bool,
    )
    return make_acyclic(edge_index, reversed_mask), reversed_mask


def make_acyclic_robust(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Break cycles with a DFS pass and a greedy fallback for residual cycles.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(acyclic_edges, reversed_mask)`` aligned to the input edge order.
    """
    if edge_index.numel() == 0:
        return edge_index.clone(), torch.zeros((0,), dtype=torch.bool)

    back_edge_mask = detect_back_edges(edge_index, num_nodes)
    acyclic_edges = make_acyclic(edge_index, back_edge_mask)
    if _is_acyclic(acyclic_edges, num_nodes):
        return acyclic_edges, back_edge_mask.to(dtype=torch.bool)
    return _greedy_fas(acyclic_edges, num_nodes)
