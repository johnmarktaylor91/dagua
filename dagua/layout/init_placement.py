"""Algorithmic initialization: topological sort + barycenter heuristic.

Good initialization is CRITICAL for convergence quality. Random init → slow
convergence, poor local minima. This module provides near-optimal starting
positions using classical graph drawing algorithms.

Sprint 3 scaling strategy:
- Barycenter ordering uses tensor ops (index_add_ / scatter) instead of Python loops
- For N > 10K: reduced passes (5 instead of 30) + tensor-based coordinate assignment
- Transpose heuristic skipped for very large graphs (diminishing returns)
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import torch

from dagua.utils import longest_path_layering


def init_positions(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: torch.Tensor,
    node_sep: float = 25.0,
    rank_sep: float = 50.0,
    device: str = "cpu",
) -> torch.Tensor:
    """Compute initial positions via topological layering + barycenter ordering.

    Returns: [N, 2] tensor of (x, y) positions.
    """
    # Step 1: Assign layers (y-coordinates) via longest-path
    layers = longest_path_layering(edge_index, num_nodes)

    # For large graphs, use fully vectorized path
    if num_nodes > 2000:
        return _init_positions_vectorized(
            edge_index, num_nodes, node_sizes, layers,
            node_sep, rank_sep, device,
        )

    # Step 2: Group nodes by layer
    layer_groups: Dict[int, List[int]] = defaultdict(list)
    for node, layer in enumerate(layers):
        layer_groups[layer].append(node)

    # Step 3: Multi-pass barycenter crossing reduction (Sugiyama Phase 2)
    if edge_index.numel() > 0:
        src = edge_index[0].tolist()
        tgt = edge_index[1].tolist()

        # Build adjacency
        children_of: Dict[int, List[int]] = defaultdict(list)
        parents_of: Dict[int, List[int]] = defaultdict(list)
        for s, t in zip(src, tgt):
            children_of[s].append(t)
            parents_of[t].append(s)

        node_order = {n: float(i) for i, n in enumerate(range(num_nodes))}
        sorted_layers = sorted(layer_groups.keys())

        num_passes = min(max(10, num_nodes // 10), 30)

        for _pass in range(num_passes):
            # Forward pass: order by barycenter of parents
            for layer_idx in sorted_layers[1:]:
                nodes = layer_groups[layer_idx]
                barycenters = []
                for node in nodes:
                    parents = parents_of[node]
                    if parents:
                        avg_x = sum(node_order[p] for p in parents) / len(parents)
                    else:
                        avg_x = node_order[node]
                    barycenters.append((avg_x, node))
                barycenters.sort()
                layer_groups[layer_idx] = [n for _, n in barycenters]

            _update_node_order(node_order, layer_groups, sorted_layers)

            # Backward pass: order by barycenter of children
            for layer_idx in reversed(sorted_layers[:-1]):
                nodes = layer_groups[layer_idx]
                barycenters = []
                for node in nodes:
                    kids = children_of[node]
                    if kids:
                        avg_x = sum(node_order[k] for k in kids) / len(kids)
                    else:
                        avg_x = node_order[node]
                    barycenters.append((avg_x, node))
                barycenters.sort()
                layer_groups[layer_idx] = [n for _, n in barycenters]

            _update_node_order(node_order, layer_groups, sorted_layers)

        # Transpose heuristic — only for small graphs where the quadratic cost is manageable
        if num_nodes <= 500:
            _transpose_heuristic(layer_groups, sorted_layers, children_of, parents_of, num_passes=5)
        elif num_nodes <= 1000:
            _transpose_heuristic(layer_groups, sorted_layers, children_of, parents_of, num_passes=2)

    # Step 4: Assign coordinates
    positions = torch.zeros(num_nodes, 2, device=device)
    node_sizes_cpu = node_sizes.cpu() if node_sizes.device.type != "cpu" else node_sizes

    for layer_idx, nodes in layer_groups.items():
        y = layer_idx * rank_sep

        total_width = sum(node_sizes_cpu[n, 0].item() for n in nodes) + node_sep * max(len(nodes) - 1, 0)
        x_start = -total_width / 2

        x_cursor = x_start
        for node in nodes:
            w = node_sizes_cpu[node, 0].item()
            positions[node, 0] = x_cursor + w / 2
            positions[node, 1] = y
            x_cursor += w + node_sep

    return positions


def _init_positions_vectorized(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: torch.Tensor,
    layers: List[int],
    node_sep: float,
    rank_sep: float,
    device: str,
) -> torch.Tensor:
    """Fully vectorized initialization for large graphs.

    Uses tensor operations for barycenter ordering:
    - index_add_ for summing parent/child positions
    - argsort within layers via composite sort key
    - Vectorized coordinate assignment

    No Python loops over individual nodes.
    """
    N = num_nodes
    layer_t = torch.tensor(layers, dtype=torch.long, device=device)
    num_layers = int(layer_t.max().item()) + 1 if N > 0 else 0

    # Build layer structure
    counts = torch.bincount(layer_t, minlength=num_layers)
    offsets = torch.zeros(num_layers + 1, dtype=torch.long, device=device)
    offsets[1:] = counts.cumsum(0)

    # Sort nodes by layer for contiguous access
    sorted_by_layer = layer_t.argsort()

    # Initialize order values: position within initial layer grouping
    order = torch.zeros(N, device=device)
    for L in range(num_layers):
        s, e = offsets[L].item(), offsets[L + 1].item()
        if e > s:
            order[sorted_by_layer[s:e]] = torch.arange(e - s, dtype=torch.float32, device=device)

    if edge_index.numel() > 0:
        src = edge_index[0].to(device)
        tgt = edge_index[1].to(device)

        # Precompute in-degree and out-degree for normalization
        in_degree = torch.zeros(N, device=device)
        out_degree = torch.zeros(N, device=device)
        in_degree.scatter_add_(0, tgt, torch.ones(tgt.shape[0], device=device))
        out_degree.scatter_add_(0, src, torch.ones(src.shape[0], device=device))

        # Barycenter passes using tensor scatter operations
        num_passes = 5  # fewer passes but tensor-based = much faster
        for _pass in range(num_passes):
            # Forward pass: each node's order = mean of parents' orders
            parent_sum = torch.zeros(N, device=device)
            parent_sum.scatter_add_(0, tgt, order[src])
            has_parents = in_degree > 0
            new_order = torch.where(has_parents, parent_sum / in_degree.clamp(min=1), order)

            # Sort within each layer by new_order (composite key trick)
            sort_key = layer_t.float() * (N + 1) + new_order
            global_sorted = sort_key.argsort()
            # Assign sequential positions within each layer
            sorted_layers = layer_t[global_sorted]
            # Use cumcount within each layer group
            layer_starts_expanded = offsets[sorted_layers]
            positions_in_sort = torch.arange(N, device=device)
            within_layer_pos = (positions_in_sort - layer_starts_expanded).float()
            # Write back to original node indices
            order[global_sorted] = within_layer_pos

            # Backward pass: each node's order = mean of children's orders
            child_sum = torch.zeros(N, device=device)
            child_sum.scatter_add_(0, src, order[tgt])
            has_children = out_degree > 0
            new_order = torch.where(has_children, child_sum / out_degree.clamp(min=1), order)

            sort_key = layer_t.float() * (N + 1) + new_order
            global_sorted = sort_key.argsort()
            sorted_layers = layer_t[global_sorted]
            layer_starts_expanded = offsets[sorted_layers]
            positions_in_sort = torch.arange(N, device=device)
            within_layer_pos = (positions_in_sort - layer_starts_expanded).float()
            order[global_sorted] = within_layer_pos

    # Assign coordinates based on final ordering
    positions = torch.zeros(N, 2, device=device)

    # Y-coordinates: layer * rank_sep
    positions[:, 1] = layer_t.float() * rank_sep

    # X-coordinates: within-layer position * (avg_width + node_sep), centered
    node_w = node_sizes[:, 0].to(device)
    avg_w = node_w.mean()
    spacing = avg_w + node_sep

    # For each layer, compute centered x positions based on order
    # x = (order - layer_width/2) * spacing
    layer_widths = counts.float()  # [L]
    node_layer_width = layer_widths[layer_t]  # [N]
    positions[:, 0] = (order - node_layer_width / 2) * spacing

    return positions


def _transpose_heuristic(
    layer_groups: Dict[int, List[int]],
    sorted_layers: List[int],
    children_of: Dict[int, List[int]],
    parents_of: Dict[int, List[int]],
    num_passes: int = 5,
) -> None:
    """Swap adjacent nodes within layers if it reduces edge crossings."""
    for _ in range(num_passes):
        improved = False
        for layer_idx in sorted_layers:
            nodes = layer_groups[layer_idx]
            if len(nodes) < 2:
                continue

            for i in range(len(nodes) - 1):
                u, v = nodes[i], nodes[i + 1]

                cross_before = _count_local_crossings(u, v, nodes, layer_groups, sorted_layers,
                                                       children_of, parents_of, layer_idx)
                nodes[i], nodes[i + 1] = v, u
                cross_after = _count_local_crossings(v, u, nodes, layer_groups, sorted_layers,
                                                      children_of, parents_of, layer_idx)
                if cross_after >= cross_before:
                    nodes[i], nodes[i + 1] = u, v
                else:
                    improved = True

        if not improved:
            break


def _count_local_crossings(
    u: int, v: int,
    nodes: List[int],
    layer_groups: Dict[int, List[int]],
    sorted_layers: List[int],
    children_of: Dict[int, List[int]],
    parents_of: Dict[int, List[int]],
    current_layer: int,
) -> int:
    """Count crossings between edges from u,v to adjacent layers."""
    crossings = 0

    pos_in_layer = {n: i for i, n in enumerate(nodes)}

    layer_idx_pos = sorted_layers.index(current_layer)
    if layer_idx_pos + 1 < len(sorted_layers):
        next_layer = sorted_layers[layer_idx_pos + 1]
        next_nodes = layer_groups[next_layer]
        next_pos = {n: i for i, n in enumerate(next_nodes)}

        u_children = [c for c in children_of.get(u, []) if c in next_pos]
        v_children = [c for c in children_of.get(v, []) if c in next_pos]

        u_pos = pos_in_layer[u]
        v_pos = pos_in_layer[v]

        for uc in u_children:
            for vc in v_children:
                if (u_pos < v_pos) != (next_pos[uc] < next_pos[vc]):
                    crossings += 1

    if layer_idx_pos > 0:
        prev_layer = sorted_layers[layer_idx_pos - 1]
        prev_nodes = layer_groups[prev_layer]
        prev_pos = {n: i for i, n in enumerate(prev_nodes)}

        u_parents = [p for p in parents_of.get(u, []) if p in prev_pos]
        v_parents = [p for p in parents_of.get(v, []) if p in prev_pos]

        u_pos = pos_in_layer[u]
        v_pos = pos_in_layer[v]

        for up in u_parents:
            for vp in v_parents:
                if (u_pos < v_pos) != (prev_pos[up] < prev_pos[vp]):
                    crossings += 1

    return crossings


def _update_node_order(
    node_order: Dict[int, float],
    layer_groups: Dict[int, List[int]],
    sorted_layers: List[int],
) -> None:
    """Update node_order dict from current layer group ordering."""
    pos_counter = 0.0
    for layer_idx in sorted_layers:
        for node in layer_groups[layer_idx]:
            node_order[node] = pos_counter
            pos_counter += 1.0
