"""Algorithmic initialization: topological sort + barycenter heuristic.

Good initialization is CRITICAL for convergence quality. Random init → slow
convergence, poor local minima. This module provides near-optimal starting
positions using classical graph drawing algorithms.
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

    # Step 2: Group nodes by layer
    layer_groups: Dict[int, List[int]] = defaultdict(list)
    for node, layer in enumerate(layers):
        layer_groups[layer].append(node)

    # Step 3: Multi-pass barycenter crossing reduction (Sugiyama Phase 2)
    # More passes = fewer crossings. Diminishing returns after ~20 passes.
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

        # Run multiple forward+backward sweeps
        # Scale passes with graph size: more passes for larger graphs
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

            # Update positions after forward pass
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

            # Update positions after backward pass
            _update_node_order(node_order, layer_groups, sorted_layers)

        # Step 3b: Transpose heuristic — swap adjacent nodes if it reduces crossings
        # This catches cases barycenter misses (local optima)
        _transpose_heuristic(layer_groups, sorted_layers, children_of, parents_of, num_passes=5)

    # Step 4: Assign coordinates
    positions = torch.zeros(num_nodes, 2, device=device)
    node_sizes_cpu = node_sizes.cpu() if node_sizes.device.type != "cpu" else node_sizes

    for layer_idx, nodes in layer_groups.items():
        # y-coordinate from layer index
        y = layer_idx * rank_sep

        # x-coordinates: space nodes within layer
        total_width = sum(node_sizes_cpu[n, 0].item() for n in nodes) + node_sep * max(len(nodes) - 1, 0)
        x_start = -total_width / 2

        x_cursor = x_start
        for node in nodes:
            w = node_sizes_cpu[node, 0].item()
            positions[node, 0] = x_cursor + w / 2
            positions[node, 1] = y
            x_cursor += w + node_sep

    return positions


def _transpose_heuristic(
    layer_groups: Dict[int, List[int]],
    sorted_layers: List[int],
    children_of: Dict[int, List[int]],
    parents_of: Dict[int, List[int]],
    num_passes: int = 5,
) -> None:
    """Swap adjacent nodes within layers if it reduces edge crossings.

    Classic Sugiyama Phase 2 refinement. For each pair of adjacent nodes
    in a layer, count crossings before and after swap; keep the swap if
    it reduces crossings.
    """
    for _ in range(num_passes):
        improved = False
        for layer_idx in sorted_layers:
            nodes = layer_groups[layer_idx]
            if len(nodes) < 2:
                continue

            for i in range(len(nodes) - 1):
                u, v = nodes[i], nodes[i + 1]

                # Count crossings involving these two nodes with their neighbors
                cross_before = _count_local_crossings(u, v, nodes, layer_groups, sorted_layers,
                                                       children_of, parents_of, layer_idx)
                nodes[i], nodes[i + 1] = v, u
                cross_after = _count_local_crossings(v, u, nodes, layer_groups, sorted_layers,
                                                      children_of, parents_of, layer_idx)
                if cross_after >= cross_before:
                    # Revert
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
    """Count crossings between edges from u,v to adjacent layers.

    Only counts crossings involving the two nodes u and v (not all crossings).
    """
    crossings = 0

    # Build position lookup for current layer
    pos_in_layer = {n: i for i, n in enumerate(nodes)}

    # Check against next layer (children)
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
                # Crossing: u is left of v but uc is right of vc (or vice versa)
                if (u_pos < v_pos) != (next_pos[uc] < next_pos[vc]):
                    crossings += 1

    # Check against previous layer (parents)
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
