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

    # Step 3: Barycenter ordering within each layer (2 passes)
    if edge_index.numel() > 0:
        src = edge_index[0].tolist()
        tgt = edge_index[1].tolist()

        # Build adjacency
        children_of: Dict[int, List[int]] = defaultdict(list)
        parents_of: Dict[int, List[int]] = defaultdict(list)
        for s, t in zip(src, tgt):
            children_of[s].append(t)
            parents_of[t].append(s)

        node_order = {n: i for i, n in enumerate(range(num_nodes))}

        # Forward pass: order by average position of parents
        sorted_layers = sorted(layer_groups.keys())
        for layer_idx in sorted_layers[1:]:  # skip first layer (no parents)
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

        # Update node_order after forward pass
        pos_counter = 0
        for layer_idx in sorted_layers:
            for node in layer_groups[layer_idx]:
                node_order[node] = pos_counter
                pos_counter += 1

        # Backward pass: order by average position of children
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
