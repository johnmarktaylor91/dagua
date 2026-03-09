"""Graph utilities: text measurement, topology helpers."""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import torch


def measure_text(text: str, font_family: str = "monospace", font_size: float = 9.0) -> Tuple[float, float]:
    """Measure text dimensions using matplotlib TextPath."""
    try:
        from matplotlib.font_manager import FontProperties
        from matplotlib.textpath import TextPath

        fp = FontProperties(family=font_family, size=font_size)
        tp = TextPath((0, 0), text, prop=fp)
        bbox = tp.get_extents()
        return max(bbox.width, 1.0), max(bbox.height, font_size)
    except Exception:
        return measure_text_fallback(text, font_size)


def measure_text_fallback(text: str, font_size: float = 9.0) -> Tuple[float, float]:
    """Fast monospace approximation (no matplotlib needed)."""
    lines = text.split("\n")
    max_chars = max(len(line) for line in lines) if lines else 1
    width = max_chars * font_size * 0.6
    height = len(lines) * font_size * 1.2
    return max(width, 1.0), max(height, font_size)


def compute_node_size(
    label: str,
    font_family: str = "monospace",
    font_size: float = 9.0,
    padding: Tuple[float, float] = (8.0, 4.0),
) -> Tuple[float, float]:
    """Compute node bounding box from label text."""
    text_w, text_h = measure_text(label, font_family, font_size)
    return text_w + padding[0] * 2, text_h + padding[1] * 2


def topological_sort(edge_index: torch.Tensor, num_nodes: int) -> List[int]:
    """Kahn's algorithm. Returns nodes in topological order.

    Falls back to BFS from roots if cycles exist.
    """
    if edge_index.numel() == 0:
        return list(range(num_nodes))

    src = edge_index[0].tolist()
    tgt = edge_index[1].tolist()

    in_degree = [0] * num_nodes
    children: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}

    for s, t in zip(src, tgt):
        in_degree[t] += 1
        children[s].append(t)

    queue = deque([i for i in range(num_nodes) if in_degree[i] == 0])
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for child in children[node]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    # If not all nodes visited (cycles), add remaining
    if len(order) < num_nodes:
        visited = set(order)
        for i in range(num_nodes):
            if i not in visited:
                order.append(i)

    return order


def longest_path_layering(edge_index: torch.Tensor, num_nodes: int) -> List[int]:
    """Assign layer indices via longest-path from sources. O(V+E)."""
    if edge_index.numel() == 0:
        return [0] * num_nodes

    src = edge_index[0].tolist()
    tgt = edge_index[1].tolist()

    children: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
    in_degree = [0] * num_nodes

    for s, t in zip(src, tgt):
        children[s].append(t)
        in_degree[t] += 1

    # BFS from sources
    layers = [0] * num_nodes
    queue = deque([i for i in range(num_nodes) if in_degree[i] == 0])

    while queue:
        node = queue.popleft()
        for child in children[node]:
            layers[child] = max(layers[child], layers[node] + 1)
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    return layers


def collect_cluster_leaves(cluster_dict) -> List[int]:
    """Recursively collect all leaf node indices from a nested cluster dict."""
    leaves = []
    if isinstance(cluster_dict, list):
        return cluster_dict
    if isinstance(cluster_dict, dict):
        for v in cluster_dict.values():
            leaves.extend(collect_cluster_leaves(v))
    return leaves
