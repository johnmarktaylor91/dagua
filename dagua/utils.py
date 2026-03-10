"""Graph utilities: text measurement, topology helpers."""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import torch


def measure_text(
    text: str,
    font_family: str = "",
    font_size: float = 8.5,
) -> Tuple[float, float]:
    """Measure text dimensions using matplotlib TextPath.

    If font_family is empty, uses the resolved best-available font.
    """
    if not font_family:
        try:
            from dagua.styles import RESOLVED_FONT
            font_family = RESOLVED_FONT
        except ImportError:
            font_family = "sans-serif"
    try:
        from matplotlib.font_manager import FontProperties
        from matplotlib.textpath import TextPath

        fp = FontProperties(family=font_family, size=font_size)
        tp = TextPath((0, 0), text, prop=fp)
        bbox = tp.get_extents()
        return max(bbox.width, 1.0), max(bbox.height, font_size)
    except Exception:
        return measure_text_fallback(text, font_size)


def measure_text_fallback(text: str, font_size: float = 8.5) -> Tuple[float, float]:
    """Fast proportional-font approximation (no matplotlib needed).

    For sans-serif fonts, average char width ≈ 0.52 × font_size.
    """
    lines = text.split("\n")
    max_chars = max(len(line) for line in lines) if lines else 1
    width = max_chars * font_size * 0.52
    height = len(lines) * font_size * 1.2
    return max(width, 1.0), max(height, font_size)


# Node sizing constants
MIN_NODE_WIDTH = 40.0
MIN_NODE_HEIGHT = 22.0
MAX_NODE_ASPECT_RATIO = 6.0


def compute_node_size(
    label: str,
    font_family: str = "",
    font_size: float = 8.5,
    padding: Tuple[float, float] = (8.0, 5.0),
) -> Tuple[float, float]:
    """Compute node bounding box from label text.

    Enforces minimum dimensions and maximum aspect ratio per style guide.
    """
    text_w, text_h = measure_text(label, font_family, font_size)
    w = text_w + padding[0] * 2
    h = text_h + padding[1] * 2

    # Enforce minimums
    w = max(w, MIN_NODE_WIDTH)
    h = max(h, MIN_NODE_HEIGHT)

    # Enforce max aspect ratio (truncation would happen at render time)
    if w / h > MAX_NODE_ASPECT_RATIO:
        w = h * MAX_NODE_ASPECT_RATIO

    return w, h


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
