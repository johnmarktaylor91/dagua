"""Graph utilities: text measurement, topology helpers, VRAM checks."""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import torch


def _vram_fits(needed_bytes: int, safety: float = 0.8) -> bool:
    """Check whether *needed_bytes* fit in free GPU VRAM (with headroom).

    Returns False when CUDA is unavailable, so CPU paths are unchanged.
    *safety* (default 0.8) reserves 20% headroom for allocator fragmentation.
    """
    if not torch.cuda.is_available():
        return False
    free, _total = torch.cuda.mem_get_info()
    return needed_bytes < int(free * safety)


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


def longest_path_layering(edge_index: torch.Tensor, num_nodes: int) -> "List[int] | torch.Tensor":
    """Assign layer indices via longest-path from sources. O(V+E).

    For large graphs (>10K nodes), uses a vectorized wave-based approach
    that avoids Python-level per-node iteration. Returns a tensor directly
    for large N to avoid expensive .tolist() conversions.
    """
    if edge_index.numel() == 0:
        if num_nodes > 10000:
            return torch.zeros(num_nodes, dtype=torch.long)
        return [0] * num_nodes

    if num_nodes > 10000:
        return _longest_path_layering_vectorized(edge_index, num_nodes)

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


_EDGE_CHUNK = 10_000_000  # edges per chunk for streaming ops
_STREAMING_NODE_THRESHOLD = 100_000_000  # switch to chunked ops above this


def _process_wave_edges_chunked(
    src: torch.Tensor,
    tgt: torch.Tensor,
    wave_set: torch.Tensor,
    layers: torch.Tensor,
    remaining: torch.Tensor,
    E: int,
) -> None:
    """Process wave edges in chunks to avoid materializing a full [E] mask.

    Mutates `layers` and `remaining` in place.
    """
    for start in range(0, E, _EDGE_CHUNK):
        end = min(start + _EDGE_CHUNK, E)
        chunk_src = src[start:end]
        chunk_tgt = tgt[start:end]
        chunk_mask = wave_set[chunk_src]
        if chunk_mask.any():
            children = chunk_tgt[chunk_mask]
            candidate = layers[chunk_src[chunk_mask]] + 1
            layers.scatter_reduce_(0, children, candidate, reduce="amax")
            ones = torch.ones(children.shape[0], dtype=torch.long)
            remaining.scatter_add_(0, children, -ones)


def _longest_path_layering_vectorized(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Longest-path layering using hybrid wave/BFS strategy.

    Wide graphs (many nodes per wave): use vectorized wave approach — each
    iteration processes an entire topological layer with tensor ops.
    Deep graphs (few nodes per wave): use CSR + numpy BFS — true O(V+E).

    For N > 100M, uses chunked edge processing to avoid [E]-sized temporaries
    (saves ~12 GB at 1B nodes).

    Heuristic: run 10 waves. If average wave size > 1000, continue with waves.
    Otherwise switch to CSR+BFS.

    Returns a tensor directly (callers that need a list can convert).
    """
    import numpy as np
    from collections import deque

    N = num_nodes
    E = edge_index.shape[1]
    src, tgt = edge_index[0], edge_index[1]
    chunked = N > _STREAMING_NODE_THRESHOLD

    # Compute in-degree — chunked for large graphs to avoid [E]-sized ones tensor
    in_degree = torch.zeros(N, dtype=torch.long)
    if chunked:
        for start in range(0, E, _EDGE_CHUNK):
            end = min(start + _EDGE_CHUNK, E)
            in_degree.scatter_add_(0, tgt[start:end], torch.ones(end - start, dtype=torch.long))
    else:
        ones_E = torch.ones(E, dtype=torch.long)
        in_degree.scatter_add_(0, tgt, ones_E)

    # Probe: run a few waves to decide strategy
    layers = torch.zeros(N, dtype=torch.long)
    remaining = in_degree.clone()
    total_processed = 0
    current_layer = 0
    probe_waves = 10

    # Pre-allocate wave_set once — reuse via .zero_() each wave
    wave_set = torch.zeros(N, dtype=torch.bool)

    for _ in range(probe_waves):
        wave = (remaining == 0).nonzero(as_tuple=True)[0]
        if wave.numel() == 0:
            break
        total_processed += wave.numel()
        layers[wave] = current_layer
        remaining[wave] = -1

        wave_set.zero_()
        wave_set[wave] = True

        if chunked:
            _process_wave_edges_chunked(src, tgt, wave_set, layers, remaining, E)
        else:
            edge_mask = wave_set[src]
            if edge_mask.any():
                children = tgt[edge_mask]
                candidate = layers[src[edge_mask]] + 1
                layers.scatter_reduce_(0, children, candidate, reduce="amax")
                ones = torch.ones(children.shape[0], dtype=torch.long)
                remaining.scatter_add_(0, children, -ones)

        current_layer += 1

    avg_wave = total_processed / max(current_layer, 1)

    # Wide graph: continue with waves (fast when few iterations needed)
    if avg_wave > 1000:
        for _ in range(N):
            wave = (remaining == 0).nonzero(as_tuple=True)[0]
            if wave.numel() == 0:
                break
            layers[wave] = current_layer
            remaining[wave] = -1

            wave_set.zero_()
            wave_set[wave] = True

            if chunked:
                _process_wave_edges_chunked(src, tgt, wave_set, layers, remaining, E)
            else:
                edge_mask = wave_set[src]
                if edge_mask.any():
                    children = tgt[edge_mask]
                    candidate = layers[src[edge_mask]] + 1
                    layers.scatter_reduce_(0, children, candidate, reduce="amax")
                    ones = torch.ones(children.shape[0], dtype=torch.long)
                    remaining.scatter_add_(0, children, -ones)

            current_layer += 1

        return layers

    # Deep graph: switch to CSR + numpy BFS (true O(V+E))
    # Reset — recompute from scratch with numpy (zero-copy from torch)
    in_deg = in_degree.numpy().copy()

    # Build CSR adjacency via sort
    order = src.argsort()
    csr_tgt = tgt[order].numpy()

    # Chunked out-degree for large graphs
    out_degree = torch.zeros(N, dtype=torch.long)
    if chunked:
        for start in range(0, E, _EDGE_CHUNK):
            end = min(start + _EDGE_CHUNK, E)
            out_degree.scatter_add_(0, src[start:end], torch.ones(end - start, dtype=torch.long))
    else:
        out_degree.scatter_add_(0, src, ones_E)
    offsets = torch.zeros(N + 1, dtype=torch.long)
    offsets[1:] = out_degree.cumsum(0)
    csr_off = offsets.numpy()

    # BFS from sources — true O(V+E)
    layer_arr = np.zeros(N, dtype=np.int64)
    queue = deque(int(i) for i in range(N) if in_deg[i] == 0)

    while queue:
        node = queue.popleft()
        child_layer = layer_arr[node] + 1
        for j in range(csr_off[node], csr_off[node + 1]):
            child = int(csr_tgt[j])
            if child_layer > layer_arr[child]:
                layer_arr[child] = child_layer
            in_deg[child] -= 1
            if in_deg[child] == 0:
                queue.append(child)

    return torch.from_numpy(layer_arr)


def collect_cluster_leaves(cluster_dict) -> List[int]:
    """Recursively collect all leaf node indices from a nested cluster dict."""
    leaves = []
    if isinstance(cluster_dict, list):
        return cluster_dict
    if isinstance(cluster_dict, dict):
        for v in cluster_dict.values():
            leaves.extend(collect_cluster_leaves(v))
    return leaves
