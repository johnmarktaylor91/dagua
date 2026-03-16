"""Graph utilities: text measurement, topology helpers, VRAM checks."""

from __future__ import annotations

from collections import deque
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import torch

RichMarkup = Dict[str, Any]
RichSegment = Tuple[str, RichMarkup]


class VRAMBudget:
    """Unified VRAM budget tracker with fragmentation-aware safety margins.

    Replaces scattered ``_vram_fits()`` calls with measurement-based decisions.
    Queries ``torch.cuda.mem_get_info()`` and adjusts the safety factor based on
    allocator fragmentation (allocated/reserved ratio).

    Parameters
    ----------
    None
        Queries GPU state at construction time.
    """

    def __init__(self) -> None:
        """Capture the current CUDA memory state."""
        if not torch.cuda.is_available():
            self._free = 0
            self._total = 0
            self._frag_ratio = 1.0
            return
        self._free, self._total = torch.cuda.mem_get_info()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        self._frag_ratio = allocated / max(reserved, 1)

    @staticmethod
    def available() -> bool:
        """Return whether CUDA is available."""
        return torch.cuda.is_available()

    @property
    def dynamic_safety(self) -> float:
        """Return a safety factor based on allocator fragmentation.

        The baseline headroom depends on total VRAM so consumer cards stay more
        conservative than large datacenter GPUs. Fragmentation can only reduce
        that baseline further.
        """
        clamped = max(0.5, min(self._frag_ratio, 1.0))
        frag_safety = 0.60 + (clamped - 0.5) * (0.85 - 0.60) / 0.5
        return min(_vram_safety_factor(self._total), frag_safety)

    def fits(self, needed_bytes: int) -> bool:
        """Return whether ``needed_bytes`` fit in free VRAM with headroom.

        Parameters
        ----------
        needed_bytes : int
            Estimated allocation size in bytes.

        Returns
        -------
        bool
            ``True`` when the request fits in the measured free VRAM budget.
        """
        if not torch.cuda.is_available():
            return False
        return needed_bytes < int(self._free * self.dynamic_safety)

    def remaining(self) -> int:
        """Return the estimated usable VRAM in bytes.

        Returns
        -------
        int
            ``free_vram * dynamic_safety`` for the captured CUDA state.
        """
        if not torch.cuda.is_available():
            return 0
        return int(self._free * self.dynamic_safety)


def _vram_fits(needed_bytes: int, safety: float = 0.8) -> bool:
    """Check whether *needed_bytes* fit in free GPU VRAM (with headroom).

    Returns False when CUDA is unavailable, so CPU paths are unchanged.
    *safety* is ignored — VRAMBudget.dynamic_safety is used instead.
    Kept for backward compatibility; new code should use VRAMBudget directly.
    """
    _ = safety
    return VRAMBudget().fits(needed_bytes)


def _vram_safety_factor(total_bytes: Optional[int] = None) -> float:
    """Return the VRAM headroom factor for the active CUDA device.

    Parameters
    ----------
    total_bytes : int, optional
        Explicit total VRAM capacity in bytes. When omitted, the value is read
        from the current CUDA device.

    Returns
    -------
    float
        Safety factor tuned by GPU memory class: ``0.75`` for cards below
        16 GB, ``0.80`` for 16-32 GB, and ``0.85`` for 32 GB or larger.
    """
    if total_bytes is None:
        if not torch.cuda.is_available():
            return 0.75
        total_bytes = torch.cuda.get_device_properties(0).total_memory

    if total_bytes < 16 * 1024**3:
        return 0.75
    if total_bytes < 32 * 1024**3:
        return 0.80
    return 0.85


def measure_text(
    text: str,
    font_family: str = "",
    font_size: float = 8.5,
    font_weight: str = "regular",
) -> Tuple[float, float]:
    """Measure plain-text dimensions.

    Parameters
    ----------
    text : str
        Label text to measure.
    font_family : str, default=""
        Preferred font family. Empty strings use Dagua's resolved default font.
    font_size : float, default=8.5
        Font size in points.
    font_weight : str, default="regular"
        Font weight passed through to matplotlib font resolution.

    Returns
    -------
    tuple[float, float]
        Approximate text width and height in points.
    """
    if not font_family:
        try:
            from dagua.styles import RESOLVED_FONT

            font_family = RESOLVED_FONT
        except ImportError:
            font_family = "sans-serif"
    try:
        return _measure_text_exact_cached(text, font_family, font_size, font_weight)
    except Exception:
        return measure_text_fallback(text, font_size, font_weight)


def measure_text_fallback(
    text: str,
    font_size: float = 8.5,
    font_weight: str = "regular",
) -> Tuple[float, float]:
    """Approximate plain-text dimensions without matplotlib.

    Parameters
    ----------
    text : str
        Label text to measure.
    font_size : float, default=8.5
        Font size in points.
    font_weight : str, default="regular"
        Font weight hint. Bold text is treated as slightly wider.

    Returns
    -------
    tuple[float, float]
        Approximate text width and height in points.
    """
    lines = text.split("\n")
    max_chars = max(len(line) for line in lines) if lines else 1
    char_width = font_size * 0.52
    if font_weight in ("bold", "heavy", "black"):
        char_width *= 1.05
    width = max_chars * char_width
    height = len(lines) * font_size * 1.2
    return max(width, 1.0), max(height, font_size)


@lru_cache(maxsize=16384)
def _measure_text_exact_cached(
    text: str,
    font_family: str,
    font_size: float,
    font_weight: str,
) -> Tuple[float, float]:
    """Measure plain text exactly with matplotlib.

    Parameters
    ----------
    text : str
        Label text to measure.
    font_family : str
        Font family name.
    font_size : float
        Font size in points.
    font_weight : str
        Font weight name.

    Returns
    -------
    tuple[float, float]
        Exact text width and height in points.
    """
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    fp = FontProperties(family=font_family, size=font_size, weight=font_weight)
    tp = TextPath((0, 0), text, prop=fp)
    bbox = tp.get_extents()
    return max(bbox.width, 1.0), max(bbox.height, font_size)


def parse_rich_markup(text: str) -> List[RichSegment]:
    """Parse lightweight rich-text markup into styled segments.

    Parameters
    ----------
    text : str
        Input text supporting ``**bold**``, ``*italic*``, `` `mono` ``,
        ``~~strike~~``, ``__underline__``, and ``{color:#RRGGBB}...{/color}``.

    Returns
    -------
    list[tuple[str, dict[str, Any]]]
        Ordered segments of literal text paired with formatting flags.
    """
    segments: List[RichSegment] = []
    buffer: List[str] = []
    current: RichMarkup = {
        "bold": False,
        "italic": False,
        "mono": False,
        "strike": False,
        "underline": False,
        "color": None,
    }
    color_stack: List[Any] = []
    idx = 0

    def flush_buffer() -> None:
        """Persist the current text buffer as a segment."""
        if not buffer:
            return
        text_value = "".join(buffer)
        buffer.clear()
        if segments and segments[-1][1] == current:
            prev_text, prev_style = segments[-1]
            segments[-1] = (prev_text + text_value, prev_style)
            return
        segments.append((text_value, dict(current)))

    while idx < len(text):
        if text.startswith("{/color}", idx):
            flush_buffer()
            current["color"] = color_stack.pop() if color_stack else None
            idx += len("{/color}")
            continue

        if text.startswith("{color:#", idx):
            end = text.find("}", idx)
            if end != -1:
                color_value = text[idx + len("{color:") : end]
                flush_buffer()
                color_stack.append(current.get("color"))
                current["color"] = color_value
                idx = end + 1
                continue

        token_map = [
            ("**", "bold"),
            ("~~", "strike"),
            ("__", "underline"),
            ("`", "mono"),
            ("*", "italic"),
        ]
        matched = False
        for token, key in token_map:
            if text.startswith(token, idx):
                flush_buffer()
                current[key] = not bool(current[key])
                idx += len(token)
                matched = True
                break
        if matched:
            continue

        buffer.append(text[idx])
        idx += 1

    flush_buffer()
    return segments or [("", dict(current))]


def measure_rich_text(
    label: str,
    font_family: str,
    font_size: float,
) -> Tuple[float, float]:
    """Measure the rendered size of a rich-format label.

    Parameters
    ----------
    label : str
        Rich-format label text.
    font_family : str
        Base font family for non-monospace segments.
    font_size : float
        Font size in points.

    Returns
    -------
    tuple[float, float]
        Estimated rich-text width and height in points.
    """
    from dagua.styles import FONT_FAMILY_MONO

    segments = parse_rich_markup(label)
    line_widths: List[float] = [0.0]
    line_heights: List[float] = [max(font_size, 1.0)]

    for segment_text, segment_style in segments:
        parts = segment_text.split("\n")
        for part_index, part in enumerate(parts):
            if part:
                segment_family = FONT_FAMILY_MONO[0] if segment_style["mono"] else font_family
                segment_weight = "bold" if segment_style["bold"] else "regular"
                width, height = measure_text(part, segment_family, font_size, segment_weight)
                line_widths[-1] += width
                line_heights[-1] = max(line_heights[-1], height)
            if part_index != len(parts) - 1:
                line_widths.append(0.0)
                line_heights.append(max(font_size * 1.2, 1.0))

    total_height = sum(max(height, font_size * 1.2) for height in line_heights)
    return max(line_widths, default=1.0), max(total_height, font_size)


# Node sizing constants
MIN_NODE_WIDTH = 40.0
MIN_NODE_HEIGHT = 22.0
MAX_NODE_ASPECT_RATIO = 6.0
MAX_LABEL_WIDTH = 200.0


def compute_node_size(
    label: str,
    font_family: str = "",
    font_size: float = 8.5,
    padding: Tuple[float, float] = (8.0, 5.0),
    shape: str = "roundrect",
    font_weight: str = "regular",
    overflow_policy: str = "shrink_text",
    min_font_size: float = 5.0,
    label_format: str = "plain",
) -> Tuple[float, float, float]:
    """Compute a node bounding box from its label.

    Parameters
    ----------
    label : str
        Node label text.
    font_family : str, default=""
        Preferred font family.
    font_size : float, default=8.5
        Font size in points.
    padding : tuple[float, float], default=(8.0, 5.0)
        Horizontal and vertical padding in points.
    shape : str, default="roundrect"
        Node shape identifier.
    font_weight : str, default="regular"
        Font weight used for plain-text labels.
    overflow_policy : str, default="shrink_text"
        Overflow policy for oversized labels.
    min_font_size : float, default=5.0
        Minimum font size for the shrink-to-fit policy.
    label_format : str, default="plain"
        Label format, either ``"plain"`` or ``"rich"``.

    Returns
    -------
    tuple[float, float, float]
        Node width, node height, and effective font size.
    """
    return _compute_node_size_cached(
        label,
        font_family,
        font_size,
        padding,
        shape,
        font_weight,
        overflow_policy,
        min_font_size,
        label_format,
    )


@lru_cache(maxsize=16384)
def _compute_node_size_cached(
    label: str,
    font_family: str,
    font_size: float,
    padding: Tuple[float, float],
    shape: str,
    font_weight: str,
    overflow_policy: str,
    min_font_size: float,
    label_format: str,
) -> Tuple[float, float, float]:
    """Cached implementation of :func:`compute_node_size`.

    Parameters
    ----------
    label : str
        Node label text.
    font_family : str
        Preferred font family.
    font_size : float
        Font size in points.
    padding : tuple[float, float]
        Horizontal and vertical padding in points.
    shape : str
        Node shape identifier.
    font_weight : str
        Font weight used for plain-text labels.
    overflow_policy : str
        Overflow policy for oversized labels.
    min_font_size : float
        Minimum font size when shrinking labels.
    label_format : str
        Label format, either ``"plain"`` or ``"rich"``.

    Returns
    -------
    tuple[float, float, float]
        Node width, node height, and effective font size.
    """
    effective_font_size = font_size

    def measure_label(current_font_size: float) -> Tuple[float, float]:
        """Measure the label using the requested formatting mode."""
        if label_format == "rich":
            return measure_rich_text(label, font_family, current_font_size)
        return measure_text(label, font_family, current_font_size, font_weight)

    if overflow_policy == "shrink_text":
        text_w, text_h = measure_label(font_size)
        while text_w > MAX_LABEL_WIDTH and effective_font_size > min_font_size:
            effective_font_size -= 0.5
            text_w, text_h = measure_label(effective_font_size)
    else:
        text_w, text_h = measure_label(font_size)

    w = text_w + padding[0] * 2
    h = text_h + padding[1] * 2

    w = max(w, MIN_NODE_WIDTH)
    h = max(h, MIN_NODE_HEIGHT)

    if shape == "diamond":
        max_dim = max(w, h)
        w = h = max_dim * 1.42
    elif shape == "circle":
        r = max(w, h)
        w = h = r

    max_ratio = 10.0 if overflow_policy == "expand_node" else MAX_NODE_ASPECT_RATIO
    if w / h > max_ratio:
        w = h * max_ratio

    return w, h, effective_font_size


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


def longest_path_layering(
    edge_index: torch.Tensor,
    num_nodes: int,
    device: str = "cpu",
) -> "List[int] | torch.Tensor":
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
        return _longest_path_layering_vectorized(edge_index, num_nodes, device)

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


def _build_flat_indices(
    starts: torch.Tensor,
    counts: torch.Tensor,
    total: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Build flattened CSR offsets for repeated adjacency expansion.

    Parameters
    ----------
    starts : torch.Tensor
        CSR start offsets for each selected source node.
    counts : torch.Tensor
        Number of outgoing edges for each selected source node.
    total : int
        Total number of flattened offsets to build.
    device : torch.device | str
        Target device for the output tensor.

    Returns
    -------
    torch.Tensor
        Flattened offsets shaped ``[total]`` with per-source ranges reset to
        zero at each CSR segment boundary.
    """
    del starts
    result = torch.ones(total, dtype=torch.long, device=device)
    boundaries = counts.cumsum(0)[:-1]
    if boundaries.numel() > 0:
        result[boundaries] -= counts[:-1]
    result[0] = 0
    return result.cumsum(0)


def _build_csr(
    src: torch.Tensor,
    tgt: torch.Tensor,
    num_nodes: int,
    device: torch.device | str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build CSR adjacency for the graph.

    Parameters
    ----------
    src : torch.Tensor
        Source node indices for each edge.
    tgt : torch.Tensor
        Target node indices for each edge.
    num_nodes : int
        Number of nodes in the graph.
    device : torch.device | str
        Device where CSR tensors should be constructed.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``(csr_offsets, csr_targets)`` describing outgoing adjacency.
    """
    if str(device) != "cpu" or (torch.cuda.is_available() and src.shape[0] > 10_000_000):
        try:
            from dagua.layout.cuda_kernels import build_csr_cuda, is_available

            if is_available():
                cuda_device = device if str(device) != "cpu" else "cuda"
                offsets, targets = build_csr_cuda(
                    src.to(cuda_device),
                    tgt.to(cuda_device),
                    num_nodes,
                )
                if str(device) == "cpu":
                    return offsets.cpu(), targets.cpu()
                return offsets, targets
        except Exception:
            pass

    edge_count = src.shape[0]
    if str(device) == "cpu" and edge_count > 1_000_000:
        try:
            return _build_csr_numpy(src, tgt, num_nodes)
        except Exception:
            pass

    chunked = num_nodes > _STREAMING_NODE_THRESHOLD
    val_dtype = torch.int32 if chunked else torch.long

    out_degree = torch.zeros(num_nodes, dtype=val_dtype, device=device)
    if chunked:
        for start in range(0, edge_count, _EDGE_CHUNK):
            end = min(start + _EDGE_CHUNK, edge_count)
            ones = torch.ones(end - start, dtype=val_dtype, device=device)
            out_degree.scatter_add_(0, src[start:end], ones)
    else:
        ones = torch.ones(edge_count, dtype=val_dtype, device=device)
        out_degree.scatter_add_(0, src, ones)

    csr_offsets = torch.zeros(num_nodes + 1, dtype=torch.long, device=device)
    csr_offsets[1:] = out_degree.to(torch.long).cumsum(0)

    order = src.argsort(stable=True)
    csr_targets = tgt[order]
    del order
    return csr_offsets, csr_targets


def _build_csr_numpy(
    src: torch.Tensor,
    tgt: torch.Tensor,
    num_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build CSR via NumPy sorting for large CPU tensors.

    Parameters
    ----------
    src : torch.Tensor
        Source node indices on CPU with shape ``(E,)``.
    tgt : torch.Tensor
        Target node indices on CPU with shape ``(E,)``.
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``(csr_offsets, csr_targets)`` on CPU.
    """
    import numpy as np

    src_np = src.detach().cpu().numpy()
    tgt_np = tgt.detach().cpu().numpy()

    order = np.argsort(src_np, kind="stable")
    csr_targets = torch.from_numpy(tgt_np[order].copy())

    out_degree = torch.bincount(src.to(torch.long), minlength=num_nodes)
    csr_offsets = torch.zeros(num_nodes + 1, dtype=torch.long)
    csr_offsets[1:] = out_degree.cumsum(0)

    del order
    return csr_offsets, csr_targets


def _process_wave_edges_chunked(
    src: torch.Tensor,
    tgt: torch.Tensor,
    wave_set: torch.Tensor,
    layers: torch.Tensor,
    remaining: torch.Tensor,
    E: int,
) -> torch.Tensor:
    """Process wave edges in chunks and return the touched child nodes.

    Parameters
    ----------
    src : torch.Tensor
        Source indices shaped ``[E]``.
    tgt : torch.Tensor
        Target indices shaped ``[E]``.
    wave_set : torch.Tensor
        Boolean membership mask shaped ``[N]`` for the current frontier.
    layers : torch.Tensor
        Layer assignments shaped ``[N]``.
    remaining : torch.Tensor
        Remaining in-degree counts shaped ``[N]``.
    E : int
        Total edge count.

    Returns
    -------
    torch.Tensor
        Unique child nodes touched by this wave. The caller filters this set
        down to children whose ``remaining`` count just reached zero.

    Notes
    -----
    This streams edge chunks to avoid materializing a full ``[E]`` mask on
    very large graphs where memory headroom matters more than extra kernel
    launches.
    """
    touched_children: List[torch.Tensor] = []
    for start in range(0, E, _EDGE_CHUNK):
        end = min(start + _EDGE_CHUNK, E)
        chunk_src = src[start:end]
        chunk_tgt = tgt[start:end]
        chunk_mask = wave_set[chunk_src]
        children = chunk_tgt[chunk_mask]
        if children.numel() > 0:
            candidate = layers[chunk_src[chunk_mask]] + 1
            layers.scatter_reduce_(0, children, candidate, reduce="amax")
            ones = torch.ones(children.shape[0], dtype=remaining.dtype, device=remaining.device)
            remaining.scatter_add_(0, children, -ones)
            touched_children.append(children.unique())

    if not touched_children:
        return torch.empty(0, dtype=torch.long, device=remaining.device)
    if len(touched_children) == 1:
        return touched_children[0]
    return torch.cat(touched_children).unique()


def _frontier_from_touched_children(
    children: torch.Tensor,
    remaining: torch.Tensor,
) -> torch.Tensor:
    """Build the next BFS frontier from children touched in the prior wave.

    Parameters
    ----------
    children : torch.Tensor
        Child node indices touched by the current wave, shaped ``[K]``.
    remaining : torch.Tensor
        Remaining in-degree counts shaped ``[N]`` after decrements.

    Returns
    -------
    torch.Tensor
        Unique child nodes whose remaining in-degree is now zero.
    """
    if children.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=remaining.device)

    unique_children = children.unique()
    child_remaining = remaining[unique_children]
    return unique_children[child_remaining == 0]


def _longest_path_layering_vectorized(
    edge_index: torch.Tensor,
    num_nodes: int,
    device: str = "cpu",
) -> torch.Tensor:
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
    from collections import deque

    import numpy as np

    N = num_nodes
    E = edge_index.shape[1]

    compute_device = "cpu"
    if device == "cuda" and VRAMBudget.available():
        estimated_bytes = N * 25 + E * 16
        if VRAMBudget().fits(estimated_bytes):
            compute_device = "cuda"

    try:
        src, tgt = edge_index[0], edge_index[1]
        if compute_device == "cuda":
            src = src.to("cuda")
            tgt = tgt.to("cuda")
        chunked = N > _STREAMING_NODE_THRESHOLD

        # Use int32 for working arrays when chunked (saves 12 GB at 1B nodes).
        # Max in-degree and layer index both fit comfortably in int32.
        val_dtype = torch.int32 if chunked else torch.long

        # Compute in-degree — chunked for large graphs to avoid [E]-sized ones tensor
        in_degree = torch.zeros(N, dtype=val_dtype, device=compute_device)
        if chunked:
            for start in range(0, E, _EDGE_CHUNK):
                end = min(start + _EDGE_CHUNK, E)
                chunk_ones = torch.ones(end - start, dtype=val_dtype, device=compute_device)
                in_degree.scatter_add_(0, tgt[start:end], chunk_ones)
        else:
            ones_E = torch.ones(E, dtype=val_dtype, device=compute_device)
            in_degree.scatter_add_(0, tgt, ones_E)

        # Probe: run a few waves to decide strategy
        layers = torch.zeros(N, dtype=val_dtype, device=compute_device)
        remaining = in_degree.clone()
        total_processed = 0
        current_layer = 0
        probe_waves = 10
        frontier = (remaining == 0).nonzero(as_tuple=True)[0]

        # Pre-allocate wave_set once — reuse via .zero_() each wave
        wave_set = torch.zeros(N, dtype=torch.bool, device=compute_device)

        for _ in range(probe_waves):
            if frontier.numel() == 0:
                break
            total_processed += frontier.numel()
            layers[frontier] = current_layer
            remaining[frontier] = -1

            wave_set.zero_()
            wave_set[frontier] = True

            if chunked:
                touched_children = _process_wave_edges_chunked(
                    src,
                    tgt,
                    wave_set,
                    layers,
                    remaining,
                    E,
                )
            else:
                edge_mask = wave_set[src]
                children = tgt[edge_mask]
                if children.numel() > 0:
                    candidate = layers[src[edge_mask]] + 1
                    layers.scatter_reduce_(0, children, candidate, reduce="amax")
                    ones = torch.ones(children.shape[0], dtype=val_dtype, device=compute_device)
                    remaining.scatter_add_(0, children, -ones)
                touched_children = children

            frontier = _frontier_from_touched_children(touched_children, remaining)

            current_layer += 1

        avg_wave = total_processed / max(current_layer, 1)

        # Build CSR for efficient wave processing — O(E log E) from argsort,
        # then O(V+E) total for all remaining waves.
        csr_offsets, csr_targets = _build_csr(src, tgt, N, compute_device)

        # Wide graph: continue with waves (fast when few iterations needed)
        if avg_wave > 1000:
            for _ in range(N):
                if frontier.numel() == 0:
                    break
                layers[frontier] = current_layer
                remaining[frontier] = -1

                wave_starts = csr_offsets[frontier]
                wave_ends = csr_offsets[frontier + 1]
                edge_counts = wave_ends - wave_starts
                total_children = int(edge_counts.sum().item())

                if total_children > 0:
                    wave_expanded = torch.repeat_interleave(frontier, edge_counts)
                    if total_children < 10_000_000:
                        offsets_within = torch.cat(
                            [
                                torch.arange(
                                    int(count.item()),
                                    dtype=torch.long,
                                    device=compute_device,
                                )
                                for count in edge_counts
                            ]
                        )
                    else:
                        offsets_within = _build_flat_indices(
                            wave_starts,
                            edge_counts,
                            total_children,
                            compute_device,
                        )
                    flat_idx = torch.repeat_interleave(wave_starts, edge_counts) + offsets_within
                    children = csr_targets[flat_idx]
                    candidate = layers[wave_expanded] + 1
                    layers.scatter_reduce_(0, children, candidate, reduce="amax")
                    ones = torch.ones(total_children, dtype=val_dtype, device=compute_device)
                    remaining.scatter_add_(0, children, -ones)
                    frontier = _frontier_from_touched_children(children, remaining)
                else:
                    frontier = torch.empty(0, dtype=torch.long, device=compute_device)

                current_layer += 1

            del csr_offsets, csr_targets
            if chunked:
                del remaining, wave_set, in_degree
                return layers
            if layers.dtype != torch.long:
                del remaining, wave_set, in_degree
                layers = layers.long()
            return layers

        # Deep graph: switch to CSR + numpy BFS (true O(V+E))
        # Reset — recompute from scratch with numpy (zero-copy from torch)
        if compute_device == "cuda":
            in_degree = in_degree.cpu()
            csr_offsets = csr_offsets.cpu()
            csr_targets = csr_targets.cpu()
        in_deg = in_degree.numpy().copy()
        csr_tgt = csr_targets.numpy()
        csr_off = csr_offsets.numpy()

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
    except RuntimeError:
        if compute_device != "cuda":
            raise
        torch.cuda.empty_cache()
        return _longest_path_layering_vectorized(edge_index, num_nodes, device="cpu")


def collect_cluster_leaves(cluster_dict) -> List[int]:
    """Recursively collect all leaf node indices from a nested cluster dict."""
    leaves = []
    if isinstance(cluster_dict, list):
        return cluster_dict
    if isinstance(cluster_dict, dict):
        for v in cluster_dict.values():
            leaves.extend(collect_cluster_leaves(v))
    return leaves
