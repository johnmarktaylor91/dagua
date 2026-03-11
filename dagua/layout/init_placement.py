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

    # Vectorized path is faster even at N=100 due to tensor ops vs Python loops
    if num_nodes > 100:
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

        num_passes = min(max(15, num_nodes // 5), 40)

        for _pass in range(num_passes):
            # Alternate mean and median heuristics (median is more robust)
            use_median = (_pass % 2 == 1)

            # Forward pass: order by center of parents
            for layer_idx in sorted_layers[1:]:
                nodes = layer_groups[layer_idx]
                centers = []
                for node in nodes:
                    parents = parents_of[node]
                    if parents:
                        vals = sorted(node_order[p] for p in parents)
                        if use_median:
                            mid = len(vals) // 2
                            center = vals[mid] if len(vals) % 2 == 1 else (vals[mid - 1] + vals[mid]) / 2
                        else:
                            center = sum(vals) / len(vals)
                    else:
                        center = node_order[node]
                    centers.append((center, node))
                centers.sort()
                layer_groups[layer_idx] = [n for _, n in centers]

            _update_node_order(node_order, layer_groups, sorted_layers)

            # Backward pass: order by center of children
            for layer_idx in reversed(sorted_layers[:-1]):
                nodes = layer_groups[layer_idx]
                centers = []
                for node in nodes:
                    kids = children_of[node]
                    if kids:
                        vals = sorted(node_order[k] for k in kids)
                        if use_median:
                            mid = len(vals) // 2
                            center = vals[mid] if len(vals) % 2 == 1 else (vals[mid - 1] + vals[mid]) / 2
                        else:
                            center = sum(vals) / len(vals)
                    else:
                        center = node_order[node]
                    centers.append((center, node))
                centers.sort()
                layer_groups[layer_idx] = [n for _, n in centers]

            _update_node_order(node_order, layer_groups, sorted_layers)

        # Transpose heuristic — swap adjacent nodes if it reduces crossings
        if num_nodes <= 500:
            _transpose_heuristic(layer_groups, sorted_layers, children_of, parents_of, num_passes=8)
        elif num_nodes <= 2000:
            _transpose_heuristic(layer_groups, sorted_layers, children_of, parents_of, num_passes=3)

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

    # Post-pass: spread children of high-degree (fan-out) hubs
    if edge_index.numel() > 0:
        _spread_fanout_children(positions, edge_index, node_sizes_cpu, node_sep)

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

    For N > 10K with edges: uses spectral initialization (Fiedler vector via lobpcg)
    for x-coordinates, which captures the graph's natural left-right structure.

    For N <= 10K or no edges: uses tensor-based barycenter ordering.

    Y-coordinates always from layer assignments.
    """
    N = num_nodes
    layer_t = layers.to(dtype=torch.long, device=device) if isinstance(layers, torch.Tensor) else torch.tensor(layers, dtype=torch.long, device=device)
    num_layers = int(layer_t.max().item()) + 1 if N > 0 else 0

    # Build layer structure
    counts = torch.bincount(layer_t, minlength=num_layers)
    offsets = torch.zeros(num_layers + 1, dtype=torch.long, device=device)
    offsets[1:] = counts.cumsum(0)

    # Sort nodes by layer for contiguous access
    sorted_by_layer = layer_t.argsort()

    # Try spectral init for large graphs — provides globally-informed x-coordinates.
    # Skip if edge count is extreme (dense coarsened graphs from multilevel).
    spectral_order = None
    n_edges = edge_index.shape[1] if edge_index.numel() > 0 else 0
    if N > 10000 and N <= 2_000_000 and n_edges > 0 and n_edges < N * 10:
        spectral_order = _spectral_order(edge_index, N, device)

    if spectral_order is not None:
        # Use spectral ordering within each layer
        order = _spectral_to_layer_order(spectral_order, layer_t, counts, offsets, sorted_by_layer, N, device)
    else:
        # Fallback: barycenter ordering
        order = _barycenter_order(edge_index, N, layer_t, counts, offsets, sorted_by_layer, device)

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

    # Post-pass: spread children of high-degree (fan-out) hubs
    if edge_index.numel() > 0:
        _spread_fanout_children(positions, edge_index, node_sizes, node_sep)

    return positions


def _spectral_order(
    edge_index: torch.Tensor,
    num_nodes: int,
    device: str,
) -> Optional[torch.Tensor]:
    """Compute Fiedler vector (2nd eigenvector of graph Laplacian) via lobpcg.

    Returns [N] tensor of spectral coordinates, or None if computation fails.
    The Fiedler vector captures the graph's natural left-right partitioning.
    """
    N = num_nodes
    src = edge_index[0].to(device)
    tgt = edge_index[1].to(device)

    # Build symmetric adjacency (DAG → undirected for Laplacian)
    all_src = torch.cat([src, tgt])
    all_tgt = torch.cat([tgt, src])

    # Degree vector
    degree = torch.zeros(N, device=device)
    degree.scatter_add_(0, all_src, torch.ones(all_src.shape[0], device=device))

    # Build sparse Laplacian: L = D - A
    # Using sparse COO format
    indices = torch.stack([all_src, all_tgt])
    values = -torch.ones(all_src.shape[0], device=device)

    # Add diagonal (degree)
    diag_idx = torch.arange(N, device=device)
    indices = torch.cat([indices, torch.stack([diag_idx, diag_idx])], dim=1)
    values = torch.cat([values, degree])

    L = torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()

    # lobpcg to find 2 smallest eigenvalues (Fiedler = 2nd smallest)
    try:
        # Random initial vectors
        X0 = torch.randn(N, 2, device=device)
        eigenvalues, eigenvectors = torch.lobpcg(L, k=2, X=X0, largest=False, niter=30)
        # Fiedler vector is the 2nd eigenvector (1st is constant)
        fiedler = eigenvectors[:, 1]
        return fiedler
    except Exception:
        # lobpcg can fail on disconnected or degenerate graphs
        return None


def _spectral_to_layer_order(
    spectral: torch.Tensor,
    layer_t: torch.Tensor,
    counts: torch.Tensor,
    offsets: torch.Tensor,
    sorted_by_layer: torch.Tensor,
    num_nodes: int,
    device: str,
) -> torch.Tensor:
    """Convert spectral coordinates to within-layer ordering.

    Within each layer, sort nodes by their spectral coordinate to get
    sequential positions (0, 1, 2, ...).
    """
    N = num_nodes
    order = torch.zeros(N, device=device)

    # Composite sort key: layer * (N+1) + rank_in_spectral
    # First, normalize spectral to [0, N) for stable sorting
    s_min = spectral.min()
    s_range = spectral.max() - s_min + 1e-8
    spectral_norm = (spectral - s_min) / s_range * N

    sort_key = layer_t.float() * (N + 1) + spectral_norm
    global_sorted = sort_key.argsort()

    # Assign sequential positions within each layer
    sorted_layers = layer_t[global_sorted]
    layer_starts_expanded = offsets[sorted_layers]
    positions_in_sort = torch.arange(N, device=device)
    within_layer_pos = (positions_in_sort - layer_starts_expanded).float()
    order[global_sorted] = within_layer_pos

    return order


def _barycenter_order(
    edge_index: torch.Tensor,
    num_nodes: int,
    layer_t: torch.Tensor,
    counts: torch.Tensor,
    offsets: torch.Tensor,
    sorted_by_layer: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Tensor-based barycenter ordering for medium graphs."""
    N = num_nodes

    # Initialize order values: position within initial layer grouping
    order = torch.zeros(N, device=device)
    num_layers = counts.shape[0]
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
        num_passes = 12  # more passes for better initial ordering
        for _pass in range(num_passes):
            # Forward pass: each node's order = mean of parents' orders
            parent_sum = torch.zeros(N, device=device)
            parent_sum.scatter_add_(0, tgt, order[src])
            has_parents = in_degree > 0
            new_order = torch.where(has_parents, parent_sum / in_degree.clamp(min=1), order)

            # Sort within each layer by new_order (composite key trick)
            sort_key = layer_t.float() * (N + 1) + new_order
            global_sorted = sort_key.argsort()
            sorted_layers = layer_t[global_sorted]
            layer_starts_expanded = offsets[sorted_layers]
            positions_in_sort = torch.arange(N, device=device)
            within_layer_pos = (positions_in_sort - layer_starts_expanded).float()
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

    return order


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


def _spread_fanout_children(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    node_sep: float,
    degree_threshold: int = 8,
) -> None:
    """Re-spread children of high-degree hub nodes in a wider arc.

    After barycenter ordering, hub children may be clustered too tightly.
    This post-pass detects hubs (out_degree >= threshold) and re-distributes
    their children symmetrically around the hub's x-coordinate.

    Modifies positions in-place.
    """
    if edge_index.numel() == 0:
        return

    src = edge_index[0].tolist()
    tgt = edge_index[1].tolist()
    N = positions.shape[0]

    # Compute out-degree
    out_degree: Dict[int, int] = defaultdict(int)
    children_of: Dict[int, List[int]] = defaultdict(list)
    for s, t in zip(src, tgt):
        out_degree[s] += 1
        children_of[s].append(t)

    for hub, degree in out_degree.items():
        if degree < degree_threshold:
            continue

        children = children_of[hub]
        k = len(children)
        hub_x = positions[hub, 0].item()

        # Compute total width needed for even distribution
        child_widths = [node_sizes[c, 0].item() for c in children]
        total_width = sum(child_widths) + node_sep * (k - 1)
        # Widen by 1.5x for breathing room
        total_width *= 1.5

        # Sort children by current x to preserve relative ordering
        children_sorted = sorted(children, key=lambda c: positions[c, 0].item())

        # Distribute evenly centered on hub_x
        x_start = hub_x - total_width / 2
        x_cursor = x_start
        for c in children_sorted:
            w = node_sizes[c, 0].item()
            positions[c, 0] = x_cursor + w / 2
            x_cursor += w + node_sep * 1.5


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
