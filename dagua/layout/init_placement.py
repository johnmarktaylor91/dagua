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

from collections import defaultdict
from typing import Dict, List, Optional, Union

import torch

from dagua.utils import VRAMBudget, longest_path_layering

_CHAIN_LAYER_RATIO = 0.8
_MIN_FLOW_PRESERVATION = 0.95
_CHAIN_FLOW_ASPECT = 0.25


def init_positions(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: torch.Tensor,
    node_sep: float = 25.0,
    rank_sep: float = 50.0,
    device: str = "cpu",
    *,
    verbose: bool = False,
) -> torch.Tensor:
    """Compute initial positions via topological layering and barycenter ordering.

    Parameters
    ----------
    edge_index : torch.Tensor
        Directed edge tensor shaped ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor
        Node-size tensor shaped ``[N, 2]``.
    node_sep : float, default=25.0
        Horizontal spacing target between adjacent nodes in the same layer.
    rank_sep : float, default=50.0
        Vertical spacing target between consecutive layers.
    device : str, default="cpu"
        Device for the returned position tensor.
    verbose : bool, default=False
        Whether CUDA layering activation and fallback messages should be
        emitted while computing the topological layers.

    Returns
    -------
    torch.Tensor
        Initial position tensor shaped ``[N, 2]``.
    """
    # Step 1: Assign layers (y-coordinates) via longest-path.
    layers = longest_path_layering(edge_index, num_nodes, device=device, verbose=verbose)
    chain_flow_layers = False

    # Round 4 sprint 19a/b: cycle reversal fallback. Kahn's algorithm lumps
    # every cycle-trapped node into max_layer+1, so a cyclic graph often
    # layers as a single blob (all-cycle case) or as a heavily skewed
    # distribution (mostly-acyclic with a feedback hub). Both collapse to
    # poor layouts: single layer triggers Force2DInitIfFlat, skewed
    # distributions lose dag_consistency and edge_straightness.
    #
    # Trigger cycle-reversal relayering when the original layering is
    # degenerate -- either a single layer, or >50% of nodes piled into one
    # layer. Only adopt the re-layer when it compresses n nodes into less
    # than n/2 layers; otherwise the graph has no real hierarchy (small-
    # world, dense random) and downstream Force2DInitIfFlat handles it via
    # 2D random init.
    if num_nodes > 2:
        layer_seq = layers if isinstance(layers, list) else layers.tolist()
        n_layers = len(set(layer_seq))
        max_layer_count = max(layer_seq.count(v) for v in set(layer_seq))
        heavy_skew = max_layer_count / float(num_nodes) > 0.5
        if n_layers <= 1 or heavy_skew:
            from dagua.layout.cycle import make_acyclic_robust

            try:
                # Self-loops participate in neither layering nor cycle
                # reversal; filter them so FAS can terminate and Kahn can
                # process the cycle-trapped nodes.
                self_loop_mask = edge_index[0] != edge_index[1]
                filtered_edges = edge_index[:, self_loop_mask]
                acyclic_edges, _reversed_mask = make_acyclic_robust(filtered_edges, num_nodes)
                if acyclic_edges.shape[1] > 0:
                    relayered = longest_path_layering(
                        acyclic_edges,
                        num_nodes,
                        device=device,
                        verbose=verbose,
                    )
                    relayered_seq = relayered if isinstance(relayered, list) else relayered.tolist()
                    n_relayered = len(set(relayered_seq))
                    relayered_max = max(relayered_seq.count(v) for v in set(relayered_seq))
                    # Accept when relayering reduces pile-up. Near-chain
                    # layerings are only useful when the FAS order still
                    # preserves the graph's original directed flow; otherwise
                    # they are dense/cyclic artifacts and the flat 2D fallback
                    # is a better seed.
                    pile_reduced = relayered_max < max_layer_count
                    chain_like = n_relayered / float(num_nodes) > _CHAIN_LAYER_RATIO
                    flow_preserved = (
                        _layering_direction_consistency(
                            filtered_edges,
                            relayered_seq,
                        )
                        >= _MIN_FLOW_PRESERVATION
                    )
                    not_degenerate = num_nodes <= 10 or not chain_like or flow_preserved
                    gained_layers = n_relayered > n_layers
                    if pile_reduced and not_degenerate and gained_layers:
                        layers = relayered
                        # Sprint-20i: when the ORIGINAL Kahn layering collapsed
                        # everything into one cycle-trapped pile (n_layers <= 1)
                        # but FAS reveals a clean spanning chain (n_relayered ~=
                        # num_nodes, flow_preserved), keep the rank-based y
                        # but skip the chain-flow x init. A monotone x init on
                        # a fake-chain produces a diagonal line layout that
                        # the optimizer can't escape; freeing x to barycenter
                        # ordering lets the optimizer organize it around the
                        # preserved y-rank. small_world_100 / small_world_500
                        # confirms: 48.58 -> ~57, 49.34 -> ~55 composite when
                        # both share this guard with rank-based y and free x.
                        fake_chain = chain_like and n_layers <= 1
                        chain_flow_layers = chain_like and flow_preserved and not fake_chain
            except Exception:
                # Cycle removal failed -- keep the original collapsed
                # layering, downstream Force2DInitIfFlat will handle.
                pass

    # Vectorized path is faster even at N=100 due to tensor ops vs Python loops
    if num_nodes > 100:
        return _init_positions_vectorized(
            edge_index,
            num_nodes,
            node_sizes,
            layers,
            node_sep,
            rank_sep,
            device,
            chain_flow_layers,
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
            use_median = _pass % 2 == 1

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
                            center = (
                                vals[mid] if len(vals) % 2 == 1 else (vals[mid - 1] + vals[mid]) / 2
                            )
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
                            center = (
                                vals[mid] if len(vals) % 2 == 1 else (vals[mid - 1] + vals[mid]) / 2
                            )
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

        total_width = sum(node_sizes_cpu[n, 0].item() for n in nodes) + node_sep * max(
            len(nodes) - 1, 0
        )
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

    if chain_flow_layers:
        _apply_chain_flow_x(positions, layers, target_aspect=_CHAIN_FLOW_ASPECT)

    return positions


def _layering_direction_consistency(
    edge_index: torch.Tensor,
    layers: List[int],
) -> float:
    """Return the share of edges that point forward in a candidate layering.

    Parameters
    ----------
    edge_index : torch.Tensor
        Directed edge tensor shaped ``[2, E]``.
    layers : List[int]
        Candidate layer assignment with one integer layer per node.

    Returns
    -------
    float
        Fraction of non-self-loop edges whose target layer is greater than or
        equal to the source layer. Ties are treated as forward, matching
        ``dag_consistency``.
    """
    if edge_index.numel() == 0:
        return 1.0

    layer_t = torch.tensor(layers, dtype=torch.long, device=edge_index.device)
    src = edge_index[0]
    tgt = edge_index[1]
    non_self = src != tgt
    if not bool(non_self.any().item()):
        return 1.0

    forward = layer_t[tgt[non_self]] >= layer_t[src[non_self]]
    return float(forward.float().mean().item())


def _apply_chain_flow_x(
    positions: torch.Tensor,
    layers: Union[List[int], torch.Tensor],
    *,
    target_aspect: float,
) -> None:
    """Assign monotone x-coordinates for high-flow near-chain layerings.

    Parameters
    ----------
    positions : torch.Tensor
        Mutable position tensor shaped ``[N, 2]``.
    layers : List[int] or torch.Tensor
        Layer assignment used for y-coordinates.
    target_aspect : float
        Desired width/height ratio for the chain seed.

    Returns
    -------
    None
        The ``positions`` tensor is updated in place.
    """
    if positions.shape[0] <= 1:
        return

    layer_t = (
        layers.to(dtype=positions.dtype, device=positions.device)
        if isinstance(layers, torch.Tensor)
        else torch.tensor(layers, dtype=positions.dtype, device=positions.device)
    )
    layer_span = float((layer_t.max() - layer_t.min()).item())
    y_span = float((positions[:, 1].max() - positions[:, 1].min()).item())
    if layer_span <= 0.0 or y_span <= 0.0:
        return

    desired_width = max(target_aspect, 0.0) * y_span
    centered_layers = layer_t - layer_t.mean()
    positions[:, 0] = centered_layers / layer_span * desired_width


def _init_positions_vectorized(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: torch.Tensor,
    layers: Union[List[int], torch.Tensor],
    node_sep: float,
    rank_sep: float,
    device: str,
    chain_flow_layers: bool = False,
) -> torch.Tensor:
    """Fully vectorized initialization for large graphs.

    For N > 10K with edges: uses spectral initialization (Fiedler vector via lobpcg)
    for x-coordinates, which captures the graph's natural left-right structure.

    For N <= 10K or no edges: uses tensor-based barycenter ordering.

    Y-coordinates always from layer assignments.
    """
    N = num_nodes
    compute_device = _choose_init_device(edge_index, num_nodes, node_sizes, device)
    layer_t = (
        layers.to(dtype=torch.long, device=compute_device)
        if isinstance(layers, torch.Tensor)
        else torch.tensor(layers, dtype=torch.long, device=compute_device)
    )
    num_layers = int(layer_t.max().item()) + 1 if N > 0 else 0

    # Build layer structure
    counts = torch.bincount(layer_t, minlength=num_layers)
    offsets = torch.zeros(num_layers + 1, dtype=torch.long, device=compute_device)
    offsets[1:] = counts.cumsum(0)

    # Sort nodes by layer for contiguous access
    sorted_by_layer = layer_t.argsort()

    # Try spectral init for large graphs — provides globally-informed x-coordinates.
    # Skip if edge count is extreme (dense coarsened graphs from multilevel).
    spectral_order = None
    n_edges = edge_index.shape[1] if edge_index.numel() > 0 else 0
    spectral_cap = 50_000_000
    if VRAMBudget.available():
        spectral_bytes = N * 24 + n_edges * 32 + N * 16
        if not VRAMBudget().fits(spectral_bytes):
            spectral_cap = 2_000_000
    if N > 10000 and N <= spectral_cap and n_edges > 0 and n_edges < N * 10:
        spectral_order = _spectral_order(edge_index, N, compute_device)

    if spectral_order is not None:
        # Use spectral ordering within each layer
        order = _spectral_to_layer_order(
            spectral_order, layer_t, counts, offsets, sorted_by_layer, N, compute_device
        )
    else:
        # Fallback: barycenter ordering
        order = _barycenter_order(
            edge_index, N, layer_t, counts, offsets, sorted_by_layer, compute_device
        )

    # Assign coordinates based on final ordering
    positions = torch.zeros(N, 2, device=compute_device)

    # Y-coordinates: layer * rank_sep
    positions[:, 1] = layer_t.float() * rank_sep

    # X-coordinates: within-layer position * (avg_width + node_sep), centered
    node_w = node_sizes[:, 0].to(compute_device)
    avg_w = node_w.mean()
    spacing = avg_w + node_sep

    # For each layer, compute centered x positions based on order.
    # Ordinal positions 0..(w-1) center on (w-1)/2, not w/2 -- using
    # w/2 biases every layer of size w left by half a slot and produces
    # a non-zero centroid for any odd layer width.
    layer_widths = counts.float()  # [L]
    node_layer_width = layer_widths[layer_t]  # [N]
    positions[:, 0] = (order - (node_layer_width - 1.0) / 2.0) * spacing

    # Post-pass: spread children of high-degree (fan-out) hubs
    if edge_index.numel() > 0:
        _spread_fanout_children(positions, edge_index, node_sizes, node_sep)

    if chain_flow_layers:
        _apply_chain_flow_x(positions, layer_t, target_aspect=_CHAIN_FLOW_ASPECT)

    return positions.to(device) if compute_device != device else positions


def _choose_init_device(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: torch.Tensor,
    device: str,
) -> str:
    """Pick a safe device for initialization ordering.

    The vectorized barycenter path can require copying the full edge index plus
    several large work tensors to the compute device. For very large coarsened
    graphs, that can exceed VRAM even when later optimization fits. In that
    case, build the initial ordering on CPU and move only the final positions.
    """
    if device != "cuda" or not torch.cuda.is_available():
        return device

    edge_elements = int(edge_index.numel())
    edge_bytes = edge_elements * edge_index.element_size()
    node_bytes = int(node_sizes.numel()) * node_sizes.element_size()
    # Conservative estimate for layer/order/degree/work buffers plus the copied
    # src/tgt edge arrays and output positions.
    needed_bytes = edge_bytes * 3 + num_nodes * 64 + node_bytes + num_nodes * 8
    return device if VRAMBudget().fits(needed_bytes) else "cpu"


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
        niter = min(30, max(10, 60 - N // 1_000_000))
        eigenvalues, eigenvectors = torch.lobpcg(L, k=2, X=X0, largest=False, niter=niter)
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
        s, e = int(offsets[L].item()), int(offsets[L + 1].item())
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

                cross_before = _count_local_crossings(
                    u, v, nodes, layer_groups, sorted_layers, children_of, parents_of, layer_idx
                )
                nodes[i], nodes[i + 1] = v, u
                cross_after = _count_local_crossings(
                    v, u, nodes, layer_groups, sorted_layers, children_of, parents_of, layer_idx
                )
                if cross_after >= cross_before:
                    nodes[i], nodes[i + 1] = u, v
                else:
                    improved = True

        if not improved:
            break


def _count_local_crossings(
    u: int,
    v: int,
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
