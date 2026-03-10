"""Multilevel coarsening V-cycle for large graph layout.

The single most important technique for scaling beyond 50K nodes.
Coarsen the graph to ~2K nodes, lay out the coarsest graph, then
prolong positions back through each level with brief refinement.

Total work: O(N) if each level does O(N_level) work and reduction ≥ 2x.

Coarsening strategy: layer-aware heavy-edge matching.
- Only merge nodes within the same DAG layer (preserves layered structure)
- Pair adjacent nodes by connectivity (shared parents/children preferred)
- Reduction ratio ~50% per level
- Edges between coarse nodes sum weights for multi-edges
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from dagua.config import LayoutConfig
from dagua.utils import _vram_fits, longest_path_layering


@dataclass
class CoarseLevel:
    """One level of the coarsening hierarchy."""
    edge_index: torch.Tensor       # [2, E_c] coarsened edges
    node_sizes: torch.Tensor       # [N_c, 2]
    num_nodes: int
    fine_to_coarse: torch.Tensor   # [N_fine] maps fine node → coarse node
    num_fine: int                  # N at the finer level
    fine_layer_assignments: Optional[torch.Tensor] = None  # [N_fine] layer assignments for fine level


def coarsen_once(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: torch.Tensor,
    layer_assignments: List[int],
    device: str = "cpu",
) -> CoarseLevel:
    """Single coarsening step via layer-aware heavy-edge matching.

    Within each layer, greedily pair adjacent nodes. Matched pairs merge
    into a single coarse node. Preserves DAG layer structure.

    Fully vectorized — no Python loops over nodes.

    TODO(perf): Streaming coarsening for 1B+ nodes. Current approach holds the
    full edge_hash + argsort in memory (~19GB at 100M). Could process per-layer
    instead of globally (coarsening is already layer-aware): iterate layers,
    assign coarse IDs per-layer, chunk edge dedup in ~10M batches. Would drop
    peak memory from O(N+E) to O(max_layer_size + chunk_size), enabling ~1B
    nodes on 128GB RAM. Estimated effort: 2-3 hours.
    """
    N = num_nodes
    if isinstance(layer_assignments, list):
        layers = torch.tensor(layer_assignments, dtype=torch.long, device=device)
    else:
        layers = layer_assignments.to(device)
    num_layers = int(layers.max().item()) + 1 if N > 0 else 0

    # Sort nodes by layer for contiguous access
    layer_counts = torch.bincount(layers, minlength=num_layers)
    layer_offsets = torch.zeros(num_layers + 1, dtype=torch.long, device=device)
    layer_offsets[1:] = layer_counts.cumsum(0)

    # Build adjacency for matching priority
    match_score = torch.zeros(N, dtype=torch.float32, device=device)
    if edge_index.numel() > 0:
        src, tgt = edge_index[0], edge_index[1]
        in_deg = torch.zeros(N, device=device)
        out_deg = torch.zeros(N, device=device)
        in_deg.scatter_add_(0, tgt, torch.ones(tgt.shape[0], device=device))
        out_deg.scatter_add_(0, src, torch.ones(src.shape[0], device=device))
        match_score = in_deg + out_deg

    # Vectorized greedy matching: sort by (layer, -match_score) globally,
    # then pair consecutive same-layer nodes.
    # Composite sort key: layer dominates, match_score breaks ties (descending)
    score_norm = match_score / (match_score.max() + 1e-8)  # [0, 1)
    sort_key = layers.float() * 2.0 + (1.0 - score_norm)  # layer first, then descending score
    global_order = sort_key.argsort()  # [N] — sorted by (layer, -score)

    # Within each layer group in global_order, consecutive pairs get same coarse ID.
    # Compute within-layer position for each node in sorted order.
    sorted_layers = layers[global_order]  # [N]

    # Within-layer index: 0, 1, 2, ... for each layer
    # Use cumsum trick: positions within layer = global_position - layer_start
    # We can compute this from the sorted layer assignments
    layer_of_sorted = sorted_layers
    layer_start_of_sorted = layer_offsets[layer_of_sorted]  # [N]
    global_pos = torch.arange(N, dtype=torch.long, device=device)
    within_layer_pos = global_pos - layer_start_of_sorted  # [N]

    # Pair index within layer: pos // 3 gives pair number (merge triples for
    # ~67% reduction per level instead of 50%, halving hierarchy depth)
    pair_within_layer = within_layer_pos // 3  # [N]

    # Coarse ID = cumulative pairs up to this layer + pair_within_layer
    # Number of coarse nodes per layer: ceil(layer_count / 3)
    coarse_per_layer = (layer_counts + 2) // 3  # [L]
    coarse_offsets = torch.zeros(num_layers + 1, dtype=torch.long, device=device)
    coarse_offsets[1:] = coarse_per_layer.cumsum(0)

    # Each node's coarse ID
    coarse_base = coarse_offsets[layer_of_sorted]  # [N]
    coarse_ids_sorted = coarse_base + pair_within_layer  # [N]

    # Map back to original node order
    fine_to_coarse = torch.empty(N, dtype=torch.long, device=device)
    fine_to_coarse[global_order] = coarse_ids_sorted

    N_coarse = coarse_offsets[-1].item()

    # Build coarse node sizes (max of merged pair for each dimension)
    coarse_sizes = torch.zeros(N_coarse, 2, device=device)
    coarse_sizes.scatter_reduce_(
        0, fine_to_coarse.unsqueeze(1).expand(-1, 2),
        node_sizes, reduce="amax",
    )

    # Build coarse edges (remap and deduplicate)
    if edge_index.numel() > 0:
        coarse_src = fine_to_coarse[edge_index[0]]
        coarse_tgt = fine_to_coarse[edge_index[1]]

        # Remove self-loops (merged nodes)
        not_self = coarse_src != coarse_tgt
        coarse_src = coarse_src[not_self]
        coarse_tgt = coarse_tgt[not_self]

        if coarse_src.numel() > 0:
            # Deduplicate edges using hash (no inverse — saves a large allocation)
            edge_hash = coarse_src * N_coarse + coarse_tgt
            unique_hash = edge_hash.unique()
            # Recover src, tgt from hash
            unique_src = unique_hash // N_coarse
            unique_tgt = unique_hash % N_coarse
            coarse_edge_index = torch.stack([unique_src, unique_tgt])
        else:
            coarse_edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
    else:
        coarse_edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)

    return CoarseLevel(
        edge_index=coarse_edge_index,
        node_sizes=coarse_sizes,
        num_nodes=N_coarse,
        fine_to_coarse=fine_to_coarse,
        num_fine=N,
    )


def build_hierarchy(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: torch.Tensor,
    min_nodes: int = 2000,
    max_levels: int = 10,
    device: str = "cpu",
) -> List[CoarseLevel]:
    """Build coarsening hierarchy until num_nodes <= min_nodes.

    Returns list of CoarseLevels from finest to coarsest.
    """
    levels: List[CoarseLevel] = []
    current_ei = edge_index
    current_sizes = node_sizes
    current_n = num_nodes

    # Compute layers once on the original graph
    if current_ei.numel() > 0:
        current_layers = longest_path_layering(current_ei, current_n)
    else:
        current_layers = [0] * current_n
    current_la_tensor = torch.tensor(current_layers, dtype=torch.long) if isinstance(current_layers, list) else current_layers

    for _ in range(max_levels):
        if current_n <= min_nodes:
            break

        level = coarsen_once(
            current_ei, current_n, current_sizes,
            layer_assignments=current_layers, device=device,
        )
        level.fine_layer_assignments = current_la_tensor
        levels.append(level)

        # Move to coarser level
        current_ei = level.edge_index
        current_sizes = level.node_sizes
        current_n = level.num_nodes

        # Safety: stop if coarsening didn't reduce enough
        if current_n > level.num_fine * 0.7:
            break

        # Propagate layers: coarse node inherits layer from its fine nodes
        # (all fine nodes in a pair share the same layer by construction)
        coarse_la = torch.zeros(current_n, dtype=torch.long)
        coarse_la.scatter_reduce_(
            0, level.fine_to_coarse, current_la_tensor, reduce="amax",
        )
        current_la_tensor = coarse_la
        current_layers = coarse_la.tolist()

    return levels


def prolong_positions(
    coarse_pos: torch.Tensor,
    level: CoarseLevel,
    device: str = "cpu",
) -> torch.Tensor:
    """Map positions from coarse level back to fine level.

    Each fine node inherits its coarse parent's position with small jitter.
    Uses barycentered placement for nodes with already-placed neighbors.
    """
    fine_pos = coarse_pos[level.fine_to_coarse].clone()

    # Add small random jitter to separate merged nodes
    # Scale jitter by node size to keep it proportional
    jitter_scale = 5.0
    jitter = torch.randn(level.num_fine, 2, device=device) * jitter_scale
    fine_pos = fine_pos + jitter

    return fine_pos


def multilevel_layout(graph, config: LayoutConfig) -> torch.Tensor:
    """Multilevel V-cycle layout for large graphs.

    1. Build coarsening hierarchy
    2. Layout coarsest graph (many steps)
    3. Prolong + refine at each level (few steps)
    """
    import time as _time
    from dagua.layout.engine import ProgressContext, _layout_inner

    verbose = config.verbose
    def _vlog(msg, indent=""):
        if verbose:
            vram = ""
            if device == "cuda" and torch.cuda.is_available():
                used = torch.cuda.memory_allocated() / 1024**2
                free, total = torch.cuda.mem_get_info()
                total_mb = total / 1024**2
                vram = f" [VRAM {used:.0f}MB / {total_mb:.0f}MB]"
            print(f"[dagua] {indent}{msg}{vram}", flush=True)

    def _reset_peak():
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def _stage_peak_mb() -> float:
        if device == "cuda" and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**2
        return 0.0

    device = config.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    n = graph.num_nodes
    _t0 = _time.perf_counter()

    if config.seed is not None:
        torch.manual_seed(config.seed)

    # Build hierarchy on CPU — coarsening uses large temporary tensors
    # (edge_hash.unique() at 50M+ edges OOMs on small GPUs).
    # Keep all graph data on CPU; only move what's needed to GPU per-level.
    cpu_ei = graph.edge_index
    cpu_ns = graph.node_sizes
    min_nodes = config.multilevel_min_nodes
    _t_hier = _time.perf_counter()
    levels = build_hierarchy(cpu_ei, n, cpu_ns, min_nodes=min_nodes, device="cpu")
    _vlog(f"Phase 1/3: Building hierarchy ({n:,} nodes)... {len(levels)} levels ({_time.perf_counter() - _t_hier:.1f}s)")

    if not levels:
        # Graph is already small enough — use direct layout
        ei = cpu_ei.to(device)
        ns = cpu_ns.to(device)
        pos = _layout_inner(ei, n, ns, config, device=device,
                            progress_context=ProgressContext())
        from dagua.layout.engine import _apply_direction
        direction = config.direction if config else graph.direction
        return _apply_direction(pos, direction)

    # Helper to build a config for a given level
    def _make_config(steps, lr=config.lr, seed=config.seed):
        return LayoutConfig(
            steps=steps,
            lr=lr,
            device=device,
            seed=seed,
            verbose=config.verbose,
            node_sep=config.node_sep,
            rank_sep=config.rank_sep,
            w_dag=config.w_dag,
            w_attract=config.w_attract,
            w_attract_x_bias=config.w_attract_x_bias,
            w_repel=config.w_repel,
            w_overlap=config.w_overlap,
            w_crossing=config.w_crossing,
            w_straightness=config.w_straightness,
            w_length_variance=config.w_length_variance,
            exact_repulsion_threshold=config.exact_repulsion_threshold,
            negative_sample_k=config.negative_sample_k,
            per_loss_backward=config.per_loss_backward,
            gradient_checkpointing=config.gradient_checkpointing,
            hybrid_device=config.hybrid_device,
            num_workers=config.num_workers,
        )

    # Layout coarsest graph with many steps.
    # Pass edges on CPU — _layout_inner will stream batches to GPU.
    # Only node_sizes go to GPU (small: [N_coarse, 2]).
    coarsest = levels[-1]
    _vlog(f"Phase 2/3: Coarsest level ({coarsest.num_nodes:,} nodes, {config.multilevel_coarse_steps} steps)")
    _reset_peak()

    coarse_config = _make_config(
        steps=config.multilevel_coarse_steps,
        lr=config.lr * 2,
    )

    pos = _layout_inner(
        coarsest.edge_index,  # stays on CPU
        coarsest.num_nodes,
        coarsest.node_sizes.to(device),
        coarse_config,
        device=device,
        progress_context=ProgressContext(),
    )

    # Prolong + refine through hierarchy (coarsest → finest)
    num_refine_levels = len(levels)
    _vlog(f"Phase 3/3: Refining ({num_refine_levels} levels)")
    for i in range(len(levels) - 1, -1, -1):
        level = levels[i]

        # Determine this level's graph data
        if i == 0:
            fine_ei_cpu = cpu_ei
            fine_sizes_cpu = cpu_ns
            fine_n = n
        else:
            prev_level = levels[i - 1]
            fine_ei_cpu = prev_level.edge_index
            fine_sizes_cpu = prev_level.node_sizes
            fine_n = prev_level.num_nodes

        n_fine_edges = fine_ei_cpu.shape[1] if fine_ei_cpu.numel() > 0 else 0
        base_refine = config.multilevel_refine_steps
        if i == 0:
            # Finest level: double steps for final quality
            refine_steps = base_refine * 2
        elif i <= 2:
            # Near-finest levels: full steps
            refine_steps = base_refine
        else:
            # Coarser levels: half steps (positions refined further at finer levels)
            refine_steps = max(base_refine // 2, 5)

        level_num = len(levels) - i
        _vlog(f"Level {level_num}/{num_refine_levels}: {fine_n:,} nodes ({refine_steps} steps)", indent="  ")
        _reset_peak()

        # Free previous level's GPU memory before allocating new tensors —
        # but only when the next level won't fit alongside current allocations.
        if device == "cuda":
            from dagua.layout.engine import _estimate_gpu_memory
            next_level_mem = _estimate_gpu_memory(fine_n, n_fine_edges, per_loss_bw=True)
            if not _vram_fits(next_level_mem):
                pos = pos.cpu()
                torch.cuda.empty_cache()

        # Prolong: map coarse positions to fine level (on CPU)
        fine_to_coarse = level.fine_to_coarse
        pos_cpu = pos.cpu() if pos.device.type != "cpu" else pos
        fine_pos = pos_cpu[fine_to_coarse].clone()
        jitter = torch.randn(level.num_fine, 2) * 5.0
        fine_pos = fine_pos + jitter

        # Positions + node_sizes to GPU; edges stay on CPU (streamed in batches)
        fine_sizes = fine_sizes_cpu.to(device)
        pos = fine_pos.to(device)

        refine_config = _make_config(steps=refine_steps, seed=None)

        pos = _layout_inner(
            fine_ei_cpu,  # edges on CPU — engine streams batches to GPU
            fine_n, fine_sizes,
            refine_config,
            device=device,
            init_pos=pos,
            layer_assignments=level.fine_layer_assignments,
            progress_context=ProgressContext(indent="    "),
        )

    _vlog(f"Done \u2014 {n:,} nodes in {_time.perf_counter() - _t0:.1f}s")

    # Apply direction transform
    from dagua.layout.engine import _apply_direction
    direction = config.direction if config else graph.direction
    pos = _apply_direction(pos, direction)

    return pos
