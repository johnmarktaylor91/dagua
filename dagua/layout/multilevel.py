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
from typing import Callable, List, Optional, Tuple, Union

import torch

from dagua.config import LayoutConfig
from dagua.utils import _EDGE_CHUNK, _vram_fits, longest_path_layering

_STREAMING_THRESHOLD = 100_000_000


@dataclass
class CoarseLevel:
    """One level of the coarsening hierarchy."""
    edge_index: torch.Tensor       # [2, E_c] coarsened edges
    node_sizes: torch.Tensor       # [N_c, 2]
    num_nodes: int
    fine_to_coarse: torch.Tensor   # [N_fine] maps fine node → coarse node
    num_fine: int                  # N at the finer level
    fine_layer_assignments: Optional[torch.Tensor] = None  # [N_fine] layer assignments for fine level


def _can_prolong_on_gpu(
    pos: torch.Tensor,
    fine_to_coarse: torch.Tensor,
    fine_n: int,
    device: str,
) -> bool:
    """Return whether prolongation can stay on GPU with conservative headroom."""
    if device != "cuda" or pos.device.type != "cuda":
        return False
    if not torch.cuda.is_available():
        return False
    needed_bytes = fine_to_coarse.numel() * fine_to_coarse.element_size()
    needed_bytes += fine_n * pos.shape[1] * pos.element_size()
    needed_bytes += fine_n * pos.shape[1] * pos.element_size()
    return _vram_fits(needed_bytes, safety=0.65)


def _coarsen_once_streaming(
    edge_index: torch.Tensor,
    N: int,
    node_sizes: torch.Tensor,
    layers: torch.Tensor,
    num_layers: int,
    layer_counts: torch.Tensor,
    layer_offsets: torch.Tensor,
    device: str = "cpu",
) -> CoarseLevel:
    """Streaming coarsening for 1B+ node graphs.

    Processes edges in chunks and matches nodes per-layer to avoid
    materializing full [N]- or [E]-sized temporaries. Peak memory ~60 GB
    at 1B nodes (vs ~100 GB for the vectorized path).
    """
    E = edge_index.shape[1] if edge_index.numel() > 0 else 0
    index_dtype = torch.int32 if N <= torch.iinfo(torch.int32).max else torch.long

    # --- Phase A: Per-layer node matching ---
    # Compute min_neighbor via chunked scatter_reduce (avoids [E]-sized ones).
    # Nodes sharing a low-index neighbor become consecutive after sort →
    # grouped into the same coarse node → shared edges collapse.
    min_neighbor = torch.full((N,), N, dtype=index_dtype, device=device)
    if E > 0:
        src_all = edge_index[0].to(dtype=index_dtype)
        tgt_all = edge_index[1].to(dtype=index_dtype)
        for start in range(0, E, _EDGE_CHUNK):
            end = min(start + _EDGE_CHUNK, E)
            min_neighbor.scatter_reduce_(0, src_all[start:end], tgt_all[start:end], reduce="amin")
            min_neighbor.scatter_reduce_(0, tgt_all[start:end], src_all[start:end], reduce="amin")

    # Coarse node counts per layer
    coarse_per_layer = (layer_counts + 2) // 3
    coarse_offsets = torch.zeros(num_layers + 1, dtype=index_dtype, device=device)
    coarse_offsets[1:] = coarse_per_layer.cumsum(0)
    N_coarse = coarse_offsets[-1].item()

    # Assign coarse IDs per-layer using boolean masking (no global argsort).
    # Reuses a [N] bool mask (~1 GB at 1B) instead of sorted_by_layer (8+8 GB).
    fine_to_coarse = torch.empty(N, dtype=index_dtype, device=device)
    layer_mask = torch.empty(N, dtype=torch.bool, device=device)
    for layer_idx in range(num_layers):
        if layer_counts[layer_idx].item() == 0:
            continue
        torch.eq(layers, layer_idx, out=layer_mask)
        layer_nodes = layer_mask.nonzero(as_tuple=True)[0]
        local_order = min_neighbor[layer_nodes].argsort()  # ascending
        n_layer = layer_nodes.shape[0]
        coarse_base = coarse_offsets[layer_idx].item()
        fine_to_coarse[layer_nodes[local_order]] = (
            torch.arange(n_layer, dtype=index_dtype, device=device) // 3 + coarse_base
        )

    del layer_mask, min_neighbor

    # --- Phase B: Coarse node sizes ---
    coarse_sizes = torch.zeros(N_coarse, 2, dtype=node_sizes.dtype, device=device)
    coarse_sizes.scatter_reduce_(
        0, fine_to_coarse.unsqueeze(1).expand(-1, 2),
        node_sizes, reduce="amax",
    )

    # --- Phase C: Chunked edge dedup ---
    if E > 0:
        running_unique: Optional[torch.Tensor] = None
        for start in range(0, E, _EDGE_CHUNK):
            end = min(start + _EDGE_CHUNK, E)
            chunk_src = fine_to_coarse[edge_index[0, start:end]].long()
            chunk_tgt = fine_to_coarse[edge_index[1, start:end]].long()
            not_self = chunk_src != chunk_tgt
            chunk_src = chunk_src[not_self]
            chunk_tgt = chunk_tgt[not_self]
            if chunk_src.numel() > 0:
                chunk_hash = chunk_src * N_coarse + chunk_tgt
                chunk_unique = chunk_hash.unique()
                if running_unique is None:
                    running_unique = chunk_unique
                else:
                    # Incremental merge avoids holding all per-chunk uniques plus
                    # a final concatenation at once, which spikes memory at 1B+ scale.
                    running_unique = torch.cat([running_unique, chunk_unique]).unique()
                del chunk_hash
            del chunk_src, chunk_tgt

        if running_unique is not None:
            unique_src = running_unique // N_coarse
            unique_tgt = running_unique % N_coarse
            del running_unique
            coarse_edge_index = torch.stack([unique_src, unique_tgt])
            del unique_src, unique_tgt
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


def coarsen_once(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: torch.Tensor,
    layer_assignments: Union[List[int], torch.Tensor],
    device: str = "cpu",
) -> CoarseLevel:
    """Single coarsening step via layer-aware heavy-edge matching.

    Within each layer, greedily pair adjacent nodes. Matched pairs merge
    into a single coarse node. Preserves DAG layer structure.

    Fully vectorized — no Python loops over nodes. For N > 100M, dispatches
    to streaming path that processes edges in chunks.
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

    # Dispatch to streaming path for very large graphs
    if N > _STREAMING_THRESHOLD:
        return _coarsen_once_streaming(
            edge_index, N, node_sizes, layers, num_layers,
            layer_counts, layer_offsets, device,
        )

    # Build adjacency for matching priority: sort by minimum neighbor index
    # so nodes sharing a low-index neighbor become consecutive → grouped into
    # the same coarse node → shared edges collapse during deduplication.
    min_neighbor = torch.full((N,), N, dtype=torch.long, device=device)
    if edge_index.numel() > 0:
        src, tgt = edge_index[0], edge_index[1]
        min_neighbor.scatter_reduce_(0, src, tgt, reduce="amin")
        min_neighbor.scatter_reduce_(0, tgt, src, reduce="amin")

    # Composite sort key: (layer, min_neighbor) — int64, layer dominates
    sort_key = layers * N + min_neighbor
    global_order = sort_key.argsort()

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
    coarse_sizes = torch.zeros(N_coarse, 2, dtype=node_sizes.dtype, device=device)
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
    progress: Optional[Callable[[str], None]] = None,
) -> List[CoarseLevel]:
    """Build coarsening hierarchy until num_nodes <= min_nodes.

    Returns list of CoarseLevels from finest to coarsest.
    """
    levels: List[CoarseLevel] = []
    current_ei = edge_index
    current_sizes = node_sizes
    current_n = num_nodes

    # Compute layers once on the original graph — returns tensor for large N
    if progress is not None:
        progress(f"Layering full graph ({current_n:,} nodes)...")
    if current_ei.numel() > 0:
        current_la = longest_path_layering(current_ei, current_n)
    else:
        current_la = torch.zeros(current_n, dtype=torch.long)
    # Ensure tensor throughout — no list conversion
    if isinstance(current_la, list):
        current_la = torch.tensor(current_la, dtype=torch.long)
    if progress is not None:
        max_layer = int(current_la.max().item()) if current_la.numel() > 0 else 0
        progress(f"Layering done ({max_layer + 1:,} layers)")

    for level_idx in range(max_levels):
        if current_n <= min_nodes:
            break

        prev_edge_count = current_ei.shape[1] if current_ei.numel() > 0 else 0
        if progress is not None:
            progress(
                f"Coarsen level {level_idx + 1}: "
                f"{current_n:,} nodes, {prev_edge_count:,} edges"
            )

        level = coarsen_once(
            current_ei, current_n, current_sizes,
            layer_assignments=current_la, device=device,
        )
        level.fine_layer_assignments = current_la
        levels.append(level)

        # Move to coarser level
        current_ei = level.edge_index
        current_sizes = level.node_sizes
        current_n = level.num_nodes
        coarse_edge_count = current_ei.shape[1] if current_ei.numel() > 0 else 0
        if progress is not None:
            progress(
                f"Coarsen level {level_idx + 1} done: "
                f"{current_n:,} nodes, {coarse_edge_count:,} edges"
            )

        # Safety: stop if coarsening didn't reduce nodes or edges enough
        if current_n > level.num_fine * 0.7:
            if progress is not None:
                progress("Stopping hierarchy build: node reduction below threshold")
            break
        if prev_edge_count > 0 and coarse_edge_count > prev_edge_count * 0.9:
            if progress is not None:
                progress("Stopping hierarchy build: edge reduction below threshold")
            break  # edges barely reduced — hierarchy won't help

        # Propagate layers: coarse node inherits layer from its fine nodes
        # (all fine nodes in a pair share the same layer by construction)
        coarse_la = torch.zeros(current_n, dtype=current_la.dtype)
        coarse_la.scatter_reduce_(
            0, level.fine_to_coarse, current_la, reduce="amax",
        )
        current_la = coarse_la

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
    levels = build_hierarchy(
        cpu_ei,
        n,
        cpu_ns,
        min_nodes=min_nodes,
        device="cpu",
        progress=(lambda msg: _vlog(msg, indent="  ")) if verbose else None,
    )
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

        # Free this level's own edge_index and node_sizes — they've been consumed.
        # levels[-1]: consumed by coarsest layout (Phase 2).
        # levels[j<-1]: consumed at iteration j+1 as fine_ei_cpu/fine_sizes_cpu.
        # (levels[i-1].edge_index is still alive — consumed THIS iteration below.)
        level.edge_index = None
        level.node_sizes = None

        # Force memory return to OS — glibc holds freed pages otherwise
        if level.num_fine > 100_000_000:
            import ctypes
            import gc as _gc
            _gc.collect()
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except OSError:
                pass

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
        if verbose and fine_n > 100_000_000:
            import gc as _gc
            import os as _os
            _gc.collect()
            try:
                with open("/proc/self/statm") as _f:
                    _rss_pages = int(_f.read().split()[1])
                _rss_gb = _rss_pages * _os.sysconf("SC_PAGE_SIZE") / 1024**3
                print(f"[dagua]     RSS={_rss_gb:.1f} GB before prolongation", flush=True)
            except Exception:
                pass
        _reset_peak()

        # Free previous level's GPU memory before allocating new tensors —
        # but only when the next level won't fit alongside current allocations.
        if device == "cuda":
            from dagua.layout.engine import _estimate_gpu_memory
            next_level_mem = _estimate_gpu_memory(fine_n, n_fine_edges, per_loss_bw=True)
            if not _vram_fits(next_level_mem):
                pos = pos.cpu()
                torch.cuda.empty_cache()

        fine_to_coarse = level.fine_to_coarse
        use_gpu_prolong = _can_prolong_on_gpu(pos, fine_to_coarse, level.num_fine, device)

        if use_gpu_prolong:
            fine_to_coarse_dev = fine_to_coarse.to(device)
            fine_pos = pos[fine_to_coarse_dev]
            fine_pos.add_(torch.randn(level.num_fine, 2, device=device).mul_(5.0))
            del fine_to_coarse_dev
            pos_cpu = None
        else:
            # Prolong on CPU when GPU headroom is uncertain.
            # Use in-place ops to avoid a second fine-position allocation.
            pos_cpu = pos.cpu() if pos.device.type != "cpu" else pos
            fine_pos = pos_cpu[fine_to_coarse]
            fine_pos.add_(torch.randn(level.num_fine, 2).mul_(5.0))

        # Free fine_to_coarse — consumed above, never needed again
        del fine_to_coarse
        level.fine_to_coarse = None

        # Free old pos and pos_cpu — consumed by prolongation above
        if pos_cpu is not None:
            del pos_cpu
        pos = None

        # Positions + node_sizes to GPU; edges stay on CPU (streamed in batches)
        fine_sizes = fine_sizes_cpu.to(device)
        pos = fine_pos if fine_pos.device.type == device else fine_pos.to(device)
        del fine_pos  # pos holds the reference now

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

        # Free this hierarchy level entirely — never revisited
        levels[i] = None

    _vlog(f"Done \u2014 {n:,} nodes in {_time.perf_counter() - _t0:.1f}s")

    # Apply direction transform
    from dagua.layout.engine import _apply_direction
    direction = config.direction if config else graph.direction
    pos = _apply_direction(pos, direction)

    return pos
