"""Core optimization loop — the heart of dagua.

Takes a DaguaGraph and LayoutConfig, returns [N, 2] position tensor.
Headless: operates on tensors extracted from the graph.

Scaling strategy (tiered):
- Tier 0 (N < 500): exact O(N^2) repulsion, full overlap check
- Tier 1 (500-5K): scatter sampling repulsion, layer-local overlap
- Tier 2 (5K-50K): RVS repulsion, reduced passes, adaptive batching
- Tier 3 (N > 50K): multilevel coarsening V-cycle

Memory optimization (composable, auto-enabled for large graphs):
- Per-loss backward: backward each loss term separately (3-4x memory reduction)
- Gradient checkpointing: recompute forward during backward (2x memory, 30% more compute)
- Hybrid device: CPU for heavy losses, GPU for edge losses (enables GPU at 10M+ nodes)

Cross-cutting:
- Pre-compute LayerIndex once, pass to all layer-aware functions
- Stochastic edge batching for O(batch) instead of O(E) per step
- Adaptive overlap projection frequency
- Early stopping on convergence
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from dagua.config import LayoutConfig
from dagua.layout.constraints import (
    cluster_compactness_loss,
    cluster_separation_loss,
    crossing_loss,
    dag_ordering_loss,
    edge_attraction_loss,
    edge_length_variance_loss,
    edge_straightness_loss,
    overlap_avoidance_loss,
    repulsion_loss,
    spacing_consistency_loss,
)
from dagua.layout.init_placement import init_positions
from dagua.layout.layers import LayerIndex, build_layer_index
from dagua.layout.projection import project_overlaps
from dagua.utils import _vram_fits, longest_path_layering


@dataclass
class ProgressContext:
    """Context for formatting engine progress messages."""
    indent: str = "  "


def layout(graph, config: Optional[LayoutConfig] = None) -> torch.Tensor:
    """Compute layout positions for all nodes.

    Args:
        graph: DaguaGraph instance
        config: LayoutConfig (uses defaults if None)

    Returns:
        [N, 2] tensor of (x, y) positions
    """
    if config is None:
        config = LayoutConfig()

    device = config.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Ensure node sizes are computed
    graph.compute_node_sizes()

    n = graph.num_nodes
    if n == 0:
        return torch.zeros(0, 2, device=device)
    if n == 1:
        return torch.zeros(1, 2, device=device)

    # Set seed for determinism
    if config.seed is not None:
        torch.manual_seed(config.seed)
        if device == "cuda":
            torch.cuda.manual_seed(config.seed)

    # Tier 3: Multilevel coarsening for very large graphs
    # Don't move data to GPU yet — multilevel manages device transfers lazily
    if n > config.multilevel_threshold:
        from dagua.layout.multilevel import multilevel_layout
        return multilevel_layout(graph, config)

    # TODO(perf): Small graph speed — graphviz dot beats us on <2K node graphs.
    # Benchmark shows dagua takes 1-2s on tiny graphs (5-20 nodes) due to fixed
    # overhead: init_placement, layer computation, optimizer setup, 500 steps.
    # Investigate: (1) reduce default steps for small N (50 steps may suffice),
    # (2) skip spectral init for N<100, (3) warm-start from topological placement
    # without optimization for very small graphs, (4) profile per-step overhead
    # to find the constant-factor bottleneck.

    # Tier 0-2: Direct layout — move data to device
    edge_index = graph.edge_index.to(device)
    node_sizes = graph.node_sizes.to(device)

    if config.verbose:
        num_edges = edge_index.shape[1] if edge_index.numel() > 0 else 0
        print(f"[dagua] Layout: {n:,} nodes, {num_edges:,} edges", flush=True)

    pos = _layout_inner(
        edge_index, n, node_sizes, config,
        device=device,
        clusters=graph.clusters if hasattr(graph, 'clusters') else None,
        progress_context=ProgressContext(),
    )

    # Apply direction transform
    direction = config.direction if config else graph.direction
    pos = _apply_direction(pos, direction)

    return pos


def _layout_inner(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: torch.Tensor,
    config: LayoutConfig,
    device: str = "cpu",
    init_pos: Optional[torch.Tensor] = None,
    clusters: Optional[dict] = None,
    layer_assignments: Optional[torch.Tensor] = None,
    progress_context: Optional[ProgressContext] = None,
) -> torch.Tensor:
    """Headless optimization loop — operates on raw tensors.

    This is the core engine, usable by both direct layout and multilevel V-cycle.
    No Graph object dependency.

    Args:
        edge_index: [2, E] edge tensor
        num_nodes: number of nodes
        node_sizes: [N, 2] width/height tensor
        config: LayoutConfig with steps, weights, etc.
        device: target device
        init_pos: optional [N, 2] initial positions (for multilevel warm start)
        clusters: optional cluster dict for cluster losses
        layer_assignments: optional pre-computed layer assignments tensor (skips recomputation)
        progress_context: optional context for formatting progress messages

    Returns:
        [N, 2] position tensor (detached)
    """
    import time as _time

    n = num_nodes
    verbose = getattr(config, "verbose", False)
    _indent = progress_context.indent if progress_context else "  "
    def _vlog(msg):
        if verbose:
            vram = ""
            if device == "cuda" and torch.cuda.is_available():
                used = torch.cuda.memory_allocated() / 1024**2
                free, total = torch.cuda.mem_get_info()
                vram = f" [VRAM {used:.0f}MB / {total/1024**2:.0f}MB]"
            print(f"[dagua] {_indent}{msg}{vram}", flush=True)

    if n == 0:
        return torch.zeros(0, 2, device=device)
    if n == 1:
        return torch.zeros(1, 2, device=device)

    # Apply adaptive spacing based on graph size
    node_sep = config.node_sep
    rank_sep = config.rank_sep
    if config.adaptive_spacing:
        node_sep, rank_sep = _adaptive_spacing(n, node_sep, rank_sep)

    # Step 1: Initialization
    if init_pos is not None:
        pos = init_pos.to(device)
    else:
        pos = init_positions(
            edge_index, n, node_sizes,
            node_sep=node_sep,
            rank_sep=rank_sep,
            device=device,
        )

    # Pre-compute layer structure (used by repulsion, overlap, projection, crossing)
    layer_assignments_list: Optional[List[int]] = None
    layer_index: Optional[LayerIndex] = None
    if layer_assignments is not None:
        # Use pre-computed assignments (from multilevel V-cycle)
        layer_index = build_layer_index(layer_assignments, device=device)
        layer_assignments_list = layer_assignments.tolist() if isinstance(layer_assignments, torch.Tensor) else layer_assignments
    elif edge_index.numel() > 0:
        layer_assignments_list = longest_path_layering(edge_index, n)
        layer_index = build_layer_index(layer_assignments_list, device=device)

    # Determine adaptive parameters based on graph size
    num_edges = edge_index.shape[1] if edge_index.numel() > 0 else 0
    edge_batch = _edge_batch_size(num_edges, config)
    overlap_interval = _overlap_interval(n, config)

    # Edge streaming: if edges are too large for GPU, keep them on CPU
    # and stream batches to GPU each step.  This lets positions stay on GPU
    # while edges (which can be 800MB+) stay in CPU RAM.
    edges_on_cpu = edge_index.device.type == "cpu" and device == "cuda"
    if not edges_on_cpu and device == "cuda" and edge_index.numel() > 0:
        edge_bytes = edge_index.numel() * edge_index.element_size()
        if not _vram_fits(edge_bytes):
            edge_index = edge_index.cpu()
            edges_on_cpu = True

    # Resolve memory optimization flags (VRAM-aware when on CUDA)
    use_per_loss_bw, use_checkpointing, use_hybrid = _resolve_memory_strategy(
        n, num_edges, device, config,
    )
    # Create thread pool for parallel hybrid losses
    executor = None
    if use_hybrid and use_per_loss_bw and getattr(config, 'num_workers', 0) > 0:
        from concurrent.futures import ThreadPoolExecutor
        executor = ThreadPoolExecutor(max_workers=config.num_workers)

    flags = []
    if use_per_loss_bw: flags.append("per_loss_bw")
    if use_checkpointing: flags.append("checkpoint")
    if use_hybrid: flags.append("hybrid")
    if edges_on_cpu: flags.append("edge_stream")
    if executor is not None: flags.append(f"workers={config.num_workers}")
    _vlog(f"init done, {num_edges:,} edges, batch={edge_batch}, strategy=[{', '.join(flags) or 'standard'}]")
    # Keep a reference to CPU edges for batching; full edge_index may be CPU
    cpu_edges_ref = edge_index if edges_on_cpu else None

    # Hybrid device: keep positions on GPU, create CPU mirror for heavy losses
    cpu_node_sizes = None
    cpu_layer_index = None
    cpu_edge_index = edge_index if edge_index.device.type == "cpu" else None
    if use_hybrid:
        cpu_node_sizes = node_sizes.cpu()
        if cpu_edge_index is None:
            cpu_edge_index = edge_index.cpu()
        cpu_layer_index = build_layer_index(layer_assignments_list, device="cpu") if layer_assignments_list else None
    elif edges_on_cpu:
        cpu_edge_index = edge_index

    # Step 2: Set up optimization
    pos = pos.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([pos], lr=config.lr)

    # Step 3: Build loss functions ONCE (reuse across steps via mutable refs)
    # batch_edges_ref is a mutable container so lambdas capture the reference
    batch_edges_ref: List[Optional[torch.Tensor]] = [None]

    # Static loss functions — each returns (base_weight_key, loss_fn, is_heavy, is_annealed)
    # base_weight_key is a string to look up the annealed weight at each step
    loss_fns: List[tuple] = []

    if config.w_dag > 0:
        loss_fns.append(("w_dag", lambda p, ns, li: dag_ordering_loss(
            p, batch_edges_ref[0] if p.device == batch_edges_ref[0].device else cpu_edge_index,
            ns, rank_sep), False, True))

    if config.w_attract > 0:
        loss_fns.append(("w_attract", lambda p, ns, li: edge_attraction_loss(
            p, batch_edges_ref[0] if p.device == batch_edges_ref[0].device else cpu_edge_index,
            x_bias=config.w_attract_x_bias), False, False))

    if config.w_repel > 0:
        loss_fns.append(("w_repel", lambda p, ns, li: repulsion_loss(
            p, n,
            threshold=config.exact_repulsion_threshold,
            sample_k=config.negative_sample_k,
            layer_index=li,
            node_sizes=ns,
            rvs_threshold=config.rvs_threshold,
            rvs_nn_k=config.rvs_nn_k,
        ), True, True))

    if config.w_overlap > 0:
        loss_fns.append(("w_overlap", lambda p, ns, li: overlap_avoidance_loss(
            p, ns, layer_index=li,
            rvs_threshold=config.rvs_threshold,
        ), True, True))

    if config.w_cluster > 0 and clusters:
        loss_fns.append(("w_cluster", lambda p, ns, li: cluster_compactness_loss(
            p, clusters, device=p.device), False, False))
        loss_fns.append(("w_cluster_sep", lambda p, ns, li: cluster_separation_loss(
            p, ns, clusters, device=p.device), False, False))

    if config.w_crossing > 0:
        # alpha is annealed per-step, captured via mutable ref
        crossing_alpha_ref: List[float] = [3.0]
        loss_fns.append(("w_crossing", lambda p, ns, li: crossing_loss(
            p, batch_edges_ref[0] if p.device == batch_edges_ref[0].device else cpu_edge_index,
            alpha=crossing_alpha_ref[0], layer_assignments=layer_assignments_list,
        ), False, True))

    if config.w_straightness > 0:
        loss_fns.append(("w_straightness", lambda p, ns, li: edge_straightness_loss(
            p, batch_edges_ref[0] if p.device == batch_edges_ref[0].device else cpu_edge_index), False, True))

    if config.w_length_variance > 0:
        loss_fns.append(("w_length_variance", lambda p, ns, li: edge_length_variance_loss(
            p, batch_edges_ref[0] if p.device == batch_edges_ref[0].device else cpu_edge_index), False, False))

    if config.w_spacing > 0 and layer_index is not None:
        loss_fns.append(("w_spacing", lambda p, ns, li: spacing_consistency_loss(
            p, ns, li, target_gap=node_sep,
        ), False, False))

    # Pre-allocate edge batch buffer (avoids per-step tensor allocation)
    batch_buf = torch.empty(2, edge_batch, dtype=torch.long, device=device) if edge_batch > 0 and num_edges > edge_batch else None

    # Optimization loop with annealing
    steps = config.steps
    prev_unweighted = float("inf")
    stall_count = 0
    _t_loop = _time.perf_counter()
    _log_interval = max(steps // 4, 1)  # log at 25%, 50%, 75%, 100%

    for step in range(steps):
        t = step / max(steps - 1, 1)  # 0 → 1

        if verbose and step > 0 and step % _log_interval == 0:
            _vlog(f"step {step}/{steps} ({100*step//steps}%) loss={prev_unweighted:.2f} [{_time.perf_counter() - _t_loop:.1f}s]")

        optimizer.zero_grad()

        # Sample edge batch for this step — reuse pre-allocated buffer
        if batch_buf is not None:
            perm = torch.randint(0, num_edges, (edge_batch,), device="cpu")
            batch_buf.copy_(edge_index[:, perm])
            batch_edges_ref[0] = batch_buf
        elif edges_on_cpu:
            batch_edges_ref[0] = edge_index.to(device)
        else:
            batch_edges_ref[0] = edge_index

        # Compute annealed weights for this step
        w_dag = config.w_dag * (1 - 0.5 * t)
        w_repel = config.w_repel * (1 + 2 * t)
        w_overlap = config.w_overlap * (1 + t)
        t_cross = min(t / 0.3, 1.0)
        w_crossing = config.w_crossing * t_cross
        w_straightness = config.w_straightness * (1 + 0.5 * t)

        # Update crossing alpha via mutable ref
        if config.w_crossing > 0:
            crossing_alpha_ref[0] = 3.0 + 7.0 * t_cross

        # Map weight keys to current annealed values
        weight_map = {
            "w_dag": w_dag,
            "w_attract": config.w_attract,
            "w_repel": w_repel,
            "w_overlap": w_overlap,
            "w_cluster": config.w_cluster,
            "w_cluster_sep": config.w_cluster * 0.5,
            "w_crossing": w_crossing,
            "w_straightness": w_straightness,
            "w_length_variance": config.w_length_variance,
            "w_spacing": config.w_spacing,
        }

        # Build loss_terms from static functions + current weights
        loss_terms: List[tuple] = []
        for key, loss_fn, is_heavy, _is_annealed in loss_fns:
            w = weight_map[key]
            if w > 0:
                loss_terms.append((w, loss_fn, is_heavy))

        # Execute loss terms with selected memory strategy
        total_loss_val = 0.0
        unweighted_loss_val = 0.0

        if use_per_loss_bw:
            if executor is not None:
                # Parallel: heavy losses on CPU threads, light losses on GPU
                heavy_futures = []
                light_terms_list = []
                for weight, loss_fn, is_heavy in loss_terms:
                    if is_heavy:
                        future = executor.submit(
                            _hybrid_loss, pos, weight, loss_fn,
                            cpu_node_sizes, cpu_layer_index,
                        )
                        heavy_futures.append((weight, future))
                    else:
                        light_terms_list.append((weight, loss_fn))

                # Compute light losses on GPU while heavy run on CPU
                for weight, loss_fn in light_terms_list:
                    term = weight * loss_fn(pos, node_sizes, layer_index)
                    if term is not None and term.requires_grad:
                        term.backward()
                        val = term.item()
                        total_loss_val += val
                        unweighted_loss_val += val / weight if weight else 0.0

                # Gather heavy loss results
                for weight, future in heavy_futures:
                    term = future.result()
                    if term is not None and term.requires_grad:
                        term.backward()
                        val = term.item()
                        total_loss_val += val
                        unweighted_loss_val += val / weight if weight else 0.0
            else:
                # Sequential per-loss backward
                for weight, loss_fn, is_heavy in loss_terms:
                    term = _compute_loss_term(
                        pos, weight, loss_fn, is_heavy,
                        use_hybrid=use_hybrid,
                        use_checkpointing=use_checkpointing,
                        node_sizes=node_sizes,
                        layer_index=layer_index,
                        cpu_node_sizes=cpu_node_sizes if use_hybrid else None,
                        cpu_layer_index=cpu_layer_index if use_hybrid else None,
                    )
                    if term is not None and term.requires_grad:
                        term.backward()
                        val = term.item()
                        total_loss_val += val
                        unweighted_loss_val += val / weight if weight else 0.0
        else:
            # Standard: accumulate all losses, single backward
            loss = torch.tensor(0.0, device=device)
            for weight, loss_fn, is_heavy in loss_terms:
                term = _compute_loss_term(
                    pos, weight, loss_fn, is_heavy,
                    use_hybrid=use_hybrid,
                    use_checkpointing=use_checkpointing,
                    node_sizes=node_sizes,
                    layer_index=layer_index,
                    cpu_node_sizes=cpu_node_sizes if use_hybrid else None,
                    cpu_layer_index=cpu_layer_index if use_hybrid else None,
                )
                if term is not None:
                    loss = loss + term
                    unweighted_loss_val += term.item() / weight if weight else 0.0
            loss.backward()
            total_loss_val = loss.item()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_([pos], max_norm=100.0)

        optimizer.step()

        # Hard overlap projection (adaptive frequency + adaptive iterations)
        if step % overlap_interval == 0 or step == steps - 1:
            proj_iters = 5 if n <= 500_000 else 3 if n <= 5_000_000 else 2
            project_overlaps(
                pos, node_sizes, padding=2.0, iterations=proj_iters,
                layer_index=layer_index,
            )

        # Early stopping check (use unweighted loss — immune to annealing)
        if step > 10 and abs(prev_unweighted - unweighted_loss_val) < prev_unweighted * 1e-4:
            stall_count += 1
            if stall_count >= 5:
                break
        else:
            stall_count = 0
        prev_unweighted = unweighted_loss_val

    _vlog(f"done ({_time.perf_counter() - _t_loop:.1f}s)")

    # Final aggressive overlap projection (scale iterations with N)
    final_proj_iters = 20 if n <= 500_000 else 10 if n <= 5_000_000 else 3
    _vlog(f"final projection ({final_proj_iters} iters)...")
    _t_proj = _time.perf_counter()
    project_overlaps(
        pos, node_sizes, padding=2.0, iterations=final_proj_iters,
        layer_index=layer_index,
    )
    _vlog(f"projection done in {_time.perf_counter() - _t_proj:.1f}s")

    if executor is not None:
        executor.shutdown(wait=False)

    return pos.detach()


def _resolve_memory_strategy(
    n: int,
    num_edges: int,
    device: str,
    config: LayoutConfig,
) -> tuple:
    """Resolve memory optimization flags, VRAM-aware when on CUDA.

    Returns (use_per_loss_bw, use_checkpointing, use_hybrid).

    When all flags are "auto", queries available VRAM and estimates peak memory
    to pick the lightest strategy that fits. When flags are "on"/"off", those
    override the auto logic.
    """
    plb = config.per_loss_backward
    gcp = config.gradient_checkpointing
    hyb = config.hybrid_device

    # Handle explicit on/off overrides
    if plb != "auto" and gcp != "auto" and hyb != "auto":
        return plb == "on", gcp == "on", hyb == "on"

    # CPU mode: per_loss_bw helps with RAM at scale, others don't apply
    if device != "cuda":
        use_plb = (plb == "on") or (plb == "auto" and n > 50000)
        return use_plb, False, False

    # CUDA mode: query available VRAM
    free_vram, total_vram = torch.cuda.mem_get_info()
    usable = int(free_vram * 0.80)  # 20% headroom for fragmentation

    # Estimate peak GPU memory for different strategies
    mem_full = _estimate_gpu_memory(n, num_edges, per_loss_bw=False)
    mem_plb = _estimate_gpu_memory(n, num_edges, per_loss_bw=True)
    mem_plb_ckpt = mem_plb // 2  # checkpointing halves intermediate storage
    mem_hybrid = _estimate_hybrid_gpu_memory(n, num_edges)

    # Pick lightest strategy that fits (escalating memory savings)
    if plb == "off" and gcp == "off" and hyb == "off":
        return False, False, False

    # Level 0: everything on GPU, standard backward
    if mem_full < usable and plb != "on" and gcp != "on" and hyb != "on":
        return False, False, False

    # Level 1: per-loss backward (3-4x reduction, no speed cost)
    use_plb = plb != "off"
    if mem_plb < usable and gcp != "on" and hyb != "on":
        return use_plb, False, False

    # Level 2: per-loss backward + checkpointing (6-8x reduction, ~30% slower)
    use_ckpt = gcp != "off"
    if mem_plb_ckpt < usable and hyb != "on":
        return use_plb, use_ckpt, False

    # Level 3: hybrid device (heavy losses on CPU, edge losses on GPU)
    use_hyb = hyb != "off"
    return use_plb, use_ckpt, use_hyb


def _estimate_gpu_memory(n: int, num_edges: int, per_loss_bw: bool = False) -> int:
    """Estimate peak GPU memory in bytes for full-GPU layout.

    Accounts for: positions, Adam state, gradients, graph data, and
    per-step autograd intermediates (forward + backward retained tensors).

    Includes a 3x safety factor for PyTorch allocator fragmentation,
    autograd graph metadata, and multilevel hierarchy overhead.
    Calibrated against empirical measurements: 5M nodes ≈ 5GB actual.
    """
    # Base allocations (persist across steps)
    # pos + adam (momentum + variance) + grad: 4 * [N,2] f32
    base = n * 2 * 4 * 4  # 32 bytes/node
    # node_sizes [N,2] f32 + layer_index arrays ~3*[N] i64
    base += n * (2 * 4 + 3 * 8)  # 32 bytes/node
    # edge_index [2,E] i64
    base += num_edges * 2 * 8

    # Per-step intermediate tensors (autograd retains for backward)
    n_active = max(int(n ** 0.75), min(n, 256))

    # RVS repulsion: [A, K_total, 2] diffs + [A, K_total] dist + size factors
    k_repul = 70  # n_random + k_nn
    repul = n_active * k_repul * 4 * 8  # ~8 intermediate tensors, f32

    # Active-subset overlap: [A, K, 2] similar structure
    k_overlap = 64
    overlap = n_active * k_overlap * 4 * 8

    # Edge-based losses: [E_batch] tensors for dag, attract, straightness, etc.
    e_batch = min(num_edges, 200000)
    edge_losses = e_batch * 4 * 6 * 4  # ~6 tensors per loss, ~4 edge losses

    # Autograd backward roughly doubles forward intermediates
    autograd_factor = 2

    # Safety factor: PyTorch allocator overhead, fragmentation, autograd graph
    # metadata, multilevel hierarchy tensors, and Python/framework overhead.
    # 2x provides sufficient headroom; per_loss_backward and hybrid mode
    # are available as automatic fallbacks if VRAM is still tight.
    safety = 2

    if per_loss_bw:
        # Only one term's intermediates alive at a time
        peak_intermediate = max(repul, overlap, edge_losses)
        return (base + peak_intermediate * autograd_factor) * safety
    else:
        # All terms alive simultaneously
        return (base + (repul + overlap + edge_losses) * autograd_factor) * safety


def _estimate_hybrid_gpu_memory(n: int, num_edges: int) -> int:
    """Estimate GPU memory when heavy losses run on CPU (hybrid mode).

    Only positions, Adam state, edge data, and edge-loss intermediates on GPU.
    Repulsion and overlap intermediates stay on CPU.
    """
    # Base: pos + adam + grad + node_sizes + layer_index + edge_index
    base = n * 64 + num_edges * 16

    # Only edge-based loss intermediates on GPU
    e_batch = min(num_edges, 200000)
    edge_losses = e_batch * 4 * 6 * 4 * 2  # forward + backward

    # CPU→GPU gradient transfer: [N, 2] f32 per heavy loss backward
    grad_transfer = n * 2 * 4

    # Same 3x safety factor
    return (base + edge_losses + grad_transfer) * 3


def _compute_loss_term(
    pos: torch.Tensor,
    weight: float,
    loss_fn: Callable,
    is_heavy: bool,
    use_hybrid: bool,
    use_checkpointing: bool,
    node_sizes: torch.Tensor,
    layer_index: Optional[LayerIndex],
    cpu_node_sizes: Optional[torch.Tensor] = None,
    cpu_layer_index: Optional[LayerIndex] = None,
) -> Optional[torch.Tensor]:
    """Compute a single weighted loss term with memory optimizations.

    loss_fn signature: (pos, node_sizes, layer_index) -> scalar tensor.

    For hybrid mode, heavy losses are computed on CPU with a gradient bridge.
    For checkpointing mode, forward is recomputed during backward.
    """
    if use_hybrid and is_heavy:
        return _hybrid_loss(pos, weight, loss_fn, cpu_node_sizes, cpu_layer_index)

    if use_checkpointing and is_heavy:
        def _checkpointed_fn(p):
            return weight * loss_fn(p, node_sizes, layer_index)
        return torch_checkpoint(_checkpointed_fn, pos, use_reentrant=False)

    # Standard
    return weight * loss_fn(pos, node_sizes, layer_index)


def _hybrid_loss(
    pos_gpu: torch.Tensor,
    weight: float,
    loss_fn: Callable,
    cpu_node_sizes: Optional[torch.Tensor],
    cpu_layer_index: Optional[LayerIndex],
) -> torch.Tensor:
    """Compute a heavy loss on CPU, bridge gradient to GPU pos.

    Creates a CPU copy of positions, computes the loss on CPU (where memory
    is plentiful), then transfers only the [N, 2] gradient back to GPU.
    """
    pos_cpu = pos_gpu.detach().cpu().requires_grad_(True)
    cpu_loss = weight * loss_fn(pos_cpu, cpu_node_sizes, cpu_layer_index)

    if not cpu_loss.requires_grad:
        return torch.tensor(0.0, device=pos_gpu.device)

    cpu_loss.backward()
    cpu_grad = pos_cpu.grad  # [N, 2] on CPU

    # Bridge: return a GPU scalar whose backward() injects the CPU gradient
    return _GradBridge.apply(pos_gpu, cpu_grad.to(pos_gpu.device), cpu_loss.item())


class _GradBridge(torch.autograd.Function):
    """Autograd bridge: returns a scalar on GPU whose backward injects a pre-computed gradient."""

    @staticmethod
    def forward(ctx, pos_gpu, cpu_grad_on_gpu, loss_val):
        ctx.save_for_backward(cpu_grad_on_gpu)
        return torch.tensor(loss_val, device=pos_gpu.device, requires_grad=True)

    @staticmethod
    def backward(ctx, grad_output):
        cpu_grad_on_gpu, = ctx.saved_tensors
        return cpu_grad_on_gpu * grad_output, None, None


def _edge_batch_size(num_edges: int, config: LayoutConfig) -> int:
    """Determine edge batch size based on graph scale.

    Returns 0 for "use all edges" (no batching).
    """
    if hasattr(config, "edge_batch_size") and config.edge_batch_size > 0:
        return config.edge_batch_size

    # Auto-scale: batch at 10K+ edges
    if num_edges <= 10000:
        return 0  # use all edges
    elif num_edges <= 100000:
        return 50000
    elif num_edges <= 1000000:
        return 100000
    else:
        return 200000


def _overlap_interval(num_nodes: int, config: LayoutConfig) -> int:
    """How often to run overlap projection (every N steps)."""
    if hasattr(config, "overlap_check_interval") and config.overlap_check_interval > 0:
        return config.overlap_check_interval

    if num_nodes <= 5000:
        return 5
    elif num_nodes <= 50000:
        return 10
    elif num_nodes <= 1_000_000:
        return 20
    else:
        return 40


def _adaptive_spacing(
    num_nodes: int,
    base_node_sep: float = 25.0,
    base_rank_sep: float = 50.0,
) -> tuple:
    """Scale spacing based on graph size for density adaptation.

    Small graphs (<20): more breathing room (1.3x)
    Medium (<200): standard (1.0x)
    Large (<1000): slightly tighter (0.85x)
    Very large (1000+): compact (0.7x)
    """
    if num_nodes < 20:
        scale = 1.3
    elif num_nodes < 200:
        scale = 1.0
    elif num_nodes < 1000:
        scale = 0.85
    else:
        scale = 0.7
    return base_node_sep * scale, base_rank_sep * scale


def _apply_direction(pos: torch.Tensor, direction: str) -> torch.Tensor:
    """Transform positions based on layout direction."""
    if direction == "TB":
        return pos
    elif direction == "BT":
        result = pos.clone()
        result[:, 1] = -result[:, 1]
        return result
    elif direction == "LR":
        result = pos.clone()
        result[:, 0] = pos[:, 1]
        result[:, 1] = pos[:, 0]
        return result
    elif direction == "RL":
        result = pos.clone()
        result[:, 0] = -pos[:, 1]
        result[:, 1] = pos[:, 0]
        return result
    return pos
