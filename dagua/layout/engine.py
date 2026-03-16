"""Core optimization loop — the heart of dagua.

Takes a DaguaGraph and LayoutConfig, returns [N, 2] position tensor.
Headless: operates on tensors extracted from the graph.

Scaling strategy (tiered):
- Tier 0 (N < 500): exact O(N^2) repulsion, full overlap check
- Tier 1 (500-20K): scatter sampling repulsion, layer-local overlap
- Tier 2 (N > 20K by default): multilevel coarsening V-cycle

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

import copy
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from dagua.config import LayoutConfig
from dagua.layout.constraints import (
    alignment_loss,
    back_edge_compactness_loss,
    cluster_compactness_loss,
    cluster_containment_loss,
    cluster_separation_loss,
    crossing_loss,
    dag_ordering_loss,
    edge_attraction_loss,
    edge_length_variance_loss,
    edge_straightness_loss,
    fanout_distribution_loss,
    flex_spacing_loss,
    overlap_avoidance_loss,
    position_pin_loss,
    project_hard_pins,
    repulsion_loss,
    spacing_consistency_loss,
)
from dagua.layout.graph_classify import GraphFamily, GraphStructure, classify_graph
from dagua.layout.init_placement import init_positions
from dagua.layout.layers import LayerIndex, build_layer_index
from dagua.layout.projection import project_overlaps
from dagua.utils import VRAMBudget, longest_path_layering

LossFn = Callable[[torch.Tensor, torch.Tensor, Optional[LayerIndex]], torch.Tensor]

SAMPLED_NODE_CONTEXT_CAP = 1_000_000
MIN_SAMPLED_ACTIVE_SET = 10_000
SAMPLED_SAME_LAYER_K = 64
REPULSION_ESTIMATE_NN_K = 20
AUTOGRAD_INTERMEDIATE_FACTOR = 1.3
INTERMEDIATE_SAFETY_FACTOR = 1.5
CHECKPOINT_MEMORY_REDUCTION = 2
CUDA_ACTIVE_SET_HEADROOM = 0.85


@dataclass
class ProgressContext:
    """Context for formatting engine progress messages."""

    indent: str = "  "


@dataclass
class EdgeBatchContext:
    """Pre-computed edge batch data shared across all edge-based losses."""

    src: torch.Tensor
    tgt: torch.Tensor
    dx: torch.Tensor
    dy: torch.Tensor
    dist_sq: torch.Tensor


@dataclass
class SampledNodeContext:
    """Shared sampling state for repulsion and overlap losses."""

    active_idx: torch.Tensor
    sampled: torch.Tensor


def _build_sampled_node_context(
    num_nodes: int,
    layer_index: LayerIndex,
    device: Union[str, torch.device],
    rvs_nn_k: int,
) -> SampledNodeContext:
    """Build shared sampled-node state for large-node repulsion and overlap terms.

    Parameters
    ----------
    num_nodes : int
        Total number of nodes in the current layout problem.
    layer_index : LayerIndex
        Pre-computed layer structure for the graph.
    device : str | torch.device
        Device where the sampled tensors should be allocated.
    rvs_nn_k : int
        Same-layer sample count used by the repulsion RVS path.

    Returns
    -------
    SampledNodeContext
        Shared active-node indices and structured neighbor samples. The sampled
        tensor stores same-layer samples first, followed by adjacent-layer
        random samples.
    """
    n_active, _, _ = _sampled_node_context_sizes(num_nodes, rvs_nn_k)
    return _build_sampled_node_context_for_active(
        num_nodes,
        layer_index,
        device,
        rvs_nn_k,
        n_active,
    )


def _sampled_node_context_sizes(
    num_nodes: int,
    rvs_nn_k: int,
    n_active_override: Optional[int] = None,
) -> tuple[int, int, int]:
    """Return the sampled-node context dimensions for a graph size.

    Parameters
    ----------
    num_nodes : int
        Total number of nodes in the current layout problem.
    rvs_nn_k : int
        Same-layer sample count used by the repulsion RVS path.
    n_active_override : int, optional
        Explicit active-node count to use instead of the default ``N**0.75``
        scaling.

    Returns
    -------
    tuple[int, int, int]
        ``(n_active, n_random, k_same)`` for the sampled context.
    """
    default_n_active = min(max(int(num_nodes**0.75), min(num_nodes, 256)), SAMPLED_NODE_CONTEXT_CAP)
    if n_active_override is None:
        n_active = default_n_active
    else:
        n_active = max(min(n_active_override, default_n_active), 0)
    n_random = max(int(num_nodes**0.25), 4)
    k_same = max(SAMPLED_SAME_LAYER_K, min(rvs_nn_k, max(num_nodes - 1, 0)))
    return n_active, n_random, k_same


def _build_sampled_node_context_for_active(
    num_nodes: int,
    layer_index: LayerIndex,
    device: Union[str, torch.device],
    rvs_nn_k: int,
    n_active: int,
) -> SampledNodeContext:
    """Build a sampled-node context with an explicit active-set size.

    Parameters
    ----------
    num_nodes : int
        Total number of nodes in the current layout problem.
    layer_index : LayerIndex
        Pre-computed layer structure for the graph.
    device : str | torch.device
        Device where the sampled tensors should be allocated.
    rvs_nn_k : int
        Same-layer sample count used by the repulsion RVS path.
    n_active : int
        Number of active nodes to sample for the shared context.

    Returns
    -------
    SampledNodeContext
        Shared active-node indices and structured neighbor samples sized for the
        requested active set.
    """
    _, n_random, k_same = _sampled_node_context_sizes(
        num_nodes,
        rvs_nn_k,
        n_active_override=n_active,
    )
    if n_active <= 0:
        empty_idx = torch.zeros(0, dtype=torch.long, device=device)
        empty_sampled = torch.zeros((0, 0), dtype=torch.long, device=device)
        return SampledNodeContext(active_idx=empty_idx, sampled=empty_sampled)

    active_idx = torch.randint(0, num_nodes, (n_active,), device=device)
    layers = layer_index.node_to_layer
    layers_long = layers if layers.dtype == torch.long else layers.to(dtype=torch.long)
    offsets = layer_index.layer_offsets
    sorted_nodes = layer_index.sorted_nodes

    active_layers = layers_long[active_idx]

    same_start = offsets[active_layers]
    same_end = offsets[active_layers + 1]
    same_range = (same_end - same_start).to(dtype=torch.float32)
    same_rand = torch.rand(n_active, k_same, device=device)
    same_indices = same_start.unsqueeze(1) + (same_rand * same_range.unsqueeze(1)).long()
    same_indices = same_indices.clamp(min=0, max=num_nodes - 1)
    same_sampled = sorted_nodes[same_indices]

    adj_lo = (active_layers - 1).clamp(min=0)
    adj_hi = (active_layers + 2).clamp(max=layer_index.num_layers)
    adj_start = offsets[adj_lo]
    adj_end = offsets[adj_hi]
    adj_range = (adj_end - adj_start).to(dtype=torch.float32)
    adj_rand = torch.rand(n_active, n_random, device=device)
    adj_indices = adj_start.unsqueeze(1) + (adj_rand * adj_range.unsqueeze(1)).long()
    adj_indices = adj_indices.clamp(min=0, max=num_nodes - 1)
    random_sampled = sorted_nodes[adj_indices]

    sampled = torch.cat([same_sampled, random_sampled], dim=1)
    return SampledNodeContext(active_idx=active_idx, sampled=sampled)


def _estimate_gpu_memory_terms(
    n: int,
    num_edges: int,
    *,
    edges_on_cpu: bool = False,
    edge_batch: int = 0,
    n_active_override: Optional[int] = None,
) -> tuple[int, int, int, int]:
    """Estimate the major GPU memory terms for the layout engine.

    Parameters
    ----------
    n : int
        Number of nodes in the layout problem.
    num_edges : int
        Number of edges in the layout problem.
    edges_on_cpu : bool, default=False
        Whether edges are streamed from CPU instead of kept resident on GPU.
    edge_batch : int, default=0
        Edge batch size. ``0`` means the full edge set is transferred.
    n_active_override : int, optional
        Explicit active-node count to use for the sampled context and large-node
        losses.

    Returns
    -------
    tuple[int, int, int, int]
        ``(base, repulsion_forward, overlap_forward, edge_forward)`` in bytes.
    """
    base = n * 2 * 4 * 4
    base += n * (2 * 4 + 2 * 8)

    actual_batch = num_edges if edge_batch <= 0 else min(edge_batch, num_edges)
    if edges_on_cpu:
        base += actual_batch * 2 * 8
    else:
        base += num_edges * 2 * 8

    n_active, n_random, k_same = _sampled_node_context_sizes(
        n,
        SAMPLED_SAME_LAYER_K,
        n_active_override=n_active_override,
    )
    k_total = k_same + n_random
    sampled_ctx = n_active * (k_total * 8 + 8)
    base += sampled_ctx

    k_repul = min(REPULSION_ESTIMATE_NN_K, max(n - 1, 0)) + n_random
    repul_forward = (
        n_active * k_repul * 2 * 4
        + n_active * k_repul * 2 * 4
        + n_active * k_repul * 4
        + n_active * k_repul * 4 * 4
        + n_active * k_repul * 4
        + n_active * k_repul * 1
        + n_active * k_repul * 8
    )

    k_overlap = 64
    overlap_forward = (
        n_active * k_overlap * 4 * 4
        + n_active * k_overlap * 4 * 2
        + n_active * k_overlap * 4 * 2
        + n_active * k_overlap * 4
        + n_active * k_overlap * 1
        + n_active * k_overlap * 8
    )

    edge_forward = actual_batch * 4 * 8
    return base, repul_forward, overlap_forward, edge_forward


def _cap_sampled_active_nodes_for_budget(
    n: int,
    num_edges: int,
    budget_bytes: int,
    *,
    edges_on_cpu: bool = False,
    edge_batch: int = 0,
    use_checkpointing: bool = False,
) -> int:
    """Return the largest active set that fits the provided GPU budget.

    Parameters
    ----------
    n : int
        Number of nodes in the layout problem.
    num_edges : int
        Number of edges in the layout problem.
    budget_bytes : int
        Total GPU budget available for the layout peak estimate.
    edges_on_cpu : bool, default=False
        Whether edges are streamed from CPU instead of kept resident on GPU.
    edge_batch : int, default=0
        Edge batch size. ``0`` means the full edge set is transferred.
    use_checkpointing : bool, default=False
        Whether gradient checkpointing is enabled for heavy losses.

    Returns
    -------
    int
        Active-node count that fits the budget, or ``0`` when even the minimum
        active set would exceed the budget.
    """
    default_n_active, _, _ = _sampled_node_context_sizes(n, SAMPLED_SAME_LAYER_K)
    if default_n_active <= MIN_SAMPLED_ACTIVE_SET:
        estimated = _estimate_gpu_memory(
            n,
            num_edges,
            per_loss_bw=True,
            edges_on_cpu=edges_on_cpu,
            edge_batch=edge_batch,
            n_active_override=default_n_active,
        )
        if use_checkpointing:
            estimated //= CHECKPOINT_MEMORY_REDUCTION
        return default_n_active if estimated <= budget_bytes else 0

    base, repul_forward, overlap_forward, edge_forward = _estimate_gpu_memory_terms(
        n,
        num_edges,
        edges_on_cpu=edges_on_cpu,
        edge_batch=edge_batch,
        n_active_override=default_n_active,
    )
    peak_intermediate = max(repul_forward, overlap_forward, edge_forward)
    scale = AUTOGRAD_INTERMEDIATE_FACTOR * INTERMEDIATE_SAFETY_FACTOR
    if use_checkpointing:
        scale /= CHECKPOINT_MEMORY_REDUCTION
    estimated = int(base + peak_intermediate * scale)
    if estimated <= budget_bytes:
        return default_n_active
    if budget_bytes <= 0:
        return 0

    non_sampled_base = _estimate_gpu_memory_terms(
        n,
        num_edges,
        edges_on_cpu=edges_on_cpu,
        edge_batch=edge_batch,
        n_active_override=0,
    )[0]
    if budget_bytes <= non_sampled_base:
        return 0

    min_active = min(default_n_active, MIN_SAMPLED_ACTIVE_SET)
    allowed_intermediate = max(int((budget_bytes - non_sampled_base) / scale), 0)
    scaled_active = int(default_n_active * allowed_intermediate / max(peak_intermediate, 1))
    candidate = max(min(scaled_active, default_n_active), min_active)
    candidate_estimate = _estimate_gpu_memory(
        n,
        num_edges,
        per_loss_bw=True,
        edges_on_cpu=edges_on_cpu,
        edge_batch=edge_batch,
        n_active_override=candidate,
    )
    if use_checkpointing:
        candidate_estimate //= CHECKPOINT_MEMORY_REDUCTION
    if candidate_estimate <= budget_bytes:
        return candidate

    min_estimate = _estimate_gpu_memory(
        n,
        num_edges,
        per_loss_bw=True,
        edges_on_cpu=edges_on_cpu,
        edge_batch=edge_batch,
        n_active_override=min_active,
    )
    if use_checkpointing:
        min_estimate //= CHECKPOINT_MEMORY_REDUCTION
    return min_active if min_estimate <= budget_bytes else 0


def _auto_layout_steps(num_nodes: int) -> int:
    """Return the automatic optimization step count for a graph size.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    int
        Auto-selected number of optimization steps.
    """
    if num_nodes <= 10:
        return 50
    if num_nodes <= 50:
        return 100
    if num_nodes <= 200:
        return 150
    if num_nodes <= 500:
        return 200
    if num_nodes <= 2000:
        return 250
    if num_nodes <= 5000:
        return 300
    if num_nodes <= 10000:
        return 400
    return 500


def _override_for_tree(config: LayoutConfig) -> LayoutConfig:
    """Return a config copy with tree-appropriate weights.

    Parameters
    ----------
    config : LayoutConfig
        Base layout configuration.

    Returns
    -------
    LayoutConfig
        Shallow copy with tree-specific loss weights disabled.
    """
    tree_config = copy.copy(config)
    tree_config.w_crossing = 0.0
    tree_config.w_straightness = 0.0
    tree_config.w_length_variance = 0.0
    return tree_config


def layout(graph, config: Optional[LayoutConfig] = None, trace=None) -> torch.Tensor:
    """Compute layout positions for all nodes.

    Args:
        graph: DaguaGraph instance
        config: LayoutConfig (uses defaults if None)

    Returns:
        [N, 2] tensor of (x, y) positions
    """
    if config is None:
        config = LayoutConfig()

    # Ensure node sizes are computed
    graph.compute_node_sizes()

    n = graph.num_nodes
    device = config.device
    if device == "cuda" and n < 1000:
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if n == 0:
        pos = torch.zeros(0, 2, device=device)
        graph.cache_layout(pos)
        return pos
    if n == 1:
        pos = torch.zeros(1, 2, device=device)
        graph.cache_layout(pos)
        return pos

    # Set seed for determinism
    if config.seed is not None:
        torch.manual_seed(config.seed)
        if device == "cuda":
            torch.cuda.manual_seed(config.seed)

    # Handle cycles: reverse back edges so the engine sees a DAG
    graph._prepare_for_layout()
    try:
        # Tier 2: Multilevel coarsening for large graphs (N > 20K default)
        # Coarsening is faster than direct optimization at this scale.
        # Don't move data to GPU yet — multilevel manages device transfers lazily
        if n > config.multilevel_threshold:
            from dagua.layout.multilevel import multilevel_layout

            pos = multilevel_layout(graph, config, trace=trace)
            graph.cache_layout(pos)
            return pos

        # Tier 0-2: Direct layout — move data to device
        edge_index = graph.edge_index.to(device)
        node_sizes = graph.node_sizes.to(device)

        if config.verbose:
            num_edges = edge_index.shape[1] if edge_index.numel() > 0 else 0
            print(f"[dagua] Layout: {n:,} nodes, {num_edges:,} edges", flush=True)

        # Resolve flex node IDs to integer indices before headless engine
        effective_config = _resolve_flex_ids(config, graph)
        if effective_config.device != device:
            effective_config = copy.copy(effective_config)
            effective_config.device = device

        # Also pick up flex from graph.flex if config doesn't have one
        if effective_config.flex is None and getattr(graph, "flex", None) is not None:
            import copy as _c

            effective_config = _c.copy(effective_config)
            effective_config.flex = _resolve_graph_flex(graph.flex, graph._id_to_index)

        pos = _layout_inner(
            edge_index,
            n,
            node_sizes,
            effective_config,
            device=device,
            clusters=graph.clusters if hasattr(graph, "clusters") else None,
            cluster_parents=graph.cluster_parents if hasattr(graph, "cluster_parents") else None,
            progress_context=ProgressContext(),
            trace=trace,
        )

        # Apply direction transform
        direction = config.direction if config else graph.direction
        pos = _apply_direction(pos, direction)
        graph.cache_layout(pos)
        return pos
    finally:
        graph._restore_after_layout()


def _layout_inner(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: torch.Tensor,
    config: LayoutConfig,
    device: str = "cpu",
    init_pos: Optional[torch.Tensor] = None,
    clusters: Optional[dict] = None,
    cluster_parents: Optional[dict] = None,
    layer_assignments: Optional[torch.Tensor] = None,
    progress_context: Optional[ProgressContext] = None,
    trace: Optional[object] = None,
    *,
    graph_structure: Optional[GraphStructure] = None,
    prebuilt_layer_index: Optional[LayerIndex] = None,
    skip_classification: bool = False,
) -> torch.Tensor:
    """Optimize node positions for a tensor-only layout problem.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor shaped ``[2, E]``.
    num_nodes : int
        Number of nodes in the current layout problem.
    node_sizes : torch.Tensor
        Node size tensor shaped ``[N, 2]`` or broadcastable from ``[N]`` /
        ``[N, 1]``.
    config : LayoutConfig
        Layout configuration for the current solve.
    device : str, default="cpu"
        Target device for optimization.
    init_pos : torch.Tensor, optional
        Warm-start positions shaped ``[N, 2]``.
    clusters : dict, optional
        Cluster membership data for cluster losses.
    cluster_parents : dict, optional
        Parent-cluster mapping used by containment/separation losses.
    layer_assignments : torch.Tensor, optional
        Raw layer assignments shaped ``[N]``.
    progress_context : ProgressContext, optional
        Prefix context for verbose logging.
    trace : object, optional
        Trace collector used by debugging and benchmarking flows.
    graph_structure : GraphStructure, optional
        Pre-classified structure metadata for the current graph.
    prebuilt_layer_index : LayerIndex, optional
        Pre-computed layer index for the current graph.
    skip_classification : bool, default=False
        When ``True``, skip structural classification and use the general layout
        path. This is used during multilevel refinement where the optimization
        strategy is already fixed by the V-cycle.

    Returns
    -------
    torch.Tensor
        Detached position tensor shaped ``[N, 2]``.
    """
    import time as _time

    n = num_nodes
    device_type = torch.device(device).type
    if node_sizes.ndim == 1:
        node_sizes = node_sizes.unsqueeze(1).expand(-1, 2).contiguous()
    elif node_sizes.ndim == 2 and node_sizes.shape[1] == 1:
        node_sizes = node_sizes.expand(-1, 2).contiguous()

    verbose = getattr(config, "verbose", False)
    _indent = progress_context.indent if progress_context else "  "

    def _vlog(msg):
        if verbose:
            vram = ""
            if device == "cuda" and torch.cuda.is_available():
                used = torch.cuda.memory_allocated() / 1024**2
                free, total = torch.cuda.mem_get_info()
                vram = f" [VRAM {used:.0f}MB / {total / 1024**2:.0f}MB]"
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
            edge_index,
            n,
            node_sizes,
            node_sep=node_sep,
            rank_sep=rank_sep,
            device=device,
        )

    if trace is not None and hasattr(trace, "capture_layout_positions"):
        trace.capture_layout_positions(
            phase="initialization",
            step=0,
            total_steps=max(config.steps, 1),
            positions=pos.detach(),
        )

    # Pre-compute layer structure (used by repulsion, overlap, projection, crossing)
    layer_assignments_raw: Optional[Union[List[int], torch.Tensor]] = None
    layer_index: Optional[LayerIndex] = None
    if prebuilt_layer_index is not None:
        layer_index = prebuilt_layer_index
        if layer_index.node_to_layer.device.type != device_type:
            layer_index = build_layer_index(layer_index.node_to_layer, device=device)
        layer_assignments_raw = layer_index.node_to_layer
    elif layer_assignments is not None:
        layer_index = build_layer_index(layer_assignments, device=device)
        layer_assignments_raw = layer_assignments
    elif edge_index.numel() > 0:
        layer_assignments_raw = longest_path_layering(edge_index, n, device=device)
        layer_index = build_layer_index(layer_assignments_raw, device=device)

    # Determine adaptive parameters based on graph size
    num_edges = edge_index.shape[1] if edge_index.numel() > 0 else 0
    edge_batch = _edge_batch_size(num_edges, config)
    overlap_interval = _overlap_interval(n, config)
    steps = config.steps if config.steps > 0 else _auto_layout_steps(n)

    classification_layers: Optional[torch.Tensor] = None
    if isinstance(layer_assignments_raw, torch.Tensor):
        classification_layers = layer_assignments_raw
    elif layer_assignments_raw is not None:
        classification_layers = torch.tensor(layer_assignments_raw, dtype=torch.long)

    if skip_classification:
        structure = GraphStructure(
            family=GraphFamily.GENERAL,
            num_components=1,
            max_degree=0,
            num_layers=0,
            avg_layer_width=0.0,
            is_planar_hint=False,
        )
    elif graph_structure is not None:
        structure = graph_structure
    else:
        structure = classify_graph(edge_index, n, layer_assignments=classification_layers)

    if structure.family in {GraphFamily.TREE, GraphFamily.CHAIN}:
        config = _override_for_tree(config)
    if structure.family == GraphFamily.CHAIN:
        steps = min(steps, 50)

    _vlog(
        "structure="
        f"{structure.family.name.lower()} "
        f"(components={structure.num_components}, "
        f"max_degree={structure.max_degree}, "
        f"layers={structure.num_layers})"
    )

    # Edge streaming: if edges are too large for GPU, keep them on CPU
    # and stream batches to GPU each step.  This lets positions stay on GPU
    # while edges (which can be 800MB+) stay in CPU RAM.
    edges_on_cpu = edge_index.device.type == "cpu" and device == "cuda"
    if not edges_on_cpu and device == "cuda" and edge_index.numel() > 0:
        edge_bytes = edge_index.numel() * edge_index.element_size()
        if not VRAMBudget().fits(edge_bytes):
            edge_index = edge_index.cpu()
            edges_on_cpu = True

    # Resolve memory optimization flags (VRAM-aware when on CUDA)
    use_per_loss_bw, use_checkpointing, use_hybrid = _resolve_memory_strategy(
        n,
        num_edges,
        device,
        config,
        edges_on_cpu=edges_on_cpu,
        edge_batch=edge_batch,
    )
    # Create thread pool for parallel hybrid losses
    executor = None
    if use_hybrid and use_per_loss_bw and getattr(config, "num_workers", 0) > 0:
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=config.num_workers)

    flags = []
    if use_per_loss_bw:
        flags.append("per_loss_bw")
    if use_checkpointing:
        flags.append("checkpoint")
    if use_hybrid:
        flags.append("hybrid")
    if edges_on_cpu:
        flags.append("edge_stream")
    if executor is not None:
        flags.append(f"workers={config.num_workers}")
    _vlog(
        "init done, "
        f"{num_edges:,} edges, batch={edge_batch}, "
        f"strategy=[{', '.join(flags) or 'standard'}]"
    )

    # Hybrid device: keep positions on GPU, create CPU mirror for heavy losses
    cpu_node_sizes = None
    cpu_layer_index = None
    cpu_edge_index = edge_index if edge_index.device.type == "cpu" else None
    if use_hybrid:
        cpu_node_sizes = node_sizes.cpu()
        if cpu_edge_index is None:
            cpu_edge_index = edge_index.cpu()
        if prebuilt_layer_index is not None:
            if prebuilt_layer_index.node_to_layer.device.type == "cpu":
                cpu_layer_index = prebuilt_layer_index
            else:
                cpu_layer_index = build_layer_index(
                    prebuilt_layer_index.node_to_layer,
                    device="cpu",
                )
        elif layer_assignments_raw is not None:
            cpu_layer_index = build_layer_index(layer_assignments_raw, device="cpu")
    elif edges_on_cpu:
        cpu_edge_index = edge_index

    # Step 2: Set up optimization
    pos = pos.clone().detach().requires_grad_(True)
    del init_pos  # Free pre-clone positions (8 GB at 1B nodes)
    optimizer = torch.optim.Adam([pos], lr=config.lr)

    # Step 3: Build loss functions ONCE (reuse across steps via mutable refs)
    # batch_edges_ref is a mutable container so lambdas capture the reference
    batch_edges_ref: List[Optional[torch.Tensor]] = [None]
    edge_ctx_ref: List[Optional[EdgeBatchContext]] = [None]
    sampled_ctx_ref: List[Optional[SampledNodeContext]] = [None]

    def _active_edges(p: torch.Tensor) -> torch.Tensor:
        if batch_edges_ref[0] is not None and p.device == batch_edges_ref[0].device:
            return batch_edges_ref[0]
        if cpu_edge_index is not None:
            return cpu_edge_index
        return edge_index

    def _active_edge_ctx(p: torch.Tensor) -> Optional[EdgeBatchContext]:
        if use_per_loss_bw:
            return None
        if edge_ctx_ref[0] is not None and p.device == edge_ctx_ref[0].src.device:
            return edge_ctx_ref[0]
        return None

    def _active_sampled_ctx(p: torch.Tensor) -> Optional[SampledNodeContext]:
        if sampled_ctx_ref[0] is not None and p.device == sampled_ctx_ref[0].active_idx.device:
            return sampled_ctx_ref[0]
        return None

    # Static loss functions — each returns (base_weight_key, loss_fn, is_heavy, is_annealed)
    # base_weight_key is a string to look up the annealed weight at each step
    loss_fns: List[tuple[str, LossFn, bool, bool]] = []

    if config.w_dag > 0:
        loss_fns.append(
            (
                "w_dag",
                lambda p, ns, li: dag_ordering_loss(
                    p,
                    _active_edges(p),
                    ns,
                    rank_sep,
                    edge_ctx=_active_edge_ctx(p),
                ),
                False,
                True,
            )
        )

    if config.w_attract > 0:
        loss_fns.append(
            (
                "w_attract",
                lambda p, ns, li: edge_attraction_loss(
                    p,
                    _active_edges(p),
                    x_bias=config.w_attract_x_bias,
                    edge_ctx=_active_edge_ctx(p),
                ),
                False,
                False,
            )
        )

    if config.w_repel > 0:

        def repel_fn(
            p: torch.Tensor,
            ns: torch.Tensor,
            li: Optional[LayerIndex],
        ) -> torch.Tensor:
            """Compute the repulsion term for the current node positions."""
            return repulsion_loss(
                p,
                n,
                threshold=config.exact_repulsion_threshold,
                sample_k=config.negative_sample_k,
                layer_index=li,
                node_sizes=ns,
                rvs_threshold=config.rvs_threshold,
                rvs_nn_k=config.rvs_nn_k,
                sampled_ctx=_active_sampled_ctx(p),
            )

        repel_is_heavy = True
        if n > config.repel_amortize_threshold and config.repel_amortize_interval > 1:
            repel_fn = _make_amortized_loss(
                repel_fn,
                skip_every=config.repel_amortize_interval,
            )
            # Amortized losses must not go through gradient checkpointing —
            # skip steps return torch.tensor(0.0) which changes saved tensor count.
            repel_is_heavy = False
        loss_fns.append(("w_repel", repel_fn, repel_is_heavy, True))

    if config.w_overlap > 0:
        loss_fns.append(
            (
                "w_overlap",
                lambda p, ns, li: overlap_avoidance_loss(
                    p,
                    ns,
                    layer_index=li,
                    rvs_threshold=config.rvs_threshold,
                    sampled_ctx=_active_sampled_ctx(p),
                ),
                True,
                True,
            )
        )

    if config.w_cluster > 0 and clusters:
        loss_fns.append(
            (
                "w_cluster",
                lambda p, ns, li: cluster_compactness_loss(p, clusters, device=p.device),
                False,
                False,
            )
        )
        loss_fns.append(
            (
                "w_cluster_sep",
                lambda p, ns, li: cluster_separation_loss(
                    p, ns, clusters, device=p.device, cluster_parents=cluster_parents
                ),
                False,
                False,
            )
        )

    if config.w_cluster_contain > 0 and clusters and cluster_parents:
        loss_fns.append(
            (
                "w_cluster_contain",
                lambda p, ns, li: cluster_containment_loss(
                    p, ns, clusters, cluster_parents, device=p.device
                ),
                False,
                False,
            )
        )

    if config.w_crossing > 0 and num_edges >= 4:
        # alpha is annealed per-step, captured via mutable ref
        crossing_alpha_ref: List[float] = [3.0]
        # Crossing loss is O(E²) — amortize by computing every N steps.
        # Positions change slowly enough that every-step is wasteful.
        crossing_interval = 3 if n > 500 else 5 if n > 50 else 10
        crossing_step_ref: List[int] = [0]
        # Scale weight up to compensate for skipped steps
        crossing_weight_scale = float(crossing_interval)

        def _crossing_fn(p, ns, li):
            crossing_step_ref[0] += 1
            if crossing_step_ref[0] % crossing_interval != 0:
                return torch.tensor(0.0, device=p.device, requires_grad=True)
            return crossing_weight_scale * crossing_loss(
                p,
                _active_edges(p),
                alpha=crossing_alpha_ref[0],
                layer_assignments=layer_assignments_raw,
                max_pairs=500,
            )

        loss_fns.append(("w_crossing", _crossing_fn, False, True))

    if config.w_straightness > 0:
        loss_fns.append(
            (
                "w_straightness",
                lambda p, ns, li: edge_straightness_loss(
                    p,
                    _active_edges(p),
                    edge_ctx=_active_edge_ctx(p),
                ),
                False,
                True,
            )
        )

    if config.w_length_variance > 0:
        loss_fns.append(
            (
                "w_length_variance",
                lambda p, ns, li: edge_length_variance_loss(
                    p,
                    _active_edges(p),
                    edge_ctx=_active_edge_ctx(p),
                ),
                False,
                False,
            )
        )

    if config.w_spacing > 0 and layer_index is not None:
        loss_fns.append(
            (
                "w_spacing",
                lambda p, ns, li: spacing_consistency_loss(
                    p,
                    ns,
                    li,
                    target_gap=node_sep,
                ),
                False,
                False,
            )
        )

    if config.w_fanout > 0:

        def fanout_fn(
            p: torch.Tensor,
            ns: torch.Tensor,
            li: Optional[LayerIndex],
        ) -> torch.Tensor:
            """Compute the fanout-distribution regularizer for active edges."""
            del ns, li
            return fanout_distribution_loss(
                p,
                _active_edges(p),
            )

        if n > config.fanout_amortize_threshold and config.fanout_amortize_interval > 1:
            fanout_fn = _make_amortized_loss(
                fanout_fn,
                skip_every=config.fanout_amortize_interval,
            )
        loss_fns.append(("w_fanout", fanout_fn, False, False))

    if config.w_back_edge > 0:
        loss_fns.append(
            (
                "w_back_edge",
                lambda p, ns, li: back_edge_compactness_loss(
                    p,
                    _active_edges(p),
                    edge_ctx=_active_edge_ctx(p),
                ),
                False,
                False,
            )
        )

    # --- Flex constraints: pins, alignment, flex spacing ---
    flex_data = _prepare_flex_data(config, n, device)

    if flex_data["has_soft_pins"]:
        _pin_idx = flex_data["pin_indices"]
        _pin_tgt = flex_data["pin_targets"]
        _pin_wt = flex_data["pin_weights"]
        _pin_mask = flex_data["soft_pin_mask"]
        loss_fns.append(
            (
                "w_pin",
                lambda p, ns, li: position_pin_loss(p, _pin_idx, _pin_tgt, _pin_wt, _pin_mask),
                False,
                False,
            )
        )

    if flex_data["align_groups"]:
        _ag = flex_data["align_groups"]
        loss_fns.append(("w_align_flex", lambda p, ns, li: alignment_loss(p, _ag), False, False))

    if flex_data["flex_node_sep"] is not None:
        _fsep = flex_data["flex_node_sep"]
        _fwt = flex_data["flex_node_sep_weight"]
        loss_fns.append(
            (
                "w_flex_spacing",
                lambda p, ns, li: flex_spacing_loss(p, ns, li, _fsep, _fwt),
                False,
                False,
            )
        )

    # Pre-allocate edge batch buffer (avoids per-step tensor allocation)
    batch_buf = (
        torch.empty(2, edge_batch, dtype=torch.long, device=device)
        if edge_batch > 0 and num_edges > edge_batch
        else None
    )
    perm_buf = torch.empty(edge_batch, dtype=torch.long, device="cpu") if edge_batch > 0 else None

    # Optimization loop with annealing
    prev_unweighted = float("inf")
    stall_count = 0
    _t_loop = _time.perf_counter()
    _log_interval = max(steps // 4, 1)  # log at 25%, 50%, 75%, 100%
    use_sampled_context = layer_index is not None and (config.w_repel > 0 or config.w_overlap > 0)
    sample_refresh_interval = (
        config.repel_amortize_interval
        if n > config.repel_amortize_threshold and config.repel_amortize_interval > 1
        else 1
    )
    sampled_ctx: Optional[SampledNodeContext] = None
    random_interval = (
        max(int(1.0 / config.edge_random_fraction), 1)
        if 0.0 < config.edge_random_fraction < 1.0
        else 0
    )

    for step in range(steps):
        t = step / max(steps - 1, 1)  # 0 → 1

        if verbose and step > 0 and step % _log_interval == 0:
            _vlog(
                f"step {step}/{steps} ({100 * step // steps}%) "
                f"loss={prev_unweighted:.2f} "
                f"[{_time.perf_counter() - _t_loop:.1f}s]"
            )

        optimizer.zero_grad(set_to_none=True)
        edge_ctx_ref[0] = None

        # Sample edge batch for this step — reuse pre-allocated buffer
        if batch_buf is not None:
            # Contiguous chunks are cache-friendly; random sampling re-mixes periodically.
            use_random_sampling = config.edge_random_fraction >= 1.0 or (
                random_interval > 0 and step % random_interval == 0
            )
            if use_random_sampling:
                assert perm_buf is not None
                torch.randint(0, num_edges, (edge_batch,), out=perm_buf)
                batch_buf.copy_(edge_index[:, perm_buf])
            else:
                start = (step * edge_batch) % num_edges
                end = min(start + edge_batch, num_edges)
                actual = end - start
                batch_buf[:, :actual].copy_(edge_index[:, start:end])
                if actual < edge_batch:
                    remaining = edge_batch - actual
                    batch_buf[:, actual:].copy_(edge_index[:, :remaining])
            batch_edges_ref[0] = batch_buf
        elif edges_on_cpu:
            batch_edges_ref[0] = edge_index.to(device)
        else:
            batch_edges_ref[0] = edge_index

        be = batch_edges_ref[0]
        if be is not None and be.numel() > 0:
            keep = be[0] != be[1]
            if not keep.all():
                batch_edges_ref[0] = be[:, keep]
        be = batch_edges_ref[0]
        if be is not None:
            if not use_per_loss_bw:
                src, tgt = be[0], be[1]
                src_pos = pos[src]
                tgt_pos = pos[tgt]
                dx = src_pos[:, 0] - tgt_pos[:, 0]
                dy = src_pos[:, 1] - tgt_pos[:, 1]
                dist_sq = dx * dx + dy * dy
                edge_ctx_ref[0] = EdgeBatchContext(
                    src=src,
                    tgt=tgt,
                    dx=dx,
                    dy=dy,
                    dist_sq=dist_sq,
                )

        if use_sampled_context and (step == 0 or step % sample_refresh_interval == 0):
            sampled_layer_index = (
                cpu_layer_index if use_hybrid and cpu_layer_index is not None else layer_index
            )
            sampled_device: Union[str, torch.device] = "cpu" if use_hybrid else pos.device
            sampled_n_active, _, _ = _sampled_node_context_sizes(n, config.rvs_nn_k)
            default_n_active = sampled_n_active
            if (
                device == "cuda"
                and not use_hybrid
                and use_per_loss_bw
                and torch.cuda.is_available()
            ):
                free, _ = torch.cuda.mem_get_info()
                total_budget = int(
                    (free + torch.cuda.memory_allocated()) * CUDA_ACTIVE_SET_HEADROOM
                )
                capped_n_active = _cap_sampled_active_nodes_for_budget(
                    n,
                    num_edges,
                    total_budget,
                    edges_on_cpu=edges_on_cpu,
                    edge_batch=edge_batch,
                    use_checkpointing=use_checkpointing,
                )
                if 0 < capped_n_active < sampled_n_active:
                    sampled_n_active = capped_n_active
                    _vlog(
                        "capped sampled active set to "
                        f"{sampled_n_active:,} nodes to keep per-loss GPU execution in budget"
                    )
            if sampled_n_active == default_n_active:
                sampled_ctx = _build_sampled_node_context(
                    n,
                    sampled_layer_index,
                    sampled_device,
                    config.rvs_nn_k,
                )
            else:
                sampled_ctx = _build_sampled_node_context_for_active(
                    n,
                    sampled_layer_index,
                    sampled_device,
                    config.rvs_nn_k,
                    sampled_n_active,
                )
        elif not use_sampled_context:
            sampled_ctx = None
        sampled_ctx_ref[0] = sampled_ctx

        # Compute annealed weights for this step
        w_dag = config.w_dag * (1 - 0.5 * t)
        w_repel = config.w_repel * (1 + 2 * t)
        w_overlap = config.w_overlap * (1 + t)
        t_cross = min(t / 0.3, 1.0)
        w_crossing = config.w_crossing * t_cross
        w_straightness = config.w_straightness * (1 + 0.5 * t)

        # Update crossing alpha via mutable ref (only exists when crossing loss is active)
        if config.w_crossing > 0 and num_edges >= 4:
            crossing_alpha_ref[0] = 3.0 + 7.0 * t_cross

        # Map weight keys to current annealed values
        weight_map = {
            "w_dag": w_dag,
            "w_attract": config.w_attract,
            "w_repel": w_repel,
            "w_overlap": w_overlap,
            "w_cluster": config.w_cluster,
            "w_cluster_sep": config.w_cluster * 0.5,
            "w_cluster_contain": config.w_cluster_contain,
            "w_crossing": w_crossing,
            "w_straightness": w_straightness,
            "w_length_variance": config.w_length_variance,
            "w_spacing": config.w_spacing,
            "w_fanout": config.w_fanout,
            "w_back_edge": config.w_back_edge,
            # Flex constraint weights (constant — not annealed)
            "w_pin": 1.0,
            "w_align_flex": 1.0,
            "w_flex_spacing": 1.0,
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
                            _hybrid_loss,
                            pos,
                            weight,
                            loss_fn,
                            cpu_node_sizes,
                            cpu_layer_index,
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
                        pos,
                        weight,
                        loss_fn,
                        is_heavy,
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
                    pos,
                    weight,
                    loss_fn,
                    is_heavy,
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

        # Hard-pin projection (weight=inf pins snapped to exact positions)
        if flex_data["has_hard_pins"]:
            project_hard_pins(
                pos,
                flex_data["pin_indices"],
                flex_data["pin_targets"],
                flex_data["hard_pin_mask"],
            )

        # Hard overlap projection (adaptive frequency + adaptive iterations)
        if step % overlap_interval == 0 or step == steps - 1:
            proj_iters = 5 if n <= 500_000 else 3 if n <= 5_000_000 else 2
            project_overlaps(
                pos,
                node_sizes,
                padding=2.0,
                iterations=proj_iters,
                layer_index=layer_index,
            )

        if trace is not None and hasattr(trace, "capture_layout_positions"):
            trace.capture_layout_positions(
                phase="node_optimization",
                step=step + 1,
                total_steps=steps,
                positions=pos.detach(),
            )

        # Early stopping check (use unweighted loss — immune to annealing)
        # Small graphs converge faster — detect convergence sooner
        rel_threshold = 5e-4 if n <= 200 else 1e-4
        stall_limit = 3 if n <= 200 else 5
        effective_stall_limit = stall_limit
        if n <= 5000:
            effective_stall_limit = max(stall_limit - 2, 3)
        rel_change = abs(prev_unweighted - unweighted_loss_val) / max(abs(prev_unweighted), 1e-8)
        if step > 10 and rel_change < rel_threshold:
            stall_count += 1
            if stall_count >= effective_stall_limit:
                break
        else:
            stall_count = 0
        prev_unweighted = unweighted_loss_val

    _vlog(f"done ({_time.perf_counter() - _t_loop:.1f}s)")

    # Free optimizer and grad before final projection — no longer needed
    del optimizer
    pos.grad = None

    # Final aggressive overlap projection (scale iterations with N)
    if n <= 50:
        final_proj_iters = 5
    elif n <= 200:
        final_proj_iters = 10
    elif n <= 500_000:
        final_proj_iters = 20
    elif n <= 5_000_000:
        final_proj_iters = 10
    else:
        final_proj_iters = 3
    _vlog(f"final projection ({final_proj_iters} iters)...")
    _t_proj = _time.perf_counter()
    project_overlaps(
        pos,
        node_sizes,
        padding=2.0,
        iterations=final_proj_iters,
        layer_index=layer_index,
    )
    _vlog(f"projection done in {_time.perf_counter() - _t_proj:.1f}s")

    if trace is not None and hasattr(trace, "capture_layout_positions"):
        trace.capture_layout_positions(
            phase="final_projection",
            step=steps,
            total_steps=steps,
            positions=pos.detach(),
        )

    if executor is not None:
        executor.shutdown(wait=False)

    return pos.detach()


def _resolve_memory_strategy(
    n: int,
    num_edges: int,
    device: str,
    config: LayoutConfig,
    edges_on_cpu: bool = False,
    edge_batch: int = 0,
) -> tuple[bool, bool, bool]:
    """Resolve memory optimization flags, VRAM-aware when on CUDA.

    Parameters
    ----------
    n : int
        Number of nodes in the layout problem.
    num_edges : int
        Number of edges in the layout problem.
    device : str
        Target compute device.
    config : LayoutConfig
        Layout configuration that may force or disable memory strategies.
    edges_on_cpu : bool, default=False
        Whether edges are streamed from CPU instead of stored on GPU.
    edge_batch : int, default=0
        Edge batch size. ``0`` means the full edge set is transferred.

    Returns
    -------
    tuple[bool, bool, bool]
        ``(use_per_loss_bw, use_checkpointing, use_hybrid)``.
    """
    plb = config.per_loss_backward
    gcp = config.gradient_checkpointing
    hyb = config.hybrid_device

    # Handle explicit on/off overrides
    if plb != "auto" and gcp != "auto" and hyb != "auto":
        return plb == "on", gcp == "on", hyb == "on"

    # CPU mode: per_loss_bw frees each loss's forward graph after backward,
    # reducing cache pressure at scale. This is faster than single backward
    # for large graphs despite multiple backward() calls.
    if device != "cuda":
        use_plb = (plb == "on") or (plb == "auto" and n > 50000)
        return use_plb, False, False

    # CUDA mode: query available VRAM
    budget = VRAMBudget()
    usable = budget.remaining()

    # Estimate peak GPU memory for different strategies
    mem_full = _estimate_gpu_memory(
        n,
        num_edges,
        per_loss_bw=False,
        edges_on_cpu=edges_on_cpu,
        edge_batch=edge_batch,
    )
    mem_plb = _estimate_gpu_memory(
        n,
        num_edges,
        per_loss_bw=True,
        edges_on_cpu=edges_on_cpu,
        edge_batch=edge_batch,
    )
    mem_plb_ckpt = mem_plb // CHECKPOINT_MEMORY_REDUCTION

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
    capped_active = _cap_sampled_active_nodes_for_budget(
        n,
        num_edges,
        usable,
        edges_on_cpu=edges_on_cpu,
        edge_batch=edge_batch,
    )
    if capped_active > 0 and gcp != "on" and hyb != "on":
        return use_plb, False, False

    # Level 2: per-loss backward + checkpointing (6-8x reduction, ~30% slower)
    use_ckpt = gcp != "off"
    if mem_plb_ckpt < usable and hyb != "on":
        return use_plb, use_ckpt, False

    # Level 3: hybrid device (heavy losses on CPU, edge losses on GPU)
    use_hyb = hyb != "off"
    return use_plb, use_ckpt, use_hyb


def _estimate_gpu_memory(
    n: int,
    num_edges: int,
    per_loss_bw: bool = False,
    *,
    edges_on_cpu: bool = False,
    edge_batch: int = 0,
    n_active_override: Optional[int] = None,
) -> int:
    """Estimate peak GPU memory in bytes for full-GPU layout.

    Parameters
    ----------
    n : int
        Number of nodes in the layout problem.
    num_edges : int
        Number of edges in the layout problem.
    per_loss_bw : bool, default=False
        Whether per-loss backward frees each loss graph before the next heavy
        term is evaluated.
    edges_on_cpu : bool, default=False
        Whether edges are streamed from CPU instead of kept resident on GPU.
    edge_batch : int, default=0
        Edge batch size. ``0`` means the full edge set is transferred.
    n_active_override : int, optional
        Explicit active-node count for the sampled-node context estimate.

    Returns
    -------
    int
        Estimated peak GPU memory in bytes.
    """
    base, repul_forward, overlap_forward, edge_forward = _estimate_gpu_memory_terms(
        n,
        num_edges,
        edges_on_cpu=edges_on_cpu,
        edge_batch=edge_batch,
        n_active_override=n_active_override,
    )
    if per_loss_bw:
        peak_intermediate = max(repul_forward, overlap_forward, edge_forward)
        return int(
            base + peak_intermediate * AUTOGRAD_INTERMEDIATE_FACTOR * INTERMEDIATE_SAFETY_FACTOR
        )
    all_intermediates = repul_forward + overlap_forward + edge_forward
    return int(base + all_intermediates * AUTOGRAD_INTERMEDIATE_FACTOR * INTERMEDIATE_SAFETY_FACTOR)


def _estimate_hybrid_gpu_memory(
    n: int,
    num_edges: int,
    *,
    edges_on_cpu: bool = False,
    edge_batch: int = 0,
) -> int:
    """Estimate GPU memory when heavy losses run on CPU (hybrid mode).

    Parameters
    ----------
    n : int
        Number of nodes in the layout problem.
    num_edges : int
        Number of edges in the layout problem.
    edges_on_cpu : bool, default=False
        Whether edges are streamed from CPU instead of kept resident on GPU.
    edge_batch : int, default=0
        Edge batch size. ``0`` means the full edge set is transferred.

    Returns
    -------
    int
        Estimated peak GPU memory in bytes while hybrid mode keeps the heavy
        node losses on CPU.
    """
    base = n * 2 * 4 * 4
    base += n * (2 * 4 + 2 * 8)
    actual_batch = num_edges if edge_batch <= 0 else min(edge_batch, num_edges)
    if edges_on_cpu:
        base += actual_batch * 2 * 8
    else:
        base += num_edges * 2 * 8

    edge_forward = actual_batch * 4 * 8
    grad_transfer = n * 2 * 4
    return int(
        base
        + edge_forward * AUTOGRAD_INTERMEDIATE_FACTOR * INTERMEDIATE_SAFETY_FACTOR
        + grad_transfer
    )


def _compute_loss_term(
    pos: torch.Tensor,
    weight: float,
    loss_fn: LossFn,
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
    loss_fn: LossFn,
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
    assert cpu_grad is not None

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
        (cpu_grad_on_gpu,) = ctx.saved_tensors
        return cpu_grad_on_gpu * grad_output, None, None


def _edge_batch_size(num_edges: int, config: LayoutConfig) -> int:
    """Determine edge batch size based on graph scale.

    Returns 0 for "use all edges" (no batching).
    """
    if config.edge_batch_size > 0:
        return config.edge_batch_size

    if num_edges <= 10000:
        return 0

    if config.device == "cuda":
        try:
            free, _ = torch.cuda.mem_get_info()
            max_safe = int(free * 0.3 / 128)
            if num_edges <= max_safe:
                return 0
            return min(max_safe, num_edges)
        except Exception:
            pass

    if num_edges <= 100000:
        return 50000
    if num_edges <= 500000:
        return 200000
    if num_edges <= 2_000_000:
        return 500_000
    if num_edges <= 20_000_000:
        return 2_000_000
    return 5_000_000


def _make_amortized_loss(loss_fn: LossFn, skip_every: int) -> LossFn:
    """Return a wrapper that skips every ``skip_every`` call.

    Parameters
    ----------
    loss_fn : LossFn
        Base loss function to evaluate on non-skipped calls.
    skip_every : int
        Call interval to skip. ``2`` skips every other call, ``3`` skips every
        third call, and so on.

    Returns
    -------
    LossFn
        Wrapped loss function with internal call-count state.
    """
    step_ref: List[int] = [0]

    def _amortized_loss(
        pos: torch.Tensor,
        node_sizes: torch.Tensor,
        layer_index: Optional[LayerIndex],
    ) -> torch.Tensor:
        """Evaluate the wrapped loss except on periodic skipped steps."""
        step_ref[0] += 1
        if step_ref[0] % skip_every == 0:
            return pos.sum() * 0.0
        return loss_fn(pos, node_sizes, layer_index)

    return _amortized_loss


def _overlap_interval(num_nodes: int, config: LayoutConfig) -> int:
    """How often to run overlap projection (every N steps)."""
    if config.overlap_check_interval > 0:
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
    base_rank_sep: float = 45.0,
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


def _prepare_flex_data(
    config: LayoutConfig,
    num_nodes: int,
    device: str,
) -> dict:
    """Extract flex constraints from config into tensor form for the optimization loop.

    Returns a dict with pre-computed tensors for pins, alignment groups, and flex spacing.
    All node ID → index resolution happens here (not per-step).
    """
    result = {
        "has_soft_pins": False,
        "has_hard_pins": False,
        "pin_indices": torch.zeros(0, dtype=torch.long, device=device),
        "pin_targets": torch.zeros(0, 2, dtype=torch.float32, device=device),
        "pin_weights": torch.zeros(0, 2, dtype=torch.float32, device=device),
        "soft_pin_mask": torch.zeros(0, 2, dtype=torch.bool, device=device),
        "hard_pin_mask": torch.zeros(0, 2, dtype=torch.bool, device=device),
        "align_groups": [],
        "flex_node_sep": None,
        "flex_node_sep_weight": 0.0,
    }

    flex = config.flex
    if flex is None:
        return result

    # --- Pins ---
    if flex.pins:
        indices = []
        targets = []
        weights = []
        soft_mask = []
        hard_mask = []

        for node_id, (fx, fy) in flex.pins.items():
            # node_id is already an index (int) in headless mode
            if isinstance(node_id, int) and 0 <= node_id < num_nodes:
                idx = node_id
            else:
                continue  # Skip unresolvable IDs in headless mode

            tx = fx.target if fx is not None else 0.0
            ty = fy.target if fy is not None else 0.0
            wx = fx.weight if fx is not None else 0.0
            wy = fy.weight if fy is not None else 0.0
            has_x = fx is not None
            has_y = fy is not None
            hard_x = bool(has_x and fx is not None and fx.is_hard)
            hard_y = bool(has_y and fy is not None and fy.is_hard)
            soft_x = has_x and not hard_x
            soft_y = has_y and not hard_y

            indices.append(idx)
            targets.append([tx, ty])
            weights.append([wx if not hard_x else 0.0, wy if not hard_y else 0.0])
            soft_mask.append([soft_x, soft_y])
            hard_mask.append([hard_x, hard_y])

        if indices:
            result["pin_indices"] = torch.tensor(indices, dtype=torch.long, device=device)
            result["pin_targets"] = torch.tensor(targets, dtype=torch.float32, device=device)
            result["pin_weights"] = torch.tensor(weights, dtype=torch.float32, device=device)
            result["soft_pin_mask"] = torch.tensor(soft_mask, dtype=torch.bool, device=device)
            result["hard_pin_mask"] = torch.tensor(hard_mask, dtype=torch.bool, device=device)
            result["has_soft_pins"] = any(any(row) for row in soft_mask)
            result["has_hard_pins"] = any(any(row) for row in hard_mask)

    # --- Alignment groups ---
    align_groups = []
    for groups, axis in [(flex.align_x, 0), (flex.align_y, 1)]:
        if groups is None:
            continue
        for group in groups:
            idx_list = []
            for node_id in group.nodes:
                if isinstance(node_id, int) and 0 <= node_id < num_nodes:
                    idx_list.append(node_id)
            if len(idx_list) >= 2:
                align_groups.append(
                    (
                        torch.tensor(idx_list, dtype=torch.long, device=device),
                        group.weight,
                        axis,
                    )
                )
    result["align_groups"] = align_groups

    # --- Flex spacing ---
    if flex.node_sep is not None:
        result["flex_node_sep"] = flex.node_sep.target
        result["flex_node_sep_weight"] = flex.node_sep.weight

    return result


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


def _resolve_flex_ids(config: LayoutConfig, graph) -> LayoutConfig:
    """Resolve flex node IDs in config to integer indices using graph._id_to_index.

    Returns config unchanged if no flex, or a shallow copy with resolved flex.
    """
    if config.flex is None:
        return config
    return _resolve_config_flex(config, graph._id_to_index)


def _resolve_config_flex(config: LayoutConfig, id_to_index: dict) -> LayoutConfig:
    """Create a config copy with flex node IDs resolved to indices."""
    import copy as _c

    resolved_flex = _resolve_graph_flex(config.flex, id_to_index)
    if resolved_flex is config.flex:
        return config
    new_config = _c.copy(config)
    new_config.flex = resolved_flex
    return new_config


def _resolve_graph_flex(flex, id_to_index: dict):
    """Resolve node IDs in a LayoutFlex to integer indices."""
    from dagua.flex import AlignGroup, LayoutFlex

    changed = False

    # Resolve pins
    new_pins = None
    if flex.pins:
        new_pins = {}
        for node_id, val in flex.pins.items():
            if isinstance(node_id, int):
                new_pins[node_id] = val
            elif node_id in id_to_index:
                new_pins[id_to_index[node_id]] = val
                changed = True
            # else: skip unresolvable

    # Resolve align groups
    new_align_x = None
    if flex.align_x:
        new_align_x = []
        for group in flex.align_x:
            resolved_nodes = []
            for nid in group.nodes:
                if isinstance(nid, int):
                    resolved_nodes.append(nid)
                elif nid in id_to_index:
                    resolved_nodes.append(id_to_index[nid])
                    changed = True
            new_align_x.append(AlignGroup(nodes=resolved_nodes, weight=group.weight))

    new_align_y = None
    if flex.align_y:
        new_align_y = []
        for group in flex.align_y:
            resolved_nodes = []
            for nid in group.nodes:
                if isinstance(nid, int):
                    resolved_nodes.append(nid)
                elif nid in id_to_index:
                    resolved_nodes.append(id_to_index[nid])
                    changed = True
            new_align_y.append(AlignGroup(nodes=resolved_nodes, weight=group.weight))

    if not changed:
        return flex

    return LayoutFlex(
        node_sep=flex.node_sep,
        rank_sep=flex.rank_sep,
        pins=new_pins if new_pins is not None else flex.pins,
        align_x=new_align_x if new_align_x is not None else flex.align_x,
        align_y=new_align_y if new_align_y is not None else flex.align_y,
    )
