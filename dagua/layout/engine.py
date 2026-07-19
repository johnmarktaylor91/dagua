"""Core optimization loop -- the dispatch + legacy engine body.

status (2026-04-22):
- Top-level `layout()` is the active dispatcher.
  * `algorithm=None` -> "dagua_native" ops pipeline (the new default).
  * `algorithm="_legacy"` -> the legacy `_layout_inner` body in this file
    (opt-in, unadvertised; preserved until ports the memory
    features it owns into ops).
  * `algorithm=<other>` -> existing pipeline dispatch.
- Animation (trace argument) keeps the legacy path until op-level snapshot
  hooks land.
- The `_layout_inner` body and its helpers below are SCHEDULED FOR ARCHIVE
  to `dagua/layout/_archive/legacy_engine/` once demonstrates
  memory parity and re-implements per-loss backward, gradient checkpointing,
  and hybrid device support as registered ops. Per 09 Q1 (user resolution:
  archive). Do NOT add new features to `_layout_inner`; new optimization
  work goes into ops + resolve.

Scaling strategy of the LEGACY body (tiered):
- Tier 0 (N < 500): exact O(N^2) repulsion, full overlap check
- Tier 1 (500-20K): scatter sampling repulsion, layer-local overlap
- Tier 2 (N > 20K by default): multilevel coarsening V-cycle

Memory optimization in the LEGACY body (composable, auto-enabled for large graphs):
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
import json
import math
import os
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Literal, Optional, Union, cast

import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from dagua.config import LayoutConfig
from dagua.layout.graph_classify import GraphFamily, GraphStructure, classify_graph
from dagua.layout.init_placement import init_positions
from dagua.layout.layers import LayerIndex, build_layer_index
from dagua.layout.losses import (
    alignment_loss,
    back_edge_compactness_loss,
    cluster_compactness_loss,
    cluster_containment_loss,
    cluster_separation_loss,
    constraint_anchor_loss,
    constraint_contain_loss,
    constraint_emphasize_loss,
    constraint_focus_loss,
    constraint_group_loss,
    constraint_order_loss,
    constraint_separate_loss,
    crossing_loss,
    dag_ordering_loss,
    edge_attraction_loss,
    edge_length_variance_loss,
    edge_straightness_loss,
    fanout_distribution_loss,
    flex_spacing_loss,
    overlap_avoidance_loss,
    position_pin_loss,
    project_hard_contains,
    project_hard_pins,
    repulsion_loss,
    spacing_consistency_loss,
    warn_legacy_cluster_loss_config,
)
from dagua.layout.projection import project_overlaps
from dagua.utils import VRAMBudget, longest_path_layering

LossFn = Callable[[torch.Tensor, torch.Tensor, Optional[LayerIndex]], torch.Tensor]

_SAMPLED_NODE_BYTES_PER_ROW = 1_200
_SAMPLED_NODE_VRAM_FRACTION = 0.30
_SAMPLED_NODE_MAX_CAP = 2_000_000
_BYTES_PER_EDGE_ESTIMATE = 120
_EDGE_BATCH_VRAM_FRACTION = 0.60
_EDGE_BATCH_MIN = 1_000_000
_EDGE_BATCH_MAX = 50_000_000
SAMPLED_NODE_CONTEXT_CAP = _SAMPLED_NODE_MAX_CAP
MIN_SAMPLED_ACTIVE_SET = 10_000
SAMPLED_SAME_LAYER_K = 64
REPULSION_ESTIMATE_NN_K = 20
AUTOGRAD_INTERMEDIATE_FACTOR = 2.0
INTERMEDIATE_SAFETY_FACTOR = 1.5
CHECKPOINT_MEMORY_REDUCTION = 2
CUDA_ACTIVE_SET_HEADROOM = 0.85
DEFAULT_HYBRID_EDGE_BATCH = _EDGE_BATCH_MIN
CUDA_CONTEXT_OVERHEAD_BYTES = 500 * 1024 * 1024
TILED_GPU_MIN_NODES = 50_000_000
STANDARD_CUDA_RESIDENT_HEADROOM = 0.85
STANDARD_CUDA_WORKING_SET_BYTES = 2_000_000_000
SUBSET_GPU_REQUIRED_THRESHOLD = 50_000_000
SUBSET_GPU_SAMPLED_BASE_BYTES = 256 * 1024 * 1024
SUBSET_GPU_BYTES_PER_SAMPLE = 64
HYBRID_OPTIMIZER_LR_MULTIPLIERS = {
    "adam": 1.0,
    "sgd_nesterov": 3.0,
    "sgd": 5.0,
}
_PROGRESS_LOG_INTERVAL_SECONDS = 30.0
_PROGRESS_FILE_INTERVAL_SECONDS = 60.0
_PROGRESS_BATCH_FRACTION = 0.25
_PROGRESS_FILE_NODE_THRESHOLD = 1_000_000
_DEFAULT_PROGRESS_FILENAME = "progress.json"
OptimizerType = Literal["adam", "sgd_nesterov", "sgd"]
ExecutionMode = Literal["standard", "subset_gpu"]


def _build_cluster_inner_pipeline(algorithm: str, config: LayoutConfig) -> Optional[Any]:
    """Build an op pipeline suitable for recursive cluster placement.

    Parameters
    ----------
    algorithm : str
        Configured algorithm name.
    config : LayoutConfig
        Layout configuration used to forward common algorithm parameters.

    Returns
    -------
    Any or None
        Pipeline object for supported algorithms, otherwise ``None``.
    """
    steps = config.steps if config.steps > 0 else 50
    normalized = algorithm.lower()
    if normalized == "fr":
        from dagua.layout.ops.pipelines.fr import build_fr_pipeline

        return build_fr_pipeline(steps=steps)
    if normalized == "kk":
        from dagua.layout.ops.pipelines.kk import build_kk_pipeline

        return build_kk_pipeline(steps=steps)
    if normalized == "fa2":
        from dagua.layout.ops.pipelines.fa2 import FA2Config, build_fa2_pipeline

        return build_fa2_pipeline(config=FA2Config(steps=steps))
    if normalized == "sfdp":
        from dagua.layout.ops.pipelines.sfdp import build_sfdp_pipeline

        return build_sfdp_pipeline(steps=steps)
    if normalized == "native_stress":
        from dagua.layout.ops.base import Pipeline
        from dagua.layout.ops.cluster_driver import NativeClusterLevelLayout

        return Pipeline(
            [NativeClusterLevelLayout(algorithm=normalized, config=config)],
            name=f"{normalized}_cluster_inner",
        )
    return None


def _effective_cluster_side_padding(graph: Any, config: LayoutConfig) -> float:
    """Resolve placement padding against currently rendered cluster styles.

    Parameters
    ----------
    graph : Any
        Graph exposing cluster style metadata.
    config : LayoutConfig
        Layout configuration with the public cluster padding knob.

    Returns
    -------
    float
        Padding value used for cluster placeholder footprints.
    """
    padding = float(config.cluster_side_padding_pt)
    default_style = getattr(graph, "default_cluster_style", None)
    if default_style is not None:
        padding = max(padding, float(getattr(default_style, "padding", padding)))
    theme = getattr(graph, "_theme", None)
    theme_style = getattr(theme, "cluster_style", None)
    if theme_style is not None:
        padding = max(padding, float(getattr(theme_style, "padding", padding)))
    for style in getattr(graph, "cluster_styles", {}).values():
        padding = max(padding, float(getattr(style, "padding", padding)))
    return padding


def _layout_cluster_aware_pipeline(graph: Any, config: LayoutConfig) -> Optional[torch.Tensor]:
    """Run the recursive cluster-aware driver for supported algorithms.

    Parameters
    ----------
    graph : Any
        Graph-like object exposing edge, node, and cluster metadata.
    config : LayoutConfig
        User layout configuration.

    Returns
    -------
    torch.Tensor or None
        Final positions when the selected algorithm supports recursive cluster
        placement, otherwise ``None`` so the caller can fall back.

    Raises
    ------
    RuntimeError
        If the recursive driver fails to produce positions.
    """
    if config.algorithm is None:
        return None
    inner_pipeline = _build_cluster_inner_pipeline(config.algorithm, config)
    if inner_pipeline is None:
        return None

    from dagua.layout.ops.cluster_driver import ClusterAwareDriver
    from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState

    problem = LayoutProblem(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        node_sizes=graph.node_sizes,
        direction=config.direction,
        clusters=graph.clusters,
        cluster_parents=graph.cluster_parents,
        edge_weights=getattr(graph, "edge_weights", None),
        seed=config.seed or 42,
    )
    driver = ClusterAwareDriver(
        inner_pipeline=inner_pipeline.ops,
        side_padding_pt=_effective_cluster_side_padding(graph, config),
        label_band_pt=config.cluster_label_band_pt,
        external_clearance_pt=config.cluster_external_clearance_pt,
        cluster_compactness_weight=config.w_cluster,
    )
    ctx = RuntimeContext(plan=ExecutionPlan(device=str(graph.node_sizes.device)))
    final_state = driver.apply(problem, SolveState(), ctx)
    if final_state.pos is None:
        raise RuntimeError("Cluster-aware layout did not produce positions.")
    return final_state.pos


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


def _resolve_progress_file_path(
    config: LayoutConfig,
    progress_file_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Resolve the monitoring file path for large layout runs.

    Parameters
    ----------
    config : LayoutConfig
        Layout configuration that may expose a ``progress_path`` override.
    progress_file_path : str | Path, optional
        Explicit progress file path forwarded by the caller.

    Returns
    -------
    Path
        Absolute path for the JSON progress file.
    """
    configured_path = progress_file_path
    if configured_path is None:
        configured_path = cast(Optional[Union[str, Path]], getattr(config, "progress_path", None))
    if configured_path is None:
        configured_path = Path.cwd() / _DEFAULT_PROGRESS_FILENAME
    return Path(configured_path).expanduser().resolve()


def _current_vram_mb() -> int:
    """Return the currently allocated VRAM in megabytes.

    Returns
    -------
    int
        Current CUDA allocation rounded to the nearest megabyte, or ``0`` when
        CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return 0
    return int(round(torch.cuda.memory_allocated() / 1024**2))


def _current_rss_gb() -> float:
    """Return the current resident set size in gigabytes.

    Returns
    -------
    float
        Current process RSS in gigabytes. Linux ``/proc`` is used when
        available so monitoring reflects the live resident set instead of the
        peak value exposed by ``resource``.
    """
    statm_path = Path("/proc/self/statm")
    if statm_path.exists():
        try:
            pages = statm_path.read_text(encoding="utf-8").split()
            if len(pages) >= 2:
                rss_pages = int(pages[1])
                page_size = os.sysconf("SC_PAGE_SIZE")
                return rss_pages * page_size / 1024**3
        except (OSError, ValueError):
            return 0.0
    return 0.0


def _write_progress_json(path: Path, payload: dict[str, object]) -> None:
    """Write progress data atomically for external polling.

    Parameters
    ----------
    path : Path
        Destination JSON path.
    payload : dict[str, object]
        Serializable progress payload to write.

    Returns
    -------
    None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def _compute_edge_batch_progress(
    step: int,
    num_edges: int,
    edge_batch: int,
    random_interval: int,
) -> Optional[tuple[int, int]]:
    """Return the current edge-batch position for verbose progress logs.

    Parameters
    ----------
    step : int
        Zero-based optimizer step.
    num_edges : int
        Total edge count for the current solve.
    edge_batch : int
        Active edge batch size. ``0`` means the full edge set is resident.
    random_interval : int
        Sampling cadence for random edge re-mixing. A value of ``0`` means
        contiguous chunks are always used.

    Returns
    -------
    tuple[int, int] | None
        One-based current batch index and total number of batches when edge
        streaming is active; otherwise ``None``.
    """
    if edge_batch <= 0 or num_edges <= edge_batch:
        return None
    total_batches = max(int(math.ceil(num_edges / edge_batch)), 1)
    if random_interval > 0:
        batch_index = (step % total_batches) + 1
    else:
        start = (step * edge_batch) % num_edges
        batch_index = (start // edge_batch) + 1
    return batch_index, total_batches


def _progress_strategy(
    use_per_loss_bw: bool,
    use_checkpointing: bool,
    use_hybrid: bool,
    edges_on_cpu: bool,
    tiled_compute_active: bool,
    execution_mode: ExecutionMode,
) -> str:
    """Return the compact strategy string stored in ``progress.json``.

    Parameters
    ----------
    use_per_loss_bw : bool
        Whether per-loss backward execution is active.
    use_checkpointing : bool
        Whether gradient checkpointing is active.
    use_hybrid : bool
        Whether hybrid CPU/GPU execution is active.
    edges_on_cpu : bool
        Whether edge tensors are streamed from CPU.
    tiled_compute_active : bool
        Whether tiled GPU execution is active.
    execution_mode : {"standard", "subset_gpu"}
        Effective execution mode for the current solve.

    Returns
    -------
    str
        Comma-separated strategy flags, or ``"standard"`` when no special
        strategy is active.
    """
    flags: list[str] = []
    if use_per_loss_bw:
        flags.append("per_loss_bw")
    if use_checkpointing:
        flags.append("checkpoint")
    if use_hybrid:
        flags.append("hybrid")
    if edges_on_cpu:
        flags.append("edge_stream")
    if tiled_compute_active:
        flags.append("tiled_gpu")
    if execution_mode == "subset_gpu":
        flags.append("subset_gpu")
    return ",".join(flags) if flags else "standard"


def _format_auto_count(value: int) -> str:
    """Return a compact string for an auto-sized count.

    Parameters
    ----------
    value : int
        Absolute count selected by an auto-sizing helper.

    Returns
    -------
    str
        Human-readable count using ``k`` and ``M`` suffixes where helpful.
    """
    if value >= 1_000_000:
        scaled = value / 1_000_000.0
        return f"{scaled:.1f}".rstrip("0").rstrip(".") + "M"
    if value >= 1_000:
        scaled = value / 1_000.0
        return f"{scaled:.1f}".rstrip("0").rstrip(".") + "k"
    return f"{value:,}"


def _format_auto_gb(num_bytes: int) -> str:
    """Return a compact string for a byte budget in gigabytes.

    Parameters
    ----------
    num_bytes : int
        Memory quantity in bytes.

    Returns
    -------
    str
        Rounded value in gigabytes.
    """
    return f"{num_bytes / 1024**3:.1f}GB"


def _log_auto_allocation(
    label: str,
    value: int,
    free_bytes: int,
    budget_fraction: float,
    verbose: bool,
) -> None:
    """Emit a verbose auto-allocation decision log.

    Parameters
    ----------
    label : str
        Resource label being sized.
    value : int
        Auto-selected count or byte budget.
    free_bytes : int
        Free memory observed while sizing the resource.
    budget_fraction : float
        Fraction of free memory reserved for this resource.
    verbose : bool
        Whether verbose logging is enabled for the current layout call.
    """
    if not verbose:
        return
    print(
        f"[dagua] Auto {label}: {_format_auto_count(value)} "
        f"({_format_auto_gb(free_bytes)} free, {budget_fraction:.0%} budget)",
        flush=True,
    )


def _auto_sampled_node_cap(verbose: bool = False) -> int:
    """Choose the sampled-node active-set cap from free VRAM.

    Parameters
    ----------
    verbose : bool, default=False
        Whether to log the selected cap.

    Returns
    -------
    int
        Active-row cap for sampled repulsion and overlap contexts. Falls back
        to ``MIN_SAMPLED_ACTIVE_SET`` when CUDA telemetry is unavailable.
    """
    max_cap = max(SAMPLED_NODE_CONTEXT_CAP, 0)
    fallback_cap = min(MIN_SAMPLED_ACTIVE_SET, max_cap)
    if not torch.cuda.is_available():
        if verbose:
            print(
                f"[dagua] Auto sampled node cap: {fallback_cap:,} (CUDA unavailable, fallback)",
                flush=True,
            )
        return fallback_cap
    try:
        free_vram, _ = torch.cuda.mem_get_info()
    except RuntimeError:
        if verbose:
            print(
                f"[dagua] Auto sampled node cap: {fallback_cap:,} "
                "(VRAM telemetry unavailable, fallback)",
                flush=True,
            )
        return fallback_cap
    safe_budget = int(free_vram * _SAMPLED_NODE_VRAM_FRACTION)
    sampled_cap = max(MIN_SAMPLED_ACTIVE_SET, safe_budget // _SAMPLED_NODE_BYTES_PER_ROW)
    sampled_cap = min(sampled_cap, _SAMPLED_NODE_MAX_CAP, max_cap)
    _log_auto_allocation(
        label="sampled node cap",
        value=sampled_cap,
        free_bytes=int(free_vram),
        budget_fraction=_SAMPLED_NODE_VRAM_FRACTION,
        verbose=verbose,
    )
    return sampled_cap


def _auto_edge_batch_size(verbose: bool = False) -> int:
    """Choose the largest edge batch that fits in available VRAM.

    Parameters
    ----------
    verbose : bool, default=False
        Whether to log the selected batch size.

    Returns
    -------
    int
        Edge batch size scaled to current free VRAM. Falls back to the
        minimum batch when CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        if verbose:
            print(
                f"[dagua] Auto edge batch: {DEFAULT_HYBRID_EDGE_BATCH:,} "
                "(CUDA unavailable, fallback)",
                flush=True,
            )
        return DEFAULT_HYBRID_EDGE_BATCH
    try:
        free_vram, _ = torch.cuda.mem_get_info()
    except RuntimeError:
        if verbose:
            print(
                f"[dagua] Auto edge batch: {DEFAULT_HYBRID_EDGE_BATCH:,} "
                "(VRAM telemetry unavailable, fallback)",
                flush=True,
            )
        return DEFAULT_HYBRID_EDGE_BATCH
    safe_budget = int(free_vram * _EDGE_BATCH_VRAM_FRACTION)
    batch = max(_EDGE_BATCH_MIN, safe_budget // _BYTES_PER_EDGE_ESTIMATE)
    batch = min(batch, _EDGE_BATCH_MAX)
    _log_auto_allocation(
        label="edge batch",
        value=batch,
        free_bytes=int(free_vram),
        budget_fraction=_EDGE_BATCH_VRAM_FRACTION,
        verbose=verbose,
    )
    return batch


def _auto_subset_gpu_sampled_budget(verbose: bool = False) -> int:
    """Choose the subset-GPU sampled transfer budget from free VRAM.

    Parameters
    ----------
    verbose : bool, default=False
        Whether to log the selected transfer budget.

    Returns
    -------
    int
        Byte budget reserved for copying sampled node subsets to CUDA. Falls
        back to ``SUBSET_GPU_SAMPLED_BASE_BYTES`` when CUDA telemetry is
        unavailable.
    """
    if not torch.cuda.is_available():
        if verbose:
            print(
                "[dagua] Auto subset-GPU sampled budget: "
                f"{_format_auto_gb(SUBSET_GPU_SAMPLED_BASE_BYTES)} "
                "(CUDA unavailable, fallback)",
                flush=True,
            )
        return SUBSET_GPU_SAMPLED_BASE_BYTES
    try:
        free_vram, _ = torch.cuda.mem_get_info()
    except RuntimeError:
        if verbose:
            print(
                "[dagua] Auto subset-GPU sampled budget: "
                f"{_format_auto_gb(SUBSET_GPU_SAMPLED_BASE_BYTES)} "
                "(VRAM telemetry unavailable, fallback)",
                flush=True,
            )
        return SUBSET_GPU_SAMPLED_BASE_BYTES
    budget = max(int(free_vram * 0.25), 0)
    if verbose:
        print(
            f"[dagua] Auto subset-GPU sampled budget: {_format_auto_gb(budget)} "
            f"({_format_auto_gb(int(free_vram))} free, 25% budget)",
            flush=True,
        )
    return budget


def _build_sampled_node_context(
    num_nodes: int,
    layer_index: LayerIndex,
    device: Union[str, torch.device],
    rvs_nn_k: int,
    verbose: bool = False,
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
    verbose : bool, default=False
        Whether to log the auto-selected sampled-node cap.

    Returns
    -------
    SampledNodeContext
        Shared active-node indices and structured neighbor samples. The sampled
        tensor stores same-layer samples first, followed by adjacent-layer
        random samples.
    """
    n_active, _, _ = _sampled_node_context_sizes(num_nodes, rvs_nn_k, verbose=verbose)
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
    verbose: bool = False,
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
    verbose : bool, default=False
        Whether to log the auto-selected sampled-node cap.

    Returns
    -------
    tuple[int, int, int]
        ``(n_active, n_random, k_same)`` for the sampled context.
    """
    auto_cap = _auto_sampled_node_cap(verbose=verbose)
    default_n_active = min(max(int(num_nodes**0.75), min(num_nodes, 256)), auto_cap)
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


def _config_or_graph_has_constraints(config: LayoutConfig, graph: Any) -> bool:
    """Return whether a layout request carries R9 user constraints.

    Parameters
    ----------
    config : LayoutConfig
        Layout configuration.
    graph : Any
        Graph-like object that may carry graph-scoped flex constraints.

    Returns
    -------
    bool
        ``True`` when graph or config flex contains at least one constraint.
    """
    config_flex = getattr(config, "flex", None)
    graph_flex = getattr(graph, "flex", None)
    return bool(getattr(config_flex, "constraints", None)) or bool(
        getattr(graph_flex, "constraints", None)
    )


def _config_or_graph_has_flex(config: LayoutConfig, graph: Any) -> bool:
    """Return whether a layout request carries scalar flex or constraints.

    Parameters
    ----------
    config : LayoutConfig
        Layout configuration.
    graph : Any
        Graph-like object that may carry graph-scoped flex.

    Returns
    -------
    bool
        ``True`` when a non-empty flex field should be forwarded.
    """
    for flex in (getattr(config, "flex", None), getattr(graph, "flex", None)):
        if flex is None:
            continue
        if (
            getattr(flex, "node_sep", None) is not None
            or getattr(flex, "rank_sep", None) is not None
        ):
            return True
        if getattr(flex, "constraints", None):
            return True
    return False


def _effective_constraint_config(config: LayoutConfig, graph: Any) -> LayoutConfig:
    """Return a config whose flex constraints are the canonical effective set.

    Parameters
    ----------
    config : LayoutConfig
        User layout configuration.
    graph : Any
        Graph-like object that may carry graph-owned constraints.

    Returns
    -------
    LayoutConfig
        Shallow config copy with graph and config constraints merged unless
        ``replace_constraints`` suppresses graph-owned constraints.
    """
    from dagua.flex import LayoutFlex

    if bool(getattr(config, "_dagua_effective_constraints", False)):
        return config

    graph_flex = getattr(graph, "flex", None)
    config_flex = getattr(config, "flex", None)
    graph_constraints = (
        [] if config.replace_constraints else list(getattr(graph_flex, "constraints", None) or [])
    )
    config_constraints = list(getattr(config_flex, "constraints", None) or [])
    if (
        (graph_constraints or config_constraints)
        and float(getattr(config, "cluster_cohesion", 1.0)) > 0.0
        and getattr(graph, "clusters", None)
    ):
        from dagua.constraints import Group

        system_strength = 0.5 * float(getattr(config, "cluster_cohesion", 1.0))
        for cluster_name, members in graph.clusters.items():
            member_list = list(members)
            if len(member_list) >= 2:
                graph_constraints.append(
                    Group(
                        member_list,
                        strength=system_strength,
                        name=f"cluster:{cluster_name}",
                        system=True,
                        tags=("system", "cluster"),
                    )
                )
    if not graph_constraints and not config_constraints:
        if config_flex is None and graph_flex is not None:
            effective = copy.copy(config)
            effective.flex = graph_flex
            resolved = _resolve_flex_ids(effective, graph)
            setattr(resolved, "_dagua_effective_constraints", True)
            return resolved
        resolved = _resolve_flex_ids(config, graph)
        setattr(resolved, "_dagua_effective_constraints", True)
        return resolved

    effective = copy.copy(config)
    effective.flex = LayoutFlex(
        node_sep=(
            getattr(config_flex, "node_sep", None)
            if getattr(config_flex, "node_sep", None) is not None
            else getattr(graph_flex, "node_sep", None)
        ),
        rank_sep=(
            getattr(config_flex, "rank_sep", None)
            if getattr(config_flex, "rank_sep", None) is not None
            else getattr(graph_flex, "rank_sep", None)
        ),
        constraints=graph_constraints + config_constraints,
    )
    resolved = _resolve_flex_ids(effective, graph)
    setattr(resolved, "_dagua_effective_constraints", True)
    return resolved


def _update_graph_constraint_report(graph: Any, pos: torch.Tensor, config: LayoutConfig) -> None:
    """Attach a lightweight constraint report to the graph constraint set.

    Parameters
    ----------
    graph : Any
        Graph-like object carrying a ``constraints`` property.
    pos : torch.Tensor
        Final position tensor with shape ``[N, 2]``.
    config : LayoutConfig
        Layout configuration containing the active policy.

    Returns
    -------
    None
        Stores the report on ``graph.constraints`` when constraints exist.
    """
    effective_config = _effective_constraint_config(config, graph)
    effective_flex = getattr(effective_config, "flex", None)
    constraints = list(getattr(effective_flex, "constraints", None) or [])
    if not constraints:
        return
    _apply_label_satellite_constraints(graph, pos, constraints)
    policy = getattr(config, "constraint_policy", "warn")
    from dagua.constraints import build_constraint_report

    report = build_constraint_report(
        constraints,
        pos,
        graph,
        policy=policy,
        direction=getattr(effective_config, "direction", "TB"),
    )
    graph.constraints.set_report(report)
    hard_violations = [row for row in report.violations if row.constraint.is_hard]
    if hard_violations and policy == "strict":
        from dagua.constraints import ConstraintConflictError

        raise ConstraintConflictError(report.summary())
    if report.violations and policy == "warn":
        import warnings

        warnings.warn(report.summary(), RuntimeWarning, stacklevel=2)


def _apply_label_satellite_constraints(
    graph: Any,
    pos: torch.Tensor,
    constraints: List[Any],
) -> None:
    """Record constrained label satellite positions without moving owner nodes.

    Parameters
    ----------
    graph : Any
        Graph-like object that owns label metadata.
    pos : torch.Tensor
        Final node positions with shape ``[N, 2]``.
    constraints : list[Any]
        Effective constraints to scan for label pins.

    Returns
    -------
    None
        Stores a private ``_r9_label_positions`` mapping on ``graph``.
    """
    from dagua.constraints import Pin, Selector, resolve_canvas_point

    label_positions = dict(getattr(graph, "_r9_label_positions", {}) or {})
    for constraint in constraints:
        if not isinstance(constraint, Pin):
            continue
        if not isinstance(constraint.sel, Selector) or constraint.sel.kind not in {
            "label",
            "labels",
        }:
            continue
        at_x, at_y = resolve_canvas_point(constraint.at, pos)
        target_x = constraint.x if constraint.x is not None else at_x
        target_y = constraint.y if constraint.y is not None else at_y
        if target_x is None and target_y is None:
            continue
        label_key = constraint.sel
        owner_indices = []
        try:
            from dagua.constraints import resolve_label_selection

            owner_indices = resolve_label_selection(label_key, graph, allow_empty=True)
        except Exception:
            owner_indices = []
        for owner_index in owner_indices:
            current = pos[int(owner_index)].detach().cpu()
            label_positions[(label_key.kind, repr(label_key.args), int(owner_index))] = (
                float(current[0].item()) if target_x is None else float(target_x),
                float(current[1].item()) if target_y is None else float(target_y),
            )
    setattr(graph, "_r9_label_positions", label_positions)


def _apply_constrained_polish(
    pos: torch.Tensor,
    graph: Any,
    config: LayoutConfig,
) -> torch.Tensor:
    """Polish pipeline output with R9 constraint losses.

    Parameters
    ----------
    pos : torch.Tensor
        Pipeline output positions with shape ``[N, 2]``.
    graph : Any
        Graph carrying topology, node sizes, and graph-owned constraints.
    config : LayoutConfig
        Layout configuration with optional constraint overlays.

    Returns
    -------
    torch.Tensor
        Polished positions. Unconstrained layouts are returned byte-identical.
    """
    if not _config_or_graph_has_constraints(config, graph):
        return pos
    effective_config = _effective_constraint_config(config, graph)
    if effective_config.flex is None:
        return pos

    device = str(pos.device)
    graph.compute_node_sizes()
    edge_index = graph.edge_index.to(device)
    node_sizes = graph.node_sizes.to(device)
    flex_data = _prepare_flex_data(
        effective_config,
        graph.num_nodes,
        device,
        edge_index=edge_index,
        node_sizes=node_sizes,
    )
    if not _flex_data_has_constraints(flex_data):
        return pos

    anchor = pos.detach().clone().to(device=device, dtype=torch.float32)
    polished = anchor.clone().detach().requires_grad_(True)
    steps = max(5, min(50, int(effective_config.steps or 20)))
    optimizer = torch.optim.Adam([polished], lr=max(float(effective_config.lr), 0.01))
    for _step in range(steps):
        optimizer.zero_grad()
        loss = (polished - anchor).square().mean() * 10.0
        loss = loss + _constraint_loss_from_flex_data(polished, flex_data)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([polished], max_norm=100.0)
        optimizer.step()
        _project_hard_constraints_from_flex_data(polished, flex_data)
    polished = polished.detach()
    _project_hard_constraints_from_flex_data(polished, flex_data)
    return polished.to(dtype=pos.dtype, device=pos.device)


def _flex_data_has_constraints(flex_data: dict[str, Any]) -> bool:
    """Return whether lowered flex data contains any R9 constraints.

    Parameters
    ----------
    flex_data : dict[str, Any]
        Lowered constraint payloads.

    Returns
    -------
    bool
        ``True`` when a constraint payload is present.
    """
    keys = (
        "has_soft_pins",
        "has_hard_pins",
        "align_groups",
        "hard_align_groups",
        "order_pairs",
        "hard_order_pairs",
        "separate_pairs",
        "group_constraints",
        "hard_groups",
        "anchor_constraints",
        "emphasize_paths",
        "focus_constraints",
        "soft_contain",
        "hard_contain",
        "custom_losses",
        "custom_projectors",
    )
    return any(bool(flex_data[key]) for key in keys)


def _constraint_loss_from_flex_data(pos: torch.Tensor, flex_data: dict[str, Any]) -> torch.Tensor:
    """Evaluate finite R9 constraint losses from lowered payloads.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    flex_data : dict[str, Any]
        Lowered constraint payloads.

    Returns
    -------
    torch.Tensor
        Scalar loss tensor.
    """
    loss = torch.zeros((), device=pos.device, dtype=pos.dtype)
    if flex_data["has_soft_pins"]:
        loss = loss + position_pin_loss(
            pos,
            flex_data["pin_indices"],
            flex_data["pin_targets"],
            flex_data["pin_weights"],
            flex_data["soft_pin_mask"],
        )
    if flex_data["align_groups"]:
        loss = loss + alignment_loss(pos, flex_data["align_groups"])
    if flex_data["order_pairs"]:
        loss = loss + constraint_order_loss(pos, flex_data["order_pairs"])
    if flex_data["separate_pairs"]:
        loss = loss + constraint_separate_loss(pos, flex_data["separate_pairs"])
    if flex_data["group_constraints"]:
        loss = loss + constraint_group_loss(pos, flex_data["group_constraints"])
    if flex_data["anchor_constraints"]:
        loss = loss + constraint_anchor_loss(pos, flex_data["anchor_constraints"])
    if flex_data["emphasize_paths"]:
        loss = loss + constraint_emphasize_loss(pos, flex_data["emphasize_paths"])
    if flex_data["focus_constraints"]:
        loss = loss + constraint_focus_loss(pos, flex_data["focus_constraints"])
    if flex_data["soft_contain"]:
        loss = loss + constraint_contain_loss(pos, flex_data["soft_contain"])
    for custom_constraint in flex_data["custom_losses"]:
        loss = loss + custom_constraint.loss(pos, flex_data["constraint_context"])
    return loss


def _project_hard_constraints_from_flex_data(pos: torch.Tensor, flex_data: dict[str, Any]) -> None:
    """Run hard R9 projections in the documented order.

    Parameters
    ----------
    pos : torch.Tensor
        Mutable position tensor with shape ``[N, 2]``.
    flex_data : dict[str, Any]
        Lowered constraint payloads.

    Returns
    -------
    None
        Mutates ``pos`` in place.
    """
    if flex_data["hard_contain"]:
        project_hard_contains(pos, flex_data["hard_contain"])
    if flex_data["hard_groups"]:
        with torch.no_grad():
            for indices, _padding, _weight in flex_data["hard_groups"]:
                if indices.numel() > 1:
                    pos[indices] = pos[indices].mean(dim=0, keepdim=True)
    if flex_data["hard_align_groups"] or flex_data["hard_order_pairs"]:
        _project_hard_vpsc_constraints(
            pos,
            flex_data["hard_align_groups"],
            flex_data["hard_order_pairs"],
        )
    if flex_data["custom_projectors"]:
        with torch.no_grad():
            for custom_constraint in flex_data["custom_projectors"]:
                custom_constraint.project(pos, flex_data["constraint_context"])
    if flex_data["has_hard_pins"]:
        project_hard_pins(
            pos,
            flex_data["pin_indices"],
            flex_data["pin_targets"],
            flex_data["hard_pin_mask"],
        )


def layout(graph: Any, config: Optional[LayoutConfig] = None, trace: Any = None) -> torch.Tensor:
    """Compute layout positions for all nodes.

    Parameters
    ----------
    graph : Any
        Graph-like object exposing the Dagua layout interface, including
        topology tensors, node-size preparation, and layout-cache helpers.
    config : LayoutConfig, optional
        Layout configuration. Uses ``LayoutConfig()`` when omitted.
    trace : Any, optional
        Optional tracing sink forwarded through the legacy engine path.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    if config is None:
        config = LayoutConfig()

    # Task 0.2: default flipped to dagua_native pipeline.
    # algorithm=None -> "dagua_native"; algorithm="_legacy" -> pre-decomposition
    # engine body (opt-in, unadvertised). Animation uses the legacy path
    # because layout-snapshot capture is not yet plumbed into ops pipelines
    #
    #
    # fallback: relax_steps > 0 also stays on the legacy path because
    # the relaxation pass (re-run with w_dag=0) is not yet a registered op.
    # will add a RelaxationStage op and drop this fallback.
    trace_fallback = trace is not None and config.algorithm is None
    relax_fallback = config.relax_steps > 0 and config.algorithm is None
    if trace_fallback:
        import warnings

        warnings.warn(
            "dagua.layout: trace-enabled runs still use the legacy engine "
            "Set "
            'algorithm="_legacy" to silence this warning, or pass trace=None '
            "to use the new default pipeline.",
            DeprecationWarning,
            stacklevel=2,
        )
    if relax_fallback:
        import warnings

        warnings.warn(
            "dagua.layout: relax_steps>0 still uses the legacy engine because "
            'Set algorithm="_legacy" '
            "to silence, or set relax_steps=0 to use the new default.",
            DeprecationWarning,
            stacklevel=2,
        )
    remapped_from_default = config.algorithm is None and trace is None and config.relax_steps <= 0
    # Algorithm=None now enters dagua_native, whose adapter
    # dispatches to tree/layered-DAG/force-directed/hybrid sub-pipelines after
    # classification. The route_flat_to_stress flag remains as a
    # compatibility shim: disabling it selects the preserved monolith.
    if remapped_from_default and not getattr(config, "route_flat_to_stress", True):
        config = copy.copy(config)
        config.force_pipeline = "legacy_monolith"
    if config.algorithm == "_legacy":
        config = copy.copy(config)
        config.algorithm = None
        # Falls through to the legacy engine body below.
    elif remapped_from_default:
        config = copy.copy(config)
        config.algorithm = "dagua_native"
        graph_direction = getattr(graph, "direction", None)
        if graph_direction in {"TB", "BT", "LR", "RL"}:
            config.direction = graph_direction

    if config.algorithm is not None:
        import inspect
        import warnings

        from dagua.layout.ops.pipelines import get_pipeline_function

        pipeline_fn = get_pipeline_function(config.algorithm)
        graph.compute_node_sizes()
        if getattr(config, "cluster_aware", True) and hasattr(graph, "clusters") and graph.clusters:
            warn_legacy_cluster_loss_config(
                cluster_aware=config.cluster_aware,
                w_cluster_containment=config.w_cluster_contain,
            )
            cluster_pos = _layout_cluster_aware_pipeline(graph, config)
            if cluster_pos is not None:
                return cluster_pos
            if config.algorithm == "dagua_native":
                warnings.warn(
                    "dagua.layout: recursive cluster placement is not yet available for "
                    "layered/DAG clusters with algorithm='dagua_native'; using flat native "
                    "placement with cluster losses.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    f"dagua.layout: cluster_aware=True is not yet supported for "
                    f"algorithm={config.algorithm!r}; falling back to legacy flat placement.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        kwargs: dict[str, object] = {
            "edge_index": graph.edge_index,
            "num_nodes": graph.num_nodes,
            "node_sizes": graph.node_sizes,
            "seed": config.seed,
        }
        if graph.edge_weights is not None:
            kwargs["edge_weights"] = graph.edge_weights
        kwargs.update(config.algorithm_params)
        kwargs["fidelity_dtype"] = config.fidelity_dtype
        # Forward steps if the pipeline accepts it
        sig = inspect.signature(pipeline_fn)
        if "steps" in sig.parameters:
            kwargs["steps"] = config.steps

        # Classify once here, where the real DaguaGraph is in scope, so an
        # explicit graph.is_semantically_directed declaration reaches
        # routing. Headless pipelines below this point only ever see
        # tensors, so this is the one place classify_graph can see
        # ``graph=``; the resulting structure flows down through the
        # existing graph_structure kwarg the pipelines already accept.
        # Gated on an explicit declaration so undeclared graphs keep the
        # exact prior code path (pipelines classify internally, including
        # after fidelity-mode edge preprocessing, and the heuristic
        # inference they run is already the fixed one).
        if (
            "graph_structure" in sig.parameters
            and "graph_structure" not in kwargs
            and getattr(graph, "is_semantically_directed", None) is not None
        ):
            kwargs["graph_structure"] = classify_graph(
                graph.edge_index,
                graph.num_nodes,
                graph=graph,
            )

        # Forward user-facing state into the pipeline regardless of how the
        # algorithm was selected. Previously this block was gated on
        # ``remapped_from_default`` (i.e. ``algorithm=None``) which meant
        # explicit ``algorithm="dagua_native"`` calls silently dropped the
        # user's config (including booleans like ``edge_equalize_polish``,
        # cluster metadata, and flex constraints). Forwarding always closes
        # that gap; downstream pipelines whose signatures don't accept a
        # particular kwarg are filtered by the ``accepted`` step below.
        if _config_or_graph_has_constraints(config, graph):
            config = _effective_constraint_config(config, graph)
        else:
            if config.flex is not None:
                config = _resolve_flex_ids(config, graph)
            if config.flex is None and getattr(graph, "flex", None) is not None:
                config = copy.copy(config)
                config.flex = _resolve_graph_flex(graph.flex, graph)
        kwargs["config"] = config
        if hasattr(graph, "clusters") and graph.clusters:
            kwargs["clusters"] = graph.clusters
        if hasattr(graph, "cluster_parents") and graph.cluster_parents:
            kwargs["cluster_parents"] = graph.cluster_parents
        if hasattr(graph, "cluster_labels") and graph.cluster_labels:
            kwargs["cluster_labels"] = graph.cluster_labels
        if hasattr(graph, "edge_labels") and graph.edge_labels:
            kwargs["edge_labels"] = graph.edge_labels
        if hasattr(graph, "last_label_positions") and graph.last_label_positions is not None:
            kwargs["label_positions"] = graph.last_label_positions
        if hasattr(graph, "node_styles") and graph.node_styles:
            kwargs["node_shapes"] = [
                getattr(graph.get_style_for_node(index), "shape", "rectangle")
                for index in range(graph.num_nodes)
            ]

        accepted = set(sig.parameters.keys())
        kwargs = {k: v for k, v in kwargs.items() if k in accepted}

        if remapped_from_default:
            graph._prepare_for_layout()
            try:
                pos = pipeline_fn(**kwargs)
                pos = _apply_constrained_polish(pos, graph, config)
                if _config_or_graph_has_constraints(config, graph):
                    pos = _project_final_hard_pins(
                        pos,
                        getattr(config, "flex", None),
                        getattr(config, "direction", "TB"),
                    )
                pos = pos.to(dtype=torch.float32)
                _update_graph_constraint_report(graph, pos, config)
                graph.cache_layout(pos)
                return pos
            finally:
                graph._restore_after_layout()
        result = pipeline_fn(**kwargs)
        if isinstance(result, tuple):
            pos, *rest = result
            if isinstance(pos, torch.Tensor):
                pos = _apply_constrained_polish(pos, graph, config)
                if _config_or_graph_has_constraints(config, graph):
                    pos = _project_final_hard_pins(
                        pos,
                        getattr(config, "flex", None),
                        getattr(config, "direction", "TB"),
                    )
                _update_graph_constraint_report(graph, pos, config)
                return (pos.to(dtype=torch.float32), *rest)
            return result
        pos = _apply_constrained_polish(result, graph, config)
        if _config_or_graph_has_constraints(config, graph):
            pos = _project_final_hard_pins(
                pos,
                getattr(config, "flex", None),
                getattr(config, "direction", "TB"),
            )
        _update_graph_constraint_report(graph, pos, config)
        return pos.to(dtype=torch.float32)

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
    # Populated in the direct (non-multilevel) branch below, where the real
    # DaguaGraph is in scope for classification; stays None for the
    # multilevel path (multilevel_layout classifies internally) so the
    # relax-pass _layout_inner call below falls back to its prior behavior.
    legacy_graph_structure: Optional[GraphStructure] = None
    try:
        # Tier 2: Multilevel coarsening for large graphs (N > 20K default)
        # Coarsening is faster than direct optimization at this scale.
        # Don't move data to GPU yet — multilevel manages device transfers lazily
        if n > config.multilevel_threshold:
            from dagua.layout.multilevel import multilevel_layout

            pos = multilevel_layout(graph, config, trace=trace)
        else:
            # Tier 0-2: Direct layout — move data to device
            # Resolve flex node IDs to integer indices before headless engine
            effective_config = _effective_constraint_config(config, graph)
            if effective_config.device != device:
                effective_config = copy.copy(effective_config)
                effective_config.device = device

            execution_mode = _resolve_execution_mode(effective_config, device, n)
            tensor_device = "cpu" if execution_mode == "subset_gpu" else device
            edge_index = graph.edge_index.to(tensor_device)
            node_sizes = graph.node_sizes.to(tensor_device)

            if config.verbose:
                num_edges = edge_index.shape[1] if edge_index.numel() > 0 else 0
                print(f"[dagua] Layout: {n:,} nodes, {num_edges:,} edges", flush=True)

            # Classify with graph= here, where the real DaguaGraph is still
            # in scope, so an explicit graph.is_semantically_directed
            # declaration reaches this legacy solve path too. Undeclared
            # graphs keep the exact prior path (None -> _layout_inner
            # classifies internally, identical inputs).
            if getattr(graph, "is_semantically_directed", None) is not None:
                legacy_graph_structure = classify_graph(edge_index, n, graph=graph)

            pos = _layout_inner(
                edge_index,
                n,
                node_sizes,
                effective_config,
                device=device,
                clusters=graph.clusters if hasattr(graph, "clusters") else None,
                cluster_parents=(
                    graph.cluster_parents if hasattr(graph, "cluster_parents") else None
                ),
                cluster_labels=(graph.cluster_labels if hasattr(graph, "cluster_labels") else None),
                edge_labels=graph.edge_labels if hasattr(graph, "edge_labels") else None,
                label_positions=(
                    graph.last_label_positions if hasattr(graph, "last_label_positions") else None
                ),
                node_shapes=(
                    [
                        getattr(graph.get_style_for_node(index), "shape", "rectangle")
                        for index in range(graph.num_nodes)
                    ]
                    if hasattr(graph, "node_styles")
                    else None
                ),
                progress_context=ProgressContext(),
                trace=trace,
                graph_structure=legacy_graph_structure,
            )

        # Force-directed relaxation: re-run with w_dag=0 to soften rigid
        # layer structure into a more organic layout.
        if config.relax_steps > 0:
            if config.verbose:
                print(
                    f"[dagua] Relaxation pass ({config.relax_steps} steps, w_dag=0)",
                    flush=True,
                )
            relax_config = copy.copy(config)
            relax_config.w_dag = 0.0
            relax_config.steps = config.relax_steps
            relax_config.lr = config.lr * 0.5
            relax_config.relax_steps = 0  # no recursive relaxation

            exec_mode = _resolve_execution_mode(relax_config, device, n)
            tensor_device_r = "cpu" if exec_mode == "subset_gpu" else device
            edge_index_r = graph.edge_index.to(tensor_device_r)
            node_sizes_r = graph.node_sizes.to(tensor_device_r)

            pos = _layout_inner(
                edge_index_r,
                n,
                node_sizes_r,
                relax_config,
                device=device,
                init_pos=pos,
                progress_context=ProgressContext(),
                trace=trace,
                graph_structure=legacy_graph_structure,
            )

        final_config = _effective_constraint_config(config, graph)
        # Apply direction transform
        graph_direction = getattr(graph, "direction", None)
        direction = (
            graph_direction
            if config.direction == "TB" and graph_direction in {"TB", "BT", "LR", "RL"}
            else config.direction
        )
        pos = _apply_direction(pos, direction)
        pos = _project_final_hard_pins(pos, getattr(final_config, "flex", None), direction)
        final_config.direction = direction
        _update_graph_constraint_report(graph, pos, final_config)
        graph.cache_layout(pos)
        return pos
    finally:
        graph._restore_after_layout()


def _create_optimizer(
    pos: torch.Tensor,
    config: LayoutConfig,
    optimizer_type: OptimizerType = "adam",
) -> torch.optim.Optimizer:
    """Create the position optimizer for the current layout solve.

    Parameters
    ----------
    pos : torch.Tensor
        Learnable position tensor shaped ``[N, 2]``.
    config : LayoutConfig
        Layout configuration that provides the base learning rate.
    optimizer_type : {"adam", "sgd_nesterov", "sgd"}, default="adam"
        Optimizer tier selected for the current solve.

    Returns
    -------
    torch.optim.Optimizer
        Optimizer configured for the requested tier.
    """
    lr = config.lr * HYBRID_OPTIMIZER_LR_MULTIPLIERS[optimizer_type]
    if optimizer_type == "sgd_nesterov":
        return torch.optim.SGD(
            [pos],
            lr=lr,
            momentum=0.9,
            nesterov=True,
        )
    if optimizer_type == "sgd":
        return torch.optim.SGD([pos], lr=lr)
    return torch.optim.Adam([pos], lr=lr)


def _should_use_tiled_gpu(
    requested_device: str,
    execution_device: str,
    num_nodes: int,
    num_edges: int,
) -> bool:
    """Return whether CPU execution should switch to tiled GPU loss evaluation.

    Parameters
    ----------
    requested_device : str
        Device stored on the current layout config. This is retained for API
        compatibility, but tiled GPU eligibility is determined by the actual
        execution mode because multilevel refinement normalizes configs to the
        per-level device.
    execution_device : str
        Device selected for the current engine call.
    num_nodes : int
        Number of nodes in the current layout problem.
    num_edges : int
        Number of edges in the current layout problem.

    Returns
    -------
    bool
        ``True`` when the current solve is running on CPU, the graph is large
        enough to justify tiling, CUDA is available, and a full resident GPU
        layout estimate does not fit the active VRAM budget.
    """
    return _tiled_gpu_decision(
        requested_device=requested_device,
        execution_device=execution_device,
        num_nodes=num_nodes,
        num_edges=num_edges,
    )[0]


def _tiled_gpu_decision(
    requested_device: str,
    execution_device: str,
    num_nodes: int,
    num_edges: int,
) -> tuple[bool, str]:
    """Return the tiled-GPU decision together with the skip reason.

    Parameters
    ----------
    requested_device : str
        Device stored on the current layout config. This is retained for API
        compatibility, but tiled GPU eligibility is determined by the actual
        execution mode because multilevel refinement normalizes configs to the
        per-level device.
    execution_device : str
        Device selected for the current engine call.
    num_nodes : int
        Number of nodes in the current layout problem.
    num_edges : int
        Number of edges in the current layout problem.

    Returns
    -------
    tuple[bool, str]
        ``(enabled, reason)`` where ``reason`` explains why tiled GPU is
        skipped when ``enabled`` is ``False``.
    """
    del requested_device
    if execution_device != "cpu":
        return False, f"execution_device={execution_device}"
    if num_nodes < TILED_GPU_MIN_NODES:
        return False, f"below threshold ({num_nodes:,} < {TILED_GPU_MIN_NODES:,})"
    if not torch.cuda.is_available():
        return False, "no CUDA"
    resident_batched_gpu_memory = _estimate_batched_cuda_resident_memory(num_nodes, num_edges)
    total_vram = _cuda_total_vram_bytes()
    if total_vram > 0 and resident_batched_gpu_memory < int(
        total_vram * STANDARD_CUDA_RESIDENT_HEADROOM
    ):
        return False, "fits in VRAM"
    full_gpu_memory = _estimate_gpu_memory(
        num_nodes,
        num_edges,
        per_loss_bw=False,
        edges_on_cpu=False,
        edge_batch=0,
    )
    if VRAMBudget().fits(full_gpu_memory):
        return False, "fits in VRAM"
    return True, "resident GPU layout exceeds VRAM budget"


def _resolve_execution_mode(
    config: LayoutConfig,
    device: str,
    num_nodes: int,
) -> ExecutionMode:
    """Resolve the actual execution mode for the current solve.

    Parameters
    ----------
    config : LayoutConfig
        Layout configuration for the current solve.
    device : str
        Requested execution device passed into the engine.
    num_nodes : int
        Number of nodes in the layout problem.

    Returns
    -------
    {"standard", "subset_gpu"}
        Effective execution mode after applying device availability and
        large-graph safety thresholds.
    """
    requested_mode = getattr(config, "execution_mode", "auto")
    if requested_mode == "standard":
        return "standard"
    if device != "cuda":
        return "standard"
    if requested_mode == "subset_gpu":
        return "subset_gpu"

    subset_threshold = max(int(getattr(config, "subset_gpu_threshold", 10_000_000)), 1)
    if num_nodes >= SUBSET_GPU_REQUIRED_THRESHOLD:
        return "subset_gpu"
    if num_nodes > subset_threshold:
        return "subset_gpu"
    return "standard"


def _cap_subset_gpu_sampled_context(
    sampled_ctx: SampledNodeContext,
    verbose: bool = False,
) -> SampledNodeContext:
    """Cap sampled active rows to keep one subset transfer within VRAM budget.

    Parameters
    ----------
    sampled_ctx : SampledNodeContext
        CPU-resident sampled-node context for the current step.
    verbose : bool, default=False
        Whether to log the auto-selected subset transfer budget.

    Returns
    -------
    SampledNodeContext
        Original or truncated sampled context sized for the budget.
    """
    budget_bytes = _auto_subset_gpu_sampled_budget(verbose=verbose)
    active_rows = sampled_ctx.active_idx.shape[0]
    if active_rows == 0 or sampled_ctx.sampled.numel() == 0:
        return sampled_ctx
    if budget_bytes <= SUBSET_GPU_SAMPLED_BASE_BYTES:
        empty_idx = sampled_ctx.active_idx[:0]
        empty_sampled = sampled_ctx.sampled[:0]
        return SampledNodeContext(active_idx=empty_idx, sampled=empty_sampled)

    samples_per_row = max(sampled_ctx.sampled.shape[1], 1)
    bytes_per_row = samples_per_row * SUBSET_GPU_BYTES_PER_SAMPLE + 64
    max_rows = max(int((budget_bytes - SUBSET_GPU_SAMPLED_BASE_BYTES) / bytes_per_row), 0)
    if max_rows <= 0:
        empty_idx = sampled_ctx.active_idx[:0]
        empty_sampled = sampled_ctx.sampled[:0]
        return SampledNodeContext(active_idx=empty_idx, sampled=empty_sampled)
    if max_rows >= active_rows:
        return sampled_ctx
    return SampledNodeContext(
        active_idx=sampled_ctx.active_idx[:max_rows].clone(),
        sampled=sampled_ctx.sampled[:max_rows].clone(),
    )


def _cuda_total_vram_bytes() -> int:
    """Return the active CUDA device capacity in bytes.

    Returns
    -------
    int
        Total VRAM for the current CUDA device, or ``0`` when CUDA is
        unavailable.
    """
    if not torch.cuda.is_available():
        return 0
    try:
        return int(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory)
    except Exception:
        return 0


def _estimate_batched_cuda_resident_memory(
    num_nodes: int,
    num_edges: int,
    edge_batch: int = DEFAULT_HYBRID_EDGE_BATCH,
) -> int:
    """Estimate CUDA residency when positions stay on GPU and edges are batched.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the current layout problem.
    num_edges : int
        Number of edges in the current layout problem.
    edge_batch : int, default=5_000_000
        Maximum number of edges transferred to GPU per optimization step.

    Returns
    -------
    int
        Estimated resident bytes for positions, gradients, one edge batch, and
        a conservative working-set margin.
    """
    position_bytes = num_nodes * 2 * 4
    gradient_bytes = position_bytes
    batched_edge_bytes = min(num_edges, edge_batch) * 2 * 8
    return int(
        position_bytes + gradient_bytes + batched_edge_bytes + STANDARD_CUDA_WORKING_SET_BYTES
    )


def _backward_standard_loss_terms(
    pos: torch.Tensor,
    loss_terms: list[tuple[float, LossFn]],
    node_sizes: torch.Tensor,
    layer_index: Optional[LayerIndex],
    progress_callback: Optional[Callable[[float], None]] = None,
) -> tuple[float, float]:
    """Evaluate standard loss terms and backpropagate them immediately.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor shaped ``[N, 2]`` on the active execution device.
    loss_terms : list[tuple[float, LossFn]]
        Weighted loss functions to evaluate in standard eager mode.
    node_sizes : torch.Tensor
        Node size tensor shaped ``[N, 2]`` aligned with ``pos``.
    layer_index : LayerIndex, optional
        Pre-computed layer structure for the current graph.
    progress_callback : Callable[[float], None], optional
        Callback invoked after each term with the cumulative unweighted loss.

    Returns
    -------
    tuple[float, float]
        Weighted and unweighted loss totals for the evaluated terms.
    """
    total_loss_val = 0.0
    unweighted_loss_val = 0.0

    for weight, loss_fn in loss_terms:
        term = weight * loss_fn(pos, node_sizes, layer_index)
        if term is not None and term.requires_grad:
            term.backward()
            val = float(term.item())
            total_loss_val += val
            unweighted_loss_val += val / weight if weight else 0.0
            if progress_callback is not None:
                progress_callback(unweighted_loss_val)

    return total_loss_val, unweighted_loss_val


def _is_cuda_oom_error(exc: BaseException) -> bool:
    """Return whether the exception represents a CUDA out-of-memory failure.

    Parameters
    ----------
    exc : BaseException
        Exception raised while running the tiled GPU step.

    Returns
    -------
    bool
        ``True`` when the exception is a CUDA OOM from modern or older
        PyTorch versions.
    """
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "CUDA out of memory" in str(exc)


def _layout_inner(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: torch.Tensor,
    config: LayoutConfig,
    device: str = "cpu",
    optimizer_type: OptimizerType = "adam",
    init_pos: Optional[torch.Tensor] = None,
    clusters: Optional[dict] = None,
    cluster_parents: Optional[dict] = None,
    cluster_labels: Optional[dict] = None,
    label_positions: Optional[Any] = None,
    edge_labels: Optional[Any] = None,
    node_shapes: Optional[list[str]] = None,
    edge_label_boxes: Optional[torch.Tensor] = None,
    cluster_label_boxes: Optional[dict] = None,
    layer_assignments: Optional[torch.Tensor] = None,
    progress_context: Optional[ProgressContext] = None,
    trace: Optional[object] = None,
    *,
    graph_structure: Optional[GraphStructure] = None,
    prebuilt_layer_index: Optional[LayerIndex] = None,
    skip_classification: bool = False,
    progress_file_path: Optional[Union[str, Path]] = None,
    progress_metadata: Optional[dict[str, object]] = None,
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
    optimizer_type : {"adam", "sgd_nesterov", "sgd"}, default="adam"
        Optimizer tier selected for this solve. Non-Adam tiers are used only
        for VRAM-constrained hybrid refinement.
    init_pos : torch.Tensor, optional
        Warm-start positions shaped ``[N, 2]``.
    clusters : dict, optional
        Cluster membership data for cluster losses.
    cluster_parents : dict, optional
        Parent-cluster mapping used by containment/separation losses.
    cluster_labels : dict, optional
        Cluster-label text keyed by cluster name.
    label_positions : Any, optional
        Edge-label anchors aligned to the edge tensor.
    edge_labels : Any, optional
        Edge-label text aligned to the edge tensor.
    node_shapes : list[str], optional
        Node-shape metadata aligned to graph nodes.
    edge_label_boxes : torch.Tensor, optional
        Edge-label geometry boxes with shape ``[E, 2]``.
    cluster_label_boxes : dict, optional
        Cluster-label geometry payload keyed by cluster name.
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
    progress_file_path : str | Path, optional
        Optional output path for atomic JSON progress snapshots.
    progress_metadata : dict[str, object], optional
        Extra metadata to merge into each progress snapshot. Multilevel
        refinement uses this to attach phase and level information.

    Returns
    -------
    torch.Tensor
        Detached position tensor shaped ``[N, 2]``.
    """
    if config.use_pipeline:
        return _layout_inner_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            node_sizes=node_sizes,
            config=config,
            device=device,
            optimizer_type=optimizer_type,
            init_pos=init_pos,
            clusters=clusters,
            cluster_parents=cluster_parents,
            layer_assignments=layer_assignments,
            graph_structure=graph_structure,
            prebuilt_layer_index=prebuilt_layer_index,
            skip_classification=skip_classification,
        )

    import time as _time

    n = num_nodes
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    execution_mode = _resolve_execution_mode(config, device, n)
    if device == "cuda" and not torch.cuda.is_available():
        if getattr(config, "verbose", False):
            print("[dagua]   Layout solve: CPU fallback (no CUDA)", flush=True)
        device = "cpu"
    resident_device = "cpu" if execution_mode == "subset_gpu" else device
    resident_device_type = torch.device(resident_device).type
    if node_sizes.ndim == 1:
        node_sizes = node_sizes.unsqueeze(1).expand(-1, 2).contiguous()
    elif node_sizes.ndim == 2 and node_sizes.shape[1] == 1:
        node_sizes = node_sizes.expand(-1, 2).contiguous()
    if node_sizes.device.type != resident_device_type:
        node_sizes = node_sizes.to(resident_device)
    if execution_mode == "subset_gpu" and edge_index.device.type != "cpu":
        edge_index = edge_index.cpu()

    verbose = getattr(config, "verbose", False)
    _indent = progress_context.indent if progress_context else "  "
    progress_path = (
        _resolve_progress_file_path(config, progress_file_path)
        if n > _PROGRESS_FILE_NODE_THRESHOLD
        else None
    )

    def _vlog(msg):
        if verbose:
            vram = ""
            if device == "cuda" and torch.cuda.is_available():
                used = torch.cuda.memory_allocated() / 1024**2
                free, total = torch.cuda.mem_get_info()
                vram = f" [VRAM {used:.0f}MB / {total / 1024**2:.0f}MB]"
            print(f"[dagua] {_indent}{msg}{vram}", flush=True)

    if n == 0:
        return torch.zeros(0, 2, device=resident_device)
    if n == 1:
        return torch.zeros(1, 2, device=resident_device)

    subset_threshold = max(int(getattr(config, "subset_gpu_threshold", 10_000_000)), 1)
    if execution_mode == "subset_gpu":
        _vlog(f"Execution mode: subset_gpu (N={n:,}, threshold={subset_threshold:,})")

    # Apply adaptive spacing based on graph size
    node_sep = config.node_sep
    rank_sep = config.rank_sep
    if config.adaptive_spacing:
        node_sep, rank_sep = _adaptive_spacing(n, node_sep, rank_sep)

    # Step 1: Initialization
    if init_pos is not None:
        pos = init_pos.to(resident_device)
    else:
        pos = init_positions(
            edge_index,
            n,
            node_sizes,
            node_sep=node_sep,
            rank_sep=rank_sep,
            device=resident_device,
            verbose=verbose,
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
        if layer_index.node_to_layer.device.type != resident_device_type:
            layer_index = build_layer_index(
                layer_index.node_to_layer,
                device=resident_device,
                verbose=verbose,
            )
        layer_assignments_raw = layer_index.node_to_layer
    elif layer_assignments is not None:
        layer_index = build_layer_index(
            layer_assignments,
            device=resident_device,
            verbose=verbose,
        )
        layer_assignments_raw = layer_assignments
    elif edge_index.numel() > 0:
        layering_device = "cuda" if torch.cuda.is_available() else "cpu"
        layer_assignments_raw = longest_path_layering(
            edge_index,
            n,
            device=layering_device,
            verbose=verbose,
        )
        layer_index = build_layer_index(
            layer_assignments_raw,
            device=resident_device,
            verbose=verbose,
        )

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
    edges_on_cpu = execution_mode == "subset_gpu" or (
        edge_index.device.type == "cpu" and device == "cuda"
    )
    if not edges_on_cpu and device == "cuda" and edge_index.numel() > 0:
        edge_bytes = edge_index.numel() * edge_index.element_size()
        if not VRAMBudget().fits(edge_bytes):
            edge_index = edge_index.cpu()
            edges_on_cpu = True

    tiled_compute = None
    tiled_enabled = False
    tiled_reason = f"execution_mode={execution_mode}"
    if execution_mode == "standard":
        tiled_enabled, tiled_reason = _tiled_gpu_decision(config.device, device, n, num_edges)
    if execution_mode == "standard" and tiled_enabled:
        from dagua.layout.tiled_compute import TiledGPUCompute

        tiled_compute = TiledGPUCompute(
            num_nodes=n,
            edge_index=edge_index if edge_index.device.type == "cpu" else edge_index.cpu(),
            node_sizes=node_sizes if node_sizes.device.type == "cpu" else node_sizes.cpu(),
            device="cuda",
            vram_budget=VRAMBudget().remaining(),
        )
        if layer_assignments_raw is not None:
            if isinstance(layer_assignments_raw, torch.Tensor):
                tiled_compute.layer_assignments = layer_assignments_raw.to(device="cpu")
            else:
                tiled_compute.layer_assignments = torch.tensor(
                    layer_assignments_raw,
                    dtype=torch.long,
                )
        print(f"[dagua] {_indent}TILED GPU active: {n:,} nodes in tiles", flush=True)
    elif verbose:
        _vlog(f"Tiled GPU: SKIPPED ({tiled_reason})")

    # Resolve memory optimization flags (VRAM-aware when on CUDA)
    if execution_mode == "subset_gpu":
        use_per_loss_bw, use_checkpointing, use_hybrid = False, False, False
    elif tiled_compute is None:
        use_per_loss_bw, use_checkpointing, use_hybrid = _resolve_memory_strategy(
            n,
            num_edges,
            device,
            config,
            edges_on_cpu=edges_on_cpu,
            edge_batch=edge_batch,
        )
    else:
        use_per_loss_bw, use_checkpointing, use_hybrid = False, False, False
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
    if tiled_compute is not None:
        flags.append("tiled_gpu")
    if execution_mode == "subset_gpu":
        flags.append("subset_gpu")
    if executor is not None:
        flags.append(f"workers={config.num_workers}")
    use_sampled_context = (
        tiled_compute is None
        and layer_index is not None
        and (config.w_repel > 0 or config.w_overlap > 0)
    )
    if use_sampled_context:
        flags.append("sampled_ctx")
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
                    verbose=verbose,
                )
        elif layer_assignments_raw is not None:
            cpu_layer_index = build_layer_index(
                layer_assignments_raw,
                device="cpu",
                verbose=verbose,
            )
    elif edges_on_cpu:
        cpu_edge_index = edge_index

    # Step 2: Set up optimization
    pos = pos.clone().detach().requires_grad_(True)
    del init_pos  # Free pre-clone positions (8 GB at 1B nodes)
    optimizer = _create_optimizer(pos, config, optimizer_type=optimizer_type)

    # Step 3: Build loss functions ONCE (reuse across steps via mutable refs)
    # batch_edges_ref is a mutable container so lambdas capture the reference
    batch_edges_ref: List[Optional[torch.Tensor]] = [None]
    edge_ctx_ref: List[Optional[EdgeBatchContext]] = [None]
    sampled_ctx_ref: List[Optional[SampledNodeContext]] = [None]
    current_step_ref: List[int] = [0]

    def _active_edges(p: torch.Tensor) -> torch.Tensor:
        if (
            tiled_compute is not None
            and tiled_compute.current_edge_index is not None
            and p.device == tiled_compute.current_edge_index.device
        ):
            return tiled_compute.current_edge_index
        if batch_edges_ref[0] is not None and p.device == batch_edges_ref[0].device:
            return batch_edges_ref[0]
        if cpu_edge_index is not None:
            return cpu_edge_index
        return edge_index

    def _active_edge_ctx(p: torch.Tensor) -> Optional[EdgeBatchContext]:
        if (
            tiled_compute is not None
            and tiled_compute.current_edge_ctx is not None
            and p.device == tiled_compute.current_edge_ctx.src.device
        ):
            return cast(EdgeBatchContext, tiled_compute.current_edge_ctx)
        if use_per_loss_bw:
            return None
        if edge_ctx_ref[0] is not None and p.device == edge_ctx_ref[0].src.device:
            return edge_ctx_ref[0]
        return None

    def _active_sampled_ctx(p: torch.Tensor) -> Optional[SampledNodeContext]:
        if sampled_ctx_ref[0] is not None and p.device == sampled_ctx_ref[0].active_idx.device:
            return sampled_ctx_ref[0]
        return None

    # Static loss functions — each returns
    # (base_weight_key, loss_fn, is_heavy, is_annealed, tiled_ok, cross_tile_ok)
    # base_weight_key is a string to look up the annealed weight at each step
    loss_fns: List[tuple[str, LossFn, bool, bool, bool, bool, str]] = []

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
                True,
                True,
                "edge",
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
                True,
                True,
                "edge",
            )
        )

    if config.w_repel > 0:
        repel_fn: LossFn

        def _repel_fn(
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

        repel_fn = _repel_fn
        if n > config.repel_amortize_threshold and config.repel_amortize_interval > 1:
            repel_fn = _make_amortized_loss(
                repel_fn,
                skip_every=config.repel_amortize_interval,
            )
        # Repulsion is always heavy (needs hybrid routing for large graphs).
        # Amortized skip steps return 0.0 which is safe for hybrid but NOT
        # for checkpointing (variable saved tensor count). Checkpointing is
        # handled in _compute_loss_term by checking the actual loss value.
        loss_fns.append(("w_repel", repel_fn, True, True, True, False, "sampled"))

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
                    debug_callback=_log_overlap_path if verbose else None,
                ),
                True,
                True,
                True,
                False,
                "sampled",
            )
        )

    if config.w_cluster > 0 and clusters:
        loss_fns.append(
            (
                "w_cluster",
                lambda p, ns, li: cluster_compactness_loss(p, clusters, device=p.device),
                False,
                False,
                False,
                False,
                "global",
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
                False,
                False,
                "global",
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
                False,
                False,
                "global",
            )
        )

    if config.w_crossing > 0 and num_edges >= 4:
        # alpha is annealed per-step, captured via mutable ref
        crossing_alpha_ref: List[float] = [3.0]
        # Crossing loss is O(E²) — amortize by computing every N steps.
        # Positions change slowly enough that every-step is wasteful.
        crossing_interval = _crossing_interval(n, config)
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

        loss_fns.append(("w_crossing", _crossing_fn, False, True, False, False, "global"))

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
                True,
                True,
                "edge",
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
                True,
                True,
                "edge",
            )
        )

    if config.w_spacing > 0 and layer_index is not None:
        # Spacing creates O(N) sort + gap tensors; mark heavy at scale
        # so hybrid mode routes it to CPU when VRAM is tight.
        spacing_is_heavy = n > 1_000_000
        loss_fns.append(
            (
                "w_spacing",
                lambda p, ns, li: spacing_consistency_loss(
                    p,
                    ns,
                    li,
                    target_gap=node_sep,
                ),
                spacing_is_heavy,
                False,
                False,
                False,
                "global",
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
            active_edges = _active_edges(p)
            active_edge_ctx = _active_edge_ctx(p)
            edge_is_sampled = active_edges.shape[1] < num_edges
            return fanout_distribution_loss(
                p,
                active_edges,
                edge_ctx=active_edge_ctx,
                step=current_step_ref[0],
                edge_is_sampled=edge_is_sampled,
            )

        loss_fns.append(("w_fanout", fanout_fn, False, False, True, True, "edge"))

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
                True,
                True,
                "edge",
            )
        )

    # --- Flex constraints: pins, alignment, flex spacing ---
    flex_data = _prepare_flex_data(
        config,
        n,
        resident_device,
        edge_index=edge_index,
        node_sizes=node_sizes,
    )

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
                False,
                False,
                "global",
            )
        )

    if flex_data["align_groups"]:
        _ag = flex_data["align_groups"]
        loss_fns.append(
            (
                "w_align_flex",
                lambda p, ns, li: alignment_loss(p, _ag),
                False,
                False,
                False,
                False,
                "global",
            )
        )

    if flex_data["order_pairs"]:
        _order_pairs = flex_data["order_pairs"]
        loss_fns.append(
            (
                "w_constraints",
                lambda p, ns, li: constraint_order_loss(p, _order_pairs),
                False,
                False,
                False,
                False,
                "global",
            )
        )

    if flex_data["separate_pairs"]:
        _separate_pairs = flex_data["separate_pairs"]
        loss_fns.append(
            (
                "w_constraints",
                lambda p, ns, li: constraint_separate_loss(p, _separate_pairs),
                False,
                False,
                False,
                False,
                "global",
            )
        )

    if flex_data["group_constraints"]:
        _group_constraints = flex_data["group_constraints"]
        loss_fns.append(
            (
                "w_constraints",
                lambda p, ns, li: constraint_group_loss(p, _group_constraints),
                False,
                False,
                False,
                False,
                "global",
            )
        )

    if flex_data["anchor_constraints"]:
        _anchor_constraints = flex_data["anchor_constraints"]
        loss_fns.append(
            (
                "w_constraints",
                lambda p, ns, li: constraint_anchor_loss(p, _anchor_constraints),
                False,
                False,
                False,
                False,
                "global",
            )
        )

    if flex_data["emphasize_paths"]:
        _emphasize_paths = flex_data["emphasize_paths"]
        loss_fns.append(
            (
                "w_constraints",
                lambda p, ns, li: constraint_emphasize_loss(p, _emphasize_paths),
                False,
                False,
                False,
                False,
                "global",
            )
        )

    if flex_data["focus_constraints"]:
        _focus_constraints = flex_data["focus_constraints"]
        loss_fns.append(
            (
                "w_constraints",
                lambda p, ns, li: constraint_focus_loss(p, _focus_constraints),
                False,
                False,
                False,
                False,
                "global",
            )
        )

    if flex_data["soft_contain"]:
        _soft_contain = flex_data["soft_contain"]
        loss_fns.append(
            (
                "w_constraints",
                lambda p, ns, li: constraint_contain_loss(p, _soft_contain),
                False,
                False,
                False,
                False,
                "global",
            )
        )

    for _custom_constraint in flex_data["custom_losses"]:
        _ctx = flex_data["constraint_context"]

        def _custom_loss_fn(
            p: torch.Tensor,
            ns: torch.Tensor,
            li: Optional[LayerIndex],
            c: Any = _custom_constraint,
            ctx: Any = _ctx,
        ) -> torch.Tensor:
            """Evaluate one user custom loss closure."""
            del ns, li
            return c.loss(p, ctx)

        loss_fns.append(
            (
                "w_constraints",
                _custom_loss_fn,
                False,
                False,
                False,
                False,
                getattr(_custom_constraint, "access", "global"),
            )
        )

    if flex_data["flex_node_sep"] is not None:
        _fsep = flex_data["flex_node_sep"]
        _fwt = flex_data["flex_node_sep_weight"]
        loss_fns.append(
            (
                "w_flex_spacing",
                lambda p, ns, li: flex_spacing_loss(p, ns, li, _fsep, _fwt),
                False,
                False,
                False,
                False,
                "global",
            )
        )

    # Pre-allocate edge batch buffer (avoids per-step tensor allocation)
    batch_device = "cpu" if execution_mode == "subset_gpu" else device
    batch_buf = (
        torch.empty(2, edge_batch, dtype=torch.long, device=batch_device)
        if tiled_compute is None and edge_batch > 0 and num_edges > edge_batch
        else None
    )
    perm_buf = (
        torch.empty(edge_batch, dtype=torch.long, device="cpu")
        if tiled_compute is None and edge_batch > 0
        else None
    )

    # Optimization loop with annealing
    prev_unweighted = float("inf")
    stall_count = 0
    _t_loop = _time.perf_counter()
    _log_interval = max(steps // 4, 1)  # log at 25%, 50%, 75%, 100%
    last_progress_write = _t_loop
    last_edge_batch_log = _t_loop
    next_edge_batch_fraction = _PROGRESS_BATCH_FRACTION
    last_edge_batch_number = 0
    sample_refresh_interval = (
        config.repel_amortize_interval
        if n > config.repel_amortize_threshold and config.repel_amortize_interval > 1
        else 1
    )
    sampled_ctx: Optional[SampledNodeContext] = None
    overlap_path_logged = False
    random_interval = (
        max(int(1.0 / config.edge_random_fraction), 1)
        if 0.0 < config.edge_random_fraction < 1.0
        else 0
    )
    progress_strategy = _progress_strategy(
        use_per_loss_bw=use_per_loss_bw,
        use_checkpointing=use_checkpointing,
        use_hybrid=use_hybrid,
        edges_on_cpu=edges_on_cpu,
        tiled_compute_active=tiled_compute is not None,
        execution_mode=execution_mode,
    )

    def _log_overlap_path(msg: str) -> None:
        """Emit the overlap-path selection once per layout solve."""
        nonlocal overlap_path_logged
        if overlap_path_logged:
            return
        _vlog(msg)
        overlap_path_logged = True

    def _maybe_write_progress_snapshot(
        step_index: int,
        loss_value: float,
        *,
        force: bool = False,
    ) -> None:
        """Write ``progress.json`` on the configured large-graph cadence.

        Parameters
        ----------
        step_index : int
            One-based optimizer step associated with the snapshot.
        loss_value : float
            Current unweighted loss estimate.
        force : bool, default=False
            Whether to bypass the normal write interval.

        Returns
        -------
        None
        """
        nonlocal last_progress_write
        if progress_path is None:
            return
        now = _time.perf_counter()
        if not force and now - last_progress_write < _PROGRESS_FILE_INTERVAL_SECONDS:
            return
        payload: dict[str, object] = {
            "step": step_index,
            "total_steps": steps,
            "step_pct": step_index / max(steps, 1),
            "loss": float(loss_value),
            "elapsed_seconds": now - _t_loop,
            "vram_mb": _current_vram_mb(),
            "rss_gb": round(_current_rss_gb(), 1),
            "strategy": progress_strategy,
            "num_nodes": n,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        if progress_metadata is not None:
            payload.update(progress_metadata)
        _write_progress_json(progress_path, payload)
        last_progress_write = now

    def _maybe_log_edge_batch_progress(
        batch_progress: Optional[tuple[int, int]],
        loss_value: float,
        *,
        force: bool = False,
    ) -> None:
        """Emit periodic edge-batch progress for verbose streamed-edge runs.

        Parameters
        ----------
        batch_progress : tuple[int, int] | None
            One-based current batch index and total number of batches.
        loss_value : float
            Current unweighted loss estimate.
        force : bool, default=False
            Whether to bypass the normal time threshold.

        Returns
        -------
        None
        """
        nonlocal last_edge_batch_log, next_edge_batch_fraction, last_edge_batch_number
        if not verbose or batch_progress is None:
            return
        batch_number, total_batches = batch_progress
        if total_batches <= 1:
            return
        if batch_number < last_edge_batch_number:
            next_edge_batch_fraction = _PROGRESS_BATCH_FRACTION
        batch_fraction = batch_number / max(total_batches, 1)
        now = _time.perf_counter()
        time_due = now - last_edge_batch_log >= _PROGRESS_LOG_INTERVAL_SECONDS
        fraction_due = batch_fraction >= next_edge_batch_fraction
        if not (force or time_due or fraction_due):
            last_edge_batch_number = batch_number
            return
        batch_pct = int(round(batch_fraction * 100))
        print(
            f"[dagua] {_indent}edge batch {batch_number}/{total_batches} "
            f"({batch_pct}%) loss={loss_value:.1f} "
            f"({now - _t_loop:.1f}s elapsed)",
            flush=True,
        )
        last_edge_batch_log = now
        while next_edge_batch_fraction <= batch_fraction:
            next_edge_batch_fraction += _PROGRESS_BATCH_FRACTION
        last_edge_batch_number = batch_number

    # Cross-step cache for the sampled access pattern — avoids redundant
    # torch.unique + searchsorted when sampled_ctx hasn't been refreshed.
    _cached_sampled_pattern: Optional[object] = None
    _cached_sampled_ctx_id: int = -1

    for step in range(steps):
        t = step / max(steps - 1, 1)  # 0 → 1
        current_step_ref[0] = step
        batch_progress = _compute_edge_batch_progress(step, num_edges, edge_batch, random_interval)

        if verbose and step > 0 and step % _log_interval == 0:
            _vlog(
                f"step {step}/{steps} ({100 * step // steps}%) "
                f"loss={prev_unweighted:.2f} "
                f"[{_time.perf_counter() - _t_loop:.1f}s]"
            )

        optimizer.zero_grad(set_to_none=True)
        edge_ctx_ref[0] = None

        # Sample edge batch for this step — reuse pre-allocated buffer
        if tiled_compute is not None:
            batch_edges_ref[0] = None
        elif batch_buf is not None:
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
            batch_edges_ref[0] = (
                edge_index if execution_mode == "subset_gpu" else edge_index.to(device)
            )
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
            assert sampled_layer_index is not None
            auto_log = verbose and step == 0
            sampled_device: Union[str, torch.device] = (
                "cpu" if use_hybrid or execution_mode == "subset_gpu" else pos.device
            )
            sampled_n_active, _, _ = _sampled_node_context_sizes(
                n,
                config.rvs_nn_k,
                verbose=auto_log,
            )
            default_n_active = sampled_n_active
            if (
                device == "cuda"
                and not use_hybrid
                and use_per_loss_bw
                and torch.cuda.is_available()
            ):
                free, _ = torch.cuda.mem_get_info()
                # Include PyTorch's cached-but-unallocated pool as available
                cached_free = max(
                    0,
                    torch.cuda.memory_reserved() - torch.cuda.memory_allocated(),
                )
                total_budget = int((free + cached_free) * CUDA_ACTIVE_SET_HEADROOM)
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
                    verbose=auto_log,
                )
            else:
                sampled_ctx = _build_sampled_node_context_for_active(
                    n,
                    sampled_layer_index,
                    sampled_device,
                    config.rvs_nn_k,
                    sampled_n_active,
                )
            if execution_mode == "subset_gpu" and device == "cuda" and torch.cuda.is_available():
                capped_ctx = _cap_subset_gpu_sampled_context(sampled_ctx, verbose=auto_log)
                if capped_ctx.active_idx.shape[0] < sampled_ctx.active_idx.shape[0]:
                    _vlog(
                        "capped subset-GPU sampled active set to "
                        f"{capped_ctx.active_idx.shape[0]:,} rows for VRAM headroom"
                    )
                sampled_ctx = capped_ctx
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
            "w_constraints": 1.0,
        }

        # Build loss_terms from static functions + current weights
        loss_terms: List[tuple] = []
        for (
            key,
            loss_fn,
            is_heavy,
            _is_annealed,
            tiled_ok,
            cross_tile_ok,
            access_kind,
        ) in loss_fns:
            w = weight_map[key]
            if w > 0:
                loss_terms.append((key, w, loss_fn, is_heavy, tiled_ok, cross_tile_ok, access_kind))

        # Execute loss terms with selected memory strategy
        total_loss_val = 0.0
        unweighted_loss_val = 0.0

        def _on_term_progress(current_unweighted_loss: float) -> None:
            """Handle periodic logging and JSON snapshots during one step.

            Parameters
            ----------
            current_unweighted_loss : float
                Current cumulative unweighted loss for the active step.

            Returns
            -------
            None
            """
            _maybe_log_edge_batch_progress(batch_progress, current_unweighted_loss)
            _maybe_write_progress_snapshot(step + 1, current_unweighted_loss)

        if execution_mode == "subset_gpu":
            from typing import Union as _Union

            from dagua.layout.subset_gpu import (
                EdgeAccessPattern,
                GlobalAccessPattern,
                SampledAccessPattern,
                SubsetGPUExecutor,
                SubsetGPULossTerm,
            )

            subset_executor = SubsetGPUExecutor(
                node_sizes=node_sizes,
                layer_index=layer_index,
                execution_device=device,
                batch_edges_ref=batch_edges_ref,
                edge_ctx_ref=edge_ctx_ref,
                sampled_ctx_ref=sampled_ctx_ref,
                verbose=verbose,
            )

            # Build ONE shared sampled access pattern for all sampled terms,
            # and cache it across steps when sampled_ctx hasn't been refreshed.
            shared_sampled_pattern: Optional[SampledAccessPattern] = None
            active_sampled_ctx = _active_sampled_ctx(pos)
            if active_sampled_ctx is not None:
                ctx_id = id(active_sampled_ctx)
                if ctx_id != _cached_sampled_ctx_id:
                    shared_sampled_pattern = SampledAccessPattern(active_sampled_ctx)
                    _cached_sampled_pattern = shared_sampled_pattern
                    _cached_sampled_ctx_id = ctx_id
                else:
                    shared_sampled_pattern = _cached_sampled_pattern  # type: ignore[assignment]

            subset_terms: List[SubsetGPULossTerm] = []
            for (
                key,
                weight,
                loss_fn,
                _is_heavy,
                _tiled_ok,
                _cross_tile_ok,
                access_kind,
            ) in loss_terms:
                access_pattern: _Union[
                    EdgeAccessPattern,
                    SampledAccessPattern,
                    GlobalAccessPattern,
                ]
                # Skip heavy global terms at very large scale — they run on
                # the full N-node tensor on CPU (e.g. spacing does argsort of
                # all N nodes) and dominate step time.  The edge + sampled
                # terms already cover layout quality at this scale.
                if access_kind == "global" and _is_heavy and n > 50_000_000:
                    continue
                if access_kind == "edge":
                    access_pattern = EdgeAccessPattern(_active_edges(pos))
                elif access_kind == "sampled":
                    access_pattern = (
                        shared_sampled_pattern
                        if shared_sampled_pattern is not None
                        else GlobalAccessPattern()
                    )
                else:
                    access_pattern = GlobalAccessPattern()
                subset_terms.append(
                    SubsetGPULossTerm(
                        name=key,
                        weight=weight,
                        loss_fn=loss_fn,
                        access_pattern=access_pattern,
                    )
                )

            pos.grad = subset_executor.compute_step(
                pos,
                subset_terms,
                progress_callback=_on_term_progress,
            )
            total_loss_val = subset_executor.last_total_loss
            unweighted_loss_val = subset_executor.last_unweighted_loss
        elif tiled_compute is not None:
            tiled_loss_fns: List[LossFn] = []
            tiled_loss_weights: List[float] = []
            tiled_cross_mask: List[bool] = []
            fallback_terms: List[tuple[float, LossFn, bool]] = []

            for (
                _key,
                weight,
                loss_fn,
                is_heavy,
                tiled_ok,
                cross_tile_ok,
                _access_kind,
            ) in loss_terms:
                if tiled_ok:
                    tiled_loss_fns.append(loss_fn)
                    tiled_loss_weights.append(weight)
                    tiled_cross_mask.append(cross_tile_ok)
                else:
                    fallback_terms.append((weight, loss_fn, is_heavy))

            if tiled_loss_fns:
                tiled_compute.cross_tile_loss_mask = tiled_cross_mask
                try:
                    total_loss_val += tiled_compute.compute_step(
                        pos,
                        tiled_loss_fns,
                        tiled_loss_weights,
                    )
                    unweighted_loss_val += tiled_compute.last_unweighted_loss
                except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                    if not _is_cuda_oom_error(exc):
                        raise
                    if hasattr(tiled_compute, "_reset_runtime_state"):
                        tiled_compute._reset_runtime_state()
                    print(
                        f"[dagua] {_indent}TILED GPU OOM - falling back to CPU for this step",
                        flush=True,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    fallback_total, fallback_unweighted = _backward_standard_loss_terms(
                        pos,
                        list(zip(tiled_loss_weights, tiled_loss_fns)),
                        node_sizes,
                        layer_index,
                        progress_callback=_on_term_progress,
                    )
                    total_loss_val += fallback_total
                    unweighted_loss_val += fallback_unweighted
                    tiled_compute.last_unweighted_loss = fallback_unweighted

            fallback_total, fallback_unweighted = _backward_standard_loss_terms(
                pos,
                [(weight, loss_fn) for weight, loss_fn, _is_heavy in fallback_terms],
                node_sizes,
                layer_index,
                progress_callback=_on_term_progress,
            )
            total_loss_val += fallback_total
            unweighted_loss_val += fallback_unweighted
        elif use_per_loss_bw:
            if executor is not None:
                # Parallel: heavy losses on CPU threads, light losses on GPU
                heavy_futures = []
                light_terms_list = []
                for (
                    _key,
                    weight,
                    loss_fn,
                    is_heavy,
                    _tiled_ok,
                    _cross_tile_ok,
                    _access_kind,
                ) in loss_terms:
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
                        _on_term_progress(unweighted_loss_val)

                # Gather heavy loss results
                for weight, future in heavy_futures:
                    term = future.result()
                    if term is not None and term.requires_grad:
                        term.backward()
                        val = term.item()
                        total_loss_val += val
                        unweighted_loss_val += val / weight if weight else 0.0
                        _on_term_progress(unweighted_loss_val)
            else:
                # Sequential per-loss backward
                for (
                    _key,
                    weight,
                    loss_fn,
                    is_heavy,
                    _tiled_ok,
                    _cross_tile_ok,
                    _access_kind,
                ) in loss_terms:
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
                        _on_term_progress(unweighted_loss_val)
        else:
            # Standard: accumulate all losses, single backward
            loss = torch.tensor(0.0, device=pos.device)
            for (
                _key,
                weight,
                loss_fn,
                is_heavy,
                _tiled_ok,
                _cross_tile_ok,
                _access_kind,
            ) in loss_terms:
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
                    _on_term_progress(unweighted_loss_val)
            loss.backward()
            total_loss_val = loss.item()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_([pos], max_norm=100.0)

        optimizer.step()

        if flex_data["hard_contain"]:
            project_hard_contains(pos, flex_data["hard_contain"])

        if flex_data["hard_align_groups"] or flex_data["hard_order_pairs"]:
            _project_hard_vpsc_constraints(
                pos,
                flex_data["hard_align_groups"],
                flex_data["hard_order_pairs"],
            )

        if flex_data["custom_projectors"]:
            _project_ctx = flex_data["constraint_context"]
            with torch.no_grad():
                for custom_constraint in flex_data["custom_projectors"]:
                    custom_constraint.project(pos, _project_ctx)

        # Hard-pin projection (weight=inf pins snapped to exact positions)
        if flex_data["has_hard_pins"]:
            project_hard_pins(
                pos,
                flex_data["pin_indices"],
                flex_data["pin_targets"],
                flex_data["hard_pin_mask"],
            )

        # Hard overlap projection (adaptive frequency + adaptive iterations)
        # Skip step 0 — positions are fresh from init and haven't moved yet.
        if step > 0 and (step % overlap_interval == 0 or step == steps - 1):
            proj_iters = 5 if n <= 500_000 else 3 if n <= 5_000_000 else 2
            if verbose and n > 1_000_000:
                _vlog(f"overlap projection (step {step}, N={n:,}, iters={proj_iters})")
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

        _maybe_log_edge_batch_progress(batch_progress, unweighted_loss_val)
        _maybe_write_progress_snapshot(step + 1, unweighted_loss_val)
        completed_steps = step + 1

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
    if completed_steps > 0:
        _maybe_write_progress_snapshot(completed_steps, prev_unweighted, force=True)

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
    if flex_data["hard_contain"]:
        project_hard_contains(pos, flex_data["hard_contain"])

    if flex_data["hard_align_groups"] or flex_data["hard_order_pairs"]:
        _project_hard_vpsc_constraints(
            pos,
            flex_data["hard_align_groups"],
            flex_data["hard_order_pairs"],
        )
    if flex_data["custom_projectors"]:
        _project_ctx = flex_data["constraint_context"]
        with torch.no_grad():
            for custom_constraint in flex_data["custom_projectors"]:
                custom_constraint.project(pos, _project_ctx)
    if flex_data["has_hard_pins"]:
        project_hard_pins(
            pos,
            flex_data["pin_indices"],
            flex_data["pin_targets"],
            flex_data["hard_pin_mask"],
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


def _layout_inner_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: torch.Tensor,
    config: LayoutConfig,
    device: str = "cpu",
    optimizer_type: OptimizerType = "adam",
    init_pos: Optional[torch.Tensor] = None,
    clusters: Optional[dict] = None,
    cluster_parents: Optional[dict] = None,
    cluster_labels: Optional[dict] = None,
    label_positions: Optional[Any] = None,
    edge_labels: Optional[Any] = None,
    node_shapes: Optional[list[str]] = None,
    edge_label_boxes: Optional[torch.Tensor] = None,
    cluster_label_boxes: Optional[dict] = None,
    layer_assignments: Optional[torch.Tensor] = None,
    *,
    graph_structure: Optional[GraphStructure] = None,
    prebuilt_layer_index: Optional[LayerIndex] = None,
    skip_classification: bool = False,
) -> torch.Tensor:
    """Delegate native direct-layout execution through the composable pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor shaped ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor
        Node size tensor shaped ``[N, 2]`` or broadcastable from ``[N]`` /
        ``[N, 1]``.
    config : LayoutConfig
        Layout configuration for the current solve.
    device : str, default="cpu"
        Target device for optimization.
    optimizer_type : {"adam", "sgd_nesterov", "sgd"}, default="adam"
        Optimizer tier requested by the caller.
    init_pos : torch.Tensor, optional
        Warm-start positions shaped ``[N, 2]``.
    clusters : dict, optional
        Cluster membership data for cluster losses.
    cluster_parents : dict, optional
        Parent-cluster mapping used by containment/separation losses.
    cluster_labels : dict, optional
        Cluster-label text keyed by cluster name.
    label_positions : Any, optional
        Edge-label anchors aligned to the edge tensor.
    edge_labels : Any, optional
        Edge-label text aligned to the edge tensor.
    node_shapes : list[str], optional
        Node-shape metadata aligned to graph nodes.
    edge_label_boxes : torch.Tensor, optional
        Edge-label geometry boxes with shape ``[E, 2]``.
    cluster_label_boxes : dict, optional
        Cluster-label geometry payload keyed by cluster name.
    layer_assignments : torch.Tensor, optional
        Raw layer assignments shaped ``[N]``.
    graph_structure : GraphStructure, optional
        Pre-classified structure metadata for the current graph.
    prebuilt_layer_index : LayerIndex, optional
        Pre-computed layer index for the current graph.
    skip_classification : bool, default=False
        When ``True``, skip graph-family classification before pipeline setup.

    Returns
    -------
    torch.Tensor
        Detached position tensor shaped ``[N, 2]``.
    """
    from dagua.layout.ops.pipelines.dagua_native import layout_dagua_native_pipeline

    return layout_dagua_native_pipeline(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        config=config,
        device=device,
        optimizer_type=optimizer_type,
        init_pos=init_pos,
        clusters=clusters,
        cluster_parents=cluster_parents,
        cluster_labels=cluster_labels,
        label_positions=label_positions,
        edge_labels=edge_labels,
        node_shapes=node_shapes,
        edge_label_boxes=edge_label_boxes,
        cluster_label_boxes=cluster_label_boxes,
        layer_assignments=layer_assignments,
        prebuilt_layer_index=prebuilt_layer_index,
        graph_structure=graph_structure,
        skip_classification=skip_classification,
    )


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
    edge_batch: int = DEFAULT_HYBRID_EDGE_BATCH,
    optimizer_type: OptimizerType = "adam",
) -> int:
    """Estimate GPU memory when heavy losses run on CPU (hybrid mode).

    Parameters
    ----------
    n : int
        Number of nodes in the layout problem.
    num_edges : int
        Number of edges in the layout problem.
    edge_batch : int, default=5_000_000
        Edge batch size resident on GPU for the current step. ``0`` means the
        full edge set is transferred.
    optimizer_type : {"adam", "sgd_nesterov", "sgd"}, default="adam"
        Optimizer tier whose state must fit alongside the GPU position tensor.

    Returns
    -------
    int
        Estimated peak GPU memory in bytes while hybrid mode keeps the heavy
        node losses on CPU.
    """
    actual_batch = num_edges if edge_batch <= 0 else min(edge_batch, num_edges)
    bytes_per_pos = n * 2 * 4
    optimizer_mem = bytes_per_pos * {"adam": 4, "sgd_nesterov": 3, "sgd": 2}[optimizer_type]
    edge_mem = actual_batch * 2 * 8
    edge_forward = actual_batch * 4 * 8
    autograd_overhead = edge_forward * AUTOGRAD_INTERMEDIATE_FACTOR * INTERMEDIATE_SAFETY_FACTOR
    grad_transfer = bytes_per_pos
    return int(
        optimizer_mem + edge_mem + autograd_overhead + grad_transfer + CUDA_CONTEXT_OVERHEAD_BYTES
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
        try:

            def _checkpointed_fn(p):
                return weight * loss_fn(p, node_sizes, layer_index)

            return torch_checkpoint(_checkpointed_fn, pos, use_reentrant=False)
        except Exception:
            # Amortized losses return 0.0 on skip steps, which changes
            # saved tensor count and breaks checkpointing. Fall through
            # to standard path.
            pass

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
    assert cpu_node_sizes is not None
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

    Parameters
    ----------
    num_edges : int
        Number of edges in the current layout problem.
    config : LayoutConfig
        Layout configuration that may provide an explicit batch override.

    Returns
    -------
    int
        Edge batch size for the current solve. Returns ``0`` for "use all
        edges" when the graph fits comfortably within the active VRAM budget.
    """
    if config.edge_batch_size > 0:
        return config.edge_batch_size

    if num_edges <= 10000:
        return 0

    if config.device == "cuda":
        auto_batch = _auto_edge_batch_size(verbose=getattr(config, "verbose", False))
        if num_edges <= auto_batch:
            return 0
        return min(auto_batch, num_edges)

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
            return torch.zeros(1, device=pos.device, dtype=pos.dtype, requires_grad=True)
        return loss_fn(pos, node_sizes, layer_index)

    return _amortized_loss


def _crossing_interval(num_nodes: int, config: LayoutConfig) -> int:
    """Return how often to evaluate crossing loss for the current solve.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the current layout problem.
    config : LayoutConfig
        Layout configuration for the current solve. Multilevel refinement may
        attach a private override for huge final-level runs.

    Returns
    -------
    int
        Crossing-loss evaluation interval in optimizer steps.
    """
    override = getattr(config, "_dagua_crossing_interval_override", None)
    if override is not None and int(override) > 0:
        return int(override)

    if num_nodes > 500:
        return 3
    if num_nodes > 50:
        return 5
    return 10


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
    elif num_nodes <= 50_000_000:
        return 40
    else:
        # At 50M+ nodes projection is O(N log N) per iteration on CPU
        # (argsort of the full position tensor). Run sparingly — the overlap
        # avoidance loss already provides soft resolution every step.
        return 200


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


def _constraint_axis_index(axis: str, direction: str) -> int:
    """Resolve a constraint axis to a tensor coordinate index.

    Parameters
    ----------
    axis : str
        Raw or semantic axis name.
    direction : str
        Layout direction.

    Returns
    -------
    int
        Coordinate index, where x is ``0`` and y is ``1``.
    """
    if axis == "x":
        return 0
    if axis == "y":
        return 1
    if axis == "flow":
        return 0 if direction in {"LR", "RL"} else 1
    if axis == "cross":
        return 1 if direction in {"LR", "RL"} else 0
    raise ValueError(f"Unknown constraint axis: {axis!r}")


def _selection_indices(selection: Any, num_nodes: int) -> List[int]:
    """Return integer indices from a lowered headless selection.

    Parameters
    ----------
    selection : Any
        Integer, sequence, or tensor of node indices.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    list[int]
        Valid node indices.
    """
    if isinstance(selection, torch.Tensor):
        values = [int(value) for value in selection.detach().cpu().flatten().tolist()]
    elif isinstance(selection, int):
        values = [int(selection)]
    elif isinstance(selection, (list, tuple)) and not (
        len(selection) == 2 and all(isinstance(item, float) for item in selection)
    ):
        values = []
        for item in selection:
            values.extend(_selection_indices(item, num_nodes))
    else:
        values = []
    return [value for value in values if 0 <= value < num_nodes]


def _selection_tensor(selection: Any, num_nodes: int, device: str) -> torch.Tensor:
    """Return a device tensor for a lowered headless selection.

    Parameters
    ----------
    selection : Any
        Integer, sequence, or tensor of node indices.
    num_nodes : int
        Number of graph nodes.
    device : str
        Target tensor device.

    Returns
    -------
    torch.Tensor
        Long tensor of valid node indices.
    """
    return torch.tensor(_selection_indices(selection, num_nodes), dtype=torch.long, device=device)


def _selection_or_canvas_tensor(selection: Any, num_nodes: int, device: str) -> Any:
    """Return node indices while preserving typed canvas selectors.

    Parameters
    ----------
    selection : Any
        Lowered node selection or canvas pseudo-element.
    num_nodes : int
        Number of graph nodes.
    device : str
        Target tensor device.

    Returns
    -------
    Any
        Long tensor for node selections, otherwise the canvas selector.
    """
    from dagua.constraints import CanvasSelector

    if isinstance(selection, CanvasSelector):
        return selection
    return _selection_tensor(selection, num_nodes, device)


def _project_anchor_target(target: Any, projection: Any) -> List[float]:
    """Return projected anchor coordinates.

    Parameters
    ----------
    target : Any
        Two-coordinate anchor target.
    projection : Any
        ``None``, ``"mercator"``, or a callable projection.

    Returns
    -------
    list[float]
        Projected ``[x, y]`` target.
    """
    x_value = float(target[0])
    y_value = float(target[1])
    if projection is None:
        return [x_value, y_value]
    if projection == "mercator":
        latitude = max(min(y_value, 89.5), -89.5)
        x_projected = x_value
        y_projected = math.log(math.tan(math.pi / 4.0 + math.radians(latitude) / 2.0))
        return [x_projected, y_projected]
    if callable(projection):
        projected = projection((x_value, y_value))
        return [float(projected[0]), float(projected[1])]
    raise ValueError(f"Unknown anchor projection: {projection!r}")


def _project_hard_vpsc_constraints(
    pos: torch.Tensor,
    hard_align_groups: List[tuple],
    hard_order_pairs: List[tuple],
) -> None:
    """Project hard one-dimensional constraints with VPSC.

    Parameters
    ----------
    pos : torch.Tensor
        Mutable position tensor with shape ``[N, 2]``.
    hard_align_groups : list[tuple]
        Alignment payloads as index tensor, weight, and axis.
    hard_order_pairs : list[tuple]
        Order payloads as left tensor, right tensor, axis, gap, and weight.

    Returns
    -------
    None
        Mutates ``pos`` in place under ``torch.no_grad``.
    """
    from dagua.layout.ops.webcola import solve_vpsc_1d

    if pos.numel() == 0:
        return
    with torch.no_grad():
        for axis in (0, 1):
            constraints = []
            for indices, _weight, group_axis in hard_align_groups:
                if int(group_axis) != axis or indices.numel() < 2:
                    continue
                base = int(indices[0].item())
                for other in indices[1:]:
                    constraints.append((base, int(other.item()), 0.0, True))
            for left, right, pair_axis, gap, _weight in hard_order_pairs:
                if int(pair_axis) != axis or left.numel() == 0 or right.numel() == 0:
                    continue
                for left_idx in left:
                    for right_idx in right:
                        constraints.append(
                            (int(left_idx.item()), int(right_idx.item()), float(gap), False)
                        )
            if not constraints:
                continue
            desired = pos[:, axis].detach().to(device="cpu", dtype=torch.float64).tolist()
            projected = solve_vpsc_1d(desired, constraints)
            pos[:, axis] = torch.tensor(projected, dtype=pos.dtype, device=pos.device)


def _prepare_flex_data(
    config: LayoutConfig,
    num_nodes: int,
    device: str,
    *,
    edge_index: Optional[torch.Tensor] = None,
    node_sizes: Optional[torch.Tensor] = None,
) -> dict:
    """Extract flex constraints from config into tensor form for the optimization loop.

    Returns a dict with pre-computed tensors for pins, alignment groups, and flex spacing.
    All node ID → index resolution happens here (not per-step).
    """
    result: dict[str, Any] = {
        "has_soft_pins": False,
        "has_hard_pins": False,
        "pin_indices": torch.zeros(0, dtype=torch.long, device=device),
        "pin_targets": torch.zeros(0, 2, dtype=torch.float32, device=device),
        "pin_weights": torch.zeros(0, 2, dtype=torch.float32, device=device),
        "soft_pin_mask": torch.zeros(0, 2, dtype=torch.bool, device=device),
        "hard_pin_mask": torch.zeros(0, 2, dtype=torch.bool, device=device),
        "align_groups": [],
        "hard_align_groups": [],
        "order_pairs": [],
        "hard_order_pairs": [],
        "separate_pairs": [],
        "group_constraints": [],
        "hard_groups": [],
        "anchor_constraints": [],
        "emphasize_paths": [],
        "focus_constraints": [],
        "custom_losses": [],
        "custom_projectors": [],
        "hard_contain": [],
        "soft_contain": [],
        "context_graph": None,
        "constraint_context": None,
        "flex_node_sep": None,
        "flex_node_sep_weight": 0.0,
    }

    flex = config.flex
    if flex is None:
        return result
    context_graph = getattr(flex, "_constraint_context_graph", None)
    result["context_graph"] = context_graph
    from dagua.constraints import ConstraintContext

    result["constraint_context"] = ConstraintContext(
        graph=context_graph,
        node_id_to_index=dict(getattr(context_graph, "_id_to_index", {})),
        edge_index=edge_index.detach().cpu() if edge_index is not None else None,
        node_sizes=node_sizes.detach().cpu() if node_sizes is not None else None,
        direction=config.direction,
    )

    constraints = list(getattr(flex, "constraints", None) or [])
    if constraints:
        from dagua.constraints import (
            Align,
            Anchor,
            Contain,
            Custom,
            Emphasize,
            Focus,
            Group,
            Order,
            Pin,
            Separate,
        )

        pin_indices: List[int] = []
        pin_targets: List[List[float]] = []
        pin_weights: List[List[float]] = []
        soft_mask: List[List[bool]] = []
        hard_mask: List[List[bool]] = []

        for constraint in constraints:
            if isinstance(constraint, Pin):
                from dagua.constraints import resolve_canvas_point

                at_x, at_y = resolve_canvas_point(constraint.at)
                for idx in _selection_indices(constraint.sel, num_nodes):
                    target_x_value = constraint.x if constraint.x is not None else at_x
                    target_y_value = constraint.y if constraint.y is not None else at_y
                    has_x = target_x_value is not None
                    has_y = target_y_value is not None
                    if not has_x and not has_y:
                        continue
                    hard = constraint.is_hard
                    weight = 0.0 if hard else float(constraint.weight)
                    pin_indices.append(idx)
                    target_x = 0.0 if target_x_value is None else float(target_x_value)
                    target_y = 0.0 if target_y_value is None else float(target_y_value)
                    pin_targets.append([target_x, target_y])
                    pin_weights.append([weight if has_x else 0.0, weight if has_y else 0.0])
                    soft_mask.append([has_x and not hard, has_y and not hard])
                    hard_mask.append([has_x and hard, has_y and hard])
            elif isinstance(constraint, Align):
                from dagua.constraints import resolve_canvas_point

                axis = _constraint_axis_index(constraint.axis, config.direction)
                idx_list = []
                for sel in tuple(constraint.sels):
                    idx_list.extend(_selection_indices(sel, num_nodes))
                at_x, at_y = resolve_canvas_point(constraint.at)
                at_value = at_x if axis == 0 else at_y
                if at_value is not None and idx_list:
                    hard = constraint.is_hard
                    weight = 0.0 if hard else float(constraint.weight)
                    for idx in idx_list:
                        target = [0.0, 0.0]
                        target[axis] = float(at_value)
                        mask = [False, False]
                        mask[axis] = True
                        pin_indices.append(idx)
                        pin_targets.append(target)
                        pin_weights.append([weight if mask[0] else 0.0, weight if mask[1] else 0.0])
                        soft_mask.append([mask[0] and not hard, mask[1] and not hard])
                        hard_mask.append([mask[0] and hard, mask[1] and hard])
                if len(idx_list) >= 2:
                    align_payload = (
                        torch.tensor(idx_list, dtype=torch.long, device=device),
                        float(constraint.weight),
                        axis,
                    )
                    bucket = "hard_align_groups" if constraint.is_hard else "align_groups"
                    result[bucket].append(align_payload)
            elif isinstance(constraint, Order):
                axis = _constraint_axis_index(constraint.axis, config.direction)
                gap = float(constraint.gap if constraint.gap is not None else 0.0)
                items = [
                    _selection_tensor(item, num_nodes, device) for item in tuple(constraint.items)
                ]
                for left, right in zip(items, items[1:]):
                    order_payload = (left, right, axis, gap, float(constraint.weight))
                    bucket = "hard_order_pairs" if constraint.is_hard else "order_pairs"
                    result[bucket].append(order_payload)
            elif isinstance(constraint, Separate):
                a = _selection_tensor(constraint.a, num_nodes, device)
                b = _selection_tensor(constraint.b, num_nodes, device)
                if a.numel() == 0 or b.numel() == 0:
                    continue
                gap = float(constraint.gap if constraint.gap is not None else 0.0)
                separate_axis: Optional[int] = (
                    _constraint_axis_index(constraint.axis, config.direction)
                    if constraint.axis
                    else None
                )
                if constraint.is_hard:
                    side = constraint.side
                    if side == "auto":
                        side = "left" if int(a[0].item()) <= int(b[0].item()) else "right"
                    left, right = (a, b) if side in {"left", "above", "before"} else (b, a)
                    result["hard_order_pairs"].append(
                        (left, right, separate_axis or 0, gap, float("inf"))
                    )
                else:
                    result["separate_pairs"].append(
                        (a, b, separate_axis, gap, float(constraint.weight))
                    )
            elif isinstance(constraint, Group):
                idx_list = []
                for sel in tuple(constraint.sels):
                    idx_list.extend(_selection_indices(sel, num_nodes))
                if len(idx_list) >= 2:
                    payload = (
                        torch.tensor(idx_list, dtype=torch.long, device=device),
                        float(constraint.padding or 0.0),
                        float(constraint.weight),
                    )
                    if constraint.is_hard:
                        result["hard_groups"].append(payload)
                    else:
                        result["group_constraints"].append(payload)
            elif isinstance(constraint, Anchor):
                indices: List[int] = []
                anchor_targets: List[List[float]] = []
                for node_id, anchor_target in constraint.mapping.items():
                    resolved = _selection_indices(node_id, num_nodes)
                    if resolved:
                        indices.append(resolved[0])
                        anchor_targets.append(
                            _project_anchor_target(anchor_target, constraint.projection)
                        )
                if indices:
                    target_tensor = torch.tensor(anchor_targets, dtype=torch.float32, device=device)
                    if constraint.fit == "fixed":
                        hard = constraint.is_hard
                        weight = 0.0 if hard else float(constraint.weight)
                        for idx, target_pair in zip(indices, anchor_targets):
                            pin_indices.append(idx)
                            pin_targets.append(target_pair)
                            pin_weights.append([weight, weight])
                            soft_mask.append([not hard, not hard])
                            hard_mask.append([hard, hard])
                    elif not constraint.is_hard:
                        result["anchor_constraints"].append(
                            (
                                torch.tensor(indices, dtype=torch.long, device=device),
                                target_tensor,
                                float(constraint.weight),
                                str(constraint.fit),
                            )
                        )
            elif isinstance(constraint, Emphasize):
                for item in tuple(constraint.path_or_edges):
                    indices = _selection_indices(item, num_nodes)
                    if len(indices) >= 2:
                        result["emphasize_paths"].append(
                            (
                                torch.tensor(indices, dtype=torch.long, device=device),
                                float(constraint.weight),
                            )
                        )
            elif isinstance(constraint, Focus):
                from dagua.constraints import resolve_canvas_point

                at_x, at_y = resolve_canvas_point(constraint.at)
                result["focus_constraints"].append(
                    (
                        _selection_tensor(constraint.sel, num_nodes, device),
                        torch.tensor(
                            [
                                0.0 if at_x is None else float(at_x),
                                0.0 if at_y is None else float(at_y),
                            ],
                            dtype=torch.float32,
                            device=device,
                        ),
                        float(constraint.zoom),
                        float(constraint.weight),
                    )
                )
            elif isinstance(constraint, Custom):
                if constraint.loss_fn is not None and not constraint.is_hard:
                    result["custom_losses"].append(constraint)
                if constraint.project_fn is not None:
                    result["custom_projectors"].append(constraint)
            elif isinstance(constraint, Contain):
                sel = _selection_tensor(constraint.sel, num_nodes, device)
                within = _selection_or_canvas_tensor(constraint.within, num_nodes, device)
                contain_payload = (sel, within, float(constraint.padding), float(constraint.weight))
                if constraint.is_hard:
                    result["hard_contain"].append(contain_payload)
                else:
                    result["soft_contain"].append(contain_payload)

        if pin_indices:
            result["pin_indices"] = torch.tensor(pin_indices, dtype=torch.long, device=device)
            result["pin_targets"] = torch.tensor(pin_targets, dtype=torch.float32, device=device)
            result["pin_weights"] = torch.tensor(pin_weights, dtype=torch.float32, device=device)
            result["soft_pin_mask"] = torch.tensor(soft_mask, dtype=torch.bool, device=device)
            result["hard_pin_mask"] = torch.tensor(hard_mask, dtype=torch.bool, device=device)
            result["has_soft_pins"] = any(any(row) for row in soft_mask)
            result["has_hard_pins"] = any(any(row) for row in hard_mask)

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


def _project_final_hard_pins(pos: torch.Tensor, flex: Any, direction: str = "TB") -> torch.Tensor:
    """Apply hard R9 constraints to final returned view-frame coordinates.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    flex : Any
        Optional ``LayoutFlex``-like object whose constraints have already
        been resolved to integer node indices.
    direction : str, default="TB"
        User-facing layout direction used for semantic axes.

    Returns
    -------
    torch.Tensor
        Position tensor with exact hard constraints re-asserted. The input is
        returned unchanged when no hard constraints are present.
    """
    if flex is None or not getattr(flex, "constraints", None):
        return pos
    final_config = LayoutConfig(flex=flex, direction=direction, constraint_policy="report")
    flex_data = _prepare_flex_data(final_config, int(pos.shape[0]), str(pos.device))
    if not _flex_data_has_constraints(flex_data):
        return pos
    projected = pos.clone()
    _project_hard_constraints_from_flex_data(projected, flex_data)
    return projected


def _resolve_flex_ids(config: LayoutConfig, graph) -> LayoutConfig:
    """Resolve flex node IDs in config to integer indices using graph._id_to_index.

    Returns config unchanged if no flex, or a shallow copy with resolved flex.
    """
    if config.flex is None:
        return config
    return _resolve_config_flex(config, graph)


def _resolve_config_flex(config: LayoutConfig, graph) -> LayoutConfig:
    """Create a config copy with flex node IDs resolved to indices."""
    import copy as _c

    resolved_flex = _resolve_graph_flex(config.flex, graph)
    if resolved_flex is config.flex:
        setattr(config.flex, "_constraint_context_graph", graph)
        return config
    new_config = _c.copy(config)
    new_config.flex = resolved_flex
    setattr(new_config.flex, "_constraint_context_graph", graph)
    return new_config


def _resolve_constraint_point_selection(selection: Any, graph: Any) -> Any:
    """Resolve node-like constraint selections while preserving canvas targets.

    Parameters
    ----------
    selection : Any
        User selection from a constraint object.
    graph : Any
        DaguaGraph-like object.

    Returns
    -------
    Any
        Node index list or typed canvas selector.
    """
    from dagua.constraints import (
        CanvasSelector,
        Selector,
        resolve_edge_selection,
        resolve_label_selection,
        resolve_node_selection,
    )

    if isinstance(selection, CanvasSelector):
        return selection
    if isinstance(selection, Selector) and selection.kind in {"label", "labels"}:
        return resolve_label_selection(selection, graph)
    if isinstance(selection, Selector) and selection.kind == "edges":
        edge_index = graph.edge_index
        endpoints: List[int] = []
        for edge_id in resolve_edge_selection(selection, graph):
            endpoints.append(int(edge_index[0, edge_id].item()))
            endpoints.append(int(edge_index[1, edge_id].item()))
        return sorted(set(endpoints))
    return resolve_node_selection(selection, graph)


def _resolve_graph_flex(flex, graph):
    """Resolve node IDs in a LayoutFlex to integer indices."""
    from dagua.constraints import (
        Align,
        Anchor,
        CanvasSelector,
        Contain,
        Emphasize,
        Focus,
        Group,
        Order,
        Pin,
        Selector,
        Separate,
        resolve_edge_selection,
        resolve_node_selection,
    )
    from dagua.flex import LayoutFlex

    changed = False
    resolved_constraints = []
    for constraint in list(getattr(flex, "constraints", None) or []):
        if isinstance(constraint, Pin):
            if isinstance(constraint.sel, Selector) and constraint.sel.kind in {"label", "labels"}:
                resolved = constraint
            else:
                resolved = dataclass_replace(
                    constraint,
                    sel=resolve_node_selection(constraint.sel, graph),
                )
        elif isinstance(constraint, Align):
            sels = tuple(resolve_node_selection(sel, graph) for sel in tuple(constraint.sels))
            resolved = Align(
                *sels,
                axis=constraint.axis,
                at=constraint.at,
                spacing=constraint.spacing,
                strength=constraint.strength,
                name=constraint.name,
                system=constraint.system,
                tags=constraint.tags,
            )
        elif isinstance(constraint, Order):
            items = tuple(resolve_node_selection(item, graph) for item in tuple(constraint.items))
            resolved = Order(
                *items,
                axis=constraint.axis,
                gap=constraint.gap,
                strength=constraint.strength,
                name=constraint.name,
                system=constraint.system,
                tags=constraint.tags,
            )
        elif isinstance(constraint, Group):
            sels = tuple(resolve_node_selection(sel, graph) for sel in tuple(constraint.sels))
            resolved = Group(
                *sels,
                padding=constraint.padding,
                strength=constraint.strength,
                name=constraint.name,
                system=constraint.system,
                tags=constraint.tags,
            )
        elif isinstance(constraint, Separate):
            resolved = dataclass_replace(
                constraint,
                a=_resolve_constraint_point_selection(constraint.a, graph),
                b=_resolve_constraint_point_selection(constraint.b, graph),
            )
        elif isinstance(constraint, Anchor):
            mapping = {
                resolve_node_selection(node_id, graph)[0]: target
                for node_id, target in constraint.mapping.items()
            }
            resolved = dataclass_replace(constraint, mapping=mapping)
        elif isinstance(constraint, Focus):
            resolved = dataclass_replace(
                constraint,
                sel=_resolve_constraint_point_selection(constraint.sel, graph),
            )
        elif isinstance(constraint, Contain):
            resolved = dataclass_replace(
                constraint,
                sel=_resolve_constraint_point_selection(constraint.sel, graph),
                within=(
                    constraint.within
                    if isinstance(constraint.within, CanvasSelector)
                    else _resolve_constraint_point_selection(constraint.within, graph)
                ),
            )
        elif isinstance(constraint, Emphasize):
            resolved_items = []
            for item in constraint.path_or_edges:
                if isinstance(item, Selector) and item.kind == "edges":
                    edge_index = graph.edge_index
                    for edge_id in resolve_edge_selection(item, graph):
                        resolved_items.append(
                            [
                                int(edge_index[0, edge_id].item()),
                                int(edge_index[1, edge_id].item()),
                            ]
                        )
                else:
                    resolved_items.append(resolve_node_selection(item, graph))
            resolved = Emphasize(
                *resolved_items,
                lane=constraint.lane,
                strength=constraint.strength,
                name=constraint.name,
                system=constraint.system,
                tags=constraint.tags,
            )
        else:
            resolved = constraint
        resolved_constraints.append(resolved)
        changed = changed or resolved is not constraint

    if not changed:
        setattr(flex, "_constraint_context_graph", graph)
        return flex

    resolved_flex = LayoutFlex(
        node_sep=flex.node_sep,
        rank_sep=flex.rank_sep,
        constraints=resolved_constraints,
    )
    setattr(resolved_flex, "_constraint_context_graph", graph)
    return resolved_flex
