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

import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import numpy as np
import numpy.typing as npt
import torch

from dagua.config import LayoutConfig
from dagua.layout.graph_classify import GraphFamily, classify_graph
from dagua.layout.layers import build_layer_index
from dagua.utils import _EDGE_CHUNK, VRAMBudget, _vram_fits, longest_path_layering

_STREAMING_THRESHOLD = 100_000_000
_DEDUP_BUCKET_TARGET = 150_000_000
_DEFAULT_NODE_SIZE = 20.0
_MIN_HUB_THRESHOLD = 8
_HUB_PERCENTILE = 90.0
_SKIP_ANCHOR_DEGREE = 2
_SKIP_ANCHOR_SPAN = 1.5

try:
    from numba import njit as _numba_njit
except ImportError:  # pragma: no cover - optional acceleration
    _numba_njit = None


def _match_scan_python(
    is_hub: npt.NDArray[np.bool_],
    pair_ok: npt.NDArray[np.bool_],
    triple_ok: npt.NDArray[np.bool_],
    n: int,
) -> npt.NDArray[np.int64]:
    """Assign coarse-group IDs with the original variable-stride scan.

    Parameters
    ----------
    is_hub : numpy.ndarray
        Boolean array of shape ``[N]`` marking scan-head nodes that must stay
        singleton.
    pair_ok : numpy.ndarray
        Boolean array of shape ``[N - 1]`` for pair compatibility between
        consecutive ordered nodes.
    triple_ok : numpy.ndarray
        Boolean array of shape ``[N - 2]`` for extending a matched pair to a
        triple using the second and third nodes.
    n : int
        Number of ordered nodes.

    Returns
    -------
    numpy.ndarray
        Integer array of shape ``[N]`` mapping each ordered node to a coarse
        group ID.
    """
    group_ids = np.empty(n, dtype=np.int64)
    group = 0
    i = 0
    while i < n:
        if is_hub[i]:
            group_ids[i] = group
            group += 1
            i += 1
        elif i + 1 < n and pair_ok[i]:
            if i + 2 < n and i < len(triple_ok) and triple_ok[i]:
                group_ids[i] = group
                group_ids[i + 1] = group
                group_ids[i + 2] = group
                group += 1
                i += 3
            else:
                group_ids[i] = group
                group_ids[i + 1] = group
                group += 1
                i += 2
        else:
            group_ids[i] = group
            group += 1
            i += 1
    return group_ids


def _build_match_scan() -> Callable[..., npt.NDArray[np.int64]]:
    """Return the fastest available coarse matching scan implementation.

    Returns
    -------
    Callable[..., numpy.ndarray]
        Python or numba-jitted group assignment function.
    """
    if _numba_njit is None:
        return _match_scan_python
    return _numba_njit(cache=True)(_match_scan_python)


def _compute_hub_thresholds(
    total_degree: npt.NDArray[np.int64],
    skip_degree: npt.NDArray[np.int64],
    global_order: npt.NDArray[np.int64],
    layer_offsets: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """Compute per-layer degree thresholds for singleton hub isolation.

    Parameters
    ----------
    total_degree : numpy.ndarray
        Total in+out degree per node with shape ``[N]``.
    skip_degree : numpy.ndarray
        Skip-edge degree per node with shape ``[N]``.
    global_order : numpy.ndarray
        Nodes grouped by layer with shape ``[N]``.
    layer_offsets : numpy.ndarray
        Layer start offsets with shape ``[num_layers + 1]``.

    Returns
    -------
    numpy.ndarray
        Integer threshold per layer with shape ``[num_layers]``.
    """
    num_layers = max(0, layer_offsets.shape[0] - 1)
    thresholds = np.full(num_layers, _MIN_HUB_THRESHOLD, dtype=np.int64)
    for layer_idx in range(num_layers):
        start = int(layer_offsets[layer_idx])
        end = int(layer_offsets[layer_idx + 1])
        if start >= end:
            continue
        layer_nodes = global_order[start:end]
        layer_degree = total_degree[layer_nodes] + 2 * skip_degree[layer_nodes]
        thresholds[layer_idx] = max(
            _MIN_HUB_THRESHOLD,
            int(np.ceil(np.percentile(layer_degree, _HUB_PERCENTILE))),
        )
    return thresholds


_match_scan = _build_match_scan()


def _offload_level_to_disk(level: CoarseLevel, level_idx: int, tmpdir: Path) -> Path:
    """Save a hierarchy level's large tensors to disk and free them from memory.

    Parameters
    ----------
    level : CoarseLevel
        The hierarchy level to offload.
    level_idx : int
        Level index used for the filename.
    tmpdir : Path
        Temporary directory that stores offloaded hierarchy data.

    Returns
    -------
    Path
        Path to the serialized level payload.
    """
    assert level.edge_index is not None
    assert level.node_sizes is not None

    path = tmpdir / f"level_{level_idx:02d}.pt"
    torch.save(
        {
            "edge_index": level.edge_index,
            "node_sizes": level.node_sizes,
        },
        path,
    )
    level.edge_index = None
    level.node_sizes = None
    level.offload_path = path
    level.offload_dir = tmpdir
    return path


def _reload_level_from_disk(level: CoarseLevel, path: Path) -> None:
    """Reload a hierarchy level's large tensors from disk.

    Parameters
    ----------
    level : CoarseLevel
        The hierarchy level to restore.
    path : Path
        Path to the serialized level payload.
    """
    data = torch.load(path, map_location="cpu")
    level.edge_index = data["edge_index"]
    level.node_sizes = data["node_sizes"]
    path.unlink(missing_ok=True)
    level.offload_path = None


def _cleanup_offloaded_hierarchy(levels: List[CoarseLevel]) -> None:
    """Remove temporary files and directories created for hierarchy offload.

    Parameters
    ----------
    levels : list[CoarseLevel]
        Hierarchy levels that may own offload metadata.
    """
    offload_dirs: set[Path] = set()
    for level in levels:
        if level.offload_path is not None:
            level.offload_path.unlink(missing_ok=True)
            offload_dirs.add(level.offload_path.parent)
            level.offload_path = None
        if level.offload_dir is not None:
            offload_dirs.add(level.offload_dir)
            level.offload_dir = None

    for offload_dir in offload_dirs:
        if offload_dir.name.startswith("dagua_hierarchy_"):
            shutil.rmtree(offload_dir, ignore_errors=True)


def _ensure_node_sizes_2d(
    node_sizes: Optional[torch.Tensor],
    num_nodes: Optional[int] = None,
) -> torch.Tensor:
    """Normalize node sizes to ``[N, 2]`` for coarsening/refinement code.

    The multilevel path touches node sizes in a few large-graph-specific places,
    so this helper centralizes the shape contract. For synthetic or partially
    populated graphs we allow a conservative fallback rather than crashing deep
    inside a scatter kernel.
    """
    if node_sizes is None:
        if num_nodes is None:
            raise ValueError("node_sizes cannot be None unless num_nodes is provided")
        return torch.full((num_nodes, 2), _DEFAULT_NODE_SIZE, dtype=torch.float32)

    if node_sizes.numel() == 0:
        if num_nodes is None:
            raise ValueError("empty node_sizes requires num_nodes for fallback sizing")
        return torch.full(
            (num_nodes, 2), _DEFAULT_NODE_SIZE, dtype=node_sizes.dtype, device=node_sizes.device
        )

    if node_sizes.ndim == 1:
        return torch.stack([node_sizes, node_sizes], dim=1)
    if node_sizes.ndim == 2 and node_sizes.shape[1] == 1:
        return node_sizes.expand(-1, 2)
    if node_sizes.ndim != 2 or node_sizes.shape[1] != 2:
        raise ValueError(
            f"node_sizes must have shape [N], [N, 1], or [N, 2]; got {tuple(node_sizes.shape)}"
        )
    if num_nodes is not None and node_sizes.shape[0] != num_nodes:
        if node_sizes.shape[0] == 1:
            return node_sizes.expand(num_nodes, 2)
        raise ValueError(
            f"node_sizes row count {node_sizes.shape[0]} does not match num_nodes={num_nodes}"
        )
    return node_sizes


@dataclass
class CoarseLevel:
    """One level of the coarsening hierarchy."""

    edge_index: Optional[torch.Tensor]  # [2, E_c] coarsened edges
    node_sizes: Optional[torch.Tensor]  # [N_c, 2]
    num_nodes: int
    fine_to_coarse: Optional[torch.Tensor]  # [N_fine] maps fine node → coarse node
    num_fine: int  # N at the finer level
    fine_layer_assignments: Optional[torch.Tensor] = (
        None  # [N_fine] layer assignments for fine level
    )
    coarse_layer_assignments: Optional[torch.Tensor] = (
        None  # [N_coarse] propagated layer assignments
    )
    offload_path: Optional[Path] = None
    offload_dir: Optional[Path] = None


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
    return VRAMBudget().fits(needed_bytes)


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
    node_sizes = _ensure_node_sizes_2d(node_sizes, N)
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

    # Coarse node counts per layer. Grouping is still local-in-layer; the
    # streaming path just avoids the global sort/materialization cost.
    coarse_per_layer = (layer_counts + 2) // 3
    coarse_offsets = torch.zeros(num_layers + 1, dtype=index_dtype, device=device)
    coarse_offsets[1:] = coarse_per_layer.cumsum(0)
    N_coarse = int(coarse_offsets[-1].item())

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
        coarse_base = int(coarse_offsets[layer_idx].item())
        fine_to_coarse[layer_nodes[local_order]] = (
            torch.arange(n_layer, dtype=index_dtype, device=device) // 3 + coarse_base
        )

    del layer_mask, min_neighbor

    # --- Phase B: Coarse node sizes ---
    # Every coarse node takes the max width/height of its assigned fine nodes.
    # This keeps cluster/refinement spacing conservative after coarsening.
    coarse_sizes = torch.zeros(N_coarse, 2, dtype=node_sizes.dtype, device=device)
    coarse_sizes[:, 0].scatter_reduce_(0, fine_to_coarse, node_sizes[:, 0], reduce="amax")
    coarse_sizes[:, 1].scatter_reduce_(0, fine_to_coarse, node_sizes[:, 1], reduce="amax")

    # --- Phase C: Chunked edge dedup ---
    # We hash coarse edges instead of building a giant dense adjacency. Bucketed
    # dedup keeps the peak memory lower on billion-edge runs.
    if E > 0:
        bucket_count = max(1, (E + _DEDUP_BUCKET_TARGET - 1) // _DEDUP_BUCKET_TARGET)
        bucket_uniques: List[torch.Tensor] = []
        for bucket_idx in range(bucket_count):
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
                    if bucket_count > 1:
                        bucket_mask = torch.remainder(chunk_hash, bucket_count) == bucket_idx
                        chunk_hash = chunk_hash[bucket_mask]
                    if chunk_hash.numel() > 0:
                        chunk_unique = chunk_hash.unique()
                        if running_unique is None:
                            running_unique = chunk_unique
                        else:
                            merged = torch.cat([running_unique, chunk_unique]).sort().values
                            running_unique = torch.unique_consecutive(merged)
                            del merged
                    del chunk_hash
                del chunk_src, chunk_tgt
            if running_unique is not None and running_unique.numel() > 0:
                bucket_uniques.append(running_unique)

        if bucket_uniques:
            all_unique = torch.cat(bucket_uniques) if len(bucket_uniques) > 1 else bucket_uniques[0]
            unique_src = all_unique // N_coarse
            unique_tgt = all_unique % N_coarse
            del bucket_uniques, all_unique
            coarse_edge_index = torch.stack([unique_src, unique_tgt])
            del unique_src, unique_tgt
        else:
            coarse_edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
    else:
        coarse_edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)

    return CoarseLevel(
        edge_index=coarse_edge_index,
        node_sizes=coarse_sizes,
        num_nodes=int(N_coarse),
        fine_to_coarse=fine_to_coarse,
        num_fine=N,
    )


def coarsen_once(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: torch.Tensor,
    layer_assignments: Union[List[int], torch.Tensor],
    device: str = "cpu",
    cluster_ids: Optional[torch.Tensor] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> CoarseLevel:
    """Single coarsening step via layer-aware heavy-edge matching.

    Within each layer, greedily pair adjacent nodes. Matched pairs merge
    into a single coarse node. Preserves DAG layer structure.

    Compatibility checks are vectorized across the full layer-sorted order,
    then a thin variable-stride scan assigns final coarse groups. For
    N > 100M, dispatches to the existing streaming path that processes edges
    in chunks.
    """
    N = num_nodes
    node_sizes = _ensure_node_sizes_2d(node_sizes, N)
    index_dtype = torch.int32 if N <= torch.iinfo(torch.int32).max else torch.long
    if isinstance(layer_assignments, list):
        layers = torch.tensor(layer_assignments, dtype=torch.long, device=device)
    else:
        layers = layer_assignments.to(device)
    cluster_ids = cluster_ids.to(device) if cluster_ids is not None else None
    num_layers = int(layers.max().item()) + 1 if N > 0 else 0
    t_features = 0.0
    t_match = 0.0
    t_dedup = 0.0

    # Sort nodes by layer for contiguous access
    layer_counts = torch.bincount(layers, minlength=num_layers)
    layer_offsets = torch.zeros(num_layers + 1, dtype=torch.long, device=device)
    layer_offsets[1:] = layer_counts.cumsum(0)

    # Dispatch to streaming path for very large graphs
    if N > _STREAMING_THRESHOLD:
        return _coarsen_once_streaming(
            edge_index,
            N,
            node_sizes,
            layers,
            num_layers,
            layer_counts,
            layer_offsets,
            device,
        )

    # Build adjacency features used for smarter within-layer ordering.
    # We keep the streaming 1B+ path unchanged and only spend extra work here,
    # where the goal is better coarsening quality for wide and skip-heavy graphs.
    scatter_device = device
    if device == "cpu" and VRAMBudget.available():
        E = edge_index.shape[1] if edge_index.numel() > 0 else 0
        scatter_bytes = N * 64 + E * 16
        if VRAMBudget().fits(scatter_bytes):
            scatter_device = "cuda"

    scatter_candidates = [scatter_device]
    if scatter_device == "cuda":
        scatter_candidates.append("cpu")

    for scatter_candidate in scatter_candidates:
        try:
            t_features_start = time.perf_counter()
            min_neighbor = torch.full((N,), N, dtype=torch.long, device=scatter_candidate)
            min_parent = torch.full((N,), N, dtype=torch.long, device=scatter_candidate)
            min_child = torch.full((N,), N, dtype=torch.long, device=scatter_candidate)
            in_degree = torch.zeros(N, dtype=torch.long, device=scatter_candidate)
            out_degree = torch.zeros(N, dtype=torch.long, device=scatter_candidate)
            skip_degree = torch.zeros(N, dtype=torch.long, device=scatter_candidate)
            mean_span = torch.zeros(N, dtype=torch.float32, device=scatter_candidate)
            if edge_index.numel() > 0:
                src = edge_index[0].to(device=scatter_candidate, dtype=torch.long)
                tgt = edge_index[1].to(device=scatter_candidate, dtype=torch.long)
                layers_sd = layers.to(scatter_candidate)
                min_neighbor.scatter_reduce_(0, src, tgt, reduce="amin")
                min_neighbor.scatter_reduce_(0, tgt, src, reduce="amin")
                min_parent.scatter_reduce_(0, tgt, src, reduce="amin")
                min_child.scatter_reduce_(0, src, tgt, reduce="amin")
                one_src = torch.ones_like(src)
                one_tgt = torch.ones_like(tgt)
                out_degree.scatter_add_(0, src, one_src)
                in_degree.scatter_add_(0, tgt, one_tgt)
                span = (layers_sd[tgt] - layers_sd[src]).abs().to(torch.float32)
                mean_span.scatter_add_(0, src, span)
                mean_span.scatter_add_(0, tgt, span)
                mean_span = mean_span / (in_degree + out_degree).clamp_min(1).to(torch.float32)
                skip_mask = span > 1.0
                skip_src = src[skip_mask]
                skip_tgt = tgt[skip_mask]
                if skip_src.numel() > 0:
                    skip_one = torch.ones(
                        skip_src.shape[0], dtype=torch.long, device=scatter_candidate
                    )
                    skip_degree.scatter_add_(0, skip_src, skip_one)
                    skip_degree.scatter_add_(0, skip_tgt, skip_one)

            if scatter_candidate != device:
                min_neighbor = min_neighbor.to(device)
                min_parent = min_parent.to(device)
                min_child = min_child.to(device)
                in_degree = in_degree.to(device)
                out_degree = out_degree.to(device)
                skip_degree = skip_degree.to(device)
                mean_span = mean_span.to(device)
            t_features = time.perf_counter() - t_features_start
            break
        except RuntimeError:
            if scatter_candidate != "cuda":
                raise
            torch.cuda.empty_cache()
    else:
        raise RuntimeError("Failed to compute coarsening adjacency features")

    # Smart within-layer ordering:
    # - similar parent/child signatures stay adjacent
    # - hubs can be kept isolated
    # - grouping adapts to local compatibility instead of blindly taking triples
    t_match_start = time.perf_counter()
    fine_to_coarse = torch.empty(N, dtype=index_dtype, device=device)
    total_degree = in_degree + out_degree

    min_neighbor_np = min_neighbor.cpu().numpy()
    min_parent_np = min_parent.cpu().numpy()
    min_child_np = min_child.cpu().numpy()
    total_degree_np = total_degree.cpu().numpy()
    skip_degree_np = skip_degree.cpu().numpy()
    mean_span_np = mean_span.cpu().numpy()
    layers_np = layers.cpu().numpy()
    cluster_ids_np = None if cluster_ids is None else cluster_ids.cpu().numpy()
    layer_offsets_np = layer_offsets.cpu().numpy()
    global_order = layers.argsort(stable=True)
    global_order_np = global_order.cpu().numpy()
    hub_threshold_per_layer = _compute_hub_thresholds(
        total_degree_np,
        skip_degree_np,
        global_order_np,
        layer_offsets_np,
    )

    all_ordered = np.empty(N, dtype=np.int64)
    max_cluster_key = np.iinfo(np.int64).max
    for layer_idx in range(num_layers):
        start = int(layer_offsets_np[layer_idx])
        end = int(layer_offsets_np[layer_idx + 1])
        if start >= end:
            continue

        layer_nodes_np = global_order_np[start:end]
        if cluster_ids_np is not None:
            cluster_key = np.where(
                cluster_ids_np[layer_nodes_np] >= 0,
                cluster_ids_np[layer_nodes_np],
                max_cluster_key,
            )
        else:
            cluster_key = np.full(end - start, max_cluster_key, dtype=np.int64)

        order = np.lexsort(
            (
                np.rint(mean_span_np[layer_nodes_np]).astype(np.int64),
                np.clip(total_degree_np[layer_nodes_np], 0, 31),
                min_child_np[layer_nodes_np],
                min_parent_np[layer_nodes_np],
                min_neighbor_np[layer_nodes_np],
                -np.clip(skip_degree_np[layer_nodes_np], 0, 31),
                cluster_key,
            )
        )
        all_ordered[start:end] = layer_nodes_np[order]

    all_layers = layers_np[all_ordered]
    all_min_neighbor = min_neighbor_np[all_ordered]
    all_min_parent = min_parent_np[all_ordered]
    all_min_child = min_child_np[all_ordered]
    all_total_degree = total_degree_np[all_ordered]
    all_mean_span = mean_span_np[all_ordered]
    all_skip_degree = skip_degree_np[all_ordered]
    hub_threshold_per_node = (
        hub_threshold_per_layer[all_layers] if N > 0 else np.empty(0, dtype=np.int64)
    )
    is_hub = (all_total_degree >= hub_threshold_per_node) | (
        (all_skip_degree >= _SKIP_ANCHOR_DEGREE) & (all_mean_span > _SKIP_ANCHOR_SPAN)
    )
    below_hub_threshold = all_total_degree < hub_threshold_per_node

    if N > 1:
        same_layer = all_layers[:-1] == all_layers[1:]
        shares_structure = (
            (all_min_neighbor[:-1] == all_min_neighbor[1:])
            | (all_min_parent[:-1] == all_min_parent[1:])
            | (all_min_child[:-1] == all_min_child[1:])
        )
        similar_shape = (
            (np.abs(all_total_degree[:-1] - all_total_degree[1:]) <= 1)
            & (np.abs(all_mean_span[:-1] - all_mean_span[1:]) <= 1.0)
            & (np.abs(all_skip_degree[:-1] - all_skip_degree[1:]) <= 1)
        )
        if cluster_ids_np is not None:
            all_cluster = cluster_ids_np[all_ordered]
            cluster_compatible = (
                (all_cluster[:-1] < 0)
                | (all_cluster[1:] < 0)
                | (all_cluster[:-1] == all_cluster[1:])
            )
            same_cluster = (all_cluster[:-1] >= 0) & (all_cluster[:-1] == all_cluster[1:])
        else:
            cluster_compatible = np.ones(N - 1, dtype=bool)
            same_cluster = np.zeros(N - 1, dtype=bool)
        pair_ok = (
            same_layer
            & ~is_hub[:-1]
            & below_hub_threshold[1:]
            & cluster_compatible
            & (same_cluster | shares_structure | similar_shape)
        )
    else:
        same_layer = np.empty(0, dtype=bool)
        pair_ok = np.empty(0, dtype=bool)

    if N > 2:
        triple_shares = (
            (all_min_parent[1:-1] == all_min_parent[2:])
            | (all_min_child[1:-1] == all_min_child[2:])
            | (all_min_neighbor[1:-1] == all_min_neighbor[2:])
        )
        triple_shape = (
            (np.abs(all_total_degree[1:-1] - all_total_degree[2:]) <= 1)
            & (np.abs(all_mean_span[1:-1] - all_mean_span[2:]) <= 1.0)
            & (np.abs(all_skip_degree[1:-1] - all_skip_degree[2:]) <= 1)
        )
        if cluster_ids_np is not None:
            triple_cluster = (
                (all_cluster[1:-1] < 0)
                | (all_cluster[2:] < 0)
                | (all_cluster[1:-1] == all_cluster[2:])
            )
        else:
            triple_cluster = np.ones(N - 2, dtype=bool)
        triple_ok = (
            same_layer[1:]
            & below_hub_threshold[2:]
            & triple_cluster
            & (triple_shares | triple_shape)
        )
    else:
        triple_ok = np.empty(0, dtype=bool)

    group_ids = _match_scan(is_hub, pair_ok, triple_ok, N)
    N_coarse = int(group_ids[-1]) + 1 if N > 0 else 0
    fine_to_coarse[torch.from_numpy(all_ordered).to(device=device, dtype=torch.long)] = (
        torch.from_numpy(group_ids).to(device=device, dtype=index_dtype)
    )
    t_match = time.perf_counter() - t_match_start

    # Build coarse node sizes (max of merged pair for each dimension)
    coarse_sizes = torch.zeros(N_coarse, 2, dtype=node_sizes.dtype, device=device)
    coarse_sizes.scatter_reduce_(
        0,
        fine_to_coarse.unsqueeze(1).expand(-1, 2),
        node_sizes,
        reduce="amax",
    )

    # Build coarse edges (remap and deduplicate)
    t_dedup_start = time.perf_counter()
    if edge_index.numel() > 0:
        edge_E = edge_index.shape[1]
        dedup_device = device
        if device == "cpu" and VRAMBudget.available():
            dedup_bytes = edge_E * 3 * 8
            if VRAMBudget().fits(dedup_bytes):
                dedup_device = "cuda"

        dedup_candidates = [dedup_device]
        if dedup_device == "cuda":
            dedup_candidates.append("cpu")

        for dedup_candidate in dedup_candidates:
            try:
                ei_dd = (
                    edge_index.to(dedup_candidate)
                    if dedup_candidate != edge_index.device.type
                    else edge_index
                )
                ftc_dd = (
                    fine_to_coarse.to(dedup_candidate)
                    if dedup_candidate != fine_to_coarse.device.type
                    else fine_to_coarse
                )

                coarse_src = ftc_dd[ei_dd[0]]
                coarse_tgt = ftc_dd[ei_dd[1]]

                not_self = coarse_src != coarse_tgt
                coarse_src = coarse_src[not_self]
                coarse_tgt = coarse_tgt[not_self]

                if coarse_src.numel() > 0:
                    coarse_src_long = coarse_src.to(dtype=torch.long)
                    coarse_tgt_long = coarse_tgt.to(dtype=torch.long)
                    edge_hash = coarse_src_long * N_coarse + coarse_tgt_long
                    unique_hash = edge_hash.unique(sorted=False)
                    unique_src = unique_hash // N_coarse
                    unique_tgt = unique_hash % N_coarse
                    coarse_edge_index = torch.stack([unique_src, unique_tgt]).to(device)
                else:
                    coarse_edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
                break
            except RuntimeError:
                if dedup_candidate != "cuda":
                    raise
                torch.cuda.empty_cache()
        else:
            raise RuntimeError("Failed to deduplicate coarse edges")
    else:
        coarse_edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
    t_dedup = time.perf_counter() - t_dedup_start

    if progress is not None:
        progress(f"features={t_features:.1f}s match={t_match:.1f}s dedup={t_dedup:.1f}s")

    return CoarseLevel(
        edge_index=coarse_edge_index,
        node_sizes=coarse_sizes,
        num_nodes=int(N_coarse),
        fine_to_coarse=fine_to_coarse,
        num_fine=N,
    )


def build_hierarchy(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: torch.Tensor,
    min_nodes: int = 2000,
    max_levels: int = 20,
    device: str = "cpu",
    progress: Optional[Callable[[str], None]] = None,
    cluster_ids: Optional[torch.Tensor] = None,
    initial_layer_assignments: Optional[torch.Tensor] = None,
    layer_assignments_callback: Optional[Callable[[torch.Tensor], None]] = None,
    level_callback: Optional[Callable[[List[CoarseLevel]], None]] = None,
    offload_to_disk: bool = True,
) -> List[CoarseLevel]:
    """Build coarsening hierarchy until num_nodes <= min_nodes.

    Returns list of CoarseLevels from finest to coarsest.

    For very large graphs, completed finer levels can have their large tensor
    payloads offloaded to temporary files during the build phase.
    """
    levels: List[CoarseLevel] = []
    current_ei = edge_index
    current_n = num_nodes
    current_sizes = _ensure_node_sizes_2d(node_sizes, current_n)
    current_cluster_ids = cluster_ids
    layer_dtype = torch.int32 if current_n <= torch.iinfo(torch.int32).max else torch.long
    offload_dir: Optional[Path] = None
    if offload_to_disk and current_n > 10_000_000:
        offload_dir = Path(tempfile.mkdtemp(prefix="dagua_hierarchy_"))

    # Compute layers once on the original graph — returns tensor for large N.
    # Allow a precomputed checkpoint for giant runs so retries can skip the
    # longest-path layering pass entirely.
    if initial_layer_assignments is not None:
        # Preserve the checkpoint dtype when resuming giant runs. Upcasting a
        # billion-element layering tensor to int64 duplicates several GB right
        # before coarsening and has no algorithmic benefit here.
        current_la = initial_layer_assignments.to(device="cpu")
        if progress is not None:
            max_layer = int(current_la.max().item()) if current_la.numel() > 0 else 0
            progress(f"Restored layering ({max_layer + 1:,} layers)")
    else:
        if progress is not None:
            progress(f"Layering full graph ({current_n:,} nodes)...")
        if current_ei.numel() > 0:
            raw_current_la = longest_path_layering(current_ei, current_n, device="cpu")
        else:
            raw_current_la = torch.zeros(current_n, dtype=layer_dtype)
        if isinstance(raw_current_la, list):
            current_la = torch.tensor(raw_current_la, dtype=layer_dtype)
        else:
            current_la = raw_current_la.to(device="cpu", dtype=layer_dtype)
        if layer_assignments_callback is not None:
            layer_assignments_callback(current_la.detach().cpu())
        if progress is not None:
            max_layer = int(current_la.max().item()) if current_la.numel() > 0 else 0
            progress(f"Layering done ({max_layer + 1:,} layers)")

    for level_idx in range(max_levels):
        if current_n <= min_nodes:
            break

        prev_edge_count = current_ei.shape[1] if current_ei.numel() > 0 else 0
        t_coarsen_start = time.perf_counter()
        level = coarsen_once(
            current_ei,
            current_n,
            current_sizes,
            layer_assignments=current_la,
            device=device,
            cluster_ids=current_cluster_ids,
            progress=progress,
        )
        t_coarsen = time.perf_counter() - t_coarsen_start
        level.fine_layer_assignments = current_la

        # Move to coarser level
        assert level.edge_index is not None
        assert level.node_sizes is not None
        current_ei = level.edge_index
        current_sizes = level.node_sizes
        current_n = level.num_nodes
        if progress is not None:
            progress(
                f"Coarsen level {level_idx + 1}: "
                f"{level.num_fine:,} nodes, {prev_edge_count:,} edges "
                f"[coarsen={t_coarsen:.1f}s]"
            )

        # Propagate layers: coarse node inherits layer from its fine nodes
        # (all fine nodes in a pair share the same layer by construction)
        assert level.fine_to_coarse is not None
        coarse_la = torch.zeros(current_n, dtype=current_la.dtype)
        coarse_la.scatter_reduce_(
            0,
            level.fine_to_coarse,
            current_la,
            reduce="amax",
        )
        level.coarse_layer_assignments = coarse_la
        level.offload_dir = offload_dir
        levels.append(level)
        if level_callback is not None:
            level_callback(levels)

        # Keep only one level's large tensors resident once the next coarse
        # level exists. Small graphs stay entirely in memory.
        if offload_dir is not None and len(levels) >= 2:
            prev_level = levels[-2]
            if prev_level.edge_index is not None:
                _offload_level_to_disk(prev_level, len(levels) - 2, offload_dir)

        # Safety: stop if coarsening didn't reduce nodes enough.
        # Edge count alone is not a reliable stopping signal for wide DAGs:
        # cross-layer edges can survive merging while node count still drops
        # enough for coarsening to remain beneficial.
        if current_n > level.num_fine * 0.7:
            if progress is not None:
                progress("Stopping hierarchy build: node reduction below threshold")
            break

        current_la = coarse_la
        if current_cluster_ids is not None:
            shifted = current_cluster_ids + 1
            coarse_min = torch.full((current_n,), shifted.max().item() + 1, dtype=shifted.dtype)
            coarse_max = torch.zeros(current_n, dtype=shifted.dtype)
            coarse_min.scatter_reduce_(0, level.fine_to_coarse, shifted, reduce="amin")
            coarse_max.scatter_reduce_(0, level.fine_to_coarse, shifted, reduce="amax")
            current_cluster_ids = torch.where(
                coarse_min == coarse_max,
                coarse_min - 1,
                torch.full_like(coarse_min, -1),
            )

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


def multilevel_layout(
    graph: Any, config: LayoutConfig, trace: Optional[Any] = None
) -> torch.Tensor:
    """Multilevel V-cycle layout for large graphs.

    1. Build coarsening hierarchy
    2. Layout coarsest graph (many steps)
    3. Prolong + refine at each level (few steps)
    """
    import time as _time

    from dagua.layout.engine import ProgressContext, _layout_inner

    verbose = config.verbose

    def _vlog(msg: str, indent: str = "") -> None:
        if verbose:
            vram = ""
            if device == "cuda" and torch.cuda.is_available():
                used = torch.cuda.memory_allocated() / 1024**2
                free, total = torch.cuda.mem_get_info()
                total_mb = total / 1024**2
                vram = f" [VRAM {used:.0f}MB / {total_mb:.0f}MB]"
            print(f"[dagua] {indent}{msg}{vram}", flush=True)

    def _reset_peak() -> None:
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
    precomputed_layers = getattr(graph, "_precomputed_layer_assignments", None)

    if config.seed is not None:
        torch.manual_seed(config.seed)

    # Build hierarchy on CPU — coarsening uses large temporary tensors
    # (edge_hash.unique() at 50M+ edges OOMs on small GPUs).
    # Keep all graph data on CPU; only move what's needed to GPU per-level.
    cpu_ei = graph.edge_index
    cpu_ns = graph.node_sizes
    min_nodes = config.multilevel_min_nodes
    precomputed_levels = getattr(graph, "_precomputed_hierarchy_levels", None)
    levels: List[CoarseLevel] = []
    _original_graph_path: Optional[Path] = None
    try:
        structure = classify_graph(cpu_ei, n, layer_assignments=precomputed_layers)
        if structure.family in {GraphFamily.TREE, GraphFamily.CHAIN}:
            _vlog(f"Phase 1/3: Tree fast path ({n:,} nodes, {structure.family.name.lower()})...")
            ei = cpu_ei.to(device)
            ns = cpu_ns.to(device)
            direct_layer_index = (
                build_layer_index(precomputed_layers, device=device)
                if precomputed_layers is not None
                else None
            )
            if trace is not None and hasattr(trace, "mark_phase"):
                trace.mark_phase("Direct Layout", f"{n:,} nodes")
            direct_pos = _layout_inner(
                ei,
                n,
                ns,
                config,
                device=device,
                layer_assignments=precomputed_layers,
                progress_context=ProgressContext(),
                trace=trace,
                graph_structure=structure,
                prebuilt_layer_index=direct_layer_index,
            )
            from dagua.layout.engine import _apply_direction

            direction = config.direction if config else graph.direction
            return _apply_direction(direct_pos, direction)

        if precomputed_levels is not None:
            levels = precomputed_levels
            _vlog(f"Phase 1/3: Restored hierarchy ({n:,} nodes)... {len(levels)} levels")
        else:
            _t_hier = _time.perf_counter()
            levels = build_hierarchy(
                cpu_ei,
                n,
                cpu_ns,
                min_nodes=min_nodes,
                device="cpu",
                progress=(lambda msg: _vlog(msg, indent="  ")) if verbose else None,
                cluster_ids=graph.cluster_ids,
                initial_layer_assignments=getattr(graph, "_precomputed_layer_assignments", None),
                layer_assignments_callback=getattr(graph, "_layer_assignments_callback", None),
                level_callback=getattr(graph, "_hierarchy_levels_callback", None),
                offload_to_disk=config.offload_to_disk,
            )
            hierarchy_complete_callback = getattr(
                graph, "_hierarchy_levels_complete_callback", None
            )
            if hierarchy_complete_callback is not None:
                hierarchy_complete_callback(levels)
            _vlog(
                f"Phase 1/3: Building hierarchy ({n:,} nodes)... "
                f"{len(levels)} levels ({_time.perf_counter() - _t_hier:.1f}s)"
            )

        if not levels:
            # Graph is already small enough — use direct layout
            ei = cpu_ei.to(device)
            ns = cpu_ns.to(device)
            direct_layer_index = (
                build_layer_index(precomputed_layers, device=device)
                if precomputed_layers is not None
                else None
            )
            if trace is not None and hasattr(trace, "mark_phase"):
                trace.mark_phase("Direct Layout", f"{n:,} nodes")
            direct_pos = _layout_inner(
                ei,
                n,
                ns,
                config,
                device=device,
                layer_assignments=precomputed_layers,
                progress_context=ProgressContext(),
                trace=trace,
                graph_structure=structure,
                prebuilt_layer_index=direct_layer_index,
            )
            from dagua.layout.engine import _apply_direction

            direction = config.direction if config else graph.direction
            return _apply_direction(direct_pos, direction)

        def _make_config(
            steps: int, lr: float = config.lr, seed: Optional[int] = config.seed
        ) -> LayoutConfig:
            """Build a level-specific layout config."""
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

        # Offload original graph to disk during Phase 2 — not needed until
        # refinement level i=0. Saves ~32GB at 1B scale.
        if config.offload_to_disk and n > 10_000_000:
            import tempfile as _tmpfile

            _orig_dir = _tmpfile.mkdtemp(prefix="dagua_orig_graph_")
            _original_graph_path = Path(_orig_dir) / "original_graph.pt"
            torch.save({"edge_index": cpu_ei, "node_sizes": cpu_ns}, _original_graph_path)
            del cpu_ei, cpu_ns
            import gc as _gc

            _gc.collect()
            try:
                import ctypes

                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except OSError:
                pass

        # Layout coarsest graph with many steps.
        # Pass edges on CPU — _layout_inner will stream batches to GPU.
        # Only node_sizes go to GPU (small: [N_coarse, 2]).
        coarsest = levels[-1]
        coarse_steps = config.multilevel_coarse_steps
        if coarsest.num_nodes > 50_000:
            coarse_steps = min(coarse_steps, 20)
        elif coarsest.num_nodes > 10_000:
            coarse_steps = min(coarse_steps, 30)
        _vlog(f"Phase 2/3: Coarsest level ({coarsest.num_nodes:,} nodes, {coarse_steps} steps)")
        _reset_peak()
        if trace is not None and hasattr(trace, "mark_phase"):
            trace.mark_phase("Hierarchy Build", f"{len(levels)} levels")
            trace.mark_phase("Coarsest Layout", f"{coarsest.num_nodes:,} supernodes")

        coarse_config = _make_config(
            steps=coarse_steps,
            lr=config.lr * 2,
        )

        assert coarsest.edge_index is not None
        assert coarsest.node_sizes is not None
        precomputed_coarsest_pos = getattr(graph, "_precomputed_coarsest_positions", None)
        if precomputed_coarsest_pos is not None:
            pos = precomputed_coarsest_pos.to(device)
            _vlog(f"Restored coarsest positions ({coarsest.num_nodes:,} nodes)", indent="  ")
        else:
            coarsest_layer_index = (
                build_layer_index(coarsest.coarse_layer_assignments, device="cpu")
                if coarsest.coarse_layer_assignments is not None
                else None
            )
            pos = _layout_inner(
                coarsest.edge_index,
                coarsest.num_nodes,
                coarsest.node_sizes.to(device),
                coarse_config,
                device=device,
                layer_assignments=coarsest.coarse_layer_assignments,
                progress_context=ProgressContext(),
                prebuilt_layer_index=coarsest_layer_index,
                skip_classification=True,
            )
            coarsest_pos_callback = getattr(graph, "_coarsest_positions_callback", None)
            if coarsest_pos_callback is not None:
                coarsest_pos_callback(pos.detach().cpu())

        num_refine_levels = len(levels)
        _vlog(f"Phase 3/3: Refining ({num_refine_levels} levels)")
        for i in range(len(levels) - 1, -1, -1):
            level = levels[i]
            assert level is not None

            level.edge_index = None
            level.node_sizes = None

            if level.num_fine > 100_000_000:
                import ctypes
                import gc as _gc

                _gc.collect()
                try:
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except OSError:
                    pass

            if i == 0:
                if _original_graph_path is not None and _original_graph_path.exists():
                    _orig_data = torch.load(_original_graph_path, map_location="cpu")
                    cpu_ei = _orig_data["edge_index"]
                    cpu_ns = _orig_data["node_sizes"]
                    _original_graph_path.unlink(missing_ok=True)
                    _original_graph_path.parent.rmdir()
                    _original_graph_path = None
                    del _orig_data
                fine_ei_cpu = cpu_ei
                fine_sizes_cpu = cpu_ns
                fine_n = n
            else:
                prev_level = levels[i - 1]
                assert prev_level is not None
                if prev_level.edge_index is None and prev_level.offload_path is not None:
                    _reload_level_from_disk(prev_level, prev_level.offload_path)
                assert prev_level.edge_index is not None
                assert prev_level.node_sizes is not None
                fine_ei_cpu = prev_level.edge_index
                fine_sizes_cpu = prev_level.node_sizes
                fine_n = prev_level.num_nodes

            n_fine_edges = fine_ei_cpu.shape[1] if fine_ei_cpu.numel() > 0 else 0
            base_refine = config.multilevel_refine_steps
            if i == 0:
                refine_steps = base_refine * 2
            elif i <= 2:
                refine_steps = base_refine
            else:
                refine_steps = max(base_refine // 2, 5)

            level_num = len(levels) - i
            _vlog(
                f"Level {level_num}/{num_refine_levels}: {fine_n:,} nodes ({refine_steps} steps)",
                indent="  ",
            )
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

            force_hybrid = False
            force_cpu = False
            if device == "cuda":
                from dagua.layout.engine import (
                    _estimate_gpu_memory,
                    _estimate_hybrid_gpu_memory,
                )

                next_level_mem = _estimate_gpu_memory(
                    fine_n,
                    n_fine_edges,
                    per_loss_bw=True,
                    edges_on_cpu=True,
                    edge_batch=config.edge_batch_size or 5_000_000,
                )
                if not _vram_fits(next_level_mem):
                    # Check if hybrid fits (pos + optimizer + edge losses only)
                    hybrid_mem = _estimate_hybrid_gpu_memory(fine_n, n_fine_edges)
                    if _vram_fits(hybrid_mem):
                        force_hybrid = True
                    else:
                        # Even hybrid doesn't fit — fall back to full CPU
                        force_cpu = True

            assert pos is not None
            assert level.fine_to_coarse is not None
            fine_to_coarse = level.fine_to_coarse
            use_gpu_prolong = _can_prolong_on_gpu(pos, fine_to_coarse, level.num_fine, device)

            pos_cpu: Optional[torch.Tensor]
            if use_gpu_prolong:
                fine_to_coarse_dev = fine_to_coarse.to(device)
                fine_pos = pos[fine_to_coarse_dev]
                fine_pos.add_(torch.randn(level.num_fine, 2, device=device).mul_(5.0))
                del fine_to_coarse_dev
                pos_cpu = None
            else:
                pos_cpu = pos.cpu() if pos.device.type != "cpu" else pos
                fine_pos = pos_cpu[fine_to_coarse]
                fine_pos.add_(torch.randn(level.num_fine, 2).mul_(5.0))

            del fine_to_coarse
            level.fine_to_coarse = None

            if pos_cpu is not None:
                del pos_cpu
            pos = None

            level_device = "cpu" if force_cpu else device
            fine_sizes = fine_sizes_cpu.to(level_device)
            pos = fine_pos if fine_pos.device.type == level_device else fine_pos.to(level_device)
            if force_cpu and torch.cuda.is_available():
                torch.cuda.empty_cache()
            del fine_pos

            refine_config = _make_config(steps=refine_steps, seed=None)
            if force_hybrid:
                import copy as _copy

                refine_config = _copy.copy(refine_config)
                refine_config.hybrid_device = "on"
            level_layer_index = (
                build_layer_index(level.fine_layer_assignments, device="cpu")
                if level.fine_layer_assignments is not None
                else None
            )
            level_trace = None
            if i == 0 and trace is not None:
                level_trace = trace
                if hasattr(trace, "mark_phase"):
                    trace.mark_phase("Final Refinement", f"{fine_n:,} nodes")

            pos = _layout_inner(
                fine_ei_cpu,
                fine_n,
                fine_sizes,
                refine_config,
                device=level_device,
                init_pos=pos,
                layer_assignments=level.fine_layer_assignments,
                progress_context=ProgressContext(indent="    "),
                trace=level_trace,
                prebuilt_layer_index=level_layer_index,
                skip_classification=True,
            )

            levels[i] = CoarseLevel(
                edge_index=None,
                node_sizes=None,
                num_nodes=level.num_nodes,
                fine_to_coarse=None,
                num_fine=level.num_fine,
                fine_layer_assignments=None,
                offload_dir=level.offload_dir,
            )

        _vlog(f"Done — {n:,} nodes in {_time.perf_counter() - _t0:.1f}s")

        from dagua.layout.engine import _apply_direction

        direction = config.direction if config else graph.direction
        assert pos is not None
        pos = _apply_direction(pos, direction)

        return pos
    finally:
        if _original_graph_path is not None:
            _original_graph_path.unlink(missing_ok=True)
            if _original_graph_path.parent.exists():
                import shutil

                shutil.rmtree(_original_graph_path.parent, ignore_errors=True)
        _cleanup_offloaded_hierarchy(levels)
