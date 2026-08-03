"""FIELD scale strategy for cyclic and general large graphs."""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Any, Optional, cast

import torch

from dagua.config import LayoutConfig
from dagua.layout.ops.pipelines.stress_sgd import layout_stress_sgd_pipeline
from dagua.layout.ops.state import LayoutProblem
from dagua.layout.scale.checkpoint import ScaleCheckpointManager, release_memory
from dagua.layout.scale.coarsen import (
    ScaleGraph,
    ScaleHierarchy,
    build_scale_hierarchy,
    prolong_positions,
)
from dagua.layout.scale.coarsest import anytime_native_coarsest
from dagua.layout.scale.pyramid import (
    build_grid_pyramid,
    build_grid_pyramid_streaming,
    choose_pyramid_device,
    choose_streaming_chunk_device,
    far_field_repulsion_force,
    pyramid_tensor_bytes,
)
from dagua.layout.scale.sketch import TopologySketch

_DEFAULT_COARSEST_TARGET = 2_000
_DEFAULT_TOPK = 16
_DEFAULT_EDGE_CHUNK = 2_000_000
_DEFAULT_BASE_CELL_MULTIPLIER = 1.2
_DEFAULT_MAX_GRID_AXIS = 1_024
_DEFAULT_PYRAMID_LEVELS = 10
_DEFAULT_STREAMING_NODE_THRESHOLD = 50_000_000
_DEFAULT_STREAMING_CHUNK_NODES = 2_000_000
_DEFAULT_STREAMING_EDGE_CHUNK = 5_000_000


@dataclass(frozen=True)
class FieldTelemetry:
    """Execution telemetry for one FIELD run.

    Parameters
    ----------
    coarsest_nodes : int
        Number of nodes in the final coarsest graph.
    levels : int
        Number of fine-to-coarse hierarchy transitions.
    coarsest_solver : str
        Coarsest solver mode actually requested.
    wall_s : float
        End-to-end FIELD wall time.
    pyramid_device : str
        Device kind used by the last pyramid build.
    checkpoint_armed : bool
        Whether checkpointing was armed for the run.
    resumed : bool
        Whether the run resumed from an existing checkpoint.
    """

    coarsest_nodes: int
    levels: int
    coarsest_solver: str
    wall_s: float
    pyramid_device: str
    checkpoint_armed: bool = False
    resumed: bool = False

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly telemetry.

        Returns
        -------
        dict[str, object]
            FIELD telemetry payload.
        """
        return {
            "coarsest_nodes": int(self.coarsest_nodes),
            "levels": int(self.levels),
            "coarsest_solver": self.coarsest_solver,
            "wall_s": float(self.wall_s),
            "pyramid_device": self.pyramid_device,
            "checkpoint_armed": bool(self.checkpoint_armed),
            "resumed": bool(self.resumed),
        }


class FieldScaleStrategy:
    """Family-agnostic FIELD strategy for above-gate general graphs."""

    def layout(
        self,
        graph: Any,
        config: LayoutConfig,
        sketch: TopologySketch,
        *,
        trace: Any = None,
    ) -> torch.Tensor:
        """Compute a FIELD layout for a prepared graph.

        Parameters
        ----------
        graph : Any
            Prepared graph-like object exposing ``edge_index`` and node sizes.
        config : LayoutConfig
            Layout configuration.
        sketch : TopologySketch
            Topology sketch used to route to FIELD.
        trace : Any, optional
            Optional trace sink, currently unused.

        Returns
        -------
        torch.Tensor
            Finite positions with shape ``[N, 2]``.
        """
        del trace
        started = time.perf_counter()
        params = config.algorithm_params
        seed = int(config.seed if config.seed is not None else 42)
        checkpoint = ScaleCheckpointManager.from_config(config, sketch, strategy="FIELD")
        final_record = checkpoint.record_for("final", 0) if checkpoint.armed else None
        if final_record is not None:
            pos = checkpoint.load_tensors("final", 0)["pos"].to(dtype=torch.float32)
            telemetry = FieldTelemetry(
                coarsest_nodes=int(cast(Any, final_record.telemetry.get("coarsest_nodes", 0))),
                levels=int(cast(Any, final_record.telemetry.get("levels", 0))),
                coarsest_solver=str(final_record.telemetry.get("coarsest_solver", "unknown")),
                wall_s=time.perf_counter() - started,
                pyramid_device=str(final_record.telemetry.get("pyramid_device", "unknown")),
                checkpoint_armed=True,
                resumed=True,
            )
            setattr(config, "_dagua_field_telemetry", telemetry.to_dict())
            return pos
        if _should_use_streaming_field(graph, config):
            return _layout_streaming_field(
                graph,
                config,
                sketch,
                checkpoint,
                started=started,
                seed=seed,
            )
        node_sizes = _resolved_graph_node_sizes(graph)
        hierarchy = build_scale_hierarchy(
            graph.edge_index,
            int(graph.num_nodes),
            node_sizes,
            getattr(graph, "edge_weights", None),
            target_nodes=int(params.get("field_coarsest_target", _DEFAULT_COARSEST_TARGET)),
            max_levels=int(params.get("field_max_levels", 24)),
            seed=seed,
            topk_per_node=int(params.get("field_topk_per_node", _DEFAULT_TOPK)),
            min_shrink_ratio=float(params.get("field_min_shrink_ratio", 0.50)),
        )
        if checkpoint.armed:
            _checkpoint_hierarchy(checkpoint, hierarchy)
        coarsest_solver = str(params.get("field_coarsest_solver", "auto")).lower()
        resumed = False
        latest_refine = _latest_refine_record(checkpoint) if checkpoint.armed else None
        if latest_refine is not None:
            pos = checkpoint.load_tensors("refine", latest_refine.level)["pos"].to(
                dtype=torch.float32
            )
            start_level = int(latest_refine.level) - 1
            resumed = True
        else:
            pos = _solve_coarsest(hierarchy, config, seed=seed, solver=coarsest_solver)
            start_level = len(hierarchy.levels) - 1
            if checkpoint.armed:
                _record_refine_checkpoint(
                    checkpoint,
                    level=len(hierarchy.levels),
                    pos=pos,
                    telemetry={
                        "phase": "coarsest",
                        "coarsest_nodes": hierarchy.coarsest_graph.num_nodes,
                    },
                )
        for level_index in range(start_level, -1, -1):
            transition = hierarchy.levels[level_index]
            jitter = _jitter_scale_for_level(config, transition.fine_num_nodes)
            pos = _expand_for_prolongation(
                pos,
                fine_num_nodes=transition.fine_num_nodes,
                coarse_num_nodes=transition.coarse_num_nodes,
                expand_gain=float(params.get("field_prolong_expand", 1.0)),
            )
            pos = prolong_positions(
                pos,
                transition.fine_to_coarse,
                seed=seed + level_index,
                jitter_scale=jitter,
            )
            level_graph = _fine_graph_for_transition(hierarchy, level_index)
            level_sizes = _fine_sizes_for_transition(
                hierarchy,
                level_index,
                int(graph.num_nodes),
                node_sizes,
            )
            level_masses = _fine_masses_for_transition(
                hierarchy,
                level_index,
                int(graph.num_nodes),
            )
            pos = _refine_level(
                pos,
                level_graph,
                level_sizes,
                level_masses,
                config,
                seed=seed + level_index,
                is_finest=level_index == 0,
            )
            if checkpoint.armed:
                _record_refine_checkpoint(
                    checkpoint,
                    level=level_index,
                    pos=pos,
                    telemetry={
                        "phase": "refine",
                        "fine_num_nodes": transition.fine_num_nodes,
                        "coarse_num_nodes": transition.coarse_num_nodes,
                    },
                )
                _maybe_exit_after_checkpoint(config, level=level_index)
        if not hierarchy.levels:
            pos = _refine_level(
                pos,
                hierarchy.finest_graph,
                node_sizes,
                torch.ones((int(graph.num_nodes),), dtype=torch.float32),
                config,
                seed=seed,
                is_finest=True,
            )
        if _should_pack_components(int(graph.num_nodes), params):
            pos = _pack_components(
                pos,
                hierarchy.finest_graph.edge_index,
                int(graph.num_nodes),
                base_sep=_target_separation(config, node_sizes),
            )
        pos = pos.detach().to(device="cpu", dtype=torch.float32)
        if not bool(torch.isfinite(pos).all().item()):
            raise RuntimeError("FIELD strategy produced non-finite positions.")
        telemetry = FieldTelemetry(
            coarsest_nodes=hierarchy.coarsest_graph.num_nodes,
            levels=len(hierarchy.levels),
            coarsest_solver=coarsest_solver,
            wall_s=time.perf_counter() - started,
            pyramid_device=str(getattr(config, "_dagua_field_last_pyramid_device", "cpu")),
            checkpoint_armed=checkpoint.armed,
            resumed=resumed,
        )
        if checkpoint.armed:
            _record_refine_checkpoint(
                checkpoint,
                level=0,
                pos=pos,
                phase="final",
                telemetry=telemetry.to_dict(),
            )
        setattr(config, "_dagua_field_telemetry", telemetry.to_dict())
        return pos


def layout_field(
    graph: Any,
    config: LayoutConfig,
    sketch: TopologySketch,
    *,
    trace: Any = None,
) -> torch.Tensor:
    """Compute a FIELD layout using the default strategy instance.

    Parameters
    ----------
    graph : Any
        Prepared graph-like object.
    config : LayoutConfig
        Layout configuration.
    sketch : TopologySketch
        Scale topology sketch.
    trace : Any, optional
        Optional trace sink.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    return FieldScaleStrategy().layout(graph, config, sketch, trace=trace)


def _checkpoint_hierarchy(checkpoint: ScaleCheckpointManager, hierarchy: ScaleHierarchy) -> None:
    """Persist hierarchy tensors as independent per-level records.

    Parameters
    ----------
    checkpoint : ScaleCheckpointManager
        Armed checkpoint manager.
    hierarchy : ScaleHierarchy
        Scale hierarchy to persist.

    Returns
    -------
    None
        Existing records are left in place.
    """
    if checkpoint.record_for("coarsest", 0) is None:
        before, after, _released = release_memory()
        checkpoint.record(
            phase="coarsest",
            level=0,
            tensors={
                "edge_index": hierarchy.coarsest_graph.edge_index,
                "edge_weight": hierarchy.coarsest_graph.edge_weight,
                "node_sizes": hierarchy.coarsest_node_sizes,
                "node_masses": hierarchy.coarsest_node_masses,
            },
            telemetry={
                "coarsest_nodes": hierarchy.coarsest_graph.num_nodes,
                "levels": len(hierarchy.levels),
            },
            rss_before_release=before,
            rss_after_release=after,
        )
    for level_index, transition in enumerate(hierarchy.levels):
        if checkpoint.record_for("hierarchy", level_index) is not None:
            continue
        before, after, _released = release_memory()
        checkpoint.record(
            phase="hierarchy",
            level=level_index,
            tensors={
                "fine_to_coarse": transition.fine_to_coarse,
                "edge_index": transition.edge_index,
                "edge_weight": transition.edge_weight,
                "node_sizes": transition.node_sizes,
                "node_masses": transition.node_masses,
            },
            telemetry={
                "fine_num_nodes": transition.fine_num_nodes,
                "coarse_num_nodes": transition.coarse_num_nodes,
            },
            rss_before_release=before,
            rss_after_release=after,
        )


def _latest_refine_record(checkpoint: ScaleCheckpointManager) -> Any:
    """Return the most recently written FIELD refine record.

    Parameters
    ----------
    checkpoint : ScaleCheckpointManager
        Checkpoint manager.

    Returns
    -------
    Any
        Latest checkpoint record or ``None``.
    """
    manifest = checkpoint.manifest
    if manifest is None:
        return None
    records = [
        record for record in manifest.records if record.phase == "refine" and record.completed
    ]
    if not records:
        return None
    return max(records, key=lambda record: record.written_at)


def _record_refine_checkpoint(
    checkpoint: ScaleCheckpointManager,
    *,
    level: int,
    pos: torch.Tensor,
    phase: str = "refine",
    telemetry: Optional[dict[str, object]] = None,
) -> None:
    """Persist one FIELD position checkpoint.

    Parameters
    ----------
    checkpoint : ScaleCheckpointManager
        Armed checkpoint manager.
    level : int
        Completed V-cycle level.
    pos : torch.Tensor
        Position tensor with shape ``[N_level, 2]``.
    phase : str, default="refine"
        Checkpoint phase.
    telemetry : dict[str, object], optional
        JSON-friendly telemetry.

    Returns
    -------
    None
        Position tensor is saved on CPU.
    """
    before, after, _released = release_memory()
    checkpoint.record(
        phase=phase,
        level=int(level),
        tensors={"pos": pos},
        telemetry=dict(telemetry or {}),
        rss_before_release=before,
        rss_after_release=after,
    )


def _maybe_exit_after_checkpoint(config: LayoutConfig, *, level: int) -> None:
    """Terminate for checkpoint kill/resume tests when explicitly requested.

    Parameters
    ----------
    config : LayoutConfig
        Layout config carrying ``scale_checkpoint_exit_after_level``.
    level : int
        Just-completed level.

    Returns
    -------
    None
        Calls ``os._exit(137)`` only for the private explicit test knob.
    """
    value = config.algorithm_params.get("scale_checkpoint_exit_after_level", None)
    if value is not None and int(value) == int(level):
        import os

        os._exit(137)


def _layout_streaming_field(
    graph: Any,
    config: LayoutConfig,
    sketch: TopologySketch,
    checkpoint: ScaleCheckpointManager,
    *,
    started: float,
    seed: int,
) -> torch.Tensor:
    """Compute a FIELD rung without Python adjacency or full-GPU residency.

    Parameters
    ----------
    graph : Any
        Tensor graph exposing ``edge_index`` and ``num_nodes``.
    config : LayoutConfig
        Layout configuration.
    sketch : TopologySketch
        Topology sketch used for manifest fingerprints.
    checkpoint : ScaleCheckpointManager
        Checkpoint manager for this run.
    started : float
        ``time.perf_counter`` value captured by the caller.
    seed : int
        Deterministic seed.

    Returns
    -------
    torch.Tensor
        Finite positions with shape ``[N, 2]``.
    """
    del sketch
    params = config.algorithm_params
    n = int(graph.num_nodes)
    node_sizes = _resolved_graph_node_sizes_for_streaming(graph, config)
    hierarchy = build_scale_hierarchy(
        graph.edge_index,
        n,
        node_sizes,
        getattr(graph, "edge_weights", None),
        target_nodes=int(params.get("field_coarsest_target", _DEFAULT_COARSEST_TARGET)),
        max_levels=int(params.get("field_max_levels", 24)),
        seed=seed,
        topk_per_node=int(params.get("field_topk_per_node", _DEFAULT_TOPK)),
        min_shrink_ratio=float(params.get("field_min_shrink_ratio", 0.50)),
    )
    if checkpoint.armed:
        _checkpoint_hierarchy(checkpoint, hierarchy)

    if hierarchy.levels:
        coarsest_solver = str(params.get("field_coarsest_solver", "auto")).lower()
        pos = _solve_coarsest(hierarchy, config, seed=seed, solver=coarsest_solver)
        if checkpoint.armed:
            _record_refine_checkpoint(
                checkpoint,
                level=len(hierarchy.levels),
                pos=pos,
                telemetry={
                    "phase": "streaming_hierarchy_coarsest",
                    "coarsest_nodes": hierarchy.coarsest_graph.num_nodes,
                },
            )
        for level_index in range(len(hierarchy.levels) - 1, -1, -1):
            transition = hierarchy.levels[level_index]
            jitter = _jitter_scale_for_level(config, transition.fine_num_nodes)
            pos = _expand_for_prolongation(
                pos,
                fine_num_nodes=transition.fine_num_nodes,
                coarse_num_nodes=transition.coarse_num_nodes,
                expand_gain=float(params.get("field_prolong_expand", 1.0)),
            )
            pos = prolong_positions(
                pos,
                transition.fine_to_coarse,
                seed=seed + level_index,
                jitter_scale=jitter,
            )
            level_graph = _fine_graph_for_transition(hierarchy, level_index)
            level_sizes = _fine_sizes_for_transition(hierarchy, level_index, n, node_sizes)
            level_masses = _fine_masses_for_transition(hierarchy, level_index, n)
            pos = _refine_streaming_level(
                pos,
                level_graph.edge_index,
                level_graph.edge_weight,
                level_sizes,
                level_masses,
                config,
                seed=seed + level_index,
            )
            if checkpoint.armed:
                _record_refine_checkpoint(
                    checkpoint,
                    level=level_index,
                    pos=pos,
                    phase="streaming_refine",
                    telemetry={
                        "phase": "streaming_hierarchy_refine",
                        "fine_num_nodes": transition.fine_num_nodes,
                        "coarse_num_nodes": transition.coarse_num_nodes,
                    },
                )
        resumed = False
        coarsest_solver = f"streaming_hierarchy_{coarsest_solver}"
    else:
        base_sep = _target_separation(config, node_sizes)
        init_record = checkpoint.record_for("streaming", 0) if checkpoint.armed else None
        if init_record is not None:
            pos = checkpoint.load_tensors("streaming", 0)["pos"].to(dtype=torch.float32)
            resumed = True
        else:
            pos = _streaming_initial_positions_from_edges(
                graph.edge_index,
                n,
                base_sep=base_sep,
                seed=seed,
                chunk_size=int(
                    params.get("field_streaming_edge_chunk", _DEFAULT_STREAMING_EDGE_CHUNK)
                ),
            )
            resumed = False
            if checkpoint.armed:
                _record_refine_checkpoint(
                    checkpoint,
                    level=0,
                    pos=pos,
                    phase="streaming",
                    telemetry={"phase": "streaming_init", "num_nodes": n},
                )
                _maybe_exit_after_checkpoint(config, level=0)
        masses = torch.ones((n,), dtype=torch.float32)
        edge_weight = _streaming_edge_weight(graph)
        steps = max(0, int(params.get("field_streaming_refine_steps", 1)))
        for step in range(steps):
            pos = _refine_streaming_level(
                pos,
                graph.edge_index,
                edge_weight,
                node_sizes,
                masses,
                config,
                seed=seed + step,
            )
            if checkpoint.armed:
                _record_refine_checkpoint(
                    checkpoint,
                    level=step + 1,
                    pos=pos,
                    phase="streaming_refine",
                    telemetry={"phase": "streaming_refine", "step": step + 1},
                )
        coarsest_solver = "streaming_cell_field"
    pos = pos.detach().to(device="cpu", dtype=torch.float32)
    if not bool(torch.isfinite(pos).all().item()):
        raise RuntimeError("streaming FIELD strategy produced non-finite positions.")
    telemetry = FieldTelemetry(
        coarsest_nodes=hierarchy.coarsest_graph.num_nodes,
        levels=len(hierarchy.levels),
        coarsest_solver=coarsest_solver,
        wall_s=time.perf_counter() - started,
        pyramid_device=str(getattr(config, "_dagua_field_last_pyramid_device", "cpu")),
        checkpoint_armed=checkpoint.armed,
        resumed=resumed,
    )
    if checkpoint.armed:
        _record_refine_checkpoint(
            checkpoint,
            level=0,
            pos=pos,
            phase="final",
            telemetry=telemetry.to_dict(),
        )
    setattr(config, "_dagua_field_telemetry", telemetry.to_dict())
    return pos


def _should_use_streaming_field(graph: Any, config: LayoutConfig) -> bool:
    """Return whether FIELD must use the CPU-resident streaming regime.

    Parameters
    ----------
    graph : Any
        Graph-like object exposing ``num_nodes`` and ``edge_index``.
    config : LayoutConfig
        Layout configuration.

    Returns
    -------
    bool
        ``True`` when explicitly enabled, the configured node threshold is
        crossed, or the ordinary CUDA refine working set exceeds 70% VRAM.
    """
    params = config.algorithm_params
    n = int(graph.num_nodes)
    e = int(graph.edge_index.shape[1])
    threshold = int(params.get("field_streaming_node_threshold", _DEFAULT_STREAMING_NODE_THRESHOLD))
    if bool(params.get("field_force_streaming", False)):
        return True
    if bool(params.get("field_allow_streaming_fallback", False)) and n >= threshold:
        return True
    if n < threshold:
        return False
    if not str(config.device).startswith("cuda") or not torch.cuda.is_available():
        return bool(params.get("field_streaming_cpu_above_threshold", True))
    estimated_peak = _estimate_refine_peak_bytes(n, e)
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
    except RuntimeError:
        return True
    return estimated_peak > int(free_bytes * 0.70)


def _streaming_initial_positions(num_nodes: int, *, base_sep: float, seed: int) -> torch.Tensor:
    """Return deterministic low-memory initial positions for FIELD streaming.

    Parameters
    ----------
    num_nodes : int
        Node count.
    base_sep : float
        Target local separation.
    seed : int
        Seed folded into the angular phase.

    Returns
    -------
    torch.Tensor
        Initial positions with shape ``[N, 2]``.
    """
    node = torch.arange(int(num_nodes), dtype=torch.float32)
    angle = (node * 2.39996323 + float(seed) * 0.0174533).remainder(6.283185307179586)
    radius = torch.sqrt(node + 1.0) * max(float(base_sep), 1.0) * 0.35
    return torch.stack((torch.cos(angle) * radius, torch.sin(angle) * radius), dim=1)


def _streaming_initial_positions_from_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    base_sep: float,
    seed: int,
    chunk_size: int,
) -> torch.Tensor:
    """Return graph-informed deterministic positions for streaming FIELD.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Node count.
    base_sep : float
        Target local separation.
    seed : int
        Seed folded into deterministic tie-break offsets.
    chunk_size : int
        Edge rows processed per chunk.

    Returns
    -------
    torch.Tensor
        CPU positions with shape ``[N, 2]``.

    Notes
    -----
    The old 100M fallback was a golden-angle spiral. This initializer instead
    builds a one-pass neighborhood barycenter field and assigns nodes to
    deterministic cells by that field, giving edges local baselines before the
    chunked force pass.
    """
    n = int(num_nodes)
    if n <= 0:
        return torch.empty((0, 2), dtype=torch.float32)
    if edge_index.numel() == 0:
        return _streaming_block_positions(n, base_sep=base_sep, seed=seed)

    degree = torch.zeros((n,), dtype=torch.float32)
    neighbor_sum = torch.zeros((n,), dtype=torch.float32)
    edges = edge_index.detach().to(device="cpu", dtype=torch.long)
    chunk = max(1, int(chunk_size))
    for start in range(0, int(edges.shape[1]), chunk):
        end = min(int(edges.shape[1]), start + chunk)
        src = edges[0, start:end]
        dst = edges[1, start:end]
        valid = (src >= 0) & (src < n) & (dst >= 0) & (dst < n) & (src != dst)
        if not bool(valid.any()):
            continue
        src = src[valid]
        dst = dst[valid]
        ones = torch.ones((int(src.numel()),), dtype=torch.float32)
        degree.scatter_add_(0, src, ones)
        degree.scatter_add_(0, dst, ones)
        neighbor_sum.scatter_add_(0, src, dst.to(dtype=torch.float32))
        neighbor_sum.scatter_add_(0, dst, src.to(dtype=torch.float32))

    node = torch.arange(n, dtype=torch.float32)
    anchor = torch.where(degree > 0.0, neighbor_sum / degree.clamp_min(1.0), node)
    del degree, neighbor_sum
    width = int(math.ceil(math.sqrt(n)))
    bucket = max(1, int(math.ceil(n / width)))
    cell = torch.div(anchor.to(dtype=torch.long), bucket, rounding_mode="floor")
    local = torch.remainder(torch.arange(n, dtype=torch.long), width).to(dtype=torch.float32)
    x = (cell.to(dtype=torch.float32) - float(width) * 0.5) * float(base_sep)
    y = (local - float(width) * 0.5) * float(base_sep)
    phase = (node * 12.9898 + anchor * 78.233 + float(seed) * 0.0174533).remainder(
        6.283185307179586
    )
    jitter = max(float(base_sep), 1.0) * 0.18
    x = x + torch.cos(phase) * jitter
    y = y + torch.sin(phase) * jitter
    return _normalize_positions(torch.stack((x, y), dim=1))


def _streaming_block_positions(num_nodes: int, *, base_sep: float, seed: int) -> torch.Tensor:
    """Return deterministic grid-block positions for edgeless streaming inputs.

    Parameters
    ----------
    num_nodes : int
        Node count.
    base_sep : float
        Target local separation.
    seed : int
        Seed folded into deterministic cell jitter.

    Returns
    -------
    torch.Tensor
        CPU positions with shape ``[N, 2]``.
    """
    n = int(num_nodes)
    side = int(math.ceil(math.sqrt(max(1, n))))
    node = torch.arange(n, dtype=torch.float32)
    x = node.remainder(side)
    y = torch.div(node, side, rounding_mode="floor")
    phase = (node * 2.39996323 + float(seed) * 0.0174533).remainder(6.283185307179586)
    jitter = max(float(base_sep), 1.0) * 0.12
    pos = torch.stack(
        (
            (x - float(side) * 0.5) * float(base_sep) + torch.cos(phase) * jitter,
            (y - float(side) * 0.5) * float(base_sep) + torch.sin(phase) * jitter,
        ),
        dim=1,
    )
    return _normalize_positions(pos)


def _refine_streaming_level(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    node_sizes: torch.Tensor,
    node_masses: torch.Tensor,
    config: LayoutConfig,
    *,
    seed: int,
) -> torch.Tensor:
    """Refine CPU-resident FIELD positions with chunked force passes.

    Parameters
    ----------
    pos : torch.Tensor
        CPU positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        CPU edge tensor with shape ``[2, E]``.
    edge_weight : torch.Tensor
        Edge weights with shape ``[E]``.
    node_sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.
    node_masses : torch.Tensor
        Node masses with shape ``[N]``.
    config : LayoutConfig
        Layout configuration.
    seed : int
        Deterministic seed for this streaming pass.

    Returns
    -------
    torch.Tensor
        Refined CPU positions with shape ``[N, 2]``.
    """
    del seed
    params = config.algorithm_params
    work = pos.detach().to(device="cpu", dtype=torch.float32).contiguous()
    base_sep = _target_separation(config, node_sizes)
    step_cap = float(params.get("field_streaming_step_cap", base_sep * 0.75))
    edge_strength = float(params.get("field_edge_strength", 0.15))
    repel_strength = float(params.get("field_repel_strength", base_sep * base_sep * 0.08))
    spring = _streaming_edge_spring_displacement(
        work,
        edge_index,
        edge_weight,
        target_length=base_sep,
        strength=edge_strength,
        chunk_size=int(params.get("field_streaming_edge_chunk", _DEFAULT_STREAMING_EDGE_CHUNK)),
    )
    pyramid = build_grid_pyramid_streaming(
        work,
        node_masses,
        base_cell_size=base_sep
        * float(params.get("field_base_cell_multiplier", _DEFAULT_BASE_CELL_MULTIPLIER)),
        max_cells_per_axis=int(params.get("field_max_grid_axis", _DEFAULT_MAX_GRID_AXIS)),
        max_levels=int(params.get("field_pyramid_levels", _DEFAULT_PYRAMID_LEVELS)),
        chunk_nodes=int(params.get("field_streaming_chunk_nodes", _DEFAULT_STREAMING_CHUNK_NODES)),
    )
    repel = _streaming_far_field_displacement(
        work,
        pyramid,
        config,
        strength=repel_strength,
        softening=base_sep * 0.25,
        max_displacement=step_cap * 4.0,
    )
    disp = spring + repel
    norm = torch.linalg.norm(disp, dim=1, keepdim=True).clamp_min(1.0e-9)
    disp = disp * torch.clamp(step_cap / norm, max=1.0)
    return _normalize_positions(work + disp)


def _streaming_edge_spring_displacement(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    *,
    target_length: float,
    strength: float,
    chunk_size: int,
) -> torch.Tensor:
    """Return deterministic CPU edge-spring displacement in chunks.

    Parameters
    ----------
    pos : torch.Tensor
        CPU positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    edge_weight : torch.Tensor
        Edge weights with shape ``[E]``.
    target_length : float
        Preferred edge length.
    strength : float
        Spring multiplier.
    chunk_size : int
        Number of edges processed per chunk.

    Returns
    -------
    torch.Tensor
        CPU displacement with shape ``[N, 2]``.
    """
    disp = torch.zeros_like(pos)
    if edge_index.numel() == 0 or float(strength) == 0.0:
        return disp
    edges = edge_index.detach().to(device="cpu", dtype=torch.long)
    weights = edge_weight.detach().to(device="cpu", dtype=torch.float32)
    weights = weights / weights.mean().clamp_min(1.0e-6)
    chunk = max(1, int(chunk_size))
    for start in range(0, int(edges.shape[1]), chunk):
        end = min(int(edges.shape[1]), start + chunk)
        src = edges[0, start:end]
        dst = edges[1, start:end]
        delta = pos[dst] - pos[src]
        dist = torch.linalg.norm(delta, dim=1, keepdim=True).clamp_min(1.0e-6)
        magnitude = dist * dist / max(float(target_length), 1.0e-6)
        force = delta / dist * (magnitude * weights[start:end].unsqueeze(1))
        force = force * float(strength)
        disp.index_add_(0, src, force)
        disp.index_add_(0, dst, -force)
    return disp


def _streaming_far_field_displacement(
    pos: torch.Tensor,
    pyramid: Any,
    config: LayoutConfig,
    *,
    strength: float,
    softening: float,
    max_displacement: float,
) -> torch.Tensor:
    """Return far-field displacement by processing CPU positions in chunks.

    Parameters
    ----------
    pos : torch.Tensor
        CPU positions with shape ``[N, 2]``.
    pyramid : Any
        Grid pyramid built from the full CPU-resident positions.
    config : LayoutConfig
        Layout configuration.
    strength : float
        Repulsion strength.
    softening : float
        Distance softening.
    max_displacement : float
        Per-node displacement cap.

    Returns
    -------
    torch.Tensor
        CPU displacement with shape ``[N, 2]``.
    """
    params = config.algorithm_params
    chunk = max(1, int(params.get("field_streaming_chunk_nodes", _DEFAULT_STREAMING_CHUNK_NODES)))
    resident_bytes = pyramid_tensor_bytes(pyramid)
    out = torch.empty_like(pos)
    used_device = "cpu"
    for start in range(0, int(pos.shape[0]), chunk):
        end = min(int(pos.shape[0]), start + chunk)
        chunk_device = choose_streaming_chunk_device(
            end - start,
            str(config.device),
            resident_pyramid_bytes=resident_bytes,
        )
        used_device = "cuda" if chunk_device == "cuda" else used_device
        chunk_pos = pos[start:end].to(device=chunk_device, dtype=torch.float32)
        chunk_pyramid = (
            _pyramid_to_device(pyramid, chunk_device) if chunk_device == "cuda" else pyramid
        )
        out[start:end] = far_field_repulsion_force(
            chunk_pos,
            chunk_pyramid,
            strength=float(strength),
            softening=float(softening),
            max_displacement=float(max_displacement),
            level_stride=int(params.get("field_pyramid_stride", 2)),
        ).to(device="cpu")
        if chunk_device == "cuda":
            del chunk_pos, chunk_pyramid
            torch.cuda.empty_cache()
    setattr(
        config,
        "_dagua_field_last_pyramid_device",
        f"{pyramid.device_kind}+{used_device}_chunks",
    )
    return out


def _pyramid_to_device(pyramid: Any, device: str) -> Any:
    """Copy a small grid pyramid to the requested chunk device.

    Parameters
    ----------
    pyramid : Any
        Grid pyramid.
    device : str
        Target device.

    Returns
    -------
    Any
        Grid pyramid with all level tensors on ``device``.
    """
    from dagua.layout.scale.pyramid import GridLevel, GridPyramid

    levels = [
        GridLevel(
            mass=level.mass.to(device=device),
            centroid=level.centroid.to(device=device),
            cell_size=level.cell_size.to(device=device),
            origin=level.origin.to(device=device),
        )
        for level in pyramid.levels
    ]
    return GridPyramid(levels=levels, device_kind=str(device))


def _resolved_graph_node_sizes_for_streaming(graph: Any, config: LayoutConfig) -> torch.Tensor:
    """Return node sizes for streaming FIELD.

    Parameters
    ----------
    graph : Any
        Graph-like object with optional ``node_sizes``.
    config : LayoutConfig
        Layout config used for a default size.

    Returns
    -------
    torch.Tensor
        CPU node-size tensor with shape ``[N, 2]``.
    """
    if getattr(graph, "node_sizes", None) is not None:
        return _resolved_graph_node_sizes(graph)
    default_size = max(1.0, float(getattr(config, "node_sep", 70.0)) * 0.20)
    return torch.full((int(graph.num_nodes), 2), default_size, dtype=torch.float32)


def _streaming_edge_weight(graph: Any) -> torch.Tensor:
    """Return CPU edge weights for a tensor-only graph.

    Parameters
    ----------
    graph : Any
        Graph-like object exposing optional ``edge_weights``.

    Returns
    -------
    torch.Tensor
        Edge weights with shape ``[E]``.
    """
    edge_weights = getattr(graph, "edge_weights", None)
    edge_count = int(graph.edge_index.shape[1])
    if edge_weights is None:
        return torch.ones((edge_count,), dtype=torch.float32)
    return edge_weights.detach().to(device="cpu", dtype=torch.float32)


def _resolved_graph_node_sizes(graph: Any) -> torch.Tensor:
    """Return graph node sizes as CPU float boxes.

    Parameters
    ----------
    graph : Any
        Graph-like object with optional ``node_sizes``.

    Returns
    -------
    torch.Tensor
        Node sizes with shape ``[N, 2]``.
    """
    node_sizes = getattr(graph, "node_sizes", None)
    if node_sizes is None:
        return torch.ones((int(graph.num_nodes), 2), dtype=torch.float32)
    sizes = node_sizes.detach().to(device="cpu", dtype=torch.float32)
    if sizes.ndim == 1:
        sizes = sizes.unsqueeze(1).expand(-1, 2).contiguous()
    if tuple(sizes.shape) != (int(graph.num_nodes), 2):
        raise ValueError("graph.node_sizes must have shape [N, 2].")
    return sizes


def _solve_coarsest(
    hierarchy: ScaleHierarchy,
    config: LayoutConfig,
    *,
    seed: int,
    solver: str,
) -> torch.Tensor:
    """Solve the coarsest FIELD graph.

    Parameters
    ----------
    hierarchy : ScaleHierarchy
        Built scale hierarchy.
    config : LayoutConfig
        Layout configuration.
    seed : int
        Deterministic seed.
    solver : str
        ``"stress_sgd"``, ``"native"``, or ``"auto"``.

    Returns
    -------
    torch.Tensor
        Coarsest positions with shape ``[N_coarse, 2]``.
    """
    graph = hierarchy.coarsest_graph
    problem = LayoutProblem(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        node_sizes=hierarchy.coarsest_node_sizes,
        direction=config.direction,
        edge_weights=graph.edge_weight,
        seed=int(seed),
    )
    if solver == "stress_sgd":
        return _stress_sgd_coarsest(problem, seed)
    if solver == "native":
        native_config = copy.copy(config)
        native_config.algorithm_params = dict(config.algorithm_params)
        native_config.algorithm_params["scale_native_max_nodes"] = max(
            int(graph.num_nodes),
            int(native_config.algorithm_params.get("scale_native_max_nodes", graph.num_nodes)),
        )
        return anytime_native_coarsest(
            problem,
            native_config,
            time_budget_s=float(config.algorithm_params.get("field_coarsest_budget_s", 8.0)),
            seed=int(seed),
        )
    return anytime_native_coarsest(
        problem,
        config,
        time_budget_s=float(config.algorithm_params.get("field_coarsest_budget_s", 4.0)),
        seed=int(seed),
    )


def _stress_sgd_coarsest(problem: LayoutProblem, seed: int) -> torch.Tensor:
    """Run the deterministic Stress-SGD coarsest solver.

    Parameters
    ----------
    problem : LayoutProblem
        Coarsest problem.
    seed : int
        Deterministic seed.

    Returns
    -------
    torch.Tensor
        Finite coarsest positions with shape ``[N, 2]``.
    """
    result = layout_stress_sgd_pipeline(
        edge_index=problem.edge_index,
        num_nodes=int(problem.num_nodes),
        node_sizes=problem.node_sizes,
        steps=30,
        seed=int(seed),
        max_exact_nodes=512,
        edge_weights=problem.edge_weights,
    )
    pos = result[0] if isinstance(result, tuple) else result
    pos = pos.detach().to(device="cpu", dtype=torch.float32)
    if not bool(torch.isfinite(pos).all().item()):
        raise RuntimeError("stress_sgd coarsest produced non-finite positions.")
    return pos


def _fine_graph_for_transition(hierarchy: ScaleHierarchy, level_index: int) -> ScaleGraph:
    """Return the fine graph refined after a prolongation transition.

    Parameters
    ----------
    hierarchy : ScaleHierarchy
        Scale hierarchy.
    level_index : int
        Transition index.

    Returns
    -------
    ScaleGraph
        Graph at the fine side of the transition.
    """
    if int(level_index) == 0:
        return hierarchy.finest_graph
    previous = hierarchy.levels[int(level_index) - 1]
    return ScaleGraph(
        num_nodes=previous.coarse_num_nodes,
        edge_index=previous.edge_index,
        edge_weight=previous.edge_weight,
        adjacency=[],
    )


def _fine_sizes_for_transition(
    hierarchy: ScaleHierarchy,
    level_index: int,
    finest_num_nodes: int,
    finest_sizes: torch.Tensor,
) -> torch.Tensor:
    """Return node sizes at the fine side of a transition.

    Parameters
    ----------
    hierarchy : ScaleHierarchy
        Scale hierarchy.
    level_index : int
        Transition index.
    finest_num_nodes : int
        Number of finest graph nodes.
    finest_sizes : torch.Tensor
        Finest node sizes with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Fine-side node sizes with shape ``[N_level, 2]``.
    """
    del finest_num_nodes
    if int(level_index) == 0:
        return finest_sizes
    return hierarchy.levels[int(level_index) - 1].node_sizes


def _fine_masses_for_transition(
    hierarchy: ScaleHierarchy,
    level_index: int,
    finest_num_nodes: int,
) -> torch.Tensor:
    """Return node masses at the fine side of a transition.

    Parameters
    ----------
    hierarchy : ScaleHierarchy
        Scale hierarchy.
    level_index : int
        Transition index.
    finest_num_nodes : int
        Number of finest nodes.

    Returns
    -------
    torch.Tensor
        Fine-side masses with shape ``[N_level]``.
    """
    if int(level_index) == 0:
        return torch.ones((int(finest_num_nodes),), dtype=torch.float32)
    return hierarchy.levels[int(level_index) - 1].node_masses


def _refine_level(
    pos: torch.Tensor,
    graph: ScaleGraph,
    node_sizes: torch.Tensor,
    node_masses: torch.Tensor,
    config: LayoutConfig,
    *,
    seed: int,
    is_finest: bool,
) -> torch.Tensor:
    """Refine one FIELD level with spring and grid-pyramid forces.

    Parameters
    ----------
    pos : torch.Tensor
        Current positions with shape ``[N, 2]``.
    graph : ScaleGraph
        Level graph.
    node_sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.
    node_masses : torch.Tensor
        Node masses with shape ``[N]``.
    config : LayoutConfig
        Layout configuration.
    seed : int
        Deterministic seed.
    is_finest : bool
        Whether this is the final finest refinement.

    Returns
    -------
    torch.Tensor
        Refined positions with shape ``[N, 2]``.
    """
    del seed
    params = config.algorithm_params
    steps = _refine_steps_for_level(graph.num_nodes, params, is_finest=is_finest)
    if steps <= 0:
        return pos.detach().to(dtype=torch.float32)
    refine_device = _choose_refine_device(
        graph.num_nodes,
        int(graph.edge_index.shape[1]),
        str(config.device),
    )
    work = pos.detach().to(device=refine_device, dtype=torch.float32).clone()
    sizes = node_sizes.detach().to(device=refine_device, dtype=torch.float32)
    masses = node_masses.detach().to(device=refine_device, dtype=torch.float32)
    base_sep = _target_separation(config, sizes)
    mean_mass = float(masses.mean().item()) if masses.numel() else 1.0
    edge_strength = float(params.get("field_edge_strength", 0.15))
    repel_strength = float(
        params.get(
            "field_repel_strength",
            base_sep * base_sep * 0.12 / max(1.0, mean_mass),
        )
    )
    cooling = float(params.get("field_cooling", 0.90))
    step0 = float(params.get("field_step0", base_sep * 0.90))
    overlap_strength = float(params.get("field_overlap_strength", 0.0))
    for step in range(steps):
        step_cap = step0 * cooling**step
        spring = _edge_spring_displacement(
            work,
            graph.edge_index,
            graph.edge_weight,
            target_length=base_sep,
            strength=edge_strength,
            chunk_size=int(params.get("field_edge_chunk", _DEFAULT_EDGE_CHUNK)),
            spring_model=str(params.get("field_spring_model", "fr")),
        )
        pyramid_device = choose_pyramid_device(work.shape[0], str(config.device))
        setattr(config, "_dagua_field_last_pyramid_device", pyramid_device)
        pyramid_work = work.to(device=pyramid_device) if pyramid_device == "cuda" else work
        pyramid_masses = masses.to(device=pyramid_device) if pyramid_device == "cuda" else masses
        pyramid = build_grid_pyramid(
            pyramid_work,
            pyramid_masses,
            base_cell_size=base_sep
            * float(
                params.get(
                    "field_base_cell_multiplier",
                    _DEFAULT_BASE_CELL_MULTIPLIER,
                )
            ),
            max_cells_per_axis=int(params.get("field_max_grid_axis", _DEFAULT_MAX_GRID_AXIS)),
            max_levels=int(params.get("field_pyramid_levels", _DEFAULT_PYRAMID_LEVELS)),
            force_cpu=pyramid_device == "cpu",
        )
        repel = far_field_repulsion_force(
            pyramid_work,
            pyramid,
            strength=repel_strength,
            softening=base_sep * 0.25,
            max_displacement=step0 * 4.0,
            level_stride=int(params.get("field_pyramid_stride", 2)),
        ).to(device=work.device)
        disp = spring + repel
        norm = torch.linalg.norm(disp, dim=1, keepdim=True).clamp_min(1.0e-9)
        disp = disp * torch.clamp(step_cap / norm, max=1.0)
        work = work + disp
        if overlap_strength > 0.0:
            work = _grid_local_overlap_projection(
                work,
                sizes,
                cell_size=base_sep,
                strength=overlap_strength,
                max_displacement=step_cap * 0.5,
            )
    return _normalize_positions(work).to(device="cpu", dtype=torch.float32)


def _choose_refine_device(num_nodes: int, num_edges: int, requested_device: str) -> str:
    """Choose the measured-safe device for one FIELD refinement level.

    Parameters
    ----------
    num_nodes : int
        Current level node count.
    num_edges : int
        Current level edge count.
    requested_device : str
        User-requested layout device.

    Returns
    -------
    str
        ``"cuda"`` when the level's positions, edges, and spring temporaries
        fit under 70% of free VRAM; otherwise ``"cpu"``.
    """
    if not str(requested_device).startswith("cuda") or not torch.cuda.is_available():
        return "cpu"
    estimated_peak = _estimate_refine_peak_bytes(int(num_nodes), int(num_edges))
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
    except RuntimeError:
        return "cpu"
    if estimated_peak > int(free_bytes * 0.70):
        return "cpu"
    return "cuda"


def _estimate_refine_peak_bytes(num_nodes: int, num_edges: int) -> int:
    """Estimate FIELD refinement peak VRAM for a single level.

    Parameters
    ----------
    num_nodes : int
        Level node count.
    num_edges : int
        Level edge count.

    Returns
    -------
    int
        Conservative bytes for positions, displacement, sizes, masses, copied
        edge tensors, weights, and chunk-local spring buffers.
    """
    node_bytes = int(num_nodes) * (2 * 4 + 2 * 4 + 2 * 4 + 4)
    edge_bytes = int(num_edges) * (2 * 8 + 4)
    chunk_edges = min(int(num_edges), _DEFAULT_EDGE_CHUNK)
    chunk_bytes = chunk_edges * (2 * 8 + 2 * 2 * 4 + 4 + 2 * 4)
    return int((node_bytes + edge_bytes + chunk_bytes) * 1.5)


def _refine_steps_for_level(
    num_nodes: int,
    params: dict[str, Any],
    *,
    is_finest: bool,
) -> int:
    """Return bounded refinement steps for a FIELD level.

    Parameters
    ----------
    num_nodes : int
        Current level node count.
    params : dict[str, Any]
        Algorithm parameters.
    is_finest : bool
        Whether this is the finest level.

    Returns
    -------
    int
        Number of refinement steps.
    """
    if "field_refine_steps" in params:
        return max(0, int(params["field_refine_steps"]))
    n = int(num_nodes)
    if n <= 5_000:
        base = 40
    elif n <= 30_000:
        base = 22
    elif n <= 120_000:
        base = 10
    elif n <= 400_000:
        base = 8
    else:
        base = 6
    return base + (5 if is_finest and n <= 30_000 else 0)


def _should_pack_components(num_nodes: int, params: dict[str, Any]) -> bool:
    """Return whether FIELD should run the component packing pass.

    Parameters
    ----------
    num_nodes : int
        Finest graph node count.
    params : dict[str, Any]
        Algorithm parameters.

    Returns
    -------
    bool
        ``True`` when the exact component pass is within the configured node
        budget. The pass is skipped by default at million-scale because it is
        a repeated global edge scan after the layout has already converged.
    """
    threshold = int(params.get("field_pack_components_node_threshold", 1_000_000))
    return int(num_nodes) <= threshold


def _edge_spring_displacement(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    *,
    target_length: float,
    strength: float,
    chunk_size: int,
    spring_model: str = "fr",
) -> torch.Tensor:
    """Return chunked edge-spring displacement.

    Parameters
    ----------
    pos : torch.Tensor
        Positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    edge_weight : torch.Tensor
        Edge weights with shape ``[E]``.
    target_length : float
        Preferred edge length.
    strength : float
        Spring multiplier.
    chunk_size : int
        Number of edges processed per chunk.
    spring_model : str, default="fr"
        ``"fr"`` uses Fruchterman-Reingold attraction ``d^2 / target`` so edge
        lengths adapt to local density (the sfdp model); ``"linear"`` pulls
        every edge toward ``target_length``.

    Returns
    -------
    torch.Tensor
        Displacement tensor with shape ``[N, 2]``.
    """
    disp = torch.zeros_like(pos)
    if edge_index.numel() == 0 or strength == 0.0:
        return disp
    edges = edge_index.to(device=pos.device, dtype=torch.long)
    weights = edge_weight.to(device=pos.device, dtype=torch.float32)
    weights = weights / weights.mean().clamp_min(1.0e-6)
    if pos.device.type == "cuda":
        return _edge_spring_displacement_sorted(
            pos,
            edges,
            weights,
            target_length=target_length,
            strength=strength,
            spring_model=spring_model,
        )
    edge_count = int(edges.shape[1])
    chunk = max(1, int(chunk_size))
    for start in range(0, edge_count, chunk):
        end = min(edge_count, start + chunk)
        src = edges[0, start:end]
        dst = edges[1, start:end]
        delta = pos[dst] - pos[src]
        dist = torch.linalg.norm(delta, dim=1, keepdim=True).clamp_min(1.0e-6)
        if spring_model == "fr":
            magnitude = dist * dist / max(float(target_length), 1.0e-6)
        else:
            magnitude = dist - float(target_length)
        force = delta / dist * (magnitude * weights[start:end].unsqueeze(1))
        force = force * float(strength)
        disp.index_add_(0, src, force)
        disp.index_add_(0, dst, -force)
    return disp


def _edge_spring_displacement_sorted(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    *,
    target_length: float,
    strength: float,
    spring_model: str,
) -> torch.Tensor:
    """Return deterministic edge-spring displacement by sorted reductions.

    Parameters
    ----------
    pos : torch.Tensor
        Positions with shape ``[N, 2]`` on CUDA.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]`` on ``pos.device``.
    edge_weight : torch.Tensor
        Normalized edge weights with shape ``[E]`` on ``pos.device``.
    target_length : float
        Preferred edge length.
    strength : float
        Spring multiplier.
    spring_model : str
        ``"fr"`` or ``"linear"`` spring model.

    Returns
    -------
    torch.Tensor
        Displacement tensor with shape ``[N, 2]``.
    """
    src = edge_index[0]
    dst = edge_index[1]
    delta = pos[dst] - pos[src]
    dist = torch.linalg.norm(delta, dim=1, keepdim=True).clamp_min(1.0e-6)
    if spring_model == "fr":
        magnitude = dist * dist / max(float(target_length), 1.0e-6)
    else:
        magnitude = dist - float(target_length)
    force = delta / dist * (magnitude * edge_weight.unsqueeze(1))
    force = force * float(strength)
    node = torch.cat((src, dst))
    contribution = torch.cat((force, -force), dim=0)
    order = node.argsort(stable=True)
    sorted_node = node[order]
    first = torch.ones_like(sorted_node, dtype=torch.bool)
    first[1:] = sorted_node[1:] != sorted_node[:-1]
    starts = torch.nonzero(first, as_tuple=False).flatten()
    lengths = torch.diff(
        torch.cat(
            (
                starts,
                torch.tensor([sorted_node.numel()], dtype=torch.long, device=pos.device),
            )
        )
    )
    summed_x = torch.segment_reduce(contribution[order, 0], "sum", lengths=lengths)
    summed_y = torch.segment_reduce(contribution[order, 1], "sum", lengths=lengths)
    disp = torch.zeros_like(pos)
    unique_node = sorted_node[starts]
    disp[unique_node, 0] = summed_x
    disp[unique_node, 1] = summed_y
    return disp


def _grid_local_overlap_projection(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    *,
    cell_size: float,
    strength: float,
    max_displacement: float,
) -> torch.Tensor:
    """Apply grid-local density separation without a global argsort.

    Parameters
    ----------
    pos : torch.Tensor
        Positions with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.
    cell_size : float
        Projection grid cell size.
    strength : float
        Projection multiplier.
    max_displacement : float
        Per-node displacement cap.

    Returns
    -------
    torch.Tensor
        Projected positions with shape ``[N, 2]``.
    """
    if pos.shape[0] <= 1 or strength <= 0.0:
        return pos
    min_xy = pos.min(dim=0).values
    max_xy = pos.max(dim=0).values
    span = torch.clamp(max_xy - min_xy, min=float(cell_size))
    counts = torch.clamp(
        torch.ceil(span / max(float(cell_size), 1.0e-6)).to(dtype=torch.long),
        min=1,
        max=512,
    )
    width = int(counts[0].item())
    height = int(counts[1].item())
    grid_size = torch.tensor([width, height], dtype=torch.long, device=pos.device)
    origin = min_xy - 0.5 * float(cell_size)
    ij = torch.floor((pos - origin) / max(float(cell_size), 1.0e-6)).to(dtype=torch.long)
    ij = torch.minimum(torch.maximum(ij, torch.zeros_like(ij)), grid_size.unsqueeze(0) - 1)
    cell_id = ij[:, 1] * width + ij[:, 0]
    flat_count = torch.bincount(cell_id, minlength=width * height).to(
        device=pos.device,
        dtype=torch.float32,
    )
    crowded = flat_count[cell_id].unsqueeze(1).clamp_min(1.0)
    cell_center = origin + (ij.to(dtype=torch.float32) + 0.5) * float(cell_size)
    delta = pos - cell_center
    norm = torch.linalg.norm(delta, dim=1, keepdim=True)
    fallback_angle = (
        torch.arange(pos.shape[0], device=pos.device, dtype=torch.float32) * 2.39996323
    ).unsqueeze(1)
    fallback = torch.cat((torch.cos(fallback_angle), torch.sin(fallback_angle)), dim=1)
    direction = torch.where(norm > 1.0e-6, delta / norm.clamp_min(1.0e-6), fallback)
    size_scale = (
        node_sizes.to(device=pos.device, dtype=torch.float32)
        .mean(dim=1, keepdim=True)
        .clamp_min(1.0)
    )
    amount = (crowded - 1.0).clamp_min(0.0) / crowded * size_scale * float(strength)
    amount = torch.clamp(amount, max=float(max_displacement))
    return pos + direction * amount


def _connected_component_labels(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Label weakly connected components with deterministic min-label hooking.

    Uses scatter-min hooking plus pointer-jumping compression, converging in
    ``O(log N)`` rounds without any non-torch dependency.

    Parameters
    ----------
    edge_index : torch.Tensor
        Undirected edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    torch.Tensor
        Per-node component labels with shape ``[N]`` (min node id per component).
    """
    labels = torch.arange(int(num_nodes), dtype=torch.long)
    if edge_index.numel() == 0:
        return labels
    src = edge_index[0].to(dtype=torch.long)
    dst = edge_index[1].to(dtype=torch.long)
    for _ in range(64):
        prev = labels.clone()
        merged = torch.minimum(labels[src], labels[dst])
        labels.scatter_reduce_(0, src, merged, reduce="amin")
        labels.scatter_reduce_(0, dst, merged, reduce="amin")
        labels = torch.minimum(labels, labels[labels])
        labels = torch.minimum(labels, labels[labels])
        if bool(torch.equal(labels, prev)):
            break
    return labels


def _pack_components(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    base_sep: float,
) -> torch.Tensor:
    """Shelf-pack non-giant components below the giant component.

    Mirrors the graphviz pack behavior: the giant component keeps its layout
    while every smaller component (including singletons) is moved into
    deterministic rows below the giant bounding box instead of being flung
    outward by the global far-field repulsion.

    Parameters
    ----------
    pos : torch.Tensor
        Finest positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Finest undirected edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    base_sep : float
        Target separation used as packing padding.

    Returns
    -------
    torch.Tensor
        Packed positions with shape ``[N, 2]``.
    """
    if int(num_nodes) <= 1:
        return pos
    labels = _connected_component_labels(edge_index, int(num_nodes))
    uniq, inverse, counts = torch.unique(labels, return_inverse=True, return_counts=True)
    if int(uniq.numel()) <= 1:
        return pos
    comp_count = int(uniq.numel())
    finite_pos = pos.detach().to(dtype=torch.float32)
    lo = torch.full((comp_count, 2), float("inf"), dtype=torch.float32)
    hi = torch.full((comp_count, 2), float("-inf"), dtype=torch.float32)
    index2 = inverse.unsqueeze(1).expand(-1, 2)
    lo.scatter_reduce_(0, index2, finite_pos, reduce="amin")
    hi.scatter_reduce_(0, index2, finite_pos, reduce="amax")
    giant = int(torch.argmax(counts).item())
    pad = max(1.0, float(base_sep))
    widths = (hi[:, 0] - lo[:, 0]).clamp_min(pad).tolist()
    heights = (hi[:, 1] - lo[:, 1]).clamp_min(pad).tolist()
    lo_list = lo.tolist()
    order = sorted(
        (index for index in range(comp_count) if index != giant),
        key=lambda index: (-heights[index], index),
    )
    strip_width = max(
        widths[giant],
        math.sqrt(sum((widths[i] + pad) * (heights[i] + pad) for i in order)),
    )
    offset_rows = [[0.0, 0.0] for _ in range(comp_count)]
    cursor_x = 0.0
    cursor_y = lo_list[giant][1] - 2.0 * pad
    row_height = 0.0
    origin_x = lo_list[giant][0]
    for index in order:
        w = widths[index]
        h = heights[index]
        if cursor_x > 0.0 and cursor_x + w > strip_width:
            cursor_x = 0.0
            cursor_y -= row_height + pad
            row_height = 0.0
        offset_rows[index][0] = origin_x + cursor_x - lo_list[index][0]
        offset_rows[index][1] = (cursor_y - h) - lo_list[index][1]
        cursor_x += w + pad
        row_height = max(row_height, h)
    offsets = torch.tensor(offset_rows, dtype=torch.float32)
    return finite_pos + offsets[inverse]


def _expand_for_prolongation(
    pos: torch.Tensor,
    *,
    fine_num_nodes: int,
    coarse_num_nodes: int,
    expand_gain: float,
) -> torch.Tensor:
    """Expand coarse positions before prolongation to preserve density.

    The fine level carries ``fine_num_nodes / coarse_num_nodes`` more nodes
    than the coarse level, so the drawing area must grow by the same factor
    (linear scale ``sqrt(ratio)``) or every prolongation collapses siblings
    onto their parents (the FM3 multilevel scale-up step).

    Parameters
    ----------
    pos : torch.Tensor
        Coarse positions with shape ``[N_coarse, 2]``.
    fine_num_nodes : int
        Node count at the fine side of the transition.
    coarse_num_nodes : int
        Node count at the coarse side of the transition.
    expand_gain : float
        User multiplier on the density-preserving expansion factor.

    Returns
    -------
    torch.Tensor
        Expanded coarse positions with shape ``[N_coarse, 2]``.
    """
    if pos.shape[0] <= 1:
        return pos
    ratio = float(fine_num_nodes) / float(max(1, coarse_num_nodes))
    scale = math.sqrt(max(1.0, ratio)) * float(expand_gain)
    scale = min(max(scale, 1.0), 4.0)
    center = pos.mean(dim=0, keepdim=True)
    return center + (pos - center) * scale


def _target_separation(config: LayoutConfig, node_sizes: torch.Tensor) -> float:
    """Return FIELD's local target separation.

    Parameters
    ----------
    config : LayoutConfig
        Layout configuration.
    node_sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.

    Returns
    -------
    float
        Positive target separation in layout units.
    """
    size_term = float(node_sizes.mean().item()) if node_sizes.numel() else 1.0
    configured = float(getattr(config, "node_sep", 70.0))
    return max(1.0, min(configured, max(configured * 0.35, size_term * 1.5)))


def _jitter_scale_for_level(config: LayoutConfig, fine_num_nodes: int) -> float:
    """Return deterministic prolongation jitter for a level.

    Parameters
    ----------
    config : LayoutConfig
        Layout configuration.
    fine_num_nodes : int
        Fine node count.

    Returns
    -------
    float
        Jitter radius in layout units.
    """
    base = float(
        config.algorithm_params.get(
            "field_prolong_jitter",
            max(1.0, config.node_sep * 0.12),
        )
    )
    if int(fine_num_nodes) >= 1_000_000:
        return base * 0.5
    return base


def _normalize_positions(pos: torch.Tensor) -> torch.Tensor:
    """Center positions and replace accidental NaN/Inf values.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Finite centered positions with shape ``[N, 2]``.
    """
    clean = torch.nan_to_num(
        pos.detach().to(dtype=torch.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if clean.numel() == 0:
        return clean
    clean = clean - clean.mean(dim=0, keepdim=True)
    span = clean.max(dim=0).values - clean.min(dim=0).values
    if float(span.max().item()) <= 1.0e-6:
        n = clean.shape[0]
        side = int(math.ceil(math.sqrt(max(1, n))))
        idx = torch.arange(n, dtype=torch.float32, device=clean.device)
        clean = torch.stack(
            (idx.remainder(side), torch.div(idx, side, rounding_mode="floor")),
            dim=1,
        )
    return clean.contiguous()


__all__ = ["FieldScaleStrategy", "FieldTelemetry", "layout_field"]
