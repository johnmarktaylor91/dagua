"""LAYERS scale strategy for very large acyclic graphs."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional, cast

import torch

from dagua.config import LayoutConfig
from dagua.layout.layers import build_layer_index
from dagua.layout.scale.checkpoint import ScaleCheckpointManager, release_memory
from dagua.layout.scale.sketch import TopologySketch
from dagua.utils import longest_path_layering


@dataclass(frozen=True)
class LayersTelemetry:
    """Execution telemetry for one LAYERS run.

    Parameters
    ----------
    num_layers : int
        Number of computed DAG ranks.
    max_layer_width : int
        Maximum nodes in one rank.
    wall_s : float
        End-to-end wall time.
    checkpoint_armed : bool
        Whether checkpointing was armed for the run.
    resumed : bool
        Whether final positions were restored from checkpoint.
    """

    num_layers: int
    max_layer_width: int
    wall_s: float
    checkpoint_armed: bool
    resumed: bool

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly telemetry.

        Returns
        -------
        dict[str, object]
            Serialized telemetry payload.
        """
        return {
            "num_layers": int(self.num_layers),
            "max_layer_width": int(self.max_layer_width),
            "wall_s": float(self.wall_s),
            "checkpoint_armed": bool(self.checkpoint_armed),
            "resumed": bool(self.resumed),
        }


class LayersScaleStrategy:
    """Tensor-native LAYERS strategy for above-gate DAGs."""

    def layout(
        self,
        graph: Any,
        config: LayoutConfig,
        sketch: TopologySketch,
        *,
        trace: Any = None,
    ) -> torch.Tensor:
        """Compute a deterministic layered layout for a DAG.

        Parameters
        ----------
        graph : Any
            Prepared graph-like object exposing ``edge_index`` and ``num_nodes``.
        config : LayoutConfig
            Layout configuration.
        sketch : TopologySketch
            Topology sketch used by the scale router.
        trace : Any, optional
            Optional trace sink, currently unused.

        Returns
        -------
        torch.Tensor
            Finite positions with shape ``[N, 2]``.
        """
        del trace
        started = time.perf_counter()
        checkpoint = ScaleCheckpointManager.from_config(config, sketch, strategy="LAYERS")
        final_record = checkpoint.record_for("final", 0) if checkpoint.armed else None
        if final_record is not None:
            pos = checkpoint.load_tensors("final", 0)["pos"].to(dtype=torch.float32)
            telemetry = LayersTelemetry(
                num_layers=int(cast(Any, final_record.telemetry.get("num_layers", 0))),
                max_layer_width=int(cast(Any, final_record.telemetry.get("max_layer_width", 0))),
                wall_s=time.perf_counter() - started,
                checkpoint_armed=True,
                resumed=True,
            )
            setattr(config, "_dagua_layers_telemetry", telemetry.to_dict())
            return pos

        layer_assignments = _load_or_compute_layers(graph, config, checkpoint)
        pos, num_layers, max_width = _positions_from_layers(graph, config, layer_assignments)
        pos = pos.detach().to(device="cpu", dtype=torch.float32)
        if not bool(torch.isfinite(pos).all().item()):
            raise RuntimeError("LAYERS strategy produced non-finite positions.")
        telemetry_payload = {
            "num_layers": int(num_layers),
            "max_layer_width": int(max_width),
            "graph_fingerprint": sketch.fingerprint,
        }
        if checkpoint.armed:
            before, after, _released = release_memory()
            checkpoint.record(
                phase="final",
                level=0,
                tensors={"pos": pos},
                telemetry=telemetry_payload,
                rss_before_release=before,
                rss_after_release=after,
            )
        telemetry = LayersTelemetry(
            num_layers=int(num_layers),
            max_layer_width=int(max_width),
            wall_s=time.perf_counter() - started,
            checkpoint_armed=checkpoint.armed,
            resumed=bool(getattr(config, "_dagua_layers_resumed_from_checkpoint", False)),
        )
        setattr(config, "_dagua_layers_telemetry", telemetry.to_dict())
        return pos


def layout_layers(
    graph: Any,
    config: LayoutConfig,
    sketch: TopologySketch,
    *,
    trace: Any = None,
) -> torch.Tensor:
    """Compute a LAYERS layout using the default strategy instance.

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
    return LayersScaleStrategy().layout(graph, config, sketch, trace=trace)


def _load_or_compute_layers(
    graph: Any,
    config: LayoutConfig,
    checkpoint: ScaleCheckpointManager,
) -> torch.Tensor:
    """Load layer assignments or compute them from the edge tensor.

    Parameters
    ----------
    graph : Any
        Graph-like object exposing ``edge_index`` and ``num_nodes``.
    config : LayoutConfig
        Layout configuration.
    checkpoint : ScaleCheckpointManager
        Optional checkpoint manager.

    Returns
    -------
    torch.Tensor
        Layer assignment tensor with shape ``[N]``.
    """
    if checkpoint.armed and checkpoint.record_for("layers", 0) is not None:
        setattr(config, "_dagua_layers_resumed_from_checkpoint", True)
        return checkpoint.load_tensors("layers", 0)["layer_assignments"].to(dtype=torch.long)
    raw_layers = longest_path_layering(
        graph.edge_index,
        int(graph.num_nodes),
        device=str(config.device),
        verbose=bool(config.verbose),
    )
    layer_assignments = (
        raw_layers.detach().to(device="cpu", dtype=torch.long)
        if isinstance(raw_layers, torch.Tensor)
        else torch.tensor(raw_layers, dtype=torch.long)
    )
    if checkpoint.armed:
        before, after, _released = release_memory()
        checkpoint.record(
            phase="layers",
            level=0,
            tensors={"layer_assignments": layer_assignments},
            telemetry={"num_layers": _num_layers(layer_assignments)},
            rss_before_release=before,
            rss_after_release=after,
        )
        _maybe_exit_after_checkpoint(config, level=0)
    return layer_assignments


def _positions_from_layers(
    graph: Any,
    config: LayoutConfig,
    layer_assignments: torch.Tensor,
) -> tuple[torch.Tensor, int, int]:
    """Build deterministic positions from layer assignments.

    Parameters
    ----------
    graph : Any
        Graph-like object exposing optional ``node_sizes``.
    config : LayoutConfig
        Layout configuration.
    layer_assignments : torch.Tensor
        Layer assignment tensor with shape ``[N]``.

    Returns
    -------
    tuple[torch.Tensor, int, int]
        Positions ``[N, 2]``, number of layers, and maximum rank width.
    """
    n = int(layer_assignments.shape[0])
    if n == 0:
        return torch.empty((0, 2), dtype=torch.float32), 0, 0
    layer_index = build_layer_index(layer_assignments, device="cpu", enable_cuda_sort=False)
    sorted_nodes = layer_index.sorted_nodes.to(dtype=torch.long)
    counts = layer_index.layer_sizes().to(dtype=torch.long)
    max_width = int(counts.max().item()) if counts.numel() else 0
    ordinal_sorted = torch.arange(n, dtype=torch.float32) - torch.repeat_interleave(
        layer_index.layer_offsets[:-1].to(dtype=torch.float32),
        counts,
    )
    width_sorted = torch.repeat_interleave(counts.to(dtype=torch.float32), counts).clamp_min(1.0)
    node_sep = _node_separation(graph, config)
    rank_sep = max(float(config.rank_sep), node_sep)
    x_sorted = (ordinal_sorted - (width_sorted - 1.0) * 0.5) * node_sep
    y_sorted = layer_assignments[sorted_nodes].to(dtype=torch.float32) * rank_sep
    pos_sorted = torch.stack((x_sorted, y_sorted), dim=1)
    pos = torch.empty((n, 2), dtype=torch.float32)
    pos[sorted_nodes] = pos_sorted
    return pos, int(layer_index.num_layers), max_width


def _node_separation(graph: Any, config: LayoutConfig) -> float:
    """Return horizontal rank spacing large enough to avoid node overlap.

    Parameters
    ----------
    graph : Any
        Graph-like object exposing optional ``node_sizes``.
    config : LayoutConfig
        Layout configuration.

    Returns
    -------
    float
        Positive separation in layout units.
    """
    node_sizes = getattr(graph, "node_sizes", None)
    if node_sizes is None or int(getattr(node_sizes, "numel", lambda: 0)()) == 0:
        return max(1.0, float(config.node_sep))
    sizes = node_sizes.detach().to(device="cpu", dtype=torch.float32)
    width = float(sizes[:, 0].max().item()) if sizes.ndim == 2 and sizes.shape[1] >= 1 else 1.0
    return max(float(config.node_sep), width * 1.05, 1.0)


def _num_layers(layer_assignments: torch.Tensor) -> int:
    """Return the number of layers in an assignment tensor.

    Parameters
    ----------
    layer_assignments : torch.Tensor
        Layer IDs with shape ``[N]``.

    Returns
    -------
    int
        Number of non-negative layers.
    """
    return int(layer_assignments.max().item()) + 1 if layer_assignments.numel() else 0


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
    requested: Optional[int]
    value = config.algorithm_params.get("scale_checkpoint_exit_after_level", None)
    requested = int(value) if value is not None else None
    if requested == int(level):
        import os

        os._exit(137)


__all__ = ["LayersScaleStrategy", "LayersTelemetry", "layout_layers"]
