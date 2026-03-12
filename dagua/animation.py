"""Post-hoc optimization animation for Dagua layouts.

Captures faithful optimization snapshots during layout/edge optimization, then
renders the full graph after the fact into GIF/video formats.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from dagua.config import LayoutConfig
from dagua.edges import BezierCurve, place_edge_labels, route_edges
from dagua.layout import layout
from dagua.render import render


_ANIM_RASTER_FORMATS = {"gif", "webp"}
_ANIM_VIDEO_FORMATS = {"mp4", "m4v", "mov", "avi"}


@dataclass
class AnimationConfig:
    """Animation output and capture controls."""

    fps: int = 20
    dpi: int = 140
    figsize: Optional[Tuple[float, float]] = None
    format: Optional[str] = None
    camera: str = "adaptive"  # adaptive, global, focus, bbox
    center_on: Optional[Any] = None
    bounds: Optional[Tuple[float, float, float, float]] = None
    follow_padding: float = 80.0
    hold_start_frames: int = 6
    hold_end_frames: int = 12
    transition_frames: int = 6
    show_phase_cards: bool = True
    max_layout_frames: int = 36
    max_edge_frames: int = 18
    overlay_title: bool = True
    output: Optional[str] = None
    save_frames_dir: Optional[str] = None
    codec: str = "libx264"
    bitrate: str = "6M"


@dataclass
class AnimationResult:
    """Summary of an exported animation."""

    output: Optional[str]
    format: str
    frame_count: int
    layout_snapshots: int
    edge_snapshots: int


@dataclass
class _Snapshot:
    kind: str
    phase: str
    step: int
    total_steps: int
    positions: Optional[torch.Tensor] = None
    endpoints: Optional[torch.Tensor] = None
    control_points: Optional[torch.Tensor] = None
    title: Optional[str] = None
    subtitle: Optional[str] = None


def _detect_animation_format(output: Optional[str], format: Optional[str]) -> str:
    if format is not None:
        return format.lower().lstrip(".")
    if output is None:
        return "gif"
    suffix = Path(output).suffix.lower().lstrip(".")
    return suffix or "gif"


def _step_interval(total_steps: int, max_frames: int) -> int:
    if total_steps <= 0:
        return 1
    return max(1, int(np.ceil(total_steps / max(max_frames, 1))))


class _TraceRecorder:
    """Collects faithful snapshots during optimization with bounded overhead."""

    def __init__(self, graph, config: AnimationConfig):
        self.graph = graph
        self.config = config
        self.snapshots: List[_Snapshot] = []
        self._layout_interval = 1
        self._edge_interval = 1

    def mark_phase(self, title: str, subtitle: Optional[str] = None) -> None:
        if not self.config.show_phase_cards:
            return
        self.snapshots.append(
            _Snapshot(kind="card", phase="card", step=0, total_steps=0, title=title, subtitle=subtitle)
        )

    def capture_layout_positions(
        self,
        phase: str,
        step: int,
        total_steps: int,
        positions: torch.Tensor,
    ) -> None:
        if phase == "initialization":
            self._layout_interval = _step_interval(total_steps, self.config.max_layout_frames)
        if not self._should_keep(step, total_steps, self._layout_interval):
            return
        self.snapshots.append(
            _Snapshot(
                kind="layout",
                phase=phase,
                step=step,
                total_steps=total_steps,
                positions=positions.detach().cpu().clone(),
            )
        )

    def capture_edge_controls(
        self,
        phase: str,
        step: int,
        total_steps: int,
        positions: torch.Tensor,
        endpoints: torch.Tensor,
        control_points: torch.Tensor,
    ) -> None:
        self._edge_interval = _step_interval(total_steps, self.config.max_edge_frames)
        if not self._should_keep(step, total_steps, self._edge_interval):
            return
        self.snapshots.append(
            _Snapshot(
                kind="edge",
                phase=phase,
                step=step,
                total_steps=total_steps,
                positions=positions.detach().cpu().clone(),
                endpoints=endpoints.detach().cpu().clone(),
                control_points=control_points.detach().cpu().clone(),
            )
        )

    @staticmethod
    def _should_keep(step: int, total_steps: int, interval: int) -> bool:
        if step in (0, total_steps):
            return True
        return step % interval == 0


def _resolve_focus_index(graph, center_on: Any) -> Optional[int]:
    if center_on is None:
        return None
    if isinstance(center_on, int):
        return center_on if 0 <= center_on < graph.num_nodes else None
    return graph._id_to_index.get(center_on)


def _snapshot_bounds(graph, positions: torch.Tensor) -> Tuple[float, float, float, float]:
    graph.compute_node_sizes()
    pos = positions.detach().cpu().numpy()
    sizes = graph.node_sizes.detach().cpu().numpy()
    margin = graph.graph_style.margin + 12.0
    x_min = float((pos[:, 0] - sizes[:, 0] / 2).min() - margin)
    x_max = float((pos[:, 0] + sizes[:, 0] / 2).max() + margin)
    y_min = float((pos[:, 1] - sizes[:, 1] / 2).min() - margin)
    y_max = float((pos[:, 1] + sizes[:, 1] / 2).max() + margin)
    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0
    return x_min, x_max, y_min, y_max


def _compute_camera_bounds(
    graph,
    snapshots: Sequence[_Snapshot],
    config: AnimationConfig,
) -> List[Tuple[float, float, float, float]]:
    bounds_per_frame: List[Tuple[float, float, float, float]] = []
    real_snaps = [s for s in snapshots if s.positions is not None]
    if not real_snaps:
        return bounds_per_frame

    raw_bounds = [_snapshot_bounds(graph, snap.positions) for snap in real_snaps]
    global_bounds = (
        min(b[0] for b in raw_bounds),
        max(b[1] for b in raw_bounds),
        min(b[2] for b in raw_bounds),
        max(b[3] for b in raw_bounds),
    )
    global_w = global_bounds[1] - global_bounds[0]
    global_h = global_bounds[3] - global_bounds[2]
    min_w = max(global_w * 0.35, 80.0)
    min_h = max(global_h * 0.35, 80.0)

    focus_idx = _resolve_focus_index(graph, config.center_on)
    smoothed = raw_bounds[0]
    raw_idx = 0

    for snap in snapshots:
        if snap.positions is None:
            bounds_per_frame.append(smoothed)
            continue
        current_raw = raw_bounds[raw_idx]
        raw_idx += 1
        if config.bounds is not None:
            target = config.bounds
        elif config.camera == "bbox":
            target = current_raw
        elif config.camera == "global":
            target = global_bounds
        elif config.camera == "focus" and focus_idx is not None:
            pos = snap.positions[focus_idx].detach().cpu().numpy()
            half_w = max(min_w * 0.5, config.follow_padding)
            half_h = max(min_h * 0.5, config.follow_padding)
            target = (float(pos[0] - half_w), float(pos[0] + half_w), float(pos[1] - half_h), float(pos[1] + half_h))
        else:
            target = current_raw

        if config.camera == "adaptive":
            alpha = 0.18
            smoothed = tuple((1 - alpha) * smoothed[i] + alpha * target[i] for i in range(4))
        else:
            smoothed = target

        cx = (smoothed[0] + smoothed[1]) / 2
        cy = (smoothed[2] + smoothed[3]) / 2
        w = max(smoothed[1] - smoothed[0], min_w)
        h = max(smoothed[3] - smoothed[2], min_h)
        smoothed = (cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2)
        bounds_per_frame.append(smoothed)

    return bounds_per_frame


def _default_animation_figsize(
    graph,
    bounds_per_frame: Sequence[Tuple[float, float, float, float]],
) -> Tuple[float, float]:
    gs = graph.graph_style
    if not bounds_per_frame:
        return (6.0, 4.0)
    x_min = min(b[0] for b in bounds_per_frame)
    x_max = max(b[1] for b in bounds_per_frame)
    y_min = min(b[2] for b in bounds_per_frame)
    y_max = max(b[3] for b in bounds_per_frame)
    width = x_max - x_min
    height = y_max - y_min
    max_w, max_h = gs.max_figsize
    min_w, min_h = gs.min_figsize
    scale = max(1.0, min(width / 100, max_w))
    aspect = height / max(width, 1.0)
    fig_w = min(max(scale, min_w), max_w)
    fig_h = min(max(fig_w * aspect, min_h), max_h)
    return (fig_w, fig_h)


def _curves_from_snapshot(snap: _Snapshot) -> List[BezierCurve]:
    assert snap.endpoints is not None
    assert snap.control_points is not None
    curves: List[BezierCurve] = []
    endpoints = snap.endpoints
    cp = snap.control_points
    for i in range(endpoints.shape[0]):
        curves.append(
            BezierCurve(
                p0=(endpoints[i, 0, 0].item(), endpoints[i, 0, 1].item()),
                cp1=(cp[i, 0, 0].item(), cp[i, 0, 1].item()),
                cp2=(cp[i, 1, 0].item(), cp[i, 1, 1].item()),
                p1=(endpoints[i, 1, 0].item(), endpoints[i, 1, 1].item()),
            )
        )
    return curves


def _overlay_text(ax, graph, title: str, subtitle: Optional[str]) -> None:
    gs = graph.graph_style
    x = 0.015
    y = 0.985
    ax.text(
        x, y, title,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=gs.title_font_size + 1,
        fontweight="bold",
        color=gs.title_font_color,
        fontfamily=gs.title_font_family or "sans-serif",
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "#FFFFFFE6",
            "edgecolor": "#D9D9D9",
            "linewidth": 0.6,
        },
        zorder=10,
    )
    if subtitle:
        ax.text(
            x, y - 0.055, subtitle,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=max(gs.title_font_size - 1, 8),
            color="#5F6368",
            fontfamily=gs.title_font_family or "sans-serif",
            zorder=10,
        )


def _frame_title(snap: _Snapshot) -> Tuple[str, Optional[str]]:
    if snap.kind == "card":
        return snap.title or "Optimization", snap.subtitle
    if snap.phase == "initialization":
        return "Initial Placement", "Topology-aware warm start"
    if snap.phase == "node_optimization":
        pct = 100 * snap.step / max(snap.total_steps, 1)
        return "Node Optimization", f"step {snap.step}/{snap.total_steps} ({pct:.0f}%)"
    if snap.phase == "final_projection":
        return "Final Projection", "Resolving residual overlaps"
    if snap.phase == "heuristic_routing":
        return "Edge Routing", "Initial routed curves"
    if snap.phase == "edge_optimization":
        pct = 100 * snap.step / max(snap.total_steps, 1)
        return "Edge Refinement", f"step {snap.step}/{snap.total_steps} ({pct:.0f}%)"
    return "Optimization", None


def _card_frame(graph, title: str, subtitle: Optional[str], figsize, dpi: int) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gs = graph.graph_style
    fig, ax = plt.subplots(1, 1, figsize=figsize or (7.0, 4.8), dpi=dpi)
    fig.patch.set_facecolor(gs.background_color)
    ax.set_facecolor(gs.background_color)
    ax.axis("off")
    ax.text(
        0.5, 0.58, title,
        ha="center", va="center",
        transform=ax.transAxes,
        fontsize=22,
        fontweight="bold",
        color=gs.title_font_color,
        fontfamily=gs.title_font_family or "sans-serif",
    )
    if subtitle:
        ax.text(
            0.5, 0.42, subtitle,
            ha="center", va="center",
            transform=ax.transAxes,
            fontsize=11,
            color="#6B7280",
            fontfamily=gs.title_font_family or "sans-serif",
        )
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def _render_snapshot_frame(
    graph,
    snap: _Snapshot,
    camera_bounds: Tuple[float, float, float, float],
    config: LayoutConfig,
    anim_cfg: AnimationConfig,
) -> np.ndarray:
    import matplotlib.pyplot as plt

    assert snap.positions is not None
    positions = snap.positions
    if snap.kind == "edge":
        curves = _curves_from_snapshot(snap)
    else:
        curves = route_edges(positions, graph.edge_index, graph.node_sizes, graph.direction, graph)
    label_positions = place_edge_labels(curves, positions, graph.node_sizes, graph.edge_labels, graph)

    fig, ax = render(
        graph,
        positions,
        config=config,
        curves=curves,
        label_positions=label_positions,
        figsize=anim_cfg.figsize,
        dpi=anim_cfg.dpi,
    )
    fig.set_size_inches(*anim_cfg.figsize)
    fig.set_dpi(anim_cfg.dpi)
    ax.set_xlim(camera_bounds[0], camera_bounds[1])
    ax.set_ylim(camera_bounds[2], camera_bounds[3])
    if anim_cfg.overlay_title:
        title, subtitle = _frame_title(snap)
        _overlay_text(ax, graph, title, subtitle)
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def _blend_frames(a: np.ndarray, b: np.ndarray, n: int) -> List[np.ndarray]:
    if n <= 0:
        return []
    blended = []
    for i in range(1, n + 1):
        alpha = i / (n + 1)
        mixed = np.clip((1 - alpha) * a + alpha * b, 0, 255).astype(np.uint8)
        blended.append(mixed)
    return blended


def _write_frames(frames: Sequence[np.ndarray], output: str, fmt: str, config: AnimationConfig) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)

    if config.save_frames_dir:
        frames_dir = Path(config.save_frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            from PIL import Image
            Image.fromarray(frame).save(frames_dir / f"frame_{i:04d}.png")

    if fmt in _ANIM_RASTER_FORMATS:
        from PIL import Image
        images = [Image.fromarray(frame) for frame in frames]
        duration = max(int(1000 / max(config.fps, 1)), 1)
        save_kwargs: Dict[str, Any] = {
            "save_all": True,
            "append_images": images[1:],
            "duration": duration,
            "loop": 0,
            "optimize": False,
        }
        if fmt == "webp":
            save_kwargs.update(lossless=True, quality=90, method=6)
        images[0].save(path, format=fmt.upper(), **save_kwargs)
        return

    if fmt in _ANIM_VIDEO_FORMATS:
        import imageio.v2 as imageio
        writer = imageio.get_writer(
            path,
            fps=config.fps,
            codec=config.codec,
            bitrate=config.bitrate,
            macro_block_size=None,
        )
        try:
            for frame in frames:
                writer.append_data(frame)
        finally:
            writer.close()
        return

    raise ValueError(
        f"Unsupported animation output format: {fmt!r}. "
        "Supported formats include GIF, WebP, MP4, MOV, M4V, and AVI."
    )


def animate(
    graph,
    config: Optional[LayoutConfig] = None,
    output: Optional[str] = None,
    animation_config: Optional[AnimationConfig] = None,
    **kwargs,
) -> AnimationResult:
    """Render a faithful optimization animation after layout completes."""
    if config is None:
        from dagua.defaults import get_default_device, get_default_layout_overrides
        layout_overrides = get_default_layout_overrides()
        config = LayoutConfig(device=get_default_device(), **layout_overrides)

    anim_cfg = animation_config or AnimationConfig()
    if output is not None:
        anim_cfg.output = output
    for key, value in kwargs.items():
        if hasattr(anim_cfg, key):
            setattr(anim_cfg, key, value)

    fmt = _detect_animation_format(anim_cfg.output, anim_cfg.format)
    recorder = _TraceRecorder(graph, anim_cfg)
    final_positions = layout(graph, config, trace=recorder)

    graph.compute_node_sizes()
    curves = route_edges(final_positions, graph.edge_index, graph.node_sizes, graph.direction, graph)
    if getattr(config, "edge_opt_steps", 0) >= 0:
        from dagua.layout.edge_optimization import optimize_edges
        curves = optimize_edges(
            curves,
            final_positions,
            graph.edge_index,
            graph.node_sizes,
            config,
            graph,
            trace=recorder,
        )
    recorder.mark_phase("Final Render", "Labels and styling settle into place")

    final_endpoints = torch.zeros(len(curves), 2, 2)
    final_cp = torch.zeros(len(curves), 2, 2)
    for i, curve in enumerate(curves):
        final_endpoints[i, 0] = torch.tensor(curve.p0)
        final_endpoints[i, 1] = torch.tensor(curve.p1)
        final_cp[i, 0] = torch.tensor(curve.cp1)
        final_cp[i, 1] = torch.tensor(curve.cp2)
    recorder.capture_edge_controls(
        phase="edge_optimization",
        step=max(getattr(config, "edge_opt_steps", 0), 1),
        total_steps=max(getattr(config, "edge_opt_steps", 0), 1),
        positions=final_positions,
        endpoints=final_endpoints,
        control_points=final_cp,
    )

    if anim_cfg.output is None:
        anim_cfg.output = str(Path("dagua_optimization.gif").resolve())

    bounds_per_frame = _compute_camera_bounds(graph, recorder.snapshots, anim_cfg)
    if anim_cfg.figsize is None:
        anim_cfg.figsize = _default_animation_figsize(graph, bounds_per_frame)
    frames: List[np.ndarray] = []
    bounds_iter = iter(bounds_per_frame)
    last_visual: Optional[np.ndarray] = None

    for snap in recorder.snapshots:
        if snap.kind == "card":
            frame = _card_frame(graph, snap.title or "Optimization", snap.subtitle, anim_cfg.figsize, anim_cfg.dpi)
        else:
            frame = _render_snapshot_frame(graph, snap, next(bounds_iter), config, anim_cfg)

        if last_visual is not None and anim_cfg.transition_frames > 0:
            frames.extend(_blend_frames(last_visual, frame, anim_cfg.transition_frames))
        frames.append(frame)
        last_visual = frame

    if frames:
        frames = [frames[0]] * anim_cfg.hold_start_frames + frames + [frames[-1]] * anim_cfg.hold_end_frames
    _write_frames(frames, anim_cfg.output, fmt, anim_cfg)

    layout_count = sum(1 for snap in recorder.snapshots if snap.kind == "layout")
    edge_count = sum(1 for snap in recorder.snapshots if snap.kind == "edge")
    return AnimationResult(
        output=anim_cfg.output,
        format=fmt,
        frame_count=len(frames),
        layout_snapshots=layout_count,
        edge_snapshots=edge_count,
    )
