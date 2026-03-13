"""Post-hoc optimization animation for Dagua layouts.

Captures faithful optimization snapshots during layout/edge optimization, then
renders the full graph after the fact into GIF/video formats.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import torch

from dagua.config import LayoutConfig
from dagua.edges import BezierCurve, place_edge_labels, route_edges
from dagua.layout import layout
from dagua.render import render


_ANIM_RASTER_FORMATS = {"gif", "webp"}
_ANIM_VIDEO_FORMATS = {"mp4", "m4v", "mov", "avi"}
_STILL_RASTER_FORMATS = {"png", "jpg", "jpeg", "webp", "tiff", "bmp"}
_STILL_VECTOR_FORMATS = {"pdf", "svg", "eps", "ps"}


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
class PosterConfig:
    """Single-frame showcase export using the cinematic tour camera logic."""

    dpi: int = 220
    figsize: Optional[Tuple[float, float]] = None
    format: Optional[str] = None
    output: Optional[str] = None
    scene: str = "auto"
    keyframes: Optional[List["CameraKeyframe"]] = None
    keyframe_index: Optional[int] = None
    show_titles: bool = True
    follow_padding: float = 80.0
    lod_threshold: int = 100_000
    detail_node_limit: int = 12_000
    label_node_limit: int = 160
    edge_sample_limit: int = 60_000
    density_bins: int = 280
    density_gamma: float = 0.52


@dataclass
class PosterResult:
    """Summary of an exported still render."""

    output: Optional[str]
    format: str
    bounds: Tuple[float, float, float, float]
    used_large_lod: bool


@dataclass
class CameraKeyframe:
    """One camera waypoint for a tour animation."""

    duration_frames: int = 24
    easing: str = "ease_in_out"  # linear, ease_in, ease_out, ease_in_out
    center_on: Optional[Any] = None
    center: Optional[Tuple[float, float]] = None
    bounds: Optional[Tuple[float, float, float, float]] = None
    scale: Optional[float] = None  # relative to the target bounds width/height
    title: Optional[str] = None
    subtitle: Optional[str] = None


@dataclass
class TourConfig:
    """Camera tour controls for final/current graph showcases."""

    fps: int = 24
    dpi: int = 160
    figsize: Optional[Tuple[float, float]] = None
    format: Optional[str] = None
    output: Optional[str] = None
    scene: str = "auto"  # auto, powers_of_ten, zoom_pan, panorama, layer_sweep, cathedral, motif_orbit, keyframes
    keyframes: Optional[List[CameraKeyframe]] = None
    hold_start_frames: int = 10
    hold_end_frames: int = 14
    show_titles: bool = True
    follow_padding: float = 80.0
    codec: str = "libx264"
    bitrate: str = "8M"
    save_frames_dir: Optional[str] = None
    lod_threshold: int = 100_000
    detail_node_limit: int = 8_000
    label_node_limit: int = 120
    edge_sample_limit: int = 40_000
    density_bins: int = 220
    density_gamma: float = 0.55


@dataclass
class _LargeTourState:
    pos: np.ndarray
    sizes: np.ndarray
    sampled_edges: Optional[np.ndarray]


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


def _detect_still_format(output: Optional[str], format: Optional[str]) -> str:
    if format is not None:
        return format.lower().lstrip(".")
    if output is None:
        return "png"
    suffix = Path(output).suffix.lower().lstrip(".")
    return suffix or "png"


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

    raw_bounds = [_snapshot_bounds(graph, cast(torch.Tensor, snap.positions)) for snap in real_snaps]
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
            smoothed = cast(
                Tuple[float, float, float, float],
                tuple((1 - alpha) * smoothed[i] + alpha * target[i] for i in range(4)),
            )
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


def _prepare_large_tour_state(
    graph,
    positions: torch.Tensor,
    config: TourConfig,
) -> _LargeTourState:
    pos = positions.detach().cpu().numpy()
    graph.compute_node_sizes()
    sizes = graph.node_sizes.detach().cpu().numpy()
    sampled_edges = None
    if graph.edge_index.numel() > 0 and config.edge_sample_limit > 0:
        ei = graph.edge_index.detach().cpu()
        edge_count = ei.shape[1]
        sample_n = min(edge_count, config.edge_sample_limit)
        sample_idx = torch.randint(0, edge_count, (sample_n,), dtype=torch.long)
        sampled_edges = ei[:, sample_idx].numpy().T
    return _LargeTourState(pos=pos, sizes=sizes, sampled_edges=sampled_edges)


def _global_bounds(graph, positions: torch.Tensor) -> Tuple[float, float, float, float]:
    return _snapshot_bounds(graph, positions)


def _ease_value(t: float, easing: str) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    if easing == "linear":
        return t
    if easing == "ease_in":
        return t * t
    if easing == "ease_out":
        return 1 - (1 - t) * (1 - t)
    return t * t * (3 - 2 * t)


def _interesting_node_indices(graph, positions: torch.Tensor, k: int = 4) -> List[int]:
    n = graph.num_nodes
    if n == 0:
        return []
    pos = positions.detach().cpu()
    ei = graph.edge_index.detach().cpu() if graph.edge_index.numel() > 0 else torch.zeros(2, 0, dtype=torch.long)
    degree = torch.zeros(n, dtype=torch.float32)
    long_span = torch.zeros(n, dtype=torch.float32)
    cluster_bonus = torch.zeros(n, dtype=torch.float32)
    if ei.numel() > 0:
        ones = torch.ones(ei.shape[1], dtype=torch.float32)
        degree.scatter_add_(0, ei[0], ones)
        degree.scatter_add_(0, ei[1], ones)
        edge_lengths = torch.norm(pos[ei[1]] - pos[ei[0]], dim=1)
        if edge_lengths.numel() > 0:
            threshold = torch.quantile(edge_lengths, 0.85)
            mask = edge_lengths >= threshold
            if mask.any():
                bonus = edge_lengths[mask] / max(edge_lengths.median().item(), 1.0)
                long_span.scatter_add_(0, ei[0][mask], bonus)
                long_span.scatter_add_(0, ei[1][mask], bonus)
    if graph.clusters:
        for name, members in graph.clusters.items():
            depth = graph.cluster_depth(name) + 1
            member_ids = graph.leaf_cluster_members(name)
            if member_ids:
                idx = torch.tensor(member_ids, dtype=torch.long)
                cluster_bonus[idx] = torch.maximum(cluster_bonus[idx], torch.full_like(cluster_bonus[idx], float(depth)))
    center = pos.mean(dim=0, keepdim=True)
    dist = torch.norm(pos - center, dim=1)
    score = degree + 0.8 * long_span + 1.5 * cluster_bonus + 0.02 * dist
    top = torch.argsort(score, descending=True)[: max(k * 3, k)].tolist()

    picked: List[int] = []
    for idx in top:
        if not picked:
            picked.append(idx)
            continue
        if all(torch.norm(pos[idx] - pos[prev]).item() > 0.12 * max(dist.max().item(), 1.0) for prev in picked):
            picked.append(idx)
        if len(picked) >= k:
            break
    return picked[:k] if picked else [int(torch.argmax(score).item())]


def _focus_bounds_for_node(
    graph,
    positions: torch.Tensor,
    node_idx: int,
    padding: float,
    min_fraction: float = 0.14,
) -> Tuple[float, float, float, float]:
    global_bounds = _global_bounds(graph, positions)
    global_w = global_bounds[1] - global_bounds[0]
    global_h = global_bounds[3] - global_bounds[2]
    graph.compute_node_sizes()
    pos = positions.detach().cpu()
    sizes = graph.node_sizes.detach().cpu()
    cx = float(pos[node_idx, 0].item())
    cy = float(pos[node_idx, 1].item())
    half_w = max(float(sizes[node_idx, 0].item()) * 1.2, global_w * min_fraction, padding)
    half_h = max(float(sizes[node_idx, 1].item()) * 1.2, global_h * min_fraction, padding)
    return (cx - half_w, cx + half_w, cy - half_h, cy + half_h)


def _interpolate_bounds(
    start: Tuple[float, float, float, float],
    end: Tuple[float, float, float, float],
    duration_frames: int,
    easing: str,
) -> List[Tuple[float, float, float, float]]:
    frames: List[Tuple[float, float, float, float]] = []
    for i in range(max(duration_frames, 1)):
        t = 1.0 if duration_frames <= 1 else i / (duration_frames - 1)
        w = _ease_value(t, easing)
        frames.append(
            cast(
                Tuple[float, float, float, float],
                tuple((1 - w) * start[j] + w * end[j] for j in range(4)),
            )
        )
    return frames


def _scale_bounds(
    bounds: Tuple[float, float, float, float],
    scale: float,
) -> Tuple[float, float, float, float]:
    cx = (bounds[0] + bounds[1]) / 2
    cy = (bounds[2] + bounds[3]) / 2
    half_w = (bounds[1] - bounds[0]) * 0.5 * scale
    half_h = (bounds[3] - bounds[2]) * 0.5 * scale
    return (cx - half_w, cx + half_w, cy - half_h, cy + half_h)


def _keyframe_target_bounds(
    graph,
    positions: torch.Tensor,
    kf: CameraKeyframe,
    default_bounds: Tuple[float, float, float, float],
    padding: float,
) -> Tuple[float, float, float, float]:
    if kf.bounds is not None:
        bounds = kf.bounds
    elif kf.center is not None:
        cx, cy = kf.center
        global_w = default_bounds[1] - default_bounds[0]
        global_h = default_bounds[3] - default_bounds[2]
        half_w = max(global_w * 0.18, padding)
        half_h = max(global_h * 0.18, padding)
        bounds = (cx - half_w, cx + half_w, cy - half_h, cy + half_h)
    elif kf.center_on is not None:
        idx = _resolve_focus_index(graph, kf.center_on)
        if idx is None:
            bounds = default_bounds
        else:
            bounds = _focus_bounds_for_node(graph, positions, idx, padding)
    else:
        bounds = default_bounds
    if kf.scale is not None:
        bounds = _scale_bounds(bounds, kf.scale)
    return bounds


def _default_tour_keyframes(graph, positions: torch.Tensor, config: TourConfig) -> List[CameraKeyframe]:
    global_bounds = _global_bounds(graph, positions)
    interesting = _interesting_node_indices(graph, positions, k=4)
    focus_nodes = interesting[: max(1, min(3, len(interesting)))]
    primary_bounds = _focus_bounds_for_node(graph, positions, focus_nodes[0], config.follow_padding, min_fraction=0.09)
    if config.scene == "zoom_pan":
        x0, x1, y0, y1 = primary_bounds
        w = x1 - x0
        h = y1 - y0
        horizontal = (global_bounds[1] - global_bounds[0]) >= (global_bounds[3] - global_bounds[2])
        if horizontal:
            pan_a = (x0 - 0.35 * w, x0 + 0.65 * w, y0, y1)
            pan_b = (x1 - 0.65 * w, x1 + 0.35 * w, y0, y1)
        else:
            pan_a = (x0, x1, y0 - 0.35 * h, y0 + 0.65 * h)
            pan_b = (x0, x1, y1 - 0.65 * h, y1 + 0.35 * h)
        return [
            CameraKeyframe(duration_frames=26, bounds=global_bounds, title="Whole Graph", subtitle="Full composition"),
            CameraKeyframe(duration_frames=28, bounds=_scale_bounds(primary_bounds, 1.35), title="Zoom In", subtitle="Entering an interesting region"),
            CameraKeyframe(duration_frames=34, bounds=pan_a, title="Pan Across", subtitle="Following visible local structure"),
            CameraKeyframe(duration_frames=34, bounds=pan_b, title="Pan Across", subtitle="Continuing across the motif"),
            CameraKeyframe(duration_frames=26, bounds=_scale_bounds(global_bounds, 0.92), title="Back to Context", subtitle="Detail within the whole"),
        ]
    if config.scene == "powers_of_ten":
        return [
            CameraKeyframe(duration_frames=28, bounds=_scale_bounds(global_bounds, 1.0), title="Whole Graph", subtitle="Starting from the full structure"),
            CameraKeyframe(duration_frames=24, bounds=_scale_bounds(global_bounds, 0.55), center_on=focus_nodes[0], title="First Zoom", subtitle="Major structure comes forward"),
            CameraKeyframe(duration_frames=24, bounds=primary_bounds, title="Detail", subtitle="Local neighborhood"),
            CameraKeyframe(duration_frames=24, bounds=_scale_bounds(global_bounds, 0.8), title="Context Return", subtitle="Detail back into context"),
        ]
    if config.scene == "cathedral":
        x0, x1, y0, y1 = global_bounds
        if graph.direction in ("TB", "BT"):
            tall = max((y1 - y0) * 0.55, config.follow_padding * 3)
            nave = ((x0 + x1) * 0.5 - (x1 - x0) * 0.22, (x0 + x1) * 0.5 + (x1 - x0) * 0.22, y0, y0 + tall)
            transept = (x0, x1, (y0 + y1) * 0.5 - tall * 0.22, (y0 + y1) * 0.5 + tall * 0.22)
        else:
            wide = max((x1 - x0) * 0.55, config.follow_padding * 3)
            nave = (x0, x0 + wide, (y0 + y1) * 0.5 - (y1 - y0) * 0.22, (y0 + y1) * 0.5 + (y1 - y0) * 0.22)
            transept = ((x0 + x1) * 0.5 - wide * 0.22, (x0 + x1) * 0.5 + wide * 0.22, y0, y1)
        return [
            CameraKeyframe(duration_frames=30, bounds=_scale_bounds(global_bounds, 1.0), title="Cathedral View", subtitle="Large-scale flow and symmetry"),
            CameraKeyframe(duration_frames=36, bounds=nave, title="Central Spine", subtitle="The main structural axis"),
            CameraKeyframe(duration_frames=36, bounds=transept, title="Cross Structure", subtitle="Lateral connections and branches"),
            CameraKeyframe(duration_frames=26, bounds=_scale_bounds(global_bounds, 0.86), title="Whole Graph", subtitle="The full architecture again"),
        ]
    if config.scene == "motif_orbit":
        motifs = focus_nodes[:3] if len(focus_nodes) >= 3 else focus_nodes
        frames = [CameraKeyframe(duration_frames=24, bounds=global_bounds, title="Motif Orbit", subtitle="A tour of strong local structures")]
        subtitles = ["Cluster-rich region", "Long-span connection hub", "Branching motif"]
        for i, node_idx in enumerate(motifs):
            frames.append(
                CameraKeyframe(
                    duration_frames=24,
                    bounds=_scale_bounds(_focus_bounds_for_node(graph, positions, node_idx, config.follow_padding, min_fraction=0.09), 1.1),
                    title=f"Motif {i + 1}",
                    subtitle=subtitles[i] if i < len(subtitles) else "Interesting local structure",
                )
            )
        frames.append(CameraKeyframe(duration_frames=22, bounds=_scale_bounds(global_bounds, 0.95), title="Whole Graph", subtitle="Orbit complete"))
        return frames
    if config.scene == "panorama":
        return [
            CameraKeyframe(duration_frames=20, bounds=_scale_bounds(global_bounds, 0.85), title="Panorama", subtitle="Broad structural sweep"),
            CameraKeyframe(duration_frames=36, bounds=(global_bounds[0], (global_bounds[0] + global_bounds[1]) / 2, global_bounds[2], global_bounds[3]), title="Left Half", subtitle="One side of the graph"),
            CameraKeyframe(duration_frames=36, bounds=((global_bounds[0] + global_bounds[1]) / 2, global_bounds[1], global_bounds[2], global_bounds[3]), title="Right Half", subtitle="Counterpart structures"),
            CameraKeyframe(duration_frames=24, bounds=_scale_bounds(global_bounds, 1.0), title="Whole Graph", subtitle="Back to the whole composition"),
        ]
    if config.scene == "layer_sweep":
        x0, x1, y0, y1 = global_bounds
        if graph.direction in ("TB", "BT"):
            h = max((y1 - y0) * 0.35, config.follow_padding * 2)
            return [
                CameraKeyframe(duration_frames=20, bounds=global_bounds, title="Layer Sweep", subtitle="Scanning along the graph flow"),
                CameraKeyframe(duration_frames=36, bounds=(x0, x1, y0, y0 + h), title="Early Layers", subtitle="Inputs and first branches"),
                CameraKeyframe(duration_frames=36, bounds=(x0, x1, (y0 + y1 - h) / 2, (y0 + y1 + h) / 2), title="Middle Layers", subtitle="Cross-links and merges"),
                CameraKeyframe(duration_frames=36, bounds=(x0, x1, y1 - h, y1), title="Late Layers", subtitle="Outputs and terminal structure"),
            ]
        w = max((x1 - x0) * 0.35, config.follow_padding * 2)
        return [
            CameraKeyframe(duration_frames=20, bounds=global_bounds, title="Layer Sweep", subtitle="Scanning along the graph flow"),
            CameraKeyframe(duration_frames=36, bounds=(x0, x0 + w, y0, y1), title="Early Layers", subtitle="Inputs and first branches"),
            CameraKeyframe(duration_frames=36, bounds=((x0 + x1 - w) / 2, (x0 + x1 + w) / 2, y0, y1), title="Middle Layers", subtitle="Cross-links and merges"),
            CameraKeyframe(duration_frames=36, bounds=(x1 - w, x1, y0, y1), title="Late Layers", subtitle="Outputs and terminal structure"),
        ]

    # auto and focus_hops default
    keyframes = [
        CameraKeyframe(duration_frames=24, bounds=global_bounds, title="Whole Graph", subtitle="Overall topology and balance"),
    ]
    subtitles = ["High-degree junction", "Secondary hub", "Tertiary motif"]
    for idx, node_idx in enumerate(focus_nodes):
        keyframes.append(
            CameraKeyframe(
                duration_frames=28,
                center_on=node_idx,
                scale=0.9 if idx == 0 else 1.0,
                title=f"Focus {idx + 1}",
                subtitle=subtitles[idx] if idx < len(subtitles) else "Interesting structure",
            )
        )
    keyframes.append(
        CameraKeyframe(duration_frames=28, bounds=_scale_bounds(global_bounds, 0.92), title="Whole Graph", subtitle="Return to full context")
    )
    return keyframes


def _tour_keyframes(
    graph,
    positions: torch.Tensor,
    config: TourConfig,
) -> List[CameraKeyframe]:
    if config.keyframes:
        return config.keyframes
    return _default_tour_keyframes(graph, positions, config)


def _poster_keyframe(
    graph,
    positions: torch.Tensor,
    config: PosterConfig,
) -> CameraKeyframe:
    keyframes = _tour_keyframes(
        graph,
        positions,
        TourConfig(
            dpi=config.dpi,
            figsize=config.figsize,
            format=config.format,
            output=config.output,
            scene=config.scene,
            keyframes=config.keyframes,
            show_titles=config.show_titles,
            follow_padding=config.follow_padding,
            lod_threshold=config.lod_threshold,
            detail_node_limit=config.detail_node_limit,
            label_node_limit=config.label_node_limit,
            edge_sample_limit=config.edge_sample_limit,
            density_bins=config.density_bins,
            density_gamma=config.density_gamma,
        ),
    )
    if config.keyframe_index is not None:
        idx = max(0, min(config.keyframe_index, len(keyframes) - 1))
        return keyframes[idx]

    preferred_idx = {
        "auto": 1,
        "powers_of_ten": 2,
        "zoom_pan": 3,
        "panorama": 1,
        "layer_sweep": 1,
        "cathedral": 1,
        "motif_orbit": 1,
        "keyframes": 0,
    }.get(config.scene, 0)
    idx = max(0, min(preferred_idx, len(keyframes) - 1))
    return keyframes[idx]


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
    canvas = cast(Any, fig.canvas)
    canvas.draw()
    frame = np.asarray(canvas.buffer_rgba())[..., :3].copy()
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
    canvas = cast(Any, fig.canvas)
    canvas.draw()
    frame = np.asarray(canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def _render_tour_frame(
    graph,
    positions: torch.Tensor,
    curves: List[BezierCurve],
    label_positions,
    camera_bounds: Tuple[float, float, float, float],
    config: LayoutConfig,
    tour_cfg: TourConfig,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
) -> np.ndarray:
    import matplotlib.pyplot as plt

    fig, ax = render(
        graph,
        positions,
        config=config,
        curves=curves,
        label_positions=label_positions,
        figsize=tour_cfg.figsize,
        dpi=tour_cfg.dpi,
    )
    fig.set_size_inches(*tour_cfg.figsize)
    fig.set_dpi(tour_cfg.dpi)
    ax.set_xlim(camera_bounds[0], camera_bounds[1])
    ax.set_ylim(camera_bounds[2], camera_bounds[3])
    if tour_cfg.show_titles and title:
        _overlay_text(ax, graph, title, subtitle)
    canvas = cast(Any, fig.canvas)
    canvas.draw()
    frame = np.asarray(canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def _render_large_tour_frame(
    graph,
    large_state: _LargeTourState,
    camera_bounds: Tuple[float, float, float, float],
    tour_cfg: TourConfig,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap

    x_min, x_max, y_min, y_max = camera_bounds
    pos = large_state.pos
    visible = (
        (pos[:, 0] >= x_min) & (pos[:, 0] <= x_max) &
        (pos[:, 1] >= y_min) & (pos[:, 1] <= y_max)
    )
    visible_count = int(visible.sum())

    fig, ax = plt.subplots(1, 1, figsize=tour_cfg.figsize, dpi=tour_cfg.dpi)
    bg = graph.graph_style.background_color
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    view_x = pos[visible, 0] if visible_count > 0 else np.array([], dtype=np.float32)
    view_y = pos[visible, 1] if visible_count > 0 else np.array([], dtype=np.float32)

    if visible_count > 0:
        bins = int(np.clip(np.sqrt(max(visible_count, 1)) * 1.25, 48, tour_cfg.density_bins))
        hist, xedges, yedges = np.histogram2d(
            view_x, view_y,
            bins=[bins, bins],
            range=[[x_min, x_max], [y_min, y_max]],
        )
        if hist.max() > 0:
            hist = np.power(hist / hist.max(), tour_cfg.density_gamma)
            cmap = LinearSegmentedColormap.from_list(
                "dagua_density",
                [bg, "#d6ecf7", "#77b9da", "#2f7ea8", "#16516e"],
            )
            ax.imshow(
                hist.T,
                origin="lower",
                extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]),
                cmap=cmap,
                interpolation="bilinear",
                aspect="auto",
                alpha=0.95,
                zorder=1,
            )

    if large_state.sampled_edges is not None:
        sampled = large_state.sampled_edges
        src = sampled[:, 0]
        tgt = sampled[:, 1]
        src_xy = pos[src]
        tgt_xy = pos[tgt]
        edge_mask = (
            (src_xy[:, 0] >= x_min) & (src_xy[:, 0] <= x_max) &
            (src_xy[:, 1] >= y_min) & (src_xy[:, 1] <= y_max) &
            (tgt_xy[:, 0] >= x_min) & (tgt_xy[:, 0] <= x_max) &
            (tgt_xy[:, 1] >= y_min) & (tgt_xy[:, 1] <= y_max)
        )
        if edge_mask.any():
            segs = np.stack([src_xy[edge_mask], tgt_xy[edge_mask]], axis=1)
            if len(segs) > 6000:
                step = max(len(segs) // 6000, 1)
                segs = segs[::step]
            lc = LineCollection(cast(Any, segs), colors="#266b8ccc", linewidths=0.35, alpha=0.16, zorder=2)
            ax.add_collection(lc)

    if 0 < visible_count <= tour_cfg.detail_node_limit:
        visible_idx = np.nonzero(visible)[0]
        sizes = np.clip(600.0 / np.sqrt(max(visible_count, 1)), 2.0, 18.0)
        ax.scatter(
            pos[visible_idx, 0],
            pos[visible_idx, 1],
            s=sizes,
            c="#1d6a8a",
            alpha=0.7,
            linewidths=0.0,
            zorder=3,
        )
        if visible_count <= tour_cfg.label_node_limit:
            for idx in visible_idx[: tour_cfg.label_node_limit]:
                ax.text(
                    pos[idx, 0],
                    pos[idx, 1],
                    graph.node_labels[idx],
                    fontsize=7.0,
                    color="#243238",
                    ha="center",
                    va="center",
                    zorder=4,
                )
    elif visible_count > 0:
        sample_n = min(2500, visible_count)
        sampled_idx = np.linspace(0, visible_count - 1, sample_n, dtype=int)
        vx = view_x[sampled_idx]
        vy = view_y[sampled_idx]
        ax.scatter(vx, vy, s=1.0, c="#114b65", alpha=0.18, linewidths=0.0, zorder=3)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.axis("off")
    if tour_cfg.show_titles and title:
        sub = subtitle
        if visible_count > tour_cfg.detail_node_limit:
            trailer = f"{visible_count:,} visible nodes"
            sub = f"{subtitle} · {trailer}" if subtitle else trailer
        _overlay_text(ax, graph, title, sub)

    canvas = cast(Any, fig.canvas)
    canvas.draw()
    frame = np.asarray(canvas.buffer_rgba())[..., :3].copy()
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


def _write_still_image(frame: np.ndarray, output: str, fmt: str, dpi: int) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt in _STILL_RASTER_FORMATS:
        from PIL import Image

        save_kwargs: Dict[str, Any] = {}
        if fmt in {"jpg", "jpeg"}:
            save_kwargs["quality"] = 95
            save_kwargs["subsampling"] = 0
            pil_format = "JPEG"
        elif fmt == "webp":
            save_kwargs["quality"] = 95
            save_kwargs["method"] = 6
            pil_format = "WEBP"
        else:
            pil_format = fmt.upper()
        Image.fromarray(frame).save(path, format=pil_format, **save_kwargs)
        return

    if fmt in _STILL_VECTOR_FORMATS:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        h, w = frame.shape[:2]
        fig, ax = plt.subplots(1, 1, figsize=(w / dpi, h / dpi), dpi=dpi)
        ax.imshow(frame)
        ax.axis("off")
        fig.subplots_adjust(0, 0, 1, 1)
        fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight", pad_inches=0.0, facecolor="white")
        plt.close(fig)
        return

    raise ValueError(f"Unsupported still format: {fmt!r}.")


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


def tour(
    graph,
    positions: Optional[torch.Tensor] = None,
    config: Optional[LayoutConfig] = None,
    output: Optional[str] = None,
    tour_config: Optional[TourConfig] = None,
    **kwargs,
) -> AnimationResult:
    """Render a cinematic tour of the final or current graph layout."""
    if config is None:
        from dagua.defaults import get_default_device, get_default_layout_overrides
        layout_overrides = get_default_layout_overrides()
        config = LayoutConfig(device=get_default_device(), **layout_overrides)

    tour_cfg = tour_config or TourConfig()
    if output is not None:
        tour_cfg.output = output
    for key, value in kwargs.items():
        if hasattr(tour_cfg, key):
            setattr(tour_cfg, key, value)

    if positions is None:
        positions = layout(graph, config)

    graph.compute_node_sizes()
    use_large_lod = graph.num_nodes >= tour_cfg.lod_threshold
    curves = None
    label_positions = None
    large_state = None
    if use_large_lod:
        large_state = _prepare_large_tour_state(graph, positions, tour_cfg)
    else:
        curves = route_edges(positions, graph.edge_index, graph.node_sizes, graph.direction, graph)
        if getattr(config, "edge_opt_steps", 0) >= 0:
            from dagua.layout.edge_optimization import optimize_edges
            curves = optimize_edges(curves, positions, graph.edge_index, graph.node_sizes, config, graph)
        label_positions = place_edge_labels(curves, positions, graph.node_sizes, graph.edge_labels, graph)

    if tour_cfg.output is None:
        tour_cfg.output = str(Path("dagua_tour.mp4").resolve())
    fmt = _detect_animation_format(tour_cfg.output, tour_cfg.format)

    keyframes = _tour_keyframes(graph, positions, tour_cfg)
    global_bounds = _global_bounds(graph, positions)
    realized: List[Tuple[CameraKeyframe, Tuple[float, float, float, float]]] = []
    for kf in keyframes:
        realized.append((kf, _keyframe_target_bounds(graph, positions, kf, global_bounds, tour_cfg.follow_padding)))

    if tour_cfg.figsize is None:
        tour_cfg.figsize = _default_animation_figsize(graph, [b for _, b in realized])

    frames: List[np.ndarray] = []
    for idx, (kf, bounds) in enumerate(realized):
        start_bounds = bounds if idx == 0 else realized[idx - 1][1]
        path = _interpolate_bounds(start_bounds, bounds, kf.duration_frames, kf.easing)
        for frame_idx, camera_bounds in enumerate(path):
            title = kf.title if frame_idx >= max(len(path) // 4, 1) else None
            subtitle = kf.subtitle if title else None
            if use_large_lod:
                assert large_state is not None
                frames.append(
                    _render_large_tour_frame(
                        graph,
                        large_state,
                        camera_bounds,
                        tour_cfg,
                        title=title,
                        subtitle=subtitle,
                    )
                )
            else:
                assert curves is not None
                frames.append(
                    _render_tour_frame(
                        graph,
                        positions,
                        curves,
                        label_positions,
                        camera_bounds,
                        config,
                        tour_cfg,
                        title=title,
                        subtitle=subtitle,
                    )
                )

    if frames:
        frames = [frames[0]] * tour_cfg.hold_start_frames + frames + [frames[-1]] * tour_cfg.hold_end_frames

    # Reuse the same writer path by adapting tour config to animation config shape.
    writer_cfg = AnimationConfig(
        fps=tour_cfg.fps,
        dpi=tour_cfg.dpi,
        figsize=tour_cfg.figsize,
        format=tour_cfg.format,
        output=tour_cfg.output,
        save_frames_dir=tour_cfg.save_frames_dir,
        codec=tour_cfg.codec,
        bitrate=tour_cfg.bitrate,
    )
    _write_frames(frames, tour_cfg.output, fmt, writer_cfg)

    return AnimationResult(
        output=tour_cfg.output,
        format=fmt,
        frame_count=len(frames),
        layout_snapshots=0,
        edge_snapshots=0,
    )


def poster(
    graph,
    positions: Optional[torch.Tensor] = None,
    config: Optional[LayoutConfig] = None,
    output: Optional[str] = None,
    poster_config: Optional[PosterConfig] = None,
    **kwargs,
) -> PosterResult:
    """Render a single cinematic still, with LOD support for huge graphs."""
    if config is None:
        from dagua.defaults import get_default_device, get_default_layout_overrides

        layout_overrides = get_default_layout_overrides()
        config = LayoutConfig(device=get_default_device(), **layout_overrides)

    poster_cfg = poster_config or PosterConfig()
    if output is not None:
        poster_cfg.output = output
    for key, value in kwargs.items():
        if hasattr(poster_cfg, key):
            setattr(poster_cfg, key, value)

    if positions is None:
        positions = layout(graph, config)

    graph.compute_node_sizes()
    if poster_cfg.output is None:
        poster_cfg.output = str(Path("dagua_poster.png").resolve())
    fmt = _detect_still_format(poster_cfg.output, poster_cfg.format)
    use_large_lod = graph.num_nodes >= poster_cfg.lod_threshold

    if poster_cfg.figsize is None:
        poster_cfg.figsize = _default_animation_figsize(graph, [_global_bounds(graph, positions)])

    selected = _poster_keyframe(graph, positions, poster_cfg)
    bounds = _keyframe_target_bounds(
        graph,
        positions,
        selected,
        _global_bounds(graph, positions),
        poster_cfg.follow_padding,
    )

    base_tour_cfg = TourConfig(
        dpi=poster_cfg.dpi,
        figsize=poster_cfg.figsize,
        format=poster_cfg.format,
        output=poster_cfg.output,
        scene=poster_cfg.scene,
        keyframes=poster_cfg.keyframes,
        show_titles=poster_cfg.show_titles,
        follow_padding=poster_cfg.follow_padding,
        lod_threshold=poster_cfg.lod_threshold,
        detail_node_limit=poster_cfg.detail_node_limit,
        label_node_limit=poster_cfg.label_node_limit,
        edge_sample_limit=poster_cfg.edge_sample_limit,
        density_bins=poster_cfg.density_bins,
        density_gamma=poster_cfg.density_gamma,
    )

    if use_large_lod:
        frame = _render_large_tour_frame(
            graph,
            _prepare_large_tour_state(graph, positions, base_tour_cfg),
            bounds,
            base_tour_cfg,
            title=selected.title if poster_cfg.show_titles else None,
            subtitle=selected.subtitle if poster_cfg.show_titles else None,
        )
    else:
        curves = route_edges(positions, graph.edge_index, graph.node_sizes, graph.direction, graph)
        if getattr(config, "edge_opt_steps", 0) >= 0:
            from dagua.layout.edge_optimization import optimize_edges

            curves = optimize_edges(curves, positions, graph.edge_index, graph.node_sizes, config, graph)
        label_positions = place_edge_labels(curves, positions, graph.node_sizes, graph.edge_labels, graph)
        frame = _render_tour_frame(
            graph,
            positions,
            curves,
            label_positions,
            bounds,
            config,
            base_tour_cfg,
            title=selected.title if poster_cfg.show_titles else None,
            subtitle=selected.subtitle if poster_cfg.show_titles else None,
        )

    _write_still_image(frame, poster_cfg.output, fmt, poster_cfg.dpi)
    return PosterResult(
        output=poster_cfg.output,
        format=fmt,
        bounds=bounds,
        used_large_lod=use_large_lod,
    )
