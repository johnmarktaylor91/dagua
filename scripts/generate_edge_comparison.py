#!/usr/bin/env python
# ruff: noqa: E402
"""Generate extended custom-edge comparison renders.

This script exercises Dagua's custom edge renderer across a focused set of
visual stress cases and saves the resulting PNG files into
``eval_output/edge_comparison/`` by default.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dagua import DaguaGraph, EdgeStyle, GraphStyle, NodeStyle, render
from dagua.graphviz_utils import layout_with_graphviz, render_graphviz_native
from dagua.render.edges import CubicBezier, DaguaEdge, DaguaEdgeCollection, build_arrowhead
from dagua.render.edges.ribbon import polyline_ribbon_path, trim_polyline_end

WHITE = "#FFFFFF"
NODE_FILL = "#E7EFF8"
NODE_STROKE = "#355C7D"
TEXT_COLOR = "#142235"
EDGE_COLORS = (
    "#2F6690",
    "#3A7D44",
    "#A23B72",
    "#C1661A",
    "#5E548E",
    "#00798C",
    "#8F2D56",
    "#556B2F",
    "#4D4D4D",
    "#8A5A44",
)
IMAGE_DPI = 190
EXPECTED_OUTPUT_FILENAMES: Tuple[str, ...] = (
    "orthogonal_routing.png",
    "polyline_routing.png",
    "edge_labels.png",
    "self_loops.png",
    "tapered_edges.png",
    "custom_dash_patterns.png",
    "tail_arrows.png",
    "short_vs_long.png",
    "node_shape_endpoints.png",
    "bidirectional.png",
    "graphviz_comparison.png",
    "mpl_vs_dagua_curved.png",
    "linestyle_gallery.png",
    "mega_stress.png",
    "mixed_styles_one_graph.png",
    "extreme_width_range.png",
    "parallel_multiedge_styles.png",
)


@dataclass(frozen=True)
class EdgeComparisonResult:
    """Paths emitted by the edge comparison generator.

    Parameters
    ----------
    output_dir : str
        Directory containing the rendered images.
    image_paths : list[str]
        Absolute paths to the rendered PNG files.
    """

    output_dir: str
    image_paths: List[str]


def _output_path(output_dir: Path, filename: str) -> Path:
    """Return the destination path for one artifact.

    Parameters
    ----------
    output_dir : Path
        Root output directory.
    filename : str
        Artifact filename.

    Returns
    -------
    Path
        Fully qualified output path.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename


def _new_figure(figsize: Tuple[float, float]) -> Tuple[Figure, Axes]:
    """Create a white, axis-free figure for edge showcase renders.

    Parameters
    ----------
    figsize : tuple[float, float]
        Figure size in inches.

    Returns
    -------
    tuple[Figure, Axes]
        Fresh matplotlib figure and axes.
    """

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def _save_figure(fig: Figure, path: Path) -> str:
    """Save a figure with consistent raster defaults.

    Parameters
    ----------
    fig : Figure
        Figure to save.
    path : Path
        Destination path.

    Returns
    -------
    str
        Saved path as a string.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=IMAGE_DPI, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    return str(path)


def _finish_axes(
    ax: Axes,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    title: str,
    subtitle: str = "",
) -> None:
    """Apply final bounds and titles to one showcase axes.

    Parameters
    ----------
    ax : Axes
        Target matplotlib axes.
    xlim : tuple[float, float]
        X-axis bounds in data units.
    ylim : tuple[float, float]
        Y-axis bounds in data units.
    title : str
        Main title.
    subtitle : str, default=""
        Optional subtitle shown below the title.

    Returns
    -------
    None
        The axes are modified in place.
    """

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if subtitle:
        ax.set_title(f"{title}\n{subtitle}", fontsize=15, color=TEXT_COLOR, pad=14)
    else:
        ax.set_title(title, fontsize=15, color=TEXT_COLOR, pad=14)


def _draw_node(
    ax: Axes,
    center: Tuple[float, float],
    label: str,
    size: Tuple[float, float] = (22.0, 14.0),
) -> None:
    """Draw a compact rounded endpoint node.

    Parameters
    ----------
    ax : Axes
        Axes receiving the node.
    center : tuple[float, float]
        Node center position.
    label : str
        Node label.
    size : tuple[float, float], default=(22.0, 14.0)
        Node width and height in data units.

    Returns
    -------
    None
        The node is drawn in place.
    """

    x, y = center
    width, height = size
    patch = FancyBboxPatch(
        (x - width / 2.0, y - height / 2.0),
        width,
        height,
        boxstyle="round,pad=0.2,rounding_size=4",
        facecolor=NODE_FILL,
        edgecolor=NODE_STROKE,
        linewidth=1.6,
        zorder=5,
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=10,
        color=TEXT_COLOR,
        zorder=6,
    )


def _draw_curve_nodes(
    ax: Axes,
    rows: Sequence[Tuple[Tuple[float, float], Tuple[float, float], str, str]],
) -> None:
    """Draw endpoint nodes for a batch of edge rows.

    Parameters
    ----------
    ax : Axes
        Axes receiving the node drawings.
    rows : Sequence[tuple[tuple[float, float], tuple[float, float], str, str]]
        Row specifications of start point, end point, start label, and end label.

    Returns
    -------
    None
        Nodes are drawn in place.
    """

    for start, end, start_label, end_label in rows:
        _draw_node(ax, start, start_label)
        _draw_node(ax, end, end_label)


def _horizontal_curve(
    start: Tuple[float, float],
    end: Tuple[float, float],
    bend: float,
) -> CubicBezier:
    """Build a horizontal cubic used in direct-edge examples.

    Parameters
    ----------
    start : tuple[float, float]
        Curve start point.
    end : tuple[float, float]
        Curve end point.
    bend : float
        Vertical control-point offset.

    Returns
    -------
    CubicBezier
        Cubic curve spanning the requested row.
    """

    sx, sy = start
    tx, ty = end
    dx = tx - sx
    return CubicBezier.from_points(
        (sx, sy),
        (sx + dx * 0.28, sy + bend),
        (sx + dx * 0.72, ty + bend),
        (tx, ty),
    )


def _draw_polyline_edge(
    ax: Axes,
    points: Sequence[Tuple[float, float]],
    width: float,
    color: str,
    arrowhead: str = "normal",
    tail_arrow: str = "none",
    arrow_fill: str = "filled",
    arrow_length: float = 12.0,
    arrow_width: float = 9.0,
) -> None:
    """Draw a polyline edge using the low-level ribbon and arrowhead helpers.

    Parameters
    ----------
    ax : Axes
        Axes receiving the edge.
    points : Sequence[tuple[float, float]]
        Polyline centerline vertices.
    width : float
        Ribbon width in data units.
    color : str
        Edge color.
    arrowhead : str, default="normal"
        Arrowhead spec at the target end.
    tail_arrow : str, default="none"
        Arrowhead spec at the source end.
    arrow_fill : str, default="filled"
        Arrowhead fill mode.
    arrow_length : float, default=12.0
        Arrowhead length in data units.
    arrow_width : float, default=9.0
        Arrowhead width in data units.

    Returns
    -------
    None
        The edge is drawn in place.
    """

    centerline = np.asarray(points, dtype=np.float64)
    head_result = None
    tail_result = None
    body_points = centerline

    if arrowhead != "none":
        head_direction = centerline[-2] - centerline[-1]
        head_result = build_arrowhead(
            arrowhead,
            tip=centerline[-1],
            tangent=head_direction,
            length=arrow_length,
            width=arrow_width,
            body_width=width,
            fill_mode=arrow_fill,
        )
        body_points = trim_polyline_end(body_points, head_result.trim_contour)

    if tail_arrow != "none":
        tail_direction = centerline[1] - centerline[0]
        tail_result = build_arrowhead(
            tail_arrow,
            tip=centerline[0],
            tangent=tail_direction,
            length=arrow_length,
            width=arrow_width,
            body_width=width,
            fill_mode=arrow_fill,
        )
        reversed_points = trim_polyline_end(body_points[::-1], tail_result.trim_contour)
        body_points = reversed_points[::-1]

    body_patch = PathPatch(
        polyline_ribbon_path(body_points, width=width, cap_start="butt", cap_end="butt"),
        facecolor=color,
        edgecolor="none",
        alpha=0.96,
        zorder=1,
    )
    ax.add_patch(body_patch)

    for result in (head_result, tail_result):
        if result is None:
            continue
        for path in result.filled_paths:
            ax.add_patch(PathPatch(path, facecolor=color, edgecolor="none", alpha=0.96, zorder=2))
        for path in result.stroked_paths:
            ax.add_patch(
                PathPatch(
                    path,
                    facecolor="none",
                    edgecolor=color,
                    linewidth=max(1.2, width * result.stroke_width_scale * 0.32),
                    alpha=0.96,
                    capstyle="round",
                    joinstyle="round",
                    zorder=2,
                )
            )


def _apply_graph_style(graph: DaguaGraph, figsize: Tuple[float, float]) -> None:
    """Configure a graph for white-background comparison output.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to configure.
    figsize : tuple[float, float]
        Minimum and maximum figure size target.

    Returns
    -------
    None
        The graph style is mutated in place.
    """

    style: GraphStyle = graph.graph_style
    style.background_color = WHITE
    style.margin = 20.0
    style.min_figsize = figsize
    style.max_figsize = figsize
    style.edge_label_background = WHITE


def _render_graph_case(
    graph: DaguaGraph,
    positions: torch.Tensor,
    output_path: Path,
    figsize: Tuple[float, float],
    title: str,
) -> str:
    """Render a fixed-position graph artifact.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to render.
    positions : torch.Tensor
        Fixed node positions with shape ``[N, 2]``.
    output_path : Path
        Destination path.
    figsize : tuple[float, float]
        Figure size for the render.
    title : str
        Figure title.

    Returns
    -------
    str
        Saved path.
    """

    _apply_graph_style(graph, figsize)
    graph.compute_node_sizes()
    fig, _ = render(
        graph,
        positions=positions,
        output=str(output_path),
        figsize=figsize,
        dpi=IMAGE_DPI,
        title=title,
    )
    plt.close(fig)
    return str(output_path)


def _build_orthogonal_routing(output_dir: Path) -> str:
    """Render exact elbow-routed edges with multiple arrowhead styles.

    Parameters
    ----------
    output_dir : Path
        Root output directory.

    Returns
    -------
    str
        Saved path.
    """

    path = _output_path(output_dir, "orthogonal_routing.png")
    fig, ax = _new_figure((11.0, 6.8))
    rows = [
        ((-120.0, 90.0), (120.0, 90.0), "A", "B"),
        ((-120.0, 30.0), (120.0, 30.0), "C", "D"),
        ((-120.0, -30.0), (120.0, -30.0), "E", "F"),
        ((-120.0, -90.0), (120.0, -90.0), "G", "H"),
    ]
    specs = [
        ("normal", EDGE_COLORS[0], 4.0),
        ("vee", EDGE_COLORS[1], 5.0),
        ("diamond", EDGE_COLORS[2], 6.0),
        ("crow", EDGE_COLORS[3], 5.0),
    ]
    for (start, end, _, _), (arrow, color, width) in zip(rows, specs):
        sx, sy = start
        tx, ty = end
        mid_x = 10.0
        points = (
            (sx + 12.0, sy),
            (mid_x, sy),
            (mid_x, ty - 22.0),
            (tx - 12.0, ty - 22.0),
            (tx - 12.0, ty),
        )
        _draw_polyline_edge(ax, points, width=width, color=color, arrowhead=arrow)
        ax.text(
            0.0,
            sy + 16.0,
            f"{arrow} arrow",
            ha="center",
            va="bottom",
            fontsize=10,
            color=TEXT_COLOR,
        )
    _draw_curve_nodes(ax, rows)
    _finish_axes(
        ax,
        (-160.0, 160.0),
        (-125.0, 125.0),
        "Orthogonal Routing",
        "Exact right-angle polylines with varied target arrowheads",
    )
    return _save_figure(fig, path)


def _build_polyline_routing(output_dir: Path) -> str:
    """Render multi-segment straight polyline edges.

    Parameters
    ----------
    output_dir : Path
        Root output directory.

    Returns
    -------
    str
        Saved path.
    """

    path = _output_path(output_dir, "polyline_routing.png")
    fig, ax = _new_figure((11.6, 7.0))
    examples = [
        (
            (
                (-130.0, 85.0),
                (-40.0, 85.0),
                (-40.0, 35.0),
                (55.0, 35.0),
                (55.0, 85.0),
                (130.0, 85.0),
            ),
            EDGE_COLORS[0],
            "5 segments",
        ),
        (
            (
                (-130.0, 15.0),
                (-75.0, 15.0),
                (-25.0, -30.0),
                (25.0, 60.0),
                (80.0, 15.0),
                (130.0, 15.0),
            ),
            EDGE_COLORS[4],
            "zig-zag",
        ),
        (
            (
                (-130.0, -75.0),
                (-60.0, -75.0),
                (-60.0, -110.0),
                (10.0, -110.0),
                (10.0, -40.0),
                (85.0, -40.0),
                (85.0, -75.0),
                (130.0, -75.0),
            ),
            EDGE_COLORS[6],
            "stair-step",
        ),
    ]
    rows = [
        ((-142.0, 85.0), (142.0, 85.0), "S1", "T1"),
        ((-142.0, 15.0), (142.0, 15.0), "S2", "T2"),
        ((-142.0, -75.0), (142.0, -75.0), "S3", "T3"),
    ]
    for (points, color, label), (_, _, start_label, end_label) in zip(examples, rows):
        _draw_polyline_edge(ax, points, width=4.6, color=color, arrowhead="normal")
        ax.text(
            0.0,
            points[0][1] + 18.0,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            color=TEXT_COLOR,
        )
        _draw_node(ax, points[0], start_label)
        _draw_node(ax, points[-1], end_label)
    _finish_axes(
        ax,
        (-170.0, 170.0),
        (-135.0, 120.0),
        "Polyline Routing",
        "Multi-segment straight bodies built from the ribbon polyline path",
    )
    return _save_figure(fig, path)


def _build_edge_labels(output_dir: Path) -> str:
    """Render edge labels across positions and rotation modes.

    Parameters
    ----------
    output_dir : Path
        Root output directory.

    Returns
    -------
    str
        Saved path.
    """

    path = _output_path(output_dir, "edge_labels.png")
    fig, ax = _new_figure((12.4, 7.8))
    edges: List[DaguaEdge] = []
    endpoints: List[Tuple[Tuple[float, float], Tuple[float, float], str, str]] = []
    positions = (0.0, 0.25, 0.5, 0.75, 1.0)
    for row_index, rotate in enumerate((False, True)):
        y = 70.0 if not rotate else -60.0
        start = (-145.0, y)
        end = (145.0, y)
        for idx, label_pos in enumerate(positions):
            color = EDGE_COLORS[idx]
            bend = 50.0 if rotate else 30.0
            curve = _horizontal_curve(start, end, bend=bend)
            edges.append(
                DaguaEdge(
                    curve=curve,
                    width=3.4,
                    color=color,
                    arrowhead="normal",
                    stroke_width=2.0,
                    label=f"{label_pos:.2f}",
                    label_position=label_pos,
                    label_offset=14.0 if rotate else 11.0,
                    label_rotate=rotate,
                    label_side="left" if idx % 2 == 0 else "right",
                    label_font_size=10.0,
                )
            )
        endpoints.append((start, end, "Start", "End"))
        ax.text(
            0.0,
            y + 44.0,
            "Horizontal labels" if not rotate else "Rotated labels",
            ha="center",
            va="bottom",
            fontsize=12,
            color=TEXT_COLOR,
        )
    DaguaEdgeCollection(edges).render(ax)
    _draw_curve_nodes(ax, endpoints)
    _finish_axes(
        ax,
        (-180.0, 180.0),
        (-120.0, 120.0),
        "Edge Labels",
        "Label positions 0.00 to 1.00, rendered horizontally and tangent-rotated",
    )
    return _save_figure(fig, path)


def _build_self_loops(output_dir: Path) -> str:
    """Render self-loops with varied arrowheads and widths.

    Parameters
    ----------
    output_dir : Path
        Root output directory.

    Returns
    -------
    str
        Saved path.
    """

    graph = DaguaGraph(direction="TB")
    node_specs = (
        ("normal", -150.0, 0.0, "normal", 1.2),
        ("vee", -50.0, 0.0, "vee", 2.0),
        ("diamond", 50.0, 0.0, "diamond", 3.0),
        ("crow", 150.0, 0.0, "crow", 4.0),
    )
    for name, _, _, _, _ in node_specs:
        graph.add_node(name, label=name)
    for index, (name, _, _, arrow, width) in enumerate(node_specs):
        graph.add_edge(
            name,
            name,
            label=f"{arrow}, w={width:.1f}",
            style=EdgeStyle(color=EDGE_COLORS[index], arrow=arrow, width=width),
        )
    positions = torch.tensor([[x, y] for _, x, y, _, _ in node_specs], dtype=torch.float32)
    return _render_graph_case(
        graph,
        positions,
        _output_path(output_dir, "self_loops.png"),
        figsize=(11.8, 4.4),
        title="Self-Loops",
    )


def _build_tapered_edges(output_dir: Path) -> str:
    """Render an approximate taper by stepping body widths along the route.

    Parameters
    ----------
    output_dir : Path
        Root output directory.

    Returns
    -------
    str
        Saved path.
    """

    path = _output_path(output_dir, "tapered_edges.png")
    fig, ax = _new_figure((11.0, 6.6))
    rows = [
        ((-130.0, 80.0), (130.0, 80.0), "T1", "T2"),
        ((-130.0, 0.0), (130.0, 0.0), "T3", "T4"),
        ((-130.0, -80.0), (130.0, -80.0), "T5", "T6"),
    ]
    width_sets = (
        (10.0, 8.0, 6.0, 4.0, 2.5),
        (12.0, 9.0, 6.0, 3.0, 1.5),
        (8.0, 6.5, 5.0, 3.5, 2.0),
    )
    for row, widths, color in zip(rows, width_sets, EDGE_COLORS[:3]):
        start, end, _, _ = row
        curve = _horizontal_curve(start, end, bend=35.0)
        segments = np.linspace(0.0, 1.0, num=len(widths) + 1)
        edges: List[DaguaEdge] = []
        for idx, width in enumerate(widths):
            subcurve = CubicBezier.from_points(
                *tuple(
                    np.asarray(point)
                    for point in (
                        curve.p0 if idx == 0 else _evaluate_curve(curve, float(segments[idx])),
                        _evaluate_curve(curve, float((segments[idx] + segments[idx + 1]) / 2.0)),
                        _evaluate_curve(curve, float((segments[idx] + segments[idx + 1]) / 2.0)),
                        _evaluate_curve(curve, float(segments[idx + 1])),
                    )
                )
            )
            edges.append(
                DaguaEdge(
                    curve=subcurve,
                    width=width,
                    color=color,
                    alpha=0.94,
                    arrowhead="none" if idx < len(widths) - 1 else "normal",
                    stroke_width=max(1.0, width * 0.35),
                )
            )
        DaguaEdgeCollection(edges).render(ax)
        ax.text(
            0.0,
            start[1] + 20.0,
            f"{widths[0]:.1f} -> {widths[-1]:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=TEXT_COLOR,
        )
    _draw_curve_nodes(ax, rows)
    _finish_axes(
        ax,
        (-165.0, 165.0),
        (-120.0, 120.0),
        "Tapered Edges",
        "Approximated with width-stepped subcurves "
        "because the current edge API is constant-width per segment",
    )
    return _save_figure(fig, path)


def _build_custom_dash_patterns(output_dir: Path) -> str:
    """Render arbitrary dash sequences using direct edge primitives.

    Parameters
    ----------
    output_dir : Path
        Root output directory.

    Returns
    -------
    str
        Saved path.
    """

    path = _output_path(output_dir, "custom_dash_patterns.png")
    fig, ax = _new_figure((11.6, 7.0))
    dash_patterns: Tuple[Tuple[float, ...], ...] = (
        (12.0, 4.0),
        (10.0, 3.0, 2.0, 3.0),
        (7.0, 2.0, 2.0, 2.0, 2.0, 2.0),
        (14.0, 3.0, 1.5, 3.0, 1.5, 5.0),
    )
    edges: List[DaguaEdge] = []
    rows: List[Tuple[Tuple[float, float], Tuple[float, float], str, str]] = []
    for idx, pattern in enumerate(dash_patterns):
        y = 90.0 - idx * 60.0
        start = (-140.0, y)
        end = (140.0, y)
        rows.append((start, end, f"D{idx + 1}", f"E{idx + 1}"))
        edges.append(
            DaguaEdge(
                curve=_horizontal_curve(start, end, bend=18.0),
                width=3.6,
                color=EDGE_COLORS[idx],
                linestyle=pattern,
                arrowhead="normal",
                stroke_width=2.0,
            )
        )
        ax.text(
            0.0,
            y + 17.0,
            str(pattern),
            ha="center",
            va="bottom",
            fontsize=10,
            color=TEXT_COLOR,
        )
    DaguaEdgeCollection(edges).render(ax)
    _draw_curve_nodes(ax, rows)
    _finish_axes(
        ax,
        (-175.0, 175.0),
        (-110.0, 130.0),
        "Custom Dash Patterns",
        "Arbitrary on/off sequences routed through arc-length dash placement",
    )
    return _save_figure(fig, path)


def _build_tail_arrows(output_dir: Path) -> str:
    """Render edges with visible tail and head arrowheads.

    Parameters
    ----------
    output_dir : Path
        Root output directory.

    Returns
    -------
    str
        Saved path.
    """

    path = _output_path(output_dir, "tail_arrows.png")
    fig, ax = _new_figure((11.6, 6.6))
    specs = (
        ("dot", "normal", EDGE_COLORS[0]),
        ("diamond", "vee", EDGE_COLORS[1]),
        ("tee", "crow", EDGE_COLORS[2]),
        ("vee", "diamond", EDGE_COLORS[3]),
    )
    rows: List[Tuple[Tuple[float, float], Tuple[float, float], str, str]] = []
    edges: List[DaguaEdge] = []
    for idx, (tail, head, color) in enumerate(specs):
        y = 95.0 - idx * 60.0
        start = (-135.0, y)
        end = (135.0, y)
        rows.append((start, end, "src", "dst"))
        edges.append(
            DaguaEdge(
                curve=_horizontal_curve(start, end, bend=26.0),
                width=4.0,
                color=color,
                arrowhead=head,
                tail_arrow=tail,
                stroke_width=2.2,
                label=f"{tail} -> {head}",
                label_position=0.5,
                label_offset=12.0,
                label_font_size=9.0,
            )
        )
    DaguaEdgeCollection(edges).render(ax)
    _draw_curve_nodes(ax, rows)
    _finish_axes(
        ax,
        (-170.0, 170.0),
        (-110.0, 130.0),
        "Tail Arrows",
        "Arrow markers rendered on both source and target terminals",
    )
    return _save_figure(fig, path)


def _build_short_vs_long(output_dir: Path) -> str:
    """Render near-touching and long-span edges side by side.

    Parameters
    ----------
    output_dir : Path
        Root output directory.

    Returns
    -------
    str
        Saved path.
    """

    path = _output_path(output_dir, "short_vs_long.png")
    fig, ax = _new_figure((12.0, 5.8))
    edges = [
        DaguaEdge(
            curve=_horizontal_curve((-70.0, 0.0), (-18.0, 0.0), bend=10.0),
            width=4.0,
            color=EDGE_COLORS[0],
            arrowhead="normal",
            stroke_width=2.0,
            label="short",
            label_position=0.55,
            label_offset=10.0,
            label_font_size=10.0,
        ),
        DaguaEdge(
            curve=_horizontal_curve((20.0, 0.0), (170.0, 0.0), bend=46.0),
            width=4.0,
            color=EDGE_COLORS[4],
            arrowhead="normal",
            stroke_width=2.0,
            label="long",
            label_position=0.5,
            label_offset=12.0,
            label_font_size=10.0,
        ),
    ]
    DaguaEdgeCollection(edges).render(ax)
    for center, label in [
        ((-82.0, 0.0), "A"),
        ((-6.0, 0.0), "B"),
        ((8.0, 0.0), "C"),
        ((182.0, 0.0), "D"),
    ]:
        _draw_node(ax, center, label, size=(20.0, 13.0))
    ax.text(
        -40.0,
        44.0,
        "Nearly touching nodes",
        ha="center",
        va="bottom",
        fontsize=12,
        color=TEXT_COLOR,
    )
    ax.text(
        96.0,
        70.0,
        "Long span",
        ha="center",
        va="bottom",
        fontsize=12,
        color=TEXT_COLOR,
    )
    _finish_axes(
        ax,
        (-110.0, 210.0),
        (-45.0, 95.0),
        "Short vs Long",
        "Terminal trimming, labels, and arrowheads across extreme edge lengths",
    )
    return _save_figure(fig, path)


def _build_node_shape_endpoints(output_dir: Path) -> str:
    """Render shape-aware endpoint termination across multiple node shapes.

    Parameters
    ----------
    output_dir : Path
        Root output directory.

    Returns
    -------
    str
        Saved path.
    """

    graph = DaguaGraph(direction="LR")
    graph.add_node(
        "src", label="Source", style=NodeStyle(shape="circle", fill="#F7E8D2", stroke="#B7791F")
    )
    target_specs = (
        ("rect", 120.0, 95.0),
        ("ellipse", 120.0, 30.0),
        ("diamond", 120.0, -35.0),
        ("roundrect", 120.0, -100.0),
    )
    for shape, _, _ in target_specs:
        graph.add_node(
            shape, label=shape, style=NodeStyle(shape=shape, fill=NODE_FILL, stroke=NODE_STROKE)
        )
        graph.add_edge(
            "src",
            shape,
            style=EdgeStyle(
                color=EDGE_COLORS[len(graph.edge_labels) % len(EDGE_COLORS)],
                width=2.2,
                arrow="normal",
                routing="straight",
            ),
        )
    positions = torch.tensor(
        [[-70.0, 0.0]] + [[x, y] for _, x, y in target_specs],
        dtype=torch.float32,
    )
    return _render_graph_case(
        graph,
        positions,
        _output_path(output_dir, "node_shape_endpoints.png"),
        figsize=(9.8, 6.8),
        title="Node Shape Endpoints",
    )


def _build_bidirectional(output_dir: Path) -> str:
    """Render parallel A->B and B->A edges with distinct styles.

    Parameters
    ----------
    output_dir : Path
        Root output directory.

    Returns
    -------
    str
        Saved path.
    """

    graph = DaguaGraph(direction="LR")
    graph.add_node("A", label="A")
    graph.add_node("B", label="B")
    graph.add_edge(
        "A",
        "B",
        label="A -> B",
        style=EdgeStyle(
            color=EDGE_COLORS[0], width=2.4, arrow="normal", style="dashed", curvature=0.35
        ),
    )
    graph.add_edge(
        "B",
        "A",
        label="B -> A",
        style=EdgeStyle(
            color=EDGE_COLORS[2], width=2.4, arrow="vee", style="dotted", curvature=0.75
        ),
    )
    positions = torch.tensor([[0.0, 0.0], [150.0, 0.0]], dtype=torch.float32)
    return _render_graph_case(
        graph,
        positions,
        _output_path(output_dir, "bidirectional.png"),
        figsize=(7.2, 4.8),
        title="Bidirectional",
    )


def _build_graphviz_comparison(output_dir: Path) -> str:
    """Render Graphviz native output beside Dagua custom edges.

    Parameters
    ----------
    output_dir : Path
        Root output directory.

    Returns
    -------
    str
        Saved path.
    """

    if shutil.which("dot") is None:
        raise RuntimeError("Graphviz 'dot' is required for graphviz_comparison.png")

    graph = DaguaGraph(direction="TB")
    for node in ("Input", "Encode", "Fuse", "Decode", "Output"):
        graph.add_node(node, label=node)
    edge_specs = (
        ("Input", "Encode", EdgeStyle(color=EDGE_COLORS[0], width=2.0, arrow="normal")),
        ("Input", "Fuse", EdgeStyle(color=EDGE_COLORS[5], width=1.8, arrow="vee", style="dashed")),
        ("Encode", "Decode", EdgeStyle(color=EDGE_COLORS[1], width=2.1, arrow="diamond")),
        (
            "Fuse",
            "Decode",
            EdgeStyle(color=EDGE_COLORS[2], width=1.9, arrow="normal", style="dotted"),
        ),
        ("Decode", "Output", EdgeStyle(color=EDGE_COLORS[3], width=2.2, arrow="normal")),
    )
    for source, target, style in edge_specs:
        graph.add_edge(source, target, style=style)

    dagua_output = output_dir / "_graphviz_cmp_dagua.png"
    native_output = output_dir / "_graphviz_cmp_native.png"
    final_output = _output_path(output_dir, "graphviz_comparison.png")

    graphviz_positions = layout_with_graphviz(graph, engine="dot")
    graphviz_positions[:, 1] *= -1.0
    _apply_graph_style(graph, (6.0, 5.8))
    render(
        graph,
        positions=graphviz_positions,
        output=str(dagua_output),
        figsize=(6.0, 5.8),
        dpi=IMAGE_DPI,
        title="Dagua custom edges on dot layout",
    )
    render_graphviz_native(graph, str(native_output), engine="dot")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.0))
    fig.patch.set_facecolor(WHITE)
    for axis, image_path, title in (
        (axes[0], native_output, "Graphviz dot (native)"),
        (axes[1], dagua_output, "Dagua custom edges"),
    ):
        axis.imshow(np.asarray(Image.open(image_path)))
        axis.set_title(title, fontsize=14, color=TEXT_COLOR, pad=10)
        axis.axis("off")
    fig.suptitle("Graphviz Comparison", fontsize=16, color=TEXT_COLOR)
    image_path = _save_figure(fig, final_output)
    dagua_output.unlink(missing_ok=True)
    native_output.unlink(missing_ok=True)
    return image_path


def _build_mpl_vs_dagua_curved(output_dir: Path) -> str:
    """Render FancyArrowPatch and Dagua curves side by side.

    Parameters
    ----------
    output_dir : Path
        Root output directory.

    Returns
    -------
    str
        Saved path.
    """

    path = _output_path(output_dir, "mpl_vs_dagua_curved.png")
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13.0, 6.8))
    fig.patch.set_facecolor(WHITE)
    for axis in (ax_left, ax_right):
        axis.set_facecolor(WHITE)
        axis.set_aspect("equal")
        axis.axis("off")
        axis.set_xlim(-30.0, 180.0)
        axis.set_ylim(-110.0, 110.0)

    curvatures = (0.12, 0.24, 0.38, 0.54)
    dagua_edges: List[DaguaEdge] = []
    rows: List[Tuple[Tuple[float, float], Tuple[float, float], str, str]] = []
    for idx, curvature in enumerate(curvatures):
        y = 80.0 - idx * 52.0
        start = (0.0, y)
        end = (150.0, y)
        color = EDGE_COLORS[idx]
        rows.append((start, end, "A", "B"))
        rad = curvature
        patch = FancyArrowPatch(
            posA=start,
            posB=end,
            arrowstyle="-|>",
            mutation_scale=18.0,
            linewidth=2.6,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            linestyle="solid",
            zorder=2,
        )
        ax_left.add_patch(patch)
        ax_left.text(
            75.0,
            y + 20.0,
            f"rad={rad:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=TEXT_COLOR,
        )
        dagua_edges.append(
            DaguaEdge(
                curve=_horizontal_curve(start, end, bend=rad * 140.0),
                width=3.5,
                color=color,
                arrowhead="normal",
                stroke_width=2.0,
            )
        )
        ax_right.text(
            75.0,
            y + 20.0,
            f"bend={rad * 140.0:.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=TEXT_COLOR,
        )

    DaguaEdgeCollection(dagua_edges).render(ax_right)
    _draw_curve_nodes(ax_left, rows)
    _draw_curve_nodes(ax_right, rows)
    ax_left.set_title("matplotlib FancyArrowPatch", fontsize=14, color=TEXT_COLOR, pad=12)
    ax_right.set_title("Dagua custom curved edges", fontsize=14, color=TEXT_COLOR, pad=12)
    fig.suptitle("MPL vs Dagua Curved Edges", fontsize=16, color=TEXT_COLOR)
    return _save_figure(fig, path)


def _build_linestyle_gallery(output_dir: Path) -> str:
    """Render a restrained line-style gallery that emphasizes dash rhythm.

    Parameters
    ----------
    output_dir : Path
        Root output directory.

    Returns
    -------
    str
        Saved path.
    """

    path = _output_path(output_dir, "linestyle_gallery.png")
    fig, ax = _new_figure((10.2, 7.1))
    color = EDGE_COLORS[0]
    width = 1.25
    specs: Tuple[Tuple[str, str], ...] = (
        ("solid", "Solid"),
        ("dashed", "Dashed"),
        ("dotted", "Dotted"),
        ("dashdot", "Dash Dot"),
    )
    edges: List[DaguaEdge] = []
    for index, (pattern, label) in enumerate(specs):
        y = 108.0 - index * 62.0
        start = (-145.0, y)
        end = (165.0, y)
        edges.append(
            DaguaEdge(
                curve=_horizontal_curve(start, end, bend=0.0),
                width=width,
                color=color,
                alpha=0.96,
                linestyle=pattern,
                arrowhead="none",
                stroke_width=0.95,
            )
        )
        ax.text(
            -152.0,
            y,
            label,
            ha="right",
            va="center",
            fontsize=11,
            color=TEXT_COLOR,
        )
    DaguaEdgeCollection(edges).render(ax)
    _finish_axes(
        ax,
        (-205.0, 182.0),
        (-92.0, 138.0),
        "Line Style Gallery",
    )
    return _save_figure(fig, path)


def _build_mega_stress(output_dir: Path) -> str:
    """Render the hardest single custom edge combination.

    Parameters
    ----------
    output_dir : Path
        Root output directory.

    Returns
    -------
    str
        Saved path.
    """

    path = _output_path(output_dir, "mega_stress.png")
    fig, ax = _new_figure((10.0, 5.6))
    edge = DaguaEdge(
        curve=_horizontal_curve((-120.0, 0.0), (120.0, 0.0), bend=70.0),
        width=5.4,
        color="#0F766E",
        linestyle="dashed",
        arrowhead="vee",
        tail_arrow="dot",
        arrow_fill="hollow",
        stroke_width=4.0,
        label="4pt dashed hollow-vee with tail arrow",
        label_position=0.58,
        label_offset=16.0,
        label_rotate=True,
        label_font_size=11.0,
        label_side="left",
    )
    DaguaEdgeCollection([edge]).render(ax)
    _draw_curve_nodes(ax, [((-120.0, 0.0), (120.0, 0.0), "Input", "Target")])
    _finish_axes(
        ax,
        (-150.0, 150.0),
        (-50.0, 110.0),
        "Mega Stress",
        "Thick + curved + dashed + hollow vee + tail arrow + rotated label",
    )
    return _save_figure(fig, path)


def _build_mixed_styles_one_graph(output_dir: Path) -> str:
    """Render a mixed-style graph where each edge uses a distinct style.

    Parameters
    ----------
    output_dir : Path
        Root output directory.

    Returns
    -------
    str
        Saved path.
    """

    graph = DaguaGraph(direction="TB")
    positions = []
    for idx in range(10):
        x = (idx % 5) * 90.0
        y = 80.0 if idx < 5 else -30.0
        node_id = f"N{idx}"
        graph.add_node(node_id, label=node_id)
        positions.append((x, y))
    style_specs = (
        EdgeStyle(
            color=EDGE_COLORS[0], width=1.4, arrow="normal", routing="bezier", curvature=0.25
        ),
        EdgeStyle(
            color=EDGE_COLORS[1],
            width=1.8,
            arrow="vee",
            style="dashed",
            routing="bezier",
            curvature=0.40,
        ),
        EdgeStyle(
            color=EDGE_COLORS[2], width=2.2, arrow="diamond", style="dotted", routing="straight"
        ),
        EdgeStyle(color=EDGE_COLORS[3], width=2.6, arrow="crow", routing="bezier", curvature=0.60),
        EdgeStyle(
            color=EDGE_COLORS[4],
            width=3.0,
            arrow="tee",
            tail_arrow="dot",
            routing="bezier",
            curvature=0.52,
        ),
        EdgeStyle(color=EDGE_COLORS[5], width=1.6, arrow="dot", routing="straight"),
        EdgeStyle(color=EDGE_COLORS[6], width=2.0, arrow="box", routing="bezier", curvature=0.32),
        EdgeStyle(
            color=EDGE_COLORS[7],
            width=2.4,
            arrow="normal",
            tail_arrow="vee",
            routing="bezier",
            curvature=0.70,
        ),
        EdgeStyle(
            color=EDGE_COLORS[8], width=2.8, arrow="diamond", routing="bezier", curvature=0.18
        ),
        EdgeStyle(
            color=EDGE_COLORS[9], width=1.9, arrow="normal", style="dashed", routing="straight"
        ),
    )
    edge_pairs = (
        ("N0", "N5"),
        ("N0", "N6"),
        ("N1", "N6"),
        ("N1", "N7"),
        ("N2", "N7"),
        ("N2", "N8"),
        ("N3", "N8"),
        ("N3", "N9"),
        ("N4", "N9"),
        ("N4", "N5"),
    )
    for idx, ((source, target), style) in enumerate(zip(edge_pairs, style_specs)):
        graph.add_edge(source, target, label=f"e{idx}", style=style)
    return _render_graph_case(
        graph,
        torch.tensor(positions, dtype=torch.float32),
        _output_path(output_dir, "mixed_styles_one_graph.png"),
        figsize=(10.6, 5.8),
        title="Mixed Styles in One Graph",
    )


def _build_extreme_width_range(output_dir: Path) -> str:
    """Render a width ladder from near-hairline to very heavy edges.

    Parameters
    ----------
    output_dir : Path
        Root output directory.

    Returns
    -------
    str
        Saved path.
    """

    path = _output_path(output_dir, "extreme_width_range.png")
    fig, ax = _new_figure((11.8, 7.4))
    widths = (0.1, 0.35, 0.8, 1.6, 3.2, 6.4, 12.0)
    edges: List[DaguaEdge] = []
    rows: List[Tuple[Tuple[float, float], Tuple[float, float], str, str]] = []
    for idx, width in enumerate(widths):
        y = 120.0 - idx * 40.0
        start = (-130.0, y)
        end = (130.0, y)
        rows.append((start, end, "L", "R"))
        edges.append(
            DaguaEdge(
                curve=_horizontal_curve(start, end, bend=14.0),
                width=width,
                color=EDGE_COLORS[idx % len(EDGE_COLORS)],
                arrowhead="normal",
                stroke_width=max(0.6, width * 0.35),
            )
        )
        ax.text(
            0.0,
            y + 14.0,
            f"width={width:.2f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=TEXT_COLOR,
        )
    DaguaEdgeCollection(edges).render(ax)
    _draw_curve_nodes(ax, rows)
    _finish_axes(
        ax,
        (-170.0, 170.0),
        (-150.0, 145.0),
        "Extreme Width Range",
        "Body widths from 0.10 to 12.00 data units",
    )
    return _save_figure(fig, path)


def _build_parallel_multiedge_styles(output_dir: Path) -> str:
    """Render five parallel same-direction edges with different styles.

    Parameters
    ----------
    output_dir : Path
        Root output directory.

    Returns
    -------
    str
        Saved path.
    """

    graph = DaguaGraph(direction="LR")
    graph.add_node("A", label="A")
    graph.add_node("B", label="B")
    styles = (
        EdgeStyle(
            color=EDGE_COLORS[0], width=1.6, arrow="normal", routing="bezier", curvature=0.22
        ),
        EdgeStyle(
            color=EDGE_COLORS[1],
            width=1.9,
            arrow="vee",
            style="dashed",
            routing="bezier",
            curvature=0.32,
        ),
        EdgeStyle(
            color=EDGE_COLORS[2],
            width=2.2,
            arrow="diamond",
            style="dotted",
            routing="bezier",
            curvature=0.42,
        ),
        EdgeStyle(color=EDGE_COLORS[3], width=2.5, arrow="crow", routing="bezier", curvature=0.52),
        EdgeStyle(
            color=EDGE_COLORS[4],
            width=2.8,
            arrow="tee",
            tail_arrow="dot",
            routing="bezier",
            curvature=0.62,
        ),
    )
    for idx, style in enumerate(styles):
        graph.add_edge("A", "B", label=f"m{idx + 1}", style=style)
    positions = torch.tensor([[0.0, 0.0], [180.0, 0.0]], dtype=torch.float32)
    return _render_graph_case(
        graph,
        positions,
        _output_path(output_dir, "parallel_multiedge_styles.png"),
        figsize=(8.8, 5.0),
        title="Parallel Multiedge Styles",
    )


def _build_all_builders() -> Mapping[str, Callable[[Path], str]]:
    """Return the artifact builder registry.

    Parameters
    ----------
    None
        The builder table is static.

    Returns
    -------
    Mapping[str, Callable[[Path], str]]
        Filename-to-builder mapping.
    """

    return {
        "orthogonal_routing.png": _build_orthogonal_routing,
        "polyline_routing.png": _build_polyline_routing,
        "edge_labels.png": _build_edge_labels,
        "self_loops.png": _build_self_loops,
        "tapered_edges.png": _build_tapered_edges,
        "custom_dash_patterns.png": _build_custom_dash_patterns,
        "tail_arrows.png": _build_tail_arrows,
        "short_vs_long.png": _build_short_vs_long,
        "node_shape_endpoints.png": _build_node_shape_endpoints,
        "bidirectional.png": _build_bidirectional,
        "graphviz_comparison.png": _build_graphviz_comparison,
        "mpl_vs_dagua_curved.png": _build_mpl_vs_dagua_curved,
        "linestyle_gallery.png": _build_linestyle_gallery,
        "mega_stress.png": _build_mega_stress,
        "mixed_styles_one_graph.png": _build_mixed_styles_one_graph,
        "extreme_width_range.png": _build_extreme_width_range,
        "parallel_multiedge_styles.png": _build_parallel_multiedge_styles,
    }


def _evaluate_curve(curve: CubicBezier, t: float) -> np.ndarray:
    """Evaluate a cubic bezier into a NumPy point.

    Parameters
    ----------
    curve : CubicBezier
        Curve to evaluate.
    t : float
        Parameter in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        Evaluated point with shape ``[2]``.
    """

    u = 1.0 - t
    p0 = np.asarray(curve.p0, dtype=np.float64)
    p1 = np.asarray(curve.cp1, dtype=np.float64)
    p2 = np.asarray(curve.cp2, dtype=np.float64)
    p3 = np.asarray(curve.p1, dtype=np.float64)
    point = (u**3) * p0 + 3.0 * (u**2) * t * p1 + 3.0 * u * (t**2) * p2 + (t**3) * p3
    return point


def build_edge_comparison_suite(
    output_dir: str = "eval_output/edge_comparison",
) -> EdgeComparisonResult:
    """Generate the full extended edge comparison image suite.

    Parameters
    ----------
    output_dir : str, default="eval_output/edge_comparison"
        Destination directory.

    Returns
    -------
    EdgeComparisonResult
        Paths to the generated images.
    """

    target_dir = Path(output_dir).resolve()
    builders = _build_all_builders()
    image_paths = [builders[filename](target_dir) for filename in EXPECTED_OUTPUT_FILENAMES]
    return EdgeComparisonResult(output_dir=str(target_dir), image_paths=image_paths)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the edge comparison generator.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Explicit CLI arguments. ``None`` uses ``sys.argv``.

    Returns
    -------
    int
        Process exit status.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="eval_output/edge_comparison",
        help="Directory where PNG files will be written.",
    )
    args = parser.parse_args(argv)

    result = build_edge_comparison_suite(output_dir=args.output_dir)
    print(f"Generated {len(result.image_paths)} images in {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
