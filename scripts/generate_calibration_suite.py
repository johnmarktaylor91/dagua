#!/usr/bin/env python
# ruff: noqa: E402
"""Generate a three-way rendering calibration suite.

The suite renders matching scenes with three backends:
1. Dagua's matplotlib renderer
2. Graphviz's native renderer
3. A plain matplotlib reference scene built from patches and lines

Usage
-----
python scripts/generate_calibration_suite.py
python scripts/generate_calibration_suite.py --category edge_options
python scripts/generate_calibration_suite.py --refresh-refs
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
import numpy as np
import torch
from matplotlib.colors import to_hex, to_rgba
from matplotlib.path import Path as MplPath
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (
    Arc,
    Circle,
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
)

from dagua import DaguaGraph, render
from dagua.render.edges.arrowheads import available_arrowheads
from dagua.styles import ClusterStyle, EdgeStyle, GraphStyle, NodeStyle
from dagua.utils import parse_rich_markup

DEFAULT_OUTPUT_DIR = "eval_output/calibration"
REF_CACHE_DIRNAME = ".ref_cache"
RAW_RENDER_DPI = 200
COMPARISON_DPI = 170
DEFAULT_COMPARISON_SIZE = (15.5, 5.8)
AUTO_PANEL_FILL_FRACTION = 0.8
AUTO_DATA_UNITS_PER_INCH = 74.0
AUTO_MIN_FIGSIZE = (1.4, 1.4)
WHITE = "#FFFFFF"
NODE_FILL = "#FFFFFF"
NODE_STROKE = "#2F3A47"
EDGE_COLOR = "#4A5563"
CLUSTER_FILL = "#DDE8F399"
CLUSTER_STROKE = "#64748B"
PANEL_SIZE = (980, 700)
PANEL_MARGIN = 40
CONTENT_CROP_PADDING = 12
PAIR_VERTICAL_GAP = 88.0
GRID_X_STEP = 250.0
GRID_Y_STEP = 195.0
LONG_LABEL = "This is a deliberately long label used for calibration"
CATEGORY_ORDER = [
    "edge_options",
    "node_options",
    "text_options",
    "cluster_options",
    "combinations_2way",
    "combinations_3way",
    "extreme_values",
    "scaling",
]
NODE_SHAPES = [
    "rect",
    "roundrect",
    "ellipse",
    "diamond",
    "circle",
    "triangle",
    "hexagon",
    "parallelogram",
    "pentagon",
    "octagon",
    "star",
    "cylinder",
    "trapezoid",
]


@dataclass
class GraphvizSpec:
    """Graphviz-native render overrides for a calibration scene.

    Parameters
    ----------
    engine : str, default="dot"
        Graphviz executable used for native rendering.
    graph_attrs : dict[str, str]
        Graph-level DOT attributes.
    default_node_attrs : dict[str, str]
        Default node attributes.
    default_edge_attrs : dict[str, str]
        Default edge attributes.
    node_attrs : dict[int, dict[str, str]]
        Per-node attribute overrides keyed by node index.
    edge_attrs : dict[int, dict[str, str]]
        Per-edge attribute overrides keyed by edge index.
    cluster_attrs : dict[str, dict[str, str]]
        Per-cluster attribute overrides keyed by cluster name.
    """

    engine: str = "dot"
    graph_attrs: Dict[str, str] = field(default_factory=dict)
    default_node_attrs: Dict[str, str] = field(default_factory=dict)
    default_edge_attrs: Dict[str, str] = field(default_factory=dict)
    node_attrs: Dict[int, Dict[str, str]] = field(default_factory=dict)
    edge_attrs: Dict[int, Dict[str, str]] = field(default_factory=dict)
    cluster_attrs: Dict[str, Dict[str, str]] = field(default_factory=dict)


@dataclass
class CalibrationScene:
    """Concrete scene rendered by all three comparison backends.

    Parameters
    ----------
    graph : DaguaGraph
        Graph with styles, labels, and cluster metadata.
    positions : torch.Tensor
        Fixed node positions with shape ``[N, 2]``.
    graphviz : GraphvizSpec
        Graphviz render configuration.
    figsize : tuple[float, float] | None, default=None
        Optional scene size metadata retained for callers that need to track an
        explicit composed size alongside the content-derived raw render size.
    mpl_renderer : callable, optional
        Optional custom matplotlib renderer for scenes that need bespoke text
        or geometry beyond the generic patch-based renderer.
    """

    graph: DaguaGraph
    positions: torch.Tensor
    graphviz: GraphvizSpec = field(default_factory=GraphvizSpec)
    figsize: Optional[Tuple[float, float]] = None
    mpl_renderer: Optional[Callable[[Any, "CalibrationScene"], None]] = None


@dataclass
class CalibrationCase:
    """One generated output image in the calibration suite.

    Parameters
    ----------
    case_id : str
        Stable filename stem and manifest identifier.
    category : str
        Output directory grouping.
    description : str
        Human-readable description shown above the comparison image.
    build_scene : callable
        Returns a freshly constructed scene for rendering.
    comparison_figsize : tuple[float, float], default=DEFAULT_COMPARISON_SIZE
        Final composed figure size in inches.
    """

    case_id: str
    category: str
    description: str
    build_scene: Callable[[], CalibrationScene]
    comparison_figsize: Tuple[float, float] = DEFAULT_COMPARISON_SIZE


@dataclass
class CalibrationSuiteResult:
    """Output metadata for a generated calibration suite.

    Parameters
    ----------
    output_dir : str
        Root output directory.
    manifest_path : str
        Path to the emitted JSON manifest.
    image_paths : list[str]
        Final comparison image paths.
    """

    output_dir: str
    manifest_path: str
    image_paths: List[str]


def _graphviz_available() -> bool:
    """Return whether the required Graphviz executable is available.

    Returns
    -------
    bool
        ``True`` when ``dot`` is present on ``PATH``.
    """

    return shutil.which("dot") is not None


def _base_graph_style() -> GraphStyle:
    """Create the default graph-wide render style for calibration scenes.

    Returns
    -------
    GraphStyle
        White-background graph style with compact margins.
    """

    return GraphStyle(
        background_color=WHITE,
        margin=10.0,
        min_figsize=(3.2, 3.0),
        max_figsize=(10.0, 8.0),
        title_font_size=10.0,
        edge_label_font_size=7.0,
        edge_label_background=WHITE,
        edge_label_background_opacity=1.0,
    )


def _base_node_style(**overrides: Any) -> NodeStyle:
    """Return the baseline node style used by most calibration cases.

    Parameters
    ----------
    **overrides : Any
        ``NodeStyle`` field overrides.

    Returns
    -------
    NodeStyle
        Node style with readable, neutral defaults.
    """

    defaults: Dict[str, Any] = {
        "shape": "roundrect",
        "fill": NODE_FILL,
        "stroke": NODE_STROKE,
        "stroke_width": 1.0,
        "font_size": 10.0,
        "font_color": "#111827",
        "padding": (14.0, 8.0),
        "corner_radius": 8.0,
        "opacity": 1.0,
        "min_width": 62.0,
        "min_height": 36.0,
    }
    defaults.update(overrides)
    return NodeStyle(**defaults)


def _base_edge_style(**overrides: Any) -> EdgeStyle:
    """Return the baseline edge style used by most calibration cases.

    Parameters
    ----------
    **overrides : Any
        ``EdgeStyle`` field overrides.

    Returns
    -------
    EdgeStyle
        Edge style with a neutral dark stroke and visible arrowhead.
    """

    defaults: Dict[str, Any] = {
        "color": EDGE_COLOR,
        "width": 1.2,
        "arrow": "normal",
        "arrow_fill": "filled",
        "arrow_length": 11.0,
        "arrow_width": 8.0,
        "arrow_node_fraction": 0.0,
        "opacity": 1.0,
        "style": "solid",
        "routing": "straight",
        "label_font_size": 7.0,
        "label_background": WHITE,
        "label_font_color": "#374151",
    }
    defaults.update(overrides)
    return EdgeStyle(**defaults)


def _base_cluster_style(**overrides: Any) -> ClusterStyle:
    """Return the baseline cluster style used by calibration scenes.

    Parameters
    ----------
    **overrides : Any
        ``ClusterStyle`` field overrides.

    Returns
    -------
    ClusterStyle
        Cluster style with a light fill and visible border.
    """

    defaults: Dict[str, Any] = {
        "fill": CLUSTER_FILL,
        "stroke": CLUSTER_STROKE,
        "stroke_width": 1.4,
        "padding": 40.0,
        "corner_radius": 12.0,
        "opacity": 0.7,
        "font_size": 10.0,
        "label_offset": (10.0, 16.0),
    }
    defaults.update(overrides)
    return ClusterStyle(**defaults)


def _set_graph_style(graph: DaguaGraph, graph_style: GraphStyle) -> None:
    """Apply a graph-wide style object to a graph.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to update.
    graph_style : GraphStyle
        Graph-wide style.

    Returns
    -------
    None
        Mutates ``graph`` in place.
    """

    graph._theme.graph_style = graph_style


def _grid_origins(
    count: int,
    columns: int,
    x_step: float = GRID_X_STEP,
    y_step: float = GRID_Y_STEP,
) -> List[Tuple[float, float]]:
    """Return sample-cell origins centered around the origin.

    Parameters
    ----------
    count : int
        Number of sample cells.
    columns : int
        Number of columns in the grid.
    x_step : float, default=GRID_X_STEP
        Horizontal distance between cell centers.
    y_step : float, default=GRID_Y_STEP
        Vertical distance between cell centers.

    Returns
    -------
    list[tuple[float, float]]
        Sample-cell centers ordered row-major.
    """

    rows = int(math.ceil(count / max(columns, 1)))
    x_offset = (columns - 1) * x_step / 2.0
    y_offset = (rows - 1) * y_step / 2.0
    origins: List[Tuple[float, float]] = []
    for index in range(count):
        row = index // columns
        column = index % columns
        origins.append((column * x_step - x_offset, y_offset - row * y_step))
    return origins


def _apply_manual_node_sizes(
    graph: DaguaGraph,
    sizes: Sequence[Tuple[float, float]],
    font_sizes: Optional[Sequence[float]] = None,
) -> None:
    """Pin explicit node sizes for a scene.

    Parameters
    ----------
    graph : DaguaGraph
        Graph whose node-size cache should be populated.
    sizes : sequence[tuple[float, float]]
        Explicit node widths and heights in data units.
    font_sizes : sequence[float], optional
        Optional per-node effective font sizes in points.

    Returns
    -------
    None
        Mutates ``graph`` in place.
    """

    graph.node_sizes = torch.tensor(sizes, dtype=torch.float32)
    if font_sizes is None:
        graph.node_font_sizes = torch.tensor(
            [graph.get_style_for_node(index).font_size for index in range(graph.num_nodes)],
            dtype=torch.float32,
        )
    else:
        graph.node_font_sizes = torch.tensor(font_sizes, dtype=torch.float32)
    graph._node_sizes_revision = graph.revision


def _build_graph(
    node_labels: Sequence[str],
    positions: Sequence[Tuple[float, float]],
    edges: Sequence[Tuple[int, int]],
    node_styles: Optional[Sequence[Optional[NodeStyle]]] = None,
    edge_styles: Optional[Sequence[Optional[EdgeStyle]]] = None,
    edge_labels: Optional[Sequence[Optional[str]]] = None,
    clusters: Optional[
        Sequence[Tuple[str, Sequence[int], ClusterStyle, str, Optional[str]]]
    ] = None,
    direction: str = "TB",
    graph_style: Optional[GraphStyle] = None,
    node_sizes: Optional[Sequence[Tuple[float, float]]] = None,
    node_font_sizes: Optional[Sequence[float]] = None,
    graphviz: Optional[GraphvizSpec] = None,
    mpl_renderer: Optional[Callable[[Any, CalibrationScene], None]] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> CalibrationScene:
    """Construct a calibration scene from explicit graph components.

    Parameters
    ----------
    node_labels : sequence[str]
        Node labels in node-index order.
    positions : sequence[tuple[float, float]]
        Fixed node positions.
    edges : sequence[tuple[int, int]]
        Directed edges using node indices.
    node_styles : sequence[NodeStyle | None], optional
        Per-node style overrides.
    edge_styles : sequence[EdgeStyle | None], optional
        Per-edge style overrides.
    edge_labels : sequence[str | None], optional
        Per-edge labels.
    clusters : sequence[tuple[str, sequence[int], ClusterStyle, str, str | None]], optional
        Cluster specifications ``(name, members, style, label, parent)``.
    direction : str, default="TB"
        Graph direction.
    graph_style : GraphStyle, optional
        Graph-wide render style.
    node_sizes : sequence[tuple[float, float]], optional
        Explicit node sizes.
    node_font_sizes : sequence[float], optional
        Explicit effective node font sizes.
    graphviz : GraphvizSpec, optional
        Graphviz render overrides.
    mpl_renderer : callable, optional
        Custom matplotlib scene renderer.
    figsize : tuple[float, float], optional
        Optional scene size metadata preserved on the returned scene.

    Returns
    -------
    CalibrationScene
        Fully configured scene.
    """

    graph = DaguaGraph(direction=direction)
    for index, label in enumerate(node_labels):
        style = None if node_styles is None else node_styles[index]
        graph.add_node(index, label=label, style=style)
    for edge_index, (source, target) in enumerate(edges):
        style = None if edge_styles is None else edge_styles[edge_index]
        label = None if edge_labels is None else edge_labels[edge_index]
        graph.add_edge(source, target, label=label, style=style)
    if clusters is not None:
        for name, members, style, label, parent in clusters:
            graph.add_cluster(name, list(members), style=style, label=label, parent=parent)
    _set_graph_style(graph, graph_style or _base_graph_style())
    if node_sizes is not None:
        _apply_manual_node_sizes(graph, node_sizes, font_sizes=node_font_sizes)
    return CalibrationScene(
        graph=graph,
        positions=torch.tensor(positions, dtype=torch.float32),
        graphviz=graphviz or GraphvizSpec(),
        figsize=figsize,
        mpl_renderer=mpl_renderer,
    )


def _scene_content_bounds(scene: CalibrationScene) -> Tuple[float, float, float, float]:
    """Return the scene bounds in data units, including clusters and self-loops.

    Parameters
    ----------
    scene : CalibrationScene
        Scene whose visible content should be measured.

    Returns
    -------
    tuple[float, float, float, float]
        ``(x_min, y_min, x_max, y_max)`` bounds for the visible content.
    """

    graph = scene.graph
    graph.compute_node_sizes()
    positions = scene.positions.detach().cpu().numpy()
    sizes = graph.node_sizes.detach().cpu().numpy()

    x_min = float((positions[:, 0] - sizes[:, 0] / 2.0).min())
    x_max = float((positions[:, 0] + sizes[:, 0] / 2.0).max())
    y_min = float((positions[:, 1] - sizes[:, 1] / 2.0).min())
    y_max = float((positions[:, 1] + sizes[:, 1] / 2.0).max())

    for cluster_name in graph.clusters:
        members = graph.leaf_cluster_members(cluster_name)
        if not members:
            continue
        cluster_positions = positions[members]
        cluster_sizes = sizes[members]
        style = graph.get_style_for_cluster(cluster_name)
        label_band = max(float(style.font_size), float(style.label_offset[1])) + 8.0
        x_min = min(
            x_min,
            float((cluster_positions[:, 0] - cluster_sizes[:, 0] / 2.0).min() - style.padding),
        )
        x_max = max(
            x_max,
            float((cluster_positions[:, 0] + cluster_sizes[:, 0] / 2.0).max() + style.padding),
        )
        y_min = min(
            y_min,
            float((cluster_positions[:, 1] - cluster_sizes[:, 1] / 2.0).min() - style.padding),
        )
        y_max = max(
            y_max,
            float(
                (cluster_positions[:, 1] + cluster_sizes[:, 1] / 2.0).max()
                + style.padding
                + label_band
            ),
        )

    for edge_index in range(int(graph.edge_index.shape[1])):
        source = int(graph.edge_index[0, edge_index].item())
        target = int(graph.edge_index[1, edge_index].item())
        if source != target:
            continue
        style = graph.get_style_for_edge(edge_index)
        center_x = float(positions[source, 0])
        center_y = float(positions[source, 1])
        node_width = float(sizes[source, 0])
        node_height = float(sizes[source, 1])
        loop_radius = max(node_width, node_height) * 0.6
        label_pad = float(style.label_font_size) + float(style.label_offset) + 8.0
        arrow_pad = float(style.arrow_length) + float(style.arrow_width)
        if graph.direction == "BT":
            anchor_y = center_y - node_height / 2.0
            y_min = min(y_min, anchor_y - loop_radius - label_pad)
            x_min = min(x_min, center_x - loop_radius - arrow_pad * 0.4)
            x_max = max(x_max, center_x + loop_radius + arrow_pad * 0.4)
        elif graph.direction == "LR":
            anchor_x = center_x - node_width / 2.0
            x_min = min(x_min, anchor_x - loop_radius - label_pad)
            y_min = min(y_min, center_y - loop_radius - arrow_pad * 0.4)
            y_max = max(y_max, center_y + loop_radius + arrow_pad * 0.4)
        elif graph.direction == "RL":
            anchor_x = center_x + node_width / 2.0
            x_max = max(x_max, anchor_x + loop_radius + label_pad)
            y_min = min(y_min, center_y - loop_radius - arrow_pad * 0.4)
            y_max = max(y_max, center_y + loop_radius + arrow_pad * 0.4)
        else:
            anchor_y = center_y + node_height / 2.0
            y_max = max(y_max, anchor_y + loop_radius + label_pad)
            x_min = min(x_min, center_x - loop_radius - arrow_pad * 0.4)
            x_max = max(x_max, center_x + loop_radius + arrow_pad * 0.4)

    margin = float(graph.graph_style.margin)
    return x_min - margin, y_min - margin, x_max + margin, y_max + margin


def _resolved_scene_figsize(scene: CalibrationScene) -> Tuple[float, float]:
    """Return an auto-sized raw figure size for one scene.

    Parameters
    ----------
    scene : CalibrationScene
        Scene to size.

    Returns
    -------
    tuple[float, float]
        Raw render figure size in inches.
    """

    x_min, y_min, x_max, y_max = _scene_content_bounds(scene)
    width = max(x_max - x_min, 1.0)
    height = max(y_max - y_min, 1.0)
    scale = AUTO_DATA_UNITS_PER_INCH * AUTO_PANEL_FILL_FRACTION
    fig_width = width / scale
    fig_height = height / scale

    min_width, min_height = AUTO_MIN_FIGSIZE
    if fig_width < min_width or fig_height < min_height:
        grow = max(min_width / max(fig_width, 1e-6), min_height / max(fig_height, 1e-6), 1.0)
        fig_width *= grow
        fig_height *= grow

    # Raw Dagua renders always size from content bounds so scaling cases can
    # reuse one render geometry while varying only the composed comparison size.
    max_width, max_height = scene.graph.graph_style.max_figsize
    if fig_width > max_width or fig_height > max_height:
        shrink = min(max_width / fig_width, max_height / fig_height)
        fig_width *= shrink
        fig_height *= shrink

    return fig_width, fig_height


def _content_bounded_figsize(scene: CalibrationScene, figure_width: float) -> Tuple[float, float]:
    """Return a raw figure size that matches the scene content aspect ratio.

    Parameters
    ----------
    scene : CalibrationScene
        Scene whose visible content should determine the aspect ratio.
    figure_width : float
        Requested figure width in inches.

    Returns
    -------
    tuple[float, float]
        Figure size whose height is derived from the content bounds so the
        rendered graph fills the canvas instead of sitting inside extra space.
    """

    x_min, y_min, x_max, y_max = _scene_content_bounds(scene)
    content_width = max(x_max - x_min, 1.0)
    content_height = max(y_max - y_min, 1.0)
    aspect_ratio = content_height / content_width
    return figure_width, max(figure_width * aspect_ratio, AUTO_MIN_FIGSIZE[1])


def _edge_scene(
    samples: Sequence[Tuple[str, EdgeStyle]],
    columns: int,
    base_node_style: Optional[NodeStyle] = None,
    figsize: Optional[Tuple[float, float]] = None,
    graphviz_splines: str = "polyline",
) -> CalibrationScene:
    """Build a grid of disconnected source-target edge samples.

    Parameters
    ----------
    samples : sequence[tuple[str, EdgeStyle]]
        Pair caption and edge style per sample.
    columns : int
        Number of grid columns.
    base_node_style : NodeStyle, optional
        Shared node style.
    figsize : tuple[float, float], optional
        Optional scene size metadata preserved on the returned scene.
    graphviz_splines : str, default="polyline"
        Graphviz ``splines`` attribute.

    Returns
    -------
    CalibrationScene
        Scene containing one vertical pair per sample.
    """

    node_labels: List[str] = []
    positions: List[Tuple[float, float]] = []
    edges: List[Tuple[int, int]] = []
    node_styles: List[Optional[NodeStyle]] = []
    edge_styles: List[Optional[EdgeStyle]] = []
    edge_labels: List[Optional[str]] = []

    for sample_index, (caption, style) in enumerate(samples):
        origin_x, origin_y = _grid_origins(len(samples), columns)[sample_index]
        source_index = sample_index * 2
        target_index = source_index + 1
        node_labels.extend(["A", "B"])
        positions.extend(
            [
                (origin_x, origin_y + PAIR_VERTICAL_GAP / 2.0),
                (origin_x, origin_y - PAIR_VERTICAL_GAP / 2.0),
            ]
        )
        shared_style = base_node_style or _base_node_style()
        node_styles.extend([shared_style, shared_style])
        edges.append((source_index, target_index))
        edge_styles.append(style)
        edge_labels.append(caption)

    width = max(6.0, min(10.0, columns * 2.8))
    rows = int(math.ceil(len(samples) / max(columns, 1)))
    raw_figsize = figsize or (width, max(4.5, rows * 2.3))
    return _build_graph(
        node_labels=node_labels,
        positions=positions,
        edges=edges,
        node_styles=node_styles,
        edge_styles=edge_styles,
        edge_labels=edge_labels,
        figsize=raw_figsize,
        graphviz=GraphvizSpec(graph_attrs={"splines": graphviz_splines}),
    )


def _node_scene(
    samples: Sequence[Tuple[str, NodeStyle]],
    columns: int,
    figsize: Optional[Tuple[float, float]] = None,
    node_sizes: Optional[Sequence[Tuple[float, float]]] = None,
    node_font_sizes: Optional[Sequence[float]] = None,
    mpl_renderer: Optional[Callable[[Any, CalibrationScene], None]] = None,
    graphviz: Optional[GraphvizSpec] = None,
) -> CalibrationScene:
    """Build a grid of single-node samples.

    Parameters
    ----------
    samples : sequence[tuple[str, NodeStyle]]
        Node label and style per sample.
    columns : int
        Number of grid columns.
    figsize : tuple[float, float], optional
        Optional scene size metadata preserved on the returned scene.
    node_sizes : sequence[tuple[float, float]], optional
        Optional explicit node sizes aligned with ``samples``.
    node_font_sizes : sequence[float], optional
        Optional explicit node font sizes aligned with ``samples``.
    mpl_renderer : callable, optional
        Custom matplotlib renderer.
    graphviz : GraphvizSpec, optional
        Graphviz render overrides.

    Returns
    -------
    CalibrationScene
        Scene containing one node per sample.
    """

    origins = _grid_origins(len(samples), columns)
    raw_figsize = figsize or (
        max(6.0, min(10.0, columns * 2.7)),
        max(4.5, math.ceil(len(samples) / max(columns, 1)) * 2.2),
    )
    return _build_graph(
        node_labels=[label for label, _ in samples],
        positions=origins,
        edges=[],
        node_styles=[style for _, style in samples],
        node_sizes=node_sizes,
        node_font_sizes=node_font_sizes,
        figsize=raw_figsize,
        mpl_renderer=mpl_renderer,
        graphviz=graphviz,
    )


def _graphviz_color(color: str, alpha: float = 1.0) -> str:
    """Convert a matplotlib-compatible color into Graphviz RGBA hex.

    Parameters
    ----------
    color : str
        Any color accepted by ``matplotlib.colors.to_rgba``.
    alpha : float, default=1.0
        Extra multiplicative alpha.

    Returns
    -------
    str
        Graphviz-compatible ``#RRGGBBAA`` color string.
    """

    red, green, blue, rgba_alpha = to_rgba(color)
    return to_hex((red, green, blue, max(0.0, min(1.0, rgba_alpha * alpha))), keep_alpha=True)


def _format_dot_value(value: Any) -> str:
    """Format one value for DOT output.

    Parameters
    ----------
    value : Any
        DOT attribute value.

    Returns
    -------
    str
        Serialized DOT literal.
    """

    if isinstance(value, (int, float)):
        return str(value)
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _format_dot_attrs(attrs: Mapping[str, Any]) -> str:
    """Format a DOT attribute dictionary.

    Parameters
    ----------
    attrs : Mapping[str, Any]
        Attribute mapping.

    Returns
    -------
    str
        DOT attribute block, or an empty string when ``attrs`` is empty.
    """

    if not attrs:
        return ""
    parts = [f"{key}={_format_dot_value(value)}" for key, value in attrs.items()]
    return " [" + ", ".join(parts) + "]"


def _node_shape_attrs(style: NodeStyle) -> Dict[str, str]:
    """Map a Dagua node shape into Graphviz node attributes.

    Parameters
    ----------
    style : NodeStyle
        Node style to map.

    Returns
    -------
    dict[str, str]
        Closest Graphviz shape attributes.
    """

    if style.shape == "rect":
        return {"shape": "box"}
    if style.shape == "roundrect":
        return {"shape": "box", "style": "rounded,filled"}
    if style.shape == "trapezoid":
        return {"shape": "trapezium"}
    return {"shape": style.shape}


def _edge_arrow_attr(style: EdgeStyle) -> str:
    """Map a Dagua arrow style into the closest Graphviz arrowhead token.

    Parameters
    ----------
    style : EdgeStyle
        Edge style to map.

    Returns
    -------
    str
        Graphviz arrowhead token.
    """

    if style.arrow == "circle":
        return "odot"
    return style.arrow


def _style_tokens(line_style: str, *, filled: bool, rounded: bool = False) -> str:
    """Compose a Graphviz ``style=`` token string.

    Parameters
    ----------
    line_style : str
        Base line-style token such as ``solid`` or ``dashed``.
    filled : bool
        Whether to include ``filled``.
    rounded : bool, default=False
        Whether to include ``rounded``.

    Returns
    -------
    str
        Comma-separated Graphviz style token string.
    """

    tokens: List[str] = []
    if filled:
        tokens.append("filled")
    if rounded:
        tokens.append("rounded")
    if line_style in {"dashed", "dotted", "solid"}:
        tokens.append(line_style)
    return ",".join(tokens)


def _cluster_label_attrs(style: ClusterStyle) -> Dict[str, str]:
    """Return Graphviz cluster label placement attributes.

    Parameters
    ----------
    style : ClusterStyle
        Cluster style.

    Returns
    -------
    dict[str, str]
        DOT attributes affecting cluster-label placement.
    """

    attrs = {"labelloc": "t"}
    if style.label_position == "top-left":
        attrs["labeljust"] = "l"
    elif style.label_position == "top-right":
        attrs["labeljust"] = "r"
    else:
        attrs["labeljust"] = "c"
    return attrs


def _build_graphviz_dot(scene: CalibrationScene) -> str:
    """Serialize one calibration scene into Graphviz DOT.

    Parameters
    ----------
    scene : CalibrationScene
        Scene to serialize.

    Returns
    -------
    str
        DOT source with explicit node positions available for fixed-position
        Graphviz engines when needed.
    """

    graph = scene.graph
    graph.compute_node_sizes()
    positions = scene.positions.detach().cpu().numpy()
    sizes = graph.node_sizes.detach().cpu().numpy()

    graph_attrs = {
        "bgcolor": WHITE,
        "outputorder": "edgesfirst",
        "overlap": "false",
        "splines": "polyline",
    }
    graph_attrs.update(scene.graphviz.graph_attrs)

    node_defaults = {
        "fontname": _base_node_style().font_family,
        "fontsize": str(_base_node_style().font_size),
        "margin": "0.05,0.03",
        "fixedsize": "true",
    }
    node_defaults.update(scene.graphviz.default_node_attrs)

    edge_defaults = {
        "fontname": _base_node_style().font_family,
        "fontsize": str(_base_edge_style().label_font_size),
        "decorate": "false",
    }
    edge_defaults.update(scene.graphviz.default_edge_attrs)

    children: Dict[Optional[str], List[str]] = {}
    for name in graph.clusters:
        children.setdefault(graph.cluster_parents.get(name), []).append(name)
    for cluster_list in children.values():
        cluster_list.sort()

    emitted: set[int] = set()
    lines = ["digraph G {"]
    for key, value in graph_attrs.items():
        lines.append(f"  {key}={_format_dot_value(value)};")
    lines.append(f"  node{_format_dot_attrs(node_defaults)};")
    lines.append(f"  edge{_format_dot_attrs(edge_defaults)};")

    def emit_cluster(name: str, indent: int = 1) -> None:
        """Emit one cluster block recursively."""
        prefix = "  " * indent
        style = graph.cluster_styles.get(name, _base_cluster_style())
        members = graph.clusters[name]
        attrs: Dict[str, Any] = {
            "label": graph.cluster_labels.get(name, name),
            "color": _graphviz_color(style.stroke),
            "fillcolor": _graphviz_color(style.fill, style.opacity),
            "penwidth": f"{style.stroke_width:.2f}",
            "style": _style_tokens(style.stroke_dash, filled=True, rounded=style.corner_radius > 0),
            "margin": f"{style.padding / 72.0:.3f}",
            "fontname": style.font_family or _base_node_style().font_family,
            "fontsize": f"{style.font_size:.2f}",
            "fontcolor": _graphviz_color(style.font_color),
        }
        attrs.update(_cluster_label_attrs(style))
        attrs.update(scene.graphviz.cluster_attrs.get(name, {}))
        lines.append(f"{prefix}subgraph cluster_{name} {{")
        for key, value in attrs.items():
            lines.append(f"{prefix}  {key}={_format_dot_value(value)};")
        for child in children.get(name, []):
            emit_cluster(child, indent + 1)
        for node_index in members:
            if node_index in emitted:
                continue
            node_attrs = _graphviz_node_attrs(
                graph, node_index, positions[node_index], sizes[node_index], scene
            )
            lines.append(f"{prefix}  n{node_index}{_format_dot_attrs(node_attrs)};")
            emitted.add(node_index)
        lines.append(f"{prefix}}}")

    for root_cluster in children.get(None, []):
        emit_cluster(root_cluster)

    for node_index in range(graph.num_nodes):
        if node_index in emitted:
            continue
        node_attrs = _graphviz_node_attrs(
            graph, node_index, positions[node_index], sizes[node_index], scene
        )
        lines.append(f"  n{node_index}{_format_dot_attrs(node_attrs)};")

    for edge_index in range(int(graph.edge_index.shape[1])):
        source = int(graph.edge_index[0, edge_index].item())
        target = int(graph.edge_index[1, edge_index].item())
        edge_attrs = _graphviz_edge_attrs(graph, edge_index, scene)
        lines.append(f"  n{source} -> n{target}{_format_dot_attrs(edge_attrs)};")

    lines.append("}")
    return "\n".join(lines)


def _graphviz_node_attrs(
    graph: DaguaGraph,
    node_index: int,
    position: np.ndarray,
    size: np.ndarray,
    scene: CalibrationScene,
) -> Dict[str, Any]:
    """Build DOT attributes for one node.

    Parameters
    ----------
    graph : DaguaGraph
        Scene graph.
    node_index : int
        Node index.
    position : numpy.ndarray
        Node position with shape ``[2]``.
    size : numpy.ndarray
        Node size with shape ``[2]``.
    scene : CalibrationScene
        Parent scene.

    Returns
    -------
    dict[str, Any]
        DOT attributes for the node.
    """

    style = graph.get_style_for_node(node_index)
    attrs: Dict[str, Any] = {
        "label": graph.node_labels[node_index],
        "fillcolor": _graphviz_color(style.fill, style.opacity),
        "color": _graphviz_color(style.stroke, style.opacity * style.border_opacity),
        "fontcolor": _graphviz_color(style.font_color),
        "fontsize": f"{style.font_size:.2f}",
        "fontname": style.font_family,
        "penwidth": f"{style.stroke_width:.2f}",
        "width": f"{float(size[0]) / 72.0:.3f}",
        "height": f"{float(size[1]) / 72.0:.3f}",
        "pos": f"{float(position[0]) / 72.0:.3f},{-float(position[1]) / 72.0:.3f}!",
    }
    attrs.update(_node_shape_attrs(style))
    if style.shape == "roundrect":
        attrs["style"] = _style_tokens(style.stroke_dash, filled=True, rounded=True)
    else:
        attrs["style"] = _style_tokens(style.stroke_dash, filled=True)
    attrs.update(scene.graphviz.node_attrs.get(node_index, {}))
    return attrs


def _graphviz_edge_attrs(
    graph: DaguaGraph,
    edge_index: int,
    scene: CalibrationScene,
) -> Dict[str, Any]:
    """Build DOT attributes for one edge.

    Parameters
    ----------
    graph : DaguaGraph
        Scene graph.
    edge_index : int
        Edge index.
    scene : CalibrationScene
        Parent scene.

    Returns
    -------
    dict[str, Any]
        DOT attributes for the edge.
    """

    style = graph.get_style_for_edge(edge_index)
    attrs: Dict[str, Any] = {
        "color": _graphviz_color(style.color, style.opacity),
        "fontcolor": _graphviz_color(style.label_font_color),
        "fontsize": f"{style.label_font_size:.2f}",
        "penwidth": f"{style.width:.2f}",
        "arrowhead": _edge_arrow_attr(style),
        "arrowsize": f"{max(style.arrow_length / 10.0, 0.3):.2f}",
    }
    if style.arrow == "none":
        attrs["arrowhead"] = "none"
    if style.style in {"dashed", "dotted", "solid"}:
        attrs["style"] = style.style
    if graph.edge_labels[edge_index]:
        attrs["label"] = graph.edge_labels[edge_index]
    if style.routing == "straight":
        attrs["constraint"] = "true"
    attrs.update(scene.graphviz.edge_attrs.get(edge_index, {}))
    return attrs


def _render_graphviz_png(dot_source: str, output_path: Path, engine: str) -> None:
    """Render a DOT scene to a PNG using Graphviz.

    Parameters
    ----------
    dot_source : str
        DOT source code.
    output_path : Path
        Destination PNG path.
    engine : str
        Graphviz executable.

    Returns
    -------
    None
        Writes the PNG to ``output_path``.
    """

    has_positions = _dot_has_pos_for_all_nodes(dot_source)
    resolved_engine = engine
    command = [resolved_engine]
    if engine != "dot" and has_positions:
        command.append("-n2")
    else:
        resolved_engine = "dot"
        command = [resolved_engine]
    result = subprocess.run(
        [*command, f"-Gdpi={RAW_RENDER_DPI}", "-Tpng", "-o", str(output_path)],
        input=dot_source,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"Graphviz {resolved_engine} render failed")


def _dot_has_pos_for_all_nodes(dot_source: str) -> bool:
    """Return whether every node declaration in a DOT string carries ``pos=``.

    Parameters
    ----------
    dot_source : str
        DOT source string.

    Returns
    -------
    bool
        ``True`` when every emitted node line includes ``pos=``.
    """

    node_lines = [
        line for line in dot_source.splitlines() if re.match(r"^\s*n\d+\s*\[", line) is not None
    ]
    return bool(node_lines) and all("pos=" in line for line in node_lines)


def _render_dagua_png(scene: CalibrationScene, output_path: Path) -> None:
    """Render the Dagua panel for one scene.

    Parameters
    ----------
    scene : CalibrationScene
        Scene to render.
    output_path : Path
        Destination PNG path.

    Returns
    -------
    None
        Writes the PNG to ``output_path``.
    """

    scene.graph.compute_node_sizes()
    fig, _ = render(
        scene.graph,
        scene.positions,
        output=str(output_path),
        figsize=_resolved_scene_figsize(scene),
        dpi=RAW_RENDER_DPI,
    )
    plt.close(fig)


def _node_patch(style: NodeStyle, x: float, y: float, width: float, height: float) -> Any:
    """Build a plain matplotlib patch for one node.

    Parameters
    ----------
    style : NodeStyle
        Node style.
    x : float
        Node center x-coordinate.
    y : float
        Node center y-coordinate.
    width : float
        Node width in data units.
    height : float
        Node height in data units.

    Returns
    -------
    Any
        Matplotlib patch.
    """

    facecolor = to_rgba(style.fill, style.opacity)
    edgecolor = to_rgba(style.stroke, style.opacity * style.border_opacity)
    linewidth = style.stroke_width
    linestyle = {
        "solid": "-",
        "dashed": "--",
        "dotted": ":",
        "dashdot": "-.",
    }.get(style.stroke_dash, "-")
    if style.shape in {"rect", "roundrect"}:
        return FancyBboxPatch(
            (x - width / 2.0, y - height / 2.0),
            width,
            height,
            boxstyle=(
                f"round,pad=0,rounding_size={style.corner_radius}"
                if style.shape == "roundrect"
                else "square,pad=0"
            ),
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            linestyle=linestyle,
        )
    if style.shape == "ellipse":
        return Ellipse(
            (x, y),
            width,
            height,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            linestyle=linestyle,
        )
    if style.shape == "circle":
        return Circle(
            (x, y),
            radius=max(width, height) / 2.0,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            linestyle=linestyle,
        )
    if style.shape == "diamond":
        return Polygon(
            [
                (x, y + height / 2.0),
                (x + width / 2.0, y),
                (x, y - height / 2.0),
                (x - width / 2.0, y),
            ],
            closed=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            linestyle=linestyle,
        )
    if style.shape == "triangle":
        return Polygon(
            [
                (x, y + height / 2.0),
                (x + width / 2.0, y - height / 2.0),
                (x - width / 2.0, y - height / 2.0),
            ],
            closed=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            linestyle=linestyle,
        )
    if style.shape == "hexagon":
        return Polygon(
            _regular_polygon_vertices(6, x, y, width, height),
            closed=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            linestyle=linestyle,
        )
    if style.shape == "pentagon":
        return Polygon(
            _regular_polygon_vertices(5, x, y, width, height),
            closed=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            linestyle=linestyle,
        )
    if style.shape == "octagon":
        return Polygon(
            _regular_polygon_vertices(8, x, y, width, height, rotation=np.pi / 8.0),
            closed=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            linestyle=linestyle,
        )
    if style.shape == "star":
        return Polygon(
            _star_vertices(x, y, width, height),
            closed=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            linestyle=linestyle,
            joinstyle="round",
        )
    if style.shape == "parallelogram":
        skew = width * 0.28
        return Polygon(
            [
                (x - width / 2.0 + skew, y + height / 2.0),
                (x + width / 2.0, y + height / 2.0),
                (x + width / 2.0 - skew, y - height / 2.0),
                (x - width / 2.0, y - height / 2.0),
            ],
            closed=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            linestyle=linestyle,
        )
    if style.shape == "trapezoid":
        inset = width * 0.28
        return Polygon(
            [
                (x - width / 2.0 + inset, y + height / 2.0),
                (x + width / 2.0 - inset, y + height / 2.0),
                (x + width / 2.0, y - height / 2.0),
                (x - width / 2.0, y - height / 2.0),
            ],
            closed=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            linestyle=linestyle,
        )
    cap_height = max(height * 0.16, 1.0)
    path = MplPath(
        [
            (x - width / 2.0, y + height / 2.0 - cap_height),
            (x - width / 2.0, y + height / 2.0),
            (x + width / 2.0, y + height / 2.0),
            (x + width / 2.0, y + height / 2.0 - cap_height),
            (x + width / 2.0, y - height / 2.0 + cap_height),
            (x + width / 2.0, y - height / 2.0),
            (x - width / 2.0, y - height / 2.0),
            (x - width / 2.0, y - height / 2.0 + cap_height),
            (x - width / 2.0, y + height / 2.0 - cap_height),
        ],
        [
            MplPath.MOVETO,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.LINETO,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CLOSEPOLY,
        ],
    )
    return matplotlib.patches.PathPatch(
        path,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
    )


def _regular_polygon_vertices(
    count: int,
    x: float,
    y: float,
    width: float,
    height: float,
    rotation: float = np.pi / 2.0,
) -> np.ndarray:
    """Return regular-polygon vertices for a node patch.

    Parameters
    ----------
    count : int
        Number of polygon corners.
    x : float
        Center x-coordinate.
    y : float
        Center y-coordinate.
    width : float
        Polygon width.
    height : float
        Polygon height.
    rotation : float, default=pi/2
        Initial rotation in radians.

    Returns
    -------
    numpy.ndarray
        Vertices with shape ``[count, 2]``.
    """

    angles = rotation + 2.0 * np.pi * np.arange(count, dtype=np.float64) / count
    return np.column_stack([x + width / 2.0 * np.cos(angles), y + height / 2.0 * np.sin(angles)])


def _star_vertices(x: float, y: float, width: float, height: float) -> np.ndarray:
    """Return five-point-star vertices for a node patch.

    Parameters
    ----------
    x : float
        Center x-coordinate.
    y : float
        Center y-coordinate.
    width : float
        Star width.
    height : float
        Star height.

    Returns
    -------
    numpy.ndarray
        Vertices with shape ``[10, 2]``.
    """

    vertices: List[Tuple[float, float]] = []
    outer_rx = width / 2.0
    outer_ry = height / 2.0
    inner_rx = outer_rx * 0.32
    inner_ry = outer_ry * 0.32
    for index in range(10):
        angle = np.pi / 2.0 + index * np.pi / 5.0
        radius_x = outer_rx if index % 2 == 0 else inner_rx
        radius_y = outer_ry if index % 2 == 0 else inner_ry
        vertices.append((x + radius_x * np.cos(angle), y + radius_y * np.sin(angle)))
    return np.asarray(vertices, dtype=np.float64)


def _edge_line_style(style: EdgeStyle) -> str:
    """Map a Dagua edge style token into a matplotlib line style.

    Parameters
    ----------
    style : EdgeStyle
        Edge style.

    Returns
    -------
    str
        Matplotlib line-style token.
    """

    return {
        "solid": "-",
        "dashed": "--",
        "dotted": ":",
        "dashdot": "-.",
    }.get(style.style, "-")


def _arrow_polygon(
    tip: Tuple[float, float],
    direction: np.ndarray,
    length: float,
    width: float,
    shape: str,
) -> np.ndarray:
    """Return polygon vertices for a manual arrowhead approximation.

    Parameters
    ----------
    tip : tuple[float, float]
        Arrow tip position.
    direction : numpy.ndarray
        Unit vector pointing toward the arrow tip.
    length : float
        Arrowhead length in data units.
    width : float
        Arrowhead width in data units.
    shape : str
        Arrowhead name.

    Returns
    -------
    numpy.ndarray
        Polygon vertices.
    """

    ux, uy = direction
    px, py = -uy, ux
    tip_x, tip_y = tip
    back_x = tip_x - ux * length
    back_y = tip_y - uy * length
    half = width / 2.0
    if shape in {"normal", "simple", "fancy", "wedge"}:
        return np.asarray(
            [
                (tip_x, tip_y),
                (back_x + px * half, back_y + py * half),
                (back_x - px * half, back_y - py * half),
            ],
            dtype=np.float64,
        )
    if shape == "diamond" or shape == "odiamond":
        mid_x = tip_x - ux * (length / 2.0)
        mid_y = tip_y - uy * (length / 2.0)
        return np.asarray(
            [
                (tip_x, tip_y),
                (mid_x + px * half, mid_y + py * half),
                (back_x, back_y),
                (mid_x - px * half, mid_y - py * half),
            ],
            dtype=np.float64,
        )
    if shape in {"box", "obox"}:
        front_x = tip_x - ux * (length * 0.15)
        front_y = tip_y - uy * (length * 0.15)
        box_back_x = front_x - ux * length
        box_back_y = front_y - uy * length
        return np.asarray(
            [
                (front_x + px * half, front_y + py * half),
                (front_x - px * half, front_y - py * half),
                (box_back_x - px * half, box_back_y - py * half),
                (box_back_x + px * half, box_back_y + py * half),
            ],
            dtype=np.float64,
        )
    return np.asarray(
        [
            (tip_x, tip_y),
            (back_x + px * half, back_y + py * half),
            (back_x - px * half, back_y - py * half),
        ],
        dtype=np.float64,
    )


def _draw_manual_arrowhead(
    ax: Any,
    tip: Tuple[float, float],
    direction: np.ndarray,
    style: EdgeStyle,
    *,
    length_scale: float = 1.0,
) -> None:
    """Draw a manual arrowhead approximation with matplotlib patches.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    tip : tuple[float, float]
        Arrow tip position.
    direction : numpy.ndarray
        Unit vector pointing toward the arrow tip.
    style : EdgeStyle
        Edge style controlling marker appearance.
    length_scale : float, default=1.0
        Extra scaling factor used by some extreme cases.

    Returns
    -------
    None
        Mutates ``ax`` by adding one or more marker artists.
    """

    if style.arrow == "none":
        return
    color = to_rgba(style.arrow_color or style.color, 1.0)
    length = float(style.arrow_length) * length_scale
    width = float(style.arrow_width) * length_scale
    head = style.arrow
    tip_x, tip_y = tip
    ux, uy = direction
    px, py = -uy, ux

    if head in {"dot", "odot", "circle"}:
        circle = Circle(
            (tip_x - ux * length * 0.45, tip_y - uy * length * 0.45),
            radius=max(width * 0.28, 2.5),
            facecolor=color if head == "dot" else "none",
            edgecolor=color,
            linewidth=max(style.width, 1.0),
        )
        ax.add_patch(circle)
        return
    if head in {"open", "vee", "curve", "icurve"}:
        spread = 0.6 if head != "vee" else 0.8
        back_x = tip_x - ux * length
        back_y = tip_y - uy * length
        left = [
            (tip_x, tip_y),
            (back_x + px * width * spread / 2.0, back_y + py * width * spread / 2.0),
        ]
        right = [
            (tip_x, tip_y),
            (back_x - px * width * spread / 2.0, back_y - py * width * spread / 2.0),
        ]
        if head in {"curve", "icurve"}:
            patch_left = FancyArrowPatch(
                posA=left[1],
                posB=left[0],
                arrowstyle="-",
                connectionstyle=f"arc3,rad={0.25 if head == 'curve' else -0.25}",
                color=color,
                linewidth=max(style.width, 1.0),
            )
            patch_right = FancyArrowPatch(
                posA=right[1],
                posB=right[0],
                arrowstyle="-",
                connectionstyle=f"arc3,rad={-0.25 if head == 'curve' else 0.25}",
                color=color,
                linewidth=max(style.width, 1.0),
            )
            ax.add_patch(patch_left)
            ax.add_patch(patch_right)
            return
        ax.plot(
            [left[0][0], left[1][0]],
            [left[0][1], left[1][1]],
            color=color,
            linewidth=max(style.width, 1.0),
        )
        ax.plot(
            [right[0][0], right[1][0]],
            [right[0][1], right[1][1]],
            color=color,
            linewidth=max(style.width, 1.0),
        )
        return
    if head in {"tee", "bar"}:
        bar_center_x = tip_x - ux * length * 0.45
        bar_center_y = tip_y - uy * length * 0.45
        ax.plot(
            [bar_center_x + px * width / 2.0, bar_center_x - px * width / 2.0],
            [bar_center_y + py * width / 2.0, bar_center_y - py * width / 2.0],
            color=color,
            linewidth=max(style.width * 1.2, 1.0),
        )
        return
    if head == "crow":
        back_x = tip_x - ux * length
        back_y = tip_y - uy * length
        for offset in (-0.55, 0.0, 0.55):
            ax.plot(
                [tip_x, back_x + px * width * offset],
                [tip_y, back_y + py * width * offset],
                color=color,
                linewidth=max(style.width, 1.0),
            )
        return
    if head == "bracket":
        back_x = tip_x - ux * length
        back_y = tip_y - uy * length
        left_x = back_x + px * width / 2.0
        left_y = back_y + py * width / 2.0
        right_x = back_x - px * width / 2.0
        right_y = back_y - py * width / 2.0
        ax.plot([left_x, right_x], [left_y, right_y], color=color, linewidth=max(style.width, 1.0))
        ax.plot([left_x, tip_x], [left_y, tip_y], color=color, linewidth=max(style.width, 1.0))
        ax.plot([right_x, tip_x], [right_y, tip_y], color=color, linewidth=max(style.width, 1.0))
        return
    if head == "inv":
        polygon = _arrow_polygon(tip, -direction, length, width, "normal")
        ax.add_patch(
            Polygon(
                polygon,
                closed=True,
                facecolor=color if style.arrow_fill == "filled" else "none",
                edgecolor=color,
                linewidth=max(style.width, 1.0),
            )
        )
        return

    polygon = _arrow_polygon(tip, direction, length, width, head)
    ax.add_patch(
        Polygon(
            polygon,
            closed=True,
            facecolor=color
            if head not in {"open", "obox", "odiamond"} and style.arrow_fill == "filled"
            else "none",
            edgecolor=color,
            linewidth=max(style.width, 1.0),
            joinstyle="round",
        )
    )


def _edge_midpoint(points: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    """Return the midpoint of a polyline.

    Parameters
    ----------
    points : sequence[tuple[float, float]]
        Polyline control points.

    Returns
    -------
    tuple[float, float]
        Midpoint of the longest segment or the center segment.
    """

    if len(points) == 2:
        return ((points[0][0] + points[1][0]) / 2.0, (points[0][1] + points[1][1]) / 2.0)
    segment_lengths = [
        math.hypot(points[index + 1][0] - points[index][0], points[index + 1][1] - points[index][1])
        for index in range(len(points) - 1)
    ]
    longest = int(np.argmax(segment_lengths))
    start = points[longest]
    end = points[longest + 1]
    return ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)


def _draw_generic_scene(ax: Any, scene: CalibrationScene, *, draw_node_labels: bool = True) -> None:
    """Draw a calibration scene with plain matplotlib primitives.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    scene : CalibrationScene
        Scene to render.
    draw_node_labels : bool, default=True
        Whether node labels should be drawn.

    Returns
    -------
    None
        Mutates ``ax`` in place.
    """

    graph = scene.graph
    graph.compute_node_sizes()
    positions = scene.positions.detach().cpu().numpy()
    sizes = graph.node_sizes.detach().cpu().numpy()

    ax.set_facecolor(WHITE)
    ax.axis("off")
    ax.set_aspect("equal")

    x_min, y_min, x_max, y_max = _scene_content_bounds(scene)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    _draw_generic_clusters(ax, scene, positions, sizes)
    _draw_generic_edges(ax, scene, positions, sizes)
    _draw_generic_nodes(ax, scene, positions, sizes, draw_node_labels=draw_node_labels)


def _draw_generic_clusters(
    ax: Any,
    scene: CalibrationScene,
    positions: np.ndarray,
    sizes: np.ndarray,
) -> None:
    """Draw cluster boxes for the generic matplotlib renderer.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    scene : CalibrationScene
        Scene to render.
    positions : numpy.ndarray
        Node positions with shape ``[N, 2]``.
    sizes : numpy.ndarray
        Node sizes with shape ``[N, 2]``.

    Returns
    -------
    None
        Mutates ``ax`` in place.
    """

    graph = scene.graph
    if not graph.clusters:
        return
    names = sorted(graph.clusters, key=lambda name: graph.cluster_depth(name))
    for name in names:
        members = graph.clusters[name]
        member_positions = positions[list(members)]
        member_sizes = sizes[list(members)]
        style = graph.cluster_styles.get(name, _base_cluster_style())
        padding = float(style.padding)
        x_min = float((member_positions[:, 0] - member_sizes[:, 0] / 2.0).min() - padding)
        x_max = float((member_positions[:, 0] + member_sizes[:, 0] / 2.0).max() + padding)
        y_min = float((member_positions[:, 1] - member_sizes[:, 1] / 2.0).min() - padding)
        y_max = float((member_positions[:, 1] + member_sizes[:, 1] / 2.0).max() + padding)
        patch = FancyBboxPatch(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            boxstyle=f"round,pad=0,rounding_size={style.corner_radius}",
            facecolor=to_rgba(style.fill, style.opacity),
            edgecolor=to_rgba(style.stroke, style.opacity),
            linewidth=style.stroke_width,
            linestyle={"solid": "-", "dashed": "--", "dotted": ":"}.get(style.stroke_dash, "-"),
            zorder=0.8,
        )
        ax.add_patch(patch)
        label_x = x_min + style.label_offset[0]
        if style.label_position == "top-center":
            label_x = (x_min + x_max) / 2.0
        elif style.label_position == "top-right":
            label_x = x_max - style.label_offset[0]
        ha = (
            "left"
            if style.label_position == "top-left"
            else "center"
            if style.label_position == "top-center"
            else "right"
        )
        ax.text(
            label_x,
            y_max - style.label_offset[1],
            graph.cluster_labels.get(name, name),
            fontsize=style.font_size,
            fontweight=style.font_weight,
            color=style.font_color,
            ha=ha,
            va="center",
            zorder=0.9,
        )


def _draw_generic_edges(
    ax: Any,
    scene: CalibrationScene,
    positions: np.ndarray,
    sizes: np.ndarray,
) -> None:
    """Draw edges for the generic matplotlib renderer.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    scene : CalibrationScene
        Scene to render.
    positions : numpy.ndarray
        Node positions with shape ``[N, 2]``.
    sizes : numpy.ndarray
        Node sizes with shape ``[N, 2]``.

    Returns
    -------
    None
        Mutates ``ax`` in place.
    """

    graph = scene.graph
    group_counts: Dict[Tuple[int, int], List[int]] = {}
    for edge_index in range(int(graph.edge_index.shape[1])):
        source = int(graph.edge_index[0, edge_index].item())
        target = int(graph.edge_index[1, edge_index].item())
        group_counts.setdefault((source, target), []).append(edge_index)

    for edge_index in range(int(graph.edge_index.shape[1])):
        style = graph.get_style_for_edge(edge_index)
        source = int(graph.edge_index[0, edge_index].item())
        target = int(graph.edge_index[1, edge_index].item())
        source_pos = positions[source]
        target_pos = positions[target]
        label = graph.edge_labels[edge_index]
        siblings = group_counts[(source, target)]
        sibling_rank = siblings.index(edge_index)
        group_offset = sibling_rank - (len(siblings) - 1) / 2.0
        color = to_rgba(style.color, style.opacity)
        linewidth = max(style.width, 0.25)
        linestyle = _edge_line_style(style)

        if source == target:
            radius = max(float(sizes[source, 0]), float(sizes[source, 1])) * 0.55 + 18.0 * (
                abs(group_offset) + 1.0
            )
            if graph.direction == "BT":
                center_x = float(source_pos[0])
                center_y = float(source_pos[1] - radius * 0.72)
                loop = Arc(
                    (center_x, center_y),
                    width=radius * 1.35,
                    height=radius * 1.2,
                    theta1=210,
                    theta2=510,
                    color=color,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    zorder=1.2,
                )
                tip = (center_x - radius * 0.05, center_y + radius * 0.58)
                direction = np.array([-0.15, 1.0], dtype=np.float64)
                label_x = center_x
                label_y = center_y - radius * 0.92
            elif graph.direction == "LR":
                center_x = float(source_pos[0] - radius * 0.72)
                center_y = float(source_pos[1])
                loop = Arc(
                    (center_x, center_y),
                    width=radius * 1.2,
                    height=radius * 1.35,
                    theta1=120,
                    theta2=420,
                    color=color,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    zorder=1.2,
                )
                tip = (center_x + radius * 0.58, center_y - radius * 0.05)
                direction = np.array([1.0, -0.15], dtype=np.float64)
                label_x = center_x - radius * 0.92
                label_y = center_y
            elif graph.direction == "RL":
                center_x = float(source_pos[0] + radius * 0.72)
                center_y = float(source_pos[1])
                loop = Arc(
                    (center_x, center_y),
                    width=radius * 1.2,
                    height=radius * 1.35,
                    theta1=300,
                    theta2=600,
                    color=color,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    zorder=1.2,
                )
                tip = (center_x - radius * 0.58, center_y - radius * 0.05)
                direction = np.array([-1.0, -0.15], dtype=np.float64)
                label_x = center_x + radius * 0.92
                label_y = center_y
            else:
                center_x = float(source_pos[0])
                center_y = float(source_pos[1] + radius * 0.72)
                loop = Arc(
                    (center_x, center_y),
                    width=radius * 1.35,
                    height=radius * 1.2,
                    theta1=30,
                    theta2=330,
                    color=color,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    zorder=1.2,
                )
                tip = (center_x + radius * 0.05, center_y - radius * 0.58)
                direction = np.array([0.15, -1.0], dtype=np.float64)
                label_x = center_x
                label_y = center_y + radius * 0.92
            ax.add_patch(loop)
            direction /= np.linalg.norm(direction)
            _draw_manual_arrowhead(ax, tip, direction, style)
            if label:
                ax.text(
                    label_x,
                    label_y,
                    label,
                    fontsize=style.label_font_size,
                    color=style.label_font_color,
                    ha="center",
                    va="center",
                    bbox={"facecolor": style.label_background, "edgecolor": "none", "pad": 1.4},
                    zorder=1.3,
                )
            continue

        delta = target_pos - source_pos
        norm = np.linalg.norm(delta)
        if norm <= 1e-9:
            continue
        direction = delta / norm
        start = source_pos + direction * (sizes[source, 1] / 2.4)
        end = target_pos - direction * (sizes[target, 1] / 2.4)

        points: List[Tuple[float, float]]
        if style.routing == "ortho":
            mid_y = (float(start[1]) + float(end[1])) / 2.0
            points = [
                (float(start[0]), float(start[1])),
                (float(start[0]), mid_y),
                (float(end[0]), mid_y),
                (float(end[0]), float(end[1])),
            ]
            ax.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                zorder=1.1,
            )
            arrow_direction = np.array(
                [points[-1][0] - points[-2][0], points[-1][1] - points[-2][1]], dtype=np.float64
            )
            arrow_direction /= np.linalg.norm(arrow_direction)
            _draw_manual_arrowhead(ax, points[-1], arrow_direction, style)
        else:
            rad = 0.0 if style.routing == "straight" else max(float(style.curvature), 0.15)
            if group_offset != 0.0:
                rad = (
                    rad + 0.18 * group_offset
                    if style.routing != "straight"
                    else 0.22 * group_offset
                )
            patch = FancyArrowPatch(
                posA=(float(start[0]), float(start[1])),
                posB=(float(end[0]), float(end[1])),
                arrowstyle="-",
                linewidth=linewidth,
                linestyle=linestyle,
                color=color,
                connectionstyle=f"arc3,rad={rad}",
                zorder=1.1,
            )
            ax.add_patch(patch)
            tangent = np.array(
                [float(end[0] - start[0]), float(end[1] - start[1])], dtype=np.float64
            )
            tangent /= np.linalg.norm(tangent)
            _draw_manual_arrowhead(ax, (float(end[0]), float(end[1])), tangent, style)
            points = [(float(start[0]), float(start[1])), (float(end[0]), float(end[1]))]

        if label:
            label_x, label_y = _edge_midpoint(points)
            ax.text(
                label_x,
                label_y + style.label_offset,
                label,
                fontsize=style.label_font_size,
                color=style.label_font_color,
                ha="center",
                va="center",
                bbox={"facecolor": style.label_background, "edgecolor": "none", "pad": 1.4},
                zorder=1.3,
            )


def _draw_generic_nodes(
    ax: Any,
    scene: CalibrationScene,
    positions: np.ndarray,
    sizes: np.ndarray,
    *,
    draw_node_labels: bool,
) -> None:
    """Draw nodes and plain-text labels for the generic renderer.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    scene : CalibrationScene
        Scene to render.
    positions : numpy.ndarray
        Node positions with shape ``[N, 2]``.
    sizes : numpy.ndarray
        Node sizes with shape ``[N, 2]``.
    draw_node_labels : bool
        Whether labels should be drawn.

    Returns
    -------
    None
        Mutates ``ax`` in place.
    """

    graph = scene.graph
    for index in range(graph.num_nodes):
        style = graph.get_style_for_node(index)
        x = float(positions[index, 0])
        y = float(positions[index, 1])
        width = float(sizes[index, 0])
        height = float(sizes[index, 1])
        patch = _node_patch(style, x, y, width, height)
        patch.set_zorder(2.0)
        ax.add_patch(patch)
        if style.shape == "cylinder":
            cap_height = max(height * 0.16, 1.0)
            rim = Ellipse(
                (x, y + height / 2.0 - cap_height),
                width,
                cap_height * 2.0,
                facecolor="none",
                edgecolor=to_rgba(style.stroke, style.opacity),
                linewidth=style.stroke_width,
                zorder=2.05,
            )
            ax.add_patch(rim)
        if not draw_node_labels:
            continue
        if not graph.node_labels[index]:
            continue
        fontsize = (
            float(graph.node_font_sizes[index].item())
            if graph.node_font_sizes is not None
            else float(style.font_size)
        )
        ax.text(
            x,
            y,
            graph.node_labels[index],
            fontsize=fontsize,
            fontweight=style.font_weight,
            fontstyle=style.font_style,
            color=style.font_color,
            ha=style.text_align,
            va=style.text_valign,
            zorder=2.2,
            clip_on=False,
            multialignment=style.text_align,
        )


def _rich_text_mpl_renderer(ax: Any, scene: CalibrationScene) -> None:
    """Render the rich-text case with plain matplotlib text runs.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    scene : CalibrationScene
        Scene to render.

    Returns
    -------
    None
        Mutates ``ax`` in place.
    """

    _draw_generic_scene(ax, scene, draw_node_labels=False)
    x = float(scene.positions[0, 0].item())
    y = float(scene.positions[0, 1].item())
    style = scene.graph.get_style_for_node(0)
    segments = parse_rich_markup(scene.graph.node_labels[0])
    widths = [max(len(text), 1) * style.font_size * 0.55 for text, _ in segments]
    total_width = sum(widths)
    cursor = x - total_width / 2.0
    for (text, flags), width in zip(segments, widths):
        ax.text(
            cursor,
            y,
            text,
            fontsize=style.font_size,
            fontweight="bold" if flags["bold"] else "regular",
            fontstyle="italic" if flags["italic"] else "normal",
            color=flags["color"] or style.font_color,
            ha="left",
            va="center",
            zorder=2.3,
        )
        cursor += width


def _render_matplotlib_png(scene: CalibrationScene, output_path: Path) -> None:
    """Render the plain-matplotlib reference panel for one scene.

    Parameters
    ----------
    scene : CalibrationScene
        Scene to render.
    output_path : Path
        Destination PNG path.

    Returns
    -------
    None
        Writes the PNG to ``output_path``.
    """

    fig, ax = plt.subplots(figsize=_resolved_scene_figsize(scene), dpi=RAW_RENDER_DPI)
    fig.patch.set_facecolor(WHITE)
    renderer = scene.mpl_renderer or _draw_generic_scene
    renderer(ax, scene)
    fig.savefig(
        output_path, dpi=RAW_RENDER_DPI, facecolor=WHITE, bbox_inches="tight", pad_inches=0.05
    )
    plt.close(fig)


def _content_crop_box(image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """Return a padded crop box around visible content.

    Parameters
    ----------
    image : PIL.Image.Image
        Source image.

    Returns
    -------
    tuple[int, int, int, int] | None
        Crop box in PIL coordinates or ``None`` when no content is found.
    """

    rgba = np.asarray(image.convert("RGBA"))
    mask = (rgba[:, :, 3] > 0) & np.any(rgba[:, :, :3] < 252, axis=2)
    if not bool(mask.any()):
        return None
    ys, xs = np.nonzero(mask)
    left = max(int(xs.min()) - CONTENT_CROP_PADDING, 0)
    top = max(int(ys.min()) - CONTENT_CROP_PADDING, 0)
    right = min(int(xs.max()) + CONTENT_CROP_PADDING + 1, image.width)
    bottom = min(int(ys.max()) + CONTENT_CROP_PADDING + 1, image.height)
    return left, top, right, bottom


def _normalize_panel_image(image_path: Path) -> Image.Image:
    """Resize and center a raw panel image onto a fixed white canvas.

    Parameters
    ----------
    image_path : Path
        Source image path.

    Returns
    -------
    PIL.Image.Image
        Normalized RGB panel image.
    """

    with Image.open(image_path) as image:
        crop_box = _content_crop_box(image)
        cropped = image if crop_box is None else image.crop(crop_box)
        rgba = cropped.convert("RGBA")
        rgba.thumbnail((PANEL_SIZE[0] - PANEL_MARGIN, PANEL_SIZE[1] - PANEL_MARGIN), Image.LANCZOS)
        canvas = Image.new("RGBA", PANEL_SIZE, WHITE)
        offset = ((PANEL_SIZE[0] - rgba.width) // 2, (PANEL_SIZE[1] - rgba.height) // 2)
        canvas.paste(rgba, offset, rgba)
    return canvas.convert("RGB")


def _compose_comparison(
    dagua_image: Path,
    graphviz_image: Path,
    mpl_image: Path,
    output_path: Path,
    description: str,
    category: str,
    case_id: str,
    figsize: Tuple[float, float],
) -> None:
    """Compose the final three-panel comparison figure.

    Parameters
    ----------
    dagua_image : Path
        Raw Dagua panel image.
    graphviz_image : Path
        Raw Graphviz panel image.
    mpl_image : Path
        Raw matplotlib panel image.
    output_path : Path
        Destination composed image.
    description : str
        Human-readable case description.
    category : str
        Case category.
    case_id : str
        Stable case identifier.
    figsize : tuple[float, float]
        Final figure size in inches.

    Returns
    -------
    None
        Writes the comparison PNG to ``output_path``.
    """

    panels = [
        ("Dagua", _normalize_panel_image(dagua_image)),
        ("Graphviz", _normalize_panel_image(graphviz_image)),
        ("matplotlib", _normalize_panel_image(mpl_image)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=COMPARISON_DPI)
    fig.patch.set_facecolor(WHITE)
    for axis, (title, image) in zip(axes, panels):
        axis.imshow(image)
        axis.axis("off")
        axis.set_title(title, fontsize=12, fontweight="bold", pad=14)
    fig.suptitle(description, fontsize=13, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.93,
        f"Category: {category} | Case: {case_id}",
        ha="center",
        va="center",
        fontsize=9,
        color="#6B7280",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    fig.savefig(output_path, dpi=COMPARISON_DPI, facecolor=WHITE, bbox_inches="tight")
    plt.close(fig)


def _reference_cache_path(output_root: Path, case: CalibrationCase, backend: str) -> Path:
    """Return the cached raw-render path for one backend.

    Parameters
    ----------
    output_root : Path
        Suite output root.
    case : CalibrationCase
        Case being rendered.
    backend : str
        Backend name such as ``"graphviz"`` or ``"matplotlib"``.

    Returns
    -------
    Path
        Cache path for the backend PNG.
    """

    return output_root / REF_CACHE_DIRNAME / backend / f"{case.case_id}.png"


def _render_case(
    case: CalibrationCase,
    output_root: Path,
    refresh_refs: bool,
) -> Dict[str, Any]:
    """Render one calibration case and return its manifest row.

    Parameters
    ----------
    case : CalibrationCase
        Case to render.
    output_root : Path
        Suite root directory.
    refresh_refs : bool
        Whether cached Graphviz/matplotlib references should be regenerated.

    Returns
    -------
    dict[str, Any]
        JSON-serializable manifest row.
    """

    scene = case.build_scene()
    output_path = output_root / case.category / f"{case.case_id}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    graphviz_cache = _reference_cache_path(output_root, case, "graphviz")
    mpl_cache = _reference_cache_path(output_root, case, "matplotlib")
    graphviz_cache.parent.mkdir(parents=True, exist_ok=True)
    mpl_cache.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="dagua_calibration_") as temp_dir:
        temp_root = Path(temp_dir)
        dagua_raw = temp_root / "dagua.png"
        _render_dagua_png(scene, dagua_raw)

        graphviz_rendered = False
        if refresh_refs or not graphviz_cache.exists():
            _render_graphviz_png(_build_graphviz_dot(scene), graphviz_cache, scene.graphviz.engine)
            graphviz_rendered = True

        mpl_rendered = False
        if refresh_refs or not mpl_cache.exists():
            _render_matplotlib_png(scene, mpl_cache)
            mpl_rendered = True

        _compose_comparison(
            dagua_image=dagua_raw,
            graphviz_image=graphviz_cache,
            mpl_image=mpl_cache,
            output_path=output_path,
            description=case.description,
            category=case.category,
            case_id=case.case_id,
            figsize=case.comparison_figsize,
        )

    return {
        "case_id": case.case_id,
        "category": case.category,
        "description": case.description,
        "output_path": str(output_path),
        "graphviz_cache": str(graphviz_cache),
        "matplotlib_cache": str(mpl_cache),
        "graphviz_rendered": graphviz_rendered,
        "matplotlib_rendered": mpl_rendered,
    }


def _edge_option_cases() -> List[CalibrationCase]:
    """Build the edge-options category cases.

    Returns
    -------
    list[CalibrationCase]
        Edge calibration cases.
    """

    cases: List[CalibrationCase] = []
    for weight in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]:
        cases.append(
            CalibrationCase(
                case_id=f"lineweight_{weight}",
                category="edge_options",
                description=f"Edge line weight {weight:.2f}pt",
                build_scene=lambda weight=weight: _edge_scene(
                    [(f"{weight:.2f} pt", _base_edge_style(width=weight))],
                    columns=1,
                ),
            )
        )
    for style in ["solid", "dashed", "dotted", "dashdot"]:
        cases.append(
            CalibrationCase(
                case_id=f"linestyle_{style}",
                category="edge_options",
                description=f"Edge line style {style}",
                build_scene=lambda style=style: _edge_scene(
                    [(style, _base_edge_style(style=style))],
                    columns=1,
                ),
            )
        )
    color_specs = [
        ("color_named", "Named color red", "red"),
        ("color_hex", "Hex color #FF0000", "#FF0000"),
        ("color_alpha", "Hex color with alpha #FF000080", "#FF000080"),
    ]
    for case_id, description, color in color_specs:
        cases.append(
            CalibrationCase(
                case_id=case_id,
                category="edge_options",
                description=description,
                build_scene=lambda color=color: _edge_scene(
                    [(color, _base_edge_style(color=color))],
                    columns=1,
                ),
            )
        )
    for opacity in [0.1, 0.3, 0.5, 0.7, 1.0]:
        cases.append(
            CalibrationCase(
                case_id=f"opacity_{opacity:.1f}",
                category="edge_options",
                description=f"Edge opacity {opacity:.1f}",
                build_scene=lambda opacity=opacity: _edge_scene(
                    [(f"alpha={opacity:.1f}", _base_edge_style(opacity=opacity, color="#2563EB"))],
                    columns=1,
                ),
            )
        )
    for arrow in available_arrowheads():
        cases.append(
            CalibrationCase(
                case_id=f"arrowhead_{arrow}",
                category="edge_options",
                description=f"Arrowhead style {arrow}",
                build_scene=lambda arrow=arrow: _edge_scene(
                    [(arrow, _base_edge_style(arrow=arrow, color="#1F2937"))],
                    columns=1,
                ),
            )
        )
    cases.append(
        CalibrationCase(
            case_id="arrowhead_scaling",
            category="edge_options",
            description="Arrowhead size scaling against edge width",
            build_scene=lambda: _edge_scene(
                [
                    (f"w={width:.1f}", _base_edge_style(width=width, arrow="normal"))
                    for width in [0.5, 1.0, 2.0, 4.0]
                ],
                columns=2,
            ),
        )
    )
    cases.append(
        CalibrationCase(
            case_id="no_arrowhead",
            category="edge_options",
            description="No-arrowhead edge mode",
            build_scene=lambda: _edge_scene(
                [("arrow=none", _base_edge_style(arrow="none"))],
                columns=1,
            ),
        )
    )
    return cases


def _node_option_cases() -> List[CalibrationCase]:
    """Build the node-options category cases.

    Returns
    -------
    list[CalibrationCase]
        Node calibration cases.
    """

    cases: List[CalibrationCase] = []
    for shape in NODE_SHAPES:
        cases.append(
            CalibrationCase(
                case_id=f"shape_{shape}",
                category="node_options",
                description=f"Node shape {shape}",
                build_scene=lambda shape=shape: _node_scene(
                    [(shape, _base_node_style(shape=shape))],
                    columns=1,
                ),
            )
        )
    cases.append(
        CalibrationCase(
            case_id="fill_colors",
            category="node_options",
            description="Node fill colors with opaque and alpha variants",
            build_scene=lambda: _node_scene(
                [
                    ("named\nsky", _base_node_style(fill="skyblue", stroke="#1D4ED8")),
                    ("hex\nblue", _base_node_style(fill="#60A5FA", stroke="#1D4ED8")),
                    ("alpha\nblue", _base_node_style(fill="#2563EB88", stroke="#1D4ED8")),
                    ("warm\nred", _base_node_style(fill="#FCA5A5", stroke="#B91C1C")),
                ],
                columns=2,
            ),
        )
    )
    for weight in [0.25, 0.5, 1.0, 2.0, 4.0]:
        cases.append(
            CalibrationCase(
                case_id=f"border_weight_{weight}",
                category="node_options",
                description=f"Node border width {weight:.2f}pt",
                build_scene=lambda weight=weight: _node_scene(
                    [("Border", _base_node_style(stroke_width=weight))],
                    columns=1,
                ),
            )
        )
    for style in ["solid", "dashed", "dotted"]:
        cases.append(
            CalibrationCase(
                case_id=f"border_style_{style}",
                category="node_options",
                description=f"Node border style {style}",
                build_scene=lambda style=style: _node_scene(
                    [(style, _base_node_style(stroke_dash=style))],
                    columns=1,
                ),
            )
        )
    for radius in [0.0, 3.0, 6.0, 12.0, 20.0]:
        cases.append(
            CalibrationCase(
                case_id=f"corner_radius_{int(radius)}",
                category="node_options",
                description=f"Round-rectangle corner radius {int(radius)}pt",
                build_scene=lambda radius=radius: _node_scene(
                    [
                        (
                            f"r={int(radius)}",
                            _base_node_style(shape="roundrect", corner_radius=radius),
                        )
                    ],
                    columns=1,
                ),
            )
        )
    cases.append(
        CalibrationCase(
            case_id="sizing_auto",
            category="node_options",
            description="Automatic node sizing from short and long labels",
            build_scene=lambda: _node_scene(
                [
                    ("Short", _base_node_style()),
                    ("A much longer label", _base_node_style()),
                    ("Multi\nLine", _base_node_style()),
                ],
                columns=3,
            ),
        )
    )
    cases.append(
        CalibrationCase(
            case_id="sizing_explicit",
            category="node_options",
            description="Explicit node size floors and oversized labels",
            build_scene=lambda: _node_scene(
                [
                    ("Min 90x40", _base_node_style(min_width=90.0, min_height=40.0)),
                    ("Min 140x60", _base_node_style(min_width=140.0, min_height=60.0)),
                    ("Wide node", _base_node_style(min_width=220.0, min_height=44.0)),
                ],
                columns=3,
            ),
        )
    )
    return cases


def _text_option_cases() -> List[CalibrationCase]:
    """Build the text-options category cases.

    Returns
    -------
    list[CalibrationCase]
        Text calibration cases.
    """

    cases: List[CalibrationCase] = []
    for font_size in [6.0, 8.0, 10.0, 12.0, 16.0, 24.0]:
        cases.append(
            CalibrationCase(
                case_id=f"fontsize_{int(font_size)}",
                category="text_options",
                description=f"Node font size {int(font_size)}pt",
                build_scene=lambda font_size=font_size: _node_scene(
                    [
                        (
                            f"{int(font_size)} pt",
                            _base_node_style(font_size=font_size, min_width=110.0, min_height=50.0),
                        )
                    ],
                    columns=1,
                ),
            )
        )
    for weight in ["regular", "bold"]:
        cases.append(
            CalibrationCase(
                case_id=f"fontweight_{weight}",
                category="text_options",
                description=f"Node font weight {weight}",
                build_scene=lambda weight=weight: _node_scene(
                    [
                        (
                            weight,
                            _base_node_style(font_weight=weight, min_width=140.0, min_height=56.0),
                        )
                    ],
                    columns=1,
                ),
            )
        )
    align_samples = []
    for horizontal in ["left", "center", "right"]:
        for vertical in ["top", "center", "bottom"]:
            align_samples.append(
                (
                    f"{horizontal[0].upper()}/{vertical[0].upper()}",
                    _base_node_style(
                        text_align=horizontal,
                        text_valign=vertical,
                        min_width=110.0,
                        min_height=64.0,
                    ),
                )
            )
    cases.append(
        CalibrationCase(
            case_id="alignment_grid",
            category="text_options",
            description="3x3 text-alignment grid: horizontal x vertical",
            build_scene=lambda: _node_scene(align_samples, columns=3, figsize=(7.5, 7.2)),
        )
    )
    cases.append(
        CalibrationCase(
            case_id="multiline_2",
            category="text_options",
            description="Two-line node label",
            build_scene=lambda: _node_scene(
                [("Line one\nLine two", _base_node_style(min_width=130.0, min_height=70.0))],
                columns=1,
            ),
        )
    )
    cases.append(
        CalibrationCase(
            case_id="multiline_3",
            category="text_options",
            description="Three-line node label",
            build_scene=lambda: _node_scene(
                [("One\nTwo\nThree", _base_node_style(min_width=140.0, min_height=90.0))],
                columns=1,
            ),
        )
    )
    cases.append(
        CalibrationCase(
            case_id="long_label",
            category="text_options",
            description="Long label handling: shrink, overflow, and expand",
            build_scene=_long_label_scene,
            comparison_figsize=(16.0, 6.0),
        )
    )
    cases.append(
        CalibrationCase(
            case_id="rich_text",
            category="text_options",
            description="Rich text label with bold, italic, and color spans",
            build_scene=_rich_text_scene,
        )
    )
    return cases


def _long_label_scene() -> CalibrationScene:
    """Build the long-label handling scene.

    Returns
    -------
    CalibrationScene
        Scene showing three explicit text-handling strategies.
    """

    styles = [
        _base_node_style(text_align="center", min_width=150.0, min_height=70.0),
        _base_node_style(text_align="center", min_width=120.0, min_height=70.0),
        _base_node_style(text_align="center", min_width=240.0, min_height=70.0),
    ]
    scene = _node_scene(
        [
            ("Shrink\n" + LONG_LABEL, styles[0]),
            ("Overflow\n" + LONG_LABEL, styles[1]),
            ("Expand\n" + LONG_LABEL, styles[2]),
        ],
        columns=3,
        node_sizes=[(150.0, 70.0), (120.0, 70.0), (240.0, 70.0)],
        node_font_sizes=[7.0, 10.0, 10.0],
    )
    scene.graphviz.node_attrs = {
        0: {"label": "Shrink\\n" + LONG_LABEL},
        1: {"label": "Overflow\\n" + LONG_LABEL},
        2: {"label": "Expand\\n" + LONG_LABEL},
    }
    return scene


def _rich_text_scene() -> CalibrationScene:
    """Build the rich-text comparison scene.

    Returns
    -------
    CalibrationScene
        Scene with one rich-text node and custom Graphviz/matplotlib handling.
    """

    rich_label = "**Bold** *Italic* {color:#D55E00}Color{/color}"
    scene = _node_scene(
        [(rich_label, _base_node_style(label_format="rich", min_width=240.0, min_height=64.0))],
        columns=1,
        mpl_renderer=_rich_text_mpl_renderer,
        graphviz=GraphvizSpec(
            node_attrs={
                0: {
                    "label": '<<B>Bold</B> <I>Italic</I> <FONT COLOR="#D55E00">Color</FONT>>',
                    "shape": "box",
                    "style": "rounded,filled",
                }
            }
        ),
    )
    return scene


def _cluster_option_cases() -> List[CalibrationCase]:
    """Build the cluster-options category cases.

    Returns
    -------
    list[CalibrationCase]
        Cluster calibration cases.
    """

    return [
        CalibrationCase(
            case_id="border_solid",
            category="cluster_options",
            description="Cluster border style solid",
            build_scene=lambda: _cluster_chain_scene(
                _base_cluster_style(stroke_dash="solid"), depth=1
            ),
        ),
        CalibrationCase(
            case_id="border_dashed",
            category="cluster_options",
            description="Cluster border style dashed",
            build_scene=lambda: _cluster_chain_scene(
                _base_cluster_style(stroke_dash="dashed"), depth=1
            ),
        ),
        CalibrationCase(
            case_id="fill_alpha",
            category="cluster_options",
            description="Cluster fill color and opacity variants",
            build_scene=_cluster_fill_alpha_scene,
            comparison_figsize=(16.0, 6.2),
        ),
        CalibrationCase(
            case_id="nested_1",
            category="cluster_options",
            description="Single-level cluster",
            build_scene=lambda: _cluster_chain_scene(_base_cluster_style(), depth=1),
        ),
        CalibrationCase(
            case_id="nested_2",
            category="cluster_options",
            description="Two-level nested clusters",
            build_scene=lambda: _cluster_chain_scene(_base_cluster_style(), depth=2),
        ),
        CalibrationCase(
            case_id="nested_3",
            category="cluster_options",
            description="Three-level nested clusters",
            build_scene=lambda: _cluster_chain_scene(_base_cluster_style(), depth=3),
        ),
        CalibrationCase(
            case_id="label_positions",
            category="cluster_options",
            description="Cluster label placement: top-left, top-center, top-right",
            build_scene=_cluster_label_position_scene,
            comparison_figsize=(16.0, 6.2),
        ),
    ]


def _cluster_chain_scene(style: ClusterStyle, depth: int) -> CalibrationScene:
    """Build a simple clustered chain scene.

    Parameters
    ----------
    style : ClusterStyle
        Base cluster style.
    depth : int
        Cluster nesting depth.

    Returns
    -------
    CalibrationScene
        Clustered chain scene.
    """

    scene = _build_graph(
        node_labels=["In", "Core", "Out"],
        positions=[(0.0, 90.0), (0.0, 0.0), (0.0, -90.0)],
        edges=[(0, 1), (1, 2)],
        node_styles=[_base_node_style(), _base_node_style(), _base_node_style()],
        edge_styles=[_base_edge_style(), _base_edge_style()],
        clusters=[
            ("group_1", [0, 1, 2], style, "Group 1", None),
            *(
                (
                    f"group_{level}",
                    [1] if level == depth else [0, 1, 2],
                    _base_cluster_style(
                        fill=f"#DBEAFE{max(80 - level * 12, 32):02X}",
                        stroke=["#60A5FA", "#2563EB", "#1D4ED8"][min(level - 1, 2)],
                        label_position="top-left",
                    ),
                    f"Group {level}",
                    f"group_{level - 1}",
                )
                for level in range(2, depth + 1)
            ),
        ],
        figsize=(4.6, 5.0),
        graphviz=GraphvizSpec(graph_attrs={"splines": "polyline"}),
    )
    return scene


def _cluster_fill_alpha_scene() -> CalibrationScene:
    """Build a scene comparing several cluster fill-alpha combinations.

    Returns
    -------
    CalibrationScene
        Multi-cluster fill-alpha scene.
    """

    graph = DaguaGraph(direction="TB")
    node_styles: List[NodeStyle] = []
    positions: List[Tuple[float, float]] = []
    cluster_specs: List[Tuple[str, Sequence[int], ClusterStyle, str, Optional[str]]] = []
    for index, (title, fill) in enumerate(
        [
            ("Blue 0.2", "#93C5FD33"),
            ("Blue 0.5", "#60A5FA80"),
            ("Red 0.5", "#FCA5A580"),
        ]
    ):
        origin_x, origin_y = _grid_origins(3, 3)[index]
        for local, node_label in enumerate(["A", "B", "C"]):
            graph.add_node(index * 3 + local, label=node_label, style=_base_node_style())
            positions.append((origin_x, origin_y + (1 - local) * 70.0))
            node_styles.append(_base_node_style())
        graph.add_edge(index * 3, index * 3 + 1, style=_base_edge_style())
        graph.add_edge(index * 3 + 1, index * 3 + 2, style=_base_edge_style())
        cluster_specs.append(
            (
                f"cluster_{index}",
                [index * 3, index * 3 + 1, index * 3 + 2],
                _base_cluster_style(fill=fill),
                title,
                None,
            )
        )
    scene = _build_graph(
        node_labels=[graph.node_labels[index] for index in range(graph.num_nodes)],
        positions=positions,
        edges=[(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8)],
        node_styles=node_styles,
        edge_styles=[_base_edge_style() for _ in range(6)],
        clusters=cluster_specs,
        figsize=(5.8, 4.0),
    )
    return scene


def _cluster_label_position_scene() -> CalibrationScene:
    """Build a scene comparing three cluster label positions.

    Returns
    -------
    CalibrationScene
        Multi-cluster label-position scene.
    """

    positions = _grid_origins(3, 3)
    node_labels: List[str] = []
    node_positions: List[Tuple[float, float]] = []
    node_styles: List[NodeStyle] = []
    edges: List[Tuple[int, int]] = []
    edge_styles: List[EdgeStyle] = []
    clusters: List[Tuple[str, Sequence[int], ClusterStyle, str, Optional[str]]] = []
    cluster_positions = ["top-left", "top-center", "top-right"]
    for cluster_index, label_position in enumerate(cluster_positions):
        origin_x, origin_y = positions[cluster_index]
        base_index = cluster_index * 2
        node_labels.extend(["A", "B"])
        node_positions.extend(
            [(origin_x - 30.0, origin_y + 20.0), (origin_x + 30.0, origin_y - 20.0)]
        )
        node_styles.extend([_base_node_style(), _base_node_style()])
        edges.append((base_index, base_index + 1))
        edge_styles.append(_base_edge_style())
        clusters.append(
            (
                f"cluster_{cluster_index}",
                [base_index, base_index + 1],
                _base_cluster_style(label_position=label_position),
                label_position,
                None,
            )
        )
    return _build_graph(
        node_labels=node_labels,
        positions=node_positions,
        edges=edges,
        node_styles=node_styles,
        edge_styles=edge_styles,
        clusters=clusters,
        figsize=(5.8, 3.8),
    )


def _combination_2way_cases() -> List[CalibrationCase]:
    """Build the two-way combination category cases.

    Returns
    -------
    list[CalibrationCase]
        Two-way combination cases.
    """

    weights = [0.5, 1.0, 2.0, 4.0]
    styles = ["solid", "dashed", "dotted"]
    arrows = ["normal", "vee", "dot", "diamond", "tee"]
    shapes = ["rect", "roundrect", "ellipse", "diamond", "circle"]
    border_styles = ["solid", "dashed", "dotted"]
    return [
        CalibrationCase(
            case_id="weight_x_style",
            category="combinations_2way",
            description="Line weight x line style matrix",
            build_scene=lambda: _edge_scene(
                [
                    (f"{weight:g} / {style}", _base_edge_style(width=weight, style=style))
                    for weight in weights
                    for style in styles
                ],
                columns=3,
                figsize=(8.8, 8.4),
            ),
            comparison_figsize=(17.0, 6.4),
        ),
        CalibrationCase(
            case_id="weight_x_arrowhead",
            category="combinations_2way",
            description="Line weight x arrowhead matrix",
            build_scene=lambda: _edge_scene(
                [
                    (f"{weight:g} / {arrow}", _base_edge_style(width=weight, arrow=arrow))
                    for weight in weights
                    for arrow in arrows
                ],
                columns=5,
                figsize=(9.8, 8.2),
            ),
            comparison_figsize=(18.5, 6.6),
        ),
        CalibrationCase(
            case_id="style_x_arrowhead",
            category="combinations_2way",
            description="Line style x arrowhead matrix",
            build_scene=lambda: _edge_scene(
                [
                    (f"{style} / {arrow}", _base_edge_style(style=style, arrow=arrow))
                    for style in styles
                    for arrow in arrows
                ],
                columns=5,
                figsize=(9.6, 6.5),
            ),
            comparison_figsize=(18.5, 6.4),
        ),
        CalibrationCase(
            case_id="shape_x_border",
            category="combinations_2way",
            description="Node shape x border style matrix",
            build_scene=lambda: _node_scene(
                [
                    (
                        f"{shape}\n{border}",
                        _base_node_style(
                            shape=shape, stroke_dash=border, min_width=115.0, min_height=70.0
                        ),
                    )
                    for shape in shapes
                    for border in border_styles
                ],
                columns=3,
                figsize=(8.4, 9.2),
            ),
            comparison_figsize=(16.5, 6.8),
        ),
        CalibrationCase(
            case_id="fontsize_x_nodesize",
            category="combinations_2way",
            description="Font size x node size interactions",
            build_scene=lambda: _node_scene(
                [
                    (
                        "small text\nlarge node",
                        _base_node_style(font_size=8.0, min_width=220.0, min_height=90.0),
                    ),
                    (
                        "large text\nsmall node",
                        _base_node_style(font_size=20.0, min_width=120.0, min_height=52.0),
                    ),
                    (
                        "matching",
                        _base_node_style(font_size=12.0, min_width=150.0, min_height=64.0),
                    ),
                ],
                columns=3,
                figsize=(8.0, 4.8),
            ),
        ),
    ]


def _combination_3way_cases() -> List[CalibrationCase]:
    """Build the three-way combination category cases.

    Returns
    -------
    list[CalibrationCase]
        Three-way combination cases.
    """

    return [
        CalibrationCase(
            case_id="weight_x_style_x_arrowhead",
            category="combinations_3way",
            description="Weight x style x arrowhead combinations",
            build_scene=lambda: _edge_scene(
                [
                    (
                        f"{weight:g} / {style} / {arrow}",
                        _base_edge_style(width=weight, style=style, arrow=arrow),
                    )
                    for weight in [1.0, 3.0]
                    for style in ["solid", "dashed"]
                    for arrow in ["normal", "vee", "dot"]
                ],
                columns=3,
                figsize=(8.4, 8.0),
            ),
            comparison_figsize=(17.0, 6.4),
        ),
        CalibrationCase(
            case_id="weight_x_style_x_curve",
            category="combinations_3way",
            description="Weight x style x curve combinations",
            build_scene=lambda: _edge_scene(
                [
                    (
                        f"{weight:g} / {style} / {routing}",
                        _base_edge_style(
                            width=weight,
                            style=style,
                            routing="straight" if routing == "straight" else "bezier",
                            curvature=0.38,
                        ),
                    )
                    for weight in [1.0, 3.0]
                    for style in ["solid", "dashed"]
                    for routing in ["straight", "curved"]
                ],
                columns=3,
                figsize=(8.4, 6.6),
                graphviz_splines="spline",
            ),
            comparison_figsize=(17.0, 6.4),
        ),
        CalibrationCase(
            case_id="shape_x_fill_x_border",
            category="combinations_3way",
            description="Node shape x fill x border combinations",
            build_scene=lambda: _node_scene(
                [
                    (
                        f"{shape}\n{fill_name}\n{border}",
                        _base_node_style(
                            shape=shape,
                            fill=fill,
                            stroke_dash=border,
                            min_width=128.0,
                            min_height=82.0,
                        ),
                    )
                    for shape in ["rect", "ellipse", "diamond"]
                    for fill_name, fill in [("blue", "#93C5FD"), ("red", "#FCA5A5")]
                    for border in ["solid", "dashed"]
                ],
                columns=3,
                figsize=(8.4, 8.0),
            ),
            comparison_figsize=(16.8, 6.5),
        ),
    ]


def _extreme_value_cases() -> List[CalibrationCase]:
    """Build the extreme-values category cases.

    Returns
    -------
    list[CalibrationCase]
        Extreme-value cases.
    """

    return [
        CalibrationCase(
            case_id="thinnest",
            category="extreme_values",
            description="Thinnest line with the smallest arrowhead",
            build_scene=lambda: _edge_scene(
                [("0.25 pt", _base_edge_style(width=0.25, arrow_length=6.0, arrow_width=4.0))],
                columns=1,
            ),
        ),
        CalibrationCase(
            case_id="thickest",
            category="extreme_values",
            description="Thickest line with the largest arrowhead",
            build_scene=lambda: _edge_scene(
                [("8.0 pt", _base_edge_style(width=8.0, arrow_length=20.0, arrow_width=15.0))],
                columns=1,
            ),
        ),
        CalibrationCase(
            case_id="zero_alpha",
            category="extreme_values",
            description="Zero-alpha nodes and edges",
            build_scene=lambda: _build_graph(
                node_labels=["Hidden", "Visible"],
                positions=[(-80.0, 0.0), (80.0, 0.0)],
                edges=[(0, 1)],
                node_styles=[_base_node_style(opacity=0.0), _base_node_style()],
                edge_styles=[_base_edge_style(opacity=0.0)],
                edge_labels=["alpha=0"],
                figsize=(6.8, 4.6),
            ),
        ),
        CalibrationCase(
            case_id="long_label_small_node",
            category="extreme_values",
            description="Very long label inside a very small node",
            build_scene=lambda: _node_scene(
                [(LONG_LABEL, _base_node_style(min_width=90.0, min_height=48.0))],
                columns=1,
                node_sizes=[(90.0, 48.0)],
                node_font_sizes=[10.0],
            ),
        ),
        CalibrationCase(
            case_id="short_label_large_node",
            category="extreme_values",
            description='Very short label "X" inside a very large node',
            build_scene=lambda: _node_scene(
                [("X", _base_node_style(min_width=260.0, min_height=140.0))],
                columns=1,
                node_sizes=[(260.0, 140.0)],
            ),
        ),
        CalibrationCase(
            case_id="self_loops",
            category="extreme_values",
            description="Self-loops with several styles",
            build_scene=_self_loop_scene,
        ),
        CalibrationCase(
            case_id="parallel_edges",
            category="extreme_values",
            description="Parallel edges with different styles",
            build_scene=_parallel_edge_scene,
        ),
    ]


def _self_loop_scene() -> CalibrationScene:
    """Build the self-loop calibration scene.

    Returns
    -------
    CalibrationScene
        Scene with multiple self-loops.
    """

    return _build_graph(
        node_labels=["Loop A", "Loop B", "Loop C"],
        positions=[(-124.0, 0.0), (0.0, 0.0), (124.0, 0.0)],
        edges=[(0, 0), (1, 1), (2, 2)],
        node_styles=[_base_node_style(), _base_node_style(), _base_node_style()],
        edge_styles=[
            _base_edge_style(style="solid", arrow="normal"),
            _base_edge_style(style="dashed", arrow="vee", color="#2563EB"),
            _base_edge_style(style="dotted", arrow="dot", color="#DC2626"),
        ],
        edge_labels=["solid", "dashed vee", "dotted dot"],
        direction="TB",
        figsize=(6.4, 4.2),
    )


def _parallel_edge_scene() -> CalibrationScene:
    """Build the parallel-edge calibration scene.

    Returns
    -------
    CalibrationScene
        Scene with three parallel edges.
    """

    return _build_graph(
        node_labels=["A", "B"],
        positions=[(0.0, 80.0), (0.0, -80.0)],
        edges=[(0, 1), (0, 1), (0, 1)],
        node_styles=[_base_node_style(), _base_node_style()],
        edge_styles=[
            _base_edge_style(style="solid", arrow="normal"),
            _base_edge_style(style="dashed", arrow="vee", color="#2563EB"),
            _base_edge_style(style="dotted", arrow="diamond", color="#DC2626"),
        ],
        edge_labels=["solid", "dashed", "dotted"],
        figsize=(5.2, 5.6),
        graphviz=GraphvizSpec(engine="dot", graph_attrs={"splines": "polyline"}),
    )


def _scaling_cases() -> List[CalibrationCase]:
    """Build the scaling category cases.

    Returns
    -------
    list[CalibrationCase]
        Scaling cases.
    """

    return [
        CalibrationCase(
            case_id=f"graph_{int(width)}in",
            category="scaling",
            description=f"Scaling check at {int(width)}in figure width",
            build_scene=_scaling_scene,
            comparison_figsize=(width, max(4.2, width / 2.8)),
        )
        for width in [4.0, 8.0, 16.0, 32.0]
    ]


def _scaling_scene() -> CalibrationScene:
    """Build the representative multi-component scaling scene.

    Returns
    -------
    CalibrationScene
        Scene with three representative graph motifs.
    """

    graph = DaguaGraph(direction="TB")
    positions: List[Tuple[float, float]] = []
    node_styles: List[NodeStyle] = []
    edges: List[Tuple[int, int]] = []
    edge_styles: List[EdgeStyle] = []
    edge_labels: List[str] = []
    clusters: List[Tuple[str, Sequence[int], ClusterStyle, str, Optional[str]]] = []

    # Component 1: simple pipeline
    component_positions = [(-120.0, 70.0), (-120.0, 0.0), (-120.0, -70.0)]
    for label, position in zip(["In", "Core", "Out"], component_positions):
        graph.add_node(graph.num_nodes, label=label, style=_base_node_style())
        positions.append(position)
        node_styles.append(_base_node_style())
    edges.extend([(0, 1), (1, 2)])
    edge_styles.extend([_base_edge_style(), _base_edge_style(style="dashed")])
    edge_labels.extend(["solid", "dashed"])
    clusters.append(("pipeline", [0, 1, 2], _base_cluster_style(), "Pipeline", None))

    # Component 2: shape comparison
    offset = graph.num_nodes
    for label, position, shape in zip(
        ["Ellipse", "Diamond", "Hex"],
        [(0.0, 70.0), (0.0, 0.0), (0.0, -70.0)],
        ["ellipse", "diamond", "hexagon"],
    ):
        graph.add_node(graph.num_nodes, label=label, style=_base_node_style(shape=shape))
        positions.append(position)
        node_styles.append(_base_node_style(shape=shape))
    edges.extend([(offset, offset + 1), (offset + 1, offset + 2)])
    edge_styles.extend(
        [_base_edge_style(arrow="vee"), _base_edge_style(style="dotted", arrow="diamond")]
    )
    edge_labels.extend(["vee", "diamond"])

    # Component 3: text and cluster
    offset = graph.num_nodes
    for label, position in zip(
        ["Multi\nLine", "Bold", "Rich"],
        [(120.0, 70.0), (120.0, 0.0), (120.0, -70.0)],
    ):
        style = _base_node_style(font_weight="bold" if label == "Bold" else "regular")
        if label == "Rich":
            style = _base_node_style(label_format="rich", min_width=140.0)
            label = "**Bold** *Color*"
        graph.add_node(graph.num_nodes, label=label, style=style)
        positions.append(position)
        node_styles.append(style)
    edges.extend([(offset, offset + 1), (offset + 1, offset + 2)])
    edge_styles.extend([_base_edge_style(), _base_edge_style(style="dashdot")])
    edge_labels.extend(["text", "dashdot"])
    clusters.append(
        (
            "text_cluster",
            [offset, offset + 1, offset + 2],
            _base_cluster_style(fill="#FDE68A66"),
            "Text",
            None,
        )
    )

    scene = _build_graph(
        node_labels=[graph.node_labels[index] for index in range(graph.num_nodes)],
        positions=positions,
        edges=edges,
        node_styles=node_styles,
        edge_styles=edge_styles,
        edge_labels=edge_labels,
        clusters=clusters,
        graphviz=GraphvizSpec(
            graph_attrs={"splines": "spline"},
            node_attrs={offset + 2: {"label": '<<B>Bold</B> <FONT COLOR="#D55E00">Color</FONT>>'}},
        ),
    )
    scene.mpl_renderer = _scaling_mpl_renderer
    return scene


def _scaling_mpl_renderer(ax: Any, scene: CalibrationScene) -> None:
    """Render the scaling scene with custom rich text on the right component.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    scene : CalibrationScene
        Scene to render.

    Returns
    -------
    None
        Mutates ``ax`` in place.
    """

    _draw_generic_scene(ax, scene, draw_node_labels=False)
    positions = scene.positions.detach().cpu().numpy()
    for index in range(scene.graph.num_nodes):
        style = scene.graph.get_style_for_node(index)
        label = scene.graph.node_labels[index]
        if label == "**Bold** *Color*":
            rich_scene = _rich_text_scene()
            rich_scene.positions[0, 0] = float(positions[index, 0])
            rich_scene.positions[0, 1] = float(positions[index, 1])
            _rich_text_mpl_renderer(ax, rich_scene)
            continue
        ax.text(
            float(positions[index, 0]),
            float(positions[index, 1]),
            label,
            fontsize=style.font_size,
            fontweight=style.font_weight,
            fontstyle=style.font_style,
            color=style.font_color,
            ha=style.text_align,
            va=style.text_valign,
            zorder=2.3,
        )


def build_case_catalog() -> List[CalibrationCase]:
    """Build the full calibration case catalog.

    Returns
    -------
    list[CalibrationCase]
        All calibration cases in output order.
    """

    cases: List[CalibrationCase] = []
    cases.extend(_edge_option_cases())
    cases.extend(_node_option_cases())
    cases.extend(_text_option_cases())
    cases.extend(_cluster_option_cases())
    cases.extend(_combination_2way_cases())
    cases.extend(_combination_3way_cases())
    cases.extend(_extreme_value_cases())
    cases.extend(_scaling_cases())
    return cases


def _select_cases(
    catalog: Sequence[CalibrationCase],
    categories: Optional[Sequence[str]] = None,
    case_ids: Optional[Sequence[str]] = None,
) -> List[CalibrationCase]:
    """Filter the catalog by category or case identifier.

    Parameters
    ----------
    catalog : sequence[CalibrationCase]
        Full case catalog.
    categories : sequence[str], optional
        Category filters.
    case_ids : sequence[str], optional
        Exact case identifiers.

    Returns
    -------
    list[CalibrationCase]
        Selected cases in catalog order.
    """

    selected = list(catalog)
    if categories is not None:
        allowed = set(categories)
        selected = [case for case in selected if case.category in allowed]
    if case_ids is not None:
        allowed_ids = set(case_ids)
        selected = [case for case in selected if case.case_id in allowed_ids]
    return selected


def build_calibration_suite(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    categories: Optional[Sequence[str]] = None,
    case_ids: Optional[Sequence[str]] = None,
    refresh_refs: bool = False,
) -> CalibrationSuiteResult:
    """Render the calibration suite and emit a manifest.

    Parameters
    ----------
    output_dir : str, default=DEFAULT_OUTPUT_DIR
        Output directory root.
    categories : sequence[str], optional
        Optional category filters.
    case_ids : sequence[str], optional
        Optional exact case-id filters.
    refresh_refs : bool, default=False
        Whether cached Graphviz and matplotlib references should be regenerated.

    Returns
    -------
    CalibrationSuiteResult
        Output metadata for the rendered suite.
    """

    if not _graphviz_available():
        raise RuntimeError("Graphviz `dot` is required for the calibration suite.")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    catalog = build_case_catalog()
    selected_cases = _select_cases(catalog, categories=categories, case_ids=case_ids)
    if not selected_cases:
        raise ValueError("No calibration cases matched the requested filters.")

    image_paths: List[str] = []
    manifest_rows: List[Dict[str, Any]] = []
    category_counts: Dict[str, int] = {}
    for category in {case.category for case in selected_cases}:
        (root / category).mkdir(parents=True, exist_ok=True)

    for case in selected_cases:
        category_counts[case.category] = category_counts.get(case.category, 0) + 1
        row = _render_case(case, root, refresh_refs=refresh_refs)
        manifest_rows.append(row)
        image_paths.append(row["output_path"])

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(root),
        "total_images": len(image_paths),
        "category_counts": category_counts,
        "cases": manifest_rows,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(f"{json.dumps(manifest, indent=2)}\n", encoding="utf-8")
    return CalibrationSuiteResult(
        output_dir=str(root),
        manifest_path=str(manifest_path),
        image_paths=image_paths,
    )


def main() -> int:
    """Parse CLI arguments and generate the requested calibration cases.

    Returns
    -------
    int
        Process exit code.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        default=None,
        help="Category directory to render. Repeat for multiple categories.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        default=None,
        help="Exact case identifier to render. Repeat for multiple case IDs.",
    )
    parser.add_argument(
        "--refresh-refs",
        action="store_true",
        help="Regenerate cached Graphviz and matplotlib reference panels.",
    )
    args = parser.parse_args()

    result = build_calibration_suite(
        output_dir=args.output_dir,
        categories=args.categories,
        case_ids=args.case_ids,
        refresh_refs=bool(args.refresh_refs),
    )
    print(f"Generated {len(result.image_paths)} calibration images in {result.output_dir}")
    print(f"Manifest: {result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
