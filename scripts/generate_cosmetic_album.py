#!/usr/bin/env python
# ruff: noqa: E402
"""Generate an exhaustive cosmetic comparison album for Dagua renders.

The album compares Dagua's renderer against Graphviz's native renderer for
overlapping cosmetic features, and emits Dagua-only sheets for features that do
not have a close Graphviz analogue in this codebase.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dagua import DaguaGraph, render
from dagua.styles import (
    GRAPHVIZ_MATCH_DEFAULTS,
    ClusterStyle,
    EdgeStyle,
    GraphStyle,
    NodeStyle,
)

PANEL_SIZE: Tuple[int, int] = (920, 680)
COMPARISON_FIGSIZE: Tuple[float, float] = (12.0, 6.0)
SOLO_FIGSIZE: Tuple[float, float] = (6.2, 6.0)
ALBUM_DPI = 180
RAW_RENDER_DPI = 200
GRAPHVIZ_COMPETITOR = "Graphviz dot"
WHITE = "#FFFFFF"
NODE_FILL = "#DCEBFA"
NODE_STROKE = "#4C77A3"
EDGE_COLOR = "#5F6C7B"
CLUSTER_FILL = "#EAF1F8"
CLUSTER_STROKE = "#A9B8C7"
GRAPHVIZ_MIN_NODE_WIDTH = 40.0
GRAPHVIZ_PAIR_VERTICAL_GAP = 80.0
PANEL_CONTENT_MARGIN = 36
CONTENT_CROP_PADDING = 12
ERROR_PLACEHOLDER_FIGSIZE: Tuple[float, float] = (6.2, 6.0)
ERROR_TEXT_WRAP = 44
COMBO_GRADIENT_COLOR = "#9FC7EE"
COMBO_EDGE_GRADIENT_END = "#D55E00"
COMBO_STRIPED_FILL_COLORS: Tuple[str, str] = (NODE_FILL, "#9FC7EE")
COMBO_SHADOW_OFFSET: Tuple[float, float] = (4.0, -4.0)
COMBO_SHADOW_COLOR = "#00000040"
COMBO_SHADOW_BLUR = 3.0
COMBO_TEXT_MAX_WIDTH = 92.0
COMBO_MIN_NODE_WIDTH = 96.0
COMBO_PAIR_VERTICAL_GAP = 120.0
COMBO_DIAMOND_PAIR_VERTICAL_GAP = 140.0
VARIED_EXTERNAL_LABELS: Tuple[str, ...] = ("v1.2", "stable", "beta", "new", "legacy")


@dataclass
class GraphvizRenderSpec:
    """Configuration for a Graphviz-native render.

    Parameters
    ----------
    graph_attrs : dict[str, str]
        Graph-level DOT attributes.
    default_node_attrs : dict[str, str]
        Default DOT node attributes.
    default_edge_attrs : dict[str, str]
        Default DOT edge attributes.
    node_attrs : dict[int, dict[str, str]]
        Per-node attribute overrides keyed by node index.
    edge_attrs : dict[int, dict[str, str]]
        Per-edge attribute overrides keyed by edge index.
    cluster_attrs : dict[str, dict[str, str]]
        Per-cluster attribute overrides keyed by cluster name.
    engine : str, default="dot"
        Graphviz engine executable.
    competitor_label : str, default="Graphviz dot"
        Label shown in composed comparison images.
    """

    graph_attrs: Dict[str, str] = field(default_factory=dict)
    default_node_attrs: Dict[str, str] = field(default_factory=dict)
    default_edge_attrs: Dict[str, str] = field(default_factory=dict)
    node_attrs: Dict[int, Dict[str, str]] = field(default_factory=dict)
    edge_attrs: Dict[int, Dict[str, str]] = field(default_factory=dict)
    cluster_attrs: Dict[str, Dict[str, str]] = field(default_factory=dict)
    engine: str = "dot"
    competitor_label: str = GRAPHVIZ_COMPETITOR


@dataclass
class AlbumCase:
    """Single cosmetic album artifact specification.

    Parameters
    ----------
    case_id : str
        Stable identifier used in the manifest and tests.
    category : str
        Output subdirectory name.
    filename : str
        Final image filename.
    title : str
        Human-readable image title.
    graph : DaguaGraph
        Preconfigured graph used for the Dagua render.
    positions : torch.Tensor
        Fixed node positions with shape ``[N, 2]``.
    settings : dict[str, object]
        JSON-serializable metadata describing the cosmetic option.
    graphviz : GraphvizRenderSpec | None, default=None
        Graphviz-native render configuration for shared comparisons.
    """

    case_id: str
    category: str
    filename: str
    title: str
    graph: DaguaGraph
    positions: torch.Tensor
    settings: Dict[str, object]
    graphviz: Optional[GraphvizRenderSpec] = None


@dataclass
class CosmeticAlbumResult:
    """Paths emitted by the album generator.

    Parameters
    ----------
    output_dir : str
        Root album directory.
    manifest_path : str
        Manifest JSON path.
    image_paths : list[str]
        Final album image paths.
    """

    output_dir: str
    manifest_path: str
    image_paths: List[str]


def _graphviz_available() -> bool:
    """Return whether Graphviz's ``dot`` executable is available.

    Returns
    -------
    bool
        ``True`` when ``dot`` is present on ``PATH``.
    """

    return shutil.which("dot") is not None


def _graphviz_match_node_style_defaults() -> Dict[str, Any]:
    """Return album-only node style overrides that mimic Graphviz's scale.

    Returns
    -------
    dict[str, Any]
        Keyword arguments suitable for ``NodeStyle`` construction.
    """

    return {
        "stroke_width": GRAPHVIZ_MATCH_DEFAULTS["stroke_width"],
        "padding": GRAPHVIZ_MATCH_DEFAULTS["padding"],
        "font_size": GRAPHVIZ_MATCH_DEFAULTS["font_size"],
        "min_width": GRAPHVIZ_MIN_NODE_WIDTH,
        "min_height": GRAPHVIZ_MATCH_DEFAULTS["min_height"],
    }


def _graphviz_match_edge_style_defaults() -> Dict[str, Any]:
    """Return album-only edge style overrides that mimic Graphviz's weight.

    Returns
    -------
    dict[str, Any]
        Keyword arguments suitable for ``EdgeStyle`` construction.
    """

    return {
        "width": GRAPHVIZ_MATCH_DEFAULTS["edge_width"],
        "opacity": GRAPHVIZ_MATCH_DEFAULTS["edge_opacity"],
        "arrow_length": GRAPHVIZ_MATCH_DEFAULTS["arrow_length"],
        "arrow_width": GRAPHVIZ_MATCH_DEFAULTS["arrow_width"],
        "arrow_scale": GRAPHVIZ_MATCH_DEFAULTS["arrow_scale"],
        "arrow_node_fraction": 0.45,
        "arrow_width_ratio": 0.75,
    }


def _top_bottom_pair_positions(
    vertical_gap: float = GRAPHVIZ_PAIR_VERTICAL_GAP,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Return symmetric top-to-bottom positions for a two-node comparison.

    Parameters
    ----------
    vertical_gap : float, default=GRAPHVIZ_PAIR_VERTICAL_GAP
        Distance between the source and target nodes.

    Returns
    -------
    tuple[tuple[float, float], tuple[float, float]]
        Source position first, target position second.
    """

    half_gap = vertical_gap / 2.0
    return (0.0, half_gap), (0.0, -half_gap)


def _base_graph_style() -> GraphStyle:
    """Create the graph-wide render defaults used by album cases.

    Returns
    -------
    GraphStyle
        White-background graph style with modest margins.
    """

    return GraphStyle(
        background_color=WHITE,
        margin=8.0,
        min_figsize=(2.0, 1.5),
        max_figsize=(8.0, 6.0),
    )


def _base_node_style(**overrides: Any) -> NodeStyle:
    """Create a readable base node style for album cases.

    Parameters
    ----------
    **overrides : Any
        NodeStyle field overrides.

    Returns
    -------
    NodeStyle
        Node style with the requested overrides applied.
    """

    style = NodeStyle(
        shape="roundrect",
        fill=NODE_FILL,
        stroke=NODE_STROKE,
        font_color="#1F2937",
        corner_radius=6.0,
        opacity=1.0,
        gradient="none",
        font_weight="regular",
        font_style="normal",
        shadow=False,
        **_graphviz_match_node_style_defaults(),
    )
    for field_name, value in overrides.items():
        setattr(style, field_name, value)
    return style


def _base_edge_style(**overrides: Any) -> EdgeStyle:
    """Create a readable base edge style for album cases.

    Parameters
    ----------
    **overrides : Any
        EdgeStyle field overrides.

    Returns
    -------
    EdgeStyle
        Edge style with the requested overrides applied.
    """

    style = EdgeStyle(
        color=EDGE_COLOR,
        arrow="normal",
        tail_arrow="none",
        arrow_fill="filled",
        style="solid",
        routing="bezier",
        **_graphviz_match_edge_style_defaults(),
    )
    for field_name, value in overrides.items():
        setattr(style, field_name, value)
    return style


def _base_cluster_style(**overrides: Any) -> ClusterStyle:
    """Create a readable base cluster style for album cases.

    Parameters
    ----------
    **overrides : Any
        ClusterStyle field overrides.

    Returns
    -------
    ClusterStyle
        Cluster style with the requested overrides applied.
    """

    style = ClusterStyle(
        fill=CLUSTER_FILL,
        stroke=CLUSTER_STROKE,
        stroke_width=2.0,
        stroke_dash="solid",
        corner_radius=10.0,
        padding=50.0,
        label_position="top-left",
        font_size=13.0,
        font_weight="bold",
        font_color="#374151",
        opacity=0.9,
        label_offset=(12.0, 10.0),
    )
    for field_name, value in overrides.items():
        setattr(style, field_name, value)
    return style


def _combo_node_style(**overrides: Any) -> NodeStyle:
    """Create a node style with visibility defaults for combo cases.

    Parameters
    ----------
    **overrides : Any
        NodeStyle field overrides for the combo case.

    Returns
    -------
    NodeStyle
        Node style with combo-focused defaults applied.
    """

    resolved_overrides = dict(overrides)
    resolved_overrides.setdefault("min_width", COMBO_MIN_NODE_WIDTH)
    if bool(resolved_overrides.get("shadow")):
        resolved_overrides.setdefault("shadow_offset", COMBO_SHADOW_OFFSET)
        resolved_overrides.setdefault("shadow_color", COMBO_SHADOW_COLOR)
        resolved_overrides.setdefault("shadow_blur", COMBO_SHADOW_BLUR)
    if str(resolved_overrides.get("gradient", "none")) != "none":
        resolved_overrides.setdefault("gradient_color", COMBO_GRADIENT_COLOR)
    if str(resolved_overrides.get("fill_pattern", "solid")) == "striped":
        resolved_overrides.setdefault("fill_pattern_colors", list(COMBO_STRIPED_FILL_COLORS))
        resolved_overrides.setdefault("fill_pattern_angle", 30.0)

    opacity = resolved_overrides.get("opacity")
    if opacity is not None:
        resolved_overrides.setdefault("border_opacity", float(opacity))

    return _base_node_style(**resolved_overrides)


def _combo_edge_style(**overrides: Any) -> EdgeStyle:
    """Create an edge style with visibility defaults for combo cases.

    Parameters
    ----------
    **overrides : Any
        EdgeStyle field overrides for the combo case.

    Returns
    -------
    EdgeStyle
        Edge style with combo-focused defaults applied.
    """

    resolved_overrides = dict(overrides)
    if bool(resolved_overrides.get("taper")):
        resolved_overrides.setdefault("taper_width_start", 3.0)
        resolved_overrides.setdefault("taper_width_end", 0.5)
    if str(resolved_overrides.get("color_gradient", "none")) == "source_to_target":
        resolved_overrides.setdefault("color_gradient_end", COMBO_EDGE_GRADIENT_END)
    return _base_edge_style(**resolved_overrides)


def _combo_case(
    slug: str,
    category: str,
    title: str,
    graph: DaguaGraph,
    positions: torch.Tensor,
    settings: Dict[str, object],
) -> AlbumCase:
    """Build a Dagua-only cosmetic combo case.

    Parameters
    ----------
    slug : str
        Stable case slug without the ``combo_`` prefix.
    category : str
        Combo category name such as ``combo_2way``.
    title : str
        Human-readable combo title suffix.
    graph : DaguaGraph
        Configured graph for the case.
    positions : torch.Tensor
        Fixed node positions with shape ``[N, 2]``.
    settings : dict[str, object]
        JSON-serializable combo metadata.

    Returns
    -------
    AlbumCase
        Dagua-only combo case definition.
    """

    return AlbumCase(
        case_id=f"combo_{slug}",
        category=category,
        filename=f"{slug}_dagua.png",
        title=f"Combo: {title}",
        graph=graph,
        positions=positions,
        settings=settings,
        graphviz=None,
    )


def _apply_graph_style(graph: DaguaGraph) -> None:
    """Apply shared graph-level render settings in place.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to update.

    Returns
    -------
    None
        The graph is mutated in place.
    """

    graph._theme.graph_style = _base_graph_style()


def _set_all_node_styles(graph: DaguaGraph, style: NodeStyle) -> None:
    """Assign the same node style to every node in the graph.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to mutate.
    style : NodeStyle
        Style copied onto every node.

    Returns
    -------
    None
        The graph is mutated in place.
    """

    graph.node_styles = [style for _ in range(graph.num_nodes)]


def _set_varied_external_label_styles(graph: DaguaGraph, **overrides: Any) -> None:
    """Assign per-node styles with distinct external labels.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to mutate.
    **overrides : Any
        ``NodeStyle`` overrides passed through ``_combo_node_style``.

    Returns
    -------
    None
        The graph is mutated in place.
    """

    graph.node_styles = [
        _combo_node_style(
            external_label=VARIED_EXTERNAL_LABELS[index % len(VARIED_EXTERNAL_LABELS)],
            **overrides,
        )
        for index in range(graph.num_nodes)
    ]


def _set_all_edge_styles(graph: DaguaGraph, style: EdgeStyle) -> None:
    """Assign the same edge style to every edge in the graph.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to mutate.
    style : EdgeStyle
        Style copied onto every edge.

    Returns
    -------
    None
        The graph is mutated in place.
    """

    edge_count = int(graph.edge_index.shape[1])
    graph.edge_styles = [style for _ in range(edge_count)]


def _pair_graph(
    positions: Sequence[Tuple[float, float]],
    labels: Sequence[str],
    direction: str = "TB",
) -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a two-node one-edge graph with fixed positions.

    Parameters
    ----------
    positions : sequence[tuple[float, float]]
        Positions for the two nodes.
    labels : sequence[str]
        Node labels in node order.
    direction : str, default="TB"
        Graph direction field used by routing.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        The configured graph and a ``[2, 2]`` position tensor.
    """

    graph = DaguaGraph(direction=direction)
    _apply_graph_style(graph)
    graph.add_node("A", label=labels[0])
    graph.add_node("B", label=labels[1])
    graph.add_edge("A", "B")
    return graph, torch.tensor(positions, dtype=torch.float32)


def _combo_pair_graph(
    labels: Sequence[str],
    direction: str = "TB",
    vertical_gap: float = COMBO_PAIR_VERTICAL_GAP,
) -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a two-node combo graph with room for larger cosmetic treatments.

    Parameters
    ----------
    labels : sequence[str]
        Node labels in node order.
    direction : str, default="TB"
        Graph direction field used by routing.
    vertical_gap : float, default=COMBO_PAIR_VERTICAL_GAP
        Distance between the source and target nodes for top-bottom layouts.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        The configured graph and a ``[2, 2]`` position tensor.
    """

    return _pair_graph(
        positions=_top_bottom_pair_positions(vertical_gap=vertical_gap),
        labels=labels,
        direction=direction,
    )


def _single_node_graph(
    label: str,
    position: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a one-node graph with a fixed position.

    Parameters
    ----------
    label : str
        Node label.
    position : tuple[float, float], default=(0.0, 0.0)
        Node position.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        The configured graph and a ``[1, 2]`` position tensor.
    """

    graph = DaguaGraph()
    _apply_graph_style(graph)
    graph.add_node("A", label=label)
    return graph, torch.tensor([position], dtype=torch.float32)


def _direction_graph(direction: str) -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a minimal three-node chain for direction comparisons.

    Parameters
    ----------
    direction : str
        One of ``TB``, ``BT``, ``LR``, or ``RL``.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        The configured graph and a ``[3, 2]`` position tensor.
    """

    if direction == "TB":
        positions = [(0.0, 140.0), (0.0, 70.0), (0.0, 0.0)]
    elif direction == "BT":
        positions = [(0.0, 0.0), (0.0, 70.0), (0.0, 140.0)]
    elif direction == "LR":
        positions = [(0.0, 0.0), (160.0, 0.0), (320.0, 0.0)]
    elif direction == "RL":
        positions = [(320.0, 0.0), (160.0, 0.0), (0.0, 0.0)]
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    graph = DaguaGraph(direction=direction)
    _apply_graph_style(graph)
    graph.add_node("A", label="Stage A")
    graph.add_node("B", label="Stage B")
    graph.add_node("C", label="Stage C")
    graph.add_edge("A", "B")
    graph.add_edge("B", "C")
    return graph, torch.tensor(positions, dtype=torch.float32)


def _basic_cluster_graph() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a small cluster-focused graph.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Cluster demo graph and fixed positions.
    """

    graph = DaguaGraph()
    _apply_graph_style(graph)
    graph.add_node("A", label="One")
    graph.add_node("B", label="Two")
    graph.add_node("C", label="Three")
    graph.add_edge("A", "B")
    graph.add_edge("B", "C")
    graph.add_cluster("group", ["A", "B"], label="Cluster")
    positions = torch.tensor(
        [[0.0, 80.0], [0.0, 0.0], [0.0, -80.0]],
        dtype=torch.float32,
    )
    return graph, positions


def _nested_cluster_graph() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a nested cluster graph for hierarchy comparisons.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Nested-cluster graph and fixed positions.
    """

    graph = DaguaGraph()
    _apply_graph_style(graph)
    graph.add_node("A", label="Outer A")
    graph.add_node("B", label="Inner B")
    graph.add_node("C", label="Inner C")
    graph.add_node("D", label="Outer D")
    graph.add_edge("A", "B")
    graph.add_edge("B", "C")
    graph.add_edge("C", "D")
    graph.add_cluster("outer", ["A", "B", "C", "D"], label="Outer")
    graph.add_cluster("inner", ["B", "C"], label="Inner", parent="outer")
    positions = torch.tensor(
        [[0.0, 120.0], [0.0, 40.0], [0.0, -40.0], [0.0, -120.0]],
        dtype=torch.float32,
    )
    return graph, positions


def _overlapping_nodes_graph() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build an overlapping-node graph to make opacity obvious.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Two-node graph and fixed positions.
    """

    graph = DaguaGraph()
    _apply_graph_style(graph)
    graph.add_node("A", label="Alpha")
    graph.add_node("B", label="Beta")
    positions = torch.tensor([[-40.0, 0.0], [40.0, 0.0]], dtype=torch.float32)
    return graph, positions


def _crossing_edges_graph() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a small crossing-edge graph for edge-opacity comparisons.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Four-node graph and fixed positions.
    """

    graph = DaguaGraph()
    _apply_graph_style(graph)
    graph.add_node("A", label="A")
    graph.add_node("B", label="B")
    graph.add_node("C", label="C")
    graph.add_node("D", label="D")
    graph.add_edge("A", "D")
    graph.add_edge("B", "C")
    positions = torch.tensor(
        [[-130.0, -50.0], [130.0, -50.0], [-130.0, 110.0], [130.0, 110.0]],
        dtype=torch.float32,
    )
    return graph, positions


def _stacked_nodes_graph() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a five-node graph with intentionally overlapping positions.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Five-node graph and a ``[5, 2]`` position tensor with slight offsets.
    """

    graph = DaguaGraph()
    _apply_graph_style(graph)
    for index, label in enumerate(("A", "B", "C", "D", "E")):
        graph.add_node(f"n{index}", label=label)
    positions = torch.tensor(
        [
            [0.0, 0.0],
            [4.0, 4.0],
            [-4.0, 4.0],
            [4.0, -4.0],
            [-4.0, -4.0],
        ],
        dtype=torch.float32,
    )
    return graph, positions


def _deep_cluster_graph() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a four-level nested cluster graph for pathological coverage.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Deep-cluster graph and a ``[8, 2]`` position tensor.
    """

    graph = DaguaGraph()
    _apply_graph_style(graph)
    for index in range(8):
        graph.add_node(f"n{index}", label=f"Node {index}")
    for source, target in (
        ("n0", "n1"),
        ("n1", "n2"),
        ("n2", "n3"),
        ("n3", "n4"),
        ("n4", "n5"),
        ("n5", "n6"),
        ("n6", "n7"),
        ("n0", "n4"),
        ("n2", "n6"),
    ):
        graph.add_edge(source, target)

    graph.add_cluster("level_1", [f"n{index}" for index in range(8)], label="Level 1")
    graph.add_cluster(
        "level_2", [f"n{index}" for index in range(1, 7)], label="Level 2", parent="level_1"
    )
    graph.add_cluster(
        "level_3", [f"n{index}" for index in range(2, 6)], label="Level 3", parent="level_2"
    )
    graph.add_cluster(
        "level_4", [f"n{index}" for index in range(3, 5)], label="Level 4", parent="level_3"
    )

    positions = torch.tensor(
        [
            [-180.0, 130.0],
            [-60.0, 130.0],
            [60.0, 130.0],
            [180.0, 130.0],
            [-180.0, -10.0],
            [-60.0, -10.0],
            [60.0, -10.0],
            [180.0, -10.0],
        ],
        dtype=torch.float32,
    )
    return graph, positions


def _many_arrow_graph() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a hub-and-spoke graph with eight incoming edges.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Hub graph and a ``[9, 2]`` position tensor.
    """

    graph = DaguaGraph()
    _apply_graph_style(graph)
    graph.add_node("hub", label="Hub")
    for index in range(8):
        graph.add_node(f"src_{index}", label=f"S{index}")
        graph.add_edge(f"src_{index}", "hub")

    positions = torch.tensor(
        [
            [0.0, 0.0],
            [-180.0, 140.0],
            [-70.0, 170.0],
            [70.0, 170.0],
            [180.0, 140.0],
            [-180.0, -140.0],
            [-70.0, -170.0],
            [70.0, -170.0],
            [180.0, -140.0],
        ],
        dtype=torch.float32,
    )
    return graph, positions


def _self_loop_graph() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a single-node graph with a self-loop edge.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Self-loop graph and a ``[1, 2]`` position tensor.
    """

    graph = DaguaGraph()
    _apply_graph_style(graph)
    graph.add_node("loop", label="Loop")
    graph.add_edge("loop", "loop")
    return graph, torch.tensor([[0.0, 0.0]], dtype=torch.float32)


def _graphviz_base_node_attrs() -> Dict[str, str]:
    """Return common Graphviz node attributes for album comparisons.

    Returns
    -------
    dict[str, str]
        Default Graphviz node attributes.
    """

    return {
        "shape": "box",
        "style": "filled",
        "fillcolor": NODE_FILL,
        "color": NODE_STROKE,
        "penwidth": "2.0",
        "fontname": "Helvetica",
        "fontsize": "14",
        "fontcolor": "#1F2937",
    }


def _graphviz_base_edge_attrs() -> Dict[str, str]:
    """Return common Graphviz edge attributes for album comparisons.

    Returns
    -------
    dict[str, str]
        Default Graphviz edge attributes.
    """

    return {
        "color": EDGE_COLOR,
        "penwidth": "2.0",
        "arrowsize": "1.1",
    }


def _escape_dot_string(value: str) -> str:
    """Escape a string for quoted DOT output.

    Parameters
    ----------
    value : str
        Raw text.

    Returns
    -------
    str
        Escaped DOT string content.
    """

    return value.replace("\\", "\\\\").replace('"', '\\"')


def _format_dot_value(value: str) -> str:
    """Format a DOT attribute value with conservative quoting.

    Parameters
    ----------
    value : str
        Raw attribute value.

    Returns
    -------
    str
        DOT-ready attribute value.
    """

    stripped = value.strip()
    if stripped == "":
        return '""'
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.:+-"
    if all(char in allowed for char in stripped):
        return stripped
    return f'"{_escape_dot_string(stripped)}"'


def _format_dot_attrs(attrs: Mapping[str, str]) -> str:
    """Format DOT attributes for a node, edge, or graph statement.

    Parameters
    ----------
    attrs : Mapping[str, str]
        Attribute mapping.

    Returns
    -------
    str
        Formatted ``[key=value, ...]`` string, or an empty string.
    """

    if not attrs:
        return ""
    parts = [f"{key}={_format_dot_value(value)}" for key, value in attrs.items()]
    return f" [{', '.join(parts)}]"


def _cluster_children(graph: DaguaGraph) -> Dict[Optional[str], List[str]]:
    """Return the cluster hierarchy keyed by parent cluster name.

    Parameters
    ----------
    graph : DaguaGraph
        Graph containing clusters.

    Returns
    -------
    dict[Optional[str], list[str]]
        Parent-to-children cluster mapping.
    """

    children: Dict[Optional[str], List[str]] = {}
    for name in sorted(graph.clusters):
        parent = graph.cluster_parents.get(name)
        if parent not in graph.clusters:
            parent = None
        children.setdefault(parent, []).append(name)
    return children


def _cluster_members(graph: DaguaGraph, name: str) -> List[int]:
    """Return flattened leaf members for a cluster.

    Parameters
    ----------
    graph : DaguaGraph
        Graph containing the cluster.
    name : str
        Cluster name.

    Returns
    -------
    list[int]
        Leaf node indices.
    """

    members = graph.clusters.get(name, [])
    if isinstance(members, dict):
        from dagua.utils import collect_cluster_leaves

        return [int(index) for index in collect_cluster_leaves(members)]
    return [int(index) for index in members]


def _emit_cluster_block(
    lines: List[str],
    graph: DaguaGraph,
    name: str,
    spec: GraphvizRenderSpec,
    children: Mapping[Optional[str], List[str]],
    emitted: set[int],
    depth: int,
) -> None:
    """Append DOT source for a cluster and its nested children.

    Parameters
    ----------
    lines : list[str]
        Mutable line buffer.
    graph : DaguaGraph
        Source graph.
    name : str
        Cluster name.
    spec : GraphvizRenderSpec
        Graphviz render configuration.
    children : Mapping[Optional[str], list[str]]
        Cluster hierarchy.
    emitted : set[int]
        Nodes already emitted into cluster scopes.
    depth : int
        Indentation depth.

    Returns
    -------
    None
        The DOT line buffer is mutated in place.
    """

    indent = "  " * (depth + 1)
    cluster_id = name.replace(".", "_")
    attrs = {"label": graph.cluster_labels.get(name, name)}
    attrs.update(spec.cluster_attrs.get(name, {}))
    lines.append(f"{indent}subgraph cluster_{cluster_id} {{")
    for key, value in attrs.items():
        lines.append(f"{indent}  {key}={_format_dot_value(value)};")

    child_names = children.get(name, [])
    for child in child_names:
        _emit_cluster_block(lines, graph, child, spec, children, emitted, depth + 1)

    nested_members: set[int] = set()
    for child in child_names:
        nested_members.update(_cluster_members(graph, child))

    for node_index in _cluster_members(graph, name):
        if node_index in nested_members or node_index in emitted:
            continue
        node_attrs = {"label": graph.node_labels[node_index]}
        node_attrs.update(spec.node_attrs.get(node_index, {}))
        lines.append(f"{indent}  n{node_index}{_format_dot_attrs(node_attrs)};")
        emitted.add(node_index)

    lines.append(f"{indent}}}")


def _build_graphviz_dot(graph: DaguaGraph, spec: GraphvizRenderSpec) -> str:
    """Serialize a DaguaGraph into DOT using Graphviz render overrides.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to serialize.
    spec : GraphvizRenderSpec
        Graphviz-native render configuration.

    Returns
    -------
    str
        DOT source code.
    """

    graph_attrs = {"bgcolor": WHITE}
    graph_attrs.update(spec.graph_attrs)
    node_defaults = _graphviz_base_node_attrs()
    node_defaults.update(spec.default_node_attrs)
    edge_defaults = _graphviz_base_edge_attrs()
    edge_defaults.update(spec.default_edge_attrs)

    lines = ["digraph G {"]
    for key, value in graph_attrs.items():
        lines.append(f"  {key}={_format_dot_value(value)};")
    lines.append(f"  node{_format_dot_attrs(node_defaults)};")
    lines.append(f"  edge{_format_dot_attrs(edge_defaults)};")

    emitted: set[int] = set()
    if graph.clusters:
        children = _cluster_children(graph)
        for root_cluster in children.get(None, []):
            _emit_cluster_block(lines, graph, root_cluster, spec, children, emitted, 0)

    for node_index in range(graph.num_nodes):
        if node_index in emitted:
            continue
        node_attrs = {"label": graph.node_labels[node_index]}
        node_attrs.update(spec.node_attrs.get(node_index, {}))
        lines.append(f"  n{node_index}{_format_dot_attrs(node_attrs)};")

    edge_count = int(graph.edge_index.shape[1])
    for edge_index in range(edge_count):
        source = int(graph.edge_index[0, edge_index].item())
        target = int(graph.edge_index[1, edge_index].item())
        edge_attrs = dict(spec.edge_attrs.get(edge_index, {}))
        if edge_index < len(graph.edge_labels) and graph.edge_labels[edge_index]:
            edge_attrs.setdefault("label", str(graph.edge_labels[edge_index]))
        lines.append(f"  n{source} -> n{target}{_format_dot_attrs(edge_attrs)};")

    lines.append("}")
    return "\n".join(lines)


def _render_graphviz_png(
    dot_source: str,
    output_path: Path,
    engine: str,
    dpi: int = RAW_RENDER_DPI,
) -> None:
    """Render DOT source to a PNG with Graphviz.

    Parameters
    ----------
    dot_source : str
        DOT source code.
    output_path : Path
        Destination PNG path.
    engine : str
        Graphviz engine executable.
    dpi : int, default=RAW_RENDER_DPI
        Rasterization DPI passed through to Graphviz.

    Returns
    -------
    None
        The rendered PNG is written to ``output_path``.
    """

    result = subprocess.run(
        [engine, f"-Gdpi={dpi}", "-Tpng", "-o", str(output_path)],
        input=dot_source,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Graphviz render failed")


def _render_dagua_png(
    graph: DaguaGraph,
    positions: torch.Tensor,
    output_path: Path,
    dpi: int = RAW_RENDER_DPI,
) -> None:
    """Render a Dagua graph to a PNG using fixed positions.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to render.
    positions : torch.Tensor
        Fixed node positions with shape ``[N, 2]``.
    output_path : Path
        Destination PNG path.
    dpi : int, default=RAW_RENDER_DPI
        Rasterization DPI passed through to Dagua's renderer.

    Returns
    -------
    None
        The rendered PNG is written to ``output_path``.
    """

    graph.compute_node_sizes()
    fig, _ = render(graph, positions, output=str(output_path), dpi=dpi)
    plt.close(fig)


def _content_crop_box(image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """Return a padded crop box around non-white content.

    Parameters
    ----------
    image : PIL.Image.Image
        Source image.

    Returns
    -------
    tuple[int, int, int, int] | None
        Crop bounds in PIL coordinates, or ``None`` when no content is found.
    """

    data = np.asarray(image.convert("RGBA"))
    content_mask = (data[:, :, 3] > 0) & np.any(data[:, :, :3] < 252, axis=2)
    if not bool(content_mask.any()):
        return None

    ys, xs = np.nonzero(content_mask)
    left = max(int(xs.min()) - CONTENT_CROP_PADDING, 0)
    top = max(int(ys.min()) - CONTENT_CROP_PADDING, 0)
    right = min(int(xs.max()) + CONTENT_CROP_PADDING + 1, image.width)
    bottom = min(int(ys.max()) + CONTENT_CROP_PADDING + 1, image.height)
    return left, top, right, bottom


def _crop_to_content(image: Image.Image) -> Image.Image:
    """Crop an image to its visible content when possible.

    Parameters
    ----------
    image : PIL.Image.Image
        Source image.

    Returns
    -------
    PIL.Image.Image
        Cropped image, or the original image when no content bounds are found.
    """

    crop_box = _content_crop_box(image)
    if crop_box is None:
        return image
    return image.crop(crop_box)


def _normalize_panel_image(image_path: Path, panel_size: Tuple[int, int]) -> Image.Image:
    """Resize and center an image onto a fixed white canvas.

    Parameters
    ----------
    image_path : Path
        Source image path.
    panel_size : tuple[int, int]
        Target panel size.

    Returns
    -------
    PIL.Image.Image
        Normalized RGB image.
    """

    with Image.open(image_path) as image:
        rgba = _crop_to_content(image.convert("RGBA"))
        rgba.thumbnail(
            (panel_size[0] - PANEL_CONTENT_MARGIN, panel_size[1] - PANEL_CONTENT_MARGIN),
            Image.LANCZOS,
        )
        canvas = Image.new("RGBA", panel_size, WHITE)
        offset = ((panel_size[0] - rgba.width) // 2, (panel_size[1] - rgba.height) // 2)
        canvas.paste(rgba, offset, rgba)
    return canvas.convert("RGB")


def _compose_comparison_image(
    dagua_image: Path,
    competitor_image: Path,
    title: str,
    competitor_label: str,
    output_path: Path,
) -> None:
    """Compose a two-panel comparison image with a shared title.

    Parameters
    ----------
    dagua_image : Path
        Rendered Dagua image.
    competitor_image : Path
        Rendered competitor image.
    title : str
        Figure title.
    competitor_label : str
        Label for the right-hand panel.
    output_path : Path
        Final output path.

    Returns
    -------
    None
        The composed image is written to ``output_path``.
    """

    dagua_panel = np.asarray(_normalize_panel_image(dagua_image, PANEL_SIZE))
    competitor_panel = np.asarray(_normalize_panel_image(competitor_image, PANEL_SIZE))

    fig, axes = plt.subplots(1, 2, figsize=COMPARISON_FIGSIZE)
    fig.patch.set_facecolor(WHITE)
    fig.suptitle(title, fontsize=15, fontweight="bold")

    axes[0].imshow(dagua_panel)
    axes[0].set_title("dagua", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(competitor_panel)
    axes[1].set_title(competitor_label, fontsize=12)
    axes[1].axis("off")

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(output_path, dpi=ALBUM_DPI, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)


def _compose_solo_image(
    dagua_image: Path,
    title: str,
    output_path: Path,
) -> None:
    """Compose a single-panel album image for Dagua-only features.

    Parameters
    ----------
    dagua_image : Path
        Rendered Dagua image.
    title : str
        Figure title.
    output_path : Path
        Final output path.

    Returns
    -------
    None
        The composed image is written to ``output_path``.
    """

    dagua_panel = np.asarray(_normalize_panel_image(dagua_image, PANEL_SIZE))

    fig, axis = plt.subplots(1, 1, figsize=SOLO_FIGSIZE)
    fig.patch.set_facecolor(WHITE)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    axis.imshow(dagua_panel)
    axis.set_title("dagua", fontsize=12)
    axis.axis("off")
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(output_path, dpi=ALBUM_DPI, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)


def _error_message(error: Exception) -> str:
    """Return a short, user-readable exception summary.

    Parameters
    ----------
    error : Exception
        Render-time exception.

    Returns
    -------
    str
        Compact ``"<type>: <message>"`` summary.
    """

    message = str(error).strip()
    if not message:
        return error.__class__.__name__
    return f"{error.__class__.__name__}: {message}"


def _compose_error_placeholder(title: str, error_text: str, output_path: Path) -> None:
    """Write a placeholder image that captures a per-case render failure.

    Parameters
    ----------
    title : str
        Album title for the failed case.
    error_text : str
        Human-readable error message.
    output_path : Path
        Destination placeholder path.

    Returns
    -------
    None
        The placeholder image is written to ``output_path``.
    """

    wrapped_error = textwrap.fill(error_text, width=ERROR_TEXT_WRAP)
    fig, axis = plt.subplots(1, 1, figsize=ERROR_PLACEHOLDER_FIGSIZE)
    fig.patch.set_facecolor(WHITE)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    axis.set_facecolor("#F8FAFC")
    axis.axis("off")
    axis.text(
        0.5,
        0.66,
        "Render failed",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="#B42318",
        transform=axis.transAxes,
    )
    axis.text(
        0.5,
        0.38,
        wrapped_error,
        ha="center",
        va="center",
        fontsize=11,
        color="#344054",
        family="monospace",
        transform=axis.transAxes,
    )
    fig.savefig(output_path, dpi=ALBUM_DPI, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)


def _compose_comparison(
    dagua_path: Path,
    competitor_path: Path,
    title: str,
    output_path: Path,
    competitor_label: str = GRAPHVIZ_COMPETITOR,
) -> None:
    """Compatibility wrapper exposing the shared comparison composer.

    Parameters
    ----------
    dagua_path : Path
        Rendered Dagua panel source.
    competitor_path : Path
        Rendered competitor panel source.
    title : str
        Figure title.
    output_path : Path
        Final composed output path.
    competitor_label : str, default=GRAPHVIZ_COMPETITOR
        Right-hand panel label.

    Returns
    -------
    None
        The composed image is written to ``output_path``.
    """

    _compose_comparison_image(
        dagua_image=dagua_path,
        competitor_image=competitor_path,
        title=title,
        competitor_label=competitor_label,
        output_path=output_path,
    )


def _compose_solo(
    dagua_path: Path,
    title: str,
    output_path: Path,
) -> None:
    """Compatibility wrapper exposing the shared solo composer.

    Parameters
    ----------
    dagua_path : Path
        Rendered Dagua panel source.
    title : str
        Figure title.
    output_path : Path
        Final composed output path.

    Returns
    -------
    None
        The composed image is written to ``output_path``.
    """

    _compose_solo_image(dagua_image=dagua_path, title=title, output_path=output_path)


def _competitor_cache_path(root: Path, case: AlbumCase) -> Path:
    """Return the persistent cache path for a competitor render.

    Parameters
    ----------
    root : Path
        Album root directory.
    case : AlbumCase
        Album case being rendered.

    Returns
    -------
    Path
        Cache path for the competitor PNG.
    """

    if case.graphviz is None:
        raise ValueError("Competitor cache path requested for a Dagua-only case.")
    return root / "_cache" / "competitor" / f"{case.case_id}_{case.graphviz.engine}.png"


def _resolve_competitor_image(
    case: AlbumCase,
    root: Path,
    temp_root: Path,
    dagua_only: bool,
    cache_competitor: bool,
) -> Path:
    """Return a competitor image path, rendering or reusing a cached PNG.

    Parameters
    ----------
    case : AlbumCase
        Album case being rendered.
    root : Path
        Album root directory.
    temp_root : Path
        Temporary directory for uncached renders.
    dagua_only : bool
        Whether competitor renders should be skipped in favor of cached images.
    cache_competitor : bool
        Whether to persist and reuse competitor renders.

    Returns
    -------
    Path
        Path to the competitor PNG.
    """

    if case.graphviz is None:
        raise ValueError("Competitor image requested for a Dagua-only case.")

    cache_path = _competitor_cache_path(root, case)
    if cache_competitor and cache_path.exists():
        return cache_path
    if dagua_only:
        raise RuntimeError(
            f"Missing cached competitor render for {case.case_id!r}. "
            "Run once without --dagua-only to populate the cache."
        )

    output_path = cache_path if cache_competitor else temp_root / f"{case.case_id}_graphviz.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dot_source = _build_graphviz_dot(case.graph, case.graphviz)
    _render_graphviz_png(dot_source, output_path, case.graphviz.engine)
    return output_path


def _case_output_path(root: Path, case: AlbumCase) -> Path:
    """Return the final output path for a case.

    Parameters
    ----------
    root : Path
        Album root directory.
    case : AlbumCase
        Album case.

    Returns
    -------
    Path
        Final artifact path.
    """

    return root / case.category / case.filename


def _manifest_entry(
    case: AlbumCase,
    output_path: Path,
    render_error: Optional[str] = None,
) -> Dict[str, object]:
    """Build the manifest row for one case.

    Parameters
    ----------
    case : AlbumCase
        Album case that was rendered.
    output_path : Path
        Final output path.
    render_error : str | None, default=None
        Optional render failure summary stored when a placeholder was emitted.

    Returns
    -------
    dict[str, object]
        JSON-serializable manifest record.
    """

    positions = case.positions.detach().cpu().tolist()
    node_positions = [
        {
            "node_index": index,
            "label": case.graph.node_labels[index],
            "position": positions[index],
        }
        for index in range(case.graph.num_nodes)
    ]
    entry: Dict[str, object] = {
        "case_id": case.case_id,
        "category": case.category,
        "filename": case.filename,
        "title": case.title,
        "output_path": str(output_path),
        "comparison": case.graphviz is not None,
        "competitor": case.graphviz.competitor_label if case.graphviz is not None else None,
        "num_nodes": case.graph.num_nodes,
        "num_edges": int(case.graph.edge_index.shape[1]),
        "node_positions": node_positions,
        "settings": case.settings,
        "placeholder": render_error is not None,
        "render_error": render_error,
    }
    if case.graphviz is not None:
        entry["graphviz_engine"] = case.graphviz.engine
        try:
            entry["graphviz_dot"] = _build_graphviz_dot(case.graph, case.graphviz)
        except Exception as error:  # pragma: no cover - defensive manifest fallback
            entry["graphviz_dot"] = None
            entry["graphviz_dot_error"] = _error_message(error)
    return entry


def _render_album_case(
    case: AlbumCase,
    root: Path,
    temp_root: Path,
    output_path: Path,
    dagua_only: bool,
    cache_competitor: bool,
) -> Optional[str]:
    """Render one album case or emit a placeholder image on failure.

    Parameters
    ----------
    case : AlbumCase
        Album case being rendered.
    root : Path
        Album root directory.
    temp_root : Path
        Temporary directory for intermediate PNGs.
    output_path : Path
        Final output path.
    dagua_only : bool
        Whether competitor renders should be reused from cache.
    cache_competitor : bool
        Whether competitor PNGs should be persisted under the album root.

    Returns
    -------
    str | None
        Error summary when a placeholder was emitted, otherwise ``None``.
    """

    try:
        dagua_raw = temp_root / f"{case.case_id}_dagua.png"
        _render_dagua_png(case.graph, case.positions, dagua_raw)

        if case.graphviz is not None:
            graphviz_raw = _resolve_competitor_image(
                case=case,
                root=root,
                temp_root=temp_root,
                dagua_only=dagua_only,
                cache_competitor=cache_competitor,
            )
            _compose_comparison_image(
                dagua_image=dagua_raw,
                competitor_image=graphviz_raw,
                title=case.title,
                competitor_label=case.graphviz.competitor_label,
                output_path=output_path,
            )
        else:
            _compose_solo_image(dagua_raw, case.title, output_path)
    except Exception as error:
        error_text = _error_message(error)
        _compose_error_placeholder(case.title, error_text, output_path)
        return error_text
    return None


def _node_shape_cases() -> List[AlbumCase]:
    """Build the exhaustive node-shape comparison cases.

    Returns
    -------
    list[AlbumCase]
        Node-shape cases.
    """

    shape_specs = [
        ("rectangle", "Rectangle", "rect", {"shape": "box"}),
        ("roundrect", "Round Rectangle", "roundrect", {"shape": "box", "style": "filled,rounded"}),
        ("ellipse", "Ellipse", "ellipse", {"shape": "ellipse"}),
        ("diamond", "Diamond", "diamond", {"shape": "diamond"}),
        ("circle", "Circle", "circle", {"shape": "circle"}),
        ("triangle", "Triangle", "triangle", {"shape": "triangle"}),
        ("hexagon", "Hexagon", "hexagon", {"shape": "hexagon"}),
        ("parallelogram", "Parallelogram", "parallelogram", {"shape": "parallelogram"}),
        ("pentagon", "Pentagon", "pentagon", {"shape": "pentagon"}),
        ("octagon", "Octagon", "octagon", {"shape": "octagon"}),
        ("star", "Star", "star", {"shape": "star"}),
        ("cylinder", "Cylinder", "cylinder", {"shape": "cylinder"}),
        ("trapezoid", "Trapezoid", "trapezoid", {"shape": "trapezium"}),
    ]
    cases: List[AlbumCase] = []
    for slug, title_name, dagua_shape, gv_node_attrs in shape_specs:
        graph, positions = _pair_graph(_top_bottom_pair_positions(), ["Source", "Target"])
        _set_all_node_styles(graph, _base_node_style(shape=dagua_shape))
        _set_all_edge_styles(graph, _base_edge_style())
        cases.append(
            AlbumCase(
                case_id=f"node_shape_{slug}",
                category="node_shapes",
                filename=f"{slug}_dagua_vs_graphviz.png",
                title=f"Node Shape: {title_name} - dagua vs Graphviz dot",
                graph=graph,
                positions=positions,
                settings={"kind": "node_shape", "dagua_shape": dagua_shape},
                graphviz=GraphvizRenderSpec(default_node_attrs=gv_node_attrs),
            )
        )
    return cases


def _arrow_type_cases() -> List[AlbumCase]:
    """Build the exhaustive arrow-type comparison cases.

    Returns
    -------
    list[AlbumCase]
        Arrow-type cases, including head, tail, and fill variants.
    """

    arrow_specs = [
        ("normal", "Normal", "normal", "normal"),
        ("vee", "Vee", "vee", "vee"),
        ("dot", "Dot", "dot", "dot"),
        ("diamond_arrow", "Diamond", "diamond", "diamond"),
        ("tee", "Tee", "tee", "tee"),
        ("crow", "Crow", "crow", "crow"),
        ("circle", "Circle", "circle", "circle"),
        ("none", "None", "none", "none"),
    ]
    cases: List[AlbumCase] = []
    for slug, title_name, dagua_arrow, gv_arrow in arrow_specs:
        graph, positions = _pair_graph(_top_bottom_pair_positions(), ["Upstream", "Downstream"])
        _set_all_node_styles(graph, _base_node_style())
        _set_all_edge_styles(graph, _base_edge_style(arrow=dagua_arrow))
        cases.append(
            AlbumCase(
                case_id=f"arrow_head_{slug}",
                category="arrow_types",
                filename=f"{slug}_dagua_vs_graphviz.png",
                title=f"Arrow Type: {title_name} - dagua vs Graphviz dot",
                graph=graph,
                positions=positions,
                settings={"kind": "arrow_head", "arrow": dagua_arrow},
                graphviz=GraphvizRenderSpec(default_edge_attrs={"arrowhead": gv_arrow}),
            )
        )

    for slug, title_name, dagua_arrow, gv_arrow in arrow_specs:
        graph, positions = _pair_graph(_top_bottom_pair_positions(), ["Start", "End"])
        _set_all_node_styles(graph, _base_node_style())
        _set_all_edge_styles(graph, _base_edge_style(arrow="none", tail_arrow=dagua_arrow))
        cases.append(
            AlbumCase(
                case_id=f"arrow_tail_{slug}",
                category="arrow_types",
                filename=f"tail_{slug}_dagua_vs_graphviz.png",
                title=f"Arrow Tail: {title_name} - dagua vs Graphviz dot",
                graph=graph,
                positions=positions,
                settings={"kind": "arrow_tail", "tail_arrow": dagua_arrow},
                graphviz=GraphvizRenderSpec(
                    default_edge_attrs={"dir": "back", "arrowhead": "none", "arrowtail": gv_arrow}
                ),
            )
        )

    fill_specs = [
        ("normal_filled", "Filled", "filled", "normal"),
        ("normal_hollow", "Hollow", "hollow", "empty"),
    ]
    for slug, title_name, dagua_fill, gv_arrow in fill_specs:
        graph, positions = _pair_graph(_top_bottom_pair_positions(), ["Filled", "Target"])
        _set_all_node_styles(graph, _base_node_style())
        _set_all_edge_styles(graph, _base_edge_style(arrow="normal", arrow_fill=dagua_fill))
        cases.append(
            AlbumCase(
                case_id=f"arrow_fill_{slug}",
                category="arrow_types",
                filename=f"{slug}_dagua_vs_graphviz.png",
                title=f"Arrow Fill: {title_name} - dagua vs Graphviz dot",
                graph=graph,
                positions=positions,
                settings={"kind": "arrow_fill", "arrow_fill": dagua_fill},
                graphviz=GraphvizRenderSpec(default_edge_attrs={"arrowhead": gv_arrow}),
            )
        )
    return cases


def _border_style_cases() -> List[AlbumCase]:
    """Build the node-border style comparison cases.

    Returns
    -------
    list[AlbumCase]
        Border-style cases.
    """

    border_specs = [
        ("solid", "Solid", "solid", "filled"),
        ("dashed", "Dashed", "dashed", "filled,dashed"),
        ("dotted", "Dotted", "dotted", "filled,dotted"),
    ]
    cases: List[AlbumCase] = []
    for slug, title_name, dagua_dash, gv_style in border_specs:
        graph, positions = _pair_graph(_top_bottom_pair_positions(), ["Node A", "Node B"])
        _set_all_node_styles(graph, _base_node_style(stroke_dash=dagua_dash))
        _set_all_edge_styles(graph, _base_edge_style(arrow="none"))
        cases.append(
            AlbumCase(
                case_id=f"border_style_{slug}",
                category="border_styles",
                filename=f"{slug}_dagua_vs_graphviz.png",
                title=f"Border Style: {title_name} - dagua vs Graphviz dot",
                graph=graph,
                positions=positions,
                settings={"kind": "border_style", "stroke_dash": dagua_dash},
                graphviz=GraphvizRenderSpec(default_node_attrs={"style": gv_style}),
            )
        )
    return cases


def _edge_style_cases() -> List[AlbumCase]:
    """Build the edge-line style comparison cases.

    Returns
    -------
    list[AlbumCase]
        Edge-style cases.
    """

    edge_specs = [
        ("solid", "Solid"),
        ("dashed", "Dashed"),
        ("dotted", "Dotted"),
    ]
    cases: List[AlbumCase] = []
    for slug, title_name in edge_specs:
        graph, positions = _pair_graph(_top_bottom_pair_positions(), ["Source", "Target"])
        _set_all_node_styles(graph, _base_node_style())
        _set_all_edge_styles(graph, _base_edge_style(style=slug))
        cases.append(
            AlbumCase(
                case_id=f"edge_style_{slug}",
                category="edge_styles",
                filename=f"{slug}_dagua_vs_graphviz.png",
                title=f"Edge Style: {title_name} - dagua vs Graphviz dot",
                graph=graph,
                positions=positions,
                settings={"kind": "edge_style", "style": slug},
                graphviz=GraphvizRenderSpec(default_edge_attrs={"style": slug}),
            )
        )
    return cases


def _edge_routing_cases() -> List[AlbumCase]:
    """Build the edge-routing comparison cases.

    Returns
    -------
    list[AlbumCase]
        Edge-routing cases.
    """

    routing_specs = [
        ("bezier", "Bezier", "bezier", "true"),
        ("straight", "Straight", "straight", "false"),
        ("ortho", "Orthogonal", "ortho", "ortho"),
    ]
    cases: List[AlbumCase] = []
    for slug, title_name, dagua_routing, gv_splines in routing_specs:
        if dagua_routing == "ortho":
            graph = DaguaGraph()
            _apply_graph_style(graph)
            graph.add_node("A", label="From")
            graph.add_node("B", label="To")
            graph.add_edge("A", "B")
            # Offset x so the orthogonal route shows a visible right-angle turn.
            positions = torch.tensor(
                [[-40.0, 55.0], [40.0, -55.0]],
                dtype=torch.float32,
            )
        else:
            graph, positions = _pair_graph(_top_bottom_pair_positions(), ["From", "To"])
        _set_all_node_styles(graph, _base_node_style())
        _set_all_edge_styles(graph, _base_edge_style(routing=dagua_routing))
        cases.append(
            AlbumCase(
                case_id=f"edge_routing_{slug}",
                category="edge_routing",
                filename=f"{slug}_dagua_vs_graphviz.png",
                title=f"Edge Routing: {title_name} - dagua vs Graphviz dot",
                graph=graph,
                positions=positions,
                settings={"kind": "edge_routing", "routing": dagua_routing},
                graphviz=GraphvizRenderSpec(graph_attrs={"splines": gv_splines}),
            )
        )
    return cases


def _text_formatting_cases() -> List[AlbumCase]:
    """Build font-weight, font-style, and alignment cases.

    Returns
    -------
    list[AlbumCase]
        Text-formatting cases.
    """

    cases: List[AlbumCase] = []

    weight_specs = [
        ("font_weight_regular", "Font Weight: Regular", "regular", "Helvetica"),
        ("font_weight_bold", "Font Weight: Bold", "bold", "Helvetica Bold"),
    ]
    for slug, title_name, dagua_weight, gv_font in weight_specs:
        graph, positions = _single_node_graph("Readable Title")
        _set_all_node_styles(graph, _base_node_style(font_weight=dagua_weight, min_width=180.0))
        cases.append(
            AlbumCase(
                case_id=slug,
                category="text_formatting",
                filename=f"{slug}_dagua_vs_graphviz.png",
                title=f"{title_name} - dagua vs Graphviz dot",
                graph=graph,
                positions=positions,
                settings={"kind": "font_weight", "font_weight": dagua_weight},
                graphviz=GraphvizRenderSpec(default_node_attrs={"fontname": gv_font}),
            )
        )

    style_specs = [
        ("font_style_normal", "Font Style: Normal", "normal", "Helvetica"),
        ("font_style_italic", "Font Style: Italic", "italic", "Helvetica Oblique"),
    ]
    for slug, title_name, dagua_style, gv_font in style_specs:
        graph, positions = _single_node_graph("Readable Title")
        _set_all_node_styles(graph, _base_node_style(font_style=dagua_style, min_width=180.0))
        cases.append(
            AlbumCase(
                case_id=slug,
                category="text_formatting",
                filename=f"{slug}_dagua_vs_graphviz.png",
                title=f"{title_name} - dagua vs Graphviz dot",
                graph=graph,
                positions=positions,
                settings={"kind": "font_style", "font_style": dagua_style},
                graphviz=GraphvizRenderSpec(default_node_attrs={"fontname": gv_font}),
            )
        )

    align_specs = [
        ("align_left", "left", "Left\naligned\nlabel"),
        ("align_center", "center", "Center\naligned\nlabel"),
        ("align_right", "right", "Right\naligned\nlabel"),
    ]
    for slug, align, label in align_specs:
        graph, positions = _single_node_graph(label)
        _set_all_node_styles(
            graph,
            _base_node_style(text_align=align, min_width=220.0, padding=(18.0, 12.0)),
        )
        cases.append(
            AlbumCase(
                case_id=f"text_alignment_{align}",
                category="text_formatting",
                filename=f"{slug}_dagua.png",
                title=f"Text Alignment: {align.title()} - dagua",
                graph=graph,
                positions=positions,
                settings={"kind": "text_alignment", "text_align": align},
            )
        )

    return cases


def _opacity_cases() -> List[AlbumCase]:
    """Build the node- and edge-opacity Dagua-only cases.

    Returns
    -------
    list[AlbumCase]
        Opacity cases.
    """

    cases: List[AlbumCase] = []

    for opacity, slug in [(1.0, "100"), (0.5, "50")]:
        graph, positions = _overlapping_nodes_graph()
        _set_all_node_styles(graph, _base_node_style(opacity=opacity, border_opacity=opacity))
        cases.append(
            AlbumCase(
                case_id=f"node_opacity_{slug}",
                category="opacity",
                filename=f"node_opacity_{slug}_dagua.png",
                title=f"Node Opacity: {slug}% - dagua",
                graph=graph,
                positions=positions,
                settings={"kind": "node_opacity", "opacity": opacity},
            )
        )

    for opacity, slug in [(1.0, "100"), (0.3, "30")]:
        graph, positions = _crossing_edges_graph()
        _set_all_node_styles(graph, _base_node_style(fill="#F4F7FA"))
        _set_all_edge_styles(graph, _base_edge_style(opacity=opacity))
        cases.append(
            AlbumCase(
                case_id=f"edge_opacity_{slug}",
                category="opacity",
                filename=f"edge_opacity_{slug}_dagua.png",
                title=f"Edge Opacity: {slug}% - dagua",
                graph=graph,
                positions=positions,
                settings={"kind": "edge_opacity", "opacity": opacity},
            )
        )
    return cases


def _shadow_cases() -> List[AlbumCase]:
    """Build the Dagua-only node shadow cases.

    Returns
    -------
    list[AlbumCase]
        Shadow cases.
    """

    cases: List[AlbumCase] = []
    for enabled in (True, False):
        slug = "on" if enabled else "off"
        graph, positions = _single_node_graph("Shadow")
        _set_all_node_styles(
            graph,
            _base_node_style(
                shadow=enabled,
                shadow_offset=(3.0, -3.0),
                shadow_color="#00000040",
                shadow_blur=3.0,
                min_width=160.0,
            ),
        )
        cases.append(
            AlbumCase(
                case_id=f"shadow_{slug}",
                category="shadows",
                filename=f"shadow_{slug}_dagua.png",
                title=f"Node Shadow: {slug.title()} - dagua",
                graph=graph,
                positions=positions,
                settings={"kind": "shadow", "shadow": enabled},
            )
        )
    return cases


def _gradient_cases() -> List[AlbumCase]:
    """Build the Dagua-only gradient cases.

    Returns
    -------
    list[AlbumCase]
        Gradient cases.
    """

    gradient_specs = [
        ("none", "None", "none"),
        ("linear", "Linear", "linear"),
        ("radial", "Radial", "radial"),
    ]
    cases: List[AlbumCase] = []
    for slug, title_name, gradient in gradient_specs:
        graph, positions = _single_node_graph("Gradient")
        _set_all_node_styles(
            graph,
            _base_node_style(
                gradient=gradient,
                gradient_color="#9FC7EE",
                min_width=180.0,
                corner_radius=14.0,
            ),
        )
        cases.append(
            AlbumCase(
                case_id=f"gradient_{slug}",
                category="gradients",
                filename=(
                    f"{slug}_gradient_dagua.png" if slug != "none" else "no_gradient_dagua.png"
                ),
                title=f"Node Gradient: {title_name} - dagua",
                graph=graph,
                positions=positions,
                settings={"kind": "gradient", "gradient": gradient},
            )
        )
    return cases


def _cluster_cases() -> List[AlbumCase]:
    """Build the cluster comparison cases.

    Returns
    -------
    list[AlbumCase]
        Cluster-style cases.
    """

    cases: List[AlbumCase] = []

    graph, positions = _basic_cluster_graph()
    _set_all_node_styles(graph, _base_node_style())
    _set_all_edge_styles(graph, _base_edge_style())
    graph.cluster_styles["group"] = _base_cluster_style(fill="#E6EEF7", opacity=0.55)
    cases.append(
        AlbumCase(
            case_id="cluster_fill",
            category="clusters",
            filename="cluster_fill_dagua_vs_graphviz.png",
            title="Cluster Style: Filled - dagua vs Graphviz dot",
            graph=graph,
            positions=positions,
            settings={"kind": "cluster_style", "variant": "fill"},
            graphviz=GraphvizRenderSpec(
                cluster_attrs={
                    "group": {
                        "style": "filled",
                        "color": CLUSTER_STROKE,
                        "fillcolor": "#E6EEF7",
                    }
                }
            ),
        )
    )

    graph, positions = _basic_cluster_graph()
    _set_all_node_styles(graph, _base_node_style())
    _set_all_edge_styles(graph, _base_edge_style())
    graph.cluster_styles["group"] = _base_cluster_style(
        fill=WHITE,
        stroke_dash="dashed",
    )
    cases.append(
        AlbumCase(
            case_id="cluster_border",
            category="clusters",
            filename="cluster_border_dagua_vs_graphviz.png",
            title="Cluster Style: Dashed Border - dagua vs Graphviz dot",
            graph=graph,
            positions=positions,
            settings={"kind": "cluster_style", "variant": "dashed_border"},
            graphviz=GraphvizRenderSpec(
                cluster_attrs={"group": {"style": "dashed", "color": CLUSTER_STROKE}}
            ),
        )
    )

    graph, positions = _nested_cluster_graph()
    _set_all_node_styles(graph, _base_node_style())
    _set_all_edge_styles(graph, _base_edge_style())
    graph.cluster_styles["outer"] = _base_cluster_style(fill="#EDF3F9", opacity=0.52)
    graph.cluster_styles["inner"] = _base_cluster_style(fill="#DDE8F4", opacity=0.65)
    cases.append(
        AlbumCase(
            case_id="cluster_nested",
            category="clusters",
            filename="cluster_nested_dagua_vs_graphviz.png",
            title="Cluster Style: Nested Clusters - dagua vs Graphviz dot",
            graph=graph,
            positions=positions,
            settings={"kind": "cluster_style", "variant": "nested"},
            graphviz=GraphvizRenderSpec(
                cluster_attrs={
                    "outer": {"style": "filled", "color": CLUSTER_STROKE, "fillcolor": "#EDF3F9"},
                    "inner": {"style": "filled", "color": "#8DA9C4", "fillcolor": "#DDE8F4"},
                }
            ),
        )
    )

    return cases


def _direction_cases() -> List[AlbumCase]:
    """Build the direction comparison cases.

    Returns
    -------
    list[AlbumCase]
        Direction cases.
    """

    cases: List[AlbumCase] = []
    for direction in ("TB", "BT", "LR", "RL"):
        graph, positions = _direction_graph(direction)
        _set_all_node_styles(graph, _base_node_style(min_width=140.0))
        _set_all_edge_styles(graph, _base_edge_style())
        cases.append(
            AlbumCase(
                case_id=f"direction_{direction.lower()}",
                category="direction",
                filename=f"{direction.lower()}_dagua_vs_graphviz.png",
                title=f"Direction: {direction} - dagua vs Graphviz dot",
                graph=graph,
                positions=positions,
                settings={"kind": "direction", "direction": direction},
                graphviz=GraphvizRenderSpec(graph_attrs={"rankdir": direction}),
            )
        )
    return cases


def _corner_radius_cases() -> List[AlbumCase]:
    """Build the Dagua-only corner-radius cases.

    Returns
    -------
    list[AlbumCase]
        Corner-radius cases.
    """

    cases: List[AlbumCase] = []
    for radius in (0, 6, 15):
        graph, positions = _single_node_graph("Rounded Corners")
        _set_all_node_styles(
            graph,
            _base_node_style(shape="roundrect", corner_radius=float(radius), min_width=210.0),
        )
        cases.append(
            AlbumCase(
                case_id=f"corner_radius_{radius}",
                category="corner_radius",
                filename=f"corner_radius_{radius}_dagua.png",
                title=f"Corner Radius: {radius} - dagua",
                graph=graph,
                positions=positions,
                settings={"kind": "corner_radius", "corner_radius": radius},
            )
        )
    return cases


def _rich_label_cases() -> List[AlbumCase]:
    """Build the Dagua-only rich-label cases.

    Returns
    -------
    list[AlbumCase]
        Rich-label cases.
    """

    cases: List[AlbumCase] = []

    graph, positions = _single_node_graph("**Bold** mixed with normal")
    _set_all_node_styles(
        graph,
        _base_node_style(label_format="rich", min_width=260.0, text_align="left"),
    )
    cases.append(
        AlbumCase(
            case_id="rich_label_bold_mixed",
            category="rich_labels",
            filename="bold_mixed_dagua.png",
            title="Rich Label: Bold Mixed - dagua",
            graph=graph,
            positions=positions,
            settings={"kind": "rich_label", "variant": "bold_mixed"},
        )
    )

    graph, positions = _single_node_graph("*Italic* mixed with normal")
    _set_all_node_styles(
        graph,
        _base_node_style(label_format="rich", min_width=220.0, text_align="left"),
    )
    cases.append(
        AlbumCase(
            case_id="rich_label_italic_mixed",
            category="rich_labels",
            filename="italic_mixed_dagua.png",
            title="Rich Label: Italic Mixed - dagua",
            graph=graph,
            positions=positions,
            settings={"kind": "rich_label", "variant": "italic_mixed"},
        )
    )

    return cases


def _combo_2way_cases() -> List[AlbumCase]:
    """Build the 2-way cosmetic combination cases.

    Returns
    -------
    list[AlbumCase]
        Dagua-only 2-way combo cases.
    """

    cases: List[AlbumCase] = []

    graph, positions = _combo_pair_graph(["Source", "Target"])
    _set_all_node_styles(graph, _combo_node_style(shadow=True, gradient="linear"))
    _set_all_edge_styles(graph, _combo_edge_style())
    cases.append(
        _combo_case(
            "shadow_gradient",
            "combo_2way",
            "Shadow + Gradient",
            graph,
            positions,
            {"combo": "2way", "shadow": True, "gradient": "linear"},
        )
    )

    graph, positions = _combo_pair_graph(["Start", "Finish"])
    _set_all_node_styles(graph, _combo_node_style(stroke_dash="dashed"))
    _set_all_edge_styles(graph, _combo_edge_style(arrow="diamond"))
    cases.append(
        _combo_case(
            "dashed_border_arrow",
            "combo_2way",
            "Dashed Border + Diamond Arrow",
            graph,
            positions,
            {"combo": "2way", "stroke_dash": "dashed", "arrow": "diamond"},
        )
    )

    graph, positions = _combo_pair_graph(["Bold italic source", "Bold italic target"])
    _set_all_node_styles(graph, _combo_node_style(font_weight="bold", font_style="italic"))
    cases.append(
        _combo_case(
            "bold_italic",
            "combo_2way",
            "Bold + Italic",
            graph,
            positions,
            {"combo": "2way", "font_weight": "bold", "font_style": "italic"},
        )
    )

    graph, positions = _overlapping_nodes_graph()
    _set_all_node_styles(graph, _combo_node_style(opacity=0.5, shadow=True))
    cases.append(
        _combo_case(
            "opacity_shadow",
            "combo_2way",
            "Opacity + Shadow",
            graph,
            positions,
            {"combo": "2way", "opacity": 0.5, "shadow": True},
        )
    )

    graph, positions = _combo_pair_graph(["Rounded source", "Rounded target"])
    _set_all_node_styles(graph, _combo_node_style(gradient="radial", corner_radius=12.0))
    cases.append(
        _combo_case(
            "gradient_rounded",
            "combo_2way",
            "Gradient + Rounded Corners",
            graph,
            positions,
            {"combo": "2way", "gradient": "radial", "corner_radius": 12},
        )
    )

    graph, positions = _combo_pair_graph(["A", "B"])
    _set_all_node_styles(graph, _combo_node_style())
    _set_all_edge_styles(graph, _combo_edge_style(style="dotted", arrow="vee"))
    cases.append(
        _combo_case(
            "dotted_edge_vee",
            "combo_2way",
            "Dotted Edge + Vee Arrow",
            graph,
            positions,
            {"combo": "2way", "edge_style": "dotted", "arrow": "vee"},
        )
    )

    graph, positions = _basic_cluster_graph()
    _set_all_node_styles(graph, _combo_node_style(gradient="linear"))
    _set_all_edge_styles(graph, _combo_edge_style())
    graph.cluster_styles["group"] = _base_cluster_style(fill="#E6EEF7", opacity=0.55)
    cases.append(
        _combo_case(
            "cluster_gradient",
            "combo_2way",
            "Cluster Fill + Gradient",
            graph,
            positions,
            {"combo": "2way", "cluster": True, "gradient": "linear"},
        )
    )

    graph, positions = _combo_pair_graph(["Double border", "Shadow target"])
    _set_all_node_styles(graph, _combo_node_style(border_count=2, shadow=True))
    cases.append(
        _combo_case(
            "double_border_shadow",
            "combo_2way",
            "Double Border + Shadow",
            graph,
            positions,
            {"combo": "2way", "border_count": 2, "shadow": True},
        )
    )

    graph, positions = _combo_pair_graph(["Wrapped bold source label", "Wrapped bold target label"])
    _set_all_node_styles(
        graph,
        _combo_node_style(
            font_weight="bold",
            text_wrap="wrap",
            text_max_width=COMBO_TEXT_MAX_WIDTH,
            min_width=160.0,
            padding=(12.0, 9.0),
        ),
    )
    cases.append(
        _combo_case(
            "text_wrap_bold",
            "combo_2way",
            "Text Wrap + Bold",
            graph,
            positions,
            {"combo": "2way", "text_wrap": "wrap", "font_weight": "bold"},
        )
    )

    graph, positions = _combo_pair_graph(["Striped source", "Dashed target"])
    _set_all_node_styles(graph, _combo_node_style(fill_pattern="striped", stroke_dash="dashed"))
    cases.append(
        _combo_case(
            "striped_fill_dashed",
            "combo_2way",
            "Striped Fill + Dashed Border",
            graph,
            positions,
            {"combo": "2way", "fill_pattern": "striped", "stroke_dash": "dashed"},
        )
    )

    graph, positions = _combo_pair_graph(["Source", "Target"])
    _set_all_node_styles(graph, _combo_node_style())
    _set_all_edge_styles(graph, _combo_edge_style(taper=True, color_gradient="source_to_target"))
    cases.append(
        _combo_case(
            "taper_gradient_edge",
            "combo_2way",
            "Taper + Edge Gradient",
            graph,
            positions,
            {"combo": "2way", "taper": True, "color_gradient": "source_to_target"},
        )
    )

    graph, positions = _crossing_edges_graph()
    _set_all_node_styles(graph, _combo_node_style(fill="#F4F7FA"))
    _set_all_edge_styles(graph, _combo_edge_style(crossing_style="arc", width=3.0))
    cases.append(
        _combo_case(
            "crossing_thick_edge",
            "combo_2way",
            "Crossing Arc + Thick Edge",
            graph,
            positions,
            {"combo": "2way", "crossing_style": "arc", "width": 3.0},
        )
    )

    graph, positions = _crossing_edges_graph()
    _set_all_node_styles(graph, _combo_node_style(fill="#F4F7FA"))
    _set_all_edge_styles(graph, _combo_edge_style(crossing_style="sharp", style="dashed"))
    cases.append(
        _combo_case(
            "crossing_sharp_dashed",
            "combo_2way",
            "Crossing Sharp + Dashed",
            graph,
            positions,
            {"combo": "2way", "crossing_style": "sharp", "edge_style": "dashed"},
        )
    )

    graph, positions = _combo_pair_graph(["Hexagon source", "Gradient target"])
    _set_all_node_styles(graph, _combo_node_style(shape="hexagon", gradient="linear"))
    cases.append(
        _combo_case(
            "hexagon_gradient",
            "combo_2way",
            "Hexagon + Gradient",
            graph,
            positions,
            {"combo": "2way", "shape": "hexagon", "gradient": "linear"},
        )
    )

    graph, positions = _combo_pair_graph(
        ["Diamond source", "Shadow target"],
        vertical_gap=COMBO_DIAMOND_PAIR_VERTICAL_GAP,
    )
    _set_all_node_styles(graph, _combo_node_style(shape="diamond", shadow=True))
    cases.append(
        _combo_case(
            "diamond_shadow",
            "combo_2way",
            "Diamond + Shadow",
            graph,
            positions,
            {"combo": "2way", "shape": "diamond", "shadow": True},
        )
    )

    graph, positions = _combo_pair_graph(["Star source", "Dotted target"])
    _set_all_node_styles(graph, _combo_node_style(shape="star", stroke_dash="dotted"))
    cases.append(
        _combo_case(
            "star_dotted",
            "combo_2way",
            "Star + Dotted Border",
            graph,
            positions,
            {"combo": "2way", "shape": "star", "stroke_dash": "dotted"},
        )
    )

    graph, positions = _combo_pair_graph(["Circle source", "Border target"])
    _set_all_node_styles(
        graph,
        _combo_node_style(shape="circle", border_count=2, min_width=120.0),
    )
    cases.append(
        _combo_case(
            "circle_double_border",
            "combo_2way",
            "Circle + Double Border",
            graph,
            positions,
            {"combo": "2way", "shape": "circle", "border_count": 2},
        )
    )

    graph, positions = _direction_graph("LR")
    _set_all_node_styles(graph, _combo_node_style(min_width=140.0))
    _set_all_edge_styles(graph, _combo_edge_style(routing="ortho"))
    cases.append(
        _combo_case(
            "lr_direction_ortho",
            "combo_2way",
            "LR Direction + Orthogonal Routing",
            graph,
            positions,
            {"combo": "2way", "direction": "LR", "routing": "ortho"},
        )
    )

    graph, positions = _overlapping_nodes_graph()
    _set_all_node_styles(graph, _combo_node_style(opacity=0.7, gradient="radial"))
    cases.append(
        _combo_case(
            "opacity_gradient",
            "combo_2way",
            "Opacity + Gradient",
            graph,
            positions,
            {"combo": "2way", "opacity": 0.7, "gradient": "radial"},
        )
    )

    graph, positions = _combo_pair_graph(["Italic source", "Size 12 target"])
    _set_all_node_styles(graph, _combo_node_style(font_style="italic", font_size=12.0))
    cases.append(
        _combo_case(
            "italic_large_font",
            "combo_2way",
            "Italic + Font Size 12",
            graph,
            positions,
            {"combo": "2way", "font_style": "italic", "font_size": 12},
        )
    )

    graph, positions = _combo_pair_graph(["Rounded source", "Shadow target"])
    _set_all_node_styles(
        graph,
        _combo_node_style(corner_radius=10.0, stroke_dash="dashed", shadow=True),
    )
    _set_all_edge_styles(graph, _combo_edge_style())
    cases.append(
        _combo_case(
            "rounded_dashed_shadow",
            "combo_2way",
            "Rounded Corners + Dashed Border + Shadow",
            graph,
            positions,
            {"combo": "2way", "corner_radius": 10, "stroke_dash": "dashed", "shadow": True},
        )
    )

    graph, positions = _combo_pair_graph(["Alpha", "Beta"])
    _set_all_node_styles(graph, _combo_node_style())
    _set_all_edge_styles(
        graph,
        _combo_edge_style(routing="bezier", color_gradient="source_to_target"),
    )
    cases.append(
        _combo_case(
            "taxi_gradient_edge",
            "combo_2way",
            "Bezier + Edge Gradient",
            graph,
            positions,
            {"combo": "2way", "routing": "bezier", "color_gradient": "source_to_target"},
        )
    )

    graph, positions = _combo_pair_graph(["Input", "Output"])
    _set_all_node_styles(graph, _combo_node_style(shadow=True))
    _set_all_edge_styles(graph, _combo_edge_style(routing="straight"))
    cases.append(
        _combo_case(
            "straight_shadow",
            "combo_2way",
            "Straight + Shadow",
            graph,
            positions,
            {"combo": "2way", "routing": "straight", "shadow": True},
        )
    )

    graph, positions = _combo_pair_graph(
        ["In", "Out"],
        vertical_gap=COMBO_DIAMOND_PAIR_VERTICAL_GAP,
    )
    _set_all_node_styles(
        graph,
        _combo_node_style(
            fill_pattern="pie",
            fill_pattern_colors=["#56B4E9", "#D55E00", "#009E73"],
            fill_pattern_values=[3, 2, 1],
            font_weight="bold",
            min_width=130.0,
        ),
    )
    _set_all_edge_styles(graph, _combo_edge_style())
    cases.append(
        _combo_case(
            "pie_bold",
            "combo_2way",
            "Pie Fill + Bold",
            graph,
            positions,
            {
                "combo": "2way",
                "fill_pattern": "pie",
                "fill_pattern_colors": ["#56B4E9", "#D55E00", "#009E73"],
                "fill_pattern_values": [3, 2, 1],
                "font_weight": "bold",
            },
        )
    )

    graph, positions = _combo_pair_graph(
        ["In", "Out"],
        vertical_gap=COMBO_DIAMOND_PAIR_VERTICAL_GAP,
    )
    _set_all_node_styles(
        graph,
        _combo_node_style(
            fill_pattern="pie",
            fill_pattern_colors=["#56B4E9", "#D55E00", "#009E73"],
            fill_pattern_values=[3, 2, 1],
            fill_pattern_hole=0.4,
            shadow=True,
            min_width=130.0,
        ),
    )
    _set_all_edge_styles(graph, _combo_edge_style())
    cases.append(
        _combo_case(
            "donut_shadow",
            "combo_2way",
            "Donut + Shadow",
            graph,
            positions,
            {
                "combo": "2way",
                "fill_pattern": "pie",
                "fill_pattern_colors": ["#56B4E9", "#D55E00", "#009E73"],
                "fill_pattern_values": [3, 2, 1],
                "fill_pattern_hole": 0.4,
                "shadow": True,
            },
        )
    )

    graph, positions = _combo_pair_graph(["Node", "Info"])
    _set_varied_external_label_styles(
        graph,
        external_label_position="bottom",
        corner_radius=12.0,
    )
    _set_all_edge_styles(graph, _combo_edge_style())
    cases.append(
        _combo_case(
            "external_label_rounded",
            "combo_2way",
            "External Label + Rounded",
            graph,
            positions,
            {
                "combo": "2way",
                "external_label_varied": True,
                "external_label_position": "bottom",
                "corner_radius": 12,
            },
        )
    )

    graph, positions = _combo_pair_graph(["Text", "Glow"])
    _set_all_node_styles(
        graph,
        _combo_node_style(
            text_outline=True,
            text_outline_color="#FFFFFF",
            text_outline_width=2.0,
            gradient="linear",
        ),
    )
    _set_all_edge_styles(graph, _combo_edge_style())
    cases.append(
        _combo_case(
            "text_outline_gradient",
            "combo_2way",
            "Text Outline + Gradient",
            graph,
            positions,
            {
                "combo": "2way",
                "text_outline": True,
                "text_outline_color": "#FFFFFF",
                "text_outline_width": 2.0,
                "gradient": "linear",
            },
        )
    )

    graph, positions = _combo_pair_graph(["Store", "Sink"])
    _set_all_node_styles(graph, _combo_node_style(shape="cylinder", stroke_dash="dashed"))
    _set_all_edge_styles(graph, _combo_edge_style())
    cases.append(
        _combo_case(
            "cylinder_dashed",
            "combo_2way",
            "Cylinder + Dashed",
            graph,
            positions,
            {"combo": "2way", "shape": "cylinder", "stroke_dash": "dashed"},
        )
    )

    graph, positions = _combo_pair_graph(["Sky", "Rain"])
    _set_all_node_styles(graph, _combo_node_style(shape="cloud", shadow=True))
    _set_all_edge_styles(graph, _combo_edge_style())
    cases.append(
        _combo_case(
            "cloud_shadow",
            "combo_2way",
            "Cloud + Shadow",
            graph,
            positions,
            {"combo": "2way", "shape": "cloud", "shadow": True},
        )
    )

    graph, positions = _combo_pair_graph(["Run", "Done"])
    _set_all_node_styles(graph, _combo_node_style(shape="stadium", gradient="linear"))
    _set_all_edge_styles(graph, _combo_edge_style())
    cases.append(
        _combo_case(
            "stadium_gradient",
            "combo_2way",
            "Stadium + Gradient",
            graph,
            positions,
            {"combo": "2way", "shape": "stadium", "gradient": "linear"},
        )
    )

    graph, positions = _combo_pair_graph(["Tab", "Pane"])
    _set_all_node_styles(graph, _combo_node_style(shape="tab", font_weight="bold"))
    _set_all_edge_styles(graph, _combo_edge_style())
    cases.append(
        _combo_case(
            "tab_bold",
            "combo_2way",
            "Tab + Bold",
            graph,
            positions,
            {"combo": "2way", "shape": "tab", "font_weight": "bold"},
        )
    )

    graph, positions = _combo_pair_graph(["Note", "Memo"])
    _set_all_node_styles(graph, _combo_node_style(shape="note", font_style="italic"))
    _set_all_edge_styles(graph, _combo_edge_style())
    cases.append(
        _combo_case(
            "note_italic",
            "combo_2way",
            "Note + Italic",
            graph,
            positions,
            {"combo": "2way", "shape": "note", "font_style": "italic"},
        )
    )

    graph, positions = _combo_pair_graph(["Doc", "File"])
    _set_all_node_styles(graph, _combo_node_style(shape="document", shadow=True))
    _set_all_edge_styles(graph, _combo_edge_style())
    cases.append(
        _combo_case(
            "document_shadow",
            "combo_2way",
            "Document + Shadow",
            graph,
            positions,
            {"combo": "2way", "shape": "document", "shadow": True},
        )
    )

    graph, positions = _combo_pair_graph(["3D", "Box"])
    _set_all_node_styles(graph, _combo_node_style(shape="box3d", gradient="linear"))
    _set_all_edge_styles(graph, _combo_edge_style())
    cases.append(
        _combo_case(
            "box3d_gradient",
            "combo_2way",
            "Box3D + Gradient",
            graph,
            positions,
            {"combo": "2way", "shape": "box3d", "gradient": "linear"},
        )
    )

    graph, positions = _combo_pair_graph(["In", "Out"])
    _set_all_node_styles(graph, _combo_node_style(shape="parallelogram"))
    _set_all_edge_styles(graph, _combo_edge_style(style="dotted"))
    cases.append(
        _combo_case(
            "parallelogram_dotted",
            "combo_2way",
            "Parallelogram + Dotted",
            graph,
            positions,
            {"combo": "2way", "shape": "parallelogram", "edge_style": "dotted"},
        )
    )

    graph, positions = _combo_pair_graph(["Trap", "Flow"])
    _set_all_node_styles(graph, _combo_node_style(shape="trapezoid", gradient="radial"))
    _set_all_edge_styles(graph, _combo_edge_style())
    cases.append(
        _combo_case(
            "trapezoid_gradient",
            "combo_2way",
            "Trapezoid + Gradient",
            graph,
            positions,
            {"combo": "2way", "shape": "trapezoid", "gradient": "radial"},
        )
    )

    graph, positions = _combo_pair_graph(["Pent", "Node"])
    _set_all_node_styles(graph, _combo_node_style(shape="pentagon", shadow=True))
    _set_all_edge_styles(graph, _combo_edge_style())
    cases.append(
        _combo_case(
            "pentagon_shadow",
            "combo_2way",
            "Pentagon + Shadow",
            graph,
            positions,
            {"combo": "2way", "shape": "pentagon", "shadow": True},
        )
    )

    graph, positions = _combo_pair_graph(["Oct", "Mark"])
    _set_all_node_styles(graph, _combo_node_style(shape="octagon", border_count=2))
    _set_all_edge_styles(graph, _combo_edge_style())
    cases.append(
        _combo_case(
            "octagon_double_border",
            "combo_2way",
            "Octagon + Double Border",
            graph,
            positions,
            {"combo": "2way", "shape": "octagon", "border_count": 2},
        )
    )

    graph, positions = _crossing_edges_graph()
    _set_all_node_styles(graph, _combo_node_style(fill="#F4F7FA"))
    _set_all_edge_styles(
        graph,
        _combo_edge_style(
            routing="straight",
            crossing_style="gap",
            crossing_size=20.0,
            width=3.5,
        ),
    )
    cases.append(
        _combo_case(
            "crossing_gap_thick",
            "combo_2way",
            "Crossing Gap + Thick",
            graph,
            positions,
            {
                "combo": "2way",
                "routing": "straight",
                "crossing_style": "gap",
                "crossing_size": 20.0,
                "width": 3.5,
            },
        )
    )

    graph, positions = _combo_pair_graph(["Input", "Output"])
    _set_all_node_styles(
        graph,
        _combo_node_style(fill_pattern="hatched", gradient="linear"),
    )
    _set_all_edge_styles(graph, _combo_edge_style())
    cases.append(
        _combo_case(
            "hatched_gradient",
            "combo_2way",
            "Hatched + Gradient",
            graph,
            positions,
            {"combo": "2way", "fill_pattern": "hatched", "gradient": "linear"},
        )
    )

    graph, positions = _direction_graph("BT")
    _set_all_node_styles(graph, _combo_node_style(shadow=True, min_width=140.0))
    _set_all_edge_styles(graph, _combo_edge_style())
    cases.append(
        _combo_case(
            "bt_direction_shadow",
            "combo_2way",
            "BT + Shadow",
            graph,
            positions,
            {"combo": "2way", "direction": "BT", "shadow": True},
        )
    )

    graph, positions = _direction_graph("RL")
    _set_all_node_styles(graph, _combo_node_style(gradient="linear", min_width=140.0))
    _set_all_edge_styles(graph, _combo_edge_style())
    cases.append(
        _combo_case(
            "rl_direction_gradient",
            "combo_2way",
            "RL + Gradient",
            graph,
            positions,
            {"combo": "2way", "direction": "RL", "gradient": "linear"},
        )
    )

    graph, positions = _combo_pair_graph(["Src", "Dst"])
    _set_all_node_styles(graph, _combo_node_style(font_weight="bold", font_size=12.0))
    _set_all_edge_styles(graph, _combo_edge_style(arrow="crow"))
    cases.append(
        _combo_case(
            "crow_arrow_bold",
            "combo_2way",
            "Crow Arrow + Bold",
            graph,
            positions,
            {"combo": "2way", "arrow": "crow", "font_weight": "bold", "font_size": 12.0},
        )
    )

    return cases


def _combo_3way_cases() -> List[AlbumCase]:
    """Build the 3-way cosmetic combination cases.

    Returns
    -------
    list[AlbumCase]
        Dagua-only 3-way combo cases.
    """

    cases: List[AlbumCase] = []

    graph, positions = _pair_graph(
        _top_bottom_pair_positions(),
        ["Bold shadow", "Gradient target"],
    )
    _set_all_node_styles(
        graph,
        _combo_node_style(
            font_weight="bold",
            shadow=True,
            gradient="linear",
            min_height=40.0,
        ),
    )
    _set_all_edge_styles(graph, _combo_edge_style())
    cases.append(
        _combo_case(
            "bold_shadow_gradient",
            "combo_3way",
            "Bold + Shadow + Gradient",
            graph,
            positions,
            {"combo": "3way", "font_weight": "bold", "shadow": True, "gradient": "linear"},
        )
    )

    graph, positions = _overlapping_nodes_graph()
    _set_all_node_styles(
        graph,
        _combo_node_style(stroke_dash="dashed", shape="diamond", opacity=0.6),
    )
    cases.append(
        _combo_case(
            "dashed_diamond_opacity",
            "combo_3way",
            "Dashed Border + Diamond + Opacity",
            graph,
            positions,
            {"combo": "3way", "stroke_dash": "dashed", "shape": "diamond", "opacity": 0.6},
        )
    )

    graph, positions = _basic_cluster_graph()
    _set_all_node_styles(graph, _combo_node_style(corner_radius=8.0, gradient="radial"))
    _set_all_edge_styles(graph, _combo_edge_style())
    graph.cluster_styles["group"] = _base_cluster_style(fill="#E6EEF7", opacity=0.55)
    cases.append(
        _combo_case(
            "cluster_rounded_gradient",
            "combo_3way",
            "Cluster + Rounded Corners + Gradient",
            graph,
            positions,
            {"combo": "3way", "cluster": True, "corner_radius": 8, "gradient": "radial"},
        )
    )

    graph, positions = _crossing_edges_graph()
    _set_all_node_styles(graph, _combo_node_style(fill="#F4F7FA"))
    _set_all_edge_styles(graph, _combo_edge_style(taper=True, crossing_style="arc", width=2.5))
    cases.append(
        _combo_case(
            "taper_crossing_thick",
            "combo_3way",
            "Taper + Crossing Arc + Thick Edge",
            graph,
            positions,
            {"combo": "3way", "taper": True, "crossing_style": "arc", "width": 2.5},
        )
    )

    graph, positions = _pair_graph(
        _top_bottom_pair_positions(),
        ["Italic hexagon", "Shadow target"],
    )
    _set_all_node_styles(
        graph,
        _combo_node_style(font_style="italic", shape="hexagon", shadow=True),
    )
    cases.append(
        _combo_case(
            "italic_hexagon_shadow",
            "combo_3way",
            "Italic + Hexagon + Shadow",
            graph,
            positions,
            {"combo": "3way", "font_style": "italic", "shape": "hexagon", "shadow": True},
        )
    )

    graph, positions = _pair_graph(
        _top_bottom_pair_positions(),
        ["Double gradient", "Rounded target"],
    )
    _set_all_node_styles(
        graph,
        _combo_node_style(border_count=2, gradient="linear", corner_radius=10.0),
    )
    cases.append(
        _combo_case(
            "double_border_gradient_rounded",
            "combo_3way",
            "Double Border + Gradient + Rounded Corners",
            graph,
            positions,
            {"combo": "3way", "border_count": 2, "gradient": "linear", "corner_radius": 10},
        )
    )

    graph, positions = _pair_graph(_top_bottom_pair_positions(), ["Pattern", "Route"])
    _set_all_node_styles(graph, _combo_node_style(fill_pattern="striped", shadow=True))
    _set_all_edge_styles(graph, _combo_edge_style(style="dashed"))
    cases.append(
        _combo_case(
            "striped_shadow_dashed_edge",
            "combo_3way",
            "Striped Fill + Shadow + Dashed Edge",
            graph,
            positions,
            {"combo": "3way", "fill_pattern": "striped", "shadow": True, "edge_style": "dashed"},
        )
    )

    graph, positions = _pair_graph(
        _top_bottom_pair_positions(),
        ["Rounded highlighted source", "Rounded highlighted target"],
    )
    _set_all_node_styles(
        graph,
        _combo_node_style(
            text_background="#FFE0B2",
            font_weight="bold",
            corner_radius=8.0,
            min_width=200.0,
        ),
    )
    cases.append(
        _combo_case(
            "text_bg_bold_rounded",
            "combo_3way",
            "Text Background + Bold + Rounded Corners",
            graph,
            positions,
            {
                "combo": "3way",
                "text_background": "#FFE0B2",
                "font_weight": "bold",
                "corner_radius": 8,
            },
        )
    )

    graph, positions = _direction_graph("LR")
    _set_all_node_styles(graph, _combo_node_style(min_width=140.0))
    _set_all_edge_styles(graph, _combo_edge_style(routing="bezier", arrow="vee"))
    cases.append(
        _combo_case(
            "lr_bezier_vee",
            "combo_3way",
            "LR Direction + Bezier Routing + Vee Arrow",
            graph,
            positions,
            {"combo": "3way", "direction": "LR", "routing": "bezier", "arrow": "vee"},
        )
    )

    graph, positions = _basic_cluster_graph()
    _set_all_node_styles(graph, _combo_node_style(opacity=0.6, stroke_dash="dashed"))
    _set_all_edge_styles(graph, _combo_edge_style())
    graph.cluster_styles["group"] = _base_cluster_style(fill="#E6EEF7", opacity=0.55)
    cases.append(
        _combo_case(
            "opacity_cluster_dashed",
            "combo_3way",
            "Opacity + Cluster + Dashed Border",
            graph,
            positions,
            {"combo": "3way", "opacity": 0.6, "cluster": True, "stroke_dash": "dashed"},
        )
    )

    graph, positions = _combo_pair_graph(["A", "B"])
    _set_all_node_styles(graph, _combo_node_style(shadow=True, gradient="linear"))
    _set_all_edge_styles(graph, _combo_edge_style(routing="taxi"))
    cases.append(
        _combo_case(
            "taxi_shadow_gradient",
            "combo_3way",
            "Taxi + Shadow + Gradient",
            graph,
            positions,
            {"combo": "3way", "routing": "taxi", "shadow": True, "gradient": "linear"},
        )
    )

    graph, positions = _combo_pair_graph(
        ["In", "Out"],
        vertical_gap=COMBO_DIAMOND_PAIR_VERTICAL_GAP,
    )
    _set_all_node_styles(
        graph,
        _combo_node_style(
            fill_pattern="pie",
            fill_pattern_colors=["#56B4E9", "#D55E00", "#009E73"],
            fill_pattern_values=[3, 2, 1],
            gradient="linear",
            font_weight="bold",
            min_width=130.0,
        ),
    )
    _set_all_edge_styles(graph, _combo_edge_style())
    cases.append(
        _combo_case(
            "pie_gradient_bold",
            "combo_3way",
            "Pie + Gradient + Bold",
            graph,
            positions,
            {
                "combo": "3way",
                "fill_pattern": "pie",
                "fill_pattern_colors": ["#56B4E9", "#D55E00", "#009E73"],
                "fill_pattern_values": [3, 2, 1],
                "gradient": "linear",
                "font_weight": "bold",
            },
        )
    )

    graph, positions = _combo_pair_graph(
        ["Dia", "Tag"],
        vertical_gap=COMBO_DIAMOND_PAIR_VERTICAL_GAP,
    )
    _set_varied_external_label_styles(
        graph,
        external_label_position="bottom",
        shape="diamond",
        shadow=True,
    )
    _set_all_edge_styles(graph, _combo_edge_style())
    cases.append(
        _combo_case(
            "external_label_diamond_shadow",
            "combo_3way",
            "Ext Label + Diamond + Shadow",
            graph,
            positions,
            {
                "combo": "3way",
                "external_label_varied": True,
                "external_label_position": "bottom",
                "shape": "diamond",
                "shadow": True,
            },
        )
    )

    graph, positions = _combo_pair_graph(["Cld", "Txt"])
    _set_all_node_styles(
        graph,
        _combo_node_style(shape="cloud", gradient="linear", font_style="italic"),
    )
    _set_all_edge_styles(graph, _combo_edge_style())
    cases.append(
        _combo_case(
            "cloud_gradient_italic",
            "combo_3way",
            "Cloud + Gradient + Italic",
            graph,
            positions,
            {"combo": "3way", "shape": "cloud", "gradient": "linear", "font_style": "italic"},
        )
    )

    graph, positions = _combo_pair_graph(["Run", "Stop"])
    _set_all_node_styles(
        graph,
        _combo_node_style(shape="stadium", fill_pattern="striped", shadow=True),
    )
    _set_all_edge_styles(graph, _combo_edge_style())
    cases.append(
        _combo_case(
            "stadium_striped_shadow",
            "combo_3way",
            "Stadium + Striped + Shadow",
            graph,
            positions,
            {"combo": "3way", "shape": "stadium", "fill_pattern": "striped", "shadow": True},
        )
    )

    graph, positions = _combo_pair_graph(["Hat", "Ink"])
    _set_all_node_styles(
        graph,
        _combo_node_style(fill_pattern="hatched", shadow=True, font_weight="bold"),
    )
    _set_all_edge_styles(graph, _combo_edge_style())
    cases.append(
        _combo_case(
            "hatched_shadow_bold",
            "combo_3way",
            "Hatched + Shadow + Bold",
            graph,
            positions,
            {"combo": "3way", "fill_pattern": "hatched", "shadow": True, "font_weight": "bold"},
        )
    )

    graph, positions = _combo_pair_graph(["Src", "Dst"])
    _set_all_node_styles(graph, _combo_node_style())
    _set_all_edge_styles(
        graph,
        _combo_edge_style(
            head_label="H",
            tail_label="T",
            head_label_offset=18.0,
            tail_label_offset=18.0,
            routing="ortho",
            arrow="none",
        ),
    )
    cases.append(
        _combo_case(
            "head_tail_labels_ortho",
            "combo_3way",
            "Head/Tail Labels + Ortho",
            graph,
            positions,
            {
                "combo": "3way",
                "head_label": "H",
                "tail_label": "T",
                "head_label_offset": 18.0,
                "tail_label_offset": 18.0,
                "routing": "ortho",
                "arrow": "none",
            },
        )
    )

    graph, positions = _basic_cluster_graph()
    graph.direction = "BT"
    _set_all_node_styles(graph, _combo_node_style(corner_radius=10.0))
    _set_all_edge_styles(graph, _combo_edge_style())
    graph.cluster_styles["group"] = _base_cluster_style(fill="#E6EEF7", opacity=0.55)
    cases.append(
        _combo_case(
            "bt_cluster_rounded",
            "combo_3way",
            "BT + Cluster + Rounded",
            graph,
            positions,
            {"combo": "3way", "direction": "BT", "cluster": True, "corner_radius": 10},
        )
    )

    graph, positions = _combo_pair_graph(["Txt", "Out"])
    _set_all_node_styles(
        graph,
        _combo_node_style(
            text_outline=True,
            text_outline_color="#333333",
            text_outline_width=2.0,
            shadow=True,
            font_weight="bold",
        ),
    )
    _set_all_edge_styles(graph, _combo_edge_style())
    cases.append(
        _combo_case(
            "text_outline_shadow_bold",
            "combo_3way",
            "Text Outline + Shadow + Bold",
            graph,
            positions,
            {
                "combo": "3way",
                "text_outline": True,
                "text_outline_color": "#333333",
                "text_outline_width": 2.0,
                "shadow": True,
                "font_weight": "bold",
            },
        )
    )

    graph, positions = _combo_pair_graph(["A", "B"])
    _set_all_node_styles(graph, _combo_node_style())
    _set_all_edge_styles(
        graph,
        _combo_edge_style(color_gradient="source_to_target", taper=True, width=4.0),
    )
    cases.append(
        _combo_case(
            "color_gradient_taper_thick",
            "combo_3way",
            "Edge Gradient + Taper + Thick",
            graph,
            positions,
            {"combo": "3way", "color_gradient": "source_to_target", "taper": True, "width": 4.0},
        )
    )

    return cases


def _combo_4way_cases() -> List[AlbumCase]:
    """Build the 4-way cosmetic combination cases.

    Returns
    -------
    list[AlbumCase]
        Dagua-only 4-way combo cases.
    """

    cases: List[AlbumCase] = []

    graph, positions = _pair_graph(
        _top_bottom_pair_positions(),
        ["Rounded bold", "Gradient target"],
    )
    _set_all_node_styles(
        graph,
        _combo_node_style(
            shape="roundrect",
            font_weight="bold",
            shadow=True,
            gradient="linear",
            corner_radius=12.0,
            min_width=210.0,
        ),
    )
    _set_all_edge_styles(graph, _combo_edge_style())
    cases.append(
        _combo_case(
            "bold_shadow_gradient_rounded",
            "combo_4way",
            "Bold + Shadow + Gradient + Rounded Corners",
            graph,
            positions,
            {
                "combo": "4way",
                "font_weight": "bold",
                "shadow": True,
                "gradient": "linear",
                "corner_radius": 12,
            },
        )
    )

    graph, positions = _overlapping_nodes_graph()
    _set_all_node_styles(
        graph,
        _combo_node_style(shape="diamond", stroke_dash="dashed", opacity=0.6, font_style="italic"),
    )
    cases.append(
        _combo_case(
            "diamond_dashed_opacity_italic",
            "combo_4way",
            "Diamond + Dashed Border + Opacity + Italic",
            graph,
            positions,
            {
                "combo": "4way",
                "shape": "diamond",
                "stroke_dash": "dashed",
                "opacity": 0.6,
                "font_style": "italic",
            },
        )
    )

    graph, positions = _basic_cluster_graph()
    _set_all_node_styles(graph, _combo_node_style(gradient="radial", shadow=True, border_count=2))
    _set_all_edge_styles(graph, _combo_edge_style())
    graph.cluster_styles["group"] = _base_cluster_style(fill="#E6EEF7", opacity=0.55)
    cases.append(
        _combo_case(
            "cluster_gradient_shadow_double_border",
            "combo_4way",
            "Cluster + Gradient + Shadow + Double Border",
            graph,
            positions,
            {
                "combo": "4way",
                "cluster": True,
                "gradient": "radial",
                "shadow": True,
                "border_count": 2,
            },
        )
    )

    graph, positions = _crossing_edges_graph()
    _set_all_node_styles(graph, _combo_node_style(fill="#F4F7FA"))
    _set_all_edge_styles(
        graph,
        _combo_edge_style(
            taper=True,
            crossing_style="arc",
            color_gradient="source_to_target",
            width=2.5,
        ),
    )
    cases.append(
        _combo_case(
            "taper_crossing_gradient_thick",
            "combo_4way",
            "Taper + Crossing Arc + Edge Gradient + Thick Edge",
            graph,
            positions,
            {
                "combo": "4way",
                "taper": True,
                "crossing_style": "arc",
                "color_gradient": "source_to_target",
                "width": 2.5,
            },
        )
    )

    graph, positions = _pair_graph(
        _top_bottom_pair_positions(),
        ["Bold striped", "Hexagon target"],
    )
    _set_all_node_styles(
        graph,
        _combo_node_style(shape="hexagon", fill_pattern="striped", shadow=True, font_weight="bold"),
    )
    cases.append(
        _combo_case(
            "hexagon_striped_shadow_bold",
            "combo_4way",
            "Hexagon + Striped Fill + Shadow + Bold",
            graph,
            positions,
            {
                "combo": "4way",
                "shape": "hexagon",
                "fill_pattern": "striped",
                "shadow": True,
                "font_weight": "bold",
            },
        )
    )

    graph, positions = _combo_pair_graph(
        ["In", "Out"],
        vertical_gap=COMBO_DIAMOND_PAIR_VERTICAL_GAP,
    )
    _set_all_node_styles(
        graph,
        _combo_node_style(
            fill_pattern="pie",
            fill_pattern_colors=["#56B4E9", "#D55E00", "#009E73"],
            fill_pattern_values=[3, 2, 1],
            shadow=True,
            gradient="linear",
            font_weight="bold",
            min_width=130.0,
        ),
    )
    _set_all_edge_styles(graph, _combo_edge_style())
    cases.append(
        _combo_case(
            "pie_shadow_gradient_bold",
            "combo_4way",
            "Pie + Shadow + Gradient + Bold",
            graph,
            positions,
            {
                "combo": "4way",
                "fill_pattern": "pie",
                "fill_pattern_colors": ["#56B4E9", "#D55E00", "#009E73"],
                "fill_pattern_values": [3, 2, 1],
                "shadow": True,
                "gradient": "linear",
                "font_weight": "bold",
            },
        )
    )

    graph, positions = _combo_pair_graph(["DB", "ET"])
    _set_all_node_styles(
        graph,
        _combo_node_style(
            shape="cylinder",
            stroke_dash="dashed",
            shadow=True,
            gradient="linear",
        ),
    )
    _set_all_edge_styles(graph, _combo_edge_style())
    cases.append(
        _combo_case(
            "cylinder_dashed_shadow_gradient",
            "combo_4way",
            "Cylinder + Dashed + Shadow + Gradient",
            graph,
            positions,
            {
                "combo": "4way",
                "shape": "cylinder",
                "stroke_dash": "dashed",
                "shadow": True,
                "gradient": "linear",
            },
        )
    )

    graph, positions = _crossing_edges_graph()
    _set_all_node_styles(graph, _combo_node_style(fill="#F4F7FA", shadow=True, gradient="linear"))
    _set_all_edge_styles(graph, _combo_edge_style(routing="taxi", crossing_style="gap"))
    cases.append(
        _combo_case(
            "taxi_crossing_gap_gradient",
            "combo_4way",
            "Taxi + Crossing Gap + Gradient + Shadow",
            graph,
            positions,
            {
                "combo": "4way",
                "routing": "taxi",
                "crossing_style": "gap",
                "gradient": "linear",
                "shadow": True,
            },
        )
    )

    graph, positions = _combo_pair_graph(["CL", "IT"])
    _set_all_node_styles(
        graph,
        _combo_node_style(
            shape="cloud",
            fill_pattern="striped",
            shadow=True,
            font_style="italic",
        ),
    )
    _set_all_edge_styles(graph, _combo_edge_style())
    cases.append(
        _combo_case(
            "cloud_striped_shadow_italic",
            "combo_4way",
            "Cloud + Striped + Shadow + Italic",
            graph,
            positions,
            {
                "combo": "4way",
                "shape": "cloud",
                "fill_pattern": "striped",
                "shadow": True,
                "font_style": "italic",
            },
        )
    )

    graph, positions = _combo_pair_graph(["HX", "MD"])
    _set_varied_external_label_styles(
        graph,
        external_label_position="bottom",
        shape="hexagon",
        gradient="linear",
        font_weight="bold",
    )
    _set_all_edge_styles(graph, _combo_edge_style())
    cases.append(
        _combo_case(
            "ext_label_hexagon_gradient_bold",
            "combo_4way",
            "Ext Label + Hexagon + Gradient + Bold",
            graph,
            positions,
            {
                "combo": "4way",
                "external_label_varied": True,
                "external_label_position": "bottom",
                "shape": "hexagon",
                "gradient": "linear",
                "font_weight": "bold",
            },
        )
    )

    return cases


def _combo_5way_cases() -> List[AlbumCase]:
    """Build the 5-way cosmetic combination cases.

    Returns
    -------
    list[AlbumCase]
        Dagua-only 5-way combo cases.
    """

    cases: List[AlbumCase] = []

    graph, positions = _crossing_edges_graph()
    _set_all_node_styles(
        graph,
        _combo_node_style(
            shape="hexagon",
            shadow=True,
            gradient="linear",
            stroke_dash="dashed",
            font_weight="bold",
        ),
    )
    _set_all_edge_styles(graph, _combo_edge_style())
    cases.append(
        _combo_case(
            "kitchen_sink_1",
            "combo_5way",
            "Hexagon + Shadow + Gradient + Dashed Border + Bold",
            graph,
            positions,
            {
                "combo": "5way",
                "shape": "hexagon",
                "shadow": True,
                "gradient": "linear",
                "stroke_dash": "dashed",
                "font_weight": "bold",
            },
        )
    )

    graph, positions = _crossing_edges_graph()
    _set_all_node_styles(
        graph,
        _combo_node_style(
            shape="roundrect",
            border_count=2,
            opacity=0.7,
            font_style="italic",
            corner_radius=8.0,
        ),
    )
    _set_all_edge_styles(graph, _combo_edge_style())
    cases.append(
        _combo_case(
            "kitchen_sink_2",
            "combo_5way",
            "Roundrect + Double Border + Opacity + Italic + Rounded Corners",
            graph,
            positions,
            {
                "combo": "5way",
                "shape": "roundrect",
                "border_count": 2,
                "opacity": 0.7,
                "font_style": "italic",
                "corner_radius": 8,
            },
        )
    )

    graph, positions = _crossing_edges_graph()
    _set_all_node_styles(graph, _combo_node_style(fill_pattern="striped", shadow=True))
    _set_all_edge_styles(
        graph,
        _combo_edge_style(taper=True, crossing_style="arc", color_gradient="source_to_target"),
    )
    cases.append(
        _combo_case(
            "kitchen_sink_3",
            "combo_5way",
            "Striped Fill + Shadow + Taper + Crossing Arc + Edge Gradient",
            graph,
            positions,
            {
                "combo": "5way",
                "fill_pattern": "striped",
                "shadow": True,
                "taper": True,
                "crossing_style": "arc",
                "color_gradient": "source_to_target",
            },
        )
    )

    graph, positions = _combo_pair_graph(["B3D", "EG"])
    _set_all_node_styles(
        graph,
        _combo_node_style(
            shape="box3d",
            shadow=True,
            text_outline=True,
            text_outline_color="#FFFFFF",
            text_outline_width=2.0,
        ),
    )
    _set_all_edge_styles(
        graph,
        _combo_edge_style(taper=True, color_gradient="source_to_target"),
    )
    cases.append(
        _combo_case(
            "kitchen_sink_4",
            "combo_5way",
            "Box3D + Shadow + Outline + Taper + Edge Gradient",
            graph,
            positions,
            {
                "combo": "5way",
                "shape": "box3d",
                "shadow": True,
                "text_outline": True,
                "text_outline_color": "#FFFFFF",
                "text_outline_width": 2.0,
                "taper": True,
                "color_gradient": "source_to_target",
            },
        )
    )

    graph, positions = _combo_pair_graph(
        ["Pie", "Lbl"],
        vertical_gap=COMBO_DIAMOND_PAIR_VERTICAL_GAP,
    )
    _set_varied_external_label_styles(
        graph,
        fill_pattern="pie",
        fill_pattern_colors=["#56B4E9", "#D55E00", "#009E73"],
        fill_pattern_values=[3, 2, 1],
        fill_pattern_hole=0.3,
        gradient="linear",
        font_weight="bold",
        external_label_position="bottom",
        min_width=140.0,
    )
    _set_all_edge_styles(graph, _combo_edge_style())
    cases.append(
        _combo_case(
            "kitchen_sink_5",
            "combo_5way",
            "Pie Donut + Gradient + Bold + Ext Label",
            graph,
            positions,
            {
                "combo": "5way",
                "fill_pattern": "pie",
                "fill_pattern_colors": ["#56B4E9", "#D55E00", "#009E73"],
                "fill_pattern_values": [3, 2, 1],
                "fill_pattern_hole": 0.3,
                "gradient": "linear",
                "font_weight": "bold",
                "external_label_varied": True,
                "external_label_position": "bottom",
            },
        )
    )

    graph, positions = _crossing_edges_graph()
    _set_all_node_styles(
        graph,
        _combo_node_style(shape="stadium", shadow=True, gradient="linear", fill="#F4F7FA"),
    )
    _set_all_edge_styles(graph, _combo_edge_style(routing="taxi", crossing_style="gap"))
    cases.append(
        _combo_case(
            "kitchen_sink_6",
            "combo_5way",
            "Stadium + Taxi + Shadow + Gradient + Crossing Gap",
            graph,
            positions,
            {
                "combo": "5way",
                "shape": "stadium",
                "routing": "taxi",
                "shadow": True,
                "gradient": "linear",
                "crossing_style": "gap",
            },
        )
    )

    return cases


def _evil_combo_cases() -> List[AlbumCase]:
    """Build pathological renderer stress cases.

    Returns
    -------
    list[AlbumCase]
        Dagua-only cases covering extreme, contradictory, and stress scenarios.
    """

    cases: List[AlbumCase] = []

    graph, positions = _pair_graph(_top_bottom_pair_positions(), ["Thin", "Target"])
    _set_all_node_styles(graph, _base_node_style())
    _set_all_edge_styles(graph, _base_edge_style(arrow_length=30.0, arrow_width=20.0, width=0.5))
    cases.append(
        AlbumCase(
            case_id="evil_huge_arrows",
            category="evil_combos",
            filename="evil_huge_arrows_dagua.png",
            title="Evil Combo: Huge Arrows - dagua",
            graph=graph,
            positions=positions,
            settings={
                "kind": "evil_combo",
                "variant": "huge_arrows",
                "arrow_length": 30.0,
                "arrow_width": 20.0,
                "edge_width": 0.5,
            },
        )
    )

    graph, positions = _single_node_graph("Text Does Not Fit")
    _set_all_node_styles(
        graph,
        _base_node_style(
            min_width=10.0,
            min_height=8.0,
            font_size=14.0,
            overflow_policy="overflow",
        ),
    )
    cases.append(
        AlbumCase(
            case_id="evil_tiny_nodes_big_text",
            category="evil_combos",
            filename="evil_tiny_nodes_big_text_dagua.png",
            title="Evil Combo: Tiny Nodes, Big Text - dagua",
            graph=graph,
            positions=positions,
            settings={
                "kind": "evil_combo",
                "variant": "tiny_nodes_big_text",
                "min_width": 10.0,
                "min_height": 8.0,
                "font_size": 14.0,
            },
        )
    )

    graph, positions = _stacked_nodes_graph()
    graph.node_styles = [
        _base_node_style(
            fill=fill, opacity=1.0, border_opacity=1.0, min_width=72.0, min_height=48.0
        )
        for fill in ("#D55E00", "#56B4E9", "#009E73", "#E69F00", "#CC79A7")
    ]
    cases.append(
        AlbumCase(
            case_id="evil_max_opacity_stack",
            category="evil_combos",
            filename="evil_max_opacity_stack_dagua.png",
            title="Evil Combo: Max Opacity Stack - dagua",
            graph=graph,
            positions=positions,
            settings={
                "kind": "evil_combo",
                "variant": "max_opacity_stack",
                "node_count": 5,
                "opacity": 1.0,
            },
        )
    )

    graph, positions = _single_node_graph("x")
    _set_all_node_styles(graph, _base_node_style(min_width=1.0, min_height=1.0, font_size=1.0))
    cases.append(
        AlbumCase(
            case_id="evil_zero_size",
            category="evil_combos",
            filename="evil_zero_size_dagua.png",
            title="Evil Combo: Zero Size - dagua",
            graph=graph,
            positions=positions,
            settings={
                "kind": "evil_combo",
                "variant": "zero_size",
                "min_width": 1.0,
                "min_height": 1.0,
                "font_size": 1.0,
            },
        )
    )

    graph = DaguaGraph()
    _apply_graph_style(graph)
    graph.add_node("A", label="Curve A")
    graph.add_node("B", label="Curve B")
    graph.add_edge("A", "B")
    positions = torch.tensor([[-90.0, 60.0], [90.0, -60.0]], dtype=torch.float32)
    _set_all_node_styles(graph, _base_node_style())
    _set_all_edge_styles(graph, _base_edge_style(curvature=2.0, width=4.0))
    cases.append(
        AlbumCase(
            case_id="evil_extreme_curvature",
            category="evil_combos",
            filename="evil_extreme_curvature_dagua.png",
            title="Evil Combo: Extreme Curvature - dagua",
            graph=graph,
            positions=positions,
            settings={
                "kind": "evil_combo",
                "variant": "extreme_curvature",
                "curvature": 2.0,
                "edge_width": 4.0,
            },
        )
    )

    graph, positions = _single_node_graph("Ghost")
    _set_all_node_styles(
        graph,
        _base_node_style(fill=WHITE, stroke=WHITE, font_color=WHITE, min_width=140.0),
    )
    cases.append(
        AlbumCase(
            case_id="evil_invisible_on_invisible",
            category="evil_combos",
            filename="evil_invisible_on_invisible_dagua.png",
            title="Evil Combo: Invisible on Invisible - dagua",
            graph=graph,
            positions=positions,
            settings={
                "kind": "evil_combo",
                "variant": "invisible_on_invisible",
                "fill": WHITE,
                "stroke": WHITE,
            },
        )
    )

    graph = DaguaGraph()
    _apply_graph_style(graph)
    graph.add_node("A", label="From")
    graph.add_node("B", label="To")
    graph.add_edge("A", "B")
    positions = torch.tensor([[-40.0, 55.0], [40.0, -55.0]], dtype=torch.float32)
    _set_all_node_styles(graph, _base_node_style(shape="diamond"))
    _set_all_edge_styles(graph, _base_edge_style(routing="ortho"))
    cases.append(
        AlbumCase(
            case_id="evil_ortho_diamond",
            category="evil_combos",
            filename="evil_ortho_diamond_dagua.png",
            title="Evil Combo: Ortho Diamond - dagua",
            graph=graph,
            positions=positions,
            settings={
                "kind": "evil_combo",
                "variant": "ortho_diamond",
                "routing": "ortho",
                "shape": "diamond",
            },
        )
    )

    graph, positions = _crossing_edges_graph()
    _set_all_node_styles(graph, _base_node_style(fill="#F4F7FA"))
    _set_all_edge_styles(graph, _base_edge_style(taper=True, crossing_style="arc", style="dashed"))
    for edge_style in graph.edge_styles:
        if edge_style is not None:
            edge_style.taper_width_start = 4.0
            edge_style.taper_width_end = 0.7
    cases.append(
        AlbumCase(
            case_id="evil_taper_crossing_dashed",
            category="evil_combos",
            filename="evil_taper_crossing_dashed_dagua.png",
            title="Evil Combo: Taper Crossing Dashed - dagua",
            graph=graph,
            positions=positions,
            settings={
                "kind": "evil_combo",
                "variant": "taper_crossing_dashed",
                "taper": True,
                "crossing_style": "arc",
                "style": "dashed",
            },
        )
    )

    graph, positions = _single_node_graph("Pie Shadow Gradient")
    _set_all_node_styles(
        graph,
        _base_node_style(
            shadow=True,
            shadow_blur=4.0,
            gradient="linear",
            gradient_color="#A0C4E8",
            fill_pattern="pie",
            fill_pattern_colors=["#56B4E9", "#D55E00", "#009E73"],
            fill_pattern_values=[3.0, 2.0, 1.0],
            min_width=200.0,
            min_height=130.0,
        ),
    )
    cases.append(
        AlbumCase(
            case_id="evil_pie_shadow_gradient",
            category="evil_combos",
            filename="evil_pie_shadow_gradient_dagua.png",
            title="Evil Combo: Pie Shadow Gradient - dagua",
            graph=graph,
            positions=positions,
            settings={
                "kind": "evil_combo",
                "variant": "pie_shadow_gradient",
                "fill_pattern": "pie",
                "shadow": True,
                "gradient": "linear",
            },
        )
    )

    graph, positions = _single_node_graph("Double Border")
    _set_all_node_styles(
        graph,
        _base_node_style(
            border_count=2,
            fill_pattern="hatched",
            gradient="radial",
            gradient_color="#9FC7EE",
            min_width=200.0,
        ),
    )
    cases.append(
        AlbumCase(
            case_id="evil_double_border_striped",
            category="evil_combos",
            filename="evil_double_border_striped_dagua.png",
            title="Evil Combo: Double Border Striped - dagua",
            graph=graph,
            positions=positions,
            settings={
                "kind": "evil_combo",
                "variant": "double_border_striped",
                "border_count": 2,
                "fill_pattern": "hatched",
                "gradient": "radial",
            },
        )
    )

    graph, positions = _deep_cluster_graph()
    _set_all_node_styles(graph, _base_node_style())
    _set_all_edge_styles(graph, _base_edge_style())
    for cluster_name, fill, stroke in (
        ("level_1", "#F2F6FB", "#9AAFC4"),
        ("level_2", "#E7F0F8", "#7E9CB8"),
        ("level_3", "#D8E7F3", "#6489AB"),
        ("level_4", "#C7DDEF", "#466F98"),
    ):
        graph.cluster_styles[cluster_name] = _base_cluster_style(
            fill=fill,
            stroke=stroke,
            stroke_width=2.0,
            opacity=0.7,
        )
    cases.append(
        AlbumCase(
            case_id="evil_deep_clusters",
            category="evil_combos",
            filename="evil_deep_clusters_dagua.png",
            title="Evil Combo: Deep Clusters - dagua",
            graph=graph,
            positions=positions,
            settings={
                "kind": "evil_combo",
                "variant": "deep_clusters",
                "cluster_depth": 4,
            },
        )
    )

    graph, positions = _many_arrow_graph()
    _set_all_node_styles(graph, _base_node_style())
    arrow_types = ("normal", "vee", "dot", "diamond", "tee", "crow", "circle", "open")
    graph.edge_styles = [
        _base_edge_style(arrow=arrow_type, width=2.0, arrow_length=16.0, arrow_width=12.0)
        for arrow_type in arrow_types
    ]
    cases.append(
        AlbumCase(
            case_id="evil_many_arrows",
            category="evil_combos",
            filename="evil_many_arrows_dagua.png",
            title="Evil Combo: Many Arrows - dagua",
            graph=graph,
            positions=positions,
            settings={
                "kind": "evil_combo",
                "variant": "many_arrows",
                "arrow_types": list(arrow_types),
            },
        )
    )

    graph, positions = _single_node_graph("This is a very long label that should overflow")
    _set_all_node_styles(
        graph,
        _base_node_style(shape="diamond", min_width=120.0, overflow_policy="overflow"),
    )
    cases.append(
        AlbumCase(
            case_id="evil_long_label_diamond",
            category="evil_combos",
            filename="evil_long_label_diamond_dagua.png",
            title="Evil Combo: Long Label Diamond - dagua",
            graph=graph,
            positions=positions,
            settings={
                "kind": "evil_combo",
                "variant": "long_label_diamond",
                "shape": "diamond",
                "label": "This is a very long label that should overflow",
            },
        )
    )

    graph, positions = _self_loop_graph()
    _set_all_node_styles(graph, _base_node_style(shape="circle", min_width=120.0, min_height=120.0))
    _set_all_edge_styles(
        graph,
        _base_edge_style(
            taper=True,
            arrow="crow",
            color_gradient="source_to_target",
            color_gradient_end="#D55E00",
            width=2.4,
        ),
    )
    if graph.edge_styles[0] is not None:
        graph.edge_styles[0].taper_width_start = 4.2
        graph.edge_styles[0].taper_width_end = 0.6
    cases.append(
        AlbumCase(
            case_id="evil_self_loop_styled",
            category="evil_combos",
            filename="evil_self_loop_styled_dagua.png",
            title="Evil Combo: Self Loop Styled - dagua",
            graph=graph,
            positions=positions,
            settings={
                "kind": "evil_combo",
                "variant": "self_loop_styled",
                "taper": True,
                "arrow": "crow",
                "color_gradient": "source_to_target",
            },
        )
    )

    graph, positions = _crossing_edges_graph()
    _set_all_node_styles(
        graph,
        _base_node_style(
            shape="hexagon",
            shadow=True,
            gradient="linear",
            gradient_color="#A0C4E8",
            border_count=2,
            stroke_dash="dashed",
            font_weight="bold",
            font_style="italic",
            opacity=0.8,
            corner_radius=8.0,
            fill_pattern="striped",
            fill_pattern_colors=["#56B4E9", "#D55E00"],
            fill_pattern_angle=35.0,
            min_width=120.0,
        ),
    )
    _set_all_edge_styles(
        graph,
        _base_edge_style(
            taper=True,
            crossing_style="arc",
            arrow="diamond",
            color_gradient="source_to_target",
            color_gradient_end="#D55E00",
            width=2.5,
            style="dashed",
        ),
    )
    for edge_style in graph.edge_styles:
        if edge_style is not None:
            edge_style.taper_width_start = 4.0
            edge_style.taper_width_end = 0.8
    cases.append(
        AlbumCase(
            case_id="evil_all_at_once",
            category="evil_combos",
            filename="evil_all_at_once_dagua.png",
            title="Evil Combo: All At Once - dagua",
            graph=graph,
            positions=positions,
            settings={
                "kind": "evil_combo",
                "variant": "all_at_once",
                "node_shape": "hexagon",
                "edge_arrow": "diamond",
            },
        )
    )

    graph, positions = _self_loop_graph()
    _set_all_node_styles(graph, _base_node_style(shape="star", min_width=120.0, min_height=120.0))
    _set_all_edge_styles(
        graph,
        _base_edge_style(width=2.0, arrow="normal", arrow_length=5.0, arrow_width=3.5),
    )
    cases.append(
        AlbumCase(
            case_id="evil_self_loop_star",
            category="evil_combos",
            filename="evil_self_loop_star_dagua.png",
            title="Evil Combo: Self Loop Star - dagua",
            graph=graph,
            positions=positions,
            settings={"kind": "evil_combo", "variant": "self_loop_star", "shape": "star"},
        )
    )

    graph, positions = _self_loop_graph()
    _set_all_node_styles(
        graph,
        _base_node_style(shape="diamond", min_width=120.0, min_height=120.0),
    )
    _set_all_edge_styles(
        graph,
        _base_edge_style(width=2.0, arrow="normal", arrow_length=5.0, arrow_width=3.5),
    )
    cases.append(
        AlbumCase(
            case_id="evil_self_loop_diamond",
            category="evil_combos",
            filename="evil_self_loop_diamond_dagua.png",
            title="Evil Combo: Self Loop Diamond - dagua",
            graph=graph,
            positions=positions,
            settings={"kind": "evil_combo", "variant": "self_loop_diamond", "shape": "diamond"},
        )
    )

    graph, positions = _self_loop_graph()
    _set_all_node_styles(
        graph,
        _base_node_style(shape="triangle", min_width=120.0, min_height=120.0),
    )
    _set_all_edge_styles(
        graph,
        _base_edge_style(width=2.0, arrow="normal", arrow_length=5.0, arrow_width=3.5),
    )
    cases.append(
        AlbumCase(
            case_id="evil_self_loop_triangle",
            category="evil_combos",
            filename="evil_self_loop_triangle_dagua.png",
            title="Evil Combo: Self Loop Triangle - dagua",
            graph=graph,
            positions=positions,
            settings={
                "kind": "evil_combo",
                "variant": "self_loop_triangle",
                "shape": "triangle",
            },
        )
    )

    graph, positions = _single_node_graph("Long text wrapping inside concave shape")
    _set_all_node_styles(
        graph,
        _base_node_style(
            shape="star",
            text_wrap="wrap",
            text_max_width=80.0,
            min_width=100.0,
            min_height=100.0,
            font_size=7.0,
            overflow_policy="shrink_text",
            min_font_size=4.0,
        ),
    )
    cases.append(
        AlbumCase(
            case_id="evil_long_wrap_star",
            category="evil_combos",
            filename="evil_long_wrap_star_dagua.png",
            title="Evil Combo: Long Wrap Star - dagua",
            graph=graph,
            positions=positions,
            settings={
                "kind": "evil_combo",
                "variant": "long_wrap_star",
                "shape": "star",
                "text_wrap": "wrap",
            },
        )
    )

    graph, positions = _single_node_graph("Long text wrapping inside concave shape")
    _set_all_node_styles(
        graph,
        _base_node_style(
            shape="triangle",
            text_wrap="wrap",
            text_max_width=80.0,
            min_width=100.0,
            min_height=100.0,
            font_size=7.0,
            overflow_policy="shrink_text",
            min_font_size=4.0,
        ),
    )
    cases.append(
        AlbumCase(
            case_id="evil_long_wrap_triangle",
            category="evil_combos",
            filename="evil_long_wrap_triangle_dagua.png",
            title="Evil Combo: Long Wrap Triangle - dagua",
            graph=graph,
            positions=positions,
            settings={
                "kind": "evil_combo",
                "variant": "long_wrap_triangle",
                "shape": "triangle",
                "text_wrap": "wrap",
            },
        )
    )

    graph = DaguaGraph()
    _apply_graph_style(graph)
    graph.add_node("hub", label="Hub")
    for index in range(24):
        graph.add_node(f"s{index}", label=f"S{index}")
        graph.add_edge(f"s{index}", "hub")
    spoke_positions: List[List[float]] = []
    for index in range(24):
        angle = 2.0 * math.pi * index / 24.0
        spoke_positions.append([220.0 * math.cos(angle), 220.0 * math.sin(angle)])
    positions = torch.tensor([[0.0, 0.0]] + spoke_positions, dtype=torch.float32)
    _set_all_node_styles(graph, _base_node_style(min_width=40.0, font_size=8.0))
    _set_all_edge_styles(graph, _base_edge_style(width=1.0))
    cases.append(
        AlbumCase(
            case_id="evil_mega_hub",
            category="evil_combos",
            filename="evil_mega_hub_dagua.png",
            title="Evil Combo: Mega Hub - dagua",
            graph=graph,
            positions=positions,
            settings={"kind": "evil_combo", "variant": "mega_hub", "edge_count": 24},
        )
    )

    graph, positions = _combo_pair_graph(["Src", "Dst"])
    _set_all_node_styles(graph, _base_node_style())
    _set_all_edge_styles(
        graph,
        _base_edge_style(width=0.1, arrow_length=25.0, arrow_width=18.0),
    )
    cases.append(
        AlbumCase(
            case_id="evil_zero_width_big_arrow",
            category="evil_combos",
            filename="evil_zero_width_big_arrow_dagua.png",
            title="Evil Combo: Zero Width Big Arrow - dagua",
            graph=graph,
            positions=positions,
            settings={
                "kind": "evil_combo",
                "variant": "zero_width_big_arrow",
                "width": 0.1,
                "arrow_length": 25,
            },
        )
    )

    graph = DaguaGraph()
    _apply_graph_style(graph)
    labels = ["Overflow text here", "Shrink me down", "Expand to fit"]
    policies = ["overflow", "shrink_text", "expand_node"]
    for index, label in enumerate(labels):
        graph.add_node(f"n{index}", label=label)
    for index in range(2):
        graph.add_edge(f"n{index}", f"n{index + 1}")
    positions = torch.tensor(
        [[0.0, 150.0], [0.0, 0.0], [0.0, -150.0]],
        dtype=torch.float32,
    )
    graph.node_styles = [
        _base_node_style(
            overflow_policy=policy,
            shape="rect",
            min_width=180.0,
            font_size=10.0,
            text_wrap="wrap",
            text_max_width=150.0,
        )
        for policy in policies
    ]
    _set_all_edge_styles(graph, _base_edge_style())
    cases.append(
        AlbumCase(
            case_id="evil_mixed_overflow",
            category="evil_combos",
            filename="evil_mixed_overflow_dagua.png",
            title="Evil Combo: Mixed Overflow - dagua",
            graph=graph,
            positions=positions,
            settings={
                "kind": "evil_combo",
                "variant": "mixed_overflow",
                "policies": policies,
            },
        )
    )

    graph = DaguaGraph()
    _apply_graph_style(graph)
    for index in range(4):
        graph.add_node(f"n{index}", label="")
    graph.add_edge("n0", "n1")
    graph.add_edge("n2", "n3")
    positions = torch.tensor(
        [[-80.0, 60.0], [80.0, 60.0], [-80.0, -60.0], [80.0, -60.0]],
        dtype=torch.float32,
    )
    _set_all_node_styles(
        graph,
        _base_node_style(
            shadow=True,
            gradient="linear",
            gradient_color="#9FC7EE",
            border_count=2,
            min_width=80.0,
        ),
    )
    _set_all_edge_styles(
        graph,
        _base_edge_style(
            taper=True,
            color_gradient="source_to_target",
            color_gradient_end="#D55E00",
        ),
    )
    cases.append(
        AlbumCase(
            case_id="evil_empty_labels",
            category="evil_combos",
            filename="evil_empty_labels_dagua.png",
            title="Evil Combo: Empty Labels - dagua",
            graph=graph,
            positions=positions,
            settings={"kind": "evil_combo", "variant": "empty_labels", "labels": ""},
        )
    )

    graph = DaguaGraph()
    _apply_graph_style(graph)
    unicode_labels = ["Cafe\u0301", "\u03b1\u2192\u03b2", "\u00fcber", "2\u00b2+3\u00b2=13"]
    for index, label in enumerate(unicode_labels):
        graph.add_node(f"n{index}", label=label)
    graph.add_edge("n0", "n1")
    graph.add_edge("n2", "n3")
    positions = torch.tensor(
        [[-80.0, 60.0], [80.0, 60.0], [-80.0, -60.0], [80.0, -60.0]],
        dtype=torch.float32,
    )
    _set_all_node_styles(graph, _base_node_style(min_width=90.0))
    _set_all_edge_styles(graph, _base_edge_style())
    cases.append(
        AlbumCase(
            case_id="evil_unicode_labels",
            category="evil_combos",
            filename="evil_unicode_labels_dagua.png",
            title="Evil Combo: Unicode Labels - dagua",
            graph=graph,
            positions=positions,
            settings={"kind": "evil_combo", "variant": "unicode_labels"},
        )
    )

    graph = DaguaGraph()
    _apply_graph_style(graph)
    graph.add_node("A", label="Neg Curve A")
    graph.add_node("B", label="Neg Curve B")
    graph.add_edge("A", "B")
    positions = torch.tensor([[-90.0, 60.0], [90.0, -60.0]], dtype=torch.float32)
    _set_all_node_styles(graph, _base_node_style(min_width=120.0))
    _set_all_edge_styles(graph, _base_edge_style(curvature=-1.5, width=3.0))
    cases.append(
        AlbumCase(
            case_id="evil_negative_curvature",
            category="evil_combos",
            filename="evil_negative_curvature_dagua.png",
            title="Evil Combo: Negative Curvature - dagua",
            graph=graph,
            positions=positions,
            settings={
                "kind": "evil_combo",
                "variant": "negative_curvature",
                "curvature": -1.5,
            },
        )
    )

    graph = DaguaGraph()
    _apply_graph_style(graph)
    for row in range(10):
        for col in range(10):
            graph.add_node(f"n{row}_{col}", label=f"{row},{col}")
    for row in range(10):
        for col in range(10):
            if col < 9:
                graph.add_edge(f"n{row}_{col}", f"n{row}_{col + 1}")
            if row < 9:
                graph.add_edge(f"n{row}_{col}", f"n{row + 1}_{col}")
    positions = torch.tensor(
        [[col * 60.0 - 270.0, row * 60.0 - 270.0] for row in range(10) for col in range(10)],
        dtype=torch.float32,
    )
    _set_all_node_styles(
        graph,
        _base_node_style(
            min_width=40.0,
            min_height=28.0,
            font_size=7.0,
            gradient="linear",
            gradient_color="#B0D4F1",
            shadow=True,
            shadow_blur=2.0,
        ),
    )
    _set_all_edge_styles(graph, _base_edge_style(width=1.0, arrow="normal"))
    cases.append(
        AlbumCase(
            case_id="evil_hundred_nodes",
            category="evil_combos",
            filename="evil_hundred_nodes_dagua.png",
            title="Evil Combo: Hundred Nodes - dagua",
            graph=graph,
            positions=positions,
            settings={"kind": "evil_combo", "variant": "hundred_nodes", "node_count": 100},
        )
    )

    graph = DaguaGraph()
    _apply_graph_style(graph)
    for index in range(8):
        graph.add_node(f"n{index}", label=f"N{index}")
    for index in range(7):
        graph.add_edge(f"n{index}", f"n{index + 1}")
    graph.add_cluster("L1", [f"n{index}" for index in range(8)], label="Level 1")
    graph.add_cluster("L2", [f"n{index}" for index in range(1, 7)], label="Level 2", parent="L1")
    graph.add_cluster("L3", [f"n{index}" for index in range(2, 6)], label="Level 3", parent="L2")
    graph.add_cluster("L4", [f"n{index}" for index in range(3, 5)], label="Level 4", parent="L3")
    graph.add_cluster("L5", ["n3", "n4"], label="Level 5", parent="L4")
    positions = torch.tensor(
        [
            [-120.0, 60.0],
            [0.0, 60.0],
            [120.0, 60.0],
            [240.0, 60.0],
            [-120.0, -60.0],
            [0.0, -60.0],
            [120.0, -60.0],
            [240.0, -60.0],
        ],
        dtype=torch.float32,
    )
    _set_all_node_styles(graph, _base_node_style(min_width=50.0, font_size=9.0))
    _set_all_edge_styles(graph, _base_edge_style(width=1.5))
    blues = [
        "#F0F5FA",
        "#E0EAF4",
        "#D0DFF0",
        "#C0D4EA",
        "#B0C9E4",
    ]
    for depth, fill in enumerate(blues, start=1):
        graph.cluster_styles[f"L{depth}"] = _base_cluster_style(
            fill=fill,
            stroke="#6A8CAF",
            stroke_width=2.0,
            opacity=0.7,
            font_size=10.0,
            padding=35.0,
        )
    cases.append(
        AlbumCase(
            case_id="evil_8_deep_clusters",
            category="evil_combos",
            filename="evil_8_deep_clusters_dagua.png",
            title="Evil Combo: 5 Deep Clusters - dagua",
            graph=graph,
            positions=positions,
            settings={"kind": "evil_combo", "variant": "5_deep_clusters", "cluster_depth": 5},
        )
    )

    graph, positions = _single_node_graph("Pie")
    _set_all_node_styles(
        graph,
        _base_node_style(
            shape="star",
            fill_pattern="pie",
            fill_pattern_colors=["#56B4E9", "#D55E00", "#009E73", "#E69F00"],
            fill_pattern_values=[4.0, 3.0, 2.0, 1.0],
            min_width=160.0,
            min_height=160.0,
        ),
    )
    cases.append(
        AlbumCase(
            case_id="evil_pie_star",
            category="evil_combos",
            filename="evil_pie_star_dagua.png",
            title="Evil Combo: Pie Star - dagua",
            graph=graph,
            positions=positions,
            settings={
                "kind": "evil_combo",
                "variant": "pie_star",
                "shape": "star",
                "fill_pattern": "pie",
            },
        )
    )

    graph, positions = _single_node_graph("Donut")
    _set_all_node_styles(
        graph,
        _base_node_style(
            shape="diamond",
            fill_pattern="pie",
            fill_pattern_colors=["#56B4E9", "#D55E00", "#009E73"],
            fill_pattern_values=[3.0, 2.0, 1.0],
            fill_pattern_hole=0.4,
            min_width=150.0,
            min_height=150.0,
        ),
    )
    cases.append(
        AlbumCase(
            case_id="evil_donut_diamond",
            category="evil_combos",
            filename="evil_donut_diamond_dagua.png",
            title="Evil Combo: Donut Diamond - dagua",
            graph=graph,
            positions=positions,
            settings={
                "kind": "evil_combo",
                "variant": "donut_diamond",
                "shape": "diamond",
                "fill_pattern": "pie",
                "fill_pattern_hole": 0.4,
            },
        )
    )

    graph, positions = _self_loop_graph()
    _set_all_node_styles(graph, _base_node_style(min_width=100.0, min_height=70.0))
    _set_all_edge_styles(graph, _base_edge_style(routing="taxi", width=2.0, arrow="vee"))
    cases.append(
        AlbumCase(
            case_id="evil_taxi_self_loop",
            category="evil_combos",
            filename="evil_taxi_self_loop_dagua.png",
            title="Evil Combo: Taxi Self Loop - dagua",
            graph=graph,
            positions=positions,
            settings={"kind": "evil_combo", "variant": "taxi_self_loop", "routing": "taxi"},
        )
    )

    graph = DaguaGraph()
    _apply_graph_style(graph)
    graph.add_node("hub", label="Hub")
    arrows = [
        "normal",
        "vee",
        "dot",
        "diamond",
        "tee",
        "crow",
        "curve",
        "bracket",
        "inv",
        "box",
        "simple",
        "fancy",
    ]
    for index, arrow in enumerate(arrows):
        graph.add_node(f"s{index}", label=arrow[:4])
        graph.add_edge(f"s{index}", "hub")
    spoke_positions = [
        [
            180.0 * math.cos(2.0 * math.pi * index / 12.0),
            180.0 * math.sin(2.0 * math.pi * index / 12.0),
        ]
        for index in range(12)
    ]
    positions = torch.tensor([[0.0, 0.0]] + spoke_positions, dtype=torch.float32)
    _set_all_node_styles(graph, _base_node_style(min_width=50.0, font_size=8.0))
    graph.edge_styles = [
        _base_edge_style(
            arrow=arrow,
            color_gradient="source_to_target",
            color_gradient_end="#D55E00",
            width=2.0,
            arrow_length=12.0,
            arrow_width=8.0,
        )
        for arrow in arrows
    ]
    cases.append(
        AlbumCase(
            case_id="evil_all_arrows_gradient",
            category="evil_combos",
            filename="evil_all_arrows_gradient_dagua.png",
            title="Evil Combo: All Arrows Gradient - dagua",
            graph=graph,
            positions=positions,
            settings={
                "kind": "evil_combo",
                "variant": "all_arrows_gradient",
                "arrow_count": 12,
            },
        )
    )

    graph, positions = _single_node_graph("Ghost Gradient")
    _set_all_node_styles(
        graph,
        _base_node_style(
            fill=WHITE,
            gradient="linear",
            gradient_color=WHITE,
            stroke="#EEEEEE",
            font_color="#DDDDDD",
            min_width=140.0,
        ),
    )
    cases.append(
        AlbumCase(
            case_id="evil_white_gradient",
            category="evil_combos",
            filename="evil_white_gradient_dagua.png",
            title="Evil Combo: White Gradient - dagua",
            graph=graph,
            positions=positions,
            settings={
                "kind": "evil_combo",
                "variant": "white_gradient",
                "fill": WHITE,
                "gradient_color": WHITE,
            },
        )
    )

    graph, positions = _crossing_edges_graph()
    _set_all_node_styles(graph, _base_node_style(fill="#F4F7FA"))
    _set_all_edge_styles(
        graph,
        _base_edge_style(taper=True, crossing_style="arc", width=6.0),
    )
    for edge_style in graph.edge_styles:
        if edge_style is not None:
            edge_style.taper_width_start = 10.0
            edge_style.taper_width_end = 0.3
    cases.append(
        AlbumCase(
            case_id="evil_extreme_taper_crossing",
            category="evil_combos",
            filename="evil_extreme_taper_crossing_dagua.png",
            title="Evil Combo: Extreme Taper Crossing - dagua",
            graph=graph,
            positions=positions,
            settings={
                "kind": "evil_combo",
                "variant": "extreme_taper_crossing",
                "taper_width_start": 10,
                "crossing_style": "arc",
            },
        )
    )

    graph = DaguaGraph()
    _apply_graph_style(graph)
    specs = [
        (
            "n0",
            "Tiny",
            _base_node_style(
                shape="circle",
                min_width=30.0,
                min_height=30.0,
                font_size=6.0,
                fill="#FF6B6B",
            ),
        ),
        (
            "n1",
            "HUGE",
            _base_node_style(
                shape="rect",
                min_width=200.0,
                min_height=120.0,
                font_size=20.0,
                fill="#4ECDC4",
                font_weight="bold",
            ),
        ),
        (
            "n2",
            "Star",
            _base_node_style(
                shape="star",
                min_width=140.0,
                min_height=140.0,
                fill="#FFE66D",
                shadow=True,
                gradient="radial",
                gradient_color="#FFA07A",
            ),
        ),
        (
            "n3",
            "",
            _base_node_style(
                shape="diamond",
                min_width=80.0,
                min_height=80.0,
                fill="#95E1D3",
                border_count=2,
                opacity=0.6,
            ),
        ),
        (
            "n4",
            "Cylinder\nDouble",
            _base_node_style(
                shape="cylinder",
                min_width=90.0,
                min_height=70.0,
                fill="#F8B4B4",
                stroke_dash="dashed",
            ),
        ),
    ]
    for node_id, label, _style in specs:
        graph.add_node(node_id, label=label)
    graph.add_edge("n0", "n1")
    graph.add_edge("n1", "n2")
    graph.add_edge("n2", "n3")
    graph.add_edge("n3", "n4")
    graph.add_edge("n4", "n0")
    positions = torch.tensor(
        [[-120.0, 80.0], [120.0, 80.0], [180.0, -40.0], [0.0, -120.0], [-180.0, -40.0]],
        dtype=torch.float32,
    )
    graph.node_styles = [style for _, _, style in specs]
    _set_all_edge_styles(graph, _base_edge_style(width=2.0))
    cases.append(
        AlbumCase(
            case_id="evil_contradictory_styles",
            category="evil_combos",
            filename="evil_contradictory_styles_dagua.png",
            title="Evil Combo: Contradictory Styles - dagua",
            graph=graph,
            positions=positions,
            settings={
                "kind": "evil_combo",
                "variant": "contradictory_styles",
                "node_count": 5,
            },
        )
    )

    return cases


def build_case_catalog() -> List[AlbumCase]:
    """Build the full album case catalog.

    Returns
    -------
    list[AlbumCase]
        All album cases in output order.
    """

    cases: List[AlbumCase] = []
    cases.extend(_node_shape_cases())
    cases.extend(_arrow_type_cases())
    cases.extend(_border_style_cases())
    cases.extend(_edge_style_cases())
    cases.extend(_edge_routing_cases())
    cases.extend(_text_formatting_cases())
    cases.extend(_opacity_cases())
    cases.extend(_shadow_cases())
    cases.extend(_gradient_cases())
    cases.extend(_cluster_cases())
    cases.extend(_direction_cases())
    cases.extend(_corner_radius_cases())
    cases.extend(_rich_label_cases())
    cases.extend(_combo_2way_cases())
    cases.extend(_combo_3way_cases())
    cases.extend(_combo_4way_cases())
    cases.extend(_combo_5way_cases())
    cases.extend(_evil_combo_cases())
    return cases


def _normalize_filter_values(values: Optional[Sequence[str]]) -> Optional[List[str]]:
    """Normalize CLI filter values, including comma-delimited arguments.

    Parameters
    ----------
    values : sequence[str] | None
        Raw CLI or API filter values.

    Returns
    -------
    list[str] | None
        Flattened filter values with comma-delimited entries split out.
    """

    if values is None:
        return None

    normalized: List[str] = []
    for value in values:
        normalized.extend(part.strip() for part in value.split(",") if part.strip())
    return normalized


def _select_cases(
    cases: Sequence[AlbumCase],
    categories: Optional[Sequence[str]] = None,
    case_ids: Optional[Sequence[str]] = None,
) -> List[AlbumCase]:
    """Filter the catalog by category and/or case identifier.

    Parameters
    ----------
    cases : sequence[AlbumCase]
        Full case catalog.
    categories : sequence[str] | None, default=None
        Optional category filter.
    case_ids : sequence[str] | None, default=None
        Optional case-id filter.

    Returns
    -------
    list[AlbumCase]
        Selected cases in catalog order.
    """

    selected = list(cases)
    if categories is not None:
        category_set = set(categories)
        selected = [case for case in selected if case.category in category_set]
    if case_ids is not None:
        case_id_set = set(case_ids)
        selected = [case for case in selected if case.case_id in case_id_set]
    return selected


def build_cosmetic_album(
    output_dir: str = "eval_output/cosmetic_album",
    categories: Optional[Sequence[str]] = None,
    case_ids: Optional[Sequence[str]] = None,
    dagua_only: bool = False,
    cache_competitor: bool = False,
) -> CosmeticAlbumResult:
    """Render the cosmetic comparison album and manifest.

    Parameters
    ----------
    output_dir : str, default="eval_output/cosmetic_album"
        Album root directory.
    categories : sequence[str] | None, default=None
        Optional subset of categories to render.
    case_ids : sequence[str] | None, default=None
        Optional subset of case identifiers to render.
    dagua_only : bool, default=False
        Reuse cached competitor renders instead of invoking Graphviz again.
    cache_competitor : bool, default=False
        Persist competitor PNGs under ``output_dir/_cache/competitor``.

    Returns
    -------
    CosmeticAlbumResult
        Output paths for the generated album.
    """

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    catalog = build_case_catalog()
    normalized_categories = _normalize_filter_values(categories)
    normalized_case_ids = _normalize_filter_values(case_ids)
    selected_cases = _select_cases(
        catalog,
        categories=normalized_categories,
        case_ids=normalized_case_ids,
    )
    if not selected_cases:
        raise ValueError("No cosmetic album cases matched the requested filters.")

    needs_graphviz_cases = any(case.graphviz is not None for case in selected_cases)
    if dagua_only and needs_graphviz_cases and not cache_competitor:
        raise ValueError("--dagua-only requires --cache-competitor for comparison cases.")

    if needs_graphviz_cases and not dagua_only and not _graphviz_available():
        raise RuntimeError("Graphviz `dot` is required for comparison cases but is not installed.")

    for category in sorted({case.category for case in selected_cases}):
        (root / category).mkdir(parents=True, exist_ok=True)
    if cache_competitor:
        (root / "_cache" / "competitor").mkdir(parents=True, exist_ok=True)

    image_paths: List[str] = []
    manifest_cases: List[Dict[str, object]] = []
    category_counts: Dict[str, int] = {}

    with tempfile.TemporaryDirectory(prefix="dagua_cosmetic_album_") as temp_dir:
        temp_root = Path(temp_dir)
        for case in selected_cases:
            category_counts[case.category] = category_counts.get(case.category, 0) + 1
            output_path = _case_output_path(root, case)
            render_error = _render_album_case(
                case=case,
                root=root,
                temp_root=temp_root,
                output_path=output_path,
                dagua_only=dagua_only,
                cache_competitor=cache_competitor,
            )
            image_paths.append(str(output_path))
            manifest_cases.append(_manifest_entry(case, output_path, render_error=render_error))

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(root),
        "total_images": len(image_paths),
        "category_counts": category_counts,
        "cases": manifest_cases,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(f"{json.dumps(manifest, indent=2)}\n", encoding="utf-8")

    return CosmeticAlbumResult(
        output_dir=str(root),
        manifest_path=str(manifest_path),
        image_paths=image_paths,
    )


def main() -> int:
    """Parse CLI arguments and generate the cosmetic album.

    Returns
    -------
    int
        Process exit code.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="eval_output/cosmetic_album")
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="Optional subset of category directory names.",
    )
    parser.add_argument(
        "--case-ids",
        nargs="*",
        default=None,
        help="Optional subset of stable case identifiers.",
    )
    parser.add_argument(
        "--dagua-only",
        action="store_true",
        help="Reuse cached competitor renders and regenerate only the Dagua side.",
    )
    parser.add_argument(
        "--cache-competitor",
        action="store_true",
        help="Persist Graphviz comparison panels under the output directory cache.",
    )
    args = parser.parse_args()

    result = build_cosmetic_album(
        output_dir=args.output_dir,
        categories=args.categories,
        case_ids=args.case_ids,
        dagua_only=args.dagua_only,
        cache_competitor=args.cache_competitor,
    )
    print(result.output_dir)
    print(result.manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
