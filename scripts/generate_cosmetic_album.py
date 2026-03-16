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
import shutil
import subprocess
import sys
import tempfile
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
GRAPHVIZ_PAIR_VERTICAL_GAP = 110.0
PANEL_CONTENT_MARGIN = 36
CONTENT_CROP_PADDING = 12


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
        stroke_width=float(GRAPHVIZ_MATCH_DEFAULTS["stroke_width"]),
        stroke_dash="solid",
        corner_radius=10.0,
        padding=24.0,
        font_size=11.0,
        font_weight="bold",
        font_color="#374151",
        opacity=0.5,
    )
    for field_name, value in overrides.items():
        setattr(style, field_name, value)
    return style


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
        positions = [(0.0, 0.0), (100.0, 0.0), (200.0, 0.0)]
    elif direction == "RL":
        positions = [(200.0, 0.0), (100.0, 0.0), (0.0, 0.0)]
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
        [[-60.0, 0.0], [60.0, 0.0], [0.0, 90.0]],
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


def _manifest_entry(case: AlbumCase, output_path: Path) -> Dict[str, object]:
    """Build the manifest row for one case.

    Parameters
    ----------
    case : AlbumCase
        Album case that was rendered.
    output_path : Path
        Final output path.

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
    }
    if case.graphviz is not None:
        entry["graphviz_engine"] = case.graphviz.engine
        entry["graphviz_dot"] = _build_graphviz_dot(case.graph, case.graphviz)
    return entry


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
        opacity=0.1,
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
    return cases


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
    selected_cases = _select_cases(catalog, categories=categories, case_ids=case_ids)
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
            dagua_raw = temp_root / f"{case.case_id}_dagua.png"
            _render_dagua_png(case.graph, case.positions, dagua_raw)

            output_path = _case_output_path(root, case)
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

            image_paths.append(str(output_path))
            manifest_cases.append(_manifest_entry(case, output_path))

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
