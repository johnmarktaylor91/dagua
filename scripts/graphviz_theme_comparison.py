#!/usr/bin/env python
# ruff: noqa: E402
"""Render Graphviz-theme comparisons across native Graphviz and Dagua.

This script builds a gallery with up to three columns per graph:
1. Native Graphviz ``dot`` using Graphviz defaults as the cosmetic reference.
2. Dagua rendered with the ``graphviz_strict`` theme.
3. Dagua rendered with the ``graphviz`` improved theme.
"""

from __future__ import annotations

import argparse
import copy
import html
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import dagua
from dagua import DaguaGraph
from dagua.graphs import list_graphs, load
from dagua.graphviz_utils import layout_with_graphviz
from dagua.styles import ClusterStyle, EdgeStyle, NodeStyle, get_theme

WHITE = "#FFFFFF"
WARM_WHITE = "#FAFAFA"
THREE_WAY_DIR = "three_way"
STRICT_DIR = "dagua_strict"
IMPROVED_DIR = "dagua_improved"
GRAPHVIZ_DIR = "graphviz_native"
INDEX_NAME = "index.html"
STRICT_THEME_NAME = "graphviz_strict"
IMPROVED_THEME_NAME = "graphviz"
PANEL_SIZE: Tuple[int, int] = (900, 700)
HEADER_HEIGHT = 40
TITLE_HEIGHT = 54
COMPOSE_DPI = 200
CONTENT_MARGIN = 20
TEXT_COLOR = "#111111"
LINE_COLOR = "#D7D7D7"
SUPPORTED_SHAPES: Tuple[str, ...] = (
    "ellipse",
    "rect",
    "roundrect",
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
)
ARROW_TYPES: Tuple[str, ...] = (
    "normal",
    "vee",
    "dot",
    "diamond",
    "tee",
    "crow",
    "circle",
    "open",
    "none",
)
DOT_ALLOWED_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.:+"
CLUSTER_NAME_SANITIZER = re.compile(r"[^0-9A-Za-z_]")
SLUG_SANITIZER = re.compile(r"[^0-9A-Za-z]+")


@dataclass(frozen=True)
class GraphCase:
    """Graph to render through the comparison pipeline.

    Parameters
    ----------
    slug : str
        Stable filesystem-friendly identifier.
    title : str
        Human-readable graph title.
    graph : DaguaGraph
        Source graph definition.
    """

    slug: str
    title: str
    graph: DaguaGraph


def _graphviz_node_attr_map(graph: DaguaGraph) -> Dict[int, Dict[str, str]]:
    """Return the per-node Graphviz override map stored on a graph.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph.

    Returns
    -------
    dict[int, dict[str, str]]
        Mutable node override map.
    """

    if not hasattr(graph, "_graphviz_node_attrs"):
        graph._graphviz_node_attrs = {}
    return graph._graphviz_node_attrs


def _graphviz_edge_attr_map(graph: DaguaGraph) -> Dict[int, Dict[str, str]]:
    """Return the per-edge Graphviz override map stored on a graph.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph.

    Returns
    -------
    dict[int, dict[str, str]]
        Mutable edge override map.
    """

    if not hasattr(graph, "_graphviz_edge_attrs"):
        graph._graphviz_edge_attrs = {}
    return graph._graphviz_edge_attrs


def _graphviz_cluster_attr_map(graph: DaguaGraph) -> Dict[str, Dict[str, str]]:
    """Return the per-cluster Graphviz override map stored on a graph.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph.

    Returns
    -------
    dict[str, dict[str, str]]
        Mutable cluster override map.
    """

    if not hasattr(graph, "_graphviz_cluster_attrs"):
        graph._graphviz_cluster_attrs = {}
    return graph._graphviz_cluster_attrs


def _set_graphviz_node_attrs(graph: DaguaGraph, node_index: int, attrs: Dict[str, str]) -> None:
    """Attach native-Graphviz node overrides to a graph node.

    Parameters
    ----------
    graph : DaguaGraph
        Target graph.
    node_index : int
        Node index.
    attrs : dict[str, str]
        Graphviz node attributes.

    Returns
    -------
    None
        The graph metadata is mutated in place.
    """

    _graphviz_node_attr_map(graph)[node_index] = dict(attrs)


def _set_graphviz_edge_attrs(graph: DaguaGraph, edge_index: int, attrs: Dict[str, str]) -> None:
    """Attach native-Graphviz edge overrides to a graph edge.

    Parameters
    ----------
    graph : DaguaGraph
        Target graph.
    edge_index : int
        Edge index.
    attrs : dict[str, str]
        Graphviz edge attributes.

    Returns
    -------
    None
        The graph metadata is mutated in place.
    """

    _graphviz_edge_attr_map(graph)[edge_index] = dict(attrs)


def _set_graphviz_cluster_attrs(graph: DaguaGraph, name: str, attrs: Dict[str, str]) -> None:
    """Attach native-Graphviz cluster overrides to a graph cluster.

    Parameters
    ----------
    graph : DaguaGraph
        Target graph.
    name : str
        Cluster name.
    attrs : dict[str, str]
        Graphviz cluster attributes.

    Returns
    -------
    None
        The graph metadata is mutated in place.
    """

    _graphviz_cluster_attr_map(graph)[name] = dict(attrs)


def _add_edge_with_graphviz_attrs(
    graph: DaguaGraph,
    source: Any,
    target: Any,
    label: Optional[str] = None,
    type: str = "normal",
    style: Optional[EdgeStyle] = None,
    graphviz_attrs: Optional[Dict[str, str]] = None,
) -> int:
    """Add an edge and optionally register native Graphviz override attrs.

    Parameters
    ----------
    graph : DaguaGraph
        Target graph.
    source : Any
        Source node identifier.
    target : Any
        Target node identifier.
    label : str, optional
        Edge label.
    type : str, default="normal"
        Edge type.
    style : EdgeStyle, optional
        Dagua edge style override.
    graphviz_attrs : dict[str, str], optional
        Native Graphviz edge attributes.

    Returns
    -------
    int
        Edge index.
    """

    edge_index = len(graph.edge_labels)
    graph.add_edge(source, target, label=label, type=type, style=style)
    if graphviz_attrs:
        _set_graphviz_edge_attrs(graph, edge_index, graphviz_attrs)
    return edge_index


def _slugify(value: str) -> str:
    """Convert a title into a stable lowercase filesystem slug.

    Parameters
    ----------
    value : str
        Source title.

    Returns
    -------
    str
        Slugified string.
    """

    normalized = SLUG_SANITIZER.sub("_", value.strip().lower()).strip("_")
    return normalized or "graph"


def _safe_cluster_id(name: str) -> str:
    """Return a DOT-safe cluster identifier.

    Parameters
    ----------
    name : str
        Cluster name.

    Returns
    -------
    str
        DOT-safe cluster identifier with Graphviz's ``cluster_`` prefix.
    """

    return f"cluster_{CLUSTER_NAME_SANITIZER.sub('_', name)}"


def _make_case(title: str, graph: DaguaGraph, slug: Optional[str] = None) -> GraphCase:
    """Wrap a graph and title into a render case.

    Parameters
    ----------
    title : str
        Human-readable graph title.
    graph : DaguaGraph
        Graph definition.
    slug : str, optional
        Explicit slug override.

    Returns
    -------
    GraphCase
        Render case metadata.
    """

    return GraphCase(slug=slug or _slugify(title), title=title, graph=graph)


def _make_node_shapes_showcase() -> Tuple[DaguaGraph, str]:
    """Create a vertical chain covering every supported node shape.

    Returns
    -------
    tuple[DaguaGraph, str]
        Graph and display title.
    """

    graph = DaguaGraph(direction="TB")
    previous_id: Optional[str] = None
    for shape in SUPPORTED_SHAPES:
        node_id = f"shape_{shape}"
        node_index = graph.add_node(node_id, label=shape, style=NodeStyle(shape=shape))
        _set_graphviz_node_attrs(graph, node_index, _graphviz_shape_attrs(NodeStyle(shape=shape)))
        if previous_id is not None:
            graph.add_edge(previous_id, node_id)
        previous_id = node_id
    return graph, "Node Shapes Showcase"


def _make_label_variety() -> Tuple[DaguaGraph, str]:
    """Create a DAG spanning short, long, and multiline node labels.

    Returns
    -------
    tuple[DaguaGraph, str]
        Graph and display title.
    """

    graph = DaguaGraph(direction="TB")
    labels = [
        ("a", "A", "input"),
        ("go", "Go", "default"),
        ("relu", "relu", "default"),
        ("dense", "dense_layer", "default"),
        ("fifteen", "attention_block", "default"),
        ("multi", "batch\nnorm", "default"),
        ("wide", "residual\nconnection\nskip", "default"),
        ("long", "conv2d_batch_norm_relu_dropout_3", "output"),
    ]
    for node_id, label, node_type in labels:
        graph.add_node(node_id, label=label, type=node_type)

    edges = [
        ("a", "go"),
        ("a", "relu"),
        ("go", "dense"),
        ("relu", "dense"),
        ("dense", "fifteen"),
        ("dense", "multi"),
        ("fifteen", "wide"),
        ("multi", "wide"),
        ("wide", "long"),
    ]
    for source, target in edges:
        graph.add_edge(source, target)
    return graph, "Label Variety"


def _make_edge_styles_showcase() -> Tuple[DaguaGraph, str]:
    """Create edge pairs that vary style, width, and label treatment.

    Returns
    -------
    tuple[DaguaGraph, str]
        Graph and display title.
    """

    graph = DaguaGraph(direction="TB")
    specs: List[Tuple[str, float, Optional[str], str]] = [
        ("solid", 1.0, None, "solid"),
        ("solid", 2.0, "thick solid", "solid"),
        ("dashed", 1.0, None, "dashed"),
        ("dashed", 0.5, "thin dashed", "dashed"),
        ("dotted", 1.0, "dotted link", "dotted"),
        ("dotted", 2.0, None, "dotted"),
    ]
    for index, (line_style, width, label, edge_type) in enumerate(specs):
        left = f"left_{index}"
        right = f"right_{index}"
        graph.add_node(left, label=f"{line_style} {width:g}x")
        graph.add_node(right, label=f"target {index + 1}")
        edge_style = EdgeStyle(style=line_style, width=width, label_font_size=10.0)
        graphviz_attrs = {
            "style": line_style,
            "penwidth": f"{width:.2f}",
        }
        _add_edge_with_graphviz_attrs(
            graph,
            left,
            right,
            label=label,
            type=edge_type,
            style=edge_style,
            graphviz_attrs=graphviz_attrs,
        )
        if index > 0:
            _add_edge_with_graphviz_attrs(
                graph,
                f"left_{index - 1}",
                left,
                style=EdgeStyle(style="solid", width=0.6),
                graphviz_attrs={"penwidth": "0.60"},
            )
    return graph, "Edge Styles Showcase"


def _make_arrow_types() -> Tuple[DaguaGraph, str]:
    """Create one source-target pair per supported arrowhead type.

    Returns
    -------
    tuple[DaguaGraph, str]
        Graph and display title.
    """

    graph = DaguaGraph(direction="TB")
    for arrow in ARROW_TYPES:
        source = f"{arrow}_src"
        target = f"{arrow}_dst"
        graph.add_node(source, label=arrow)
        graph.add_node(target, label="target")
        edge_style = EdgeStyle(
            arrow=arrow,
            arrow_fill="hollow" if arrow in {"circle", "open"} else "filled",
            width=1.2,
            label_font_size=10.0,
        )
        _add_edge_with_graphviz_attrs(
            graph,
            source,
            target,
            label=arrow,
            style=edge_style,
            graphviz_attrs={
                "arrowhead": _graphviz_arrowhead(edge_style),
                "penwidth": "1.20",
            },
        )
    return graph, "Arrow Types"


def _make_fan_pattern() -> Tuple[DaguaGraph, str]:
    """Create high-degree fan-out and fan-in hub patterns.

    Returns
    -------
    tuple[DaguaGraph, str]
        Graph and display title.
    """

    graph = DaguaGraph(direction="TB")
    graph.add_node("fan_out", label="fan-out", type="input")
    graph.add_node("fan_in", label="fan-in", type="output")
    for index in range(8):
        out_leaf = f"out_{index}"
        in_leaf = f"in_{index}"
        graph.add_node(out_leaf, label=f"out {index + 1}")
        graph.add_node(in_leaf, label=f"in {index + 1}")
        graph.add_edge("fan_out", out_leaf, style=EdgeStyle(width=1.0 + (index % 3) * 0.35))
        graph.add_edge(in_leaf, "fan_in", style=EdgeStyle(width=1.0 + (index % 2) * 0.4))
        if index < 7:
            graph.add_edge(out_leaf, in_leaf, style=EdgeStyle(style="dotted", opacity=0.45))
    return graph, "Fan Pattern"


def _make_cluster_showcase() -> Tuple[DaguaGraph, str]:
    """Create sibling and nested clusters with cross-cluster connectivity.

    Returns
    -------
    tuple[DaguaGraph, str]
        Graph and display title.
    """

    graph = DaguaGraph(direction="TB")
    small_nodes = [f"small_{index}" for index in range(2)]
    medium_nodes = [f"medium_{index}" for index in range(4)]
    large_nodes = [f"large_{index}" for index in range(6)]
    outer_nodes = ["outer_a", "outer_b"]
    inner_nodes = ["inner_a", "inner_b", "inner_c"]
    for node_id in small_nodes + medium_nodes + large_nodes + outer_nodes + inner_nodes:
        graph.add_node(node_id, label=node_id.replace("_", " "))

    for members, prefix in (
        (small_nodes, "small"),
        (medium_nodes, "medium"),
        (large_nodes, "large"),
        (outer_nodes + inner_nodes, "outer"),
        (inner_nodes, "inner"),
    ):
        for index in range(len(members) - 1):
            graph.add_edge(
                members[index],
                members[index + 1],
                style=EdgeStyle(width=1.0 if prefix != "inner" else 0.9),
            )

    graph.add_cluster("cluster_small", small_nodes, label="Tiny Cluster")
    graph.add_cluster("cluster_medium", medium_nodes, label="Medium Cluster")
    graph.add_cluster(
        "cluster_large",
        large_nodes,
        label="Large Cluster With Longer Label",
    )
    graph.add_cluster("cluster_inner", inner_nodes, label="Nested Inner", parent="cluster_outer")
    graph.add_cluster(
        "cluster_outer",
        outer_nodes + inner_nodes,
        label="Outer Cluster",
    )

    cross_edges = [
        (small_nodes[0], medium_nodes[0]),
        (small_nodes[1], medium_nodes[-1]),
        (medium_nodes[1], large_nodes[0]),
        (medium_nodes[-1], large_nodes[-1]),
        (outer_nodes[0], medium_nodes[2]),
        (inner_nodes[-1], large_nodes[2]),
        (large_nodes[3], outer_nodes[1]),
    ]
    for source, target in cross_edges:
        graph.add_edge(source, target, style=EdgeStyle(width=1.0, opacity=0.8))

    return graph, "Cluster Showcase"


def _make_mixed_styles() -> Tuple[DaguaGraph, str]:
    """Create a graph mixing node and edge-level cosmetic overrides.

    Returns
    -------
    tuple[DaguaGraph, str]
        Graph and display title.
    """

    graph = DaguaGraph(direction="TB")
    graph.add_node(
        "start",
        label="Start",
        type="input",
        style=NodeStyle(fill="#E6F4EA", stroke="#2D6A4F", font_weight="bold"),
    )
    graph.add_node(
        "process",
        label="Process",
        style=NodeStyle(font_size=14.0, fill="#F7F7F7", stroke="#444444"),
    )
    graph.add_node(
        "review",
        label="Review",
        style=NodeStyle(fill="#FFF7E6", stroke="#B7791F", font_weight="bold"),
    )
    graph.add_node(
        "fallback",
        label="Fallback",
        style=NodeStyle(font_size=10.0, fill="#F3F0FF", stroke="#6B46C1"),
    )
    graph.add_node(
        "done",
        label="Done",
        type="output",
        style=NodeStyle(fill="#FFF1F2", stroke="#B42318", font_size=11.0),
    )
    graph.add_edge("start", "process", style=EdgeStyle(style="solid", width=1.4))
    graph.add_edge(
        "process",
        "review",
        label="needs sign-off",
        style=EdgeStyle(style="dashed", width=1.1, color="#A16207", label_font_size=9.0),
    )
    graph.add_edge(
        "review",
        "done",
        label="approved",
        style=EdgeStyle(style="solid", width=1.2, color="#047857", label_font_size=9.0),
    )
    graph.add_edge(
        "review",
        "fallback",
        label="rework",
        style=EdgeStyle(style="dotted", width=1.0, color="#6B7280", label_font_size=9.0),
    )
    graph.add_edge(
        "fallback",
        "process",
        label="retry",
        type="if",
        style=EdgeStyle(style="dashed", width=1.0, color="#9A3412", label_font_size=8.5),
    )
    return graph, "Mixed Styles"


def _make_tiny_graph() -> Tuple[DaguaGraph, str]:
    """Create the smallest non-trivial three-node DAG.

    Returns
    -------
    tuple[DaguaGraph, str]
        Graph and display title.
    """

    graph = DaguaGraph(direction="TB")
    graph.add_node("in", label="In", type="input")
    graph.add_node("mid", label="Mid")
    graph.add_node("out", label="Out", type="output")
    graph.add_edge("in", "mid")
    graph.add_edge("mid", "out")
    return graph, "Tiny Graph"


def _make_dense_graph() -> Tuple[DaguaGraph, str]:
    """Create a moderately dense synthetic DAG with 40 nodes and ~80 edges.

    Returns
    -------
    tuple[DaguaGraph, str]
        Graph and display title.
    """

    graph = DaguaGraph(direction="TB")
    for index in range(40):
        node_type = "default"
        if index < 3:
            node_type = "input"
        elif index >= 37:
            node_type = "output"
        graph.add_node(f"n{index}", label=f"n{index}", type=node_type)

    for index in range(39):
        graph.add_edge(f"n{index}", f"n{index + 1}")

    for source in range(40):
        for step in (3, 5):
            target = source + step
            if target < 40:
                edge_type = "buffer" if step == 5 else "default"
                graph.add_edge(
                    f"n{source}",
                    f"n{target}",
                    type=edge_type,
                    style=EdgeStyle(
                        width=1.0 if step == 3 else 0.8,
                        style="solid" if step == 3 else "dotted",
                        opacity=0.75 if step == 3 else 0.55,
                    ),
                )

    extra_edges = [
        (0, 10),
        (2, 14),
        (6, 18),
        (8, 24),
        (11, 29),
        (15, 33),
        (19, 36),
        (21, 39),
    ]
    for source, target in extra_edges:
        graph.add_edge(
            f"n{source}",
            f"n{target}",
            style=EdgeStyle(style="dashed", width=1.0, opacity=0.7),
        )

    return graph, "Dense Graph"


def _make_colors_showcase() -> Tuple[DaguaGraph, str]:
    """Create nodes with explicit fill colors to test theme override behavior.

    Returns
    -------
    tuple[DaguaGraph, str]
        Graph and display title.
    """

    graph = DaguaGraph(direction="TB")
    colors = [
        ("red", "#F87171", "#991B1B"),
        ("blue", "#60A5FA", "#1D4ED8"),
        ("green", "#4ADE80", "#166534"),
        ("yellow", "#FACC15", "#854D0E"),
        ("purple", "#C084FC", "#6B21A8"),
        ("orange", "#FB923C", "#9A3412"),
    ]
    previous_id: Optional[str] = None
    for name, fill, stroke in colors:
        node_id = f"color_{name}"
        node_index = graph.add_node(
            node_id,
            label=name.title(),
            style=NodeStyle(fill=fill, stroke=stroke),
        )
        _set_graphviz_node_attrs(
            graph,
            node_index,
            {"style": "filled", "fillcolor": fill, "color": stroke},
        )
        if previous_id is not None:
            graph.add_edge(previous_id, node_id)
        previous_id = node_id
    return graph, "Colors Showcase"


def _custom_graph_cases() -> List[GraphCase]:
    """Create the full set of programmatic showcase cases.

    Returns
    -------
    list[GraphCase]
        Render cases for the custom showcase graphs.
    """

    generators: Sequence[Any] = (
        _make_node_shapes_showcase,
        _make_label_variety,
        _make_edge_styles_showcase,
        _make_arrow_types,
        _make_fan_pattern,
        _make_cluster_showcase,
        _make_mixed_styles,
        _make_tiny_graph,
        _make_dense_graph,
        _make_colors_showcase,
    )
    cases: List[GraphCase] = []
    for generator in generators:
        graph, title = generator()
        cases.append(_make_case(title, graph))
    return cases


def _bundled_graph_cases(max_nodes: int = 200) -> List[GraphCase]:
    """Load bundled YAML graphs, skipping large cases for iteration speed.

    Parameters
    ----------
    max_nodes : int, default=200
        Maximum node count to include.

    Returns
    -------
    list[GraphCase]
        Loaded bundled graph cases under the size limit.
    """

    cases: List[GraphCase] = []
    for graph_name in list_graphs():
        graph = load(graph_name)
        if graph.num_nodes > max_nodes:
            continue
        cases.append(_make_case(graph_name, graph, slug=graph_name))
    return cases


def _graphviz_available() -> bool:
    """Return whether native Graphviz ``dot`` is available.

    Returns
    -------
    bool
        ``True`` when ``dot`` is on ``PATH``.
    """

    return shutil.which("dot") is not None


def _escape_dot_string(value: str) -> str:
    """Escape a string for DOT quoted-string syntax.

    Parameters
    ----------
    value : str
        Raw string value.

    Returns
    -------
    str
        Escaped string safe for use inside DOT quotes.
    """

    return value.replace("\\", "\\\\").replace('"', '\\"')


def _format_dot_value(value: str) -> str:
    """Format a DOT value with conservative quoting.

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
    if all(char in DOT_ALLOWED_CHARS for char in stripped):
        return stripped
    return f'"{_escape_dot_string(stripped)}"'


def _format_dot_attrs(attrs: Mapping[str, str]) -> str:
    """Render a DOT attribute list.

    Parameters
    ----------
    attrs : Mapping[str, str]
        Attribute mapping.

    Returns
    -------
    str
        ``[key=value, ...]`` block or an empty string.
    """

    if not attrs:
        return ""
    parts = [f"{key}={_format_dot_value(value)}" for key, value in attrs.items()]
    return f" [{', '.join(parts)}]"


def _cluster_children(graph: DaguaGraph) -> Dict[Optional[str], List[str]]:
    """Build the cluster hierarchy keyed by parent cluster name.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph.

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
    """Return flattened member indices for a cluster.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph.
    name : str
        Cluster name.

    Returns
    -------
    list[int]
        Member node indices.
    """

    members = graph.clusters.get(name, [])
    if isinstance(members, dict):
        from dagua.utils import collect_cluster_leaves

        return [int(index) for index in collect_cluster_leaves(members)]
    return [int(index) for index in members]


def _graphviz_shape_attrs(style: NodeStyle) -> Dict[str, str]:
    """Map a Dagua node shape into Graphviz node attributes.

    Parameters
    ----------
    style : NodeStyle
        Effective node style.

    Returns
    -------
    dict[str, str]
        Graphviz node shape attributes.
    """

    if style.shape == "rect":
        return {"shape": "box"}
    if style.shape == "roundrect":
        return {"shape": "box", "style": "rounded"}
    if style.shape == "trapezoid":
        return {"shape": "trapezium"}
    return {"shape": style.shape}


def _graphviz_arrowhead(style: EdgeStyle) -> str:
    """Map a Dagua arrow style into the closest Graphviz arrowhead token.

    Parameters
    ----------
    style : EdgeStyle
        Effective edge style.

    Returns
    -------
    str
        Graphviz arrowhead token.
    """

    if style.arrow == "normal":
        return "normal" if style.arrow_fill == "filled" else "empty"
    if style.arrow == "diamond":
        return "diamond" if style.arrow_fill == "filled" else "odiamond"
    if style.arrow == "dot":
        return "dot" if style.arrow_fill == "filled" else "odot"
    if style.arrow == "circle":
        return "odot"
    return style.arrow


def _graphviz_style_tokens(style: Optional[str], filled: bool, rounded: bool = False) -> str:
    """Compose Graphviz ``style=`` tokens.

    Parameters
    ----------
    style : str, optional
        Base line style token such as ``solid`` or ``dashed``.
    filled : bool
        Whether ``filled`` should be included.
    rounded : bool, default=False
        Whether ``rounded`` should be included.

    Returns
    -------
    str
        Comma-delimited Graphviz style string.
    """

    tokens: List[str] = []
    if filled:
        tokens.append("filled")
    if rounded:
        tokens.append("rounded")
    if style is not None and style not in {"solid", ""}:
        tokens.append(style)
    return ",".join(tokens)


def _node_override_attrs(graph: DaguaGraph, index: int) -> Dict[str, str]:
    """Return node-level DOT attributes needed beyond Graphviz defaults.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph.
    index : int
        Node index.

    Returns
    -------
    dict[str, str]
        Explicit node attribute overrides.
    """

    attrs: Dict[str, str] = {"label": graph.node_labels[index]}
    attrs.update(_graphviz_node_attr_map(graph).get(index, {}))

    style = graph.node_styles[index] if index < len(graph.node_styles) else None
    if style is None:
        return attrs

    default_style = NodeStyle()
    if style.shape != default_style.shape and "shape" not in attrs:
        attrs.update(_graphviz_shape_attrs(style))
    if style.fill != default_style.fill:
        attrs.setdefault("fillcolor", style.fill)
        attrs["style"] = _graphviz_style_tokens(style=style.stroke_dash, filled=True, rounded=False)
    if style.stroke != default_style.stroke:
        attrs.setdefault("color", style.stroke)
    if style.stroke_width != default_style.stroke_width:
        attrs.setdefault("penwidth", f"{style.stroke_width:.2f}")
    if style.stroke_dash != default_style.stroke_dash:
        existing_style = attrs.get("style")
        attrs["style"] = _graphviz_style_tokens(
            style=style.stroke_dash,
            filled="fillcolor" in attrs or bool(existing_style and "filled" in existing_style),
            rounded=style.shape == "roundrect",
        )
    if style.font_family != default_style.font_family:
        attrs.setdefault("fontname", style.font_family)
    if style.font_size != default_style.font_size:
        attrs.setdefault("fontsize", f"{style.font_size:.1f}")
    if style.font_color != default_style.font_color:
        attrs.setdefault("fontcolor", style.font_color)
    return attrs


def _edge_override_attrs(graph: DaguaGraph, index: int) -> Dict[str, str]:
    """Return edge-level DOT attributes needed beyond Graphviz defaults.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph.
    index : int
        Edge index.

    Returns
    -------
    dict[str, str]
        Explicit edge attribute overrides.
    """

    attrs: Dict[str, str] = {}
    attrs.update(_graphviz_edge_attr_map(graph).get(index, {}))
    if index < len(graph.edge_labels) and graph.edge_labels[index]:
        attrs["label"] = str(graph.edge_labels[index])
    style = graph.edge_styles[index] if index < len(graph.edge_styles) else None
    if style is None:
        return attrs

    default_style = EdgeStyle()
    if style.color != default_style.color:
        attrs["color"] = style.color
    if style.width != default_style.width:
        attrs["penwidth"] = f"{style.width:.2f}"
    if style.style != default_style.style:
        attrs["style"] = style.style
    if style.arrow != default_style.arrow or style.arrow_fill != default_style.arrow_fill:
        attrs["arrowhead"] = _graphviz_arrowhead(style)
    return attrs


def _cluster_override_attrs(style: ClusterStyle) -> Dict[str, str]:
    """Return cluster-level DOT attributes for explicit cluster styling.

    Parameters
    ----------
    style : ClusterStyle
        Effective cluster style.

    Returns
    -------
    dict[str, str]
        Cluster DOT attributes.
    """

    theme_style = get_theme(STRICT_THEME_NAME).cluster_style
    attrs: Dict[str, str] = {}
    if style.stroke != theme_style.stroke:
        attrs["color"] = style.stroke
    if style.stroke_width != theme_style.stroke_width:
        attrs["penwidth"] = f"{style.stroke_width:.2f}"
    if style.stroke_dash != theme_style.stroke_dash:
        attrs["style"] = style.stroke_dash
    if style.fill not in {"none", "", theme_style.fill}:
        attrs["style"] = _graphviz_style_tokens(style=style.stroke_dash, filled=True)
        attrs["fillcolor"] = style.fill
    if style.font_size != theme_style.font_size:
        attrs["fontsize"] = f"{style.font_size:.1f}"
    if style.font_color != theme_style.font_color:
        attrs["fontcolor"] = style.font_color
    if style.font_family:
        attrs["fontname"] = style.font_family
    return attrs


def _emit_cluster_block(
    lines: List[str],
    graph: DaguaGraph,
    name: str,
    children: Mapping[Optional[str], List[str]],
    emitted_nodes: Set[int],
    depth: int,
) -> None:
    """Append DOT lines for one cluster and its nested descendants.

    Parameters
    ----------
    lines : list[str]
        Mutable DOT buffer.
    graph : DaguaGraph
        Source graph.
    name : str
        Cluster name.
    children : Mapping[Optional[str], list[str]]
        Parent-to-children cluster mapping.
    emitted_nodes : set[int]
        Nodes already emitted into prior cluster scopes.
    depth : int
        Nesting depth used for indentation.

    Returns
    -------
    None
        Mutates ``lines`` in place.
    """

    indent = "  " * (depth + 1)
    label = graph.cluster_labels.get(name, name)
    cluster_style = graph.get_style_for_cluster(name)
    attrs = {"label": label}
    attrs.update(_cluster_override_attrs(cluster_style))
    attrs.update(_graphviz_cluster_attr_map(graph).get(name, {}))

    lines.append(f"{indent}subgraph {_safe_cluster_id(name)} {{")
    for key, value in attrs.items():
        lines.append(f"{indent}  {key}={_format_dot_value(value)};")

    child_names = children.get(name, [])
    for child in child_names:
        _emit_cluster_block(lines, graph, child, children, emitted_nodes, depth + 1)

    nested_members: Set[int] = set()
    for child in child_names:
        nested_members.update(_cluster_members(graph, child))

    for node_index in _cluster_members(graph, name):
        if node_index in nested_members or node_index in emitted_nodes:
            continue
        node_attrs = _format_dot_attrs(_node_override_attrs(graph, node_index))
        lines.append(f"{indent}  n{node_index}{node_attrs};")
        emitted_nodes.add(node_index)

    lines.append(f"{indent}}}")


def graph_to_dot(graph: DaguaGraph) -> str:
    """Convert a graph into DOT using Graphviz defaults plus explicit overrides.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph.

    Returns
    -------
    str
        DOT source string.
    """

    lines = ["digraph G {", "  rankdir=TB;"]
    emitted_nodes: Set[int] = set()

    if graph.clusters:
        children = _cluster_children(graph)
        for root_cluster in children.get(None, []):
            _emit_cluster_block(lines, graph, root_cluster, children, emitted_nodes, 0)

    for node_index in range(graph.num_nodes):
        if node_index in emitted_nodes:
            continue
        node_attrs = _format_dot_attrs(_node_override_attrs(graph, node_index))
        lines.append(f"  n{node_index}{node_attrs};")

    edge_count = int(graph.edge_index.shape[1])
    for edge_index in range(edge_count):
        source = int(graph.edge_index[0, edge_index].item())
        target = int(graph.edge_index[1, edge_index].item())
        attrs = _edge_override_attrs(graph, edge_index)
        lines.append(f"  n{source} -> n{target}{_format_dot_attrs(attrs)};")

    lines.append("}")
    return "\n".join(lines)


def render_graphviz_native(graph: DaguaGraph, output_path: Path) -> bool:
    """Render a graph through native Graphviz ``dot``.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph.
    output_path : Path
        Destination PNG path.

    Returns
    -------
    bool
        ``True`` when rendering succeeded.
    """

    dot_source = graph_to_dot(graph)
    with tempfile.NamedTemporaryFile(suffix=".dot", mode="w", delete=False) as handle:
        handle.write(dot_source)
        handle.flush()
        dot_path = Path(handle.name)

    try:
        result = subprocess.run(
            ["dot", "-Tpng", "-Gdpi=210", str(dot_path), "-o", str(output_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        dot_path.unlink(missing_ok=True)
        return False
    finally:
        dot_path.unlink(missing_ok=True)

    return result.returncode == 0


def _copy_graph_with_theme(graph: DaguaGraph, theme_name: str) -> DaguaGraph:
    """Clone a graph and apply a named theme to the clone.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph.
    theme_name : str
        Built-in theme name.

    Returns
    -------
    DaguaGraph
        Themed graph clone ready for layout and rendering.
    """

    themed_graph = copy.deepcopy(graph)
    dagua.set_theme(theme_name)
    themed_graph._theme = get_theme(theme_name)
    themed_graph.invalidate_layout()

    return themed_graph


def render_dagua_theme(graph: DaguaGraph, theme_name: str, output_path: Path) -> None:
    """Render a graph with a specific Dagua theme on Graphviz positions.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph.
    theme_name : str
        Built-in theme name.
    output_path : Path
        Destination PNG path.

    Returns
    -------
    None
        The render is written to ``output_path``.
    """

    themed_graph = _copy_graph_with_theme(graph, theme_name)
    themed_graph.compute_node_sizes()
    positions = layout_with_graphviz(themed_graph, engine="dot")
    # layout_with_graphviz negates y for dagua's y-down/TB convention.
    # Keep the negated positions so the edge router produces correct
    # arrowhead orientation for top-down flow. The render will be inverted
    # (source at bottom) but we fix this by inverting the y-axis after render.

    # Compute figsize so Graphviz's point-space positions render at the same
    # physical scale as native Graphviz (72 points = 1 inch).
    sizes_np = themed_graph.node_sizes.detach().cpu().numpy()
    margin = 18.0  # points
    x_min = float((positions[:, 0] - sizes_np[:, 0] / 2).min()) - margin
    x_max = float((positions[:, 0] + sizes_np[:, 0] / 2).max()) + margin
    y_min = float((positions[:, 1] - sizes_np[:, 1] / 2).min()) - margin
    y_max = float((positions[:, 1] + sizes_np[:, 1] / 2).max()) + margin
    # 72 points per inch (Graphviz convention)
    fig_w = (x_max - x_min) / 72.0
    fig_h = (y_max - y_min) / 72.0
    # Clamp to reasonable range
    fig_w = max(2.0, min(fig_w, 16.0))
    fig_h = max(2.0, min(fig_h, 16.0))
    fig, ax = dagua.render(
        themed_graph,
        positions,
        figsize=(fig_w, fig_h),
        dpi=210,
    )
    # Invert y-axis so the graph reads top-down matching Graphviz.
    # The positions use dagua's y-down convention (negated from Graphviz),
    # so inverting the display axis restores Graphviz's visual orientation
    # while keeping arrowheads pointing in the correct direction.
    ax.invert_yaxis()
    fig.savefig(str(output_path), dpi=210, bbox_inches="tight", facecolor=fig.get_facecolor())
    import matplotlib.pyplot as _plt

    _plt.close(fig)


def _content_crop_box(image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """Compute a content crop box by trimming near-white margins.

    Parameters
    ----------
    image : PIL.Image.Image
        Source image.

    Returns
    -------
    tuple[int, int, int, int] | None
        PIL crop bounds, or ``None`` when no content is detected.
    """

    data = np.asarray(image.convert("RGBA"))
    content_mask = (data[:, :, 3] > 0) & np.any(data[:, :, :3] < 250, axis=2)
    if not bool(content_mask.any()):
        return None
    ys, xs = np.nonzero(content_mask)
    left = max(int(xs.min()) - 12, 0)
    top = max(int(ys.min()) - 12, 0)
    right = min(int(xs.max()) + 13, image.width)
    bottom = min(int(ys.max()) + 13, image.height)
    return left, top, right, bottom


def _normalize_panel_image(image_path: Path, panel_size: Tuple[int, int]) -> Image.Image:
    """Resize and center an image onto a fixed white canvas.

    Parameters
    ----------
    image_path : Path
        Source render path.
    panel_size : tuple[int, int]
        Target panel size in pixels.

    Returns
    -------
    PIL.Image.Image
        Normalized RGB panel image.
    """

    with Image.open(image_path) as image:
        rgba = image.convert("RGBA")
        crop_box = _content_crop_box(rgba)
        if crop_box is not None:
            rgba = rgba.crop(crop_box)
        rgba.thumbnail(
            (panel_size[0] - CONTENT_MARGIN * 2, panel_size[1] - CONTENT_MARGIN * 2),
            Image.LANCZOS,
        )
        canvas = Image.new("RGBA", panel_size, WHITE)
        offset = ((panel_size[0] - rgba.width) // 2, (panel_size[1] - rgba.height) // 2)
        canvas.paste(rgba, offset, rgba)
    return canvas.convert("RGB")


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Load a simple local font, falling back to PIL's default font.

    Parameters
    ----------
    size : int
        Desired point size.
    bold : bool, default=False
        Whether to prefer a bold variant.

    Returns
    -------
    PIL.ImageFont.ImageFont
        Loaded font object.
    """

    font_candidates = ["DejaVuSans-Bold.ttf", "DejaVuSans.ttf"] if bold else ["DejaVuSans.ttf"]
    for candidate in font_candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    bounds: Tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    """Draw centered text inside a rectangular region.

    Parameters
    ----------
    draw : PIL.ImageDraw.ImageDraw
        Drawing context.
    bounds : tuple[int, int, int, int]
        Bounding rectangle as ``(left, top, right, bottom)``.
    text : str
        Text to draw.
    font : PIL.ImageFont.ImageFont
        Font object.
    fill : str
        Text color.

    Returns
    -------
    None
        Mutates ``draw`` in place.
    """

    text_box = draw.textbbox((0, 0), text, font=font)
    width = text_box[2] - text_box[0]
    height = text_box[3] - text_box[1]
    left, top, right, bottom = bounds
    x = left + ((right - left) - width) / 2
    y = top + ((bottom - top) - height) / 2
    draw.text((x, y), text, font=font, fill=fill)


def compose_three_way(
    strict_path: Path,
    improved_path: Path,
    title: str,
    output_path: Path,
    graphviz_path: Optional[Path] = None,
) -> None:
    """Compose a two- or three-panel comparison image with labeled headers.

    Parameters
    ----------
    strict_path : Path
        Dagua strict render path.
    improved_path : Path
        Dagua improved render path.
    title : str
        Graph title shown as the row header.
    output_path : Path
        Destination composite image path.
    graphviz_path : Path, optional
        Native Graphviz render path. When omitted, only the two Dagua columns
        are shown.

    Returns
    -------
    None
        The composite PNG is written to ``output_path``.
    """

    panels: List[Tuple[str, Image.Image]] = []
    if graphviz_path is not None and graphviz_path.exists():
        panels.append(("Graphviz dot", _normalize_panel_image(graphviz_path, PANEL_SIZE)))
    panels.append(("Dagua (strict)", _normalize_panel_image(strict_path, PANEL_SIZE)))
    panels.append(("Dagua (improved)", _normalize_panel_image(improved_path, PANEL_SIZE)))

    width = PANEL_SIZE[0] * len(panels)
    height = TITLE_HEIGHT + HEADER_HEIGHT + PANEL_SIZE[1]
    canvas = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(22, bold=True)
    header_font = _load_font(18, bold=False)

    _draw_centered_text(draw, (0, 0, width, TITLE_HEIGHT), title, title_font, TEXT_COLOR)
    draw.line((0, TITLE_HEIGHT, width, TITLE_HEIGHT), fill=LINE_COLOR, width=1)
    draw.line(
        (0, TITLE_HEIGHT + HEADER_HEIGHT, width, TITLE_HEIGHT + HEADER_HEIGHT),
        fill=LINE_COLOR,
        width=1,
    )

    for index, (header, panel_image) in enumerate(panels):
        left = index * PANEL_SIZE[0]
        right = left + PANEL_SIZE[0]
        _draw_centered_text(
            draw,
            (left, TITLE_HEIGHT, right, TITLE_HEIGHT + HEADER_HEIGHT),
            header,
            header_font,
            TEXT_COLOR,
        )
        canvas.paste(panel_image, (left, TITLE_HEIGHT + HEADER_HEIGHT))
        if index > 0:
            draw.line((left, 0, left, height), fill=LINE_COLOR, width=1)

    canvas.save(output_path, format="PNG", dpi=(COMPOSE_DPI, COMPOSE_DPI))


def _write_index_html(cases: Sequence[GraphCase], output_dir: Path) -> None:
    """Write a local-file-friendly gallery page for the composed comparisons.

    Parameters
    ----------
    cases : Sequence[GraphCase]
        Cases that produced composed images.
    output_dir : Path
        Root output directory.

    Returns
    -------
    None
        The gallery file is written to ``output_dir / INDEX_NAME``.
    """

    items: List[str] = []
    for case in cases:
        image_rel = f"{THREE_WAY_DIR}/{case.slug}.png"
        items.append(
            "\n".join(
                [
                    '<section class="entry">',
                    f"  <h2>{html.escape(case.title)}</h2>",
                    f'  <img src="{html.escape(image_rel)}" alt="{html.escape(case.title)}">',
                    "</section>",
                ]
            )
        )

    html_text = "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            '  <meta charset="utf-8">',
            "  <title>Graphviz Theme Comparison</title>",
            "  <style>",
            "    body {",
            "      font-family: Arial, sans-serif;",
            "      margin: 0;",
            "      background: #f5f5f5;",
            "      color: #111;",
            "    }",
            "    main { max-width: 1880px; margin: 0 auto; padding: 24px; }",
            "    h1 { margin: 0 0 20px; font-size: 28px; }",
            "    .entry {",
            "      margin: 0 0 28px;",
            "      padding: 18px;",
            "      background: #fff;",
            "      border: 1px solid #ddd;",
            "    }",
            "    .entry h2 { margin: 0 0 12px; font-size: 20px; }",
            "    .entry img {",
            "      display: block;",
            "      width: 100%;",
            "      height: auto;",
            "      border: 1px solid #e5e5e5;",
            "    }",
            "  </style>",
            "</head>",
            "<body>",
            "  <main>",
            "    <h1>Graphviz Theme Comparison</h1>",
            *items,
            "  </main>",
            "</body>",
            "</html>",
        ]
    )
    (output_dir / INDEX_NAME).write_text(html_text, encoding="utf-8")


def _ensure_output_dirs(output_dir: Path) -> Dict[str, Path]:
    """Create and return the output directory layout.

    Parameters
    ----------
    output_dir : Path
        Root output directory.

    Returns
    -------
    dict[str, Path]
        Named output subdirectories.
    """

    directories = {
        THREE_WAY_DIR: output_dir / THREE_WAY_DIR,
        STRICT_DIR: output_dir / STRICT_DIR,
        IMPROVED_DIR: output_dir / IMPROVED_DIR,
        GRAPHVIZ_DIR: output_dir / GRAPHVIZ_DIR,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def _iter_cases(quick: bool) -> List[GraphCase]:
    """Return the ordered case list for this run.

    Parameters
    ----------
    quick : bool
        Whether to skip bundled YAML graphs.

    Returns
    -------
    list[GraphCase]
        Cases to render.
    """

    cases = _custom_graph_cases()
    if not quick:
        cases.extend(_bundled_graph_cases())
    return cases


def _render_case(
    case: GraphCase,
    directories: Mapping[str, Path],
    use_graphviz: bool,
) -> None:
    """Render one graph case into all requested outputs.

    Parameters
    ----------
    case : GraphCase
        Graph case to render.
    directories : Mapping[str, Path]
        Output directory mapping.
    use_graphviz : bool
        Whether to attempt the native Graphviz column.

    Returns
    -------
    None
        The outputs are written to disk.
    """

    strict_path = directories[STRICT_DIR] / f"{case.slug}.png"
    improved_path = directories[IMPROVED_DIR] / f"{case.slug}.png"
    graphviz_path = directories[GRAPHVIZ_DIR] / f"{case.slug}.png"
    composed_path = directories[THREE_WAY_DIR] / f"{case.slug}.png"

    render_dagua_theme(case.graph, STRICT_THEME_NAME, strict_path)
    render_dagua_theme(case.graph, IMPROVED_THEME_NAME, improved_path)

    native_available = False
    if use_graphviz:
        native_available = render_graphviz_native(case.graph, graphviz_path)

    compose_three_way(
        strict_path=strict_path,
        improved_path=improved_path,
        title=case.title,
        output_path=composed_path,
        graphviz_path=graphviz_path if native_available else None,
    )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the comparison pipeline.

    Parameters
    ----------
    argv : Sequence[str], optional
        Explicit argument vector.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="eval_output/graphviz_theme",
        help="Directory to receive comparison outputs.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Render only the programmatic showcase graphs.",
    )
    parser.add_argument(
        "--no-graphviz",
        action="store_true",
        help="Skip native Graphviz rendering even if dot is installed.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the three-way Graphviz theme comparison pipeline.

    Parameters
    ----------
    argv : Sequence[str], optional
        Explicit argument vector.

    Returns
    -------
    int
        Process exit status.
    """

    args = _parse_args(argv)
    output_dir = Path(args.output_dir)
    directories = _ensure_output_dirs(output_dir)
    cases = _iter_cases(quick=bool(args.quick))
    use_graphviz = not bool(args.no_graphviz) and _graphviz_available()

    for case in cases:
        print(f"Rendering {case.slug}...", flush=True)
        _render_case(case, directories, use_graphviz)

    _write_index_html(cases, output_dir)
    print(f"Wrote {len(cases)} comparison rows to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
