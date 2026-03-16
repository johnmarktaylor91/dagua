#!/usr/bin/env python
# ruff: noqa: E402
"""Generate a cosmetic combination album comparing Dagua and Graphviz.

This album extends the single-option cosmetic album by stressing combinations
of visual choices, such as shape plus border treatment or arrow style plus
routing. Cases without a practical Graphviz analogue are rendered as Dagua-only
panels.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from PIL import Image

from dagua import DaguaGraph
from dagua.styles import EdgeStyle, GraphStyle, NodeStyle
from scripts.generate_cosmetic_album import (
    AlbumCase,
    CosmeticAlbumResult,
    GraphvizRenderSpec,
    _apply_graph_style,
    _base_cluster_style,
    _base_edge_style,
    _base_node_style,
    _build_graphviz_dot,
    _compose_comparison,
    _compose_solo,
    _pair_graph,
    _render_dagua_png,
    _render_graphviz_png,
    _set_all_edge_styles,
    _set_all_node_styles,
    _single_node_graph,
)

DEFAULT_OUTPUT_DIR = "eval_output/cosmetic_combos"
GRAPHVIZ_LABEL = "Graphviz"
RAW_RENDER_DPI = 200

CATEGORY_DESCRIPTIONS: Dict[str, str] = {
    "01_shape_x_border": "Tests dashed and dotted borders on complex node shapes.",
    "02_shape_x_gradient": "Tests whether gradients clip cleanly inside non-rectangular nodes.",
    "03_arrow_x_edgestyle": "Tests arrowheads combined with dashed and dotted edges.",
    "04_arrow_x_routing": "Tests arrowheads at ortho, straight, and bezier endpoints.",
    "05_arrow_proportions": "Tests the balance between edge width and arrowhead size.",
    "06_arrow_head_tail": "Tests mixed head and tail arrow combinations on short spans.",
    "07_text_overflow": "Tests long text, multiline text, and typography stress inside nodes.",
    "08_edge_labels": "Tests edge labels against routing, dash styles, and crowding.",
    "09_short_edges": "Tests rendering when nodes are intentionally packed close together.",
    "10_self_loops_parallel": (
        "Tests self-loops and parallel edges, which are classic stress cases."
    ),
    "11_opacity_interactions": "Tests opacity combined with gradients, borders, and arrows.",
    "12_shadow_interactions": (
        "Tests shadows combined with borders, gradients, and non-rectangular nodes."
    ),
    "13_direction_x_routing": "Tests direction changes combined with routing, arrows, and labels.",
    "14_cluster_combos": "Tests cluster styles combined with nodes, edges, and hierarchy depth.",
    "15_color_contrast": "Tests readable and intentionally bad color combinations.",
    "16_dark_mode": "Tests gradients, shadows, and opacity on a dark background.",
    "17_extreme_params": "Pushes render parameters to extremes to expose breakpoints.",
    "18_dense_mixed": "Tests larger mixed-style scenes where visual clutter becomes obvious.",
    "19_real_world_patterns": (
        "Tests realistic workflow, state-machine, and org-chart combinations."
    ),
    "20_kitchen_sink": "Combines three or more options per case for full stress testing.",
}

CATEGORY_RISK: Dict[str, str] = {
    "01_shape_x_border": "high",
    "02_shape_x_gradient": "high",
    "03_arrow_x_edgestyle": "high",
    "04_arrow_x_routing": "medium",
    "05_arrow_proportions": "medium",
    "06_arrow_head_tail": "high",
    "07_text_overflow": "high",
    "08_edge_labels": "high",
    "09_short_edges": "high",
    "10_self_loops_parallel": "high",
    "11_opacity_interactions": "medium",
    "12_shadow_interactions": "medium",
    "13_direction_x_routing": "medium",
    "14_cluster_combos": "high",
    "15_color_contrast": "high",
    "16_dark_mode": "high",
    "17_extreme_params": "medium",
    "18_dense_mixed": "high",
    "19_real_world_patterns": "high",
    "20_kitchen_sink": "high",
}

CategoryBuilder = Callable[[], List[AlbumCase]]


def _pair_positions(
    gap: float = 170.0,
    direction: str = "TB",
) -> List[Tuple[float, float]]:
    """Return fixed two-node positions for a directional pair.

    Parameters
    ----------
    gap : float, default=170.0
        Distance between the two nodes.
    direction : str, default="TB"
        One of ``TB``, ``BT``, ``LR``, or ``RL``.

    Returns
    -------
    list[tuple[float, float]]
        Source position first and target position second.
    """

    half_gap = gap / 2.0
    if direction == "TB":
        return [(0.0, half_gap), (0.0, -half_gap)]
    if direction == "BT":
        return [(0.0, -half_gap), (0.0, half_gap)]
    if direction == "LR":
        return [(-half_gap, 0.0), (half_gap, 0.0)]
    if direction == "RL":
        return [(half_gap, 0.0), (-half_gap, 0.0)]
    raise ValueError(f"Unsupported pair direction: {direction}")


def _chain_graph(
    n: int,
    labels: Sequence[str],
    direction: str = "TB",
    spacing: float = 170.0,
) -> Tuple[DaguaGraph, torch.Tensor]:
    """Build an ``n``-node chain with fixed directional positions.

    Parameters
    ----------
    n : int
        Number of nodes in the chain.
    labels : sequence[str]
        Node labels in order.
    direction : str, default="TB"
        One of ``TB``, ``BT``, ``LR``, or ``RL``.
    spacing : float, default=170.0
        Distance between consecutive nodes.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        The configured graph and a position tensor with shape ``[n, 2]``.
    """

    graph = DaguaGraph(direction=direction)
    _apply_graph_style(graph)
    for index, label in enumerate(labels):
        graph.add_node(f"n{index}", label=label)
    for index in range(n - 1):
        graph.add_edge(f"n{index}", f"n{index + 1}")
    if direction in ("TB", "BT"):
        sign = -1.0 if direction == "TB" else 1.0
        positions = [(0.0, sign * index * spacing) for index in range(n)]
    else:
        sign = 1.0 if direction == "LR" else -1.0
        positions = [(sign * index * spacing, 0.0) for index in range(n)]
    return graph, torch.tensor(positions, dtype=torch.float32)


def _diamond_dag(
    labels: Optional[Sequence[str]] = None,
) -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a four-node diamond DAG with fixed positions.

    Parameters
    ----------
    labels : sequence[str] | None, default=None
        Optional node labels. Defaults to ``["Start", "Left", "Right", "End"]``.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        The configured graph and a position tensor with shape ``[4, 2]``.
    """

    node_labels = list(labels) if labels is not None else ["Start", "Left", "Right", "End"]
    graph = DaguaGraph(direction="TB")
    _apply_graph_style(graph)
    for index, label in enumerate(node_labels):
        graph.add_node(f"n{index}", label=label)
    graph.add_edge("n0", "n1")
    graph.add_edge("n0", "n2")
    graph.add_edge("n1", "n3")
    graph.add_edge("n2", "n3")
    positions = [(0.0, 170.0), (-120.0, 0.0), (120.0, 0.0), (0.0, -170.0)]
    return graph, torch.tensor(positions, dtype=torch.float32)


def _fan_graph(
    center_label: str,
    leaf_labels: Sequence[str],
    direction: str = "TB",
) -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a star / fan-out graph with fixed positions.

    Parameters
    ----------
    center_label : str
        Label for the center node.
    leaf_labels : sequence[str]
        Labels for leaf nodes.
    direction : str, default="TB"
        One of ``TB``, ``BT``, ``LR``, or ``RL``.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Configured graph and position tensor.
    """

    graph = DaguaGraph(direction=direction)
    _apply_graph_style(graph)
    graph.add_node("center", label=center_label)
    for index, label in enumerate(leaf_labels):
        node_id = f"leaf{index}"
        graph.add_node(node_id, label=label)
        graph.add_edge("center", node_id)

    count = len(leaf_labels)
    midpoint = (count - 1) / 2.0
    positions: List[Tuple[float, float]] = []
    if direction == "TB":
        positions.append((0.0, 120.0))
        positions.extend(((index - midpoint) * 120.0, -40.0) for index in range(count))
    elif direction == "BT":
        positions.append((0.0, -120.0))
        positions.extend(((index - midpoint) * 120.0, 40.0) for index in range(count))
    elif direction == "LR":
        positions.append((-150.0, 0.0))
        positions.extend((60.0, (midpoint - index) * 120.0) for index in range(count))
    elif direction == "RL":
        positions.append((150.0, 0.0))
        positions.extend((-60.0, (midpoint - index) * 120.0) for index in range(count))
    else:
        raise ValueError(f"Unsupported fan direction: {direction}")
    return graph, torch.tensor(positions, dtype=torch.float32)


def _grid_graph(rows: int, cols: int) -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a right-and-down grid graph with fixed positions.

    Parameters
    ----------
    rows : int
        Number of rows.
    cols : int
        Number of columns.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Configured graph and position tensor.
    """

    graph = DaguaGraph(direction="TB")
    _apply_graph_style(graph)
    positions: List[Tuple[float, float]] = []
    for row in range(rows):
        for col in range(cols):
            node_id = f"n{row}_{col}"
            graph.add_node(node_id, label=f"{row},{col}")
            positions.append(((col - (cols - 1) / 2.0) * 120.0, ((rows - 1) / 2.0 - row) * 110.0))
            if col > 0:
                graph.add_edge(f"n{row}_{col - 1}", node_id)
            if row > 0:
                graph.add_edge(f"n{row - 1}_{col}", node_id)
    return graph, torch.tensor(positions, dtype=torch.float32)


def _mixed_shape_graph(
    shapes: Sequence[str],
    labels: Sequence[str],
    edges: Sequence[Tuple[int, int]],
) -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a graph with per-node shapes and fixed positions.

    Parameters
    ----------
    shapes : sequence[str]
        Per-node shapes.
    labels : sequence[str]
        Per-node labels.
    edges : sequence[tuple[int, int]]
        Directed edges using node indices.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Configured graph and position tensor.
    """

    graph = DaguaGraph(direction="TB")
    _apply_graph_style(graph)
    positions: List[Tuple[float, float]] = []
    cols = min(3, max(len(labels), 1))
    for index, label in enumerate(labels):
        node_id = f"n{index}"
        graph.add_node(node_id, label=label)
        row = index // cols
        col = index % cols
        positions.append(((col - (cols - 1) / 2.0) * 140.0, (1 - row) * 130.0))
    for source, target in edges:
        graph.add_edge(f"n{source}", f"n{target}")
    graph.node_styles = [_base_node_style(shape=shape) for shape in shapes]
    return graph, torch.tensor(positions, dtype=torch.float32)


def _format_float(value: float) -> str:
    """Return a compact decimal string for DOT attributes.

    Parameters
    ----------
    value : float
        Numeric value to serialize.

    Returns
    -------
    str
        Decimal string without trailing zeros.
    """

    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _pinned_graphviz_node_attrs(positions: torch.Tensor) -> Dict[int, Dict[str, str]]:
    """Return Graphviz node attributes that pin nodes to fixed positions.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.

    Returns
    -------
    dict[int, dict[str, str]]
        Per-node ``pos`` and ``pin`` overrides keyed by node index.
    """

    attrs: Dict[int, Dict[str, str]] = {}
    for index, xy in enumerate(positions.detach().cpu().tolist()):
        x_coord, y_coord = xy
        attrs[index] = {"pos": f"{_format_float(x_coord)},{_format_float(y_coord)}!", "pin": "true"}
    return attrs


def _pinned_graphviz_spec(
    positions: torch.Tensor,
    *,
    graph_attrs: Optional[Mapping[str, str]] = None,
    default_node_attrs: Optional[Mapping[str, str]] = None,
    default_edge_attrs: Optional[Mapping[str, str]] = None,
    node_attrs: Optional[Mapping[int, Mapping[str, str]]] = None,
    edge_attrs: Optional[Mapping[int, Mapping[str, str]]] = None,
    cluster_attrs: Optional[Mapping[str, Mapping[str, str]]] = None,
    engine: str = "neato",
) -> GraphvizRenderSpec:
    """Create a Graphviz render spec that preserves the provided node positions.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    graph_attrs : Mapping[str, str] | None, default=None
        Graph-level DOT attribute overrides.
    default_node_attrs : Mapping[str, str] | None, default=None
        Default DOT node attributes.
    default_edge_attrs : Mapping[str, str] | None, default=None
        Default DOT edge attributes.
    node_attrs : Mapping[int, Mapping[str, str]] | None, default=None
        Additional per-node overrides.
    edge_attrs : Mapping[int, Mapping[str, str]] | None, default=None
        Per-edge overrides keyed by edge index.
    cluster_attrs : Mapping[str, Mapping[str, str]] | None, default=None
        Per-cluster DOT overrides keyed by cluster name.
    engine : str, default="neato"
        Graphviz executable to run.

    Returns
    -------
    GraphvizRenderSpec
        Render configuration with pinned node positions.
    """

    merged_graph_attrs: Dict[str, str] = {"overlap": "false", "notranslate": "true"}
    if graph_attrs is not None:
        merged_graph_attrs.update(graph_attrs)

    merged_node_attrs = _pinned_graphviz_node_attrs(positions)
    if node_attrs is not None:
        for index, attrs in node_attrs.items():
            current = dict(merged_node_attrs.get(index, {}))
            current.update(attrs)
            merged_node_attrs[index] = current

    edge_overrides = {index: dict(attrs) for index, attrs in (edge_attrs or {}).items()}
    return GraphvizRenderSpec(
        graph_attrs=merged_graph_attrs,
        default_node_attrs=dict(default_node_attrs or {}),
        default_edge_attrs=dict(default_edge_attrs or {}),
        node_attrs=merged_node_attrs,
        edge_attrs=edge_overrides,
        cluster_attrs={name: dict(attrs) for name, attrs in (cluster_attrs or {}).items()},
        engine=engine,
        competitor_label=GRAPHVIZ_LABEL,
    )


def _make_case(
    *,
    case_id: str,
    category: str,
    title: str,
    graph: DaguaGraph,
    positions: torch.Tensor,
    options_tested: Sequence[str],
    graphviz: Optional[GraphvizRenderSpec],
) -> AlbumCase:
    """Create an album case with manifest-friendly metadata.

    Parameters
    ----------
    case_id : str
        Stable case identifier.
    category : str
        Output category directory.
    title : str
        Human-readable case title.
    graph : DaguaGraph
        Graph to render.
    positions : torch.Tensor
        Fixed node positions with shape ``[N, 2]``.
    options_tested : sequence[str]
        High-level option summary for the manifest.
    graphviz : GraphvizRenderSpec | None
        Competitor render configuration, or ``None`` for Dagua-only cases.

    Returns
    -------
    AlbumCase
        Configured album case.
    """

    return AlbumCase(
        case_id=case_id,
        category=category,
        filename=f"{case_id}.png",
        title=title,
        graph=graph,
        positions=positions,
        settings={"options_tested": list(options_tested)},
        graphviz=graphviz,
    )


def _shape_x_border_cases() -> List[AlbumCase]:
    """Build shape-plus-border comparison cases.

    Returns
    -------
    list[AlbumCase]
        Requested ``01_shape_x_border`` cases.
    """

    specs = [
        ("star_dashed", "star", "dashed", "star"),
        ("star_dotted", "star", "dotted", "star"),
        ("hexagon_dashed", "hexagon", "dashed", "hexagon"),
        ("hexagon_dotted", "hexagon", "dotted", "hexagon"),
        ("octagon_dashed", "octagon", "dashed", "octagon"),
        ("diamond_dashed", "diamond", "dashed", "diamond"),
        ("diamond_dotted", "diamond", "dotted", "diamond"),
        ("cylinder_dashed", "cylinder", "dashed", "cylinder"),
        ("triangle_dotted", "triangle", "dotted", "triangle"),
        ("parallelogram_dashed", "parallelogram", "dashed", "parallelogram"),
        ("trapezoid_dotted", "trapezoid", "dotted", "trapezium"),
        ("circle_dashed", "circle", "dashed", "circle"),
    ]

    cases: List[AlbumCase] = []
    for case_id, shape, border, gv_shape in specs:
        graph, positions = _pair_graph(_pair_positions(), ["Alpha", "Beta"])
        _set_all_node_styles(graph, _base_node_style(shape=shape, stroke_dash=border))
        graphviz = _pinned_graphviz_spec(
            positions,
            default_node_attrs={"shape": gv_shape, "style": f"filled,{border}"},
        )
        cases.append(
            _make_case(
                case_id=case_id,
                category="01_shape_x_border",
                title=f"{shape.title()} + {border} border",
                graph=graph,
                positions=positions,
                options_tested=[f"shape={shape}", f"stroke_dash={border}"],
                graphviz=graphviz,
            )
        )
    return cases


def _shape_x_gradient_cases() -> List[AlbumCase]:
    """Build Dagua-only shape-plus-gradient cases.

    Returns
    -------
    list[AlbumCase]
        Requested ``02_shape_x_gradient`` cases.
    """

    specs = [
        ("diamond_linear", "diamond", "linear"),
        ("diamond_radial", "diamond", "radial"),
        ("circle_linear", "circle", "linear"),
        ("circle_radial", "circle", "radial"),
        ("star_linear", "star", "linear"),
        ("hexagon_radial", "hexagon", "radial"),
        ("cylinder_linear", "cylinder", "linear"),
        ("triangle_radial", "triangle", "radial"),
    ]

    cases: List[AlbumCase] = []
    for case_id, shape, gradient in specs:
        graph, positions = _single_node_graph(shape.title())
        graph.node_styles[0] = _base_node_style(
            shape=shape,
            gradient=gradient,
            gradient_color="#A0C4E8",
        )
        cases.append(
            _make_case(
                case_id=case_id,
                category="02_shape_x_gradient",
                title=f"{shape.title()} + {gradient} gradient",
                graph=graph,
                positions=positions,
                options_tested=[
                    f"shape={shape}",
                    f"gradient={gradient}",
                    "gradient_color=#A0C4E8",
                ],
                graphviz=None,
            )
        )
    return cases


def _arrow_x_edgestyle_cases() -> List[AlbumCase]:
    """Build arrow-plus-edge-style comparison cases.

    Returns
    -------
    list[AlbumCase]
        Requested ``03_arrow_x_edgestyle`` cases.
    """

    specs = [
        ("vee_dashed", "vee", "dashed", "vee"),
        ("vee_dotted", "vee", "dotted", "vee"),
        ("diamond_dashed", "diamond", "dashed", "diamond"),
        ("diamond_dotted", "diamond", "dotted", "diamond"),
        ("dot_dashed", "dot", "dashed", "dot"),
        ("dot_dotted", "dot", "dotted", "dot"),
        ("tee_dashed", "tee", "dashed", "tee"),
        ("tee_dotted", "tee", "dotted", "tee"),
        ("crow_dashed", "crow", "dashed", "crow"),
        ("crow_dotted", "crow", "dotted", "crow"),
        ("normal_dashed", "normal", "dashed", "normal"),
        ("normal_dotted", "normal", "dotted", "normal"),
    ]

    cases: List[AlbumCase] = []
    for case_id, arrow, edge_style, gv_arrow in specs:
        graph, positions = _pair_graph(_pair_positions(), ["Source", "Target"])
        _set_all_edge_styles(graph, _base_edge_style(arrow=arrow, style=edge_style))
        graphviz = _pinned_graphviz_spec(
            positions,
            default_edge_attrs={"arrowhead": gv_arrow, "style": edge_style},
        )
        cases.append(
            _make_case(
                case_id=case_id,
                category="03_arrow_x_edgestyle",
                title=f"{arrow.title()} arrow + {edge_style} edge",
                graph=graph,
                positions=positions,
                options_tested=[f"arrow={arrow}", f"edge_style={edge_style}"],
                graphviz=graphviz,
            )
        )
    return cases


def _arrow_x_routing_cases() -> List[AlbumCase]:
    """Build arrow-plus-routing comparison cases.

    Returns
    -------
    list[AlbumCase]
        Requested ``04_arrow_x_routing`` cases.
    """

    specs = [
        ("normal_ortho", "normal", "ortho", "normal", "ortho"),
        ("vee_ortho", "vee", "ortho", "vee", "ortho"),
        ("diamond_ortho", "diamond", "ortho", "diamond", "ortho"),
        ("tee_ortho", "tee", "ortho", "tee", "ortho"),
        ("crow_ortho", "crow", "ortho", "crow", "ortho"),
        ("normal_straight", "normal", "straight", "normal", "false"),
        ("vee_straight", "vee", "straight", "vee", "false"),
        ("diamond_straight", "diamond", "straight", "diamond", "false"),
        ("normal_bezier", "normal", "bezier", "normal", "true"),
    ]

    positions = [(0.0, 170.0), (120.0, 0.0)]
    cases: List[AlbumCase] = []
    for case_id, arrow, routing, gv_arrow, gv_splines in specs:
        graph, tensor = _pair_graph(positions, ["Input", "Output"])
        _set_all_edge_styles(graph, _base_edge_style(arrow=arrow, routing=routing))
        graphviz = _pinned_graphviz_spec(
            tensor,
            graph_attrs={"splines": gv_splines},
            default_edge_attrs={"arrowhead": gv_arrow},
        )
        cases.append(
            _make_case(
                case_id=case_id,
                category="04_arrow_x_routing",
                title=f"{arrow.title()} arrow + {routing} routing",
                graph=graph,
                positions=tensor,
                options_tested=[f"arrow={arrow}", f"routing={routing}"],
                graphviz=graphviz,
            )
        )
    return cases


def _arrow_proportions_cases() -> List[AlbumCase]:
    """Build arrow proportion comparison cases.

    Returns
    -------
    list[AlbumCase]
        Requested ``05_arrow_proportions`` cases.
    """

    specs = [
        ("normal_thin", "normal", 0.5, 14.0, 10.0, 0.9),
        ("normal_thick", "normal", 4.0, 14.0, 10.0, 0.9),
        ("large_thin", "normal", 0.8, 22.0, 16.0, 1.5),
        ("large_thick", "normal", 4.0, 22.0, 16.0, 1.5),
        ("small_thin", "normal", 0.8, 6.0, 4.0, 0.4),
        ("small_thick", "normal", 4.0, 6.0, 4.0, 0.4),
        ("vee_thick", "vee", 3.0, 14.0, 10.0, 0.9),
        ("diamond_thick", "diamond", 3.0, 14.0, 10.0, 0.9),
        ("dot_thick", "dot", 3.0, 14.0, 10.0, 0.9),
    ]

    cases: List[AlbumCase] = []
    for case_id, arrow, width, arrow_length, arrow_width, gv_arrowsize in specs:
        graph, positions = _pair_graph(_pair_positions(), ["Source", "Target"])
        _set_all_edge_styles(
            graph,
            _base_edge_style(
                arrow=arrow,
                width=width,
                arrow_length=arrow_length,
                arrow_width=arrow_width,
            ),
        )
        default_edge_attrs = {
            "penwidth": _format_float(width),
            "arrowsize": _format_float(gv_arrowsize),
        }
        if arrow != "normal":
            default_edge_attrs["arrowhead"] = arrow
        graphviz = _pinned_graphviz_spec(positions, default_edge_attrs=default_edge_attrs)
        cases.append(
            _make_case(
                case_id=case_id,
                category="05_arrow_proportions",
                title=f"{arrow.title()} arrow proportions",
                graph=graph,
                positions=positions,
                options_tested=[
                    f"arrow={arrow}",
                    f"width={_format_float(width)}",
                    f"arrow_length={_format_float(arrow_length)}",
                    f"arrow_width={_format_float(arrow_width)}",
                ],
                graphviz=graphviz,
            )
        )
    return cases


def _arrow_head_tail_cases() -> List[AlbumCase]:
    """Build bidirectional head-tail comparison cases.

    Returns
    -------
    list[AlbumCase]
        Requested ``06_arrow_head_tail`` cases.
    """

    specs = [
        ("normal_dot", "normal", "dot", "normal", "dot", None),
        ("normal_diamond", "normal", "diamond", "normal", "diamond", None),
        ("vee_tee", "vee", "tee", "vee", "tee", None),
        ("diamond_diamond", "diamond", "diamond", "diamond", "diamond", None),
        ("crow_dot", "crow", "dot", "crow", "dot", None),
        ("filled_hollow", "normal", "open", "normal", "empty", "filled"),
        ("circle_circle", "circle", "circle", "circle", "circle", None),
        ("normal_normal", "normal", "normal", "normal", "normal", None),
    ]

    cases: List[AlbumCase] = []
    for case_id, arrow, tail_arrow, gv_head, gv_tail, arrow_fill in specs:
        graph, positions = _pair_graph(_pair_positions(gap=120.0), ["Source", "Target"])
        edge_kwargs: Dict[str, object] = {"arrow": arrow, "tail_arrow": tail_arrow}
        if arrow_fill is not None:
            edge_kwargs["arrow_fill"] = arrow_fill
        _set_all_edge_styles(graph, _base_edge_style(**edge_kwargs))
        graphviz = _pinned_graphviz_spec(
            positions,
            default_edge_attrs={"dir": "both", "arrowhead": gv_head, "arrowtail": gv_tail},
        )
        options = [f"arrow={arrow}", f"tail_arrow={tail_arrow}"]
        if arrow_fill is not None:
            options.append(f"arrow_fill={arrow_fill}")
        cases.append(
            _make_case(
                case_id=case_id,
                category="06_arrow_head_tail",
                title=f"{arrow.title()} head + {tail_arrow.title()} tail",
                graph=graph,
                positions=positions,
                options_tested=options,
                graphviz=graphviz,
            )
        )
    return cases


def _graphviz_font_name(font_weight: str, font_style: str) -> str:
    """Return a conservative Graphviz font name for weight/style combinations.

    Parameters
    ----------
    font_weight : str
        Requested font weight.
    font_style : str
        Requested font style.

    Returns
    -------
    str
        Graphviz font name string.
    """

    if font_weight == "bold" and font_style == "italic":
        return "Helvetica Bold Oblique"
    if font_weight == "bold":
        return "Helvetica Bold"
    if font_style == "italic":
        return "Helvetica Oblique"
    return "Helvetica"


def _text_overflow_cases() -> List[AlbumCase]:
    """Build text-overflow and typography cases.

    Returns
    -------
    list[AlbumCase]
        Requested ``07_text_overflow`` cases.
    """

    specs = [
        ("long_circle", "Supercalifragilisticexpialidocious", "circle", {}, True),
        ("long_diamond", "Supercalifragilisticexpialidocious", "diamond", {}, True),
        ("long_star", "Supercalifragilisticexpialidocious", "star", {}, True),
        ("threeword_circle", "Input Processing Layer", "circle", {}, True),
        ("threeword_diamond", "Input Processing Layer", "diamond", {}, True),
        ("multiline_diamond", "First Line\nSecond Line", "diamond", {}, True),
        ("multiline_triangle", "First Line\nSecond Line", "triangle", {}, True),
        (
            "bold_italic_circle",
            "Important Node",
            "circle",
            {"font_weight": "bold", "font_style": "italic"},
            True,
        ),
        ("bold_hexagon", "Important Node", "hexagon", {"font_weight": "bold"}, True),
        ("tiny_font_rect", "Small Text Example", "rect", {"font_size": 7.0}, False),
        ("huge_font_rect", "Big", "rect", {"font_size": 24.0}, False),
        ("expand_circle", "Long Label Here", "circle", {"overflow_policy": "expand_node"}, False),
    ]

    cases: List[AlbumCase] = []
    for case_id, label, shape, overrides, include_graphviz in specs:
        graph, positions = _single_node_graph(label)
        graph.node_styles[0] = _base_node_style(shape=shape, **overrides)

        graphviz: Optional[GraphvizRenderSpec] = None
        if include_graphviz:
            gv_node_attrs = {
                "shape": shape,
                "fontname": _graphviz_font_name(
                    str(overrides.get("font_weight", "regular")),
                    str(overrides.get("font_style", "normal")),
                ),
            }
            graphviz = _pinned_graphviz_spec(positions, default_node_attrs=gv_node_attrs)

        option_list = [f"shape={shape}", f"label={label!r}"]
        for key, value in overrides.items():
            option_list.append(f"{key}={value}")
        cases.append(
            _make_case(
                case_id=case_id,
                category="07_text_overflow",
                title=case_id.replace("_", " ").title(),
                graph=graph,
                positions=positions,
                options_tested=option_list,
                graphviz=graphviz,
            )
        )
    return cases


def _set_all_edge_labels(graph: DaguaGraph, labels: Sequence[Optional[str]]) -> None:
    """Assign edge labels to every edge in order.

    Parameters
    ----------
    graph : DaguaGraph
        Graph whose edge labels should be updated.
    labels : sequence[str | None]
        Labels to assign in edge order.

    Returns
    -------
    None
        The graph is mutated in place.
    """

    graph.edge_labels = list(labels)


def _edge_label_cases() -> List[AlbumCase]:
    """Build edge-label comparison cases.

    Returns
    -------
    list[AlbumCase]
        Requested ``08_edge_labels`` cases.
    """

    specs = [
        ("label_bezier", "flow", "bezier", "solid", "normal", "pair", 170.0, None),
        ("label_ortho", "flow", "ortho", "solid", "normal", "pair", 170.0, None),
        ("label_straight", "flow", "straight", "solid", "normal", "pair", 170.0, None),
        ("label_dashed", "flow", "bezier", "dashed", "normal", "pair", 170.0, None),
        ("label_dotted", "flow", "bezier", "dotted", "normal", "pair", 170.0, None),
        ("label_vee", "flow", "bezier", "solid", "vee", "pair", 170.0, None),
        ("label_short", "flow", "bezier", "solid", "normal", "pair", 60.0, None),
        (
            "label_long",
            "data transformation pipeline",
            "bezier",
            "solid",
            "normal",
            "pair",
            170.0,
            None,
        ),
        ("label_multi", "", "bezier", "solid", "normal", "diamond", 170.0, ["a", "b", "c", "d"]),
        ("label_thick", "flow", "bezier", "solid", "normal", "pair", 170.0, None),
    ]

    cases: List[AlbumCase] = []
    for case_id, label, routing, edge_style, arrow, graph_kind, gap, multi_labels in specs:
        if graph_kind == "diamond":
            graph, positions = _diamond_dag()
            labels = cast(Sequence[Optional[str]], multi_labels)
            _set_all_edge_styles(
                graph, _base_edge_style(arrow=arrow, routing=routing, style=edge_style)
            )
            _set_all_edge_labels(graph, labels)
            graphviz = _pinned_graphviz_spec(
                positions,
                graph_attrs={"splines": "true"},
                default_edge_attrs={"style": edge_style},
            )
            options = [f"routing={routing}", "edge_labels=a,b,c,d"]
        else:
            graph, positions = _pair_graph(_pair_positions(gap=gap), ["Source", "Target"])
            edge_kwargs: Dict[str, object] = {
                "arrow": arrow,
                "routing": routing,
                "style": edge_style,
            }
            if case_id == "label_thick":
                edge_kwargs["width"] = 3.0
            _set_all_edge_styles(graph, _base_edge_style(**edge_kwargs))
            _set_all_edge_labels(graph, [label])
            gv_edge_attrs = {"style": edge_style}
            if arrow != "normal":
                gv_edge_attrs["arrowhead"] = arrow
            if case_id == "label_thick":
                gv_edge_attrs["penwidth"] = "3"
            graphviz = _pinned_graphviz_spec(
                positions,
                graph_attrs={
                    "splines": {"bezier": "true", "ortho": "ortho", "straight": "false"}[routing]
                },
                default_edge_attrs=gv_edge_attrs,
            )
            options = [
                f"label={label!r}",
                f"routing={routing}",
                f"edge_style={edge_style}",
                f"arrow={arrow}",
            ]
            if case_id == "label_thick":
                options.append("width=3")

        cases.append(
            _make_case(
                case_id=case_id,
                category="08_edge_labels",
                title=case_id.replace("_", " ").title(),
                graph=graph,
                positions=positions,
                options_tested=options,
                graphviz=graphviz,
            )
        )
    return cases


def _short_edge_cases() -> List[AlbumCase]:
    """Build short-edge comparison cases.

    Returns
    -------
    list[AlbumCase]
        Requested ``09_short_edges`` cases.
    """

    specs = [
        ("short_normal", {"arrow": "normal"}, {}),
        ("short_large_arrow", {"arrow": "normal", "arrow_length": 20.0}, {"arrowsize": "1.35"}),
        ("short_dashed", {"arrow": "normal", "style": "dashed"}, {"style": "dashed"}),
        ("short_dotted", {"arrow": "normal", "style": "dotted"}, {"style": "dotted"}),
        ("short_label", {"arrow": "normal"}, {}),
        ("short_vee", {"arrow": "vee"}, {"arrowhead": "vee"}),
        (
            "short_head_tail",
            {"arrow": "normal", "tail_arrow": "dot"},
            {"dir": "both", "arrowtail": "dot"},
        ),
        ("short_thick", {"arrow": "normal", "width": 3.0}, {"penwidth": "3"}),
    ]

    cases: List[AlbumCase] = []
    for case_id, edge_overrides, gv_overrides in specs:
        graph, positions = _pair_graph(_pair_positions(gap=50.0), ["A", "B"])
        _set_all_edge_styles(graph, _base_edge_style(**edge_overrides))
        if case_id == "short_label":
            _set_all_edge_labels(graph, ["x"])
        graphviz = _pinned_graphviz_spec(positions, default_edge_attrs=gv_overrides)
        options = [f"{key}={value}" for key, value in edge_overrides.items()]
        if case_id == "short_label":
            options.append("label='x'")
        cases.append(
            _make_case(
                case_id=case_id,
                category="09_short_edges",
                title=case_id.replace("_", " ").title(),
                graph=graph,
                positions=positions,
                options_tested=options,
                graphviz=graphviz,
            )
        )
    return cases


def _self_loop_graph(label: str = "Loop") -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a one-node self-loop graph.

    Parameters
    ----------
    label : str, default="Loop"
        Node label.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph with a single self-loop and fixed position.
    """

    graph = DaguaGraph()
    _apply_graph_style(graph)
    graph.add_node("A", label=label)
    graph.add_edge("A", "A")
    return graph, torch.tensor([[0.0, 0.0]], dtype=torch.float32)


def _parallel_graph(edge_count: int) -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a two-node graph with multiple parallel edges.

    Parameters
    ----------
    edge_count : int
        Number of parallel edges from ``A`` to ``B``.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph with fixed two-node positions.
    """

    graph, positions = _pair_graph(_pair_positions(gap=140.0), ["A", "B"])
    for _ in range(edge_count - 1):
        graph.add_edge("A", "B")
    return graph, positions


def _self_loop_plus_normal_graph() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a graph containing both a self-loop and a normal outgoing edge.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph with two nodes and one self-loop on the source.
    """

    graph = DaguaGraph()
    _apply_graph_style(graph)
    graph.add_node("A", label="A")
    graph.add_node("B", label="B")
    graph.add_edge("A", "A")
    graph.add_edge("A", "B")
    positions = torch.tensor([[0.0, 70.0], [0.0, -70.0]], dtype=torch.float32)
    return graph, positions


def _self_loops_parallel_cases() -> List[AlbumCase]:
    """Build self-loop and parallel-edge comparison cases.

    Returns
    -------
    list[AlbumCase]
        Requested ``10_self_loops_parallel`` cases.
    """

    cases: List[AlbumCase] = []

    graph, positions = _self_loop_graph()
    _set_all_edge_styles(graph, _base_edge_style())
    cases.append(
        _make_case(
            case_id="selfloop_normal",
            category="10_self_loops_parallel",
            title="Self-loop normal arrow",
            graph=graph,
            positions=positions,
            options_tested=["arrow=normal"],
            graphviz=_pinned_graphviz_spec(positions, graph_attrs={"splines": "true"}),
        )
    )

    graph, positions = _self_loop_graph()
    _set_all_edge_styles(graph, _base_edge_style(arrow="vee"))
    cases.append(
        _make_case(
            case_id="selfloop_vee",
            category="10_self_loops_parallel",
            title="Self-loop vee arrow",
            graph=graph,
            positions=positions,
            options_tested=["arrow=vee"],
            graphviz=_pinned_graphviz_spec(
                positions,
                graph_attrs={"splines": "true"},
                default_edge_attrs={"arrowhead": "vee"},
            ),
        )
    )

    graph, positions = _self_loop_graph()
    _set_all_edge_styles(graph, _base_edge_style(style="dashed"))
    cases.append(
        _make_case(
            case_id="selfloop_dashed",
            category="10_self_loops_parallel",
            title="Self-loop dashed edge",
            graph=graph,
            positions=positions,
            options_tested=["style=dashed"],
            graphviz=_pinned_graphviz_spec(
                positions,
                graph_attrs={"splines": "true"},
                default_edge_attrs={"style": "dashed"},
            ),
        )
    )

    graph, positions = _self_loop_graph()
    _set_all_edge_styles(graph, _base_edge_style())
    _set_all_edge_labels(graph, ["retry"])
    cases.append(
        _make_case(
            case_id="selfloop_label",
            category="10_self_loops_parallel",
            title="Self-loop with label",
            graph=graph,
            positions=positions,
            options_tested=["label='retry'"],
            graphviz=_pinned_graphviz_spec(positions, graph_attrs={"splines": "true"}),
        )
    )

    graph, positions = _parallel_graph(2)
    graph.edge_styles[0] = _base_edge_style(style="solid")
    graph.edge_styles[1] = _base_edge_style(style="dashed")
    cases.append(
        _make_case(
            case_id="parallel_mixed_style",
            category="10_self_loops_parallel",
            title="Parallel edges with mixed styles",
            graph=graph,
            positions=positions,
            options_tested=["edge[0].style=solid", "edge[1].style=dashed"],
            graphviz=_pinned_graphviz_spec(
                positions,
                graph_attrs={"splines": "true"},
                edge_attrs={0: {"style": "solid"}, 1: {"style": "dashed"}},
            ),
        )
    )

    graph, positions = _parallel_graph(2)
    graph.edge_styles[0] = _base_edge_style(arrow="normal")
    graph.edge_styles[1] = _base_edge_style(arrow="vee")
    cases.append(
        _make_case(
            case_id="parallel_mixed_arrow",
            category="10_self_loops_parallel",
            title="Parallel edges with mixed arrows",
            graph=graph,
            positions=positions,
            options_tested=["edge[0].arrow=normal", "edge[1].arrow=vee"],
            graphviz=_pinned_graphviz_spec(
                positions,
                graph_attrs={"splines": "true"},
                edge_attrs={0: {"arrowhead": "normal"}, 1: {"arrowhead": "vee"}},
            ),
        )
    )

    graph, positions = _parallel_graph(3)
    _set_all_edge_styles(graph, _base_edge_style())
    cases.append(
        _make_case(
            case_id="parallel_three",
            category="10_self_loops_parallel",
            title="Three parallel edges",
            graph=graph,
            positions=positions,
            options_tested=["parallel_edges=3"],
            graphviz=_pinned_graphviz_spec(positions, graph_attrs={"splines": "true"}),
        )
    )

    graph, positions = _self_loop_plus_normal_graph()
    _set_all_edge_styles(graph, _base_edge_style())
    cases.append(
        _make_case(
            case_id="selfloop_plus_normal",
            category="10_self_loops_parallel",
            title="Self-loop plus outgoing edge",
            graph=graph,
            positions=positions,
            options_tested=["edges=A->A,A->B"],
            graphviz=_pinned_graphviz_spec(positions, graph_attrs={"splines": "true"}),
        )
    )

    return cases


def _opacity_interaction_cases() -> List[AlbumCase]:
    """Build opacity interaction cases.

    Returns
    -------
    list[AlbumCase]
        Requested ``11_opacity_interactions`` cases.
    """

    specs = [
        ("opacity_gradient", {"opacity": 0.4, "gradient": "linear"}, {"opacity": 0.65}),
        ("opacity_shadow", {"opacity": 0.4, "shadow": True}, {"opacity": 0.65}),
        ("opacity_dashed_border", {"opacity": 0.3, "stroke_dash": "dashed"}, {"opacity": 0.65}),
        ("edge_opacity_dotted", {}, {"opacity": 0.3, "style": "dotted"}),
        ("edge_opacity_diamond", {}, {"opacity": 0.3, "arrow": "diamond"}),
        ("opacity_nodes_edges", {"opacity": 0.5}, {"opacity": 0.5}),
    ]

    cases: List[AlbumCase] = []
    for case_id, node_overrides, edge_overrides in specs:
        graph, positions = _pair_graph(_pair_positions(), ["Near", "Far"])
        _set_all_node_styles(graph, _base_node_style(**node_overrides))
        _set_all_edge_styles(graph, _base_edge_style(**edge_overrides))
        options = [f"node_{key}={value}" for key, value in node_overrides.items()]
        options.extend(f"edge_{key}={value}" for key, value in edge_overrides.items())
        cases.append(
            _make_case(
                case_id=case_id,
                category="11_opacity_interactions",
                title=case_id.replace("_", " ").title(),
                graph=graph,
                positions=positions,
                options_tested=options,
                graphviz=None,
            )
        )
    return cases


def _shadow_interaction_cases() -> List[AlbumCase]:
    """Build shadow interaction cases.

    Returns
    -------
    list[AlbumCase]
        Requested ``12_shadow_interactions`` cases.
    """

    specs = [
        (
            "shadow_dashed",
            {"shadow": True, "stroke_dash": "solid"},
        ),
        ("shadow_gradient", {"shadow": True, "gradient": "linear"}),
        ("shadow_radius", {"shadow": True, "corner_radius": 20.0}),
        ("shadow_circle", {"shadow": True, "shape": "circle"}),
        ("shadow_star", {"shadow": True, "shape": "star"}),
        ("shadow_opacity", {"shadow": True, "opacity": 0.5}),
    ]

    cases: List[AlbumCase] = []
    for case_id, node_overrides in specs:
        graph, positions = _single_node_graph(case_id.replace("_", "\n"))
        graph.node_styles[0] = _base_node_style(**node_overrides)
        cases.append(
            _make_case(
                case_id=case_id,
                category="12_shadow_interactions",
                title=case_id.replace("_", " ").title(),
                graph=graph,
                positions=positions,
                options_tested=[f"{key}={value}" for key, value in node_overrides.items()],
                graphviz=None,
            )
        )
    return cases


def _direction_x_routing_cases() -> List[AlbumCase]:
    """Build direction-versus-routing cases.

    Returns
    -------
    list[AlbumCase]
        Requested ``13_direction_x_routing`` cases.
    """

    specs = [
        ("lr_ortho_normal", "LR", "ortho", "normal", None, None),
        ("lr_ortho_vee", "LR", "ortho", "vee", None, None),
        ("rl_bezier_diamond", "RL", "bezier", "diamond", None, None),
        ("bt_ortho_normal", "BT", "ortho", "normal", None, None),
        ("bt_straight_tee", "BT", "straight", "tee", None, None),
        ("lr_bezier_label", "LR", "bezier", "normal", "flow", None),
        ("rl_ortho_label", "RL", "ortho", "normal", "flow", None),
        ("lr_bezier_dashed", "LR", "bezier", "normal", None, "dashed"),
    ]

    cases: List[AlbumCase] = []
    for case_id, direction, routing, arrow, label, style in specs:
        graph, positions = _chain_graph(3, ["A", "B", "C"], direction=direction)
        graph.edge_styles = [
            _base_edge_style(arrow=arrow, routing=routing, style=style or "solid"),
            _base_edge_style(arrow=arrow, routing=routing, style=style or "solid"),
        ]
        if label is not None:
            _set_all_edge_labels(graph, [label, label])
        gv_attrs = {
            "splines": {"ortho": "ortho", "bezier": "true", "straight": "false"}[routing],
            "rankdir": direction,
        }
        gv_edge_attrs: Dict[str, str] = {"arrowhead": arrow}
        if style is not None:
            gv_edge_attrs["style"] = style
        cases.append(
            _make_case(
                case_id=case_id,
                category="13_direction_x_routing",
                title=case_id.replace("_", " ").title(),
                graph=graph,
                positions=positions,
                options_tested=[
                    f"direction={direction}",
                    f"routing={routing}",
                    f"arrow={arrow}",
                    *(["label='flow'"] if label is not None else []),
                    *([f"style={style}"] if style is not None else []),
                ],
                graphviz=_pinned_graphviz_spec(
                    positions,
                    graph_attrs=gv_attrs,
                    default_edge_attrs=gv_edge_attrs,
                ),
            )
        )
    return cases


def _cluster_combos_cases() -> List[AlbumCase]:
    """Build cluster combination cases.

    Returns
    -------
    list[AlbumCase]
        Requested ``14_cluster_combos`` cases.
    """

    cases: List[AlbumCase] = []

    base_positions = torch.tensor([[-80.0, 0.0], [80.0, 0.0], [0.0, -150.0]], dtype=torch.float32)

    graph = DaguaGraph()
    _apply_graph_style(graph)
    graph.add_node("A", label="One")
    graph.add_node("B", label="Two")
    graph.add_node("C", label="Three")
    graph.add_edge("A", "B")
    graph.add_edge("B", "C")
    graph.add_cluster("group", ["A", "B"], label="Group")
    graph.node_styles = [
        _base_node_style(stroke_dash="dashed"),
        _base_node_style(stroke_dash="dashed"),
        _base_node_style(),
    ]
    graph.edge_styles = [_base_edge_style(), _base_edge_style()]
    graph.cluster_styles["group"] = _base_cluster_style(stroke_dash="dashed")
    cases.append(
        _make_case(
            case_id="cluster_dashed_nodes",
            category="14_cluster_combos",
            title="Dashed cluster + dashed nodes",
            graph=graph,
            positions=base_positions,
            options_tested=["cluster_dash=dashed", "node_dash=dashed"],
            graphviz=_pinned_graphviz_spec(
                base_positions,
                cluster_attrs={"group": {"style": "dashed"}},
                node_attrs={0: {"style": "filled,dashed"}, 1: {"style": "filled,dashed"}},
            ),
        )
    )

    graph = DaguaGraph()
    _apply_graph_style(graph)
    graph.add_node("A", label="One")
    graph.add_node("B", label="Two")
    graph.add_node("C", label="Three")
    graph.add_edge("A", "B")
    graph.add_edge("B", "C")
    graph.add_cluster("group", ["A", "B"], label="Group")
    _set_all_node_styles(graph, _base_node_style())
    _set_all_edge_styles(graph, _base_edge_style())
    graph.cluster_styles["group"] = _base_cluster_style(stroke_dash="dashed")
    cases.append(
        _make_case(
            case_id="cluster_dashed_solid_nodes",
            category="14_cluster_combos",
            title="Dashed cluster + solid nodes",
            graph=graph,
            positions=base_positions,
            options_tested=["cluster_dash=dashed", "node_dash=solid"],
            graphviz=_pinned_graphviz_spec(
                base_positions, cluster_attrs={"group": {"style": "dashed"}}
            ),
        )
    )

    graph = DaguaGraph()
    _apply_graph_style(graph)
    graph.add_node("A", label="Grad A")
    graph.add_node("B", label="Grad B")
    graph.add_node("C", label="Outside")
    graph.add_edge("A", "B")
    graph.add_edge("B", "C")
    graph.add_cluster("group", ["A", "B"], label="Filled")
    graph.node_styles = [
        _base_node_style(gradient="linear"),
        _base_node_style(gradient="linear"),
        _base_node_style(),
    ]
    graph.edge_styles = [_base_edge_style(), _base_edge_style()]
    graph.cluster_styles["group"] = _base_cluster_style(fill="#EAF1F8", opacity=0.5)
    cases.append(
        _make_case(
            case_id="cluster_gradient_nodes",
            category="14_cluster_combos",
            title="Cluster fill + gradient nodes",
            graph=graph,
            positions=base_positions,
            options_tested=["cluster_fill=true", "node_gradient=linear"],
            graphviz=None,
        )
    )

    graph, positions = _diamond_dag(["Outer", "Inner L", "Inner R", "Exit"])
    graph.add_cluster("outer", ["n0", "n1", "n2", "n3"], label="Outer")
    graph.add_cluster("inner", ["n1", "n2"], label="Inner", parent="outer")
    graph.node_styles = [
        _base_node_style(shape="roundrect"),
        _base_node_style(shape="diamond"),
        _base_node_style(shape="hexagon"),
        _base_node_style(shape="roundrect"),
    ]
    _set_all_edge_styles(graph, _base_edge_style())
    graph.cluster_styles["outer"] = _base_cluster_style()
    graph.cluster_styles["inner"] = _base_cluster_style(fill="#DCEBFA")
    cases.append(
        _make_case(
            case_id="cluster_nested_shapes",
            category="14_cluster_combos",
            title="Nested cluster + mixed shapes",
            graph=graph,
            positions=positions,
            options_tested=["nested_clusters=2", "mixed_shapes=true"],
            graphviz=_pinned_graphviz_spec(
                positions,
                node_attrs={1: {"shape": "diamond"}, 2: {"shape": "hexagon"}},
            ),
        )
    )

    graph, positions = _diamond_dag(["In", "Inside", "Outside", "Out"])
    graph.add_cluster("group", ["n1", "n2"], label="Boundary")
    graph.edge_styles = [
        _base_edge_style(width=3.0),
        _base_edge_style(width=3.0),
        _base_edge_style(),
        _base_edge_style(),
    ]
    graph.cluster_styles["group"] = _base_cluster_style()
    cases.append(
        _make_case(
            case_id="cluster_thick_crossing",
            category="14_cluster_combos",
            title="Cluster + thick crossing edge",
            graph=graph,
            positions=positions,
            options_tested=["cluster=true", "edge_width=3"],
            graphviz=_pinned_graphviz_spec(
                positions,
                edge_attrs={0: {"penwidth": "3"}, 1: {"penwidth": "3"}},
            ),
        )
    )

    graph, positions = _diamond_dag(["Top", "Left", "Right", "Bottom"])
    graph.add_cluster("group", ["n1", "n2"], label="Middle")
    _set_all_edge_styles(graph, _base_edge_style(routing="ortho"))
    graph.cluster_styles["group"] = _base_cluster_style()
    cases.append(
        _make_case(
            case_id="cluster_ortho",
            category="14_cluster_combos",
            title="Cluster + ortho routing",
            graph=graph,
            positions=positions,
            options_tested=["cluster=true", "routing=ortho"],
            graphviz=_pinned_graphviz_spec(positions, graph_attrs={"splines": "ortho"}),
        )
    )

    graph = DaguaGraph()
    _apply_graph_style(graph)
    graph.add_node("A", label="Very Long Node A")
    graph.add_node("B", label="Very Long Node B")
    graph.add_node("C", label="Exit")
    graph.add_edge("A", "B")
    graph.add_edge("B", "C")
    graph.add_cluster("group", ["A", "B"], label="My Group")
    _set_all_node_styles(graph, _base_node_style())
    _set_all_edge_styles(graph, _base_edge_style())
    graph.cluster_styles["group"] = _base_cluster_style()
    cases.append(
        _make_case(
            case_id="cluster_long_labels",
            category="14_cluster_combos",
            title="Cluster label + long node labels",
            graph=graph,
            positions=base_positions,
            options_tested=["cluster_label='My Group'", "long_node_labels=true"],
            graphviz=_pinned_graphviz_spec(base_positions),
        )
    )

    graph = DaguaGraph()
    _apply_graph_style(graph)
    graph.add_node("A", label="Shadow")
    graph.add_node("B", label="Shadow")
    graph.add_node("C", label="Outside")
    graph.add_edge("A", "B")
    graph.add_edge("B", "C")
    graph.add_cluster("group", ["A", "B"], label="Shadow")
    graph.node_styles = [
        _base_node_style(shadow=True),
        _base_node_style(shadow=True),
        _base_node_style(),
    ]
    _set_all_edge_styles(graph, _base_edge_style())
    graph.cluster_styles["group"] = _base_cluster_style(fill="#EAF1F8")
    cases.append(
        _make_case(
            case_id="cluster_shadow_nodes",
            category="14_cluster_combos",
            title="Cluster fill + shadow nodes",
            graph=graph,
            positions=base_positions,
            options_tested=["cluster_fill=true", "shadow=true"],
            graphviz=None,
        )
    )

    graph = DaguaGraph(direction="LR")
    _apply_graph_style(graph)
    for node_id, label in [("A", "A1"), ("B", "A2"), ("C", "B1"), ("D", "B2")]:
        graph.add_node(node_id, label=label)
    graph.add_edge("A", "B")
    graph.add_edge("C", "D")
    graph.add_cluster("left", ["A", "B"], label="Blue")
    graph.add_cluster("right", ["C", "D"], label="Green")
    side_positions = torch.tensor(
        [[-220.0, 20.0], [-80.0, 20.0], [80.0, 20.0], [220.0, 20.0]], dtype=torch.float32
    )
    _set_all_node_styles(graph, _base_node_style())
    _set_all_edge_styles(graph, _base_edge_style())
    graph.cluster_styles["left"] = _base_cluster_style(fill="#DCEBFA")
    graph.cluster_styles["right"] = _base_cluster_style(fill="#D9F2E2")
    cases.append(
        _make_case(
            case_id="cluster_side_by_side",
            category="14_cluster_combos",
            title="Side-by-side clusters",
            graph=graph,
            positions=side_positions,
            options_tested=["clusters=2", "different_cluster_fills=true"],
            graphviz=_pinned_graphviz_spec(
                side_positions,
                cluster_attrs={
                    "left": {"style": "filled", "fillcolor": "#DCEBFA"},
                    "right": {"style": "filled", "fillcolor": "#D9F2E2"},
                },
            ),
        )
    )

    graph = DaguaGraph()
    _apply_graph_style(graph)
    for node_id, label in [("A", "Outer"), ("B", "Middle"), ("C", "Inner"), ("D", "Leaf")]:
        graph.add_node(node_id, label=label)
    graph.add_edge("A", "B")
    graph.add_edge("B", "C")
    graph.add_edge("C", "D")
    graph.add_cluster("outer", ["A", "B", "C", "D"], label="Outer")
    graph.add_cluster("middle", ["B", "C", "D"], label="Middle", parent="outer")
    graph.add_cluster("inner", ["C", "D"], label="Inner", parent="middle")
    stack_positions = torch.tensor(
        [[0.0, 180.0], [0.0, 60.0], [0.0, -50.0], [0.0, -160.0]], dtype=torch.float32
    )
    _set_all_node_styles(graph, _base_node_style())
    _set_all_edge_styles(graph, _base_edge_style())
    graph.cluster_styles["outer"] = _base_cluster_style()
    graph.cluster_styles["middle"] = _base_cluster_style(fill="#EAF1F8")
    graph.cluster_styles["inner"] = _base_cluster_style(fill="#DCEBFA")
    cases.append(
        _make_case(
            case_id="cluster_three_levels",
            category="14_cluster_combos",
            title="Nested 3-level cluster",
            graph=graph,
            positions=stack_positions,
            options_tested=["nested_clusters=3"],
            graphviz=_pinned_graphviz_spec(stack_positions),
        )
    )

    return cases


def _color_contrast_cases() -> List[AlbumCase]:
    """Build color contrast cases.

    Returns
    -------
    list[AlbumCase]
        Requested ``15_color_contrast`` cases.
    """

    specs = [
        ("contrast_dark_dark", {"fill": "#1A1A1A", "font_color": "#333333"}, True),
        ("contrast_dark_light", {"fill": "#1A1A1A", "font_color": "#FFFFFF"}, True),
        ("contrast_light_light", {"fill": "#F5F5F5", "stroke": "#E0E0E0"}, True),
        ("contrast_bright", {"fill": "#FF0000", "stroke": "#00FF00"}, True),
        ("contrast_pastel", {"fill": "#E8D5E8", "stroke": "#D5E8D5"}, True),
        ("contrast_white_white", {"fill": "#FFFFFF", "stroke": "#FFFFFF"}, True),
        ("contrast_black_gradient", {"fill": "#000000", "gradient": "linear"}, False),
    ]

    cases: List[AlbumCase] = []
    for case_id, overrides, compare in specs:
        graph, positions = _single_node_graph("Sample")
        graph.node_styles[0] = _base_node_style(**overrides)
        cases.append(
            _make_case(
                case_id=case_id,
                category="15_color_contrast",
                title=case_id.replace("_", " ").title(),
                graph=graph,
                positions=positions,
                options_tested=[f"{key}={value}" for key, value in overrides.items()],
                graphviz=_pinned_graphviz_spec(positions) if compare else None,
            )
        )

    graph, positions = _pair_graph(
        _pair_positions(direction="LR"), ["Red", "Green"], direction="LR"
    )
    graph.node_styles = [
        _base_node_style(fill="#FF0000", stroke="#C00000"),
        _base_node_style(fill="#00FF00", stroke="#00C000"),
    ]
    cases.append(
        _make_case(
            case_id="contrast_red_green_pair",
            category="15_color_contrast",
            title="Red/green side by side",
            graph=graph,
            positions=positions,
            options_tested=["left_fill=#FF0000", "right_fill=#00FF00"],
            graphviz=_pinned_graphviz_spec(
                positions,
                node_attrs={
                    0: {"fillcolor": "#FF0000", "color": "#C00000"},
                    1: {"fillcolor": "#00FF00", "color": "#00C000"},
                },
            ),
        )
    )
    return cases


def _dark_mode_graph_style() -> GraphStyle:
    """Return the graph style used for dark-mode cases.

    Returns
    -------
    GraphStyle
        Dark-mode graph style.
    """

    return GraphStyle(
        background_color="#1A1E24",
        margin=8.0,
        edge_label_background="#1A1E24",
        edge_label_background_opacity=0.95,
        min_figsize=(2.0, 1.5),
    )


def _dark_node_style(**overrides: object) -> NodeStyle:
    """Return a light-on-dark node style.

    Parameters
    ----------
    **overrides : object
        Field overrides applied to the base style.

    Returns
    -------
    NodeStyle
        Configured node style.
    """

    style = _base_node_style(fill="#2A3A4A", stroke="#89B5DA", font_color="#E8F2FC")
    for key, value in overrides.items():
        setattr(style, key, value)
    return style


def _dark_edge_style(**overrides: object) -> EdgeStyle:
    """Return a light-on-dark edge style.

    Parameters
    ----------
    **overrides : object
        Field overrides applied to the base style.

    Returns
    -------
    EdgeStyle
        Configured edge style.
    """

    style = _base_edge_style(color="#9EC4E6")
    for key, value in overrides.items():
        setattr(style, key, value)
    return style


def _dark_mode_cases() -> List[AlbumCase]:
    """Build dark-mode cases.

    Returns
    -------
    list[AlbumCase]
        Requested ``16_dark_mode`` cases.
    """

    specs = [
        ("dark_baseline", "baseline"),
        ("dark_gradient", "gradient"),
        ("dark_shadow", "shadow"),
        ("dark_opacity", "opacity"),
        ("dark_dashed", "dashed"),
        ("dark_cluster", "cluster"),
    ]

    cases: List[AlbumCase] = []
    for case_id, variant in specs:
        graph, positions = _chain_graph(3, ["In", "Process", "Out"], direction="LR")
        graph._theme.graph_style = _dark_mode_graph_style()
        graph.node_styles = [_dark_node_style() for _ in range(3)]
        graph.edge_styles = [_dark_edge_style() for _ in range(2)]
        if variant == "gradient":
            graph.node_styles = [_dark_node_style(gradient="linear") for _ in range(3)]
        elif variant == "shadow":
            graph.node_styles = [_dark_node_style(shadow=True) for _ in range(3)]
        elif variant == "opacity":
            graph.node_styles = [_dark_node_style(opacity=0.6) for _ in range(3)]
        elif variant == "dashed":
            graph.edge_styles = [_dark_edge_style(style="dashed") for _ in range(2)]
        elif variant == "cluster":
            graph.add_cluster("group", ["n0", "n1"], label="Cluster")
            graph.cluster_styles["group"] = _base_cluster_style(
                fill="#223244", stroke="#6E93B7", font_color="#E8F2FC"
            )
        cases.append(
            _make_case(
                case_id=case_id,
                category="16_dark_mode",
                title=case_id.replace("_", " ").title(),
                graph=graph,
                positions=positions,
                options_tested=[f"variant={variant}", "background=#1A1E24"],
                graphviz=None,
            )
        )
    return cases


def _extreme_params_cases() -> List[AlbumCase]:
    """Build extreme-parameter cases.

    Returns
    -------
    list[AlbumCase]
        Requested ``17_extreme_params`` cases.
    """

    node_specs = [
        ("extreme_font_6", {"shape": "diamond", "font_size": 6.0}, True),
        ("extreme_font_24", {"shape": "roundrect", "font_size": 24.0}, True),
        ("extreme_stroke_5", {"shape": "roundrect", "stroke_width": 5.0}, True),
        ("extreme_stroke_0_2", {"shape": "diamond", "stroke_width": 0.2}, True),
        ("extreme_radius_25", {"shape": "roundrect", "corner_radius": 25.0}, False),
        ("extreme_radius_0", {"shape": "roundrect", "corner_radius": 0.0}, False),
    ]
    edge_specs = [
        ("extreme_arrow_huge", {"arrow_length": 25.0, "arrow_width": 20.0}, True),
        ("extreme_arrow_tiny", {"arrow_length": 4.0, "arrow_width": 3.0}, True),
        ("extreme_curvature_1", {"curvature": 1.0}, False),
        ("extreme_curvature_0", {"curvature": 0.0}, False),
    ]

    cases: List[AlbumCase] = []
    for case_id, overrides, compare in node_specs:
        graph, positions = _single_node_graph("Extreme")
        graph.node_styles[0] = _base_node_style(**overrides)
        cases.append(
            _make_case(
                case_id=case_id,
                category="17_extreme_params",
                title=case_id.replace("_", " ").title(),
                graph=graph,
                positions=positions,
                options_tested=[f"{key}={value}" for key, value in overrides.items()],
                graphviz=_pinned_graphviz_spec(positions) if compare else None,
            )
        )

    for case_id, overrides, compare in edge_specs:
        graph, positions = _pair_graph(_pair_positions(), ["A", "B"])
        _set_all_edge_styles(graph, _base_edge_style(**overrides))
        cases.append(
            _make_case(
                case_id=case_id,
                category="17_extreme_params",
                title=case_id.replace("_", " ").title(),
                graph=graph,
                positions=positions,
                options_tested=[f"{key}={value}" for key, value in overrides.items()],
                graphviz=_pinned_graphviz_spec(positions) if compare else None,
            )
        )
    return cases


def _dense_mixed_cases() -> List[AlbumCase]:
    """Build dense mixed-style cases.

    Returns
    -------
    list[AlbumCase]
        Requested ``18_dense_mixed`` cases.
    """

    cases: List[AlbumCase] = []

    graph, positions = _mixed_shape_graph(
        ["roundrect", "circle", "diamond", "hexagon", "trapezoid", "star"],
        ["A", "B", "C", "D", "E", "F"],
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
    )
    cases.append(
        _make_case(
            case_id="dense_shapes",
            category="18_dense_mixed",
            title="Mixed shape chain",
            graph=graph,
            positions=positions,
            options_tested=["mixed_shapes=true"],
            graphviz=_pinned_graphviz_spec(positions),
        )
    )

    graph, positions = _chain_graph(
        6, ["A", "B", "C", "D", "E", "F"], direction="LR", spacing=120.0
    )
    fills = ["#DCEBFA", "#F8D6CC", "#D9F2E2", "#FBE7B8", "#E8D8F0", "#F7F2D2"]
    graph.node_styles = [_base_node_style(fill=fill) for fill in fills]
    _set_all_edge_styles(graph, _base_edge_style())
    cases.append(
        _make_case(
            case_id="dense_colors",
            category="18_dense_mixed",
            title="Mixed color chain",
            graph=graph,
            positions=positions,
            options_tested=["mixed_fills=true"],
            graphviz=_pinned_graphviz_spec(positions),
        )
    )

    graph, positions = _diamond_dag()
    graph.edge_styles = [
        _base_edge_style(style="solid"),
        _base_edge_style(style="dashed"),
        _base_edge_style(style="dotted"),
        _base_edge_style(width=3.0),
    ]
    cases.append(
        _make_case(
            case_id="dense_edge_styles",
            category="18_dense_mixed",
            title="Diamond DAG mixed edge styles",
            graph=graph,
            positions=positions,
            options_tested=["mixed_edge_styles=true"],
            graphviz=_pinned_graphviz_spec(positions),
        )
    )

    graph, positions = _diamond_dag()
    graph.edge_styles = [
        _base_edge_style(arrow="normal"),
        _base_edge_style(arrow="vee"),
        _base_edge_style(arrow="diamond"),
        _base_edge_style(arrow="crow"),
    ]
    cases.append(
        _make_case(
            case_id="dense_edge_arrows",
            category="18_dense_mixed",
            title="Diamond DAG mixed arrows",
            graph=graph,
            positions=positions,
            options_tested=["mixed_arrows=true"],
            graphviz=_pinned_graphviz_spec(positions),
        )
    )

    graph, positions = _mixed_shape_graph(
        [
            "roundrect",
            "circle",
            "diamond",
            "hexagon",
            "trapezoid",
            "roundrect",
            "circle",
            "diamond",
        ],
        ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"],
        [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (1, 5), (2, 6)],
    )
    graph.add_cluster("left", ["n0", "n1", "n2", "n3"], label="Left")
    graph.add_cluster("right", ["n4", "n5", "n6", "n7"], label="Right")
    graph.cluster_styles["left"] = _base_cluster_style()
    graph.cluster_styles["right"] = _base_cluster_style(fill="#EAF1F8")
    _set_all_edge_styles(graph, _base_edge_style())
    cases.append(
        _make_case(
            case_id="dense_clusters",
            category="18_dense_mixed",
            title="Dense clustered mixed shapes",
            graph=graph,
            positions=positions,
            options_tested=["clusters=2", "mixed_shapes=true"],
            graphviz=_pinned_graphviz_spec(positions),
        )
    )

    graph, positions = _chain_graph(
        6, ["N0", "N1", "N2", "N3", "N4", "N5"], direction="LR", spacing=120.0
    )
    graph.node_styles = [
        _base_node_style(gradient="linear"),
        _base_node_style(gradient="radial"),
        _base_node_style(gradient="linear"),
        _base_node_style(shadow=True),
        _base_node_style(shadow=True),
        _base_node_style(shadow=True),
    ]
    _set_all_edge_styles(graph, _base_edge_style())
    cases.append(
        _make_case(
            case_id="dense_gradient_shadow",
            category="18_dense_mixed",
            title="Gradient and shadow mix",
            graph=graph,
            positions=positions,
            options_tested=["gradients=3", "shadows=3"],
            graphviz=None,
        )
    )

    graph, positions = _fan_graph("Hub", ["L1", "L2", "L3", "L4"])
    graph.node_styles = [
        _base_node_style(),
        _base_node_style(shape="circle", fill="#DCEBFA"),
        _base_node_style(shape="diamond", fill="#F8D6CC"),
        _base_node_style(shape="hexagon", fill="#D9F2E2"),
        _base_node_style(shape="trapezoid", fill="#FBE7B8"),
    ]
    _set_all_edge_styles(graph, _base_edge_style())
    cases.append(
        _make_case(
            case_id="dense_fan_out",
            category="18_dense_mixed",
            title="Fan-out mixed leaves",
            graph=graph,
            positions=positions,
            options_tested=["fan_out=true", "mixed_leaf_shapes=true"],
            graphviz=_pinned_graphviz_spec(positions),
        )
    )

    graph, positions = _grid_graph(3, 3)
    graph.node_styles = []
    for index in range(9):
        graph.node_styles.append(
            _base_node_style(
                shape="rect" if index % 2 == 0 else "circle",
                stroke_dash="solid" if index % 3 == 0 else "dashed",
            )
        )
    _set_all_edge_styles(graph, _base_edge_style())
    cases.append(
        _make_case(
            case_id="dense_grid",
            category="18_dense_mixed",
            title="Alternating grid",
            graph=graph,
            positions=positions,
            options_tested=["grid=3x3", "alternating_shapes=true"],
            graphviz=_pinned_graphviz_spec(positions),
        )
    )
    return cases


def _real_world_pattern_cases() -> List[AlbumCase]:
    """Build real-world pattern cases.

    Returns
    -------
    list[AlbumCase]
        Requested ``19_real_world_patterns`` cases.
    """

    cases: List[AlbumCase] = []

    graph, positions = _chain_graph(
        5, ["Input", "Parse", "Transform", "Validate", "Output"], direction="LR", spacing=130.0
    )
    _set_all_node_styles(graph, _base_node_style(shape="roundrect"))
    graph.edge_styles = [
        _base_edge_style(routing="ortho"),
        _base_edge_style(routing="ortho", style="dashed"),
        _base_edge_style(routing="ortho"),
        _base_edge_style(routing="ortho"),
    ]
    _set_all_edge_labels(graph, ["data", "transform", "validate", "output"])
    graph.add_cluster("core", ["n1", "n2", "n3"], label="Core")
    graph.cluster_styles["core"] = _base_cluster_style()
    cases.append(
        _make_case(
            case_id="real_pipeline",
            category="19_real_world_patterns",
            title="Pipeline / workflow",
            graph=graph,
            positions=positions,
            options_tested=["direction=LR", "routing=ortho", "cluster=true", "edge_labels=true"],
            graphviz=_pinned_graphviz_spec(
                positions, graph_attrs={"rankdir": "LR", "splines": "ortho"}
            ),
        )
    )

    graph = DaguaGraph(direction="LR")
    _apply_graph_style(graph)
    for node_id, label in [
        ("start", "Start"),
        ("idle", "Idle"),
        ("active", "Active"),
        ("processing", "Processing"),
        ("done", "Done"),
    ]:
        graph.add_node(node_id, label=label)
    for source, target, label in [
        ("start", "idle", "boot"),
        ("idle", "active", "activate"),
        ("active", "idle", "deactivate"),
        ("active", "processing", "run"),
        ("processing", "processing", "retry"),
        ("processing", "done", "finish"),
    ]:
        graph.add_edge(source, target, label=label)
    positions = torch.tensor(
        [[-300.0, 0.0], [-150.0, 0.0], [0.0, 0.0], [160.0, 0.0], [320.0, 0.0]], dtype=torch.float32
    )
    graph.node_styles = [
        _base_node_style(
            shape="circle", min_width=24.0, fill="#1F2937", stroke="#1F2937", font_color="#FFFFFF"
        ),
        _base_node_style(shape="circle"),
        _base_node_style(shape="circle"),
        _base_node_style(shape="circle"),
        _base_node_style(shape="circle"),
    ]
    graph.edge_styles = [_base_edge_style() for _ in range(6)]
    cases.append(
        _make_case(
            case_id="real_state_machine",
            category="19_real_world_patterns",
            title="State machine",
            graph=graph,
            positions=positions,
            options_tested=[
                "direction=LR",
                "self_loop=true",
                "parallel_edges=true",
                "edge_labels=true",
            ],
            graphviz=_pinned_graphviz_spec(positions, graph_attrs={"rankdir": "LR"}),
        )
    )

    graph = DaguaGraph(direction="TB")
    _apply_graph_style(graph)
    for node_id, label in [
        ("start", "Start"),
        ("decision", "Valid?"),
        ("process", "Process"),
        ("error", "Error"),
        ("end", "End"),
    ]:
        graph.add_node(node_id, label=label)
    for source, target, label in [
        ("start", "decision", None),
        ("decision", "process", "yes"),
        ("decision", "error", "no"),
        ("process", "end", None),
        ("error", "end", "error"),
    ]:
        graph.add_edge(source, target, label=label)
    positions = torch.tensor(
        [[0.0, 220.0], [0.0, 80.0], [-140.0, -40.0], [140.0, -40.0], [0.0, -180.0]],
        dtype=torch.float32,
    )
    graph.node_styles = [
        _base_node_style(shape="roundrect"),
        _base_node_style(shape="diamond"),
        _base_node_style(shape="rect"),
        _base_node_style(shape="rect"),
        _base_node_style(shape="roundrect"),
    ]
    graph.edge_styles = [
        _base_edge_style(),
        _base_edge_style(),
        _base_edge_style(style="dashed"),
        _base_edge_style(),
        _base_edge_style(style="dashed"),
    ]
    cases.append(
        _make_case(
            case_id="real_flowchart",
            category="19_real_world_patterns",
            title="Flowchart",
            graph=graph,
            positions=positions,
            options_tested=["diamond=true", "edge_labels=yes/no", "dashed_error=true"],
            graphviz=_pinned_graphviz_spec(positions),
        )
    )

    graph = DaguaGraph(direction="TB")
    _apply_graph_style(graph)
    for index, label in enumerate(
        ["Input", "Enc 1", "Enc 2", "Bridge", "Dec 1", "Dec 2", "Output", "Skip"]
    ):
        graph.add_node(f"n{index}", label=label)
    for edge in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (0, 3), (2, 5), (1, 7), (7, 5)]:
        graph.add_edge(f"n{edge[0]}", f"n{edge[1]}")
    positions = torch.tensor(
        [
            [0.0, 300.0],
            [0.0, 210.0],
            [0.0, 120.0],
            [0.0, 20.0],
            [0.0, -80.0],
            [0.0, -170.0],
            [0.0, -260.0],
            [180.0, -40.0],
        ],
        dtype=torch.float32,
    )
    graph.node_styles = [_base_node_style(font_size=10.0) for _ in range(8)]
    graph.edge_styles = [_base_edge_style() for _ in range(10)]
    graph.add_cluster("encoder", ["n0", "n1", "n2"], label="Encoder")
    graph.add_cluster("decoder", ["n4", "n5", "n6"], label="Decoder")
    graph.cluster_styles["encoder"] = _base_cluster_style()
    graph.cluster_styles["decoder"] = _base_cluster_style(fill="#DCEBFA")
    cases.append(
        _make_case(
            case_id="real_neural_net",
            category="19_real_world_patterns",
            title="Neural network",
            graph=graph,
            positions=positions,
            options_tested=["skip_connections=true", "clusters=encoder_decoder"],
            graphviz=_pinned_graphviz_spec(positions),
        )
    )

    graph = DaguaGraph(direction="TB")
    _apply_graph_style(graph)
    for index, label in enumerate(
        ["CEO", "Ops", "Sales", "Tech", "Ops Rep", "Sales Rep", "Tech Rep"]
    ):
        graph.add_node(f"n{index}", label=label)
    for edge in [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 2)]:
        graph.add_edge(f"n{edge[0]}", f"n{edge[1]}")
    positions = torch.tensor(
        [
            [0.0, 240.0],
            [-220.0, 80.0],
            [0.0, 80.0],
            [220.0, 80.0],
            [-220.0, -100.0],
            [0.0, -100.0],
            [220.0, -100.0],
        ],
        dtype=torch.float32,
    )
    graph.node_styles = [_base_node_style(shape="roundrect") for _ in range(7)]
    graph.edge_styles = [
        _base_edge_style(width=2.4),
        _base_edge_style(width=2.4),
        _base_edge_style(width=2.4),
        _base_edge_style(width=1.4),
        _base_edge_style(width=1.4),
        _base_edge_style(width=1.4),
        _base_edge_style(width=1.0, style="dashed"),
    ]
    graph.add_cluster("ops", ["n1", "n4"], label="Ops")
    graph.add_cluster("sales", ["n2", "n5"], label="Sales")
    graph.add_cluster("tech", ["n3", "n6"], label="Tech")
    graph.cluster_styles["ops"] = _base_cluster_style()
    graph.cluster_styles["sales"] = _base_cluster_style(fill="#EAF1F8")
    graph.cluster_styles["tech"] = _base_cluster_style(fill="#FBE7B8")
    cases.append(
        _make_case(
            case_id="real_org_chart",
            category="19_real_world_patterns",
            title="Organizational chart",
            graph=graph,
            positions=positions,
            options_tested=["tree=true", "dashed_reporting=true", "clusters=departments"],
            graphviz=_pinned_graphviz_spec(positions),
        )
    )

    graph = DaguaGraph(direction="LR")
    _apply_graph_style(graph)
    shapes = ["rect", "rect", "cylinder", "rect", "ellipse", "rect"]
    labels = ["API", "ETL", "DB", "Cache", "Client", "Queue"]
    for index, label in enumerate(labels):
        graph.add_node(f"n{index}", label=label)
    for edge in [(0, 1), (1, 2), (1, 3), (4, 0), (2, 5), (5, 3)]:
        graph.add_edge(f"n{edge[0]}", f"n{edge[1]}")
    positions = torch.tensor(
        [[-250.0, 0.0], [-90.0, 0.0], [90.0, 80.0], [90.0, -80.0], [-410.0, 0.0], [250.0, 0.0]],
        dtype=torch.float32,
    )
    graph.node_styles = [_base_node_style(shape=shape) for shape in shapes]
    graph.edge_styles = [_base_edge_style(routing="ortho") for _ in range(6)]
    _set_all_edge_labels(graph, ["request", "write", "warm", "input", "events", "invalidate"])
    graph.add_cluster("backend", ["n0", "n1", "n2", "n3", "n5"], label="Backend")
    graph.cluster_styles["backend"] = _base_cluster_style()
    cases.append(
        _make_case(
            case_id="real_data_flow",
            category="19_real_world_patterns",
            title="Data flow",
            graph=graph,
            positions=positions,
            options_tested=[
                "mixed_shapes=true",
                "routing=ortho",
                "edge_labels=true",
                "cluster=backend",
            ],
            graphviz=_pinned_graphviz_spec(
                positions, graph_attrs={"rankdir": "LR", "splines": "ortho"}
            ),
        )
    )

    return cases


def _kitchen_sink_cases() -> List[AlbumCase]:
    """Build kitchen-sink stress cases.

    Returns
    -------
    list[AlbumCase]
        Requested ``20_kitchen_sink`` cases.
    """

    cases: List[AlbumCase] = []

    graph, positions = _single_node_graph("Diamond")
    graph.node_styles[0] = _base_node_style(
        shape="diamond",
        stroke_dash="solid",
        gradient="linear",
        shadow=True,
    )
    cases.append(
        _make_case(
            case_id="kitchen_diamond",
            category="20_kitchen_sink",
            title="Diamond + dashed + gradient + shadow",
            graph=graph,
            positions=positions,
            options_tested=["shape=diamond", "dashed=true", "gradient=linear", "shadow=true"],
            graphviz=None,
        )
    )

    graph, positions = _pair_graph(_pair_positions(), ["Bold", "Target"])
    graph.node_styles = [
        _base_node_style(shape="circle", font_weight="bold", font_style="italic") for _ in range(2)
    ]
    graph.edge_styles = [_base_edge_style(arrow="vee", style="dotted")]
    cases.append(
        _make_case(
            case_id="kitchen_circle_vee",
            category="20_kitchen_sink",
            title="Circle + bold italic + vee + dotted",
            graph=graph,
            positions=positions,
            options_tested=["shape=circle", "font=bold_italic", "arrow=vee", "edge_style=dotted"],
            graphviz=_pinned_graphviz_spec(positions),
        )
    )

    graph, positions = _single_node_graph("Hex")
    graph.node_styles[0] = _base_node_style(
        shape="hexagon", gradient="radial", opacity=0.7, shadow=True
    )
    cases.append(
        _make_case(
            case_id="kitchen_hexagon",
            category="20_kitchen_sink",
            title="Hexagon + radial + opacity + shadow",
            graph=graph,
            positions=positions,
            options_tested=["shape=hexagon", "gradient=radial", "opacity=0.7", "shadow=true"],
            graphviz=None,
        )
    )

    graph, positions = _single_node_graph("Star")
    graph.node_styles[0] = _base_node_style(
        shape="star", stroke_width=3.0, stroke_dash="dotted", font_style="italic"
    )
    cases.append(
        _make_case(
            case_id="kitchen_star",
            category="20_kitchen_sink",
            title="Star + thick dotted + italic",
            graph=graph,
            positions=positions,
            options_tested=[
                "shape=star",
                "stroke_width=3",
                "stroke_dash=dotted",
                "font_style=italic",
            ],
            graphviz=_pinned_graphviz_spec(positions),
        )
    )

    graph, positions = _diamond_dag()
    graph.add_cluster("group", ["n1", "n2"], label="Group")
    graph.node_styles = [_base_node_style(shape="diamond") for _ in range(4)]
    graph.edge_styles = [_base_edge_style(style="dashed", routing="ortho") for _ in range(4)]
    graph.cluster_styles["group"] = _base_cluster_style()
    cases.append(
        _make_case(
            case_id="kitchen_cluster_diamond",
            category="20_kitchen_sink",
            title="Cluster + diamond nodes + dashed ortho",
            graph=graph,
            positions=positions,
            options_tested=["cluster=true", "diamond_nodes=true", "dashed=true", "routing=ortho"],
            graphviz=_pinned_graphviz_spec(positions, graph_attrs={"splines": "ortho"}),
        )
    )

    graph, positions = _chain_graph(3, ["Input", "Work", "Output"], direction="LR")
    graph.node_styles = [_base_node_style(shape="trapezoid") for _ in range(3)]
    graph.edge_styles = [_base_edge_style(arrow="crow", width=3.0) for _ in range(2)]
    cases.append(
        _make_case(
            case_id="kitchen_lr_trapezoid",
            category="20_kitchen_sink",
            title="LR + trapezoid + crow + thick",
            graph=graph,
            positions=positions,
            options_tested=["direction=LR", "shape=trapezoid", "arrow=crow", "width=3"],
            graphviz=_pinned_graphviz_spec(positions, graph_attrs={"rankdir": "LR"}),
        )
    )

    graph, positions = _diamond_dag()
    graph._theme.graph_style = _dark_mode_graph_style()
    graph.add_cluster("dark", ["n1", "n2"], label="Dark")
    graph.node_styles = [_dark_node_style(gradient="linear", shadow=True) for _ in range(4)]
    graph.edge_styles = [_dark_edge_style(style="dashed") for _ in range(4)]
    graph.cluster_styles["dark"] = _base_cluster_style(
        fill="#223244", stroke="#6E93B7", font_color="#E8F2FC"
    )
    cases.append(
        _make_case(
            case_id="kitchen_dark_cluster",
            category="20_kitchen_sink",
            title="Dark bg + gradient + shadow + dashed + cluster",
            graph=graph,
            positions=positions,
            options_tested=[
                "dark_mode=true",
                "gradient=true",
                "shadow=true",
                "dashed=true",
                "cluster=true",
            ],
            graphviz=None,
        )
    )

    graph, positions = _pair_graph(_pair_positions(gap=50.0), ["Near", "Nearer"])
    graph.edge_styles = [_base_edge_style(style="dashed", tail_arrow="dot")]
    _set_all_edge_labels(graph, ["flow"])
    cases.append(
        _make_case(
            case_id="kitchen_short_dense",
            category="20_kitchen_sink",
            title="Short edge + head tail + label + dashed",
            graph=graph,
            positions=positions,
            options_tested=["gap=50", "tail_arrow=dot", "label='flow'", "style=dashed"],
            graphviz=_pinned_graphviz_spec(positions),
        )
    )

    graph = DaguaGraph()
    _apply_graph_style(graph)
    for node_id, label in [("A", "Outer"), ("B", "Inner"), ("C", "Leaf"), ("D", "Exit")]:
        graph.add_node(node_id, label=label)
    graph.add_edge("A", "B", label="enter")
    graph.add_edge("B", "C", label="branch")
    graph.add_edge("C", "D", label="ship")
    graph.add_cluster("outer", ["A", "B", "C", "D"], label="Outer")
    graph.add_cluster("inner", ["B", "C"], label="Inner", parent="outer")
    stack_positions = torch.tensor(
        [[0.0, 180.0], [0.0, 60.0], [0.0, -50.0], [0.0, -160.0]], dtype=torch.float32
    )
    graph.node_styles = [
        _base_node_style(shape="roundrect"),
        _base_node_style(shape="diamond", gradient="linear"),
        _base_node_style(shape="hexagon", gradient="radial"),
        _base_node_style(shape="circle"),
    ]
    graph.edge_styles = [_base_edge_style() for _ in range(3)]
    graph.cluster_styles["outer"] = _base_cluster_style()
    graph.cluster_styles["inner"] = _base_cluster_style(fill="#DCEBFA")
    cases.append(
        _make_case(
            case_id="kitchen_nested_gradient",
            category="20_kitchen_sink",
            title="Nested cluster + mixed shapes + gradient + labels",
            graph=graph,
            positions=stack_positions,
            options_tested=[
                "nested_clusters=true",
                "mixed_shapes=true",
                "gradient=true",
                "edge_labels=true",
            ],
            graphviz=None,
        )
    )

    graph, positions = _pair_graph(_pair_positions(), ["Source", "Sink"])
    graph.edge_styles = [
        _base_edge_style(style="dotted", width=3.0, arrow="diamond", arrow_length=22.0)
    ]
    _set_all_edge_labels(graph, ["Important"])
    cases.append(
        _make_case(
            case_id="kitchen_thick_dotted",
            category="20_kitchen_sink",
            title="Thick dotted + large diamond + label",
            graph=graph,
            positions=positions,
            options_tested=["style=dotted", "width=3", "arrow=diamond", "label='Important'"],
            graphviz=_pinned_graphviz_spec(positions),
        )
    )

    graph, positions = _parallel_graph(3)
    graph.edge_styles = [
        _base_edge_style(style="solid", arrow="normal"),
        _base_edge_style(style="dashed", arrow="vee"),
        _base_edge_style(style="dotted", arrow="diamond"),
    ]
    cases.append(
        _make_case(
            case_id="kitchen_parallel",
            category="20_kitchen_sink",
            title="Parallel mixed edges",
            graph=graph,
            positions=positions,
            options_tested=["parallel_edges=3", "mixed_styles=true", "mixed_arrows=true"],
            graphviz=_pinned_graphviz_spec(positions, graph_attrs={"splines": "true"}),
        )
    )

    graph, positions = _self_loop_plus_normal_graph()
    graph.node_styles[0] = _base_node_style(shadow=True)
    graph.edge_styles[0] = _base_edge_style(style="dashed", arrow="vee")
    graph.edge_styles[1] = _base_edge_style()
    _set_all_edge_labels(graph, ["retry", None])
    cases.append(
        _make_case(
            case_id="kitchen_selfloop",
            category="20_kitchen_sink",
            title="Self-loop + dashed + vee + label + shadow",
            graph=graph,
            positions=positions,
            options_tested=["self_loop=true", "style=dashed", "arrow=vee", "shadow=true"],
            graphviz=None,
        )
    )

    return cases


def _legacy_build_case_catalog() -> List[AlbumCase]:
    """Build the legacy partial combo album case catalog.

    Returns
    -------
    list[AlbumCase]
        All album cases in output order.
    """

    builders: Sequence[CategoryBuilder] = (
        _shape_x_border_cases,
        _shape_x_gradient_cases,
        _arrow_x_edgestyle_cases,
        _arrow_x_routing_cases,
        _arrow_proportions_cases,
        _arrow_head_tail_cases,
        _text_overflow_cases,
        _edge_label_cases,
        _short_edge_cases,
        _self_loops_parallel_cases,
        _opacity_interaction_cases,
        _shadow_interaction_cases,
        _direction_x_routing_cases,
        _cluster_combos_cases,
        _color_contrast_cases,
        _dark_mode_cases,
        _extreme_params_cases,
        _dense_mixed_cases,
        _real_world_pattern_cases,
        _kitchen_sink_cases,
    )
    cases: List[AlbumCase] = []
    for builder in builders:
        cases.extend(builder())

    seen_outputs: set[Tuple[str, str]] = set()
    for case in cases:
        output_key = (case.category, case.filename)
        if output_key in seen_outputs:
            raise ValueError(
                f"Duplicate combo album output target: {case.category}/{case.filename}"
            )
        seen_outputs.add(output_key)
    return cases


def _graphviz_shape_attrs(shape: str, stroke_dash: str = "solid") -> Dict[str, str]:
    """Return Graphviz node attributes that approximate a Dagua shape.

    Parameters
    ----------
    shape : str
        Dagua node shape name.
    stroke_dash : str, default="solid"
        Requested node border dash pattern.

    Returns
    -------
    dict[str, str]
        Per-node Graphviz attribute overrides.
    """

    mapped_shape = {"rect": "box", "roundrect": "box", "trapezoid": "trapezium"}.get(shape, shape)
    attrs: Dict[str, str] = {"shape": mapped_shape}
    style_tokens = ["filled"]
    if shape == "roundrect":
        style_tokens.append("rounded")
    if stroke_dash in {"dashed", "dotted"}:
        style_tokens.append(stroke_dash)
    if style_tokens != ["filled"]:
        attrs["style"] = ",".join(style_tokens)
    return attrs


def _graphviz_splines_for_routing(routing: str) -> str:
    """Return the Graphviz ``splines`` value for a Dagua routing mode.

    Parameters
    ----------
    routing : str
        Dagua routing mode.

    Returns
    -------
    str
        Graphviz ``splines`` attribute value.
    """

    return {"bezier": "true", "ortho": "ortho", "straight": "false"}[routing]


def _cluster_graph_custom(
    node_specs: Sequence[Tuple[str, str, Mapping[str, object]]],
    edges: Sequence[Tuple[str, str]],
    clusters: Sequence[Tuple[str, str, Sequence[str], Mapping[str, object]]],
    positions: Optional[Mapping[str, Tuple[float, float]]] = None,
    direction: str = "TB",
    cluster_parents: Optional[Mapping[str, str]] = None,
) -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a clustered graph from explicit node, edge, and cluster specs.

    Parameters
    ----------
    node_specs : sequence[tuple[str, str, Mapping[str, object]]]
        Tuples of ``(node_id, label, node_style_overrides)`` in node order.
    edges : sequence[tuple[str, str]]
        Directed edges by node ID.
    clusters : sequence[tuple[str, str, sequence[str], Mapping[str, object]]]
        Tuples of ``(cluster_name, label, member_ids, cluster_style_overrides)``.
    positions : Mapping[str, tuple[float, float]] | None, default=None
        Optional node position overrides keyed by node ID.
    direction : str, default="TB"
        Graph direction field used by routing.
    cluster_parents : Mapping[str, str] | None, default=None
        Optional parent cluster mapping keyed by cluster name.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Configured graph and node position tensor with shape ``[N, 2]``.
    """

    graph = DaguaGraph(direction=direction)
    _apply_graph_style(graph)

    node_styles: List[NodeStyle] = []
    ordered_positions: List[Tuple[float, float]] = []
    for index, (node_id, label, style_overrides) in enumerate(node_specs):
        graph.add_node(node_id, label=label)
        node_styles.append(_base_node_style(**dict(style_overrides)))
        if positions is None:
            ordered_positions.append((0.0, -140.0 * index))
        else:
            ordered_positions.append(positions[node_id])
    graph.node_styles = node_styles

    for source, target in edges:
        graph.add_edge(source, target)
    _set_all_edge_styles(graph, _base_edge_style())

    for cluster_name, label, member_ids, style_overrides in clusters:
        graph.add_cluster(
            cluster_name,
            list(member_ids),
            label=label,
            parent=None if cluster_parents is None else cluster_parents.get(cluster_name),
        )
        graph.cluster_styles[cluster_name] = _base_cluster_style(**dict(style_overrides))

    return graph, torch.tensor(ordered_positions, dtype=torch.float32)


def _cat11_opacity_interactions_cases() -> List[AlbumCase]:
    """Build category 11 opacity interaction cases.

    Returns
    -------
    list[AlbumCase]
        Requested ``11_opacity_interactions`` cases.
    """

    node_specs = [
        ("opacity_gradient", {"opacity": 0.4, "gradient": "linear"}),
        ("opacity_shadow", {"opacity": 0.4, "shadow": True}),
        ("opacity_dashed_border", {"opacity": 0.3, "stroke_dash": "dashed"}),
    ]
    edge_specs = [
        ("edge_opacity_dotted", {"opacity": 0.3, "style": "dotted"}),
        ("edge_opacity_diamond", {"opacity": 0.3, "arrow": "diamond"}),
    ]

    cases: List[AlbumCase] = []
    for case_id, node_overrides in node_specs:
        graph, positions = _pair_graph(_pair_positions(), ["Source", "Target"])
        _set_all_node_styles(graph, _base_node_style(**node_overrides))
        _set_all_edge_styles(graph, _base_edge_style())
        cases.append(
            _make_case(
                case_id=case_id,
                category="11_opacity_interactions",
                title=case_id.replace("_", " ").title(),
                graph=graph,
                positions=positions,
                options_tested=[f"node.{key}={value}" for key, value in node_overrides.items()],
                graphviz=None,
            )
        )

    for case_id, edge_overrides in edge_specs:
        graph, positions = _pair_graph(_pair_positions(), ["Near", "Far"])
        _set_all_node_styles(graph, _base_node_style())
        graph.edge_styles = [_base_edge_style(**edge_overrides) for _ in range(2)]
        cases.append(
            _make_case(
                case_id=case_id,
                category="11_opacity_interactions",
                title=case_id.replace("_", " ").title(),
                graph=graph,
                positions=positions,
                options_tested=[f"edge.{key}={value}" for key, value in edge_overrides.items()],
                graphviz=None,
            )
        )

    graph, positions = _pair_graph(_pair_positions(), ["Near", "Far"])
    _set_all_node_styles(graph, _base_node_style(opacity=0.5))
    _set_all_edge_styles(graph, _base_edge_style(opacity=0.5))
    cases.append(
        _make_case(
            case_id="both_faded",
            category="11_opacity_interactions",
            title="Both Faded",
            graph=graph,
            positions=positions,
            options_tested=["node.opacity=0.5", "edge.opacity=0.5"],
            graphviz=None,
        )
    )
    return cases


def _cat12_shadow_interactions_cases() -> List[AlbumCase]:
    """Build category 12 shadow interaction cases.

    Returns
    -------
    list[AlbumCase]
        Requested ``12_shadow_interactions`` cases.
    """

    specs = [
        ("shadow_dashed", {"shadow": True, "stroke_dash": "dashed"}),
        ("shadow_gradient", {"shadow": True, "gradient": "linear"}),
        ("shadow_large_radius", {"shadow": True, "corner_radius": 20.0}),
        ("shadow_circle", {"shadow": True, "shape": "circle"}),
        ("shadow_star", {"shadow": True, "shape": "star"}),
        ("shadow_opacity", {"shadow": True, "opacity": 0.5}),
    ]

    cases: List[AlbumCase] = []
    for case_id, node_overrides in specs:
        graph, positions = _single_node_graph(case_id.replace("_", "\n"))
        graph.node_styles[0] = _base_node_style(**node_overrides)
        cases.append(
            _make_case(
                case_id=case_id,
                category="12_shadow_interactions",
                title=case_id.replace("_", " ").title(),
                graph=graph,
                positions=positions,
                options_tested=[f"node.{key}={value}" for key, value in node_overrides.items()],
                graphviz=None,
            )
        )
    return cases


def _cat13_direction_x_routing_cases() -> List[AlbumCase]:
    """Build category 13 direction-versus-routing cases.

    Returns
    -------
    list[AlbumCase]
        Requested ``13_direction_x_routing`` cases.
    """

    specs = [
        ("lr_ortho_normal", "LR", "ortho", "normal", None, None),
        ("lr_ortho_vee", "LR", "ortho", "vee", None, None),
        ("rl_bezier_diamond", "RL", "bezier", "diamond", None, None),
        ("bt_ortho_normal", "BT", "ortho", "normal", None, None),
        ("bt_straight_tee", "BT", "straight", "tee", None, None),
        ("lr_bezier_label", "LR", "bezier", "normal", ("step 1", "step 2"), None),
        ("rl_ortho_label", "RL", "ortho", "normal", ("step 1", "step 2"), None),
        ("lr_dashed", "LR", "bezier", "normal", None, "dashed"),
    ]

    cases: List[AlbumCase] = []
    for case_id, direction, routing, arrow, labels, edge_style in specs:
        graph, positions = _chain_graph(3, ["Start", "Middle", "End"], direction=direction)
        graph.edge_styles = [
            _base_edge_style(arrow=arrow, routing=routing, style=edge_style or "solid"),
            _base_edge_style(arrow=arrow, routing=routing, style=edge_style or "solid"),
        ]
        if labels is not None:
            _set_all_edge_labels(graph, list(labels))

        gv_edge_attrs: Dict[str, str] = {"arrowhead": arrow}
        if edge_style is not None:
            gv_edge_attrs["style"] = edge_style

        cases.append(
            _make_case(
                case_id=case_id,
                category="13_direction_x_routing",
                title=case_id.replace("_", " ").title(),
                graph=graph,
                positions=positions,
                options_tested=[
                    f"direction={direction}",
                    f"routing={routing}",
                    f"arrow={arrow}",
                    *([] if labels is None else ["labels=step 1,step 2"]),
                    *([] if edge_style is None else [f"style={edge_style}"]),
                ],
                graphviz=_pinned_graphviz_spec(
                    positions,
                    graph_attrs={
                        "rankdir": direction,
                        "splines": _graphviz_splines_for_routing(routing),
                    },
                    default_edge_attrs=gv_edge_attrs,
                ),
            )
        )
    return cases


def _cat14_cluster_combos_cases() -> List[AlbumCase]:
    """Build category 14 cluster combination cases.

    Returns
    -------
    list[AlbumCase]
        Requested ``14_cluster_combos`` cases.
    """

    cases: List[AlbumCase] = []

    base_positions = {"a": (-140.0, 30.0), "b": (0.0, 30.0), "c": (140.0, -120.0)}
    graph, positions = _cluster_graph_custom(
        [
            ("a", "One", {"stroke_dash": "dashed"}),
            ("b", "Two", {"stroke_dash": "dashed"}),
            ("c", "Three", {}),
        ],
        [("a", "b"), ("b", "c")],
        [("alpha", "Alpha", ["a", "b"], {"stroke_dash": "dashed"})],
        positions=base_positions,
    )
    cases.append(
        _make_case(
            case_id="dashed_cluster_dashed_nodes",
            category="14_cluster_combos",
            title="Dashed Cluster Dashed Nodes",
            graph=graph,
            positions=positions,
            options_tested=["cluster.stroke_dash=dashed", "node.stroke_dash=dashed"],
            graphviz=_pinned_graphviz_spec(
                positions,
                cluster_attrs={"alpha": {"style": "dashed"}},
                node_attrs={
                    0: _graphviz_shape_attrs("roundrect", "dashed"),
                    1: _graphviz_shape_attrs("roundrect", "dashed"),
                },
            ),
        )
    )

    graph, positions = _cluster_graph_custom(
        [("a", "One", {}), ("b", "Two", {}), ("c", "Three", {})],
        [("a", "b"), ("b", "c")],
        [("alpha", "Alpha", ["a", "b"], {"stroke_dash": "dashed"})],
        positions=base_positions,
    )
    cases.append(
        _make_case(
            case_id="dashed_cluster_solid_nodes",
            category="14_cluster_combos",
            title="Dashed Cluster Solid Nodes",
            graph=graph,
            positions=positions,
            options_tested=["cluster.stroke_dash=dashed", "node.stroke_dash=solid"],
            graphviz=_pinned_graphviz_spec(positions, cluster_attrs={"alpha": {"style": "dashed"}}),
        )
    )

    graph, positions = _cluster_graph_custom(
        [
            ("a", "Gradient A", {"gradient": "linear"}),
            ("b", "Gradient B", {"gradient": "linear"}),
            ("c", "Outside", {}),
        ],
        [("a", "b"), ("b", "c")],
        [("alpha", "Gradient Cluster", ["a", "b"], {})],
        positions=base_positions,
    )
    cases.append(
        _make_case(
            case_id="cluster_gradient_nodes",
            category="14_cluster_combos",
            title="Cluster Gradient Nodes",
            graph=graph,
            positions=positions,
            options_tested=["cluster=true", "node.gradient=linear"],
            graphviz=None,
        )
    )

    nested_positions = {
        "entry": (0.0, 170.0),
        "left": (-120.0, 20.0),
        "right": (120.0, 20.0),
        "exit": (0.0, -140.0),
    }
    graph, positions = _cluster_graph_custom(
        [
            ("entry", "Entry", {"shape": "roundrect"}),
            ("left", "Inner L", {"shape": "diamond"}),
            ("right", "Inner R", {"shape": "hexagon"}),
            ("exit", "Exit", {"shape": "circle"}),
        ],
        [("entry", "left"), ("entry", "right"), ("left", "exit"), ("right", "exit")],
        [
            ("outer", "Outer", ["entry", "left", "right", "exit"], {}),
            ("inner", "Inner", ["left", "right"], {"fill": "#DCEBFA"}),
        ],
        positions=nested_positions,
        cluster_parents={"inner": "outer"},
    )
    cases.append(
        _make_case(
            case_id="nested_mixed_shapes",
            category="14_cluster_combos",
            title="Nested Mixed Shapes",
            graph=graph,
            positions=positions,
            options_tested=["nested_clusters=2", "mixed_shapes=true"],
            graphviz=_pinned_graphviz_spec(
                positions,
                node_attrs={
                    0: _graphviz_shape_attrs("roundrect"),
                    1: _graphviz_shape_attrs("diamond"),
                    2: _graphviz_shape_attrs("hexagon"),
                    3: _graphviz_shape_attrs("circle"),
                },
                cluster_attrs={"inner": {"style": "filled", "fillcolor": "#DCEBFA"}},
            ),
        )
    )

    thick_positions = {
        "inlet": (-180.0, 110.0),
        "alpha": (-60.0, 0.0),
        "beta": (60.0, 0.0),
        "outlet": (180.0, -110.0),
    }
    graph, positions = _cluster_graph_custom(
        [
            ("inlet", "In", {}),
            ("alpha", "Alpha", {}),
            ("beta", "Beta", {}),
            ("outlet", "Out", {}),
        ],
        [("inlet", "alpha"), ("alpha", "beta"), ("beta", "outlet"), ("inlet", "outlet")],
        [("group", "Boundary", ["alpha", "beta"], {})],
        positions=thick_positions,
    )
    graph.edge_styles = [
        _base_edge_style(),
        _base_edge_style(width=3.0),
        _base_edge_style(),
        _base_edge_style(),
    ]
    cases.append(
        _make_case(
            case_id="cluster_thick_edge",
            category="14_cluster_combos",
            title="Cluster Thick Edge",
            graph=graph,
            positions=positions,
            options_tested=["cluster=true", "edge.width=3.0"],
            graphviz=_pinned_graphviz_spec(positions, edge_attrs={1: {"penwidth": "3"}}),
        )
    )

    graph, positions = _cluster_graph_custom(
        [
            ("top", "Top", {}),
            ("left", "Left", {}),
            ("right", "Right", {}),
            ("bottom", "Bottom", {}),
        ],
        [("top", "left"), ("top", "right"), ("left", "bottom"), ("right", "bottom")],
        [("middle", "Middle", ["left", "right"], {})],
        positions={
            "top": (0.0, 170.0),
            "left": (-120.0, 20.0),
            "right": (120.0, 20.0),
            "bottom": (0.0, -140.0),
        },
    )
    _set_all_edge_styles(graph, _base_edge_style(routing="ortho"))
    cases.append(
        _make_case(
            case_id="cluster_ortho",
            category="14_cluster_combos",
            title="Cluster Ortho",
            graph=graph,
            positions=positions,
            options_tested=["cluster=true", "routing=ortho"],
            graphviz=_pinned_graphviz_spec(positions, graph_attrs={"splines": "ortho"}),
        )
    )

    long_positions = {"a": (-150.0, 40.0), "b": (10.0, 40.0), "c": (160.0, -120.0)}
    graph, positions = _cluster_graph_custom(
        [
            ("a", "Very Long Ingest Step", {}),
            ("b", "Very Long Validation Step", {}),
            ("c", "Exit", {}),
        ],
        [("a", "b"), ("b", "c")],
        [("group", "Processing Cluster", ["a", "b"], {})],
        positions=long_positions,
    )
    cases.append(
        _make_case(
            case_id="cluster_long_labels",
            category="14_cluster_combos",
            title="Cluster Long Labels",
            graph=graph,
            positions=positions,
            options_tested=["cluster.label=Processing Cluster", "long_labels=true"],
            graphviz=_pinned_graphviz_spec(positions),
        )
    )

    graph, positions = _cluster_graph_custom(
        [
            ("a", "Shadow A", {"shadow": True}),
            ("b", "Shadow B", {"shadow": True}),
            ("c", "Outside", {}),
        ],
        [("a", "b"), ("b", "c")],
        [("shadow_box", "Shadow Cluster", ["a", "b"], {})],
        positions=base_positions,
    )
    cases.append(
        _make_case(
            case_id="cluster_shadow_nodes",
            category="14_cluster_combos",
            title="Cluster Shadow Nodes",
            graph=graph,
            positions=positions,
            options_tested=["cluster=true", "node.shadow=true"],
            graphviz=None,
        )
    )

    twin_positions = {
        "a1": (-260.0, 20.0),
        "a2": (-120.0, 20.0),
        "b1": (120.0, 20.0),
        "b2": (260.0, 20.0),
    }
    graph, positions = _cluster_graph_custom(
        [("a1", "A1", {}), ("a2", "A2", {}), ("b1", "B1", {}), ("b2", "B2", {})],
        [("a1", "a2"), ("b1", "b2")],
        [
            ("left", "Blue", ["a1", "a2"], {"fill": "#DCEBFA"}),
            ("right", "Green", ["b1", "b2"], {"fill": "#D9F2E2"}),
        ],
        positions=twin_positions,
        direction="LR",
    )
    cases.append(
        _make_case(
            case_id="two_clusters_colors",
            category="14_cluster_combos",
            title="Two Clusters Colors",
            graph=graph,
            positions=positions,
            options_tested=["clusters=2", "cluster_fill=two_colors"],
            graphviz=_pinned_graphviz_spec(
                positions,
                cluster_attrs={
                    "left": {"style": "filled", "fillcolor": "#DCEBFA"},
                    "right": {"style": "filled", "fillcolor": "#D9F2E2"},
                },
            ),
        )
    )

    stack_positions = {
        "outer": (0.0, 180.0),
        "middle": (0.0, 70.0),
        "inner": (0.0, -40.0),
        "leaf": (0.0, -150.0),
    }
    graph, positions = _cluster_graph_custom(
        [
            ("outer", "Outer", {}),
            ("middle", "Middle", {}),
            ("inner", "Inner", {}),
            ("leaf", "Leaf", {}),
        ],
        [("outer", "middle"), ("middle", "inner"), ("inner", "leaf")],
        [
            ("outer_box", "Outer", ["outer", "middle", "inner", "leaf"], {}),
            ("middle_box", "Middle", ["middle", "inner", "leaf"], {"fill": "#EAF1F8"}),
            ("inner_box", "Inner", ["inner", "leaf"], {"fill": "#DCEBFA"}),
        ],
        positions=stack_positions,
        cluster_parents={"middle_box": "outer_box", "inner_box": "middle_box"},
    )
    cases.append(
        _make_case(
            case_id="nested_three_level",
            category="14_cluster_combos",
            title="Nested Three Level",
            graph=graph,
            positions=positions,
            options_tested=["nested_clusters=3"],
            graphviz=_pinned_graphviz_spec(
                positions,
                cluster_attrs={
                    "middle_box": {"style": "filled", "fillcolor": "#EAF1F8"},
                    "inner_box": {"style": "filled", "fillcolor": "#DCEBFA"},
                },
            ),
        )
    )
    return cases


def _cat15_color_contrast_cases() -> List[AlbumCase]:
    """Build category 15 color contrast cases.

    Returns
    -------
    list[AlbumCase]
        Requested ``15_color_contrast`` cases.
    """

    specs = [
        ("dark_dark_text", "#1A1A1A", "#4C77A3", "#333333", True),
        ("dark_light_text", "#1A1A1A", "#5A8AB0", "#FFFFFF", True),
        ("light_light_stroke", "#F5F5F5", "#E0E0E0", "#333333", True),
        ("garish", "#FF6B6B", "#4ECB71", "#1A1A1A", True),
        ("pastel", "#E8D5E8", "#D5E8D5", "#555555", True),
        ("invisible", "#FFFFFF", "#FFFFFF", "#333333", True),
        ("black_gradient", "#1A1A1A", "#333333", "#FFFFFF", False),
    ]

    cases: List[AlbumCase] = []
    for case_id, fill, stroke, font_color, compare in specs:
        graph, positions = _pair_graph(
            _pair_positions(direction="LR"), ["Alpha", "Beta"], direction="LR"
        )
        node_style = _base_node_style(
            fill=fill,
            stroke=stroke,
            font_color=font_color,
            gradient="linear" if case_id == "black_gradient" else "none",
        )
        _set_all_node_styles(graph, node_style)
        _set_all_edge_styles(graph, _base_edge_style())
        graphviz = None
        if compare:
            graphviz = _pinned_graphviz_spec(
                positions,
                default_node_attrs={"fillcolor": fill, "color": stroke, "fontcolor": font_color},
            )
        cases.append(
            _make_case(
                case_id=case_id,
                category="15_color_contrast",
                title=case_id.replace("_", " ").title(),
                graph=graph,
                positions=positions,
                options_tested=[f"fill={fill}", f"stroke={stroke}", f"font_color={font_color}"],
                graphviz=graphviz,
            )
        )

    graph, positions = _pair_graph(
        _pair_positions(direction="LR"), ["Red", "Green"], direction="LR"
    )
    _set_all_edge_styles(graph, _base_edge_style())
    graph.node_styles = [
        _base_node_style(fill="#FF6B6B", stroke="#FF6B6B", font_color="#1A1A1A"),
        _base_node_style(fill="#6BCB77", stroke="#6BCB77", font_color="#1A1A1A"),
    ]
    cases.append(
        _make_case(
            case_id="red_green",
            category="15_color_contrast",
            title="Red Green",
            graph=graph,
            positions=positions,
            options_tested=["left.fill=#FF6B6B", "right.fill=#6BCB77"],
            graphviz=_pinned_graphviz_spec(
                positions,
                node_attrs={
                    0: {"fillcolor": "#FF6B6B", "color": "#FF6B6B", "fontcolor": "#1A1A1A"},
                    1: {"fillcolor": "#6BCB77", "color": "#6BCB77", "fontcolor": "#1A1A1A"},
                },
            ),
        )
    )
    return cases


def _dark_mode_graph_style() -> GraphStyle:
    """Return the graph style used for dark-mode cases.

    Returns
    -------
    GraphStyle
        Dark background graph style.
    """

    return GraphStyle(
        background_color="#1A1E24",
        margin=8.0,
        edge_label_background="#1A1E24",
        edge_label_background_opacity=0.95,
        min_figsize=(2.0, 1.5),
    )


def _dark_node_style(**overrides: object) -> NodeStyle:
    """Return the dark-mode node style for combo album cases.

    Parameters
    ----------
    **overrides : object
        Field overrides applied to the base style.

    Returns
    -------
    NodeStyle
        Configured node style.
    """

    style = _base_node_style(fill="#2A3A4A", stroke="#5A8AB0", font_color="#E0E0E0")
    for key, value in overrides.items():
        setattr(style, key, value)
    return style


def _dark_edge_style(**overrides: object) -> EdgeStyle:
    """Return the dark-mode edge style for combo album cases.

    Parameters
    ----------
    **overrides : object
        Field overrides applied to the base style.

    Returns
    -------
    EdgeStyle
        Configured edge style.
    """

    style = _base_edge_style(color="#5A8AB0")
    for key, value in overrides.items():
        setattr(style, key, value)
    return style


def _cat16_dark_mode_cases() -> List[AlbumCase]:
    """Build category 16 dark-mode cases.

    Returns
    -------
    list[AlbumCase]
        Requested ``16_dark_mode`` cases.
    """

    specs = [
        ("dark_baseline", "baseline"),
        ("dark_gradient", "gradient"),
        ("dark_shadow", "shadow"),
        ("dark_opacity", "opacity"),
        ("dark_dashed", "dashed"),
        ("dark_cluster", "cluster"),
    ]

    cases: List[AlbumCase] = []
    for case_id, variant in specs:
        graph, positions = _chain_graph(3, ["Input", "Process", "Output"], direction="LR")
        graph._theme.graph_style = _dark_mode_graph_style()
        graph.node_styles = [_dark_node_style() for _ in range(3)]
        graph.edge_styles = [_dark_edge_style() for _ in range(2)]
        if variant == "gradient":
            graph.node_styles = [_dark_node_style(gradient="linear") for _ in range(3)]
        elif variant == "shadow":
            graph.node_styles = [_dark_node_style(shadow=True) for _ in range(3)]
        elif variant == "opacity":
            graph.node_styles = [_dark_node_style(opacity=0.6) for _ in range(3)]
        elif variant == "dashed":
            graph.edge_styles = [_dark_edge_style(style="dashed") for _ in range(2)]
        elif variant == "cluster":
            graph.add_cluster("dark_cluster", ["n0", "n1"], label="Backend")
            graph.cluster_styles["dark_cluster"] = _base_cluster_style(
                fill="#2A3A4A44",
                stroke="#5A8AB0",
                font_color="#E0E0E0",
            )
        cases.append(
            _make_case(
                case_id=case_id,
                category="16_dark_mode",
                title=case_id.replace("_", " ").title(),
                graph=graph,
                positions=positions,
                options_tested=[f"variant={variant}", "background=#1A1E24"],
                graphviz=None,
            )
        )
    return cases


def _cat17_extreme_params_cases() -> List[AlbumCase]:
    """Build category 17 extreme-parameter cases.

    Returns
    -------
    list[AlbumCase]
        Requested ``17_extreme_params`` cases.
    """

    cases: List[AlbumCase] = []

    node_specs = [
        (
            "tiny_font",
            {"shape": "diamond", "font_size": 6.0},
            {"shape": "diamond", "fontsize": "6"},
        ),
        (
            "huge_font",
            {"shape": "roundrect", "font_size": 24.0},
            {"shape": "box", "style": "filled,rounded", "fontsize": "24"},
        ),
        (
            "thick_border",
            {"shape": "roundrect", "stroke_width": 5.0},
            {"shape": "box", "style": "filled,rounded", "penwidth": "5"},
        ),
        (
            "thin_border",
            {"shape": "diamond", "stroke_width": 0.2},
            {"shape": "diamond", "penwidth": "0.2"},
        ),
        ("pill_shape", {"shape": "roundrect", "corner_radius": 25.0}, None),
        ("sharp_roundrect", {"shape": "roundrect", "corner_radius": 0.0}, None),
    ]
    for case_id, overrides, gv_attrs in node_specs:
        graph, positions = _single_node_graph("Extreme")
        graph.node_styles[0] = _base_node_style(**overrides)
        cases.append(
            _make_case(
                case_id=case_id,
                category="17_extreme_params",
                title=case_id.replace("_", " ").title(),
                graph=graph,
                positions=positions,
                options_tested=[f"node.{key}={value}" for key, value in overrides.items()],
                graphviz=None
                if gv_attrs is None
                else _pinned_graphviz_spec(positions, node_attrs={0: gv_attrs}),
            )
        )

    edge_specs = [
        ("huge_arrow", {"arrow_length": 25.0, "arrow_width": 20.0}, {"arrowsize": "2.0"}),
        ("tiny_arrow", {"arrow_length": 4.0, "arrow_width": 3.0}, {"arrowsize": "0.3"}),
        ("max_curvature", {"curvature": 1.0}, None),
        ("zero_curvature", {"curvature": 0.0}, None),
    ]
    for case_id, overrides, gv_attrs in edge_specs:
        graph, positions = _pair_graph(_pair_positions(direction="LR"), ["A", "B"], direction="LR")
        _set_all_edge_styles(graph, _base_edge_style(**overrides))
        cases.append(
            _make_case(
                case_id=case_id,
                category="17_extreme_params",
                title=case_id.replace("_", " ").title(),
                graph=graph,
                positions=positions,
                options_tested=[f"edge.{key}={value}" for key, value in overrides.items()],
                graphviz=None
                if gv_attrs is None
                else _pinned_graphviz_spec(positions, default_edge_attrs=gv_attrs),
            )
        )
    return cases


def _cat18_dense_mixed_cases() -> List[AlbumCase]:
    """Build category 18 dense mixed-style cases.

    Returns
    -------
    list[AlbumCase]
        Requested ``18_dense_mixed`` cases.
    """

    cases: List[AlbumCase] = []

    graph, positions = _chain_graph(
        6,
        ["Rect", "Circle", "Diamond", "Hex", "Trap", "Star"],
        direction="LR",
        spacing=130.0,
    )
    graph.node_styles = [
        _base_node_style(shape="roundrect"),
        _base_node_style(shape="circle"),
        _base_node_style(shape="diamond"),
        _base_node_style(shape="hexagon"),
        _base_node_style(shape="trapezoid"),
        _base_node_style(shape="star"),
    ]
    _set_all_edge_styles(graph, _base_edge_style())
    cases.append(
        _make_case(
            case_id="six_shapes_chain",
            category="18_dense_mixed",
            title="Six Shapes Chain",
            graph=graph,
            positions=positions,
            options_tested=["nodes=6", "mixed_shapes=true"],
            graphviz=_pinned_graphviz_spec(
                positions,
                node_attrs={
                    index: _graphviz_shape_attrs(style.shape)
                    for index, style in enumerate(graph.node_styles)
                },
            ),
        )
    )

    fills = ["#DCEBFA", "#F8D6CC", "#D9F2E2", "#FBE7B8", "#E8D8F0", "#F7F2D2"]
    graph, positions = _chain_graph(
        6, ["A", "B", "C", "D", "E", "F"], direction="LR", spacing=130.0
    )
    graph.node_styles = [_base_node_style(fill=fill) for fill in fills]
    _set_all_edge_styles(graph, _base_edge_style())
    cases.append(
        _make_case(
            case_id="six_colors_chain",
            category="18_dense_mixed",
            title="Six Colors Chain",
            graph=graph,
            positions=positions,
            options_tested=["nodes=6", "mixed_fills=true"],
            graphviz=_pinned_graphviz_spec(
                positions,
                node_attrs={
                    index: {"fillcolor": fill, "color": graph.node_styles[index].stroke}
                    for index, fill in enumerate(fills)
                },
            ),
        )
    )

    graph, positions = _diamond_dag(["Start", "Left", "Right", "End"])
    graph.edge_styles = [
        _base_edge_style(style="solid"),
        _base_edge_style(style="dashed"),
        _base_edge_style(style="dotted"),
        _base_edge_style(width=3.0),
    ]
    cases.append(
        _make_case(
            case_id="four_edge_styles",
            category="18_dense_mixed",
            title="Four Edge Styles",
            graph=graph,
            positions=positions,
            options_tested=["diamond_dag=true", "mixed_edge_styles=true"],
            graphviz=_pinned_graphviz_spec(
                positions,
                edge_attrs={
                    0: {"style": "solid"},
                    1: {"style": "dashed"},
                    2: {"style": "dotted"},
                    3: {"penwidth": "3"},
                },
            ),
        )
    )

    graph, positions = _diamond_dag(["Start", "Left", "Right", "End"])
    graph.edge_styles = [
        _base_edge_style(arrow="normal"),
        _base_edge_style(arrow="vee"),
        _base_edge_style(arrow="diamond"),
        _base_edge_style(arrow="crow"),
    ]
    cases.append(
        _make_case(
            case_id="four_arrows",
            category="18_dense_mixed",
            title="Four Arrows",
            graph=graph,
            positions=positions,
            options_tested=["diamond_dag=true", "mixed_arrows=true"],
            graphviz=_pinned_graphviz_spec(
                positions,
                edge_attrs={
                    0: {"arrowhead": "normal"},
                    1: {"arrowhead": "vee"},
                    2: {"arrowhead": "diamond"},
                    3: {"arrowhead": "crow"},
                },
            ),
        )
    )

    cluster_positions = {
        "a0": (-280.0, 90.0),
        "a1": (-160.0, 90.0),
        "a2": (-280.0, -30.0),
        "a3": (-160.0, -30.0),
        "b0": (160.0, 90.0),
        "b1": (280.0, 90.0),
        "b2": (160.0, -30.0),
        "b3": (280.0, -30.0),
    }
    graph, positions = _cluster_graph_custom(
        [
            ("a0", "A0", {"shape": "roundrect"}),
            ("a1", "A1", {"shape": "circle"}),
            ("a2", "A2", {"shape": "diamond"}),
            ("a3", "A3", {"shape": "hexagon"}),
            ("b0", "B0", {"shape": "trapezoid"}),
            ("b1", "B1", {"shape": "star"}),
            ("b2", "B2", {"shape": "roundrect"}),
            ("b3", "B3", {"shape": "circle"}),
        ],
        [
            ("a0", "a1"),
            ("a1", "a3"),
            ("a0", "a2"),
            ("b0", "b1"),
            ("b1", "b3"),
            ("b0", "b2"),
            ("a3", "b0"),
            ("a2", "b2"),
        ],
        [
            ("left", "Left", ["a0", "a1", "a2", "a3"], {}),
            ("right", "Right", ["b0", "b1", "b2", "b3"], {"fill": "#EAF1F8"}),
        ],
        positions=cluster_positions,
        direction="LR",
    )
    cases.append(
        _make_case(
            case_id="two_clusters_mixed",
            category="18_dense_mixed",
            title="Two Clusters Mixed",
            graph=graph,
            positions=positions,
            options_tested=["nodes=8", "clusters=2", "mixed_shapes=true"],
            graphviz=_pinned_graphviz_spec(
                positions,
                node_attrs={
                    index: _graphviz_shape_attrs(style.shape)
                    for index, style in enumerate(graph.node_styles)
                },
                cluster_attrs={"right": {"style": "filled", "fillcolor": "#EAF1F8"}},
            ),
        )
    )

    graph, positions = _chain_graph(
        6, ["G0", "G1", "G2", "S0", "S1", "S2"], direction="LR", spacing=125.0
    )
    graph.node_styles = [
        _base_node_style(gradient="linear"),
        _base_node_style(gradient="radial"),
        _base_node_style(gradient="linear"),
        _base_node_style(shadow=True),
        _base_node_style(shadow=True),
        _base_node_style(shadow=True),
    ]
    _set_all_edge_styles(graph, _base_edge_style())
    cases.append(
        _make_case(
            case_id="gradient_shadow_mix",
            category="18_dense_mixed",
            title="Gradient Shadow Mix",
            graph=graph,
            positions=positions,
            options_tested=["gradients=3", "shadows=3"],
            graphviz=None,
        )
    )

    graph, positions = _fan_graph("Hub", ["Circle", "Diamond", "Hex", "Trap"])
    graph.node_styles = [
        _base_node_style(),
        _base_node_style(shape="circle", fill="#DCEBFA"),
        _base_node_style(shape="diamond", fill="#F8D6CC"),
        _base_node_style(shape="hexagon", fill="#D9F2E2"),
        _base_node_style(shape="trapezoid", fill="#FBE7B8"),
    ]
    _set_all_edge_styles(graph, _base_edge_style())
    cases.append(
        _make_case(
            case_id="fan_shapes",
            category="18_dense_mixed",
            title="Fan Shapes",
            graph=graph,
            positions=positions,
            options_tested=["fan_out=true", "mixed_leaf_shapes=true", "mixed_leaf_colors=true"],
            graphviz=_pinned_graphviz_spec(
                positions,
                node_attrs={
                    index: _graphviz_shape_attrs(style.shape)
                    for index, style in enumerate(graph.node_styles)
                },
            ),
        )
    )

    graph, positions = _grid_graph(3, 3)
    graph.node_styles = [
        _base_node_style(
            shape="rect" if (index + row) % 2 == 0 else "circle",
            stroke_dash="solid" if (index + row) % 2 == 0 else "dashed",
        )
        for row in range(3)
        for index in range(3)
    ]
    _set_all_edge_styles(graph, _base_edge_style())
    cases.append(
        _make_case(
            case_id="grid_checkerboard",
            category="18_dense_mixed",
            title="Grid Checkerboard",
            graph=graph,
            positions=positions,
            options_tested=["grid=3x3", "checkerboard_shapes=true", "checkerboard_borders=true"],
            graphviz=_pinned_graphviz_spec(
                positions,
                node_attrs={
                    index: _graphviz_shape_attrs(style.shape, style.stroke_dash)
                    for index, style in enumerate(graph.node_styles)
                },
            ),
        )
    )
    return cases


def _cat19_real_world_patterns_cases() -> List[AlbumCase]:
    """Build category 19 real-world pattern cases.

    Returns
    -------
    list[AlbumCase]
        Requested ``19_real_world_patterns`` cases.
    """

    cases: List[AlbumCase] = []

    graph, positions = _chain_graph(
        5,
        ["Input", "Parse", "Transform", "Validate", "Output"],
        direction="LR",
        spacing=150.0,
    )
    _set_all_node_styles(graph, _base_node_style(shape="roundrect"))
    graph.edge_styles = [
        _base_edge_style(routing="ortho"),
        _base_edge_style(routing="ortho"),
        _base_edge_style(routing="ortho"),
        _base_edge_style(routing="ortho", style="dashed"),
    ]
    _set_all_edge_labels(graph, ["data", "transform", "validate", "output"])
    graph.add_cluster("processing", ["n1", "n2", "n3"], label="Processing")
    graph.cluster_styles["processing"] = _base_cluster_style()
    cases.append(
        _make_case(
            case_id="pipeline",
            category="19_real_world_patterns",
            title="Pipeline",
            graph=graph,
            positions=positions,
            options_tested=[
                "direction=LR",
                "routing=ortho",
                "cluster=Processing",
                "optional_last_edge=true",
            ],
            graphviz=_pinned_graphviz_spec(
                positions,
                graph_attrs={"rankdir": "LR", "splines": "ortho"},
                node_attrs={index: _graphviz_shape_attrs("roundrect") for index in range(5)},
                edge_attrs={3: {"style": "dashed"}},
            ),
        )
    )

    state_positions = torch.tensor(
        [[-340.0, 0.0], [-180.0, 0.0], [-20.0, 0.0], [160.0, 0.0], [340.0, 0.0]],
        dtype=torch.float32,
    )
    graph = DaguaGraph(direction="LR")
    _apply_graph_style(graph)
    for node_id, label in [
        ("start", ""),
        ("idle", "Idle"),
        ("active", "Active"),
        ("processing", "Processing"),
        ("done", "Done"),
    ]:
        graph.add_node(node_id, label=label)
    for source, target in [
        ("start", "idle"),
        ("idle", "active"),
        ("active", "processing"),
        ("processing", "done"),
        ("processing", "idle"),
        ("processing", "processing"),
    ]:
        graph.add_edge(source, target)
    graph.node_styles = [
        _base_node_style(
            shape="circle", fill="#1F2937", stroke="#1F2937", min_width=20.0, padding=(4.0, 4.0)
        ),
        _base_node_style(shape="circle"),
        _base_node_style(shape="circle"),
        _base_node_style(shape="circle"),
        _base_node_style(shape="circle"),
    ]
    graph.edge_styles = [_base_edge_style() for _ in range(6)]
    _set_all_edge_labels(graph, [None, "start", "process", "complete", "timeout", "retry"])
    cases.append(
        _make_case(
            case_id="state_machine",
            category="19_real_world_patterns",
            title="State Machine",
            graph=graph,
            positions=state_positions,
            options_tested=[
                "direction=LR",
                "self_loop=retry",
                "edge_labels=start,process,complete,timeout",
            ],
            graphviz=_pinned_graphviz_spec(
                state_positions,
                graph_attrs={"rankdir": "LR", "splines": "true"},
                node_attrs={index: _graphviz_shape_attrs("circle") for index in range(5)},
            ),
        )
    )

    flow_positions = torch.tensor(
        [[0.0, 240.0], [0.0, 100.0], [-150.0, -40.0], [150.0, -40.0], [0.0, -210.0]],
        dtype=torch.float32,
    )
    graph = DaguaGraph(direction="TB")
    _apply_graph_style(graph)
    for node_id, label in [
        ("start", "Start"),
        ("decision", "Decision"),
        ("a", "Process A"),
        ("b", "Process B"),
        ("end", "End"),
    ]:
        graph.add_node(node_id, label=label)
    for source, target in [
        ("start", "decision"),
        ("decision", "a"),
        ("decision", "b"),
        ("a", "end"),
        ("b", "end"),
    ]:
        graph.add_edge(source, target)
    graph.node_styles = [
        _base_node_style(shape="roundrect"),
        _base_node_style(shape="diamond"),
        _base_node_style(shape="rect"),
        _base_node_style(shape="rect"),
        _base_node_style(shape="roundrect"),
    ]
    graph.edge_styles = [
        _base_edge_style(),
        _base_edge_style(),
        _base_edge_style(),
        _base_edge_style(),
        _base_edge_style(style="dashed"),
    ]
    _set_all_edge_labels(graph, [None, "yes", "no", None, "error"])
    cases.append(
        _make_case(
            case_id="flowchart",
            category="19_real_world_patterns",
            title="Flowchart",
            graph=graph,
            positions=flow_positions,
            options_tested=["decision=true", "decision_labels=yes,no", "error_path=dashed"],
            graphviz=_pinned_graphviz_spec(
                flow_positions,
                node_attrs={
                    0: _graphviz_shape_attrs("roundrect"),
                    1: _graphviz_shape_attrs("diamond"),
                    2: _graphviz_shape_attrs("rect"),
                    3: _graphviz_shape_attrs("rect"),
                    4: _graphviz_shape_attrs("roundrect"),
                },
                edge_attrs={4: {"style": "dashed"}},
            ),
        )
    )

    neural_positions = torch.tensor(
        [
            [0.0, 320.0],
            [0.0, 230.0],
            [0.0, 140.0],
            [0.0, 50.0],
            [0.0, -40.0],
            [0.0, -130.0],
            [0.0, -220.0],
            [0.0, -310.0],
        ],
        dtype=torch.float32,
    )
    graph = DaguaGraph(direction="TB")
    _apply_graph_style(graph)
    for index in range(8):
        graph.add_node(f"n{index}", label=f"Layer {index}")
    for source, target in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (1, 5)]:
        graph.add_edge(f"n{source}", f"n{target}")
    graph.node_styles = [_base_node_style(font_size=9.0) for _ in range(8)]
    _set_all_edge_styles(graph, _base_edge_style())
    graph.add_cluster("encoder", ["n0", "n1", "n2", "n3"], label="Encoder")
    graph.add_cluster("decoder", ["n4", "n5", "n6", "n7"], label="Decoder")
    graph.cluster_styles["encoder"] = _base_cluster_style()
    graph.cluster_styles["decoder"] = _base_cluster_style(fill="#EAF1F8")
    cases.append(
        _make_case(
            case_id="neural_net",
            category="19_real_world_patterns",
            title="Neural Net",
            graph=graph,
            positions=neural_positions,
            options_tested=[
                "layers=8",
                "skip_connection=n1->n5",
                "font_size=9",
                "clusters=Encoder,Decoder",
            ],
            graphviz=_pinned_graphviz_spec(
                neural_positions,
                cluster_attrs={"decoder": {"style": "filled", "fillcolor": "#EAF1F8"}},
            ),
        )
    )

    org_positions = torch.tensor(
        [
            [0.0, 250.0],
            [-240.0, 80.0],
            [0.0, 80.0],
            [240.0, 80.0],
            [-240.0, -100.0],
            [0.0, -100.0],
            [240.0, -100.0],
        ],
        dtype=torch.float32,
    )
    graph = DaguaGraph(direction="TB")
    _apply_graph_style(graph)
    for node_id, label in [
        ("ceo", "CEO"),
        ("vp_ops", "VP Ops"),
        ("vp_sales", "VP Sales"),
        ("vp_eng", "VP Eng"),
        ("dir_ops", "Director Ops"),
        ("dir_sales", "Director Sales"),
        ("dir_eng", "Director Eng"),
    ]:
        graph.add_node(node_id, label=label)
    for source, target in [
        ("ceo", "vp_ops"),
        ("ceo", "vp_sales"),
        ("ceo", "vp_eng"),
        ("vp_ops", "dir_ops"),
        ("vp_sales", "dir_sales"),
        ("vp_eng", "dir_eng"),
        ("dir_ops", "vp_sales"),
    ]:
        graph.add_edge(source, target)
    graph.node_styles = [_base_node_style(shape="roundrect") for _ in range(7)]
    graph.edge_styles = [
        _base_edge_style(width=2.0),
        _base_edge_style(width=2.0),
        _base_edge_style(width=2.0),
        _base_edge_style(width=2.0),
        _base_edge_style(width=2.0),
        _base_edge_style(width=2.0),
        _base_edge_style(width=1.0, style="dashed"),
    ]
    cases.append(
        _make_case(
            case_id="org_chart",
            category="19_real_world_patterns",
            title="Org Chart",
            graph=graph,
            positions=org_positions,
            options_tested=["tree=true", "main_edge_width=2.0", "reporting_edge=dashed"],
            graphviz=_pinned_graphviz_spec(
                org_positions,
                node_attrs={index: _graphviz_shape_attrs("roundrect") for index in range(7)},
                edge_attrs={
                    0: {"penwidth": "2"},
                    1: {"penwidth": "2"},
                    2: {"penwidth": "2"},
                    3: {"penwidth": "2"},
                    4: {"penwidth": "2"},
                    5: {"penwidth": "2"},
                    6: {"style": "dashed", "penwidth": "1"},
                },
            ),
        )
    )

    data_positions = torch.tensor(
        [
            [-320.0, 40.0],
            [-130.0, 40.0],
            [40.0, 120.0],
            [40.0, -40.0],
            [220.0, 40.0],
            [390.0, 40.0],
        ],
        dtype=torch.float32,
    )
    graph = DaguaGraph(direction="LR")
    _apply_graph_style(graph)
    for node_id, label in [
        ("client", "Client"),
        ("router", "Router"),
        ("api", "API Server"),
        ("auth", "Auth Service"),
        ("db", "Database"),
        ("response", "Response"),
    ]:
        graph.add_node(node_id, label=label)
    for source, target in [
        ("client", "router"),
        ("router", "api"),
        ("router", "auth"),
        ("api", "db"),
        ("db", "response"),
        ("auth", "response"),
    ]:
        graph.add_edge(source, target)
    graph.node_styles = [
        _base_node_style(shape="ellipse"),
        _base_node_style(shape="diamond"),
        _base_node_style(shape="rect"),
        _base_node_style(shape="rect"),
        _base_node_style(shape="cylinder"),
        _base_node_style(shape="ellipse"),
    ]
    graph.edge_styles = [_base_edge_style(routing="ortho") for _ in range(6)]
    _set_all_edge_labels(graph, ["request", "route", "auth", "query", "result", "token"])
    graph.add_cluster("backend", ["api", "auth", "db"], label="Backend")
    graph.cluster_styles["backend"] = _base_cluster_style()
    cases.append(
        _make_case(
            case_id="data_flow",
            category="19_real_world_patterns",
            title="Data Flow",
            graph=graph,
            positions=data_positions,
            options_tested=[
                "mixed_shapes=true",
                "routing=ortho",
                "edge_labels=true",
                "cluster=Backend",
            ],
            graphviz=_pinned_graphviz_spec(
                data_positions,
                graph_attrs={"rankdir": "LR", "splines": "ortho"},
                node_attrs={
                    0: _graphviz_shape_attrs("ellipse"),
                    1: _graphviz_shape_attrs("diamond"),
                    2: _graphviz_shape_attrs("rect"),
                    3: _graphviz_shape_attrs("rect"),
                    4: _graphviz_shape_attrs("cylinder"),
                    5: _graphviz_shape_attrs("ellipse"),
                },
            ),
        )
    )
    return cases


def _cat20_kitchen_sink_cases() -> List[AlbumCase]:
    """Build category 20 kitchen-sink stress cases.

    Returns
    -------
    list[AlbumCase]
        Requested ``20_kitchen_sink`` cases.
    """

    cases: List[AlbumCase] = []

    graph, positions = _single_node_graph("Diamond")
    graph.node_styles[0] = _base_node_style(
        shape="diamond", stroke_dash="dashed", gradient="linear", shadow=True
    )
    cases.append(
        _make_case(
            case_id="diamond_dashed_gradient_shadow",
            category="20_kitchen_sink",
            title="Diamond Dashed Gradient Shadow",
            graph=graph,
            positions=positions,
            options_tested=[
                "shape=diamond",
                "stroke_dash=dashed",
                "gradient=linear",
                "shadow=true",
            ],
            graphviz=None,
        )
    )

    graph, positions = _pair_graph(
        _pair_positions(direction="LR"), ["Focus", "Target"], direction="LR"
    )
    graph.node_styles = [
        _base_node_style(shape="circle", font_weight="bold", font_style="italic"),
        _base_node_style(shape="circle", font_weight="bold", font_style="italic"),
    ]
    graph.edge_styles = [_base_edge_style(arrow="vee", style="dotted")]
    cases.append(
        _make_case(
            case_id="circle_bold_italic_vee_dotted",
            category="20_kitchen_sink",
            title="Circle Bold Italic Vee Dotted",
            graph=graph,
            positions=positions,
            options_tested=[
                "shape=circle",
                "font_weight=bold",
                "font_style=italic",
                "arrow=vee",
                "style=dotted",
            ],
            graphviz=_pinned_graphviz_spec(
                positions,
                node_attrs={
                    0: {"shape": "circle", "fontname": _graphviz_font_name("bold", "italic")},
                    1: {"shape": "circle", "fontname": _graphviz_font_name("bold", "italic")},
                },
                default_edge_attrs={"arrowhead": "vee", "style": "dotted"},
            ),
        )
    )

    graph, positions = _single_node_graph("Hex")
    graph.node_styles[0] = _base_node_style(
        shape="hexagon", gradient="radial", opacity=0.7, shadow=True
    )
    cases.append(
        _make_case(
            case_id="hexagon_radial_opacity_shadow",
            category="20_kitchen_sink",
            title="Hexagon Radial Opacity Shadow",
            graph=graph,
            positions=positions,
            options_tested=["shape=hexagon", "gradient=radial", "opacity=0.7", "shadow=true"],
            graphviz=None,
        )
    )

    graph, positions = _single_node_graph("Star")
    graph.node_styles[0] = _base_node_style(
        shape="star", stroke_width=3.0, stroke_dash="dotted", font_style="italic"
    )
    cases.append(
        _make_case(
            case_id="star_thick_dotted_italic",
            category="20_kitchen_sink",
            title="Star Thick Dotted Italic",
            graph=graph,
            positions=positions,
            options_tested=[
                "shape=star",
                "stroke_width=3",
                "stroke_dash=dotted",
                "font_style=italic",
            ],
            graphviz=_pinned_graphviz_spec(
                positions,
                node_attrs={
                    0: {
                        "shape": "star",
                        "style": "filled,dotted",
                        "penwidth": "3",
                        "fontname": _graphviz_font_name("regular", "italic"),
                    }
                },
            ),
        )
    )

    graph, positions = _diamond_dag(["Entry", "Left", "Right", "Exit"])
    graph.add_cluster("group", ["n1", "n2"], label="Cluster")
    graph.node_styles = [_base_node_style(shape="diamond") for _ in range(4)]
    graph.edge_styles = [_base_edge_style(style="dashed", routing="ortho") for _ in range(4)]
    graph.cluster_styles["group"] = _base_cluster_style()
    cases.append(
        _make_case(
            case_id="cluster_diamond_dashed_ortho",
            category="20_kitchen_sink",
            title="Cluster Diamond Dashed Ortho",
            graph=graph,
            positions=positions,
            options_tested=["cluster=true", "shape=diamond", "style=dashed", "routing=ortho"],
            graphviz=_pinned_graphviz_spec(
                positions,
                graph_attrs={"splines": "ortho"},
                node_attrs={index: _graphviz_shape_attrs("diamond") for index in range(4)},
                default_edge_attrs={"style": "dashed"},
            ),
        )
    )

    graph, positions = _chain_graph(3, ["Input", "Work", "Output"], direction="LR")
    graph.node_styles = [_base_node_style(shape="trapezoid") for _ in range(3)]
    graph.edge_styles = [_base_edge_style(arrow="crow", width=3.0) for _ in range(2)]
    cases.append(
        _make_case(
            case_id="lr_trapezoid_crow_thick",
            category="20_kitchen_sink",
            title="LR Trapezoid Crow Thick",
            graph=graph,
            positions=positions,
            options_tested=["direction=LR", "shape=trapezoid", "arrow=crow", "width=3"],
            graphviz=_pinned_graphviz_spec(
                positions,
                graph_attrs={"rankdir": "LR"},
                node_attrs={index: _graphviz_shape_attrs("trapezoid") for index in range(3)},
                default_edge_attrs={"arrowhead": "crow", "penwidth": "3"},
            ),
        )
    )

    graph, positions = _diamond_dag(["In", "Prep", "Ship", "Out"])
    graph._theme.graph_style = _dark_mode_graph_style()
    graph.add_cluster("dark_cluster", ["n1", "n2"], label="Cluster")
    graph.node_styles = [_dark_node_style(gradient="linear", shadow=True) for _ in range(4)]
    graph.edge_styles = [_dark_edge_style(style="dashed") for _ in range(4)]
    graph.cluster_styles["dark_cluster"] = _base_cluster_style(
        fill="#2A3A4A44", stroke="#5A8AB0", font_color="#E0E0E0"
    )
    cases.append(
        _make_case(
            case_id="dark_gradient_shadow_dashed_cluster",
            category="20_kitchen_sink",
            title="Dark Gradient Shadow Dashed Cluster",
            graph=graph,
            positions=positions,
            options_tested=[
                "dark_mode=true",
                "gradient=true",
                "shadow=true",
                "dashed=true",
                "cluster=true",
            ],
            graphviz=None,
        )
    )

    graph, positions = _pair_graph(_pair_positions(gap=55.0), ["Near", "Nearer"])
    graph.edge_styles = [_base_edge_style(style="dashed", tail_arrow="dot")]
    _set_all_edge_labels(graph, ["flow"])
    cases.append(
        _make_case(
            case_id="short_headtail_label_dashed",
            category="20_kitchen_sink",
            title="Short Headtail Label Dashed",
            graph=graph,
            positions=positions,
            options_tested=["short_edge=true", "tail_arrow=dot", "label=flow", "style=dashed"],
            graphviz=_pinned_graphviz_spec(
                positions,
                default_edge_attrs={"dir": "both", "arrowtail": "dot", "style": "dashed"},
            ),
        )
    )

    nested_positions = {
        "a": (0.0, 180.0),
        "b": (-100.0, 50.0),
        "c": (100.0, 50.0),
        "d": (0.0, -120.0),
    }
    graph, positions = _cluster_graph_custom(
        [
            ("a", "Outer", {"shape": "roundrect"}),
            ("b", "Inner L", {"shape": "diamond", "gradient": "linear"}),
            ("c", "Inner R", {"shape": "hexagon", "gradient": "radial"}),
            ("d", "Exit", {"shape": "circle"}),
        ],
        [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")],
        [
            ("outer", "Outer", ["a", "b", "c", "d"], {}),
            ("inner", "Inner", ["b", "c"], {"fill": "#DCEBFA"}),
        ],
        positions=nested_positions,
        cluster_parents={"inner": "outer"},
    )
    _set_all_edge_labels(graph, ["enter", "branch", "left", "right"])
    cases.append(
        _make_case(
            case_id="nested_mixed_gradient_labels",
            category="20_kitchen_sink",
            title="Nested Mixed Gradient Labels",
            graph=graph,
            positions=positions,
            options_tested=[
                "nested_clusters=true",
                "mixed_shapes=true",
                "gradients=true",
                "edge_labels=true",
            ],
            graphviz=None,
        )
    )

    graph, positions = _pair_graph(
        _pair_positions(direction="LR"), ["Source", "Sink"], direction="LR"
    )
    graph.node_styles = [
        _base_node_style(font_weight="bold"),
        _base_node_style(font_weight="bold"),
    ]
    graph.edge_styles = [
        _base_edge_style(style="dotted", width=3.0, arrow="diamond", arrow_length=25.0)
    ]
    _set_all_edge_labels(graph, ["Important"])
    cases.append(
        _make_case(
            case_id="thick_dotted_large_diamond_bold",
            category="20_kitchen_sink",
            title="Thick Dotted Large Diamond Bold",
            graph=graph,
            positions=positions,
            options_tested=[
                "style=dotted",
                "width=3",
                "arrow=diamond",
                "large_arrow=true",
                "font_weight=bold",
            ],
            graphviz=_pinned_graphviz_spec(
                positions,
                node_attrs={
                    0: {"fontname": _graphviz_font_name("bold", "normal")},
                    1: {"fontname": _graphviz_font_name("bold", "normal")},
                },
                default_edge_attrs={
                    "style": "dotted",
                    "penwidth": "3",
                    "arrowhead": "diamond",
                    "arrowsize": "2.0",
                },
            ),
        )
    )

    graph, positions = _parallel_graph(3)
    graph.edge_styles = [
        _base_edge_style(style="solid", arrow="normal"),
        _base_edge_style(style="dashed", arrow="vee"),
        _base_edge_style(style="dotted", arrow="diamond"),
    ]
    cases.append(
        _make_case(
            case_id="parallel_mixed_all",
            category="20_kitchen_sink",
            title="Parallel Mixed All",
            graph=graph,
            positions=positions,
            options_tested=["parallel_edges=3", "mixed_styles=true", "mixed_arrows=true"],
            graphviz=_pinned_graphviz_spec(
                positions,
                graph_attrs={"splines": "true"},
                edge_attrs={
                    0: {"style": "solid", "arrowhead": "normal"},
                    1: {"style": "dashed", "arrowhead": "vee"},
                    2: {"style": "dotted", "arrowhead": "diamond"},
                },
            ),
        )
    )

    graph, positions = _self_loop_plus_normal_graph()
    graph.node_styles[0] = _base_node_style(shadow=True)
    graph.edge_styles = [_base_edge_style(style="dashed", arrow="vee"), _base_edge_style()]
    _set_all_edge_labels(graph, ["retry", None])
    cases.append(
        _make_case(
            case_id="selfloop_dashed_vee_label_shadow",
            category="20_kitchen_sink",
            title="Selfloop Dashed Vee Label Shadow",
            graph=graph,
            positions=positions,
            options_tested=[
                "self_loop=true",
                "style=dashed",
                "arrow=vee",
                "label=retry",
                "shadow=true",
            ],
            graphviz=None,
        )
    )
    return cases


def build_combo_catalog() -> List[AlbumCase]:
    """Build the full combo album case catalog.

    Returns
    -------
    list[AlbumCase]
        All album cases in output order.
    """

    builders: Sequence[CategoryBuilder] = (
        _shape_x_border_cases,
        _shape_x_gradient_cases,
        _arrow_x_edgestyle_cases,
        _arrow_x_routing_cases,
        _arrow_proportions_cases,
        _arrow_head_tail_cases,
        _text_overflow_cases,
        _edge_label_cases,
        _short_edge_cases,
        _self_loops_parallel_cases,
        _cat11_opacity_interactions_cases,
        _cat12_shadow_interactions_cases,
        _cat13_direction_x_routing_cases,
        _cat14_cluster_combos_cases,
        _cat15_color_contrast_cases,
        _cat16_dark_mode_cases,
        _cat17_extreme_params_cases,
        _cat18_dense_mixed_cases,
        _cat19_real_world_patterns_cases,
        _cat20_kitchen_sink_cases,
    )
    cases: List[AlbumCase] = []
    for builder in builders:
        cases.extend(builder())

    seen_outputs: set[Tuple[str, str]] = set()
    for case in cases:
        output_key = (case.category, case.filename)
        if output_key in seen_outputs:
            raise ValueError(
                f"Duplicate combo album output target: {case.category}/{case.filename}"
            )
        seen_outputs.add(output_key)
    return cases


def build_case_catalog() -> List[AlbumCase]:
    """Return the combo album case catalog.

    Returns
    -------
    list[AlbumCase]
        Full combo album case catalog.

    Notes
    -----
    This compatibility alias preserves the original Part 1 entry point while
    exposing the spec-requested ``build_combo_catalog`` name.
    """

    return build_combo_catalog()


def _select_cases(
    cases: Sequence[AlbumCase],
    categories: Optional[Sequence[str]] = None,
    case_ids: Optional[Sequence[str]] = None,
) -> List[AlbumCase]:
    """Filter combo cases by category.

    Parameters
    ----------
    cases : sequence[AlbumCase]
        Full case catalog.
    categories : sequence[str] | None, default=None
        Optional category names to keep.
    case_ids : sequence[str] | None, default=None
        Optional case identifiers to keep.

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


def _graphviz_available() -> bool:
    """Return whether the Graphviz ``dot`` executable is available.

    Returns
    -------
    bool
        ``True`` when Graphviz is installed.
    """

    return shutil.which("dot") is not None


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
        Final case image path.
    """

    return root / case.category / case.filename


def _options_tested(case: AlbumCase) -> List[str]:
    """Return the manifest option list stored on a case.

    Parameters
    ----------
    case : AlbumCase
        Album case.

    Returns
    -------
    list[str]
        Case option summary.
    """

    raw_options = case.settings.get("options_tested", [])
    return list(cast(Sequence[str], raw_options))


def _manifest_entry(case: AlbumCase, output_path: Path, render_mode: str) -> Dict[str, object]:
    """Build a manifest record for a rendered case.

    Parameters
    ----------
    case : AlbumCase
        Case that was rendered.
    output_path : Path
        Final image path.
    render_mode : str
        ``comparison`` or ``dagua_only`` depending on the emitted panel.

    Returns
    -------
    dict[str, object]
        JSON-serializable manifest row.
    """

    return {
        "case_id": case.case_id,
        "category": case.category,
        "filename": case.filename,
        "title": case.title,
        "comparison": case.graphviz is not None,
        "competitor": GRAPHVIZ_LABEL if case.graphviz is not None else None,
        "render_mode": render_mode,
        "output_path": str(output_path),
        "options_tested": _options_tested(case),
        "risk_level": CATEGORY_RISK.get(case.category, "medium"),
    }


def _write_summary(root: Path, cases: Sequence[AlbumCase], dagua_only: bool) -> Path:
    """Write a Markdown summary of the rendered combo cases.

    Parameters
    ----------
    root : Path
        Album root directory.
    cases : sequence[AlbumCase]
        Rendered cases in output order.
    dagua_only : bool
        Whether the output was forced to Dagua-only panels.

    Returns
    -------
    Path
        Summary Markdown path.
    """

    grouped: Dict[str, List[AlbumCase]] = defaultdict(list)
    for case in cases:
        grouped[case.category].append(case)

    lines = [
        "# Cosmetic Combination Album",
        "",
        f"Generated: {datetime.now().date().isoformat()}",
        f"Total images: {len(cases)}",
        f"Categories: {len(grouped)}",
        "",
        "## Category Index",
        "",
    ]
    if dagua_only:
        lines.append(
            "Rendered in Dagua-only mode. Comparison-capable cases were exported as solo panels."
        )
        lines.append("")
    for category in sorted(grouped):
        lines.append(f"### {category} ({len(grouped[category])} images)")
        lines.append(CATEGORY_DESCRIPTIONS.get(category, ""))
        lines.append(f"Risk: {CATEGORY_RISK.get(category, 'medium').title()}")
        lines.append("")
        lines.append("| Case | Options | Comparison | Notes |")
        lines.append("| --- | --- | --- | --- |")
        for case in grouped[category]:
            option_text = ", ".join(f"`{option}`" for option in _options_tested(case))
            comparison_text = "dagua-only" if dagua_only or case.graphviz is None else "vs Graphviz"
            lines.append(f"| {case.case_id} | {option_text} | {comparison_text} | {case.title} |")
        lines.append("")

    lines.extend(
        [
            "## Methodology",
            "",
            "- All positions are fixed to isolate rendering from layout behavior.",
            "- Graphviz comparisons use pinned node positions via `neato` to keep geometry stable.",
            (
                "- Comparison-capable cases use Graphviz-oriented defaults on the "
                "dagua side for a fairer visual baseline."
            ),
            "- Dagua-only cases skip Graphviz when the feature has no useful analogue.",
            "",
            "## Pass/Fail Rubric",
            "",
            "- FAIL: Text is clipped, unreadable, or collides badly with other marks.",
            "- FAIL: Arrowheads overlap nodes or detach visibly from the edge stroke.",
            "- FAIL: Elements disappear because of opacity or poor color contrast.",
            "- FAIL: Dash, loop, shadow, or gradient artefacts are visually obvious.",
            (
                "- NOTE: Proportional imbalance and visual clutter are still defects "
                "even when geometry is technically valid."
            ),
            "",
        ]
    )

    summary_path = root / "summary.md"
    summary_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return summary_path


def _verify_outputs(root: Path, image_paths: Sequence[str]) -> None:
    """Verify that all generated images exist and are readable.

    Parameters
    ----------
    root : Path
        Album root directory.
    image_paths : sequence[str]
        Expected image paths.

    Returns
    -------
    None
        Raises when the output set is incomplete or corrupt.
    """

    _ = root
    for raw_path in image_paths:
        path = Path(raw_path)
        if not path.exists():
            raise RuntimeError(f"Missing generated image: {path}")
        if path.stat().st_size <= 0:
            raise RuntimeError(f"Generated empty image: {path}")
        with Image.open(path) as image:
            image.verify()


def build_combo_album(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    categories: Optional[Sequence[str]] = None,
    case_ids: Optional[Sequence[str]] = None,
    dagua_only: bool = False,
) -> CosmeticAlbumResult:
    """Render the cosmetic combo album and its manifest.

    Parameters
    ----------
    output_dir : str, default=DEFAULT_OUTPUT_DIR
        Album root directory.
    categories : sequence[str] | None, default=None
        Optional subset of category names to render.
    case_ids : sequence[str] | None, default=None
        Optional subset of case identifiers to render.
    dagua_only : bool, default=False
        Render every case as a Dagua-only panel and skip Graphviz execution.

    Returns
    -------
    CosmeticAlbumResult
        Output directory, manifest path, and image paths.
    """

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    catalog = build_case_catalog()
    selected_cases = _select_cases(catalog, categories=categories, case_ids=case_ids)
    if not selected_cases:
        raise ValueError("No combo album cases matched the requested filters.")

    needs_graphviz = any(case.graphviz is not None for case in selected_cases)
    if needs_graphviz and not dagua_only and not _graphviz_available():
        raise RuntimeError("Graphviz is required for comparison cases but is not installed.")

    for category in sorted({case.category for case in selected_cases}):
        (root / category).mkdir(parents=True, exist_ok=True)

    image_paths: List[str] = []
    manifest_cases: List[Dict[str, object]] = []
    category_counts: Dict[str, int] = {}

    with tempfile.TemporaryDirectory(prefix="dagua_combo_album_") as temp_dir:
        temp_root = Path(temp_dir)
        total = len(selected_cases)
        for index, case in enumerate(selected_cases, start=1):
            print(f"[{index}/{total}] {case.category}/{case.case_id}", flush=True)
            category_counts[case.category] = category_counts.get(case.category, 0) + 1
            dagua_raw = temp_root / f"{case.case_id}_dagua.png"
            _render_dagua_png(case.graph, case.positions, dagua_raw, dpi=RAW_RENDER_DPI)

            output_path = _case_output_path(root, case)
            comparison = case.graphviz is not None and not dagua_only
            if comparison:
                graphviz_raw = temp_root / f"{case.case_id}_graphviz.png"
                dot_source = _build_graphviz_dot(
                    case.graph, cast(GraphvizRenderSpec, case.graphviz)
                )
                _render_graphviz_png(
                    dot_source,
                    graphviz_raw,
                    cast(GraphvizRenderSpec, case.graphviz).engine,
                    dpi=RAW_RENDER_DPI,
                )
                _compose_comparison(
                    dagua_path=dagua_raw,
                    competitor_path=graphviz_raw,
                    title=case.title,
                    output_path=output_path,
                    competitor_label=cast(GraphvizRenderSpec, case.graphviz).competitor_label,
                )
            else:
                _compose_solo(dagua_path=dagua_raw, title=case.title, output_path=output_path)

            image_paths.append(str(output_path))
            manifest_cases.append(
                _manifest_entry(case, output_path, "comparison" if comparison else "dagua_only")
            )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(root),
        "dagua_only": dagua_only,
        "total_images": len(image_paths),
        "category_counts": category_counts,
        "cases": manifest_cases,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(f"{json.dumps(manifest, indent=2)}\n", encoding="utf-8")
    _write_summary(root, selected_cases, dagua_only)
    _verify_outputs(root, image_paths)

    return CosmeticAlbumResult(
        output_dir=str(root),
        manifest_path=str(manifest_path),
        image_paths=image_paths,
    )


def main() -> int:
    """Parse CLI arguments and generate the combo album.

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
        default=None,
        help="Optional category directory to render. Repeat to render multiple categories.",
    )
    parser.add_argument(
        "--dagua-only",
        action="store_true",
        help="Render every selected case as a Dagua-only panel and skip Graphviz.",
    )
    args = parser.parse_args()

    result = build_combo_album(
        output_dir=args.output_dir,
        categories=args.category,
        dagua_only=args.dagua_only,
    )
    print(result.output_dir)
    print(result.manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
