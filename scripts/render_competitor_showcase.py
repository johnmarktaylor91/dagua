#!/usr/bin/env python
"""Render visual checks for cross-package cosmetics and cluster labels."""

from __future__ import annotations

import argparse
import io
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import cairosvg
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

import dagua
from dagua import ClusterStyle, DaguaGraph, EdgeStyle, NodeStyle, get_theme

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "eval_output"
SHOWCASE_PATH = OUTPUT_DIR / "competitor_showcase.png"
CLUSTER_CHECK_PATH = OUTPUT_DIR / "cluster_label_check.png"
PANEL_SIZE = (620, 390)
HEADER_HEIGHT = 76
GALLERY_COLUMNS = 4
GALLERY_ROWS = 4
GALLERY_BACKGROUND = "#E8ECF2"
PANEL_BACKGROUND = "#FFFFFF"
INK = "#172033"
MUTED_INK = "#536078"
DIVIDER = "#C5CEDA"


@dataclass(frozen=True)
class ShowcaseCell:
    """Describe one independently rendered cosmetic feature cell.

    Parameters
    ----------
    title : str
        Human-readable feature name.
    property_text : str
        Style property and representative value shown in the cell.
    build_scene : Callable[[], Tuple[DaguaGraph, torch.Tensor]]
        Factory returning the graph and fixed positions with shape ``[N, 2]``.
    """

    title: str
    property_text: str
    build_scene: Callable[[], Tuple[DaguaGraph, torch.Tensor]]


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a readable gallery font with a portable fallback.

    Parameters
    ----------
    size : int
        Font size in pixels.
    bold : bool, default=False
        Whether to request the bold face.

    Returns
    -------
    PIL.ImageFont.FreeTypeFont | PIL.ImageFont.ImageFont
        Loaded font instance.
    """
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(filename, size=size)
    except OSError:
        return ImageFont.load_default()


def _base_graph() -> DaguaGraph:
    """Return a graph with neutral gallery defaults.

    Returns
    -------
    DaguaGraph
        Empty graph using a white canvas and restrained default styles.
    """
    graph = DaguaGraph()
    graph._theme.graph_style.background_color = PANEL_BACKGROUND
    graph._theme.graph_style.margin = 13.0
    graph.default_node_style = NodeStyle(
        shape="roundrect",
        fill="#EAF1FA",
        stroke="#344A67",
        stroke_width=1.4,
        font_size=10.0,
        font_color=INK,
        padding=(12.0, 8.0),
    )
    graph.default_edge_style = EdgeStyle(
        color="#315B87",
        width=2.6,
        opacity=1.0,
        arrow="normal",
        arrow_length=13.0,
        arrow_width=10.0,
        arrow_node_fraction=0.0,
        label_font_size=9.0,
        label_font_color=INK,
        label_background=PANEL_BACKGROUND,
        label_background_opacity=1.0,
        avoid_nodes=False,
        curvature=0.0,
    )
    return graph


def _positions(values: Sequence[Tuple[float, float]]) -> torch.Tensor:
    """Convert fixed position pairs to a float tensor.

    Parameters
    ----------
    values : sequence[tuple[float, float]]
        Position pairs in renderer data coordinates.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    return torch.tensor(values, dtype=torch.float32)


def _two_node_scene(
    edge_style: EdgeStyle,
    *,
    label: Optional[str] = None,
    diagonal: bool = False,
    source_label: str = "SOURCE",
    target_label: str = "TARGET",
) -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a two-node edge demonstration.

    Parameters
    ----------
    edge_style : EdgeStyle
        Per-edge cosmetics to demonstrate.
    label : str | None, optional
        Optional edge label.
    diagonal : bool, default=False
        Whether the target should sit above the source.
    source_label : str, default="SOURCE"
        Source node label.
    target_label : str, default="TARGET"
        Target node label.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and fixed positions with shape ``[2, 2]``.
    """
    graph = _base_graph()
    graph.add_node("source", source_label)
    graph.add_node("target", target_label)
    graph.add_edge("source", "target", label=label, style=edge_style)
    target_y = 38.0 if diagonal else 0.0
    return graph, _positions([(-72.0, -24.0 if diagonal else 0.0), (72.0, target_y)])


def _fill_opacity_scene() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build opaque and translucent node fills for direct comparison.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and fixed positions with shape ``[2, 2]``.
    """
    graph = _base_graph()
    graph.add_node(
        "opaque",
        "100% fill",
        style=NodeStyle(fill="#EF476F", fill_opacity=1.0, stroke="#8C1735", font_size=11.0),
    )
    graph.add_node(
        "translucent",
        "25% fill",
        style=NodeStyle(fill="#EF476F", fill_opacity=0.25, stroke="#8C1735", font_size=11.0),
    )
    return graph, _positions([(-58.0, 0.0), (58.0, 0.0)])


def _text_opacity_scene() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build node labels at full and reduced text opacity.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and fixed positions with shape ``[2, 2]``.
    """
    graph = _base_graph()
    common = {"fill": "#CDE8FF", "stroke": "#1769AA", "font_size": 14.0}
    graph.add_node("full", "FULL TEXT", style=NodeStyle(**common, text_opacity=1.0))
    graph.add_node("faint", "FAINT TEXT", style=NodeStyle(**common, text_opacity=0.22))
    return graph, _positions([(-62.0, 0.0), (62.0, 0.0)])


def _outline_scene() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a node with a visibly offset dashed outline.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and fixed positions with shape ``[1, 2]``.
    """
    graph = _base_graph()
    graph.add_node(
        "outline",
        "OUTLINE",
        style=NodeStyle(
            shape="roundrect",
            fill="#FFF0B8",
            stroke="#543C00",
            stroke_width=2.0,
            outline_color="#E23D28",
            outline_width=3.0,
            outline_offset=5.0,
            outline_style="dashed",
            font_size=13.0,
            padding=(20.0, 12.0),
        ),
    )
    return graph, _positions([(0.0, 0.0)])


def _text_shadow_scene() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a node label with a strong blurred shadow.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and fixed positions with shape ``[1, 2]``.
    """
    graph = _base_graph()
    graph.add_node(
        "shadow",
        "SHADOW",
        style=NodeStyle(
            fill="#F6F0FF",
            stroke="#6F3FA0",
            font_color="#552178",
            font_size=17.0,
            font_weight="bold",
            text_shadow_color="#111827B8",
            text_shadow_offset=(4.0, -4.0),
            text_shadow_blur=2.5,
            padding=(23.0, 15.0),
        ),
    )
    return graph, _positions([(0.0, 0.0)])


def _rounded_polygons_scene() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build the five rounded polygon node variants.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and fixed positions with shape ``[5, 2]``.
    """
    graph = _base_graph()
    shapes = (
        ("round_triangle", "T"),
        ("round_diamond", "D"),
        ("round_pentagon", "P"),
        ("round_hexagon", "H"),
        ("round_octagon", "O"),
    )
    colors = ("#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9", "#BAE1FF")
    for index, ((shape, label), fill) in enumerate(zip(shapes, colors)):
        graph.add_node(
            str(index),
            label,
            style=NodeStyle(
                shape=shape,
                fill=fill,
                stroke="#334155",
                stroke_width=1.7,
                font_size=11.0,
                min_width=38.0,
                min_height=38.0,
            ),
        )
    return graph, _positions(
        [(-100.0, 38.0), (0.0, 38.0), (100.0, 38.0), (-55.0, -42.0), (55.0, -42.0)]
    )


def _custom_polygon_scene() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a custom concave polygon node.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and fixed positions with shape ``[1, 2]``.
    """
    graph = _base_graph()
    crown_points = [
        (-1.0, -0.8),
        (-1.0, 0.7),
        (-0.45, 0.15),
        (0.0, 1.0),
        (0.45, 0.15),
        (1.0, 0.7),
        (1.0, -0.8),
    ]
    graph.add_node(
        "polygon",
        "CUSTOM",
        style=NodeStyle(
            shape="polygon",
            polygon_points=crown_points,
            fill="#F9A8D4",
            stroke="#831843",
            stroke_width=2.0,
            font_size=11.0,
            min_width=92.0,
            min_height=66.0,
        ),
    )
    return graph, _positions([(0.0, 0.0)])


def _brewer_scene() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build nodes filled by one-based ColorBrewer scheme indices.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and fixed positions with shape ``[4, 2]``.
    """
    graph = _base_graph()
    for index, color_index in enumerate((1, 2, 4, 5)):
        graph.add_node(
            str(index),
            str(color_index),
            style=NodeStyle(
                shape="circle",
                color_scheme="set19",
                fill=str(color_index),
                stroke="#253047",
                font_size=12.0,
                min_width=42.0,
                min_height=42.0,
            ),
        )
    return graph, _positions([(-72.0, 0.0), (-24.0, 0.0), (24.0, 0.0), (72.0, 0.0)])


def _dash_scene() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build an edge using a custom numeric dash array.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and fixed positions with shape ``[2, 2]``.
    """
    return _two_node_scene(
        EdgeStyle(
            color="#006D77",
            width=3.4,
            arrow="none",
            line_dash_pattern=(10.0, 4.0, 2.0, 4.0),
            opacity=1.0,
            curvature=0.0,
            avoid_nodes=False,
        ),
        source_label="A",
        target_label="B",
    )


def _label_halo_scene() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build an edge label outlined for contrast over its edge.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and fixed positions with shape ``[2, 2]``.
    """
    return _two_node_scene(
        EdgeStyle(
            color="#D62828",
            width=5.0,
            arrow="none",
            label_font_size=15.0,
            label_font_weight="bold",
            label_font_color="#111827",
            label_background="",
            text_outline_color="#FFFFFF",
            text_outline_width=3.2,
            label_offset=0.0,
            opacity=1.0,
            curvature=0.0,
            avoid_nodes=False,
        ),
        label="HALO",
        source_label="A",
        target_label="B",
    )


def _autorotate_scene() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a diagonal edge with a tangent-aligned label.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and fixed positions with shape ``[2, 2]``.
    """
    return _two_node_scene(
        EdgeStyle(
            color="#5B21B6",
            width=2.8,
            arrow="normal",
            label_font_size=12.0,
            label_font_weight="bold",
            label_font_color="#5B21B6",
            label_background="#FFFFFF",
            label_background_opacity=0.95,
            label_autorotate=True,
            label_offset=4.0,
            opacity=1.0,
            curvature=0.0,
            avoid_nodes=False,
        ),
        label="follows tangent",
        diagonal=True,
        source_label="LOW",
        target_label="HIGH",
    )


def _wavy_scene() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a high-amplitude wavy edge.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and fixed positions with shape ``[2, 2]``.
    """
    return _two_node_scene(
        EdgeStyle(
            color="#0077B6",
            width=3.2,
            arrow="none",
            line_wave=True,
            line_wave_amplitude=7.0,
            line_wave_wavelength=22.0,
            opacity=1.0,
            curvature=0.0,
            avoid_nodes=False,
        ),
        source_label="A",
        target_label="B",
    )


def _source_arrow_scene() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build an edge carrying an extra marker at its source.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and fixed positions with shape ``[2, 2]``.
    """
    return _two_node_scene(
        EdgeStyle(
            color="#A23E48",
            width=3.0,
            arrow="normal",
            source_arrow="diamond",
            arrow_color="#A23E48",
            arrow_length=17.0,
            arrow_width=14.0,
            arrow_node_fraction=0.0,
            opacity=1.0,
            curvature=0.0,
            avoid_nodes=False,
        ),
        source_label="SOURCE",
        target_label="TARGET",
    )


def _mid_arrow_scene() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build an edge carrying a marker at its midpoint.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and fixed positions with shape ``[2, 2]``.
    """
    return _two_node_scene(
        EdgeStyle(
            color="#2A9D8F",
            width=3.0,
            arrow="none",
            mid_arrow="vee",
            arrow_color="#2A9D8F",
            arrow_length=18.0,
            arrow_width=15.0,
            arrow_node_fraction=0.0,
            opacity=1.0,
            curvature=0.0,
            avoid_nodes=False,
        ),
        source_label="A",
        target_label="B",
    )


def _cross_arrow_scene() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build an edge terminated by a cross/X arrowhead.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and fixed positions with shape ``[2, 2]``.
    """
    return _two_node_scene(
        EdgeStyle(
            color="#7F1D1D",
            width=3.0,
            arrow="cross",
            arrow_color="#DC2626",
            arrow_length=22.0,
            arrow_width=22.0,
            arrow_node_fraction=0.0,
            opacity=1.0,
            curvature=0.0,
            avoid_nodes=False,
        ),
        source_label="FROM",
        target_label="X",
    )


def _cluster_padding_scene() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build an asymmetrically padded cluster.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and fixed positions with shape ``[1, 2]``.
    """
    graph = _base_graph()
    graph.add_node("member", "MEMBER", style=NodeStyle(fill="#FFFFFF", stroke="#1D4ED8"))
    graph.add_cluster(
        "padding",
        ["member"],
        label="TOP 34",
        style=ClusterStyle(
            fill="#DBEAFE",
            fill_opacity=1.0,
            stroke="#2563EB",
            stroke_width=2.0,
            corner_radius=5.0,
            padding=8.0,
            padding_top=34.0,
            padding_right=42.0,
            padding_bottom=20.0,
            padding_left=12.0,
            label_position="top-left",
            label_offset=(6.0, 7.0),
            font_size=10.0,
            font_color="#1E3A8A",
            opacity=1.0,
        ),
    )
    return graph, _positions([(0.0, 0.0)])


def _external_labels_scene() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build the nine supported external-label anchor positions.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and fixed positions with shape ``[9, 2]``.
    """
    graph = _base_graph()
    anchors = (
        "top-left",
        "top-center",
        "top-right",
        "middle-left",
        "center",
        "middle-right",
        "bottom-left",
        "bottom-center",
        "bottom-right",
    )
    abbreviations = ("TL", "TC", "TR", "ML", "C", "MR", "BL", "BC", "BR")
    positions: List[Tuple[float, float]] = []
    for index, (anchor, abbreviation) in enumerate(zip(anchors, abbreviations)):
        graph.add_node(
            str(index),
            "•",
            style=NodeStyle(
                shape="circle",
                fill="#FFE4E6",
                stroke="#BE123C",
                min_width=22.0,
                min_height=22.0,
                padding=(2.0, 2.0),
                external_label=abbreviation,
                external_label_position=anchor,
                external_label_font_size=7.5,
                external_label_font_color="#881337",
                external_label_offset=4.0,
            ),
        )
        row, column = divmod(index, 3)
        positions.append(((column - 1) * 76.0, (1 - row) * 62.0))
    return graph, _positions(positions)


def _showcase_cells() -> Tuple[ShowcaseCell, ...]:
    """Return the complete ordered showcase cell catalog.

    Returns
    -------
    tuple[ShowcaseCell, ...]
        Sixteen cells covering every requested cosmetic feature.
    """
    return (
        ShowcaseCell("Node fill opacity", "fill_opacity: 1.0 vs 0.25", _fill_opacity_scene),
        ShowcaseCell("Node text opacity", "text_opacity: 1.0 vs 0.22", _text_opacity_scene),
        ShowcaseCell(
            "Node outline border",
            "outline: red · 3pt · 5pt offset · dashed",
            _outline_scene,
        ),
        ShowcaseCell(
            "Node text shadow",
            "text_shadow: 4,-4 offset · 2.5 blur",
            _text_shadow_scene,
        ),
        ShowcaseCell(
            "Rounded polygons",
            "round_triangle / diamond / pentagon / hexagon / octagon",
            _rounded_polygons_scene,
        ),
        ShowcaseCell(
            "Custom polygon node",
            "shape='polygon' · polygon_points=7 vertices",
            _custom_polygon_scene,
        ),
        ShowcaseCell("Brewer fill", "color_scheme='set19' · fill=1,2,4,5", _brewer_scene),
        ShowcaseCell(
            "Custom edge dashes",
            "line_dash_pattern=(10,4,2,4)",
            _dash_scene,
        ),
        ShowcaseCell(
            "Edge-label halo",
            "text_outline_color=white · width=3.2",
            _label_halo_scene,
        ),
        ShowcaseCell("Edge-label autorotate", "label_autorotate=True", _autorotate_scene),
        ShowcaseCell(
            "Wavy edge",
            "line_wave=True · amplitude=7 · wavelength=22",
            _wavy_scene,
        ),
        ShowcaseCell("Source arrow", "source_arrow='diamond'", _source_arrow_scene),
        ShowcaseCell("Mid-edge arrow", "mid_arrow='vee'", _mid_arrow_scene),
        ShowcaseCell("Cross/X arrowhead", "arrow='cross'", _cross_arrow_scene),
        ShowcaseCell(
            "Per-side cluster padding",
            "top=34 · right=42 · bottom=20 · left=12",
            _cluster_padding_scene,
        ),
        ShowcaseCell(
            "Nine-way external labels",
            "NodeStyle.external_label_position: 3×3 anchors",
            _external_labels_scene,
        ),
    )


def _render_scene(graph: DaguaGraph, positions: torch.Tensor) -> Image.Image:
    """Render one graph scene through Dagua's public API.

    Parameters
    ----------
    graph : DaguaGraph
        Fully styled graph scene.
    positions : torch.Tensor
        Fixed node positions with shape ``[N, 2]``.

    Returns
    -------
    PIL.Image.Image
        RGB panel image at ``PANEL_SIZE``.
    """
    with tempfile.NamedTemporaryFile(suffix=".png") as temporary:
        figure, _ = dagua.render(
            graph,
            positions=positions,
            output=temporary.name,
            format="png",
            figsize=(PANEL_SIZE[0] / 150.0, (PANEL_SIZE[1] - HEADER_HEIGHT) / 150.0),
            dpi=150,
            show=False,
            fit_to_canvas=0.08,
        )
        plt.close(figure)
        image = Image.open(temporary.name).convert("RGB")
        return image.resize(
            (PANEL_SIZE[0], PANEL_SIZE[1] - HEADER_HEIGHT),
            Image.Resampling.LANCZOS,
        )


def _compose_cell(cell: ShowcaseCell) -> Image.Image:
    """Render and label one showcase cell.

    Parameters
    ----------
    cell : ShowcaseCell
        Cell metadata and graph-scene factory.

    Returns
    -------
    PIL.Image.Image
        Labeled RGB gallery cell.
    """
    graph, positions = cell.build_scene()
    scene = _render_scene(graph, positions)
    canvas = Image.new("RGB", PANEL_SIZE, PANEL_BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 10), cell.title, font=_font(23, bold=True), fill=INK)
    draw.text((18, 43), cell.property_text, font=_font(15), fill=MUTED_INK)
    draw.line((0, HEADER_HEIGHT - 1, PANEL_SIZE[0], HEADER_HEIGHT - 1), fill=DIVIDER, width=2)
    canvas.paste(scene, (0, HEADER_HEIGHT))
    draw.rectangle((0, 0, PANEL_SIZE[0] - 1, PANEL_SIZE[1] - 1), outline=DIVIDER, width=2)
    return canvas


def render_showcase(output_path: Path = SHOWCASE_PATH) -> Path:
    """Render the complete 4×4 competitor-cosmetics gallery.

    Parameters
    ----------
    output_path : pathlib.Path, default=SHOWCASE_PATH
        Destination PNG path.

    Returns
    -------
    pathlib.Path
        Written gallery path.
    """
    cells = _showcase_cells()
    if len(cells) != GALLERY_COLUMNS * GALLERY_ROWS:
        raise RuntimeError("The showcase must contain exactly one feature per 4×4 grid cell")

    gutter = 16
    title_height = 108
    width = GALLERY_COLUMNS * PANEL_SIZE[0] + (GALLERY_COLUMNS + 1) * gutter
    height = title_height + GALLERY_ROWS * PANEL_SIZE[1] + (GALLERY_ROWS + 1) * gutter
    gallery = Image.new("RGB", (width, height), GALLERY_BACKGROUND)
    draw = ImageDraw.Draw(gallery)
    draw.text(
        (gutter, 16),
        "Dagua competitor-cosmetics verification",
        font=_font(34, bold=True),
        fill=INK,
    )
    draw.text(
        (gutter, 62),
        "Each cell isolates one cross-package visual property using dagua.render().",
        font=_font(19),
        fill=MUTED_INK,
    )

    for index, cell in enumerate(cells):
        row, column = divmod(index, GALLERY_COLUMNS)
        x = gutter + column * (PANEL_SIZE[0] + gutter)
        y = title_height + gutter + row * (PANEL_SIZE[1] + gutter)
        gallery.paste(_compose_cell(cell), (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gallery.save(output_path, format="PNG", optimize=True)
    return output_path


def _graphviz_cluster_svg() -> bytes:
    """Render the filled-cluster reference through Graphviz ``dot``.

    Returns
    -------
    bytes
        Native Graphviz SVG document.
    """
    source = r'''
digraph G {
  graph [bgcolor="white", pad="0.25", rankdir="LR"];
  node [shape=ellipse, style=filled, fillcolor="white", color="#1E3A5F",
        fontname="Times", fontsize=14];
  subgraph cluster_filled {
    label="Filled cluster";
    labelloc="t";
    labeljust="c";
    style="filled";
    fillcolor="#DCEEFF";
    color="#2B6CB0";
    penwidth=2;
    margin=18;
    a -> b;
  }
}
'''
    return subprocess.run(
        ["dot", "-Tsvg"],
        input=source.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout


def _graphviz_has_cluster_label_patch(svg_bytes: bytes) -> bool:
    """Detect a distinct filled shape behind Graphviz's cluster label.

    Parameters
    ----------
    svg_bytes : bytes
        Graphviz SVG document for the simple reference scene.

    Returns
    -------
    bool
        ``True`` when the cluster group contains more than its one filled
        boundary shape, indicating a separate label-background patch.
    """
    root = ET.fromstring(svg_bytes)
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    for group in root.findall(".//svg:g", namespace):
        if group.attrib.get("class") != "cluster":
            continue
        filled_shapes = []
        for tag in ("polygon", "path", "rect"):
            for element in group.findall(f"svg:{tag}", namespace):
                fill = element.attrib.get("fill", "none").lower()
                if fill not in {"", "none", "transparent"}:
                    filled_shapes.append(element)
        return len(filled_shapes) > 1
    raise RuntimeError("Graphviz SVG did not contain a cluster group")


def _graphviz_panel(svg_bytes: bytes) -> Image.Image:
    """Rasterize Graphviz SVG into a comparison panel.

    Parameters
    ----------
    svg_bytes : bytes
        Native Graphviz SVG document.

    Returns
    -------
    PIL.Image.Image
        RGB panel at ``PANEL_SIZE``.
    """
    png_bytes = cairosvg.svg2png(
        bytestring=svg_bytes,
        scale=2.0,
        background_color="white",
    )
    rendered = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    fitted = ImageOps.contain(
        rendered,
        (PANEL_SIZE[0] - 30, PANEL_SIZE[1] - HEADER_HEIGHT - 20),
        Image.Resampling.LANCZOS,
    )
    panel = Image.new("RGB", (PANEL_SIZE[0], PANEL_SIZE[1] - HEADER_HEIGHT), "white")
    panel.paste(
        fitted,
        ((panel.width - fitted.width) // 2, (panel.height - fitted.height) // 2),
    )
    return panel


def _dagua_cluster_panel() -> Tuple[Image.Image, bool]:
    """Render Dagua's filled graphviz-strict cluster and detect a label patch.

    Returns
    -------
    tuple[PIL.Image.Image, bool]
        RGB panel and whether Dagua emitted a distinct cluster-label background.
    """
    graph = DaguaGraph()
    graph._theme = get_theme("graphviz_strict")
    graph.add_node("a", "a")
    graph.add_node("b", "b")
    graph.add_edge("a", "b")
    graph.add_cluster(
        "filled",
        ["a", "b"],
        label="Filled cluster",
        style=ClusterStyle(
            fill="#DCEEFF",
            fill_opacity=1.0,
            stroke="#2B6CB0",
            stroke_width=2.0,
            padding=18.0,
            label_position="top-center",
            label_background="",
            label_background_opacity=0.0,
            corner_radius=0.0,
            opacity=1.0,
        ),
    )
    positions = _positions([(-42.0, 0.0), (42.0, 0.0)])
    figure, axes = dagua.render(
        graph,
        positions=positions,
        figsize=(PANEL_SIZE[0] / 150.0, (PANEL_SIZE[1] - HEADER_HEIGHT) / 150.0),
        dpi=150,
        show=False,
        fit_to_canvas=0.08,
    )
    has_patch = any(
        isinstance(patch.get_gid(), str)
        and patch.get_gid().startswith("dagua-cluster-label-filled")
        and patch.get_gid().endswith("-background")
        for patch in axes.patches
    )
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=150, facecolor="white")
    plt.close(figure)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    image = image.resize(
        (PANEL_SIZE[0], PANEL_SIZE[1] - HEADER_HEIGHT),
        Image.Resampling.LANCZOS,
    )
    return image, has_patch


def _labeled_comparison_panel(title: str, subtitle: str, scene: Image.Image) -> Image.Image:
    """Add a title and diagnostic subtitle above a comparison scene.

    Parameters
    ----------
    title : str
        Panel heading.
    subtitle : str
        Label-background diagnostic.
    scene : PIL.Image.Image
        Rendered scene content.

    Returns
    -------
    PIL.Image.Image
        Labeled panel image.
    """
    panel = Image.new("RGB", PANEL_SIZE, PANEL_BACKGROUND)
    draw = ImageDraw.Draw(panel)
    draw.text((18, 10), title, font=_font(23, bold=True), fill=INK)
    draw.text((18, 43), subtitle, font=_font(15), fill=MUTED_INK)
    draw.line((0, HEADER_HEIGHT - 1, PANEL_SIZE[0], HEADER_HEIGHT - 1), fill=DIVIDER, width=2)
    panel.paste(scene, (0, HEADER_HEIGHT))
    draw.rectangle((0, 0, PANEL_SIZE[0] - 1, PANEL_SIZE[1] - 1), outline=DIVIDER, width=2)
    return panel


def render_cluster_label_check(output_path: Path = CLUSTER_CHECK_PATH) -> Tuple[Path, bool, bool]:
    """Render Graphviz and Dagua filled-cluster label treatments side by side.

    Parameters
    ----------
    output_path : pathlib.Path, default=CLUSTER_CHECK_PATH
        Destination PNG path.

    Returns
    -------
    tuple[pathlib.Path, bool, bool]
        Written path, Graphviz patch presence, and Dagua patch presence.
    """
    svg_bytes = _graphviz_cluster_svg()
    graphviz_has_patch = _graphviz_has_cluster_label_patch(svg_bytes)
    dagua_scene, dagua_has_patch = _dagua_cluster_panel()
    graphviz_scene = _graphviz_panel(svg_bytes)

    gutter = 22
    title_height = 108
    width = PANEL_SIZE[0] * 2 + gutter * 3
    height = title_height + PANEL_SIZE[1] + gutter * 2
    comparison = Image.new("RGB", (width, height), GALLERY_BACKGROUND)
    draw = ImageDraw.Draw(comparison)
    draw.text((gutter, 16), "Filled cluster label treatment", font=_font(34, bold=True), fill=INK)
    verdict = "MATCH" if graphviz_has_patch == dagua_has_patch else "MISMATCH"
    draw.text(
        (gutter, 62),
        (
            "Separate label-background patch: "
            f"Graphviz={graphviz_has_patch} · Dagua={dagua_has_patch} · {verdict}"
        ),
        font=_font(19),
        fill="#166534" if verdict == "MATCH" else "#B91C1C",
    )
    graphviz_panel = _labeled_comparison_panel(
        "Graphviz dot → SVG → CairoSVG",
        f"distinct label patch: {graphviz_has_patch}",
        graphviz_scene,
    )
    dagua_panel = _labeled_comparison_panel(
        "Dagua graphviz_strict",
        f"label_background='' · distinct patch: {dagua_has_patch}",
        dagua_scene,
    )
    comparison.paste(graphviz_panel, (gutter, title_height + gutter))
    comparison.paste(dagua_panel, (PANEL_SIZE[0] + gutter * 2, title_height + gutter))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.save(output_path, format="PNG", optimize=True)
    return output_path, graphviz_has_patch, dagua_has_patch


def parse_args() -> argparse.Namespace:
    """Parse command-line options.

    Returns
    -------
    argparse.Namespace
        Parsed output-directory option.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for competitor_showcase.png and cluster_label_check.png",
    )
    return parser.parse_args()


def main() -> None:
    """Render both visual verification deliverables.

    Returns
    -------
    None
        Writes two PNG files and prints patch-detection diagnostics.
    """
    args = parse_args()
    showcase_path = render_showcase(args.output_dir / SHOWCASE_PATH.name)
    cluster_path, graphviz_has_patch, dagua_has_patch = render_cluster_label_check(
        args.output_dir / CLUSTER_CHECK_PATH.name
    )
    print(f"showcase: {showcase_path}")
    print(f"cluster check: {cluster_path}")
    print(f"Graphviz label patch: {graphviz_has_patch}")
    print(f"Dagua label patch: {dagua_has_patch}")
    result = "MATCH" if graphviz_has_patch == dagua_has_patch else "MISMATCH"
    print(f"cluster label result: {result}")


if __name__ == "__main__":
    main()
