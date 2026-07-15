#!/usr/bin/env python
# ruff: noqa: E402
"""Build the user-facing Dagua feature reference gallery.

The gallery is a static set of PNG reference images plus an HTML index for
local browsing. Each image is rendered at a fixed `800x600` size so the
artifacts read consistently in documentation and release bundles.
"""

from __future__ import annotations

import argparse
import html
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

from dagua import DaguaGraph, render
from dagua.render.edges import available_arrowheads
from dagua.styles import (
    PALETTE,
    THEME_REGISTRY,
    ClusterStyle,
    EdgeStyle,
    GraphStyle,
    NodeStyle,
    border_from_fill,
    get_theme,
    make_fill,
)

INDEX_NAME = "index.html"
WHITE = "#FFFFFF"
TEXT_COLOR = "#1F2933"
SUBTLE_TEXT_COLOR = "#5B6670"
CARD_BORDER = "#E5E7EB"
IMAGE_DPI = 200
IMAGE_SIZE_PX: Tuple[int, int] = (800, 600)
IMAGE_SIZE_IN: Tuple[float, float] = (
    IMAGE_SIZE_PX[0] / IMAGE_DPI,
    IMAGE_SIZE_PX[1] / IMAGE_DPI,
)
SHAPE_NAMES: Tuple[str, ...] = (
    "rect",
    "roundrect",
    "ellipse",
    "circle",
    "diamond",
    "triangle",
    "hexagon",
    "pentagon",
    "octagon",
    "star",
    "cylinder",
    "parallelogram",
    "trapezoid",
    "double_circle",
    "cloud",
    "stadium",
    "tab",
    "note",
    "document",
    "box3d",
)
FEATURE_NAMES: Tuple[str, ...] = (
    "borders",
    "fills",
    "text",
    "edges",
    "clusters",
    "effects",
    "pie_charts",
)
NON_INSETTABLE_SHAPES = {
    "double_circle",
    "cloud",
    "stadium",
    "tab",
    "note",
    "document",
    "box3d",
}
SHAPE_FILE_OVERRIDES: Mapping[str, str] = {
    "rect": "rectangle",
    "roundrect": "rounded_rectangle",
}
DISPLAY_NAME_OVERRIDES: Mapping[str, str] = {
    "rect": "Rectangle",
    "roundrect": "Rounded Rectangle",
    "double_circle": "Double Circle",
    "box3d": "3D Box",
    "igraph_r": "igraph R",
    "graph_tool": "graph-tool",
    "n8n": "n8n",
    "visjs": "Vis.js",
    "drawio": "draw.io",
    "neo4j": "Neo4j",
}


@dataclass(frozen=True)
class DemoScene:
    """One rendered gallery scene.

    Parameters
    ----------
    graph : DaguaGraph
        Graph configured for rendering.
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    title : str
        In-image title shown in the overlay card.
    subtitle : str
        Smaller descriptive line shown below ``title``.
    """

    graph: DaguaGraph
    positions: torch.Tensor
    title: str
    subtitle: str


@dataclass(frozen=True)
class GalleryItem:
    """One card in the generated HTML gallery.

    Parameters
    ----------
    label : str
        Human-readable card label.
    relative_path : str
        Image path relative to the gallery root.
    """

    label: str
    relative_path: str


@dataclass(frozen=True)
class GallerySection:
    """One logical section in the generated HTML.

    Parameters
    ----------
    slug : str
        Stable anchor ID.
    title : str
        Section heading.
    items : tuple[GalleryItem, ...]
        Rendered cards in display order.
    """

    slug: str
    title: str
    items: Tuple[GalleryItem, ...]


@dataclass(frozen=True)
class GalleryBuildResult:
    """Paths and metadata emitted by the gallery builder.

    Parameters
    ----------
    output_dir : str
        Root output directory.
    index_path : str
        HTML index path.
    sections : tuple[GallerySection, ...]
        Rendered gallery sections in display order.
    """

    output_dir: str
    index_path: str
    sections: Tuple[GallerySection, ...]


def _friendly_name(name: str) -> str:
    """Return a human-readable display label.

    Parameters
    ----------
    name : str
        Internal slug such as ``"double_circle"``.

    Returns
    -------
    str
        Display label suitable for HTML and image overlays.
    """

    if name in DISPLAY_NAME_OVERRIDES:
        return DISPLAY_NAME_OVERRIDES[name]
    return name.replace("_", " ").title()


def _shape_filename(shape_name: str) -> str:
    """Return the output filename stem for a node shape.

    Parameters
    ----------
    shape_name : str
        Internal Dagua shape name.

    Returns
    -------
    str
        Filename stem without extension.
    """

    return SHAPE_FILE_OVERRIDES.get(shape_name, shape_name)


def _relative_path(root: Path, target: Path) -> str:
    """Return a POSIX relative path for HTML output.

    Parameters
    ----------
    root : Path
        Gallery root directory.
    target : Path
        Asset path below ``root``.

    Returns
    -------
    str
        Relative POSIX path.
    """

    return target.relative_to(root).as_posix()


def _validate_requested_names(
    requested: Optional[Sequence[str]],
    allowed: Sequence[str],
    label: str,
) -> Tuple[str, ...]:
    """Validate a requested subset against the supported names.

    Parameters
    ----------
    requested : Sequence[str] | None
        Optional user-specified subset.
    allowed : Sequence[str]
        Supported names in canonical order.
    label : str
        Error-label prefix such as ``"shape"``.

    Returns
    -------
    tuple[str, ...]
        Canonical ordered subset.

    Raises
    ------
    ValueError
        If any requested value is unsupported.
    """

    if requested is None:
        return tuple(allowed)

    allowed_set = set(allowed)
    unknown = [name for name in requested if name not in allowed_set]
    if unknown:
        raise ValueError(f"Unknown {label} names: {unknown!r}")

    requested_set = set(requested)
    return tuple(name for name in allowed if name in requested_set)


def _clear_directory(directory: Path) -> None:
    """Remove all existing content inside a managed output directory.

    Parameters
    ----------
    directory : Path
        Managed directory to clear.

    Returns
    -------
    None
        The directory is recreated empty.
    """

    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


def _prepare_output_directory(output_dir: Path) -> Dict[str, Path]:
    """Create and clear the gallery output tree.

    Parameters
    ----------
    output_dir : Path
        Gallery root directory.

    Returns
    -------
    dict[str, Path]
        Named subdirectories for the generated assets.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / INDEX_NAME).exists():
        (output_dir / INDEX_NAME).unlink()

    managed_dirs = {
        "shapes": output_dir / "shapes",
        "arrows": output_dir / "arrows",
        "themes": output_dir / "themes",
        "features": output_dir / "features",
    }
    for directory in managed_dirs.values():
        _clear_directory(directory)
    return managed_dirs


def _reference_graph_style(background_color: str) -> GraphStyle:
    """Return the fixed graph-level style used for reference demos.

    Parameters
    ----------
    background_color : str
        Graph background color.

    Returns
    -------
    GraphStyle
        Minimal graph styling with fixed canvas constraints.
    """

    return GraphStyle(
        background_color=background_color,
        margin=20.0,
        min_figsize=IMAGE_SIZE_IN,
        max_figsize=IMAGE_SIZE_IN,
    )


def _apply_reference_canvas(graph: DaguaGraph, background_color: str = WHITE) -> None:
    """Force a clean render canvas for non-theme demo graphs.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to update.
    background_color : str, default=WHITE
        Canvas background color.

    Returns
    -------
    None
        The graph theme is updated in place.
    """

    graph._theme.graph_style = _reference_graph_style(background_color)


def _positions(points: Sequence[Tuple[float, float]]) -> torch.Tensor:
    """Build a float32 position tensor from Python tuples.

    Parameters
    ----------
    points : Sequence[tuple[float, float]]
        Node positions in graph order.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """

    return torch.tensor(points, dtype=torch.float32)


def _base_node_style(base_color: str, shape: str = "roundrect") -> NodeStyle:
    """Return a readable node style for small documentation scenes.

    Parameters
    ----------
    base_color : str
        Base color used to derive fill and stroke.
    shape : str, default="roundrect"
        Node shape name.

    Returns
    -------
    NodeStyle
        Node style tuned for small gallery renders.
    """

    return NodeStyle(
        base_color=base_color,
        shape=shape,
        stroke_width=1.4,
        min_width=82.0,
        min_height=46.0,
        padding=(12.0, 9.0),
        shadow=False,
    )


def _accent_edge_style(color: str, **overrides: object) -> EdgeStyle:
    """Return a readable edge style for highlighted examples.

    Parameters
    ----------
    color : str
        Edge color.
    **overrides : object
        Extra ``EdgeStyle`` field overrides.

    Returns
    -------
    EdgeStyle
        Edge style with the requested overrides applied.
    """

    style_kwargs: Dict[str, object] = {"color": color, "width": 2.4, "opacity": 0.95}
    style_kwargs.update(overrides)
    return EdgeStyle(**style_kwargs)


def _hex_luminance(hex_color: str) -> float:
    """Estimate relative luminance for a hex color.

    Parameters
    ----------
    hex_color : str
        Color in ``#RRGGBB`` or ``#RRGGBBAA`` form.

    Returns
    -------
    float
        Relative luminance on ``[0, 1]``.
    """

    color = hex_color.lstrip("#")
    if len(color) < 6:
        return 1.0
    red = int(color[0:2], 16) / 255.0
    green = int(color[2:4], 16) / 255.0
    blue = int(color[4:6], 16) / 255.0
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _overlay_colors(background_color: str) -> Tuple[str, str, str]:
    """Choose overlay colors that contrast with the render background.

    Parameters
    ----------
    background_color : str
        Scene background color.

    Returns
    -------
    tuple[str, str, str]
        Text color, box fill color, and box border color.
    """

    if _hex_luminance(background_color) < 0.45:
        return "#F8FAFC", "#111827CC", "#E5E7EB44"
    return TEXT_COLOR, "#FFFFFFE6", "#D1D5DB"


def _annotate_scene(ax: Axes, title: str, subtitle: str, background_color: str) -> None:
    """Add a compact documentation overlay to a rendered axes.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes returned by the renderer.
    title : str
        Primary title line.
    subtitle : str
        Secondary explanatory line.
    background_color : str
        Scene background color used for contrast selection.

    Returns
    -------
    None
        The axes are annotated in place.
    """

    text_color, box_fill, box_edge = _overlay_colors(background_color)
    ax.text(
        0.03,
        0.96,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13.0,
        fontweight="bold",
        color=text_color,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": box_fill,
            "edgecolor": box_edge,
            "linewidth": 0.9,
        },
        zorder=20,
    )
    ax.text(
        0.03,
        0.885,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.6,
        color=text_color,
        zorder=21,
    )


def _save_figure(fig: Figure, output_path: Path, background_color: str) -> None:
    """Save one scene at the required fixed pixel size.

    Parameters
    ----------
    fig : Figure
        Figure to save.
    output_path : Path
        Destination PNG path.
    background_color : str
        Figure background color.

    Returns
    -------
    None
        The figure is saved and closed.

    Raises
    ------
    ValueError
        If the generated PNG is empty.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=IMAGE_DPI,
        facecolor=background_color,
        edgecolor=background_color,
        transparent=False,
    )
    plt.close(fig)
    if output_path.stat().st_size <= 0:
        raise ValueError(f"Generated empty PNG: {output_path}")


def _render_scene(scene: DemoScene, output_path: Path) -> None:
    """Render and save one documentation scene.

    Parameters
    ----------
    scene : DemoScene
        Scene definition to render.
    output_path : Path
        Destination PNG path.

    Returns
    -------
    None
        The rendered scene is written to disk.
    """

    figure, axes = render(
        scene.graph,
        positions=scene.positions,
        figsize=IMAGE_SIZE_IN,
        dpi=IMAGE_DPI,
        show=False,
    )
    background_color = scene.graph.graph_style.background_color
    _annotate_scene(axes, scene.title, scene.subtitle, background_color)
    _save_figure(figure, output_path, background_color)


def _build_shape_scene(shape_name: str) -> DemoScene:
    """Build the reference graph for one node shape.

    Parameters
    ----------
    shape_name : str
        Supported Dagua node shape name.

    Returns
    -------
    DemoScene
        Render-ready scene for the requested shape.
    """

    graph = DaguaGraph(direction="LR")
    _apply_reference_canvas(graph)
    graph.add_node("input", label="Input", style=_base_node_style(PALETTE["bluish_green"]))
    stroke_width = 0.0 if shape_name in NON_INSETTABLE_SHAPES else 1.6
    graph.add_node(
        "feature",
        label="Feature",
        style=NodeStyle(
            base_color=PALETTE["blue"],
            shape=shape_name,
            stroke_width=stroke_width,
            min_width=114.0,
            min_height=74.0,
            padding=(14.0, 11.0),
            shadow=True,
            shadow_offset=(2.5, -2.5),
            shadow_color="#00000022",
        ),
    )
    graph.add_node("output", label="Output", style=_base_node_style(PALETTE["amber"]))
    graph.add_edge("input", "feature", style=_accent_edge_style("#6B7280", arrow="none"))
    graph.add_edge("feature", "output", style=_accent_edge_style(PALETTE["blue"], arrow="vee"))
    return DemoScene(
        graph=graph,
        positions=_positions([(-150.0, 0.0), (0.0, 0.0), (150.0, 0.0)]),
        title=_friendly_name(shape_name),
        subtitle="Node shape reference",
    )


def _build_arrow_scene(arrow_name: str) -> DemoScene:
    """Build the reference graph for one arrowhead type.

    Parameters
    ----------
    arrow_name : str
        Supported arrowhead name.

    Returns
    -------
    DemoScene
        Render-ready scene for the requested arrowhead.
    """

    graph = DaguaGraph(direction="LR")
    _apply_reference_canvas(graph)
    muted_edge = _accent_edge_style("#A0AEC0", arrow="none", width=1.8, opacity=0.75)
    highlight_edge = _accent_edge_style(
        PALETTE["vermillion"],
        arrow=arrow_name,
        width=2.8,
        curvature=0.15,
    )
    graph.add_node("draft", label="Draft", style=_base_node_style(PALETTE["sky"]))
    graph.add_node("review", label="Review", style=_base_node_style(PALETTE["blue"]))
    graph.add_node("release", label="Release", style=_base_node_style(PALETTE["bluish_green"]))
    graph.add_edge("draft", "review", style=muted_edge)
    graph.add_edge("review", "release", label="highlight", style=highlight_edge)
    return DemoScene(
        graph=graph,
        positions=_positions([(-150.0, 25.0), (0.0, 0.0), (150.0, 25.0)]),
        title=_friendly_name(arrow_name),
        subtitle="Arrowhead reference",
    )


def _build_theme_scene(theme_name: str) -> DemoScene:
    """Build the shared reference graph used for one theme.

    Parameters
    ----------
    theme_name : str
        Built-in theme registry key.

    Returns
    -------
    DemoScene
        Render-ready scene for the requested theme.
    """

    graph = DaguaGraph(direction="LR")
    graph._theme = get_theme(theme_name)
    for style in graph._theme.node_styles.values():
        if style.shape in NON_INSETTABLE_SHAPES:
            style.stroke_width = 0.0
    graph.add_node("Input", label="Input", type="input")
    graph.add_node("Process", label="Process")
    graph.add_node("Output", label="Output", type="output")
    graph.add_node("Cache", label="Cache", type="buffer")
    graph.add_edge("Input", "Process")
    graph.add_edge("Process", "Output")
    graph.add_edge("Input", "Cache")
    graph.add_edge("Cache", "Output")
    return DemoScene(
        graph=graph,
        positions=_positions([(-150.0, 70.0), (0.0, 70.0), (150.0, 70.0), (0.0, -70.0)]),
        title=_friendly_name(theme_name),
        subtitle="Built-in theme",
    )


def _build_borders_scene() -> DemoScene:
    """Build the border style reference scene.

    Returns
    -------
    DemoScene
        Render-ready border example scene.
    """

    graph = DaguaGraph(direction="LR")
    _apply_reference_canvas(graph)
    labels_and_dashes = (
        ("Solid", "solid", PALETTE["blue"]),
        ("Dashed", "dashed", PALETTE["amber"]),
        ("Dotted", "dotted", PALETTE["vermillion"]),
    )
    for label, dash, color in labels_and_dashes:
        graph.add_node(
            label.lower(),
            label=label,
            style=NodeStyle(
                base_color=color,
                stroke_dash=dash,
                stroke_width=1.7,
                min_width=106.0,
                min_height=58.0,
            ),
        )
    return DemoScene(
        graph=graph,
        positions=_positions([(-150.0, 0.0), (0.0, 0.0), (150.0, 0.0)]),
        title="Border styles",
        subtitle="Solid, dashed, and dotted node outlines",
    )


def _build_fills_scene() -> DemoScene:
    """Build the fill style reference scene.

    Returns
    -------
    DemoScene
        Render-ready fill example scene.
    """

    graph = DaguaGraph(direction="LR")
    _apply_reference_canvas(graph)
    graph.add_node("solid", label="Solid", style=_base_node_style(PALETTE["sky"]))
    graph.add_node(
        "gradient",
        label="Gradient",
        style=NodeStyle(
            base_color=PALETTE["blue"],
            gradient="linear",
            gradient_color="#9FC7EE",
            gradient_angle=90.0,
            min_width=110.0,
            min_height=58.0,
        ),
    )
    graph.add_node(
        "pattern",
        label="Pattern",
        style=NodeStyle(
            base_color=PALETTE["amber"],
            fill_pattern="striped",
            fill_pattern_colors=[make_fill(PALETTE["amber"]), "#F7D88A"],
            fill_pattern_angle=28.0,
            min_width=110.0,
            min_height=58.0,
        ),
    )
    return DemoScene(
        graph=graph,
        positions=_positions([(-150.0, 0.0), (0.0, 0.0), (150.0, 0.0)]),
        title="Fill treatments",
        subtitle="Solid, gradient, and patterned nodes",
    )


def _build_text_scene() -> DemoScene:
    """Build the typography reference scene.

    Returns
    -------
    DemoScene
        Render-ready text feature scene.
    """

    graph = DaguaGraph(direction="TB")
    _apply_reference_canvas(graph)
    graph.add_node(
        "bold",
        label="Bold",
        style=NodeStyle(base_color=PALETTE["blue"], font_weight="bold", min_width=92.0),
    )
    graph.add_node(
        "italic",
        label="Italic",
        style=NodeStyle(base_color=PALETTE["amber"], font_style="italic", min_width=92.0),
    )
    graph.add_node(
        "wrap",
        label="Wrapped labels stay readable",
        style=NodeStyle(
            base_color=PALETTE["bluish_green"],
            text_wrap="wrap",
            text_max_width=80.0,
            min_width=98.0,
            min_height=64.0,
        ),
    )
    graph.add_node(
        "outline",
        label="Outline",
        style=NodeStyle(
            base_color=PALETTE["reddish_purple"],
            text_transform="uppercase",
            text_outline=True,
            text_outline_color="#FFFFFF",
            text_outline_width=2.2,
            min_width=100.0,
        ),
    )
    return DemoScene(
        graph=graph,
        positions=_positions([(-120.0, 65.0), (120.0, 65.0), (-120.0, -65.0), (120.0, -65.0)]),
        title="Text features",
        subtitle="Bold, italic, wrapped, and outlined labels",
    )


def _build_edges_scene() -> DemoScene:
    """Build the edge feature reference scene.

    Returns
    -------
    DemoScene
        Render-ready edge feature scene.
    """

    graph = DaguaGraph(direction="LR")
    _apply_reference_canvas(graph)
    for node_id, label, color in (
        ("input", "Input", PALETTE["bluish_green"]),
        ("rules", "Rules", PALETTE["blue"]),
        ("queue", "Queue", PALETTE["amber"]),
        ("output", "Output", PALETTE["vermillion"]),
    ):
        graph.add_node(node_id, label=label, style=_base_node_style(color))
    graph.add_edge(
        "input",
        "rules",
        label="bezier",
        style=_accent_edge_style(PALETTE["blue"], routing="bezier", curvature=0.42),
    )
    graph.add_edge(
        "input",
        "queue",
        label="ortho",
        style=_accent_edge_style(PALETTE["amber"], routing="ortho", arrow="vee"),
    )
    graph.add_edge(
        "rules",
        "output",
        label="taper + gradient",
        style=_accent_edge_style(
            PALETTE["blue"],
            taper=True,
            taper_width_start=4.0,
            taper_width_end=0.8,
            color_gradient="source_to_target",
            color_gradient_end=PALETTE["vermillion"],
        ),
    )
    graph.add_edge(
        "queue",
        "output",
        label="taxi",
        style=_accent_edge_style(PALETTE["vermillion"], routing="taxi"),
    )
    return DemoScene(
        graph=graph,
        positions=_positions([(-170.0, 25.0), (-30.0, 90.0), (-30.0, -85.0), (165.0, 0.0)]),
        title="Edge styling",
        subtitle="Routing, tapering, and color transitions",
    )


def _build_clusters_scene() -> DemoScene:
    """Build the nested cluster reference scene.

    Returns
    -------
    DemoScene
        Render-ready cluster feature scene.
    """

    graph = DaguaGraph(direction="LR")
    _apply_reference_canvas(graph)
    graph.add_node("ingest", label="Ingest", style=_base_node_style(PALETTE["sky"]))
    graph.add_node("validate", label="Validate", style=_base_node_style(PALETTE["blue"]))
    graph.add_node("train", label="Train", style=_base_node_style(PALETTE["amber"]))
    graph.add_node("serve", label="Serve", style=_base_node_style(PALETTE["bluish_green"]))
    graph.add_node("monitor", label="Monitor", style=_base_node_style(PALETTE["vermillion"]))
    graph.add_edge("ingest", "validate", style=_accent_edge_style("#6B7280", arrow="none"))
    graph.add_edge("validate", "train", style=_accent_edge_style(PALETTE["blue"]))
    graph.add_edge("train", "serve", style=_accent_edge_style(PALETTE["amber"]))
    graph.add_edge("serve", "monitor", style=_accent_edge_style(PALETTE["bluish_green"]))

    platform_style = ClusterStyle(
        fill=make_fill(PALETTE["blue"], bg_hex=WHITE, blend=0.10),
        stroke=border_from_fill(PALETTE["blue"], darken=0.10),
        opacity=0.28,
        padding=44.0,
    )
    prep_style = ClusterStyle(
        fill=make_fill(PALETTE["amber"], bg_hex=WHITE, blend=0.12),
        stroke=border_from_fill(PALETTE["amber"], darken=0.18),
        opacity=0.34,
        padding=36.0,
    )
    serving_style = ClusterStyle(
        fill=make_fill(PALETTE["bluish_green"], bg_hex=WHITE, blend=0.10),
        stroke=border_from_fill(PALETTE["bluish_green"], darken=0.18),
        opacity=0.32,
        padding=36.0,
    )
    graph.add_cluster(
        "platform",
        ["ingest", "validate", "train", "serve", "monitor"],
        label="Platform",
        style=platform_style,
    )
    graph.add_cluster(
        "prep",
        ["ingest", "validate", "train"],
        label="Prep",
        style=prep_style,
        parent="platform",
    )
    graph.add_cluster(
        "serving",
        ["serve", "monitor"],
        label="Serving",
        style=serving_style,
        parent="platform",
    )
    return DemoScene(
        graph=graph,
        positions=_positions(
            [(-170.0, 10.0), (-70.0, 10.0), (35.0, 10.0), (135.0, 65.0), (135.0, -55.0)]
        ),
        title="Cluster nesting",
        subtitle="Parent and child groups with independent styling",
    )


def _build_effects_scene() -> DemoScene:
    """Build the effects reference scene.

    Returns
    -------
    DemoScene
        Render-ready effects scene.
    """

    graph = DaguaGraph(direction="LR")
    _apply_reference_canvas(graph)
    graph.add_node(
        "shadow",
        label="Shadow",
        style=NodeStyle(
            base_color=PALETTE["blue"],
            shadow=True,
            shadow_offset=(3.0, -3.0),
            shadow_color="#00000033",
            min_width=98.0,
        ),
    )
    graph.add_node("cross_a", label="Cross A", style=_base_node_style(PALETTE["sky"]))
    graph.add_node("cross_b", label="Cross B", style=_base_node_style(PALETTE["amber"]))
    graph.add_node(
        "opacity",
        label="Opacity",
        style=NodeStyle(base_color=PALETTE["vermillion"], opacity=0.58, min_width=98.0),
    )
    crossing_style = _accent_edge_style("#5F6C7B", crossing_style="arc", crossing_size=10.0)
    graph.add_edge("shadow", "opacity", label="crossing jump", style=crossing_style)
    graph.add_edge("cross_a", "cross_b", style=crossing_style)
    return DemoScene(
        graph=graph,
        positions=_positions([(-150.0, 90.0), (120.0, 90.0), (-120.0, -90.0), (150.0, -90.0)]),
        title="Effects",
        subtitle="Shadow, transparency, and crossing jumps",
    )


def _build_pie_charts_scene() -> DemoScene:
    """Build the pie and donut chart reference scene.

    Returns
    -------
    DemoScene
        Render-ready pie chart scene.
    """

    graph = DaguaGraph(direction="LR")
    _apply_reference_canvas(graph)
    pie_colors = ["#4F8BC9", "#76C3A5", "#F0B35A", "#E16B5A"]
    graph.add_node(
        "pie",
        label="Pie",
        style=NodeStyle(
            shape="circle",
            fill_pattern="pie",
            fill_pattern_colors=pie_colors[:3],
            fill_pattern_values=[45.0, 35.0, 20.0],
            min_width=90.0,
            min_height=90.0,
        ),
    )
    graph.add_node(
        "donut",
        label="Donut",
        style=NodeStyle(
            shape="circle",
            fill_pattern="pie",
            fill_pattern_colors=pie_colors,
            fill_pattern_values=[35.0, 25.0, 20.0, 20.0],
            fill_pattern_hole=0.42,
            min_width=96.0,
            min_height=96.0,
        ),
    )
    graph.add_node(
        "split",
        label="Split",
        style=NodeStyle(
            shape="circle",
            fill_pattern="pie",
            fill_pattern_colors=pie_colors[:4],
            fill_pattern_values=[1.0, 1.0, 1.0, 1.0],
            min_width=92.0,
            min_height=92.0,
        ),
    )
    return DemoScene(
        graph=graph,
        positions=_positions([(-150.0, 0.0), (0.0, 0.0), (150.0, 0.0)]),
        title="Pie charts and donuts",
        subtitle="Categorical splits rendered directly inside nodes",
    )


FEATURE_BUILDERS: Mapping[str, Callable[[], DemoScene]] = {
    "borders": _build_borders_scene,
    "fills": _build_fills_scene,
    "text": _build_text_scene,
    "edges": _build_edges_scene,
    "clusters": _build_clusters_scene,
    "effects": _build_effects_scene,
    "pie_charts": _build_pie_charts_scene,
}


def _write_index_html(output_dir: Path, sections: Sequence[GallerySection]) -> None:
    """Write the gallery HTML index page.

    Parameters
    ----------
    output_dir : Path
        Gallery root directory.
    sections : Sequence[GallerySection]
        Sections to render in the index.

    Returns
    -------
    None
        The HTML page is written to disk.
    """

    nav_links = "\n".join(
        f'        <a href="#{html.escape(section.slug)}">{html.escape(section.title)}</a>'
        for section in sections
    )
    section_blocks: List[str] = []
    for section in sections:
        cards = "\n".join(
            (
                '        <article class="card">'
                f'<img src="{html.escape(item.relative_path)}" alt="{html.escape(item.label)}">'
                f'<div class="label">{html.escape(item.label)}</div>'
                "</article>"
            )
            for item in section.items
        )
        section_blocks.append(
            "\n".join(
                [
                    f'    <section id="{html.escape(section.slug)}" class="section">',
                    f"      <h2>{html.escape(section.title)}</h2>",
                    '      <div class="gallery">',
                    cards,
                    "      </div>",
                    "    </section>",
                ]
            )
        )

    html_text = "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            '  <meta charset="utf-8">',
            '  <meta name="viewport" content="width=device-width, initial-scale=1.0">',
            "  <title>Dagua Feature Reference</title>",
            "  <style>",
            "    :root { color-scheme: light; }",
            (
                "    body { font-family: system-ui, sans-serif; max-width: 1200px; "
                "margin: 0 auto; padding: 24px; color: #1f2933; background: #ffffff; }"
            ),
            "    h1 { margin: 0 0 8px; padding-bottom: 12px; border-bottom: 2px solid #1f2933; }",
            "    p.lede { margin: 0 0 20px; color: #5b6670; max-width: 70ch; line-height: 1.5; }",
            "    nav { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 28px; }",
            "    nav a { color: #24577a; text-decoration: none; font-weight: 600; }",
            "    nav a:hover { text-decoration: underline; }",
            "    .section { margin-bottom: 34px; }",
            "    .section h2 { margin: 0 0 14px; }",
            (
                "    .gallery { display: grid; grid-template-columns: repeat(auto-fill, "
                "minmax(300px, 1fr)); gap: 20px; }"
            ),
            (
                "    .card { border: 1px solid #e0e0e0; border-radius: 8px; "
                "overflow: hidden; background: #ffffff; "
                "box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05); }"
            ),
            "    .card img { width: 100%; display: block; background: #ffffff; }",
            "    .card .label { padding: 8px 12px; font-size: 14px; color: #333333; }",
            "  </style>",
            "</head>",
            "<body>",
            "  <h1>Dagua Feature Reference</h1>",
            (
                '  <p class="lede">A user-facing reference gallery covering supported '
                "node shapes, arrowheads, built-in themes, and the main cosmetic "
                "feature families available in Dagua.</p>"
            ),
            "  <nav>",
            nav_links,
            "  </nav>",
            *section_blocks,
            "</body>",
            "</html>",
        ]
    )
    (output_dir / INDEX_NAME).write_text(html_text, encoding="utf-8")


def build_feature_reference(
    output_dir: Union[str, Path],
    shapes: Optional[Sequence[str]] = None,
    arrowheads: Optional[Sequence[str]] = None,
    theme_names: Optional[Sequence[str]] = None,
    feature_names: Optional[Sequence[str]] = None,
) -> GalleryBuildResult:
    """Build the full documentation gallery on disk.

    Parameters
    ----------
    output_dir : str | Path
        Destination gallery root.
    shapes : Sequence[str] | None, default=None
        Optional subset of node shapes to render.
    arrowheads : Sequence[str] | None, default=None
        Optional subset of arrowheads to render.
    theme_names : Sequence[str] | None, default=None
        Optional subset of themes to render.
    feature_names : Sequence[str] | None, default=None
        Optional subset of feature scenes to render.

    Returns
    -------
    GalleryBuildResult
        Output paths and section metadata.
    """

    output_root = Path(output_dir)
    directories = _prepare_output_directory(output_root)
    selected_shapes = _validate_requested_names(shapes, SHAPE_NAMES, "shape")
    selected_arrowheads = _validate_requested_names(
        arrowheads,
        tuple(available_arrowheads()),
        "arrowhead",
    )
    selected_themes = _validate_requested_names(
        theme_names,
        tuple(THEME_REGISTRY.keys()),
        "theme",
    )
    selected_features = _validate_requested_names(feature_names, FEATURE_NAMES, "feature")

    sections: List[GallerySection] = []

    shape_items: List[GalleryItem] = []
    for shape_name in selected_shapes:
        target_path = directories["shapes"] / f"{_shape_filename(shape_name)}.png"
        _render_scene(_build_shape_scene(shape_name), target_path)
        shape_items.append(
            GalleryItem(
                label=_friendly_name(shape_name),
                relative_path=_relative_path(output_root, target_path),
            )
        )
    sections.append(GallerySection(slug="shapes", title="Shapes", items=tuple(shape_items)))

    arrow_items: List[GalleryItem] = []
    for arrow_name in selected_arrowheads:
        target_path = directories["arrows"] / f"{arrow_name}.png"
        _render_scene(_build_arrow_scene(arrow_name), target_path)
        arrow_items.append(
            GalleryItem(
                label=_friendly_name(arrow_name),
                relative_path=_relative_path(output_root, target_path),
            )
        )
    sections.append(GallerySection(slug="arrows", title="Arrowheads", items=tuple(arrow_items)))

    theme_items: List[GalleryItem] = []
    for theme_name in selected_themes:
        target_path = directories["themes"] / f"{theme_name}.png"
        _render_scene(_build_theme_scene(theme_name), target_path)
        theme_items.append(
            GalleryItem(
                label=_friendly_name(theme_name),
                relative_path=_relative_path(output_root, target_path),
            )
        )
    sections.append(GallerySection(slug="themes", title="Themes", items=tuple(theme_items)))

    feature_items: List[GalleryItem] = []
    for feature_name in selected_features:
        target_path = directories["features"] / f"{feature_name}.png"
        _render_scene(FEATURE_BUILDERS[feature_name](), target_path)
        feature_items.append(
            GalleryItem(
                label=_friendly_name(feature_name),
                relative_path=_relative_path(output_root, target_path),
            )
        )
    sections.append(
        GallerySection(slug="features", title="Feature Examples", items=tuple(feature_items))
    )

    _write_index_html(output_root, sections)
    return GalleryBuildResult(
        output_dir=str(output_root),
        index_path=str(output_root / INDEX_NAME),
        sections=tuple(sections),
    )


# ==============================================================================
# v2: matrix/manifest/ledger-driven docs/visual_reference/ generator
# (FINAL_DESIGN.md section 9, Lane D item 5)
#
# Unlike the v1 gallery above (which hand-builds DemoScene specimens), v2 is
# purely a READER: it never renders new images. It reads
# card_manifest.json + coverage_matrix.json + ledger.json (all
# schema-version-checked via scripts.visual_parity.io) and fills competitor
# side-by-side slots from the parity loop's own refcache -- "the loop's
# cache IS the guide's comparison content, zero extra render cost." Layout
# rules (Sol, section 9): no marketing hero -- first screen is index/TOC +
# compact status summary; mobile-first (2-column pairs collapse to stacked
# Reference/Dagua); simple sections/tables/grids, no nested cards.
# ==============================================================================

V2_DOMAINS: Tuple[str, ...] = (
    "shapes",
    "arrowheads",
    "edges",
    "fills",
    "text",
    "clusters",
    "themes",
    "other",
)

DEFAULT_V2_CARD_MANIFEST = ".project-context/research/sprint_visual_parity_v2/card_manifest.json"
DEFAULT_V2_COVERAGE_MATRIX = (
    ".project-context/research/sprint_visual_parity_v2/coverage_matrix.json"
)
DEFAULT_V2_LEDGER = ".project-context/research/sprint_visual_parity_v2/ledger.json"
DEFAULT_V2_REFCACHE = "eval_output/visual_parity_v2/refcache"
DEFAULT_V2_OUTPUT_DIR = "docs/visual_reference"
DEFAULT_V2_MARKDOWN_INDEX = "docs/VISUAL_REFERENCE.md"


@dataclass(frozen=True)
class V2CardEntry:
    """One resolved card row ready for v2 page rendering.

    Parameters
    ----------
    case_id
        Stable case identifier.
    category
        Source category (manifest ``category`` field).
    description
        Human-readable description.
    domain
        Classified v2 domain page (see ``V2_DOMAINS``).
    status_badge
        Short status label derived from the coverage matrix, if a matching
        cell exists (``"untested"`` otherwise).
    departure_text
        Waiver/residual text, if any.
    competitor_image
        Relative path (under the output directory) to a copied competitor
        specimen image, or ``None`` when no refcache entry exists yet.
    """

    case_id: str
    category: str
    description: str
    domain: str
    status_badge: str
    departure_text: str
    competitor_image: Optional[str]


@dataclass(frozen=True)
class V2BuildResult:
    """Output metadata for a v2 visual reference guide build.

    Parameters
    ----------
    output_dir
        Guide root directory.
    index_path
        Path to the generated ``index.html``.
    markdown_index_path
        Path to the generated ``docs/VISUAL_REFERENCE.md``.
    page_paths
        Domain slug to generated HTML page path.
    domain_counts
        Domain slug to card count.
    filled_competitor_slots
        Number of cards with a resolved competitor specimen image.
    """

    output_dir: str
    index_path: str
    markdown_index_path: str
    page_paths: Dict[str, str]
    domain_counts: Dict[str, int]
    filled_competitor_slots: int


def _classify_v2_domain(card: Mapping[str, object]) -> str:
    """Classify a card manifest row into a v2 domain page.

    Parameters
    ----------
    card
        One row from ``card_manifest.json``.

    Returns
    -------
    str
        A member of ``V2_DOMAINS``.
    """

    category = str(card.get("category", ""))
    case_id = str(card.get("case_id", ""))
    kind = card.get("kind")

    if case_id.startswith("arrowhead_") or "arrowhead" in category:
        return "arrowheads"
    if category == "node_options" and case_id.startswith("shape_"):
        return "shapes"
    if "shape" in category or (kind == "reference" and "shape" in category.lower()):
        return "shapes"
    if "fill" in category or "opacity" in category or "color" in category:
        return "fills"
    if category == "text_options" or "text" in category or "label" in category:
        return "text"
    if category == "cluster_options" or "cluster" in category:
        return "clusters"
    if category == "edge_options" or "edge" in category or "spline" in category:
        return "edges"
    if "theme" in category:
        return "themes"
    return "other"


def _status_badge_for_cell(
    coverage_cell_id: Optional[str],
    cells_by_id: Mapping[str, Mapping[str, object]],
) -> str:
    """Return a short status badge string for a coverage cell.

    Parameters
    ----------
    coverage_cell_id
        Linked coverage cell id, if any.
    cells_by_id
        Coverage matrix cells keyed by ``cell_id``.

    Returns
    -------
    str
        A short badge label, e.g. ``"in_tolerance"`` or ``"untested"``.
    """

    if not coverage_cell_id:
        return "untested"
    cell = cells_by_id.get(coverage_cell_id)
    if cell is None:
        return "untested"
    return str(cell.get("parity_status", "untested"))


def _departure_text_for_cell(
    coverage_cell_id: Optional[str],
    cells_by_id: Mapping[str, Mapping[str, object]],
) -> str:
    """Return waiver/residual departure text for a coverage cell, if any.

    Parameters
    ----------
    coverage_cell_id
        Linked coverage cell id, if any.
    cells_by_id
        Coverage matrix cells keyed by ``cell_id``.

    Returns
    -------
    str
        Human-readable departure text, or an empty string.
    """

    if not coverage_cell_id:
        return ""
    cell = cells_by_id.get(coverage_cell_id)
    if cell is None:
        return ""
    waiver = cell.get("waiver")
    if isinstance(waiver, Mapping) and waiver.get("user_visible_impact"):
        return str(waiver["user_visible_impact"])
    return ""


def _resolve_competitor_image(
    case_id: str,
    refcache_root: Path,
    output_dir: Path,
) -> Optional[str]:
    """Copy a refcache competitor specimen into the guide and return its path.

    Parameters
    ----------
    case_id
        Card case id to look up under ``refcache_root/graphviz/{case_id}.png``.
    refcache_root
        Root of the parity loop's refcache directory.
    output_dir
        Guide output root; images are copied to ``{output_dir}/img/``.

    Returns
    -------
    str or None
        Relative path (from ``output_dir``) to the copied image, or ``None``
        when no refcache entry exists for this case.
    """

    source = refcache_root / "graphviz" / f"{case_id}.png"
    if not source.exists():
        return None
    dest_dir = output_dir / "img"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{case_id}.png"
    shutil.copyfile(source, dest)
    return f"img/{case_id}.png"


def _load_v2_cards(
    card_manifest_path: Union[str, Path],
    coverage_matrix_path: Union[str, Path],
    refcache_root: Union[str, Path],
    output_dir: Path,
) -> List[V2CardEntry]:
    """Load and resolve card manifest rows into v2 page entries.

    Parameters
    ----------
    card_manifest_path
        Path to ``card_manifest.json``.
    coverage_matrix_path
        Path to ``coverage_matrix.json``.
    refcache_root
        Root of the parity loop's refcache directory.
    output_dir
        Guide output root (competitor images are copied here).

    Returns
    -------
    list[V2CardEntry]
        Resolved entries in manifest order.
    """

    from scripts.visual_parity.io import read_card_manifest, read_coverage_matrix

    manifest = read_card_manifest(card_manifest_path)
    matrix = read_coverage_matrix(coverage_matrix_path)
    cells_by_id = {
        str(cell.get("cell_id")): cell
        for cell in matrix.get("cells", [])
        if isinstance(cell, Mapping)
    }
    refcache_path = Path(refcache_root)

    entries: List[V2CardEntry] = []
    for card in manifest.get("cards", []):
        if not isinstance(card, Mapping):
            continue
        case_id = str(card.get("case_id", ""))
        coverage_cell_id = card.get("coverage_cell_id")
        entries.append(
            V2CardEntry(
                case_id=case_id,
                category=str(card.get("category", "")),
                description=str(card.get("description", "")),
                domain=_classify_v2_domain(card),
                status_badge=_status_badge_for_cell(coverage_cell_id, cells_by_id),
                departure_text=_departure_text_for_cell(coverage_cell_id, cells_by_id),
                competitor_image=_resolve_competitor_image(case_id, refcache_path, output_dir),
            )
        )
    return entries


def _write_v2_domain_page(output_dir: Path, domain: str, entries: Sequence[V2CardEntry]) -> Path:
    """Write one mobile-first HTML page for a v2 domain.

    Parameters
    ----------
    output_dir
        Guide output root.
    domain
        Domain slug (member of ``V2_DOMAINS``).
    entries
        Cards belonging to this domain.

    Returns
    -------
    Path
        Path to the written HTML page.
    """

    rows: List[str] = []
    for entry in entries:
        if entry.competitor_image:
            reference_cell = f'<img src="{html.escape(entry.competitor_image)}" alt="reference">'
        else:
            reference_cell = '<span class="missing">no reference yet</span>'
        departure_html = (
            f'<p class="departure">{html.escape(entry.departure_text)}</p>'
            if entry.departure_text
            else ""
        )
        rows.append(
            "\n".join(
                [
                    '      <div class="pair">',
                    f"        <h3>{html.escape(entry.case_id)}</h3>",
                    f'        <p class="desc">{html.escape(entry.description)}</p>',
                    f'        <span class="badge">{html.escape(entry.status_badge)}</span>',
                    '        <div class="cols">',
                    f'          <div class="col"><h4>Reference</h4>{reference_cell}</div>',
                    '          <div class="col"><h4>Dagua</h4>'
                    '<span class="missing">no specimen yet</span></div>',
                    "        </div>",
                    departure_html,
                    "      </div>",
                ]
            )
        )

    page_html = "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            '  <meta charset="utf-8">',
            '  <meta name="viewport" content="width=device-width, initial-scale=1.0">',
            f"  <title>Dagua Visual Reference -- {html.escape(domain)}</title>",
            "  <style>",
            "    :root { color-scheme: light; }",
            "    body { font-family: system-ui, sans-serif; max-width: 900px; "
            "margin: 0 auto; padding: 16px; color: #1f2933; }",
            "    h1 { font-size: 20px; }",
            "    .pair { border-top: 1px solid #e0e0e0; padding: 12px 0; }",
            "    .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }",
            "    @media (max-width: 600px) { .cols { grid-template-columns: 1fr; } }",
            "    .col img { max-width: 100%; }",
            "    .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; "
            "background: #eef2f6; font-size: 12px; }",
            "    .missing { color: #9aa5b1; font-size: 13px; }",
            "    .departure { color: #92400e; font-size: 13px; }",
            "    a.back { display: inline-block; margin-bottom: 12px; }",
            "  </style>",
            "</head>",
            "<body>",
            '  <a class="back" href="index.html">&larr; Index</a>',
            f"  <h1>{html.escape(domain)} ({len(entries)})</h1>",
            *rows,
            "</body>",
            "</html>",
        ]
    )
    page_path = output_dir / f"{domain}.html"
    page_path.write_text(page_html, encoding="utf-8")
    return page_path


def _write_v2_index_html(
    output_dir: Path,
    domain_counts: Mapping[str, int],
    filled_competitor_slots: int,
    total_cards: int,
) -> Path:
    """Write the v2 guide's index page (TOC + compact status summary, no hero).

    Parameters
    ----------
    output_dir
        Guide output root.
    domain_counts
        Domain slug to card count.
    filled_competitor_slots
        Number of cards with a resolved competitor specimen image.
    total_cards
        Total number of cards across all domains.

    Returns
    -------
    Path
        Path to the written ``index.html``.
    """

    rows = "\n".join(
        f'    <li><a href="{html.escape(domain)}.html">{html.escape(domain)}</a> ({count})</li>'
        for domain, count in domain_counts.items()
        if count > 0
    )
    page_html = "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            '  <meta charset="utf-8">',
            '  <meta name="viewport" content="width=device-width, initial-scale=1.0">',
            "  <title>Dagua Visual Reference</title>",
            "  <style>",
            "    body { font-family: system-ui, sans-serif; max-width: 700px; "
            "margin: 0 auto; padding: 16px; color: #1f2933; }",
            "    h1 { font-size: 20px; }",
            "    ul { padding-left: 20px; }",
            "    .status { font-size: 13px; color: #5b6670; }",
            "  </style>",
            "</head>",
            "<body>",
            "  <h1>Dagua Visual Reference</h1>",
            f'  <p class="status">{total_cards} cards indexed; '
            f"{filled_competitor_slots} with a filled reference specimen.</p>",
            "  <ul>",
            rows,
            "  </ul>",
            "</body>",
            "</html>",
        ]
    )
    index_path = output_dir / INDEX_NAME
    index_path.write_text(page_html, encoding="utf-8")
    return index_path


def _write_v2_markdown_index(
    markdown_path: Union[str, Path],
    output_dir: Path,
    domain_counts: Mapping[str, int],
    filled_competitor_slots: int,
    total_cards: int,
) -> None:
    """Write the ``docs/VISUAL_REFERENCE.md`` markdown index.

    Parameters
    ----------
    markdown_path
        Destination markdown path.
    output_dir
        Guide HTML output root (linked from the markdown index).
    domain_counts
        Domain slug to card count.
    filled_competitor_slots
        Number of cards with a resolved competitor specimen image.
    total_cards
        Total number of cards across all domains.

    Returns
    -------
    None
        The markdown file is written to disk.
    """

    markdown_dir = Path(markdown_path).resolve().parent
    relative_dir = Path(os.path.relpath(Path(output_dir).resolve(), markdown_dir)).as_posix()
    lines = [
        "# Dagua Visual Reference",
        "",
        f"{total_cards} cards indexed; {filled_competitor_slots} with a filled "
        "reference specimen from the visual parity v2 refcache.",
        "",
        "| domain | cards | page |",
        "|---|---|---|",
    ]
    for domain, count in domain_counts.items():
        if count == 0:
            continue
        lines.append(f"| {domain} | {count} | [{domain}.html]({relative_dir}/{domain}.html) |")
    lines.append("")
    lines.append(f"Full browsable index: [{relative_dir}/index.html]({relative_dir}/index.html)")
    lines.append("")
    Path(markdown_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_feature_reference_v2(
    output_dir: Union[str, Path] = DEFAULT_V2_OUTPUT_DIR,
    card_manifest_path: Union[str, Path] = DEFAULT_V2_CARD_MANIFEST,
    coverage_matrix_path: Union[str, Path] = DEFAULT_V2_COVERAGE_MATRIX,
    ledger_path: Union[str, Path] = DEFAULT_V2_LEDGER,
    refcache_root: Union[str, Path] = DEFAULT_V2_REFCACHE,
    markdown_index_path: Union[str, Path] = DEFAULT_V2_MARKDOWN_INDEX,
) -> V2BuildResult:
    """Build the v2 matrix/manifest/ledger-driven visual reference guide.

    This never renders new images -- it reads card_manifest.json +
    coverage_matrix.json (+ ledger.json, reserved for future ratchet/lock
    annotations) and fills competitor slots from the parity loop's refcache.

    Parameters
    ----------
    output_dir
        Guide HTML output root (``docs/visual_reference`` by default).
    card_manifest_path
        Path to ``card_manifest.json``.
    coverage_matrix_path
        Path to ``coverage_matrix.json``.
    ledger_path
        Path to ``ledger.json`` (read for schema validation; not yet used
        for per-row ratchet/lock annotations -- reserved for a future pass
        once Lane C's ledger rows are populated).
    refcache_root
        Root of the parity loop's refcache directory.
    markdown_index_path
        Destination for the ``docs/VISUAL_REFERENCE.md`` markdown index.

    Returns
    -------
    V2BuildResult
        Output paths and per-domain counts.
    """

    from scripts.visual_parity.io import read_ledger

    read_ledger(ledger_path)  # schema-version validation only, for now.

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    entries = _load_v2_cards(card_manifest_path, coverage_matrix_path, refcache_root, output_root)
    by_domain: Dict[str, List[V2CardEntry]] = {domain: [] for domain in V2_DOMAINS}
    for entry in entries:
        by_domain.setdefault(entry.domain, []).append(entry)

    page_paths: Dict[str, str] = {}
    domain_counts: Dict[str, int] = {}
    for domain, domain_entries in by_domain.items():
        domain_counts[domain] = len(domain_entries)
        if domain_entries:
            page_paths[domain] = str(_write_v2_domain_page(output_root, domain, domain_entries))

    filled_competitor_slots = sum(1 for entry in entries if entry.competitor_image)
    index_path = _write_v2_index_html(
        output_root, domain_counts, filled_competitor_slots, len(entries)
    )
    _write_v2_markdown_index(
        markdown_index_path, output_root, domain_counts, filled_competitor_slots, len(entries)
    )

    return V2BuildResult(
        output_dir=str(output_root),
        index_path=str(index_path),
        markdown_index_path=str(markdown_index_path),
        page_paths=page_paths,
        domain_counts=domain_counts,
        filled_competitor_slots=filled_competitor_slots,
    )


def main() -> int:
    """Parse CLI arguments and build the gallery.

    Returns
    -------
    int
        Process exit code.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="docs/gallery")
    parser.add_argument(
        "--v2",
        action="store_true",
        help=(
            "Build the matrix/manifest/ledger-driven docs/visual_reference/ guide "
            "instead of the v1 hand-built demo-scene gallery."
        ),
    )
    parser.add_argument("--card-manifest", default=DEFAULT_V2_CARD_MANIFEST)
    parser.add_argument("--coverage-matrix", default=DEFAULT_V2_COVERAGE_MATRIX)
    parser.add_argument("--ledger", default=DEFAULT_V2_LEDGER)
    parser.add_argument("--refcache", default=DEFAULT_V2_REFCACHE)
    parser.add_argument("--markdown-index", default=DEFAULT_V2_MARKDOWN_INDEX)
    args = parser.parse_args()

    if args.v2:
        v2_output_dir = (
            args.output_dir if args.output_dir != "docs/gallery" else DEFAULT_V2_OUTPUT_DIR
        )
        v2_result = build_feature_reference_v2(
            output_dir=v2_output_dir,
            card_manifest_path=args.card_manifest,
            coverage_matrix_path=args.coverage_matrix,
            ledger_path=args.ledger,
            refcache_root=args.refcache,
            markdown_index_path=args.markdown_index,
        )
        print(f"Cards indexed: {sum(v2_result.domain_counts.values())}")
        print(f"Filled competitor slots: {v2_result.filled_competitor_slots}")
        print(v2_result.index_path)
        print(v2_result.markdown_index_path)
        return 0

    result = build_feature_reference(output_dir=args.output_dir)
    section_counts = {section.slug: len(section.items) for section in result.sections}
    print(f"Shapes: {section_counts['shapes']}")
    print(f"Arrowheads: {section_counts['arrows']}")
    print(f"Themes: {section_counts['themes']}")
    print(f"Features: {section_counts['features']}")
    print(result.index_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
