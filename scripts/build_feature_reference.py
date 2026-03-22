#!/usr/bin/env python
# ruff: noqa: E402
"""Build a browsable visual reference gallery for Dagua rendering features.

The gallery is organized as static PNG specimens plus a local-file-friendly
HTML index. It is intentionally structured to accept future side-by-side
competitor renders under ``competitors/<tool>/...`` without changing the
artifact layout.
"""

from __future__ import annotations

import argparse
import html
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch

from dagua import DaguaGraph, EdgeStyle, NodeStyle, render
from dagua.render.borders import ShapeSpec, build_shape_path

INDEX_NAME = "index.html"
OUTPUT_DIRNAME = "feature_reference"
SPECIMEN_BACKGROUND = "#F8F8F8"
SPECIMEN_BORDER = "#D7D7D7"
TEXT_COLOR = "#333333"
CAPTION_COLOR = "#666666"
IMAGE_DPI = 150
COMPETITOR_PLACEHOLDERS: Tuple[str, ...] = ("mermaid", "d3", "cytoscape")

ALL_SHAPES: Tuple[str, ...] = (
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
ALL_ARROWHEADS: Tuple[str, ...] = (
    "normal",
    "inv",
    "dot",
    "box",
    "vee",
    "tee",
    "crow",
    "diamond",
    "curve",
    "icurve",
    "simple",
    "fancy",
    "wedge",
    "bracket",
    "none",
    "crows_foot_one",
    "crows_foot_many",
    "crows_foot_one_mandatory",
    "crows_foot_many_mandatory",
    "crows_foot_many_optional",
    "triangle_tee",
    "open",
    "circle",
)
ALL_ROUTING_MODES: Tuple[str, ...] = ("bezier", "straight", "ortho", "taxi")


@dataclass(frozen=True)
class SpecimenItem:
    """One gallery specimen entry.

    Parameters
    ----------
    name : str
        Human-readable display name.
    path : str
        Relative image path from the gallery root.
    """

    name: str
    path: str


def _specimen_axes(fig: Figure) -> Axes:
    """Return the primary axes for a rendered specimen figure.

    Parameters
    ----------
    fig : Figure
        Rendered matplotlib figure.

    Returns
    -------
    Axes
        First axes from the figure.
    """

    if not fig.axes:
        raise ValueError("Rendered specimen figure has no axes.")
    return fig.axes[0]


def _finalize_axes(ax: Axes) -> None:
    """Apply shared axis cosmetics for specimen renders.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to update.

    Returns
    -------
    None
        The axes are modified in place.
    """

    ax.set_facecolor(SPECIMEN_BACKGROUND)
    ax.set_aspect("equal")
    ax.axis("off")


def _save_figure(fig: Figure, output_path: Path) -> None:
    """Persist one specimen figure with consistent raster settings.

    Parameters
    ----------
    fig : Figure
        Figure to save.
    output_path : Path
        Destination PNG path.

    Returns
    -------
    None
        The file is written and the figure is closed.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=IMAGE_DPI,
        bbox_inches="tight",
        facecolor=SPECIMEN_BACKGROUND,
        edgecolor=SPECIMEN_BACKGROUND,
        pad_inches=0.1,
    )
    plt.close(fig)


def _new_specimen_figure(figsize: Tuple[float, float]) -> Tuple[Figure, Axes]:
    """Create a background-colored figure for direct specimen drawing.

    Parameters
    ----------
    figsize : tuple[float, float]
        Figure size in inches.

    Returns
    -------
    tuple[Figure, Axes]
        Fresh figure and axes.
    """

    fig, ax = plt.subplots(figsize=figsize, dpi=IMAGE_DPI)
    fig.patch.set_facecolor(SPECIMEN_BACKGROUND)
    _finalize_axes(ax)
    return fig, ax


def _render_graph_figure(
    graph: DaguaGraph,
    positions: torch.Tensor,
    figsize: Tuple[float, float],
) -> Figure:
    """Render a graph into a matplotlib figure.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to render.
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    figsize : tuple[float, float]
        Output figure size in inches.

    Returns
    -------
    Figure
        Rendered figure.
    """

    fig, _ = render(graph, positions=positions, figsize=figsize, dpi=IMAGE_DPI, show=False)
    fig.patch.set_facecolor(SPECIMEN_BACKGROUND)
    _finalize_axes(_specimen_axes(fig))
    return fig


def _minimal_endpoint_style() -> NodeStyle:
    """Return a nearly invisible endpoint node style for edge specimens.

    Parameters
    ----------
    None

    Returns
    -------
    NodeStyle
        Style that leaves routing anchors in place without distracting from
        the rendered edge or arrowhead.
    """

    return NodeStyle(
        shape="circle",
        fill=SPECIMEN_BACKGROUND,
        stroke=SPECIMEN_BACKGROUND,
        opacity=0.0,
        border_opacity=0.0,
        padding=(0.0, 0.0),
        min_width=8.0,
        min_height=8.0,
    )


def _single_node_graph(label: str, style: NodeStyle) -> DaguaGraph:
    """Build a one-node graph for node-centric specimens.

    Parameters
    ----------
    label : str
        Node label.
    style : NodeStyle
        Per-node style override.

    Returns
    -------
    DaguaGraph
        Graph containing one labeled node.
    """

    graph = DaguaGraph()
    graph.add_node("feature", label=label, style=style)
    graph.compute_node_sizes()
    graph.cache_layout(torch.tensor([[0.0, 0.0]], dtype=torch.float32))
    return graph


def _caption_figure(fig: Figure, text: str) -> None:
    """Add a small caption beneath a specimen render.

    Parameters
    ----------
    fig : Figure
        Figure receiving the caption.
    text : str
        Caption text.

    Returns
    -------
    None
        The figure is modified in place.
    """

    fig.text(
        0.5,
        0.04,
        text,
        ha="center",
        va="bottom",
        fontsize=8,
        family="sans-serif",
        color=CAPTION_COLOR,
    )


def render_shape_specimen(shape: str, output_path: Path) -> None:
    """Render one node shape as a labeled specimen PNG.

    Parameters
    ----------
    shape : str
        Node shape name.
    output_path : Path
        Destination PNG path.

    Returns
    -------
    None
        The specimen image is written to disk.
    """

    fig, ax = _new_specimen_figure(figsize=(3.0, 2.0))
    ax.set_xlim(-60.0, 60.0)
    ax.set_ylim(-45.0, 45.0)

    spec = ShapeSpec(center_x=0.0, center_y=0.0, width=88.0, height=54.0, shape=shape)
    patch = PathPatch(
        build_shape_path(spec),
        facecolor="#FAFBFC",
        edgecolor=TEXT_COLOR,
        linewidth=1.5,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        0.0,
        0.0,
        shape,
        ha="center",
        va="center",
        fontsize=8.5,
        family="sans-serif",
        color=TEXT_COLOR,
        zorder=3,
    )
    _save_figure(fig, output_path)


def render_arrowhead_specimen(name: str, output_path: Path) -> None:
    """Render one arrowhead using Dagua's actual edge renderer.

    Parameters
    ----------
    name : str
        Arrowhead name.
    output_path : Path
        Destination PNG path.

    Returns
    -------
    None
        The specimen image is written to disk.
    """

    graph = DaguaGraph()
    endpoint_style = _minimal_endpoint_style()
    graph.add_node("src", label="", style=endpoint_style)
    graph.add_node("tgt", label="", style=endpoint_style)
    graph.add_edge(
        "src",
        "tgt",
        style=EdgeStyle(
            arrow=name,
            routing="straight",
            width=1.5,
            color=TEXT_COLOR,
            opacity=1.0,
            arrow_color=TEXT_COLOR,
            arrow_length=11.0,
            arrow_width=8.0,
        ),
    )
    graph.compute_node_sizes()
    positions = torch.tensor([[-30.0, 0.0], [30.0, 0.0]], dtype=torch.float32)
    graph.cache_layout(positions)

    fig = _render_graph_figure(graph=graph, positions=positions, figsize=(3.0, 1.2))
    _caption_figure(fig, name)
    _save_figure(fig, output_path)


def render_routing_specimen(mode: str, output_path: Path) -> None:
    """Render a compact four-node graph using one routing mode.

    Parameters
    ----------
    mode : str
        Edge routing mode.
    output_path : Path
        Destination PNG path.

    Returns
    -------
    None
        The specimen image is written to disk.
    """

    graph = DaguaGraph()
    graph.default_edge_style = EdgeStyle(routing=mode, arrow="normal", opacity=0.9, width=1.35)
    for node_id in range(4):
        graph.add_node(
            node_id,
            label=str(node_id),
            style=NodeStyle(shape="circle", min_width=24.0, min_height=24.0),
        )
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(1, 3)
    graph.add_edge(2, 3)
    graph.compute_node_sizes()
    positions = torch.tensor(
        [[0.0, 40.0], [-40.0, 0.0], [40.0, 0.0], [0.0, -40.0]],
        dtype=torch.float32,
    )
    graph.cache_layout(positions)

    fig = _render_graph_figure(graph=graph, positions=positions, figsize=(3.0, 3.0))
    _specimen_axes(fig).set_title(f"routing={mode}", fontsize=10, color=TEXT_COLOR, pad=6)
    _save_figure(fig, output_path)


def render_gradient_specimen(mode: str, output_path: Path) -> None:
    """Render a node specimen with a gradient fill.

    Parameters
    ----------
    mode : str
        Gradient mode, typically ``"linear"`` or ``"radial"``.
    output_path : Path
        Destination PNG path.

    Returns
    -------
    None
        The specimen image is written to disk.
    """

    graph = _single_node_graph(
        label="gradient",
        style=NodeStyle(
            shape="roundrect",
            gradient=mode,
            fill="#4A90D9",
            gradient_color="#FFFFFF",
            min_width=96.0,
            min_height=56.0,
        ),
    )
    fig = _render_graph_figure(
        graph=graph,
        positions=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        figsize=(3.0, 2.0),
    )
    _specimen_axes(fig).set_title(f"gradient={mode}", fontsize=10, color=TEXT_COLOR, pad=6)
    _save_figure(fig, output_path)


def render_text_background_specimen(output_path: Path) -> None:
    """Render a node specimen showcasing label background styling.

    Parameters
    ----------
    output_path : Path
        Destination PNG path.

    Returns
    -------
    None
        The specimen image is written to disk.
    """

    graph = _single_node_graph(
        label="text bg",
        style=NodeStyle(
            shape="roundrect",
            fill="#FFFFFF",
            stroke="#9CA3AF",
            text_background="#FDE68A",
            text_background_opacity=0.95,
            text_background_padding=(5.0, 3.0),
            text_background_corner_radius=4.0,
            min_width=96.0,
            min_height=56.0,
        ),
    )
    fig = _render_graph_figure(
        graph=graph,
        positions=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        figsize=(3.0, 2.0),
    )
    _specimen_axes(fig).set_title("text background", fontsize=10, color=TEXT_COLOR, pad=6)
    _save_figure(fig, output_path)


def render_shadow_specimen(output_path: Path) -> None:
    """Render a node specimen showcasing node shadow styling.

    Parameters
    ----------
    output_path : Path
        Destination PNG path.

    Returns
    -------
    None
        The specimen image is written to disk.
    """

    graph = _single_node_graph(
        label="shadow",
        style=NodeStyle(
            shape="roundrect",
            fill="#FAFBFC",
            stroke="#4B5563",
            shadow=True,
            shadow_offset=(3.0, -3.0),
            shadow_color="#00000040",
            shadow_blur=3.0,
            min_width=96.0,
            min_height=56.0,
        ),
    )
    fig = _render_graph_figure(
        graph=graph,
        positions=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        figsize=(3.0, 2.0),
    )
    _specimen_axes(fig).set_title("shadow", fontsize=10, color=TEXT_COLOR, pad=6)
    _save_figure(fig, output_path)


def _slugify(name: str) -> str:
    """Return a filesystem-friendly slug for a feature name.

    Parameters
    ----------
    name : str
        Input display name.

    Returns
    -------
    str
        Lowercase slug using underscores between tokens.
    """

    return name.strip().lower().replace(" ", "_")


def _ensure_output_dirs(output_dir: Path) -> Dict[str, Path]:
    """Create the managed output directory layout for the gallery.

    Parameters
    ----------
    output_dir : Path
        Root output directory.

    Returns
    -------
    dict[str, Path]
        Named subdirectories used by the build.
    """

    directories = {
        "shapes": output_dir / "shapes",
        "arrowheads": output_dir / "arrowheads",
        "routing": output_dir / "routing",
        "effects": output_dir / "effects",
        "competitors": output_dir / "competitors",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    for competitor_name in COMPETITOR_PLACEHOLDERS:
        (directories["competitors"] / competitor_name).mkdir(parents=True, exist_ok=True)
    return directories


def write_gallery_html(
    output_dir: Path,
    specimens: Mapping[str, Sequence[SpecimenItem]],
) -> None:
    """Write the main HTML gallery index.

    Parameters
    ----------
    output_dir : Path
        Root output directory.
    specimens : Mapping[str, Sequence[SpecimenItem]]
        Section name to rendered specimen list mapping.

    Returns
    -------
    None
        The HTML gallery is written to ``output_dir / "index.html"``.
    """

    section_blocks = []
    for section_name, items in specimens.items():
        cards = []
        for item in items:
            cards.extend(
                [
                    '      <div class="specimen">',
                    f'        <img src="{html.escape(item.path)}" alt="{html.escape(item.name)}">',
                    f'        <div class="caption">{html.escape(item.name)}</div>',
                    "      </div>",
                ]
            )
        section_blocks.extend(
            [
                '    <section class="section">',
                f"      <h2>{html.escape(section_name)}</h2>",
                '      <div class="grid">',
                *cards,
                "      </div>",
                '      <div class="competitor-placeholder">',
                "        Competitor side-by-side renders will be added during theme sprints.",
                "      </div>",
                "    </section>",
            ]
        )

    competitor_names = ", ".join(COMPETITOR_PLACEHOLDERS)
    html_text = "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            '  <meta charset="utf-8">',
            "  <title>Dagua Feature Reference</title>",
            "  <style>",
            "    :root {",
            f"      --bg: {SPECIMEN_BACKGROUND};",
            "      --page: #FFFFFF;",
            f"      --border: {SPECIMEN_BORDER};",
            "      --text: #111111;",
            "      --muted: #666666;",
            "      --accent: #2B6CB0;",
            "      --placeholder: #FFF7D6;",
            "      --placeholder-border: #E2B100;",
            "    }",
            "    body {",
            "      margin: 0;",
            "      background: linear-gradient(180deg, #FFFFFF 0%, #F4F5F7 100%);",
            "      color: var(--text);",
            '      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;',
            "    }",
            "    main { max-width: 1440px; margin: 0 auto; padding: 28px 24px 56px; }",
            "    h1 { margin: 0 0 10px; font-size: 32px; }",
            "    p.lede { max-width: 920px; color: var(--muted); line-height: 1.5; }",
            "    .section { margin-top: 32px; }",
            "    h2 { margin: 0 0 14px; font-size: 22px; }",
            "    .grid {",
            "      display: grid;",
            "      grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));",
            "      gap: 14px;",
            "    }",
            "    .specimen {",
            "      background: var(--page);",
            "      border: 1px solid var(--border);",
            "      border-radius: 10px;",
            "      padding: 10px;",
            "      text-align: center;",
            "      box-shadow: 0 10px 24px rgba(17, 24, 39, 0.05);",
            "    }",
            "    .specimen img {",
            "      display: block;",
            "      width: 100%;",
            "      height: auto;",
            "      background: var(--bg);",
            "      border-radius: 6px;",
            "    }",
            "    .caption { margin-top: 8px; font-size: 12px; color: var(--muted); }",
            "    .competitor-placeholder {",
            "      margin-top: 12px;",
            "      padding: 14px 16px;",
            "      background: var(--placeholder);",
            "      border: 1px dashed var(--placeholder-border);",
            "      border-radius: 10px;",
            "      color: #6B5300;",
            "      font-style: italic;",
            "    }",
            "    code { color: var(--accent); }",
            "  </style>",
            "</head>",
            "<body>",
            "  <main>",
            "    <h1>Dagua Feature Reference Gallery</h1>",
            '    <p class="lede">',
            "      Visual catalog of node shapes, arrowheads, routing modes, and render effects.",
            (
                "      Placeholder competitor directories are ready under "
                f"<code>competitors/</code> for {html.escape(competitor_names)}."
            ),
            "    </p>",
            *section_blocks,
            "  </main>",
            "</body>",
            "</html>",
        ]
    )
    (output_dir / INDEX_NAME).write_text(html_text, encoding="utf-8")


def build_gallery(
    output_dir: Path,
    shapes: Sequence[str] = ALL_SHAPES,
    arrowheads: Sequence[str] = ALL_ARROWHEADS,
    routing_modes: Sequence[str] = ALL_ROUTING_MODES,
) -> Mapping[str, Sequence[SpecimenItem]]:
    """Render the complete feature gallery into the output directory.

    Parameters
    ----------
    output_dir : Path
        Root output directory.
    shapes : Sequence[str], default=ALL_SHAPES
        Node shapes to render.
    arrowheads : Sequence[str], default=ALL_ARROWHEADS
        Arrowheads to render.
    routing_modes : Sequence[str], default=ALL_ROUTING_MODES
        Edge routing modes to render.

    Returns
    -------
    Mapping[str, Sequence[SpecimenItem]]
        Ordered specimen mapping used to build the HTML index.
    """

    directories = _ensure_output_dirs(output_dir)
    specimens: Dict[str, Sequence[SpecimenItem]] = {}

    shape_items = []
    for shape in shapes:
        output_path = directories["shapes"] / f"{_slugify(shape)}.png"
        print(f"Rendering shape: {shape}", flush=True)
        render_shape_specimen(shape, output_path)
        shape_items.append(SpecimenItem(name=shape, path=f"shapes/{output_path.name}"))
    specimens[f"Node Shapes ({len(shape_items)})"] = shape_items

    arrow_items = []
    for name in arrowheads:
        output_path = directories["arrowheads"] / f"{_slugify(name)}.png"
        print(f"Rendering arrowhead: {name}", flush=True)
        render_arrowhead_specimen(name, output_path)
        arrow_items.append(SpecimenItem(name=name, path=f"arrowheads/{output_path.name}"))
    specimens[f"Arrowheads ({len(arrow_items)})"] = arrow_items

    routing_items = []
    for mode in routing_modes:
        output_path = directories["routing"] / f"{_slugify(mode)}.png"
        print(f"Rendering routing: {mode}", flush=True)
        render_routing_specimen(mode, output_path)
        routing_items.append(SpecimenItem(name=mode, path=f"routing/{output_path.name}"))
    specimens[f"Edge Routing ({len(routing_items)})"] = routing_items

    effect_items = []
    gradient_modes = ("linear", "radial")
    for mode in gradient_modes:
        output_path = directories["effects"] / f"gradient_{mode}.png"
        print(f"Rendering gradient: {mode}", flush=True)
        render_gradient_specimen(mode, output_path)
        effect_items.append(
            SpecimenItem(name=f"gradient={mode}", path=f"effects/{output_path.name}")
        )

    text_background_path = directories["effects"] / "text_background.png"
    print("Rendering effect: text_background", flush=True)
    render_text_background_specimen(text_background_path)
    effect_items.append(
        SpecimenItem(name="text_background", path=f"effects/{text_background_path.name}")
    )

    shadow_path = directories["effects"] / "shadow.png"
    print("Rendering effect: shadow", flush=True)
    render_shadow_specimen(shadow_path)
    effect_items.append(SpecimenItem(name="shadow", path=f"effects/{shadow_path.name}"))

    specimens["Visual Effects"] = effect_items
    write_gallery_html(output_dir, specimens)
    return specimens


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the gallery builder.

    Parameters
    ----------
    argv : Sequence[str], optional
        Explicit argument vector.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(description="Build feature reference gallery")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=f"eval_output/{OUTPUT_DIRNAME}",
        help="Directory to receive the rendered gallery artifacts.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Build the feature reference gallery.

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
    build_gallery(output_dir=output_dir)
    print(f"Gallery written to {output_dir / INDEX_NAME}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
