#!/usr/bin/env python
# ruff: noqa: E402
"""Generate node border comparison images for the matplotlib renderer.

The requested image set is intentionally reproducible: every PNG in
``eval_output/node_comparison`` is generated from code in this script rather
than assembled by hand.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dagua.config import LayoutConfig
from dagua.edges import route_edges
from dagua.graph import DaguaGraph
from dagua.graphviz_utils import to_dot
from dagua.layout import layout
from dagua.render.mpl import (
    _build_node_patch,
    _draw_clusters,
    _draw_edges,
    _draw_node_labels,
    _draw_node_shape_extras,
    _draw_nodes,
    _node_linestyle,
)
from dagua.styles import (
    PALETTE_ORDER,
    ClusterStyle,
    EdgeStyle,
    GraphStyle,
    NodeStyle,
    border_from_fill,
    make_fill,
)

WHITE = "#FFFFFF"
INK = "#1F2937"
SOFT_BORDER = "#CBD5E1"
SOFT_PANEL = "#F7FAFC"
TRANSPARENT_BG = "#E7EEF7"
FIGURE_DPI = 220
PANEL_CONTENT_FRACTION = 0.91
DEFAULT_OUTPUT_DIR = REPO_ROOT / "eval_output" / "node_comparison"
GRAPHVIZ_COMPARISON_DOT = """digraph G {
    graph [rankdir=TB, nodesep=0.34, ranksep=0.50, margin=0.02, pad=0.02];
    node [
        shape=box,
        style=filled,
        fillcolor="#E8F0FE",
        color="#7A8797",
        penwidth=1.15,
        margin="0.18,0.10"
    ];
    edge [color="#6B7280", penwidth=1.15, arrowsize=0.78];
    Input -> Process;
    Input -> Transform;
    Process -> Validate;
    Transform -> Validate;
    Validate -> Output;
}
"""
SHAPES: Tuple[str, ...] = (
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
)
REQUESTED_FILENAMES: Tuple[str, ...] = (
    "all_shapes_solid.png",
    "default_nodes.png",
    "graphviz_comparison.png",
    "all_shapes_dashed.png",
    "all_shapes_dotted.png",
    "border_weight_ladder.png",
    "corner_radius_range.png",
    "transparent_borders.png",
    "thick_border_stress.png",
    "mixed_shapes_graph.png",
    "cluster_nesting.png",
    "mpl_native_comparison.png",
    "shapes_fill_gradient.png",
    "star_diamond_border.png",
    "cylinder_border.png",
    "tiny_nodes_thick_border.png",
)


@dataclass(frozen=True)
class NodePanelSpec:
    """Describe one manually arranged node in a comparison sheet.

    Parameters
    ----------
    caption : str
        External label shown near the node.
    style : NodeStyle
        Style applied to the node.
    width : float
        Node width in data units.
    height : float
        Node height in data units.
    """

    caption: str
    style: NodeStyle
    width: float
    height: float


def _base_graph_style(background_color: str = WHITE) -> GraphStyle:
    """Return the shared graph-level style used by this generator.

    Parameters
    ----------
    background_color : str, default=WHITE
        Figure and axes background color.

    Returns
    -------
    GraphStyle
        Graph style with white-paper defaults suitable for comparison sheets.
    """

    return GraphStyle(
        background_color=background_color,
        margin=10.0,
        min_figsize=(2.0, 1.5),
        max_figsize=(18.0, 12.0),
        title_font_size=16.0,
        title_font_weight="regular",
        title_font_color=INK,
    )


def _apply_graph_style(graph: DaguaGraph, background_color: str = WHITE) -> None:
    """Assign the shared graph style to a graph in place.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to mutate.
    background_color : str, default=WHITE
        Graph background color.

    Returns
    -------
    None
        The graph theme is updated in place.
    """

    graph._theme.graph_style = _base_graph_style(background_color=background_color)


def _apply_straight_edge_style(graph: DaguaGraph) -> None:
    """Force straight edge routing for fixed-position graph scenes.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to mutate in place.

    Returns
    -------
    None
        The graph's default edge style is updated in place.
    """

    # These comparison scenes use manually authored node positions, so bezier
    # crossover loops read like phantom unlabeled nodes at merge points.
    graph.default_edge_style = EdgeStyle(routing="straight")


def _palette_pair(index: int, blend: float = 0.25) -> Tuple[str, str]:
    """Return a muted fill and darker stroke from the shared palette.

    Parameters
    ----------
    index : int
        Palette index.
    blend : float, default=0.25
        Blend ratio passed to :func:`dagua.styles.make_fill`.

    Returns
    -------
    tuple[str, str]
        Fill and stroke colors.
    """

    base = PALETTE_ORDER[index % len(PALETTE_ORDER)]
    fill = make_fill(base, blend=blend)
    stroke = border_from_fill(base, darken=0.38)
    return fill, stroke


def _node_style(
    *,
    shape: str = "roundrect",
    fill: Optional[str] = None,
    stroke: Optional[str] = None,
    stroke_width: float = 1.4,
    stroke_dash: str = "solid",
    border_opacity: float = 1.0,
    corner_radius: float = 10.0,
    gradient: str = "none",
    gradient_color: str = "",
    gradient_angle: float = 18.0,
    opacity: float = 1.0,
) -> NodeStyle:
    """Build a readable node style for comparison panels.

    Parameters
    ----------
    shape : str, default="roundrect"
        Dagua node shape name.
    fill : str | None, optional
        Node fill color. When omitted, a palette-derived fill is used.
    stroke : str | None, optional
        Node border color. When omitted, a palette-derived stroke is used.
    stroke_width : float, default=1.4
        Border width in typographic points.
    stroke_dash : str, default="solid"
        Border dash style.
    border_opacity : float, default=1.0
        Border alpha multiplier.
    corner_radius : float, default=10.0
        Rounded-rectangle corner radius.
    gradient : str, default="none"
        Fill gradient mode.
    gradient_color : str, default=""
        Secondary gradient color.
    gradient_angle : float, default=18.0
        Linear gradient angle in degrees.
    opacity : float, default=1.0
        Overall node opacity.

    Returns
    -------
    NodeStyle
        Configured node style.
    """

    computed_fill, computed_stroke = _palette_pair(0)
    final_fill = fill or computed_fill
    final_stroke = stroke or computed_stroke
    return NodeStyle(
        shape=shape,
        fill=final_fill,
        stroke=final_stroke,
        stroke_width=stroke_width,
        stroke_dash=stroke_dash,
        border_opacity=border_opacity,
        font_size=10.0,
        font_color=INK,
        padding=(8.0, 5.0),
        corner_radius=corner_radius,
        opacity=opacity,
        gradient=gradient,
        gradient_color=gradient_color,
        gradient_angle=gradient_angle,
        base_color=final_fill,
    )


def _build_manual_node_graph(
    specs: Sequence[NodePanelSpec],
    background_color: str = WHITE,
) -> DaguaGraph:
    """Build a graph used by the manual node-sheet renderer.

    Parameters
    ----------
    specs : sequence[NodePanelSpec]
        Node specifications in render order.
    background_color : str, default=WHITE
        Graph background color.

    Returns
    -------
    DaguaGraph
        Graph with one node per panel spec and no edges.
    """

    graph = DaguaGraph()
    _apply_graph_style(graph, background_color=background_color)
    for index, spec in enumerate(specs):
        graph.add_node(f"n{index}", label="", style=spec.style)
    return graph


def _grid_positions(
    count: int,
    *,
    columns: int,
    horizontal_gap: float,
    vertical_gap: float,
) -> np.ndarray:
    """Return centered grid positions for a comparison sheet.

    Parameters
    ----------
    count : int
        Number of positions to generate.
    columns : int
        Number of columns in the grid.
    horizontal_gap : float
        Spacing between columns in data units.
    vertical_gap : float
        Spacing between rows in data units.

    Returns
    -------
    numpy.ndarray
        Position array with shape ``[count, 2]``.
    """

    rows = math.ceil(count / columns)
    positions: List[Tuple[float, float]] = []
    for index in range(count):
        row = index // columns
        column = index % columns
        x = (column - (columns - 1) / 2.0) * horizontal_gap
        y = ((rows - 1) / 2.0 - row) * vertical_gap
        positions.append((x, y))
    return np.asarray(positions, dtype=np.float64)


def _set_fixed_node_sizes(graph: DaguaGraph, sizes: np.ndarray) -> None:
    """Populate a graph's node size cache with fixed dimensions.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to update.
    sizes : numpy.ndarray
        Size array with shape ``[N, 2]``.

    Returns
    -------
    None
        The graph size cache is updated in place.
    """

    tensor_sizes = torch.tensor(sizes, dtype=torch.float32)
    graph.node_sizes = tensor_sizes
    graph.node_font_sizes = torch.full((tensor_sizes.shape[0],), 10.0, dtype=torch.float32)
    graph._node_sizes_revision = graph.revision


def _set_axes_frame(ax: Axes, positions: np.ndarray, sizes: np.ndarray, margin: float) -> None:
    """Configure axes limits tightly around positioned nodes.

    Parameters
    ----------
    ax : Axes
        Axes to configure.
    positions : numpy.ndarray
        Node positions with shape ``[N, 2]``.
    sizes : numpy.ndarray
        Node sizes with shape ``[N, 2]``.
    margin : float
        Extra frame padding in data units.

    Returns
    -------
    None
        The axes limits and display properties are updated in place.
    """

    x_min = float(np.min(positions[:, 0] - sizes[:, 0] / 2.0) - margin)
    x_max = float(np.max(positions[:, 0] + sizes[:, 0] / 2.0) + margin)
    y_min = float(np.min(positions[:, 1] - sizes[:, 1] / 2.0) - margin)
    y_max = float(np.max(positions[:, 1] + sizes[:, 1] / 2.0) + margin)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.axis("off")


def _save_figure(fig: Figure, output_path: Path) -> None:
    """Save a Matplotlib figure with consistent white margins.

    Parameters
    ----------
    fig : Figure
        Figure to save.
    output_path : Path
        Target PNG path.

    Returns
    -------
    None
        The figure is written to disk and then closed.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        pad_inches=0.12,
        facecolor=fig.get_facecolor(),
        edgecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def _render_manual_node_grid(
    output_path: Path,
    title: str,
    specs: Sequence[NodePanelSpec],
    *,
    columns: int,
    horizontal_gap: float,
    vertical_gap: float,
    figure_size: Tuple[float, float],
    background_color: str = WHITE,
    caption_gap: float = 18.0,
) -> None:
    """Render a manually positioned node comparison grid.

    Parameters
    ----------
    output_path : Path
        Final PNG path.
    title : str
        Figure title.
    specs : sequence[NodePanelSpec]
        Node specifications.
    columns : int
        Number of grid columns.
    horizontal_gap : float
        Column spacing in data units.
    vertical_gap : float
        Row spacing in data units.
    figure_size : tuple[float, float]
        Matplotlib figure size in inches.
    background_color : str, default=WHITE
        Figure background color.
    caption_gap : float, default=18.0
        Offset between node bottom and external caption baseline.

    Returns
    -------
    None
        The PNG is written to ``output_path``.
    """

    graph = _build_manual_node_graph(specs, background_color=background_color)
    positions = _grid_positions(
        len(specs),
        columns=columns,
        horizontal_gap=horizontal_gap,
        vertical_gap=vertical_gap,
    )
    sizes = np.asarray([[spec.width, spec.height] for spec in specs], dtype=np.float64)
    _set_fixed_node_sizes(graph, sizes)

    fig, ax = plt.subplots(figsize=figure_size)
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    _set_axes_frame(ax, positions, sizes, margin=48.0)
    clip_patches = _draw_nodes(ax=ax, graph=graph, pos=positions, sizes=sizes)
    if clip_patches:
        # The manual panels rely on external captions, so the internal label pass
        # intentionally stays disabled.
        del clip_patches

    for (x, y), spec in zip(positions, specs):
        ax.text(
            float(x),
            float(y - spec.height / 2.0 - caption_gap),
            spec.caption,
            ha="center",
            va="top",
            fontsize=11.0,
            color=INK,
        )

    ax.set_title(title, fontsize=18.0, color=INK, pad=14.0)
    _save_figure(fig, output_path)


def _draw_dagua_scene(
    ax: Axes,
    graph: DaguaGraph,
    positions: torch.Tensor,
    *,
    draw_labels: bool = True,
    margin: float = 34.0,
) -> None:
    """Render a graph onto an existing axes using Dagua's matplotlib codepath.

    Parameters
    ----------
    ax : Axes
        Destination axes.
    graph : DaguaGraph
        Graph to render.
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    draw_labels : bool, default=True
        Whether to render node labels.
    margin : float, default=34.0
        Axes padding in data units.

    Returns
    -------
    None
        The scene is drawn directly onto ``ax``.
    """

    pos = positions.detach().cpu().numpy().astype(np.float64)
    graph.compute_node_sizes()
    assert graph.node_sizes is not None
    sizes = graph.node_sizes.detach().cpu().numpy().astype(np.float64)

    _set_axes_frame(ax, pos, sizes, margin=margin)
    ax.figure.patch.set_facecolor(graph.graph_style.background_color)
    ax.set_facecolor(graph.graph_style.background_color)

    _draw_clusters(ax, graph, pos, sizes)
    if int(graph.edge_index.shape[1]) > 0:
        curves = route_edges(positions, graph.edge_index, graph.node_sizes, graph.direction, graph)
        _draw_edges(ax, graph, curves)
    clip_patches = _draw_nodes(ax, graph, pos, sizes)
    if draw_labels:
        _draw_node_labels(ax, graph, pos, sizes, clip_patches)


def _render_dagua_graph(
    output_path: Path,
    title: Optional[str],
    graph: DaguaGraph,
    positions: torch.Tensor,
    *,
    figure_size: Tuple[float, float],
    margin: float = 26.0,
) -> None:
    """Render one graph-only panel with Dagua's Matplotlib drawing helpers.

    Parameters
    ----------
    output_path : Path
        Final PNG path.
    title : str | None
        Figure title. When ``None``, the graph panel is rendered without an
        internal title so composed comparison sheets can provide the header.
    graph : DaguaGraph
        Graph to render.
    positions : torch.Tensor
        Fixed node positions with shape ``[N, 2]``.
    figure_size : tuple[float, float]
        Matplotlib figure size in inches.
    margin : float, default=26.0
        Axes padding in data units.

    Returns
    -------
    None
        The renderer writes the PNG to ``output_path``.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figure_size)
    fig.patch.set_facecolor(graph.graph_style.background_color)
    ax.set_facecolor(graph.graph_style.background_color)
    _draw_dagua_scene(ax, graph, positions, margin=margin)
    if title is not None:
        ax.set_title(title, fontsize=17.0, color=INK, pad=14.0)
    _save_figure(fig, output_path)


def _trim_raster_to_content(image: np.ndarray, threshold: int = 18) -> np.ndarray:
    """Crop a white-backed raster to its non-background content bounds.

    Parameters
    ----------
    image : numpy.ndarray
        Raster image with shape ``[H, W, 3]``.
    threshold : int, default=18
        Sum-of-channel distance from white required to mark a pixel as content.

    Returns
    -------
    numpy.ndarray
        Cropped image. If no content is detected, the original image is returned.
    """

    rgb = image[..., :3].astype(np.int16, copy=False)
    content_mask = np.sum(np.abs(rgb - 255), axis=2) > threshold
    if not np.any(content_mask):
        return image
    y_indices, x_indices = np.nonzero(content_mask)
    y_min = int(y_indices.min())
    y_max = int(y_indices.max()) + 1
    x_min = int(x_indices.min())
    x_max = int(x_indices.max()) + 1
    return image[y_min:y_max, x_min:x_max]


def _normalize_panel_raster(
    image_path: Path,
    *,
    canvas_size: Tuple[int, int],
    content_fraction: float = PANEL_CONTENT_FRACTION,
) -> np.ndarray:
    """Trim and scale a raster into a shared comparison-panel canvas.

    Parameters
    ----------
    image_path : Path
        Source raster to normalize.
    canvas_size : tuple[int, int]
        Output canvas size as ``(width, height)`` in pixels.
    content_fraction : float, default=PANEL_CONTENT_FRACTION
        Fraction of the canvas each cropped image should occupy after scaling.

    Returns
    -------
    numpy.ndarray
        Normalized white-backed raster with shape ``[H, W, 3]``.
    """

    source_image = Image.open(image_path).convert("RGBA")
    white_backdrop = Image.new("RGBA", source_image.size, (255, 255, 255, 255))
    white_backdrop.alpha_composite(source_image)
    cropped = _trim_raster_to_content(np.asarray(white_backdrop.convert("RGB")))

    crop_height, crop_width = cropped.shape[:2]
    target_width = max(1, int(round(canvas_size[0] * content_fraction)))
    target_height = max(1, int(round(canvas_size[1] * content_fraction)))
    scale = min(target_width / crop_width, target_height / crop_height)
    scaled_width = max(1, int(round(crop_width * scale)))
    scaled_height = max(1, int(round(crop_height * scale)))

    resampling = getattr(Image, "Resampling", Image).LANCZOS
    resized = Image.fromarray(cropped).resize((scaled_width, scaled_height), resampling)
    panel = Image.new("RGB", canvas_size, WHITE)
    offset_x = (canvas_size[0] - scaled_width) // 2
    offset_y = (canvas_size[1] - scaled_height) // 2
    panel.paste(resized, (offset_x, offset_y))
    return np.asarray(panel)


def _compose_image_panels(
    output_path: Path,
    title: str,
    panels: Sequence[Tuple[str, Path]],
    *,
    figure_size: Tuple[float, float],
) -> None:
    """Compose pre-rendered raster panels into a single side-by-side figure.

    Parameters
    ----------
    output_path : Path
        Final PNG path.
    title : str
        Figure title.
    panels : sequence[tuple[str, Path]]
        Sequence of panel title and image path pairs.
    figure_size : tuple[float, float]
        Matplotlib figure size in inches.

    Returns
    -------
    None
        The composed image is written to disk.
    """

    fig, axes = plt.subplots(1, len(panels), figsize=figure_size)
    fig.patch.set_facecolor(WHITE)
    if len(panels) == 1:
        axes = [axes]

    panel_canvas = (
        max(1080, int(round((figure_size[0] * FIGURE_DPI * 0.92) / max(len(panels), 1)))),
        max(780, int(round(figure_size[1] * FIGURE_DPI * 0.74))),
    )
    for ax, (panel_title, image_path) in zip(axes, panels):
        image = _normalize_panel_raster(image_path, canvas_size=panel_canvas)
        ax.set_facecolor(SOFT_PANEL)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_edgecolor(SOFT_BORDER)
        ax.text(
            0.05,
            0.965,
            panel_title,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14.5,
            fontweight="semibold",
            color=INK,
        )
        ax.plot(
            [0.04, 0.96],
            [0.885, 0.885],
            transform=ax.transAxes,
            color=SOFT_BORDER,
            linewidth=0.9,
            solid_capstyle="round",
        )
        image_ax = ax.inset_axes([0.035, 0.07, 0.93, 0.79])
        image_ax.imshow(image, interpolation="lanczos")
        image_ax.set_facecolor(WHITE)
        image_ax.set_xticks([])
        image_ax.set_yticks([])
        for spine in image_ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(title, fontsize=18.0, fontweight="semibold", color=INK, y=0.96)
    plt.subplots_adjust(left=0.022, right=0.978, top=0.87, bottom=0.06, wspace=0.045)
    _save_figure(fig, output_path)


def _render_graphviz_native(
    graph: DaguaGraph,
    output_path: Path,
    *,
    engine: str,
    positions: Optional[torch.Tensor] = None,
    dpi: int = FIGURE_DPI,
) -> None:
    """Render a graph with Graphviz.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to export.
    output_path : Path
        Final PNG path.
    engine : str
        Graphviz engine executable to invoke.
    positions : torch.Tensor | None, optional
        Fixed node positions with shape ``[N, 2]``. When provided, the helper
        reuses those coordinates via Graphviz's ``-n2`` mode.
    dpi : int, default=FIGURE_DPI
        Graphviz raster DPI passed through as a graph attribute.

    Returns
    -------
    None
        Graphviz writes the PNG to ``output_path``.

    Raises
    ------
    RuntimeError
        If the requested Graphviz executable is unavailable or rendering fails.
    """

    executable = shutil.which(engine)
    if executable is None:
        raise RuntimeError(f"Graphviz {engine} is required for graphviz_comparison.png")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dot_source = to_dot(graph, positions=positions)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".dot", delete=False) as handle:
        handle.write(dot_source)
        dot_path = Path(handle.name)

    try:
        command = [executable, f"-Gdpi={dpi}"]
        if positions is not None:
            command.append("-n2")
        command.extend(["-Tpng", "-o", str(output_path), str(dot_path)])
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        dot_path.unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(f"Graphviz native render failed: {result.stderr.strip()}")


def _render_graphviz_dot_png(dot_source: str, output_path: Path) -> None:
    """Render a DOT source string to PNG via the Graphviz ``dot`` executable.

    Parameters
    ----------
    dot_source : str
        DOT source describing the graph to render.
    output_path : Path
        Final PNG path.

    Returns
    -------
    None
        Graphviz writes the PNG to ``output_path``.

    Raises
    ------
    RuntimeError
        If ``dot`` is unavailable or the subprocess exits with a non-zero code.
    """

    dot_executable = shutil.which("dot")
    if dot_executable is None:
        raise RuntimeError("Graphviz dot is required for graphviz_comparison.png")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".dot",
        prefix="graphviz_comparison_",
        delete=False,
        dir="/tmp",
    ) as handle:
        handle.write(dot_source)
        dot_path = Path(handle.name)

    try:
        result = subprocess.run(
            [dot_executable, "-Tpng", str(dot_path), "-o", str(output_path)],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        dot_path.unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(f"Graphviz dot render failed: {result.stderr.strip()}")


def _render_graphviz_comparison_dagua(graph: DaguaGraph, output_path: Path) -> None:
    """Layout and render the Graphviz comparison DAG with Dagua.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to lay out and render.
    output_path : Path
        Final PNG path.

    Returns
    -------
    None
        The Dagua-rendered PNG is written to ``output_path``.
    """

    positions = layout(
        graph,
        LayoutConfig(
            direction="TB",
            device="cpu",
            seed=42,
            steps=90,
            node_sep=48.0,
            rank_sep=70.0,
            adaptive_spacing=False,
            edge_opt_steps=-1,
        ),
    )
    # Graphviz's PNG uses image-space Y with the root visually at the top,
    # while Dagua's render helpers treat larger Y as higher on the canvas.
    # Flipping the solved layout keeps both panels in the same top-to-bottom
    # reading order without changing relative geometry.
    positions = positions.clone()
    positions[:, 1] = -positions[:, 1]
    _render_dagua_graph(
        output_path,
        None,
        graph,
        positions,
        figure_size=(6.4, 4.8),
        margin=18.0,
    )


def _shape_demo_specs(stroke_dash: str) -> List[NodePanelSpec]:
    """Return the 13-shape gallery specs for a given border style.

    Parameters
    ----------
    stroke_dash : str
        Border dash style used for all nodes.

    Returns
    -------
    list[NodePanelSpec]
        One spec per supported node shape. The solid showcase intentionally
        uses one shared fill and stroke so shape geometry, not palette
        differences, carries the comparison.
    """

    specs: List[NodePanelSpec] = []
    shared_fill, shared_stroke = _palette_pair(0, blend=0.22)
    for index, shape in enumerate(SHAPES):
        if stroke_dash == "solid":
            fill = shared_fill
            stroke = shared_stroke
        else:
            fill, stroke = _palette_pair(index, blend=0.22)
        width = 82.0
        height = 54.0
        if shape in {"circle", "star"}:
            width = 62.0
            height = 62.0
        if shape == "triangle":
            height = 60.0
        if shape == "cylinder":
            width = 86.0
            height = 58.0
        caption = shape.replace("roundrect", "roundrect").title()
        specs.append(
            NodePanelSpec(
                caption=caption,
                style=_node_style(
                    shape=shape,
                    fill=fill,
                    stroke=stroke,
                    stroke_width=1.8,
                    stroke_dash=stroke_dash,
                    corner_radius=12.0,
                ),
                width=width,
                height=height,
            )
        )
    return specs


def _core_demo_dag() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build the shared five-node DAG used in the core node showcase images.

    Parameters
    ----------
    None
        This helper accepts no arguments.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and fixed positions with shape ``[5, 2]``.
    """

    graph = DaguaGraph(direction="TB")
    _apply_graph_style(graph)
    graph.default_node_style = _node_style(
        stroke_width=0.9,
        corner_radius=9.0,
        opacity=1.0,
    )
    graph.default_edge_style = EdgeStyle(
        routing="straight",
        width=1.3,
        opacity=0.72,
        arrow_length=9.0,
        arrow_width=6.2,
    )
    graph.add_node("input", label="Input")
    graph.add_node("process", label="Process")
    graph.add_node("transform", label="Transform")
    graph.add_node("validate", label="Validate")
    graph.add_node("output", label="Output")
    graph.add_edge("input", "process")
    graph.add_edge("input", "transform")
    graph.add_edge("process", "validate")
    graph.add_edge("transform", "validate")
    graph.add_edge("validate", "output")
    positions = torch.tensor(
        [
            [0.0, 118.0],
            [-96.0, 34.0],
            [96.0, 34.0],
            [0.0, -48.0],
            [0.0, -126.0],
        ],
        dtype=torch.float32,
    )
    return graph, positions


def _default_dag() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build the shared five-node DAG used in the default Dagua showcase.

    Parameters
    ----------
    None
        This helper accepts no arguments.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and fixed positions with shape ``[5, 2]``.
    """

    return _core_demo_dag()


def _graphviz_comparison_dag() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build the five-node DAG used in the Graphviz comparison panel.

    Parameters
    ----------
    None
        This helper accepts no arguments.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and fixed positions with shape ``[5, 2]``.
    """

    return _core_demo_dag()


def _build_graphviz_comparison_graph() -> DaguaGraph:
    """Build the Dagua graph matching ``GRAPHVIZ_COMPARISON_DOT``.

    Parameters
    ----------
    None
        This helper accepts no arguments.

    Returns
    -------
    DaguaGraph
        Styled graph with the same topology and labels as the DOT source.
    """

    graph = DaguaGraph(direction="TB")
    _apply_graph_style(graph)
    graph.default_node_style = _node_style(
        shape="rect",
        fill="#D6E5F9",
        stroke="#426081",
        stroke_width=1.9,
        corner_radius=0.0,
    )
    graph.default_node_style.font_size = 11.0
    graph.default_node_style.font_weight = "bold"
    graph.default_node_style.padding = (10.0, 6.0)
    graph.default_edge_style = EdgeStyle(
        routing="straight",
        color="#5A6D84",
        width=1.9,
        opacity=0.95,
        arrow_length=10.0,
        arrow_width=6.8,
    )

    node_specs = (
        ("input", "Input"),
        ("process", "Process"),
        ("transform", "Transform"),
        ("validate", "Validate"),
        ("output", "Output"),
    )
    for node_id, label in node_specs:
        graph.add_node(node_id, label=label)

    edge_specs = (
        ("input", "process"),
        ("input", "transform"),
        ("process", "validate"),
        ("transform", "validate"),
        ("validate", "output"),
    )
    for source_id, target_id in edge_specs:
        graph.add_edge(source_id, target_id)

    return graph


def _mixed_shapes_graph() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a small graph mixing major node shapes in one scene.

    Parameters
    ----------
    None
        This helper accepts no arguments.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and fixed positions with shape ``[5, 2]``.
    """

    graph = DaguaGraph(direction="TB")
    _apply_graph_style(graph)
    _apply_straight_edge_style(graph)
    node_specs = [
        ("source", "Source", "rect", 0),
        ("router", "Router", "roundrect", 1),
        ("merge", "Merge", "ellipse", 2),
        ("decision", "Decision", "diamond", 3),
        ("cache", "Cache", "circle", 4),
    ]
    for node_id, label, shape, color_index in node_specs:
        fill, stroke = _palette_pair(color_index, blend=0.2)
        graph.add_node(
            node_id,
            label=label,
            style=_node_style(
                shape=shape,
                fill=fill,
                stroke=stroke,
                stroke_width=1.7,
                corner_radius=12.0,
            ),
        )
    graph.add_edge("source", "router")
    graph.add_edge("source", "merge")
    graph.add_edge("router", "decision")
    graph.add_edge("merge", "decision")
    graph.add_edge("decision", "cache")
    positions = torch.tensor(
        [
            [0.0, 170.0],
            [-130.0, 60.0],
            [130.0, 60.0],
            [0.0, -55.0],
            [0.0, -180.0],
        ],
        dtype=torch.float32,
    )
    return graph, positions


def _cluster_graph() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a three-level nested cluster example.

    Parameters
    ----------
    None
        This helper accepts no arguments.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and fixed positions with shape ``[6, 2]``.
    """

    graph = DaguaGraph(direction="TB")
    _apply_graph_style(graph)
    _apply_straight_edge_style(graph)
    labels = [
        ("a", "Ingress"),
        ("b", "Queue"),
        ("c", "Worker"),
        ("d", "Model"),
        ("e", "Audit"),
        ("f", "Export"),
    ]
    for index, (node_id, label) in enumerate(labels):
        fill, stroke = _palette_pair(index, blend=0.2)
        graph.add_node(
            node_id,
            label=label,
            style=_node_style(fill=fill, stroke=stroke, stroke_width=1.4, corner_radius=10.0),
        )

    graph.add_edge("a", "b")
    graph.add_edge("b", "c")
    graph.add_edge("c", "d")
    graph.add_edge("d", "e")
    graph.add_edge("e", "f")

    outer_style = ClusterStyle(
        fill="#F7F5EF",
        stroke="#D6C9A8",
        stroke_width=1.6,
        stroke_dash="solid",
        corner_radius=14.0,
        padding=34.0,
        opacity=0.65,
        label_offset=(10.0, 14.0),
    )
    middle_style = ClusterStyle(
        fill="#EFF6FF",
        stroke="#9DB5D9",
        stroke_width=1.5,
        stroke_dash="dashed",
        corner_radius=12.0,
        padding=28.0,
        opacity=0.7,
        label_offset=(10.0, 14.0),
    )
    inner_style = ClusterStyle(
        fill="#F3FAF7",
        stroke="#90B8A0",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=10.0,
        padding=24.0,
        opacity=0.78,
        label_offset=(10.0, 14.0),
    )
    graph.add_cluster("system", ["a", "b", "c", "d", "e", "f"], style=outer_style, label="System")
    graph.add_cluster(
        "processing",
        ["b", "c", "d", "e"],
        style=middle_style,
        label="Processing",
        parent="system",
    )
    graph.add_cluster(
        "core",
        ["c", "d"],
        style=inner_style,
        label="Core",
        parent="processing",
    )

    positions = torch.tensor(
        [
            [0.0, 240.0],
            [0.0, 150.0],
            [-85.0, 45.0],
            [85.0, 45.0],
            [0.0, -70.0],
            [0.0, -180.0],
        ],
        dtype=torch.float32,
    )
    return graph, positions


def _native_vs_data_specs() -> Tuple[DaguaGraph, np.ndarray, np.ndarray]:
    """Return a shared node set for the native-vs-Dagua border comparison.

    Parameters
    ----------
    None
        This helper accepts no arguments.

    Returns
    -------
    tuple[DaguaGraph, numpy.ndarray, numpy.ndarray]
        Graph, positions, and fixed sizes.
    """

    specs = [
        NodePanelSpec(
            caption="roundrect",
            style=_node_style(
                shape="roundrect",
                fill="#E9F3FF",
                stroke="#406A8B",
                stroke_width=4.0,
                stroke_dash="dashed",
                corner_radius=14.0,
            ),
            width=96.0,
            height=60.0,
        ),
        NodePanelSpec(
            caption="ellipse",
            style=_node_style(
                shape="ellipse",
                fill="#EEF8F2",
                stroke="#467C62",
                stroke_width=4.0,
                stroke_dash="dashed",
            ),
            width=96.0,
            height=60.0,
        ),
        NodePanelSpec(
            caption="diamond",
            style=_node_style(
                shape="diamond",
                fill="#FFF3E8",
                stroke="#A5662E",
                stroke_width=4.0,
                stroke_dash="dashed",
            ),
            width=92.0,
            height=64.0,
        ),
        NodePanelSpec(
            caption="star",
            style=_node_style(
                shape="star",
                fill="#FAEAF4",
                stroke="#8E5376",
                stroke_width=4.0,
                stroke_dash="dashed",
            ),
            width=72.0,
            height=72.0,
        ),
        NodePanelSpec(
            caption="cylinder",
            style=_node_style(
                shape="cylinder",
                fill="#ECF5F7",
                stroke="#3E7380",
                stroke_width=4.0,
                stroke_dash="dotted",
            ),
            width=102.0,
            height=62.0,
        ),
    ]
    graph = _build_manual_node_graph(specs)
    positions = np.asarray(
        [[-220.0, 0.0], [-110.0, 0.0], [0.0, 0.0], [120.0, 0.0], [245.0, 0.0]],
        dtype=np.float64,
    )
    sizes = np.asarray([[spec.width, spec.height] for spec in specs], dtype=np.float64)
    _set_fixed_node_sizes(graph, sizes)
    return graph, positions, sizes


def _render_mpl_native_vs_dagua(output_path: Path) -> None:
    """Render native Matplotlib strokes beside Dagua's data-coordinate borders.

    Parameters
    ----------
    output_path : Path
        Final PNG path.

    Returns
    -------
    None
        The comparison image is written to disk.
    """

    graph, positions, sizes = _native_vs_data_specs()
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))
    fig.patch.set_facecolor(WHITE)

    for ax in axes:
        ax.set_facecolor(WHITE)
        _set_axes_frame(ax, positions, sizes, margin=52.0)

    axes[0].set_title("Native Matplotlib stroke", fontsize=16.0, color=INK, pad=12.0)
    axes[1].set_title("Dagua data-coordinate borders", fontsize=16.0, color=INK, pad=12.0)

    for index in range(graph.num_nodes):
        x = float(positions[index, 0])
        y = float(positions[index, 1])
        width = float(sizes[index, 0])
        height = float(sizes[index, 1])
        style = graph.get_style_for_node(index)
        facecolor = style.fill
        edgecolor = style.stroke
        patch = _build_node_patch(
            x,
            y,
            width,
            height,
            style,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=float(style.stroke_width),
            linestyle=_node_linestyle(style),
            zorder=2.0,
        )
        axes[0].add_patch(patch)
        _draw_node_shape_extras(axes[0], x, y, width, height, style, edgecolor, 2.05)
        axes[0].text(
            x,
            y - height / 2.0 - 20.0,
            style.shape,
            ha="center",
            va="top",
            fontsize=11.0,
            color=INK,
        )

    _draw_nodes(axes[1], graph, positions, sizes)
    for index in range(graph.num_nodes):
        x = float(positions[index, 0])
        y = float(positions[index, 1])
        height = float(sizes[index, 1])
        style = graph.get_style_for_node(index)
        axes[1].text(
            x,
            y - height / 2.0 - 20.0,
            style.shape,
            ha="center",
            va="top",
            fontsize=11.0,
            color=INK,
        )

    fig.suptitle("Matplotlib stroke vs Dagua border geometry", fontsize=18.0, color=INK, y=0.98)
    plt.subplots_adjust(left=0.03, right=0.97, top=0.84, bottom=0.08, wspace=0.08)
    _save_figure(fig, output_path)


def _render_graphviz_comparison(output_path: Path) -> None:
    """Render Dagua and Graphviz versions of the same five-node DAG.

    Parameters
    ----------
    output_path : Path
        Final PNG path.

    Returns
    -------
    None
        The composed comparison image is written to disk.
    """

    graph = _build_graphviz_comparison_graph()
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
        temp_dir = Path(tmp_dir)
        dagua_path = temp_dir / "dagua_render.png"
        graphviz_path = temp_dir / "graphviz_render.png"
        _render_graphviz_dot_png(GRAPHVIZ_COMPARISON_DOT, graphviz_path)
        _render_graphviz_comparison_dagua(graph, dagua_path)
        _compose_image_panels(
            output_path,
            "Graphviz vs Dagua",
            (("Graphviz", graphviz_path), ("Dagua", dagua_path)),
            figure_size=(13.8, 5.8),
        )


def _render_all_shapes_solid(output_path: Path) -> None:
    """Render the solid-border 13-shape gallery.

    Parameters
    ----------
    output_path : Path
        Final PNG path.

    Returns
    -------
    None
        The PNG is written to disk.
    """

    _render_manual_node_grid(
        output_path,
        "All supported node shapes · solid borders",
        _shape_demo_specs("solid"),
        columns=4,
        horizontal_gap=170.0,
        vertical_gap=150.0,
        figure_size=(13.5, 10.0),
    )


def _render_default_nodes(output_path: Path) -> None:
    """Render the default five-node Dagua styling sample.

    Parameters
    ----------
    output_path : Path
        Final PNG path.

    Returns
    -------
    None
        The PNG is written to disk.
    """

    graph, positions = _default_dag()
    _render_dagua_graph(
        output_path,
        "Default Dagua nodes",
        graph,
        positions,
        figure_size=(7.4, 5.8),
        margin=24.0,
    )


def _render_all_shapes_dashed(output_path: Path) -> None:
    """Render the dashed-border 13-shape gallery.

    Parameters
    ----------
    output_path : Path
        Final PNG path.

    Returns
    -------
    None
        The PNG is written to disk.
    """

    _render_manual_node_grid(
        output_path,
        "All supported node shapes · dashed borders",
        _shape_demo_specs("dashed"),
        columns=4,
        horizontal_gap=170.0,
        vertical_gap=150.0,
        figure_size=(13.5, 10.0),
    )


def _render_all_shapes_dotted(output_path: Path) -> None:
    """Render the dotted-border 13-shape gallery.

    Parameters
    ----------
    output_path : Path
        Final PNG path.

    Returns
    -------
    None
        The PNG is written to disk.
    """

    _render_manual_node_grid(
        output_path,
        "All supported node shapes · dotted borders",
        _shape_demo_specs("dotted"),
        columns=4,
        horizontal_gap=170.0,
        vertical_gap=150.0,
        figure_size=(13.5, 10.0),
    )


def _render_border_weight_ladder(output_path: Path) -> None:
    """Render a roundrect border-weight ladder.

    Parameters
    ----------
    output_path : Path
        Final PNG path.

    Returns
    -------
    None
        The PNG is written to disk.
    """

    widths = [0.25, 0.5, 1.0, 2.0, 4.0]
    specs: List[NodePanelSpec] = []
    for index, width in enumerate(widths):
        fill, stroke = _palette_pair(index + 1, blend=0.2)
        specs.append(
            NodePanelSpec(
                caption=f"{width:g} pt",
                style=_node_style(
                    shape="roundrect",
                    fill=fill,
                    stroke=stroke,
                    stroke_width=width,
                    corner_radius=12.0,
                ),
                width=110.0,
                height=62.0,
            )
        )
    _render_manual_node_grid(
        output_path,
        "Border weight ladder",
        specs,
        columns=5,
        horizontal_gap=138.0,
        vertical_gap=110.0,
        figure_size=(13.2, 3.8),
    )


def _render_corner_radius_range(output_path: Path) -> None:
    """Render a roundrect corner-radius range.

    Parameters
    ----------
    output_path : Path
        Final PNG path.

    Returns
    -------
    None
        The PNG is written to disk.
    """

    radii = [0.0, 2.0, 4.0, 8.0, 12.0, 20.0]
    specs: List[NodePanelSpec] = []
    for index, radius in enumerate(radii):
        fill, stroke = _palette_pair(index + 1, blend=0.2)
        specs.append(
            NodePanelSpec(
                caption=f"{int(radius)}",
                style=_node_style(
                    shape="roundrect",
                    fill=fill,
                    stroke=stroke,
                    stroke_width=1.6,
                    corner_radius=radius,
                ),
                width=108.0,
                height=60.0,
            )
        )
    _render_manual_node_grid(
        output_path,
        "Corner radius range",
        specs,
        columns=6,
        horizontal_gap=126.0,
        vertical_gap=110.0,
        figure_size=(14.2, 3.8),
    )


def _render_transparent_borders(output_path: Path) -> None:
    """Render semi-transparent borders on a colored field.

    Parameters
    ----------
    output_path : Path
        Final PNG path.

    Returns
    -------
    None
        The PNG is written to disk.
    """

    specs: List[NodePanelSpec] = []
    for index, shape in enumerate(("roundrect", "ellipse", "diamond", "circle", "hexagon")):
        fill, stroke = _palette_pair(index, blend=0.12)
        specs.append(
            NodePanelSpec(
                caption=f"{shape} · 50% border",
                style=_node_style(
                    shape=shape,
                    fill=fill,
                    stroke=stroke,
                    stroke_width=2.2,
                    border_opacity=0.5,
                    corner_radius=12.0,
                ),
                width=90.0 if shape != "circle" else 68.0,
                height=58.0 if shape != "circle" else 68.0,
            )
        )
    _render_manual_node_grid(
        output_path,
        "Transparent borders on a colored background",
        specs,
        columns=5,
        horizontal_gap=145.0,
        vertical_gap=110.0,
        figure_size=(14.0, 4.0),
        background_color=TRANSPARENT_BG,
    )


def _render_thick_border_stress(output_path: Path) -> None:
    """Render small-node thick-border stress cases.

    Parameters
    ----------
    output_path : Path
        Final PNG path.

    Returns
    -------
    None
        The PNG is written to disk.
    """

    cases = [
        ("roundrect", 26.0, 18.0, 4.0),
        ("roundrect", 22.0, 16.0, 3.5),
        ("ellipse", 24.0, 18.0, 4.0),
        ("diamond", 22.0, 18.0, 4.0),
        ("circle", 18.0, 18.0, 3.5),
        ("cylinder", 24.0, 18.0, 4.0),
    ]
    specs: List[NodePanelSpec] = []
    for index, (shape, width, height, stroke_width) in enumerate(cases):
        fill, stroke = _palette_pair(index + 1, blend=0.18)
        specs.append(
            NodePanelSpec(
                caption=f"{shape} · {int(width)}×{int(height)} · {stroke_width:g} pt",
                style=_node_style(
                    shape=shape,
                    fill=fill,
                    stroke=stroke,
                    stroke_width=stroke_width,
                    corner_radius=10.0,
                ),
                width=width,
                height=height,
            )
        )
    _render_manual_node_grid(
        output_path,
        "Thick border stress on small nodes",
        specs,
        columns=3,
        horizontal_gap=120.0,
        vertical_gap=120.0,
        figure_size=(12.0, 5.8),
        caption_gap=22.0,
    )


def _render_mixed_shapes_graph(output_path: Path) -> None:
    """Render a mixed-shape graph scene.

    Parameters
    ----------
    output_path : Path
        Final PNG path.

    Returns
    -------
    None
        The PNG is written to disk.
    """

    graph, positions = _mixed_shapes_graph()
    _render_dagua_graph(
        output_path,
        "Mixed-shape graph",
        graph,
        positions,
        figure_size=(8.4, 7.8),
    )


def _render_cluster_nesting(output_path: Path) -> None:
    """Render the three-level cluster nesting scene.

    Parameters
    ----------
    output_path : Path
        Final PNG path.

    Returns
    -------
    None
        The PNG is written to disk.
    """

    graph, positions = _cluster_graph()
    _render_dagua_graph(
        output_path,
        "Cluster nesting",
        graph,
        positions,
        figure_size=(8.6, 9.2),
    )


def _render_shapes_fill_gradient(output_path: Path) -> None:
    """Render gradient-filled shapes to verify clipping against borders.

    Parameters
    ----------
    output_path : Path
        Final PNG path.

    Returns
    -------
    None
        The PNG is written to disk.
    """

    shapes = ("roundrect", "ellipse", "diamond", "hexagon", "star", "cylinder")
    specs: List[NodePanelSpec] = []
    for index, shape in enumerate(shapes):
        fill, stroke = _palette_pair(index, blend=0.16)
        specs.append(
            NodePanelSpec(
                caption=f"{shape} · {'radial' if index % 2 else 'linear'}",
                style=_node_style(
                    shape=shape,
                    fill=fill,
                    stroke=stroke,
                    stroke_width=1.8,
                    corner_radius=12.0,
                    gradient="radial" if index % 2 else "linear",
                    gradient_color=stroke,
                    gradient_angle=22.0 + 18.0 * index,
                ),
                width=92.0 if shape != "star" else 76.0,
                height=60.0 if shape != "star" else 76.0,
            )
        )
    _render_manual_node_grid(
        output_path,
        "Gradient fills clipped by node borders",
        specs,
        columns=3,
        horizontal_gap=170.0,
        vertical_gap=150.0,
        figure_size=(11.6, 6.8),
    )


def _render_star_diamond_border(output_path: Path) -> None:
    """Render adversarial star and diamond thick dashed borders.

    Parameters
    ----------
    output_path : Path
        Final PNG path.

    Returns
    -------
    None
        The PNG is written to disk.
    """

    specs = [
        NodePanelSpec(
            caption="star · 4 pt dashed",
            style=_node_style(
                shape="star",
                fill="#FAEAF4",
                stroke="#8E5376",
                stroke_width=4.0,
                stroke_dash="dashed",
            ),
            width=120.0,
            height=120.0,
        ),
        NodePanelSpec(
            caption="diamond · 4 pt dashed",
            style=_node_style(
                shape="diamond",
                fill="#FFF3E8",
                stroke="#A5662E",
                stroke_width=4.0,
                stroke_dash="dashed",
            ),
            width=128.0,
            height=92.0,
        ),
    ]
    _render_manual_node_grid(
        output_path,
        "Adversarial dashed borders",
        specs,
        columns=2,
        horizontal_gap=220.0,
        vertical_gap=120.0,
        figure_size=(10.0, 5.2),
        caption_gap=24.0,
    )


def _render_cylinder_border(output_path: Path) -> None:
    """Render cylinders at multiple border widths.

    Parameters
    ----------
    output_path : Path
        Final PNG path.

    Returns
    -------
    None
        The PNG is written to disk.
    """

    widths = [0.5, 1.0, 2.0, 4.0]
    specs: List[NodePanelSpec] = []
    for index, width in enumerate(widths):
        fill, stroke = _palette_pair(index + 2, blend=0.18)
        specs.append(
            NodePanelSpec(
                caption=f"{width:g} pt",
                style=_node_style(
                    shape="cylinder",
                    fill=fill,
                    stroke=stroke,
                    stroke_width=width,
                ),
                width=110.0,
                height=68.0,
            )
        )
    _render_manual_node_grid(
        output_path,
        "Cylinder border widths",
        specs,
        columns=4,
        horizontal_gap=150.0,
        vertical_gap=120.0,
        figure_size=(11.8, 4.0),
    )


def _render_tiny_nodes_thick_border(output_path: Path) -> None:
    """Render very small nodes with intentionally heavy borders.

    Parameters
    ----------
    output_path : Path
        Final PNG path.

    Returns
    -------
    None
        The PNG is written to disk.
    """

    shapes = ("roundrect", "ellipse", "diamond", "circle", "star")
    specs: List[NodePanelSpec] = []
    for index, shape in enumerate(shapes):
        fill, stroke = _palette_pair(index + 1, blend=0.16)
        specs.append(
            NodePanelSpec(
                caption=f"{shape} · 10×8 · 3 pt",
                style=_node_style(
                    shape=shape,
                    fill=fill,
                    stroke=stroke,
                    stroke_width=3.0,
                    corner_radius=8.0,
                ),
                width=10.0 if shape != "circle" else 10.0,
                height=8.0 if shape != "circle" else 10.0,
            )
        )
    _render_manual_node_grid(
        output_path,
        "Tiny nodes with thick borders",
        specs,
        columns=5,
        horizontal_gap=62.0,
        vertical_gap=80.0,
        figure_size=(11.8, 3.4),
        caption_gap=12.0,
    )


def _render_generator_map() -> Mapping[str, Callable[[Path], None]]:
    """Return the filename-to-renderer mapping for this generator.

    Parameters
    ----------
    None
        This helper accepts no arguments.

    Returns
    -------
    Mapping[str, Callable[[Path], None]]
        Stable mapping for the requested PNG set.
    """

    return {
        "all_shapes_solid.png": _render_all_shapes_solid,
        "default_nodes.png": _render_default_nodes,
        "graphviz_comparison.png": _render_graphviz_comparison,
        "all_shapes_dashed.png": _render_all_shapes_dashed,
        "all_shapes_dotted.png": _render_all_shapes_dotted,
        "border_weight_ladder.png": _render_border_weight_ladder,
        "corner_radius_range.png": _render_corner_radius_range,
        "transparent_borders.png": _render_transparent_borders,
        "thick_border_stress.png": _render_thick_border_stress,
        "mixed_shapes_graph.png": _render_mixed_shapes_graph,
        "cluster_nesting.png": _render_cluster_nesting,
        "mpl_native_comparison.png": _render_mpl_native_vs_dagua,
        "shapes_fill_gradient.png": _render_shapes_fill_gradient,
        "star_diamond_border.png": _render_star_diamond_border,
        "cylinder_border.png": _render_cylinder_border,
        "tiny_nodes_thick_border.png": _render_tiny_nodes_thick_border,
    }


def generate_node_border_comparisons(
    output_dir: str,
    filenames: Optional[Sequence[str]] = None,
) -> List[str]:
    """Generate the requested node-border comparison PNG set.

    Parameters
    ----------
    output_dir : str
        Output directory for rendered images.
    filenames : sequence[str] | None, optional
        Optional subset of filenames to render. When omitted, all requested
        images are generated.

    Returns
    -------
    list[str]
        Absolute output paths in render order.

    Raises
    ------
    ValueError
        If ``filenames`` contains an unsupported entry.
    """

    generator_map = _render_generator_map()
    ordered_names = list(filenames or REQUESTED_FILENAMES)
    unknown = sorted(set(ordered_names) - set(generator_map))
    if unknown:
        raise ValueError(f"Unsupported node comparison image(s): {', '.join(unknown)}")

    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rendered_paths: List[str] = []
    for filename in ordered_names:
        output_path = output_root / filename
        generator_map[filename](output_path)
        rendered_paths.append(str(output_path))
    return rendered_paths


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the node-border comparison generator.

    Parameters
    ----------
    argv : sequence[str] | None, optional
        Optional explicit argument list.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for generated PNGs.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional subset of exact filenames to generate.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the CLI entry point for node-border comparison generation.

    Parameters
    ----------
    argv : sequence[str] | None, optional
        Optional explicit argument list.

    Returns
    -------
    int
        Process exit status.
    """

    args = _parse_args(argv)
    generate_node_border_comparisons(args.output_dir, filenames=args.only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
