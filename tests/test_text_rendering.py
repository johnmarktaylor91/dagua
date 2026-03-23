"""Tests for standalone data-coordinate text rendering."""

from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D

from dagua.graph import DaguaGraph
from dagua.render.mpl import _compute_display_scale
from dagua.render.text import (
    DaguaText,
    background_rect_path,
    get_font_metrics,
    layout_plain_text,
    layout_rich_text,
    measure_text_data,
    render_text,
    strikethrough_path,
    text_to_glyphs,
    underline_path,
)
from dagua.render.text.paths import _cached_font_metrics, _cached_glyph_data
from dagua.styles import RESOLVED_FONT, NodeStyle
from dagua.utils import compute_node_size, measure_text

text_collection = importlib.import_module("dagua.render.text.collection")
_IMAGE_DPI = 160
_IMAGE_SIZE = (12.0, 8.0)


def _comparison_dir() -> Path:
    """Return the comparison-artifact directory and create it if needed.

    Returns
    -------
    pathlib.Path
        Directory used for text rendering comparison artifacts.
    """
    directory = Path("eval_output/text_comparison")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _make_axes(
    xlim: Tuple[float, float] = (0.0, 100.0),
    ylim: Tuple[float, float] = (0.0, 70.0),
    facecolor: str = "#FFFFFF",
) -> Tuple[Any, Any]:
    """Create an equal-aspect matplotlib figure for text scenes.

    Parameters
    ----------
    xlim : tuple[float, float], default=(0.0, 100.0)
        X-axis limits.
    ylim : tuple[float, float], default=(0.0, 70.0)
        Y-axis limits.
    facecolor : str, default="#FFFFFF"
        Figure and axes background color.

    Returns
    -------
    tuple[Any, Any]
        ``(fig, ax)`` pair configured for data-coordinate rendering.
    """
    fig, ax = plt.subplots(figsize=_IMAGE_SIZE, dpi=_IMAGE_DPI)
    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor(facecolor)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def _render_specs(ax: Any, specs: Sequence[DaguaText]) -> List[Any]:
    """Render text specs using the standalone text package.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    specs : sequence[DaguaText]
        Render specifications.

    Returns
    -------
    list[Any]
        Artists created by the renderer.
    """
    return render_text(ax, specs, _compute_display_scale(ax))


def _save_scene(fig: Any, filename: str) -> Path:
    """Persist a rendered scene to the comparison directory.

    Parameters
    ----------
    fig : Any
        Matplotlib figure.
    filename : str
        Output filename.

    Returns
    -------
    pathlib.Path
        Saved image path.
    """
    output = _comparison_dir() / filename
    fig.savefig(output, dpi=_IMAGE_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    return output


def _rectangle_clip(ax: Any, x: float, y: float, width: float, height: float) -> PathPatch:
    """Build a rectangular clip patch in data coordinates.

    Parameters
    ----------
    ax : Any
        Matplotlib axes providing the data transform.
    x : float
        Center x-coordinate.
    y : float
        Center y-coordinate.
    width : float
        Rectangle width.
    height : float
        Rectangle height.

    Returns
    -------
    PathPatch
        Unattached clip patch.
    """
    patch = Rectangle(
        (x - width / 2.0, y - height / 2.0),
        width,
        height,
        facecolor="none",
        edgecolor="none",
        transform=ax.transData,
    )
    return patch


def _regular_polygon_path(
    center_x: float,
    center_y: float,
    radius: float,
    sides: int,
    rotation_degrees: float = 0.0,
) -> MplPath:
    """Build a closed regular-polygon path.

    Parameters
    ----------
    center_x : float
        Center x-coordinate.
    center_y : float
        Center y-coordinate.
    radius : float
        Circumradius.
    sides : int
        Number of polygon sides.
    rotation_degrees : float, default=0.0
        Polygon rotation.

    Returns
    -------
    matplotlib.path.Path
        Closed polygon path.
    """
    angles = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    rotation = math.radians(rotation_degrees)
    vertices = np.column_stack(
        [
            center_x + radius * np.cos(angles + rotation),
            center_y + radius * np.sin(angles + rotation),
        ]
    )
    vertices = np.vstack([vertices, vertices[0]])
    codes = np.full(vertices.shape[0], MplPath.LINETO, dtype=np.uint8)
    codes[0] = MplPath.MOVETO
    codes[-1] = MplPath.CLOSEPOLY
    return MplPath(vertices, codes)


def _star_path(
    center_x: float,
    center_y: float,
    outer_radius: float,
    inner_radius: float,
    points: int = 5,
) -> MplPath:
    """Build a closed star-shaped path.

    Parameters
    ----------
    center_x : float
        Center x-coordinate.
    center_y : float
        Center y-coordinate.
    outer_radius : float
        Radius of star tips.
    inner_radius : float
        Radius of inner vertices.
    points : int, default=5
        Number of star points.

    Returns
    -------
    matplotlib.path.Path
        Closed star path.
    """
    vertices: List[Tuple[float, float]] = []
    for index in range(points * 2):
        angle = math.pi / points * index - math.pi / 2.0
        radius = outer_radius if index % 2 == 0 else inner_radius
        vertices.append(
            (
                center_x + radius * math.cos(angle),
                center_y + radius * math.sin(angle),
            )
        )
    vertices.append(vertices[0])
    codes = np.full(len(vertices), MplPath.LINETO, dtype=np.uint8)
    codes[0] = MplPath.MOVETO
    codes[-1] = MplPath.CLOSEPOLY
    return MplPath(np.asarray(vertices, dtype=float), codes)


def _scene_basic_text() -> Tuple[Any, Any, List[Any]]:
    """Render the basic text comparison scene.

    Returns
    -------
    tuple[Any, Any, list[Any]]
        Figure, axes, and created artists.
    """
    fig, ax = _make_axes()
    specs = [
        DaguaText(20.0, 52.0, "Regular", font_size=14.0, ha="center"),
        DaguaText(50.0, 52.0, "Bold", font_size=14.0, font_weight="bold", ha="center"),
        DaguaText(80.0, 52.0, "Italic", font_size=14.0, font_style="italic", ha="center"),
        DaguaText(
            50.0,
            24.0,
            "The quick brown fox",
            font_size=16.0,
            ha="center",
            background="#F4F1E8",
            background_padding=(4.0, 3.0),
        ),
    ]
    artists = _render_specs(ax, specs)
    return fig, ax, artists


def _scene_alignment_grid() -> Tuple[Any, Any, List[Any]]:
    """Render the alignment-grid comparison scene.

    Returns
    -------
    tuple[Any, Any, list[Any]]
        Figure, axes, and created artists.
    """
    fig, ax = _make_axes()
    artists: List[Any] = []
    has = ["left", "center", "right"]
    vas = ["top", "center", "bottom"]
    for row, va in enumerate(vas):
        for col, ha in enumerate(has):
            cx = 20.0 + col * 30.0
            cy = 50.0 - row * 20.0
            ax.add_patch(
                Rectangle(
                    (cx - 8.0, cy - 6.0),
                    16.0,
                    12.0,
                    fill=False,
                    lw=0.8,
                    ec="#C8CDD4",
                )
            )
            ax.plot([cx - 8.0, cx + 8.0], [cy, cy], color="#D6DCE3", lw=0.6)
            ax.plot([cx, cx], [cy - 6.0, cy + 6.0], color="#D6DCE3", lw=0.6)
            artists.extend(
                _render_specs(
                    ax,
                    [
                        DaguaText(
                            cx,
                            cy,
                            f"{ha}\n{va}",
                            font_size=7.5,
                            ha=ha,
                            va=va,
                            line_spacing=1.15,
                        )
                    ],
                )
            )
    return fig, ax, artists


def _scene_font_ladder() -> Tuple[Any, Any, List[Any]]:
    """Render the font-size ladder scene.

    Returns
    -------
    tuple[Any, Any, list[Any]]
        Figure, axes, and created artists.
    """
    fig, ax = _make_axes()
    sizes = [6.0, 9.0, 12.0, 16.0, 24.0, 36.0]
    specs = [
        DaguaText(18.0 + index * 13.0, 35.0, f"{int(size)}pt", font_size=size, ha="center")
        for index, size in enumerate(sizes)
    ]
    artists = _render_specs(ax, specs)
    return fig, ax, artists


def _scene_rich_text_showcase() -> Tuple[Any, Any, List[Any]]:
    """Render the rich-text showcase scene.

    Returns
    -------
    tuple[Any, Any, list[Any]]
        Figure, axes, and created artists.
    """
    fig, ax = _make_axes()
    specs = [
        DaguaText(
            50.0,
            52.0,
            (
                "**Encoder** uses *cross-attention* with `KV cache` and "
                "{color:#B8572B}fused ops{/color}."
            ),
            rich=True,
            font_size=12.0,
            ha="center",
            va="center",
            background="#F7F2E8",
            background_padding=(5.0, 4.0),
            outline=True,
        )
    ]
    artists = _render_specs(ax, specs)
    return fig, ax, artists


def _scene_decorations() -> Tuple[Any, Any, List[Any]]:
    """Render the decoration showcase scene.

    Returns
    -------
    tuple[Any, Any, list[Any]]
        Figure, axes, and created artists.
    """
    fig, ax = _make_axes()
    specs = [
        DaguaText(20.0, 50.0, "Underline", font_size=14.0, underline=True),
        DaguaText(50.0, 50.0, "Strike", font_size=14.0, strikethrough=True),
        DaguaText(
            80.0,
            50.0,
            "Outline",
            font_size=14.0,
            outline=True,
            outline_color="#FFFFFF",
            background="#2E3B45",
            font_color="#F5F7FA",
        ),
    ]
    ax.set_facecolor("#E8EDF1")
    artists = _render_specs(ax, specs)
    return fig, ax, artists


def _scene_rotation_fan() -> Tuple[Any, Any, List[Any]]:
    """Render the rotation fan scene.

    Returns
    -------
    tuple[Any, Any, list[Any]]
        Figure, axes, and created artists.
    """
    fig, ax = _make_axes(xlim=(0.0, 100.0), ylim=(0.0, 86.0))
    artists: List[Any] = []
    center = (50.0, 35.0)
    ax.add_patch(Rectangle((8.0, 6.0), 84.0, 74.0, fill=False, lw=0.8, ec="#D7DDE5"))
    for index, angle in enumerate([0, 15, 30, 45, 60, 75, 90]):
        radius = 6.0 + index * 5.5
        radians = math.radians(angle)
        x = center[0] + radius * math.cos(radians)
        y = center[1] + radius * math.sin(radians)
        ax.plot([center[0], x], [center[1], y], color="#D2D7DE", lw=0.8)
        artists.extend(
            _render_specs(
                ax,
                [
                    DaguaText(
                        x,
                        y,
                        f"{angle} deg",
                        rotation=float(angle),
                        font_size=9.0,
                        background="#F7F4EC",
                        background_padding=(3.5, 2.5),
                    )
                ],
            )
        )
    return fig, ax, artists


def _scene_multiline_alignment() -> Tuple[Any, Any, List[Any]]:
    """Render multiline alignment examples.

    Returns
    -------
    tuple[Any, Any, list[Any]]
        Figure, axes, and created artists.
    """
    fig, ax = _make_axes()
    artists: List[Any] = []
    for index, ha in enumerate(["left", "center", "right"]):
        x = 20.0 + index * 30.0
        ax.add_patch(Rectangle((x - 10.0, 20.0), 20.0, 30.0, fill=False, lw=0.8, ec="#C8CDD4"))
        artists.extend(
            _render_specs(
                ax,
                [
                    DaguaText(
                        x,
                        35.0,
                        "Line 1\nLine 2\nLine 3",
                        font_size=9.0,
                        ha=ha,
                        va="center",
                        secondary_scale=0.82,
                    )
                ],
            )
        )
    return fig, ax, artists


def _scene_node_label_simulation() -> Tuple[Any, Any, List[Any]]:
    """Render a node-label simulation scene.

    Returns
    -------
    tuple[Any, Any, list[Any]]
        Figure, axes, and created artists.
    """
    fig, ax = _make_axes()
    artists: List[Any] = []
    for row in range(3):
        for col in range(4):
            cx = 16.0 + col * 22.0
            cy = 52.0 - row * 18.0
            ax.add_patch(
                Rectangle(
                    (cx - 8.5, cy - 5.5),
                    17.0,
                    11.0,
                    facecolor="#EDF2F5",
                    edgecolor="#9CA9B4",
                    lw=0.9,
                )
            )
            artists.extend(
                _render_specs(
                    ax,
                    [
                        DaguaText(
                            cx,
                            cy,
                            f"Node {row * 4 + col + 1}",
                            font_size=8.5,
                            max_width=13.0,
                        )
                    ],
                )
            )
    return fig, ax, artists


def _scene_edge_label_simulation() -> Tuple[Any, Any, List[Any]]:
    """Render an edge-label simulation scene.

    Returns
    -------
    tuple[Any, Any, list[Any]]
        Figure, axes, and created artists.
    """
    fig, ax = _make_axes()
    artists: List[Any] = []
    segments = [
        ((12.0, 15.0), (88.0, 22.0), 5.0),
        ((12.0, 35.0), (88.0, 35.0), 0.0),
        ((12.0, 55.0), (88.0, 48.0), -6.0),
    ]
    for index, (start, end, angle) in enumerate(segments):
        ax.plot([start[0], end[0]], [start[1], end[1]], color="#9BA6B2", lw=2.0)
        mx = (start[0] + end[0]) / 2.0
        my = (start[1] + end[1]) / 2.0
        artists.extend(
            _render_specs(
                ax,
                [
                    DaguaText(
                        mx,
                        my,
                        f"edge_{index}",
                        font_size=8.5,
                        rotation=angle,
                        background="#FBF7EF",
                        background_padding=(3.0, 2.0),
                    )
                ],
            )
        )
    return fig, ax, artists


def _scene_cluster_label_simulation() -> Tuple[Any, Any, List[Any]]:
    """Render a cluster-label simulation scene.

    Returns
    -------
    tuple[Any, Any, list[Any]]
        Figure, axes, and created artists.
    """
    fig, ax = _make_axes()
    artists: List[Any] = []
    rectangles = [
        (50.0, 35.0, 72.0, 48.0, "#EEF3F6", "#A3B4C2"),
        (48.0, 34.0, 52.0, 32.0, "#F5F7F9", "#BBC7D1"),
        (46.0, 33.0, 32.0, 18.0, "#FBFCFD", "#CCD5DC"),
    ]
    for cx, cy, width, height, fill, edge in rectangles:
        ax.add_patch(
            Rectangle(
                (cx - width / 2.0, cy - height / 2.0),
                width,
                height,
                facecolor=fill,
                edgecolor=edge,
                lw=1.0,
            )
        )
    artists.extend(
        _render_specs(
            ax,
            [
                DaguaText(
                    16.0,
                    57.0,
                    "Outer Cluster",
                    ha="left",
                    va="top",
                    font_size=12.0,
                    clip_on=False,
                ),
                DaguaText(
                    24.0,
                    48.0,
                    "Nested Cluster",
                    ha="left",
                    va="top",
                    font_size=10.5,
                    clip_on=False,
                ),
                DaguaText(
                    32.0,
                    40.0,
                    "Leaf Cluster",
                    ha="left",
                    va="top",
                    font_size=9.0,
                    clip_on=False,
                ),
            ],
        )
    )
    return fig, ax, artists


def _scene_dark_theme() -> Tuple[Any, Any, List[Any]]:
    """Render the dark-theme readability scene.

    Returns
    -------
    tuple[Any, Any, list[Any]]
        Figure, axes, and created artists.
    """
    fig, ax = _make_axes(facecolor="#182028")
    specs = [
        DaguaText(
            50.0,
            38.0,
            "Dark Theme Label",
            font_size=18.0,
            font_color="#F6F8FB",
            outline=True,
            outline_color="#11161C",
            background="#2C3945",
            background_padding=(5.0, 4.0),
        )
    ]
    artists = _render_specs(ax, specs)
    return fig, ax, artists


def _scene_zoom_scaling() -> Tuple[Any, Any, List[Any]]:
    """Render labels at multiple data-coordinate scales.

    Returns
    -------
    tuple[Any, Any, list[Any]]
        Figure, axes, and created artists.
    """
    fig, ax = _make_axes()
    artists: List[Any] = []
    for index, scale in enumerate([1.0, 2.0, 4.0]):
        base_x = 18.0 + index * 28.0
        ax.add_patch(
            Rectangle(
                (base_x - 10.0, 20.0),
                20.0,
                30.0,
                fill=False,
                lw=0.8,
                ec="#C8CDD4",
            )
        )
        artists.extend(
            _render_specs(
                ax,
                [
                    DaguaText(
                        base_x,
                        35.0,
                        f"{scale:.0f}x\nzoom",
                        font_size=9.0 * scale,
                        secondary_scale=0.8,
                    )
                ],
            )
        )
    return fig, ax, artists


def _scene_matplotlib_comparison() -> Tuple[Any, Any, List[Any]]:
    """Render the side-by-side matplotlib comparison scene.

    Uses a SINGLE axes so both samples share the same display_scale,
    ensuring TextPath and ax.text() appear at the same visual size.

    Returns
    -------
    tuple[Any, Any, list[Any]]
        Figure, axes, and created artists.
    """
    fig, ax = plt.subplots(1, 1, figsize=_IMAGE_SIZE, dpi=_IMAGE_DPI)
    artists: List[Any] = []
    comparison_text = "The quick brown fox"
    sample_font_size = 18.0
    ax.set_xlim(0.0, 200.0)
    ax.set_ylim(0.0, 70.0)
    ax.set_aspect("equal")
    ax.axis("off")
    # Left panel: TextPath
    left_cx = 55.0
    right_cx = 145.0
    header_y = 58.0
    sample_y = 34.0
    ax.add_patch(Rectangle((5.0, 16.0), 90.0, 32.0, fill=False, lw=0.8, ec="#D9DFE7"))
    ax.add_patch(Rectangle((105.0, 16.0), 90.0, 32.0, fill=False, lw=0.8, ec="#D9DFE7"))
    ax.plot([5.0, 95.0], [sample_y, sample_y], color="#E3E8EE", lw=0.6)
    ax.plot([105.0, 195.0], [sample_y, sample_y], color="#E3E8EE", lw=0.6)
    # Force layout so axes transform is finalized before computing display_scale
    fig.canvas.draw()
    ds = _compute_display_scale(ax)
    artists.extend(
        render_text(
            ax,
            [
                DaguaText(
                    left_cx, header_y, "TextPath / render_text()", font_size=9.0, ha="center"
                ),
                DaguaText(
                    left_cx,
                    sample_y,
                    comparison_text,
                    font_size=sample_font_size,
                    ha="center",
                    va="center",
                    background="#F7F3EA",
                    background_padding=(4.0, 3.0),
                ),
            ],
            ds,
        )
    )
    # Right panel: native ax.text()
    ax.text(
        right_cx,
        header_y,
        "Matplotlib / ax.text()",
        ha="center",
        va="center",
        fontsize=9.0,
        color="#111111",
    )
    ax.text(
        right_cx,
        sample_y,
        comparison_text,
        ha="center",
        va="center",
        fontsize=sample_font_size,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "#F7F3EA", "edgecolor": "none"},
    )
    return fig, ax, artists


def _scene_dense_graph_labels() -> Tuple[Any, Any, List[Any]]:
    """Render a dense grid of labels.

    Returns
    -------
    tuple[Any, Any, list[Any]]
        Figure, axes, and created artists.
    """
    fig, ax = _make_axes()
    artists: List[Any] = []
    specs: List[DaguaText] = []
    for row in range(5):
        for col in range(6):
            x = 12.0 + col * 15.0
            y = 58.0 - row * 11.0
            ax.plot(x, y, "o", color="#AAB6C2", ms=4.0)
            specs.append(DaguaText(x, y + 2.0, f"n{row}{col}", font_size=6.5))
    artists.extend(_render_specs(ax, specs))
    return fig, ax, artists


def _scene_special_characters() -> Tuple[Any, Any, List[Any]]:
    """Render a special-character showcase scene.

    Returns
    -------
    tuple[Any, Any, list[Any]]
        Figure, axes, and created artists.
    """
    fig, ax = _make_axes()
    specs = [
        DaguaText(50.0, 54.0, "cafe -> café    naïve    résumé    façade", font_size=12.5),
        DaguaText(50.0, 40.0, "α + β = γ    x × y    3 ≤ 4", font_size=12.5),
        DaguaText(50.0, 26.0, "Arrows: ← ↑ → ↓    Quotes: “curly”", font_size=12.5),
        DaguaText(50.0, 12.0, "em dash — en dash – minus −", font_size=12.5),
    ]
    artists = _render_specs(ax, specs)
    return fig, ax, artists


def _scene_weight_spectrum() -> Tuple[Any, Any, List[Any]]:
    """Render a font-weight spectrum scene.

    Returns
    -------
    tuple[Any, Any, list[Any]]
        Figure, axes, and created artists.
    """
    fig, ax = _make_axes()
    weights = ["regular", "medium", "semibold", "bold", "black"]
    baseline_y = 34.0
    ax.plot([8.0, 92.0], [baseline_y, baseline_y], color="#D4DBE3", lw=0.8)
    specs = [
        DaguaText(
            14.0 + index * 18.0,
            baseline_y,
            weight,
            font_size=11.0,
            font_weight=weight,
            font_style="normal",
            ha="center",
            va="baseline",
        )
        for index, weight in enumerate(weights)
    ]
    artists = _render_specs(ax, specs)
    return fig, ax, artists


def _scene_mixed_features() -> Tuple[Any, Any, List[Any]]:
    """Render a scene that combines most supported text features.

    Returns
    -------
    tuple[Any, Any, list[Any]]
        Figure, axes, and created artists.
    """
    fig, ax = _make_axes()
    specs = [
        DaguaText(
            50.0,
            35.0,
            "**Mixed**\n{color:#0F7B6C}Features{/color}",
            rich=True,
            font_size=14.0,
            secondary_scale=0.85,
            rotation=18.0,
            background="#F9F3E8",
            outline=True,
            outline_color="#FFFFFF",
            background_padding=(6.0, 4.0),
        )
    ]
    artists = _render_specs(ax, specs)
    return fig, ax, artists


def _scene_clip_shapes() -> Tuple[Any, Any, List[Any]]:
    """Render labels clipped to several shapes.

    Returns
    -------
    tuple[Any, Any, list[Any]]
        Figure, axes, and created artists.
    """
    fig, ax = _make_axes()
    artists: List[Any] = []
    ellipse = MplPath.unit_circle().transformed(Affine2D().scale(8.0, 5.0).translate(20.0, 35.0))
    diamond = _regular_polygon_path(45.0, 35.0, 8.0, 4, rotation_degrees=45.0)
    triangle = _regular_polygon_path(70.0, 35.0, 8.0, 3, rotation_degrees=90.0)
    star = _star_path(90.0, 35.0, 8.0, 4.0)
    for path in [ellipse, diamond, triangle, star]:
        ax.add_patch(PathPatch(path, fill=False, edgecolor="#A7B4BF", lw=0.8))
    specs = [
        DaguaText(
            20.0,
            35.0,
            "Ellipse",
            font_size=11.0,
            clip_patch=PathPatch(ellipse, transform=ax.transData),
        ),
        DaguaText(
            45.0,
            35.0,
            "Diamond",
            font_size=11.0,
            clip_patch=PathPatch(diamond, transform=ax.transData),
        ),
        DaguaText(
            70.0,
            35.0,
            "Triangle",
            font_size=11.0,
            clip_patch=PathPatch(triangle, transform=ax.transData),
        ),
        DaguaText(
            90.0,
            35.0,
            "Star",
            font_size=10.5,
            clip_patch=PathPatch(star, transform=ax.transData),
        ),
    ]
    artists.extend(_render_specs(ax, specs))
    return fig, ax, artists


def _scene_spacing_test() -> Tuple[Any, Any, List[Any]]:
    """Render rich-text spacing cases that rely on advance widths.

    Returns
    -------
    tuple[Any, Any, list[Any]]
        Figure, axes, and created artists.
    """
    fig, ax = _make_axes()
    x_left = 16.0
    ax.add_patch(Rectangle((10.0, 8.0), 80.0, 54.0, fill=False, lw=0.8, ec="#D8DEE6"))
    specs = [
        DaguaText(x_left, 56.0, "AV VA To AT LT fy", font_size=14.0, ha="left"),
        DaguaText(
            x_left,
            45.0,
            "**CPU:**  98%",
            rich=True,
            font_size=14.0,
            ha="left",
            background="#F8F3EB",
        ),
        DaguaText(x_left, 34.0, "WAVE", font_size=14.0, ha="left"),
        DaguaText(x_left, 23.0, "wave", font_size=14.0, ha="left"),
        DaguaText(x_left, 12.0, "Room  204    Gate  A1", font_size=14.0, ha="left"),
    ]
    artists = _render_specs(ax, specs)
    return fig, ax, artists


def _scene_consistent_baselines() -> Tuple[Any, Any, List[Any]]:
    """Render two multiline blocks that should share the same line spacing.

    Returns
    -------
    tuple[Any, Any, list[Any]]
        Figure, axes, and created artists.
    """
    fig, ax = _make_axes()
    ax.add_patch(Rectangle((20.0, 18.0), 24.0, 28.0, fill=False, lw=0.8, ec="#C8CDD4"))
    ax.add_patch(Rectangle((56.0, 18.0), 24.0, 28.0, fill=False, lw=0.8, ec="#C8CDD4"))
    specs = [
        DaguaText(32.0, 32.0, "ABC\nDEF", font_size=11.0, ha="center", va="center"),
        DaguaText(68.0, 32.0, "gyp\nqj", font_size=11.0, ha="center", va="center"),
    ]
    artists = _render_specs(ax, specs)
    return fig, ax, artists


_SCENES: List[Tuple[str, str, Callable[[], Tuple[Any, Any, List[Any]]]]] = [
    ("basic_text", "basic_text.png", _scene_basic_text),
    ("alignment_grid", "alignment_grid.png", _scene_alignment_grid),
    ("font_ladder", "font_ladder.png", _scene_font_ladder),
    ("rich_text_showcase", "rich_text_showcase.png", _scene_rich_text_showcase),
    ("decorations", "decorations.png", _scene_decorations),
    ("rotation_fan", "rotation_fan.png", _scene_rotation_fan),
    ("multiline_alignment", "multiline_alignment.png", _scene_multiline_alignment),
    ("node_label_simulation", "node_label_simulation.png", _scene_node_label_simulation),
    ("edge_label_simulation", "edge_label_simulation.png", _scene_edge_label_simulation),
    ("cluster_label_simulation", "cluster_label_simulation.png", _scene_cluster_label_simulation),
    ("dark_theme", "dark_theme.png", _scene_dark_theme),
    ("zoom_scaling", "zoom_scaling.png", _scene_zoom_scaling),
    ("matplotlib_comparison", "matplotlib_comparison.png", _scene_matplotlib_comparison),
    ("dense_graph_labels", "dense_graph_labels.png", _scene_dense_graph_labels),
    ("special_characters", "special_characters.png", _scene_special_characters),
    ("weight_spectrum", "weight_spectrum.png", _scene_weight_spectrum),
    ("mixed_features", "mixed_features.png", _scene_mixed_features),
    ("clip_shapes", "clip_shapes.png", _scene_clip_shapes),
    ("spacing_test", "spacing_test.png", _scene_spacing_test),
    ("consistent_baselines", "consistent_baselines.png", _scene_consistent_baselines),
]


def test_font_metrics_stable() -> None:
    """Stable metrics must not depend on the specific glyph content."""
    metrics_a = get_font_metrics(12.0, font_family="DejaVu Sans")
    metrics_b = get_font_metrics(12.0, font_family="DejaVu Sans")
    assert metrics_a == metrics_b


def test_font_metrics_scales() -> None:
    """Doubling the size must double stable metrics."""
    small = get_font_metrics(8.0, font_family="DejaVu Sans")
    large = get_font_metrics(16.0, font_family="DejaVu Sans")
    assert large.ascent == pytest.approx(small.ascent * 2.0)
    assert large.descent == pytest.approx(small.descent * 2.0)
    assert large.line_height == pytest.approx(small.line_height * 2.0)


def test_font_metrics_bold_different() -> None:
    """Bold text should expose different stable metrics than regular text."""
    regular = get_font_metrics(12.0, font_family="DejaVu Sans", font_weight="regular")
    bold = get_font_metrics(12.0, font_family="DejaVu Sans", font_weight="bold")
    assert regular != bold


def test_text_to_glyphs_basic() -> None:
    """Basic glyph conversion must return a valid non-empty path."""
    glyph_run = text_to_glyphs("Hello", 12.0, font_family="DejaVu Sans")
    assert glyph_run.advance_width > 0.0
    assert glyph_run.path.vertices.shape[0] > 0


def test_text_to_glyphs_empty() -> None:
    """Empty strings should produce zero-width empty paths."""
    glyph_run = text_to_glyphs("", 12.0, font_family="DejaVu Sans")
    assert glyph_run.advance_width == pytest.approx(0.0)
    assert glyph_run.path.vertices.shape == (0, 2)


def test_text_to_glyphs_space() -> None:
    """Space-only strings must preserve advance width even without outlines."""
    glyph_run = text_to_glyphs(" ", 12.0, font_family="DejaVu Sans")
    assert glyph_run.advance_width > 0.0
    assert glyph_run.path.vertices.shape == (0, 2)


def test_text_to_glyphs_multichar() -> None:
    """Longer strings should have greater advance widths."""
    short = text_to_glyphs("A", 12.0, font_family="DejaVu Sans")
    long = text_to_glyphs("AAAA", 12.0, font_family="DejaVu Sans")
    assert long.advance_width > short.advance_width


def test_text_to_glyphs_bold_wider() -> None:
    """Bold glyph runs should use wider advance widths for the same content."""
    regular = text_to_glyphs("Hello", 12.0, font_family="DejaVu Sans", font_weight="regular")
    bold = text_to_glyphs("Hello", 12.0, font_family="DejaVu Sans", font_weight="bold")
    assert bold.advance_width > regular.advance_width


def test_text_to_glyphs_size_scales() -> None:
    """Doubling size must double path advances and metrics."""
    small = text_to_glyphs("Scale", 8.0, font_family="DejaVu Sans")
    large = text_to_glyphs("Scale", 16.0, font_family="DejaVu Sans")
    assert large.advance_width == pytest.approx(small.advance_width * 2.0)
    assert large.metrics.line_height == pytest.approx(small.metrics.line_height * 2.0)


def test_text_to_glyphs_italic() -> None:
    """Italic text should still produce a valid glyph path."""
    glyph_run = text_to_glyphs("office", 12.0, font_family="DejaVu Serif", font_style="italic")
    assert glyph_run.advance_width > 0.0
    assert glyph_run.path.vertices.shape[0] > 0


def test_advance_width_vs_bbox() -> None:
    """Advance widths must differ from glyph extents for italic overhang cases."""
    glyph_run = text_to_glyphs("office", 12.0, font_family="DejaVu Serif", font_style="italic")
    bbox_width = glyph_run.path.get_extents().width
    assert glyph_run.advance_width != pytest.approx(bbox_width)


def test_measure_matches_glyphs() -> None:
    """Single-line measurement must match glyph conversion metrics."""
    width, height = measure_text_data("Hello", 12.0, font_family="DejaVu Sans")
    glyph_run = text_to_glyphs("Hello", 12.0, font_family="DejaVu Sans")
    assert width == pytest.approx(glyph_run.advance_width)
    assert height == pytest.approx(glyph_run.metrics.line_height)


def test_layout_plain_single_line() -> None:
    """Single-line layout must create one line with positive size."""
    block = layout_plain_text("Hello", 12.0, font_family="DejaVu Sans")
    assert len(block.lines) == 1
    assert block.width > 0.0
    assert block.height > 0.0


def test_layout_plain_multiline() -> None:
    """Multiline layout height must use stable line spacing."""
    block = layout_plain_text("One\nTwo", 12.0, font_family="DejaVu Sans")
    metrics = get_font_metrics(12.0, font_family="DejaVu Sans")
    assert block.height == pytest.approx(2.0 * metrics.line_height * 1.2)


def test_layout_plain_consistent_height() -> None:
    """Different glyph content must not change multiline block height."""
    letters = layout_plain_text("ABC\nDEF", 12.0, font_family="DejaVu Sans")
    descenders = layout_plain_text("gyp\nqj", 12.0, font_family="DejaVu Sans")
    assert letters.height == pytest.approx(descenders.height)


@pytest.mark.parametrize(
    ("ha", "expected"),
    [
        ("left", 0.0),
        ("center", None),
        ("right", None),
    ],
)
def test_layout_plain_horizontal_alignment(ha: str, expected: Optional[float]) -> None:
    """Horizontal alignment must shift the block origin correctly."""
    block = layout_plain_text("Hello", 12.0, ha=ha, font_family="DejaVu Sans")
    if expected is not None:
        assert block.x_offset == pytest.approx(expected)
    elif ha == "center":
        assert block.x_offset == pytest.approx(-block.width / 2.0)
    else:
        assert block.x_offset == pytest.approx(-block.width)


@pytest.mark.parametrize(
    ("va", "expected_name"),
    [
        ("top", "top"),
        ("center", "center"),
        ("bottom", "bottom"),
        ("baseline", "baseline"),
    ],
)
def test_layout_plain_vertical_alignment(va: str, expected_name: str) -> None:
    """Vertical alignment must use block height or ascent as specified."""
    block = layout_plain_text("Hello", 12.0, va=va, font_family="DejaVu Sans")
    metrics = get_font_metrics(12.0, font_family="DejaVu Sans")
    if expected_name == "top":
        assert block.y_offset == pytest.approx(0.0)
    elif expected_name == "center":
        assert block.y_offset == pytest.approx(block.height / 2.0)
    elif expected_name == "bottom":
        assert block.y_offset == pytest.approx(block.height)
    else:
        assert block.y_offset == pytest.approx(metrics.ascent)


def test_layout_rich_bold() -> None:
    """Rich-text bold markup must promote segment weight."""
    block = layout_rich_text("**Bold**", 12.0, font_family="DejaVu Sans")
    assert block.lines[0].segments[0].is_bold is True


def test_layout_rich_italic() -> None:
    """Rich-text italic markup must promote segment style."""
    block = layout_rich_text("*Italic*", 12.0, font_family="DejaVu Sans")
    assert block.lines[0].segments[0].is_italic is True


def test_layout_rich_color() -> None:
    """Rich-text color markup must override segment color."""
    block = layout_rich_text("{color:#ff0000}Red{/color}", 12.0, font_family="DejaVu Sans")
    assert block.lines[0].segments[0].color == "#ff0000"


def test_layout_rich_mixed() -> None:
    """Mixed rich markup must create multiple styled segments."""
    block = layout_rich_text("**Bold** *Italic* `Mono`", 12.0, font_family="DejaVu Sans")
    assert len(block.lines[0].segments) >= 3


def test_layout_rich_multiline() -> None:
    """Rich-text newlines must create multiple lines."""
    block = layout_rich_text("**A**\n*B*", 12.0, font_family="DejaVu Sans")
    assert len(block.lines) == 2


def test_layout_rich_spaces() -> None:
    """Double spaces in rich text must preserve measurable width."""
    spaced = layout_rich_text("**CPU:**  98%", 12.0, font_family="DejaVu Sans")
    compact = layout_rich_text("**CPU:** 98%", 12.0, font_family="DejaVu Sans")
    assert spaced.width > compact.width


def test_layout_secondary_scale() -> None:
    """Secondary lines must use a reduced glyph scale when requested."""
    block = layout_plain_text(
        "Primary\nSecondary",
        12.0,
        font_family="DejaVu Sans",
        secondary_scale=0.8,
    )
    first_metrics = block.lines[0].segments[0].glyph_run.metrics
    second_metrics = block.lines[1].segments[0].glyph_run.metrics
    assert second_metrics.line_height < first_metrics.line_height


def test_max_width_shrink() -> None:
    """Shrink-to-fit must reduce the line width below the requested maximum."""
    original = layout_plain_text("A very long label", 12.0, font_family="DejaVu Sans")
    shrunk = layout_plain_text(
        "A very long label",
        12.0,
        font_family="DejaVu Sans",
        max_width=original.width * 0.6,
        min_size_data=4.0,
    )
    assert shrunk.width <= original.width * 0.6 + 1e-6
    assert (
        shrunk.lines[0].segments[0].glyph_run.metrics.line_height
        < original.lines[0].segments[0].glyph_run.metrics.line_height
    )


def test_max_width_floor() -> None:
    """Shrink-to-fit must stop at the minimum requested size."""
    min_size_data = 6.0
    block = layout_plain_text(
        "A very long label",
        14.0,
        font_family="DejaVu Sans",
        max_width=1.0,
        min_size_data=min_size_data,
    )
    floor_metrics = get_font_metrics(min_size_data, font_family="DejaVu Sans")
    assert block.lines[0].segments[0].glyph_run.metrics.line_height == pytest.approx(
        floor_metrics.line_height
    )


def test_background_rect_path() -> None:
    """Background rectangles must have the requested padded bounds."""
    path = background_rect_path(0.0, 0.0, 10.0, 4.0, 2.0, 1.0, 0.0)
    bbox = path.get_extents()
    assert bbox.width == pytest.approx(14.0)
    assert bbox.height == pytest.approx(6.0)


def test_background_rect_rounded() -> None:
    """Rounded backgrounds must include Bezier curve segments."""
    path = background_rect_path(0.0, 0.0, 10.0, 4.0, 2.0, 1.0, 1.5)
    assert MplPath.CURVE4 in path.codes


def test_underline_path() -> None:
    """Underline paths must build horizontal rectangles."""
    path = underline_path(0.0, 10.0, -1.0, 0.5)
    bbox = path.get_extents()
    assert bbox.width == pytest.approx(10.0)
    assert bbox.height == pytest.approx(0.5)


def test_strikethrough_path() -> None:
    """Strikethrough paths must build horizontal rectangles."""
    path = strikethrough_path(0.0, 10.0, 1.0, 0.5)
    bbox = path.get_extents()
    assert bbox.width == pytest.approx(10.0)
    assert bbox.height == pytest.approx(0.5)


def test_cache_hit() -> None:
    """Repeated glyph conversions must hit the glyph cache."""
    before = _cached_glyph_data.cache_info().hits
    text_to_glyphs("Cache", 12.0, font_family="DejaVu Sans")
    text_to_glyphs("Cache", 16.0, font_family="DejaVu Sans")
    after = _cached_glyph_data.cache_info().hits
    assert after > before


def test_cache_float_drift() -> None:
    """Rounded metric cache keys must absorb tiny float drift."""
    before = _cached_font_metrics.cache_info().hits
    get_font_metrics(1.00001, font_family="DejaVu Sans")
    get_font_metrics(1.00009, font_family="DejaVu Sans")
    after = _cached_font_metrics.cache_info().hits
    assert after > before


def test_measure_text_multiline() -> None:
    """Shared utility measurement must use stable multiline height."""
    line_height = measure_text("Hg", font_family=RESOLVED_FONT, font_size=10.0)[1]
    width, height = measure_text("ABC\nDEF", font_family=RESOLVED_FONT, font_size=10.0)
    single_width = max(
        measure_text("ABC", font_family=RESOLVED_FONT, font_size=10.0)[0],
        measure_text("DEF", font_family=RESOLVED_FONT, font_size=10.0)[0],
    )
    assert width == pytest.approx(single_width)
    assert height == pytest.approx(2.0 * line_height * 1.2)


def test_measure_text_italic_wider() -> None:
    """Italic measurement must differ for widths when the font variant changes."""
    normal_width = measure_text(
        "office",
        font_family=RESOLVED_FONT,
        font_size=12.0,
        font_style="normal",
    )[0]
    italic_width = measure_text(
        "office",
        font_family=RESOLVED_FONT,
        font_size=12.0,
        font_style="italic",
    )[0]
    assert italic_width != pytest.approx(normal_width)


def test_compute_node_size_italic() -> None:
    """Italic-aware node sizing must propagate font style through the sizing chain."""
    normal_width, _, _ = compute_node_size(
        "office",
        font_family=RESOLVED_FONT,
        font_size=12.0,
        font_style="normal",
    )
    italic_width, _, _ = compute_node_size(
        "office",
        font_family=RESOLVED_FONT,
        font_size=12.0,
        font_style="italic",
    )
    assert italic_width != pytest.approx(normal_width)


def test_graph_compute_node_sizes_forwards_font_style() -> None:
    """Graph-level node sizing must include style font-style overrides."""
    graph = DaguaGraph.from_edge_list([("office", "other")])
    graph.node_styles[0] = NodeStyle(font_style="italic")
    graph.compute_node_sizes()
    assert graph.node_sizes is not None
    assert graph.node_sizes[0, 0].item() > 0.0


def test_render_text_basic_artist_count() -> None:
    """Rendering a simple label should create at least one artist."""
    fig, ax = _make_axes()
    artists = _render_specs(ax, [DaguaText(50.0, 35.0, "Hello", font_size=12.0)])
    plt.close(fig)
    assert artists


def test_render_text_background_and_outline() -> None:
    """Backgrounds and outlines should add extra artists to the scene."""
    fig, ax = _make_axes()
    artists = _render_specs(
        ax,
        [
            DaguaText(
                50.0,
                35.0,
                "Readable",
                font_size=12.0,
                background="#F6F1E7",
                outline=True,
            )
        ],
    )
    plt.close(fig)
    assert len(artists) >= 3


def test_render_text_outline_clamps_to_visible_width() -> None:
    """Outline strokes should keep a minimum width even for tiny style values."""

    fig, ax = _make_axes()
    artists = _render_specs(
        ax,
        [
            DaguaText(
                50.0,
                35.0,
                "Outline",
                font_size=12.0,
                outline=True,
                outline_width=0.5,
                gid="outline-clamp",
            )
        ],
    )
    plt.close(fig)

    outline_patches = [
        artist for artist in artists if isinstance(artist, PathPatch) and artist.get_gid()
    ]
    assert any(
        str(patch.get_gid()) == "outline-clamp-outline-0-0"
        and float(patch.get_linewidth()) == pytest.approx(2.0)
        for patch in outline_patches
    )


def test_render_text_bold_adds_emphasis_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bold labels should add a same-color emphasis stroke behind the fill."""

    monkeypatch.setattr(text_collection, "_segment_needs_bold_emphasis", lambda spec, segment: True)
    fig, ax = _make_axes()
    artists = _render_specs(
        ax,
        [
            DaguaText(
                50.0,
                35.0,
                "Bold",
                font_size=12.0,
                font_weight="bold",
                gid="bold-emphasis",
            )
        ],
    )
    plt.close(fig)

    assert any(
        isinstance(artist, PathPatch) and str(artist.get_gid()) == "bold-emphasis-embolden-0-0"
        for artist in artists
    )


def test_render_text_clip_patch() -> None:
    """Clip patches must be accepted as artist-based clipping proxies."""
    fig, ax = _make_axes()
    clip_patch = _rectangle_clip(ax, 50.0, 35.0, 12.0, 8.0)
    artists = _render_specs(
        ax,
        [DaguaText(50.0, 35.0, "Clipped", font_size=12.0, clip_patch=clip_patch)],
    )
    plt.close(fig)
    assert artists


@pytest.mark.parametrize(
    ("scene_name", "filename", "builder"),
    _SCENES,
    ids=[scene[0] for scene in _SCENES],
)
def test_render_comparison_scene(
    scene_name: str,
    filename: str,
    builder: Callable[[], Tuple[Any, Any, List[Any]]],
) -> None:
    """Render and save each requested comparison scene."""
    _ = scene_name
    fig, _, artists = builder()
    output = _save_scene(fig, filename)
    plt.close(fig)
    assert output.exists()
    assert output.stat().st_size > 0
    assert artists or filename == "matplotlib_comparison.png"
