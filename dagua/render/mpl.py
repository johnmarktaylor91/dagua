"""Matplotlib renderer for DaguaGraph.

Publication-quality rendering following the Dagua Aesthetic Style Guide:
- Wong/Okabe-Ito colorblind-safe palette
- Muted fills, strong borders, quiet edges
- Helvetica/Arial typography
- Warm white background (#FAFAFA)
- Layered rendering: clusters → edges → nodes → labels
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from dagua.edges import BezierCurve, evaluate_bezier, route_edges
from dagua.styles import (
    ClusterStyle,
    EdgeStyle,
    FONT_FAMILY,
    MEDIUM_GRAY,
    NEAR_BLACK,
    RESOLVED_FONT,
    WARM_WHITE,
)
from dagua.utils import collect_cluster_leaves


def render(
    graph,
    positions,
    config=None,
    output: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 150,
    show: bool = False,
    title: Optional[str] = None,
):
    """Render graph with computed positions.

    Args:
        graph: DaguaGraph instance
        positions: [N, 2] tensor of node positions
        config: LayoutConfig (optional)
        output: file path to save (PNG, SVG, PDF)
        figsize: figure size in inches
        dpi: resolution for raster output
        show: whether to call plt.show()
        title: optional title for the figure

    Returns:
        (fig, ax) matplotlib objects
    """
    import warnings
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Set global font preferences (use resolved font to avoid warnings)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "findfont")
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = FONT_FAMILY

    pos = positions.detach().cpu().numpy()
    n = graph.num_nodes

    if n == 0:
        fig, ax = plt.subplots(1, 1, figsize=figsize or (6, 4))
        fig.patch.set_facecolor(WARM_WHITE)
        if output:
            fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=WARM_WHITE)
        return fig, ax

    # Compute figure bounds
    graph.compute_node_sizes()
    sizes = graph.node_sizes.detach().cpu().numpy()

    margin = 30
    x_min = (pos[:, 0] - sizes[:, 0] / 2).min() - margin
    x_max = (pos[:, 0] + sizes[:, 0] / 2).max() + margin
    y_min = (pos[:, 1] - sizes[:, 1] / 2).min() - margin
    y_max = (pos[:, 1] + sizes[:, 1] / 2).max() + margin

    width = x_max - x_min
    height = y_max - y_min

    if figsize is None:
        scale = max(1.0, min(width / 100, 30))
        aspect = height / max(width, 1)
        fig_w = min(max(scale, 4), 30)
        fig_h = min(max(fig_w * aspect, 3), 40)
        figsize = (fig_w, fig_h)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.patch.set_facecolor(WARM_WHITE)
    ax.set_facecolor(WARM_WHITE)

    # --- Layer 0: Cluster backgrounds ---
    _draw_clusters(ax, graph, pos, sizes)

    # --- Layer 1: Edges ---
    curves = route_edges(positions, graph.edge_index, graph.node_sizes, graph.direction)
    _draw_edges(ax, graph, curves)

    # --- Layer 2: Nodes ---
    _draw_nodes(ax, graph, pos, sizes)

    # --- Layer 3: Node labels ---
    _draw_node_labels(ax, graph, pos, sizes)

    # --- Layer 4: Edge labels ---
    _draw_edge_labels(ax, graph, curves)

    # Configure axes
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=10, fontweight="regular", color=NEAR_BLACK,
                      fontfamily=FONT_FAMILY[0])

    plt.tight_layout()

    if output:
        fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=WARM_WHITE)

    if show:
        plt.show()

    return fig, ax


def _draw_nodes(ax, graph, pos, sizes):
    """Draw node shapes with muted fills and strong borders."""
    from matplotlib.patches import FancyBboxPatch, Ellipse, Circle

    for i in range(graph.num_nodes):
        x, y = pos[i, 0], pos[i, 1]
        w, h = sizes[i, 0], sizes[i, 1]
        style = graph.get_style_for_node(i)

        # Corner radius proportional to shorter dimension (18%)
        shorter = min(w, h)
        cr = shorter * 0.18

        if style.shape in ("roundrect", "rect"):
            pad = cr * 0.01 if style.shape == "roundrect" else 0
            boxstyle = f"round,pad={pad}" if style.shape == "roundrect" else "square,pad=0"
            patch = FancyBboxPatch(
                (x - w / 2, y - h / 2), w, h,
                boxstyle=boxstyle,
                facecolor=style.fill,
                edgecolor=style.stroke,
                linewidth=style.stroke_width,
                linestyle="--" if style.stroke_dash == "dashed" else "-",
                alpha=style.opacity,
                zorder=2,
            )
        elif style.shape == "ellipse":
            patch = Ellipse(
                (x, y), w, h,
                facecolor=style.fill,
                edgecolor=style.stroke,
                linewidth=style.stroke_width,
                linestyle="--" if style.stroke_dash == "dashed" else "-",
                alpha=style.opacity,
                zorder=2,
            )
        elif style.shape == "circle":
            r = max(w, h) / 2
            patch = Circle(
                (x, y), r,
                facecolor=style.fill,
                edgecolor=style.stroke,
                linewidth=style.stroke_width,
                alpha=style.opacity,
                zorder=2,
            )
        elif style.shape == "diamond":
            from matplotlib.patches import Polygon
            pts = np.array([
                [x, y + h / 2],
                [x + w / 2, y],
                [x, y - h / 2],
                [x - w / 2, y],
            ])
            patch = Polygon(
                pts, closed=True,
                facecolor=style.fill,
                edgecolor=style.stroke,
                linewidth=style.stroke_width,
                alpha=style.opacity,
                zorder=2,
            )
        else:
            patch = FancyBboxPatch(
                (x - w / 2, y - h / 2), w, h,
                boxstyle="round,pad=0.02",
                facecolor=style.fill,
                edgecolor=style.stroke,
                linewidth=style.stroke_width,
                alpha=style.opacity,
                zorder=2,
            )

        ax.add_patch(patch)


def _draw_node_labels(ax, graph, pos, sizes):
    """Draw centered text labels inside nodes."""
    for i in range(graph.num_nodes):
        x, y = pos[i, 0], pos[i, 1]
        style = graph.get_style_for_node(i)
        label = graph.node_labels[i]

        lines = label.split("\n")
        fontsize = style.font_size
        font_family = style.font_family_list

        if len(lines) == 1:
            ax.text(
                x, y, label,
                ha="center", va="center",
                fontsize=fontsize,
                fontfamily=font_family[0],
                color=style.font_color,
                fontweight="regular",
                zorder=3,
                clip_on=True,
            )
        else:
            line_height = fontsize * 1.2
            total_height = line_height * len(lines)
            start_y = y + total_height / 2 - line_height / 2

            for j, line in enumerate(lines):
                ly = start_y - j * line_height
                # Secondary lines slightly smaller
                fs = fontsize if j == 0 else fontsize * 0.85
                ax.text(
                    x, ly, line,
                    ha="center", va="center",
                    fontsize=fs,
                    fontfamily=font_family[0],
                    color=style.font_color,
                    fontweight="regular",
                    zorder=3,
                    clip_on=True,
                )


def _draw_edges(ax, graph, curves: List[BezierCurve]):
    """Draw bezier edges with arrowheads.

    Edges are quiet: medium gray at 70% opacity, 0.75pt width.
    Arrowheads: small filled triangles (5pt × 3.5pt).
    """
    from matplotlib.path import Path
    from matplotlib.patches import FancyArrowPatch

    for e_idx, curve in enumerate(curves):
        style = graph.get_style_for_edge(e_idx)

        verts = [curve.p0, curve.cp1, curve.cp2, curve.p1]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        path = Path(verts, codes)

        # Arrow style: small filled triangle per style guide
        arrow_l = style.arrow_length
        arrow_w = style.arrow_width
        arrowstyle = f"->,head_length={arrow_l},head_width={arrow_w}"

        linestyle = "-"
        if style.style == "dashed":
            linestyle = "--"
        elif style.style == "dotted":
            linestyle = "-."

        arrow = FancyArrowPatch(
            path=path,
            arrowstyle=arrowstyle,
            color=style.color,
            linewidth=style.width,
            linestyle=linestyle,
            alpha=style.opacity,
            zorder=1,
            mutation_scale=1,
        )
        ax.add_patch(arrow)


def _draw_edge_labels(ax, graph, curves: List[BezierCurve]):
    """Draw edge labels offset from the curve midpoint.

    Labels: 7pt regular, dark gray, with subtle background.
    """
    for e_idx, curve in enumerate(curves):
        if e_idx >= len(graph.edge_labels):
            break
        label = graph.edge_labels[e_idx]
        if not label:
            continue

        mid = evaluate_bezier(curve, 0.5)

        # Offset 4pt perpendicular (approximated as upward)
        label_offset = 4.0
        ax.text(
            mid[0], mid[1] + label_offset, label,
            ha="center", va="center",
            fontsize=7.0,
            fontweight="regular",
            fontfamily=FONT_FAMILY[0],
            color=NEAR_BLACK,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor=WARM_WHITE,
                edgecolor="none",
                alpha=0.85,
            ),
            zorder=4,
        )


def _draw_clusters(ax, graph, pos, sizes):
    """Draw cluster background boxes.

    Barely-there fills, thin borders, progressive darkening for nesting.
    Labels: top-left, left-aligned.
    """
    from matplotlib.patches import FancyBboxPatch

    if not graph.clusters:
        return

    sorted_clusters = sorted(
        graph.clusters.items(),
        key=lambda kv: len(collect_cluster_leaves(kv[1]) if isinstance(kv[1], dict) else kv[1]),
        reverse=True,
    )

    for depth, (name, members) in enumerate(sorted_clusters):
        if isinstance(members, dict):
            indices = collect_cluster_leaves(members)
        else:
            indices = members

        if not indices:
            continue

        style = graph.cluster_styles.get(name, ClusterStyle())
        padding = style.padding

        member_pos = pos[indices]
        member_sizes = sizes[indices]

        x_min = (member_pos[:, 0] - member_sizes[:, 0] / 2).min() - padding
        x_max = (member_pos[:, 0] + member_sizes[:, 0] / 2).max() + padding
        y_min = (member_pos[:, 1] - member_sizes[:, 1] / 2).min() - padding
        y_max = (member_pos[:, 1] + member_sizes[:, 1] / 2).max() + padding + 14  # space for label

        # Progressive nesting: deeper clusters get slightly darker fills
        level = min(depth, len(ClusterStyle.LEVEL_FILLS) - 1)
        fill_color = ClusterStyle.LEVEL_FILLS[level]
        stroke_color = ClusterStyle.LEVEL_STROKES[level]

        # Opacity decreases with depth
        max_depth = len(sorted_clusters)
        opacity = style.opacity * (1 - depth * 0.15 / max(max_depth, 1))
        opacity = max(opacity, 0.08)

        patch = FancyBboxPatch(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            boxstyle="round,pad=0",
            facecolor=fill_color,
            edgecolor=stroke_color,
            linewidth=style.stroke_width,
            alpha=opacity,
            zorder=0,
        )
        ax.add_patch(patch)

        # Cluster label: top-left, left-aligned
        label = graph.cluster_labels.get(name, name)
        # Font size decreases by 1pt per nesting level (min 7pt)
        label_fontsize = max(style.font_size - depth * 1.0, 7.0)
        ax.text(
            x_min + 6, y_max - 6, label,
            fontsize=label_fontsize,
            fontweight=style.font_weight,
            fontfamily=FONT_FAMILY[0],
            color=style.font_color,
            va="top", ha="left",
            zorder=0.5,
        )
