"""Matplotlib renderer for DaguaGraph.

Publication-quality rendering following the Dagua Aesthetic Style Guide:
- Wong/Okabe-Ito colorblind-safe palette
- Muted fills, strong borders, quiet edges
- Helvetica/Arial typography
- Warm white background (#FAFAFA)
- Layered rendering: clusters -> edges -> nodes -> labels
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
    darken_hex,
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

    gs = graph.graph_style

    # Set global font preferences (use resolved font to avoid warnings)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "findfont")
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = FONT_FAMILY

    pos = positions.detach().cpu().numpy()
    n = graph.num_nodes
    bg = gs.background_color

    if n == 0:
        fig, ax = plt.subplots(1, 1, figsize=figsize or (6, 4))
        fig.patch.set_facecolor(bg)
        if output:
            fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=bg)
        return fig, ax

    # Compute figure bounds
    graph.compute_node_sizes()
    sizes = graph.node_sizes.detach().cpu().numpy()

    margin = gs.margin
    x_min = (pos[:, 0] - sizes[:, 0] / 2).min() - margin
    x_max = (pos[:, 0] + sizes[:, 0] / 2).max() + margin
    y_min = (pos[:, 1] - sizes[:, 1] / 2).min() - margin
    y_max = (pos[:, 1] + sizes[:, 1] / 2).max() + margin

    width = x_max - x_min
    height = y_max - y_min

    if figsize is None:
        max_w, max_h = gs.max_figsize
        min_w, min_h = gs.min_figsize
        scale = max(1.0, min(width / 100, max_w))
        aspect = height / max(width, 1)
        fig_w = min(max(scale, min_w), max_w)
        fig_h = min(max(fig_w * aspect, min_h), max_h)
        figsize = (fig_w, fig_h)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    # --- Layer 0: Cluster backgrounds ---
    _draw_clusters(ax, graph, pos, sizes)

    # --- Layer 1: Edges ---
    curves = route_edges(positions, graph.edge_index, graph.node_sizes, graph.direction, graph)
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
        title_ff = gs.title_font_family or FONT_FAMILY[0]
        ax.set_title(title, fontsize=gs.title_font_size, fontweight=gs.title_font_weight,
                      color=gs.title_font_color, fontfamily=title_ff)

    plt.tight_layout()

    if output:
        fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=bg)

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

        cr = style.corner_radius

        linestyle = "-"
        if style.stroke_dash == "dashed":
            linestyle = "--"

        # Shadow (render-only decoration)
        if style.shadow:
            _draw_shadow(ax, x, y, w, h, style)

        if style.shape in ("roundrect", "rect"):
            pad = cr * 0.01 if style.shape == "roundrect" else 0
            boxstyle = f"round,pad={pad}" if style.shape == "roundrect" else "square,pad=0"
            patch = FancyBboxPatch(
                (x - w / 2, y - h / 2), w, h,
                boxstyle=boxstyle,
                facecolor=style.fill,
                edgecolor=style.stroke,
                linewidth=style.stroke_width,
                linestyle=linestyle,
                alpha=style.opacity,
                zorder=2,
            )
        elif style.shape == "ellipse":
            patch = Ellipse(
                (x, y), w, h,
                facecolor=style.fill,
                edgecolor=style.stroke,
                linewidth=style.stroke_width,
                linestyle=linestyle,
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
                linestyle=linestyle,
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
                linestyle=linestyle,
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
                linestyle=linestyle,
                alpha=style.opacity,
                zorder=2,
            )

        ax.add_patch(patch)


def _draw_shadow(ax, x, y, w, h, style):
    """Draw a shadow offset duplicate patch behind the node."""
    from matplotlib.patches import FancyBboxPatch, Ellipse, Circle

    ox, oy = style.shadow_offset
    shadow_color = style.shadow_color

    if style.shape in ("roundrect", "rect"):
        cr = style.corner_radius
        pad = cr * 0.01 if style.shape == "roundrect" else 0
        boxstyle = f"round,pad={pad}" if style.shape == "roundrect" else "square,pad=0"
        shadow = FancyBboxPatch(
            (x - w / 2 + ox, y - h / 2 + oy), w, h,
            boxstyle=boxstyle,
            facecolor=shadow_color,
            edgecolor="none",
            zorder=1.5,
        )
    elif style.shape == "ellipse":
        shadow = Ellipse(
            (x + ox, y + oy), w, h,
            facecolor=shadow_color,
            edgecolor="none",
            zorder=1.5,
        )
    elif style.shape == "circle":
        r = max(w, h) / 2
        shadow = Circle(
            (x + ox, y + oy), r,
            facecolor=shadow_color,
            edgecolor="none",
            zorder=1.5,
        )
    elif style.shape == "diamond":
        from matplotlib.patches import Polygon
        pts = np.array([
            [x + ox, y + h / 2 + oy],
            [x + w / 2 + ox, y + oy],
            [x + ox, y - h / 2 + oy],
            [x - w / 2 + ox, y + oy],
        ])
        shadow = Polygon(pts, closed=True, facecolor=shadow_color, edgecolor="none", zorder=1.5)
    else:
        shadow = FancyBboxPatch(
            (x - w / 2 + ox, y - h / 2 + oy), w, h,
            boxstyle="round,pad=0.02",
            facecolor=shadow_color,
            edgecolor="none",
            zorder=1.5,
        )

    ax.add_patch(shadow)


def _draw_node_labels(ax, graph, pos, sizes):
    """Draw centered text labels inside nodes."""
    gs = graph.graph_style

    for i in range(graph.num_nodes):
        x, y = pos[i, 0], pos[i, 1]
        style = graph.get_style_for_node(i)
        label = graph.node_labels[i]

        lines = label.split("\n")
        fontsize = style.font_size
        font_family = style.font_family_list
        font_weight = style.font_weight
        font_style = style.font_style

        if len(lines) == 1:
            ax.text(
                x, y, label,
                ha="center", va="center",
                fontsize=fontsize,
                fontfamily=font_family[0],
                color=style.font_color,
                fontweight=font_weight,
                fontstyle=font_style,
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
                fs = fontsize if j == 0 else fontsize * gs.node_label_secondary_scale
                ax.text(
                    x, ly, line,
                    ha="center", va="center",
                    fontsize=fs,
                    fontfamily=font_family[0],
                    color=style.font_color,
                    fontweight=font_weight,
                    fontstyle=font_style,
                    zorder=3,
                    clip_on=True,
                )


def _draw_edges(ax, graph, curves: List[BezierCurve]):
    """Draw bezier edges with arrowheads.

    Edges are quiet: medium gray at 70% opacity, 0.75pt width.
    Arrowheads: small filled triangles (5pt x 3.5pt).
    """
    from matplotlib.path import Path
    from matplotlib.patches import FancyArrowPatch

    for e_idx, curve in enumerate(curves):
        style = graph.get_style_for_edge(e_idx)

        verts = [curve.p0, curve.cp1, curve.cp2, curve.p1]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        path = Path(verts, codes)

        # Arrow style: "none" -> no arrowhead, otherwise small filled triangle
        if style.arrow == "none":
            arrowstyle = "-"
        else:
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

    Uses per-edge style for font size/color/background, with fallback to graph_style.
    """
    gs = graph.graph_style

    for e_idx, curve in enumerate(curves):
        if e_idx >= len(graph.edge_labels):
            break
        label = graph.edge_labels[e_idx]
        if not label:
            continue

        style = graph.get_style_for_edge(e_idx)
        mid = evaluate_bezier(curve, 0.5)

        # Per-edge style or fall back to graph style
        font_size = style.label_font_size
        font_color = style.label_font_color
        label_bg = style.label_background
        bg_opacity = gs.edge_label_background_opacity

        # Offset 4pt perpendicular (approximated as upward)
        label_offset = 4.0
        ax.text(
            mid[0], mid[1] + label_offset, label,
            ha="center", va="center",
            fontsize=font_size,
            fontweight="regular",
            fontfamily=FONT_FAMILY[0],
            color=font_color,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor=label_bg,
                edgecolor="none",
                alpha=bg_opacity,
            ),
            zorder=4,
        )


def _draw_clusters(ax, graph, pos, sizes):
    """Draw cluster background boxes.

    Barely-there fills, thin borders, progressive darkening for nesting.
    Labels: configurable position via label_position.
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

        style = graph.get_style_for_cluster(name)
        padding = style.padding

        member_pos = pos[indices]
        member_sizes = sizes[indices]

        x_min = (member_pos[:, 0] - member_sizes[:, 0] / 2).min() - padding
        x_max = (member_pos[:, 0] + member_sizes[:, 0] / 2).max() + padding
        y_min = (member_pos[:, 1] - member_sizes[:, 1] / 2).min() - padding
        y_max = (member_pos[:, 1] + member_sizes[:, 1] / 2).max() + padding + 14  # space for label

        # Progressive depth darkening using HSL (replaces LEVEL_FILLS/LEVEL_STROKES)
        fill_color = darken_hex(style.fill, depth * style.depth_fill_step)
        stroke_color = darken_hex(style.stroke, depth * style.depth_stroke_step)

        # Opacity decreases with depth
        max_depth = len(sorted_clusters)
        opacity = style.opacity * (1 - depth * 0.15 / max(max_depth, 1))
        opacity = max(opacity, 0.08)

        # Corner radius
        cr = style.corner_radius
        boxstyle = f"round,pad=0" if cr > 0 else "square,pad=0"

        # Stroke dash
        linestyle = "-"
        if style.stroke_dash == "dashed":
            linestyle = "--"
        elif style.stroke_dash == "dotted":
            linestyle = "-."

        patch = FancyBboxPatch(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            boxstyle=boxstyle,
            facecolor=fill_color,
            edgecolor=stroke_color,
            linewidth=style.stroke_width,
            linestyle=linestyle,
            alpha=opacity,
            zorder=0,
        )
        ax.add_patch(patch)

        # Cluster label: position from style
        label = graph.cluster_labels.get(name, name)
        label_fontsize = max(style.font_size - depth * 1.0, 7.0)
        label_ff = style.font_family or FONT_FAMILY[0]
        label_ox, label_oy = style.label_offset

        if style.label_position == "top-center":
            lx = (x_min + x_max) / 2
            ha = "center"
        elif style.label_position == "top-right":
            lx = x_max - label_ox
            ha = "right"
        else:  # "top-left" (default)
            lx = x_min + label_ox
            ha = "left"

        ly = y_max - label_oy

        ax.text(
            lx, ly, label,
            fontsize=label_fontsize,
            fontweight=style.font_weight,
            fontfamily=label_ff,
            color=style.font_color,
            va="top", ha=ha,
            zorder=0.5,
        )
