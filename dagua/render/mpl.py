"""Matplotlib renderer for DaguaGraph.

Renders nodes, edges, clusters, and labels using matplotlib.
Supports PNG, SVG, PDF output.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from dagua.edges import BezierCurve, evaluate_bezier, route_edges
from dagua.styles import ClusterStyle, EdgeStyle, NodeStyle
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.path import Path
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch

    pos = positions.detach().cpu().numpy()
    n = graph.num_nodes

    if n == 0:
        fig, ax = plt.subplots(1, 1, figsize=figsize or (6, 4))
        if output:
            fig.savefig(output, dpi=dpi, bbox_inches="tight")
        return fig, ax

    # Compute figure bounds
    graph.compute_node_sizes()
    sizes = graph.node_sizes.detach().cpu().numpy()

    x_min = (pos[:, 0] - sizes[:, 0] / 2).min() - 30
    x_max = (pos[:, 0] + sizes[:, 0] / 2).max() + 30
    y_min = (pos[:, 1] - sizes[:, 1] / 2).min() - 30
    y_max = (pos[:, 1] + sizes[:, 1] / 2).max() + 30

    width = x_max - x_min
    height = y_max - y_min

    if figsize is None:
        # Auto-size: aim for reasonable display
        scale = max(1.0, min(width / 100, 30))
        aspect = height / max(width, 1)
        fig_w = min(max(scale, 4), 30)
        fig_h = min(max(fig_w * aspect, 3), 40)
        figsize = (fig_w, fig_h)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

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
        ax.set_title(title, fontsize=12, fontweight="bold")

    plt.tight_layout()

    if output:
        fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")

    if show:
        plt.show()

    return fig, ax


def _draw_nodes(ax, graph, pos, sizes):
    """Draw node shapes."""
    from matplotlib.patches import FancyBboxPatch, Ellipse, Circle, Rectangle

    for i in range(graph.num_nodes):
        x, y = pos[i, 0], pos[i, 1]
        w, h = sizes[i, 0], sizes[i, 1]
        style = graph.get_style_for_node(i)

        if style.shape in ("roundrect", "rect"):
            rounding = style.corner_radius if style.shape == "roundrect" else 0
            pad = rounding * 0.01  # FancyBboxPatch uses relative pad
            patch = FancyBboxPatch(
                (x - w / 2, y - h / 2), w, h,
                boxstyle=f"round,pad={pad}" if style.shape == "roundrect" else "square,pad=0",
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
            # Fallback to rectangle
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

        # Split multi-line labels
        lines = label.split("\n")
        fontsize = style.font_size

        if len(lines) == 1:
            ax.text(
                x, y, label,
                ha="center", va="center",
                fontsize=fontsize,
                fontfamily=style.font_family,
                color=style.font_color,
                zorder=3,
                clip_on=True,
            )
        else:
            # Multi-line: stack vertically
            line_height = fontsize * 1.2
            total_height = line_height * len(lines)
            start_y = y + total_height / 2 - line_height / 2

            for j, line in enumerate(lines):
                ly = start_y - j * line_height
                fs = fontsize if j == 0 else fontsize * 0.8
                ax.text(
                    x, ly, line,
                    ha="center", va="center",
                    fontsize=fs,
                    fontfamily=style.font_family,
                    color=style.font_color,
                    zorder=3,
                    clip_on=True,
                )


def _draw_edges(ax, graph, curves: List[BezierCurve]):
    """Draw bezier edges with arrowheads."""
    from matplotlib.path import Path
    from matplotlib.patches import FancyArrowPatch

    for e_idx, curve in enumerate(curves):
        style = graph.get_style_for_edge(e_idx)

        # Build bezier path
        verts = [curve.p0, curve.cp1, curve.cp2, curve.p1]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        path = Path(verts, codes)

        arrow = FancyArrowPatch(
            path=path,
            arrowstyle="->,head_length=6,head_width=4",
            color=style.color,
            linewidth=style.width,
            linestyle="--" if style.style == "dashed" else ("-." if style.style == "dotted" else "-"),
            alpha=style.opacity,
            zorder=1,
            mutation_scale=1,
        )
        ax.add_patch(arrow)


def _draw_edge_labels(ax, graph, curves: List[BezierCurve]):
    """Draw edge labels at the midpoint of each curve."""
    for e_idx, curve in enumerate(curves):
        if e_idx >= len(graph.edge_labels):
            break
        label = graph.edge_labels[e_idx]
        if not label:
            continue

        # Evaluate at midpoint
        mid = evaluate_bezier(curve, 0.5)

        ax.text(
            mid[0], mid[1], label,
            ha="center", va="center",
            fontsize=8,
            fontweight="bold",
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8),
            zorder=4,
        )


def _draw_clusters(ax, graph, pos, sizes):
    """Draw cluster background boxes."""
    from matplotlib.patches import FancyBboxPatch

    if not graph.clusters:
        return

    # Sort clusters by size (largest first = outermost)
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

        # Compute bbox from member positions
        member_pos = pos[indices]
        member_sizes = sizes[indices]

        x_min = (member_pos[:, 0] - member_sizes[:, 0] / 2).min() - padding
        x_max = (member_pos[:, 0] + member_sizes[:, 0] / 2).max() + padding
        y_min = (member_pos[:, 1] - member_sizes[:, 1] / 2).min() - padding
        y_max = (member_pos[:, 1] + member_sizes[:, 1] / 2).max() + padding + 12  # extra for label

        # Opacity decreases with depth
        max_depth = len(sorted_clusters)
        opacity = style.opacity * (1 - depth / (max_depth + 1))

        patch = FancyBboxPatch(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            boxstyle=f"round,pad=0",
            facecolor=style.fill,
            edgecolor=style.stroke,
            linewidth=style.stroke_width,
            alpha=max(opacity, 0.05),
            zorder=0,
        )
        ax.add_patch(patch)

        # Cluster label
        label = graph.cluster_labels.get(name, name)
        ax.text(
            x_min + 4, y_max - 4, label,
            fontsize=style.font_size,
            fontweight=style.font_weight,
            color=style.font_color,
            va="top", ha="left",
            zorder=0.5,
        )
