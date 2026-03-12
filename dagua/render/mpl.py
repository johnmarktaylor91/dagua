"""Matplotlib renderer for DaguaGraph.

Publication-quality rendering following the Dagua Aesthetic Style Guide:
- Wong/Okabe-Ito colorblind-safe palette
- Muted fills, strong borders, quiet edges
- Helvetica/Arial typography
- Warm white background (#FAFAFA)
- Layered rendering: clusters -> edges -> nodes -> labels
"""

from __future__ import annotations

import io
import gzip
from pathlib import Path
from typing import List, Optional, Tuple
import xml.etree.ElementTree as ET

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


_VECTOR_FORMATS = {"pdf", "ps", "eps", "svg", "svgz"}
_RASTER_FORMATS = {"png", "jpg", "jpeg", "webp", "tif", "tiff", "bmp"}


def _detect_output_format(output: Optional[str], format: Optional[str]) -> Optional[str]:
    if format is not None:
        return format.lower().lstrip(".")
    if output is None:
        return None
    suffix = Path(output).suffix.lower().lstrip(".")
    return suffix or "png"


def _save_figure(fig, output: str, bg: str, dpi: int, format: Optional[str] = None) -> None:
    """Save figures with consistent defaults across raster formats."""
    fmt = _detect_output_format(output, format)
    if fmt is None:
        fmt = "png"
    svg_hover_map = getattr(fig, "_dagua_svg_hover_map", None)

    common = {
        "bbox_inches": "tight",
        "pad_inches": 0.05,
        "facecolor": bg,
        "edgecolor": bg,
        "transparent": False,
    }

    if fmt in _VECTOR_FORMATS:
        fig.savefig(output, format=fmt, **common)
        if fmt in {"svg", "svgz"} and svg_hover_map:
            _inject_svg_hover_text(output, svg_hover_map, compressed=(fmt == "svgz"))
        return

    if fmt not in _RASTER_FORMATS:
        raise ValueError(
            f"Unsupported render output format: {fmt!r}. "
            "Supported formats include PNG, JPEG, WebP, TIFF, BMP, SVG, and PDF."
        )

    try:
        from PIL import Image
    except ImportError:
        fig.savefig(output, format=fmt, dpi=dpi, **common)
        return

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, **common)
    buf.seek(0)
    with Image.open(buf) as img:
        if fmt in {"jpg", "jpeg", "bmp"}:
            img = img.convert("RGB")
        save_kwargs = {}
        if fmt in {"jpg", "jpeg"}:
            save_kwargs.update(quality=95, optimize=True, progressive=False, subsampling=0)
        elif fmt == "webp":
            save_kwargs.update(quality=95, method=6)
        elif fmt in {"png", "tif", "tiff"}:
            save_kwargs.update(compress_level=6 if fmt == "png" else None)
        clean_kwargs = {k: v for k, v in save_kwargs.items() if v is not None}
        target_format = {"jpg": "JPEG", "jpeg": "JPEG", "tif": "TIFF"}.get(fmt, fmt.upper())
        img.save(output, format=target_format, **clean_kwargs)


def render(
    graph,
    positions,
    config=None,
    output: Optional[str] = None,
    format: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 150,
    show: bool = False,
    title: Optional[str] = None,
    curves: Optional[List[BezierCurve]] = None,
    label_positions: Optional[List[Optional[Tuple[float, float]]]] = None,
    svg_hover_text: bool = True,
):
    """Render graph with computed positions.

    Args:
        graph: DaguaGraph instance
        positions: [N, 2] tensor of node positions
        config: LayoutConfig (optional)
        output: file path to save
        format: explicit output format override. If None, inferred from output path.
        figsize: figure size in inches
        dpi: resolution for raster output
        show: whether to call plt.show()
        title: optional title for the figure
        curves: pre-computed BezierCurve list (skips re-routing if provided)
        label_positions: pre-computed (x, y) per edge label (from place_edge_labels)

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
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = [RESOLVED_FONT, *[f for f in FONT_FAMILY if f != RESOLVED_FONT]]

    pos = positions.detach().cpu().numpy()
    n = graph.num_nodes
    bg = gs.background_color

    if n == 0:
        fig, ax = plt.subplots(1, 1, figsize=figsize or (6, 4))
        fig.patch.set_facecolor(bg)
        if output:
            _save_figure(fig, output, bg, dpi=dpi, format=format)
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
    setattr(fig, "_dagua_svg_hover_map", {} if svg_hover_text else None)
    svg_hover_map = getattr(fig, "_dagua_svg_hover_map")

    # --- Layer 0: Cluster backgrounds ---
    _draw_clusters(ax, graph, pos, sizes, svg_hover_map=svg_hover_map)

    # --- Layer 1: Edges ---
    if curves is None:
        curves = route_edges(positions, graph.edge_index, graph.node_sizes, graph.direction, graph)
    _draw_edges(ax, graph, curves, svg_hover_map=svg_hover_map)

    # --- Layer 2: Nodes ---
    _draw_nodes(ax, graph, pos, sizes, svg_hover_map=svg_hover_map)

    # --- Layer 3: Node labels ---
    _draw_node_labels(ax, graph, pos, sizes, svg_hover_map=svg_hover_map)

    # --- Layer 4: Edge labels ---
    _draw_edge_labels(ax, graph, curves, label_positions=label_positions, svg_hover_map=svg_hover_map)

    # Configure axes
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.axis("off")

    if title:
        title_ff = gs.title_font_family or RESOLVED_FONT
        ax.set_title(title, fontsize=gs.title_font_size, fontweight=gs.title_font_weight,
                      color=gs.title_font_color, fontfamily=title_ff)

    plt.tight_layout()

    if output:
        _save_figure(fig, output, bg, dpi=dpi, format=format)

    if show:
        plt.show()

    return fig, ax


def _set_svg_hover(artist, gid: str, text: str, svg_hover_map) -> None:
    if svg_hover_map is None:
        return
    artist.set_gid(gid)
    svg_hover_map[gid] = text


def _edge_hover_text(graph, edge_idx: int) -> str:
    src_idx = int(graph.edge_index[0, edge_idx])
    dst_idx = int(graph.edge_index[1, edge_idx])
    src = graph.node_labels[src_idx]
    dst = graph.node_labels[dst_idx]
    label = graph.edge_labels[edge_idx] if edge_idx < len(graph.edge_labels) else None
    return f"{src} -> {dst}: {label}" if label else f"{src} -> {dst}"


def _cluster_hover_text(name: str, graph, indices: List[int]) -> str:
    label = graph.cluster_labels.get(name, name)
    return f"Cluster: {label} ({len(indices)} members)"


def _inject_svg_hover_text(output: str, svg_hover_map, compressed: bool = False) -> None:
    if compressed:
        with gzip.open(output, "rt", encoding="utf-8") as f:
            svg_text = f.read()
    else:
        svg_text = Path(output).read_text(encoding="utf-8")

    root = ET.fromstring(svg_text)
    title_tag = "{http://www.w3.org/2000/svg}title"
    for elem in root.iter():
        gid = elem.attrib.get("id")
        if gid and gid in svg_hover_map:
            title = elem.find(title_tag)
            if title is None:
                title = ET.Element("title")
                elem.insert(0, title)
            title.text = svg_hover_map[gid]

    svg_text = ET.tostring(root, encoding="unicode")
    if compressed:
        with gzip.open(output, "wt", encoding="utf-8") as f:
            f.write(svg_text)
    else:
        Path(output).write_text(svg_text, encoding="utf-8")


def _draw_nodes(ax, graph, pos, sizes, svg_hover_map=None):
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
        _set_svg_hover(patch, f"dagua-node-{i}", graph.node_labels[i], svg_hover_map)


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


def _draw_node_labels(ax, graph, pos, sizes, svg_hover_map=None):
    """Draw centered text labels inside nodes."""
    gs = graph.graph_style

    for i in range(graph.num_nodes):
        x, y = pos[i, 0], pos[i, 1]
        style = graph.get_style_for_node(i)
        label = graph.node_labels[i]

        lines = label.split("\n")
        # Use per-node effective font size when available, fall back to style
        if graph.node_font_sizes is not None and i < graph.node_font_sizes.shape[0]:
            fontsize = graph.node_font_sizes[i].item()
        else:
            fontsize = style.font_size
        font_family = style.font_family_list
        font_weight = style.font_weight
        font_style = style.font_style

        if len(lines) == 1:
            text_artist = ax.text(
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
            _set_svg_hover(text_artist, f"dagua-node-label-{i}", label, svg_hover_map)
        else:
            line_height = fontsize * 1.2
            total_height = line_height * len(lines)
            start_y = y + total_height / 2 - line_height / 2

            for j, line in enumerate(lines):
                ly = start_y - j * line_height
                # Secondary lines slightly smaller
                fs = fontsize if j == 0 else fontsize * gs.node_label_secondary_scale
                text_artist = ax.text(
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
                _set_svg_hover(text_artist, f"dagua-node-label-{i}-{j}", label, svg_hover_map)


def _draw_edges(ax, graph, curves: List[BezierCurve], svg_hover_map=None):
    """Draw bezier edges with arrowheads.

    Edges are quiet: medium gray at 70% opacity, 0.75pt width.
    Arrowheads: small filled triangles (5pt x 3.5pt).
    """
    from matplotlib.path import Path
    from matplotlib.patches import FancyArrowPatch

    for e_idx, curve in enumerate(curves):
        style = graph.get_style_for_edge(e_idx)

        # Extend p1 slightly into the target node so arrowhead visually
        # touches the node border (3px inset along curve direction)
        p1 = curve.p1
        if style.arrow != "none":
            dx = p1[0] - curve.cp2[0]
            dy = p1[1] - curve.cp2[1]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist > 1e-6:
                inset = 3.0
                p1 = (p1[0] + dx / dist * inset, p1[1] + dy / dist * inset)

        verts = [curve.p0, curve.cp1, curve.cp2, p1]
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
        _set_svg_hover(arrow, f"dagua-edge-{e_idx}", _edge_hover_text(graph, e_idx), svg_hover_map)


def _draw_edge_labels(
    ax, graph, curves: List[BezierCurve],
    label_positions: Optional[List[Optional[Tuple[float, float]]]] = None,
    svg_hover_map=None,
):
    """Draw edge labels offset from the curve midpoint.

    Uses per-edge style for font size/color/background, with fallback to graph_style.
    When label_positions is provided, uses pre-computed (x, y) positions.
    """
    gs = graph.graph_style

    for e_idx, curve in enumerate(curves):
        if e_idx >= len(graph.edge_labels):
            break
        label = graph.edge_labels[e_idx]
        if not label:
            continue

        style = graph.get_style_for_edge(e_idx)

        # Use pre-computed position if available
        if label_positions is not None and e_idx < len(label_positions) and label_positions[e_idx] is not None:
            lx, ly = label_positions[e_idx]
        else:
            mid = evaluate_bezier(curve, style.label_position)
            lx, ly = mid[0], mid[1] + 4.0  # default offset

        font_size = style.label_font_size
        font_color = style.label_font_color
        label_bg = style.label_background
        bg_opacity = gs.edge_label_background_opacity

        text_artist = ax.text(
            lx, ly, label,
            ha="center", va="center",
            fontsize=font_size,
            fontweight="regular",
            fontfamily=RESOLVED_FONT,
            color=font_color,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor=label_bg,
                edgecolor="none",
                alpha=bg_opacity,
            ),
            zorder=4,
        )
        _set_svg_hover(text_artist, f"dagua-edge-label-{e_idx}", _edge_hover_text(graph, e_idx), svg_hover_map)


def _draw_clusters(ax, graph, pos, sizes, svg_hover_map=None):
    """Draw cluster background boxes.

    Barely-there fills, thin borders, progressive darkening for nesting.
    Labels: configurable position via label_position.
    """
    from matplotlib.patches import FancyBboxPatch

    if not graph.clusters:
        return

    # Compute true hierarchy depth per cluster via parent chain
    cluster_parents = getattr(graph, 'cluster_parents', {})
    if cluster_parents:
        cluster_depths = {}
        for name in graph.clusters:
            d, cur = 0, name
            while cluster_parents.get(cur):
                cur = cluster_parents[cur]
                d += 1
            cluster_depths[name] = d

        # Sort: shallowest first (deeper clusters render on top)
        sorted_clusters = sorted(
            graph.clusters.items(),
            key=lambda kv: cluster_depths.get(kv[0], 0),
        )
    else:
        # Legacy: sort by member count (largest first)
        sorted_clusters = sorted(
            graph.clusters.items(),
            key=lambda kv: len(collect_cluster_leaves(kv[1]) if isinstance(kv[1], dict) else kv[1]),
            reverse=True,
        )
        cluster_depths = {name: i for i, (name, _) in enumerate(sorted_clusters)}

    for name, members in sorted_clusters:
        depth = cluster_depths.get(name, 0)
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

        # Ensure cluster bbox is wide enough to fit the label text
        label = graph.cluster_labels.get(name, name)
        label_fontsize = max(style.font_size - depth * 1.0, 7.0)
        label_ox = style.label_offset[0]
        # Rough estimate: ~0.55 * font_size per character
        est_label_width = len(label) * label_fontsize * 0.55 + label_ox * 2
        content_width = x_max - x_min
        if est_label_width > content_width:
            expand = (est_label_width - content_width) / 2
            x_min -= expand
            x_max += expand

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
        _set_svg_hover(patch, f"dagua-cluster-{name}", _cluster_hover_text(name, graph, indices), svg_hover_map)

        # Cluster label: position from style (label, label_fontsize already computed above)
        label_ff = style.font_family or RESOLVED_FONT
        label_oy = style.label_offset[1]

        if style.label_position == "top-center":
            lx = (x_min + x_max) / 2
            ha = "center"
        elif style.label_position == "top-right":
            lx = x_max - label_ox
            ha = "right"
        else:  # "top-left" (default)
            lx = x_min + label_ox
            ha = "left"

        # Offset label further down for nested clusters to prevent overlap
        depth_label_offset = depth * label_fontsize * 1.4
        ly = y_max - label_oy - depth_label_offset

        text_artist = ax.text(
            lx, ly, label,
            fontsize=label_fontsize,
            fontweight=style.font_weight,
            fontfamily=label_ff,
            color=style.font_color,
            va="top", ha=ha,
            zorder=0.5,
        )
        _set_svg_hover(text_artist, f"dagua-cluster-label-{name}", _cluster_hover_text(name, graph, indices), svg_hover_map)
