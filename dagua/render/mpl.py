"""Matplotlib renderer for DaguaGraph.

Publication-quality rendering following the Dagua Aesthetic Style Guide:
- Wong/Okabe-Ito colorblind-safe palette
- Muted fills, strong borders, quiet edges
- Helvetica/Arial typography
- Warm white background (#FAFAFA)
- Layered rendering: clusters -> edges -> nodes -> labels
"""

from __future__ import annotations

import gzip
import io
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from dagua.edges import BezierCurve, preferred_edge_label_position, route_edges
from dagua.render.borders import (
    ShapeSpec,
    add_filled_collections,
    annular_path,
    build_shape_path,
    clamp_border_width,
    dash_ribbon_paths,
    inset_shape_path,
    make_clip_proxy,
)
from dagua.render.edges import CubicBezier as RenderBezier
from dagua.render.edges import DaguaEdge, DaguaEdgeCollection
from dagua.render.text import DaguaText, render_text
from dagua.styles import (
    FONT_FAMILY,
    RESOLVED_FONT,
    darken_hex,
)
from dagua.utils import (
    collect_cluster_leaves,
    measure_text,
)

_VECTOR_FORMATS = {"pdf", "ps", "eps", "svg", "svgz"}
_RASTER_FORMATS = {"png", "jpg", "jpeg", "webp", "tif", "tiff", "bmp"}
_GRAPHVIZ_DASH_PATTERN: Tuple[float, float] = (5.0, 3.0)
_GRAPHVIZ_DOT_PATTERN: Tuple[float, float] = (0.1, 3.0)
_ARROWHEAD_REFERENCE_WIDTH_POINTS = 1.2


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
    with Image.open(buf) as opened_img:
        img = opened_img.convert("RGB") if fmt in {"jpg", "jpeg", "bmp"} else opened_img
        save_kwargs: dict[str, Any] = {}
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
    positions=None,
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
    """Render a graph with computed node positions.

    Parameters
    ----------
    graph : Any
        Graph object exposing Dagua's render-facing API.
    positions : Any, optional
        Node positions with shape ``[N, 2]``. When omitted, the graph must have
        a fresh cached layout.
    config : Any, optional
        Unused render-time layout config placeholder kept for API compatibility.
    output : str, optional
        Output file path.
    format : str, optional
        Explicit output format override. When omitted, the renderer infers the
        format from ``output``.
    figsize : tuple[float, float], optional
        Figure size in inches.
    dpi : int, default=150
        Raster output resolution.
    show : bool, default=False
        Whether to call ``plt.show()``.
    title : str, optional
        Figure title.
    curves : list[BezierCurve], optional
        Pre-routed bezier curves for edges.
    label_positions : list[tuple[float, float] | None], optional
        Pre-computed positions for edge labels.
    svg_hover_text : bool, default=True
        Whether to embed hover tooltips in SVG outputs.

    Returns
    -------
    tuple[Any, Any]
        Matplotlib ``(figure, axes)``.
    """
    import warnings

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gs = graph.graph_style

    if positions is None:
        if graph.has_fresh_layout:
            positions = graph.last_positions
        else:
            raise ValueError(
                f"positions=None but graph layout is {graph.layout_status}. "
                "Call dagua.layout(), dagua.draw(), or pass explicit positions."
            )

    # Set global font preferences (use resolved font to avoid warnings)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "findfont")
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = [
            RESOLVED_FONT,
            *[f for f in FONT_FAMILY if f != RESOLVED_FONT],
        ]

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

    # Expand figure bounds for cluster headers and minimum width.
    # Cluster rendering adds header space above y_max and may expand x_min/x_max
    # for minimum width. Account for this so labels are not clipped.
    if graph.clusters:
        for cname in graph.clusters:
            cstyle = graph.get_style_for_cluster(cname)
            cindices = graph.leaf_cluster_members(cname)
            if not cindices:
                continue
            ci = np.array(cindices)
            cp = pos[ci]
            cs = sizes[ci]
            cpad = cstyle.padding
            cy_max_member = (cp[:, 1] + cs[:, 1] / 2).max()
            # Header expansion: at least 14pt + label height
            header = max(14.0, cstyle.font_size * 1.2) * 2.0
            cy_max = cy_max_member + cpad + header
            cx_min = (cp[:, 0] - cs[:, 0] / 2).min() - cpad
            cx_max = (cp[:, 0] + cs[:, 0] / 2).max() + cpad
            # Minimum width
            ch = cy_max - ((cp[:, 1] - cs[:, 1] / 2).min() - cpad)
            min_cw = ch * 0.8
            cw = cx_max - cx_min
            if cw < min_cw:
                expand_cw = (min_cw - cw) / 2.0
                cx_min -= expand_cw
                cx_max += expand_cw
            x_min = min(x_min, cx_min - margin)
            x_max = max(x_max, cx_max + margin)
            y_max = max(y_max, cy_max + margin)

    # Expand figure bounds for self-loop arcs that extend beyond nodes.
    edge_index = graph.edge_index.detach().cpu().numpy()
    direction = getattr(graph, "direction", "TB")
    for e_idx in range(edge_index.shape[1]):
        src, tgt = int(edge_index[0, e_idx]), int(edge_index[1, e_idx])
        if src == tgt:
            sx, sy = float(pos[src, 0]), float(pos[src, 1])
            sw, sh = float(sizes[src, 0]), float(sizes[src, 1])
            loop_size = max(sw, sh)
            loop_w = loop_size * 0.9
            loop_h = loop_size * 2.0
            if direction == "TB":
                y_max = max(y_max, sy + sh / 2 + loop_h + margin)
                x_min = min(x_min, sx - loop_w - margin)
                x_max = max(x_max, sx + loop_w + margin)
            elif direction == "BT":
                y_min = min(y_min, sy - sh / 2 - loop_h - margin)
            elif direction == "LR":
                x_min = min(x_min, sx - sw / 2 - loop_h - margin)
            elif direction == "RL":
                x_max = max(x_max, sx + sw / 2 + loop_h + margin)

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

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.axis("off")

    # --- Layer 0: Cluster backgrounds ---
    _draw_clusters(ax, graph, pos, sizes, svg_hover_map=svg_hover_map)

    # --- Layer 1: Edges ---
    if curves is None:
        curves = route_edges(positions, graph.edge_index, graph.node_sizes, graph.direction, graph)
    edge_collection = _draw_edges(ax, graph, curves, svg_hover_map=svg_hover_map)

    # --- Layer 2: Nodes ---
    clip_patches = _draw_nodes(ax, graph, pos, sizes, svg_hover_map=svg_hover_map)

    # --- Layer 3: Node labels ---
    _draw_node_labels(ax, graph, pos, sizes, clip_patches, svg_hover_map=svg_hover_map)

    # --- Layer 4: Edge labels ---
    _draw_edge_labels(
        ax,
        graph,
        curves,
        label_positions=label_positions,
        svg_hover_map=svg_hover_map,
        edge_collection=edge_collection,
    )

    if title:
        title_ff = gs.title_font_family or RESOLVED_FONT
        ax.set_title(
            title,
            fontsize=gs.title_font_size,
            fontweight=gs.title_font_weight,
            color=gs.title_font_color,
            fontfamily=title_ff,
        )

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


def _node_linestyle(style: Any) -> Any:
    """Resolve the matplotlib linestyle for node borders.

    Parameters
    ----------
    style : Any
        Node style object.

    Returns
    -------
    Any
        Matplotlib linestyle string or dash tuple.
    """
    if style.stroke_dash_pattern is not None:
        return (0, style.stroke_dash_pattern)
    if style.stroke_dash == "dashed":
        return (0, _GRAPHVIZ_DASH_PATTERN)
    if style.stroke_dash == "dotted":
        return (0, _GRAPHVIZ_DOT_PATTERN)
    return "-"


def _edge_linestyle(style: Any) -> Any:
    """Resolve the matplotlib linestyle for an edge body.

    Parameters
    ----------
    style : Any
        Edge style object.

    Returns
    -------
    Any
        Matplotlib linestyle string or dash tuple.
    """
    if style.style == "dashed":
        return (0, _GRAPHVIZ_DASH_PATTERN)
    if style.style == "dotted":
        return (0, _GRAPHVIZ_DOT_PATTERN)
    return "-"


def _triangle_vertices(x: float, y: float, w: float, h: float) -> np.ndarray:
    """Return vertices for a wide triangle matching Graphviz proportions.

    Parameters
    ----------
    x : float
        Node center x-coordinate.
    y : float
        Node center y-coordinate.
    w : float
        Node width.
    h : float
        Node height.

    Returns
    -------
    numpy.ndarray
        Triangle vertices with shape ``[3, 2]``.
    """
    # Graphviz renders triangles wider than tall, so fill the requested node
    # bounds instead of forcing an equilateral silhouette.
    half_width = w / 2.0
    half_height = h / 2.0
    return np.array(
        [
            [x, y + half_height],
            [x + half_width, y - half_height],
            [x - half_width, y - half_height],
        ]
    )


def _cluster_linestyle(stroke_dash: str) -> Any:
    """Resolve the matplotlib linestyle for cluster borders.

    Parameters
    ----------
    stroke_dash : str
        Cluster dash style name.

    Returns
    -------
    Any
        Matplotlib linestyle string or dash tuple.
    """
    if stroke_dash == "dashed":
        return (0, _GRAPHVIZ_DASH_PATTERN)
    if stroke_dash == "dotted":
        return (0, _GRAPHVIZ_DOT_PATTERN)
    return "-"


def _regular_polygon_vertices(
    num_vertices: int,
    x: float,
    y: float,
    w: float,
    h: float,
    rotation: float = np.pi / 2,
) -> np.ndarray:
    """Return vertices for a polygon inscribed in a node bounding box.

    Parameters
    ----------
    num_vertices : int
        Number of polygon corners.
    x : float
        Node center x-coordinate.
    y : float
        Node center y-coordinate.
    w : float
        Node width.
    h : float
        Node height.
    rotation : float, default=np.pi / 2
        Initial rotation in radians.

    Returns
    -------
    numpy.ndarray
        Polygon vertices with shape ``[num_vertices, 2]``.
    """
    angles = rotation + (2.0 * np.pi * np.arange(num_vertices) / num_vertices)
    return np.column_stack((x + (w / 2) * np.cos(angles), y + (h / 2) * np.sin(angles)))


def _star_vertices(x: float, y: float, w: float, h: float) -> np.ndarray:
    """Return vertices for a five-point star.

    Parameters
    ----------
    x : float
        Node center x-coordinate.
    y : float
        Node center y-coordinate.
    w : float
        Node width.
    h : float
        Node height.

    Returns
    -------
    numpy.ndarray
        Star vertices with shape ``[10, 2]``.
    """
    points: List[Tuple[float, float]] = []
    outer_rx = w / 2
    outer_ry = h / 2
    inner_rx = outer_rx * 0.32
    inner_ry = outer_ry * 0.32
    for idx in range(10):
        angle = np.pi / 2 + idx * np.pi / 5
        rx = outer_rx if idx % 2 == 0 else inner_rx
        ry = outer_ry if idx % 2 == 0 else inner_ry
        points.append((x + rx * np.cos(angle), y + ry * np.sin(angle)))
    return np.array(points)


def _build_node_patch(
    x: float,
    y: float,
    w: float,
    h: float,
    style: Any,
    facecolor: Any,
    edgecolor: Any,
    linewidth: float,
    linestyle: Any,
    zorder: float,
) -> Any:
    """Build a matplotlib patch for a node shape.

    Parameters
    ----------
    x : float
        Node center x-coordinate.
    y : float
        Node center y-coordinate.
    w : float
        Node width.
    h : float
        Node height.
    style : Any
        Node style object.
    facecolor : Any
        Matplotlib-compatible face color.
    edgecolor : Any
        Matplotlib-compatible edge color.
    linewidth : float
        Border width.
    linestyle : Any
        Matplotlib linestyle string or dash tuple.
    zorder : float
        Patch z-order.

    Returns
    -------
    Any
        Matplotlib patch instance.
    """
    from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, PathPatch, Polygon
    from matplotlib.path import Path

    shape = style.shape
    if shape in ("roundrect", "rect"):
        corner_radius = style.corner_radius if shape == "roundrect" else 0.0
        if corner_radius > 0:
            boxstyle = f"round,pad=0,rounding_size={corner_radius}"
        else:
            boxstyle = "square,pad=0"
        return FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle=boxstyle,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder,
        )
    if shape == "ellipse":
        return Ellipse(
            (x, y),
            w,
            h,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder,
        )
    if shape == "circle":
        return Circle(
            (x, y),
            max(w, h) / 2,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder,
        )
    if shape == "diamond":
        return Polygon(
            np.array(
                [
                    [x, y + h / 2],
                    [x + w / 2, y],
                    [x, y - h / 2],
                    [x - w / 2, y],
                ]
            ),
            closed=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder,
        )
    if shape == "triangle":
        vertices = _triangle_vertices(x, y, w, h)
        return Polygon(
            vertices,
            closed=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder,
        )
    if shape == "hexagon":
        return Polygon(
            _regular_polygon_vertices(6, x, y, w, h),
            closed=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder,
        )
    if shape == "pentagon":
        return Polygon(
            _regular_polygon_vertices(5, x, y, w, h),
            closed=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder,
        )
    if shape == "octagon":
        return Polygon(
            _regular_polygon_vertices(8, x, y, w, h, rotation=np.pi / 8),
            closed=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder,
        )
    if shape == "star":
        return Polygon(
            _star_vertices(x, y, w, h),
            closed=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            linestyle=linestyle,
            joinstyle="round",
            zorder=zorder,
        )
    if shape == "parallelogram":
        skew = w * 0.28
        return Polygon(
            np.array(
                [
                    [x - w / 2 + skew, y + h / 2],
                    [x + w / 2, y + h / 2],
                    [x + w / 2 - skew, y - h / 2],
                    [x - w / 2, y - h / 2],
                ]
            ),
            closed=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder,
        )
    if shape == "trapezoid":
        inset = w * 0.28
        return Polygon(
            np.array(
                [
                    [x - w / 2 + inset, y + h / 2],  # top-left (narrower)
                    [x + w / 2 - inset, y + h / 2],  # top-right (narrower)
                    [x + w / 2, y - h / 2],  # bottom-right (wider)
                    [x - w / 2, y - h / 2],  # bottom-left (wider)
                ]
            ),
            closed=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder,
        )
    if shape == "cylinder":
        cap_h = max(h * 0.16, 1.0)
        top_cy = y + h / 2 - cap_h
        bottom_cy = y - h / 2 + cap_h
        vertices = [
            (x - w / 2, top_cy),
            (x - w / 2, top_cy + cap_h),
            (x + w / 2, top_cy + cap_h),
            (x + w / 2, top_cy),
            (x + w / 2, bottom_cy),
            (x + w / 2, bottom_cy - cap_h),
            (x - w / 2, bottom_cy - cap_h),
            (x - w / 2, bottom_cy),
            (x - w / 2, top_cy),
        ]
        codes = [
            Path.MOVETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.LINETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.CLOSEPOLY,
        ]
        return PathPatch(
            Path(vertices, codes),
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder,
        )
    return FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0,rounding_size=6",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )


def _draw_node_shape_extras(
    ax: Any,
    x: float,
    y: float,
    w: float,
    h: float,
    style: Any,
    edgecolor: Any,
    zorder: float,
) -> None:
    """Draw shape-specific decorative details after the main node patch.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    x : float
        Node center x-coordinate.
    y : float
        Node center y-coordinate.
    w : float
        Node width.
    h : float
        Node height.
    style : Any
        Node style object.
    edgecolor : Any
        Matplotlib-compatible edge color.
    zorder : float
        Artist z-order.
    """
    if style.shape != "cylinder":
        return

    from matplotlib.patches import Ellipse

    cap_h = max(h * 0.16, 1.0)
    rim = Ellipse(
        (x, y + h / 2 - cap_h),
        w,
        cap_h * 2,
        facecolor="none",
        edgecolor=edgecolor,
        linewidth=style.stroke_width,
        linestyle=_node_linestyle(style),
        zorder=zorder,
    )
    ax.add_patch(rim)


def _draw_gradient_fill(
    ax: Any, patch: Any, x: float, y: float, w: float, h: float, style: Any
) -> None:
    """Draw a gradient image clipped to a node patch.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    patch : Any
        Clip patch for the gradient image.
    x : float
        Node center x-coordinate.
    y : float
        Node center y-coordinate.
    w : float
        Node width.
    h : float
        Node height.
    style : Any
        Node style object.
    """
    from matplotlib.colors import LinearSegmentedColormap

    resolution = 128
    grid = np.linspace(-1.0, 1.0, resolution)
    xx, yy = np.meshgrid(grid, grid)

    if style.gradient == "radial":
        data = np.clip(np.sqrt(xx**2 + yy**2), 0.0, 1.0)
    else:
        angle = np.deg2rad(style.gradient_angle)
        projection = xx * np.cos(angle) + yy * np.sin(angle)
        data = np.clip((projection + 1.0) / 2.0, 0.0, 1.0)

    gradient_color = style.gradient_color or style.stroke or darken_hex(style.fill, 0.12)
    cmap = LinearSegmentedColormap.from_list("dagua-node-gradient", [style.fill, gradient_color])
    image = ax.imshow(
        data,
        extent=(x - w / 2, x + w / 2, y - h / 2, y + h / 2),
        origin="lower",
        cmap=cmap,
        interpolation="bicubic",
        alpha=style.opacity,
        zorder=1.95,
        aspect="auto",
    )
    image.set_clip_path(patch)


def _scaled_node_style(style: Any, display_scale: float) -> Any:
    """Convert node geometry-style fields from points into data units.

    Parameters
    ----------
    style : Any
        Node style object.
    display_scale : float
        Point-to-data conversion factor.

    Returns
    -------
    Any
        Style copy whose data-geometry fields are converted for the current
        axes scale.
    """

    return replace(
        style,
        corner_radius=float(style.corner_radius) * display_scale,
        shadow_offset=(
            float(style.shadow_offset[0]) * display_scale,
            float(style.shadow_offset[1]) * display_scale,
        ),
    )


def _node_border_pattern(style: Any, display_scale: float) -> Any:
    """Resolve a node border dash pattern in data units.

    Parameters
    ----------
    style : Any
        Node style object.
    display_scale : float
        Point-to-data conversion factor.

    Returns
    -------
    Any
        Either a built-in dash name or a custom dash tuple in data units.
    """

    if style.stroke_dash_pattern is None:
        return style.stroke_dash
    return tuple(float(value) * display_scale for value in style.stroke_dash_pattern)


def _cluster_render_order(graph: Any) -> List[str]:
    """Return clusters in parent-first depth-first traversal order.

    Parameters
    ----------
    graph : Any
        Graph exposing ``clusters`` and optional ``cluster_parents``.

    Returns
    -------
    list[str]
        Cluster names in render order.
    """

    if not getattr(graph, "cluster_parents", {}):
        return list(graph.clusters.keys())

    children: Dict[Optional[str], List[str]] = {}
    for name in graph.clusters:
        parent = graph.cluster_parents.get(name)
        children.setdefault(parent, []).append(name)

    ordered: List[str] = []

    def visit(name: str) -> None:
        """Visit one cluster and its descendants."""

        ordered.append(name)
        for child in children.get(name, []):
            visit(child)

    for root in children.get(None, []):
        visit(root)

    if len(ordered) != len(graph.clusters):
        for name in graph.clusters:
            if name not in ordered:
                visit(name)
    return ordered


def _cluster_depths(graph: Any, ordered_clusters: Sequence[str]) -> Dict[str, int]:
    """Compute cluster nesting depth from parent links or fallback order.

    Parameters
    ----------
    graph : Any
        Graph exposing ``cluster_parents``.
    ordered_clusters : sequence[str]
        Flattened render order.

    Returns
    -------
    dict[str, int]
        Cluster depth per cluster name.
    """

    cluster_parents = getattr(graph, "cluster_parents", {})
    if not cluster_parents:
        return {name: index for index, name in enumerate(ordered_clusters)}

    depths: Dict[str, int] = {}
    for name in ordered_clusters:
        depth = 0
        current = name
        while cluster_parents.get(current) is not None:
            current = cluster_parents[current]
            depth += 1
        depths[name] = depth
    return depths


def _draw_nodes(
    ax: Any,
    graph: Any,
    pos: np.ndarray,
    sizes: np.ndarray,
    svg_hover_map: Optional[Dict[str, str]] = None,
) -> List[Any]:
    """Draw node shapes and return clip patches for labels.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    graph : Any
        Graph exposing Dagua's node-style API.
    pos : numpy.ndarray
        Node positions with shape ``[N, 2]``.
    sizes : numpy.ndarray
        Node sizes with shape ``[N, 2]``.
    svg_hover_map : dict[str, str], optional
        SVG hover text accumulator.

    Returns
    -------
    list[Any]
        Primary node patches used to clip node labels.
    """
    from matplotlib.colors import to_rgba

    display_scale = _compute_display_scale(ax)
    clip_patches: List[Any] = []
    fill_paths: List[Any] = []
    fill_colors: List[Any] = []
    border_paths: List[Any] = []
    border_colors: List[Any] = []

    for i in range(graph.num_nodes):
        x, y = float(pos[i, 0]), float(pos[i, 1])
        w, h = float(sizes[i, 0]), float(sizes[i, 1])
        style = graph.get_style_for_node(i)
        scaled_style = _scaled_node_style(style, display_scale)
        border_width = clamp_border_width(float(style.stroke_width) * display_scale, w, h)
        shape_spec = ShapeSpec(
            center_x=x,
            center_y=y,
            width=w,
            height=h,
            shape=str(style.shape),
            corner_radius=float(scaled_style.corner_radius),
        )
        outer_path = build_shape_path(shape_spec)
        fill_path = inset_shape_path(shape_spec, border_width) if border_width > 0.0 else outer_path

        if style.shadow:
            _draw_shadow(ax, x, y, w, h, scaled_style)

        facecolor = to_rgba(style.fill, style.opacity)
        edgecolor = to_rgba(style.stroke, style.opacity * style.border_opacity)
        clip_patch = make_clip_proxy(fill_path, ax.transData)
        if style.gradient == "none":
            fill_paths.append(fill_path)
            fill_colors.append(facecolor)
        elif style.opacity > 0.0:
            _draw_gradient_fill(ax, clip_patch, x, y, w, h, style)

        if border_width > 0.0 and edgecolor[-1] > 0.0:
            if style.stroke_dash == "solid" and style.stroke_dash_pattern is None:
                border_paths.append(annular_path(outer_path, fill_path))
                border_colors.append(edgecolor)
            else:
                centerline_path = inset_shape_path(shape_spec, border_width / 2.0)
                dash_pattern = _node_border_pattern(style, display_scale)
                ribbons = dash_ribbon_paths(centerline_path, dash_pattern, border_width)
                border_paths.extend(ribbons)
                border_colors.extend([edgecolor] * len(ribbons))

        clip_patches.append(clip_patch)
        if style.gradient != "none":
            _set_svg_hover(clip_patch, f"dagua-node-{i}", graph.node_labels[i], svg_hover_map)
        _draw_node_shape_extras(ax, x, y, w, h, style, edgecolor, zorder=2.08)

    add_filled_collections(
        ax=ax,
        fill_paths=fill_paths,
        fill_colors=fill_colors,
        border_paths=border_paths,
        border_colors=border_colors,
        fill_zorder=2.0,
        border_zorder=2.05,
    )
    return clip_patches


def _draw_shadow(ax: Any, x: float, y: float, w: float, h: float, style: Any) -> None:
    """Draw a node shadow, approximating blur with layered fills.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    x : float
        Node center x-coordinate.
    y : float
        Node center y-coordinate.
    w : float
        Node width.
    h : float
        Node height.
    style : Any
        Node style object.
    """
    from matplotlib.colors import to_rgba

    ox, oy = style.shadow_offset
    base_r, base_g, base_b, base_a = to_rgba(style.shadow_color)
    steps = 1 if style.shadow_blur <= 0 else min(max(int(np.ceil(style.shadow_blur)), 2), 6)
    for idx in range(steps, 0, -1):
        scale = 1.0 + (0.01 * style.shadow_blur * idx)
        alpha = base_a / (idx + 1) if steps > 1 else base_a
        shadow = _build_node_patch(
            x + ox,
            y + oy,
            w * scale,
            h * scale,
            style,
            (base_r, base_g, base_b, alpha),
            "none",
            0.0,
            "-",
            zorder=1.4 - idx * 0.01,
        )
        ax.add_patch(shadow)


def _points_to_data_units(ax: Any, points: float, axis: str) -> float:
    """Convert typographic points to data units along one axis.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    points : float
        Distance in points.
    axis : str
        Axis name, either ``"x"`` or ``"y"``.

    Returns
    -------
    float
        Distance in data units.
    """
    pixels = points * ax.figure.dpi / 72.0
    transformed = ax.transData.transform([(0.0, 0.0), (1.0, 1.0)])
    if axis == "x":
        scale = abs(transformed[1][0] - transformed[0][0])
    else:
        scale = abs(transformed[1][1] - transformed[0][1])
    if scale <= 1e-9:
        return 0.0
    return pixels / scale


def _compute_display_scale(ax: Any) -> float:
    """Compute the point-to-data conversion factor for display-sized geometry.

    Parameters
    ----------
    ax : Any
        Matplotlib axes with established data limits.

    Returns
    -------
    float
        Multiplicative factor such that ``data_units = points * scale``.

    Notes
    -----
    Use this only for geometry constructed in data coordinates whose intended
    visual size is specified in points, such as node and cluster border bodies,
    node corner radii, shadow offsets, arrowhead polygons, and cluster label
    offsets. Matplotlib already interprets ``linewidth`` and ``fontsize`` in
    points natively.
    """
    scale_x = _points_to_data_units(ax, 1.0, "x")
    scale_y = _points_to_data_units(ax, 1.0, "y")
    scale = min(scale_x, scale_y)
    return scale if scale > 1e-9 else 1.0


def _marker_data_size(
    ax: Any,
    style: Any,
    length: float,
    width: float,
    node_height: float = 0.0,
) -> Tuple[float, float]:
    """Convert marker dimensions to data units.

    Parameters
    ----------
    ax : Any
        Matplotlib axes receiving the marker patch.
    style : Any
        Edge style object. ``arrow_scale`` is intentionally ignored.
    length : float
        Marker length in typographic points.
    width : float
        Marker width in typographic points.
    node_height : float, default=0.0
        Target node height in data units. When ``style.arrow_node_fraction`` is
        positive, marker sizing becomes proportional to this height.

    Returns
    -------
    tuple[float, float]
        Marker ``(length, width)`` in data units.

    Notes
    -----
    When ``arrow_node_fraction > 0`` on the style, arrowheads scale with the
    connected node height. Otherwise the renderer falls back to converting fixed
    point dimensions into data units so marker size stays stable in display space.
    """
    fraction = float(getattr(style, "arrow_node_fraction", 0.0))
    if fraction > 0.0 and node_height > 0.0:
        width_ratio = float(getattr(style, "arrow_width_ratio", 0.7))
        data_length = node_height * fraction
        data_width = data_length * width_ratio
        return data_length, data_width

    scale = _compute_display_scale(ax)
    return length * scale, width * scale


def _scaled_arrowhead_dimensions(
    length_points: float,
    width_points: float,
    edge_width_points: float,
    reference_width_points: float = _ARROWHEAD_REFERENCE_WIDTH_POINTS,
) -> Tuple[float, float]:
    """Scale arrowhead dimensions sublinearly with the edge stroke weight.

    Parameters
    ----------
    length_points : float
        Base arrowhead length in typographic points.
    width_points : float
        Base arrowhead width in typographic points.
    edge_width_points : float
        Edge stroke width in typographic points.
    reference_width_points : float, default=1.2
        Width treated as the unscaled baseline.

    Returns
    -------
    tuple[float, float]
        Scaled ``(length, width)`` in points.
    """

    safe_reference = max(reference_width_points, 1e-6)
    width_ratio = max(edge_width_points, 0.0) / safe_reference
    scale = max(width_ratio, 1e-6) ** 0.5
    return length_points * scale, width_points * scale


def _edge_width_data_units(ax: Any, width_points: float) -> float:
    """Convert an edge body width from points to data units.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    width_points : float
        Edge width in typographic points.

    Returns
    -------
    float
        Width in data units.
    """
    width_x = _points_to_data_units(ax, width_points, "x")
    width_y = _points_to_data_units(ax, width_points, "y")
    width = min(width_x, width_y)
    return width if width > 1e-6 else 1e-6


def _curve_to_render_bezier(curve: BezierCurve) -> RenderBezier:
    """Convert a routed bezier curve to the custom-renderer type.

    Parameters
    ----------
    curve : BezierCurve
        Routed edge curve.

    Returns
    -------
    dagua.render.edges.geometry.CubicBezier
        Equivalent render-space curve.
    """
    return RenderBezier.from_points(curve.p0, curve.cp1, curve.cp2, curve.p1)


def _build_custom_edge_collection(
    ax: Any,
    graph: Any,
    curves: List[BezierCurve],
) -> DaguaEdgeCollection:
    """Translate graph edge styles into the custom edge collection.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    graph : Any
        Graph exposing Dagua's edge-style API.
    curves : list[BezierCurve]
        Routed edge curves.

    Returns
    -------
    DaguaEdgeCollection
        Prepared custom edge collection.
    """
    edges: List[DaguaEdge] = []
    for e_idx, curve in enumerate(curves):
        style = graph.get_style_for_edge(e_idx)
        src_idx = int(graph.edge_index[0, e_idx])
        tgt_idx = int(graph.edge_index[1, e_idx])
        src_node_height = float(graph.node_sizes[src_idx, 1])
        tgt_node_height = float(graph.node_sizes[tgt_idx, 1])
        scaled_head_length, scaled_head_width = _scaled_arrowhead_dimensions(
            float(style.arrow_length),
            float(style.arrow_width),
            float(style.width),
        )
        head_length, head_width = _marker_data_size(
            ax,
            style,
            scaled_head_length,
            scaled_head_width,
            node_height=tgt_node_height,
        )
        tail_length, tail_width = _marker_data_size(
            ax,
            style,
            scaled_head_length,
            scaled_head_width,
            node_height=src_node_height,
        )
        label = graph.edge_labels[e_idx] if e_idx < len(graph.edge_labels) else None
        edges.append(
            DaguaEdge(
                curve=_curve_to_render_bezier(curve),
                width=_edge_width_data_units(ax, float(style.width)),
                color=str(style.color or "#8C8C8C"),
                alpha=float(style.opacity if style.opacity is not None else 0.7),
                linestyle=style.style,
                arrowhead=str(style.arrow),
                tail_arrow=str(style.tail_arrow),
                arrowhead_length=head_length,
                arrowhead_width=head_width,
                tail_arrow_length=tail_length,
                tail_arrow_width=tail_width,
                arrow_fill=str(style.arrow_fill),
                arrow_color=str(style.arrow_color) if style.arrow_color else None,
                stroke_width=_edge_width_data_units(ax, float(style.width)),
                label=label,
                label_position=float(style.label_position),
                label_offset=float(style.label_offset),
                label_rotate=False,
                label_side=str(style.label_side),
                label_font_size=float(style.label_font_size),
                label_font_color=str(style.label_font_color),
                label_background=str(style.label_background),
                label_font_family=str(style.label_font_family),
                label_font_weight=str(style.label_font_weight),
                group_key=(src_idx, tgt_idx),
                source_node=src_idx,
                target_node=tgt_idx,
            )
        )
    return DaguaEdgeCollection(edges)


def _label_reference_y(y: float, h: float, shape: str) -> float:
    """Adjust the vertical label anchor for shapes with off-center centroids.

    Parameters
    ----------
    y : float
        Node center y-coordinate.
    h : float
        Node height.
    shape : str
        Node shape name.

    Returns
    -------
    float
        Shape-adjusted y-coordinate used as the label anchor reference.
    """
    if shape == "triangle":
        # Upright triangles read visually lower than their bounding-box center,
        # so shift labels toward the centroid to match Graphviz.
        return y - h / 6
    return y


def _draw_node_labels(
    ax: Any,
    graph: Any,
    pos: np.ndarray,
    sizes: np.ndarray,
    clip_patches: Optional[Sequence[Any]] = None,
    svg_hover_map: Optional[Dict[str, str]] = None,
) -> None:
    """Draw node labels with alignment, rich-text, and outline support.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    graph : Any
        Graph exposing Dagua's node-label API.
    pos : numpy.ndarray
        Node positions with shape ``[N, 2]``.
    sizes : numpy.ndarray
        Node sizes with shape ``[N, 2]``.
    clip_patches : sequence[Any], optional
        Clip patches returned by :func:`_draw_nodes`. When omitted, labels are
        rendered without per-node clipping for backward compatibility with
        internal utility callers.
    svg_hover_map : dict[str, str], optional
        SVG hover text accumulator.
    """
    gs = graph.graph_style
    display_scale = _compute_display_scale(ax)
    clip_patch_seq: Sequence[Any] = clip_patches or []
    specs: List[DaguaText] = []

    for i in range(graph.num_nodes):
        label = graph.node_labels[i]
        if not label:
            continue

        x, y = float(pos[i, 0]), float(pos[i, 1])
        w, h = float(sizes[i, 0]), float(sizes[i, 1])
        style = graph.get_style_for_node(i)
        clip_patch = clip_patch_seq[i] if i < len(clip_patch_seq) else None
        label_y = _label_reference_y(y, h, style.shape)

        if graph.node_font_sizes is not None and i < graph.node_font_sizes.shape[0]:
            fontsize = float(graph.node_font_sizes[i].item())
        else:
            fontsize = float(style.font_size)

        pad_x = float(style.padding[0]) * display_scale
        pad_y = float(style.padding[1]) * display_scale
        max_width: Optional[float] = None
        if style.overflow_policy == "shrink_text":
            max_width = w - 2.0 * pad_x

        if style.text_align == "left":
            text_x = x - w / 2.0 + pad_x
        elif style.text_align == "right":
            text_x = x + w / 2 - pad_x
        else:
            text_x = x

        if style.text_valign == "top":
            text_y = label_y + h / 2.0 - pad_y
        elif style.text_valign == "bottom":
            text_y = label_y - h / 2.0 + pad_y
        else:
            text_y = label_y

        is_rich = style.label_format == "rich"
        secondary = gs.node_label_secondary_scale if not is_rich else 1.0
        specs.append(
            DaguaText(
                x=text_x,
                y=text_y,
                text=label,
                font_size=fontsize,
                font_family=style.font_family_list[0],
                font_weight=style.font_weight,
                font_style=style.font_style,
                font_color=style.font_color,
                alpha=1.0,
                ha=style.text_align,
                va=style.text_valign,
                rich=is_rich,
                line_spacing=1.2,
                secondary_scale=secondary,
                max_width=max_width,
                min_font_size=style.min_font_size,
                outline=style.text_outline,
                outline_color=style.text_outline_color,
                outline_width=style.text_outline_width,
                clip_patch=clip_patch,
                clip_on=True,
                zorder=3.0,
                gid=f"dagua-node-label-{i}",
            )
        )

    render_text(ax, specs, display_scale, svg_hover_map)


def _draw_edge_marker(
    ax: Any,
    point: Tuple[float, float],
    direction: Tuple[float, float],
    marker: str,
    style: Any,
    node_height: float = 0.0,
) -> None:
    """Draw a custom edge endpoint marker.

    Parameters
    ----------
    ax : Any
        Matplotlib axes receiving the marker patch.
    point : tuple[float, float]
        Marker tip position in data coordinates.
    direction : tuple[float, float]
        Direction vector pointing out of the endpoint.
    marker : str
        Marker name such as ``"normal"`` or ``"diamond"``.
    style : Any
        Edge style object providing arrow geometry and color settings.
    node_height : float, default=0.0
        Height of the connected node in data units. Used only when the style
        enables node-relative arrow sizing.

    Returns
    -------
    None
        Mutates ``ax`` in place by adding the marker artist when applicable.
    """
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Polygon

    dx, dy = direction
    dist = float(np.hypot(dx, dy))
    if dist <= 1e-9 or marker == "none":
        return

    ux, uy = dx / dist, dy / dist
    px, py = -uy, ux
    length = float(style.arrow_length)
    width = float(style.arrow_width)
    manual_length, manual_width = _marker_data_size(
        ax,
        style,
        length,
        width,
        node_height=node_height,
    )
    # Graphviz-style calibration expects arrowheads to read slightly heavier
    # than the edge stroke, so keep marker fill/outline fully opaque.
    color = to_rgba(style.arrow_color or style.color, 1.0)
    filled = style.arrow_fill == "filled" and marker not in {"open", "vee", "tee", "crow"}
    tip_x, tip_y = point

    if marker == "normal":
        # Filled triangle with tip at the edge endpoint (node boundary)
        # and body extending into the gap between nodes.
        base_x = tip_x - ux * manual_length
        base_y = tip_y - uy * manual_length
        polygon = Polygon(
            [
                (tip_x, tip_y),
                (base_x + px * manual_width * 0.6, base_y + py * manual_width * 0.6),
                (base_x - px * manual_width * 0.6, base_y - py * manual_width * 0.6),
            ],
            closed=True,
            facecolor=color if filled else "none",
            edgecolor=color,
            linewidth=style.width,
            joinstyle="round",
            zorder=3,
        )
        ax.add_patch(polygon)
        return

    if marker == "vee":
        base_x = tip_x - ux * manual_length
        base_y = tip_y - uy * manual_length
        polygon = Polygon(
            [
                (base_x + px * manual_width * 0.7, base_y + py * manual_width * 0.7),
                (tip_x, tip_y),
                (base_x - px * manual_width * 0.7, base_y - py * manual_width * 0.7),
            ],
            closed=False,
            facecolor="none",
            edgecolor=color,
            linewidth=max(style.width * 1.8, 2.0),
            joinstyle="round",
            capstyle="round",
            zorder=3,
        )
        ax.add_patch(polygon)
        return

    if marker == "open":
        base_x = tip_x - ux * manual_length
        base_y = tip_y - uy * manual_length
        polygon = Polygon(
            [
                (tip_x, tip_y),
                (base_x + px * manual_width * 0.6, base_y + py * manual_width * 0.6),
                (base_x - px * manual_width * 0.6, base_y - py * manual_width * 0.6),
            ],
            closed=True,
            facecolor="none",
            edgecolor=color,
            linewidth=style.width,
            joinstyle="round",
            zorder=3,
        )
        ax.add_patch(polygon)
        return

    if marker in {"dot", "circle"}:
        radius = manual_width * (0.55 if marker == "dot" else 0.85)
        center_x = tip_x - ux * radius
        center_y = tip_y - uy * radius
        # Graphviz uses a filled dot marker but a hollow circle marker.
        is_filled = marker == "dot" and filled
        circle_patch = Circle(
            (center_x, center_y),
            radius,
            facecolor=color if is_filled else "none",
            edgecolor=color,
            linewidth=style.width,
            zorder=3,
        )
        ax.add_patch(circle_patch)
        return

    if marker == "diamond":
        mid_x = tip_x - ux * (manual_length / 2)
        mid_y = tip_y - uy * (manual_length / 2)
        back_x = tip_x - ux * manual_length
        back_y = tip_y - uy * manual_length
        diamond = Polygon(
            [
                (tip_x, tip_y),
                (mid_x + px * manual_width / 2, mid_y + py * manual_width / 2),
                (back_x, back_y),
                (mid_x - px * manual_width / 2, mid_y - py * manual_width / 2),
            ],
            closed=True,
            facecolor=color if filled else "none",
            edgecolor=color,
            linewidth=style.width,
            joinstyle="round",
            zorder=3,
        )
        ax.add_patch(diamond)
        return

    if marker == "tee":
        # Use a thin rectangle instead of a thick line so the tee reads as a
        # wide, flat bar instead of a square cap at small render sizes.
        bar_x = tip_x - ux * (manual_length / 4)
        bar_y = tip_y - uy * (manual_length / 4)
        bar_half_span = manual_width * 1.3
        bar_half_thick = manual_length / 6
        polygon = Polygon(
            [
                (
                    bar_x + px * bar_half_span + ux * bar_half_thick,
                    bar_y + py * bar_half_span + uy * bar_half_thick,
                ),
                (
                    bar_x + px * bar_half_span - ux * bar_half_thick,
                    bar_y + py * bar_half_span - uy * bar_half_thick,
                ),
                (
                    bar_x - px * bar_half_span - ux * bar_half_thick,
                    bar_y - py * bar_half_span - uy * bar_half_thick,
                ),
                (
                    bar_x - px * bar_half_span + ux * bar_half_thick,
                    bar_y - py * bar_half_span + uy * bar_half_thick,
                ),
            ],
            closed=True,
            facecolor=color,
            edgecolor=color,
            linewidth=0.5,
            zorder=3,
        )
        ax.add_patch(polygon)
        return

    if marker == "crow":
        back_x = tip_x - ux * manual_length
        back_y = tip_y - uy * manual_length
        for end_x, end_y in (
            (back_x, back_y),
            (back_x + px * manual_width * 0.85, back_y + py * manual_width * 0.85),
            (back_x - px * manual_width * 0.85, back_y - py * manual_width * 0.85),
        ):
            ax.add_line(
                Line2D(
                    [tip_x, end_x],
                    [tip_y, end_y],
                    color=color,
                    linewidth=max(style.width * 1.8, 2.0),
                    zorder=3,
                )
            )


def _draw_edges(
    ax: Any,
    graph: Any,
    curves: List[BezierCurve],
    svg_hover_map: Optional[Dict[str, str]] = None,
) -> Optional[DaguaEdgeCollection]:
    """Draw edge bodies and arrowheads with the custom batched renderer.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    graph : Any
        Graph exposing Dagua's edge-style API.
    curves : list[BezierCurve]
        Routed edge curves.
    svg_hover_map : dict[str, str], optional
        SVG hover text accumulator.
    Returns
    -------
    dagua.render.edges.collection.DaguaEdgeCollection | None
        The prepared custom collection so the label pass can reuse it.
    """
    if not curves:
        return None
    collection = _build_custom_edge_collection(ax, graph, curves)
    collection.render_bodies(ax)
    collection.render_heads(ax)
    return collection


def _draw_edge_labels(
    ax: Any,
    graph: Any,
    curves: List[BezierCurve],
    label_positions: Optional[List[Optional[Tuple[float, float]]]] = None,
    svg_hover_map: Optional[Dict[str, str]] = None,
    edge_collection: Optional[DaguaEdgeCollection] = None,
) -> None:
    """Draw edge labels using per-edge font settings.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    graph : Any
        Graph exposing Dagua's edge-label API.
    curves : list[BezierCurve]
        Routed edge curves.
    label_positions : list[tuple[float, float] | None], optional
        Pre-computed label positions.
    svg_hover_map : dict[str, str], optional
        SVG hover text accumulator.
    edge_collection : DaguaEdgeCollection | None, optional
        Prepared collection whose label geometry should be reused.
    """
    gs = graph.graph_style
    display_scale = _compute_display_scale(ax)

    if edge_collection is not None and label_positions is None:
        if svg_hover_map is not None:
            for e_idx, prepared in enumerate(edge_collection.prepared_edges):
                if not prepared.edge.label:
                    continue
                hover_text = _edge_hover_text(graph, e_idx)
                svg_hover_map[f"dagua-edge-label-{e_idx}"] = hover_text
                svg_hover_map[f"dagua-edge-label-{e_idx}-background"] = hover_text
        edge_collection.render_labels(
            ax,
            display_scale=display_scale,
            label_background_alpha=gs.edge_label_background_opacity,
            svg_hover_map=svg_hover_map,
        )
        return

    specs: List[DaguaText] = []
    for e_idx, curve in enumerate(curves):
        if e_idx >= len(graph.edge_labels):
            break
        label = graph.edge_labels[e_idx]
        if not label:
            continue

        style = graph.get_style_for_edge(e_idx)
        if (
            label_positions is not None
            and e_idx < len(label_positions)
            and label_positions[e_idx] is not None
        ):
            label_pos = label_positions[e_idx]
            assert label_pos is not None
            lx, ly = label_pos
        else:
            lx, ly = preferred_edge_label_position(
                curve,
                label_position=style.label_position,
                label_offset=style.label_offset,
                label_side=style.label_side,
            )

        specs.append(
            DaguaText(
                x=lx,
                y=ly,
                text=label,
                font_size=float(style.label_font_size),
                font_family=str(style.label_font_family or RESOLVED_FONT),
                font_weight=str(style.label_font_weight),
                font_color=str(style.label_font_color),
                ha="center",
                va="center",
                background=str(style.label_background),
                background_alpha=float(gs.edge_label_background_opacity),
                background_padding=(
                    float(style.label_font_size) * 0.15,
                    float(style.label_font_size) * 0.15,
                ),
                background_corner_radius=float(style.label_font_size) * 0.15,
                clip_on=False,
                zorder=4.0,
                gid=f"dagua-edge-label-{e_idx}",
            )
        )
        if svg_hover_map is not None:
            svg_hover_map[f"dagua-edge-label-{e_idx}"] = _edge_hover_text(graph, e_idx)
            svg_hover_map[f"dagua-edge-label-{e_idx}-background"] = _edge_hover_text(graph, e_idx)

    render_text(ax, specs, display_scale, svg_hover_map)


def _draw_clusters(
    ax: Any,
    graph: Any,
    pos: np.ndarray,
    sizes: np.ndarray,
    svg_hover_map: Optional[Dict[str, str]] = None,
) -> None:
    """Draw cluster background boxes and labels.

    Parameters
    ----------
    ax : Any
        Matplotlib axes receiving cluster artists.
    graph : Any
        Graph object exposing cluster membership, labels, and styles.
    pos : numpy.ndarray
        Node positions with shape ``[N, 2]`` in render coordinates.
    sizes : numpy.ndarray
        Node box sizes with shape ``[N, 2]`` in points.
    svg_hover_map : dict[str, str] | None, default=None
        Optional SVG hover-text map populated with cluster metadata.

    Returns
    -------
    None
        Mutates ``ax`` in place by adding cluster patches and labels.
    """
    from matplotlib.colors import to_rgba

    if not graph.clusters:
        return

    ordered_clusters = _cluster_render_order(graph)
    cluster_depths = _cluster_depths(graph, ordered_clusters)
    max_depth = max(cluster_depths.values(), default=0)
    display_scale = _compute_display_scale(ax)

    fill_paths_by_depth: Dict[int, List[Any]] = {}
    fill_colors_by_depth: Dict[int, List[Any]] = {}
    border_paths_by_depth: Dict[int, List[Any]] = {}
    border_colors_by_depth: Dict[int, List[Any]] = {}
    border_outline_patches_by_depth: Dict[int, List[Any]] = {}
    cluster_label_specs: List[DaguaText] = []

    from matplotlib.patches import PathPatch

    for name in ordered_clusters:
        members = graph.clusters[name]
        depth = cluster_depths.get(name, 0)
        indices = collect_cluster_leaves(members) if isinstance(members, dict) else members

        if not indices:
            continue

        style = graph.get_style_for_cluster(name)
        padding = style.padding

        member_pos = pos[indices]
        member_sizes = sizes[indices]

        x_min = (member_pos[:, 0] - member_sizes[:, 0] / 2).min() - padding
        x_max = (member_pos[:, 0] + member_sizes[:, 0] / 2).max() + padding
        label = graph.cluster_labels.get(name, name)
        label_fontsize = max(style.font_size - depth * 1.0, 7.0)
        label_ff = style.font_family or RESOLVED_FONT
        label_ox = style.label_offset[0] * display_scale
        label_oy = style.label_offset[1] * display_scale
        label_width_pt, label_height_pt = measure_text(
            label,
            font_family=label_ff,
            font_size=label_fontsize,
            font_weight=style.font_weight,
        )
        label_width = _points_to_data_units(ax, label_width_pt, "x")
        label_height = _points_to_data_units(ax, label_height_pt, "y")

        y_min = (member_pos[:, 1] - member_sizes[:, 1] / 2).min() - padding
        y_max = (
            (member_pos[:, 1] + member_sizes[:, 1] / 2).max()
            + padding
            + max(
                _points_to_data_units(ax, 14.0, "y"),
                label_height,
            )
        )

        # Enforce a modest minimum cluster width so tall vertical stacks do not
        # collapse into needle-thin boxes, while still allowing nested
        # clusters to stay closer to the matplotlib reference proportions.
        cluster_height = y_max - y_min
        cluster_width = x_max - x_min
        min_cluster_width = cluster_height * 0.65
        if cluster_width < min_cluster_width:
            expand_w = (min_cluster_width - cluster_width) / 2.0
            x_min -= expand_w
            x_max += expand_w

        # Cluster labels are few and measure_text is cached, so use the actual
        # measured width instead of a character-count heuristic.
        est_label_width = label_width + label_ox * 2
        content_width = x_max - x_min
        if est_label_width > content_width:
            expand = (est_label_width - content_width) / 2
            x_min -= expand
            x_max += expand

        # Progressive depth darkening using HSL (replaces LEVEL_FILLS/LEVEL_STROKES)
        fill_color = darken_hex(style.fill, depth * style.depth_fill_step)
        stroke_color = darken_hex(style.stroke, depth * style.depth_stroke_step)

        fill_alpha = style.opacity * (1.0 - depth * 0.15 / max(max_depth, 1))
        fill_alpha = max(fill_alpha, 0.08)
        border_alpha = min(
            max(style.opacity * 2.5, 0.6) * (1.0 - depth * 0.15 / max(max_depth, 1)), 1.0
        )

        width = x_max - x_min
        height = y_max - y_min
        border_width = clamp_border_width(float(style.stroke_width) * display_scale, width, height)
        shape_spec = ShapeSpec(
            center_x=(x_min + x_max) / 2.0,
            center_y=(y_min + y_max) / 2.0,
            width=width,
            height=height,
            shape="roundrect",
            corner_radius=float(style.corner_radius) * display_scale,
        )
        outer_path = build_shape_path(shape_spec)
        fill_path = inset_shape_path(shape_spec, border_width) if border_width > 0.0 else outer_path

        fill_paths_by_depth.setdefault(depth, []).append(fill_path)
        fill_colors_by_depth.setdefault(depth, []).append(to_rgba(fill_color, fill_alpha))
        if border_width > 0.0:
            if style.stroke_dash == "solid":
                border_paths = [annular_path(outer_path, fill_path)]
            else:
                centerline_path = inset_shape_path(shape_spec, border_width / 2.0)
                border_paths = dash_ribbon_paths(centerline_path, style.stroke_dash, border_width)
            border_paths_by_depth.setdefault(depth, []).extend(border_paths)
            border_colors_by_depth.setdefault(depth, []).extend(
                [to_rgba(stroke_color, border_alpha)] * len(border_paths)
            )
            border_outline_patches_by_depth.setdefault(depth, []).append(
                PathPatch(
                    outer_path,
                    facecolor="none",
                    edgecolor=to_rgba(stroke_color, border_alpha),
                    linewidth=max(float(style.stroke_width), 0.7),
                    linestyle=_cluster_linestyle(style.stroke_dash),
                    joinstyle="round",
                    zorder=0.075 + depth * 0.01,
                )
            )

        # Cluster label: position from style (label, label_fontsize already computed above)
        if style.label_position == "top-center":
            lx = (x_min + x_max) / 2
            ha = "center"
        elif style.label_position == "top-right":
            lx = x_max - label_ox
            ha = "right"
        else:  # "top-left" (default)
            lx = x_min + label_ox
            ha = "left"

        depth_label_offset = depth * (_points_to_data_units(ax, label_fontsize, "y")) * 1.4
        ly = y_max - label_oy - depth_label_offset

        if label:
            cluster_label_specs.append(
                DaguaText(
                    x=lx,
                    y=ly,
                    text=label,
                    font_size=label_fontsize,
                    font_family=label_ff,
                    font_weight=style.font_weight,
                    font_color=style.font_color,
                    alpha=1.0,
                    ha=ha,
                    va="top",
                    clip_on=False,
                    zorder=0.1 + depth * 0.01,
                    gid=f"dagua-cluster-label-{name}",
                )
            )
            if svg_hover_map is not None:
                svg_hover_map[f"dagua-cluster-label-{name}"] = _cluster_hover_text(
                    name,
                    graph,
                    indices,
                )

    for depth in sorted(fill_paths_by_depth):
        add_filled_collections(
            ax=ax,
            fill_paths=fill_paths_by_depth.get(depth, []),
            fill_colors=fill_colors_by_depth.get(depth, []),
            border_paths=border_paths_by_depth.get(depth, []),
            border_colors=border_colors_by_depth.get(depth, []),
            fill_zorder=0.0 + depth * 0.01,
            border_zorder=0.05 + depth * 0.01,
        )
        for border_patch in border_outline_patches_by_depth.get(depth, []):
            ax.add_patch(border_patch)

    if cluster_label_specs:
        render_text(ax, cluster_label_specs, display_scale, svg_hover_map)
