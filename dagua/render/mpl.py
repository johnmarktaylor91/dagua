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
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from dagua.edges import BezierCurve, preferred_edge_label_position, route_edges
from dagua.styles import (
    FONT_FAMILY,
    FONT_FAMILY_MONO,
    RESOLVED_FONT,
    darken_hex,
)
from dagua.utils import collect_cluster_leaves, measure_text, parse_rich_markup

_VECTOR_FORMATS = {"pdf", "ps", "eps", "svg", "svgz"}
_RASTER_FORMATS = {"png", "jpg", "jpeg", "webp", "tif", "tiff", "bmp"}
_GRAPHVIZ_DASH_PATTERN: Tuple[float, float] = (5.0, 3.0)
_GRAPHVIZ_DOT_PATTERN: Tuple[float, float] = (0.1, 3.0)


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
    _draw_edges(ax, graph, curves, svg_hover_map=svg_hover_map)

    # --- Layer 2: Nodes ---
    clip_patches = _draw_nodes(ax, graph, pos, sizes, svg_hover_map=svg_hover_map)

    # --- Layer 3: Node labels ---
    _draw_node_labels(ax, graph, pos, sizes, clip_patches, svg_hover_map=svg_hover_map)

    # --- Layer 4: Edge labels ---
    _draw_edge_labels(
        ax, graph, curves, label_positions=label_positions, svg_hover_map=svg_hover_map
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

    clip_patches: List[Any] = []
    for i in range(graph.num_nodes):
        x, y = float(pos[i, 0]), float(pos[i, 1])
        w, h = float(sizes[i, 0]), float(sizes[i, 1])
        style = graph.get_style_for_node(i)

        if style.shadow:
            _draw_shadow(ax, x, y, w, h, style)

        facecolor = to_rgba(style.fill, style.opacity)
        edgecolor = to_rgba(style.stroke, style.opacity * style.border_opacity)
        patch_face = "none" if style.gradient != "none" else facecolor
        patch = _build_node_patch(
            x,
            y,
            w,
            h,
            style,
            patch_face,
            edgecolor,
            style.stroke_width,
            _node_linestyle(style),
            zorder=2,
        )
        ax.add_patch(patch)
        if style.gradient != "none":
            _draw_gradient_fill(ax, patch, x, y, w, h, style)
        _draw_node_shape_extras(ax, x, y, w, h, style, edgecolor, zorder=2.05)
        clip_patches.append(patch)
        _set_svg_hover(patch, f"dagua-node-{i}", graph.node_labels[i], svg_hover_map)
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
    visual size is specified in points, such as arrowhead polygons, cluster
    corner radii, and cluster label offsets. Matplotlib already interprets
    ``linewidth``, ``fontsize``, and dash patterns in points natively.
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
) -> Tuple[float, float]:
    """Convert point-based marker dimensions to data units.

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

    Returns
    -------
    tuple[float, float]
        Marker ``(length, width)`` in data units.

    Notes
    -----
    Arrowhead polygons are drawn in data coordinates, but their visual size
    should stay stable in display space. Converting once here keeps marker
    geometry consistent across different graph extents and figure sizes.
    """
    _ = style
    scale = _compute_display_scale(ax)
    return length * scale, width * scale


def _label_anchor_x(align: str, x: float, w: float, pad_x: float, line_width: float) -> float:
    """Resolve the x anchor for a label line.

    Parameters
    ----------
    align : str
        Horizontal alignment value.
    x : float
        Node center x-coordinate.
    w : float
        Node width.
    pad_x : float
        Horizontal padding in data units.
    line_width : float
        Line width in data units.

    Returns
    -------
    float
        X coordinate for the line anchor.
    """
    if align == "left":
        return x - w / 2 + pad_x
    if align == "right":
        return x + w / 2 - pad_x - line_width
    return x - line_width / 2


def _label_anchor_y(valign: str, y: float, h: float, pad_y: float, block_height: float) -> float:
    """Resolve the top of a multi-line label block.

    Parameters
    ----------
    valign : str
        Vertical alignment value.
    y : float
        Node center y-coordinate.
    h : float
        Node height.
    pad_y : float
        Vertical padding in data units.
    block_height : float
        Total block height in data units.

    Returns
    -------
    float
        Top edge of the laid-out text block.
    """
    if valign == "top":
        return y + h / 2 - pad_y
    if valign == "bottom":
        return y - h / 2 + pad_y + block_height
    return y + block_height / 2


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


def _segment_font_properties(
    segment_style: Dict[str, Any], style: Any
) -> Tuple[str, str, str, str]:
    """Resolve font properties for one rich-text segment.

    Parameters
    ----------
    segment_style : dict[str, Any]
        Parsed rich-text formatting flags.
    style : Any
        Base node style.

    Returns
    -------
    tuple[str, str, str, str]
        Font family, font weight, font style, and color.
    """
    font_family = FONT_FAMILY_MONO[0] if segment_style.get("mono") else style.font_family_list[0]
    font_weight = "bold" if segment_style.get("bold") else style.font_weight
    font_style = "italic" if segment_style.get("italic") else style.font_style
    color = str(segment_style.get("color") or style.font_color)
    return font_family, font_weight, font_style, color


def _split_rich_lines(
    segments: Sequence[Tuple[str, Dict[str, Any]]],
) -> List[List[Tuple[str, Dict[str, Any]]]]:
    """Split rich-text segments into line-wise segment groups.

    Parameters
    ----------
    segments : sequence[tuple[str, dict[str, Any]]]
        Parsed rich-text segments.

    Returns
    -------
    list[list[tuple[str, dict[str, Any]]]]
        Segments grouped per output line.
    """
    lines: List[List[Tuple[str, Dict[str, Any]]]] = [[]]
    for text, segment_style in segments:
        parts = text.split("\n")
        for part_index, part in enumerate(parts):
            if part:
                lines[-1].append((part, segment_style))
            if part_index != len(parts) - 1:
                lines.append([])
    return lines


def _apply_text_effects(text_artist: Any, style: Any) -> None:
    """Apply optional outline effects to a text artist.

    Parameters
    ----------
    text_artist : Any
        Matplotlib text artist.
    style : Any
        Node style object.
    """
    if not style.text_outline:
        return

    import matplotlib.patheffects as pe

    text_artist.set_path_effects(
        [
            pe.withStroke(
                linewidth=style.text_outline_width,
                foreground=style.text_outline_color,
            )
        ]
    )


def _draw_text_decoration(
    ax: Any,
    x0: float,
    x1: float,
    y: float,
    color: str,
    clip_patch: Optional[Any],
    zorder: float,
) -> None:
    """Draw an underline or strike-through for a rich-text segment.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    x0 : float
        Decoration start x-coordinate.
    x1 : float
        Decoration end x-coordinate.
    y : float
        Decoration y-coordinate.
    color : str
        Decoration color.
    clip_patch : Any, optional
        Clip patch for keeping the decoration inside the node.
    zorder : float
        Artist z-order.
    """
    from matplotlib.lines import Line2D

    line = Line2D([x0, x1], [y, y], color=color, linewidth=1.0, zorder=zorder)
    if clip_patch is not None:
        line.set_clip_path(clip_patch)
    ax.add_line(line)


def _render_rich_label(
    ax: Any,
    label: str,
    x: float,
    y: float,
    w: float,
    h: float,
    font_size: float,
    style: Any,
    clip_patch: Optional[Any],
    base_gid: str,
    svg_hover_map: Optional[Dict[str, str]],
) -> None:
    """Render mixed-format node labels as per-segment text artists.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    label : str
        Rich-format label text.
    x : float
        Node center x-coordinate.
    y : float
        Node center y-coordinate.
    w : float
        Node width.
    h : float
        Node height.
    font_size : float
        Base font size in points.
    style : Any
        Node style object.
    clip_patch : Any, optional
        Clip patch for label rendering.
    base_gid : str
        SVG hover identifier prefix.
    svg_hover_map : dict[str, str], optional
        SVG hover text accumulator.
    """
    segments = parse_rich_markup(label)
    lines = _split_rich_lines(segments)
    pad_x = _points_to_data_units(ax, style.padding[0], "x")
    pad_y = _points_to_data_units(ax, style.padding[1], "y")
    line_height = _points_to_data_units(ax, font_size * 1.2, "y")
    total_height = max(line_height * len(lines), line_height)
    label_y = _label_reference_y(y, h, style.shape)
    block_top = _label_anchor_y(style.text_valign, label_y, h, pad_y, total_height)

    for line_index, line_segments in enumerate(lines):
        segment_widths: List[float] = []
        for segment_text, segment_style in line_segments:
            family, weight, _, _ = _segment_font_properties(segment_style, style)
            width_pt, _ = measure_text(segment_text, family, font_size, weight)
            segment_widths.append(_points_to_data_units(ax, width_pt, "x"))

        line_width = sum(segment_widths)
        current_x = _label_anchor_x(style.text_align, x, w, pad_x, line_width)
        line_y = block_top - (line_index + 0.5) * line_height

        for segment_index, (segment, segment_style) in enumerate(line_segments):
            family, weight, font_style, color = _segment_font_properties(segment_style, style)
            text_artist = ax.text(
                current_x,
                line_y,
                segment,
                ha="left",
                va="center",
                fontsize=font_size,
                fontfamily=family,
                fontweight=weight,
                fontstyle=font_style,
                color=color,
                zorder=3,
                clip_on=True,
            )
            if clip_patch is not None:
                text_artist.set_clip_path(clip_patch)
            _apply_text_effects(text_artist, style)
            _set_svg_hover(
                text_artist, f"{base_gid}-{line_index}-{segment_index}", label, svg_hover_map
            )

            segment_width = segment_widths[segment_index]
            if segment_style.get("underline"):
                underline_y = line_y - _points_to_data_units(ax, font_size * 0.32, "y")
                _draw_text_decoration(
                    ax,
                    current_x,
                    current_x + segment_width,
                    underline_y,
                    color,
                    clip_patch,
                    zorder=3.05,
                )
            if segment_style.get("strike"):
                strike_y = line_y + _points_to_data_units(ax, font_size * 0.08, "y")
                _draw_text_decoration(
                    ax,
                    current_x,
                    current_x + segment_width,
                    strike_y,
                    color,
                    clip_patch,
                    zorder=3.05,
                )
            current_x += segment_width


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
    clip_patch_seq: Sequence[Any] = clip_patches or []

    for i in range(graph.num_nodes):
        x, y = float(pos[i, 0]), float(pos[i, 1])
        w, h = float(sizes[i, 0]), float(sizes[i, 1])
        style = graph.get_style_for_node(i)
        label = graph.node_labels[i]
        clip_patch = clip_patch_seq[i] if i < len(clip_patch_seq) else None
        label_y = _label_reference_y(y, h, style.shape)

        if graph.node_font_sizes is not None and i < graph.node_font_sizes.shape[0]:
            fontsize = float(graph.node_font_sizes[i].item())
        else:
            fontsize = float(style.font_size)

        if style.label_format == "rich":
            _render_rich_label(
                ax,
                label,
                x,
                y,
                w,
                h,
                fontsize,
                style,
                clip_patch,
                f"dagua-node-label-{i}",
                svg_hover_map,
            )
            continue

        font_family = style.font_family_list[0]
        pad_x = _points_to_data_units(ax, style.padding[0], "x")
        pad_y = _points_to_data_units(ax, style.padding[1], "y")
        if "\n" not in label:
            text_x = _label_anchor_x(style.text_align, x, w, pad_x, 0.0)
            if style.text_align == "center":
                text_x = x
            elif style.text_align == "right":
                text_x = x + w / 2 - pad_x

            if style.text_valign == "top":
                text_y = label_y + h / 2 - pad_y
            elif style.text_valign == "bottom":
                text_y = label_y - h / 2 + pad_y
            else:
                text_y = label_y

            text_artist = ax.text(
                text_x,
                text_y,
                label,
                ha=style.text_align,
                va=style.text_valign,
                fontsize=fontsize,
                fontfamily=font_family,
                color=style.font_color,
                fontweight=style.font_weight,
                fontstyle=style.font_style,
                zorder=3,
                clip_on=True,
            )
            if clip_patch is not None:
                text_artist.set_clip_path(clip_patch)
            _apply_text_effects(text_artist, style)
            _set_svg_hover(text_artist, f"dagua-node-label-{i}", label, svg_hover_map)
            continue

        lines = label.split("\n")
        line_height = _points_to_data_units(ax, fontsize * 1.2, "y")
        total_height = line_height * len(lines)
        block_top = _label_anchor_y(style.text_valign, label_y, h, pad_y, total_height)
        text_x = (
            x
            if style.text_align == "center"
            else _label_anchor_x(style.text_align, x, w, pad_x, 0.0)
        )
        if style.text_align == "right":
            text_x = x + w / 2 - pad_x

        for j, line in enumerate(lines):
            line_y = block_top - (j + 0.5) * line_height
            line_font_size = fontsize if j == 0 else fontsize * gs.node_label_secondary_scale
            text_artist = ax.text(
                text_x,
                line_y,
                line,
                ha=style.text_align,
                va="center",
                fontsize=line_font_size,
                fontfamily=font_family,
                color=style.font_color,
                fontweight=style.font_weight,
                fontstyle=style.font_style,
                zorder=3,
                clip_on=True,
            )
            if clip_patch is not None:
                text_artist.set_clip_path(clip_patch)
            _apply_text_effects(text_artist, style)
            _set_svg_hover(text_artist, f"dagua-node-label-{i}-{j}", label, svg_hover_map)


def _draw_edge_marker(
    ax: Any,
    point: Tuple[float, float],
    direction: Tuple[float, float],
    marker: str,
    style: Any,
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
    manual_length, manual_width = _marker_data_size(ax, style, length, width)
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
) -> None:
    """Draw bezier edges with configurable endpoint markers.

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
    """
    from matplotlib.colors import to_rgba
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    for e_idx, curve in enumerate(curves):
        style = graph.get_style_for_edge(e_idx)
        verts = [curve.p0, curve.cp1, curve.cp2, curve.p1]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        path_patch = PathPatch(
            Path(verts, codes),
            facecolor="none",
            edgecolor=to_rgba(style.color, style.opacity),
            linewidth=style.width,
            linestyle=_edge_linestyle(style),
            capstyle="round",
            joinstyle="round",
            zorder=1,
        )
        ax.add_patch(path_patch)
        _set_svg_hover(
            path_patch, f"dagua-edge-{e_idx}", _edge_hover_text(graph, e_idx), svg_hover_map
        )
        # Target arrow: direction continues past the endpoint (same sense as
        # the curve's tangent at p1).  For bezier cp2-p1 already points that
        # way; for straight routing the control point collapses onto p1, so
        # we fall back to the overall edge direction p1-p0.
        head_dx = curve.cp2[0] - curve.p1[0]
        head_dy = curve.cp2[1] - curve.p1[1]
        if head_dx * head_dx + head_dy * head_dy < 1e-12:
            head_dx = curve.p1[0] - curve.p0[0]
            head_dy = curve.p1[1] - curve.p0[1]
        _draw_edge_marker(
            ax,
            curve.p1,
            (head_dx, head_dy),
            style.arrow,
            style,
        )
        # Source tail arrow: same tangent logic — cp1-p0 continues past the
        # source; fallback to p0-p1 for straight routing where cp1==p0.
        tail_dx = curve.cp1[0] - curve.p0[0]
        tail_dy = curve.cp1[1] - curve.p0[1]
        if tail_dx * tail_dx + tail_dy * tail_dy < 1e-12:
            tail_dx = curve.p0[0] - curve.p1[0]
            tail_dy = curve.p0[1] - curve.p1[1]
        _draw_edge_marker(
            ax,
            curve.p0,
            (tail_dx, tail_dy),
            style.tail_arrow,
            style,
        )


def _draw_edge_labels(
    ax: Any,
    graph: Any,
    curves: List[BezierCurve],
    label_positions: Optional[List[Optional[Tuple[float, float]]]] = None,
    svg_hover_map: Optional[Dict[str, str]] = None,
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
    """
    gs = graph.graph_style

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

        font_family = style.label_font_family or RESOLVED_FONT
        text_artist = ax.text(
            lx,
            ly,
            label,
            ha="center",
            va="center",
            fontsize=style.label_font_size,
            fontweight=style.label_font_weight,
            fontfamily=font_family,
            color=style.label_font_color,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor=style.label_background,
                edgecolor="none",
                alpha=gs.edge_label_background_opacity,
            ),
            zorder=4,
        )
        _set_svg_hover(
            text_artist, f"dagua-edge-label-{e_idx}", _edge_hover_text(graph, e_idx), svg_hover_map
        )


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
    from matplotlib.patches import FancyBboxPatch

    if not graph.clusters:
        return

    # Compute true hierarchy depth per cluster via parent chain
    cluster_parents = getattr(graph, "cluster_parents", {})
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
        label = graph.cluster_labels.get(name, name)
        label_fontsize = max(style.font_size - depth * 1.0, 7.0)
        label_ff = style.font_family or RESOLVED_FONT
        display_scale = _compute_display_scale(ax)
        label_ox = style.label_offset[0] * display_scale
        label_oy = style.label_offset[1] * display_scale
        label_width, label_height = measure_text(
            label,
            font_family=label_ff,
            font_size=label_fontsize,
            font_weight=style.font_weight,
        )

        y_min = (member_pos[:, 1] - member_sizes[:, 1] / 2).min() - padding
        y_max = (
            (member_pos[:, 1] + member_sizes[:, 1] / 2).max() + padding + max(14.0, label_height)
        )

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

        # Opacity decreases with depth
        max_depth = len(sorted_clusters)
        opacity = style.opacity * (1 - depth * 0.15 / max(max_depth, 1))
        opacity = max(opacity, 0.08)

        # Corner radius
        corner_radius = style.corner_radius * display_scale
        if corner_radius > 0:
            boxstyle = f"round,pad=0,rounding_size={corner_radius}"
        else:
            boxstyle = "square,pad=0"

        # Stroke dash
        patch = FancyBboxPatch(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            boxstyle=boxstyle,
            facecolor=fill_color,
            edgecolor=stroke_color,
            linewidth=style.stroke_width,
            linestyle=_cluster_linestyle(style.stroke_dash),
            alpha=opacity,
            zorder=0,
        )
        ax.add_patch(patch)
        _set_svg_hover(
            patch, f"dagua-cluster-{name}", _cluster_hover_text(name, graph, indices), svg_hover_map
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

        # Offset label further down for nested clusters to prevent overlap
        depth_label_offset = depth * label_fontsize * 1.4
        ly = y_max - label_oy - depth_label_offset

        text_artist = ax.text(
            lx,
            ly,
            label,
            fontsize=label_fontsize,
            fontweight=style.font_weight,
            fontfamily=label_ff,
            color=style.font_color,
            va="top",
            ha=ha,
            zorder=0.5,
            clip_on=False,
        )
        _set_svg_hover(
            text_artist,
            f"dagua-cluster-label-{name}",
            _cluster_hover_text(name, graph, indices),
            svg_hover_map,
        )
