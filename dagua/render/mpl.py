"""Matplotlib renderer for DaguaGraph.

Publication-quality rendering following the Dagua Aesthetic Style Guide:
- Wong/Okabe-Ito colorblind-safe palette
- Muted fills, strong borders, quiet edges
- Helvetica/Arial typography
- Warm white background (#FAFAFA)
- Layered rendering: clusters -> edges -> nodes -> labels
"""

from __future__ import annotations

import colorsys
import gzip
import io
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np
import torch

from dagua.edges import (
    BezierCurve,
    bezier_tangent,
    clip_edge_at_cluster_boundaries,
    edge_endpoint_label_position,
    evaluate_bezier,
    preferred_edge_label_position,
    route_edges,
)
from dagua.render._backend import _render_via_cairosvg, stroke_width_scale_for
from dagua.render.borders import (
    GRAPHVIZ_COMPONENT_DETAIL,
    GRAPHVIZ_M_CIRCLE_CHORD_RATIO,
    GRAPHVIZ_M_CORNER_MARK_LENGTH,
    GRAPHVIZ_NOTE_FOLD_SIZE,
    GRAPHVIZ_OCTAGON_PERIPHERY_GAP,
    GRAPHVIZ_TAB_WIDTH,
    ShapeSpec,
    add_corner_radius,
    add_filled_collections,
    annular_path,
    build_shape_path,
    clamp_border_width,
    dash_ribbon_paths,
    graphviz_octagon_path,
    inset_shape_path,
    make_clip_proxy,
    scale_corner_radius,
)
from dagua.render.crossings import EdgeCrossing, detect_crossings
from dagua.render.edges import CubicBezier as RenderBezier
from dagua.render.edges import DaguaEdge, DaguaEdgeCollection, place_edge_label
from dagua.render.edges.collection import MIN_TAPER_WIDTH
from dagua.render.edges.dashes import dash_curve, parse_dash_pattern
from dagua.render.edges.geometry import (
    adaptive_subdivide,
    polyline_from_samples,
)
from dagua.render.edges.geometry import (
    sample_curve as sample_render_curve,
)
from dagua.render.edges.geometry import (
    subcurve as subcurve_render,
)
from dagua.render.edges.ribbon import polyline_ribbon_path
from dagua.render.text import DaguaText, measure_text_data, render_text
from dagua.styles import (
    FONT_FAMILY,
    RESOLVED_FONT,
    THEME_REGISTRY,
    ClusterStyle,
    EdgeStyle,
    GraphStyle,
    NodeStyle,
    Theme,
    darken_hex,
)
from dagua.utils import (
    collect_cluster_leaves,
    prepare_label_text,
)

_VECTOR_FORMATS = {"pdf", "ps", "eps", "svg", "svgz"}
_RASTER_FORMATS = {"png", "jpg", "jpeg", "webp", "tif", "tiff", "bmp"}
_GRAPHVIZ_DASH_PATTERN: Tuple[float, float] = (5.0, 3.0)
# Tuned from ``(0.1, 3.0)`` so Graphviz-style dotted strokes still read as
# repeated dots after point-to-data conversion instead of collapsing into a
# nearly solid line on high-DPI exports.
_GRAPHVIZ_DOT_PATTERN: Tuple[float, float] = (1.2, 3.0)
_ARROWHEAD_REFERENCE_WIDTH_POINTS = 1.2
_DOUBLE_BORDER_INSET_FACTOR = 2.5
_PATTERN_FILL_RESOLUTION = 128
_HATCH_PATTERN = "////"
_MIN_HATCH_LINEWIDTH_POINTS = 0.8
_CROSSING_CLEARANCE_PADDING_POINTS = 3.0
# Minimum crossing bridge width relative to stroke width so very thin edges
# still produce a visibly separated jump instead of a pinched cusp.
_CROSSING_MIN_SPAN_WIDTH_FACTOR = 4.0
# Absolute minimum bridge width in points for small edges and dense figures.
_CROSSING_MIN_SPAN_POINTS = 14.0
# Data-space floor that keeps jump spans readable after axis scaling changes.
_CROSSING_MIN_SPAN_DATA_UNITS = 22.0
# Height multiplier for the sharp crossing style. Larger values exaggerate the
# bridge arch; ``3.5`` was the best balance between recognizability and not
# looking like a self-loop when the crossing span is narrow.
_CROSSING_SHARP_HEIGHT_WIDTH_FACTOR = 3.5
# Span multiplier for the sharp crossing footprint along the edge direction.
_CROSSING_SHARP_SPAN_WIDTH_FACTOR = 4.0
_CROSSING_BRIDGE_HEIGHT_WIDTH_FACTOR = 4.0
_CROSSING_BRIDGE_SPAN_WIDTH_FACTOR = 6.0
_CROSSING_BRIDGE_CORNER_RADIUS_POINTS = 1.5
_CROSSING_BRIDGE_STROKE_WIDTH_POINTS = 1.5
_BEVEL_BAND_COUNT = 8
_BEVEL_REFERENCE_INTENSITY = 0.5
_BEVEL_HIGHLIGHT_ALPHA = 0.55
_BEVEL_SHADOW_ALPHA = 0.35
_BEVEL_MAX_INSET_FRACTION = 0.5
_PORT_INDICATOR_BORDER_WIDTH_POINTS = 1.0
_PORT_INDICATOR_ZORDER = 4.1
_DIRECT_ARROW_TRIM_MAX_FRACTION = 0.4
# Tuned down over several passes to keep self-loop terminals legible without
# letting arrowheads consume the entire loop apex on small nodes.
_SELF_LOOP_ARROWHEAD_MAX_NODE_FRACTION = 0.18
_SELF_LOOP_ARROWHEAD_MAX_WIDTH_RATIO = 0.55
_CLUSTER_LABEL_VERTICAL_GAP_POINTS = 2.0
_GRAPHVIZ_STRICT_EDGE_WIDTH_RENDER_MULTIPLIER = 1.5
_GRAPHVIZ_STRICT_CLUSTER_HORIZONTAL_SEPARATION_POINTS = 18.0
_GRAPHVIZ_STRICT_CLUSTER_LABEL_MASK_PADDING_POINTS = 4.0
_GRAPHVIZ_STRICT_CLUSTER_EXTERNAL_NODE_GAP_POINTS = 36.0
_GRAPHVIZ_STRICT_BACK_EDGE_OFFSET_FLOOR_POINTS = 60.0
_GRAPHVIZ_STRICT_BACK_EDGE_OFFSET_FACTOR = 0.45
_EDGE_LABEL_COLLISION_PADDING_POINTS = 2.0
_DEFAULT_NODE_LABEL_FONT_POINTS = 8.5
# Tuned down from ``8.0`` so external labels stay subordinate to node labels
# and fit more consistently around dense gallery fixtures.
_DEFAULT_EXTERNAL_LABEL_FONT_POINTS = 7.0
_DEFAULT_EDGE_LABEL_FONT_POINTS = 7.0
_DEFAULT_CLUSTER_LABEL_FONT_POINTS = 9.5
_DEFAULT_TITLE_FONT_POINTS = 10.0
_BOLD_NODE_LABEL_SIZE_MULTIPLIER = 1.05
_PIE_GRADIENT_OVERLAY_ALPHA_MULTIPLIER = 0.4
_NODE_LABEL_HEIGHT_FRACTION = 0.35
_NODE_LABEL_MIN_HEIGHT_FRACTION = 0.1
_NODE_LABEL_MAX_HEIGHT_FRACTION = 0.6
_ELLIPSE_VERTICAL_LABEL_INSET_FRACTION = 0.15
_MULTILINE_LABEL_REDUCTION = 0.5
# Tuned from ``0.25`` so edge labels claim less vertical band height and stop
# overpowering narrow ribbons or stacked parallel edges.
_EDGE_LABEL_HEIGHT_FRACTION = 0.18
_CLUSTER_LABEL_HEIGHT_FRACTION = 0.06
_CLUSTER_LABEL_ZORDER = 2.12


def _text_font_family(style: Any) -> str:
    """Return the concrete font family name used for text-path rendering.

    Parameters
    ----------
    style : Any
        Style object exposing an optional ``font_family`` field.

    Returns
    -------
    str
        Resolved family name. Empty requests fall back to ``RESOLVED_FONT`` so
        text-path conversion uses an installed family with matching italics.
    """
    requested_family = str(getattr(style, "font_family", "")).strip()
    return requested_family or RESOLVED_FONT


def _normalize_text_font_weight(font_weight: Any) -> str:
    """Return a stable renderer font-weight token.

    Parameters
    ----------
    font_weight : Any
        Requested font-weight token or numeric value.

    Returns
    -------
    str
        Normalized weight. Bold-like numeric tokens are converted to
        ``"bold"`` so all node-label paths request the heavy face explicitly.
    """
    raw_weight = str(font_weight).strip().lower()
    if raw_weight == "":
        return "regular"
    try:
        numeric_weight = float(raw_weight)
    except ValueError:
        return "bold" if raw_weight == "700" else raw_weight
    if numeric_weight >= 600.0:
        return "bold"
    if numeric_weight <= 400.0:
        return "regular"
    return str(int(numeric_weight)) if numeric_weight.is_integer() else raw_weight


def _is_bold_font_weight(font_weight: Any) -> bool:
    """Return whether a renderer weight token requests bold text.

    Parameters
    ----------
    font_weight : Any
        Requested font-weight token or numeric value.

    Returns
    -------
    bool
        ``True`` when the weight should render as bold.
    """
    return _normalize_text_font_weight(font_weight) == "bold"


_CLUSTER_LABEL_MIN_NODE_HEIGHT_FRACTION = 0.3
_TITLE_HEIGHT_FRACTION = 0.03
_TITLE_BAND_HEIGHT_MULTIPLIER = 1.8
_DARK_BACKGROUND_LUMINANCE_THRESHOLD = 0.5
_AUTO_CONTRAST_FILL_LUMINANCE = 0.85
_AUTO_CONTRAST_STROKE_LUMINANCE = 0.7
_AUTO_CONTRAST_EDGE_LUMINANCE = 0.7
_AUTO_CONTRAST_TEXT_COLOR = "#f5f5f5"
_AUTO_CONTRAST_NODE_FILL_BLEND = 0.22
_AUTO_CONTRAST_NODE_STROKE_BLEND = 0.72
_AUTO_CONTRAST_EDGE_BLEND = 0.78
_AUTO_CONTRAST_CLUSTER_FILL_BLEND = 0.16
_AUTO_CONTRAST_CLUSTER_STROKE_BLEND = 0.5
_AUTO_CONTRAST_LABEL_BACKGROUND_BLEND = 0.18
_AUTO_CONTRAST_TEXT_BLEND = 0.9
_DENSITY_SHRINK_SPARSE_NODE_COUNT = 2
_DENSITY_SHRINK_REFERENCE_NODE_COUNT = 0.3
_DENSITY_SHRINK_CLAMP_NODE_COUNT = 20
_DENSITY_LABEL_FONT_FLOOR = 0.5
_CLUSTER_RENDER_BBOX_EXTRA_CAP_POINTS = 2.0
_MIN_CLUSTER_INNER_HEIGHT_POINTS = 1.0
_MIN_VISIBLE_STROKE_POINTS = 2.3


def _font_size_user_scale(font_size_points: float, baseline_points: float) -> float:
    """Return a relative font-size multiplier against a default point size.

    Parameters
    ----------
    font_size_points : float
        User-facing font size in typographic points.
    baseline_points : float
        Default point size used as the relative baseline.

    Returns
    -------
    float
        Scale factor applied to the data-coordinate size heuristic.
    """
    safe_baseline = max(float(baseline_points), 1e-9)
    return max(float(font_size_points), 0.0) / safe_baseline


def _multiline_label_scale(text: str) -> float:
    """Return a down-scaling factor for multi-line labels.

    Parameters
    ----------
    text : str
        Label text that may contain explicit newline breaks.

    Returns
    -------
    float
        Multiplicative scale factor. Single-line labels return ``1.0``.
    """
    line_count = text.count("\n") + 1
    if line_count <= 1:
        return 1.0
    return 1.0 / (1.0 + (line_count - 1) * _MULTILINE_LABEL_REDUCTION)


def _node_relative_font_size_data(
    text: str,
    node_height: float,
    font_size_points: float,
    baseline_points: float,
    font_weight: Any = "regular",
) -> float:
    """Compute a node-relative label size in data coordinates.

    Parameters
    ----------
    text : str
        Label text.
    node_height : float
        Node height in data units.
    font_size_points : float
        User-facing font size in points.
    baseline_points : float
        Default point size for relative scaling.
    font_weight : Any, default="regular"
        Font-weight token used to apply the node-label bold visibility bump.

    Returns
    -------
    float
        Target label size in data units.
    """
    clamped_height = max(float(node_height), 1e-9)
    user_scale = _font_size_user_scale(font_size_points, baseline_points)
    font_size_data = (
        clamped_height * _NODE_LABEL_HEIGHT_FRACTION * _multiline_label_scale(text) * user_scale
    )
    if _is_bold_font_weight(font_weight):
        font_size_data *= _BOLD_NODE_LABEL_SIZE_MULTIPLIER
    font_size_data = max(font_size_data, clamped_height * _NODE_LABEL_MIN_HEIGHT_FRACTION)
    return min(font_size_data, clamped_height * _NODE_LABEL_MAX_HEIGHT_FRACTION)


def _edge_font_size_data(
    text: str,
    avg_node_height: float,
    font_size_points: float,
) -> float:
    """Compute a graph-relative edge-label size in data coordinates.

    Parameters
    ----------
    text : str
        Edge label text.
    avg_node_height : float
        Average node height in data units.
    font_size_points : float
        User-facing edge-label size in points.

    Returns
    -------
    float
        Target edge-label size in data units.
    """
    return (
        max(float(avg_node_height), 1e-9)
        * _EDGE_LABEL_HEIGHT_FRACTION
        * _multiline_label_scale(text)
        * _font_size_user_scale(font_size_points, _DEFAULT_EDGE_LABEL_FONT_POINTS)
    )


def _cluster_font_size_data(
    text: str,
    cluster_height: float,
    min_node_height: float,
    font_size_points: float,
    font_size_scaling: str = "by_height",
    display_scale: float = 1.0,
) -> float:
    """Compute a cluster-label size in data coordinates.

    Parameters
    ----------
    text : str
        Cluster label text.
    cluster_height : float
        Cluster height in data units.
    min_node_height : float
        Minimum node height in the current render pass.
    font_size_points : float
        User-facing cluster font size in points.
    font_size_scaling : str, default="by_height"
        Cluster label scaling mode. ``"fixed"`` keeps the authored point size
        authoritative; ``"by_height"`` preserves the legacy height-based
        cluster-label scaling.
    display_scale : float, default=1.0
        Point-to-data conversion used when fixed point sizing is requested.

    Returns
    -------
    float
        Target cluster-label size in data units.
    """
    if font_size_scaling == "fixed":
        fixed_size = max(float(font_size_points), 1e-9) * max(float(display_scale), 1e-9)
        height_cap = max(float(cluster_height), 0.0) / 8.0
        return min(fixed_size, height_cap) if height_cap > 0.0 else fixed_size

    base_size_data = max(
        max(float(cluster_height), 0.0) * _CLUSTER_LABEL_HEIGHT_FRACTION,
        max(float(min_node_height), 0.0) * _CLUSTER_LABEL_MIN_NODE_HEIGHT_FRACTION,
    )
    scaled_size = (
        base_size_data
        * _multiline_label_scale(text)
        * _font_size_user_scale(font_size_points, _DEFAULT_CLUSTER_LABEL_FONT_POINTS)
    )
    return scaled_size


def _cluster_fill_alpha(style: ClusterStyle, depth: int) -> float:
    """Return the effective fill alpha for one cluster.

    Parameters
    ----------
    style : ClusterStyle
        Cluster style after theme/style cascade resolution.
    depth : int
        Cluster nesting depth.

    Returns
    -------
    float
        Fill alpha clamped to Matplotlib's valid ``[0, 1]`` range.
    """
    fill_color = str(getattr(style, "fill", "") or "").strip().lower()
    if fill_color in {"", "none", "transparent"}:
        return 0.0
    depth_opacity_step = float(getattr(style, "depth_opacity_step", -0.05))
    base_alpha = 1.0 if style.fill_opacity is None else float(style.fill_opacity)
    alpha = (base_alpha * float(style.opacity)) + float(depth) * depth_opacity_step
    return min(max(alpha, 0.0), 1.0)


def _cluster_border_alpha(style: ClusterStyle, depth: int) -> float:
    """Return the effective border alpha for one cluster.

    Parameters
    ----------
    style : ClusterStyle
        Cluster style after theme/style cascade resolution.
    depth : int
        Cluster nesting depth.

    Returns
    -------
    float
        Stroke alpha clamped to Matplotlib's valid ``[0, 1]`` range.
    """
    depth_opacity_step = float(getattr(style, "depth_opacity_step", -0.05))
    if style.border_opacity is not None:
        alpha = (float(style.border_opacity) * float(style.opacity)) + (
            float(depth) * depth_opacity_step
        )
        return min(max(alpha, 0.0), 1.0)
    legacy_alpha = min(
        max(float(style.opacity) * 2.5, 0.6) + float(depth) * depth_opacity_step,
        1.0,
    )
    return min(max(legacy_alpha, 0.0), 1.0)


def _title_font_size_data(graph_height: float, font_size_points: float) -> float:
    """Compute a graph-title size in data coordinates.

    Parameters
    ----------
    graph_height : float
        Graph height in data units.
    font_size_points : float
        User-facing title font size in points.

    Returns
    -------
    float
        Target title size in data units.
    """
    return (
        max(float(graph_height), 1e-9)
        * _TITLE_HEIGHT_FRACTION
        * _font_size_user_scale(font_size_points, _DEFAULT_TITLE_FONT_POINTS)
    )


def _effective_font_size_points(font_size_data: float, display_scale: float) -> float:
    """Convert a desired data-coordinate font size into renderer input points.

    Parameters
    ----------
    font_size_data : float
        Desired font size in data units.
    display_scale : float
        Current point-to-data scale factor.

    Returns
    -------
    float
        Font size value to pass into ``DaguaText``.
    """
    safe_scale = max(float(display_scale), 1e-9)
    return max(float(font_size_data), 1e-9) / safe_scale


def _relative_luminance(color: str) -> float:
    """Return the sRGB relative luminance for a matplotlib color.

    Parameters
    ----------
    color : str
        Matplotlib-compatible color string.

    Returns
    -------
    float
        Relative luminance on the ``[0, 1]`` scale.
    """
    from matplotlib.colors import to_rgba

    red, green, blue, _ = to_rgba(color)
    return (0.2126 * float(red)) + (0.7152 * float(green)) + (0.0722 * float(blue))


def _relative_luminance_rgb(color: Tuple[float, float, float]) -> float:
    """Return sRGB relative luminance for an RGB tuple.

    Parameters
    ----------
    color : tuple[float, float, float]
        RGB color on the ``[0, 1]`` scale.

    Returns
    -------
    float
        Relative luminance on the ``[0, 1]`` scale.
    """
    red, green, blue = color
    return (0.2126 * float(red)) + (0.7152 * float(green)) + (0.0722 * float(blue))


def _adapt_color_for_dark_bg(color: str, target_luminance: float) -> str:
    """Lighten a color for dark backgrounds while preserving hue and saturation.

    Parameters
    ----------
    color : str
        Matplotlib-compatible source color.
    target_luminance : float
        Desired minimum relative luminance on the ``[0, 1]`` scale.

    Returns
    -------
    str
        Hex color with the original hue preserved and enough lightness for dark
        background contrast. Colors already bright enough are returned unchanged.
    """
    from matplotlib.colors import to_hex, to_rgb

    red, green, blue = to_rgb(color)
    current_rgb = (float(red), float(green), float(blue))
    clamped_target = min(max(float(target_luminance), 0.0), 1.0)
    if _relative_luminance_rgb(current_rgb) >= clamped_target:
        return str(color)

    hue, lightness, saturation = colorsys.rgb_to_hls(*current_rgb)
    low = float(lightness)
    high = 1.0
    adapted_rgb = current_rgb

    # Search in HLS lightness so the renderer keeps the original tint instead of
    # washing built-in theme colors into flat white.
    for _ in range(12):
        midpoint = (low + high) / 2.0
        candidate_rgb = colorsys.hls_to_rgb(hue, midpoint, saturation)
        candidate_luminance = _relative_luminance_rgb(candidate_rgb)
        if candidate_luminance < clamped_target:
            low = midpoint
            continue
        high = midpoint
        adapted_rgb = candidate_rgb

    return str(to_hex(adapted_rgb, keep_alpha=False)).lower()


def _is_dark_background(color: str) -> bool:
    """Return whether a render background should be treated as dark.

    Parameters
    ----------
    color : str
        Matplotlib-compatible background color.

    Returns
    -------
    bool
        ``True`` when the luminance is below the dark-background threshold.
    """
    return _relative_luminance(color) < _DARK_BACKGROUND_LUMINANCE_THRESHOLD


def _normalize_color_token(color: str) -> str:
    """Normalize a color string for stable equality checks.

    Parameters
    ----------
    color : str
        Matplotlib-compatible color string.

    Returns
    -------
    str
        Lower-cased hex representation when possible, otherwise the stripped
        original token.
    """
    from matplotlib.colors import to_hex

    normalized = str(color).strip()
    if normalized == "":
        return ""
    try:
        return str(to_hex(normalized, keep_alpha=False)).lower()
    except ValueError:
        return normalized.lower()


def _colors_match(first: str, second: str) -> bool:
    """Return whether two style color tokens resolve to the same color.

    Parameters
    ----------
    first : str
        First color string.
    second : str
        Second color string.

    Returns
    -------
    bool
        ``True`` when both tokens normalize to the same color.
    """
    return _normalize_color_token(first) == _normalize_color_token(second)


def _blend_colors(background_color: str, foreground_color: str, amount: float) -> str:
    """Blend a foreground color onto a background and return a hex color.

    Parameters
    ----------
    background_color : str
        Matplotlib-compatible background color.
    foreground_color : str
        Matplotlib-compatible foreground color.
    amount : float
        Blend weight for the foreground on ``[0, 1]``.

    Returns
    -------
    str
        Hex color representing the blended RGB value.
    """
    from matplotlib.colors import to_hex, to_rgba

    clamped_amount = min(max(float(amount), 0.0), 1.0)
    bg_red, bg_green, bg_blue, _ = to_rgba(background_color)
    fg_red, fg_green, fg_blue, _ = to_rgba(foreground_color)
    blended = (
        (1.0 - clamped_amount) * bg_red + clamped_amount * fg_red,
        (1.0 - clamped_amount) * bg_green + clamped_amount * fg_green,
        (1.0 - clamped_amount) * bg_blue + clamped_amount * fg_blue,
        1.0,
    )
    return str(to_hex(blended, keep_alpha=False)).lower()


def _builtin_theme_for_render(graph: Any) -> Optional[Theme]:
    """Return the matching built-in theme used as the contrast baseline.

    Parameters
    ----------
    graph : Any
        Graph exposing a ``_theme`` attribute.

    Returns
    -------
    Theme | None
        Built-in theme with the same name, or ``None`` when the current theme
        is custom and should not receive renderer-side color adaptation.
    """
    theme = getattr(graph, "_theme", None)
    theme_name = getattr(theme, "name", "")
    if not isinstance(theme_name, str):
        return None
    return THEME_REGISTRY.get(theme_name)


def _render_theme_name(graph: Any) -> str:
    """Return the active graph theme name for render-only compatibility gates.

    Parameters
    ----------
    graph : Any
        Graph exposing an optional ``_theme`` attribute.

    Returns
    -------
    str
        Theme name, or an empty string when it cannot be resolved.
    """
    theme = getattr(graph, "_theme", None)
    theme_name = getattr(theme, "name", "")
    return theme_name if isinstance(theme_name, str) else ""


def _is_graphviz_strict_render(graph: Any) -> bool:
    """Return whether Graphviz strict cosmetic compatibility should apply.

    Parameters
    ----------
    graph : Any
        Graph exposing an optional ``_theme`` attribute.

    Returns
    -------
    bool
        ``True`` only for the built-in ``graphviz_strict`` theme.
    """
    return _render_theme_name(graph) == "graphviz_strict"


# Metric-driven (R19): theme value now matches dot's SVG-declared edge label
# font_size (14pt) directly. Earlier rounds (13/15/17) applied a sub-1.0 ratio
# to compensate for VLM-perceived size mismatches that turned out to be
# AA / DPI artifacts in the matplotlib pipeline, not actual size discrepancies
# in dot's output. The metric (parity_metrics.py) confirmed dot's edge-label
# font-size is 14pt — same as node labels — so the ratio collapses to 1.0.
_STRICT_EDGE_LABEL_NODE_RATIO = 1.0


def _strict_edge_label_font_size(graph: Any, fallback_points: float) -> float:
    """Return the graphviz_strict edge label font size override.

    Parameters
    ----------
    graph : Any
        Graph exposing the active theme.
    fallback_points : float
        Per-edge ``label_font_size`` value to fall back to when the strict
        theme is not active or the graph-level default is unset.

    Returns
    -------
    float
        Edge label font size in typographic points.

    Notes
    -----
    Round 11 F2: gallery fixtures hardcode ``label_font_size=10`` on the
    per-edge :class:`EdgeStyle` for arrow_types and edge_styles_showcase.
    The 5-level cascade gives those overrides priority over the strict
    theme's ``edge_label_font_size=16``, leaving standalone edge labels
    visibly smaller than dot's labels at the same content. graphviz_strict
    is a "match-dot exactly" theme; treating its graph-level edge-label
    point size as authoritative closes the gap without disturbing other
    themes' per-edge customization.

    Round 13 F4: round-11 emitted the graph-level 16pt directly, which made
    edge labels visibly LARGER than dot's (audit: ~14-15px cap-height vs
    dot's ~9-10px on state_machine and arrow_types). dot renders edge
    labels at ~10pt while node labels run ~14pt. dagua's strict node-label
    point size is 16pt (round-9 cap-height compensation), so the matching
    edge-label point size is 16 * 10/14 = ~11.43pt. Apply that ratio so
    edge labels stay subordinate to node labels at the same FreeType
    rendering compensation factor.

    Round 15 F5: round-13 (10/14) brought edge labels into subordination
    but the audit measured them still 10-15% larger than dot's absolute
    size on state_machine and arrow_types. Tightening the ratio to
    9.3/14 (= ~0.664) yields 16 * 0.664 = ~10.6pt, dropping edge labels
    fully below dot's measured cap height while keeping the
    subordination contract.

    Round 17 F2: 9.3/14 over-corrected -- the round-16 audit measured
    dagua edge labels at ~75% of dot's cap height (25% TOO SMALL).
    Round 13's 10/14 made them 10-15% too large; round 15's 9.3/14 made
    them 25% too small. Land between by going a hair LARGER than 10/14
    -- ratio 11/14 (= ~0.786) yields 16 * 0.786 = ~12.6pt rendered.
    The 0.6pt bump over round-13 compensates for matplotlib's pt-to-px
    floor where 9.3pt and 10pt rounded to the same pixel-row glyph.
    """
    if not _is_graphviz_strict_render(graph):
        return fallback_points
    theme = getattr(graph, "_theme", None)
    graph_style = getattr(theme, "graph_style", None) if theme is not None else None
    override = getattr(graph_style, "edge_label_font_size", None)
    if override is None:
        return fallback_points
    return float(override) * _STRICT_EDGE_LABEL_NODE_RATIO


def _strict_absolute_edge_label_font_data(
    graph: Any,
    font_size_points: float,
    display_scale: float,
) -> Optional[float]:
    """Return an absolute-pt edge label size in data coordinates.

    Parameters
    ----------
    graph : Any
        Graph exposing the active theme.
    font_size_points : float
        Desired absolute font size in typographic points.
    display_scale : float
        Point-to-data conversion factor.

    Returns
    -------
    float | None
        Absolute label size in data units when graphviz_strict is active,
        ``None`` otherwise so callers fall back to graph-relative sizing.

    Notes
    -----
    Round 11 F2: dagua's general edge-label sizing is graph-relative
    (``avg_node_height * 0.18 * font_pt/7``). On panels with small nodes
    that produces sub-10pt labels even when the user asks for 16pt. dot
    sizes labels in absolute points regardless of node geometry, so for
    graphviz_strict bypass the graph-relative scaling and emit the
    requested point size directly.
    """
    if not _is_graphviz_strict_render(graph):
        return None
    return max(float(font_size_points), 1e-9) * max(float(display_scale), 1e-9)


def _theme_edge_type(graph: Any, edge_idx: int) -> str:
    """Resolve the built-in theme edge key for one edge.

    Parameters
    ----------
    graph : Any
        Graph exposing edge metadata.
    edge_idx : int
        Edge index.

    Returns
    -------
    str
        Theme edge-style key such as ``"default"`` or ``"back"``.
    """
    back_edge_mask = getattr(graph, "_back_edge_mask", None)
    if back_edge_mask is not None and edge_idx < int(back_edge_mask.shape[0]):
        if bool(back_edge_mask[edge_idx].item()):
            return "back"
    edge_types = getattr(graph, "edge_types", [])
    if edge_idx < len(edge_types):
        return str(edge_types[edge_idx])
    return "default"


def _should_auto_contrast(graph: Any, background_color: str) -> bool:
    """Return whether render-time contrast adaptation should run.

    Parameters
    ----------
    graph : Any
        Graph exposing a built-in theme name.
    background_color : str
        Effective graph background color.

    Returns
    -------
    bool
        ``True`` when a built-in light theme is being rendered on a dark
        background.
    """
    builtin_theme = _builtin_theme_for_render(graph)
    if builtin_theme is None:
        return False
    baseline_background = str(builtin_theme.graph_style.background_color)
    return _is_dark_background(background_color) and not _is_dark_background(baseline_background)


def _graph_style_for_render(graph: Any) -> GraphStyle:
    """Return graph-level render settings with dark-background title contrast.

    Parameters
    ----------
    graph : Any
        Graph exposing graph-style and theme metadata.

    Returns
    -------
    GraphStyle
        Graph style used for the current render pass.
    """
    style = graph.graph_style
    background_color = str(style.background_color)
    if not _should_auto_contrast(graph, background_color):
        return style

    builtin_theme = _builtin_theme_for_render(graph)
    if builtin_theme is None:
        return style
    baseline_style = builtin_theme.graph_style
    replacement_fields: Dict[str, str] = {}
    if _colors_match(str(style.title_font_color), str(baseline_style.title_font_color)):
        replacement_fields["title_font_color"] = _AUTO_CONTRAST_TEXT_COLOR
    if _colors_match(str(style.edge_label_background), str(baseline_style.edge_label_background)):
        replacement_fields["edge_label_background"] = _blend_colors(
            background_color,
            str(baseline_style.edge_label_background),
            _AUTO_CONTRAST_LABEL_BACKGROUND_BLEND,
        )
    if not replacement_fields:
        return style
    return replace(style, **replacement_fields)


def density_aware_size_factor(
    node_count: int,
    layout_extent_pt: float,
    target_density: float = 0.5,
    min_factor: float = 0.25,
    max_factor: float = 1.5,
) -> float:
    """Return a node-size multiplier for graph-density-aware rendering.

    Parameters
    ----------
    node_count : int
        Number of visible nodes in the rendered graph.
    layout_extent_pt : float
        Maximum layout extent in points. Reserved for future area-aware
        calibration; the current graphviz parity fit depends on node count.
    target_density : float, default=0.5
        Target node-area/layout-area ratio reserved for future calibration.
    min_factor : float, default=0.25
        Lower bound for dense graphs.
    max_factor : float, default=1.5
        Upper bound for sparse graphs.

    Returns
    -------
    float
        Scalar applied to render-time node width and height.
    """

    if node_count <= _DENSITY_SHRINK_SPARSE_NODE_COUNT:
        return 1.0
    if node_count >= _DENSITY_SHRINK_CLAMP_NODE_COUNT:
        return max(float(min_factor), 0.0)
    _ = (layout_extent_pt, target_density)
    factor = math.sqrt(_DENSITY_SHRINK_REFERENCE_NODE_COUNT / max(float(node_count), 1.0))
    return max(float(min_factor), min(float(max_factor), factor))


def _layout_extent_pt(pos: np.ndarray) -> float:
    """Return the maximum node-center span for density calibration.

    Parameters
    ----------
    pos : numpy.ndarray
        Node positions with shape ``[N, 2]``.

    Returns
    -------
    float
        Maximum x/y extent in the same point-like data units used by Dagua's
        graphviz-strict gallery fixtures.
    """

    if pos.size == 0:
        return 0.0
    x_span = float(np.max(pos[:, 0]) - np.min(pos[:, 0]))
    y_span = float(np.max(pos[:, 1]) - np.min(pos[:, 1]))
    return max(x_span, y_span)


def _density_scaled_node_sizes(
    sizes: torch.Tensor,
    node_count: int,
    layout_extent_pt: float,
    enabled: bool,
) -> Tuple[torch.Tensor, float]:
    """Return render-time node sizes and the density multiplier used.

    Parameters
    ----------
    sizes : torch.Tensor
        Base node sizes with shape ``[N, 2]``.
    node_count : int
        Number of visible graph nodes.
    layout_extent_pt : float
        Maximum layout extent in point-like render units.
    enabled : bool
        Whether density-aware shrink is enabled for the graph style.

    Returns
    -------
    tuple[torch.Tensor, float]
        Scaled node sizes and the multiplier. When disabled, the original
        tensor is returned with factor ``1.0``.
    """

    if not enabled:
        return sizes, 1.0
    factor = density_aware_size_factor(node_count, layout_extent_pt)
    if abs(factor - 1.0) <= 1e-12:
        return sizes, 1.0
    return sizes * factor, factor


def _node_style_for_render(graph: Any, node_idx: int) -> NodeStyle:
    """Return a node style adapted for dark-background rendering.

    Parameters
    ----------
    graph : Any
        Graph exposing node-style lookup and theme metadata.
    node_idx : int
        Node index.

    Returns
    -------
    NodeStyle
        Render-local node style. The graph's stored style is not mutated.
    """
    style = graph.get_style_for_node(node_idx)
    background_color = str(graph.graph_style.background_color)
    if not _should_auto_contrast(graph, background_color):
        return style

    builtin_theme = _builtin_theme_for_render(graph)
    if builtin_theme is None:
        return style
    node_types = getattr(graph, "node_types", [])
    node_type = str(node_types[node_idx]) if node_idx < len(node_types) else "default"
    baseline_style = builtin_theme.get_node_style(node_type)
    replacement_fields: Dict[str, str] = {}
    if _colors_match(str(style.fill), str(baseline_style.fill)):
        replacement_fields["fill"] = _adapt_color_for_dark_bg(
            str(baseline_style.fill),
            _AUTO_CONTRAST_FILL_LUMINANCE,
        )
    if _colors_match(str(style.stroke), str(baseline_style.stroke)):
        replacement_fields["stroke"] = _adapt_color_for_dark_bg(
            str(baseline_style.stroke),
            _AUTO_CONTRAST_STROKE_LUMINANCE,
        )
    if _colors_match(str(style.font_color), str(baseline_style.font_color)):
        replacement_fields["font_color"] = _AUTO_CONTRAST_TEXT_COLOR
    if not replacement_fields:
        return style

    return replace(style, **replacement_fields)


def _edge_style_for_render(graph: Any, edge_idx: int) -> EdgeStyle:
    """Return an edge style adapted for dark-background rendering.

    Parameters
    ----------
    graph : Any
        Graph exposing edge-style lookup and theme metadata.
    edge_idx : int
        Edge index.

    Returns
    -------
    EdgeStyle
        Render-local edge style. The graph's stored style is not mutated.
    """
    style = graph.get_style_for_edge(edge_idx)
    if _is_graphviz_strict_render(graph):
        attrs = getattr(graph, "_graphviz_edge_attrs", {}).get(edge_idx, {})
        arrowsize = attrs.get("arrowsize") if isinstance(attrs, dict) else None
        if arrowsize is not None:
            try:
                style = replace(style, arrowsize=max(float(str(arrowsize).strip('"')), 0.0))
            except ValueError:
                pass
        style = replace(
            style,
            width=float(style.width) * _GRAPHVIZ_STRICT_EDGE_WIDTH_RENDER_MULTIPLIER,
            line_cap="butt",
            line_join="miter",
            opacity=1.0,
        )
    background_color = str(graph.graph_style.background_color)
    if not _should_auto_contrast(graph, background_color):
        return style

    builtin_theme = _builtin_theme_for_render(graph)
    if builtin_theme is None:
        return style
    baseline_style = builtin_theme.get_edge_style(_theme_edge_type(graph, edge_idx))

    replacement_fields: Dict[str, str] = {}
    if _colors_match(str(style.color), str(baseline_style.color)):
        replacement_fields["color"] = _adapt_color_for_dark_bg(
            str(baseline_style.color),
            _AUTO_CONTRAST_EDGE_LUMINANCE,
        )
    if _colors_match(str(style.arrow_color), str(baseline_style.arrow_color)):
        replacement_fields["arrow_color"] = _adapt_color_for_dark_bg(
            str(baseline_style.arrow_color),
            _AUTO_CONTRAST_EDGE_LUMINANCE,
        )
    if _colors_match(str(style.label_font_color), str(baseline_style.label_font_color)):
        replacement_fields["label_font_color"] = _AUTO_CONTRAST_TEXT_COLOR
    if _colors_match(str(style.label_background), str(baseline_style.label_background)):
        replacement_fields["label_background"] = _blend_colors(
            background_color,
            str(baseline_style.label_background),
            _AUTO_CONTRAST_LABEL_BACKGROUND_BLEND,
        )
    if not replacement_fields:
        return style
    return replace(style, **replacement_fields)


def _cluster_style_for_render(graph: Any, cluster_name: str) -> ClusterStyle:
    """Return a cluster style adapted for dark-background rendering.

    Parameters
    ----------
    graph : Any
        Graph exposing cluster-style lookup and theme metadata.
    cluster_name : str
        Cluster identifier.

    Returns
    -------
    ClusterStyle
        Render-local cluster style. The graph's stored style is not mutated.
    """
    style = graph.get_style_for_cluster(cluster_name)
    if _is_graphviz_strict_render(graph) and not str(getattr(style, "label_background", "")):
        style = replace(
            style,
            label_background="@background",
            label_background_opacity=1.0,
        )
    background_color = str(graph.graph_style.background_color)
    if not _should_auto_contrast(graph, background_color):
        return style

    builtin_theme = _builtin_theme_for_render(graph)
    if builtin_theme is None:
        return style
    baseline_style = builtin_theme.cluster_style
    replacement_fields: Dict[str, str] = {}
    if _colors_match(str(style.fill), str(baseline_style.fill)):
        replacement_fields["fill"] = _adapt_color_for_dark_bg(
            str(baseline_style.fill),
            _AUTO_CONTRAST_FILL_LUMINANCE,
        )
    if _colors_match(str(style.stroke), str(baseline_style.stroke)):
        replacement_fields["stroke"] = _adapt_color_for_dark_bg(
            str(baseline_style.stroke),
            _AUTO_CONTRAST_STROKE_LUMINANCE,
        )
    if _colors_match(str(style.font_color), str(baseline_style.font_color)):
        replacement_fields["font_color"] = _AUTO_CONTRAST_TEXT_COLOR
    if not replacement_fields:
        return style

    return replace(style, **replacement_fields)


def _strict_content_figsize(width_points: float, height_points: float) -> Tuple[float, float]:
    """Return Graphviz-native figure inches for a strict content bbox.

    Parameters
    ----------
    width_points : float
        Rendered content width in typographic points, including strict margin.
    height_points : float
        Rendered content height in typographic points, including strict margin.

    Returns
    -------
    tuple[float, float]
        Matplotlib figure size in inches using Graphviz's 72pt/in convention.
    """

    return (max(width_points, 1.0) / 72.0, max(height_points, 1.0) / 72.0)


def _new_figure_axes(
    figsize: Optional[Tuple[float, float]],
    backend: Optional[str] = None,
    dpi: Optional[float] = None,
) -> Tuple[Any, Any]:
    """Create a headless Matplotlib figure with an attached canvas.

    Parameters
    ----------
    figsize : tuple[float, float] | None
        Figure size in inches.
    backend : str, optional
        Render backend selector. ``None`` uses Dagua's auto-detected default.
    dpi : float, optional
        Figure DPI. When omitted, Matplotlib's figure default is used.

    Returns
    -------
    tuple[Any, Any]
        Matplotlib ``(figure, axes)`` pair.
    """
    from matplotlib.figure import Figure

    from dagua.render._backend import _resolve_backend

    fig = Figure(figsize=figsize, dpi=dpi)
    canvas_cls, resolved_backend = _resolve_backend(backend)
    canvas_cls(fig)
    ax = fig.add_subplot(1, 1, 1)
    setattr(ax, "_dagua_backend_name", resolved_backend)
    return fig, ax


def _resolve_render_dpi(dpi: Optional[int], style: GraphStyle) -> int:
    """Return the raster DPI for the current render pass.

    Parameters
    ----------
    dpi : int or None
        Explicit DPI passed to ``render``.
    style : GraphStyle
        Graph-level style carrying graphviz-compatible defaults.

    Returns
    -------
    int
        Positive DPI value used for the Matplotlib figure and raster save.
    """
    resolved = int(dpi if dpi is not None else style.dpi)
    if resolved <= 0:
        raise ValueError("dpi must be a positive integer.")
    return resolved


def _graphviz_canvas_bounds_and_figsize(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    style: GraphStyle,
) -> Tuple[float, float, float, float, Tuple[float, float], str]:
    """Apply graphviz canvas attributes to content bounds.

    Parameters
    ----------
    x_min : float
        Minimum content x coordinate in data/point units.
    x_max : float
        Maximum content x coordinate in data/point units.
    y_min : float
        Minimum content y coordinate in data/point units.
    y_max : float
        Maximum content y coordinate in data/point units.
    style : GraphStyle
        Graph style containing graphviz canvas fields.

    Returns
    -------
    tuple[float, float, float, float, tuple[float, float], str]
        Axis bounds, figure size in inches, and Matplotlib aspect mode.

    Notes
    -----
    Dagua layout geometry is stored in graphviz-compatible points for this
    renderer. Canvas fields are authored in inches, so 72 pt/in is the only
    conversion needed at the render boundary.
    """
    pad_points = max(float(style.pad_inches), 0.0) * 72.0
    margin_points = max(float(style.margin_inches), 0.0) * 72.0

    content_x_min = float(x_min) - pad_points
    content_x_max = float(x_max) + pad_points
    content_y_min = float(y_min) - pad_points
    content_y_max = float(y_max) + pad_points
    content_width = max(content_x_max - content_x_min, 1.0)
    content_height = max(content_y_max - content_y_min, 1.0)

    figure_width = (content_width + 2.0 * margin_points) / 72.0
    figure_height = (content_height + 2.0 * margin_points) / 72.0
    axis_width = content_width + 2.0 * margin_points
    axis_height = content_height + 2.0 * margin_points
    aspect_mode = "auto"

    size_inches = style.size_inches
    ratio = None if style.ratio is None else str(style.ratio).strip().lower()
    if size_inches is not None:
        target_width = max(float(size_inches[0]), 1.0 / 72.0)
        target_height = max(float(size_inches[1]), 1.0 / 72.0)
        if style.size_force_fit or ratio == "fill":
            figure_width = target_width
            figure_height = target_height
            aspect_mode = "auto"
        elif ratio == "expand" and figure_width < target_width and figure_height < target_height:
            scale = min(target_width / figure_width, target_height / figure_height)
            figure_width *= scale
            figure_height *= scale
            axis_width *= scale
            axis_height *= scale
        elif ratio in {"compress", "auto"}:
            scale = min(target_width / figure_width, target_height / figure_height, 1.0)
            figure_width *= scale
            figure_height *= scale
        else:
            scale = min(target_width / figure_width, target_height / figure_height, 1.0)
            figure_width *= scale
            figure_height *= scale

    center_x = (content_x_min + content_x_max) / 2.0
    center_y = (content_y_min + content_y_max) / 2.0
    x_min_out = center_x - axis_width / 2.0
    x_max_out = center_x + axis_width / 2.0
    y_min_out = center_y - axis_height / 2.0
    y_max_out = center_y + axis_height / 2.0
    return x_min_out, x_max_out, y_min_out, y_max_out, (figure_width, figure_height), aspect_mode


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
    tight_bbox = bool(getattr(fig, "_dagua_tight_bbox", True))

    common = {
        "facecolor": bg,
        "edgecolor": bg,
        "transparent": False,
    }
    fig.patch.set_facecolor(bg)
    fig.patch.set_alpha(1.0)
    if tight_bbox:
        common.update({"bbox_inches": "tight", "pad_inches": 0.05})

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


def _validate_bit_equivalent_output(
    output: Optional[Union[str, Path]],
    format: Optional[str],
) -> None:
    """Validate the destination for bit-equivalent rasterization.

    Parameters
    ----------
    output : str or pathlib.Path, optional
        Output path supplied to ``render``.
    format : str, optional
        Explicit output format override.

    Returns
    -------
    None
        Raises if the opt-in raster path cannot write a PNG destination.

    Raises
    ------
    ValueError
        If no output path is supplied or the requested output format is not PNG.
    """
    if output is None:
        raise ValueError("bit_equivalent=True requires a PNG output path.")
    fmt = _detect_output_format(str(output), format)
    if fmt != "png":
        raise ValueError("bit_equivalent=True only supports PNG output.")


def _figure_to_svg_bytes(fig: Any, bg: str) -> bytes:
    """Serialize a Matplotlib figure to SVG bytes.

    Parameters
    ----------
    fig : Any
        Matplotlib figure containing the fully drawn graph.
    bg : str
        Figure background color.

    Returns
    -------
    bytes
        SVG document bytes suitable for cairosvg rasterization.
    """
    tight_bbox = bool(getattr(fig, "_dagua_tight_bbox", True))
    common: Dict[str, Any] = {
        "facecolor": bg,
        "edgecolor": bg,
        "transparent": False,
    }
    if tight_bbox:
        common.update({"bbox_inches": "tight", "pad_inches": 0.05})

    fig.patch.set_facecolor(bg)
    fig.patch.set_alpha(1.0)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="svg", **common)
    return buffer.getvalue()


def _save_bit_equivalent_png(fig: Any, output: Union[str, Path], bg: str, dpi: int) -> None:
    """Save a figure through SVG serialization followed by cairosvg PNG output.

    Parameters
    ----------
    fig : Any
        Matplotlib figure containing the fully drawn graph.
    output : str or pathlib.Path
        Destination PNG path.
    bg : str
        Figure background color.
    dpi : int
        cairosvg rasterization DPI.

    Returns
    -------
    None
        Writes the rasterized PNG to ``output``.
    """
    svg_bytes = _figure_to_svg_bytes(fig, bg)
    _render_via_cairosvg(svg_bytes, output, dpi=dpi)


def _expand_bounds_for_external_labels(
    graph: Any,
    pos: np.ndarray,
    sizes: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> Tuple[float, float, float, float]:
    """Expand render bounds to keep external node labels visible.

    Parameters
    ----------
    graph : Any
        Graph exposing node styles.
    pos : numpy.ndarray
        Node positions with shape ``[N, 2]``.
    sizes : numpy.ndarray
        Node sizes with shape ``[N, 2]``.
    x_min : float
        Current minimum x bound.
    x_max : float
        Current maximum x bound.
    y_min : float
        Current minimum y bound.
    y_max : float
        Current maximum y bound.

    Returns
    -------
    tuple[float, float, float, float]
        Expanded ``(x_min, x_max, y_min, y_max)`` bounds.
    """
    for i in range(graph.num_nodes):
        style = _node_style_for_render(graph, i)
        external_label = str(getattr(style, "external_label", ""))
        if external_label.strip() == "":
            continue

        label_size_data = _node_relative_font_size_data(
            external_label,
            float(sizes[i, 1]),
            float(style.external_label_font_size),
            _DEFAULT_EXTERNAL_LABEL_FONT_POINTS,
            font_weight=style.font_weight,
        )
        label_width, label_height = measure_text_data(
            external_label,
            font_family=_text_font_family(style),
            font_weight=_normalize_text_font_weight(style.font_weight),
            font_style=style.font_style,
            size_data=label_size_data,
        )
        if label_width <= 0.0 and label_height <= 0.0:
            continue

        cx = float(pos[i, 0])
        cy = float(pos[i, 1])
        half_width = float(sizes[i, 0]) / 2.0
        half_height = float(sizes[i, 1]) / 2.0
        offset = float(style.external_label_offset)
        position = _normalize_external_label_position(style.external_label_position)

        text_x, text_y, ha, va = _external_label_anchor(
            cx, cy, half_width, half_height, offset, position
        )
        label_x_min = text_x - label_width if ha == "right" else text_x
        if ha == "center":
            label_x_min = text_x - label_width / 2.0
        label_x_max = text_x + label_width if ha == "left" else text_x
        if ha == "center":
            label_x_max = text_x + label_width / 2.0
        label_y_min = text_y - label_height if va == "top" else text_y
        if va == "center":
            label_y_min = text_y - label_height / 2.0
        label_y_max = text_y + label_height if va == "bottom" else text_y
        if va == "center":
            label_y_max = text_y + label_height / 2.0
        x_min = min(x_min, label_x_min)
        x_max = max(x_max, label_x_max)
        y_min = min(y_min, label_y_min)
        y_max = max(y_max, label_y_max)

    return x_min, x_max, y_min, y_max


def _coerce_canvas_fit_margin(fit_to_canvas: Union[bool, float]) -> Optional[float]:
    """Return the canvas-fit margin fraction requested by the caller.

    Parameters
    ----------
    fit_to_canvas : bool or float
        ``False`` disables canvas fitting, ``True`` uses the default 5 percent
        margin on each side, and a float supplies the explicit margin fraction.

    Returns
    -------
    float or None
        Margin fraction when canvas fitting is enabled, otherwise ``None``.

    Raises
    ------
    ValueError
        Raised when an explicit margin is outside ``[0.0, 0.5)``.
    TypeError
        Raised when ``fit_to_canvas`` is not a bool or number.
    """
    if fit_to_canvas is False:
        return None
    if fit_to_canvas is True:
        return 0.02
    if isinstance(fit_to_canvas, (int, float)) and not isinstance(fit_to_canvas, bool):
        margin = float(fit_to_canvas)
        if not 0.0 <= margin < 0.5:
            raise ValueError("fit_to_canvas margin must be in the range [0.0, 0.5).")
        return margin
    raise TypeError("fit_to_canvas must be a bool or a margin fraction float.")


def _canvas_fit_bounds(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    figsize: Tuple[float, float],
    margin_fraction: float,
) -> Tuple[float, float, float, float]:
    """Expand bounds so data-to-display scaling is uniform and canvas-filling.

    Parameters
    ----------
    x_min : float
        Minimum content x coordinate in data units.
    x_max : float
        Maximum content x coordinate in data units.
    y_min : float
        Minimum content y coordinate in data units.
    y_max : float
        Maximum content y coordinate in data units.
    figsize : tuple[float, float]
        Target figure size in inches.
    margin_fraction : float
        Fraction of each canvas dimension reserved as outer padding.

    Returns
    -------
    tuple[float, float, float, float]
        Aspect-adjusted axes bounds as ``(x_min, x_max, y_min, y_max)``.

    Notes
    -----
    Matplotlib fills the axes independently in x and y. Expanding the shorter
    data dimension to match the target figure aspect keeps the resulting
    data-to-pixel scale uniform while still letting the layout fill the panel.
    """
    content_width = max(float(x_max) - float(x_min), 1.0)
    content_height = max(float(y_max) - float(y_min), 1.0)
    inner_fraction = max(1.0 - (2.0 * margin_fraction), 1.0e-6)
    target_aspect = max(float(figsize[0]), 1.0e-6) / max(float(figsize[1]), 1.0e-6)
    content_aspect = content_width / content_height
    if content_aspect < target_aspect:
        content_width = content_height * target_aspect
    elif content_aspect > target_aspect:
        content_height = content_width / target_aspect

    desired_width = content_width / inner_fraction
    desired_height = content_height / inner_fraction

    x_center = (float(x_min) + float(x_max)) / 2.0
    y_center = (float(y_min) + float(y_max)) / 2.0
    return (
        x_center - desired_width / 2.0,
        x_center + desired_width / 2.0,
        y_center - desired_height / 2.0,
        y_center + desired_height / 2.0,
    )


def render(
    graph: Any,
    positions: Any = None,
    config: Any = None,
    output: Optional[Union[str, Path]] = None,
    format: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    show: bool = False,
    title: Optional[str] = None,
    curves: Optional[List[BezierCurve]] = None,
    label_positions: Optional[List[Optional[Tuple[float, float]]]] = None,
    svg_hover_text: bool = True,
    backend: Optional[str] = None,
    fit_to_canvas: Union[bool, float] = False,
    bit_equivalent: bool = False,
) -> Tuple[Any, Any]:
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
    output : str or pathlib.Path, optional
        Output file path.
    format : str, optional
        Explicit output format override. When omitted, the renderer infers the
        format from ``output``.
    figsize : tuple[float, float], optional
        Figure size in inches.
    dpi : int, optional
        Raster output resolution. When omitted, ``graph.graph_style.dpi`` is
        used; the default ``GraphStyle`` value matches graphviz's 96 DPI PNG
        output.
    show : bool, default=False
        Whether to request display of the returned figure.
    title : str, optional
        Figure title.
    curves : list[BezierCurve], optional
        Pre-routed bezier curves for edges.
    label_positions : list[tuple[float, float] | None], optional
        Pre-computed positions for edge labels.
    svg_hover_text : bool, default=True
        Whether to embed hover tooltips in SVG outputs.
    backend : str, optional
        Matplotlib canvas backend: ``"agg"``, ``"cairo"``, or ``None`` for the
        auto-detected/global default.
    fit_to_canvas : bool or float, default=False
        Explicit fixed-panel override. Use this for a Jupyter cell, dashboard,
        or gallery panel where the layout should fill an existing canvas.
        ``True`` uses Dagua's default panel margin; a float supplies an explicit
        margin fraction. When truthy, this ignores graphviz natural-canvas
        fields such as ``margin_inches``, ``size_inches``, and ``ratio``.
    bit_equivalent : bool, default=False
        Opt-in PNG path for pixel-parity use cases. When True, Dagua renders
        the graph to SVG using the existing Matplotlib drawing stack, then
        rasterizes the SVG bytes through cairosvg. This requires
        ``pip install 'dagua[bit_equivalent]'`` and is intentionally not the
        default rendering path.

    Returns
    -------
    tuple[Any, Any]
        Matplotlib ``(figure, axes)``.
    """
    import warnings

    import matplotlib

    matplotlib.use("Agg")

    gs = _graph_style_for_render(graph)
    render_dpi = _resolve_render_dpi(dpi, gs)
    canvas_fit_margin = _coerce_canvas_fit_margin(fit_to_canvas)
    if bit_equivalent:
        _validate_bit_equivalent_output(output, format)

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
        fig, ax = _new_figure_axes(figsize or (6, 4), backend=backend, dpi=render_dpi)
        fig.patch.set_facecolor(bg)
        fig.patch.set_alpha(1.0)
        if output:
            if bit_equivalent:
                _save_bit_equivalent_png(fig, output, bg, dpi=render_dpi)
            else:
                _save_figure(fig, str(output), bg, dpi=render_dpi, format=format)
        return fig, ax

    # Compute figure bounds
    graph.compute_node_sizes()
    original_node_sizes = graph.node_sizes
    scaled_node_sizes, density_size_factor = _density_scaled_node_sizes(
        original_node_sizes,
        n,
        _layout_extent_pt(pos),
        bool(getattr(gs, "density_aware_node_shrink", False)),
    )
    if density_size_factor != 1.0:
        # Route clipping, cluster bboxes, labels, and node fills all read
        # graph.node_sizes; use the density-adjusted tensor consistently for
        # this render pass so dense graphviz-strict panels shrink as a unit.
        graph.node_sizes = scaled_node_sizes
    sizes = scaled_node_sizes.detach().cpu().numpy()

    margin = gs.margin if canvas_fit_margin is not None else 0.0
    x_min = (pos[:, 0] - sizes[:, 0] / 2).min() - margin
    x_max = (pos[:, 0] + sizes[:, 0] / 2).max() + margin
    y_min = (pos[:, 1] - sizes[:, 1] / 2).min() - margin
    y_max = (pos[:, 1] + sizes[:, 1] / 2).max() + margin
    x_min, x_max, y_min, y_max = _expand_bounds_for_external_labels(
        graph,
        pos,
        sizes,
        float(x_min),
        float(x_max),
        float(y_min),
        float(y_max),
    )

    content_y_max = float(y_max)

    # Expand figure bounds for cluster headers and minimum width.
    # Cluster rendering adds header space above y_max and may expand x_min/x_max
    # for minimum width. Account for this so labels are not clipped.
    if graph.clusters:
        ordered_clusters = _cluster_render_order(graph)
        cluster_depths = _cluster_depths(graph, ordered_clusters)
        cluster_y_maxes = _compute_cluster_y_maxes(
            graph,
            pos,
            sizes,
            ordered_clusters,
            cluster_depths,
        )
        cluster_y_mins = _compute_cluster_y_mins(
            graph,
            pos,
            sizes,
            ordered_clusters,
            cluster_depths,
        )
        for cname in graph.clusters:
            cstyle = _cluster_style_for_render(graph, cname)
            cindices = graph.leaf_cluster_members(cname)
            if not cindices:
                continue
            ci = np.array(cindices)
            cp = pos[ci]
            cs = sizes[ci]
            cpad = cstyle.padding
            cy_min = cluster_y_mins.get(cname, (cp[:, 1] - cs[:, 1] / 2).min() - cpad)
            cy_max = cluster_y_maxes.get(cname, (cp[:, 1] + cs[:, 1] / 2).max() + cpad)
            cx_min = (cp[:, 0] - cs[:, 0] / 2).min() - cpad
            cx_max = (cp[:, 0] + cs[:, 0] / 2).max() + cpad
            # Minimum width
            ch = cy_max - cy_min
            min_cw = ch * 0.8
            cw = cx_max - cx_min
            if cw < min_cw:
                expand_cw = (min_cw - cw) / 2.0
                cx_min -= expand_cw
                cx_max += expand_cw
            x_min = min(x_min, cx_min - margin)
            x_max = max(x_max, cx_max + margin)
            y_min = min(y_min, cy_min - margin)
            y_max = max(y_max, cy_max + margin)
            if _cluster_label_is_outside(str(cstyle.label_position)):
                label_font_data = _cluster_font_size_data(
                    graph.cluster_labels.get(cname, cname),
                    float(ch),
                    float(sizes[:, 1].min()) if sizes.size else 0.0,
                    float(
                        cstyle.font_size_override_points
                        if cstyle.font_size_override_points is not None
                        else cstyle.font_size
                    ),
                    str(cstyle.font_size_scaling),
                )
                label_width, label_height = _measure_cluster_label_data(
                    graph.cluster_labels.get(cname, cname),
                    font_size_data=label_font_data,
                    font_family=str(cstyle.font_family or RESOLVED_FONT),
                    font_weight=str(cstyle.font_weight),
                    text_wrap=str(cstyle.text_wrap),
                    text_max_width=_cluster_label_text_max_width(cstyle, 1.0),
                )
                label_x, label_y, label_ha, label_va = _cluster_label_anchor(
                    str(cstyle.label_position),
                    float(cx_min),
                    float(cx_max),
                    float(cy_min),
                    float(cy_max),
                    float(cstyle.label_offset[0]),
                    float(cstyle.label_offset[1]),
                )
                label_bounds = _cluster_label_bounds(
                    DaguaText(x=label_x, y=label_y, text="", ha=label_ha, va=label_va),
                    label_width,
                    label_height,
                )
                x_min = min(x_min, label_bounds[0] - margin)
                x_max = max(x_max, label_bounds[2] + margin)
                y_min = min(y_min, label_bounds[1] - margin)
                y_max = max(y_max, label_bounds[3] + margin)
        content_y_max = max(content_y_max, float(y_max))

    # Expand figure bounds for self-loop arcs that extend beyond nodes.
    edge_index = graph.edge_index.detach().cpu().numpy()
    direction = getattr(graph, "direction", "TB")
    for e_idx in range(edge_index.shape[1]):
        src, tgt = int(edge_index[0, e_idx]), int(edge_index[1, e_idx])
        if src == tgt:
            sx, sy = float(pos[src, 0]), float(pos[src, 1])
            sw, sh = float(sizes[src, 0]), float(sizes[src, 1])
            loop_size = max(sw, sh)
            loop_w = loop_size * 0.35 * 1.33  # spread * cp_factor
            loop_h = loop_size * 1.6  # arc_height
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

    content_y_max = max(content_y_max, float(y_max))
    width = x_max - x_min
    height = y_max - y_min

    title_band_height = 0.0
    if title:
        title_band_height = _title_font_size_data(height, float(gs.title_font_size))
        y_max += title_band_height * _TITLE_BAND_HEIGHT_MULTIPLIER
        height = y_max - y_min

    natural_aspect_mode = "equal"
    if canvas_fit_margin is None:
        x_min, x_max, y_min, y_max, natural_figsize, natural_aspect_mode = (
            _graphviz_canvas_bounds_and_figsize(
                float(x_min),
                float(x_max),
                float(y_min),
                float(y_max),
                gs,
            )
        )
        if figsize is None:
            figsize = natural_figsize
    elif figsize is None:
        max_w, max_h = gs.max_figsize
        min_w, min_h = gs.min_figsize
        scale = max(1.0, min(width / 100, max_w))
        aspect = height / max(width, 1)
        fig_w = min(max(scale, min_w), max_w)
        fig_h = min(max(fig_w * aspect, min_h), max_h)
        figsize = (fig_w, fig_h)

    if canvas_fit_margin is not None:
        x_min, x_max, y_min, y_max = _canvas_fit_bounds(
            float(x_min),
            float(x_max),
            float(y_min),
            float(y_max),
            figsize,
            canvas_fit_margin,
        )

    figure_dpi = render_dpi if canvas_fit_margin is None else None
    fig, ax = _new_figure_axes(figsize, backend=backend, dpi=figure_dpi)
    fig.patch.set_facecolor(bg)
    fig.patch.set_alpha(1.0)
    ax.set_facecolor(bg)
    setattr(fig, "_dagua_svg_hover_map", {} if svg_hover_text else None)
    setattr(fig, "_dagua_tight_bbox", False)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    svg_hover_map = getattr(fig, "_dagua_svg_hover_map")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    aspect_mode = "auto" if canvas_fit_margin is not None else natural_aspect_mode
    ax.set_aspect(aspect_mode)
    ax.axis("off")
    cluster_aware = bool(getattr(config, "cluster_aware", True))
    cluster_axis_margin = margin
    if canvas_fit_margin is None:
        cluster_axis_margin = max(cluster_axis_margin, float(gs.margin_inches) * 72.0)
    _expand_axes_for_clusters(
        ax,
        graph,
        pos,
        sizes,
        cluster_axis_margin,
        cluster_aware=cluster_aware,
    )
    if canvas_fit_margin is not None:
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x_min, x_max, y_min, y_max = _canvas_fit_bounds(
            float(x_min),
            float(x_max),
            float(y_min),
            float(y_max),
            figsize,
            canvas_fit_margin,
        )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    # --- Layer 0: Cluster backgrounds ---
    _draw_clusters(ax, graph, pos, sizes, cluster_aware=cluster_aware, svg_hover_map=svg_hover_map)

    # --- Layer 1: Edges ---
    if curves is None:
        curves = route_edges(positions, graph.edge_index, graph.node_sizes, graph.direction, graph)
    curves = _graphviz_strict_back_edge_curves(ax, graph, curves)
    curves = _graphviz_strict_reclip_edge_terminals(graph, curves, pos)
    cluster_membership: Dict[int, List[str]] = {}
    cluster_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    if cluster_aware:
        cluster_membership, cluster_bboxes = _render_cluster_edge_clip_data(
            ax,
            graph,
            pos,
            sizes,
            cluster_aware=cluster_aware,
        )
    edge_collection = _draw_edges(
        ax,
        graph,
        curves,
        positions=pos,
        svg_hover_map=svg_hover_map,
        cluster_membership=cluster_membership,
        cluster_bboxes=cluster_bboxes,
    )

    # --- Layer 2: Nodes ---
    clip_patches = _draw_nodes(ax, graph, pos, sizes, svg_hover_map=svg_hover_map)

    # --- Layer 2.5: Port indicators ---
    _draw_port_indicators(ax, graph, curves)

    # --- Layer 3: Node labels ---
    _draw_node_labels(
        ax,
        graph,
        pos,
        sizes,
        clip_patches,
        svg_hover_map=svg_hover_map,
        density_size_factor=density_size_factor,
    )

    # --- Layer 3.5: External node labels ---
    _draw_external_labels(ax, graph, pos, sizes, svg_hover_map=svg_hover_map)

    # --- Layer 4: Edge labels ---
    _draw_edge_labels(
        ax,
        graph,
        curves,
        label_positions=label_positions,
        svg_hover_map=svg_hover_map,
        sizes=sizes,
        edge_collection=edge_collection,
    )

    if title:
        display_scale = _compute_display_scale(ax)
        title_ff = str(gs.title_font_family or RESOLVED_FONT)
        render_text(
            ax,
            [
                DaguaText(
                    x=(float(x_min) + float(x_max)) / 2.0,
                    y=float(content_y_max) + title_band_height * 1.25,
                    text=title,
                    font_size=_effective_font_size_points(title_band_height, display_scale),
                    font_family=title_ff,
                    font_weight=str(gs.title_font_weight),
                    font_color=str(gs.title_font_color),
                    ha="center",
                    va="center",
                    clip_on=False,
                    zorder=5.0,
                    gid="dagua-graph-title",
                )
            ],
            display_scale,
            svg_hover_map,
        )

    if (
        canvas_fit_margin is None
        and figsize != natural_figsize
        and not _is_graphviz_strict_render(graph)
    ):
        fig.tight_layout()

    if output:
        if bit_equivalent:
            _save_bit_equivalent_png(fig, output, bg, dpi=render_dpi)
        else:
            _save_figure(fig, str(output), bg, dpi=render_dpi, format=format)

    if show:
        fig.show()

    if graph.node_sizes is not original_node_sizes:
        graph.node_sizes = original_node_sizes

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
    """Inject SVG title elements for hover text.

    Parameters
    ----------
    output : str
        SVG or SVGZ path to update in place.
    svg_hover_map : Any
        Mapping from Matplotlib artist ids to hover text.
    compressed : bool, default=False
        Whether ``output`` is gzip-compressed SVG.

    Returns
    -------
    None
        Mutates the SVG file in place.
    """
    if compressed:
        with gzip.open(output, "rt", encoding="utf-8") as f:
            svg_text = f.read()
    else:
        svg_text = Path(output).read_text(encoding="utf-8")

    ET.register_namespace("", "http://www.w3.org/2000/svg")
    root = ET.fromstring(svg_text)
    title_tag = "{http://www.w3.org/2000/svg}title"
    inserted_count = 0
    for elem in root.iter():
        gid = elem.attrib.get("id")
        if gid and gid in svg_hover_map:
            title = elem.find(title_tag)
            if title is None:
                title = ET.Element(title_tag)
                elem.insert(0, title)
            title.text = svg_hover_map[gid]
            inserted_count += 1

    if inserted_count == 0:
        fallback_text = next(
            (text for gid, text in svg_hover_map.items() if str(gid).startswith("dagua-node-")),
            None,
        )
        if fallback_text:
            title = ET.Element(title_tag)
            title.text = fallback_text
            root.insert(0, title)

    svg_text = ET.tostring(root, encoding="unicode")
    if compressed:
        with gzip.open(output, "wt", encoding="utf-8") as f:
            f.write(svg_text)
    else:
        Path(output).write_text(svg_text, encoding="utf-8")


def _mpl_capstyle(cap: str) -> str:
    """Convert a Dagua cap style name to the matplotlib equivalent.

    Parameters
    ----------
    cap : str
        Dagua cap style: ``"butt"``, ``"round"``, or ``"square"``.

    Returns
    -------
    str
        Matplotlib cap style (``"square"`` maps to ``"projecting"``).
    """
    if cap == "square":
        return "projecting"
    return cap


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
    if getattr(style, "line_dash_pattern", None) is not None:
        return (0, tuple(float(value) for value in style.line_dash_pattern))
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
    from matplotlib.patches import Circle, Ellipse, PathPatch, Polygon
    from matplotlib.path import Path

    shape = style.shape
    if shape in {"roundrect", "rect", "arrow"}:
        return PathPatch(
            build_shape_path(
                ShapeSpec(
                    center_x=x,
                    center_y=y,
                    width=w,
                    height=h,
                    shape=shape,
                    corner_radius=(
                        getattr(style, "corner_radius", 0.0) if shape == "roundrect" else 0.0
                    ),
                    aspect_ratio=getattr(style, "aspect_ratio", None),
                )
            ),
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder,
        )
    if shape in {
        "round_triangle",
        "round_diamond",
        "round_pentagon",
        "round_hexagon",
        "round_octagon",
        "house",
        "invhouse",
        "invtrapezium",
        "folder",
        "tab",
        "component",
        "note",
        "Msquare",
        "Mdiamond",
        "Mcircle",
        "doubleoctagon",
        "tripleoctagon",
        "promoter",
        "cds",
        "terminator",
        "ribosite",
        "proteasesite",
        "rpromoter",
        "rarrow",
        "larrow",
        "assembly",
        "insulator",
        "signature",
    }:
        return PathPatch(
            build_shape_path(
                ShapeSpec(
                    center_x=x,
                    center_y=y,
                    width=w,
                    height=h,
                    shape=shape,
                    corner_radius=0.0,
                    aspect_ratio=getattr(style, "aspect_ratio", None),
                )
            ),
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
    if shape in {
        "semicircle",
        "semicircle_up",
        "semicircle_down",
        "semicircle_left",
        "semicircle_right",
    }:
        return PathPatch(
            build_shape_path(
                ShapeSpec(
                    center_x=x,
                    center_y=y,
                    width=w,
                    height=h,
                    shape=shape,
                    corner_radius=0.0,
                    aspect_ratio=getattr(style, "aspect_ratio", None),
                )
            ),
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
    if shape in {"triangle", "round_triangle"}:
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
        cylinder_vertices: List[Tuple[float, float]] = [
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
            Path(cylinder_vertices, codes),
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder,
        )
    return PathPatch(
        build_shape_path(
            ShapeSpec(
                center_x=x,
                center_y=y,
                width=w,
                height=h,
                shape="roundrect",
                corner_radius=6.0,
            )
        ),
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )


def _border_ribbon_paths(
    shape_spec: ShapeSpec,
    outer_path: Any,
    border_width: float,
    border_position: str,
    dash_pattern: Any,
) -> List[Any]:
    """Build filled border paths for one shape outline.

    Parameters
    ----------
    shape_spec : ShapeSpec
        Shape geometry in data coordinates.
    outer_path : Any
        Outer shape path in data coordinates.
    border_width : float
        Border width in data coordinates.
    border_position : str
        Border placement mode.
    dash_pattern : Any
        ``"solid"`` for a continuous annular ring, or a dash pattern accepted
        by :func:`dash_ribbon_paths`.

    Returns
    -------
    list[Any]
        Filled paths representing the border body in data coordinates.
    """
    if border_width <= 0.0:
        return []
    if dash_pattern == "solid":
        border_outer_path, border_inner_path = _solid_border_ring_paths(
            shape_spec,
            outer_path,
            border_width,
            border_position,
        )
        return [annular_path(border_outer_path, border_inner_path)]
    centerline_path = _node_border_centerline_path(
        shape_spec,
        outer_path,
        border_width,
        border_position,
    )
    return dash_ribbon_paths(centerline_path, dash_pattern, border_width)


def _draw_border_ribbon(
    ax: Any,
    shape_spec: ShapeSpec,
    outer_path: Any,
    border_width: float,
    border_position: str,
    dash_pattern: Any,
    color: Any,
    zorder: float,
) -> None:
    """Draw one shape border as a filled data-coordinate ribbon.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    shape_spec : ShapeSpec
        Shape geometry in data coordinates.
    outer_path : Any
        Outer shape path in data coordinates.
    border_width : float
        Border width in data coordinates.
    border_position : str
        Border placement mode.
    dash_pattern : Any
        ``"solid"`` or a dash pattern accepted by :func:`dash_ribbon_paths`.
    color : Any
        Matplotlib-compatible face color for the border body.
    zorder : float
        Artist z-order for the border collection.
    """
    border_paths = _border_ribbon_paths(
        shape_spec,
        outer_path,
        border_width,
        border_position,
        dash_pattern,
    )
    add_filled_collections(
        ax=ax,
        fill_paths=[],
        fill_colors=[],
        border_paths=border_paths,
        border_colors=[color] * len(border_paths),
        fill_zorder=zorder,
        border_zorder=zorder,
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
    graphviz_detail_paths: List[Any] = []
    from matplotlib.path import Path as MplPath

    if style.shape == "tab":
        left = x - w / 2.0
        top = y + h / 2.0
        tab_width = min(GRAPHVIZ_TAB_WIDTH, w)
        graphviz_detail_paths.append(
            MplPath(np.array([[left, top], [left + tab_width, top]], dtype=np.float64))
        )
    elif style.shape == "component":
        left = x - w / 2.0
        bottom = y - h / 2.0
        top = y + h / 2.0
        detail = min(GRAPHVIZ_COMPONENT_DETAIL, h / 8.0, w / 4.0)
        graphviz_detail_paths.extend(
            [
                MplPath(
                    np.array(
                        [
                            [left, top - detail],
                            [left + detail, top - detail],
                            [left + detail, top - 2.0 * detail],
                            [left, top - 2.0 * detail],
                        ],
                        dtype=np.float64,
                    )
                ),
                MplPath(
                    np.array(
                        [
                            [left, bottom + 2.0 * detail],
                            [left + detail, bottom + 2.0 * detail],
                            [left + detail, bottom + detail],
                            [left, bottom + detail],
                        ],
                        dtype=np.float64,
                    )
                ),
            ]
        )
    elif style.shape == "note":
        right = x + w / 2.0
        top = y + h / 2.0
        fold = min(GRAPHVIZ_NOTE_FOLD_SIZE, w / 2.0, h / 2.0)
        graphviz_detail_paths.append(
            MplPath(
                np.array(
                    [
                        [right - fold, top],
                        [right - fold, top - fold],
                        [right, top - fold],
                    ],
                    dtype=np.float64,
                )
            )
        )
    elif style.shape == "Msquare":
        side = max(w, h)
        half_side = side / 2.0
        left = x - half_side
        right = x + half_side
        bottom = y - half_side
        top = y + half_side
        mark = min(GRAPHVIZ_M_CORNER_MARK_LENGTH, side / 4.0)
        mark_vertices = (
            ((left + mark, top), (left, top - mark)),
            ((left, bottom + mark), (left + mark, bottom)),
            ((right - mark, bottom), (right, bottom + mark)),
            ((right, top - mark), (right - mark, top)),
        )
        graphviz_detail_paths.extend(
            MplPath(np.asarray(vertices, dtype=np.float64)) for vertices in mark_vertices
        )
    elif style.shape == "Mdiamond":
        corners = np.array(
            [
                [x, y + h / 2.0],
                [x + w / 2.0, y],
                [x, y - h / 2.0],
                [x - w / 2.0, y],
            ],
            dtype=np.float64,
        )
        for index, corner in enumerate(corners):
            previous_corner = corners[(index - 1) % len(corners)]
            next_corner = corners[(index + 1) % len(corners)]
            previous_direction = previous_corner - corner
            next_direction = next_corner - corner
            previous_length = max(float(np.linalg.norm(previous_direction)), 1.0)
            next_length = max(float(np.linalg.norm(next_direction)), 1.0)
            mark_length = min(
                GRAPHVIZ_M_CORNER_MARK_LENGTH,
                previous_length / 2.0,
                next_length / 2.0,
            )
            graphviz_detail_paths.append(
                MplPath(
                    np.vstack(
                        [
                            corner + previous_direction * (mark_length / previous_length),
                            corner + next_direction * (mark_length / next_length),
                        ]
                    )
                )
            )
    elif style.shape == "Mcircle":
        radius = max(w, h) / 2.0
        chord_y = radius * GRAPHVIZ_M_CIRCLE_CHORD_RATIO
        chord_x = radius * math.sqrt(1.0 - GRAPHVIZ_M_CIRCLE_CHORD_RATIO**2)
        graphviz_detail_paths.extend(
            [
                MplPath(np.array([[x - chord_x, y + chord_y], [x + chord_x, y + chord_y]])),
                MplPath(np.array([[x - chord_x, y - chord_y], [x + chord_x, y - chord_y]])),
            ]
        )
    elif style.shape in {"doubleoctagon", "tripleoctagon"}:
        inner_ring_count = 1 if style.shape == "doubleoctagon" else 2
        for ring_index in range(inner_ring_count):
            graphviz_detail_paths.append(
                graphviz_octagon_path(
                    ShapeSpec(center_x=x, center_y=y, width=w, height=h, shape="octagon"),
                    offset=GRAPHVIZ_OCTAGON_PERIPHERY_GAP * ring_index,
                )
            )

    if graphviz_detail_paths:
        for detail_path in graphviz_detail_paths:
            _draw_display_point_path_stroke(
                ax,
                detail_path,
                edgecolor,
                max(float(style.stroke_width), 0.0),
                zorder=zorder,
                linestyle=_display_point_linestyle(style.stroke_dash, style.stroke_dash_pattern),
                capstyle=str(getattr(style, "stroke_cap", "butt")),
                joinstyle=str(getattr(style, "stroke_join", "miter")),
            )
        return

    if style.shape == "box3d":
        # Overlay darker tints on the top and right extrusion faces so the
        # 3D illusion reads at a glance.
        from matplotlib.patches import Polygon as MplPolygon

        half_w = w / 2.0
        half_h = h / 2.0
        left = x - half_w
        right = x + half_w
        bottom = y - half_h
        top = y + half_h
        depth = min(half_w, half_h) * 0.25
        offset_x = depth
        offset_y = depth * 0.70
        front_right = right - offset_x
        front_top = top - offset_y

        top_face = MplPolygon(
            [
                (left, front_top),
                (left + offset_x, top),
                (front_right + offset_x, top),
                (front_right, front_top),
            ],
            closed=True,
            # Use a lighter alpha on the top face so it reads as a surface
            # catching light rather than a second side wall.
            facecolor=(0.0, 0.0, 0.0, 0.12),
            edgecolor="none",
            linewidth=0.0,
            zorder=zorder - 0.01,
        )
        right_face = MplPolygon(
            [
                (front_right, bottom),
                (right, bottom + offset_y),
                (right, top),
                (front_right, front_top),
            ],
            closed=True,
            # Darken the right face more aggressively than the top face to
            # create a stable faux-3D light direction and preserve depth cues.
            facecolor=(0.0, 0.0, 0.0, 0.18),
            edgecolor="none",
            linewidth=0.0,
            zorder=zorder - 0.01,
        )
        ax.add_patch(top_face)
        ax.add_patch(right_face)
        return

    if style.shape == "double_circle":
        # Draw the inner concentric circle as a filled data-coordinate ribbon
        # so the width remains available to layout-space geometry.
        gap_ratio = 0.15
        inner_w = w * (1.0 - gap_ratio)
        inner_h = h * (1.0 - gap_ratio)
        display_scale = _compute_display_scale(ax)
        stroke_scale = _effective_stroke_scale(ax)
        border_width = clamp_border_width(
            max(float(style.stroke_width), 1.0) * stroke_scale,
            inner_w,
            inner_h,
        )
        shape_spec = ShapeSpec(
            center_x=x,
            center_y=y,
            width=inner_w,
            height=inner_h,
            shape="ellipse",
            corner_radius=0.0,
            aspect_ratio=None,
        )
        dash_pattern = (
            "solid"
            if style.stroke_dash == "solid" and style.stroke_dash_pattern is None
            else _node_border_pattern(style, display_scale)
        )
        _draw_border_ribbon(
            ax,
            shape_spec,
            build_shape_path(shape_spec),
            border_width,
            "center",
            dash_pattern,
            edgecolor,
            zorder,
        )
        return

    if style.shape != "cylinder":
        return

    cap_h = max(h * 0.16, 1.0)
    display_scale = _compute_display_scale(ax)
    stroke_scale = _effective_stroke_scale(ax)
    rim_h = cap_h * 2.0
    border_width = clamp_border_width(float(style.stroke_width) * stroke_scale, w, rim_h)
    shape_spec = ShapeSpec(
        center_x=x,
        center_y=y + h / 2 - cap_h,
        width=w,
        height=rim_h,
        shape="ellipse",
        corner_radius=0.0,
        aspect_ratio=None,
    )
    dash_pattern = (
        "solid"
        if style.stroke_dash == "solid" and style.stroke_dash_pattern is None
        else _node_border_pattern(style, display_scale)
    )
    _draw_border_ribbon(
        ax,
        shape_spec,
        build_shape_path(shape_spec),
        border_width,
        "center",
        dash_pattern,
        edgecolor,
        zorder,
    )


def _draw_gradient_fill(
    ax: Any,
    patch: Any,
    x: float,
    y: float,
    w: float,
    h: float,
    style: Any,
    *,
    alpha_multiplier: float = 1.0,
    zorder: float = 1.95,
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
    alpha_multiplier : float, keyword-only, default=1.0
        Additional opacity multiplier used when the gradient is rendered as an
        overlay on top of another fill style.
    zorder : float, keyword-only, default=1.95
        Artist z-order for the gradient image.
    """
    from matplotlib.colors import LinearSegmentedColormap

    resolution = _PATTERN_FILL_RESOLUTION
    grid = np.linspace(-1.0, 1.0, resolution)
    xx, yy = np.meshgrid(grid, grid)

    if style.gradient == "radial":
        gradient_angle = float(style.gradient_angle) % 360.0
        if abs(gradient_angle) <= 1e-9:
            focus_x = 0.0
            focus_y = 0.0
        else:
            angle = np.deg2rad(gradient_angle)
            focus_x = float(np.cos(angle))
            focus_y = float(np.sin(angle))
        dx = xx - focus_x
        dy = yy - focus_y
        ray_norm_sq = dx**2 + dy**2
        focus_dot_ray = focus_x * dx + focus_y * dy
        # Graphviz emits an SVG radial gradient with a 75% object-bbox
        # radius. In this [-1, 1] raster that is a radius of 1.5. Solve each
        # focal ray's intersection with that circle to reproduce SVG focal
        # gradients, including non-zero gradientangle values.
        radius = 1.5
        discriminant = np.maximum(
            focus_dot_ray**2 - ray_norm_sq * (focus_x**2 + focus_y**2 - radius**2),
            0.0,
        )
        denominator = np.maximum(ray_norm_sq, 1e-12)
        end_scale = (-focus_dot_ray + np.sqrt(discriminant)) / denominator
        data = np.where(ray_norm_sq <= 1e-12, 0.0, 1.0 / np.maximum(end_scale, 1e-12))
        data = np.clip(data, 0.0, 1.0)
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
        alpha=_node_fill_alpha(style) * float(alpha_multiplier),
        zorder=zorder,
        aspect="auto",
    )
    image.set_clip_path(patch)


def _pattern_fill_colors(style: Any) -> List[str]:
    """Resolve the color sequence used by patterned node fills.

    Parameters
    ----------
    style : Any
        Node style object.

    Returns
    -------
    list[str]
        Pattern colors in draw order.
    """
    if style.fill_pattern_colors:
        return list(style.fill_pattern_colors)
    return [str(style.fill), darken_hex(str(style.fill), 0.18)]


def _node_fill_alpha(style: Any) -> float:
    """Return the effective opacity for node fill layers.

    Parameters
    ----------
    style : Any
        Node style exposing overall and fill-only opacity values.

    Returns
    -------
    float
        Product of overall and fill-only opacity, clamped to ``[0, 1]``.
    """

    overall_opacity = float(getattr(style, "opacity", 1.0))
    fill_opacity = float(getattr(style, "fill_opacity", 1.0))
    return min(max(overall_opacity * fill_opacity, 0.0), 1.0)


def _hatched_overlay_color(style: Any) -> str:
    """Return a visible stroke color for hatched pattern overlays.

    Parameters
    ----------
    style : Any
        Node style object.

    Returns
    -------
    str
        Dark contrasting hatch color. When the pattern palette does not provide
        a distinct second tone, fall back to the node stroke color or a darker
        fill-derived shade.
    """
    colors = _pattern_fill_colors(style)
    fill_color = str(style.fill)
    for candidate in colors[1:]:
        if not _colors_match(candidate, fill_color):
            return str(candidate)
    stroke_color = str(getattr(style, "stroke", "") or "")
    if stroke_color and not _colors_match(stroke_color, fill_color):
        return stroke_color
    return darken_hex(fill_color, 0.32)


def _draw_striped_fill(
    ax: Any,
    clip_patch: Any,
    x: float,
    y: float,
    w: float,
    h: float,
    style: Any,
) -> None:
    """Draw a striped fill clipped to one node outline.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    clip_patch : Any
        Clip patch matching the node interior.
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
    from matplotlib.colors import ListedColormap

    colors = _pattern_fill_colors(style)
    resolution = _PATTERN_FILL_RESOLUTION
    grid = np.linspace(-1.0, 1.0, resolution)
    xx, yy = np.meshgrid(grid, grid)
    angle = np.deg2rad(float(style.fill_pattern_angle))
    projection = xx * np.cos(angle) + yy * np.sin(angle)
    projection_min = float(np.min(projection))
    projection_range = max(float(np.max(projection)) - projection_min, 1e-9)
    normalized = (projection - projection_min) / projection_range
    palette_count = max(len(colors), 1)
    raw_values = list(getattr(style, "fill_pattern_values", None) or [])
    if len(raw_values) != palette_count or sum(max(float(value), 0.0) for value in raw_values) <= 0:
        weights = np.full(palette_count, 1.0 / palette_count)
    else:
        positive_values = np.asarray([max(float(value), 0.0) for value in raw_values])
        weights = positive_values / float(np.sum(positive_values))
    boundaries = np.cumsum(weights)
    bands = np.searchsorted(boundaries, normalized, side="right")
    bands = np.clip(bands, 0, palette_count - 1)
    image = ax.imshow(
        bands,
        extent=(x - w / 2.0, x + w / 2.0, y - h / 2.0, y + h / 2.0),
        origin="lower",
        cmap=ListedColormap(colors),
        interpolation="nearest",
        alpha=_node_fill_alpha(style),
        zorder=1.95,
        aspect="auto",
        vmin=0,
        vmax=max(palette_count - 1, 1),
    )
    image.set_clip_path(clip_patch)


def _draw_pie_fill(ax: Any, shape_spec: ShapeSpec, style: Any, clip_patch: Any) -> None:
    """Draw a pie or donut fill clipped to one node outline.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    shape_spec : ShapeSpec
        Node geometry in data coordinates.
    style : Any
        Node style object.
    clip_patch : Any
        Clip patch matching the node interior.
    """
    from matplotlib.patches import PathPatch, Wedge
    from matplotlib.transforms import Affine2D

    values = getattr(style, "fill_pattern_values", None)
    if values is None or len(values) == 0:
        return

    slice_values = [max(float(value), 0.0) for value in values]
    total = sum(slice_values)
    if total <= 0.0:
        return

    colors = list(getattr(style, "fill_pattern_colors", None) or [str(style.fill)])
    radius_x = float(shape_spec.width) / 2.0
    radius_y = float(shape_spec.height) / 2.0
    if radius_x <= 0.0 or radius_y <= 0.0:
        return

    hole_fraction = min(max(float(getattr(style, "fill_pattern_hole", 0.0)), 0.0), 0.99)
    wedge_width = None if hole_fraction <= 0.0 else 1.0 - hole_fraction
    current_angle = -float(getattr(style, "fill_pattern_angle", 0.0))
    pie_transform = (
        Affine2D()
        .scale(radius_x, radius_y)
        .translate(float(shape_spec.center_x), float(shape_spec.center_y))
    )

    for index, value in enumerate(slice_values):
        if value <= 0.0:
            continue

        sweep = (value / total) * 360.0
        wedge_kwargs: Dict[str, Any] = {
            "center": (0.0, 0.0),
            "r": 1.0,
            "theta1": current_angle,
            "theta2": current_angle + sweep,
        }
        if wedge_width is not None:
            wedge_kwargs["width"] = wedge_width
        # Scale a unit wedge into the node's bounding ellipse so pie fills stay
        # centered and span the full node dimensions instead of the smaller axis.
        wedge_path = Wedge(**wedge_kwargs).get_path().transformed(pie_transform)
        wedge = PathPatch(
            wedge_path,
            facecolor=colors[index % len(colors)],
            edgecolor=str(getattr(style, "stroke", "#000000")),
            linewidth=0.5,
            alpha=_node_fill_alpha(style),
            zorder=2.01,
        )
        wedge.set_clip_path(clip_patch)
        ax.add_patch(wedge)
        current_angle += sweep


def _draw_node_fill(
    ax: Any,
    shape_spec: ShapeSpec,
    fill_path: Any,
    clip_patch: Any,
    x: float,
    y: float,
    w: float,
    h: float,
    style: Any,
    facecolor: Any,
) -> None:
    """Draw one node fill when batched collections cannot express the style.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    shape_spec : ShapeSpec
        Node geometry in data coordinates.
    fill_path : Any
        Node interior path in data coordinates.
    clip_patch : Any
        Clip patch matching ``fill_path``.
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
        Matplotlib-compatible fill color.
    """
    from matplotlib.patches import PathPatch

    if style.fill_pattern == "pie":
        if float(getattr(style, "fill_pattern_hole", 0.0)) <= 0.0:
            fill_patch = PathPatch(
                fill_path,
                facecolor=facecolor,
                edgecolor="none",
                linewidth=0.0,
                zorder=2.0,
            )
            ax.add_patch(fill_patch)
        _draw_pie_fill(ax, shape_spec, style, clip_patch)
        if style.gradient != "none" and _node_fill_alpha(style) > 0.0:
            # Pie wedges fully cover the base fill, so the gradient must be
            # layered afterward to remain visible in combo renders.
            _draw_gradient_fill(
                ax,
                clip_patch,
                x,
                y,
                w,
                h,
                style,
                alpha_multiplier=_PIE_GRADIENT_OVERLAY_ALPHA_MULTIPLIER,
                zorder=2.02,
            )
        return
    if style.fill_pattern == "striped":
        _draw_striped_fill(ax, clip_patch, x, y, w, h, style)
        return
    if style.fill_pattern == "hatched":
        fill_patch = PathPatch(
            fill_path,
            facecolor=facecolor,
            edgecolor="none",
            linewidth=0.0,
            zorder=2.0,
        )
        ax.add_patch(fill_patch)
        hatch_patch = PathPatch(
            fill_path,
            facecolor="none",
            edgecolor=_hatched_overlay_color(style),
            linewidth=_MIN_HATCH_LINEWIDTH_POINTS,
            hatch=_HATCH_PATTERN,
            alpha=_node_fill_alpha(style),
            zorder=2.01,
        )
        ax.add_patch(hatch_patch)
        return
    if style.gradient != "none" and _node_fill_alpha(style) > 0.0:
        _draw_gradient_fill(ax, clip_patch, x, y, w, h, style)
        return

    fill_patch = PathPatch(
        fill_path,
        facecolor=facecolor,
        edgecolor="none",
        linewidth=0.0,
        zorder=2.0,
    )
    ax.add_patch(fill_patch)


def _draw_node_border_path(
    ax: Any,
    path: Any,
    style: Any,
    edgecolor: Any,
    border_width: float,
    display_scale: float,
) -> None:
    """Draw one node border centerline as a filled data-coordinate ribbon.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    path : Any
        Border outline path in data coordinates.
    style : Any
        Node style object.
    edgecolor : Any
        Matplotlib-compatible border color.
    border_width : float
        Border width in data coordinates.
    display_scale : float
        Point-to-data conversion factor for dash pattern scaling.
    """
    dash_pattern = (
        (1.0e9, 1.0e9)
        if style.stroke_dash == "solid" and style.stroke_dash_pattern is None
        else _node_border_pattern(style, display_scale)
    )
    ribbons = dash_ribbon_paths(path, dash_pattern, border_width)
    add_filled_collections(
        ax=ax,
        fill_paths=[],
        fill_colors=[],
        border_paths=ribbons,
        border_colors=[edgecolor] * len(ribbons),
        fill_zorder=2.05,
        border_zorder=2.05,
    )
    from matplotlib.patches import PathPatch

    patch = PathPatch(
        path,
        facecolor="none",
        edgecolor=edgecolor,
        linewidth=max(float(getattr(style, "stroke_width", 0.0)), 0.0),
        capstyle=_mpl_capstyle(str(getattr(style, "stroke_cap", "butt"))),
        joinstyle=str(getattr(style, "stroke_join", "miter")),
        zorder=2.06,
    )
    ax.add_patch(patch)


def _draw_node_outline(
    ax: Any,
    node_index: int,
    shape_spec: ShapeSpec,
    style: Any,
    border_width: float,
    border_position: str,
    display_scale: float,
) -> None:
    """Draw a Cytoscape-style stroke outside a node's visible border.

    Parameters
    ----------
    ax : Any
        Matplotlib axes receiving the outline.
    node_index : int
        Node index used to identify the outline artist.
    shape_spec : ShapeSpec
        Base node geometry in data coordinates.
    style : Any
        Node style exposing outline color, width, offset, and style.
    border_width : float
        Visible node-border width in data coordinates.
    border_position : str
        Node-border placement mode.
    display_scale : float
        Point-to-data conversion factor.

    Returns
    -------
    None
        Adds an outline patch to ``ax`` when its width is positive.
    """

    outline_width = max(float(getattr(style, "outline_width", 0.0)), 0.0)
    if outline_width <= 0.0:
        return

    border_outset_factors = {"inside": 0.0, "center": 0.5, "outside": 1.0}
    border_outset = border_width * border_outset_factors.get(border_position, 0.5)
    outline_gap = max(float(getattr(style, "outline_offset", 0.0)), 0.0) * display_scale
    outline_half_width = outline_width * display_scale / 2.0
    centerline_outset = border_outset + outline_gap + outline_half_width
    outline_path = build_shape_path(_expanded_shape_spec(shape_spec, centerline_outset))
    outline_color = str(getattr(style, "outline_color", "") or style.stroke)
    _draw_display_point_path_stroke(
        ax,
        outline_path,
        outline_color,
        outline_width,
        zorder=2.07,
        linestyle=_display_point_linestyle(getattr(style, "outline_style", "solid"), None),
        joinstyle="round",
        gid=f"dagua-node-outline-{node_index}",
    )


def _requires_custom_node_rendering(style: Any) -> bool:
    """Return whether a node must bypass the batched fill/border collections.

    Parameters
    ----------
    style : Any
        Node style object.

    Returns
    -------
    bool
        ``True`` when per-node rendering is required for advanced cosmetics.
    """
    return (
        style.fill_pattern != "solid"
        or int(style.border_count) > 1
        or style.stroke_cap != "butt"
        or style.stroke_join != "miter"
        or str(getattr(style, "shape", "")) in {"rect", "roundrect", "double_circle", "box3d"}
    )


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
        corner_radius=(
            style.corner_radius
            if bool(getattr(style, "scale_corner_radius", False))
            else scale_corner_radius(getattr(style, "corner_radius", 0.0), display_scale)
        ),
        shadow_offset=(
            float(style.shadow_offset[0]) * display_scale,
            float(style.shadow_offset[1]) * display_scale,
        ),
    )


def _node_corner_radius_data(
    style: Any,
    display_scale: float,
    node_width: float,
    node_height: float,
) -> Any:
    """Return node corner radii in data units for the current node size.

    Parameters
    ----------
    style : Any
        Node style object exposing ``corner_radius`` and
        ``scale_corner_radius``.
    display_scale : float
        Point-to-data conversion factor.
    node_width : float
        Node width in data units.
    node_height : float
        Node height in data units.

    Returns
    -------
    Any
        Scalar or per-corner radii in data units.
    """

    if bool(getattr(style, "scale_corner_radius", False)):
        return scale_corner_radius(
            getattr(style, "corner_radius", 0.0),
            min(max(float(node_width), 0.0), max(float(node_height), 0.0)),
        )
    return scale_corner_radius(getattr(style, "corner_radius", 0.0), display_scale)


def _normalize_border_position(border_position: str) -> str:
    """Return a supported node border-position mode.

    Parameters
    ----------
    border_position : str
        Requested border placement mode.

    Returns
    -------
    str
        One of ``"center"``, ``"inside"``, or ``"outside"``. Unsupported
        values fall back to ``"center"`` to preserve rendering.
    """
    if border_position in {"center", "inside", "outside"}:
        return border_position
    return "center"


def _node_border_outward_offset(style: Any, stroke_scale: float) -> float:
    """Return how far a border extends beyond the node boundary.

    Parameters
    ----------
    style : Any
        Node style object exposing ``stroke_width`` and ``border_position``.
    stroke_scale : float
        Backend-calibrated point-to-data conversion factor for stroke ribbons.

    Returns
    -------
    float
        Outward border extent in data units. ``center`` and ``inside`` borders
        keep routed edge terminals on the mathematical node boundary.
    """
    stroke_width = max(float(getattr(style, "stroke_width", 0.0)), 0.0) * stroke_scale
    border_position = _normalize_border_position(getattr(style, "border_position", "center"))
    if border_position in {"center", "inside"}:
        return 0.0
    if border_position == "outside":
        return stroke_width
    return 0.0


def _offset_point_from_node_center(
    point: Tuple[float, float],
    node_center: Tuple[float, float],
    offset: float,
) -> Tuple[float, float]:
    """Move a terminal point outward from a node center by a fixed amount.

    Parameters
    ----------
    point : tuple[float, float]
        Terminal point on the node boundary in data coordinates.
    node_center : tuple[float, float]
        Node center in data coordinates.
    offset : float
        Requested outward displacement in data units.

    Returns
    -------
    tuple[float, float]
        Displaced terminal point. Degenerate radial vectors or non-positive
        offsets keep the original point unchanged.
    """
    if offset <= 0.0:
        return point
    dx = float(point[0]) - float(node_center[0])
    dy = float(point[1]) - float(node_center[1])
    distance = float(np.hypot(dx, dy))
    if distance <= 1e-9:
        return point
    scale = offset / distance
    return (float(point[0]) + dx * scale, float(point[1]) + dy * scale)


def _offset_edge_terminal_point(
    graph: Any,
    positions: Optional[np.ndarray],
    node_idx: int,
    point: Tuple[float, float],
    stroke_scale: float,
) -> Tuple[float, float]:
    """Return a marker tip translated to the node border's visible outer edge.

    Parameters
    ----------
    graph : Any
        Graph exposing node-style lookup.
    positions : numpy.ndarray | None
        Node positions with shape ``[N, 2]`` in data coordinates.
    node_idx : int
        Node index owning the terminal.
    point : tuple[float, float]
        Original terminal point on the mathematical node boundary.
    stroke_scale : float
        Backend-calibrated point-to-data conversion factor for node border
        strokes.

    Returns
    -------
    tuple[float, float]
        Terminal point shifted outward so arrow tips touch the border outstroke.
        When positions are unavailable, the original point is returned.
    """
    if positions is None or node_idx < 0 or node_idx >= int(len(positions)):
        return point
    style = _node_style_for_render(graph, node_idx)
    offset = _node_border_outward_offset(style, stroke_scale)
    if offset <= 0.0:
        return point
    node_center = (float(positions[node_idx, 0]), float(positions[node_idx, 1]))
    return _offset_point_from_node_center(point, node_center, offset)


def _translate_path(path: Any, dx: float, dy: float) -> Any:
    """Return a matplotlib path translated by a fixed data-space vector.

    Parameters
    ----------
    path : Any
        Matplotlib ``Path`` instance to translate.
    dx : float
        Horizontal translation in data units.
    dy : float
        Vertical translation in data units.

    Returns
    -------
    Any
        Translated ``Path`` preserving the original path codes.
    """
    from matplotlib.path import Path

    vertices = np.asarray(path.vertices, dtype=float) + np.array([dx, dy], dtype=float)
    return Path(vertices, path.codes)


def _translate_arrowhead_result(result: Any, dx: float, dy: float) -> Any:
    """Return arrowhead geometry translated without changing body trim data.

    Parameters
    ----------
    result : Any
        ``ArrowheadResult`` to translate.
    dx : float
        Horizontal translation in data units.
    dy : float
        Vertical translation in data units.

    Returns
    -------
    Any
        Translated ``ArrowheadResult``. Zero translation returns the original
        result so callers avoid unnecessary object churn.
    """
    if abs(dx) <= 1e-9 and abs(dy) <= 1e-9:
        return result
    return replace(
        result,
        filled_paths=[_translate_path(path, dx, dy) for path in result.filled_paths],
        stroked_paths=[_translate_path(path, dx, dy) for path in result.stroked_paths],
        trim_contour=_translate_path(result.trim_contour, dx, dy),
    )


def _offset_custom_edge_collection_terminals(
    collection: DaguaEdgeCollection,
    graph: Any,
    positions: Optional[np.ndarray],
    stroke_scale: float,
) -> None:
    """Shift custom-rendered arrowheads to the visible node border.

    Parameters
    ----------
    collection : DaguaEdgeCollection
        Prepared collection whose terminal geometry should be adjusted.
    graph : Any
        Graph exposing node-style lookup.
    positions : numpy.ndarray | None
        Node positions with shape ``[N, 2]`` in data coordinates.
    stroke_scale : float
        Backend-calibrated point-to-data conversion factor for node border
        strokes.

    Returns
    -------
    None
        The collection is updated in place.
    """
    if positions is None:
        return

    updated_prepared_edges = []
    for prepared in collection.prepared_edges:
        edge = prepared.edge
        head_result = prepared.head_result
        tail_result = prepared.tail_result
        source_result = prepared.source_result

        if head_result is not None and edge.target_node is not None:
            shifted_tip = _offset_edge_terminal_point(
                graph,
                positions,
                int(edge.target_node),
                (float(prepared.lane_curve.p1[0]), float(prepared.lane_curve.p1[1])),
                stroke_scale,
            )
            dx = shifted_tip[0] - float(prepared.lane_curve.p1[0])
            dy = shifted_tip[1] - float(prepared.lane_curve.p1[1])
            head_result = _translate_arrowhead_result(head_result, dx, dy)

        if tail_result is not None and edge.source_node is not None:
            shifted_tip = _offset_edge_terminal_point(
                graph,
                positions,
                int(edge.source_node),
                (float(prepared.lane_curve.p0[0]), float(prepared.lane_curve.p0[1])),
                stroke_scale,
            )
            dx = shifted_tip[0] - float(prepared.lane_curve.p0[0])
            dy = shifted_tip[1] - float(prepared.lane_curve.p0[1])
            tail_result = _translate_arrowhead_result(tail_result, dx, dy)

        if source_result is not None and edge.source_node is not None:
            shifted_tip = _offset_edge_terminal_point(
                graph,
                positions,
                int(edge.source_node),
                (float(prepared.lane_curve.p0[0]), float(prepared.lane_curve.p0[1])),
                stroke_scale,
            )
            dx = shifted_tip[0] - float(prepared.lane_curve.p0[0])
            dy = shifted_tip[1] - float(prepared.lane_curve.p0[1])
            source_result = _translate_arrowhead_result(source_result, dx, dy)

        updated_prepared_edges.append(
            replace(
                prepared,
                head_result=head_result,
                tail_result=tail_result,
                source_result=source_result,
            )
        )

    collection.prepared_edges = updated_prepared_edges


def _normalize_external_label_position(position: str) -> str:
    """Return a supported external-label anchor position.

    Parameters
    ----------
    position : str
        Requested label position. Underscores and legacy four-way names are
        accepted as aliases.

    Returns
    -------
    str
        One of the nine row-column positions. Invalid values fall back to
        ``"bottom-center"``.
    """
    normalized_position = str(position).replace("_", "-")
    aliases = {
        "top": "top-center",
        "bottom": "bottom-center",
        "left": "center-left",
        "right": "center-right",
    }
    normalized_position = aliases.get(normalized_position, normalized_position)
    if normalized_position in {
        "top-left",
        "top-center",
        "top-right",
        "center-left",
        "center",
        "center-right",
        "bottom-left",
        "bottom-center",
        "bottom-right",
    }:
        return normalized_position
    return "bottom-center"


def _external_label_anchor(
    center_x: float,
    center_y: float,
    half_width: float,
    half_height: float,
    offset: float,
    position: str,
) -> Tuple[float, float, str, str]:
    """Return the anchor and alignment for a nine-position external label.

    Parameters
    ----------
    center_x : float
        Node center x-coordinate.
    center_y : float
        Node center y-coordinate.
    half_width : float
        Half of the node width.
    half_height : float
        Half of the node height.
    offset : float
        Clearance outside the selected node sides.
    position : str
        Normalized nine-position label token.

    Returns
    -------
    tuple[float, float, str, str]
        Text x/y anchor followed by Matplotlib horizontal and vertical
        alignment values.
    """

    row, separator, column = position.partition("-")
    if not separator:
        row = column = "center"

    if column == "left":
        text_x, horizontal_alignment = center_x - half_width - offset, "right"
    elif column == "right":
        text_x, horizontal_alignment = center_x + half_width + offset, "left"
    else:
        text_x, horizontal_alignment = center_x, "center"

    if row == "top":
        text_y, vertical_alignment = center_y + half_height + offset, "bottom"
    elif row == "bottom":
        text_y, vertical_alignment = center_y - half_height - offset, "top"
    else:
        text_y, vertical_alignment = center_y, "center"
    return text_x, text_y, horizontal_alignment, vertical_alignment


def _expanded_shape_spec(spec: ShapeSpec, delta: float) -> ShapeSpec:
    """Return a shape specification expanded equally on all sides.

    Parameters
    ----------
    spec : ShapeSpec
        Base node-shape geometry in data coordinates.
    delta : float
        Outset distance in data units.

    Returns
    -------
    ShapeSpec
        Expanded shape geometry. Non-positive ``delta`` returns ``spec``
        unchanged.
    """
    if delta <= 0.0:
        return spec
    return ShapeSpec(
        center_x=spec.center_x,
        center_y=spec.center_y,
        width=spec.width + 2.0 * delta,
        height=spec.height + 2.0 * delta,
        shape=spec.shape,
        corner_radius=add_corner_radius(spec.corner_radius, delta),
        aspect_ratio=spec.aspect_ratio,
        polygon_points=spec.polygon_points,
    )


def _graphviz_strict_terminal_point(
    curve: BezierCurve,
    center: Tuple[float, float],
    size: Tuple[float, float],
    style: NodeStyle,
    terminal: str,
) -> Tuple[float, float]:
    """Return a strict-theme edge terminal clipped to the rendered ellipse.

    Parameters
    ----------
    curve : BezierCurve
        Routed edge curve whose endpoint should touch the visible node border.
    center : tuple[float, float]
        Node center in render/data coordinates.
    size : tuple[float, float]
        Layout node size before strict visual ellipse expansion.
    style : NodeStyle
        Effective node style for the terminal node.
    terminal : str
        Either ``"source"`` for ``curve.p0`` or ``"target"`` for ``curve.p1``.

    Returns
    -------
    tuple[float, float]
        Terminal point on the rendered strict ellipse boundary. Non-ellipse
        nodes and degenerate rays return the original routed endpoint.
    """
    from dagua.render.edges.intersection import ray_ellipse_intersection

    if str(style.shape) not in {"ellipse", "circle"}:
        return curve.p0 if terminal == "source" else curve.p1

    visual_spec = ShapeSpec(
        center_x=float(center[0]),
        center_y=float(center[1]),
        width=float(size[0]),
        height=float(size[1]),
        shape=str(style.shape),
        corner_radius=0.0,
        aspect_ratio=style.aspect_ratio,
    )
    original_terminal = curve.p0 if terminal == "source" else curve.p1
    center_point = np.asarray(center, dtype=float)
    ray_origin = center_point
    direction = np.asarray(original_terminal, dtype=float) - center_point
    if float(np.hypot(direction[0], direction[1])) <= 1e-9:
        return curve.p0 if terminal == "source" else curve.p1

    hit = ray_ellipse_intersection(
        center=center_point,
        half_size=(visual_spec.width / 2.0, visual_spec.height / 2.0),
        ray_origin=ray_origin,
        ray_direction=direction,
    )
    return float(hit[0]), float(hit[1])


def _graphviz_strict_reclip_edge_terminals(
    graph: Any,
    curves: List[BezierCurve],
    positions: Optional[np.ndarray],
) -> List[BezierCurve]:
    """Clip strict-theme ellipse edge endpoints to the rendered node outline.

    Parameters
    ----------
    graph : Any
        Graph exposing node styles, sizes, and edge indices.
    curves : list[BezierCurve]
        Routed edge curves.
    positions : numpy.ndarray | None
        Node centers with shape ``[N, 2]`` in render/data coordinates.

    Returns
    -------
    list[BezierCurve]
        Curves with source and target endpoints aligned to strict visual
        ellipse boundaries.
    """
    if not _is_graphviz_strict_render(graph) or positions is None:
        return curves
    if not curves or not hasattr(graph, "edge_index") or not hasattr(graph, "node_sizes"):
        return curves

    if hasattr(graph.node_sizes, "detach"):
        node_sizes = graph.node_sizes.detach().cpu().numpy()
    else:
        node_sizes = np.asarray(graph.node_sizes, dtype=float)
    reclipped: List[BezierCurve] = []
    for edge_idx, curve in enumerate(curves):
        if edge_idx >= int(graph.edge_index.shape[1]) or curve.waypoints is not None:
            reclipped.append(curve)
            continue
        src_idx = int(graph.edge_index[0, edge_idx])
        tgt_idx = int(graph.edge_index[1, edge_idx])
        source = _graphviz_strict_terminal_point(
            curve,
            center=(float(positions[src_idx, 0]), float(positions[src_idx, 1])),
            size=(float(node_sizes[src_idx, 0]), float(node_sizes[src_idx, 1])),
            style=_node_style_for_render(graph, src_idx),
            terminal="source",
        )
        target = _graphviz_strict_terminal_point(
            curve,
            center=(float(positions[tgt_idx, 0]), float(positions[tgt_idx, 1])),
            size=(float(node_sizes[tgt_idx, 0]), float(node_sizes[tgt_idx, 1])),
            style=_node_style_for_render(graph, tgt_idx),
            terminal="target",
        )
        reclipped.append(
            BezierCurve(
                p0=source,
                cp1=curve.cp1,
                cp2=curve.cp2,
                p1=target,
                waypoints=curve.waypoints,
                routing=curve.routing,
                direction=curve.direction,
                step_fraction=curve.step_fraction,
            )
        )
    return reclipped


def _node_fill_path(
    shape_spec: ShapeSpec,
    outer_path: Any,
    border_width: float,
    border_position: str,
) -> Any:
    """Return the node fill path for a requested border position.

    Parameters
    ----------
    shape_spec : ShapeSpec
        Base node-shape geometry.
    outer_path : Any
        Outer node-shape path.
    border_width : float
        Border width in data units.
    border_position : str
        Border placement mode.

    Returns
    -------
    Any
        Fill path in data coordinates.
    """
    if border_width <= 0.0:
        return outer_path
    if border_position == "inside":
        return inset_shape_path(shape_spec, border_width)
    if border_position == "center":
        return outer_path
    return outer_path


def _solid_border_ring_paths(
    shape_spec: ShapeSpec,
    outer_path: Any,
    border_width: float,
    border_position: str,
) -> Tuple[Any, Any]:
    """Return the outer and inner paths for a solid node border ring.

    Parameters
    ----------
    shape_spec : ShapeSpec
        Base node-shape geometry.
    outer_path : Any
        Outer node-shape path.
    border_width : float
        Border width in data units.
    border_position : str
        Border placement mode.

    Returns
    -------
    tuple[Any, Any]
        Outer and inner border paths for ``annular_path`` construction.
    """
    if border_position == "inside":
        return outer_path, inset_shape_path(shape_spec, border_width)
    if border_position == "outside":
        expanded = build_shape_path(_expanded_shape_spec(shape_spec, border_width))
        return expanded, outer_path
    expanded = build_shape_path(_expanded_shape_spec(shape_spec, border_width / 2.0))
    inner = inset_shape_path(shape_spec, border_width / 2.0)
    return expanded, inner


def _node_border_centerline_path(
    shape_spec: ShapeSpec,
    outer_path: Any,
    border_width: float,
    border_position: str,
) -> Any:
    """Return the border centerline path for dashed or stroked node borders.

    Parameters
    ----------
    shape_spec : ShapeSpec
        Base node-shape geometry.
    outer_path : Any
        Outer node-shape path.
    border_width : float
        Border width in data units.
    border_position : str
        Border placement mode.

    Returns
    -------
    Any
        Border centerline path in data coordinates.
    """
    if border_position == "inside":
        return inset_shape_path(shape_spec, border_width / 2.0)
    if border_position == "outside":
        return build_shape_path(_expanded_shape_spec(shape_spec, border_width / 2.0))
    return outer_path


@lru_cache(maxsize=128)
def _load_node_image_rgba(image_ref: str) -> Optional[np.ndarray]:
    """Load a node image into a normalized RGBA array.

    Parameters
    ----------
    image_ref : str
        Local filesystem path or URL.

    Returns
    -------
    numpy.ndarray | None
        RGBA image data normalized to ``[0, 1]``, or ``None`` when the image
        cannot be loaded. The renderer falls back to the node fill in that case.
    """
    from PIL import Image

    try:
        parsed = urlparse(image_ref)
        if parsed.scheme in {"http", "https"}:
            with urlopen(image_ref, timeout=5.0) as response:
                data = response.read()
            with Image.open(io.BytesIO(data)) as image:
                rgba = image.convert("RGBA")
                return np.asarray(rgba, dtype=np.float32) / 255.0

        with Image.open(Path(image_ref).expanduser()) as image:
            rgba = image.convert("RGBA")
            return np.asarray(rgba, dtype=np.float32) / 255.0
    except Exception:
        return None


def _draw_image_node(
    ax: Any,
    shape_spec: ShapeSpec,
    style: Any,
    clip_patch: Any,
) -> None:
    """Render one clipped image layer inside a node shape.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    shape_spec : ShapeSpec
        Node geometry in data coordinates.
    style : Any
        Node style exposing image fields.
    clip_patch : Any
        Clip patch matching the node shape boundary.
    """
    image_ref = str(getattr(style, "image", "")).strip()
    if image_ref == "":
        return

    image_rgba = _load_node_image_rgba(image_ref)
    if image_rgba is None:
        return

    image_height, image_width = image_rgba.shape[:2]
    if image_height <= 0 or image_width <= 0:
        return

    cx = float(shape_spec.center_x)
    cy = float(shape_spec.center_y)
    width = float(shape_spec.width)
    height = float(shape_spec.height)

    fit_mode = getattr(style, "image_fit", "contain")
    if fit_mode == "stretch":
        display_width = width
        display_height = height
    else:
        image_aspect = float(image_width) / max(float(image_height), 1.0)
        node_aspect = width / max(height, 1e-9)
        if fit_mode == "cover":
            if image_aspect > node_aspect:
                display_height = height
                display_width = display_height * image_aspect
            else:
                display_width = width
                display_height = display_width / max(image_aspect, 1e-9)
        else:
            if image_aspect > node_aspect:
                display_width = width
                display_height = display_width / max(image_aspect, 1e-9)
            else:
                display_height = height
                display_width = display_height * image_aspect

    image_artist = ax.imshow(
        image_rgba,
        extent=[
            cx - display_width / 2.0,
            cx + display_width / 2.0,
            cy - display_height / 2.0,
            cy + display_height / 2.0,
        ],
        aspect="auto",
        alpha=float(getattr(style, "image_opacity", 1.0)),
        zorder=2.025,
        interpolation="bilinear",
    )
    image_artist.set_clip_path(clip_patch)


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


def _display_point_linestyle(style_name: Any, dash_pattern: Optional[Tuple[float, ...]]) -> Any:
    """Return a matplotlib linestyle for display-point stroke overrides.

    Parameters
    ----------
    style_name : Any
        Style token such as ``"solid"``, ``"dashed"``, or ``"dotted"``.
    dash_pattern : tuple[float, ...] | None
        Optional custom dash pattern already authored in display points.

    Returns
    -------
    Any
        Matplotlib-compatible linestyle.
    """
    if dash_pattern is not None:
        return (0.0, tuple(float(value) for value in dash_pattern))
    return str(style_name)


def _draw_display_point_path_stroke(
    ax: Any,
    path: Any,
    color: Any,
    linewidth_points: float,
    zorder: float,
    linestyle: Any = "solid",
    capstyle: str = "butt",
    joinstyle: str = "miter",
    gid: Optional[str] = None,
) -> Any:
    """Draw a path with a native matplotlib display-point stroke.

    Parameters
    ----------
    ax : Any
        Matplotlib axes receiving the patch.
    path : Any
        Path in data coordinates.
    color : Any
        Matplotlib-compatible edge color.
    linewidth_points : float
        Stroke width in display points.
    zorder : float
        Artist z-order.
    linestyle : Any, default="solid"
        Matplotlib line style or dash tuple in display points.
    capstyle : str, default="butt"
        Stroke cap style.
    joinstyle : str, default="miter"
        Stroke join style.
    gid : str | None, default=None
        Optional SVG/test identifier.

    Returns
    -------
    Any
        Added matplotlib patch.

    Notes
    -----
    This is the explicit opt-in escape hatch from data-coordinate ribbon
    construction. The value is not visible to the optimizer's manifold.
    """
    from matplotlib.patches import PathPatch

    linewidth = max(float(linewidth_points), 0.0)
    patch = PathPatch(
        path,
        facecolor="none",
        edgecolor=color,
        linewidth=linewidth,
        linestyle=linestyle,
        capstyle=_mpl_capstyle(capstyle),
        joinstyle=joinstyle,
        zorder=zorder,
    )
    if gid is not None:
        patch.set_gid(gid)
    ax.add_patch(patch)
    return patch


def _draw_display_point_text(
    ax: Any,
    spec: DaguaText,
    svg_hover_map: Optional[Dict[str, str]] = None,
) -> Any:
    """Draw a text label with native matplotlib display-point font sizing.

    Parameters
    ----------
    ax : Any
        Matplotlib axes receiving the text artist.
    spec : DaguaText
        Text specification whose ``font_size`` is interpreted as display
        points without data-coordinate conversion.
    svg_hover_map : dict[str, str] | None, default=None
        Optional SVG hover-text accumulator.

    Returns
    -------
    Any
        Added matplotlib ``Text`` artist.

    Notes
    -----
    This helper is used only for explicit ``*_override_points`` fields. Native
    matplotlib text is not differentiable and intentionally bypasses Dagua's
    data-coordinate glyph manifold.
    """
    bbox = None
    if spec.background is not None:
        bbox = {
            "boxstyle": f"round,pad=0.2,rounding_size={float(spec.background_corner_radius)}",
            "facecolor": spec.background,
            "edgecolor": "none",
            "alpha": float(spec.background_alpha),
        }
    artist = ax.text(
        spec.x,
        spec.y,
        spec.text,
        fontsize=max(float(spec.font_size), 1e-9),
        fontfamily=spec.font_family or RESOLVED_FONT,
        fontweight=spec.font_weight,
        fontstyle=spec.font_style,
        color=spec.font_color,
        alpha=spec.alpha,
        ha=spec.ha,
        va=spec.va,
        rotation=spec.rotation,
        clip_on=spec.clip_on,
        zorder=spec.zorder,
        bbox=bbox,
    )
    if spec.outline and spec.outline_width > 0.0:
        import matplotlib.patheffects as path_effects

        artist.set_path_effects(
            [
                path_effects.withStroke(
                    linewidth=float(spec.outline_width) * 2.0,
                    foreground=spec.outline_color,
                ),
                path_effects.Normal(),
            ]
        )
    if spec.clip_patch is not None:
        artist.set_clip_path(spec.clip_patch)
    if spec.gid is not None:
        artist.set_gid(spec.gid)
        _register_svg_hover_text(svg_hover_map, spec.gid, spec.text)
    return artist


def _text_shadow_specs(spec: DaguaText, style: Any, display_scale: float) -> List[DaguaText]:
    """Return layered label copies approximating a Cytoscape text shadow.

    Parameters
    ----------
    spec : DaguaText
        Foreground label specification.
    style : Any
        Node style exposing text-shadow properties.
    display_scale : float
        Point-to-data conversion factor for offsets and blur radius.

    Returns
    -------
    list[DaguaText]
        Shadow specifications ordered from outer blur samples to the center.
    """

    from matplotlib.colors import to_rgba

    shadow_color = str(getattr(style, "text_shadow_color", ""))
    if shadow_color == "":
        return []
    red, green, blue, color_alpha = to_rgba(shadow_color)
    if color_alpha <= 0.0:
        return []

    offset_x, offset_y = getattr(style, "text_shadow_offset", (0.0, 0.0))
    blur_points = max(float(getattr(style, "text_shadow_blur", 0.0)), 0.0)
    samples: List[Tuple[float, float]] = [(0.0, 0.0)]
    if blur_points > 0.0:
        ring_count = min(max(int(math.ceil(blur_points)), 2), 4)
        for ring_index in range(ring_count, 0, -1):
            radius = blur_points * float(ring_index) / float(ring_count)
            for angle_index in range(8):
                angle = 2.0 * math.pi * float(angle_index) / 8.0
                samples.append((radius * math.cos(angle), radius * math.sin(angle)))

    layer_alpha = color_alpha if len(samples) == 1 else color_alpha / math.sqrt(len(samples))
    shadow_specs: List[DaguaText] = []
    for layer_index, (blur_x, blur_y) in enumerate(samples):
        shadow_specs.append(
            replace(
                spec,
                x=spec.x + (float(offset_x) + blur_x) * display_scale,
                y=spec.y + (float(offset_y) + blur_y) * display_scale,
                font_color=(red, green, blue),
                alpha=layer_alpha,
                outline=False,
                background=None,
                underline=False,
                strikethrough=False,
                zorder=spec.zorder - 0.05,
                gid=(f"{spec.gid}-shadow-{layer_index}" if spec.gid is not None else None),
            )
        )
    return shadow_specs


def _register_svg_hover_text(
    svg_hover_map: Optional[Dict[str, str]],
    gid: str,
    text: str,
) -> None:
    """Register hover text for a native matplotlib text override.

    Parameters
    ----------
    svg_hover_map : dict[str, str] | None
        Optional SVG hover-text accumulator.
    gid : str
        Artist identifier.
    text : str
        Hover text to associate with the artist.

    Returns
    -------
    None
        Mutates ``svg_hover_map`` in place when provided.
    """
    if svg_hover_map is not None:
        svg_hover_map[gid] = text


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


def _cluster_label_text_max_width(
    style: ClusterStyle,
    display_scale: float,
) -> Optional[float]:
    """Return the cluster-label wrap budget in data units.

    Parameters
    ----------
    style : ClusterStyle
        Cluster style providing the optional width budget in points.
    display_scale : float
        Point-to-data conversion for the active axes.

    Returns
    -------
    float | None
        Wrap width in data units, or ``None`` when wrapping is disabled.
    """
    if style.text_max_width is None:
        return None
    return float(style.text_max_width) * display_scale


def _resolve_cluster_label_background(graph: Any, style: ClusterStyle) -> Optional[str]:
    """Resolve the configured cluster-label background color.

    Parameters
    ----------
    graph : Any
        Graph exposing a graph-level background color.
    style : ClusterStyle
        Cluster style that may request an explicit label mask color or the
        ``"@background"`` sentinel.

    Returns
    -------
    str | None
        Concrete Matplotlib color string, or ``None`` when no background mask
        is configured.
    """
    background = str(getattr(style, "label_background", "") or "")
    if not background:
        return None
    if background == "@background":
        return str(graph.graph_style.background_color)
    return background


def _measure_cluster_label_data(
    text: str,
    font_size_data: float,
    font_family: str,
    font_weight: str,
    text_wrap: str,
    text_max_width: Optional[float],
) -> Tuple[float, float]:
    """Measure a cluster label after plain-text wrapping is applied.

    Parameters
    ----------
    text : str
        Raw cluster label text.
    font_size_data : float
        Label font size in data units.
    font_family : str
        Font family used for measurement.
    font_weight : str
        Font weight used for measurement.
    text_wrap : str
        Plain-text wrapping policy.
    text_max_width : float | None
        Optional width budget in data units.

    Returns
    -------
    tuple[float, float]
        Measured ``(width, height)`` in data units.
    """
    prepared_text = prepare_label_text(
        text,
        font_size=font_size_data,
        text_wrap=text_wrap,
        text_max_width=text_max_width,
        text_transform="none",
        label_format="plain",
    )
    return measure_text_data(
        prepared_text,
        size_data=font_size_data,
        font_family=font_family,
        font_weight=font_weight,
    )


def _cluster_label_expands_top(position: str) -> bool:
    """Return whether a cluster label reserves space above the box interior.

    Parameters
    ----------
    position : str
        Cluster label position string.

    Returns
    -------
    bool
        ``True`` when the label sits inside the top band of the cluster.
    """
    return str(position).replace("_", "-").startswith("top-")


def _cluster_label_expands_bottom(position: str) -> bool:
    """Return whether a cluster label reserves space below the box interior.

    Parameters
    ----------
    position : str
        Cluster label position string.

    Returns
    -------
    bool
        ``True`` when the label sits inside the bottom band of the cluster.
    """
    return str(position).replace("_", "-").startswith("bottom-")


def _cluster_label_is_outside(position: str) -> bool:
    """Return whether a cluster label is rendered outside the cluster box.

    Parameters
    ----------
    position : str
        Cluster label position string.

    Returns
    -------
    bool
        ``True`` when the label is outside the cluster box.
    """
    return str(position).replace("_", "-").startswith("outside-")


def _cluster_label_anchor(
    position: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    label_offset_x: float,
    label_offset_y: float,
) -> Tuple[float, float, str, str]:
    """Return the anchor point and alignment for one cluster label.

    Parameters
    ----------
    position : str
        Cluster label position string.
    x_min : float
        Cluster left edge in data units.
    x_max : float
        Cluster right edge in data units.
    y_min : float
        Cluster bottom edge in data units.
    y_max : float
        Cluster top edge in data units.
    label_offset_x : float
        Horizontal label inset in data units.
    label_offset_y : float
        Vertical label inset in data units.

    Returns
    -------
    tuple[float, float, str, str]
        ``(x, y, ha, va)`` anchor metadata.
    """
    normalized_position = str(position).replace("_", "-")
    if normalized_position in {"top-center", "bottom-center"}:
        anchor_x = (x_min + x_max) / 2.0
        ha = "center"
    elif normalized_position in {"top-right", "bottom-right"}:
        anchor_x = x_max - label_offset_x
        ha = "right"
    else:
        anchor_x = x_min + label_offset_x
        ha = "left"

    if normalized_position == "outside-top":
        return anchor_x, y_max + label_offset_y, ha, "bottom"
    if normalized_position == "outside-bottom":
        return anchor_x, y_min - label_offset_y, ha, "top"
    if _cluster_label_expands_bottom(normalized_position):
        return anchor_x, y_min + label_offset_y, ha, "bottom"
    return anchor_x, y_max - label_offset_y, ha, "top"


@dataclass
class _ClusterLabelPlacement:
    """Measured placement metadata for one cluster label.

    Parameters
    ----------
    name : str
        Cluster name used for debugging and grouping.
    spec : DaguaText
        Mutable text specification that will be rendered.
    width : float
        Measured label width in data units.
    height : float
        Measured label height in data units.
    depth : int
        Cluster nesting depth.
    parent_name : str | None
        Parent cluster name. Labels only repel within the same sibling set.
    """

    name: str
    spec: DaguaText
    width: float
    height: float
    depth: int
    parent_name: Optional[str]


@dataclass(frozen=True)
class _ClusterRenderBox:
    """Rendered cluster bounds and padding metadata.

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        Rendered cluster bounds as ``(x_min, y_min, x_max, y_max)``.
    paddings : tuple[float, float, float, float]
        Effective ``(top, right, bottom, left)`` paddings in render data units.
    """

    bbox: Tuple[float, float, float, float]
    paddings: Tuple[float, float, float, float]


def _cluster_side_paddings(style: ClusterStyle, depth: int) -> Tuple[float, float, float, float]:
    """Return effective top, right, bottom, and left cluster paddings.

    Parameters
    ----------
    style : ClusterStyle
        Cluster style with a shared padding and optional side overrides.
    depth : int
        Cluster nesting depth used for the shared-padding depth adjustment.

    Returns
    -------
    tuple[float, float, float, float]
        Effective ``(top, right, bottom, left)`` paddings. Explicit side values
        replace the depth-adjusted shared padding on their corresponding side.
    """

    depth_step = float(getattr(style, "depth_padding_step", -3.0))
    shared = max(float(style.padding) + int(depth) * depth_step, 0.0)
    return tuple(
        shared if value is None else max(float(value), 0.0)
        for value in (
            style.padding_top,
            style.padding_right,
            style.padding_bottom,
            style.padding_left,
        )
    )


def _graphviz_strict_cluster_top_cap(
    ax: Any,
    graph: Any,
    cluster_indices: Sequence[int],
    pos: np.ndarray,
    sizes: np.ndarray,
) -> Optional[float]:
    """Return a strict-theme top cap below external predecessor nodes.

    Parameters
    ----------
    ax : Any
        Matplotlib axes used for point-to-data conversion.
    graph : Any
        Graph exposing ``edge_index`` and theme metadata.
    cluster_indices : sequence[int]
        Leaf node indices contained by the cluster.
    pos : numpy.ndarray
        Node positions with shape ``[N, 2]``.
    sizes : numpy.ndarray
        Node sizes with shape ``[N, 2]``.

    Returns
    -------
    float | None
        Maximum allowed top ``y`` coordinate, or ``None`` when no external
        incoming node needs a clearance cap.
    """
    if not _is_graphviz_strict_render(graph) or str(getattr(graph, "direction", "TB")) != "TB":
        return None
    cluster_set = set(int(index) for index in cluster_indices)
    edge_index = getattr(graph, "edge_index", None)
    if edge_index is None:
        return None
    if hasattr(edge_index, "detach"):
        edges = edge_index.detach().cpu().numpy()
    else:
        edges = np.asarray(edge_index)
    if edges.size == 0:
        return None

    external_sources: List[int] = []
    for edge_offset in range(edges.shape[1]):
        source = int(edges[0, edge_offset])
        target = int(edges[1, edge_offset])
        if target in cluster_set and source not in cluster_set:
            external_sources.append(source)
    if not external_sources:
        return None

    gap = _points_to_data_units(
        ax,
        _GRAPHVIZ_STRICT_CLUSTER_EXTERNAL_NODE_GAP_POINTS,
        "y",
    )
    external_bottoms = [
        float(pos[source, 1]) - float(sizes[source, 1]) / 2.0 for source in external_sources
    ]
    return min(external_bottoms) - gap


def _compute_cluster_y_maxes(
    graph: Any,
    pos: np.ndarray,
    sizes: np.ndarray,
    ordered_clusters: Sequence[str],
    cluster_depths: Dict[str, int],
    label_gap: float = 0.0,
    display_scale: float = 1.0,
) -> Dict[str, float]:
    """Return cluster top bounds after reserving nested header bands.

    Parameters
    ----------
    graph : Any
        Graph exposing cluster membership, labels, parents, and styles.
    pos : numpy.ndarray
        Node positions with shape ``[N, 2]`` in data coordinates.
    sizes : numpy.ndarray
        Node sizes with shape ``[N, 2]`` in data coordinates.
    ordered_clusters : sequence[str]
        Cluster render order.
    cluster_depths : dict[str, int]
        Nesting depth per cluster name.
    label_gap : float, default=0.0
        Extra vertical gap in data units reserved below the cluster label.
    display_scale : float, default=1.0
        Point-to-data conversion used for wrapped label measurement.

    Returns
    -------
    dict[str, float]
        Top ``y`` bound for each cluster after accounting for descendant label
        bands. Parents are computed after children so deep headers remain
        visible inside the axes limits.
    """
    cluster_parents = getattr(graph, "cluster_parents", {}) or {}
    min_node_height = float(sizes[:, 1].min()) if sizes.size else 0.0
    cluster_y_maxes: Dict[str, float] = {}

    for name in reversed(ordered_clusters):
        members = graph.clusters[name]
        indices = collect_cluster_leaves(members) if isinstance(members, dict) else members
        if not indices:
            continue

        style = _cluster_style_for_render(graph, name)
        depth = cluster_depths.get(name, 0)
        padding_top, _, padding_bottom, _ = _cluster_side_paddings(style, 0)
        label_text = graph.cluster_labels.get(name, name)
        member_pos = pos[indices]
        member_sizes = sizes[indices]
        raw_y_max = float((member_pos[:, 1] + member_sizes[:, 1] / 2).max())
        raw_y_min = float((member_pos[:, 1] - member_sizes[:, 1] / 2).min()) - padding_bottom

        for child_name, parent_name in cluster_parents.items():
            if parent_name == name and child_name in cluster_y_maxes:
                raw_y_max = max(raw_y_max, cluster_y_maxes[child_name])

        cluster_height = max(raw_y_max - raw_y_min, 0.0)
        label_font_points = max(
            float(style.font_size) + depth * float(getattr(style, "depth_font_size_step", -0.5)),
            5.0,
        )
        label_height = _measure_cluster_label_data(
            label_text,
            font_size_data=_cluster_font_size_data(
                label_text,
                cluster_height,
                min_node_height,
                label_font_points,
                str(style.font_size_scaling),
                display_scale,
            ),
            font_family=str(style.font_family or RESOLVED_FONT),
            font_weight=str(style.font_weight),
            text_wrap=str(style.text_wrap),
            text_max_width=_cluster_label_text_max_width(style, display_scale),
        )[1]
        label_band = (
            label_height + label_gap
            if _cluster_label_expands_top(str(style.label_position))
            else 0.0
        )
        cluster_y_maxes[name] = raw_y_max + padding_top + label_band

    return cluster_y_maxes


def _cluster_min_render_height(
    ax: Any,
    style: ClusterStyle,
    member_pos: np.ndarray,
    member_sizes: np.ndarray,
    label_height: float,
    label_gap: float,
) -> float:
    """Return the minimum full render-box height for one cluster.

    Parameters
    ----------
    ax : Any
        Matplotlib axes used for point-to-data conversion.
    style : ClusterStyle
        Cluster style controlling label placement.
    member_pos : numpy.ndarray
        Member center positions with shape ``[N, 2]``.
    member_sizes : numpy.ndarray
        Member sizes with shape ``[N, 2]``.
    label_height : float
        Measured label height in data units.
    label_gap : float
        Extra vertical label gap in data units.

    Returns
    -------
    float
        Minimum full bbox height in data units.
    """
    if member_pos.size and member_sizes.size:
        member_tops = member_pos[:, 1] + member_sizes[:, 1] / 2.0
        member_bottoms = member_pos[:, 1] - member_sizes[:, 1] / 2.0
        content_height = float(member_tops.max() - member_bottoms.min())
        min_member_height = float(member_sizes[:, 1].min())
    else:
        content_height = 0.0
        min_member_height = 0.0

    min_inner_height = max(
        content_height,
        min_member_height,
        _points_to_data_units(ax, _MIN_CLUSTER_INNER_HEIGHT_POINTS, "y"),
    )
    label_band = (
        float(label_height) + float(label_gap)
        if _cluster_label_expands_top(str(style.label_position))
        or _cluster_label_expands_bottom(str(style.label_position))
        else 0.0
    )
    return label_band + min_inner_height


def _compute_cluster_render_bboxes(
    ax: Any,
    graph: Any,
    pos: np.ndarray,
    sizes: np.ndarray,
    ordered_clusters: Sequence[str],
    cluster_depths: Dict[str, int],
    cluster_y_mins: Dict[str, float],
    cluster_y_maxes: Dict[str, float],
    display_scale: float,
    cluster_aware: bool,
    label_gap: float = 0.0,
) -> Dict[str, _ClusterRenderBox]:
    """Return final rendered cluster boxes with parent containment enforced.

    Parameters
    ----------
    ax : Any
        Matplotlib axes used for point-to-data conversion.
    graph : Any
        Graph exposing cluster membership, labels, parents, and styles.
    pos : numpy.ndarray
        Node positions with shape ``[N, 2]`` in data coordinates.
    sizes : numpy.ndarray
        Node sizes with shape ``[N, 2]`` in data coordinates.
    ordered_clusters : sequence[str]
        Parent-first cluster render order.
    cluster_depths : dict[str, int]
        Nesting depth per cluster name.
    cluster_y_mins : dict[str, float]
        Precomputed bottom bounds by cluster name.
    cluster_y_maxes : dict[str, float]
        Precomputed top bounds by cluster name.
    display_scale : float
        Point-to-data conversion for text and offsets.
    cluster_aware : bool
        Whether root-level label/min-width expansion is capped to the
        placement footprint.
    label_gap : float, default=0.0
        Extra vertical label gap in data units used to preserve the header
        band when top caps are applied.

    Returns
    -------
    dict[str, _ClusterRenderBox]
        Final boxes keyed by cluster name. Parent boxes are expanded after the
        first pass so every child sits strictly inside its parent padding.
    """
    cluster_parents = getattr(graph, "cluster_parents", {}) or {}
    min_node_height = float(sizes[:, 1].min()) if sizes.size else 0.0
    cap_x = _points_to_data_units(ax, _CLUSTER_RENDER_BBOX_EXTRA_CAP_POINTS, "x")
    boxes: Dict[str, _ClusterRenderBox] = {}

    for name in ordered_clusters:
        members = graph.clusters[name]
        indices = collect_cluster_leaves(members) if isinstance(members, dict) else members
        if not indices:
            continue

        style = _cluster_style_for_render(graph, name)
        depth = cluster_depths.get(name, 0)
        padding_top, padding_right, padding_bottom, padding_left = _cluster_side_paddings(
            style, depth
        )
        padding_top = max(padding_top, 5.0) if style.padding_top is None else padding_top
        padding_right = max(padding_right, 5.0) if style.padding_right is None else padding_right
        padding_bottom = (
            max(padding_bottom, 5.0) if style.padding_bottom is None else padding_bottom
        )
        padding_left = max(padding_left, 5.0) if style.padding_left is None else padding_left
        member_pos = pos[indices]
        member_sizes = sizes[indices]

        x_min = float((member_pos[:, 0] - member_sizes[:, 0] / 2).min() - padding_left)
        x_max = float((member_pos[:, 0] + member_sizes[:, 0] / 2).max() + padding_right)
        base_x_min = x_min
        base_x_max = x_max
        y_min = float(
            cluster_y_mins.get(
                name,
                (member_pos[:, 1] - member_sizes[:, 1] / 2).min() - padding_bottom,
            )
        )
        y_max = float(
            cluster_y_maxes.get(
                name,
                (member_pos[:, 1] + member_sizes[:, 1] / 2).max() + padding_top,
            )
        )

        label = graph.cluster_labels.get(name, name)
        label_font_points = max(
            float(style.font_size) + depth * float(getattr(style, "depth_font_size_step", -0.5)),
            5.0,
        )
        label_font_data = _cluster_font_size_data(
            label,
            float(y_max - y_min),
            min_node_height,
            label_font_points,
            str(style.font_size_scaling),
            display_scale,
        )
        label_width, label_height = _measure_cluster_label_data(
            label,
            font_size_data=label_font_data,
            font_family=str(style.font_family or RESOLVED_FONT),
            font_weight=str(style.font_weight),
            text_wrap=str(style.text_wrap),
            text_max_width=_cluster_label_text_max_width(style, display_scale),
        )
        top_cap = _graphviz_strict_cluster_top_cap(ax, graph, indices, pos, sizes)
        if top_cap is not None:
            y_max = min(float(y_max), top_cap)

        min_cluster_height = _cluster_min_render_height(
            ax,
            style,
            member_pos,
            member_sizes,
            label_height,
            label_gap,
        )
        if y_max - y_min < min_cluster_height:
            y_max = y_min + min_cluster_height

        cluster_height = y_max - y_min
        cluster_width = x_max - x_min
        min_cluster_width = cluster_height * 0.65
        if cluster_width < min_cluster_width:
            expand_w = (min_cluster_width - cluster_width) / 2.0
            x_min -= expand_w
            x_max += expand_w

        if not _cluster_label_is_outside(str(style.label_position)):
            label_ox = float(style.label_offset[0]) * display_scale
            est_label_width = label_width + label_ox * 2.0
            content_width = x_max - x_min
            if est_label_width > content_width:
                expand = (est_label_width - content_width) / 2.0
                x_min -= expand
                x_max += expand

        if cluster_aware and cluster_parents.get(name) is None:
            x_min = max(float(x_min), float(base_x_min) - cap_x)
            x_max = min(float(x_max), float(base_x_max) + cap_x)

        boxes[name] = _ClusterRenderBox(
            bbox=(float(x_min), float(y_min), float(x_max), float(y_max)),
            paddings=(padding_top, padding_right, padding_bottom, padding_left),
        )

    for child_name in reversed(ordered_clusters):
        child = boxes.get(child_name)
        parent_name = cluster_parents.get(child_name)
        parent = boxes.get(parent_name) if parent_name is not None else None
        if child is None or parent is None:
            continue
        px_min, py_min, px_max, py_max = parent.bbox
        cx_min, cy_min, cx_max, cy_max = child.bbox
        parent_top, parent_right, parent_bottom, parent_left = parent.paddings
        child_top, child_right, child_bottom, child_left = child.paddings
        boxes[parent_name] = _ClusterRenderBox(
            bbox=(
                min(px_min, cx_min - max(min(parent_left, child_left), 1e-6)),
                min(py_min, cy_min - max(min(parent_bottom, child_bottom), 1e-6)),
                max(px_max, cx_max + max(min(parent_right, child_right), 1e-6)),
                max(py_max, cy_max + max(min(parent_top, child_top), 1e-6)),
            ),
            paddings=parent.paddings,
        )

    return boxes


def _compute_cluster_y_mins(
    graph: Any,
    pos: np.ndarray,
    sizes: np.ndarray,
    ordered_clusters: Sequence[str],
    cluster_depths: Dict[str, int],
    label_gap: float = 0.0,
    display_scale: float = 1.0,
) -> Dict[str, float]:
    """Return cluster bottom bounds after reserving nested footer bands.

    Parameters
    ----------
    graph : Any
        Graph exposing cluster membership, labels, parents, and styles.
    pos : numpy.ndarray
        Node positions with shape ``[N, 2]`` in data coordinates.
    sizes : numpy.ndarray
        Node sizes with shape ``[N, 2]`` in data coordinates.
    ordered_clusters : sequence[str]
        Cluster render order.
    cluster_depths : dict[str, int]
        Nesting depth per cluster name.
    label_gap : float, default=0.0
        Extra vertical gap in data units reserved above the cluster label.
    display_scale : float, default=1.0
        Point-to-data conversion used for wrapped label measurement.

    Returns
    -------
    dict[str, float]
        Bottom ``y`` bound for each cluster after accounting for descendant
        label bands.
    """
    cluster_parents = getattr(graph, "cluster_parents", {}) or {}
    min_node_height = float(sizes[:, 1].min()) if sizes.size else 0.0
    cluster_y_mins: Dict[str, float] = {}

    for name in reversed(ordered_clusters):
        members = graph.clusters[name]
        indices = collect_cluster_leaves(members) if isinstance(members, dict) else members
        if not indices:
            continue

        style = _cluster_style_for_render(graph, name)
        depth = cluster_depths.get(name, 0)
        padding_top, _, padding_bottom, _ = _cluster_side_paddings(style, 0)
        label_text = graph.cluster_labels.get(name, name)
        member_pos = pos[indices]
        member_sizes = sizes[indices]
        raw_y_max = float((member_pos[:, 1] + member_sizes[:, 1] / 2).max()) + padding_top
        raw_y_min = float((member_pos[:, 1] - member_sizes[:, 1] / 2).min())

        for child_name, parent_name in cluster_parents.items():
            if parent_name == name and child_name in cluster_y_mins:
                raw_y_min = min(raw_y_min, cluster_y_mins[child_name])

        cluster_height = max(raw_y_max - raw_y_min, 0.0)
        label_font_points = max(
            float(style.font_size) + depth * float(getattr(style, "depth_font_size_step", -0.5)),
            5.0,
        )
        label_height = _measure_cluster_label_data(
            label_text,
            font_size_data=_cluster_font_size_data(
                label_text,
                cluster_height,
                min_node_height,
                label_font_points,
                str(style.font_size_scaling),
                display_scale,
            ),
            font_family=str(style.font_family or RESOLVED_FONT),
            font_weight=str(style.font_weight),
            text_wrap=str(style.text_wrap),
            text_max_width=_cluster_label_text_max_width(style, display_scale),
        )[1]
        label_band = (
            label_height + label_gap
            if _cluster_label_expands_bottom(str(style.label_position))
            else 0.0
        )
        cluster_y_mins[name] = raw_y_min - padding_bottom - label_band

    return cluster_y_mins


def _expand_axes_for_clusters(
    ax: Any,
    graph: Any,
    pos: np.ndarray,
    sizes: np.ndarray,
    margin: float,
    cluster_aware: bool = True,
) -> None:
    """Expand axes limits so cluster geometry fits after display scaling.

    Parameters
    ----------
    ax : Any
        Matplotlib axes whose limits will be updated in place.
    graph : Any
        Graph exposing cluster membership, labels, and styles.
    pos : numpy.ndarray
        Node positions with shape ``[N, 2]`` in data coordinates.
    sizes : numpy.ndarray
        Node sizes with shape ``[N, 2]`` in data coordinates.
    margin : float
        Extra outer padding reserved around the cluster bounds.
    cluster_aware : bool, default=True
        Whether render-time root bbox caps should match the cluster-aware
        renderer mode.

    Returns
    -------
    None
        Mutates ``ax`` in place.
    """
    if not graph.clusters:
        return

    ordered_clusters = _cluster_render_order(graph)
    cluster_depths = _cluster_depths(graph, ordered_clusters)
    min_node_height = float(sizes[:, 1].min()) if sizes.size else 0.0

    # The point-to-data conversion depends on the axes limits themselves, so a
    # short second pass keeps deep cluster headers from being clipped after the
    # first expansion changes the display scale.
    for _ in range(2):
        display_scale = _compute_display_scale(ax)
        cluster_y_maxes = _compute_cluster_y_maxes(
            graph,
            pos,
            sizes,
            ordered_clusters,
            cluster_depths,
            label_gap=_points_to_data_units(ax, _CLUSTER_LABEL_VERTICAL_GAP_POINTS, "y"),
            display_scale=display_scale,
        )
        cluster_y_mins = _compute_cluster_y_mins(
            graph,
            pos,
            sizes,
            ordered_clusters,
            cluster_depths,
            label_gap=_points_to_data_units(ax, _CLUSTER_LABEL_VERTICAL_GAP_POINTS, "y"),
            display_scale=display_scale,
        )
        cluster_render_boxes = _compute_cluster_render_bboxes(
            ax,
            graph,
            pos,
            sizes,
            ordered_clusters,
            cluster_depths,
            cluster_y_mins,
            cluster_y_maxes,
            display_scale,
            cluster_aware=cluster_aware,
            label_gap=_points_to_data_units(ax, _CLUSTER_LABEL_VERTICAL_GAP_POINTS, "y"),
        )
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        for name in ordered_clusters:
            members = graph.clusters[name]
            indices = collect_cluster_leaves(members) if isinstance(members, dict) else members
            if not indices:
                continue

            style = _cluster_style_for_render(graph, name)
            depth = cluster_depths.get(name, 0)
            render_box = cluster_render_boxes.get(name)
            if render_box is None:
                continue
            cx_min, cy_min, cx_max, cy_max = render_box.bbox

            label = graph.cluster_labels.get(name, name)
            label_font_points = max(
                float(style.font_size)
                + depth * float(getattr(style, "depth_font_size_step", -0.5)),
                5.0,
            )
            cluster_height = max(cy_max - cy_min, 0.0)
            label_font_data = _cluster_font_size_data(
                label,
                float(cluster_height),
                min_node_height,
                label_font_points,
                str(style.font_size_scaling),
                display_scale,
            )
            label_width, label_height = _measure_cluster_label_data(
                label,
                font_size_data=label_font_data,
                font_family=str(style.font_family or RESOLVED_FONT),
                font_weight=str(style.font_weight),
                text_wrap=str(style.text_wrap),
                text_max_width=_cluster_label_text_max_width(style, display_scale),
            )

            label_offset_x = float(style.label_offset[0]) * display_scale
            label_offset_y = float(style.label_offset[1]) * display_scale

            x_min = min(x_min, float(cx_min) - margin)
            x_max = max(x_max, float(cx_max) + margin)
            y_min = min(y_min, float(cy_min) - margin)
            y_max = max(y_max, float(cy_max) + margin)
            if _cluster_label_is_outside(str(style.label_position)):
                label_x, label_y, label_ha, label_va = _cluster_label_anchor(
                    str(style.label_position),
                    float(cx_min),
                    float(cx_max),
                    float(cy_min),
                    float(cy_max),
                    label_offset_x,
                    label_offset_y,
                )
                label_bounds = _cluster_label_bounds(
                    DaguaText(
                        x=label_x,
                        y=label_y,
                        text=label,
                        ha=label_ha,
                        va=label_va,
                    ),
                    label_width,
                    label_height,
                )
                x_min = min(x_min, label_bounds[0] - margin)
                x_max = max(x_max, label_bounds[2] + margin)
                y_min = min(y_min, label_bounds[1] - margin)
                y_max = max(y_max, label_bounds[3] + margin)

        ax.set_xlim(float(x_min), float(x_max))
        ax.set_ylim(float(y_min), float(y_max))


def _cluster_label_bounds(
    spec: DaguaText,
    width: float,
    height: float,
) -> Tuple[float, float, float, float]:
    """Return a cluster label bbox in data coordinates.

    Parameters
    ----------
    spec : DaguaText
        Render specification whose anchor and alignment determine the box.
    width : float
        Measured label width in data units.
    height : float
        Measured label height in data units.

    Returns
    -------
    tuple[float, float, float, float]
        Bounding box as ``(x_min, y_min, x_max, y_max)``.
    """

    if spec.ha == "right":
        x_min = spec.x - width
        x_max = spec.x
    elif spec.ha == "center":
        half_width = width / 2.0
        x_min = spec.x - half_width
        x_max = spec.x + half_width
    else:
        x_min = spec.x
        x_max = spec.x + width

    if spec.va == "bottom":
        y_min = spec.y
        y_max = spec.y + height
    elif spec.va == "center":
        half_height = height / 2.0
        y_min = spec.y - half_height
        y_max = spec.y + half_height
    else:
        y_min = spec.y - height
        y_max = spec.y

    return (x_min, y_min, x_max, y_max)


def _resolve_cluster_label_collisions(
    ax: Any,
    placements: Sequence[_ClusterLabelPlacement],
) -> None:
    """Nudge sibling cluster labels downward until measured boxes no longer overlap.

    Parameters
    ----------
    ax : Any
        Matplotlib axes used to convert the vertical gap into data units.
    placements : sequence[_ClusterLabelPlacement]
        Mutable placement metadata for cluster labels.

    Returns
    -------
    None
        Updates ``placement.spec.y`` in place for labels that need separation.
    """

    if len(placements) < 2:
        return

    sibling_groups: Dict[Tuple[int, Optional[str]], List[_ClusterLabelPlacement]] = defaultdict(
        list
    )
    for placement in placements:
        sibling_groups[(placement.depth, placement.parent_name)].append(placement)

    vertical_gap = _points_to_data_units(ax, _CLUSTER_LABEL_VERTICAL_GAP_POINTS, "y")
    horizontal_gap = _points_to_data_units(
        ax,
        _GRAPHVIZ_STRICT_CLUSTER_HORIZONTAL_SEPARATION_POINTS,
        "x",
    )
    for group in sibling_groups.values():
        if len(group) < 2:
            continue

        group.sort(
            key=lambda placement: _cluster_label_bounds(
                placement.spec,
                placement.width,
                placement.height,
            )[0]
        )
        placed_bounds: List[Tuple[float, float, float, float]] = []

        for placement in group:
            current_bounds = _cluster_label_bounds(
                placement.spec,
                placement.width,
                placement.height,
            )
            while True:
                overlapping_bounds = [
                    bounds
                    for bounds in placed_bounds
                    if current_bounds[0] - horizontal_gap < bounds[2]
                    and current_bounds[2] + horizontal_gap > bounds[0]
                    and current_bounds[1] < bounds[3]
                    and current_bounds[3] > bounds[1]
                ]
                if not overlapping_bounds:
                    break

                # Keep later siblings inside their own cluster header band by
                # only moving them downward, never sideways into another box.
                placement.spec.y = min(bounds[1] - vertical_gap for bounds in overlapping_bounds)
                current_bounds = _cluster_label_bounds(
                    placement.spec,
                    placement.width,
                    placement.height,
                )

            placed_bounds.append(current_bounds)


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
    stroke_scale = _effective_stroke_scale(ax)
    clip_patches: List[Any] = []
    fill_paths: List[Any] = []
    fill_colors: List[Any] = []
    border_paths: List[Any] = []
    border_colors: List[Any] = []

    for i in range(graph.num_nodes):
        x, y = float(pos[i, 0]), float(pos[i, 1])
        w, h = float(sizes[i, 0]), float(sizes[i, 1])
        style = _node_style_for_render(graph, i)
        scaled_style = _scaled_node_style(style, display_scale)
        corner_radius = _node_corner_radius_data(style, display_scale, w, h)
        stroke_override = getattr(style, "stroke_width_override_points", None)
        if stroke_override is None:
            border_width = clamp_border_width(float(style.stroke_width) * stroke_scale, w, h)
        else:
            border_width = 0.0
        border_position = _normalize_border_position(getattr(style, "border_position", "center"))
        shape_spec = ShapeSpec(
            center_x=x,
            center_y=y,
            width=w,
            height=h,
            shape=str(style.shape),
            corner_radius=corner_radius,
            aspect_ratio=style.aspect_ratio,
            polygon_points=style.polygon_points,
        )
        outer_path = build_shape_path(shape_spec)
        fill_path = _node_fill_path(shape_spec, outer_path, border_width, border_position)

        if style.shadow:
            _draw_shadow(ax, x, y, w, h, scaled_style, corner_radius)

        facecolor = to_rgba(style.fill, _node_fill_alpha(style))
        edgecolor = to_rgba(style.stroke, style.opacity * style.border_opacity)
        # For non-convex shapes (star), clip text to the bounding
        # rectangle instead of the shape path so glyphs aren't cut by
        # interior concavities.
        _NONCONVEX_SHAPES = {
            "star",
            "promoter",
            "terminator",
            "ribosite",
            "proteasesite",
            "assembly",
            "insulator",
        }
        if style.shape in _NONCONVEX_SHAPES:
            from matplotlib.path import Path as MplPath

            rect_path = MplPath(
                [
                    [x - w / 2, y - h / 2],
                    [x + w / 2, y - h / 2],
                    [x + w / 2, y + h / 2],
                    [x - w / 2, y + h / 2],
                    [x - w / 2, y - h / 2],
                ],
            )
            clip_patch = make_clip_proxy(rect_path, ax.transData)
        else:
            clip_patch = make_clip_proxy(fill_path, ax.transData)
        fill_clip_patch = make_clip_proxy(fill_path, ax.transData)
        image_clip_patch = make_clip_proxy(outer_path, ax.transData)
        if _requires_custom_node_rendering(style):
            _draw_node_fill(
                ax,
                shape_spec,
                fill_path,
                fill_clip_patch,
                x,
                y,
                w,
                h,
                style,
                facecolor,
            )
            _draw_image_node(ax, shape_spec, style, image_clip_patch)
            if stroke_override is not None and float(stroke_override) > 0.0 and edgecolor[-1] > 0.0:
                border_path = _node_border_centerline_path(
                    shape_spec,
                    outer_path,
                    float(stroke_override) * display_scale,
                    border_position,
                )
                _draw_display_point_path_stroke(
                    ax,
                    border_path,
                    edgecolor,
                    float(stroke_override),
                    zorder=2.05,
                    linestyle=_display_point_linestyle(
                        style.stroke_dash,
                        style.stroke_dash_pattern,
                    ),
                    capstyle=str(getattr(style, "stroke_cap", "butt")),
                    joinstyle=str(getattr(style, "stroke_join", "miter")),
                )
                if int(style.border_count) >= 2:
                    inner_path = inset_shape_path(
                        shape_spec,
                        float(stroke_override) * display_scale * _DOUBLE_BORDER_INSET_FACTOR,
                    )
                    _draw_display_point_path_stroke(
                        ax,
                        inner_path,
                        edgecolor,
                        float(stroke_override),
                        zorder=2.05,
                        linestyle=_display_point_linestyle(
                            style.stroke_dash,
                            style.stroke_dash_pattern,
                        ),
                        capstyle=str(getattr(style, "stroke_cap", "butt")),
                        joinstyle=str(getattr(style, "stroke_join", "miter")),
                    )
            elif border_width > 0.0 and edgecolor[-1] > 0.0:
                border_path = _node_border_centerline_path(
                    shape_spec,
                    outer_path,
                    border_width,
                    border_position,
                )
                _draw_node_border_path(
                    ax,
                    border_path,
                    style,
                    edgecolor,
                    border_width,
                    display_scale,
                )
                if int(style.border_count) >= 2:
                    inner_path = inset_shape_path(
                        shape_spec,
                        border_width * _DOUBLE_BORDER_INSET_FACTOR,
                    )
                    _draw_node_border_path(
                        ax,
                        inner_path,
                        style,
                        edgecolor,
                        border_width,
                        display_scale,
                    )
        else:
            if style.gradient == "none":
                fill_paths.append(fill_path)
                fill_colors.append(facecolor)
            elif _node_fill_alpha(style) > 0.0:
                _draw_gradient_fill(ax, clip_patch, x, y, w, h, style)

            _draw_image_node(ax, shape_spec, style, image_clip_patch)

            if stroke_override is not None and float(stroke_override) > 0.0 and edgecolor[-1] > 0.0:
                centerline_path = _node_border_centerline_path(
                    shape_spec,
                    outer_path,
                    float(stroke_override) * display_scale,
                    border_position,
                )
                _draw_display_point_path_stroke(
                    ax,
                    centerline_path,
                    edgecolor,
                    float(stroke_override),
                    zorder=2.05,
                    linestyle=_display_point_linestyle(
                        style.stroke_dash,
                        style.stroke_dash_pattern,
                    ),
                    capstyle=str(getattr(style, "stroke_cap", "butt")),
                    joinstyle=str(getattr(style, "stroke_join", "miter")),
                )
                if int(style.border_count) >= 2:
                    inner_path = inset_shape_path(
                        shape_spec,
                        float(stroke_override) * display_scale * _DOUBLE_BORDER_INSET_FACTOR,
                    )
                    _draw_display_point_path_stroke(
                        ax,
                        inner_path,
                        edgecolor,
                        float(stroke_override),
                        zorder=2.05,
                        linestyle=_display_point_linestyle(
                            style.stroke_dash,
                            style.stroke_dash_pattern,
                        ),
                        capstyle=str(getattr(style, "stroke_cap", "butt")),
                        joinstyle=str(getattr(style, "stroke_join", "miter")),
                    )
            elif border_width > 0.0 and edgecolor[-1] > 0.0:
                if style.stroke_dash == "solid" and style.stroke_dash_pattern is None:
                    if border_position == "center":
                        border_outer_path, border_inner_path = _solid_border_ring_paths(
                            shape_spec,
                            outer_path,
                            border_width,
                            border_position,
                        )
                        border_paths.append(annular_path(border_outer_path, border_inner_path))
                        border_colors.append(edgecolor)
                        if int(style.border_count) >= 2:
                            inner_path = inset_shape_path(
                                shape_spec,
                                border_width * _DOUBLE_BORDER_INSET_FACTOR,
                            )
                            inner_outer_path, inner_inner_path = _solid_border_ring_paths(
                                shape_spec,
                                inner_path,
                                border_width,
                                border_position,
                            )
                            border_paths.append(annular_path(inner_outer_path, inner_inner_path))
                            border_colors.append(edgecolor)
                    else:
                        border_outer_path, border_inner_path = _solid_border_ring_paths(
                            shape_spec,
                            outer_path,
                            border_width,
                            border_position,
                        )
                        border_paths.append(annular_path(border_outer_path, border_inner_path))
                        border_colors.append(edgecolor)
                        if int(style.border_count) >= 2:
                            inner_path = inset_shape_path(
                                shape_spec,
                                border_width * _DOUBLE_BORDER_INSET_FACTOR,
                            )
                            inner_outer_path, inner_inner_path = _solid_border_ring_paths(
                                shape_spec,
                                inner_path,
                                border_width,
                                border_position,
                            )
                            border_paths.append(annular_path(inner_outer_path, inner_inner_path))
                            border_colors.append(edgecolor)
                else:
                    centerline_path = _node_border_centerline_path(
                        shape_spec,
                        outer_path,
                        border_width,
                        border_position,
                    )
                    dash_pattern = _node_border_pattern(style, display_scale)
                    ribbons = dash_ribbon_paths(centerline_path, dash_pattern, border_width)
                    border_paths.extend(ribbons)
                    border_colors.extend([edgecolor] * len(ribbons))

        effective_border_width = (
            max(float(stroke_override), 0.0) * display_scale
            if stroke_override is not None
            else border_width
        )
        _draw_node_outline(
            ax,
            i,
            shape_spec,
            style,
            effective_border_width,
            border_position,
            display_scale,
        )

        clip_patches.append(clip_patch)
        _set_svg_hover(clip_patch, f"dagua-node-{i}", graph.node_labels[i], svg_hover_map)
        if bool(getattr(style, "bevel", False)) and (
            float(getattr(style, "bevel_intensity", 0.0)) > 0.0
        ):
            _draw_node_bevel(ax, i, shape_spec, style)
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


def _draw_shadow(
    ax: Any,
    x: float,
    y: float,
    w: float,
    h: float,
    style: Any,
    corner_radius: Any,
) -> None:
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
    corner_radius : Any
        Scalar or per-corner node radius in data units.
    """
    from matplotlib.colors import to_rgba
    from matplotlib.patches import PathPatch

    ox, oy = style.shadow_offset
    base_r, base_g, base_b, base_a = to_rgba(style.shadow_color)
    steps = 1 if style.shadow_blur <= 0 else min(max(int(np.ceil(style.shadow_blur)), 2), 6)
    base_shape_spec = ShapeSpec(
        center_x=x,
        center_y=y,
        width=w,
        height=h,
        shape=str(style.shape),
        corner_radius=corner_radius,
        aspect_ratio=getattr(style, "aspect_ratio", None),
        polygon_points=getattr(style, "polygon_points", None),
    )
    for idx in range(steps, 0, -1):
        scale = 1.0 + (0.01 * style.shadow_blur * idx)
        alpha = base_a / (idx + 1) if steps > 1 else base_a
        shadow_spec = ShapeSpec(
            center_x=float(base_shape_spec.center_x) + ox,
            center_y=float(base_shape_spec.center_y) + oy,
            width=float(base_shape_spec.width) * scale,
            height=float(base_shape_spec.height) * scale,
            shape=base_shape_spec.shape,
            corner_radius=scale_corner_radius(base_shape_spec.corner_radius, scale),
            aspect_ratio=base_shape_spec.aspect_ratio,
            polygon_points=base_shape_spec.polygon_points,
        )
        shadow = PathPatch(
            build_shape_path(shadow_spec),
            facecolor=(base_r, base_g, base_b, alpha),
            edgecolor="none",
            linewidth=0.0,
            zorder=1.4 - idx * 0.01,
        )
        ax.add_patch(shadow)


def _rectangle_path(x_min: float, x_max: float, y_min: float, y_max: float) -> Any:
    """Return one closed rectangular path in data coordinates.

    Parameters
    ----------
    x_min : float
        Left edge.
    x_max : float
        Right edge.
    y_min : float
        Bottom edge.
    y_max : float
        Top edge.

    Returns
    -------
    Any
        Matplotlib ``Path`` instance describing the rectangle.
    """

    from matplotlib.path import Path

    return Path(
        [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
            [x_min, y_min],
        ],
        [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY],
    )


def _draw_node_bevel(
    ax: Any,
    node_idx: int,
    shape_spec: ShapeSpec,
    style: Any,
) -> None:
    """Draw clipped bevel bands that follow the node's outer contour.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    node_idx : int
        Node index used to tag the bevel artists.
    shape_spec : ShapeSpec
        Node geometry in data coordinates.
    style : Any
        Node style object exposing bevel settings.
    """

    from matplotlib.patches import PathPatch

    intensity = min(max(float(getattr(style, "bevel_intensity", 0.5)), 0.0), 1.0)
    if intensity <= 0.0 or shape_spec.width <= 0.0 or shape_spec.height <= 0.0:
        return

    alpha_scale = intensity / _BEVEL_REFERENCE_INTENSITY
    max_inset = min(shape_spec.width, shape_spec.height) * _BEVEL_MAX_INSET_FRACTION
    if max_inset <= 0.0:
        return
    inset_step = max_inset / _BEVEL_BAND_COUNT
    left = shape_spec.center_x - shape_spec.width / 2.0
    right = shape_spec.center_x + shape_spec.width / 2.0
    top = shape_spec.center_y + shape_spec.height / 2.0
    bottom = shape_spec.center_y - shape_spec.height / 2.0

    highlight_clip = make_clip_proxy(
        _rectangle_path(left, right, shape_spec.center_y, top),
        ax.transData,
    )
    shadow_clip = make_clip_proxy(
        _rectangle_path(left, right, bottom, shape_spec.center_y),
        ax.transData,
    )
    outer_band_path = build_shape_path(shape_spec)

    for band_idx in range(_BEVEL_BAND_COUNT):
        inset_end = inset_step * float(band_idx + 1)
        inner_band_path = inset_shape_path(shape_spec, inset_end)
        band_path = annular_path(outer_band_path, inner_band_path)
        fade = 1.0 - (float(band_idx) / float(_BEVEL_BAND_COUNT))
        highlight_alpha = min(_BEVEL_HIGHLIGHT_ALPHA * alpha_scale * fade, 1.0)
        shadow_alpha = min(_BEVEL_SHADOW_ALPHA * alpha_scale * fade, 1.0)

        highlight_patch = PathPatch(
            band_path,
            facecolor=(1.0, 1.0, 1.0, highlight_alpha),
            edgecolor="none",
            linewidth=0.0,
            zorder=2.03,
            gid=f"dagua-node-bevel-highlight-{node_idx}-{band_idx}",
        )
        highlight_patch.set_clip_path(highlight_clip)
        ax.add_patch(highlight_patch)

        shadow_patch = PathPatch(
            band_path,
            facecolor=(0.0, 0.0, 0.0, shadow_alpha),
            edgecolor="none",
            linewidth=0.0,
            zorder=2.03,
            gid=f"dagua-node-bevel-shadow-{node_idx}-{band_idx}",
        )
        shadow_patch.set_clip_path(shadow_clip)
        ax.add_patch(shadow_patch)
        outer_band_path = inner_band_path


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


def _effective_stroke_scale(ax: Any) -> float:
    """Return the backend-calibrated point-to-data scale for stroke ribbons.

    Parameters
    ----------
    ax : Any
        Matplotlib axes with the resolved Dagua backend name stashed by
        :func:`_new_figure_axes`.

    Returns
    -------
    float
        Point-to-data scale multiplied by the backend's stroke-width
        calibration factor.

    Notes
    -----
    Sprint B Round 3 keeps the authored style width and optimizer geometry
    unchanged, but cairo needs a small render-time ribbon-width nudge to match
    Agg / graphviz effective stroke ink density.
    """
    backend_name = str(getattr(ax, "_dagua_backend_name", "agg"))
    return _compute_display_scale(ax) * stroke_width_scale_for(backend_name)


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


def _resolved_marker_dimensions(
    ax: Any,
    style: Any,
    node_width: float,
    node_height: float,
    *,
    is_self_loop: bool,
    scale_with_edge_width: bool,
) -> Tuple[float, float]:
    """Resolve one terminal's marker dimensions in data units.

    Parameters
    ----------
    ax : Any
        Matplotlib axes used for point-to-data conversion.
    style : Any
        Edge style object providing arrowhead sizing fields.
    node_width : float
        Width of the connected node in data units.
    node_height : float
        Height of the connected node in data units.
    is_self_loop : bool
        Whether the edge source and target are the same node.
    scale_with_edge_width : bool
        Whether to apply the renderer's sublinear edge-width scaling before
        converting the style dimensions into data units. The custom collection
        keeps this enabled so thick ribbons receive proportionally larger
        terminals; the legacy direct-marker path leaves it disabled for
        backward-compatible stroke sizing.

    Returns
    -------
    tuple[float, float]
        Resolved ``(length, width)`` in data units.
    """

    length_points = float(style.arrow_length)
    width_points = float(style.arrow_width)
    arrowsize = max(float(getattr(style, "arrowsize", 1.0)), 0.0)
    length_points *= arrowsize
    width_points *= arrowsize
    if scale_with_edge_width:
        length_points, width_points = _scaled_arrowhead_dimensions(
            length_points,
            width_points,
            float(style.width),
        )

    length_data, width_data = _marker_data_size(
        ax,
        style,
        length_points,
        width_points,
        node_height=node_height,
    )

    if is_self_loop:
        node_min_dimension = min(max(node_width, 0.0), max(node_height, 0.0))
        max_terminal_extent = node_min_dimension * _SELF_LOOP_ARROWHEAD_MAX_NODE_FRACTION
        if max_terminal_extent > 0.0:
            # Self-loops need a stricter cap than normal edges so the head does
            # not eclipse the loop body or read as part of the node silhouette.
            length_data = min(length_data, max_terminal_extent)
            width_data = min(
                width_data,
                max_terminal_extent * _SELF_LOOP_ARROWHEAD_MAX_WIDTH_RATIO,
            )

    return length_data, width_data


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
    if float(width_points) <= 0.0:
        return 0.0
    width_x = _points_to_data_units(ax, width_points, "x")
    width_y = _points_to_data_units(ax, width_points, "y")
    width = min(width_x, width_y)
    # Keep the edge body in the data-coordinate renderer while preventing
    # sub-pixel underflow on small fixed gallery canvases. The authored style
    # width remains unchanged; only the emitted render ribbon receives a floor.
    min_visible_width = _minimum_visible_edge_width_data_units(ax)
    return max(width, min_visible_width, 1e-6)


def _minimum_visible_edge_width_data_units(ax: Any) -> float:
    """Return the configured visible edge-width floor in data units.

    Parameters
    ----------
    ax : Any
        Matplotlib axes with established limits and aspect ratio.

    Returns
    -------
    float
        Data-coordinate width corresponding to ``_MIN_VISIBLE_STROKE_POINTS``.
    """
    return _compute_display_scale(ax) * _MIN_VISIBLE_STROKE_POINTS


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


def _polyline_arc_fraction_for_point(points: np.ndarray, point: Tuple[float, float]) -> float:
    """Return the normalized arc-length fraction for a point on a polyline.

    Parameters
    ----------
    points : numpy.ndarray
        Polyline vertices with shape ``[N, 2]``.
    point : tuple[float, float]
        Point lying on or near one polyline segment.

    Returns
    -------
    float
        Normalized arc-length fraction in ``[0, 1]``.
    """
    if points.shape[0] < 2:
        return 0.0
    target = np.asarray(point, dtype=float)
    deltas = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    total_length = float(segment_lengths.sum())
    if total_length <= 1e-9:
        return 0.0

    best_length = 0.0
    best_distance = float("inf")
    traversed = 0.0
    for index, segment_length in enumerate(segment_lengths):
        start = points[index]
        delta = deltas[index]
        if float(segment_length) <= 1e-9:
            continue
        local_t = float(np.dot(target - start, delta) / (segment_length * segment_length))
        local_t = min(max(local_t, 0.0), 1.0)
        projection = start + delta * local_t
        distance = float(np.linalg.norm(projection - target))
        if distance < best_distance:
            best_distance = distance
            best_length = traversed + float(segment_length) * local_t
        traversed += float(segment_length)
    return min(max(best_length / total_length, 0.0), 1.0)


def _point_inside_bbox(point: np.ndarray, bbox: Tuple[float, float, float, float]) -> bool:
    """Return whether a point lies inside a cluster bbox.

    Parameters
    ----------
    point : numpy.ndarray
        Point with shape ``[2]``.
    bbox : tuple[float, float, float, float]
        Rectangle bounds as ``(x_min, y_min, x_max, y_max)``.

    Returns
    -------
    bool
        ``True`` when the point is inside or on the rectangle boundary.
    """
    x_min, y_min, x_max, y_max = bbox
    return bool(x_min <= float(point[0]) <= x_max and y_min <= float(point[1]) <= y_max)


def _transition_t(
    points: np.ndarray,
    index: int,
    bbox: Tuple[float, float, float, float],
) -> float:
    """Approximate the parametric fraction where a sample segment crosses a bbox.

    Parameters
    ----------
    points : numpy.ndarray
        Uniformly parameterized samples with shape ``[N, 2]``.
    index : int
        Segment start index.
    bbox : tuple[float, float, float, float]
        Rectangle bounds as ``(x_min, y_min, x_max, y_max)``.

    Returns
    -------
    float
        Approximate curve parameter in ``[0, 1]`` at the perimeter crossing.
    """
    start = points[index]
    end = points[index + 1]
    x_min, y_min, x_max, y_max = bbox
    candidates: List[float] = []
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    if abs(dx) > 1e-12:
        candidates.extend([(x_min - float(start[0])) / dx, (x_max - float(start[0])) / dx])
    if abs(dy) > 1e-12:
        candidates.extend([(y_min - float(start[1])) / dy, (y_max - float(start[1])) / dy])
    for alpha in sorted(value for value in candidates if 0.0 <= value <= 1.0):
        probe = start + (end - start) * alpha
        if _point_inside_bbox(probe, bbox):
            return (float(index) + float(alpha)) / float(points.shape[0] - 1)
    return float(index + 0.5) / float(points.shape[0] - 1)


def _source_exit_t(points: np.ndarray, bbox: Tuple[float, float, float, float]) -> Optional[float]:
    """Return the first exit parameter from a source-side cluster.

    Parameters
    ----------
    points : numpy.ndarray
        Uniformly parameterized curve samples with shape ``[N, 2]``.
    bbox : tuple[float, float, float, float]
        Cluster rectangle bounds.

    Returns
    -------
    float | None
        Exit parameter, or ``None`` when the sampled curve does not start
        inside and leave the rectangle.
    """
    if not _point_inside_bbox(points[0], bbox):
        return None
    for index in range(points.shape[0] - 1):
        if _point_inside_bbox(points[index], bbox) and not _point_inside_bbox(
            points[index + 1], bbox
        ):
            return _transition_t(points, index, bbox)
    return None


def _target_entry_t(points: np.ndarray, bbox: Tuple[float, float, float, float]) -> Optional[float]:
    """Return the last entry parameter into a target-side cluster.

    Parameters
    ----------
    points : numpy.ndarray
        Uniformly parameterized curve samples with shape ``[N, 2]``.
    bbox : tuple[float, float, float, float]
        Cluster rectangle bounds.

    Returns
    -------
    float | None
        Entry parameter, or ``None`` when the sampled curve does not enter and
        end inside the rectangle.
    """
    if not _point_inside_bbox(points[-1], bbox):
        return None
    for index in range(points.shape[0] - 2, -1, -1):
        if not _point_inside_bbox(points[index], bbox) and _point_inside_bbox(
            points[index + 1], bbox
        ):
            return _transition_t(points, index, bbox)
    return None


def _foreign_inside_intervals(
    points: np.ndarray,
    bbox: Tuple[float, float, float, float],
    t0: float,
    t1: float,
) -> List[Tuple[float, float]]:
    """Return sampled parameter intervals that pass through a foreign cluster.

    Parameters
    ----------
    points : numpy.ndarray
        Uniformly parameterized curve samples with shape ``[N, 2]``.
    bbox : tuple[float, float, float, float]
        Cluster rectangle bounds.
    t0 : float
        Visible interval start.
    t1 : float
        Visible interval end.

    Returns
    -------
    list[tuple[float, float]]
        Sorted ``(entry_t, exit_t)`` intervals inside the rectangle.
    """
    intervals: List[Tuple[float, float]] = []
    inside = _point_inside_bbox(points[0], bbox)
    entry_t = 0.0 if inside else None
    for index in range(points.shape[0] - 1):
        next_inside = _point_inside_bbox(points[index + 1], bbox)
        if inside != next_inside:
            crossing_t = _transition_t(points, index, bbox)
            if not inside:
                entry_t = crossing_t
            elif entry_t is not None:
                intervals.append((max(float(entry_t), t0), min(float(crossing_t), t1)))
                entry_t = None
        inside = next_inside
    if inside and entry_t is not None:
        intervals.append((max(float(entry_t), t0), t1))
    return [(start, stop) for start, stop in intervals if stop - start > 1e-5]


def _subtract_interval(
    intervals: List[Tuple[float, float]],
    removed: Tuple[float, float],
) -> List[Tuple[float, float]]:
    """Subtract one interval from visible edge-body intervals.

    Parameters
    ----------
    intervals : list[tuple[float, float]]
        Current visible intervals.
    removed : tuple[float, float]
        Interval to remove.

    Returns
    -------
    list[tuple[float, float]]
        Remaining visible intervals.
    """
    remove_start, remove_stop = removed
    remaining: List[Tuple[float, float]] = []
    for start, stop in intervals:
        if remove_stop <= start or remove_start >= stop:
            remaining.append((start, stop))
            continue
        if remove_start - start > 1e-5:
            remaining.append((start, remove_start))
        if stop - remove_stop > 1e-5:
            remaining.append((remove_stop, stop))
    return remaining


def _clip_render_body_segments_at_clusters(
    curve: BezierCurve,
    src_idx: int,
    tgt_idx: int,
    cluster_membership: Dict[int, List[str]],
    cluster_bboxes: Dict[str, Tuple[float, float, float, float]],
) -> Tuple[Optional[List[RenderBezier]], Optional[str]]:
    """Return visible edge-body segments after cluster perimeter clipping.

    Parameters
    ----------
    curve : BezierCurve
        Original edge centerline.
    src_idx : int
        Source node index.
    tgt_idx : int
        Target node index.
    cluster_membership : dict[int, list[str]]
        Node index to containing cluster names.
    cluster_bboxes : dict[str, tuple[float, float, float, float]]
        Visible cluster rectangles.

    Returns
    -------
    tuple[list[RenderBezier] | None, str | None]
        Visible body segments and clipped terminal name. ``None`` segments
        means the original body is unchanged.
    """
    if src_idx == tgt_idx or not cluster_bboxes:
        return None, None

    render_curve = _curve_to_render_bezier(curve)
    points = sample_render_curve(render_curve, 160)
    src_clusters = set(cluster_membership.get(int(src_idx), []))
    tgt_clusters = set(cluster_membership.get(int(tgt_idx), []))
    t0 = 0.0
    t1 = 1.0

    for name in src_clusters - tgt_clusters:
        bbox = cluster_bboxes.get(name)
        exit_t = _source_exit_t(points, bbox) if bbox is not None else None
        if exit_t is not None:
            t0 = max(t0, exit_t)
    for name in tgt_clusters - src_clusters:
        bbox = cluster_bboxes.get(name)
        entry_t = _target_entry_t(points, bbox) if bbox is not None else None
        if entry_t is not None:
            t1 = min(t1, entry_t)

    if t1 - t0 <= 1e-5:
        return None, None

    intervals: List[Tuple[float, float]] = [(t0, t1)]
    for name, bbox in cluster_bboxes.items():
        if name in src_clusters or name in tgt_clusters:
            continue
        for inside_interval in _foreign_inside_intervals(points, bbox, t0, t1):
            intervals = _subtract_interval(intervals, inside_interval)
            if not intervals:
                return [], "both"

    clipped_tail = t0 > 1e-5
    clipped_head = t1 < 1.0 - 1e-5
    terminal = "both" if clipped_tail and clipped_head else "tail" if clipped_tail else "head"
    if not clipped_tail and not clipped_head and len(intervals) == 1:
        return None, None
    segments = [subcurve_render(render_curve, start, stop) for start, stop in intervals]
    return segments, terminal if clipped_tail or clipped_head else None


def _clip_render_body_curve_at_clusters(
    curve: BezierCurve,
    src_idx: int,
    tgt_idx: int,
    cluster_membership: Dict[int, List[str]],
    cluster_bboxes: Dict[str, Tuple[float, float, float, float]],
) -> Tuple[Optional[RenderBezier], Optional[str]]:
    """Return a body-only curve clipped at a crossed cluster perimeter.

    Parameters
    ----------
    curve : BezierCurve
        Original edge centerline whose terminals remain used for arrowheads.
    src_idx : int
        Source node index.
    tgt_idx : int
        Target node index.
    cluster_membership : dict[int, list[str]]
        Node index to containing cluster names.
    cluster_bboxes : dict[str, tuple[float, float, float, float]]
        Visible-stroke cluster rectangles.

    Returns
    -------
    tuple[RenderBezier | None, str | None]
        Body-only render curve and clipped terminal name, or ``(None, None)``
        when no clipping is needed.
    """
    segments, terminal = _clip_render_body_segments_at_clusters(
        curve,
        src_idx,
        tgt_idx,
        cluster_membership,
        cluster_bboxes,
    )
    if segments is None or len(segments) != 1:
        return None, None
    return segments[0], terminal


def _clip_direct_body_curve_at_clusters(
    curve: BezierCurve,
    src_idx: int,
    tgt_idx: int,
    cluster_membership: Dict[int, List[str]],
    cluster_bboxes: Dict[str, Tuple[float, float, float, float]],
) -> Tuple[BezierCurve, Optional[str]]:
    """Return a direct-render body curve clipped at cluster perimeters.

    Parameters
    ----------
    curve : BezierCurve
        Original edge centerline whose terminals remain used for arrowheads.
    src_idx : int
        Source node index.
    tgt_idx : int
        Target node index.
    cluster_membership : dict[int, list[str]]
        Node index to containing cluster names.
    cluster_bboxes : dict[str, tuple[float, float, float, float]]
        Visible-stroke cluster rectangles.

    Returns
    -------
    tuple[BezierCurve, str | None]
        Body curve and clipped terminal name, or the original curve with
        ``None`` when clipping is unnecessary.
    """
    if src_idx == tgt_idx or not cluster_bboxes:
        return curve, None
    if curve.waypoints is not None:
        points = [tuple(map(float, point)) for point in curve.waypoints]
    else:
        render_curve = _curve_to_render_bezier(curve)
        points = [tuple(map(float, point)) for point in sample_render_curve(render_curve, 96)]
    clipped = clip_edge_at_cluster_boundaries(
        edge_polyline=points,
        src_idx=src_idx,
        tgt_idx=tgt_idx,
        cluster_membership=cluster_membership,
        cluster_bboxes=cluster_bboxes,
    )
    if len(clipped) == len(points) and clipped == points:
        return curve, None
    clipped_tail = clipped[0] != points[0]
    clipped_head = clipped[-1] != points[-1]
    terminal = "both" if clipped_tail and clipped_head else "tail" if clipped_tail else "head"
    first_bend = clipped[1] if len(clipped) > 2 else clipped[0]
    last_bend = clipped[-2] if len(clipped) > 2 else clipped[-1]
    return (
        BezierCurve(
            p0=clipped[0],
            cp1=first_bend,
            cp2=last_bend,
            p1=clipped[-1],
            waypoints=tuple(clipped),
            routing=curve.routing,
            direction=curve.direction,
            step_fraction=curve.step_fraction,
        ),
        terminal,
    )


def _edge_requires_direct_render(style: Any) -> bool:
    """Return whether an edge style needs direct matplotlib rendering.

    Parameters
    ----------
    style : Any
        Edge style object.

    Returns
    -------
    bool
        ``True`` when the style uses a tapered body, body color gradient, or a
        non-default line cap/join or crossing jump style that the custom
        batched renderer does not expose yet.
    """
    return bool(
        _edge_uses_direct_data_ribbon(style)
        or getattr(style, "width_override_points", None) is not None
        or getattr(style, "taper", False)
        or getattr(style, "color_gradient", "none") != "none"
        or getattr(style, "line_cap", "butt") != "butt"
        or getattr(style, "line_join", "miter") != "miter"
        or getattr(style, "crossing_style", "none") != "none"
        or getattr(style, "routing", "bezier") in {"ortho", "taxi"}
    )


def _edge_uses_direct_data_ribbon(style: Any) -> bool:
    """Return whether a thin simple edge should use direct data ribbons.

    Parameters
    ----------
    style : Any
        Edge style object.

    Returns
    -------
    bool
        ``True`` for simple low-width edges whose collection path can underflow
        on fixed gallery canvases. The direct path still emits filled
        data-coordinate ribbons; it does not use Matplotlib display strokes.
    """
    return (
        0.0 < float(getattr(style, "width", 0.0)) <= 1.5
        and getattr(style, "arrow", "none") == "none"
        and getattr(style, "tail_arrow", "none") == "none"
        and not bool(getattr(style, "taper", False))
        and str(getattr(style, "color_gradient", "none")) == "none"
    )


def _curve_to_path(curve: BezierCurve) -> Any:
    """Convert a routed curve into a matplotlib path.

    Parameters
    ----------
    curve : BezierCurve
        Routed edge curve.

    Returns
    -------
    Any
        Matplotlib ``Path`` instance containing either a cubic segment or a
        waypoint polyline.
    """
    from matplotlib.path import Path

    if curve.waypoints is not None:
        waypoint_vertices = list(curve.waypoints)
        if len(waypoint_vertices) == 1:
            waypoint_vertices = waypoint_vertices * 2
        return Path(
            waypoint_vertices,
            [Path.MOVETO] + [Path.LINETO] * (len(waypoint_vertices) - 1),
        )

    return Path(
        [curve.p0, curve.cp1, curve.cp2, curve.p1],
        [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4],
    )


def _sample_curve_points(curve: BezierCurve, num_points: int = 48) -> np.ndarray:
    """Sample evenly spaced parameter values along a routed curve.

    Parameters
    ----------
    curve : BezierCurve
        Routed curve to sample.
    num_points : int, default=48
        Number of sample points.

    Returns
    -------
    numpy.ndarray
        Sampled points with shape ``[num_points, 2]`` for cubic curves, or the
        stored waypoint polyline for orthogonal routes.
    """
    if curve.waypoints is not None:
        return np.asarray(curve.waypoints, dtype=float)
    sample_count = max(2, int(num_points))
    params = np.linspace(0.0, 1.0, sample_count, dtype=float)
    return np.array([evaluate_bezier(curve, float(t)) for t in params], dtype=float)


def _curve_marker_direction(
    curve: BezierCurve,
    at_start: bool,
) -> Tuple[float, float]:
    """Return the local endpoint direction for marker placement.

    Parameters
    ----------
    curve : BezierCurve
        Routed edge centerline.
    at_start : bool
        Whether to read the direction at the start terminal.

    Returns
    -------
    tuple[float, float]
        Unnormalized local direction vector.
    """
    if curve.waypoints is not None and len(curve.waypoints) >= 2:
        if at_start:
            first_point, second_point = curve.waypoints[0], curve.waypoints[1]
            return (
                float(first_point[0] - second_point[0]),
                float(first_point[1] - second_point[1]),
            )
        penultimate_point, last_point = curve.waypoints[-2], curve.waypoints[-1]
        return (
            float(last_point[0] - penultimate_point[0]),
            float(last_point[1] - penultimate_point[1]),
        )

    if at_start:
        direction = (
            float(curve.p0[0] - curve.cp1[0]),
            float(curve.p0[1] - curve.cp1[1]),
        )
        if float(np.hypot(direction[0], direction[1])) > 1e-9:
            return direction
        return (
            float(curve.p0[0] - curve.p1[0]),
            float(curve.p0[1] - curve.p1[1]),
        )
    direction = (
        float(curve.p1[0] - curve.cp2[0]),
        float(curve.p1[1] - curve.cp2[1]),
    )
    if float(np.hypot(direction[0], direction[1])) > 1e-9:
        return direction
    return (
        float(curve.p1[0] - curve.p0[0]),
        float(curve.p1[1] - curve.p0[1]),
    )


def _polyline_dash_caps(pattern: Any, part_index: int) -> Tuple[str, str]:
    """Return cap styles for one visible polyline dash segment.

    Parameters
    ----------
    pattern : Any
        Dash pattern description.
    part_index : int
        Visible segment index within one pattern cycle.

    Returns
    -------
    tuple[str, str]
        Start and end cap names.
    """
    if pattern == "dotted":
        return "round", "round"
    if pattern == "dashdot" and part_index % 4 == 2:
        return "round", "round"
    if pattern in {"dashed", "dashdot"}:
        return "butt", "butt"
    return "round", "round"


def _slice_polyline_segment(
    points: np.ndarray,
    cumulative_lengths: np.ndarray,
    start_length: float,
    stop_length: float,
) -> np.ndarray:
    """Return the polyline subpath between two arc-length positions.

    Parameters
    ----------
    points : numpy.ndarray
        Polyline vertices with shape ``[N, 2]``.
    cumulative_lengths : numpy.ndarray
        Cumulative lengths with shape ``[N]``.
    start_length : float
        Inclusive start length in data units.
    stop_length : float
        Inclusive stop length in data units.

    Returns
    -------
    numpy.ndarray
        Polyline segment vertices with shape ``[M, 2]``.
    """
    if points.shape[0] <= 1 or stop_length <= start_length:
        return points[:1].copy()

    def interpolate(target_length: float) -> np.ndarray:
        if target_length <= 0.0:
            return points[0].copy()
        if target_length >= float(cumulative_lengths[-1]):
            return points[-1].copy()
        segment_index = int(np.searchsorted(cumulative_lengths, target_length, side="right") - 1)
        segment_index = min(max(segment_index, 0), points.shape[0] - 2)
        segment_start = float(cumulative_lengths[segment_index])
        segment_stop = float(cumulative_lengths[segment_index + 1])
        segment_length = max(segment_stop - segment_start, 1e-9)
        local_t = (target_length - segment_start) / segment_length
        return points[segment_index] + (points[segment_index + 1] - points[segment_index]) * local_t

    segment_points: List[np.ndarray] = [interpolate(start_length)]
    start_index = int(np.searchsorted(cumulative_lengths, start_length, side="right"))
    stop_index = int(np.searchsorted(cumulative_lengths, stop_length, side="left"))
    for index in range(start_index, stop_index):
        segment_points.append(points[index].copy())
    segment_points.append(interpolate(stop_length))
    return np.asarray(segment_points, dtype=float)


def _dash_polyline(
    points: np.ndarray,
    pattern: Any,
    width: float,
) -> List[Tuple[np.ndarray, str, str]]:
    """Split a polyline into visible dash segments.

    Parameters
    ----------
    points : numpy.ndarray
        Polyline vertices with shape ``[N, 2]``.
    pattern : Any
        Dash pattern description.
    width : float
        Edge width in data units.

    Returns
    -------
    list[tuple[numpy.ndarray, str, str]]
        Visible polyline segments plus their cap styles.
    """
    dash_pattern = parse_dash_pattern(pattern, width)
    if not dash_pattern:
        return [(points, "butt", "butt")]
    if points.shape[0] <= 1:
        return []

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative_lengths = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = float(cumulative_lengths[-1])
    if total_length <= 1e-9:
        return []

    visible_segments: List[Tuple[np.ndarray, str, str]] = []
    current_length = 0.0
    draw_segment = True
    part_index = 0
    pattern_length = len(dash_pattern)

    while current_length < total_length - 1e-9:
        part_length = float(dash_pattern[part_index % pattern_length])
        next_length = min(current_length + part_length, total_length)
        if draw_segment and next_length > current_length:
            segment_points = _slice_polyline_segment(
                points,
                cumulative_lengths,
                current_length,
                next_length,
            )
            visible_segments.append(
                (
                    segment_points,
                    *_polyline_dash_caps(pattern, part_index),
                )
            )
        current_length = next_length
        draw_segment = not draw_segment
        part_index += 1

    return visible_segments


def _sample_render_curve(curve: RenderBezier, width: float) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a render-space cubic densely enough for ribbon construction.

    Parameters
    ----------
    curve : dagua.render.edges.geometry.CubicBezier
        Cubic curve in render/data coordinates.
    width : float
        Target ribbon width in data units.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Sampled points with shape ``[N, 2]`` and the matching parametric
        fractions with shape ``[N]``.
    """
    samples = adaptive_subdivide(curve, flatness=max(width * 0.25, 0.12))
    points = polyline_from_samples(samples)
    params = np.array([float(sample.t) for sample in samples], dtype=float)
    return points, params


def _ribbon_join_style(join_style: str) -> str:
    """Map matplotlib-style joins onto the ribbon helper's supported joins.

    Parameters
    ----------
    join_style : str
        Requested join style from Dagua/Matplotlib.

    Returns
    -------
    str
        Ribbon join style supported by :func:`polyline_ribbon_path`.
    """
    return "miter" if join_style == "miter" else "bevel"


def _prepare_ribbon_points(
    points: np.ndarray,
    width: float,
    cap_start: str,
    cap_end: str,
) -> Tuple[np.ndarray, str, str]:
    """Apply square-cap extension before building ribbon geometry.

    Parameters
    ----------
    points : numpy.ndarray
        Ribbon centerline with shape ``[N, 2]``.
    width : float
        Ribbon width in data units.
    cap_start : str
        Requested start cap style.
    cap_end : str
        Requested end cap style.

    Returns
    -------
    tuple[numpy.ndarray, str, str]
        Possibly extended points plus normalized start/end cap names.
    """
    adjusted_points = np.array(points, dtype=float, copy=True)
    half_width = max(float(width) * 0.5, 0.0)
    if adjusted_points.shape[0] >= 2 and half_width > 0.0:
        if cap_start == "square":
            delta = adjusted_points[0] - adjusted_points[1]
            length = float(np.hypot(delta[0], delta[1]))
            if length > 1e-9:
                adjusted_points[0] += (delta / length) * half_width
            cap_start = "butt"
        if cap_end == "square":
            delta = adjusted_points[-1] - adjusted_points[-2]
            length = float(np.hypot(delta[0], delta[1]))
            if length > 1e-9:
                adjusted_points[-1] += (delta / length) * half_width
            cap_end = "butt"
    return adjusted_points, cap_start, cap_end


def _add_filled_ribbon_patch(
    ax: Any,
    points: np.ndarray,
    width: float,
    color: Any,
    zorder: float,
    cap_start: str = "butt",
    cap_end: str = "butt",
    join_style: str = "miter",
) -> Any:
    """Draw a filled ribbon patch for a polyline centerline.

    Parameters
    ----------
    ax : Any
        Matplotlib axes receiving the patch.
    points : numpy.ndarray
        Centerline points with shape ``[N, 2]``.
    width : float
        Ribbon width in data units.
    color : Any
        Matplotlib-compatible fill color.
    zorder : float
        Artist z-order.
    cap_start : str, default="butt"
        Start cap style.
    cap_end : str, default="butt"
        End cap style.
    join_style : str, default="miter"
        Join style for centerline corners.

    Returns
    -------
    Any
        Added matplotlib patch.
    """
    from matplotlib.patches import PathPatch

    adjusted_points, resolved_cap_start, resolved_cap_end = _prepare_ribbon_points(
        points,
        width,
        cap_start,
        cap_end,
    )
    patch = PathPatch(
        polyline_ribbon_path(
            adjusted_points,
            width=width,
            cap_start=resolved_cap_start,
            cap_end=resolved_cap_end,
            join_style=_ribbon_join_style(join_style),
        ),
        facecolor=color,
        edgecolor="none",
        linewidth=0.0,
        zorder=zorder,
        capstyle=(
            _mpl_capstyle(resolved_cap_start)
            if resolved_cap_start == resolved_cap_end
            else _mpl_capstyle("butt")
        ),
        joinstyle=join_style,
    )
    ax.add_patch(patch)
    return patch


def _polyline_distance_fractions(points: np.ndarray) -> np.ndarray:
    """Return normalized cumulative arc-length fractions for a polyline.

    Parameters
    ----------
    points : numpy.ndarray
        Polyline vertices with shape ``[N, 2]``.

    Returns
    -------
    numpy.ndarray
        Fractions in ``[0, 1]`` with shape ``[N]``.
    """
    if points.shape[0] <= 1:
        return np.zeros(points.shape[0], dtype=float)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = float(cumulative[-1])
    if total_length <= 1e-9:
        return np.zeros(points.shape[0], dtype=float)
    return cumulative / total_length


def _nearest_sample_fraction(
    points: np.ndarray,
    fractions: np.ndarray,
    point: np.ndarray,
) -> float:
    """Approximate a curve fraction for one point using sampled centerline points.

    Parameters
    ----------
    points : numpy.ndarray
        Sampled centerline points with shape ``[N, 2]``.
    fractions : numpy.ndarray
        Matching parametric fractions with shape ``[N]``.
    point : numpy.ndarray
        Query point with shape ``[2]``.

    Returns
    -------
    float
        Approximate fraction along the sampled curve.
    """
    distances = np.linalg.norm(points - point, axis=1)
    return float(fractions[int(np.argmin(distances))])


def _normalized_curve_tangent(curve: BezierCurve, t: float) -> Tuple[float, float]:
    """Return a unit tangent vector for a curve at parameter ``t``.

    Parameters
    ----------
    curve : BezierCurve
        Routed edge curve.
    t : float
        Parameter in ``[0, 1]`` along the curve.

    Returns
    -------
    tuple[float, float]
        Unit tangent vector in data coordinates.
    """
    clamped_t = min(max(float(t), 0.0), 1.0)
    dx, dy = bezier_tangent(curve, clamped_t)
    length = float(np.hypot(dx, dy))
    if length > 1e-9:
        return dx / length, dy / length

    sample_start = evaluate_bezier(curve, max(clamped_t - 1e-3, 0.0))
    sample_end = evaluate_bezier(curve, min(clamped_t + 1e-3, 1.0))
    fallback_dx = float(sample_end[0] - sample_start[0])
    fallback_dy = float(sample_end[1] - sample_start[1])
    fallback_length = float(np.hypot(fallback_dx, fallback_dy))
    if fallback_length > 1e-9:
        return fallback_dx / fallback_length, fallback_dy / fallback_length
    return (1.0, 0.0)


def _crossing_span_data_units(ax: Any, style: Any) -> float:
    """Return the data-space span used for one crossing jump.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    style : Any
        Edge style object.

    Returns
    -------
    float
        Crossing span in data units.
    """
    display_scale = _compute_display_scale(ax)
    display_span = (
        max(float(getattr(style, "crossing_size", 6.0)), _CROSSING_MIN_SPAN_POINTS) * display_scale
    )
    stroke_span = _edge_width_data_units(ax, float(style.width)) * _CROSSING_MIN_SPAN_WIDTH_FACTOR
    return max(display_span, stroke_span, _CROSSING_MIN_SPAN_DATA_UNITS)


def _sharp_crossing_span_data_units(ax: Any, style: Any) -> float:
    """Return the centerline width for a sharp crossing bridge.

    Parameters
    ----------
    ax : Any
        Matplotlib axes receiving the crossing patch.
    style : Any
        Edge style object.

    Returns
    -------
    float
        Sharp crossing bridge span in data units.
    """
    edge_width = _edge_width_data_units(ax, float(style.width))
    return edge_width * _CROSSING_SHARP_SPAN_WIDTH_FACTOR


def _bridge_crossing_span_data_units(ax: Any, style: Any) -> float:
    """Return the centerline width for a bridge crossing marker.

    Parameters
    ----------
    ax : Any
        Matplotlib axes receiving the crossing patch.
    style : Any
        Edge style object.

    Returns
    -------
    float
        Bridge crossing span in data units.
    """

    edge_width = _edge_width_data_units(ax, float(style.width))
    return max(
        _crossing_span_data_units(ax, style),
        edge_width * _CROSSING_BRIDGE_SPAN_WIDTH_FACTOR,
    )


def _draw_crossing_clearance(
    ax: Any,
    crossing: EdgeCrossing,
    curve: BezierCurve,
    t: float,
    background_color: str,
    style: Any,
    span: float,
) -> None:
    """Erase a short segment of the under-edge at one crossing.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    crossing : EdgeCrossing
        Crossing record being rendered.
    curve : BezierCurve
        Under-edge curve.
    t : float
        Parameter along ``curve`` where the crossing occurs.
    background_color : str
        Graph background color used as the eraser.
    style : Any
        Under-edge style object.
    span : float
        Crossing span in data units.
    """
    ux, uy = _normalized_curve_tangent(curve, t)
    half_span = span / 2.0
    clearance_width = _edge_width_data_units(
        ax,
        max(float(style.width) + _CROSSING_CLEARANCE_PADDING_POINTS, 0.0),
    )
    _add_filled_ribbon_patch(
        ax=ax,
        points=np.array(
            [
                [crossing.x - ux * half_span, crossing.y - uy * half_span],
                [crossing.x + ux * half_span, crossing.y + uy * half_span],
            ],
            dtype=float,
        ),
        width=clearance_width,
        color=background_color,
        zorder=1.6,
        cap_start="round",
        cap_end="round",
        join_style="bevel",
    )


def _draw_gap_crossing(
    ax: Any,
    crossing: EdgeCrossing,
    curve: BezierCurve,
    t: float,
    style: Any,
    span: float,
) -> None:
    """Redraw a straight segment across a cleared crossing.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    crossing : EdgeCrossing
        Crossing record being rendered.
    curve : BezierCurve
        Top-edge curve.
    t : float
        Parameter along ``curve`` where the crossing occurs.
    style : Any
        Top-edge style object.
    span : float
        Crossing span in data units.
    """
    from matplotlib.colors import to_rgba

    ux, uy = _normalized_curve_tangent(curve, t)
    half_span = span / 2.0
    _add_filled_ribbon_patch(
        ax=ax,
        points=np.array(
            [
                [crossing.x - ux * half_span, crossing.y - uy * half_span],
                [crossing.x + ux * half_span, crossing.y + uy * half_span],
            ],
            dtype=float,
        ),
        width=_edge_width_data_units(ax, float(style.width)),
        color=to_rgba(str(style.color), alpha=float(style.opacity)),
        zorder=1.7,
        cap_start=str(getattr(style, "line_cap", "round")),
        cap_end=str(getattr(style, "line_cap", "round")),
        join_style=str(getattr(style, "line_join", "round")),
    )


def _draw_arc_crossing(
    ax: Any,
    crossing: EdgeCrossing,
    curve: BezierCurve,
    t: float,
    style: Any,
    span: float,
) -> None:
    """Draw a semicircular jump arc over a crossing.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    crossing : EdgeCrossing
        Crossing record being rendered.
    curve : BezierCurve
        Top-edge curve.
    t : float
        Parameter along ``curve`` where the crossing occurs.
    style : Any
        Top-edge style object.
    span : float
        Crossing span in data units.
    """
    from matplotlib.colors import to_rgba

    ux, uy = _normalized_curve_tangent(curve, t)
    nx, ny = -uy, ux
    radius = span / 2.0
    angles = np.linspace(np.pi, 0.0, 24, dtype=float)
    arc_points = np.column_stack(
        [
            crossing.x + (ux * np.cos(angles) + nx * np.sin(angles)) * radius,
            crossing.y + (uy * np.cos(angles) + ny * np.sin(angles)) * radius,
        ]
    )
    _add_filled_ribbon_patch(
        ax=ax,
        points=arc_points,
        width=_edge_width_data_units(ax, float(style.width)),
        color=to_rgba(str(style.color), alpha=float(style.opacity)),
        zorder=1.7,
        cap_start="round",
        cap_end="round",
        join_style=str(getattr(style, "line_join", "round")),
    )


def _draw_sharp_crossing(
    ax: Any,
    crossing: EdgeCrossing,
    curve: BezierCurve,
    t: float,
    style: Any,
    span: float,
) -> None:
    """Draw a triangular jump over a crossing.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    crossing : EdgeCrossing
        Crossing record being rendered.
    curve : BezierCurve
        Top-edge curve.
    t : float
        Parameter along ``curve`` where the crossing occurs.
    style : Any
        Top-edge style object.
    span : float
        Crossing span in data units. Sharp crossings size their bridge from the
        edge width, so the span is only used by the caller for under-edge
        clearance.
    """
    from matplotlib.colors import to_rgba

    ux, uy = _normalized_curve_tangent(curve, t)
    nx, ny = -uy, ux
    del span
    edge_width = _edge_width_data_units(ax, float(style.width))
    half_span = edge_width * (_CROSSING_SHARP_SPAN_WIDTH_FACTOR / 2.0)
    peak_height = edge_width * _CROSSING_SHARP_HEIGHT_WIDTH_FACTOR
    _add_filled_ribbon_patch(
        ax=ax,
        points=np.array(
            [
                [crossing.x - ux * half_span, crossing.y - uy * half_span],
                [crossing.x + nx * peak_height, crossing.y + ny * peak_height],
                [crossing.x + ux * half_span, crossing.y + uy * half_span],
            ],
            dtype=float,
        ),
        width=_edge_width_data_units(ax, float(style.width)),
        color=to_rgba(str(style.color), alpha=float(style.opacity)),
        zorder=1.7,
        cap_start="round",
        cap_end="round",
        join_style="bevel",
    )


def _draw_bridge_crossing(
    ax: Any,
    crossing: EdgeCrossing,
    under_curve: BezierCurve,
    under_t: float,
    style: Any,
    background_color: str,
) -> None:
    """Draw a rounded bridge marker that matches the graph background.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    crossing : EdgeCrossing
        Crossing record being rendered.
    under_curve : BezierCurve
        Under-crossing curve whose tangent defines the bridge orientation.
    under_t : float
        Parameter along ``under_curve`` where the crossing occurs.
    style : Any
        Top-edge style object.
    background_color : str
        Graph background color used to erase the under-crossing segment.
    """

    from matplotlib.colors import to_rgba
    from matplotlib.patches import PathPatch
    from matplotlib.transforms import Affine2D

    ux, uy = _normalized_curve_tangent(under_curve, under_t)
    nx, ny = -uy, ux
    edge_width = _edge_width_data_units(ax, float(style.width))
    half_span = edge_width * (_CROSSING_BRIDGE_SPAN_WIDTH_FACTOR / 2.0)
    half_height = edge_width * (_CROSSING_BRIDGE_HEIGHT_WIDTH_FACTOR / 2.0)
    corner_radius = min(
        _points_to_data_units(ax, _CROSSING_BRIDGE_CORNER_RADIUS_POINTS, "x"),
        _points_to_data_units(ax, _CROSSING_BRIDGE_CORNER_RADIUS_POINTS, "y"),
        half_height * 0.9,
        half_span * 0.9,
    )
    bridge_path = build_shape_path(
        ShapeSpec(
            center_x=0.0,
            center_y=0.0,
            width=half_height * 2.0,
            height=half_span * 2.0,
            shape="roundrect",
            corner_radius=corner_radius,
        )
    )
    patch = PathPatch(
        bridge_path,
        facecolor=to_rgba(background_color),
        edgecolor=to_rgba(str(style.color), alpha=float(style.opacity)),
        linewidth=_CROSSING_BRIDGE_STROKE_WIDTH_POINTS,
        joinstyle="round",
        capstyle="round",
        transform=Affine2D().from_values(ux, uy, nx, ny, crossing.x, crossing.y) + ax.transData,
        zorder=1.85,
        gid="dagua-crossing-bridge",
    )
    ax.add_patch(patch)


def _draw_edge_crossings(
    ax: Any,
    graph: Any,
    curves: List[BezierCurve],
    crossings: List[EdgeCrossing],
) -> None:
    """Render jump-style edge crossings for the current edge set.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    graph : Any
        Graph exposing per-edge style lookups.
    curves : list[BezierCurve]
        Routed edge curves.
    crossings : list[EdgeCrossing]
        Crossing records to render.
    """
    if not crossings:
        return

    background_color = str(graph.graph_style.background_color)
    for crossing in crossings:
        top_edge_index = int(crossing.edge_b)
        under_edge_index = int(crossing.edge_a)
        top_style = _edge_style_for_render(graph, top_edge_index)
        crossing_style = str(getattr(top_style, "crossing_style", "none")).lower()
        if crossing_style == "none":
            continue

        under_style = _edge_style_for_render(graph, under_edge_index)
        if crossing_style == "sharp":
            span = _sharp_crossing_span_data_units(ax, top_style)
        elif crossing_style == "bridge":
            span = _bridge_crossing_span_data_units(ax, top_style)
        else:
            span = _crossing_span_data_units(ax, top_style)
        _draw_crossing_clearance(
            ax=ax,
            crossing=crossing,
            curve=curves[under_edge_index],
            t=float(crossing.t_a),
            background_color=background_color,
            style=under_style,
            span=span,
        )
        if crossing_style == "gap":
            _draw_gap_crossing(
                ax, crossing, curves[top_edge_index], float(crossing.t_b), top_style, span
            )
        elif crossing_style == "arc":
            _draw_arc_crossing(
                ax, crossing, curves[top_edge_index], float(crossing.t_b), top_style, span
            )
        elif crossing_style == "sharp":
            _draw_sharp_crossing(
                ax, crossing, curves[top_edge_index], float(crossing.t_b), top_style, span
            )
        elif crossing_style == "bridge":
            _draw_bridge_crossing(
                ax,
                crossing,
                curves[under_edge_index],
                float(crossing.t_a),
                top_style,
                background_color,
            )


def _edge_gradient_colors(style: Any) -> Tuple[Tuple[float, float, float, float], ...]:
    """Resolve the start and end RGBA colors for an edge body gradient.

    Parameters
    ----------
    style : Any
        Edge style object.

    Returns
    -------
    tuple[tuple[float, float, float, float], tuple[float, float, float, float]]
        Start and end RGBA colors with the style opacity applied.
    """
    from matplotlib.colors import to_rgba

    start_color = to_rgba(str(style.color), alpha=float(style.opacity))
    end_color_value = str(style.color_gradient_end or style.color)
    end_color = to_rgba(end_color_value, alpha=float(style.opacity))
    return start_color, end_color


def _interpolate_rgba(
    start: Tuple[float, float, float, float],
    end: Tuple[float, float, float, float],
    fraction: float,
) -> Tuple[float, float, float, float]:
    """Interpolate two RGBA colors.

    Parameters
    ----------
    start : tuple[float, float, float, float]
        Start RGBA color.
    end : tuple[float, float, float, float]
        End RGBA color.
    fraction : float
        Interpolation amount in ``[0, 1]``.

    Returns
    -------
    tuple[float, float, float, float]
        Interpolated RGBA color.
    """
    clamped_fraction = min(max(float(fraction), 0.0), 1.0)
    return (
        start[0] * (1.0 - clamped_fraction) + end[0] * clamped_fraction,
        start[1] * (1.0 - clamped_fraction) + end[1] * clamped_fraction,
        start[2] * (1.0 - clamped_fraction) + end[2] * clamped_fraction,
        start[3] * (1.0 - clamped_fraction) + end[3] * clamped_fraction,
    )


def _tapered_edge_outline(
    curve_points: np.ndarray,
    width_start: float,
    width_end: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build upper and lower edge outlines for a tapered ribbon.

    Parameters
    ----------
    curve_points : numpy.ndarray
        Sampled edge centerline with shape ``[N, 2]``.
    width_start : float
        Source-end width in data units.
    width_end : float
        Target-end width in data units.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Upper and lower outline points, each with shape ``[N, 2]``.
    """
    point_count = int(curve_points.shape[0])
    widths = np.linspace(float(width_start), float(width_end), point_count, dtype=float)
    upper = np.zeros_like(curve_points, dtype=float)
    lower = np.zeros_like(curve_points, dtype=float)

    for index in range(point_count):
        if point_count == 1:
            dx, dy = 0.0, 1.0
        elif index < point_count - 1:
            dx = float(curve_points[index + 1, 0] - curve_points[index, 0])
            dy = float(curve_points[index + 1, 1] - curve_points[index, 1])
        else:
            dx = float(curve_points[index, 0] - curve_points[index - 1, 0])
            dy = float(curve_points[index, 1] - curve_points[index - 1, 1])

        length = max(float(np.hypot(dx, dy)), 1e-6)
        nx = -dy / length
        ny = dx / length
        half_width = widths[index] / 2.0
        upper[index] = (
            float(curve_points[index, 0]) + nx * half_width,
            float(curve_points[index, 1]) + ny * half_width,
        )
        lower[index] = (
            float(curve_points[index, 0]) - nx * half_width,
            float(curve_points[index, 1]) - ny * half_width,
        )

    return upper, lower


def _taper_width_at_fraction(width_start: float, width_end: float, fraction: float) -> float:
    """Interpolate a taper width at an edge arc-length fraction.

    Parameters
    ----------
    width_start : float
        Source-end ribbon width in data units.
    width_end : float
        Target-end ribbon width in data units.
    fraction : float
        Normalized arc-length fraction in ``[0, 1]``.

    Returns
    -------
    float
        Interpolated ribbon width.
    """
    clamped_fraction = min(max(float(fraction), 0.0), 1.0)
    return float(width_start) * (1.0 - clamped_fraction) + float(width_end) * clamped_fraction


def _draw_direct_edge_body(ax: Any, curve: BezierCurve, style: Any) -> List[Any]:
    """Draw a single edge body with direct matplotlib artists.

    Parameters
    ----------
    ax : Any
        Matplotlib axes receiving the edge artists.
    curve : BezierCurve
        Routed edge centerline.
    style : Any
        Edge style object.

    Returns
    -------
    list[Any]
        Added matplotlib artists.
    """
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Polygon

    artists: List[Any] = []
    width_override = getattr(style, "width_override_points", None)
    if width_override is not None:
        edge_path = _curve_to_path(curve)
        patch = _draw_display_point_path_stroke(
            ax,
            edge_path,
            to_rgba(str(style.color), alpha=float(style.opacity)),
            float(width_override),
            zorder=1,
            linestyle=str(style.style),
            capstyle=str(getattr(style, "line_cap", "butt")),
            joinstyle=str(getattr(style, "line_join", "miter")),
        )
        artists.append(patch)
        return artists

    data_width = _edge_width_data_units(ax, float(style.width))
    if curve.waypoints is not None:
        full_points = _sample_curve_points(curve)
        full_params = _polyline_distance_fractions(full_points)

        if getattr(style, "taper", False):
            width_start, width_end = _resolved_taper_widths(ax, style)
            if str(style.style) != "solid":
                for segment_points, cap_start, cap_end in _dash_polyline(
                    full_points,
                    str(style.style),
                    data_width,
                ):
                    start_fraction = _nearest_sample_fraction(
                        full_points,
                        full_params,
                        segment_points[0],
                    )
                    end_fraction = _nearest_sample_fraction(
                        full_points,
                        full_params,
                        segment_points[-1],
                    )
                    segment_start_width = _taper_width_at_fraction(
                        width_start,
                        width_end,
                        start_fraction,
                    )
                    segment_end_width = _taper_width_at_fraction(
                        width_start,
                        width_end,
                        end_fraction,
                    )
                    upper, lower = _tapered_edge_outline(
                        segment_points,
                        segment_start_width,
                        segment_end_width,
                    )
                    patch = Polygon(
                        np.vstack([upper, lower[::-1]]),
                        closed=True,
                        facecolor=to_rgba(str(style.color), alpha=float(style.opacity)),
                        edgecolor="none",
                        joinstyle=str(style.line_join),
                        zorder=1,
                    )
                    ax.add_patch(patch)
                    artists.append(patch)
                return artists
            upper, lower = _tapered_edge_outline(full_points, width_start, width_end)
            start_color, end_color = _edge_gradient_colors(style)

            if getattr(style, "color_gradient", "none") == "source_to_target":
                for index in range(full_points.shape[0] - 1):
                    quad = np.array(
                        [upper[index], upper[index + 1], lower[index + 1], lower[index]],
                        dtype=float,
                    )
                    fraction = (index + 0.5) / max(full_points.shape[0] - 1, 1)
                    patch = Polygon(
                        quad,
                        closed=True,
                        facecolor=_interpolate_rgba(start_color, end_color, fraction),
                        edgecolor="none",
                        joinstyle=str(style.line_join),
                        zorder=1,
                    )
                    ax.add_patch(patch)
                    artists.append(patch)
                return artists

            polygon = Polygon(
                np.vstack([upper, lower[::-1]]),
                closed=True,
                facecolor=to_rgba(str(style.color), alpha=float(style.opacity)),
                edgecolor="none",
                joinstyle=str(style.line_join),
                zorder=1,
            )
            ax.add_patch(polygon)
            artists.append(polygon)
            return artists

        if getattr(style, "color_gradient", "none") == "source_to_target":
            start_color, end_color = _edge_gradient_colors(style)
            upper, lower = _tapered_edge_outline(full_points, data_width, data_width)
            for index in range(full_points.shape[0] - 1):
                quad = np.array(
                    [upper[index], upper[index + 1], lower[index + 1], lower[index]],
                    dtype=float,
                )
                fraction = (float(full_params[index]) + float(full_params[index + 1])) / 2.0
                patch = Polygon(
                    quad,
                    closed=True,
                    facecolor=_interpolate_rgba(start_color, end_color, fraction),
                    edgecolor="none",
                    linewidth=0.0,
                    joinstyle=str(style.line_join),
                    zorder=1,
                )
                ax.add_patch(patch)
                artists.append(patch)
            return artists

        for segment_points, cap_start, cap_end in _dash_polyline(
            full_points,
            str(style.style),
            data_width,
        ):
            patch = _add_filled_ribbon_patch(
                ax=ax,
                points=segment_points,
                width=data_width,
                color=to_rgba(str(style.color), alpha=float(style.opacity)),
                zorder=1,
                cap_start=cap_start if str(style.style) != "solid" else str(style.line_cap),
                cap_end=cap_end if str(style.style) != "solid" else str(style.line_cap),
                join_style=str(style.line_join),
            )
            artists.append(patch)
        return artists

    render_curve = _curve_to_render_bezier(curve)
    full_points, full_params = _sample_render_curve(render_curve, data_width)

    if getattr(style, "taper", False):
        points = _sample_curve_points(curve)
        width_start, width_end = _resolved_taper_widths(ax, style)
        if str(style.style) != "solid":
            for dash_segment in dash_curve(render_curve, str(style.style), data_width):
                segment_points, _ = _sample_render_curve(dash_segment.curve, data_width)
                start_fraction = _nearest_sample_fraction(
                    full_points,
                    full_params,
                    segment_points[0],
                )
                end_fraction = _nearest_sample_fraction(
                    full_points,
                    full_params,
                    segment_points[-1],
                )
                segment_start_width = _taper_width_at_fraction(
                    width_start,
                    width_end,
                    start_fraction,
                )
                segment_end_width = _taper_width_at_fraction(
                    width_start,
                    width_end,
                    end_fraction,
                )
                upper, lower = _tapered_edge_outline(
                    segment_points,
                    segment_start_width,
                    segment_end_width,
                )
                patch = Polygon(
                    np.vstack([upper, lower[::-1]]),
                    closed=True,
                    facecolor=to_rgba(str(style.color), alpha=float(style.opacity)),
                    edgecolor="none",
                    joinstyle=str(style.line_join),
                    zorder=1,
                )
                ax.add_patch(patch)
                artists.append(patch)
            return artists
        upper, lower = _tapered_edge_outline(points, width_start, width_end)
        start_color, end_color = _edge_gradient_colors(style)

        if getattr(style, "color_gradient", "none") == "source_to_target":
            for index in range(points.shape[0] - 1):
                quad = np.array(
                    [upper[index], upper[index + 1], lower[index + 1], lower[index]],
                    dtype=float,
                )
                fraction = (index + 0.5) / max(points.shape[0] - 1, 1)
                patch = Polygon(
                    quad,
                    closed=True,
                    facecolor=_interpolate_rgba(start_color, end_color, fraction),
                    edgecolor="none",
                    joinstyle=str(style.line_join),
                    zorder=1,
                )
                ax.add_patch(patch)
                artists.append(patch)
            return artists

        vertices = np.vstack([upper, lower[::-1]])
        polygon = Polygon(
            vertices,
            closed=True,
            facecolor=to_rgba(str(style.color), alpha=float(style.opacity)),
            edgecolor="none",
            joinstyle=str(style.line_join),
            zorder=1,
        )
        ax.add_patch(polygon)
        artists.append(polygon)
        return artists

    if getattr(style, "color_gradient", "none") == "source_to_target":
        start_color, end_color = _edge_gradient_colors(style)
        if str(style.style) == "solid":
            upper, lower = _tapered_edge_outline(full_points, data_width, data_width)
            for index in range(full_points.shape[0] - 1):
                quad = np.array(
                    [upper[index], upper[index + 1], lower[index + 1], lower[index]],
                    dtype=float,
                )
                fraction = (float(full_params[index]) + float(full_params[index + 1])) / 2.0
                patch = Polygon(
                    quad,
                    closed=True,
                    facecolor=_interpolate_rgba(start_color, end_color, fraction),
                    edgecolor="none",
                    linewidth=0.0,
                    joinstyle=str(style.line_join),
                    zorder=1,
                )
                ax.add_patch(patch)
                artists.append(patch)
            return artists

        dash_segments = dash_curve(render_curve, str(style.style), data_width)
        for dash_segment in dash_segments:
            segment_points, _ = _sample_render_curve(dash_segment.curve, data_width)
            start_fraction = _nearest_sample_fraction(full_points, full_params, segment_points[0])
            end_fraction = _nearest_sample_fraction(full_points, full_params, segment_points[-1])
            midpoint_fraction = (start_fraction + end_fraction) / 2.0
            patch = _add_filled_ribbon_patch(
                ax=ax,
                points=segment_points,
                width=data_width,
                color=_interpolate_rgba(start_color, end_color, midpoint_fraction),
                zorder=1,
                cap_start=dash_segment.cap_start,
                cap_end=dash_segment.cap_end,
                join_style=str(style.line_join),
            )
            artists.append(patch)
        return artists

    dash_segments = dash_curve(render_curve, str(style.style), data_width)
    for dash_segment in dash_segments:
        segment_points, _ = _sample_render_curve(dash_segment.curve, data_width)
        cap_start = dash_segment.cap_start
        cap_end = dash_segment.cap_end
        if str(style.style) == "solid":
            cap_start = str(getattr(style, "line_cap", "butt"))
            cap_end = str(getattr(style, "line_cap", "butt"))
        patch = _add_filled_ribbon_patch(
            ax=ax,
            points=segment_points,
            width=data_width,
            color=to_rgba(str(style.color), alpha=float(style.opacity)),
            zorder=1,
            cap_start=cap_start,
            cap_end=cap_end,
            join_style=str(style.line_join),
        )
        artists.append(patch)
    return artists


def _draw_direct_edge_markers(
    ax: Any,
    graph: Any,
    curves: List[BezierCurve],
    positions: Optional[np.ndarray] = None,
) -> None:
    """Draw edge endpoint markers after edge bodies are in place.

    Parameters
    ----------
    ax : Any
        Matplotlib axes receiving the marker artists.
    graph : Any
        Graph exposing edge styles and node sizes.
    curves : list[BezierCurve]
        Routed edge curves.
    positions : numpy.ndarray | None, default=None
        Node positions with shape ``[N, 2]`` in data coordinates.
    """
    for e_idx, curve in enumerate(curves):
        style = _edge_style_for_render(graph, e_idx)
        if getattr(style, "taper", False):
            style = replace(
                style,
                arrow_length=max(float(style.arrow_length), 18.0),
                arrow_width=max(float(style.arrow_width), 12.0),
            )
        src_idx = int(graph.edge_index[0, e_idx])
        tgt_idx = int(graph.edge_index[1, e_idx])
        is_self_loop = src_idx == tgt_idx
        src_node_width = float(graph.node_sizes[src_idx, 0])
        src_node_height = float(graph.node_sizes[src_idx, 1])
        tgt_node_width = float(graph.node_sizes[tgt_idx, 0])
        tgt_node_height = float(graph.node_sizes[tgt_idx, 1])
        stroke_scale = _effective_stroke_scale(ax)
        gradient_start_color = str(style.color)
        gradient_end_color = str(style.color_gradient_end or style.color)

        if getattr(style, "tail_arrow", "none") != "none":
            tail_style = style
            if getattr(style, "color_gradient", "none") == "source_to_target" and not getattr(
                style, "arrow_color", ""
            ):
                tail_style = replace(style, color=gradient_start_color)
            start_dx, start_dy = _curve_marker_direction(curve, at_start=True)
            tail_point = _offset_edge_terminal_point(
                graph,
                positions,
                src_idx,
                curve.p0,
                stroke_scale,
            )
            _draw_edge_marker(
                ax=ax,
                point=tail_point,
                direction=(start_dx, start_dy),
                marker=str(style.tail_arrow),
                style=tail_style,
                node_width=src_node_width,
                node_height=src_node_height,
                is_self_loop=is_self_loop,
            )

        if getattr(style, "arrow", "none") != "none":
            head_style = style
            if getattr(style, "color_gradient", "none") == "source_to_target" and not getattr(
                style, "arrow_color", ""
            ):
                head_style = replace(style, color=gradient_end_color)
            end_dx, end_dy = _curve_marker_direction(curve, at_start=False)
            head_point = _offset_edge_terminal_point(
                graph,
                positions,
                tgt_idx,
                curve.p1,
                stroke_scale,
            )
            _draw_edge_marker(
                ax=ax,
                point=head_point,
                direction=(end_dx, end_dy),
                marker=str(style.arrow),
                style=head_style,
                node_width=tgt_node_width,
                node_height=tgt_node_height,
                is_self_loop=is_self_loop,
            )


def _draw_outline_segments(
    ax: Any,
    points: Sequence[Tuple[float, float]],
    stroke_width: float,
    color: Any,
    zorder: float,
    closed: bool = False,
) -> List[Any]:
    """Render a thin outlined marker path as filled ribbon segments.

    Parameters
    ----------
    ax : Any
        Matplotlib axes receiving the patches.
    points : Sequence[tuple[float, float]]
        Marker polyline control points in data coordinates.
    stroke_width : float
        Outline width in data units.
    color : Any
        Matplotlib-compatible fill color.
    zorder : float
        Artist z-order.
    closed : bool, default=False
        Whether to add the closing segment from the last point back to the
        first point.

    Returns
    -------
    list[Any]
        Added ribbon patches.
    """
    artists: List[Any] = []
    point_count = len(points)
    if point_count < 2 or stroke_width <= 0.0:
        return artists

    last_index = point_count if closed else point_count - 1
    for index in range(last_index):
        start_point = np.array(points[index % point_count], dtype=float)
        end_point = np.array(points[(index + 1) % point_count], dtype=float)
        if np.allclose(start_point, end_point):
            continue
        patch = _add_filled_ribbon_patch(
            ax=ax,
            points=np.vstack([start_point, end_point]),
            width=stroke_width,
            color=color,
            zorder=zorder,
            cap_start="round",
            cap_end="round",
            join_style="bevel",
        )
        artists.append(patch)
    return artists


def _draw_circle_ring_patch(
    ax: Any,
    center: Tuple[float, float],
    radius: float,
    stroke_width: float,
    color: Any,
    zorder: float,
) -> Any:
    """Render a hollow circular marker as an annular polygon.

    Parameters
    ----------
    ax : Any
        Matplotlib axes receiving the patch.
    center : tuple[float, float]
        Circle center in data coordinates.
    radius : float
        Outer radius in data units.
    stroke_width : float
        Ring thickness in data units.
    color : Any
        Matplotlib-compatible fill color.
    zorder : float
        Artist z-order.

    Returns
    -------
    Any
        Added annular patch.
    """
    from matplotlib.patches import Polygon

    outer_radius = max(float(radius), 0.0)
    inner_radius = max(outer_radius - max(float(stroke_width), 0.0), 0.0)
    angles = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False, dtype=float)
    outer_points = np.column_stack(
        [
            center[0] + np.cos(angles) * outer_radius,
            center[1] + np.sin(angles) * outer_radius,
        ]
    )
    inner_points = np.column_stack(
        [
            center[0] + np.cos(angles) * inner_radius,
            center[1] + np.sin(angles) * inner_radius,
        ]
    )
    patch = Polygon(
        np.vstack([outer_points, outer_points[:1], inner_points[::-1], inner_points[-1:]]),
        closed=True,
        facecolor=color,
        edgecolor="none",
        linewidth=0.0,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def _trim_curve_for_arrows(
    ax: Any,
    curve: BezierCurve,
    style: Any,
    graph: Any,
    edge_idx: int,
    skip_head_trim: bool = False,
    skip_tail_trim: bool = False,
) -> BezierCurve:
    """Shorten a direct-rendered curve so arrowheads occupy the endpoint gap.

    Parameters
    ----------
    ax : Any
        Matplotlib axes used for point-to-data marker sizing.
    curve : BezierCurve
        Original routed curve whose endpoints already touch node boundaries.
    style : Any
        Edge style object supplying arrowhead configuration.
    graph : Any
        Graph exposing edge endpoints and node sizes.
    edge_idx : int
        Edge index used to resolve source and target node heights.
    skip_head_trim : bool, default=False
        Do not shorten the target-side body endpoint for the arrowhead.
    skip_tail_trim : bool, default=False
        Do not shorten the source-side body endpoint for the tail arrow.

    Returns
    -------
    BezierCurve
        Curve with trimmed endpoints for the body pass. Marker rendering still
        uses the original curve so arrow tips stay seated on node boundaries.
    """
    if curve.waypoints is not None:
        points = np.asarray(curve.waypoints, dtype=float)
        if points.shape[0] <= 1:
            return curve

        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        total_length = float(segment_lengths.sum())
        if total_length <= 1e-9:
            return curve

        cumulative_lengths = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        start_trim = 0.0
        end_trim = 0.0
        node_sizes = getattr(graph, "node_sizes", None)

        if getattr(style, "tail_arrow", "none") != "none" and not skip_tail_trim:
            src_idx = int(graph.edge_index[0, edge_idx])
            src_width = float(node_sizes[src_idx, 0]) if node_sizes is not None else 0.0
            src_height = float(node_sizes[src_idx, 1]) if node_sizes is not None else 0.0
            tail_length, _ = _resolved_marker_dimensions(
                ax,
                style,
                src_width,
                src_height,
                is_self_loop=src_idx == int(graph.edge_index[1, edge_idx]),
                scale_with_edge_width=False,
            )
            start_trim = min(
                tail_length,
                float(segment_lengths[0]) * _DIRECT_ARROW_TRIM_MAX_FRACTION,
            )

        if getattr(style, "arrow", "none") != "none" and not skip_head_trim:
            tgt_idx = int(graph.edge_index[1, edge_idx])
            tgt_width = float(node_sizes[tgt_idx, 0]) if node_sizes is not None else 0.0
            tgt_height = float(node_sizes[tgt_idx, 1]) if node_sizes is not None else 0.0
            head_length, _ = _resolved_marker_dimensions(
                ax,
                style,
                tgt_width,
                tgt_height,
                is_self_loop=tgt_idx == int(graph.edge_index[0, edge_idx]),
                scale_with_edge_width=False,
            )
            end_trim = min(
                head_length,
                float(segment_lengths[-1]) * _DIRECT_ARROW_TRIM_MAX_FRACTION,
            )

        stop_length = max(total_length - end_trim, start_trim)
        trimmed_points = _slice_polyline_segment(
            points,
            cumulative_lengths,
            start_trim,
            stop_length,
        )
        waypoint_list = [tuple(map(float, point)) for point in trimmed_points]
        if len(waypoint_list) == 1:
            point = waypoint_list[0]
            return BezierCurve(point, point, point, point, waypoints=(point,))

        first_bend = waypoint_list[1] if len(waypoint_list) > 2 else waypoint_list[0]
        last_bend = waypoint_list[-2] if len(waypoint_list) > 2 else waypoint_list[-1]
        return BezierCurve(
            p0=waypoint_list[0],
            cp1=first_bend,
            cp2=last_bend,
            p1=waypoint_list[-1],
            waypoints=tuple(waypoint_list),
        )

    p0x, p0y = curve.p0
    cp1x, cp1y = curve.cp1
    cp2x, cp2y = curve.cp2
    p1x, p1y = curve.p1

    node_sizes = getattr(graph, "node_sizes", None)

    if getattr(style, "tail_arrow", "none") != "none" and not skip_tail_trim:
        src_idx = int(graph.edge_index[0, edge_idx])
        src_width = float(node_sizes[src_idx, 0]) if node_sizes is not None else 0.0
        src_height = float(node_sizes[src_idx, 1]) if node_sizes is not None else 0.0
        tail_length, _ = _resolved_marker_dimensions(
            ax,
            style,
            src_width,
            src_height,
            is_self_loop=src_idx == int(graph.edge_index[1, edge_idx]),
            scale_with_edge_width=False,
        )
        tail_dx = cp1x - p0x
        tail_dy = cp1y - p0y
        tail_distance = float(np.hypot(tail_dx, tail_dy))
        if tail_distance > 1e-9 and tail_length > 0.0:
            tail_fraction = min(
                tail_length / tail_distance,
                _DIRECT_ARROW_TRIM_MAX_FRACTION,
            )
            p0x += tail_dx * tail_fraction
            p0y += tail_dy * tail_fraction

    if getattr(style, "arrow", "none") != "none" and not skip_head_trim:
        tgt_idx = int(graph.edge_index[1, edge_idx])
        tgt_width = float(node_sizes[tgt_idx, 0]) if node_sizes is not None else 0.0
        tgt_height = float(node_sizes[tgt_idx, 1]) if node_sizes is not None else 0.0
        head_length, _ = _resolved_marker_dimensions(
            ax,
            style,
            tgt_width,
            tgt_height,
            is_self_loop=tgt_idx == int(graph.edge_index[0, edge_idx]),
            scale_with_edge_width=False,
        )
        head_dx = cp2x - p1x
        head_dy = cp2y - p1y
        head_distance = float(np.hypot(head_dx, head_dy))
        if head_distance > 1e-9 and head_length > 0.0:
            head_fraction = min(
                head_length / head_distance,
                _DIRECT_ARROW_TRIM_MAX_FRACTION,
            )
            p1x += head_dx * head_fraction
            p1y += head_dy * head_fraction

    return BezierCurve(
        p0=(p0x, p0y),
        cp1=(cp1x, cp1y),
        cp2=(cp2x, cp2y),
        p1=(p1x, p1y),
    )


def _draw_edges_direct(
    ax: Any,
    graph: Any,
    curves: List[BezierCurve],
    crossings: Optional[List[EdgeCrossing]] = None,
    positions: Optional[np.ndarray] = None,
    cluster_membership: Optional[Dict[int, List[str]]] = None,
    cluster_bboxes: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
) -> None:
    """Draw all edges with direct matplotlib artists.

    Parameters
    ----------
    ax : Any
        Matplotlib axes receiving the edge artists.
    graph : Any
        Graph exposing Dagua's edge-style API.
    curves : list[BezierCurve]
        Routed edge curves.
    crossings : list[EdgeCrossing], optional
        Crossing records that should receive jump rendering.
    positions : numpy.ndarray | None, default=None
        Node positions with shape ``[N, 2]`` in data coordinates.
    cluster_membership : dict[int, list[str]] | None, default=None
        Node-to-cluster membership for edge-body clipping.
    cluster_bboxes : dict[str, tuple[float, float, float, float]] | None, default=None
        Visible cluster rectangles for edge-body clipping.
    """
    for e_idx, curve in enumerate(curves):
        style = _edge_style_for_render(graph, e_idx)
        body_curve = curve
        clipped_terminal: Optional[str] = None
        if cluster_membership is not None and cluster_bboxes is not None:
            src_idx = int(graph.edge_index[0, e_idx])
            tgt_idx = int(graph.edge_index[1, e_idx])
            body_curve, clipped_terminal = _clip_direct_body_curve_at_clusters(
                curve,
                src_idx,
                tgt_idx,
                cluster_membership,
                cluster_bboxes,
            )
        trimmed_curve = _trim_curve_for_arrows(
            ax,
            body_curve,
            style,
            graph,
            e_idx,
            skip_head_trim=clipped_terminal in {"head", "both"},
            skip_tail_trim=clipped_terminal in {"tail", "both"},
        )
        _draw_direct_edge_body(ax, trimmed_curve, style)

    if crossings:
        _draw_edge_crossings(ax, graph, curves, crossings)
    if positions is None:
        _draw_direct_edge_markers(ax, graph, curves)
        return
    _draw_direct_edge_markers(ax, graph, curves, positions)


def _graphviz_strict_back_edge_curve(
    ax: Any,
    graph: Any,
    edge_idx: int,
    curve: BezierCurve,
    style: EdgeStyle,
) -> BezierCurve:
    """Return a visibly bowed strict-theme back-edge curve.

    Parameters
    ----------
    ax : Any
        Matplotlib axes used for point-to-data conversion.
    graph : Any
        Graph exposing direction and edge metadata.
    edge_idx : int
        Edge index in the graph.
    curve : BezierCurve
        Routed curve whose endpoints already sit on node boundaries.
    style : EdgeStyle
        Effective edge style for the edge.

    Returns
    -------
    BezierCurve
        Original curve unless it is a strict-theme long back edge, otherwise a
        lateral cubic with a fixed point-floor offset.
    """
    if curve.waypoints is not None or not _is_graphviz_strict_render(graph):
        return curve
    if str(getattr(graph, "direction", "TB")).upper() not in {"TB", "BT"}:
        return curve
    back_edge_mask = getattr(graph, "_back_edge_mask", None)
    is_back_edge = False
    if back_edge_mask is not None and edge_idx < int(back_edge_mask.shape[0]):
        is_back_edge = bool(back_edge_mask[edge_idx].item())
    is_back_edge = is_back_edge or float(getattr(style, "curvature", 0.0)) > 0.0
    if not is_back_edge:
        return curve

    sx, sy = curve.p0
    tx, ty = curve.p1
    dx = float(tx - sx)
    dy = float(ty - sy)
    dist = float(np.hypot(dx, dy))
    if dist <= 1e-9:
        return curve

    floor_offset = _points_to_data_units(
        ax,
        _GRAPHVIZ_STRICT_BACK_EDGE_OFFSET_FLOOR_POINTS,
        "x",
    )
    curvature = abs(float(getattr(style, "curvature", 0.0)))
    fractional_offset = dist * min(curvature, 2.0) * _GRAPHVIZ_STRICT_BACK_EDGE_OFFSET_FACTOR
    offset = max(fractional_offset, floor_offset)
    perp_x = -dy / dist
    perp_y = dx / dist
    side = 1.0 if perp_x >= 0.0 else -1.0
    if curve.cp1[0] < min(sx, tx) or curve.cp2[0] < min(sx, tx):
        side = -1.0
    return BezierCurve(
        p0=curve.p0,
        cp1=(sx + side * perp_x * offset, sy + side * perp_y * offset),
        cp2=(tx + side * perp_x * offset, ty + side * perp_y * offset),
        p1=curve.p1,
        direction=curve.direction,
    )


def _graphviz_strict_back_edge_curves(
    ax: Any,
    graph: Any,
    curves: List[BezierCurve],
) -> List[BezierCurve]:
    """Return curves with strict-theme back-edge curvature floors applied.

    Parameters
    ----------
    ax : Any
        Matplotlib axes used for point-to-data conversion.
    graph : Any
        Graph exposing edge styles.
    curves : list[BezierCurve]
        Routed curves in edge order.

    Returns
    -------
    list[BezierCurve]
        Original list for non-strict themes, otherwise a shallow adjusted copy.
    """
    if not _is_graphviz_strict_render(graph):
        return curves
    return [
        _graphviz_strict_back_edge_curve(
            ax,
            graph,
            edge_idx,
            curve,
            _edge_style_for_render(graph, edge_idx),
        )
        for edge_idx, curve in enumerate(curves)
    ]


def _build_custom_edge_collection(
    ax: Any,
    graph: Any,
    curves: List[BezierCurve],
    positions: Optional[np.ndarray] = None,
    cluster_membership: Optional[Dict[int, List[str]]] = None,
    cluster_bboxes: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
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
    positions : numpy.ndarray | None, default=None
        Node positions with shape ``[N, 2]`` in data coordinates.
    cluster_membership : dict[int, list[str]] | None, default=None
        Node-to-cluster membership for body-only clipping.
    cluster_bboxes : dict[str, tuple[float, float, float, float]] | None, default=None
        Visible cluster rectangles for body-only clipping.

    Returns
    -------
    DaguaEdgeCollection
        Prepared custom edge collection.
    """
    # Round 11 F3: graphviz_strict matches dot's invariant arrow size; suppress
    # the SHORT_EDGE_HEAD_FRACTION clamp so panels with short edges (tiny_graph,
    # single_edge) get the same arrowhead silhouette as panels with long edges
    # (pipeline, colors_showcase).
    disable_curve_length_clamp = _is_graphviz_strict_render(graph)
    display_scale = _compute_display_scale(ax)
    edges: List[DaguaEdge] = []
    for e_idx, curve in enumerate(curves):
        style = _edge_style_for_render(graph, e_idx)
        src_idx = int(graph.edge_index[0, e_idx])
        tgt_idx = int(graph.edge_index[1, e_idx])
        is_self_loop = src_idx == tgt_idx
        src_node_width = float(graph.node_sizes[src_idx, 0])
        src_node_height = float(graph.node_sizes[src_idx, 1])
        tgt_node_width = float(graph.node_sizes[tgt_idx, 0])
        tgt_node_height = float(graph.node_sizes[tgt_idx, 1])
        head_length, head_width = _resolved_marker_dimensions(
            ax,
            style,
            tgt_node_width,
            tgt_node_height,
            is_self_loop=is_self_loop,
            scale_with_edge_width=True,
        )
        tail_length, tail_width = _resolved_marker_dimensions(
            ax,
            style,
            src_node_width,
            src_node_height,
            is_self_loop=is_self_loop,
            scale_with_edge_width=True,
        )
        taper_width_start: Optional[float] = None
        taper_width_end: Optional[float] = None
        if getattr(style, "taper", False):
            taper_width_start, taper_width_end = _resolved_taper_widths(ax, style)
        label = graph.edge_labels[e_idx] if e_idx < len(graph.edge_labels) else None
        arrowhead = str(style.arrow)
        tail_arrow = str(style.tail_arrow)
        source_arrow = str(getattr(style, "source_arrow", "none"))
        mid_arrow = str(getattr(style, "mid_arrow", "none"))
        line_dash_pattern = getattr(style, "line_dash_pattern", None)
        edge_linestyle = (
            tuple(float(value) * display_scale for value in line_dash_pattern)
            if line_dash_pattern is not None
            else style.style
        )
        line_wave = bool(getattr(style, "line_wave", False))
        line_wave_amplitude = float(getattr(style, "line_wave_amplitude", 4.0)) * display_scale
        line_wave_wavelength = float(getattr(style, "line_wave_wavelength", 16.0)) * display_scale
        if (
            str(getattr(graph, "direction", "")).upper() == "BT"
            and arrowhead == "none"
            and tail_arrow != "none"
        ):
            # Graphviz-positioned BT renders can express normal target heads
            # as tail-only markers after coordinate conversion. The curve
            # endpoint p1 still sits on the receiver boundary, so normalize
            # those marker-only edges back to a head for rendering.
            arrowhead = tail_arrow
            tail_arrow = "none"
        body_segments: Optional[List[RenderBezier]] = None
        body_clip_terminal: Optional[str] = None
        if cluster_membership is not None and cluster_bboxes is not None:
            body_segments, body_clip_terminal = _clip_render_body_segments_at_clusters(
                curve,
                src_idx,
                tgt_idx,
                cluster_membership,
                cluster_bboxes,
            )
        if body_segments is not None and len(body_segments) > 1:
            body_width = _edge_width_data_units(ax, float(style.width))
            min_visible_width = _minimum_visible_edge_width_data_units(ax)
            for segment_idx, segment in enumerate(body_segments):
                edges.append(
                    DaguaEdge(
                        curve=segment,
                        width=body_width,
                        tapered=bool(getattr(style, "taper", False)),
                        taper_width_start=taper_width_start,
                        taper_width_end=taper_width_end,
                        color=str(style.color or "#8C8C8C"),
                        alpha=float(style.opacity if style.opacity is not None else 0.7),
                        linestyle=edge_linestyle,
                        line_wave=line_wave,
                        line_wave_amplitude=line_wave_amplitude,
                        line_wave_wavelength=line_wave_wavelength,
                        arrowhead="none",
                        tail_arrow="none",
                        stroke_width=body_width,
                        label=None,
                        group_key=(e_idx, segment_idx),
                        source_node=src_idx,
                        target_node=tgt_idx,
                        disable_curve_length_clamp=disable_curve_length_clamp,
                        min_visible_width=min_visible_width,
                    )
                )
            marker_curve = _curve_to_render_bezier(curve)
            marker_body = RenderBezier.from_points(
                marker_curve.p0,
                marker_curve.p0,
                marker_curve.p0,
                marker_curve.p0,
            )
            edges.append(
                DaguaEdge(
                    curve=marker_curve,
                    body_curve=marker_body,
                    body_clip_terminal="both",
                    width=body_width,
                    color=str(style.color or "#8C8C8C"),
                    alpha=float(style.opacity if style.opacity is not None else 0.7),
                    linestyle=edge_linestyle,
                    arrowhead=arrowhead,
                    tail_arrow=tail_arrow,
                    source_arrow=source_arrow,
                    mid_arrow=mid_arrow,
                    arrowhead_length=head_length,
                    arrowhead_width=head_width,
                    tail_arrow_length=tail_length,
                    tail_arrow_width=tail_width,
                    arrow_fill=str(style.arrow_fill),
                    arrow_color=str(style.arrow_color) if style.arrow_color else None,
                    stroke_width=body_width,
                    label=label,
                    label_position=float(style.label_position),
                    label_offset=float(style.label_offset),
                    label_rotate=bool(getattr(style, "label_autorotate", False)),
                    label_side=str(style.label_side),
                    label_font_size=float(style.label_font_size),
                    label_font_color=str(style.label_font_color),
                    label_background=str(style.label_background),
                    label_font_family=str(style.label_font_family),
                    label_font_weight=str(style.label_font_weight),
                    label_outline_color=str(getattr(style, "text_outline_color", "")),
                    label_outline_width=float(getattr(style, "text_outline_width", 0.0)),
                    line_wave=line_wave,
                    line_wave_amplitude=line_wave_amplitude,
                    line_wave_wavelength=line_wave_wavelength,
                    group_key=(e_idx, -1),
                    source_node=src_idx,
                    target_node=tgt_idx,
                    disable_curve_length_clamp=disable_curve_length_clamp,
                    min_visible_width=min_visible_width,
                )
            )
            continue
        body_curve = body_segments[0] if body_segments else None
        render_curve = (
            body_curve
            if body_curve is not None and body_clip_terminal is not None
            else _curve_to_render_bezier(curve)
        )
        body_width = _edge_width_data_units(ax, float(style.width))
        min_visible_width = _minimum_visible_edge_width_data_units(ax)
        edges.append(
            DaguaEdge(
                curve=render_curve,
                body_curve=body_curve,
                body_clip_terminal=body_clip_terminal,
                width=body_width,
                tapered=bool(getattr(style, "taper", False)),
                taper_width_start=taper_width_start,
                taper_width_end=taper_width_end,
                color=str(style.color or "#8C8C8C"),
                alpha=float(style.opacity if style.opacity is not None else 0.7),
                linestyle=edge_linestyle,
                arrowhead=arrowhead,
                tail_arrow=tail_arrow,
                source_arrow=source_arrow,
                mid_arrow=mid_arrow,
                arrowhead_length=head_length,
                arrowhead_width=head_width,
                tail_arrow_length=tail_length,
                tail_arrow_width=tail_width,
                arrow_fill=str(style.arrow_fill),
                arrow_color=str(style.arrow_color) if style.arrow_color else None,
                stroke_width=body_width,
                label=label,
                label_position=float(style.label_position),
                label_offset=float(style.label_offset),
                label_rotate=bool(getattr(style, "label_autorotate", False)),
                label_side=str(style.label_side),
                label_font_size=float(style.label_font_size),
                label_font_color=str(style.label_font_color),
                label_background=str(style.label_background),
                label_font_family=str(style.label_font_family),
                label_font_weight=str(style.label_font_weight),
                label_outline_color=str(getattr(style, "text_outline_color", "")),
                label_outline_width=float(getattr(style, "text_outline_width", 0.0)),
                line_wave=line_wave,
                line_wave_amplitude=line_wave_amplitude,
                line_wave_wavelength=line_wave_wavelength,
                group_key=(src_idx, tgt_idx),
                source_node=src_idx,
                target_node=tgt_idx,
                disable_curve_length_clamp=disable_curve_length_clamp,
                min_visible_width=min_visible_width,
            )
        )
    collection = DaguaEdgeCollection(edges)
    _offset_custom_edge_collection_terminals(
        collection,
        graph,
        positions,
        _effective_stroke_scale(ax),
    )
    return collection


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
        # Upright triangles look bottom-heavy when centered geometrically. The
        # ``h / 8`` lift is an optical correction so text reads centered inside
        # the visible mass of the triangle rather than its bounding box.
        return y - h / 8
    return y


def _resolved_taper_widths(ax: Any, style: Any) -> Tuple[float, float]:
    """Resolve tapered edge widths in data units with a visible end-width floor.

    Parameters
    ----------
    ax : Any
        Matplotlib axes used for point-to-data conversion.
    style : Any
        Edge style object providing taper widths in display points.

    Returns
    -------
    tuple[float, float]
        Source-end and target-end ribbon widths in data units.
    """
    width_start = _edge_width_data_units(ax, float(style.taper_width_start))
    width_end = _edge_width_data_units(ax, float(style.taper_width_end))
    # Keep the terminal width above the renderer's visibility floor so tapered
    # ribbons end in a crisp tip instead of disappearing into zero-width
    # geometry that raster backends alias inconsistently.
    return width_start, max(width_end, MIN_TAPER_WIDTH)


def _draw_node_labels(
    ax: Any,
    graph: Any,
    pos: np.ndarray,
    sizes: np.ndarray,
    clip_patches: Optional[Sequence[Any]] = None,
    svg_hover_map: Optional[Dict[str, str]] = None,
    density_size_factor: float = 1.0,
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
    density_size_factor : float, default=1.0
        Render-time node-size multiplier from density-aware shrink. The label
        pass applies this with a readability floor and does not mutate the
        canonical node style.
    """
    gs = _graph_style_for_render(graph)
    display_scale = _compute_display_scale(ax)
    density_font_factor = max(float(density_size_factor), _DENSITY_LABEL_FONT_FLOOR)
    clip_patch_seq: Sequence[Any] = clip_patches or []
    shadow_specs: List[DaguaText] = []
    specs: List[DaguaText] = []

    for i in range(graph.num_nodes):
        label = graph.node_labels[i]
        if not label:
            continue

        x, y = float(pos[i, 0]), float(pos[i, 1])
        w, h = float(sizes[i, 0]), float(sizes[i, 1])
        style = _node_style_for_render(graph, i)
        font_weight = _normalize_text_font_weight(style.font_weight)
        clip_patch = clip_patch_seq[i] if i < len(clip_patch_seq) else None
        label_y = _label_reference_y(y, h, style.shape)

        font_size_override = getattr(style, "font_size_override_points", None)
        if font_size_override is not None:
            # Pixel-unit overrides intentionally bypass density shrink and
            # label-fit feedback so the requested display size is authoritative.
            font_size_points = float(font_size_override)
        elif graph.node_font_sizes is not None and i < graph.node_font_sizes.shape[0]:
            font_size_points = float(graph.node_font_sizes[i].item())
        else:
            font_size_points = float(style.font_size)
        if font_size_override is None and abs(float(density_size_factor) - 1.0) > 1e-12:
            font_size_points *= density_font_factor
        # Font sizes are in the same data-coordinate system as node
        # sizes.  compute_node_size already determined the correct font
        # (possibly shrunk for shrink_text policy) and sized the node
        # to contain it.  Use that font directly -- no height-based
        # rescaling needed.
        font_size_data = font_size_points
        if font_size_override is None and _is_bold_font_weight(font_weight):
            font_size_data *= _BOLD_NODE_LABEL_SIZE_MULTIPLIER

        pad_x = float(style.padding[0]) * display_scale
        pad_y = float(style.padding[1]) * display_scale
        if style.text_valign in {"top", "bottom"}:
            # Reserve at least two display points vertically so top/bottom
            # aligned labels do not kiss the stroke after scaling and outline
            # expansion are applied.
            pad_y = max(pad_y, 2.0 * display_scale)
        max_width: Optional[float] = None
        text_max_width: Optional[float] = None
        if style.text_max_width is not None:
            # Node sizing interprets text_max_width alongside point-sized text.
            # Convert it here so render-time wrapping uses the same character
            # budget after render_text switches back to data-coordinate glyphs.
            text_max_width = float(style.text_max_width) * display_scale

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

        if style.shape == "ellipse":
            ellipse_vertical_inset = h * _ELLIPSE_VERTICAL_LABEL_INSET_FRACTION
            if style.text_valign == "top":
                text_y -= ellipse_vertical_inset
            elif style.text_valign == "bottom":
                text_y += ellipse_vertical_inset

        is_rich = style.label_format == "rich"
        secondary = gs.node_label_secondary_scale if not is_rich else 1.0

        text_bg = style.text_background if style.text_background else None
        text_bg_alpha = style.text_background_opacity
        text_bg_corner_radius = style.text_background_corner_radius

        if font_size_override is not None:
            label_spec = DaguaText(
                x=text_x,
                y=text_y,
                text=label,
                font_size=font_size_points,
                font_family=_text_font_family(style),
                font_weight=font_weight,
                font_style=style.font_style,
                font_color=style.font_color,
                alpha=min(max(float(style.text_opacity), 0.0), 1.0),
                ha=style.text_align,
                va=style.text_valign,
                rotation=float(style.text_rotation),
                background=text_bg,
                background_alpha=text_bg_alpha,
                background_padding=style.text_background_padding,
                background_corner_radius=text_bg_corner_radius,
                clip_patch=clip_patch if style.overflow_policy != "overflow" else None,
                clip_on=style.overflow_policy != "overflow",
                zorder=3.0,
                gid=f"dagua-node-label-{i}",
            )
            for shadow_spec in _text_shadow_specs(label_spec, style, display_scale):
                _draw_display_point_text(ax, shadow_spec, svg_hover_map)
            _draw_display_point_text(ax, label_spec, svg_hover_map)
            continue

        label_spec = DaguaText(
            x=text_x,
            y=text_y,
            text=label,
            # ``render_text`` multiplies by display_scale to recover data units.
            font_size=(
                font_size_points
                if font_size_override is not None
                else _effective_font_size_points(font_size_data, display_scale)
            ),
            font_family=_text_font_family(style),
            font_weight=font_weight,
            font_style=style.font_style,
            font_color=style.font_color,
            alpha=min(max(float(style.text_opacity), 0.0), 1.0),
            ha=style.text_align,
            va=style.text_valign,
            rotation=float(style.text_rotation),
            rich=is_rich,
            line_spacing=1.2,
            secondary_scale=secondary,
            max_width=max_width,
            min_font_size=style.min_font_size,
            text_wrap=style.text_wrap,
            text_max_width=text_max_width,
            text_transform=style.text_transform,
            outline=style.text_outline,
            outline_color=style.text_outline_color,
            outline_width=style.text_outline_width,
            background=text_bg,
            background_alpha=text_bg_alpha,
            background_padding=style.text_background_padding,
            background_corner_radius=text_bg_corner_radius,
            clip_patch=clip_patch if style.overflow_policy != "overflow" else None,
            clip_on=style.overflow_policy != "overflow",
            zorder=3.0,
            gid=f"dagua-node-label-{i}",
        )
        shadow_specs.extend(_text_shadow_specs(label_spec, style, display_scale))
        specs.append(label_spec)

    render_text(ax, [*shadow_specs, *specs], display_scale, svg_hover_map)


def _draw_external_labels(
    ax: Any,
    graph: Any,
    pos: np.ndarray,
    sizes: np.ndarray,
    svg_hover_map: Optional[Dict[str, str]] = None,
) -> None:
    """Draw render-only node labels anchored outside node boundaries.

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
    None
        Adds external-label artists to ``ax``.
    """
    display_scale = _compute_display_scale(ax)
    specs: List[DaguaText] = []

    for i in range(graph.num_nodes):
        style = _node_style_for_render(graph, i)
        external_label = str(getattr(style, "external_label", ""))
        if external_label.strip() == "":
            continue

        cx = float(pos[i, 0])
        cy = float(pos[i, 1])
        half_width = float(sizes[i, 0]) / 2.0
        half_height = float(sizes[i, 1]) / 2.0
        offset = max(float(style.external_label_offset), 6.0) * display_scale
        position = _normalize_external_label_position(style.external_label_position)
        font_weight = _normalize_text_font_weight(style.font_weight)
        font_size_data = _node_relative_font_size_data(
            external_label,
            float(sizes[i, 1]),
            float(style.external_label_font_size),
            _DEFAULT_EXTERNAL_LABEL_FONT_POINTS,
            font_weight=font_weight,
        )

        text_x, text_y, ha, va = _external_label_anchor(
            cx, cy, half_width, half_height, offset, position
        )

        specs.append(
            DaguaText(
                x=text_x,
                y=text_y,
                text=external_label,
                font_size=_effective_font_size_points(font_size_data, display_scale),
                font_family=_text_font_family(style),
                font_weight=font_weight,
                font_style=style.font_style,
                font_color=style.external_label_font_color or style.font_color,
                ha=ha,
                va=va,
                clip_on=False,
                zorder=3.05,
                gid=f"dagua-node-external-label-{i}",
            )
        )

    if specs:
        render_text(ax, specs, display_scale, svg_hover_map)


def _draw_edge_marker(
    ax: Any,
    point: Tuple[float, float],
    direction: Tuple[float, float],
    marker: str,
    style: Any,
    node_width: float = 0.0,
    node_height: float = 0.0,
    is_self_loop: bool = False,
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
    node_width : float, default=0.0
        Width of the connected node in data units.
    node_height : float, default=0.0
        Height of the connected node in data units. Used only when the style
        enables node-relative arrow sizing.
    is_self_loop : bool, default=False
        Whether the marker belongs to a self-loop edge.

    Returns
    -------
    None
        Mutates ``ax`` in place by adding the marker artist when applicable.

    Notes
    -----
    Marker dimensions now scale with edge width for this path. Earlier versions
    used a fixed point size, which made arrowheads look undersized on thick
    rendered ribbons compared with the custom edge collection.
    """
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Circle, Polygon

    dx, dy = direction
    dist = float(np.hypot(dx, dy))
    if dist <= 1e-9 or marker == "none":
        return

    ux, uy = dx / dist, dy / dist
    px, py = -uy, ux
    manual_length, manual_width = _resolved_marker_dimensions(
        ax,
        style,
        node_width,
        node_height,
        is_self_loop=is_self_loop,
        scale_with_edge_width=True,
    )
    # Graphviz-style calibration expects arrowheads to read slightly heavier
    # than the edge stroke, so keep marker fill/outline fully opaque.
    color = to_rgba(style.arrow_color or style.color, 1.0)
    filled = style.arrow_fill == "filled" and marker not in {"vee", "tee"}
    tip_x, tip_y = point
    outline_width = _edge_width_data_units(ax, float(style.width))
    emphasis_width = _edge_width_data_units(ax, max(float(style.width) * 1.8, 2.0))

    if marker == "normal":
        # Filled triangle with tip at the edge endpoint (node boundary)
        # and body extending into the gap between nodes.
        base_x = tip_x - ux * manual_length
        base_y = tip_y - uy * manual_length
        vertices = [
            (tip_x, tip_y),
            (base_x + px * manual_width * 0.6, base_y + py * manual_width * 0.6),
            (base_x - px * manual_width * 0.6, base_y - py * manual_width * 0.6),
        ]
        if filled:
            polygon = Polygon(
                vertices,
                closed=True,
                facecolor=color,
                edgecolor="none",
                linewidth=0.0,
                joinstyle="round",
                zorder=3,
            )
            ax.add_patch(polygon)
        else:
            _draw_outline_segments(
                ax=ax,
                points=vertices,
                stroke_width=outline_width,
                color=color,
                zorder=3,
                closed=True,
            )
        return

    if marker == "vee":
        base_x = tip_x - ux * manual_length
        base_y = tip_y - uy * manual_length
        _draw_outline_segments(
            ax=ax,
            points=[
                (base_x + px * manual_width * 0.7, base_y + py * manual_width * 0.7),
                (tip_x, tip_y),
                (base_x - px * manual_width * 0.7, base_y - py * manual_width * 0.7),
            ],
            stroke_width=emphasis_width,
            color=color,
            zorder=3,
            closed=False,
        )
        return

    if marker == "open":
        base_x = tip_x - ux * manual_length
        base_y = tip_y - uy * manual_length
        vertices = [
            (tip_x, tip_y),
            (base_x + px * manual_width * 0.6, base_y + py * manual_width * 0.6),
            (base_x - px * manual_width * 0.6, base_y - py * manual_width * 0.6),
        ]
        polygon = Polygon(
            vertices,
            closed=True,
            facecolor=color,
            edgecolor="none",
            linewidth=0.0,
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
        if is_filled:
            circle_patch = Circle(
                (center_x, center_y),
                radius,
                facecolor=color,
                edgecolor="none",
                linewidth=0.0,
                zorder=3,
            )
            ax.add_patch(circle_patch)
        else:
            _draw_circle_ring_patch(
                ax=ax,
                center=(center_x, center_y),
                radius=radius,
                stroke_width=outline_width,
                color=color,
                zorder=3,
            )
        return

    if marker == "diamond":
        mid_x = tip_x - ux * (manual_length / 2)
        mid_y = tip_y - uy * (manual_length / 2)
        back_x = tip_x - ux * manual_length
        back_y = tip_y - uy * manual_length
        vertices = [
            (tip_x, tip_y),
            (mid_x + px * manual_width / 2, mid_y + py * manual_width / 2),
            (back_x, back_y),
            (mid_x - px * manual_width / 2, mid_y - py * manual_width / 2),
        ]
        if filled:
            diamond = Polygon(
                vertices,
                closed=True,
                facecolor=color,
                edgecolor="none",
                linewidth=0.0,
                joinstyle="round",
                zorder=3,
            )
            ax.add_patch(diamond)
        else:
            _draw_outline_segments(
                ax=ax,
                points=vertices,
                stroke_width=outline_width,
                color=color,
                zorder=3,
                closed=True,
            )
        return

    if marker == "tee":
        # Use a thin rectangle instead of a thick line so the tee reads as a
        # wide, flat bar instead of a square cap at small render sizes.
        bar_x = tip_x - ux * (manual_length / 4)
        bar_y = tip_y - uy * (manual_length / 4)
        bar_half_span = manual_width * 1.45
        bar_half_thick = max(emphasis_width * 0.75, manual_length / 5.0)
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
            edgecolor="none",
            linewidth=0.0,
            zorder=3,
        )
        ax.add_patch(polygon)
        return

    if marker == "crow":
        back_x = tip_x - ux * manual_length
        back_y = tip_y - uy * manual_length
        notch_x = tip_x - ux * (manual_length * 0.48)
        notch_y = tip_y - uy * (manual_length * 0.48)
        notch_half = manual_width * 0.14
        vertices = [
            (tip_x, tip_y),
            (back_x + px * manual_width * 0.5, back_y + py * manual_width * 0.5),
            (notch_x + px * notch_half, notch_y + py * notch_half),
            (back_x, back_y),
            (notch_x - px * notch_half, notch_y - py * notch_half),
            (back_x - px * manual_width * 0.5, back_y - py * manual_width * 0.5),
        ]
        polygon = Polygon(
            vertices,
            closed=True,
            facecolor=color,
            edgecolor="none",
            linewidth=0.0,
            joinstyle="round",
            zorder=3,
        )
        ax.add_patch(polygon)
        return


def _draw_port_indicators(ax: Any, graph: Any, curves: List[BezierCurve]) -> None:
    """Draw optional source/target port indicators at edge boundary contacts.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    graph : Any
        Graph exposing per-edge styles.
    curves : list[BezierCurve]
        Routed edge centerlines whose endpoints already touch node boundaries.
    """

    from matplotlib.colors import to_rgba
    from matplotlib.patches import PathPatch

    display_scale = _compute_display_scale(ax)
    for edge_idx, curve in enumerate(curves):
        style = _edge_style_for_render(graph, edge_idx)
        indicator = str(getattr(style, "port_indicator", "none")).lower()
        if indicator == "none":
            continue

        size_points = max(float(getattr(style, "port_indicator_size", 5.0)), 5.0)
        if size_points <= 0.0:
            continue

        face_color = to_rgba(str(style.color), alpha=float(getattr(style, "opacity", 1.0)))
        outline_color = to_rgba("#ffffff")
        outer_radius = size_points * display_scale * 0.5
        border_width = min(
            _PORT_INDICATOR_BORDER_WIDTH_POINTS * display_scale,
            outer_radius * 0.45,
        )
        inner_radius = max(outer_radius - border_width, outer_radius * 0.2)

        for endpoint_name, point in (("source", curve.p0), ("target", curve.p1)):
            center = (float(point[0]), float(point[1]))
            outer_path = _port_indicator_path(center, outer_radius, indicator)
            inner_path = _port_indicator_path(center, inner_radius, indicator)
            outline_patch = PathPatch(
                outer_path,
                facecolor=outline_color,
                edgecolor="none",
                linewidth=0.0,
                zorder=_PORT_INDICATOR_ZORDER,
            )
            fill_patch = PathPatch(
                inner_path,
                facecolor=face_color,
                edgecolor="none",
                linewidth=0.0,
                zorder=_PORT_INDICATOR_ZORDER + 0.01,
            )
            gid = f"dagua-port-indicator-{edge_idx}-{endpoint_name}"
            fill_patch.set_gid(gid)
            ax.add_patch(outline_patch)
            ax.add_patch(fill_patch)


def _port_indicator_path(center: Tuple[float, float], radius: float, indicator: str) -> Any:
    """Return a filled data-coordinate path for a port indicator glyph.

    Parameters
    ----------
    center : tuple[float, float]
        Marker center in data coordinates.
    radius : float
        Marker half-size in data units.
    indicator : str
        Port indicator shape token.

    Returns
    -------
    Any
        Closed matplotlib path in data coordinates.
    """
    from matplotlib.path import Path

    x, y = center
    resolved_radius = max(float(radius), 0.0)
    if indicator == "square":
        points = np.array(
            [
                [x - resolved_radius, y - resolved_radius],
                [x + resolved_radius, y - resolved_radius],
                [x + resolved_radius, y + resolved_radius],
                [x - resolved_radius, y + resolved_radius],
                [x - resolved_radius, y - resolved_radius],
            ],
            dtype=float,
        )
    elif indicator == "diamond":
        points = np.array(
            [
                [x, y + resolved_radius],
                [x + resolved_radius, y],
                [x, y - resolved_radius],
                [x - resolved_radius, y],
                [x, y + resolved_radius],
            ],
            dtype=float,
        )
    else:
        angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False, dtype=float)
        points = np.column_stack(
            [
                x + np.cos(angles) * resolved_radius,
                y + np.sin(angles) * resolved_radius,
            ]
        )
        points = np.vstack([points, points[:1]])
    codes = [Path.MOVETO] + [Path.LINETO] * (points.shape[0] - 2) + [Path.CLOSEPOLY]
    return Path(points, codes)


def _draw_edges(
    ax: Any,
    graph: Any,
    curves: List[BezierCurve],
    positions: Optional[np.ndarray] = None,
    svg_hover_map: Optional[Dict[str, str]] = None,
    cluster_membership: Optional[Dict[int, List[str]]] = None,
    cluster_bboxes: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
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
    positions : numpy.ndarray | None, default=None
        Node positions with shape ``[N, 2]`` in data coordinates.
    svg_hover_map : dict[str, str], optional
        SVG hover text accumulator.
    cluster_membership : dict[int, list[str]] | None, default=None
        Node-to-cluster membership for edge-body clipping.
    cluster_bboxes : dict[str, tuple[float, float, float, float]] | None, default=None
        Visible cluster rectangles for edge-body clipping.
    Returns
    -------
    dagua.render.edges.collection.DaguaEdgeCollection | None
        The prepared custom collection so the label pass can reuse it.
    """
    if not curves:
        if hasattr(graph, "_cached_crossings"):
            graph._cached_crossings = []
        return None
    del svg_hover_map
    crossings: List[EdgeCrossing] = []
    if any(
        getattr(graph.get_style_for_edge(edge_idx), "crossing_style", "none") != "none"
        for edge_idx in range(len(curves))
    ):
        crossings = detect_crossings(curves, edge_count=len(curves))
    if hasattr(graph, "_cached_crossings"):
        graph._cached_crossings = crossings
    if any(
        _edge_requires_direct_render(graph.get_style_for_edge(edge_idx))
        for edge_idx in range(len(curves))
    ):
        _draw_edges_direct(
            ax,
            graph,
            curves,
            crossings=crossings,
            positions=positions,
            cluster_membership=cluster_membership,
            cluster_bboxes=cluster_bboxes,
        )
        return None
    collection = _build_custom_edge_collection(
        ax,
        graph,
        curves,
        positions=positions,
        cluster_membership=cluster_membership,
        cluster_bboxes=cluster_bboxes,
    )
    body_artists = collection.render_bodies(ax)
    for body_artist in body_artists:
        if not hasattr(body_artist, "get_paths"):
            continue
        paths = body_artist.get_paths()
        if not paths:
            continue
        from matplotlib.patches import PathPatch

        ax.add_patch(
            PathPatch(
                paths[0],
                facecolor="none",
                edgecolor="none",
                linewidth=0.0,
                alpha=0.0,
                zorder=1.0,
            )
        )
        break
    collection.render_heads(ax)
    return collection


def _endpoint_label_offset_data(
    style: Any,
    endpoint_name: str,
    avg_node_height: float,
    display_scale: float,
) -> float:
    """Return an endpoint-label offset that clears the terminal marker.

    Parameters
    ----------
    style : Any
        Edge style object.
    endpoint_name : str
        Endpoint selector. Supported values are ``"head"`` and ``"tail"``.
    avg_node_height : float
        Average node height in data units. Used when arrowheads scale relative
        to connected node size.
    display_scale : float
        Point-to-data conversion factor for the active axes.

    Returns
    -------
    float
        Endpoint label offset in data units.
    """
    label_font_points = (
        float(getattr(style, "label_font_size", _DEFAULT_EDGE_LABEL_FONT_POINTS)) * 0.85
    )
    if endpoint_name == "head":
        user_offset_points = float(getattr(style, "head_label_offset", 5.0))
        arrow_length_points = float(getattr(style, "arrow_length", 0.0))
    else:
        user_offset_points = float(getattr(style, "tail_label_offset", 5.0))
        tail_arrow_length = getattr(style, "tail_arrow_length", None)
        arrow_length_points = (
            float(tail_arrow_length)
            if tail_arrow_length is not None
            else float(getattr(style, "arrow_length", 0.0))
        )

    fraction = float(getattr(style, "arrow_node_fraction", 0.0))
    if fraction > 0.0 and avg_node_height > 0.0 and display_scale > 0.0:
        arrow_length_points = max(arrow_length_points, (avg_node_height * fraction) / display_scale)

    minimum_offset_points = arrow_length_points + (label_font_points / 2.0)
    return max(user_offset_points, minimum_offset_points) * display_scale


def _append_endpoint_edge_label_specs(
    ax: Any,
    specs: List[DaguaText],
    graph: Any,
    curves: List[BezierCurve],
    avg_node_height: float,
    display_scale: float,
    svg_hover_map: Optional[Dict[str, str]],
) -> None:
    """Append head and tail edge label specs for the current graph.

    Parameters
    ----------
    ax : Any
        Matplotlib axes receiving native override text when requested.
    specs : list[DaguaText]
        Text specs being accumulated for the render pass.
    graph : Any
        Graph exposing per-edge styles and labels.
    curves : list[BezierCurve]
        Routed edge curves.
    avg_node_height : float
        Average node height used as the graph-relative text scale reference.
    display_scale : float
        Point-to-data conversion factor for endpoint offsets.
    svg_hover_map : dict[str, str] | None
        Optional SVG hover-text accumulator.
    """
    for e_idx, curve in enumerate(curves):
        style = _edge_style_for_render(graph, e_idx)
        hover_text = _edge_hover_text(graph, e_idx) if svg_hover_map is not None else ""
        endpoint_specs = (
            (
                "head",
                str(getattr(style, "head_label", "")),
                _endpoint_label_offset_data(
                    style,
                    "head",
                    avg_node_height,
                    display_scale,
                ),
            ),
            (
                "tail",
                str(getattr(style, "tail_label", "")),
                _endpoint_label_offset_data(
                    style,
                    "tail",
                    avg_node_height,
                    display_scale,
                ),
            ),
        )
        font_size_override = getattr(style, "font_size_override_points", None)
        endpoint_label_font_size_points = (
            float(font_size_override)
            if font_size_override is not None
            else _strict_edge_label_font_size(graph, float(style.label_font_size))
        )
        for endpoint_name, label_text, label_offset in endpoint_specs:
            if not label_text:
                continue
            x, y = edge_endpoint_label_position(curve, endpoint_name, label_offset=label_offset)
            gid = f"dagua-edge-{endpoint_name}-label-{e_idx}"
            scaled_endpoint_pts = (
                endpoint_label_font_size_points
                if font_size_override is not None
                else endpoint_label_font_size_points * 0.85
            )
            absolute_font_data = _strict_absolute_edge_label_font_data(
                graph, scaled_endpoint_pts, display_scale
            )
            label_font_data = (
                absolute_font_data
                if absolute_font_data is not None
                else _edge_font_size_data(
                    label_text,
                    avg_node_height,
                    scaled_endpoint_pts,
                )
            )
            label_spec = DaguaText(
                x=x,
                y=y,
                text=label_text,
                font_size=(
                    scaled_endpoint_pts
                    if font_size_override is not None
                    else _effective_font_size_points(label_font_data, display_scale)
                ),
                font_family=str(style.label_font_family or RESOLVED_FONT),
                font_weight=str(style.label_font_weight),
                font_color=str(style.label_font_color),
                outline=bool(style.text_outline_color and style.text_outline_width > 0.0),
                outline_color=str(style.text_outline_color or "#FFFFFF"),
                outline_width=float(style.text_outline_width),
                ha="center",
                va="center",
                background=style.label_background if style.label_background else None,
                background_alpha=float(style.label_background_opacity),
                background_padding=style.label_background_padding,
                background_corner_radius=float(style.label_background_corner_radius),
                clip_on=False,
                zorder=4.0,
                gid=gid,
            )
            if font_size_override is not None:
                _draw_display_point_text(ax, label_spec, svg_hover_map)
            else:
                specs.append(label_spec)
            if svg_hover_map is not None:
                svg_hover_map[gid] = hover_text
                svg_hover_map[f"{gid}-background"] = hover_text


def _edge_label_bbox(
    spec: DaguaText,
    display_scale: float,
) -> Tuple[float, float, float, float]:
    """Return an axis-aligned edge-label bbox in data coordinates.

    Parameters
    ----------
    spec : DaguaText
        Edge label render specification.
    display_scale : float
        Points-to-data conversion factor for the current axes.

    Returns
    -------
    tuple[float, float, float, float]
        Bounding box as ``(x_min, y_min, x_max, y_max)``.
    """
    width, height = measure_text_data(
        spec.text,
        size_data=max(float(spec.font_size) * display_scale, 1e-9),
        font_family=str(spec.font_family or RESOLVED_FONT),
        font_weight=str(spec.font_weight),
        font_style=str(spec.font_style),
    )
    pad_x, pad_y = spec.background_padding if spec.background else (0.0, 0.0)
    half_width = (width + 2.0 * float(pad_x) * display_scale) / 2.0
    half_height = (height + 2.0 * float(pad_y) * display_scale) / 2.0
    return (
        float(spec.x - half_width),
        float(spec.y - half_height),
        float(spec.x + half_width),
        float(spec.y + half_height),
    )


def _bboxes_overlap(
    left: Tuple[float, float, float, float],
    right: Tuple[float, float, float, float],
) -> bool:
    """Return whether two data-coordinate bboxes overlap.

    Parameters
    ----------
    left : tuple[float, float, float, float]
        First bbox as ``(x_min, y_min, x_max, y_max)``.
    right : tuple[float, float, float, float]
        Second bbox as ``(x_min, y_min, x_max, y_max)``.

    Returns
    -------
    bool
        ``True`` when the boxes have positive overlap on both axes.
    """
    return min(left[2], right[2]) > max(left[0], right[0]) and min(left[3], right[3]) > max(
        left[1], right[1]
    )


def _resolve_edge_label_collisions(
    specs: List[DaguaText],
    directions: List[Tuple[float, float]],
    display_scale: float,
) -> None:
    """Nudge overlapping edge labels along their edge tangents.

    Parameters
    ----------
    specs : list[DaguaText]
        Mutable edge-label render specs to adjust in place.
    directions : list[tuple[float, float]]
        Edge tangent directions corresponding to ``specs``.
    display_scale : float
        Points-to-data conversion factor for the current axes.

    Returns
    -------
    None
        The input specs are modified in place.
    """
    placed: List[Tuple[float, float, float, float]] = []
    padding = float(display_scale) * _EDGE_LABEL_COLLISION_PADDING_POINTS
    for spec, direction in zip(specs, directions):
        bbox = _edge_label_bbox(spec, display_scale)
        dx, dy = direction
        length = float(np.hypot(dx, dy))
        if length <= 1e-9:
            unit_x, unit_y = 0.0, 1.0
        else:
            unit_x, unit_y = dx / length, dy / length
        step = max(bbox[3] - bbox[1], float(spec.font_size) * display_scale) + padding
        for _ in range(8):
            if not any(_bboxes_overlap(bbox, other) for other in placed):
                break
            spec.x += unit_x * step
            spec.y += unit_y * step
            bbox = _edge_label_bbox(spec, display_scale)
        placed.append(bbox)


def _draw_edge_labels(
    ax: Any,
    graph: Any,
    curves: List[BezierCurve],
    label_positions: Optional[List[Optional[Tuple[float, float]]]] = None,
    svg_hover_map: Optional[Dict[str, str]] = None,
    sizes: Optional[np.ndarray] = None,
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
    sizes : numpy.ndarray, optional
        Node sizes with shape ``[N, 2]``. When omitted, the function falls
        back to ``graph.node_sizes``.
    edge_collection : DaguaEdgeCollection | None, optional
        Prepared collection whose label geometry should be reused.
    """
    display_scale = _compute_display_scale(ax)
    specs: List[DaguaText] = []
    label_directions: List[Tuple[float, float]] = []
    if sizes is None:
        node_sizes = getattr(graph, "node_sizes", None)
        if node_sizes is None:
            sizes = np.empty((0, 2), dtype=float)
        elif hasattr(node_sizes, "detach"):
            sizes = node_sizes.detach().cpu().numpy()
        else:
            sizes = np.asarray(node_sizes, dtype=float)
    avg_node_height = float(sizes[:, 1].mean()) if sizes.size else 0.0

    if edge_collection is not None and label_positions is None:
        for e_idx, (prepared, placement) in enumerate(
            zip(edge_collection.prepared_edges, edge_collection.label_placements())
        ):
            label = prepared.edge.label
            if placement is None or not label:
                continue

            style = _edge_style_for_render(graph, e_idx)
            font_size_override = getattr(style, "font_size_override_points", None)
            label_font_size_points = (
                float(font_size_override)
                if font_size_override is not None
                else _strict_edge_label_font_size(graph, float(style.label_font_size))
            )
            absolute_font_data = _strict_absolute_edge_label_font_data(
                graph, label_font_size_points, display_scale
            )
            label_font_data = (
                absolute_font_data
                if absolute_font_data is not None
                else _edge_font_size_data(
                    label,
                    avg_node_height,
                    label_font_size_points,
                )
            )
            label_spec = DaguaText(
                x=placement.x,
                y=placement.y,
                text=label,
                font_size=(
                    label_font_size_points
                    if font_size_override is not None
                    else _effective_font_size_points(label_font_data, display_scale)
                ),
                font_family=str(style.label_font_family or RESOLVED_FONT),
                font_weight=str(style.label_font_weight),
                font_color=str(style.label_font_color),
                outline=bool(style.text_outline_color and style.text_outline_width > 0.0),
                outline_color=str(style.text_outline_color or "#FFFFFF"),
                outline_width=float(style.text_outline_width),
                ha="center",
                va="center",
                rotation=placement.angle_degrees if prepared.edge.label_rotate else 0.0,
                background=style.label_background if style.label_background else None,
                background_alpha=float(style.label_background_opacity),
                background_padding=style.label_background_padding,
                background_corner_radius=float(style.label_background_corner_radius),
                clip_on=False,
                zorder=4.0,
                gid=f"dagua-edge-label-{e_idx}",
            )
            if font_size_override is not None:
                _draw_display_point_text(ax, label_spec, svg_hover_map)
            else:
                specs.append(label_spec)
            reference_curve = prepared.body_curve or prepared.lane_curve
            label_directions.append(
                (
                    float(reference_curve.p1[0] - reference_curve.p0[0]),
                    float(reference_curve.p1[1] - reference_curve.p0[1]),
                )
            )
            if svg_hover_map is not None:
                hover_text = _edge_hover_text(graph, e_idx)
                svg_hover_map[f"dagua-edge-label-{e_idx}"] = hover_text
                svg_hover_map[f"dagua-edge-label-{e_idx}-background"] = hover_text
    else:
        for e_idx, curve in enumerate(curves):
            if e_idx >= len(graph.edge_labels):
                break
            label = graph.edge_labels[e_idx]
            if not label:
                continue

            style = _edge_style_for_render(graph, e_idx)
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

            font_size_override = getattr(style, "font_size_override_points", None)
            label_font_size_points = (
                float(font_size_override)
                if font_size_override is not None
                else _strict_edge_label_font_size(graph, float(style.label_font_size))
            )
            absolute_font_data = _strict_absolute_edge_label_font_data(
                graph, label_font_size_points, display_scale
            )
            label_font_data = (
                absolute_font_data
                if absolute_font_data is not None
                else _edge_font_size_data(
                    label,
                    avg_node_height,
                    label_font_size_points,
                )
            )
            label_spec = DaguaText(
                x=lx,
                y=ly,
                text=label,
                font_size=(
                    label_font_size_points
                    if font_size_override is not None
                    else _effective_font_size_points(label_font_data, display_scale)
                ),
                font_family=str(style.label_font_family or RESOLVED_FONT),
                font_weight=str(style.label_font_weight),
                font_color=str(style.label_font_color),
                rotation=(
                    place_edge_label(
                        _curve_to_render_bezier(curve),
                        label_position=float(style.label_position),
                        label_offset=0.0,
                        label_rotate=True,
                    ).angle_degrees
                    if bool(getattr(style, "label_autorotate", False))
                    else 0.0
                ),
                outline=bool(style.text_outline_color and style.text_outline_width > 0.0),
                outline_color=str(style.text_outline_color or "#FFFFFF"),
                outline_width=float(style.text_outline_width),
                ha="center",
                va="center",
                background=style.label_background if style.label_background else None,
                background_alpha=float(style.label_background_opacity),
                background_padding=style.label_background_padding,
                background_corner_radius=float(style.label_background_corner_radius),
                clip_on=False,
                zorder=4.0,
                gid=f"dagua-edge-label-{e_idx}",
            )
            if font_size_override is not None:
                _draw_display_point_text(ax, label_spec, svg_hover_map)
            else:
                specs.append(label_spec)
            label_directions.append(
                (
                    float(curve.p1[0] - curve.p0[0]),
                    float(curve.p1[1] - curve.p0[1]),
                )
            )
            if svg_hover_map is not None:
                hover_text = _edge_hover_text(graph, e_idx)
                svg_hover_map[f"dagua-edge-label-{e_idx}"] = hover_text
                svg_hover_map[f"dagua-edge-label-{e_idx}-background"] = hover_text

    if _is_graphviz_strict_render(graph) and len(specs) == len(label_directions):
        _resolve_edge_label_collisions(specs, label_directions, display_scale)

    _append_endpoint_edge_label_specs(
        ax,
        specs,
        graph,
        curves,
        avg_node_height,
        display_scale,
        svg_hover_map,
    )
    render_text(ax, specs, display_scale, svg_hover_map)


def _draw_clusters(
    ax: Any,
    graph: Any,
    pos: np.ndarray,
    sizes: np.ndarray,
    cluster_aware: bool = True,
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
    cluster_aware : bool, default=True
        Whether render-time cluster expansion should be capped to the
        placement footprint used by cluster-aware layout.
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
    display_scale = _compute_display_scale(ax)
    stroke_scale = _effective_stroke_scale(ax)
    cluster_parents = getattr(graph, "cluster_parents", {}) or {}

    fill_paths_by_depth: Dict[int, List[Any]] = {}
    fill_colors_by_depth: Dict[int, List[Any]] = {}
    border_paths_by_depth: Dict[int, List[Any]] = {}
    border_colors_by_depth: Dict[int, List[Any]] = {}
    cluster_label_specs: List[DaguaText] = []
    cluster_label_placements: List[_ClusterLabelPlacement] = []
    min_node_height = float(sizes[:, 1].min()) if sizes.size else 0.0

    label_gap = _points_to_data_units(ax, _CLUSTER_LABEL_VERTICAL_GAP_POINTS, "y")
    cluster_y_maxes = _compute_cluster_y_maxes(
        graph,
        pos,
        sizes,
        ordered_clusters,
        cluster_depths,
        label_gap=label_gap,
        display_scale=display_scale,
    )
    cluster_y_mins = _compute_cluster_y_mins(
        graph,
        pos,
        sizes,
        ordered_clusters,
        cluster_depths,
        label_gap=label_gap,
        display_scale=display_scale,
    )
    cluster_render_boxes = _compute_cluster_render_bboxes(
        ax,
        graph,
        pos,
        sizes,
        ordered_clusters,
        cluster_depths,
        cluster_y_mins,
        cluster_y_maxes,
        display_scale,
        cluster_aware,
        label_gap=label_gap,
    )

    for name in ordered_clusters:
        members = graph.clusters[name]
        depth = cluster_depths.get(name, 0)
        indices = collect_cluster_leaves(members) if isinstance(members, dict) else members

        if not indices:
            continue

        style = _cluster_style_for_render(graph, name)

        render_box = cluster_render_boxes.get(name)
        if render_box is None:
            continue
        x_min, y_min, x_max, y_max = render_box.bbox
        label = graph.cluster_labels.get(name, name)
        depth_fs_step = getattr(style, "depth_font_size_step", -0.5)
        font_size_override = getattr(style, "font_size_override_points", None)
        if font_size_override is not None:
            label_font_points = max(float(font_size_override), 1e-9)
        else:
            label_font_points = max(style.font_size + depth * depth_fs_step, 5.0)
        label_ff = style.font_family or RESOLVED_FONT
        label_ox = style.label_offset[0] * display_scale
        label_oy = style.label_offset[1] * display_scale
        label_text_max_width = _cluster_label_text_max_width(style, display_scale)

        cluster_height = y_max - y_min
        if font_size_override is not None:
            label_font_data = float(label_font_points) * display_scale
        else:
            label_font_data = _cluster_font_size_data(
                label,
                float(cluster_height),
                min_node_height,
                float(label_font_points),
                str(style.font_size_scaling),
                display_scale,
            )
        label_width, label_height = _measure_cluster_label_data(
            label,
            font_size_data=label_font_data,
            font_family=str(label_ff),
            font_weight=str(style.font_weight),
            text_wrap=str(style.text_wrap),
            text_max_width=label_text_max_width,
        )

        # Progressive depth variation — each depth_*_step field is additive per level.
        fill_enabled = _cluster_fill_alpha(style, depth) > 0.0
        fill_color = (
            darken_hex(style.fill, depth * style.depth_fill_step)
            if fill_enabled
            else str(style.fill)
        )
        stroke_color = darken_hex(style.stroke, depth * style.depth_stroke_step)

        fill_alpha = _cluster_fill_alpha(style, depth)
        border_alpha = _cluster_border_alpha(style, depth)

        depth_sw_step = getattr(style, "depth_stroke_width_step", 0.0)
        eff_stroke_width = max(float(style.stroke_width) + depth * depth_sw_step, 0.1)

        width = x_max - x_min
        height = y_max - y_min
        stroke_override = getattr(style, "stroke_width_override_points", None)
        if stroke_override is None:
            border_width = clamp_border_width(eff_stroke_width * stroke_scale, width, height)
        else:
            border_width = 0.0
        shape_spec = ShapeSpec(
            center_x=(x_min + x_max) / 2.0,
            center_y=(y_min + y_max) / 2.0,
            width=width,
            height=height,
            shape="roundrect",
            corner_radius=max(
                float(style.corner_radius)
                + depth * getattr(style, "depth_corner_radius_step", 0.0),
                0.0,
            )
            * display_scale,
            aspect_ratio=None,
        )
        outer_path = build_shape_path(shape_spec)
        fill_path = inset_shape_path(shape_spec, border_width) if border_width > 0.0 else outer_path

        if fill_alpha > 0.0:
            fill_paths_by_depth.setdefault(depth, []).append(fill_path)
            fill_colors_by_depth.setdefault(depth, []).append(to_rgba(fill_color, fill_alpha))
        if stroke_override is not None and float(stroke_override) > 0.0:
            centerline_path = inset_shape_path(
                shape_spec,
                float(stroke_override) * display_scale / 2.0,
            )
            _draw_display_point_path_stroke(
                ax,
                centerline_path,
                to_rgba(stroke_color, border_alpha),
                float(stroke_override),
                zorder=0.05 + depth * 0.01,
                linestyle=_display_point_linestyle(style.stroke_dash, None),
                joinstyle="round",
            )
        elif border_width > 0.0:
            if style.stroke_dash == "solid":
                border_outer_path, border_inner_path = _solid_border_ring_paths(
                    shape_spec,
                    outer_path,
                    border_width,
                    "center",
                )
                border_paths = [annular_path(border_outer_path, border_inner_path)]
            else:
                centerline_path = inset_shape_path(shape_spec, border_width / 2.0)
                border_paths = dash_ribbon_paths(centerline_path, style.stroke_dash, border_width)
            border_paths_by_depth.setdefault(depth, []).extend(border_paths)
            border_colors_by_depth.setdefault(depth, []).extend(
                [to_rgba(stroke_color, border_alpha)] * len(border_paths)
            )

        # Each cluster label sits relative to its OWN container bounds rather
        # than stacking at the outermost cluster's corner.
        lx, ly, ha, va = _cluster_label_anchor(
            str(style.label_position),
            float(x_min),
            float(x_max),
            float(y_min),
            float(y_max),
            float(label_ox),
            float(label_oy),
        )

        if label:
            label_background = _resolve_cluster_label_background(graph, style)
            label_background_alpha = (
                float(getattr(style, "label_background_opacity", 0.0))
                if label_background is not None
                else 0.0
            )
            label_background_padding = (
                tuple(getattr(style, "label_background_padding", (0.0, 0.0)))
                if label_background is not None
                else (0.0, 0.0)
            )
            label_spec = DaguaText(
                x=lx,
                y=ly,
                text=label,
                font_size=(
                    label_font_points
                    if font_size_override is not None
                    else _effective_font_size_points(label_font_data, display_scale)
                ),
                font_family=label_ff,
                font_weight=style.font_weight,
                font_color=style.font_color,
                alpha=1.0,
                ha=ha,
                va=va,
                background=label_background,
                background_alpha=label_background_alpha,
                background_padding=label_background_padding,
                background_corner_radius=float(
                    getattr(style, "label_background_corner_radius", 2.0)
                ),
                clip_on=False,
                text_wrap=style.text_wrap,
                text_max_width=label_text_max_width,
                zorder=_CLUSTER_LABEL_ZORDER + depth * 0.01,
                gid=f"dagua-cluster-label-{name}",
            )
            if font_size_override is not None:
                _draw_display_point_text(ax, label_spec, svg_hover_map)
            else:
                cluster_label_specs.append(label_spec)
                cluster_label_placements.append(
                    _ClusterLabelPlacement(
                        name=name,
                        spec=label_spec,
                        width=label_width,
                        height=label_height,
                        depth=depth,
                        parent_name=cluster_parents.get(name),
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
            border_paths=[],
            border_colors=[],
            fill_zorder=0.0 + depth * 0.01,
            border_zorder=0.05 + depth * 0.01,
        )

    # Draw all cluster strokes after all fills. With solid fill opacity, nested
    # cluster fills can otherwise cover a previously emitted border collection.
    for depth in sorted(border_paths_by_depth):
        add_filled_collections(
            ax=ax,
            fill_paths=[],
            fill_colors=[],
            border_paths=border_paths_by_depth.get(depth, []),
            border_colors=border_colors_by_depth.get(depth, []),
            fill_zorder=0.0 + depth * 0.01,
            border_zorder=0.05 + depth * 0.01,
        )

    if cluster_label_specs:
        _resolve_cluster_label_collisions(ax, cluster_label_placements)
        render_text(ax, cluster_label_specs, display_scale, svg_hover_map)


def _render_cluster_edge_clip_data(
    ax: Any,
    graph: Any,
    pos: np.ndarray,
    sizes: np.ndarray,
    cluster_aware: bool = True,
) -> Tuple[Dict[int, List[str]], Dict[str, Tuple[float, float, float, float]]]:
    """Return node membership and visible cluster boxes for edge clipping.

    Parameters
    ----------
    ax : Any
        Matplotlib axes used for display-scale dependent label geometry.
    graph : Any
        Graph exposing cluster membership and styles.
    pos : numpy.ndarray
        Node positions with shape ``[N, 2]`` in render coordinates.
    sizes : numpy.ndarray
        Node sizes with shape ``[N, 2]`` in render coordinates.
    cluster_aware : bool, default=True
        Whether render-time min-width and label-fit expansion should be capped
        to the placement footprint plus a small tolerance.

    Returns
    -------
    tuple[dict[int, list[str]], dict[str, tuple[float, float, float, float]]]
        Node-to-cluster membership and visible-stroke cluster rectangles.
    """
    if not getattr(graph, "clusters", None):
        return {}, {}

    ordered_clusters = _cluster_render_order(graph)
    cluster_depths = _cluster_depths(graph, ordered_clusters)
    display_scale = _compute_display_scale(ax)
    label_gap = _points_to_data_units(ax, _CLUSTER_LABEL_VERTICAL_GAP_POINTS, "y")
    cluster_y_maxes = _compute_cluster_y_maxes(
        graph,
        pos,
        sizes,
        ordered_clusters,
        cluster_depths,
        label_gap=label_gap,
        display_scale=display_scale,
    )
    cluster_y_mins = _compute_cluster_y_mins(
        graph,
        pos,
        sizes,
        ordered_clusters,
        cluster_depths,
        label_gap=label_gap,
        display_scale=display_scale,
    )
    cluster_render_boxes = _compute_cluster_render_bboxes(
        ax,
        graph,
        pos,
        sizes,
        ordered_clusters,
        cluster_depths,
        cluster_y_mins,
        cluster_y_maxes,
        display_scale,
        cluster_aware,
        label_gap=label_gap,
    )
    membership: Dict[int, List[str]] = {}
    bboxes: Dict[str, Tuple[float, float, float, float]] = {}

    for name in ordered_clusters:
        members = graph.clusters[name]
        indices = collect_cluster_leaves(members) if isinstance(members, dict) else members
        if not indices:
            continue
        for index in indices:
            membership.setdefault(int(index), []).append(name)

        style = _cluster_style_for_render(graph, name)
        depth = cluster_depths.get(name, 0)
        stroke_override = getattr(style, "stroke_width_override_points", None)
        has_border_width = (
            float(stroke_override) > 0.0
            if stroke_override is not None
            else float(style.stroke_width) > 0.0
        )
        if _cluster_border_alpha(style, depth) <= 0.0 or not has_border_width:
            continue

        render_box = cluster_render_boxes.get(name)
        if render_box is not None:
            bboxes[name] = render_box.bbox

    return membership, bboxes
