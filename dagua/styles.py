"""NodeStyle, EdgeStyle, ClusterStyle, GraphStyle, Theme dataclasses, color utilities.

Color system: Wong/Okabe-Ito colorblind-safe palette (Wong, B. 2011. Nature Methods 8:441).
Typography: Helvetica/Arial sans-serif per Nature/Science figure guidelines.
Aesthetics: publication-quality defaults — muted fills, strong borders, quiet edges.
"""

# Unit system notes
# -----------------
# Dagua keeps layout-facing node and edge geometry in data coordinates, but many
# cosmetic style fields are authored in display-space points so they stay
# visually stable across zoom levels and figure sizes.
#
# Point-based fields (converted by the renderer when geometry is constructed):
# - NodeStyle.stroke_width, NodeStyle.padding, NodeStyle.corner_radius
# - NodeStyle.font_size, NodeStyle.text_outline_width, NodeStyle.shadow_offset
# - EdgeStyle.width, EdgeStyle.arrow_length, EdgeStyle.arrow_width
# - EdgeStyle.taper_width_start, EdgeStyle.taper_width_end
# - EdgeStyle.label_font_size, EdgeStyle.label_offset
# - EdgeStyle.head_label_offset, EdgeStyle.tail_label_offset
# - ClusterStyle.stroke_width, ClusterStyle.padding, ClusterStyle.corner_radius
# - ClusterStyle.font_size, ClusterStyle.label_offset
# - GraphStyle margin/title and label typography fields
#
# Data-coordinate fields (used directly without point conversion):
# - Graph/node positions and computed node_sizes tensors
# - Edge routing control points and cluster bounds
# - Explicit scene geometry passed to the renderer
#
# ``dagua.render.mpl._compute_display_scale`` converts point-authored geometry
# into data units at draw time. Matplotlib-native properties such as
# ``linewidth`` and ``fontsize`` still receive point values directly because the
# backend interprets those in display space on its own.

# TODO: Add support for pixel-unit overrides (e.g., "2pt") for users who want
# fixed-size elements regardless of zoom/data scale. This would require a
# unit-aware value type like Union[float, str] where strings like "2pt" are
# parsed as display-point values and floats remain data-coordinate values.
#
# TODO: Expose additional text rendering capabilities as style fields:
# - NodeStyle.text_underline, text_strikethrough (text decorations)
# - EdgeStyle.label_outline (outline on edge labels for readability)
# - ClusterStyle.label_outline (outline on cluster labels)
# These capabilities exist in dagua/render/text/ but are not yet exposed in styles.

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from typing import Any, Dict, List, Optional, Tuple

# ─── Wong/Okabe-Ito Colorblind-Safe Palette ────────────────────────────────

PALETTE = {
    "sky": "#56B4E9",
    "vermillion": "#D55E00",
    "bluish_green": "#009E73",
    "amber": "#E69F00",
    "reddish_purple": "#CC79A7",
    "blue": "#0072B2",
    "yellow": "#F0E442",
}

# Ordered for automatic assignment when > 1 category
PALETTE_ORDER: List[str] = [
    "#56B4E9",  # sky — default
    "#0072B2",  # blue — primary computation
    "#009E73",  # bluish green
    "#E69F00",  # amber
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#F0E442",  # yellow
]

# Neutrals
NEAR_BLACK = "#2D2D2D"
DARK_GRAY = "#4A4A4A"
MEDIUM_GRAY = "#6B7280"  # darkened from #8C8C8C for edge visibility
LIGHT_GRAY = "#D4D4D4"
WARM_WHITE = "#FAFAFA"
PAPER = "#F5F5F0"

# Preferred font stack (Nature/Science figure standard)
# Helvetica Neue/Helvetica are preferred but proprietary; Arial is the
# standard substitute (metrically near-identical, universally available).
FONT_FAMILY = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
FONT_FAMILY_MONO = ["SF Mono", "Menlo", "Consolas", "DejaVu Sans Mono"]


def _resolve_font() -> str:
    """Find the best available font from the preference stack."""
    try:
        from matplotlib.font_manager import FontProperties, findfont

        for name in FONT_FAMILY:
            try:
                findfont(FontProperties(family=name), fallback_to_default=False)
                return name
            except ValueError:
                continue
    except ImportError:
        pass
    return "sans-serif"  # matplotlib's built-in fallback


# Resolved at import time — avoids repeated "font not found" warnings
RESOLVED_FONT: str = _resolve_font()

_STYLE_REPR_PRIORITY_FIELDS: Dict[str, List[str]] = {
    "NodeStyle": ["shape", "fill", "stroke", "font_size", "font_family", "font_color"],
    "EdgeStyle": ["color", "width", "arrow", "routing", "style", "opacity"],
    "ClusterStyle": ["fill", "stroke", "font_size", "padding", "opacity"],
    "GraphStyle": ["background_color", "margin"],
}


# ─── Color Utilities ────────────────────────────────────────────────────────


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def _hex_to_hsl(hex_color: str) -> Tuple[float, float, float]:
    r, g, b = [x / 255.0 for x in _hex_to_rgb(hex_color)]
    mx, mn = max(r, g, b), min(r, g, b)
    lightness = (mx + mn) / 2.0
    if mx == mn:
        h = s = 0.0
    else:
        d = mx - mn
        s = d / (2.0 - mx - mn) if lightness > 0.5 else d / (mx + mn)
        if mx == r:
            h = (g - b) / d + (6.0 if g < b else 0.0)
        elif mx == g:
            h = (b - r) / d + 2.0
        else:
            h = (r - g) / d + 4.0
        h /= 6.0
    return h, s, lightness


def _hsl_to_hex(h: float, s: float, lightness: float) -> str:
    if s == 0:
        v = int(round(lightness * 255))
        return _rgb_to_hex(v, v, v)

    def hue_to_rgb(p, q, t):
        if t < 0:
            t += 1
        if t > 1:
            t -= 1
        if t < 1 / 6:
            return p + (q - p) * 6 * t
        if t < 1 / 2:
            return q
        if t < 2 / 3:
            return p + (q - p) * (2 / 3 - t) * 6
        return p

    q = lightness * (1 + s) if lightness < 0.5 else lightness + s - lightness * s
    p = 2 * lightness - q
    r = int(round(hue_to_rgb(p, q, h + 1 / 3) * 255))
    g = int(round(hue_to_rgb(p, q, h) * 255))
    b = int(round(hue_to_rgb(p, q, h - 1 / 3) * 255))
    return _rgb_to_hex(r, g, b)


def make_fill(base_hex: str, bg_hex: str = WARM_WHITE, blend: float = 0.25) -> str:
    """Blend base color toward background for a muted fill.

    blend=0.25 means 25% base color, 75% background.
    """
    br, bg, bb = _hex_to_rgb(base_hex)
    bgr, bgg, bgb = _hex_to_rgb(bg_hex)
    r = int(round(br * blend + bgr * (1 - blend)))
    g = int(round(bg * blend + bgg * (1 - blend)))
    b = int(round(bb * blend + bgb * (1 - blend)))
    return _rgb_to_hex(r, g, b)


def border_from_fill(fill_hex: str, darken: float = 0.5) -> str:
    """Derive border color by darkening the fill in HSL lightness space."""
    h, s, lightness = _hex_to_hsl(fill_hex)
    return _hsl_to_hex(h, min(s * 1.2, 1.0), lightness * (1 - darken))


def darken_hex(hex_color: str, amount: float) -> str:
    """Darken a hex color by reducing HSL lightness by `amount` (0-1 scale)."""
    h, s, lightness = _hex_to_hsl(hex_color)
    return _hsl_to_hex(h, s, max(lightness - amount, 0.0))


def make_node_colors(base_hex: str) -> Tuple[str, str]:
    """Return (fill, stroke) pair from a base palette color.

    Fill: muted (25% blend toward warm white).
    Stroke: full-saturation base, darkened 50%.
    """
    fill = make_fill(base_hex, WARM_WHITE, blend=0.25)
    stroke = border_from_fill(base_hex, darken=0.4)
    return fill, stroke


def _compact_style_repr(style: Any) -> str:
    """Build a compact repr showing only non-default fields.

    Parameters
    ----------
    style : Any
        Dataclass style instance to summarize.

    Returns
    -------
    str
        Compact repr string with at most six changed fields.
    """
    default = style.__class__()
    changed: Dict[str, Any] = {}
    for field_info in dataclass_fields(style):
        value = getattr(style, field_info.name)
        default_value = getattr(default, field_info.name)
        if value != default_value:
            changed[field_info.name] = value

    class_name = style.__class__.__name__
    if not changed:
        return f"{class_name}()"

    ordered: List[Tuple[str, Any]] = []
    for key in _STYLE_REPR_PRIORITY_FIELDS.get(class_name, []):
        if key in changed:
            ordered.append((key, changed.pop(key)))
    ordered.extend(changed.items())

    shown = ordered[:6]
    remaining = len(ordered) - len(shown)
    parts = [f"{key}={value!r}" for key, value in shown]
    if remaining > 0:
        parts.append(f"...+{remaining}")
    return f"{class_name}({', '.join(parts)})"


# ─── Style Dataclasses ──────────────────────────────────────────────────────


@dataclass
class NodeStyle:
    """Visual style for a node.

    Notes
    -----
    Supported ``shape`` values are ``"rect"``, ``"roundrect"``, ``"ellipse"``,
    ``"diamond"``, ``"circle"``, ``"triangle"``, ``"hexagon"``,
    ``"parallelogram"``, ``"pentagon"``, ``"octagon"``, ``"star"``,
    ``"cylinder"``, ``"trapezoid"``, ``"double_circle"``, ``"cloud"``,
    ``"stadium"``, ``"tab"``, ``"note"``, ``"document"``, and ``"box3d"``.

    The ``_set_fields`` attribute tracks which fields were explicitly modified
    after construction, allowing the style cascade to distinguish between
    "field matches the class default" and "field was never set."  Call
    ``style.mark_set("field_name")`` after assigning a value that matches the
    dataclass default to ensure the cascade respects it.
    """

    shape: str = "roundrect"
    fill: str = ""  # empty = computed from base_color
    stroke: str = ""  # empty = computed from base_color
    stroke_width: float = 0.57
    stroke_dash: str = "solid"  # solid, dashed, dotted
    stroke_dash_pattern: Optional[Tuple[float, ...]] = None  # custom (on, off, ...)
    border_opacity: float = 1.0
    font_family: str = ""  # empty = use FONT_FAMILY default
    font_size: float = 9.0
    font_color: str = NEAR_BLACK
    text_align: str = "center"  # left, center, right
    text_valign: str = "center"  # top, center, bottom
    text_rotation: float = 0.0  # render-only degrees, counter-clockwise
    text_wrap: str = "none"  # Layout-affecting: none, wrap, ellipsis
    text_max_width: Optional[float] = None  # Layout-affecting width limit before wrapping
    text_transform: str = "none"  # Layout-affecting: none, uppercase, lowercase
    text_outline: bool = False
    text_outline_color: str = "#FFFFFF"
    text_outline_width: float = 2.0
    text_background: str = ""  # Background color behind node label (empty = none)
    text_background_opacity: float = 0.85
    text_background_padding: Tuple[float, float] = (3.0, 2.0)
    text_background_corner_radius: float = 2.0
    external_label: str = ""  # render-only label outside the node boundary
    external_label_position: str = "bottom"  # render-only: top, bottom, left, right
    external_label_font_size: float = 8.0  # render-only size in points
    external_label_font_color: str = ""  # render-only; empty = use font_color
    external_label_offset: float = 4.0  # render-only distance from boundary in points
    padding: Tuple[float, float] = (11.0, 9.0)  # horizontal, vertical
    corner_radius: float = 6.0
    opacity: float = 1.0
    gradient: str = "none"  # none, linear, radial
    gradient_color: str = ""  # empty = computed from fill
    gradient_angle: float = 0.0  # degrees for linear gradients
    base_color: str = PALETTE["sky"]  # Wong palette color
    # New fields (Part 2)
    font_weight: str = "regular"  # Layout-affecting: changes text width
    font_style: str = "normal"  # Layout-affecting: changes measured text width
    shadow: bool = False  # render-only decoration
    shadow_offset: Tuple[float, float] = (1.5, -1.5)  # render-only
    shadow_color: str = "#00000020"  # render-only
    shadow_blur: float = 0.0
    min_width: Optional[float] = None  # Layout-affecting: floor on node width
    min_height: Optional[float] = None  # Layout-affecting: floor on node height
    # New fields (Part 3) — overflow policy
    overflow_policy: str = "shrink_text"  # "shrink_text", "expand_node", "overflow"
    min_font_size: float = 5.0  # Floor for shrink_text policy
    label_format: str = "plain"  # plain, rich
    border_count: int = 1  # render-only: 1 = single border, 2 = double border
    border_position: str = "center"  # render-only: center, inside, outside
    stroke_cap: str = "butt"  # render-only: butt, round, square
    stroke_join: str = "miter"  # render-only: miter, bevel, round
    fill_pattern: str = "solid"  # render-only: solid, striped, hatched, pie
    fill_pattern_colors: Optional[List[str]] = None  # render-only palette for stripes/pie slices
    fill_pattern_values: Optional[List[float]] = None  # render-only pie slice proportions
    fill_pattern_angle: float = 0.0  # render-only stripe angle in degrees
    fill_pattern_hole: float = 0.0  # render-only inner radius fraction for donut pies
    image: str = ""  # render-only path or URL for node image content
    image_fit: str = "contain"  # render-only: contain, cover, stretch
    image_opacity: float = 1.0  # render-only alpha for the image layer

    def __post_init__(self):
        """Populate derived defaults after dataclass initialization."""
        object.__setattr__(self, "_set_fields", set())
        object.__setattr__(self, "_init_done", False)
        if not self.fill:
            self.fill = make_fill(self.base_color)
        if not self.stroke:
            self.stroke = border_from_fill(self.base_color, darken=0.4)
        if not self.font_family:
            self.font_family = RESOLVED_FONT
        object.__setattr__(self, "_init_done", True)

    def __setattr__(self, name: str, value: object) -> None:
        """Track fields explicitly set after initialization."""
        object.__setattr__(self, name, value)
        if getattr(self, "_init_done", False) and not name.startswith("_"):
            self._set_fields.add(name)

    def mark_set(self, field_name: str) -> None:
        """Mark a field as explicitly set for cascade priority."""
        self._set_fields.add(field_name)

    def __repr__(self) -> str:
        """Return a compact repr showing only non-default fields."""
        return _compact_style_repr(self)

    @property
    def font_family_list(self) -> List[str]:
        """Return font family as a list for matplotlib."""
        if self.font_family in (FONT_FAMILY[0], ""):
            return FONT_FAMILY
        return [self.font_family]


@dataclass
class EdgeStyle:
    """Visual style for an edge."""

    color: str = "#6B7280"  # medium gray — visible but recedes behind nodes
    width: float = 1.4
    arrow: str = "normal"  # normal, vee, dot, diamond, tee, crow, circle, open, none
    tail_arrow: str = "none"
    arrow_fill: str = "filled"  # filled, hollow
    arrow_color: str = ""  # empty = use edge color
    arrow_length: float = 12.0
    arrow_width: float = 9.0
    arrow_scale: Optional[float] = None  # Legacy field; matplotlib renderer ignores it
    arrow_node_fraction: float = 0.35  # fraction of target node height (0 = use fixed arrow_length)
    arrow_width_ratio: float = 0.85  # width = length * this ratio (for node-relative mode)
    style: str = "solid"  # solid, dashed, dotted
    line_cap: str = "butt"  # render-only: butt, round, square
    line_join: str = "miter"  # render-only: miter, bevel, round
    opacity: float = 0.75
    # New fields (Part 2)
    routing: str = "bezier"  # bezier, straight, ortho, taxi — post-layout
    label_font_size: float = 7.0  # render-only
    label_font_color: str = NEAR_BLACK  # render-only
    label_background: str = WARM_WHITE  # render-only
    label_background_opacity: float = 0.85
    label_background_padding: Tuple[float, float] = (3.0, 2.0)
    label_background_corner_radius: float = 2.0
    label_font_family: str = ""  # empty = use default
    label_font_weight: str = "regular"  # regular, bold
    # New fields (Part 3) — edge aesthetics
    label_position: float = 0.5  # Position along curve (0=start, 1=end)
    label_offset: float = 8.0  # Perpendicular distance from edge centerline
    label_side: str = "auto"  # "auto", "left", or "right" relative to edge direction
    curvature: float = 0.4  # Control point offset factor (0=straight, 1=max curve)
    port_style: str = "distributed"  # "distributed" or "center"
    label_avoidance: bool = True  # Whether to avoid label collisions
    taper: bool = False  # Taper edge body from source width to target width
    taper_width_start: float = 3.0
    taper_width_end: float = 0.5
    head_label: str = ""  # Label near the target endpoint
    tail_label: str = ""  # Label near the source endpoint
    head_label_offset: float = 5.0
    tail_label_offset: float = 5.0
    color_gradient: str = "none"  # "none", "source_to_target"
    color_gradient_end: str = ""  # empty = use the edge color for both ends
    crossing_style: str = "none"  # none, arc, gap, sharp
    crossing_size: float = 6.0  # jump marker size in points

    def __repr__(self) -> str:
        """Return a compact repr showing only non-default fields."""
        return _compact_style_repr(self)


@dataclass
class ClusterStyle:
    """Visual style for a cluster box."""

    fill: str = PAPER
    stroke: str = LIGHT_GRAY
    stroke_width: float = 0.7
    stroke_dash: str = "solid"
    corner_radius: float = 8.0
    padding: float = 38.0
    label_position: str = "top-left"  # top-left, top-center, top-right
    font_size: float = 9.5
    font_weight: str = "bold"
    font_color: str = DARK_GRAY
    opacity: float = 0.32
    # New fields (Part 2)
    font_family: str = ""  # empty = use FONT_FAMILY default, render-only
    label_offset: Tuple[float, float] = (
        10.0,
        12.0,
    )  # render-only (y-offset prevents nested label overlap)
    depth_fill_step: float = 0.03  # HSL lightness step per depth level
    depth_stroke_step: float = 0.05  # HSL lightness step per depth level
    depth_stroke_width_step: float = 0.0  # additive stroke_width change per depth (points)
    depth_opacity_step: float = -0.05  # additive opacity change per depth level
    depth_font_size_step: float = -0.5  # additive font_size change per depth (points)
    depth_padding_step: float = -3.0  # additive padding change per depth (points)
    depth_corner_radius_step: float = 0.0  # additive corner_radius change per depth (points)
    # Member style overrides — applied to all nodes/edges within this cluster
    member_node_style: Optional[NodeStyle] = None
    member_edge_style: Optional[EdgeStyle] = None

    # Legacy constants kept for reference but replaced by depth_*_step
    LEVEL_FILLS = [PAPER, "#EDEDE8", "#E5E5E0"]
    LEVEL_STROKES = [LIGHT_GRAY, "#C8C8C8", "#BCBCBC"]

    def __repr__(self) -> str:
        """Return a compact repr showing only non-default fields."""
        return _compact_style_repr(self)


@dataclass
class GraphStyle:
    """Graph-level visual settings (all render-only, no layout effect)."""

    background_color: str = WARM_WHITE
    margin: float = 18.0
    title_font_size: float = 10.0
    title_font_weight: str = "regular"
    title_font_color: str = NEAR_BLACK
    title_font_family: str = ""
    edge_label_font_size: float = 7.0
    edge_label_background: str = WARM_WHITE
    edge_label_background_opacity: float = 0.85
    node_label_secondary_scale: float = 0.85
    max_figsize: Tuple[float, float] = (30.0, 40.0)
    min_figsize: Tuple[float, float] = (4.0, 3.0)

    def __repr__(self) -> str:
        """Return a compact repr showing only non-default fields."""
        return _compact_style_repr(self)


# ─── Theme System ───────────────────────────────────────────────────────────


@dataclass
class Theme:
    """Unified theme bundling all style defaults for a graph."""

    name: str = "default"
    node_styles: Dict[str, NodeStyle] = field(default_factory=dict)
    edge_styles: Dict[str, EdgeStyle] = field(default_factory=dict)
    cluster_style: ClusterStyle = field(default_factory=ClusterStyle)
    graph_style: GraphStyle = field(default_factory=GraphStyle)

    def get_node_style(self, node_type: str) -> NodeStyle:
        """Look up node style: type > "default" > NodeStyle()."""
        if node_type in self.node_styles:
            return self.node_styles[node_type]
        if "default" in self.node_styles:
            return self.node_styles["default"]
        return NodeStyle()

    def get_edge_style(self, edge_type: str) -> EdgeStyle:
        """Look up edge style: type > "default" > EdgeStyle()."""
        if edge_type in self.edge_styles:
            return self.edge_styles[edge_type]
        if "default" in self.edge_styles:
            return self.edge_styles["default"]
        return EdgeStyle()

    def copy(self) -> Theme:
        """Deep copy for user modification."""
        return copy.deepcopy(self)


# ─── Built-in Node Style Dicts (backwards compat) ──────────────────────────

# Default: all nodes use Sky blue (Wong palette default)
_sky_fill, _sky_stroke = make_node_colors(PALETTE["sky"])
_blue_fill, _blue_stroke = make_node_colors(PALETTE["blue"])
_green_fill, _green_stroke = make_node_colors(PALETTE["bluish_green"])
_vermillion_fill, _vermillion_stroke = make_node_colors(PALETTE["vermillion"])
_amber_fill, _amber_stroke = make_node_colors(PALETTE["amber"])
_purple_fill, _purple_stroke = make_node_colors(PALETTE["reddish_purple"])
_yellow_fill, _yellow_stroke = make_node_colors(PALETTE["yellow"])

# Legacy name: bare Dict[str, NodeStyle] for backwards compat
DEFAULT_NODE_STYLES: Dict[str, NodeStyle] = {
    "default": NodeStyle(base_color=PALETTE["sky"]),
    "input": NodeStyle(base_color=PALETTE["bluish_green"], padding=(11.0, 9.0)),
    "output": NodeStyle(base_color=PALETTE["vermillion"], padding=(11.0, 9.0)),
    "buffer": NodeStyle(base_color=MEDIUM_GRAY),
    "bool": NodeStyle(base_color=PALETTE["amber"]),
    "trainable_params": NodeStyle(base_color=PALETTE["blue"]),
    "frozen_params": NodeStyle(base_color=MEDIUM_GRAY),
    "mixed_params": NodeStyle(base_color=PALETTE["reddish_purple"]),
    "module": NodeStyle(base_color=PALETTE["blue"]),
}

GRAPHVIZ_MATCH_NODE_STYLES: Dict[str, NodeStyle] = {
    "default": NodeStyle(
        fill="#FFFFFF",
        stroke="#000000",
        shape="ellipse",
        font_family="serif",
        font_size=14.0,
        font_color="#000000",
        base_color="#000000",
    ),
    "input": NodeStyle(
        fill="#98FB98",
        stroke="#000000",
        shape="ellipse",
        font_family="serif",
        font_size=14.0,
        font_color="#000000",
        base_color="#98FB98",
    ),
    "output": NodeStyle(
        fill="#FF9999",
        stroke="#000000",
        shape="ellipse",
        font_family="serif",
        font_size=14.0,
        font_color="#000000",
        base_color="#FF9999",
    ),
}

# Backwards-compatible aliases
DEFAULT_THEME: Dict[str, NodeStyle] = DEFAULT_NODE_STYLES
GRAPHVIZ_MATCH_THEME: Dict[str, NodeStyle] = GRAPHVIZ_MATCH_NODE_STYLES
GRAPHVIZ_MATCH_DEFAULTS: Dict[str, Any] = {
    "stroke_width": 1.6,
    "padding": (7.0, 4.0),
    "font_size": 12.0,
    "arrow_length": 14.0,
    "arrow_width": 10.0,
    "arrow_scale": 16.0,
    "edge_width": 1.6,
    "edge_opacity": 1.0,
    "min_height": 22.0,
}

# ─── Full Theme Objects ────────────────────────────────────────────────────

DEFAULT_THEME_OBJ = Theme(
    name="default",
    node_styles=dict(DEFAULT_NODE_STYLES),
    edge_styles={
        "default": EdgeStyle(),
        "if": EdgeStyle(style="dashed", color=PALETTE["amber"]),
        "then": EdgeStyle(style="dashed", color=PALETTE["bluish_green"]),
        "buffer": EdgeStyle(style="dotted", opacity=0.5),
        "back": EdgeStyle(curvature=0.6),
    },
    cluster_style=ClusterStyle(),
    graph_style=GraphStyle(),
)

DARK_THEME = Theme(
    name="dark",
    node_styles={
        "default": NodeStyle(
            base_color=PALETTE["sky"],
            fill="#2A3A4A",
            stroke="#5A8AB0",
            font_color="#E0E0E0",
        ),
        "input": NodeStyle(
            base_color=PALETTE["bluish_green"],
            fill="#1A3A2A",
            stroke="#4A9A73",
            font_color="#E0E0E0",
        ),
        "output": NodeStyle(
            base_color=PALETTE["vermillion"],
            fill="#3A2A1A",
            stroke="#B05A3A",
            font_color="#E0E0E0",
        ),
        "buffer": NodeStyle(
            base_color=MEDIUM_GRAY,
            fill="#2A2A2A",
            stroke="#6A6A6A",
            font_color="#B0B0B0",
        ),
        "bool": NodeStyle(
            base_color=PALETTE["amber"],
            fill="#3A3A1A",
            stroke="#B09A3A",
            font_color="#E0E0E0",
        ),
        "trainable_params": NodeStyle(
            base_color=PALETTE["blue"],
            fill="#1A2A3A",
            stroke="#4A7AB0",
            font_color="#E0E0E0",
        ),
    },
    edge_styles={
        "default": EdgeStyle(color="#606060", opacity=0.6),
        "if": EdgeStyle(style="dashed", color="#B09A3A", opacity=0.6),
        "then": EdgeStyle(style="dashed", color="#4A9A73", opacity=0.6),
        "buffer": EdgeStyle(style="dotted", color="#505050", opacity=0.4),
        "back": EdgeStyle(color="#606060", opacity=0.6, curvature=0.6),
    },
    cluster_style=ClusterStyle(
        fill="#1E2228",
        stroke="#3A3E44",
        font_color="#A0A0A0",
        opacity=0.5,
    ),
    graph_style=GraphStyle(
        background_color="#1A1E24",
        title_font_color="#E0E0E0",
        edge_label_background="#1A1E24",
    ),
)

MINIMAL_THEME = Theme(
    name="minimal",
    node_styles={
        "default": NodeStyle(
            shape="rect",
            base_color="#000000",
            fill="#FFFFFF",
            stroke="#000000",
            stroke_width=0.5,
            corner_radius=0.0,
            font_color="#000000",
        ),
        "input": NodeStyle(
            shape="rect",
            base_color="#009E73",
            fill="#E8F5E9",
            stroke="#2E7D32",
            stroke_width=0.5,
            corner_radius=0.0,
            font_color="#000000",
        ),
        "output": NodeStyle(
            shape="rect",
            base_color="#D55E00",
            fill="#FBE9E7",
            stroke="#BF360C",
            stroke_width=0.5,
            corner_radius=0.0,
            font_color="#000000",
        ),
    },
    edge_styles={
        "default": EdgeStyle(color="#000000", width=0.5, opacity=0.5, curvature=0.0),
        "back": EdgeStyle(color="#000000", width=0.5, opacity=0.5, curvature=0.6),
    },
    cluster_style=ClusterStyle(
        fill="#FFFFFF",
        stroke="#CCCCCC",
        stroke_width=0.5,
        corner_radius=0.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(
        background_color="#FFFFFF",
    ),
)

TORCHLENS_THEME = Theme(
    name="torchlens",
    node_styles={
        "default": NodeStyle(
            base_color="#5E6C84",
            fill="#F4F1EB",
            stroke="#455468",
            font_size=8.0,
            padding=(8.0, 5.0),
            font_color="#1F2933",
        ),
        "input": NodeStyle(
            base_color="#2F9E75",
            fill="#DFF4EA",
            stroke="#23785A",
            font_size=8.4,
            padding=(10.0, 6.0),
            font_weight="bold",
        ),
        "output": NodeStyle(
            base_color="#D9684D",
            fill="#F8E2DB",
            stroke="#A94B33",
            font_size=8.4,
            padding=(10.0, 6.0),
            font_weight="bold",
        ),
        "buffer": NodeStyle(
            base_color="#7C8797",
            fill="#ECEEF2",
            stroke="#6E7784",
            font_size=7.8,
            font_color="#4B5563",
            padding=(9.0, 5.0),
        ),
        "bool": NodeStyle(
            base_color="#C48A1D",
            fill="#F8E8BF",
            stroke="#8E671B",
            font_size=8.1,
            font_weight="bold",
            padding=(9.0, 5.0),
        ),
        "trainable_params": NodeStyle(
            base_color="#3B82A6",
            fill="#DDEEF5",
            stroke="#2E637C",
            font_size=8.0,
            padding=(9.0, 5.0),
        ),
        "frozen_params": NodeStyle(
            base_color="#7B8594",
            fill="#E8EBEF",
            stroke="#687180",
            font_size=8.0,
            padding=(9.0, 5.0),
        ),
        "mixed_params": NodeStyle(
            base_color="#8A5D91",
            fill="#EFE3F1",
            stroke="#68476E",
            font_size=8.0,
            padding=(9.0, 5.0),
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#5F6B78", width=1.1, opacity=0.62, curvature=0.22, label_font_size=6.8
        ),
        "skip": EdgeStyle(
            color="#567DA4",
            width=1.25,
            opacity=0.52,
            curvature=0.34,
            style="solid",
            label_font_size=6.8,
        ),
        "recurrent": EdgeStyle(
            color="#A97728",
            width=1.2,
            opacity=0.65,
            curvature=0.55,
            style="dashed",
            label_font_size=6.8,
        ),
        "if": EdgeStyle(
            color="#B6841F", width=1.25, opacity=0.8, style="dashed", label_font_size=7.0
        ),
        "then": EdgeStyle(
            color="#2D8A68", width=1.25, opacity=0.8, style="dashed", label_font_size=7.0
        ),
        "buffer": EdgeStyle(
            color="#7B8594", width=1.0, opacity=0.45, style="dotted", label_font_size=6.7
        ),
        "back": EdgeStyle(
            color="#8C93D6",
            width=1.0,
            opacity=0.55,
            curvature=0.56,
            style="dotted",
            label_font_size=6.7,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#EDF2F6",
        stroke="#9FB3C3",
        stroke_width=0.85,
        corner_radius=10.0,
        padding=26.0,
        font_size=8.4,
        font_color="#3A4A5A",
        opacity=0.28,
        label_offset=(10.0, 24.0),
        depth_fill_step=0.04,
        depth_stroke_step=0.06,
    ),
    graph_style=GraphStyle(
        background_color="#FBFAF7",
        margin=20.0,
        title_font_size=7.6,
        title_font_weight="regular",
        title_font_color="#334155",
        edge_label_font_size=6.3,
        edge_label_background="#FBFAF7",
        edge_label_background_opacity=0.88,
        node_label_secondary_scale=0.8,
        max_figsize=(34.0, 44.0),
        min_figsize=(4.0, 3.0),
    ),
)

_GRAPHVIZ_STRICT_DEFAULT_NODE_STYLE = NodeStyle(
    shape="ellipse",
    fill="#FFFFFF",
    stroke="#000000",
    stroke_width=1.3,  # slightly above 1.0 to compensate for AA thinning
    font_family="Times New Roman",
    font_size=14.0,
    font_color="#000000",
    padding=(8.0, 4.0),  # Graphviz margin: 0.11in x 0.055in = ~8pt x 4pt
    corner_radius=0.0,
    opacity=1.0,
    base_color="#000000",
    min_width=54.0,  # Graphviz default: 0.75in = 54pt
    min_height=36.0,  # Graphviz default: 0.5in = 36pt
    overflow_policy="expand_node",
)

GRAPHVIZ_STRICT_THEME = Theme(
    name="graphviz_strict",
    node_styles={
        "default": copy.deepcopy(_GRAPHVIZ_STRICT_DEFAULT_NODE_STYLE),
        "input": copy.deepcopy(_GRAPHVIZ_STRICT_DEFAULT_NODE_STYLE),
        "output": copy.deepcopy(_GRAPHVIZ_STRICT_DEFAULT_NODE_STYLE),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#000000",
            width=1.0,  # Graphviz default penwidth
            arrow="normal",
            arrow_fill="filled",
            arrow_length=7.0,  # slim compact triangle
            arrow_width=4.5,  # narrow, proportional to 1.0pt edge
            arrow_scale=None,
            arrow_node_fraction=0.0,  # fixed size, not node-relative
            arrow_width_ratio=0.7,
            style="solid",
            opacity=1.0,
            label_font_size=14.0,
            label_font_color="#000000",
            label_background="#FFFFFF",
            label_font_family="Times New Roman",
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F0F0F0",  # very subtle gray (Graphviz defaults to no fill)
        stroke="#666666",  # medium gray like Graphviz's light border
        stroke_width=0.8,  # thin border matching Graphviz
        corner_radius=0.0,
        padding=16.0,  # generous padding like Graphviz
        label_position="top-left",
        font_size=12.0,
        font_weight="regular",
        font_color="#000000",
        font_family="Times New Roman",
        opacity=0.6,  # subtle fill, not heavy
    ),
    graph_style=GraphStyle(
        background_color="#FFFFFF",
        margin=18.0,
        title_font_size=14.0,
        title_font_color="#000000",
        edge_label_font_size=14.0,
        edge_label_background="#FFFFFF",
        edge_label_background_opacity=1.0,
    ),
)

GRAPHVIZ_THEME = Theme(
    name="graphviz",
    node_styles={
        "default": NodeStyle(
            shape="ellipse",
            fill="#FAFBFC",  # DEPARTURE: very subtle off-white instead of pure white
            stroke="#333333",  # DEPARTURE: dark gray instead of pure black (softer)
            stroke_width=1.2,  # DEPARTURE: slightly heavier for crispness at screen res
            font_family="",  # DEPARTURE: uses system sans-serif instead of Times-Roman
            font_size=12.0,  # DEPARTURE: 12pt instead of 14pt (less cluttered)
            font_color="#1A1A1A",  # DEPARTURE: near-black instead of pure black
            padding=(9.0, 5.0),  # DEPARTURE: slightly more generous padding
            corner_radius=0.0,
            opacity=1.0,
            base_color="#333333",
            min_width=44.0,  # DEPARTURE: ellipse width factor now provides extra width
            min_height=34.0,  # DEPARTURE: scaled down with the reduced minimum width
        ),
        "input": NodeStyle(
            shape="ellipse",
            fill="#F0FAF0",  # DEPARTURE: very subtle green tint for inputs
            stroke="#2D6A2D",  # DEPARTURE: dark green instead of black
            stroke_width=1.2,
            font_family="",  # DEPARTURE: uses system sans-serif instead of Times-Roman
            font_size=12.0,  # DEPARTURE: 12pt instead of 14pt (less cluttered)
            font_color="#1A1A1A",  # DEPARTURE: near-black instead of pure black
            padding=(9.0, 5.0),  # DEPARTURE: slightly more generous padding
            corner_radius=0.0,
            base_color="#2D6A2D",
        ),
        "output": NodeStyle(
            shape="ellipse",
            fill="#FFF0F0",  # DEPARTURE: very subtle red tint for outputs
            stroke="#8B2D2D",  # DEPARTURE: dark red instead of black
            stroke_width=1.2,
            font_family="",  # DEPARTURE: uses system sans-serif instead of Times-Roman
            font_size=12.0,  # DEPARTURE: 12pt instead of 14pt (less cluttered)
            font_color="#1A1A1A",  # DEPARTURE: near-black instead of pure black
            padding=(9.0, 5.0),  # DEPARTURE: slightly more generous padding
            corner_radius=0.0,
            base_color="#8B2D2D",
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#4A4A4A",  # DEPARTURE: dark gray instead of black
            width=1.1,  # DEPARTURE: slightly thicker
            arrow="normal",
            arrow_fill="filled",
            arrow_length=20.0,  # points — slightly smaller than strict
            arrow_width=14.0,  # points — stocky triangle
            arrow_scale=None,  # ignored; unified display scaling handles conversion
            arrow_node_fraction=0.35,  # keep Graphviz-like heads visually prominent
            arrow_width_ratio=0.7,
            arrow_color="#333333",  # DEPARTURE: darker arrowheads for Graphviz-like contrast
            style="solid",
            opacity=0.92,  # DEPARTURE: edges recede slightly behind nodes
            label_font_size=11.0,  # DEPARTURE: slightly smaller than node text
            label_font_color="#333333",
            label_background="#FAFBFC",
        ),
        "if": EdgeStyle(color="#B08A1F", width=1.1, style="dashed", opacity=0.92),
        "then": EdgeStyle(color="#2D8A68", width=1.1, style="dashed", opacity=0.92),
        "buffer": EdgeStyle(color="#7B8594", width=1.0, style="dotted", opacity=0.7),
    },
    cluster_style=ClusterStyle(
        fill="#F5F5F5",  # DEPARTURE: subtle fill instead of transparent
        stroke="#999999",  # DEPARTURE: medium gray instead of black
        stroke_width=1.0,
        corner_radius=3.0,  # DEPARTURE: very subtle rounding
        padding=12.0,
        label_position="top-left",
        font_size=12.0,  # DEPARTURE: matches node font size
        font_weight="bold",
        font_color="#333333",  # DEPARTURE: dark gray instead of black
        opacity=0.8,  # DEPARTURE: semi-transparent for layering
    ),
    graph_style=GraphStyle(
        background_color="#FAFAFA",  # DEPARTURE: warm white
        margin=18.0,
        title_font_size=12.0,
        title_font_color="#1A1A1A",
        edge_label_font_size=10.0,
        edge_label_background="#FAFAFA",
        edge_label_background_opacity=1.0,
    ),
)

MERMAID_THEME = Theme(
    name="mermaid",
    node_styles={
        "default": NodeStyle(
            shape="roundrect",
            fill="#ECECFF",
            stroke="#9370DB",
            stroke_width=2.0,
            font_family="Trebuchet MS",
            font_size=13.0,
            font_color="#333333",
            corner_radius=10.0,
            padding=(15.0, 10.0),
        ),
        "input": NodeStyle(
            shape="stadium",
            fill="#ECECFF",
            stroke="#9370DB",
            stroke_width=2.0,
            font_family="Trebuchet MS",
            font_size=13.0,
            font_color="#333333",
            corner_radius=10.0,
            padding=(15.0, 10.0),
        ),
        "output": NodeStyle(
            shape="stadium",
            fill="#ECECFF",
            stroke="#9370DB",
            stroke_width=2.0,
            font_family="Trebuchet MS",
            font_size=13.0,
            font_color="#333333",
            corner_radius=10.0,
            padding=(15.0, 10.0),
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#555555",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#555555",
            routing="bezier",
            label_font_size=11.0,
            label_font_color="#333333",
            label_background="#E8E8E8",
        ),
        "back": EdgeStyle(
            color="#555555",
            width=1.5,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#555555",
            routing="bezier",
        ),
    },
    cluster_style=ClusterStyle(
        fill="#FFFFDE",
        stroke="#AAAA33",
        stroke_width=1.5,
        corner_radius=10.0,
        font_size=13.0,
        font_color="#333333",
        font_weight="bold",
        padding=15.0,
        opacity=0.8,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

D3_THEME = Theme(
    name="d3",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#1f77b4",
            stroke="#FFFFFF",
            stroke_width=1.5,
            font_family="system-ui, -apple-system, sans-serif",
            font_size=10.0,
            font_color="#333333",
            padding=(6.0, 6.0),
            min_width=20.0,
            min_height=20.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#2ca02c",
            stroke="#FFFFFF",
            stroke_width=1.5,
            font_family="system-ui, sans-serif",
            font_size=10.0,
            font_color="#333333",
            padding=(6.0, 6.0),
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#d62728",
            stroke="#FFFFFF",
            stroke_width=1.5,
            font_family="system-ui, sans-serif",
            font_size=10.0,
            font_color="#333333",
            padding=(6.0, 6.0),
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#999999",
            width=1.5,
            opacity=0.6,
            style="solid",
            arrow="none",
            routing="straight",
        ),
        "back": EdgeStyle(
            color="#999999",
            width=1.5,
            opacity=0.4,
            style="dashed",
            arrow="none",
            routing="straight",
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F0F0F0",
        stroke="#CCCCCC",
        stroke_width=1.0,
        corner_radius=0.0,
        font_size=11.0,
        font_color="#666666",
        opacity=0.3,
        padding=10.0,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

CYTOSCAPE_THEME = Theme(
    name="cytoscape",
    node_styles={
        "default": NodeStyle(
            shape="ellipse",
            fill="#888888",
            stroke="#666666",
            stroke_width=0.5,
            font_family="Helvetica Neue",
            font_size=11.0,
            font_color="#FFFFFF",
            padding=(8.0, 8.0),
            min_width=30.0,
            min_height=30.0,
        ),
        "input": NodeStyle(
            shape="ellipse",
            fill="#5B8FF9",
            stroke="#4070CC",
            stroke_width=0.5,
            font_family="Helvetica Neue",
            font_size=11.0,
            font_color="#FFFFFF",
            padding=(8.0, 8.0),
        ),
        "output": NodeStyle(
            shape="ellipse",
            fill="#F6903D",
            stroke="#D07030",
            stroke_width=0.5,
            font_family="Helvetica Neue",
            font_size=11.0,
            font_color="#FFFFFF",
            padding=(8.0, 8.0),
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#AAAAAA",
            width=1.0,
            style="solid",
            arrow="none",
            routing="bezier",
            opacity=0.9,
        ),
        "back": EdgeStyle(
            color="#CC6666",
            width=1.0,
            style="dashed",
            arrow="none",
            routing="bezier",
            opacity=0.6,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#E8E8E8",
        stroke="#AAAAAA",
        stroke_width=1.0,
        corner_radius=5.0,
        font_size=11.0,
        font_color="#555555",
        opacity=0.5,
        padding=10.0,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

GEPHI_THEME = Theme(
    name="gephi",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#6A8EAE",
            stroke="#5A7E9E",
            stroke_width=0.5,
            font_family="Arial",
            font_size=9.0,
            font_color="#000000",
            padding=(4.0, 4.0),
            min_width=18.0,
            min_height=18.0,
            opacity=0.9,
            text_outline=True,
            text_outline_color="#FFFFFF",
            text_outline_width=2.5,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#4CAF50",
            stroke="#3C9F40",
            stroke_width=0.5,
            font_family="Arial",
            font_size=9.0,
            font_color="#000000",
            padding=(4.0, 4.0),
            opacity=0.9,
            text_outline=True,
            text_outline_color="#FFFFFF",
            text_outline_width=2.5,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#E57373",
            stroke="#D56363",
            stroke_width=0.5,
            font_family="Arial, sans-serif",
            font_size=9.0,
            font_color="#000000",
            padding=(4.0, 4.0),
            opacity=0.9,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#888888",
            width=0.8,
            style="solid",
            arrow="simple",
            arrow_fill="filled",
            arrow_length=5.0,
            arrow_width=3.5,
            opacity=0.25,
            routing="bezier",
            curvature=0.3,
        ),
        "back": EdgeStyle(
            color="#CC8888",
            width=0.8,
            style="solid",
            arrow="simple",
            arrow_fill="filled",
            arrow_length=5.0,
            arrow_width=3.5,
            opacity=0.2,
            routing="bezier",
            curvature=0.3,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F5F5F5",
        stroke="#CCCCCC",
        stroke_width=0.5,
        corner_radius=0.0,
        font_size=10.0,
        font_color="#666666",
        opacity=0.3,
        padding=8.0,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

OBSIDIAN_THEME = Theme(
    name="obsidian",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#8B7BDE",
            stroke="#8B7BDE",
            stroke_width=0.0,
            font_family="Inter",
            font_size=8.0,
            font_color="#DCDDDE",
            padding=(4.0, 4.0),
            min_width=12.0,
            min_height=12.0,
            opacity=0.85,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#08B94E",
            stroke="#08B94E",
            stroke_width=0.0,
            font_family="Inter",
            font_size=8.0,
            font_color="#DCDDDE",
            padding=(4.0, 4.0),
            opacity=0.85,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#E0AC00",
            stroke="#E0AC00",
            stroke_width=0.0,
            font_family="Inter",
            font_size=8.0,
            font_color="#DCDDDE",
            padding=(4.0, 4.0),
            opacity=0.85,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#5C5C5C",
            width=1.0,
            style="solid",
            arrow="none",
            routing="straight",
            opacity=0.4,
        ),
        "back": EdgeStyle(
            color="#5C5C5C",
            width=1.0,
            style="dashed",
            arrow="none",
            routing="straight",
            opacity=0.3,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#2D2D2D",
        stroke="#404040",
        stroke_width=0.5,
        corner_radius=4.0,
        font_size=9.0,
        font_color="#999999",
        opacity=0.4,
        padding=8.0,
    ),
    graph_style=GraphStyle(background_color="#1E1E1E"),
)

YED_THEME = Theme(
    name="yed",
    node_styles={
        "default": NodeStyle(
            shape="roundrect",
            fill="#E8E8E8",
            stroke="#000000",
            stroke_width=1.0,
            font_family="Arial",
            font_size=12.0,
            font_color="#000000",
            corner_radius=3.0,
            padding=(10.0, 6.0),
            shadow=True,
            shadow_color="#00000044",
            shadow_offset=(3.0, -3.0),
            shadow_blur=0.0,
        ),
        "input": NodeStyle(
            shape="roundrect",
            fill="#C7E5C0",
            stroke="#000000",
            stroke_width=1.0,
            font_family="Arial",
            font_size=12.0,
            font_color="#000000",
            corner_radius=3.0,
            padding=(10.0, 6.0),
            shadow=True,
            shadow_color="#00000044",
            shadow_offset=(3.0, -3.0),
        ),
        "output": NodeStyle(
            shape="roundrect",
            fill="#FFC9C9",
            stroke="#000000",
            stroke_width=1.0,
            font_family="Arial",
            font_size=12.0,
            font_color="#000000",
            corner_radius=3.0,
            padding=(10.0, 6.0),
            shadow=True,
            shadow_color="#00000033",
            shadow_offset=(2.0, -2.0),
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#AAAAAA",
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#AAAAAA",
            routing="ortho",
        ),
        "back": EdgeStyle(
            color="#AAAAAA",
            width=1.0,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#AAAAAA",
            routing="ortho",
        ),
    },
    cluster_style=ClusterStyle(
        fill="#EEEEEE",
        stroke="#000000",
        stroke_width=1.0,
        corner_radius=0.0,
        font_size=12.0,
        font_color="#000000",
        font_weight="bold",
        padding=12.0,
        opacity=0.8,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

DRAWIO_THEME = Theme(
    name="drawio",
    node_styles={
        "default": NodeStyle(
            shape="roundrect",
            fill="#DAE8FC",
            stroke="#6C8EBF",
            stroke_width=1.0,
            font_family="Arial",
            font_size=12.0,
            font_color="#333333",
            corner_radius=5.0,
            padding=(10.0, 6.0),
            shadow=True,
            shadow_color="#00000022",
            shadow_offset=(2.0, -3.0),
            shadow_blur=0.0,
        ),
        "input": NodeStyle(
            shape="roundrect",
            fill="#D5E8D4",
            stroke="#82B366",
            stroke_width=1.0,
            font_family="Arial",
            font_size=12.0,
            font_color="#333333",
            corner_radius=5.0,
            padding=(10.0, 6.0),
            shadow=True,
            shadow_color="#00000022",
            shadow_offset=(2.0, -3.0),
        ),
        "output": NodeStyle(
            shape="roundrect",
            fill="#F8CECC",
            stroke="#B85450",
            stroke_width=1.0,
            font_family="Arial",
            font_size=12.0,
            font_color="#333333",
            corner_radius=5.0,
            padding=(10.0, 6.0),
            shadow=True,
            shadow_color="#00000022",
            shadow_offset=(2.0, -3.0),
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#666666",
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#666666",
            routing="ortho",
            label_font_size=11.0,
            label_background="#FFFFFF",
        ),
        "back": EdgeStyle(
            color="#999999",
            width=1.0,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            routing="ortho",
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F5F5F5",
        stroke="#666666",
        stroke_width=1.0,
        stroke_dash="dashed",
        corner_radius=5.0,
        font_size=12.0,
        font_color="#333333",
        font_weight="bold",
        padding=12.0,
        opacity=0.7,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)


NEO4J_THEME = Theme(
    name="neo4j",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#57C7E3",  # Neo4j signature teal
            stroke="#23B3D7",
            stroke_width=2.0,
            font_family="Arial",
            font_size=10.0,
            font_color="#FFFFFF",
            padding=(6.0, 6.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#8DCC93",  # Neo4j palette green
            stroke="#6DBB73",
            stroke_width=2.0,
            font_family="Arial",
            font_size=10.0,
            font_color="#2A2C34",
            padding=(6.0, 6.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#F16667",  # Neo4j palette coral red
            stroke="#D14B4C",
            stroke_width=2.0,
            font_family="Arial",
            font_size=10.0,
            font_color="#FFFFFF",
            padding=(6.0, 6.0),
            min_width=28.0,
            min_height=28.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#A5ABB6",
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#A5ABB6",
            arrow_length=6.0,
            arrow_width=4.0,
            routing="straight",
            label_font_size=8.0,
            label_font_color="#333333",
            label_background="#F9FCFF",
        ),
        "back": EdgeStyle(
            color="#A5ABB6",
            width=1.0,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#A5ABB6",
            routing="straight",
            label_font_size=8.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F0F4F8",
        stroke="#A5ABB6",
        stroke_width=1.0,
        corner_radius=8.0,
        font_size=11.0,
        font_color="#2A2C34",
        font_weight="bold",
        padding=12.0,
        opacity=0.6,
    ),
    graph_style=GraphStyle(
        background_color="#F9FCFF",  # Neo4j pale blue-white
    ),
)


NETWORKX_THEME = Theme(
    name="networkx",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#1F78B4",  # ColorBrewer Paired blue -- the iconic NX color
            stroke="#1F78B4",  # border matches face
            stroke_width=1.0,
            font_family="DejaVu Sans",
            font_size=10.0,
            font_color="#000000",
            padding=(6.0, 6.0),
            min_width=22.0,
            min_height=22.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#33A02C",  # ColorBrewer Paired green
            stroke="#33A02C",
            stroke_width=1.0,
            font_family="DejaVu Sans",
            font_size=10.0,
            font_color="#000000",
            padding=(6.0, 6.0),
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#E31A1C",  # ColorBrewer Paired red
            stroke="#E31A1C",
            stroke_width=1.0,
            font_family="DejaVu Sans",
            font_size=10.0,
            font_color="#000000",
            padding=(6.0, 6.0),
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#000000",
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#000000",
            routing="straight",
        ),
        "back": EdgeStyle(
            color="#000000",
            width=1.0,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            routing="straight",
        ),
    },
    cluster_style=ClusterStyle(
        fill="#E0E0E0",
        stroke="#999999",
        stroke_width=1.0,
        corner_radius=0.0,
        font_size=10.0,
        font_color="#000000",
        padding=8.0,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

TIKZ_THEME = Theme(
    name="tikz",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#ABD7E6",  # tikz-network default vertex fill
            stroke="#000000",
            stroke_width=0.6,  # thin LaTeX line
            font_family="DejaVu Serif",  # closest to Computer Modern
            font_size=8.0,  # \scriptsize equivalent
            font_color="#000000",
            padding=(4.0, 4.0),
            min_width=20.0,
            min_height=20.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#C8E6C0",  # light green
            stroke="#000000",
            stroke_width=0.6,
            font_family="DejaVu Serif",
            font_size=8.0,
            font_color="#000000",
            padding=(4.0, 4.0),
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#F0C0C0",  # light red
            stroke="#000000",
            stroke_width=0.6,
            font_family="DejaVu Serif",
            font_size=8.0,
            font_color="#000000",
            padding=(4.0, 4.0),
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#404040",  # black!75
            width=1.2,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#404040",
            routing="straight",
            label_font_size=8.0,
            label_background="#FFFFFF",
        ),
        "back": EdgeStyle(
            color="#404040",
            width=1.2,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            routing="straight",
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F5F5F5",
        stroke="#000000",
        stroke_width=0.4,
        stroke_dash="dashed",
        corner_radius=0.0,
        font_size=9.0,
        font_color="#000000",
        padding=8.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

SIGMA_THEME = Theme(
    name="sigma",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#999999",  # sigma default gray
            stroke="#999999",
            stroke_width=0.0,  # no border in default circle program
            font_family="Arial",
            font_size=10.0,
            font_color="#000000",
            padding=(4.0, 4.0),
            min_width=16.0,
            min_height=16.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#5B8FF9",
            stroke="#5B8FF9",
            stroke_width=0.0,
            font_family="Arial",
            font_size=10.0,
            font_color="#000000",
            padding=(4.0, 4.0),
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#F6903D",
            stroke="#F6903D",
            stroke_width=0.0,
            font_family="Arial",
            font_size=10.0,
            font_color="#000000",
            padding=(4.0, 4.0),
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#CCCCCC",  # sigma default light gray
            width=1.0,
            style="solid",
            arrow="none",
            routing="straight",
            opacity=0.8,
        ),
        "back": EdgeStyle(
            color="#CCCCCC",
            width=1.0,
            style="dashed",
            arrow="none",
            routing="straight",
            opacity=0.5,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F0F0F0",
        stroke="#CCCCCC",
        stroke_width=0.5,
        corner_radius=0.0,
        font_size=10.0,
        font_color="#666666",
        padding=8.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

VISJS_THEME = Theme(
    name="visjs",
    node_styles={
        "default": NodeStyle(
            shape="ellipse",
            fill="#97C2FC",  # vis.js signature cornflower blue
            stroke="#2B7CE9",  # darker blue border
            stroke_width=1.0,
            font_family="Arial",
            font_size=11.0,
            font_color="#343434",
            padding=(10.0, 8.0),
            min_width=25.0,
            min_height=20.0,
        ),
        "input": NodeStyle(
            shape="ellipse",
            fill="#A8E6A1",  # light green
            stroke="#4CAF50",
            stroke_width=1.0,
            font_family="Arial",
            font_size=11.0,
            font_color="#343434",
            padding=(10.0, 8.0),
        ),
        "output": NodeStyle(
            shape="ellipse",
            fill="#FFAAAA",  # light red
            stroke="#E53935",
            stroke_width=1.0,
            font_family="Arial",
            font_size=11.0,
            font_color="#343434",
            padding=(10.0, 8.0),
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#848484",  # vis.js gray
            width=1.0,
            style="solid",
            arrow="none",  # vis.js arrows off by default
            routing="bezier",
            curvature=0.3,  # smooth dynamic curves
        ),
        "back": EdgeStyle(
            color="#848484",
            width=1.0,
            style="dashed",
            arrow="none",
            routing="bezier",
            curvature=0.3,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#EEF2FF",
        stroke="#2B7CE9",
        stroke_width=1.0,
        corner_radius=4.0,
        font_size=11.0,
        font_color="#343434",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

GRAPHISTRY_THEME = Theme(
    name="graphistry",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#6C8EBF",  # muted blue (structure-colored default)
            stroke="#6C8EBF",
            stroke_width=0.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#E0E0E0",
            padding=(3.0, 3.0),
            min_width=14.0,
            min_height=14.0,
            opacity=0.9,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#4CAF50",
            stroke="#4CAF50",
            stroke_width=0.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#E0E0E0",
            padding=(3.0, 3.0),
            opacity=0.9,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#FF7043",
            stroke="#FF7043",
            stroke_width=0.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#E0E0E0",
            padding=(3.0, 3.0),
            opacity=0.9,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#555555",
            width=0.8,
            style="solid",
            arrow="simple",
            arrow_fill="filled",
            arrow_color="#555555",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.2,
            opacity=0.3,  # signature low opacity for density handling
        ),
        "back": EdgeStyle(
            color="#555555",
            width=0.8,
            style="solid",
            arrow="simple",
            opacity=0.2,
            routing="bezier",
            curvature=0.2,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#2A2A2A",
        stroke="#444444",
        stroke_width=0.5,
        corner_radius=4.0,
        font_size=9.0,
        font_color="#AAAAAA",
        padding=8.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(
        background_color="#1E1E1E",  # signature dark background
    ),
)


IGRAPH_R_THEME = Theme(
    name="igraph_r",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#7EC0EE",  # R SkyBlue2
            stroke="#000000",
            stroke_width=1.0,
            font_family="DejaVu Serif",  # R serif default
            font_size=10.0,
            font_color="#000000",
            padding=(6.0, 6.0),
            min_width=24.0,
            min_height=24.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#56B4E9",  # R colorblind palette blue
            stroke="#000000",
            stroke_width=1.0,
            font_family="DejaVu Serif",
            font_size=10.0,
            font_color="#000000",
            padding=(6.0, 6.0),
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#D55E00",  # R colorblind palette vermillion
            stroke="#000000",
            stroke_width=1.0,
            font_family="DejaVu Serif",
            font_size=10.0,
            font_color="#000000",
            padding=(6.0, 6.0),
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#A9A9A9",  # darkgrey
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#A9A9A9",
            routing="straight",
        ),
        "back": EdgeStyle(
            color="#A9A9A9",
            width=1.0,
            style="dashed",
            arrow="normal",
            routing="straight",
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F0F0F0",
        stroke="#999999",
        stroke_width=1.0,
        corner_radius=0.0,
        font_size=10.0,
        font_color="#000000",
        padding=8.0,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

GRAPH_TOOL_THEME = Theme(
    name="graph_tool",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#A50F15",  # graph-tool dark crimson
            stroke="#999999",
            stroke_width=0.8,
            font_family="DejaVu Serif",
            font_size=9.0,
            font_color="#000000",
            padding=(3.0, 3.0),
            min_width=14.0,
            min_height=14.0,
            opacity=0.8,  # graph-tool signature 80% opacity
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#2171B5",  # blue variant
            stroke="#999999",
            stroke_width=0.8,
            font_family="DejaVu Serif",
            font_size=9.0,
            font_color="#000000",
            padding=(3.0, 3.0),
            opacity=0.8,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#238B45",  # green variant
            stroke="#999999",
            stroke_width=0.8,
            font_family="DejaVu Serif",
            font_size=9.0,
            font_color="#000000",
            padding=(3.0, 3.0),
            opacity=0.8,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#2D3436",  # near-black charcoal
            width=1.0,
            style="solid",
            arrow="none",  # graph-tool: no arrows by default
            routing="straight",
            opacity=0.8,
        ),
        "back": EdgeStyle(
            color="#2D3436",
            width=1.0,
            style="dashed",
            arrow="none",
            routing="straight",
            opacity=0.6,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F5F5F5",
        stroke="#CCCCCC",
        stroke_width=0.5,
        corner_radius=0.0,
        font_size=9.0,
        font_color="#333333",
        padding=6.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)


EXCALIDRAW_THEME = Theme(
    name="excalidraw",
    node_styles={
        "default": NodeStyle(
            shape="roundrect",
            fill="#A5D8FF",  # Open Color blue-2
            stroke="#1E1E1E",
            stroke_width=1.5,
            stroke_dash="solid",
            font_family="Comic Sans MS",  # closest to Virgil/Excalifont
            font_size=12.0,
            font_color="#1E1E1E",
            corner_radius=6.0,
            padding=(12.0, 8.0),
        ),
        "input": NodeStyle(
            shape="roundrect",
            fill="#B2F2BB",  # Open Color green-2
            stroke="#1E1E1E",
            stroke_width=1.5,
            font_family="Comic Sans MS",
            font_size=12.0,
            font_color="#1E1E1E",
            corner_radius=6.0,
            padding=(12.0, 8.0),
        ),
        "output": NodeStyle(
            shape="roundrect",
            fill="#FFC9C9",  # Open Color red-2
            stroke="#1E1E1E",
            stroke_width=1.5,
            font_family="Comic Sans MS",
            font_size=12.0,
            font_color="#1E1E1E",
            corner_radius=6.0,
            padding=(12.0, 8.0),
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#1E1E1E",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#1E1E1E",
            routing="straight",
        ),
        "back": EdgeStyle(
            color="#1E1E1E",
            width=1.5,
            style="dashed",
            arrow="normal",
            routing="straight",
        ),
    },
    cluster_style=ClusterStyle(
        fill="#FFEC99",  # Open Color yellow-2
        stroke="#1E1E1E",
        stroke_width=1.5,
        corner_radius=6.0,
        font_size=12.0,
        font_color="#1E1E1E",
        font_weight="bold",
        padding=12.0,
        opacity=0.6,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

GITHUB_THEME = Theme(
    name="github",
    node_styles={
        "default": NodeStyle(
            shape="roundrect",
            fill="#F6F8FA",  # Primer gray-0
            stroke="#D0D7DE",  # Primer gray-3
            stroke_width=1.0,
            font_family="Arial",
            font_size=11.0,
            font_color="#1F2328",  # Primer fg-default
            corner_radius=6.0,
            padding=(12.0, 8.0),
        ),
        "input": NodeStyle(
            shape="roundrect",
            fill="#DAFBE1",  # Primer green-1
            stroke="#1A7F37",  # Primer green-6
            stroke_width=1.0,
            font_family="Arial",
            font_size=11.0,
            font_color="#1F2328",
            corner_radius=6.0,
            padding=(12.0, 8.0),
        ),
        "output": NodeStyle(
            shape="roundrect",
            fill="#FFEBE9",  # Primer red-1
            stroke="#CF222E",  # Primer red-5
            stroke_width=1.0,
            font_family="Arial",
            font_size=11.0,
            font_color="#1F2328",
            corner_radius=6.0,
            padding=(12.0, 8.0),
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#656D76",  # Primer gray-5
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#656D76",
            routing="ortho",
        ),
        "back": EdgeStyle(
            color="#656D76",
            width=1.5,
            style="dashed",
            arrow="normal",
            routing="ortho",
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F6F8FA",
        stroke="#D0D7DE",
        stroke_width=1.0,
        corner_radius=6.0,
        font_size=11.0,
        font_color="#656D76",
        font_weight="bold",
        padding=12.0,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

LINEAR_THEME = Theme(
    name="linear",
    node_styles={
        "default": NodeStyle(
            shape="roundrect",
            fill="#1E1E20",  # dark surface
            stroke="#2A2A2E",  # barely visible border
            stroke_width=1.0,
            font_family="Inter",
            font_size=11.0,
            font_color="#EEEFF1",  # off-white
            corner_radius=6.0,
            padding=(12.0, 8.0),
        ),
        "input": NodeStyle(
            shape="roundrect",
            fill="#1E1E20",
            stroke="#5E6AD2",  # indigo accent border
            stroke_width=1.5,
            font_family="Inter",
            font_size=11.0,
            font_color="#EEEFF1",
            corner_radius=6.0,
            padding=(12.0, 8.0),
        ),
        "output": NodeStyle(
            shape="roundrect",
            fill="#1E1E20",
            stroke="#E5484D",  # red accent
            stroke_width=1.5,
            font_family="Inter",
            font_size=11.0,
            font_color="#EEEFF1",
            corner_radius=6.0,
            padding=(12.0, 8.0),
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#3A3A3E",
            width=1.0,
            style="solid",
            arrow="simple",
            arrow_fill="filled",
            arrow_color="#3A3A3E",
            routing="bezier",
            opacity=0.7,
        ),
        "back": EdgeStyle(
            color="#3A3A3E",
            width=1.0,
            style="dashed",
            arrow="simple",
            routing="bezier",
            opacity=0.5,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#161618",
        stroke="#2A2A2E",
        stroke_width=0.5,
        corner_radius=8.0,
        font_size=10.0,
        font_color="#7C7C84",
        font_weight="bold",
        padding=10.0,
        opacity=0.6,
    ),
    graph_style=GraphStyle(
        background_color="#0F0F10",  # "Woodsmoke" ultra-dark
    ),
)

N8N_THEME = Theme(
    name="n8n",
    node_styles={
        "default": NodeStyle(
            shape="roundrect",
            fill="#FFFFFF",
            stroke="#DBDFE7",
            stroke_width=1.0,
            font_family="Arial",
            font_size=11.0,
            font_color="#525356",
            corner_radius=8.0,
            padding=(14.0, 10.0),
            shadow=True,
            shadow_color="#00000011",
            shadow_offset=(2.0, -2.0),
            shadow_blur=0.0,
        ),
        "input": NodeStyle(
            shape="roundrect",
            fill="#FFFFFF",
            stroke="#10B981",  # green accent
            stroke_width=2.0,
            font_family="Arial",
            font_size=11.0,
            font_color="#525356",
            corner_radius=8.0,
            padding=(14.0, 10.0),
            shadow=True,
            shadow_color="#00000011",
            shadow_offset=(2.0, -2.0),
        ),
        "output": NodeStyle(
            shape="roundrect",
            fill="#FFFFFF",
            stroke="#EA4B71",  # n8n brand pink
            stroke_width=2.0,
            font_family="Arial",
            font_size=11.0,
            font_color="#525356",
            corner_radius=8.0,
            padding=(14.0, 10.0),
            shadow=True,
            shadow_color="#00000011",
            shadow_offset=(2.0, -2.0),
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#B0B0B0",
            width=1.5,
            style="solid",
            arrow="none",  # n8n uses connection dots, not arrowheads
            routing="bezier",
            curvature=0.4,
        ),
        "back": EdgeStyle(
            color="#B0B0B0",
            width=1.5,
            style="dashed",
            arrow="none",
            routing="bezier",
            curvature=0.4,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F5F5F5",
        stroke="#DBDFE7",
        stroke_width=1.0,
        corner_radius=8.0,
        font_size=11.0,
        font_color="#525356",
        font_weight="bold",
        padding=12.0,
    ),
    graph_style=GraphStyle(background_color="#F5F5F5"),
)

AIRFLOW_THEME = Theme(
    name="airflow",
    node_styles={
        "default": NodeStyle(
            shape="roundrect",
            fill="#FFFFFF",
            stroke="#017CEE",  # Airflow blue
            stroke_width=1.5,
            font_family="Arial",
            font_size=10.0,
            font_color="#333333",
            corner_radius=4.0,
            padding=(10.0, 6.0),
        ),
        "input": NodeStyle(
            shape="roundrect",
            fill="#E6F1F2",  # sensor cyan
            stroke="#4BB8A9",
            stroke_width=1.5,
            font_family="Arial",
            font_size=10.0,
            font_color="#333333",
            corner_radius=4.0,
            padding=(10.0, 6.0),
        ),
        "output": NodeStyle(
            shape="roundrect",
            fill="#FFE0B2",  # operator orange
            stroke="#F4A460",
            stroke_width=1.5,
            font_family="Arial",
            font_size=10.0,
            font_color="#333333",
            corner_radius=4.0,
            padding=(10.0, 6.0),
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#666666",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#666666",
            routing="straight",
        ),
        "back": EdgeStyle(
            color="#999999",
            width=1.5,
            style="dashed",
            arrow="normal",
            routing="straight",
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F0F4F8",
        stroke="#017CEE",
        stroke_width=1.0,
        corner_radius=4.0,
        font_size=10.0,
        font_color="#017CEE",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

DAGSTER_THEME = Theme(
    name="dagster",
    node_styles={
        "default": NodeStyle(
            shape="roundrect",
            fill="#252830",  # dark surface
            stroke="#333740",
            stroke_width=1.0,
            font_family="Inter",
            font_size=10.0,
            font_color="#E0E0E0",
            corner_radius=6.0,
            padding=(12.0, 8.0),
        ),
        "input": NodeStyle(
            shape="roundrect",
            fill="#252830",
            stroke="#7C3AED",  # Dagster purple accent
            stroke_width=1.5,
            font_family="Inter",
            font_size=10.0,
            font_color="#E0E0E0",
            corner_radius=6.0,
            padding=(12.0, 8.0),
        ),
        "output": NodeStyle(
            shape="roundrect",
            fill="#252830",
            stroke="#22C55E",  # success green
            stroke_width=1.5,
            font_family="Inter",
            font_size=10.0,
            font_color="#E0E0E0",
            corner_radius=6.0,
            padding=(12.0, 8.0),
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#444444",
            width=1.0,
            style="solid",
            arrow="simple",
            arrow_fill="filled",
            arrow_color="#444444",
            routing="bezier",
            curvature=0.3,
            opacity=0.6,
        ),
        "back": EdgeStyle(
            color="#444444",
            width=1.0,
            style="dashed",
            arrow="simple",
            routing="bezier",
            opacity=0.4,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#1C1F25",
        stroke="#333740",
        stroke_width=0.5,
        corner_radius=6.0,
        font_size=10.0,
        font_color="#888888",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(
        background_color="#1C1F25",
    ),
)


# ─── Theme Registry ──────────────────────────────────────────────────────

THEME_REGISTRY: Dict[str, Theme] = {
    "default": GRAPHVIZ_THEME,
    "dark": DARK_THEME,
    "minimal": MINIMAL_THEME,
    "torchlens": TORCHLENS_THEME,
    "graphviz": GRAPHVIZ_THEME,
    "graphviz_strict": GRAPHVIZ_STRICT_THEME,
    "mermaid": MERMAID_THEME,
    "d3": D3_THEME,
    "cytoscape": CYTOSCAPE_THEME,
    "gephi": GEPHI_THEME,
    "obsidian": OBSIDIAN_THEME,
    "yed": YED_THEME,
    "drawio": DRAWIO_THEME,
    "neo4j": NEO4J_THEME,
    "networkx": NETWORKX_THEME,
    "tikz": TIKZ_THEME,
    "sigma": SIGMA_THEME,
    "visjs": VISJS_THEME,
    "graphistry": GRAPHISTRY_THEME,
    "igraph_r": IGRAPH_R_THEME,
    "graph_tool": GRAPH_TOOL_THEME,
}

NEURON_THEME = Theme(
    name="neuron",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#F5EDE3",  # warm parchment soma
            stroke="#2C1810",  # dark sepia border
            stroke_width=2.0,
            font_family="DejaVu Serif",
            font_size=9.0,
            font_color="#2C1810",
            padding=(5.0, 5.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#D4E8C2",  # dendrite green (sensory)
            stroke="#2C1810",
            stroke_width=2.0,
            font_family="DejaVu Serif",
            font_size=9.0,
            font_color="#2C1810",
            padding=(5.0, 5.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#E8C2C2",  # axon terminal pink (motor)
            stroke="#2C1810",
            stroke_width=2.0,
            font_family="DejaVu Serif",
            font_size=9.0,
            font_color="#2C1810",
            padding=(5.0, 5.0),
            min_width=26.0,
            min_height=26.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#2C1810",  # dark sepia axon
            width=1.2,
            style="solid",
            arrow="dot",  # synaptic terminal bulb
            arrow_fill="filled",
            arrow_color="#2C1810",
            arrow_length=5.0,
            arrow_width=5.0,
            routing="bezier",
            curvature=0.3,
        ),
        "back": EdgeStyle(
            color="#8B7355",  # lighter sepia for inhibitory
            width=1.0,
            style="dashed",
            arrow="dot",
            arrow_fill="filled",
            arrow_color="#8B7355",
            arrow_length=4.0,
            arrow_width=4.0,
            routing="bezier",
            curvature=0.4,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#FAF6F0",  # light parchment
        stroke="#8B7355",
        stroke_width=0.5,
        stroke_dash="dashed",
        corner_radius=0.0,
        font_size=10.0,
        font_color="#8B7355",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(
        background_color="#FFFDF7",  # aged paper
    ),
)

BLUEPRINT_THEME = Theme(
    name="blueprint",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#1B3A5C",  # slightly lighter than bg
            stroke="#FFFFFF",
            stroke_width=1.5,
            font_family="DejaVu Sans",
            font_size=9.0,
            font_color="#FFFFFF",
            padding=(5.0, 5.0),
            min_width=22.0,
            min_height=22.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#1B3A5C",
            stroke="#88CCFF",  # highlight blue
            stroke_width=2.0,
            font_family="DejaVu Sans",
            font_size=9.0,
            font_color="#88CCFF",
            padding=(5.0, 5.0),
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#1B3A5C",
            stroke="#FFD700",  # gold highlight
            stroke_width=2.0,
            font_family="DejaVu Sans",
            font_size=9.0,
            font_color="#FFD700",
            padding=(5.0, 5.0),
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#FFFFFF",
            width=0.8,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#FFFFFF",
            routing="straight",
            opacity=0.7,
        ),
        "back": EdgeStyle(
            color="#FFFFFF",
            width=0.8,
            style="dashed",
            arrow="normal",
            routing="straight",
            opacity=0.4,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#162D4A",
        stroke="#FFFFFF",
        stroke_width=0.5,
        stroke_dash="dashed",
        corner_radius=0.0,
        font_size=9.0,
        font_color="#88CCFF",
        padding=8.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(
        background_color="#0F2744",  # Prussian blue
    ),
)

CHALKBOARD_THEME = Theme(
    name="chalkboard",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#2B4034",  # slightly lighter than board
            stroke="#E8E4D4",  # chalk white
            stroke_width=1.5,
            font_family="DejaVu Serif",
            font_size=10.0,
            font_color="#E8E4D4",
            padding=(5.0, 5.0),
            min_width=24.0,
            min_height=24.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#2B4034",
            stroke="#FFD59A",  # yellow chalk
            stroke_width=1.5,
            font_family="DejaVu Serif",
            font_size=10.0,
            font_color="#FFD59A",
            padding=(5.0, 5.0),
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#2B4034",
            stroke="#FF9A9A",  # pink chalk
            stroke_width=1.5,
            font_family="DejaVu Serif",
            font_size=10.0,
            font_color="#FF9A9A",
            padding=(5.0, 5.0),
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#E8E4D4",  # chalk white
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#E8E4D4",
            routing="straight",
            opacity=0.8,
        ),
        "back": EdgeStyle(
            color="#E8E4D4",
            width=1.0,
            style="dashed",
            arrow="normal",
            routing="straight",
            opacity=0.5,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#263A2E",
        stroke="#E8E4D4",
        stroke_width=0.5,
        stroke_dash="dashed",
        corner_radius=0.0,
        font_size=10.0,
        font_color="#E8E4D4",
        padding=8.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(
        background_color="#1E3228",  # dark green slate
    ),
)

SUBWAY_THEME = Theme(
    name="subway",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#FFFFFF",
            stroke="#333333",
            stroke_width=2.5,
            font_family="Arial",
            font_size=9.0,
            font_color="#333333",
            padding=(4.0, 4.0),
            min_width=18.0,
            min_height=18.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#FFFFFF",
            stroke="#E32636",  # Central line red
            stroke_width=3.0,
            font_family="Arial",
            font_size=9.0,
            font_color="#333333",
            padding=(4.0, 4.0),
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#FFFFFF",
            stroke="#003688",  # Piccadilly line blue
            stroke_width=3.0,
            font_family="Arial",
            font_size=9.0,
            font_color="#333333",
            padding=(4.0, 4.0),
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#0098D4",  # Victoria line cyan
            width=4.0,  # thick transit lines
            style="solid",
            arrow="none",  # subway lines have no arrows
            routing="ortho",  # Beck's orthogonal routing
            line_cap="round",
        ),
        "back": EdgeStyle(
            color="#9B0056",  # Metropolitan line magenta
            width=4.0,
            style="solid",
            arrow="none",
            routing="ortho",
            line_cap="round",
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F5F5F5",
        stroke="#CCCCCC",
        stroke_width=1.0,
        corner_radius=0.0,
        font_size=10.0,
        font_color="#333333",
        font_weight="bold",
        padding=10.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

VINTAGE_TEXTBOOK_THEME = Theme(
    name="vintage_textbook",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#FAF3E6",  # cream paper
            stroke="#333333",
            stroke_width=1.0,
            font_family="DejaVu Serif",
            font_size=11.0,
            font_color="#333333",
            font_style="italic",
            padding=(5.0, 5.0),
            min_width=22.0,
            min_height=22.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#FAF3E6",
            stroke="#333333",
            stroke_width=1.5,
            font_family="DejaVu Serif",
            font_size=11.0,
            font_color="#333333",
            font_style="italic",
            padding=(5.0, 5.0),
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#333333",  # filled black (common in textbooks)
            stroke="#333333",
            stroke_width=1.0,
            font_family="DejaVu Serif",
            font_size=11.0,
            font_color="#FAF3E6",
            font_style="italic",
            padding=(5.0, 5.0),
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#333333",
            width=0.8,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#333333",
            routing="straight",
        ),
        "back": EdgeStyle(
            color="#333333",
            width=0.8,
            style="dashed",
            arrow="normal",
            routing="straight",
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F5ECD8",
        stroke="#333333",
        stroke_width=0.5,
        stroke_dash="dashed",
        corner_radius=0.0,
        font_size=10.0,
        font_color="#333333",
        font_weight="bold",
        padding=8.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(
        background_color="#FAF3E6",  # cream textbook paper
    ),
)

FEYNMAN_THEME = Theme(
    name="feynman",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#FFFFFF",
            stroke="#000000",
            stroke_width=0.0,  # vertices are just dots in Feynman diagrams
            font_family="DejaVu Serif",
            font_size=10.0,
            font_color="#000000",
            font_style="italic",
            padding=(2.0, 2.0),
            min_width=8.0,  # small vertex dots
            min_height=8.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#000000",  # filled vertex
            stroke="#000000",
            stroke_width=0.0,
            font_family="DejaVu Serif",
            font_size=10.0,
            font_color="#000000",
            font_style="italic",
            padding=(2.0, 2.0),
            min_width=8.0,
            min_height=8.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#000000",
            stroke="#000000",
            stroke_width=0.0,
            font_family="DejaVu Serif",
            font_size=10.0,
            font_color="#000000",
            font_style="italic",
            padding=(2.0, 2.0),
            min_width=8.0,
            min_height=8.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#000000",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#000000",
            routing="straight",
        ),
        "back": EdgeStyle(
            color="#000000",
            width=1.5,
            style="dashed",  # dashed = virtual particle
            arrow="normal",
            routing="bezier",
            curvature=0.5,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F8F8F8",
        stroke="#CCCCCC",
        stroke_width=0.5,
        stroke_dash="dashed",
        corner_radius=0.0,
        font_size=10.0,
        font_color="#666666",
        padding=8.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

BAUHAUS_THEME = Theme(
    name="bauhaus",
    node_styles={
        "default": NodeStyle(
            shape="rect",
            fill="#FFFFFF",
            stroke="#000000",
            stroke_width=2.5,
            font_family="Arial",
            font_size=11.0,
            font_color="#000000",
            padding=(10.0, 8.0),
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#D40000",  # Bauhaus red
            stroke="#000000",
            stroke_width=2.5,
            font_family="Arial",
            font_size=11.0,
            font_color="#FFFFFF",
            padding=(8.0, 8.0),
        ),
        "output": NodeStyle(
            shape="rect",
            fill="#003DA5",  # Bauhaus blue
            stroke="#000000",
            stroke_width=2.5,
            font_family="Arial",
            font_size=11.0,
            font_color="#FFFFFF",
            padding=(10.0, 8.0),
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#000000",
            width=2.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#000000",
            routing="straight",
        ),
        "back": EdgeStyle(
            color="#000000",
            width=2.0,
            style="dashed",
            arrow="normal",
            routing="straight",
        ),
    },
    cluster_style=ClusterStyle(
        fill="#FFD700",  # Bauhaus yellow
        stroke="#000000",
        stroke_width=2.0,
        corner_radius=0.0,
        font_size=12.0,
        font_color="#000000",
        font_weight="bold",
        padding=10.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#F5F0E8"),
)

ART_DECO_THEME = Theme(
    name="art_deco",
    node_styles={
        "default": NodeStyle(
            shape="diamond",
            fill="#1A1A2E",  # deep navy
            stroke="#C9A94E",  # gold
            stroke_width=2.0,
            font_family="DejaVu Serif",
            font_size=10.0,
            font_color="#C9A94E",
            padding=(10.0, 10.0),
            min_width=30.0,
            min_height=30.0,
        ),
        "input": NodeStyle(
            shape="hexagon",
            fill="#1A1A2E",
            stroke="#C9A94E",
            stroke_width=2.0,
            font_family="DejaVu Serif",
            font_size=10.0,
            font_color="#C9A94E",
            padding=(10.0, 10.0),
        ),
        "output": NodeStyle(
            shape="octagon",
            fill="#1A1A2E",
            stroke="#C9A94E",
            stroke_width=2.0,
            font_family="DejaVu Serif",
            font_size=10.0,
            font_color="#C9A94E",
            padding=(10.0, 10.0),
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#C9A94E",
            width=1.5,
            style="solid",
            arrow="diamond",
            arrow_fill="filled",
            arrow_color="#C9A94E",
            routing="straight",
            opacity=0.8,
        ),
        "back": EdgeStyle(
            color="#C9A94E",
            width=1.0,
            style="dashed",
            arrow="diamond",
            routing="straight",
            opacity=0.5,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#16213E",
        stroke="#C9A94E",
        stroke_width=1.5,
        corner_radius=0.0,
        font_size=11.0,
        font_color="#C9A94E",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#0F0F23"),
)

NEON_THEME = Theme(
    name="neon",
    node_styles={
        "default": NodeStyle(
            shape="roundrect",
            fill="#0D0D0D",
            stroke="#00FFFF",  # cyan neon
            stroke_width=2.0,
            font_family="Arial",
            font_size=10.0,
            font_color="#00FFFF",
            corner_radius=4.0,
            padding=(10.0, 6.0),
        ),
        "input": NodeStyle(
            shape="roundrect",
            fill="#0D0D0D",
            stroke="#FF00FF",  # hot pink
            stroke_width=2.0,
            font_family="Arial",
            font_size=10.0,
            font_color="#FF00FF",
            corner_radius=4.0,
            padding=(10.0, 6.0),
        ),
        "output": NodeStyle(
            shape="roundrect",
            fill="#0D0D0D",
            stroke="#39FF14",  # neon green
            stroke_width=2.0,
            font_family="Arial",
            font_size=10.0,
            font_color="#39FF14",
            corner_radius=4.0,
            padding=(10.0, 6.0),
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#00FFFF",
            width=1.5,
            style="solid",
            arrow="simple",
            arrow_fill="filled",
            arrow_color="#00FFFF",
            routing="bezier",
            opacity=0.7,
        ),
        "back": EdgeStyle(
            color="#FF00FF",
            width=1.5,
            style="dashed",
            arrow="simple",
            routing="bezier",
            opacity=0.5,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#111111",
        stroke="#00FFFF",
        stroke_width=1.0,
        corner_radius=4.0,
        font_size=10.0,
        font_color="#00FFFF",
        padding=10.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#0A0A0A"),
)

TERMINAL_THEME = Theme(
    name="terminal",
    node_styles={
        "default": NodeStyle(
            shape="rect",
            fill="#0C0C0C",
            stroke="#33FF33",  # phosphor green
            stroke_width=1.0,
            font_family="DejaVu Sans Mono",
            font_size=9.0,
            font_color="#33FF33",
            padding=(8.0, 4.0),
        ),
        "input": NodeStyle(
            shape="rect",
            fill="#0C0C0C",
            stroke="#33FF33",
            stroke_width=1.5,
            font_family="DejaVu Sans Mono",
            font_size=9.0,
            font_color="#33FF33",
            padding=(8.0, 4.0),
        ),
        "output": NodeStyle(
            shape="rect",
            fill="#0C0C0C",
            stroke="#FFAA00",  # amber terminal
            stroke_width=1.5,
            font_family="DejaVu Sans Mono",
            font_size=9.0,
            font_color="#FFAA00",
            padding=(8.0, 4.0),
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#33FF33",
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#33FF33",
            routing="straight",
            opacity=0.6,
        ),
        "back": EdgeStyle(
            color="#33FF33",
            width=1.0,
            style="dashed",
            arrow="normal",
            routing="straight",
            opacity=0.3,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#0A0A0A",
        stroke="#33FF33",
        stroke_width=0.5,
        stroke_dash="dashed",
        corner_radius=0.0,
        font_size=9.0,
        font_color="#33FF33",
        padding=6.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#0C0C0C"),
)

NAPKIN_THEME = Theme(
    name="napkin",
    node_styles={
        "default": NodeStyle(
            shape="roundrect",
            fill="#FFFFFF",
            stroke="#444444",
            stroke_width=1.0,
            font_family="Comic Sans MS",
            font_size=11.0,
            font_color="#333333",
            corner_radius=8.0,
            padding=(10.0, 6.0),
        ),
        "input": NodeStyle(
            shape="roundrect",
            fill="#FFFFFF",
            stroke="#2266CC",
            stroke_width=1.5,
            font_family="Comic Sans MS",
            font_size=11.0,
            font_color="#2266CC",
            corner_radius=8.0,
            padding=(10.0, 6.0),
        ),
        "output": NodeStyle(
            shape="roundrect",
            fill="#FFFFFF",
            stroke="#CC3333",
            stroke_width=1.5,
            font_family="Comic Sans MS",
            font_size=11.0,
            font_color="#CC3333",
            corner_radius=8.0,
            padding=(10.0, 6.0),
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#444444",
            width=1.0,
            style="solid",
            arrow="simple",
            arrow_fill="filled",
            arrow_color="#444444",
            routing="bezier",
            curvature=0.5,
        ),
        "back": EdgeStyle(
            color="#444444",
            width=1.0,
            style="dashed",
            arrow="simple",
            routing="bezier",
            curvature=0.5,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F8F8F8",
        stroke="#999999",
        stroke_width=0.5,
        stroke_dash="dashed",
        corner_radius=8.0,
        font_size=11.0,
        font_color="#666666",
        padding=10.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

MOLECULAR_THEME = Theme(
    name="molecular",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#909090",  # CPK carbon gray
            stroke="#666666",
            stroke_width=0.5,
            font_family="Arial",
            font_size=10.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#3050F8",  # CPK nitrogen blue
            stroke="#2040D0",
            stroke_width=0.5,
            font_family="Arial",
            font_size=10.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#FF0D0D",  # CPK oxygen red
            stroke="#CC0000",
            stroke_width=0.5,
            font_family="Arial",
            font_size=10.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=28.0,
            min_height=28.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#555555",
            width=4.0,  # thick bond sticks
            style="solid",
            arrow="none",  # bonds have no direction
            routing="straight",
            line_cap="round",
        ),
        "back": EdgeStyle(
            color="#555555",
            width=2.0,
            style="dashed",  # dashed = weak bond
            arrow="none",
            routing="straight",
            line_cap="round",
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F0F0F0",
        stroke="#CCCCCC",
        stroke_width=0.5,
        corner_radius=8.0,
        font_size=9.0,
        font_color="#666666",
        padding=8.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

CIRCUIT_THEME = Theme(
    name="circuit",
    node_styles={
        "default": NodeStyle(
            shape="rect",
            fill="#1A5C1A",  # PCB green
            stroke="#C87533",  # copper
            stroke_width=1.5,
            font_family="DejaVu Sans Mono",
            font_size=8.0,
            font_color="#FFFFFF",
            padding=(8.0, 4.0),
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#C87533",  # copper pad
            stroke="#A06020",
            stroke_width=1.5,
            font_family="DejaVu Sans Mono",
            font_size=8.0,
            font_color="#FFFFFF",
            padding=(4.0, 4.0),
            min_width=18.0,
            min_height=18.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#C87533",
            stroke="#A06020",
            stroke_width=1.5,
            font_family="DejaVu Sans Mono",
            font_size=8.0,
            font_color="#FFFFFF",
            padding=(4.0, 4.0),
            min_width=18.0,
            min_height=18.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#C87533",  # copper trace
            width=2.5,
            style="solid",
            arrow="none",
            routing="ortho",  # PCB traces are orthogonal
            line_cap="projecting",
        ),
        "back": EdgeStyle(
            color="#8B5A2B",
            width=2.0,
            style="solid",
            arrow="none",
            routing="ortho",
            line_cap="projecting",
        ),
    },
    cluster_style=ClusterStyle(
        fill="#164016",
        stroke="#C87533",
        stroke_width=1.0,
        corner_radius=0.0,
        font_size=8.0,
        font_color="#C87533",
        padding=8.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#1A5C1A"),
)

CONSTELLATION_THEME = Theme(
    name="constellation",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#FFFFFF",
            stroke="#FFFFFF",
            stroke_width=0.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#8899BB",
            padding=(2.0, 2.0),
            min_width=6.0,  # tiny star dots
            min_height=6.0,
            opacity=0.9,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#FFD700",  # bright star gold
            stroke="#FFD700",
            stroke_width=0.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFD700",
            padding=(2.0, 2.0),
            min_width=8.0,
            min_height=8.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#FF6B6B",  # red giant
            stroke="#FF6B6B",
            stroke_width=0.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#FF6B6B",
            padding=(2.0, 2.0),
            min_width=8.0,
            min_height=8.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#445577",
            width=0.5,
            style="solid",
            arrow="none",
            routing="straight",
            opacity=0.4,
        ),
        "back": EdgeStyle(
            color="#445577",
            width=0.5,
            style="dashed",
            arrow="none",
            routing="straight",
            opacity=0.2,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#0A0E1A",
        stroke="#223355",
        stroke_width=0.3,
        stroke_dash="dashed",
        corner_radius=0.0,
        font_size=8.0,
        font_color="#556688",
        padding=6.0,
        opacity=0.2,
    ),
    graph_style=GraphStyle(background_color="#070B14"),  # deep space
)

GENEALOGY_THEME = Theme(
    name="genealogy",
    node_styles={
        "default": NodeStyle(
            shape="roundrect",
            fill="#FFF8F0",  # warm cream
            stroke="#8B6914",  # antique gold
            stroke_width=1.5,
            font_family="DejaVu Serif",
            font_size=10.0,
            font_color="#4A3000",
            corner_radius=4.0,
            padding=(12.0, 8.0),
            shadow=True,
            shadow_color="#00000015",
            shadow_offset=(2.0, -2.0),
        ),
        "input": NodeStyle(
            shape="roundrect",
            fill="#E8F0E8",  # pale green (paternal)
            stroke="#5C7A3A",
            stroke_width=1.5,
            font_family="DejaVu Serif",
            font_size=10.0,
            font_color="#3A5020",
            corner_radius=4.0,
            padding=(12.0, 8.0),
            shadow=True,
            shadow_color="#00000015",
            shadow_offset=(2.0, -2.0),
        ),
        "output": NodeStyle(
            shape="roundrect",
            fill="#F0E8F0",  # pale lavender (maternal)
            stroke="#7A5C7A",
            stroke_width=1.5,
            font_family="DejaVu Serif",
            font_size=10.0,
            font_color="#503A50",
            corner_radius=4.0,
            padding=(12.0, 8.0),
            shadow=True,
            shadow_color="#00000015",
            shadow_offset=(2.0, -2.0),
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#8B7355",
            width=1.0,
            style="solid",
            arrow="none",  # family trees typically no arrows
            routing="ortho",
        ),
        "back": EdgeStyle(
            color="#8B7355",
            width=1.0,
            style="dashed",
            arrow="none",
            routing="ortho",
        ),
    },
    cluster_style=ClusterStyle(
        fill="#FFF5E6",
        stroke="#C4A66A",
        stroke_width=1.0,
        corner_radius=4.0,
        font_size=10.0,
        font_color="#8B6914",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#FFFDF5"),
)

DARK_ACADEMIA_THEME = Theme(
    name="dark_academia",
    node_styles={
        "default": NodeStyle(
            shape="roundrect",
            fill="#2C1810",  # deep mahogany
            stroke="#8B6914",  # antique gold trim
            stroke_width=1.0,
            font_family="DejaVu Serif",
            font_size=10.0,
            font_color="#D4C5A9",  # parchment text
            corner_radius=3.0,
            padding=(10.0, 6.0),
        ),
        "input": NodeStyle(
            shape="roundrect",
            fill="#3D1E10",  # burgundy leather
            stroke="#8B6914",
            stroke_width=1.0,
            font_family="DejaVu Serif",
            font_size=10.0,
            font_color="#D4C5A9",
            corner_radius=3.0,
            padding=(10.0, 6.0),
        ),
        "output": NodeStyle(
            shape="roundrect",
            fill="#1A2A1A",  # dark green cloth
            stroke="#8B6914",
            stroke_width=1.0,
            font_family="DejaVu Serif",
            font_size=10.0,
            font_color="#D4C5A9",
            corner_radius=3.0,
            padding=(10.0, 6.0),
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#8B7355",
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#8B7355",
            routing="bezier",
            opacity=0.6,
        ),
        "back": EdgeStyle(
            color="#8B7355",
            width=1.0,
            style="dashed",
            arrow="normal",
            routing="bezier",
            opacity=0.4,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#1A1008",
        stroke="#6B5530",
        stroke_width=0.5,
        corner_radius=3.0,
        font_size=10.0,
        font_color="#A0906A",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#1A1008"),
)

PASTEL_THEME = Theme(
    name="pastel",
    node_styles={
        "default": NodeStyle(
            shape="roundrect",
            fill="#E8D5F5",  # soft lavender
            stroke="#C4A6E0",
            stroke_width=1.5,
            font_family="Arial",
            font_size=11.0,
            font_color="#6B4D8A",
            corner_radius=12.0,  # very rounded
            padding=(14.0, 10.0),
        ),
        "input": NodeStyle(
            shape="roundrect",
            fill="#D5F5E3",  # soft mint
            stroke="#A6E0C0",
            stroke_width=1.5,
            font_family="Arial",
            font_size=11.0,
            font_color="#4D8A6B",
            corner_radius=12.0,
            padding=(14.0, 10.0),
        ),
        "output": NodeStyle(
            shape="roundrect",
            fill="#F5D5D5",  # soft rose
            stroke="#E0A6A6",
            stroke_width=1.5,
            font_family="Arial",
            font_size=11.0,
            font_color="#8A4D4D",
            corner_radius=12.0,
            padding=(14.0, 10.0),
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#C0B0D0",
            width=1.5,
            style="solid",
            arrow="simple",
            arrow_fill="filled",
            arrow_color="#C0B0D0",
            routing="bezier",
            curvature=0.4,
        ),
        "back": EdgeStyle(
            color="#C0B0D0",
            width=1.5,
            style="dashed",
            arrow="simple",
            routing="bezier",
            opacity=0.6,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F5F0FA",
        stroke="#D0C0E0",
        stroke_width=1.0,
        corner_radius=12.0,
        font_size=11.0,
        font_color="#8A7AA0",
        font_weight="bold",
        padding=12.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#FDFBFF"),
)

THEME_REGISTRY["bauhaus"] = BAUHAUS_THEME
THEME_REGISTRY["art_deco"] = ART_DECO_THEME
THEME_REGISTRY["neon"] = NEON_THEME
THEME_REGISTRY["terminal"] = TERMINAL_THEME
THEME_REGISTRY["napkin"] = NAPKIN_THEME
THEME_REGISTRY["molecular"] = MOLECULAR_THEME
THEME_REGISTRY["circuit"] = CIRCUIT_THEME
THEME_REGISTRY["constellation"] = CONSTELLATION_THEME
THEME_REGISTRY["genealogy"] = GENEALOGY_THEME
THEME_REGISTRY["dark_academia"] = DARK_ACADEMIA_THEME
THEME_REGISTRY["pastel"] = PASTEL_THEME
THEME_REGISTRY["blueprint"] = BLUEPRINT_THEME
THEME_REGISTRY["chalkboard"] = CHALKBOARD_THEME
THEME_REGISTRY["subway"] = SUBWAY_THEME
THEME_REGISTRY["vintage_textbook"] = VINTAGE_TEXTBOOK_THEME
THEME_REGISTRY["feynman"] = FEYNMAN_THEME
THEME_REGISTRY["neuron"] = NEURON_THEME
THEME_REGISTRY["excalidraw"] = EXCALIDRAW_THEME
THEME_REGISTRY["github"] = GITHUB_THEME
THEME_REGISTRY["linear"] = LINEAR_THEME
THEME_REGISTRY["n8n"] = N8N_THEME
THEME_REGISTRY["airflow"] = AIRFLOW_THEME
THEME_REGISTRY["dagster"] = DAGSTER_THEME

# ── Neuroscience & biology themes ────────────────────────────────────────

# Van Essen -- inspired by David Van Essen's cortical wiring diagrams
# (flat colored region boxes, hierarchical connectivity, warm earth palette)
VAN_ESSEN_THEME = Theme(
    name="van_essen",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#E8D5B7",  # tan cortical region
            stroke="#5C4033",  # dark brown border
            stroke_width=1.8,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#3B2716",
            font_weight="bold",
            padding=(6.0, 3.0),
            min_width=36.0,
            min_height=18.0,
            corner_radius=2.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#A8C6E0",  # cool blue (primary sensory)
            stroke="#3B5E80",
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#1E3448",
            font_weight="bold",
            padding=(6.0, 3.0),
            min_width=36.0,
            min_height=18.0,
            corner_radius=2.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#D4A0A0",  # muted red (motor cortex)
            stroke="#804040",
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#4A2020",
            font_weight="bold",
            padding=(6.0, 3.0),
            min_width=36.0,
            min_height=18.0,
            corner_radius=2.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#6B5344",  # warm brown fiber tract
            width=1.2,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#6B5344",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.25,
        ),
        "back": EdgeStyle(
            color="#9B8B7B",  # lighter feedback connection
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#9B8B7B",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.35,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F5ECD8",  # light parchment lobe boundary
        stroke="#8B7B68",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=4.0,
        font_size=9.0,
        font_color="#5C4033",
        font_weight="bold",
        padding=12.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(
        background_color="#FEFCF5",  # off-white paper
    ),
)

# Ramon y Cajal -- inspired by Santiago Ramon y Cajal's histological ink
# drawings (1890s): fine sepia pen strokes on aged parchment, hand-drawn
# feel with delicate branching structures
CAJAL_THEME = Theme(
    name="cajal",
    node_styles={
        "default": NodeStyle(
            shape="ellipse",
            fill="#E6D5BC",  # parchment soma body
            stroke="#1A0E07",  # near-black india ink
            stroke_width=1.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#1A0E07",
            font_style="italic",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=20.0,
        ),
        "input": NodeStyle(
            shape="ellipse",
            fill="#C7B897",  # darker parchment (sensory neuron)
            stroke="#1A0E07",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#1A0E07",
            font_style="italic",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=20.0,
        ),
        "output": NodeStyle(
            shape="ellipse",
            fill="#D8C4A8",  # warm parchment (motor neuron)
            stroke="#1A0E07",
            stroke_width=2.5,  # heavier ink for emphasis
            font_family="Georgia",
            font_size=8.0,
            font_color="#1A0E07",
            font_style="italic",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=20.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#2C1A0E",  # dark sepia ink -- axon fibers
            width=0.8,  # thin pen strokes
            style="solid",
            arrow="vee",  # fine pointed tip
            arrow_fill="filled",
            arrow_color="#2C1A0E",
            arrow_length=4.0,
            arrow_width=2.5,
            routing="bezier",
            curvature=0.35,  # organic curves
        ),
        "back": EdgeStyle(
            color="#6B5038",  # lighter sepia -- recurrent collaterals
            width=0.6,
            style="solid",
            arrow="vee",
            arrow_fill="filled",
            arrow_color="#6B5038",
            arrow_length=3.5,
            arrow_width=2.0,
            routing="bezier",
            curvature=0.45,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F0E6D2",  # light parchment
        stroke="#8B7355",
        stroke_width=0.3,  # barely-there boundary
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=9.0,
        font_color="#6B5038",
        font_weight="normal",
        padding=8.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(
        background_color="#F7EDD8",  # aged yellowed paper
    ),
)

# Connectome -- inspired by fMRI connectome matrices and circular
# connectivity diagrams: saturated jewel tones on dark background,
# glowing edges, modern clinical neuroimaging aesthetic
CONNECTOME_THEME = Theme(
    name="connectome",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#2A5F8F",  # deep cortical blue
            stroke="#60B0FF",  # bright MRI highlight
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#C0DCFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#2D8B57",  # temporal lobe green
            stroke="#50E890",
            stroke_width=2.5,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#B0F0C8",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#8B2D5E",  # frontal lobe magenta
            stroke="#E850A0",
            stroke_width=2.5,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#F0B0D0",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#4090D0",  # tract blue with glow feel
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#4090D0",
            arrow_length=4.5,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.3,
            opacity=0.7,
        ),
        "back": EdgeStyle(
            color="#D06040",  # warm orange-red for feedback
            width=0.8,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#D06040",
            arrow_length=4.0,
            arrow_width=2.5,
            routing="bezier",
            curvature=0.4,
            opacity=0.6,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#1A2840",  # dark cluster region
        stroke="#3070A0",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=8.0,
        font_size=9.0,
        font_color="#80B0E0",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(
        background_color="#0D1520",  # dark MRI void
    ),
)

# Pathway -- inspired by textbook biochemical pathway diagrams (Krebs
# cycle, glycolysis, electron transport chain): curved reaction arrows
# connecting metabolite ovals, enzyme names in green, cofactors in
# orange, warm off-white textbook paper background
PATHWAY_THEME = Theme(
    name="pathway",
    node_styles={
        "default": NodeStyle(
            shape="ellipse",  # metabolite ovals (citrate, malate, etc.)
            fill="#FEFEFE",
            stroke="#3B6EA5",  # textbook blue
            stroke_width=1.8,
            font_family="Palatino",
            font_size=8.0,
            font_color="#2C3E50",
            padding=(6.0, 3.0),
            min_width=38.0,
            min_height=20.0,
        ),
        "input": NodeStyle(
            shape="rectangle",  # enzymes as rounded rectangles
            fill="#E6F4E6",  # light enzyme green
            stroke="#2D8B47",  # classic enzyme green
            stroke_width=1.8,
            font_family="Palatino",
            font_size=7.5,
            font_color="#1A5C2E",
            font_style="italic",  # enzyme names always italic
            padding=(6.0, 3.0),
            min_width=38.0,
            min_height=18.0,
            corner_radius=3.0,
        ),
        "output": NodeStyle(
            shape="ellipse",  # cofactors / products (NAD+, ATP, CO2)
            fill="#FFF5E6",  # warm cofactor cream
            stroke="#D4851F",  # amber cofactor border
            stroke_width=1.5,
            font_family="Palatino",
            font_size=7.5,
            font_color="#8B5A1A",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=18.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#444444",  # dark gray reaction arrow
            width=1.4,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#444444",
            arrow_length=6.0,
            arrow_width=4.0,
            routing="bezier",  # curved like Krebs cycle arrows
            curvature=0.25,
        ),
        "back": EdgeStyle(
            color="#B03030",  # red feedback / inhibition
            width=1.0,
            style="dashed",
            arrow="tee",  # inhibition bar
            arrow_fill="filled",
            arrow_color="#B03030",
            arrow_length=5.0,
            arrow_width=6.0,
            routing="bezier",
            curvature=0.3,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F5F0E8",  # warm parchment compartment (mitochondria, cytosol)
        stroke="#C0B090",
        stroke_width=1.5,
        stroke_dash="dashed",  # dashed membrane boundary
        corner_radius=8.0,
        font_size=9.0,
        font_color="#7A6B55",
        font_weight="bold",
        padding=12.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(
        background_color="#FDFAF3",  # textbook off-white
    ),
)

# Roadmap -- inspired by highway maps and route planning diagrams:
# nodes as map pins/junctions, edges as road segments with varying
# width, muted cartographic palette, subtle terrain background
ROADMAP_THEME = Theme(
    name="roadmap",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#FFFFFF",  # white junction marker
            stroke="#3D3D3D",  # dark road gray
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#2C2C2C",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
        "input": NodeStyle(
            shape="diamond",  # origin marker
            fill="#2E7D32",  # highway green
            stroke="#1B5E20",
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "output": NodeStyle(
            shape="diamond",  # destination marker
            fill="#C62828",  # exit red
            stroke="#8E0000",
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=26.0,
            min_height=26.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#F9A825",  # highway yellow-orange
            width=2.5,  # thick road
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#F9A825",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",  # gentle road curves
            curvature=0.2,
        ),
        "back": EdgeStyle(
            color="#90A4AE",  # secondary road gray-blue
            width=1.2,  # thinner side road
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#90A4AE",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.3,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#E8EAE6",  # muted terrain region
        stroke="#A0A898",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=6.0,
        font_size=9.0,
        font_color="#5D6458",
        font_weight="bold",
        padding=12.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(
        background_color="#F0EDE4",  # parchment map background
    ),
)

# Phylogeny -- inspired by evolutionary cladograms and phylogenetic
# trees: minimal decoration, thin precise branching lines, taxa names
# as the focus, clean academic journal aesthetic
PHYLOGENY_THEME = Theme(
    name="phylogeny",
    node_styles={
        "default": NodeStyle(
            shape="none",  # taxa are just labels, no box
            fill="transparent",
            stroke="transparent",
            stroke_width=0.0,
            font_family="Times New Roman",
            font_size=8.5,
            font_color="#1A1A1A",
            font_style="italic",  # species names in italic
            padding=(2.0, 1.0),
            min_width=0.0,
            min_height=0.0,
        ),
        "input": NodeStyle(
            shape="circle",  # root / ancestral node
            fill="#2C2C2C",
            stroke="#2C2C2C",
            stroke_width=1.0,
            font_family="Times New Roman",
            font_size=8.5,
            font_color="#1A1A1A",
            font_weight="bold",
            padding=(3.0, 3.0),
            min_width=6.0,  # small dot
            min_height=6.0,
        ),
        "output": NodeStyle(
            shape="none",  # extant taxa -- label only
            fill="transparent",
            stroke="transparent",
            stroke_width=0.0,
            font_family="Times New Roman",
            font_size=8.5,
            font_color="#1A1A1A",
            font_style="italic",
            font_weight="bold",
            padding=(2.0, 1.0),
            min_width=0.0,
            min_height=0.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#1A1A1A",  # crisp black branch
            width=1.0,
            style="solid",
            arrow="none",  # cladograms have no arrowheads
            routing="ortho",  # right-angle branching
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#999999",  # gray for horizontal transfer / reticulation
            width=0.7,
            style="dashed",
            arrow="none",
            routing="bezier",
            curvature=0.3,
        ),
    },
    cluster_style=ClusterStyle(
        fill="transparent",
        stroke="#AAAAAA",
        stroke_width=0.5,
        stroke_dash="dashed",  # subtle clade grouping
        corner_radius=0.0,
        font_size=8.0,
        font_color="#666666",
        font_weight="normal",
        padding=6.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(
        background_color="#FFFFFF",  # clean journal white
    ),
)

# Branches -- inspired by botanical tree branching: bark brown edges
# that taper outward, leaf-green terminal nodes, bud nodes at forks,
# dappled sunlight background
BRANCHES_THEME = Theme(
    name="branches",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#6B8E4E",  # muted leaf green
            stroke="#3E5427",  # darker leaf edge
            stroke_width=1.5,
            font_family="Georgia",
            font_size=7.5,
            font_color="#2A3B1A",
            padding=(4.0, 4.0),
            min_width=20.0,
            min_height=20.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#8B6944",  # heartwood brown (trunk / root)
            stroke="#5C4030",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#F5EDD8",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#A8C96A",  # bright new-leaf green
            stroke="#6B8E4E",
            stroke_width=1.0,
            font_family="Georgia",
            font_size=7.5,
            font_color="#2A3B1A",
            padding=(3.0, 3.0),
            min_width=16.0,
            min_height=16.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#6B5038",  # bark brown
            width=2.0,
            style="solid",
            arrow="none",  # branches don't have arrows
            routing="bezier",  # organic curves
            curvature=0.35,
        ),
        "back": EdgeStyle(
            color="#A09070",  # lighter deadwood
            width=1.0,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.45,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#D5E8B8",  # canopy green
        stroke="#8BA86A",
        stroke_width=0.8,
        stroke_dash="solid",
        corner_radius=12.0,  # soft organic shape
        font_size=8.5,
        font_color="#4A6030",
        font_weight="bold",
        padding=10.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(
        background_color="#F2EDE0",  # warm sunlit parchment
    ),
)

# Spiderweb -- radial silk threads on dark night background, dew-drop
# nodes glistening at intersections, gossamer connections
SPIDERWEB_THEME = Theme(
    name="spiderweb",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#C8D8E8",  # dewdrop pale blue
            stroke="#E8F0F8",  # bright glint
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=7.0,
            font_color="#D0D8E0",
            padding=(3.0, 3.0),
            min_width=12.0,  # small dewdrops
            min_height=12.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#E0E8F0",  # larger central dewdrop
            stroke="#F0F4F8",
            stroke_width=2.5,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#D0D8E0",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=20.0,
            min_height=20.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#A0B8D0",  # slightly darker outer dewdrop
            stroke="#C0D0E0",
            stroke_width=1.0,
            font_family="Helvetica",
            font_size=7.0,
            font_color="#B0C0D0",
            padding=(3.0, 3.0),
            min_width=10.0,
            min_height=10.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#8090A0",  # silver silk thread
            width=0.6,  # gossamer thin
            style="solid",
            arrow="none",  # webs don't have direction
            routing="straight",
            curvature=0.0,
            opacity=0.7,
        ),
        "back": EdgeStyle(
            color="#607080",  # deeper silk for spiral threads
            width=0.4,
            style="solid",
            arrow="none",
            routing="bezier",  # gentle spiral curve
            curvature=0.5,
            opacity=0.5,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#1A2030",  # dark night pocket
        stroke="#3A4858",
        stroke_width=0.5,
        stroke_dash="solid",
        corner_radius=20.0,  # soft circular web region
        font_size=7.5,
        font_color="#607888",
        font_weight="normal",
        padding=8.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(
        background_color="#0E141E",  # deep night sky
    ),
)

# ── Nature themes ─────────────────────────────────────────────────────

# Coral -- deep ocean reef: bioluminescent nodes, dark abyssal water
CORAL_THEME = Theme(
    name="coral",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#E87461",  # living coral
            stroke="#C0504D",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#FFE8E0",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#40C9A2",  # sea anemone teal
            stroke="#2A9D78",
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#E0FFF4",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#F4A940",  # clownfish orange
            stroke="#D08820",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#FFF0D0",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#60B8D0",  # bioluminescent blue
            width=1.2,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#60B8D0",
            arrow_length=4.5,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.35,
            opacity=0.75,
        ),
        "back": EdgeStyle(
            color="#8860B0",  # deep purple current
            width=0.8,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#8860B0",
            arrow_length=4.0,
            arrow_width=2.5,
            routing="bezier",
            curvature=0.4,
            opacity=0.6,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#0C2840",
        stroke="#2080A0",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=10.0,
        font_size=8.0,
        font_color="#60A8C0",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#081828"),
)

# Autumn -- fall foliage: warm reds, burnt orange, gold on woody brown
AUTUMN_THEME = Theme(
    name="autumn",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#D4763A",  # burnt orange leaf
            stroke="#A05020",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#3A1E0A",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#C0272D",  # deep maple red
            stroke="#8B1A1A",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#FFE0D0",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#E8B830",  # golden aspen
            stroke="#B89020",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#3A2A0A",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#6B4226",  # bare branch brown
            width=1.5,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.3,
        ),
        "back": EdgeStyle(
            color="#8B7355",
            width=0.8,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.4,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F0D8A0",
        stroke="#C0A060",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=8.0,
        font_size=8.5,
        font_color="#6B4226",
        font_weight="bold",
        padding=10.0,
        opacity=0.35,
    ),
    graph_style=GraphStyle(background_color="#F5ECD0"),
)

# Aurora -- northern lights on arctic night sky
AURORA_THEME = Theme(
    name="aurora",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#30A878",  # aurora green
            stroke="#50E0A0",
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#C0FFD8",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#6838B0",  # aurora purple
            stroke="#A060E8",
            stroke_width=2.5,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#D8C0F8",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#C83880",  # aurora pink
            stroke="#E860A8",
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#FFD0E8",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#40D890",
            width=1.2,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#40D890",
            arrow_length=4.5,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.3,
            opacity=0.65,
        ),
        "back": EdgeStyle(
            color="#9050D0",
            width=0.8,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#9050D0",
            arrow_length=4.0,
            arrow_width=2.5,
            routing="bezier",
            curvature=0.4,
            opacity=0.5,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#102030",
        stroke="#305050",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=12.0,
        font_size=8.0,
        font_color="#60A088",
        font_weight="bold",
        padding=10.0,
        opacity=0.35,
    ),
    graph_style=GraphStyle(background_color="#080E18"),
)

# Cave -- Lascaux cave paintings: ochre and charcoal on dark stone
CAVE_THEME = Theme(
    name="cave",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#C4883C",  # ochre pigment
            stroke="#8B6020",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#2A1A0A",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#B03020",  # red ochre
            stroke="#801810",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#F0D8C0",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#E0C878",  # yellow ochre
            stroke="#B0A050",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#2A1A0A",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#3A2A1A",  # charcoal
            width=1.8,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.4,
        ),
        "back": EdgeStyle(
            color="#5A4A38",
            width=1.0,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.5,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#3A3028",
        stroke="#5A4838",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=8.5,
        font_color="#A08860",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#2A221A"),
)

# ── Art & design themes ──────────────────────────────────────────────

# Stained glass -- cathedral jewel tones, thick black lead came
STAINED_GLASS_THEME = Theme(
    name="stained_glass",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#2860A8",  # cobalt blue glass
            stroke="#1A1A1A",  # lead came
            stroke_width=3.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#D0E0F8",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=22.0,
            corner_radius=2.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#C82030",  # ruby red glass
            stroke="#1A1A1A",
            stroke_width=3.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#FFD0D0",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=22.0,
            corner_radius=2.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#D4A010",  # amber gold glass
            stroke="#1A1A1A",
            stroke_width=3.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#3A2A00",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=22.0,
            corner_radius=2.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#1A1A1A",  # lead came
            width=2.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#1A1A1A",
            arrow_length=5.0,
            arrow_width=4.0,
            routing="straight",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#3A3A3A",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#3A3A3A",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="straight",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#182028",
        stroke="#1A1A1A",
        stroke_width=3.0,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=9.0,
        font_color="#C0B080",
        font_weight="bold",
        padding=12.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#101418"),
)

# Watercolor -- soft translucent washes on textured paper
WATERCOLOR_THEME = Theme(
    name="watercolor",
    node_styles={
        "default": NodeStyle(
            shape="ellipse",
            fill="#88B8D8",  # cerulean wash
            stroke="#5090B0",
            stroke_width=0.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#2A3A4A",
            padding=(5.0, 4.0),
            min_width=28.0,
            min_height=22.0,
            opacity=0.7,
        ),
        "input": NodeStyle(
            shape="ellipse",
            fill="#D88888",  # alizarin wash
            stroke="#B06060",
            stroke_width=0.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#4A2020",
            font_weight="bold",
            padding=(5.0, 4.0),
            min_width=28.0,
            min_height=22.0,
            opacity=0.7,
        ),
        "output": NodeStyle(
            shape="ellipse",
            fill="#C8C878",  # yellow ochre wash
            stroke="#A0A050",
            stroke_width=0.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#3A3A1A",
            padding=(5.0, 4.0),
            min_width=28.0,
            min_height=22.0,
            opacity=0.7,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#6888A0",
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#6888A0",
            arrow_length=4.5,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.3,
            opacity=0.5,
        ),
        "back": EdgeStyle(
            color="#A08888",
            width=0.7,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#A08888",
            arrow_length=4.0,
            arrow_width=2.5,
            routing="bezier",
            curvature=0.4,
            opacity=0.4,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#D0D8C8",
        stroke="#A0A890",
        stroke_width=0.3,
        stroke_dash="solid",
        corner_radius=12.0,
        font_size=8.5,
        font_color="#607058",
        font_weight="normal",
        padding=10.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#F5F0E5"),
)

# Ukiyo-e -- Japanese woodblock print: indigo, vermillion, gold
UKIYO_E_THEME = Theme(
    name="ukiyo_e",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#2B4570",  # deep indigo
            stroke="#1A2A40",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#E8D8C0",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=1.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#C03020",  # vermillion
            stroke="#801810",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#FFE8D0",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=1.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#C8A028",  # gold leaf
            stroke="#987818",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#2A2008",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=1.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#1A2A40",  # sumi ink
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#1A2A40",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.25,
        ),
        "back": EdgeStyle(
            color="#506880",
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#506880",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.35,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#E8D8C0",
        stroke="#8B7355",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=9.0,
        font_color="#3A2A1A",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#F0E4D0"),  # rice paper
)

# Illuminated -- medieval manuscript: gold leaf, lapis blue, vellum
ILLUMINATED_THEME = Theme(
    name="illuminated",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#1A3C6E",  # lapis lazuli blue
            stroke="#C8A030",  # gold leaf border
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#E8D8A0",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=22.0,
            corner_radius=2.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#8B1A2B",  # deep carmine
            stroke="#C8A030",
            stroke_width=3.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#F0D8A0",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=22.0,
            corner_radius=2.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#C8A030",  # gold leaf fill
            stroke="#8B6D18",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#2A1A08",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=22.0,
            corner_radius=2.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#C8A030",  # gold ink
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#C8A030",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.25,
        ),
        "back": EdgeStyle(
            color="#6B5028",  # dark gold
            width=1.0,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#6B5028",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.35,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#2A1E30",
        stroke="#C8A030",
        stroke_width=2.0,
        stroke_dash="solid",
        corner_radius=3.0,
        font_size=9.0,
        font_color="#C8A030",
        font_weight="bold",
        padding=12.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#E8D8B8"),  # vellum
)

# ── Sci-fi & pop culture themes ──────────────────────────────────────

# Matrix -- green phosphor on black, digital rain
MATRIX_THEME = Theme(
    name="matrix",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#0A1A0A",
            stroke="#00CC00",  # terminal green
            stroke_width=1.5,
            font_family="Courier New",
            font_size=8.0,
            font_color="#00FF00",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=18.0,
            corner_radius=0.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#003300",
            stroke="#00FF00",
            stroke_width=2.0,
            font_family="Courier New",
            font_size=8.0,
            font_color="#00FF00",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=18.0,
            corner_radius=0.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#0A1A0A",
            stroke="#00AA00",
            stroke_width=1.5,
            font_family="Courier New",
            font_size=8.0,
            font_color="#00DD00",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=18.0,
            corner_radius=0.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#00AA00",
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#00AA00",
            arrow_length=5.0,
            arrow_width=3.0,
            routing="straight",
            curvature=0.0,
            opacity=0.7,
        ),
        "back": EdgeStyle(
            color="#006600",
            width=0.6,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#006600",
            arrow_length=4.0,
            arrow_width=2.5,
            routing="straight",
            curvature=0.0,
            opacity=0.5,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#001A00",
        stroke="#004400",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=8.0,
        font_color="#008800",
        font_weight="bold",
        padding=8.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#000000"),
)

# Tron -- neon cyan grid on pure black
TRON_THEME = Theme(
    name="tron",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#0A0A14",
            stroke="#00D4FF",  # neon cyan
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#00D4FF",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=20.0,
            corner_radius=1.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#0A0A14",
            stroke="#FF6600",  # orange program
            stroke_width=2.5,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FF8830",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=20.0,
            corner_radius=1.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#0A0A14",
            stroke="#FFFFFF",  # white user
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=20.0,
            corner_radius=1.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#00D4FF",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#00D4FF",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="ortho",  # grid lines
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#0088AA",
            width=0.8,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#0088AA",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="ortho",
            curvature=0.0,
            opacity=0.6,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#0A0A14",
        stroke="#00688B",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=8.0,
        font_color="#00A0CC",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#000000"),
)

# Steampunk -- brass/copper nodes, Victorian riveted edges
STEAMPUNK_THEME = Theme(
    name="steampunk",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#B8860B",  # dark goldenrod brass
            stroke="#8B6914",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#2A1A08",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#B87333",  # copper
            stroke="#8B5A2B",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#FFE8D0",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#708090",  # steel gray
            stroke="#4A5A68",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#E8E8E8",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=26.0,
            min_height=26.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#5A4020",  # dark brass pipe
            width=2.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#5A4020",
            arrow_length=5.0,
            arrow_width=4.0,
            routing="ortho",  # rigid pipes
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#786040",
            width=1.2,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#786040",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="ortho",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#3A3028",
        stroke="#8B7355",
        stroke_width=2.0,
        stroke_dash="solid",
        corner_radius=4.0,
        font_size=9.0,
        font_color="#B8960B",
        font_weight="bold",
        padding=12.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#2A2218"),
)

# Pixel -- 8-bit retro game aesthetic
PIXEL_THEME = Theme(
    name="pixel",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#5B6EE1",  # NES blue
            stroke="#222034",
            stroke_width=2.0,
            font_family="Courier New",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 2.0),
            min_width=28.0,
            min_height=18.0,
            corner_radius=0.0,  # sharp pixel edges
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#AC3232",  # NES red
            stroke="#222034",
            stroke_width=2.0,
            font_family="Courier New",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 2.0),
            min_width=28.0,
            min_height=18.0,
            corner_radius=0.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#6ABE30",  # NES green
            stroke="#222034",
            stroke_width=2.0,
            font_family="Courier New",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 2.0),
            min_width=28.0,
            min_height=18.0,
            corner_radius=0.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#FBFB36",  # NES yellow
            width=2.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#FBFB36",
            arrow_length=5.0,
            arrow_width=4.0,
            routing="ortho",  # pixel-perfect right angles
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#76428A",  # NES purple
            width=1.5,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#76428A",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="ortho",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#222034",
        stroke="#524B6B",
        stroke_width=2.0,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=8.0,
        font_color="#CBDBFC",
        font_weight="bold",
        padding=8.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#222034"),
)

# ── Science visualization themes ─────────────────────────────────────

# X-ray -- radiograph: blue-white on black, translucent
XRAY_THEME = Theme(
    name="xray",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#C0D8E8",
            stroke="#E0F0FF",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#E0F0FF",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
            opacity=0.7,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#E8F0FF",
            stroke="#FFFFFF",
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
            opacity=0.85,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#8098B0",
            stroke="#A0B8D0",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#C0D8E8",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
            opacity=0.6,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#80A8C8",
            width=0.8,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#80A8C8",
            arrow_length=4.0,
            arrow_width=2.5,
            routing="straight",
            curvature=0.0,
            opacity=0.5,
        ),
        "back": EdgeStyle(
            color="#506878",
            width=0.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#506878",
            arrow_length=3.5,
            arrow_width=2.0,
            routing="straight",
            curvature=0.0,
            opacity=0.35,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#101820",
        stroke="#304050",
        stroke_width=0.5,
        stroke_dash="solid",
        corner_radius=6.0,
        font_size=7.5,
        font_color="#607888",
        font_weight="normal",
        padding=8.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#000000"),
)

# Thermal -- infrared heat map: cool blue to hot red
THERMAL_THEME = Theme(
    name="thermal",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#E8C020",  # warm yellow
            stroke="#D0A010",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#1A1A1A",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#E03030",  # hot red
            stroke="#B02020",
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FFE0D0",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#3060C0",  # cool blue
            stroke="#2048A0",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#D0E0FF",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#D08020",  # warm orange
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#D08020",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.2,
            opacity=0.7,
        ),
        "back": EdgeStyle(
            color="#4070A0",
            width=0.8,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#4070A0",
            arrow_length=4.0,
            arrow_width=2.5,
            routing="bezier",
            curvature=0.3,
            opacity=0.5,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#1A1018",
        stroke="#403028",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=4.0,
        font_size=8.0,
        font_color="#A08060",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#0A0808"),
)

# Microscopy -- electron microscope: grayscale with false-color accents
MICROSCOPY_THEME = Theme(
    name="microscopy",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#A0A0A0",  # mid-gray specimen
            stroke="#C8C8C8",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#E0E0E0",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#30A060",  # false-color green (fluorescence)
            stroke="#20C070",
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#D0FFE0",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#C030A0",  # false-color magenta
            stroke="#E050C0",
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#FFD0F0",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#888888",
            width=0.8,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#888888",
            arrow_length=4.0,
            arrow_width=2.5,
            routing="bezier",
            curvature=0.2,
            opacity=0.6,
        ),
        "back": EdgeStyle(
            color="#555555",
            width=0.5,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#555555",
            arrow_length=3.5,
            arrow_width=2.0,
            routing="bezier",
            curvature=0.3,
            opacity=0.4,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#1A1A1A",
        stroke="#404040",
        stroke_width=0.5,
        stroke_dash="dashed",
        corner_radius=6.0,
        font_size=7.5,
        font_color="#808080",
        font_weight="normal",
        padding=8.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#0A0A0A"),
)

# Topographic -- contour map: earth tones, elevation feel
TOPOGRAPHIC_THEME = Theme(
    name="topographic",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#8B7E6A",  # earth brown
            stroke="#6B5E4A",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#2A2218",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=20.0,
            min_height=20.0,
        ),
        "input": NodeStyle(
            shape="triangle",  # peak marker
            fill="#4A7848",  # forest green
            stroke="#3A5838",
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#E0F0E0",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=24.0,
            min_height=24.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#4878A8",  # lake blue
            stroke="#305880",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#D0E0F0",
            padding=(4.0, 4.0),
            min_width=20.0,
            min_height=20.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#8B6844",  # trail brown
            width=1.2,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#8B6844",
            arrow_length=4.5,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.3,
        ),
        "back": EdgeStyle(
            color="#A09880",  # contour line
            width=0.6,
            style="dashed",
            arrow="none",
            routing="bezier",
            curvature=0.4,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#D8D0B8",
        stroke="#A09878",
        stroke_width=0.8,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=8.0,
        font_color="#6B5E4A",
        font_weight="bold",
        padding=10.0,
        opacity=0.35,
    ),
    graph_style=GraphStyle(background_color="#E8E0D0"),
)

# ── History themes ───────────────────────────────────────────────────

# Hieroglyph -- Egyptian: papyrus, gold, lapis blue
HIEROGLYPH_THEME = Theme(
    name="hieroglyph",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#C8A040",  # gold cartouche
            stroke="#8B6D18",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#1A1208",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=22.0,
            corner_radius=3.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#1A3868",  # lapis lazuli
            stroke="#C8A040",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#D8C898",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=22.0,
            corner_radius=3.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#A82818",  # carnelian red
            stroke="#C8A040",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#F0D8A0",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=22.0,
            corner_radius=3.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#8B6D18",  # dark gold
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#8B6D18",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="ortho",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#A09060",
            width=1.0,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#A09060",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="ortho",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#D8C898",
        stroke="#8B6D18",
        stroke_width=2.0,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=9.0,
        font_color="#4A3A18",
        font_weight="bold",
        padding=12.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#E8D8B0"),  # papyrus
)

# Roman mosaic -- tessellated earth tones, stone texture feel
ROMAN_MOSAIC_THEME = Theme(
    name="roman_mosaic",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#C8B898",  # limestone tesserae
            stroke="#8B7E68",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#3A3028",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=28.0,
            min_height=22.0,
            corner_radius=1.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#8B3020",  # terracotta red
            stroke="#6B2018",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#F0D8C0",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=28.0,
            min_height=22.0,
            corner_radius=1.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#2A5060",  # dark teal tesserae
            stroke="#1A3A48",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#C0D8E0",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=28.0,
            min_height=22.0,
            corner_radius=1.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#5A4A38",  # grout line
            width=2.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#5A4A38",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="straight",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#7A6A58",
            width=1.2,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#7A6A58",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="straight",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#B8A888",
        stroke="#8B7E68",
        stroke_width=2.5,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=9.0,
        font_color="#4A3828",
        font_weight="bold",
        padding=12.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#D8C8A8"),
)

# ── Game & geography themes ──────────────────────────────────────────

# Catan -- Settlers of Catan: hex terrain colors, roads, settlements
CATAN_THEME = Theme(
    name="catan",
    node_styles={
        "default": NodeStyle(
            shape="hexagon",
            fill="#C8A848",  # wheat/grain hex
            stroke="#8B7428",
            stroke_width=2.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#3A2A08",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "input": NodeStyle(
            shape="pentagon",  # settlement
            fill="#CC4422",  # brick red settlement
            stroke="#8B2E15",
            stroke_width=2.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFE0D0",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "output": NodeStyle(
            shape="rectangle",  # city
            fill="#4A6E8B",  # ore blue-gray city
            stroke="#2A4A68",
            stroke_width=2.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#D0E0F0",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=24.0,
            corner_radius=2.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#8B6020",  # dirt road
            width=2.5,
            style="solid",
            arrow="none",  # roads don't have direction
            routing="straight",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#607850",  # forest path
            width=1.5,
            style="dashed",
            arrow="none",
            routing="bezier",
            curvature=0.2,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#B8D080",  # sheep pasture green
        stroke="#80A048",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=6.0,
        font_size=9.0,
        font_color="#4A5828",
        font_weight="bold",
        padding=12.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#70A0C8"),  # ocean blue
)

# Archipelago -- islands and bridges: sandy nodes, blue water, bridge edges
ARCHIPELAGO_THEME = Theme(
    name="archipelago",
    node_styles={
        "default": NodeStyle(
            shape="ellipse",
            fill="#E8D8A0",  # sandy island
            stroke="#C0A868",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#4A3A18",
            padding=(6.0, 4.0),
            min_width=32.0,
            min_height=22.0,
        ),
        "input": NodeStyle(
            shape="ellipse",
            fill="#68A860",  # lush main island
            stroke="#488840",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#1A3A10",
            font_weight="bold",
            padding=(6.0, 4.0),
            min_width=36.0,
            min_height=26.0,
        ),
        "output": NodeStyle(
            shape="ellipse",
            fill="#D0B880",  # atoll sand
            stroke="#A89060",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=7.5,
            font_color="#5A4A28",
            padding=(5.0, 3.0),
            min_width=28.0,
            min_height=18.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#8B7040",  # wooden bridge / rope bridge
            width=2.0,
            style="solid",
            arrow="none",  # bridges are bidirectional
            routing="bezier",
            curvature=0.2,
        ),
        "back": EdgeStyle(
            color="#5088A0",  # sea route (lighter)
            width=1.0,
            style="dashed",  # dotted sea lane
            arrow="none",
            routing="bezier",
            curvature=0.35,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#90C8B0",  # shallow lagoon
        stroke="#60A088",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=14.0,  # organic lagoon shape
        font_size=8.5,
        font_color="#2A5040",
        font_weight="bold",
        padding=12.0,
        opacity=0.35,
    ),
    graph_style=GraphStyle(background_color="#3878A8"),  # deep ocean
)

# Mario overworld -- Super Mario World map: green pipes, brick paths,
# castle nodes, bright Nintendo palette on sky blue
MARIO_THEME = Theme(
    name="mario",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#E8B830",  # ? block gold
            stroke="#C09020",
            stroke_width=2.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#3A2A08",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
        "input": NodeStyle(
            shape="rectangle",  # castle
            fill="#B83830",  # castle brick red
            stroke="#882020",
            stroke_width=3.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFE0D0",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=24.0,
            corner_radius=0.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#30A830",  # warp pipe green
            stroke="#208020",
            stroke_width=2.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#E0FFE0",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#C8A060",  # dirt path
            width=3.0,  # chunky overworld road
            style="solid",
            arrow="none",  # paths don't have arrows
            routing="straight",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#68A830",  # grass path shortcut
            width=1.5,
            style="dashed",
            arrow="none",
            routing="bezier",
            curvature=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#68B838",  # grass world
        stroke="#488828",
        stroke_width=2.0,
        stroke_dash="solid",
        corner_radius=8.0,
        font_size=9.0,
        font_color="#1A4A08",
        font_weight="bold",
        padding=12.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#6898F8"),  # sky blue
)

# Mycelium -- underground fungal network: pale filaments radiating
# through dark soil, fruiting body nodes, organic branching
MYCELIUM_THEME = Theme(
    name="mycelium",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#D8C8A8",  # pale hyphal knot
            stroke="#C0A878",
            stroke_width=1.0,
            font_family="Georgia",
            font_size=7.0,
            font_color="#E8D8C0",
            padding=(3.0, 3.0),
            min_width=14.0,  # small junction points
            min_height=14.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#C89060",  # mushroom cap brown
            stroke="#A87040",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#F0E0C8",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=26.0,  # fruiting body
            min_height=26.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#E0D0B0",  # pale mycelium tip
            stroke="#C8B890",
            stroke_width=0.8,
            font_family="Georgia",
            font_size=7.0,
            font_color="#D0C0A0",
            padding=(2.0, 2.0),
            min_width=10.0,
            min_height=10.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#C8B890",  # pale filament
            width=0.7,  # gossamer hyphae
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.4,  # organic branching
            opacity=0.6,
        ),
        "back": EdgeStyle(
            color="#A09870",  # deeper filament
            width=0.4,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.55,
            opacity=0.4,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#1A1810",
        stroke="#302818",
        stroke_width=0.5,
        stroke_dash="solid",
        corner_radius=10.0,
        font_size=7.5,
        font_color="#807050",
        font_weight="normal",
        padding=8.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#100E08"),  # dark soil
)

# xkcd -- Randall Munroe stick-figure style: wobbly hand-drawn feel,
# simple black-on-white, Comic Sans (sorry not sorry)
XKCD_THEME = Theme(
    name="xkcd",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#FFFFFF",
            stroke="#000000",
            stroke_width=1.5,
            font_family="Comic Sans MS",
            font_size=9.0,
            font_color="#000000",
            padding=(5.0, 5.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#FFFFFF",
            stroke="#000000",
            stroke_width=2.5,
            font_family="Comic Sans MS",
            font_size=9.0,
            font_color="#000000",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#DDDDDD",
            stroke="#000000",
            stroke_width=1.5,
            font_family="Comic Sans MS",
            font_size=9.0,
            font_color="#000000",
            padding=(5.0, 5.0),
            min_width=26.0,
            min_height=26.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#000000",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#000000",
            arrow_length=6.0,
            arrow_width=4.0,
            routing="bezier",
            curvature=0.15,
        ),
        "back": EdgeStyle(
            color="#888888",
            width=1.0,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#888888",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.2,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F8F8F8",
        stroke="#000000",
        stroke_width=1.0,
        stroke_dash="dashed",
        corner_radius=8.0,
        font_size=10.0,
        font_color="#000000",
        font_weight="bold",
        padding=10.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

# Slime mold -- Physarum polycephalum solving a maze: bright yellow
# plasmodium threads on dark agar, pulsating network
SLIME_MOLD_THEME = Theme(
    name="slime_mold",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#E8D020",  # bright slime yellow
            stroke="#C8B018",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=7.0,
            font_color="#1A1A08",
            padding=(3.0, 3.0),
            min_width=16.0,
            min_height=16.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#F0E030",  # oat flake (food source)
            stroke="#D0C020",
            stroke_width=2.5,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#2A2808",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#C8B020",  # fading tendril tip
            stroke="#A89018",
            stroke_width=1.0,
            font_family="Helvetica",
            font_size=7.0,
            font_color="#303008",
            padding=(2.0, 2.0),
            min_width=12.0,
            min_height=12.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#D8C020",  # plasmodium tube
            width=1.8,
            style="solid",
            arrow="none",  # network is undirected
            routing="bezier",
            curvature=0.3,
            opacity=0.8,
        ),
        "back": EdgeStyle(
            color="#A89818",  # thinner exploratory tube
            width=0.6,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.5,
            opacity=0.4,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#181810",
        stroke="#2A2818",
        stroke_width=0.5,
        stroke_dash="solid",
        corner_radius=6.0,
        font_size=7.5,
        font_color="#606020",
        font_weight="normal",
        padding=8.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#0A0A08"),  # dark agar
)

# Cavern -- underground cave system: stalactites, subterranean pools,
# limestone passages, headlamp glow
CAVERN_THEME = Theme(
    name="cavern",
    node_styles={
        "default": NodeStyle(
            shape="ellipse",
            fill="#585050",  # limestone chamber
            stroke="#787068",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=7.5,
            font_color="#D0C8B8",
            padding=(5.0, 4.0),
            min_width=28.0,
            min_height=20.0,
        ),
        "input": NodeStyle(
            shape="ellipse",
            fill="#384858",  # underground pool
            stroke="#5878A0",  # water glint
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#A0C0E0",
            font_weight="bold",
            padding=(5.0, 4.0),
            min_width=30.0,
            min_height=22.0,
        ),
        "output": NodeStyle(
            shape="ellipse",
            fill="#C8A858",  # headlamp-lit formation
            stroke="#E0C870",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=7.5,
            font_color="#2A2218",
            padding=(5.0, 4.0),
            min_width=26.0,
            min_height=18.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#605848",  # passage walls
            width=2.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#605848",
            arrow_length=4.5,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.35,
        ),
        "back": EdgeStyle(
            color="#484038",  # narrow squeeze
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#484038",
            arrow_length=3.5,
            arrow_width=2.5,
            routing="bezier",
            curvature=0.5,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#282420",  # cavern chamber
        stroke="#484038",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=10.0,
        font_size=8.0,
        font_color="#908070",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#1A1614"),  # deep underground
)

# Flight map -- airline route map: city dots on dark globe, great-circle
# arcs, hub airports larger, classic SkyTeam / in-flight magazine feel
FLIGHT_MAP_THEME = Theme(
    name="flight_map",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#E8E0D0",  # city dot
            stroke="#C0B8A8",
            stroke_width=1.0,
            font_family="Helvetica",
            font_size=7.0,
            font_color="#D0C8B8",
            padding=(3.0, 3.0),
            min_width=10.0,  # small city
            min_height=10.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#FFFFFF",  # major hub
            stroke="#E83030",  # airline red
            stroke_width=2.5,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=22.0,  # hub airport
            min_height=22.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#D0C8B8",
            stroke="#A09888",
            stroke_width=1.0,
            font_family="Helvetica",
            font_size=7.0,
            font_color="#C0B8A0",
            padding=(3.0, 3.0),
            min_width=8.0,  # small destination
            min_height=8.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#E83030",  # airline red route
            width=0.8,
            style="solid",
            arrow="none",  # routes are bidirectional
            routing="bezier",  # great-circle arc
            curvature=0.3,
            opacity=0.5,
        ),
        "back": EdgeStyle(
            color="#F8A828",  # secondary route orange
            width=0.5,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.35,
            opacity=0.35,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#1A2030",
        stroke="#304050",
        stroke_width=0.5,
        stroke_dash="dashed",
        corner_radius=8.0,
        font_size=7.5,
        font_color="#607080",
        font_weight="bold",
        padding=10.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#141E28"),  # dark globe
)

# Telecom -- phone/telecom network: cell towers, signal paths,
# copper/fiber trunk lines, technical blue palette
TELECOM_THEME = Theme(
    name="telecom",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#E8E8E8",  # equipment gray
            stroke="#3070A0",  # telco blue
            stroke_width=1.5,
            font_family="Arial",
            font_size=7.5,
            font_color="#1A3050",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=18.0,
            corner_radius=2.0,
        ),
        "input": NodeStyle(
            shape="triangle",  # cell tower
            fill="#3070A0",
            stroke="#1A4878",
            stroke_width=2.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#D0E0F0",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#F0F0F0",  # endpoint device
            stroke="#3070A0",
            stroke_width=1.5,
            font_family="Arial",
            font_size=7.5,
            font_color="#1A3050",
            padding=(4.0, 4.0),
            min_width=20.0,
            min_height=20.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#3070A0",  # fiber trunk
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#3070A0",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="ortho",  # structured cable runs
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#90B0D0",  # wireless link
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#90B0D0",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="straight",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#E0E8F0",
        stroke="#3070A0",
        stroke_width=1.0,
        stroke_dash="dashed",
        corner_radius=4.0,
        font_size=8.0,
        font_color="#2050A0",
        font_weight="bold",
        padding=10.0,
        opacity=0.35,
    ),
    graph_style=GraphStyle(background_color="#F8F8FC"),
)

# Social -- social network graph: profile-pic circles, relationship
# edges, Facebook/LinkedIn blue tones, clean modern
SOCIAL_THEME = Theme(
    name="social",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#4267B2",  # Facebook blue
            stroke="#365899",
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#E44D26",  # influencer / high-degree orange-red
            stroke="#C03018",
            stroke_width=2.5,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=30.0,
            min_height=30.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#7FB3D8",  # casual connection light blue
            stroke="#5A9BC0",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#FFFFFF",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#8B9DC3",  # friendship line
            width=1.0,
            style="solid",
            arrow="none",  # friendships are mutual
            routing="bezier",
            curvature=0.2,
            opacity=0.5,
        ),
        "back": EdgeStyle(
            color="#DFE3EE",  # weak tie
            width=0.5,
            style="dashed",
            arrow="none",
            routing="bezier",
            curvature=0.3,
            opacity=0.35,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#E9EBEE",  # group background
        stroke="#8B9DC3",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=10.0,
        font_size=8.5,
        font_color="#4267B2",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

# Ant colony -- pheromone trails on sandy earth: dark ant nodes,
# amber pheromone paths of varying intensity, nest chambers
ANT_COLONY_THEME = Theme(
    name="ant_colony",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#2A1A0E",  # dark ant body
            stroke="#4A3020",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=7.0,
            font_color="#D0B888",
            padding=(3.0, 3.0),
            min_width=14.0,
            min_height=14.0,
        ),
        "input": NodeStyle(
            shape="ellipse",  # nest chamber
            fill="#5A4030",  # packed earth
            stroke="#8B6844",
            stroke_width=2.5,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#E0C8A0",
            font_weight="bold",
            padding=(6.0, 4.0),
            min_width=32.0,
            min_height=22.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#608838",  # food source leaf green
            stroke="#487028",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#E0F0D0",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=20.0,
            min_height=20.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#C89838",  # strong pheromone trail (amber)
            width=1.8,
            style="solid",
            arrow="none",  # trails are bidirectional
            routing="bezier",
            curvature=0.25,
            opacity=0.7,
        ),
        "back": EdgeStyle(
            color="#A08030",  # fading pheromone
            width=0.6,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.4,
            opacity=0.3,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#2A2018",  # tunnel chamber
        stroke="#4A3828",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=8.0,
        font_size=7.5,
        font_color="#907850",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#18140E"),  # underground earth
)

# Noir -- black & white film noir: high contrast, deep shadows
NOIR_THEME = Theme(
    name="noir",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#1A1A1A",
            stroke="#E0E0E0",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#E0E0E0",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=20.0,
            corner_radius=1.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#F0F0F0",  # bright under the streetlamp
            stroke="#1A1A1A",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#1A1A1A",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=22.0,
            corner_radius=1.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#3A3A3A",  # deep shadow
            stroke="#A0A0A0",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#C0C0C0",
            font_style="italic",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=20.0,
            corner_radius=1.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#C0C0C0",
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#C0C0C0",
            arrow_length=5.0,
            arrow_width=3.0,
            routing="straight",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#606060",
            width=0.6,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#606060",
            arrow_length=4.0,
            arrow_width=2.5,
            routing="straight",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#0A0A0A",
        stroke="#404040",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=9.0,
        font_color="#808080",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#000000"),
)

# Cyberpunk -- neon pink/cyan on dark rainy city
CYBERPUNK_THEME = Theme(
    name="cyberpunk",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#1A0A20",
            stroke="#FF2D95",  # hot pink neon
            stroke_width=2.0,
            font_family="Courier New",
            font_size=8.0,
            font_color="#FF2D95",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=18.0,
            corner_radius=1.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#1A0A20",
            stroke="#00FFF0",  # electric cyan
            stroke_width=2.5,
            font_family="Courier New",
            font_size=8.0,
            font_color="#00FFF0",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=1.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#1A0A20",
            stroke="#B030FF",  # purple neon
            stroke_width=2.0,
            font_family="Courier New",
            font_size=8.0,
            font_color="#C060FF",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=18.0,
            corner_radius=1.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#FF2D95",
            width=1.2,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#FF2D95",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.2,
            opacity=0.7,
        ),
        "back": EdgeStyle(
            color="#00FFF0",
            width=0.8,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#00FFF0",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.3,
            opacity=0.5,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#0A0418",
        stroke="#402060",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=2.0,
        font_size=8.0,
        font_color="#8040C0",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#0A0410"),
)

# Vascular -- arterial red / venous blue on anatomy beige
VASCULAR_THEME = Theme(
    name="vascular",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#E0D0C0",  # tissue beige
            stroke="#A08878",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#4A3828",
            padding=(4.0, 4.0),
            min_width=18.0,
            min_height=18.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#CC2020",  # arterial red (heart)
            stroke="#991010",
            stroke_width=2.5,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FFD0D0",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#3060A0",  # venous blue
            stroke="#204080",
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#C0D0E8",
            padding=(4.0, 4.0),
            min_width=20.0,
            min_height=20.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#CC2020",  # artery
            width=1.8,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#CC2020",
            arrow_length=4.5,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.3,
        ),
        "back": EdgeStyle(
            color="#3060A0",  # vein
            width=1.2,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#3060A0",
            arrow_length=4.0,
            arrow_width=2.5,
            routing="bezier",
            curvature=0.35,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F0E0D0",
        stroke="#C0A890",
        stroke_width=1.0,
        stroke_dash="dashed",
        corner_radius=10.0,
        font_size=8.0,
        font_color="#806050",
        font_weight="bold",
        padding=10.0,
        opacity=0.35,
    ),
    graph_style=GraphStyle(background_color="#F8EEE0"),
)

# Nebula -- deep space gas clouds, purple/teal wisps
NEBULA_THEME = Theme(
    name="nebula",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#D0D8F8",  # star white-blue
            stroke="#E8E8FF",
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#E0E8FF",
            padding=(3.0, 3.0),
            min_width=16.0,
            min_height=16.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#F8E8A0",  # bright star gold
            stroke="#FFE860",
            stroke_width=2.5,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FFF0C0",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#8070B0",  # dim purple star
            stroke="#A090D0",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#C0B8E0",
            padding=(3.0, 3.0),
            min_width=14.0,
            min_height=14.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#6050A0",  # purple gas wisp
            width=1.0,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.4,
            opacity=0.45,
        ),
        "back": EdgeStyle(
            color="#307880",  # teal gas wisp
            width=0.6,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.5,
            opacity=0.3,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#1A1030",
        stroke="#302050",
        stroke_width=0.5,
        stroke_dash="solid",
        corner_radius=16.0,
        font_size=7.5,
        font_color="#605088",
        font_weight="normal",
        padding=10.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#08060E"),
)

# Lava -- volcanic obsidian with glowing orange cracks
LAVA_THEME = Theme(
    name="lava",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#1A1A1A",  # obsidian
            stroke="#E85020",  # magma glow
            stroke_width=2.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#F0A040",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#E83010",  # molten core
            stroke="#FF6030",
            stroke_width=3.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFE0A0",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#2A2020",  # cooled basalt
            stroke="#803020",
            stroke_width=1.5,
            font_family="Arial",
            font_size=7.5,
            font_color="#C08060",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#E05018",  # lava flow
            width=2.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#E05018",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.3,
            opacity=0.8,
        ),
        "back": EdgeStyle(
            color="#803010",  # cooling flow
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#803010",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.4,
            opacity=0.5,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#1A0A08",
        stroke="#402010",
        stroke_width=2.0,
        stroke_dash="solid",
        corner_radius=4.0,
        font_size=8.0,
        font_color="#C06030",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#0A0606"),
)

# Frost -- ice crystals, cold blue-white on steel gray
FROST_THEME = Theme(
    name="frost",
    node_styles={
        "default": NodeStyle(
            shape="diamond",
            fill="#D8E8F8",  # ice crystal
            stroke="#A0C0E0",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#2A4060",
            padding=(5.0, 5.0),
            min_width=22.0,
            min_height=22.0,
        ),
        "input": NodeStyle(
            shape="diamond",
            fill="#FFFFFF",  # bright ice
            stroke="#80B0D8",
            stroke_width=2.5,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#1A3050",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "output": NodeStyle(
            shape="diamond",
            fill="#B0C8E0",  # deep ice blue
            stroke="#7898B8",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#E0E8F0",
            padding=(5.0, 5.0),
            min_width=20.0,
            min_height=20.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#88B0D0",
            width=0.8,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#88B0D0",
            arrow_length=4.5,
            arrow_width=3.0,
            routing="straight",  # crystalline straight lines
            curvature=0.0,
            opacity=0.6,
        ),
        "back": EdgeStyle(
            color="#6080A0",
            width=0.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#6080A0",
            arrow_length=3.5,
            arrow_width=2.5,
            routing="straight",
            curvature=0.0,
            opacity=0.4,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#C0D0E0",
        stroke="#90A8C0",
        stroke_width=0.5,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=8.0,
        font_color="#4A6880",
        font_weight="bold",
        padding=10.0,
        opacity=0.25,
    ),
    graph_style=GraphStyle(background_color="#D8E0E8"),
)

# Treasure map -- pirate cartography, parchment, X marks the spot
TREASURE_MAP_THEME = Theme(
    name="treasure_map",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#C8A060",  # map marker
            stroke="#6B4226",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#3A2010",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
        "input": NodeStyle(
            shape="star",  # X marks the spot
            fill="#C82020",  # red X
            stroke="#8B1010",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#FFE0C0",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#4878A0",  # port town blue
            stroke="#305878",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=7.5,
            font_color="#D0E0F0",
            padding=(4.0, 4.0),
            min_width=20.0,
            min_height=20.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#6B4226",  # ink route
            width=1.5,
            style="dashed",  # dotted treasure route
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#6B4226",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.2,
        ),
        "back": EdgeStyle(
            color="#8B7355",
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#8B7355",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.3,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#E0C890",
        stroke="#8B7355",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=9.0,
        font_color="#4A3218",
        font_weight="bold",
        padding=12.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#E8D8B0"),
)

# Propaganda -- Soviet constructivist: bold red/black/cream, angular
PROPAGANDA_THEME = Theme(
    name="propaganda",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#CC1818",  # bold red
            stroke="#000000",
            stroke_width=3.0,
            font_family="Arial",
            font_size=9.0,
            font_color="#F0E0C0",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=22.0,
            corner_radius=0.0,  # sharp angles
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#000000",
            stroke="#CC1818",
            stroke_width=3.5,
            font_family="Arial",
            font_size=9.0,
            font_color="#F0E0C0",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=36.0,
            min_height=24.0,
            corner_radius=0.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#F0E0C0",  # cream poster paper
            stroke="#000000",
            stroke_width=2.5,
            font_family="Arial",
            font_size=9.0,
            font_color="#000000",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=22.0,
            corner_radius=0.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#000000",
            width=2.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#000000",
            arrow_length=6.0,
            arrow_width=5.0,
            routing="straight",  # bold direct lines
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#CC1818",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#CC1818",
            arrow_length=5.0,
            arrow_width=4.0,
            routing="straight",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F0E0C0",
        stroke="#000000",
        stroke_width=3.0,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=10.0,
        font_color="#CC1818",
        font_weight="bold",
        padding=12.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#E8D8B8"),
)

# Gothic -- dark cathedral stone, pointed shapes, gargoyle gray
GOTHIC_THEME = Theme(
    name="gothic",
    node_styles={
        "default": NodeStyle(
            shape="diamond",  # pointed arch evocation
            fill="#3A3840",  # dark stone
            stroke="#5A5860",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#C0B8B0",
            padding=(5.0, 5.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "input": NodeStyle(
            shape="diamond",
            fill="#4A2838",  # deep burgundy stone
            stroke="#6A4858",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#D8C0C8",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="diamond",
            fill="#2A3038",  # blue-black stone
            stroke="#4A5060",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#A0A8B0",
            padding=(5.0, 5.0),
            min_width=24.0,
            min_height=24.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#686068",  # iron fitting
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#686068",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.2,
        ),
        "back": EdgeStyle(
            color="#484048",
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#484048",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.3,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#1A181C",
        stroke="#3A3840",
        stroke_width=2.0,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=9.0,
        font_color="#706868",
        font_weight="bold",
        padding=12.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#121014"),
)

# Graffiti -- spray paint on concrete wall, neon tags
GRAFFITI_THEME = Theme(
    name="graffiti",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#E82878",  # hot pink tag
            stroke="#1A1A1A",
            stroke_width=2.5,
            font_family="Arial",
            font_size=9.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=20.0,
            corner_radius=3.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#20D0E0",  # cyan spray
            stroke="#1A1A1A",
            stroke_width=3.0,
            font_family="Arial",
            font_size=9.0,
            font_color="#1A1A1A",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=22.0,
            corner_radius=3.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#A0E828",  # lime green
            stroke="#1A1A1A",
            stroke_width=2.5,
            font_family="Arial",
            font_size=9.0,
            font_color="#1A1A1A",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=20.0,
            corner_radius=3.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#F8D030",  # yellow drip line
            width=2.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#F8D030",
            arrow_length=6.0,
            arrow_width=4.5,
            routing="bezier",
            curvature=0.25,
        ),
        "back": EdgeStyle(
            color="#FF6020",  # orange
            width=1.2,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#FF6020",
            arrow_length=5.0,
            arrow_width=4.0,
            routing="bezier",
            curvature=0.35,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#484848",
        stroke="#1A1A1A",
        stroke_width=2.5,
        stroke_dash="solid",
        corner_radius=4.0,
        font_size=10.0,
        font_color="#E0E0E0",
        font_weight="bold",
        padding=12.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#606060"),  # concrete wall
)

# Plumbing -- pipe network: copper/PVC joints, valve nodes,
# thick pipe edges, utility basement feel
PLUMBING_THEME = Theme(
    name="plumbing",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#B87333",  # copper fitting
            stroke="#8B5A2B",
            stroke_width=2.5,
            font_family="Arial",
            font_size=7.5,
            font_color="#FFE8D0",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
        "input": NodeStyle(
            shape="rectangle",  # valve/shutoff
            fill="#CC2020",  # red shutoff valve
            stroke="#991010",
            stroke_width=3.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFE0D0",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=28.0,
            min_height=22.0,
            corner_radius=2.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#E0E0E0",  # PVC white joint
            stroke="#A0A0A0",
            stroke_width=2.0,
            font_family="Arial",
            font_size=7.5,
            font_color="#3A3A3A",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#6B5040",  # copper pipe
            width=3.5,  # thicc pipe
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#6B5040",
            arrow_length=5.0,
            arrow_width=5.0,
            routing="ortho",  # right-angle pipe runs
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#808080",  # PVC drain pipe
            width=2.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#808080",
            arrow_length=4.0,
            arrow_width=4.0,
            routing="ortho",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#E8E0D0",  # concrete wall
        stroke="#A09080",
        stroke_width=2.0,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=8.5,
        font_color="#5A4A38",
        font_weight="bold",
        padding=12.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#D8D0C0"),  # basement concrete
)

# Flowchart -- classic business flowchart: diamond decisions,
# rectangle processes, clean corporate palette
FLOWCHART_THEME = Theme(
    name="flowchart",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#D6EAF8",  # process light blue
            stroke="#2980B9",
            stroke_width=1.5,
            font_family="Arial",
            font_size=8.5,
            font_color="#1A3C5E",
            padding=(6.0, 4.0),
            min_width=40.0,
            min_height=22.0,
            corner_radius=3.0,
        ),
        "input": NodeStyle(
            shape="diamond",  # decision diamond
            fill="#FDEBD0",  # decision amber
            stroke="#E67E22",
            stroke_width=2.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#7E5109",
            font_weight="bold",
            padding=(8.0, 6.0),
            min_width=36.0,
            min_height=36.0,
        ),
        "output": NodeStyle(
            shape="ellipse",  # terminator oval
            fill="#D5F5E3",  # terminal green
            stroke="#27AE60",
            stroke_width=2.0,
            font_family="Arial",
            font_size=8.5,
            font_color="#1E7B40",
            font_weight="bold",
            padding=(6.0, 4.0),
            min_width=36.0,
            min_height=22.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#2C3E50",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#2C3E50",
            arrow_length=6.0,
            arrow_width=4.0,
            routing="ortho",  # right-angle flowchart lines
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#7F8C8D",
            width=1.0,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#7F8C8D",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="ortho",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F8F9FA",
        stroke="#BDC3C7",
        stroke_width=1.5,
        stroke_dash="dashed",
        corner_radius=4.0,
        font_size=9.0,
        font_color="#5D6D7E",
        font_weight="bold",
        padding=12.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

# Choose your own adventure -- paperback book page, branching story,
# page-number nodes, narrative edges
ADVENTURE_THEME = Theme(
    name="adventure",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#FFF8E8",  # book page cream
            stroke="#8B7355",  # aged page edge
            stroke_width=1.5,
            font_family="Georgia",
            font_size=8.5,
            font_color="#2A1E10",
            padding=(6.0, 4.0),
            min_width=36.0,
            min_height=22.0,
            corner_radius=1.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#E8D0A0",  # title page gold
            stroke="#6B4226",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=9.0,
            font_color="#2A1E10",
            font_weight="bold",
            padding=(6.0, 4.0),
            min_width=40.0,
            min_height=24.0,
            corner_radius=1.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#C82020",  # THE END (red)
            stroke="#8B1010",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.5,
            font_color="#FFE0D0",
            font_weight="bold",
            padding=(6.0, 4.0),
            min_width=36.0,
            min_height=22.0,
            corner_radius=1.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#6B4226",  # ink brown
            width=1.2,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#6B4226",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.2,
        ),
        "back": EdgeStyle(
            color="#A08060",  # lighter pencil
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#A08060",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.3,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F0E8D0",
        stroke="#8B7355",
        stroke_width=1.0,
        stroke_dash="dashed",
        corner_radius=2.0,
        font_size=9.0,
        font_color="#6B4226",
        font_weight="bold",
        padding=10.0,
        opacity=0.35,
    ),
    graph_style=GraphStyle(background_color="#F8F0E0"),
)

# Aqueduct -- Roman aqueduct: stone arches, water channels,
# classical Mediterranean palette
AQUEDUCT_THEME = Theme(
    name="aqueduct",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#D8CDB8",  # travertine stone
            stroke="#9B8E78",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#3A3028",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=22.0,
            corner_radius=0.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#4878A0",  # water source blue
            stroke="#305878",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#D0E0F0",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=24.0,
            corner_radius=0.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#C8A060",  # golden sandstone fountain
            stroke="#A08040",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#3A2A10",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=22.0,
            corner_radius=0.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#5088A0",  # water channel
            width=2.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#5088A0",
            arrow_length=5.0,
            arrow_width=4.0,
            routing="ortho",  # stone channel right angles
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#9B8E78",  # dry overflow channel
            width=1.2,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#9B8E78",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="ortho",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#E8D8C0",
        stroke="#9B8E78",
        stroke_width=2.5,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=9.0,
        font_color="#5A4A38",
        font_weight="bold",
        padding=12.0,
        opacity=0.45,
    ),
    graph_style=GraphStyle(background_color="#70A8C8"),  # Mediterranean sky
)

# DNA -- genetic network: double helix colors, nucleotide base pairs
DNA_THEME = Theme(
    name="dna",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#3498DB",  # adenine blue
            stroke="#2178B8",
            stroke_width=1.5,
            font_family="Courier New",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#E74C3C",  # thymine red
            stroke="#C0392B",
            stroke_width=2.0,
            font_family="Courier New",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#2ECC71",  # guanine green
            stroke="#27AE60",
            stroke_width=1.5,
            font_family="Courier New",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#7F8C8D",  # backbone bond
            width=1.5,
            style="solid",
            arrow="none",  # chemical bonds
            routing="bezier",
            curvature=0.3,
        ),
        "back": EdgeStyle(
            color="#BDC3C7",  # hydrogen bond (weaker)
            width=0.8,
            style="dashed",
            arrow="none",
            routing="straight",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#1A2530",
        stroke="#2C3E50",
        stroke_width=1.0,
        stroke_dash="dashed",
        corner_radius=6.0,
        font_size=8.0,
        font_color="#5D6D7E",
        font_weight="bold",
        padding=10.0,
        opacity=0.35,
    ),
    graph_style=GraphStyle(background_color="#0E1A28"),
)

# Origami -- Japanese paper folds: crisp edges, pastel paper colors
ORIGAMI_THEME = Theme(
    name="origami",
    node_styles={
        "default": NodeStyle(
            shape="diamond",  # folded paper shape
            fill="#F0C8C8",  # pink washi
            stroke="#D0A0A0",
            stroke_width=1.0,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#5A3030",
            padding=(5.0, 5.0),
            min_width=24.0,
            min_height=24.0,
        ),
        "input": NodeStyle(
            shape="diamond",
            fill="#C8D8F0",  # blue washi
            stroke="#A0B8D0",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#2A3A5A",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "output": NodeStyle(
            shape="diamond",
            fill="#D8F0C8",  # green washi
            stroke="#B0D0A0",
            stroke_width=1.0,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#2A4A2A",
            padding=(5.0, 5.0),
            min_width=24.0,
            min_height=24.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#A09090",  # fold crease
            width=0.8,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#A09090",
            arrow_length=4.0,
            arrow_width=2.5,
            routing="straight",  # crisp folds
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#C0B8B0",
            width=0.5,
            style="dashed",  # valley fold
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#C0B8B0",
            arrow_length=3.5,
            arrow_width=2.0,
            routing="straight",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F8F0E8",
        stroke="#D0C8C0",
        stroke_width=0.5,
        stroke_dash="dashed",
        corner_radius=0.0,
        font_size=8.0,
        font_color="#908080",
        font_weight="normal",
        padding=8.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#FAF6F0"),
)

# Clockwork -- precision watch mechanism: brass gears, jewel bearings
CLOCKWORK_THEME = Theme(
    name="clockwork",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#C8B070",  # brass gear
            stroke="#A08840",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=7.5,
            font_color="#2A2008",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#C03038",  # ruby jewel bearing
            stroke="#901820",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#FFD0D0",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#E0E0E0",  # steel escapement
            stroke="#A0A0A0",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=7.5,
            font_color="#2A2A2A",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#8B7030",  # brass linkage
            width=1.8,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#8B7030",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="straight",  # precise mechanical
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#606060",  # steel spring
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#606060",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="straight",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#2A2418",
        stroke="#5A4830",
        stroke_width=2.0,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=8.5,
        font_color="#B0A070",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#1A1810"),
)

# Tapestry -- medieval woven threads, rich textile colors
TAPESTRY_THEME = Theme(
    name="tapestry",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#8B2252",  # madder rose thread
            stroke="#6B1842",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#F0D0D8",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=20.0,
            corner_radius=0.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#1A4878",  # woad blue thread
            stroke="#0A3060",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#C0D0E8",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=22.0,
            corner_radius=0.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#C8A030",  # gold thread
            stroke="#A08020",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#2A2008",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=20.0,
            corner_radius=0.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#4A3020",  # dark warp thread
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#4A3020",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="straight",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#7A6050",  # lighter weft
            width=0.8,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#7A6050",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="straight",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#3A2828",
        stroke="#5A4038",
        stroke_width=2.0,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=9.0,
        font_color="#C8A868",
        font_weight="bold",
        padding=12.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#2A2018"),
)

# Railway -- train track network: rail ties, junction switches,
# signal colors, infrastructure gray
RAILWAY_THEME = Theme(
    name="railway",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#E8E0D0",  # platform marker
            stroke="#4A4A4A",
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#2A2A2A",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=20.0,
            min_height=20.0,
        ),
        "input": NodeStyle(
            shape="rectangle",  # station building
            fill="#8B1A1A",  # classic station red
            stroke="#5A0A0A",
            stroke_width=2.5,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#F0D8C0",
            font_weight="bold",
            padding=(6.0, 3.0),
            min_width=34.0,
            min_height=22.0,
            corner_radius=1.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#2A7830",  # signal green (clear)
            stroke="#1A5820",
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#D0F0D0",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=20.0,
            min_height=20.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#3A3A3A",  # rail track
            width=3.0,  # heavy rail
            style="solid",
            arrow="none",  # tracks are bidirectional
            routing="bezier",
            curvature=0.15,  # gentle curves
        ),
        "back": EdgeStyle(
            color="#808080",  # siding / branch line
            width=1.5,
            style="dashed",
            arrow="none",
            routing="bezier",
            curvature=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#E0D8C8",
        stroke="#A09880",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=2.0,
        font_size=8.5,
        font_color="#4A4A4A",
        font_weight="bold",
        padding=12.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#D0C8B8"),  # gravel ballast
)

# Jungle -- tropical vines and canopy: lush greens, hanging vine
# edges, flower-bright nodes, dense humid atmosphere
JUNGLE_THEME = Theme(
    name="jungle",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#2D8B46",  # deep leaf green
            stroke="#1A6830",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=7.5,
            font_color="#E0F8E0",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#E8384F",  # hibiscus red
            stroke="#C02040",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#FFE0D8",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#F0C830",  # tropical bird yellow
            stroke="#C8A020",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=7.5,
            font_color="#3A2A08",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#3A6830",  # vine green
            width=1.8,
            style="solid",
            arrow="none",  # vines just connect
            routing="bezier",
            curvature=0.45,  # drooping vines
        ),
        "back": EdgeStyle(
            color="#5A8848",  # lighter tendril
            width=0.7,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.6,  # extra droopy
            opacity=0.5,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#1A3818",  # dense canopy shadow
        stroke="#2A5828",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=12.0,
        font_size=8.0,
        font_color="#80B868",
        font_weight="bold",
        padding=10.0,
        opacity=0.35,
    ),
    graph_style=GraphStyle(background_color="#0E2810"),  # deep jungle floor
)

# Power grid -- electrical power distribution: energized lines glow
# bright, de-energized lines are dull gray. Transformer nodes,
# substation clusters, utility infrastructure palette
POWER_GRID_THEME = Theme(
    name="power_grid",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#2A2A2A",  # switchgear housing
            stroke="#F0C020",  # warning yellow
            stroke_width=2.0,
            font_family="Courier New",
            font_size=7.5,
            font_color="#F0C020",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=20.0,
            corner_radius=1.0,
        ),
        "input": NodeStyle(
            shape="rectangle",  # generator / power source
            fill="#C82020",  # high voltage red
            stroke="#F0C020",
            stroke_width=3.0,
            font_family="Courier New",
            font_size=8.0,
            font_color="#FFE860",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=22.0,
            corner_radius=1.0,
        ),
        "output": NodeStyle(
            shape="circle",  # load / consumer
            fill="#1A1A1A",
            stroke="#606060",  # unpowered gray
            stroke_width=1.5,
            font_family="Courier New",
            font_size=7.5,
            font_color="#909090",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#F0C020",  # ENERGIZED -- bright yellow power line
            width=2.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#F0C020",
            arrow_length=5.0,
            arrow_width=4.0,
            routing="ortho",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#484848",  # DE-ENERGIZED -- dull gray, no power
            width=1.5,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#484848",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="ortho",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#1A1A1A",  # substation enclosure
        stroke="#F0C020",
        stroke_width=2.0,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=8.0,
        font_color="#F0C020",
        font_weight="bold",
        padding=12.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#0A0A0A"),
)

# Catacombs -- Parisian ossuary tunnels: bone white on deep stone,
# flickering torchlight amber, skull alcove nodes, narrow passages
CATACOMBS_THEME = Theme(
    name="catacombs",
    node_styles={
        "default": NodeStyle(
            shape="ellipse",
            fill="#C8B898",  # bone white-yellow
            stroke="#A09070",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=7.5,
            font_color="#2A2018",
            padding=(5.0, 3.0),
            min_width=26.0,
            min_height=18.0,
        ),
        "input": NodeStyle(
            shape="ellipse",
            fill="#D8A030",  # torchlit alcove amber
            stroke="#B08020",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#1A1008",
            font_weight="bold",
            padding=(5.0, 4.0),
            min_width=30.0,
            min_height=22.0,
        ),
        "output": NodeStyle(
            shape="ellipse",
            fill="#887868",  # dark stone niche
            stroke="#685848",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=7.5,
            font_color="#C0B098",
            padding=(5.0, 3.0),
            min_width=24.0,
            min_height=16.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#685848",  # narrow passage stone
            width=1.8,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#685848",
            arrow_length=4.5,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.3,
        ),
        "back": EdgeStyle(
            color="#484038",  # deeper tunnel
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#484038",
            arrow_length=3.5,
            arrow_width=2.5,
            routing="bezier",
            curvature=0.45,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#1A1610",  # chamber darkness
        stroke="#3A3028",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=6.0,
        font_size=8.0,
        font_color="#807060",
        font_weight="bold",
        padding=10.0,
        opacity=0.45,
    ),
    graph_style=GraphStyle(background_color="#100E0A"),  # deep underground
)

# Fortress -- medieval fortification: stone tower nodes, thick
# curtain wall edges, crenellated battlements, castle keep palette
FORTRESS_THEME = Theme(
    name="fortress",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#8B8070",  # gray fieldstone tower
            stroke="#5A5248",
            stroke_width=3.0,  # thick masonry
            font_family="Georgia",
            font_size=8.0,
            font_color="#E8E0D0",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=26.0,
            corner_radius=0.0,  # square towers
        ),
        "input": NodeStyle(
            shape="rectangle",  # keep / gatehouse
            fill="#5A4838",  # dark stone keep
            stroke="#3A3028",
            stroke_width=4.0,
            font_family="Georgia",
            font_size=8.5,
            font_color="#D8C8A8",
            font_weight="bold",
            padding=(6.0, 4.0),
            min_width=36.0,
            min_height=30.0,
            corner_radius=0.0,
        ),
        "output": NodeStyle(
            shape="circle",  # watchtower turret
            fill="#A09880",
            stroke="#787068",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=7.5,
            font_color="#E8E0D0",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#5A5248",  # curtain wall
            width=4.0,  # massive stone wall
            style="solid",
            arrow="none",  # walls connect both ways
            routing="straight",  # walls are straight
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#787068",  # inner bailey wall (thinner)
            width=2.0,
            style="solid",
            arrow="none",
            routing="straight",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#3A3428",  # castle courtyard
        stroke="#5A5248",
        stroke_width=3.0,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=9.0,
        font_color="#C0B898",
        font_weight="bold",
        padding=14.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#4A6838"),  # grassy field outside walls
)

# ── Code editor colorscheme themes ────────────────────────────────────

# Solarized Light -- Ethan Schoonover's famous warm light palette
SOLARIZED_LIGHT_THEME = Theme(
    name="solarized_light",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#eee8d5",  # base2
            stroke="#268bd2",  # blue
            stroke_width=1.5,
            font_family="Courier New",
            font_size=8.0,
            font_color="#657b83",  # base00
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=3.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#eee8d5",
            stroke="#dc322f",  # red
            stroke_width=2.0,
            font_family="Courier New",
            font_size=8.0,
            font_color="#cb4b16",  # orange
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=3.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#eee8d5",
            stroke="#859900",  # green
            stroke_width=1.5,
            font_family="Courier New",
            font_size=8.0,
            font_color="#586e75",  # base01
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=3.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#586e75",  # base01
            width=1.2,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#586e75",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.2,
        ),
        "back": EdgeStyle(
            color="#93a1a1",  # base1
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#93a1a1",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#fdf6e3",  # base3
        stroke="#93a1a1",
        stroke_width=1.0,
        stroke_dash="dashed",
        corner_radius=4.0,
        font_size=8.0,
        font_color="#657b83",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#fdf6e3"),
)

# Solarized Dark
SOLARIZED_DARK_THEME = Theme(
    name="solarized_dark",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#073642",  # base02
            stroke="#268bd2",  # blue
            stroke_width=1.5,
            font_family="Courier New",
            font_size=8.0,
            font_color="#839496",  # base0
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=3.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#073642",
            stroke="#dc322f",  # red
            stroke_width=2.0,
            font_family="Courier New",
            font_size=8.0,
            font_color="#cb4b16",  # orange
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=3.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#073642",
            stroke="#859900",  # green
            stroke_width=1.5,
            font_family="Courier New",
            font_size=8.0,
            font_color="#93a1a1",  # base1
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=3.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#839496",
            width=1.2,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#839496",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.2,
        ),
        "back": EdgeStyle(
            color="#586e75",
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#586e75",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#002b36",  # base03
        stroke="#586e75",
        stroke_width=1.0,
        stroke_dash="dashed",
        corner_radius=4.0,
        font_size=8.0,
        font_color="#839496",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#002b36"),
)

# Monokai -- Sublime Text classic
MONOKAI_THEME = Theme(
    name="monokai",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#3E3D32",
            stroke="#66D9EF",  # cyan
            stroke_width=1.5,
            font_family="Courier New",
            font_size=8.0,
            font_color="#F8F8F2",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=3.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#3E3D32",
            stroke="#F92672",  # pink-red
            stroke_width=2.0,
            font_family="Courier New",
            font_size=8.0,
            font_color="#FD971F",  # orange
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=3.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#3E3D32",
            stroke="#A6E22E",  # green
            stroke_width=1.5,
            font_family="Courier New",
            font_size=8.0,
            font_color="#E6DB74",  # yellow
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=3.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#F8F8F2",
            width=1.2,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#F8F8F2",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.2,
        ),
        "back": EdgeStyle(
            color="#75715E",  # comment gray
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#75715E",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#272822",
        stroke="#75715E",
        stroke_width=1.0,
        stroke_dash="dashed",
        corner_radius=4.0,
        font_size=8.0,
        font_color="#F8F8F2",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#272822"),
)

# Dracula
DRACULA_THEME = Theme(
    name="dracula",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#44475a",  # current line
            stroke="#bd93f9",  # purple
            stroke_width=1.5,
            font_family="Courier New",
            font_size=8.0,
            font_color="#f8f8f2",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=3.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#44475a",
            stroke="#ff79c6",  # pink
            stroke_width=2.0,
            font_family="Courier New",
            font_size=8.0,
            font_color="#ffb86c",  # orange
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=3.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#44475a",
            stroke="#50fa7b",  # green
            stroke_width=1.5,
            font_family="Courier New",
            font_size=8.0,
            font_color="#8be9fd",  # cyan
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=3.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#bd93f9",
            width=1.2,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#bd93f9",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.2,
        ),
        "back": EdgeStyle(
            color="#6272a4",  # comment
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#6272a4",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#282a36",
        stroke="#6272a4",
        stroke_width=1.0,
        stroke_dash="dashed",
        corner_radius=4.0,
        font_size=8.0,
        font_color="#f8f8f2",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#282a36"),
)

# Nord -- Arctic north-bluish
NORD_THEME = Theme(
    name="nord",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#3b4252",  # polar night 1
            stroke="#88c0d0",  # frost 1
            stroke_width=1.5,
            font_family="Courier New",
            font_size=8.0,
            font_color="#d8dee9",  # snow storm 0
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=3.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#3b4252",
            stroke="#bf616a",  # aurora red
            stroke_width=2.0,
            font_family="Courier New",
            font_size=8.0,
            font_color="#d08770",  # aurora orange
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=3.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#3b4252",
            stroke="#a3be8c",  # aurora green
            stroke_width=1.5,
            font_family="Courier New",
            font_size=8.0,
            font_color="#ebcb8b",  # aurora yellow
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=3.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#81a1c1",  # frost 2
            width=1.2,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#81a1c1",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.2,
        ),
        "back": EdgeStyle(
            color="#4c566a",  # polar night 3
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#4c566a",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#2e3440",
        stroke="#4c566a",
        stroke_width=1.0,
        stroke_dash="dashed",
        corner_radius=4.0,
        font_size=8.0,
        font_color="#d8dee9",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#2e3440"),
)

# Gruvbox Dark -- retro groove
GRUVBOX_THEME = Theme(
    name="gruvbox",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#3c3836",
            stroke="#83a598",  # bright blue
            stroke_width=1.5,
            font_family="Courier New",
            font_size=8.0,
            font_color="#ebdbb2",  # fg
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=3.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#3c3836",
            stroke="#fb4934",  # bright red
            stroke_width=2.0,
            font_family="Courier New",
            font_size=8.0,
            font_color="#fe8019",  # bright orange
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=3.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#3c3836",
            stroke="#b8bb26",  # bright green
            stroke_width=1.5,
            font_family="Courier New",
            font_size=8.0,
            font_color="#fabd2f",  # bright yellow
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=3.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#ebdbb2",
            width=1.2,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#ebdbb2",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.2,
        ),
        "back": EdgeStyle(
            color="#665c54",
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#665c54",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#282828",
        stroke="#665c54",
        stroke_width=1.0,
        stroke_dash="dashed",
        corner_radius=4.0,
        font_size=8.0,
        font_color="#ebdbb2",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#282828"),
)

# One Dark -- Atom editor default
ONE_DARK_THEME = Theme(
    name="one_dark",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#3b4048",
            stroke="#61afef",  # blue
            stroke_width=1.5,
            font_family="Courier New",
            font_size=8.0,
            font_color="#abb2bf",  # fg
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=3.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#3b4048",
            stroke="#e06c75",  # red
            stroke_width=2.0,
            font_family="Courier New",
            font_size=8.0,
            font_color="#c678dd",  # magenta
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=3.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#3b4048",
            stroke="#98c379",  # green
            stroke_width=1.5,
            font_family="Courier New",
            font_size=8.0,
            font_color="#e5c07b",  # yellow
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=3.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#abb2bf",
            width=1.2,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#abb2bf",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.2,
        ),
        "back": EdgeStyle(
            color="#5c6370",  # comment gray
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#5c6370",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#282c34",
        stroke="#5c6370",
        stroke_width=1.0,
        stroke_dash="dashed",
        corner_radius=4.0,
        font_size=8.0,
        font_color="#abb2bf",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#282c34"),
)

# Catppuccin Mocha -- soothing pastels on dark
CATPPUCCIN_THEME = Theme(
    name="catppuccin",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#313244",  # surface0
            stroke="#89b4fa",  # blue
            stroke_width=1.5,
            font_family="Courier New",
            font_size=8.0,
            font_color="#cdd6f4",  # text
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=4.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#313244",
            stroke="#f38ba8",  # red
            stroke_width=2.0,
            font_family="Courier New",
            font_size=8.0,
            font_color="#fab387",  # peach
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=4.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#313244",
            stroke="#a6e3a1",  # green
            stroke_width=1.5,
            font_family="Courier New",
            font_size=8.0,
            font_color="#94e2d5",  # teal
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=4.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#cba6f7",  # mauve
            width=1.2,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#cba6f7",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.2,
        ),
        "back": EdgeStyle(
            color="#585b70",  # surface2
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#585b70",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#1e1e2e",  # base
        stroke="#585b70",
        stroke_width=1.0,
        stroke_dash="dashed",
        corner_radius=6.0,
        font_size=8.0,
        font_color="#cdd6f4",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#1e1e2e"),
)

# Tokyo Night
TOKYO_NIGHT_THEME = Theme(
    name="tokyo_night",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#24283b",
            stroke="#7aa2f7",  # blue
            stroke_width=1.5,
            font_family="Courier New",
            font_size=8.0,
            font_color="#c0caf5",  # fg
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=3.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#24283b",
            stroke="#f7768e",  # red
            stroke_width=2.0,
            font_family="Courier New",
            font_size=8.0,
            font_color="#bb9af7",  # magenta
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=3.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#24283b",
            stroke="#9ece6a",  # green
            stroke_width=1.5,
            font_family="Courier New",
            font_size=8.0,
            font_color="#e0af68",  # yellow
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=3.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#7dcfff",  # cyan
            width=1.2,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#7dcfff",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.2,
        ),
        "back": EdgeStyle(
            color="#565f89",  # comment
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#565f89",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#1a1b26",
        stroke="#565f89",
        stroke_width=1.0,
        stroke_dash="dashed",
        corner_radius=4.0,
        font_size=8.0,
        font_color="#c0caf5",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#1a1b26"),
)

# Lilypad -- pond surface: green pads floating on dark water,
# delicate tendrils connecting beneath
LILYPAD_THEME = Theme(
    name="lilypad",
    node_styles={
        "default": NodeStyle(
            shape="ellipse",
            fill="#5A9E4B",  # lilypad green
            stroke="#3A7830",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=7.5,
            font_color="#E0F8D8",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=20.0,
        ),
        "input": NodeStyle(
            shape="ellipse",
            fill="#E898B0",  # lotus pink
            stroke="#C07088",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#3A1820",
            font_weight="bold",
            padding=(5.0, 4.0),
            min_width=32.0,
            min_height=24.0,
        ),
        "output": NodeStyle(
            shape="ellipse",
            fill="#78B868",  # young pad bright green
            stroke="#58A048",
            stroke_width=1.0,
            font_family="Georgia",
            font_size=7.0,
            font_color="#1A3A10",
            padding=(4.0, 3.0),
            min_width=24.0,
            min_height=16.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#4A8838",  # underwater tendril
            width=0.8,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.4,
            opacity=0.5,
        ),
        "back": EdgeStyle(
            color="#386828",
            width=0.5,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.55,
            opacity=0.3,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#1A3028",
        stroke="#2A5038",
        stroke_width=0.5,
        stroke_dash="solid",
        corner_radius=16.0,
        font_size=7.5,
        font_color="#60A050",
        font_weight="normal",
        padding=10.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#0E2018"),  # dark pond water
)

# Garden -- flower garden: bright blooms, green stems, earthy beds
GARDEN_THEME = Theme(
    name="garden",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#D84888",  # peony pink
            stroke="#B03068",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=7.5,
            font_color="#FFE0E8",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#E8C030",  # sunflower gold
            stroke="#C0A020",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#3A2A08",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#7848B8",  # lavender purple
            stroke="#5830A0",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=7.5,
            font_color="#E0D0F8",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#408830",  # garden stem
            width=1.5,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.3,
        ),
        "back": EdgeStyle(
            color="#60A848",  # lighter vine
            width=0.8,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.45,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#D8C8A0",  # mulch bed
        stroke="#A09060",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=10.0,
        font_size=8.5,
        font_color="#5A4A28",
        font_weight="bold",
        padding=10.0,
        opacity=0.35,
    ),
    graph_style=GraphStyle(background_color="#3A6828"),  # lawn green
)

# River -- waterways connecting pools: blue currents, sandy banks
RIVER_THEME = Theme(
    name="river",
    node_styles={
        "default": NodeStyle(
            shape="ellipse",
            fill="#3880B0",  # deep pool blue
            stroke="#286898",
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#D0E8F8",
            padding=(5.0, 4.0),
            min_width=30.0,
            min_height=22.0,
        ),
        "input": NodeStyle(
            shape="ellipse",
            fill="#2068A0",  # mountain spring source
            stroke="#104878",
            stroke_width=2.5,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#C0D8F0",
            font_weight="bold",
            padding=(6.0, 4.0),
            min_width=34.0,
            min_height=26.0,
        ),
        "output": NodeStyle(
            shape="ellipse",
            fill="#68A8C8",  # shallow estuary
            stroke="#4890B0",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#1A3850",
            padding=(5.0, 3.0),
            min_width=28.0,
            min_height=18.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#4898C0",  # flowing current
            width=2.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#4898C0",
            arrow_length=5.0,
            arrow_width=4.0,
            routing="bezier",
            curvature=0.35,  # meandering
        ),
        "back": EdgeStyle(
            color="#78B0D0",  # tributary
            width=1.2,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#78B0D0",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.45,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#C8B888",  # sandy bank
        stroke="#A09868",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=12.0,
        font_size=8.5,
        font_color="#5A4A28",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#88A878"),  # riparian green
)

# Jetstream -- high-altitude wind currents: streaky fast edges,
# pressure system nodes, stratospheric palette
JETSTREAM_THEME = Theme(
    name="jetstream",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#E0E8F0",  # high-pressure white
            stroke="#A0B8D0",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#2A4060",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#C83030",  # warm front red
            stroke="#A01818",
            stroke_width=2.5,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FFE0D0",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#2868B0",  # cold front blue
            stroke="#184890",
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#C0D8F0",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#6090C0",  # jet stream blue
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#6090C0",
            arrow_length=6.0,  # elongated wind arrow
            arrow_width=3.0,
            routing="bezier",
            curvature=0.3,
            opacity=0.6,
        ),
        "back": EdgeStyle(
            color="#90B0D0",
            width=0.7,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#90B0D0",
            arrow_length=5.0,
            arrow_width=2.5,
            routing="bezier",
            curvature=0.4,
            opacity=0.4,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#D0D8E0",
        stroke="#90A0B0",
        stroke_width=0.8,
        stroke_dash="dashed",
        corner_radius=10.0,
        font_size=8.0,
        font_color="#4A6080",
        font_weight="bold",
        padding=10.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#E8ECF0"),
)

# Weather -- TV weather map: isobars, fronts, radar palette
WEATHER_THEME = Theme(
    name="weather",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#F8F8F8",  # station marker
            stroke="#2A2A2A",
            stroke_width=1.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#1A1A1A",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=20.0,
            min_height=20.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#E02020",  # high pressure H
            stroke="#B01010",
            stroke_width=2.5,
            font_family="Arial",
            font_size=9.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#2060D0",  # low pressure L
            stroke="#1040A0",
            stroke_width=2.5,
            font_family="Arial",
            font_size=9.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=26.0,
            min_height=26.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#E02020",  # warm front (red with bumps)
            width=2.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#E02020",
            arrow_length=5.0,
            arrow_width=4.0,
            routing="bezier",
            curvature=0.25,
        ),
        "back": EdgeStyle(
            color="#2060D0",  # cold front (blue with triangles)
            width=2.0,
            style="solid",
            arrow="vee",
            arrow_fill="filled",
            arrow_color="#2060D0",
            arrow_length=5.0,
            arrow_width=5.0,
            routing="bezier",
            curvature=0.3,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#90D070",  # radar green zone
        stroke="#60A040",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=8.0,
        font_size=8.5,
        font_color="#2A4A18",
        font_weight="bold",
        padding=10.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#C8D8C0"),  # map green
)

# NYT -- New York Times: elegant serif, restrained gray palette,
# thin rules, old-paper dignity
NYT_THEME = Theme(
    name="nyt",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#FFFFFF",
            stroke="#333333",
            stroke_width=1.0,
            font_family="Georgia",
            font_size=8.5,
            font_color="#333333",
            padding=(6.0, 3.0),
            min_width=38.0,
            min_height=20.0,
            corner_radius=0.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#FFFFFF",
            stroke="#333333",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=9.0,
            font_color="#121212",
            font_weight="bold",
            padding=(6.0, 3.0),
            min_width=40.0,
            min_height=22.0,
            corner_radius=0.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#F7F7F5",
            stroke="#CCCCCC",
            stroke_width=1.0,
            font_family="Georgia",
            font_size=8.5,
            font_color="#666666",
            padding=(6.0, 3.0),
            min_width=38.0,
            min_height=20.0,
            corner_radius=0.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#888888",
            width=0.8,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#888888",
            arrow_length=5.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.15,
        ),
        "back": EdgeStyle(
            color="#CCCCCC",
            width=0.5,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#CCCCCC",
            arrow_length=4.0,
            arrow_width=2.5,
            routing="bezier",
            curvature=0.2,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F7F7F5",
        stroke="#DDDDDD",
        stroke_width=0.5,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=9.0,
        font_color="#333333",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

# Economist -- The Economist: distinctive red/navy, clean sans-serif
ECONOMIST_THEME = Theme(
    name="economist",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#FFFFFF",
            stroke="#0D47A1",  # Economist navy
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#1A1A1A",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=20.0,
            corner_radius=0.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#D32F2F",  # Economist red
            stroke="#B71C1C",
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=36.0,
            min_height=22.0,
            corner_radius=0.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#E8E8E8",
            stroke="#0D47A1",
            stroke_width=1.0,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#333333",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=20.0,
            corner_radius=0.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#0D47A1",
            width=1.2,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#0D47A1",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.15,
        ),
        "back": EdgeStyle(
            color="#90A4AE",
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#90A4AE",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.2,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F5F5F5",
        stroke="#BDBDBD",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=8.5,
        font_color="#0D47A1",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

# FT -- Financial Times: salmon pink, dark text, authoritative
FT_THEME = Theme(
    name="ft",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#FFF1E5",  # FT pink
            stroke="#333333",
            stroke_width=1.0,
            font_family="Georgia",
            font_size=8.5,
            font_color="#333333",
            padding=(6.0, 3.0),
            min_width=36.0,
            min_height=20.0,
            corner_radius=0.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#0D7680",  # FT teal
            stroke="#0A5C64",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.5,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(6.0, 3.0),
            min_width=38.0,
            min_height=22.0,
            corner_radius=0.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#F2DFCE",  # deeper salmon
            stroke="#333333",
            stroke_width=1.0,
            font_family="Georgia",
            font_size=8.5,
            font_color="#666666",
            padding=(6.0, 3.0),
            min_width=36.0,
            min_height=20.0,
            corner_radius=0.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#333333",
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#333333",
            arrow_length=5.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.15,
        ),
        "back": EdgeStyle(
            color="#999999",
            width=0.6,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#999999",
            arrow_length=4.0,
            arrow_width=2.5,
            routing="bezier",
            curvature=0.2,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#FFF1E5",
        stroke="#CCBBAA",
        stroke_width=0.8,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=9.0,
        font_color="#333333",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#FFF1E5"),
)

# Canyon -- network of desert canyons: red sandstone walls,
# dry riverbeds, mesa plateaus
CANYON_THEME = Theme(
    name="canyon",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#C8785A",  # red sandstone mesa
            stroke="#A05838",
            stroke_width=2.5,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FFE8D0",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=22.0,
            corner_radius=2.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#D89868",  # sunlit cliff face
            stroke="#B87848",
            stroke_width=3.0,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#3A1A08",
            font_weight="bold",
            padding=(6.0, 4.0),
            min_width=34.0,
            min_height=24.0,
            corner_radius=2.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#A06040",  # shadowed alcove
            stroke="#784830",
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#E0C8A8",
            padding=(5.0, 3.0),
            min_width=28.0,
            min_height=20.0,
            corner_radius=2.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#887060",  # dry canyon floor
            width=2.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#887060",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.3,
        ),
        "back": EdgeStyle(
            color="#A89880",  # sandy wash
            width=1.2,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#A89880",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.4,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#B89070",
        stroke="#987050",
        stroke_width=2.0,
        stroke_dash="solid",
        corner_radius=4.0,
        font_size=8.5,
        font_color="#3A2010",
        font_weight="bold",
        padding=12.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#E8C8A0"),  # desert sand
)

# Pacman -- arcade maze: dark blue walls, pellet dots,
# ghost colors, black maze background
PACMAN_THEME = Theme(
    name="pacman",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#FFFF00",  # pac-dot yellow
            stroke="#E0E000",
            stroke_width=1.0,
            font_family="Courier New",
            font_size=7.0,
            font_color="#000000",
            font_weight="bold",
            padding=(3.0, 3.0),
            min_width=14.0,  # small pellet
            min_height=14.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#FFFF00",  # pacman himself
            stroke="#E0C000",
            stroke_width=2.5,
            font_family="Courier New",
            font_size=8.0,
            font_color="#000000",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#FF0000",  # Blinky ghost red
            stroke="#CC0000",
            stroke_width=2.0,
            font_family="Courier New",
            font_size=7.5,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#2121DE",  # maze wall blue
            width=3.0,
            style="solid",
            arrow="none",  # maze corridors
            routing="ortho",  # right-angle maze
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#FFB8FF",  # frightened ghost pink
            width=1.5,
            style="dashed",
            arrow="none",
            routing="ortho",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#000000",
        stroke="#2121DE",
        stroke_width=2.5,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=8.0,
        font_color="#FFFF00",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#000000"),
)

# Fracture -- pressure cracks in something about to shatter:
# glowing stress lines on dark surface, hot fissure edges
FRACTURE_THEME = Theme(
    name="fracture",
    node_styles={
        "default": NodeStyle(
            shape="diamond",
            fill="#1A1A1A",  # dark stressed material
            stroke="#FF4020",  # glowing crack edge
            stroke_width=2.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#FF8060",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=24.0,
            min_height=24.0,
        ),
        "input": NodeStyle(
            shape="diamond",
            fill="#2A0808",  # critical stress point
            stroke="#FF2000",
            stroke_width=3.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFD0A0",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="diamond",
            fill="#1A1A1A",
            stroke="#C06030",  # cooling crack
            stroke_width=2.0,
            font_family="Arial",
            font_size=7.5,
            font_color="#D09060",
            padding=(5.0, 5.0),
            min_width=22.0,
            min_height=22.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#FF3018",  # active fracture line
            width=1.5,
            style="solid",
            arrow="none",
            routing="straight",  # cracks are jagged-straight
            curvature=0.0,
            opacity=0.85,
        ),
        "back": EdgeStyle(
            color="#803020",  # hairline micro-crack
            width=0.6,
            style="solid",
            arrow="none",
            routing="straight",
            curvature=0.0,
            opacity=0.5,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#0A0808",
        stroke="#601810",
        stroke_width=2.0,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=8.0,
        font_color="#C04020",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#0A0A0A"),
)

# ── School & sports colorschemes ─────────────────────────────────────

# Yale -- blue and white
YALE_THEME = Theme(
    name="yale",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#00356B",
            stroke="#002B5B",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=2.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#FFFFFF",
            stroke="#00356B",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#00356B",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=22.0,
            corner_radius=2.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#00356B",
            stroke="#4A90D0",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#C0D8F0",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=2.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#00356B",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#00356B",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.2,
        ),
        "back": EdgeStyle(
            color="#8090A0",
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#8090A0",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#E8EEF4",
        stroke="#00356B",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=3.0,
        font_size=8.5,
        font_color="#00356B",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#F8F8F8"),
)

# Harvard -- crimson and black
HARVARD_THEME = Theme(
    name="harvard",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#A51C30",
            stroke="#8A1728",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=2.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#1E1E1E",
            stroke="#A51C30",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#A51C30",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=22.0,
            corner_radius=2.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#A51C30",
            stroke="#D04050",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#F0D0D0",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=2.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#A51C30",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#A51C30",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.2,
        ),
        "back": EdgeStyle(
            color="#808080",
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#808080",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F4E8EA",
        stroke="#A51C30",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=3.0,
        font_size=8.5,
        font_color="#A51C30",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

# Princeton -- orange and black
PRINCETON_THEME = Theme(
    name="princeton",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#E87722",
            stroke="#C86018",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#000000",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=2.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#000000",
            stroke="#E87722",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#E87722",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=22.0,
            corner_radius=2.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#E87722",
            stroke="#F0A050",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#1A1A1A",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=2.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#000000",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#000000",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.2,
        ),
        "back": EdgeStyle(
            color="#E87722",
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#E87722",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#FFF0E0",
        stroke="#E87722",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=3.0,
        font_size=8.5,
        font_color="#000000",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

# Lakers -- purple and gold
LAKERS_THEME = Theme(
    name="lakers",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#552583",
            stroke="#3A1060",
            stroke_width=2.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#FDB927",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#FDB927",
            stroke="#D89E20",
            stroke_width=2.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#552583",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#552583",
            stroke="#7848A8",
            stroke_width=1.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#FDB927",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#FDB927",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#FDB927",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.2,
        ),
        "back": EdgeStyle(
            color="#7848A8",
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#7848A8",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#3A1860",
        stroke="#552583",
        stroke_width=2.0,
        stroke_dash="solid",
        corner_radius=6.0,
        font_size=8.5,
        font_color="#FDB927",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#1A0A30"),
)

# Yankees -- navy pinstripe and white
YANKEES_THEME = Theme(
    name="yankees",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#1C2841",
            stroke="#0C1830",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=1.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#FFFFFF",
            stroke="#1C2841",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#1C2841",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=22.0,
            corner_radius=1.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#1C2841",
            stroke="#4A6088",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#C0D0E0",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=1.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#C0C8D0",
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#C0C8D0",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="straight",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#4A6088",
            width=0.6,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#4A6088",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="straight",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#E8ECF0",
        stroke="#1C2841",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=8.5,
        font_color="#1C2841",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#F8F8F8"),
)

# Celtics -- green and white (parquet)
CELTICS_THEME = Theme(
    name="celtics",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#007A33",
            stroke="#005A24",
            stroke_width=2.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#FFFFFF",
            stroke="#007A33",
            stroke_width=2.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#007A33",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#BA9653",
            stroke="#9A7843",
            stroke_width=1.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#007A33",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#007A33",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.2,
        ),
        "back": EdgeStyle(
            color="#BA9653",
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#BA9653",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#E0F0E8",
        stroke="#007A33",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=4.0,
        font_size=8.5,
        font_color="#007A33",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#F0E8D8"),  # parquet floor
)

# Ferrari -- racing red, Italian motorsport
FERRARI_THEME = Theme(
    name="ferrari",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#DC0000",
            stroke="#A80000",
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#FFD700",
            stroke="#D0B000",
            stroke_width=2.5,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#1A1A1A",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#1A1A1A",
            stroke="#DC0000",
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#DC0000",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#DC0000",
            width=2.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#DC0000",
            arrow_length=5.0,
            arrow_width=4.0,
            routing="bezier",
            curvature=0.15,
        ),
        "back": EdgeStyle(
            color="#808080",
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#808080",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.2,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#2A0A0A",
        stroke="#DC0000",
        stroke_width=2.0,
        stroke_dash="solid",
        corner_radius=4.0,
        font_size=8.5,
        font_color="#FFD700",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#1A1A1A"),
)

# Seaborn -- muted pastels, white grid, statistical plotting feel
SEABORN_THEME = Theme(
    name="seaborn",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#4C72B0",
            stroke="#3A5A90",
            stroke_width=1.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFFFFF",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#DD8452",
            stroke="#C06A38",
            stroke_width=2.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#55A868",
            stroke="#3A8848",
            stroke_width=1.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFFFFF",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#4C72B0",
            width=1.2,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#4C72B0",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.2,
            opacity=0.7,
        ),
        "back": EdgeStyle(
            color="#CCCCCC",
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#CCCCCC",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#EAEAF2",
        stroke="#CCCCCC",
        stroke_width=0.5,
        stroke_dash="solid",
        corner_radius=4.0,
        font_size=8.5,
        font_color="#333333",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#EAEAF2"),
)

# Matplotlib -- classic matplotlib defaults: tab10 colors, white bg
MATPLOTLIB_THEME = Theme(
    name="matplotlib",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#1F77B4",
            stroke="#155A8A",
            stroke_width=1.5,
            font_family="DejaVu Sans",
            font_size=8.0,
            font_color="#FFFFFF",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#FF7F0E",
            stroke="#D06A0A",
            stroke_width=2.0,
            font_family="DejaVu Sans",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#2CA02C",
            stroke="#1A801A",
            stroke_width=1.5,
            font_family="DejaVu Sans",
            font_size=8.0,
            font_color="#FFFFFF",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#1F77B4",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#1F77B4",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.2,
        ),
        "back": EdgeStyle(
            color="#BCBD22",
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#BCBD22",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F0F0F0",
        stroke="#CCCCCC",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=8.5,
        font_color="#333333",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

# ggplot -- R's ggplot2: gray panel background, subtle gridlines feel
GGPLOT_THEME = Theme(
    name="ggplot",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#F8766D",
            stroke="#D85A50",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FFFFFF",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#00BA38",
            stroke="#009828",
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#619CFF",
            stroke="#4880D8",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FFFFFF",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#333333",
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#333333",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.2,
        ),
        "back": EdgeStyle(
            color="#999999",
            width=0.6,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#999999",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#EBEBEB",
        stroke="#FFFFFF",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=8.5,
        font_color="#333333",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#EBEBEB"),
)

# Apple -- clean Cupertino: SF Pro feel, space gray, accent blue
APPLE_THEME = Theme(
    name="apple",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#FFFFFF",
            stroke="#D1D1D6",
            stroke_width=1.0,
            font_family="Helvetica",
            font_size=8.5,
            font_color="#1D1D1F",
            padding=(6.0, 4.0),
            min_width=36.0,
            min_height=22.0,
            corner_radius=8.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#007AFF",
            stroke="#0066D6",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=8.5,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(6.0, 4.0),
            min_width=38.0,
            min_height=24.0,
            corner_radius=8.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#F5F5F7",
            stroke="#D1D1D6",
            stroke_width=1.0,
            font_family="Helvetica",
            font_size=8.5,
            font_color="#86868B",
            padding=(6.0, 4.0),
            min_width=36.0,
            min_height=22.0,
            corner_radius=8.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#86868B",
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#86868B",
            arrow_length=4.5,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.15,
        ),
        "back": EdgeStyle(
            color="#D1D1D6",
            width=0.6,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#D1D1D6",
            arrow_length=4.0,
            arrow_width=2.5,
            routing="bezier",
            curvature=0.2,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F5F5F7",
        stroke="#D1D1D6",
        stroke_width=0.5,
        stroke_dash="solid",
        corner_radius=12.0,
        font_size=8.5,
        font_color="#1D1D1F",
        font_weight="bold",
        padding=12.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

# Google Material -- Material Design: shadow cards, vibrant primaries
MATERIAL_THEME = Theme(
    name="material",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#FFFFFF",
            stroke="#E0E0E0",
            stroke_width=1.0,
            font_family="Arial",
            font_size=8.5,
            font_color="#212121",
            padding=(6.0, 4.0),
            min_width=36.0,
            min_height=22.0,
            corner_radius=4.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#4285F4",
            stroke="#3367D6",
            stroke_width=1.5,
            font_family="Arial",
            font_size=8.5,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(6.0, 4.0),
            min_width=38.0,
            min_height=24.0,
            corner_radius=4.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#34A853",
            stroke="#2A8A44",
            stroke_width=1.5,
            font_family="Arial",
            font_size=8.5,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(6.0, 4.0),
            min_width=36.0,
            min_height=22.0,
            corner_radius=4.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#757575",
            width=1.2,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#757575",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.15,
        ),
        "back": EdgeStyle(
            color="#BDBDBD",
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#BDBDBD",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.2,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#FAFAFA",
        stroke="#E0E0E0",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=8.0,
        font_size=8.5,
        font_color="#212121",
        font_weight="bold",
        padding=12.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#FAFAFA"),
)

# Spotify -- dark green/black, vibrant green accent
SPOTIFY_THEME = Theme(
    name="spotify",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#282828",
            stroke="#1DB954",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FFFFFF",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=6.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#1DB954",
            stroke="#18A349",
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#000000",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=22.0,
            corner_radius=6.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#181818",
            stroke="#535353",
            stroke_width=1.0,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#B3B3B3",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=6.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#1DB954",
            width=1.2,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#1DB954",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.2,
        ),
        "back": EdgeStyle(
            color="#535353",
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#535353",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#181818",
        stroke="#282828",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=8.0,
        font_size=8.0,
        font_color="#1DB954",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#121212"),
)

# Slack -- purple workspace, vibrant accents
SLACK_THEME = Theme(
    name="slack",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#FFFFFF",
            stroke="#E8E8E8",
            stroke_width=1.0,
            font_family="Arial",
            font_size=8.5,
            font_color="#1D1C1D",
            padding=(6.0, 4.0),
            min_width=36.0,
            min_height=22.0,
            corner_radius=6.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#4A154B",
            stroke="#3A0A3B",
            stroke_width=2.0,
            font_family="Arial",
            font_size=8.5,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(6.0, 4.0),
            min_width=38.0,
            min_height=24.0,
            corner_radius=6.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#36C5F0",
            stroke="#28A8D0",
            stroke_width=1.5,
            font_family="Arial",
            font_size=8.5,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(6.0, 4.0),
            min_width=36.0,
            min_height=22.0,
            corner_radius=6.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#4A154B",
            width=1.2,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#4A154B",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.15,
        ),
        "back": EdgeStyle(
            color="#E01E5A",
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#E01E5A",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.2,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F8F8F8",
        stroke="#DDDDDD",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=8.0,
        font_size=8.5,
        font_color="#4A154B",
        font_weight="bold",
        padding=12.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

# Coca-Cola -- classic red and white, Spencerian script energy
COLA_THEME = Theme(
    name="cola",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#F40009",
            stroke="#C00008",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#FFFFFF",
            stroke="#F40009",
            stroke_width=3.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#F40009",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#F40009",
            stroke="#FF4040",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#FFD0D0",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#FFFFFF",
            width=2.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#FFFFFF",
            arrow_length=5.0,
            arrow_width=4.0,
            routing="bezier",
            curvature=0.2,
        ),
        "back": EdgeStyle(
            color="#F40009",
            width=1.0,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#F40009",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#D00008",
        stroke="#F40009",
        stroke_width=2.0,
        stroke_dash="solid",
        corner_radius=8.0,
        font_size=9.0,
        font_color="#FFFFFF",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#F40009"),
)

# ── Fun & whimsy themes ───────────────────────────────────────────────

ZEN_THEME = Theme(
    name="zen",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#C8C0B0",
            stroke="#A09888",
            stroke_width=1.0,
            font_family="Georgia",
            font_size=7.0,
            font_color="#5A5048",
            padding=(4.0, 4.0),
            min_width=18.0,
            min_height=18.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#8B8478",
            stroke="#6B6458",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=7.5,
            font_color="#E8E0D8",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=24.0,
            min_height=24.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#D8D0C4",
            stroke="#B0A898",
            stroke_width=0.8,
            font_family="Georgia",
            font_size=7.0,
            font_color="#6A6258",
            padding=(3.0, 3.0),
            min_width=14.0,
            min_height=14.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#A09888",
            width=0.5,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.2,
            opacity=0.4,
        ),
        "back": EdgeStyle(
            color="#C0B8A8",
            width=0.3,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.3,
            opacity=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#E8E0D0",
        stroke="#C0B8A8",
        stroke_width=0.3,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=7.0,
        font_color="#8A8278",
        font_weight="normal",
        padding=8.0,
        opacity=0.2,
    ),
    graph_style=GraphStyle(background_color="#E8E0D0"),
)

HONEYCOMB_THEME = Theme(
    name="honeycomb",
    node_styles={
        "default": NodeStyle(
            shape="hexagon",
            fill="#F0C030",
            stroke="#D0A020",
            stroke_width=2.0,
            font_family="Arial",
            font_size=7.5,
            font_color="#3A2A08",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "input": NodeStyle(
            shape="hexagon",
            fill="#E8A010",
            stroke="#C88808",
            stroke_width=2.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#2A1A00",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="hexagon",
            fill="#F8D860",
            stroke="#E0C040",
            stroke_width=1.5,
            font_family="Arial",
            font_size=7.5,
            font_color="#4A3808",
            padding=(5.0, 5.0),
            min_width=24.0,
            min_height=24.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#8B6D18",
            width=2.0,
            style="solid",
            arrow="none",
            routing="straight",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#C0A040",
            width=1.0,
            style="solid",
            arrow="none",
            routing="straight",
            curvature=0.0,
            opacity=0.5,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F8E898",
        stroke="#D0A828",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=8.0,
        font_color="#5A4010",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#F8E8B0"),
)

CAMPFIRE_THEME = Theme(
    name="campfire",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#E07020",
            stroke="#C05818",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#FFE8C0",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#F0A830",
            stroke="#D09020",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#2A1808",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#C04818",
            stroke="#A03010",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=7.5,
            font_color="#F0D0A0",
            padding=(4.0, 4.0),
            min_width=20.0,
            min_height=20.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#D08028",
            width=1.5,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.3,
            opacity=0.7,
        ),
        "back": EdgeStyle(
            color="#804020",
            width=0.8,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.4,
            opacity=0.4,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#1A1208",
        stroke="#3A2818",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=8.0,
        font_size=8.0,
        font_color="#C09040",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#0E0A06"),
)

RUST_THEME = Theme(
    name="rust",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#8B4513",
            stroke="#6B3010",
            stroke_width=2.0,
            font_family="Courier New",
            font_size=8.0,
            font_color="#E0C0A0",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=20.0,
            corner_radius=0.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#B05020",
            stroke="#903810",
            stroke_width=2.5,
            font_family="Courier New",
            font_size=8.0,
            font_color="#FFE0C0",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=22.0,
            corner_radius=0.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#5A3A28",
            stroke="#3A2818",
            stroke_width=1.5,
            font_family="Courier New",
            font_size=8.0,
            font_color="#C0A080",
            padding=(5.0, 3.0),
            min_width=28.0,
            min_height=18.0,
            corner_radius=0.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#A06030",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#A06030",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="straight",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#685040",
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#685040",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="straight",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#3A2818",
        stroke="#5A4030",
        stroke_width=2.0,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=8.5,
        font_color="#C09060",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#2A1E14"),
)

SYNTHWAVE_THEME = Theme(
    name="synthwave",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#1A0830",
            stroke="#FF6EC7",
            stroke_width=2.0,
            font_family="Courier New",
            font_size=8.0,
            font_color="#FF6EC7",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=18.0,
            corner_radius=2.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#1A0830",
            stroke="#FCEE09",
            stroke_width=2.5,
            font_family="Courier New",
            font_size=8.0,
            font_color="#FCEE09",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=2.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#1A0830",
            stroke="#00B4D8",
            stroke_width=2.0,
            font_family="Courier New",
            font_size=8.0,
            font_color="#00B4D8",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=18.0,
            corner_radius=2.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#FF6EC7",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#FF6EC7",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.2,
            opacity=0.7,
        ),
        "back": EdgeStyle(
            color="#7B2FBE",
            width=0.8,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#7B2FBE",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.3,
            opacity=0.5,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#0D0420",
        stroke="#3A1860",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=2.0,
        font_size=8.0,
        font_color="#FF6EC7",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#0D0420"),
)

PRIDE_THEME = Theme(
    name="pride",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#E40303",
            stroke="#C00000",
            stroke_width=1.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#FF8C00",
            stroke="#D87400",
            stroke_width=2.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#004DFF",
            stroke="#0038C0",
            stroke_width=1.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#750787",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#750787",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.25,
        ),
        "back": EdgeStyle(
            color="#008026",
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#008026",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.3,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#FFED00",
        stroke="#D0C000",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=8.0,
        font_size=8.5,
        font_color="#333333",
        font_weight="bold",
        padding=10.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

RETRO_DINER_THEME = Theme(
    name="retro_diner",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#E0E0E0",
            stroke="#C0C0C0",
            stroke_width=2.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#1A1A1A",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#CC1818",
            stroke="#A01010",
            stroke_width=2.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#18A8B8",
            stroke="#108898",
            stroke_width=2.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#C0C0C0",
            width=2.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#C0C0C0",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.15,
        ),
        "back": EdgeStyle(
            color="#E8D0B0",
            width=1.0,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#E8D0B0",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.2,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#1A1A1A",
        stroke="#C0C0C0",
        stroke_width=2.0,
        stroke_dash="solid",
        corner_radius=6.0,
        font_size=9.0,
        font_color="#CC1818",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#F0E8D8"),
)

TAROT_THEME = Theme(
    name="tarot",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#1A0A30",
            stroke="#C8A030",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#C8A030",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=22.0,
            corner_radius=3.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#4A1868",
            stroke="#E8C040",
            stroke_width=3.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#F0D860",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=24.0,
            corner_radius=3.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#0A0820",
            stroke="#9080A0",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#B0A0C0",
            font_style="italic",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=22.0,
            corner_radius=3.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#C8A030",
            width=1.2,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#C8A030",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.3,
        ),
        "back": EdgeStyle(
            color="#605070",
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#605070",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.35,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#0A0820",
        stroke="#C8A030",
        stroke_width=2.0,
        stroke_dash="solid",
        corner_radius=4.0,
        font_size=9.0,
        font_color="#C8A030",
        font_weight="bold",
        padding=12.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#0A0618"),
)

BOB_ROSS_THEME = Theme(
    name="bob_ross",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#2D5A27",
            stroke="#1A3A18",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=7.5,
            font_color="#E0F0D8",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#8B4513",
            stroke="#6B3010",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#F0E0C0",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#4682B4",
            stroke="#326898",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=7.5,
            font_color="#D8E8F0",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#6B8E4E",
            width=1.5,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.35,
        ),
        "back": EdgeStyle(
            color="#A0B888",
            width=0.8,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.45,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#87CEEB",
        stroke="#6AAEC8",
        stroke_width=0.5,
        stroke_dash="solid",
        corner_radius=10.0,
        font_size=8.0,
        font_color="#2A4A68",
        font_weight="normal",
        padding=10.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#87CEEB"),
)

MINECRAFT_THEME = Theme(
    name="minecraft",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#8B8B8B",
            stroke="#6D6D6D",
            stroke_width=2.0,
            font_family="Courier New",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
            corner_radius=0.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#7B5B2D",
            stroke="#5A4020",
            stroke_width=2.5,
            font_family="Courier New",
            font_size=8.0,
            font_color="#C8A870",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=26.0,
            min_height=26.0,
            corner_radius=0.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#5A8B32",
            stroke="#3A6B18",
            stroke_width=2.0,
            font_family="Courier New",
            font_size=8.0,
            font_color="#D8F0B8",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
            corner_radius=0.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#4A4A4A",
            width=2.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#4A4A4A",
            arrow_length=5.0,
            arrow_width=4.0,
            routing="ortho",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#7B5B2D",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#7B5B2D",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="ortho",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#3A5A1A",
        stroke="#2A4A0A",
        stroke_width=2.0,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=8.0,
        font_color="#B8D890",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#78B0E0"),
)

TETRIS_THEME = Theme(
    name="tetris",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#00F0F0",
            stroke="#00C8C8",
            stroke_width=2.0,
            font_family="Courier New",
            font_size=8.0,
            font_color="#000000",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
            corner_radius=0.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#F0F000",
            stroke="#C8C800",
            stroke_width=2.5,
            font_family="Courier New",
            font_size=8.0,
            font_color="#000000",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
            corner_radius=0.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#A000F0",
            stroke="#8000C8",
            stroke_width=2.0,
            font_family="Courier New",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
            corner_radius=0.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#F0A000",
            width=2.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#F0A000",
            arrow_length=5.0,
            arrow_width=4.0,
            routing="ortho",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#00F000",
            width=1.5,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#00F000",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="ortho",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#000000",
        stroke="#333333",
        stroke_width=2.0,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=8.0,
        font_color="#FFFFFF",
        font_weight="bold",
        padding=8.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#000000"),
)

BUBBLEGUM_THEME = Theme(
    name="bubblegum",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#FF69B4",
            stroke="#E050A0",
            stroke_width=1.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#FF1493",
            stroke="#D01080",
            stroke_width=2.5,
            font_family="Arial",
            font_size=8.5,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=30.0,
            min_height=30.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#FFB6C1",
            stroke="#E0A0A8",
            stroke_width=1.0,
            font_family="Arial",
            font_size=7.5,
            font_color="#8B2050",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#FF69B4",
            width=2.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#FF69B4",
            arrow_length=5.0,
            arrow_width=4.0,
            routing="bezier",
            curvature=0.3,
        ),
        "back": EdgeStyle(
            color="#FFB6C1",
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#FFB6C1",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.35,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#FFF0F5",
        stroke="#FFB6C1",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=16.0,
        font_size=8.5,
        font_color="#FF1493",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#FFF5F8"),
)

POLAROID_THEME = Theme(
    name="polaroid",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#F5F0E0",
            stroke="#E0D8C8",
            stroke_width=1.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#6A5A48",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=1.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#C89068",
            stroke="#A87050",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#FFE8D0",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=22.0,
            corner_radius=1.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#88A8A0",
            stroke="#688888",
            stroke_width=1.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#E0F0E8",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=1.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#A09080",
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#A09080",
            arrow_length=4.5,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.2,
        ),
        "back": EdgeStyle(
            color="#C0B8A8",
            width=0.6,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#C0B8A8",
            arrow_length=4.0,
            arrow_width=2.5,
            routing="bezier",
            curvature=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#FFFFFF",
        stroke="#D8D0C0",
        stroke_width=4.0,
        stroke_dash="solid",
        corner_radius=1.0,
        font_size=8.0,
        font_color="#8A7A68",
        font_weight="normal",
        padding=14.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#E8E0D0"),
)

ART_NOUVEAU_THEME = Theme(
    name="art_nouveau",
    node_styles={
        "default": NodeStyle(
            shape="ellipse",
            fill="#C4A882",
            stroke="#8B7355",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#3A2A18",
            font_style="italic",
            padding=(5.0, 4.0),
            min_width=30.0,
            min_height=22.0,
        ),
        "input": NodeStyle(
            shape="ellipse",
            fill="#6A8E5A",
            stroke="#4A6E3A",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#E0F0D8",
            font_style="italic",
            font_weight="bold",
            padding=(5.0, 4.0),
            min_width=32.0,
            min_height=24.0,
        ),
        "output": NodeStyle(
            shape="ellipse",
            fill="#9E7B9E",
            stroke="#7E5B7E",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#F0E0F0",
            font_style="italic",
            padding=(5.0, 4.0),
            min_width=30.0,
            min_height=22.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#8B7355",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#8B7355",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.45,
        ),
        "back": EdgeStyle(
            color="#A09070",
            width=0.8,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#A09070",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.55,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F0E8D8",
        stroke="#8B7355",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=12.0,
        font_size=9.0,
        font_color="#5A4A30",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#F5ECD8"),
)

TIM_BURTON_THEME = Theme(
    name="tim_burton",
    node_styles={
        "default": NodeStyle(
            shape="ellipse",
            fill="#1A1A1A",
            stroke="#E0E0E0",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#E0E0E0",
            padding=(5.0, 4.0),
            min_width=28.0,
            min_height=20.0,
        ),
        "input": NodeStyle(
            shape="ellipse",
            fill="#8B1A1A",
            stroke="#E0E0E0",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#E0E0E0",
            font_weight="bold",
            padding=(5.0, 4.0),
            min_width=30.0,
            min_height=22.0,
        ),
        "output": NodeStyle(
            shape="ellipse",
            fill="#E0E0E0",
            stroke="#1A1A1A",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#1A1A1A",
            padding=(5.0, 4.0),
            min_width=28.0,
            min_height=20.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#E0E0E0",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#E0E0E0",
            arrow_length=5.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.4,
        ),
        "back": EdgeStyle(
            color="#606060",
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#606060",
            arrow_length=4.0,
            arrow_width=2.5,
            routing="bezier",
            curvature=0.5,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#0A0A0A",
        stroke="#404040",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=9.0,
        font_color="#A0A0A0",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#0A0A0A"),
)

# ── Film & animation themes ──────────────────────────────────────────

DISNEY_THEME = Theme(
    name="disney",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#1A73E8",
            stroke="#1060C8",
            stroke_width=2.0,
            font_family="Arial",
            font_size=8.5,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#FFD700",
            stroke="#E0B800",
            stroke_width=2.5,
            font_family="Arial",
            font_size=8.5,
            font_color="#1A1A1A",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#E040A0",
            stroke="#C02880",
            stroke_width=2.0,
            font_family="Arial",
            font_size=8.5,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#FFD700",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#FFD700",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.25,
        ),
        "back": EdgeStyle(
            color="#A0A0A0",
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#A0A0A0",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.3,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F0F8FF",
        stroke="#87CEEB",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=10.0,
        font_size=9.0,
        font_color="#1A73E8",
        font_weight="bold",
        padding=12.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#0C1E3C"),
)

GHIBLI_THEME = Theme(
    name="ghibli",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#6BAF7A",
            stroke="#4A8F5A",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#1A3A20",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#E88868",
            stroke="#C86848",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#3A1A10",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#87CEEB",
            stroke="#68A8C8",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#1A3A4A",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#5A8A5A",
            width=1.2,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.35,
        ),
        "back": EdgeStyle(
            color="#90B890",
            width=0.7,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.45,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#E8F0D8",
        stroke="#A0C090",
        stroke_width=0.8,
        stroke_dash="solid",
        corner_radius=10.0,
        font_size=8.0,
        font_color="#4A6A40",
        font_weight="normal",
        padding=10.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#D0E8F0"),
)

# ── Painter themes ───────────────────────────────────────────────────

# Picasso -- cubism: angular fragments, muted blues/browns/terra cotta
PICASSO_THEME = Theme(
    name="picasso",
    node_styles={
        "default": NodeStyle(
            shape="diamond",
            fill="#8B7D6B",
            stroke="#6B5D4B",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#E8D8C0",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "input": NodeStyle(
            shape="diamond",
            fill="#4A6888",
            stroke="#2A4868",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#C0D0E0",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="diamond",
            fill="#B87848",
            stroke="#986038",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#F0E0C8",
            padding=(5.0, 5.0),
            min_width=24.0,
            min_height=24.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#4A4038",
            width=2.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#4A4038",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="straight",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#7A6858",
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#7A6858",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="straight",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#D8C8B0",
        stroke="#8B7D6B",
        stroke_width=2.0,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=9.0,
        font_color="#4A3828",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#E8D8C0"),
)

# Pollock -- abstract expressionism: chaotic splatters on beige canvas
POLLOCK_THEME = Theme(
    name="pollock",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#1A1A1A",
            stroke="#000000",
            stroke_width=2.5,
            font_family="Arial",
            font_size=7.0,
            font_color="#F0E8D0",
            padding=(3.0, 3.0),
            min_width=16.0,
            min_height=16.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#CC2020",
            stroke="#A01010",
            stroke_width=3.0,
            font_family="Arial",
            font_size=7.5,
            font_color="#FFE0C0",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=20.0,
            min_height=20.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#2060A0",
            stroke="#104080",
            stroke_width=2.5,
            font_family="Arial",
            font_size=7.0,
            font_color="#C0D8F0",
            padding=(3.0, 3.0),
            min_width=16.0,
            min_height=16.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#1A1A1A", width=1.0, style="solid", arrow="none", routing="bezier", curvature=0.6
        ),
        "back": EdgeStyle(
            color="#C8A030",
            width=0.6,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.7,
            opacity=0.6,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#E0D8C0",
        stroke="#C0B898",
        stroke_width=0.5,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=7.0,
        font_color="#6A5A40",
        font_weight="normal",
        padding=6.0,
        opacity=0.2,
    ),
    graph_style=GraphStyle(background_color="#E8DCC8"),
)

# Riley -- Bridget Riley op art: vivid parallel stripes, high contrast
RILEY_THEME = Theme(
    name="riley",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#000000",
            stroke="#FFFFFF",
            stroke_width=2.5,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=28.0,
            min_height=18.0,
            corner_radius=0.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#FF0000",
            stroke="#000000",
            stroke_width=3.0,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=20.0,
            corner_radius=0.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#0000FF",
            stroke="#000000",
            stroke_width=2.5,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=28.0,
            min_height=18.0,
            corner_radius=0.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#000000",
            width=3.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#000000",
            arrow_length=6.0,
            arrow_width=5.0,
            routing="straight",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#FFFFFF",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#FFFFFF",
            arrow_length=5.0,
            arrow_width=4.0,
            routing="straight",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#E0E0E0",
        stroke="#000000",
        stroke_width=3.0,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=9.0,
        font_color="#000000",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

# Renoir -- impressionist warmth: soft golden light, lush colors
RENOIR_THEME = Theme(
    name="renoir",
    node_styles={
        "default": NodeStyle(
            shape="ellipse",
            fill="#D4956B",
            stroke="#B47848",
            stroke_width=1.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#3A2010",
            padding=(5.0, 4.0),
            min_width=28.0,
            min_height=22.0,
            opacity=0.8,
        ),
        "input": NodeStyle(
            shape="ellipse",
            fill="#C85060",
            stroke="#A83848",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#FFE0E0",
            font_weight="bold",
            padding=(5.0, 4.0),
            min_width=30.0,
            min_height=24.0,
            opacity=0.8,
        ),
        "output": NodeStyle(
            shape="ellipse",
            fill="#6888A0",
            stroke="#487088",
            stroke_width=1.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#E0E8F0",
            padding=(5.0, 4.0),
            min_width=28.0,
            min_height=22.0,
            opacity=0.8,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#8B7050",
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#8B7050",
            arrow_length=4.5,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.3,
            opacity=0.5,
        ),
        "back": EdgeStyle(
            color="#B0A080",
            width=0.6,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#B0A080",
            arrow_length=4.0,
            arrow_width=2.5,
            routing="bezier",
            curvature=0.4,
            opacity=0.35,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#E8D8B8",
        stroke="#C0A878",
        stroke_width=0.5,
        stroke_dash="solid",
        corner_radius=10.0,
        font_size=8.0,
        font_color="#6A5838",
        font_weight="normal",
        padding=10.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#F0E4D0"),
)

# Da Vinci -- technical diagrams: sepia ink, anatomical precision
DA_VINCI_THEME = Theme(
    name="da_vinci",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#E0D0B8",
            stroke="#2C1A0E",
            stroke_width=1.0,
            font_family="Georgia",
            font_size=7.5,
            font_color="#2C1A0E",
            font_style="italic",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#D0C0A0",
            stroke="#2C1A0E",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#2C1A0E",
            font_style="italic",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#E8D8C0",
            stroke="#2C1A0E",
            stroke_width=0.8,
            font_family="Georgia",
            font_size=7.5,
            font_color="#2C1A0E",
            font_style="italic",
            padding=(4.0, 4.0),
            min_width=20.0,
            min_height=20.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#2C1A0E",
            width=0.6,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#2C1A0E",
            arrow_length=4.0,
            arrow_width=2.0,
            routing="straight",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#6B5038",
            width=0.4,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#6B5038",
            arrow_length=3.5,
            arrow_width=1.5,
            routing="straight",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#E8D8C0",
        stroke="#8B7355",
        stroke_width=0.3,
        stroke_dash="dashed",
        corner_radius=0.0,
        font_size=7.5,
        font_color="#6B5038",
        font_weight="normal",
        padding=8.0,
        opacity=0.2,
    ),
    graph_style=GraphStyle(background_color="#E8D8C0"),
)

# Mondrian -- primary color grid: red/blue/yellow blocks, black lines
MONDRIAN_THEME = Theme(
    name="mondrian",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#FFFFFF",
            stroke="#000000",
            stroke_width=4.0,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#000000",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=24.0,
            corner_radius=0.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#DD0100",
            stroke="#000000",
            stroke_width=4.0,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=26.0,
            corner_radius=0.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#0039A6",
            stroke="#000000",
            stroke_width=4.0,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=24.0,
            corner_radius=0.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#000000", width=4.0, style="solid", arrow="none", routing="ortho", curvature=0.0
        ),
        "back": EdgeStyle(
            color="#000000", width=2.0, style="solid", arrow="none", routing="ortho", curvature=0.0
        ),
    },
    cluster_style=ClusterStyle(
        fill="#FADA5E",
        stroke="#000000",
        stroke_width=4.0,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=9.0,
        font_color="#000000",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#F0F0E8"),
)

# Van Gogh -- swirling post-impressionism: bold brushstrokes, starry night
VAN_GOGH_THEME = Theme(
    name="van_gogh",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#2A5CAA",
            stroke="#1A3C8A",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#F0E8A0",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#F0C828",
            stroke="#D0A818",
            stroke_width=3.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#1A2A4A",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#2A7A4A",
            stroke="#1A5A30",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#D0F0C0",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#F0C828", width=2.0, style="solid", arrow="none", routing="bezier", curvature=0.5
        ),
        "back": EdgeStyle(
            color="#6A8AB8", width=1.0, style="solid", arrow="none", routing="bezier", curvature=0.6
        ),
    },
    cluster_style=ClusterStyle(
        fill="#1A2A4A",
        stroke="#2A5CAA",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=8.0,
        font_size=8.0,
        font_color="#F0E8A0",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#1A2040"),
)

# Klimt -- gold leaf, decorative patterns, Viennese secession
KLIMT_THEME = Theme(
    name="klimt",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#C8A030",
            stroke="#A08020",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#1A1208",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=22.0,
            corner_radius=3.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#2A5838",
            stroke="#1A4828",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#D0C890",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=24.0,
            corner_radius=3.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#8B3040",
            stroke="#6B1828",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#E8C898",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=22.0,
            corner_radius=3.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#C8A030",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#C8A030",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.3,
        ),
        "back": EdgeStyle(
            color="#8B7030",
            width=0.8,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#8B7030",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.35,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#1A1808",
        stroke="#C8A030",
        stroke_width=2.0,
        stroke_dash="solid",
        corner_radius=4.0,
        font_size=9.0,
        font_color="#C8A030",
        font_weight="bold",
        padding=12.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#1A1608"),
)

# Warhol -- pop art: screen-print brights, repetition
WARHOL_THEME = Theme(
    name="warhol",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#FF6EC7",
            stroke="#D04898",
            stroke_width=2.0,
            font_family="Arial",
            font_size=8.5,
            font_color="#000000",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=20.0,
            corner_radius=0.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#00CED1",
            stroke="#00A8A8",
            stroke_width=2.5,
            font_family="Arial",
            font_size=8.5,
            font_color="#000000",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=22.0,
            corner_radius=0.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#FFD700",
            stroke="#D0B000",
            stroke_width=2.0,
            font_family="Arial",
            font_size=8.5,
            font_color="#000000",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=20.0,
            corner_radius=0.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#000000",
            width=2.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#000000",
            arrow_length=6.0,
            arrow_width=4.5,
            routing="straight",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#FF6EC7",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#FF6EC7",
            arrow_length=5.0,
            arrow_width=4.0,
            routing="straight",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#FF4500",
        stroke="#000000",
        stroke_width=3.0,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=10.0,
        font_color="#FFFFFF",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

# ── More science & fantasy themes ────────────────────────────────────

# Crystal -- gemstone facets: prismatic refractions, transparent geometry
CRYSTAL_THEME = Theme(
    name="crystal",
    node_styles={
        "default": NodeStyle(
            shape="diamond",
            fill="#C8D8F0",
            stroke="#90A8D0",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#2A3A5A",
            padding=(5.0, 5.0),
            min_width=24.0,
            min_height=24.0,
            opacity=0.75,
        ),
        "input": NodeStyle(
            shape="diamond",
            fill="#D8A0C8",
            stroke="#B878A8",
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#3A1A30",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=26.0,
            min_height=26.0,
            opacity=0.8,
        ),
        "output": NodeStyle(
            shape="diamond",
            fill="#A0D8C8",
            stroke="#78B8A8",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#1A3A30",
            padding=(5.0, 5.0),
            min_width=22.0,
            min_height=22.0,
            opacity=0.7,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#90A8D0",
            width=0.8,
            style="solid",
            arrow="none",
            routing="straight",
            curvature=0.0,
            opacity=0.5,
        ),
        "back": EdgeStyle(
            color="#B0C0D8",
            width=0.5,
            style="solid",
            arrow="none",
            routing="straight",
            curvature=0.0,
            opacity=0.3,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#E8E0F0",
        stroke="#B0A8C8",
        stroke_width=0.5,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=7.5,
        font_color="#5A5078",
        font_weight="normal",
        padding=8.0,
        opacity=0.25,
    ),
    graph_style=GraphStyle(background_color="#F0ECF8"),
)

# Neon sign -- bent glass tubes: glowing colored tubes on dark brick
NEON_SIGN_THEME = Theme(
    name="neon_sign",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#1A1A1A",
            stroke="#FF355E",
            stroke_width=2.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#FF355E",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#1A1A1A",
            stroke="#50FF50",
            stroke_width=3.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#50FF50",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#1A1A1A",
            stroke="#1BFFFF",
            stroke_width=2.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#1BFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#FF355E",
            width=2.0,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.25,
            opacity=0.8,
        ),
        "back": EdgeStyle(
            color="#6050FF",
            width=1.5,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.35,
            opacity=0.6,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#1A1010",
        stroke="#3A2020",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=4.0,
        font_size=8.5,
        font_color="#FF8888",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#1A1010"),
)

# Fantasy map -- Tolkien/GoT cartography: parchment, hand-drawn mountains
FANTASY_MAP_THEME = Theme(
    name="fantasy_map",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#C8A868",
            stroke="#8B6D30",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#3A2818",
            font_weight="bold",
            font_style="italic",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
        "input": NodeStyle(
            shape="star",
            fill="#C83020",
            stroke="#A01818",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.5,
            font_color="#FFE0C0",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#4A7858",
            stroke="#2A5838",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=7.5,
            font_color="#D0E8D0",
            font_style="italic",
            padding=(4.0, 4.0),
            min_width=20.0,
            min_height=20.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#6B4226",
            width=1.2,
            style="dashed",
            arrow="none",
            routing="bezier",
            curvature=0.2,
        ),
        "back": EdgeStyle(
            color="#8B7355",
            width=0.6,
            style="dashed",
            arrow="none",
            routing="bezier",
            curvature=0.3,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#E0D0A8",
        stroke="#8B7355",
        stroke_width=1.0,
        stroke_dash="dashed",
        corner_radius=0.0,
        font_size=9.0,
        font_color="#4A3218",
        font_weight="bold",
        padding=12.0,
        opacity=0.35,
    ),
    graph_style=GraphStyle(background_color="#E8D8B0"),
)

# Tech tree -- video game research/skill tree: locked/unlocked nodes
TECH_TREE_THEME = Theme(
    name="tech_tree",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#2A3040",
            stroke="#4A90D0",
            stroke_width=2.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#C0D8F0",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=22.0,
            corner_radius=4.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#4A90D0",
            stroke="#3070B0",
            stroke_width=2.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=36.0,
            min_height=24.0,
            corner_radius=4.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#1A2030",
            stroke="#3A4858",
            stroke_width=1.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#607080",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=22.0,
            corner_radius=4.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#4A90D0",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#4A90D0",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="ortho",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#3A4858",
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#3A4858",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="ortho",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#1A2030",
        stroke="#2A3848",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=6.0,
        font_size=8.5,
        font_color="#6090C0",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#10181F"),
)

# SimCity -- Maxis city builder: zoning colors (green residential,
# blue commercial, yellow industrial), road grid, isometric feel
SIMCITY_THEME = Theme(
    name="simcity",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#4CAF50",
            stroke="#388E3C",
            stroke_width=2.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=28.0,
            min_height=22.0,
            corner_radius=1.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#2196F3",
            stroke="#1976D2",
            stroke_width=2.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=24.0,
            corner_radius=1.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#FFC107",
            stroke="#FFA000",
            stroke_width=2.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#1A1A1A",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=28.0,
            min_height=22.0,
            corner_radius=1.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#757575", width=2.5, style="solid", arrow="none", routing="ortho", curvature=0.0
        ),
        "back": EdgeStyle(
            color="#BDBDBD", width=1.2, style="dashed", arrow="none", routing="ortho", curvature=0.0
        ),
    },
    cluster_style=ClusterStyle(
        fill="#E8F5E9",
        stroke="#81C784",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=2.0,
        font_size=8.5,
        font_color="#2E7D32",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#8BC34A"),
)

# ── Countries ─────────────────────────────────────────────────────────

USA_THEME = Theme(
    name="usa",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#FFFFFF",
            stroke="#002868",
            stroke_width=2.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#002868",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=2.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#BF0A30",
            stroke="#8F0720",
            stroke_width=2.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=22.0,
            corner_radius=2.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#002868",
            stroke="#001848",
            stroke_width=2.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=2.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#BF0A30",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#BF0A30",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.2,
        ),
        "back": EdgeStyle(
            color="#002868",
            width=1.0,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#002868",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F0F0F8",
        stroke="#002868",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=2.0,
        font_size=8.5,
        font_color="#002868",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

# Brazil -- verde e amarelo
BRAZIL_THEME = Theme(
    name="brazil",
    node_styles={
        "default": NodeStyle(
            shape="diamond",
            fill="#009C3B",
            stroke="#007A2E",
            stroke_width=2.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "input": NodeStyle(
            shape="diamond",
            fill="#FFDF00",
            stroke="#D0B800",
            stroke_width=2.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#002776",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#002776",
            stroke="#001A5A",
            stroke_width=2.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#009C3B",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#009C3B",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.25,
        ),
        "back": EdgeStyle(
            color="#FFDF00",
            width=1.0,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#FFDF00",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.3,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#E0F8E0",
        stroke="#009C3B",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=4.0,
        font_size=8.5,
        font_color="#009C3B",
        font_weight="bold",
        padding=10.0,
        opacity=0.35,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

# Japan -- hinomaru: clean white, rising sun red, elegant restraint
JAPAN_THEME = Theme(
    name="japan",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#FFFFFF",
            stroke="#BC002D",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#333333",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#BC002D",
            stroke="#8C0020",
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#F8F8F8",
            stroke="#CCCCCC",
            stroke_width=1.0,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#666666",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#333333",
            width=0.8,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#333333",
            arrow_length=4.5,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.15,
        ),
        "back": EdgeStyle(
            color="#CCCCCC",
            width=0.5,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#CCCCCC",
            arrow_length=4.0,
            arrow_width=2.5,
            routing="bezier",
            curvature=0.2,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#FFFFFF",
        stroke="#BC002D",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=8.0,
        font_color="#BC002D",
        font_weight="bold",
        padding=10.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

# India -- saffron/white/green with Ashoka Chakra navy
INDIA_THEME = Theme(
    name="india",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#FFFFFF",
            stroke="#000080",
            stroke_width=1.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#000080",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=2.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#FF9933",
            stroke="#D88020",
            stroke_width=2.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=22.0,
            corner_radius=2.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#138808",
            stroke="#0A6806",
            stroke_width=2.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=2.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#000080",
            width=1.2,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#000080",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.2,
        ),
        "back": EdgeStyle(
            color="#FF9933",
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#FF9933",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F8F0E8",
        stroke="#000080",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=3.0,
        font_size=8.5,
        font_color="#000080",
        font_weight="bold",
        padding=10.0,
        opacity=0.35,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

# ── Sports & competition themes ──────────────────────────────────────

# Tournament bracket
BRACKET_THEME = Theme(
    name="bracket",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#FFFFFF",
            stroke="#333333",
            stroke_width=1.5,
            font_family="Arial",
            font_size=8.5,
            font_color="#1A1A1A",
            padding=(6.0, 3.0),
            min_width=42.0,
            min_height=18.0,
            corner_radius=2.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#FFD700",
            stroke="#C0A000",
            stroke_width=2.5,
            font_family="Arial",
            font_size=9.0,
            font_color="#1A1A1A",
            font_weight="bold",
            padding=(6.0, 3.0),
            min_width=44.0,
            min_height=20.0,
            corner_radius=2.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#E8E8E8",
            stroke="#AAAAAA",
            stroke_width=1.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#666666",
            padding=(6.0, 3.0),
            min_width=42.0,
            min_height=18.0,
            corner_radius=2.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#333333", width=1.5, style="solid", arrow="none", routing="ortho", curvature=0.0
        ),
        "back": EdgeStyle(
            color="#AAAAAA", width=0.8, style="dashed", arrow="none", routing="ortho", curvature=0.0
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F8F8F0",
        stroke="#CCCCCC",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=9.0,
        font_color="#333333",
        font_weight="bold",
        padding=12.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

# Fantasy football -- draft board / matchup card feel
FANTASY_FOOTBALL_THEME = Theme(
    name="fantasy_football",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#1A472A",
            stroke="#0D3318",
            stroke_width=2.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=22.0,
            corner_radius=3.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#C8AA2C",
            stroke="#A08820",
            stroke_width=2.5,
            font_family="Arial",
            font_size=8.5,
            font_color="#1A1A1A",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=36.0,
            min_height=24.0,
            corner_radius=3.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#8B0000",
            stroke="#6B0000",
            stroke_width=2.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=22.0,
            corner_radius=3.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#C8AA2C",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#C8AA2C",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.15,
        ),
        "back": EdgeStyle(
            color="#4A6A4A",
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#4A6A4A",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.2,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#0D3318",
        stroke="#1A472A",
        stroke_width=2.0,
        stroke_dash="solid",
        corner_radius=4.0,
        font_size=9.0,
        font_color="#C8AA2C",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#0A2010"),
)

# Seahawks -- action green / navy / wolf gray
SEAHAWKS_THEME = Theme(
    name="seahawks",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#002244",
            stroke="#001830",
            stroke_width=2.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#69BE28",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#69BE28",
            stroke="#4A9E18",
            stroke_width=2.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#002244",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#A5ACAF",
            stroke="#858C8F",
            stroke_width=1.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#002244",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#69BE28",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#69BE28",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.2,
        ),
        "back": EdgeStyle(
            color="#A5ACAF",
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#A5ACAF",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#001830",
        stroke="#002244",
        stroke_width=2.0,
        stroke_dash="solid",
        corner_radius=4.0,
        font_size=8.5,
        font_color="#69BE28",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#001020"),
)

# ── Math theme ────────────────────────────────────────────────────────

# Category theory -- commutative diagrams: clean, minimal,
# labeled morphism arrows, mathematical typesetting feel
CATEGORY_THEORY_THEME = Theme(
    name="category_theory",
    node_styles={
        "default": NodeStyle(
            shape="none",
            fill="transparent",
            stroke="transparent",
            stroke_width=0.0,
            font_family="Times New Roman",
            font_size=10.0,
            font_color="#000000",
            font_style="italic",
            padding=(2.0, 1.0),
            min_width=0.0,
            min_height=0.0,
        ),
        "input": NodeStyle(
            shape="none",
            fill="transparent",
            stroke="transparent",
            stroke_width=0.0,
            font_family="Times New Roman",
            font_size=10.0,
            font_color="#000000",
            font_style="italic",
            font_weight="bold",
            padding=(2.0, 1.0),
            min_width=0.0,
            min_height=0.0,
        ),
        "output": NodeStyle(
            shape="none",
            fill="transparent",
            stroke="transparent",
            stroke_width=0.0,
            font_family="Times New Roman",
            font_size=10.0,
            font_color="#000000",
            font_style="italic",
            padding=(2.0, 1.0),
            min_width=0.0,
            min_height=0.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#000000",
            width=0.8,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#000000",
            arrow_length=5.0,
            arrow_width=3.0,
            routing="straight",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#000000",
            width=0.6,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#000000",
            arrow_length=4.5,
            arrow_width=2.5,
            routing="bezier",
            curvature=0.3,
        ),
    },
    cluster_style=ClusterStyle(
        fill="transparent",
        stroke="#888888",
        stroke_width=0.3,
        stroke_dash="dashed",
        corner_radius=0.0,
        font_size=9.0,
        font_color="#444444",
        font_weight="normal",
        padding=6.0,
        opacity=0.2,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

# Syntax tree -- linguistics sentence diagram: labeled brackets,
# clean academic, part-of-speech categories
SYNTAX_TREE_THEME = Theme(
    name="syntax_tree",
    node_styles={
        "default": NodeStyle(
            shape="none",
            fill="transparent",
            stroke="transparent",
            stroke_width=0.0,
            font_family="Times New Roman",
            font_size=9.0,
            font_color="#000000",
            padding=(2.0, 1.0),
            min_width=0.0,
            min_height=0.0,
        ),
        "input": NodeStyle(
            shape="none",
            fill="transparent",
            stroke="transparent",
            stroke_width=0.0,
            font_family="Times New Roman",
            font_size=9.0,
            font_color="#000000",
            font_weight="bold",
            padding=(2.0, 1.0),
            min_width=0.0,
            min_height=0.0,
        ),
        "output": NodeStyle(
            shape="none",
            fill="transparent",
            stroke="transparent",
            stroke_width=0.0,
            font_family="Times New Roman",
            font_size=9.0,
            font_color="#444444",
            font_style="italic",
            padding=(2.0, 1.0),
            min_width=0.0,
            min_height=0.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#000000",
            width=0.8,
            style="solid",
            arrow="none",
            routing="straight",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#888888",
            width=0.5,
            style="dashed",
            arrow="none",
            routing="bezier",
            curvature=0.2,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F8F8F8",
        stroke="#CCCCCC",
        stroke_width=0.5,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=8.5,
        font_color="#333333",
        font_weight="bold",
        padding=6.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

# Data structure -- CS textbook: pointer arrows, struct boxes,
# null terminators, academic clean
DATA_STRUCTURE_THEME = Theme(
    name="data_structure",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#E8F4FD",
            stroke="#1976D2",
            stroke_width=1.5,
            font_family="Courier New",
            font_size=8.0,
            font_color="#0D47A1",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=20.0,
            corner_radius=1.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#1976D2",
            stroke="#0D47A1",
            stroke_width=2.0,
            font_family="Courier New",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=36.0,
            min_height=22.0,
            corner_radius=1.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#FFEBEE",
            stroke="#E53935",
            stroke_width=1.5,
            font_family="Courier New",
            font_size=8.0,
            font_color="#B71C1C",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=20.0,
            corner_radius=1.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#1976D2",
            width=1.2,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#1976D2",
            arrow_length=6.0,
            arrow_width=4.0,
            routing="ortho",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#E53935",
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#E53935",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="ortho",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F5F5F5",
        stroke="#BDBDBD",
        stroke_width=1.0,
        stroke_dash="dashed",
        corner_radius=2.0,
        font_size=8.5,
        font_color="#424242",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

OLYMPIC_THEME = Theme(
    name="olympic",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#0085C7",
            stroke="#006AA5",
            stroke_width=2.5,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#F4C300",
            stroke="#D0A500",
            stroke_width=3.0,
            font_family="Helvetica",
            font_size=8.5,
            font_color="#1A1A1A",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#009F3D",
            stroke="#007A2E",
            stroke_width=2.5,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=26.0,
            min_height=26.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#DF0024", width=2.0, style="solid", arrow="none", routing="bezier", curvature=0.2
        ),
        "back": EdgeStyle(
            color="#000000",
            width=1.0,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F8F8F8",
        stroke="#CCCCCC",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=8.0,
        font_size=9.0,
        font_color="#333333",
        font_weight="bold",
        padding=12.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

CARVED_THEME = Theme(
    name="carved",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#8B8070",
            stroke="#A09888",
            stroke_width=0.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#C8C0B0",
            font_style="italic",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=18.0,
            corner_radius=0.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#8B8070",
            stroke="#B0A898",
            stroke_width=1.0,
            font_family="Georgia",
            font_size=8.5,
            font_color="#D8D0C0",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=0.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#8B8070",
            stroke="#A09888",
            stroke_width=0.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#B8B0A0",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=18.0,
            corner_radius=0.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#A09888",
            width=0.8,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#A09888",
            arrow_length=4.0,
            arrow_width=2.5,
            routing="straight",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#908878",
            width=0.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#908878",
            arrow_length=3.5,
            arrow_width=2.0,
            routing="straight",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#7A7268",
        stroke="#8A8278",
        stroke_width=0.5,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=8.0,
        font_color="#B0A898",
        font_weight="normal",
        padding=8.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#7A7268"),
)

CHUTES_LADDERS_THEME = Theme(
    name="chutes_ladders",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#FFE4B5",
            stroke="#DEB887",
            stroke_width=1.5,
            font_family="Arial",
            font_size=9.0,
            font_color="#333333",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
            corner_radius=2.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#FFD700",
            stroke="#DAA520",
            stroke_width=2.5,
            font_family="Arial",
            font_size=9.0,
            font_color="#1A1A1A",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
            corner_radius=2.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#98FB98",
            stroke="#3CB371",
            stroke_width=2.0,
            font_family="Arial",
            font_size=9.0,
            font_color="#1A4A1A",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
            corner_radius=2.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#228B22",
            width=2.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#228B22",
            arrow_length=5.0,
            arrow_width=4.0,
            routing="straight",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#DC143C",
            width=2.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#DC143C",
            arrow_length=5.0,
            arrow_width=4.0,
            routing="bezier",
            curvature=0.3,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#FFF8DC",
        stroke="#DEB887",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=4.0,
        font_size=9.0,
        font_color="#8B4513",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#FFF8DC"),
)

BARK_CARVING_THEME = Theme(
    name="bark_carving",
    node_styles={
        "default": NodeStyle(
            shape="ellipse",
            fill="#6B4C33",
            stroke="#8B6D4C",
            stroke_width=0.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#C8B898",
            font_style="italic",
            padding=(5.0, 3.0),
            min_width=28.0,
            min_height=20.0,
        ),
        "input": NodeStyle(
            shape="ellipse",
            fill="#6B4C33",
            stroke="#A89070",
            stroke_width=1.0,
            font_family="Georgia",
            font_size=8.5,
            font_color="#D8C8A8",
            font_weight="bold",
            padding=(5.0, 4.0),
            min_width=30.0,
            min_height=22.0,
        ),
        "output": NodeStyle(
            shape="ellipse",
            fill="#6B4C33",
            stroke="#7A5E40",
            stroke_width=0.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#B0A080",
            padding=(5.0, 3.0),
            min_width=26.0,
            min_height=18.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#A89070", width=0.8, style="solid", arrow="none", routing="bezier", curvature=0.2
        ),
        "back": EdgeStyle(
            color="#8B7355", width=0.5, style="solid", arrow="none", routing="bezier", curvature=0.3
        ),
    },
    cluster_style=ClusterStyle(
        fill="#5A3E28",
        stroke="#6B4C33",
        stroke_width=0.5,
        stroke_dash="solid",
        corner_radius=6.0,
        font_size=8.0,
        font_color="#A89070",
        font_weight="normal",
        padding=8.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#5A3E28"),
)

CHISEL_THEME = Theme(
    name="chisel",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#6A6A6A",
            stroke="#8A8A8A",
            stroke_width=0.8,
            font_family="Georgia",
            font_size=8.0,
            font_color="#B8B8B8",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=0.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#6A6A6A",
            stroke="#A0A0A0",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=8.5,
            font_color="#D0D0D0",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=22.0,
            corner_radius=0.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#6A6A6A",
            stroke="#808080",
            stroke_width=0.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#A0A0A0",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=18.0,
            corner_radius=0.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#909090",
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#909090",
            arrow_length=4.5,
            arrow_width=3.0,
            routing="straight",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#787878",
            width=0.6,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#787878",
            arrow_length=4.0,
            arrow_width=2.5,
            routing="straight",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#5A5A5A",
        stroke="#6A6A6A",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=8.5,
        font_color="#A8A8A8",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#5A5A5A"),
)

CHECKERS_THEME = Theme(
    name="checkers",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#CC0000",
            stroke="#990000",
            stroke_width=2.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#1A1A1A",
            stroke="#000000",
            stroke_width=2.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#CC0000",
            stroke="#FFD700",
            stroke_width=3.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFD700",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=26.0,
            min_height=26.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#8B4513",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#8B4513",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="straight",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#D2B48C",
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#D2B48C",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="straight",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#DEB887",
        stroke="#8B4513",
        stroke_width=2.0,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=9.0,
        font_color="#4A2A0A",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#F5DEB3"),
)

CHESS_THEME = Theme(
    name="chess",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#F0D9B5",
            stroke="#B58863",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.5,
            font_color="#1A1A1A",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=28.0,
            min_height=28.0,
            corner_radius=0.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#1A1A1A",
            stroke="#000000",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=9.0,
            font_color="#F0D9B5",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=30.0,
            corner_radius=0.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#FFFFFF",
            stroke="#B58863",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.5,
            font_color="#1A1A1A",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=28.0,
            min_height=28.0,
            corner_radius=0.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#1A1A1A",
            width=1.2,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#1A1A1A",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="straight",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#B58863",
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#B58863",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.2,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#D4A76A",
        stroke="#8B6D4C",
        stroke_width=2.5,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=9.0,
        font_color="#1A1A1A",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#B58863"),
)

GO_BOARD_THEME = Theme(
    name="go",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#1A1A1A",
            stroke="#000000",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=7.0,
            font_color="#E0E0E0",
            padding=(3.0, 3.0),
            min_width=20.0,
            min_height=20.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#FFFFFF",
            stroke="#CCCCCC",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=7.0,
            font_color="#1A1A1A",
            font_weight="bold",
            padding=(3.0, 3.0),
            min_width=20.0,
            min_height=20.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#1A1A1A",
            stroke="#444444",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=7.0,
            font_color="#C0C0C0",
            padding=(3.0, 3.0),
            min_width=20.0,
            min_height=20.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#2A2A2A",
            width=0.8,
            style="solid",
            arrow="none",
            routing="straight",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#4A4A4A",
            width=0.5,
            style="solid",
            arrow="none",
            routing="straight",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#DCB35C",
        stroke="#C8A040",
        stroke_width=0.5,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=7.5,
        font_color="#4A3818",
        font_weight="bold",
        padding=6.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#DCB35C"),
)

SCRABBLE_THEME = Theme(
    name="scrabble",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#F2E0C0",
            stroke="#C8A878",
            stroke_width=1.5,
            font_family="Arial",
            font_size=9.0,
            font_color="#1A1A1A",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
            corner_radius=1.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#C83030",
            stroke="#A02020",
            stroke_width=2.0,
            font_family="Arial",
            font_size=9.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=26.0,
            min_height=26.0,
            corner_radius=1.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#2878C8",
            stroke="#1860A8",
            stroke_width=2.0,
            font_family="Arial",
            font_size=9.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
            corner_radius=1.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#1A6838", width=1.5, style="solid", arrow="none", routing="ortho", curvature=0.0
        ),
        "back": EdgeStyle(
            color="#4A9858", width=0.8, style="dashed", arrow="none", routing="ortho", curvature=0.0
        ),
    },
    cluster_style=ClusterStyle(
        fill="#1A6838",
        stroke="#0D4820",
        stroke_width=2.0,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=9.0,
        font_color="#F2E0C0",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#1A6838"),
)

PINBALL_THEME = Theme(
    name="pinball",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#FF4500",
            stroke="#FF6030",
            stroke_width=2.5,
            font_family="Arial",
            font_size=7.5,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#FFD700",
            stroke="#FFA500",
            stroke_width=3.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#1A1A1A",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#FF1493",
            stroke="#E0107A",
            stroke_width=2.5,
            font_family="Arial",
            font_size=7.5,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#C0C0C0",
            width=2.0,
            style="solid",
            arrow="none",
            routing="straight",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#808080", width=1.0, style="solid", arrow="none", routing="bezier", curvature=0.3
        ),
    },
    cluster_style=ClusterStyle(
        fill="#0A0A0A",
        stroke="#333333",
        stroke_width=2.0,
        stroke_dash="solid",
        corner_radius=6.0,
        font_size=8.0,
        font_color="#FFD700",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#0A0A0A"),
)

KALEIDOSCOPE_THEME = Theme(
    name="kaleidoscope",
    node_styles={
        "default": NodeStyle(
            shape="diamond",
            fill="#E040E0",
            stroke="#C020C0",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#FFFFFF",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
        "input": NodeStyle(
            shape="diamond",
            fill="#40E0D0",
            stroke="#20C0B0",
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#1A3A38",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "output": NodeStyle(
            shape="diamond",
            fill="#FFD700",
            stroke="#E0B800",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#3A2A00",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#FF4500",
            width=1.2,
            style="solid",
            arrow="none",
            routing="straight",
            curvature=0.0,
            opacity=0.6,
        ),
        "back": EdgeStyle(
            color="#4169E1",
            width=0.8,
            style="solid",
            arrow="none",
            routing="straight",
            curvature=0.0,
            opacity=0.4,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#1A0A20",
        stroke="#4A2060",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=7.5,
        font_color="#E040E0",
        font_weight="bold",
        padding=8.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#1A0A20"),
)

MORSE_CODE_THEME = Theme(
    name="morse_code",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#C8A030",
            stroke="#A88020",
            stroke_width=2.0,
            font_family="Courier New",
            font_size=8.0,
            font_color="#1A1208",
            font_weight="bold",
            padding=(3.0, 3.0),
            min_width=14.0,
            min_height=14.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#C8A030",
            stroke="#A88020",
            stroke_width=2.0,
            font_family="Courier New",
            font_size=8.0,
            font_color="#1A1208",
            font_weight="bold",
            padding=(8.0, 3.0),
            min_width=32.0,
            min_height=14.0,
            corner_radius=0.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#C8A030",
            stroke="#A88020",
            stroke_width=1.5,
            font_family="Courier New",
            font_size=7.5,
            font_color="#2A1A08",
            padding=(3.0, 3.0),
            min_width=12.0,
            min_height=12.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#4A3818",
            width=1.5,
            style="solid",
            arrow="none",
            routing="straight",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#6A5838",
            width=0.8,
            style="dashed",
            arrow="none",
            routing="straight",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#2A2218",
        stroke="#4A3828",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=8.0,
        font_color="#C8A030",
        font_weight="bold",
        padding=8.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#2A2218"),
)

EMBROIDERY_THEME = Theme(
    name="embroidery",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#E85858",
            stroke="#C83838",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=7.5,
            font_color="#FFFFFF",
            padding=(4.0, 4.0),
            min_width=18.0,
            min_height=18.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#5878D8",
            stroke="#3858B8",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=20.0,
            min_height=20.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#58B868",
            stroke="#389848",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=7.5,
            font_color="#FFFFFF",
            padding=(4.0, 4.0),
            min_width=18.0,
            min_height=18.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#C83838",
            width=1.0,
            style="dashed",
            arrow="none",
            routing="straight",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#3858B8",
            width=0.8,
            style="dashed",
            arrow="none",
            routing="straight",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F5F0E5",
        stroke="#D8C8B0",
        stroke_width=0.5,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=7.5,
        font_color="#8A7A68",
        font_weight="normal",
        padding=8.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#F5F0E5"),
)

DOMINO_THEME = Theme(
    name="domino",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#1A1A1A",
            stroke="#E8E8E8",
            stroke_width=2.0,
            font_family="Arial",
            font_size=9.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=24.0,
            min_height=28.0,
            corner_radius=3.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#FFFFFF",
            stroke="#1A1A1A",
            stroke_width=2.5,
            font_family="Arial",
            font_size=9.0,
            font_color="#1A1A1A",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=26.0,
            min_height=30.0,
            corner_radius=3.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#1A1A1A",
            stroke="#C0C0C0",
            stroke_width=1.5,
            font_family="Arial",
            font_size=9.0,
            font_color="#E0E0E0",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=24.0,
            min_height=28.0,
            corner_radius=3.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#E8E8E8",
            width=1.5,
            style="solid",
            arrow="none",
            routing="straight",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#808080",
            width=0.8,
            style="dashed",
            arrow="none",
            routing="straight",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#2A5A2A",
        stroke="#1A4A1A",
        stroke_width=2.0,
        stroke_dash="solid",
        corner_radius=4.0,
        font_size=9.0,
        font_color="#E8E8E8",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#2A5A2A"),
)

NEWSPAPER_THEME = Theme(
    name="newspaper",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#FFFFF0",
            stroke="#000000",
            stroke_width=0.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#1A1A1A",
            padding=(6.0, 3.0),
            min_width=40.0,
            min_height=18.0,
            corner_radius=0.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#FFFFF0",
            stroke="#000000",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=10.0,
            font_color="#000000",
            font_weight="bold",
            padding=(6.0, 3.0),
            min_width=44.0,
            min_height=22.0,
            corner_radius=0.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#F8F8E8",
            stroke="#000000",
            stroke_width=0.5,
            font_family="Georgia",
            font_size=7.5,
            font_color="#444444",
            font_style="italic",
            padding=(6.0, 3.0),
            min_width=38.0,
            min_height=16.0,
            corner_radius=0.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#000000",
            width=0.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#000000",
            arrow_length=4.0,
            arrow_width=2.5,
            routing="ortho",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#888888",
            width=0.3,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#888888",
            arrow_length=3.5,
            arrow_width=2.0,
            routing="ortho",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#FFFFF0",
        stroke="#000000",
        stroke_width=0.5,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=10.0,
        font_color="#000000",
        font_weight="bold",
        padding=10.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#FFFFF0"),
)

THEME_REGISTRY["scrabble"] = SCRABBLE_THEME
THEME_REGISTRY["pinball"] = PINBALL_THEME
THEME_REGISTRY["kaleidoscope"] = KALEIDOSCOPE_THEME
THEME_REGISTRY["morse_code"] = MORSE_CODE_THEME
THEME_REGISTRY["embroidery"] = EMBROIDERY_THEME
THEME_REGISTRY["domino"] = DOMINO_THEME
# Conspiracy -- Pepe Silvia bulletin board: pushpin nodes, red string
# connections crisscrossing everywhere, cork board background
CONSPIRACY_THEME = Theme(
    name="conspiracy",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#FFFFF0",
            stroke="#C8C0A8",
            stroke_width=0.5,
            font_family="Courier New",
            font_size=7.5,
            font_color="#1A1A1A",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=20.0,
            corner_radius=0.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#CC2020",
            stroke="#AA1010",
            stroke_width=1.0,
            font_family="Courier New",
            font_size=6.0,
            font_color="#CC2020",
            padding=(2.0, 2.0),
            min_width=10.0,
            min_height=10.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#F8F0D0",
            stroke="#B0A880",
            stroke_width=0.5,
            font_family="Courier New",
            font_size=7.5,
            font_color="#333333",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=20.0,
            corner_radius=0.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#CC2020",
            width=0.8,
            style="solid",
            arrow="none",
            routing="straight",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#CC2020",
            width=0.5,
            style="solid",
            arrow="none",
            routing="straight",
            curvature=0.0,
            opacity=0.5,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#C8A868",
        stroke="#A88848",
        stroke_width=0.5,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=8.0,
        font_color="#4A3818",
        font_weight="bold",
        padding=10.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#C8A868"),
)

HOME_ASSISTANT_THEME = Theme(
    name="home_assistant",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#1C1C1C",
            stroke="#03A9F4",
            stroke_width=1.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFFFFF",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=22.0,
            corner_radius=8.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#03A9F4",
            stroke="#0288D1",
            stroke_width=2.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=36.0,
            min_height=24.0,
            corner_radius=8.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#1C1C1C",
            stroke="#44739E",
            stroke_width=1.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#9E9E9E",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=22.0,
            corner_radius=8.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#03A9F4",
            width=1.2,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#03A9F4",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.15,
        ),
        "back": EdgeStyle(
            color="#44739E",
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#44739E",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.2,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#111111",
        stroke="#333333",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=12.0,
        font_size=8.0,
        font_color="#03A9F4",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#111111"),
)
MOLESKINE_THEME = Theme(
    name="moleskine",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#F8F4E8",
            stroke="#D0C8B0",
            stroke_width=0.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#2A2218",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=18.0,
            corner_radius=1.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#F8F4E8",
            stroke="#2A2218",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=8.5,
            font_color="#2A2218",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=20.0,
            corner_radius=1.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#F8F4E8",
            stroke="#B0A890",
            stroke_width=0.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#5A5040",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=18.0,
            corner_radius=1.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#2A2218",
            width=0.6,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#2A2218",
            arrow_length=4.0,
            arrow_width=2.5,
            routing="bezier",
            curvature=0.15,
        ),
        "back": EdgeStyle(
            color="#A09880",
            width=0.4,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#A09880",
            arrow_length=3.5,
            arrow_width=2.0,
            routing="bezier",
            curvature=0.2,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F8F4E8",
        stroke="#D0C8B0",
        stroke_width=0.3,
        stroke_dash="dashed",
        corner_radius=0.0,
        font_size=8.0,
        font_color="#5A5040",
        font_weight="normal",
        padding=8.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#F8F4E8"),
)
YELLOW_LEGAL_THEME = Theme(
    name="yellow_legal",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#FFFACD",
            stroke="#DAA520",
            stroke_width=0.5,
            font_family="Courier New",
            font_size=8.0,
            font_color="#1A1A1A",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=18.0,
            corner_radius=0.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#FFFACD",
            stroke="#1A1A1A",
            stroke_width=1.5,
            font_family="Courier New",
            font_size=8.5,
            font_color="#1A1A1A",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=36.0,
            min_height=20.0,
            corner_radius=0.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#FFFACD",
            stroke="#B8860B",
            stroke_width=0.5,
            font_family="Courier New",
            font_size=8.0,
            font_color="#555555",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=18.0,
            corner_radius=0.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#4169E1",
            width=0.8,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#4169E1",
            arrow_length=4.5,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.15,
        ),
        "back": EdgeStyle(
            color="#4169E1",
            width=0.5,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#4169E1",
            arrow_length=4.0,
            arrow_width=2.5,
            routing="bezier",
            curvature=0.2,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#FFFACD",
        stroke="#DAA520",
        stroke_width=0.5,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=8.5,
        font_color="#1A1A1A",
        font_weight="bold",
        padding=10.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#FFFACD"),
)
NOTEBOOK_THEME = Theme(
    name="notebook",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#FFFFFF",
            stroke="#CCCCCC",
            stroke_width=0.5,
            font_family="Comic Sans MS",
            font_size=8.5,
            font_color="#333333",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=18.0,
            corner_radius=2.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#FFFFFF",
            stroke="#333333",
            stroke_width=1.5,
            font_family="Comic Sans MS",
            font_size=9.0,
            font_color="#333333",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=36.0,
            min_height=20.0,
            corner_radius=2.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#FFFFFF",
            stroke="#AAAAAA",
            stroke_width=0.5,
            font_family="Comic Sans MS",
            font_size=8.5,
            font_color="#666666",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=18.0,
            corner_radius=2.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#2B5CDB",
            width=0.8,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#2B5CDB",
            arrow_length=4.5,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.2,
        ),
        "back": EdgeStyle(
            color="#DB2B2B",
            width=0.6,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#DB2B2B",
            arrow_length=4.0,
            arrow_width=2.5,
            routing="bezier",
            curvature=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#FFFFFF",
        stroke="#CCCCCC",
        stroke_width=0.5,
        stroke_dash="dashed",
        corner_radius=4.0,
        font_size=9.0,
        font_color="#333333",
        font_weight="bold",
        padding=10.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)
CHRISTMAS_THEME = Theme(
    name="christmas",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#CC0000",
            stroke="#990000",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "input": NodeStyle(
            shape="star",
            fill="#FFD700",
            stroke="#DAA520",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#1A1A1A",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#006400",
            stroke="#004D00",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#006400",
            width=2.0,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.25,
        ),
        "back": EdgeStyle(
            color="#CC0000", width=1.0, style="solid", arrow="none", routing="bezier", curvature=0.3
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F0F8F0",
        stroke="#006400",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=6.0,
        font_size=9.0,
        font_color="#006400",
        font_weight="bold",
        padding=10.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)
ETCH_A_SKETCH_THEME = Theme(
    name="etch_a_sketch",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#C0C0C0",
            stroke="#A0A0A0",
            stroke_width=0.5,
            font_family="Courier New",
            font_size=7.5,
            font_color="#333333",
            padding=(4.0, 2.0),
            min_width=26.0,
            min_height=16.0,
            corner_radius=0.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#C0C0C0",
            stroke="#666666",
            stroke_width=1.0,
            font_family="Courier New",
            font_size=8.0,
            font_color="#1A1A1A",
            font_weight="bold",
            padding=(4.0, 2.0),
            min_width=28.0,
            min_height=18.0,
            corner_radius=0.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#C0C0C0",
            stroke="#888888",
            stroke_width=0.5,
            font_family="Courier New",
            font_size=7.5,
            font_color="#555555",
            padding=(4.0, 2.0),
            min_width=26.0,
            min_height=16.0,
            corner_radius=0.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#333333", width=1.5, style="solid", arrow="none", routing="ortho", curvature=0.0
        ),
        "back": EdgeStyle(
            color="#666666", width=0.8, style="solid", arrow="none", routing="ortho", curvature=0.0
        ),
    },
    cluster_style=ClusterStyle(
        fill="#CC0000",
        stroke="#AA0000",
        stroke_width=3.0,
        stroke_dash="solid",
        corner_radius=8.0,
        font_size=8.0,
        font_color="#FFFFFF",
        font_weight="bold",
        padding=14.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#C0C0C0"),
)
REDDIT_THEME = Theme(
    name="reddit",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#FFFFFF",
            stroke="#EDEFF1",
            stroke_width=1.0,
            font_family="Arial",
            font_size=8.5,
            font_color="#1A1A1B",
            padding=(6.0, 4.0),
            min_width=36.0,
            min_height=22.0,
            corner_radius=4.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#FF4500",
            stroke="#CC3700",
            stroke_width=2.0,
            font_family="Arial",
            font_size=8.5,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(6.0, 4.0),
            min_width=38.0,
            min_height=24.0,
            corner_radius=4.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#0079D3",
            stroke="#0060A8",
            stroke_width=1.5,
            font_family="Arial",
            font_size=8.5,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(6.0, 4.0),
            min_width=36.0,
            min_height=22.0,
            corner_radius=4.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#878A8C",
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#878A8C",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.15,
        ),
        "back": EdgeStyle(
            color="#DAE0E6",
            width=0.6,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#DAE0E6",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.2,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#DAE0E6",
        stroke="#EDEFF1",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=4.0,
        font_size=8.5,
        font_color="#1A1A1B",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#DAE0E6"),
)
CLOUDS_THEME = Theme(
    name="clouds",
    node_styles={
        "default": NodeStyle(
            shape="ellipse",
            fill="#FFFFFF",
            stroke="#D0D8E0",
            stroke_width=1.0,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#4A5A6A",
            padding=(6.0, 4.0),
            min_width=34.0,
            min_height=22.0,
            opacity=0.85,
        ),
        "input": NodeStyle(
            shape="ellipse",
            fill="#FFFFFF",
            stroke="#B0C0D0",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=8.5,
            font_color="#2A3A4A",
            font_weight="bold",
            padding=(6.0, 4.0),
            min_width=38.0,
            min_height=26.0,
            opacity=0.9,
        ),
        "output": NodeStyle(
            shape="ellipse",
            fill="#F0F4F8",
            stroke="#D8E0E8",
            stroke_width=0.8,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#6A7A8A",
            padding=(5.0, 3.0),
            min_width=28.0,
            min_height=18.0,
            opacity=0.7,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#A0B0C0",
            width=0.6,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.3,
            opacity=0.4,
        ),
        "back": EdgeStyle(
            color="#C0D0E0",
            width=0.4,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.4,
            opacity=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#E8F0F8",
        stroke="#C0D0E0",
        stroke_width=0.3,
        stroke_dash="solid",
        corner_radius=16.0,
        font_size=8.0,
        font_color="#6A7A8A",
        font_weight="normal",
        padding=10.0,
        opacity=0.25,
    ),
    graph_style=GraphStyle(background_color="#87CEEB"),
)
TWITTER_THEME = Theme(
    name="twitter",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#FFFFFF",
            stroke="#CFD9DE",
            stroke_width=1.0,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#0F1419",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#1D9BF0",
            stroke="#1A8CD8",
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#F7F9F9",
            stroke="#CFD9DE",
            stroke_width=1.0,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#536471",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#1D9BF0",
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#1D9BF0",
            arrow_length=4.5,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.2,
        ),
        "back": EdgeStyle(
            color="#CFD9DE",
            width=0.6,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#CFD9DE",
            arrow_length=4.0,
            arrow_width=2.5,
            routing="bezier",
            curvature=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F7F9F9",
        stroke="#EFF3F4",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=12.0,
        font_size=8.0,
        font_color="#0F1419",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)
HERALDRY_THEME = Theme(
    name="heraldry",
    node_styles={
        "default": NodeStyle(
            shape="pentagon",
            fill="#1A1A6B",
            stroke="#C8A030",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#C8A030",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "input": NodeStyle(
            shape="pentagon",
            fill="#8B1A1A",
            stroke="#C8A030",
            stroke_width=3.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#E8D898",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=30.0,
            min_height=30.0,
        ),
        "output": NodeStyle(
            shape="pentagon",
            fill="#FFFFFF",
            stroke="#C8A030",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#1A1A6B",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=26.0,
            min_height=26.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#C8A030",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#C8A030",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="straight",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#8A7020",
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#8A7020",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="straight",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F0E8D0",
        stroke="#C8A030",
        stroke_width=2.5,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=9.0,
        font_color="#1A1A6B",
        font_weight="bold",
        padding=12.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#F0E8D0"),
)
MAZE_THEME = Theme(
    name="maze",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#FFFFFF",
            stroke="#333333",
            stroke_width=1.0,
            font_family="Arial",
            font_size=7.0,
            font_color="#333333",
            padding=(3.0, 3.0),
            min_width=12.0,
            min_height=12.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#00CC00",
            stroke="#009900",
            stroke_width=2.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=18.0,
            min_height=18.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#CC0000",
            stroke="#990000",
            stroke_width=2.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=18.0,
            min_height=18.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#333333", width=2.5, style="solid", arrow="none", routing="ortho", curvature=0.0
        ),
        "back": EdgeStyle(
            color="#CCCCCC", width=1.0, style="dashed", arrow="none", routing="ortho", curvature=0.0
        ),
    },
    cluster_style=ClusterStyle(
        fill="#FFFFFF",
        stroke="#333333",
        stroke_width=2.0,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=8.0,
        font_color="#333333",
        font_weight="bold",
        padding=8.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)
CROP_CIRCLES_THEME = Theme(
    name="crop_circles",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#8B9E3A",
            stroke="#6B7E2A",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=7.0,
            font_color="#D8E8A0",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#C8D870",
            stroke="#A8B850",
            stroke_width=2.5,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#2A3A08",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#A0B848",
            stroke="#80A030",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=7.0,
            font_color="#2A3808",
            padding=(3.0, 3.0),
            min_width=16.0,
            min_height=16.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#C8D870",
            width=1.0,
            style="solid",
            arrow="none",
            routing="straight",
            curvature=0.0,
            opacity=0.6,
        ),
        "back": EdgeStyle(
            color="#A8B850",
            width=0.5,
            style="solid",
            arrow="none",
            routing="straight",
            curvature=0.0,
            opacity=0.4,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#6B7E2A",
        stroke="#8B9E3A",
        stroke_width=0.5,
        stroke_dash="solid",
        corner_radius=20.0,
        font_size=7.5,
        font_color="#D8E8A0",
        font_weight="normal",
        padding=10.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#5A6E28"),
)
SHEET_MUSIC_THEME = Theme(
    name="sheet_music",
    node_styles={
        "default": NodeStyle(
            shape="ellipse",
            fill="#1A1A1A",
            stroke="#000000",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=7.0,
            font_color="#FFFFFF",
            padding=(3.0, 2.0),
            min_width=16.0,
            min_height=12.0,
        ),
        "input": NodeStyle(
            shape="ellipse",
            fill="#1A1A1A",
            stroke="#000000",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 3.0),
            min_width=20.0,
            min_height=14.0,
        ),
        "output": NodeStyle(
            shape="ellipse",
            fill="#FFFFFF",
            stroke="#000000",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=7.0,
            font_color="#000000",
            padding=(3.0, 2.0),
            min_width=16.0,
            min_height=12.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#000000",
            width=0.8,
            style="solid",
            arrow="none",
            routing="straight",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#888888", width=0.5, style="solid", arrow="none", routing="bezier", curvature=0.3
        ),
    },
    cluster_style=ClusterStyle(
        fill="#FFFFF0",
        stroke="#CCCCCC",
        stroke_width=0.5,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=8.0,
        font_color="#333333",
        font_weight="bold",
        padding=8.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#FFFFF0"),
)
MARKOV_THEME = Theme(
    name="markov",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#FFFFFF",
            stroke="#333333",
            stroke_width=2.0,
            font_family="Times New Roman",
            font_size=9.0,
            font_color="#000000",
            font_style="italic",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#E8E8E8",
            stroke="#000000",
            stroke_width=2.5,
            font_family="Times New Roman",
            font_size=9.0,
            font_color="#000000",
            font_style="italic",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=30.0,
            min_height=30.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#FFFFFF",
            stroke="#333333",
            stroke_width=2.0,
            font_family="Times New Roman",
            font_size=9.0,
            font_color="#333333",
            font_style="italic",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#000000",
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#000000",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.3,
        ),
        "back": EdgeStyle(
            color="#666666",
            width=0.6,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#666666",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.4,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F8F8F8",
        stroke="#CCCCCC",
        stroke_width=0.5,
        stroke_dash="dashed",
        corner_radius=0.0,
        font_size=9.0,
        font_color="#333333",
        font_weight="normal",
        padding=8.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)
CALLIGRAPHY_THEME = Theme(
    name="calligraphy",
    node_styles={
        "default": NodeStyle(
            shape="ellipse",
            fill="#F8F0E0",
            stroke="#2A1A0A",
            stroke_width=0.8,
            font_family="Georgia",
            font_size=9.0,
            font_color="#2A1A0A",
            font_style="italic",
            padding=(6.0, 3.0),
            min_width=32.0,
            min_height=18.0,
        ),
        "input": NodeStyle(
            shape="ellipse",
            fill="#F8F0E0",
            stroke="#2A1A0A",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=10.0,
            font_color="#2A1A0A",
            font_style="italic",
            font_weight="bold",
            padding=(6.0, 3.0),
            min_width=36.0,
            min_height=20.0,
        ),
        "output": NodeStyle(
            shape="ellipse",
            fill="#F8F0E0",
            stroke="#6B5038",
            stroke_width=0.5,
            font_family="Georgia",
            font_size=8.5,
            font_color="#6B5038",
            font_style="italic",
            padding=(6.0, 3.0),
            min_width=30.0,
            min_height=16.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#2A1A0A",
            width=0.8,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#2A1A0A",
            arrow_length=5.0,
            arrow_width=2.5,
            routing="bezier",
            curvature=0.25,
        ),
        "back": EdgeStyle(
            color="#8B7355",
            width=0.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#8B7355",
            arrow_length=4.0,
            arrow_width=2.0,
            routing="bezier",
            curvature=0.35,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F8F0E0",
        stroke="#C0A878",
        stroke_width=0.3,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=9.0,
        font_color="#6B5038",
        font_weight="normal",
        padding=8.0,
        opacity=0.2,
    ),
    graph_style=GraphStyle(background_color="#F8F0E0"),
)
BALDERDASH_THEME = Theme(
    name="balderdash",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#2A0845",
            stroke="#6C2EB9",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.5,
            font_color="#E8D0F0",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=22.0,
            corner_radius=4.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#6C2EB9",
            stroke="#4A1A88",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=9.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=36.0,
            min_height=24.0,
            corner_radius=4.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#FF6F00",
            stroke="#D05A00",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.5,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=22.0,
            corner_radius=4.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#6C2EB9",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#6C2EB9",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.2,
        ),
        "back": EdgeStyle(
            color="#FF6F00",
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#FF6F00",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#1A0830",
        stroke="#6C2EB9",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=6.0,
        font_size=9.0,
        font_color="#E8D0F0",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#1A0830"),
)
BALLOONS_THEME = Theme(
    name="balloons",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#FF6B6B",
            stroke="#E05050",
            stroke_width=1.0,
            font_family="Arial",
            font_size=7.5,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#4ECDC4",
            stroke="#38B8B0",
            stroke_width=1.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#FFE66D",
            stroke="#E8D050",
            stroke_width=1.0,
            font_family="Arial",
            font_size=7.5,
            font_color="#4A4A1A",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#CCCCCC", width=0.5, style="solid", arrow="none", routing="bezier", curvature=0.4
        ),
        "back": EdgeStyle(
            color="#DDDDDD", width=0.3, style="solid", arrow="none", routing="bezier", curvature=0.5
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F0F8FF",
        stroke="#D0E0F0",
        stroke_width=0.5,
        stroke_dash="solid",
        corner_radius=14.0,
        font_size=8.0,
        font_color="#6A8AAA",
        font_weight="bold",
        padding=10.0,
        opacity=0.25,
    ),
    graph_style=GraphStyle(background_color="#87CEEB"),
)
CHAIN_LINK_THEME = Theme(
    name="chain_link",
    node_styles={
        "default": NodeStyle(
            shape="ellipse",
            fill="#808080",
            stroke="#606060",
            stroke_width=2.5,
            font_family="Arial",
            font_size=7.5,
            font_color="#E0E0E0",
            font_weight="bold",
            padding=(4.0, 3.0),
            min_width=22.0,
            min_height=16.0,
        ),
        "input": NodeStyle(
            shape="ellipse",
            fill="#A0A0A0",
            stroke="#707070",
            stroke_width=3.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 4.0),
            min_width=26.0,
            min_height=18.0,
        ),
        "output": NodeStyle(
            shape="ellipse",
            fill="#606060",
            stroke="#484848",
            stroke_width=2.0,
            font_family="Arial",
            font_size=7.5,
            font_color="#D0D0D0",
            padding=(4.0, 3.0),
            min_width=20.0,
            min_height=14.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#707070",
            width=3.0,
            style="solid",
            arrow="none",
            routing="straight",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#909090",
            width=1.5,
            style="solid",
            arrow="none",
            routing="straight",
            curvature=0.0,
            opacity=0.6,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#2A2A2A",
        stroke="#4A4A4A",
        stroke_width=2.0,
        stroke_dash="solid",
        corner_radius=4.0,
        font_size=8.0,
        font_color="#A0A0A0",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#1A1A1A"),
)
MYSPACE_THEME = Theme(
    name="myspace",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#003366",
            stroke="#004488",
            stroke_width=1.5,
            font_family="Verdana",
            font_size=8.0,
            font_color="#FFFFFF",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=2.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#FF6600",
            stroke="#DD5500",
            stroke_width=2.0,
            font_family="Verdana",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=22.0,
            corner_radius=2.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#660099",
            stroke="#440077",
            stroke_width=1.5,
            font_family="Verdana",
            font_size=8.0,
            font_color="#FFFFFF",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=2.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#FF6600",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#FF6600",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.2,
        ),
        "back": EdgeStyle(
            color="#003366",
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#003366",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#001A33",
        stroke="#003366",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=4.0,
        font_size=8.0,
        font_color="#FF6600",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#000000"),
)
AOL_THEME = Theme(
    name="aol",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#EEEEEE",
            stroke="#336699",
            stroke_width=1.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#003366",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=3.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#336699",
            stroke="#224477",
            stroke_width=2.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=22.0,
            corner_radius=3.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#FFCC00",
            stroke="#DDAA00",
            stroke_width=1.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#003366",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=3.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#336699",
            width=1.2,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#336699",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.2,
        ),
        "back": EdgeStyle(
            color="#999999",
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#999999",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#DDDDDD",
        stroke="#336699",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=4.0,
        font_size=8.5,
        font_color="#003366",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)
INSTAGRAM_THEME = Theme(
    name="instagram",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#FFFFFF",
            stroke="#DBDBDB",
            stroke_width=1.0,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#262626",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=22.0,
            corner_radius=8.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#E1306C",
            stroke="#C02060",
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=36.0,
            min_height=24.0,
            corner_radius=8.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#FCAF45",
            stroke="#E09030",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=22.0,
            corner_radius=8.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#833AB4",
            width=1.2,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#833AB4",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.2,
        ),
        "back": EdgeStyle(
            color="#DBDBDB",
            width=0.6,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#DBDBDB",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#FAFAFA",
        stroke="#DBDBDB",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=12.0,
        font_size=8.0,
        font_color="#262626",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#FAFAFA"),
)
TIKTOK_THEME = Theme(
    name="tiktok",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#010101",
            stroke="#25F4EE",
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=20.0,
            corner_radius=4.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#010101",
            stroke="#FE2C55",
            stroke_width=2.5,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FE2C55",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=22.0,
            corner_radius=4.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#010101",
            stroke="#FFFFFF",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FFFFFF",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=20.0,
            corner_radius=4.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#25F4EE",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#25F4EE",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.2,
        ),
        "back": EdgeStyle(
            color="#FE2C55",
            width=0.8,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#FE2C55",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#010101",
        stroke="#333333",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=6.0,
        font_size=8.0,
        font_color="#FFFFFF",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#010101"),
)
OREGON_TRAIL_THEME = Theme(
    name="oregon_trail",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#000000",
            stroke="#33FF33",
            stroke_width=1.0,
            font_family="Courier New",
            font_size=8.0,
            font_color="#33FF33",
            padding=(4.0, 2.0),
            min_width=32.0,
            min_height=16.0,
            corner_radius=0.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#003300",
            stroke="#33FF33",
            stroke_width=2.0,
            font_family="Courier New",
            font_size=8.0,
            font_color="#33FF33",
            font_weight="bold",
            padding=(4.0, 2.0),
            min_width=34.0,
            min_height=18.0,
            corner_radius=0.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#000000",
            stroke="#22AA22",
            stroke_width=1.0,
            font_family="Courier New",
            font_size=8.0,
            font_color="#22AA22",
            padding=(4.0, 2.0),
            min_width=32.0,
            min_height=16.0,
            corner_radius=0.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#33FF33",
            width=1.0,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#33FF33",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.15,
        ),
        "back": EdgeStyle(
            color="#116611",
            width=0.6,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#116611",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.2,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#000000",
        stroke="#22AA22",
        stroke_width=1.0,
        stroke_dash="dashed",
        corner_radius=0.0,
        font_size=8.0,
        font_color="#33FF33",
        font_weight="bold",
        padding=8.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#000000"),
)
SPRING_MASS_THEME = Theme(
    name="spring_mass",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#4A90D0",
            stroke="#3070B0",
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#E04040",
            stroke="#C02020",
            stroke_width=2.5,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=30.0,
            min_height=30.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#40A040",
            stroke="#208020",
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#888888",
            width=1.5,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.15,
        ),
        "back": EdgeStyle(
            color="#AAAAAA",
            width=0.8,
            style="dashed",
            arrow="none",
            routing="bezier",
            curvature=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F0F0F0",
        stroke="#CCCCCC",
        stroke_width=1.0,
        stroke_dash="dashed",
        corner_radius=4.0,
        font_size=8.0,
        font_color="#333333",
        font_weight="bold",
        padding=10.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)
GOOGLE_MAPS_THEME = Theme(
    name="google_maps",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#EA4335",
            stroke="#C5221F",
            stroke_width=1.5,
            font_family="Arial",
            font_size=7.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(3.0, 3.0),
            min_width=16.0,
            min_height=16.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#FBBC04",
            stroke="#E0A800",
            stroke_width=2.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#1A1A1A",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#34A853",
            stroke="#2A8A44",
            stroke_width=1.5,
            font_family="Arial",
            font_size=7.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(3.0, 3.0),
            min_width=14.0,
            min_height=14.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#4285F4", width=2.0, style="solid", arrow="none", routing="bezier", curvature=0.2
        ),
        "back": EdgeStyle(
            color="#9AA0A6",
            width=1.0,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F1F3F4",
        stroke="#DADCE0",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=8.0,
        font_size=8.0,
        font_color="#3C4043",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#E8F0E8"),
)

THEME_REGISTRY["home_assistant"] = HOME_ASSISTANT_THEME
THEME_REGISTRY["moleskine"] = MOLESKINE_THEME
THEME_REGISTRY["yellow_legal"] = YELLOW_LEGAL_THEME
THEME_REGISTRY["notebook"] = NOTEBOOK_THEME
THEME_REGISTRY["christmas"] = CHRISTMAS_THEME
THEME_REGISTRY["etch_a_sketch"] = ETCH_A_SKETCH_THEME
THEME_REGISTRY["reddit"] = REDDIT_THEME
THEME_REGISTRY["clouds"] = CLOUDS_THEME
THEME_REGISTRY["twitter"] = TWITTER_THEME
THEME_REGISTRY["heraldry"] = HERALDRY_THEME
THEME_REGISTRY["maze"] = MAZE_THEME
THEME_REGISTRY["crop_circles"] = CROP_CIRCLES_THEME
THEME_REGISTRY["sheet_music"] = SHEET_MUSIC_THEME
THEME_REGISTRY["markov"] = MARKOV_THEME
THEME_REGISTRY["calligraphy"] = CALLIGRAPHY_THEME
THEME_REGISTRY["balderdash"] = BALDERDASH_THEME
THEME_REGISTRY["balloons"] = BALLOONS_THEME
THEME_REGISTRY["chain_link"] = CHAIN_LINK_THEME
THEME_REGISTRY["myspace"] = MYSPACE_THEME
THEME_REGISTRY["aol"] = AOL_THEME
THEME_REGISTRY["instagram"] = INSTAGRAM_THEME
THEME_REGISTRY["tiktok"] = TIKTOK_THEME
THEME_REGISTRY["oregon_trail"] = OREGON_TRAIL_THEME
THEME_REGISTRY["spring_mass"] = SPRING_MASS_THEME
WIKIPEDIA_THEME = Theme(
    name="wikipedia",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#FFFFFF",
            stroke="#A2A9B1",
            stroke_width=1.0,
            font_family="Georgia",
            font_size=8.5,
            font_color="#202122",
            padding=(6.0, 3.0),
            min_width=38.0,
            min_height=20.0,
            corner_radius=2.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#FFFFFF",
            stroke="#3366CC",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=8.5,
            font_color="#3366CC",
            font_weight="bold",
            padding=(6.0, 3.0),
            min_width=40.0,
            min_height=22.0,
            corner_radius=2.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#F8F9FA",
            stroke="#A2A9B1",
            stroke_width=1.0,
            font_family="Georgia",
            font_size=8.5,
            font_color="#54595D",
            padding=(6.0, 3.0),
            min_width=38.0,
            min_height=20.0,
            corner_radius=2.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#3366CC",
            width=0.8,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#3366CC",
            arrow_length=4.5,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.15,
        ),
        "back": EdgeStyle(
            color="#A2A9B1",
            width=0.5,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#A2A9B1",
            arrow_length=4.0,
            arrow_width=2.5,
            routing="bezier",
            curvature=0.2,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F8F9FA",
        stroke="#A2A9B1",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=2.0,
        font_size=9.0,
        font_color="#202122",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)
NATURE_JOURNAL_THEME = Theme(
    name="nature_journal",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#FFFFFF",
            stroke="#C7282D",
            stroke_width=1.0,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#333333",
            padding=(5.0, 3.0),
            min_width=36.0,
            min_height=20.0,
            corner_radius=0.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#C7282D",
            stroke="#A01A20",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=38.0,
            min_height=22.0,
            corner_radius=0.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#F5F5F5",
            stroke="#CCCCCC",
            stroke_width=0.8,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#666666",
            padding=(5.0, 3.0),
            min_width=36.0,
            min_height=20.0,
            corner_radius=0.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#333333",
            width=0.8,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#333333",
            arrow_length=5.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.15,
        ),
        "back": EdgeStyle(
            color="#999999",
            width=0.5,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#999999",
            arrow_length=4.0,
            arrow_width=2.5,
            routing="bezier",
            curvature=0.2,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F8F8F8",
        stroke="#DDDDDD",
        stroke_width=0.5,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=8.5,
        font_color="#C7282D",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

BRUTALIST_THEME = Theme(
    name="brutalist",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#808080",
            stroke="#5A5A5A",
            stroke_width=3.0,
            font_family="Helvetica",
            font_size=9.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=26.0,
            corner_radius=0.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#5A5A5A",
            stroke="#3A3A3A",
            stroke_width=4.0,
            font_family="Helvetica",
            font_size=9.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=38.0,
            min_height=30.0,
            corner_radius=0.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#9A9A9A",
            stroke="#6A6A6A",
            stroke_width=2.5,
            font_family="Helvetica",
            font_size=9.0,
            font_color="#1A1A1A",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=26.0,
            corner_radius=0.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#3A3A3A",
            width=3.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#3A3A3A",
            arrow_length=6.0,
            arrow_width=5.0,
            routing="ortho",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#6A6A6A",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#6A6A6A",
            arrow_length=5.0,
            arrow_width=4.0,
            routing="ortho",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#6A6A6A",
        stroke="#4A4A4A",
        stroke_width=3.0,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=10.0,
        font_color="#FFFFFF",
        font_weight="bold",
        padding=14.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#A0A0A0"),
)
LOVECRAFT_THEME = Theme(
    name="lovecraft",
    node_styles={
        "default": NodeStyle(
            shape="ellipse",
            fill="#0A1828",
            stroke="#1A4040",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#5A8878",
            font_style="italic",
            padding=(5.0, 4.0),
            min_width=28.0,
            min_height=22.0,
        ),
        "input": NodeStyle(
            shape="ellipse",
            fill="#1A3050",
            stroke="#2A6060",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#80C8A0",
            font_style="italic",
            font_weight="bold",
            padding=(5.0, 4.0),
            min_width=30.0,
            min_height=24.0,
        ),
        "output": NodeStyle(
            shape="ellipse",
            fill="#08101A",
            stroke="#183030",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#406858",
            font_style="italic",
            padding=(5.0, 4.0),
            min_width=26.0,
            min_height=20.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#1A4040",
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#1A4040",
            arrow_length=5.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.45,
            opacity=0.6,
        ),
        "back": EdgeStyle(
            color="#0A2828",
            width=0.6,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#0A2828",
            arrow_length=4.0,
            arrow_width=2.5,
            routing="bezier",
            curvature=0.55,
            opacity=0.4,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#06101A",
        stroke="#1A3030",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=8.0,
        font_color="#305848",
        font_weight="normal",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#040A10"),
)
SATELLITE_THEME = Theme(
    name="satellite",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#1A4A28",
            stroke="#30A848",
            stroke_width=1.5,
            font_family="Courier New",
            font_size=7.5,
            font_color="#90E8A0",
            padding=(4.0, 4.0),
            min_width=20.0,
            min_height=20.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#A83020",
            stroke="#E84030",
            stroke_width=2.0,
            font_family="Courier New",
            font_size=8.0,
            font_color="#FFD0C0",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#2040A0",
            stroke="#4070E0",
            stroke_width=1.5,
            font_family="Courier New",
            font_size=7.5,
            font_color="#A0C0F0",
            padding=(4.0, 4.0),
            min_width=20.0,
            min_height=20.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#60A060",
            width=0.8,
            style="solid",
            arrow="none",
            routing="straight",
            curvature=0.0,
            opacity=0.5,
        ),
        "back": EdgeStyle(
            color="#408040",
            width=0.5,
            style="solid",
            arrow="none",
            routing="straight",
            curvature=0.0,
            opacity=0.3,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#0A2010",
        stroke="#204028",
        stroke_width=0.5,
        stroke_dash="solid",
        corner_radius=4.0,
        font_size=7.5,
        font_color="#60A068",
        font_weight="bold",
        padding=8.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#0A1A10"),
)
NAUTICAL_THEME = Theme(
    name="nautical",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#FFFFFF",
            stroke="#003366",
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#003366",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=18.0,
            min_height=18.0,
        ),
        "input": NodeStyle(
            shape="triangle",
            fill="#CC0000",
            stroke="#990000",
            stroke_width=2.5,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=24.0,
            min_height=24.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#003366",
            stroke="#002244",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#C0D8E8",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=16.0,
            min_height=16.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#003366",
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#003366",
            arrow_length=5.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.2,
        ),
        "back": EdgeStyle(
            color="#6688AA",
            width=0.6,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#6688AA",
            arrow_length=4.0,
            arrow_width=2.5,
            routing="bezier",
            curvature=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#E8E0C8",
        stroke="#003366",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=8.5,
        font_color="#003366",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#E8E0C8"),
)
WINE_LABEL_THEME = Theme(
    name="wine_label",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#F8F0E8",
            stroke="#722F37",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#722F37",
            font_style="italic",
            padding=(6.0, 3.0),
            min_width=36.0,
            min_height=20.0,
            corner_radius=1.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#722F37",
            stroke="#5A1A24",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.5,
            font_color="#F0E0D0",
            font_weight="bold",
            padding=(6.0, 3.0),
            min_width=38.0,
            min_height=22.0,
            corner_radius=1.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#F8F0E8",
            stroke="#C8A878",
            stroke_width=1.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#8B6D4C",
            font_style="italic",
            padding=(6.0, 3.0),
            min_width=36.0,
            min_height=20.0,
            corner_radius=1.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#722F37",
            width=0.8,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#722F37",
            arrow_length=4.5,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.15,
        ),
        "back": EdgeStyle(
            color="#C8A878",
            width=0.5,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#C8A878",
            arrow_length=4.0,
            arrow_width=2.5,
            routing="bezier",
            curvature=0.2,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F8F0E8",
        stroke="#722F37",
        stroke_width=0.8,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=9.0,
        font_color="#722F37",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#F8F0E8"),
)
MARBLE_THEME = Theme(
    name="marble",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#E8E4E0",
            stroke="#C0B8B0",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#4A4440",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=22.0,
            corner_radius=2.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#D0C8C0",
            stroke="#A09890",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#2A2420",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=24.0,
            corner_radius=2.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#F0ECE8",
            stroke="#D0C8C0",
            stroke_width=1.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#6A6460",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=22.0,
            corner_radius=2.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#908880",
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#908880",
            arrow_length=5.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.15,
        ),
        "back": EdgeStyle(
            color="#C0B8B0",
            width=0.6,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#C0B8B0",
            arrow_length=4.0,
            arrow_width=2.5,
            routing="bezier",
            curvature=0.2,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#E0D8D0",
        stroke="#B0A8A0",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=3.0,
        font_size=8.5,
        font_color="#4A4440",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#F0ECE8"),
)
VALENTINE_THEME = Theme(
    name="valentine",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#E8284A",
            stroke="#C01838",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#FF69B4",
            stroke="#E04898",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#FFB6C1",
            stroke="#E0A0A8",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#8B2050",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#E8284A",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#E8284A",
            arrow_length=5.0,
            arrow_width=4.0,
            routing="bezier",
            curvature=0.3,
        ),
        "back": EdgeStyle(
            color="#FFB6C1",
            width=0.8,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#FFB6C1",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.35,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#FFF0F5",
        stroke="#FFB6C1",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=12.0,
        font_size=9.0,
        font_color="#E8284A",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#FFF5F8"),
)
HALLOWEEN_THEME = Theme(
    name="halloween",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#FF6600",
            stroke="#DD5500",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#000000",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#000000",
            stroke="#333333",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#FF6600",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#6B0080",
            stroke="#4A0060",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#E0C0F0",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#FF6600",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#FF6600",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.25,
        ),
        "back": EdgeStyle(
            color="#333333",
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#333333",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.3,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#1A0A20",
        stroke="#4A0060",
        stroke_width=2.0,
        stroke_dash="solid",
        corner_radius=6.0,
        font_size=9.0,
        font_color="#FF6600",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#0A0A0A"),
)
LETTERPRESS_THEME = Theme(
    name="letterpress",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#F4ECD8",
            stroke="#2A2218",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=9.0,
            font_color="#2A2218",
            padding=(6.0, 3.0),
            min_width=36.0,
            min_height=20.0,
            corner_radius=0.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#F4ECD8",
            stroke="#2A2218",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=10.0,
            font_color="#2A2218",
            font_weight="bold",
            padding=(6.0, 3.0),
            min_width=38.0,
            min_height=22.0,
            corner_radius=0.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#F4ECD8",
            stroke="#5A4A38",
            stroke_width=1.0,
            font_family="Georgia",
            font_size=8.5,
            font_color="#5A4A38",
            padding=(6.0, 3.0),
            min_width=36.0,
            min_height=20.0,
            corner_radius=0.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#2A2218",
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#2A2218",
            arrow_length=5.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.15,
        ),
        "back": EdgeStyle(
            color="#8B7355",
            width=0.6,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#8B7355",
            arrow_length=4.0,
            arrow_width=2.5,
            routing="bezier",
            curvature=0.2,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F4ECD8",
        stroke="#2A2218",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=9.5,
        font_color="#2A2218",
        font_weight="bold",
        padding=10.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#F4ECD8"),
)
RISOGRAPH_THEME = Theme(
    name="risograph",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#FF6B8A",
            stroke="#E05070",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#0078BF",
            stroke="#0060A0",
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#FFB521",
            stroke="#E09818",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#1A1A1A",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#0078BF",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#0078BF",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.2,
            opacity=0.7,
        ),
        "back": EdgeStyle(
            color="#FF6B8A",
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#FF6B8A",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.25,
            opacity=0.6,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#FFF5E8",
        stroke="#FFB521",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=6.0,
        font_size=8.5,
        font_color="#0078BF",
        font_weight="bold",
        padding=10.0,
        opacity=0.35,
    ),
    graph_style=GraphStyle(background_color="#FFF8F0"),
)
JAZZ_THEME = Theme(
    name="jazz",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#1A1A2E",
            stroke="#C8A030",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#C8A030",
            font_style="italic",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#C8A030",
            stroke="#A88020",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#1A1A2E",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#1A1A2E",
            stroke="#8B6D30",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#A08840",
            font_style="italic",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#C8A030",
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#C8A030",
            arrow_length=5.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.35,
        ),
        "back": EdgeStyle(
            color="#5A4A28",
            width=0.6,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#5A4A28",
            arrow_length=4.0,
            arrow_width=2.5,
            routing="bezier",
            curvature=0.45,
            opacity=0.5,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#0E0E1A",
        stroke="#3A3048",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=6.0,
        font_size=8.5,
        font_color="#C8A030",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#0E0E1A"),
)

HOLLYWOOD_THEME = Theme(
    name="hollywood",
    node_styles={
        "default": NodeStyle(
            shape="star",
            fill="#FFD700",
            stroke="#DAA520",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#1A1A1A",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "input": NodeStyle(
            shape="star",
            fill="#FFD700",
            stroke="#B8860B",
            stroke_width=3.0,
            font_family="Georgia",
            font_size=9.0,
            font_color="#1A1A1A",
            font_weight="bold",
            padding=(6.0, 6.0),
            min_width=30.0,
            min_height=30.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#1A1A1A",
            stroke="#FFD700",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#FFD700",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#FFD700",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#FFD700",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.2,
        ),
        "back": EdgeStyle(
            color="#8B7028",
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#8B7028",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#1A1A1A",
        stroke="#FFD700",
        stroke_width=2.0,
        stroke_dash="solid",
        corner_radius=4.0,
        font_size=9.0,
        font_color="#FFD700",
        font_weight="bold",
        padding=12.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#1A1A1A"),
)

FROSTED_THEME = Theme(
    name="frosted",
    node_styles={
        "default": NodeStyle(
            shape="ellipse",
            fill="#D8E8F0",
            stroke="#B0C8D8",
            stroke_width=0.5,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#4A6878",
            padding=(5.0, 4.0),
            min_width=28.0,
            min_height=20.0,
            opacity=0.6,
        ),
        "input": NodeStyle(
            shape="ellipse",
            fill="#C0D8E8",
            stroke="#90B0C8",
            stroke_width=1.0,
            font_family="Helvetica",
            font_size=8.5,
            font_color="#2A4858",
            font_weight="bold",
            padding=(5.0, 4.0),
            min_width=32.0,
            min_height=24.0,
            opacity=0.7,
        ),
        "output": NodeStyle(
            shape="ellipse",
            fill="#E0EAF0",
            stroke="#C0D0D8",
            stroke_width=0.3,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#6A8898",
            padding=(5.0, 3.0),
            min_width=26.0,
            min_height=18.0,
            opacity=0.5,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#8AA8B8",
            width=1.5,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.2,
            opacity=0.5,
        ),
        "back": EdgeStyle(
            color="#A0B8C8",
            width=0.8,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.3,
            opacity=0.3,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#C8D8E0",
        stroke="#A0B8C8",
        stroke_width=0.3,
        stroke_dash="solid",
        corner_radius=12.0,
        font_size=8.0,
        font_color="#5A7888",
        font_weight="normal",
        padding=10.0,
        opacity=0.2,
    ),
    graph_style=GraphStyle(background_color="#C8D8E0"),
)

FROSTING_THEME = Theme(
    name="frosting",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#FFB6C1",
            stroke="#E898A8",
            stroke_width=2.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#FFFFFF",
            stroke="#FFD700",
            stroke_width=3.0,
            font_family="Arial",
            font_size=8.5,
            font_color="#D87093",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=30.0,
            min_height=30.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#98D8C8",
            stroke="#78B8A8",
            stroke_width=2.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#FFB6C1", width=3.0, style="solid", arrow="none", routing="bezier", curvature=0.3
        ),
        "back": EdgeStyle(
            color="#98D8C8",
            width=2.0,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.35,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#FFF0F5",
        stroke="#FFB6C1",
        stroke_width=2.0,
        stroke_dash="solid",
        corner_radius=12.0,
        font_size=9.0,
        font_color="#D87093",
        font_weight="bold",
        padding=12.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#FFF8F0"),
)
LEGO_THEME = Theme(
    name="lego",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#D01012",
            stroke="#A00E10",
            stroke_width=2.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
            corner_radius=2.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#F5CD2F",
            stroke="#D0AE20",
            stroke_width=3.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#1A1A1A",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
            corner_radius=2.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#0057A8",
            stroke="#004088",
            stroke_width=2.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
            corner_radius=2.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#4A4A4A", width=2.5, style="solid", arrow="none", routing="ortho", curvature=0.0
        ),
        "back": EdgeStyle(
            color="#808080", width=1.5, style="solid", arrow="none", routing="ortho", curvature=0.0
        ),
    },
    cluster_style=ClusterStyle(
        fill="#237841",
        stroke="#1A5A30",
        stroke_width=2.5,
        stroke_dash="solid",
        corner_radius=2.0,
        font_size=9.0,
        font_color="#FFFFFF",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#237841"),
)
CRAYON_THEME = Theme(
    name="crayon",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#FC2847",
            stroke="#E01830",
            stroke_width=2.0,
            font_family="Comic Sans MS",
            font_size=8.5,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#1F75FE",
            stroke="#1860D0",
            stroke_width=2.5,
            font_family="Comic Sans MS",
            font_size=9.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#1CAC78",
            stroke="#148A60",
            stroke_width=2.0,
            font_family="Comic Sans MS",
            font_size=8.5,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#EE204D",
            width=2.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#EE204D",
            arrow_length=5.0,
            arrow_width=4.0,
            routing="bezier",
            curvature=0.25,
        ),
        "back": EdgeStyle(
            color="#FCE883",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#FCE883",
            arrow_length=4.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.3,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#FFEB99",
        stroke="#FCE883",
        stroke_width=2.0,
        stroke_dash="solid",
        corner_radius=8.0,
        font_size=9.0,
        font_color="#B4674D",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#FFF5D7"),
)
QUILT_THEME = Theme(
    name="quilt",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#C06050",
            stroke="#8B4040",
            stroke_width=1.0,
            font_family="Georgia",
            font_size=7.5,
            font_color="#FFE8E0",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
            corner_radius=0.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#4878A0",
            stroke="#305878",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#D0E0F0",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=26.0,
            min_height=26.0,
            corner_radius=0.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#608848",
            stroke="#406828",
            stroke_width=1.0,
            font_family="Georgia",
            font_size=7.5,
            font_color="#D8F0C8",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
            corner_radius=0.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#F8F0E0",
            width=1.5,
            style="dashed",
            arrow="none",
            routing="straight",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#E0D8C8",
            width=1.0,
            style="dashed",
            arrow="none",
            routing="straight",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F0E8D8",
        stroke="#D0C8B0",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=8.0,
        font_color="#5A4A38",
        font_weight="bold",
        padding=8.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#F0E8D8"),
)
FIREWORKS_THEME = Theme(
    name="fireworks",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#FFD700",
            stroke="#FFA500",
            stroke_width=2.0,
            font_family="Arial",
            font_size=7.5,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(3.0, 3.0),
            min_width=16.0,
            min_height=16.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#FF1493",
            stroke="#E01080",
            stroke_width=2.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#00CED1",
            stroke="#00A8B0",
            stroke_width=2.0,
            font_family="Arial",
            font_size=7.5,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(3.0, 3.0),
            min_width=14.0,
            min_height=14.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#FFD700",
            width=0.8,
            style="solid",
            arrow="none",
            routing="straight",
            curvature=0.0,
            opacity=0.6,
        ),
        "back": EdgeStyle(
            color="#FF6347",
            width=0.5,
            style="solid",
            arrow="none",
            routing="straight",
            curvature=0.0,
            opacity=0.4,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#0A0A1A",
        stroke="#1A1A3A",
        stroke_width=0.5,
        stroke_dash="solid",
        corner_radius=10.0,
        font_size=7.5,
        font_color="#808090",
        font_weight="normal",
        padding=8.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#0A0A1A"),
)
TATTOO_THEME = Theme(
    name="tattoo",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#F0D0A0",
            stroke="#1A1A1A",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#1A1A1A",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#CC0000",
            stroke="#1A1A1A",
            stroke_width=3.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#003366",
            stroke="#1A1A1A",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#C0D8E8",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#1A1A1A",
            width=2.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#1A1A1A",
            arrow_length=6.0,
            arrow_width=4.5,
            routing="bezier",
            curvature=0.2,
        ),
        "back": EdgeStyle(
            color="#006600",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#006600",
            arrow_length=5.0,
            arrow_width=4.0,
            routing="bezier",
            curvature=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F0D0A0",
        stroke="#1A1A1A",
        stroke_width=2.0,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=9.0,
        font_color="#1A1A1A",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#F0D0A0"),
)
KNITTING_THEME = Theme(
    name="knitting",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#D06850",
            stroke="#B04838",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=7.5,
            font_color="#FFFFFF",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#4878A0",
            stroke="#306088",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#E8C860",
            stroke="#C8A840",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=7.5,
            font_color="#3A2A08",
            padding=(4.0, 4.0),
            min_width=20.0,
            min_height=20.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#D06850",
            width=2.0,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.35,
        ),
        "back": EdgeStyle(
            color="#4878A0", width=1.5, style="solid", arrow="none", routing="bezier", curvature=0.4
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F8F0E8",
        stroke="#D0C0A8",
        stroke_width=0.8,
        stroke_dash="solid",
        corner_radius=8.0,
        font_size=8.0,
        font_color="#6A5040",
        font_weight="normal",
        padding=10.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#F8F0E8"),
)
BUBBLE_BATH_THEME = Theme(
    name="bubble_bath",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#E8F0F8",
            stroke="#C0D8E8",
            stroke_width=1.0,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#4A6878",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
            opacity=0.7,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#D8E8F8",
            stroke="#B0C8E0",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#2A4858",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=30.0,
            min_height=30.0,
            opacity=0.75,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#F0F4F8",
            stroke="#D8E0E8",
            stroke_width=0.8,
            font_family="Helvetica",
            font_size=7.0,
            font_color="#6A8898",
            padding=(3.0, 3.0),
            min_width=16.0,
            min_height=16.0,
            opacity=0.6,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#B0C8E0",
            width=0.5,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.3,
            opacity=0.3,
        ),
        "back": EdgeStyle(
            color="#C8D8E8",
            width=0.3,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.4,
            opacity=0.2,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#E0E8F0",
        stroke="#C0D0E0",
        stroke_width=0.3,
        stroke_dash="solid",
        corner_radius=20.0,
        font_size=7.5,
        font_color="#6A8898",
        font_weight="normal",
        padding=10.0,
        opacity=0.2,
    ),
    graph_style=GraphStyle(background_color="#E8F0F8"),
)
SAND_CASTLE_THEME = Theme(
    name="sand_castle",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#E8D8B0",
            stroke="#C8B888",
            stroke_width=2.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#5A4A28",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=28.0,
            min_height=24.0,
            corner_radius=0.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#D0C090",
            stroke="#B0A068",
            stroke_width=2.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#3A2A10",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=28.0,
            corner_radius=0.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#E8D8B0",
            stroke="#C8B888",
            stroke_width=1.5,
            font_family="Arial",
            font_size=7.5,
            font_color="#6A5A38",
            padding=(4.0, 4.0),
            min_width=18.0,
            min_height=18.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#C8B888",
            width=2.5,
            style="solid",
            arrow="none",
            routing="straight",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#D8C898",
            width=1.2,
            style="dashed",
            arrow="none",
            routing="straight",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#D8C898",
        stroke="#B8A878",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=8.0,
        font_color="#5A4A28",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#4A90C0"),
)
BLACKLIGHT_THEME = Theme(
    name="blacklight",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#1A0030",
            stroke="#FF00FF",
            stroke_width=2.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#FF00FF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#1A0030",
            stroke="#00FF00",
            stroke_width=3.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#00FF00",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#1A0030",
            stroke="#FF6600",
            stroke_width=2.5,
            font_family="Arial",
            font_size=8.0,
            font_color="#FF6600",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#FFFFFF",
            width=1.5,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.2,
            opacity=0.8,
        ),
        "back": EdgeStyle(
            color="#00FFFF",
            width=0.8,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.3,
            opacity=0.5,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#0A0018",
        stroke="#3A0060",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=6.0,
        font_size=8.0,
        font_color="#FF00FF",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#0A0018"),
)
SPAGHETTI_THEME = Theme(
    name="spaghetti",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#8B4513",
            stroke="#6B3010",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=7.5,
            font_color="#FFE8D0",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#CC2020",
            stroke="#A01010",
            stroke_width=3.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#6B8E23",
            stroke="#4A6E13",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=7.5,
            font_color="#E8F0D0",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#F0D888", width=2.0, style="solid", arrow="none", routing="bezier", curvature=0.5
        ),
        "back": EdgeStyle(
            color="#E8C868", width=1.5, style="solid", arrow="none", routing="bezier", curvature=0.6
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F8F0E0",
        stroke="#E0D0B0",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=10.0,
        font_size=8.0,
        font_color="#8B4513",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#F8F0E0"),
)

FIVETHIRTYEIGHT_THEME = Theme(
    name="fivethirtyeight",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#008FD5",
            stroke="#0070A8",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#FC4F30",
            stroke="#D03820",
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#E5AE38",
            stroke="#C89828",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#008FD5",
            width=1.2,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#008FD5",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.15,
        ),
        "back": EdgeStyle(
            color="#CCCCCC",
            width=0.6,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#CCCCCC",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.2,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F0F0F0",
        stroke="#CCCCCC",
        stroke_width=0.5,
        stroke_dash="solid",
        corner_radius=4.0,
        font_size=8.5,
        font_color="#333333",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#F0F0F0"),
)

# Food web -- trophic levels: green producers at bottom, herbivore
# nodes, predator red at top, energy flow arrows
FOOD_WEB_THEME = Theme(
    name="food_web",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#8BC34A",
            stroke="#689F38",
            stroke_width=1.5,
            font_family="Helvetica",
            font_size=7.5,
            font_color="#1A3A08",
            padding=(4.0, 4.0),
            min_width=24.0,
            min_height=24.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#4CAF50",
            stroke="#388E3C",
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#D32F2F",
            stroke="#B71C1C",
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=26.0,
            min_height=26.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#5D4037",
            width=1.2,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#5D4037",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="bezier",
            curvature=0.2,
        ),
        "back": EdgeStyle(
            color="#A1887F",
            width=0.6,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#A1887F",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#E8F5E9",
        stroke="#A5D6A7",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=6.0,
        font_size=8.5,
        font_color="#2E7D32",
        font_weight="bold",
        padding=10.0,
        opacity=0.35,
    ),
    graph_style=GraphStyle(background_color="#FFFFF8"),
)

# Process -- industrial/manufacturing process flow: clean ISO-style,
# operation boxes, inspection diamonds, precise technical
PROCESS_THEME = Theme(
    name="process",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#E3F2FD",
            stroke="#1565C0",
            stroke_width=1.5,
            font_family="Arial",
            font_size=8.5,
            font_color="#0D47A1",
            padding=(6.0, 4.0),
            min_width=42.0,
            min_height=22.0,
            corner_radius=2.0,
        ),
        "input": NodeStyle(
            shape="diamond",
            fill="#FFF3E0",
            stroke="#E65100",
            stroke_width=2.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#BF360C",
            font_weight="bold",
            padding=(8.0, 6.0),
            min_width=34.0,
            min_height=34.0,
        ),
        "output": NodeStyle(
            shape="ellipse",
            fill="#E8F5E9",
            stroke="#2E7D32",
            stroke_width=1.5,
            font_family="Arial",
            font_size=8.5,
            font_color="#1B5E20",
            font_weight="bold",
            padding=(6.0, 4.0),
            min_width=36.0,
            min_height=22.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#37474F",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#37474F",
            arrow_length=6.0,
            arrow_width=4.0,
            routing="ortho",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#90A4AE",
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#90A4AE",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="ortho",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F5F5F5",
        stroke="#BDBDBD",
        stroke_width=1.5,
        stroke_dash="dashed",
        corner_radius=4.0,
        font_size=9.0,
        font_color="#37474F",
        font_weight="bold",
        padding=12.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

# Instruction manual -- IKEA-style assembly: simple line art, numbered
# steps, minimal color, functional clarity
INSTRUCTION_THEME = Theme(
    name="instruction",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#FFFFFF",
            stroke="#1A1A1A",
            stroke_width=1.0,
            font_family="Helvetica",
            font_size=8.5,
            font_color="#1A1A1A",
            padding=(6.0, 4.0),
            min_width=38.0,
            min_height=22.0,
            corner_radius=3.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#0051BA",
            stroke="#003A8A",
            stroke_width=2.0,
            font_family="Helvetica",
            font_size=9.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#F8F8F8",
            stroke="#CCCCCC",
            stroke_width=0.8,
            font_family="Helvetica",
            font_size=8.0,
            font_color="#666666",
            padding=(6.0, 4.0),
            min_width=36.0,
            min_height=20.0,
            corner_radius=3.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#1A1A1A",
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#1A1A1A",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="ortho",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#AAAAAA",
            width=0.6,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#AAAAAA",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="ortho",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#FFFFFF",
        stroke="#DDDDDD",
        stroke_width=1.0,
        stroke_dash="dashed",
        corner_radius=4.0,
        font_size=9.0,
        font_color="#1A1A1A",
        font_weight="bold",
        padding=10.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

# Ecology -- field biology: earthy greens, habitat zones, species
# interaction arrows, field notebook feel
ECOLOGY_THEME = Theme(
    name="ecology",
    node_styles={
        "default": NodeStyle(
            shape="ellipse",
            fill="#A5D6A7",
            stroke="#66BB6A",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#1B5E20",
            font_style="italic",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=20.0,
        ),
        "input": NodeStyle(
            shape="ellipse",
            fill="#FFD54F",
            stroke="#FFB300",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#5D4037",
            font_weight="bold",
            padding=(5.0, 4.0),
            min_width=34.0,
            min_height=24.0,
        ),
        "output": NodeStyle(
            shape="ellipse",
            fill="#81D4FA",
            stroke="#4FC3F7",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#01579B",
            font_style="italic",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=20.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#5D4037",
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#5D4037",
            arrow_length=5.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.25,
        ),
        "back": EdgeStyle(
            color="#8D6E63",
            width=0.6,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#8D6E63",
            arrow_length=4.0,
            arrow_width=2.5,
            routing="bezier",
            curvature=0.3,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#E8F5E9",
        stroke="#81C784",
        stroke_width=1.0,
        stroke_dash="dashed",
        corner_radius=8.0,
        font_size=8.5,
        font_color="#2E7D32",
        font_weight="bold",
        padding=10.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#F1F8E9"),
)

PSEUDOCODE_THEME = Theme(
    name="pseudocode",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#FFFFFF",
            stroke="#CCCCCC",
            stroke_width=0.5,
            font_family="Courier New",
            font_size=8.5,
            font_color="#000080",
            padding=(6.0, 3.0),
            min_width=38.0,
            min_height=18.0,
            corner_radius=0.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#FFFFFF",
            stroke="#CCCCCC",
            stroke_width=1.0,
            font_family="Courier New",
            font_size=8.5,
            font_color="#800000",
            font_weight="bold",
            padding=(6.0, 3.0),
            min_width=40.0,
            min_height=20.0,
            corner_radius=0.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#FFFFFF",
            stroke="#CCCCCC",
            stroke_width=0.5,
            font_family="Courier New",
            font_size=8.5,
            font_color="#006400",
            padding=(6.0, 3.0),
            min_width=38.0,
            min_height=18.0,
            corner_radius=0.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#333333",
            width=0.8,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#333333",
            arrow_length=5.0,
            arrow_width=3.0,
            routing="ortho",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#999999",
            width=0.5,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#999999",
            arrow_length=4.0,
            arrow_width=2.5,
            routing="ortho",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F8F8F8",
        stroke="#CCCCCC",
        stroke_width=0.5,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=9.0,
        font_color="#333333",
        font_weight="bold",
        padding=10.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

THEME_REGISTRY["pseudocode"] = PSEUDOCODE_THEME
THEME_REGISTRY["food_web"] = FOOD_WEB_THEME
THEME_REGISTRY["process"] = PROCESS_THEME
THEME_REGISTRY["instruction"] = INSTRUCTION_THEME
THEME_REGISTRY["ecology"] = ECOLOGY_THEME

# SCM -- Judea Pearl structural causal model: clean academic DAG,
# variable names in italic, unidirectional causal arrows, white
CAUSAL_THEME = Theme(
    name="causal",
    node_styles={
        "default": NodeStyle(
            shape="ellipse",
            fill="#FFFFFF",
            stroke="#2A66A6",
            stroke_width=1.5,
            font_family="Times New Roman",
            font_size=9.0,
            font_color="#1A1A1A",
            font_style="italic",
            padding=(6.0, 3.0),
            min_width=32.0,
            min_height=20.0,
        ),
        "input": NodeStyle(
            shape="ellipse",
            fill="#FFFFFF",
            stroke="#2A66A6",
            stroke_width=2.5,
            font_family="Times New Roman",
            font_size=9.0,
            font_color="#1A1A1A",
            font_style="italic",
            font_weight="bold",
            padding=(6.0, 3.0),
            min_width=34.0,
            min_height=22.0,
        ),
        "output": NodeStyle(
            shape="ellipse",
            fill="#E8F0F8",
            stroke="#2A66A6",
            stroke_width=1.5,
            font_family="Times New Roman",
            font_size=9.0,
            font_color="#1A1A1A",
            font_style="italic",
            padding=(6.0, 3.0),
            min_width=32.0,
            min_height=20.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#1A1A1A",
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#1A1A1A",
            arrow_length=6.0,
            arrow_width=3.5,
            routing="straight",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#CC3333",
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#CC3333",
            arrow_length=5.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.3,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F8F8FF",
        stroke="#BBBBBB",
        stroke_width=0.5,
        stroke_dash="dashed",
        corner_radius=4.0,
        font_size=9.0,
        font_color="#333333",
        font_weight="bold",
        padding=10.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

# Concept map -- cognitive science: rounded bubbles, labeled cross-links,
# warm educational palette
CONCEPT_MAP_THEME = Theme(
    name="concept_map",
    node_styles={
        "default": NodeStyle(
            shape="ellipse",
            fill="#DCEEFB",
            stroke="#4A90C8",
            stroke_width=1.5,
            font_family="Arial",
            font_size=8.5,
            font_color="#1A3A5A",
            padding=(6.0, 4.0),
            min_width=36.0,
            min_height=22.0,
        ),
        "input": NodeStyle(
            shape="ellipse",
            fill="#FFE4B5",
            stroke="#E0A040",
            stroke_width=2.0,
            font_family="Arial",
            font_size=9.0,
            font_color="#5A3A10",
            font_weight="bold",
            padding=(6.0, 4.0),
            min_width=40.0,
            min_height=26.0,
        ),
        "output": NodeStyle(
            shape="ellipse",
            fill="#D5F5D5",
            stroke="#60A860",
            stroke_width=1.5,
            font_family="Arial",
            font_size=8.5,
            font_color="#1A4A1A",
            padding=(6.0, 4.0),
            min_width=36.0,
            min_height=22.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#666666",
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#666666",
            arrow_length=5.0,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.25,
        ),
        "back": EdgeStyle(
            color="#AAAAAA",
            width=0.6,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#AAAAAA",
            arrow_length=4.0,
            arrow_width=2.5,
            routing="bezier",
            curvature=0.3,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#FFF8F0",
        stroke="#E0C8A0",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=10.0,
        font_size=9.0,
        font_color="#5A4A28",
        font_weight="bold",
        padding=12.0,
        opacity=0.35,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

# Bayesian network -- probabilistic graphical model: clean plates,
# shaded observed nodes, open latent nodes
BAYESIAN_THEME = Theme(
    name="bayesian",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#FFFFFF",
            stroke="#333333",
            stroke_width=1.5,
            font_family="Times New Roman",
            font_size=9.0,
            font_color="#1A1A1A",
            font_style="italic",
            padding=(5.0, 5.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#D0D0D0",
            stroke="#333333",
            stroke_width=2.0,
            font_family="Times New Roman",
            font_size=9.0,
            font_color="#1A1A1A",
            font_style="italic",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#FFFFFF",
            stroke="#333333",
            stroke_width=1.5,
            font_family="Times New Roman",
            font_size=9.0,
            font_color="#333333",
            font_style="italic",
            padding=(5.0, 5.0),
            min_width=26.0,
            min_height=26.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#333333",
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#333333",
            arrow_length=5.5,
            arrow_width=3.5,
            routing="straight",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#888888",
            width=0.6,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#888888",
            arrow_length=4.5,
            arrow_width=3.0,
            routing="bezier",
            curvature=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#FFFFFF",
        stroke="#333333",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=4.0,
        font_size=9.0,
        font_color="#333333",
        font_weight="bold",
        padding=10.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

# Org chart -- corporate hierarchy: crisp cards, title boxes,
# reporting lines, professional blue-gray
ORG_CHART_THEME = Theme(
    name="org_chart",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#FFFFFF",
            stroke="#34495E",
            stroke_width=1.5,
            font_family="Arial",
            font_size=8.5,
            font_color="#2C3E50",
            padding=(8.0, 4.0),
            min_width=44.0,
            min_height=24.0,
            corner_radius=4.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#2C3E50",
            stroke="#1A252F",
            stroke_width=2.0,
            font_family="Arial",
            font_size=9.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(8.0, 4.0),
            min_width=48.0,
            min_height=28.0,
            corner_radius=4.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#F8F9FA",
            stroke="#BDC3C7",
            stroke_width=1.0,
            font_family="Arial",
            font_size=8.0,
            font_color="#7F8C8D",
            padding=(8.0, 4.0),
            min_width=42.0,
            min_height=22.0,
            corner_radius=4.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#34495E", width=1.2, style="solid", arrow="none", routing="ortho", curvature=0.0
        ),
        "back": EdgeStyle(
            color="#BDC3C7", width=0.8, style="dashed", arrow="none", routing="ortho", curvature=0.0
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F0F3F5",
        stroke="#BDC3C7",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=6.0,
        font_size=9.0,
        font_color="#2C3E50",
        font_weight="bold",
        padding=14.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

# UML -- software engineering class/sequence diagrams: precise boxes,
# typed arrows, stereotype brackets, modeling blue
UML_THEME = Theme(
    name="uml",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#FFFFCC",
            stroke="#000000",
            stroke_width=1.0,
            font_family="Courier New",
            font_size=8.0,
            font_color="#000000",
            padding=(6.0, 3.0),
            min_width=40.0,
            min_height=22.0,
            corner_radius=0.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#CCFFCC",
            stroke="#000000",
            stroke_width=1.5,
            font_family="Courier New",
            font_size=8.0,
            font_color="#000000",
            font_weight="bold",
            padding=(6.0, 3.0),
            min_width=42.0,
            min_height=24.0,
            corner_radius=0.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#CCCCFF",
            stroke="#000000",
            stroke_width=1.0,
            font_family="Courier New",
            font_size=8.0,
            font_color="#000000",
            padding=(6.0, 3.0),
            min_width=40.0,
            min_height=22.0,
            corner_radius=0.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#000000",
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#000000",
            arrow_length=6.0,
            arrow_width=4.0,
            routing="ortho",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#000000",
            width=0.8,
            style="dashed",
            arrow="vee",
            arrow_fill="filled",
            arrow_color="#000000",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="ortho",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F0F0FF",
        stroke="#000000",
        stroke_width=1.0,
        stroke_dash="dashed",
        corner_radius=0.0,
        font_size=8.5,
        font_color="#000000",
        font_weight="bold",
        padding=12.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

THEME_REGISTRY["causal"] = CAUSAL_THEME
THEME_REGISTRY["concept_map"] = CONCEPT_MAP_THEME
THEME_REGISTRY["bayesian"] = BAYESIAN_THEME
THEME_REGISTRY["org_chart"] = ORG_CHART_THEME
THEME_REGISTRY["uml"] = UML_THEME

# Linear algebra -- 3Blue1Brown style: dark background, grid feel,
# bold colored vector arrows, coordinate dot nodes, i-hat/j-hat energy
LINEAR_ALGEBRA_THEME = Theme(
    name="linear_algebra",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#FFFF00",
            stroke="#E0E000",
            stroke_width=1.5,
            font_family="Courier New",
            font_size=7.5,
            font_color="#FFFF00",
            font_weight="bold",
            padding=(3.0, 3.0),
            min_width=14.0,
            min_height=14.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#FF4040",
            stroke="#D02020",
            stroke_width=2.0,
            font_family="Courier New",
            font_size=8.0,
            font_color="#FF8080",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=18.0,
            min_height=18.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#58C4DD",
            stroke="#40A8C0",
            stroke_width=1.5,
            font_family="Courier New",
            font_size=7.5,
            font_color="#A0E0F0",
            font_weight="bold",
            padding=(3.0, 3.0),
            min_width=14.0,
            min_height=14.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#83C167",
            width=2.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#83C167",
            arrow_length=6.0,
            arrow_width=4.0,
            routing="straight",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#FF8080",
            width=1.2,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#FF8080",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="straight",
            curvature=0.0,
            opacity=0.6,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#1A1A2E",
        stroke="#2A2A48",
        stroke_width=0.5,
        stroke_dash="dashed",
        corner_radius=0.0,
        font_size=8.0,
        font_color="#58C4DD",
        font_weight="bold",
        padding=8.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#1A1A2E"),
)

THEME_REGISTRY["linear_algebra"] = LINEAR_ALGEBRA_THEME

# Beacons -- hilltop signal fires connected by beams of light
# across dark landscape, Gondor-calls-for-aid energy
BEACONS_THEME = Theme(
    name="beacons",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#F0A020",
            stroke="#E88010",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=7.5,
            font_color="#FFE8A0",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=20.0,
            min_height=20.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#F8D040",
            stroke="#E0B828",
            stroke_width=3.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#3A2808",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#C07818",
            stroke="#A06010",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=7.5,
            font_color="#F0D898",
            padding=(4.0, 4.0),
            min_width=16.0,
            min_height=16.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#F0C040",
            width=1.0,
            style="solid",
            arrow="none",
            routing="straight",
            curvature=0.0,
            opacity=0.5,
        ),
        "back": EdgeStyle(
            color="#C09828",
            width=0.5,
            style="solid",
            arrow="none",
            routing="straight",
            curvature=0.0,
            opacity=0.25,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#0A1018",
        stroke="#1A2028",
        stroke_width=0.5,
        stroke_dash="solid",
        corner_radius=6.0,
        font_size=7.5,
        font_color="#A08840",
        font_weight="bold",
        padding=10.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#0A1018"),
)

THEME_REGISTRY["beacons"] = BEACONS_THEME

# ── Great thinkers themes ─────────────────────────────────────────────

# Euler / Konigsberg -- where graph theory was born (1736): 18th century
# cartography, river blue, bridge brown, landmass green, parchment map
KONIGSBERG_THEME = Theme(
    name="konigsberg",
    node_styles={
        "default": NodeStyle(
            shape="ellipse",
            fill="#8BAA78",
            stroke="#5A7848",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.5,
            font_color="#1A2A10",
            font_weight="bold",
            padding=(6.0, 4.0),
            min_width=34.0,
            min_height=24.0,
        ),
        "input": NodeStyle(
            shape="ellipse",
            fill="#6A8A58",
            stroke="#4A6838",
            stroke_width=3.0,
            font_family="Georgia",
            font_size=9.0,
            font_color="#E8F0D8",
            font_weight="bold",
            padding=(6.0, 4.0),
            min_width=38.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="ellipse",
            fill="#A0BA88",
            stroke="#7A9A68",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#2A3A18",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=20.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#8B6844",
            width=3.0,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.25,
        ),
        "back": EdgeStyle(
            color="#A08060", width=2.0, style="solid", arrow="none", routing="bezier", curvature=0.3
        ),
    },
    cluster_style=ClusterStyle(
        fill="#E0D8C0",
        stroke="#A09878",
        stroke_width=1.0,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=9.0,
        font_color="#5A4A28",
        font_weight="bold",
        padding=12.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#4878A8"),
)

# Deleuze & Guattari -- the rhizome: anti-hierarchical, any-to-any
# connections, nomadic, deterritorialized. Earthy but restless.
RHIZOME_THEME = Theme(
    name="rhizome",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#5A7040",
            stroke="#3A5028",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=7.5,
            font_color="#D0E0B8",
            font_style="italic",
            padding=(3.0, 3.0),
            min_width=16.0,
            min_height=16.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#7A9058",
            stroke="#5A7040",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#E8F0D8",
            font_style="italic",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=20.0,
            min_height=20.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#4A6030",
            stroke="#2A4018",
            stroke_width=1.0,
            font_family="Georgia",
            font_size=7.0,
            font_color="#B0C898",
            font_style="italic",
            padding=(3.0, 3.0),
            min_width=12.0,
            min_height=12.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#6A8048",
            width=0.7,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.45,
        ),
        "back": EdgeStyle(
            color="#8AA068",
            width=0.5,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.55,
            opacity=0.5,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#1A2810",
        stroke="#2A3818",
        stroke_width=0.5,
        stroke_dash="solid",
        corner_radius=8.0,
        font_size=7.5,
        font_color="#7A9058",
        font_weight="normal",
        padding=8.0,
        opacity=0.25,
    ),
    graph_style=GraphStyle(background_color="#1A2010"),
)

# Kabbalah -- Tree of Life (Sefirot): 10 emanations, mystical gold
# and deep blue on cosmic black, sacred geometry
KABBALAH_THEME = Theme(
    name="kabbalah",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#1A1040",
            stroke="#C8A030",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#C8A030",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#C8A030",
            stroke="#E8C848",
            stroke_width=3.0,
            font_family="Georgia",
            font_size=8.5,
            font_color="#1A1040",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=32.0,
            min_height=32.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#1A1040",
            stroke="#8070A0",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#A090C8",
            padding=(5.0, 5.0),
            min_width=26.0,
            min_height=26.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#C8A030",
            width=1.5,
            style="solid",
            arrow="none",
            routing="straight",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#6858A0",
            width=0.8,
            style="solid",
            arrow="none",
            routing="straight",
            curvature=0.0,
            opacity=0.5,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#0A0820",
        stroke="#2A2048",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=9.0,
        font_color="#C8A030",
        font_weight="bold",
        padding=12.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#0A0820"),
)

# Indra's Net -- Buddhist/Hindu infinite reflective web: each node
# a jewel mirroring all others, luminous on void
INDRA_NET_THEME = Theme(
    name="indra_net",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#D0D8E8",
            stroke="#E8E8FF",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=7.0,
            font_color="#8890B0",
            padding=(3.0, 3.0),
            min_width=16.0,
            min_height=16.0,
            opacity=0.7,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#E8E0F8",
            stroke="#F0E8FF",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=7.5,
            font_color="#6068A0",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=20.0,
            min_height=20.0,
            opacity=0.8,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#B8C0D8",
            stroke="#D0D0E8",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=7.0,
            font_color="#9098B0",
            padding=(3.0, 3.0),
            min_width=14.0,
            min_height=14.0,
            opacity=0.6,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#8088B0",
            width=0.4,
            style="solid",
            arrow="none",
            routing="straight",
            curvature=0.0,
            opacity=0.3,
        ),
        "back": EdgeStyle(
            color="#6068A0",
            width=0.3,
            style="solid",
            arrow="none",
            routing="straight",
            curvature=0.0,
            opacity=0.15,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#080810",
        stroke="#181828",
        stroke_width=0.3,
        stroke_dash="solid",
        corner_radius=16.0,
        font_size=7.0,
        font_color="#5058A0",
        font_weight="normal",
        padding=8.0,
        opacity=0.15,
    ),
    graph_style=GraphStyle(background_color="#080810"),
)

# Darwin -- the "I think" notebook sketch (1837): scribbled ink tree
# on cream notebook paper, raw scientific excitement
DARWIN_THEME = Theme(
    name="darwin",
    node_styles={
        "default": NodeStyle(
            shape="none",
            fill="transparent",
            stroke="transparent",
            stroke_width=0.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#3A2A18",
            font_style="italic",
            padding=(2.0, 1.0),
            min_width=0.0,
            min_height=0.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#3A2A18",
            stroke="#2A1A08",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=8.5,
            font_color="#E8D8C0",
            font_weight="bold",
            padding=(3.0, 3.0),
            min_width=10.0,
            min_height=10.0,
        ),
        "output": NodeStyle(
            shape="none",
            fill="transparent",
            stroke="transparent",
            stroke_width=0.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#5A4A30",
            font_style="italic",
            padding=(2.0, 1.0),
            min_width=0.0,
            min_height=0.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#3A2A18", width=1.0, style="solid", arrow="none", routing="bezier", curvature=0.2
        ),
        "back": EdgeStyle(
            color="#6A5A40", width=0.5, style="solid", arrow="none", routing="bezier", curvature=0.3
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F0E8D0",
        stroke="#C0A878",
        stroke_width=0.3,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=8.0,
        font_color="#6A5A40",
        font_weight="normal",
        padding=6.0,
        opacity=0.2,
    ),
    graph_style=GraphStyle(background_color="#F0E8D0"),
)

# Peirce -- existential graphs: logical notation drawn as nested
# ovals on a "sheet of assertion," precise academic formalism
PEIRCE_THEME = Theme(
    name="peirce",
    node_styles={
        "default": NodeStyle(
            shape="ellipse",
            fill="#FFFFFF",
            stroke="#000000",
            stroke_width=1.5,
            font_family="Times New Roman",
            font_size=9.0,
            font_color="#000000",
            font_style="italic",
            padding=(5.0, 3.0),
            min_width=28.0,
            min_height=18.0,
        ),
        "input": NodeStyle(
            shape="ellipse",
            fill="#FFFFFF",
            stroke="#000000",
            stroke_width=2.5,
            font_family="Times New Roman",
            font_size=9.0,
            font_color="#000000",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=20.0,
        ),
        "output": NodeStyle(
            shape="ellipse",
            fill="#F0F0F0",
            stroke="#000000",
            stroke_width=1.0,
            font_family="Times New Roman",
            font_size=9.0,
            font_color="#333333",
            font_style="italic",
            padding=(5.0, 3.0),
            min_width=26.0,
            min_height=16.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#000000",
            width=1.0,
            style="solid",
            arrow="none",
            routing="straight",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#666666",
            width=0.6,
            style="dashed",
            arrow="none",
            routing="straight",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#FFFFFF",
        stroke="#000000",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=20.0,
        font_size=9.0,
        font_color="#000000",
        font_weight="bold",
        padding=10.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

# Kepler -- Mysterium Cosmographicum: planetary orbits, nested
# Platonic solids, Renaissance astronomical diagrams, copperplate
KEPLER_THEME = Theme(
    name="kepler",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#E8D8B8",
            stroke="#8B7355",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#3A2A18",
            font_style="italic",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#C8A030",
            stroke="#A88020",
            stroke_width=2.5,
            font_family="Georgia",
            font_size=8.5,
            font_color="#2A1A08",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#D0C0A0",
            stroke="#A09878",
            stroke_width=1.0,
            font_family="Georgia",
            font_size=7.5,
            font_color="#5A4A30",
            font_style="italic",
            padding=(4.0, 4.0),
            min_width=18.0,
            min_height=18.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#5A4A30", width=0.6, style="solid", arrow="none", routing="bezier", curvature=0.3
        ),
        "back": EdgeStyle(
            color="#8B7355",
            width=0.4,
            style="dashed",
            arrow="none",
            routing="bezier",
            curvature=0.35,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F0E8D0",
        stroke="#8B7355",
        stroke_width=0.5,
        stroke_dash="dashed",
        corner_radius=16.0,
        font_size=8.0,
        font_color="#5A4A30",
        font_weight="normal",
        padding=10.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#F0E8D0"),
)

# Vesalius -- De Humani Corporis Fabrica (1543): woodcut engravings,
# fine crosshatching, anatomical precision on cream paper
VESALIUS_THEME = Theme(
    name="vesalius",
    node_styles={
        "default": NodeStyle(
            shape="ellipse",
            fill="#E8D8C0",
            stroke="#2A1A0E",
            stroke_width=1.0,
            font_family="Georgia",
            font_size=7.5,
            font_color="#2A1A0E",
            font_style="italic",
            padding=(5.0, 3.0),
            min_width=28.0,
            min_height=18.0,
        ),
        "input": NodeStyle(
            shape="ellipse",
            fill="#D8C8A8",
            stroke="#2A1A0E",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#2A1A0E",
            font_style="italic",
            font_weight="bold",
            padding=(5.0, 4.0),
            min_width=32.0,
            min_height=22.0,
        ),
        "output": NodeStyle(
            shape="ellipse",
            fill="#E8D8C0",
            stroke="#5A4A30",
            stroke_width=0.8,
            font_family="Georgia",
            font_size=7.5,
            font_color="#5A4A30",
            font_style="italic",
            padding=(5.0, 3.0),
            min_width=26.0,
            min_height=16.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#2A1A0E",
            width=0.7,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#2A1A0E",
            arrow_length=4.0,
            arrow_width=2.0,
            routing="bezier",
            curvature=0.25,
        ),
        "back": EdgeStyle(
            color="#5A4A30",
            width=0.4,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#5A4A30",
            arrow_length=3.5,
            arrow_width=1.5,
            routing="bezier",
            curvature=0.3,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#E8D8C0",
        stroke="#8B7355",
        stroke_width=0.3,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=8.0,
        font_color="#5A4A30",
        font_weight="normal",
        padding=8.0,
        opacity=0.2,
    ),
    graph_style=GraphStyle(background_color="#E8D8C0"),
)

THEME_REGISTRY["konigsberg"] = KONIGSBERG_THEME
THEME_REGISTRY["rhizome"] = RHIZOME_THEME
THEME_REGISTRY["kabbalah"] = KABBALAH_THEME
THEME_REGISTRY["indra_net"] = INDRA_NET_THEME
THEME_REGISTRY["darwin"] = DARWIN_THEME
THEME_REGISTRY["peirce"] = PEIRCE_THEME
THEME_REGISTRY["kepler"] = KEPLER_THEME
THEME_REGISTRY["vesalius"] = VESALIUS_THEME

# Runes -- ancient mystical inscriptions: carved angular symbols on
# dark weathered stone, faint eldritch glow
RUNES_THEME = Theme(
    name="runes",
    node_styles={
        "default": NodeStyle(
            shape="diamond",
            fill="#1A1820",
            stroke="#6A8870",
            stroke_width=2.0,
            font_family="Georgia",
            font_size=8.5,
            font_color="#6A8870",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=26.0,
            min_height=26.0,
        ),
        "input": NodeStyle(
            shape="diamond",
            fill="#1A1820",
            stroke="#90C898",
            stroke_width=3.0,
            font_family="Georgia",
            font_size=9.0,
            font_color="#90C898",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=30.0,
            min_height=30.0,
        ),
        "output": NodeStyle(
            shape="diamond",
            fill="#1A1820",
            stroke="#4A6850",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=8.0,
            font_color="#4A6850",
            padding=(5.0, 5.0),
            min_width=24.0,
            min_height=24.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#6A8870",
            width=1.0,
            style="solid",
            arrow="none",
            routing="straight",
            curvature=0.0,
            opacity=0.6,
        ),
        "back": EdgeStyle(
            color="#3A5840",
            width=0.5,
            style="solid",
            arrow="none",
            routing="straight",
            curvature=0.0,
            opacity=0.35,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#10100E",
        stroke="#3A4838",
        stroke_width=1.5,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=8.5,
        font_color="#6A8870",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#10100E"),
)

# Monad -- Leibniz's monads: self-contained windowless substances,
# each reflecting the universe. Isolated circles with no connecting
# edges visible (very faint), deep philosophical darkness
MONAD_THEME = Theme(
    name="monad",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#18141E",
            stroke="#4838A0",
            stroke_width=1.5,
            font_family="Times New Roman",
            font_size=8.0,
            font_color="#6858C8",
            font_style="italic",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#201830",
            stroke="#6050D0",
            stroke_width=2.5,
            font_family="Times New Roman",
            font_size=8.5,
            font_color="#8878E8",
            font_style="italic",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=32.0,
            min_height=32.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#14101A",
            stroke="#383068",
            stroke_width=1.0,
            font_family="Times New Roman",
            font_size=8.0,
            font_color="#504898",
            font_style="italic",
            padding=(5.0, 5.0),
            min_width=26.0,
            min_height=26.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#2A2048",
            width=0.4,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.2,
            opacity=0.2,
        ),
        "back": EdgeStyle(
            color="#1A1830",
            width=0.3,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.3,
            opacity=0.1,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#0A080E",
        stroke="#201830",
        stroke_width=0.5,
        stroke_dash="solid",
        corner_radius=20.0,
        font_size=8.0,
        font_color="#383068",
        font_weight="normal",
        padding=10.0,
        opacity=0.2,
    ),
    graph_style=GraphStyle(background_color="#0A080E"),
)

THEME_REGISTRY["runes"] = RUNES_THEME
THEME_REGISTRY["monad"] = MONAD_THEME

EUCLID_THEME = Theme(
    name="euclid",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#FFFFFF",
            stroke="#1A1A1A",
            stroke_width=1.0,
            font_family="Times New Roman",
            font_size=9.0,
            font_color="#1A1A1A",
            font_style="italic",
            padding=(4.0, 4.0),
            min_width=20.0,
            min_height=20.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#FFFFFF",
            stroke="#1A1A1A",
            stroke_width=1.5,
            font_family="Times New Roman",
            font_size=9.0,
            font_color="#1A1A1A",
            font_style="italic",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=22.0,
            min_height=22.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#FFFFFF",
            stroke="#1A1A1A",
            stroke_width=1.0,
            font_family="Times New Roman",
            font_size=9.0,
            font_color="#1A1A1A",
            font_style="italic",
            padding=(4.0, 4.0),
            min_width=20.0,
            min_height=20.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#1A1A1A",
            width=0.8,
            style="solid",
            arrow="none",
            routing="straight",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#1A1A1A",
            width=0.5,
            style="dashed",
            arrow="none",
            routing="straight",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#FFFFFF",
        stroke="#1A1A1A",
        stroke_width=0.5,
        stroke_dash="dashed",
        corner_radius=0.0,
        font_size=9.0,
        font_color="#1A1A1A",
        font_weight="normal",
        padding=8.0,
        opacity=0.2,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

THEME_REGISTRY["euclid"] = EUCLID_THEME

SIDEWALK_CHALK_THEME = Theme(
    name="sidewalk_chalk",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#FF6B8A",
            stroke="#E05070",
            stroke_width=2.5,
            font_family="Comic Sans MS",
            font_size=9.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=28.0,
            min_height=28.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#4DC9F6",
            stroke="#30A8D8",
            stroke_width=3.0,
            font_family="Comic Sans MS",
            font_size=9.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=30.0,
            min_height=30.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#A0E030",
            stroke="#80C018",
            stroke_width=2.5,
            font_family="Comic Sans MS",
            font_size=9.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(4.0, 4.0),
            min_width=26.0,
            min_height=26.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#F9E94E",
            width=3.0,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.25,
        ),
        "back": EdgeStyle(
            color="#FF9F43", width=2.0, style="solid", arrow="none", routing="bezier", curvature=0.3
        ),
    },
    cluster_style=ClusterStyle(
        fill="#808080",
        stroke="#909090",
        stroke_width=2.0,
        stroke_dash="solid",
        corner_radius=8.0,
        font_size=9.0,
        font_color="#FFFFFF",
        font_weight="bold",
        padding=10.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#808080"),
)
SAND_TRACE_THEME = Theme(
    name="sand_trace",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#D8C498",
            stroke="#C8B488",
            stroke_width=0.5,
            font_family="Georgia",
            font_size=7.5,
            font_color="#8A7A58",
            padding=(4.0, 4.0),
            min_width=18.0,
            min_height=18.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#C8B488",
            stroke="#B8A478",
            stroke_width=1.0,
            font_family="Georgia",
            font_size=8.0,
            font_color="#6A5A38",
            font_weight="bold",
            padding=(5.0, 5.0),
            min_width=22.0,
            min_height=22.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#E0D0A8",
            stroke="#D0C098",
            stroke_width=0.3,
            font_family="Georgia",
            font_size=7.0,
            font_color="#9A8A68",
            padding=(3.0, 3.0),
            min_width=14.0,
            min_height=14.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#B8A878",
            width=1.2,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.3,
            opacity=0.6,
        ),
        "back": EdgeStyle(
            color="#C8B888",
            width=0.6,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.4,
            opacity=0.35,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#D8C898",
        stroke="#C8B888",
        stroke_width=0.3,
        stroke_dash="solid",
        corner_radius=10.0,
        font_size=7.5,
        font_color="#8A7A58",
        font_weight="normal",
        padding=8.0,
        opacity=0.2,
    ),
    graph_style=GraphStyle(background_color="#D8C898"),
)

THEME_REGISTRY["sidewalk_chalk"] = SIDEWALK_CHALK_THEME
THEME_REGISTRY["sand_trace"] = SAND_TRACE_THEME

STENCIL_THEME = Theme(
    name="stencil",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#2A2A2A",
            stroke="#E8E8E8",
            stroke_width=0.5,
            font_family="Arial",
            font_size=9.0,
            font_color="#E8E8E8",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=0.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#2A2A2A",
            stroke="#FFFFFF",
            stroke_width=1.5,
            font_family="Arial",
            font_size=10.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=22.0,
            corner_radius=0.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#2A2A2A",
            stroke="#C0C0C0",
            stroke_width=0.5,
            font_family="Arial",
            font_size=9.0,
            font_color="#C0C0C0",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=32.0,
            min_height=20.0,
            corner_radius=0.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#E8E8E8",
            width=1.5,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#E8E8E8",
            arrow_length=5.0,
            arrow_width=3.5,
            routing="straight",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#808080",
            width=0.8,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#808080",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="straight",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#1A1A1A",
        stroke="#4A4A4A",
        stroke_width=1.0,
        stroke_dash="dashed",
        corner_radius=0.0,
        font_size=9.0,
        font_color="#E8E8E8",
        font_weight="bold",
        padding=10.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#2A2A2A"),
)

THEME_REGISTRY["stencil"] = STENCIL_THEME
THEME_REGISTRY["fivethirtyeight"] = FIVETHIRTYEIGHT_THEME
THEME_REGISTRY["frosting"] = FROSTING_THEME
THEME_REGISTRY["lego"] = LEGO_THEME
THEME_REGISTRY["crayon"] = CRAYON_THEME
THEME_REGISTRY["quilt"] = QUILT_THEME
THEME_REGISTRY["fireworks"] = FIREWORKS_THEME
THEME_REGISTRY["tattoo"] = TATTOO_THEME
THEME_REGISTRY["knitting"] = KNITTING_THEME
THEME_REGISTRY["bubble_bath"] = BUBBLE_BATH_THEME
THEME_REGISTRY["sand_castle"] = SAND_CASTLE_THEME
THEME_REGISTRY["blacklight"] = BLACKLIGHT_THEME
THEME_REGISTRY["spaghetti"] = SPAGHETTI_THEME
THEME_REGISTRY["frosted"] = FROSTED_THEME
THEME_REGISTRY["hollywood"] = HOLLYWOOD_THEME
THEME_REGISTRY["brutalist"] = BRUTALIST_THEME
THEME_REGISTRY["lovecraft"] = LOVECRAFT_THEME
THEME_REGISTRY["satellite"] = SATELLITE_THEME
THEME_REGISTRY["nautical"] = NAUTICAL_THEME
THEME_REGISTRY["wine_label"] = WINE_LABEL_THEME
THEME_REGISTRY["marble"] = MARBLE_THEME
THEME_REGISTRY["valentine"] = VALENTINE_THEME
THEME_REGISTRY["halloween"] = HALLOWEEN_THEME
THEME_REGISTRY["letterpress"] = LETTERPRESS_THEME
THEME_REGISTRY["risograph"] = RISOGRAPH_THEME
THEME_REGISTRY["jazz"] = JAZZ_THEME
THEME_REGISTRY["wikipedia"] = WIKIPEDIA_THEME
THEME_REGISTRY["nature_journal"] = NATURE_JOURNAL_THEME
THEME_REGISTRY["google_maps"] = GOOGLE_MAPS_THEME
THEME_REGISTRY["conspiracy"] = CONSPIRACY_THEME
THEME_REGISTRY["newspaper"] = NEWSPAPER_THEME
THEME_REGISTRY["go"] = GO_BOARD_THEME
THEME_REGISTRY["chess"] = CHESS_THEME
THEME_REGISTRY["checkers"] = CHECKERS_THEME
THEME_REGISTRY["bark_carving"] = BARK_CARVING_THEME
THEME_REGISTRY["chisel"] = CHISEL_THEME
THEME_REGISTRY["chutes_ladders"] = CHUTES_LADDERS_THEME
THEME_REGISTRY["carved"] = CARVED_THEME
THEME_REGISTRY["olympic"] = OLYMPIC_THEME
THEME_REGISTRY["syntax_tree"] = SYNTAX_TREE_THEME
THEME_REGISTRY["data_structure"] = DATA_STRUCTURE_THEME
THEME_REGISTRY["usa"] = USA_THEME
THEME_REGISTRY["brazil"] = BRAZIL_THEME
THEME_REGISTRY["japan"] = JAPAN_THEME
THEME_REGISTRY["india"] = INDIA_THEME
THEME_REGISTRY["bracket"] = BRACKET_THEME
THEME_REGISTRY["fantasy_football"] = FANTASY_FOOTBALL_THEME
THEME_REGISTRY["seahawks"] = SEAHAWKS_THEME
THEME_REGISTRY["category_theory"] = CATEGORY_THEORY_THEME

# ASCII -- monospace text art, classic green-on-black terminal
ASCII_GREEN_THEME = Theme(
    name="ascii_green",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#000000",
            stroke="#33FF33",
            stroke_width=1.0,
            font_family="Courier New",
            font_size=9.0,
            font_color="#33FF33",
            padding=(4.0, 2.0),
            min_width=28.0,
            min_height=16.0,
            corner_radius=0.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#003300",
            stroke="#33FF33",
            stroke_width=2.0,
            font_family="Courier New",
            font_size=9.0,
            font_color="#33FF33",
            font_weight="bold",
            padding=(4.0, 2.0),
            min_width=30.0,
            min_height=18.0,
            corner_radius=0.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#000000",
            stroke="#22AA22",
            stroke_width=1.0,
            font_family="Courier New",
            font_size=9.0,
            font_color="#22AA22",
            padding=(4.0, 2.0),
            min_width=28.0,
            min_height=16.0,
            corner_radius=0.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#33FF33",
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#33FF33",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="ortho",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#116611",
            width=0.6,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#116611",
            arrow_length=3.5,
            arrow_width=2.5,
            routing="ortho",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#000000",
        stroke="#22AA22",
        stroke_width=1.0,
        stroke_dash="dashed",
        corner_radius=0.0,
        font_size=9.0,
        font_color="#33FF33",
        font_weight="bold",
        padding=8.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#000000"),
)

# ASCII amber -- warm phosphor CRT
ASCII_AMBER_THEME = Theme(
    name="ascii_amber",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#000000",
            stroke="#FFB000",
            stroke_width=1.0,
            font_family="Courier New",
            font_size=9.0,
            font_color="#FFB000",
            padding=(4.0, 2.0),
            min_width=28.0,
            min_height=16.0,
            corner_radius=0.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#1A1000",
            stroke="#FFB000",
            stroke_width=2.0,
            font_family="Courier New",
            font_size=9.0,
            font_color="#FFB000",
            font_weight="bold",
            padding=(4.0, 2.0),
            min_width=30.0,
            min_height=18.0,
            corner_radius=0.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#000000",
            stroke="#CC8800",
            stroke_width=1.0,
            font_family="Courier New",
            font_size=9.0,
            font_color="#CC8800",
            padding=(4.0, 2.0),
            min_width=28.0,
            min_height=16.0,
            corner_radius=0.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#FFB000",
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#FFB000",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="ortho",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#886600",
            width=0.6,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#886600",
            arrow_length=3.5,
            arrow_width=2.5,
            routing="ortho",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#000000",
        stroke="#CC8800",
        stroke_width=1.0,
        stroke_dash="dashed",
        corner_radius=0.0,
        font_size=9.0,
        font_color="#FFB000",
        font_weight="bold",
        padding=8.0,
        opacity=0.4,
    ),
    graph_style=GraphStyle(background_color="#000000"),
)

# ASCII white -- clean monochrome paper printout
ASCII_WHITE_THEME = Theme(
    name="ascii_white",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#FFFFFF",
            stroke="#000000",
            stroke_width=1.0,
            font_family="Courier New",
            font_size=9.0,
            font_color="#000000",
            padding=(4.0, 2.0),
            min_width=28.0,
            min_height=16.0,
            corner_radius=0.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#FFFFFF",
            stroke="#000000",
            stroke_width=2.0,
            font_family="Courier New",
            font_size=9.0,
            font_color="#000000",
            font_weight="bold",
            padding=(4.0, 2.0),
            min_width=30.0,
            min_height=18.0,
            corner_radius=0.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#EEEEEE",
            stroke="#000000",
            stroke_width=1.0,
            font_family="Courier New",
            font_size=9.0,
            font_color="#333333",
            padding=(4.0, 2.0),
            min_width=28.0,
            min_height=16.0,
            corner_radius=0.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#000000",
            width=1.0,
            style="solid",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#000000",
            arrow_length=4.0,
            arrow_width=3.0,
            routing="ortho",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#888888",
            width=0.6,
            style="dashed",
            arrow="normal",
            arrow_fill="filled",
            arrow_color="#888888",
            arrow_length=3.5,
            arrow_width=2.5,
            routing="ortho",
            curvature=0.0,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#FFFFFF",
        stroke="#000000",
        stroke_width=1.0,
        stroke_dash="dashed",
        corner_radius=0.0,
        font_size=9.0,
        font_color="#000000",
        font_weight="bold",
        padding=8.0,
        opacity=0.3,
    ),
    graph_style=GraphStyle(background_color="#FFFFFF"),
)

THEME_REGISTRY["ascii_green"] = ASCII_GREEN_THEME
THEME_REGISTRY["ascii_amber"] = ASCII_AMBER_THEME
THEME_REGISTRY["ascii_white"] = ASCII_WHITE_THEME

CATS_CRADLE_THEME = Theme(
    name="cats_cradle",
    node_styles={
        "default": NodeStyle(
            shape="circle",
            fill="#F0E0D0",
            stroke="#C8A888",
            stroke_width=1.0,
            font_family="Georgia",
            font_size=7.0,
            font_color="#6A5040",
            padding=(3.0, 3.0),
            min_width=12.0,
            min_height=12.0,
        ),
        "input": NodeStyle(
            shape="circle",
            fill="#E8D0B8",
            stroke="#B89878",
            stroke_width=1.5,
            font_family="Georgia",
            font_size=7.5,
            font_color="#5A4030",
            font_weight="bold",
            padding=(3.0, 3.0),
            min_width=14.0,
            min_height=14.0,
        ),
        "output": NodeStyle(
            shape="circle",
            fill="#F0E0D0",
            stroke="#C8A888",
            stroke_width=1.0,
            font_family="Georgia",
            font_size=7.0,
            font_color="#6A5040",
            padding=(3.0, 3.0),
            min_width=12.0,
            min_height=12.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#E04040",
            width=1.2,
            style="solid",
            arrow="none",
            routing="straight",
            curvature=0.0,
        ),
        "back": EdgeStyle(
            color="#4080E0",
            width=1.0,
            style="solid",
            arrow="none",
            routing="straight",
            curvature=0.0,
            opacity=0.7,
        ),
    },
    cluster_style=ClusterStyle(
        fill="#F8F0E8",
        stroke="#D8C8B0",
        stroke_width=0.3,
        stroke_dash="solid",
        corner_radius=0.0,
        font_size=7.0,
        font_color="#8A7A68",
        font_weight="normal",
        padding=6.0,
        opacity=0.2,
    ),
    graph_style=GraphStyle(background_color="#F8F0E8"),
)

DEFUSE_THEME = Theme(
    name="defuse",
    node_styles={
        "default": NodeStyle(
            shape="rectangle",
            fill="#2A2A2A",
            stroke="#555555",
            stroke_width=2.0,
            font_family="Courier New",
            font_size=8.0,
            font_color="#E0E0E0",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=20.0,
            corner_radius=2.0,
        ),
        "input": NodeStyle(
            shape="rectangle",
            fill="#CC0000",
            stroke="#FF0000",
            stroke_width=3.0,
            font_family="Courier New",
            font_size=9.0,
            font_color="#FFFFFF",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=34.0,
            min_height=24.0,
            corner_radius=2.0,
        ),
        "output": NodeStyle(
            shape="rectangle",
            fill="#1A1A1A",
            stroke="#00CC00",
            stroke_width=2.0,
            font_family="Courier New",
            font_size=8.0,
            font_color="#00FF00",
            font_weight="bold",
            padding=(5.0, 3.0),
            min_width=30.0,
            min_height=20.0,
            corner_radius=2.0,
        ),
    },
    edge_styles={
        "default": EdgeStyle(
            color="#FF0000",
            width=2.5,
            style="solid",
            arrow="none",
            routing="bezier",
            curvature=0.15,
        ),
        "back": EdgeStyle(
            color="#0088FF", width=2.5, style="solid", arrow="none", routing="bezier", curvature=0.2
        ),
    },
    cluster_style=ClusterStyle(
        fill="#1A1A1A",
        stroke="#444444",
        stroke_width=2.0,
        stroke_dash="solid",
        corner_radius=3.0,
        font_size=8.0,
        font_color="#FF0000",
        font_weight="bold",
        padding=10.0,
        opacity=0.5,
    ),
    graph_style=GraphStyle(background_color="#0A0A0A"),
)

THEME_REGISTRY["cats_cradle"] = CATS_CRADLE_THEME
THEME_REGISTRY["defuse"] = DEFUSE_THEME
THEME_REGISTRY["simcity"] = SIMCITY_THEME
THEME_REGISTRY["zen"] = ZEN_THEME
THEME_REGISTRY["honeycomb"] = HONEYCOMB_THEME
THEME_REGISTRY["campfire"] = CAMPFIRE_THEME
THEME_REGISTRY["rust"] = RUST_THEME
THEME_REGISTRY["synthwave"] = SYNTHWAVE_THEME
THEME_REGISTRY["pride"] = PRIDE_THEME
THEME_REGISTRY["retro_diner"] = RETRO_DINER_THEME
THEME_REGISTRY["tarot"] = TAROT_THEME
THEME_REGISTRY["bob_ross"] = BOB_ROSS_THEME
THEME_REGISTRY["minecraft"] = MINECRAFT_THEME
THEME_REGISTRY["tetris"] = TETRIS_THEME
THEME_REGISTRY["bubblegum"] = BUBBLEGUM_THEME
THEME_REGISTRY["polaroid"] = POLAROID_THEME
THEME_REGISTRY["art_nouveau"] = ART_NOUVEAU_THEME
THEME_REGISTRY["tim_burton"] = TIM_BURTON_THEME
THEME_REGISTRY["disney"] = DISNEY_THEME
THEME_REGISTRY["ghibli"] = GHIBLI_THEME
THEME_REGISTRY["picasso"] = PICASSO_THEME
THEME_REGISTRY["pollock"] = POLLOCK_THEME
THEME_REGISTRY["riley"] = RILEY_THEME
THEME_REGISTRY["renoir"] = RENOIR_THEME
THEME_REGISTRY["da_vinci"] = DA_VINCI_THEME
THEME_REGISTRY["mondrian"] = MONDRIAN_THEME
THEME_REGISTRY["van_gogh"] = VAN_GOGH_THEME
THEME_REGISTRY["klimt"] = KLIMT_THEME
THEME_REGISTRY["warhol"] = WARHOL_THEME
THEME_REGISTRY["crystal"] = CRYSTAL_THEME
THEME_REGISTRY["neon_sign"] = NEON_SIGN_THEME
THEME_REGISTRY["fantasy_map"] = FANTASY_MAP_THEME
THEME_REGISTRY["tech_tree"] = TECH_TREE_THEME
THEME_REGISTRY["cola"] = COLA_THEME
THEME_REGISTRY["apple"] = APPLE_THEME
THEME_REGISTRY["material"] = MATERIAL_THEME
THEME_REGISTRY["spotify"] = SPOTIFY_THEME
THEME_REGISTRY["slack"] = SLACK_THEME
THEME_REGISTRY["seaborn"] = SEABORN_THEME
THEME_REGISTRY["matplotlib"] = MATPLOTLIB_THEME
THEME_REGISTRY["ggplot"] = GGPLOT_THEME
THEME_REGISTRY["fracture"] = FRACTURE_THEME
THEME_REGISTRY["yale"] = YALE_THEME
THEME_REGISTRY["harvard"] = HARVARD_THEME
THEME_REGISTRY["princeton"] = PRINCETON_THEME
THEME_REGISTRY["lakers"] = LAKERS_THEME
THEME_REGISTRY["yankees"] = YANKEES_THEME
THEME_REGISTRY["celtics"] = CELTICS_THEME
THEME_REGISTRY["ferrari"] = FERRARI_THEME
THEME_REGISTRY["canyon"] = CANYON_THEME
THEME_REGISTRY["pacman"] = PACMAN_THEME
THEME_REGISTRY["lilypad"] = LILYPAD_THEME
THEME_REGISTRY["garden"] = GARDEN_THEME
THEME_REGISTRY["river"] = RIVER_THEME
THEME_REGISTRY["jetstream"] = JETSTREAM_THEME
THEME_REGISTRY["weather"] = WEATHER_THEME
THEME_REGISTRY["nyt"] = NYT_THEME
THEME_REGISTRY["economist"] = ECONOMIST_THEME
THEME_REGISTRY["ft"] = FT_THEME
THEME_REGISTRY["solarized_light"] = SOLARIZED_LIGHT_THEME
THEME_REGISTRY["solarized_dark"] = SOLARIZED_DARK_THEME
THEME_REGISTRY["monokai"] = MONOKAI_THEME
THEME_REGISTRY["dracula"] = DRACULA_THEME
THEME_REGISTRY["nord"] = NORD_THEME
THEME_REGISTRY["gruvbox"] = GRUVBOX_THEME
THEME_REGISTRY["one_dark"] = ONE_DARK_THEME
THEME_REGISTRY["catppuccin"] = CATPPUCCIN_THEME
THEME_REGISTRY["tokyo_night"] = TOKYO_NIGHT_THEME
THEME_REGISTRY["fortress"] = FORTRESS_THEME
THEME_REGISTRY["catacombs"] = CATACOMBS_THEME
THEME_REGISTRY["power_grid"] = POWER_GRID_THEME
THEME_REGISTRY["jungle"] = JUNGLE_THEME
THEME_REGISTRY["railway"] = RAILWAY_THEME
THEME_REGISTRY["flowchart"] = FLOWCHART_THEME
THEME_REGISTRY["adventure"] = ADVENTURE_THEME
THEME_REGISTRY["aqueduct"] = AQUEDUCT_THEME
THEME_REGISTRY["dna"] = DNA_THEME
THEME_REGISTRY["origami"] = ORIGAMI_THEME
THEME_REGISTRY["clockwork"] = CLOCKWORK_THEME
THEME_REGISTRY["tapestry"] = TAPESTRY_THEME
THEME_REGISTRY["plumbing"] = PLUMBING_THEME
THEME_REGISTRY["noir"] = NOIR_THEME
THEME_REGISTRY["cyberpunk"] = CYBERPUNK_THEME
THEME_REGISTRY["vascular"] = VASCULAR_THEME
THEME_REGISTRY["nebula"] = NEBULA_THEME
THEME_REGISTRY["lava"] = LAVA_THEME
THEME_REGISTRY["frost"] = FROST_THEME
THEME_REGISTRY["treasure_map"] = TREASURE_MAP_THEME
THEME_REGISTRY["propaganda"] = PROPAGANDA_THEME
THEME_REGISTRY["gothic"] = GOTHIC_THEME
THEME_REGISTRY["graffiti"] = GRAFFITI_THEME
THEME_REGISTRY["ant_colony"] = ANT_COLONY_THEME
THEME_REGISTRY["telecom"] = TELECOM_THEME
THEME_REGISTRY["social"] = SOCIAL_THEME
THEME_REGISTRY["flight_map"] = FLIGHT_MAP_THEME
THEME_REGISTRY["mario"] = MARIO_THEME
THEME_REGISTRY["mycelium"] = MYCELIUM_THEME
THEME_REGISTRY["xkcd"] = XKCD_THEME
THEME_REGISTRY["slime_mold"] = SLIME_MOLD_THEME
THEME_REGISTRY["cavern"] = CAVERN_THEME
THEME_REGISTRY["coral"] = CORAL_THEME
THEME_REGISTRY["autumn"] = AUTUMN_THEME
THEME_REGISTRY["aurora"] = AURORA_THEME
THEME_REGISTRY["cave"] = CAVE_THEME
THEME_REGISTRY["stained_glass"] = STAINED_GLASS_THEME
THEME_REGISTRY["watercolor"] = WATERCOLOR_THEME
THEME_REGISTRY["ukiyo_e"] = UKIYO_E_THEME
THEME_REGISTRY["illuminated"] = ILLUMINATED_THEME
THEME_REGISTRY["matrix"] = MATRIX_THEME
THEME_REGISTRY["tron"] = TRON_THEME
THEME_REGISTRY["steampunk"] = STEAMPUNK_THEME
THEME_REGISTRY["pixel"] = PIXEL_THEME
THEME_REGISTRY["xray"] = XRAY_THEME
THEME_REGISTRY["thermal"] = THERMAL_THEME
THEME_REGISTRY["microscopy"] = MICROSCOPY_THEME
THEME_REGISTRY["topographic"] = TOPOGRAPHIC_THEME
THEME_REGISTRY["hieroglyph"] = HIEROGLYPH_THEME
THEME_REGISTRY["roman_mosaic"] = ROMAN_MOSAIC_THEME
THEME_REGISTRY["catan"] = CATAN_THEME
THEME_REGISTRY["archipelago"] = ARCHIPELAGO_THEME
THEME_REGISTRY["branches"] = BRANCHES_THEME
THEME_REGISTRY["spiderweb"] = SPIDERWEB_THEME
THEME_REGISTRY["phylogeny"] = PHYLOGENY_THEME
THEME_REGISTRY["roadmap"] = ROADMAP_THEME
THEME_REGISTRY["van_essen"] = VAN_ESSEN_THEME
THEME_REGISTRY["cajal"] = CAJAL_THEME
THEME_REGISTRY["connectome"] = CONNECTOME_THEME
THEME_REGISTRY["pathway"] = PATHWAY_THEME


def get_theme(name: str) -> Theme:
    """Look up a built-in theme by name. Returns a deep copy."""
    if name not in THEME_REGISTRY:
        raise ValueError(f"Unknown theme: {name!r}. Available: {list(THEME_REGISTRY.keys())}")
    return copy.deepcopy(THEME_REGISTRY[name])


# ─── Style Cascade Resolution ─────────────────────────────────────────────


def _is_default_value(style_obj, field_name: str) -> bool:
    """Check if a field on a style dataclass is still its default value."""
    field_val = getattr(style_obj, field_name)
    for f in dataclass_fields(type(style_obj)):
        if f.name == field_name:
            if f.default is not dataclasses.MISSING:
                return field_val == f.default
            if f.default_factory is not dataclasses.MISSING:
                return field_val == f.default_factory()
            return False
    return False


def resolve_node_style(
    per_element: Optional[NodeStyle],
    cluster_member_styles: Optional[List[Optional[NodeStyle]]],
    theme_style: NodeStyle,
    graph_default: Optional[NodeStyle] = None,
    global_default: Optional[NodeStyle] = None,
) -> NodeStyle:
    """Field-level merge: most-specific scope wins.

    Cascade order (highest priority first):
    1. per_element — per-node override
    2. cluster_member_styles — deepest cluster first
    3. theme_style — from Theme.get_node_style()
    4. graph_default — Graph.default_node_style
    5. global_default — dagua.configure() overrides

    For each field, picks the first non-default value walking the cascade.
    """
    sources: List[Optional[NodeStyle]] = [per_element]
    if cluster_member_styles:
        sources.extend(cluster_member_styles)
    sources.append(theme_style)
    sources.append(graph_default)
    sources.append(global_default)

    return _merge_style(NodeStyle, sources)


def resolve_edge_style(
    per_element: Optional[EdgeStyle],
    cluster_member_styles: Optional[List[Optional[EdgeStyle]]],
    theme_style: EdgeStyle,
    graph_default: Optional[EdgeStyle] = None,
    global_default: Optional[EdgeStyle] = None,
) -> EdgeStyle:
    """Field-level merge for edge styles. Same cascade as resolve_node_style."""
    sources: List[Optional[EdgeStyle]] = [per_element]
    if cluster_member_styles:
        sources.extend(cluster_member_styles)
    sources.append(theme_style)
    sources.append(graph_default)
    sources.append(global_default)

    return _merge_style(EdgeStyle, sources)


def resolve_cluster_style(
    per_cluster: Optional[ClusterStyle],
    theme_style: ClusterStyle,
    global_default: Optional[ClusterStyle] = None,
) -> ClusterStyle:
    """Field-level merge for cluster styles."""
    sources: List[Optional[ClusterStyle]] = [per_cluster, theme_style, global_default]
    return _merge_style(ClusterStyle, sources)


def _merge_style(cls, sources: List[Optional[Any]]):
    """Generic field-level merge across a cascade of style sources.

    For each field, picks the first non-default value walking the cascade.
    The per-element source (sources[0]) is special: ALL of its non-None
    fields win unconditionally, even if they match the class default.
    This prevents a theme from overriding an explicit per-element choice
    that happens to match the dataclass default value.

    Lower-priority sources (theme, graph default, global default) only
    contribute fields that differ from the class default.
    """
    import dataclasses as _dc

    defaults_instance = cls()
    defaults_dict = {f.name: getattr(defaults_instance, f.name) for f in _dc.fields(cls)}

    # Track which fields the per-element source explicitly set.
    # A per-element source is sources[0].  If it exists and has a
    # _set_fields attribute, use that.  Otherwise fall back to checking
    # non-default values for all sources uniformly.
    per_element = sources[0] if sources else None
    per_element_fields: set = set()
    if per_element is not None and hasattr(per_element, "_set_fields"):
        per_element_fields = per_element._set_fields

    result_kwargs = {}
    for f in _dc.fields(cls):
        if f.name in ("LEVEL_FILLS", "LEVEL_STROKES"):
            continue
        for idx, source in enumerate(sources):
            if source is None:
                continue
            val = getattr(source, f.name)
            # Per-element source (idx 0): accept if it has a _set_fields
            # tracker saying this field was explicitly assigned, OR if the
            # value is non-default (backward compatible).
            if idx == 0 and f.name in per_element_fields:
                result_kwargs[f.name] = val
                break
            if val != defaults_dict[f.name]:
                result_kwargs[f.name] = val
                break

    return cls(**result_kwargs)
