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
    fill_pattern: str = "solid"  # render-only: solid, striped, hatched
    fill_pattern_colors: Optional[List[str]] = None  # render-only stripe palette
    fill_pattern_angle: float = 0.0  # render-only stripe angle in degrees
    image: str = ""  # render-only path or URL for node image content
    image_fit: str = "contain"  # render-only: contain, cover, stretch
    image_opacity: float = 1.0  # render-only alpha for the image layer

    def __post_init__(self):
        if not self.fill:
            self.fill = make_fill(self.base_color)
        if not self.stroke:
            self.stroke = border_from_fill(self.base_color, darken=0.4)
        if not self.font_family:
            self.font_family = RESOLVED_FONT

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
    width: float = 1.2
    arrow: str = "normal"  # normal, vee, dot, diamond, tee, crow, circle, open, none
    tail_arrow: str = "none"
    arrow_fill: str = "filled"  # filled, hollow
    arrow_color: str = ""  # empty = use edge color
    arrow_length: float = 10.0
    arrow_width: float = 7.0
    arrow_scale: Optional[float] = None  # Legacy field; matplotlib renderer ignores it
    arrow_node_fraction: float = (
        0.0  # 0 = use fixed arrow_length; >0 = fraction of target node height
    )
    arrow_width_ratio: float = 0.7  # width = length * this ratio (for node-relative mode)
    style: str = "solid"  # solid, dashed, dotted
    line_cap: str = "butt"  # render-only: butt, round, square
    line_join: str = "miter"  # render-only: miter, bevel, round
    opacity: float = 0.65
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
    # Member style overrides — applied to all nodes/edges within this cluster
    member_node_style: Optional[NodeStyle] = None
    member_edge_style: Optional[EdgeStyle] = None

    # Legacy constants kept for reference but replaced by depth_*_step
    LEVEL_FILLS = [PAPER, "#EDEDE8", "#E5E5E0"]
    LEVEL_STROKES = [LIGHT_GRAY, "#C8C8C8", "#BCBCBC"]


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
    "stroke_width": 1.4,
    "padding": (7.0, 4.0),
    "font_size": 12.0,
    "arrow_length": 10.0,
    "arrow_width": 7.0,
    "arrow_scale": 16.0,
    "edge_width": 1.4,
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
            arrow_node_fraction=0.24,  # slightly smaller than strict
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

    For each field, picks the first non-default value from the sources list.
    Falls back to the class default if no source overrides a field.
    """
    import dataclasses as _dc

    defaults_instance = cls()
    defaults_dict = {f.name: getattr(defaults_instance, f.name) for f in _dc.fields(cls)}

    result_kwargs = {}
    for f in _dc.fields(cls):
        # Skip class-level constants (not constructor params)
        if f.name in ("LEVEL_FILLS", "LEVEL_STROKES"):
            continue
        for source in sources:
            if source is None:
                continue
            val = getattr(source, f.name)
            if val != defaults_dict[f.name]:
                result_kwargs[f.name] = val
                break

    return cls(**result_kwargs)
