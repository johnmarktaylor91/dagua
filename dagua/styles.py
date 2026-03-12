"""NodeStyle, EdgeStyle, ClusterStyle, GraphStyle, Theme dataclasses, color utilities.

Color system: Wong/Okabe-Ito colorblind-safe palette (Wong, B. 2011. Nature Methods 8:441).
Typography: Helvetica/Arial sans-serif per Nature/Science figure guidelines.
Aesthetics: publication-quality defaults — muted fills, strong borders, quiet edges.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field, fields as dataclass_fields
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
        from matplotlib.font_manager import findfont, FontProperties
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
    l = (mx + mn) / 2.0
    if mx == mn:
        h = s = 0.0
    else:
        d = mx - mn
        s = d / (2.0 - mx - mn) if l > 0.5 else d / (mx + mn)
        if mx == r:
            h = (g - b) / d + (6.0 if g < b else 0.0)
        elif mx == g:
            h = (b - r) / d + 2.0
        else:
            h = (r - g) / d + 4.0
        h /= 6.0
    return h, s, l


def _hsl_to_hex(h: float, s: float, l: float) -> str:
    if s == 0:
        v = int(round(l * 255))
        return _rgb_to_hex(v, v, v)

    def hue_to_rgb(p, q, t):
        if t < 0: t += 1
        if t > 1: t -= 1
        if t < 1/6: return p + (q - p) * 6 * t
        if t < 1/2: return q
        if t < 2/3: return p + (q - p) * (2/3 - t) * 6
        return p

    q = l * (1 + s) if l < 0.5 else l + s - l * s
    p = 2 * l - q
    r = int(round(hue_to_rgb(p, q, h + 1/3) * 255))
    g = int(round(hue_to_rgb(p, q, h) * 255))
    b = int(round(hue_to_rgb(p, q, h - 1/3) * 255))
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
    h, s, l = _hex_to_hsl(fill_hex)
    return _hsl_to_hex(h, min(s * 1.2, 1.0), l * (1 - darken))


def darken_hex(hex_color: str, amount: float) -> str:
    """Darken a hex color by reducing HSL lightness by `amount` (0-1 scale)."""
    h, s, l = _hex_to_hsl(hex_color)
    return _hsl_to_hex(h, s, max(l - amount, 0.0))


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
    """Visual style for a node."""

    shape: str = "roundrect"  # rect, roundrect, ellipse, diamond, circle
    fill: str = ""  # empty = computed from base_color
    stroke: str = ""  # empty = computed from base_color
    stroke_width: float = 0.57
    stroke_dash: str = "solid"  # solid, dashed
    font_family: str = ""  # empty = use FONT_FAMILY default
    font_size: float = 9.0
    font_color: str = NEAR_BLACK
    padding: Tuple[float, float] = (10.0, 6.0)  # horizontal, vertical
    corner_radius: float = 6.0
    opacity: float = 1.0
    base_color: str = PALETTE["sky"]  # Wong palette color
    # New fields (Part 2)
    font_weight: str = "regular"  # Layout-affecting: changes text width
    font_style: str = "normal"  # normal, italic — render-only
    shadow: bool = False  # render-only decoration
    shadow_offset: Tuple[float, float] = (1.5, -1.5)  # render-only
    shadow_color: str = "#00000020"  # render-only
    min_width: Optional[float] = None  # Layout-affecting: floor on node width
    # New fields (Part 3) — overflow policy
    overflow_policy: str = "shrink_text"  # "shrink_text", "expand_node", "overflow"
    min_font_size: float = 5.0  # Floor for shrink_text policy

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
    arrow: str = "normal"  # normal, none
    arrow_length: float = 10.0
    arrow_width: float = 7.0
    style: str = "solid"  # solid, dashed, dotted
    opacity: float = 0.65
    # New fields (Part 2)
    routing: str = "bezier"  # bezier, straight, ortho — post-layout
    label_font_size: float = 7.0  # render-only
    label_font_color: str = NEAR_BLACK  # render-only
    label_background: str = WARM_WHITE  # render-only
    # New fields (Part 3) — edge aesthetics
    label_position: float = 0.5  # Position along curve (0=start, 1=end)
    curvature: float = 0.4  # Control point offset factor (0=straight, 1=max curve)
    port_style: str = "distributed"  # "distributed" or "center"
    label_avoidance: bool = True  # Whether to avoid label collisions


@dataclass
class ClusterStyle:
    """Visual style for a cluster box."""

    fill: str = PAPER
    stroke: str = LIGHT_GRAY
    stroke_width: float = 0.7
    stroke_dash: str = "solid"
    corner_radius: float = 8.0
    padding: float = 25.0
    label_position: str = "top-left"  # top-left, top-center, top-right
    font_size: float = 9.5
    font_weight: str = "bold"
    font_color: str = DARK_GRAY
    opacity: float = 0.32
    # New fields (Part 2)
    font_family: str = ""  # empty = use FONT_FAMILY default, render-only
    label_offset: Tuple[float, float] = (8.0, 20.0)  # render-only (y-offset prevents nested label overlap)
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
    "input": NodeStyle(base_color=PALETTE["bluish_green"], padding=(14.0, 8.0)),
    "output": NodeStyle(base_color=PALETTE["vermillion"], padding=(14.0, 8.0)),
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


# ─── Theme Registry ──────────────────────────────────────────────────────

THEME_REGISTRY: Dict[str, Theme] = {
    "default": DEFAULT_THEME_OBJ,
    "dark": DARK_THEME,
    "minimal": MINIMAL_THEME,
}


def get_theme(name: str) -> Theme:
    """Look up a built-in theme by name. Returns a deep copy."""
    if name not in THEME_REGISTRY:
        raise ValueError(
            f"Unknown theme: {name!r}. Available: {list(THEME_REGISTRY.keys())}"
        )
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
