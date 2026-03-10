"""NodeStyle, EdgeStyle, ClusterStyle dataclasses, color utilities, and theme system.

Color system: Wong/Okabe-Ito colorblind-safe palette (Wong, B. 2011. Nature Methods 8:441).
Typography: Helvetica/Arial sans-serif per Nature/Science figure guidelines.
Aesthetics: publication-quality defaults — muted fills, strong borders, quiet edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


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
MEDIUM_GRAY = "#8C8C8C"
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
    stroke_width: float = 0.75
    stroke_dash: str = "solid"  # solid, dashed
    font_family: str = ""  # empty = use FONT_FAMILY default
    font_size: float = 8.5
    font_color: str = NEAR_BLACK
    padding: Tuple[float, float] = (8.0, 5.0)  # horizontal, vertical
    corner_radius: float = 4.0
    opacity: float = 1.0
    base_color: str = PALETTE["sky"]  # Wong palette color

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

    color: str = MEDIUM_GRAY
    width: float = 0.75
    arrow: str = "normal"  # normal, none
    arrow_length: float = 5.0
    arrow_width: float = 3.5
    style: str = "solid"  # solid, dashed, dotted
    opacity: float = 0.7


@dataclass
class ClusterStyle:
    """Visual style for a cluster box."""

    fill: str = PAPER
    stroke: str = LIGHT_GRAY
    stroke_width: float = 0.5
    stroke_dash: str = "solid"
    corner_radius: float = 7.0
    padding: float = 18.0
    label_position: str = "top-left"
    font_size: float = 9.5
    font_weight: str = "medium"
    font_color: str = DARK_GRAY
    opacity: float = 0.4

    # Nested cluster level colors (progressively darken)
    LEVEL_FILLS = [PAPER, "#EDEDE8", "#E5E5E0"]
    LEVEL_STROKES = [LIGHT_GRAY, "#C8C8C8", "#BCBCBC"]


# ─── Theme System ───────────────────────────────────────────────────────────

# Default: all nodes use Sky blue (Wong palette default)
_sky_fill, _sky_stroke = make_node_colors(PALETTE["sky"])
_blue_fill, _blue_stroke = make_node_colors(PALETTE["blue"])
_green_fill, _green_stroke = make_node_colors(PALETTE["bluish_green"])
_vermillion_fill, _vermillion_stroke = make_node_colors(PALETTE["vermillion"])
_amber_fill, _amber_stroke = make_node_colors(PALETTE["amber"])
_purple_fill, _purple_stroke = make_node_colors(PALETTE["reddish_purple"])
_yellow_fill, _yellow_stroke = make_node_colors(PALETTE["yellow"])

DEFAULT_THEME: Dict[str, NodeStyle] = {
    "default": NodeStyle(base_color=PALETTE["sky"]),
    "input": NodeStyle(base_color=PALETTE["bluish_green"]),
    "output": NodeStyle(base_color=PALETTE["vermillion"]),
    "buffer": NodeStyle(base_color=MEDIUM_GRAY),
    "bool": NodeStyle(base_color=PALETTE["amber"]),
    "trainable_params": NodeStyle(base_color=PALETTE["blue"]),
    "frozen_params": NodeStyle(base_color=MEDIUM_GRAY),
    "mixed_params": NodeStyle(base_color=PALETTE["reddish_purple"]),
    "module": NodeStyle(base_color=PALETTE["blue"]),
}

# Legacy Graphviz-matching theme (for backwards compatibility)
GRAPHVIZ_MATCH_THEME: Dict[str, NodeStyle] = {
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
