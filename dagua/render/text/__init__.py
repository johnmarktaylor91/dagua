"""Public API for data-coordinate text rendering.

This module converts text into filled geometric paths in data coordinates so
labels scale with zoom and pan like other graph geometry.

Limitations
-----------
- Requires equal-aspect axes.
- Rich-text segments are positioned independently, so inter-segment kerning is
  not preserved across style boundaries.
- Only explicit newline breaks are supported.
- Complex shaping (RTL, Arabic, Indic, emoji/color fonts) is out of scope.
"""

from dagua.render.text.collection import DaguaText, render_text
from dagua.render.text.decorations import background_rect_path, strikethrough_path, underline_path
from dagua.render.text.layout import (
    LayoutLine,
    LayoutSegment,
    TextBlock,
    layout_plain_text,
    layout_rich_text,
)
from dagua.render.text.paths import (
    FontMetrics,
    GlyphRun,
    get_font_metrics,
    measure_text_data,
    text_to_glyphs,
)

__all__ = [
    "DaguaText",
    "FontMetrics",
    "GlyphRun",
    "LayoutLine",
    "LayoutSegment",
    "TextBlock",
    "background_rect_path",
    "get_font_metrics",
    "layout_plain_text",
    "layout_rich_text",
    "measure_text_data",
    "render_text",
    "strikethrough_path",
    "text_to_glyphs",
    "underline_path",
]
