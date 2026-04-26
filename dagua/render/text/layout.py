"""Layout helpers for plain and rich text blocks in data coordinates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from dagua.render.text.paths import FontMetrics, GlyphRun, get_font_metrics, text_to_glyphs
from dagua.styles import FONT_FAMILY_MONO
from dagua.utils import _split_rich_lines, parse_rich_markup


@dataclass
class LayoutSegment:
    """One styled run within a laid-out line."""

    glyph_run: GlyphRun
    x_offset: float
    color: str
    is_bold: bool = False
    is_italic: bool = False
    is_mono: bool = False
    underline: bool = False
    strikethrough: bool = False


@dataclass
class LayoutLine:
    """One line of laid-out text."""

    segments: List[LayoutSegment]
    width: float
    height: float
    baseline_y: float


@dataclass
class TextBlock:
    """Complete laid-out text block relative to an anchor point."""

    lines: List[LayoutLine] = field(default_factory=list)
    width: float = 0.0
    height: float = 0.0
    x_offset: float = 0.0
    y_offset: float = 0.0


@dataclass
class _LineBuild:
    """Internal line build before block-level alignment is applied."""

    segments: List[LayoutSegment]
    width: float
    size_data: float


def _effective_max_width(max_width: Optional[float]) -> Optional[float]:
    """Normalize maximum-width constraints for shrink-to-fit layout.

    Parameters
    ----------
    max_width : float | None
        Requested width limit in data units.

    Returns
    -------
    float | None
        Positive width limit or ``None`` when shrinking is disabled.
    """
    if max_width is None or max_width <= 0.0:
        return None
    return float(max_width)


def _fit_size_to_width(
    initial_size: float,
    min_size: Optional[float],
    max_width: Optional[float],
    measure_width: Callable[[float], float],
) -> float:
    """Shrink a line's font size until its measured width fits.

    Parameters
    ----------
    initial_size : float
        Starting font size in data units.
    min_size : float | None
        Minimum allowed size in data units.
    max_width : float | None
        Width limit in data units.
    measure_width : callable
        Width measurement callback for a candidate size.

    Returns
    -------
    float
        Fitted line size in data units.
    """
    safe_initial = max(float(initial_size), 1e-9)
    effective_max_width = _effective_max_width(max_width)
    if effective_max_width is None or measure_width(safe_initial) <= effective_max_width:
        return safe_initial

    floor = max(float(min_size) if min_size is not None else 1e-9, 1e-9)
    floor = min(floor, safe_initial)
    if measure_width(floor) > effective_max_width:
        return floor

    low = floor
    high = safe_initial
    for _ in range(16):
        mid = (low + high) / 2.0
        if measure_width(mid) <= effective_max_width:
            low = mid
        else:
            high = mid
    return low


def _horizontal_offsets(ha: str, block_width: float, line_width: float) -> Tuple[float, float]:
    """Compute block and line offsets for horizontal alignment.

    Parameters
    ----------
    ha : str
        Horizontal alignment name.
    block_width : float
        Maximum width across all lines.
    line_width : float
        Current line width.

    Returns
    -------
    tuple[float, float]
        ``(block_x_offset, line_x_offset)``.
    """
    if ha == "left":
        return 0.0, 0.0
    if ha == "center":
        return -block_width / 2.0, (block_width - line_width) / 2.0
    if ha == "right":
        return -block_width, block_width - line_width
    raise ValueError(f"Unsupported horizontal alignment: {ha!r}")


def _vertical_offset(va: str, block_height: float, metrics: FontMetrics) -> float:
    """Compute a block offset for vertical alignment.

    Parameters
    ----------
    va : str
        Vertical alignment name.
    block_height : float
        Total block height in data units.
    metrics : FontMetrics
        Stable metrics for the first line.

    Returns
    -------
    float
        Block-level y offset.
    """
    if va == "top":
        return 0.0
    if va == "center":
        return block_height / 2.0
    if va == "bottom":
        return block_height
    if va == "baseline":
        return metrics.ascent
    raise ValueError(f"Unsupported vertical alignment: {va!r}")


def _build_text_block(
    line_builds: Sequence[_LineBuild],
    base_metrics: FontMetrics,
    line_spacing: float,
    ha: str,
    va: str,
) -> TextBlock:
    """Convert per-line builds into an aligned text block.

    Parameters
    ----------
    line_builds : sequence[_LineBuild]
        Prepared line data before block alignment.
    base_metrics : FontMetrics
        Stable metrics used for baseline spacing.
    line_spacing : float
        Line-height multiplier.
    ha : str
        Horizontal alignment name.
    va : str
        Vertical alignment name.

    Returns
    -------
    TextBlock
        Fully aligned text block.
    """
    line_height = base_metrics.line_height * line_spacing
    block_width = max((line.width for line in line_builds), default=0.0)
    block_height = max(len(line_builds), 1) * line_height
    block_x_offset, _ = _horizontal_offsets(ha, block_width, block_width)
    y_offset = _vertical_offset(va, block_height, base_metrics)

    lines: List[LayoutLine] = []
    for line_index, line_build in enumerate(line_builds):
        _, line_x_offset = _horizontal_offsets(ha, block_width, line_build.width)
        for segment in line_build.segments:
            segment.x_offset += line_x_offset
        lines.append(
            LayoutLine(
                segments=line_build.segments,
                width=line_build.width,
                height=line_height,
                baseline_y=-base_metrics.ascent - line_index * line_height,
            )
        )

    return TextBlock(
        lines=lines,
        width=block_width,
        height=block_height,
        x_offset=block_x_offset,
        y_offset=y_offset,
    )


def layout_plain_text(
    text: str,
    size_data: float,
    ha: str = "center",
    va: str = "center",
    font_family: str = "sans-serif",
    font_weight: str = "regular",
    font_style: str = "normal",
    font_color: str = "#111111",
    line_spacing: float = 1.2,
    secondary_scale: float = 1.0,
    max_width: Optional[float] = None,
    min_size_data: Optional[float] = None,
) -> TextBlock:
    """Lay out plain text into a multiline text block.

    Parameters
    ----------
    text : str
        Plain text with optional explicit ``\\n`` line breaks.
    size_data : float
        Base font size in data units.
    ha : str, default="center"
        Horizontal alignment.
    va : str, default="center"
        Vertical alignment.
    font_family : str, default="sans-serif"
        Font family name.
    font_weight : str, default="regular"
        Font weight.
    font_style : str, default="normal"
        Font style.
    font_color : str, default="#111111"
        Text color.
    line_spacing : float, default=1.2
        Stable line-height multiplier.
    secondary_scale : float, default=1.0
        Size multiplier for lines after the first.
    max_width : float | None, default=None
        Optional width constraint in data units.
    min_size_data : float | None, default=None
        Minimum shrink-to-fit size in data units.

    Returns
    -------
    TextBlock
        Laid-out text block.
    """
    lines = text.split("\n") if text else [""]
    safe_size = max(float(size_data), 1e-9)
    line_builds: List[_LineBuild] = []

    for line_index, line_text in enumerate(lines):
        line_size = safe_size if line_index == 0 else safe_size * secondary_scale
        fitted_size = _fit_size_to_width(
            line_size,
            min_size_data,
            max_width,
            lambda candidate_size: (
                text_to_glyphs(
                    line_text,
                    candidate_size,
                    font_family=font_family,
                    font_weight=font_weight,
                    font_style=font_style,
                ).advance_width
            ),
        )
        glyph_run = text_to_glyphs(
            line_text,
            fitted_size,
            font_family=font_family,
            font_weight=font_weight,
            font_style=font_style,
        )
        line_builds.append(
            _LineBuild(
                segments=[
                    LayoutSegment(
                        glyph_run=glyph_run,
                        x_offset=0.0,
                        color=font_color,
                    )
                ],
                width=glyph_run.advance_width,
                size_data=fitted_size,
            )
        )

    first_line_size = line_builds[0].size_data if line_builds else safe_size
    base_metrics = get_font_metrics(first_line_size, font_family, font_weight, font_style)
    return _build_text_block(line_builds, base_metrics, line_spacing, ha, va)


def _resolve_segment_style(
    segment_style: Dict[str, object],
    font_family: str,
    font_weight: str,
    font_style: str,
    font_color: str,
) -> Tuple[str, str, str, str]:
    """Resolve font and color properties for one rich-text segment.

    Parameters
    ----------
    segment_style : dict[str, object]
        Parsed markup flags for the segment.
    font_family : str
        Base font family.
    font_weight : str
        Base font weight.
    font_style : str
        Base font style.
    font_color : str
        Base text color.

    Returns
    -------
    tuple[str, str, str, str]
        ``(family, weight, style, color)`` for the segment.
    """
    segment_family = FONT_FAMILY_MONO[0] if bool(segment_style.get("mono")) else font_family
    segment_weight = "bold" if bool(segment_style.get("bold")) else font_weight
    segment_font_style = "italic" if bool(segment_style.get("italic")) else font_style
    segment_color = str(segment_style.get("color") or font_color)
    return segment_family, segment_weight, segment_font_style, segment_color


def layout_rich_text(
    text: str,
    size_data: float,
    ha: str = "center",
    va: str = "center",
    font_family: str = "sans-serif",
    font_weight: str = "regular",
    font_style: str = "normal",
    font_color: str = "#111111",
    line_spacing: float = 1.2,
    secondary_scale: float = 1.0,
    max_width: Optional[float] = None,
    min_size_data: Optional[float] = None,
) -> TextBlock:
    """Lay out rich-text markup into a multiline text block.

    Parameters
    ----------
    text : str
        Rich-text markup string.
    size_data : float
        Base font size in data units.
    ha : str, default="center"
        Horizontal alignment.
    va : str, default="center"
        Vertical alignment.
    font_family : str, default="sans-serif"
        Base font family.
    font_weight : str, default="regular"
        Base font weight.
    font_style : str, default="normal"
        Base font style.
    font_color : str, default="#111111"
        Base text color.
    line_spacing : float, default=1.2
        Stable line-height multiplier.
    secondary_scale : float, default=1.0
        Size multiplier for lines after the first.
    max_width : float | None, default=None
        Optional width constraint in data units.
    min_size_data : float | None, default=None
        Minimum shrink-to-fit size in data units.

    Returns
    -------
    TextBlock
        Laid-out rich text block.
    """
    parsed_segments = parse_rich_markup(text)
    lines = _split_rich_lines(parsed_segments)
    safe_size = max(float(size_data), 1e-9)
    line_builds: List[_LineBuild] = []

    for line_index, line_segments in enumerate(lines):
        line_size = safe_size if line_index == 0 else safe_size * secondary_scale

        def measure_width(candidate_size: float) -> float:
            total_width = 0.0
            for segment_text, segment_style in line_segments:
                family, weight, style_name, _ = _resolve_segment_style(
                    segment_style,
                    font_family,
                    font_weight,
                    font_style,
                    font_color,
                )
                total_width += text_to_glyphs(
                    segment_text,
                    candidate_size,
                    font_family=family,
                    font_weight=weight,
                    font_style=style_name,
                ).advance_width
            return total_width

        fitted_size = _fit_size_to_width(line_size, min_size_data, max_width, measure_width)
        segments: List[LayoutSegment] = []
        x_offset = 0.0
        for segment_text, segment_style in line_segments:
            family, weight, style_name, color = _resolve_segment_style(
                segment_style,
                font_family,
                font_weight,
                font_style,
                font_color,
            )
            glyph_run = text_to_glyphs(
                segment_text,
                fitted_size,
                font_family=family,
                font_weight=weight,
                font_style=style_name,
            )
            segments.append(
                LayoutSegment(
                    glyph_run=glyph_run,
                    x_offset=x_offset,
                    color=color,
                    is_bold=bool(segment_style.get("bold")),
                    is_italic=bool(segment_style.get("italic")),
                    is_mono=bool(segment_style.get("mono")),
                    underline=bool(segment_style.get("underline")),
                    strikethrough=bool(segment_style.get("strike")),
                )
            )
            x_offset += glyph_run.advance_width
        line_builds.append(_LineBuild(segments=segments, width=x_offset, size_data=fitted_size))

    first_line_size = line_builds[0].size_data if line_builds else safe_size
    base_metrics = get_font_metrics(first_line_size, font_family, font_weight, font_style)
    return _build_text_block(line_builds, base_metrics, line_spacing, ha, va)
