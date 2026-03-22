"""Path generation and font metrics for data-coordinate text rendering."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Tuple

import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.textpath import TextToPath

from dagua.styles import RESOLVED_FONT

FONT_SCALE = 100.0
LINE_SPACING = 1.2
_CACHE_PRECISION = 3
_TEXT_TO_PATH = TextToPath()


@dataclass(frozen=True)
class FontMetrics:
    """Stable font metrics independent of glyph content.

    Parameters
    ----------
    ascent : float
        Distance from baseline to the tallest ascender in data units.
    descent : float
        Distance from baseline to descenders in data units.
    line_height : float
        Stable line height for layout in data units.
    x_height : float
        Lowercase x-height in data units.
    space_width : float
        Advance width of a single ASCII space in data units.
    """

    ascent: float
    descent: float
    line_height: float
    x_height: float
    space_width: float


@dataclass(frozen=True)
class GlyphRun:
    """One styled run of text converted to a filled path.

    Parameters
    ----------
    path : matplotlib.path.Path
        Glyph outlines positioned at the origin in data units.
    advance_width : float
        Typographic advance width in data units.
    metrics : FontMetrics
        Stable font metrics for the run's font.
    text : str
        Original text content.
    font_size_pts : float
        Original font size in typographic points.
    """

    path: Path
    advance_width: float
    metrics: FontMetrics
    text: str
    font_size_pts: float


def _resolve_font_family(font_family: str) -> str:
    """Resolve empty font-family requests to Dagua's default font.

    Parameters
    ----------
    font_family : str
        Requested family name.

    Returns
    -------
    str
        Concrete family name.
    """
    return font_family or RESOLVED_FONT


def _build_font_properties(
    font_family: str,
    font_weight: str,
    font_style: str,
    size: float,
) -> FontProperties:
    """Build matplotlib font properties for text-path conversion.

    Parameters
    ----------
    font_family : str
        Concrete font family name.
    font_weight : str
        Font weight name.
    font_style : str
        Font style name.
    size : float
        Font size in data units or points, depending on caller context.

    Returns
    -------
    FontProperties
        Matplotlib font properties object.
    """
    return FontProperties(
        family=_resolve_font_family(font_family),
        weight=font_weight,
        style=_normalize_font_style(font_style),
        size=size,
    )


def _normalize_font_style(font_style: str) -> Literal["normal", "italic", "oblique"]:
    """Clamp arbitrary font-style strings to matplotlib's supported literals.

    Parameters
    ----------
    font_style : str
        Requested font style.

    Returns
    -------
    Literal["normal", "italic", "oblique"]
        Style accepted by ``FontProperties``.
    """
    if font_style == "italic":
        return "italic"
    if font_style == "oblique":
        return "oblique"
    return "normal"


def _empty_path() -> Path:
    """Return a reusable empty path instance.

    Returns
    -------
    matplotlib.path.Path
        Empty path with numpy-backed vertices and codes.
    """
    return Path(np.empty((0, 2), dtype=float), np.empty((0,), dtype=np.uint8))


def _normalize_text_path_payload(
    vertices: object,
    codes: object,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert ``TextToPath`` outputs into numpy arrays.

    Parameters
    ----------
    vertices : object
        Vertex payload returned by ``TextToPath``.
    codes : object
        Code payload returned by ``TextToPath``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Vertex array with shape ``[N, 2]`` and code array with shape ``[N]``.
    """
    vertex_array = np.asarray(vertices, dtype=float).reshape(-1, 2)
    code_array = np.asarray(codes, dtype=np.uint8).reshape(-1)
    return vertex_array, code_array


def _round_size(size_data: float) -> float:
    """Round data-unit font sizes for stable metric cache keys.

    Parameters
    ----------
    size_data : float
        Font size in data units.

    Returns
    -------
    float
        Rounded cache key.
    """
    return round(float(size_data), _CACHE_PRECISION)


def get_font_metrics(
    size_data: float,
    font_family: str = "sans-serif",
    font_weight: str = "regular",
    font_style: str = "normal",
) -> FontMetrics:
    """Compute stable font metrics for layout decisions.

    Parameters
    ----------
    size_data : float
        Font size in data units.
    font_family : str, default="sans-serif"
        Font family name.
    font_weight : str, default="regular"
        Font weight.
    font_style : str, default="normal"
        Font style.

    Returns
    -------
    FontMetrics
        Stable metrics in data coordinates.
    """
    resolved_family = _resolve_font_family(font_family)
    rounded_size = max(_round_size(size_data), 1e-9)
    ascent, descent, line_height, x_height, space_width = _cached_font_metrics(
        resolved_family,
        font_weight,
        font_style,
        rounded_size,
    )
    return FontMetrics(
        ascent=ascent,
        descent=descent,
        line_height=line_height,
        x_height=x_height,
        space_width=space_width,
    )


@lru_cache(maxsize=4096)
def _cached_font_metrics(
    font_family: str,
    font_weight: str,
    font_style: str,
    size_data: float,
) -> Tuple[float, float, float, float, float]:
    """Cache stable font metrics for one font/size combination.

    Parameters
    ----------
    font_family : str
        Concrete font family name.
    font_weight : str
        Font weight.
    font_style : str
        Font style.
    size_data : float
        Rounded font size in data units.

    Returns
    -------
    tuple[float, float, float, float, float]
        ``(ascent, descent, line_height, x_height, space_width)``.
    """
    font_properties = _build_font_properties(font_family, font_weight, font_style, size_data)
    _, hg_height, hg_descent = _TEXT_TO_PATH.get_text_width_height_descent(
        "Hg",
        font_properties,
        ismath=False,
    )
    _, x_height, _ = _TEXT_TO_PATH.get_text_width_height_descent(
        "x",
        font_properties,
        ismath=False,
    )
    width_with_space, _, _ = _TEXT_TO_PATH.get_text_width_height_descent(
        "x x",
        font_properties,
        ismath=False,
    )
    width_without_space, _, _ = _TEXT_TO_PATH.get_text_width_height_descent(
        "xx",
        font_properties,
        ismath=False,
    )
    line_height = max(float(hg_height), 1e-9)
    descent = max(float(hg_descent), 0.0)
    ascent = max(line_height - descent, 0.0)
    return (
        ascent,
        descent,
        line_height,
        max(float(x_height), 0.0),
        max(float(width_with_space - width_without_space), 0.0),
    )


def text_to_glyphs(
    text: str,
    size_data: float,
    font_family: str = "sans-serif",
    font_weight: str = "regular",
    font_style: str = "normal",
    font_size_pts: float = 0.0,
) -> GlyphRun:
    """Convert text to a filled glyph path in data coordinates.

    Parameters
    ----------
    text : str
        Text to convert. Newlines are not supported.
    size_data : float
        Font size in data units.
    font_family : str, default="sans-serif"
        Font family name.
    font_weight : str, default="regular"
        Font weight.
    font_style : str, default="normal"
        Font style.
    font_size_pts : float, default=0.0
        Original typographic point size retained for future selectable overlays.

    Returns
    -------
    GlyphRun
        Glyph outlines and stable metrics in data coordinates.
    """
    if "\n" in text:
        raise ValueError("text_to_glyphs() does not support embedded newlines.")

    resolved_family = _resolve_font_family(font_family)
    safe_size = max(float(size_data), 1e-9)
    metrics = get_font_metrics(safe_size, resolved_family, font_weight, font_style)
    ref_vertices, ref_codes, advance_width_ref = _cached_glyph_data(
        text,
        resolved_family,
        font_weight,
        font_style,
    )
    scale = safe_size / FONT_SCALE
    vertices = ref_vertices * scale
    advance_width = advance_width_ref * scale
    if vertices.size == 0:
        path = _empty_path()
    else:
        path = Path(vertices, ref_codes)
    return GlyphRun(
        path=path,
        advance_width=advance_width,
        metrics=metrics,
        text=text,
        font_size_pts=float(font_size_pts),
    )


@lru_cache(maxsize=16384)
def _cached_glyph_data(
    text: str,
    family: str,
    weight: str,
    style: str,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Cache glyph outlines and advance widths at ``FONT_SCALE``.

    Parameters
    ----------
    text : str
        Text content without newlines.
    family : str
        Concrete font family.
    weight : str
        Font weight.
    style : str
        Font style.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, float]
        ``(vertices, codes, advance_width_ref)`` at ``FONT_SCALE``.
    """
    font_properties = _build_font_properties(family, weight, style, FONT_SCALE)
    raw_vertices, raw_codes = _TEXT_TO_PATH.get_text_path(font_properties, text, ismath=False)
    vertices, codes = _normalize_text_path_payload(raw_vertices, raw_codes)
    advance_width_ref, _, _ = _TEXT_TO_PATH.get_text_width_height_descent(
        text,
        font_properties,
        ismath=False,
    )
    return vertices, codes, float(advance_width_ref)


def _measure_line_advance(
    text: str,
    size_data: float,
    font_family: str,
    font_weight: str,
    font_style: str,
) -> float:
    """Measure typographic advance width for one line of text.

    Parameters
    ----------
    text : str
        Single-line text.
    size_data : float
        Font size in data units.
    font_family : str
        Concrete font family.
    font_weight : str
        Font weight.
    font_style : str
        Font style.

    Returns
    -------
    float
        Advance width in data units.
    """
    font_properties = _build_font_properties(font_family, font_weight, font_style, size_data)
    advance_width, _, _ = _TEXT_TO_PATH.get_text_width_height_descent(
        text,
        font_properties,
        ismath=False,
    )
    return float(advance_width)


def measure_text_data(
    text: str,
    size_data: float,
    font_family: str = "sans-serif",
    font_weight: str = "regular",
    font_style: str = "normal",
) -> Tuple[float, float]:
    """Measure text in data coordinates using the same metrics as glyph conversion.

    Parameters
    ----------
    text : str
        Text content. Explicit newlines create multiple lines.
    size_data : float
        Font size in data units.
    font_family : str, default="sans-serif"
        Font family name.
    font_weight : str, default="regular"
        Font weight.
    font_style : str, default="normal"
        Font style.

    Returns
    -------
    tuple[float, float]
        Width and height in data units.
    """
    resolved_family = _resolve_font_family(font_family)
    safe_size = max(float(size_data), 1e-9)
    metrics = get_font_metrics(safe_size, resolved_family, font_weight, font_style)
    lines = text.split("\n")
    if len(lines) == 1:
        glyph_run = text_to_glyphs(
            text,
            safe_size,
            font_family=resolved_family,
            font_weight=font_weight,
            font_style=font_style,
        )
        return glyph_run.advance_width, glyph_run.metrics.line_height

    max_width = 0.0
    for line in lines:
        max_width = max(
            max_width,
            _measure_line_advance(line, safe_size, resolved_family, font_weight, font_style),
        )
    total_height = max(len(lines), 1) * metrics.line_height * LINE_SPACING
    return max_width, total_height
