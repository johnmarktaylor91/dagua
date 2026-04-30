"""Artist creation for data-coordinate text labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.transforms import Affine2D

from dagua.render.edges.ribbon import polyline_ribbon_path
from dagua.render.text.decorations import (
    background_rect_path,
    strikethrough_path,
    underline_path,
)
from dagua.render.text.layout import (
    LayoutLine,
    LayoutSegment,
    TextBlock,
    layout_plain_text,
    layout_rich_text,
)
from dagua.render.text.paths import text_to_glyphs
from dagua.styles import RESOLVED_FONT
from dagua.utils import prepare_label_text

_MIN_VISIBLE_OUTLINE_WIDTH_POINTS = 2.0
_BOLD_EMPHASIS_WIDTH_RATIO = 0.08
_MIN_BOLD_EMPHASIS_WIDTH_POINTS = 0.75
_POINTS_PER_INCH = 72.0


@dataclass
class DaguaText:
    """Specification for one text element rendered as filled paths."""

    x: float
    y: float
    text: str
    font_size: float = 9.0
    font_family: str = ""
    font_weight: str = "regular"
    font_style: str = "normal"
    font_color: str = "#111111"
    alpha: float = 1.0
    ha: str = "center"
    va: str = "center"
    rotation: float = 0.0
    rich: bool = False
    line_spacing: float = 1.2
    secondary_scale: float = 1.0
    max_width: Optional[float] = None
    min_font_size: float = 5.0
    text_wrap: str = "none"
    text_max_width: Optional[float] = None
    text_transform: str = "none"
    outline: bool = False
    outline_color: str = "#FFFFFF"
    outline_width: float = 2.0
    background: Optional[str] = None
    background_alpha: float = 0.85
    background_padding: Tuple[float, float] = (3.0, 2.0)
    background_corner_radius: float = 2.0
    underline: bool = False
    strikethrough: bool = False
    clip_patch: Any = None
    clip_on: bool = True
    zorder: float = 3.0
    gid: Optional[str] = None


def _apply_clip(patch: PathPatch, spec: DaguaText) -> None:
    """Apply clip-path settings to an artist.

    Parameters
    ----------
    patch : PathPatch
        Patch to configure.
    spec : DaguaText
        Source render specification.
    """
    if spec.clip_patch is not None:
        patch.set_clip_path(spec.clip_patch)
    patch.set_clip_on(spec.clip_on)


def _transform_path(path: Path, spec: DaguaText) -> Path:
    """Rotate a path around the label anchor when requested.

    Parameters
    ----------
    path : Path
        Path in absolute data coordinates.
    spec : DaguaText
        Render specification providing anchor and rotation.

    Returns
    -------
    Path
        Transformed path.
    """
    if spec.rotation == 0.0:
        return path
    transform = Affine2D().rotate_deg_around(spec.x, spec.y, spec.rotation)
    return path.transformed(transform)


def _background_zorder(spec: DaguaText) -> float:
    """Return the background patch z-order for one text spec.

    Parameters
    ----------
    spec : DaguaText
        Text render specification.

    Returns
    -------
    float
        Z-order used for the background patch.
    """
    if isinstance(spec.gid, str) and spec.gid.startswith("dagua-cluster-label-"):
        return float(spec.zorder)
    return float(spec.zorder) - 0.1


def _segment_path(
    spec: DaguaText,
    block: TextBlock,
    line: LayoutLine,
    segment: LayoutSegment,
) -> Path:
    """Position one glyph run in data coordinates.

    Parameters
    ----------
    spec : DaguaText
        Render specification.
    block : TextBlock
        Laid-out text block.
    line : LayoutLine
        Owning line.
    segment : LayoutSegment
        Segment to position.

    Returns
    -------
    Path
        Positioned segment path.
    """
    translate = Affine2D().translate(
        spec.x + block.x_offset + segment.x_offset,
        spec.y + block.y_offset + line.baseline_y,
    )
    return _transform_path(segment.glyph_run.path.transformed(translate), spec)


def _register_hover(
    svg_hover_map: Optional[Dict[str, str]],
    gid: Optional[str],
    text: str,
) -> None:
    """Register hover text for SVG export when an id is available.

    Parameters
    ----------
    svg_hover_map : dict[str, str] | None
        Hover-text accumulator.
    gid : str | None
        Artist group id.
    text : str
        Hover text content.
    """
    if svg_hover_map is not None and gid:
        svg_hover_map.setdefault(gid, text)


def _path_is_empty(path: Path) -> bool:
    """Return whether a matplotlib path has any drawable vertices.

    Parameters
    ----------
    path : Path
        Path to inspect.

    Returns
    -------
    bool
        ``True`` when the path has no vertices.
    """
    return np.asarray(path.vertices).size == 0


def _segment_gid(
    spec: DaguaText,
    line_index: int,
    segment_index: int,
    line_count: int,
) -> Optional[str]:
    """Derive the artist gid for one rendered segment.

    Parameters
    ----------
    spec : DaguaText
        Source render specification.
    line_index : int
        Zero-based line index.
    segment_index : int
        Zero-based segment index.
    line_count : int
        Total number of lines in the block.

    Returns
    -------
    str | None
        Derived gid for the artist.
    """
    if spec.gid is None:
        return None
    if spec.rich:
        return f"{spec.gid}-{line_index}-{segment_index}"
    if line_count > 1:
        return f"{spec.gid}-{line_index}"
    return spec.gid


def _patch_gid(
    spec: DaguaText,
    role: str,
    line_index: Optional[int] = None,
    segment_index: Optional[int] = None,
) -> Optional[str]:
    """Derive a unique gid for a non-fill patch within one text spec.

    Parameters
    ----------
    spec : DaguaText
        Source render specification.
    role : str
        Patch role such as ``"background"`` or ``"outline"``.
    line_index : int | None, default=None
        Optional line index for per-line patches.
    segment_index : int | None, default=None
        Optional segment index for per-segment patches.

    Returns
    -------
    str | None
        Unique gid or ``None`` when the spec is not hover-addressable.
    """
    if spec.gid is None:
        return None
    suffix = role
    if line_index is not None:
        suffix = f"{suffix}-{line_index}"
    if segment_index is not None:
        suffix = f"{suffix}-{segment_index}"
    return f"{spec.gid}-{suffix}"


def _add_patch(ax: Any, patch: PathPatch, spec: DaguaText, artists: List[Any]) -> None:
    """Attach a patch artist to the axes and result list.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    patch : PathPatch
        Patch to add.
    spec : DaguaText
        Source render specification.
    artists : list[Any]
        Output artist accumulator.
    """
    _apply_clip(patch, spec)
    ax.add_patch(patch)
    artists.append(patch)


def _matplotlib_text_background_path(
    ax: Any,
    spec: DaguaText,
    prepared_text: str,
    font_family: str,
    font_size_pts: float,
) -> Optional[Path]:
    """Return a data-space text background from Matplotlib's rasterized bbox.

    Parameters
    ----------
    ax : Any
        Axes used to resolve display and data transforms.
    spec : DaguaText
        Text render specification.
    prepared_text : str
        Final label string after wrapping and transforms.
    font_family : str
        Font family used for raster text measurement.
    font_size_pts : float
        Font size in typographic points.

    Returns
    -------
    matplotlib.path.Path | None
        Absolute data-coordinate background path, or ``None`` when the
        fallback vector-layout measurement should be used.
    """

    if spec.rich or abs(float(spec.rotation)) > 1e-9:
        return None
    renderer = ax.figure.canvas.get_renderer()
    if renderer is None:
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
    probe = ax.text(
        spec.x,
        spec.y,
        prepared_text,
        fontsize=font_size_pts,
        fontfamily=font_family,
        fontweight=spec.font_weight,
        fontstyle=spec.font_style,
        ha=spec.ha,
        va=spec.va,
        alpha=0.0,
    )
    try:
        bbox = probe.get_window_extent(renderer=renderer)
    finally:
        probe.remove()
    pad_x = spec.background_padding[0] * ax.figure.dpi / _POINTS_PER_INCH
    pad_y = spec.background_padding[1] * ax.figure.dpi / _POINTS_PER_INCH
    x0 = bbox.x0 - pad_x
    x1 = bbox.x1 + pad_x
    y0 = bbox.y0 - pad_y
    y1 = bbox.y1 + pad_y
    corners = ax.transData.inverted().transform(
        [
            [x0, y0],
            [x1, y1],
        ]
    )
    x_min, y_min = corners[0]
    x_max, y_max = corners[1]
    radius_px = spec.background_corner_radius * ax.figure.dpi / _POINTS_PER_INCH
    radius_corners = ax.transData.inverted().transform([[0.0, 0.0], [radius_px, radius_px]])
    corner_radius = abs(float(radius_corners[1][0]) - float(radius_corners[0][0]))
    return background_rect_path(
        (float(x_min) + float(x_max)) / 2.0,
        (float(y_min) + float(y_max)) / 2.0,
        abs(float(x_max) - float(x_min)),
        abs(float(y_max) - float(y_min)),
        0.0,
        0.0,
        corner_radius,
    )


def _is_bold_weight(font_weight: str) -> bool:
    """Return whether a font-weight token requests a bold face.

    Parameters
    ----------
    font_weight : str
        Requested weight token.

    Returns
    -------
    bool
        ``True`` when the token maps to a visibly bold weight.
    """
    return font_weight.strip().lower() in {
        "bold",
        "demibold",
        "semibold",
        "heavy",
        "black",
        "extrabold",
    }


def _segment_needs_bold_emphasis(spec: DaguaText, segment: LayoutSegment) -> bool:
    """Return whether one rendered segment should receive synthetic emboldening.

    Parameters
    ----------
    spec : DaguaText
        Source render specification.
    segment : LayoutSegment
        Laid-out text segment.

    Returns
    -------
    bool
        ``True`` when the segment should receive a same-color stroke behind the
        fill so bold text stays visibly heavier even with fallback fonts.
    """
    resolved_weight = "bold" if segment.is_bold else spec.font_weight
    if not _is_bold_weight(resolved_weight):
        return False

    resolved_family = spec.font_family or RESOLVED_FONT
    resolved_style = "italic" if segment.is_italic else spec.font_style
    regular_run = text_to_glyphs(
        segment.glyph_run.text,
        segment.glyph_run.size_data,
        font_family=resolved_family,
        font_weight="normal",
        font_style=resolved_style,
    )
    width_delta = abs(float(segment.glyph_run.advance_width) - float(regular_run.advance_width))
    if width_delta > max(float(regular_run.advance_width) * 0.02, 1e-3):
        return False
    if segment.glyph_run.path.vertices.shape != regular_run.path.vertices.shape:
        return False
    return bool(
        np.allclose(
            segment.glyph_run.path.vertices,
            regular_run.path.vertices,
            atol=1e-3,
        )
    )


def _outline_linewidth(spec: DaguaText) -> float:
    """Return a visible outline width for one text specification.

    Parameters
    ----------
    spec : DaguaText
        Text render specification.

    Returns
    -------
    float
        Outline stroke width in typographic points.
    """
    return max(float(spec.outline_width), _MIN_VISIBLE_OUTLINE_WIDTH_POINTS)


def _bold_emphasis_linewidth(spec: DaguaText) -> float:
    """Return the synthetic stroke width used to reinforce bold glyphs.

    Parameters
    ----------
    spec : DaguaText
        Text render specification.

    Returns
    -------
    float
        Stroke width in typographic points.
    """
    return max(
        float(spec.font_size) * _BOLD_EMPHASIS_WIDTH_RATIO,
        _MIN_BOLD_EMPHASIS_WIDTH_POINTS,
    )


def _dedupe_polyline(points: np.ndarray) -> np.ndarray:
    """Remove adjacent duplicate points from a flattened path contour.

    Parameters
    ----------
    points : numpy.ndarray
        Contour vertices with shape ``[N, 2]``.

    Returns
    -------
    numpy.ndarray
        De-duplicated contour vertices with shape ``[M, 2]``.
    """
    if points.shape[0] <= 1:
        return points
    keep = [True]
    for index in range(1, points.shape[0]):
        keep.append(not np.allclose(points[index], points[index - 1]))
    return points[np.array(keep, dtype=bool)]


def _glyph_stroke_ribbon_paths(path: Path, stroke_width: float) -> List[Path]:
    """Build filled data-coordinate ribbons for a glyph-outline stroke.

    Parameters
    ----------
    path : Path
        Positioned glyph outline path in data coordinates.
    stroke_width : float
        Stroke body width in data units.

    Returns
    -------
    list[Path]
        Filled ribbon polygons approximating the glyph stroke.
    """
    if stroke_width <= 0.0 or _path_is_empty(path):
        return []

    ribbons: List[Path] = []
    for polygon in path.to_polygons(closed_only=False):
        points = _dedupe_polyline(np.asarray(polygon, dtype=np.float64))
        if points.shape[0] < 2:
            continue
        if not np.allclose(points[0], points[-1]):
            points = np.vstack([points, points[0]])
        if points.shape[0] < 3:
            continue
        ribbons.append(
            polyline_ribbon_path(
                points,
                width=stroke_width,
                cap_start="butt",
                cap_end="butt",
                join_style="bevel",
            )
        )
    return ribbons


def render_text(
    ax: Any,
    specs: Sequence[DaguaText],
    display_scale: float,
    svg_hover_map: Optional[Dict[str, str]] = None,
) -> List[Any]:
    """Render text labels as filled paths in data coordinates.

    Parameters
    ----------
    ax : Any
        Matplotlib axes with equal aspect ratio.
    specs : sequence[DaguaText]
        Text render specifications.
    display_scale : float
        Points-to-data conversion factor.
    svg_hover_map : dict[str, str] | None, default=None
        Optional SVG hover-text accumulator.

    Returns
    -------
    list[Any]
        Artists created for all labels.
    """
    safe_scale = max(float(display_scale), 1e-9)
    artists: List[Any] = []

    for spec in specs:
        font_family = spec.font_family or RESOLVED_FONT
        font_size_pts = spec.font_size if spec.font_size > 0.0 else spec.min_font_size
        font_size_pts = max(font_size_pts, 1e-9)
        size_data = max(font_size_pts * safe_scale, 1e-9)
        min_size_data = max(spec.min_font_size * safe_scale, 1e-9)
        prepared_text = prepare_label_text(
            spec.text,
            font_size=size_data,
            text_wrap=spec.text_wrap,
            text_max_width=spec.text_max_width,
            text_transform=spec.text_transform,
            label_format="rich" if spec.rich else "plain",
        )
        if prepared_text.strip() == "" and spec.background is None:
            continue

        block = (
            layout_rich_text(
                prepared_text,
                size_data=size_data,
                ha=spec.ha,
                va=spec.va,
                font_family=font_family,
                font_weight=spec.font_weight,
                font_style=spec.font_style,
                font_color=spec.font_color,
                line_spacing=spec.line_spacing,
                secondary_scale=spec.secondary_scale,
                max_width=spec.max_width,
                min_size_data=min_size_data,
            )
            if spec.rich
            else layout_plain_text(
                prepared_text,
                size_data=size_data,
                ha=spec.ha,
                va=spec.va,
                font_family=font_family,
                font_weight=spec.font_weight,
                font_style=spec.font_style,
                font_color=spec.font_color,
                line_spacing=spec.line_spacing,
                secondary_scale=spec.secondary_scale,
                max_width=spec.max_width,
                min_size_data=min_size_data,
            )
        )

        if spec.background is not None:
            background_path = _matplotlib_text_background_path(
                ax,
                spec,
                prepared_text,
                font_family,
                font_size_pts,
            )
            if background_path is None:
                pad_x = spec.background_padding[0] * safe_scale
                pad_y = spec.background_padding[1] * safe_scale
                corner_radius = spec.background_corner_radius * safe_scale
                background_path = background_rect_path(
                    spec.x + block.x_offset + block.width / 2.0,
                    spec.y + block.y_offset - block.height / 2.0,
                    block.width,
                    block.height,
                    pad_x,
                    pad_y,
                    corner_radius,
                )
            background_patch = PathPatch(
                _transform_path(background_path, spec),
                facecolor=spec.background,
                edgecolor="none",
                linewidth=0.0,
                alpha=spec.background_alpha,
                zorder=_background_zorder(spec),
            )
            background_gid = _patch_gid(spec, "background")
            if background_gid is not None:
                background_patch.set_gid(background_gid)
            _register_hover(svg_hover_map, background_gid, spec.text)
            _add_patch(ax, background_patch, spec, artists)

        if spec.outline:
            for line_index, line in enumerate(block.lines):
                for segment_index, segment in enumerate(line.segments):
                    if _path_is_empty(segment.glyph_run.path):
                        continue
                    outline_gid = _patch_gid(spec, "outline", line_index, segment_index)
                    outline_width = _outline_linewidth(spec) * safe_scale
                    for outline_path in _glyph_stroke_ribbon_paths(
                        _segment_path(spec, block, line, segment),
                        outline_width,
                    ):
                        outline_patch = PathPatch(
                            outline_path,
                            facecolor=spec.outline_color,
                            edgecolor="none",
                            linewidth=0.0,
                            alpha=spec.alpha,
                            zorder=spec.zorder - 0.02,
                        )
                        if outline_gid is not None:
                            outline_patch.set_gid(outline_gid)
                        _register_hover(svg_hover_map, outline_gid, spec.text)
                        _add_patch(ax, outline_patch, spec, artists)

        for line_index, line in enumerate(block.lines):
            for segment_index, segment in enumerate(line.segments):
                if _path_is_empty(segment.glyph_run.path):
                    continue
                if _segment_needs_bold_emphasis(spec, segment):
                    emphasis_gid = _patch_gid(spec, "embolden", line_index, segment_index)
                    emphasis_width = _bold_emphasis_linewidth(spec) * safe_scale
                    for emphasis_path in _glyph_stroke_ribbon_paths(
                        _segment_path(spec, block, line, segment),
                        emphasis_width,
                    ):
                        emphasis_patch = PathPatch(
                            emphasis_path,
                            facecolor=segment.color,
                            edgecolor="none",
                            linewidth=0.0,
                            alpha=spec.alpha,
                            zorder=spec.zorder - 0.01,
                        )
                        if emphasis_gid is not None:
                            emphasis_patch.set_gid(emphasis_gid)
                        _register_hover(svg_hover_map, emphasis_gid, spec.text)
                        _add_patch(ax, emphasis_patch, spec, artists)
                fill_patch = PathPatch(
                    _segment_path(spec, block, line, segment),
                    facecolor=segment.color,
                    edgecolor="none",
                    linewidth=0.0,
                    alpha=spec.alpha,
                    zorder=spec.zorder,
                )
                gid = _segment_gid(spec, line_index, segment_index, len(block.lines))
                if gid is not None:
                    fill_patch.set_gid(gid)
                _register_hover(svg_hover_map, gid, spec.text)
                _add_patch(ax, fill_patch, spec, artists)

        for line_index, line in enumerate(block.lines):
            baseline = spec.y + block.y_offset + line.baseline_y
            for segment_index, segment in enumerate(line.segments):
                segment_start = spec.x + block.x_offset + segment.x_offset
                segment_end = segment_start + segment.glyph_run.advance_width
                metrics = segment.glyph_run.metrics
                thickness = metrics.line_height * 0.06
                if segment.underline or (not spec.rich and spec.underline):
                    underline_patch = PathPatch(
                        _transform_path(
                            underline_path(
                                segment_start,
                                segment_end,
                                baseline - metrics.descent * 0.3,
                                thickness,
                            ),
                            spec,
                        ),
                        facecolor=segment.color,
                        edgecolor="none",
                        linewidth=0.0,
                        alpha=spec.alpha,
                        zorder=spec.zorder + 0.02,
                    )
                    underline_gid = _patch_gid(spec, "underline", line_index, segment_index)
                    if underline_gid is not None:
                        underline_patch.set_gid(underline_gid)
                    _register_hover(svg_hover_map, underline_gid, spec.text)
                    _add_patch(ax, underline_patch, spec, artists)
                if segment.strikethrough or (not spec.rich and spec.strikethrough):
                    strike_patch = PathPatch(
                        _transform_path(
                            strikethrough_path(
                                segment_start,
                                segment_end,
                                baseline + metrics.x_height * 0.5,
                                thickness,
                            ),
                            spec,
                        ),
                        facecolor=segment.color,
                        edgecolor="none",
                        linewidth=0.0,
                        alpha=spec.alpha,
                        zorder=spec.zorder + 0.02,
                    )
                    strike_gid = _patch_gid(spec, "strikethrough", line_index, segment_index)
                    if strike_gid is not None:
                        strike_patch.set_gid(strike_gid)
                    _register_hover(svg_hover_map, strike_gid, spec.text)
                    _add_patch(ax, strike_patch, spec, artists)

    return artists
