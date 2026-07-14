"""Shared two-panel visual parity compositor."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts.visual_parity.types import AlignmentManifest, GeometryMode

WHITE = "#FFFFFF"
TEXT_COLOR = "#111111"
LINE_COLOR = "#D7D7D7"
HEADER_FILL = "#F7F7F7"
HEADER_HEIGHT = 28
DIVIDER_WIDTH = 4
CONTENT_MARGIN_PX = 12
MAX_OUTPUT_SIDE_PX = 2000


def _load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    """Load a local font for compositor labels.

    Parameters
    ----------
    size
        Requested font size.
    bold
        Whether to prefer a bold face.

    Returns
    -------
    PIL.ImageFont.ImageFont
        Loaded font object.
    """

    names = ["DejaVuSans-Bold.ttf", "DejaVuSans.ttf"] if bold else ["DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _ink_bbox(image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """Return a near-white-trimmed ink bounding box.

    Parameters
    ----------
    image
        Image to inspect.

    Returns
    -------
    tuple[int, int, int, int] or None
        PIL crop box, or ``None`` when no ink is detected.
    """

    data = np.asarray(image.convert("RGBA"))
    rgb = data[:, :, :3].astype(np.int16)
    alpha = data[:, :, 3]
    mask = (alpha > 0) & np.any(rgb < 248, axis=2)
    if not bool(mask.any()):
        return None
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _paste_on_canvas(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """Paste an image onto a white canvas without scaling.

    Parameters
    ----------
    image
        Source image.
    size
        Canvas ``(width, height)``.

    Returns
    -------
    PIL.Image.Image
        RGB canvas containing ``image`` at the top-left origin.
    """

    canvas = Image.new("RGB", size, WHITE)
    canvas.paste(image.convert("RGB"), (0, 0))
    return canvas


def union_content_crop(
    reference: Image.Image,
    dagua: Image.Image,
    *,
    margin_px: int = CONTENT_MARGIN_PX,
) -> Tuple[Image.Image, Image.Image, Optional[Tuple[int, int, int, int]]]:
    """Crop both panels to one union ink box with no per-side normalization.

    Parameters
    ----------
    reference
        Reference panel.
    dagua
        Dagua panel.
    margin_px
        Margin added around the union ink bounds.

    Returns
    -------
    tuple[PIL.Image.Image, PIL.Image.Image, tuple[int, int, int, int] | None]
        Cropped panels and the crop box in shared pixel coordinates.
    """

    shared_size = (max(reference.width, dagua.width), max(reference.height, dagua.height))
    ref_canvas = _paste_on_canvas(reference, shared_size)
    dagua_canvas = _paste_on_canvas(dagua, shared_size)
    boxes = [box for box in (_ink_bbox(ref_canvas), _ink_bbox(dagua_canvas)) if box is not None]
    if not boxes:
        return ref_canvas, dagua_canvas, None
    left = max(min(box[0] for box in boxes) - margin_px, 0)
    top = max(min(box[1] for box in boxes) - margin_px, 0)
    right = min(max(box[2] for box in boxes) + margin_px, shared_size[0])
    bottom = min(max(box[3] for box in boxes) + margin_px, shared_size[1])
    crop_box = (left, top, right, bottom)
    return ref_canvas.crop(crop_box), dagua_canvas.crop(crop_box), crop_box


def _draw_text_fit(
    draw: ImageDraw.ImageDraw,
    bounds: Tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
) -> None:
    """Draw text clipped to a single bounded header line.

    Parameters
    ----------
    draw
        Drawing context.
    bounds
        Text bounds.
    text
        Text to draw.
    font
        Font object.

    Returns
    -------
    None
        The image is mutated in place.
    """

    left, top, right, bottom = bounds
    max_width = max(right - left - 8, 1)
    if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
        candidate = text
    else:
        low = 0
        high = len(text)
        candidate = "..."
        while low <= high:
            mid = (low + high) // 2
            trial = f"{text[:mid].rstrip()}..."
            if draw.textbbox((0, 0), trial, font=font)[2] <= max_width:
                candidate = trial
                low = mid + 1
            else:
                high = mid - 1
    bbox = draw.textbbox((0, 0), candidate, font=font)
    y = top + ((bottom - top) - (bbox[3] - bbox[1])) / 2
    draw.text((left + 4, y), candidate, font=font, fill=TEXT_COLOR)


def _compose_tile(
    reference: Image.Image,
    dagua: Image.Image,
    *,
    header: str,
    y0: int,
    y1: int,
) -> Image.Image:
    """Compose one vertical tile of the two-panel comparison.

    Parameters
    ----------
    reference
        Cropped reference panel.
    dagua
        Cropped Dagua panel.
    header
        Header strip text.
    y0
        Inclusive tile top in panel coordinates.
    y1
        Exclusive tile bottom in panel coordinates.

    Returns
    -------
    PIL.Image.Image
        Two-column tile image.
    """

    ref_tile = reference.crop((0, y0, reference.width, y1))
    dagua_tile = dagua.crop((0, y0, dagua.width, y1))
    panel_w = max(ref_tile.width, dagua_tile.width)
    panel_h = max(ref_tile.height, dagua_tile.height)
    width = panel_w * 2 + DIVIDER_WIDTH
    height = HEADER_HEIGHT + panel_h
    canvas = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(canvas)
    header_font = _load_font(12)
    draw.rectangle((0, 0, width, HEADER_HEIGHT), fill=HEADER_FILL)
    _draw_text_fit(draw, (0, 0, width, HEADER_HEIGHT), header, header_font)
    canvas.paste(_paste_on_canvas(ref_tile, (panel_w, panel_h)), (0, HEADER_HEIGHT))
    divider_x = panel_w
    draw.rectangle((divider_x, 0, divider_x + DIVIDER_WIDTH - 1, height), fill=LINE_COLOR)
    canvas.paste(
        _paste_on_canvas(dagua_tile, (panel_w, panel_h)),
        (panel_w + DIVIDER_WIDTH, HEADER_HEIGHT),
    )
    return canvas


def _scale_under_cap(image: Image.Image) -> Tuple[Image.Image, float]:
    """Scale an image uniformly when its longest side exceeds the cap.

    Parameters
    ----------
    image
        Image to cap.

    Returns
    -------
    tuple[PIL.Image.Image, float]
        Capped image and scale factor.
    """

    longest = max(image.size)
    if longest <= MAX_OUTPUT_SIDE_PX:
        return image, 1.0
    scale = MAX_OUTPUT_SIDE_PX / float(longest)
    size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
    return image.resize(size, Image.Resampling.LANCZOS), scale


def compose_pair(
    reference_path: Union[str, Path],
    dagua_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    case_id: str,
    round_id: str,
    reference_label: str,
    geometry_mode: GeometryMode,
    l1: Optional[float] = None,
    feature_ok: Optional[Tuple[int, int]] = None,
    canvas_pt: Tuple[float, float] = (0.0, 0.0),
    dpi: float = 200.0,
    manifest_path: Optional[Union[str, Path]] = None,
) -> AlignmentManifest:
    """Compose a two-panel comparison image and emit its alignment manifest.

    Parameters
    ----------
    reference_path
        Reference image path.
    dagua_path
        Dagua image path.
    output_path
        Composite image destination.
    case_id
        Case id included in the header.
    round_id
        Round id included in the header.
    reference_label
        Reference provenance label.
    geometry_mode
        Geometry source label.
    l1
        Optional L1 value included in the header.
    feature_ok
        Optional ``(passed, total)`` feature count included in the header.
    canvas_pt
        Declared canvas size in points.
    dpi
        Raster DPI used for both panels.
    manifest_path
        Optional JSON manifest destination.

    Returns
    -------
    scripts.visual_parity.types.AlignmentManifest
        Alignment manifest for the composed image.
    """

    with Image.open(reference_path) as ref_image, Image.open(dagua_path) as dagua_image:
        cropped_ref, cropped_dagua, crop_box = union_content_crop(ref_image, dagua_image)
    l1_text = "na" if l1 is None else f"{l1:.4f}"
    feat_text = "na" if feature_ok is None else f"{feature_ok[0]}/{feature_ok[1]}"
    header = (
        f"{case_id} | round {round_id} | ref={reference_label} | "
        f"mode={geometry_mode.value} | L1={l1_text} feat_ok={feat_text}"
    )
    panel_h = max(cropped_ref.height, cropped_dagua.height)
    max_tile_content_h = max(MAX_OUTPUT_SIDE_PX - HEADER_HEIGHT, 1)
    tiles: List[Image.Image] = []
    for y0 in range(0, panel_h, max_tile_content_h):
        y1 = min(y0 + max_tile_content_h, panel_h)
        tile = _compose_tile(cropped_ref, cropped_dagua, header=header, y0=y0, y1=y1)
        tile, _scale = _scale_under_cap(tile)
        if max(tile.size) > MAX_OUTPUT_SIDE_PX:
            raise AssertionError(f"composite tile exceeds {MAX_OUTPUT_SIDE_PX}px: {tile.size}")
        tiles.append(tile)
    if len(tiles) == 1:
        composite = tiles[0]
        scale_factor = 1.0
    else:
        tile_w = max(tile.width for tile in tiles)
        composite_h = sum(tile.height for tile in tiles)
        composite = Image.new("RGB", (tile_w, composite_h), WHITE)
        offset = 0
        for tile in tiles:
            composite.paste(tile, (0, offset))
            offset += tile.height
        composite, scale_factor = _scale_under_cap(composite)
    if max(composite.size) > MAX_OUTPUT_SIDE_PX:
        raise AssertionError(f"composite exceeds {MAX_OUTPUT_SIDE_PX}px: {composite.size}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    composite.save(output, format="PNG")
    manifest = AlignmentManifest(
        canvas_pt=canvas_pt,
        dpi=float(dpi),
        pixel_size=(int(max(cropped_ref.width, cropped_dagua.width)), int(panel_h)),
        crop_box_px=crop_box,
        crop_reason="union_content_12px" if crop_box is not None else None,
        metric_uses_crop=False,
        pad_or_crop_applied=crop_box is not None or cropped_ref.size != cropped_dagua.size,
    )
    manifest_payload: Dict[str, Any] = asdict(manifest)
    manifest_payload["composite_path"] = str(output)
    manifest_payload["composite_scale_factor"] = scale_factor
    manifest_payload["max_output_side_px"] = MAX_OUTPUT_SIDE_PX
    if manifest_path is not None:
        path = Path(manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    return manifest


def write_roi_pair(
    reference: Image.Image,
    dagua: Image.Image,
    roi_box_px: Tuple[int, int, int, int],
    output_path: Union[str, Path],
    *,
    scale: float = 1.0,
) -> None:
    """Write a two-column ROI crop pair under the output cap.

    Parameters
    ----------
    reference
        Reference panel.
    dagua
        Dagua panel.
    roi_box_px
        ROI crop box in reference pixel coordinates.
    output_path
        Destination PNG path.
    scale
        Optional pre-crop scale factor, auto-reduced under the cap.

    Returns
    -------
    None
        ROI pair image is written.
    """

    crop_ref = reference.crop(roi_box_px).convert("RGB")
    crop_dagua = dagua.crop(roi_box_px).convert("RGB")
    if scale != 1.0:
        crop_ref = crop_ref.resize(
            (max(1, int(crop_ref.width * scale)), max(1, int(crop_ref.height * scale))),
            Image.Resampling.LANCZOS,
        )
        crop_dagua = crop_dagua.resize(crop_ref.size, Image.Resampling.LANCZOS)
    pair = _compose_tile(crop_ref, crop_dagua, header="ROI", y0=0, y1=crop_ref.height)
    pair, _scale = _scale_under_cap(pair)
    if max(pair.size) > MAX_OUTPUT_SIDE_PX:
        raise AssertionError(f"ROI pair exceeds {MAX_OUTPUT_SIDE_PX}px: {pair.size}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pair.save(output, format="PNG")
