#!/usr/bin/env python
# ruff: noqa: E402
"""Pixel-level parity checks for the ``graphviz_strict`` cosmetic theme.

This script complements ``scripts/parity_metrics.py``. The declarative metric
checks extracted attributes; this raster pass catches font hinting,
anti-aliasing, sub-pixel rounding, and other differences that only appear
after both renderers produce pixels.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import dagua  # noqa: E402
import scripts.graphviz_theme_comparison as gthc  # noqa: E402
import scripts.parity_metrics as pmetrics  # noqa: E402
from dagua.graphviz_utils import layout_with_graphviz  # noqa: E402

DEFAULT_OUT_DIR = Path("eval_output/parity_pixel_diff")
STRICT_THEME_NAME = "graphviz_strict"
MAX_HIRES_SIDE_PX = 2000
POINTS_PER_INCH = 72.0
SVG_NS = "http://www.w3.org/2000/svg"
WHITE = "#FFFFFF"
TEXT_COLOR = "#111111"
LINE_COLOR = "#D7D7D7"
HEADER_HEIGHT = 34


@dataclass(frozen=True)
class RenderPair:
    """Rendered image pair and matching reference metadata.

    Parameters
    ----------
    dot_png : Path
        Native Graphviz PNG path.
    dagua_png : Path
        Dagua strict PNG path with dimensions normalized to ``dot_png``.
    dot_svg : str
        SVG payload emitted from the same DOT source used for the PNG.
    dimensions : tuple[int, int]
        Image dimensions as ``(width, height)`` in pixels.
    effective_dpi : int
        DPI used for both renderers after any hi-res cap adjustment.
    """

    dot_png: Path
    dagua_png: Path
    dot_svg: str
    dimensions: Tuple[int, int]
    effective_dpi: int


@dataclass(frozen=True)
class RegionMasks:
    """Boolean masks used to split pixel error by graph region.

    Parameters
    ----------
    text : numpy.ndarray
        Approximate text/node-label mask, shape ``[H, W]``.
    node : numpy.ndarray
        Node ellipse mask, shape ``[H, W]``.
    edge : numpy.ndarray
        Edge path/arrow mask, shape ``[H, W]``.
    background : numpy.ndarray
        Background mask outside node and edge regions, shape ``[H, W]``.
    """

    text: np.ndarray
    node: np.ndarray
    edge: np.ndarray
    background: np.ndarray


def _strip_ns(tag: str) -> str:
    """Remove an XML namespace from an element tag.

    Parameters
    ----------
    tag : str
        Raw XML element tag.

    Returns
    -------
    str
        Local tag name.
    """

    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def _parse_float(value: Optional[str], default: float = 0.0) -> float:
    """Parse a float from an SVG attribute.

    Parameters
    ----------
    value : str, optional
        Raw attribute value.
    default : float, default=0.0
        Fallback value.

    Returns
    -------
    float
        Parsed number or ``default``.
    """

    if value is None:
        return default
    cleaned = value.strip()
    if cleaned.endswith("pt"):
        cleaned = cleaned[:-2].strip()
    try:
        return float(cleaned)
    except ValueError:
        return default


def _parse_svg_viewbox(root: ET.Element) -> Tuple[float, float, float, float]:
    """Return an SVG viewBox in point units.

    Parameters
    ----------
    root : xml.etree.ElementTree.Element
        SVG root element.

    Returns
    -------
    tuple[float, float, float, float]
        ``(min_x, min_y, width, height)``.
    """

    raw = root.attrib.get("viewBox", "")
    parts = [part for part in raw.replace(",", " ").split() if part]
    if len(parts) == 4:
        try:
            return tuple(float(part) for part in parts)  # type: ignore[return-value]
        except ValueError:
            pass
    return (
        0.0,
        0.0,
        _parse_float(root.attrib.get("width")),
        _parse_float(root.attrib.get("height")),
    )


def _dot_source(graph: Any) -> str:
    """Build DOT source for the native Graphviz reference.

    Parameters
    ----------
    graph : Any
        Source ``DaguaGraph`` from the comparison harness.

    Returns
    -------
    str
        DOT document.
    """

    themed = pmetrics._apply_strict_theme(graph)
    return gthc.graph_to_dot(themed)


def _run_dot(dot_source: str, png_path: Path, dpi: int) -> str:
    """Render DOT to PNG and SVG through native ``dot``.

    Parameters
    ----------
    dot_source : str
        DOT document.
    png_path : pathlib.Path
        PNG destination.
    dpi : int
        Raster DPI.

    Returns
    -------
    str
        SVG payload from the same DOT source.

    Raises
    ------
    RuntimeError
        If Graphviz returns a non-zero exit status.
    """

    png_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dagua-parity-dot-") as tmp:
        dot_path = Path(tmp) / "graph.dot"
        svg_path = Path(tmp) / "graph.svg"
        dot_path.write_text(dot_source, encoding="utf-8")
        png_result = subprocess.run(
            ["dot", "-Tpng", f"-Gdpi={dpi}", str(dot_path), "-o", str(png_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if png_result.returncode != 0:
            raise RuntimeError(f"dot -Tpng failed: {png_result.stderr.strip()}")
        svg_result = subprocess.run(
            ["dot", "-Tsvg", str(dot_path), "-o", str(svg_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if svg_result.returncode != 0:
            raise RuntimeError(f"dot -Tsvg failed: {svg_result.stderr.strip()}")
        return svg_path.read_text(encoding="utf-8")


def _copy_graph_with_strict_theme(graph: Any) -> Any:
    """Clone a graph and bind ``graphviz_strict`` without altering globals.

    Parameters
    ----------
    graph : Any
        Source ``DaguaGraph``.

    Returns
    -------
    Any
        Themed graph clone.
    """

    return pmetrics._apply_strict_theme(graph)


def _graphviz_positions(graph: Any) -> Any:
    """Return Graphviz positions in Dagua's render coordinate convention.

    Parameters
    ----------
    graph : Any
        Strict-themed graph clone.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """

    positions = layout_with_graphviz(graph, engine="dot")
    positions[:, 1] = -positions[:, 1]
    graph.direction = "BT"
    return positions


def _swap_arrowheads_for_graphviz_yup(graph: Any) -> None:
    """Move head arrows to tail arrows for BT rendering on Graphviz coordinates.

    Parameters
    ----------
    graph : Any
        Strict-themed graph clone to mutate.

    Returns
    -------
    None
        Edge styles are updated in place.
    """

    from dataclasses import replace as dc_replace

    for edge_index in range(int(graph.edge_index.shape[1])):
        style = graph.get_style_for_edge(edge_index)
        if style.arrow != "none" and style.tail_arrow == "none":
            graph.edge_styles[edge_index] = dc_replace(
                style,
                arrow="none",
                tail_arrow=style.arrow,
            )


def _center_crop_or_pad(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """Center-crop or pad an image to exact target dimensions.

    Parameters
    ----------
    image : PIL.Image.Image
        Source image.
    target_size : tuple[int, int]
        Desired ``(width, height)``.

    Returns
    -------
    PIL.Image.Image
        RGB image with exact target dimensions.
    """

    rgb = image.convert("RGB")
    target_w, target_h = target_size
    if rgb.width > target_w or rgb.height > target_h:
        left = max((rgb.width - target_w) // 2, 0)
        top = max((rgb.height - target_h) // 2, 0)
        rgb = rgb.crop(
            (
                left,
                top,
                min(left + target_w, rgb.width),
                min(top + target_h, rgb.height),
            )
        )
    canvas = Image.new("RGB", target_size, WHITE)
    offset = ((target_w - rgb.width) // 2, (target_h - rgb.height) // 2)
    canvas.paste(rgb, offset)
    return canvas


def _render_dagua_strict(
    graph: Any,
    output_path: Path,
    target_size: Tuple[int, int],
    dpi: int,
) -> None:
    """Render Dagua strict output and normalize it to a reference canvas.

    Parameters
    ----------
    graph : Any
        Source ``DaguaGraph``.
    output_path : pathlib.Path
        Destination PNG path.
    target_size : tuple[int, int]
        Desired output size in pixels.
    dpi : int
        Raster DPI.

    Returns
    -------
    None
        The normalized PNG is written to ``output_path``.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    themed = _copy_graph_with_strict_theme(graph)
    themed.compute_node_sizes()
    positions = _graphviz_positions(themed)
    _swap_arrowheads_for_graphviz_yup(themed)
    fig_w = target_size[0] / float(dpi)
    fig_h = target_size[1] / float(dpi)
    raw_path = output_path.with_name(f"{output_path.stem}.raw.png")
    fig, _ax = dagua.render(
        themed,
        positions,
        output=str(raw_path),
        figsize=(fig_w, fig_h),
        dpi=dpi,
    )
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass
    with Image.open(raw_path) as raw:
        normalized = _center_crop_or_pad(raw, target_size)
        normalized.save(output_path, format="PNG", dpi=(dpi, dpi))
    raw_path.unlink(missing_ok=True)


def _scaled_hires_dpi(dot_source: str, requested_dpi: int) -> int:
    """Choose a hi-res DPI that keeps native output under the image cap.

    Parameters
    ----------
    dot_source : str
        DOT document.
    requested_dpi : int
        Requested hi-res DPI.

    Returns
    -------
    int
        Effective DPI, never larger than ``requested_dpi``.
    """

    svg_text = subprocess.run(
        ["dot", "-Tsvg"],
        input=dot_source,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout
    try:
        root = ET.fromstring(svg_text)
        _min_x, _min_y, width_pt, height_pt = _parse_svg_viewbox(root)
    except ET.ParseError:
        return requested_dpi
    longest = max(width_pt, height_pt)
    if longest <= 0:
        return requested_dpi
    capped = math.floor(MAX_HIRES_SIDE_PX * POINTS_PER_INCH / longest)
    return max(72, min(requested_dpi, capped))


def render_pair(
    case: gthc.GraphCase,
    out_dir: Path,
    dpi: int,
    *,
    cap_longest_side: bool = False,
) -> RenderPair:
    """Render native dot and Dagua strict rasters for one case.

    Parameters
    ----------
    case : scripts.graphviz_theme_comparison.GraphCase
        Panel to render.
    out_dir : pathlib.Path
        Directory for intermediate image files.
    dpi : int
        Requested DPI.
    cap_longest_side : bool, default=False
        Whether to reduce DPI so the longest side stays below 2000 pixels.

    Returns
    -------
    RenderPair
        Paths and metadata for the rendered pair.
    """

    dot_source = _dot_source(case.graph)
    effective_dpi = _scaled_hires_dpi(dot_source, dpi) if cap_longest_side else dpi
    dot_path = out_dir / "dot" / f"{case.slug}.png"
    dagua_path = out_dir / "dagua" / f"{case.slug}.png"
    svg_text = _run_dot(dot_source, dot_path, effective_dpi)
    with Image.open(dot_path) as dot_image:
        target_size = dot_image.size
    _render_dagua_strict(case.graph, dagua_path, target_size, effective_dpi)
    return RenderPair(
        dot_png=dot_path,
        dagua_png=dagua_path,
        dot_svg=svg_text,
        dimensions=target_size,
        effective_dpi=effective_dpi,
    )


def _svg_to_pixel(
    x: float,
    y: float,
    viewbox: Tuple[float, float, float, float],
    dimensions: Tuple[int, int],
) -> Tuple[float, float]:
    """Map SVG point coordinates to image pixel coordinates.

    Parameters
    ----------
    x : float
        SVG x coordinate.
    y : float
        SVG y coordinate.
    viewbox : tuple[float, float, float, float]
        SVG viewBox.
    dimensions : tuple[int, int]
        Image dimensions.

    Returns
    -------
    tuple[float, float]
        Pixel ``(x, y)``.
    """

    min_x, min_y, width_pt, height_pt = viewbox
    width_px, height_px = dimensions
    px = (x - min_x) / max(width_pt, 1e-9) * width_px
    py = (y - min_y) / max(height_pt, 1e-9) * height_px
    return px, py


def _draw_svg_ellipse(
    draw: ImageDraw.ImageDraw,
    ellipse: ET.Element,
    viewbox: Tuple[float, float, float, float],
    dimensions: Tuple[int, int],
    *,
    expand_px: int,
) -> None:
    """Draw an SVG ellipse into a pixel mask.

    Parameters
    ----------
    draw : PIL.ImageDraw.ImageDraw
        Mask drawing context.
    ellipse : xml.etree.ElementTree.Element
        SVG ellipse element.
    viewbox : tuple[float, float, float, float]
        SVG viewBox.
    dimensions : tuple[int, int]
        Pixel dimensions.
    expand_px : int
        Extra pixels added around the ellipse.

    Returns
    -------
    None
        The mask is mutated in place.
    """

    cx = _parse_float(ellipse.attrib.get("cx"))
    cy = _parse_float(ellipse.attrib.get("cy"))
    rx = _parse_float(ellipse.attrib.get("rx"))
    ry = _parse_float(ellipse.attrib.get("ry"))
    x0, y0 = _svg_to_pixel(cx - rx, cy - ry, viewbox, dimensions)
    x1, y1 = _svg_to_pixel(cx + rx, cy + ry, viewbox, dimensions)
    draw.ellipse(
        (
            min(x0, x1) - expand_px,
            min(y0, y1) - expand_px,
            max(x0, x1) + expand_px,
            max(y0, y1) + expand_px,
        ),
        fill=255,
    )


def _draw_svg_text_box(
    draw: ImageDraw.ImageDraw,
    text: ET.Element,
    viewbox: Tuple[float, float, float, float],
    dimensions: Tuple[int, int],
    *,
    expand_px: int,
) -> None:
    """Draw an approximate text bounding box into a pixel mask.

    Parameters
    ----------
    draw : PIL.ImageDraw.ImageDraw
        Mask drawing context.
    text : xml.etree.ElementTree.Element
        SVG text element.
    viewbox : tuple[float, float, float, float]
        SVG viewBox.
    dimensions : tuple[int, int]
        Pixel dimensions.
    expand_px : int
        Padding added around the estimated text box.

    Returns
    -------
    None
        The mask is mutated in place.
    """

    label = text.text or ""
    font_size = _parse_float(text.attrib.get("font-size"), default=14.0)
    x = _parse_float(text.attrib.get("x"))
    y = _parse_float(text.attrib.get("y"))
    px, py = _svg_to_pixel(x, y, viewbox, dimensions)
    scale_x = dimensions[0] / max(viewbox[2], 1e-9)
    scale_y = dimensions[1] / max(viewbox[3], 1e-9)
    width = max(len(label), 1) * font_size * 0.58 * scale_x
    height = font_size * 1.25 * scale_y
    draw.rectangle(
        (
            px - width / 2 - expand_px,
            py - height + font_size * 0.25 * scale_y - expand_px,
            px + width / 2 + expand_px,
            py + font_size * 0.35 * scale_y + expand_px,
        ),
        fill=255,
    )


def _parse_path_numbers(path_data: str) -> List[float]:
    """Extract numeric coordinates from simple SVG path data.

    Parameters
    ----------
    path_data : str
        SVG path ``d`` attribute.

    Returns
    -------
    list[float]
        Coordinate numbers in source order.
    """

    normalized = path_data
    for char in "MmCcLlQqSsTtHhVvAaZz,":
        normalized = normalized.replace(char, " ")
    numbers: List[float] = []
    for token in normalized.split():
        try:
            numbers.append(float(token))
        except ValueError:
            continue
    return numbers


def _draw_svg_path_bbox(
    draw: ImageDraw.ImageDraw,
    path: ET.Element,
    viewbox: Tuple[float, float, float, float],
    dimensions: Tuple[int, int],
    *,
    expand_px: int,
) -> None:
    """Draw a conservative path bounding box into an edge mask.

    Parameters
    ----------
    draw : PIL.ImageDraw.ImageDraw
        Mask drawing context.
    path : xml.etree.ElementTree.Element
        SVG path element.
    viewbox : tuple[float, float, float, float]
        SVG viewBox.
    dimensions : tuple[int, int]
        Pixel dimensions.
    expand_px : int
        Padding around the path bbox.

    Returns
    -------
    None
        The mask is mutated in place.
    """

    numbers = _parse_path_numbers(path.attrib.get("d", ""))
    points = [(numbers[i], numbers[i + 1]) for i in range(0, len(numbers) - 1, 2)]
    if not points:
        return
    px_points = [_svg_to_pixel(x, y, viewbox, dimensions) for x, y in points]
    xs = [p[0] for p in px_points]
    ys = [p[1] for p in px_points]
    draw.rectangle(
        (
            min(xs) - expand_px,
            min(ys) - expand_px,
            max(xs) + expand_px,
            max(ys) + expand_px,
        ),
        fill=255,
    )


def _draw_svg_polygon(
    draw: ImageDraw.ImageDraw,
    polygon: ET.Element,
    viewbox: Tuple[float, float, float, float],
    dimensions: Tuple[int, int],
    *,
    expand_px: int,
) -> None:
    """Draw an SVG polygon into a pixel mask.

    Parameters
    ----------
    draw : PIL.ImageDraw.ImageDraw
        Mask drawing context.
    polygon : xml.etree.ElementTree.Element
        SVG polygon element.
    viewbox : tuple[float, float, float, float]
        SVG viewBox.
    dimensions : tuple[int, int]
        Pixel dimensions.
    expand_px : int
        Padding around the polygon bbox.

    Returns
    -------
    None
        The mask is mutated in place.
    """

    vertices = pmetrics._parse_polygon_points(polygon.attrib.get("points", ""))
    if not vertices:
        return
    px_points = [_svg_to_pixel(x, y, viewbox, dimensions) for x, y in vertices]
    xs = [p[0] for p in px_points]
    ys = [p[1] for p in px_points]
    draw.rectangle(
        (
            min(xs) - expand_px,
            min(ys) - expand_px,
            max(xs) + expand_px,
            max(ys) + expand_px,
        ),
        fill=255,
    )


def build_region_masks(dot_svg: str, dimensions: Tuple[int, int]) -> RegionMasks:
    """Create approximate region masks from Graphviz SVG geometry.

    Parameters
    ----------
    dot_svg : str
        Native Graphviz SVG payload.
    dimensions : tuple[int, int]
        Matching PNG dimensions.

    Returns
    -------
    RegionMasks
        Region masks for scalar error breakdowns.
    """

    root = ET.fromstring(dot_svg)
    viewbox = _parse_svg_viewbox(root)
    node_img = Image.new("L", dimensions, 0)
    text_img = Image.new("L", dimensions, 0)
    edge_img = Image.new("L", dimensions, 0)
    node_draw = ImageDraw.Draw(node_img)
    text_draw = ImageDraw.Draw(text_img)
    edge_draw = ImageDraw.Draw(edge_img)

    for group in root.iter():
        if _strip_ns(group.tag) != "g":
            continue
        css_class = group.attrib.get("class", "")
        if css_class == "node":
            for child in group:
                if _strip_ns(child.tag) == "ellipse":
                    _draw_svg_ellipse(node_draw, child, viewbox, dimensions, expand_px=2)
                elif _strip_ns(child.tag) == "text":
                    _draw_svg_text_box(text_draw, child, viewbox, dimensions, expand_px=2)
        elif css_class == "edge":
            for child in group:
                tag = _strip_ns(child.tag)
                if tag == "path":
                    _draw_svg_path_bbox(edge_draw, child, viewbox, dimensions, expand_px=5)
                elif tag in {"polygon", "polyline"}:
                    _draw_svg_polygon(edge_draw, child, viewbox, dimensions, expand_px=5)

    node = np.asarray(node_img) > 0
    text = np.asarray(text_img) > 0
    edge = np.asarray(edge_img) > 0
    background = ~(node | edge)
    return RegionMasks(text=text, node=node, edge=edge, background=background)


def _load_rgb(path: Path) -> np.ndarray:
    """Load an image as an RGB uint8 array.

    Parameters
    ----------
    path : pathlib.Path
        Source image path.

    Returns
    -------
    numpy.ndarray
        Array with shape ``[H, W, 3]``.
    """

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _mean_l1(error: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Compute mean per-pixel RGB L1 error.

    Parameters
    ----------
    error : numpy.ndarray
        Absolute RGB difference with shape ``[H, W, 3]``.
    mask : numpy.ndarray, optional
        Boolean region mask with shape ``[H, W]``.

    Returns
    -------
    float
        Mean absolute channel error per pixel.
    """

    if mask is not None:
        if not bool(mask.any()):
            return 0.0
        values = error[mask]
    else:
        values = error.reshape(-1, 3)
    return float(values.mean()) if values.size else 0.0


def _ssim_fallback(left: np.ndarray, right: np.ndarray) -> float:
    """Compute a bounded SSIM-like fallback from mean squared error.

    Parameters
    ----------
    left : numpy.ndarray
        Left RGB image, shape ``[H, W, 3]``.
    right : numpy.ndarray
        Right RGB image, shape ``[H, W, 3]``.

    Returns
    -------
    float
        Similarity score in ``[0, 1]``.
    """

    mse = float(np.mean((left.astype(np.float32) - right.astype(np.float32)) ** 2))
    return max(0.0, min(1.0, 1.0 - mse / (255.0**2)))


def _compute_ssim(left: np.ndarray, right: np.ndarray) -> float:
    """Compute global structural similarity.

    Parameters
    ----------
    left : numpy.ndarray
        Left RGB image, shape ``[H, W, 3]``.
    right : numpy.ndarray
        Right RGB image, shape ``[H, W, 3]``.

    Returns
    -------
    float
        SSIM score. Uses scikit-image when installed.
    """

    try:
        from skimage.metrics import structural_similarity

        return float(structural_similarity(left, right, channel_axis=2, data_range=255))
    except Exception:
        return _ssim_fallback(left, right)


def write_heatmap(error: np.ndarray, output_path: Path) -> None:
    """Write a transparent red heatmap for a pixel error image.

    Parameters
    ----------
    error : numpy.ndarray
        Absolute RGB difference with shape ``[H, W, 3]``.
    output_path : pathlib.Path
        PNG destination.

    Returns
    -------
    None
        The heatmap is written to disk.
    """

    intensity = error.mean(axis=2).astype(np.float32)
    alpha = np.clip((intensity / 64.0) * 255.0, 0, 255).astype(np.uint8)
    rgba = np.zeros((error.shape[0], error.shape[1], 4), dtype=np.uint8)
    rgba[:, :, 0] = 255
    rgba[:, :, 3] = alpha
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(output_path, format="PNG")


def score_pair(case: gthc.GraphCase, pair: RenderPair, out_dir: Path) -> Dict[str, Any]:
    """Compute pixel metrics and write heatmap/composite outputs.

    Parameters
    ----------
    case : scripts.graphviz_theme_comparison.GraphCase
        Scored panel.
    pair : RenderPair
        Rendered image pair.
    out_dir : pathlib.Path
        Output directory.

    Returns
    -------
    dict[str, Any]
        JSON-serialisable scalar metrics for the panel.
    """

    dot = _load_rgb(pair.dot_png)
    dagua_img = _load_rgb(pair.dagua_png)
    if dot.shape != dagua_img.shape:
        raise RuntimeError(f"{case.slug}: image shapes differ: {dot.shape} vs {dagua_img.shape}")
    error = np.abs(dot.astype(np.int16) - dagua_img.astype(np.int16)).astype(np.uint8)
    masks = build_region_masks(pair.dot_svg, pair.dimensions)
    heatmap_path = out_dir / "heatmaps" / f"{case.slug}.png"
    write_heatmap(error, heatmap_path)
    composite_path = out_dir / f"{case.slug}.png"
    compose_comparison(
        title=case.title,
        dot_path=pair.dot_png,
        dagua_path=pair.dagua_png,
        heatmap_path=heatmap_path,
        output_path=composite_path,
    )
    metrics: Dict[str, Any] = {
        "slug": case.slug,
        "title": case.title,
        "dpi": pair.effective_dpi,
        "width_px": pair.dimensions[0],
        "height_px": pair.dimensions[1],
        "l1_rgb_per_pixel": round(_mean_l1(error), 4),
        "ssim": round(_compute_ssim(dot, dagua_img), 6),
        "regions": {
            "text_l1_rgb_per_pixel": round(_mean_l1(error, masks.text), 4),
            "node_l1_rgb_per_pixel": round(_mean_l1(error, masks.node), 4),
            "background_l1_rgb_per_pixel": round(_mean_l1(error, masks.background), 4),
            "edge_arrow_l1_rgb_per_pixel": round(_mean_l1(error, masks.edge), 4),
        },
        "paths": {
            "dot": str(pair.dot_png),
            "dagua": str(pair.dagua_png),
            "heatmap": str(heatmap_path),
            "composite": str(composite_path),
        },
    }
    (out_dir / f"{case.slug}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    """Load a local font for comparison headers.

    Parameters
    ----------
    size : int
        Font size.
    bold : bool, default=False
        Whether to prefer a bold font.

    Returns
    -------
    PIL.ImageFont.ImageFont
        Loaded font.
    """

    names = ["DejaVuSans-Bold.ttf", "DejaVuSans.ttf"] if bold else ["DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    bounds: Tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
) -> None:
    """Draw centered header text.

    Parameters
    ----------
    draw : PIL.ImageDraw.ImageDraw
        Draw context.
    bounds : tuple[int, int, int, int]
        Text bounds.
    text : str
        Text value.
    font : PIL.ImageFont.ImageFont
        Font object.

    Returns
    -------
    None
        The image is mutated in place.
    """

    text_box = draw.textbbox((0, 0), text, font=font)
    width = text_box[2] - text_box[0]
    height = text_box[3] - text_box[1]
    left, top, right, bottom = bounds
    x = left + ((right - left) - width) / 2
    y = top + ((bottom - top) - height) / 2
    draw.text((x, y), text, font=font, fill=TEXT_COLOR)


def compose_comparison(
    title: str,
    dot_path: Path,
    dagua_path: Path,
    heatmap_path: Path,
    output_path: Path,
) -> None:
    """Compose native, Dagua, and heatmap panels side by side.

    Parameters
    ----------
    title : str
        Panel title.
    dot_path : pathlib.Path
        Native dot PNG path.
    dagua_path : pathlib.Path
        Dagua strict PNG path.
    heatmap_path : pathlib.Path
        Heatmap PNG path.
    output_path : pathlib.Path
        Composite PNG destination.

    Returns
    -------
    None
        The composite image is written to disk.
    """

    with (
        Image.open(dot_path) as dot_img,
        Image.open(dagua_path) as dagua_img,
        Image.open(heatmap_path) as heatmap_img,
    ):
        panels = [
            ("native dot", dot_img.convert("RGB")),
            ("dagua strict", dagua_img.convert("RGB")),
            (
                "diff heatmap",
                Image.alpha_composite(
                    Image.new("RGBA", heatmap_img.size, WHITE), heatmap_img
                ).convert("RGB"),
            ),
        ]
        panel_w, panel_h = panels[0][1].size
        width = panel_w * 3
        height = HEADER_HEIGHT * 2 + panel_h
        canvas = Image.new("RGB", (width, height), WHITE)
        draw = ImageDraw.Draw(canvas)
        title_font = _load_font(18, bold=True)
        header_font = _load_font(15)
        _draw_centered_text(draw, (0, 0, width, HEADER_HEIGHT), title, title_font)
        draw.line((0, HEADER_HEIGHT, width, HEADER_HEIGHT), fill=LINE_COLOR, width=1)
        for index, (label, image) in enumerate(panels):
            x = index * panel_w
            _draw_centered_text(
                draw,
                (x, HEADER_HEIGHT, x + panel_w, HEADER_HEIGHT * 2),
                label,
                header_font,
            )
            canvas.paste(image, (x, HEADER_HEIGHT * 2))
            if index > 0:
                draw.line((x, 0, x, height), fill=LINE_COLOR, width=1)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path, format="PNG")


def build_summary(metrics: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-panel pixel metrics.

    Parameters
    ----------
    metrics : Sequence[Mapping[str, Any]]
        Per-panel metric dictionaries.

    Returns
    -------
    dict[str, Any]
        Aggregate summary.
    """

    l1_values = [float(item["l1_rgb_per_pixel"]) for item in metrics]
    ssim_values = [float(item["ssim"]) for item in metrics]
    worst_l1 = sorted(metrics, key=lambda item: float(item["l1_rgb_per_pixel"]), reverse=True)
    worst_ssim = sorted(metrics, key=lambda item: float(item["ssim"]))
    return {
        "total_panels": len(metrics),
        "mean_l1_rgb_per_pixel": round(statistics.fmean(l1_values), 4) if l1_values else 0.0,
        "median_l1_rgb_per_pixel": round(statistics.median(l1_values), 4) if l1_values else 0.0,
        "max_l1_rgb_per_pixel": round(max(l1_values), 4) if l1_values else 0.0,
        "mean_ssim": round(statistics.fmean(ssim_values), 6) if ssim_values else 0.0,
        "median_ssim": round(statistics.median(ssim_values), 6) if ssim_values else 0.0,
        "min_ssim": round(min(ssim_values), 6) if ssim_values else 0.0,
        "panels_by_worst_ssim": [
            {
                "slug": str(item["slug"]),
                "ssim": float(item["ssim"]),
                "l1_rgb_per_pixel": float(item["l1_rgb_per_pixel"]),
            }
            for item in worst_ssim[:10]
        ],
        "panels_by_worst_l1": [
            {
                "slug": str(item["slug"]),
                "l1_rgb_per_pixel": float(item["l1_rgb_per_pixel"]),
                "ssim": float(item["ssim"]),
            }
            for item in worst_l1[:10]
        ],
    }


def write_summary_files(
    metrics: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    out_dir: Path,
) -> None:
    """Write aggregate JSON and Markdown reports.

    Parameters
    ----------
    metrics : Sequence[Mapping[str, Any]]
        Per-panel metrics.
    summary : Mapping[str, Any]
        Aggregate summary.
    out_dir : pathlib.Path
        Output directory.

    Returns
    -------
    None
        ``summary.json`` and ``summary.md`` are written.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"summary": dict(summary), "panels": list(metrics)}
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Graphviz Strict Pixel Diff Summary",
        "",
        f"- Panels: {summary['total_panels']}",
        f"- Mean L1 RGB / pixel: {summary['mean_l1_rgb_per_pixel']:.4f}",
        f"- Median L1 RGB / pixel: {summary['median_l1_rgb_per_pixel']:.4f}",
        f"- Max L1 RGB / pixel: {summary['max_l1_rgb_per_pixel']:.4f}",
        f"- Mean SSIM: {summary['mean_ssim']:.6f}",
        f"- Median SSIM: {summary['median_ssim']:.6f}",
        f"- Min SSIM: {summary['min_ssim']:.6f}",
        "",
        "## Worst Panels By SSIM",
        "",
        "| Rank | Panel | SSIM | L1 RGB / px |",
        "| ---: | --- | ---: | ---: |",
    ]
    for rank, item in enumerate(summary["panels_by_worst_ssim"], start=1):
        lines.append(
            f"| {rank} | `{item['slug']}` | {item['ssim']:.6f} | {item['l1_rgb_per_pixel']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Worst Panels By L1",
            "",
            "| Rank | Panel | L1 RGB / px | SSIM |",
            "| ---: | --- | ---: | ---: |",
        ]
    )
    for rank, item in enumerate(summary["panels_by_worst_l1"], start=1):
        lines.append(
            f"| {rank} | `{item['slug']}` | {item['l1_rgb_per_pixel']:.4f} | {item['ssim']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Per-Panel Region Metrics",
            "",
            "| Panel | L1 | SSIM | Text L1 | Node L1 | Edge/Arrow L1 | Background L1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in sorted(metrics, key=lambda value: float(value["ssim"])):
        regions = item["regions"]
        lines.append(
            f"| `{item['slug']}` | {float(item['l1_rgb_per_pixel']):.4f} | "
            f"{float(item['ssim']):.6f} | {float(regions['text_l1_rgb_per_pixel']):.4f} | "
            f"{float(regions['node_l1_rgb_per_pixel']):.4f} | "
            f"{float(regions['edge_arrow_l1_rgb_per_pixel']):.4f} | "
            f"{float(regions['background_l1_rgb_per_pixel']):.4f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _select_cases(requested: Sequence[str], quick: bool = False) -> List[gthc.GraphCase]:
    """Select comparison harness cases by slug.

    Parameters
    ----------
    requested : Sequence[str]
        Requested slugs; empty means all cases.
    quick : bool, default=False
        Whether to skip bundled YAML cases.

    Returns
    -------
    list[scripts.graphviz_theme_comparison.GraphCase]
        Selected cases.
    """

    cases = gthc._iter_cases(quick=quick)
    if not requested:
        return cases
    requested_set = {slug.strip() for slug in requested if slug.strip()}
    selected = [case for case in cases if case.slug in requested_set]
    missing = requested_set - {case.slug for case in selected}
    if missing:
        print(f"[warn] unknown case slugs: {sorted(missing)}", file=sys.stderr)
    return selected


def run_diff(cases: Sequence[gthc.GraphCase], out_dir: Path, dpi: int) -> Dict[str, Any]:
    """Run pixel diff for selected cases.

    Parameters
    ----------
    cases : Sequence[scripts.graphviz_theme_comparison.GraphCase]
        Cases to score.
    out_dir : pathlib.Path
        Output directory.
    dpi : int
        Raster DPI.

    Returns
    -------
    dict[str, Any]
        Aggregate payload.
    """

    metrics: List[Mapping[str, Any]] = []
    for case in cases:
        print(f"[render] {case.slug}")
        pair = render_pair(case, out_dir, dpi)
        metrics.append(score_pair(case, pair, out_dir))
    summary = build_summary(metrics)
    write_summary_files(metrics, summary, out_dir)
    return {"summary": summary, "panels": metrics}


def run_hires(cases: Sequence[gthc.GraphCase], out_dir: Path, dpi: int) -> None:
    """Render separate hi-res dot and Dagua images for selected cases.

    Parameters
    ----------
    cases : Sequence[scripts.graphviz_theme_comparison.GraphCase]
        Cases to render.
    out_dir : pathlib.Path
        Root output directory.
    dpi : int
        Requested hi-res DPI.

    Returns
    -------
    None
        Images are written under ``out_dir / "hires" / slug``.
    """

    for case in cases:
        case_dir = out_dir / "hires" / case.slug
        print(f"[hires] {case.slug}")
        pair = render_pair(case, case_dir, dpi, cap_longest_side=True)
        final_dot = case_dir / "dot.png"
        final_dagua = case_dir / "dagua.png"
        pair.dot_png.replace(final_dot)
        pair.dagua_png.replace(final_dagua)
        metadata = {
            "slug": case.slug,
            "title": case.title,
            "requested_dpi": dpi,
            "effective_dpi": pair.effective_dpi,
            "width_px": pair.dimensions[0],
            "height_px": pair.dimensions[1],
            "dot": str(final_dot),
            "dagua": str(final_dagua),
        }
        (case_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : Sequence[str], optional
        Explicit argument vector.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(description="Pixel-level graphviz_strict parity diff.")
    parser.add_argument(
        "--cases",
        default="",
        help="Comma-separated case slugs to diff; default is every panel.",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="Output directory.")
    parser.add_argument("--dpi", type=int, default=200, help="Pixel diff DPI.")
    parser.add_argument(
        "--hires",
        default="",
        help="Comma-separated case slugs to render as separate hi-res images.",
    )
    parser.add_argument("--hires-dpi", type=int, default=400, help="Requested hi-res DPI.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use only programmatic showcase cases when no explicit case list is given.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Parameters
    ----------
    argv : Sequence[str], optional
        Explicit argument vector.

    Returns
    -------
    int
        Process exit status.
    """

    args = _parse_args(argv)
    if shutil.which("dot") is None:
        print("[error] 'dot' binary not found on PATH", file=sys.stderr)
        return 1
    out_dir = Path(args.out)
    if args.hires:
        requested = [slug for slug in args.hires.split(",") if slug]
        cases = _select_cases(requested, quick=False)
        if not cases:
            print("[error] no hi-res cases selected", file=sys.stderr)
            return 1
        run_hires(cases, out_dir, int(args.hires_dpi))
        return 0
    requested = [slug for slug in args.cases.split(",") if slug] if args.cases else []
    cases = _select_cases(requested, quick=bool(args.quick))
    if not cases:
        print("[error] no cases selected", file=sys.stderr)
        return 1
    payload = run_diff(cases, out_dir, int(args.dpi))
    summary = payload["summary"]
    print()
    print("Pixel parity summary")
    print("--------------------")
    print(f"  panels:          {summary['total_panels']}")
    print(f"  mean L1 RGB/px:  {summary['mean_l1_rgb_per_pixel']:.4f}")
    print(f"  mean SSIM:       {summary['mean_ssim']:.6f}")
    print(f"  worst SSIM:      {summary['min_ssim']:.6f}")
    print(f"Wrote outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
