"""SVG feature extractors for visual parity v2 metrics.

The routines in this module are intentionally small and deterministic so they
can validate metric plumbing without invoking the full render harness. They
operate on SVG snippets emitted by Graphviz and return JSON-compatible
records consumed by ``scripts/parity_metrics.py`` and the tripwire tests.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXTRACTOR_VERSION = "vp2-lane-b-1"
SVG_NS = "http://www.w3.org/2000/svg"

Point = Tuple[float, float]


@dataclass(frozen=True)
class ShapePathFeature:
    """Feature record for a non-ellipse SVG node outline.

    Parameters
    ----------
    element_id
        SVG group title or id associated with the shape.
    tag
        SVG primitive tag, such as ``"path"`` or ``"polygon"``.
    command_inventory
        Counts of SVG path commands observed in the outline.
    bbox
        Bounding box in points as ``(min_x, min_y, max_x, max_y)``.
    area
        Polygonal area in square points.
    centroid
        Shape centroid in points.
    path_iou
        Raster-style path IoU when a candidate shape is supplied. Reference
        self-comparison returns ``1.0``.
    """

    element_id: str
    tag: str
    command_inventory: Dict[str, int]
    bbox: Tuple[float, float, float, float]
    area: float
    centroid: Point
    path_iou: float


def _strip_ns(tag: str) -> str:
    """Return an XML tag without its namespace.

    Parameters
    ----------
    tag
        ElementTree tag string.

    Returns
    -------
    str
        Namespace-free tag name.
    """

    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def _parse_float_tokens(text: str) -> List[float]:
    """Parse every numeric token from an SVG attribute.

    Parameters
    ----------
    text
        Raw SVG attribute text.

    Returns
    -------
    list[float]
        Parsed numeric values in source order.
    """

    return [float(token) for token in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text)]


def parse_points(points_attr: str) -> List[Point]:
    """Parse an SVG ``points`` attribute.

    Parameters
    ----------
    points_attr
        Raw ``points`` attribute.

    Returns
    -------
    list[tuple[float, float]]
        Parsed point sequence.
    """

    nums = _parse_float_tokens(points_attr)
    return [(nums[index], nums[index + 1]) for index in range(0, len(nums) - 1, 2)]


def path_command_inventory(path_data: str) -> Dict[str, int]:
    """Count command letters in an SVG path.

    Parameters
    ----------
    path_data
        Raw SVG ``d`` attribute.

    Returns
    -------
    dict[str, int]
        Upper-case command counts.
    """

    counts: Dict[str, int] = {}
    for command in re.findall(r"[A-Za-z]", path_data):
        key = command.upper()
        counts[key] = counts.get(key, 0) + 1
    return counts


def approximate_path_points(path_data: str) -> List[Point]:
    """Approximate an SVG path by its explicit coordinate pairs.

    Parameters
    ----------
    path_data
        Raw SVG ``d`` attribute.

    Returns
    -------
    list[tuple[float, float]]
        Coordinate pairs in source order.

    Notes
    -----
    This lightweight parser is sufficient for Graphviz node outlines, which
    are polygonal for the target non-ellipse shapes in the v2 fast tests.
    Curves are represented by their control/end points for bbox and centroid
    extraction; exact curve integration is deferred to Lane E1 geometry.
    """

    nums = _parse_float_tokens(path_data)
    return [(nums[index], nums[index + 1]) for index in range(0, len(nums) - 1, 2)]


def polygon_bbox(points: Sequence[Point]) -> Tuple[float, float, float, float]:
    """Compute a point sequence bounding box.

    Parameters
    ----------
    points
        Sequence of ``(x, y)`` points.

    Returns
    -------
    tuple[float, float, float, float]
        ``(min_x, min_y, max_x, max_y)`` or all zeros for an empty sequence.
    """

    if not points:
        return 0.0, 0.0, 0.0, 0.0
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def polygon_area(points: Sequence[Point]) -> float:
    """Compute polygon area with the shoelace formula.

    Parameters
    ----------
    points
        Polygon vertices.

    Returns
    -------
    float
        Absolute polygon area in square points.
    """

    if len(points) < 3:
        return 0.0
    total = 0.0
    for index, point in enumerate(points):
        nxt = points[(index + 1) % len(points)]
        total += point[0] * nxt[1] - nxt[0] * point[1]
    return abs(total) / 2.0


def polygon_centroid(points: Sequence[Point]) -> Point:
    """Compute a polygon centroid.

    Parameters
    ----------
    points
        Polygon vertices.

    Returns
    -------
    tuple[float, float]
        Centroid point. Empty input returns ``(0.0, 0.0)``.
    """

    if not points:
        return 0.0, 0.0
    signed_area = 0.0
    cx = 0.0
    cy = 0.0
    for index, point in enumerate(points):
        nxt = points[(index + 1) % len(points)]
        cross = point[0] * nxt[1] - nxt[0] * point[1]
        signed_area += cross
        cx += (point[0] + nxt[0]) * cross
        cy += (point[1] + nxt[1]) * cross
    if abs(signed_area) < 1e-9:
        return (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )
    factor = 1.0 / (3.0 * signed_area)
    return cx * factor, cy * factor


def bbox_iou(
    left: Tuple[float, float, float, float],
    right: Tuple[float, float, float, float],
) -> float:
    """Compute IoU between two axis-aligned boxes.

    Parameters
    ----------
    left
        First bbox as ``(min_x, min_y, max_x, max_y)``.
    right
        Second bbox as ``(min_x, min_y, max_x, max_y)``.

    Returns
    -------
    float
        Intersection-over-union in ``[0, 1]``.
    """

    ix0 = max(left[0], right[0])
    iy0 = max(left[1], right[1])
    ix1 = min(left[2], right[2])
    iy1 = min(left[3], right[3])
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - inter
    return inter / union if union > 0.0 else 0.0


def _shape_points(element: ET.Element) -> Tuple[str, Dict[str, int], List[Point]]:
    """Extract outline points from one SVG shape element.

    Parameters
    ----------
    element
        SVG outline element.

    Returns
    -------
    tuple[str, dict[str, int], list[tuple[float, float]]]
        Shape tag, command inventory, and approximate outline points.
    """

    tag = _strip_ns(element.tag)
    if tag in {"polygon", "polyline"}:
        points = parse_points(element.attrib.get("points", ""))
        return tag, {"POINT": len(points)}, points
    if tag == "path":
        path_data = element.attrib.get("d", "")
        return tag, path_command_inventory(path_data), approximate_path_points(path_data)
    if tag == "rect":
        x = float(element.attrib.get("x", "0") or 0.0)
        y = float(element.attrib.get("y", "0") or 0.0)
        w = float(element.attrib.get("width", "0") or 0.0)
        h = float(element.attrib.get("height", "0") or 0.0)
        return tag, {"RECT": 1}, [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    return tag, {}, []


def extract_shape_paths(svg_text: str) -> List[Dict[str, Any]]:
    """Extract non-ellipse node outline features from SVG text.

    Parameters
    ----------
    svg_text
        SVG document text.

    Returns
    -------
    list[dict[str, Any]]
        JSON-compatible shape feature records.
    """

    root = ET.fromstring(svg_text)
    records: List[Dict[str, Any]] = []
    for group in root.iter():
        if _strip_ns(group.tag) != "g" or group.attrib.get("class") != "node":
            continue
        title_el = group.find(f"{{{SVG_NS}}}title")
        if title_el is not None:
            element_id = (title_el.text or group.attrib.get("id", "node")).strip()
        else:
            element_id = group.attrib.get("id", "node")
        for child in group:
            tag = _strip_ns(child.tag)
            if tag in {"ellipse", "text", "title"}:
                continue
            if tag not in {"polygon", "polyline", "path", "rect"}:
                continue
            shape_tag, commands, points = _shape_points(child)
            if not points:
                continue
            bbox = polygon_bbox(points)
            record = ShapePathFeature(
                element_id=element_id,
                tag=shape_tag,
                command_inventory=commands,
                bbox=bbox,
                area=polygon_area(points),
                centroid=polygon_centroid(points),
                path_iou=1.0,
            )
            records.append(
                {
                    "element_id": record.element_id,
                    "tag": record.tag,
                    "command_inventory": record.command_inventory,
                    "bbox": list(record.bbox),
                    "area": round(record.area, 4),
                    "centroid": [round(record.centroid[0], 4), round(record.centroid[1], 4)],
                    "path_iou": record.path_iou,
                }
            )
            break
    return records


def polygon_projected_metrics(points: Sequence[Point], tangent: Point) -> Dict[str, float]:
    """Measure polygon length and width along an edge tangent.

    Parameters
    ----------
    points
        Arrow polygon vertices.
    tangent
        Edge tangent vector.

    Returns
    -------
    dict[str, float]
        Projected ``length`` and ``width`` in points.
    """

    if not points:
        return {"length": 0.0, "width": 0.0}
    norm = math.hypot(tangent[0], tangent[1]) or 1.0
    ux = tangent[0] / norm
    uy = tangent[1] / norm
    px = -uy
    py = ux
    axial = [point[0] * ux + point[1] * uy for point in points]
    lateral = [point[0] * px + point[1] * py for point in points]
    return {"length": max(axial) - min(axial), "width": max(lateral) - min(lateral)}


def classify_fill_mode(fill: str, stroke: str = "") -> str:
    """Classify an SVG arrowhead fill mode.

    Parameters
    ----------
    fill
        SVG fill value.
    stroke
        SVG stroke value.

    Returns
    -------
    str
        ``"filled"``, ``"hollow"``, or ``"stroked"``.
    """

    fill_norm = fill.strip().lower()
    stroke_norm = stroke.strip().lower()
    if fill_norm in {"", "none", "transparent"}:
        return "stroked" if stroke_norm not in {"", "none", "transparent"} else "hollow"
    if fill_norm in {"white", "#ffffff", "#fff"} and stroke_norm not in {"", "none"}:
        return "hollow"
    return "filled"


def parse_arrow_compound_order(arrow_name: str) -> List[str]:
    """Parse a Graphviz compound arrow name into primitive order.

    Parameters
    ----------
    arrow_name
        Arrow name such as ``"normalopen"`` or ``"lteevee"``.

    Returns
    -------
    list[str]
        Best-effort primitive sequence with side/fill modifiers stripped.
    """

    primitives = [
        "crow",
        "curve",
        "icurve",
        "diamond",
        "odiamond",
        "normal",
        "inv",
        "dot",
        "odot",
        "box",
        "obox",
        "tee",
        "vee",
        "open",
        "none",
    ]
    lowered = arrow_name.lower()
    cleaned = lowered.replace("l", "", 1) if lowered.startswith("l") else lowered
    cleaned = cleaned.replace("r", "", 1) if cleaned.startswith("r") else cleaned
    order: List[str] = []
    cursor = 0
    while cursor < len(cleaned):
        match = next((name for name in primitives if cleaned.startswith(name, cursor)), None)
        if match is None:
            cursor += 1
            continue
        order.append(match)
        cursor += len(match)
    return order or [arrow_name]


def side_clip(arrow_name: str) -> str:
    """Return the Graphviz l/r side clip modifier for an arrow.

    Parameters
    ----------
    arrow_name
        Arrow name.

    Returns
    -------
    str
        ``"left"``, ``"right"``, or ``"none"``.
    """

    lowered = arrow_name.lower()
    if lowered.startswith("l"):
        return "left"
    if lowered.startswith("r"):
        return "right"
    return "none"


def arrow_metric_family(
    reference_points: Sequence[Point],
    candidate_points: Sequence[Point],
    tangent: Point = (1.0, 0.0),
    reference_fill: str = "black",
    candidate_fill: str = "black",
    reference_arrow: str = "normal",
    candidate_arrow: str = "normal",
) -> Dict[str, Any]:
    """Compute the v2 arrow metric family for one arrowhead.

    Parameters
    ----------
    reference_points
        Reference arrow polygon vertices.
    candidate_points
        Candidate arrow polygon vertices.
    tangent
        Edge tangent vector used for projected length/width.
    reference_fill
        Reference SVG fill value.
    candidate_fill
        Candidate SVG fill value.
    reference_arrow
        Reference Graphviz arrow name.
    candidate_arrow
        Candidate Graphviz arrow name.

    Returns
    -------
    dict[str, Any]
        Metrics for polygon IoU, projected length/width, fill mode, compound
        order, and side clip.
    """

    ref_proj = polygon_projected_metrics(reference_points, tangent)
    cand_proj = polygon_projected_metrics(candidate_points, tangent)
    ref_bbox = polygon_bbox(reference_points)
    cand_bbox = polygon_bbox(candidate_points)
    ref_order = parse_arrow_compound_order(reference_arrow)
    cand_order = parse_arrow_compound_order(candidate_arrow)
    return {
        "arrow_polygon_iou": bbox_iou(ref_bbox, cand_bbox),
        "arrow_len_pct": _relative_delta_pct(ref_proj["length"], cand_proj["length"]),
        "arrow_width_pct": _relative_delta_pct(ref_proj["width"], cand_proj["width"]),
        "arrow_fill_mode": {
            "target": classify_fill_mode(reference_fill),
            "dagua": classify_fill_mode(candidate_fill),
            "match": classify_fill_mode(reference_fill) == classify_fill_mode(candidate_fill),
        },
        "arrow_compound_order": {
            "target": ref_order,
            "dagua": cand_order,
            "match": ref_order == cand_order,
        },
        "arrow_side_clip": {
            "target": side_clip(reference_arrow),
            "dagua": side_clip(candidate_arrow),
            "match": side_clip(reference_arrow) == side_clip(candidate_arrow),
        },
    }


def _relative_delta_pct(target: float, candidate: float) -> float:
    """Return absolute relative delta percentage.

    Parameters
    ----------
    target
        Target value.
    candidate
        Candidate value.

    Returns
    -------
    float
        Absolute percent delta.
    """

    if abs(target) < 1e-9:
        return 0.0 if abs(candidate) < 1e-9 else 100.0
    return abs(candidate - target) / abs(target) * 100.0


def label_glyph_extent(
    text: str,
    font_size_pt: float,
    target_kind: str,
    font_resolver: str = "matplotlib",
    resolved_font_file: Optional[str] = None,
    width_scale: float = 0.6,
) -> Dict[str, Any]:
    """Estimate a label glyph ink extent with provenance.

    Parameters
    ----------
    text
        Label text.
    font_size_pt
        Font size in points.
    target_kind
        Target lane, e.g. ``"svg_declared"``.
    font_resolver
        Resolver name.
    resolved_font_file
        Resolved font file path when known.
    width_scale
        Average glyph-width multiplier.

    Returns
    -------
    dict[str, Any]
        Width/height estimate and required provenance fields.
    """

    font_file = resolved_font_file
    if font_file is None:
        font_file = str(Path("unknown"))
    return {
        "width_pt": round(len(text) * font_size_pt * width_scale, 4),
        "height_pt": round(font_size_pt, 4),
        "font_resolver": font_resolver,
        "resolved_font_file": font_file,
        "target_kind": target_kind,
    }


def cluster_rect_features(svg_text: str) -> List[Dict[str, Any]]:
    """Extract cluster rectangle and border segment features.

    Parameters
    ----------
    svg_text
        SVG document text.

    Returns
    -------
    list[dict[str, Any]]
        Cluster geometry records.
    """

    root = ET.fromstring(svg_text)
    records: List[Dict[str, Any]] = []
    for group in root.iter():
        if _strip_ns(group.tag) != "g" or group.attrib.get("class") != "cluster":
            continue
        title_el = group.find(f"{{{SVG_NS}}}title")
        if title_el is not None:
            title = (title_el.text or group.attrib.get("id", "cluster")).strip()
        else:
            title = group.attrib.get("id", "cluster")
        polygon = group.find(f"{{{SVG_NS}}}polygon")
        if polygon is None:
            continue
        points = parse_points(polygon.attrib.get("points", ""))
        bbox = polygon_bbox(points)
        records.append(
            {
                "cluster_id": title,
                "bbox": list(bbox),
                "border_segments": len(points),
                "area": round(polygon_area(points), 4),
            }
        )
    return records


def edge_trim_distance(edge_endpoint: Point, node_bbox: Tuple[float, float, float, float]) -> float:
    """Measure distance from an edge endpoint to a node boundary box.

    Parameters
    ----------
    edge_endpoint
        Endpoint point in points.
    node_bbox
        Node boundary box as ``(min_x, min_y, max_x, max_y)``.

    Returns
    -------
    float
        Distance in points. Points inside or on the box return ``0.0``.
    """

    x, y = edge_endpoint
    dx = max(node_bbox[0] - x, 0.0, x - node_bbox[2])
    dy = max(node_bbox[1] - y, 0.0, y - node_bbox[3])
    return math.hypot(dx, dy)


def symmetric_mean_point_to_polyline(left: Sequence[Point], right: Sequence[Point]) -> float:
    """Compute symmetric mean point-to-polyline distance.

    Parameters
    ----------
    left
        First polyline.
    right
        Second polyline.

    Returns
    -------
    float
        Mean symmetric distance in points.
    """

    if not left and not right:
        return 0.0
    if not left or not right:
        return float("inf")
    left_mean = sum(_point_to_polyline_distance(point, right) for point in left) / len(left)
    right_mean = sum(_point_to_polyline_distance(point, left) for point in right) / len(right)
    return (left_mean + right_mean) / 2.0


def _point_to_polyline_distance(point: Point, polyline: Sequence[Point]) -> float:
    """Compute point-to-polyline distance.

    Parameters
    ----------
    point
        Query point.
    polyline
        Polyline vertices.

    Returns
    -------
    float
        Minimum segment distance.
    """

    if len(polyline) == 1:
        return math.hypot(point[0] - polyline[0][0], point[1] - polyline[0][1])
    return min(
        _point_to_segment_distance(point, polyline[index], polyline[index + 1])
        for index in range(len(polyline) - 1)
    )


def _point_to_segment_distance(point: Point, start: Point, end: Point) -> float:
    """Compute point-to-segment distance.

    Parameters
    ----------
    point
        Query point.
    start
        Segment start.
    end
        Segment end.

    Returns
    -------
    float
        Euclidean distance.
    """

    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return math.hypot(point[0] - sx, point[1] - sy)
    t = max(0.0, min(1.0, ((point[0] - sx) * dx + (point[1] - sy) * dy) / denom))
    px = sx + t * dx
    py = sy + t * dy
    return math.hypot(point[0] - px, point[1] - py)


def extract_svg_features(svg_text: str) -> Dict[str, Any]:
    """Extract all Lane B SVG-side v2 feature families.

    Parameters
    ----------
    svg_text
        SVG document text.

    Returns
    -------
    dict[str, Any]
        Feature-family payload with extractor provenance.
    """

    return {
        "extractor_version": EXTRACTOR_VERSION,
        "shape_paths": extract_shape_paths(svg_text),
        "clusters": cluster_rect_features(svg_text),
    }
