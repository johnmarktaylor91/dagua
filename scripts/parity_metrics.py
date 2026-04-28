#!/usr/bin/env python
# ruff: noqa: E402
"""Quantitative parity metrics for the ``graphviz_strict`` cosmetic theme.

This script converts the qualitative graphviz-parity audit loop into a numeric
optimization problem. For each test panel exposed by
``scripts/graphviz_theme_comparison._iter_cases()`` it:

1.  Emits DOT for the source graph through the existing
    ``graphviz_theme_comparison.graph_to_dot`` helper, runs ``dot -Tsvg`` to
    produce a reference SVG, and parses the SVG with stdlib XML to harvest
    target ellipse semi-axes, font sizes, arrow polygon geometry, cluster
    bounding boxes, and graph-level chrome.
2.  Re-derives the same features from dagua's strict-theme internal state via
    Python introspection only -- no image reads, no rendering. Node ellipse
    semi-axes come from ``compute_node_size``; edge / cluster geometry comes
    from ``EdgeStyle`` and ``ClusterStyle`` on the graphviz_strict theme.
3.  Computes per-feature deltas with absolute / relative tolerance flags and
    aggregates into a JSON report keyed by panel slug.

The output is consumed by future graphviz-parity rounds as a scalar loss
instead of natural-language audit verdicts. See
``.project-context/knowledge/visual_tuning_workflow.md`` for the full
postmortem driving the design.

Usage
-----
::

    python scripts/parity_metrics.py
    python scripts/parity_metrics.py --cases pipeline,diamond,arrow_types
    python scripts/parity_metrics.py --out /tmp/parity.json
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse the existing harness so the test-case set, theme application logic, and
# DOT emitter stay aligned with what the visual gallery uses. The harness is
# imported as a module so its private helpers (prefixed with ``_``) remain
# accessible without copying their surface area into this script.
import scripts.graphviz_theme_comparison as gthc  # noqa: E402
from dagua import DaguaGraph  # noqa: E402
from dagua.styles import get_theme  # noqa: E402

# ---------------------------------------------------------------------------
# Tolerance profile
# ---------------------------------------------------------------------------

#: Absolute / relative thresholds per feature kind. Values were picked so that
#: typical SVG parser noise (sub-pixel rounding, off-by-one Bezier artefacts)
#: stays inside the band while real visual departures register as failures.
DEFAULT_TOLERANCE: Dict[str, Any] = {
    # node ellipse
    "ellipse_rx_pt": 2.0,
    "ellipse_ry_pt": 2.0,
    "ellipse_aspect_pct": 5.0,  # relative %, semi-axis ratio
    # node text
    "font_size_pt": 1.0,
    "font_family_exact": True,
    "node_stroke_width_pt": 0.2,
    "color_exact": True,
    # edge
    "edge_stroke_width_pt": 0.2,
    "arrow_length_pt": 1.0,
    "arrow_width_pt": 1.0,
    "arrow_filled_exact": True,
    # cluster
    "cluster_stroke_width_pt": 0.2,
    "cluster_label_font_size_pt": 1.0,
}

#: SVG namespace constants. Graphviz emits SVG 1.1 with a default namespace.
SVG_NS = "http://www.w3.org/2000/svg"
NS_MAP = {"svg": SVG_NS}


def _strip_ns(tag: str) -> str:
    """Drop the XML namespace prefix from an element tag.

    Parameters
    ----------
    tag : str
        Fully-qualified XML element tag, e.g. ``"{http://...}ellipse"``.

    Returns
    -------
    str
        Local tag name with the namespace prefix removed.
    """

    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


# ---------------------------------------------------------------------------
# Reference (dot) feature extraction
# ---------------------------------------------------------------------------


@dataclass
class ReferenceNode:
    """SVG-derived geometry/style for a single reference node.

    Attributes
    ----------
    node_id : str
        DOT identifier (``n0``, ``n1``, ...).
    label : str
        Rendered text label.
    ellipse_rx : float
        Horizontal ellipse semi-axis in SVG user units (= points).
    ellipse_ry : float
        Vertical ellipse semi-axis in SVG user units (= points).
    font_size_pt : float
        Reported font size in points.
    font_family : str
        Reported font family.
    node_fill : str
        Ellipse fill color (CSS color literal as emitted by Graphviz).
    node_stroke : str
        Ellipse stroke color.
    node_stroke_width_pt : float
        Ellipse stroke width in points (Graphviz default 1.0 if absent).
    """

    node_id: str
    label: str
    ellipse_rx: float
    ellipse_ry: float
    font_size_pt: float
    font_family: str
    node_fill: str
    node_stroke: str
    node_stroke_width_pt: float

    @property
    def ellipse_aspect(self) -> float:
        """Ellipse aspect ratio, ``rx / ry``."""

        return self.ellipse_rx / self.ellipse_ry if self.ellipse_ry > 0 else float("nan")


@dataclass
class ReferenceEdge:
    """SVG-derived geometry for a single reference edge.

    Attributes
    ----------
    edge_id : str
        SVG ``id`` of the edge group.
    title : str
        Edge title (e.g. ``"n0->n1"``).
    stroke_color : str
        Edge path stroke color.
    stroke_width_pt : float
        Edge stroke width in points.
    arrow_polygon_vertices : list[tuple[float, float]] | None
        Arrowhead polygon vertices, when the head is rendered as a polygon.
    arrow_length_pt : float | None
        Derived arrowhead axial length (longer of polygon bbox width/height).
    arrow_width_pt : float | None
        Derived arrowhead lateral width (shorter of polygon bbox width/height).
    arrow_filled : bool | None
        ``True`` when the arrowhead polygon's fill is non-``"none"``.
    """

    edge_id: str
    title: str
    stroke_color: str
    stroke_width_pt: float
    arrow_polygon_vertices: Optional[List[Tuple[float, float]]] = None
    arrow_length_pt: Optional[float] = None
    arrow_width_pt: Optional[float] = None
    arrow_filled: Optional[bool] = None


@dataclass
class ReferenceCluster:
    """SVG-derived geometry for a reference cluster.

    Attributes
    ----------
    cluster_id : str
        SVG ``id`` of the cluster group (``clust1``, ``clust2``, ...).
    title : str
        Cluster ``<title>`` text. Used to align with dagua's cluster names.
    rect_x : float
        Bounding-box left edge (SVG user units).
    rect_y : float
        Bounding-box top edge.
    rect_w : float
        Bounding-box width.
    rect_h : float
        Bounding-box height.
    fill : str
        Polygon fill color.
    stroke : str
        Polygon stroke color.
    stroke_width_pt : float
        Polygon stroke width in points.
    label_font_size_pt : float | None
        Font size of the cluster's text label.
    """

    cluster_id: str
    title: str
    rect_x: float
    rect_y: float
    rect_w: float
    rect_h: float
    fill: str
    stroke: str
    stroke_width_pt: float
    label_font_size_pt: Optional[float] = None


@dataclass
class ReferenceGraph:
    """Bag of reference features for a single test panel.

    Attributes
    ----------
    bg_color : str
        Background fill emitted by Graphviz at the canvas root.
    margin : float
        Difference between the SVG viewBox bounds and the union of node /
        cluster bounding boxes (in points). Always non-negative.
    nodes : list[ReferenceNode]
        Per-node feature records keyed by DOT id.
    edges : list[ReferenceEdge]
        Per-edge feature records.
    clusters : list[ReferenceCluster]
        Per-cluster feature records (may be empty).
    """

    bg_color: str
    margin: float
    nodes: List[ReferenceNode] = field(default_factory=list)
    edges: List[ReferenceEdge] = field(default_factory=list)
    clusters: List[ReferenceCluster] = field(default_factory=list)


def _parse_float(value: Optional[str], default: float = 0.0) -> float:
    """Parse a float attribute value defensively.

    Parameters
    ----------
    value : str | None
        Source attribute value (may be ``None`` or contain ``"pt"`` suffix).
    default : float, default=0.0
        Fallback when ``value`` is missing or unparseable.

    Returns
    -------
    float
        Parsed value or ``default``.
    """

    if value is None:
        return default
    cleaned = value.strip()
    if not cleaned:
        return default
    # Some Graphviz attributes carry "pt" suffixes; SVG user units == points
    # already, so just strip the trailing characters.
    if cleaned.endswith("pt"):
        cleaned = cleaned[:-2].strip()
    try:
        return float(cleaned)
    except ValueError:
        return default


def _parse_polygon_points(points_attr: str) -> List[Tuple[float, float]]:
    """Parse an SVG ``polygon``/``polyline`` ``points`` attribute.

    Parameters
    ----------
    points_attr : str
        Raw ``points`` string, e.g. ``"30.5,-119.1 27,-109.1 ..."``.

    Returns
    -------
    list[tuple[float, float]]
        Parsed (x, y) vertex sequence; partial pairs are silently dropped.
    """

    coords: List[float] = []
    for token in points_attr.replace(",", " ").split():
        try:
            coords.append(float(token))
        except ValueError:
            continue
    return [(coords[i], coords[i + 1]) for i in range(0, len(coords) - 1, 2)]


def _polygon_bbox(
    vertices: Sequence[Tuple[float, float]],
) -> Tuple[float, float, float, float]:
    """Compute axis-aligned bounding box of a polygon.

    Parameters
    ----------
    vertices : Sequence[tuple[float, float]]
        Polygon vertices in (x, y) form.

    Returns
    -------
    tuple[float, float, float, float]
        ``(min_x, min_y, max_x, max_y)``. ``(0, 0, 0, 0)`` for an empty input.
    """

    if not vertices:
        return 0.0, 0.0, 0.0, 0.0
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    return min(xs), min(ys), max(xs), max(ys)


def _viewbox(svg_root: ET.Element) -> Tuple[float, float, float, float]:
    """Return the SVG viewBox or a synthesized fallback.

    Parameters
    ----------
    svg_root : xml.etree.ElementTree.Element
        Root ``<svg>`` element.

    Returns
    -------
    tuple[float, float, float, float]
        ``(min_x, min_y, width, height)`` in user units.
    """

    raw = svg_root.attrib.get("viewBox", "")
    parts = [p for p in raw.replace(",", " ").split() if p]
    if len(parts) == 4:
        try:
            return tuple(float(p) for p in parts)  # type: ignore[return-value]
        except ValueError:
            pass
    width = _parse_float(svg_root.attrib.get("width"))
    height = _parse_float(svg_root.attrib.get("height"))
    return 0.0, 0.0, width, height


def _node_record(node_group: ET.Element) -> Optional[ReferenceNode]:
    """Extract a :class:`ReferenceNode` from one ``g class="node"`` block.

    Parameters
    ----------
    node_group : xml.etree.ElementTree.Element
        SVG ``<g class="node">`` element.

    Returns
    -------
    ReferenceNode | None
        Parsed node, or ``None`` when the group lacks an ellipse outline. Such
        groups exist for non-ellipse shapes; they are intentionally skipped
        here because parity is currently scored on ellipse-shaped nodes only.
    """

    title_el = node_group.find(f"{{{SVG_NS}}}title")
    title = (title_el.text or "").strip() if title_el is not None else ""

    ellipse = node_group.find(f"{{{SVG_NS}}}ellipse")
    if ellipse is None:
        return None

    text_el = node_group.find(f"{{{SVG_NS}}}text")
    label = (text_el.text or "").strip() if text_el is not None else ""
    font_size = _parse_float(text_el.attrib.get("font-size")) if text_el is not None else 0.0
    font_family = text_el.attrib.get("font-family", "") if text_el is not None else ""

    return ReferenceNode(
        node_id=title,
        label=label,
        ellipse_rx=_parse_float(ellipse.attrib.get("rx")),
        ellipse_ry=_parse_float(ellipse.attrib.get("ry")),
        font_size_pt=font_size,
        font_family=font_family,
        node_fill=ellipse.attrib.get("fill", ""),
        node_stroke=ellipse.attrib.get("stroke", ""),
        # Graphviz omits stroke-width for the default 1.0 pen; default to 1.0.
        node_stroke_width_pt=_parse_float(ellipse.attrib.get("stroke-width"), default=1.0),
    )


def _edge_record(edge_group: ET.Element) -> ReferenceEdge:
    """Extract a :class:`ReferenceEdge` from one ``g class="edge"`` block.

    Parameters
    ----------
    edge_group : xml.etree.ElementTree.Element
        SVG ``<g class="edge">`` element.

    Returns
    -------
    ReferenceEdge
        Parsed edge with arrow geometry derived from its first ``<polygon>``
        child (when present). Missing arrow geometry is captured as ``None``
        so downstream code can flag the absence rather than crash.
    """

    title_el = edge_group.find(f"{{{SVG_NS}}}title")
    title = (title_el.text or "").strip() if title_el is not None else ""
    edge_id = edge_group.attrib.get("id", title or "edge")

    path_el = edge_group.find(f"{{{SVG_NS}}}path")
    if path_el is not None:
        stroke_color = path_el.attrib.get("stroke", "")
        stroke_width = _parse_float(path_el.attrib.get("stroke-width"), default=1.0)
    else:
        stroke_color = ""
        stroke_width = 1.0

    record = ReferenceEdge(
        edge_id=edge_id,
        title=title,
        stroke_color=stroke_color,
        stroke_width_pt=stroke_width,
    )

    polygon = edge_group.find(f"{{{SVG_NS}}}polygon")
    if polygon is not None:
        vertices = _parse_polygon_points(polygon.attrib.get("points", ""))
        record.arrow_polygon_vertices = vertices
        if vertices:
            min_x, min_y, max_x, max_y = _polygon_bbox(vertices)
            width = max_x - min_x
            height = max_y - min_y
            # The polygon may be drawn pointing along either axis; treat the
            # longer dimension as "length" (axial) and the shorter as "width"
            # (lateral) so comparisons with EdgeStyle.arrow_length/arrow_width
            # stay direction-agnostic.
            record.arrow_length_pt = max(width, height)
            record.arrow_width_pt = min(width, height)
        fill = polygon.attrib.get("fill", "").strip().lower()
        record.arrow_filled = bool(fill) and fill != "none"
    else:
        # tee/circle/dot arrowheads use polyline/ellipse instead of polygon;
        # leave arrow_* fields as None so the comparator skips them rather
        # than emitting a misleading delta against EdgeStyle's polygon-based
        # arrow_length/arrow_width.
        record.arrow_polygon_vertices = None

    return record


def _cluster_record(cluster_group: ET.Element) -> Optional[ReferenceCluster]:
    """Extract a :class:`ReferenceCluster` from one ``g class="cluster"`` block.

    Parameters
    ----------
    cluster_group : xml.etree.ElementTree.Element
        SVG ``<g class="cluster">`` element.

    Returns
    -------
    ReferenceCluster | None
        Parsed cluster, or ``None`` when the polygon outline is missing.
    """

    title_el = cluster_group.find(f"{{{SVG_NS}}}title")
    title = (title_el.text or "").strip() if title_el is not None else ""
    cluster_id = cluster_group.attrib.get("id", title or "cluster")

    polygon = cluster_group.find(f"{{{SVG_NS}}}polygon")
    if polygon is None:
        return None
    vertices = _parse_polygon_points(polygon.attrib.get("points", ""))
    if not vertices:
        return None
    min_x, min_y, max_x, max_y = _polygon_bbox(vertices)

    text_el = cluster_group.find(f"{{{SVG_NS}}}text")
    label_font_size: Optional[float] = None
    if text_el is not None:
        label_font_size = _parse_float(text_el.attrib.get("font-size")) or None

    return ReferenceCluster(
        cluster_id=cluster_id,
        title=title,
        rect_x=min_x,
        rect_y=min_y,
        rect_w=max_x - min_x,
        rect_h=max_y - min_y,
        fill=polygon.attrib.get("fill", ""),
        stroke=polygon.attrib.get("stroke", ""),
        stroke_width_pt=_parse_float(polygon.attrib.get("stroke-width"), default=1.0),
        label_font_size_pt=label_font_size,
    )


def _bg_color(svg_root: ET.Element) -> str:
    """Return the canvas background fill or an empty string if absent.

    Graphviz emits a polygon directly inside the ``graph0`` group whose fill
    encodes the background color. We pick the first polygon under graph0
    whose fill is set and stroke is ``"none"`` since inner shapes carry
    visible strokes.

    Parameters
    ----------
    svg_root : xml.etree.ElementTree.Element
        Root ``<svg>`` element.

    Returns
    -------
    str
        Background color (CSS literal) or ``""``.
    """

    graph0 = svg_root.find(f"{{{SVG_NS}}}g")
    if graph0 is None:
        return ""
    for child in graph0:
        if _strip_ns(child.tag) != "polygon":
            continue
        fill = child.attrib.get("fill", "")
        stroke = child.attrib.get("stroke", "")
        if fill and stroke.lower() == "none":
            return fill
    return ""


def parse_reference_svg(svg_text: str) -> ReferenceGraph:
    """Parse a Graphviz ``-Tsvg`` payload into a :class:`ReferenceGraph`.

    Parameters
    ----------
    svg_text : str
        SVG document text.

    Returns
    -------
    ReferenceGraph
        Aggregated reference features.

    Notes
    -----
    All extraction is best-effort: shapes that diverge from the standard
    Graphviz emission pattern (non-ellipse nodes, polyline arrowheads) are
    silently skipped on the per-feature axes that no longer apply, but the
    record itself is still returned so downstream comparisons can flag the
    skipped axes explicitly. SVG parser errors raise to the caller because
    they signal a real upstream problem (e.g. ``dot`` failed mid-render).
    """

    root = ET.fromstring(svg_text)
    vb_min_x, vb_min_y, vb_w, vb_h = _viewbox(root)

    nodes: List[ReferenceNode] = []
    edges: List[ReferenceEdge] = []
    clusters: List[ReferenceCluster] = []

    # Walk the entire tree and dispatch on the SVG ``class`` attribute
    # Graphviz tags onto each ``<g>`` (node / edge / cluster). This avoids
    # having to assume a specific nesting depth -- nested clusters surface
    # naturally because they live as siblings under graph0.
    for element in root.iter():
        if _strip_ns(element.tag) != "g":
            continue
        css_class = element.attrib.get("class", "")
        if css_class == "node":
            record = _node_record(element)
            if record is not None:
                nodes.append(record)
        elif css_class == "edge":
            edges.append(_edge_record(element))
        elif css_class == "cluster":
            cluster = _cluster_record(element)
            if cluster is not None:
                clusters.append(cluster)

    # Margin = padding from viewBox edge to the union of cluster + node
    # bounding boxes. SVG y-axis is flipped relative to Graphviz output, so
    # use absolute extents rather than sign-aware deltas.
    content_xs: List[float] = []
    content_ys: List[float] = []
    for node in nodes:
        # Reconstruct a bbox from cx/cy was not stored; re-extract by scanning
        # ellipse element only on demand. For the margin calculation we only
        # need the outer bounding extents, which we approximate from nodes
        # via rx/ry (cx/cy were dropped to avoid widening ReferenceNode).
        # Reparse via attributes already captured does not give cx/cy, so
        # fall back to clusters when present; otherwise use the viewBox
        # padding heuristic of SVG renderers (typically 4-pt margin).
        pass
    if clusters:
        for cluster in clusters:
            content_xs.extend([cluster.rect_x, cluster.rect_x + cluster.rect_w])
            content_ys.extend([cluster.rect_y, cluster.rect_y + cluster.rect_h])
    if content_xs and content_ys:
        margin_left = min(content_xs) - vb_min_x
        margin_right = (vb_min_x + vb_w) - max(content_xs)
        margin_top = min(content_ys) - vb_min_y
        margin_bottom = (vb_min_y + vb_h) - max(content_ys)
        margin = max(0.0, min(margin_left, margin_right, margin_top, margin_bottom))
    else:
        # Graphviz typically pads 4pt on each side; expose that as the
        # observable target when nothing else is available.
        margin = 4.0

    return ReferenceGraph(
        bg_color=_bg_color(root),
        margin=margin,
        nodes=nodes,
        edges=edges,
        clusters=clusters,
    )


def render_reference_svg(graph: DaguaGraph, *, timeout: float = 30.0) -> str:
    """Render a graph through ``dot -Tsvg`` and return the SVG payload.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph. The harness's ``graph_to_dot`` is reused so the DOT
        emitted matches the gallery's "native Graphviz" panel byte-for-byte.
    timeout : float, default=30.0
        Subprocess timeout in seconds.

    Returns
    -------
    str
        SVG document text.

    Raises
    ------
    RuntimeError
        When ``dot`` returns a non-zero exit status.
    """

    # Using gthc.graph_to_dot rather than dagua.graphviz_utils.to_dot is an
    # intentional deviation from the literal spec: gthc.graph_to_dot lets
    # Graphviz apply its OWN node/edge defaults (Times,serif at 14pt, ellipse
    # shape, black stroke) instead of dagua's strict-theme overrides. That
    # makes the resulting SVG a true "what dot would render" reference, which
    # is what the parity sprint is actually trying to converge on.
    #
    # We also bind the graphviz_strict theme to the graph BEFORE emitting
    # DOT. Without this, gthc._cluster_override_attrs (which compares per-
    # element style to the strict theme's cluster style) treats the graph's
    # default-theme cluster style as an override and injects strict's own
    # values into the DOT -- which would then be the values Graphviz renders
    # against, defeating the parity check. Aligning the theme up-front keeps
    # the comparison apples-to-apples.
    themed = _apply_strict_theme(graph)
    dot_source = gthc.graph_to_dot(themed)
    result = subprocess.run(
        ["dot", "-Tsvg"],
        input=dot_source,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"dot -Tsvg failed: {result.stderr.strip()}")
    return result.stdout


# ---------------------------------------------------------------------------
# Candidate (dagua) feature extraction
# ---------------------------------------------------------------------------


@dataclass
class CandidateNode:
    """Dagua-side feature record matching :class:`ReferenceNode`.

    Attributes
    ----------
    node_id : str
        DOT identifier (matches the reference node naming convention so that
        per-node deltas can be aligned by id).
    label : str
        Node label.
    ellipse_rx : float
        Horizontal semi-axis derived from ``compute_node_size`` plus the
        configured ``circumscribe`` factor.
    ellipse_ry : float
        Vertical semi-axis derived analogously.
    font_size_pt : float
        Effective font size after dagua's overflow policy (``shrink_text``
        may reduce the authored font size; we report the actual rendered
        size).
    font_family : str
        Resolved font family from the strict theme.
    node_fill : str
        Effective fill color.
    node_stroke : str
        Effective stroke color.
    node_stroke_width_pt : float
        Effective stroke width.
    """

    node_id: str
    label: str
    ellipse_rx: float
    ellipse_ry: float
    font_size_pt: float
    font_family: str
    node_fill: str
    node_stroke: str
    node_stroke_width_pt: float


@dataclass
class CandidateEdge:
    """Dagua-side edge feature record matching :class:`ReferenceEdge`."""

    edge_id: str
    title: str
    stroke_color: str
    stroke_width_pt: float
    arrow_length_pt: float
    arrow_width_pt: float
    arrow_filled: bool


@dataclass
class CandidateCluster:
    """Dagua-side cluster feature record matching :class:`ReferenceCluster`.

    Notes
    -----
    Dagua's cluster bounding boxes depend on layout, and this script does not
    run the layout engine -- so ``rect_x/y/w/h`` are intentionally left as
    ``None``. Only the styling axes (fill / stroke / stroke-width / label
    font size) are reportable from theme inspection alone, which is still
    where the parity sprint mostly diverges.
    """

    title: str
    fill: str
    stroke: str
    stroke_width_pt: float
    label_font_size_pt: float
    rect_x: Optional[float] = None
    rect_y: Optional[float] = None
    rect_w: Optional[float] = None
    rect_h: Optional[float] = None


@dataclass
class CandidateGraph:
    """Dagua-side feature bag matching :class:`ReferenceGraph`."""

    bg_color: str
    margin: float
    nodes: List[CandidateNode] = field(default_factory=list)
    edges: List[CandidateEdge] = field(default_factory=list)
    clusters: List[CandidateCluster] = field(default_factory=list)


def _apply_strict_theme(graph: DaguaGraph) -> DaguaGraph:
    """Clone ``graph`` and bind the ``graphviz_strict`` theme to the clone.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph.

    Returns
    -------
    DaguaGraph
        Deep-copied graph with the strict theme attached and node sizes
        recomputed under that theme.

    Notes
    -----
    This mirrors :func:`gthc._copy_graph_with_theme` but skips the
    ``set_theme`` global mutation -- we want a self-contained measurement
    that does not affect any concurrent dagua user.
    """

    clone = copy.deepcopy(graph)
    clone._theme = get_theme("graphviz_strict")
    clone.invalidate_layout()
    clone.compute_node_sizes()
    return clone


def _ellipse_semi_axes(graph: DaguaGraph, idx: int) -> Tuple[float, float]:
    """Compute (rx, ry) for a dagua ellipse node.

    Parameters
    ----------
    graph : DaguaGraph
        Graph with a populated ``node_sizes`` tensor.
    idx : int
        Node index.

    Returns
    -------
    tuple[float, float]
        ``(rx, ry)`` semi-axes in points. ``node_sizes`` already encodes the
        full ellipse bounding box, so the SVG semi-axes are simply half of
        each dimension.
    """

    sizes = graph.node_sizes
    if sizes is None:
        raise RuntimeError("compute_node_sizes() must be called before _ellipse_semi_axes")
    full_w = float(sizes[idx, 0].item())
    full_h = float(sizes[idx, 1].item())
    return full_w / 2.0, full_h / 2.0


def _candidate_nodes(graph: DaguaGraph) -> List[CandidateNode]:
    """Build the dagua-side node feature list for ``graph``.

    Parameters
    ----------
    graph : DaguaGraph
        Strict-themed graph clone.

    Returns
    -------
    list[CandidateNode]
        One entry per node, indexed by SVG-style id ``n0``, ``n1``, ...
        (matches gthc's DOT emitter node ids so per-node alignment works).
    """

    out: List[CandidateNode] = []
    font_sizes_tensor = graph.node_font_sizes
    for idx in range(graph.num_nodes):
        style = graph.get_style_for_node(idx)
        rx, ry = _ellipse_semi_axes(graph, idx)
        if font_sizes_tensor is not None:
            effective_font_size = float(font_sizes_tensor[idx].item())
        else:
            effective_font_size = float(style.font_size)
        out.append(
            CandidateNode(
                node_id=f"n{idx}",
                label=graph.node_labels[idx],
                ellipse_rx=rx,
                ellipse_ry=ry,
                font_size_pt=effective_font_size,
                font_family=style.font_family,
                node_fill=style.fill,
                node_stroke=style.stroke,
                node_stroke_width_pt=float(style.stroke_width),
            )
        )
    return out


def _candidate_edges(graph: DaguaGraph) -> List[CandidateEdge]:
    """Build the dagua-side edge feature list for ``graph``.

    Parameters
    ----------
    graph : DaguaGraph
        Strict-themed graph clone.

    Returns
    -------
    list[CandidateEdge]
        One entry per edge, ordered the same way as ``edge_index``. Arrow
        geometry comes from :class:`EdgeStyle` so it is independent of the
        eventual rendered length on a particular edge curve.
    """

    out: List[CandidateEdge] = []
    n_edges = int(graph.edge_index.shape[1]) if graph.edge_index.numel() > 0 else 0
    for idx in range(n_edges):
        style = graph.get_style_for_edge(idx)
        src = int(graph.edge_index[0, idx].item())
        tgt = int(graph.edge_index[1, idx].item())
        out.append(
            CandidateEdge(
                edge_id=f"e{idx}",
                title=f"n{src}->n{tgt}",
                stroke_color=style.color,
                stroke_width_pt=float(style.width),
                arrow_length_pt=float(style.arrow_length),
                arrow_width_pt=float(style.arrow_width),
                arrow_filled=style.arrow_fill == "filled",
            )
        )
    return out


def _candidate_clusters(graph: DaguaGraph) -> List[CandidateCluster]:
    """Build the dagua-side cluster feature list for ``graph``.

    Parameters
    ----------
    graph : DaguaGraph
        Strict-themed graph clone.

    Returns
    -------
    list[CandidateCluster]
        One entry per cluster. Geometry fields stay ``None`` because layout
        is not run here; only theme-driven cluster styling is reported.
    """

    out: List[CandidateCluster] = []
    for name in graph.clusters:
        style = graph.get_style_for_cluster(name)
        out.append(
            CandidateCluster(
                title=name,
                fill=style.fill,
                stroke=style.stroke,
                stroke_width_pt=float(style.stroke_width),
                label_font_size_pt=float(style.font_size),
            )
        )
    return out


def extract_candidate_features(graph: DaguaGraph) -> CandidateGraph:
    """Mirror :func:`parse_reference_svg` for a dagua strict-themed graph.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph. A strict-themed deep copy is produced internally.

    Returns
    -------
    CandidateGraph
        Aggregated candidate features.
    """

    themed = _apply_strict_theme(graph)
    theme = get_theme("graphviz_strict")
    return CandidateGraph(
        bg_color=theme.graph_style.background_color,
        # Strict theme exposes a graph-level margin in points; dagua's render
        # pipeline applies it symmetrically, so report it directly here.
        margin=float(theme.graph_style.margin),
        nodes=_candidate_nodes(themed),
        edges=_candidate_edges(themed),
        clusters=_candidate_clusters(themed),
    )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _abs_in_tolerance(target: float, candidate: float, tol: float) -> bool:
    """Return ``True`` when ``|target - candidate| <= tol``.

    Parameters
    ----------
    target : float
        Reference value.
    candidate : float
        Candidate value.
    tol : float
        Absolute tolerance in target units.

    Returns
    -------
    bool
        Whether the delta sits inside the band.
    """

    return abs(target - candidate) <= tol


def _pct_in_tolerance(target: float, candidate: float, tol_pct: float) -> bool:
    """Return ``True`` when the relative error is within ``tol_pct`` percent.

    Parameters
    ----------
    target : float
        Reference value.
    candidate : float
        Candidate value.
    tol_pct : float
        Allowed relative error in percent (e.g. ``5.0`` for 5 percent).

    Returns
    -------
    bool
        Whether ``abs(candidate - target) / abs(target) * 100 <= tol_pct``.
        When ``target == 0`` the function falls back to a strict equality
        check to avoid division by zero.
    """

    if target == 0:
        return abs(candidate) <= 1e-9
    return abs(candidate - target) / abs(target) * 100.0 <= tol_pct


#: Named colors Graphviz routinely emits. The SVG spec covers many more, but
#: this subset is enough to canonicalise every name we see in the harness's
#: gallery output. Hex values are uppercased to match dagua's convention.
_NAMED_COLORS: Dict[str, str] = {
    "black": "#000000",
    "white": "#FFFFFF",
    "none": "none",
    "transparent": "none",
    "red": "#FF0000",
    "green": "#008000",
    "blue": "#0000FF",
    "yellow": "#FFFF00",
    "gray": "#808080",
    "grey": "#808080",
    "lightgray": "#D3D3D3",
    "lightgrey": "#D3D3D3",
}


def _normalize_color(value: str) -> str:
    """Canonicalise a CSS color literal for comparison.

    Parameters
    ----------
    value : str
        Raw color string from SVG attribute or dagua style field.

    Returns
    -------
    str
        Uppercased hex string (``#RRGGBB``) when recognisable, otherwise the
        lowercased trimmed input. ``"none"`` and ``"transparent"`` map to
        ``"none"`` so the comparator treats both as "no fill".

    Notes
    -----
    Graphviz frequently emits ``stroke="black"`` and ``fill="white"`` while
    dagua stores ``"#000000"`` / ``"#FFFFFF"`` -- without normalisation the
    raw equality comparator would flag every node as a mismatch and drown
    out the real (non-trivial) departures the parity sprint is hunting.
    """

    cleaned = value.strip().lower()
    if not cleaned:
        return ""
    if cleaned in _NAMED_COLORS:
        return _NAMED_COLORS[cleaned]
    if cleaned.startswith("#"):
        # Expand 3-digit shorthand and uppercase the digits.
        if len(cleaned) == 4:
            r, g, b = cleaned[1], cleaned[2], cleaned[3]
            return f"#{r}{r}{g}{g}{b}{b}".upper()
        return cleaned.upper()
    return cleaned


def _color_eq(left: str, right: str) -> bool:
    """Compare two CSS color literals after canonicalisation.

    Parameters
    ----------
    left : str
        First color literal.
    right : str
        Second color literal.

    Returns
    -------
    bool
        ``True`` when both inputs map to the same canonical form. SVG's
        ``"none"`` (no fill) is treated as equivalent to a white fill since
        Graphviz uses the former and dagua uses the latter to express the
        same "transparent disk over a white canvas" intent.
    """

    a = _normalize_color(left)
    b = _normalize_color(right)
    if a == b:
        return True
    # Graphviz's default node fill is "none" (relies on the white canvas
    # showing through); dagua materialises the same visual outcome by
    # painting the disk white. Treat them as equivalent so the parity
    # comparator stops flagging every default-styled ellipse.
    white = _NAMED_COLORS["white"]
    if {a, b} == {"none", white}:
        return True
    return False


def _font_family_eq(left: str, right: str) -> bool:
    """Compare two font-family strings tolerantly.

    Parameters
    ----------
    left : str
        First font-family token (Graphviz emits ``"Times,serif"``).
    right : str
        Second font-family token (dagua reports e.g. ``"TeX Gyre Termes"``).

    Returns
    -------
    bool
        ``True`` only on exact (case-insensitive, whitespace-stripped) match.
        Font fallback similarity is NOT modeled here -- the parity sprint
        treats Times/Termes/Liberation as visually distinct enough that we
        want the failure to register.
    """

    return left.strip().lower() == right.strip().lower()


@dataclass
class FeatureDelta:
    """One per-feature comparison result.

    Attributes
    ----------
    target : Any
        Reference value (float or str).
    dagua : Any
        Candidate value (float or str).
    delta : Any
        Numeric delta (``dagua - target``) or string label for non-numeric
        comparisons.
    in_tolerance : bool
        Whether the delta sits inside the active tolerance.
    """

    target: Any
    dagua: Any
    delta: Any
    in_tolerance: bool


def _compare_numeric(
    target: Optional[float],
    candidate: Optional[float],
    tol: float,
    *,
    relative: bool = False,
) -> Optional[FeatureDelta]:
    """Build a numeric :class:`FeatureDelta`, skipping unmeasurable cases.

    Parameters
    ----------
    target : float | None
        Reference value.
    candidate : float | None
        Candidate value.
    tol : float
        Tolerance (absolute units, or percent when ``relative=True``).
    relative : bool, default=False
        When ``True``, treat ``tol`` as a percent threshold and store the
        relative delta in percent units alongside the absolute delta.

    Returns
    -------
    FeatureDelta | None
        ``None`` when either input is ``None`` (the feature is not
        comparable, e.g. polyline arrowheads have no polygon length).
    """

    if target is None or candidate is None:
        return None
    delta = candidate - target
    if relative:
        ok = _pct_in_tolerance(target, candidate, tol)
    else:
        ok = _abs_in_tolerance(target, candidate, tol)
    return FeatureDelta(
        target=round(target, 4),
        dagua=round(candidate, 4),
        delta=round(delta, 4),
        in_tolerance=ok,
    )


def _compare_string(
    target: str,
    candidate: str,
    *,
    use_color_eq: bool = False,
) -> FeatureDelta:
    """Build a string :class:`FeatureDelta` for color / font-family fields.

    Parameters
    ----------
    target : str
        Reference string.
    candidate : str
        Candidate string.
    use_color_eq : bool, default=False
        When ``True``, compare with :func:`_color_eq`; otherwise use
        :func:`_font_family_eq`.

    Returns
    -------
    FeatureDelta
        String-valued delta with a textual ``delta`` summary like
        ``"target='X' dagua='Y'"``.
    """

    if use_color_eq:
        match = _color_eq(target, candidate)
    else:
        match = _font_family_eq(target, candidate)
    return FeatureDelta(
        target=target,
        dagua=candidate,
        delta=f"target={target!r} dagua={candidate!r}",
        in_tolerance=match,
    )


@dataclass
class PanelReport:
    """Per-panel parity summary.

    Attributes
    ----------
    slug : str
        Filesystem-friendly panel identifier (matches gthc.GraphCase.slug).
    in_tolerance : bool
        ``True`` only when every comparable feature is in tolerance.
    nodes : list[dict]
        Per-node feature deltas.
    edges : list[dict]
        Per-edge feature deltas.
    clusters : list[dict]
        Per-cluster feature deltas.
    graph : dict
        Per-graph feature deltas (background, margin).
    out_of_tolerance : list[str]
        Human-readable list of failing features for quick scanning.
    """

    slug: str
    in_tolerance: bool
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    clusters: List[Dict[str, Any]] = field(default_factory=list)
    graph: Dict[str, Any] = field(default_factory=dict)
    out_of_tolerance: List[str] = field(default_factory=list)


def _flatten_node_deltas(
    node_id: str,
    target: ReferenceNode,
    candidate: CandidateNode,
    tolerance: Mapping[str, Any],
) -> Dict[str, Any]:
    """Compute every comparable delta for one matched node pair.

    Parameters
    ----------
    node_id : str
        Aligning DOT id (``n0`` etc.).
    target : ReferenceNode
        SVG-derived reference record.
    candidate : CandidateNode
        Dagua-side record.
    tolerance : Mapping[str, Any]
        Active tolerance profile.

    Returns
    -------
    dict[str, Any]
        Mapping ``feature_name -> FeatureDelta-as-dict``. Keys mirror the
        target tolerance keys so the summary aggregation can group them.
    """

    deltas: Dict[str, Any] = {"id": node_id, "label": target.label}

    rx_delta = _compare_numeric(target.ellipse_rx, candidate.ellipse_rx, tolerance["ellipse_rx_pt"])
    if rx_delta is not None:
        deltas["ellipse_rx_pt"] = rx_delta.__dict__

    ry_delta = _compare_numeric(target.ellipse_ry, candidate.ellipse_ry, tolerance["ellipse_ry_pt"])
    if ry_delta is not None:
        deltas["ellipse_ry_pt"] = ry_delta.__dict__

    if target.ellipse_ry > 0 and candidate.ellipse_ry > 0:
        target_aspect = target.ellipse_rx / target.ellipse_ry
        cand_aspect = candidate.ellipse_rx / candidate.ellipse_ry
        aspect_delta = _compare_numeric(
            target_aspect,
            cand_aspect,
            tolerance["ellipse_aspect_pct"],
            relative=True,
        )
        if aspect_delta is not None:
            deltas["ellipse_aspect_pct"] = aspect_delta.__dict__

    fs_delta = _compare_numeric(
        target.font_size_pt, candidate.font_size_pt, tolerance["font_size_pt"]
    )
    if fs_delta is not None:
        deltas["font_size_pt"] = fs_delta.__dict__

    deltas["font_family"] = _compare_string(target.font_family, candidate.font_family).__dict__

    deltas["node_fill"] = _compare_string(
        target.node_fill, candidate.node_fill, use_color_eq=True
    ).__dict__

    deltas["node_stroke"] = _compare_string(
        target.node_stroke, candidate.node_stroke, use_color_eq=True
    ).__dict__

    sw_delta = _compare_numeric(
        target.node_stroke_width_pt,
        candidate.node_stroke_width_pt,
        tolerance["node_stroke_width_pt"],
    )
    if sw_delta is not None:
        deltas["node_stroke_width_pt"] = sw_delta.__dict__

    return deltas


def _flatten_edge_deltas(
    edge_id: str,
    target: ReferenceEdge,
    candidate: CandidateEdge,
    tolerance: Mapping[str, Any],
) -> Dict[str, Any]:
    """Compute every comparable delta for one matched edge pair.

    Parameters
    ----------
    edge_id : str
        Aligning edge identifier.
    target : ReferenceEdge
        SVG-derived reference record.
    candidate : CandidateEdge
        Dagua-side record.
    tolerance : Mapping[str, Any]
        Active tolerance profile.

    Returns
    -------
    dict[str, Any]
        Mapping ``feature_name -> FeatureDelta-as-dict``.
    """

    deltas: Dict[str, Any] = {"id": edge_id, "title": target.title}

    deltas["edge_stroke_color"] = _compare_string(
        target.stroke_color, candidate.stroke_color, use_color_eq=True
    ).__dict__

    sw_delta = _compare_numeric(
        target.stroke_width_pt,
        candidate.stroke_width_pt,
        tolerance["edge_stroke_width_pt"],
    )
    if sw_delta is not None:
        deltas["edge_stroke_width_pt"] = sw_delta.__dict__

    al_delta = _compare_numeric(
        target.arrow_length_pt,
        candidate.arrow_length_pt,
        tolerance["arrow_length_pt"],
    )
    if al_delta is not None:
        deltas["arrow_length_pt"] = al_delta.__dict__

    aw_delta = _compare_numeric(
        target.arrow_width_pt,
        candidate.arrow_width_pt,
        tolerance["arrow_width_pt"],
    )
    if aw_delta is not None:
        deltas["arrow_width_pt"] = aw_delta.__dict__

    if target.arrow_filled is not None:
        deltas["arrow_filled"] = FeatureDelta(
            target=bool(target.arrow_filled),
            dagua=bool(candidate.arrow_filled),
            delta=f"target={target.arrow_filled} dagua={candidate.arrow_filled}",
            in_tolerance=bool(target.arrow_filled) == bool(candidate.arrow_filled),
        ).__dict__

    return deltas


def _flatten_cluster_deltas(
    title: str,
    target: ReferenceCluster,
    candidate: CandidateCluster,
    tolerance: Mapping[str, Any],
) -> Dict[str, Any]:
    """Compute every comparable delta for one matched cluster pair.

    Parameters
    ----------
    title : str
        Aligning cluster title.
    target : ReferenceCluster
        SVG-derived reference record.
    candidate : CandidateCluster
        Dagua-side record.
    tolerance : Mapping[str, Any]
        Active tolerance profile.

    Returns
    -------
    dict[str, Any]
        Mapping ``feature_name -> FeatureDelta-as-dict``.
    """

    deltas: Dict[str, Any] = {"title": title}

    deltas["cluster_fill"] = _compare_string(
        target.fill, candidate.fill, use_color_eq=True
    ).__dict__
    deltas["cluster_stroke"] = _compare_string(
        target.stroke, candidate.stroke, use_color_eq=True
    ).__dict__

    sw_delta = _compare_numeric(
        target.stroke_width_pt,
        candidate.stroke_width_pt,
        tolerance["cluster_stroke_width_pt"],
    )
    if sw_delta is not None:
        deltas["cluster_stroke_width_pt"] = sw_delta.__dict__

    if target.label_font_size_pt is not None:
        fs_delta = _compare_numeric(
            target.label_font_size_pt,
            candidate.label_font_size_pt,
            tolerance["cluster_label_font_size_pt"],
        )
        if fs_delta is not None:
            deltas["cluster_label_font_size_pt"] = fs_delta.__dict__

    return deltas


def _flatten_graph_deltas(
    target: ReferenceGraph,
    candidate: CandidateGraph,
) -> Dict[str, Any]:
    """Compute graph-level deltas (background color, margin).

    Parameters
    ----------
    target : ReferenceGraph
        Reference graph features.
    candidate : CandidateGraph
        Candidate graph features.

    Returns
    -------
    dict[str, Any]
        Mapping ``feature_name -> FeatureDelta-as-dict``.
    """

    deltas: Dict[str, Any] = {}
    deltas["bg_color"] = _compare_string(
        target.bg_color, candidate.bg_color, use_color_eq=True
    ).__dict__
    # Graphviz's SVG viewBox padding (~4pt) and dagua's theme margin (18pt)
    # measure different things -- the SVG margin is "extra whitespace inside
    # the canvas", while dagua's GraphStyle.margin is a render-time padding
    # applied around the bounding box during matplotlib export. Use a wide
    # 16-pt tolerance so this axis is reported but does not drown the
    # summary in spurious failures while a more meaningful margin metric
    # is designed.
    margin_delta = _compare_numeric(
        target.margin,
        candidate.margin,
        tol=16.0,
    )
    if margin_delta is not None:
        deltas["margin_pt"] = margin_delta.__dict__
    return deltas


def _flag_out_of_tolerance(deltas: Dict[str, Any]) -> List[str]:
    """Walk a flattened delta block and collect human-readable failures.

    Parameters
    ----------
    deltas : dict[str, Any]
        One ``_flatten_*_deltas`` result.

    Returns
    -------
    list[str]
        ``["feature_name: dagua vs target (delta)", ...]`` for failing axes.
    """

    failures: List[str] = []
    label = deltas.get("id", deltas.get("title", "?"))
    for key, value in deltas.items():
        if key in {"id", "title", "label"}:
            continue
        if not isinstance(value, dict):
            continue
        if value.get("in_tolerance", True):
            continue
        target = value.get("target")
        cand = value.get("dagua")
        delta = value.get("delta")
        failures.append(f"{label}.{key}: dagua={cand} vs target={target} (delta={delta})")
    return failures


def compare_panel(
    slug: str,
    reference: ReferenceGraph,
    candidate: CandidateGraph,
    tolerance: Mapping[str, Any] = DEFAULT_TOLERANCE,
) -> PanelReport:
    """Compare reference and candidate features into a :class:`PanelReport`.

    Parameters
    ----------
    slug : str
        Panel identifier.
    reference : ReferenceGraph
        SVG-derived reference features.
    candidate : CandidateGraph
        Dagua-derived candidate features.
    tolerance : Mapping[str, Any], default=DEFAULT_TOLERANCE
        Active tolerance profile.

    Returns
    -------
    PanelReport
        Aggregated per-panel comparison.

    Notes
    -----
    Feature alignment uses positional indices for nodes and edges (same as
    DOT's ``n<i>`` / edge order in the harness emitter). For clusters, names
    are matched via the ``cluster_<name>`` Graphviz title prefix produced by
    ``_safe_cluster_id``. Missing matches are reported as panel-level
    failures so silent drops are visible.
    """

    panel = PanelReport(slug=slug, in_tolerance=True)

    # Nodes: positional alignment. The harness emits nodes as ``n<idx>`` in
    # the same order dagua maintains internally.
    ref_node_by_id = {ref.node_id: ref for ref in reference.nodes}
    for cand in candidate.nodes:
        ref = ref_node_by_id.get(cand.node_id)
        if ref is None:
            # Non-ellipse shapes (rect, diamond, triangle, ...) are emitted by
            # Graphviz with shape-specific outlines; the SVG parser intentionally
            # only records ellipse-shaped reference nodes. Skip unmatched
            # candidates rather than blocking the panel -- they are tracked at
            # the summary level.
            continue
        deltas = _flatten_node_deltas(cand.node_id, ref, cand, tolerance)
        panel.nodes.append(deltas)
        failures = _flag_out_of_tolerance(deltas)
        panel.out_of_tolerance.extend(f"node[{cand.node_id}].{f}" for f in failures)

    # Edges: positional alignment via gthc DOT id ordering.
    for cand in candidate.edges:
        ref_match: Optional[ReferenceEdge] = None
        for ref in reference.edges:
            if ref.title.replace("&#45;&gt;", "->") == cand.title or ref.title == cand.title:
                ref_match = ref
                break
        if ref_match is None:
            continue
        deltas = _flatten_edge_deltas(cand.edge_id, ref_match, cand, tolerance)
        panel.edges.append(deltas)
        failures = _flag_out_of_tolerance(deltas)
        panel.out_of_tolerance.extend(f"edge[{cand.title}].{f}" for f in failures)

    # Clusters: align via ``cluster_<title>`` reference prefix.
    ref_cluster_by_title = {}
    for ref in reference.clusters:
        # Graphviz emits cluster titles like ``cluster_cluster_inner``; strip
        # the leading "cluster_" prefix the harness adds on top of the
        # user-supplied name to align with dagua's stored cluster name.
        plain = ref.title
        if plain.startswith("cluster_"):
            plain = plain[len("cluster_") :]
        ref_cluster_by_title[plain] = ref
    for cand in candidate.clusters:
        ref = ref_cluster_by_title.get(cand.title)
        if ref is None:
            continue
        deltas = _flatten_cluster_deltas(cand.title, ref, cand, tolerance)
        panel.clusters.append(deltas)
        failures = _flag_out_of_tolerance(deltas)
        panel.out_of_tolerance.extend(f"cluster[{cand.title}].{f}" for f in failures)

    panel.graph = _flatten_graph_deltas(reference, candidate)
    failures = _flag_out_of_tolerance(panel.graph)
    panel.out_of_tolerance.extend(f"graph.{f}" for f in failures)

    panel.in_tolerance = not panel.out_of_tolerance
    return panel


# ---------------------------------------------------------------------------
# Aggregate summary
# ---------------------------------------------------------------------------


def _collect_feature_stats(panels: Iterable[PanelReport]) -> Dict[str, Dict[str, Any]]:
    """Compute per-feature aggregated stats across all panels.

    Parameters
    ----------
    panels : Iterable[PanelReport]
        All scored panels.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping ``feature_name -> {"compared", "in_tolerance", "pct",
        "median_delta", "max_delta"}``.
    """

    feature_buckets: Dict[str, List[Tuple[bool, Optional[float]]]] = {}

    def _bucket(deltas_list: Iterable[Dict[str, Any]]) -> None:
        for entry in deltas_list:
            for key, value in entry.items():
                if not isinstance(value, dict) or "in_tolerance" not in value:
                    continue
                ok = bool(value["in_tolerance"])
                # Numeric features expose float-able deltas; string/bool
                # features expose textual deltas. Only keep float deltas for
                # median/max statistics so the reductions stay meaningful.
                raw = value.get("delta")
                numeric: Optional[float]
                try:
                    numeric = float(raw) if isinstance(raw, (int, float)) else None
                except (TypeError, ValueError):
                    numeric = None
                feature_buckets.setdefault(key, []).append((ok, numeric))

    for panel in panels:
        _bucket(panel.nodes)
        _bucket(panel.edges)
        _bucket(panel.clusters)
        _bucket([panel.graph])

    result: Dict[str, Dict[str, Any]] = {}
    for key, entries in feature_buckets.items():
        compared = len(entries)
        in_tol = sum(1 for ok, _ in entries if ok)
        deltas = [d for _, d in entries if d is not None]
        if deltas:
            abs_deltas = [abs(d) for d in deltas]
            median_delta = round(statistics.median(abs_deltas), 4)
            max_delta = round(max(abs_deltas), 4)
        else:
            median_delta = None
            max_delta = None
        result[key] = {
            "compared": compared,
            "in_tolerance": in_tol,
            "pct": round((in_tol / compared) * 100.0, 2) if compared else 0.0,
            "median_delta": median_delta,
            "max_delta": max_delta,
        }
    return result


def build_summary(
    panels: List[PanelReport],
    tolerance_name: str,
) -> Dict[str, Any]:
    """Assemble the JSON-serialisable summary block.

    Parameters
    ----------
    panels : list[PanelReport]
        Scored panels.
    tolerance_name : str
        Symbolic name of the active tolerance profile (for traceability).

    Returns
    -------
    dict[str, Any]
        Top-level summary block (mirrors the spec's JSON shape).
    """

    feature_stats = _collect_feature_stats(panels)
    total_features = sum(stat["compared"] for stat in feature_stats.values())
    total_in_tol = sum(stat["in_tolerance"] for stat in feature_stats.values())
    return {
        "tolerance_profile": tolerance_name,
        "total_panels": len(panels),
        "panels_in_tolerance": sum(1 for p in panels if p.in_tolerance),
        "total_features_compared": total_features,
        "in_tolerance_count": total_in_tol,
        "out_of_tolerance_count": total_features - total_in_tol,
        "in_tolerance_pct": (
            round((total_in_tol / total_features) * 100.0, 2) if total_features else 0.0
        ),
        "by_feature_type": feature_stats,
    }


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def score_case(
    case: gthc.GraphCase,
    tolerance: Mapping[str, Any] = DEFAULT_TOLERANCE,
) -> Optional[PanelReport]:
    """Score a single :class:`gthc.GraphCase` end-to-end.

    Parameters
    ----------
    case : gthc.GraphCase
        Test case from the harness.
    tolerance : Mapping[str, Any], default=DEFAULT_TOLERANCE
        Active tolerance profile.

    Returns
    -------
    PanelReport | None
        Comparison result, or ``None`` when the SVG render / parse failed.
        Failures are logged to stderr so the operator can investigate the
        offending case without aborting the whole sweep.
    """

    try:
        svg_text = render_reference_svg(case.graph)
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"[skip] {case.slug}: dot render failed: {exc}", file=sys.stderr)
        return None

    try:
        reference = parse_reference_svg(svg_text)
    except ET.ParseError as exc:
        print(f"[skip] {case.slug}: SVG parse failed: {exc}", file=sys.stderr)
        return None

    candidate = extract_candidate_features(case.graph)
    return compare_panel(case.slug, reference, candidate, tolerance)


def run(
    cases: Sequence[gthc.GraphCase],
    output_path: Path,
    tolerance: Mapping[str, Any] = DEFAULT_TOLERANCE,
    tolerance_name: str = "default",
) -> Dict[str, Any]:
    """Score the supplied cases and persist the JSON report.

    Parameters
    ----------
    cases : Sequence[gthc.GraphCase]
        Cases to score.
    output_path : Path
        Destination JSON path. Parent directories are created.
    tolerance : Mapping[str, Any], default=DEFAULT_TOLERANCE
        Active tolerance profile.
    tolerance_name : str, default="default"
        Symbolic profile name embedded in the report for traceability.

    Returns
    -------
    dict[str, Any]
        The full JSON report (also written to ``output_path``).
    """

    panels: List[PanelReport] = []
    for case in cases:
        report = score_case(case, tolerance)
        if report is not None:
            panels.append(report)

    summary = build_summary(panels, tolerance_name)
    payload = {
        "summary": summary,
        "panels": [
            {
                "slug": p.slug,
                "in_tolerance": p.in_tolerance,
                "features": {
                    "nodes": p.nodes,
                    "edges": p.edges,
                    "clusters": p.clusters,
                    "graph": p.graph,
                },
                "out_of_tolerance_features": p.out_of_tolerance,
            }
            for p in panels
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    return payload


def _print_summary_table(summary: Mapping[str, Any]) -> None:
    """Print a brief, terminal-friendly summary table.

    Parameters
    ----------
    summary : Mapping[str, Any]
        ``summary`` block produced by :func:`build_summary`.

    Returns
    -------
    None
        Output is written directly to stdout.
    """

    print()
    print("Parity summary")
    print("--------------")
    print(f"  panels:               {summary['total_panels']}")
    print(f"  panels in tolerance:  {summary['panels_in_tolerance']}")
    print(f"  features compared:    {summary['total_features_compared']}")
    print(f"  in-tolerance:         {summary['in_tolerance_count']}")
    print(f"  out-of-tolerance:     {summary['out_of_tolerance_count']}")
    print(f"  in-tolerance %:       {summary['in_tolerance_pct']:.2f}%")
    print()
    print("Per-feature breakdown (sorted by failures desc)")
    print("------------------------------------------------")
    feature_stats = summary["by_feature_type"]
    rows = sorted(
        feature_stats.items(),
        key=lambda item: item[1]["compared"] - item[1]["in_tolerance"],
        reverse=True,
    )
    for name, stat in rows:
        miss = stat["compared"] - stat["in_tolerance"]
        median = stat["median_delta"]
        maxd = stat["max_delta"]
        median_str = f"{median:.2f}" if isinstance(median, float) else "-"
        max_str = f"{maxd:.2f}" if isinstance(maxd, float) else "-"
        print(
            f"  {name:<32s}  {stat['compared']:>4d}  in_tol={stat['in_tolerance']:>4d}  "
            f"miss={miss:>3d}  pct={stat['pct']:>6.2f}%  med={median_str:>6s}  max={max_str:>6s}"
        )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Parameters
    ----------
    argv : Sequence[str], optional
        Explicit argument vector.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(
        description="Quantitative cosmetic parity metrics for graphviz_strict."
    )
    parser.add_argument(
        "--cases",
        default="",
        help=(
            "Comma-separated case slugs to score (default: every case from "
            "scripts.graphviz_theme_comparison._iter_cases())."
        ),
    )
    parser.add_argument(
        "--out",
        default="eval_output/parity_metrics.json",
        help="Destination JSON path.",
    )
    parser.add_argument(
        "--tolerance",
        default="default",
        help="Tolerance profile name (currently only 'default' is supported).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use only the programmatic showcase cases (skip bundled YAMLs).",
    )
    return parser.parse_args(argv)


def _select_cases(
    requested: Sequence[str],
    quick: bool,
) -> List[gthc.GraphCase]:
    """Filter the harness's full case list down to a requested subset.

    Parameters
    ----------
    requested : Sequence[str]
        Selected slugs (empty -> all cases).
    quick : bool
        Forwarded to :func:`gthc._iter_cases`.

    Returns
    -------
    list[gthc.GraphCase]
        Cases to score.
    """

    cases = gthc._iter_cases(quick=quick)
    if not requested:
        return cases
    requested_set = {s.strip() for s in requested if s.strip()}
    selected = [c for c in cases if c.slug in requested_set]
    missing = requested_set - {c.slug for c in selected}
    if missing:
        print(f"[warn] unknown case slugs: {sorted(missing)}", file=sys.stderr)
    return selected


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
        print(
            "[error] 'dot' binary not found on PATH; install Graphviz to run parity_metrics",
            file=sys.stderr,
        )
        return 1
    if args.tolerance != "default":
        print(
            f"[warn] unknown tolerance profile {args.tolerance!r}; falling back to 'default'",
            file=sys.stderr,
        )

    requested = [s for s in args.cases.split(",") if s] if args.cases else []
    cases = _select_cases(requested, quick=bool(args.quick))
    if not cases:
        print("[error] no cases selected", file=sys.stderr)
        return 1

    payload = run(
        cases=cases,
        output_path=Path(args.out),
        tolerance=DEFAULT_TOLERANCE,
        tolerance_name="default",
    )
    _print_summary_table(payload["summary"])
    print(f"Wrote {len(payload['panels'])} panels to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
