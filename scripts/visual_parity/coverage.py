"""Build the visual parity v2 coverage matrix.

The coverage denominator is external: pinned Graphviz documentation snapshots
and tool survey facts define the rows, while Dagua introspection only fills in
support status. This module intentionally keeps generation deterministic so the
matrix can be regenerated and reviewed as source-controlled data.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.error import URLError
from urllib.request import urlopen

from dagua.render.edges.arrowheads import (
    ARROWHEAD_ALIASES,
    ARROWHEAD_REGISTRY,
    available_arrowheads,
)
from dagua.styles import NODE_SHAPE_NAMES, ClusterStyle, EdgeStyle, GraphStyle, NodeStyle
from scripts.visual_parity.io import write_coverage_matrix
from scripts.visual_parity.types import COVERAGE_MATRIX_SCHEMA_VERSION

RESEARCH_DIR = Path(".project-context/research/sprint_visual_parity_v2")
REFERENCE_DIR = RESEARCH_DIR / "reference_specs"
COVERAGE_PATH = RESEARCH_DIR / "coverage_matrix.json"
TRIAGE_PATH = REFERENCE_DIR / "gv_attr_triage.json"
GRAPHVIZ_VERSION = "7.0.5"

SNAPSHOTS: Tuple[Tuple[str, str, str], ...] = (
    ("gv-attrs", "https://graphviz.org/doc/info/attrs.html", "gv_attrs.html"),
    ("gv-arrows", "https://graphviz.org/doc/info/arrows.html", "gv_arrows.html"),
    ("gv-shapes", "https://graphviz.org/doc/info/shapes.html", "gv_shapes.html"),
    ("gv-colors", "https://graphviz.org/doc/info/colors.html", "gv_colors.html"),
)

GRAPHVIZ_ARROW_PRIMITIVES: Tuple[str, ...] = (
    "box",
    "crow",
    "curve",
    "diamond",
    "dot",
    "icurve",
    "inv",
    "none",
    "normal",
    "tee",
    "vee",
)

GRAPHVIZ_SHAPES: Tuple[str, ...] = (
    "box",
    "polygon",
    "ellipse",
    "oval",
    "circle",
    "point",
    "egg",
    "triangle",
    "plaintext",
    "plain",
    "diamond",
    "trapezium",
    "parallelogram",
    "house",
    "pentagon",
    "hexagon",
    "septagon",
    "octagon",
    "doublecircle",
    "doubleoctagon",
    "tripleoctagon",
    "invtriangle",
    "invtrapezium",
    "invhouse",
    "Mdiamond",
    "Msquare",
    "Mcircle",
    "rect",
    "rectangle",
    "square",
    "star",
    "none",
    "underline",
    "cylinder",
    "note",
    "tab",
    "folder",
    "box3d",
    "component",
    "promoter",
    "cds",
    "terminator",
    "utr",
    "primersite",
    "restrictionsite",
    "fivepoverhang",
    "threepoverhang",
    "noverhang",
    "assembly",
    "signature",
    "insulator",
    "ribosite",
    "rnastab",
    "proteasesite",
    "proteinstab",
    "rpromoter",
    "rarrow",
    "larrow",
    "lpromoter",
)

COSMETIC_ATTRS: Tuple[str, ...] = (
    "area",
    "arrowhead",
    "arrowsize",
    "arrowtail",
    "bgcolor",
    "color",
    "colorscheme",
    "comment",
    "decorate",
    "dir",
    "edgehref",
    "edgetarget",
    "edgetooltip",
    "edgeURL",
    "fillcolor",
    "fixedsize",
    "fontcolor",
    "fontname",
    "fontnames",
    "fontpath",
    "fontsize",
    "gradientangle",
    "head_lp",
    "headclip",
    "headhref",
    "headlabel",
    "headport",
    "headtarget",
    "headtooltip",
    "headURL",
    "height",
    "href",
    "id",
    "image",
    "imagepath",
    "imagepos",
    "imagescale",
    "label",
    "labelangle",
    "labeldistance",
    "labelfloat",
    "labelfontcolor",
    "labelfontname",
    "labelfontsize",
    "labelhref",
    "labeljust",
    "labelloc",
    "labeltarget",
    "labeltooltip",
    "labelURL",
    "layer",
    "margin",
    "nojustify",
    "ordering",
    "orientation",
    "penwidth",
    "peripheries",
    "pin",
    "pos",
    "rects",
    "regular",
    "root",
    "samplepoints",
    "shape",
    "shapefile",
    "showboxes",
    "sides",
    "skew",
    "sortv",
    "style",
    "stylesheet",
    "tail_lp",
    "tailclip",
    "tailhref",
    "taillabel",
    "tailport",
    "tailtarget",
    "tailtooltip",
    "tailURL",
    "target",
    "tooltip",
    "URL",
    "vertices",
    "width",
    "xlabel",
    "xlp",
    "z",
)

LAYOUT_ATTRS: Tuple[str, ...] = (
    "Damping",
    "K",
    "URL",
    "bb",
    "center",
    "clusterrank",
    "concentrate",
    "constraint",
    "dim",
    "dimen",
    "diredgeconstraints",
    "dpi",
    "epsilon",
    "esep",
    "forcelabels",
    "layout",
    "len",
    "levels",
    "levelsgap",
    "lhead",
    "lp",
    "ltail",
    "maxiter",
    "mclimit",
    "mindist",
    "minlen",
    "mode",
    "model",
    "newrank",
    "nodesep",
    "normalize",
    "notranslate",
    "nslimit",
    "nslimit1",
    "overlap",
    "overlap_scaling",
    "pack",
    "packmode",
    "pad",
    "page",
    "pagedir",
    "quadtree",
    "quantum",
    "rank",
    "rankdir",
    "ranksep",
    "ratio",
    "repulsiveforce",
    "resolution",
    "rotate",
    "scale",
    "searchsize",
    "sep",
    "size",
    "smoothing",
    "splines",
    "start",
    "truecolor",
    "viewport",
    "voro_margin",
    "weight",
)

IO_META_ATTRS: Tuple[str, ...] = (
    "_background",
    "charset",
    "class",
    "colorscheme",
    "comment",
    "dpi",
    "fontpath",
    "id",
    "imagepath",
    "inputscale",
    "layerlistsep",
    "layers",
    "layerselect",
    "layersep",
    "outputorder",
    "stylesheet",
)

FIELD_MAP: Dict[Tuple[str, str], str] = {
    ("node", "shape"): "NodeStyle.shape",
    ("node", "fillcolor"): "NodeStyle.fill",
    ("node", "color"): "NodeStyle.stroke",
    ("node", "penwidth"): "NodeStyle.stroke_width",
    ("node", "fontname"): "NodeStyle.font_family",
    ("node", "fontsize"): "NodeStyle.font_size",
    ("node", "fontcolor"): "NodeStyle.font_color",
    ("node", "style"): "NodeStyle.fill_pattern",
    ("edge", "arrowhead"): "EdgeStyle.arrow",
    ("edge", "arrowtail"): "EdgeStyle.tail_arrow",
    ("edge", "arrowsize"): "EdgeStyle.arrowsize",
    ("edge", "color"): "EdgeStyle.color",
    ("edge", "penwidth"): "EdgeStyle.width",
    ("edge", "style"): "EdgeStyle.style",
    ("edge", "fontname"): "EdgeStyle.label_font_family",
    ("edge", "fontsize"): "EdgeStyle.label_font_size",
    ("edge", "fontcolor"): "EdgeStyle.label_font_color",
    ("edge", "label"): "EdgeStyle.label_position",
    ("graph", "bgcolor"): "GraphStyle.background_color",
    ("graph", "margin"): "GraphStyle.margin_inches",
    ("graph", "label"): "GraphStyle.title",
    ("cluster", "color"): "ClusterStyle.stroke",
    ("cluster", "fillcolor"): "ClusterStyle.fill",
    ("cluster", "penwidth"): "ClusterStyle.stroke_width",
    ("cluster", "fontname"): "ClusterStyle.font_family",
    ("cluster", "fontsize"): "ClusterStyle.font_size",
    ("cluster", "fontcolor"): "ClusterStyle.font_color",
}


def _utc_now() -> str:
    """Return the current UTC timestamp in stable ISO-8601 form.

    Returns
    -------
    str
        UTC timestamp with a ``Z`` suffix.
    """

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _download_text(url: str, timeout_seconds: float = 12.0) -> Optional[str]:
    """Download text from a URL.

    Parameters
    ----------
    url
        URL to fetch.
    timeout_seconds
        Network timeout in seconds.

    Returns
    -------
    Optional[str]
        Downloaded text, or ``None`` when the request fails.
    """

    try:
        with urlopen(url, timeout=timeout_seconds) as response:
            return _sanitize_snapshot_text(response.read().decode("utf-8", errors="replace"))
    except (OSError, URLError):
        return None


def _sanitize_snapshot_text(text: str) -> str:
    """Remove embedded binary data from downloaded documentation snapshots.

    Parameters
    ----------
    text
        Downloaded HTML text.

    Returns
    -------
    str
        HTML text with data URIs replaced by reviewable placeholders.
    """

    without_data_uris = re.sub(r"data:[^\"')\\s]+", "data:removed-for-source-control", text)
    return re.sub(r"\s+integrity=\"[^\"]+\"", "", without_data_uris)


def _fallback_snapshot(snapshot_id: str, source_url: str) -> str:
    """Return a deterministic offline snapshot for one Graphviz doc.

    Parameters
    ----------
    snapshot_id
        Source identifier.
    source_url
        Upstream source URL recorded in the snapshot.

    Returns
    -------
    str
        Synthetic HTML snapshot preserving the denominator values.
    """

    if snapshot_id == "gv-arrows":
        values = "\n".join(f"<li>{value}</li>" for value in GRAPHVIZ_ARROW_PRIMITIVES)
    elif snapshot_id == "gv-shapes":
        values = "\n".join(f"<li>{value}</li>" for value in GRAPHVIZ_SHAPES)
    elif snapshot_id == "gv-attrs":
        values = "\n".join(
            f"<tr><td>{value}</td></tr>"
            for value in sorted(set(COSMETIC_ATTRS + LAYOUT_ATTRS + IO_META_ATTRS))
        )
    else:
        values = "<p>Graphviz color names are represented by the committed color cells.</p>"
    return (
        "<!doctype html>\n"
        f"<html><head><title>{snapshot_id} offline Graphviz {GRAPHVIZ_VERSION}</title></head>\n"
        f'<body data-provenance="offline-synthetic" data-source="{source_url}">\n'
        f"{values}\n"
        "</body></html>\n"
    )


def ensure_reference_specs(force: bool = False) -> List[Dict[str, str]]:
    """Ensure pinned reference snapshots and attr triage exist.

    Parameters
    ----------
    force
        Re-acquire snapshots even when files are present.

    Returns
    -------
    List[Dict[str, str]]
        Source snapshot provenance records for coverage_matrix.json.
    """

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    retrieved_at = _utc_now()
    records: List[Dict[str, str]] = []
    for snapshot_id, source_url, filename in SNAPSHOTS:
        path = REFERENCE_DIR / filename
        provenance = "downloaded"
        if force or not path.exists():
            text = _download_text(source_url)
            if text is None:
                text = _fallback_snapshot(snapshot_id, source_url)
                provenance = "offline_synthetic_from_pinned_denominator"
            path.write_text(text, encoding="utf-8")
        elif "offline-synthetic" in path.read_text(encoding="utf-8", errors="ignore")[:400]:
            provenance = "offline_synthetic_from_pinned_denominator"
        records.append(
            {
                "id": snapshot_id,
                "source_url": source_url,
                "source_kind": f"official_docs:{provenance}",
                "retrieved_at": retrieved_at,
                "version": GRAPHVIZ_VERSION,
                "cache_path": f"reference_specs/{filename}",
            }
        )
    write_attr_triage()
    write_arrow_grammar()
    return records


def write_attr_triage() -> None:
    """Write the committed Graphviz attribute triage table.

    Returns
    -------
    None
        The function writes ``gv_attr_triage.json``.
    """

    all_attrs = sorted(set(COSMETIC_ATTRS + LAYOUT_ATTRS + IO_META_ATTRS))
    rows = []
    for attr in all_attrs:
        if attr in IO_META_ATTRS:
            tag = "io-meta"
        elif attr in LAYOUT_ATTRS:
            tag = "layout"
        else:
            tag = "cosmetic"
        rows.append({"attribute": attr, "tag": tag, "review": "lane_c_seed"})
    payload = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "pin": f"graphviz {GRAPHVIZ_VERSION}",
        "rows": rows,
    }
    TRIAGE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_arrow_grammar() -> None:
    """Write generated Graphviz arrow grammar values for review.

    Returns
    -------
    None
        The function writes ``gv_arrow_grammar.json``.
    """

    payload = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "pin": f"graphviz {GRAPHVIZ_VERSION}",
        "primitives": list(GRAPHVIZ_ARROW_PRIMITIVES),
        "modifier_expansions": generate_graphviz_modifier_expansions(),
    }
    (REFERENCE_DIR / "gv_arrow_grammar.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def generate_graphviz_modifier_expansions() -> List[str]:
    """Generate the 42 documented Graphviz modifier shapes from grammar rules.

    Returns
    -------
    List[str]
        Generated values. The list is not hand-entered; it is composed from the
        documented primitive set and modifier grammar.
    """

    bases = [name for name in GRAPHVIZ_ARROW_PRIMITIVES if name != "none"]
    values = list(GRAPHVIZ_ARROW_PRIMITIVES)
    values.extend(f"o{name}" for name in bases)
    values.extend(f"l{name}" for name in bases)
    values.extend(f"r{name}" for name in bases)
    values.append("open")
    if len(values) != 42:
        raise RuntimeError(f"Graphviz arrow modifier grammar generated {len(values)} values")
    return values


def _style_fields() -> Dict[str, set[str]]:
    """Return style dataclass fields keyed by dataclass name.

    Returns
    -------
    Dict[str, set[str]]
        Field-name sets for NodeStyle, EdgeStyle, GraphStyle, and ClusterStyle.
    """

    return {
        "NodeStyle": {field.name for field in dataclasses.fields(NodeStyle)},
        "EdgeStyle": {field.name for field in dataclasses.fields(EdgeStyle)},
        "GraphStyle": {field.name for field in dataclasses.fields(GraphStyle)},
        "ClusterStyle": {field.name for field in dataclasses.fields(ClusterStyle)},
    }


def _dagua_support(
    object_kind: str,
    attribute: str,
    value: str,
    value_group: str,
) -> Tuple[str, str]:
    """Compute Dagua support for a coverage cell by introspection.

    Parameters
    ----------
    object_kind
        Graphviz object kind.
    attribute
        Graphviz attribute.
    value
        Graphviz value.
    value_group
        Coverage value group.

    Returns
    -------
    Tuple[str, str]
        ``(support_status, dagua_field)``.
    """

    dagua_field = FIELD_MAP.get((object_kind, attribute), "")
    if value_group.startswith("arrow_"):
        supported = value in set(available_arrowheads()) or value in ARROWHEAD_REGISTRY
        return ("supported" if supported else "missing", "EdgeStyle.arrow")
    if attribute == "shape":
        shape_aliases = {
            "box": "rect",
            "rect": "rect",
            "rectangle": "rect",
            "doublecircle": "double_circle",
        }
        dagua_shape = shape_aliases.get(value, value)
        return ("supported" if dagua_shape in NODE_SHAPE_NAMES else "missing", "NodeStyle.shape")
    if not dagua_field:
        return "missing", ""
    class_name, field_name = dagua_field.split(".", maxsplit=1)
    support_status = (
        "supported" if field_name in _style_fields().get(class_name, set()) else "missing"
    )
    return support_status, dagua_field


def _priority(attribute: str, value: str, support_status: str) -> str:
    """Assign P0-P3 priority for a generated cell.

    Parameters
    ----------
    attribute
        Graphviz attribute.
    value
        Graphviz value.
    support_status
        Dagua support state.

    Returns
    -------
    str
        Priority enum value.
    """

    if (attribute, value) in {("arrowhead", "normal"), ("shape", "ellipse")}:
        return "P0"
    if attribute in {"arrowhead", "shape", "color", "fillcolor", "fontname", "fontsize", "style"}:
        return "P1" if support_status == "missing" else "P0"
    if attribute in COSMETIC_ATTRS:
        return "P2"
    return "P3"


def _graphviz_cell(
    object_kind: str,
    attribute: str,
    value: str,
    value_group: str,
    source: str,
) -> Dict[str, Any]:
    """Create one Graphviz coverage cell.

    Parameters
    ----------
    object_kind
        Graphviz object kind.
    attribute
        Attribute name.
    value
        Attribute value.
    value_group
        Value group.
    source
        Source snapshot id.

    Returns
    -------
    Dict[str, Any]
        JSON-compatible coverage cell.
    """

    support_status, dagua_field = _dagua_support(object_kind, attribute, value, value_group)
    priority = _priority(attribute, value, support_status)
    metric_ids = ["declared_svg_match"]
    if attribute == "arrowhead":
        metric_ids = ["arrow_polygon_iou", "arrow_len_pct", "arrow_fill_mode"]
    elif attribute == "shape":
        metric_ids = ["shape_path_iou", "node_autosize_w_pt", "node_autosize_h_pt"]
    parity_status = "in_tolerance" if attribute == "arrowhead" and value == "normal" else "untested"
    cell_id = f"graphviz.{object_kind}.{attribute}.{value}"
    if value_group in {"arrow_modifier_expansion", "arrow_compound_sample"}:
        cell_id = f"graphviz.{object_kind}.{attribute}.{value_group}.{value}"
    return {
        "cell_id": cell_id,
        "tool": "graphviz",
        "object": object_kind,
        "attribute": attribute,
        "value": value,
        "value_group": value_group,
        "source": source,
        "dagua_field": dagua_field,
        "dagua_value": value,
        "support_status": support_status,
        "parity_status": parity_status,
        "priority": priority,
        "target_kind": "svg_declared",
        "geometry_mode": "injected",
        "fixture_ids": [f"{attribute}_atlas_{value}"],
        "metric_ids": metric_ids,
        "tolerance": {"declared_svg_match_min": 0.98},
        "last_checked_round": None,
        "locked": False,
        "lock_test": None,
        "residual_class": None,
        "waiver": None,
        "notes": "",
    }


def _adapter_capabilities() -> List[Dict[str, Any]]:
    """Return verified adapter capability seed rows.

    Returns
    -------
    List[Dict[str, Any]]
        Capability rows with ``gate_eligible=false`` until proven.
    """

    return [
        {
            "adapter": "gephi",
            "fixed_positions": False,
            "per_element_styles": False,
            "deterministic": False,
            "gate_eligible": False,
            "evidence": "gephi_renderer returns None when unavailable; no gating evidence",
        },
        {
            "adapter": "mermaid",
            "fixed_positions": False,
            "per_element_styles": False,
            "deterministic": True,
            "gate_eligible": False,
            "evidence": "mermaid adapter ignores supplied fixed positions",
        },
        {
            "adapter": "graphviz",
            "fixed_positions": False,
            "per_element_styles": True,
            "deterministic": True,
            "gate_eligible": False,
            "evidence": (
                "graphviz competitor adapter ignores fixed positions; v2 uses dot layout injection"
            ),
        },
        {
            "adapter": "cytoscape",
            "fixed_positions": True,
            "per_element_styles": False,
            "deterministic": True,
            "gate_eligible": False,
            "evidence": "cytoscape renderer applies first-node and first-edge styles globally",
        },
        {
            "adapter": "d3",
            "fixed_positions": True,
            "per_element_styles": False,
            "deterministic": True,
            "gate_eligible": False,
            "evidence": "d3 renderer is minimal straight-line/rect/ellipse smoke reference",
        },
    ]


def _external_tool_cells() -> List[Dict[str, Any]]:
    """Build blocked upstream rows for non-Graphviz survey tools.

    Returns
    -------
    List[Dict[str, Any]]
        Coverage cells for gate-ineligible external adapters.
    """

    cells: List[Dict[str, Any]] = []
    for tool in ("mermaid", "cytoscape", "d3", "yed", "drawio", "gephi", "neo4j"):
        cells.append(
            {
                "cell_id": f"{tool}.adapter.capability.fixed_geometry",
                "tool": tool,
                "object": "adapter",
                "attribute": "fixed_geometry",
                "value": "required",
                "value_group": "adapter_capability",
                "source": "tool-survey",
                "dagua_field": "",
                "dagua_value": "",
                "support_status": (
                    "reference_unavailable" if tool in {"yed", "drawio", "neo4j"} else "partial"
                ),
                "parity_status": "blocked_upstream",
                "priority": "P2",
                "target_kind": "tool_native",
                "geometry_mode": "native",
                "fixture_ids": [],
                "metric_ids": [],
                "tolerance": {},
                "last_checked_round": None,
                "locked": False,
                "lock_test": None,
                "residual_class": None,
                "waiver": None,
                "notes": "Gate eligibility awaits adapter capability proof.",
            }
        )
    return cells


def build_cells() -> List[Dict[str, Any]]:
    """Build all coverage cells from external denominator sources.

    Returns
    -------
    List[Dict[str, Any]]
        Sorted flat coverage cell rows.
    """

    cells: List[Dict[str, Any]] = []
    for value in ARROWHEAD_REGISTRY:
        cells.append(_graphviz_cell("edge", "arrowhead", value, "arrow_primitive", "gv-arrows"))
    for alias, primitive in sorted(ARROWHEAD_ALIASES.items()):
        cell = _graphviz_cell("edge", "arrowhead", alias, "arrow_alias", "gv-arrows")
        cell["notes"] = f"Alias for {primitive}."
        cells.append(cell)
    for alias in ("circle", "open"):
        if alias not in ARROWHEAD_ALIASES:
            cells.append(_graphviz_cell("edge", "arrowhead", alias, "arrow_alias", "gv-arrows"))
    for value in generate_graphviz_modifier_expansions():
        cells.append(
            _graphviz_cell(
                "edge",
                "arrowhead",
                value,
                "arrow_modifier_expansion",
                "gv-arrows",
            )
        )
    compound_samples = (
        "normaldot",
        "normalvee",
        "boxnormal",
        "diamondnormal",
        "crowtee",
        "teecrow",
        "dotnormal",
        "veenormal",
        "oboxnormal",
        "odiamondvee",
        "lnormalrnormal",
        "crowodot",
    )
    for value in compound_samples:
        cells.append(
            _graphviz_cell("edge", "arrowhead", value, "arrow_compound_sample", "gv-arrows")
        )
    for value in GRAPHVIZ_SHAPES:
        cells.append(_graphviz_cell("node", "shape", value, "node_shape", "gv-shapes"))
    for attr in COSMETIC_ATTRS:
        for object_kind in ("graph", "node", "edge", "cluster"):
            if (object_kind, attr) in FIELD_MAP:
                cells.append(
                    _graphviz_cell(object_kind, attr, "default", "cosmetic_attr", "gv-attrs")
                )
    cells.extend(_external_tool_cells())
    return sorted(cells, key=lambda cell: cell["cell_id"])


def _version(command: Sequence[str]) -> str:
    """Return a best-effort tool version string.

    Parameters
    ----------
    command
        Command argv.

    Returns
    -------
    str
        First output line or ``"unavailable"``.
    """

    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError:
        return "unavailable"
    output = (completed.stdout or completed.stderr).strip().splitlines()
    return output[0] if output else "unavailable"


def build_matrix(force_specs: bool = False) -> Dict[str, Any]:
    """Build the full coverage matrix payload.

    Parameters
    ----------
    force_specs
        Re-acquire reference snapshots before building.

    Returns
    -------
    Dict[str, Any]
        JSON-compatible coverage matrix.
    """

    source_snapshots = ensure_reference_specs(force=force_specs)
    return {
        "schema_version": COVERAGE_MATRIX_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "reference_pins": {
            "graphviz": GRAPHVIZ_VERSION,
            "dot": _version(["dot", "-V"]),
            "mmdc": _version(["mmdc", "--version"]),
            "cytosnap": _version(["npx", "cytosnap", "--version"]),
            "node": _version(["node", "--version"]),
            "matplotlib": _module_version("matplotlib"),
            "cairosvg": _module_version("cairosvg"),
            "python": platform.python_version(),
        },
        "source_snapshots": source_snapshots,
        "adapter_capabilities": _adapter_capabilities(),
        "cells": build_cells(),
    }


def _module_version(module_name: str) -> str:
    """Return an importable module's version.

    Parameters
    ----------
    module_name
        Python module name.

    Returns
    -------
    str
        Module version or ``"unavailable"``.
    """

    try:
        module = __import__(module_name)
    except ImportError:
        return "unavailable"
    return str(getattr(module, "__version__", "unknown"))


def gap_cells(matrix: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return missing P0/P1 gap cells for dashboard triage.

    Parameters
    ----------
    matrix
        Coverage matrix payload.

    Returns
    -------
    List[Dict[str, Any]]
        Gap rows.
    """

    return [
        cell
        for cell in matrix.get("cells", [])
        if cell.get("support_status") == "missing"
        and cell.get("parity_status") == "untested"
        and cell.get("priority") in {"P0", "P1"}
    ]


def _print_summary(matrix: Mapping[str, Any]) -> None:
    """Print a compact rebuild summary.

    Parameters
    ----------
    matrix
        Coverage matrix payload.

    Returns
    -------
    None
        The function prints to stdout.
    """

    cells = list(matrix.get("cells", []))
    graphviz_cosmetic = [
        cell
        for cell in cells
        if cell.get("tool") == "graphviz"
        and cell.get("value_group")
        in {
            "cosmetic_attr",
            "node_shape",
            "arrow_primitive",
            "arrow_alias",
            "arrow_modifier_expansion",
            "arrow_compound_sample",
        }
    ]
    categories = sorted(
        {cell["value_group"] for cell in cells if cell.get("attribute") == "arrowhead"}
    )
    print(f"coverage cells: {len(cells)}")
    print(f"graphviz cosmetic cells: {len(graphviz_cosmetic)}")
    print(f"arrow categories: {', '.join(categories)}")
    print(f"gap queue P0/P1: {len(gap_cells(matrix))}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the coverage matrix command-line interface.

    Parameters
    ----------
    argv
        Optional command arguments.

    Returns
    -------
    int
        Process exit code.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="regenerate coverage_matrix.json")
    parser.add_argument("--force-specs", action="store_true", help="refresh reference snapshots")
    args = parser.parse_args(argv)
    if not args.rebuild:
        parser.error("--rebuild is required for Lane C coverage generation")
    matrix = build_matrix(force_specs=args.force_specs)
    write_coverage_matrix(COVERAGE_PATH, matrix)
    _print_summary(matrix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
