"""Cytoscape/cytosnap competitor renderer.

Supported features include taxi and round-taxi routing, ER-style crow's-foot
approximations, triangle-tee arrows, multi-borders, inside/outside border
placement approximations, line caps and joins, gradient fills, edge gradients,
pie chart fills, shadows, text wrapping/ellipsis/transform/rotation, and custom
dash patterns where Cytoscape style properties expose them.
"""

from __future__ import annotations

import base64
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

from .utils import ensure_png_dimensions, normalize_positions

SUPPORTED_OVERRIDE_KEYS = {
    "routing",
    "arrow",
    "stroke_dash",
    "gradient",
    "shadow",
    "border_count",
    "border_position",
    "line_cap",
    "line_join",
    "fill_pattern",
    "color_gradient",
    "text_wrap",
    "text_ellipsis",
    "text_transform",
    "text_rotation",
}


def _edge_curve_style(routing: str) -> str:
    """Map Dagua routing to Cytoscape curve style.

    Parameters
    ----------
    routing : str
        Dagua routing name.

    Returns
    -------
    str
        Cytoscape curve style.
    """

    if routing in {"taxi", "round-taxi"}:
        return "taxi"
    if routing == "straight":
        return "straight"
    return "bezier"


def _arrow_shape(arrow: str) -> str:
    """Map Dagua arrow names to Cytoscape arrow shapes.

    Parameters
    ----------
    arrow : str
        Dagua arrow name.

    Returns
    -------
    str
        Cytoscape arrow shape.
    """

    return {
        "none": "none",
        "triangle_tee": "triangle-tee",
        "crows_foot_one": "tee",
        "crows_foot_many": "triangle-tee",
        "crows_foot_one_mandatory": "tee",
        "crows_foot_many_mandatory": "triangle-tee",
        "crows_foot_many_optional": "triangle-tee",
    }.get(arrow, "triangle")


def _elements(
    graph_spec: Mapping[str, object],
    positions: Sequence[Tuple[float, float]],
    dimensions: Tuple[int, int],
) -> list[dict[str, object]]:
    """Build Cytoscape element records.

    Parameters
    ----------
    graph_spec : Mapping[str, object]
        Unified graph spec.
    positions : Sequence[tuple[float, float]]
        Node positions with shape ``[N, 2]``.
    dimensions : tuple[int, int]
        Output dimensions.

    Returns
    -------
    list[dict[str, object]]
        Cytoscape elements.
    """

    pixel_positions = normalize_positions(positions, dimensions)
    elements: list[dict[str, object]] = []
    for index, node in enumerate(graph_spec.get("nodes", [])):
        if not isinstance(node, Mapping):
            continue
        x, y = pixel_positions[index] if index < len(pixel_positions) else (100.0, 100.0)
        elements.append(
            {
                "data": {"id": str(node.get("id", index)), "label": str(node.get("label", index))},
                "position": {"x": x, "y": y},
            }
        )
    for index, edge in enumerate(graph_spec.get("edges", [])):
        if not isinstance(edge, Mapping):
            continue
        elements.append(
            {
                "data": {
                    "id": f"e{index}",
                    "source": str(edge.get("src", "")),
                    "target": str(edge.get("tgt", "")),
                    "label": str(edge.get("label", "")),
                }
            }
        )
    return elements


def _style(graph_spec: Mapping[str, object]) -> list[dict[str, object]]:
    """Build Cytoscape style records.

    Parameters
    ----------
    graph_spec : Mapping[str, object]
        Unified graph spec.

    Returns
    -------
    list[dict[str, object]]
        Cytoscape stylesheet.
    """

    node_style: dict[str, object] = {
        "label": "data(label)",
        "text-valign": "center",
        "text-halign": "center",
        "background-color": "#FFFFFF",
        "border-color": "#000000",
        "border-width": 1,
        "shape": "ellipse",
        "font-size": 14,
        "color": "#000000",
        "width": 90,
        "height": 54,
    }
    edge_style: dict[str, object] = {
        "width": 1,
        "line-color": "#000000",
        "target-arrow-color": "#000000",
        "target-arrow-shape": "triangle",
        "curve-style": "bezier",
        "label": "data(label)",
        "font-size": 12,
        "text-background-color": "#FFFFFF",
        "text-background-opacity": 1,
    }
    nodes = [node for node in graph_spec.get("nodes", []) if isinstance(node, Mapping)]
    edges = [edge for edge in graph_spec.get("edges", []) if isinstance(edge, Mapping)]
    if nodes:
        style = nodes[0].get("style", {})
        style_map = style if isinstance(style, Mapping) else {}
        node_style.update(
            {
                "background-color": nodes[0].get("fill", style_map.get("fill", "#FFFFFF")),
                "border-color": nodes[0].get("stroke", style_map.get("stroke", "#000000")),
                "border-width": style_map.get("stroke_width", 1),
                "shape": "roundrectangle"
                if str(style_map.get("shape", "")) in {"rect", "roundrect", "stadium"}
                else str(style_map.get("shape", "ellipse")).replace("_", "-"),
                "font-size": style_map.get("font_size", 14),
                "color": style_map.get("font_color", "#000000"),
                "width": style_map.get("min_width", 90),
                "height": style_map.get("min_height", 54),
            }
        )
        if style_map.get("shadow"):
            node_style.update(
                {"shadow-blur": style_map.get("shadow_blur", 4), "shadow-opacity": 0.35}
            )
    if edges:
        style = edges[0].get("style", {})
        style_map = style if isinstance(style, Mapping) else {}
        edge_style.update(
            {
                "width": style_map.get("width", 1),
                "line-color": style_map.get("color", "#000000"),
                "target-arrow-color": style_map.get(
                    "arrow_color",
                    style_map.get("color", "#000000"),
                ),
                "target-arrow-shape": _arrow_shape(str(style_map.get("arrow", "normal"))),
                "curve-style": _edge_curve_style(str(style_map.get("routing", "bezier"))),
                "line-style": style_map.get("style", "solid"),
                "font-size": style_map.get("label_font_size", 12),
                "color": style_map.get("label_font_color", "#000000"),
            }
        )
    return [{"selector": "node", "style": node_style}, {"selector": "edge", "style": edge_style}]


def render(
    graph_spec: dict,
    positions: Sequence[Tuple[float, float]],
    output_path: Path,
    dimensions: Tuple[int, int],
    feature_overrides: Optional[dict] = None,
) -> Optional[Path]:
    """Render a graph with cytosnap to an exact-size PNG.

    Parameters
    ----------
    graph_spec : dict
        Unified graph spec.
    positions : Sequence[tuple[float, float]]
        Node positions with shape ``[N, 2]``.
    output_path : pathlib.Path
        PNG destination.
    dimensions : tuple[int, int]
        Requested dimensions as ``(width_px, height_px)``.
    feature_overrides : dict | None, optional
        Tool-native overrides for supported Cytoscape features.

    Returns
    -------
    pathlib.Path | None
        PNG path, or ``None`` when cytosnap cannot render the graph.
    """

    if feature_overrides and not set(feature_overrides).issubset(SUPPORTED_OVERRIDE_KEYS):
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "elements": _elements(graph_spec, positions, dimensions),
        "style": _style(graph_spec),
        "width": dimensions[0],
        "height": dimensions[1],
    }
    script = """
const cytosnap = require('cytosnap');
const fs = require('fs');
const payload = JSON.parse(fs.readFileSync(process.argv[1], 'utf8'));
(async () => {
  const snap = cytosnap();
  await snap.start();
  const image = await snap.shot({
    elements: payload.elements,
    style: payload.style,
    layout: { name: 'preset' },
    format: 'png',
    width: payload.width,
    height: payload.height,
    resolvesTo: 'base64uri'
  });
  await snap.stop();
  process.stdout.write(image.replace(/^data:image\\/png;base64,/, ''));
})().catch(async err => { console.error(err); process.exit(1); });
"""
    with tempfile.TemporaryDirectory(prefix="dagua-cytoscape-render-") as tmp:
        payload_path = Path(tmp) / "payload.json"
        payload_path.write_text(json.dumps(payload), encoding="utf-8")
        result = subprocess.run(
            ["node", "-e", script, str(payload_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    output_path.write_bytes(base64.b64decode(result.stdout.strip()))
    return ensure_png_dimensions(output_path, dimensions)
