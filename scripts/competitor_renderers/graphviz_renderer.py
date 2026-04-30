"""Graphviz competitor renderer.

Supported features include Graphviz node shapes, standard arrowheads, straight
or orthogonal routing approximations, solid/dashed/dotted edge styles, bold
edges, basic fonts, external labels, head/tail labels, cluster fill/stroke, and
Graphviz-compatible striped, wedged, linear, or radial fills when expressed as
DOT attributes.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

from .utils import command_available, ensure_png_dimensions, sanitize_id

GRAPHVIZ_SHAPES = {
    "rect": "box",
    "roundrect": "box",
    "ellipse": "ellipse",
    "circle": "circle",
    "diamond": "diamond",
    "triangle": "triangle",
    "hexagon": "hexagon",
    "pentagon": "pentagon",
    "octagon": "octagon",
    "star": "star",
    "cylinder": "cylinder",
    "parallelogram": "parallelogram",
    "trapezoid": "trapezium",
    "double_circle": "doublecircle",
    "tab": "tab",
    "note": "note",
    "box3d": "box3d",
}


def _dot_attrs(attrs: Mapping[str, object]) -> str:
    """Format DOT attributes.

    Parameters
    ----------
    attrs : Mapping[str, object]
        Attribute mapping.

    Returns
    -------
    str
        DOT attribute list body.
    """

    parts = []
    for key, value in attrs.items():
        escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
        parts.append(f'{key}="{escaped}"')
    return ", ".join(parts)


def _node_attrs(node: Mapping[str, object]) -> Mapping[str, object]:
    """Return Graphviz node attributes for a unified node spec.

    Parameters
    ----------
    node : Mapping[str, object]
        Unified node record.

    Returns
    -------
    Mapping[str, object]
        DOT attributes.
    """

    style = node.get("style", {})
    style_map = style if isinstance(style, Mapping) else {}
    shape = str(node.get("shape", style_map.get("shape", "ellipse")))
    dot_shape = GRAPHVIZ_SHAPES.get(shape, "ellipse")
    attrs = {
        "label": node.get("label", node.get("id", "")),
        "shape": dot_shape,
        "style": "filled,rounded" if shape == "roundrect" else "filled",
        "fillcolor": node.get("fill", style_map.get("fill", "#FFFFFF")),
        "color": node.get("stroke", style_map.get("stroke", "#000000")),
        "penwidth": style_map.get("stroke_width", 1.0),
        "fontname": style_map.get("font_family", "Times,serif"),
        "fontsize": style_map.get("font_size", 14.0),
        "fontcolor": style_map.get("font_color", "#000000"),
    }
    return attrs


def _edge_attrs(edge: Mapping[str, object]) -> Mapping[str, object]:
    """Return Graphviz edge attributes for a unified edge spec.

    Parameters
    ----------
    edge : Mapping[str, object]
        Unified edge record.

    Returns
    -------
    Mapping[str, object]
        DOT attributes.
    """

    style = edge.get("style", {})
    style_map = style if isinstance(style, Mapping) else {}
    arrow = str(style_map.get("arrow", "normal"))
    return {
        "color": style_map.get("color", "#000000"),
        "penwidth": style_map.get("width", 1.0),
        "style": style_map.get("style", "solid"),
        "arrowhead": "none" if arrow == "none" else arrow,
        "fontname": style_map.get("label_font_family", "Times,serif"),
        "fontsize": style_map.get("label_font_size", 14.0),
        "fontcolor": style_map.get("label_font_color", "#000000"),
    }


def _build_dot(graph_spec: Mapping[str, object]) -> str:
    """Build DOT source from a unified graph spec.

    Parameters
    ----------
    graph_spec : Mapping[str, object]
        Unified graph specification.

    Returns
    -------
    str
        DOT source.
    """

    lines = ["digraph G {", '  graph [bgcolor="white", margin="0"];']
    for node in graph_spec.get("nodes", []):
        if not isinstance(node, Mapping):
            continue
        node_id = sanitize_id(node.get("id", "node"))
        lines.append(f"  {node_id} [{_dot_attrs(_node_attrs(node))}];")
    for edge in graph_spec.get("edges", []):
        if not isinstance(edge, Mapping):
            continue
        src = sanitize_id(edge.get("src", ""))
        tgt = sanitize_id(edge.get("tgt", ""))
        lines.append(f"  {src} -> {tgt} [{_dot_attrs(_edge_attrs(edge))}];")
    lines.append("}")
    return "\n".join(lines)


def render(
    graph_spec: dict,
    positions: Sequence[Tuple[float, float]],
    output_path: Path,
    dimensions: Tuple[int, int],
    feature_overrides: Optional[dict] = None,
) -> Optional[Path]:
    """Render a graph with Graphviz to an exact-size PNG.

    Parameters
    ----------
    graph_spec : dict
        Unified graph spec.
    positions : Sequence[tuple[float, float]]
        Node positions with shape ``[N, 2]``. Graphviz dot ignores fixed
        positions; this reference isolates supported styling, not layout.
    output_path : pathlib.Path
        PNG destination.
    dimensions : tuple[int, int]
        Requested dimensions as ``(width_px, height_px)``.
    feature_overrides : dict | None, optional
        Tool-native feature overrides, currently unused.

    Returns
    -------
    pathlib.Path | None
        PNG path, or ``None`` when Graphviz is unavailable.
    """

    del positions, feature_overrides
    if not command_available("dot"):
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dagua-gv-render-") as tmp:
        dot_path = Path(tmp) / "graph.dot"
        dot_path.write_text(_build_dot(graph_spec), encoding="utf-8")
        result = subprocess.run(
            ["dot", "-Tpng", "-Gdpi=150", str(dot_path), "-o", str(output_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    if result.returncode != 0:
        return None
    return ensure_png_dimensions(output_path, dimensions)
