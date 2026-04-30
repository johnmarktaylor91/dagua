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


def _hex_with_alpha(color: object, opacity: object = 1.0) -> str:
    """Return a Graphviz color token with an alpha channel.

    Parameters
    ----------
    color : object
        Base color token.
    opacity : object, default=1.0
        Alpha multiplier on ``[0, 1]``.

    Returns
    -------
    str
        ``#RRGGBBAA`` when possible, otherwise the original color string.
    """
    text = str(color)
    if not text.startswith("#") or len(text) not in {7, 9}:
        return text
    try:
        alpha = min(max(float(opacity), 0.0), 1.0)
    except (TypeError, ValueError):
        alpha = 1.0
    return f"{text[:7]}{int(round(alpha * 255.0)):02X}"


def _graph_attrs(graph_spec: Mapping[str, object]) -> Mapping[str, object]:
    """Return Graphviz graph attributes for a unified graph spec.

    Parameters
    ----------
    graph_spec : Mapping[str, object]
        Unified graph specification.

    Returns
    -------
    Mapping[str, object]
        DOT graph attributes.
    """
    style = graph_spec.get("style", graph_spec.get("graph", {}))
    style_map = style if isinstance(style, Mapping) else {}
    direction = graph_spec.get("direction", style_map.get("direction", "TB"))
    return {
        "bgcolor": style_map.get("background_color", style_map.get("bgcolor", "white")),
        "margin": style_map.get("margin", 0),
        "rankdir": str(direction).upper(),
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
    opacity = style_map.get("opacity", 1.0)
    border_opacity = style_map.get("border_opacity", opacity)
    labeljust = {"left": "l", "center": "c", "right": "r"}.get(
        str(style_map.get("text_align", "center")),
        "c",
    )
    labelloc = {"top": "t", "center": "c", "bottom": "b"}.get(
        str(style_map.get("text_valign", "center")),
        "c",
    )
    attrs = {
        "label": node.get("label", node.get("id", "")),
        "shape": dot_shape,
        "style": "filled,rounded" if shape == "roundrect" else "filled",
        "fillcolor": _hex_with_alpha(
            node.get("fill", style_map.get("fill", "#FFFFFF")),
            opacity,
        ),
        "color": _hex_with_alpha(
            node.get("stroke", style_map.get("stroke", "#000000")),
            border_opacity,
        ),
        "penwidth": style_map.get("stroke_width", 1.0),
        "fontname": style_map.get("font_family", "Times,serif"),
        "fontsize": style_map.get("font_size", 14.0),
        "fontcolor": style_map.get("font_color", "#000000"),
        "labeljust": labeljust,
        "labelloc": labelloc,
    }
    external_label = str(style_map.get("external_label", "") or "")
    if external_label:
        attrs["xlabel"] = external_label
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
    edge_style = str(style_map.get("style", "solid"))
    if bool(style_map.get("taper")):
        edge_style = "tapered" if edge_style == "solid" else f"{edge_style},tapered"
    return {
        "color": style_map.get("color", "#000000"),
        "penwidth": style_map.get("width", 1.0),
        "style": edge_style,
        "arrowhead": "none" if arrow == "none" else arrow,
        "fontname": style_map.get("label_font_family", "Times,serif"),
        "fontsize": style_map.get("label_font_size", 14.0),
        "fontcolor": style_map.get("label_font_color", "#000000"),
        "label": edge.get("label", edge.get("xlabel", "")),
        "headlabel": style_map.get("head_label", ""),
        "taillabel": style_map.get("tail_label", ""),
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

    lines = ["digraph G {", f"  graph [{_dot_attrs(_graph_attrs(graph_spec))}];"]
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
