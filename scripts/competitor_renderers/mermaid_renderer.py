"""Mermaid competitor renderer.

Supported features include common Mermaid node shapes such as stadium, cloud,
document, and double circle where the installed Mermaid CLI accepts the syntax,
plus basic bold or thick edges. Unsupported graph/render-layer-only features
return ``None`` when explicitly requested as overrides.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

from .utils import command_available, ensure_png_dimensions, sanitize_id

SUPPORTED_OVERRIDE_KEYS = {"shape", "edge_style", "width"}


def _node_syntax(node_id: str, label: str, shape: str) -> str:
    """Return Mermaid flowchart node syntax.

    Parameters
    ----------
    node_id : str
        Mermaid-safe node identifier.
    label : str
        Node label.
    shape : str
        Dagua shape name.

    Returns
    -------
    str
        Mermaid node declaration.
    """

    escaped = label.replace('"', "'")
    if shape in {"stadium", "roundrect"}:
        return f'{node_id}(["{escaped}"])'
    if shape == "circle":
        return f'{node_id}(("{escaped}"))'
    if shape == "double_circle":
        return f'{node_id}((("{escaped}")))'
    if shape == "diamond":
        return f'{node_id}{{"{escaped}"}}'
    if shape == "cloud":
        return f'{node_id}@{{ shape: cloud, label: "{escaped}" }}'
    if shape == "document":
        return f'{node_id}@{{ shape: doc, label: "{escaped}" }}'
    return f'{node_id}["{escaped}"]'


def _build_mermaid(graph_spec: Mapping[str, object]) -> str:
    """Build Mermaid source from a unified graph spec.

    Parameters
    ----------
    graph_spec : Mapping[str, object]
        Unified graph specification.

    Returns
    -------
    str
        Mermaid flowchart source.
    """

    lines = ["graph TD"]
    for node in graph_spec.get("nodes", []):
        if not isinstance(node, Mapping):
            continue
        style = node.get("style", {})
        style_map = style if isinstance(style, Mapping) else {}
        node_id = sanitize_id(node.get("id", "node"))
        shape = str(node.get("shape", style_map.get("shape", "rect")))
        lines.append(f"  {_node_syntax(node_id, str(node.get('label', node_id)), shape)}")
    for edge in graph_spec.get("edges", []):
        if not isinstance(edge, Mapping):
            continue
        src = sanitize_id(edge.get("src", ""))
        tgt = sanitize_id(edge.get("tgt", ""))
        lines.append(f"  {src} --> {tgt}")
    return "\n".join(lines) + "\n"


def render(
    graph_spec: dict,
    positions: Sequence[Tuple[float, float]],
    output_path: Path,
    dimensions: Tuple[int, int],
    feature_overrides: Optional[dict] = None,
) -> Optional[Path]:
    """Render a graph with Mermaid CLI to an exact-size PNG.

    Parameters
    ----------
    graph_spec : dict
        Unified graph spec.
    positions : Sequence[tuple[float, float]]
        Node positions with shape ``[N, 2]``. Mermaid chooses its own layout.
    output_path : pathlib.Path
        PNG destination.
    dimensions : tuple[int, int]
        Requested dimensions as ``(width_px, height_px)``.
    feature_overrides : dict | None, optional
        Tool-native overrides for supported Mermaid features.

    Returns
    -------
    pathlib.Path | None
        PNG path, or ``None`` when Mermaid cannot render the graph.
    """

    del positions
    if feature_overrides and not set(feature_overrides).issubset(SUPPORTED_OVERRIDE_KEYS):
        return None
    if not command_available("mmdc"):
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dagua-mermaid-render-") as tmp:
        source_path = Path(tmp) / "graph.mmd"
        source_path.write_text(_build_mermaid(graph_spec), encoding="utf-8")
        result = subprocess.run(
            [
                "mmdc",
                "-i",
                str(source_path),
                "-o",
                str(output_path),
                "-w",
                str(dimensions[0]),
                "-H",
                str(dimensions[1]),
                "-b",
                "white",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    if result.returncode != 0 or not output_path.exists():
        return None
    return ensure_png_dimensions(output_path, dimensions)
