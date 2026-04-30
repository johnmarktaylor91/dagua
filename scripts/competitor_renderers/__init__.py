"""Registry for cosmetic competitor render adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

from . import cytoscape_renderer, d3_renderer, gephi_renderer, graphviz_renderer, mermaid_renderer

Renderer = Callable[
    [dict, Sequence[Tuple[float, float]], Path, Tuple[int, int], Optional[dict]],
    Optional[Path],
]

RENDERERS: Dict[str, Renderer] = {
    "graphviz": graphviz_renderer.render,
    "mermaid": mermaid_renderer.render,
    "cytoscape": cytoscape_renderer.render,
    "d3": d3_renderer.render,
    "gephi": gephi_renderer.render,
}


def render_competitor(
    tool_name: str,
    graph_spec: dict,
    positions: Sequence[Tuple[float, float]],
    output_path: Path,
    dimensions: Tuple[int, int],
    feature_overrides: Optional[dict] = None,
) -> Optional[Path]:
    """Dispatch to one registered competitor renderer.

    Parameters
    ----------
    tool_name : str
        Renderer key such as ``"graphviz"`` or ``"cytoscape"``.
    graph_spec : dict
        Unified graph spec.
    positions : Sequence[tuple[float, float]]
        Node positions with shape ``[N, 2]``.
    output_path : pathlib.Path
        PNG destination.
    dimensions : tuple[int, int]
        Requested dimensions as ``(width_px, height_px)``.
    feature_overrides : dict | None, optional
        Tool-native feature overrides.

    Returns
    -------
    pathlib.Path | None
        PNG path, or ``None`` when unavailable/unsupported.

    Raises
    ------
    KeyError
        Raised when ``tool_name`` is not registered.
    """

    return RENDERERS[tool_name](graph_spec, positions, output_path, dimensions, feature_overrides)
