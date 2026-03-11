"""Dagua: GPU-accelerated differentiable graph layout engine built on PyTorch."""

__version__ = "0.0.2"

from dagua.graph import DaguaGraph
from dagua.styles import (
    NodeStyle, EdgeStyle, ClusterStyle, GraphStyle, Theme,
    PALETTE, make_fill, border_from_fill,
    DEFAULT_THEME_OBJ, DARK_THEME, MINIMAL_THEME,
    # Backwards-compatible aliases
    DEFAULT_THEME, DEFAULT_NODE_STYLES, GRAPHVIZ_MATCH_THEME, GRAPHVIZ_MATCH_NODE_STYLES,
)
from dagua.config import LayoutConfig
from dagua.layout import layout
from dagua.render import render
from dagua.io import graph_from_image as from_image
from dagua.io import theme_from_image


def draw(graph, config=None, output=None, **kwargs):
    """Layout + render in one call. Convenience function."""
    positions = layout(graph, config)
    return render(graph, positions, config, output=output, **kwargs)


__all__ = [
    "DaguaGraph",
    "NodeStyle",
    "EdgeStyle",
    "ClusterStyle",
    "GraphStyle",
    "Theme",
    "LayoutConfig",
    "layout",
    "render",
    "draw",
    "PALETTE",
    "make_fill",
    "border_from_fill",
    "from_image",
    "theme_from_image",
    "DEFAULT_THEME_OBJ",
    "DARK_THEME",
    "MINIMAL_THEME",
    "DEFAULT_THEME",
    "DEFAULT_NODE_STYLES",
    "GRAPHVIZ_MATCH_THEME",
    "GRAPHVIZ_MATCH_NODE_STYLES",
]
