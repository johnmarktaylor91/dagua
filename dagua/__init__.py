"""Dagua: GPU-accelerated differentiable graph layout engine built on PyTorch."""

__version__ = "0.0.2"

from dagua.graph import DaguaGraph
from dagua.styles import NodeStyle, EdgeStyle, ClusterStyle, PALETTE, make_fill, border_from_fill
from dagua.config import LayoutConfig
from dagua.layout import layout
from dagua.render import render


def draw(graph, config=None, output=None, **kwargs):
    """Layout + render in one call. Convenience function."""
    positions = layout(graph, config)
    return render(graph, positions, config, output=output, **kwargs)


__all__ = [
    "DaguaGraph",
    "NodeStyle",
    "EdgeStyle",
    "ClusterStyle",
    "LayoutConfig",
    "layout",
    "render",
    "draw",
    "PALETTE",
    "make_fill",
    "border_from_fill",
]
