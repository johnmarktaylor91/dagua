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
from dagua.edges import place_edge_labels, route_edges
from dagua.layout import layout
from dagua.render import render
from dagua.io import graph_from_image as from_image
from dagua.io import theme_from_image


def draw(graph, config=None, output=None, **kwargs):
    """Layout + render in one call. Convenience function.

    Full pipeline: layout → route_edges → optimize_edges → place_edge_labels → render.
    Edge optimization is controlled by config.edge_opt_steps (0=auto, -1=skip, >0=explicit).
    """
    config = config or LayoutConfig()
    positions = layout(graph, config)
    graph.compute_node_sizes()
    curves = route_edges(positions, graph.edge_index, graph.node_sizes, graph.direction, graph)

    if getattr(config, "edge_opt_steps", 0) >= 0:
        from dagua.layout.edge_optimization import optimize_edges
        curves = optimize_edges(curves, positions, graph.edge_index, graph.node_sizes, config, graph)

    label_positions = place_edge_labels(curves, positions, graph.node_sizes, graph.edge_labels, graph)

    return render(graph, positions, config, output=output,
                  curves=curves, label_positions=label_positions, **kwargs)


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
    "route_edges",
    "place_edge_labels",
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
