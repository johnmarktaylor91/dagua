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
from dagua.animation import (
    animate, AnimationConfig, AnimationResult,
    tour, TourConfig, CameraKeyframe,
    poster, PosterConfig, PosterResult,
)
from dagua.io import graph_from_image as from_image
from dagua.io import theme_from_image
from dagua.io import (
    load, save,
    load_style, save_style,
    graph_from_json, graph_to_json,
    graph_from_yaml, graph_to_yaml,
    to_networkx, to_igraph, to_pyg, to_scipy,
    from_igraph, from_scipy, from_dot,
)
from dagua.styles import get_theme
from dagua.flex import Flex, LayoutFlex, AlignGroup
from dagua.defaults import (
    set_theme, set_device, configure, defaults,
    get_defaults, export_config, reset,
)


def draw(graph, config=None, output=None, **kwargs):
    """Layout + render in one call. Convenience function.

    Full pipeline: layout → route_edges → optimize_edges → place_edge_labels → render.
    Edge optimization is controlled by config.edge_opt_steps (0=auto, -1=skip, >0=explicit).

    When config=None, consults global defaults (dagua.configure()) for
    device, layout overrides, and theme settings.
    """
    if config is None:
        from dagua.defaults import get_default_device, get_default_layout_overrides
        layout_overrides = get_default_layout_overrides()
        config = LayoutConfig(device=get_default_device(), **layout_overrides)

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
    # Core
    "DaguaGraph",
    "LayoutConfig",
    "layout",
    "render",
    "draw",
    "animate",
    "tour",
    "poster",
    "AnimationConfig",
    "AnimationResult",
    "TourConfig",
    "CameraKeyframe",
    "PosterConfig",
    "PosterResult",
    # Styles
    "NodeStyle",
    "EdgeStyle",
    "ClusterStyle",
    "GraphStyle",
    "Theme",
    "PALETTE",
    "make_fill",
    "border_from_fill",
    "get_theme",
    "DEFAULT_THEME_OBJ",
    "DARK_THEME",
    "MINIMAL_THEME",
    "DEFAULT_THEME",
    "DEFAULT_NODE_STYLES",
    "GRAPHVIZ_MATCH_THEME",
    "GRAPHVIZ_MATCH_NODE_STYLES",
    # Flex system
    "Flex",
    "LayoutFlex",
    "AlignGroup",
    # Global defaults
    "set_theme",
    "set_device",
    "configure",
    "defaults",
    "get_defaults",
    "export_config",
    "reset",
    # IO
    "route_edges",
    "place_edge_labels",
    "from_image",
    "theme_from_image",
    "load",
    "save",
    "load_style",
    "save_style",
    "graph_from_json",
    "graph_to_json",
    "graph_from_yaml",
    "graph_to_yaml",
    # Interop exports
    "to_networkx",
    "to_igraph",
    "to_pyg",
    "to_scipy",
    # Interop imports
    "from_igraph",
    "from_scipy",
    "from_dot",
]
