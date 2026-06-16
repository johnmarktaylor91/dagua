"""Dagua: GPU-accelerated differentiable graph layout engine built on PyTorch."""

__version__ = "0.3.0"

# Enable expandable CUDA memory segments to prevent allocator fragmentation.
# Without this, large graphs (100M+ nodes) can OOM even with sufficient total
# VRAM because the allocator holds reserved-but-unusable memory blocks.
import os as _os

if "PYTORCH_CUDA_ALLOC_CONF" not in _os.environ:
    _os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from dataclasses import replace
from typing import Any, Optional, Tuple

from dagua.animation import (
    AnimationConfig,
    AnimationResult,
    CameraKeyframe,
    PosterConfig,
    PosterResult,
    TourConfig,
    animate,
    poster,
    tour,
)
from dagua.config import LayoutConfig
from dagua.defaults import (
    configure,
    defaults,
    export_config,
    get_defaults,
    reset,
    set_device,
    set_theme,
)
from dagua.edges import place_edge_labels, route_edges
from dagua.eval.compare import layout_all
from dagua.flex import AlignGroup, Flex, LayoutFlex
from dagua.generators import STRUCTURES as STRUCTURES
from dagua.generators import generate_graph as generate_graph
from dagua.graph import DaguaGraph
from dagua.io import (
    ImageAIConfig,
    configure_image_ai,
    from_dot,
    from_igraph,
    from_scipy,
    get_image_ai_config,
    graph_code_from_image,
    graph_dict_from_image,
    graph_from_json,
    graph_from_yaml,
    graph_script_from_dict,
    graph_to_json,
    graph_to_yaml,
    load,
    load_style,
    save,
    save_style,
    theme_code_from_image,
    theme_dict_from_image,
    theme_from_image,
    to_igraph,
    to_networkx,
    to_pyg,
    to_scipy,
)
from dagua.io import graph_from_image as from_image
from dagua.layout import layout
from dagua.metrics import evaluate, layout_similarity
from dagua.playground import launch_playground
from dagua.render import render
from dagua.render._backend import get_default_backend, set_default_backend
from dagua.styles import (
    DARK_THEME,
    DEFAULT_NODE_STYLES,
    # Backwards-compatible aliases
    DEFAULT_THEME,
    DEFAULT_THEME_OBJ,
    GRAPHVIZ_MATCH_NODE_STYLES,
    GRAPHVIZ_MATCH_THEME,
    GRAPHVIZ_STRICT_THEME,
    MINIMAL_THEME,
    PALETTE,
    TORCHLENS_THEME,
    ClusterStyle,
    EdgeStyle,
    GraphStyle,
    NodeStyle,
    Theme,
    border_from_fill,
    get_theme,
    make_fill,
)


def draw(
    graph: Any,
    config: Optional[LayoutConfig] = None,
    output: Optional[str] = None,
    relayout: Optional[bool] = None,
    use_cached: bool = False,
    direction: Optional[str] = None,
    backend: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[Any, Any]:
    """Layout and render a graph in one convenience call.

    Parameters
    ----------
    graph : Any
        Dagua graph object to lay out and render.
    config : LayoutConfig, optional
        Layout configuration. When omitted, global defaults provide device,
        layout overrides, and theme settings.
    output : str, optional
        Output file path passed to the renderer.
    relayout : bool, optional
        Force or skip layout recomputation.
    use_cached : bool, default=False
        Reuse cached graph layout positions.
    direction : str, optional
        Layout direction override.
    backend : str, optional
        Matplotlib render backend: ``"agg"``, ``"cairo"``, or ``None`` for the
        auto-detected/global default.
    **kwargs : Any
        Additional keyword arguments forwarded to ``dagua.render``.

    Returns
    -------
    tuple[Any, Any]
        Matplotlib ``(figure, axes)`` pair from ``dagua.render``.
    """
    user_supplied_config = config is not None
    if config is None:
        from dagua.defaults import get_default_device, get_default_layout_overrides

        layout_overrides = get_default_layout_overrides()
        effective_direction = direction or graph.direction
        config = LayoutConfig(
            device=get_default_device(),
            direction=effective_direction,
            **layout_overrides,
        )
    else:
        effective_direction = direction or config.direction
        if direction is not None and config.direction != direction:
            config = replace(config, direction=direction)

    effective_direction = config.direction

    if relayout is False or use_cached:
        if not graph.has_fresh_layout:
            raise ValueError(
                f"Graph layout is {graph.layout_status}. Re-run dagua.layout() or dagua.draw(), "
                "or pass relayout=True."
            )
        positions = graph.last_positions.to(config.device)
    elif relayout is None and graph.has_fresh_layout and not user_supplied_config:
        positions = graph.last_positions.to(config.device)
    else:
        positions = layout(graph, config)

    graph.compute_node_sizes()
    curves = route_edges(positions, graph.edge_index, graph.node_sizes, effective_direction, graph)

    # Sprint 6: edge_routing takes precedence over edge_opt_steps. Only
    # the "differentiable" mode runs optimize_edges; "heuristic" keeps
    # the bezier curves that route_edges produced (zero gradient work).
    if (
        getattr(config, "edge_routing", "differentiable") == "differentiable"
        and getattr(config, "edge_opt_steps", 0) >= 0
    ):
        # Sprint 6 r3: adaptive skip when the heuristic already produces
        # a near-optimal routing. The Sprint 6 audit revealed that nested
        # cluster graphs (nested_2lvl/3lvl/4lvl) have zero or very few
        # edge-node crossings under heuristic routing, and CP refinement
        # CREATES new crossings there (nested_2lvl went 0 -> 33 under
        # Sprint-5 defaults; the heuristic path already respects cluster
        # locality which the six CP losses don't model). Skipping when
        # heuristic crossings < adaptive_skip_threshold (default 5) closes
        # that regression without touching graph families where
        # differentiable mode meaningfully wins.
        from dagua.metrics import edge_node_crossing_count

        heur_crossings = edge_node_crossing_count(
            curves, positions, graph.node_sizes, graph.edge_index
        )["edge_node_crossings"]
        threshold = getattr(config, "edge_routing_auto_skip_threshold", 5)
        has_rectilinear_routes = any(
            getattr(curve, "routing", "bezier") in ("ortho", "taxi") for curve in curves
        )
        if heur_crossings >= threshold or has_rectilinear_routes:
            from dagua.layout.edge_optimization import optimize_edges

            curves = optimize_edges(
                curves, positions, graph.edge_index, graph.node_sizes, config, graph
            )

    label_positions = place_edge_labels(
        curves, positions, graph.node_sizes, graph.edge_labels, graph
    )
    graph.cache_routing(curves, label_positions)

    return render(
        graph,
        positions,
        config,
        output=output,
        curves=curves,
        label_positions=label_positions,
        backend=backend,
        **kwargs,
    )


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
    "TORCHLENS_THEME",
    "DEFAULT_THEME",
    "DEFAULT_NODE_STYLES",
    "GRAPHVIZ_MATCH_THEME",
    "GRAPHVIZ_MATCH_NODE_STYLES",
    "GRAPHVIZ_STRICT_THEME",
    # Flex system
    "Flex",
    "LayoutFlex",
    "AlignGroup",
    "layout_all",
    "layout_similarity",
    "evaluate",
    # Global defaults
    "set_theme",
    "set_device",
    "configure",
    "defaults",
    "get_defaults",
    "export_config",
    "reset",
    "set_default_backend",
    "get_default_backend",
    # IO
    "route_edges",
    "place_edge_labels",
    "from_image",
    "theme_from_image",
    "graph_dict_from_image",
    "graph_code_from_image",
    "graph_script_from_dict",
    "theme_dict_from_image",
    "theme_code_from_image",
    "ImageAIConfig",
    "configure_image_ai",
    "get_image_ai_config",
    "load",
    "save",
    "load_style",
    "save_style",
    "graph_from_json",
    "graph_to_json",
    "graph_from_yaml",
    "graph_to_yaml",
    "launch_playground",
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
