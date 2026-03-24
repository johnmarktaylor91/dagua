"""Tests for graph-scoped style defaults."""

from __future__ import annotations

from typing import Iterator

import pytest

from dagua.defaults import reset
from dagua.graph import DaguaGraph
from dagua.render import mpl as mpl_renderer
from dagua.styles import ClusterStyle, NodeStyle, Theme


@pytest.fixture(autouse=True)
def _reset_defaults() -> Iterator[None]:
    """Reset global defaults around each test."""
    reset()
    yield
    reset()


def test_graph_configure_sets_default_node_style() -> None:
    """`DaguaGraph.configure()` should populate node defaults."""
    graph = DaguaGraph()
    graph.add_node("a", label="A")

    graph.configure(overflow_policy="expand_node")

    assert graph.default_node_style is not None
    assert graph.default_node_style.overflow_policy == "expand_node"


def test_graph_configure_routes_edge_and_cluster_fields() -> None:
    """Flat kwargs should route to edge and cluster graph defaults."""
    graph = DaguaGraph.from_edge_list([("a", "b")])

    graph.configure(edge_width=2.0, color="#FF0000", cluster_padding=20.0)

    assert graph.default_edge_style is not None
    assert graph.default_edge_style.width == 2.0
    assert graph.default_edge_style.color == "#FF0000"
    assert graph.default_cluster_style is not None
    assert graph.default_cluster_style.padding == 20.0


def test_graph_default_node_style_cascades_to_render_and_node_sizing() -> None:
    """Graph-level node defaults should affect render lookup and sizing."""
    graph = DaguaGraph()
    graph._theme = Theme()
    graph.add_node("a", label="A")
    graph.default_node_style = NodeStyle(font_size=14.0, overflow_policy="expand_node")

    render_style = mpl_renderer._node_style_for_render(graph, 0)
    graph.compute_node_sizes()

    assert render_style.font_size == pytest.approx(14.0)
    assert render_style.overflow_policy == "expand_node"
    assert graph.node_font_sizes is not None
    assert float(graph.node_font_sizes[0].item()) == pytest.approx(14.0)


def test_per_node_overrides_graph_default() -> None:
    """Per-node style overrides should beat graph defaults."""
    graph = DaguaGraph()
    graph._theme = Theme()
    graph.add_node("a", label="A")
    graph.default_node_style = NodeStyle(font_size=14.0, overflow_policy="expand_node")
    graph.node_styles[0] = NodeStyle(font_size=8.0, overflow_policy="expand_node")

    render_style = mpl_renderer._node_style_for_render(graph, 0)
    graph.compute_node_sizes()

    assert render_style.font_size == pytest.approx(8.0)
    assert graph.node_font_sizes is not None
    assert float(graph.node_font_sizes[0].item()) == pytest.approx(8.0)


def test_graph_default_cluster_style_cascades_to_render() -> None:
    """Graph-level cluster defaults should fill fields the theme leaves unset."""
    graph = DaguaGraph()
    graph._theme = Theme(cluster_style=ClusterStyle())
    graph.add_node("a")
    graph.add_cluster("grp", members=["a"])
    graph.default_cluster_style = ClusterStyle(depth_fill_step=0.08)

    render_style = mpl_renderer._cluster_style_for_render(graph, "grp")

    assert render_style.depth_fill_step == pytest.approx(0.08)
