"""Tests for classic competitor adapters."""

from __future__ import annotations

import pytest

from dagua.eval.competitors import get_available_competitors
from dagua.graph import DaguaGraph

EXPECTED_CLASSIC_NAMES = {
    "classic_fr",
    "classic_kk",
    "classic_fa2",
    "classic_stress_sgd",
    "classic_sugiyama",
    "classic_spectral",
    "classic_pivot_mds",
    "classic_linlog",
    "classic_gem",
    "classic_tsnet",
    "classic_maxent_stress",
    "classic_davidson_harel",
    "classic_fmmm",
}


def _make_small_graph() -> DaguaGraph:
    """Create a connected 10-node graph suitable for all classic layouts.

    Returns
    -------
    DaguaGraph
        Graph with computed node sizes.
    """
    edges = [(f"node_{index}", f"node_{index + 1}") for index in range(9)]
    graph = DaguaGraph.from_edge_list(edges)
    graph.compute_node_sizes()
    if graph.node_sizes is None:
        raise AssertionError("node sizes should be available after compute_node_sizes()")
    return graph


def _make_clustered_graph() -> DaguaGraph:
    """Create a small clustered graph for compound-layout competitors.

    Returns
    -------
    DaguaGraph
        Four-node path graph with two sibling clusters.
    """
    graph = DaguaGraph()
    graph.add_node("a")
    graph.add_node("b")
    graph.add_node("c")
    graph.add_node("d")
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")
    graph.add_edge("c", "d")
    graph.add_cluster("group1", ["a", "b"])
    graph.add_cluster("group2", ["c", "d"])
    graph.compute_node_sizes()
    if graph.node_sizes is None:
        raise AssertionError("node sizes should be available after compute_node_sizes()")
    return graph


def test_classic_competitors_are_discoverable() -> None:
    """All classic competitors should appear in the available registry."""
    competitor_names = {competitor.name for competitor in get_available_competitors()}
    assert EXPECTED_CLASSIC_NAMES.issubset(competitor_names)


def test_classic_competitor_names_match_expected_values() -> None:
    """The registered classic competitor names should match the expected set."""
    classic_names = {
        competitor.name
        for competitor in get_available_competitors()
        if competitor.name.startswith("classic_")
    }
    assert classic_names == EXPECTED_CLASSIC_NAMES


def test_each_classic_competitor_produces_a_valid_result() -> None:
    """Each classic competitor should return a successful layout result."""
    graph = _make_small_graph()
    competitors = {
        competitor.name: competitor
        for competitor in get_available_competitors()
        if competitor.name in EXPECTED_CLASSIC_NAMES
    }

    for competitor_name in EXPECTED_CLASSIC_NAMES:
        result = competitors[competitor_name].layout(graph)
        assert result.name == competitor_name
        assert result.error is None
        assert result.pos is not None
        assert result.runtime_seconds >= 0.0


def test_classic_competitor_positions_have_expected_shape() -> None:
    """Classic competitor layouts should return position tensors with shape ``[N, 2]``."""
    graph = _make_small_graph()
    competitors = {
        competitor.name: competitor
        for competitor in get_available_competitors()
        if competitor.name in EXPECTED_CLASSIC_NAMES
    }

    for competitor_name in EXPECTED_CLASSIC_NAMES:
        result = competitors[competitor_name].layout(graph)
        assert result.pos is not None
        assert tuple(result.pos.shape) == (graph.num_nodes, 2)


def test_graphviz_dot_with_clusters() -> None:
    """Graphviz dot should lay out clustered graphs without dropping nodes.

    Returns
    -------
    None
        The assertion validates clustered layout output.
    """
    graph = _make_clustered_graph()

    from dagua.eval.competitors.graphviz_competitor import GraphvizDot

    competitor = GraphvizDot()
    if not competitor.available():
        pytest.skip("Graphviz dot is not installed")

    result = competitor.layout(graph)
    assert result.error is None
    assert result.pos is not None
    assert tuple(result.pos.shape) == (4, 2)


def test_supports_clusters_flag() -> None:
    """Cluster-capable competitors should be explicitly marked.

    Returns
    -------
    None
        The assertion validates the registry metadata.
    """
    from dagua.eval.competitors import get_competitors

    cluster_names = {
        competitor.name for competitor in get_competitors() if competitor.supports_clusters
    }
    assert {"graphviz_dot", "elk_layered", "dagre"}.issubset(cluster_names)
