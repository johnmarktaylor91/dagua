"""Tests for igraph competitor adapters."""

import pytest

igraph = pytest.importorskip("igraph")

from dagua.graph import DaguaGraph
from dagua.eval.competitors.igraph_competitor import (
    IgraphSugiyama,
    IgraphFR,
    IgraphRT,
    _graph_to_igraph,
    _igraph_pos_to_tensor,
)


def _make_simple_graph():
    g = DaguaGraph()
    for i in range(5):
        g.add_node(i)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(3, 4)
    g.add_edge(0, 2)
    return g


class TestIgraphConversion:
    def test_graph_to_igraph(self):
        g = _make_simple_graph()
        ig = _graph_to_igraph(g)
        assert ig.vcount() == 5
        assert ig.ecount() == 5

    def test_pos_to_tensor(self):
        layout = [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]
        pos = _igraph_pos_to_tensor(layout, 3)
        assert pos.shape == (3, 2)
        assert pos[0, 0].item() == pytest.approx(0.0)
        assert pos[1, 0].item() == pytest.approx(100.0)


class TestIgraphCompetitors:
    def test_sugiyama_available(self):
        comp = IgraphSugiyama()
        assert comp.available()
        assert comp.name == "igraph_sugiyama"

    def test_sugiyama_layout(self):
        g = _make_simple_graph()
        comp = IgraphSugiyama()
        result = comp.layout(g)
        assert result.pos is not None
        assert result.pos.shape == (5, 2)
        assert result.error is None
        assert result.runtime_seconds > 0

    def test_fr_layout(self):
        g = _make_simple_graph()
        comp = IgraphFR()
        result = comp.layout(g)
        assert result.pos is not None
        assert result.pos.shape == (5, 2)

    def test_rt_layout(self):
        g = _make_simple_graph()
        comp = IgraphRT()
        result = comp.layout(g)
        assert result.pos is not None
        assert result.pos.shape == (5, 2)


class TestIgraphRegistration:
    def test_registered_in_competitors(self):
        from dagua.eval.competitors import get_available_competitors

        names = [c.name for c in get_available_competitors()]
        assert "igraph_sugiyama" in names
        assert "igraph_fr" in names
        assert "igraph_rt" in names
