"""Tests for igraph competitor adapters."""

import pytest
import torch

igraph = pytest.importorskip("igraph")

from dagua.eval.competitors.igraph_competitor import (  # noqa: E402
    IgraphDavidsonHarel,
    IgraphDRL,
    IgraphFR,
    IgraphKamadaKawai,
    IgraphLGL,
    IgraphMDS,
    IgraphRT,
    IgraphSugiyama,
    _graph_to_igraph,
    _igraph_pos_to_tensor,
)
from dagua.graph import DaguaGraph  # noqa: E402


def _make_simple_graph() -> DaguaGraph:
    """Create a small connected graph for igraph adapter tests.

    Returns
    -------
    DaguaGraph
        Five-node graph with one shortcut edge.
    """
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
    def test_graph_to_igraph(self) -> None:
        """The conversion helper should preserve node and edge counts.

        Returns
        -------
        None
            This test asserts on the converted igraph structure.
        """
        g = _make_simple_graph()
        ig = _graph_to_igraph(g)
        assert ig.vcount() == 5
        assert ig.ecount() == 5

    def test_pos_to_tensor(self) -> None:
        """The coordinate helper should scale igraph units into tensors.

        Returns
        -------
        None
            This test asserts on the converted tensor contents.
        """
        layout = [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]
        pos = _igraph_pos_to_tensor(layout, 3)
        assert pos.shape == (3, 2)
        assert pos[0, 0].item() == pytest.approx(0.0)
        assert pos[1, 0].item() == pytest.approx(100.0)


class TestIgraphCompetitors:
    def test_sugiyama_available(self) -> None:
        """The Sugiyama adapter should report igraph availability.

        Returns
        -------
        None
            This test asserts on the availability probe.
        """
        comp = IgraphSugiyama()
        assert comp.available()
        assert comp.name == "igraph_sugiyama"

    def test_sugiyama_layout(self) -> None:
        """The Sugiyama adapter should return positions on a small graph.

        Returns
        -------
        None
            This test asserts on the layout result payload.
        """
        g = _make_simple_graph()
        comp = IgraphSugiyama()
        result = comp.layout(g)
        assert result.pos is not None
        assert result.pos.shape == (5, 2)
        assert result.error is None
        assert result.runtime_seconds > 0

    def test_fr_layout(self) -> None:
        """The FR adapter should return positions on a small graph.

        Returns
        -------
        None
            This test asserts on the layout result payload.
        """
        g = _make_simple_graph()
        comp = IgraphFR()
        result = comp.layout(g)
        assert result.pos is not None
        assert result.pos.shape == (5, 2)

    def test_rt_layout(self) -> None:
        """The Reingold-Tilford adapter should return positions.

        Returns
        -------
        None
            This test asserts on the layout result payload.
        """
        g = _make_simple_graph()
        comp = IgraphRT()
        result = comp.layout(g)
        assert result.pos is not None
        assert result.pos.shape == (5, 2)

    def test_davidson_harel_layout(self) -> None:
        """The Davidson-Harel adapter should return positions.

        Returns
        -------
        None
            This test asserts on the layout result payload.
        """
        g = _make_simple_graph()
        comp = IgraphDavidsonHarel()
        result = comp.layout(g)
        assert result.pos is not None
        assert result.pos.shape == (5, 2)

    def test_kamada_kawai_layout(self) -> None:
        """The Kamada-Kawai adapter should return positions.

        Returns
        -------
        None
            This test asserts on the layout result payload.
        """
        g = _make_simple_graph()
        comp = IgraphKamadaKawai()
        result = comp.layout(g)
        assert result.pos is not None
        assert result.pos.shape == (5, 2)

    def test_mds_layout(self) -> None:
        """The MDS adapter should return positions.

        Returns
        -------
        None
            This test asserts on the layout result payload.
        """
        g = _make_simple_graph()
        comp = IgraphMDS()
        result = comp.layout(g)
        assert result.pos is not None
        assert result.pos.shape == (5, 2)

    def test_drl_layout_is_reproducible_for_same_seed(self) -> None:
        """The DRL adapter should seed igraph's internal RNG."""
        g = _make_simple_graph()
        comp = IgraphDRL()

        first = comp.layout(g, seed=7)
        second = comp.layout(g, seed=7)
        third = comp.layout(g, seed=8)

        assert first.pos is not None
        assert second.pos is not None
        assert third.pos is not None
        torch.testing.assert_close(first.pos, second.pos)
        assert not torch.allclose(first.pos, third.pos)

    def test_lgl_layout_is_reproducible_for_same_seed(self) -> None:
        """The LGL adapter should seed igraph's internal RNG."""
        g = _make_simple_graph()
        comp = IgraphLGL()

        first = comp.layout(g, seed=11)
        second = comp.layout(g, seed=11)
        third = comp.layout(g, seed=12)

        assert first.pos is not None
        assert second.pos is not None
        assert third.pos is not None
        torch.testing.assert_close(first.pos, second.pos)
        assert not torch.allclose(first.pos, third.pos)


class TestIgraphRegistration:
    def test_registered_in_competitors(self) -> None:
        """The igraph adapters should be discoverable through the registry.

        Returns
        -------
        None
            This test asserts on the available competitor names.
        """
        from dagua.eval.competitors import get_available_competitors

        names = [c.name for c in get_available_competitors()]
        assert "igraph_sugiyama" in names
        assert "igraph_fr" in names
        assert "igraph_rt" in names
        assert "igraph_davidson_harel" in names
        assert "igraph_kamada_kawai" in names
        assert "igraph_mds" in names
