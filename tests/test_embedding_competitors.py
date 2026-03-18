"""Tests for embedding-based competitor adapters."""

from __future__ import annotations

import pytest
import torch

from dagua.eval.competitors import get_competitors
from dagua.eval.competitors.sgd2_competitor import SGD2
from dagua.eval.competitors.umap_competitor import UMAPGraph
from dagua.graph import DaguaGraph

SGD2_AVAILABLE = SGD2().available()
UMAP_AVAILABLE = UMAPGraph().available()


def _make_small_graph() -> DaguaGraph:
    """Create a small connected graph for embedding competitors.

    Returns
    -------
    DaguaGraph
        Six-node chain graph.
    """
    graph = DaguaGraph()
    for node_idx in range(6):
        graph.add_node(str(node_idx), label=str(node_idx))
    for node_idx in range(5):
        graph.add_edge(str(node_idx), str(node_idx + 1))
    return graph


@pytest.mark.smoke
def test_embedding_competitors_registered() -> None:
    """The new embedding competitors should be registered on import.

    Returns
    -------
    None
        This test asserts on the global competitor registry contents.
    """
    names = {competitor.name for competitor in get_competitors()}
    assert {"sgd2", "tsne_graph", "umap_graph"} <= names


@pytest.mark.smoke
class TestSGD2:
    """Smoke coverage for the ``s_gd2`` competitor adapter."""

    def test_available_check(self) -> None:
        """The availability probe should return a boolean.

        Returns
        -------
        None
            This test asserts on the availability probe result.
        """
        competitor = SGD2()
        assert isinstance(competitor.available(), bool)

    @pytest.mark.skipif(
        not SGD2_AVAILABLE,
        reason="s_gd2 not installed",
    )
    def test_layout_returns_positions(self) -> None:
        """The adapter should return 2D positions for a small graph.

        Returns
        -------
        None
            This test asserts on the returned position tensor.
        """
        graph = _make_small_graph()
        result = SGD2().layout(graph, timeout=30.0)
        assert result.pos is not None
        assert result.pos.shape == (6, 2)
        assert result.error is None


@pytest.mark.smoke
class TestTSNEGraph:
    """Smoke coverage for the graph-distance t-SNE competitor."""

    def test_layout_returns_positions(self) -> None:
        """The adapter should return 2D positions for a small graph.

        Returns
        -------
        None
            This test asserts on the returned position tensor.
        """
        from dagua.eval.competitors.tsne_competitor import TSNEGraph

        graph = _make_small_graph()
        result = TSNEGraph().layout(graph, timeout=60.0)
        assert result.pos is not None
        assert result.pos.shape == (6, 2)
        assert result.error is None
        assert result.pos.dtype == torch.float32


@pytest.mark.smoke
class TestUMAPGraph:
    """Smoke coverage for the graph-distance UMAP competitor."""

    def test_available_check(self) -> None:
        """The availability probe should return a boolean.

        Returns
        -------
        None
            This test asserts on the availability probe result.
        """
        competitor = UMAPGraph()
        assert isinstance(competitor.available(), bool)

    @pytest.mark.skipif(
        not UMAP_AVAILABLE,
        reason="umap not installed or unusable",
    )
    def test_layout_returns_positions(self) -> None:
        """The adapter should return 2D positions for a small graph.

        Returns
        -------
        None
            This test asserts on the returned position tensor.
        """
        graph = _make_small_graph()
        result = UMAPGraph().layout(graph, timeout=60.0)
        assert result.pos is not None
        assert result.pos.shape == (6, 2)
        assert result.error is None
        assert result.pos.dtype == torch.float32
