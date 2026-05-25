"""Tests for embedding-based competitor adapters."""

from __future__ import annotations

import pytest
import torch

from dagua.eval.competitors import get_competitors, neulay_competitor
from dagua.eval.competitors.neulay_competitor import NeuLayReference
from dagua.eval.competitors.sgd2_competitor import SGD2, SGD2MDS
from dagua.eval.competitors.umap_competitor import UMAPGraph
from dagua.graph import DaguaGraph

NEULAY_AVAILABLE = NeuLayReference().available()
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
    assert {"sgd2", "sgd2_mds", "neulay", "tsne_graph", "umap_graph"} <= names


@pytest.mark.smoke
class TestNeuLay:
    """Smoke coverage for the recovered NeuLay competitor."""

    def test_available_check(self) -> None:
        """The availability probe should return a boolean.

        Returns
        -------
        None
            This test asserts on the availability probe result.
        """
        competitor = NeuLayReference()
        assert isinstance(competitor.available(), bool)

    def test_available_is_false_without_upstream_reference(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The adapter should surface recovered-reference import failures."""
        monkeypatch.setattr(neulay_competitor, "_load_upstream_neulay", lambda: None)
        competitor = NeuLayReference()
        assert competitor.available() is False

    def test_layout_uses_upstream_reference_when_present(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The adapter should forward full NeuLay defaults to the upstream callable."""
        graph = _make_small_graph()
        observed: dict[str, object] = {}

        def _fake_upstream(
            edge_index: torch.Tensor,
            num_nodes: int,
            *,
            node_sizes: object,
            seed: int,
            steps: int,
            gcn_steps: int,
            use_gcn: bool,
            lr: float,
            radius: float,
        ) -> torch.Tensor:
            """Capture NeuLay adapter arguments for the regression test.

            Parameters
            ----------
            edge_index : torch.Tensor
                Graph connectivity tensor with shape ``[2, E]``.
            num_nodes : int
                Number of graph nodes.
            node_sizes : object
                Node-size payload forwarded by the adapter.
            seed : int
                Random seed forwarded by the adapter.
            steps : int
                Total NeuLay optimization budget.
            gcn_steps : int
                GCN warm-start optimization budget.
            use_gcn : bool
                Whether the GCN phase is enabled.
            lr : float
                Direct-refinement learning rate.
            radius : float
                Gaussian repulsion radius.

            Returns
            -------
            torch.Tensor
                Zero-valued position tensor with shape ``[N, 2]``.
            """
            observed["edge_shape"] = tuple(edge_index.shape)
            observed["num_nodes"] = num_nodes
            observed["node_sizes"] = node_sizes
            observed["seed"] = seed
            observed["steps"] = steps
            observed["gcn_steps"] = gcn_steps
            observed["use_gcn"] = use_gcn
            observed["lr"] = lr
            observed["radius"] = radius
            return torch.zeros((num_nodes, 2), dtype=torch.float32)

        monkeypatch.setattr(neulay_competitor, "_load_upstream_neulay", lambda: _fake_upstream)

        result = NeuLayReference().layout(graph, seed=9)

        assert result.error is None
        assert result.pos is not None
        assert observed["edge_shape"] == tuple(graph.edge_index.shape)
        assert observed["num_nodes"] == graph.num_nodes
        assert observed["seed"] == 9
        assert observed["steps"] == 20_000
        assert observed["gcn_steps"] == 2_000
        assert observed["use_gcn"] is True
        assert observed["lr"] == 0.1
        assert observed["radius"] == 0.4

    @pytest.mark.skipif(
        not NEULAY_AVAILABLE,
        reason="recovered NeuLay reference unavailable",
    )
    def test_layout_returns_positions(self) -> None:
        """The adapter should return positions for a small graph.

        Returns
        -------
        None
            This test asserts on the returned position tensor.
        """
        graph = _make_small_graph()
        result = NeuLayReference().layout_with_variant(
            graph,
            timeout=30.0,
            variant_params={"steps": 4, "gcn_steps": 0, "use_gcn": False},
        )
        assert result.pos is not None
        assert result.pos.shape == (6, 2)
        assert result.error is None


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
class TestSGD2MDS:
    """Smoke coverage for the ``s_gd2`` classical MDS adapter."""

    def test_available_check(self) -> None:
        """The availability probe should return a boolean.

        Returns
        -------
        None
            This test asserts on the availability probe result.
        """
        competitor = SGD2MDS()
        assert isinstance(competitor.available(), bool)

    @pytest.mark.skipif(
        not SGD2_AVAILABLE,
        reason="s_gd2 not installed",
    )
    def test_layout_returns_positions(self) -> None:
        """The adapter should return 2D positions for a connected graph.

        Returns
        -------
        None
            This test asserts on the returned position tensor.
        """
        graph = _make_small_graph()
        result = SGD2MDS().layout(graph, timeout=30.0)
        assert result.pos is not None
        assert result.pos.shape == (6, 2)
        assert result.error is None

    @pytest.mark.skipif(
        not SGD2_AVAILABLE,
        reason="s_gd2 not installed",
    )
    def test_disconnected_graph_returns_error(self) -> None:
        """Disconnected graphs should surface the MDS precondition failure.

        Returns
        -------
        None
            This test asserts on the adapter's error payload.
        """
        graph = DaguaGraph()
        graph.add_node("0")
        graph.add_node("1")
        result = SGD2MDS().layout(graph, timeout=30.0)
        assert result.pos is None
        assert result.error == "graph is disconnected"


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

    @pytest.mark.skipif(
        not UMAP_AVAILABLE,
        reason="umap not installed or unusable",
    )
    def test_tiny_graph_layout_is_seeded(self) -> None:
        """Tiny-graph fallback should use the benchmark seed."""
        graph = DaguaGraph()
        graph.add_node("0")
        graph.add_node("1")
        graph.add_node("2")
        graph.add_edge("0", "1")

        competitor = UMAPGraph()
        first = competitor.layout(graph, seed=5)
        second = competitor.layout(graph, seed=5)
        third = competitor.layout(graph, seed=6)

        assert first.pos is not None
        assert second.pos is not None
        assert third.pos is not None
        torch.testing.assert_close(first.pos, second.pos)
        assert not torch.allclose(first.pos, third.pos)


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
