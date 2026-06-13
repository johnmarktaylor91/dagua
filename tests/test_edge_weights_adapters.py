"""Tests for edge-weight forwarding through evaluation adapters."""

from __future__ import annotations

import sys
import types
from typing import Any, Optional

import pytest
import torch

from dagua.eval.competitors.classic_competitor import _quick_classic
from dagua.eval.competitors.igraph_competitor import IgraphDRL
from dagua.eval.competitors.networkx_competitor import _graph_to_nx
from dagua.eval.competitors.sgd2_competitor import (
    _build_condensed_distances,
    _symmetrized_unique_edges,
)
from dagua.eval.competitors.tsne_competitor import _distance_matrix as _tsne_distance_matrix
from dagua.eval.competitors.umap_competitor import _distance_matrix as _umap_distance_matrix
from dagua.eval.graphs import _undirected_to_dag, get_test_graphs
from dagua.eval.variants import get_variant
from dagua.graph import DaguaGraph


def _weighted_path_graph() -> DaguaGraph:
    """Create a small weighted path graph for adapter regression tests.

    Returns
    -------
    DaguaGraph
        Three-node path with edge weights ``[2.0, 3.0]``.
    """
    graph = DaguaGraph()
    graph.add_edge(0, 1, weight=2.0)
    graph.add_edge(1, 2, weight=3.0)
    _ = graph.edge_index
    graph.compute_node_sizes()
    return graph


class TestWeightedTestGraphs:
    """Coverage for weighted evaluation graph registration."""

    def test_weighted_graphs_exist(self) -> None:
        """Weighted test graphs are registered."""
        graphs = get_test_graphs(tags={"weighted"})
        names = {graph.name for graph in graphs}
        assert "weighted_chain_20" in names
        assert "weighted_clusters_3x10" in names
        assert "weighted_karate_34" in names

    def test_weighted_chain_has_weights(self) -> None:
        """The weighted chain graph exposes non-uniform edge weights."""
        graphs = get_test_graphs(tags={"weighted"})
        chain = [graph for graph in graphs if graph.name == "weighted_chain_20"][0]
        assert chain.graph.edge_weights is not None
        assert chain.graph.edge_weights.shape[0] > 0
        assert chain.graph.edge_weights.min().item() < chain.graph.edge_weights.max().item()

    def test_weighted_clusters_has_weights(self) -> None:
        """The weighted clustered graph retains explicit edge weights."""
        graphs = get_test_graphs(tags={"weighted"})
        clusters = [graph for graph in graphs if graph.name == "weighted_clusters_3x10"][0]
        assert clusters.graph.edge_weights is not None

    def test_weighted_karate_has_weights(self) -> None:
        """The weighted karate graph preserves weights after DAG orientation."""
        graphs = get_test_graphs(tags={"weighted"})
        karate = [graph for graph in graphs if graph.name == "weighted_karate_34"][0]
        assert karate.graph.edge_weights is not None

    def test_undirected_to_dag_preserves_edge_weights(self) -> None:
        """Reorienting undirected edges keeps aligned edge weights."""
        graph = DaguaGraph()
        graph.add_edge(0, 1, weight=2.0)
        graph.add_edge(1, 0, weight=5.0)
        graph.add_edge(2, 1, weight=7.0)
        graph.compute_node_sizes()

        dag_graph = _undirected_to_dag(graph)

        assert dag_graph.edge_weights is not None
        assert dag_graph.edge_index.tolist() == [[0, 1], [1, 2]]
        assert dag_graph.edge_weights.tolist() == [2.0, 7.0]


class TestClassicAdapterWeightForwarding:
    """Coverage for classic-adapter weight forwarding."""

    def test_quick_classic_forwards_weights(self, monkeypatch: Any) -> None:
        """``_quick_classic`` passes ``edge_weights`` to the layout function."""
        graph = _weighted_path_graph()
        seen: dict[str, Optional[torch.Tensor]] = {"edge_weights": None}
        module_name = "tests.fake_weighted_layout_module"
        fake_module = types.ModuleType(module_name)

        def _layout_fake(
            edge_index: torch.Tensor,
            num_nodes: int,
            node_sizes: Optional[torch.Tensor] = None,
            seed: int = 42,
            edge_weights: Optional[torch.Tensor] = None,
            **kwargs: Any,
        ) -> torch.Tensor:
            """Capture forwarded kwargs and return a trivial layout tensor.

            Parameters
            ----------
            edge_index : torch.Tensor
                Input edge tensor.
            num_nodes : int
                Number of nodes in the graph.
            node_sizes : torch.Tensor | None, default=None
                Unused compatibility node sizes.
            seed : int, default=42
                Unused seed forwarded by the adapter.
            edge_weights : torch.Tensor | None, default=None
                Forwarded edge weights under test.
            **kwargs : Any
                Additional compatibility kwargs.

            Returns
            -------
            torch.Tensor
                Zero coordinates shaped ``[N, 2]``.
            """
            del edge_index, node_sizes, seed, kwargs
            seen["edge_weights"] = None if edge_weights is None else edge_weights.clone()
            return torch.zeros((num_nodes, 2), dtype=torch.float32)

        fake_module.layout_fake = _layout_fake
        monkeypatch.setitem(sys.modules, module_name, fake_module)

        result = _quick_classic(
            "classic_test",
            module_name,
            "layout_fake",
            graph,
            seed=42,
            steps=5,
        )

        assert result.pos is not None
        assert result.error is None
        assert seen["edge_weights"] is not None
        assert seen["edge_weights"].tolist() == [2.0, 3.0]


class TestExternalAdapterWeightEncoding:
    """Coverage for non-classic adapter preprocessing helpers."""

    def test_igraph_drl_forwards_weight_attribute_name(self, monkeypatch: Any) -> None:
        """The igraph DrL adapter should pass weighted graphs via ``weights``."""
        graph = _weighted_path_graph()
        seen: dict[str, Any] = {}

        class _FakeEdgeSeq:
            """Minimal edge-sequence stand-in that stores edge attributes."""

            def __init__(self) -> None:
                """Create empty edge-attribute storage."""
                self._attrs: dict[str, list[float]] = {}

            def __setitem__(self, key: str, value: list[float]) -> None:
                """Store an edge attribute list.

                Parameters
                ----------
                key : str
                    Attribute name.
                value : list[float]
                    Attribute values aligned with graph edges.

                Returns
                -------
                None
                    The attribute store is updated in place.
                """
                self._attrs[key] = value

            def attribute_names(self) -> list[str]:
                """Return stored edge-attribute names.

                Returns
                -------
                list[str]
                    Attribute names assigned by the adapter.
                """
                return list(self._attrs)

        class _FakeGraph:
            """Minimal igraph.Graph replacement for adapter keyword capture."""

            def __init__(self, directed: bool = True) -> None:
                """Create a fake graph.

                Parameters
                ----------
                directed : bool, default=True
                    Whether the graph should be treated as directed.
                """
                self.directed = directed
                self.es = _FakeEdgeSeq()

            def add_vertices(self, count: int) -> None:
                """Record the requested vertex count.

                Parameters
                ----------
                count : int
                    Number of vertices to add.

                Returns
                -------
                None
                    The count is stored for inspection.
                """
                seen["vertices"] = count

            def add_edges(self, edges: list[tuple[int, int]]) -> None:
                """Record the requested edge list.

                Parameters
                ----------
                edges : list[tuple[int, int]]
                    Edges to add.

                Returns
                -------
                None
                    The edge list is stored for inspection.
                """
                seen["edges"] = edges

            def layout(self, algo: str, **kwargs: Any) -> list[list[float]]:
                """Capture layout kwargs and return trivial coordinates.

                Parameters
                ----------
                algo : str
                    igraph layout algorithm name.
                **kwargs : Any
                    Layout keyword arguments forwarded by the adapter.

                Returns
                -------
                list[list[float]]
                    Coordinates shaped ``[N, 2]``.
                """
                seen["algo"] = algo
                seen["kwargs"] = kwargs
                return [[0.0, 0.0] for _ in range(seen["vertices"])]

        def _set_random_number_generator(generator: object) -> None:
            """Capture igraph RNG hook calls.

            Parameters
            ----------
            generator : object
                RNG object or ``None`` requested by the adapter.

            Returns
            -------
            None
                The call count is tracked in ``seen``.
            """
            seen["rng_calls"] = int(seen.get("rng_calls", 0)) + 1

        fake_igraph = types.SimpleNamespace(
            Graph=_FakeGraph,
            set_random_number_generator=_set_random_number_generator,
        )
        monkeypatch.setitem(sys.modules, "igraph", fake_igraph)

        result = IgraphDRL().layout(graph, seed=42)

        assert result.error is None
        assert seen["algo"] == "drl"
        assert seen["kwargs"]["weights"] == "weight"

    def test_graph_to_nx_sets_weight_attribute(self) -> None:
        """NetworkX conversion copies edge weights to the ``weight`` attribute."""
        graph = _weighted_path_graph()

        nx_graph = _graph_to_nx(graph)

        assert nx_graph[0][1]["weight"] == 2.0
        assert nx_graph[1][2]["weight"] == 3.0

    def test_sgd2_symmetrized_edges_include_weights(self) -> None:
        """The s_gd2 preprocessing helper returns summed symmetrized weights."""
        graph = DaguaGraph()
        graph.add_edge(0, 1, weight=2.0)
        graph.add_edge(1, 0, weight=3.0)
        graph.compute_node_sizes()

        sources, targets, weights = _symmetrized_unique_edges(graph)

        assert sources.tolist() == [0, 1]
        assert targets.tolist() == [1, 0]
        assert weights is not None
        assert weights.tolist() == [5.0, 5.0]

    def test_sgd2_condensed_distances_use_edge_weights(self) -> None:
        """Weighted shortest-path distances feed the s_gd2 MDS helper."""
        graph = _weighted_path_graph()

        distances, weights = _build_condensed_distances(graph)

        assert distances.tolist() == [2.0, 5.0, 3.0]
        assert weights.tolist() == pytest.approx([0.25, 0.04, 1.0 / 9.0])

    def test_tsne_distance_matrix_uses_edge_weights(self) -> None:
        """t-SNE shortest-path preprocessing uses weighted adjacency."""
        graph = _weighted_path_graph()

        distances = _tsne_distance_matrix(graph)

        assert distances.tolist() == [[0.0, 2.0, 5.0], [2.0, 0.0, 3.0], [5.0, 3.0, 0.0]]

    def test_umap_distance_matrix_uses_edge_weights(self) -> None:
        """UMAP shortest-path preprocessing uses weighted adjacency."""
        graph = _weighted_path_graph()

        distances = _umap_distance_matrix(graph)

        assert distances.tolist() == [[0.0, 2.0, 5.0], [2.0, 0.0, 3.0], [5.0, 3.0, 0.0]]


class TestNewVariants:
    """Coverage for the newly registered ForceAtlas2 variants."""

    def test_fa2_dissuade_hubs_variant_exists(self) -> None:
        """The FA2 dissuade-hubs variant is registered."""
        variant = get_variant("classic_fa2_dissuade_hubs")
        assert variant is not None
        assert variant.reimpl_params["dissuade_hubs"] is True

    def test_fa2_linlog_variant_exists(self) -> None:
        """The FA2 linlog variant is registered."""
        variant = get_variant("classic_fa2_linlog")
        assert variant is not None
        assert variant.reimpl_params["linlog"] is True

    def test_fa2_barnes_hut_variant_exists(self) -> None:
        """The FA2 Barnes-Hut variant is registered."""
        variant = get_variant("classic_fa2_barnes_hut")
        assert variant is not None
        assert variant.reimpl_params["barnes_hut"] is True

    def test_fa2_exact_variant_exists(self) -> None:
        """The FA2 exact variant is registered."""
        variant = get_variant("classic_fa2_exact")
        assert variant is not None
        assert variant.reimpl_params["barnes_hut"] is False
