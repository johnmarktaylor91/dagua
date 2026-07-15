"""Unit tests for the standalone planar layout ops."""

from __future__ import annotations

import ast
import inspect

import networkx as nx
import pytest
import torch

from dagua.layout.ops.pipelines import planar
from dagua.layout.ops.pipelines.planar import (
    PlanarityError,
    check_planarity,
    combinatorial_embedding_to_pos,
    layout_planar_pipeline,
)


def _edge_index_from_graph(graph: nx.Graph) -> torch.Tensor:
    """Return an edge tensor from a NetworkX graph.

    Parameters
    ----------
    graph : nx.Graph
        Graph with integer node ids.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    edges = list(graph.edges())
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def test_planarity_embedding_matches_networkx_cycle_grid_and_k4() -> None:
    """Left-Right planarity should match the reference cyclic adjacency order."""
    graphs = [
        nx.cycle_graph(6),
        nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 3)),
        nx.complete_graph(4),
    ]

    for graph in graphs:
        edge_index = _edge_index_from_graph(graph)
        is_planar, embedding = check_planarity(edge_index, graph.number_of_nodes())
        reference_is_planar, reference_embedding = nx.check_planarity(graph)

        assert is_planar == reference_is_planar
        assert embedding is not None
        assert embedding.get_data() == reference_embedding.get_data()


def test_shift_method_raw_integer_positions_match_networkx() -> None:
    """Chrobak-Payne raw grid coordinates should match the reference helper."""
    graph = nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 3))
    edge_index = _edge_index_from_graph(graph)
    _, embedding = check_planarity(edge_index, graph.number_of_nodes())
    _, reference_embedding = nx.check_planarity(graph)

    assert embedding is not None
    assert combinatorial_embedding_to_pos(embedding) == nx.combinatorial_embedding_to_pos(
        reference_embedding
    )


def test_non_planar_graph_raises_precise_reason() -> None:
    """K5 should be rejected with the public non-planar reason."""
    graph = nx.complete_graph(5)

    with pytest.raises(PlanarityError, match="G is not planar\\."):
        layout_planar_pipeline(_edge_index_from_graph(graph), graph.number_of_nodes())


def test_planar_pipeline_has_no_runtime_networkx_delegation() -> None:
    """The runtime planar pipeline must not import or call NetworkX."""
    source = inspect.getsource(planar)
    tree = ast.parse(source)
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and any(alias.name == "networkx" for alias in node.names)
    ]

    assert imports == []
    assert "import networkx" not in source
    assert "nx." not in source
