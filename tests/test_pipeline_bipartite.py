"""Pipeline pins for the NetworkX bipartite layout."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import dagua
from dagua.config import LayoutConfig
from dagua.layout.ops.networkx_simple import nx_bipartite_node_set
from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.bipartite import build_bipartite_pipeline, layout_bipartite_pipeline


def _path_edges() -> torch.Tensor:
    """Return a fixed four-node path edge tensor.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, 3]``.
    """
    return torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)


def test_bipartite_pipeline_is_registered() -> None:
    """Register the public bipartite algorithm.

    Returns
    -------
    None
        Registry lookups must resolve the bipartite entrypoint.
    """
    assert PIPELINE_REGISTRY["bipartite"] == (
        "dagua.layout.ops.pipelines.bipartite",
        "layout_bipartite_pipeline",
    )
    assert get_pipeline_function("BIPARTITE") is layout_bipartite_pipeline
    assert [operation.name for operation in build_bipartite_pipeline().ops] == [
        "networkx_simple_layout"
    ]


def test_bipartite_matches_networkx_reference() -> None:
    """Match NetworkX bipartite coordinates with the pinned node set.

    Returns
    -------
    None
        Dagua's source port must match NetworkX exactly.
    """
    nx = pytest.importorskip("networkx")
    edge_index = _path_edges()
    graph = nx.DiGraph()
    graph.add_nodes_from(range(4))
    graph.add_edges_from([(0, 1), (1, 2), (2, 3)])
    nodes = nx_bipartite_node_set(edge_index, 4)
    expected = np.vstack([nx.bipartite_layout(graph, nodes=nodes)[node] for node in range(4)])

    actual = layout_bipartite_pipeline(edge_index, 4, nodes=nodes).numpy()

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-12)


def test_layout_config_algorithm_bipartite_dispatches() -> None:
    """Exercise public engine dispatch for ``algorithm='bipartite'``.

    Returns
    -------
    None
        Dispatch must return finite coordinates.
    """
    graph = dagua.DaguaGraph.from_edge_list([("a", "b"), ("b", "c")])
    positions = dagua.layout(graph, LayoutConfig(algorithm="bipartite"))

    assert positions.shape == (3, 2)
    assert torch.isfinite(positions).all()
