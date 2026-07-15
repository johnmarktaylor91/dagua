"""Pipeline pins for the NetworkX multipartite layout."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import dagua
from dagua.config import LayoutConfig
from dagua.layout.ops.networkx_simple import nx_bfs_layers
from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.multipartite import (
    build_multipartite_pipeline,
    layout_multipartite_pipeline,
)


def _path_edges() -> torch.Tensor:
    """Return a fixed four-node path edge tensor.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, 3]``.
    """
    return torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)


def test_multipartite_pipeline_is_registered() -> None:
    """Register the public multipartite algorithm.

    Returns
    -------
    None
        Registry lookups must resolve the multipartite entrypoint.
    """
    assert PIPELINE_REGISTRY["multipartite"] == (
        "dagua.layout.ops.pipelines.multipartite",
        "layout_multipartite_pipeline",
    )
    assert get_pipeline_function("MULTIPARTITE") is layout_multipartite_pipeline
    assert [operation.name for operation in build_multipartite_pipeline().ops] == [
        "networkx_simple_layout"
    ]


def test_multipartite_matches_networkx_reference() -> None:
    """Match NetworkX multipartite coordinates with pinned BFS layers.

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
    layers = nx_bfs_layers(edge_index, 4)
    expected = np.vstack(
        [nx.multipartite_layout(graph, subset_key=layers)[node] for node in range(4)]
    )

    actual = layout_multipartite_pipeline(edge_index, 4, layers=layers).numpy()

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-12)


def test_layout_config_algorithm_multipartite_dispatches() -> None:
    """Exercise public engine dispatch for ``algorithm='multipartite'``.

    Returns
    -------
    None
        Dispatch must return finite coordinates.
    """
    graph = dagua.DaguaGraph.from_edge_list([("a", "b"), ("b", "c")])
    positions = dagua.layout(graph, LayoutConfig(algorithm="multipartite"))

    assert positions.shape == (3, 2)
    assert torch.isfinite(positions).all()
