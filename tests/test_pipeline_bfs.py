"""Pipeline pins for the NetworkX BFS layout."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import dagua
from dagua.config import LayoutConfig
from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.bfs import build_bfs_pipeline, layout_bfs_pipeline


def _path_edges() -> torch.Tensor:
    """Return a fixed four-node path edge tensor.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, 3]``.
    """
    return torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)


def test_bfs_pipeline_is_registered() -> None:
    """Register the public BFS algorithm.

    Returns
    -------
    None
        Registry lookups must resolve the BFS entrypoint.
    """
    assert PIPELINE_REGISTRY["bfs"] == (
        "dagua.layout.ops.pipelines.bfs",
        "layout_bfs_pipeline",
    )
    assert get_pipeline_function("BFS") is layout_bfs_pipeline
    assert [operation.name for operation in build_bfs_pipeline().ops] == ["networkx_simple_layout"]


def test_bfs_matches_networkx_reference() -> None:
    """Match NetworkX BFS coordinates on a connected small graph.

    Returns
    -------
    None
        Dagua's source port must match NetworkX exactly.
    """
    nx = pytest.importorskip("networkx")
    graph = nx.DiGraph()
    graph.add_nodes_from(range(4))
    graph.add_edges_from([(0, 1), (1, 2), (2, 3)])
    expected = np.vstack([nx.bfs_layout(graph, 0)[node] for node in range(4)])

    actual = layout_bfs_pipeline(_path_edges(), 4).numpy()

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-12)


def test_layout_config_algorithm_bfs_dispatches() -> None:
    """Exercise public engine dispatch for ``algorithm='bfs'``.

    Returns
    -------
    None
        Dispatch must return finite coordinates.
    """
    graph = dagua.DaguaGraph.from_edge_list([("a", "b"), ("b", "c")])
    positions = dagua.layout(graph, LayoutConfig(algorithm="bfs"))

    assert positions.shape == (3, 2)
    assert torch.isfinite(positions).all()
