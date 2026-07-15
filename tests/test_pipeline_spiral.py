"""Pipeline pins for the NetworkX spiral layout."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import dagua
from dagua.config import LayoutConfig
from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.spiral import build_spiral_pipeline, layout_spiral_pipeline


def _path_edges() -> torch.Tensor:
    """Return a fixed four-node path edge tensor.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, 3]``.
    """
    return torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)


def test_spiral_pipeline_is_registered() -> None:
    """Register the public spiral algorithm.

    Returns
    -------
    None
        Registry lookups must resolve the spiral entrypoint.
    """
    assert PIPELINE_REGISTRY["spiral"] == (
        "dagua.layout.ops.pipelines.spiral",
        "layout_spiral_pipeline",
    )
    assert get_pipeline_function("SPIRAL") is layout_spiral_pipeline
    assert [operation.name for operation in build_spiral_pipeline().ops] == [
        "networkx_simple_layout"
    ]


def test_spiral_matches_networkx_reference() -> None:
    """Match NetworkX spiral coordinates on a small graph.

    Returns
    -------
    None
        Dagua's source port must match NetworkX exactly.
    """
    nx = pytest.importorskip("networkx")
    graph = nx.DiGraph()
    graph.add_nodes_from(range(4))
    graph.add_edges_from([(0, 1), (1, 2), (2, 3)])
    expected = np.vstack([nx.spiral_layout(graph)[node] for node in range(4)])

    actual = layout_spiral_pipeline(_path_edges(), 4).numpy()

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-12)


def test_layout_config_algorithm_spiral_dispatches() -> None:
    """Exercise public engine dispatch for ``algorithm='spiral'``.

    Returns
    -------
    None
        Dispatch must return finite coordinates.
    """
    graph = dagua.DaguaGraph.from_edge_list([("a", "b"), ("b", "c")])
    positions = dagua.layout(graph, LayoutConfig(algorithm="spiral"))

    assert positions.shape == (3, 2)
    assert torch.isfinite(positions).all()
