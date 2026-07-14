"""Pipeline pins for the NetworkX ARF layout."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import dagua
from dagua.config import LayoutConfig
from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.arf import build_arf_pipeline, layout_arf_pipeline


def _path_edges() -> torch.Tensor:
    """Return a fixed four-node path edge tensor.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, 3]``.
    """
    return torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)


def test_arf_pipeline_is_registered() -> None:
    """Register the public ARF algorithm.

    Returns
    -------
    None
        Registry lookups must resolve the ARF entrypoint.
    """
    assert PIPELINE_REGISTRY["arf"] == (
        "dagua.layout.ops.pipelines.arf",
        "layout_arf_pipeline",
    )
    assert get_pipeline_function("ARF") is layout_arf_pipeline
    assert [operation.name for operation in build_arf_pipeline().ops] == ["networkx_simple_layout"]


def test_arf_matches_networkx_reference() -> None:
    """Match NetworkX ARF coordinates with the pinned seed.

    Returns
    -------
    None
        Dagua's source port must match NetworkX exactly.
    """
    nx = pytest.importorskip("networkx")
    graph = nx.DiGraph()
    graph.add_nodes_from(range(4))
    graph.add_edges_from([(0, 1), (1, 2), (2, 3)])
    expected = np.vstack([nx.arf_layout(graph, seed=42)[node] for node in range(4)])

    actual = layout_arf_pipeline(_path_edges(), 4, seed=42).numpy()

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-12)


def test_layout_config_algorithm_arf_dispatches() -> None:
    """Exercise public engine dispatch for ``algorithm='arf'``.

    Returns
    -------
    None
        Dispatch must return finite coordinates.
    """
    graph = dagua.DaguaGraph.from_edge_list([("a", "b"), ("b", "c")])
    positions = dagua.layout(graph, LayoutConfig(algorithm="arf", seed=42))

    assert positions.shape == (3, 2)
    assert torch.isfinite(positions).all()
