"""Pipeline pins for the NetworkX shell layout."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import dagua
from dagua.config import LayoutConfig
from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.shell import build_shell_pipeline, layout_shell_pipeline


def _path_edges() -> torch.Tensor:
    """Return a fixed four-node path edge tensor.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, 3]``.
    """
    return torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)


def test_shell_pipeline_is_registered() -> None:
    """Register the public shell algorithm.

    Returns
    -------
    None
        Registry lookups must resolve the shell entrypoint.
    """
    assert PIPELINE_REGISTRY["shell"] == (
        "dagua.layout.ops.pipelines.shell",
        "layout_shell_pipeline",
    )
    assert get_pipeline_function("SHELL") is layout_shell_pipeline
    assert [operation.name for operation in build_shell_pipeline().ops] == [
        "networkx_simple_layout"
    ]


def test_shell_matches_networkx_reference() -> None:
    """Match NetworkX shell coordinates on a small graph.

    Returns
    -------
    None
        Dagua's source port must match NetworkX exactly for the default shell.
    """
    nx = pytest.importorskip("networkx")
    graph = nx.DiGraph()
    graph.add_nodes_from(range(4))
    graph.add_edges_from([(0, 1), (1, 2), (2, 3)])
    expected = np.vstack([nx.shell_layout(graph)[node] for node in range(4)])

    actual = layout_shell_pipeline(_path_edges(), 4).numpy()

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-12)


def test_layout_config_algorithm_shell_dispatches() -> None:
    """Exercise public engine dispatch for ``algorithm='shell'``.

    Returns
    -------
    None
        Dispatch must return finite coordinates.
    """
    graph = dagua.DaguaGraph.from_edge_list([("a", "b"), ("b", "c")])
    positions = dagua.layout(graph, LayoutConfig(algorithm="shell"))

    assert positions.shape == (3, 2)
    assert torch.isfinite(positions).all()
