"""Pipeline pins for the NetworkX circular layout."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import dagua
from dagua.config import LayoutConfig
from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.circular import build_circular_pipeline, layout_circular_pipeline
from dagua.layout.ops.taxonomy import get_op_class


def _path_edges() -> torch.Tensor:
    """Return a fixed four-node path edge tensor.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, 3]``.
    """
    return torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)


def test_circular_pipeline_is_registered() -> None:
    """Register the public circular algorithm and shared NetworkX op.

    Returns
    -------
    None
        Registry lookups must resolve the circular entrypoint and op class.
    """
    assert PIPELINE_REGISTRY["circular"] == (
        "dagua.layout.ops.pipelines.circular",
        "layout_circular_pipeline",
    )
    assert get_pipeline_function("CIRCULAR") is layout_circular_pipeline
    assert get_op_class("networkx_simple_layout").__name__ == "NetworkXSimpleLayout"
    assert [operation.name for operation in build_circular_pipeline().ops] == [
        "networkx_simple_layout"
    ]


def test_circular_matches_networkx_reference() -> None:
    """Match NetworkX circular coordinates on a small graph.

    Returns
    -------
    None
        Dagua's source port must match NetworkX within float32 trigonometric
        tolerance.
    """
    nx = pytest.importorskip("networkx")
    graph = nx.DiGraph()
    graph.add_nodes_from(range(4))
    graph.add_edges_from([(0, 1), (1, 2), (2, 3)])
    expected = np.vstack([nx.circular_layout(graph)[node] for node in range(4)])

    actual = layout_circular_pipeline(_path_edges(), 4).numpy()

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=5.0e-8)


def test_layout_config_algorithm_circular_dispatches() -> None:
    """Exercise public engine dispatch for ``algorithm='circular'``.

    Returns
    -------
    None
        Dispatch must return finite coordinates.
    """
    graph = dagua.DaguaGraph.from_edge_list([("a", "b"), ("b", "c")])
    positions = dagua.layout(graph, LayoutConfig(algorithm="circular"))

    assert positions.shape == (3, 2)
    assert torch.isfinite(positions).all()


def test_networkx_simple_pipelines_have_no_runtime_delegation() -> None:
    """Guard production NetworkX-simple pipelines against oracle delegation.

    Returns
    -------
    None
        Production source must not import NetworkX or call competitor adapters.
    """
    root = Path(__file__).parents[1]
    source_paths = [
        root / "dagua" / "layout" / "ops" / "networkx_simple.py",
        root / "dagua" / "layout" / "ops" / "pipelines" / "circular.py",
        root / "dagua" / "layout" / "ops" / "pipelines" / "shell.py",
        root / "dagua" / "layout" / "ops" / "pipelines" / "spiral.py",
        root / "dagua" / "layout" / "ops" / "pipelines" / "bipartite.py",
        root / "dagua" / "layout" / "ops" / "pipelines" / "multipartite.py",
        root / "dagua" / "layout" / "ops" / "pipelines" / "bfs.py",
        root / "dagua" / "layout" / "ops" / "pipelines" / "arf.py",
    ]
    source = "\n".join(path.read_text() for path in source_paths)
    assert "import networkx" not in source
    assert "networkx_competitor" not in source
