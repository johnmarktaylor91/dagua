"""Regression tests for the native pipeline multi-start wrapper."""

from __future__ import annotations

import torch

from dagua.config import LayoutConfig
from dagua.eval.graphs import get_test_graphs
from dagua.graph import DaguaGraph
from dagua.layout.engine import layout


def _graph(name: str) -> DaguaGraph:
    """Return a named evaluation graph.

    Parameters
    ----------
    name : str
        Name of the graph fixture to return.

    Returns
    -------
    DaguaGraph
        Matching evaluation graph.
    """
    return next(test_graph.graph for test_graph in get_test_graphs() if test_graph.name == name)


def test_multi_start_k_one_preserves_default() -> None:
    """Verify explicit single-start native layout matches the default."""
    g = _graph("random_dag_50")
    g.compute_node_sizes()
    a = layout(g, LayoutConfig(seed=42))
    b = layout(g, LayoutConfig(seed=42, multi_start_k=1))
    assert torch.equal(a, b)


def test_multi_start_k_three_scores_at_least_single() -> None:
    """Verify best-of-three does not score below the single seeded run.

    The internal best-of-k selection (``_score_native_result``) seeds torch
    to a fixed value before calling ``full()`` so rankings are deterministic
    across candidates. Match that convention here so the external re-score
    sees the same stochastic samples as the wrapper's picker.
    """
    from dagua.metrics import composite, full

    g = _graph("random_dag_50")
    g.compute_node_sizes()
    single = layout(g, LayoutConfig(seed=42, multi_start_k=1))
    multi = layout(g, LayoutConfig(seed=42, multi_start_k=3))
    torch.manual_seed(0)
    s1 = composite(full(single, g.edge_index, node_sizes=g.node_sizes))
    torch.manual_seed(0)
    s3 = composite(full(multi, g.edge_index, node_sizes=g.node_sizes))
    assert s3 >= s1
