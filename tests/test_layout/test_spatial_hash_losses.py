"""Integration tests for spatial-hash native losses."""

from __future__ import annotations

import torch

from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.layout import layout
from dagua.metrics import composite, full


def _make_synthetic_500_node_graph() -> DaguaGraph:
    """Build a deterministic 500-node cyclic graph for native layout tests.

    Returns
    -------
    DaguaGraph
        Graph with precomputed node sizes and direct edge tensor storage.
    """
    num_nodes = 500
    nodes = torch.arange(num_nodes, dtype=torch.long)
    ring_sources = nodes
    ring_targets = (nodes + 1) % num_nodes
    chord_sources = nodes[::2]
    chord_targets = (chord_sources + 37) % num_nodes
    graph = DaguaGraph(num_nodes=num_nodes)
    graph.edge_index = torch.cat(
        [
            torch.stack([ring_sources, ring_targets]),
            torch.stack([chord_sources, chord_targets]),
        ],
        dim=1,
    )
    graph.node_sizes = torch.full((num_nodes, 2), 16.0, dtype=torch.float32)
    return graph


def _layout_score(graph: DaguaGraph, exact_repulsion: bool) -> tuple[torch.Tensor, float]:
    """Run a short default-native layout and score its composite metric.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to lay out.
    exact_repulsion : bool
        Whether to force the legacy exact pairwise losses.

    Returns
    -------
    tuple[torch.Tensor, float]
        Layout positions and composite score.
    """
    config = LayoutConfig(
        steps=2,
        seed=123,
        exact_repulsion=exact_repulsion,
        w_crossing=0.0,
    )
    pos = layout(graph, config)
    metrics = full(
        pos,
        graph.edge_index,
        node_sizes=graph.node_sizes,
        crossing_samples=10_000,
        neighborhood_samples=1000,
    )
    return pos, float(composite(metrics))


def test_default_native_cell_list_500_node_layout_matches_exact_composite() -> None:
    """Cell-list and exact losses should produce close finite 500-node layouts."""
    graph = _make_synthetic_500_node_graph()

    cell_pos, cell_score = _layout_score(graph, exact_repulsion=False)
    exact_pos, exact_score = _layout_score(graph, exact_repulsion=True)

    assert torch.isfinite(cell_pos).all()
    assert torch.isfinite(exact_pos).all()
    assert cell_pos.shape == exact_pos.shape == (500, 2)
    assert abs(cell_score - exact_score) <= 1.0
