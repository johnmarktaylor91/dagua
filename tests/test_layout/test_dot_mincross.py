"""Regression tests for Graphviz dot mincross fidelity helpers."""

from __future__ import annotations

import torch

from dagua.layout.ops._dot_mincross import graphviz_mincross
from dagua.layout.ops.pipelines.sugiyama import layout_sugiyama_pipeline


def test_graphviz_mincross_two_rank_golden_order() -> None:
    """Match a two-rank golden ordering captured from Graphviz 7.0.5 dot."""
    ranks = [[0, 1], [2, 3]]
    edges = torch.tensor([[0, 1], [3, 2]], dtype=torch.long)

    ordering = graphviz_mincross(ranks=ranks, edges=edges, iterations=24)

    assert ordering == [[0, 1], [3, 2]]


def test_graphviz_mincross_three_rank_golden_order() -> None:
    """Match a three-rank golden ordering captured from Graphviz 7.0.5 dot."""
    ranks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    edges = [
        (0, 4),
        (1, 3),
        (2, 5),
        (3, 7),
        (4, 6),
        (5, 8),
    ]

    ordering = graphviz_mincross(ranks=ranks, edges=edges, iterations=24)

    assert ordering == [[0, 1, 2], [4, 3, 5], [6, 7, 8]]


def test_graphviz_mincross_ignores_unexpanded_long_edges() -> None:
    """Require callers to pass dot-style adjacent-rank virtual-node chains."""
    ranks = [[0], [1], [2]]
    edges = [(0, 2)]

    ordering = graphviz_mincross(ranks=ranks, edges=edges, iterations=24)

    assert ordering == ranks


def test_sugiyama_dot_fidelity_uses_graphviz_mincross() -> None:
    """Run the public Sugiyama pipeline with Graphviz dot mincross enabled."""
    edge_index = torch.tensor([[0, 1], [3, 2]], dtype=torch.long)

    positions = layout_sugiyama_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        fidelity_mode="dot",
        center_coordinates=False,
    )

    assert isinstance(positions, torch.Tensor)
    assert float(positions[3, 0].item()) < float(positions[2, 0].item())
