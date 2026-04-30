"""Tests for Graphviz neato-compatible dispatch."""

from __future__ import annotations

import torch

import dagua


def test_public_neato_algorithm_dispatch() -> None:
    """Public ``dagua.layout`` accepts ``algorithm="neato"``."""
    graph = dagua.DaguaGraph.from_edge_list([("a", "b"), ("b", "c"), ("c", "a")])

    pos = dagua.layout(graph, dagua.LayoutConfig(algorithm="neato", seed=42))

    assert pos.shape == (3, 2)
    assert torch.isfinite(pos).all()


def test_stress_majorization_graphviz_neato_fidelity_switch() -> None:
    """The stress pipeline accepts the neato fidelity compatibility switch."""
    graph = dagua.DaguaGraph.from_edge_list([("a", "b"), ("b", "c"), ("d", "e")])

    pos = dagua.layout(
        graph,
        dagua.LayoutConfig(
            algorithm="stress_majorization",
            seed=42,
            algorithm_params={"graphviz_neato_fidelity": True},
        ),
    )

    assert pos.shape == (5, 2)
    assert torch.isfinite(pos).all()
