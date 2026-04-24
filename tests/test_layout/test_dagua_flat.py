"""Tests for dagua_flat pipeline + direction-aware dispatcher."""

from __future__ import annotations

import torch

from dagua.config import LayoutConfig
from dagua.eval.graphs import get_test_graphs
from dagua.graph import DaguaGraph
from dagua.layout.engine import layout
from dagua.layout.ops.pipelines.dagua_flat import layout_dagua_flat_pipeline


def _get_graph(name: str) -> DaguaGraph:
    """Return the named benchmark graph by its registered test name."""
    return next(tg.graph for tg in get_test_graphs() if tg.name == name)


def test_dagua_flat_smoke_on_ring() -> None:
    """layout_dagua_flat_pipeline returns finite positions on a small ring."""
    graph = DaguaGraph.from_edge_list([(i, (i + 1) % 8) for i in range(8)], num_nodes=8)
    graph.compute_node_sizes()
    pos = layout_dagua_flat_pipeline(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        node_sizes=graph.node_sizes,
        steps=20,
        seed=42,
    )
    assert pos.shape == (8, 2)
    assert torch.isfinite(pos).all()


def test_dagua_flat_algorithm_dispatch() -> None:
    """algorithm='dagua_flat' dispatches to the flat pipeline."""
    graph = DaguaGraph.from_edge_list([(i, (i + 1) % 8) for i in range(8)], num_nodes=8)
    graph.compute_node_sizes()
    pos = layout(graph, LayoutConfig(algorithm="dagua_flat", seed=42, steps=20))
    assert pos.shape == (8, 2)
    assert torch.isfinite(pos).all()


def test_auto_route_respects_is_semantically_directed_true() -> None:
    """Explicit is_semantically_directed=True keeps the layered native path."""
    graph = DaguaGraph.from_edge_list(
        [(i, i + 1) for i in range(9)],
        num_nodes=10,
        is_semantically_directed=True,
    )
    graph.compute_node_sizes()
    pos_routed = layout(graph, LayoutConfig(seed=42, route_flat_to_stress=True))
    pos_bypass = layout(graph, LayoutConfig(seed=42, route_flat_to_stress=False))
    # Directed chain should follow the same (native) path either way.
    assert torch.allclose(pos_routed, pos_bypass)


def test_auto_route_redirects_undirected_to_flat() -> None:
    """Undirected ring routes to dagua_flat under route_flat_to_stress=True."""
    graph = DaguaGraph.from_edge_list(
        [(i, (i + 1) % 12) for i in range(12)],
        num_nodes=12,
        is_semantically_directed=False,
    )
    graph.compute_node_sizes()
    pos_routed = layout(graph, LayoutConfig(seed=42, route_flat_to_stress=True))
    pos_bypass = layout(graph, LayoutConfig(seed=42, route_flat_to_stress=False))
    # With routing on, the flat pipeline should produce a materially
    # different layout than the native path on this undirected graph.
    assert not torch.allclose(pos_routed, pos_bypass)


def test_auto_route_off_preserves_default() -> None:
    """route_flat_to_stress=False preserves the current default dispatch."""
    graph = _get_graph("random_dag_50")
    graph.compute_node_sizes()
    pos_off = layout(graph, LayoutConfig(seed=42, route_flat_to_stress=False))
    pos_default = layout(graph, LayoutConfig(seed=42, route_flat_to_stress=True))
    # random_dag_50 is a real DAG; both runs should produce the same output.
    assert torch.allclose(pos_off, pos_default)
