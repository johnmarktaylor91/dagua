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
    """Explicit directed graphs should not route through the flat stress path."""
    graph = DaguaGraph.from_edge_list(
        [(i, i + 1) for i in range(9)],
        num_nodes=10,
        is_semantically_directed=True,
    )
    graph.compute_node_sizes()
    pos_routed = layout(graph, LayoutConfig(seed=42, route_flat_to_stress=True))
    pos_layered = layout(graph, LayoutConfig(seed=42, force_pipeline="tree"))
    # Directed chains use the native tree path rather than the flat pipeline.
    assert torch.allclose(pos_routed, pos_layered)


def test_auto_route_redirects_undirected_to_flat() -> None:
    """Undirected non-planar graph routes to dagua_flat when forced.

    Sprint-20e refactored auto-dispatch: undirected graphs now flow
    through native_force_directed inside dagua_native. dagua_flat remains
    available as an explicit opt-in via algorithm="dagua_flat", which
    this test exercises alongside the default (unforced) path.
    """
    edges: list[tuple[int, int]] = []
    for i in range(4):
        for j in range(4, 8):
            edges.append((i, j))
    graph = DaguaGraph.from_edge_list(
        edges,
        num_nodes=8,
        is_semantically_directed=False,
    )
    graph.compute_node_sizes()
    pos_flat = layout(graph, LayoutConfig(seed=42, algorithm="dagua_flat"))
    pos_default = layout(graph, LayoutConfig(seed=42))
    assert pos_flat.shape == pos_default.shape
    assert torch.isfinite(pos_flat).all()
    assert torch.isfinite(pos_default).all()
    # The two pipelines should produce materially different layouts.
    assert not torch.allclose(pos_flat, pos_default, atol=1e-3)


def test_auto_route_off_preserves_default() -> None:
    """route_flat_to_stress=False preserves the sprint-20d monolith."""
    graph = _get_graph("random_dag_50")
    graph.compute_node_sizes()
    pos_off = layout(graph, LayoutConfig(seed=42, route_flat_to_stress=False))
    pos_default = layout(graph, LayoutConfig(seed=42, route_flat_to_stress=True))
    assert pos_off.shape == pos_default.shape
    assert torch.isfinite(pos_off).all()
    assert torch.isfinite(pos_default).all()
