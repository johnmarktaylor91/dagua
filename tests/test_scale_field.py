"""Tests for FIELD scale coarsening, pyramid force, and dispatch."""

from __future__ import annotations

from typing import List, Tuple

import torch

import dagua
from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.layout.scale.coarsen import build_scale_hierarchy, prolong_positions
from dagua.layout.scale.pyramid import build_grid_pyramid, far_field_repulsion_force


def _graph_from_edges(edges: List[Tuple[int, int]], num_nodes: int) -> DaguaGraph:
    """Build a graph from integer edge pairs.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Directed edge list.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    DaguaGraph
        Graph with a CPU edge tensor.
    """
    edge_index = (
        torch.tensor(edges, dtype=torch.long).transpose(0, 1).contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    return DaguaGraph.from_edge_index(edge_index, num_nodes=num_nodes)


def _cyclic_fixture(num_nodes: int = 24) -> DaguaGraph:
    """Return a deterministic cyclic graph routed to FIELD.

    Parameters
    ----------
    num_nodes : int, default=24
        Number of graph nodes.

    Returns
    -------
    DaguaGraph
        Strongly connected cyclic graph.
    """
    edges = [(node, (node + 1) % num_nodes) for node in range(num_nodes)]
    edges.extend((node, (node + 5) % num_nodes) for node in range(num_nodes))
    return _graph_from_edges(edges, num_nodes)


def test_scale_hierarchy_reduces_and_prolongs_deterministically() -> None:
    """Family-agnostic coarsening reduces cyclic graphs and prolongs byte-identically."""
    graph = _cyclic_fixture(64)
    sizes = torch.full((64, 2), 10.0, dtype=torch.float32)

    hierarchy = build_scale_hierarchy(
        graph.edge_index,
        graph.num_nodes,
        sizes,
        target_nodes=12,
        seed=42,
        topk_per_node=4,
    )

    assert hierarchy.levels
    assert hierarchy.coarsest_graph.num_nodes <= 12
    assert hierarchy.coarsest_graph.edge_index.shape[1] <= hierarchy.coarsest_graph.num_nodes * 4
    coarse = torch.arange(
        hierarchy.coarsest_graph.num_nodes * 2,
        dtype=torch.float32,
    ).reshape(-1, 2)
    first = prolong_positions(coarse, hierarchy.levels[-1].fine_to_coarse, seed=42)
    second = prolong_positions(coarse, hierarchy.levels[-1].fine_to_coarse, seed=42)
    assert torch.equal(first, second)


def test_grid_pyramid_force_is_deterministic_and_repulsive() -> None:
    """Grid-pyramid far-field force is byte-deterministic and points outward."""
    pos = torch.tensor([[-10.0, 0.0], [10.0, 0.0]], dtype=torch.float32)
    pyramid = build_grid_pyramid(pos, base_cell_size=5.0, max_cells_per_axis=8, max_levels=3)

    first = far_field_repulsion_force(pos, pyramid, strength=1.0, max_displacement=100.0)
    second = far_field_repulsion_force(pos, pyramid, strength=1.0, max_displacement=100.0)

    assert torch.equal(first, second)
    assert first[0, 0] < 0.0
    assert first[1, 0] > 0.0


def test_field_dispatch_completes_and_is_byte_deterministic() -> None:
    """Above-gate cyclic default dispatch uses FIELD and is deterministic."""
    config = LayoutConfig(
        algorithm_params={
            "scale_node_gate": 10,
            "scale_edge_gate": 10_000,
            "field_coarsest_target": 8,
            "field_coarsest_solver": "stress_sgd",
            "field_refine_steps": 1,
            "field_max_grid_axis": 16,
        },
        seed=42,
    )
    first_graph = _cyclic_fixture(24)
    second_graph = _cyclic_fixture(24)

    first = dagua.layout(first_graph, config)
    second = dagua.layout(second_graph, config)

    assert torch.isfinite(first).all()
    assert torch.equal(first, second)
    assert first.shape == (24, 2)
    metadata = getattr(first_graph, "_dagua_scale_route_decision")
    assert metadata["decision"]["strategy"] == "FIELD"
    assert "temporary_fallback" not in metadata
    assert metadata["field"]["coarsest_nodes"] <= 8
