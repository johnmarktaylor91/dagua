"""Tests for FIELD scale coarsening, pyramid force, and dispatch."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List, Tuple

import pytest
import torch

import dagua
from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.layout.scale import coarsen as scale_coarsen
from dagua.layout.scale.coarsen import build_scale_hierarchy, prolong_positions
from dagua.layout.scale.pyramid import build_grid_pyramid, far_field_repulsion_force
from dagua.layout.scale.strategies import field as field_strategy
from dagua.layout.scale.strategies.field import (
    _estimate_edge_spring_sort_workspace_bytes,
    _estimate_refine_peak_bytes,
    _should_use_streaming_field,
    _streaming_initial_positions_from_edges,
)

_L40_48GB_BYTES = 48_000_000_000


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


def _shape_only_scale_graph(num_nodes: int, num_edges: int) -> Any:
    """Build a graph stub that exposes only selector-required shape fields.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes to report.
    num_edges : int
        Number of graph edges to report.

    Returns
    -------
    Any
        Lightweight graph stub with ``num_nodes`` and ``edge_index.shape``.
    """
    return SimpleNamespace(
        num_nodes=int(num_nodes),
        edge_index=SimpleNamespace(shape=(2, int(num_edges))),
    )


def _cuda_field_config() -> LayoutConfig:
    """Return a FIELD config that requests CUDA selection.

    Returns
    -------
    LayoutConfig
        Configuration with a CUDA device string and deterministic seed.
    """
    return LayoutConfig(device="cuda", seed=42)


def test_field_refine_peak_estimate_includes_stable_sort_workspace() -> None:
    """FIELD peak VRAM model includes the missing sorted spring argsort workspace."""
    edge_count = 600_000_000

    sort_workspace = _estimate_edge_spring_sort_workspace_bytes(edge_count)
    peak = _estimate_refine_peak_bytes(300_000_000, edge_count)

    assert sort_workspace == 28_800_000_000
    assert peak >= sort_workspace


def test_field_auto_selector_streams_large_cuda_levels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Large FIELD graphs stream immediately on a 48GB L40 budget."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (_L40_48GB_BYTES, _L40_48GB_BYTES))
    config = _cuda_field_config()

    assert _should_use_streaming_field(_shape_only_scale_graph(300_000_000, 600_000_000), config)
    assert _should_use_streaming_field(
        _shape_only_scale_graph(1_000_000_000, 3_000_000_000),
        config,
    )


def test_field_auto_selector_keeps_measured_fit_cuda_levels_resident(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Measured-fit 10M and 100M FIELD rungs keep the resident CUDA path."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (_L40_48GB_BYTES, _L40_48GB_BYTES))
    config = _cuda_field_config()

    assert not _should_use_streaming_field(_shape_only_scale_graph(10_000_000, 10_000_000), config)
    assert not _should_use_streaming_field(
        _shape_only_scale_graph(100_000_000, 100_000_000),
        config,
    )


def test_field_auto_selector_uses_vram_bound_below_node_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Edge-heavy FIELD levels stream from a VRAM-derived bound, not a bare N gate."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (_L40_48GB_BYTES, _L40_48GB_BYTES))
    config = LayoutConfig(
        device="cuda",
        algorithm_params={"field_streaming_node_threshold": 1_000_000_000},
        seed=42,
    )

    assert _should_use_streaming_field(_shape_only_scale_graph(300_000_000, 600_000_000), config)


def test_field_refine_device_uses_corrected_peak_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resident CUDA refinement is rejected before launching the sorted spring argsort."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (_L40_48GB_BYTES, _L40_48GB_BYTES))

    assert field_strategy._choose_refine_device(300_000_000, 600_000_000, "cuda") == "cpu"
    assert field_strategy._choose_refine_device(10_000_000, 10_000_000, "cuda") == "cuda"


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


def test_field_streaming_threshold_auto_selects_streaming_regime() -> None:
    """FIELD streams through the real hierarchy when the threshold is crossed."""
    config = LayoutConfig(
        algorithm_params={
            "scale_node_gate": 10,
            "scale_edge_gate": 10_000,
            "field_streaming_node_threshold": 16,
            "field_coarsest_target": 8,
            "field_coarsest_solver": "stress_sgd",
            "field_refine_steps": 0,
            "field_max_grid_axis": 16,
        },
        seed=42,
    )
    graph = _cyclic_fixture(24)

    pos = dagua.layout(graph, config)

    assert torch.isfinite(pos).all()
    metadata = getattr(graph, "_dagua_scale_route_decision")
    assert str(metadata["field"]["coarsest_solver"]).startswith("streaming_hierarchy_")
    assert metadata["field"]["levels"] > 0
    assert metadata["field"]["coarsest_nodes"] <= 8


def test_field_streaming_branch_is_explicit_fallback_only() -> None:
    """FIELD streaming hierarchy is deterministic only when explicitly allowed."""
    config = LayoutConfig(
        algorithm_params={
            "scale_node_gate": 10,
            "scale_edge_gate": 10_000,
            "field_streaming_node_threshold": 16,
            "field_allow_streaming_fallback": True,
            "field_coarsest_target": 8,
            "field_streaming_refine_steps": 0,
        },
        seed=42,
    )
    first_graph = _cyclic_fixture(24)
    second_graph = _cyclic_fixture(24)

    first = dagua.layout(first_graph, config)
    second = dagua.layout(second_graph, config)

    assert torch.isfinite(first).all()
    assert torch.equal(first, second)
    metadata = getattr(first_graph, "_dagua_scale_route_decision")
    assert str(metadata["field"]["coarsest_solver"]).startswith("streaming_hierarchy_")
    assert metadata["field"]["levels"] > 0


def test_field_streaming_initializer_is_graph_informed_not_spiral() -> None:
    """Streaming FIELD initialization uses edge neighborhoods, not radial node order."""
    graph = _cyclic_fixture(32)

    first = _streaming_initial_positions_from_edges(
        graph.edge_index,
        graph.num_nodes,
        base_sep=10.0,
        seed=42,
        chunk_size=8,
    )
    second = _streaming_initial_positions_from_edges(
        graph.edge_index,
        graph.num_nodes,
        base_sep=10.0,
        seed=42,
        chunk_size=8,
    )
    radius = torch.linalg.norm(first, dim=1)

    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert not bool(torch.all(radius[1:] >= radius[:-1]).item())


def test_scale_hierarchy_uses_tensor_coarsening_without_large_adjacency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Large FIELD levels avoid Python adjacency while still reducing."""
    monkeypatch.setattr(scale_coarsen, "_ADJACENCY_BUILD_NODE_LIMIT", 8)
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
    assert hierarchy.finest_graph.adjacency == []
    assert hierarchy.coarsest_graph.num_nodes <= 12
