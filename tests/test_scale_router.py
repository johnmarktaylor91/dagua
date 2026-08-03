"""Tests for M1 topology sketching, scale routing, and engine gate behavior."""

from __future__ import annotations

from importlib import import_module
from typing import List, Tuple

import pytest
import torch

import dagua
from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.layout.scale.router import ScaleStrategy, route, should_enter_scale_gate
from dagua.layout.scale.sketch import TopologySketch


def _graph_from_edges(edges: List[Tuple[int, int]], num_nodes: int) -> DaguaGraph:
    """Build a Dagua graph from integer edges.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Directed edge list.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    DaguaGraph
        Graph with a CPU long edge tensor.
    """
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).transpose(0, 1).contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return DaguaGraph.from_edge_index(edge_index, num_nodes=num_nodes)


def _chain_graph(num_nodes: int) -> DaguaGraph:
    """Build a directed chain graph.

    Parameters
    ----------
    num_nodes : int
        Number of chain nodes.

    Returns
    -------
    DaguaGraph
        Chain graph with ``N - 1`` edges.
    """
    return _graph_from_edges([(node, node + 1) for node in range(num_nodes - 1)], num_nodes)


def _wide_layered_dag(width: int = 12) -> DaguaGraph:
    """Build a shallow wide DAG fixture.

    Parameters
    ----------
    width : int, default=12
        Nodes per layer.

    Returns
    -------
    DaguaGraph
        Three-layer DAG that should route to ``LAYERS``.
    """
    edges = []
    for node in range(width):
        edges.append((node, width + node))
        edges.append((width + node, 2 * width + node))
    return _graph_from_edges(edges, 3 * width)


def _cyclic_er_like(num_nodes: int = 16) -> DaguaGraph:
    """Build a strongly connected cyclic fixture.

    Parameters
    ----------
    num_nodes : int, default=16
        Number of cyclic nodes.

    Returns
    -------
    DaguaGraph
        Cyclic graph with one giant SCC.
    """
    edges = [(node, (node + 1) % num_nodes) for node in range(num_nodes)]
    edges.extend((node, (node + 3) % num_nodes) for node in range(num_nodes))
    return _graph_from_edges(edges, num_nodes)


def _sketch(graph: DaguaGraph, *, depth_cap: int = 8) -> TopologySketch:
    """Build a sketch for a test graph.

    Parameters
    ----------
    graph : DaguaGraph
        Graph fixture.
    depth_cap : int, default=8
        Capped-depth threshold.

    Returns
    -------
    TopologySketch
        Exact topology sketch.
    """
    return TopologySketch.from_edge_index(graph.edge_index, graph.num_nodes, depth_cap=depth_cap)


def test_scc_exactness_detects_cyclic_giant_and_dag_singletons() -> None:
    """Exact SCC fields distinguish a giant cyclic SCC from a DAG."""
    cyclic = _sketch(_cyclic_er_like(12))
    dag = _sketch(_wide_layered_dag(5))

    assert cyclic.largest_scc_size == 12
    assert cyclic.scc_count == 1
    assert not cyclic.is_acyclic
    assert dag.largest_scc_size == 1
    assert dag.scc_count == dag.num_nodes
    assert dag.is_acyclic


def test_route_fixtures_lock_strategy_and_reason_codes() -> None:
    """Router decisions are stable on representative topology fixtures."""
    config = LayoutConfig()

    wide = route(_sketch(_wide_layered_dag()), config)
    assert wide.strategy == ScaleStrategy.LAYERS
    assert wide.reason_codes == ["acyclic", "depth_cap_passed"]

    cyclic = route(_sketch(_cyclic_er_like()), config)
    assert cyclic.strategy == ScaleStrategy.FIELD
    assert cyclic.reason_codes == ["nontrivial_giant_scc", "cyclic_field_required"]

    deep = route(_sketch(_chain_graph(12), depth_cap=4), config)
    assert deep.strategy == ScaleStrategy.FIELD
    assert deep.reason_codes == ["depth_cap_tripped", "acyclic_hostile_field"]

    stall_graph = _graph_from_edges([(0, node) for node in range(1, 20)], 20)
    stall_config = LayoutConfig(
        algorithm_params={
            "scale_reduction_stall_degree": 6,
            "scale_reduction_stall_fraction": 0.5,
        }
    )
    stall = route(_sketch(stall_graph), stall_config)
    assert stall.strategy == ScaleStrategy.FIELD
    assert stall.reason_codes == ["reduction_stall_risk", "acyclic_hostile_field"]


def test_scale_gate_boundary_and_edge_conditional_trigger() -> None:
    """Node and edge thresholds trigger the one scale gate at the boundary."""
    config = LayoutConfig(algorithm_params={"scale_node_gate": 10, "scale_edge_gate": 20})

    assert not should_enter_scale_gate(10, 20, config)
    assert should_enter_scale_gate(11, 20, config)
    assert should_enter_scale_gate(10, 21, config)


def test_below_gate_default_never_constructs_sketch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sub-gate native default layout remains byte-inert with no sketch build."""

    def forbidden_from_edge_index(*args: object, **kwargs: object) -> TopologySketch:
        """Fail if the below-gate path reaches scale sketching."""
        del args, kwargs
        raise AssertionError("below-gate default path constructed a sketch")

    monkeypatch.setattr(TopologySketch, "from_edge_index", forbidden_from_edge_index)

    graph = _chain_graph(4)
    pos = dagua.layout(
        graph,
        LayoutConfig(
            steps=1,
            algorithm_params={"scale_node_gate": 10, "scale_edge_gate": 20},
        ),
    )

    assert pos.shape == (4, 2)
    assert not hasattr(graph, "_dagua_scale_route_decision")


def test_above_gate_layers_dispatches_to_legacy_multilevel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Above-gate acyclic layouts route through the temporary LAYERS fallback."""
    calls = {"multilevel": 0}

    def fake_multilevel_layout(
        graph: DaguaGraph,
        config: LayoutConfig,
        trace: object = None,
    ) -> torch.Tensor:
        """Return deterministic finite positions for the routed graph."""
        del config, trace
        calls["multilevel"] += 1
        return torch.zeros((graph.num_nodes, 2), dtype=torch.float32)

    multilevel_module = import_module("dagua.layout.multilevel")
    monkeypatch.setattr(multilevel_module, "multilevel_layout", fake_multilevel_layout)

    graph = _wide_layered_dag(4)
    pos = dagua.layout(
        graph,
        LayoutConfig(
            algorithm_params={
                "scale_node_gate": 10,
                "scale_edge_gate": 200,
                "scale_depth_cap": 8,
            }
        ),
    )

    assert calls == {"multilevel": 1}
    assert torch.isfinite(pos).all()
    assert pos.shape == (12, 2)
    metadata = getattr(graph, "_dagua_scale_route_decision")
    assert metadata["decision"]["strategy"] == "LAYERS"
    assert metadata["decision"]["reason_codes"] == ["acyclic", "depth_cap_passed"]


def test_above_gate_field_dispatches_to_temporary_legacy_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Above-gate cyclic layouts route to FIELD metadata and complete via fallback."""

    def fake_multilevel_layout(
        graph: DaguaGraph,
        config: LayoutConfig,
        trace: object = None,
    ) -> torch.Tensor:
        """Return deterministic finite positions for the routed cyclic graph."""
        del config, trace
        return torch.ones((graph.num_nodes, 2), dtype=torch.float32)

    multilevel_module = import_module("dagua.layout.multilevel")
    monkeypatch.setattr(multilevel_module, "multilevel_layout", fake_multilevel_layout)

    graph = _cyclic_er_like(8)
    pos = dagua.layout(
        graph,
        LayoutConfig(
            algorithm_params={
                "scale_node_gate": 5,
                "scale_edge_gate": 200,
                "scale_depth_cap": 8,
            }
        ),
    )

    assert torch.isfinite(pos).all()
    assert pos.shape == (8, 2)
    metadata = getattr(graph, "_dagua_scale_route_decision")
    assert metadata["decision"]["strategy"] == "FIELD"
    assert metadata["temporary_fallback"] == "legacy_multilevel"
