"""Tests for M1 topology sketching, scale routing, and engine gate behavior."""

from __future__ import annotations

from types import SimpleNamespace
from typing import List, Tuple

import pytest
import torch

import dagua
from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.layout.scale.router import ScaleStrategy, route, should_enter_scale_gate
from dagua.layout.scale.sketch import (
    TopologySketch,
    estimate_bounded_topology_peak_bytes,
    estimate_declared_topology_peak_bytes,
    estimate_topology_peak_bytes,
)
from dagua.layout.scale.strategies.layers import _positions_from_layers


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


def _assert_layered_positions_have_no_overlap(
    pos: torch.Tensor,
    layer_assignments: torch.Tensor,
    *,
    node_width: float,
) -> None:
    """Assert each rank has strictly increasing x positions without overlap.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    layer_assignments : torch.Tensor
        Layer IDs with shape ``[N]``.
    node_width : float
        Width of each synthetic node.

    Returns
    -------
    None
        Raises an assertion failure when any same-rank pair overlaps.
    """
    num_layers = int(layer_assignments.max().item()) + 1 if layer_assignments.numel() else 0
    for layer_id in range(num_layers):
        nodes = torch.nonzero(layer_assignments == layer_id, as_tuple=False).flatten()
        if nodes.numel() <= 1:
            continue
        ordered_x = pos[nodes, 0].sort(stable=True).values
        diffs = ordered_x[1:] - ordered_x[:-1]
        assert bool((diffs > 0.0).all().item())
        assert int((diffs < float(node_width)).sum().item()) == 0


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


def test_bounded_sketch_routes_backward_edges_to_field() -> None:
    """Over-budget bounded sketches conservatively route cyclic-looking graphs to FIELD."""
    graph = _cyclic_er_like(16)
    sketch = TopologySketch.from_edge_index_bounded(
        graph.edge_index,
        graph.num_nodes,
        depth_cap=8,
        sample_edges=8,
        chunk_edges=4,
    )

    decision = route(sketch, LayoutConfig())

    assert not sketch.is_acyclic
    assert decision.strategy == ScaleStrategy.FIELD
    assert decision.reason_codes == ["nontrivial_giant_scc", "cyclic_field_required"]


@pytest.mark.parametrize(
    ("graph", "depth_cap"),
    [
        (_wide_layered_dag(), 8),
        (_cyclic_er_like(), 8),
        (_chain_graph(12), 4),
    ],
)
def test_bounded_sketch_route_decision_matches_exact_fixtures(
    graph: DaguaGraph,
    depth_cap: int,
) -> None:
    """Bounded sketches preserve exact route decisions on locked fixtures."""
    exact = TopologySketch.from_edge_index(graph.edge_index, graph.num_nodes, depth_cap=depth_cap)
    bounded = TopologySketch.from_edge_index_bounded(
        graph.edge_index,
        graph.num_nodes,
        depth_cap=depth_cap,
        sample_edges=8,
        chunk_edges=4,
    )

    assert route(bounded, LayoutConfig()) == route(exact, LayoutConfig())
    assert bounded.degree_p50 == exact.degree_p50
    assert bounded.degree_p90 == exact.degree_p90
    assert bounded.degree_p99 == exact.degree_p99
    assert bounded.max_degree == exact.max_degree


def test_declared_topology_bypass_route_decision_matches_exact_fixtures() -> None:
    """Declared topology skips SCC/depth work while preserving fixture routes."""
    config = LayoutConfig()

    wide = _wide_layered_dag()
    wide_exact = TopologySketch.from_edge_index(wide.edge_index, wide.num_nodes, depth_cap=8)
    wide_declared = TopologySketch.from_declared_topology(
        wide.edge_index,
        wide.num_nodes,
        "directed_acyclic",
        depth_cap=8,
        depth=2,
        chunk_edges=4,
    )
    assert route(wide_declared, config) == route(wide_exact, config)

    cyclic = _cyclic_er_like()
    cyclic_exact = TopologySketch.from_edge_index(cyclic.edge_index, cyclic.num_nodes, depth_cap=8)
    cyclic_declared = TopologySketch.from_declared_topology(
        cyclic.edge_index,
        cyclic.num_nodes,
        "directed_cyclic",
        depth_cap=8,
        chunk_edges=4,
    )
    assert route(cyclic_declared, config) == route(cyclic_exact, config)

    deep = _chain_graph(12)
    deep_exact = TopologySketch.from_edge_index(deep.edge_index, deep.num_nodes, depth_cap=4)
    deep_declared = TopologySketch.from_declared_topology(
        deep.edge_index,
        deep.num_nodes,
        "directed_acyclic",
        depth_cap=4,
        depth_cap_tripped=True,
        chunk_edges=4,
    )
    assert route(deep_declared, config) == route(deep_exact, config)


def test_layers_positions_two_million_single_rank_has_no_float32_x_collapse() -> None:
    """A 2M-node rank should keep strictly increasing, non-overlapping x values."""
    num_nodes = 2_000_000
    node_width = 1.0
    layer_assignments = torch.zeros(num_nodes, dtype=torch.long)
    graph = SimpleNamespace(node_sizes=torch.full((num_nodes, 2), node_width, dtype=torch.float32))
    config = LayoutConfig(adaptive_spacing=False, node_sep=1.1, rank_sep=10.0)

    pos, num_layers, max_width = _positions_from_layers(graph, config, layer_assignments)

    assert num_layers == 1
    assert max_width == num_nodes
    _assert_layered_positions_have_no_overlap(
        pos,
        layer_assignments,
        node_width=node_width,
    )


@pytest.mark.parametrize("num_nodes", [10_000, 100_000])
def test_layers_positions_wide_dag_preserves_flow_and_overlap(num_nodes: int) -> None:
    """Wide DAG LAYERS coordinates should preserve flow and avoid rank overlap."""
    num_layers = 10
    node_width = 1.0
    width = num_nodes // num_layers
    layer_assignments = torch.arange(num_nodes, dtype=torch.long) // width
    graph = SimpleNamespace(node_sizes=torch.full((num_nodes, 2), node_width, dtype=torch.float32))
    config = LayoutConfig(adaptive_spacing=False, node_sep=1.1, rank_sep=8.0)

    pos, computed_layers, max_width = _positions_from_layers(graph, config, layer_assignments)
    src = torch.arange(0, num_nodes - width, dtype=torch.long)
    tgt = src + width

    assert computed_layers == num_layers
    assert max_width == width
    assert bool((pos[tgt, 1] > pos[src, 1]).all().item())
    _assert_layered_positions_have_no_overlap(
        pos,
        layer_assignments,
        node_width=node_width,
    )


def test_topology_peak_byte_model_unblocks_billion_declared_path() -> None:
    """Modeled topology-sketch peaks stay under the 600GB 1B-node target."""
    points = [
        (10_000_000, 30_000_000),
        (100_000_000, 300_000_000),
        (1_000_000_000, 3_000_000_000),
    ]

    modeled = [
        (
            estimate_topology_peak_bytes(nodes, edges),
            estimate_bounded_topology_peak_bytes(nodes, edges),
            estimate_declared_topology_peak_bytes(nodes, edges),
        )
        for nodes, edges in points
    ]

    assert modeled[0] == (1_670_000_024, 960_000_000, 960_000_000)
    assert modeled[1] == (16_700_000_024, 7_800_000_000, 7_800_000_000)
    assert modeled[2] == (175_000_000_024, 80_200_000_000, 80_200_000_000)
    assert modeled[-1][1] < 600_000_000_000
    assert modeled[-1][2] < 600_000_000_000


def test_scale_dispatch_uses_bounded_sketch_when_exact_sketch_over_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Engine dispatch falls back to bounded sketching before FIELD selection."""
    from dagua.layout.scale import budget as scale_budget

    original_check = scale_budget.BudgetGuard.check

    def check_with_sketch_wall(
        self: scale_budget.BudgetGuard,
        stage: str,
        declared_peak_bytes: int,
    ) -> scale_budget.BudgetCheck:
        """Raise only for exact sketch preflight in this regression."""
        if stage == "scale_sketch":
            raise MemoryError("forced exact sketch wall")
        return original_check(self, stage, declared_peak_bytes)

    monkeypatch.setattr(scale_budget.BudgetGuard, "check", check_with_sketch_wall)
    graph = _cyclic_er_like(24)

    pos = dagua.layout(
        graph,
        LayoutConfig(
            algorithm_params={
                "scale_node_gate": 10,
                "scale_edge_gate": 10_000,
                "field_streaming_node_threshold": 16,
                "field_streaming_refine_steps": 0,
            },
            seed=42,
        ),
    )

    metadata = getattr(graph, "_dagua_scale_route_decision")
    assert torch.isfinite(pos).all()
    assert metadata["sketch_mode"] == "bounded"
    assert metadata["decision"]["strategy"] == "FIELD"


def test_scale_dispatch_uses_declared_topology_bypass() -> None:
    """Engine dispatch skips exact SCC/depth when topology is declared."""
    graph = _wide_layered_dag(4)
    pos = dagua.layout(
        graph,
        LayoutConfig(
            algorithm_params={
                "scale_node_gate": 10,
                "scale_edge_gate": 200,
                "scale_depth_cap": 8,
                "scale_declared_topology": "directed_acyclic",
                "scale_declared_depth": 2,
            },
            seed=42,
        ),
    )

    metadata = getattr(graph, "_dagua_scale_route_decision")
    assert torch.isfinite(pos).all()
    assert metadata["sketch_mode"] == "declared"
    assert metadata["decision"]["strategy"] == "LAYERS"


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


def test_above_gate_layers_dispatches_to_layers_strategy() -> None:
    """Above-gate acyclic layouts route through the tensor-native LAYERS path."""
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

    assert torch.isfinite(pos).all()
    assert pos.shape == (12, 2)
    metadata = getattr(graph, "_dagua_scale_route_decision")
    assert metadata["decision"]["strategy"] == "LAYERS"
    assert metadata["decision"]["reason_codes"] == ["acyclic", "depth_cap_passed"]
    assert metadata["layers"]["num_layers"] == 3


def test_above_gate_field_dispatches_to_field_strategy() -> None:
    """Above-gate cyclic layouts route to FIELD and no longer use legacy fallback."""
    graph = _cyclic_er_like(8)
    pos = dagua.layout(
        graph,
        LayoutConfig(
            algorithm_params={
                "scale_node_gate": 5,
                "scale_edge_gate": 200,
                "scale_depth_cap": 8,
                "field_coarsest_target": 4,
                "field_coarsest_solver": "stress_sgd",
                "field_refine_steps": 1,
                "field_max_grid_axis": 8,
            }
        ),
    )

    assert torch.isfinite(pos).all()
    assert pos.shape == (8, 2)
    metadata = getattr(graph, "_dagua_scale_route_decision")
    assert metadata["decision"]["strategy"] == "FIELD"
    assert "temporary_fallback" not in metadata
    assert metadata["field"]["coarsest_nodes"] <= 4
