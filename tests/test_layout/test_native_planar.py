"""Tests for the sprint-20g native planar pipeline."""

from __future__ import annotations

import copy

import networkx as nx
import pytest
import torch

from dagua.config import LayoutConfig
from dagua.eval.graphs import (
    _make_hexagonal_lattice_graph,
    _make_sierpinski_graph,
    _random_dag,
    get_test_graphs,
)
from dagua.graph import DaguaGraph
from dagua.layout import layout
from dagua.layout.graph_classify import classify_graph
from dagua.layout.ops.pipelines.dagua_native import _choose_native_pipeline
from dagua.layout.ops.pipelines.native_planar import (
    PlanarityFailure,
    _FacePreservingConstraint,
    build_planar_pipeline,
    layout_native_planar_pipeline,
)
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.metrics import composite, count_crossings, full


def _edge_index_from_networkx(graph: nx.Graph) -> torch.Tensor:
    """Return a Dagua edge tensor from a NetworkX graph.

    Parameters
    ----------
    graph : nx.Graph
        NetworkX graph with integer node ids.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    edges = list(graph.edges())
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _benchmark_graph(name: str) -> DaguaGraph:
    """Return a copied benchmark graph with node sizes computed.

    Parameters
    ----------
    name : str
        Benchmark graph name.

    Returns
    -------
    DaguaGraph
        Graph ready for layout.
    """
    graphs = {entry.name: entry.graph for entry in get_test_graphs()}
    graph = copy.deepcopy(graphs[name])
    graph.compute_node_sizes()
    return graph


def _composite_score(graph: DaguaGraph, config: LayoutConfig) -> tuple[float, int]:
    """Return composite score and exact crossing count for one layout.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to lay out.
    config : LayoutConfig
        Layout configuration.

    Returns
    -------
    tuple[float, int]
        Composite score and exact crossing count.
    """
    pos = layout(graph, config)
    metrics = full(
        pos,
        graph.edge_index,
        node_sizes=graph.node_sizes,
        crossing_samples=50_000,
        neighborhood_samples=1000,
    )
    return float(composite(metrics)), int(count_crossings(pos, graph.edge_index))


def test_classify_graph_flags_hexagonal_planar_and_k5_non_planar() -> None:
    """Classifier should populate exact planarity metadata."""
    hex_graph = _make_hexagonal_lattice_graph(rows=6, cols=7)
    complete = nx.complete_graph(5)

    hex_structure = classify_graph(hex_graph.edge_index, hex_graph.num_nodes)
    complete_structure = classify_graph(
        _edge_index_from_networkx(complete),
        complete.number_of_nodes(),
    )

    assert hex_structure.is_planar
    assert hex_structure.planar_embedding is not None
    assert not complete_structure.is_planar
    assert complete_structure.planar_embedding is None


def test_schnyder_init_produces_zero_crossing_coordinates() -> None:
    """Planar embedding initialization should produce finite zero-crossing coordinates."""
    graph = nx.cycle_graph(6)
    edge_index = _edge_index_from_networkx(graph)
    structure = classify_graph(edge_index, graph.number_of_nodes())
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=graph.number_of_nodes(),
        structure=structure,
    )
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))

    state = build_planar_pipeline(LayoutConfig(steps=0)).apply(problem, SolveState(), ctx)

    assert state.pos is not None
    assert torch.isfinite(state.pos).all()
    assert count_crossings(state.pos, edge_index) == 0


def test_face_preserving_constraint_detects_inverted_face() -> None:
    """Face-preserving loss should be zero for consistent winding and positive when inverted."""
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    problem = LayoutProblem(edge_index=edge_index, num_nodes=4)
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    constraint = _FacePreservingConstraint(faces=[[0, 1, 2, 3]])
    consistent = SolveState(pos=torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
    inverted = SolveState(pos=torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]))

    assert constraint.evaluate(problem, consistent, ctx).item() == pytest.approx(0.0)
    assert constraint.evaluate(problem, inverted, ctx).item() > 0.0


def test_native_planar_hexagonal_lattice_zero_crossings_and_beats_baseline() -> None:
    """Explicit native_planar should preserve crossings and beat the prior default baseline."""
    graph = _make_hexagonal_lattice_graph(rows=6, cols=7)
    graph.compute_node_sizes()

    baseline_score, _ = _composite_score(
        graph,
        LayoutConfig(seed=42, steps=0, try_planar_first=False),
    )
    planar_score, planar_crossings = _composite_score(
        graph,
        LayoutConfig(algorithm="native_planar", seed=42, steps=10),
    )

    assert planar_crossings == 0
    assert planar_score > baseline_score


def test_auto_dispatch_selects_planar_for_planar_targets_and_layered_for_random_dag() -> None:
    """Native topology dispatch should try planar only for exact planar targets."""
    planar_graph = _benchmark_graph("planar_60")
    hex_graph = _make_hexagonal_lattice_graph(rows=6, cols=7)
    sierpinski_graph = _make_sierpinski_graph(depth=3)
    random_graph = _random_dag(200, 300, seed=42)
    config = LayoutConfig()

    for graph in (hex_graph, sierpinski_graph, planar_graph):
        structure = classify_graph(graph.edge_index, graph.num_nodes)
        assert structure.is_planar
        assert _choose_native_pipeline(structure, config) == "planar"

    random_structure = classify_graph(random_graph.edge_index, random_graph.num_nodes)
    assert not random_structure.is_planar
    assert _choose_native_pipeline(random_structure, config) == "layered_dag"


def test_non_planar_graphs_fall_back_or_fail_only_when_forced() -> None:
    """Default routing should handle non-planar graphs while forced planar raises."""
    small_world = nx.watts_strogatz_graph(100, 6, 0.25, seed=42)
    bipartite = nx.complete_bipartite_graph(5, 5)
    small_world_graph = DaguaGraph.from_networkx(small_world)
    small_world_graph.compute_node_sizes()
    bipartite_edge_index = _edge_index_from_networkx(bipartite)

    pos = layout(small_world_graph, LayoutConfig(seed=42, steps=5))

    assert pos.shape == (small_world_graph.num_nodes, 2)
    assert not classify_graph(bipartite_edge_index, bipartite.number_of_nodes()).is_planar
    with pytest.raises(PlanarityFailure):
        layout_native_planar_pipeline(
            bipartite_edge_index,
            bipartite.number_of_nodes(),
            node_sizes=torch.full((bipartite.number_of_nodes(), 2), 20.0),
            steps=0,
        )
