"""Regression tests for native Brandes-Koepf horizontal refinement."""

from __future__ import annotations

import copy

import torch

from dagua.config import LayoutConfig
from dagua.eval.graphs import get_test_graphs
from dagua.graph import DaguaGraph
from dagua.layout import layout
from dagua.layout.graph_classify import GraphFamily, GraphStructure
from dagua.layout.ops.coordinate import (
    BrandesKoepfHorizontalRefine,
    BrandesKoepfHorizontalRefineConfig,
)
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.metrics import composite, full


def _edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    """Build a directed edge tensor from Python edge tuples.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Directed edge tuples.

    Returns
    -------
    torch.Tensor
        Long edge tensor with shape ``[2, E]``.
    """
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)
    sources, targets = zip(*edges)
    return torch.tensor([list(sources), list(targets)], dtype=torch.long)


def _layered_dag_edges(num_layers: int, width: int) -> list[tuple[int, int]]:
    """Return connected multi-edge DAG edges for a layered test graph.

    Parameters
    ----------
    num_layers : int
        Number of layers to create.
    width : int
        Nodes per layer.

    Returns
    -------
    list[tuple[int, int]]
        Directed edges advancing one layer at a time.
    """
    edges: list[tuple[int, int]] = []
    for layer_index in range(num_layers - 1):
        upper_offset = layer_index * width
        lower_offset = (layer_index + 1) * width
        for column in range(width):
            edges.append((upper_offset + column, lower_offset + column))
            edges.append((upper_offset + column, lower_offset + ((column + 1) % width)))
    return edges


def _benchmark_graph(name: str) -> DaguaGraph:
    """Return a deep copy of one benchmark graph with node sizes.

    Parameters
    ----------
    name : str
        Benchmark graph name.

    Returns
    -------
    DaguaGraph
        Graph ready for layout and metric evaluation.
    """
    graphs = {entry.name: entry.graph for entry in get_test_graphs()}
    graph = copy.deepcopy(graphs[name])
    graph.compute_node_sizes()
    return graph


def _composite_and_edge_cv(graph: DaguaGraph, config: LayoutConfig) -> tuple[float, float]:
    """Return composite score and edge-length CV for one graph layout.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to lay out.
    config : LayoutConfig
        Layout configuration to evaluate.

    Returns
    -------
    tuple[float, float]
        ``(composite_score, edge_length_cv)`` for the resulting layout.
    """
    pos = layout(graph, config)
    metrics = full(pos, graph.edge_index, node_sizes=graph.node_sizes)
    return float(composite(metrics)), float(metrics["edge_length_cv"])


def test_brandes_koepf_refinement_preserves_layer_y_assignments() -> None:
    """BK refinement should only rewrite x coordinates for layered DAGs."""
    num_layers = 6
    width = 3
    num_nodes = num_layers * width
    layers = torch.arange(num_layers, dtype=torch.long).repeat_interleave(width)
    y_values = layers.to(dtype=torch.float32) * 11.0
    x_values = torch.tensor([2.0, -1.0, 4.0] * num_layers, dtype=torch.float32)
    state = SolveState(pos=torch.stack((x_values, y_values), dim=1), layers=layers)
    problem = LayoutProblem(
        edge_index=_edge_index(_layered_dag_edges(num_layers=num_layers, width=width)),
        num_nodes=num_nodes,
        node_sizes=torch.full((num_nodes, 2), 1.0, dtype=torch.float32),
    )

    result = BrandesKoepfHorizontalRefine(
        BrandesKoepfHorizontalRefineConfig(
            node_sep=1.0,
            structure=GraphStructure(
                family=GraphFamily.GENERAL,
                num_components=1,
                max_degree=4,
                num_layers=num_layers,
                avg_layer_width=float(width),
                is_planar_hint=False,
                is_directed_acyclic=True,
            ),
        )
    ).apply(
        problem,
        state,
        RuntimeContext(),
    )

    assert result.pos is not None
    assert torch.equal(result.pos[:, 1], y_values)
    assert not torch.allclose(result.pos[:, 0], x_values)
    assert result.extras["brandes_koepf_horizontal_refine_applied"] is True


def test_brandes_koepf_refinement_admits_multi_component_forward_dag() -> None:
    """BK refinement should run on disconnected DAGs with strict forward layers."""
    component_layers = torch.arange(6, dtype=torch.long)
    layers = torch.cat((component_layers, component_layers))
    y_values = layers.to(dtype=torch.float32) * 11.0
    x_values = torch.tensor(
        [0.0, 0.2, -0.1, 0.1, -0.2, 0.0, 8.0, 8.2, 7.9, 8.1, 7.8, 8.0],
        dtype=torch.float32,
    )
    original_pos = torch.stack((x_values, y_values), dim=1)
    problem = LayoutProblem(
        edge_index=_edge_index(
            [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
            ],
        ),
        num_nodes=12,
        node_sizes=torch.full((12, 2), 1.0, dtype=torch.float32),
        structure=GraphStructure(
            family=GraphFamily.GENERAL,
            num_components=2,
            max_degree=2,
            num_layers=6,
            avg_layer_width=2.0,
            is_planar_hint=True,
            is_directed_acyclic=True,
        ),
    )
    state = SolveState(pos=original_pos.clone(), layers=layers)

    result = BrandesKoepfHorizontalRefine().apply(problem, state, RuntimeContext())

    assert result.pos is not None
    assert torch.equal(result.pos[:, 1], y_values)
    assert not torch.allclose(result.pos[:, 0], x_values)
    assert result.extras["brandes_koepf_horizontal_refine_applied"] is True


def test_brandes_koepf_refinement_skips_cyclic_graph() -> None:
    """A back edge in the layer assignment should keep BK refinement off."""
    num_nodes = 6
    layers = torch.arange(num_nodes, dtype=torch.long)
    original_pos = torch.stack(
        (torch.arange(num_nodes, dtype=torch.float32), layers.float()),
        dim=1,
    )
    problem = LayoutProblem(
        edge_index=_edge_index([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]),
        num_nodes=num_nodes,
        node_sizes=torch.full((num_nodes, 2), 1.0, dtype=torch.float32),
    )
    state = SolveState(pos=original_pos.clone(), layers=layers)

    result = BrandesKoepfHorizontalRefine().apply(problem, state, RuntimeContext())

    assert result.pos is not None
    assert torch.equal(result.pos, original_pos)
    assert result.extras["brandes_koepf_horizontal_refine_applied"] is False


def test_dagua_native_hexagonal_lattice_42_bk_improves_edge_cv_or_composite() -> None:
    """Native BK should improve edge CV or composite on the hex lattice target."""
    graph = _benchmark_graph("hexagonal_lattice_42")
    baseline_score, baseline_edge_cv = _composite_and_edge_cv(
        graph,
        LayoutConfig(seed=42, steps=80, brandes_koepf_refine=False),
    )
    enabled_score, enabled_edge_cv = _composite_and_edge_cv(
        graph,
        LayoutConfig(seed=42, steps=80, brandes_koepf_refine=True),
    )

    assert enabled_edge_cv <= baseline_edge_cv or enabled_score > baseline_score


def test_dagua_native_random_dag_200_bk_within_prior_composite_window() -> None:
    """BK should keep random_dag_200 within 99% of the pre-BK native score."""
    graph = _benchmark_graph("random_dag_200")
    baseline_score, _ = _composite_and_edge_cv(
        graph,
        LayoutConfig(seed=42, steps=60, brandes_koepf_refine=False),
    )
    enabled_score, _ = _composite_and_edge_cv(
        graph,
        LayoutConfig(seed=42, steps=60, brandes_koepf_refine=True),
    )

    assert enabled_score >= baseline_score * 0.99
