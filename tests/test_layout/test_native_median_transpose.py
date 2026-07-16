"""Regression tests for native median + transpose crossing reduction."""

from __future__ import annotations

import copy

import pytest
import torch

from dagua.config import LayoutConfig
from dagua.eval.graphs import get_test_graphs, make_sparse_dense_pair
from dagua.graph import DaguaGraph
from dagua.layout import layout
from dagua.layout.graph_classify import GraphFamily, GraphStructure
from dagua.layout.ops.ordering import (
    MedianSweep,
    MedianSweepConfig,
    TransposeHeuristic,
    TransposeHeuristicConfig,
)
from dagua.layout.ops.pipelines.dagua_native import build_dagua_pipeline
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.metrics import evaluate, sampled_crossing_rate


def _edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    """Build an edge tensor from a Python edge list.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Directed edge tuples.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)
    sources, targets = zip(*edges)
    return torch.tensor([list(sources), list(targets)], dtype=torch.long)


def _crossings_between_adjacent_layers(
    edge_index: torch.Tensor,
    layers: torch.Tensor,
    ordering: torch.Tensor,
) -> int:
    """Count crossings between adjacent-layer edge pairs.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    layers : torch.Tensor
        Layer assignment tensor with shape ``[N]``.
    ordering : torch.Tensor
        Per-node in-layer ordering tensor with shape ``[N]``.

    Returns
    -------
    int
        Number of adjacent-layer crossings.
    """
    edges: list[tuple[int, int, int]] = []
    for source, target in edge_index.t().tolist():
        source_layer = int(layers[source].item())
        target_layer = int(layers[target].item())
        if abs(source_layer - target_layer) != 1:
            continue
        if source_layer < target_layer:
            edges.append((source_layer, source, target))
        else:
            edges.append((target_layer, target, source))

    crossings = 0
    for index, first in enumerate(edges):
        first_layer, first_upper, first_lower = first
        for second_layer, second_upper, second_lower in edges[index + 1 :]:
            if first_layer != second_layer:
                continue
            upper_delta = int(ordering[first_upper].item()) - int(ordering[second_upper].item())
            lower_delta = int(ordering[first_lower].item()) - int(ordering[second_lower].item())
            if upper_delta * lower_delta < 0:
                crossings += 1
    return crossings


def _rank_by_x(pos: torch.Tensor, layers: torch.Tensor) -> torch.Tensor:
    """Return per-node in-layer ranks implied by x coordinates.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    layers : torch.Tensor
        Layer assignment tensor with shape ``[N]``.

    Returns
    -------
    torch.Tensor
        Long tensor with shape ``[N]`` containing x-order ranks.
    """
    ranks = torch.zeros((pos.shape[0],), dtype=torch.long)
    for layer_id in torch.unique(layers).tolist():
        nodes = torch.nonzero(layers == int(layer_id), as_tuple=False).flatten()
        ordered = sorted(nodes.tolist(), key=lambda node: (float(pos[node, 0].item()), node))
        for rank, node in enumerate(ordered):
            ranks[node] = rank
    return ranks


def _benchmark_graph(name: str) -> DaguaGraph:
    """Return a deep copy of one synthetic benchmark graph.

    Parameters
    ----------
    name : str
        Benchmark graph name.

    Returns
    -------
    DaguaGraph
        Graph with node sizes computed.
    """
    graphs = {entry.name: entry.graph for entry in get_test_graphs(max_nodes=250)}
    graph = copy.deepcopy(graphs[name])
    graph.compute_node_sizes()
    return graph


def _crossing_rate(graph: DaguaGraph, pos: torch.Tensor) -> float:
    """Return a deterministic sampled crossing rate for a graph layout.

    Parameters
    ----------
    graph : DaguaGraph
        Graph whose edges are evaluated.
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    float
        Sampled crossing rate.
    """
    metrics = sampled_crossing_rate(pos, graph.edge_index, n_samples=200_000, seed=42)
    return float(metrics["crossing_rate"])


def test_median_sweep_reduces_known_two_layer_crossings_and_updates_positions() -> None:
    """MedianSweep should reduce a known bipartite crossing and project x-slots."""
    problem = LayoutProblem(
        edge_index=_edge_index([(0, 3), (1, 2), (0, 5), (1, 4)]),
        num_nodes=6,
    )
    layers = torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.long)
    initial_ordering = torch.tensor([0, 1, 0, 1, 2, 3], dtype=torch.long)
    state = SolveState(
        layers=layers,
        pos=torch.tensor(
            [
                [0.0, 0.0],
                [10.0, 0.0],
                [0.0, 1.0],
                [10.0, 1.0],
                [20.0, 1.0],
                [30.0, 1.0],
            ],
            dtype=torch.float32,
        ),
    )
    initial_crossings = _crossings_between_adjacent_layers(
        problem.edge_index,
        layers,
        initial_ordering,
    )

    result = MedianSweep(MedianSweepConfig(passes=4)).apply(problem, state, RuntimeContext())

    assert result.ordering is not None
    assert result.pos is not None
    assert _crossings_between_adjacent_layers(problem.edge_index, layers, result.ordering) < (
        initial_crossings
    )
    assert torch.equal(_rank_by_x(result.pos, layers), result.ordering.cpu())


def test_transpose_heuristic_finds_optimal_three_node_adjacent_swap() -> None:
    """TransposeHeuristic should find the exact optimal swap on a 3x3 layer pair."""
    problem = LayoutProblem(
        edge_index=_edge_index([(0, 3), (1, 4), (2, 5)]),
        num_nodes=6,
    )
    layers = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    state = SolveState(
        layers=layers,
        ordering=torch.tensor([0, 1, 2, 1, 0, 2], dtype=torch.long),
        pos=torch.tensor(
            [
                [0.0, 0.0],
                [10.0, 0.0],
                [20.0, 0.0],
                [10.0, 1.0],
                [0.0, 1.0],
                [20.0, 1.0],
            ],
            dtype=torch.float32,
        ),
    )

    result = TransposeHeuristic(TransposeHeuristicConfig(passes=4)).apply(
        problem,
        state,
        RuntimeContext(),
    )

    assert result.ordering is not None
    assert result.pos is not None
    assert result.ordering.tolist() == [1, 0, 2, 1, 0, 2]


def test_dot_mincross_numba_reorder_matches_python(monkeypatch: pytest.MonkeyPatch) -> None:
    """Numba and Python rank reordering produce identical mincross output.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to force the Python fallback after the numba run.

    Returns
    -------
    None
        This test asserts exact ordered-rank identity.
    """
    from dagua.layout.ops import _dot_mincross

    if not _dot_mincross._NUMBA_AVAILABLE:
        pytest.skip("numba is not installed")

    ranks = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    edges = [
        (0, 7),
        (1, 6),
        (2, 5),
        (3, 4),
        (4, 11),
        (5, 10),
        (6, 9),
        (7, 8),
        (0, 5),
        (2, 7),
    ]

    numba_order = _dot_mincross.graphviz_mincross(ranks, edges, iterations=6)
    monkeypatch.setattr(_dot_mincross, "_NUMBA_AVAILABLE", False)
    python_order = _dot_mincross.graphviz_mincross(ranks, edges, iterations=6)

    assert numba_order == python_order


def test_dagua_native_dense_pair_50_reduces_crossings() -> None:
    """Median-transpose ordering should reduce edge-crossing rate on
    ``dense_pair_50`` relative to the same pipeline with median-transpose
    disabled.

    This test originally also asserted that the composite score improved.
    That second assertion was dropped after the audit found that
    composite-level assertions silently double as benchmark-score
    regression tests, which encourages overfitting the algorithm to one
    metric definition. The reduction in crossing rate is the behavioural
    invariant median-transpose is *designed* to deliver; whether that
    translates into a higher composite for any specific graph is a
    metric-weighting question that should be measured by the suite, not
    asserted here.
    """
    _, graph = make_sparse_dense_pair(n=50, seed=42)
    graph.compute_node_sizes()
    # r80: dense_pair_50 is mechanically oriented and infers undirected, so
    # the default router would send it to the undirected portfolio contest,
    # where an sfdp challenger can win both runs and mask the flag under
    # test. Declare the graph directed (the public routing override) so both
    # configs take the exact pre-r80 layered path, keeping this a
    # median-transpose mechanism test.
    graph.is_semantically_directed = True
    baseline_config = LayoutConfig(
        seed=42,
        steps=80,
        use_native_median_transpose=False,
        brandes_koepf_refine=False,
    )
    enabled_config = LayoutConfig(
        seed=42,
        steps=80,
        use_native_median_transpose=True,
        brandes_koepf_refine=False,
    )

    baseline_pos = layout(graph, baseline_config)
    enabled_pos = layout(graph, enabled_config)

    assert _crossing_rate(graph, enabled_pos) < _crossing_rate(graph, baseline_pos)


def test_dagua_native_cyclic_graph_skips_median_transpose_and_preserves_composite() -> None:
    """Cyclic graphs should skip the new phase and keep the previous layout."""
    graph = _benchmark_graph("recurrent_feedback_cell")
    baseline_config = LayoutConfig(
        seed=42,
        steps=40,
        use_native_median_transpose=False,
        brandes_koepf_refine=False,
    )
    enabled_config = LayoutConfig(
        seed=42,
        steps=40,
        use_native_median_transpose=True,
        brandes_koepf_refine=False,
    )

    baseline_pos = layout(graph, baseline_config)
    enabled_pos = layout(graph, enabled_config)
    baseline_composite = float(evaluate(graph, baseline_pos, tier="full")["composite_score"])
    enabled_composite = float(evaluate(graph, enabled_pos, tier="full")["composite_score"])

    assert torch.allclose(enabled_pos, baseline_pos)
    assert _crossing_rate(graph, enabled_pos) == _crossing_rate(graph, baseline_pos)
    assert enabled_composite == baseline_composite


def test_native_pipeline_omits_median_transpose_when_structure_is_cyclic() -> None:
    """The DAG gate should keep the new ordering ops out of cyclic pipelines."""
    config = LayoutConfig(seed=42)
    setattr(
        config,
        "_dagua_native_structure",
        GraphStructure(
            family=GraphFamily.GENERAL,
            num_components=1,
            max_degree=3,
            num_layers=2,
            avg_layer_width=2.0,
            is_planar_hint=True,
            is_acyclic=False,
            is_directed_acyclic=False,
        ),
    )

    op_names = [op.name for op in build_dagua_pipeline(config).ops]

    assert "barycenter_reorder" in op_names
    assert "median_sweep" not in op_names
    assert "transpose_heuristic" not in op_names
