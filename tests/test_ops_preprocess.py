"""Tests for preprocessing layout ops."""

from __future__ import annotations

import pytest
import torch

from dagua.layout.classic._graph_distances import (
    build_directed_adjacency,
    build_undirected_adjacency,
)
from dagua.layout.ops.preprocess import (
    BuildAdjacency,
    BuildAdjacencyConfig,
    ClassifyGraph,
    ClassifyGraphConfig,
    DetectComponents,
    DetectCycles,
    DetectCyclesConfig,
    MakeAcyclic,
)
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState


def _edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    """Build a ``[2, E]`` edge tensor from a Python edge list.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Source-target edge tuples.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)
    src, tgt = zip(*edges)
    return torch.tensor([list(src), list(tgt)], dtype=torch.long)


def _problem(
    edges: list[tuple[int, int]],
    num_nodes: int,
    edge_weights: torch.Tensor | None = None,
) -> LayoutProblem:
    """Create a layout problem for op tests.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Source-target edge tuples.
    num_nodes : int
        Number of graph nodes.
    edge_weights : torch.Tensor | None, optional
        Optional per-edge weights with shape ``[E]``.

    Returns
    -------
    LayoutProblem
        Test problem instance.
    """
    return LayoutProblem(
        edge_index=_edge_index(edges),
        num_nodes=num_nodes,
        edge_weights=edge_weights,
        seed=17,
    )


def test_detect_cycles_and_make_acyclic_follow_cycle_reference() -> None:
    """DetectCycles should produce a mask that MakeAcyclic turns into a DAG."""
    problem = _problem([(0, 1), (1, 2), (2, 0), (2, 3)], num_nodes=4)
    state = SolveState()
    ctx = RuntimeContext()

    DetectCycles(DetectCyclesConfig(method="dfs_then_greedy")).apply(problem, state, ctx)
    MakeAcyclic().apply(problem, state, ctx)

    assert state.back_edge_mask is not None
    assert state.back_edge_mask.dtype == torch.bool
    assert state.back_edge_mask.tolist() == [False, False, True, False]

    acyclic_edges = state.extras["preprocess_edge_index"]
    assert torch.equal(
        acyclic_edges,
        torch.tensor([[0, 1, 0, 2], [1, 2, 2, 3]], dtype=torch.long),
    )


def test_classify_graph_writes_problem_structure_for_small_graphs() -> None:
    """ClassifyGraph should map the reference classifier into problem.structure."""
    problem = _problem([(0, 1), (1, 2), (2, 3)], num_nodes=4)
    state = SolveState()

    ClassifyGraph(ClassifyGraphConfig(large_graph_cutoff=10)).apply(
        problem,
        state,
        RuntimeContext(),
    )

    assert problem.structure is not None
    assert problem.structure.family == "chain"
    assert problem.structure.num_nodes == 4
    assert problem.structure.num_edges == 3
    assert problem.structure.max_degree == 2
    assert problem.structure.num_components == 1
    assert problem.structure.is_dag is True


def test_build_adjacency_matches_unweighted_reference_list() -> None:
    """Default BuildAdjacency should match the undirected classic adjacency helper."""
    problem = _problem([(0, 1), (1, 2), (0, 1)], num_nodes=3)
    state = SolveState()

    BuildAdjacency().apply(problem, state, RuntimeContext())

    expected = build_undirected_adjacency(problem.edge_index, problem.num_nodes)
    assert state.adjacency == expected


def test_build_adjacency_supports_weighted_directed_lists() -> None:
    """Directed weighted adjacency should match the classic directed helper."""
    weights = torch.tensor([2.0, 4.0, 1.5], dtype=torch.float32)
    problem = _problem([(0, 1), (1, 2), (0, 1)], num_nodes=3, edge_weights=weights)
    state = SolveState()

    BuildAdjacency(BuildAdjacencyConfig(directed=True, weighted=True)).apply(
        problem,
        state,
        RuntimeContext(),
    )

    expected = build_directed_adjacency(problem.edge_index, problem.num_nodes, weights)
    assert state.adjacency == expected


def test_build_adjacency_can_emit_dense_and_csr_formats() -> None:
    """Non-list adjacency formats should be built without losing edge costs."""
    weights = torch.tensor([1.0, 3.0], dtype=torch.float32)
    problem = _problem([(0, 1), (1, 2)], num_nodes=3, edge_weights=weights)

    dense_state = SolveState()
    BuildAdjacency(BuildAdjacencyConfig(format="dense", weighted=True)).apply(
        problem,
        dense_state,
        RuntimeContext(),
    )
    assert isinstance(dense_state.adjacency, torch.Tensor)
    assert dense_state.adjacency.shape == (3, 3)
    assert float(dense_state.adjacency[0, 1].item()) == 1.0
    assert float(dense_state.adjacency[1, 0].item()) == 1.0
    assert torch.isinf(dense_state.adjacency[0, 2])

    csr_state = SolveState()
    BuildAdjacency(BuildAdjacencyConfig(format="csr", weighted=True)).apply(
        problem,
        csr_state,
        RuntimeContext(),
    )
    assert isinstance(csr_state.adjacency, dict)
    assert csr_state.adjacency["indptr"].tolist() == [0, 1, 3, 4]
    assert csr_state.adjacency["indices"].tolist() == [1, 0, 2, 1]


def test_detect_components_labels_disconnected_graphs() -> None:
    """DetectComponents should assign stable weak-component labels."""
    problem = _problem([(0, 1), (2, 3)], num_nodes=5)
    state = SolveState()

    DetectComponents().apply(problem, state, RuntimeContext())

    assert state.component_ids is not None
    assert state.component_ids.tolist() == [0, 0, 1, 1, 2]


@pytest.mark.parametrize("format_name", ["list", "dense", "csr"])
def test_build_adjacency_handles_empty_graph_in_all_formats(format_name: str) -> None:
    """BuildAdjacency should support empty graphs across every public format."""

    problem = _problem([], num_nodes=0)
    state = SolveState()

    BuildAdjacency(BuildAdjacencyConfig(format=format_name)).apply(
        problem,
        state,
        RuntimeContext(),
    )

    if format_name == "list":
        assert state.adjacency == []
    elif format_name == "dense":
        assert isinstance(state.adjacency, torch.Tensor)
        assert state.adjacency.shape == (0, 0)
    else:
        assert isinstance(state.adjacency, dict)
        assert state.adjacency["indptr"].tolist() == [0]
        assert state.adjacency["indices"].numel() == 0
        assert state.adjacency["weights"].numel() == 0


def test_build_adjacency_handles_single_node_without_edges() -> None:
    """A single isolated node should yield a valid empty neighbor row."""

    problem = _problem([], num_nodes=1)
    state = SolveState()

    BuildAdjacency().apply(problem, state, RuntimeContext())

    assert state.adjacency == [[]]


def test_build_adjacency_undirected_drops_self_loops_and_dedups_parallel_edges() -> None:
    """Undirected adjacency should ignore self-loops and aggregate duplicates."""

    weights = torch.tensor([9.0, 3.0, 2.0, 5.0], dtype=torch.float32)
    problem = _problem([(0, 0), (0, 1), (0, 1), (1, 2)], num_nodes=3, edge_weights=weights)
    state = SolveState()

    BuildAdjacency(BuildAdjacencyConfig(weighted=True)).apply(problem, state, RuntimeContext())

    assert state.adjacency == [
        [(1, 2.0)],
        [(0, 2.0), (2, 5.0)],
        [(1, 5.0)],
    ]


def test_build_adjacency_directed_preserves_self_loops_and_parallel_edges() -> None:
    """Directed multiplicity mode should keep exact loop and duplicate entries."""

    weights = torch.tensor([9.0, 3.0, 2.0, 5.0], dtype=torch.float32)
    problem = _problem([(0, 0), (0, 1), (0, 1), (1, 2)], num_nodes=3, edge_weights=weights)
    state = SolveState()

    BuildAdjacency(
        BuildAdjacencyConfig(
            directed=True,
            weighted=True,
            dedup="keep_all",
            keep_multiplicity=True,
        )
    ).apply(problem, state, RuntimeContext())

    assert state.adjacency == [
        [(0, 9.0), (1, 2.0), (1, 3.0)],
        [(2, 5.0)],
        [],
    ]


def test_build_adjacency_sum_dedup_aggregates_duplicate_weights() -> None:
    """The ``sum`` dedup mode should add parallel-edge costs together."""

    weights = torch.tensor([1.5, 2.5, 4.0], dtype=torch.float32)
    problem = _problem([(0, 1), (0, 1), (1, 2)], num_nodes=3, edge_weights=weights)
    state = SolveState()

    BuildAdjacency(BuildAdjacencyConfig(weighted=True, dedup="sum")).apply(
        problem,
        state,
        RuntimeContext(),
    )

    assert state.adjacency == [
        [(1, 4.0)],
        [(0, 4.0), (2, 4.0)],
        [(1, 4.0)],
    ]


def test_detect_components_handles_edgeless_pair() -> None:
    """Two isolated nodes should land in distinct singleton components."""

    problem = _problem([], num_nodes=2)
    state = SolveState()

    DetectComponents().apply(problem, state, RuntimeContext())

    assert state.component_ids is not None
    assert state.component_ids.tolist() == [0, 1]


def test_detect_components_labels_large_fan_out_as_one_component() -> None:
    """A star graph should be labeled as a single weak component."""

    edges = [(0, leaf) for leaf in range(1, 100)]
    problem = _problem(edges, num_nodes=100)
    state = SolveState()

    DetectComponents().apply(problem, state, RuntimeContext())

    assert state.component_ids is not None
    assert state.component_ids.unique().tolist() == [0]


def test_build_adjacency_large_fan_out_has_expected_hub_degree() -> None:
    """A 99-leaf hub should expose every leaf in the hub adjacency row."""

    edges = [(0, leaf) for leaf in range(1, 100)]
    problem = _problem(edges, num_nodes=100)
    state = SolveState()

    BuildAdjacency().apply(problem, state, RuntimeContext())

    assert isinstance(state.adjacency, list)
    assert len(state.adjacency[0]) == 99
    assert all(len(state.adjacency[node]) == 1 for node in range(1, 100))


def test_classify_graph_handles_empty_graph() -> None:
    """ClassifyGraph should populate a valid structure for an empty graph."""

    problem = _problem([], num_nodes=0)
    state = SolveState()

    ClassifyGraph(ClassifyGraphConfig(large_graph_cutoff=10)).apply(
        problem,
        state,
        RuntimeContext(),
    )

    assert problem.structure is not None
    assert problem.structure.num_nodes == 0
    assert problem.structure.num_edges == 0
    assert problem.structure.num_components == 0
