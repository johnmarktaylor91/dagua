"""Tests for coarsening ops."""

from __future__ import annotations

import torch

from dagua.layout.ops.coarsen import HeavyEdgeMatching
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState


def _path_problem(num_nodes: int) -> LayoutProblem:
    """Create a path graph for coarsening tests.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the path.

    Returns
    -------
    LayoutProblem
        Path graph with unit node sizes.
    """
    sources = list(range(num_nodes - 1))
    targets = list(range(1, num_nodes))
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    return LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=torch.ones((num_nodes, 2), dtype=torch.float32),
        seed=7,
    )


def test_heavy_edge_matching_builds_valid_hierarchy_for_20_node_graph() -> None:
    """Heavy-edge matching should produce a valid finest-to-coarsest hierarchy."""
    problem = _path_problem(20)
    state = HeavyEdgeMatching().apply(problem, SolveState(), RuntimeContext())

    assert state.hierarchy is not None
    assert state.hierarchy

    expected_fine_nodes = problem.num_nodes
    previous_coarse_nodes = problem.num_nodes
    for level in state.hierarchy:
        assert level.fine_to_coarse is not None
        assert level.edge_index is not None
        assert level.node_sizes is not None
        assert level.num_fine == expected_fine_nodes
        assert level.fine_to_coarse.shape == (level.num_fine,)
        assert level.node_sizes.shape == (level.num_nodes, 2)
        assert level.num_nodes < previous_coarse_nodes
        assert int(level.fine_to_coarse.min().item()) >= 0
        assert int(level.fine_to_coarse.max().item()) == level.num_nodes - 1

        if level.edge_index.numel() > 0:
            assert int(level.edge_index.min().item()) >= 0
            assert int(level.edge_index.max().item()) < level.num_nodes

        expected_fine_nodes = level.num_nodes
        previous_coarse_nodes = level.num_nodes


def test_heavy_edge_matching_first_level_has_fewer_nodes_than_input() -> None:
    """Heavy-edge matching should coarsen the graph on its first hierarchy level."""

    problem = _path_problem(12)

    result = HeavyEdgeMatching().apply(problem, SolveState(), RuntimeContext())

    assert result.hierarchy is not None
    assert result.hierarchy[0].num_nodes < problem.num_nodes
    assert result.hierarchy[0].fine_to_coarse is not None
    assert result.hierarchy[0].fine_to_coarse.shape == (problem.num_nodes,)
