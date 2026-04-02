"""Tests for prolongation ops."""

from __future__ import annotations

import torch

from dagua.layout.ops.prolong import (
    DirectMapping,
    DirectMappingConfig,
    NeighborSmoothing,
    NeighborSmoothingConfig,
)
from dagua.layout.ops.state import HierarchyLevel, LayoutProblem, RuntimeContext, SolveState


def _empty_problem(num_nodes: int) -> LayoutProblem:
    """Create an edgeless problem for prolongation tests.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.

    Returns
    -------
    LayoutProblem
        Edgeless problem instance.
    """
    return LayoutProblem(
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        num_nodes=num_nodes,
        seed=11,
    )


def test_direct_mapping_prolongs_to_the_expected_fine_node_count() -> None:
    """DirectMapping should expand coarse coordinates to the mapped fine graph."""
    hierarchy = [
        HierarchyLevel(
            num_nodes=2,
            num_fine=4,
            fine_to_coarse=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        )
    ]
    state = SolveState(
        pos=torch.tensor([[1.0, 2.0], [5.0, 6.0]], dtype=torch.float32),
        hierarchy=hierarchy,
    )

    DirectMapping(DirectMappingConfig(jitter_scale=0.0)).apply(
        _empty_problem(4),
        state,
        RuntimeContext(),
    )

    assert state.pos is not None
    assert state.pos.shape == (4, 2)
    assert torch.equal(
        state.pos,
        torch.tensor(
            [
                [1.0, 2.0],
                [1.0, 2.0],
                [5.0, 6.0],
                [5.0, 6.0],
            ],
            dtype=torch.float32,
        ),
    )


def test_neighbor_smoothing_moves_nodes_toward_neighbor_means() -> None:
    """NeighborSmoothing should pull each node toward the mean of its neighbors."""
    state = SolveState(
        pos=torch.tensor(
            [
                [0.0, 0.0],
                [10.0, 0.0],
                [20.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        adjacency=[[1], [0, 2], [1]],
    )

    NeighborSmoothing(NeighborSmoothingConfig(blend_factor=0.5)).apply(
        _empty_problem(3),
        state,
        RuntimeContext(),
    )

    assert state.pos is not None
    assert torch.allclose(
        state.pos,
        torch.tensor(
            [
                [5.0, 0.0],
                [10.0, 0.0],
                [15.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    )


def test_direct_mapping_respects_nontrivial_mapping_indices() -> None:
    """DirectMapping should copy each coarse coordinate to its mapped fine nodes."""

    hierarchy = [
        HierarchyLevel(
            num_nodes=3,
            num_fine=5,
            fine_to_coarse=torch.tensor([2, 0, 1, 2, 1], dtype=torch.long),
        )
    ]
    state = SolveState(
        pos=torch.tensor([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]], dtype=torch.float32),
        hierarchy=hierarchy,
    )

    result = DirectMapping(DirectMappingConfig(jitter_scale=0.0)).apply(
        _empty_problem(5),
        state,
        RuntimeContext(),
    )

    assert result.pos is not None
    torch.testing.assert_close(
        result.pos,
        torch.tensor(
            [[5.0, 5.0], [1.0, 1.0], [3.0, 3.0], [5.0, 5.0], [3.0, 3.0]],
            dtype=torch.float32,
        ),
    )


def test_neighbor_smoothing_leaves_isolated_nodes_unchanged() -> None:
    """NeighborSmoothing should skip nodes without neighbors."""

    state = SolveState(
        pos=torch.tensor([[0.0, 0.0], [4.0, 0.0], [10.0, 0.0]], dtype=torch.float32),
        adjacency=[[1], [0], []],
    )

    result = NeighborSmoothing(NeighborSmoothingConfig(blend_factor=0.25)).apply(
        _empty_problem(3),
        state,
        RuntimeContext(),
    )

    assert result.pos is not None
    torch.testing.assert_close(result.pos[2], torch.tensor([10.0, 0.0]))
