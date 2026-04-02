"""Tests for prolongation ops."""

from __future__ import annotations

import pytest
import torch

from dagua.layout.classic.fmmm import _TYPE_MOON, _TYPE_PLANET, _TYPE_SUN
from dagua.layout.ops.coarsen import SolarHierarchyStep
from dagua.layout.ops.prolong import (
    DirectMapping,
    DirectMappingConfig,
    LambdaInterpolation,
    LambdaInterpolationConfig,
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


def _direct_mapping_state(mapping: torch.Tensor) -> SolveState:
    """Create a direct-mapping solve state for one active hierarchy level.

    Parameters
    ----------
    mapping : torch.Tensor
        Fine-to-coarse mapping with shape ``[N_fine]``.

    Returns
    -------
    SolveState
        Solve state positioned at the coarse level referenced by ``mapping``.
    """
    coarse_nodes = int(mapping.max().item()) + 1 if mapping.numel() > 0 else 0
    hierarchy = [
        HierarchyLevel(
            num_nodes=coarse_nodes,
            num_fine=mapping.numel(),
            fine_to_coarse=mapping,
        )
    ]
    return SolveState(
        pos=torch.tensor([[1.0, 2.0], [5.0, 6.0], [9.0, 10.0]], dtype=torch.float32)[:coarse_nodes],
        hierarchy=hierarchy,
    )


def _lambda_problem(seed: int = 11) -> LayoutProblem:
    """Create a small problem for lambda-interpolation tests.

    Parameters
    ----------
    seed : int, default=11
        Base random seed for prolongation.

    Returns
    -------
    LayoutProblem
        Minimal problem instance with four nodes.
    """
    return LayoutProblem(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        num_nodes=4,
        seed=seed,
    )


def _lambda_hierarchy() -> list[HierarchyLevel]:
    """Create a one-level hierarchy for lambda-interpolation tests.

    Returns
    -------
    list[HierarchyLevel]
        Single active hierarchy level.
    """
    return [
        HierarchyLevel(
            num_nodes=2,
            num_fine=4,
            fine_to_coarse=torch.tensor([0, 0, 0, 1], dtype=torch.long),
        )
    ]


def _lambda_step() -> SolarHierarchyStep:
    """Create deterministic solar-system metadata for exact interpolation tests.

    Returns
    -------
    SolarHierarchyStep
        Metadata with one sun, one planet, one moon, and one neighboring sun.
    """
    return SolarHierarchyStep(
        mapping=torch.tensor([0, 0, 0, 1], dtype=torch.long),
        node_types=[_TYPE_SUN, _TYPE_PLANET, _TYPE_MOON, _TYPE_SUN],
        dedicated_sun=[0, 0, 0, 3],
        dedicated_sun_distance=[0.0, 2.0, 1.0, 0.0],
        pm_nodes=[],
        moon_children=[[], [], [], []],
        lambda_values=[[], [0.5], [0.25], []],
        neighbor_suns=[[], [3], [3], []],
    )


def _lambda_state(seed: int = 11) -> SolveState:
    """Create a solve state for lambda-interpolation tests.

    Parameters
    ----------
    seed : int, default=11
        Unused but kept aligned with helper callers.

    Returns
    -------
    SolveState
        Solve state with two coarse positions and cached solar metadata.
    """
    del seed
    return SolveState(
        pos=torch.tensor([[0.0, 0.0], [10.0, 0.0]], dtype=torch.float32),
        hierarchy=_lambda_hierarchy(),
        extras={"solar_system_steps": [_lambda_step()]},
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


def test_direct_mapping_matches_coarse_positions_at_mapping_indices() -> None:
    """DirectMapping should place each fine node at its mapped coarse position."""
    state = _direct_mapping_state(torch.tensor([2, 0, 1, 2, 1], dtype=torch.long))

    result = DirectMapping(DirectMappingConfig(jitter_scale=0.0)).apply(
        _empty_problem(5),
        state,
        RuntimeContext(),
    )

    assert result.pos is not None
    torch.testing.assert_close(
        result.pos,
        torch.tensor(
            [[9.0, 10.0], [1.0, 2.0], [5.0, 6.0], [9.0, 10.0], [5.0, 6.0]],
            dtype=torch.float32,
        ),
    )


def test_direct_mapping_jitter_scale_zero_gives_exact_copy() -> None:
    """DirectMapping should be exact when jitter is disabled."""
    mapping = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    state = _direct_mapping_state(mapping)

    result = DirectMapping(DirectMappingConfig(jitter_scale=0.0)).apply(
        _empty_problem(4),
        state,
        RuntimeContext(),
    )

    assert result.pos is not None
    torch.testing.assert_close(
        result.pos,
        torch.tensor(
            [[1.0, 2.0], [1.0, 2.0], [5.0, 6.0], [5.0, 6.0]],
            dtype=torch.float32,
        ),
    )


def test_direct_mapping_jitter_adds_noise() -> None:
    """DirectMapping should add Gaussian noise when jitter is enabled."""
    mapping = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    exact_state = _direct_mapping_state(mapping)
    jitter_state = _direct_mapping_state(mapping)

    exact = DirectMapping(DirectMappingConfig(jitter_scale=0.0)).apply(
        _empty_problem(4),
        exact_state,
        RuntimeContext(),
    )
    jittered = DirectMapping(DirectMappingConfig(jitter_scale=1.0)).apply(
        _empty_problem(4),
        jitter_state,
        RuntimeContext(),
    )

    assert exact.pos is not None
    assert jittered.pos is not None
    assert not torch.allclose(jittered.pos, exact.pos)


def test_direct_mapping_is_reproducible_for_same_seed() -> None:
    """DirectMapping should draw the same jitter for repeated runs with the same seed."""
    mapping = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    first = DirectMapping(DirectMappingConfig(jitter_scale=1.0)).apply(
        _empty_problem(4),
        _direct_mapping_state(mapping),
        RuntimeContext(),
    )
    second = DirectMapping(DirectMappingConfig(jitter_scale=1.0)).apply(
        _empty_problem(4),
        _direct_mapping_state(mapping),
        RuntimeContext(),
    )

    assert first.pos is not None
    assert second.pos is not None
    torch.testing.assert_close(first.pos, second.pos)


def test_direct_mapping_handles_multi_level_hierarchy_by_active_node_count() -> None:
    """DirectMapping should choose the hierarchy level matching the current coarse positions."""
    hierarchy = [
        HierarchyLevel(
            num_nodes=4,
            num_fine=8,
            fine_to_coarse=torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long),
        ),
        HierarchyLevel(
            num_nodes=2,
            num_fine=4,
            fine_to_coarse=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        ),
    ]
    state = SolveState(
        pos=torch.tensor([[2.0, 1.0], [6.0, 5.0]], dtype=torch.float32),
        hierarchy=hierarchy,
    )

    result = DirectMapping(DirectMappingConfig(jitter_scale=0.0)).apply(
        _empty_problem(4),
        state,
        RuntimeContext(),
    )

    assert result.pos is not None
    torch.testing.assert_close(
        result.pos,
        torch.tensor(
            [[2.0, 1.0], [2.0, 1.0], [6.0, 5.0], [6.0, 5.0]],
            dtype=torch.float32,
        ),
    )


def test_lambda_interpolation_waggle_factor_zero_gives_exact_interpolation() -> None:
    """LambdaInterpolation should reduce to exact source-target interpolation with zero waggle."""
    state = _lambda_state()

    result = LambdaInterpolation(LambdaInterpolationConfig(waggle_factor=0.0)).apply(
        _lambda_problem(),
        state,
        RuntimeContext(),
    )

    assert result.pos is not None
    torch.testing.assert_close(
        result.pos,
        torch.tensor(
            [[0.0, 0.0], [5.0, 0.0], [2.5, 0.0], [10.0, 0.0]],
            dtype=torch.float32,
        ),
    )


def test_lambda_interpolation_waggle_adds_randomness() -> None:
    """LambdaInterpolation should move points off the exact line when waggle is enabled."""
    exact_state = _lambda_state()
    waggled_state = _lambda_state()

    exact = LambdaInterpolation(LambdaInterpolationConfig(waggle_factor=0.0)).apply(
        _lambda_problem(),
        exact_state,
        RuntimeContext(),
    )
    waggled = LambdaInterpolation(LambdaInterpolationConfig(waggle_factor=0.5)).apply(
        _lambda_problem(),
        waggled_state,
        RuntimeContext(),
    )

    assert exact.pos is not None
    assert waggled.pos is not None
    assert not torch.allclose(waggled.pos[1:], exact.pos[1:])


def test_lambda_interpolation_is_reproducible_for_same_seed() -> None:
    """LambdaInterpolation should be deterministic for the same seed and metadata."""
    first = LambdaInterpolation(LambdaInterpolationConfig(waggle_factor=0.5)).apply(
        _lambda_problem(seed=13),
        _lambda_state(),
        RuntimeContext(),
    )
    second = LambdaInterpolation(LambdaInterpolationConfig(waggle_factor=0.5)).apply(
        _lambda_problem(seed=13),
        _lambda_state(),
        RuntimeContext(),
    )

    assert first.pos is not None
    assert second.pos is not None
    torch.testing.assert_close(first.pos, second.pos)


def test_lambda_interpolation_different_seed_changes_waggled_positions() -> None:
    """LambdaInterpolation should produce different waggle for different seeds."""
    first = LambdaInterpolation(LambdaInterpolationConfig(waggle_factor=0.5)).apply(
        _lambda_problem(seed=13),
        _lambda_state(),
        RuntimeContext(),
    )
    second = LambdaInterpolation(LambdaInterpolationConfig(waggle_factor=0.5)).apply(
        _lambda_problem(seed=29),
        _lambda_state(),
        RuntimeContext(),
    )

    assert first.pos is not None
    assert second.pos is not None
    assert not torch.allclose(first.pos, second.pos)


def test_lambda_interpolation_uses_active_hierarchy_level_in_multi_level_state() -> None:
    """LambdaInterpolation should pick the level matching the current coarse node count."""
    hierarchy = [
        HierarchyLevel(
            num_nodes=4,
            num_fine=8,
            fine_to_coarse=torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long),
        ),
        _lambda_hierarchy()[0],
    ]
    state = SolveState(
        pos=torch.tensor([[0.0, 0.0], [10.0, 0.0]], dtype=torch.float32),
        hierarchy=hierarchy,
        extras={
            "solar_system_steps": [
                SolarHierarchyStep(
                    mapping=torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long),
                    node_types=[_TYPE_SUN] * 8,
                    dedicated_sun=list(range(8)),
                    dedicated_sun_distance=[0.0] * 8,
                    pm_nodes=[],
                    moon_children=[[] for _ in range(8)],
                    lambda_values=[[] for _ in range(8)],
                    neighbor_suns=[[] for _ in range(8)],
                ),
                _lambda_step(),
            ]
        },
    )

    result = LambdaInterpolation(LambdaInterpolationConfig(waggle_factor=0.0)).apply(
        _lambda_problem(),
        state,
        RuntimeContext(),
    )

    assert result.pos is not None
    torch.testing.assert_close(
        result.pos,
        torch.tensor(
            [[0.0, 0.0], [5.0, 0.0], [2.5, 0.0], [10.0, 0.0]],
            dtype=torch.float32,
        ),
    )


def test_neighbor_smoothing_blend_factor_zero_gives_exact_neighbor_mean() -> None:
    """NeighborSmoothing should fully move to the neighbor mean when blend_factor is zero."""
    state = SolveState(
        pos=torch.tensor([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]], dtype=torch.float32),
        adjacency=[[1], [0, 2], [1]],
    )

    result = NeighborSmoothing(NeighborSmoothingConfig(blend_factor=0.0)).apply(
        _empty_problem(3),
        state,
        RuntimeContext(),
    )

    assert result.pos is not None
    torch.testing.assert_close(
        result.pos,
        torch.tensor([[10.0, 0.0], [10.0, 0.0], [10.0, 0.0]], dtype=torch.float32),
    )


def test_neighbor_smoothing_blend_factor_one_keeps_original_positions() -> None:
    """NeighborSmoothing should preserve positions when blend_factor is one."""
    original = torch.tensor([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]], dtype=torch.float32)
    state = SolveState(pos=original.clone(), adjacency=[[1], [0, 2], [1]])

    result = NeighborSmoothing(NeighborSmoothingConfig(blend_factor=1.0)).apply(
        _empty_problem(3),
        state,
        RuntimeContext(),
    )

    assert result.pos is not None
    torch.testing.assert_close(result.pos, original)


def test_neighbor_smoothing_supports_tuple_based_adjacency_entries() -> None:
    """NeighborSmoothing should accept adjacency entries that include edge weights."""
    state = SolveState(
        pos=torch.tensor([[0.0, 0.0], [6.0, 0.0], [12.0, 0.0]], dtype=torch.float32),
        adjacency=[[(1, 2.0)], [(0, 2.0), (2, 3.0)], [(1, 3.0)]],
    )

    result = NeighborSmoothing(NeighborSmoothingConfig(blend_factor=0.5)).apply(
        _empty_problem(3),
        state,
        RuntimeContext(),
    )

    assert result.pos is not None
    torch.testing.assert_close(
        result.pos,
        torch.tensor([[3.0, 0.0], [6.0, 0.0], [9.0, 0.0]], dtype=torch.float32),
    )


def test_neighbor_smoothing_rejects_invalid_adjacency_length() -> None:
    """NeighborSmoothing should reject adjacency lists that do not match the node count."""
    state = SolveState(
        pos=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32),
        adjacency=[[1]],
    )

    with pytest.raises(ValueError, match="state.adjacency length"):
        NeighborSmoothing().apply(_empty_problem(2), state, RuntimeContext())
