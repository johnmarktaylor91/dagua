"""Tests for native stress-layout support operations."""

from __future__ import annotations

import torch

from dagua.layout.ops.native_stress import (
    PrepareWarmStartStressMajorization,
    PrepareWarmStartStressMajorizationConfig,
)
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState


def _path_problem(node_sizes: torch.Tensor | None) -> LayoutProblem:
    """Build a 4-node path problem with optional node sizes.

    Parameters
    ----------
    node_sizes : torch.Tensor | None
        Optional node-size tensor with shape ``[4, 2]``.

    Returns
    -------
    LayoutProblem
        Path graph ``0-1-2-3``.
    """
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    return LayoutProblem(edge_index=edge_index, num_nodes=4, node_sizes=node_sizes, seed=42)


def _warm_start_state() -> SolveState:
    """Build a solve state carrying deterministic warm-start positions.

    Returns
    -------
    SolveState
        State with a fixed ``[4, 2]`` position tensor.
    """
    positions = torch.tensor(
        [[0.0, 0.0], [1.0, 0.2], [2.0, -0.1], [3.0, 0.3]],
        dtype=torch.float32,
    )
    return SolveState(pos=positions)


def test_prepare_warm_start_size_aware_skips_inflation_without_node_sizes() -> None:
    """Default size-aware prep must not index empty radii when sizes are absent.

    Regression: with ``size_aware=True`` (the default), ``node_sizes=None``,
    and at least one edge, the radii vector is empty and the inflation helper
    used to raise ``IndexError``. Absent node sizes there is nothing to
    inflate by, so the result must match ``size_aware=False`` exactly.
    """

    problem = _path_problem(node_sizes=None)

    aware = PrepareWarmStartStressMajorization().apply(
        problem, _warm_start_state(), RuntimeContext()
    )
    unaware = PrepareWarmStartStressMajorization(
        PrepareWarmStartStressMajorizationConfig(size_aware=False)
    ).apply(problem, _warm_start_state(), RuntimeContext())

    assert aware.distance_matrix is not None
    assert unaware.distance_matrix is not None
    torch.testing.assert_close(aware.distance_matrix, unaware.distance_matrix, rtol=0.0, atol=0.0)


def test_prepare_warm_start_size_aware_still_inflates_with_node_sizes() -> None:
    """Size-aware prep must keep inflating adjacent targets when sizes exist."""

    sizes = torch.full((4, 2), 10.0, dtype=torch.float32)
    problem = _path_problem(node_sizes=sizes)

    aware = PrepareWarmStartStressMajorization().apply(
        problem, _warm_start_state(), RuntimeContext()
    )
    unaware = PrepareWarmStartStressMajorization(
        PrepareWarmStartStressMajorizationConfig(size_aware=False)
    ).apply(problem, _warm_start_state(), RuntimeContext())

    assert aware.distance_matrix is not None
    assert unaware.distance_matrix is not None
    # Adjacent pair (0, 1) is inflated by the summed endpoint radii.
    assert float(aware.distance_matrix[0, 1]) > float(unaware.distance_matrix[0, 1])
