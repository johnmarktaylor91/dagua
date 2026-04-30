"""Regression tests for stress-majorization fidelity modes."""

from __future__ import annotations

import numpy as np
import torch

from dagua.layout.ops.pipelines.stress_majorization import layout_stress_majorization_pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.stress import (
    CURRENT_POSITIONS_KEY,
    InitializeStressMajorizationPositions,
    InitializeStressMajorizationPositionsConfig,
    PrepareStressMajorizationState,
    PrepareStressMajorizationStateConfig,
    SmacofStep,
    SmacofStepConfig,
)


def _disconnected_problem() -> LayoutProblem:
    """Build a small disconnected graph with diameter different from ``sqrt(N)``.

    Returns
    -------
    LayoutProblem
        Six-node graph with three connected components.
    """
    edge_index = torch.tensor(
        [[0, 1, 3], [1, 2, 4]],
        dtype=torch.long,
    )
    return LayoutProblem(edge_index=edge_index, num_nodes=6)


def test_stress_maj_ogdf_fill_uses_sqrt_node_count() -> None:
    """OGDF fidelity mode fills unreachable distances with ``sqrt(N)``."""
    problem = _disconnected_problem()
    state = PrepareStressMajorizationState(
        config=PrepareStressMajorizationStateConfig(distance_fill="ogdf")
    ).apply(problem, SolveState(), RuntimeContext(plan=ExecutionPlan(device="cpu")))

    distances = state.distance_matrix
    assert isinstance(distances, torch.Tensor)
    assert np.isclose(float(distances[0, 3].item()), 6.0**0.5)
    assert np.isclose(float(distances[2, 5].item()), 6.0**0.5)


def test_stress_maj_default_fill_remains_classic_diameter_plus_one() -> None:
    """Default stress-majorization fill preserves dagua's existing policy."""
    problem = _disconnected_problem()
    state = PrepareStressMajorizationState().apply(
        problem,
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device="cpu")),
    )

    distances = state.distance_matrix
    assert isinstance(distances, torch.Tensor)
    assert np.isclose(float(distances[0, 3].item()), 3.0)


def test_stress_maj_ogdf_mode_is_seed_independent() -> None:
    """The OGDF fidelity path disables dagua's seeded warm-start jitter."""
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    first = layout_stress_majorization_pipeline(
        edge_index=edge_index,
        num_nodes=5,
        iterations=3,
        seed=1,
        fidelity_mode="ogdf",
    )
    second = layout_stress_majorization_pipeline(
        edge_index=edge_index,
        num_nodes=5,
        iterations=3,
        seed=99,
        fidelity_mode="ogdf",
    )

    assert isinstance(first, torch.Tensor)
    assert isinstance(second, torch.Tensor)
    assert torch.allclose(first, second)


def test_stress_maj_ogdf_serial_sweep_updates_in_place() -> None:
    """The OGDF serial mode updates later nodes from earlier accepted votes."""
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    problem = LayoutProblem(edge_index=edge_index, num_nodes=3)
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    state = PrepareStressMajorizationState().apply(problem, SolveState(), ctx)
    state = InitializeStressMajorizationPositions(
        config=InitializeStressMajorizationPositionsConfig(jitter_scale=0.0)
    ).apply(problem, state, ctx)
    before = state.extras[CURRENT_POSITIONS_KEY].copy()
    state = SmacofStep(config=SmacofStepConfig(update_mode="ogdf_serial")).apply(
        problem,
        state,
        ctx,
    )

    after = state.extras[CURRENT_POSITIONS_KEY]
    assert isinstance(after, np.ndarray)
    assert not np.allclose(after, before)
