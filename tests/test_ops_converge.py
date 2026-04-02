"""Tests for convergence ops."""

from __future__ import annotations

import torch

from dagua.layout.ops import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.converge import (
    DisplacementThreshold,
    DisplacementThresholdConfig,
    FixedSteps,
    FixedStepsConfig,
    LRThreshold,
    LRThresholdConfig,
    SlidingWindowRelative,
    SlidingWindowRelativeConfig,
    StallCount,
    StallCountConfig,
    TemperatureThreshold,
    TemperatureThresholdConfig,
)


def _make_problem(num_nodes: int = 3) -> LayoutProblem:
    """Create a minimal layout problem for convergence-op tests.

    Parameters
    ----------
    num_nodes : int, default=3
        Number of nodes in the synthetic test problem.

    Returns
    -------
    LayoutProblem
        Minimal problem instance.
    """
    edge_count = max(num_nodes - 1, 0)
    if edge_count == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.stack(
            [
                torch.arange(0, edge_count, dtype=torch.long),
                torch.arange(1, num_nodes, dtype=torch.long),
            ],
            dim=0,
        )
    return LayoutProblem(edge_index=edge_index, num_nodes=num_nodes)


def test_fixed_steps_sets_total_steps() -> None:
    """FixedSteps should update the state's total step budget."""

    state = SolveState(total_steps=1)

    result = FixedSteps(FixedStepsConfig(n=25)).apply(_make_problem(), state, RuntimeContext())

    assert result.total_steps == 25


def test_displacement_threshold_convergence() -> None:
    """DisplacementThreshold should converge after a sufficiently small move."""

    op = DisplacementThreshold(DisplacementThresholdConfig(threshold=1.0e-3))
    state = SolveState(pos=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32))
    problem = _make_problem(2)

    op.apply(problem, state, RuntimeContext())
    state.pos = torch.tensor([[0.0, 0.0], [1.0, 1.0005]], dtype=torch.float32)
    result = op.apply(problem, state, RuntimeContext())

    assert result.converged


def test_displacement_threshold_converges_for_zero_displacement() -> None:
    """Zero movement between calls should satisfy the displacement threshold."""

    op = DisplacementThreshold(DisplacementThresholdConfig(threshold=1.0e-9))
    problem = _make_problem(2)
    state = SolveState(pos=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32))

    op.apply(problem, state, RuntimeContext())
    result = op.apply(problem, state, RuntimeContext())

    assert result.converged


def test_temperature_threshold_converges_below_minimum() -> None:
    """TemperatureThreshold should mark convergence at low temperature."""

    state = SolveState(temperature=0.001)

    result = TemperatureThreshold(TemperatureThresholdConfig(min_temp=0.005)).apply(
        _make_problem(),
        state,
        RuntimeContext(),
    )

    assert result.converged


def test_sliding_window_relative_converges_for_flat_loss_window() -> None:
    """SlidingWindowRelative should match the NeuLay-style loss-window rule."""

    state = SolveState(
        pos=torch.zeros((4, 2), dtype=torch.float32),
        extras={"loss_window": [10.0, 10.0, 9.9999, 10.0]},
    )

    result = SlidingWindowRelative(SlidingWindowRelativeConfig(window=4, tol=1.0e-4)).apply(
        _make_problem(4),
        state,
        RuntimeContext(),
    )

    assert result.converged


def test_stall_count_convergence_after_repeated_identical_losses() -> None:
    """StallCount should converge after enough repeated loss values."""

    op = StallCount(StallCountConfig(limit=3, rel_threshold=1.0e-6))
    state = SolveState(prev_loss=1.0)
    problem = _make_problem()

    for _ in range(4):
        op.apply(problem, state, RuntimeContext())

    assert state.converged
    assert state.stall_count == 3


def test_stall_count_converges_after_five_identical_losses() -> None:
    """Five repeated losses should trip a StallCount limit of five."""

    op = StallCount(StallCountConfig(limit=5, rel_threshold=1.0e-6))
    state = SolveState(prev_loss=2.0)
    problem = _make_problem()

    for _ in range(6):
        op.apply(problem, state, RuntimeContext())

    assert state.converged
    assert state.stall_count == 5


def test_stall_count_does_not_converge_for_improving_losses() -> None:
    """Meaningful loss improvements should keep the stall counter reset."""

    op = StallCount(StallCountConfig(limit=5, rel_threshold=1.0e-4))
    problem = _make_problem()
    state = SolveState(prev_loss=10.0)

    for loss in [9.0, 8.0, 7.0, 6.0, 5.0]:
        state.prev_loss = loss
        op.apply(problem, state, RuntimeContext())

    assert state.converged is False
    assert state.stall_count == 0


def test_lr_threshold_converges_when_lr_is_small() -> None:
    """LRThreshold should read the current LR from the default optimizer."""

    pos = torch.zeros((1, 2), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.SGD([pos], lr=1.0e-6)
    state = SolveState(pos=pos, optimizer=optimizer)

    result = LRThreshold(LRThresholdConfig(min_lr=1.0e-5)).apply(
        _make_problem(1),
        state,
        RuntimeContext(),
    )

    assert result.converged


def test_lr_threshold_converges_after_scheduler_reaches_minimum() -> None:
    """LRThreshold should detect a scheduled learning rate once it crosses the floor."""

    pos = torch.zeros((1, 2), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.SGD([pos], lr=0.1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
    state = SolveState(pos=pos, optimizer=optimizer)

    for _ in range(5):
        optimizer.step()
        scheduler.step()

    result = LRThreshold(LRThresholdConfig(min_lr=1.0e-5)).apply(
        _make_problem(1),
        state,
        RuntimeContext(),
    )

    assert result.converged


def test_sliding_window_relative_does_not_converge_for_varying_losses() -> None:
    """A visibly changing loss window should remain above the relative flatness rule."""

    state = SolveState(
        pos=torch.zeros((4, 2), dtype=torch.float32),
        extras={"loss_window": [10.0, 9.0, 8.0, 7.0]},
    )

    result = SlidingWindowRelative(SlidingWindowRelativeConfig(window=4, tol=1.0e-4)).apply(
        _make_problem(4),
        state,
        RuntimeContext(),
    )

    assert result.converged is False
