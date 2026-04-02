"""Tests for annealing ops."""

from __future__ import annotations

import math

import pytest
import torch

from dagua.layout.ops.anneal import (
    EarlyExaggeration,
    EarlyExaggerationConfig,
    ExponentialCool,
    ExponentialCoolConfig,
    LinearCool,
    LinearCoolConfig,
    LRDecay,
    LRDecayConfig,
    PerNodeTemperature,
    PerNodeTemperatureConfig,
    PhaseSchedule,
    PhaseScheduleConfig,
    PhaseSpec,
    WeightAnnealing,
)
from dagua.layout.ops.state import AnnealingSchedule, LayoutProblem, RuntimeContext, SolveState


def _make_problem(num_nodes: int = 2) -> LayoutProblem:
    """Create a minimal layout problem for annealing-op tests.

    Parameters
    ----------
    num_nodes : int, default=2
        Number of nodes in the synthetic graph.

    Returns
    -------
    LayoutProblem
        Minimal immutable problem instance.
    """
    return LayoutProblem(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=num_nodes)


def test_linear_cool_decreases_temperature() -> None:
    """LinearCool should subtract the configured per-step rate."""

    state = SolveState(temperature=1.0)

    result = LinearCool(LinearCoolConfig(rate=0.2)).apply(_make_problem(), state, RuntimeContext())

    assert math.isclose(float(result.temperature), 0.8, rel_tol=1.0e-6)


def test_exponential_cool_applies_known_factor() -> None:
    """ExponentialCool should multiply the current temperature by its factor."""

    state = SolveState(temperature=2.5)

    result = ExponentialCool().apply(_make_problem(), state, RuntimeContext())

    assert math.isclose(float(result.temperature), 2.5 * 0.99, rel_tol=1.0e-6)


def test_weight_annealing_updates_current_weights_from_schedule_fns() -> None:
    """WeightAnnealing should refresh the active weight snapshot from schedules."""

    annealing = AnnealingSchedule()
    annealing.schedule_fns = {
        "w_repel": lambda step, total: float(step + total),
        "w_attract": lambda step, total: float(total - step),
    }
    state = SolveState(annealing=annealing, step=3, total_steps=10)

    result = WeightAnnealing().apply(_make_problem(), state, RuntimeContext())

    assert result.annealing is not None
    assert result.annealing.current_weights == {"w_repel": 13.0, "w_attract": 7.0}


def test_linear_cool_reaches_near_zero_after_total_steps_plus_one_updates() -> None:
    """Default linear cooling should drain the initial temperature to zero."""

    state = SolveState(temperature=1.2, total_steps=5)
    op = LinearCool()

    for step in range(6):
        state.step = step
        state = op.apply(_make_problem(), state, RuntimeContext())

    assert float(state.temperature) == pytest.approx(0.0, abs=1.0e-6)


def test_exponential_cool_matches_factor_power_after_multiple_steps() -> None:
    """Repeated exponential cooling should match ``temperature * factor**n``."""

    factor = 0.8
    initial_temperature = 5.0
    state = SolveState(temperature=initial_temperature)
    op = ExponentialCool(ExponentialCoolConfig(factor=factor))

    for _ in range(4):
        state = op.apply(_make_problem(), state, RuntimeContext())

    assert float(state.temperature) == pytest.approx(initial_temperature * (factor**4), rel=1.0e-6)


def test_per_node_temperature_decreases_for_reversed_impulses() -> None:
    """Per-node temperatures should cool when the impulse direction reverses."""

    problem = _make_problem(num_nodes=3)
    op = PerNodeTemperature(PerNodeTemperatureConfig(init_temp=5.0, min_temp=0.1))
    state = SolveState(
        forces=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32),
        old_forces=torch.zeros((3, 2), dtype=torch.float32),
    )

    state = op.apply(problem, state, RuntimeContext())
    initial = state.extras["local_temperatures"].clone()
    state.old_forces = state.forces.clone()
    state.forces = -state.forces.clone()
    state = op.apply(problem, state, RuntimeContext())

    assert torch.all(state.extras["local_temperatures"] < initial)
    assert torch.all(state.extras["local_temperatures"] >= 0.1)


def test_phase_schedule_transitions_at_configured_boundaries() -> None:
    """PhaseSchedule should select the expected phase for each step interval."""

    op = PhaseSchedule(
        PhaseScheduleConfig(
            phases=[
                PhaseSpec(name="warmup", iterations=2, temperature=3.0),
                PhaseSpec(name="settle", iterations=3, temperature=1.5),
                PhaseSpec(name="final", iterations=4, temperature=0.5),
            ]
        )
    )

    states = []
    for step in (0, 1, 2, 4, 5, 9):
        state = op.apply(_make_problem(), SolveState(step=step), RuntimeContext())
        states.append((step, state.temperature, state.extras["phase_schedule_state"]["name"]))

    assert states == [
        (0, 3.0, "warmup"),
        (1, 3.0, "warmup"),
        (2, 1.5, "settle"),
        (4, 1.5, "settle"),
        (5, 0.5, "final"),
        (9, 0.5, "final"),
    ]


def test_weight_annealing_calls_schedule_functions_with_step_and_total_steps() -> None:
    """WeightAnnealing should pass ``(step, total_steps)`` to every schedule function."""

    calls: list[tuple[int, int]] = []

    def _record(step: int, total_steps: int) -> float:
        """Record the annealing-call arguments and return a sentinel weight."""

        calls.append((step, total_steps))
        return float(step - total_steps)

    annealing = AnnealingSchedule(weight_fns={"w_test": _record})
    state = SolveState(annealing=annealing, step=7, total_steps=11)

    WeightAnnealing().apply(_make_problem(), state, RuntimeContext())

    assert calls == [(7, 11)]
    assert state.annealing is not None
    assert state.annealing.current_weights == {"w_test": -4.0}


def test_lr_decay_updates_optimizer_learning_rate() -> None:
    """LRDecay should rewrite the optimizer LR according to the configured schedule."""

    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.SGD([parameter], lr=0.5)
    state = SolveState(optimizer=optimizer, step=4, total_steps=9)

    LRDecay(LRDecayConfig(mode="linear", start_lr=0.5, end_lr=0.1)).apply(
        _make_problem(),
        state,
        RuntimeContext(),
    )

    expected = 0.5 + (0.1 - 0.5) * ((4 + 1) / 9)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(expected, rel=1.0e-6)


def test_early_exaggeration_applies_only_before_threshold_step() -> None:
    """EarlyExaggeration should stop applying its multiplier at ``until_step``."""

    op = EarlyExaggeration(EarlyExaggerationConfig(multiplier=8.0, until_step=3))

    before = op.apply(_make_problem(), SolveState(step=2), RuntimeContext())
    after = op.apply(_make_problem(), SolveState(step=3), RuntimeContext())

    assert before.extras["exaggeration"] == 8.0
    assert after.extras["exaggeration"] == 1.0
