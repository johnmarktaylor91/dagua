"""Tests for annealing ops."""

from __future__ import annotations

import math

import pytest
import torch

from dagua.layout.ops.anneal import (
    AdaptiveCool,
    AdaptiveCoolConfig,
    EarlyExaggeration,
    EarlyExaggerationConfig,
    ExponentialCool,
    ExponentialCoolConfig,
    IdealLengthDecay,
    IdealLengthDecayConfig,
    LinearCool,
    LinearCoolConfig,
    LRDecay,
    LRDecayConfig,
    PerNodeTemperature,
    PerNodeTemperatureConfig,
    PhaseSchedule,
    PhaseScheduleConfig,
    PhaseSpec,
    ReduceLROnPlateau,
    ReduceLROnPlateauConfig,
    SmoothStepsSchedule,
    SmoothStepsScheduleConfig,
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


def _make_optimizer(lr: float = 1.0) -> torch.optim.Optimizer:
    """Create a one-parameter SGD optimizer for scheduler tests.

    Parameters
    ----------
    lr : float, default=1.0
        Initial learning rate.

    Returns
    -------
    torch.optim.Optimizer
        Optimizer with a single scalar parameter.
    """
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    return torch.optim.SGD([parameter], lr=lr)


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


def test_linear_cool_clamps_temperature_at_zero() -> None:
    """LinearCool should never produce negative temperatures."""

    state = SolveState(temperature=0.1)

    result = LinearCool(LinearCoolConfig(rate=0.5)).apply(_make_problem(), state, RuntimeContext())

    assert result.temperature == pytest.approx(0.0, abs=1.0e-8)


def test_exponential_cool_factor_one_leaves_temperature_unchanged() -> None:
    """ExponentialCool with factor ``1`` should be a no-op."""

    state = SolveState(temperature=3.75)

    result = ExponentialCool(ExponentialCoolConfig(factor=1.0)).apply(
        _make_problem(),
        state,
        RuntimeContext(),
    )

    assert result.temperature == pytest.approx(3.75, rel=1.0e-6)


def test_adaptive_cool_increases_temperature_when_force_norm_improves() -> None:
    """AdaptiveCool should heat up when the force norm drops materially."""

    state = SolveState(
        temperature=2.0,
        forces=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        old_forces=torch.tensor([[2.0, 0.0]], dtype=torch.float32),
    )

    result = AdaptiveCool(AdaptiveCoolConfig(up_factor=1.2, down_factor=0.8)).apply(
        _make_problem(),
        state,
        RuntimeContext(),
    )

    assert result.temperature == pytest.approx(2.4, rel=1.0e-6)


def test_adaptive_cool_decreases_temperature_when_force_norm_grows() -> None:
    """AdaptiveCool should cool down when the force norm gets worse."""

    state = SolveState(
        temperature=2.0,
        forces=torch.tensor([[3.0, 4.0]], dtype=torch.float32),
        old_forces=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
    )

    result = AdaptiveCool(AdaptiveCoolConfig(up_factor=1.2, down_factor=0.8)).apply(
        _make_problem(),
        state,
        RuntimeContext(),
    )

    assert result.temperature == pytest.approx(1.6, rel=1.0e-6)


def test_adaptive_cool_uses_old_forces_as_the_initial_reference() -> None:
    """AdaptiveCool should consult ``old_forces`` on the first update when no history exists."""

    state = SolveState(
        temperature=1.5,
        forces=torch.tensor([[0.5, 0.0]], dtype=torch.float32),
        old_forces=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
    )

    result = AdaptiveCool().apply(_make_problem(), state, RuntimeContext())

    assert result.temperature is not None
    assert float(result.temperature) > 1.5


def test_per_node_temperature_initializes_from_configured_init_temp() -> None:
    """PerNodeTemperature should seed every node with ``init_temp`` when history is absent."""

    state = SolveState()

    result = PerNodeTemperature(PerNodeTemperatureConfig(init_temp=7.0)).apply(
        _make_problem(num_nodes=3),
        state,
        RuntimeContext(),
    )

    torch.testing.assert_close(
        result.extras["local_temperatures"],
        torch.full((3,), 7.0, dtype=torch.float32),
    )


def test_per_node_temperature_respects_the_min_temp_floor() -> None:
    """Per-node temperatures should not fall below the configured floor."""

    op = PerNodeTemperature(PerNodeTemperatureConfig(init_temp=1.0, min_temp=0.25))
    state = SolveState(
        forces=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        old_forces=torch.zeros((1, 2), dtype=torch.float32),
    )
    state = op.apply(_make_problem(num_nodes=1), state, RuntimeContext())
    state.old_forces = state.forces.clone()
    state.forces = -10.0 * state.forces

    result = op.apply(_make_problem(num_nodes=1), state, RuntimeContext())

    assert float(result.extras["local_temperatures"][0].item()) >= 0.25


def test_per_node_temperature_handles_zero_node_graphs() -> None:
    """PerNodeTemperature should store an empty tensor for empty problems."""

    result = PerNodeTemperature().apply(_make_problem(num_nodes=0), SolveState(), RuntimeContext())

    assert result.extras["local_temperatures"].shape == (0,)


def test_phase_schedule_stores_phase_specific_parameters() -> None:
    """PhaseSchedule should expose the selected phase metadata in ``state.extras``."""

    op = PhaseSchedule(
        PhaseScheduleConfig(
            phases=[
                PhaseSpec(name="warmup", iterations=2, temperature=3.0, attraction=0.5),
                PhaseSpec(name="cooldown", iterations=2, temperature=1.0, damping_mult=0.25),
            ]
        )
    )

    result = op.apply(_make_problem(), SolveState(step=3), RuntimeContext())
    phase_state = result.extras["phase_schedule_state"]

    assert phase_state["name"] == "cooldown"
    assert phase_state["temperature"] == pytest.approx(1.0)
    assert phase_state["damping_mult"] == pytest.approx(0.25)


def test_phase_schedule_negative_steps_clamp_to_the_first_phase() -> None:
    """Negative steps should behave like the first phase step."""

    op = PhaseSchedule(
        PhaseScheduleConfig(
            phases=[
                PhaseSpec(name="start", iterations=3, temperature=2.0),
                PhaseSpec(name="end", iterations=3, temperature=1.0),
            ]
        )
    )

    result = op.apply(_make_problem(), SolveState(step=-5), RuntimeContext())

    assert result.temperature == pytest.approx(2.0)
    assert result.extras["phase_schedule_state"]["name"] == "start"


def test_phase_schedule_with_no_phases_is_a_noop() -> None:
    """An empty phase schedule should leave the state untouched."""

    state = SolveState(step=4, temperature=9.0)

    result = PhaseSchedule(PhaseScheduleConfig(phases=[])).apply(
        _make_problem(),
        state,
        RuntimeContext(),
    )

    assert result.temperature == pytest.approx(9.0)
    assert "phase_schedule_state" not in result.extras


@pytest.mark.parametrize("step", [0, 10])
def test_smooth_steps_schedule_hits_edge_keyframes_exactly(step: int) -> None:
    """SmoothStepsSchedule should match the configured endpoint keyframes exactly."""

    op = SmoothStepsSchedule(SmoothStepsScheduleConfig(keyframes={"w": {0: 1.0, 10: 3.0}}))

    result = op.apply(_make_problem(), SolveState(step=step), RuntimeContext())

    assert result.extras["criterion_weights"]["w"] == pytest.approx(1.0 if step == 0 else 3.0)


def test_smooth_steps_schedule_interpolates_between_keyframes() -> None:
    """SmoothStepsSchedule should smoothly interpolate interior steps."""

    op = SmoothStepsSchedule(SmoothStepsScheduleConfig(keyframes={"w": {0: 0.0, 10: 1.0}}))

    result = op.apply(_make_problem(), SolveState(step=5), RuntimeContext())

    assert result.extras["criterion_weights"]["w"] == pytest.approx(0.5, rel=1.0e-6)


def test_smooth_steps_schedule_supports_sequence_keyframes() -> None:
    """SmoothStepsSchedule should accept sequence-based keyframes."""

    op = SmoothStepsSchedule(SmoothStepsScheduleConfig(keyframes={"w": [(0, 1.0), (4, 5.0)]}))

    result = op.apply(_make_problem(), SolveState(step=2), RuntimeContext())

    assert 1.0 < result.extras["criterion_weights"]["w"] < 5.0


def test_weight_annealing_uses_weight_fns_when_schedule_fns_are_absent() -> None:
    """WeightAnnealing should fall back to ``weight_fns`` on the schedule object."""

    annealing = AnnealingSchedule(weight_fns={"alpha": lambda step, total: float(step * total)})
    state = SolveState(annealing=annealing, step=2, total_steps=5)

    result = WeightAnnealing().apply(_make_problem(), state, RuntimeContext())

    assert result.annealing is not None
    assert result.annealing.current_weights == {"alpha": 10.0}


def test_weight_annealing_is_a_noop_without_an_annealing_schedule() -> None:
    """WeightAnnealing should not fail or create state when annealing is absent."""

    state = SolveState(step=3, total_steps=6)

    result = WeightAnnealing().apply(_make_problem(), state, RuntimeContext())

    assert result.annealing is None


def test_lr_decay_exponential_mode_matches_geometric_interpolation() -> None:
    """LRDecay exponential mode should match geometric interpolation."""

    optimizer = _make_optimizer(lr=1.0)
    state = SolveState(optimizer=optimizer, step=4, total_steps=9)

    LRDecay(LRDecayConfig(mode="exponential", start_lr=1.0, end_lr=0.1)).apply(
        _make_problem(),
        state,
        RuntimeContext(),
    )

    expected = 1.0 * ((0.1 / 1.0) ** ((4 + 1) / 9))
    assert optimizer.param_groups[0]["lr"] == pytest.approx(expected, rel=1.0e-6)


def test_lr_decay_caches_the_default_end_lr_in_state() -> None:
    """LRDecay should store the derived default end LR in ``state.extras``."""

    optimizer = _make_optimizer(lr=0.8)
    state = SolveState(optimizer=optimizer, step=0, total_steps=16)

    LRDecay().apply(_make_problem(), state, RuntimeContext())

    assert "lr_decay_end_lr" in state.extras
    assert float(state.extras["lr_decay_end_lr"]) > 0.0


def test_lr_decay_clamps_to_the_end_lr_after_total_steps() -> None:
    """Once the step exceeds the horizon, LRDecay should stay at ``end_lr``."""

    optimizer = _make_optimizer(lr=0.6)
    state = SolveState(optimizer=optimizer, step=20, total_steps=10)

    LRDecay(LRDecayConfig(mode="linear", start_lr=0.6, end_lr=0.2)).apply(
        _make_problem(),
        state,
        RuntimeContext(),
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.2, rel=1.0e-6)


def test_early_exaggeration_uses_the_configured_multiplier() -> None:
    """EarlyExaggeration should store the configured exaggeration factor."""

    result = EarlyExaggeration(EarlyExaggerationConfig(multiplier=5.0, until_step=10)).apply(
        _make_problem(),
        SolveState(step=4),
        RuntimeContext(),
    )

    assert result.extras["exaggeration"] == pytest.approx(5.0)


def test_early_exaggeration_applies_to_negative_steps() -> None:
    """Negative steps are still inside the early-exaggeration phase."""

    result = EarlyExaggeration(EarlyExaggerationConfig(multiplier=3.0, until_step=2)).apply(
        _make_problem(),
        SolveState(step=-1),
        RuntimeContext(),
    )

    assert result.extras["exaggeration"] == pytest.approx(3.0)


def test_reduce_lr_on_plateau_reduces_lr_after_patience_is_exhausted() -> None:
    """ReduceLROnPlateau should lower LR after repeated non-improving checkpoints."""

    optimizer = _make_optimizer(lr=1.0)
    op = ReduceLROnPlateau(ReduceLROnPlateauConfig(factor=0.5, patience=0, min_lr=0.01))
    state = SolveState(optimizer=optimizer, prev_loss=1.0)

    for step, loss in ((0, 1.0), (10, 2.0)):
        state.step = step
        state.prev_loss = loss
        op.apply(_make_problem(), state, RuntimeContext())

    assert optimizer.param_groups[0]["lr"] < 1.0


def test_reduce_lr_on_plateau_respects_the_factor_configuration() -> None:
    """ReduceLROnPlateau should apply the configured multiplicative drop."""

    optimizer = _make_optimizer(lr=1.0)
    op = ReduceLROnPlateau(ReduceLROnPlateauConfig(factor=0.25, patience=0, min_lr=0.01))
    state = SolveState(optimizer=optimizer, prev_loss=1.0)

    for step, loss in ((0, 1.0), (10, 2.0)):
        state.step = step
        state.prev_loss = loss
        op.apply(_make_problem(), state, RuntimeContext())

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.25, rel=1.0e-6)


def test_reduce_lr_on_plateau_respects_the_min_lr_floor() -> None:
    """ReduceLROnPlateau should stop decaying once the configured floor is reached."""

    optimizer = _make_optimizer(lr=1.0)
    op = ReduceLROnPlateau(ReduceLROnPlateauConfig(factor=0.5, patience=0, min_lr=0.2))
    state = SolveState(optimizer=optimizer, prev_loss=1.0)

    for step, loss in ((0, 1.0), (10, 2.0), (20, 3.0), (30, 4.0), (40, 5.0)):
        state.step = step
        state.prev_loss = loss
        op.apply(_make_problem(), state, RuntimeContext())

    assert optimizer.param_groups[0]["lr"] >= 0.2
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.2, rel=1.0e-6)


def test_ideal_length_decay_multiplies_the_existing_length_each_call() -> None:
    """IdealLengthDecay should apply ``decay_factor`` on every call."""

    state = SolveState(extras={"ideal_length": 8.0})
    op = IdealLengthDecay(IdealLengthDecayConfig(decay_factor=0.5))

    op.apply(_make_problem(), state, RuntimeContext())
    op.apply(_make_problem(), state, RuntimeContext())

    assert state.extras["ideal_length"] == pytest.approx(2.0, rel=1.0e-6)


def test_ideal_length_decay_initializes_from_spring_length_mean() -> None:
    """IdealLengthDecay should bootstrap from ``state.spring_lengths`` when needed."""

    state = SolveState(spring_lengths=torch.tensor([2.0, 4.0, 6.0], dtype=torch.float32))

    result = IdealLengthDecay(IdealLengthDecayConfig(decay_factor=0.5)).apply(
        _make_problem(),
        state,
        RuntimeContext(),
    )

    assert result.extras["ideal_length"] == pytest.approx(2.0, rel=1.0e-6)


def test_ideal_length_decay_respects_the_minimum_floor() -> None:
    """IdealLengthDecay should not decay below the built-in minimum."""

    state = SolveState(extras={"ideal_length": 1.0e-8})

    result = IdealLengthDecay(IdealLengthDecayConfig(decay_factor=0.1)).apply(
        _make_problem(),
        state,
        RuntimeContext(),
    )

    assert result.extras["ideal_length"] == pytest.approx(1.0e-6, rel=1.0e-6)
