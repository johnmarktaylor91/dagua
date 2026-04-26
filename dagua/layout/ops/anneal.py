"""Annealing, cooling, and schedule operations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, Mapping, Optional, Sequence, Tuple

import torch

from dagua.layout.ops.base import Op
from dagua.layout.ops.state import AnnealingSchedule, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_DEFAULT_MIN_IDEAL_LENGTH = 1.0e-6
_PLATEAU_EMA_DECAY = 0.5 ** (1.0 / 100.0)
_LINEAR_COOL_INITIAL_KEY = "linear_cool_initial_temperature"
_ADAPTIVE_COOL_PREV_FORCE_KEY = "adaptive_cool_prev_force_norm"
_PER_NODE_SKEW_KEY = "per_node_temperature_skew_gauge"
_PER_NODE_PREV_IMPULSE_KEY = "per_node_temperature_prev_impulse"
_PHASE_STATE_KEY = "phase_schedule_state"
_PLATEAU_SCHEDULER_KEY = "reduce_lr_on_plateau_scheduler"
_PLATEAU_EMA_SUM_KEY = "reduce_lr_on_plateau_ema_weighted_sum"
_PLATEAU_EMA_WEIGHT_KEY = "reduce_lr_on_plateau_ema_total_weight"
_LR_DECAY_START_KEY = "lr_decay_start_lr"
_LR_DECAY_END_KEY = "lr_decay_end_lr"
_MIN_DISTANCE = 1.0e-9
_ROTATION_SINE_THRESHOLD = math.sin((math.pi / 2.0) + (math.pi / 3.0 / 2.0))
_OSCILLATION_COSINE_THRESHOLD = math.cos((math.pi / 2.0) / 2.0)
_ROTATION_SENSITIVITY = 0.01
_OSCILLATION_SENSITIVITY = 0.3


def _gem_update_temperatures(
    temperatures: torch.Tensor,
    current_impulse: torch.Tensor,
    previous_impulse: torch.Tensor,
    skew_gauge: torch.Tensor,
    initial_temperature: float,
    min_distance: float,
    rotation_sine_threshold: float,
    oscillation_cosine_threshold: float,
    rotation_sensitivity: float,
    oscillation_sensitivity: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adjust per-node temperatures with OGDF's oscillation and skew rules.

    Parameters
    ----------
    temperatures : torch.Tensor
        Per-node temperature tensor with shape ``[N]``.
    current_impulse : torch.Tensor
        Current scaled node movements with shape ``[N, 2]``.
    previous_impulse : torch.Tensor
        Previous scaled node movements with shape ``[N, 2]``.
    skew_gauge : torch.Tensor
        Per-node rotation accumulator with shape ``[N]``.
    initial_temperature : float
        Maximum allowed per-node temperature.
    min_distance : float
        Safety floor applied to impulse-norm products.
    rotation_sine_threshold : float
        Sine threshold that triggers skew-gauge growth.
    oscillation_cosine_threshold : float
        Cosine threshold that triggers oscillation cooling.
    rotation_sensitivity : float
        Increment added to the skew gauge after a rotation event.
    oscillation_sensitivity : float
        Multiplier applied to temperature after an oscillation event.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Updated temperatures and skew-gauge values, both with shape ``[N]``.
    """
    current_norm = torch.linalg.norm(current_impulse, dim=1)
    previous_norm = torch.linalg.norm(previous_impulse, dim=1)
    product = current_norm * previous_norm
    valid = product > min_distance

    if not bool(valid.any()):
        return temperatures, skew_gauge

    safe_product = product.clamp(min=min_distance)
    sin_beta = (
        current_impulse[:, 0] * previous_impulse[:, 0]
        - current_impulse[:, 1] * previous_impulse[:, 1]
    ) / safe_product
    cos_beta = (
        current_impulse[:, 0] * previous_impulse[:, 0]
        + current_impulse[:, 1] * previous_impulse[:, 1]
    ) / safe_product

    rotation_mask = valid & (sin_beta > rotation_sine_threshold)
    skew_gauge = torch.where(rotation_mask, skew_gauge + rotation_sensitivity, skew_gauge)

    oscillation_mask = valid & (cos_beta.abs() > oscillation_cosine_threshold)
    oscillation_scale = 1.0 + cos_beta * oscillation_sensitivity
    temperatures = torch.where(oscillation_mask, temperatures * oscillation_scale, temperatures)
    temperatures = temperatures * (1.0 - skew_gauge.abs())
    temperatures = torch.minimum(temperatures, torch.full_like(temperatures, initial_temperature))
    return temperatures, skew_gauge


def _require_temperature(state: SolveState, op_name: str) -> float:
    """Return the global temperature required by a cooling op.

    Parameters
    ----------
    state : SolveState
        Mutable solve state.
    op_name : str
        Name of the operation requesting the temperature.

    Returns
    -------
    float
        Current temperature.

    Raises
    ------
    ValueError
        If ``state.temperature`` is not set.
    """
    if state.temperature is None:
        raise ValueError(f"{op_name} requires state.temperature to be set.")
    return float(state.temperature)


def _require_optimizer(state: SolveState, op_name: str) -> torch.optim.Optimizer:
    """Return the default optimizer required by an annealing op.

    Parameters
    ----------
    state : SolveState
        Mutable solve state.
    op_name : str
        Name of the operation requesting the optimizer.

    Returns
    -------
    torch.optim.Optimizer
        Default optimizer stored in ``state.optimizer``.

    Raises
    ------
    ValueError
        If ``state.optimizer`` is missing.
    TypeError
        If ``state.optimizer`` is not a PyTorch optimizer.
    """
    if state.optimizer is None:
        raise ValueError(f"{op_name} requires state.optimizer to be set.")
    if not isinstance(state.optimizer, torch.optim.Optimizer):
        raise TypeError(f"{op_name} requires a torch.optim.Optimizer in state.optimizer.")
    return state.optimizer


def _schedule_functions(annealing: AnnealingSchedule) -> Mapping[str, Callable[[int, int], float]]:
    """Return annealing schedule functions from either supported attribute name.

    Parameters
    ----------
    annealing : AnnealingSchedule
        Annealing schedule container.

    Returns
    -------
    Mapping[str, Callable[[int, int], float]]
        Resolved schedule mapping.
    """
    schedule_fns = getattr(annealing, "schedule_fns", None)
    if isinstance(schedule_fns, Mapping):
        return schedule_fns
    return annealing.weight_fns


def _smooth_step(x: float) -> float:
    """Evaluate the cubic Hermite smooth-step interpolant.

    Parameters
    ----------
    x : float
        Raw interpolation fraction.

    Returns
    -------
    float
        Smoothed fraction in ``[0, 1]``.
    """
    x_clamped = min(max(x, 0.0), 1.0)
    return 3.0 * x_clamped * x_clamped - 2.0 * x_clamped * x_clamped * x_clamped


def _coerce_keyframes(spec: Any) -> tuple[list[int], list[float]]:
    """Normalize one criterion schedule specification into times and values.

    Parameters
    ----------
    spec : Any
        Keyframe specification. Supported formats are ``{time: value}``,
        ``{"times": [...], "values": [...]}``, or ``[(time, value), ...]``.

    Returns
    -------
    tuple[list[int], list[float]]
        Monotone times and aligned values.

    Raises
    ------
    TypeError
        If the specification format is unsupported.
    ValueError
        If the keyframes are empty or malformed.
    """
    if isinstance(spec, Mapping):
        if "times" in spec and "values" in spec:
            times = [int(time) for time in spec["times"]]
            values = [float(value) for value in spec["values"]]
        else:
            items = sorted((int(time), float(value)) for time, value in spec.items())
            times = [time for time, _ in items]
            values = [value for _, value in items]
    elif isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
        items = [(int(time), float(value)) for time, value in spec]
        items.sort(key=lambda item: item[0])
        times = [time for time, _ in items]
        values = [value for _, value in items]
    else:
        raise TypeError("SmoothStepsSchedule keyframes must be mappings or sequences of pairs.")

    if not times or len(times) != len(values):
        raise ValueError("SmoothStepsSchedule keyframes must be non-empty and aligned.")
    if any(right < left for left, right in zip(times, times[1:])):
        raise ValueError("SmoothStepsSchedule keyframe times must be non-decreasing.")
    return times, values


def _evaluate_smooth_schedule(times: Sequence[int], values: Sequence[float], step: int) -> float:
    """Evaluate a smooth-step schedule at the given step.

    Parameters
    ----------
    times : sequence[int]
        Monotone keyframe times.
    values : sequence[float]
        Keyframe values aligned with ``times``.
    step : int
        Current optimization step.

    Returns
    -------
    float
        Interpolated value at ``step``.
    """
    if step <= times[0]:
        return float(values[0])
    for index in range(len(times) - 1):
        left = int(times[index])
        right = int(times[index + 1])
        if step <= right:
            span = max(right - left, 1)
            fraction = float(step - left) / float(span)
            return float(values[index]) + _smooth_step(fraction) * (
                float(values[index + 1]) - float(values[index])
            )
    return float(values[-1])


def _optimizer_lr(optimizer: torch.optim.Optimizer) -> float:
    """Return the learning rate of the first optimizer param group.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer whose learning rate should be inspected.

    Returns
    -------
    float
        Learning rate from the first parameter group.
    """
    return float(optimizer.param_groups[0]["lr"])


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Assign a learning rate to every optimizer parameter group.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer to mutate.
    lr : float
        New learning rate.

    Returns
    -------
    None
        The optimizer is updated in place.
    """
    for group in optimizer.param_groups:
        group["lr"] = lr


def _default_lr_decay_end_lr(
    start_lr: float,
    total_steps: int,
    end_lr_multiplier: float,
) -> float:
    """Compute the LinLog-style default final learning rate.

    Parameters
    ----------
    start_lr : float
        Initial learning rate.
    total_steps : int
        Planned optimization horizon.
    end_lr_multiplier : float
        Fraction of ``start_lr`` used by the linear fallback endpoint.

    Returns
    -------
    float
        Default final learning rate.
    """
    return max(
        start_lr * end_lr_multiplier,
        start_lr / math.sqrt(float(max(total_steps, 1))),
    )


def _state_device(problem: LayoutProblem, state: SolveState) -> torch.device:
    """Pick a device for new schedule tensors.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs.
    state : SolveState
        Mutable solve state.

    Returns
    -------
    torch.device
        Device chosen from the active state tensors when possible.
    """
    if state.pos is not None:
        return state.pos.device
    if state.forces is not None:
        return state.forces.device
    if problem.edge_index.numel() > 0:
        return problem.edge_index.device
    return torch.device("cpu")


@dataclass(frozen=True)
class LinearCoolConfig:
    """Configuration for :class:`LinearCool`.

    Parameters
    ----------
    rate : float or None, default=None
        Per-step temperature decrement. When ``None``, the first observed
        temperature is cooled linearly to zero over ``total_steps + 1`` steps.
    """

    rate: Optional[float] = None


@register_op
@dataclass(frozen=True)
class LinearCool(Op):
    """Apply Fruchterman-Reingold style linear cooling."""

    config: LinearCoolConfig = field(default_factory=LinearCoolConfig)

    name: ClassVar[str] = "linear_cool"
    category: ClassVar[OpCategory] = OpCategory.ANNEAL
    reads: ClassVar[Tuple[str, ...]] = (
        "temperature",
        "step",
        "total_steps",
        f"extras.{_LINEAR_COOL_INITIAL_KEY}",
    )
    writes: ClassVar[Tuple[str, ...]] = ("temperature", f"extras.{_LINEAR_COOL_INITIAL_KEY}")
    requires: ClassVar[Tuple[str, ...]] = ("temperature",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Decrease temperature by a fixed linear step.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with a cooled global temperature.
        """
        del problem, ctx

        temperature = _require_temperature(state=state, op_name=self.name)
        if self.config.rate is None:
            initial_temperature = state.extras.get(_LINEAR_COOL_INITIAL_KEY)
            if not isinstance(initial_temperature, (float, int)):
                initial_temperature = temperature
                state.extras[_LINEAR_COOL_INITIAL_KEY] = float(initial_temperature)
            rate = float(initial_temperature) / float(max(state.total_steps + 1, 1))
        else:
            rate = float(self.config.rate)
        state.temperature = max(temperature - rate, 0.0)
        return state


@dataclass(frozen=True)
class InitTemperatureFromExtentConfig:
    """Configuration for :class:`InitTemperatureFromExtent`.

    Parameters
    ----------
    scale : float, default=0.1
        Multiplier applied to the larger axis extent.
    """

    scale: float = 0.1


@register_op
@dataclass(frozen=True)
class InitTemperatureFromExtent(Op):
    """Initialize FR temperature from the current position extent."""

    config: InitTemperatureFromExtentConfig = field(default_factory=InitTemperatureFromExtentConfig)

    name: ClassVar[str] = "init_temperature_from_extent"
    category: ClassVar[OpCategory] = OpCategory.ANNEAL
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ("temperature",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Set ``state.temperature`` to ``max(x_extent, y_extent) * scale``.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state containing the current positions.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with the initialized global temperature.

        Raises
        ------
        ValueError
            If ``state.pos`` is missing or ``scale`` is negative.
        """
        del problem, ctx

        if self.config.scale < 0.0:
            raise ValueError("InitTemperatureFromExtent scale must be non-negative.")
        if state.pos is None:
            raise ValueError("InitTemperatureFromExtent requires state.pos to be set.")

        if state.pos.shape[0] == 0:
            state.temperature = 0.0
            return state

        x_extent = float((state.pos[:, 0].max() - state.pos[:, 0].min()).item())
        y_extent = float((state.pos[:, 1].max() - state.pos[:, 1].min()).item())
        state.temperature = max(x_extent, y_extent) * self.config.scale
        return state


@dataclass(frozen=True)
class ExponentialCoolConfig:
    """Configuration for :class:`ExponentialCool`.

    Parameters
    ----------
    factor : float, default=0.99
        Multiplicative temperature decay factor.
    """

    factor: float = 0.99


@register_op
@dataclass(frozen=True)
class ExponentialCool(Op):
    """Apply multiplicative temperature decay."""

    config: ExponentialCoolConfig = field(default_factory=ExponentialCoolConfig)

    name: ClassVar[str] = "exponential_cool"
    category: ClassVar[OpCategory] = OpCategory.ANNEAL
    reads: ClassVar[Tuple[str, ...]] = ("temperature",)
    writes: ClassVar[Tuple[str, ...]] = ("temperature",)
    requires: ClassVar[Tuple[str, ...]] = ("temperature",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Multiply the current temperature by ``factor``.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with a cooled global temperature.
        """
        del problem, ctx

        state.temperature = (
            _require_temperature(state=state, op_name=self.name) * self.config.factor
        )
        return state


@dataclass(frozen=True)
class AdaptiveCoolConfig:
    """Configuration for :class:`AdaptiveCool`.

    Parameters
    ----------
    up_factor : float, default=1.1
        Multiplicative factor used when the force norm decreases materially.
    down_factor : float, default=0.9
        Multiplicative factor used when the force norm does not improve.
    improvement_threshold : float, default=0.95
        Force-norm ratio below which the up-factor is applied.
    """

    up_factor: float = 1.1
    down_factor: float = 0.9
    improvement_threshold: float = 0.95


@register_op
@dataclass(frozen=True)
class AdaptiveCool(Op):
    """Apply the Graphviz SFDP adaptive cooling rule."""

    config: AdaptiveCoolConfig = field(default_factory=AdaptiveCoolConfig)

    name: ClassVar[str] = "adaptive_cool"
    category: ClassVar[OpCategory] = OpCategory.ANNEAL
    reads: ClassVar[Tuple[str, ...]] = (
        "forces",
        "old_forces",
        "temperature",
        f"extras.{_ADAPTIVE_COOL_PREV_FORCE_KEY}",
    )
    writes: ClassVar[Tuple[str, ...]] = (
        "temperature",
        f"extras.{_ADAPTIVE_COOL_PREV_FORCE_KEY}",
    )
    requires: ClassVar[Tuple[str, ...]] = ("temperature",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Adapt the temperature from force-norm progress.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with an updated global temperature.
        """
        del problem, ctx

        temperature = _require_temperature(state=state, op_name=self.name)
        if state.forces is None:
            return state

        current_force_norm = float(torch.linalg.vector_norm(state.forces).item())
        previous_force_norm = state.extras.get(_ADAPTIVE_COOL_PREV_FORCE_KEY)
        if not isinstance(previous_force_norm, (float, int)):
            if state.old_forces is not None:
                previous_force_norm = float(torch.linalg.vector_norm(state.old_forces).item())
            else:
                previous_force_norm = float("inf")

        if math.isfinite(float(previous_force_norm)):
            if current_force_norm >= float(previous_force_norm):
                temperature = temperature * self.config.down_factor
            elif current_force_norm <= self.config.improvement_threshold * float(
                previous_force_norm
            ):
                temperature = temperature * self.config.up_factor

        state.temperature = temperature
        state.extras[_ADAPTIVE_COOL_PREV_FORCE_KEY] = current_force_norm
        return state


@dataclass(frozen=True)
class PerNodeTemperatureConfig:
    """Configuration for :class:`PerNodeTemperature`.

    Parameters
    ----------
    init_temp : float, default=12.0
        Initial local temperature for newly initialized nodes.
    min_temp : float, default=0.005
        Lower bound applied after each update.
    min_distance : float, default=1e-9
        Safety floor applied to impulse-norm products.
    rotation_sine_threshold : float, default=sin(pi/2 + pi/6)
        Sine threshold that grows the skew gauge.
    oscillation_cosine_threshold : float, default=cos(pi/4)
        Cosine threshold that triggers oscillation cooling.
    rotation_sensitivity : float, default=0.01
        Increment added to the skew gauge after a rotation event.
    oscillation_sensitivity : float, default=0.3
        Multiplier applied to temperature after an oscillation event.
    """

    init_temp: float = 12.0
    min_temp: float = 0.005
    min_distance: float = _MIN_DISTANCE
    rotation_sine_threshold: float = _ROTATION_SINE_THRESHOLD
    oscillation_cosine_threshold: float = _OSCILLATION_COSINE_THRESHOLD
    rotation_sensitivity: float = _ROTATION_SENSITIVITY
    oscillation_sensitivity: float = _OSCILLATION_SENSITIVITY


@register_op
@dataclass(frozen=True)
class PerNodeTemperature(Op):
    """Maintain GEM-style local temperatures in ``state.extras``."""

    config: PerNodeTemperatureConfig = field(default_factory=PerNodeTemperatureConfig)

    name: ClassVar[str] = "per_node_temperature"
    category: ClassVar[OpCategory] = OpCategory.ANNEAL
    reads: ClassVar[Tuple[str, ...]] = (
        "forces",
        "old_forces",
        "pos",
        "extras.local_temperatures",
        f"extras.{_PER_NODE_SKEW_KEY}",
        f"extras.{_PER_NODE_PREV_IMPULSE_KEY}",
    )
    writes: ClassVar[Tuple[str, ...]] = (
        "extras.local_temperatures",
        f"extras.{_PER_NODE_SKEW_KEY}",
        f"extras.{_PER_NODE_PREV_IMPULSE_KEY}",
    )
    requires: ClassVar[Tuple[str, ...]] = ()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Initialize or update per-node local temperatures.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with ``extras["local_temperatures"]`` updated.
        """
        del ctx

        device = _state_device(problem=problem, state=state)
        if problem.num_nodes == 0:
            state.extras["local_temperatures"] = torch.empty(
                (0,),
                dtype=torch.float32,
                device=device,
            )
            return state

        base_dtype = torch.float32
        if state.forces is not None:
            device = state.forces.device
            base_dtype = state.forces.dtype
        elif state.pos is not None:
            device = state.pos.device
            base_dtype = state.pos.dtype

        existing = state.extras.get("local_temperatures")
        if isinstance(existing, torch.Tensor) and existing.shape == (problem.num_nodes,):
            local_temperatures = existing.to(device=device, dtype=base_dtype)
        else:
            local_temperatures = torch.full(
                (problem.num_nodes,),
                float(self.config.init_temp),
                device=device,
                dtype=base_dtype,
            )

        skew = state.extras.get(_PER_NODE_SKEW_KEY)
        if not isinstance(skew, torch.Tensor) or skew.shape != (problem.num_nodes,):
            skew = torch.zeros((problem.num_nodes,), device=device, dtype=base_dtype)
        else:
            skew = skew.to(device=device, dtype=base_dtype)

        current_impulse = (
            state.forces.to(device=device, dtype=base_dtype)
            if state.forces is not None
            else torch.zeros((problem.num_nodes, 2), device=device, dtype=base_dtype)
        )
        previous_impulse = state.extras.get(_PER_NODE_PREV_IMPULSE_KEY)
        if (
            not isinstance(previous_impulse, torch.Tensor)
            or previous_impulse.shape != current_impulse.shape
        ):
            if state.old_forces is not None and state.old_forces.shape == current_impulse.shape:
                previous_impulse = state.old_forces.to(device=device, dtype=base_dtype)
            else:
                previous_impulse = torch.zeros_like(current_impulse)
        else:
            previous_impulse = previous_impulse.to(device=device, dtype=base_dtype)

        updated, updated_skew = _gem_update_temperatures(
            temperatures=local_temperatures,
            current_impulse=current_impulse,
            previous_impulse=previous_impulse,
            skew_gauge=skew,
            initial_temperature=float(self.config.init_temp),
            min_distance=float(self.config.min_distance),
            rotation_sine_threshold=float(self.config.rotation_sine_threshold),
            oscillation_cosine_threshold=float(self.config.oscillation_cosine_threshold),
            rotation_sensitivity=float(self.config.rotation_sensitivity),
            oscillation_sensitivity=float(self.config.oscillation_sensitivity),
        )
        updated = updated.clamp(min=float(self.config.min_temp))

        state.extras["local_temperatures"] = updated
        state.extras[_PER_NODE_SKEW_KEY] = updated_skew
        state.extras[_PER_NODE_PREV_IMPULSE_KEY] = current_impulse.detach().clone()
        return state


@dataclass(frozen=True)
class PhaseSpec:
    """One named phase in a phase-based annealing schedule.

    Parameters
    ----------
    name : str
        Human-readable phase identifier.
    iterations : int
        Number of steps spent in the phase.
    temperature : float
        Temperature assigned while the phase is active.
    attraction : float, default=1.0
        Optional phase-local attraction coefficient stored in extras.
    damping_mult : float, default=1.0
        Optional phase-local damping coefficient stored in extras.
    """

    name: str
    iterations: int
    temperature: float
    attraction: float = 1.0
    damping_mult: float = 1.0


@dataclass(frozen=True)
class PhaseScheduleConfig:
    """Configuration for :class:`PhaseSchedule`.

    Parameters
    ----------
    phases : list[PhaseSpec], default=[]
        Ordered phase definitions.
    """

    phases: list[PhaseSpec] = field(default_factory=list)


@register_op
@dataclass(frozen=True)
class PhaseSchedule(Op):
    """Select the active DRL-style phase from the current step."""

    config: PhaseScheduleConfig = field(default_factory=PhaseScheduleConfig)

    name: ClassVar[str] = "phase_schedule"
    category: ClassVar[OpCategory] = OpCategory.ANNEAL
    reads: ClassVar[Tuple[str, ...]] = ("step",)
    writes: ClassVar[Tuple[str, ...]] = ("temperature", f"extras.{_PHASE_STATE_KEY}")
    requires: ClassVar[Tuple[str, ...]] = ()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Set phase-derived temperature and extras phase metadata.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with phase metadata stored in ``state.extras``.
        """
        del problem, ctx

        if not self.config.phases:
            return state

        step_index = max(state.step, 0)
        cumulative = 0
        phase_index = len(self.config.phases) - 1
        active_phase = self.config.phases[-1]
        phase_start = cumulative
        for index, phase in enumerate(self.config.phases):
            next_cumulative = cumulative + max(int(phase.iterations), 0)
            # Clamp negative iteration counts to zero so malformed phase specs
            # still preserve the legacy "fall through to the last phase" behavior.
            if step_index < next_cumulative or index == len(self.config.phases) - 1:
                phase_index = index
                active_phase = phase
                phase_start = cumulative
                break
            cumulative = next_cumulative

        state.temperature = float(active_phase.temperature)
        state.extras[_PHASE_STATE_KEY] = {
            "index": phase_index,
            "name": active_phase.name,
            "iterations": int(active_phase.iterations),
            "temperature": float(active_phase.temperature),
            "attraction": float(active_phase.attraction),
            "damping_mult": float(active_phase.damping_mult),
            "phase_step": max(step_index - phase_start, 0),
        }
        return state


@dataclass(frozen=True)
class SmoothStepsScheduleConfig:
    """Configuration for :class:`SmoothStepsSchedule`.

    Parameters
    ----------
    keyframes : dict[str, Any], default={}
        Criterion schedule specifications keyed by criterion name.
    """

    keyframes: Dict[str, Any] = field(default_factory=dict)


@register_op
@dataclass(frozen=True)
class SmoothStepsSchedule(Op):
    """Evaluate smooth criterion-weight schedules into ``state.extras``."""

    config: SmoothStepsScheduleConfig = field(default_factory=SmoothStepsScheduleConfig)

    name: ClassVar[str] = "smooth_steps_schedule"
    category: ClassVar[OpCategory] = OpCategory.ANNEAL
    reads: ClassVar[Tuple[str, ...]] = ("step",)
    writes: ClassVar[Tuple[str, ...]] = ("extras.criterion_weights",)
    requires: ClassVar[Tuple[str, ...]] = ()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Update multicriteria weights for the current optimization step.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with ``extras["criterion_weights"]`` updated.
        """
        del problem, ctx

        weights: Dict[str, float] = {}
        for criterion, spec in self.config.keyframes.items():
            times, values = _coerce_keyframes(spec)
            weights[str(criterion)] = _evaluate_smooth_schedule(
                times=times,
                values=values,
                step=state.step,
            )
        state.extras["criterion_weights"] = weights
        return state


@register_op
class WeightAnnealing(Op):
    """Refresh ``annealing.current_weights`` from the active schedule functions.

    Notes
    -----
    The op accepts both ``annealing.schedule_fns`` and the older
    ``annealing.weight_fns`` attribute for backward compatibility.
    """

    name: ClassVar[str] = "weight_annealing"
    category: ClassVar[OpCategory] = OpCategory.ANNEAL
    reads: ClassVar[Tuple[str, ...]] = ("annealing", "step", "total_steps")
    writes: ClassVar[Tuple[str, ...]] = ("annealing",)
    requires: ClassVar[Tuple[str, ...]] = ("annealing",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Evaluate all annealed weight schedules at the current step.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with refreshed ``state.annealing.current_weights``.
        """
        del problem, ctx

        if state.annealing is None:
            return state

        state.annealing.current_weights = {
            key: float(schedule(state.step, state.total_steps))
            for key, schedule in _schedule_functions(state.annealing).items()
        }
        return state


@dataclass(frozen=True)
class InitAnnealingScheduleConfig:
    """Configuration for :class:`InitAnnealingSchedule`.

    Parameters
    ----------
    w_dag : float, default=10.0
        Base DAG ordering loss weight.
    w_attract : float, default=2.0
        Base edge-attraction loss weight.
    w_repel : float, default=0.1
        Base repulsion loss weight.
    w_overlap : float, default=5.0
        Base overlap-avoidance loss weight.
    w_cluster : float, default=1.0
        Base cluster-compactness loss weight.
    w_cluster_contain : float, default=2.0
        Base cluster-containment loss weight.
    w_crossing : float, default=1.8
        Base edge-crossing loss weight.
    w_straightness : float, default=2.2
        Base edge-straightness loss weight.
    w_length_variance : float, default=0.7
        Base edge-length variance loss weight.
    w_spacing : float, default=0.3
        Base same-layer spacing-consistency loss weight.
    w_fanout : float, default=0.3
        Base fanout-distribution loss weight.
    w_back_edge : float, default=0.3
        Base back-edge compactness loss weight.
    crossing_ramp_fraction : float, default=0.3
        Fraction of the solve horizon used to ramp crossing penalties.
    crossing_alpha_start : float, default=3.0
        Initial crossing-loss sharpness retained for state bookkeeping.
    crossing_alpha_end : float, default=10.0
        Final crossing-loss sharpness retained for state bookkeeping.
    """

    w_dag: float = 10.0
    w_attract: float = 2.0
    w_repel: float = 0.1
    w_overlap: float = 5.0
    w_cluster: float = 1.0
    w_cluster_contain: float = 2.0
    w_crossing: float = 1.8
    w_straightness: float = 2.2
    w_length_variance: float = 8.0  # Sprint 11: bumped for CV^2 formulation
    # Sprint 15: opt-in pivot-stress weight. Default 0.0 (disabled).
    w_stress: float = 0.0
    w_spacing: float = 0.3
    w_fanout: float = 0.3
    w_back_edge: float = 0.3
    crossing_ramp_fraction: float = 0.3
    crossing_alpha_start: float = 3.0
    crossing_alpha_end: float = 10.0


@register_op
@dataclass(frozen=True)
class InitAnnealingSchedule(Op):
    """Seed the native engine's per-loss annealing schedule.

    Notes
    -----
    This op mirrors the monolithic native engine's weight map, including the
    late crossing-weight ramp and the monotone boosts for repulsion, overlap,
    and straightness. Crossing alpha is stored on ``state.annealing`` for
    parity with the native bookkeeping even though the current loss op reads a
    fixed alpha from its own config.
    """

    config: InitAnnealingScheduleConfig = field(default_factory=InitAnnealingScheduleConfig)

    name: ClassVar[str] = "init_annealing_schedule"
    category: ClassVar[OpCategory] = OpCategory.ANNEAL
    reads: ClassVar[Tuple[str, ...]] = ("total_steps", "step")
    writes: ClassVar[Tuple[str, ...]] = ("annealing",)
    requires: ClassVar[Tuple[str, ...]] = ()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Initialize native-engine annealing functions on the solve state.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state carrying ``total_steps`` and ``step``.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with a populated ``annealing`` schedule and initial weights.
        """
        del problem, ctx

        def _progress(step: int, total_steps: int) -> float:
            """Return the native engine's normalized step fraction."""
            return float(step) / float(max(total_steps - 1, 1))

        def _crossing_progress(step: int, total_steps: int) -> float:
            """Return the crossing-specific warm-up fraction."""
            progress = _progress(step=step, total_steps=total_steps)
            return min(progress / max(self.config.crossing_ramp_fraction, 1.0e-8), 1.0)

        schedule = AnnealingSchedule(
            weight_fns={
                "w_dag": lambda step, total_steps: (
                    self.config.w_dag * (1.0 - 0.5 * _progress(step, total_steps))
                ),
                "w_attract": lambda step, total_steps: self.config.w_attract,
                "w_repel": lambda step, total_steps: (
                    self.config.w_repel * (1.0 + 2.0 * _progress(step, total_steps))
                ),
                "w_overlap": lambda step, total_steps: (
                    self.config.w_overlap * (1.0 + _progress(step, total_steps))
                ),
                "w_cluster": lambda step, total_steps: self.config.w_cluster,
                "w_cluster_sep": lambda step, total_steps: self.config.w_cluster * 0.5,
                "w_cluster_contain": lambda step, total_steps: self.config.w_cluster_contain,
                "w_crossing": lambda step, total_steps: (
                    self.config.w_crossing * _crossing_progress(step, total_steps)
                ),
                "w_straightness": lambda step, total_steps: (
                    self.config.w_straightness * (1.0 + 0.5 * _progress(step, total_steps))
                ),
                "w_length_variance": lambda step, total_steps: self.config.w_length_variance,
                "w_spacing": lambda step, total_steps: self.config.w_spacing,
                "w_fanout": lambda step, total_steps: self.config.w_fanout,
                "w_back_edge": lambda step, total_steps: self.config.w_back_edge,
                "w_pin": lambda step, total_steps: 1.0,
                "w_align_flex": lambda step, total_steps: 1.0,
                "w_flex_spacing": lambda step, total_steps: 1.0,
                # Sprint 15: PivotApproxStressLoss weight_key is "stress".
                # Honour the user's config.w_stress (default 0.0, i.e.
                # disabled) so an opt-in w_stress=0.1 actually binds to
                # the loss.
                "stress": lambda step, total_steps: getattr(self.config, "w_stress", 0.0),
            },
            crossing_alpha=self.config.crossing_alpha_start,
        )
        schedule.current_weights = {
            key: float(weight_fn(state.step, state.total_steps))
            for key, weight_fn in schedule.weight_fns.items()
        }
        crossing_fraction = _crossing_progress(state.step, state.total_steps)
        schedule.crossing_alpha = (
            self.config.crossing_alpha_start
            + (self.config.crossing_alpha_end - self.config.crossing_alpha_start)
            * crossing_fraction
        )
        state.annealing = schedule
        return state


@dataclass(frozen=True)
class LRDecayConfig:
    """Configuration for :class:`LRDecay`.

    Parameters
    ----------
    mode : str, default="linear"
        Decay mode. ``"linear"`` linearly interpolates between the endpoints.
        ``"exponential"`` uses geometric interpolation.
    start_lr : float or None, default=None
        Initial learning rate. When ``None``, use the optimizer's first-seen LR.
    end_lr : float or None, default=None
        Final learning rate. When ``None``, use the LinLog reference rule.
    end_lr_multiplier : float, default=0.1
        Fraction of ``start_lr`` used by the default linear endpoint rule.
    """

    mode: str = "linear"
    start_lr: Optional[float] = None
    end_lr: Optional[float] = None
    end_lr_multiplier: float = 0.1


@register_op
@dataclass(frozen=True)
class LRDecay(Op):
    """Apply a learning-rate schedule to the default optimizer."""

    config: LRDecayConfig = field(default_factory=LRDecayConfig)

    name: ClassVar[str] = "lr_decay"
    category: ClassVar[OpCategory] = OpCategory.ANNEAL
    reads: ClassVar[Tuple[str, ...]] = (
        "optimizer",
        "step",
        "total_steps",
        f"extras.{_LR_DECAY_START_KEY}",
        f"extras.{_LR_DECAY_END_KEY}",
    )
    writes: ClassVar[Tuple[str, ...]] = (
        "optimizer",
        f"extras.{_LR_DECAY_START_KEY}",
        f"extras.{_LR_DECAY_END_KEY}",
    )
    requires: ClassVar[Tuple[str, ...]] = ()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Update the optimizer learning rate for the current step.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State whose optimizer param groups have updated learning rates.
        """
        del problem, ctx

        optimizer = _require_optimizer(state=state, op_name=self.name)
        start_lr = state.extras.get(_LR_DECAY_START_KEY)
        if not isinstance(start_lr, (float, int)):
            start_lr = (
                self.config.start_lr
                if self.config.start_lr is not None
                else _optimizer_lr(optimizer)
            )
            state.extras[_LR_DECAY_START_KEY] = float(start_lr)
        start_lr = float(start_lr)

        end_lr = state.extras.get(_LR_DECAY_END_KEY)
        if not isinstance(end_lr, (float, int)):
            if self.config.end_lr is None:
                end_lr = _default_lr_decay_end_lr(
                    start_lr=start_lr,
                    total_steps=state.total_steps,
                    end_lr_multiplier=float(self.config.end_lr_multiplier),
                )
            else:
                end_lr = float(self.config.end_lr)
            state.extras[_LR_DECAY_END_KEY] = float(end_lr)
        end_lr = float(end_lr)

        fraction = min(max(float(state.step + 1) / float(max(state.total_steps, 1)), 0.0), 1.0)
        if self.config.mode == "linear":
            lr = start_lr + (end_lr - start_lr) * fraction
        elif self.config.mode == "exponential":
            if start_lr <= 0.0 or end_lr <= 0.0:
                raise ValueError(
                    "LRDecay(mode='exponential') requires positive start_lr and end_lr."
                )
            lr = start_lr * ((end_lr / start_lr) ** fraction)
        else:
            raise ValueError("LRDecay mode must be 'linear' or 'exponential'.")

        _set_optimizer_lr(optimizer=optimizer, lr=lr)
        return state


@dataclass(frozen=True)
class EarlyExaggerationConfig:
    """Configuration for :class:`EarlyExaggeration`.

    Parameters
    ----------
    multiplier : float, default=12.0
        Exaggeration factor applied before ``until_step``.
    until_step : int, default=250
        Exclusive upper step bound for early exaggeration.
    """

    multiplier: float = 12.0
    until_step: int = 250


@register_op
@dataclass(frozen=True)
class EarlyExaggeration(Op):
    """Track the active t-SNE exaggeration multiplier in ``state.extras``."""

    config: EarlyExaggerationConfig = field(default_factory=EarlyExaggerationConfig)

    name: ClassVar[str] = "early_exaggeration"
    category: ClassVar[OpCategory] = OpCategory.ANNEAL
    reads: ClassVar[Tuple[str, ...]] = ("step",)
    writes: ClassVar[Tuple[str, ...]] = ("extras.exaggeration",)
    requires: ClassVar[Tuple[str, ...]] = ()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Set ``extras['exaggeration']`` for the current step.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with the current exaggeration multiplier stored in extras.
        """
        del problem, ctx

        state.extras["exaggeration"] = (
            float(self.config.multiplier) if state.step < self.config.until_step else 1.0
        )
        return state


@dataclass(frozen=True)
class ReduceLROnPlateauConfig:
    """Configuration for :class:`ReduceLROnPlateau`.

    Parameters
    ----------
    factor : float, default=0.9
        Multiplicative LR decay factor.
    patience : int, default=20000
        Plateau patience passed to PyTorch's scheduler.
    min_lr : float, default=1e-5
        Lower LR floor passed to the scheduler.
    step_interval : int, default=10
        How often (in optimization steps) to feed the EMA loss to the scheduler.
    ema_decay : float, default=0.5**(1/100)
        Exponential moving-average decay used between scheduler steps.
    """

    factor: float = 0.9
    patience: int = 20000
    min_lr: float = 1.0e-5
    step_interval: int = 10
    ema_decay: float = _PLATEAU_EMA_DECAY


@register_op
@dataclass(frozen=True)
class ReduceLROnPlateau(Op):
    """Maintain a plateau scheduler using the SGD^2 EMA stepping pattern."""

    config: ReduceLROnPlateauConfig = field(default_factory=ReduceLROnPlateauConfig)

    name: ClassVar[str] = "reduce_lr_on_plateau"
    category: ClassVar[OpCategory] = OpCategory.ANNEAL
    reads: ClassVar[Tuple[str, ...]] = (
        "optimizer",
        "prev_loss",
        "step",
        f"extras.{_PLATEAU_SCHEDULER_KEY}",
        f"extras.{_PLATEAU_EMA_SUM_KEY}",
        f"extras.{_PLATEAU_EMA_WEIGHT_KEY}",
    )
    writes: ClassVar[Tuple[str, ...]] = (
        "optimizer",
        f"extras.{_PLATEAU_SCHEDULER_KEY}",
        f"extras.{_PLATEAU_EMA_SUM_KEY}",
        f"extras.{_PLATEAU_EMA_WEIGHT_KEY}",
    )
    requires: ClassVar[Tuple[str, ...]] = ()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Step a plateau scheduler at a fixed interval using an EMA loss.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with scheduler bookkeeping stored in ``state.extras``.
        """
        del problem, ctx

        optimizer = _require_optimizer(state=state, op_name=self.name)
        scheduler = state.extras.get(_PLATEAU_SCHEDULER_KEY)
        if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.config.factor,
                patience=self.config.patience,
                min_lr=self.config.min_lr,
            )
            state.extras[_PLATEAU_SCHEDULER_KEY] = scheduler

        loss_value = float(state.prev_loss)
        if not math.isfinite(loss_value):
            return state

        ema_weighted_sum = float(state.extras.get(_PLATEAU_EMA_SUM_KEY, 0.0))
        ema_total_weight = float(state.extras.get(_PLATEAU_EMA_WEIGHT_KEY, 0.0))
        # Keep the scheduler input smooth so noisy single-step losses do not
        # cause unnecessary LR drops.
        ema_weighted_sum = ema_weighted_sum * self.config.ema_decay + loss_value
        ema_total_weight = ema_total_weight * self.config.ema_decay + 1.0
        state.extras[_PLATEAU_EMA_SUM_KEY] = ema_weighted_sum
        state.extras[_PLATEAU_EMA_WEIGHT_KEY] = ema_total_weight

        if state.step % self.config.step_interval == 0 and ema_total_weight > 0.0:
            scheduler.step(ema_weighted_sum / ema_total_weight)
        return state


@dataclass(frozen=True)
class IdealLengthDecayConfig:
    """Configuration for :class:`IdealLengthDecay`.

    Parameters
    ----------
    decay_factor : float, default=0.75
        Multiplicative decay applied to ``extras["ideal_length"]``.
    default_initial_length : float, default=1.0
        Fallback ideal length used when no cached or measured value exists.
    min_ideal_length : float, default=1e-6
        Lower bound applied after each decay step.
    """

    decay_factor: float = 0.75
    default_initial_length: float = 1.0
    min_ideal_length: float = _DEFAULT_MIN_IDEAL_LENGTH


@register_op
@dataclass(frozen=True)
class IdealLengthDecay(Op):
    """Decay SFDP's ideal edge length stored in ``state.extras``."""

    config: IdealLengthDecayConfig = field(default_factory=IdealLengthDecayConfig)

    name: ClassVar[str] = "ideal_length_decay"
    category: ClassVar[OpCategory] = OpCategory.ANNEAL
    reads: ClassVar[Tuple[str, ...]] = ("spring_lengths", "extras.ideal_length")
    writes: ClassVar[Tuple[str, ...]] = ("extras.ideal_length",)
    requires: ClassVar[Tuple[str, ...]] = ()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Decay or initialize the ideal length scalar.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with ``extras["ideal_length"]`` updated.
        """
        del problem, ctx

        current = state.extras.get("ideal_length")
        if not isinstance(current, (float, int)):
            if state.spring_lengths is not None and state.spring_lengths.numel() > 0:
                current = float(state.spring_lengths.to(dtype=torch.float32).mean().item())
            else:
                current = float(self.config.default_initial_length)
        state.extras["ideal_length"] = max(
            float(current) * self.config.decay_factor,
            float(self.config.min_ideal_length),
        )
        return state
