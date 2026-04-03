"""Annealing, cooling, and schedule operations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, Mapping, Optional, Sequence, Tuple

import torch

from dagua.layout.classic.gem import _update_temperatures as _gem_update_temperatures
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


def _default_lr_decay_end_lr(start_lr: float, total_steps: int) -> float:
    """Compute the LinLog-style default final learning rate.

    Parameters
    ----------
    start_lr : float
        Initial learning rate.
    total_steps : int
        Planned optimization horizon.

    Returns
    -------
    float
        Default final learning rate.
    """
    return max(start_lr * 0.1, start_lr / math.sqrt(float(max(total_steps, 1))))


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
    reads: ClassVar[Tuple[str, ...]] = ("temperature", "step", "total_steps")
    writes: ClassVar[Tuple[str, ...]] = ("temperature",)
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
    """

    up_factor: float = 1.1
    down_factor: float = 0.9


@register_op
@dataclass(frozen=True)
class AdaptiveCool(Op):
    """Apply the Graphviz SFDP adaptive cooling rule."""

    config: AdaptiveCoolConfig = field(default_factory=AdaptiveCoolConfig)

    name: ClassVar[str] = "adaptive_cool"
    category: ClassVar[OpCategory] = OpCategory.ANNEAL
    reads: ClassVar[Tuple[str, ...]] = ("forces", "old_forces", "temperature")
    writes: ClassVar[Tuple[str, ...]] = ("temperature",)
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
            elif current_force_norm <= 0.95 * float(previous_force_norm):
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
    """

    init_temp: float = 12.0
    min_temp: float = 0.005


@register_op
@dataclass(frozen=True)
class PerNodeTemperature(Op):
    """Maintain GEM-style local temperatures in ``state.extras``."""

    config: PerNodeTemperatureConfig = field(default_factory=PerNodeTemperatureConfig)

    name: ClassVar[str] = "per_node_temperature"
    category: ClassVar[OpCategory] = OpCategory.ANNEAL
    reads: ClassVar[Tuple[str, ...]] = ("forces", "old_forces")
    writes: ClassVar[Tuple[str, ...]] = ()
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
    writes: ClassVar[Tuple[str, ...]] = ("temperature",)
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
    writes: ClassVar[Tuple[str, ...]] = ()
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
    """Refresh ``annealing.current_weights`` from the active schedule functions."""

    name = "weight_annealing"
    category = OpCategory.ANNEAL
    reads = ("annealing", "step", "total_steps")
    writes = ()
    requires = ("annealing",)

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
    """

    mode: str = "linear"
    start_lr: Optional[float] = None
    end_lr: Optional[float] = None


@register_op
@dataclass(frozen=True)
class LRDecay(Op):
    """Apply a learning-rate schedule to the default optimizer."""

    config: LRDecayConfig = field(default_factory=LRDecayConfig)

    name: ClassVar[str] = "lr_decay"
    category: ClassVar[OpCategory] = OpCategory.ANNEAL
    reads: ClassVar[Tuple[str, ...]] = ("step", "total_steps")
    writes: ClassVar[Tuple[str, ...]] = ()
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
                end_lr = _default_lr_decay_end_lr(start_lr=start_lr, total_steps=state.total_steps)
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
    writes: ClassVar[Tuple[str, ...]] = ()
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
    """

    factor: float = 0.9
    patience: int = 20000
    min_lr: float = 1.0e-5


@register_op
@dataclass(frozen=True)
class ReduceLROnPlateau(Op):
    """Maintain a plateau scheduler using the SGD^2 EMA stepping pattern."""

    config: ReduceLROnPlateauConfig = field(default_factory=ReduceLROnPlateauConfig)

    name: ClassVar[str] = "reduce_lr_on_plateau"
    category: ClassVar[OpCategory] = OpCategory.ANNEAL
    reads: ClassVar[Tuple[str, ...]] = ("prev_loss", "step")
    writes: ClassVar[Tuple[str, ...]] = ()
    requires: ClassVar[Tuple[str, ...]] = ()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Step a plateau scheduler every ten iterations using an EMA loss.

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
        ema_weighted_sum = ema_weighted_sum * _PLATEAU_EMA_DECAY + loss_value
        ema_total_weight = ema_total_weight * _PLATEAU_EMA_DECAY + 1.0
        state.extras[_PLATEAU_EMA_SUM_KEY] = ema_weighted_sum
        state.extras[_PLATEAU_EMA_WEIGHT_KEY] = ema_total_weight

        if state.step % 10 == 0 and ema_total_weight > 0.0:
            scheduler.step(ema_weighted_sum / ema_total_weight)
        return state


@dataclass(frozen=True)
class IdealLengthDecayConfig:
    """Configuration for :class:`IdealLengthDecay`.

    Parameters
    ----------
    decay_factor : float, default=0.75
        Multiplicative decay applied to ``extras["ideal_length"]``.
    """

    decay_factor: float = 0.75


@register_op
@dataclass(frozen=True)
class IdealLengthDecay(Op):
    """Decay SFDP's ideal edge length stored in ``state.extras``."""

    config: IdealLengthDecayConfig = field(default_factory=IdealLengthDecayConfig)

    name: ClassVar[str] = "ideal_length_decay"
    category: ClassVar[OpCategory] = OpCategory.ANNEAL
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = ()
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
                current = 1.0
        state.extras["ideal_length"] = max(
            float(current) * self.config.decay_factor,
            _DEFAULT_MIN_IDEAL_LENGTH,
        )
        return state
