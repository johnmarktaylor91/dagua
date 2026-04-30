"""Convergence and stopping operations for composable layout pipelines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple

import torch

from dagua.layout.ops.base import Op
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_PREV_POS_KEY = "converge_prev_pos"
_FR_LAST_DELTA_POS_KEY = "fr_last_delta_pos"
_STALL_LAST_LOSS_KEY = "stall_last_loss"


def _relative_window_difference(loss_window: list[float]) -> float:
    """Measure the NeuLay relative-loss window difference.

    Parameters
    ----------
    loss_window : list[float]
        Sliding loss window values.

    Returns
    -------
    float
        Relative range ``(max - min) / max``. Returns ``0`` when the window
        does not contain a positive value.
    """
    if not loss_window:
        return float("inf")
    max_loss = max(loss_window)
    if max_loss <= 0.0:
        return 0.0
    return (max_loss - min(loss_window)) / max_loss


def _current_optimizer_lr(state: SolveState) -> Optional[float]:
    """Read the current learning rate from the default optimizer.

    Parameters
    ----------
    state : SolveState
        Mutable solve state.

    Returns
    -------
    float or None
        First param-group learning rate, or ``None`` when unavailable.
    """
    if not isinstance(state.optimizer, torch.optim.Optimizer):
        return None
    if not state.optimizer.param_groups:
        return None
    lr = state.optimizer.param_groups[0].get("lr")
    if lr is None:
        return None
    return float(lr)


@dataclass(frozen=True)
class FixedStepsConfig:
    """Configuration for :class:`FixedSteps`.

    Attributes
    ----------
    n : int, default=100
        Total optimization step budget.
    """

    n: int = 100


@register_op
class FixedSteps(Op):
    """Set the total optimization budget.

    Randomness
    ----------
    This op does not use randomness.
    """

    name: ClassVar[str] = "FixedSteps"
    category: ClassVar[OpCategory] = OpCategory.CONVERGE
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = ("total_steps",)
    requires: ClassVar[Tuple[str, ...]] = ()

    def __init__(self, config: Optional[FixedStepsConfig] = None) -> None:
        """Store the frozen configuration for this op.

        Parameters
        ----------
        config : FixedStepsConfig, optional
            Step-budget configuration. When omitted, defaults are used.

        Returns
        -------
        None
            This constructor stores configuration only.
        """
        self.config = config or FixedStepsConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Set ``state.total_steps`` from the configured budget.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context. Unused by this op.

        Returns
        -------
        SolveState
            State with the updated step budget.
        """
        del problem, ctx

        if self.config.n < 0:
            raise ValueError("FixedSteps n must be non-negative.")
        state.total_steps = self.config.n
        return state


@dataclass(frozen=True)
class DisplacementThresholdConfig:
    """Configuration for :class:`DisplacementThreshold`.

    Attributes
    ----------
    threshold : float, default=1e-4
        Mean-displacement convergence threshold.
    """

    threshold: float = 1.0e-4


@register_op
class DisplacementThreshold(Op):
    """Converge when mean node displacement falls below a threshold.

    Notes
    -----
    The op persists the previous position tensor in
    ``state.extras["converge_prev_pos"]``. On the first call it only seeds the
    baseline and cannot mark convergence.
    """

    name: ClassVar[str] = "DisplacementThreshold"
    category: ClassVar[OpCategory] = OpCategory.CONVERGE
    reads: ClassVar[Tuple[str, ...]] = ("pos", f"extras.{_PREV_POS_KEY}")
    writes: ClassVar[Tuple[str, ...]] = ("converged", f"extras.{_PREV_POS_KEY}")
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def __init__(self, config: Optional[DisplacementThresholdConfig] = None) -> None:
        """Store the frozen configuration for this op.

        Parameters
        ----------
        config : DisplacementThresholdConfig, optional
            Convergence configuration. When omitted, defaults are used.

        Returns
        -------
        None
            This constructor stores configuration only.
        """
        self.config = config or DisplacementThresholdConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Update the convergence flag from inter-step displacement.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context. Unused by this op.

        Returns
        -------
        SolveState
            State with an updated convergence flag and stored baseline.
        """
        del problem, ctx

        if self.config.threshold < 0.0:
            raise ValueError("DisplacementThreshold threshold must be non-negative.")
        if state.pos is None:
            return state

        previous = state.extras.get(_PREV_POS_KEY)
        state.extras[_PREV_POS_KEY] = state.pos.detach().clone()
        previous_shape_matches = isinstance(previous, torch.Tensor) and (
            tuple(previous.shape) == tuple(state.pos.shape)
        )
        if not previous_shape_matches:
            # The first call only seeds the baseline for the next displacement check.
            return state

        displacement = (state.pos.detach() - previous.to(device=state.pos.device)).norm(dim=1)
        mean_displacement = float(displacement.mean().item())
        state.converged = state.converged or mean_displacement <= self.config.threshold
        return state


@dataclass(frozen=True)
class FRConvergenceCheckConfig:
    """Configuration for :class:`FRConvergenceCheck`.

    Attributes
    ----------
    threshold : float, default=1e-4
        Classic FR convergence threshold applied to ``||delta_pos||_F / N``.
    min_force_norm : float, default=1e-2
        Safety floor applied before normalizing node-force vectors.
    """

    threshold: float = 1.0e-4
    min_force_norm: float = 1.0e-2


@register_op
class FRConvergenceCheck(Op):
    """Converge when classic FR's Frobenius displacement rule is satisfied."""

    name: ClassVar[str] = "fr_convergence_check"
    category: ClassVar[OpCategory] = OpCategory.CONVERGE
    reads: ClassVar[Tuple[str, ...]] = (
        "forces",
        "temperature",
        f"extras.{_FR_LAST_DELTA_POS_KEY}",
    )
    writes: ClassVar[Tuple[str, ...]] = ("converged",)
    requires: ClassVar[Tuple[str, ...]] = ("forces", "temperature")

    def __init__(self, config: Optional[FRConvergenceCheckConfig] = None) -> None:
        """Store the frozen configuration for this op.

        Parameters
        ----------
        config : FRConvergenceCheckConfig, optional
            Convergence configuration. When omitted, defaults are used.

        Returns
        -------
        None
            This constructor stores configuration only.
        """
        self.config = config or FRConvergenceCheckConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Update convergence using classic FR's ``||delta_pos||_F / N`` rule.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state containing the current force buffer.
        ctx : RuntimeContext
            Execution context. Unused by this op.

        Returns
        -------
        SolveState
            State with an updated convergence flag.

        Raises
        ------
        ValueError
            If ``threshold`` is negative.
        """
        del ctx

        if self.config.threshold < 0.0:
            raise ValueError("FRConvergenceCheck threshold must be non-negative.")
        if problem.num_nodes <= 0 or state.forces is None or state.temperature is None:
            return state

        cached_delta_pos = state.extras.get(_FR_LAST_DELTA_POS_KEY)
        if isinstance(cached_delta_pos, torch.Tensor):
            delta_pos = cached_delta_pos
        else:
            length = torch.linalg.vector_norm(state.forces, dim=1).clamp(
                min=float(self.config.min_force_norm)
            )
            delta_pos = state.forces * (float(state.temperature) / length).unsqueeze(1)
        mean_displacement = float(torch.linalg.norm(delta_pos).item()) / float(problem.num_nodes)
        state.converged = state.converged or mean_displacement < self.config.threshold
        return state


@dataclass(frozen=True)
class TemperatureThresholdConfig:
    """Configuration for :class:`TemperatureThreshold`.

    Attributes
    ----------
    min_temp : float, default=0.005
        Minimum temperature before the solve is marked converged.
    """

    min_temp: float = 0.005


@register_op
class TemperatureThreshold(Op):
    """Converge once the annealing temperature reaches a minimum.

    Randomness
    ----------
    This op does not use randomness.
    """

    name: ClassVar[str] = "TemperatureThreshold"
    category: ClassVar[OpCategory] = OpCategory.CONVERGE
    reads: ClassVar[Tuple[str, ...]] = ("temperature",)
    writes: ClassVar[Tuple[str, ...]] = ("converged",)
    requires: ClassVar[Tuple[str, ...]] = ()

    def __init__(self, config: Optional[TemperatureThresholdConfig] = None) -> None:
        """Store the frozen configuration for this op.

        Parameters
        ----------
        config : TemperatureThresholdConfig, optional
            Convergence configuration. When omitted, defaults are used.

        Returns
        -------
        None
            This constructor stores configuration only.
        """
        self.config = config or TemperatureThresholdConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Update convergence from ``state.temperature``.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context. Unused by this op.

        Returns
        -------
        SolveState
            State with an updated convergence flag.
        """
        del problem, ctx

        if self.config.min_temp < 0.0:
            raise ValueError("TemperatureThreshold min_temp must be non-negative.")
        if state.temperature is None:
            return state
        state.converged = state.converged or state.temperature <= self.config.min_temp
        return state


@dataclass(frozen=True)
class SlidingWindowRelativeConfig:
    """Configuration for :class:`SlidingWindowRelative`.

    Attributes
    ----------
    window : int, default=10
        Window length required before evaluating convergence.
    tol : float, default=1e-4
        Relative tolerance coefficient multiplied by ``sqrt(N)``.
    """

    window: int = 10
    tol: float = 1.0e-4


@register_op
class SlidingWindowRelative(Op):
    """Converge when a loss window becomes sufficiently flat.

    Notes
    -----
    The op reads ``state.extras["loss_window"]`` and matches the classic
    NeuLay criterion ``relative_range < tol * sqrt(N)``.
    """

    name: ClassVar[str] = "SlidingWindowRelative"
    category: ClassVar[OpCategory] = OpCategory.CONVERGE
    reads: ClassVar[Tuple[str, ...]] = ("pos", "extras.loss_window")
    writes: ClassVar[Tuple[str, ...]] = ("converged",)
    requires: ClassVar[Tuple[str, ...]] = ("extras.loss_window",)

    def __init__(self, config: Optional[SlidingWindowRelativeConfig] = None) -> None:
        """Store the frozen configuration for this op.

        Parameters
        ----------
        config : SlidingWindowRelativeConfig, optional
            Convergence configuration. When omitted, defaults are used.

        Returns
        -------
        None
            This constructor stores configuration only.
        """
        self.config = config or SlidingWindowRelativeConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Update convergence from ``extras['loss_window']``.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context. Unused by this op.

        Returns
        -------
        SolveState
            State with an updated convergence flag.
        """
        del ctx

        if self.config.window <= 0:
            raise ValueError("SlidingWindowRelative window must be positive.")
        if self.config.tol < 0.0:
            raise ValueError("SlidingWindowRelative tol must be non-negative.")

        raw_window = state.extras.get("loss_window")
        if not isinstance(raw_window, list):
            return state
        numeric_window = [float(value) for value in raw_window[-self.config.window :]]
        if len(numeric_window) < self.config.window:
            return state

        num_nodes = int(state.pos.shape[0]) if state.pos is not None else problem.num_nodes
        criterion = _relative_window_difference(numeric_window) < (
            self.config.tol * math.sqrt(float(max(num_nodes, 1)))
        )
        state.converged = state.converged or criterion
        return state


@dataclass(frozen=True)
class StallCountConfig:
    """Configuration for :class:`StallCount`.

    Attributes
    ----------
    limit : int, default=5
        Number of consecutive stalled iterations required for convergence.
    rel_threshold : float, default=1e-4
        Relative-improvement threshold below which a step counts as stalled.
    denominator_floor : float, default=1e-12
        Safety floor used when normalizing the relative-loss change.
    """

    limit: int = 5
    rel_threshold: float = 1.0e-4
    denominator_floor: float = 1.0e-12


@register_op
class StallCount(Op):
    """Converge after repeated relative-loss stalls.

    Notes
    -----
    The current loss is read from ``state.prev_loss``. The previous observed
    loss is persisted in ``state.extras["stall_last_loss"]`` so this op can be
    used even though ``SolveState`` only carries one scalar loss slot.
    """

    name: ClassVar[str] = "StallCount"
    category: ClassVar[OpCategory] = OpCategory.CONVERGE
    reads: ClassVar[Tuple[str, ...]] = (
        "prev_loss",
        "stall_count",
        f"extras.{_STALL_LAST_LOSS_KEY}",
    )
    writes: ClassVar[Tuple[str, ...]] = (
        "stall_count",
        "converged",
        f"extras.{_STALL_LAST_LOSS_KEY}",
    )
    requires: ClassVar[Tuple[str, ...]] = ("prev_loss",)

    def __init__(self, config: Optional[StallCountConfig] = None) -> None:
        """Store the frozen configuration for this op.

        Parameters
        ----------
        config : StallCountConfig, optional
            Convergence configuration. When omitted, defaults are used.

        Returns
        -------
        None
            This constructor stores configuration only.
        """
        self.config = config or StallCountConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Update the stall counter from the latest scalar loss.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context. Unused by this op.

        Returns
        -------
        SolveState
            State with updated stall bookkeeping and convergence flag.
        """
        del problem, ctx

        if self.config.limit < 0:
            raise ValueError("StallCount limit must be non-negative.")
        if self.config.rel_threshold < 0.0:
            raise ValueError("StallCount rel_threshold must be non-negative.")

        current_loss = float(state.prev_loss)
        previous_loss = state.extras.get(_STALL_LAST_LOSS_KEY)
        state.extras[_STALL_LAST_LOSS_KEY] = current_loss
        if previous_loss is None:
            return state

        denominator = max(abs(float(previous_loss)), float(self.config.denominator_floor))
        relative_change = abs(float(previous_loss) - current_loss) / denominator
        if relative_change <= self.config.rel_threshold:
            state.stall_count += 1
        else:
            state.stall_count = 0
        state.converged = state.converged or state.stall_count >= self.config.limit
        return state


@dataclass(frozen=True)
class LRThresholdConfig:
    """Configuration for :class:`LRThreshold`.

    Attributes
    ----------
    min_lr : float, default=1e-5
        Minimum learning rate before the solve is marked converged.
    """

    min_lr: float = 1.0e-5


@register_op
class LRThreshold(Op):
    """Converge once the default optimizer learning rate falls below a floor.

    Randomness
    ----------
    This op does not use randomness.
    """

    name: ClassVar[str] = "LRThreshold"
    category: ClassVar[OpCategory] = OpCategory.CONVERGE
    reads: ClassVar[Tuple[str, ...]] = ("optimizer",)
    writes: ClassVar[Tuple[str, ...]] = ("converged",)
    requires: ClassVar[Tuple[str, ...]] = ()

    def __init__(self, config: Optional[LRThresholdConfig] = None) -> None:
        """Store the frozen configuration for this op.

        Parameters
        ----------
        config : LRThresholdConfig, optional
            Convergence configuration. When omitted, defaults are used.

        Returns
        -------
        None
            This constructor stores configuration only.
        """
        self.config = config or LRThresholdConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Update convergence from the default optimizer learning rate.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context. Unused by this op.

        Returns
        -------
        SolveState
            State with an updated convergence flag.
        """
        del problem, ctx

        if self.config.min_lr < 0.0:
            raise ValueError("LRThreshold min_lr must be non-negative.")
        lr = _current_optimizer_lr(state)
        if lr is None:
            return state
        state.converged = state.converged or lr <= self.config.min_lr
        return state
