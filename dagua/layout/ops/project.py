"""Projection operations for hard layout constraints.

These ops enforce non-differentiable constraints after a layout update:
overlap removal, hard pins, domain clamping, displacement clamping, and
monotonic acceptance safeguards for iterative solvers.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Optional, Tuple

import torch

from dagua.layout.constraints import project_hard_pins
from dagua.layout.ops.base import Op
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op
from dagua.layout.projection import project_overlaps

_MIN_MOVEMENT_NORM = 1.0e-12
_MIN_BOUNDARY_EXTENT = 1.0
_MONOTONE_OBJECTIVE_KEY = "monotone_objective"
_MONOTONE_PREV_POS_KEY = "monotone_previous_pos"
_MONOTONE_PREV_VALUE_KEY = "monotone_previous_value"


def _require_positions(state: SolveState, op_name: str) -> torch.Tensor:
    """Return the position tensor required by a projection op.

    Parameters
    ----------
    state : SolveState
        Mutable solve state.
    op_name : str
        Name of the operation requesting positions.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If ``state.pos`` is not available.
    """
    if state.pos is None:
        raise ValueError(f"{op_name} requires state.pos to be set.")
    return state.pos


def _require_forces(state: SolveState, op_name: str) -> torch.Tensor:
    """Return the force tensor required by a projection op.

    Parameters
    ----------
    state : SolveState
        Mutable solve state.
    op_name : str
        Name of the operation requesting forces.

    Returns
    -------
    torch.Tensor
        Force tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If ``state.forces`` is not available.
    """
    if state.forces is None:
        raise ValueError(f"{op_name} requires state.forces to be set.")
    return state.forces


def _auto_boundary_extent(problem: LayoutProblem) -> float:
    """Estimate a stable clamping extent from problem size and node sizes.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs.

    Returns
    -------
    float
        Bounding-box half-width used by ``BoundaryClamp`` when no explicit
        extent is provided.
    """
    if problem.node_sizes is None or problem.node_sizes.numel() == 0:
        return max(float(max(problem.num_nodes, 1)) ** 0.5 * 5.0, _MIN_BOUNDARY_EXTENT)

    max_size = float(problem.node_sizes.to(dtype=torch.float32, device="cpu").max().item())
    return max(
        max_size * max(float(max(problem.num_nodes, 1)) ** 0.5, 1.0) * 2.0,
        _MIN_BOUNDARY_EXTENT,
    )


def _resolve_monotone_objective_value(
    objective: Callable[..., Any],
    problem: LayoutProblem,
    state: SolveState,
    ctx: RuntimeContext,
    pos: torch.Tensor,
) -> float:
    """Evaluate a monotone-safeguard objective on a candidate position tensor.

    Parameters
    ----------
    objective : callable
        Objective callable stored in ``state.extras``.
    problem : LayoutProblem
        Immutable layout inputs.
    state : SolveState
        Mutable solve state.
    ctx : RuntimeContext
        Execution infrastructure.
    pos : torch.Tensor
        Candidate positions with shape ``[N, 2]``.

    Returns
    -------
    float
        Scalar objective value.

    Raises
    ------
    TypeError
        If the objective signature is unsupported or its result is not scalar.
    """
    signature = inspect.signature(objective)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    has_varargs = any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    )

    result: Any
    if has_varargs or len(positional) >= 4:
        result = objective(problem, state, ctx, pos)
    elif len(positional) == 3:
        original_pos = state.pos
        try:
            state.pos = pos
            result = objective(problem, state, ctx)
        finally:
            state.pos = original_pos
    elif len(positional) == 1:
        result = objective(pos)
    else:
        raise TypeError(
            "MonotoneSafeguard objective must accept one positional argument "
            "(pos), three positional arguments (problem, state, ctx), "
            "or four positional arguments (problem, state, ctx, pos)."
        )

    if isinstance(result, torch.Tensor):
        if result.numel() != 1:
            raise TypeError("MonotoneSafeguard objective tensor must be scalar.")
        return float(result.detach().item())
    if isinstance(result, (float, int)):
        return float(result)
    raise TypeError("MonotoneSafeguard objective must return a scalar float or tensor.")


@dataclass(frozen=True)
class OverlapProjectionConfig:
    """Configuration for :class:`OverlapProjection`.

    Parameters
    ----------
    padding : float, default=2.0
        Extra spacing enforced between node bounding boxes.
    iterations : int, default=10
        Maximum number of projection passes.
    """

    padding: float = 2.0
    iterations: int = 10


@register_op
@dataclass(frozen=True)
class OverlapProjection(Op):
    """Resolve node overlaps using the native overlap projector."""

    config: OverlapProjectionConfig = field(default_factory=OverlapProjectionConfig)

    name: ClassVar[str] = "overlap_projection"
    category: ClassVar[OpCategory] = OpCategory.PROJECT
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Project overlapping bounding boxes apart in-place.

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
            State with overlap-projected positions.
        """
        del ctx

        positions = _require_positions(state=state, op_name=self.name)
        if problem.node_sizes is None or problem.node_sizes.numel() == 0:
            return state

        node_sizes = problem.node_sizes.to(device=positions.device, dtype=positions.dtype)
        project_overlaps(
            pos=positions,
            node_sizes=node_sizes,
            padding=self.config.padding,
            iterations=self.config.iterations,
            layer_index=state.layer_index,
        )
        state.pos = positions
        return state


@register_op
class HardPinProjection(Op):
    """Project hard-pinned node coordinates onto their target axes."""

    name: ClassVar[str] = "hard_pin_projection"
    category: ClassVar[OpCategory] = OpCategory.PROJECT
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Apply hard pin targets from ``problem.flex``.

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
            State with pinned coordinates projected onto their targets.
        """
        del ctx

        positions = _require_positions(state=state, op_name=self.name)
        flex = problem.flex
        if (
            flex is None
            or flex.pin_indices is None
            or flex.pin_targets is None
            or flex.hard_pin_mask is None
            or flex.pin_indices.numel() == 0
        ):
            return state

        pin_indices = flex.pin_indices.to(device=positions.device, dtype=torch.long)
        pin_targets = flex.pin_targets.to(device=positions.device, dtype=positions.dtype)
        hard_mask = flex.hard_pin_mask.to(device=positions.device, dtype=torch.bool)
        project_hard_pins(
            pos=positions,
            pin_indices=pin_indices,
            pin_targets=pin_targets,
            pin_mask=hard_mask,
        )
        state.pos = positions
        return state


@dataclass(frozen=True)
class BoundaryClampConfig:
    """Configuration for :class:`BoundaryClamp`.

    Parameters
    ----------
    extent : float or None, default=None
        Bounding-box half-width. When ``None``, derive an extent from the
        problem size and node sizes.
    """

    extent: Optional[float] = None


@register_op
@dataclass(frozen=True)
class BoundaryClamp(Op):
    """Clamp node positions into a symmetric axis-aligned bounding box."""

    config: BoundaryClampConfig = field(default_factory=BoundaryClampConfig)

    name: ClassVar[str] = "boundary_clamp"
    category: ClassVar[OpCategory] = OpCategory.PROJECT
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Clamp all coordinates into ``[-extent, extent]``.

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
            State with clamped positions.
        """
        del ctx

        positions = _require_positions(state=state, op_name=self.name)
        extent = (
            self.config.extent if self.config.extent is not None else _auto_boundary_extent(problem)
        )
        state.pos = positions.clamp(min=-float(extent), max=float(extent))
        return state


@dataclass(frozen=True)
class MovementClampConfig:
    """Configuration for :class:`MovementClamp`.

    Parameters
    ----------
    mode : str, default="temperature"
        Displacement budget mode. ``"temperature"`` uses ``state.temperature``.
        ``"fixed"`` and ``"explicit"`` use ``max_delta``.
    max_delta : float or None, default=None
        Explicit movement cap used by ``"fixed"`` and ``"explicit"`` modes.
    """

    mode: str = "temperature"
    max_delta: Optional[float] = None


@register_op
@dataclass(frozen=True)
class MovementClamp(Op):
    """Cap per-node force vectors to a maximum movement magnitude."""

    config: MovementClampConfig = field(default_factory=MovementClampConfig)

    name: ClassVar[str] = "movement_clamp"
    category: ClassVar[OpCategory] = OpCategory.PROJECT
    reads: ClassVar[Tuple[str, ...]] = ("forces", "temperature")
    writes: ClassVar[Tuple[str, ...]] = ("forces",)
    requires: ClassVar[Tuple[str, ...]] = ("forces",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Clamp each force vector without changing its direction.

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
            State with clamped force vectors.

        Raises
        ------
        ValueError
            If the selected mode cannot resolve a positive displacement cap.
        """
        del problem, ctx

        forces = _require_forces(state=state, op_name=self.name)
        if forces.numel() == 0:
            return state

        if self.config.mode == "temperature":
            if state.temperature is None:
                raise ValueError("MovementClamp(mode='temperature') requires state.temperature.")
            limit = float(state.temperature)
        elif self.config.mode in {"fixed", "explicit"}:
            if self.config.max_delta is None:
                raise ValueError("MovementClamp fixed/explicit mode requires config.max_delta.")
            limit = float(self.config.max_delta)
        else:
            raise ValueError("MovementClamp mode must be 'temperature', 'fixed', or 'explicit'.")

        if limit < 0.0:
            raise ValueError("MovementClamp requires a non-negative displacement limit.")

        lengths = torch.linalg.vector_norm(forces, dim=1)
        safe_lengths = lengths.clamp(min=_MIN_MOVEMENT_NORM)
        scale = torch.minimum(
            torch.ones_like(lengths),
            torch.full_like(lengths, limit) / safe_lengths,
        )
        state.forces = forces * scale.unsqueeze(1)
        return state


@dataclass(frozen=True)
class MonotoneSafeguardConfig:
    """Configuration for :class:`MonotoneSafeguard`.

    Parameters
    ----------
    max_bisections : int, default=8
        Maximum number of conservative half-way blends attempted before
        reverting to the previous accepted position.
    tolerance : float, default=1e-8
        Numerical slack allowed when comparing objective values.
    """

    max_bisections: int = 8
    tolerance: float = 1.0e-8


@register_op
@dataclass(frozen=True)
class MonotoneSafeguard(Op):
    """Prevent objective regressions by conservative bisection backtracking.

    Notes
    -----
    This op uses ``state.extras`` for its auxiliary state:

    - ``"monotone_objective"``: callable objective evaluator.
    - ``"monotone_previous_pos"``: last accepted positions.
    - ``"monotone_previous_value"``: last accepted scalar objective.
    """

    config: MonotoneSafeguardConfig = field(default_factory=MonotoneSafeguardConfig)

    name: ClassVar[str] = "monotone_safeguard"
    category: ClassVar[OpCategory] = OpCategory.PROJECT
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Accept, bisect, or revert a candidate update based on an objective.

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
            State with either the candidate, a bisected blend, or the previous
            accepted positions.
        """
        positions = _require_positions(state=state, op_name=self.name)
        objective = state.extras.get(_MONOTONE_OBJECTIVE_KEY)
        if not callable(objective):
            state.extras[_MONOTONE_PREV_POS_KEY] = positions.detach().clone()
            return state

        previous_pos = state.extras.get(_MONOTONE_PREV_POS_KEY)
        previous_value = state.extras.get(_MONOTONE_PREV_VALUE_KEY)
        candidate_value = _resolve_monotone_objective_value(
            objective=objective,
            problem=problem,
            state=state,
            ctx=ctx,
            pos=positions,
        )

        if not isinstance(previous_pos, torch.Tensor):
            state.extras[_MONOTONE_PREV_POS_KEY] = positions.detach().clone()
            state.extras[_MONOTONE_PREV_VALUE_KEY] = candidate_value
            return state

        reference_value = (
            float(previous_value)
            if isinstance(previous_value, (float, int)) and math.isfinite(float(previous_value))
            else _resolve_monotone_objective_value(
                objective=objective,
                problem=problem,
                state=state,
                ctx=ctx,
                pos=previous_pos.to(device=positions.device, dtype=positions.dtype),
            )
        )

        if candidate_value <= reference_value + self.config.tolerance:
            state.extras[_MONOTONE_PREV_POS_KEY] = positions.detach().clone()
            state.extras[_MONOTONE_PREV_VALUE_KEY] = candidate_value
            return state

        accepted = previous_pos.to(device=positions.device, dtype=positions.dtype)
        accepted_value = reference_value
        blended = positions
        for _ in range(self.config.max_bisections):
            # Bisect toward the last accepted state rather than line-searching
            # from scratch so the safeguard stays deterministic and cheap.
            blended = 0.5 * (blended + accepted)
            blended_value = _resolve_monotone_objective_value(
                objective=objective,
                problem=problem,
                state=state,
                ctx=ctx,
                pos=blended,
            )
            if blended_value <= reference_value + self.config.tolerance:
                accepted = blended.detach().clone()
                accepted_value = blended_value
                break
        else:
            accepted = accepted.detach().clone()

        state.pos = accepted
        state.extras[_MONOTONE_PREV_POS_KEY] = accepted.detach().clone()
        state.extras[_MONOTONE_PREV_VALUE_KEY] = accepted_value
        return state
