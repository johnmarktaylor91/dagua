"""Projection operations for hard layout constraints.

These ops enforce non-differentiable constraints after a layout update:
overlap removal, hard pins, domain clamping, displacement clamping, and
monotonic acceptance safeguards for iterative solvers.
"""

from __future__ import annotations

import inspect
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Optional, Tuple

import torch

from dagua.layout.constraints import project_hard_pins
from dagua.layout.ops.base import Op
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op
from dagua.layout.projection import project_overlaps
from dagua.metrics import (
    composite_undirected,
    edge_crossing_score,
    edge_length_deviation_score,
    isotonic_stress,
    node_occlusion_score,
)

_MIN_MOVEMENT_NORM = 1.0e-12
_MIN_BOUNDARY_EXTENT = 1.0
_MONOTONE_OBJECTIVE_KEY = "monotone_objective"
_MONOTONE_PREV_POS_KEY = "monotone_previous_pos"
_MONOTONE_PREV_VALUE_KEY = "monotone_previous_value"
# Fixed seed for the gate's proxy metrics -- matches the project convention
# (see dagua.metrics.full()) of pinning sampled_crossing_rate's seed so
# repeat scoring of the same positions is deterministic.
_GATE_METRIC_SEED = 0
_GATE_CROSSING_SAMPLES = 200_000

log = logging.getLogger(__name__)


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


def _visible_original_positions(
    problem: LayoutProblem,
    state: SolveState,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, Optional[object]]:
    """Return the original-node view for overlap projection.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable original graph inputs.
    state : SolveState
        Mutable solve state that may be dummy-expanded.
    positions : torch.Tensor
        Active positions with shape ``[N_active, 2]``.

    Returns
    -------
    tuple[torch.Tensor, object | None]
        Original-node position view and matching layer index.
    """
    expanded_graph = state.extras.get("expanded_graph")
    if expanded_graph is None or int(getattr(expanded_graph, "num_nodes", -1)) != int(
        positions.shape[0]
    ):
        return positions, state.layer_index
    return positions[: problem.num_nodes], state.extras.get(
        "original_layer_index",
        state.layer_index,
    )


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

        visible_positions, layer_index = _visible_original_positions(
            problem=problem,
            state=state,
            positions=positions,
        )
        node_sizes = problem.node_sizes.to(
            device=visible_positions.device,
            dtype=visible_positions.dtype,
        )
        project_overlaps(
            pos=visible_positions,
            node_sizes=node_sizes,
            padding=self.config.padding,
            iterations=self.config.iterations,
            layer_index=layer_index,
        )
        state.pos = positions
        return state


def _gate_is_semantically_directed(problem: LayoutProblem) -> bool:
    """Conservative semantic-direction lookup for gate scoring.

    Mirrors the ``direct_hint`` / ``structure_hint`` lookup chain used by
    ``dagua.layout.ops.sugiyama._directed_igraph_fidelity_gate`` so the gate
    agrees with the rest of the pipeline on whether a graph's edges carry
    directional meaning. Defaults to directed (``True``) when no hint is
    available, same conservative default as ``dagua.metrics.composite_auto``.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs, optionally carrying a direct
        ``is_semantically_directed`` hint or one nested in ``problem.structure``.

    Returns
    -------
    bool
        Whether edges should be scored as semantically directed.
    """
    direct_hint = getattr(problem, "is_semantically_directed", None)
    if direct_hint is not None:
        return bool(direct_hint)
    structure = getattr(problem, "structure", None)
    structure_hint = getattr(structure, "is_semantically_directed", None)
    if structure_hint is not None:
        return bool(structure_hint)
    return True


def _overlap_gate_proxy_composite(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    direction: str,
    is_semantically_directed: bool,
) -> float:
    """Cheap proxy composite for :class:`OverlapProjectionGated`.

    Reuses the real ``composite_auto`` weighting from ``dagua.metrics`` on a
    partial metric dict -- overlap count, sampled crossing rate (pinned to
    the project's fixed metric seed), and edge-length CV -- so the formulas
    are never reimplemented. On directed graphs, ``dag_consistency`` is
    included too (also O(|E|), no sampling -- still cheap): overlap
    projection can resolve overlaps by pushing a node against the graph's
    top-to-bottom (or configured) direction, which the other three terms
    cannot see at all but which a directed composite penalizes heavily.
    Without this, the gate accepted projections that wrecked DAG
    consistency on directed test graphs going through the native_stress
    pipeline (caught by
    ``tests/test_layout/test_quality_knob.py::test_quality_high_smoke_spends_more_and_scores_near_draft``).
    The terms this proxy still omits (depth correlation, straightness,
    angular resolution, cluster separation) fall back to ``composite``'s /
    ``composite_undirected``'s own neutral or worst-case defaults on both
    the before and after evaluation, so they contribute an identical
    constant offset that cancels out of the accept/reject delta.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor shaped ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor shaped ``[2, E]``.
    node_sizes : torch.Tensor
        Node-size tensor shaped ``[N, 2]``.
    direction : str
        Layout direction retained for API compatibility.
    is_semantically_directed : bool
        Semantic direction retained for API compatibility. Projection quality
        uses common geometry because this local gate has no declared rank metadata.

    Returns
    -------
    float
        Proxy composite score where higher is better.
    """
    pos_cpu = pos.detach().cpu()
    edge_index_cpu = edge_index.detach().cpu()
    node_sizes_cpu = node_sizes.detach().cpu()

    del direction, is_semantically_directed
    proxy_metrics = {}
    proxy_metrics.update(node_occlusion_score(pos_cpu, node_sizes_cpu, seed=_GATE_METRIC_SEED))
    proxy_metrics.update(
        edge_crossing_score(
            pos_cpu,
            edge_index_cpu,
            n_samples=_GATE_CROSSING_SAMPLES,
            seed=_GATE_METRIC_SEED,
        )
    )
    proxy_metrics.update(edge_length_deviation_score(pos_cpu, edge_index_cpu))
    proxy_metrics.update(
        isotonic_stress(
            pos_cpu,
            edge_index_cpu,
            n_sources=min(32, int(pos_cpu.shape[0])),
            n_targets=min(64, int(pos_cpu.shape[0])),
        )
    )
    return composite_undirected(proxy_metrics)


@dataclass(frozen=True)
class OverlapProjectionGatedConfig:
    """Configuration for :class:`OverlapProjectionGated`.

    Parameters
    ----------
    padding : float, default=2.0
        Extra spacing enforced between node bounding boxes.
    iterations : int, default=10
        Maximum number of projection passes.
    convergent : bool, default=True
        Use the convergent exact-path projector (this op was designed to
        gate that projector; see ``dagua.layout.projection.project_overlaps``).
    """

    padding: float = 2.0
    iterations: int = 10
    convergent: bool = True


@register_op
@dataclass(frozen=True)
class OverlapProjectionGated(Op):
    """Overlap projection with metric-gated acceptance.

    Projection is a non-differentiable, potentially destructive constraint:
    on dense overlap cliques the projector can trade a full overlap cleanup
    for materially worse crossings or edge-length uniformity (see
    ``.project-context/research/r79_native/P3B2_STRESS_FORENSICS.md``, fix
    item 5). This op computes a cheap proxy composite before and after
    projection using the real overlap/crossing/edge-length-CV metric
    formulas and keeps the projected result only if the proxy does not
    regress; otherwise it reverts to the pre-projection positions and logs a
    debug line.

    NOT wired into any default pipeline as of r80-S2b: the r80-S2 sweep +
    call-site attribution (P7_PROJECTOR_EVIDENCE.md) showed the composite
    damage from projector-trajectory changes happens in the MANY ungated
    ``periodic_overlap_projection`` calls during optimization, upstream of
    any single final-stage gate, so gating only the final projection cannot
    protect the default path. Registered and tested for future use.
    """

    config: OverlapProjectionGatedConfig = field(default_factory=OverlapProjectionGatedConfig)

    name: ClassVar[str] = "overlap_projection_gated"
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
        """Project overlaps, keeping the result only if the proxy improves or ties.

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
            State with either the projected positions (proxy accepted) or
            the original pre-projection positions (proxy rejected).
        """
        del ctx

        positions = _require_positions(state=state, op_name=self.name)
        if problem.node_sizes is None or problem.node_sizes.numel() == 0:
            return state

        visible_positions, layer_index = _visible_original_positions(
            problem=problem,
            state=state,
            positions=positions,
        )
        node_sizes = problem.node_sizes.to(
            device=visible_positions.device,
            dtype=visible_positions.dtype,
        )
        edge_index = problem.edge_index.to(device=visible_positions.device)
        direction = getattr(problem, "direction", "TB")
        is_directed = _gate_is_semantically_directed(problem)

        before = visible_positions.detach().clone()
        proxy_before = _overlap_gate_proxy_composite(
            before, edge_index, node_sizes, direction, is_directed
        )

        candidate = visible_positions.detach().clone()
        project_overlaps(
            pos=candidate,
            node_sizes=node_sizes,
            padding=self.config.padding,
            iterations=self.config.iterations,
            layer_index=layer_index,
            convergent=self.config.convergent,
        )
        proxy_after = _overlap_gate_proxy_composite(
            candidate, edge_index, node_sizes, direction, is_directed
        )

        if proxy_after >= proxy_before:
            visible_positions.data.copy_(candidate)
        else:
            log.debug(
                "%s rejected projection: proxy_before=%.4f proxy_after=%.4f "
                "(reverting to pre-projection positions)",
                self.name,
                proxy_before,
                proxy_after,
            )

        state.pos = positions
        return state


@dataclass(frozen=True)
class PeriodicOverlapProjectionConfig:
    """Configuration for :class:`PeriodicOverlapProjection`.

    Parameters
    ----------
    interval : int, default=5
        Run the overlap projector every ``interval`` optimization steps.
    padding : float, default=2.0
        Extra spacing enforced between node bounding boxes.
    iterations : int or None, default=None
        Projection iterations per trigger. When ``None``, derive the native
        engine's scale-aware default from ``problem.num_nodes``.
    run_on_last_step : bool, default=True
        Whether to force one projection on the final optimization step even
        when it does not align with ``interval``.
    """

    interval: int = 5
    padding: float = 2.0
    iterations: Optional[int] = None
    run_on_last_step: bool = True


@register_op
@dataclass(frozen=True)
class PeriodicOverlapProjection(Op):
    """Run overlap projection periodically during native optimization."""

    config: PeriodicOverlapProjectionConfig = field(default_factory=PeriodicOverlapProjectionConfig)

    name: ClassVar[str] = "periodic_overlap_projection"
    category: ClassVar[OpCategory] = OpCategory.PROJECT
    reads: ClassVar[Tuple[str, ...]] = ("pos", "step", "total_steps", "layer_index")
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Project overlaps on the configured cadence.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state carrying positions and the current step.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with positions projected when the cadence fires.
        """
        del ctx

        positions = _require_positions(state=state, op_name=self.name)
        if problem.node_sizes is None or problem.node_sizes.numel() == 0:
            return state
        if self.config.interval <= 0:
            raise ValueError("PeriodicOverlapProjection interval must be positive.")
        if state.step <= 0:
            return state

        is_interval_step = state.step % self.config.interval == 0
        is_last_step = state.step >= max(state.total_steps - 1, 0)
        if not is_interval_step and not (self.config.run_on_last_step and is_last_step):
            return state

        if self.config.iterations is None:
            if problem.num_nodes <= 500_000:
                iterations = 5
            elif problem.num_nodes <= 5_000_000:
                iterations = 3
            else:
                iterations = 2
        else:
            iterations = self.config.iterations
        if iterations <= 0:
            return state

        visible_positions, layer_index = _visible_original_positions(
            problem=problem,
            state=state,
            positions=positions,
        )
        node_sizes = problem.node_sizes.to(
            device=visible_positions.device,
            dtype=visible_positions.dtype,
        )
        project_overlaps(
            pos=visible_positions,
            node_sizes=node_sizes,
            padding=self.config.padding,
            iterations=iterations,
            layer_index=layer_index,
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
