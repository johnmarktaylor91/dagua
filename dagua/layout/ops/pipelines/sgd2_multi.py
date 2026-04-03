"""(SGD)^2 multicriteria layout expressed as a composable ops pipeline.

Produces BIT-IDENTICAL output to the classic ``layout_sgd2_multi`` for all
criterion combinations including stress, ideal edge length, neighborhood
preservation, crossings (neural detector), crossing angle maximization,
aspect ratio, angular resolution, and vertex resolution.  All numeric work
is delegated to the classic module's helper functions -- the ops only
orchestrate the call sequence.
"""

from __future__ import annotations

import math
from typing import ClassVar, Dict, Optional, Tuple

import torch
import torch.nn as nn

from dagua.layout.classic.sgd2_multi import (
    _CROSSING_DETECTOR_LR,
    SmoothSteps,
    _criterion_loss,
    _CrossingDetector,
    _CrossingLossState,
    _CyclicSampler,
    _layout_device,
    _prepare_state,
    _resolve_schedules,
    _set_seed,
    _validate_inputs,
    _VertexResolutionState,
)
from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory

# ---------------------------------------------------------------------------
# Pipeline-local ops
# ---------------------------------------------------------------------------


class _InitSGD2MultiState(Op):
    """Validate inputs, seed RNGs, and precompute graph structures.

    Handles the trivial cases (0 or 1 node) by setting ``state.converged``
    so downstream ops skip the optimization loop.
    """

    name: ClassVar[str] = "sgd2_multi_init_state"
    category: ClassVar[OpCategory] = OpCategory.INIT
    writes: ClassVar[Tuple[str, ...]] = ("pos", "extras")

    def __init__(
        self,
        steps: int = 10_000,
        lr: float = 1.0,
        momentum: float = 0.7,
        grad_clamp: float = 4.0,
        batch_size: int = 16,
        criteria: Optional[Dict[str, float]] = None,
        criteria_schedules: Optional[Dict[str, SmoothSteps]] = None,
    ) -> None:
        """Store optimization parameters.

        Parameters
        ----------
        steps : int
            Maximum number of SGD iterations.
        lr : float
            SGD learning rate.
        momentum : float
            SGD momentum.
        grad_clamp : float
            Symmetric gradient clamp.
        batch_size : int
            Mini-batch size.
        criteria : dict[str, float] | None
            Static criterion weights.
        criteria_schedules : dict[str, SmoothSteps] | None
            Per-criterion schedules.
        """
        self.steps = steps
        self.lr = lr
        self.momentum = momentum
        self.grad_clamp = grad_clamp
        self.batch_size = batch_size
        self.criteria = criteria
        self.criteria_schedules = criteria_schedules

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Validate, seed, and precompute graph structures.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure. Unused.

        Returns
        -------
        SolveState
            State with precomputed graph data in ``extras``, or final
            positions and ``converged=True`` for trivial graphs.
        """
        del ctx

        device = _layout_device(
            edge_index=problem.edge_index,
            node_sizes=problem.node_sizes,
        )
        _validate_inputs(
            edge_index=problem.edge_index,
            num_nodes=problem.num_nodes,
            steps=self.steps,
            lr=self.lr,
            momentum=self.momentum,
            grad_clamp=self.grad_clamp,
            batch_size=self.batch_size,
        )

        state.extras["sgd2_device"] = device

        # Trivial cases
        if problem.num_nodes == 0:
            state.pos = torch.empty((0, 2), dtype=torch.float32, device=device)
            state.converged = True
            return state
        if problem.num_nodes == 1:
            state.pos = torch.zeros((1, 2), dtype=torch.float32, device=device)
            state.converged = True
            return state

        # Resolve schedules and determine which structures are needed
        schedules = _resolve_schedules(
            criteria=self.criteria,
            criteria_schedules=self.criteria_schedules,
        )
        active_names = set(schedules)
        needs_distances = bool({"stress", "vertex_resolution"} & active_names)
        needs_incident_edge_pairs = "angular_resolution" in active_names
        needs_non_incident_edge_pairs = bool(
            {"crossings", "crossing_angle_maximization"} & active_names
        )

        _set_seed(problem.seed)
        prepared = _prepare_state(
            edge_index=problem.edge_index,
            num_nodes=problem.num_nodes,
            device=device,
            needs_distances=needs_distances,
            needs_incident_edge_pairs=needs_incident_edge_pairs,
            needs_non_incident_edge_pairs=needs_non_incident_edge_pairs,
            edge_weights=problem.edge_weights,
        )

        state.extras["sgd2_prepared"] = prepared
        state.extras["sgd2_schedules"] = schedules
        state.extras["sgd2_active_names"] = active_names

        return state


class _RunSGD2MultiOptimization(Op):
    """Execute the full (SGD)^2 Nesterov optimization loop.

    This op encapsulates the entire training loop including cyclic
    samplers, crossing detector training, EMA smoothing, and LR
    scheduling.  The deeply intertwined RNG state (torch.manual_seed,
    torch.randperm in cyclic samplers, torch.randint in mini-batch
    sampling, torch.randn in position init) makes it impossible to
    split the loop across separate ops without breaking bit-identity.
    """

    name: ClassVar[str] = "sgd2_multi_run_optimization"
    category: ClassVar[OpCategory] = OpCategory.OPTIMIZE
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("pos",)

    def __init__(
        self,
        steps: int = 10_000,
        lr: float = 1.0,
        momentum: float = 0.7,
        grad_clamp: float = 4.0,
        batch_size: int = 16,
    ) -> None:
        """Store optimizer hyperparameters.

        Parameters
        ----------
        steps : int
            Maximum SGD iterations.
        lr : float
            Learning rate.
        momentum : float
            SGD momentum.
        grad_clamp : float
            Symmetric gradient clamp.
        batch_size : int
            Mini-batch size.
        """
        self.steps = steps
        self.lr = lr
        self.momentum = momentum
        self.grad_clamp = grad_clamp
        self.batch_size = batch_size

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run the full (SGD)^2 optimization loop.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure. Unused.

        Returns
        -------
        SolveState
            State with optimized positions in ``pos``.
        """
        del ctx

        if state.converged:
            return state

        device = state.extras["sgd2_device"]
        prepared = state.extras["sgd2_prepared"]
        schedules = state.extras["sgd2_schedules"]
        active_names = state.extras["sgd2_active_names"]
        num_nodes = problem.num_nodes

        # Build cyclic samplers -- matches classic exactly
        samplers: Dict[str, _CyclicSampler] = {}
        for sname in schedules:
            if sname in ("stress", "vertex_resolution") and prepared.stress_pairs is not None:
                samplers[sname] = _CyclicSampler(prepared.stress_pairs.shape[1], device)
            elif sname == "ideal_edge_length" and prepared.edges.shape[1] > 0:
                samplers[sname] = _CyclicSampler(prepared.edges.shape[1], device)
            elif sname in ("neighborhood_preservation", "aspect_ratio"):
                samplers[sname] = _CyclicSampler(num_nodes, device)
            elif sname == "angular_resolution" and prepared.incident_edge_pairs is not None:
                samplers[sname] = _CyclicSampler(prepared.incident_edge_pairs.shape[1], device)
            elif (
                sname in ("crossings", "crossing_angle_maximization")
                and prepared.non_incident_edge_pairs is not None
            ):
                samplers[sname] = _CyclicSampler(prepared.non_incident_edge_pairs.shape[1], device)

        # Initialize positions -- matches classic: randn * sqrt(N)
        positions = torch.nn.Parameter(
            torch.randn((num_nodes, 2), device=device, dtype=torch.float32)
            * math.sqrt(float(num_nodes))
        )

        # Crossing detector state
        crossing_state = None
        if "crossings" in active_names:
            crossing_detector = _CrossingDetector().to(device=device)
            crossing_state = _CrossingLossState(
                detector=crossing_detector,
                optimizer=torch.optim.Adam(
                    crossing_detector.parameters(), lr=_CROSSING_DETECTOR_LR
                ),
                train_loss=nn.BCELoss(),
                position_loss=nn.BCELoss(reduction="sum"),
            )

        # Vertex resolution state
        vertex_resolution_state = None
        if "vertex_resolution" in active_names:
            vertex_resolution_state = _VertexResolutionState(
                prev_target_dist=torch.tensor(1.0, device=device, dtype=torch.float32),
                prev_weight=0.0,
            )

        # Optimizer and scheduler -- matches classic exactly
        optimizer = torch.optim.SGD([positions], lr=self.lr, momentum=self.momentum, nesterov=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.9,
            patience=20000,
            min_lr=1.0e-5,
        )

        # EMA smoothing matching reference: half-life of 100 iterations
        _ema_decay = 0.5 ** (1.0 / 100.0)
        ema_weighted_sum = 0.0
        ema_total_weight = 0.0

        for step_index in range(self.steps):
            optimizer.zero_grad(set_to_none=True)
            loss = positions.sum() * 0.0
            for name, schedule in schedules.items():
                weight = schedule(step_index)
                if weight == 0.0:
                    continue
                loss = loss + weight * _criterion_loss(
                    name=name,
                    pos=positions,
                    state=prepared,
                    batch_size=self.batch_size,
                    sampler=samplers.get(name),
                    vertex_resolution_state=vertex_resolution_state,
                    crossing_state=crossing_state,
                )

            loss.backward()
            if positions.grad is not None:
                positions.grad.clamp_(-self.grad_clamp, self.grad_clamp)
            optimizer.step()

            # EMA + scheduler step every 10 iterations (reference pattern)
            loss_value = float(loss.detach().item())
            ema_weighted_sum = ema_weighted_sum * _ema_decay + loss_value
            ema_total_weight = ema_total_weight * _ema_decay + 1.0
            if step_index % 10 == 0:
                scheduler.step(ema_weighted_sum / ema_total_weight)
            if float(optimizer.param_groups[0]["lr"]) <= 1.0e-5:
                break

        state.pos = positions.detach().to(dtype=torch.float32)
        state.converged = True
        return state


# ---------------------------------------------------------------------------
# Pipeline builder and drop-in adapter
# ---------------------------------------------------------------------------


def build_sgd2_multi_pipeline(
    steps: int = 10_000,
    criteria: Optional[Dict[str, float]] = None,
    criteria_schedules: Optional[Dict[str, SmoothSteps]] = None,
    lr: float = 1.0,
    momentum: float = 0.7,
    grad_clamp: float = 4.0,
    batch_size: int = 16,
) -> Pipeline:
    """Build an (SGD)^2 multicriteria pipeline bit-identical to classic.

    Parameters
    ----------
    steps : int, default=10000
        Maximum number of SGD iterations.
    criteria : dict[str, float] | None, default=None
        Static per-criterion weights.
    criteria_schedules : dict[str, SmoothSteps] | None, default=None
        Piecewise-smooth criterion schedules.
    lr : float, default=1.0
        SGD learning rate.
    momentum : float, default=0.7
        SGD momentum.
    grad_clamp : float, default=4.0
        Symmetric gradient clamp.
    batch_size : int, default=16
        Global mini-batch size.

    Returns
    -------
    Pipeline
        Pipeline that reproduces classic (SGD)^2's initialization,
        precomputation, optimization loop, and output conversion.

    Raises
    ------
    ValueError
        If ``steps`` is negative.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative.")

    return Pipeline(
        [
            _InitSGD2MultiState(
                steps=steps,
                lr=lr,
                momentum=momentum,
                grad_clamp=grad_clamp,
                batch_size=batch_size,
                criteria=criteria,
                criteria_schedules=criteria_schedules,
            ),
            _RunSGD2MultiOptimization(
                steps=steps,
                lr=lr,
                momentum=momentum,
                grad_clamp=grad_clamp,
                batch_size=batch_size,
            ),
        ],
        name="sgd2_multi_pipeline",
    )


def layout_sgd2_multi_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: int = 42,
    steps: int = 10_000,
    criteria: Optional[Dict[str, float]] = None,
    criteria_schedules: Optional[Dict[str, SmoothSteps]] = None,
    lr: float = 1.0,
    momentum: float = 0.7,
    grad_clamp: float = 4.0,
    batch_size: int = 16,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the (SGD)^2 multicriteria pipeline as a drop-in replacement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor | None, default=None
        Unused placeholder kept for API compatibility.
    seed : int, default=42
        Random seed.
    steps : int, default=10000
        Maximum SGD iterations.
    criteria : dict[str, float] | None, default=None
        Static per-criterion weights.
    criteria_schedules : dict[str, SmoothSteps] | None, default=None
        Piecewise-smooth criterion schedules.
    lr : float, default=1.0
        SGD learning rate.
    momentum : float, default=0.7
        SGD momentum.
    grad_clamp : float, default=4.0
        Symmetric gradient clamp.
    batch_size : int, default=16
        Mini-batch size.
    edge_weights : torch.Tensor, optional
        Optional per-edge weights with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Coordinate tensor with shape ``[N, 2]`` and dtype ``float32``.

    Raises
    ------
    ValueError
        If input arguments are invalid.
    RuntimeError
        If the pipeline fails to produce positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if lr <= 0.0:
        raise ValueError("lr must be positive.")
    if momentum < 0.0 or momentum >= 1.0:
        raise ValueError("momentum must be in [0, 1).")
    if grad_clamp <= 0.0:
        raise ValueError("grad_clamp must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
            )

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_sgd2_multi_pipeline(
        steps=steps,
        criteria=criteria,
        criteria_schedules=criteria_schedules,
        lr=lr,
        momentum=momentum,
        grad_clamp=grad_clamp,
        batch_size=batch_size,
    ).apply(problem, state, ctx)

    if final_state.pos is None:
        raise RuntimeError("(SGD)^2 pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_sgd2_multi_pipeline", "layout_sgd2_multi_pipeline"]
