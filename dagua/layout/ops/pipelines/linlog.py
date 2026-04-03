"""LinLog expressed as a composable ops pipeline."""

from __future__ import annotations

import math
from typing import ClassVar, Optional, Tuple

import torch

from dagua.layout.classic.linlog import (
    _initialize_positions,
    _layout_device,
    _layout_extent,
    _linlog_loss,
    _normalize_positions,
)
from dagua.layout.ops.base import Conditional, Op, Pipeline, Repeat
from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory

_INITIAL_LR_KEY = "linlog_initial_lr"


class _InitializeLinLogPositions(Op):
    """Initialize positions exactly like classic LinLog."""

    name: ClassVar[str] = "linlog_initialize_positions"
    category: ClassVar[OpCategory] = OpCategory.INIT
    writes: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Seed ``state.pos`` from the classic random-normal initializer.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs containing ``num_nodes`` and ``seed``.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with ``state.pos`` initialized on the classic output device.
        """
        del ctx

        output_device = _layout_device(
            edge_index=problem.edge_index,
            node_sizes=problem.node_sizes,
        )
        if problem.num_nodes == 0:
            state.pos = torch.empty((0, 2), dtype=torch.float32, device=output_device)
            return state
        if problem.num_nodes == 1:
            state.pos = torch.zeros((1, 2), dtype=torch.float32, device=output_device)
            return state

        state.pos = _initialize_positions(
            problem.num_nodes, output_device, problem.seed
        ).requires_grad_(True)
        return state


class _CreateLinLogOptimizer(Op):
    """Create the classic Adam optimizer for LinLog."""

    name: ClassVar[str] = "linlog_create_optimizer"
    category: ClassVar[OpCategory] = OpCategory.OPTIMIZE
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ("optimizer", "extras")
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Create the Adam optimizer with the classic learning-rate rule.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state containing learnable positions.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with the configured Adam optimizer stored in
            ``state.optimizer``.

        Raises
        ------
        ValueError
            If ``state.pos`` is missing.
        """
        del ctx

        if state.pos is None:
            raise ValueError("_CreateLinLogOptimizer requires state.pos to be set.")

        initial_lr = min(0.05, 0.8 / float(max(problem.num_nodes, 1)))
        state.optimizer = torch.optim.Adam([state.pos], lr=initial_lr)
        state.extras[_INITIAL_LR_KEY] = float(initial_lr)
        return state


class _BackpropLinLogLoss(Op):
    """Evaluate the exact classic LinLog loss and backpropagate it."""

    name: ClassVar[str] = "linlog_backprop_loss"
    category: ClassVar[OpCategory] = OpCategory.LOSS
    reads: ClassVar[Tuple[str, ...]] = ("pos", "optimizer", "step")
    writes: ClassVar[Tuple[str, ...]] = ("prev_loss",)
    requires: ClassVar[Tuple[str, ...]] = ("pos", "optimizer")

    def __init__(self, a: float, r: float) -> None:
        """Store the LinLog objective exponents.

        Parameters
        ----------
        a : float
            Attraction exponent.
        r : float
            Repulsion exponent.

        Returns
        -------
        None
            This constructor stores configuration only.
        """
        self.a = float(a)
        self.r = float(r)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run ``zero_grad()``, evaluate the classic loss, and backpropagate.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state containing positions and the optimizer.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with ``prev_loss`` updated to the current LinLog objective.

        Raises
        ------
        ValueError
            If ``state.pos`` or ``state.optimizer`` is missing.
        TypeError
            If ``state.optimizer`` is not a PyTorch optimizer.
        """
        del ctx

        if state.pos is None:
            raise ValueError("_BackpropLinLogLoss requires state.pos to be set.")
        if state.optimizer is None:
            raise ValueError("_BackpropLinLogLoss requires state.optimizer to be set.")
        if not isinstance(state.optimizer, torch.optim.Optimizer):
            raise TypeError("_BackpropLinLogLoss requires a torch.optim.Optimizer.")

        state.optimizer.zero_grad(set_to_none=True)
        loss = _linlog_loss(
            positions=state.pos,
            edge_index=problem.edge_index,
            seed=problem.seed,
            step=state.step,
            a=self.a,
            r=self.r,
            edge_weights=problem.edge_weights,
        )
        loss.backward()
        state.prev_loss = float(loss.item())
        return state


class _StepLinLogOptimizer(Op):
    """Apply one Adam step for LinLog."""

    name: ClassVar[str] = "linlog_step_optimizer"
    category: ClassVar[OpCategory] = OpCategory.OPTIMIZE
    reads: ClassVar[Tuple[str, ...]] = ("optimizer",)
    writes: ClassVar[Tuple[str, ...]] = ("optimizer",)
    requires: ClassVar[Tuple[str, ...]] = ("optimizer",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run the classic Adam update.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state containing the optimizer.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State after one optimizer step.

        Raises
        ------
        ValueError
            If ``state.optimizer`` is missing.
        TypeError
            If ``state.optimizer`` is not a PyTorch optimizer.
        """
        del problem, ctx

        if state.optimizer is None:
            raise ValueError("_StepLinLogOptimizer requires state.optimizer to be set.")
        if not isinstance(state.optimizer, torch.optim.Optimizer):
            raise TypeError("_StepLinLogOptimizer requires a torch.optim.Optimizer.")

        state.optimizer.step()
        return state


class _DecayLinLogLearningRate(Op):
    """Apply the classic per-step linear LR interpolation."""

    name: ClassVar[str] = "linlog_decay_learning_rate"
    category: ClassVar[OpCategory] = OpCategory.ANNEAL
    reads: ClassVar[Tuple[str, ...]] = ("optimizer", "extras", "step", "total_steps")
    writes: ClassVar[Tuple[str, ...]] = ("optimizer",)
    requires: ClassVar[Tuple[str, ...]] = ("optimizer",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Update the optimizer LR using the classic LinLog schedule.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state containing the optimizer and step counters.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with an updated optimizer learning rate.

        Raises
        ------
        ValueError
            If the optimizer or cached initial learning rate is missing.
        TypeError
            If ``state.optimizer`` is not a PyTorch optimizer.
        """
        del problem, ctx

        if state.optimizer is None:
            raise ValueError("_DecayLinLogLearningRate requires state.optimizer to be set.")
        if not isinstance(state.optimizer, torch.optim.Optimizer):
            raise TypeError("_DecayLinLogLearningRate requires a torch.optim.Optimizer.")

        initial_lr = state.extras.get(_INITIAL_LR_KEY)
        if not isinstance(initial_lr, (float, int)):
            raise ValueError("_DecayLinLogLearningRate requires a cached initial LR.")

        initial_lr = float(initial_lr)
        final_lr = max(initial_lr * 0.1, initial_lr / math.sqrt(float(max(state.total_steps, 1))))
        fraction = float(state.step + 1) / float(max(state.total_steps, 1))
        decay = initial_lr + (final_lr - initial_lr) * fraction
        state.optimizer.param_groups[0]["lr"] = decay
        return state


class _FinalizeLinLogPositions(Op):
    """Apply the classic LinLog centering and scale normalization."""

    name: ClassVar[str] = "linlog_finalize_positions"
    category: ClassVar[OpCategory] = OpCategory.POSTPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Normalize the final coordinates like classic ``layout_linlog``.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state containing final coordinates.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with final output positions.

        Raises
        ------
        ValueError
            If ``state.pos`` is missing.
        """
        del ctx

        if state.pos is None:
            raise ValueError("_FinalizeLinLogPositions requires state.pos to be set.")

        output_device = _layout_device(
            edge_index=problem.edge_index,
            node_sizes=problem.node_sizes,
        )
        if problem.num_nodes == 0:
            state.pos = torch.empty((0, 2), dtype=torch.float32, device=output_device)
            return state
        if problem.num_nodes == 1:
            state.pos = torch.zeros((1, 2), dtype=torch.float32, device=output_device)
            return state

        extent = _layout_extent(num_nodes=problem.num_nodes, node_sizes=problem.node_sizes)
        normalized = _normalize_positions(state.pos.detach(), extent=extent)
        state.pos = normalized.to(dtype=torch.float32, device=output_device)
        return state


def build_linlog_pipeline(
    steps: int = 300,
    a: float = 1.0,
    r: float = 0.0,
) -> Pipeline:
    """Build a LinLog pipeline that matches classic ``layout_linlog``.

    Parameters
    ----------
    steps : int, default=300
        Number of Adam updates.
    a : float, default=1.0
        Attraction exponent.
    r : float, default=0.0
        Repulsion exponent.

    Returns
    -------
    Pipeline
        Pipeline that reproduces classic LinLog exactly.

    Raises
    ------
    ValueError
        If ``steps``, ``a``, or ``r`` are invalid.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if a < 0.0:
        raise ValueError("a must be non-negative.")
    if r < 0.0:
        raise ValueError("r must be non-negative.")

    optimize_pipeline = Pipeline(
        [
            _CreateLinLogOptimizer(),
            Repeat(
                n=steps,
                ops=[
                    _BackpropLinLogLoss(a=a, r=r),
                    _StepLinLogOptimizer(),
                    _DecayLinLogLearningRate(),
                ],
            ),
        ],
        name="linlog_optimize",
    )

    return Pipeline(
        [
            FixedSteps(FixedStepsConfig(n=steps)),
            _InitializeLinLogPositions(),
            Conditional(
                predicate=lambda problem, state, ctx: problem.num_nodes > 1,
                op=optimize_pipeline,
            ),
            _FinalizeLinLogPositions(),
        ],
        name="linlog_pipeline",
    )


def layout_linlog_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 300,
    seed: int = 42,
    a: float = 1.0,
    r: float = 0.0,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the LinLog pipeline as a drop-in replacement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor used only to scale the final layout.
    steps : int, default=300
        Number of Adam updates.
    seed : int, default=42
        Random seed for initialization and repulsion sampling.
    a : float, default=1.0
        Attraction exponent.
    r : float, default=0.0
        Repulsion exponent.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Final position tensor with the same dtype, device, and values as
        classic ``layout_linlog``.

    Raises
    ------
    ValueError
        If the public LinLog inputs are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if a < 0.0:
        raise ValueError("a must be non-negative.")
    if r < 0.0:
        raise ValueError("r must be non-negative.")
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
    final_state = build_linlog_pipeline(steps=steps, a=a, r=r).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("LinLog pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_linlog_pipeline", "layout_linlog_pipeline"]
