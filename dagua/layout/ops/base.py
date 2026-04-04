"""Base classes for composable layout operations.

Every layout operation is a subclass of ``Op`` with a single method:
``apply(problem, state, ctx) -> SolveState``.

The three-argument signature separates concerns:
- problem: what to lay out (read-only)
- state: working data (read-write)
- ctx: how to execute (infrastructure)

Differentiable loss evaluation uses the ``LossOp`` subclass, which adds
``evaluate(problem, state, ctx) -> torch.Tensor`` returning a scalar loss.
``LossGroup`` composes multiple ``LossOp`` instances and manages the
backward pass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch

from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory

# ---------------------------------------------------------------------------
# Op -- base class for all operations
# ---------------------------------------------------------------------------


class Op(ABC):
    """Base class for all layout operations.

    Subclasses implement ``apply()`` and can document their behavior through
    metadata fields used by logging and best-effort linting.

    Class Attributes
    ----------------
    name : str
        Human-readable name for logging and provenance.
    category : OpCategory
        Op category from the taxonomy enum.
    reads : tuple[str, ...]
        SolveState fields this op reads. Documentation only.
    writes : tuple[str, ...]
        SolveState fields this op writes. Documentation only.
    requires : tuple[str, ...]
        SolveState fields that should already be set. Best-effort only.
    access_pattern : str
        How this op accesses node data: ``"global"`` (all nodes),
        ``"edge"`` (edge endpoints only), ``"sampled"`` (sampled subset).
        Used by SubsetGPU executor for device routing. Defaults to
        ``"global"``.
    """

    name: str = "unnamed_op"
    category: Union[OpCategory, str] = OpCategory.UNKNOWN
    reads: Tuple[str, ...] = ()
    writes: Tuple[str, ...] = ()
    requires: Tuple[str, ...] = ()
    access_pattern: str = "global"

    @abstractmethod
    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Apply the operation.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable problem inputs.
        state : SolveState
            Mutable working state.
        ctx : RuntimeContext
            Execution infrastructure such as memory and trace policy.

        Returns
        -------
        SolveState
            Updated solve state. Implementations may mutate in place.
        """

    def describe(self) -> str:
        """Return a one-line human-readable description.

        Returns
        -------
        str
            Operation name and category.
        """
        cat = self.category.value if isinstance(self.category, OpCategory) else self.category
        return f"{self.name} ({cat})"

    def __repr__(self) -> str:
        """Return a concise debug representation.

        Returns
        -------
        str
            Class-name-based representation of the operation.
        """
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# LossOp -- differentiable loss evaluation
# ---------------------------------------------------------------------------


class LossOp(Op):
    """Base class for differentiable loss operations.

    Unlike plain ``Op``, a ``LossOp`` exposes ``evaluate()`` which returns
    a scalar ``torch.Tensor`` (the loss value). This allows ``LossGroup``
    to accumulate multiple losses before calling ``backward()``.

    The default ``apply()`` calls ``evaluate()``, stores the loss in
    ``state.prev_loss``, and calls ``backward()`` for standalone use.
    Inside a ``LossGroup``, only ``evaluate()`` is called -- the group
    manages the backward pass.

    Subclasses must implement ``evaluate()``.

    Class Attributes
    ----------------
    weight_key : str
        Name of the weight in the annealing schedule (e.g. ``"w_repel"``).
        Used by ``LossGroup`` to look up the current annealed weight.
    default_weight : float
        Fallback weight when no annealing schedule is active.
    """

    weight_key: str = ""
    default_weight: float = 1.0

    @abstractmethod
    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Compute the scalar loss value.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable problem inputs.
        state : SolveState
            Mutable working state (positions must have ``requires_grad``).
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor suitable for ``backward()``.
        """

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Evaluate the loss and call backward (standalone mode).

        When used inside a ``LossGroup``, only ``evaluate()`` is called.
        This ``apply()`` is for standalone use where a single loss needs
        to drive a gradient update.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable problem inputs.
        state : SolveState
            Mutable working state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with ``prev_loss`` updated.
        """
        loss = self.evaluate(problem, state, ctx)
        loss.backward()
        state.prev_loss = loss.item()
        return state


# ---------------------------------------------------------------------------
# Pipeline -- sequential composition (also an Op for nesting)
# ---------------------------------------------------------------------------


class Pipeline(Op):
    """Ordered sequence of layout operations.

    ``Pipeline`` is itself an ``Op``, so it can be nested inside
    ``Conditional``, ``Repeat``, and other composition patterns.

    Parameters
    ----------
    ops : sequence[Op]
        Operations to apply in order.
    name : str, default="pipeline"
        Pipeline name used in representations and logging.
    trace_between : bool, default=False
        When ``True``, fires ``ctx.trace_sink.op_start()`` /
        ``op_end()`` and ``snapshot()`` at each op boundary.
    """

    category = OpCategory.CONTROL

    def __init__(
        self,
        ops: Sequence[Op],
        name: str = "pipeline",
        trace_between: bool = False,
    ) -> None:
        """Store the ordered list of operations.

        Parameters
        ----------
        ops : sequence[Op]
            Operations to apply in order.
        name : str, default="pipeline"
            Pipeline name used in representations and logging.
        trace_between : bool, default=False
            Fire trace events between ops.

        Returns
        -------
        None
            The pipeline stores the supplied operations.
        """
        self.ops: List[Op] = list(ops)
        # Override the class-level name with the instance name.
        self.name = name
        self.trace_between = trace_between

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run all operations in sequence.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable working state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            Final state after all operations have run.
        """
        for op in self.ops:
            if self.trace_between:
                ctx.trace_sink.op_start(op.name, state.step)
                if state.pos is not None:
                    ctx.trace_sink.snapshot(state.pos, state.step)
            state = op.apply(problem, state, ctx)
            state.ops_applied.append(op.name)
            if len(state.ops_applied) > 100:
                state.ops_applied = state.ops_applied[-100:]
            if self.trace_between:
                ctx.trace_sink.op_end(op.name, state.step)
        return state

    # Keep __call__ as an alias for backward compat.
    def __call__(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Alias for ``apply()`` -- backward compatibility.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable working state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            Final state after all operations have run.
        """
        return self.apply(problem, state, ctx)

    def lint(self, state: SolveState) -> List[str]:
        """Perform best-effort precondition checks.

        This method is intentionally advisory. Dynamic control flow and
        runtime-dependent state make static validation incomplete.

        Parameters
        ----------
        state : SolveState
            Current state to inspect.

        Returns
        -------
        list[str]
            Warning messages for missing documented preconditions.
        """
        warnings: List[str] = []
        available = {
            field_name
            for field_name in state.__dataclass_fields__
            if getattr(state, field_name) is not None
        }
        for op in self.ops:
            for requirement in op.requires:
                if requirement not in available:
                    warnings.append(f"{op.name} expects '{requirement}' to be set")
            available.update(op.writes)
        return warnings

    def dependency_graph(self) -> dict:
        """Return declared reads/writes per op for visualization.

        Returns
        -------
        dict
            Mapping from op name to ``{"reads": ..., "writes": ...}``.
        """
        return {op.name: {"reads": op.reads, "writes": op.writes} for op in self.ops}

    def __repr__(self) -> str:
        """Return a concise pipeline representation.

        Returns
        -------
        str
            Pipeline name and ordered op names.
        """
        op_names = ", ".join(op.name for op in self.ops)
        return f"Pipeline({self.name}: [{op_names}])"


# ---------------------------------------------------------------------------
# Repeat -- fixed iteration loop with convergence check
# ---------------------------------------------------------------------------


class Repeat(Op):
    """Repeat a sequence of operations for up to ``n`` iterations.

    Stops early if ``state.converged`` is set to ``True`` by an inner op
    (e.g., ``EarlyBreak``). Automatically increments ``state.step`` after
    each iteration.

    Parameters
    ----------
    n : int
        Maximum number of iterations.
    ops : sequence[Op]
        Operations to repeat.
    """

    name = "repeat"
    category = OpCategory.CONTROL

    def __init__(self, n: int, ops: Sequence[Op]) -> None:
        """Store the iteration count and inner pipeline.

        Parameters
        ----------
        n : int
            Maximum number of iterations.
        ops : sequence[Op]
            Operations to repeat.

        Returns
        -------
        None
            The repeat wrapper stores the inner pipeline.
        """
        self.n = n
        self.inner = Pipeline(ops, name="repeat_inner")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run the inner pipeline up to ``n`` times.

        Stops early if ``state.converged`` becomes ``True``.
        Increments ``state.step`` after each iteration.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable working state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            Updated state after iterations complete.
        """
        # Save and reset converged so a previous Repeat's convergence
        # does not prevent this loop from running.
        saved_converged = state.converged
        state.converged = False
        for _ in range(self.n):
            if state.converged:
                break
            state = self.inner.apply(problem, state, ctx)
            state.step += 1
        # If this loop did NOT converge, restore the previous flag
        # so outer pipelines see the original state.
        if not state.converged:
            state.converged = saved_converged
        return state


# ---------------------------------------------------------------------------
# EarlyBreak -- convergence escape for Repeat loops
# ---------------------------------------------------------------------------


class EarlyBreak(Op):
    """Set ``state.converged = True`` when a predicate holds.

    Designed for use inside ``Repeat``. When the predicate returns
    ``True``, sets the convergence flag so that ``Repeat`` stops
    after the current iteration completes.

    Parameters
    ----------
    predicate : Callable[[LayoutProblem, SolveState, RuntimeContext], bool]
        Convergence condition.
    """

    name = "early_break"
    category = OpCategory.CONVERGE

    def __init__(
        self,
        predicate: Callable[[LayoutProblem, SolveState, RuntimeContext], bool],
    ) -> None:
        """Store the convergence predicate.

        Parameters
        ----------
        predicate : Callable[[LayoutProblem, SolveState, RuntimeContext], bool]
            Returns ``True`` when the loop should stop.

        Returns
        -------
        None
            The early break stores the predicate.
        """
        self.predicate = predicate

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Check the predicate and set ``state.converged`` if met.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable working state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with ``converged`` potentially set to ``True``.
        """
        if self.predicate(problem, state, ctx):
            state.converged = True
        return state


# ---------------------------------------------------------------------------
# Conditional -- predicate-based branching (with else_op)
# ---------------------------------------------------------------------------


class Conditional(Op):
    """Run an operation when a predicate holds, optionally with an else branch.

    Parameters
    ----------
    predicate : Callable[[LayoutProblem, SolveState, RuntimeContext], bool]
        Predicate controlling which branch runs.
    op : Op
        Operation to execute when the predicate is ``True``.
    else_op : Op or None, default=None
        Operation to execute when the predicate is ``False``.
        If ``None``, the state passes through unchanged.
    """

    name = "conditional"
    category = OpCategory.CONTROL

    def __init__(
        self,
        predicate: Callable[[LayoutProblem, SolveState, RuntimeContext], bool],
        op: Op,
        else_op: Optional[Op] = None,
    ) -> None:
        """Store the predicate and branch operations.

        Parameters
        ----------
        predicate : Callable[[LayoutProblem, SolveState, RuntimeContext], bool]
            Predicate controlling which branch runs.
        op : Op
            Operation to execute when the predicate holds.
        else_op : Op or None, default=None
            Operation to execute when the predicate does not hold.

        Returns
        -------
        None
            The conditional wrapper stores the supplied objects.
        """
        self.predicate = predicate
        self.op = op
        self.else_op = else_op

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run the appropriate branch based on the predicate.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable working state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            Updated state from whichever branch ran.
        """
        if self.predicate(problem, state, ctx):
            return self.op.apply(problem, state, ctx)
        if self.else_op is not None:
            return self.else_op.apply(problem, state, ctx)
        return state


# ---------------------------------------------------------------------------
# LossGroup -- shared-context loss accumulation with backward
# ---------------------------------------------------------------------------


class LossGroup(Op):
    """Evaluate multiple ``LossOp`` instances and manage the backward pass.

    In ``"combined"`` mode (default): evaluates all losses, sums their
    weighted values, calls ``backward()`` once.

    In ``"per_loss"`` mode: evaluates and backwards each loss separately,
    freeing intermediates between losses. Uses 3-4x less memory at the
    cost of recomputing shared tensors.

    Parameters
    ----------
    losses : sequence[LossOp]
        Loss operations to evaluate.
    backward_mode : str, default="combined"
        Either ``"combined"`` or ``"per_loss"``.
    """

    name = "loss_group"
    category = OpCategory.CONTROL

    def __init__(
        self,
        losses: Sequence[LossOp],
        backward_mode: str = "combined",
    ) -> None:
        """Store the loss ops and backward mode.

        Parameters
        ----------
        losses : sequence[LossOp]
            Loss operations to evaluate.
        backward_mode : str, default="combined"
            Backward strategy: ``"combined"`` or ``"per_loss"``.

        Returns
        -------
        None
            The group stores the loss ops.
        """
        self.losses: List[LossOp] = list(losses)
        self.backward_mode = backward_mode

    def _get_weight(self, loss_op: LossOp, state: SolveState) -> float:
        """Look up the current weight for a loss op.

        Uses the annealing schedule if available, otherwise falls back
        to the loss op's ``default_weight``.

        Parameters
        ----------
        loss_op : LossOp
            The loss operation.
        state : SolveState
            Current state (checked for annealing schedule).

        Returns
        -------
        float
            The current weight for this loss.
        """
        if (
            state.annealing is not None
            and loss_op.weight_key
            and loss_op.weight_key in state.annealing.current_weights
        ):
            return state.annealing.current_weights[loss_op.weight_key]
        return loss_op.default_weight

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Evaluate all losses and call backward.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable problem inputs.
        state : SolveState
            Mutable working state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with ``prev_loss`` updated to the total weighted loss.
        """
        if self.backward_mode == "per_loss":
            total = 0.0
            for loss_op in self.losses:
                w = self._get_weight(loss_op, state)
                if w == 0.0:
                    continue
                val = loss_op.evaluate(problem, state, ctx)
                (w * val).backward()
                total += w * val.item()
            state.prev_loss = total
        else:
            # Combined mode: sum then single backward.
            # Initialize from the first nonzero loss to inherit
            # device and dtype (torch.tensor(0.0) would be CPU-only).
            total_tensor = None
            for loss_op in self.losses:
                w = self._get_weight(loss_op, state)
                if w == 0.0:
                    continue
                val = loss_op.evaluate(problem, state, ctx)
                term = w * val
                total_tensor = term if total_tensor is None else total_tensor + term
            if total_tensor is not None and total_tensor.requires_grad:
                total_tensor.backward()
            state.prev_loss = total_tensor.item() if total_tensor is not None else 0.0
        return state

    def __repr__(self) -> str:
        """Return a concise representation.

        Returns
        -------
        str
            Loss names and backward mode.
        """
        names = ", ".join(op.name for op in self.losses)
        return f"LossGroup([{names}], mode={self.backward_mode})"


# ---------------------------------------------------------------------------
