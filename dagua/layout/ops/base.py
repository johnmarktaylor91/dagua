"""Base classes for composable layout operations.

Every layout operation is a subclass of ``Op`` with a single method:
``apply(problem, state, ctx) -> SolveState``.

The three-argument signature separates concerns:
- problem: what to lay out (read-only)
- state: working data (read-write)
- ctx: how to execute (infrastructure)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Sequence, Tuple

from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState


class Op(ABC):
    """Base class for all layout operations.

    Subclasses implement ``apply()`` and can document their behavior through
    metadata fields used by logging and best-effort linting.

    Class Attributes
    ----------------
    name : str
        Human-readable name for logging and provenance.
    category : str
        Op category such as ``init`` or ``refinement``.
    reads : tuple[str, ...]
        SolveState fields this op reads. Documentation only.
    writes : tuple[str, ...]
        SolveState fields this op writes. Documentation only.
    requires : tuple[str, ...]
        SolveState fields that should already be set. Best-effort only.
    """

    name: str = "unnamed_op"
    category: str = "unknown"
    reads: Tuple[str, ...] = ()
    writes: Tuple[str, ...] = ()
    requires: Tuple[str, ...] = ()

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

    def __repr__(self) -> str:
        """Return a concise debug representation.

        Returns
        -------
        str
            Class-name-based representation of the operation.
        """
        return f"{self.__class__.__name__}()"


class Pipeline:
    """Ordered sequence of layout operations.

    Parameters
    ----------
    ops : sequence[Op]
        Operations to apply in order.
    name : str, default="pipeline"
        Pipeline name used in representations and logging.
    """

    def __init__(self, ops: Sequence[Op], name: str = "pipeline") -> None:
        """Store the ordered list of operations.

        Parameters
        ----------
        ops : sequence[Op]
            Operations to apply in order.
        name : str, default="pipeline"
            Pipeline name used in representations and logging.

        Returns
        -------
        None
            The pipeline stores the supplied operations.
        """
        self.ops: List[Op] = list(ops)
        self.name = name

    def __call__(
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
            state = op.apply(problem, state, ctx)
            state.ops_applied.append(op.name)
            if len(state.ops_applied) > 100:
                state.ops_applied = state.ops_applied[-100:]
        return state

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

    def __repr__(self) -> str:
        """Return a concise pipeline representation.

        Returns
        -------
        str
            Pipeline name and ordered op names.
        """
        op_names = ", ".join(op.name for op in self.ops)
        return f"Pipeline({self.name}: [{op_names}])"


class Repeat(Op):
    """Repeat a sequence of operations for ``n`` iterations.

    Parameters
    ----------
    n : int
        Number of iterations.
    ops : sequence[Op]
        Operations to repeat.
    """

    name = "repeat"
    category = "control"

    def __init__(self, n: int, ops: Sequence[Op]) -> None:
        """Store the iteration count and inner pipeline.

        Parameters
        ----------
        n : int
            Number of iterations.
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
        """Run the inner pipeline ``n`` times.

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
            Updated state after all iterations complete.
        """
        for _ in range(self.n):
            state = self.inner(problem, state, ctx)
        return state


class Conditional(Op):
    """Run an operation only when a predicate evaluates to ``True``.

    Parameters
    ----------
    predicate : Callable[[LayoutProblem, SolveState, RuntimeContext], bool]
        Predicate controlling whether the wrapped op runs.
    op : Op
        Operation to execute when the predicate holds.
    """

    name = "conditional"
    category = "control"

    def __init__(
        self,
        predicate: Callable[[LayoutProblem, SolveState, RuntimeContext], bool],
        op: Op,
    ) -> None:
        """Store the predicate and wrapped operation.

        Parameters
        ----------
        predicate : Callable[[LayoutProblem, SolveState, RuntimeContext], bool]
            Predicate controlling whether the wrapped op runs.
        op : Op
            Operation to execute when the predicate holds.

        Returns
        -------
        None
            The conditional wrapper stores the supplied predicate and op.
        """
        self.predicate = predicate
        self.op = op

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run the wrapped operation when the predicate returns ``True``.

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
            Updated state if the predicate held, otherwise the original state.
        """
        if self.predicate(problem, state, ctx):
            return self.op.apply(problem, state, ctx)
        return state


class MultilevelVCycle(Op):
    """Specialized multilevel V-cycle orchestrator skeleton.

    The eventual implementation will own lifecycle concerns that do not fit a
    generic ``Pipeline`` wrapper, including hierarchy build, checkpoint
    restore, offloading, per-level planning, and cleanup hooks.

    Parameters
    ----------
    coarsen_op : Op
        Operation responsible for hierarchy coarsening.
    base_layout : Pipeline
        Pipeline used at the coarsest level.
    refine : Pipeline
        Pipeline used during refinement.
    min_nodes : int, default=50
        Coarsening stop threshold.
    """

    name = "multilevel_vcycle"
    category = "structural"

    def __init__(
        self,
        coarsen_op: Op,
        base_layout: Pipeline,
        refine: Pipeline,
        min_nodes: int = 50,
    ) -> None:
        """Store V-cycle hook pipelines and threshold.

        Parameters
        ----------
        coarsen_op : Op
            Operation responsible for hierarchy coarsening.
        base_layout : Pipeline
            Pipeline used at the coarsest level.
        refine : Pipeline
            Pipeline used during refinement.
        min_nodes : int, default=50
            Coarsening stop threshold.

        Returns
        -------
        None
            The orchestrator stores the provided hook objects.
        """
        self.coarsen_op = coarsen_op
        self.base_layout = base_layout
        self.refine = refine
        self.min_nodes = min_nodes

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Raise until the concrete V-cycle implementation exists.

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
            This method never returns in the skeleton implementation.

        Raises
        ------
        NotImplementedError
            Always raised until the V-cycle is implemented.
        """
        raise NotImplementedError(
            "MultilevelVCycle.apply() is a skeleton. Actual V-cycle implementation pending."
        )

    def build_hierarchy(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build the coarsening hierarchy.

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
            This method never returns in the skeleton implementation.

        Raises
        ------
        NotImplementedError
            Always raised until subclasses implement the hook.
        """
        raise NotImplementedError

    def solve_coarsest(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Solve the coarsest hierarchy level.

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
            This method never returns in the skeleton implementation.

        Raises
        ------
        NotImplementedError
            Always raised until subclasses implement the hook.
        """
        raise NotImplementedError

    def prolong_and_refine(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
        level_idx: int,
    ) -> SolveState:
        """Prolong positions and refine a single hierarchy level.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable working state.
        ctx : RuntimeContext
            Execution infrastructure.
        level_idx : int
            Refinement level index being processed.

        Returns
        -------
        SolveState
            This method never returns in the skeleton implementation.

        Raises
        ------
        NotImplementedError
            Always raised until subclasses implement the hook.
        """
        raise NotImplementedError

    def cleanup(self, ctx: RuntimeContext) -> None:
        """Run post-V-cycle cleanup hooks.

        Parameters
        ----------
        ctx : RuntimeContext
            Execution infrastructure and resource policy.

        Returns
        -------
        None
            The default skeleton has no cleanup work.
        """
        return None
