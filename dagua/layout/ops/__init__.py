"""Composable layout operations foundation.

Three-part state model:
- LayoutProblem: immutable graph structure and user constraints
- SolveState: mutable positions, hierarchy, cached contexts
- RuntimeContext: execution plan, memory policy, trace sinks

Op base class, LossOp for differentiable losses, and composition
patterns (Pipeline, Repeat, Conditional, LossGroup, EarlyBreak,
MultilevelVCycle) for building layout workflows.

OpCategory enum and registry for operation discovery.
"""

from dagua.layout.ops.base import (
    Conditional,
    EarlyBreak,
    LossGroup,
    LossOp,
    MultilevelVCycle,
    Op,
    Pipeline,
    Repeat,
)
from dagua.layout.ops.state import (
    AnnealingSchedule,
    ExecutionPlan,
    FlexConstraints,
    GraphStructure,
    HierarchyLevel,
    LayoutProblem,
    MemoryPolicy,
    NullTraceSink,
    RuntimeContext,
    SolveState,
    TraceSink,
)
from dagua.layout.ops.taxonomy import (
    OpCategory,
    get_op_class,
    list_categories,
    list_ops,
    register_op,
    unregister_op,
)

__all__ = [
    # State model
    "LayoutProblem",
    "SolveState",
    "RuntimeContext",
    "GraphStructure",
    "HierarchyLevel",
    "ExecutionPlan",
    "MemoryPolicy",
    "AnnealingSchedule",
    "FlexConstraints",
    "TraceSink",
    "NullTraceSink",
    # Op base classes
    "Op",
    "LossOp",
    # Composition patterns
    "Pipeline",
    "Repeat",
    "Conditional",
    "EarlyBreak",
    "LossGroup",
    "MultilevelVCycle",
    # Taxonomy
    "OpCategory",
    "register_op",
    "unregister_op",
    "get_op_class",
    "list_ops",
    "list_categories",
]
