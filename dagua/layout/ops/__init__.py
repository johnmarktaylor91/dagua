"""Composable layout operations foundation.

Three-part state model:
- LayoutProblem: immutable graph structure and user constraints
- SolveState: mutable positions, hierarchy, cached contexts
- RuntimeContext: execution plan, memory policy, trace sinks

Op base class and Pipeline for composing operations.
"""

from dagua.layout.ops.base import Conditional, MultilevelVCycle, Op, Pipeline, Repeat
from dagua.layout.ops.state import (
    AnnealingSchedule,
    ExecutionPlan,
    FlexConstraints,
    GraphStructure,
    HierarchyLevel,
    LayoutProblem,
    MemoryPolicy,
    RuntimeContext,
    SolveState,
)

__all__ = [
    "LayoutProblem",
    "SolveState",
    "RuntimeContext",
    "GraphStructure",
    "HierarchyLevel",
    "ExecutionPlan",
    "MemoryPolicy",
    "AnnealingSchedule",
    "FlexConstraints",
    "Op",
    "Pipeline",
    "Repeat",
    "Conditional",
    "MultilevelVCycle",
]
