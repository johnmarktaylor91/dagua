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
from dagua.layout.ops.edge_route import (
    BezierControlPointOpt,
    BezierControlPointOptConfig,
    ReconstructEdgeRoutes,
)
from dagua.layout.ops.init import (
    CircularInit,
    CircularInitConfig,
    DeterministicInit,
    DeterministicInitConfig,
    RandomNormalInit,
    RandomNormalInitConfig,
    RandomUniformInit,
    RandomUniformInitConfig,
    XavierInit,
    XavierInitConfig,
)
from dagua.layout.ops.loss_classic import (
    CyclicSampler,
    CyclicSamplerConfig,
    DavidsonHarelEnergyLoss,
    DavidsonHarelEnergyLossConfig,
    ElasticLoss,
    EntropyLoss,
    EntropyLossConfig,
    ExactPairStressLoss,
    ExactPairStressLossConfig,
    KDTreeRepulsionLoss,
    KDTreeRepulsionLossConfig,
    KLDivergenceLoss,
    KLDivergenceLossConfig,
    LinLogAttractionLoss,
    LinLogAttractionLossConfig,
    LinLogRepulsionLoss,
    LinLogRepulsionLossConfig,
    PivotApproxStressLoss,
    SGD2CriterionLoss,
    SGD2CriterionLossConfig,
    SGD2CrossingDetectorStep,
    SGD2CrossingDetectorStepConfig,
    UMAPCrossEntropyLoss,
    UMAPCrossEntropyLossConfig,
)
from dagua.layout.ops.postprocess import (
    CenterPositions,
    DirectionTransform,
    DirectionTransformConfig,
    NormalizePositions,
    NormalizePositionsConfig,
    ScalePositions,
    ScalePositionsConfig,
    SpreadFanoutChildren,
    SpreadFanoutChildrenConfig,
    StripDummyNodes,
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
from dagua.layout.ops.utility import (
    Checkpoint,
    CheckpointConfig,
    DiskOffload,
    DiskReload,
    GarbageCollect,
    ProgressReport,
    ProgressReportConfig,
    Timer,
    VRAMGuard,
    VRAMGuardConfig,
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
    # Classic loss ops
    "ExactPairStressLoss",
    "ExactPairStressLossConfig",
    "PivotApproxStressLoss",
    "KLDivergenceLoss",
    "KLDivergenceLossConfig",
    "UMAPCrossEntropyLoss",
    "UMAPCrossEntropyLossConfig",
    "LinLogAttractionLoss",
    "LinLogAttractionLossConfig",
    "LinLogRepulsionLoss",
    "LinLogRepulsionLossConfig",
    "EntropyLoss",
    "EntropyLossConfig",
    "DavidsonHarelEnergyLoss",
    "DavidsonHarelEnergyLossConfig",
    "ElasticLoss",
    "KDTreeRepulsionLoss",
    "KDTreeRepulsionLossConfig",
    "SGD2CriterionLoss",
    "SGD2CriterionLossConfig",
    "SGD2CrossingDetectorStep",
    "SGD2CrossingDetectorStepConfig",
    "CyclicSampler",
    "CyclicSamplerConfig",
    # Edge routing ops
    "BezierControlPointOpt",
    "BezierControlPointOptConfig",
    "ReconstructEdgeRoutes",
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
    # Concrete ops
    "CircularInit",
    "CircularInitConfig",
    "DeterministicInit",
    "DeterministicInitConfig",
    "RandomNormalInit",
    "RandomNormalInitConfig",
    "RandomUniformInit",
    "RandomUniformInitConfig",
    "XavierInit",
    "XavierInitConfig",
    "CenterPositions",
    "ScalePositions",
    "ScalePositionsConfig",
    "NormalizePositions",
    "NormalizePositionsConfig",
    "DirectionTransform",
    "DirectionTransformConfig",
    "StripDummyNodes",
    "SpreadFanoutChildren",
    "SpreadFanoutChildrenConfig",
    "Checkpoint",
    "CheckpointConfig",
    "DiskOffload",
    "DiskReload",
    "GarbageCollect",
    "ProgressReport",
    "ProgressReportConfig",
    "Timer",
    "VRAMGuard",
    "VRAMGuardConfig",
]


# Late imports: trigger @register_op in all category modules.
# Placed after __all__ to avoid circular import issues.
def _register_all_ops() -> None:
    """Import every category module so their @register_op decorators fire."""
    import importlib

    for mod_name in (
        "anneal",
        "coarsen",
        "context",
        "converge",
        "coordinate",
        "distance",
        "embed",
        "force",
        "layering",
        "loss_engine",
        "optimize",
        "ordering",
        "preprocess",
        "project",
    ):
        importlib.import_module(f"dagua.layout.ops.{mod_name}")


_register_all_ops()
