"""Composable layout operations foundation.

Three-part state model:
- LayoutProblem: immutable graph structure and user constraints
- SolveState: mutable positions, hierarchy, cached contexts
- RuntimeContext: execution plan, memory policy, trace sinks

Op base class, LossOp for differentiable losses, and composition
patterns (Pipeline, Repeat, Conditional, LossGroup, EarlyBreak)
for building layout workflows.

OpCategory enum and registry for operation discovery.
"""

import importlib
import logging
from pathlib import Path
from typing import Set

from dagua.layout.ops.base import (
    Conditional,
    EarlyBreak,
    LossGroup,
    LossOp,
    Op,
    Pipeline,
    Repeat,
)
from dagua.layout.ops.cluster_driver import (
    ClusterAwareDriver,
    ClusterPlacement,
    NativeClusterLevelLayout,
)
from dagua.layout.ops.edge_route import (
    BezierControlPointOpt,
    BezierControlPointOptConfig,
    ReconstructEdgeRoutes,
)
from dagua.layout.ops.gem import (
    GEMApplyDisplacement,
    GEMComputeImpulse,
    GEMConvergenceCheck,
    GEMSequentialStep,
    GEMUpdateTemperatures,
)
from dagua.layout.ops.graphviz_radial_circular import (
    CircoAssignCircularCoordinates,
    TwopiAssignRadialCoordinates,
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
    OP_REGISTRY,
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
    "ClusterAwareDriver",
    "ClusterPlacement",
    "NativeClusterLevelLayout",
    "Pipeline",
    "Repeat",
    "Conditional",
    "EarlyBreak",
    "LossGroup",
    "GEMApplyDisplacement",
    "GEMConvergenceCheck",
    "GEMComputeImpulse",
    "GEMSequentialStep",
    "GEMUpdateTemperatures",
    "CircoAssignCircularCoordinates",
    "TwopiAssignRadialCoordinates",
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
    "OP_REGISTRY",
]

_EXPECTED_OP_MODULES = (
    "anneal",
    "barycenter",
    "brandes_koepf",
    "cluster_arrange",
    "cluster_driver",
    "coarsen",
    "context",
    "converge",
    "coordinate",
    "crossing_swap",
    "d3dag",
    "d3force",
    "davidson_harel",
    "dagre",
    "distance",
    "drl",
    "edge_route",
    "embed",
    "fcose",
    "fmmm",
    "force",
    "force_2d_init",
    "gem",
    "graphviz_radial_circular",
    "hde",
    "init",
    "layering",
    "lgl",
    "loss_classic",
    "loss_engine",
    "maxent_stress",
    "native_stress",
    "neulay",
    "optimize",
    "ordering",
    "postprocess",
    "preprocess",
    "project",
    "prolong",
    "quadtree",
    "sfdp",
    "sgd2_multi",
    "spatial_hash",
    "scc",
    "stress",
    "stress_sgd",
    "sugiyama",
    "tsnet",
    "tutte",
    "umap",
    "utility",
    "vcycle",
    "yifanhu",
)

log = logging.getLogger(__name__)


# Late imports: trigger @register_op in all category modules.
# Placed after __all__ to avoid circular import issues.
def _register_all_ops() -> None:
    """Import all operation modules to trigger ``@register_op`` decorators.

    The list of expected modules is retained for validation to catch package
    drift. If discovery and expectation diverge, a warning is emitted but import
    continues from discovered modules.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    expected_modules: Set[str] = set(_EXPECTED_OP_MODULES)
    discovered_modules: Set[str] = set()
    ops_dir = Path(__file__).resolve().parent
    excluded_modules = {
        "__init__",
        "base",
        "state",
        "taxonomy",
        "graph_utils",
        "cluster_geometry",
    }

    for module_path in ops_dir.glob("*.py"):
        stem = module_path.stem
        if stem in excluded_modules or stem.startswith("_"):
            continue
        discovered_modules.add(stem)

    if not discovered_modules:
        log.warning("No op modules discovered in %s; falling back to expected list.", ops_dir)
        discovered_modules = expected_modules

    missing_modules = expected_modules.difference(discovered_modules)
    extra_modules = discovered_modules.difference(expected_modules)
    if missing_modules or extra_modules:
        log.warning(
            "Mismatch between expected and discovered op modules. Missing: %s. Extra: %s.",
            sorted(missing_modules),
            sorted(extra_modules),
        )

    for mod_name in sorted(discovered_modules):
        importlib.import_module(f"dagua.layout.ops.{mod_name}")


_register_all_ops()
