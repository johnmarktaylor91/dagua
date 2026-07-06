"""First-class native stress pipeline for cyclic and undirected graphs."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch

from dagua.config import LayoutConfig
from dagua.layout.ops.base import Conditional, Pipeline, Repeat
from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig
from dagua.layout.ops.distance import PivotDistanceQueries, PivotSelection, PivotSelectionConfig
from dagua.layout.ops.embed import PivotMDSComputeCoordinates
from dagua.layout.ops.native_stress import (
    InflateStressTargetDistances,
    InflateStressTargetDistancesConfig,
    PrepareWarmStartStressMajorization,
    PrepareWarmStartStressMajorizationConfig,
    ResetConvergence,
    RunWarmStartStressSGDApproximateSchedule,
    RunWarmStartStressSGDApproximateScheduleConfig,
)
from dagua.layout.ops.pipelines._native_shared import (
    _extract_component_problem,
    _tile_component_positions,
)
from dagua.layout.ops.postprocess import (
    AspectRatioFit,
    AspectRatioFitConfig,
    PivotMDSFinalizePositions,
)
from dagua.layout.ops.preprocess import BuildAdjacency, BuildAdjacencyConfig, DetectComponents
from dagua.layout.ops.project import OverlapProjection, OverlapProjectionConfig
from dagua.layout.ops.sgd2_multi import (
    SmoothSteps,
    _InitSGD2MultiState,
    _RunSGD2MultiOptimization,
)
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.stress import (
    CaptureStressMajorizationStress,
    CheckStressMajorizationEpsilon,
    CollectStressMajorizationTrace,
    FinalizeStressMajorizationPositions,
    SmacofStep,
)
from dagua.layout.ops.stress_sgd import (
    InitializeStressSGDState,
    PrepareStressSGDTerms,
    RunStressSGDExactSchedule,
)
from dagua.layout.resolve import normalize_node_sizes

_DEFAULT_EPS = 0.01
_DEFAULT_MAX_EXACT_NODES = 10_000
_DEFAULT_SMACOF_MAX_NODES = 5_000


@dataclass(frozen=True)
class NativeStressConfig:
    """Configuration for the r79 native stress core.

    Parameters
    ----------
    steps : int, default=0
        Stress-SGD epoch budget. ``0`` selects an automatic graph-size budget.
    late_steps : int, default=0
        Late multicriteria SGD budget. ``0`` selects a short automatic budget.
    n_pivots : int, default=0
        Pivot-MDS landmark count. ``0`` selects an automatic graph-size count.
    eps : float, default=0.01
        Final Stress-SGD schedule shrinkage.
    max_exact_nodes : int, default=10000
        Node cutoff for exact Stress-SGD all-pairs terms.
    sample_size : int | str, default="auto"
        Approximate Stress-SGD sample budget.
    size_aware : bool, default=True
        Inflate adjacent targets by endpoint bounding radii.
    size_scale : float, default=1.0
        Multiplier for size-aware target inflation.
    repulsion_mode : {"none", "fa2_degree", "linlog"}, default="none"
        Reserved native-stress repulsion selector. Non-default modes are
        accepted for evidence probes but remain off by default.
    smacof_iters : int, default=4
        Warm-start SMACOF polish iterations.
    smacof_max_nodes : int, default=5000
        Node cutoff for dense SMACOF polish.
    overlap_padding : float, default=2.0
        Padding used by overlap projection.
    overlap_iterations : int, default=10
        Number of final overlap projection passes.
    target_aspect : float, optional
        Optional target aspect forwarded to ``AspectRatioFit``.
    seed : int, default=42
        Deterministic seed.
    """

    steps: int = 0
    late_steps: int = 0
    n_pivots: int = 0
    eps: float = _DEFAULT_EPS
    max_exact_nodes: int = _DEFAULT_MAX_EXACT_NODES
    sample_size: Union[int, str] = "auto"
    size_aware: bool = True
    size_scale: float = 1.0
    repulsion_mode: str = "none"
    smacof_iters: int = 4
    smacof_max_nodes: int = _DEFAULT_SMACOF_MAX_NODES
    overlap_padding: float = 2.0
    overlap_iterations: int = 10
    target_aspect: Optional[float] = None
    seed: int = 42


def _resolve_native_stress_config(
    num_nodes: int,
    config: Optional[NativeStressConfig] = None,
) -> NativeStressConfig:
    """Resolve automatic native-stress budgets.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    config : NativeStressConfig, optional
        User-supplied configuration.

    Returns
    -------
    NativeStressConfig
        Frozen configuration with automatic budgets materialized.
    """
    base = config or NativeStressConfig()
    if base.repulsion_mode not in {"none", "fa2_degree", "linlog"}:
        raise ValueError("repulsion_mode must be 'none', 'fa2_degree', or 'linlog'.")
    if base.sample_size != "auto" and (
        not isinstance(base.sample_size, int) or base.sample_size <= 0
    ):
        raise ValueError("sample_size must be a positive integer or 'auto'.")
    if base.eps <= 0.0:
        raise ValueError("eps must be positive.")
    if base.size_scale < 0.0:
        raise ValueError("size_scale must be nonnegative.")

    safe_nodes = max(int(num_nodes), 1)
    steps = base.steps if base.steps > 0 else int(min(180, max(40, 20 + 8 * math.sqrt(safe_nodes))))
    late_steps = (
        base.late_steps if base.late_steps > 0 else int(min(60, max(8, round(float(steps) * 0.18))))
    )
    n_pivots = (
        base.n_pivots
        if base.n_pivots > 0
        else int(min(safe_nodes, max(8, math.sqrt(safe_nodes) * 4)))
    )
    return NativeStressConfig(
        steps=steps,
        late_steps=late_steps,
        n_pivots=n_pivots,
        eps=base.eps,
        max_exact_nodes=base.max_exact_nodes,
        sample_size=base.sample_size,
        size_aware=base.size_aware,
        size_scale=base.size_scale,
        repulsion_mode=base.repulsion_mode,
        smacof_iters=base.smacof_iters,
        smacof_max_nodes=base.smacof_max_nodes,
        overlap_padding=base.overlap_padding,
        overlap_iterations=base.overlap_iterations,
        target_aspect=base.target_aspect,
        seed=base.seed,
    )


def build_native_stress_pipeline(
    config: Union[NativeStressConfig, LayoutConfig, None] = None,
) -> Pipeline:
    """Build the native stress pipeline from registered ops.

    Parameters
    ----------
    config : NativeStressConfig or LayoutConfig or None, optional
        Resolved native-stress configuration, public layout configuration, or
        ``None`` for automatic defaults.

    Returns
    -------
    Pipeline
        Pipeline composed as Pivot-MDS initialization, Stress-SGD core,
        short late multicriteria refinement, optional SMACOF polish, and
        final overlap/aspect postprocessing.
    """
    resolved = _config_from_public(
        num_nodes=int(getattr(config, "_dagua_native_num_nodes", 0)),
        config=config,
        seed=int(getattr(config, "seed", 42) if config is not None else 42),
        target_aspect=getattr(config, "_dagua_native_target_aspect", None),
    )
    criteria_schedules = {
        "stress": SmoothSteps([0, resolved.late_steps], [0.35, 0.08]),
        "vertex_resolution": SmoothSteps([0, resolved.late_steps], [0.0, 0.20]),
        "crossings": SmoothSteps([0, resolved.late_steps], [0.0, 0.10]),
    }
    if resolved.repulsion_mode != "none":
        criteria_schedules["neighborhood_preservation"] = SmoothSteps(
            [0, resolved.late_steps],
            [0.0, 0.06 if resolved.repulsion_mode == "fa2_degree" else 0.10],
        )

    return Pipeline(
        [
            BuildAdjacency(
                BuildAdjacencyConfig(weighted=True, dedup="min", format="list", directed=False)
            ),
            PivotSelection(PivotSelectionConfig(n_pivots=resolved.n_pivots)),
            PivotDistanceQueries(),
            InflateStressTargetDistances(
                InflateStressTargetDistancesConfig(
                    enabled=resolved.size_aware,
                    scale=resolved.size_scale,
                )
            ),
            PivotMDSComputeCoordinates(),
            PivotMDSFinalizePositions(),
            InitializeStressSGDState(independent_shuffle_rng=True),
            PrepareStressSGDTerms(max_exact_nodes=resolved.max_exact_nodes),
            InflateStressTargetDistances(
                InflateStressTargetDistancesConfig(
                    enabled=resolved.size_aware,
                    scale=resolved.size_scale,
                )
            ),
            RunStressSGDExactSchedule(steps=resolved.steps, eps=resolved.eps),
            RunWarmStartStressSGDApproximateSchedule(
                RunWarmStartStressSGDApproximateScheduleConfig(
                    steps=resolved.steps,
                    eps=resolved.eps,
                    sample_size=resolved.sample_size,
                )
            ),
            ResetConvergence(),
            _InitSGD2MultiState(
                steps=resolved.late_steps,
                lr=0.08,
                momentum=0.6,
                grad_clamp=3.0,
                batch_size=128,
                criteria_schedules=criteria_schedules,
            ),
            _RunSGD2MultiOptimization(
                steps=resolved.late_steps,
                lr=0.08,
                momentum=0.6,
                grad_clamp=3.0,
                batch_size=128,
                scheduler_patience=max(resolved.late_steps + 1, 1),
            ),
            ResetConvergence(),
            Conditional(
                predicate=lambda problem, state, ctx: (
                    resolved.smacof_iters > 0 and problem.num_nodes <= resolved.smacof_max_nodes
                ),
                op=Pipeline(
                    [
                        FixedSteps(FixedStepsConfig(n=resolved.smacof_iters)),
                        PrepareWarmStartStressMajorization(
                            PrepareWarmStartStressMajorizationConfig(
                                size_aware=resolved.size_aware,
                                size_scale=resolved.size_scale,
                            )
                        ),
                        Repeat(
                            n=resolved.smacof_iters,
                            ops=[
                                CaptureStressMajorizationStress(),
                                SmacofStep(),
                                CollectStressMajorizationTrace(),
                                CheckStressMajorizationEpsilon(epsilon=1.0e-5),
                            ],
                        ),
                        FinalizeStressMajorizationPositions(),
                    ],
                    name="native_stress_smacof_polish",
                ),
            ),
            OverlapProjection(
                OverlapProjectionConfig(
                    padding=resolved.overlap_padding,
                    iterations=resolved.overlap_iterations,
                )
            ),
            AspectRatioFit(AspectRatioFitConfig(target_aspect=resolved.target_aspect)),
        ],
        name="native_stress_pipeline",
    )


def _layout_connected_native_stress(
    problem: LayoutProblem,
    state: SolveState,
    ctx: RuntimeContext,
    config: NativeStressConfig,
) -> torch.Tensor:
    """Run native stress on one connected component.

    Parameters
    ----------
    problem : LayoutProblem
        Connected layout problem.
    state : SolveState
        Initial solve state, optionally carrying warm-start coordinates.
    ctx : RuntimeContext
        Runtime execution context.
    config : NativeStressConfig
        Resolved pipeline configuration.

    Returns
    -------
    torch.Tensor
        Component position tensor with shape ``[N, 2]``.
    """
    final_state = build_native_stress_pipeline(config).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("native_stress pipeline did not produce final positions.")
    return final_state.pos.detach()


def layout_native_stress_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    config: Optional[Union[NativeStressConfig, LayoutConfig]] = None,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    target_aspect: Optional[float] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Run the r79 native stress pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``.
    config : NativeStressConfig or LayoutConfig, optional
        Native-stress or public layout configuration.
    seed : int, default=42
        Deterministic seed.
    edge_weights : torch.Tensor, optional
        Optional per-edge distance costs with shape ``[E]``.
    target_aspect : float, optional
        Optional output aspect ratio.
    **kwargs : Any
        Compatibility keywords accepted by generic dispatchers.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.
    """
    del kwargs
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if edge_weights is not None and edge_weights.shape[0] != edge_index.shape[1]:
        raise ValueError("edge_weights length must match edge count.")

    output_device = edge_index.device if edge_index.numel() > 0 else torch.device("cpu")
    if node_sizes is not None:
        output_device = node_sizes.device
    if node_sizes is None:
        resolved_node_sizes = torch.ones((num_nodes, 2), dtype=torch.float32, device=output_device)
    else:
        resolved_node_sizes = normalize_node_sizes(node_sizes=node_sizes, device=output_device)

    base_config = _config_from_public(
        num_nodes=num_nodes,
        config=config,
        seed=seed,
        target_aspect=target_aspect,
    )
    problem = LayoutProblem(
        edge_index=edge_index.to(device=output_device, dtype=torch.long),
        num_nodes=num_nodes,
        node_sizes=resolved_node_sizes,
        edge_weights=None if edge_weights is None else edge_weights.to(device=output_device),
        seed=base_config.seed,
    )
    ctx = RuntimeContext(plan=ExecutionPlan(device=str(output_device)))
    component_state = DetectComponents().apply(problem, SolveState(), ctx)
    component_ids = component_state.component_ids
    if component_ids is not None and int(torch.unique(component_ids).numel()) > 1:
        return _layout_components(
            problem=problem,
            ctx=ctx,
            config=base_config,
            component_ids=component_ids,
        )
    return _layout_connected_native_stress(problem, SolveState(), ctx, base_config)


def _config_from_public(
    num_nodes: int,
    config: Optional[Union[NativeStressConfig, LayoutConfig]],
    seed: int,
    target_aspect: Optional[float],
) -> NativeStressConfig:
    """Build native-stress config from public or pipeline configuration.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    config : NativeStressConfig or LayoutConfig, optional
        User-supplied configuration.
    seed : int
        Fallback seed.
    target_aspect : float, optional
        Fallback target aspect ratio.

    Returns
    -------
    NativeStressConfig
        Resolved native-stress configuration.
    """
    if isinstance(config, NativeStressConfig):
        return _resolve_native_stress_config(num_nodes=num_nodes, config=config)
    params = getattr(config, "algorithm_params", {}) if config is not None else {}
    public_steps = int(getattr(config, "steps", 0)) if config is not None else 0
    public_seed = getattr(config, "seed", seed) if config is not None else seed
    public_target_aspect = getattr(config, "_dagua_native_target_aspect", target_aspect)
    raw_config = NativeStressConfig(
        steps=int(params.get("steps", public_steps)),
        late_steps=int(params.get("late_steps", 0)),
        n_pivots=int(params.get("n_pivots", 0)),
        eps=float(params.get("eps", _DEFAULT_EPS)),
        max_exact_nodes=int(params.get("max_exact_nodes", _DEFAULT_MAX_EXACT_NODES)),
        sample_size=params.get("sample_size", "auto"),
        size_aware=bool(params.get("size_aware", True)),
        size_scale=float(params.get("size_scale", 1.0)),
        repulsion_mode=str(params.get("repulsion_mode", "none")),
        smacof_iters=int(params.get("smacof_iters", 4)),
        smacof_max_nodes=int(params.get("smacof_max_nodes", _DEFAULT_SMACOF_MAX_NODES)),
        overlap_padding=float(params.get("overlap_padding", 2.0)),
        overlap_iterations=int(params.get("overlap_iterations", 10)),
        seed=int(public_seed if public_seed is not None else 42),
        target_aspect=public_target_aspect,
    )
    return _resolve_native_stress_config(num_nodes=num_nodes, config=raw_config)


def _layout_components(
    problem: LayoutProblem,
    ctx: RuntimeContext,
    config: NativeStressConfig,
    component_ids: torch.Tensor,
) -> torch.Tensor:
    """Lay out disconnected components independently and tile them.

    Parameters
    ----------
    problem : LayoutProblem
        Parent layout problem.
    ctx : RuntimeContext
        Runtime execution context.
    config : NativeStressConfig
        Resolved native-stress configuration.
    component_ids : torch.Tensor
        Component label tensor with shape ``[N]``.

    Returns
    -------
    torch.Tensor
        Tiled position tensor with shape ``[N, 2]``.
    """
    component_results: list[tuple[torch.Tensor, torch.Tensor]] = []
    for component_id in torch.unique(component_ids, sorted=True).tolist():
        component_nodes = torch.nonzero(component_ids == component_id, as_tuple=False).squeeze(1)
        child_problem, child_state, parent_indices, _ = _extract_component_problem(
            problem,
            SolveState(),
            component_nodes,
            layer_assignments=None,
        )
        child_config = copy.copy(config)
        child_config = _resolve_native_stress_config(
            num_nodes=child_problem.num_nodes,
            config=child_config,
        )
        if child_problem.num_nodes <= 1:
            child_pos = torch.zeros(
                (child_problem.num_nodes, 2),
                dtype=torch.float32,
                device=problem.edge_index.device,
            )
        else:
            child_pos = _layout_connected_native_stress(
                child_problem,
                child_state,
                ctx,
                child_config,
            )
        component_results.append((parent_indices, child_pos))

    tiled = _tile_component_positions(component_results, node_sep=70.0)
    outer_state = AspectRatioFit(AspectRatioFitConfig(target_aspect=config.target_aspect)).apply(
        problem,
        SolveState(pos=tiled),
        ctx,
    )
    if outer_state.pos is None:
        raise RuntimeError("native_stress component tiling did not produce positions.")
    return outer_state.pos.detach()


__all__ = [
    "NativeStressConfig",
    "build_native_stress_pipeline",
    "layout_native_stress_pipeline",
]
