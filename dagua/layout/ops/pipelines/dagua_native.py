"""Composable pipeline for dagua's native tensor layout engine.

Config-time helpers live in ``dagua.layout.resolve``. The pipeline body here
is pure composed ops; no inline helpers, no imports from
``dagua.layout.engine``.
"""

from __future__ import annotations

import copy
from typing import Any, Optional

import torch

from dagua.config import LayoutConfig
from dagua.layout.graph_classify import GraphStructure
from dagua.layout.ops.anneal import (
    InitAnnealingSchedule,
    InitAnnealingScheduleConfig,
    WeightAnnealing,
)
from dagua.layout.ops.base import EarlyBreak, LossGroup, Pipeline, Repeat
from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig, StallCount, StallCountConfig
from dagua.layout.ops.init import (
    NativeEngineInit,
    NativeEngineInitConfig,
)
from dagua.layout.ops.optimize import (
    ClipGradNorm,
    ClipGradNormConfig,
    CreateOptimizer,
    CreateOptimizerConfig,
    OptimizerStep,
    OptimizerZeroGrad,
)
from dagua.layout.ops.project import (
    HardPinProjection,
    OverlapProjection,
    OverlapProjectionConfig,
    PeriodicOverlapProjection,
    PeriodicOverlapProjectionConfig,
)
from dagua.layout.ops.state import (
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)
from dagua.layout.resolve import (
    build_flex_constraints,
    build_loss_ops,
    normalize_node_sizes,
    prepare_pipeline_config,
)


def build_gradient_core(
    losses: list,
    steps: int,
    overlap_interval: int,
    stall_limit: int,
    rel_threshold: float,
) -> Pipeline:
    """Build the inner differentiable optimizer as a named sub-pipeline.

    Sprint 1 exit criterion: "the 'inner differentiable optimizer' is its
    own named sub-pipeline (`GradientCore`), pluggable between initializers."
    Packages the Repeat loop body + PeriodicOverlapProjection +
    StallCount + EarlyBreak as a standalone Pipeline called "gradient_core"
    that composes into the top-level pipeline, letting Sprint 2+ swap
    initializers without touching the optimization loop.

    Parameters
    ----------
    losses : list
        Weighted loss operators to evaluate each step.
    steps : int
        Maximum optimization steps.
    overlap_interval : int
        Periodic overlap projection interval.
    stall_limit : int
        Consecutive-no-improve steps before early break.
    rel_threshold : float
        Relative-loss threshold for stall detection.

    Returns
    -------
    Pipeline
        Named ``gradient_core`` sub-pipeline.
    """
    return Pipeline(
        [
            Repeat(
                n=steps,
                ops=[
                    WeightAnnealing(),
                    OptimizerZeroGrad(),
                    # Sprint 1 memory port: per_loss backward frees each
                    # loss term's autograd graph immediately, cutting peak
                    # RSS 3-4x vs combined.
                    LossGroup(losses=losses, backward_mode="per_loss"),
                    ClipGradNorm(ClipGradNormConfig(max_norm=100.0)),
                    OptimizerStep(),
                    HardPinProjection(),
                    PeriodicOverlapProjection(
                        PeriodicOverlapProjectionConfig(
                            interval=overlap_interval,
                            padding=2.0,
                            iterations=None,
                            run_on_last_step=True,
                        ),
                    ),
                    StallCount(
                        StallCountConfig(
                            limit=stall_limit,
                            rel_threshold=rel_threshold,
                        ),
                    ),
                    EarlyBreak(lambda problem, state, ctx: state.converged),
                ],
            ),
        ],
        name="gradient_core",
    )


def build_dagua_pipeline(config: LayoutConfig) -> Pipeline:
    """Build the composable native-engine pipeline from a resolved config.

    Parameters
    ----------
    config : LayoutConfig
        Layout configuration annotated with resolved private pipeline
        metadata (``_dagua_native_*`` attrs set by
        :func:`dagua.layout.resolve.prepare_pipeline_config`).

    Returns
    -------
    Pipeline
        Native-engine pipeline composed purely from registered ops.
    """
    resolved_steps = int(
        getattr(config, "_dagua_native_steps", config.steps if config.steps > 0 else 0),
    )
    resolved_node_sep = float(getattr(config, "_dagua_native_node_sep", config.node_sep))
    resolved_rank_sep = float(getattr(config, "_dagua_native_rank_sep", config.rank_sep))
    resolved_device = str(getattr(config, "_dagua_native_device", config.device))
    resolved_verbose = bool(getattr(config, "_dagua_native_verbose", config.verbose))
    overlap_interval = int(getattr(config, "_dagua_native_overlap_interval", 5))
    final_projection_iterations = int(
        getattr(config, "_dagua_native_final_projection_iterations", 10),
    )
    stall_limit = int(getattr(config, "_dagua_native_stall_limit", 5))
    rel_threshold = float(getattr(config, "_dagua_native_rel_threshold", 1.0e-4))
    optimizer_type = str(getattr(config, "_dagua_native_optimizer_type", "adam"))
    losses = build_loss_ops(
        config=config,
        node_sep=resolved_node_sep,
        rank_sep=resolved_rank_sep,
    )

    return Pipeline(
        [
            FixedSteps(FixedStepsConfig(n=resolved_steps)),
            # Sprint 1 attempted FamilyConditionalInit with spectral fallback
            # for flat graphs; iteration on the held-out 30-graph suite showed
            # widespread regressions (sparse_layered 64->30, grid_square 89->45,
            # directed_dag_medium 50->29, bipartite 45->34) because the
            # layer-ratio heuristic fires too aggressively. Spectral init
            # works for large undirected graphs but not for the mixed DAG-ish
            # suite we have. Reverted to plain NativeEngineInit; the
            # FamilyConditionalInit op is registered but unused by default.
            # Revisit in Sprint 1 follow-up with a stricter family predicate.
            NativeEngineInit(
                NativeEngineInitConfig(
                    node_sep=resolved_node_sep,
                    rank_sep=resolved_rank_sep,
                    device=resolved_device,
                    verbose=resolved_verbose,
                    layer_assignments=getattr(config, "_dagua_native_layer_assignments", None),
                    prebuilt_layer_index=getattr(
                        config,
                        "_dagua_native_prebuilt_layer_index",
                        None,
                    ),
                ),
            ),
            InitAnnealingSchedule(
                InitAnnealingScheduleConfig(
                    w_dag=config.w_dag,
                    w_attract=config.w_attract,
                    w_repel=config.w_repel,
                    w_overlap=config.w_overlap,
                    w_cluster=config.w_cluster,
                    w_cluster_contain=config.w_cluster_contain,
                    w_crossing=config.w_crossing,
                    w_straightness=config.w_straightness,
                    w_length_variance=config.w_length_variance,
                    w_spacing=config.w_spacing,
                    w_fanout=config.w_fanout,
                    w_back_edge=config.w_back_edge,
                ),
            ),
            CreateOptimizer(
                CreateOptimizerConfig(
                    optimizer_type=optimizer_type,
                    lr=config.lr,
                    target="pos",
                    key="default",
                ),
            ),
            # Sprint 1: named gradient_core sub-pipeline extracted so
            # Sprint 2+ can swap initializers without touching the loop.
            build_gradient_core(
                losses=losses,
                steps=resolved_steps,
                overlap_interval=overlap_interval,
                stall_limit=stall_limit,
                rel_threshold=rel_threshold,
            ),
            OverlapProjection(
                OverlapProjectionConfig(
                    padding=2.0,
                    iterations=final_projection_iterations,
                ),
            ),
        ],
        name="dagua_native_pipeline",
    )


def layout_dagua_native_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: torch.Tensor,
    config: Optional[LayoutConfig] = None,
    device: Optional[str] = None,
    optimizer_type: str = "adam",
    init_pos: Optional[torch.Tensor] = None,
    clusters: Optional[dict[str, Any]] = None,
    cluster_parents: Optional[dict[str, str]] = None,
    layer_assignments: Optional[torch.Tensor] = None,
    prebuilt_layer_index: Optional[Any] = None,
    graph_structure: Optional[GraphStructure] = None,
    skip_classification: bool = False,
    seed: Optional[int] = None,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the native-engine pipeline as a drop-in tensor layout entrypoint.

    Thin adapter: resolves a config via :mod:`dagua.layout.resolve`,
    constructs a ``LayoutProblem``/``SolveState``/``RuntimeContext``, then
    invokes the pipeline built by :func:`build_dagua_pipeline`.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")

    effective_config = copy.copy(config) if config is not None else LayoutConfig()
    requested_device = device or effective_config.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        requested_device = "cpu"
    target_device = torch.device(requested_device)
    normalized_node_sizes = normalize_node_sizes(node_sizes=node_sizes, device=target_device)
    prepared_edge_index = edge_index.to(device=target_device, dtype=torch.long)
    prepared_init_pos = (
        init_pos.to(device=target_device, dtype=torch.float32) if init_pos is not None else None
    )
    resolved_seed = seed if seed is not None else effective_config.seed
    if resolved_seed is not None:
        torch.manual_seed(int(resolved_seed))
        if target_device.type == "cuda":
            torch.cuda.manual_seed(int(resolved_seed))

    prepared_config = prepare_pipeline_config(
        config=effective_config,
        num_nodes=num_nodes,
        edge_index=prepared_edge_index,
        device=str(target_device),
        layer_assignments=layer_assignments,
        prebuilt_layer_index=prebuilt_layer_index,
        graph_structure=graph_structure,
        skip_classification=skip_classification,
    )
    flex_constraints = build_flex_constraints(
        config=prepared_config,
        num_nodes=num_nodes,
        device=target_device,
    )
    setattr(prepared_config, "_dagua_native_optimizer_type", optimizer_type)
    problem = LayoutProblem(
        edge_index=prepared_edge_index,
        num_nodes=num_nodes,
        node_sizes=normalized_node_sizes,
        direction=prepared_config.direction,
        clusters=clusters,
        cluster_parents=cluster_parents,
        flex=flex_constraints,
        edge_weights=edge_weights,
        seed=int(resolved_seed if resolved_seed is not None else 42),
    )
    state = SolveState(pos=prepared_init_pos)
    ctx = RuntimeContext(
        plan=ExecutionPlan(
            device=str(target_device),
            optimizer_type=optimizer_type,
        ),
    )
    final_state = build_dagua_pipeline(prepared_config).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("dagua_native pipeline did not produce final positions.")
    return final_state.pos.detach()


__all__ = ["build_dagua_pipeline", "build_gradient_core", "layout_dagua_native_pipeline"]
