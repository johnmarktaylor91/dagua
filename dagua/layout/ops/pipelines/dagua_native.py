"""Composable pipeline for dagua's native tensor layout engine."""

from __future__ import annotations

import copy
from typing import Any, List, Optional

import torch

from dagua.config import LayoutConfig
from dagua.layout.graph_classify import GraphFamily, GraphStructure, classify_graph
from dagua.layout.ops.anneal import (
    InitAnnealingSchedule,
    InitAnnealingScheduleConfig,
    WeightAnnealing,
)
from dagua.layout.ops.base import EarlyBreak, LossGroup, LossOp, Pipeline, Repeat
from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig, StallCount, StallCountConfig
from dagua.layout.ops.init import NativeEngineInit, NativeEngineInitConfig
from dagua.layout.ops.loss_engine import (
    AlignmentLoss,
    BackEdgeCompactnessLoss,
    ClusterCompactnessLoss,
    ClusterContainmentLoss,
    ClusterSeparationLoss,
    CrossingLoss,
    CrossingLossConfig,
    DagOrderingLoss,
    DagOrderingLossConfig,
    EdgeAttractionLoss,
    EdgeAttractionLossConfig,
    EdgeLengthVarianceLoss,
    EdgeStraightnessLoss,
    FanoutDistributionLoss,
    FlexSpacingLoss,
    OverlapAvoidanceLoss,
    OverlapAvoidanceLossConfig,
    PositionPinLoss,
    RepulsionLoss,
    RepulsionLossConfig,
    SpacingConsistencyLoss,
    SpacingConsistencyLossConfig,
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
    FlexConstraints,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)


def _normalize_node_sizes(node_sizes: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Broadcast node sizes to ``[N, 2]`` on the requested device.

    Parameters
    ----------
    node_sizes : torch.Tensor
        Input size tensor shaped ``[N]``, ``[N, 1]``, or ``[N, 2]``.
    device : torch.device
        Target device for the normalized tensor.

    Returns
    -------
    torch.Tensor
        Float32 size tensor with shape ``[N, 2]``.
    """
    if node_sizes.ndim == 1:
        normalized = node_sizes.unsqueeze(1).expand(-1, 2).contiguous()
    elif node_sizes.ndim == 2 and node_sizes.shape[1] == 1:
        normalized = node_sizes.expand(-1, 2).contiguous()
    else:
        normalized = node_sizes.contiguous()
    return normalized.to(device=device, dtype=torch.float32)


def _final_projection_iterations(num_nodes: int) -> int:
    """Return the native engine's final overlap-projection iteration count.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    int
        Final overlap-projection iteration count.
    """
    if num_nodes <= 50:
        return 5
    if num_nodes <= 200:
        return 10
    if num_nodes <= 500_000:
        return 20
    if num_nodes <= 5_000_000:
        return 10
    return 3


def _stall_config(num_nodes: int) -> tuple[int, float]:
    """Return the native engine's early-stop stall parameters.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    tuple[int, float]
        Stall-count limit and relative-loss threshold.
    """
    rel_threshold = 5.0e-4 if num_nodes <= 200 else 1.0e-4
    stall_limit = 3 if num_nodes <= 200 else 5
    if num_nodes <= 5_000:
        stall_limit = max(stall_limit - 2, 3)
    return stall_limit, rel_threshold


def _build_flex_constraints(
    config: LayoutConfig,
    num_nodes: int,
    device: torch.device,
) -> Optional[FlexConstraints]:
    """Convert ``LayoutConfig.flex`` into pipeline-ready tensor constraints.

    Parameters
    ----------
    config : LayoutConfig
        Layout configuration that may carry flex constraints.
    num_nodes : int
        Number of graph nodes.
    device : torch.device
        Device for prepared constraint tensors.

    Returns
    -------
    FlexConstraints or None
        Prepared flex constraints, or ``None`` when flex is disabled.
    """
    if config.flex is None:
        return None

    from dagua.layout.engine import _prepare_flex_data

    flex_data = _prepare_flex_data(config=config, num_nodes=num_nodes, device=str(device))
    return FlexConstraints(
        pin_indices=flex_data["pin_indices"],
        pin_targets=flex_data["pin_targets"],
        pin_weights=flex_data["pin_weights"],
        soft_pin_mask=flex_data["soft_pin_mask"],
        hard_pin_mask=flex_data["hard_pin_mask"],
        align_groups=flex_data["align_groups"],
        flex_node_sep=flex_data["flex_node_sep"],
        flex_node_sep_weight=flex_data["flex_node_sep_weight"],
    )


def _prepare_pipeline_config(
    config: LayoutConfig,
    num_nodes: int,
    edge_index: torch.Tensor,
    device: str,
    layer_assignments: Optional[torch.Tensor],
    prebuilt_layer_index: Optional[Any],
    graph_structure: Optional[GraphStructure],
    skip_classification: bool,
) -> LayoutConfig:
    """Resolve native-engine pipeline settings for one problem instance.

    Parameters
    ----------
    config : LayoutConfig
        Base layout configuration.
    num_nodes : int
        Number of graph nodes.
    edge_index : torch.Tensor
        Edge tensor shaped ``[2, E]``.
    device : str
        Requested execution device.
    layer_assignments : torch.Tensor or None
        Optional precomputed layer assignments.
    prebuilt_layer_index : Any or None
        Optional precomputed layer index reused by the pipeline.
    graph_structure : GraphStructure or None
        Optional structural classification supplied by the caller.
    skip_classification : bool
        Whether to bypass graph-family classification.

    Returns
    -------
    LayoutConfig
        Shallow config copy annotated with resolved private pipeline metadata.
    """
    from dagua.layout.engine import (
        _adaptive_spacing,
        _auto_layout_steps,
        _overlap_interval,
        _override_for_tree,
    )

    effective_config = copy.copy(config)
    if not skip_classification:
        structure = graph_structure
        if structure is None:
            classification_layers = layer_assignments
            structure = classify_graph(
                edge_index,
                num_nodes,
                layer_assignments=classification_layers,
            )
        if structure.family in {GraphFamily.TREE, GraphFamily.CHAIN}:
            effective_config = _override_for_tree(effective_config)
        if structure.family == GraphFamily.CHAIN:
            auto_steps = _auto_layout_steps(num_nodes)
            resolved_steps = min(
                effective_config.steps if effective_config.steps > 0 else auto_steps,
                50,
            )
        else:
            resolved_steps = (
                effective_config.steps
                if effective_config.steps > 0
                else _auto_layout_steps(num_nodes)
            )
    else:
        resolved_steps = (
            effective_config.steps if effective_config.steps > 0 else _auto_layout_steps(num_nodes)
        )

    resolved_node_sep = effective_config.node_sep
    resolved_rank_sep = effective_config.rank_sep
    if effective_config.adaptive_spacing:
        resolved_node_sep, resolved_rank_sep = _adaptive_spacing(
            num_nodes=num_nodes,
            base_node_sep=resolved_node_sep,
            base_rank_sep=resolved_rank_sep,
        )

    stall_limit, rel_threshold = _stall_config(num_nodes=num_nodes)
    setattr(effective_config, "_dagua_native_steps", resolved_steps)
    setattr(effective_config, "_dagua_native_node_sep", resolved_node_sep)
    setattr(effective_config, "_dagua_native_rank_sep", resolved_rank_sep)
    setattr(effective_config, "_dagua_native_device", device)
    setattr(effective_config, "_dagua_native_verbose", effective_config.verbose)
    setattr(effective_config, "_dagua_native_layer_assignments", layer_assignments)
    setattr(effective_config, "_dagua_native_prebuilt_layer_index", prebuilt_layer_index)
    setattr(
        effective_config,
        "_dagua_native_overlap_interval",
        _overlap_interval(num_nodes=num_nodes, config=effective_config),
    )
    setattr(
        effective_config,
        "_dagua_native_final_projection_iterations",
        _final_projection_iterations(num_nodes=num_nodes),
    )
    setattr(effective_config, "_dagua_native_stall_limit", stall_limit)
    setattr(effective_config, "_dagua_native_rel_threshold", rel_threshold)
    setattr(effective_config, "_dagua_native_crossing_alpha", 3.0)
    setattr(effective_config, "_dagua_native_optimizer_type", "adam")
    return effective_config


def _build_loss_ops(config: LayoutConfig, node_sep: float, rank_sep: float) -> List[LossOp]:
    """Construct the active native-engine loss operators.

    Parameters
    ----------
    config : LayoutConfig
        Resolved native layout configuration.
    node_sep : float
        Same-layer gap target after adaptive spacing.

    Returns
    -------
    list[LossOp]
        Loss operators enabled by the configuration.
    """
    losses: List[LossOp] = []
    if config.w_dag > 0.0:
        losses.append(DagOrderingLoss(DagOrderingLossConfig(rank_sep=rank_sep)))
    if config.w_attract > 0.0:
        losses.append(EdgeAttractionLoss(EdgeAttractionLossConfig(x_bias=config.w_attract_x_bias)))
    if config.w_repel > 0.0:
        losses.append(
            RepulsionLoss(
                RepulsionLossConfig(
                    threshold=config.exact_repulsion_threshold,
                    sample_k=config.negative_sample_k,
                    rvs_threshold=config.rvs_threshold,
                    rvs_nn_k=config.rvs_nn_k,
                ),
            ),
        )
    if config.w_overlap > 0.0:
        losses.append(
            OverlapAvoidanceLoss(
                OverlapAvoidanceLossConfig(
                    padding=2.0,
                    rvs_threshold=config.rvs_threshold,
                ),
            ),
        )
    if config.w_cluster > 0.0:
        losses.append(ClusterCompactnessLoss())
        losses.append(ClusterSeparationLoss())
    if config.w_cluster_contain > 0.0:
        losses.append(ClusterContainmentLoss())
    if config.w_crossing > 0.0:
        losses.append(
            CrossingLoss(
                CrossingLossConfig(
                    alpha=float(getattr(config, "_dagua_native_crossing_alpha", 3.0)),
                    max_pairs=500,
                ),
            ),
        )
    if config.w_straightness > 0.0:
        losses.append(EdgeStraightnessLoss())
    if config.w_length_variance > 0.0:
        losses.append(EdgeLengthVarianceLoss())
    if config.w_spacing > 0.0:
        losses.append(SpacingConsistencyLoss(SpacingConsistencyLossConfig(target_gap=node_sep)))
    if config.w_fanout > 0.0:
        losses.append(FanoutDistributionLoss())
    if config.w_back_edge > 0.0:
        losses.append(BackEdgeCompactnessLoss())
    if config.flex is not None and config.flex.pins:
        losses.append(PositionPinLoss())
    if config.flex is not None and (config.flex.align_x or config.flex.align_y):
        losses.append(AlignmentLoss())
    if config.flex is not None and config.flex.node_sep is not None:
        losses.append(FlexSpacingLoss())
    return losses


def build_dagua_pipeline(config: LayoutConfig) -> Pipeline:
    """Build the composable native-engine pipeline.

    Parameters
    ----------
    config : LayoutConfig
        Layout configuration annotated with any resolved private pipeline
        metadata required for problem-specific defaults.

    Returns
    -------
    Pipeline
        Native-engine pipeline composed from registered ops.
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
    losses = _build_loss_ops(
        config=config,
        node_sep=resolved_node_sep,
        rank_sep=resolved_rank_sep,
    )

    return Pipeline(
        [
            FixedSteps(FixedStepsConfig(n=resolved_steps)),
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
            Repeat(
                n=resolved_steps,
                ops=[
                    WeightAnnealing(),
                    OptimizerZeroGrad(),
                    LossGroup(losses=losses, backward_mode="combined"),
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

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor
        Node-size tensor shaped ``[N]``, ``[N, 1]``, or ``[N, 2]``.
    config : LayoutConfig or None, default=None
        Layout configuration. Uses ``LayoutConfig()`` when omitted.
    device : str or None, default=None
        Explicit execution device override.
    optimizer_type : str, default="adam"
        Optimizer name forwarded into the runtime context.
    init_pos : torch.Tensor or None, default=None
        Optional warm-start positions shaped ``[N, 2]``.
    clusters : dict[str, Any] or None, default=None
        Optional cluster membership mapping.
    cluster_parents : dict[str, str] or None, default=None
        Optional cluster parent mapping.
    layer_assignments : torch.Tensor or None, default=None
        Optional precomputed layer assignments shaped ``[N]``.
    prebuilt_layer_index : Any or None, default=None
        Optional precomputed layer index.
    graph_structure : GraphStructure or None, default=None
        Optional precomputed graph classification.
    skip_classification : bool, default=False
        Whether to bypass graph-family classification.
    seed : int or None, default=None
        Random seed forwarded to the layout problem.
    edge_weights : torch.Tensor or None, default=None
        Optional edge weights shaped ``[E]``.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If ``num_nodes`` is negative.
    RuntimeError
        If the pipeline does not populate final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")

    effective_config = copy.copy(config) if config is not None else LayoutConfig()
    requested_device = device or effective_config.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        requested_device = "cpu"
    target_device = torch.device(requested_device)
    normalized_node_sizes = _normalize_node_sizes(node_sizes=node_sizes, device=target_device)
    prepared_edge_index = edge_index.to(device=target_device, dtype=torch.long)
    prepared_init_pos = (
        init_pos.to(device=target_device, dtype=torch.float32) if init_pos is not None else None
    )
    resolved_seed = seed if seed is not None else effective_config.seed
    if resolved_seed is not None:
        torch.manual_seed(int(resolved_seed))
        if target_device.type == "cuda":
            torch.cuda.manual_seed(int(resolved_seed))

    prepared_config = _prepare_pipeline_config(
        config=effective_config,
        num_nodes=num_nodes,
        edge_index=prepared_edge_index,
        device=str(target_device),
        layer_assignments=layer_assignments,
        prebuilt_layer_index=prebuilt_layer_index,
        graph_structure=graph_structure,
        skip_classification=skip_classification,
    )
    flex_constraints = _build_flex_constraints(
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


__all__ = ["build_dagua_pipeline", "layout_dagua_native_pipeline"]
