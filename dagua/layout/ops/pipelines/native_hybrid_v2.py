"""SCC-condensation native hybrid-v2 sub-pipeline."""

from __future__ import annotations

import copy
from typing import Any, Optional

import torch

from dagua.config import LayoutConfig
from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.postprocess import AspectRatioFit, AspectRatioFitConfig
from dagua.layout.ops.project import OverlapProjection, OverlapProjectionConfig
from dagua.layout.ops.scc import (
    SCCCondense,
    SCCExpand,
    SCCInternalLayoutConfig,
    SCCLayoutCondensationDAG,
    SCCLayoutInternals,
    SCCMetaLayoutConfig,
)
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.resolve import normalize_node_sizes


def build_native_hybrid_v2_pipeline(config: LayoutConfig) -> Pipeline:
    """Build the SCC-condensation hybrid-v2 pipeline.

    Parameters
    ----------
    config : LayoutConfig
        Prepared native configuration with ``_dagua_native_*`` metadata.

    Returns
    -------
    Pipeline
        Pipeline composed from registered SCC condensation, internal layout,
        meta-DAG layout, expansion, overlap projection, and aspect fitting ops.
    """
    seed_value = getattr(config, "seed", 42)
    seed = int(seed_value if seed_value is not None else 42)
    node_sep = float(getattr(config, "_dagua_native_node_sep", config.node_sep))
    rank_sep = float(getattr(config, "_dagua_native_rank_sep", config.rank_sep))
    steps = int(getattr(config, "_dagua_native_steps", config.steps if config.steps > 0 else 0))
    device = str(getattr(config, "_dagua_native_device", config.device))
    final_projection_iterations = int(
        getattr(config, "_dagua_native_final_projection_iterations", 10),
    )
    return Pipeline(
        [
            SCCCondense(),
            SCCLayoutInternals(
                SCCInternalLayoutConfig(
                    internal_min=int(getattr(config, "hybrid_v2_internal_min", 5)),
                    small_steps=int(getattr(config, "hybrid_v2_small_steps", 24)),
                    large_steps=int(getattr(config, "hybrid_v2_large_steps", 0)),
                    bbox_padding=float(getattr(config, "hybrid_v2_bbox_padding", 24.0)),
                    seed=seed,
                )
            ),
            SCCLayoutCondensationDAG(
                SCCMetaLayoutConfig(
                    node_sep=node_sep,
                    rank_sep=rank_sep,
                    steps=steps,
                    seed=seed,
                    device=device,
                )
            ),
            SCCExpand(),
            OverlapProjection(
                OverlapProjectionConfig(
                    padding=2.0,
                    iterations=final_projection_iterations,
                )
            ),
            AspectRatioFit(
                AspectRatioFitConfig(
                    target_aspect=getattr(config, "_dagua_native_target_aspect", None),
                )
            ),
        ],
        name="native_hybrid_v2_pipeline",
    )


def layout_native_hybrid_v2_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: torch.Tensor,
    config: Optional[LayoutConfig] = None,
    device: Optional[str] = None,
    optimizer_type: str = "adam",
    init_pos: Optional[torch.Tensor] = None,
    clusters: Optional[dict[str, Any]] = None,
    cluster_parents: Optional[dict[str, Optional[str]]] = None,
    layer_assignments: Optional[torch.Tensor] = None,
    prebuilt_layer_index: Optional[Any] = None,
    graph_structure: Optional[Any] = None,
    skip_classification: bool = False,
    seed: Optional[int] = None,
    edge_weights: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Run the SCC-condensation hybrid-v2 pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Directed graph connectivity with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.
    config : LayoutConfig, optional
        User-facing layout configuration.
    device : str, optional
        Target execution device override.
    optimizer_type : str, default="adam"
        Optimizer name stored in the runtime execution plan.
    init_pos : torch.Tensor, optional
        Ignored compatibility warm-start tensor.
    clusters : dict[str, Any], optional
        Ignored compatibility cluster metadata.
    cluster_parents : dict[str, str], optional
        Ignored compatibility nested-cluster metadata.
    layer_assignments : torch.Tensor, optional
        Ignored compatibility layer assignments.
    prebuilt_layer_index : Any, optional
        Ignored compatibility layer index.
    graph_structure : Any, optional
        Optional pre-classified graph metadata.
    skip_classification : bool, default=False
        Ignored compatibility flag.
    seed : int, optional
        Seed override forwarded from the layout dispatcher.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.
    **kwargs : Any
        Ignored compatibility keywords from generic dispatchers.

    Returns
    -------
    torch.Tensor
        Detached position tensor with shape ``[N, 2]``.
    """
    del init_pos, clusters, cluster_parents, layer_assignments, prebuilt_layer_index
    del skip_classification, edge_weights, kwargs

    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    target_device = torch.device(device or getattr(config, "device", "cpu"))
    if target_device.type == "cuda" and not torch.cuda.is_available():
        target_device = torch.device("cpu")
    if num_nodes == 0:
        return torch.zeros((0, 2), dtype=torch.float32, device=target_device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32, device=target_device)

    effective_config = copy.copy(config) if config is not None else LayoutConfig()
    resolved_seed = seed if seed is not None else effective_config.seed
    if resolved_seed is None:
        resolved_seed = 42
    effective_config.seed = int(resolved_seed)
    setattr(effective_config, "_dagua_native_num_nodes", num_nodes)
    setattr(effective_config, "_dagua_native_device", str(target_device))
    setattr(effective_config, "_dagua_native_structure", graph_structure)

    normalized_node_sizes = normalize_node_sizes(node_sizes=node_sizes, device=target_device)
    problem = LayoutProblem(
        edge_index=edge_index.to(device=target_device, dtype=torch.long),
        num_nodes=num_nodes,
        node_sizes=normalized_node_sizes,
        structure=graph_structure,
        seed=int(resolved_seed),
    )
    ctx = RuntimeContext(
        plan=ExecutionPlan(
            device=str(target_device),
            optimizer_type=optimizer_type,
        )
    )
    final_state = build_native_hybrid_v2_pipeline(effective_config).apply(
        problem,
        SolveState(),
        ctx,
    )
    if final_state.pos is None:
        raise RuntimeError("native_hybrid_v2 pipeline did not produce final positions.")
    return final_state.pos.detach()


__all__ = ["build_native_hybrid_v2_pipeline", "layout_native_hybrid_v2_pipeline"]
