"""Tree native sub-pipeline based on Reingold-Tilford placement."""

from __future__ import annotations

from typing import Any, Optional

import torch

from dagua.config import LayoutConfig
from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.coordinate import ReingoldTilfordTree, ReingoldTilfordTreeConfig
from dagua.layout.ops.postprocess import AspectRatioFit, AspectRatioFitConfig
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def build_native_tree_pipeline(config: LayoutConfig) -> Pipeline:
    """Build the native tree fast-path pipeline.

    Parameters
    ----------
    config : LayoutConfig
        Prepared native layout configuration.

    Returns
    -------
    Pipeline
        Reingold-Tilford tree placement followed by aspect-ratio fitting.
    """
    return Pipeline(
        [
            ReingoldTilfordTree(ReingoldTilfordTreeConfig()),
            AspectRatioFit(
                AspectRatioFitConfig(
                    target_aspect=getattr(config, "_dagua_native_target_aspect", None),
                )
            ),
        ],
        name="native_tree_pipeline",
    )


def layout_native_tree_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    config: Optional[LayoutConfig] = None,
    seed: Optional[int] = None,
    edge_weights: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Run the native tree sub-pipeline through the public registry.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node sizes with shape ``[N, 2]``.
    config : LayoutConfig, optional
        User-facing layout configuration.
    seed : int, optional
        RNG seed.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.
    **kwargs : Any
        Ignored compatibility keywords from generic dispatchers.

    Returns
    -------
    torch.Tensor
        Detached position tensor with shape ``[N, 2]``.
    """
    del kwargs

    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    effective_config = config if config is not None else LayoutConfig()
    resolved_seed = seed if seed is not None else effective_config.seed
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=int(resolved_seed if resolved_seed is not None else 42),
    )
    output_device = (
        edge_index.device
        if edge_index.numel() > 0
        else node_sizes.device
        if node_sizes is not None
        else torch.device("cpu")
    )
    final_state = build_native_tree_pipeline(effective_config).apply(
        problem,
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device=str(output_device))),
    )
    if final_state.pos is None:
        raise RuntimeError("native_tree pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_native_tree_pipeline", "layout_native_tree_pipeline"]
