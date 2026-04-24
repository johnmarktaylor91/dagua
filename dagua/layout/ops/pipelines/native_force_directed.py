"""Force-directed native sub-pipeline for flat or heavily cyclic graphs."""

from __future__ import annotations

from typing import Any, Optional

import torch

from dagua.config import LayoutConfig
from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.pipelines.dagua_flat import (
    build_dagua_flat_pipeline,
    layout_dagua_flat_pipeline,
)


def build_native_force_directed_pipeline(config: LayoutConfig) -> Pipeline:
    """Build the Pivot-MDS plus Stress-SGD native force-directed pipeline.

    Parameters
    ----------
    config : LayoutConfig
        Layout configuration. ``steps`` is interpreted as the Stress-SGD epoch
        count when positive; otherwise the flat pipeline default is used.

    Returns
    -------
    Pipeline
        Pipeline composed as Pivot-MDS init, Stress-SGD refinement, overlap
        projection, and aspect-ratio fit.
    """
    steps = int(config.steps) if int(getattr(config, "steps", 0)) > 0 else 30
    return build_dagua_flat_pipeline(
        steps=steps,
        n_pivots=int(getattr(config, "w_stress_n_pivots", 50)),
        target_aspect=1.0,
    )


def layout_native_force_directed_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    config: Optional[LayoutConfig] = None,
    seed: Optional[int] = None,
    edge_weights: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Run the native force-directed sub-pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node sizes with shape ``[N, 2]``.
    config : LayoutConfig, optional
        Layout configuration.
    seed : int, optional
        RNG seed.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.
    **kwargs : Any
        Ignored compatibility keywords from generic dispatchers.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    del kwargs

    effective_config = config if config is not None else LayoutConfig()
    resolved_seed = seed if seed is not None else effective_config.seed
    return layout_dagua_flat_pipeline(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        steps=int(effective_config.steps) if effective_config.steps > 0 else 30,
        n_pivots=int(getattr(effective_config, "w_stress_n_pivots", 50)),
        seed=int(resolved_seed if resolved_seed is not None else 42),
        edge_weights=edge_weights,
        target_aspect=1.0,
    )


__all__ = ["build_native_force_directed_pipeline", "layout_native_force_directed_pipeline"]
