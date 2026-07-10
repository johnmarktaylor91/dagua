"""Layered-DAG native sub-pipeline."""

from __future__ import annotations

import copy
from typing import Any, Optional

import torch

from dagua.config import LayoutConfig
from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.pipelines import dagua_native_legacy


def build_native_layered_dag_pipeline(config: LayoutConfig) -> Pipeline:
    """Build the native layered-DAG pipeline.

    Parameters
    ----------
    config : LayoutConfig
        Prepared native configuration with ``_dagua_native_*`` metadata.

    Returns
    -------
    Pipeline
        Pipeline using the sprint-19 layered DAG polish stack.
    """
    layered_config = copy.copy(config)
    layered_config.insert_dummy_nodes = bool(getattr(config, "insert_dummy_nodes", True))
    layered_config.use_native_median_transpose = bool(
        getattr(config, "use_native_median_transpose", True)
    )
    layered_config.brandes_koepf_refine = bool(getattr(config, "brandes_koepf_refine", True))
    return dagua_native_legacy.build_dagua_pipeline(layered_config)


def layout_native_layered_dag_pipeline(
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
    """Run the native layered-DAG sub-pipeline through the public registry.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.
    config : LayoutConfig, optional
        User-facing layout configuration.
    device : str, optional
        Target execution device override.
    optimizer_type : str, default="adam"
        Optimizer implementation used by the native gradient core.
    init_pos : torch.Tensor, optional
        Optional initial positions with shape ``[N, 2]``.
    clusters : dict[str, Any], optional
        Cluster membership metadata.
    cluster_parents : dict[str, str], optional
        Nested-cluster parent metadata.
    layer_assignments : torch.Tensor, optional
        Optional layer assignments with shape ``[N]``.
    prebuilt_layer_index : Any, optional
        Optional pre-built layer index.
    graph_structure : Any, optional
        Optional pre-classified graph metadata.
    skip_classification : bool, default=False
        Whether to skip graph-family classification during config prep.
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
    del kwargs

    effective_config = copy.copy(config) if config is not None else LayoutConfig()
    effective_config.insert_dummy_nodes = bool(
        getattr(effective_config, "insert_dummy_nodes", True)
    )
    effective_config.use_native_median_transpose = bool(
        getattr(effective_config, "use_native_median_transpose", True)
    )
    effective_config.brandes_koepf_refine = bool(
        getattr(effective_config, "brandes_koepf_refine", True)
    )
    return dagua_native_legacy.layout_dagua_native_pipeline(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        config=effective_config,
        device=device,
        optimizer_type=optimizer_type,
        init_pos=init_pos,
        clusters=clusters,
        cluster_parents=cluster_parents,
        layer_assignments=layer_assignments,
        prebuilt_layer_index=prebuilt_layer_index,
        graph_structure=graph_structure,
        skip_classification=skip_classification,
        seed=seed,
        edge_weights=edge_weights,
    )


__all__ = ["build_native_layered_dag_pipeline", "layout_native_layered_dag_pipeline"]
