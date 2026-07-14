"""Graphviz fdp-compatible public layout pipeline."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch

from dagua.layout.ops.pipelines.fmmm import layout_fmmm_pipeline

_GRAPHVIZ_FDP_DEFAULT_STEPS = 200


def layout_fdp_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 200,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    fidelity_dtype: torch.dtype = torch.float32,
    clusters: Optional[Mapping[str, Sequence[int]]] = None,
    cluster_parents: Optional[Mapping[str, Optional[str]]] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Run the existing Graphviz FDP fidelity path as a first-class algorithm.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]`` in point units.
    steps : int, default=200
        Graphviz FDP ``maxiter`` compatibility budget. Values less than one
        select the public Graphviz-compatible default.
    seed : int, default=42
        Graphviz-compatible start seed.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.
    fidelity_dtype : torch.dtype, default=torch.float32
        Internal fidelity dtype forwarded for dispatcher compatibility.
    clusters : Mapping[str, Sequence[int]], optional
        Cluster membership metadata forwarded to the cluster-aware FDP branch.
    cluster_parents : Mapping[str, str | None], optional
        Cluster parent metadata forwarded to the cluster-aware FDP branch.
    **kwargs : Any
        Additional generic dispatcher keywords accepted for compatibility.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.
    """
    effective_steps = _GRAPHVIZ_FDP_DEFAULT_STEPS if steps <= 0 else steps
    return layout_fmmm_pipeline(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        steps=effective_steps,
        seed=seed,
        edge_weights=edge_weights,
        fidelity_mode="graphviz_fdp",
        fidelity_dtype=fidelity_dtype,
        clusters=clusters,
        cluster_parents=cluster_parents,
        **kwargs,
    )


__all__ = ["layout_fdp_pipeline"]
