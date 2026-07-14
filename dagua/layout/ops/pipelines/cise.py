"""Cytoscape CiSE circular-cluster layout pipeline."""

from __future__ import annotations

from typing import Any, Optional

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.cytoscape import CytoscapeCircleClusters, CytoscapeFinalize
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def build_cise_pipeline(node_separation: float = 60.0) -> Pipeline:
    """Build the Cytoscape CiSE-style circular-cluster pipeline.

    Parameters
    ----------
    node_separation : float, default=60.0
        Separation used for member circles and cluster spacing.

    Returns
    -------
    Pipeline
        Composable CiSE pipeline.
    """
    return Pipeline(
        [CytoscapeCircleClusters(node_separation=node_separation), CytoscapeFinalize()],
        name="cise_pipeline",
    )


def layout_cise_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 2500,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    clusters: Optional[dict[str, Any]] = None,
    cluster_parents: Optional[dict[str, Optional[str]]] = None,
    nodeSeparation: float = 60.0,
    randomize: bool = False,
    gravity: float = 0.25,
    gravityRange: float = 3.8,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run the Cytoscape CiSE-style pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor | None, optional
        Node-size tensor with shape ``[N, 2]``.
    steps : int, default=2500
        Accepted for API consistency with Cytoscape CiSE.
    seed : int, default=42
        Accepted for API consistency.
    edge_weights : torch.Tensor | None, optional
        Accepted for API consistency.
    clusters : dict[str, Any] | None, optional
        Cluster membership mapping.
    cluster_parents : dict[str, str | None] | None, optional
        Cluster parent mapping.
    nodeSeparation : float, default=60.0
        Separation used for circular clusters.
    randomize : bool, default=False
        Accepted for API consistency.
    gravity : float, default=0.25
        Accepted for API consistency.
    gravityRange : float, default=3.8
        Accepted for API consistency.
    fidelity_dtype : torch.dtype | None, optional
        Optional output dtype override.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    del steps, seed, edge_weights, cluster_parents, randomize, gravity, gravityRange
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        clusters=clusters,
    )
    state = build_cise_pipeline(node_separation=nodeSeparation).apply(
        problem,
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device="cpu")),
    )
    if state.pos is None:
        raise RuntimeError("CiSE pipeline did not produce positions.")
    if fidelity_dtype is not None:
        return state.pos.to(dtype=fidelity_dtype)
    return state.pos


__all__ = ["build_cise_pipeline", "layout_cise_pipeline"]
