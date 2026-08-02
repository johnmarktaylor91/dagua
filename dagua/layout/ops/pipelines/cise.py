"""Cytoscape CiSE circular-cluster layout pipeline."""

from __future__ import annotations

from typing import Any, Optional

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.cytoscape import (
    CytoscapeCircleClusters,
    CytoscapeCiSERelax,
    CytoscapeFinalize,
)
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def build_cise_pipeline(
    node_separation: float = 12.5,
    steps: int = 2500,
    gravity: float = 0.25,
    gravity_range: float = 3.8,
    randomize: bool = False,
) -> Pipeline:
    """Build the Cytoscape CiSE-style circular-cluster pipeline.

    Parameters
    ----------
    node_separation : float, default=12.5
        Separation used for member circles and cluster spacing.
    steps : int, default=2500
        Maximum CiSE relaxation iteration budget per reference sub-stage. Use
        ``0`` to preserve the Step 1/2 static placement.
    gravity : float, default=0.25
        Root graph gravity strength used by the relaxation phase.
    gravity_range : float, default=3.8
        Root graph gravity range multiplier.
    randomize : bool, default=False
        Whether to include Cytoscape CiSE's randomized Step 3 reversal stage.

    Returns
    -------
    Pipeline
        Composable CiSE pipeline.
    """
    return Pipeline(
        [
            CytoscapeCircleClusters(node_separation=node_separation),
            CytoscapeCiSERelax(
                steps=steps,
                gravity=gravity,
                gravity_range=gravity_range,
                randomize=randomize,
            ),
            CytoscapeFinalize(),
        ],
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
    nodeSeparation: float = 12.5,
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
        Maximum CiSE relaxation iteration budget per reference sub-stage. Use
        ``0`` to preserve the Step 1/2 static placement.
    seed : int, default=42
        Accepted for API consistency.
    edge_weights : torch.Tensor | None, optional
        Accepted for API consistency.
    clusters : dict[str, Any] | None, optional
        Cluster membership mapping.
    cluster_parents : dict[str, str | None] | None, optional
        Cluster parent mapping.
    nodeSeparation : float, default=12.5
        Separation used for circular clusters.
    randomize : bool, default=False
        Whether to include Cytoscape CiSE's randomized Step 3 reversal stage.
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
    del seed, edge_weights, cluster_parents
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        clusters=clusters,
    )
    state = build_cise_pipeline(
        node_separation=nodeSeparation,
        steps=steps,
        gravity=gravity,
        gravity_range=gravityRange,
        randomize=randomize,
    ).apply(
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
