"""Cytoscape AVSDF circular layout pipeline."""

from __future__ import annotations

from typing import Optional

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.cytoscape import AVSDFLayoutOp, CytoscapeFinalize
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def build_avsdf_pipeline(node_separation: float = 60.0, postprocess: bool = True) -> Pipeline:
    """Build the Cytoscape AVSDF pipeline.

    Parameters
    ----------
    node_separation : float, default=60.0
        Cytoscape AVSDF node separation.
    postprocess : bool, default=True
        Whether to run AVSDF local crossing reduction.

    Returns
    -------
    Pipeline
        Composable AVSDF pipeline.
    """
    return Pipeline(
        [
            AVSDFLayoutOp(node_separation=node_separation, postprocess=postprocess),
            CytoscapeFinalize(),
        ],
        name="avsdf_pipeline",
    )


def layout_avsdf_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: Optional[int] = 42,
    edge_weights: Optional[torch.Tensor] = None,
    nodeSeparation: float = 60.0,
    postprocess: bool = True,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run the Cytoscape AVSDF pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor | None, optional
        Node-size tensor with shape ``[N, 2]``.
    seed : int | None, default=42
        Accepted for API consistency; AVSDF is deterministic.
    edge_weights : torch.Tensor | None, optional
        Accepted for API consistency; AVSDF ignores weights.
    nodeSeparation : float, default=60.0
        Cytoscape AVSDF node separation.
    postprocess : bool, default=True
        Whether to run local crossing reduction.
    fidelity_dtype : torch.dtype | None, optional
        Optional output dtype override.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    del seed, edge_weights
    problem = LayoutProblem(edge_index=edge_index, num_nodes=num_nodes, node_sizes=node_sizes)
    state = build_avsdf_pipeline(node_separation=nodeSeparation, postprocess=postprocess).apply(
        problem,
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device="cpu")),
    )
    if state.pos is None:
        raise RuntimeError("AVSDF pipeline did not produce positions.")
    if fidelity_dtype is not None:
        return state.pos.to(dtype=fidelity_dtype)
    return state.pos


__all__ = ["build_avsdf_pipeline", "layout_avsdf_pipeline"]
