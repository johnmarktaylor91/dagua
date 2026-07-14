"""Graphviz circo-compatible public layout pipeline."""

from __future__ import annotations

from typing import Any, Optional

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.graphviz_radial_circular import CircoAssignCircularCoordinates
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def build_circo_pipeline(nodesep: float = 18.0) -> Pipeline:
    """Build the composable circo pipeline.

    Parameters
    ----------
    nodesep : float, default=18.0
        Approximate separation between adjacent block nodes in points.

    Returns
    -------
    Pipeline
        Pipeline containing the circular block coordinate assignment op.
    """
    return Pipeline(
        [CircoAssignCircularCoordinates(nodesep=nodesep)],
        name="circo",
    )


def layout_circo_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    nodesep: Optional[float] = None,
    fidelity_dtype: torch.dtype = torch.float32,
    config: Optional[Any] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Run the deterministic circo circular layout pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node sizes with shape ``[N, 2]``; accepted for dispatcher
        compatibility.
    nodesep : float, optional
        Approximate node separation in points.
    fidelity_dtype : torch.dtype, default=torch.float32
        Output dtype requested by the caller.
    config : Any, optional
        Full layout config supplied by the engine.
    **kwargs : Any
        Additional dispatcher arguments accepted for compatibility.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.
    """
    del config, kwargs
    effective_nodesep = 18.0 if nodesep is None else float(nodesep)
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
    )
    state = build_circo_pipeline(nodesep=effective_nodesep).apply(
        problem,
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device=str(edge_index.device))),
    )
    return state.pos.to(dtype=fidelity_dtype)


__all__ = ["build_circo_pipeline", "layout_circo_pipeline"]
