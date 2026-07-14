"""Tutte barycentric embedding layout pipeline."""

from __future__ import annotations

from typing import Any, Optional

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.tutte import TutteBarycentricEmbedding


def build_tutte_pipeline(radius: float = 1.0) -> Pipeline:
    """Build the deterministic Tutte embedding pipeline.

    Parameters
    ----------
    radius : float, default=1.0
        Radius of the fixed convex boundary polygon.

    Returns
    -------
    Pipeline
        Pipeline containing the fixed-boundary linear solve op.
    """
    return Pipeline([TutteBarycentricEmbedding(radius=radius)], name="tutte")


def layout_tutte_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: Optional[int] = 42,
    edge_weights: Optional[torch.Tensor] = None,
    radius: float = 1.0,
    fidelity_dtype: torch.dtype = torch.float32,
    config: Optional[Any] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Run the Tutte barycentric embedding pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N``.
    node_sizes : torch.Tensor, optional
        Optional node sizes with shape ``[N, 2]``. Accepted for dispatcher
        compatibility.
    seed : int, optional
        Accepted for dispatcher compatibility. Tutte embedding is
        deterministic once the boundary is selected.
    edge_weights : torch.Tensor, optional
        Optional positive edge weights with shape ``[E]`` used in the
        barycentric linear system.
    radius : float, default=1.0
        Radius of the fixed convex boundary polygon.
    fidelity_dtype : torch.dtype, default=torch.float32
        Output dtype requested by the caller.
    config : Any, optional
        Full layout config supplied by the engine.
    **kwargs : Any
        Additional dispatcher arguments accepted for compatibility.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    del seed, config, kwargs
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
    )
    state = build_tutte_pipeline(radius=radius).apply(
        problem,
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device=str(edge_index.device))),
    )
    if state.pos is None:
        raise RuntimeError("Tutte pipeline did not produce positions.")
    return state.pos.to(dtype=fidelity_dtype, device=edge_index.device)


__all__ = ["build_tutte_pipeline", "layout_tutte_pipeline"]
