"""Deterministic star-layout pipeline."""

from __future__ import annotations

from typing import Optional

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.networkx_simple import NetworkXSimpleLayout
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def build_star_pipeline(center: Optional[int] = None, scale: float = 1.0) -> Pipeline:
    """Build the deterministic star-layout pipeline.

    Parameters
    ----------
    center : int | None, optional
        Explicit center node. ``None`` selects the maximum-degree node.
    scale : float, default=1.0
        Leaf-circle radius.

    Returns
    -------
    Pipeline
        Single-stage composable coordinate pipeline.
    """
    return Pipeline(
        [NetworkXSimpleLayout("star", {"center": center, "scale": scale})],
        name="star_pipeline",
    )


def layout_star_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: Optional[int] = 42,
    edge_weights: Optional[torch.Tensor] = None,
    center: Optional[int] = None,
    scale: float = 1.0,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run the deterministic star-layout source port.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor | None, optional
        Optional node-size tensor with shape ``[N, 2]``.
    seed : int | None, default=42
        Accepted for API consistency; star layout is deterministic.
    edge_weights : torch.Tensor | None, optional
        Accepted for API consistency; star layout ignores weights.
    center : int | None, optional
        Explicit center node.
    scale : float, default=1.0
        Leaf-circle radius.
    fidelity_dtype : torch.dtype | None, optional
        Output dtype for direct fidelity checks.

    Returns
    -------
    torch.Tensor
        Coordinate tensor with shape ``[N, 2]``.
    """
    del seed, edge_weights
    problem = LayoutProblem(edge_index=edge_index, num_nodes=num_nodes, node_sizes=node_sizes)
    state = build_star_pipeline(center=center, scale=scale).apply(
        problem,
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device="cpu")),
    )
    if state.pos is None:
        raise RuntimeError("Star pipeline did not produce positions.")
    return state.pos.to(dtype=fidelity_dtype) if fidelity_dtype is not None else state.pos


__all__ = ["build_star_pipeline", "layout_star_pipeline"]
