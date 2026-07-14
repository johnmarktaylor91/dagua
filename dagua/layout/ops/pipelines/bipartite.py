"""NetworkX bipartite-layout pipeline."""

from __future__ import annotations

from typing import Optional

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.networkx_simple import NetworkXSimpleLayout
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def build_bipartite_pipeline(
    nodes: Optional[list[int]] = None,
    align: str = "vertical",
    scale: float = 1.0,
    aspect_ratio: float = 4.0 / 3.0,
) -> Pipeline:
    """Build the NetworkX bipartite-layout pipeline.

    Parameters
    ----------
    nodes : list[int] | None, optional
        Left/top node set. ``None`` uses the pinned BFS-parity fallback.
    align : {"vertical", "horizontal"}, default="vertical"
        NetworkX alignment mode.
    scale : float, default=1.0
        NetworkX layout scale.
    aspect_ratio : float, default=4/3
        Width-to-height ratio before NetworkX rescale.

    Returns
    -------
    Pipeline
        Single-stage composable coordinate pipeline.
    """
    return Pipeline(
        [
            NetworkXSimpleLayout(
                "bipartite",
                {"nodes": nodes, "align": align, "scale": scale, "aspect_ratio": aspect_ratio},
            )
        ],
        name="bipartite_pipeline",
    )


def layout_bipartite_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: Optional[int] = 42,
    edge_weights: Optional[torch.Tensor] = None,
    nodes: Optional[list[int]] = None,
    align: str = "vertical",
    scale: float = 1.0,
    aspect_ratio: float = 4.0 / 3.0,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run the deterministic NetworkX bipartite-layout source port.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor | None, optional
        Optional node-size tensor with shape ``[N, 2]``.
    seed : int | None, default=42
        Accepted for API consistency; bipartite layout is deterministic.
    edge_weights : torch.Tensor | None, optional
        Accepted for API consistency; bipartite layout ignores weights.
    nodes : list[int] | None, optional
        Left/top node set.
    align : {"vertical", "horizontal"}, default="vertical"
        NetworkX alignment mode.
    scale : float, default=1.0
        NetworkX layout scale.
    aspect_ratio : float, default=4/3
        Width-to-height ratio before NetworkX rescale.
    fidelity_dtype : torch.dtype | None, optional
        Output dtype for direct fidelity checks.

    Returns
    -------
    torch.Tensor
        Coordinate tensor with shape ``[N, 2]``.
    """
    del seed, edge_weights
    problem = LayoutProblem(edge_index=edge_index, num_nodes=num_nodes, node_sizes=node_sizes)
    state = build_bipartite_pipeline(
        nodes=nodes,
        align=align,
        scale=scale,
        aspect_ratio=aspect_ratio,
    ).apply(problem, SolveState(), RuntimeContext(plan=ExecutionPlan(device="cpu")))
    if state.pos is None:
        raise RuntimeError("Bipartite pipeline did not produce positions.")
    return state.pos.to(dtype=fidelity_dtype) if fidelity_dtype is not None else state.pos


__all__ = ["build_bipartite_pipeline", "layout_bipartite_pipeline"]
