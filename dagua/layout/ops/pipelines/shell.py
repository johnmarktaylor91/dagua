"""NetworkX shell-layout pipeline."""

from __future__ import annotations

from typing import List, Optional, Union

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.networkx_simple import NetworkXSimpleLayout
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def build_shell_pipeline(
    nlist: Optional[Union[List[List[int]], str]] = None,
    rotate: Optional[float] = None,
    scale: float = 1.0,
) -> Pipeline:
    """Build the NetworkX shell-layout pipeline.

    Parameters
    ----------
    nlist : list[list[int]] | str | None, optional
        Shell membership. ``None`` mirrors NetworkX's single-shell default;
        ``"bfs"`` uses Dagua's pinned BFS-distance fallback.
    rotate : float | None, optional
        Starting-angle increment between shells.
    scale : float, default=1.0
        NetworkX layout scale.

    Returns
    -------
    Pipeline
        Single-stage composable coordinate pipeline.
    """
    return Pipeline(
        [NetworkXSimpleLayout("shell", {"nlist": nlist, "rotate": rotate, "scale": scale})],
        name="shell_pipeline",
    )


def layout_shell_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: Optional[int] = 42,
    edge_weights: Optional[torch.Tensor] = None,
    nlist: Optional[Union[List[List[int]], str]] = None,
    rotate: Optional[float] = None,
    scale: float = 1.0,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run the deterministic NetworkX shell-layout source port.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor | None, optional
        Optional node-size tensor with shape ``[N, 2]``.
    seed : int | None, default=42
        Accepted for API consistency; shell layout is deterministic.
    edge_weights : torch.Tensor | None, optional
        Accepted for API consistency; shell layout ignores weights.
    nlist : list[list[int]] | str | None, optional
        Shell membership.
    rotate : float | None, optional
        Starting-angle increment between shells.
    scale : float, default=1.0
        NetworkX layout scale.
    fidelity_dtype : torch.dtype | None, optional
        Output dtype for direct fidelity checks.

    Returns
    -------
    torch.Tensor
        Coordinate tensor with shape ``[N, 2]``.
    """
    del seed, edge_weights
    problem = LayoutProblem(edge_index=edge_index, num_nodes=num_nodes, node_sizes=node_sizes)
    state = build_shell_pipeline(nlist=nlist, rotate=rotate, scale=scale).apply(
        problem,
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device="cpu")),
    )
    if state.pos is None:
        raise RuntimeError("Shell pipeline did not produce positions.")
    return state.pos.to(dtype=fidelity_dtype) if fidelity_dtype is not None else state.pos


__all__ = ["build_shell_pipeline", "layout_shell_pipeline"]
