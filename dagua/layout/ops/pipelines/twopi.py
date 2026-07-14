"""Graphviz twopi-compatible public layout pipeline."""

from __future__ import annotations

from typing import Any, Optional

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.graphviz_radial_circular import TwopiAssignRadialCoordinates
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def build_twopi_pipeline(
    ranksep: float = 72.0,
    root: Optional[int] = None,
) -> Pipeline:
    """Build the composable twopi pipeline.

    Parameters
    ----------
    ranksep : float, default=72.0
        Distance between BFS rings in points.
    root : int, optional
        Explicit radial root index.

    Returns
    -------
    Pipeline
        Pipeline containing the radial coordinate assignment op.
    """
    return Pipeline(
        [TwopiAssignRadialCoordinates(ranksep=ranksep, root=root)],
        name="twopi",
    )


def layout_twopi_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    ranksep: Optional[float] = None,
    root: Optional[int] = None,
    fidelity_dtype: torch.dtype = torch.float32,
    config: Optional[Any] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Run the deterministic twopi radial layout pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node sizes with shape ``[N, 2]``; accepted for dispatcher
        compatibility.
    ranksep : float, optional
        Ring spacing in points. Defaults to Graphviz's one-inch spacing.
    root : int, optional
        Explicit root node index.
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
    del kwargs
    effective_ranksep = 72.0 if ranksep is None else float(ranksep)
    if config is not None and root is None:
        root = getattr(config, "root", None)
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
    )
    state = build_twopi_pipeline(ranksep=effective_ranksep, root=root).apply(
        problem,
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device=str(edge_index.device))),
    )
    return state.pos.to(dtype=fidelity_dtype)


__all__ = ["build_twopi_pipeline", "layout_twopi_pipeline"]
