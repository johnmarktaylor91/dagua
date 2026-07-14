"""Harel-Koren high-dimensional embedding layout pipeline."""

from __future__ import annotations

from typing import Any, Optional

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.distance import (
    PivotDistanceQueries,
    PivotDistanceQueriesConfig,
    PivotSelection,
    PivotSelectionConfig,
)
from dagua.layout.ops.hde import HDEProjectPivotDistances
from dagua.layout.ops.preprocess import BuildAdjacency, BuildAdjacencyConfig
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def build_hde_pipeline(n_pivots: int = 50) -> Pipeline:
    """Build the deterministic HDE pipeline.

    Parameters
    ----------
    n_pivots : int, default=50
        Maximum number of farthest-first pivots. The first pivot is node 0.

    Returns
    -------
    Pipeline
        Pipeline that builds undirected adjacency, selects pivots, queries
        pivot distances, and runs the reusable HDE PCA init op.

    Raises
    ------
    ValueError
        If ``n_pivots`` is not positive.
    """
    if n_pivots <= 0:
        raise ValueError("n_pivots must be positive.")
    return Pipeline(
        [
            BuildAdjacency(BuildAdjacencyConfig(directed=False, weighted=False, format="list")),
            PivotSelection(PivotSelectionConfig(n_pivots=n_pivots, first_pivot="first_node")),
            PivotDistanceQueries(PivotDistanceQueriesConfig(dtype=torch.float64)),
            HDEProjectPivotDistances(),
        ],
        name="hde",
    )


def layout_hde_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: Optional[int] = 42,
    edge_weights: Optional[torch.Tensor] = None,
    n_pivots: int = 50,
    fidelity_dtype: torch.dtype = torch.float32,
    config: Optional[Any] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Run the HDE layout pipeline.

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
        Accepted for dispatcher compatibility. HDE is deterministic because
        the first pivot is fixed to node 0.
    edge_weights : torch.Tensor, optional
        Accepted for dispatcher compatibility. HDE uses unweighted BFS hops.
    n_pivots : int, default=50
        Maximum number of pivots.
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
    del edge_weights, config, kwargs
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        seed=42 if seed is None else int(seed),
    )
    state = build_hde_pipeline(n_pivots=n_pivots).apply(
        problem,
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device=str(edge_index.device))),
    )
    if state.pos is None:
        raise RuntimeError("HDE pipeline did not produce positions.")
    return state.pos.to(dtype=fidelity_dtype, device=edge_index.device)


__all__ = ["build_hde_pipeline", "layout_hde_pipeline"]
