"""Pivot-MDS layout pipeline."""

from __future__ import annotations

from typing import Optional

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.distance import PivotDistanceQueries, PivotSelection, PivotSelectionConfig
from dagua.layout.ops.embed import PivotMDSComputeCoordinates
from dagua.layout.ops.postprocess import PivotMDSFinalizePositions
from dagua.layout.ops.preprocess import BuildAdjacency, BuildAdjacencyConfig
from dagua.layout.ops.state import (  # noqa: E402
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)


def build_pivot_mds_pipeline(
    n_pivots: int = 50,
    weighted: bool = False,
    first_pivot_index: Optional[int] = None,
) -> Pipeline:
    """Build a Pivot-MDS pipeline.

    Parameters
    ----------
    n_pivots : int, default=50
        Maximum number of pivots to select.
    weighted : bool, default=False
        Whether to treat edges as weighted during adjacency construction.
    first_pivot_index : int | None, default=None
        Optional deterministic first pivot used by reference-compatible
        callers. ``None`` preserves the seeded Pivot-MDS default.

    Returns
    -------
    Pipeline
        Pipeline implementing the Pivot-MDS algorithm. The pipeline produces
        final node coordinates by building adjacency, selecting pivots,
        computing pivot-to-node distances, solving the low-rank MDS embedding,
        and finalizing the layout.

    Raises
    ------
    ValueError
        If ``n_pivots`` is not positive.
    """
    if n_pivots <= 0:
        raise ValueError("n_pivots must be positive.")

    return Pipeline(
        [
            BuildAdjacency(
                BuildAdjacencyConfig(
                    weighted=weighted,
                    dedup="min",
                    format="list",
                ),
            ),
            PivotSelection(
                PivotSelectionConfig(
                    n_pivots=n_pivots,
                    first_pivot_index=first_pivot_index,
                )
            ),
            PivotDistanceQueries(),
            PivotMDSComputeCoordinates(),
            PivotMDSFinalizePositions(),
        ],
        name="pivot_mds_pipeline",
    )


def layout_pivot_mds_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    n_pivots: int = 50,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    first_pivot_index: Optional[int] = None,
) -> torch.Tensor:
    """Run the Pivot-MDS pipeline as a drop-in replacement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]`` used to scale the
        final drawing extent.
    n_pivots : int, default=50
        Maximum number of pivots to select.
    seed : int, default=42
        Random seed for the first pivot.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.
    first_pivot_index : int | None, default=None
        Optional deterministic first pivot used by reference-compatible
        callers. ``None`` preserves the seeded Pivot-MDS default.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``n_pivots``, or ``edge_weights`` are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if n_pivots <= 0:
        raise ValueError("n_pivots must be positive.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
            )

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_pivot_mds_pipeline(
        n_pivots=n_pivots,
        weighted=problem.edge_weights is not None,
        first_pivot_index=first_pivot_index,
    ).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("Pivot-MDS pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_pivot_mds_pipeline", "layout_pivot_mds_pipeline"]
