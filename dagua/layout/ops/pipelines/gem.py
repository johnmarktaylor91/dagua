"""GEM (Graph Embedder) expressed as a composable ops pipeline."""

from __future__ import annotations

from typing import Optional

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig
from dagua.layout.ops.gem import (
    GEMBatchedSolve,
    GEMFinalizePositions,
    GEMPrepareState,
    GEMSequentialSolve,
    InitializeGEMPositions,
)
from dagua.layout.ops.state import (  # noqa: E402
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)


def build_gem_pipeline(max_iters: int = 500) -> Pipeline:
    """Build a GEM pipeline that is bit-identical to classic ``layout_gem``.

    Parameters
    ----------
    max_iters : int, default=500
        Maximum number of OGDF node updates.

    Returns
    -------
    Pipeline
        Pipeline that reproduces classic GEM's initialization, sequential/batched
        solve, and postprocessing.

    Raises
    ------
    ValueError
        If ``max_iters`` is negative.
    """
    if max_iters < 0:
        raise ValueError("max_iters must be non-negative.")

    return Pipeline(
        [
            FixedSteps(FixedStepsConfig(n=max_iters)),
            InitializeGEMPositions(),
            GEMPrepareState(),
            GEMSequentialSolve(),
            GEMBatchedSolve(),
            GEMFinalizePositions(),
        ],
        name="gem_pipeline",
    )


def layout_gem_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    max_iters: int = 500,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the GEM pipeline as a drop-in replacement for classic ``layout_gem``.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor used only to resolve the output device.
    max_iters : int, default=500
        Maximum number of OGDF node updates.
    seed : int, default=42
        Random seed for initialization and permutations.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Final position tensor with the same dtype, device, and values as
        classic ``layout_gem``.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``max_iters``, or ``edge_weights`` are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if max_iters < 0:
        raise ValueError("max_iters must be non-negative.")
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
    final_state = build_gem_pipeline(max_iters=max_iters).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("GEM pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_gem_pipeline", "layout_gem_pipeline"]
