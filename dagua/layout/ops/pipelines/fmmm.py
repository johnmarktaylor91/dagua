"""FM^3 multilevel force-directed layout expressed as a composable ops pipeline."""

from __future__ import annotations

from typing import Optional

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.fmmm import (
    _FinalizeFMMMPositions,
    _InitializeCoarsestLevel,
    _InitializeFMMMState,
    _RefineCoarsestLevel,
    _SingleLevelFallback,
    _UncoarsenLoop,
)
from dagua.layout.ops.graph_utils import layout_device as _layout_device
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def build_fmmm_pipeline(steps: int = 100) -> Pipeline:
    """Build an FM^3 pipeline that is bit-identical to classic ``layout_fmmm``.

    Parameters
    ----------
    steps : int, default=100
        Total refinement budget across hierarchy levels.

    Returns
    -------
    Pipeline
        Pipeline that reproduces classic FM^3's hierarchy construction,
        coarse FR initialization, per-level refinement, prolongation,
        and final normalization.

    Raises
    ------
    ValueError
        If ``steps`` is negative.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative.")

    initialize_state = _InitializeFMMMState(steps=steps)
    initialize_coarsest = _InitializeCoarsestLevel()
    refine_coarsest = _RefineCoarsestLevel()
    uncoarsen_loop = _UncoarsenLoop()
    single_level_fallback = _SingleLevelFallback()
    finalize_positions = _FinalizeFMMMPositions()

    return Pipeline(
        [
            initialize_state,
            initialize_coarsest,
            refine_coarsest,
            uncoarsen_loop,
            single_level_fallback,
            finalize_positions,
        ],
        name="fmmm_pipeline",
    )


def layout_fmmm_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 100,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the FM^3 pipeline as a drop-in replacement for classic ``layout_fmmm``.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor used for extent calculation and output device.
    steps : int, default=100
        Total refinement budget across hierarchy levels.
    seed : int, default=42
        Random seed for coarsening, FR initialization, and prolongation jitter.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Final position tensor with the same dtype, device, and values as
        classic ``layout_fmmm``.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``steps``, or ``edge_weights`` are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge_count {edge_index.shape[1]}"
            )

    device = _layout_device(edge_index=edge_index, node_sizes=node_sizes)
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32, device=device)

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_fmmm_pipeline(steps=steps).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("FM^3 pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_fmmm_pipeline", "layout_fmmm_pipeline"]
