"""Fruchterman-Reingold expressed as a composable ops pipeline."""

from __future__ import annotations

from typing import Optional

import torch

from dagua.layout.ops.anneal import InitTemperatureFromExtent, LinearCool
from dagua.layout.ops.base import Pipeline, Repeat  # noqa: E402
from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig, FRConvergenceCheck  # noqa: E402
from dagua.layout.ops.force import ApplyDisplacement, FRCombinedForce
from dagua.layout.ops.init import RandomUniformInit, RandomUniformInitConfig
from dagua.layout.ops.postprocess import FRFinalizePositions
from dagua.layout.ops.preprocess import FRPrepareAdjacency
from dagua.layout.ops.state import (  # noqa: E402
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)


def build_fr_pipeline(steps: int = 50) -> Pipeline:
    """Build an FR pipeline that is bit-identical to classic ``layout_fr``.

    Parameters
    ----------
    steps : int, default=50
        Maximum number of FR iterations.

    Returns
    -------
    Pipeline
        Pipeline that reproduces classic ``layout_fr`` initialization, dense force
        update, cooling schedule, convergence rule, and final post-processing.

    Raises
    ------
    ValueError
        If ``steps`` is negative.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative.")

    return Pipeline(
        [
            FixedSteps(FixedStepsConfig(n=steps)),
            RandomUniformInit(
                RandomUniformInitConfig(
                    scale="none",
                    rng_backend="numpy",
                ),
            ),
            FRPrepareAdjacency(),
            InitTemperatureFromExtent(),
            Repeat(
                n=steps,
                ops=[
                    FRCombinedForce(),
                    ApplyDisplacement(),
                    FRConvergenceCheck(),
                    LinearCool(),
                ],
            ),
            FRFinalizePositions(),
        ],
        name="fr_pipeline",
    )


def layout_fr_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 50,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the FR pipeline as a drop-in replacement for classic ``layout_fr``.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor used only to resolve output device.
    steps : int, default=50
        Maximum number of FR iterations.
    seed : int, default=42
        Random seed for the NumPy unit-square initialization.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Final position tensor with the same dtype, device, and values as
        classic ``layout_fr``.

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
    output_device = (
        edge_index.device
        if edge_index.numel() > 0
        else node_sizes.device
        if node_sizes is not None
        else torch.device("cpu")
    )
    ctx = RuntimeContext(plan=ExecutionPlan(device=str(output_device)))
    final_state = build_fr_pipeline(steps=steps).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("FR pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_fr_pipeline", "layout_fr_pipeline"]
