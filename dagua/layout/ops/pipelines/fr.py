"""Fruchterman-Reingold force-directed layout pipeline."""

from __future__ import annotations

from typing import Optional

import torch

from dagua.layout.ops.anneal import InitTemperatureFromExtent, LinearCool
from dagua.layout.ops.base import Conditional, Pipeline, Repeat  # noqa: E402
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
    """Build a Fruchterman-Reingold force-directed layout pipeline.

    Parameters
    ----------
    steps : int, default=50
        Maximum number of cooling iterations to run.

    Returns
    -------
    Pipeline
        Pipeline implementing the classical Fruchterman-Reingold algorithm.
        The pipeline produces final node coordinates by sampling unit-square
        initial positions, building adjacency data, setting an initial
        temperature from the current extent, iterating attraction and
        repulsion force updates with displacement and linear cooling, then
        finalizing the coordinates.

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
            Conditional(
                predicate=lambda problem, state, ctx: state.pos is None,
                op=RandomUniformInit(
                    RandomUniformInitConfig(
                        scale="none",
                        rng_backend="numpy",
                    ),
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
    pos: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the Fruchterman-Reingold pipeline as a drop-in replacement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]`` used only to resolve
        the output device.
    steps : int, default=50
        Maximum number of cooling iterations to run.
    seed : int, default=42
        Random seed for the NumPy-backed unit-square initialization.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.
    pos : torch.Tensor, optional
        Initial positions with shape ``[N, 2]``. When provided, the pipeline
        starts from these coordinates instead of sampling a random
        initialization.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``steps``, ``edge_weights``, or ``pos`` are invalid.
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
    if pos is not None and pos.shape != (num_nodes, 2):
        raise ValueError(f"pos must have shape ({num_nodes}, 2), got {tuple(pos.shape)}")

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    if pos is not None:
        state.pos = pos.detach().clone().to(dtype=torch.float64)
    output_device = (
        edge_index.device
        if edge_index.numel() > 0
        else node_sizes.device
        if node_sizes is not None
        else torch.device("cpu")
    )
    if pos is not None and steps == 0:
        return state.pos.to(device=output_device, dtype=torch.float32)
    ctx = RuntimeContext(plan=ExecutionPlan(device=str(output_device)))
    final_state = build_fr_pipeline(steps=steps).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("FR pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_fr_pipeline", "layout_fr_pipeline"]
