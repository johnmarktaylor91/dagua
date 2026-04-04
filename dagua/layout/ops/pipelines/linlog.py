"""LinLog expressed as a composable ops pipeline."""

from __future__ import annotations

from typing import Optional

import torch

from dagua.layout.ops.anneal import LRDecay
from dagua.layout.ops.base import Conditional, LossGroup, Pipeline, Repeat
from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig
from dagua.layout.ops.init import LinLogInitializePositions
from dagua.layout.ops.loss_classic import LinLogLoss, LinLogLossConfig
from dagua.layout.ops.optimize import (
    LinLogCreateOptimizer,
    OptimizerStep,
    OptimizerZeroGrad,
)
from dagua.layout.ops.postprocess import LinLogFinalizePositions
from dagua.layout.ops.state import (
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)


def build_linlog_pipeline(
    steps: int = 300,
    a: float = 1.0,
    r: float = 0.0,
) -> Pipeline:
    """Build a LinLog pipeline that matches classic ``layout_linlog``.

    Parameters
    ----------
    steps : int, default=300
        Number of Adam updates.
    a : float, default=1.0
        Attraction exponent.
    r : float, default=0.0
        Repulsion exponent.

    Returns
    -------
    Pipeline
        Pipeline that reproduces classic ``layout_linlog``.

    Raises
    ------
    ValueError
        If ``steps``, ``a``, or ``r`` are invalid.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if a < 0.0:
        raise ValueError("a must be non-negative.")
    if r < 0.0:
        raise ValueError("r must be non-negative.")

    objective = LinLogLoss(
        config=LinLogLossConfig(exponent_a=a, exponent_r=r),
    )

    optimize_pipeline = Pipeline(
        [
            LinLogCreateOptimizer(),
            Repeat(
                n=steps,
                ops=[
                    OptimizerZeroGrad(),
                    LossGroup([objective]),
                    OptimizerStep(),
                    LRDecay(),
                ],
            ),
        ],
        name="linlog_optimize",
    )

    return Pipeline(
        [
            FixedSteps(FixedStepsConfig(n=steps)),
            LinLogInitializePositions(),
            Conditional(
                predicate=lambda problem, state, ctx: problem.num_nodes > 1,
                op=optimize_pipeline,
            ),
            LinLogFinalizePositions(),
        ],
        name="linlog_pipeline",
    )


def layout_linlog_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 300,
    seed: int = 42,
    a: float = 1.0,
    r: float = 0.0,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the LinLog pipeline as a drop-in replacement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor used only to scale the final layout.
    steps : int, default=300
        Number of Adam updates.
    seed : int, default=42
        Random seed for initialization and repulsion sampling.
    a : float, default=1.0
        Attraction exponent.
    r : float, default=0.0
        Repulsion exponent.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Final position tensor with the same dtype, device, and values as
        classic ``layout_linlog``.

    Raises
    ------
    ValueError
        If the public LinLog inputs are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if a < 0.0:
        raise ValueError("a must be non-negative.")
    if r < 0.0:
        raise ValueError("r must be non-negative.")
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
    final_state = build_linlog_pipeline(steps=steps, a=a, r=r).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("LinLog pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_linlog_pipeline", "layout_linlog_pipeline"]
