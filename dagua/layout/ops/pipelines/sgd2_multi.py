"""(SGD)^2 multicriteria layout expressed as a composable ops pipeline."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.sgd2_multi import (
    SmoothSteps,
    _InitSGD2MultiState,
    _RunSGD2MultiOptimization,
)
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def build_sgd2_multi_pipeline(
    steps: int = 10_000,
    criteria: Optional[Dict[str, float]] = None,
    criteria_schedules: Optional[Dict[str, SmoothSteps]] = None,
    lr: float = 1.0,
    momentum: float = 0.7,
    grad_clamp: float = 4.0,
    batch_size: int = 16,
) -> Pipeline:
    """Build an (SGD)^2 multicriteria pipeline.

    Parameters
    ----------
    steps : int, default=10000
        Maximum number of SGD iterations.
    criteria : dict[str, float] | None, default=None
        Static per-criterion weights.
    criteria_schedules : dict[str, SmoothSteps] | None, default=None
        Piecewise-smooth criterion schedules.
    lr : float, default=1.0
        SGD learning rate.
    momentum : float, default=0.7
        SGD momentum.
    grad_clamp : float, default=4.0
        Symmetric gradient clamp.
    batch_size : int, default=16
        Global mini-batch size.

    Returns
    -------
    Pipeline
        Pipeline that reproduces classic (SGD)^2 behavior.

    Raises
    ------
    ValueError
        If ``steps`` is negative.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative.")

    return Pipeline(
        [
            _InitSGD2MultiState(
                steps=steps,
                lr=lr,
                momentum=momentum,
                grad_clamp=grad_clamp,
                batch_size=batch_size,
                criteria=criteria,
                criteria_schedules=criteria_schedules,
            ),
            _RunSGD2MultiOptimization(
                steps=steps,
                lr=lr,
                momentum=momentum,
                grad_clamp=grad_clamp,
                batch_size=batch_size,
            ),
        ],
        name="sgd2_multi_pipeline",
    )


def layout_sgd2_multi_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: int = 42,
    steps: int = 10_000,
    criteria: Optional[Dict[str, float]] = None,
    criteria_schedules: Optional[Dict[str, SmoothSteps]] = None,
    lr: float = 1.0,
    momentum: float = 0.7,
    grad_clamp: float = 4.0,
    batch_size: int = 16,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the (SGD)^2 multicriteria pipeline as a drop-in replacement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor | None, default=None
        Unused placeholder kept for interface compatibility.
    seed : int, default=42
        Random seed.
    steps : int, default=10000
        Maximum SGD iterations.
    criteria : dict[str, float] | None, default=None
        Static criterion weights.
    criteria_schedules : dict[str, SmoothSteps] | None, default=None
        Piecewise-smooth criterion schedules.
    lr : float, default=1.0
        SGD learning rate.
    momentum : float, default=0.7
        SGD momentum.
    grad_clamp : float, default=4.0
        Symmetric gradient clamp.
    batch_size : int, default=16
        Mini-batch size.
    edge_weights : torch.Tensor, optional
        Optional per-edge weights with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Final coordinates with shape ``[N, 2]`` and dtype ``float32``.

    Raises
    ------
    ValueError
        If input arguments are invalid.
    RuntimeError
        If the pipeline fails to produce final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if lr <= 0.0:
        raise ValueError("lr must be positive.")
    if momentum < 0.0 or momentum >= 1.0:
        raise ValueError("momentum must be in [0, 1).")
    if grad_clamp <= 0.0:
        raise ValueError("grad_clamp must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
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
    final_state = build_sgd2_multi_pipeline(
        steps=steps,
        criteria=criteria,
        criteria_schedules=criteria_schedules,
        lr=lr,
        momentum=momentum,
        grad_clamp=grad_clamp,
        batch_size=batch_size,
    ).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("(SGD)^2 pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_sgd2_multi_pipeline", "layout_sgd2_multi_pipeline"]
