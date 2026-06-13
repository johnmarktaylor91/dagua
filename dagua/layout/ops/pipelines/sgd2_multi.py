"""(SGD)^2 multicriteria layout pipeline."""

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
    steps: int = 2_000,
    criteria: Optional[Dict[str, float]] = None,
    criteria_schedules: Optional[Dict[str, SmoothSteps]] = None,
    lr: float = 1.0,
    momentum: float = 0.7,
    grad_clamp: float = 5.0,
    batch_size: int = 128,
    fidelity_mode: bool = False,
) -> Pipeline:
    """Build an ``(SGD)^2`` multicriteria layout pipeline.

    Reference fidelity
    ------------------
    Targets: historical ``(SGD)^2`` multi-criteria graph-drawing reference
        sources from the graph-drawing project.
    Fidelity mode: ``layout_sgd2_multi_pipeline(..., fidelity_mode=True)`` is
        retained for variant compatibility and still runs this native port.
    Verified at: Round 64 replaces runtime reference delegation with a direct
        PyTorch implementation of the GD2 optimization loop.
    Known divergences:
        - The native port intentionally avoids importing measurement adapters
          or optional third-party layout packages from runtime pipelines.

    Parameters
    ----------
    steps : int, default=2000
        Maximum number of SGD iterations.
    criteria : dict[str, float] | None, default=None
        Static per-criterion weights.
    criteria_schedules : dict[str, SmoothSteps] | None, default=None
        Piecewise-smooth criterion schedules.
    lr : float, default=1.0
        SGD learning rate.
    momentum : float, default=0.7
        SGD momentum.
    grad_clamp : float, default=5.0
        Symmetric gradient clamp.
    batch_size : int, default=128
        Global mini-batch size matching the reference adapter's default
        per-criterion sample size.
    fidelity_mode : bool, default=False
        Accepted for interface parity with ``layout_sgd2_multi_pipeline``.

    Returns
    -------
    Pipeline
        Pipeline implementing the multicriteria ``(SGD)^2`` algorithm. The
        pipeline produces final node coordinates by initializing the optimizer
        state, scheduling criterion weights, and running stochastic
        multi-objective optimization until the configured budget is exhausted.

    Raises
    ------
    ValueError
        If ``steps`` is negative.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    # The public flag is retained for variant compatibility; the implementation
    # is now always the native Python port rather than a runtime delegation.
    del fidelity_mode

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
    steps: int = 2_000,
    criteria: Optional[Dict[str, float]] = None,
    criteria_schedules: Optional[Dict[str, SmoothSteps]] = None,
    lr: float = 1.0,
    momentum: float = 0.7,
    grad_clamp: float = 5.0,
    batch_size: int = 128,
    edge_weights: Optional[torch.Tensor] = None,
    use_reference_fallback: bool = False,
    fidelity_mode: bool = False,
) -> torch.Tensor:
    """Run the ``(SGD)^2`` multicriteria pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor | None, default=None
        Optional node-size tensor with shape ``[N, 2]``. Unused placeholder
        kept for interface compatibility.
    seed : int, default=42
        Random seed.
    steps : int, default=2000
        Maximum SGD iterations.
    criteria : dict[str, float] | None, default=None
        Static criterion weights.
    criteria_schedules : dict[str, SmoothSteps] | None, default=None
        Piecewise-smooth criterion schedules.
    lr : float, default=1.0
        SGD learning rate.
    momentum : float, default=0.7
        SGD momentum.
    grad_clamp : float, default=5.0
        Symmetric gradient clamp.
    batch_size : int, default=128
        Mini-batch size matching the reference adapter's default
        per-criterion sample size.
    edge_weights : torch.Tensor, optional
        Optional per-edge weights with shape ``[E]``.
    use_reference_fallback : bool, default=False
        Deprecated compatibility flag. The native pipeline ignores it.
    fidelity_mode : bool, default=False
        Deprecated compatibility flag. The native pipeline ignores it.

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

    del use_reference_fallback

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
        fidelity_mode=fidelity_mode,
    ).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("(SGD)^2 pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_sgd2_multi_pipeline", "layout_sgd2_multi_pipeline"]
