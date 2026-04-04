"""Stress majorization (SMACOF) expressed as a composable ops pipeline."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch

from dagua.layout.ops.base import Pipeline, Repeat
from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig
from dagua.layout.ops.graph_utils import layout_device as _layout_device
from dagua.layout.ops.state import (
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)
from dagua.layout.ops.stress import (
    TRACE_EVERY_KEY,
    TRACES_KEY,
    CollectStressMajorizationTrace,
    FinalizeStressMajorizationPositions,
    InitializeStressMajorizationPositions,
    PrepareStressMajorizationState,
    SmacofStep,
)


def build_stress_majorization_pipeline(
    iterations: int = 200,
    trace_every: int = 0,
) -> Pipeline:
    """Build a stress majorization pipeline matching the classic implementation.

    Parameters
    ----------
    iterations : int, default=200
        Number of SMACOF majorization steps.
    trace_every : int, default=0
        If positive, collect position snapshots at this cadence.

    Returns
    -------
    Pipeline
        Pipeline that reproduces classic stress majorization exactly.

    Raises
    ------
    ValueError
        If ``iterations`` or ``trace_every`` is negative.
    """
    if iterations < 0:
        raise ValueError("iterations must be non-negative.")
    if trace_every < 0:
        raise ValueError("trace_every must be non-negative.")

    return Pipeline(
        [
            FixedSteps(FixedStepsConfig(n=iterations)),
            PrepareStressMajorizationState(),
            InitializeStressMajorizationPositions(),
            Repeat(
                n=iterations,
                ops=[
                    SmacofStep(),
                    CollectStressMajorizationTrace(),
                ],
            ),
            FinalizeStressMajorizationPositions(),
        ],
        name="stress_majorization_pipeline",
    )


def layout_stress_majorization_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    iterations: int = 200,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    trace_every: int = 0,
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
    """Run the stress majorization pipeline as a drop-in classic replacement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``.
    iterations : int, default=200
        Number of SMACOF majorization steps.
    seed : int, default=42
        Random seed for the stochastic warm-start jitter.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.
    trace_every : int, default=0
        If positive, return periodic position snapshots.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, list[torch.Tensor]]
        Final position tensor with shape ``[N, 2]``. When ``trace_every > 0``,
        also returns periodic snapshots.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``iterations``, ``trace_every``, or ``edge_weights``
        are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if iterations < 0:
        raise ValueError("iterations must be non-negative.")
    if trace_every < 0:
        raise ValueError("trace_every must be non-negative.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must be shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
            )

    # Resolve empty and singleton graphs here to preserve direct-returns in the
    # classic implementation path and keep pipeline behavior simple.
    if num_nodes == 0:
        device = _layout_device(edge_index=edge_index, node_sizes=node_sizes)
        empty = torch.empty((0, 2), dtype=torch.float32, device=device)
        return (empty, []) if trace_every > 0 else empty
    if num_nodes == 1:
        device = _layout_device(edge_index=edge_index, node_sizes=node_sizes)
        single = torch.zeros((1, 2), dtype=torch.float32, device=device)
        return (single, []) if trace_every > 0 else single

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    state.extras[TRACE_EVERY_KEY] = trace_every
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_stress_majorization_pipeline(
        iterations=iterations,
        trace_every=trace_every,
    ).apply(problem, state, ctx)

    if final_state.pos is None:
        raise RuntimeError("Stress majorization pipeline did not produce final positions.")

    if trace_every > 0:
        traces = final_state.extras.get(TRACES_KEY, [])
        return final_state.pos, traces
    return final_state.pos


__all__ = [
    "build_stress_majorization_pipeline",
    "layout_stress_majorization_pipeline",
]
