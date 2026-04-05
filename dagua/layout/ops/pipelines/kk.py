"""Kamada-Kawai spring-embedding layout pipeline."""

from __future__ import annotations

from typing import Optional

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig
from dagua.layout.ops.distance import KamadaKawaiAllPairsShortestPaths
from dagua.layout.ops.graph_utils import layout_device as _layout_device
from dagua.layout.ops.init import KamadaKawaiInitializePositions
from dagua.layout.ops.optimize import LBFGSStep, LBFGSStepConfig
from dagua.layout.ops.postprocess import KamadaKawaiFinalizePositions
from dagua.layout.ops.state import (
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)


def build_kk_pipeline(
    steps: Optional[int] = None,
    trace_every: int = 0,
) -> Pipeline:
    """Build a Kamada-Kawai spring-embedding pipeline.

    Parameters
    ----------
    steps : int, optional
        Maximum L-BFGS-B iterations. ``None`` or ``0`` leaves ``maxiter``
        unset to match classic KK.
    trace_every : int, default=0
        Snapshot cadence for optimizer traces.

    Returns
    -------
    Pipeline
        Pipeline implementing the classical Kamada-Kawai algorithm. The
        pipeline produces final node coordinates by computing all-pairs
        shortest paths, initializing positions, minimizing the spring-energy
        objective with L-BFGS, and finalizing the layout.

    Raises
    ------
    ValueError
        If ``steps`` or ``trace_every`` are invalid.
    """
    if steps is not None and steps < 0:
        raise ValueError("steps must be non-negative.")
    if trace_every < 0:
        raise ValueError("trace_every must be non-negative.")

    optimize_config = LBFGSStepConfig(
        maxiter=steps,
        trace_every=trace_every,
        trace_key="kk_traces" if trace_every > 0 else None,
    )
    return Pipeline(
        [
            FixedSteps(FixedStepsConfig(n=0 if steps is None else steps)),
            KamadaKawaiAllPairsShortestPaths(),
            KamadaKawaiInitializePositions(),
            LBFGSStep(config=optimize_config),
            KamadaKawaiFinalizePositions(),
        ],
        name="kk_pipeline",
    )


def layout_kk_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: Optional[int] = None,
    seed: int = 42,
    trace_every: int = 0,
    solver: str = "auto",
    pos: Optional[torch.Tensor] = None,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    """Run the Kamada-Kawai pipeline as a drop-in replacement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``. Unused and accepted
        for interface compatibility.
    steps : int, optional
        Maximum L-BFGS-B iterations. ``None`` or ``0`` leaves ``maxiter``
        unset to match classic KK.
    seed : int, default=42
        Accepted for interface compatibility. The 2D classic KK path uses
        deterministic circular initialization and does not consume a seed.
    trace_every : int, default=0
        Snapshot cadence for optimizer traces.
    solver : {"auto", "newton", "adam"}, default="auto"
        Retained for interface compatibility. All accepted values resolve to
        the classic SciPy L-BFGS-B path.
    pos : torch.Tensor, optional
        Initial positions with shape ``[N, 2]``. When provided, overrides the
        default circular initialization.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, list[torch.Tensor]]
        Final positions with shape ``[N, 2]``. When ``trace_every > 0``,
        periodic optimizer snapshots are returned alongside the final layout.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``steps``, ``trace_every``, ``solver``, or ``pos``
        are invalid.
    RuntimeError
        If the pipeline does not produce final positions.
    """
    _ = seed

    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if steps is not None and steps < 0:
        raise ValueError("steps must be non-negative.")
    if trace_every < 0:
        raise ValueError("trace_every must be non-negative.")
    if solver not in {"auto", "newton", "adam"}:
        raise ValueError("solver must be one of 'auto', 'newton', or 'adam'.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
            )
    if pos is not None and pos.shape != (num_nodes, 2):
        raise ValueError(f"pos must have shape ({num_nodes}, 2), got {tuple(pos.shape)}")

    output_device = _layout_device(edge_index=edge_index, node_sizes=node_sizes)
    if num_nodes == 0:
        empty = torch.empty((0, 2), dtype=torch.float32, device=output_device)
        return (empty, []) if trace_every > 0 else empty
    if num_nodes == 1:
        single = torch.zeros((1, 2), dtype=torch.float32, device=output_device)
        return (single, []) if trace_every > 0 else single

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
    )
    state = SolveState()
    if pos is not None:
        state.extras["kk_initial_pos"] = pos
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_kk_pipeline(steps=steps, trace_every=trace_every).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("KK pipeline did not produce final positions.")

    if trace_every > 0:
        traces = final_state.extras.get("kk_traces", [])
        return final_state.pos, traces
    return final_state.pos


__all__ = ["build_kk_pipeline", "layout_kk_pipeline"]
