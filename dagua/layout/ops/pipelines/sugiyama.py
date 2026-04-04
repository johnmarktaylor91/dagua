"""Sugiyama layered graph drawing expressed as a composable ops pipeline."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch

from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.sugiyama import (
    _AssignLayers,
    _BarycenterOrdering,
    _BuildEdgeRoutes,
    _BuildNeighborStructures,
    _CoordinateAssignment,
    _ExpandDummyNodes,
    _PrepareAcyclicEdges,
    _ResolveNodeSizes,
    _StoreSpacingParams,
    _ValidateInputs,
)


def build_sugiyama_pipeline(
    rank_sep: float = 1.0,
    node_sep: float = 1.0,
    barycenter_passes: int = 24,
    seed: int = 42,
    trace_every: int = 0,
    return_edge_routes: bool = False,
) -> Pipeline:
    """Build a Sugiyama pipeline that reproduces ``layout_sugiyama``.

    Parameters
    ----------
    rank_sep : float, default=1.0
        Vertical spacing between layers.
    node_sep : float, default=1.0
        Horizontal spacing between nodes within a layer.
    barycenter_passes : int, default=24
        Number of up/down sweeps for crossing minimization.
    seed : int, default=42
        Seed for deterministic tie handling.
    trace_every : int, default=0
        If positive, emit a trace every N passes.
    return_edge_routes : bool, default=False
        If ``True``, include edge-route reconstruction in the pipeline.

    Returns
    -------
    Pipeline
        Pipeline equivalent to classic ``layout_sugiyama``.
    """
    ops: list[Op] = [
        _ValidateInputs(),
        _StoreSpacingParams(rank_sep=rank_sep, node_sep=node_sep),
        _ResolveNodeSizes(),
        _PrepareAcyclicEdges(),
        _AssignLayers(),
        _ExpandDummyNodes(),
        _BuildNeighborStructures(),
        _BarycenterOrdering(
            barycenter_passes=barycenter_passes,
            seed=seed,
            trace_every=trace_every,
        ),
        _CoordinateAssignment(),
    ]
    if return_edge_routes:
        ops.append(_BuildEdgeRoutes())
    return Pipeline(ops, name="sugiyama_pipeline")


def layout_sugiyama_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    rank_sep: float = 1.0,
    node_sep: float = 1.0,
    layer_sep: Optional[float] = None,
    seed: int = 42,
    barycenter_passes: int = 24,
    trace_every: int = 0,
    edge_weights: Optional[torch.Tensor] = None,
    return_edge_routes: bool = False,
) -> Union[
    torch.Tensor,
    Tuple[torch.Tensor, List[torch.Tensor]],
    Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]],
]:
    """Run the Sugiyama pipeline as a drop-in replacement for ``layout_sugiyama``.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor, shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``.
    rank_sep : float
        Vertical spacing between layers.
    node_sep : float
        Horizontal spacing within layers.
    layer_sep : float, optional
        Alias for ``rank_sep``. Overrides ``rank_sep`` when provided.
    seed : int
        Random seed for deterministic tie-breaking.
    barycenter_passes : int
        Number of crossing-minimization sweeps.
    trace_every : int
        If positive, snapshots are emitted every ``trace_every`` sweeps.
    edge_weights : torch.Tensor, optional
        Optional edge-weight vector with shape ``[E]``.
    return_edge_routes : bool, default=False
        If ``True``, include edge-route polylines in output.

    Returns
    -------
    torch.Tensor or tuple
        Final positions and optional traces/routes.

    Raises
    ------
    ValueError
        If ``trace_every`` is negative.
    RuntimeError
        If the pipeline fails to produce final positions.
    """
    if trace_every < 0:
        raise ValueError("trace_every must be non-negative.")
    if layer_sep is not None:
        rank_sep = layer_sep

    output_device = edge_index.device
    if node_sizes is not None:
        output_device = node_sizes.device

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device=str(output_device)))

    pipeline = build_sugiyama_pipeline(
        rank_sep=rank_sep,
        node_sep=node_sep,
        barycenter_passes=barycenter_passes,
        seed=seed,
        trace_every=trace_every,
        return_edge_routes=return_edge_routes,
    )
    final_state = pipeline.apply(problem, state, ctx)

    if final_state.pos is None:
        raise RuntimeError("Sugiyama pipeline did not produce final positions.")

    positions = final_state.pos
    traces = final_state.extras.get("sugiyama_traces", [])
    visible_traces = [trace[:num_nodes] for trace in traces]

    if not return_edge_routes:
        if trace_every > 0:
            return positions, visible_traces
        return positions

    edge_routes = final_state.edge_routes
    if edge_routes is None:
        edge_routes = []

    if trace_every > 0:
        return positions, visible_traces, edge_routes
    return positions, edge_routes


__all__ = ["build_sugiyama_pipeline", "layout_sugiyama_pipeline"]
