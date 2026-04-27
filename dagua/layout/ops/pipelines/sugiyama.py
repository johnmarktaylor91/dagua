"""Sugiyama layered graph-drawing pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch

from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.state import (
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)
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

if TYPE_CHECKING:
    from dagua.config import LayoutConfig


def build_sugiyama_pipeline(
    rank_sep: float = 1.0,
    node_sep: float = 1.0,
    barycenter_passes: int = 24,
    seed: int = 42,
    trace_every: int = 0,
    return_edge_routes: bool = False,
) -> Pipeline:
    """Build a Sugiyama layered graph-drawing pipeline.

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
        Pipeline implementing the Sugiyama framework. The pipeline produces
        final layered coordinates by validating inputs, storing spacing
        parameters, resolving node sizes, making the graph acyclic, assigning
        layers, expanding dummy nodes, running barycenter ordering sweeps,
        assigning coordinates, and optionally reconstructing edge routes.
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
    rank_sep: Optional[float] = None,
    node_sep: Optional[float] = None,
    layer_sep: Optional[float] = None,
    seed: int = 42,
    barycenter_passes: int = 24,
    trace_every: int = 0,
    edge_weights: Optional[torch.Tensor] = None,
    return_edge_routes: bool = False,
    config: Optional["LayoutConfig"] = None,
) -> Union[
    torch.Tensor,
    Tuple[torch.Tensor, List[torch.Tensor]],
    Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]],
]:
    """Run the Sugiyama layered graph-drawing pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``.
    rank_sep : float, optional
        Vertical spacing between layers. Defaults to unit spacing for direct
        calls, or ``config.rank_sep`` when invoked through ``LayoutConfig``.
    node_sep : float, optional
        Horizontal spacing within layers. Defaults to unit spacing for direct
        calls, or ``config.node_sep`` when invoked through ``LayoutConfig``.
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
    config : LayoutConfig, optional
        Full layout configuration supplied by the engine. Only spacing fields
        are read by this classic pipeline.

    Returns
    -------
    torch.Tensor or tuple
        Final positions with shape ``[N, 2]`` and optional ordering traces or
        routed edge polylines depending on ``trace_every`` and
        ``return_edge_routes``.

    Raises
    ------
    ValueError
        If ``trace_every`` is negative.
    RuntimeError
        If the pipeline fails to produce final positions.
    """
    if trace_every < 0:
        raise ValueError("trace_every must be non-negative.")
    if config is not None:
        if rank_sep is None:
            rank_sep = config.rank_sep
        if node_sep is None:
            node_sep = config.node_sep
    if rank_sep is None:
        rank_sep = 1.0
    if node_sep is None:
        node_sep = 1.0
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
