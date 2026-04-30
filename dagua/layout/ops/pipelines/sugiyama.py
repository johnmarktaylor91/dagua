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

_DOT_DEFAULT_RANK_CENTER_SEP = 72.0
_DOT_DEFAULT_NODE_SEP = 18.0

if TYPE_CHECKING:
    from dagua.config import LayoutConfig


def build_sugiyama_pipeline(
    rank_sep: float = _DOT_DEFAULT_RANK_CENTER_SEP,
    node_sep: float = _DOT_DEFAULT_NODE_SEP,
    barycenter_passes: int = 24,
    seed: int = 42,
    trace_every: int = 0,
    return_edge_routes: bool = False,
    fidelity_mode: Optional[str] = None,
) -> Pipeline:
    """Build a Sugiyama layered graph-drawing pipeline.

    Parameters
    ----------
    rank_sep : float, default=72.0
        Vertical center-to-center spacing between layers. The default matches
        Graphviz dot's common 0.5 inch rank gap plus a 0.5 inch node height in
        point units.
    node_sep : float, default=18.0
        Horizontal gap between node bounding boxes, matching Graphviz dot's
        default 0.25 inch ``nodesep`` in point units.
    barycenter_passes : int, default=24
        Number of up/down sweeps for crossing minimization.
    seed : int, default=42
        Seed for deterministic tie handling.
    trace_every : int, default=0
        If positive, emit a trace every N passes.
    return_edge_routes : bool, default=False
        If ``True``, include edge-route reconstruction in the pipeline.
    fidelity_mode : str, optional
        Optional reference-compatibility mode. ``"igraph"`` enables igraph's
        stable-order early stop and incidence-average barycenters.

    Returns
    -------
    Pipeline
        Pipeline implementing the Sugiyama framework. The pipeline produces
        final layered coordinates by validating inputs, storing spacing
        parameters, resolving node sizes, making the graph acyclic, assigning
        layers, expanding dummy nodes, running barycenter ordering sweeps,
        assigning coordinates, and optionally reconstructing edge routes.
    """
    if fidelity_mode not in (None, "igraph"):
        raise ValueError("fidelity_mode must be None or 'igraph'.")
    use_igraph_fidelity = fidelity_mode == "igraph"

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
            stop_when_stable=use_igraph_fidelity,
            use_incidence_barycenters=use_igraph_fidelity,
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
    fidelity_mode: Optional[str] = None,
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
        Vertical center-to-center spacing between layers. Defaults to a
        Graphviz-dot-compatible point spacing for direct calls, or
        ``config.rank_sep`` when invoked through ``LayoutConfig``.
    node_sep : float, optional
        Horizontal gap between node bounding boxes. Defaults to Graphviz dot's
        point-unit ``nodesep`` for direct calls, or ``config.node_sep`` when
        invoked through ``LayoutConfig``.
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
    fidelity_mode : str, optional
        Optional reference-compatibility mode. ``"igraph"`` enables igraph's
        stable-order early stop and incidence-average barycenters.
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
    if fidelity_mode not in (None, "igraph"):
        raise ValueError("fidelity_mode must be None or 'igraph'.")
    if trace_every < 0:
        raise ValueError("trace_every must be non-negative.")
    if config is not None:
        if rank_sep is None:
            rank_sep = config.rank_sep
        if node_sep is None:
            node_sep = config.node_sep
    if rank_sep is None:
        rank_sep = _DOT_DEFAULT_RANK_CENTER_SEP
    if node_sep is None:
        node_sep = _DOT_DEFAULT_NODE_SEP
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
        fidelity_mode=fidelity_mode,
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
