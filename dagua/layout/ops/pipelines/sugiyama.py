"""Sugiyama layered graph drawing expressed as a composable ops pipeline."""

from __future__ import annotations

from typing import ClassVar, List, Optional, Tuple, Union

import torch

from dagua.layout.classic.sugiyama import (
    _barycenter_ordering,
    _build_edge_routes,
    _build_neighbor_lists,
    _build_neighbor_weight_maps,
    _coordinate_assignment,
    _expand_long_edges_with_dummy_nodes,
    _longest_path_layering,
    _prepare_acyclic_edges,
    _promote_layer_assignments,
    _resolve_node_sizes,
    _validate_layout_inputs,
)
from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory


class _ValidateInputs(Op):
    """Validate Sugiyama layout inputs exactly like the classic entry point."""

    name: ClassVar[str] = "sugiyama_validate_inputs"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    writes: ClassVar[Tuple[str, ...]] = ()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run the classic validation checks on the problem inputs.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state. Unchanged by this op.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            Unmodified state after validation passes.

        Raises
        ------
        ValueError
            If edge_index, node_sizes, or edge_weights have invalid shapes.
        """
        del ctx

        _validate_layout_inputs(
            edge_index=problem.edge_index,
            num_nodes=problem.num_nodes,
            node_sizes=problem.node_sizes,
            edge_weights=problem.edge_weights,
        )
        return state


class _ResolveNodeSizes(Op):
    """Resolve node sizes to CPU float tensor for coordinate spacing."""

    name: ClassVar[str] = "sugiyama_resolve_node_sizes"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Store resolved node sizes in extras.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs containing optional node_sizes.
        state : SolveState
            Mutable solve state receiving resolved sizes.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with ``extras["sugiyama_resolved_sizes"]`` populated.
        """
        del ctx

        resolved = _resolve_node_sizes(
            node_sizes=problem.node_sizes,
            num_nodes=problem.num_nodes,
        )
        state.extras["sugiyama_resolved_sizes"] = resolved
        return state


class _PrepareAcyclicEdges(Op):
    """Break cycles to produce an acyclic edge orientation."""

    name: ClassVar[str] = "sugiyama_prepare_acyclic_edges"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run cycle removal and store the acyclic edges and reversal mask.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs containing edge_index.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with acyclic edges and reversed mask in extras.
        """
        del ctx

        acyclic_edges, reversed_mask = _prepare_acyclic_edges(
            edge_index=problem.edge_index,
            num_nodes=problem.num_nodes,
        )
        state.extras["sugiyama_acyclic_edges"] = acyclic_edges
        state.extras["sugiyama_reversed_mask"] = reversed_mask
        return state


class _AssignLayers(Op):
    """Assign nodes to layers via longest-path layering with promotion."""

    name: ClassVar[str] = "sugiyama_assign_layers"
    category: ClassVar[OpCategory] = OpCategory.LAYERING
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute longest-path layers and promote to reduce dummy nodes.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with acyclic edges already in extras.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with promoted layer assignments in extras.
        """
        del ctx

        acyclic_edges = state.extras["sugiyama_acyclic_edges"]
        layer_assignments = _longest_path_layering(
            edge_index=acyclic_edges,
            num_nodes=problem.num_nodes,
        )
        layer_assignments = _promote_layer_assignments(
            edge_index=acyclic_edges,
            layer_assignments=layer_assignments,
            num_nodes=problem.num_nodes,
        )
        state.extras["sugiyama_layer_assignments"] = layer_assignments
        return state


class _ExpandDummyNodes(Op):
    """Insert dummy nodes for edges spanning more than one layer."""

    name: ClassVar[str] = "sugiyama_expand_dummy_nodes"
    category: ClassVar[OpCategory] = OpCategory.LAYERING
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Expand long edges with dummy nodes and store the expanded graph.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with layer assignments in extras.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with expanded graph and edge weights in extras.
        """
        del ctx

        expanded_graph, expanded_edge_weights = _expand_long_edges_with_dummy_nodes(
            edge_index=state.extras["sugiyama_acyclic_edges"],
            layer_assignments=state.extras["sugiyama_layer_assignments"],
            node_sizes=state.extras["sugiyama_resolved_sizes"],
            num_original_nodes=problem.num_nodes,
            edge_weights=problem.edge_weights,
        )
        state.extras["sugiyama_expanded_graph"] = expanded_graph
        state.extras["sugiyama_expanded_edge_weights"] = expanded_edge_weights
        return state


class _BuildNeighborStructures(Op):
    """Build parent/child adjacency lists and edge-weight maps."""

    name: ClassVar[str] = "sugiyama_build_neighbor_structures"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build adjacency lists and weight maps for crossing minimization.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with expanded graph in extras.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with neighbor lists and weight maps in extras.
        """
        del ctx

        expanded_graph = state.extras["sugiyama_expanded_graph"]
        parents, children = _build_neighbor_lists(
            edge_index=expanded_graph.edge_index,
            num_nodes=expanded_graph.num_nodes,
        )
        parent_weights, child_weights = _build_neighbor_weight_maps(
            edge_index=expanded_graph.edge_index,
            num_nodes=expanded_graph.num_nodes,
            edge_weights=state.extras["sugiyama_expanded_edge_weights"],
        )
        state.extras["sugiyama_parents"] = parents
        state.extras["sugiyama_children"] = children
        state.extras["sugiyama_parent_weights"] = parent_weights
        state.extras["sugiyama_child_weights"] = child_weights
        return state


class _BarycenterOrdering(Op):
    """Minimize edge crossings via repeated barycenter sweeps."""

    name: ClassVar[str] = "sugiyama_barycenter_ordering"
    category: ClassVar[OpCategory] = OpCategory.ORDERING
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def __init__(
        self,
        barycenter_passes: int = 24,
        seed: int = 42,
        trace_every: int = 0,
    ) -> None:
        """Store barycenter sweep parameters.

        Parameters
        ----------
        barycenter_passes : int, default=24
            Number of up/down sweeps for crossing minimization.
        seed : int, default=42
            Retained for API compatibility.
        trace_every : int, default=0
            Snapshot cadence in passes. Zero disables tracing.

        Returns
        -------
        None
            This constructor stores configuration only.
        """
        self.barycenter_passes = barycenter_passes
        self.seed = seed
        self.trace_every = trace_every

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run barycenter ordering sweeps on the expanded graph.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with neighbor structures in extras.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with ordered layers and traces in extras.
        """
        del ctx

        expanded_graph = state.extras["sugiyama_expanded_graph"]
        output_device = problem.edge_index.device
        if problem.node_sizes is not None:
            output_device = problem.node_sizes.device

        rank_sep = state.extras.get("sugiyama_rank_sep", 1.0)
        node_sep = state.extras.get("sugiyama_node_sep", 1.0)

        ordered_layers, traces = _barycenter_ordering(
            layers=expanded_graph.layers,
            parents=state.extras["sugiyama_parents"],
            children=state.extras["sugiyama_children"],
            parent_weights=state.extras["sugiyama_parent_weights"],
            child_weights=state.extras["sugiyama_child_weights"],
            num_nodes=expanded_graph.num_nodes,
            num_original_nodes=problem.num_nodes,
            num_passes=self.barycenter_passes,
            seed=self.seed,
            node_sizes=expanded_graph.node_sizes,
            rank_sep=rank_sep,
            node_sep=node_sep,
            trace_every=self.trace_every,
            output_device=output_device,
        )
        state.extras["sugiyama_ordered_layers"] = ordered_layers
        state.extras["sugiyama_traces"] = traces
        return state


class _CoordinateAssignment(Op):
    """Assign (x, y) coordinates with Brandes-Kopf compaction."""

    name: ClassVar[str] = "sugiyama_coordinate_assignment"
    category: ClassVar[OpCategory] = OpCategory.COORDINATE
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run Brandes-Kopf coordinate assignment on the ordered layers.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with ordered layers in extras.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with final positions in pos (original nodes only).
        """
        del ctx

        expanded_graph = state.extras["sugiyama_expanded_graph"]
        output_device = problem.edge_index.device
        if problem.node_sizes is not None:
            output_device = problem.node_sizes.device

        rank_sep = state.extras.get("sugiyama_rank_sep", 1.0)
        node_sep = state.extras.get("sugiyama_node_sep", 1.0)

        expanded_positions = _coordinate_assignment(
            layers=state.extras["sugiyama_ordered_layers"],
            parents=state.extras["sugiyama_parents"],
            children=state.extras["sugiyama_children"],
            node_sizes=expanded_graph.node_sizes,
            num_nodes=expanded_graph.num_nodes,
            num_original_nodes=problem.num_nodes,
            rank_sep=rank_sep,
            node_sep=node_sep,
            output_device=output_device,
        )
        state.extras["sugiyama_expanded_positions"] = expanded_positions
        state.pos = expanded_positions[: problem.num_nodes]
        return state


class _BuildEdgeRoutes(Op):
    """Convert dummy-node chains into routed edge polylines."""

    name: ClassVar[str] = "sugiyama_build_edge_routes"
    category: ClassVar[OpCategory] = OpCategory.POSTPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("edge_routes",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build polyline edge routes from expanded positions and edge paths.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with expanded positions in extras.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with edge_routes populated.
        """
        del ctx

        expanded_graph = state.extras["sugiyama_expanded_graph"]
        output_device = problem.edge_index.device
        if problem.node_sizes is not None:
            output_device = problem.node_sizes.device

        state.edge_routes = _build_edge_routes(
            positions=state.extras["sugiyama_expanded_positions"],
            edge_paths=expanded_graph.edge_paths,
            reversed_edge_mask=state.extras["sugiyama_reversed_mask"],
            output_device=output_device,
        )
        return state


class _StoreSpacingParams(Op):
    """Store rank_sep and node_sep in extras for downstream ops."""

    name: ClassVar[str] = "sugiyama_store_spacing"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def __init__(self, rank_sep: float = 1.0, node_sep: float = 1.0) -> None:
        """Store spacing parameters.

        Parameters
        ----------
        rank_sep : float, default=1.0
            Vertical spacing between layers.
        node_sep : float, default=1.0
            Horizontal spacing between nodes within a layer.

        Returns
        -------
        None
            This constructor stores configuration only.
        """
        self.rank_sep = rank_sep
        self.node_sep = node_sep

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Write spacing parameters to extras.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state receiving spacing parameters.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with spacing params stored in extras.
        """
        del problem, ctx

        state.extras["sugiyama_rank_sep"] = self.rank_sep
        state.extras["sugiyama_node_sep"] = self.node_sep
        return state


def build_sugiyama_pipeline(
    rank_sep: float = 1.0,
    node_sep: float = 1.0,
    barycenter_passes: int = 24,
    seed: int = 42,
    trace_every: int = 0,
    return_edge_routes: bool = False,
) -> Pipeline:
    """Build a Sugiyama pipeline that is bit-identical to classic ``layout_sugiyama``.

    Parameters
    ----------
    rank_sep : float, default=1.0
        Vertical spacing between layers.
    node_sep : float, default=1.0
        Horizontal spacing between nodes within a layer.
    barycenter_passes : int, default=24
        Number of up/down sweeps for crossing minimization.
    seed : int, default=42
        Retained for API compatibility.
    trace_every : int, default=0
        Snapshot cadence in passes. Zero disables tracing.
    return_edge_routes : bool, default=False
        If True, include the edge route construction op.

    Returns
    -------
    Pipeline
        Pipeline that reproduces classic Sugiyama's cycle removal, layering,
        crossing minimization, and coordinate assignment phases.
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
    """Run the Sugiyama pipeline as a drop-in replacement for classic ``layout_sugiyama``.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list, shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Node sizes ``[N, 2]`` for spacing.
    rank_sep : float
        Vertical spacing between layers.
    node_sep : float
        Horizontal spacing between nodes within a layer.
    layer_sep : float, optional
        Alias for ``rank_sep``. Overrides ``rank_sep`` when provided.
    seed : int
        Retained for API compatibility.
    barycenter_passes : int
        Number of up/down sweeps for crossing minimization.
    trace_every : int
        If greater than zero, record position snapshots during barycenter
        sweeps after every ``trace_every`` full passes.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.
    return_edge_routes : bool, default=False
        If True, also return routed polylines for each input edge.

    Returns
    -------
    torch.Tensor or tuple
        Final positions ``[N, 2]`` by default. If ``trace_every > 0``, returns
        ``(positions, traces)``. If ``return_edge_routes`` is enabled, returns
        ``(positions, edge_routes)`` or ``(positions, traces, edge_routes)``.

    Raises
    ------
    ValueError
        If inputs are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    if trace_every < 0:
        raise ValueError("trace_every must be non-negative")
    if layer_sep is not None:
        rank_sep = layer_sep

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))

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
    visible_traces = [t[:num_nodes] for t in traces]

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
