"""Sugiyama layered graph-drawing pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast

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
    fidelity_dtype: torch.dtype = torch.float32,
    center_coordinates: bool = True,
    graphviz_enable_cluster_skeleton: bool = False,
) -> Pipeline:
    """Build a Sugiyama layered graph-drawing pipeline.

    Reference fidelity
    ------------------
    Targets: igraph 1.0.0 Sugiyama and Graphviz 7.0.5 dot / Sugiyama,
        Tagawa, and Toda (1981), "Methods for Visual Understanding of
        Hierarchical System Structures".
    Fidelity mode: ``fidelity_mode="igraph"`` enables igraph stable-order
        early stop and incidence-average barycenters. ``"dot"`` and
        ``"graphviz_dot"`` enable Graphviz dot median/transpose mincross.
    Verified at: final 100-seed report, strong equivalent; median RMSD 0.031
        across default, pass-count, tight, and wide variants. Round 33 dot
        bit-exact subset median remained 0.006317.
    Known divergences:
        - Graphviz dot network simplex rank assignment, mincross, x-position
          simplex, and cluster metadata are architectural residuals.
        - Edge-route reconstruction is optional and not part of the core
          coordinate fidelity contract.

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
    center_coordinates : bool, default=True
        Whether to translate the final horizontal span to be centered at zero.
    graphviz_enable_cluster_skeleton : bool, default=False
        Enable the inactive A12 cluster rank/mincross prototype.

    Returns
    -------
    Pipeline
        Pipeline implementing the Sugiyama framework. The pipeline produces
        final layered coordinates by validating inputs, storing spacing
        parameters, resolving node sizes, making the graph acyclic, assigning
        layers, expanding dummy nodes, running barycenter ordering sweeps,
        assigning coordinates, and optionally reconstructing edge routes.
    """
    if fidelity_mode not in (None, "igraph", "dot", "graphviz_dot", "graphviz"):
        raise ValueError(
            "fidelity_mode must be None, 'igraph', 'dot', 'graphviz_dot', or 'graphviz'."
        )
    use_igraph_fidelity = fidelity_mode == "igraph"
    use_graphviz_mincross = fidelity_mode in {"dot", "graphviz_dot", "graphviz"}
    use_graphviz_rank = fidelity_mode in {"dot", "graphviz_dot", "graphviz"}
    use_graphviz_xcoord = fidelity_mode == "graphviz"
    use_graphviz_node_order = fidelity_mode == "graphviz"

    ops: list[Op] = [
        _ValidateInputs(),
        _StoreSpacingParams(rank_sep=rank_sep, node_sep=node_sep),
        _ResolveNodeSizes(),
        _PrepareAcyclicEdges(),
        _AssignLayers(
            fidelity_mode=fidelity_mode if use_graphviz_rank else fidelity_mode,
            use_graphviz_cluster_skeleton=graphviz_enable_cluster_skeleton,
        ),
        _ExpandDummyNodes(use_graphviz_cluster_skeleton=graphviz_enable_cluster_skeleton),
        _BuildNeighborStructures(),
        _BarycenterOrdering(
            barycenter_passes=barycenter_passes,
            seed=seed,
            trace_every=trace_every,
            stop_when_stable=use_igraph_fidelity,
            use_incidence_barycenters=use_igraph_fidelity,
            center_coordinates=center_coordinates,
            use_graphviz_mincross=use_graphviz_mincross,
            use_graphviz_node_order=use_graphviz_node_order,
            use_graphviz_cluster_skeleton=graphviz_enable_cluster_skeleton,
        ),
        _CoordinateAssignment(
            center_coordinates=center_coordinates,
            use_graphviz_xcoord=use_graphviz_xcoord,
            use_igraph_conflicts=use_igraph_fidelity,
        ),
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
    fidelity_dtype: torch.dtype = torch.float32,
    use_node_sizes_for_spacing: Optional[bool] = None,
    center_coordinates: Optional[bool] = None,
    graphviz_node_sizes: Optional[torch.Tensor] = None,
    graphviz_edge_label_sizes: Optional[torch.Tensor] = None,
    clusters: Optional[Dict[str, Any]] = None,
    cluster_parents: Optional[Dict[str, Optional[str]]] = None,
    graphviz_apply_cluster_constraints: bool = False,
    graphviz_enable_cluster_skeleton: bool = False,
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
        Vertical center-to-center spacing between layers. Defaults to the
        classic compatibility spacing of ``1.0`` for direct calls, or
        ``config.rank_sep`` when invoked through ``LayoutConfig``.
    node_sep : float, optional
        Horizontal gap between node bounding boxes. Defaults to the classic
        compatibility spacing of ``1.0`` for direct calls, or
        ``config.node_sep`` when invoked through ``LayoutConfig``.
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
        stable-order early stop and incidence-average barycenters. ``"dot"``
        and ``"graphviz_dot"`` enable Graphviz dot median/transpose mincross.
    use_node_sizes_for_spacing : bool, optional
        Whether horizontal compaction should include node widths. Defaults to
        ``False`` in igraph fidelity mode because the igraph reference adapter
        receives topology only, and ``True`` otherwise.
    center_coordinates : bool, optional
        Whether to center final horizontal coordinates. Defaults to ``False``
        in igraph fidelity mode to match igraph's left-anchored coordinate
        frame, and ``True`` otherwise.
    graphviz_node_sizes : torch.Tensor, optional
        Point-unit Graphviz DOT node boxes with shape ``[N, 2]``. Only exact
        ``fidelity_mode="graphviz"`` consumes this override during the
        x-coordinate auxiliary solve.
    graphviz_edge_label_sizes : torch.Tensor, optional
        Point-unit Graphviz DOT edge-label boxes with shape ``[E, 2]``. Only
        exact ``fidelity_mode="graphviz"`` consumes this override when
        materializing dot's label virtual nodes.
    clusters : dict[str, Any], optional
        Cluster membership metadata from ``DaguaGraph.clusters``. Only exact
        ``fidelity_mode="graphviz"`` consumes this for Graphviz-dot cluster
        x-boundary machinery.
    cluster_parents : dict[str, str | None], optional
        Cluster hierarchy metadata from ``DaguaGraph.cluster_parents``.
    graphviz_apply_cluster_constraints : bool, default=False
        Whether Graphviz-dot cluster x-boundary machinery is allowed to consume
        ``clusters``. The benchmark wrapper enables this only for cluster-only
        DOT inputs; mixed cluster plus edge-label DOT inputs must remain on the
        pre-A9 path until the combined Graphviz machinery is ported.
    graphviz_enable_cluster_skeleton : bool, default=False
        Enable the inactive A12 cluster rank/mincross prototype. It remains
        opt-in because the x-stage integration is not benchmark-safe yet.
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
    if fidelity_mode not in (None, "igraph", "dot", "graphviz_dot", "graphviz"):
        raise ValueError(
            "fidelity_mode must be None, 'igraph', 'dot', 'graphviz_dot', or 'graphviz'."
        )
    if trace_every < 0:
        raise ValueError("trace_every must be non-negative.")
    use_igraph_fidelity = fidelity_mode == "igraph"
    if use_node_sizes_for_spacing is None:
        use_node_sizes_for_spacing = not use_igraph_fidelity
    if center_coordinates is None:
        center_coordinates = not use_igraph_fidelity
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

    if (
        use_igraph_fidelity
        and not return_edge_routes
        and trace_every == 0
        and _has_multiple_weak_components(edge_index=edge_index, num_nodes=num_nodes)
    ):
        return _layout_igraph_packed_components(
            edge_index=edge_index,
            num_nodes=num_nodes,
            node_sizes=node_sizes,
            rank_sep=rank_sep,
            node_sep=node_sep,
            seed=seed,
            barycenter_passes=barycenter_passes,
            edge_weights=edge_weights,
            use_node_sizes_for_spacing=use_node_sizes_for_spacing,
            center_coordinates=center_coordinates,
            config=config,
        )

    problem_node_sizes = node_sizes if use_node_sizes_for_spacing else None

    output_device = edge_index.device
    if problem_node_sizes is not None:
        output_device = node_sizes.device

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=problem_node_sizes,
        clusters=clusters
        if fidelity_mode == "graphviz" and graphviz_apply_cluster_constraints
        else None,
        cluster_parents=cluster_parents
        if fidelity_mode == "graphviz" and graphviz_apply_cluster_constraints
        else None,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    if graphviz_node_sizes is not None and fidelity_mode == "graphviz":
        state.extras["sugiyama_graphviz_node_sizes"] = graphviz_node_sizes.detach().to(
            device="cpu",
            dtype=torch.float32,
        )
    if graphviz_edge_label_sizes is not None and fidelity_mode == "graphviz":
        state.extras["sugiyama_graphviz_edge_label_sizes"] = graphviz_edge_label_sizes.detach().to(
            device="cpu",
            dtype=torch.float32,
        )
    ctx = RuntimeContext(plan=ExecutionPlan(device=str(output_device)))

    pipeline = build_sugiyama_pipeline(
        rank_sep=rank_sep,
        node_sep=node_sep,
        barycenter_passes=barycenter_passes,
        seed=seed,
        trace_every=trace_every,
        return_edge_routes=return_edge_routes,
        fidelity_mode=fidelity_mode,
        fidelity_dtype=fidelity_dtype,
        center_coordinates=center_coordinates,
        graphviz_enable_cluster_skeleton=graphviz_enable_cluster_skeleton,
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


def _has_multiple_weak_components(edge_index: torch.Tensor, num_nodes: int) -> bool:
    """Return whether the graph has more than one weak component.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.

    Returns
    -------
    bool
        ``True`` when at least two weak components are present.
    """
    return len(_weak_components(edge_index=edge_index, num_nodes=num_nodes)) > 1


def _weak_components(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
    """Compute weak components in deterministic node-id order.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.

    Returns
    -------
    list of list of int
        Weak components, each sorted by original node id.
    """
    if num_nodes <= 0:
        return []

    adjacency: List[List[int]] = [[] for _ in range(num_nodes)]
    edge_index_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    for source, target in zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist()):
        if source == target:
            continue
        adjacency[source].append(target)
        adjacency[target].append(source)

    seen = [False] * num_nodes
    components: List[List[int]] = []
    for start in range(num_nodes):
        if seen[start]:
            continue
        seen[start] = True
        stack = [start]
        component: List[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def _layout_igraph_packed_components(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor],
    rank_sep: float,
    node_sep: float,
    seed: int,
    barycenter_passes: int,
    edge_weights: Optional[torch.Tensor],
    use_node_sizes_for_spacing: bool,
    center_coordinates: bool,
    config: Optional["LayoutConfig"],
) -> torch.Tensor:
    """Lay out weak components independently and pack them along X.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``.
    rank_sep : float
        Vertical center-to-center spacing between layers.
    node_sep : float
        Horizontal gap between components and adjacent vertices.
    seed : int
        Seed retained for API compatibility.
    barycenter_passes : int
        Maximum number of crossing-minimization sweeps.
    edge_weights : torch.Tensor, optional
        Optional edge-weight vector with shape ``[E]``.
    use_node_sizes_for_spacing : bool
        Whether component layouts should include node widths.
    center_coordinates : bool
        Whether component layouts should center horizontal coordinates before
        the igraph-style component packing normalization.
    config : LayoutConfig, optional
        Full layout configuration supplied by the engine.

    Returns
    -------
    torch.Tensor
        Packed original-node positions with shape ``[N, 2]``.
    """
    output_device = node_sizes.device if node_sizes is not None else edge_index.device
    packed = torch.zeros((num_nodes, 2), dtype=torch.float32, device=output_device)
    edge_index_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    weights_cpu = (
        None
        if edge_weights is None
        else edge_weights.detach().to(device="cpu", dtype=torch.float32)
    )

    dx = 0.0
    for component in _weak_components(edge_index=edge_index_cpu, num_nodes=num_nodes):
        component_edges, component_weights = _slice_component_edges(
            edge_index=edge_index_cpu,
            edge_weights=weights_cpu,
            component=component,
        )
        component_sizes = node_sizes[component] if node_sizes is not None else None
        component_pos = cast(
            torch.Tensor,
            layout_sugiyama_pipeline(
                edge_index=component_edges.to(device=edge_index.device),
                num_nodes=len(component),
                node_sizes=component_sizes,
                rank_sep=rank_sep,
                node_sep=node_sep,
                seed=seed,
                barycenter_passes=barycenter_passes,
                edge_weights=(
                    None
                    if component_weights is None
                    else component_weights.to(device=edge_index.device)
                ),
                fidelity_mode="igraph",
                use_node_sizes_for_spacing=use_node_sizes_for_spacing,
                center_coordinates=center_coordinates,
                config=config,
            ),
        ).to(device=output_device, dtype=torch.float32)
        if component_pos.numel() > 0:
            component_pos[:, 0] -= float(component_pos[:, 0].min().item())
            component_pos[:, 0] += dx
            dx += float(component_pos[:, 0].max().item()) - dx + node_sep
        packed[component] = component_pos
    return packed


def _slice_component_edges(
    edge_index: torch.Tensor,
    edge_weights: Optional[torch.Tensor],
    component: List[int],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return a component-local edge list and aligned weights.

    Parameters
    ----------
    edge_index : torch.Tensor
        CPU graph connectivity tensor with shape ``[2, E]``.
    edge_weights : torch.Tensor, optional
        Optional CPU edge-weight vector with shape ``[E]``.
    component : list of int
        Original node ids in one weak component.

    Returns
    -------
    tuple
        Component-local ``edge_index`` and optional aligned edge weights.
    """
    node_to_local: Dict[int, int] = {node: index for index, node in enumerate(component)}
    sources: List[int] = []
    targets: List[int] = []
    weight_values: List[float] = []
    for edge_id, (source, target) in enumerate(zip(edge_index[0].tolist(), edge_index[1].tolist())):
        if source not in node_to_local or target not in node_to_local:
            continue
        sources.append(node_to_local[source])
        targets.append(node_to_local[target])
        if edge_weights is not None:
            weight_values.append(float(edge_weights[edge_id].item()))

    component_edges = torch.tensor([sources, targets], dtype=torch.long)
    component_weights = (
        None if edge_weights is None else torch.tensor(weight_values, dtype=torch.float32)
    )
    return component_edges, component_weights


__all__ = ["build_sugiyama_pipeline", "layout_sugiyama_pipeline"]
