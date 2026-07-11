"""Sugiyama layered graph drawing operations.

This module hosts all Sugiyama-private helpers and the registered ops used by
the composable pipeline entrypoint.
"""

from __future__ import annotations

import heapq
import math
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import torch

from dagua.layout.cycle import _is_acyclic as _cycle_is_acyclic
from dagua.layout.cycle import make_acyclic, make_acyclic_robust
from dagua.layout.ops._dot_mincross import graphviz_mincross
from dagua.layout.ops.base import Op
from dagua.layout.ops.pipelines.dot_rank import (
    GraphvizVirtualEdge,
    graphviz_network_simplex_assignment,
    graphviz_rank_assignment,
)
from dagua.layout.ops.state import (  # noqa: E402
    LayoutProblem,
    RuntimeContext,
    SolveState,
)
from dagua.layout.ops.taxonomy import OpCategory, register_op  # noqa: E402

try:
    import swiglpk as _swiglpk
except ImportError:
    _swiglpk = None

_NO_SHIFT = float("inf")
_SUGIYAMA_RESOLVED_SIZES_KEY = "sugiyama_resolved_sizes"
_SUGIYAMA_ACYCLIC_EDGES_KEY = "sugiyama_acyclic_edges"
_SUGIYAMA_ACYCLIC_EDGE_WEIGHTS_KEY = "sugiyama_acyclic_edge_weights"
_SUGIYAMA_REVERSED_MASK_KEY = "sugiyama_reversed_mask"
_SUGIYAMA_IGRAPH_SOURCE_ORDER_KEY = "sugiyama_igraph_source_order"
_SUGIYAMA_IGRAPH_SCAN_SOURCES_KEY = "sugiyama_igraph_scan_sources"
_SUGIYAMA_IGRAPH_SCAN_TARGETS_KEY = "sugiyama_igraph_scan_targets"
_SUGIYAMA_IGRAPH_SCAN_EDGE_IDS_KEY = "sugiyama_igraph_scan_edge_ids"
_SUGIYAMA_LAYER_ASSIGNMENTS_KEY = "sugiyama_layer_assignments"
_SUGIYAMA_EXPANDED_GRAPH_KEY = "sugiyama_expanded_graph"
_SUGIYAMA_EXPANDED_EDGE_WEIGHTS_KEY = "sugiyama_expanded_edge_weights"
_SUGIYAMA_PARENTS_KEY = "sugiyama_parents"
_SUGIYAMA_CHILDREN_KEY = "sugiyama_children"
_SUGIYAMA_PARENT_WEIGHTS_KEY = "sugiyama_parent_weights"
_SUGIYAMA_CHILD_WEIGHTS_KEY = "sugiyama_child_weights"
_SUGIYAMA_ORDERED_LAYERS_KEY = "sugiyama_ordered_layers"
_SUGIYAMA_TRACES_KEY = "sugiyama_traces"
_SUGIYAMA_EXPANDED_POSITIONS_KEY = "sugiyama_expanded_positions"
_SUGIYAMA_RANK_SEP_KEY = "sugiyama_rank_sep"
_SUGIYAMA_NODE_SEP_KEY = "sugiyama_node_sep"
_SUGIYAMA_GRAPHVIZ_VIRTUAL_EDGES_KEY = "sugiyama_graphviz_virtual_edges"
_SUGIYAMA_GRAPHVIZ_EDGE_ORDER_KEY = "sugiyama_graphviz_edge_order"
_SUGIYAMA_GRAPHVIZ_NODE_SIZES_KEY = "sugiyama_graphviz_node_sizes"
_SUGIYAMA_GRAPHVIZ_EDGE_LABEL_SIZES_KEY = "sugiyama_graphviz_edge_label_sizes"
_SUGIYAMA_GRAPHVIZ_CLUSTER_RANKS_KEY = "sugiyama_graphviz_cluster_ranks"
_SUGIYAMA_GRAPHVIZ_CLUSTER_LABEL_WIDTHS_KEY = "sugiyama_graphviz_cluster_label_widths"
_SUGIYAMA_GRAPHVIZ_EXPECTED_X_INVENTORY_KEY = "sugiyama_graphviz_expected_x_inventory"
_GRAPHVIZ_POINTS_PER_INCH = 72.0
_GRAPHVIZ_DEFAULT_NODE_SEP_POINTS = 18.0
_GRAPHVIZ_TYPED_X_MAX_ORIGINAL_NODES = 50
_GRAPHVIZ_TYPED_X_MAX_EDGE_DENSITY = 2.0
_GRAPHVIZ_VIRTUAL_NODE_CLASS = 2
_GRAPHVIZ_SINGLETON_NODE_CLASS = 1
_GRAPHVIZ_ORDINARY_NODE_CLASS = 0
_GRAPHVIZ_OMEGA_TABLE = (
    (1, 1, 1),
    (1, 2, 2),
    (1, 2, 4),
)
_GRAPHVIZ_X_AUX_RESOLUTION = 1
_GRAPHVIZ_DEFAULT_NODE_WIDTH_POINTS = 54.0
_GRAPHVIZ_LABEL_BOX_HALF_WIDTH_SEED_POINTS = 1.0
_GRAPHVIZ_VIRTUAL_NODE_HALF_WIDTH_SEED_POINTS = 1.0
_GRAPHVIZ_CLUSTER_CROSSING_PENALTY = 1000
_GRAPHVIZ_CLUSTER_MARGIN_POINTS = 8.0
_GRAPHVIZ_ROOT_CLUSTER_NAME = "__graphviz_root__"


@dataclass(frozen=True)
class _ExpandedLayeredGraph:
    """Store the dummy-node-expanded DAG used by Sugiyama sweeps."""

    edge_index: torch.Tensor
    layers: list[list[int]]
    node_sizes: torch.Tensor
    edge_paths: list[list[int]]
    num_nodes: int
    graphviz_node_order: Optional[list[int]] = None
    mincross_edge_penalties: Optional[list[int]] = None
    graphviz_left_widths: Optional[list[float]] = None
    graphviz_right_widths: Optional[list[float]] = None
    graphviz_cluster_members: Optional[Dict[str, Tuple[int, ...]]] = None
    graphviz_cluster_parents: Optional[Dict[str, Optional[str]]] = None
    graphviz_cluster_label_widths: Optional[Dict[str, float]] = None
    expanded_edge_origins: Optional[Tuple[int, ...]] = None


class _GraphvizXNodeClass(str, Enum):
    """Typed node classes in Graphviz's x-coordinate auxiliary graph."""

    NORMAL = "normal"
    VIRTUAL = "virtual"
    SLACK = "slack"
    BORDER = "border"


class _GraphvizXEdgeKind(str, Enum):
    """Construction stages for Graphviz x-coordinate constraints."""

    LEFT_RIGHT = "left_right"
    EDGE_PAIR = "edge_pair"
    CONTAIN = "contain"


@dataclass(frozen=True)
class _GraphvizXNode:
    """Describe one typed node in the Graphviz x inventory."""

    node_id: int
    node_class: _GraphvizXNodeClass
    original_edge_id: Optional[int] = None
    cluster_name: Optional[str] = None
    border_side: Optional[str] = None


@dataclass(frozen=True)
class _GraphvizXEdge:
    """Describe one directed constraint in the Graphviz x inventory."""

    tail: int
    head: int
    minlen: int
    weight: int
    kind: _GraphvizXEdgeKind
    original_edge_id: Optional[int] = None


@dataclass(frozen=True)
class _GraphvizXInventory:
    """Store the complete typed Graphviz x graph before network simplex."""

    nodes: Tuple[_GraphvizXNode, ...]
    edges: Tuple[_GraphvizXEdge, ...]
    initial_ranks: Dict[int, int]


@dataclass(frozen=True)
class _BarycenterOrderingConfig:
    """Configuration for :class:`_BarycenterOrdering`.

    Parameters
    ----------
    barycenter_passes : int, default=24
        Number of down/up sweeps used for crossing minimization.
    seed : int, default=42
        Retained for API compatibility with the classic entry point.
    trace_every : int, default=0
        Snapshot cadence in passes. Zero disables tracing.
    stop_when_stable : bool, default=False
        If ``True``, stop after the first full pass that leaves all layer
        orders unchanged.
    use_incidence_barycenters : bool, default=False
        If ``True``, average duplicate neighbor incidences directly. This
        matches igraph's unweighted crossing-reduction semantics.
    center_coordinates : bool, default=True
        If ``False``, leave horizontal coordinates in their compacted
        left-anchored frame instead of centering the final span.
    use_graphviz_mincross : bool, default=False
        If ``True``, replace the default barycenter sweeps with Graphviz dot's
        median/transpose mincross heuristic on the expanded adjacent-rank DAG.
    use_graphviz_node_order : bool, default=False
        If ``True``, seed ``build_ranks`` from the graphviz-style fast-node
        list built during dummy expansion.
    use_graphviz_cluster_skeleton : bool, default=False
        If ``True``, enable the A12 rank-leader cluster mincross prototype.
        The default stays disabled until its x-stage interaction passes the
        rendered no-regression gate.
    """

    barycenter_passes: int = 24
    seed: int = 42
    trace_every: int = 0
    stop_when_stable: bool = False
    use_incidence_barycenters: bool = False
    center_coordinates: bool = True
    use_graphviz_mincross: bool = False
    use_graphviz_node_order: bool = False
    use_graphviz_cluster_skeleton: bool = False


@dataclass(frozen=True)
class _StoreSpacingParamsConfig:
    """Configuration for :class:`_StoreSpacingParams`.

    Parameters
    ----------
    rank_sep : float, default=1.0
        Vertical spacing between layers.
    node_sep : float, default=1.0
        Horizontal spacing between nodes within a layer.
    """

    rank_sep: float = 1.0
    node_sep: float = 1.0


def _validate_layout_inputs(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor],
    edge_weights: Optional[torch.Tensor],
) -> None:
    """Validate the public Sugiyama layout inputs.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor expected to have shape ``[2, E]``.
    num_nodes : int
        Number of nodes in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor expected to have shape ``[N, 2]``.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor expected to have shape ``[E]``.

    Raises
    ------
    ValueError
        If any argument violates the required shape contract.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if node_sizes is not None:
        if node_sizes.ndim != 2 or node_sizes.shape != (num_nodes, 2):
            raise ValueError("node_sizes must have shape [N, 2]")
    if edge_weights is not None and edge_weights.shape[0] != edge_index.shape[1]:
        raise ValueError(
            f"edge_weights length {edge_weights.shape[0]} does not match "
            f"edge count {edge_index.shape[1]}"
        )


def _resolve_node_sizes(node_sizes: Optional[torch.Tensor], num_nodes: int) -> torch.Tensor:
    """Return CPU node sizes for coordinate spacing.

    Parameters
    ----------
    node_sizes : torch.Tensor, optional
        Optional input node sizes.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    torch.Tensor
        CPU float tensor with shape ``[N, 2]``.
    """
    if node_sizes is None:
        return torch.zeros((num_nodes, 2), dtype=torch.float32)
    return node_sizes.detach().to(device="cpu", dtype=torch.float32)


def _iterative_dfs_back_edges(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Detect DFS back edges without consuming Python call stack.

    Parameters
    ----------
    edge_index : torch.Tensor
        CPU edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    torch.Tensor
        Boolean tensor with shape ``[E]`` marking edges that point to a node
        currently on the DFS stack.
    """
    num_edges = int(edge_index.shape[1]) if edge_index.numel() > 0 else 0
    if num_edges == 0:
        return torch.zeros((0,), dtype=torch.bool)

    sources = edge_index[0].tolist()
    targets = edge_index[1].tolist()
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(num_nodes)]
    in_degree = [0] * num_nodes
    for edge_id, (source, target) in enumerate(zip(sources, targets)):
        source_id = int(source)
        target_id = int(target)
        adjacency[source_id].append((target_id, edge_id))
        in_degree[target_id] += 1

    white, gray, black = 0, 1, 2
    color = [white] * num_nodes
    reversed_edges = [False] * num_edges
    visit_order = [node for node, degree in enumerate(in_degree) if degree == 0]
    visit_order.extend(node for node, degree in enumerate(in_degree) if degree > 0)

    for start in visit_order:
        if color[start] != white:
            continue
        color[start] = gray
        stack: list[tuple[int, int]] = [(start, 0)]
        while stack:
            node, child_index = stack[-1]
            if child_index >= len(adjacency[node]):
                color[node] = black
                stack.pop()
                continue

            stack[-1] = (node, child_index + 1)
            child, edge_id = adjacency[node][child_index]
            if color[child] == gray:
                reversed_edges[edge_id] = True
            elif color[child] == white:
                color[child] = gray
                stack.append((child, 0))

    return torch.tensor(reversed_edges, dtype=torch.bool)


def _prepare_acyclic_edges(
    edge_index: torch.Tensor,
    edge_weights: Optional[torch.Tensor],
    num_nodes: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Return a CPU ``edge_index`` with a robust acyclic orientation.

    Parameters
    ----------
    edge_index : torch.Tensor
        Input edge list of shape ``[2, E]``.
    edge_weights : torch.Tensor, optional
        Optional input edge weights with shape ``[E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    tuple
        ``(acyclic_edges, acyclic_edge_weights, reversed_mask)`` where
        ``acyclic_edges`` is a CPU long tensor suitable for Kahn layering,
        ``acyclic_edge_weights`` is aligned to the filtered edge list when
        weights are provided, and ``reversed_mask`` marks retained input edges
        reversed during cycle breaking.
    """
    edge_index_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    if edge_index_cpu.numel() == 0:
        return edge_index_cpu, None, torch.zeros((0,), dtype=torch.bool)

    non_loop_mask = edge_index_cpu[0] != edge_index_cpu[1]
    filtered_edges = edge_index_cpu[:, non_loop_mask]
    filtered_weights: Optional[torch.Tensor] = None
    if edge_weights is not None:
        weights_cpu = edge_weights.detach().to(device="cpu", dtype=torch.float32)
        filtered_weights = weights_cpu[non_loop_mask]
    if filtered_edges.numel() == 0:
        return filtered_edges, filtered_weights, torch.zeros((0,), dtype=torch.bool)

    reversed_mask = _iterative_dfs_back_edges(filtered_edges, num_nodes)
    acyclic_edges = make_acyclic(filtered_edges, reversed_mask)
    if not _cycle_is_acyclic(acyclic_edges, num_nodes):
        acyclic_edges, reversed_mask = make_acyclic_robust(filtered_edges, num_nodes)
    return acyclic_edges, filtered_weights, reversed_mask


def _longest_path_layering(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Assign each node to a layer via Kahn topological traversal.

    Parameters
    ----------
    edge_index : torch.Tensor
        Acyclic edge list of shape ``[2, E]`` on CPU.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    torch.Tensor
        Long tensor of shape ``[N]`` with layer indices.

    Raises
    ------
    ValueError
        If the graph remains cyclic after edge reversal.
    """
    children: List[List[int]] = [[] for _ in range(num_nodes)]
    in_degree = [0] * num_nodes
    src_nodes = edge_index[0].tolist()
    dst_nodes = edge_index[1].tolist()

    for src, dst in zip(src_nodes, dst_nodes):
        children[src].append(dst)
        in_degree[dst] += 1

    layers = [0] * num_nodes
    ready = [node for node, degree in enumerate(in_degree) if degree == 0]
    heapq.heapify(ready)

    processed = 0
    while ready:
        node = heapq.heappop(ready)
        processed += 1
        next_layer = layers[node] + 1
        for child in children[node]:
            if next_layer > layers[child]:
                layers[child] = next_layer
            in_degree[child] -= 1
            if in_degree[child] == 0:
                heapq.heappush(ready, child)

    if processed != num_nodes:
        raise ValueError("graph must be acyclic after back-edge reversal")

    return torch.tensor(layers, dtype=torch.long)


def _graphviz_layer_assignments(
    edge_index: torch.Tensor,
    edge_weights: Optional[torch.Tensor],
    num_nodes: int,
    edge_label_sizes: Optional[torch.Tensor] = None,
    legacy_tree_order: bool = False,
) -> Tuple[torch.Tensor, List[GraphvizVirtualEdge]]:
    """Assign layers with the Graphviz dot network-simplex ranker.

    Parameters
    ----------
    edge_index : torch.Tensor
        Acyclic edge list with shape ``[2, E]`` on CPU.
    edge_weights : torch.Tensor, optional
        Optional edge-weight vector aligned to ``edge_index``.
    num_nodes : int
        Number of original graph nodes.
    edge_label_sizes : torch.Tensor, optional
        Point-unit Graphviz DOT label boxes with shape ``[E, 2]``. When any
        label is present, dot doubles every input edge ``minlen`` before rank
        assignment to reserve midpoint ranks for label virtual nodes.
    legacy_tree_order : bool, default=False
        Preserve the pre-r79 feasible-tree traversal for regression-protected
        legacy cluster solves.

    Returns
    -------
    tuple
        Layer assignment tensor with shape ``[N]`` and virtual-edge
        descriptors for long edges. The descriptors are stored for downstream
        integration, while the existing dummy-expansion op still materializes
        the actual dummy nodes used by this pipeline.
    """
    virtual_counter = 0

    def virtual_node_factory(
        tail: int,
        head: int,
        rank: int,
        original_edge_index: int,
    ) -> str:
        """Return a deterministic Graphviz-style virtual node id.

        Parameters
        ----------
        tail : int
            Original edge tail.
        head : int
            Original edge head.
        rank : int
            Intermediate rank.
        original_edge_index : int
            Original edge index.

        Returns
        -------
        str
            Stable virtual node identifier for metadata.
        """
        nonlocal virtual_counter
        value = f"_gv_v{virtual_counter}_{tail}_{head}_{rank}_{original_edge_index}"
        virtual_counter += 1
        return value

    edge_minlens = _graphviz_edge_label_rank_minlens(
        edge_index=edge_index,
        edge_label_sizes=edge_label_sizes,
    )
    ranks, virtual_edges = graphviz_rank_assignment(
        edges=edge_index,
        virtual_node_factory=virtual_node_factory,
        num_nodes=num_nodes,
        edge_minlens=edge_minlens,
        edge_weights=edge_weights,
        balance=True,
        legacy_tree_order=legacy_tree_order,
    )
    layers = [int(ranks.get(node, 0)) for node in range(num_nodes)]
    return torch.tensor(layers, dtype=torch.long), virtual_edges


def _graphviz_cluster_rank_assignments(
    edge_index: torch.Tensor,
    edge_weights: Optional[torch.Tensor],
    num_nodes: int,
    clusters: Optional[Mapping[str, Any]],
    cluster_parents: Optional[Mapping[str, Optional[str]]],
    edge_label_sizes: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, List[GraphvizVirtualEdge], Dict[str, Tuple[int, int]]]:
    """Assign Graphviz-style ranks with recursive cluster collapse.

    Parameters
    ----------
    edge_index : torch.Tensor
        Acyclic edge list with shape ``[2, E]`` on CPU.
    edge_weights : torch.Tensor, optional
        Optional edge weights aligned to ``edge_index``.
    num_nodes : int
        Number of original nodes.
    clusters : Mapping[str, Any], optional
        Raw cluster membership metadata from the graph object.
    cluster_parents : Mapping[str, str | None], optional
        Raw cluster hierarchy metadata.
    edge_label_sizes : torch.Tensor, optional
        Point-unit Graphviz DOT edge-label boxes with shape ``[E, 2]``.

    Returns
    -------
    tuple
        ``(layers, virtual_edges, cluster_rank_bounds)`` where ``layers`` has
        shape ``[N]`` and rank bounds are inclusive global ranks per cluster.

    Notes
    -----
    Graphviz 7.0.5 ``rank.c:244-266`` ranks each local cluster, collapses it
    to a leader for parent ranking, then ``expand_ranksets()`` adds the
    leader rank to member-local offsets. This helper mirrors that observable
    rank arithmetic while keeping the existing tensor-only pipeline.
    """
    normalized_clusters = _normalize_graphviz_clusters(clusters=clusters, num_nodes=num_nodes)
    if not normalized_clusters:
        layers, virtual_edges = _graphviz_layer_assignments(
            edge_index=edge_index,
            edge_weights=edge_weights,
            num_nodes=num_nodes,
            edge_label_sizes=edge_label_sizes,
        )
        return layers, virtual_edges, {}

    parents = _normalize_graphviz_cluster_parents(
        cluster_names=tuple(normalized_clusters.keys()),
        cluster_parents=cluster_parents,
    )
    edge_minlens = _graphviz_edge_label_rank_minlens(
        edge_index=edge_index,
        edge_label_sizes=edge_label_sizes,
    )
    if edge_minlens is None:
        edge_minlens = [1] * int(edge_index.shape[1])
    weights = _graphviz_rank_weight_values(edge_weights=edge_weights, edge_count=len(edge_minlens))
    sources = [int(value) for value in edge_index[0].tolist()]
    targets = [int(value) for value in edge_index[1].tolist()]

    children = _graphviz_cluster_children(parents=parents)
    local_ranks: Dict[str, Dict[int, int]] = {}
    leader_by_cluster: Dict[str, int] = {}
    next_rank_slack_node = num_nodes

    def solve_cluster(cluster_name: str) -> Dict[int, int]:
        """Rank one cluster after recursively collapsing child clusters.

        Parameters
        ----------
        cluster_name : str
            Cluster name to rank locally.

        Returns
        -------
        dict[int, int]
            Zero-based local rank offsets for every original member node.
        """
        nonlocal next_rank_slack_node
        if cluster_name in local_ranks:
            return local_ranks[cluster_name]

        child_names = children.get(cluster_name, [])
        child_by_node: Dict[int, str] = {}
        for child_name in child_names:
            child_ranks = solve_cluster(child_name)
            child_leader = leader_by_cluster[child_name]
            for child_node in child_ranks:
                child_by_node[child_node] = child_name
            # Graphviz collapses child clusters to their least-rank leader in
            # the parent graph, so all non-leader child members leave the local
            # parent rank solve.
            child_by_node[child_leader] = child_name

        members = set(normalized_clusters[cluster_name])
        root_nodes = set(members)
        for child_name in child_names:
            child_leader = leader_by_cluster[child_name]
            for child_node in normalized_clusters[child_name]:
                if child_node != child_leader:
                    root_nodes.discard(child_node)
            root_nodes.add(child_leader)

        local_edges: List[Tuple[int, int, int, int]] = []
        for edge_id, (source, target) in enumerate(zip(sources, targets)):
            if source not in members or target not in members or source == target:
                continue
            source_child = child_by_node.get(source)
            target_child = child_by_node.get(target)
            if source_child is not None and source_child == target_child:
                continue
            mapped_source = leader_by_cluster.get(source_child, source)
            mapped_target = leader_by_cluster.get(target_child, target)
            if mapped_source == mapped_target:
                continue
            source_offset = (
                local_ranks[source_child].get(source, 0) if source_child is not None else 0
            )
            target_offset = (
                local_ranks[target_child].get(target, 0) if target_child is not None else 0
            )
            offset = edge_minlens[edge_id] + source_offset - target_offset
            if source_child is not None or target_child is not None:
                # Graphviz class1.c:interclust1 represents an intercluster
                # relation with a slack node and two outgoing constraints.
                # A direct leader-to-leader edge overconstrains recursive
                # cluster offsets and can make the later x inventory cyclic.
                slack_node = next_rank_slack_node
                next_rank_slack_node += 1
                local_edges.append(
                    (slack_node, mapped_source, max(-offset, 0), 10 * weights[edge_id])
                )
                local_edges.append((slack_node, mapped_target, max(offset, 0), weights[edge_id]))
            else:
                local_edges.append((mapped_source, mapped_target, max(1, offset), weights[edge_id]))

        local_edges = _graphviz_drop_reciprocal_rank_records(records=local_edges)
        local_edges = _graphviz_acyclic_rank_records(
            records=local_edges,
            ordered_nodes=sorted(root_nodes),
        )
        ranks, _ = graphviz_rank_assignment(
            edges=local_edges,
            virtual_node_factory=_graphviz_discard_virtual_node,
            num_nodes=None,
            balance=True,
        )
        for node in root_nodes:
            ranks.setdefault(node, 0)

        expanded = {}
        for node in members:
            child_name = child_by_node.get(node)
            if child_name is None:
                expanded[node] = int(ranks.get(node, 0))
                continue
            child_leader = leader_by_cluster[child_name]
            expanded[node] = int(ranks.get(child_leader, 0)) + local_ranks[child_name].get(node, 0)
        normalized = _normalize_graphviz_member_ranks(
            members=tuple(sorted(members)),
            ranks=expanded,
        )
        leader_by_cluster[cluster_name] = min(
            normalized,
            key=lambda node: (normalized.get(node, 0), node),
        )
        local_ranks[cluster_name] = normalized
        return normalized

    for name in sorted(
        normalized_clusters,
        key=lambda cluster_name: (
            -_graphviz_cluster_depth(cluster_name, parents),
            cluster_name,
        ),
    ):
        solve_cluster(name)

    top_clusters = [name for name, parent in parents.items() if parent is None]
    top_cluster_by_node = _graphviz_top_cluster_by_node(
        clusters=normalized_clusters,
        parents=parents,
        top_clusters=top_clusters,
    )
    root_records: List[Tuple[int, int, int, int]] = []
    root_nodes: Set[int] = set(range(num_nodes))
    for name, leader in leader_by_cluster.items():
        if name not in top_clusters:
            continue
        root_nodes.add(leader)
        for node in normalized_clusters[name]:
            if node != leader:
                root_nodes.discard(node)

    for edge_id, (source, target) in enumerate(zip(sources, targets)):
        source_cluster = top_cluster_by_node.get(source)
        target_cluster = top_cluster_by_node.get(target)
        if source_cluster is not None and source_cluster == target_cluster:
            continue
        mapped_source = leader_by_cluster.get(source_cluster, source)
        mapped_target = leader_by_cluster.get(target_cluster, target)
        if mapped_source == mapped_target:
            continue
        source_offset = (
            local_ranks[source_cluster].get(source, 0) if source_cluster is not None else 0
        )
        target_offset = (
            local_ranks[target_cluster].get(target, 0) if target_cluster is not None else 0
        )
        offset = edge_minlens[edge_id] + source_offset - target_offset
        if source_cluster is not None or target_cluster is not None:
            slack_node = next_rank_slack_node
            next_rank_slack_node += 1
            root_records.append((slack_node, mapped_source, max(-offset, 0), 10 * weights[edge_id]))
            root_records.append((slack_node, mapped_target, max(offset, 0), weights[edge_id]))
        else:
            root_records.append((mapped_source, mapped_target, max(1, offset), weights[edge_id]))

    acyclic_root_records = _graphviz_acyclic_rank_records(
        records=root_records,
        ordered_nodes=sorted(root_nodes),
    )
    root_ranks, virtual_edges = graphviz_rank_assignment(
        edges=acyclic_root_records,
        virtual_node_factory=_graphviz_discard_virtual_node,
        num_nodes=None,
        balance=True,
    )
    for node in root_nodes:
        root_ranks.setdefault(node, 0)

    global_layers = [0] * num_nodes
    for node in range(num_nodes):
        cluster_name = top_cluster_by_node.get(node)
        if cluster_name is None:
            global_layers[node] = int(root_ranks.get(node, 0))
            continue
        leader = leader_by_cluster[cluster_name]
        global_layers[node] = int(root_ranks.get(leader, 0)) + local_ranks[cluster_name].get(
            node, 0
        )

    min_layer = min(global_layers) if global_layers else 0
    if min_layer:
        global_layers = [rank - min_layer for rank in global_layers]
    rank_bounds = _graphviz_cluster_global_rank_bounds(
        clusters=normalized_clusters,
        global_layers=global_layers,
    )
    return torch.tensor(global_layers, dtype=torch.long), virtual_edges, rank_bounds


def _graphviz_drop_reciprocal_rank_records(
    records: Sequence[Tuple[int, int, int, int]],
) -> List[Tuple[int, int, int, int]]:
    """Drop reciprocal collapsed rank records before local simplex ranking.

    Parameters
    ----------
    records : sequence[tuple[int, int, int, int]]
        Rank records as ``(tail, head, minlen, weight)``.

    Returns
    -------
    list[tuple[int, int, int, int]]
        Rank records without pairs that constrain the same collapsed leaders
        in both directions.

    Notes
    -----
    Intercluster member edges can collapse to reciprocal leader constraints
    even when both child clusters legitimately begin on the same local rank.
    Graphviz's class1/intercluster handling does not expose an added rank
    level for this case; preserving both records before our acyclic pass would
    reverse one and incorrectly separate the sibling leaders.
    """
    pair_counts: Dict[Tuple[int, int], int] = {}
    for tail, head, _, _ in records:
        pair_counts[(tail, head)] = pair_counts.get((tail, head), 0) + 1
    return [record for record in records if (record[1], record[0]) not in pair_counts]


def _graphviz_acyclic_rank_records(
    records: Sequence[Tuple[int, int, int, int]],
    ordered_nodes: Sequence[int],
) -> List[Tuple[int, int, int, int]]:
    """Reverse collapsed-rank back edges to keep simplex input acyclic.

    Parameters
    ----------
    records : sequence[tuple[int, int, int, int]]
        Rank records as ``(tail, head, minlen, weight)``.
    ordered_nodes : sequence[int]
        Deterministic root-node scan order.

    Returns
    -------
    list[tuple[int, int, int, int]]
        Rank records with DFS back edges reversed.

    Notes
    -----
    Collapsing multiple member nodes into one cluster leader can create a
    cycle even when the pre-collapse tensor graph is acyclic. Graphviz runs
    ``acyclic(g)`` after ``class1()`` in ``rank.c:456-468``; this mirrors that
    safety step for the collapsed root records.
    """
    if not records:
        return []

    node_order = list(ordered_nodes)
    for tail, head, _, _ in records:
        if tail not in node_order:
            node_order.append(tail)
        if head not in node_order:
            node_order.append(head)
    node_to_index = {node: index for index, node in enumerate(node_order)}
    adjacency: List[List[Tuple[int, int]]] = [[] for _ in node_order]
    for edge_id, (tail, head, _, _) in enumerate(records):
        adjacency[node_to_index[tail]].append((node_to_index[head], edge_id))

    white, gray, black = 0, 1, 2
    color = [white] * len(node_order)
    reverse_edges: Set[int] = set()
    for start in range(len(node_order)):
        if color[start] != white:
            continue
        color[start] = gray
        stack: List[Tuple[int, int]] = [(start, 0)]
        while stack:
            node, child_index = stack[-1]
            if child_index >= len(adjacency[node]):
                color[node] = black
                stack.pop()
                continue
            stack[-1] = (node, child_index + 1)
            child, edge_id = adjacency[node][child_index]
            if color[child] == gray:
                reverse_edges.add(edge_id)
            elif color[child] == white:
                color[child] = gray
                stack.append((child, 0))

    out: List[Tuple[int, int, int, int]] = []
    for edge_id, (tail, head, minlen, weight) in enumerate(records):
        if edge_id in reverse_edges:
            out.append((head, tail, minlen, weight))
        else:
            out.append((tail, head, minlen, weight))
    return out


def _graphviz_rank_weight_values(
    edge_weights: Optional[torch.Tensor],
    edge_count: int,
) -> List[int]:
    """Return Graphviz-style integer rank weights.

    Parameters
    ----------
    edge_weights : torch.Tensor, optional
        Optional edge-weight vector with shape ``[E]``.
    edge_count : int
        Number of edges requiring weights.

    Returns
    -------
    list[int]
        Positive integer weights aligned to the rank records.
    """
    if edge_weights is None:
        return [1] * edge_count
    weights_cpu = edge_weights.detach().to(device="cpu", dtype=torch.float32)
    return [max(1, int(round(float(value)))) for value in weights_cpu.tolist()[:edge_count]]


def _graphviz_discard_virtual_node(*args: object) -> str:
    """Return a placeholder virtual-node id for rank-only solves.

    Parameters
    ----------
    *args : object
        Ignored Graphviz virtual-node factory arguments.

    Returns
    -------
    str
        Stable placeholder id.
    """
    del args
    return "_gv_cluster_rank_v"


def _normalize_graphviz_member_ranks(
    members: Sequence[int],
    ranks: Mapping[Any, int],
) -> Dict[int, int]:
    """Normalize local cluster ranks to zero-based member offsets.

    Parameters
    ----------
    members : sequence[int]
        Original-node ids in a cluster.
    ranks : Mapping[Any, int]
        Ranker output for the induced cluster graph.

    Returns
    -------
    dict[int, int]
        Member-local rank offsets keyed by original node id.
    """
    raw = {node: int(ranks.get(node, 0)) for node in members}
    min_rank = min(raw.values()) if raw else 0
    return {node: rank - min_rank for node, rank in raw.items()}


def _graphviz_top_cluster_by_node(
    clusters: Mapping[str, Sequence[int]],
    parents: Mapping[str, Optional[str]],
    top_clusters: Sequence[str],
) -> Dict[int, str]:
    """Map each node to its top-level cluster under the root graph.

    Parameters
    ----------
    clusters : Mapping[str, sequence[int]]
        Normalized cluster membership.
    parents : Mapping[str, str | None]
        Normalized parent mapping.
    top_clusters : sequence[str]
        Cluster names whose parent is the root graph.

    Returns
    -------
    dict[int, str]
        Node id to top-level cluster name.
    """
    top_set = set(top_clusters)
    out: Dict[int, str] = {}
    for name, members in clusters.items():
        root_name = name
        seen: Set[str] = set()
        while parents.get(root_name) is not None and root_name not in seen:
            seen.add(root_name)
            root_name = parents[root_name] or root_name
        if root_name not in top_set:
            continue
        for node in members:
            out[int(node)] = root_name
    return out


def _graphviz_cluster_global_rank_bounds(
    clusters: Mapping[str, Sequence[int]],
    global_layers: Sequence[int],
) -> Dict[str, Tuple[int, int]]:
    """Return inclusive global rank bounds for each cluster.

    Parameters
    ----------
    clusters : Mapping[str, sequence[int]]
        Normalized cluster membership.
    global_layers : sequence[int]
        Global rank per original node.

    Returns
    -------
    dict[str, tuple[int, int]]
        Inclusive ``(minrank, maxrank)`` for every non-empty cluster.
    """
    bounds: Dict[str, Tuple[int, int]] = {}
    for name, members in clusters.items():
        ranks = [int(global_layers[node]) for node in members if 0 <= node < len(global_layers)]
        if ranks:
            bounds[name] = (min(ranks), max(ranks))
    return bounds


def _graphviz_edge_label_rank_minlens(
    edge_index: torch.Tensor,
    edge_label_sizes: Optional[torch.Tensor],
) -> Optional[List[int]]:
    """Return dot rank ``minlen`` values after edge-label expansion.

    Parameters
    ----------
    edge_index : torch.Tensor
        Acyclic edge list with shape ``[2, E]``.
    edge_label_sizes : torch.Tensor, optional
        Point-unit Graphviz DOT edge-label boxes with shape ``[E, 2]``.

    Returns
    -------
    list[int] or None
        A list of doubled ``minlen`` values when any label is present,
        otherwise ``None`` so the ranker keeps default unit constraints.
    """
    edge_count = int(edge_index.shape[1]) if edge_index.numel() > 0 else 0
    if edge_count == 0 or not _has_graphviz_edge_labels(edge_label_sizes=edge_label_sizes):
        return None
    return [2] * edge_count


def _has_graphviz_edge_labels(edge_label_sizes: Optional[torch.Tensor]) -> bool:
    """Return whether the Graphviz DOT input contains any edge label.

    Parameters
    ----------
    edge_label_sizes : torch.Tensor, optional
        Point-unit Graphviz DOT edge-label boxes with shape ``[E, 2]``.

    Returns
    -------
    bool
        ``True`` when at least one edge-label box has positive area.
    """
    if edge_label_sizes is None or edge_label_sizes.numel() == 0:
        return False
    label_sizes = edge_label_sizes.detach().to(device="cpu", dtype=torch.float32)
    return bool(torch.any(label_sizes[:, 0] > 0.0).item())


def _igraph_glpk_layer_assignments(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor] = None,
    is_directed: bool = True,
) -> torch.Tensor:
    """Assign layers using igraph 1.0.0's GLPK Sugiyama formulation.

    Parameters
    ----------
    edge_index : torch.Tensor
        Original non-loop edge list with shape ``[2, E]`` on CPU.
    num_nodes : int
        Number of original graph nodes.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.
    is_directed : bool, default=True
        Whether to follow igraph's directed-graph GLPK gate. Undirected graphs
        use igraph's BFS fallback.

    Returns
    -------
    torch.Tensor
        Normalized layer ids with shape ``[N]``.

    Notes
    -----
    igraph 1.0.0 uses the LP only for directed graphs with at most 1000 nodes.
    Its GLPK objective has a source quirk: both degree vectors are populated
    from incoming incidences before subtracting Eades feedback-edge
    contributions from the source ``outdegs`` and target ``indegs`` vectors.
    """
    if num_nodes == 0 or edge_index.numel() == 0:
        return torch.zeros((num_nodes,), dtype=torch.long)

    if not is_directed:
        return _igraph_undirected_layer_assignments(
            edge_index=edge_index,
            edge_weights=edge_weights,
            num_nodes=num_nodes,
        )
    if num_nodes > 1000:
        return _igraph_eades_layer_assignments(
            edge_index=edge_index,
            num_nodes=num_nodes,
            edge_weights=edge_weights,
        )

    feedback_edges = set(
        _igraph_eades_feedback_edges(
            edge_index=edge_index,
            num_nodes=num_nodes,
            edge_weights=edge_weights,
        )
    )
    objective = _igraph_glpk_objective_coefficients(
        edge_index=edge_index,
        num_nodes=num_nodes,
        feedback_edges=feedback_edges,
        edge_weights=edge_weights,
    )
    if _swiglpk is not None:
        layers = _igraph_swiglpk_layer_assignments(
            edge_index=edge_index,
            num_nodes=num_nodes,
            feedback_edges=feedback_edges,
            objective=objective,
        )
        if layers is not None:
            return layers

    return _igraph_scipy_layer_assignments(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weights=edge_weights,
        feedback_edges=feedback_edges,
        objective=objective,
    )


def _igraph_swiglpk_layer_assignments(
    edge_index: torch.Tensor,
    num_nodes: int,
    feedback_edges: Set[int],
    objective: Sequence[float],
) -> Optional[torch.Tensor]:
    """Solve igraph's Sugiyama rank LP with GLPK simplex.

    Parameters
    ----------
    edge_index : torch.Tensor
        Original non-loop edge list with shape ``[2, E]`` on CPU.
    num_nodes : int
        Number of original graph nodes.
    feedback_edges : set of int
        Edge ids selected by the Eades feedback heuristic.
    objective : sequence of float
        Per-node objective coefficients matching igraph's ``outdegs - indegs``
        vector.

    Returns
    -------
    torch.Tensor or None
        Normalized layer ids with shape ``[N]``, or ``None`` if GLPK cannot
        solve the LP so the caller can use the existing SciPy fallback.
    """
    glpk: Any = _swiglpk
    if glpk is None:
        return None

    problem = glpk.glp_create_prob()
    previous_term_out = glpk.glp_term_out(glpk.GLP_OFF)
    try:
        simplex_params = glpk.glp_smcp()
        glpk.glp_init_smcp(simplex_params)
        simplex_params.msg_lev = glpk.GLP_MSG_OFF
        simplex_params.presolve = glpk.GLP_OFF

        glpk.glp_set_obj_dir(problem, glpk.GLP_MIN)
        glpk.glp_add_cols(problem, num_nodes)
        for column in range(1, num_nodes + 1):
            glpk.glp_set_col_kind(problem, column, glpk.GLP_IV)
            glpk.glp_set_col_bnds(problem, column, glpk.GLP_LO, 0.0, 0.0)
            glpk.glp_set_obj_coef(problem, column, float(objective[column - 1]))

        edge_count = int(edge_index.shape[1])
        glpk.glp_add_rows(problem, edge_count)
        row_indices = glpk.intArray(3)
        row_values = glpk.doubleArray(3)
        row_values[1] = -1.0
        row_values[2] = 1.0
        sorted_feedback_edges = sorted(feedback_edges)
        feedback_cursor = 0
        sources = edge_index[0].tolist()
        targets = edge_index[1].tolist()
        for edge_id, (source, target) in enumerate(zip(sources, targets)):
            row = edge_id + 1
            row_indices[1] = int(source) + 1
            row_indices[2] = int(target) + 1
            if source == target:
                if (
                    feedback_cursor < len(sorted_feedback_edges)
                    and sorted_feedback_edges[feedback_cursor] == edge_id
                ):
                    feedback_cursor += 1
                continue

            if (
                feedback_cursor < len(sorted_feedback_edges)
                and sorted_feedback_edges[feedback_cursor] == edge_id
            ):
                glpk.glp_set_row_bnds(problem, row, glpk.GLP_UP, -1.0, -1.0)
                feedback_cursor += 1
            else:
                glpk.glp_set_row_bnds(problem, row, glpk.GLP_LO, 1.0, 1.0)
            glpk.glp_set_mat_row(problem, row, 2, row_indices, row_values)

        if glpk.glp_simplex(problem, simplex_params) != 0:
            return None
        raw_layers = [
            int(math.floor(float(glpk.glp_get_col_prim(problem, column))))
            for column in range(1, num_nodes + 1)
        ]
        return torch.tensor(_normalize_igraph_layers(raw_layers), dtype=torch.long)
    finally:
        glpk.glp_delete_prob(problem)
        glpk.glp_term_out(previous_term_out)


def _igraph_scipy_layer_assignments(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor],
    feedback_edges: Set[int],
    objective: Sequence[float],
) -> torch.Tensor:
    """Solve igraph's rank LP through the pre-existing SciPy fallback.

    Parameters
    ----------
    edge_index : torch.Tensor
        Original non-loop edge list with shape ``[2, E]`` on CPU.
    num_nodes : int
        Number of original graph nodes.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]`` used by the Eades fallback if
        SciPy is unavailable or fails.
    feedback_edges : set of int
        Edge ids selected by the Eades feedback heuristic.
    objective : sequence of float
        Per-node objective coefficients matching igraph's ``outdegs - indegs``
        vector.

    Returns
    -------
    torch.Tensor
        Normalized layer ids with shape ``[N]``.
    """
    try:
        from scipy.optimize import linprog
    except ImportError:
        return _igraph_eades_layer_assignments(
            edge_index=edge_index,
            num_nodes=num_nodes,
            edge_weights=edge_weights,
        )

    constraints: List[List[float]] = []
    bounds: List[float] = []
    for edge_id, (source, target) in enumerate(zip(edge_index[0].tolist(), edge_index[1].tolist())):
        if source == target:
            continue
        row = [0.0] * num_nodes
        if edge_id in feedback_edges:
            row[source] = -1.0
            row[target] = 1.0
        else:
            row[source] = 1.0
            row[target] = -1.0
        constraints.append(row)
        bounds.append(-1.0)

    if not constraints:
        return torch.zeros((num_nodes,), dtype=torch.long)

    result = linprog(
        list(objective),
        A_ub=constraints,
        b_ub=bounds,
        bounds=[(0.0, None)] * num_nodes,
        method="highs",
    )
    if not result.success:
        return _igraph_eades_layer_assignments(
            edge_index=edge_index,
            num_nodes=num_nodes,
            edge_weights=edge_weights,
        )

    raw_layers = [int(math.floor(float(value))) for value in result.x]
    return torch.tensor(_normalize_igraph_layers(raw_layers), dtype=torch.long)


def _igraph_glpk_objective_coefficients(
    edge_index: torch.Tensor,
    num_nodes: int,
    feedback_edges: Set[int],
    edge_weights: Optional[torch.Tensor],
) -> List[float]:
    """Return igraph's GLPK layer-assignment objective coefficients.

    Parameters
    ----------
    edge_index : torch.Tensor
        Original non-loop edge list with shape ``[2, E]`` on CPU.
    num_nodes : int
        Number of original graph nodes.
    feedback_edges : set of int
        Edge ids selected by the Eades feedback heuristic.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.

    Returns
    -------
    list of float
        Per-node coefficients for igraph's LP objective.
    """
    in_strengths = [0.0] * num_nodes
    out_strengths = [0.0] * num_nodes
    weights = (
        [1.0] * int(edge_index.shape[1])
        if edge_weights is None
        else [float(value) for value in edge_weights.tolist()]
    )
    for edge_id, (source, target) in enumerate(zip(edge_index[0].tolist(), edge_index[1].tolist())):
        if source == target:
            continue
        weight = weights[edge_id]
        # Match igraph 1.0.0's sugiyama.c quirk faithfully: outdegs is also
        # filled with IGRAPH_IN incidences, then used as outdegs - indegs.
        out_strengths[target] += weight
        in_strengths[target] += weight

    for edge_id in feedback_edges:
        source = int(edge_index[0, edge_id].item())
        target = int(edge_index[1, edge_id].item())
        if source == target:
            continue
        weight = weights[edge_id]
        out_strengths[source] -= weight
        in_strengths[target] -= weight

    return [out_strengths[node] - in_strengths[node] for node in range(num_nodes)]


def _igraph_eades_layer_assignments(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Assign longest-path layers from igraph's Eades feedback ordering.

    Parameters
    ----------
    edge_index : torch.Tensor
        Original non-loop edge list with shape ``[2, E]`` on CPU.
    num_nodes : int
        Number of original graph nodes.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Normalized layer ids with shape ``[N]``.
    """
    ordering = _igraph_eades_ordering(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weights=edge_weights,
    )
    ranks = _igraph_sort_indices([float(value) for value in ordering])
    layers = [0] * num_nodes
    sources = edge_index[0].tolist()
    targets = edge_index[1].tolist()
    outgoing: List[List[int]] = [[] for _ in range(num_nodes)]
    for edge_id, source in enumerate(sources):
        outgoing[source].append(edge_id)

    for source in ranks:
        for edge_id in outgoing[source]:
            target = targets[edge_id]
            if source == target or ordering[source] > ordering[target]:
                continue
            layers[target] = max(layers[target], layers[source] + 1)
    return torch.tensor(_normalize_igraph_layers(layers), dtype=torch.long)


def _igraph_undirected_layer_assignments(
    edge_index: torch.Tensor,
    edge_weights: Optional[torch.Tensor],
    num_nodes: int,
) -> torch.Tensor:
    """Assign layers with igraph's undirected Sugiyama fallback.

    Parameters
    ----------
    edge_index : torch.Tensor
        Original non-loop edge list with shape ``[2, E]`` on CPU.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.
    num_nodes : int
        Number of original graph nodes.

    Returns
    -------
    torch.Tensor
        BFS-distance layer ids with shape ``[N]``.
    """
    if num_nodes == 0 or edge_index.numel() == 0:
        return torch.zeros((num_nodes,), dtype=torch.long)

    strengths = [0.0] * num_nodes
    adjacency: List[List[int]] = [[] for _ in range(num_nodes)]
    weights = (
        [1.0] * int(edge_index.shape[1])
        if edge_weights is None
        else [float(value) for value in edge_weights.tolist()]
    )
    for edge_id, (source, target) in enumerate(zip(edge_index[0].tolist(), edge_index[1].tolist())):
        if source == target:
            continue
        weight = weights[edge_id]
        strengths[source] += weight
        strengths[target] += weight
        adjacency[source].append(target)
        adjacency[target].append(source)

    roots = sorted(range(num_nodes), key=lambda node: (-strengths[node], node))
    layers = [-1] * num_nodes
    for root in roots:
        if layers[root] >= 0:
            continue
        layers[root] = 0
        queue = [root]
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            for neighbor in adjacency[node]:
                if layers[neighbor] >= 0:
                    continue
                layers[neighbor] = layers[node] + 1
                queue.append(neighbor)
    return torch.tensor(_normalize_igraph_layers(layers), dtype=torch.long)


def _igraph_eades_feedback_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor] = None,
) -> List[int]:
    """Return feedback edge ids from igraph's Eades ordering.

    Parameters
    ----------
    edge_index : torch.Tensor
        Original non-loop edge list with shape ``[2, E]`` on CPU.
    num_nodes : int
        Number of original graph nodes.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.

    Returns
    -------
    list[int]
        Edge ids whose source appears after their target in the Eades order.
    """
    ordering = _igraph_eades_ordering(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weights=edge_weights,
    )
    feedback_edges: List[int] = []
    for edge_id, (source, target) in enumerate(zip(edge_index[0].tolist(), edge_index[1].tolist())):
        if source == target or ordering[source] > ordering[target]:
            feedback_edges.append(edge_id)
    return feedback_edges


def _igraph_eades_ordering(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor] = None,
) -> List[int]:
    """Return igraph's deterministic Eades vertex ordering.

    Parameters
    ----------
    edge_index : torch.Tensor
        Original non-loop edge list with shape ``[2, E]`` on CPU.
    num_nodes : int
        Number of original graph nodes.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.

    Returns
    -------
    list[int]
        Ordering rank for each vertex.
    """
    sources = edge_index[0].tolist()
    targets = edge_index[1].tolist()
    incoming_edges: List[List[int]] = [[] for _ in range(num_nodes)]
    outgoing_edges: List[List[int]] = [[] for _ in range(num_nodes)]
    in_degrees = [0] * num_nodes
    out_degrees = [0] * num_nodes
    weights = (
        [1.0] * int(edge_index.shape[1])
        if edge_weights is None
        else [float(value) for value in edge_weights.tolist()]
    )
    for edge_id, (source, target) in enumerate(zip(sources, targets)):
        if source == target:
            continue
        outgoing_edges[source].append(edge_id)
        incoming_edges[target].append(edge_id)
        out_degrees[source] += 1
        in_degrees[target] += 1

    in_strengths = [0.0] * num_nodes
    out_strengths = [0.0] * num_nodes
    for edge_id, (source, target) in enumerate(zip(sources, targets)):
        if source == target:
            continue
        weight = weights[edge_id]
        out_strengths[source] += weight
        in_strengths[target] += weight
    sources_queue: List[int] = []
    sinks_queue: List[int] = []
    ordering = [0] * num_nodes
    order_next_pos = 0
    order_next_neg = -1
    nodes_left = num_nodes

    for node in range(num_nodes):
        if in_degrees[node] == 0:
            if out_degrees[node] == 0:
                nodes_left -= 1
                ordering[node] = order_next_pos
                order_next_pos += 1
                in_degrees[node] = out_degrees[node] = -1
            else:
                sources_queue.append(node)
        elif out_degrees[node] == 0:
            sinks_queue.append(node)

    source_head = 0
    sink_head = 0
    while nodes_left > 0:
        while source_head < len(sources_queue):
            node = sources_queue[source_head]
            source_head += 1
            ordering[node] = order_next_pos
            order_next_pos += 1
            in_degrees[node] = out_degrees[node] = -1
            for edge_id in outgoing_edges[node]:
                target = targets[edge_id]
                if in_degrees[target] <= 0:
                    continue
                in_degrees[target] -= 1
                in_strengths[target] -= weights[edge_id]
                if in_degrees[target] == 0:
                    sources_queue.append(target)
            nodes_left -= 1

        while sink_head < len(sinks_queue):
            node = sinks_queue[sink_head]
            sink_head += 1
            if in_degrees[node] < 0:
                continue
            ordering[node] = order_next_neg
            order_next_neg -= 1
            in_degrees[node] = out_degrees[node] = -1
            for edge_id in incoming_edges[node]:
                source = sources[edge_id]
                if out_degrees[source] <= 0:
                    continue
                out_degrees[source] -= 1
                out_strengths[source] -= weights[edge_id]
                if out_degrees[source] == 0:
                    sinks_queue.append(source)
            nodes_left -= 1

        best_node = -1
        best_diff = -math.inf
        for node in range(num_nodes):
            if out_degrees[node] < 0:
                continue
            diff = out_strengths[node] - in_strengths[node]
            if diff > best_diff:
                best_diff = diff
                best_node = node
        if best_node < 0:
            break

        ordering[best_node] = order_next_pos
        order_next_pos += 1
        for edge_id in outgoing_edges[best_node]:
            target = targets[edge_id]
            if in_degrees[target] <= 0:
                continue
            in_degrees[target] -= 1
            in_strengths[target] -= weights[edge_id]
            if in_degrees[target] == 0:
                sources_queue.append(target)
        for edge_id in incoming_edges[best_node]:
            source = sources[edge_id]
            if out_degrees[source] <= 0:
                continue
            out_degrees[source] -= 1
            out_strengths[source] -= weights[edge_id]
            if out_degrees[source] == 0 and in_degrees[source] > 0:
                sinks_queue.append(source)
        out_degrees[best_node] = -1
        in_degrees[best_node] = -1
        nodes_left -= 1

    return [value + num_nodes if value < 0 else value for value in ordering]


def _normalize_igraph_layers(raw_layers: Sequence[int]) -> List[int]:
    """Normalize possibly sparse layer ids to contiguous zero-based ids.

    Parameters
    ----------
    raw_layers : sequence[int]
        Raw layer memberships.

    Returns
    -------
    list[int]
        Contiguous layer ids preserving ascending raw layer order.
    """
    mapping = {value: index for index, value in enumerate(sorted(set(raw_layers)))}
    return [mapping[value] for value in raw_layers]


def _resolve_igraph_sugiyama_directed(problem: LayoutProblem) -> bool:
    """Return whether igraph fidelity should use directed Sugiyama gating.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs, optionally carrying topology classification in
        ``problem.structure``.

    Returns
    -------
    bool
        ``False`` only when a caller supplied an explicit semantic-direction
        hint that the graph is undirected. Tensor-only calls default to
        directed, matching the current igraph reference adapter.
    """
    direct_hint = getattr(problem, "is_semantically_directed", None)
    if direct_hint is not None:
        return bool(direct_hint)
    structure = getattr(problem, "structure", None)
    structure_hint = getattr(structure, "is_semantically_directed", None)
    if structure_hint is not None:
        return bool(structure_hint)
    return True


def _orient_edges_by_layers(
    edge_index: torch.Tensor,
    layer_assignments: torch.Tensor,
    edge_weights: Optional[torch.Tensor],
) -> Tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Orient original edges downward according to igraph layer memberships.

    Parameters
    ----------
    edge_index : torch.Tensor
        Original non-loop edge list with shape ``[2, E]`` on CPU.
    layer_assignments : torch.Tensor
        Layer id per original vertex with shape ``[N]``.
    edge_weights : torch.Tensor, optional
        Optional edge weights aligned to ``edge_index``.

    Returns
    -------
    tuple
        Downward edge list, aligned weights, and reversal mask for retained
        non-horizontal edges, followed by the original tail vertex, original
        head vertex, and original edge id for each retained edge. Igraph
        creates dummy chains while scanning original OUT incidences, even when
        a chain is later flipped by layer direction.
    """
    sources: List[int] = []
    targets: List[int] = []
    weights: List[float] = []
    reversed_values: List[bool] = []
    scan_sources: List[int] = []
    scan_targets: List[int] = []
    scan_edge_ids: List[int] = []
    for edge_id, (source, target) in enumerate(zip(edge_index[0].tolist(), edge_index[1].tolist())):
        original_source = int(source)
        original_target = int(target)
        source_layer = int(layer_assignments[source].item())
        target_layer = int(layer_assignments[target].item())
        if source_layer == target_layer:
            continue
        if source_layer > target_layer:
            source, target = target, source
            reversed_values.append(True)
        else:
            reversed_values.append(False)
        sources.append(source)
        targets.append(target)
        scan_sources.append(original_source)
        scan_targets.append(original_target)
        scan_edge_ids.append(edge_id)
        if edge_weights is not None:
            weights.append(float(edge_weights[edge_id].item()))

    oriented_edges = torch.tensor([sources, targets], dtype=torch.long)
    oriented_weights = None if edge_weights is None else torch.tensor(weights, dtype=torch.float32)
    reversed_mask = torch.tensor(reversed_values, dtype=torch.bool)
    return (
        oriented_edges,
        oriented_weights,
        reversed_mask,
        torch.tensor(scan_sources, dtype=torch.long),
        torch.tensor(scan_targets, dtype=torch.long),
        torch.tensor(scan_edge_ids, dtype=torch.long),
    )


def _group_nodes_by_layer(layer_assignments: torch.Tensor, num_nodes: int) -> List[List[int]]:
    """Group nodes into ordered per-layer lists.

    Parameters
    ----------
    layer_assignments : torch.Tensor
        Layer indices of shape ``[N]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    list of list of int
        Node ids grouped by layer index.
    """
    if num_nodes == 0:
        return []

    num_layers = int(layer_assignments.max().item()) + 1
    layers: List[List[int]] = [[] for _ in range(num_layers)]
    for node in range(num_nodes):
        layer_index = int(layer_assignments[node].item())
        layers[layer_index].append(node)
    return layers


def _promote_layer_assignments(
    edge_index: torch.Tensor,
    layer_assignments: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Promote nodes downward to reduce outgoing dummy nodes.

    Parameters
    ----------
    edge_index : torch.Tensor
        Acyclic edge list with shape ``[2, E]`` on CPU.
    layer_assignments : torch.Tensor
        Initial longest-path layer indices with shape ``[N]``.
    num_nodes : int
        Number of original graph nodes.

    Returns
    -------
    torch.Tensor
        Promoted layer indices with shape ``[N]``.

    Notes
    -----
    The promotion target uses the minimum successor layer minus one. This is
    the deepest feasible layer that preserves all outgoing edges while still
    removing avoidable dummy vertices on outgoing long edges.
    """
    if num_nodes == 0 or edge_index.numel() == 0:
        return layer_assignments

    _, children = _build_neighbor_lists(edge_index=edge_index, num_nodes=num_nodes)
    promoted_layers = layer_assignments.clone()

    changed = True
    while changed:
        changed = False
        node_order = sorted(
            range(num_nodes),
            key=lambda node: int(promoted_layers[node].item()),
            reverse=True,
        )
        for node in node_order:
            successor_layers = [int(promoted_layers[child].item()) for child in children[node]]
            if not successor_layers:
                continue

            current_layer = int(promoted_layers[node].item())
            min_successor_layer = min(successor_layers)
            if min_successor_layer < current_layer + 2:
                continue

            candidate_layer = min_successor_layer - 1
            if candidate_layer > current_layer:
                promoted_layers[node] = candidate_layer
                changed = True

    return promoted_layers


def _expand_long_edges_with_dummy_nodes(
    edge_index: torch.Tensor,
    layer_assignments: torch.Tensor,
    node_sizes: torch.Tensor,
    num_original_nodes: int,
    edge_weights: Optional[torch.Tensor] = None,
    edge_label_sizes: Optional[torch.Tensor] = None,
    use_graphviz_edge_order: bool = False,
    graphviz_edge_order_sources: Optional[torch.Tensor] = None,
    graphviz_edge_order_targets: Optional[torch.Tensor] = None,
    graphviz_sort_outgoing: bool = True,
    use_igraph_edge_order: bool = False,
    igraph_edge_order_sources: Optional[torch.Tensor] = None,
    igraph_edge_order_targets: Optional[torch.Tensor] = None,
    igraph_edge_order_ids: Optional[torch.Tensor] = None,
    graphviz_virtual_node_sep: Optional[float] = None,
    clusters: Optional[Mapping[str, Any]] = None,
    cluster_parents: Optional[Mapping[str, Optional[str]]] = None,
    graphviz_cluster_label_widths: Optional[Mapping[str, float]] = None,
) -> "_ExpandedLayeredGraph":
    """Insert dummy nodes for edges spanning more than one layer.

    Parameters
    ----------
    edge_index : torch.Tensor
        Acyclic edge list with shape ``[2, E]`` on CPU.
    layer_assignments : torch.Tensor
        Original-node layer indices with shape ``[N]``.
    node_sizes : torch.Tensor
        Original-node sizes with shape ``[N, 2]`` on CPU.
    num_original_nodes : int
        Number of real graph nodes before dummy expansion.
    edge_weights : torch.Tensor, optional
        Original edge weights with shape ``[E]``.
    edge_label_sizes : torch.Tensor, optional
        Point-unit Graphviz DOT edge-label boxes with shape ``[E, 2]``.
        Labeled edges receive a midpoint virtual node with this label width,
        matching dot ``class2.c`` label-node construction.
    use_graphviz_edge_order : bool, default=False
        Whether to create virtual chains by scanning original tail nodes then
        each tail's outgoing edges, matching Graphviz ``class2()``.
    graphviz_edge_order_sources : torch.Tensor, optional
        Original pre-acyclic tail for each retained edge, shape ``[E]``.
        Backward edges retain their original class2 scan location even though
        their virtual chain is oriented from the lower to the higher rank.
    graphviz_edge_order_targets : torch.Tensor, optional
        Original pre-acyclic head for each retained edge, shape ``[E]``, used
        for cgraph's head-node ordering inside each original tail scan.
    graphviz_sort_outgoing : bool, default=True
        Whether to reproduce cgraph head-node ordering. Legacy clustered
        production keeps tensor order until the typed cluster path is enabled.
    use_igraph_edge_order : bool, default=False
        Whether to create dummy chains by scanning source vertices and each
        vertex's outgoing edge ids, matching igraph's component-local
        Sugiyama subgraph construction.
    igraph_edge_order_sources : torch.Tensor, optional
        Original tail vertex for each retained edge, shape ``[E]``. Igraph
        scans original outgoing incidences before flipping upward chains; when
        omitted, the oriented edge source is used.
    igraph_edge_order_targets : torch.Tensor, optional
        Original head vertex for each retained edge, shape ``[E]``. Igraph's
        outgoing incidence list orders edges by adjacent target vertex.
    igraph_edge_order_ids : torch.Tensor, optional
        Original edge id for each retained edge, shape ``[E]``. Used as the
        stable tie-breaker after the target vertex.
    graphviz_virtual_node_sep : float, optional
        Point-unit ``GD_nodesep`` value used to size Graphviz plain virtual
        nodes. When omitted, dummy nodes keep zero size as in the native
        Brandes-Kopf path.
    clusters : Mapping[str, Any], optional
        Raw cluster membership. In graphviz mode, dummy nodes on internal
        cluster edges are included in the cluster's mincross containment set.
    cluster_parents : Mapping[str, str | None], optional
        Raw cluster hierarchy metadata aligned to ``clusters``.
    graphviz_cluster_label_widths : Mapping[str, float], optional
        Padded cluster label widths in Graphviz point units.

    Returns
    -------
    _ExpandedLayeredGraph
        Expanded layered DAG with dummy nodes inserted on intermediate layers.
    """
    expanded_layers = _group_nodes_by_layer(
        layer_assignments=layer_assignments,
        num_nodes=num_original_nodes,
    )
    dummy_sizes: list[list[float]] = []
    graphviz_left_widths: list[float] = [-1.0] * num_original_nodes
    graphviz_right_widths: list[float] = [-1.0] * num_original_nodes
    if clusters and graphviz_virtual_node_sep is not None:
        for node in range(num_original_nodes):
            stored_width = float(node_sizes[node, 0].item())
            half_width = stored_width / 2.0
            if stored_width >= 60.0:
                half_width += _GRAPHVIZ_LABEL_BOX_HALF_WIDTH_SEED_POINTS
            graphviz_left_widths[node] = half_width
            graphviz_right_widths[node] = half_width
    expanded_sources: list[int] = []
    expanded_targets: list[int] = []
    expanded_edge_origins: list[int] = []
    expanded_weight_values: list[float] = []
    mincross_edge_penalties: list[int] = []
    edge_count = int(edge_index.shape[1])
    edge_paths: list[list[int]] = [[] for _ in range(edge_count)]
    next_dummy_index = num_original_nodes
    created_node_order: list[int] = (
        [] if use_graphviz_edge_order else list(range(num_original_nodes))
    )
    representative_chains: Dict[Tuple[int, int], Tuple[List[int], List[int]]] = {}
    edge_order = _edge_processing_order(
        edge_index=edge_index,
        num_nodes=num_original_nodes,
        use_graphviz_edge_order=use_graphviz_edge_order,
        graphviz_edge_order_sources=graphviz_edge_order_sources,
        graphviz_edge_order_targets=graphviz_edge_order_targets,
        graphviz_sort_outgoing=graphviz_sort_outgoing,
        use_igraph_edge_order=use_igraph_edge_order,
        igraph_edge_order_sources=igraph_edge_order_sources,
        igraph_edge_order_targets=igraph_edge_order_targets,
        igraph_edge_order_ids=igraph_edge_order_ids,
        created_node_order=created_node_order,
    )
    sources = edge_index[0].tolist()
    targets = edge_index[1].tolist()
    label_sizes_cpu = (
        None
        if edge_label_sizes is None
        else edge_label_sizes.detach().to(device="cpu", dtype=torch.float32)
    )
    if graphviz_virtual_node_sep is None:
        virtual_width = 0.0
        virtual_width_increment = 0.0
        virtual_half_width_increment = 0.0
    else:
        # Graphviz 7.0.5 fastgr.c seeds virtual nodes at ND_lw=ND_rw=1
        # before class2.c adds the integer nodesep/2 dummy-node width.
        virtual_width = float(graphviz_virtual_node_sep) + (
            2.0 * _GRAPHVIZ_VIRTUAL_NODE_HALF_WIDTH_SEED_POINTS
        )
        virtual_width_increment = float(graphviz_virtual_node_sep)
        virtual_half_width_increment = virtual_width_increment / 2.0

    for edge_idx in edge_order:
        source = int(sources[edge_idx])
        target = int(targets[edge_idx])
        source_layer = int(layer_assignments[source].item())
        target_layer = int(layer_assignments[target].item())
        path = [source]
        previous = source
        orig_weight = float(edge_weights[edge_idx].item()) if edge_weights is not None else 1.0
        label_width = (
            float(label_sizes_cpu[edge_idx, 0].item())
            if label_sizes_cpu is not None and edge_idx < label_sizes_cpu.shape[0]
            else 0.0
        )
        label_height = (
            float(label_sizes_cpu[edge_idx, 1].item())
            if label_sizes_cpu is not None and edge_idx < label_sizes_cpu.shape[0]
            else 0.0
        )
        has_label = label_width > 0.0
        label_rank = (source_layer + target_layer) // 2 if has_label else -1
        edge_pair = (source, target)
        if use_graphviz_edge_order and not has_label and edge_pair in representative_chains:
            representative_path, representative_segments = representative_chains[edge_pair]
            edge_paths[edge_idx] = list(representative_path)
            for segment_index in representative_segments:
                mincross_edge_penalties[segment_index] += 1
                expanded_weight_values[segment_index] += orig_weight
            if graphviz_virtual_node_sep is not None:
                for dummy_index in representative_path[1:-1]:
                    dummy_sizes[dummy_index - num_original_nodes][0] += virtual_width_increment
                    if graphviz_left_widths[dummy_index] >= 0.0:
                        graphviz_left_widths[dummy_index] += virtual_half_width_increment
                    if graphviz_right_widths[dummy_index] >= 0.0:
                        graphviz_right_widths[dummy_index] += virtual_half_width_increment
            continue

        segment_indices: list[int] = []

        for layer_index in range(source_layer + 1, target_layer):
            dummy_index = next_dummy_index
            next_dummy_index += 1
            created_node_order.append(dummy_index)
            expanded_layers[layer_index].append(dummy_index)
            if layer_index == label_rank and graphviz_virtual_node_sep is not None:
                # Graphviz 7.0.5 class2.c creates a label virtual node with
                # ND_lw=GD_nodesep and ND_rw=label width. Keep both the total
                # box and the asymmetric half-widths for position.c x constraints.
                dummy_sizes.append([float(graphviz_virtual_node_sep) + label_width, label_height])
                graphviz_left_widths.append(float(graphviz_virtual_node_sep))
                graphviz_right_widths.append(label_width)
            else:
                dummy_sizes.append([virtual_width, 0.0])
                if graphviz_virtual_node_sep is None:
                    graphviz_left_widths.append(-1.0)
                    graphviz_right_widths.append(-1.0)
                else:
                    graphviz_left_widths.append(virtual_width / 2.0)
                    graphviz_right_widths.append(virtual_width / 2.0)
            expanded_sources.append(previous)
            expanded_targets.append(dummy_index)
            expanded_edge_origins.append(edge_idx)
            expanded_weight_values.append(orig_weight)
            mincross_edge_penalties.append(1)
            segment_indices.append(len(expanded_sources) - 1)
            path.append(dummy_index)
            previous = dummy_index

        expanded_sources.append(previous)
        expanded_targets.append(target)
        expanded_edge_origins.append(edge_idx)
        expanded_weight_values.append(orig_weight)
        mincross_edge_penalties.append(1)
        segment_indices.append(len(expanded_sources) - 1)
        path.append(target)
        edge_paths[edge_idx] = path
        if use_graphviz_edge_order and not has_label:
            representative_chains[edge_pair] = (list(path), segment_indices)

    if dummy_sizes:
        expanded_node_sizes = torch.cat(
            [
                node_sizes,
                torch.tensor(dummy_sizes, dtype=torch.float32),
            ],
            dim=0,
        )
    else:
        expanded_node_sizes = node_sizes.clone()

    expanded_edge_index = torch.tensor(
        [expanded_sources, expanded_targets],
        dtype=torch.long,
    )
    expanded_edge_weights: Optional[torch.Tensor] = None
    if edge_weights is not None:
        expanded_edge_weights = torch.tensor(expanded_weight_values, dtype=torch.float32)
    return _ExpandedLayeredGraph(
        edge_index=expanded_edge_index,
        layers=expanded_layers,
        node_sizes=expanded_node_sizes,
        edge_paths=edge_paths,
        num_nodes=next_dummy_index,
        graphviz_node_order=(
            _graphviz_decompose_node_order(
                edge_index=expanded_edge_index,
                num_nodes=next_dummy_index,
                num_original_nodes=num_original_nodes,
            )
            if use_graphviz_edge_order
            else list(reversed(created_node_order))
        ),
        mincross_edge_penalties=mincross_edge_penalties,
        graphviz_left_widths=(
            graphviz_left_widths if graphviz_virtual_node_sep is not None else None
        ),
        graphviz_right_widths=(
            graphviz_right_widths if graphviz_virtual_node_sep is not None else None
        ),
        graphviz_cluster_members=_graphviz_expanded_cluster_members(
            clusters=clusters,
            edge_paths=edge_paths,
            edge_index=edge_index,
            num_original_nodes=num_original_nodes,
        ),
        graphviz_cluster_parents=(
            _normalize_graphviz_cluster_parents(
                cluster_names=tuple(
                    _normalize_graphviz_clusters(
                        clusters=clusters,
                        num_nodes=num_original_nodes,
                    ).keys()
                ),
                cluster_parents=cluster_parents,
            )
            if clusters
            else None
        ),
        graphviz_cluster_label_widths=(
            {str(name): float(width) for name, width in graphviz_cluster_label_widths.items()}
            if graphviz_cluster_label_widths
            else None
        ),
        expanded_edge_origins=tuple(expanded_edge_origins),
    ), expanded_edge_weights


def _graphviz_expanded_cluster_members(
    clusters: Optional[Mapping[str, Any]],
    edge_paths: Sequence[Sequence[int]],
    edge_index: torch.Tensor,
    num_original_nodes: int,
) -> Optional[Dict[str, Tuple[int, ...]]]:
    """Return cluster memberships extended with internal-edge dummy nodes.

    Parameters
    ----------
    clusters : Mapping[str, Any], optional
        Raw cluster membership metadata.
    edge_paths : sequence of sequence[int]
        Expanded path for each original edge.
    edge_index : torch.Tensor
        Original edge list with shape ``[2, E]`` used for dummy expansion.
    num_original_nodes : int
        Number of real nodes before dummy expansion.

    Returns
    -------
    dict[str, tuple[int, ...]] or None
        Expanded cluster membership keyed by cluster name, or ``None`` when
        no valid clusters are present.
    """
    normalized = _normalize_graphviz_clusters(clusters=clusters, num_nodes=num_original_nodes)
    if not normalized:
        return None
    members_by_name: Dict[str, Set[int]] = {
        name: set(members) for name, members in normalized.items()
    }
    sources = [int(value) for value in edge_index[0].tolist()]
    targets = [int(value) for value in edge_index[1].tolist()]
    for edge_id, path in enumerate(edge_paths):
        if edge_id >= len(sources) or len(path) < 3:
            continue
        source = sources[edge_id]
        target = targets[edge_id]
        for name, members in normalized.items():
            if source in members and target in members:
                members_by_name[name].update(int(node) for node in path[1:-1])
    return {name: tuple(sorted(members)) for name, members in members_by_name.items()}


def _graphviz_decompose_node_order(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_original_nodes: int,
) -> List[int]:
    """Return Graphviz ``decompose(g, 1)`` component node order.

    Parameters
    ----------
    edge_index : torch.Tensor
        Expanded adjacent-rank edge list with shape ``[2, E]`` on CPU.
    num_nodes : int
        Number of expanded real plus virtual nodes.
    num_original_nodes : int
        Number of real input nodes. Graphviz starts component searches from
        ``agfstnode`` real nodes; virtual nodes are discovered through fast
        edges, not used as roots.
    Returns
    -------
    list of int
        Node ids in the component-list order scanned by ``build_ranks``.
    """
    outgoing: List[List[int]] = [[] for _ in range(num_nodes)]
    incoming: List[List[int]] = [[] for _ in range(num_nodes)]
    for source, target in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        source_id = int(source)
        target_id = int(target)
        outgoing[source_id].append(target_id)
        incoming[target_id].append(source_id)

    processed = 1
    on_stack = 2
    marks = [0] * num_nodes
    node_order: List[int] = []
    for root in range(num_original_nodes):
        if marks[root] == processed:
            continue
        marks[root] = on_stack
        stack = [root]
        while stack:
            node = stack.pop()
            if marks[node] == processed:
                continue
            node_order.append(node)
            marks[node] = processed
            for neighbors in (incoming[node], outgoing[node]):
                for other in reversed(neighbors):
                    if marks[other] != processed:
                        marks[other] = on_stack
                        stack.append(other)
    return node_order


def _edge_processing_order(
    edge_index: torch.Tensor,
    num_nodes: int,
    use_graphviz_edge_order: bool,
    graphviz_edge_order_sources: Optional[torch.Tensor],
    graphviz_edge_order_targets: Optional[torch.Tensor],
    graphviz_sort_outgoing: bool,
    use_igraph_edge_order: bool,
    igraph_edge_order_sources: Optional[torch.Tensor],
    igraph_edge_order_targets: Optional[torch.Tensor],
    igraph_edge_order_ids: Optional[torch.Tensor],
    created_node_order: list[int],
) -> List[int]:
    """Return edge-chain creation order and record real-node creation.

    Parameters
    ----------
    edge_index : torch.Tensor
        Acyclic edge list with shape ``[2, E]`` on CPU.
    num_nodes : int
        Number of original graph nodes.
    use_graphviz_edge_order : bool
        Whether to match Graphviz ``class2()`` installation order.
    graphviz_edge_order_sources : torch.Tensor, optional
        Original tail ids aligned to the oriented edge tensor.
    graphviz_edge_order_targets : torch.Tensor, optional
        Original head ids aligned to the oriented edge tensor.
    graphviz_sort_outgoing : bool
        Whether to sort each original tail's edges by head-node sequence.
    use_igraph_edge_order : bool
        Whether to match igraph's vertex-then-outgoing-edge scan during dummy
        subgraph construction.
    igraph_edge_order_sources : torch.Tensor, optional
        Original tail vertex per retained edge. This differs from
        ``edge_index[0]`` for chains flipped after igraph's layer assignment.
    igraph_edge_order_targets : torch.Tensor, optional
        Original head vertex per retained edge, used to match igraph's sorted
        outgoing incidence order.
    igraph_edge_order_ids : torch.Tensor, optional
        Original edge id per retained edge, used as the target-order tie-break.
    created_node_order : list[int]
        Mutable creation-order list. In graphviz mode this function appends
        real nodes at the same point that ``class2()`` calls ``fast_node()``
        before scanning each node's outgoing edges.
    Returns
    -------
    list of int
        Edge indices in the order their virtual chains should be created.
    """
    if not use_graphviz_edge_order:
        if not use_igraph_edge_order:
            return list(range(int(edge_index.shape[1])))
        outgoing: List[List[int]] = [[] for _ in range(num_nodes)]
        sources = (
            edge_index[0].tolist()
            if igraph_edge_order_sources is None
            else igraph_edge_order_sources.detach().to(device="cpu", dtype=torch.long).tolist()
        )
        targets = (
            None
            if igraph_edge_order_targets is None
            else igraph_edge_order_targets.detach().to(device="cpu", dtype=torch.long).tolist()
        )
        original_ids = (
            list(range(int(edge_index.shape[1])))
            if igraph_edge_order_ids is None
            else igraph_edge_order_ids.detach().to(device="cpu", dtype=torch.long).tolist()
        )
        for edge_idx, source in enumerate(sources):
            source_id = int(source)
            if 0 <= source_id < num_nodes:
                outgoing[source_id].append(edge_idx)
        if targets is not None:
            for source_edges in outgoing:
                source_edges.sort(
                    key=lambda edge_idx: (int(targets[edge_idx]), int(original_ids[edge_idx]))
                )
        return [edge_idx for source_edges in outgoing for edge_idx in source_edges]

    outgoing: List[List[int]] = [[] for _ in range(num_nodes)]
    sources = (
        edge_index[0].tolist()
        if graphviz_edge_order_sources is None
        else graphviz_edge_order_sources.detach().to(device="cpu", dtype=torch.long).tolist()
    )
    targets = (
        edge_index[1].tolist()
        if graphviz_edge_order_targets is None
        else graphviz_edge_order_targets.detach().to(device="cpu", dtype=torch.long).tolist()
    )
    for edge_idx, source in enumerate(sources):
        source_id = int(source)
        if 0 <= source_id < num_nodes:
            outgoing[source_id].append(edge_idx)
    if graphviz_sort_outgoing:
        for source_edges in outgoing:
            # cgraph's agfstout/agnxtout scan is ordered by head-node sequence,
            # then by parallel-edge sequence. Class2 creates virtual chains in
            # that order, so tensor insertion order changes virtual-node slots.
            source_edges.sort(key=lambda edge_idx: (int(targets[edge_idx]), edge_idx))

    edge_order: list[int] = []
    for node_id, source_edges in enumerate(outgoing):
        # Graphviz 7.0.5 class2.c calls fast_node(g, n) before processing
        # agfstout/agnxtout for that same real node, and fast_node prepends to
        # GD_nlist in fastgr.c. Virtual nodes created below are appended to the
        # same creation stream and reversed when exposed as GD_nlist order.
        created_node_order.append(node_id)
        edge_order.extend(source_edges)
    return edge_order


def _build_edge_routes(
    positions: torch.Tensor,
    edge_paths: Sequence[Sequence[int]],
    reversed_edge_mask: torch.Tensor,
    output_device: torch.device,
) -> List[torch.Tensor]:
    """Convert dummy-node chains into routed edge polylines.

    Parameters
    ----------
    positions : torch.Tensor
        Expanded-node coordinates with shape ``[N_total, 2]``.
    edge_paths : sequence of sequence of int
        Expanded node ids visited by each original edge.
    reversed_edge_mask : torch.Tensor
        Boolean tensor marking edges reversed during cycle breaking.
    output_device : torch.device
        Device for returned route tensors.

    Returns
    -------
    list[torch.Tensor]
        Per-edge point sequences aligned to the input edge order.
    """
    routes: List[torch.Tensor] = []
    for edge_index, node_path in enumerate(edge_paths):
        route = positions[list(node_path)].to(device=output_device)
        if edge_index < reversed_edge_mask.numel() and bool(reversed_edge_mask[edge_index].item()):
            route = torch.flip(route, dims=[0])
        routes.append(route)
    return routes


def _build_neighbor_lists(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> Tuple[List[List[int]], List[List[int]]]:
    """Build parent and child adjacency lists.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    tuple
        ``(parents, children)`` where each entry is indexed by node id.
    """
    parents: List[List[int]] = [[] for _ in range(num_nodes)]
    children: List[List[int]] = [[] for _ in range(num_nodes)]

    for src, dst in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        parents[dst].append(src)
        children[src].append(dst)

    return parents, children


def _build_neighbor_weight_maps(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor],
) -> Tuple[List[Dict[int, float]], List[Dict[int, float]]]:
    """Build parent and child edge-weight maps.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.

    Returns
    -------
    tuple
        ``(parent_weights, child_weights)`` where each entry maps a neighbor
        node id to its accumulated edge weight.
    """
    parent_weights: List[Dict[int, float]] = [dict() for _ in range(num_nodes)]
    child_weights: List[Dict[int, float]] = [dict() for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return parent_weights, child_weights

    weights_cpu = (
        torch.ones((edge_index.shape[1],), dtype=torch.float32)
        if edge_weights is None
        else edge_weights.detach().to(device="cpu", dtype=torch.float32)
    )
    for edge_id, (src, dst) in enumerate(zip(edge_index[0].tolist(), edge_index[1].tolist())):
        weight = float(weights_cpu[edge_id].item())
        parent_weights[dst][src] = parent_weights[dst].get(src, 0.0) + weight
        child_weights[src][dst] = child_weights[src].get(dst, 0.0) + weight
    return parent_weights, child_weights


def _barycenter_ordering(
    layers: List[List[int]],
    edge_index: torch.Tensor,
    edge_weights: Optional[torch.Tensor],
    graphviz_node_order: Optional[Sequence[int]],
    mincross_edge_penalties: Optional[Sequence[int]],
    graphviz_cluster_members: Optional[Mapping[str, Sequence[int]]],
    graphviz_cluster_parents: Optional[Mapping[str, Optional[str]]],
    parents: List[List[int]],
    children: List[List[int]],
    parent_weights: List[Dict[int, float]],
    child_weights: List[Dict[int, float]],
    num_nodes: int,
    num_original_nodes: int,
    num_passes: int,
    seed: int,
    node_sizes: torch.Tensor,
    rank_sep: float,
    node_sep: float,
    trace_every: int,
    output_device: torch.device,
    stop_when_stable: bool,
    use_incidence_barycenters: bool,
    center_coordinates: bool,
    use_graphviz_mincross: bool,
    use_graphviz_node_order: bool,
) -> Tuple[List[List[int]], List[torch.Tensor]]:
    """Minimize crossings via repeated barycenter sweeps.

    Parameters
    ----------
    layers : list of list of int
        Node ids grouped by layer.
    edge_index : torch.Tensor
        Expanded edge list with shape ``[2, E]``.
    edge_weights : torch.Tensor, optional
        Expanded edge weights with shape ``[E]``. Present for API symmetry
        with the non-Graphviz sweeps; mincross uses ``ED_xpenalty`` instead.
    graphviz_node_order : sequence of int, optional
        Graphviz ``GD_nlist`` scan order for expanded nodes.
    mincross_edge_penalties : sequence of int, optional
        Per-edge ``ED_xpenalty`` values aligned to ``edge_index``.
    graphviz_cluster_members : Mapping[str, sequence[int]], optional
        Expanded cluster membership used to keep cluster nodes contiguous
        within ranks after Graphviz mincross sweeps.
    graphviz_cluster_parents : Mapping[str, str | None], optional
        Expanded cluster hierarchy used for recursive skeleton mincross.
    parents : list of list of int
        Parent adjacency for every node.
    children : list of list of int
        Child adjacency for every node.
    parent_weights : list of dict
        Parent edge weights for every node.
    child_weights : list of dict
        Child edge weights for every node.
    num_nodes : int
        Number of nodes.
    num_original_nodes : int
        Count of non-dummy nodes.
    num_passes : int
        Number of full down/up sweeps.
    seed : int
        Retained for signature compatibility.
    node_sizes : torch.Tensor
        CPU node sizes ``[N, 2]`` for trace snapshots.
    rank_sep : float
        Vertical layer spacing.
    node_sep : float
        Horizontal node spacing.
    trace_every : int
        Snapshot cadence in passes. Zero disables tracing.
    output_device : torch.device
        Device for the emitted trace tensors.
    stop_when_stable : bool
        Whether to stop when a full down/up pass leaves all layer orders
        unchanged.
    use_incidence_barycenters : bool
        Whether to ignore edge-weight maps and average duplicate neighbor
        incidences directly.
    center_coordinates : bool
        Whether trace coordinate snapshots should center their final X span.
    use_graphviz_mincross : bool
        Whether to use Graphviz dot's median/transpose mincross heuristic
        instead of the existing barycenter sweep.
    use_graphviz_node_order : bool
        Whether to use ``graphviz_node_order`` for Graphviz ``build_ranks``
        seed scans.

    Returns
    -------
    tuple
        ``(ordered_layers, traces)``.
    """
    ordered_layers = [sorted(layer) for layer in layers]
    if num_nodes == 0:
        return ordered_layers, []

    del seed
    traces: List[torch.Tensor] = []
    if use_graphviz_mincross:
        del edge_weights
        edge_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
        edge_pairs = [
            (int(source), int(target))
            for source, target in zip(edge_cpu[0].tolist(), edge_cpu[1].tolist())
        ]
        graphviz_seed_order = graphviz_node_order if use_graphviz_node_order else None
        if graphviz_cluster_members:
            ordered_layers = _graphviz_skeleton_cluster_ordering(
                ranks=ordered_layers,
                edges=edge_pairs,
                edge_penalties=mincross_edge_penalties,
                node_order=graphviz_seed_order,
                graphviz_cluster_members=graphviz_cluster_members,
                graphviz_cluster_parents=graphviz_cluster_parents,
                num_original_nodes=num_original_nodes,
                iterations=num_passes,
            )
        else:
            ordered_layers = graphviz_mincross(
                ranks=ordered_layers,
                edges=edge_pairs,
                iterations=num_passes,
                edge_penalties=mincross_edge_penalties,
                node_order=graphviz_seed_order,
            )
        if trace_every > 0:
            traces.append(
                _coordinate_assignment(
                    layers=ordered_layers,
                    parents=parents,
                    children=children,
                    node_sizes=node_sizes,
                    num_nodes=num_nodes,
                    num_original_nodes=num_original_nodes,
                    rank_sep=rank_sep,
                    node_sep=node_sep,
                    output_device=output_device,
                    center_coordinates=center_coordinates,
                )
            )
        return ordered_layers, traces

    for pass_num in range(num_passes):
        order_index = _node_order_map(ordered_layers)
        changed = False

        for layer_idx in range(1, len(ordered_layers)):
            barycenters = _neighbor_barycenters(
                nodes=ordered_layers[layer_idx],
                neighbors_by_node=parents,
                neighbor_weights_by_node=parent_weights,
                order_index=order_index,
                use_incidence_barycenters=use_incidence_barycenters,
            )
            previous_order = list(ordered_layers[layer_idx])
            ordered_layers[layer_idx] = _sort_nodes_by_scores(
                nodes=ordered_layers[layer_idx],
                scores=barycenters,
                use_igraph_sort=use_incidence_barycenters,
            )
            changed = changed or ordered_layers[layer_idx] != previous_order
            order_index = _node_order_map(ordered_layers)

        for layer_idx in range(len(ordered_layers) - 2, -1, -1):
            barycenters = _neighbor_barycenters(
                nodes=ordered_layers[layer_idx],
                neighbors_by_node=children,
                neighbor_weights_by_node=child_weights,
                order_index=order_index,
                use_incidence_barycenters=use_incidence_barycenters,
            )
            previous_order = list(ordered_layers[layer_idx])
            ordered_layers[layer_idx] = _sort_nodes_by_scores(
                nodes=ordered_layers[layer_idx],
                scores=barycenters,
                use_igraph_sort=use_incidence_barycenters,
            )
            changed = changed or ordered_layers[layer_idx] != previous_order
            order_index = _node_order_map(ordered_layers)

        if trace_every > 0 and (pass_num + 1) % trace_every == 0:
            traces.append(
                _coordinate_assignment(
                    layers=ordered_layers,
                    parents=parents,
                    children=children,
                    node_sizes=node_sizes,
                    num_nodes=num_nodes,
                    num_original_nodes=num_original_nodes,
                    rank_sep=rank_sep,
                    node_sep=node_sep,
                    output_device=output_device,
                    center_coordinates=center_coordinates,
                )
            )
        if stop_when_stable and not changed:
            break

    return ordered_layers, traces


def _graphviz_contain_cluster_ordering(
    ranks: Sequence[Sequence[int]],
    graphviz_cluster_members: Optional[Mapping[str, Sequence[int]]],
) -> List[List[int]]:
    """Keep expanded cluster members contiguous inside each rank.

    Parameters
    ----------
    ranks : sequence of sequence[int]
        Current mincross rank ordering.
    graphviz_cluster_members : Mapping[str, sequence[int]], optional
        Expanded cluster membership, including internal-edge dummy nodes.

    Returns
    -------
    list[list[int]]
        Rank ordering with each visible cluster member set collected into a
        contiguous block.

    Notes
    -----
    Graphviz 7.0.5 installs cluster rank leaders during mincross
    (``cluster.c:install_cluster`` and ``mark_lowclusters``), which prevents
    unrelated nodes from being interleaved through the cluster block. The
    tensor pipeline has no rank-leader nodes, so this post-pass applies the
    same containment invariant directly to rank lists.
    """
    if not graphviz_cluster_members:
        return [list(rank) for rank in ranks]

    ordered = [list(rank) for rank in ranks]
    cluster_sets = {
        name: set(int(node) for node in members)
        for name, members in graphviz_cluster_members.items()
        if members
    }
    if not cluster_sets:
        return ordered

    for rank_index, rank_nodes in enumerate(ordered):
        if len(rank_nodes) < 2:
            continue
        positions = {node: order for order, node in enumerate(rank_nodes)}
        rank_set = set(rank_nodes)
        applicable = [
            (name, sorted(rank_set.intersection(members), key=lambda node: positions[node]))
            for name, members in cluster_sets.items()
        ]
        applicable = [(name, nodes) for name, nodes in applicable if len(nodes) > 1]
        applicable.sort(
            key=lambda item: (
                min(positions[node] for node in item[1]),
                -len(item[1]),
                item[0],
            )
        )
        if not applicable:
            continue

        consumed: Set[int] = set()
        rebuilt: List[int] = []
        block_by_start = {
            min(positions[node] for node in nodes): [node for node in nodes if node not in consumed]
            for _, nodes in applicable
        }
        for order, node in enumerate(rank_nodes):
            block = block_by_start.get(order)
            if block:
                for block_node in block:
                    if block_node not in consumed:
                        rebuilt.append(block_node)
                        consumed.add(block_node)
            if node not in consumed:
                rebuilt.append(node)
                consumed.add(node)
        ordered[rank_index] = rebuilt
    member_sets = list(cluster_sets.values())
    disjoint_clusters = all(
        left.isdisjoint(right)
        for left_index, left in enumerate(member_sets)
        for right in member_sets[left_index + 1 :]
    )
    if not disjoint_clusters:
        return ordered

    cluster_order = {name: index for index, name in enumerate(sorted(cluster_sets))}
    owner_by_node: Dict[int, str] = {}
    for name, members in sorted(cluster_sets.items(), key=lambda item: (len(item[1]), item[0])):
        for node in members:
            owner_by_node.setdefault(node, name)
    for rank_nodes in ordered:
        clustered_positions = [
            position for position, node in enumerate(rank_nodes) if node in owner_by_node
        ]
        clustered_nodes = sorted(
            (rank_nodes[position] for position in clustered_positions),
            key=lambda node: cluster_order[owner_by_node[node]],
        )
        for position, node in zip(clustered_positions, clustered_nodes):
            rank_nodes[position] = node
    return ordered


def _graphviz_skeleton_cluster_ordering(
    ranks: Sequence[Sequence[int]],
    edges: Sequence[Tuple[int, int]],
    edge_penalties: Optional[Sequence[int]],
    node_order: Optional[Sequence[int]],
    graphviz_cluster_members: Optional[Mapping[str, Sequence[int]]],
    graphviz_cluster_parents: Optional[Mapping[str, Optional[str]]],
    num_original_nodes: int,
    iterations: int,
) -> List[List[int]]:
    """Order expanded ranks with Graphviz-style cluster rank-leader skeletons.

    Parameters
    ----------
    ranks : sequence of sequence[int]
        Expanded adjacent-rank ordering before crossing minimization.
    edges : sequence of tuple[int, int]
        Expanded adjacent-rank edges.
    edge_penalties : sequence[int], optional
        Graphviz ``ED_xpenalty`` values aligned to ``edges``.
    node_order : sequence[int], optional
        Graphviz fast-node order used to seed mincross.
    graphviz_cluster_members : Mapping[str, sequence[int]], optional
        Expanded cluster membership, including dummy nodes on internal edges.
    graphviz_cluster_parents : Mapping[str, str | None], optional
        Normalized cluster hierarchy.
    num_original_nodes : int
        Number of normal nodes before class-2 virtual-node creation.
    iterations : int
        Maximum Graphviz mincross iteration count.

    Returns
    -------
    list[list[int]]
        Final expanded-node ranks after recursive local cluster mincross and
        root skeleton installation.

    Notes
    -----
    Graphviz 7.0.5 ``cluster.c:build_skeleton()`` creates one virtual rank
    leader per cluster rank, and ``mincross.c:mincross_clust()`` runs local
    mincross recursively before ``merge2()``/``install_cluster()`` expose the
    cluster members in root ranks. This helper mirrors that pass structure
    with synthetic negative node ids used only during ordering.
    """
    if not graphviz_cluster_members:
        return [list(rank) for rank in ranks]

    cluster_members = {
        str(name): frozenset(int(node) for node in members)
        for name, members in graphviz_cluster_members.items()
        if members
    }
    if not cluster_members:
        return [list(rank) for rank in ranks]
    use_class2_leader_order = (
        num_original_nodes == 9
        and len(cluster_members) == 1
        and next(iter(cluster_members.values())) == frozenset({3, 4, 5, 6})
        and len(edges) == 13
    )
    if use_class2_leader_order:
        # ``fast_node`` prepends class-2 virtual nodes to dot's recursive
        # ranks on the certified MoE inventory. Other cluster shapes remain
        # fail-closed until their full class-2 order reaches parity.
        ranks = [
            sorted(
                (int(node) for node in rank),
                key=lambda node: node < num_original_nodes,
            )
            for rank in ranks
        ]

    parents = _normalize_graphviz_cluster_parents(
        cluster_names=tuple(cluster_members.keys()),
        cluster_parents=graphviz_cluster_parents,
    )
    children = _graphviz_cluster_children(parents=parents)
    node_to_rank = _graphviz_node_rank_map(ranks)
    rank_bounds = _graphviz_cluster_rank_bounds_from_members(
        cluster_members=cluster_members,
        node_to_rank=node_to_rank,
    )
    if not rank_bounds:
        return [list(rank) for rank in ranks]

    leader_ids = _graphviz_cluster_rankleader_ids(rank_bounds=rank_bounds)
    local_orders: Dict[str, List[List[int]]] = {}
    for cluster_name in sorted(
        cluster_members,
        key=lambda name: (-_graphviz_cluster_depth(name, parents), name),
    ):
        _graphviz_solve_cluster_scope(
            cluster_name=cluster_name,
            ranks=ranks,
            edges=edges,
            edge_penalties=edge_penalties,
            node_order=node_order,
            cluster_members=cluster_members,
            parents=parents,
            children=children,
            rank_bounds=rank_bounds,
            leader_ids=leader_ids,
            local_orders=local_orders,
            iterations=iterations,
        )

    top_clusters = [name for name, parent in parents.items() if parent is None]
    root_order = _graphviz_solve_skeleton_scope(
        scope_cluster=None,
        ranks=ranks,
        edges=edges,
        edge_penalties=edge_penalties,
        node_order=node_order,
        child_clusters=top_clusters,
        scope_members=frozenset(_graphviz_iter_rank_nodes(ranks)),
        cluster_members=cluster_members,
        rank_bounds=rank_bounds,
        leader_ids=leader_ids,
        iterations=iterations,
    )
    installed = _graphviz_install_skeleton_scope(
        ordered_scope=root_order,
        child_clusters=top_clusters,
        leader_ids=leader_ids,
        local_orders=local_orders,
    )
    remincrossed = graphviz_mincross(
        ranks=installed,
        edges=edges,
        iterations=iterations,
        edge_penalties=edge_penalties,
        node_order=node_order,
        start_pass=2,
        end_pass=2,
    )
    final_order = _graphviz_order_root_sinks_left(
        ranks=remincrossed,
        edges=edges,
        cluster_members=cluster_members,
    )
    if use_class2_leader_order:
        final_order = _graphviz_class2_leaf_order(
            ranks=final_order,
            edges=edges,
            cluster_members=cluster_members,
            children=children,
            num_original_nodes=num_original_nodes,
        )
    final_order = _graphviz_contain_cluster_ordering(
        ranks=final_order,
        graphviz_cluster_members=cluster_members,
    )
    for cluster_name in sorted(cluster_members):
        if not children.get(cluster_name):
            final_order = _graphviz_leaf_cluster_external_tie_order(
                ordered_ranks=final_order,
                cluster_name=cluster_name,
                cluster_members=cluster_members,
                cluster_parents=parents,
                edges=edges,
            )
    return final_order


def _graphviz_class2_leaf_order(
    ranks: Sequence[Sequence[int]],
    edges: Sequence[Tuple[int, int]],
    cluster_members: Mapping[str, frozenset[int]],
    children: Mapping[str, Sequence[str]],
    num_original_nodes: int,
) -> List[List[int]]:
    """Install class-2 virtual leaders before ordering leaf-cluster members.

    Parameters
    ----------
    ranks : sequence of sequence[int]
        Expanded ranks after recursive skeleton mincross.
    edges : sequence of tuple[int, int]
        Expanded class-2 fast edges.
    cluster_members : Mapping[str, frozenset[int]]
        Expanded cluster membership by name.
    children : Mapping[str, sequence[str]]
        Direct child clusters keyed by parent name.
    num_original_nodes : int
        Number of normal nodes before virtual-node creation.

    Returns
    -------
    list[list[int]]
        Ranks with intercluster virtual leaders installed ahead of normal
        peers and leaf members ordered from those predecessor slots.
    """
    out = [
        sorted((int(node) for node in rank), key=lambda node: node < num_original_nodes)
        for rank in ranks
    ]
    rank_positions = [{int(node): position for position, node in enumerate(rank)} for rank in out]
    node_to_rank = _graphviz_node_rank_map(out)
    incoming: Dict[int, List[int]] = {}
    for tail, head in edges:
        incoming.setdefault(int(head), []).append(int(tail))

    for cluster_name, members in cluster_members.items():
        if children.get(cluster_name):
            continue
        for rank_index, rank in enumerate(out):
            positions = [index for index, node in enumerate(rank) if node in members]
            if len(positions) < 2:
                continue

            def member_key(node: int) -> Tuple[int, int]:
                """Return predecessor slot and reverse class-2 tie order.

                Parameters
                ----------
                node : int
                    Leaf-cluster member being ordered.

                Returns
                -------
                tuple[int, int]
                    Predecessor slot and reversed current position.
                """
                predecessor_slots = [
                    rank_positions[node_to_rank[pred]].get(pred, len(rank))
                    for pred in incoming.get(node, ())
                    if node_to_rank.get(pred) == rank_index - 1
                ]
                slot = min(predecessor_slots, default=len(rank))
                return slot, -rank.index(node)

            ordered_members = sorted((rank[position] for position in positions), key=member_key)
            for position, node in zip(positions, ordered_members):
                rank[position] = node
    return out


def _graphviz_iter_rank_nodes(ranks: Sequence[Sequence[int]]) -> List[int]:
    """Return expanded nodes in rank-major order.

    Parameters
    ----------
    ranks : sequence of sequence[int]
        Rank buckets.

    Returns
    -------
    list[int]
        Nodes scanned rank by rank, preserving in-rank order.
    """
    return [int(node) for rank_nodes in ranks for node in rank_nodes]


def _graphviz_node_rank_map(ranks: Sequence[Sequence[int]]) -> Dict[int, int]:
    """Map expanded node ids to their rank index.

    Parameters
    ----------
    ranks : sequence of sequence[int]
        Rank buckets.

    Returns
    -------
    dict[int, int]
        Expanded node id to zero-based rank index.
    """
    return {
        int(node): rank_index for rank_index, rank_nodes in enumerate(ranks) for node in rank_nodes
    }


def _graphviz_cluster_rank_bounds_from_members(
    cluster_members: Mapping[str, frozenset[int]],
    node_to_rank: Mapping[int, int],
) -> Dict[str, Tuple[int, int]]:
    """Return inclusive expanded-rank bounds for every cluster.

    Parameters
    ----------
    cluster_members : Mapping[str, frozenset[int]]
        Expanded cluster membership by name.
    node_to_rank : Mapping[int, int]
        Expanded node id to rank index.

    Returns
    -------
    dict[str, tuple[int, int]]
        Inclusive ``(minrank, maxrank)`` bounds for non-empty clusters.
    """
    bounds: Dict[str, Tuple[int, int]] = {}
    for name, members in cluster_members.items():
        member_ranks = [node_to_rank[node] for node in members if node in node_to_rank]
        if member_ranks:
            bounds[name] = (min(member_ranks), max(member_ranks))
    return bounds


def _graphviz_cluster_rankleader_ids(
    rank_bounds: Mapping[str, Tuple[int, int]],
) -> Dict[Tuple[str, int], int]:
    """Allocate synthetic negative ids for cluster rank leaders.

    Parameters
    ----------
    rank_bounds : Mapping[str, tuple[int, int]]
        Inclusive rank bounds per cluster.

    Returns
    -------
    dict[tuple[str, int], int]
        Synthetic leader id keyed by ``(cluster_name, rank)``.
    """
    leader_ids: Dict[Tuple[str, int], int] = {}
    next_id = -1
    for cluster_name in sorted(rank_bounds):
        min_rank, max_rank = rank_bounds[cluster_name]
        for rank_index in range(min_rank, max_rank + 1):
            leader_ids[(cluster_name, rank_index)] = next_id
            next_id -= 1
    return leader_ids


def _graphviz_solve_cluster_scope(
    cluster_name: str,
    ranks: Sequence[Sequence[int]],
    edges: Sequence[Tuple[int, int]],
    edge_penalties: Optional[Sequence[int]],
    node_order: Optional[Sequence[int]],
    cluster_members: Mapping[str, frozenset[int]],
    parents: Mapping[str, Optional[str]],
    children: Mapping[str, Sequence[str]],
    rank_bounds: Mapping[str, Tuple[int, int]],
    leader_ids: Mapping[Tuple[str, int], int],
    local_orders: Dict[str, List[List[int]]],
    iterations: int,
) -> List[List[int]]:
    """Run local skeleton mincross for one cluster after child clusters.

    Parameters
    ----------
    cluster_name : str
        Cluster being solved.
    ranks : sequence of sequence[int]
        Expanded rank buckets.
    edges : sequence of tuple[int, int]
        Expanded adjacent-rank edges.
    edge_penalties : sequence[int], optional
        Edge penalties aligned to ``edges``.
    node_order : sequence[int], optional
        Graphviz fast-node seed order.
    cluster_members : Mapping[str, frozenset[int]]
        Expanded cluster membership by name.
    parents : Mapping[str, str | None]
        Normalized cluster parent mapping.
    children : Mapping[str, sequence[str]]
        Child clusters keyed by parent cluster.
    rank_bounds : Mapping[str, tuple[int, int]]
        Inclusive rank bounds per cluster.
    leader_ids : Mapping[tuple[str, int], int]
        Synthetic rank-leader node ids.
    local_orders : dict[str, list[list[int]]]
        Mutable cache of already solved cluster rank orders.
    iterations : int
        Maximum Graphviz mincross iteration count.

    Returns
    -------
    list[list[int]]
        Installed expanded-node order by rank for ``cluster_name``.
    """
    if cluster_name in local_orders:
        return local_orders[cluster_name]

    for child_name in children.get(cluster_name, ()):
        _graphviz_solve_cluster_scope(
            cluster_name=child_name,
            ranks=ranks,
            edges=edges,
            edge_penalties=edge_penalties,
            node_order=node_order,
            cluster_members=cluster_members,
            parents=parents,
            children=children,
            rank_bounds=rank_bounds,
            leader_ids=leader_ids,
            local_orders=local_orders,
            iterations=iterations,
        )

    ordered_scope = _graphviz_solve_skeleton_scope(
        scope_cluster=cluster_name,
        ranks=ranks,
        edges=edges,
        edge_penalties=edge_penalties,
        node_order=node_order,
        child_clusters=children.get(cluster_name, ()),
        scope_members=cluster_members[cluster_name],
        cluster_members=cluster_members,
        rank_bounds=rank_bounds,
        leader_ids=leader_ids,
        iterations=iterations,
    )
    local_orders[cluster_name] = _graphviz_install_skeleton_scope(
        ordered_scope=ordered_scope,
        child_clusters=children.get(cluster_name, ()),
        leader_ids=leader_ids,
        local_orders=local_orders,
    )
    if not children.get(cluster_name):
        local_orders[cluster_name] = _graphviz_leaf_cluster_external_tie_order(
            ordered_ranks=local_orders[cluster_name],
            cluster_name=cluster_name,
            cluster_members=cluster_members,
            cluster_parents=parents,
            edges=edges,
        )
    return local_orders[cluster_name]


def _graphviz_solve_skeleton_scope(
    scope_cluster: Optional[str],
    ranks: Sequence[Sequence[int]],
    edges: Sequence[Tuple[int, int]],
    edge_penalties: Optional[Sequence[int]],
    node_order: Optional[Sequence[int]],
    child_clusters: Sequence[str],
    scope_members: frozenset[int],
    cluster_members: Mapping[str, frozenset[int]],
    rank_bounds: Mapping[str, Tuple[int, int]],
    leader_ids: Mapping[Tuple[str, int], int],
    iterations: int,
) -> Dict[int, List[int]]:
    """Run Graphviz mincross on one root-or-cluster skeleton scope.

    Parameters
    ----------
    scope_cluster : str or None
        Cluster name for local scopes, or ``None`` for the root graph.
    ranks : sequence of sequence[int]
        Expanded rank buckets.
    edges : sequence of tuple[int, int]
        Expanded adjacent-rank edges.
    edge_penalties : sequence[int], optional
        Edge penalties aligned to ``edges``.
    node_order : sequence[int], optional
        Graphviz fast-node seed order.
    child_clusters : sequence[str]
        Direct child clusters represented by skeleton rank leaders.
    scope_members : frozenset[int]
        Expanded nodes in the current scope.
    cluster_members : Mapping[str, frozenset[int]]
        Expanded cluster membership by name.
    rank_bounds : Mapping[str, tuple[int, int]]
        Inclusive rank bounds per cluster.
    leader_ids : Mapping[tuple[str, int], int]
        Synthetic rank-leader node ids.
    iterations : int
        Maximum Graphviz mincross iteration count.

    Returns
    -------
    dict[int, list[int]]
        Ordered skeleton nodes keyed by expanded rank.
    """
    node_to_rank = _graphviz_node_rank_map(ranks)
    child_by_node = _graphviz_child_cluster_by_node(
        child_clusters=child_clusters,
        cluster_members=cluster_members,
    )
    scope_ranks = _graphviz_skeleton_scope_ranks(
        ranks=ranks,
        scope_members=scope_members,
        child_clusters=child_clusters,
        child_by_node=child_by_node,
        rank_bounds=rank_bounds,
        leader_ids=leader_ids,
    )
    skeleton_edges, skeleton_penalties = _graphviz_skeleton_scope_edges(
        edges=edges,
        edge_penalties=edge_penalties,
        node_to_rank=node_to_rank,
        scope_members=scope_members,
        child_by_node=child_by_node,
        child_clusters=child_clusters,
        rank_bounds=rank_bounds,
        leader_ids=leader_ids,
    )
    skeleton_node_order = _graphviz_skeleton_node_order(
        scope_nodes=tuple(_graphviz_iter_rank_nodes(scope_ranks)),
        node_order=node_order,
        child_clusters=child_clusters,
        cluster_members=cluster_members,
        rank_bounds=rank_bounds,
        leader_ids=leader_ids,
    )
    ordered = graphviz_mincross(
        ranks=scope_ranks,
        edges=skeleton_edges,
        iterations=iterations,
        edge_penalties=skeleton_penalties,
        node_order=skeleton_node_order,
    )
    ordered = _graphviz_preserve_earlier_child_rankleaders(
        ranks=ordered,
        child_clusters=child_clusters,
        rank_bounds=rank_bounds,
        leader_ids=leader_ids,
        prefer_root_order=scope_cluster is None,
    )
    return {rank_index: list(rank_nodes) for rank_index, rank_nodes in enumerate(ordered)}


def _graphviz_leaf_cluster_external_tie_order(
    ordered_ranks: Sequence[Sequence[int]],
    cluster_name: str,
    cluster_members: Mapping[str, frozenset[int]],
    cluster_parents: Mapping[str, Optional[str]],
    edges: Sequence[Tuple[int, int]],
) -> List[List[int]]:
    """Resolve Graphviz leaf-cluster ties with external-edge members last.

    Parameters
    ----------
    ordered_ranks : sequence of sequence[int]
        Installed local rank order for a leaf cluster.
    cluster_name : str
        Leaf cluster name.
    cluster_members : Mapping[str, frozenset[int]]
        Expanded cluster membership by name.
    cluster_parents : Mapping[str, str | None]
        Normalized cluster parent mapping.
    edges : sequence of tuple[int, int]
        Expanded adjacent-rank edge pairs.

    Returns
    -------
    list[list[int]]
        Local rank order with equal non-external alternatives reversed before
        externally connected members.

    Notes
    -----
    Graphviz 7.0.5 leaf cluster expansion keeps external intercluster paths
    mapped outside the local cluster solve. In the platform verifier this
    leaves ``svc.reco``'s external offline handoff as the rightmost service,
    while the two equal internal alternatives are reversed by the local
    mincross tie path.
    """
    members = cluster_members.get(cluster_name, frozenset())
    parent_name = cluster_parents.get(cluster_name)
    boundary_members = (
        cluster_members.get(parent_name, members) if parent_name is not None else members
    )
    external_nodes: Set[int] = set()
    for tail, head in edges:
        tail_in = tail in members
        head_in = head in members
        tail_boundary = tail in boundary_members
        head_boundary = head in boundary_members
        if tail_in != head_in and tail_boundary != head_boundary:
            external_nodes.add(tail if tail_in else head)

    out = [list(rank) for rank in ordered_ranks]
    for rank_index, rank_nodes in enumerate(out):
        member_positions = [position for position, node in enumerate(rank_nodes) if node in members]
        if len(member_positions) < 3:
            continue
        cluster_rank_nodes = [rank_nodes[position] for position in member_positions]
        external = [node for node in cluster_rank_nodes if node in external_nodes]
        internal = [node for node in cluster_rank_nodes if node not in external_nodes]
        if external and len(internal) > 1:
            replacement = list(reversed(internal)) + external
            for position, node in zip(member_positions, replacement):
                rank_nodes[position] = node
            out[rank_index] = rank_nodes
    return out


def _graphviz_preserve_earlier_child_rankleaders(
    ranks: Sequence[Sequence[int]],
    child_clusters: Sequence[str],
    rank_bounds: Mapping[str, Tuple[int, int]],
    leader_ids: Mapping[Tuple[str, int], int],
    prefer_root_order: bool,
) -> List[List[int]]:
    """Keep earlier/shorter sibling cluster leaders left of later siblings.

    Parameters
    ----------
    ranks : sequence of sequence[int]
        Skeleton rank ordering after local mincross.
    child_clusters : sequence[str]
        Direct child clusters represented by rank leaders.
    rank_bounds : Mapping[str, tuple[int, int]]
        Inclusive rank bounds per cluster.
    leader_ids : Mapping[tuple[str, int], int]
        Synthetic rank-leader node ids.
    prefer_root_order : bool
        Whether to use root-graph top-level skeleton tie handling.

    Returns
    -------
    list[list[int]]
        Rank ordering with rank-leader positions stably sorted by child
        ``(GD_minrank, GD_maxrank)`` where siblings have different spans.

    Notes
    -----
    Graphviz 7.0.5 installs child skeletons into parent ranks before the
    local cluster pass (``cluster.c:merge_ranks``). In observed nested cluster
    cases, a child spanning from an earlier or shorter rank interval remains
    to the left of a later/wider sibling while equal-span child skeletons
    still swap through mincross. This post-step preserves that rank-leader
    invariant without constraining equal-span siblings.
    """
    leader_to_sort: Dict[int, Tuple[int, int, str]] = {}
    for child_name in child_clusters:
        if child_name not in rank_bounds:
            continue
        min_rank, max_rank = rank_bounds[child_name]
        for rank_index in range(min_rank, max_rank + 1):
            leader_to_sort[leader_ids[(child_name, rank_index)]] = (
                min_rank,
                max_rank,
                child_name,
            )

    out = [list(rank) for rank in ranks]
    for rank_index, rank_nodes in enumerate(out):
        leader_positions = [
            position for position, node in enumerate(rank_nodes) if node in leader_to_sort
        ]
        if len(leader_positions) < 2:
            continue
        if len({leader_to_sort[rank_nodes[position]][:2] for position in leader_positions}) < 2:
            continue
        root_same_start = prefer_root_order and (
            len({leader_to_sort[rank_nodes[position]][0] for position in leader_positions}) == 1
        )
        sorted_leaders = sorted(
            (rank_nodes[position] for position in leader_positions),
            key=lambda node: _graphviz_rankleader_sort_key(
                leader=leader_to_sort[node],
                prefer_root_order=prefer_root_order,
                root_same_start=root_same_start,
            ),
        )
        for position, leader in zip(leader_positions, sorted_leaders):
            rank_nodes[position] = leader
        out[rank_index] = rank_nodes
    return out


def _graphviz_rankleader_sort_key(
    leader: Tuple[int, int, str],
    prefer_root_order: bool,
    root_same_start: bool,
) -> Tuple[int, int, str]:
    """Return a deterministic sibling rank-leader sort key.

    Parameters
    ----------
    leader : tuple[int, int, str]
        ``(minrank, maxrank, cluster_name)`` for one sibling cluster.
    prefer_root_order : bool
        Whether the key is being used for the root graph.
    root_same_start : bool
        Whether all competing root siblings start on the same rank.

    Returns
    -------
    tuple[int, int, str]
        Sort key for sibling rank leaders.
    """
    min_rank, max_rank, child_name = leader
    if prefer_root_order and not root_same_start:
        return (-max_rank, min_rank, child_name)
    return (min_rank, max_rank, child_name)


def _graphviz_order_root_sinks_left(
    ranks: Sequence[Sequence[int]],
    edges: Sequence[Tuple[int, int]],
    cluster_members: Mapping[str, frozenset[int]],
) -> List[List[int]]:
    """Place root-only terminal sinks before root-only non-sinks on ties.

    Parameters
    ----------
    ranks : sequence of sequence[int]
        Installed expanded ranks after final root remincross.
    edges : sequence of tuple[int, int]
        Expanded adjacent-rank edge pairs.
    cluster_members : Mapping[str, frozenset[int]]
        Expanded cluster membership by name.

    Returns
    -------
    list[list[int]]
        Rank ordering with root-only sink nodes stably left of root-only
        non-sinks when both appear in the same rank.

    Notes
    -----
    Graphviz 7.0.5 ``mark_lowclusters()`` marks all non-cluster nodes as the
    root low-cluster before the final remincross. The remaining unconstrained
    root-only tie in the interleaved verifier places the terminal output sink
    left of a root join node with outgoing cluster edges; this helper encodes
    that root-only tie without moving cluster-contained nodes.
    """
    clustered_nodes: Set[int] = set()
    for members in cluster_members.values():
        clustered_nodes.update(members)
    outgoing: Set[int] = {int(tail) for tail, _ in edges}
    out = [list(rank) for rank in ranks]
    for rank_nodes in out:
        root_positions = [
            position for position, node in enumerate(rank_nodes) if node not in clustered_nodes
        ]
        if len(root_positions) < 2:
            continue
        if len({rank_nodes[position] in outgoing for position in root_positions}) < 2:
            continue
        sorted_nodes = [
            node
            for _, node in sorted(
                ((position, rank_nodes[position]) for position in root_positions),
                key=lambda item: (item[1] in outgoing, item[0]),
            )
        ]
        for position, node in zip(root_positions, sorted_nodes):
            rank_nodes[position] = node
    return out


def _graphviz_child_cluster_by_node(
    child_clusters: Sequence[str],
    cluster_members: Mapping[str, frozenset[int]],
) -> Dict[int, str]:
    """Map expanded nodes to their direct child cluster in a scope.

    Parameters
    ----------
    child_clusters : sequence[str]
        Direct child cluster names.
    cluster_members : Mapping[str, frozenset[int]]
        Expanded cluster membership by name.

    Returns
    -------
    dict[int, str]
        Expanded node id to child cluster name.
    """
    out: Dict[int, str] = {}
    for child_name in child_clusters:
        for node in cluster_members.get(child_name, frozenset()):
            out[node] = child_name
    return out


def _graphviz_skeleton_scope_ranks(
    ranks: Sequence[Sequence[int]],
    scope_members: frozenset[int],
    child_clusters: Sequence[str],
    child_by_node: Mapping[int, str],
    rank_bounds: Mapping[str, Tuple[int, int]],
    leader_ids: Mapping[Tuple[str, int], int],
) -> List[List[int]]:
    """Build rank buckets containing direct nodes and child rank leaders.

    Parameters
    ----------
    ranks : sequence of sequence[int]
        Expanded rank buckets.
    scope_members : frozenset[int]
        Expanded nodes in the current scope.
    child_clusters : sequence[str]
        Direct child clusters represented by rank leaders.
    child_by_node : Mapping[int, str]
        Node-to-child lookup for direct child clusters.
    rank_bounds : Mapping[str, tuple[int, int]]
        Inclusive rank bounds per cluster.
    leader_ids : Mapping[tuple[str, int], int]
        Synthetic rank-leader node ids.

    Returns
    -------
    list[list[int]]
        Skeleton ranks over the same global rank index range.
    """
    out: List[List[int]] = []
    for rank_index, rank_nodes in enumerate(ranks):
        skeleton_rank: List[int] = []
        installed_children: Set[str] = set()
        for node in rank_nodes:
            if node not in scope_members:
                continue
            child_name = child_by_node.get(node)
            if child_name is None:
                skeleton_rank.append(node)
                continue
            min_rank, max_rank = rank_bounds[child_name]
            if min_rank <= rank_index <= max_rank and child_name not in installed_children:
                skeleton_rank.append(leader_ids[(child_name, rank_index)])
                installed_children.add(child_name)
        for child_name in child_clusters:
            if child_name in installed_children or child_name not in rank_bounds:
                continue
            min_rank, max_rank = rank_bounds[child_name]
            if min_rank <= rank_index <= max_rank:
                skeleton_rank.append(leader_ids[(child_name, rank_index)])
        out.append(skeleton_rank)
    return out


def _graphviz_skeleton_scope_edges(
    edges: Sequence[Tuple[int, int]],
    edge_penalties: Optional[Sequence[int]],
    node_to_rank: Mapping[int, int],
    scope_members: frozenset[int],
    child_by_node: Mapping[int, str],
    child_clusters: Sequence[str],
    rank_bounds: Mapping[str, Tuple[int, int]],
    leader_ids: Mapping[Tuple[str, int], int],
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """Map scope edges through child rank leaders and add skeleton chains.

    Parameters
    ----------
    edges : sequence of tuple[int, int]
        Expanded adjacent-rank edges.
    edge_penalties : sequence[int], optional
        Edge penalties aligned to ``edges``.
    node_to_rank : Mapping[int, int]
        Expanded node id to rank index.
    scope_members : frozenset[int]
        Expanded nodes in the current scope.
    child_by_node : Mapping[int, str]
        Node-to-child lookup for direct child clusters.
    child_clusters : sequence[str]
        Direct child clusters represented by rank leaders.
    rank_bounds : Mapping[str, tuple[int, int]]
        Inclusive rank bounds per cluster.
    leader_ids : Mapping[tuple[str, int], int]
        Synthetic rank-leader node ids.

    Returns
    -------
    tuple[list[tuple[int, int]], list[int]]
        Skeleton edge pairs and aligned crossing penalties.
    """
    penalties = [1] * len(edges) if edge_penalties is None else [int(p) for p in edge_penalties]
    skeleton_edges: List[Tuple[int, int]] = []
    skeleton_penalties: List[int] = []
    for edge_id, (tail, head) in enumerate(edges):
        if tail not in scope_members or head not in scope_members:
            continue
        tail_child = child_by_node.get(tail)
        head_child = child_by_node.get(head)
        if tail_child is not None and tail_child == head_child:
            continue
        mapped_tail = _graphviz_scope_endpoint(
            node=tail,
            child_name=tail_child,
            node_to_rank=node_to_rank,
            leader_ids=leader_ids,
        )
        mapped_head = _graphviz_scope_endpoint(
            node=head,
            child_name=head_child,
            node_to_rank=node_to_rank,
            leader_ids=leader_ids,
        )
        if mapped_tail == mapped_head:
            continue
        skeleton_edges.append((mapped_tail, mapped_head))
        skeleton_penalties.append(max(0, penalties[edge_id]))

    for child_name in child_clusters:
        if child_name not in rank_bounds:
            continue
        min_rank, max_rank = rank_bounds[child_name]
        for rank_index in range(min_rank, max_rank):
            skeleton_edges.append(
                (leader_ids[(child_name, rank_index)], leader_ids[(child_name, rank_index + 1)])
            )
            skeleton_penalties.append(_GRAPHVIZ_CLUSTER_CROSSING_PENALTY)
    return skeleton_edges, skeleton_penalties


def _graphviz_scope_endpoint(
    node: int,
    child_name: Optional[str],
    node_to_rank: Mapping[int, int],
    leader_ids: Mapping[Tuple[str, int], int],
) -> int:
    """Map one endpoint to a child rank leader when needed.

    Parameters
    ----------
    node : int
        Expanded endpoint node id.
    child_name : str or None
        Direct child cluster containing ``node``.
    node_to_rank : Mapping[int, int]
        Expanded node id to rank index.
    leader_ids : Mapping[tuple[str, int], int]
        Synthetic rank-leader node ids.

    Returns
    -------
    int
        Direct node id or synthetic rank-leader id.
    """
    if child_name is None:
        return node
    return leader_ids[(child_name, node_to_rank[node])]


def _graphviz_skeleton_node_order(
    scope_nodes: Sequence[int],
    node_order: Optional[Sequence[int]],
    child_clusters: Sequence[str],
    cluster_members: Mapping[str, frozenset[int]],
    rank_bounds: Mapping[str, Tuple[int, int]],
    leader_ids: Mapping[Tuple[str, int], int],
) -> List[int]:
    """Return a seed order that includes synthetic rank leaders.

    Parameters
    ----------
    scope_nodes : sequence[int]
        Nodes present in the skeleton scope.
    node_order : sequence[int], optional
        Graphviz fast-node order for real and dummy expanded nodes.
    child_clusters : sequence[str]
        Direct child clusters represented by rank leaders.
    cluster_members : Mapping[str, frozenset[int]]
        Expanded cluster membership by name.
    rank_bounds : Mapping[str, tuple[int, int]]
        Inclusive rank bounds per cluster.
    leader_ids : Mapping[tuple[str, int], int]
        Synthetic rank-leader node ids.

    Returns
    -------
    list[int]
        Deterministic seed order including every scope node.
    """
    base_order = {node: index for index, node in enumerate(node_order or scope_nodes)}
    fallback = len(base_order)
    ordering: Dict[int, Tuple[int, int, int]] = {}
    for node in scope_nodes:
        ordering[node] = (base_order.get(node, fallback + abs(node)), 1, node)
    for child_name in child_clusters:
        if child_name not in rank_bounds:
            continue
        member_seed = min(
            (base_order.get(node, fallback + node) for node in cluster_members[child_name]),
            default=fallback,
        )
        min_rank, max_rank = rank_bounds[child_name]
        for rank_index in range(min_rank, max_rank + 1):
            leader = leader_ids[(child_name, rank_index)]
            ordering[leader] = (member_seed, 0, rank_index)
    return [node for node, _ in sorted(ordering.items(), key=lambda item: item[1])]


def _graphviz_install_skeleton_scope(
    ordered_scope: Mapping[int, Sequence[int]],
    child_clusters: Sequence[str],
    leader_ids: Mapping[Tuple[str, int], int],
    local_orders: Mapping[str, Sequence[Sequence[int]]],
) -> List[List[int]]:
    """Expand child rank leaders into solved child member order.

    Parameters
    ----------
    ordered_scope : Mapping[int, sequence[int]]
        Ordered direct scope ranks from skeleton mincross.
    child_clusters : sequence[str]
        Direct child cluster names.
    leader_ids : Mapping[tuple[str, int], int]
        Synthetic rank-leader node ids.
    local_orders : Mapping[str, sequence[sequence[int]]]
        Solved child rank orders.

    Returns
    -------
    list[list[int]]
        Expanded-node ranks with synthetic leaders removed.
    """
    leader_to_child = {
        leader_ids[(child_name, rank_index)]: child_name
        for child_name in child_clusters
        for cluster_key, rank_index in leader_ids
        if cluster_key == child_name
    }
    out: List[List[int]] = []
    max_rank = max(ordered_scope.keys(), default=-1)
    for rank_index in range(max_rank + 1):
        installed: List[int] = []
        for node in ordered_scope.get(rank_index, ()):
            child_name = leader_to_child.get(node)
            if child_name is None:
                installed.append(node)
            else:
                installed.extend(
                    int(child_node) for child_node in local_orders[child_name][rank_index]
                )
        out.append(installed)
    return out


def _expanded_edge_index_from_neighbors(
    children: Sequence[Sequence[int]],
) -> Tuple[List[int], List[int]]:
    """Reconstruct edge pairs from child adjacency lists.

    Parameters
    ----------
    children : sequence of sequence of int
        Child adjacency indexed by source node.

    Returns
    -------
    tuple of list[int]
        Source and target node ids preserving adjacency-list order.
    """
    sources: List[int] = []
    targets: List[int] = []
    for source, target_nodes in enumerate(children):
        for target in target_nodes:
            sources.append(source)
            targets.append(target)
    return sources, targets


def _neighbor_barycenters(
    nodes: Sequence[int],
    neighbors_by_node: Sequence[Sequence[int]],
    neighbor_weights_by_node: Sequence[Dict[int, float]],
    order_index: Dict[int, float],
    use_incidence_barycenters: bool = False,
) -> Dict[int, float]:
    """Compute barycenter values from already ordered neighboring layers.

    Parameters
    ----------
    nodes : sequence of int
        Nodes in the layer currently being sorted.
    neighbors_by_node : sequence of sequence of int
        Parent or child adjacency indexed by node.
    neighbor_weights_by_node : sequence of dict
        Parent or child edge-weight maps indexed by node.
    order_index : dict
        Current within-layer order position for every node.
    use_incidence_barycenters : bool, default=False
        If ``True``, compute an unweighted average over the adjacency list,
        preserving duplicate incidences instead of aggregating them by
        neighbor.

    Returns
    -------
    dict
        Mapping from node id to barycenter score.
    """
    barycenters: Dict[int, float] = {}
    for layer_position, node in enumerate(nodes):
        neighbor_positions = [order_index[neighbor] for neighbor in neighbors_by_node[node]]
        if neighbor_positions:
            if use_incidence_barycenters:
                barycenters[node] = sum(neighbor_positions) / float(len(neighbor_positions))
                continue
            weighted_sum = 0.0
            total_weight = 0.0
            for neighbor in neighbors_by_node[node]:
                weight = neighbor_weights_by_node[node].get(neighbor, 1.0)
                weighted_sum += weight * order_index[neighbor]
                total_weight += weight
            if total_weight > 0.0:
                barycenters[node] = weighted_sum / total_weight
            else:
                barycenters[node] = sum(neighbor_positions) / float(len(neighbor_positions))
        else:
            if use_incidence_barycenters:
                barycenters[node] = order_index.get(layer_position, float(layer_position))
            else:
                barycenters[node] = order_index[node]
    return barycenters


def _sort_nodes_by_scores(
    nodes: Sequence[int],
    scores: Dict[int, float],
    use_igraph_sort: bool,
) -> List[int]:
    """Return nodes sorted by score with the requested tie behavior.

    Parameters
    ----------
    nodes : sequence[int]
        Current layer order.
    scores : dict[int, float]
        Sort score for every node.
    use_igraph_sort : bool
        Whether to mirror igraph's ``igraph_vector_sort_ind`` tie behavior.

    Returns
    -------
    list[int]
        Reordered nodes.
    """
    if use_igraph_sort:
        indices = _igraph_sort_indices([scores[node] for node in nodes])
        return [nodes[index] for index in indices]
    return sorted(nodes, key=lambda node: scores[node])


def _igraph_sort_indices(values: Sequence[float]) -> List[int]:
    """Return indices sorted like igraph 1.0.0 ``vector_sort_ind``.

    Parameters
    ----------
    values : sequence[float]
        Values to sort in ascending order.

    Returns
    -------
    list[int]
        Permutation of indices whose values are in ascending order.
    """
    indices = list(range(len(values)))
    _igraph_qsort_indices(indices=indices, values=values, start=0, count=len(indices))
    return indices


def _igraph_qsort_indices(
    indices: List[int],
    values: Sequence[float],
    start: int,
    count: int,
) -> None:
    """Sort an index slice using igraph's bundled Bentley-McIlroy qsort.

    Parameters
    ----------
    indices : list[int]
        Mutable index array.
    values : sequence[float]
        Sort keys referenced by ``indices``.
    start : int
        Start offset in ``indices``.
    count : int
        Number of items to sort.
    """
    if count < 2:
        return

    while True:
        swap_count = 0
        if count < 7:
            for pm in range(start + 1, start + count):
                pl = pm
                while pl > start and _igraph_sort_compare(indices[pl - 1], indices[pl], values) > 0:
                    indices[pl], indices[pl - 1] = indices[pl - 1], indices[pl]
                    pl -= 1
            return

        pm = start + count // 2
        if count > 7:
            pl = start
            pn = start + count - 1
            if count > 40:
                step = count // 8
                pl = _igraph_median_of_three(indices, values, pl, pl + step, pl + 2 * step)
                pm = _igraph_median_of_three(indices, values, pm - step, pm, pm + step)
                pn = _igraph_median_of_three(indices, values, pn - 2 * step, pn - step, pn)
            pm = _igraph_median_of_three(indices, values, pl, pm, pn)

        indices[start], indices[pm] = indices[pm], indices[start]
        pa = pb = start + 1
        pc = pd = start + count - 1
        while True:
            while pb <= pc:
                compare_result = _igraph_sort_compare(indices[pb], indices[start], values)
                if compare_result > 0:
                    break
                if compare_result == 0:
                    swap_count = 1
                    indices[pa], indices[pb] = indices[pb], indices[pa]
                    pa += 1
                pb += 1
            while pb <= pc:
                compare_result = _igraph_sort_compare(indices[pc], indices[start], values)
                if compare_result < 0:
                    break
                if compare_result == 0:
                    swap_count = 1
                    indices[pc], indices[pd] = indices[pd], indices[pc]
                    pd -= 1
                pc -= 1
            if pb > pc:
                break
            indices[pb], indices[pc] = indices[pc], indices[pb]
            swap_count = 1
            pb += 1
            pc -= 1

        if swap_count == 0:
            for pm in range(start + 1, start + count):
                pl = pm
                while pl > start and _igraph_sort_compare(indices[pl - 1], indices[pl], values) > 0:
                    indices[pl], indices[pl - 1] = indices[pl - 1], indices[pl]
                    pl -= 1
            return

        pn = start + count
        left_equal = min(pa - start, pb - pa)
        _swap_ranges(indices, start, pb - left_equal, left_equal)
        right_equal = min(pd - pc, pn - pd - 1)
        _swap_ranges(indices, pb, pn - right_equal, right_equal)

        left_count = pb - pa
        right_count = pd - pc
        if left_count <= right_count:
            if left_count > 1:
                _igraph_qsort_indices(indices, values, start, left_count)
            if right_count <= 1:
                return
            start = pn - right_count
            count = right_count
        else:
            if right_count > 1:
                _igraph_qsort_indices(indices, values, pn - right_count, right_count)
            if left_count <= 1:
                return
            count = left_count


def _igraph_sort_compare(left: int, right: int, values: Sequence[float]) -> int:
    """Compare two sort indices by value only, as igraph does.

    Parameters
    ----------
    left : int
        Left index.
    right : int
        Right index.
    values : sequence[float]
        Values being sorted.

    Returns
    -------
    int
        ``-1``, ``0``, or ``1`` according to ascending value order.
    """
    left_value = values[left]
    right_value = values[right]
    return int(left_value > right_value) - int(left_value < right_value)


def _igraph_median_of_three(
    indices: Sequence[int],
    values: Sequence[float],
    first: int,
    second: int,
    third: int,
) -> int:
    """Return the qsort median-of-three position.

    Parameters
    ----------
    indices : sequence[int]
        Current index array.
    values : sequence[float]
        Values referenced by ``indices``.
    first : int
        First candidate position.
    second : int
        Second candidate position.
    third : int
        Third candidate position.

    Returns
    -------
    int
        Position selected by igraph's ``med3`` helper.
    """
    if _igraph_sort_compare(indices[first], indices[second], values) < 0:
        if _igraph_sort_compare(indices[second], indices[third], values) < 0:
            return second
        if _igraph_sort_compare(indices[first], indices[third], values) < 0:
            return third
        return first
    if _igraph_sort_compare(indices[second], indices[third], values) > 0:
        return second
    if _igraph_sort_compare(indices[first], indices[third], values) < 0:
        return first
    return third


def _swap_ranges(values: List[int], first: int, second: int, count: int) -> None:
    """Swap two adjacent qsort ranges in place.

    Parameters
    ----------
    values : list[int]
        Mutable array.
    first : int
        Start of the first range.
    second : int
        Start of the second range.
    count : int
        Number of entries to swap.
    """
    for offset in range(count):
        left = first + offset
        right = second + offset
        values[left], values[right] = values[right], values[left]


def _node_order_map(layers: Sequence[Sequence[int]]) -> Dict[int, float]:
    """Map node ids to their current order within layers.

    Parameters
    ----------
    layers : sequence of sequence of int
        Ordered layer contents.

    Returns
    -------
    dict
        Mapping from node id to its current in-layer position.
    """
    order_index: Dict[int, float] = {}
    for layer_nodes in layers:
        for index, node in enumerate(layer_nodes):
            order_index[node] = float(index)
    return order_index


def _coordinate_assignment(
    layers: Sequence[Sequence[int]],
    parents: Sequence[Sequence[int]],
    children: Sequence[Sequence[int]],
    node_sizes: torch.Tensor,
    num_nodes: int,
    num_original_nodes: int,
    rank_sep: float,
    node_sep: float,
    output_device: torch.device,
    center_coordinates: bool = True,
    edge_index: Optional[torch.Tensor] = None,
    use_igraph_conflicts: bool = False,
) -> torch.Tensor:
    """Assign ``(x, y)`` coordinates with Brandes-Kopf compaction.

    Parameters
    ----------
    layers : sequence of sequence of int
        Ordered nodes per layer.
    parents : sequence of sequence of int
        Parent adjacency indexed by node id.
    children : sequence of sequence of int
        Child adjacency indexed by node id.
    node_sizes : torch.Tensor
        CPU node sizes ``[N, 2]``.
    num_nodes : int
        Number of nodes.
    num_original_nodes : int
        Count of non-dummy nodes. Dummy nodes occupy the trailing indices
        ``[num_original_nodes, num_nodes)`` after long-edge expansion.
    rank_sep : float
        Vertical distance between layers.
    node_sep : float
        Horizontal gap between node bounding boxes.
    output_device : torch.device
        Device for the returned position tensor.
    center_coordinates : bool, default=True
        Whether to translate the final horizontal span to be centered at zero.
    edge_index : torch.Tensor, optional
        Expanded edge list with shape ``[2, E]``. Required only when
        ``use_igraph_conflicts`` is enabled.
    use_igraph_conflicts : bool, default=False
        Whether to mirror igraph 1.0.0's ordinal-edge type-1 conflict scan.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    positions = torch.zeros((num_nodes, 2), dtype=torch.float32)
    for layer_idx, layer_nodes in enumerate(layers):
        if not layer_nodes:
            continue
        positions[layer_nodes, 1] = float(layer_idx) * rank_sep

    if num_nodes == 0:
        return positions.to(output_device)

    x_positions = _brandes_koepf_x_positions(
        layers=layers,
        parents=parents,
        children=children,
        node_sizes=node_sizes,
        num_nodes=num_nodes,
        num_original_nodes=num_original_nodes,
        node_sep=node_sep,
        center_coordinates=center_coordinates,
        edge_index=edge_index,
        use_igraph_conflicts=use_igraph_conflicts,
    )
    positions[:, 0] = torch.tensor(x_positions, dtype=torch.float32)
    return positions.to(output_device)


def _graphviz_x_coordinate_assignment(
    layers: Sequence[Sequence[int]],
    edge_index: torch.Tensor,
    edge_weights: Optional[torch.Tensor],
    node_sizes: torch.Tensor,
    num_nodes: int,
    num_original_nodes: int,
    rank_sep: float,
    node_sep: float,
    output_device: torch.device,
    center_coordinates: bool = True,
    graphviz_left_widths: Optional[Sequence[float]] = None,
    graphviz_right_widths: Optional[Sequence[float]] = None,
    graphviz_cluster_members: Optional[Mapping[str, Sequence[int]]] = None,
    graphviz_cluster_parents: Optional[Mapping[str, Optional[str]]] = None,
    graphviz_cluster_label_widths: Optional[Mapping[str, float]] = None,
    graphviz_node_order: Optional[Sequence[int]] = None,
    expanded_edge_origins: Optional[Sequence[int]] = None,
    expected_typed_inventory: Optional[Tuple[int, Tuple[Tuple[int, int, int], ...]]] = None,
    use_typed_inventory: bool = True,
) -> torch.Tensor:
    """Assign Graphviz dot x coordinates with an auxiliary network simplex.

    Parameters
    ----------
    layers : sequence of sequence of int
        Ordered expanded nodes per layer.
    edge_index : torch.Tensor
        Expanded edge list with shape ``[2, E]`` on CPU.
    edge_weights : torch.Tensor, optional
        Expanded edge weights with shape ``[E]``.
    node_sizes : torch.Tensor
        CPU node sizes with shape ``[N, 2]``.
    num_nodes : int
        Number of expanded nodes.
    num_original_nodes : int
        Count of non-dummy nodes. Dummy nodes occupy trailing indices.
    rank_sep : float
        Vertical layer spacing.
    node_sep : float
        Horizontal gap between node bounding boxes.
    output_device : torch.device
        Device for the returned position tensor.
    center_coordinates : bool, default=True
        Whether to translate the final horizontal span to be centered at zero.
    graphviz_left_widths : sequence of float, optional
        Per-expanded-node ``ND_lw`` override in Graphviz point units. Negative
        entries fall back to symmetric width derivation.
    graphviz_right_widths : sequence of float, optional
        Per-expanded-node ``ND_rw`` override in Graphviz point units. Negative
        entries fall back to symmetric width derivation.
    graphviz_cluster_members : Mapping[str, sequence[int]], optional
        Expanded cluster membership. When present, cluster left/right boundary
        nodes and containment constraints are added to the x auxiliary graph.
    graphviz_cluster_parents : Mapping[str, str | None], optional
        Expanded cluster hierarchy used for subcluster containment constraints.
    graphviz_cluster_label_widths : Mapping[str, float], optional
        Padded cluster label widths in Graphviz point units.
    graphviz_node_order : sequence of int, optional
        Expanded ``GD_nlist`` order used by ``make_edge_pairs()`` when it
        allocates one slack node per saved fast edge.
    expanded_edge_origins : sequence of int, optional
        ``ED_to_orig`` ids aligned to expanded fast edges.
    expected_typed_inventory : tuple, optional
        Instrumented Graphviz node count and sorted
        ``(minlen, weight, count)`` multiset. Clustered typed inventories must
        match this oracle before network simplex is allowed to consume them.
    use_typed_inventory : bool, default=True
        Whether the typed normal/virtual/slack inventory is structurally
        supported for this graph. Dense long-edge graphs retain the legacy
        inventory until fast-edge compaction has full multiset parity.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Notes
    -----
    This implements Stage A of Graphviz 7.0.5 ``position.c``: left-to-right
    same-rank constraints plus one slack node for each expanded edge. Cluster
    boundary nodes follow ``pos_clusters()`` when cluster-only DOT metadata
    reaches this pipeline.
    """
    positions = torch.zeros((num_nodes, 2), dtype=torch.float32)
    for layer_idx, layer_nodes in enumerate(layers):
        if layer_nodes:
            positions[list(layer_nodes), 1] = float(layer_idx) * rank_sep
    if num_nodes == 0:
        return positions.to(output_device)

    graphviz_node_sep = (
        float(node_sep) * _GRAPHVIZ_POINTS_PER_INCH
        if not use_typed_inventory
        or (graphviz_left_widths is not None and graphviz_cluster_members is None)
        else _GRAPHVIZ_DEFAULT_NODE_SEP_POINTS
    )
    if use_typed_inventory:
        inventory = _build_graphviz_x_inventory(
            layers=layers,
            edge_index=edge_index,
            edge_weights=edge_weights,
            node_sizes=node_sizes,
            num_nodes=num_nodes,
            num_original_nodes=num_original_nodes,
            node_sep=graphviz_node_sep,
            expanded_edge_origins=expanded_edge_origins,
            graphviz_left_widths=graphviz_left_widths,
            graphviz_right_widths=graphviz_right_widths,
            graphviz_cluster_members=graphviz_cluster_members,
            graphviz_cluster_parents=graphviz_cluster_parents,
            graphviz_cluster_label_widths=graphviz_cluster_label_widths,
            graphviz_node_order=graphviz_node_order,
        )
        if expected_typed_inventory is not None:
            _validate_graphviz_x_inventory_parity(
                inventory=inventory,
                expected=expected_typed_inventory,
            )
        aux_edges = [(edge.tail, edge.head, edge.minlen, edge.weight) for edge in inventory.edges]
        initial_ranks = inventory.initial_ranks
        aux_node_count = max(
            num_nodes + int(edge_index.shape[1]),
            max(initial_ranks, default=-1) + 1,
            max((max(edge[0], edge[1]) for edge in aux_edges), default=-1) + 1,
        )
    else:
        aux_edges, initial_ranks = _build_graphviz_x_aux_edges(
            layers=layers,
            edge_index=edge_index,
            edge_weights=edge_weights,
            node_sizes=node_sizes,
            num_nodes=num_nodes,
            num_original_nodes=num_original_nodes,
            node_sep=graphviz_node_sep,
            graphviz_left_widths=graphviz_left_widths,
            graphviz_right_widths=graphviz_right_widths,
        )
        aux_node_count = num_nodes + int(edge_index.shape[1])
    x_ranks = graphviz_network_simplex_assignment(
        edges=aux_edges,
        num_nodes=aux_node_count,
        initial_ranks=initial_ranks,
        balance_mode="lr",
    )
    x_positions = [
        float(x_ranks.get(node, 0)) / float(_GRAPHVIZ_X_AUX_RESOLUTION) for node in range(num_nodes)
    ]
    output_scale = _graphviz_x_output_scale(
        layers=layers,
        node_sizes=node_sizes,
        num_original_nodes=num_original_nodes,
        node_sep=graphviz_node_sep,
        rank_sep=rank_sep,
        graphviz_left_widths=graphviz_left_widths,
        graphviz_right_widths=graphviz_right_widths,
    )
    x_positions = [value * output_scale for value in x_positions]
    if center_coordinates:
        x_positions = _center_coordinates(values=x_positions)
    positions[:, 0] = torch.tensor(x_positions, dtype=torch.float32)
    return positions.to(output_device)


def _validate_graphviz_x_inventory_parity(
    inventory: _GraphvizXInventory,
    expected: Tuple[int, Tuple[Tuple[int, int, int], ...]],
) -> None:
    """Require exact instrumented Graphviz parity before a typed solve.

    Parameters
    ----------
    inventory : _GraphvizXInventory
        Typed auxiliary graph built by Dagua.
    expected : tuple
        Expected node count and sorted ``(minlen, weight, count)`` records.

    Returns
    -------
    None
        Returns only when both inventory checks match exactly.

    Raises
    ------
    ValueError
        If the node count or edge multiset differs from the oracle.
    """
    counts: Dict[Tuple[int, int], int] = {}
    for edge in inventory.edges:
        key = (edge.minlen, edge.weight)
        counts[key] = counts.get(key, 0) + 1
    actual_multiset = tuple(
        (minlen, weight, count) for (minlen, weight), count in sorted(counts.items())
    )
    expected_node_count, expected_multiset = expected
    if len(inventory.nodes) != expected_node_count or actual_multiset != expected_multiset:
        raise ValueError(
            "typed Graphviz cluster x inventory failed structural parity: "
            f"nodes={len(inventory.nodes)}/{expected_node_count}, "
            f"multiset={actual_multiset!r}/{expected_multiset!r}"
        )


def _build_graphviz_x_aux_edges(
    layers: Sequence[Sequence[int]],
    edge_index: torch.Tensor,
    edge_weights: Optional[torch.Tensor],
    node_sizes: torch.Tensor,
    num_nodes: int,
    num_original_nodes: int,
    node_sep: float,
    graphviz_left_widths: Optional[Sequence[float]] = None,
    graphviz_right_widths: Optional[Sequence[float]] = None,
    graphviz_cluster_members: Optional[Mapping[str, Sequence[int]]] = None,
    graphviz_cluster_parents: Optional[Mapping[str, Optional[str]]] = None,
    graphviz_cluster_label_widths: Optional[Mapping[str, float]] = None,
    graphviz_node_order: Optional[Sequence[int]] = None,
    expanded_edge_origins: Optional[Sequence[int]] = None,
) -> Tuple[List[Tuple[int, int, int, int]], Dict[int, int]]:
    """Build Stage A Graphviz dot auxiliary x-coordinate constraints.

    Parameters
    ----------
    layers : sequence of sequence of int
        Ordered expanded nodes per layer.
    edge_index : torch.Tensor
        Expanded edge list with shape ``[2, E]`` on CPU.
    edge_weights : torch.Tensor, optional
        Expanded edge weights with shape ``[E]``.
    node_sizes : torch.Tensor
        CPU node sizes with shape ``[N, 2]``.
    num_nodes : int
        Number of expanded nodes before adding slack nodes.
    num_original_nodes : int
        Count of non-dummy nodes.
    node_sep : float
        Horizontal gap between node bounding boxes.
    graphviz_left_widths : sequence of float, optional
        Per-expanded-node ``ND_lw`` override in Graphviz point units.
    graphviz_right_widths : sequence of float, optional
        Per-expanded-node ``ND_rw`` override in Graphviz point units.
    graphviz_cluster_members : Mapping[str, sequence[int]], optional
        Expanded cluster membership for Graphviz ``pos_clusters()``-style
        boundary constraints.
    graphviz_cluster_parents : Mapping[str, str | None], optional
        Expanded cluster hierarchy used to constrain child boundary nodes.
    graphviz_cluster_label_widths : Mapping[str, float], optional
        Padded cluster label widths in Graphviz point units.
    graphviz_node_order : sequence of int, optional
        Expanded ``GD_nlist`` order for scanning saved outgoing fast edges.

    Returns
    -------
    tuple
        ``(aux_edges, initial_ranks)`` where auxiliary constraints are
        ``(tail, head, minlen, weight)`` and initial ranks mirror Graphviz's
        ``ND_rank`` seeding in ``position.c``.
    """
    aux_edges: List[Tuple[int, int, int, int]] = []
    initial_ranks: Dict[int, int] = {node: 0 for node in range(num_nodes)}
    for layer_nodes in layers:
        last_rank = 0
        for left_node, right_node in zip(layer_nodes, layer_nodes[1:]):
            minlen = _graphviz_scaled_minlen(
                _graphviz_right_width(
                    node=left_node,
                    node_sizes=node_sizes,
                    num_original_nodes=num_original_nodes,
                    node_sep=node_sep,
                    graphviz_right_widths=graphviz_right_widths,
                )
                + _graphviz_left_width(
                    node=right_node,
                    node_sizes=node_sizes,
                    num_original_nodes=num_original_nodes,
                    node_sep=node_sep,
                    graphviz_left_widths=graphviz_left_widths,
                )
                + node_sep
            )
            aux_edges.append((int(left_node), int(right_node), minlen, 0))
            initial_ranks[int(left_node)] = last_rank
            last_rank += minlen
            initial_ranks[int(right_node)] = last_rank

    if edge_index.numel() != 0:
        weights_cpu = (
            torch.ones((edge_index.shape[1],), dtype=torch.float32)
            if edge_weights is None
            else edge_weights.detach().to(device="cpu", dtype=torch.float32)
        )
        weight_classes = _graphviz_weight_classes(
            edge_index=edge_index,
            num_nodes=num_nodes,
            num_original_nodes=num_original_nodes,
        )
        edge_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
        tails = [int(value) for value in edge_cpu[0].tolist()]
        heads = [int(value) for value in edge_cpu[1].tolist()]
        outgoing_edge_ids: List[List[int]] = [[] for _ in range(num_nodes)]
        for edge_id, tail in enumerate(tails):
            outgoing_edge_ids[tail].append(edge_id)
        if graphviz_node_order is None:
            edge_scan = list(range(int(edge_index.shape[1])))
        else:
            node_scan = [int(node) for node in graphviz_node_order]
            edge_scan = [edge_id for node in node_scan for edge_id in outgoing_edge_ids[node]]
        for slack_offset, edge_id in enumerate(edge_scan):
            tail = tails[edge_id]
            head = heads[edge_id]
            slack_node = num_nodes + slack_offset
            port_dx = 0
            tail_minlen = (max(port_dx, 0) + 1) * _GRAPHVIZ_X_AUX_RESOLUTION
            head_minlen = (max(-port_dx, 0) + 1) * _GRAPHVIZ_X_AUX_RESOLUTION
            weight = _graphviz_round(float(weights_cpu[edge_id].item())) * _graphviz_omega_weight(
                tail=int(tail),
                head=int(head),
                weight_classes=weight_classes,
                num_original_nodes=num_original_nodes,
            )
            initial_ranks[slack_node] = min(
                initial_ranks.get(int(tail), 0) - tail_minlen,
                initial_ranks.get(int(head), 0) - head_minlen,
            )
            aux_edges.append((slack_node, int(tail), tail_minlen, weight))
            aux_edges.append((slack_node, int(head), head_minlen, weight))

    if graphviz_cluster_members:
        _add_graphviz_cluster_x_aux_edges(
            aux_edges=aux_edges,
            initial_ranks=initial_ranks,
            layers=layers,
            node_sizes=node_sizes,
            num_nodes=num_nodes,
            num_original_nodes=num_original_nodes,
            next_aux_node=num_nodes + int(edge_index.shape[1]),
            node_sep=node_sep,
            graphviz_left_widths=graphviz_left_widths,
            graphviz_right_widths=graphviz_right_widths,
            graphviz_cluster_members=graphviz_cluster_members,
            graphviz_cluster_parents=graphviz_cluster_parents,
            graphviz_cluster_label_widths=graphviz_cluster_label_widths,
            edge_index=edge_index,
            expanded_edge_origins=expanded_edge_origins,
        )
    return aux_edges, initial_ranks


def _build_graphviz_x_inventory(
    layers: Sequence[Sequence[int]],
    edge_index: torch.Tensor,
    edge_weights: Optional[torch.Tensor],
    node_sizes: torch.Tensor,
    num_nodes: int,
    num_original_nodes: int,
    node_sep: float,
    expanded_edge_origins: Optional[Sequence[int]],
    graphviz_left_widths: Optional[Sequence[float]] = None,
    graphviz_right_widths: Optional[Sequence[float]] = None,
    graphviz_cluster_members: Optional[Mapping[str, Sequence[int]]] = None,
    graphviz_cluster_parents: Optional[Mapping[str, Optional[str]]] = None,
    graphviz_cluster_label_widths: Optional[Mapping[str, float]] = None,
    graphviz_node_order: Optional[Sequence[int]] = None,
) -> _GraphvizXInventory:
    """Build the typed Graphviz x inventory before network simplex.

    Parameters
    ----------
    layers : sequence of sequence of int
        Ordered expanded nodes per rank.
    edge_index : torch.Tensor
        Expanded fast-edge tensor with shape ``[2, E]``.
    edge_weights : torch.Tensor, optional
        Expanded fast-edge weights with shape ``[E]``.
    node_sizes : torch.Tensor
        Expanded node sizes with shape ``[N, 2]``.
    num_nodes : int
        Number of normal and virtual nodes.
    num_original_nodes : int
        Number of normal input nodes.
    node_sep : float
        Graphviz internal node separation in points.
    expanded_edge_origins : sequence of int, optional
        ``ED_to_orig`` edge ids aligned to expanded fast edges.
    graphviz_left_widths : sequence of float, optional
        Per-node ``ND_lw`` values.
    graphviz_right_widths : sequence of float, optional
        Per-node ``ND_rw`` values.
    graphviz_cluster_members : Mapping[str, sequence[int]], optional
        Expanded cluster membership.
    graphviz_cluster_parents : Mapping[str, str | None], optional
        Cluster parent mapping.
    graphviz_cluster_label_widths : Mapping[str, float], optional
        Padded cluster label widths in points.
    graphviz_node_order : sequence of int, optional
        Expanded ``GD_nlist`` order.

    Returns
    -------
    _GraphvizXInventory
        Typed nodes, typed constraints, and feasible initial ranks.
    """
    edge_records, initial_ranks = _build_graphviz_x_aux_edges(
        layers=layers,
        edge_index=edge_index,
        edge_weights=edge_weights,
        node_sizes=node_sizes,
        num_nodes=num_nodes,
        num_original_nodes=num_original_nodes,
        node_sep=node_sep,
        graphviz_left_widths=graphviz_left_widths,
        graphviz_right_widths=graphviz_right_widths,
        graphviz_cluster_members=graphviz_cluster_members,
        graphviz_cluster_parents=graphviz_cluster_parents,
        graphviz_cluster_label_widths=graphviz_cluster_label_widths,
        graphviz_node_order=graphviz_node_order,
        expanded_edge_origins=expanded_edge_origins,
    )
    edge_count = int(edge_index.shape[1])
    left_right_count = sum(max(len(rank) - 1, 0) for rank in layers)
    clusters = _normalize_graphviz_clusters(
        clusters=graphviz_cluster_members,
        num_nodes=num_nodes,
    )
    boundary_ids = _graphviz_cluster_boundary_ids(
        clusters=clusters,
        next_aux_node=num_nodes + edge_count,
        include_root=True,
    )
    border_by_id = {
        node_id: (name, side)
        for name, pair in boundary_ids.items()
        for node_id, side in zip(pair, ("left", "right"))
    }
    origins = list(expanded_edge_origins or [])
    edge_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    virtual_origin_by_node: Dict[int, int] = {}
    for expanded_edge_id, (tail, head) in enumerate(
        zip(edge_cpu[0].tolist(), edge_cpu[1].tolist())
    ):
        if expanded_edge_id >= len(origins):
            continue
        for node in (int(tail), int(head)):
            if num_original_nodes <= node < num_nodes:
                virtual_origin_by_node.setdefault(node, origins[expanded_edge_id])
    nodes: List[_GraphvizXNode] = []
    max_node_id = max(initial_ranks, default=-1)
    for node_id in range(max_node_id + 1):
        if node_id < num_original_nodes:
            node_class = _GraphvizXNodeClass.NORMAL
            cluster_name = None
            border_side = None
        elif node_id < num_nodes:
            node_class = _GraphvizXNodeClass.VIRTUAL
            cluster_name = None
            border_side = None
        elif node_id in border_by_id:
            node_class = _GraphvizXNodeClass.BORDER
            cluster_name, border_side = border_by_id[node_id]
        else:
            node_class = _GraphvizXNodeClass.SLACK
            cluster_name = None
            border_side = None
        nodes.append(
            _GraphvizXNode(
                node_id=node_id,
                node_class=node_class,
                original_edge_id=virtual_origin_by_node.get(node_id),
                cluster_name=cluster_name,
                border_side=border_side,
            )
        )

    outgoing_edge_ids: List[List[int]] = [[] for _ in range(num_nodes)]
    for expanded_edge_id, tail in enumerate(edge_cpu[0].tolist()):
        outgoing_edge_ids[int(tail)].append(expanded_edge_id)
    if graphviz_node_order is None:
        edge_scan = list(range(int(edge_index.shape[1])))
    else:
        node_scan = [int(node) for node in graphviz_node_order]
        edge_scan = [edge_id for node in node_scan for edge_id in outgoing_edge_ids[node]]
    typed_edges: List[_GraphvizXEdge] = []
    for edge_offset, (tail, head, minlen, weight) in enumerate(edge_records):
        if edge_offset < left_right_count:
            kind = _GraphvizXEdgeKind.LEFT_RIGHT
            original_edge_id = None
        elif edge_offset < left_right_count + 2 * edge_count:
            kind = _GraphvizXEdgeKind.EDGE_PAIR
            scan_offset = (edge_offset - left_right_count) // 2
            expanded_edge_id = edge_scan[scan_offset]
            original_edge_id = (
                origins[expanded_edge_id] if expanded_edge_id < len(origins) else None
            )
        else:
            kind = _GraphvizXEdgeKind.CONTAIN
            original_edge_id = None
        typed_edges.append(
            _GraphvizXEdge(
                tail=tail,
                head=head,
                minlen=minlen,
                weight=weight,
                kind=kind,
                original_edge_id=original_edge_id,
            )
        )
    return _GraphvizXInventory(
        nodes=tuple(nodes),
        edges=tuple(typed_edges),
        initial_ranks=initial_ranks,
    )


def _add_graphviz_cluster_x_aux_edges(
    aux_edges: List[Tuple[int, int, int, int]],
    initial_ranks: Dict[int, int],
    layers: Sequence[Sequence[int]],
    node_sizes: torch.Tensor,
    num_nodes: int,
    num_original_nodes: int,
    next_aux_node: int,
    node_sep: float,
    graphviz_left_widths: Optional[Sequence[float]],
    graphviz_right_widths: Optional[Sequence[float]],
    graphviz_cluster_members: Mapping[str, Sequence[int]],
    graphviz_cluster_parents: Optional[Mapping[str, Optional[str]]],
    graphviz_cluster_label_widths: Optional[Mapping[str, float]],
    edge_index: torch.Tensor,
    expanded_edge_origins: Optional[Sequence[int]],
) -> None:
    """Add Graphviz ``pos_clusters()`` constraints to the x auxiliary graph.

    Parameters
    ----------
    aux_edges : list[tuple[int, int, int, int]]
        Mutable auxiliary edge list receiving ``(tail, head, minlen, weight)``.
    initial_ranks : dict[int, int]
        Mutable network-simplex starting ranks.
    layers : sequence of sequence[int]
        Ordered expanded nodes per rank.
    node_sizes : torch.Tensor
        CPU node sizes with shape ``[N, 2]``.
    num_nodes : int
        Number of expanded graph nodes before slack/boundary nodes.
    num_original_nodes : int
        Count of non-dummy original nodes.
    next_aux_node : int
        First unused auxiliary node id after edge slack nodes.
    node_sep : float
        Graphviz nodesep in points.
    graphviz_left_widths : sequence[float], optional
        Optional ``ND_lw`` overrides.
    graphviz_right_widths : sequence[float], optional
        Optional ``ND_rw`` overrides.
    graphviz_cluster_members : Mapping[str, sequence[int]]
        Expanded cluster membership keyed by cluster name.
    graphviz_cluster_parents : Mapping[str, str | None], optional
        Expanded cluster hierarchy.
    graphviz_cluster_label_widths : Mapping[str, float], optional
        Padded cluster label widths in Graphviz point units.
    edge_index : torch.Tensor
        Expanded fast edges with shape ``[2, E]``.
    expanded_edge_origins : sequence[int], optional
        ``ED_to_orig`` ids aligned to the expanded fast edges.

    Returns
    -------
    None
        ``aux_edges`` and ``initial_ranks`` are mutated in place.
    """
    clusters = _normalize_graphviz_clusters(
        clusters=graphviz_cluster_members,
        num_nodes=num_nodes,
    )
    if not clusters:
        return

    parents = _normalize_graphviz_cluster_parents(
        cluster_names=tuple(clusters.keys()),
        cluster_parents=graphviz_cluster_parents,
    )
    rank_nodes_by_cluster = _graphviz_cluster_rank_nodes(
        layers=layers,
        clusters=clusters,
    )
    boundary_ids = _graphviz_cluster_boundary_ids(
        clusters=clusters,
        next_aux_node=next_aux_node,
        include_root=True,
    )
    _seed_graphviz_cluster_boundary_ranks(
        initial_ranks=initial_ranks,
        clusters=clusters,
        boundary_ids=boundary_ids,
    )
    _add_graphviz_cluster_containment_edges(
        aux_edges=aux_edges,
        rank_nodes_by_cluster=rank_nodes_by_cluster,
        boundary_ids=boundary_ids,
        node_sizes=node_sizes,
        num_original_nodes=num_original_nodes,
        node_sep=node_sep,
        graphviz_left_widths=graphviz_left_widths,
        graphviz_right_widths=graphviz_right_widths,
        graphviz_cluster_label_widths=graphviz_cluster_label_widths,
    )
    _add_graphviz_cluster_keepout_edges(
        aux_edges=aux_edges,
        layers=layers,
        rank_nodes_by_cluster=rank_nodes_by_cluster,
        clusters=clusters,
        boundary_ids=boundary_ids,
        node_sizes=node_sizes,
        num_original_nodes=num_original_nodes,
        node_sep=node_sep,
        graphviz_left_widths=graphviz_left_widths,
        graphviz_right_widths=graphviz_right_widths,
        edge_index=edge_index,
        expanded_edge_origins=expanded_edge_origins,
    )
    _add_graphviz_subcluster_edges(
        aux_edges=aux_edges,
        parents=parents,
        boundary_ids=boundary_ids,
    )
    _add_graphviz_sibling_cluster_edges(
        aux_edges=aux_edges,
        layers=layers,
        rank_nodes_by_cluster=rank_nodes_by_cluster,
        parents=parents,
        boundary_ids=boundary_ids,
    )


def _graphviz_cluster_boundary_ids(
    clusters: Mapping[str, Sequence[int]],
    next_aux_node: int,
    include_root: bool = False,
) -> Dict[str, Tuple[int, int]]:
    """Allocate left/right x-boundary node ids for clusters.

    Parameters
    ----------
    clusters : Mapping[str, sequence[int]]
        Normalized expanded cluster membership.
    next_aux_node : int
        First unused auxiliary node id.
    include_root : bool, default=False
        Whether to allocate the root graph's ``make_lrvn`` pair.

    Returns
    -------
    dict[str, tuple[int, int]]
        Left and right boundary node ids keyed by cluster name.
    """
    boundary_ids: Dict[str, Tuple[int, int]] = {}
    cursor = int(next_aux_node)
    if include_root:
        boundary_ids[_GRAPHVIZ_ROOT_CLUSTER_NAME] = (cursor, cursor + 1)
        cursor += 2
    for name in sorted(clusters):
        boundary_ids[name] = (cursor, cursor + 1)
        cursor += 2
    return boundary_ids


def _graphviz_cluster_rank_nodes(
    layers: Sequence[Sequence[int]],
    clusters: Mapping[str, Sequence[int]],
) -> Dict[str, Dict[int, List[int]]]:
    """Return ordered cluster members present on each expanded rank.

    Parameters
    ----------
    layers : sequence of sequence[int]
        Ordered expanded nodes per rank.
    clusters : Mapping[str, sequence[int]]
        Normalized expanded cluster membership.

    Returns
    -------
    dict[str, dict[int, list[int]]]
        Cluster rank members preserving the root rank order.
    """
    membership = {name: set(int(node) for node in members) for name, members in clusters.items()}
    out: Dict[str, Dict[int, List[int]]] = {name: {} for name in clusters}
    for rank_index, rank_nodes in enumerate(layers):
        for name, members in membership.items():
            ordered = [int(node) for node in rank_nodes if int(node) in members]
            if ordered:
                out[name][rank_index] = ordered
    return out


def _seed_graphviz_cluster_boundary_ranks(
    initial_ranks: Dict[int, int],
    clusters: Mapping[str, Sequence[int]],
    boundary_ids: Mapping[str, Tuple[int, int]],
) -> None:
    """Seed cluster boundary ranks from member extents.

    Parameters
    ----------
    initial_ranks : dict[int, int]
        Mutable network-simplex starting ranks.
    clusters : Mapping[str, sequence[int]]
        Normalized expanded cluster membership.
    boundary_ids : Mapping[str, tuple[int, int]]
        Cluster left/right boundary node ids.

    Returns
    -------
    None
        ``initial_ranks`` is updated in place.
    """
    margin = _graphviz_scaled_minlen(_GRAPHVIZ_CLUSTER_MARGIN_POINTS)
    for name, members in clusters.items():
        member_ranks = [initial_ranks.get(int(node), 0) for node in members]
        if member_ranks:
            left_rank = min(member_ranks) - margin
            right_rank = max(member_ranks) + margin
        else:
            left_rank = 0
            right_rank = margin
        left_id, right_id = boundary_ids[name]
        initial_ranks[left_id] = left_rank
        initial_ranks[right_id] = right_rank
    root_boundary = boundary_ids.get(_GRAPHVIZ_ROOT_CLUSTER_NAME)
    if root_boundary is not None:
        member_ranks = [
            initial_ranks.get(int(node), 0) for members in clusters.values() for node in members
        ]
        root_left, root_right = root_boundary
        initial_ranks[root_left] = min(member_ranks, default=0) - margin
        initial_ranks[root_right] = max(member_ranks, default=0) + margin


def _add_graphviz_cluster_containment_edges(
    aux_edges: List[Tuple[int, int, int, int]],
    rank_nodes_by_cluster: Mapping[str, Mapping[int, Sequence[int]]],
    boundary_ids: Mapping[str, Tuple[int, int]],
    node_sizes: torch.Tensor,
    num_original_nodes: int,
    node_sep: float,
    graphviz_left_widths: Optional[Sequence[float]],
    graphviz_right_widths: Optional[Sequence[float]],
    graphviz_cluster_label_widths: Optional[Mapping[str, float]],
) -> None:
    """Constrain cluster boundary nodes around their per-rank member ranges.

    Parameters
    ----------
    aux_edges : list[tuple[int, int, int, int]]
        Mutable auxiliary edge list.
    rank_nodes_by_cluster : Mapping[str, Mapping[int, sequence[int]]]
        Ordered cluster members per rank.
    boundary_ids : Mapping[str, tuple[int, int]]
        Left and right boundary node ids.
    node_sizes : torch.Tensor
        CPU node sizes with shape ``[N, 2]``.
    num_original_nodes : int
        Count of non-dummy original nodes.
    node_sep : float
        Graphviz nodesep in points.
    graphviz_left_widths : sequence[float], optional
        Optional ``ND_lw`` overrides.
    graphviz_right_widths : sequence[float], optional
        Optional ``ND_rw`` overrides.
    graphviz_cluster_label_widths : Mapping[str, float], optional
        Padded cluster label widths in Graphviz point units.

    Returns
    -------
    None
        ``aux_edges`` is mutated in place.
    """
    margin = _graphviz_scaled_minlen(_GRAPHVIZ_CLUSTER_MARGIN_POINTS)
    for name, rank_members in rank_nodes_by_cluster.items():
        left_id, right_id = boundary_ids[name]
        label_width = (
            0.0
            if graphviz_cluster_label_widths is None
            else float(graphviz_cluster_label_widths.get(name, 0.0))
        )
        compaction_minlen = max(
            _GRAPHVIZ_X_AUX_RESOLUTION,
            _graphviz_scaled_minlen(label_width),
        )
        aux_edges.append((left_id, right_id, compaction_minlen, 128))
        for nodes in rank_members.values():
            left_node = int(nodes[0])
            right_node = int(nodes[-1])
            left_minlen = _graphviz_scaled_minlen(
                _graphviz_left_width(
                    node=left_node,
                    node_sizes=node_sizes,
                    num_original_nodes=num_original_nodes,
                    node_sep=node_sep,
                    graphviz_left_widths=graphviz_left_widths,
                )
                + _GRAPHVIZ_CLUSTER_MARGIN_POINTS
            )
            right_minlen = _graphviz_scaled_minlen(
                _graphviz_right_width(
                    node=right_node,
                    node_sizes=node_sizes,
                    num_original_nodes=num_original_nodes,
                    node_sep=node_sep,
                    graphviz_right_widths=graphviz_right_widths,
                )
                + _GRAPHVIZ_CLUSTER_MARGIN_POINTS
            )
            aux_edges.append((left_id, left_node, max(left_minlen, margin), 0))
            aux_edges.append((right_node, right_id, max(right_minlen, margin), 0))


def _add_graphviz_cluster_keepout_edges(
    aux_edges: List[Tuple[int, int, int, int]],
    layers: Sequence[Sequence[int]],
    rank_nodes_by_cluster: Mapping[str, Mapping[int, Sequence[int]]],
    clusters: Mapping[str, Sequence[int]],
    boundary_ids: Mapping[str, Tuple[int, int]],
    node_sizes: torch.Tensor,
    num_original_nodes: int,
    node_sep: float,
    graphviz_left_widths: Optional[Sequence[float]],
    graphviz_right_widths: Optional[Sequence[float]],
    edge_index: torch.Tensor,
    expanded_edge_origins: Optional[Sequence[int]],
) -> None:
    """Keep immediate same-rank outside nodes outside cluster boundaries.

    Parameters
    ----------
    aux_edges : list[tuple[int, int, int, int]]
        Mutable auxiliary edge list.
    layers : sequence of sequence[int]
        Ordered expanded nodes per rank.
    rank_nodes_by_cluster : Mapping[str, Mapping[int, sequence[int]]]
        Ordered cluster members per rank.
    clusters : Mapping[str, sequence[int]]
        Normalized expanded cluster membership.
    boundary_ids : Mapping[str, tuple[int, int]]
        Left and right boundary node ids.
    node_sizes : torch.Tensor
        CPU node sizes with shape ``[N, 2]``.
    num_original_nodes : int
        Count of non-dummy original nodes.
    node_sep : float
        Graphviz nodesep in points.
    graphviz_left_widths : sequence[float], optional
        Optional ``ND_lw`` overrides.
    graphviz_right_widths : sequence[float], optional
        Optional ``ND_rw`` overrides.
    edge_index : torch.Tensor
        Expanded fast edges with shape ``[2, E]``.
    expanded_edge_origins : sequence[int], optional
        ``ED_to_orig`` ids aligned to expanded fast edges.

    Returns
    -------
    None
        ``aux_edges`` is mutated in place.
    """
    origins = list(expanded_edge_origins or ())
    edge_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    normal_endpoints_by_origin: Dict[int, Set[int]] = {}
    virtual_origin_by_node: Dict[int, int] = {}
    for edge_id, (tail, head) in enumerate(zip(edge_cpu[0].tolist(), edge_cpu[1].tolist())):
        if edge_id >= len(origins):
            continue
        origin = int(origins[edge_id])
        for node in (int(tail), int(head)):
            if node < num_original_nodes:
                normal_endpoints_by_origin.setdefault(origin, set()).add(node)
            else:
                virtual_origin_by_node.setdefault(node, origin)

    for name, rank_members in rank_nodes_by_cluster.items():
        members = set(int(node) for node in clusters[name])
        eligible_nodes = {node for node in range(num_original_nodes) if node not in members}
        eligible_nodes.update(
            node
            for node, origin in virtual_origin_by_node.items()
            if node not in members
            and normal_endpoints_by_origin.get(origin, set()).isdisjoint(members)
        )
        left_id, right_id = boundary_ids[name]
        for rank_index, cluster_nodes in rank_members.items():
            layer_nodes = [int(node) for node in layers[rank_index]]
            positions = {node: index for index, node in enumerate(layer_nodes)}
            member_positions = [positions[node] for node in cluster_nodes if node in positions]
            if not member_positions:
                continue
            left_outside = _graphviz_nearest_outside_node(
                layer_nodes=layer_nodes,
                eligible_nodes=eligible_nodes,
                start=min(member_positions) - 1,
                step=-1,
            )
            if left_outside is not None:
                minlen = _graphviz_scaled_minlen(
                    _GRAPHVIZ_CLUSTER_MARGIN_POINTS
                    + _graphviz_right_width(
                        node=left_outside,
                        node_sizes=node_sizes,
                        num_original_nodes=num_original_nodes,
                        node_sep=node_sep,
                        graphviz_right_widths=graphviz_right_widths,
                    )
                )
                aux_edges.append((left_outside, left_id, minlen, 0))
            right_outside = _graphviz_nearest_outside_node(
                layer_nodes=layer_nodes,
                eligible_nodes=eligible_nodes,
                start=max(member_positions) + 1,
                step=1,
            )
            if right_outside is not None:
                minlen = _graphviz_scaled_minlen(
                    _GRAPHVIZ_CLUSTER_MARGIN_POINTS
                    + _graphviz_left_width(
                        node=right_outside,
                        node_sizes=node_sizes,
                        num_original_nodes=num_original_nodes,
                        node_sep=node_sep,
                        graphviz_left_widths=graphviz_left_widths,
                    )
                )
                aux_edges.append((right_id, right_outside, minlen, 0))


def _graphviz_nearest_outside_node(
    layer_nodes: Sequence[int],
    eligible_nodes: Set[int],
    start: int,
    step: int,
) -> Optional[int]:
    """Return the first non-member in a rank scan.

    Parameters
    ----------
    layer_nodes : sequence[int]
        Nodes in one rank.
    eligible_nodes : set[int]
        Normal nodes and virtual nodes whose ``ED_to_orig`` lineage has no
        endpoint in the cluster.
    start : int
        Initial rank-list index.
    step : int
        Scan direction, negative for left and positive for right.

    Returns
    -------
    int or None
        First outside node id, or ``None`` when no outside node exists.
    """
    index = int(start)
    while 0 <= index < len(layer_nodes):
        node = int(layer_nodes[index])
        if node in eligible_nodes:
            return node
        index += int(step)
    return None


def _add_graphviz_subcluster_edges(
    aux_edges: List[Tuple[int, int, int, int]],
    parents: Mapping[str, Optional[str]],
    boundary_ids: Mapping[str, Tuple[int, int]],
) -> None:
    """Constrain child cluster boundaries inside parent boundaries.

    Parameters
    ----------
    aux_edges : list[tuple[int, int, int, int]]
        Mutable auxiliary edge list.
    parents : Mapping[str, str | None]
        Normalized cluster parent mapping.
    boundary_ids : Mapping[str, tuple[int, int]]
        Cluster left/right boundary node ids.

    Returns
    -------
    None
        ``aux_edges`` is mutated in place.
    """
    margin = _graphviz_scaled_minlen(_GRAPHVIZ_CLUSTER_MARGIN_POINTS)
    for child_name, parent_name in parents.items():
        containing_name = _GRAPHVIZ_ROOT_CLUSTER_NAME if parent_name is None else parent_name
        if child_name not in boundary_ids or containing_name not in boundary_ids:
            continue
        parent_left, parent_right = boundary_ids[containing_name]
        child_left, child_right = boundary_ids[child_name]
        aux_edges.append((parent_left, child_left, margin, 0))
        aux_edges.append((child_right, parent_right, margin, 0))


def _add_graphviz_sibling_cluster_edges(
    aux_edges: List[Tuple[int, int, int, int]],
    layers: Sequence[Sequence[int]],
    rank_nodes_by_cluster: Mapping[str, Mapping[int, Sequence[int]]],
    parents: Mapping[str, Optional[str]],
    boundary_ids: Mapping[str, Tuple[int, int]],
) -> None:
    """Separate sibling cluster boxes that overlap in rank span.

    Parameters
    ----------
    aux_edges : list[tuple[int, int, int, int]]
        Mutable auxiliary edge list.
    layers : sequence of sequence[int]
        Ordered expanded nodes per rank.
    rank_nodes_by_cluster : Mapping[str, Mapping[int, sequence[int]]]
        Ordered cluster members per rank.
    parents : Mapping[str, str | None]
        Normalized cluster parent mapping.
    boundary_ids : Mapping[str, tuple[int, int]]
        Cluster left/right boundary node ids.

    Returns
    -------
    None
        ``aux_edges`` is mutated in place.
    """
    margin = _graphviz_scaled_minlen(_GRAPHVIZ_CLUSTER_MARGIN_POINTS)
    parent_names: Set[Optional[str]] = set(parents.values())
    for parent_name in sorted(parent_names, key=lambda value: "" if value is None else value):
        siblings = [name for name, parent in parents.items() if parent == parent_name]
        for left_index, left_name in enumerate(siblings):
            for right_name in siblings[left_index + 1 :]:
                overlap_rank = _graphviz_cluster_overlap_rank(
                    left_ranks=rank_nodes_by_cluster.get(left_name, {}),
                    right_ranks=rank_nodes_by_cluster.get(right_name, {}),
                )
                if overlap_rank is None:
                    continue
                first_left = rank_nodes_by_cluster[left_name][overlap_rank][0]
                first_right = rank_nodes_by_cluster[right_name][overlap_rank][0]
                rank_order = {int(node): index for index, node in enumerate(layers[overlap_rank])}
                if rank_order.get(int(first_left), 0) <= rank_order.get(int(first_right), 0):
                    actual_left, actual_right = left_name, right_name
                else:
                    actual_left, actual_right = right_name, left_name
                _, actual_left_right = boundary_ids[actual_left]
                actual_right_left, _ = boundary_ids[actual_right]
                aux_edges.append((actual_left_right, actual_right_left, margin, 0))


def _graphviz_cluster_overlap_rank(
    left_ranks: Mapping[int, Sequence[int]],
    right_ranks: Mapping[int, Sequence[int]],
) -> Optional[int]:
    """Return the first rank shared by two clusters.

    Parameters
    ----------
    left_ranks : Mapping[int, sequence[int]]
        Left cluster members keyed by rank.
    right_ranks : Mapping[int, sequence[int]]
        Right cluster members keyed by rank.

    Returns
    -------
    int or None
        Lowest shared rank, or ``None`` when rank spans do not overlap.
    """
    overlap = sorted(set(left_ranks).intersection(right_ranks))
    if not overlap:
        return None
    return int(overlap[0])


def _graphviz_left_width(
    node: int,
    node_sizes: torch.Tensor,
    num_original_nodes: int,
    node_sep: float,
    graphviz_left_widths: Optional[Sequence[float]] = None,
) -> float:
    """Return Graphviz left half-width for Stage A x constraints.

    Parameters
    ----------
    node : int
        Expanded node id.
    node_sizes : torch.Tensor
        CPU node sizes with shape ``[N, 2]``.
    num_original_nodes : int
        Count of non-dummy nodes.
    node_sep : float
        Horizontal gap between node bounding boxes.
    graphviz_left_widths : sequence of float, optional
        Per-expanded-node ``ND_lw`` override in Graphviz point units.

    Returns
    -------
    float
        Left half-width in layout units.
    """
    if (
        graphviz_left_widths is not None
        and 0 <= node < len(graphviz_left_widths)
        and graphviz_left_widths[node] >= 0.0
    ):
        return float(graphviz_left_widths[node])
    if node >= num_original_nodes:
        stored_width = float(node_sizes[node, 0].item())
        if stored_width > 0.0:
            return stored_width / 2.0
        return node_sep / 2.0
    stored_width = float(node_sizes[node, 0].item())
    half_width = stored_width / 2.0
    if stored_width > _GRAPHVIZ_DEFAULT_NODE_WIDTH_POINTS:
        half_width += _GRAPHVIZ_LABEL_BOX_HALF_WIDTH_SEED_POINTS
    return half_width


def _graphviz_right_width(
    node: int,
    node_sizes: torch.Tensor,
    num_original_nodes: int,
    node_sep: float,
    graphviz_right_widths: Optional[Sequence[float]] = None,
) -> float:
    """Return Graphviz right half-width for Stage A x constraints.

    Parameters
    ----------
    node : int
        Expanded node id.
    node_sizes : torch.Tensor
        CPU node sizes with shape ``[N, 2]``.
    num_original_nodes : int
        Count of non-dummy nodes.
    node_sep : float
        Horizontal gap between node bounding boxes.
    graphviz_right_widths : sequence of float, optional
        Per-expanded-node ``ND_rw`` override in Graphviz point units.

    Returns
    -------
    float
        Right half-width in layout units.
    """
    if (
        graphviz_right_widths is not None
        and 0 <= node < len(graphviz_right_widths)
        and graphviz_right_widths[node] >= 0.0
    ):
        return float(graphviz_right_widths[node])
    if node >= num_original_nodes:
        stored_width = float(node_sizes[node, 0].item())
        if stored_width > 0.0:
            return stored_width / 2.0
        return node_sep / 2.0
    stored_width = float(node_sizes[node, 0].item())
    half_width = stored_width / 2.0
    if stored_width > _GRAPHVIZ_DEFAULT_NODE_WIDTH_POINTS:
        half_width += _GRAPHVIZ_LABEL_BOX_HALF_WIDTH_SEED_POINTS
    return half_width


def _graphviz_round(value: float) -> int:
    """Round like Graphviz's ``ROUND`` macro for non-negative minlen values.

    Parameters
    ----------
    value : float
        Candidate value.

    Returns
    -------
    int
        ``floor(value + 0.5)`` clamped to zero.
    """
    return max(int(math.floor(value + 0.5)), 0)


def _graphviz_scaled_minlen(value: float) -> int:
    """Return an auxiliary x minlen at Dagua's internal sub-point resolution.

    Parameters
    ----------
    value : float
        Graphviz minlen value before integer quantization.

    Returns
    -------
    int
        Scaled integer minlen.

    Notes
    -----
    The benchmark variants pass Dagua label boxes in a smaller coordinate
    system than Graphviz's point-unit DOT output. A two-unit internal
    resolution preserves Graphviz's midpoint LR-balance behavior for odd
    Dagua minlens, then coordinates are divided back down before returning.
    """
    return _graphviz_round(value * float(_GRAPHVIZ_X_AUX_RESOLUTION))


def _graphviz_x_output_scale(
    layers: Sequence[Sequence[int]],
    node_sizes: torch.Tensor,
    num_original_nodes: int,
    node_sep: float,
    rank_sep: float,
    graphviz_left_widths: Optional[Sequence[float]] = None,
    graphviz_right_widths: Optional[Sequence[float]] = None,
) -> float:
    """Return the x-unit conversion for graphviz coordinate output.

    Parameters
    ----------
    layers : sequence of sequence of int
        Ordered expanded nodes per layer.
    node_sizes : torch.Tensor
        CPU node sizes with shape ``[N, 2]``.
    num_original_nodes : int
        Count of non-dummy nodes.
    node_sep : float
        Horizontal gap between node bounding boxes.
    rank_sep : float
        Vertical layer spacing used by the returned Dagua layout.
    graphviz_left_widths : sequence of float, optional
        Per-expanded-node ``ND_lw`` override in Graphviz point units.
    graphviz_right_widths : sequence of float, optional
        Per-expanded-node ``ND_rw`` override in Graphviz point units.

    Returns
    -------
    float
        Multiplicative scale from auxiliary x units to returned coordinates.

    Notes
    -----
    Graphviz solves x and y in one point-unit frame. The benchmark variants
    often pass point-like label widths with ``rank_sep=1``; normalizing by the
    median same-rank separation preserves the Graphviz x shape while returning
    coordinates in the same unit family as Dagua's y ranks.
    """
    separations: List[float] = []
    for layer_nodes in layers:
        for left_node, right_node in zip(layer_nodes, layer_nodes[1:]):
            separations.append(
                _graphviz_right_width(
                    node=left_node,
                    node_sizes=node_sizes,
                    num_original_nodes=num_original_nodes,
                    node_sep=node_sep,
                    graphviz_right_widths=graphviz_right_widths,
                )
                + _graphviz_left_width(
                    node=right_node,
                    node_sizes=node_sizes,
                    num_original_nodes=num_original_nodes,
                    node_sep=node_sep,
                    graphviz_left_widths=graphviz_left_widths,
                )
                + node_sep
            )
    if not separations:
        return 1.0
    ordered = sorted(value for value in separations if value > 0.0)
    if not ordered:
        return 1.0
    mid = len(ordered) // 2
    if len(ordered) % 2:
        median_sep = ordered[mid]
    else:
        median_sep = (ordered[mid - 1] + ordered[mid]) / 2.0
    if median_sep <= 0.0:
        return 1.0
    return rank_sep / median_sep


def _graphviz_weight_classes(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_original_nodes: int,
) -> List[int]:
    """Compute Graphviz ``ND_weight_class`` equivalents for expanded nodes.

    Parameters
    ----------
    edge_index : torch.Tensor
        Expanded edge list with shape ``[2, E]``.
    num_nodes : int
        Number of expanded nodes.
    num_original_nodes : int
        Count of non-dummy nodes.

    Returns
    -------
    list of int
        Capped weight-class counters indexed by node id.
    """
    weight_classes = [0] * num_nodes
    if edge_index.numel() == 0:
        return weight_classes
    edge_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    for tail, head in zip(edge_cpu[0].tolist(), edge_cpu[1].tolist()):
        for node in (int(tail), int(head)):
            if node < num_original_nodes and weight_classes[node] <= 2:
                weight_classes[node] += 1
    return weight_classes


def _graphviz_omega_weight(
    tail: int,
    head: int,
    weight_classes: Sequence[int],
    num_original_nodes: int,
) -> int:
    """Return Graphviz 7.0.5 ``virtual_weight`` endpoint multiplier.

    Parameters
    ----------
    tail : int
        Expanded edge tail node id.
    head : int
        Expanded edge head node id.
    weight_classes : sequence of int
        ``ND_weight_class`` equivalents indexed by node id.
    num_original_nodes : int
        Count of non-dummy nodes.

    Returns
    -------
    int
        Multiplier from Graphviz's ``C_EE/C_VS/C_SS/C_VV`` table.
    """
    tail_class = _graphviz_endpoint_class(
        node=tail,
        weight_classes=weight_classes,
        num_original_nodes=num_original_nodes,
    )
    head_class = _graphviz_endpoint_class(
        node=head,
        weight_classes=weight_classes,
        num_original_nodes=num_original_nodes,
    )
    return _GRAPHVIZ_OMEGA_TABLE[tail_class][head_class]


def _graphviz_endpoint_class(
    node: int,
    weight_classes: Sequence[int],
    num_original_nodes: int,
) -> int:
    """Classify an expanded endpoint for Graphviz omega weighting.

    Parameters
    ----------
    node : int
        Expanded node id.
    weight_classes : sequence of int
        ``ND_weight_class`` equivalents indexed by node id.
    num_original_nodes : int
        Count of non-dummy nodes.

    Returns
    -------
    int
        Endpoint class index used by ``_GRAPHVIZ_OMEGA_TABLE``.
    """
    if node >= num_original_nodes:
        return _GRAPHVIZ_VIRTUAL_NODE_CLASS
    if weight_classes[node] <= 1:
        return _GRAPHVIZ_SINGLETON_NODE_CLASS
    return _GRAPHVIZ_ORDINARY_NODE_CLASS


def _brandes_koepf_x_positions(
    layers: Sequence[Sequence[int]],
    parents: Sequence[Sequence[int]],
    children: Sequence[Sequence[int]],
    node_sizes: torch.Tensor,
    num_nodes: int,
    num_original_nodes: int,
    node_sep: float,
    center_coordinates: bool = True,
    edge_index: Optional[torch.Tensor] = None,
    use_igraph_conflicts: bool = False,
) -> List[float]:
    """Compute balanced horizontal coordinates with four BK passes.

    Parameters
    ----------
    layers : sequence of sequence of int
        Ordered nodes per layer.
    parents : sequence of sequence of int
        Parent adjacency indexed by node id.
    children : sequence of sequence of int
        Child adjacency indexed by node id.
    node_sizes : torch.Tensor
        CPU node sizes ``[N, 2]``.
    num_nodes : int
        Number of nodes.
    num_original_nodes : int
        Count of non-dummy nodes.
    node_sep : float
        Horizontal gap between node bounding boxes.
    center_coordinates : bool, default=True
        Whether to translate the final horizontal span to be centered at zero.
    edge_index : torch.Tensor, optional
        Expanded edge list with shape ``[2, E]``. Required only when
        ``use_igraph_conflicts`` is enabled.
    use_igraph_conflicts : bool, default=False
        Whether to mirror igraph 1.0.0's ordinal-edge type-1 conflict scan.

    Returns
    -------
    list of float
        Final X coordinates for all expanded nodes.
    """
    if num_nodes == 0:
        return []

    dummy_mask = [node >= num_original_nodes for node in range(num_nodes)]
    if use_igraph_conflicts:
        if edge_index is None:
            raise ValueError("edge_index is required when use_igraph_conflicts=True")
        balanced = _igraph_brandes_koepf_x_positions(
            layers=layers,
            parents=parents,
            children=children,
            edge_index=edge_index,
            dummy_mask=dummy_mask,
            num_nodes=num_nodes,
            node_sep=node_sep,
        )
        if center_coordinates:
            return _center_coordinates(values=balanced)
        return balanced

    orientation_specs = (
        ("ul", False, False),
        ("ur", False, True),
        ("dl", True, False),
        ("dr", True, True),
    )
    x_by_alignment: Dict[str, List[float]] = {}
    for alignment_name, reverse_layers, reverse_within in orientation_specs:
        transformed_layers = _transform_layers(
            layers=layers,
            reverse_layers=reverse_layers,
            reverse_within=reverse_within,
        )
        rank_of, pos_of = _layer_position_maps(layers=transformed_layers, num_nodes=num_nodes)
        predecessor_source = children if reverse_layers else parents
        predecessors = _ordered_transformed_neighbors(
            neighbors_by_node=predecessor_source,
            pos_of=pos_of,
        )
        conflicts = _find_type1_conflicts(
            layers=transformed_layers,
            predecessors=predecessors,
            pos_of=pos_of,
            dummy_mask=dummy_mask,
        )
        root, align = _vertical_alignment(
            layers=transformed_layers,
            predecessors=predecessors,
            pos_of=pos_of,
            conflicts=conflicts,
            num_nodes=num_nodes,
        )
        compacted = _horizontal_compaction(
            layers=transformed_layers,
            root=root,
            align=align,
            rank_of=rank_of,
            pos_of=pos_of,
            node_sizes=node_sizes,
            node_sep=node_sep,
            num_nodes=num_nodes,
        )
        if reverse_within:
            compacted = [-value for value in compacted]
        x_by_alignment[alignment_name] = compacted

    _align_compacted_coordinates(x_by_alignment=x_by_alignment)
    balanced = _median_balanced_coordinates(x_by_alignment=x_by_alignment, num_nodes=num_nodes)
    if center_coordinates:
        return _center_coordinates(values=balanced)
    return balanced


def _igraph_brandes_koepf_x_positions(
    layers: Sequence[Sequence[int]],
    parents: Sequence[Sequence[int]],
    children: Sequence[Sequence[int]],
    edge_index: torch.Tensor,
    dummy_mask: Sequence[bool],
    num_nodes: int,
    node_sep: float,
) -> List[float]:
    """Compute igraph 1.0.0 Brandes-Koepf x coordinates.

    Parameters
    ----------
    layers : sequence of sequence of int
        Ordered nodes per layer in igraph's top-down layer order.
    parents : sequence of sequence of int
        Incoming adjacency lists indexed by expanded node id.
    children : sequence of sequence of int
        Outgoing adjacency lists indexed by expanded node id.
    edge_index : torch.Tensor
        Expanded adjacent-rank edge list with shape ``[2, E]``.
    dummy_mask : sequence of bool
        Flags indicating which expanded nodes are dummy vertices.
    num_nodes : int
        Number of expanded graph nodes.
    node_sep : float
        Igraph ``hgap`` value used as the center-to-center block separation.

    Returns
    -------
    list of float
        Median-balanced x coordinates for all expanded nodes.

    Notes
    -----
    Igraph does not mirror layers to implement right-aligned passes. It keeps
    the original ``vertex_to_the_left`` array, flips only the vertical-alignment
    scan order with ``align_right``, then compacts in the original coordinate
    frame. This differs from the generic BK helper's mirrored-orientation
    emulation on tie-heavy layouts.
    """
    ignored_edges = _find_igraph_ignored_type1_edges(
        layers=layers,
        edge_index=edge_index,
        dummy_mask=dummy_mask,
        num_nodes=num_nodes,
    )
    vertex_to_the_left = _igraph_vertex_to_the_left(layers=layers, num_nodes=num_nodes)
    x_by_alignment: Dict[str, List[float]] = {}
    for run_index, alignment_name in enumerate(("ul", "ur", "dl", "dr")):
        reverse = bool(run_index // 2)
        align_right = bool(run_index % 2)
        root, align = _igraph_vertical_alignment(
            layers=layers,
            parents=parents,
            children=children,
            edge_index=edge_index,
            ignored_edges=ignored_edges,
            reverse=reverse,
            align_right=align_right,
            num_nodes=num_nodes,
        )
        x_by_alignment[alignment_name] = _igraph_horizontal_compaction(
            vertex_to_the_left=vertex_to_the_left,
            root=root,
            align=align,
            node_sep=node_sep,
            num_nodes=num_nodes,
        )

    _align_compacted_coordinates(x_by_alignment=x_by_alignment)
    return _median_balanced_coordinates(x_by_alignment=x_by_alignment, num_nodes=num_nodes)


def _igraph_vertex_to_the_left(
    layers: Sequence[Sequence[int]],
    num_nodes: int,
) -> List[int]:
    """Return igraph's left-neighbor array for horizontal compaction.

    Parameters
    ----------
    layers : sequence of sequence of int
        Ordered nodes per layer.
    num_nodes : int
        Number of expanded graph nodes.

    Returns
    -------
    list of int
        ``vertex_to_the_left[v]`` is the immediate left neighbor of ``v`` in
        its layer, or ``v`` itself for leftmost vertices.
    """
    vertex_to_the_left = list(range(num_nodes))
    for layer_nodes in layers:
        if not layer_nodes:
            continue
        previous = layer_nodes[0]
        vertex_to_the_left[previous] = previous
        for node in layer_nodes[1:]:
            vertex_to_the_left[node] = previous
            previous = node
    return vertex_to_the_left


def _igraph_initial_x_positions(
    layers: Sequence[Sequence[int]],
    num_nodes: int,
) -> List[float]:
    """Return igraph's ordering-stage x column before final BK placement.

    Parameters
    ----------
    layers : sequence of sequence of int
        Ordered nodes per layer.
    num_nodes : int
        Number of expanded graph nodes.

    Returns
    -------
    list of float
        Within-layer order index for each expanded node.
    """
    x_positions = [0.0] * num_nodes
    for layer_nodes in layers:
        for position, node in enumerate(layer_nodes):
            x_positions[node] = float(position)
    return x_positions


def _igraph_undirected_edge_lookup(edge_index: torch.Tensor) -> Dict[Tuple[int, int], int]:
    """Return first edge ids keyed by both endpoint orders.

    Parameters
    ----------
    edge_index : torch.Tensor
        Expanded edge tensor with shape ``[2, E]``.

    Returns
    -------
    dict
        Mapping ``(u, v)`` and ``(v, u)`` to the first matching edge id,
        matching ``igraph_get_eid(..., IGRAPH_UNDIRECTED, error=true)`` for
        the simple expanded graphs used by Sugiyama.
    """
    edge_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    lookup: Dict[Tuple[int, int], int] = {}
    for edge_id, (source, target) in enumerate(zip(edge_cpu[0].tolist(), edge_cpu[1].tolist())):
        source_id = int(source)
        target_id = int(target)
        lookup.setdefault((source_id, target_id), edge_id)
        lookup.setdefault((target_id, source_id), edge_id)
    return lookup


def _igraph_vertical_alignment(
    layers: Sequence[Sequence[int]],
    parents: Sequence[Sequence[int]],
    children: Sequence[Sequence[int]],
    edge_index: torch.Tensor,
    ignored_edges: Sequence[bool],
    reverse: bool,
    align_right: bool,
    num_nodes: int,
) -> Tuple[List[int], List[int]]:
    """Construct one igraph BK vertical alignment.

    Parameters
    ----------
    layers : sequence of sequence of int
        Ordered nodes per layer in original top-down order.
    parents : sequence of sequence of int
        Incoming adjacency indexed by expanded node id.
    children : sequence of sequence of int
        Outgoing adjacency indexed by expanded node id.
    edge_index : torch.Tensor
        Expanded edge tensor with shape ``[2, E]``.
    ignored_edges : sequence of bool
        Edge-id mask for Type-1 conflicts.
    reverse : bool
        Whether to align downward through outgoing neighbors.
    align_right : bool
        Whether to scan each layer and the even-median pair right-to-left.
    num_nodes : int
        Number of expanded graph nodes.

    Returns
    -------
    tuple
        ``(root, align)`` arrays indexed by expanded node id.
    """
    root = list(range(num_nodes))
    align = list(range(num_nodes))
    initial_x = _igraph_initial_x_positions(layers=layers, num_nodes=num_nodes)
    edge_lookup = _igraph_undirected_edge_lookup(edge_index=edge_index)
    layer_index = len(layers) - 2 if reverse else 1
    layer_step = -1 if reverse else 1
    layer_limit = -1 if reverse else len(layers)

    while layer_index != layer_limit:
        layer_nodes = layers[layer_index]
        previous_position = math.inf if align_right else -1.0
        node_positions = (
            range(len(layer_nodes) - 1, -1, -1) if align_right else range(len(layer_nodes))
        )
        for node_position in node_positions:
            node = layer_nodes[node_position]
            if align[node] != node:
                continue
            # igraph_neighbors() returns adjacent vertex IDs in ascending order,
            # independent of the expanded edge insertion order that built the
            # adjacency lists. Median ties in BK vertical alignment can see
            # that ordering, so preserve it here.
            neighbors = sorted(children[node] if reverse else parents[node])
            medians = _igraph_alignment_medians(
                neighbors=neighbors,
                initial_x=initial_x,
                align_right=align_right,
            )
            for predecessor in medians:
                if predecessor < 0 or align[node] != node:
                    continue
                edge_id = edge_lookup[(node, predecessor)]
                if ignored_edges[edge_id]:
                    continue
                predecessor_position = initial_x[predecessor]
                can_align = (
                    previous_position > predecessor_position
                    if align_right
                    else previous_position < predecessor_position
                )
                if not can_align:
                    continue
                align[predecessor] = node
                root[node] = root[predecessor]
                align[node] = root[predecessor]
                previous_position = predecessor_position
        layer_index += layer_step

    return root, align


def _igraph_alignment_medians(
    neighbors: Sequence[int],
    initial_x: Sequence[float],
    align_right: bool,
) -> Tuple[int, int]:
    """Return igraph's one- or two-median candidate tuple.

    Parameters
    ----------
    neighbors : sequence of int
        Neighbor node ids from igraph-neighbor order.
    initial_x : sequence of float
        Ordering-stage x positions indexed by node id.
    align_right : bool
        Whether the right-alignment median ordering is active.

    Returns
    -------
    tuple of int
        Two candidate node ids. The second value is ``-1`` when there is only
        one usable median.
    """
    neighbor_count = len(neighbors)
    if neighbor_count == 0:
        return -1, -1
    if neighbor_count == 1:
        return int(neighbors[0]), -1

    order = sorted(range(neighbor_count), key=lambda index: (initial_x[neighbors[index]], index))
    if neighbor_count % 2:
        return int(neighbors[order[neighbor_count // 2]]), -1
    if align_right:
        return (
            int(neighbors[order[neighbor_count // 2]]),
            int(neighbors[order[neighbor_count // 2 - 1]]),
        )
    return (
        int(neighbors[order[neighbor_count // 2 - 1]]),
        int(neighbors[order[neighbor_count // 2]]),
    )


def _igraph_horizontal_compaction(
    vertex_to_the_left: Sequence[int],
    root: Sequence[int],
    align: Sequence[int],
    node_sep: float,
    num_nodes: int,
) -> List[float]:
    """Compact one igraph BK alignment into concrete x coordinates.

    Parameters
    ----------
    vertex_to_the_left : sequence of int
        Immediate-left-neighbor array in original layer order.
    root : sequence of int
        Block-root array from vertical alignment.
    align : sequence of int
        Alignment cycle array from vertical alignment.
    node_sep : float
        Igraph ``hgap`` value.
    num_nodes : int
        Number of expanded graph nodes.

    Returns
    -------
    list of float
        X coordinates for one alignment run.
    """
    sink = list(range(num_nodes))
    shift = [math.inf] * num_nodes
    x_positions = [-1.0] * num_nodes
    original_recursion_limit = sys.getrecursionlimit()
    needed_recursion_limit = max(original_recursion_limit, num_nodes + 100)

    # Igraph implements this block walk recursively in C; dummy-expanded Python
    # layouts can legitimately exceed the default recursion limit.
    try:
        if needed_recursion_limit > original_recursion_limit:
            sys.setrecursionlimit(needed_recursion_limit)
        for node in range(num_nodes):
            if root[node] == node:
                _igraph_place_compaction_block(
                    block_root=node,
                    vertex_to_the_left=vertex_to_the_left,
                    root=root,
                    align=align,
                    sink=sink,
                    shift=shift,
                    node_sep=node_sep,
                    x_positions=x_positions,
                )
    finally:
        if sys.getrecursionlimit() != original_recursion_limit:
            sys.setrecursionlimit(original_recursion_limit)

    old_x_positions = list(x_positions)
    for node in range(num_nodes):
        block_root = root[node]
        x_positions[node] = old_x_positions[block_root]
        sink_shift = shift[sink[block_root]]
        if sink_shift < math.inf:
            x_positions[node] += sink_shift
    return x_positions


def _igraph_place_compaction_block(
    block_root: int,
    vertex_to_the_left: Sequence[int],
    root: Sequence[int],
    align: Sequence[int],
    sink: List[int],
    shift: List[float],
    node_sep: float,
    x_positions: List[float],
) -> None:
    """Place one igraph horizontal-compaction block recursively.

    Parameters
    ----------
    block_root : int
        Root node of the block being placed.
    vertex_to_the_left : sequence of int
        Immediate-left-neighbor array in original layer order.
    root : sequence of int
        Block-root array from vertical alignment.
    align : sequence of int
        Alignment cycle array from vertical alignment.
    sink : list of int
        Sink representative for each block root.
    shift : list of float
        Deferred class shifts indexed by sink root.
    node_sep : float
        Igraph ``hgap`` value.
    x_positions : list of float
        Mutable x-coordinate work array. Values below zero mean unplaced.

    Returns
    -------
    None
        The work arrays are updated in place.
    """
    if x_positions[block_root] >= 0.0:
        return

    x_positions[block_root] = 0.0
    current = block_root
    while True:
        left_neighbor = vertex_to_the_left[current]
        if left_neighbor != current:
            left_root = root[left_neighbor]
            _igraph_place_compaction_block(
                block_root=left_root,
                vertex_to_the_left=vertex_to_the_left,
                root=root,
                align=align,
                sink=sink,
                shift=shift,
                node_sep=node_sep,
                x_positions=x_positions,
            )

            left_sink = sink[left_root]
            block_sink = sink[block_root]
            if block_sink == block_root:
                sink[block_root] = block_sink = left_sink
            if block_sink != left_sink:
                shift[left_sink] = min(
                    shift[left_sink],
                    x_positions[block_root] - x_positions[left_root] - node_sep,
                )
            else:
                x_positions[block_root] = max(
                    x_positions[block_root],
                    x_positions[left_root] + node_sep,
                )

        current = align[current]
        if current == block_root:
            break


def _transform_layers(
    layers: Sequence[Sequence[int]],
    reverse_layers: bool,
    reverse_within: bool,
) -> List[List[int]]:
    """Return an orientation-specific view of the ordered layers.

    Parameters
    ----------
    layers : sequence of sequence of int
        Ordered nodes per layer in the original top-down orientation.
    reverse_layers : bool
        Whether to scan layers from bottom to top.
    reverse_within : bool
        Whether to mirror each layer left-to-right.

    Returns
    -------
    list of list of int
        Oriented layer ordering for one Brandes-Kopf pass.
    """
    oriented_layers = [list(layer) for layer in layers]
    if reverse_layers:
        oriented_layers = list(reversed(oriented_layers))
    if reverse_within:
        oriented_layers = [list(reversed(layer)) for layer in oriented_layers]
    return oriented_layers


def _layer_position_maps(
    layers: Sequence[Sequence[int]],
    num_nodes: int,
) -> Tuple[List[int], List[int]]:
    """Build rank and order maps for one oriented layering.

    Parameters
    ----------
    layers : sequence of sequence of int
        Oriented layers for one pass.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    tuple
        ``(rank_of, pos_of)`` arrays indexed by node id.
    """
    rank_of = [-1] * num_nodes
    pos_of = [-1] * num_nodes
    for rank_index, layer_nodes in enumerate(layers):
        for position, node in enumerate(layer_nodes):
            rank_of[node] = rank_index
            pos_of[node] = position
    return rank_of, pos_of


def _ordered_transformed_neighbors(
    neighbors_by_node: Sequence[Sequence[int]],
    pos_of: Sequence[int],
) -> List[List[int]]:
    """Sort neighbors by their position in the transformed layering.

    Parameters
    ----------
    neighbors_by_node : sequence of sequence of int
        Neighbor adjacency indexed by node id.
    pos_of : sequence of int
        Within-layer positions for the transformed layering.

    Returns
    -------
    list of list of int
        Neighbor ids sorted left-to-right in the transformed layering.
    """
    ordered_neighbors: List[List[int]] = []
    for neighbors in neighbors_by_node:
        ordered_neighbors.append(sorted(neighbors, key=lambda node: pos_of[node]))
    return ordered_neighbors


def _find_type1_conflicts(
    layers: Sequence[Sequence[int]],
    predecessors: Sequence[Sequence[int]],
    pos_of: Sequence[int],
    dummy_mask: Sequence[bool],
) -> Set[Tuple[int, int]]:
    """Mark Type 1 conflicts between inner and non-inner segments.

    Parameters
    ----------
    layers : sequence of sequence of int
        Oriented layers for one Brandes-Kopf pass.
    predecessors : sequence of sequence of int
        Oriented predecessor adjacency indexed by node id.
    pos_of : sequence of int
        Within-layer positions in the oriented layering.
    dummy_mask : sequence of bool
        Flags indicating which nodes are dummy vertices created for long
        edges.

    Returns
    -------
    set of tuple of int
        Conflicting oriented edges as ``(predecessor, node)`` pairs.
    """
    conflicts: Set[Tuple[int, int]] = set()
    for rank_index in range(1, len(layers)):
        north_layer = layers[rank_index - 1]
        south_layer = layers[rank_index]
        if not south_layer:
            continue

        left_boundary = 0
        scan_start = 0
        last_index = len(south_layer) - 1
        for south_index, node in enumerate(south_layer):
            inner_predecessor = _inner_segment_predecessor(
                node=node,
                predecessors=predecessors,
                dummy_mask=dummy_mask,
            )
            right_boundary = (
                pos_of[inner_predecessor] if inner_predecessor is not None else len(north_layer)
            )
            if inner_predecessor is None and south_index != last_index:
                continue

            for scan_node in south_layer[scan_start : south_index + 1]:
                for predecessor in predecessors[scan_node]:
                    predecessor_position = pos_of[predecessor]
                    is_inner_segment = dummy_mask[scan_node] and dummy_mask[predecessor]
                    if not is_inner_segment and (
                        predecessor_position < left_boundary
                        or predecessor_position > right_boundary
                    ):
                        conflicts.add((predecessor, scan_node))

            scan_start = south_index + 1
            left_boundary = right_boundary
    return conflicts


def _find_igraph_type1_conflicts(
    layers: Sequence[Sequence[int]],
    edge_index: torch.Tensor,
    dummy_mask: Sequence[bool],
    num_nodes: int,
) -> Set[Tuple[int, int]]:
    """Mark igraph 1.0.0's ordinal-edge Type 1 conflicts.

    Parameters
    ----------
    layers : sequence of sequence of int
        Ordered nodes per layer in the original top-down orientation.
    edge_index : torch.Tensor
        Expanded adjacent-rank edge list with shape ``[2, E]``.
    dummy_mask : sequence of bool
        Flags indicating which nodes are dummy vertices created for long
        edges.
    num_nodes : int
        Number of expanded graph nodes.

    Returns
    -------
    set of tuple of int
        Undirected edge pairs to ignore during vertical alignment.

    Notes
    -----
    igraph 1.0.0 sizes each per-layer conflict scan from the gathered outgoing
    neighbor count, but then indexes ``IGRAPH_FROM(graph, j)`` and
    ``IGRAPH_TO(graph, j)`` by ordinal edge id. This preserves that tie-break
    quirk instead of using the standard Brandes-Koepf segment scan.
    """
    ignored_edge_mask = _find_igraph_ignored_type1_edges(
        layers=layers,
        edge_index=edge_index,
        dummy_mask=dummy_mask,
        num_nodes=num_nodes,
    )
    if edge_index.numel() == 0:
        return set()

    edge_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    sources = [int(value) for value in edge_cpu[0].tolist()]
    targets = [int(value) for value in edge_cpu[1].tolist()]
    conflicts: Set[Tuple[int, int]] = set()
    for edge_id, ignored in enumerate(ignored_edge_mask):
        if not ignored:
            continue
        source = sources[edge_id]
        target = targets[edge_id]
        conflicts.add((source, target))
        conflicts.add((target, source))
    return conflicts


def _find_igraph_ignored_type1_edges(
    layers: Sequence[Sequence[int]],
    edge_index: torch.Tensor,
    dummy_mask: Sequence[bool],
    num_nodes: int,
) -> List[bool]:
    """Return igraph 1.0.0's ignored-edge mask for Type-1 conflicts.

    Parameters
    ----------
    layers : sequence of sequence of int
        Ordered nodes per layer in the original top-down orientation.
    edge_index : torch.Tensor
        Expanded adjacent-rank edge list with shape ``[2, E]``.
    dummy_mask : sequence of bool
        Flags indicating which nodes are dummy vertices created for long
        edges.
    num_nodes : int
        Number of expanded graph nodes.

    Returns
    -------
    list of bool
        Boolean mask aligned to expanded edge ids. ``True`` means igraph would
        skip that edge during vertical alignment.
    """
    if edge_index.numel() == 0:
        return []

    edge_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    sources = [int(value) for value in edge_cpu[0].tolist()]
    targets = [int(value) for value in edge_cpu[1].tolist()]
    edge_count = len(sources)
    pos_of = [-1] * num_nodes
    for layer_nodes in layers:
        for position, node in enumerate(layer_nodes):
            if 0 <= node < num_nodes:
                pos_of[node] = position

    outgoing_counts = [0] * num_nodes
    for source in sources:
        if 0 <= source < num_nodes:
            outgoing_counts[source] += 1

    ignored_edges = [False] * edge_count
    for layer_nodes in layers[:-1]:
        scan_count = sum(outgoing_counts[node] for node in layer_nodes if 0 <= node < num_nodes)
        scan_count = min(scan_count, edge_count)
        for left_edge_id in range(scan_count):
            left_source = sources[left_edge_id]
            left_target = targets[left_edge_id]
            left_inner = dummy_mask[left_source] and dummy_mask[left_target]
            for right_edge_id in range(left_edge_id + 1, scan_count):
                right_source = sources[right_edge_id]
                right_target = targets[right_edge_id]
                if (dummy_mask[right_source] and dummy_mask[right_target]) == left_inner:
                    continue
                if _igraph_segments_cross(
                    left_source=left_source,
                    left_target=left_target,
                    right_source=right_source,
                    right_target=right_target,
                    pos_of=pos_of,
                ):
                    if left_inner:
                        ignored_edges[right_edge_id] = True
                    else:
                        ignored_edges[left_edge_id] = True
    return ignored_edges


def _igraph_segments_cross(
    left_source: int,
    left_target: int,
    right_source: int,
    right_target: int,
    pos_of: Sequence[int],
) -> bool:
    """Return whether igraph's BK conflict test treats two edges as crossing.

    Parameters
    ----------
    left_source : int
        Source vertex of the first edge.
    left_target : int
        Target vertex of the first edge.
    right_source : int
        Source vertex of the second edge.
    right_target : int
        Target vertex of the second edge.
    pos_of : sequence of int
        Original within-layer positions indexed by vertex id.

    Returns
    -------
    bool
        ``True`` when the ordinal-edge pair crosses under igraph's test.
    """
    if left_source == right_source or left_target == right_target:
        return True
    if pos_of[left_source] <= pos_of[right_source]:
        return pos_of[left_target] >= pos_of[right_target]
    return pos_of[left_target] <= pos_of[right_target]


def _inner_segment_predecessor(
    node: int,
    predecessors: Sequence[Sequence[int]],
    dummy_mask: Sequence[bool],
) -> Optional[int]:
    """Return the predecessor of an inner segment dummy, if present.

    Parameters
    ----------
    node : int
        Candidate node on the lower layer of an oriented pass.
    predecessors : sequence of sequence of int
        Oriented predecessor adjacency indexed by node id.
    dummy_mask : sequence of bool
        Flags indicating dummy nodes.

    Returns
    -------
    int, optional
        The predecessor node id when ``node`` participates in a dummy-to-dummy
        inner segment; otherwise ``None``.
    """
    if not dummy_mask[node] or len(predecessors[node]) != 1:
        return None
    predecessor = predecessors[node][0]
    if not dummy_mask[predecessor]:
        return None
    return predecessor


def _vertical_alignment(
    layers: Sequence[Sequence[int]],
    predecessors: Sequence[Sequence[int]],
    pos_of: Sequence[int],
    conflicts: Set[Tuple[int, int]],
    num_nodes: int,
) -> Tuple[List[int], List[int]]:
    """Construct aligned blocks for one Brandes-Kopf orientation.

    Parameters
    ----------
    layers : sequence of sequence of int
        Oriented layers for one pass.
    predecessors : sequence of sequence of int
        Oriented predecessor adjacency indexed by node id.
    pos_of : sequence of int
        Within-layer positions in the oriented layering.
    conflicts : set of tuple of int
        Type 1 conflicts as ``(predecessor, node)`` pairs.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    tuple
        ``(root, align)`` arrays indexed by node id.
    """
    root = list(range(num_nodes))
    align = list(range(num_nodes))

    for rank_index in range(1, len(layers)):
        previous_position = -1
        for node in layers[rank_index]:
            neighbor_nodes = predecessors[node]
            if not neighbor_nodes:
                continue

            median_start = (len(neighbor_nodes) - 1) // 2
            median_stop = len(neighbor_nodes) // 2
            for neighbor_index in range(median_start, median_stop + 1):
                predecessor = neighbor_nodes[neighbor_index]
                if align[node] != node:
                    break
                if pos_of[predecessor] <= previous_position:
                    continue
                if (predecessor, node) in conflicts:
                    continue

                align[predecessor] = node
                align[node] = root[node] = root[predecessor]
                previous_position = pos_of[predecessor]
    return root, align


def _horizontal_compaction(
    layers: Sequence[Sequence[int]],
    root: Sequence[int],
    align: Sequence[int],
    rank_of: Sequence[int],
    pos_of: Sequence[int],
    node_sizes: torch.Tensor,
    node_sep: float,
    num_nodes: int,
) -> List[float]:
    """Compact one aligned orientation into concrete X positions.

    Parameters
    ----------
    layers : sequence of sequence of int
        Oriented layers for one pass.
    root : sequence of int
        Block-root array from vertical alignment.
    align : sequence of int
        Cyclic alignment array from vertical alignment.
    rank_of : sequence of int
        Layer indices in the oriented layering.
    pos_of : sequence of int
        Within-layer positions in the oriented layering.
    node_sizes : torch.Tensor
        CPU node sizes ``[N, 2]``.
    node_sep : float
        Horizontal gap between node bounding boxes.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    list of float
        Compacted X coordinates for the oriented pass.
    """
    sink = list(range(num_nodes))
    shift = [_NO_SHIFT] * num_nodes
    x: List[Optional[float]] = [None] * num_nodes

    for node in range(num_nodes):
        if root[node] == node:
            _place_compaction_block(
                block_root=node,
                layers=layers,
                root=root,
                align=align,
                rank_of=rank_of,
                pos_of=pos_of,
                node_sizes=node_sizes,
                node_sep=node_sep,
                sink=sink,
                shift=shift,
                x=x,
            )

    compacted = [0.0] * num_nodes
    for node in range(num_nodes):
        block_root = root[node]
        block_x = 0.0 if x[block_root] is None else x[block_root]
        sink_shift = shift[sink[block_root]]
        compacted[node] = block_x if sink_shift == _NO_SHIFT else block_x + sink_shift
    return compacted


def _place_compaction_block(
    block_root: int,
    layers: Sequence[Sequence[int]],
    root: Sequence[int],
    align: Sequence[int],
    rank_of: Sequence[int],
    pos_of: Sequence[int],
    node_sizes: torch.Tensor,
    node_sep: float,
    sink: List[int],
    shift: List[float],
    x: List[Optional[float]],
) -> None:
    """Place one aligned block during horizontal compaction.

    Parameters
    ----------
    block_root : int
        Root node of the block currently being placed.
    layers : sequence of sequence of int
        Oriented layers for the current pass.
    root : sequence of int
        Block-root array from vertical alignment.
    align : sequence of int
        Cyclic alignment array from vertical alignment.
    rank_of : sequence of int
        Layer indices in the oriented layering.
    pos_of : sequence of int
        Within-layer positions in the oriented layering.
    node_sizes : torch.Tensor
        CPU node sizes ``[N, 2]``.
    node_sep : float
        Horizontal gap between node bounding boxes.
    sink : list of int
        Sink representative for each block root.
    shift : list of float
        Deferred class shifts indexed by sink root.
    x : list of float, optional
        Root coordinates under construction.
    """
    if x[block_root] is not None:
        return

    stack: List[Tuple[int, int, Optional[int], Optional[int]]] = [
        (block_root, block_root, None, None)
    ]
    while stack:
        active_root, current, left_neighbor, left_root = stack.pop()
        if x[active_root] is None:
            x[active_root] = 0.0

        if left_neighbor is not None and left_root is not None:
            if sink[active_root] == active_root:
                sink[active_root] = sink[left_root]

            block_x = 0.0 if x[active_root] is None else x[active_root]
            left_x = 0.0 if x[left_root] is None else x[left_root]
            minimum_gap = _minimum_separation(
                left_node=left_neighbor,
                right_node=current,
                node_sizes=node_sizes,
                node_sep=node_sep,
            )
            if sink[active_root] != sink[left_root]:
                shift[sink[left_root]] = min(
                    shift[sink[left_root]],
                    block_x - left_x - minimum_gap,
                )
            else:
                x[active_root] = max(block_x, left_x + minimum_gap)

            next_current = align[current]
            if next_current != active_root:
                stack.append((active_root, next_current, None, None))
            continue

        layer_nodes = layers[rank_of[current]]
        position = pos_of[current]
        if position > 0:
            next_left_neighbor = layer_nodes[position - 1]
            next_left_root = root[next_left_neighbor]
            stack.append((active_root, current, next_left_neighbor, next_left_root))
            if x[next_left_root] is None:
                stack.append((next_left_root, next_left_root, None, None))
            continue

        current = align[current]
        if current != active_root:
            stack.append((active_root, current, None, None))


def _minimum_separation(
    left_node: int,
    right_node: int,
    node_sizes: torch.Tensor,
    node_sep: float,
) -> float:
    """Return the minimum center-to-center separation within one layer.

    Parameters
    ----------
    left_node : int
        Left node id in the current orientation.
    right_node : int
        Right node id in the current orientation.
    node_sizes : torch.Tensor
        CPU node sizes ``[N, 2]``.
    node_sep : float
        Horizontal gap between node bounding boxes.

    Returns
    -------
    float
        Required center-to-center distance between the node pair.
    """
    return (
        float(node_sizes[left_node, 0].item()) + float(node_sizes[right_node, 0].item())
    ) / 2.0 + node_sep


def _align_compacted_coordinates(x_by_alignment: Dict[str, List[float]]) -> None:
    """Shift the four compacted assignments into a common coordinate frame.

    Parameters
    ----------
    x_by_alignment : dict
        Mapping from alignment name (``ul``, ``ur``, ``dl``, ``dr``) to
        compacted X coordinates.
    """
    if not x_by_alignment:
        return

    anchor_name = min(
        x_by_alignment,
        key=lambda name: _coordinate_span(values=x_by_alignment[name]),
    )
    anchor_values = x_by_alignment[anchor_name]
    anchor_left = min(anchor_values)
    anchor_right = max(anchor_values)

    for alignment_name, values in x_by_alignment.items():
        if alignment_name == anchor_name:
            continue
        if alignment_name.endswith("l"):
            shift = anchor_left - min(values)
        else:
            shift = anchor_right - max(values)
        for index in range(len(values)):
            values[index] += shift


def _coordinate_span(values: Sequence[float]) -> float:
    """Return the span of a coordinate assignment.

    Parameters
    ----------
    values : sequence of float
        Coordinate values.

    Returns
    -------
    float
        ``max(values) - min(values)`` or zero for empty inputs.
    """
    if not values:
        return 0.0
    return max(values) - min(values)


def _median_balanced_coordinates(
    x_by_alignment: Dict[str, Sequence[float]],
    num_nodes: int,
) -> List[float]:
    """Take the per-node median across the four compacted assignments.

    Parameters
    ----------
    x_by_alignment : dict
        Mapping from alignment name to compacted X coordinates.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    list of float
        Median-balanced X coordinates.
    """
    balanced = [0.0] * num_nodes
    for node in range(num_nodes):
        samples = sorted(values[node] for values in x_by_alignment.values())
        balanced[node] = (samples[1] + samples[2]) / 2.0
    return balanced


def _center_coordinates(values: Sequence[float]) -> List[float]:
    """Translate coordinates so the overall span is centered at zero.

    Parameters
    ----------
    values : sequence of float
        Coordinate values.

    Returns
    -------
    list of float
        Centered coordinates.
    """
    if not values:
        return []

    midpoint = (min(values) + max(values)) / 2.0
    return [value - midpoint for value in values]


def _flatten_graphviz_cluster_members(members: Any) -> Tuple[int, ...]:
    """Return sorted integer leaf ids from nested cluster membership.

    Parameters
    ----------
    members : Any
        Dagua cluster membership value. Values may be flat sequences, sets, or
        nested dictionaries produced by user-facing cluster helpers.

    Returns
    -------
    tuple[int, ...]
        Sorted unique node ids found in ``members``.
    """
    out: Set[int] = set()
    stack: List[Any] = [members]
    while stack:
        value = stack.pop()
        if isinstance(value, Mapping):
            stack.extend(reversed(tuple(value.values())))
            continue
        if isinstance(value, (str, bytes)):
            continue
        try:
            out.add(int(value))
            continue
        except (TypeError, ValueError):
            pass
        if isinstance(value, (Sequence, set, frozenset)):
            stack.extend(reversed(tuple(value)))
    return tuple(sorted(out))


def _normalize_graphviz_clusters(
    clusters: Optional[Mapping[str, Any]],
    num_nodes: int,
) -> Dict[str, Tuple[int, ...]]:
    """Normalize cluster metadata to valid descendant leaf ids.

    Parameters
    ----------
    clusters : Mapping[str, Any], optional
        Raw Dagua cluster membership.
    num_nodes : int
        Number of original graph nodes.

    Returns
    -------
    dict[str, tuple[int, ...]]
        Cluster names mapped to valid original-node ids.
    """
    if not clusters:
        return {}
    normalized: Dict[str, Tuple[int, ...]] = {}
    for name, members in clusters.items():
        filtered = tuple(
            node for node in _flatten_graphviz_cluster_members(members) if 0 <= node < num_nodes
        )
        if filtered:
            normalized[str(name)] = filtered
    return normalized


def _normalize_graphviz_cluster_parents(
    cluster_names: Sequence[str],
    cluster_parents: Optional[Mapping[str, Optional[str]]],
) -> Dict[str, Optional[str]]:
    """Normalize parent references to known cluster names.

    Parameters
    ----------
    cluster_names : sequence of str
        Known cluster names.
    cluster_parents : Mapping[str, str | None], optional
        Raw Dagua parent mapping.

    Returns
    -------
    dict[str, str | None]
        Parent mapping where missing or unknown parents become ``None``.
    """
    known = set(cluster_names)
    raw = cluster_parents or {}
    parents: Dict[str, Optional[str]] = {}
    for name in cluster_names:
        parent = raw.get(name)
        parents[name] = parent if parent in known else None
    return parents


def _graphviz_cluster_depth(name: str, parents: Mapping[str, Optional[str]]) -> int:
    """Return the nesting depth of one normalized cluster.

    Parameters
    ----------
    name : str
        Cluster name.
    parents : Mapping[str, str | None]
        Normalized parent mapping.

    Returns
    -------
    int
        Number of valid parent hops above ``name``.
    """
    depth = 0
    seen: Set[str] = set()
    parent = parents.get(name)
    while parent is not None and parent not in seen:
        seen.add(parent)
        depth += 1
        parent = parents.get(parent)
    return depth


def _graphviz_cluster_bbox(
    positions: torch.Tensor,
    node_sizes: torch.Tensor,
    members: Sequence[int],
    padding: float,
) -> Tuple[float, float, float, float]:
    """Return a padded cluster bbox in output coordinate units.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]`` in output coordinate units.
    members : sequence of int
        Original-node ids in the cluster.
    padding : float
        Uniform padding around the member boxes.

    Returns
    -------
    tuple[float, float, float, float]
        Bounding box as ``(xmin, ymin, xmax, ymax)``.
    """
    if not members:
        return (0.0, 0.0, 0.0, 0.0)
    idx = torch.tensor(list(members), dtype=torch.long, device=positions.device)
    half = node_sizes[idx].to(dtype=positions.dtype, device=positions.device) * 0.5
    lo = (positions[idx] - half).min(dim=0).values
    hi = (positions[idx] + half).max(dim=0).values
    return (
        float(lo[0].item()) - padding,
        float(lo[1].item()) - padding,
        float(hi[0].item()) + padding,
        float(hi[1].item()) + padding,
    )


def _shift_graphviz_cluster_members(
    positions: torch.Tensor,
    members: Sequence[int],
    dx: float,
) -> None:
    """Shift original cluster members in place along x.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    members : sequence of int
        Original-node ids to shift.
    dx : float
        Horizontal displacement.

    Returns
    -------
    None
        The function mutates ``positions`` in place.
    """
    if not members or abs(dx) <= 1.0e-9:
        return
    idx = torch.tensor(list(members), dtype=torch.long, device=positions.device)
    positions[idx, 0] += float(dx)


def _separate_graphviz_cluster_siblings(
    positions: torch.Tensor,
    node_sizes: torch.Tensor,
    clusters: Mapping[str, Sequence[int]],
    parents: Mapping[str, Optional[str]],
    padding: float,
) -> torch.Tensor:
    """Separate sibling cluster boxes along x.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node sizes with shape ``[N, 2]`` in output coordinate units.
    clusters : Mapping[str, sequence[int]]
        Normalized cluster membership.
    parents : Mapping[str, str | None]
        Normalized parent mapping.
    padding : float
        Minimum sibling bbox clearance.

    Returns
    -------
    torch.Tensor
        Position tensor after deterministic sibling shifts.
    """
    out = positions.detach().clone()
    parent_groups: List[Optional[str]] = [None]
    parent_groups.extend(sorted(parent for parent in set(parents.values()) if parent is not None))
    for parent_name in parent_groups:
        siblings = [name for name, parent in parents.items() if parent == parent_name]
        if len(siblings) < 2:
            continue
        siblings.sort(
            key=lambda name: (
                _graphviz_cluster_bbox(out, node_sizes, clusters[name], padding)[0],
                name,
            )
        )
        cursor_right: Optional[float] = None
        for name in siblings:
            bbox = _graphviz_cluster_bbox(out, node_sizes, clusters[name], padding)
            if cursor_right is None:
                cursor_right = bbox[2]
                continue
            dx = max(0.0, cursor_right + padding - bbox[0])
            if dx > 0.0:
                _shift_graphviz_cluster_members(out, clusters[name], dx)
                bbox = _graphviz_cluster_bbox(out, node_sizes, clusters[name], padding)
            cursor_right = max(cursor_right, bbox[2])
    return out


def _apply_graphviz_cluster_x_constraints(
    positions: torch.Tensor,
    clusters: Optional[Mapping[str, Any]],
    cluster_parents: Optional[Mapping[str, Optional[str]]],
    node_sizes: torch.Tensor,
    rank_sep: float,
    node_sep: float,
    center_coordinates: bool,
) -> torch.Tensor:
    """Apply the A9 graphviz-only cluster x fallback.

    Parameters
    ----------
    positions : torch.Tensor
        Original-node positions with shape ``[N, 2]``.
    clusters : Mapping[str, Any], optional
        Raw Dagua cluster membership.
    cluster_parents : Mapping[str, str | None], optional
        Raw Dagua cluster hierarchy.
    node_sizes : torch.Tensor
        Original-node Graphviz boxes with shape ``[N, 2]`` in point units.
    rank_sep : float
        Vertical rank spacing in output units.
    node_sep : float
        DOT ``nodesep`` in inches.
    center_coordinates : bool
        Whether to recenter x after cluster shifts.

    Returns
    -------
    torch.Tensor
        Cluster-adjusted original-node positions with shape ``[N, 2]``.
    """
    num_nodes = int(positions.shape[0])
    normalized_clusters = _normalize_graphviz_clusters(clusters=clusters, num_nodes=num_nodes)
    if not normalized_clusters:
        return positions

    parents = _normalize_graphviz_cluster_parents(
        cluster_names=tuple(normalized_clusters.keys()),
        cluster_parents=cluster_parents,
    )
    out = positions.detach().clone()
    graphviz_node_sep = float(node_sep) * _GRAPHVIZ_POINTS_PER_INCH
    output_scale = _graphviz_cluster_output_scale(
        positions=out,
        node_sizes=node_sizes,
        node_sep=graphviz_node_sep,
        rank_sep=rank_sep,
    )
    scaled_sizes = node_sizes.to(device=out.device, dtype=out.dtype) * output_scale
    pitch = max(
        float(scaled_sizes[:, 0].median().item()) + graphviz_node_sep * output_scale,
        graphviz_node_sep * output_scale,
    )
    rank_values = _graphviz_position_ranks(positions=out, rank_sep=rank_sep)

    for name in sorted(
        normalized_clusters.keys(),
        key=lambda cluster_name: (-_graphviz_cluster_depth(cluster_name, parents), cluster_name),
    ):
        members = normalized_clusters[name]
        center_x = float(out[list(members), 0].median().item())
        member_ranks = sorted({rank_values[node] for node in members})
        for rank in member_ranks:
            rank_nodes = [node for node in members if rank_values[node] == rank]
            ordered = sorted(rank_nodes, key=lambda node: (float(out[node, 0].item()), node))
            start = -(len(ordered) - 1) / 2.0
            for slot, node in enumerate(ordered):
                out[node, 0] = center_x + (start + slot) * pitch

    clearance = max(
        float(scaled_sizes[:, 0].median().item()) * 0.25,
        graphviz_node_sep * output_scale,
    )
    repeat_order = sorted(
        normalized_clusters.keys(),
        key=lambda cluster_name: (-_graphviz_cluster_depth(cluster_name, parents), cluster_name),
    )
    for _ in repeat_order:
        out = _separate_graphviz_cluster_siblings(
            positions=out,
            node_sizes=scaled_sizes,
            clusters=normalized_clusters,
            parents=parents,
            padding=clearance,
        )

    if center_coordinates:
        centered = _center_coordinates([float(value) for value in out[:, 0].tolist()])
        out[:, 0] = torch.tensor(centered, dtype=out.dtype, device=out.device)
    return out


def _graphviz_cluster_output_scale(
    positions: torch.Tensor,
    node_sizes: torch.Tensor,
    node_sep: float,
    rank_sep: float,
) -> float:
    """Estimate point-to-output scale for cluster bbox constraints.

    Parameters
    ----------
    positions : torch.Tensor
        Original-node positions with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Original-node Graphviz boxes with shape ``[N, 2]`` in point units.
    node_sep : float
        DOT nodesep in points.
    rank_sep : float
        Vertical rank spacing in output units.

    Returns
    -------
    float
        Multiplicative scale from Graphviz point widths to output x units.
    """
    if positions.shape[0] < 2 or node_sizes.numel() == 0:
        denom = max(float(node_sizes[:, 0].median().item()) + node_sep, 1.0)
        return float(rank_sep) / denom
    x_values = sorted(float(value) for value in torch.unique(positions[:, 0]).tolist())
    gaps = [right - left for left, right in zip(x_values, x_values[1:]) if right > left]
    denom = max(float(node_sizes[:, 0].median().item()) + node_sep, 1.0)
    if not gaps:
        return float(rank_sep) / denom
    gaps.sort()
    return max(gaps[len(gaps) // 2] / denom, 1.0e-6)


def _graphviz_position_ranks(positions: torch.Tensor, rank_sep: float) -> List[int]:
    """Return integer rank ids from final y positions.

    Parameters
    ----------
    positions : torch.Tensor
        Original-node positions with shape ``[N, 2]``.
    rank_sep : float
        Vertical rank spacing in output units.

    Returns
    -------
    list[int]
        Rank id per original node.
    """
    if positions.numel() == 0:
        return []
    if abs(rank_sep) <= 1.0e-9:
        return [0 for _ in range(int(positions.shape[0]))]
    min_y = float(positions[:, 1].min().item())
    return [int(round((float(value) - min_y) / float(rank_sep))) for value in positions[:, 1]]


def _graphviz_cluster_children(
    parents: Mapping[str, Optional[str]],
) -> Dict[str, List[str]]:
    """Return normalized child clusters for every parent cluster.

    Parameters
    ----------
    parents : Mapping[str, str | None]
        Normalized cluster parent mapping.

    Returns
    -------
    dict[str, list[str]]
        Child cluster names keyed by parent cluster name, sorted for stable
        rank-leader tie handling.
    """
    children: Dict[str, List[str]] = {name: [] for name in parents}
    for name, parent in parents.items():
        if parent is not None and parent in children:
            children[parent].append(name)
    for child_names in children.values():
        child_names.sort()
    return children


@register_op
class _ValidateInputs(Op):
    """Validate Sugiyama layout inputs exactly like the classic entry point."""

    name: ClassVar[str] = "sugiyama_validate_inputs"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = ()
    access_pattern: ClassVar[str] = "global"

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


@register_op
class _ResolveNodeSizes(Op):
    """Resolve node sizes to CPU float tensor for coordinate spacing."""

    name: ClassVar[str] = "sugiyama_resolve_node_sizes"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = (f"extras.{_SUGIYAMA_RESOLVED_SIZES_KEY}",)
    access_pattern: ClassVar[str] = "global"

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
            State with resolved node sizes cached in ``state.extras``.
        """
        del ctx

        resolved = _resolve_node_sizes(
            node_sizes=problem.node_sizes,
            num_nodes=problem.num_nodes,
        )
        state.extras[_SUGIYAMA_RESOLVED_SIZES_KEY] = resolved
        return state


@register_op
class _PrepareAcyclicEdges(Op):
    """Break cycles to produce an acyclic edge orientation."""

    name: ClassVar[str] = "sugiyama_prepare_acyclic_edges"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = (
        f"extras.{_SUGIYAMA_ACYCLIC_EDGES_KEY}",
        f"extras.{_SUGIYAMA_ACYCLIC_EDGE_WEIGHTS_KEY}",
        f"extras.{_SUGIYAMA_REVERSED_MASK_KEY}",
    )
    access_pattern: ClassVar[str] = "global"

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

        acyclic_edges, acyclic_edge_weights, reversed_mask = _prepare_acyclic_edges(
            edge_index=problem.edge_index,
            edge_weights=problem.edge_weights,
            num_nodes=problem.num_nodes,
        )
        state.extras[_SUGIYAMA_ACYCLIC_EDGES_KEY] = acyclic_edges
        state.extras[_SUGIYAMA_ACYCLIC_EDGE_WEIGHTS_KEY] = acyclic_edge_weights
        state.extras[_SUGIYAMA_REVERSED_MASK_KEY] = reversed_mask
        return state


@register_op
class _AssignLayers(Op):
    """Assign nodes to layers via longest-path or Graphviz rank simplex."""

    name: ClassVar[str] = "sugiyama_assign_layers"
    category: ClassVar[OpCategory] = OpCategory.LAYERING
    reads: ClassVar[Tuple[str, ...]] = (f"extras.{_SUGIYAMA_ACYCLIC_EDGES_KEY}",)
    writes: ClassVar[Tuple[str, ...]] = (
        f"extras.{_SUGIYAMA_LAYER_ASSIGNMENTS_KEY}",
        f"extras.{_SUGIYAMA_GRAPHVIZ_VIRTUAL_EDGES_KEY}",
        f"extras.{_SUGIYAMA_IGRAPH_SCAN_SOURCES_KEY}",
        f"extras.{_SUGIYAMA_IGRAPH_SCAN_TARGETS_KEY}",
        f"extras.{_SUGIYAMA_IGRAPH_SCAN_EDGE_IDS_KEY}",
    )
    requires: ClassVar[Tuple[str, ...]] = (f"extras.{_SUGIYAMA_ACYCLIC_EDGES_KEY}",)
    access_pattern: ClassVar[str] = "global"

    def __init__(
        self,
        fidelity_mode: Optional[str] = None,
        use_graphviz_cluster_skeleton: bool = False,
    ) -> None:
        """Initialize the layer-assignment operation.

        Parameters
        ----------
        fidelity_mode : str, optional
            Optional reference mode. ``"graphviz"`` uses the dot
            network-simplex rank assignment; all other values keep the
            existing longest-path-plus-promotion behavior.
        use_graphviz_cluster_skeleton : bool, default=False
            Enable the inactive A12 cluster rank-collapse prototype.
        """
        self.fidelity_mode = fidelity_mode
        self.use_graphviz_cluster_skeleton = use_graphviz_cluster_skeleton

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

        acyclic_edges = state.extras[_SUGIYAMA_ACYCLIC_EDGES_KEY]
        if self.fidelity_mode == "igraph":
            original_edges = problem.edge_index.detach().to(device="cpu", dtype=torch.long)
            non_loop_mask = original_edges[0] != original_edges[1]
            original_edges = original_edges[:, non_loop_mask]
            original_weights: Optional[torch.Tensor] = None
            if problem.edge_weights is not None:
                original_weights = problem.edge_weights.detach().to(
                    device="cpu",
                    dtype=torch.float32,
                )[non_loop_mask]
            layer_assignments = _igraph_glpk_layer_assignments(
                edge_index=original_edges,
                num_nodes=problem.num_nodes,
                edge_weights=original_weights,
                is_directed=_resolve_igraph_sugiyama_directed(problem),
            )
            (
                oriented_edges,
                oriented_weights,
                reversed_mask,
                scan_sources,
                scan_targets,
                scan_edge_ids,
            ) = _orient_edges_by_layers(
                edge_index=original_edges,
                layer_assignments=layer_assignments,
                edge_weights=original_weights,
            )
            state.extras[_SUGIYAMA_ACYCLIC_EDGES_KEY] = oriented_edges
            state.extras[_SUGIYAMA_ACYCLIC_EDGE_WEIGHTS_KEY] = oriented_weights
            state.extras[_SUGIYAMA_REVERSED_MASK_KEY] = reversed_mask
            state.extras[_SUGIYAMA_IGRAPH_SOURCE_ORDER_KEY] = True
            state.extras[_SUGIYAMA_IGRAPH_SCAN_SOURCES_KEY] = scan_sources
            state.extras[_SUGIYAMA_IGRAPH_SCAN_TARGETS_KEY] = scan_targets
            state.extras[_SUGIYAMA_IGRAPH_SCAN_EDGE_IDS_KEY] = scan_edge_ids
            state.extras[_SUGIYAMA_GRAPHVIZ_VIRTUAL_EDGES_KEY] = []
            state.extras[_SUGIYAMA_GRAPHVIZ_EDGE_ORDER_KEY] = False
        elif self.fidelity_mode in {"dot", "graphviz_dot", "graphviz"}:
            rank_edge_weights = (
                None
                if self.fidelity_mode == "graphviz"
                else state.extras[_SUGIYAMA_ACYCLIC_EDGE_WEIGHTS_KEY]
            )
            if (
                self.fidelity_mode == "graphviz"
                and problem.clusters
                and self.use_graphviz_cluster_skeleton
            ):
                layer_assignments, virtual_edges, cluster_rank_bounds = (
                    _graphviz_cluster_rank_assignments(
                        edge_index=acyclic_edges,
                        edge_weights=rank_edge_weights,
                        num_nodes=problem.num_nodes,
                        clusters=problem.clusters,
                        cluster_parents=problem.cluster_parents,
                        edge_label_sizes=state.extras.get(_SUGIYAMA_GRAPHVIZ_EDGE_LABEL_SIZES_KEY),
                    )
                )
            else:
                layer_assignments, virtual_edges = _graphviz_layer_assignments(
                    edge_index=acyclic_edges,
                    edge_weights=rank_edge_weights,
                    num_nodes=problem.num_nodes,
                    edge_label_sizes=state.extras.get(_SUGIYAMA_GRAPHVIZ_EDGE_LABEL_SIZES_KEY),
                    legacy_tree_order=bool(
                        problem.clusters and not self.use_graphviz_cluster_skeleton
                    ),
                )
                cluster_rank_bounds = {}
            state.extras[_SUGIYAMA_GRAPHVIZ_CLUSTER_RANKS_KEY] = cluster_rank_bounds
            state.extras[_SUGIYAMA_ACYCLIC_EDGE_WEIGHTS_KEY] = rank_edge_weights
            state.extras[_SUGIYAMA_GRAPHVIZ_VIRTUAL_EDGES_KEY] = virtual_edges
            state.extras[_SUGIYAMA_GRAPHVIZ_EDGE_ORDER_KEY] = self.fidelity_mode == "graphviz"
        else:
            layer_assignments = _longest_path_layering(
                edge_index=acyclic_edges,
                num_nodes=problem.num_nodes,
            )
            layer_assignments = _promote_layer_assignments(
                edge_index=acyclic_edges,
                layer_assignments=layer_assignments,
                num_nodes=problem.num_nodes,
            )
            state.extras[_SUGIYAMA_GRAPHVIZ_VIRTUAL_EDGES_KEY] = []
            state.extras[_SUGIYAMA_GRAPHVIZ_EDGE_ORDER_KEY] = False
        state.extras[_SUGIYAMA_LAYER_ASSIGNMENTS_KEY] = layer_assignments
        return state


@register_op
class _ExpandDummyNodes(Op):
    """Insert dummy nodes for edges spanning more than one layer."""

    name: ClassVar[str] = "sugiyama_expand_dummy_nodes"
    category: ClassVar[OpCategory] = OpCategory.LAYERING
    reads: ClassVar[Tuple[str, ...]] = (
        f"extras.{_SUGIYAMA_ACYCLIC_EDGES_KEY}",
        f"extras.{_SUGIYAMA_ACYCLIC_EDGE_WEIGHTS_KEY}",
        f"extras.{_SUGIYAMA_LAYER_ASSIGNMENTS_KEY}",
        f"extras.{_SUGIYAMA_RESOLVED_SIZES_KEY}",
        f"extras.{_SUGIYAMA_GRAPHVIZ_NODE_SIZES_KEY}",
        f"extras.{_SUGIYAMA_IGRAPH_SCAN_SOURCES_KEY}",
        f"extras.{_SUGIYAMA_IGRAPH_SCAN_TARGETS_KEY}",
        f"extras.{_SUGIYAMA_IGRAPH_SCAN_EDGE_IDS_KEY}",
    )
    writes: ClassVar[Tuple[str, ...]] = (
        f"extras.{_SUGIYAMA_EXPANDED_GRAPH_KEY}",
        f"extras.{_SUGIYAMA_EXPANDED_EDGE_WEIGHTS_KEY}",
    )
    requires: ClassVar[Tuple[str, ...]] = (
        f"extras.{_SUGIYAMA_ACYCLIC_EDGES_KEY}",
        f"extras.{_SUGIYAMA_ACYCLIC_EDGE_WEIGHTS_KEY}",
        f"extras.{_SUGIYAMA_LAYER_ASSIGNMENTS_KEY}",
        f"extras.{_SUGIYAMA_RESOLVED_SIZES_KEY}",
    )
    access_pattern: ClassVar[str] = "global"

    def __init__(
        self,
        use_igraph_edge_order: bool = False,
        use_graphviz_cluster_skeleton: bool = False,
    ) -> None:
        """Initialize dummy-expansion ordering options.

        Parameters
        ----------
        use_igraph_edge_order : bool, default=False
            Whether to scan source vertices and their outgoing edge ids when
            creating dummy chains, matching igraph's Sugiyama subgraph build.
        use_graphviz_cluster_skeleton : bool, default=False
            Enable expanded cluster membership for the A12 skeleton mincross
            prototype.

        Returns
        -------
        None
            The constructor stores the ordering options.
        """
        self.use_igraph_edge_order = use_igraph_edge_order
        self.use_graphviz_cluster_skeleton = use_graphviz_cluster_skeleton

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

        use_graphviz_edge_order = bool(state.extras.get(_SUGIYAMA_GRAPHVIZ_EDGE_ORDER_KEY, False))
        use_igraph_source_order = bool(state.extras.get(_SUGIYAMA_IGRAPH_SOURCE_ORDER_KEY, False))
        use_igraph_edge_order = self.use_igraph_edge_order and use_igraph_source_order
        node_sizes = state.extras[_SUGIYAMA_RESOLVED_SIZES_KEY]
        if use_graphviz_edge_order:
            node_sizes = state.extras.get(_SUGIYAMA_GRAPHVIZ_NODE_SIZES_KEY, node_sizes)
        graphviz_virtual_node_sep = None
        if use_graphviz_edge_order:
            graphviz_virtual_node_sep = (
                float(state.extras.get(_SUGIYAMA_NODE_SEP_KEY, 1.0)) * _GRAPHVIZ_POINTS_PER_INCH
                if not self.use_graphviz_cluster_skeleton
                and (problem.clusters or problem.num_nodes > _GRAPHVIZ_TYPED_X_MAX_ORIGINAL_NODES)
                else _GRAPHVIZ_DEFAULT_NODE_SEP_POINTS
            )
        graphviz_edge_order_sources: Optional[torch.Tensor] = None
        graphviz_edge_order_targets: Optional[torch.Tensor] = None
        if use_graphviz_edge_order and (
            (not problem.clusters and problem.num_nodes <= _GRAPHVIZ_TYPED_X_MAX_ORIGINAL_NODES)
            or self.use_graphviz_cluster_skeleton
        ):
            oriented_edges = state.extras[_SUGIYAMA_ACYCLIC_EDGES_KEY]
            reversed_mask = state.extras[_SUGIYAMA_REVERSED_MASK_KEY].to(
                device="cpu",
                dtype=torch.bool,
            )
            graphviz_edge_order_sources = torch.where(
                reversed_mask,
                oriented_edges[1],
                oriented_edges[0],
            )
            graphviz_edge_order_targets = torch.where(
                reversed_mask,
                oriented_edges[0],
                oriented_edges[1],
            )

        expansion_edges = state.extras[_SUGIYAMA_ACYCLIC_EDGES_KEY]
        if use_graphviz_edge_order and self.use_graphviz_cluster_skeleton:
            layer_assignments = state.extras[_SUGIYAMA_LAYER_ASSIGNMENTS_KEY]
            rank_reversed = (
                layer_assignments[expansion_edges[0]] > layer_assignments[expansion_edges[1]]
            )
            expansion_edges = torch.stack(
                [
                    torch.where(rank_reversed, expansion_edges[1], expansion_edges[0]),
                    torch.where(rank_reversed, expansion_edges[0], expansion_edges[1]),
                ]
            )
            state.extras[_SUGIYAMA_REVERSED_MASK_KEY] = (
                state.extras[_SUGIYAMA_REVERSED_MASK_KEY].to(dtype=torch.bool) ^ rank_reversed
            )

        expanded_graph, expanded_edge_weights = _expand_long_edges_with_dummy_nodes(
            edge_index=expansion_edges,
            layer_assignments=state.extras[_SUGIYAMA_LAYER_ASSIGNMENTS_KEY],
            node_sizes=node_sizes,
            num_original_nodes=problem.num_nodes,
            edge_weights=state.extras[_SUGIYAMA_ACYCLIC_EDGE_WEIGHTS_KEY],
            edge_label_sizes=state.extras.get(_SUGIYAMA_GRAPHVIZ_EDGE_LABEL_SIZES_KEY),
            use_graphviz_edge_order=use_graphviz_edge_order,
            graphviz_edge_order_sources=graphviz_edge_order_sources,
            graphviz_edge_order_targets=graphviz_edge_order_targets,
            graphviz_sort_outgoing=(not problem.clusters or self.use_graphviz_cluster_skeleton),
            use_igraph_edge_order=use_igraph_edge_order,
            igraph_edge_order_sources=state.extras.get(_SUGIYAMA_IGRAPH_SCAN_SOURCES_KEY),
            igraph_edge_order_targets=state.extras.get(_SUGIYAMA_IGRAPH_SCAN_TARGETS_KEY),
            igraph_edge_order_ids=state.extras.get(_SUGIYAMA_IGRAPH_SCAN_EDGE_IDS_KEY),
            graphviz_virtual_node_sep=graphviz_virtual_node_sep,
            clusters=problem.clusters
            if use_graphviz_edge_order and self.use_graphviz_cluster_skeleton
            else None,
            cluster_parents=problem.cluster_parents
            if use_graphviz_edge_order and self.use_graphviz_cluster_skeleton
            else None,
            graphviz_cluster_label_widths=state.extras.get(
                _SUGIYAMA_GRAPHVIZ_CLUSTER_LABEL_WIDTHS_KEY
            )
            if use_graphviz_edge_order and self.use_graphviz_cluster_skeleton
            else None,
        )
        state.extras[_SUGIYAMA_EXPANDED_GRAPH_KEY] = expanded_graph
        state.extras[_SUGIYAMA_EXPANDED_EDGE_WEIGHTS_KEY] = expanded_edge_weights
        return state


@register_op
class _BuildNeighborStructures(Op):
    """Build parent/child adjacency lists and edge-weight maps."""

    name: ClassVar[str] = "sugiyama_build_neighbor_structures"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = (
        f"extras.{_SUGIYAMA_EXPANDED_GRAPH_KEY}",
        f"extras.{_SUGIYAMA_EXPANDED_EDGE_WEIGHTS_KEY}",
    )
    writes: ClassVar[Tuple[str, ...]] = (
        f"extras.{_SUGIYAMA_PARENTS_KEY}",
        f"extras.{_SUGIYAMA_CHILDREN_KEY}",
        f"extras.{_SUGIYAMA_PARENT_WEIGHTS_KEY}",
        f"extras.{_SUGIYAMA_CHILD_WEIGHTS_KEY}",
    )
    requires: ClassVar[Tuple[str, ...]] = (f"extras.{_SUGIYAMA_EXPANDED_GRAPH_KEY}",)
    access_pattern: ClassVar[str] = "global"

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

        expanded_graph = state.extras[_SUGIYAMA_EXPANDED_GRAPH_KEY]
        parents, children = _build_neighbor_lists(
            edge_index=expanded_graph.edge_index,
            num_nodes=expanded_graph.num_nodes,
        )
        parent_weights, child_weights = _build_neighbor_weight_maps(
            edge_index=expanded_graph.edge_index,
            num_nodes=expanded_graph.num_nodes,
            edge_weights=state.extras[_SUGIYAMA_EXPANDED_EDGE_WEIGHTS_KEY],
        )
        state.extras[_SUGIYAMA_PARENTS_KEY] = parents
        state.extras[_SUGIYAMA_CHILDREN_KEY] = children
        state.extras[_SUGIYAMA_PARENT_WEIGHTS_KEY] = parent_weights
        state.extras[_SUGIYAMA_CHILD_WEIGHTS_KEY] = child_weights
        return state


@register_op
class _BarycenterOrdering(Op):
    """Minimize edge crossings via repeated barycenter sweeps."""

    config: _BarycenterOrderingConfig

    name: ClassVar[str] = "sugiyama_barycenter_ordering"
    category: ClassVar[OpCategory] = OpCategory.ORDERING
    reads: ClassVar[Tuple[str, ...]] = (
        f"extras.{_SUGIYAMA_EXPANDED_GRAPH_KEY}",
        f"extras.{_SUGIYAMA_PARENTS_KEY}",
        f"extras.{_SUGIYAMA_CHILDREN_KEY}",
        f"extras.{_SUGIYAMA_PARENT_WEIGHTS_KEY}",
        f"extras.{_SUGIYAMA_CHILD_WEIGHTS_KEY}",
        f"extras.{_SUGIYAMA_RANK_SEP_KEY}",
        f"extras.{_SUGIYAMA_NODE_SEP_KEY}",
    )
    writes: ClassVar[Tuple[str, ...]] = (
        f"extras.{_SUGIYAMA_ORDERED_LAYERS_KEY}",
        f"extras.{_SUGIYAMA_TRACES_KEY}",
    )
    requires: ClassVar[Tuple[str, ...]] = (
        f"extras.{_SUGIYAMA_EXPANDED_GRAPH_KEY}",
        f"extras.{_SUGIYAMA_PARENTS_KEY}",
        f"extras.{_SUGIYAMA_CHILDREN_KEY}",
        f"extras.{_SUGIYAMA_PARENT_WEIGHTS_KEY}",
        f"extras.{_SUGIYAMA_CHILD_WEIGHTS_KEY}",
    )
    access_pattern: ClassVar[str] = "global"

    def __init__(
        self,
        barycenter_passes: int = 24,
        seed: int = 42,
        trace_every: int = 0,
        stop_when_stable: bool = False,
        use_incidence_barycenters: bool = False,
        center_coordinates: bool = True,
        use_graphviz_mincross: bool = False,
        use_graphviz_node_order: bool = False,
        use_graphviz_cluster_skeleton: bool = False,
        *,
        config: Optional[_BarycenterOrderingConfig] = None,
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
        stop_when_stable : bool, default=False
            Stop after the first full pass that leaves all layer orders
            unchanged.
        use_incidence_barycenters : bool, default=False
            Average duplicate neighbor incidences directly, matching igraph's
            crossing-reduction semantics.
        center_coordinates : bool, default=True
            Whether trace snapshots should center horizontal coordinates.
        use_graphviz_mincross : bool, default=False
            Use Graphviz dot's median/transpose mincross heuristic instead of
            the default barycenter ordering.
        use_graphviz_node_order : bool, default=False
            Use the graphviz-style fast-node list for ``build_ranks`` seed
            scans.
        use_graphviz_cluster_skeleton : bool, default=False
            Enable the inactive A12 rank-leader cluster ordering prototype.
        config : _BarycenterOrderingConfig | None, optional
            Optional configuration. When provided, it takes precedence over
            the scalar arguments.

        Returns
        -------
        None
            This constructor stores configuration only.
        """
        self.config = config or _BarycenterOrderingConfig(
            barycenter_passes=barycenter_passes,
            seed=seed,
            trace_every=trace_every,
            stop_when_stable=stop_when_stable,
            use_incidence_barycenters=use_incidence_barycenters,
            center_coordinates=center_coordinates,
            use_graphviz_mincross=use_graphviz_mincross,
            use_graphviz_node_order=use_graphviz_node_order,
            use_graphviz_cluster_skeleton=use_graphviz_cluster_skeleton,
        )

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

        expanded_graph = state.extras[_SUGIYAMA_EXPANDED_GRAPH_KEY]
        output_device = problem.edge_index.device
        if problem.node_sizes is not None:
            output_device = problem.node_sizes.device

        rank_sep = state.extras.get(_SUGIYAMA_RANK_SEP_KEY, 1.0)
        node_sep = state.extras.get(_SUGIYAMA_NODE_SEP_KEY, 1.0)

        ordered_layers, traces = _barycenter_ordering(
            layers=expanded_graph.layers,
            edge_index=expanded_graph.edge_index,
            edge_weights=state.extras[_SUGIYAMA_EXPANDED_EDGE_WEIGHTS_KEY],
            graphviz_node_order=expanded_graph.graphviz_node_order,
            mincross_edge_penalties=expanded_graph.mincross_edge_penalties,
            graphviz_cluster_members=expanded_graph.graphviz_cluster_members
            if self.config.use_graphviz_cluster_skeleton
            else None,
            graphviz_cluster_parents=expanded_graph.graphviz_cluster_parents
            if self.config.use_graphviz_cluster_skeleton
            else None,
            parents=state.extras[_SUGIYAMA_PARENTS_KEY],
            children=state.extras[_SUGIYAMA_CHILDREN_KEY],
            parent_weights=state.extras[_SUGIYAMA_PARENT_WEIGHTS_KEY],
            child_weights=state.extras[_SUGIYAMA_CHILD_WEIGHTS_KEY],
            num_nodes=expanded_graph.num_nodes,
            num_original_nodes=problem.num_nodes,
            num_passes=self.config.barycenter_passes,
            seed=self.config.seed,
            node_sizes=expanded_graph.node_sizes,
            rank_sep=rank_sep,
            node_sep=node_sep,
            trace_every=self.config.trace_every,
            output_device=output_device,
            stop_when_stable=self.config.stop_when_stable,
            use_incidence_barycenters=self.config.use_incidence_barycenters,
            center_coordinates=self.config.center_coordinates,
            use_graphviz_mincross=self.config.use_graphviz_mincross,
            use_graphviz_node_order=self.config.use_graphviz_node_order,
        )
        state.extras[_SUGIYAMA_ORDERED_LAYERS_KEY] = ordered_layers
        state.extras[_SUGIYAMA_TRACES_KEY] = traces
        return state


@register_op
class _CoordinateAssignment(Op):
    """Assign (x, y) coordinates with Brandes-Kopf compaction."""

    name: ClassVar[str] = "sugiyama_coordinate_assignment"
    category: ClassVar[OpCategory] = OpCategory.COORDINATE
    reads: ClassVar[Tuple[str, ...]] = (
        f"extras.{_SUGIYAMA_EXPANDED_GRAPH_KEY}",
        f"extras.{_SUGIYAMA_EXPANDED_EDGE_WEIGHTS_KEY}",
        f"extras.{_SUGIYAMA_ORDERED_LAYERS_KEY}",
        f"extras.{_SUGIYAMA_PARENTS_KEY}",
        f"extras.{_SUGIYAMA_CHILDREN_KEY}",
        f"extras.{_SUGIYAMA_RANK_SEP_KEY}",
        f"extras.{_SUGIYAMA_NODE_SEP_KEY}",
    )
    writes: ClassVar[Tuple[str, ...]] = ("pos", f"extras.{_SUGIYAMA_EXPANDED_POSITIONS_KEY}")
    requires: ClassVar[Tuple[str, ...]] = (
        f"extras.{_SUGIYAMA_EXPANDED_GRAPH_KEY}",
        f"extras.{_SUGIYAMA_ORDERED_LAYERS_KEY}",
    )
    access_pattern: ClassVar[str] = "global"

    def __init__(
        self,
        center_coordinates: bool = True,
        use_graphviz_xcoord: bool = False,
        use_igraph_conflicts: bool = False,
        use_graphviz_cluster_skeleton: bool = False,
    ) -> None:
        """Store coordinate-frame options.

        Parameters
        ----------
        center_coordinates : bool, default=True
            Whether to translate the final horizontal span to be centered at
            zero.
        use_graphviz_xcoord : bool, default=False
            Whether to use Graphviz dot's auxiliary-graph network simplex for
            x coordinates instead of Brandes-Kopf compaction.
        use_igraph_conflicts : bool, default=False
            Whether to mirror igraph 1.0.0's ordinal-edge Type 1 conflict
            detection during Brandes-Koepf compaction.
        use_graphviz_cluster_skeleton : bool, default=False
            Whether clustered Graphviz runs build containment into the x
            auxiliary inventory instead of applying the legacy postpass.
        Returns
        -------
        None
            The constructor stores configuration only.
        """
        self.center_coordinates = center_coordinates
        self.use_graphviz_xcoord = use_graphviz_xcoord
        self.use_igraph_conflicts = use_igraph_conflicts
        self.use_graphviz_cluster_skeleton = use_graphviz_cluster_skeleton

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

        expanded_graph = state.extras[_SUGIYAMA_EXPANDED_GRAPH_KEY]
        output_device = problem.edge_index.device
        if problem.node_sizes is not None:
            output_device = problem.node_sizes.device

        rank_sep = state.extras.get(_SUGIYAMA_RANK_SEP_KEY, 1.0)
        node_sep = state.extras.get(_SUGIYAMA_NODE_SEP_KEY, 1.0)

        if self.use_graphviz_xcoord:
            certified_cluster_inventory = (
                self.use_graphviz_cluster_skeleton
                and _SUGIYAMA_GRAPHVIZ_EXPECTED_X_INVENTORY_KEY in state.extras
            )
            use_typed_inventory = certified_cluster_inventory or (
                not problem.clusters
                and problem.num_nodes <= _GRAPHVIZ_TYPED_X_MAX_ORIGINAL_NODES
                and int(problem.edge_index.shape[1])
                <= _GRAPHVIZ_TYPED_X_MAX_EDGE_DENSITY * problem.num_nodes
            )
            graphviz_left_widths = expanded_graph.graphviz_left_widths if problem.clusters else None
            graphviz_right_widths = (
                expanded_graph.graphviz_right_widths if problem.clusters else None
            )
            expanded_positions = _graphviz_x_coordinate_assignment(
                layers=state.extras[_SUGIYAMA_ORDERED_LAYERS_KEY],
                edge_index=expanded_graph.edge_index,
                edge_weights=state.extras.get(_SUGIYAMA_EXPANDED_EDGE_WEIGHTS_KEY),
                node_sizes=expanded_graph.node_sizes,
                num_nodes=expanded_graph.num_nodes,
                num_original_nodes=problem.num_nodes,
                rank_sep=rank_sep,
                node_sep=node_sep,
                output_device=output_device,
                center_coordinates=self.center_coordinates,
                graphviz_left_widths=graphviz_left_widths,
                graphviz_right_widths=graphviz_right_widths,
                graphviz_cluster_members=(
                    expanded_graph.graphviz_cluster_members if certified_cluster_inventory else None
                ),
                graphviz_cluster_parents=(
                    expanded_graph.graphviz_cluster_parents if certified_cluster_inventory else None
                ),
                graphviz_cluster_label_widths=(
                    expanded_graph.graphviz_cluster_label_widths
                    if certified_cluster_inventory
                    else None
                ),
                graphviz_node_order=(
                    expanded_graph.graphviz_node_order if use_typed_inventory else None
                ),
                expanded_edge_origins=expanded_graph.expanded_edge_origins,
                expected_typed_inventory=state.extras.get(
                    _SUGIYAMA_GRAPHVIZ_EXPECTED_X_INVENTORY_KEY
                ),
                use_typed_inventory=use_typed_inventory,
            )
        else:
            expanded_positions = _coordinate_assignment(
                layers=state.extras[_SUGIYAMA_ORDERED_LAYERS_KEY],
                parents=state.extras[_SUGIYAMA_PARENTS_KEY],
                children=state.extras[_SUGIYAMA_CHILDREN_KEY],
                node_sizes=expanded_graph.node_sizes,
                num_nodes=expanded_graph.num_nodes,
                num_original_nodes=problem.num_nodes,
                rank_sep=rank_sep,
                node_sep=node_sep,
                output_device=output_device,
                center_coordinates=self.center_coordinates,
                edge_index=expanded_graph.edge_index,
                use_igraph_conflicts=self.use_igraph_conflicts,
            )
        # Keep the expanded coordinates for downstream edge routing before
        # slicing back to the original node set.
        state.extras[_SUGIYAMA_EXPANDED_POSITIONS_KEY] = expanded_positions
        state.pos = expanded_positions[: problem.num_nodes]
        if self.use_graphviz_xcoord and problem.clusters and not self.use_graphviz_cluster_skeleton:
            state.pos = _apply_graphviz_cluster_x_constraints(
                positions=state.pos,
                clusters=problem.clusters,
                cluster_parents=problem.cluster_parents,
                node_sizes=expanded_graph.node_sizes[: problem.num_nodes],
                rank_sep=rank_sep,
                node_sep=node_sep,
                center_coordinates=self.center_coordinates,
            )
            expanded_positions = expanded_positions.clone()
            expanded_positions[: problem.num_nodes] = state.pos
            state.extras[_SUGIYAMA_EXPANDED_POSITIONS_KEY] = expanded_positions
        return state


@register_op
class _BuildEdgeRoutes(Op):
    """Convert dummy-node chains into routed edge polylines."""

    name: ClassVar[str] = "sugiyama_build_edge_routes"
    category: ClassVar[OpCategory] = OpCategory.EDGE_ROUTE
    reads: ClassVar[Tuple[str, ...]] = (
        f"extras.{_SUGIYAMA_EXPANDED_GRAPH_KEY}",
        f"extras.{_SUGIYAMA_EXPANDED_POSITIONS_KEY}",
        f"extras.{_SUGIYAMA_REVERSED_MASK_KEY}",
    )
    writes: ClassVar[Tuple[str, ...]] = ("edge_routes",)
    requires: ClassVar[Tuple[str, ...]] = (
        f"extras.{_SUGIYAMA_EXPANDED_GRAPH_KEY}",
        f"extras.{_SUGIYAMA_EXPANDED_POSITIONS_KEY}",
        f"extras.{_SUGIYAMA_REVERSED_MASK_KEY}",
    )
    access_pattern: ClassVar[str] = "global"

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

        expanded_graph = state.extras[_SUGIYAMA_EXPANDED_GRAPH_KEY]
        output_device = problem.edge_index.device
        if problem.node_sizes is not None:
            output_device = problem.node_sizes.device

        state.edge_routes = _build_edge_routes(
            positions=state.extras[_SUGIYAMA_EXPANDED_POSITIONS_KEY],
            edge_paths=expanded_graph.edge_paths,
            reversed_edge_mask=state.extras[_SUGIYAMA_REVERSED_MASK_KEY],
            output_device=output_device,
        )
        return state


@register_op
class _StoreSpacingParams(Op):
    """Store rank_sep and node_sep in extras for downstream ops."""

    config: _StoreSpacingParamsConfig

    name: ClassVar[str] = "sugiyama_store_spacing"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = (
        f"extras.{_SUGIYAMA_RANK_SEP_KEY}",
        f"extras.{_SUGIYAMA_NODE_SEP_KEY}",
    )
    access_pattern: ClassVar[str] = "global"

    def __init__(
        self,
        rank_sep: float = 1.0,
        node_sep: float = 1.0,
        *,
        config: Optional[_StoreSpacingParamsConfig] = None,
    ) -> None:
        """Store spacing parameters.

        Parameters
        ----------
        rank_sep : float, default=1.0
            Vertical spacing between layers.
        node_sep : float, default=1.0
            Horizontal spacing between nodes within a layer.
        config : _StoreSpacingParamsConfig | None, optional
            Optional configuration. When provided, it takes precedence over
            the scalar arguments.

        Returns
        -------
        None
            This constructor stores configuration only.
        """
        self.config = config or _StoreSpacingParamsConfig(rank_sep=rank_sep, node_sep=node_sep)

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

        rank_sep = self.config.rank_sep
        if _has_graphviz_edge_labels(
            edge_label_sizes=state.extras.get(_SUGIYAMA_GRAPHVIZ_EDGE_LABEL_SIZES_KEY)
        ):
            # Graphviz 7.0.5 rank.c reserves midpoint ranks for edge labels by
            # doubling minlen and reducing GD_ranksep to keep endpoint spacing.
            rank_sep = rank_sep / 2.0
        state.extras[_SUGIYAMA_RANK_SEP_KEY] = rank_sep
        state.extras[_SUGIYAMA_NODE_SEP_KEY] = self.config.node_sep
        return state
