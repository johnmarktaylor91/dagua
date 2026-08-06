"""Composable stages for the dagre.js 0.8.5 layered layout engine."""

from __future__ import annotations

import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import (
    Any,
    ClassVar,
    Dict,
    Hashable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import torch

from dagua.layout.ops.base import Op
from dagua.layout.ops.brandes_koepf import (
    BRANDES_KOEPF_BORDER_TYPES_KEY,
    BRANDES_KOEPF_DUMMY_NODES_KEY,
    BRANDES_KOEPF_LAYERING_KEY,
    BRANDES_KOEPF_PREDECESSORS_KEY,
    BRANDES_KOEPF_SUCCESSORS_KEY,
    BRANDES_KOEPF_WIDTHS_KEY,
    BRANDES_KOEPF_X_KEY,
)
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_DAGRE_GRAPH_KEY = "dagre_graph"
_DAGRE_RANKS_KEY = "dagre_ranks"
_DAGRE_ORDERING_KEY = "dagre_ordering"
_DAGRE_INTERNAL_POSITIONS_KEY = "dagre_internal_positions"

NodeId = Hashable


@dataclass
class _DagreNode:
    """Mutable node label used by the local Dagre stage ports."""

    width: float
    height: float
    rank: Optional[int] = None
    order: Optional[int] = None
    dummy: Optional[str] = None
    min_rank: Optional[int] = None
    max_rank: Optional[int] = None
    border_top: Optional[NodeId] = None
    border_bottom: Optional[NodeId] = None
    border_left: Dict[int, NodeId] = field(default_factory=dict)
    border_right: Dict[int, NodeId] = field(default_factory=dict)
    border_type: Optional[str] = None
    edge_source: Optional[NodeId] = None
    edge_target: Optional[NodeId] = None


@dataclass
class _DagreEdge:
    """Mutable multigraph edge used by the local Dagre stage ports."""

    source: NodeId
    target: NodeId
    weight: float
    minlen: int
    original_index: int
    active: bool = True
    reversed: bool = False
    nesting_edge: bool = False


@dataclass
class _DagreGraph:
    """Working graph shared across Dagre operations."""

    nodes: Dict[NodeId, _DagreNode]
    node_order: List[NodeId]
    edges: List[_DagreEdge]
    num_original_nodes: int
    original_node_ids: List[NodeId]
    rank_sep: float
    node_sep: float
    edge_sep: float
    rankdir: str
    ranker: str
    acyclicer: str
    self_edges: Dict[int, List[_DagreEdge]] = field(default_factory=dict)
    parents: Dict[NodeId, Optional[NodeId]] = field(default_factory=dict)
    nesting_root: Optional[NodeId] = None
    node_rank_factor: int = 1
    dummy_chains: List[NodeId] = field(default_factory=list)
    next_dummy_id: int = 0

    def add_dummy(self, dummy_type: str, width: float = 0.0, height: float = 0.0) -> NodeId:
        """Append a uniquely identified dummy node.

        Parameters
        ----------
        dummy_type : str
            Dagre dummy-node category.
        width : float, default=0.0
            Dummy box width.
        height : float, default=0.0
            Dummy box height.

        Returns
        -------
        Hashable
            New internal node id.
        """
        node_id = (dummy_type, self.next_dummy_id)
        self.next_dummy_id += 1
        self.nodes[node_id] = _DagreNode(width=width, height=height, dummy=dummy_type)
        self.node_order.append(node_id)
        return node_id

    def set_parent(self, node: NodeId, parent: Optional[NodeId]) -> None:
        """Assign one compound parent relation.

        Parameters
        ----------
        node : Hashable
            Child node or cluster id.
        parent : Hashable | None
            Parent cluster id. ``None`` makes the child a graph root.

        Returns
        -------
        None
            The parent map is updated.
        """
        if node not in self.nodes:
            raise ValueError(f"Unknown Dagre compound child: {node!r}")
        if parent is not None and parent not in self.nodes:
            raise ValueError(f"Unknown Dagre compound parent: {parent!r}")
        if parent == node:
            raise ValueError("Dagre compound node cannot parent itself.")
        self.parents[node] = parent

    def parent_of(self, node: NodeId) -> Optional[NodeId]:
        """Return the direct compound parent for one node.

        Parameters
        ----------
        node : Hashable
            Node or cluster id.

        Returns
        -------
        Hashable | None
            Parent cluster id when assigned.
        """
        return self.parents.get(node)

    def children(self, parent: Optional[NodeId] = None) -> List[NodeId]:
        """Return direct children in graph insertion order.

        Parameters
        ----------
        parent : Hashable | None, optional
            Parent cluster id. ``None`` returns graph-root children.

        Returns
        -------
        list[Hashable]
            Direct children ordered like graphlib's node list.
        """
        return [node for node in self.node_order if self.parents.get(node) == parent]

    def has_compound(self) -> bool:
        """Return whether any node participates in a compound hierarchy.

        Returns
        -------
        bool
            ``True`` when at least one node has a direct parent.
        """
        return any(parent is not None for parent in self.parents.values())

    def non_compound_node_order(self) -> List[NodeId]:
        """Return nodes included by dagre's ``asNonCompoundGraph`` helper.

        Returns
        -------
        list[Hashable]
            Nodes without children, preserving graph insertion order.
        """
        return [node for node in self.node_order if not self.children(node)]

    def add_edge(
        self,
        source: NodeId,
        target: NodeId,
        weight: float,
        minlen: int,
        original_index: int,
        reversed_edge: bool = False,
        nesting_edge: bool = False,
    ) -> _DagreEdge:
        """Append one active multigraph edge.

        Parameters
        ----------
        source : Hashable
            Edge tail.
        target : Hashable
            Edge head.
        weight : float
            Crossing/rank objective weight.
        minlen : int
            Minimum rank span.
        original_index : int
            Source edge index in the caller's tensor.
        reversed_edge : bool, default=False
            Whether the acyclic stage reversed this edge.
        nesting_edge : bool, default=False
            Whether this is a temporary nesting-graph edge.

        Returns
        -------
        _DagreEdge
            Appended edge record.
        """
        edge = _DagreEdge(
            source=source,
            target=target,
            weight=weight,
            minlen=minlen,
            original_index=original_index,
            reversed=reversed_edge,
            nesting_edge=nesting_edge,
        )
        self.edges.append(edge)
        return edge

    def active_edges(self) -> List[_DagreEdge]:
        """Return active edges in graph insertion order.

        Returns
        -------
        list[_DagreEdge]
            Active edge records.
        """
        return [edge for edge in self.edges if edge.active]

    def in_edges(self, node: NodeId) -> List[_DagreEdge]:
        """Return active incoming edges in insertion order.

        Parameters
        ----------
        node : Hashable
            Incident node id.

        Returns
        -------
        list[_DagreEdge]
            Incoming edge records.
        """
        return [edge for edge in self.edges if edge.active and edge.target == node]

    def out_edges(self, node: NodeId) -> List[_DagreEdge]:
        """Return active outgoing edges in insertion order.

        Parameters
        ----------
        node : Hashable
            Incident node id.

        Returns
        -------
        list[_DagreEdge]
            Outgoing edge records.
        """
        return [edge for edge in self.edges if edge.active and edge.source == node]

    def predecessors(self, node: NodeId) -> List[NodeId]:
        """Return distinct predecessors in first-edge order.

        Parameters
        ----------
        node : Hashable
            Target node id.

        Returns
        -------
        list[Hashable]
            Distinct predecessor ids.
        """
        return _unique(edge.source for edge in self.in_edges(node))

    def successors(self, node: NodeId) -> List[NodeId]:
        """Return distinct successors in first-edge order.

        Parameters
        ----------
        node : Hashable
            Source node id.

        Returns
        -------
        list[Hashable]
            Distinct successor ids.
        """
        return _unique(edge.target for edge in self.out_edges(node))


def _unique(values: Sequence[NodeId] | object) -> List[NodeId]:
    """Return values without duplicates while preserving order.

    Parameters
    ----------
    values : iterable[Hashable]
        Values to deduplicate.

    Returns
    -------
    list[Hashable]
        First occurrence of each value.
    """
    output: List[NodeId] = []
    seen: Set[NodeId] = set()
    for value in values:  # type: ignore[union-attr]
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _cluster_node_id(cluster_name: str) -> NodeId:
    """Return the internal node id for one declared cluster.

    Parameters
    ----------
    cluster_name : str
        External cluster identifier.

    Returns
    -------
    Hashable
        Internal cluster node id.
    """
    return ("cluster", cluster_name)


def _flatten_cluster_members(members: Any) -> List[int]:
    """Return integer leaf node ids from nested cluster membership payloads.

    Parameters
    ----------
    members : Any
        Cluster member payload from ``LayoutProblem.clusters``.

    Returns
    -------
    list[int]
        Flattened node indices in encounter order.
    """
    output: List[int] = []

    def visit(value: Any) -> None:
        """Collect leaves from one nested membership value.

        Parameters
        ----------
        value : Any
            Nested member payload.

        Returns
        -------
        None
            ``output`` is mutated.
        """
        if isinstance(value, Mapping):
            for child_value in value.values():
                visit(child_value)
            return
        if isinstance(value, (str, bytes)):
            try:
                output.append(int(value))
            except ValueError:
                return
            return
        if isinstance(value, Iterable):
            for child_value in value:
                visit(child_value)
            return
        try:
            output.append(int(value))
        except (TypeError, ValueError):
            return

    visit(members)
    return output


def _cluster_children_by_parent(
    clusters: Mapping[str, Any],
    cluster_parents: Optional[Mapping[str, Optional[str]]],
) -> Dict[Optional[str], List[str]]:
    """Build a deterministic cluster-child map.

    Parameters
    ----------
    clusters : mapping[str, Any]
        Declared cluster membership mapping.
    cluster_parents : mapping[str, str | None] | None
        Optional cluster parent mapping.

    Returns
    -------
    dict[str | None, list[str]]
        Cluster names grouped by valid parent, sorted by cluster id.
    """
    by_parent: Dict[Optional[str], List[str]] = {}
    parents = cluster_parents or {}
    for cluster_name in sorted(str(name) for name in clusters):
        parent = parents.get(cluster_name)
        if parent not in clusters:
            parent = None
        by_parent.setdefault(parent, []).append(cluster_name)
    return by_parent


def _validate_compound_tree(
    clusters: Mapping[str, Any],
    cluster_parents: Optional[Mapping[str, Optional[str]]],
) -> bool:
    """Return whether the declared cluster parent graph is a tree forest.

    Parameters
    ----------
    clusters : mapping[str, Any]
        Declared clusters.
    cluster_parents : mapping[str, str | None] | None
        Optional cluster parent map.

    Returns
    -------
    bool
        ``False`` when a parent cycle is detected.
    """
    parents = cluster_parents or {}
    for cluster_name in clusters:
        seen: Set[str] = set()
        parent = parents.get(str(cluster_name))
        while parent in clusters:
            if parent in seen:
                return False
            seen.add(parent)
            parent = parents.get(parent)
    return True


def _compound_tree_depths(graph: _DagreGraph) -> Dict[NodeId, int]:
    """Return dagre.js nesting depths for every compound-tree node.

    Parameters
    ----------
    graph : _DagreGraph
        Prepared graph carrying parent metadata.

    Returns
    -------
    dict[Hashable, int]
        Depth per graph-root subtree node.
    """
    depths: Dict[NodeId, int] = {}

    def visit(node: NodeId, depth: int) -> None:
        """Assign depths recursively.

        Parameters
        ----------
        node : Hashable
            Current node or cluster.
        depth : int
            Dagre nesting depth.

        Returns
        -------
        None
            ``depths`` is mutated.
        """
        children = graph.children(node)
        if children:
            for child in children:
                visit(child, depth + 1)
        depths[node] = depth

    for child in graph.children(None):
        visit(child, 1)
    return depths


def _remove_empty_compound_ranks(graph: _DagreGraph) -> None:
    """Remove non-node border ranks the way dagre.js ``removeEmptyRanks`` does.

    Parameters
    ----------
    graph : _DagreGraph
        Ranked compound graph.

    Returns
    -------
    None
        Node ranks are compacted in place.
    """
    ranked_nodes = [node for node in graph.node_order if graph.nodes[node].rank is not None]
    if not ranked_nodes:
        return
    offset = min(int(graph.nodes[node].rank or 0) for node in ranked_nodes)
    layers: Dict[int, List[NodeId]] = {}
    for node in ranked_nodes:
        rank = int(graph.nodes[node].rank or 0) - offset
        layers.setdefault(rank, []).append(node)
    max_rank = max(layers, default=-1)
    delta = 0
    factor = max(int(graph.node_rank_factor), 1)
    for rank in range(max_rank + 1):
        layer = layers.get(rank)
        if layer is None and rank % factor != 0:
            delta -= 1
        elif delta and layer is not None:
            for node in layer:
                old_rank = graph.nodes[node].rank
                if old_rank is not None:
                    graph.nodes[node].rank = old_rank + delta


def _normalize_ranks(graph: _DagreGraph) -> None:
    """Shift all ranked nodes so the minimum rank is zero.

    Parameters
    ----------
    graph : _DagreGraph
        Ranked working graph.

    Returns
    -------
    None
        Node ranks are shifted in place.
    """
    ranks = [int(node.rank) for node in graph.nodes.values() if node.rank is not None]
    if not ranks:
        return
    minimum = min(ranks)
    for node in graph.nodes.values():
        if node.rank is not None:
            node.rank = int(node.rank - minimum)


def _require_graph(state: SolveState) -> _DagreGraph:
    """Return the prepared Dagre working graph.

    Parameters
    ----------
    state : SolveState
        Pipeline state populated by :class:`DagrePrepareGraph`.

    Returns
    -------
    _DagreGraph
        Mutable working graph.

    Raises
    ------
    RuntimeError
        If the preparation stage has not run.
    """
    graph = state.extras.get(_DAGRE_GRAPH_KEY)
    if not isinstance(graph, _DagreGraph):
        raise RuntimeError("DagrePrepareGraph must run before this stage.")
    return graph


def _validate_rankdir(rankdir: str) -> str:
    """Normalize and validate a Dagre rank direction.

    Parameters
    ----------
    rankdir : str
        Requested direction.

    Returns
    -------
    str
        Uppercase direction.

    Raises
    ------
    ValueError
        If the direction is unsupported.
    """
    normalized = rankdir.upper()
    if normalized not in {"TB", "BT", "LR", "RL"}:
        raise ValueError("rankdir must be one of TB, BT, LR, or RL.")
    return normalized


def _validate_ranker(ranker: str) -> str:
    """Normalize and validate a Dagre ranker.

    Parameters
    ----------
    ranker : str
        Requested ranking algorithm.

    Returns
    -------
    str
        Lowercase ranker name.

    Raises
    ------
    ValueError
        If the ranker is unsupported.
    """
    normalized = ranker.lower()
    if normalized not in {"network-simplex", "tight-tree", "longest-path"}:
        raise ValueError("ranker must be network-simplex, tight-tree, or longest-path.")
    return normalized


def _validate_acyclicer(acyclicer: str) -> str:
    """Normalize and validate a Dagre acyclicer.

    Parameters
    ----------
    acyclicer : str
        Requested feedback-arc heuristic.

    Returns
    -------
    str
        Lowercase acyclicer name.

    Raises
    ------
    ValueError
        If the acyclicer is unsupported.
    """
    normalized = acyclicer.lower()
    if normalized not in {"dfs", "greedy"}:
        raise ValueError("acyclicer must be dfs or greedy.")
    return normalized


@register_op
class DagrePrepareGraph(Op):
    """Validate tensor inputs and create Dagre's mutable layout graph."""

    name: ClassVar[str] = "dagre_prepare_graph"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def __init__(
        self,
        rank_sep: float = 50.0,
        node_sep: float = 50.0,
        edge_sep: float = 20.0,
        rankdir: str = "TB",
        ranker: str = "network-simplex",
        acyclicer: str = "dfs",
    ) -> None:
        """Store canonical Dagre graph options.

        Parameters
        ----------
        rank_sep : float, default=50.0
            Gap between adjacent rank boxes.
        node_sep : float, default=50.0
            Gap between adjacent real nodes.
        edge_sep : float, default=20.0
            Gap contributed by adjacent dummy edge nodes.
        rankdir : str, default="TB"
            Layout direction.
        ranker : str, default="network-simplex"
            Rank assignment variant.
        acyclicer : str, default="dfs"
            Feedback-arc heuristic.

        Returns
        -------
        None
            Validated options are stored on the op.
        """
        if rank_sep < 0.0 or node_sep < 0.0 or edge_sep < 0.0:
            raise ValueError("Dagre separation values must be non-negative.")
        self.rank_sep = float(rank_sep)
        self.node_sep = float(node_sep)
        self.edge_sep = float(edge_sep)
        self.rankdir = _validate_rankdir(rankdir)
        self.ranker = _validate_ranker(ranker)
        self.acyclicer = _validate_acyclicer(acyclicer)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build the local multigraph and apply edge-label rank scaling.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable topology and node-size inputs.
        state : SolveState
            Mutable state receiving the working graph.
        ctx : RuntimeContext
            Runtime infrastructure; unused by this CPU reference port.

        Returns
        -------
        SolveState
            State containing ``dagre_graph``.
        """
        del ctx
        edge_index = problem.edge_index.detach().to(device="cpu", dtype=torch.long)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E].")
        if problem.num_nodes < 0:
            raise ValueError("num_nodes must be non-negative.")
        if edge_index.numel() and (
            int(edge_index.min().item()) < 0 or int(edge_index.max().item()) >= problem.num_nodes
        ):
            raise ValueError("edge_index contains an out-of-range node id.")

        if problem.node_sizes is None:
            sizes = torch.zeros((problem.num_nodes, 2), dtype=torch.float64)
        else:
            sizes = problem.node_sizes.detach().to(device="cpu", dtype=torch.float64)
            if sizes.shape != (problem.num_nodes, 2):
                if problem.num_nodes == 0 and sizes.numel() == 0:
                    # Degenerate empty graphs may carry a 0-element size
                    # tensor of any shape (e.g. ``[0]``).
                    sizes = sizes.reshape(0, 2)
                else:
                    raise ValueError("node_sizes must have shape [N, 2].")
        weights = (
            torch.ones(edge_index.shape[1], dtype=torch.float64)
            if problem.edge_weights is None
            else problem.edge_weights.detach().to(device="cpu", dtype=torch.float64)
        )
        if weights.shape != (edge_index.shape[1],):
            raise ValueError("edge_weights must have shape [E].")

        normalized_clusters: Dict[str, Any] = {
            str(name): members for name, members in (problem.clusters or {}).items()
        }
        normalized_parents: Dict[str, Optional[str]] = {
            str(name): None if parent is None else str(parent)
            for name, parent in (problem.cluster_parents or {}).items()
        }
        if not _validate_compound_tree(normalized_clusters, normalized_parents):
            normalized_clusters = {}
            normalized_parents = {}
        compound_node_order = [
            _cluster_node_id(cluster_name) for cluster_name in sorted(normalized_clusters)
        ]
        original_node_ids: List[NodeId] = (
            [str(node) for node in range(problem.num_nodes)]
            if normalized_clusters
            else list(range(problem.num_nodes))
        )
        nodes = {node: _DagreNode(width=0.0, height=0.0) for node in compound_node_order}
        nodes.update(
            {
                original_node_ids[node]: _DagreNode(
                    width=float(sizes[node, 0]),
                    height=float(sizes[node, 1]),
                )
                for node in range(problem.num_nodes)
            }
        )
        graph = _DagreGraph(
            nodes=nodes,
            node_order=[*compound_node_order, *original_node_ids],
            edges=[],
            num_original_nodes=problem.num_nodes,
            original_node_ids=original_node_ids,
            # Dagre always halves ranksep and doubles minlen to reserve the
            # half-ranks used by potential edge labels, even when labels are empty.
            rank_sep=self.rank_sep / 2.0,
            node_sep=self.node_sep,
            edge_sep=self.edge_sep,
            rankdir=self.rankdir,
            ranker=self.ranker,
            acyclicer=self.acyclicer,
        )
        if normalized_clusters:
            children_by_parent = _cluster_children_by_parent(
                normalized_clusters,
                normalized_parents,
            )
            emitted_nodes: Set[int] = set()

            def emit_cluster(cluster_name: str) -> None:
                """Assign direct dagre parents for one cluster subtree.

                Parameters
                ----------
                cluster_name : str
                    Cluster being emitted.

                Returns
                -------
                None
                    Parent metadata on ``graph`` is mutated.
                """
                cluster_id = _cluster_node_id(cluster_name)
                parent_name = normalized_parents.get(cluster_name)
                if parent_name in normalized_clusters:
                    graph.set_parent(cluster_id, _cluster_node_id(parent_name))
                child_clusters = children_by_parent.get(cluster_name, [])
                for child_name in child_clusters:
                    emit_cluster(child_name)

                descendant_members: Set[int] = set()
                for child_name in child_clusters:
                    descendant_members.update(
                        index
                        for index in _flatten_cluster_members(normalized_clusters[child_name])
                        if 0 <= index < problem.num_nodes
                    )
                for node_index in _flatten_cluster_members(normalized_clusters[cluster_name]):
                    if (
                        node_index in descendant_members
                        or node_index in emitted_nodes
                        or node_index < 0
                        or node_index >= problem.num_nodes
                    ):
                        continue
                    graph.set_parent(original_node_ids[node_index], cluster_id)
                    emitted_nodes.add(node_index)

            for root_cluster in children_by_parent.get(None, []):
                emit_cluster(root_cluster)
        edges_by_pair: Dict[Tuple[NodeId, NodeId], _DagreEdge] = {}
        for edge_index_value, (source, target) in enumerate(
            zip(edge_index[0].tolist(), edge_index[1].tolist())
        ):
            pair = (original_node_ids[int(source)], original_node_ids[int(target)])
            edge = edges_by_pair.get(pair)
            if edge is None:
                edge = graph.add_edge(
                    source=pair[0],
                    target=pair[1],
                    weight=float(weights[edge_index_value]),
                    minlen=2,
                    original_index=edge_index_value,
                )
                edges_by_pair[pair] = edge
            else:
                # The canonical adapter builds a non-multigraph graphlib.Graph;
                # setEdge on an existing pair preserves its insertion slot and
                # replaces the label with the last edge's values.
                edge.weight = float(weights[edge_index_value])
                edge.original_index = edge_index_value
            if source == target:
                edge.active = False
                self_edges = graph.self_edges.setdefault(int(source), [])
                if edge not in self_edges:
                    self_edges.append(edge)
        state.extras[_DAGRE_GRAPH_KEY] = graph
        return state


def _dfs_feedback_edges(graph: _DagreGraph) -> List[_DagreEdge]:
    """Return Dagre's DFS feedback arc set.

    Parameters
    ----------
    graph : _DagreGraph
        Prepared directed multigraph.

    Returns
    -------
    list[_DagreEdge]
        Back edges in traversal order.
    """
    feedback: List[_DagreEdge] = []
    visited: Set[NodeId] = set()
    stack: Set[NodeId] = set()

    # Iterative twin of dagre's recursive dfsFAS visit (suspended-iterator
    # stack): preserves the exact traversal order, the on-stack back-edge
    # test, and the feedback collection order while staying depth-safe on
    # path-like graphs.
    frames: List[Tuple[NodeId, Iterator[_DagreEdge]]] = []
    for start in graph.node_order:
        if start in visited:
            continue
        visited.add(start)
        stack.add(start)
        frames.append((start, iter(graph.out_edges(start))))
        while frames:
            node, edge_iter = frames[-1]
            descended = False
            for edge in edge_iter:
                if edge.target in stack:
                    feedback.append(edge)
                elif edge.target not in visited:
                    visited.add(edge.target)
                    stack.add(edge.target)
                    frames.append((edge.target, iter(graph.out_edges(edge.target))))
                    descended = True
                    break
            if not descended:
                stack.remove(node)
                frames.pop()
    return feedback


def _greedy_feedback_pairs(graph: _DagreGraph) -> List[Tuple[NodeId, NodeId]]:
    """Return weighted Eades feedback pairs matching dagre's greedy FAS.

    Parameters
    ----------
    graph : _DagreGraph
        Prepared directed multigraph.

    Returns
    -------
    list[tuple[Hashable, Hashable]]
        Simplified feedback pairs in removal order.
    """
    pair_weights: Dict[Tuple[NodeId, NodeId], float] = {}
    for edge in graph.active_edges():
        pair = (edge.source, edge.target)
        pair_weights[pair] = pair_weights.get(pair, 0.0) + edge.weight
    active_nodes: Set[NodeId] = set(graph.node_order)
    incoming: Dict[NodeId, float] = {node: 0.0 for node in graph.node_order}
    outgoing: Dict[NodeId, float] = {node: 0.0 for node in graph.node_order}
    for (source, target), weight in pair_weights.items():
        outgoing[source] += weight
        incoming[target] += weight
    feedback: List[Tuple[NodeId, NodeId]] = []

    def remove_node(node: NodeId, collect: bool) -> None:
        """Remove one FAS node and update weighted degrees.

        Parameters
        ----------
        node : Hashable
            Node to remove.
        collect : bool
            Whether incoming pairs belong to the feedback set.

        Returns
        -------
        None
            Local graph state is mutated.
        """
        if collect:
            for pair in pair_weights:
                if pair[1] == node and pair[0] in active_nodes:
                    feedback.append(pair)
        for (source, target), weight in pair_weights.items():
            if target == node and source in active_nodes:
                outgoing[source] -= weight
            if source == node and target in active_nodes:
                incoming[target] -= weight
        active_nodes.remove(node)

    while active_nodes:
        changed = True
        while changed:
            changed = False
            for node in graph.node_order:
                if node in active_nodes and outgoing[node] == 0.0:
                    remove_node(node, collect=False)
                    changed = True
                    break
        changed = True
        while changed:
            changed = False
            for node in graph.node_order:
                if node in active_nodes and incoming[node] == 0.0:
                    remove_node(node, collect=False)
                    changed = True
                    break
        if active_nodes:
            node = max(
                (candidate for candidate in graph.node_order if candidate in active_nodes),
                key=lambda candidate: outgoing[candidate] - incoming[candidate],
            )
            remove_node(node, collect=True)
    return feedback


@register_op
class DagreMakeAcyclic(Op):
    """Reverse a deterministic feedback arc set."""

    name: ClassVar[str] = "dagre_make_acyclic"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run the configured DFS or greedy acyclicer.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable inputs; unused after preparation.
        state : SolveState
            State holding the working graph.
        ctx : RuntimeContext
            Runtime infrastructure; unused.

        Returns
        -------
        SolveState
            State whose active graph is acyclic.
        """
        del problem, ctx
        graph = _require_graph(state)
        if graph.acyclicer == "greedy":
            pairs = set(_greedy_feedback_pairs(graph))
            feedback = [
                edge for edge in graph.active_edges() if (edge.source, edge.target) in pairs
            ]
        else:
            feedback = _dfs_feedback_edges(graph)
        for edge in feedback:
            edge.active = False
            graph.add_edge(
                source=edge.target,
                target=edge.source,
                weight=edge.weight,
                minlen=edge.minlen,
                original_index=edge.original_index,
                reversed_edge=True,
            )
        return state


@register_op
class DagreNestingGraph(Op):
    """Add dagre.js compound nesting nodes and ranking constraints."""

    name: ClassVar[str] = "dagre_nesting_graph"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run dagre.js ``nesting-graph.run`` for compound inputs.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable inputs; unused after graph preparation.
        state : SolveState
            State holding an acyclic compound graph.
        ctx : RuntimeContext
            Runtime infrastructure; unused.

        Returns
        -------
        SolveState
            State whose working graph has temporary nesting constraints.
        """
        del problem, ctx
        graph = _require_graph(state)
        if not graph.has_compound():
            return state
        root = graph.add_dummy("root")
        depths = _compound_tree_depths(graph)
        height = max(depths.values(), default=1) - 1
        node_sep = 2 * height + 1
        graph.nesting_root = root
        graph.node_rank_factor = node_sep
        for edge in graph.active_edges():
            edge.minlen *= node_sep
        weight = sum(edge.weight for edge in graph.active_edges()) + 1.0

        def visit(node: NodeId) -> None:
            """Create border nodes and nesting edges for one subtree.

            Parameters
            ----------
            node : Hashable
                Current compound-tree child.

            Returns
            -------
            None
                The working graph is mutated.
            """
            children = graph.children(node)
            if not children:
                if node != root:
                    graph.add_edge(root, node, weight=0.0, minlen=node_sep, original_index=-1)
                return

            top = graph.add_dummy("border")
            bottom = graph.add_dummy("border")
            label = graph.nodes[node]
            graph.set_parent(top, node)
            graph.set_parent(bottom, node)
            label.border_top = top
            label.border_bottom = bottom

            for child in children:
                visit(child)
                child_node = graph.nodes[child]
                child_top = child_node.border_top if child_node.border_top is not None else child
                child_bottom = (
                    child_node.border_bottom if child_node.border_bottom is not None else child
                )
                edge_weight = weight if child_node.border_top is not None else 2.0 * weight
                minlen = 1 if child_top != child_bottom else height - depths.get(node, 1) + 1
                graph.add_edge(
                    top,
                    child_top,
                    weight=edge_weight,
                    minlen=minlen,
                    original_index=-1,
                    nesting_edge=True,
                )
                graph.add_edge(
                    child_bottom,
                    bottom,
                    weight=edge_weight,
                    minlen=minlen,
                    original_index=-1,
                    nesting_edge=True,
                )

            if graph.parent_of(node) is None:
                graph.add_edge(
                    root,
                    top,
                    weight=0.0,
                    minlen=height + depths.get(node, 1),
                    original_index=-1,
                )

        for child in graph.children(None):
            if child != root:
                visit(child)
        return state


def _simplified_rank_edges(
    graph: _DagreGraph,
    allowed_nodes: Optional[Set[NodeId]] = None,
) -> List[Tuple[NodeId, NodeId, int, int]]:
    """Aggregate multiedges for Dagre's rank stage.

    Parameters
    ----------
    graph : _DagreGraph
        Acyclic working graph.
    allowed_nodes : set[Hashable] | None, optional
        Optional node filter matching dagre's non-compound rank view.

    Returns
    -------
    list[tuple[Hashable, Hashable, int, int]]
        Pair edges with summed integer weights and maximum ``minlen``.
    """
    records: Dict[Tuple[NodeId, NodeId], Tuple[int, int]] = {}
    for edge in graph.active_edges():
        if allowed_nodes is not None and (
            edge.source not in allowed_nodes or edge.target not in allowed_nodes
        ):
            continue
        pair = (edge.source, edge.target)
        old_weight, old_minlen = records.get(pair, (0, 1))
        records[pair] = (old_weight + int(edge.weight), max(old_minlen, edge.minlen))
    return [
        (source, target, minlen, weight) for (source, target), (weight, minlen) in records.items()
    ]


def _longest_path_ranks(
    node_order: Sequence[NodeId],
    edges: Sequence[Tuple[NodeId, NodeId, int, int]],
) -> Dict[NodeId, int]:
    """Assign Dagre's sink-anchored longest-path ranks.

    Parameters
    ----------
    node_order : sequence[Hashable]
        Graph node insertion order.
    edges : sequence[tuple]
        Simplified ``(tail, head, minlen, weight)`` records.

    Returns
    -------
    dict[Hashable, int]
        Unnormalized non-positive ranks.
    """
    outgoing: Dict[NodeId, List[Tuple[NodeId, int]]] = {node: [] for node in node_order}
    incoming_count: Dict[NodeId, int] = {node: 0 for node in node_order}
    for source, target, minlen, _weight in edges:
        outgoing[source].append((target, minlen))
        incoming_count[target] += 1
    ranks: Dict[NodeId, int] = {}

    # Iterative twin of dagre's recursive longestPath dfs: preserves the
    # provisional-zero memo visible to in-flight revisits, the candidate
    # evaluation order, and the ranks-dict insertion (discovery) order while
    # staying depth-safe on long chains. A frame carries the node, its
    # suspended outgoing iterator, the candidate ranks collected so far, and
    # the minlen owed to the parent frame when this node's rank finalizes.
    for source in node_order:
        if incoming_count[source] != 0 or source in ranks:
            continue
        ranks[source] = 0
        frames: List[Tuple[NodeId, Iterator[Tuple[NodeId, int]], List[int], int]] = [
            (source, iter(outgoing[source]), [], 0)
        ]
        while frames:
            node, targets, candidates, owed_minlen = frames[-1]
            descended = False
            for target, minlen in targets:
                if target in ranks:
                    candidates.append(ranks[target] - minlen)
                else:
                    ranks[target] = 0
                    frames.append((target, iter(outgoing[target]), [], minlen))
                    descended = True
                    break
            if descended:
                continue
            ranks[node] = min(candidates) if candidates else 0
            frames.pop()
            if frames:
                frames[-1][2].append(ranks[node] - owed_minlen)
    return ranks


def _tight_tree_ranks(
    node_order: Sequence[NodeId],
    edges: Sequence[Tuple[NodeId, NodeId, int, int]],
) -> Dict[NodeId, int]:
    """Run Dagre's feasible-tight-tree ranker.

    Parameters
    ----------
    node_order : sequence[Hashable]
        Graph node insertion order.
    edges : sequence[tuple]
        Simplified rank edges.

    Returns
    -------
    dict[Hashable, int]
        Feasible tight-tree ranks.
    """
    ranks = _longest_path_ranks(node_order, edges)
    if not node_order:
        return ranks
    tree_nodes: Set[NodeId] = {node_order[0]}

    def slack(edge: Tuple[NodeId, NodeId, int, int]) -> int:
        """Return rank slack for one edge.

        Parameters
        ----------
        edge : tuple
            Simplified rank edge.

        Returns
        -------
        int
            Current edge slack.
        """
        source, target, minlen, _weight = edge
        return ranks[target] - ranks[source] - minlen

    while len(tree_nodes) < len(node_order):
        changed = True
        while changed:
            changed = False
            for source, target, _minlen, _weight in edges:
                if slack((source, target, _minlen, _weight)) != 0:
                    continue
                if source in tree_nodes and target not in tree_nodes:
                    tree_nodes.add(target)
                    changed = True
                elif target in tree_nodes and source not in tree_nodes:
                    tree_nodes.add(source)
                    changed = True
        if len(tree_nodes) == len(node_order):
            break
        crossing_edges = [
            edge for edge in edges if (edge[0] in tree_nodes) != (edge[1] in tree_nodes)
        ]
        edge = min(crossing_edges, key=slack)
        delta = slack(edge) if edge[0] in tree_nodes else -slack(edge)
        for node in tree_nodes:
            ranks[node] += delta
    return ranks


@dataclass
class _DagreRankTreeNode:
    """Mutable Dagre network-simplex tree-node label."""

    low: int = 0
    lim: int = 0
    parent: Optional[NodeId] = None


@dataclass
class _DagreRankTreeEdge:
    """Mutable undirected tight-tree edge label."""

    left: NodeId
    right: NodeId
    cut_value: float = 0.0
    active: bool = True


@dataclass
class _DagreRankTree:
    """Undirected tight tree used by Dagre's network simplex."""

    graph_node_order: Sequence[NodeId]
    nodes: Dict[NodeId, _DagreRankTreeNode] = field(default_factory=dict)
    edges: List[_DagreRankTreeEdge] = field(default_factory=list)

    def add_node(self, node: NodeId) -> None:
        """Add one tree node if absent.

        Parameters
        ----------
        node : Hashable
            Rank-graph node id.

        Returns
        -------
        None
            Tree state is mutated.
        """
        self.nodes.setdefault(node, _DagreRankTreeNode())

    def ordered_nodes(self) -> List[NodeId]:
        """Return Graphlib-compatible node-key order.

        Returns
        -------
        list[Hashable]
            Tree nodes in the parent rank graph's key order.
        """
        return [node for node in self.graph_node_order if node in self.nodes]

    def add_edge(self, left: NodeId, right: NodeId) -> None:
        """Append an undirected edge unless it already exists.

        Parameters
        ----------
        left : Hashable
            First endpoint.
        right : Hashable
            Second endpoint.

        Returns
        -------
        None
            Tree state is mutated.
        """
        if self.edge(left, right) is not None:
            return
        self.add_node(left)
        self.add_node(right)
        canonical_left, canonical_right = _canonical_rank_pair(left, right)
        self.edges.append(_DagreRankTreeEdge(canonical_left, canonical_right))

    def edge(self, left: NodeId, right: NodeId) -> Optional[_DagreRankTreeEdge]:
        """Return an active undirected edge between two nodes.

        Parameters
        ----------
        left : Hashable
            First endpoint.
        right : Hashable
            Second endpoint.

        Returns
        -------
        _DagreRankTreeEdge | None
            Matching edge when present.
        """
        pair = frozenset((left, right))
        for edge in self.edges:
            if edge.active and frozenset((edge.left, edge.right)) == pair:
                return edge
        return None

    def remove_edge(self, edge: _DagreRankTreeEdge) -> None:
        """Remove one tree edge.

        Parameters
        ----------
        edge : _DagreRankTreeEdge
            Active edge to remove.

        Returns
        -------
        None
            The edge is marked inactive.
        """
        edge.active = False

    def active_edges(self) -> List[_DagreRankTreeEdge]:
        """Return active tree edges in insertion order.

        Returns
        -------
        list[_DagreRankTreeEdge]
            Active tree edges.
        """
        return [edge for edge in self.edges if edge.active]

    def neighbors(self, node: NodeId) -> List[NodeId]:
        """Return Graphlib undirected neighbors.

        Parameters
        ----------
        node : Hashable
            Tree node id.

        Returns
        -------
        list[Hashable]
            Canonical predecessors followed by canonical successors.
        """
        predecessors = [edge.left for edge in self.active_edges() if edge.right == node]
        successors = [edge.right for edge in self.active_edges() if edge.left == node]
        return _graphlib_key_order(predecessors) + _graphlib_key_order(successors)


def _graphlib_string(node: NodeId) -> str:
    """Return the JavaScript Graphlib string key for an internal node.

    Parameters
    ----------
    node : Hashable
        Internal rank node id.

    Returns
    -------
    str
        Graphlib-style key string.
    """
    if isinstance(node, int):
        return str(node)
    if isinstance(node, tuple) and node and node[0] == "root":
        return f"_root{node[1]}"
    return str(node)


def _canonical_rank_pair(left: NodeId, right: NodeId) -> Tuple[NodeId, NodeId]:
    """Canonicalize an undirected Graphlib edge pair.

    Parameters
    ----------
    left : Hashable
        First endpoint.
    right : Hashable
        Second endpoint.

    Returns
    -------
    tuple[Hashable, Hashable]
        Lexicographically ordered Graphlib string endpoints.
    """
    if _graphlib_string(left) > _graphlib_string(right):
        return right, left
    return left, right


def _is_js_array_index(value: str) -> bool:
    """Return whether a key receives JavaScript integer-key ordering.

    Parameters
    ----------
    value : str
        Object key string.

    Returns
    -------
    bool
        ``True`` for canonical non-negative 32-bit array indices.
    """
    if not value.isdigit():
        return False
    integer = int(value)
    return 0 <= integer < 2**32 - 1 and str(integer) == value


def _graphlib_key_order(nodes: Sequence[NodeId]) -> List[NodeId]:
    """Apply JavaScript object-key ordering to unique node ids.

    Parameters
    ----------
    nodes : sequence[Hashable]
        Node ids in property insertion order.

    Returns
    -------
    list[Hashable]
        Integer-like keys numerically sorted before other insertion-ordered keys.
    """
    unique_nodes = _unique(nodes)
    integer_nodes = [node for node in unique_nodes if _is_js_array_index(_graphlib_string(node))]
    other_nodes = [node for node in unique_nodes if node not in integer_nodes]
    integer_nodes.sort(key=lambda node: int(_graphlib_string(node)))
    return integer_nodes + other_nodes


def _rank_in_edges(
    node: NodeId,
    edges: Sequence[Tuple[NodeId, NodeId, int, int]],
) -> List[Tuple[NodeId, NodeId, int, int]]:
    """Return incoming simplified rank edges.

    Parameters
    ----------
    node : Hashable
        Target node.
    edges : sequence[tuple]
        Rank edges in insertion order.

    Returns
    -------
    list[tuple]
        Incoming records.
    """
    return [edge for edge in edges if edge[1] == node]


def _rank_out_edges(
    node: NodeId,
    edges: Sequence[Tuple[NodeId, NodeId, int, int]],
) -> List[Tuple[NodeId, NodeId, int, int]]:
    """Return outgoing simplified rank edges.

    Parameters
    ----------
    node : Hashable
        Source node.
    edges : sequence[tuple]
        Rank edges in insertion order.

    Returns
    -------
    list[tuple]
        Outgoing records.
    """
    return [edge for edge in edges if edge[0] == node]


def _rank_node_edges(
    node: NodeId,
    edges: Sequence[Tuple[NodeId, NodeId, int, int]],
) -> List[Tuple[NodeId, NodeId, int, int]]:
    """Return incoming then outgoing rank edges like Graphlib.

    Parameters
    ----------
    node : Hashable
        Incident node.
    edges : sequence[tuple]
        Rank edges in insertion order.

    Returns
    -------
    list[tuple]
        Incident edge records.
    """
    return _rank_in_edges(node, edges) + _rank_out_edges(node, edges)


def _rank_edge_between(
    left: NodeId,
    right: NodeId,
    edges: Sequence[Tuple[NodeId, NodeId, int, int]],
) -> Optional[Tuple[NodeId, NodeId, int, int]]:
    """Return the directed rank edge from ``left`` to ``right``.

    Parameters
    ----------
    left : Hashable
        Candidate source.
    right : Hashable
        Candidate target.
    edges : sequence[tuple]
        Simplified rank edges.

    Returns
    -------
    tuple | None
        Directed edge record when present.
    """
    return next((edge for edge in edges if edge[0] == left and edge[1] == right), None)


def _rank_slack(
    edge: Tuple[NodeId, NodeId, int, int],
    ranks: Mapping[NodeId, int],
) -> int:
    """Return Dagre rank slack for one edge.

    Parameters
    ----------
    edge : tuple
        ``(tail, head, minlen, weight)`` record.
    ranks : mapping[Hashable, int]
        Current feasible ranks.

    Returns
    -------
    int
        ``rank(head) - rank(tail) - minlen``.
    """
    source, target, minlen, _weight = edge
    return ranks[target] - ranks[source] - minlen


def _dagre_tight_tree(
    tree: _DagreRankTree,
    edges: Sequence[Tuple[NodeId, NodeId, int, int]],
    ranks: Mapping[NodeId, int],
) -> int:
    """Grow a maximal tree over currently tight rank edges.

    Parameters
    ----------
    tree : _DagreRankTree
        Partial tight tree.
    edges : sequence[tuple]
        Simplified rank edges.
    ranks : mapping[Hashable, int]
        Current feasible ranks.

    Returns
    -------
    int
        Number of tree nodes after growth.
    """

    def visit(node: NodeId) -> None:
        """Recursively add unseen nodes on tight incident edges.

        Parameters
        ----------
        node : Hashable
            Current tree node.

        Returns
        -------
        None
            The tree is mutated.
        """
        for edge in _rank_node_edges(node, edges):
            other = edge[1] if edge[0] == node else edge[0]
            if other not in tree.nodes and _rank_slack(edge, ranks) == 0:
                tree.add_node(other)
                tree.add_edge(node, other)
                visit(other)

    for node in tree.ordered_nodes():
        visit(node)
    return len(tree.nodes)


def _dagre_feasible_tree(
    node_order: Sequence[NodeId],
    edges: Sequence[Tuple[NodeId, NodeId, int, int]],
    ranks: Dict[NodeId, int],
) -> _DagreRankTree:
    """Construct Dagre's feasible tight spanning tree.

    Parameters
    ----------
    node_order : sequence[Hashable]
        Graphlib node-key order.
    edges : sequence[tuple]
        Simplified rank edges.
    ranks : dict[Hashable, int]
        Mutable feasible ranks.

    Returns
    -------
    _DagreRankTree
        Tight spanning tree.
    """
    tree = _DagreRankTree(graph_node_order=node_order)
    if not node_order:
        return tree
    tree.add_node(node_order[0])
    while _dagre_tight_tree(tree, edges, ranks) < len(node_order):
        crossing = [edge for edge in edges if (edge[0] in tree.nodes) != (edge[1] in tree.nodes)]
        selected = min(crossing, key=lambda edge: _rank_slack(edge, ranks))
        delta = (
            _rank_slack(selected, ranks)
            if selected[0] in tree.nodes
            else -_rank_slack(selected, ranks)
        )
        for node in tree.ordered_nodes():
            ranks[node] += delta
    return tree


def _dagre_init_low_lim(tree: _DagreRankTree, root: Optional[NodeId] = None) -> None:
    """Assign DFS low/lim intervals and parents on the tight tree.

    Parameters
    ----------
    tree : _DagreRankTree
        Tight spanning tree.
    root : Hashable | None, optional
        DFS root; defaults to Graphlib's first tree node.

    Returns
    -------
    None
        Tree-node labels are mutated.
    """
    if not tree.nodes:
        return
    resolved_root = tree.ordered_nodes()[0] if root is None else root
    visited: Set[NodeId] = set()

    def visit(node: NodeId, next_lim: int, parent: Optional[NodeId]) -> int:
        """Assign one DFS subtree interval.

        Parameters
        ----------
        node : Hashable
            Current tree node.
        next_lim : int
            Next postorder counter.
        parent : Hashable | None
            DFS parent.

        Returns
        -------
        int
            Next unused counter after this subtree.
        """
        low = next_lim
        visited.add(node)
        for neighbor in tree.neighbors(node):
            if neighbor not in visited:
                next_lim = visit(neighbor, next_lim, node)
        label = tree.nodes[node]
        label.low = low
        label.lim = next_lim
        label.parent = parent
        return next_lim + 1

    visit(resolved_root, 1, None)


def _dagre_tree_postorder(tree: _DagreRankTree, root: NodeId) -> List[NodeId]:
    """Return Graphlib DFS postorder from a connected tree root.

    Parameters
    ----------
    tree : _DagreRankTree
        Tight tree.
    root : Hashable
        Traversal root.

    Returns
    -------
    list[Hashable]
        Postorder node ids.
    """
    visited: Set[NodeId] = set()
    output: List[NodeId] = []

    def visit(node: NodeId) -> None:
        """Visit one postorder subtree.

        Parameters
        ----------
        node : Hashable
            Current tree node.

        Returns
        -------
        None
            Traversal output is mutated.
        """
        if node in visited:
            return
        visited.add(node)
        for neighbor in tree.neighbors(node):
            visit(neighbor)
        output.append(node)

    visit(root)
    return output


def _dagre_calc_cut_value(
    tree: _DagreRankTree,
    edges: Sequence[Tuple[NodeId, NodeId, int, int]],
    child: NodeId,
) -> float:
    """Calculate one Dagre network-simplex tree-edge cut value.

    Parameters
    ----------
    tree : _DagreRankTree
        Tight tree with child-parent labels.
    edges : sequence[tuple]
        Simplified directed rank edges.
    child : Hashable
        Child endpoint of the tree edge.

    Returns
    -------
    float
        Cut value used to select leaving edges.
    """
    parent = tree.nodes[child].parent
    if parent is None:
        raise RuntimeError("Cannot calculate a cut value for the tree root.")
    graph_edge = _rank_edge_between(child, parent, edges)
    child_is_tail = graph_edge is not None
    if graph_edge is None:
        graph_edge = _rank_edge_between(parent, child, edges)
    if graph_edge is None:
        raise RuntimeError("Tight-tree edge is absent from the rank graph.")
    cut_value = float(graph_edge[3])
    for edge in _rank_node_edges(child, edges):
        is_out_edge = edge[0] == child
        other = edge[1] if is_out_edge else edge[0]
        if other == parent:
            continue
        points_to_head = is_out_edge == child_is_tail
        other_weight = float(edge[3])
        cut_value += other_weight if points_to_head else -other_weight
        tree_edge = tree.edge(child, other)
        if tree_edge is not None:
            cut_value += -tree_edge.cut_value if points_to_head else tree_edge.cut_value
    return cut_value


def _dagre_init_cut_values(
    tree: _DagreRankTree,
    edges: Sequence[Tuple[NodeId, NodeId, int, int]],
) -> None:
    """Initialize all tight-tree cut values in postorder.

    Parameters
    ----------
    tree : _DagreRankTree
        Tight tree with low/lim labels.
    edges : sequence[tuple]
        Simplified directed rank edges.

    Returns
    -------
    None
        Tree-edge cut values are mutated.
    """
    if not tree.nodes:
        return
    root = tree.ordered_nodes()[0]
    for child in _dagre_tree_postorder(tree, root)[:-1]:
        parent = tree.nodes[child].parent
        if parent is None:
            continue
        tree_edge = tree.edge(child, parent)
        if tree_edge is None:
            raise RuntimeError("Dagre tight tree lost a child-parent edge.")
        tree_edge.cut_value = _dagre_calc_cut_value(tree, edges, child)


def _dagre_is_descendant(
    node_label: _DagreRankTreeNode,
    root_label: _DagreRankTreeNode,
) -> bool:
    """Return whether one low/lim label lies in another subtree.

    Parameters
    ----------
    node_label : _DagreRankTreeNode
        Candidate descendant label.
    root_label : _DagreRankTreeNode
        Candidate subtree-root label.

    Returns
    -------
    bool
        Subtree membership result.
    """
    return root_label.low <= node_label.lim <= root_label.lim


def _dagre_enter_edge(
    tree: _DagreRankTree,
    edges: Sequence[Tuple[NodeId, NodeId, int, int]],
    leaving: _DagreRankTreeEdge,
    ranks: Mapping[NodeId, int],
) -> Tuple[NodeId, NodeId, int, int]:
    """Select the minimum-slack entering edge for one exchange.

    Parameters
    ----------
    tree : _DagreRankTree
        Current tight tree.
    edges : sequence[tuple]
        Simplified directed rank edges.
    leaving : _DagreRankTreeEdge
        Negative-cut tree edge to remove.
    ranks : mapping[Hashable, int]
        Current feasible ranks.

    Returns
    -------
    tuple
        Entering simplified edge.
    """
    left = leaving.left
    right = leaving.right
    if _rank_edge_between(left, right, edges) is None:
        left, right = right, left
    left_label = tree.nodes[left]
    right_label = tree.nodes[right]
    tail_label = left_label
    flip = False
    if left_label.lim > right_label.lim:
        tail_label = right_label
        flip = True
    candidates = [
        edge
        for edge in edges
        if flip == _dagre_is_descendant(tree.nodes[edge[0]], tail_label)
        and flip != _dagre_is_descendant(tree.nodes[edge[1]], tail_label)
    ]
    return min(candidates, key=lambda edge: _rank_slack(edge, ranks))


def _dagre_tree_preorder(tree: _DagreRankTree, root: NodeId) -> List[NodeId]:
    """Return Graphlib DFS preorder from a connected tree root.

    Parameters
    ----------
    tree : _DagreRankTree
        Tight tree.
    root : Hashable
        Traversal root.

    Returns
    -------
    list[Hashable]
        Preorder node ids.
    """
    visited: Set[NodeId] = set()
    output: List[NodeId] = []

    def visit(node: NodeId) -> None:
        """Visit one preorder subtree.

        Parameters
        ----------
        node : Hashable
            Current tree node.

        Returns
        -------
        None
            Traversal output is mutated.
        """
        if node in visited:
            return
        visited.add(node)
        output.append(node)
        for neighbor in tree.neighbors(node):
            visit(neighbor)

    visit(root)
    return output


def _dagre_update_ranks(
    tree: _DagreRankTree,
    edges: Sequence[Tuple[NodeId, NodeId, int, int]],
    ranks: Dict[NodeId, int],
) -> None:
    """Recompute ranks from tight-tree edge lengths after an exchange.

    Parameters
    ----------
    tree : _DagreRankTree
        Exchanged tight tree.
    edges : sequence[tuple]
        Simplified directed rank edges.
    ranks : dict[Hashable, int]
        Mutable rank mapping.

    Returns
    -------
    None
        Ranks are updated in place.
    """
    root = tree.ordered_nodes()[0]
    for node in _dagre_tree_preorder(tree, root)[1:]:
        parent = tree.nodes[node].parent
        if parent is None:
            raise RuntimeError("Dagre preorder child has no parent label.")
        edge = _rank_edge_between(node, parent, edges)
        flipped = edge is None
        if edge is None:
            edge = _rank_edge_between(parent, node, edges)
        if edge is None:
            raise RuntimeError("Dagre tree edge is absent during rank update.")
        minlen = edge[2]
        ranks[node] = ranks[parent] + (minlen if flipped else -minlen)


def _dagre_network_simplex_ranks(
    node_order: Sequence[NodeId],
    edges: Sequence[Tuple[NodeId, NodeId, int, int]],
) -> Dict[NodeId, int]:
    """Port dagre.js 0.8.5 network simplex with Graphlib tie semantics.

    Parameters
    ----------
    node_order : sequence[Hashable]
        Graphlib node-key order.
    edges : sequence[tuple]
        Simplified ``(tail, head, minlen, weight)`` records.

    Returns
    -------
    dict[Hashable, int]
        Optimized integer ranks.
    """
    ranks = _longest_path_ranks(node_order, edges)
    # The simplex helpers below (tight-tree growth, low/lim numbering, cut
    # values, preorder/postorder walks) recurse to graph-DFS depth. Raise the
    # recursion limit for the whole run and restore it afterward (scc.py /
    # _reingold_tilford.py convention); depth is bounded by the node count.
    previous_recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(previous_recursion_limit, 2 * len(node_order) + 100))
    try:
        tree = _dagre_feasible_tree(node_order, edges, ranks)
        _dagre_init_low_lim(tree)
        _dagre_init_cut_values(tree, edges)
        while True:
            leaving = next(
                (edge for edge in tree.active_edges() if edge.cut_value < 0.0),
                None,
            )
            if leaving is None:
                break
            entering = _dagre_enter_edge(tree, edges, leaving, ranks)
            tree.remove_edge(leaving)
            tree.add_edge(entering[0], entering[1])
            _dagre_init_low_lim(tree)
            _dagre_init_cut_values(tree, edges)
            _dagre_update_ranks(tree, edges, ranks)
    finally:
        sys.setrecursionlimit(previous_recursion_limit)
    return ranks


@register_op
class DagreAssignRanks(Op):
    """Assign ranks with Dagre's selectable ranker."""

    name: ClassVar[str] = "dagre_assign_ranks"
    category: ClassVar[OpCategory] = OpCategory.LAYERING
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras", "layers")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Connect the graph through a zero-weight root and rank it.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable inputs; unused after graph preparation.
        state : SolveState
            State holding the acyclic graph.
        ctx : RuntimeContext
            Runtime infrastructure; unused.

        Returns
        -------
        SolveState
            State with node ranks and an original-node rank snapshot.
        """
        del problem, ctx
        graph = _require_graph(state)
        if graph.has_compound():
            rank_node_order = graph.non_compound_node_order()
            rank_node_set = set(rank_node_order)
            rank_edges = _simplified_rank_edges(graph, rank_node_set)
        else:
            root = graph.add_dummy("root")
            rank_node_order = list(graph.node_order)
            original_edges = _simplified_rank_edges(graph)
            rank_edges = [*original_edges]
            for node in graph.node_order:
                if node != root:
                    rank_edges.append((root, node, 1, 0))

        if graph.ranker == "longest-path":
            ranks = _longest_path_ranks(rank_node_order, rank_edges)
        elif graph.ranker == "tight-tree":
            ranks = _tight_tree_ranks(rank_node_order, rank_edges)
        else:
            ranks = _dagre_network_simplex_ranks(rank_node_order, rank_edges)

        if graph.has_compound():
            for node in rank_node_order:
                graph.nodes[node].rank = int(ranks[node])
        else:
            graph.node_order.remove(root)
            del graph.nodes[root]
            minimum = min((ranks[node] for node in graph.node_order), default=0)
            for node in graph.node_order:
                graph.nodes[node].rank = int(ranks[node] - minimum)
            original_ranks = [int(graph.nodes[node].rank or 0) for node in graph.original_node_ids]
            state.extras[_DAGRE_RANKS_KEY] = original_ranks
            state.layers = torch.tensor(original_ranks, dtype=torch.long)
        return state


@register_op
class DagreCleanupNestingGraph(Op):
    """Remove temporary nesting edges and assign cluster rank spans."""

    name: ClassVar[str] = "dagre_cleanup_nesting_graph"
    category: ClassVar[OpCategory] = OpCategory.LAYERING
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras", "layers")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run dagre.js nesting cleanup, rank compaction, and span assignment.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable inputs; unused after graph preparation.
        state : SolveState
            State holding ranked compound graph.
        ctx : RuntimeContext
            Runtime infrastructure; unused.

        Returns
        -------
        SolveState
            State with normalized ranks and original-node layer snapshot.
        """
        del problem, ctx
        graph = _require_graph(state)
        if not graph.has_compound():
            return state
        _remove_empty_compound_ranks(graph)
        if graph.nesting_root is not None and graph.nesting_root in graph.nodes:
            removed_root = graph.nesting_root
            graph.node_order.remove(graph.nesting_root)
            del graph.nodes[graph.nesting_root]
            graph.parents.pop(graph.nesting_root, None)
            for edge in graph.edges:
                if edge.source == removed_root or edge.target == removed_root:
                    edge.active = False
        graph.nesting_root = None
        for edge in graph.edges:
            if edge.nesting_edge:
                edge.active = False
        _normalize_ranks(graph)
        for node in graph.node_order:
            node_data = graph.nodes[node]
            if node_data.border_top is not None and node_data.border_bottom is not None:
                top_rank = graph.nodes[node_data.border_top].rank
                bottom_rank = graph.nodes[node_data.border_bottom].rank
                if top_rank is not None and bottom_rank is not None:
                    node_data.min_rank = int(top_rank)
                    node_data.max_rank = int(bottom_rank)
        original_ranks = [int(graph.nodes[node].rank or 0) for node in graph.original_node_ids]
        state.extras[_DAGRE_RANKS_KEY] = original_ranks
        state.layers = torch.tensor(original_ranks, dtype=torch.long)
        return state


@register_op
class DagreNormalizeEdges(Op):
    """Insert a zero-size dummy node on every intermediate edge rank."""

    name: ClassVar[str] = "dagre_normalize_edges"
    category: ClassVar[OpCategory] = OpCategory.LAYERING
    reads: ClassVar[Tuple[str, ...]] = ("extras", "layers")
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Split long edges into adjacent-rank chains.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable inputs; unused.
        state : SolveState
            State holding ranked nodes.
        ctx : RuntimeContext
            Runtime infrastructure; unused.

        Returns
        -------
        SolveState
            State with a normalized active graph.
        """
        del problem, ctx
        graph = _require_graph(state)
        for edge in list(graph.active_edges()):
            source_rank = graph.nodes[edge.source].rank
            target_rank = graph.nodes[edge.target].rank
            if source_rank is None or target_rank is None:
                raise RuntimeError("Dagre rank stage left a node unranked.")
            if target_rank == source_rank + 1:
                continue
            edge.active = False
            previous = edge.source
            first_dummy: Optional[NodeId] = None
            for rank in range(source_rank + 1, target_rank):
                dummy = graph.add_dummy("edge")
                dummy_node = graph.nodes[dummy]
                dummy_node.rank = rank
                dummy_node.edge_source = edge.source
                dummy_node.edge_target = edge.target
                if first_dummy is None:
                    first_dummy = dummy
                graph.add_edge(
                    source=previous,
                    target=dummy,
                    weight=edge.weight,
                    minlen=1,
                    original_index=edge.original_index,
                    reversed_edge=edge.reversed,
                )
                previous = dummy
            if first_dummy is not None:
                graph.dummy_chains.append(first_dummy)
            graph.add_edge(
                source=previous,
                target=edge.target,
                weight=edge.weight,
                minlen=1,
                original_index=edge.original_index,
                reversed_edge=edge.reversed,
            )
        return state


def _compound_postorder_numbers(
    graph: _DagreGraph,
) -> Dict[Optional[NodeId], Tuple[int, int]]:
    """Return dagre.js parent-tree postorder intervals.

    Parameters
    ----------
    graph : _DagreGraph
        Compound graph.

    Returns
    -------
    dict[Hashable | None, tuple[int, int]]
        ``(low, lim)`` intervals for children traversed from graph roots.
    """
    result: Dict[Optional[NodeId], Tuple[int, int]] = {}
    limit = 0

    def visit(node: NodeId) -> None:
        """Visit one compound-tree node.

        Parameters
        ----------
        node : Hashable
            Current node.

        Returns
        -------
        None
            ``result`` and ``limit`` are mutated.
        """
        nonlocal limit
        low = limit
        for child in graph.children(node):
            visit(child)
        result[node] = (low, limit)
        limit += 1

    for child in graph.children(None):
        visit(child)
    result[None] = (0, limit)
    return result


def _compound_path_through_lca(
    graph: _DagreGraph,
    postorder_nums: Mapping[Optional[NodeId], Tuple[int, int]],
    source: NodeId,
    target: NodeId,
) -> Tuple[List[Optional[NodeId]], Optional[NodeId]]:
    """Return the parent path from source to target through their LCA.

    Parameters
    ----------
    graph : _DagreGraph
        Compound graph.
    postorder_nums : mapping[Hashable | None, tuple[int, int]]
        Parent-tree intervals from :func:`_compound_postorder_numbers`.
    source : Hashable
        Original edge source.
    target : Hashable
        Original edge target.

    Returns
    -------
    tuple[list[Hashable | None], Hashable | None]
        Full parent path and lowest common ancestor.
    """
    source_low, source_lim = postorder_nums.get(source, (0, 0))
    target_low, target_lim = postorder_nums.get(target, (0, 0))
    low = min(source_low, target_low)
    lim = max(source_lim, target_lim)
    source_path: List[Optional[NodeId]] = []
    parent = source
    while True:
        parent = graph.parent_of(parent)
        source_path.append(parent)
        parent_low, parent_lim = postorder_nums.get(parent, (0, lim))
        if parent is None or (parent_low <= low and lim <= parent_lim):
            break
    lca = parent

    target_path: List[Optional[NodeId]] = []
    parent = target
    while True:
        parent = graph.parent_of(parent)
        if parent == lca:
            break
        target_path.append(parent)
        if parent is None:
            break
    return source_path + list(reversed(target_path)), lca


@register_op
class DagreParentDummyChains(Op):
    """Assign normalized edge dummies to the correct compound parent."""

    name: ClassVar[str] = "dagre_parent_dummy_chains"
    category: ClassVar[OpCategory] = OpCategory.LAYERING
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Port dagre.js ``parent-dummy-chains``.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable inputs; unused after graph preparation.
        state : SolveState
            State holding normalized dummy chains.
        ctx : RuntimeContext
            Runtime infrastructure; unused.

        Returns
        -------
        SolveState
            State with dummy parent assignments updated.
        """
        del problem, ctx
        graph = _require_graph(state)
        if not graph.has_compound() or not graph.dummy_chains:
            return state
        postorder_nums = _compound_postorder_numbers(graph)
        for chain_start in graph.dummy_chains:
            current = chain_start
            node = graph.nodes[current]
            if node.edge_source is None or node.edge_target is None:
                continue
            path, lca = _compound_path_through_lca(
                graph,
                postorder_nums,
                node.edge_source,
                node.edge_target,
            )
            path_index = 0
            path_node = path[path_index] if path else None
            ascending = True
            while current != node.edge_target:
                current_node = graph.nodes[current]
                if current_node.rank is None:
                    break
                if ascending:
                    while path_node != lca:
                        if path_node is None:
                            break
                        path_label = graph.nodes[path_node]
                        if path_label.max_rank is None or path_label.max_rank >= current_node.rank:
                            break
                        path_index += 1
                        path_node = path[path_index] if path_index < len(path) else None
                    if path_node == lca:
                        ascending = False
                if not ascending:
                    while path_index < len(path) - 1:
                        next_path_node = path[path_index + 1]
                        if next_path_node is None:
                            break
                        next_label = graph.nodes[next_path_node]
                        if next_label.min_rank is None or next_label.min_rank > current_node.rank:
                            break
                        path_index += 1
                    path_node = path[path_index] if path_index < len(path) else None
                graph.set_parent(current, path_node)
                successors = graph.successors(current)
                if not successors:
                    break
                current = successors[0]
        return state


@register_op
class DagreBorderSegments(Op):
    """Add per-rank left and right border segment dummies for clusters."""

    name: ClassVar[str] = "dagre_border_segments"
    category: ClassVar[OpCategory] = OpCategory.LAYERING
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Port dagre.js ``add-border-segments``.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable inputs; unused after graph preparation.
        state : SolveState
            State holding ranked compound graph.
        ctx : RuntimeContext
            Runtime infrastructure; unused.

        Returns
        -------
        SolveState
            State with border dummy nodes and segment edges inserted.
        """
        del problem, ctx
        graph = _require_graph(state)
        if not graph.has_compound():
            return state

        def add_border_node(
            prop: str,
            cluster: NodeId,
            cluster_node: _DagreNode,
            rank: int,
        ) -> None:
            """Append one border dummy and link it to the previous segment.

            Parameters
            ----------
            prop : str
                ``borderLeft`` or ``borderRight``.
            cluster : Hashable
                Owning cluster id.
            cluster_node : _DagreNode
                Owning cluster label.
            rank : int
                Rank for the border dummy.

            Returns
            -------
            None
                Graph state is mutated.
            """
            storage = (
                cluster_node.border_left if prop == "borderLeft" else cluster_node.border_right
            )
            previous = storage.get(rank - 1)
            current = graph.add_dummy("border")
            current_node = graph.nodes[current]
            current_node.rank = rank
            current_node.border_type = prop
            storage[rank] = current
            graph.set_parent(current, cluster)
            if previous is not None:
                graph.add_edge(previous, current, weight=1.0, minlen=1, original_index=-1)

        def visit(node: NodeId) -> None:
            """Visit one compound-tree node after children.

            Parameters
            ----------
            node : Hashable
                Current node or cluster.

            Returns
            -------
            None
                Border nodes are inserted for cluster nodes.
            """
            for child in graph.children(node):
                visit(child)
            node_data = graph.nodes[node]
            if node_data.min_rank is None or node_data.max_rank is None:
                return
            for rank in range(node_data.min_rank, node_data.max_rank + 1):
                add_border_node("borderLeft", node, node_data, rank)
                add_border_node("borderRight", node, node_data, rank)

        for child in graph.children(None):
            visit(child)
        return state


def _initial_order(graph: _DagreGraph) -> List[List[NodeId]]:
    """Build Dagre's DFS initial ordering.

    Parameters
    ----------
    graph : _DagreGraph
        Ranked normalized graph.

    Returns
    -------
    list[list[Hashable]]
        Initial nodes per rank.
    """
    max_rank = max((graph.nodes[node].rank or 0 for node in graph.node_order), default=-1)
    layers: List[List[NodeId]] = [[] for _ in range(max_rank + 1)]
    visited: Set[NodeId] = set()

    def place(node: NodeId) -> None:
        """Record one node in its rank layer.

        Parameters
        ----------
        node : Hashable
            Node being placed.

        Returns
        -------
        None
            ``layers`` and ``visited`` are mutated.
        """
        visited.add(node)
        rank = graph.nodes[node].rank
        if rank is None:
            raise RuntimeError("Dagre ordering received an unranked node.")
        layers[rank].append(node)

    # Iterative twin of dagre's recursive successor-first DFS (suspended-
    # iterator stack): the layer append order -- which seeds crossing
    # minimization -- is preserved exactly, and deep chains no longer exhaust
    # the recursion limit.
    ordered_nodes = sorted(graph.node_order, key=lambda node: graph.nodes[node].rank or 0)
    frames: List[Iterator[NodeId]] = []
    for start in ordered_nodes:
        if start in visited:
            continue
        place(start)
        frames.append(iter(graph.successors(start)))
        while frames:
            descended = False
            for successor in frames[-1]:
                if successor not in visited:
                    place(successor)
                    frames.append(iter(graph.successors(successor)))
                    descended = True
                    break
            if not descended:
                frames.pop()
    return layers


def _assign_order(graph: _DagreGraph, layers: Sequence[Sequence[NodeId]]) -> None:
    """Write layer positions onto node labels.

    Parameters
    ----------
    graph : _DagreGraph
        Working graph.
    layers : sequence[sequence[Hashable]]
        Ordered nodes per rank.

    Returns
    -------
    None
        Node labels are mutated.
    """
    for layer in layers:
        for order, node in enumerate(layer):
            graph.nodes[node].order = order


def _weighted_neighbors(
    graph: _DagreGraph,
    node: NodeId,
    relationship: str,
) -> List[Tuple[NodeId, float]]:
    """Aggregate incident edge weights by adjacent node.

    Parameters
    ----------
    graph : _DagreGraph
        Normalized graph.
    node : Hashable
        Movable node.
    relationship : str
        ``"in"`` or ``"out"`` sweep relationship.

    Returns
    -------
    list[tuple[Hashable, float]]
        Neighbor ids with summed weights in first-edge order.
    """
    edges = graph.in_edges(node) if relationship == "in" else graph.out_edges(node)
    weights: Dict[NodeId, float] = {}
    for edge in edges:
        neighbor = edge.source if relationship == "in" else edge.target
        weights[neighbor] = weights.get(neighbor, 0.0) + edge.weight
    return list(weights.items())


def _sort_rank(
    graph: _DagreGraph,
    rank: int,
    relationship: str,
    bias_right: bool,
) -> List[NodeId]:
    """Sort one rank with Dagre's weighted barycenter rule.

    Parameters
    ----------
    graph : _DagreGraph
        Normalized graph with current node orders.
    rank : int
        Rank to reorder.
    relationship : str
        Incident direction used for barycenters.
    bias_right : bool
        Reverse stable-index tie bias.

    Returns
    -------
    list[Hashable]
        New rank order.
    """
    movable = [node for node in graph.node_order if graph.nodes[node].rank == rank]
    entries: List[Tuple[NodeId, int, Optional[float], float]] = []
    for index, node in enumerate(movable):
        neighbors = _weighted_neighbors(graph, node, relationship)
        if not neighbors:
            entries.append((node, index, None, 0.0))
            continue
        weighted_sum = 0.0
        total_weight = 0.0
        for neighbor, weight in neighbors:
            order = graph.nodes[neighbor].order
            if order is None:
                raise RuntimeError("Dagre barycenter neighbor has no order.")
            weighted_sum += weight * order
            total_weight += weight
        entries.append((node, index, weighted_sum / total_weight, total_weight))

    sortable = [entry for entry in entries if entry[2] is not None]
    unsortable = sorted(
        (entry for entry in entries if entry[2] is None),
        key=lambda entry: -entry[1],
    )
    sortable.sort(
        key=lambda entry: (
            float(entry[2]),
            -entry[1] if bias_right else entry[1],
        )
    )
    output: List[NodeId] = []
    output_index = 0

    def consume_unsortable() -> None:
        """Insert fixed entries whose source index has been reached.

        Returns
        -------
        None
            Local output collections are mutated.
        """
        nonlocal output_index
        while unsortable and unsortable[-1][1] <= output_index:
            output.append(unsortable.pop()[0])
            output_index += 1

    consume_unsortable()
    for entry in sortable:
        output.append(entry[0])
        output_index += 1
        consume_unsortable()
    consume_unsortable()
    return output


def _cross_count(graph: _DagreGraph, layers: Sequence[Sequence[NodeId]]) -> float:
    """Return Dagre's weighted crossing count with indexed bilayer scans.

    Parameters
    ----------
    graph : _DagreGraph
        Ordered normalized graph.
    layers : sequence[sequence[Hashable]]
        Current layer matrix.

    Returns
    -------
    float
        Weighted adjacent-rank crossing count.
    """
    crossings = 0.0
    for north, south in zip(layers, layers[1:]):
        south_positions = {node: index for index, node in enumerate(south)}
        entries: List[Tuple[int, float]] = []
        for node in north:
            node_entries = [
                (south_positions[edge.target], edge.weight)
                for edge in graph.out_edges(node)
                if edge.target in south_positions
            ]
            entries.extend(sorted(node_entries, key=lambda entry: entry[0]))
        if len(entries) < 2:
            continue

        tree = [0.0] * (len(south_positions) + 1)
        seen_weight = 0.0
        for position, weight in entries:
            index = position + 1
            prefix_weight = 0.0
            scan = index
            while scan > 0:
                prefix_weight += tree[scan]
                scan -= scan & -scan
            crossings += weight * (seen_weight - prefix_weight)
            while index < len(tree):
                tree[index] += weight
                index += index & -index
            seen_weight += weight
    return crossings


@dataclass
class _LayerNode:
    """Layer-graph node label used by dagre's compound ordering pass."""

    order: Optional[int] = None
    border_left: Optional[NodeId] = None
    border_right: Optional[NodeId] = None


@dataclass
class _LayerGraph:
    """Small compound graph view for one ordering sweep rank."""

    root: NodeId
    nodes: Dict[NodeId, _LayerNode] = field(default_factory=dict)
    parents: Dict[NodeId, Optional[NodeId]] = field(default_factory=dict)
    edge_weights: Dict[Tuple[NodeId, NodeId], float] = field(default_factory=dict)
    node_order: List[NodeId] = field(default_factory=list)

    def set_node(self, node: NodeId, label: Optional[_LayerNode] = None) -> None:
        """Add or replace one node label.

        Parameters
        ----------
        node : Hashable
            Layer-graph node id.
        label : _LayerNode | None, optional
            Node label to store.

        Returns
        -------
        None
            The layer graph is mutated.
        """
        if node not in self.nodes:
            self.node_order.append(node)
        self.nodes[node] = label or self.nodes.get(node, _LayerNode())

    def set_parent(self, node: NodeId, parent: Optional[NodeId]) -> None:
        """Assign a parent in the layer graph.

        Parameters
        ----------
        node : Hashable
            Child node.
        parent : Hashable | None
            Parent node.

        Returns
        -------
        None
            Parent metadata is updated.
        """
        self.parents[node] = parent

    def add_edge(self, source: NodeId, target: NodeId, weight: float) -> None:
        """Add or aggregate one weighted edge.

        Parameters
        ----------
        source : Hashable
            Edge tail.
        target : Hashable
            Edge head.
        weight : float
            Edge weight to add.

        Returns
        -------
        None
            Edge weights are updated.
        """
        self.set_node(source)
        self.set_node(target)
        pair = (source, target)
        self.edge_weights[pair] = self.edge_weights.get(pair, 0.0) + weight

    def children(self, parent: NodeId) -> List[NodeId]:
        """Return direct children in insertion order.

        Parameters
        ----------
        parent : Hashable
            Parent id.

        Returns
        -------
        list[Hashable]
            Child ids.
        """
        return [node for node in self.node_order if self.parents.get(node) == parent]

    def parent_of(self, node: NodeId) -> Optional[NodeId]:
        """Return one layer-graph parent.

        Parameters
        ----------
        node : Hashable
            Node id.

        Returns
        -------
        Hashable | None
            Parent id.
        """
        return self.parents.get(node)

    def in_edges(self, node: NodeId) -> List[Tuple[NodeId, NodeId, float]]:
        """Return incoming weighted edges.

        Parameters
        ----------
        node : Hashable
            Target node.

        Returns
        -------
        list[tuple[Hashable, Hashable, float]]
            Incoming edges in insertion order.
        """
        return [
            (source, target, weight)
            for (source, target), weight in self.edge_weights.items()
            if target == node
        ]

    def predecessors(self, node: NodeId) -> List[NodeId]:
        """Return distinct predecessors.

        Parameters
        ----------
        node : Hashable
            Target node.

        Returns
        -------
        list[Hashable]
            Source node ids.
        """
        return _unique([source for source, _target, _weight in self.in_edges(node)])


@dataclass
class _SortEntry:
    """Sortable barycenter entry for compound ordering."""

    vs: List[NodeId]
    i: int
    barycenter: Optional[float] = None
    weight: float = 0.0


@dataclass
class _SortResult:
    """Recursive sort result from ``sort-subgraph``."""

    vs: List[NodeId]
    barycenter: Optional[float] = None
    weight: float = 0.0


def _compound_initial_order(graph: _DagreGraph) -> List[List[NodeId]]:
    """Build dagre's DFS initial order over simple compound nodes.

    Parameters
    ----------
    graph : _DagreGraph
        Ranked normalized compound graph.

    Returns
    -------
    list[list[Hashable]]
        Initial layer matrix.
    """
    simple_nodes = [node for node in graph.node_order if not graph.children(node)]
    max_rank = max((graph.nodes[node].rank or 0 for node in simple_nodes), default=-1)
    layers: List[List[NodeId]] = [[] for _ in range(max_rank + 1)]
    visited: Set[NodeId] = set()

    def visit(node: NodeId) -> None:
        """Visit one node in successor-first DFS order.

        Parameters
        ----------
        node : Hashable
            Current node.

        Returns
        -------
        None
            ``layers`` and ``visited`` are mutated.
        """
        if node in visited:
            return
        visited.add(node)
        rank = graph.nodes[node].rank
        if rank is None:
            return
        layers[rank].append(node)
        for successor in graph.successors(node):
            visit(successor)

    for node in sorted(simple_nodes, key=lambda item: graph.nodes[item].rank or 0):
        visit(node)
    return layers


def _build_compound_layer_graph(
    graph: _DagreGraph,
    rank: int,
    relationship: str,
) -> _LayerGraph:
    """Build dagre's subgraph-aware layer graph for one rank.

    Parameters
    ----------
    graph : _DagreGraph
        Full compound working graph.
    rank : int
        Rank to sort.
    relationship : str
        ``"in"`` or ``"out"`` incident-edge selector.

    Returns
    -------
    _LayerGraph
        Layer graph rooted at a synthetic node.
    """
    root: NodeId = ("layer_root", rank, relationship)
    layer_graph = _LayerGraph(root=root)
    layer_graph.set_node(root)
    for node in graph.node_order:
        node_data = graph.nodes[node]
        spans_rank = node_data.rank == rank or (
            node_data.min_rank is not None
            and node_data.max_rank is not None
            and node_data.min_rank <= rank <= node_data.max_rank
        )
        if not spans_rank:
            continue
        label = _LayerNode(order=node_data.order)
        if node_data.min_rank is not None:
            label.border_left = node_data.border_left.get(rank)
            label.border_right = node_data.border_right.get(rank)
        layer_graph.set_node(node, label)
        layer_graph.set_parent(node, graph.parent_of(node) or root)
        incident = graph.in_edges(node) if relationship == "in" else graph.out_edges(node)
        for edge in incident:
            other = edge.source if edge.target == node else edge.target
            other_order = graph.nodes[other].order
            layer_graph.set_node(other, _LayerNode(order=other_order))
            layer_graph.add_edge(other, node, edge.weight)
    return layer_graph


def _compound_barycenters(layer_graph: _LayerGraph, movable: Sequence[NodeId]) -> List[_SortEntry]:
    """Compute weighted barycenters for one layer-graph child list.

    Parameters
    ----------
    layer_graph : _LayerGraph
        Layer graph.
    movable : sequence[Hashable]
        Child ids to sort.

    Returns
    -------
    list[_SortEntry]
        Barycenter entries.
    """
    entries: List[_SortEntry] = []
    for index, node in enumerate(movable):
        incoming = layer_graph.in_edges(node)
        if not incoming:
            entries.append(_SortEntry(vs=[node], i=index))
            continue
        weighted_sum = 0.0
        total_weight = 0.0
        for source, _target, weight in incoming:
            source_order = layer_graph.nodes[source].order
            if source_order is None:
                continue
            weighted_sum += weight * source_order
            total_weight += weight
        if total_weight == 0.0:
            entries.append(_SortEntry(vs=[node], i=index))
        else:
            entries.append(
                _SortEntry(
                    vs=[node],
                    i=index,
                    barycenter=weighted_sum / total_weight,
                    weight=total_weight,
                )
            )
    return entries


def _merge_barycenters(target: _SortEntry, other: _SortResult) -> None:
    """Merge recursive subgraph barycenter data into a parent entry.

    Parameters
    ----------
    target : _SortEntry
        Parent entry to update.
    other : _SortResult
        Child subgraph result.

    Returns
    -------
    None
        ``target`` is mutated.
    """
    if other.barycenter is None or other.weight == 0.0:
        return
    if target.barycenter is not None:
        total = target.weight + other.weight
        target.barycenter = (
            target.barycenter * target.weight + other.barycenter * other.weight
        ) / total
        target.weight = total
    else:
        target.barycenter = other.barycenter
        target.weight = other.weight


def _resolve_compound_conflicts(
    entries: Sequence[_SortEntry],
    constraints: Sequence[Tuple[NodeId, NodeId]],
) -> List[_SortEntry]:
    """Resolve barycenter order conflicts against subgraph constraints.

    Parameters
    ----------
    entries : sequence[_SortEntry]
        Initial entries.
    constraints : sequence[tuple[Hashable, Hashable]]
        Constraint graph edges in insertion order.

    Returns
    -------
    list[_SortEntry]
        Coalesced entries.
    """
    mapped: Dict[NodeId, Dict[str, Any]] = {}
    for index, entry in enumerate(entries):
        item: Dict[str, Any] = {
            "indegree": 0,
            "in": [],
            "out": [],
            "vs": list(entry.vs),
            "i": index,
            "barycenter": entry.barycenter,
            "weight": entry.weight,
            "merged": False,
        }
        mapped[entry.vs[0]] = item
    for source, target in constraints:
        source_entry = mapped.get(source)
        target_entry = mapped.get(target)
        if source_entry is None or target_entry is None:
            continue
        target_entry["indegree"] += 1
        source_entry["out"].append(target_entry)
    source_set = [
        mapped[node] for node in _graphlib_key_order(list(mapped)) if not mapped[node]["indegree"]
    ]
    resolved: List[Dict[str, Any]] = []

    def merge_entries(target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Merge two constrained entries.

        Parameters
        ----------
        target : dict[str, Any]
            Entry that survives.
        source : dict[str, Any]
            Entry merged into target.

        Returns
        -------
        None
            Dictionaries are mutated.
        """
        total_sum = 0.0
        total_weight = 0.0
        if source.get("weight"):
            total_sum += float(source["barycenter"]) * float(source["weight"])
            total_weight += float(source["weight"])
        if target.get("weight"):
            total_sum += float(target["barycenter"]) * float(target["weight"])
            total_weight += float(target["weight"])
        target["vs"] = list(source["vs"]) + list(target["vs"])
        target["barycenter"] = total_sum / total_weight if total_weight else None
        target["weight"] = total_weight
        target["i"] = min(int(source["i"]), int(target["i"]))
        source["merged"] = True

    while source_set:
        entry = source_set.pop()
        resolved.append(entry)
        for incoming in reversed(entry["in"]):
            if incoming["merged"]:
                continue
            if (
                incoming.get("barycenter") is None
                or entry.get("barycenter") is None
                or float(incoming["barycenter"]) >= float(entry["barycenter"])
            ):
                merge_entries(entry, incoming)
        for outgoing in entry["out"]:
            outgoing["in"].append(entry)
            outgoing["indegree"] -= 1
            if outgoing["indegree"] == 0:
                source_set.append(outgoing)
    return [
        _SortEntry(
            vs=list(entry["vs"]),
            i=int(entry["i"]),
            barycenter=entry.get("barycenter"),
            weight=float(entry.get("weight") or 0.0),
        )
        for entry in resolved
        if not entry["merged"]
    ]


def _sort_compound_entries(entries: Sequence[_SortEntry], bias_right: bool) -> _SortResult:
    """Sort barycenter entries using dagre's stable bias rule.

    Parameters
    ----------
    entries : sequence[_SortEntry]
        Entries to sort.
    bias_right : bool
        Reverse tie bias.

    Returns
    -------
    _SortResult
        Flattened sorted nodes plus merged barycenter.
    """
    sortable = [entry for entry in entries if entry.barycenter is not None]
    unsortable = sorted(
        (entry for entry in entries if entry.barycenter is None),
        key=lambda entry: -entry.i,
    )
    sortable.sort(
        key=lambda entry: (
            float(entry.barycenter),
            -entry.i if bias_right else entry.i,
        )
    )
    output_chunks: List[List[NodeId]] = []
    output_index = 0
    weighted_sum = 0.0
    total_weight = 0.0

    def consume_unsortable() -> None:
        """Consume fixed entries whose insertion index has been reached.

        Returns
        -------
        None
            Local output state is mutated.
        """
        nonlocal output_index
        while unsortable and unsortable[-1].i <= output_index:
            entry = unsortable.pop()
            output_chunks.append(entry.vs)
            output_index += 1

    consume_unsortable()
    for entry in sortable:
        output_index += len(entry.vs)
        output_chunks.append(entry.vs)
        weighted_sum += float(entry.barycenter) * entry.weight
        total_weight += entry.weight
        consume_unsortable()
    flattened = [node for chunk in output_chunks for node in chunk]
    if total_weight:
        return _SortResult(
            vs=flattened,
            barycenter=weighted_sum / total_weight,
            weight=total_weight,
        )
    return _SortResult(vs=flattened)


def _sort_compound_subgraph(
    layer_graph: _LayerGraph,
    node: NodeId,
    constraints: Sequence[Tuple[NodeId, NodeId]],
    bias_right: bool,
) -> _SortResult:
    """Recursively sort a compound subgraph for one rank.

    Parameters
    ----------
    layer_graph : _LayerGraph
        Layer graph.
    node : Hashable
        Root or cluster node to sort.
    constraints : sequence[tuple[Hashable, Hashable]]
        Persistent subgraph constraint graph.
    bias_right : bool
        Reverse tie bias.

    Returns
    -------
    _SortResult
        Sorted child ids and optional barycenter.
    """
    movable = layer_graph.children(node)
    node_label = layer_graph.nodes.get(node)
    border_left = node_label.border_left if node_label is not None else None
    border_right = node_label.border_right if node_label is not None else None
    if border_left is not None and border_right is not None:
        movable = [child for child in movable if child not in {border_left, border_right}]
    subgraphs: Dict[NodeId, _SortResult] = {}
    entries = _compound_barycenters(layer_graph, movable)
    for entry in entries:
        child = entry.vs[0]
        if layer_graph.children(child):
            result = _sort_compound_subgraph(layer_graph, child, constraints, bias_right)
            subgraphs[child] = result
            _merge_barycenters(entry, result)
    resolved = _resolve_compound_conflicts(entries, constraints)
    for entry in resolved:
        expanded: List[NodeId] = []
        for child in entry.vs:
            expanded.extend(subgraphs[child].vs if child in subgraphs else [child])
        entry.vs = expanded
    result = _sort_compound_entries(resolved, bias_right)
    if border_left is not None and border_right is not None:
        result.vs = [border_left, *result.vs, border_right]
        left_predecessors = layer_graph.predecessors(border_left)
        right_predecessors = layer_graph.predecessors(border_right)
        if left_predecessors and right_predecessors:
            left_order = layer_graph.nodes[left_predecessors[0]].order
            right_order = layer_graph.nodes[right_predecessors[0]].order
            if left_order is not None and right_order is not None:
                if result.barycenter is None:
                    result.barycenter = 0.0
                    result.weight = 0.0
                result.barycenter = (
                    result.barycenter * result.weight + left_order + right_order
                ) / (result.weight + 2.0)
                result.weight += 2.0
    return result


def _add_compound_subgraph_constraints(
    layer_graph: _LayerGraph,
    constraints: List[Tuple[NodeId, NodeId]],
    constraint_set: Set[Tuple[NodeId, NodeId]],
    ordered: Sequence[NodeId],
) -> None:
    """Add dagre's ordering constraints between adjacent subgraph blocks.

    Parameters
    ----------
    layer_graph : _LayerGraph
        Layer graph.
    constraints : list[tuple[Hashable, Hashable]]
        Constraint graph to update.
    constraint_set : set[tuple[Hashable, Hashable]]
        Existing constraint edges.
    ordered : sequence[Hashable]
        Sorted flattened ids for the current rank.

    Returns
    -------
    None
        ``constraints`` is mutated.
    """
    previous_by_parent: Dict[NodeId, NodeId] = {}
    root_previous: Optional[NodeId] = None
    for node in ordered:
        child = layer_graph.parent_of(node)
        while child is not None:
            parent = layer_graph.parent_of(child)
            if parent is not None:
                previous_child = previous_by_parent.get(parent)
                previous_by_parent[parent] = child
            else:
                previous_child = root_previous
                root_previous = child
            if previous_child is not None and previous_child != child:
                edge = (previous_child, child)
                if edge not in constraint_set:
                    constraint_set.add(edge)
                    constraints.append(edge)
                return
            child = parent


def _compound_order_graph(graph: _DagreGraph) -> List[List[NodeId]]:
    """Run dagre's recursive compound ordering sweeps.

    Parameters
    ----------
    graph : _DagreGraph
        Ranked normalized compound graph.

    Returns
    -------
    list[list[Hashable]]
        Best layer matrix.
    """
    layers = _compound_initial_order(graph)
    _assign_order(graph, layers)
    max_rank = len(layers) - 1
    down_layer_graphs = [
        _build_compound_layer_graph(graph, rank, "in") for rank in range(1, max_rank + 1)
    ]
    up_layer_graphs = [
        _build_compound_layer_graph(graph, rank, "out") for rank in range(max_rank - 1, -1, -1)
    ]
    best_crossings = float("inf")
    best = [list(layer) for layer in layers]
    iteration = 0
    iterations_since_best = 0
    while iterations_since_best < 4:
        layer_graphs = down_layer_graphs if iteration % 2 else up_layer_graphs
        bias_right = iteration % 4 >= 2
        constraints: List[Tuple[NodeId, NodeId]] = []
        constraint_set: Set[Tuple[NodeId, NodeId]] = set()
        for layer_graph in layer_graphs:
            for node, label in layer_graph.nodes.items():
                if node in graph.nodes:
                    label.order = graph.nodes[node].order
            sorted_result = _sort_compound_subgraph(
                layer_graph,
                layer_graph.root,
                constraints,
                bias_right,
            )
            for order, node in enumerate(sorted_result.vs):
                layer_graph.nodes[node].order = order
                if node in graph.nodes:
                    graph.nodes[node].order = order
            _add_compound_subgraph_constraints(
                layer_graph,
                constraints,
                constraint_set,
                sorted_result.vs,
            )
        layers = _build_layer_matrix(graph)
        crossing_count = _cross_count(graph, layers)
        if crossing_count < best_crossings:
            best_crossings = crossing_count
            best = [list(layer) for layer in layers]
            iterations_since_best = 0
        iteration += 1
        iterations_since_best += 1
    _assign_order(graph, best)
    return best


def _build_layer_matrix(graph: _DagreGraph) -> List[List[NodeId]]:
    """Build a rank/order matrix from current node labels.

    Parameters
    ----------
    graph : _DagreGraph
        Ordered working graph.

    Returns
    -------
    list[list[Hashable]]
        Layer matrix.
    """
    max_rank = max(
        (int(node.rank) for node in graph.nodes.values() if node.rank is not None),
        default=-1,
    )
    layers: List[List[NodeId]] = [[] for _ in range(max_rank + 1)]
    for node in graph.node_order:
        node_data = graph.nodes[node]
        if node_data.rank is None or node_data.order is None:
            continue
        rank = int(node_data.rank)
        order = int(node_data.order)
        while len(layers[rank]) <= order:
            layers[rank].append(node)
        layers[rank][order] = node
    return layers


def _order_graph(graph: _DagreGraph) -> List[List[NodeId]]:
    """Run Dagre's alternating barycenter ordering sweeps.

    Parameters
    ----------
    graph : _DagreGraph
        Ranked normalized graph.

    Returns
    -------
    list[list[Hashable]]
        Best layer matrix by weighted crossing count.
    """
    layers = _initial_order(graph)
    _assign_order(graph, layers)
    max_rank = len(layers) - 1
    best_crossings = float("inf")
    best = [list(layer) for layer in layers]
    iteration = 0
    iterations_since_best = 0
    while iterations_since_best < 4:
        relationship = "in" if iteration % 2 else "out"
        ranks = range(1, max_rank + 1) if relationship == "in" else range(max_rank - 1, -1, -1)
        bias_right = iteration % 4 >= 2
        for rank in ranks:
            ordered = _sort_rank(graph, rank, relationship, bias_right)
            for order, node in enumerate(ordered):
                graph.nodes[node].order = order
        layers = [[] for _ in range(max_rank + 1)]
        for node in graph.node_order:
            node_data = graph.nodes[node]
            if node_data.rank is not None and node_data.order is not None:
                while len(layers[node_data.rank]) <= node_data.order:
                    layers[node_data.rank].append(node)
                layers[node_data.rank][node_data.order] = node
        crossing_count = _cross_count(graph, layers)
        if crossing_count < best_crossings:
            best_crossings = crossing_count
            best = [list(layer) for layer in layers]
            iterations_since_best = 0
        iteration += 1
        iterations_since_best += 1
    _assign_order(graph, best)
    return best


@register_op
class DagreOrderNodes(Op):
    """Minimize crossings and publish Brandes-Koepf layer metadata."""

    name: ClassVar[str] = "dagre_order_nodes"
    category: ClassVar[OpCategory] = OpCategory.ORDERING
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras", "ordering")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run ordering, insert self-edge dummies, and export BK inputs.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable inputs; unused.
        state : SolveState
            State holding the normalized graph.
        ctx : RuntimeContext
            Runtime infrastructure; unused.

        Returns
        -------
        SolveState
            State with ordering snapshots and BK metadata.
        """
        del problem, ctx
        graph = _require_graph(state)
        layers = _compound_order_graph(graph) if graph.has_compound() else _order_graph(graph)
        for rank, layer in enumerate(layers):
            expanded: List[NodeId] = []
            for node in layer:
                expanded.append(node)
                original_index_by_id = {
                    original_node: index
                    for index, original_node in enumerate(graph.original_node_ids)
                }
                original_index = original_index_by_id.get(node)
                if original_index is not None:
                    for _self_edge in graph.self_edges.get(original_index, []):
                        dummy = graph.add_dummy("selfedge")
                        graph.nodes[dummy].rank = rank
                        expanded.append(dummy)
            layers[rank] = expanded
        _assign_order(graph, layers)

        original_ordering = [0] * graph.num_original_nodes
        for node_index, node in enumerate(graph.original_node_ids):
            original_ordering[node_index] = int(graph.nodes[node].order or 0)
        state.extras[_DAGRE_ORDERING_KEY] = original_ordering
        state.ordering = torch.tensor(original_ordering, dtype=torch.long)

        predecessors = {node: graph.predecessors(node) for node in graph.node_order}
        successors = {node: graph.successors(node) for node in graph.node_order}
        horizontal = graph.rankdir in {"LR", "RL"}
        widths = {
            node: graph.nodes[node].height if horizontal else graph.nodes[node].width
            for node in graph.node_order
        }
        dummy_nodes = {node for node in graph.node_order if graph.nodes[node].dummy is not None}
        border_types = {
            node: node_data.border_type
            for node, node_data in graph.nodes.items()
            if node_data.border_type is not None
        }
        state.extras[BRANDES_KOEPF_LAYERING_KEY] = layers
        state.extras[BRANDES_KOEPF_PREDECESSORS_KEY] = predecessors
        state.extras[BRANDES_KOEPF_SUCCESSORS_KEY] = successors
        state.extras[BRANDES_KOEPF_WIDTHS_KEY] = widths
        state.extras[BRANDES_KOEPF_DUMMY_NODES_KEY] = dummy_nodes
        state.extras[BRANDES_KOEPF_BORDER_TYPES_KEY] = border_types
        return state


@register_op
class DagreAssignY(Op):
    """Assign Dagre's box-aware rank coordinates and combine them with BK x."""

    name: ClassVar[str] = "dagre_assign_y"
    category: ClassVar[OpCategory] = OpCategory.COORDINATE
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Place each rank at cumulative maximum-height offsets.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable inputs; unused.
        state : SolveState
            State containing BK x coordinates and layers.
        ctx : RuntimeContext
            Runtime infrastructure; unused.

        Returns
        -------
        SolveState
            State with internal ``(x, y)`` coordinates.
        """
        del problem, ctx
        graph = _require_graph(state)
        x_coordinates: Mapping[NodeId, float] = state.extras[BRANDES_KOEPF_X_KEY]
        layers: Sequence[Sequence[NodeId]] = state.extras[BRANDES_KOEPF_LAYERING_KEY]
        horizontal = graph.rankdir in {"LR", "RL"}
        y_coordinates: Dict[NodeId, float] = {}
        previous_y = 0.0
        for layer in layers:
            heights = [
                graph.nodes[node].width if horizontal else graph.nodes[node].height
                for node in layer
            ]
            max_height = max(heights, default=0.0)
            for node in layer:
                y_coordinates[node] = previous_y + max_height / 2.0
            previous_y += max_height + graph.rank_sep
        state.extras[_DAGRE_INTERNAL_POSITIONS_KEY] = {
            node: (x_coordinates[node], y_coordinates[node])
            for node in graph.node_order
            if node in x_coordinates and node in y_coordinates
        }
        return state


def _apply_rankdir(x: float, y: float, rankdir: str) -> Tuple[float, float]:
    """Undo Dagre's adjusted coordinate system for one point.

    Parameters
    ----------
    x : float
        Adjusted horizontal coordinate.
    y : float
        Adjusted rank coordinate.
    rankdir : str
        Uppercase rank direction.

    Returns
    -------
    tuple[float, float]
        Output coordinate in the requested orientation.
    """
    if rankdir == "BT":
        return x, -y
    if rankdir == "LR":
        return y, x
    if rankdir == "RL":
        return -y, x
    return x, y


def _project_hard_pins(positions: torch.Tensor, config: object) -> torch.Tensor:
    """Apply hard ``LayoutFlex`` pins without coupling Dagre to the engine.

    Parameters
    ----------
    positions : torch.Tensor
        Final positions with shape ``[N, 2]``.
    config : object
        Optional ``LayoutConfig``-like object.

    Returns
    -------
    torch.Tensor
        Positions with hard-pinned axes overwritten.
    """
    flex = getattr(config, "flex", None)
    pins = getattr(flex, "pins", None)
    if not pins:
        return positions
    result = positions.clone()
    for node, axes in pins.items():
        if not isinstance(node, int) or node < 0 or node >= positions.shape[0]:
            continue
        for axis, constraint in enumerate(axes):
            if constraint is not None and bool(getattr(constraint, "is_hard", False)):
                result[node, axis] = float(constraint.target)
    return result


@register_op
class DagreFinalizeCoordinates(Op):
    """Undo rank direction, translate extents, strip dummies, and apply pins."""

    name: ClassVar[str] = "dagre_finalize_coordinates"
    category: ClassVar[OpCategory] = OpCategory.POSTPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("pos",)

    def __init__(self, config: Optional[object] = None) -> None:
        """Store an optional layout config for hard pins.

        Parameters
        ----------
        config : object | None, optional
            LayoutConfig-like object carrying resolved flex constraints.

        Returns
        -------
        None
            The config reference is stored.
        """
        self.config = config

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Produce public coordinates with Dagre's positive-extent translation.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable problem used for output device selection.
        state : SolveState
            State containing adjusted internal coordinates.
        ctx : RuntimeContext
            Runtime infrastructure; unused.

        Returns
        -------
        SolveState
            State with ``pos`` shaped ``[N, 2]``.
        """
        del ctx
        graph = _require_graph(state)
        internal: Mapping[NodeId, Tuple[float, float]] = state.extras[_DAGRE_INTERNAL_POSITIONS_KEY]
        oriented = {node: _apply_rankdir(x, y, graph.rankdir) for node, (x, y) in internal.items()}
        min_x = float("inf")
        min_y = float("inf")
        max_x = 0.0
        max_y = 0.0
        for node, (x, y) in oriented.items():
            node_data = graph.nodes[node]
            width = node_data.width
            height = node_data.height
            min_x = min(min_x, x - width / 2.0)
            max_x = max(max_x, x + width / 2.0)
            min_y = min(min_y, y - height / 2.0)
            max_y = max(max_y, y + height / 2.0)
        del max_x, max_y
        if min_x == float("inf"):
            min_x = 0.0
            min_y = 0.0
        positions = torch.zeros((graph.num_original_nodes, 2), dtype=torch.float64)
        for node_index, node in enumerate(graph.original_node_ids):
            x, y = oriented[node]
            positions[node_index, 0] = x - min_x
            positions[node_index, 1] = y - min_y
        if self.config is not None:
            positions = _project_hard_pins(positions, self.config)
        state.pos = positions.to(device=problem.edge_index.device)
        return state


__all__ = [
    "DagreAssignRanks",
    "DagreAssignY",
    "DagreBorderSegments",
    "DagreCleanupNestingGraph",
    "DagreFinalizeCoordinates",
    "DagreMakeAcyclic",
    "DagreNestingGraph",
    "DagreNormalizeEdges",
    "DagreOrderNodes",
    "DagreParentDummyChains",
    "DagrePrepareGraph",
]
