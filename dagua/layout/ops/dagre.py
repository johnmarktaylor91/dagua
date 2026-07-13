"""Composable stages for the dagre.js 0.8.5 layered layout engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Dict, Hashable, List, Mapping, Optional, Sequence, Set, Tuple

import torch

from dagua.layout.ops.base import Op
from dagua.layout.ops.brandes_koepf import (
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


@dataclass
class _DagreGraph:
    """Working graph shared across Dagre operations."""

    nodes: Dict[NodeId, _DagreNode]
    node_order: List[NodeId]
    edges: List[_DagreEdge]
    num_original_nodes: int
    rank_sep: float
    node_sep: float
    edge_sep: float
    rankdir: str
    ranker: str
    acyclicer: str
    self_edges: Dict[int, List[_DagreEdge]] = field(default_factory=dict)
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

    def add_edge(
        self,
        source: NodeId,
        target: NodeId,
        weight: float,
        minlen: int,
        original_index: int,
        reversed_edge: bool = False,
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
                raise ValueError("node_sizes must have shape [N, 2].")
        weights = (
            torch.ones(edge_index.shape[1], dtype=torch.float64)
            if problem.edge_weights is None
            else problem.edge_weights.detach().to(device="cpu", dtype=torch.float64)
        )
        if weights.shape != (edge_index.shape[1],):
            raise ValueError("edge_weights must have shape [E].")

        nodes = {
            node: _DagreNode(width=float(sizes[node, 0]), height=float(sizes[node, 1]))
            for node in range(problem.num_nodes)
        }
        graph = _DagreGraph(
            nodes=nodes,
            node_order=list(range(problem.num_nodes)),
            edges=[],
            num_original_nodes=problem.num_nodes,
            # Dagre always halves ranksep and doubles minlen to reserve the
            # half-ranks used by potential edge labels, even when labels are empty.
            rank_sep=self.rank_sep / 2.0,
            node_sep=self.node_sep,
            edge_sep=self.edge_sep,
            rankdir=self.rankdir,
            ranker=self.ranker,
            acyclicer=self.acyclicer,
        )
        edges_by_pair: Dict[Tuple[int, int], _DagreEdge] = {}
        for edge_index_value, (source, target) in enumerate(
            zip(edge_index[0].tolist(), edge_index[1].tolist())
        ):
            pair = (int(source), int(target))
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

    def visit(node: NodeId) -> None:
        """Depth-first visit one node.

        Parameters
        ----------
        node : Hashable
            Node to visit.

        Returns
        -------
        None
            Traversal collections are mutated.
        """
        if node in visited:
            return
        visited.add(node)
        stack.add(node)
        for edge in graph.out_edges(node):
            if edge.target in stack:
                feedback.append(edge)
            else:
                visit(edge.target)
        stack.remove(node)

    for node in graph.node_order:
        visit(node)
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


def _simplified_rank_edges(graph: _DagreGraph) -> List[Tuple[NodeId, NodeId, int, int]]:
    """Aggregate multiedges for Dagre's rank stage.

    Parameters
    ----------
    graph : _DagreGraph
        Acyclic working graph.

    Returns
    -------
    list[tuple[Hashable, Hashable, int, int]]
        Pair edges with summed integer weights and maximum ``minlen``.
    """
    records: Dict[Tuple[NodeId, NodeId], Tuple[int, int]] = {}
    for edge in graph.active_edges():
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

    def visit(node: NodeId) -> int:
        """Return the recursively assigned rank for one node.

        Parameters
        ----------
        node : Hashable
            Node to rank.

        Returns
        -------
        int
            Sink-anchored rank.
        """
        if node in ranks:
            return ranks[node]
        ranks[node] = 0
        candidates = [visit(target) - minlen for target, minlen in outgoing[node]]
        ranks[node] = min(candidates) if candidates else 0
        return ranks[node]

    for node in node_order:
        if incoming_count[node] == 0:
            visit(node)
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
        root = graph.add_dummy("root")
        original_edges = _simplified_rank_edges(graph)
        rank_edges = [*original_edges]
        for node in graph.node_order:
            if node != root:
                rank_edges.append((root, node, 1, 0))

        if graph.ranker == "longest-path":
            ranks = _longest_path_ranks(graph.node_order, rank_edges)
        elif graph.ranker == "tight-tree":
            ranks = _tight_tree_ranks(graph.node_order, rank_edges)
        else:
            ranks = _dagre_network_simplex_ranks(graph.node_order, rank_edges)

        graph.node_order.remove(root)
        del graph.nodes[root]
        minimum = min((ranks[node] for node in graph.node_order), default=0)
        for node in graph.node_order:
            graph.nodes[node].rank = int(ranks[node] - minimum)
        original_ranks = [
            int(graph.nodes[node].rank or 0) for node in range(graph.num_original_nodes)
        ]
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
            for rank in range(source_rank + 1, target_rank):
                dummy = graph.add_dummy("edge")
                graph.nodes[dummy].rank = rank
                graph.add_edge(
                    source=previous,
                    target=dummy,
                    weight=edge.weight,
                    minlen=1,
                    original_index=edge.original_index,
                    reversed_edge=edge.reversed,
                )
                previous = dummy
            graph.add_edge(
                source=previous,
                target=edge.target,
                weight=edge.weight,
                minlen=1,
                original_index=edge.original_index,
                reversed_edge=edge.reversed,
            )
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

    def visit(node: NodeId) -> None:
        """Visit one node in successor-first DFS order.

        Parameters
        ----------
        node : Hashable
            Node to visit.

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
            raise RuntimeError("Dagre ordering received an unranked node.")
        layers[rank].append(node)
        for successor in graph.successors(node):
            visit(successor)

    ordered_nodes = sorted(graph.node_order, key=lambda node: graph.nodes[node].rank or 0)
    for node in ordered_nodes:
        visit(node)
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
    """Return Dagre's weighted crossing count.

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
        for entry_index, (position, weight) in enumerate(entries):
            for later_position, later_weight in entries[entry_index + 1 :]:
                if later_position < position:
                    crossings += weight * later_weight
    return crossings


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
        layers = _order_graph(graph)
        for rank, layer in enumerate(layers):
            expanded: List[NodeId] = []
            for node in layer:
                expanded.append(node)
                if isinstance(node, int):
                    for _self_edge in graph.self_edges.get(node, []):
                        dummy = graph.add_dummy("selfedge")
                        graph.nodes[dummy].rank = rank
                        expanded.append(dummy)
            layers[rank] = expanded
        _assign_order(graph, layers)

        original_ordering = [0] * graph.num_original_nodes
        for node in range(graph.num_original_nodes):
            original_ordering[node] = int(graph.nodes[node].order or 0)
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
        state.extras[BRANDES_KOEPF_LAYERING_KEY] = layers
        state.extras[BRANDES_KOEPF_PREDECESSORS_KEY] = predecessors
        state.extras[BRANDES_KOEPF_SUCCESSORS_KEY] = successors
        state.extras[BRANDES_KOEPF_WIDTHS_KEY] = widths
        state.extras[BRANDES_KOEPF_DUMMY_NODES_KEY] = dummy_nodes
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
            node: (x_coordinates[node], y_coordinates[node]) for node in graph.node_order
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
        for node in range(graph.num_original_nodes):
            x, y = oriented[node]
            positions[node, 0] = x - min_x
            positions[node, 1] = y - min_y
        if self.config is not None:
            positions = _project_hard_pins(positions, self.config)
        state.pos = positions.to(device=problem.edge_index.device)
        return state


__all__ = [
    "DagreAssignRanks",
    "DagreAssignY",
    "DagreFinalizeCoordinates",
    "DagreMakeAcyclic",
    "DagreNormalizeEdges",
    "DagreOrderNodes",
    "DagrePrepareGraph",
]
