"""Graphviz dot-compatible network-simplex rank assignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Hashable, List, Literal, Optional, Sequence, Tuple, Union

import torch

_GRAPHVIZ_SEARCH_SIZE = 30
_NO_TREE_INDEX = -1

NodeId = Hashable
VirtualNodeFactory = Callable[..., NodeId]
EdgeLike = Union[Tuple[NodeId, NodeId], Tuple[NodeId, NodeId, int], Tuple[NodeId, NodeId, int, int]]
BalanceMode = Literal["none", "tb", "lr"]


@dataclass(frozen=True)
class GraphvizVirtualEdge:
    """Describe the dummy-node chain inserted for one long dot edge.

    Parameters
    ----------
    original_edge_index : int
        Index of the source edge in the normalized input edge list.
    source : Hashable
        Original edge tail node.
    target : Hashable
        Original edge head node.
    virtual_nodes : tuple of Hashable
        Virtual nodes placed on intermediate ranks.
    chain : tuple of Hashable
        Complete split-edge chain, including ``source`` and ``target``.
    """

    original_edge_index: int
    source: NodeId
    target: NodeId
    virtual_nodes: Tuple[NodeId, ...]
    chain: Tuple[NodeId, ...]


@dataclass(frozen=True)
class _EdgeRecord:
    """Store one normalized rank constraint edge."""

    tail: NodeId
    head: NodeId
    minlen: int
    weight: int
    original_index: int


@dataclass
class _NSNode:
    """Mutable node record for the network-simplex port."""

    in_edges: List[int] = field(default_factory=list)
    out_edges: List[int] = field(default_factory=list)
    tree_in: List[int] = field(default_factory=list)
    tree_out: List[int] = field(default_factory=list)
    rank: int = 0
    priority: int = 0
    low: int = 0
    lim: int = 0
    par: Optional[int] = None
    subtree: int = -1


@dataclass
class _NSEdge:
    """Mutable edge record for the network-simplex port."""

    tail: int
    head: int
    minlen: int
    weight: int
    original_index: int
    cutvalue: int = 0
    tree_index: int = _NO_TREE_INDEX


@dataclass
class _Subtree:
    """Union-find record for the feasible-tight-tree construction."""

    rep: int
    size: int
    parent: int
    heap_index: int


@dataclass
class _SimplexGraph:
    """Hold Graphviz network-simplex mutable state."""

    nodes: List[_NSNode]
    edges: List[_NSEdge]
    tree_edges: List[int] = field(default_factory=list)
    search_index: int = 0
    search_size: int = _GRAPHVIZ_SEARCH_SIZE


def graphviz_rank_assignment(
    edges: Union[torch.Tensor, Sequence[EdgeLike]],
    virtual_node_factory: VirtualNodeFactory,
    num_nodes: Optional[int] = None,
    edge_minlens: Optional[Sequence[int]] = None,
    edge_weights: Optional[Union[torch.Tensor, Sequence[float]]] = None,
    maxiter: Optional[int] = None,
    search_size: int = _GRAPHVIZ_SEARCH_SIZE,
    balance: bool = True,
) -> Tuple[Dict[NodeId, int], List[GraphvizVirtualEdge]]:
    """Assign dot ranks and describe virtual-node chains for long edges.

    Parameters
    ----------
    edges : torch.Tensor or sequence
        Directed rank constraints. A tensor must have shape ``[2, E]`` and
        uses integer node ids. Sequence entries may be ``(tail, head)``,
        ``(tail, head, minlen)``, or ``(tail, head, minlen, weight)``.
    virtual_node_factory : callable
        Factory used for dummy nodes on long edges. Dagua calls it with
        ``(tail, head, rank, original_edge_index)`` when supported, then falls
        back to fewer arguments for simple factories.
    num_nodes : int, optional
        Number of integer nodes. When supplied, isolated nodes ``0..N-1`` are
        included with rank zero.
    edge_minlens : sequence of int, optional
        Per-edge minimum rank spans for tensor input or two-tuple edges.
    edge_weights : torch.Tensor or sequence of float, optional
        Per-edge objective weights. Values are coerced to Graphviz-style ints.
    maxiter : int, optional
        Maximum network-simplex pivots per weak component. ``None`` uses a
        conservative finite bound.
    search_size : int, default=30
        Graphviz ``searchsize`` equivalent for selecting leaving tree edges.
    balance : bool, default=True
        Whether to apply dot's top-bottom balancing pass after optimization.

    Returns
    -------
    tuple
        ``(ranks, virtual_edges)`` where ``ranks`` maps original nodes to
        zero-based rank integers and ``virtual_edges`` describes every original
        edge whose final span is greater than one rank.

    Raises
    ------
    ValueError
        If edge shapes are invalid or the input still contains a directed
        cycle. Graphviz calls ``acyclic()`` before ranking; callers should do
        the same for cyclic graphs.
    """
    records = _normalize_edge_records(
        edges=edges,
        num_nodes=num_nodes,
        edge_minlens=edge_minlens,
        edge_weights=edge_weights,
    )
    ordered_nodes = _ordered_nodes(records=records, num_nodes=num_nodes)
    if not ordered_nodes:
        return {}, []

    ranks: Dict[NodeId, int] = {node: 0 for node in ordered_nodes}
    for component_nodes in _weak_components(records=records, ordered_nodes=ordered_nodes):
        component_records = [
            record
            for record in records
            if record.tail in component_nodes and record.head in component_nodes
        ]
        if not component_records:
            continue
        local_nodes = [node for node in ordered_nodes if node in component_nodes]
        local_ranks = _rank_component(
            records=component_records,
            ordered_nodes=local_nodes,
            maxiter=maxiter,
            search_size=search_size,
            balance_mode="tb" if balance else "none",
        )
        ranks.update(local_ranks)

    virtual_edges = _build_virtual_edges(
        records=records,
        ranks=ranks,
        virtual_node_factory=virtual_node_factory,
    )
    return ranks, virtual_edges


def graphviz_network_simplex_assignment(
    edges: Union[torch.Tensor, Sequence[EdgeLike]],
    num_nodes: Optional[int] = None,
    edge_minlens: Optional[Sequence[int]] = None,
    edge_weights: Optional[Union[torch.Tensor, Sequence[float]]] = None,
    initial_ranks: Optional[Dict[NodeId, int]] = None,
    maxiter: Optional[int] = None,
    search_size: int = _GRAPHVIZ_SEARCH_SIZE,
    balance_mode: BalanceMode = "tb",
) -> Dict[NodeId, int]:
    """Assign node ranks for arbitrary Graphviz network-simplex constraints.

    Parameters
    ----------
    edges : torch.Tensor or sequence
        Directed rank constraints. A tensor must have shape ``[2, E]`` and
        uses integer node ids. Sequence entries may be ``(tail, head)``,
        ``(tail, head, minlen)``, or ``(tail, head, minlen, weight)``.
    num_nodes : int, optional
        Number of integer nodes. When supplied, isolated nodes ``0..N-1`` are
        included with rank zero.
    edge_minlens : sequence of int, optional
        Per-edge minimum lengths for tensor input or two-tuple edges.
    edge_weights : torch.Tensor or sequence of float, optional
        Per-edge objective weights. Values are coerced to Graphviz-style ints.
    initial_ranks : dict, optional
        Optional starting ranks keyed by node id. Graphviz's x-coordinate
        simplex seeds ranks during auxiliary-graph construction before calling
        ``rank(g, 2, ...)``.
    maxiter : int, optional
        Maximum network-simplex pivots per weak component. ``None`` uses a
        conservative finite bound.
    search_size : int, default=30
        Graphviz ``searchsize`` equivalent for selecting leaving tree edges.
    balance_mode : {"none", "tb", "lr"}, default="tb"
        Post-optimal balancing pass. ``"lr"`` matches dot ``rank(g, 2, ...)``
        for horizontal coordinate assignment.

    Returns
    -------
    dict
        Original node id to simplex rank.

    Raises
    ------
    ValueError
        If edge shapes are invalid, ``balance_mode`` is unsupported, or the
        input still contains a directed cycle.
    """
    if balance_mode not in ("none", "tb", "lr"):
        raise ValueError("balance_mode must be 'none', 'tb', or 'lr'.")

    records = _normalize_edge_records(
        edges=edges,
        num_nodes=num_nodes,
        edge_minlens=edge_minlens,
        edge_weights=edge_weights,
    )
    ordered_nodes = _ordered_nodes(records=records, num_nodes=num_nodes)
    if not ordered_nodes:
        return {}

    ranks: Dict[NodeId, int] = {node: 0 for node in ordered_nodes}
    for component_nodes in _weak_components(records=records, ordered_nodes=ordered_nodes):
        component_records = [
            record
            for record in records
            if record.tail in component_nodes and record.head in component_nodes
        ]
        if not component_records:
            continue
        local_nodes = [node for node in ordered_nodes if node in component_nodes]
        local_ranks = _rank_component(
            records=component_records,
            ordered_nodes=local_nodes,
            maxiter=maxiter,
            search_size=search_size,
            balance_mode=balance_mode,
            initial_ranks=initial_ranks,
        )
        ranks.update(local_ranks)
    return ranks


def _normalize_edge_records(
    edges: Union[torch.Tensor, Sequence[EdgeLike]],
    num_nodes: Optional[int],
    edge_minlens: Optional[Sequence[int]],
    edge_weights: Optional[Union[torch.Tensor, Sequence[float]]],
) -> List[_EdgeRecord]:
    """Convert public edge inputs into Graphviz rank constraints.

    Parameters
    ----------
    edges : torch.Tensor or sequence
        Public edge representation.
    num_nodes : int, optional
        Number of integer nodes for validation.
    edge_minlens : sequence of int, optional
        Explicit minimum lengths.
    edge_weights : torch.Tensor or sequence of float, optional
        Explicit edge weights.

    Returns
    -------
    list of _EdgeRecord
        Normalized, non-self-loop rank constraints in input order.
    """
    weights = _coerce_weight_values(edge_weights=edge_weights)
    minlens = list(edge_minlens) if edge_minlens is not None else None
    records: List[_EdgeRecord] = []
    if isinstance(edges, torch.Tensor):
        if edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError("edges tensor must have shape [2, E]")
        edge_count = int(edges.shape[1])
        if minlens is not None and len(minlens) != edge_count:
            raise ValueError("edge_minlens length must match edge count")
        if weights is not None and len(weights) != edge_count:
            raise ValueError("edge_weights length must match edge count")
        edge_cpu = edges.detach().to(device="cpu", dtype=torch.long)
        for edge_index, (tail, head) in enumerate(zip(edge_cpu[0].tolist(), edge_cpu[1].tolist())):
            if tail == head:
                continue
            _validate_integer_node(node=tail, num_nodes=num_nodes)
            _validate_integer_node(node=head, num_nodes=num_nodes)
            records.append(
                _EdgeRecord(
                    tail=int(tail),
                    head=int(head),
                    minlen=_coerce_minlen(minlens[edge_index] if minlens is not None else 1),
                    weight=_coerce_weight(weights[edge_index] if weights is not None else 1),
                    original_index=edge_index,
                )
            )
        return records

    edge_sequence = list(edges)
    if minlens is not None and len(minlens) != len(edge_sequence):
        raise ValueError("edge_minlens length must match edge count")
    if weights is not None and len(weights) != len(edge_sequence):
        raise ValueError("edge_weights length must match edge count")
    for edge_index, raw_edge in enumerate(edge_sequence):
        if len(raw_edge) < 2 or len(raw_edge) > 4:
            raise ValueError("edge tuples must have length 2, 3, or 4")
        tail = raw_edge[0]
        head = raw_edge[1]
        if tail == head:
            continue
        tuple_minlen = raw_edge[2] if len(raw_edge) >= 3 else 1
        tuple_weight = raw_edge[3] if len(raw_edge) >= 4 else 1
        records.append(
            _EdgeRecord(
                tail=tail,
                head=head,
                minlen=_coerce_minlen(minlens[edge_index] if minlens is not None else tuple_minlen),
                weight=_coerce_weight(weights[edge_index] if weights is not None else tuple_weight),
                original_index=edge_index,
            )
        )
    return records


def _coerce_weight_values(
    edge_weights: Optional[Union[torch.Tensor, Sequence[float]]],
) -> Optional[List[float]]:
    """Return edge weights as a Python list when present.

    Parameters
    ----------
    edge_weights : torch.Tensor or sequence of float, optional
        Public edge weights.

    Returns
    -------
    list of float or None
        CPU scalar weights, preserving edge order.
    """
    if edge_weights is None:
        return None
    if isinstance(edge_weights, torch.Tensor):
        return [float(value) for value in edge_weights.detach().to(device="cpu").tolist()]
    return [float(value) for value in edge_weights]


def _coerce_minlen(value: Any) -> int:
    """Coerce a Graphviz ``minlen`` value.

    Parameters
    ----------
    value : Any
        Candidate minimum rank span.

    Returns
    -------
    int
        Non-negative integer minimum length.
    """
    return max(int(value), 0)


def _coerce_weight(value: Any) -> int:
    """Coerce a Graphviz edge ``weight`` value.

    Parameters
    ----------
    value : Any
        Candidate objective weight.

    Returns
    -------
    int
        Non-negative integer edge weight.
    """
    return max(int(value), 0)


def _validate_integer_node(node: int, num_nodes: Optional[int]) -> None:
    """Validate an integer node id for tensor input.

    Parameters
    ----------
    node : int
        Node id found in the edge tensor.
    num_nodes : int, optional
        Number of allowed integer nodes.

    Raises
    ------
    ValueError
        If the node id is out of range.
    """
    if num_nodes is not None and (node < 0 or node >= num_nodes):
        raise ValueError("edge endpoint is outside num_nodes")


def _ordered_nodes(records: Sequence[_EdgeRecord], num_nodes: Optional[int]) -> List[NodeId]:
    """Return nodes in deterministic Graphviz input order.

    Parameters
    ----------
    records : sequence of _EdgeRecord
        Normalized rank constraints.
    num_nodes : int, optional
        Optional integer node count.

    Returns
    -------
    list of Hashable
        Ordered original node ids.
    """
    ordered: List[NodeId] = []
    seen: set[NodeId] = set()
    if num_nodes is not None:
        for node in range(num_nodes):
            ordered.append(node)
            seen.add(node)
    for record in records:
        for node in (record.tail, record.head):
            if node not in seen:
                ordered.append(node)
                seen.add(node)
    return ordered


def _weak_components(
    records: Sequence[_EdgeRecord],
    ordered_nodes: Sequence[NodeId],
) -> List[set[NodeId]]:
    """Compute weak components in node-list order.

    Parameters
    ----------
    records : sequence of _EdgeRecord
        Normalized rank constraints.
    ordered_nodes : sequence of Hashable
        Graph nodes in deterministic order.

    Returns
    -------
    list of set
        Weak components.
    """
    adjacency: Dict[NodeId, List[NodeId]] = {node: [] for node in ordered_nodes}
    for record in records:
        adjacency[record.tail].append(record.head)
        adjacency[record.head].append(record.tail)

    components: List[set[NodeId]] = []
    seen: set[NodeId] = set()
    for start in ordered_nodes:
        if start in seen:
            continue
        seen.add(start)
        stack = [start]
        component: set[NodeId] = set()
        while stack:
            node = stack.pop()
            component.add(node)
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def _rank_component(
    records: Sequence[_EdgeRecord],
    ordered_nodes: Sequence[NodeId],
    maxiter: Optional[int],
    search_size: int,
    balance_mode: BalanceMode,
    initial_ranks: Optional[Dict[NodeId, int]] = None,
) -> Dict[NodeId, int]:
    """Run network simplex on one weak component.

    Parameters
    ----------
    records : sequence of _EdgeRecord
        Component-local rank constraints.
    ordered_nodes : sequence of Hashable
        Component nodes in Graphviz input order.
    maxiter : int, optional
        Pivot cap.
    search_size : int
        Leaving-edge search window.
    balance_mode : {"none", "tb", "lr"}
        Post-optimal balancing pass.
    initial_ranks : dict, optional
        Optional starting ranks keyed by original node id.

    Returns
    -------
    dict
        Original node id to rank.
    """
    node_to_local = {node: index for index, node in enumerate(ordered_nodes)}
    graph = _build_simplex_graph(
        records=records,
        node_to_local=node_to_local,
        search_size=search_size,
    )
    if initial_ranks is not None:
        for node, local_id in node_to_local.items():
            graph.nodes[local_id].rank = int(initial_ranks.get(node, 0))
    if len(graph.nodes) == 1:
        return {ordered_nodes[0]: 0}
    _run_network_simplex(graph=graph, balance_mode=balance_mode, maxiter=maxiter)
    return {node: graph.nodes[node_to_local[node]].rank for node in ordered_nodes}


def _build_simplex_graph(
    records: Sequence[_EdgeRecord],
    node_to_local: Dict[NodeId, int],
    search_size: int,
) -> _SimplexGraph:
    """Build mutable simplex records.

    Parameters
    ----------
    records : sequence of _EdgeRecord
        Component-local rank constraints.
    node_to_local : dict
        Mapping from original node ids to component-local indices.
    search_size : int
        Leaving-edge search window.

    Returns
    -------
    _SimplexGraph
        Mutable graph ready for rank initialization.
    """
    nodes = [_NSNode() for _ in range(len(node_to_local))]
    graph = _SimplexGraph(nodes=nodes, edges=[], search_size=max(1, int(search_size)))
    for record in records:
        edge_id = len(graph.edges)
        tail = node_to_local[record.tail]
        head = node_to_local[record.head]
        graph.edges.append(
            _NSEdge(
                tail=tail,
                head=head,
                minlen=record.minlen,
                weight=record.weight,
                original_index=record.original_index,
            )
        )
        graph.nodes[tail].out_edges.append(edge_id)
        graph.nodes[head].in_edges.append(edge_id)
    return graph


def _run_network_simplex(
    graph: _SimplexGraph,
    balance_mode: BalanceMode,
    maxiter: Optional[int],
) -> None:
    """Apply Graphviz's network-simplex rank optimizer.

    Parameters
    ----------
    graph : _SimplexGraph
        Mutable rank graph.
    balance_mode : {"none", "tb", "lr"}
        Post-optimal balancing pass to apply after pivots.
    maxiter : int, optional
        Maximum number of pivots.

    Raises
    ------
    ValueError
        If the graph is cyclic or the initial tight tree cannot be built.
    """
    feasible = _init_graph(graph=graph)
    if not feasible:
        _init_rank(graph=graph)
    _feasible_tree(graph=graph)
    pivot_limit = (
        maxiter
        if maxiter is not None
        else max(1000, 4 * max(len(graph.nodes), 1) * max(len(graph.edges), 1))
    )
    for _ in range(max(0, int(pivot_limit))):
        leaving = _leave_edge(graph=graph)
        if leaving is None:
            break
        entering = _enter_edge(graph=graph, edge_id=leaving)
        if entering is None:
            break
        _update(graph=graph, leaving=leaving, entering=entering)
    if balance_mode == "tb":
        _top_bottom_balance(graph=graph)
    elif balance_mode == "lr":
        _left_right_balance(graph=graph)
    else:
        _scan_and_normalize(graph=graph)


def _init_graph(graph: _SimplexGraph) -> bool:
    """Initialize per-edge and per-node simplex fields.

    Parameters
    ----------
    graph : _SimplexGraph
        Mutable rank graph.

    Returns
    -------
    bool
        ``True`` when current ranks already satisfy all minlen constraints.
    """
    feasible = True
    graph.tree_edges.clear()
    graph.search_index = 0
    for node in graph.nodes:
        node.priority = 0
        node.tree_in.clear()
        node.tree_out.clear()
        node.par = None
        node.low = 0
        node.lim = 0
        node.subtree = -1
    for edge_id, edge in enumerate(graph.edges):
        edge.cutvalue = 0
        edge.tree_index = _NO_TREE_INDEX
        graph.nodes[edge.head].priority += 1
        if _length(graph=graph, edge_id=edge_id) < edge.minlen:
            feasible = False
    return feasible


def _init_rank(graph: _SimplexGraph) -> None:
    """Initialize feasible ranks using Graphviz's longest-path pass.

    Parameters
    ----------
    graph : _SimplexGraph
        Mutable rank graph.

    Raises
    ------
    ValueError
        If the input still contains a directed cycle.
    """
    queue = [index for index, node in enumerate(graph.nodes) if node.priority == 0]
    cursor = 0
    processed = 0
    while cursor < len(queue):
        node_id = queue[cursor]
        cursor += 1
        node = graph.nodes[node_id]
        node.rank = 0
        processed += 1
        for edge_id in node.in_edges:
            edge = graph.edges[edge_id]
            node.rank = max(node.rank, graph.nodes[edge.tail].rank + edge.minlen)
        for edge_id in node.out_edges:
            head = graph.edges[edge_id].head
            graph.nodes[head].priority -= 1
            if graph.nodes[head].priority <= 0:
                queue.append(head)
    if processed != len(graph.nodes):
        raise ValueError("graphviz rank assignment requires acyclic input")


def _feasible_tree(graph: _SimplexGraph) -> None:
    """Construct the initial tight spanning tree.

    Parameters
    ----------
    graph : _SimplexGraph
        Mutable rank graph.

    Raises
    ------
    ValueError
        If no inter-tree edge exists while merging tight subtrees.
    """
    subtrees = _find_tight_subtrees(graph=graph)
    if not subtrees:
        return
    active = {index for index in range(len(subtrees))}
    while len(active) > 1:
        extracted = min(active, key=lambda idx: (subtrees[idx].size, idx))
        active.remove(extracted)
        subtrees[extracted].heap_index = -1
        entering = _inter_tree_edge(graph=graph, subtrees=subtrees, subtree_id=extracted)
        if entering is None:
            raise ValueError("graphviz rank assignment could not connect tight subtrees")
        merged = _merge_trees(
            graph=graph,
            subtrees=subtrees,
            edge_id=entering,
            active=active,
        )
        active.add(merged)
        subtrees[merged].heap_index = len(active) - 1
    _dfs_range_init(graph=graph)
    _dfs_cutval(graph=graph)


def _find_tight_subtrees(graph: _SimplexGraph) -> List[_Subtree]:
    """Find initial tight subtrees and add their tree edges.

    Parameters
    ----------
    graph : _SimplexGraph
        Mutable rank graph.

    Returns
    -------
    list of _Subtree
        Initial tight subtrees.
    """
    subtrees: List[_Subtree] = []
    for node_id, node in enumerate(graph.nodes):
        if node.subtree != -1:
            continue
        subtree_id = len(subtrees)
        node.subtree = subtree_id
        size = _tight_subtree_search(graph=graph, start=node_id, subtree_id=subtree_id)
        subtrees.append(_Subtree(rep=node_id, size=size, parent=subtree_id, heap_index=subtree_id))
    return subtrees


def _tight_subtree_search(graph: _SimplexGraph, start: int, subtree_id: int) -> int:
    """Depth-first search over zero-slack edges.

    Parameters
    ----------
    graph : _SimplexGraph
        Mutable rank graph.
    start : int
        Root node id.
    subtree_id : int
        Subtree label to assign.

    Returns
    -------
    int
        Number of nodes in the tight subtree.
    """
    size = 0
    stack = [start]
    while stack:
        node_id = stack.pop()
        size += 1
        node = graph.nodes[node_id]
        for edge_id in node.in_edges:
            edge = graph.edges[edge_id]
            if _is_tree_edge(graph=graph, edge_id=edge_id):
                continue
            if graph.nodes[edge.tail].subtree == -1 and _slack(graph=graph, edge_id=edge_id) == 0:
                _add_tree_edge(graph=graph, edge_id=edge_id)
                graph.nodes[edge.tail].subtree = subtree_id
                stack.append(edge.tail)
        for edge_id in node.out_edges:
            edge = graph.edges[edge_id]
            if _is_tree_edge(graph=graph, edge_id=edge_id):
                continue
            if graph.nodes[edge.head].subtree == -1 and _slack(graph=graph, edge_id=edge_id) == 0:
                _add_tree_edge(graph=graph, edge_id=edge_id)
                graph.nodes[edge.head].subtree = subtree_id
                stack.append(edge.head)
    return size


def _inter_tree_edge(
    graph: _SimplexGraph,
    subtrees: List[_Subtree],
    subtree_id: int,
) -> Optional[int]:
    """Find the tightest edge incident to another subtree.

    Parameters
    ----------
    graph : _SimplexGraph
        Mutable rank graph.
    subtrees : list of _Subtree
        Union-find subtree records.
    subtree_id : int
        Root subtree to scan.

    Returns
    -------
    int or None
        Best non-tree edge id.
    """
    root = _find_subtree(subtrees=subtrees, subtree_id=subtree_id)
    best: Optional[int] = None
    seen: set[int] = set()
    stack = [subtrees[root].rep]
    while stack:
        node_id = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        node = graph.nodes[node_id]
        for edge_id in node.out_edges:
            edge = graph.edges[edge_id]
            if _is_tree_edge(graph=graph, edge_id=edge_id):
                stack.append(edge.head)
            elif (
                _find_subtree(subtrees=subtrees, subtree_id=graph.nodes[edge.head].subtree) != root
            ):
                if _is_better_entering_edge(graph=graph, edge_id=edge_id, best=best):
                    best = edge_id
        for edge_id in node.in_edges:
            edge = graph.edges[edge_id]
            if _is_tree_edge(graph=graph, edge_id=edge_id):
                stack.append(edge.tail)
            elif (
                _find_subtree(subtrees=subtrees, subtree_id=graph.nodes[edge.tail].subtree) != root
            ):
                if _is_better_entering_edge(graph=graph, edge_id=edge_id, best=best):
                    best = edge_id
    return best


def _merge_trees(
    graph: _SimplexGraph,
    subtrees: List[_Subtree],
    edge_id: int,
    active: set[int],
) -> int:
    """Merge two tight subtrees through an entering edge.

    Parameters
    ----------
    graph : _SimplexGraph
        Mutable rank graph.
    subtrees : list of _Subtree
        Union-find subtree records.
    edge_id : int
        Non-tree edge used for the merge.
    active : set of int
        Subtree ids still participating in the heap.

    Returns
    -------
    int
        Active root subtree after the merge.
    """
    edge = graph.edges[edge_id]
    tail_tree = _find_subtree(subtrees=subtrees, subtree_id=graph.nodes[edge.tail].subtree)
    head_tree = _find_subtree(subtrees=subtrees, subtree_id=graph.nodes[edge.head].subtree)
    slack = _slack(graph=graph, edge_id=edge_id)
    if tail_tree not in active:
        if slack:
            _tree_adjust(graph=graph, node_id=subtrees[tail_tree].rep, from_node=None, delta=slack)
        root = head_tree
        child = tail_tree
    else:
        if slack:
            _tree_adjust(graph=graph, node_id=subtrees[head_tree].rep, from_node=None, delta=-slack)
        root = tail_tree
        child = head_tree
    _add_tree_edge(graph=graph, edge_id=edge_id)
    subtrees[child].parent = root
    subtrees[root].parent = root
    subtrees[root].size += subtrees[child].size
    return root


def _is_better_entering_edge(
    graph: _SimplexGraph,
    edge_id: int,
    best: Optional[int],
) -> bool:
    """Return whether an edge improves the current entering-edge choice.

    Parameters
    ----------
    graph : _SimplexGraph
        Mutable rank graph.
    edge_id : int
        Candidate edge id.
    best : int, optional
        Current best edge id.

    Returns
    -------
    bool
        ``True`` when ``edge_id`` has smaller slack than ``best`` or no best
        edge has been selected yet.
    """
    if best is None:
        return True
    return _slack(graph=graph, edge_id=edge_id) < _slack(graph=graph, edge_id=best)


def _find_subtree(subtrees: List[_Subtree], subtree_id: int) -> int:
    """Find a subtree union-find root.

    Parameters
    ----------
    subtrees : list of _Subtree
        Union-find records.
    subtree_id : int
        Subtree id to resolve.

    Returns
    -------
    int
        Root subtree id.
    """
    path: List[int] = []
    current = subtree_id
    while subtrees[current].parent != current:
        path.append(current)
        current = subtrees[current].parent
    for path_id in path:
        subtrees[path_id].parent = current
    return current


def _tree_adjust(
    graph: _SimplexGraph,
    node_id: int,
    from_node: Optional[int],
    delta: int,
) -> None:
    """Add ``delta`` to every rank in a tree component.

    Parameters
    ----------
    graph : _SimplexGraph
        Mutable rank graph.
    node_id : int
        Tree traversal root.
    from_node : int, optional
        Previous node to avoid traversing back.
    delta : int
        Rank delta.
    """
    stack: List[Tuple[int, Optional[int]]] = [(node_id, from_node)]
    while stack:
        current, previous = stack.pop()
        graph.nodes[current].rank += delta
        for edge_id in reversed(graph.nodes[current].tree_out):
            head = graph.edges[edge_id].head
            if head != previous:
                stack.append((head, current))
        for edge_id in reversed(graph.nodes[current].tree_in):
            tail = graph.edges[edge_id].tail
            if tail != previous:
                stack.append((tail, current))


def _add_tree_edge(graph: _SimplexGraph, edge_id: int) -> None:
    """Add a non-tree edge to the current spanning tree.

    Parameters
    ----------
    graph : _SimplexGraph
        Mutable rank graph.
    edge_id : int
        Edge to add.
    """
    edge = graph.edges[edge_id]
    edge.tree_index = len(graph.tree_edges)
    graph.tree_edges.append(edge_id)
    graph.nodes[edge.tail].tree_out.append(edge_id)
    graph.nodes[edge.head].tree_in.append(edge_id)


def _exchange_tree_edges(graph: _SimplexGraph, leaving: int, entering: int) -> None:
    """Replace one tree edge with another.

    Parameters
    ----------
    graph : _SimplexGraph
        Mutable rank graph.
    leaving : int
        Current tree edge id.
    entering : int
        Non-tree edge id.
    """
    leaving_edge = graph.edges[leaving]
    entering_edge = graph.edges[entering]
    tree_index = leaving_edge.tree_index
    graph.tree_edges[tree_index] = entering
    leaving_edge.tree_index = _NO_TREE_INDEX
    entering_edge.tree_index = tree_index
    graph.nodes[leaving_edge.tail].tree_out.remove(leaving)
    graph.nodes[leaving_edge.head].tree_in.remove(leaving)
    graph.nodes[entering_edge.tail].tree_out.append(entering)
    graph.nodes[entering_edge.head].tree_in.append(entering)


def _dfs_range_init(graph: _SimplexGraph) -> None:
    """Initialize DFS low/lim intervals on the current tree.

    Parameters
    ----------
    graph : _SimplexGraph
        Mutable rank graph.
    """
    if not graph.nodes:
        return
    _dfs_range(graph=graph, node_id=0, parent_edge=None, low=1, reuse_clean=False)


def _dfs_range(
    graph: _SimplexGraph,
    node_id: int,
    parent_edge: Optional[int],
    low: int,
    reuse_clean: bool,
) -> int:
    """Assign DFS low/lim intervals over the current tree.

    Parameters
    ----------
    graph : _SimplexGraph
        Mutable rank graph.
    node_id : int
        Node to visit.
    parent_edge : int, optional
        Parent tree edge.
    low : int
        Low DFS index.
    reuse_clean : bool
        Whether unchanged ``(parent_edge, low)`` intervals may be reused.

    Returns
    -------
    int
        Next DFS index after this subtree.
    """
    root = graph.nodes[node_id]
    if reuse_clean and root.par == parent_edge and root.low == low:
        return root.lim + 1

    root.par = parent_edge
    root.low = low
    stack: List[Tuple[int, Optional[int], int, int, List[Tuple[int, int]]]] = [
        (
            node_id,
            parent_edge,
            0,
            low,
            _dfs_range_children(graph=graph, node_id=node_id, parent_edge=parent_edge),
        )
    ]
    while stack:
        current, _current_parent, child_index, lim, children = stack[-1]
        if child_index < len(children):
            edge_id, child_id = children[child_index]
            child = graph.nodes[child_id]
            if reuse_clean and child.par == edge_id and child.low == lim:
                stack[-1] = (
                    current,
                    _current_parent,
                    child_index + 1,
                    child.lim + 1,
                    children,
                )
                continue
            stack[-1] = (current, _current_parent, child_index + 1, lim, children)
            child.par = edge_id
            child.low = lim
            stack.append(
                (
                    child_id,
                    edge_id,
                    0,
                    lim,
                    _dfs_range_children(graph=graph, node_id=child_id, parent_edge=edge_id),
                )
            )
            continue

        graph.nodes[current].lim = lim
        next_low = lim + 1
        stack.pop()
        if stack:
            parent_id, grandparent, parent_index, _, parent_children = stack[-1]
            stack[-1] = (
                parent_id,
                grandparent,
                parent_index,
                next_low,
                parent_children,
            )
        else:
            return next_low
    return low


def _dfs_range_children(
    graph: _SimplexGraph,
    node_id: int,
    parent_edge: Optional[int],
) -> List[Tuple[int, int]]:
    """Return tree children in Graphviz DFS traversal order.

    Parameters
    ----------
    graph : _SimplexGraph
        Mutable rank graph.
    node_id : int
        Tree node whose children should be listed.
    parent_edge : int, optional
        Parent tree edge to skip.

    Returns
    -------
    list of tuple[int, int]
        ``(edge_id, child_node_id)`` pairs with tree-out edges before tree-in
        edges, matching Graphviz ``dfs_range_init`` and ``dfs_cutval`` order.
    """
    node = graph.nodes[node_id]
    children: List[Tuple[int, int]] = []
    for edge_id in node.tree_out:
        if edge_id != parent_edge:
            children.append((edge_id, graph.edges[edge_id].head))
    for edge_id in node.tree_in:
        if edge_id != parent_edge:
            children.append((edge_id, graph.edges[edge_id].tail))
    return children


def _dfs_cutval(graph: _SimplexGraph) -> None:
    """Compute cut values for all tree edges.

    Parameters
    ----------
    graph : _SimplexGraph
        Mutable rank graph.
    """
    if not graph.nodes:
        return

    stack: List[Tuple[int, Optional[int], int, List[Tuple[int, int]]]] = [
        (0, None, 0, _dfs_range_children(graph=graph, node_id=0, parent_edge=None))
    ]
    while stack:
        node_id, parent_edge, child_index, children = stack[-1]
        if child_index < len(children):
            edge_id, child_id = children[child_index]
            stack[-1] = (node_id, parent_edge, child_index + 1, children)
            stack.append(
                (
                    child_id,
                    edge_id,
                    0,
                    _dfs_range_children(
                        graph=graph,
                        node_id=child_id,
                        parent_edge=edge_id,
                    ),
                )
            )
            continue

        stack.pop()
        if parent_edge is not None:
            _x_cutval(graph=graph, edge_id=parent_edge)


def _x_cutval(graph: _SimplexGraph, edge_id: int) -> None:
    """Set a tree edge cut value from one already-searched side.

    Parameters
    ----------
    graph : _SimplexGraph
        Mutable rank graph.
    edge_id : int
        Tree edge id.
    """
    edge = graph.edges[edge_id]
    if graph.nodes[edge.tail].par == edge_id:
        node_id = edge.tail
        direction = 1
    else:
        node_id = edge.head
        direction = -1
    total = 0
    for out_edge in graph.nodes[node_id].out_edges:
        total += _x_val(graph=graph, edge_id=out_edge, node_id=node_id, direction=direction)
    for in_edge in graph.nodes[node_id].in_edges:
        total += _x_val(graph=graph, edge_id=in_edge, node_id=node_id, direction=direction)
    edge.cutvalue = total


def _x_val(graph: _SimplexGraph, edge_id: int, node_id: int, direction: int) -> int:
    """Return one edge's cut-value contribution.

    Parameters
    ----------
    graph : _SimplexGraph
        Mutable rank graph.
    edge_id : int
        Edge to evaluate.
    node_id : int
        Node on the searched side.
    direction : int
        Direction multiplier from Graphviz ``x_val``.

    Returns
    -------
    int
        Signed cut-value contribution.
    """
    edge = graph.edges[edge_id]
    other = edge.head if edge.tail == node_id else edge.tail
    node = graph.nodes[node_id]
    if not _seq(node.low, graph.nodes[other].lim, node.lim):
        flag = 1
        value = edge.weight
    else:
        flag = 0
        value = edge.cutvalue if _is_tree_edge(graph=graph, edge_id=edge_id) else 0
        value -= edge.weight
    if direction > 0:
        sign = 1 if edge.head == node_id else -1
    else:
        sign = 1 if edge.tail == node_id else -1
    if flag:
        sign = -sign
    return -value if sign < 0 else value


def _leave_edge(graph: _SimplexGraph) -> Optional[int]:
    """Select a negative-cut tree edge to leave the basis.

    Parameters
    ----------
    graph : _SimplexGraph
        Mutable rank graph.

    Returns
    -------
    int or None
        Leaving tree edge id.
    """
    if not graph.tree_edges:
        return None
    best: Optional[int] = None
    count = 0
    start = graph.search_index
    checked = 0
    while checked < len(graph.tree_edges):
        edge_id = graph.tree_edges[graph.search_index]
        edge = graph.edges[edge_id]
        if edge.cutvalue < 0:
            if best is None or graph.edges[best].cutvalue > edge.cutvalue:
                best = edge_id
            count += 1
            if count >= graph.search_size:
                return best
        graph.search_index = (graph.search_index + 1) % len(graph.tree_edges)
        checked += 1
        if graph.search_index == start and checked >= len(graph.tree_edges):
            break
    return best


def _enter_edge(graph: _SimplexGraph, edge_id: int) -> Optional[int]:
    """Find a non-tree edge to enter for a leaving tree edge.

    Parameters
    ----------
    graph : _SimplexGraph
        Mutable rank graph.
    edge_id : int
        Leaving tree edge id.

    Returns
    -------
    int or None
        Entering non-tree edge id.
    """
    edge = graph.edges[edge_id]
    if graph.nodes[edge.tail].lim < graph.nodes[edge.head].lim:
        return _dfs_enter_inedge(
            graph=graph,
            node_id=edge.tail,
            low=graph.nodes[edge.tail].low,
            lim=graph.nodes[edge.tail].lim,
        )
    return _dfs_enter_outedge(
        graph=graph,
        node_id=edge.head,
        low=graph.nodes[edge.head].low,
        lim=graph.nodes[edge.head].lim,
    )


def _dfs_enter_outedge(
    graph: _SimplexGraph,
    node_id: int,
    low: int,
    lim: int,
) -> Optional[int]:
    """Search a tree subtree for the tightest outgoing entering edge.

    Parameters
    ----------
    graph : _SimplexGraph
        Mutable rank graph.
    node_id : int
        Subtree root.
    low : int
        Subtree low DFS index.
    lim : int
        Subtree high DFS index.

    Returns
    -------
    int or None
        Entering edge id.
    """
    best: Optional[int] = None
    stack = [node_id]
    while stack:
        current = stack.pop()
        for edge_id in graph.nodes[current].out_edges:
            edge = graph.edges[edge_id]
            if not _is_tree_edge(graph=graph, edge_id=edge_id):
                if not _seq(low, graph.nodes[edge.head].lim, lim):
                    if _is_better_entering_edge(graph=graph, edge_id=edge_id, best=best):
                        best = edge_id
            elif graph.nodes[edge.head].lim < graph.nodes[current].lim:
                stack.append(edge.head)
        if best is not None and _slack(graph=graph, edge_id=best) == 0:
            continue
        for edge_id in graph.nodes[current].tree_in:
            edge = graph.edges[edge_id]
            if graph.nodes[edge.tail].lim < graph.nodes[current].lim:
                stack.append(edge.tail)
    return best


def _dfs_enter_inedge(
    graph: _SimplexGraph,
    node_id: int,
    low: int,
    lim: int,
) -> Optional[int]:
    """Search a tree subtree for the tightest incoming entering edge.

    Parameters
    ----------
    graph : _SimplexGraph
        Mutable rank graph.
    node_id : int
        Subtree root.
    low : int
        Subtree low DFS index.
    lim : int
        Subtree high DFS index.

    Returns
    -------
    int or None
        Entering edge id.
    """
    best: Optional[int] = None
    stack = [node_id]
    while stack:
        current = stack.pop()
        for edge_id in graph.nodes[current].in_edges:
            edge = graph.edges[edge_id]
            if not _is_tree_edge(graph=graph, edge_id=edge_id):
                if not _seq(low, graph.nodes[edge.tail].lim, lim):
                    if _is_better_entering_edge(graph=graph, edge_id=edge_id, best=best):
                        best = edge_id
            elif graph.nodes[edge.tail].lim < graph.nodes[current].lim:
                stack.append(edge.tail)
        if best is not None and _slack(graph=graph, edge_id=best) == 0:
            continue
        for edge_id in graph.nodes[current].tree_out:
            edge = graph.edges[edge_id]
            if graph.nodes[edge.head].lim < graph.nodes[current].lim:
                stack.append(edge.head)
    return best


def _update(graph: _SimplexGraph, leaving: int, entering: int) -> None:
    """Pivot the simplex tree.

    Parameters
    ----------
    graph : _SimplexGraph
        Mutable rank graph.
    leaving : int
        Leaving tree edge id.
    entering : int
        Entering non-tree edge id.
    """
    delta = _slack(graph=graph, edge_id=entering)
    if delta > 0:
        leaving_edge = graph.edges[leaving]
        tail_degree = len(graph.nodes[leaving_edge.tail].tree_in) + len(
            graph.nodes[leaving_edge.tail].tree_out
        )
        head_degree = len(graph.nodes[leaving_edge.head].tree_in) + len(
            graph.nodes[leaving_edge.head].tree_out
        )
        if tail_degree == 1:
            _rerank(graph=graph, node_id=leaving_edge.tail, parent_edge=leaving, delta=delta)
        elif head_degree == 1:
            _rerank(graph=graph, node_id=leaving_edge.head, parent_edge=leaving, delta=-delta)
        elif graph.nodes[leaving_edge.tail].lim < graph.nodes[leaving_edge.head].lim:
            _rerank(graph=graph, node_id=leaving_edge.tail, parent_edge=leaving, delta=delta)
        else:
            _rerank(graph=graph, node_id=leaving_edge.head, parent_edge=leaving, delta=-delta)
    cutvalue = graph.edges[leaving].cutvalue
    entering_edge = graph.edges[entering]
    lca = _tree_update(
        graph=graph,
        node_id=entering_edge.tail,
        target_node=entering_edge.head,
        cutvalue=cutvalue,
        direction=1,
    )
    other_lca = _tree_update(
        graph=graph,
        node_id=entering_edge.head,
        target_node=entering_edge.tail,
        cutvalue=cutvalue,
        direction=0,
    )
    if lca != other_lca:
        raise ValueError("graphviz rank assignment tree update found mismatched LCA")
    lca_low = graph.nodes[lca].low
    _invalidate_path(graph=graph, lca=lca, node_id=entering_edge.head)
    _invalidate_path(graph=graph, lca=lca, node_id=entering_edge.tail)
    graph.edges[entering].cutvalue = -cutvalue
    graph.edges[leaving].cutvalue = 0
    _exchange_tree_edges(graph=graph, leaving=leaving, entering=entering)
    _dfs_range(
        graph=graph,
        node_id=lca,
        parent_edge=graph.nodes[lca].par,
        low=lca_low,
        reuse_clean=True,
    )


def _tree_update(
    graph: _SimplexGraph,
    node_id: int,
    target_node: int,
    cutvalue: int,
    direction: int,
) -> int:
    """Update cut values while walking one endpoint toward the LCA.

    Parameters
    ----------
    graph : _SimplexGraph
        Mutable rank graph.
    node_id : int
        Endpoint node to walk upward through the old tree.
    target_node : int
        Opposite entering-edge endpoint.
    cutvalue : int
        Cut value of the leaving tree edge.
    direction : int
        Graphviz ``treeupdate`` direction flag, either ``1`` or ``0``.

    Returns
    -------
    int
        Lowest common ancestor of ``node_id`` and ``target_node`` in the old
        tree intervals.
    """
    target_lim = graph.nodes[target_node].lim
    current = node_id
    while not _seq(graph.nodes[current].low, target_lim, graph.nodes[current].lim):
        parent_edge = graph.nodes[current].par
        if parent_edge is None:
            return current
        edge = graph.edges[parent_edge]
        if current == edge.tail:
            add_cutvalue = bool(direction)
        else:
            add_cutvalue = not bool(direction)
        if add_cutvalue:
            edge.cutvalue += cutvalue
        else:
            edge.cutvalue -= cutvalue
        current = (
            edge.tail if graph.nodes[edge.tail].lim > graph.nodes[edge.head].lim else edge.head
        )
    return current


def _invalidate_path(graph: _SimplexGraph, lca: int, node_id: int) -> None:
    """Invalidate cached DFS lows from one entering endpoint to the LCA.

    Parameters
    ----------
    graph : _SimplexGraph
        Mutable rank graph.
    lca : int
        Lowest common ancestor in the old tree intervals.
    node_id : int
        Endpoint node whose path to ``lca`` should be invalidated.
    """
    current = node_id
    while True:
        node = graph.nodes[current]
        if node.low == -1:
            break
        node.low = -1
        parent_edge = node.par
        if parent_edge is None:
            break
        if node.lim >= graph.nodes[lca].lim:
            break
        edge = graph.edges[parent_edge]
        current = (
            edge.tail if graph.nodes[edge.tail].lim > graph.nodes[edge.head].lim else edge.head
        )


def _rerank(graph: _SimplexGraph, node_id: int, parent_edge: int, delta: int) -> None:
    """Apply Graphviz ``rerank`` to one side of a tree edge.

    Parameters
    ----------
    graph : _SimplexGraph
        Mutable rank graph.
    node_id : int
        Traversal root.
    parent_edge : int
        Edge not to cross.
    delta : int
        Amount subtracted from each rank.
    """
    stack: List[Tuple[int, int]] = [(node_id, parent_edge)]
    while stack:
        current, previous_edge = stack.pop()
        graph.nodes[current].rank -= delta
        for edge_id in reversed(graph.nodes[current].tree_in):
            if edge_id != previous_edge:
                stack.append((graph.edges[edge_id].tail, edge_id))
        for edge_id in reversed(graph.nodes[current].tree_out):
            if edge_id != previous_edge:
                stack.append((graph.edges[edge_id].head, edge_id))


def _scan_and_normalize(graph: _SimplexGraph) -> int:
    """Normalize ranks so the minimum rank is zero.

    Parameters
    ----------
    graph : _SimplexGraph
        Mutable rank graph.

    Returns
    -------
    int
        Maximum normalized rank.
    """
    if not graph.nodes:
        return 0
    min_rank = min(node.rank for node in graph.nodes)
    max_rank = max(node.rank for node in graph.nodes)
    for node in graph.nodes:
        node.rank -= min_rank
    return max_rank - min_rank


def _top_bottom_balance(graph: _SimplexGraph) -> None:
    """Apply dot's top-bottom balancing pass.

    Parameters
    ----------
    graph : _SimplexGraph
        Mutable rank graph.
    """
    max_rank = _scan_and_normalize(graph=graph)
    rank_counts = [0] * (max_rank + 1)
    node_order = sorted(range(len(graph.nodes)), key=lambda node_id: graph.nodes[node_id].rank)
    for node_id in node_order:
        rank_counts[graph.nodes[node_id].rank] += 1
    for node_id in node_order:
        node = graph.nodes[node_id]
        inweight = 0
        outweight = 0
        low = 0
        high = max_rank
        for edge_id in node.in_edges:
            edge = graph.edges[edge_id]
            inweight += edge.weight
            low = max(low, graph.nodes[edge.tail].rank + edge.minlen)
        for edge_id in node.out_edges:
            edge = graph.edges[edge_id]
            outweight += edge.weight
            high = min(high, graph.nodes[edge.head].rank - edge.minlen)
        if low < 0:
            low = 0
        if inweight == outweight and low <= high:
            choice = low
            for rank in range(low + 1, high + 1):
                if rank_counts[rank] < rank_counts[choice]:
                    choice = rank
            rank_counts[node.rank] -= 1
            rank_counts[choice] += 1
            node.rank = choice


def _left_right_balance(graph: _SimplexGraph) -> None:
    """Apply dot's LR balance pass for horizontal coordinate simplex ranks.

    Parameters
    ----------
    graph : _SimplexGraph
        Mutable rank graph.

    Notes
    -----
    Graphviz 7.0.5 calls this path as ``rank(g, 2, nsiter2(g))`` after
    building the x-coordinate auxiliary graph in ``position.c``. It shifts the
    tail or head side of zero-cut tree edges halfway across the currently
    available slack, preserving optimal objective value while centering
    degree-balanced nodes in their feasible horizontal range.
    """
    for edge_id in list(graph.tree_edges):
        edge = graph.edges[edge_id]
        if edge.cutvalue != 0:
            continue
        entering = _enter_edge(graph=graph, edge_id=edge_id)
        if entering is None:
            continue
        delta = _slack(graph=graph, edge_id=entering)
        if delta <= 1:
            continue
        if graph.nodes[edge.tail].lim < graph.nodes[edge.head].lim:
            _rerank(graph=graph, node_id=edge.tail, parent_edge=edge_id, delta=delta // 2)
        else:
            _rerank(graph=graph, node_id=edge.head, parent_edge=edge_id, delta=-(delta // 2))


def _build_virtual_edges(
    records: Sequence[_EdgeRecord],
    ranks: Dict[NodeId, int],
    virtual_node_factory: VirtualNodeFactory,
) -> List[GraphvizVirtualEdge]:
    """Create virtual-node descriptors for long edges.

    Parameters
    ----------
    records : sequence of _EdgeRecord
        Normalized original constraints.
    ranks : dict
        Rank assignment for original nodes.
    virtual_node_factory : callable
        Factory for virtual node ids.

    Returns
    -------
    list of GraphvizVirtualEdge
        Long-edge virtual chains in input order.
    """
    virtual_edges: List[GraphvizVirtualEdge] = []
    for record in records:
        tail_rank = ranks[record.tail]
        head_rank = ranks[record.head]
        if head_rank - tail_rank <= 1:
            continue
        virtual_nodes: List[NodeId] = []
        for rank in range(tail_rank + 1, head_rank):
            virtual_nodes.append(
                _call_virtual_node_factory(
                    factory=virtual_node_factory,
                    tail=record.tail,
                    head=record.head,
                    rank=rank,
                    original_index=record.original_index,
                )
            )
        chain = (record.tail, *virtual_nodes, record.head)
        virtual_edges.append(
            GraphvizVirtualEdge(
                original_edge_index=record.original_index,
                source=record.tail,
                target=record.head,
                virtual_nodes=tuple(virtual_nodes),
                chain=chain,
            )
        )
    return virtual_edges


def _call_virtual_node_factory(
    factory: VirtualNodeFactory,
    tail: NodeId,
    head: NodeId,
    rank: int,
    original_index: int,
) -> NodeId:
    """Call a virtual-node factory with compatible arity.

    Parameters
    ----------
    factory : callable
        User-provided virtual node factory.
    tail : Hashable
        Original edge tail.
    head : Hashable
        Original edge head.
    rank : int
        Intermediate rank.
    original_index : int
        Original edge index.

    Returns
    -------
    Hashable
        Virtual node id.
    """
    for args in (
        (tail, head, rank, original_index),
        (tail, head, rank),
        (),
    ):
        try:
            return factory(*args)
        except TypeError:
            continue
    raise TypeError("virtual_node_factory could not be called with supported arities")


def _length(graph: _SimplexGraph, edge_id: int) -> int:
    """Return the current rank length of an edge.

    Parameters
    ----------
    graph : _SimplexGraph
        Mutable rank graph.
    edge_id : int
        Edge id.

    Returns
    -------
    int
        ``rank(head) - rank(tail)``.
    """
    edge = graph.edges[edge_id]
    return graph.nodes[edge.head].rank - graph.nodes[edge.tail].rank


def _slack(graph: _SimplexGraph, edge_id: int) -> int:
    """Return an edge's minlen slack.

    Parameters
    ----------
    graph : _SimplexGraph
        Mutable rank graph.
    edge_id : int
        Edge id.

    Returns
    -------
    int
        ``length(edge) - minlen(edge)``.
    """
    return _length(graph=graph, edge_id=edge_id) - graph.edges[edge_id].minlen


def _is_tree_edge(graph: _SimplexGraph, edge_id: int) -> bool:
    """Return whether an edge is currently in the simplex tree.

    Parameters
    ----------
    graph : _SimplexGraph
        Mutable rank graph.
    edge_id : int
        Edge id.

    Returns
    -------
    bool
        ``True`` when the edge has a tree index.
    """
    return graph.edges[edge_id].tree_index >= 0


def _seq(low: int, value: int, high: int) -> bool:
    """Return whether ``value`` lies in ``[low, high]``.

    Parameters
    ----------
    low : int
        Lower bound.
    value : int
        Candidate value.
    high : int
        Upper bound.

    Returns
    -------
    bool
        Inclusive range membership.
    """
    return low <= value <= high


__all__ = [
    "GraphvizVirtualEdge",
    "graphviz_network_simplex_assignment",
    "graphviz_rank_assignment",
]
