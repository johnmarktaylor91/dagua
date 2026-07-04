"""Graphviz dot crossing-minimization fidelity helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch

_CONVERGENCE_RATIO = 0.995
_MAX_INITIAL_PASSES = 2
_MAX_INITIAL_PASS_ITERATIONS = 4
_MC_SCALE = 256
_MIN_QUIT = 8


@dataclass(frozen=True)
class _MincrossEdge:
    """Store one adjacent-rank edge and its Graphviz crossing penalty.

    Parameters
    ----------
    tail : int
        Directed edge tail.
    head : int
        Directed edge head.
    penalty : int
        Graphviz ``ED_xpenalty`` value used by mincross crossing counts.
    """

    tail: int
    head: int
    penalty: int


def graphviz_mincross(
    ranks: Sequence[Sequence[int]],
    edges: Union[torch.Tensor, Sequence[Tuple[int, int]]],
    iterations: int = 24,
    edge_penalties: Optional[Sequence[int]] = None,
    node_order: Optional[Sequence[int]] = None,
) -> List[List[int]]:
    """Order nodes within ranks using Graphviz dot's mincross heuristic.

    Parameters
    ----------
    ranks : sequence of sequence of int
        Initial rank ordering. Each inner sequence contains node ids for one
        rank, ordered from left to right.
    edges : torch.Tensor or sequence of tuple[int, int]
        Directed edges as either a tensor with shape ``[2, E]`` or ``(tail,
        head)`` pairs. The Graphviz C mincross stage expects long edges to have
        already been expanded into adjacent-rank virtual-node chains; this
        helper follows that contract and ignores non-adjacent rank edges.
    iterations : int, default=24
        Maximum number of median/transpose iterations. Graphviz dot defaults
        to ``MaxIter = 24`` when ``mclimit`` is unset.
    edge_penalties : sequence of int, optional
        Per-edge ``ED_xpenalty`` values aligned to ``edges`` before
        non-adjacent filtering. When omitted, every edge uses unit penalty.
    node_order : sequence of int, optional
        Graph node-list order used by Graphviz ``build_ranks`` seeding. When
        omitted, nodes are iterated in supplied rank order.

    Returns
    -------
    list of list of int
        Ordered ranks after median sweeps, best-order restoration, and final
        non-reverse transposition.
    """
    base_ranks = [list(rank) for rank in ranks]
    if not base_ranks:
        return []

    node_to_rank = _node_rank_map(base_ranks)
    build_order = (
        [int(node) for node in node_order]
        if node_order is not None
        else list(_iter_rank_nodes(base_ranks))
    )
    adjacent_edges = _normalize_adjacent_edges(
        edges=edges,
        node_to_rank=node_to_rank,
        edge_penalties=edge_penalties,
    )
    if not adjacent_edges:
        return base_ranks

    max_iter = max(int(iterations), 0)
    best_cross: Optional[int] = None
    best_ranks: List[List[int]] = [list(rank) for rank in base_ranks]
    current_cross = 0
    ordered_ranks = [list(rank) for rank in base_ranks]

    for pass_index in range(_MAX_INITIAL_PASSES + 1):
        if pass_index < _MAX_INITIAL_PASSES:
            max_this_pass = min(_MAX_INITIAL_PASS_ITERATIONS, max_iter)
            ordered_ranks = _build_ranks(
                base_ranks=base_ranks,
                edges=adjacent_edges,
                pass_index=pass_index,
                node_to_rank=node_to_rank,
                node_order=build_order,
            )
            incoming, outgoing = _build_rank_adjacency(
                ranks=ordered_ranks,
                edges=adjacent_edges,
                node_to_rank=node_to_rank,
            )
            _transpose(ranks=ordered_ranks, incoming=incoming, outgoing=outgoing, reverse=False)
            current_cross = _count_crossings(
                ranks=ordered_ranks,
                edges=adjacent_edges,
                node_to_rank=node_to_rank,
            )
            if best_cross is None or current_cross <= best_cross:
                best_ranks = [list(rank) for rank in ordered_ranks]
                best_cross = current_cross
        else:
            max_this_pass = max_iter
            if best_cross is not None and current_cross > best_cross:
                ordered_ranks = [list(rank) for rank in best_ranks]
            current_cross = best_cross if best_cross is not None else current_cross
            incoming, outgoing = _build_rank_adjacency(
                ranks=ordered_ranks,
                edges=adjacent_edges,
                node_to_rank=node_to_rank,
            )

        trying = 0
        for iteration in range(max_this_pass):
            if trying >= _MIN_QUIT or current_cross == 0:
                break
            trying += 1
            _mincross_step(
                ranks=ordered_ranks,
                incoming=incoming,
                outgoing=outgoing,
                iteration=iteration,
            )
            current_cross = _count_crossings(
                ranks=ordered_ranks,
                edges=adjacent_edges,
                node_to_rank=node_to_rank,
            )
            if best_cross is None or current_cross <= best_cross:
                best_ranks = [list(rank) for rank in ordered_ranks]
                if best_cross is not None and current_cross < _CONVERGENCE_RATIO * best_cross:
                    trying = 0
                best_cross = current_cross

        if current_cross == 0:
            break

    if best_cross is not None and current_cross > best_cross:
        ordered_ranks = [list(rank) for rank in best_ranks]

    if best_cross is not None and best_cross > 0:
        incoming, outgoing = _build_rank_adjacency(
            ranks=ordered_ranks,
            edges=adjacent_edges,
            node_to_rank=node_to_rank,
        )
        _transpose(ranks=ordered_ranks, incoming=incoming, outgoing=outgoing, reverse=False)

    return ordered_ranks


def _normalize_adjacent_edges(
    edges: Union[torch.Tensor, Sequence[Tuple[int, int]]],
    node_to_rank: Dict[int, int],
    edge_penalties: Optional[Sequence[int]],
) -> List[_MincrossEdge]:
    """Return adjacent-rank edges in tail-to-head orientation.

    Parameters
    ----------
    edges : torch.Tensor or sequence of tuple[int, int]
        Edge list supplied to :func:`graphviz_mincross`.
    node_to_rank : dict[int, int]
        Mapping from node id to rank index.
    edge_penalties : sequence of int, optional
        Per-edge ``ED_xpenalty`` values aligned to the supplied edge list.

    Returns
    -------
    list of _MincrossEdge
        Edges whose endpoints are present in ``ranks`` and differ by exactly
        one rank. Non-positive penalties are omitted like Graphviz medians.
    """
    edge_pairs: List[Tuple[int, int]]
    if isinstance(edges, torch.Tensor):
        edges_cpu = edges.detach().to(device="cpu", dtype=torch.long)
        if edges_cpu.numel() == 0:
            edge_pairs = []
        else:
            edge_pairs = [
                (int(tail), int(head))
                for tail, head in zip(edges_cpu[0].tolist(), edges_cpu[1].tolist())
            ]
    else:
        edge_pairs = [(int(tail), int(head)) for tail, head in edges]

    penalties = (
        [1] * len(edge_pairs)
        if edge_penalties is None
        else [max(int(penalty), 0) for penalty in edge_penalties]
    )
    if len(penalties) != len(edge_pairs):
        raise ValueError("edge_penalties length must match edge count")

    adjacent_edges: List[_MincrossEdge] = []
    for edge_index, (tail, head) in enumerate(edge_pairs):
        tail_rank = node_to_rank.get(tail)
        head_rank = node_to_rank.get(head)
        if tail_rank is None or head_rank is None:
            continue
        penalty = penalties[edge_index]
        if abs(head_rank - tail_rank) == 1 and penalty > 0:
            adjacent_edges.append(_MincrossEdge(tail=tail, head=head, penalty=penalty))
    return adjacent_edges


def _build_ranks(
    base_ranks: Sequence[Sequence[int]],
    edges: Sequence[_MincrossEdge],
    pass_index: int,
    node_to_rank: Dict[int, int],
    node_order: Sequence[int],
) -> List[List[int]]:
    """Return Graphviz ``build_ranks``-style initial ordering for one pass.

    Parameters
    ----------
    base_ranks : sequence of sequence of int
        Rank buckets in graph input order.
    edges : sequence of _MincrossEdge
        Adjacent-rank edges in graph edge order.
    pass_index : int
        Mincross initialization pass. Pass 0 starts from sources and traverses
        outgoing edges; pass 1 starts from sinks and traverses incoming edges.
    node_to_rank : dict[int, int]
        Stable node-to-rank mapping from ``base_ranks``.
    node_order : sequence of int
        Graph node-list order for source/sink seed scans.

    Returns
    -------
    list of list of int
        Initial rank ordering for the requested pass.
    """
    outgoing, incoming = _directed_adjacency(edges=edges)
    marked = {node: False for rank in base_ranks for node in rank}
    ordered_ranks: List[List[int]] = [[] for _ in base_ranks]

    for node in node_order:
        if node not in node_to_rank:
            continue
        other_edges = incoming.get(node, []) if pass_index == 0 else outgoing.get(node, [])
        if other_edges or marked[node]:
            continue
        marked[node] = True
        queue = [node]
        queue_head = 0
        while queue_head < len(queue):
            current = queue[queue_head]
            queue_head += 1
            ordered_ranks[node_to_rank[current]].append(current)
            neighbors = outgoing.get(current, []) if pass_index == 0 else incoming.get(current, [])
            for neighbor in neighbors:
                if marked[neighbor]:
                    continue
                marked[neighbor] = True
                queue.append(neighbor)

    for node in node_order:
        if node in node_to_rank and not marked[node]:
            ordered_ranks[node_to_rank[node]].append(node)
    return ordered_ranks


def _iter_rank_nodes(ranks: Sequence[Sequence[int]]) -> Iterable[int]:
    """Yield nodes in deterministic input order.

    Parameters
    ----------
    ranks : sequence of sequence of int
        Rank buckets preserving graph input order inside each rank.

    Returns
    -------
    iterable of int
        Node ids in rank-major order.
    """
    for rank in ranks:
        for node in rank:
            yield node


def _directed_adjacency(
    edges: Sequence[_MincrossEdge],
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """Build directed adjacency lists in edge-list order.

    Parameters
    ----------
    edges : sequence of _MincrossEdge
        Adjacent-rank edges.

    Returns
    -------
    tuple of dict[int, list[int]]
        ``(outgoing, incoming)`` neighbor lists keyed by node id.
    """
    outgoing: Dict[int, List[int]] = {}
    incoming: Dict[int, List[int]] = {}
    for edge in edges:
        outgoing.setdefault(edge.tail, []).append(edge.head)
        outgoing.setdefault(edge.head, [])
        incoming.setdefault(edge.head, []).append(edge.tail)
        incoming.setdefault(edge.tail, [])
    return outgoing, incoming


def _node_rank_map(ranks: Sequence[Sequence[int]]) -> Dict[int, int]:
    """Map each node id to its rank index.

    Parameters
    ----------
    ranks : sequence of sequence of int
        Ordered node ids grouped by rank.

    Returns
    -------
    dict[int, int]
        Node-to-rank lookup.
    """
    return {node: rank_index for rank_index, rank in enumerate(ranks) for node in rank}


def _node_order_map(ranks: Sequence[Sequence[int]]) -> Dict[int, int]:
    """Map each node id to its current in-rank order.

    Parameters
    ----------
    ranks : sequence of sequence of int
        Ordered node ids grouped by rank.

    Returns
    -------
    dict[int, int]
        Node-to-order lookup.
    """
    return {node: order for rank in ranks for order, node in enumerate(rank)}


def _rank_order_maps(ranks: Sequence[Sequence[int]]) -> List[Dict[int, int]]:
    """Map every node id to its in-rank order, grouped by rank.

    Parameters
    ----------
    ranks : sequence of sequence of int
        Ordered node ids grouped by rank.

    Returns
    -------
    list of dict[int, int]
        One node-to-order lookup per rank.
    """
    return [{node: order for order, node in enumerate(rank)} for rank in ranks]


def _build_rank_adjacency(
    ranks: Sequence[Sequence[int]],
    edges: Sequence[_MincrossEdge],
    node_to_rank: Dict[int, int],
) -> Tuple[Dict[int, List[Tuple[int, int]]], Dict[int, List[Tuple[int, int]]]]:
    """Build Graphviz-style in/out adjacency lists for mincross.

    Parameters
    ----------
    ranks : sequence of sequence of int
        Ordered node ids grouped by rank.
    edges : sequence of _MincrossEdge
        Adjacent-rank edges.
    node_to_rank : dict[int, int]
        Mapping from node id to rank index.

    Returns
    -------
    tuple of dict[int, list[tuple[int, int]]]
        ``(incoming, outgoing)`` neighbor lists keyed by node id.
    """
    incoming: Dict[int, List[Tuple[int, int]]] = {node: [] for rank in ranks for node in rank}
    outgoing: Dict[int, List[Tuple[int, int]]] = {node: [] for rank in ranks for node in rank}
    for edge in edges:
        tail_rank = node_to_rank[edge.tail]
        head_rank = node_to_rank[edge.head]
        if tail_rank < head_rank:
            outgoing[edge.tail].append((edge.head, edge.penalty))
            incoming[edge.head].append((edge.tail, edge.penalty))
        else:
            outgoing[edge.head].append((edge.tail, edge.penalty))
            incoming[edge.tail].append((edge.head, edge.penalty))
    return incoming, outgoing


def _mincross_step(
    ranks: List[List[int]],
    incoming: Dict[int, List[Tuple[int, int]]],
    outgoing: Dict[int, List[Tuple[int, int]]],
    iteration: int,
) -> None:
    """Run one Graphviz median pass followed by transposition.

    Parameters
    ----------
    ranks : list of list of int
        Mutable ordered ranks.
    incoming : dict[int, list[tuple[int, int]]]
        Incoming neighbor lists from the immediately preceding rank.
    outgoing : dict[int, list[tuple[int, int]]]
        Outgoing neighbor lists to the immediately following rank.
    iteration : int
        Zero-based iteration number, used for Graphviz's pass direction and
        reverse tie rules.

    Returns
    -------
    None
        ``ranks`` is updated in place.
    """
    reverse = iteration % 4 < 2
    if iteration % 2 == 0:
        rank_iter = range(1, len(ranks))
        reference = incoming
    else:
        rank_iter = range(len(ranks) - 2, -1, -1)
        reference = outgoing

    for rank_index in rank_iter:
        hasfixed = any(not incoming[node] and not outgoing[node] for node in ranks[rank_index])
        mvals = _median_values(
            rank_nodes=ranks[rank_index],
            neighbors_by_node=reference,
            order_map=_node_order_map(ranks),
        )
        _reorder_rank(
            rank_nodes=ranks[rank_index],
            mvals=mvals,
            reverse=reverse,
            hasfixed=hasfixed,
        )

    _transpose(ranks=ranks, incoming=incoming, outgoing=outgoing, reverse=not reverse)


def _median_values(
    rank_nodes: Sequence[int],
    neighbors_by_node: Dict[int, List[Tuple[int, int]]],
    order_map: Dict[int, int],
) -> Dict[int, float]:
    """Compute Graphviz ``ND_mval`` values for one rank.

    Parameters
    ----------
    rank_nodes : sequence of int
        Nodes in the rank currently being reordered.
    neighbors_by_node : dict[int, list[tuple[int, int]]]
        Neighbor lists in the reference rank.
    order_map : dict[int, int]
        Current in-rank order for every node.

    Returns
    -------
    dict[int, float]
        Median values. Isolated nodes receive ``-1.0`` to match Graphviz's
        fixed-node sentinel.
    """
    mvals: Dict[int, float] = {}
    for node in rank_nodes:
        values = [
            _MC_SCALE * order_map[neighbor]
            for neighbor, penalty in neighbors_by_node[node]
            if penalty > 0
        ]
        count = len(values)
        if count == 0:
            mvals[node] = -1.0
        elif count == 1:
            mvals[node] = float(values[0])
        elif count == 2:
            mvals[node] = float((values[0] + values[1]) // 2)
        else:
            values.sort()
            if count % 2 == 1:
                mvals[node] = float(values[count // 2])
            else:
                right_mid = count // 2
                left_mid = right_mid - 1
                right_span = values[-1] - values[right_mid]
                left_span = values[left_mid] - values[0]
                if left_span == right_span:
                    mvals[node] = float((values[left_mid] + values[right_mid]) // 2)
                else:
                    numerator = values[left_mid] * float(right_span) + values[right_mid] * float(
                        left_span
                    )
                    mvals[node] = numerator / float(left_span + right_span)
    return mvals


def _reorder_rank(
    rank_nodes: List[int],
    mvals: Dict[int, float],
    reverse: bool,
    hasfixed: bool = False,
) -> None:
    """Reorder one rank using Graphviz's pair-exchange median rule.

    Parameters
    ----------
    rank_nodes : list[int]
        Mutable rank contents.
    mvals : dict[int, float]
        Median values for nodes in ``rank_nodes``.
    reverse : bool
        Whether equal median values should be reversed during this pass.
    hasfixed : bool, default=False
        Whether the rank contains Graphviz fixed sentinel nodes. Graphviz keeps
        the full scan window when fixed nodes are present.

    Returns
    -------
    None
        ``rank_nodes`` is updated in place.
    """
    end = len(rank_nodes)
    for _ in range(len(rank_nodes) - 1, -1, -1):
        left = 0
        while left < end:
            while left < end and mvals[rank_nodes[left]] < 0.0:
                left += 1
            if left >= end:
                break
            right = left + 1
            while right < end and mvals[rank_nodes[right]] < 0.0:
                right += 1
            if right >= end:
                break

            left_value = mvals[rank_nodes[left]]
            right_value = mvals[rank_nodes[right]]
            if left_value > right_value or (left_value >= right_value and reverse):
                rank_nodes[left], rank_nodes[right] = rank_nodes[right], rank_nodes[left]
            left = right

        if not hasfixed and not reverse:
            end -= 1


def _transpose(
    ranks: List[List[int]],
    incoming: Dict[int, List[Tuple[int, int]]],
    outgoing: Dict[int, List[Tuple[int, int]]],
    reverse: bool,
) -> None:
    """Run Graphviz's adjacent transposition refinement to convergence.

    Parameters
    ----------
    ranks : list of list of int
        Mutable ordered ranks.
    incoming : dict[int, list[tuple[int, int]]]
        Incoming neighbor lists from preceding ranks.
    outgoing : dict[int, list[tuple[int, int]]]
        Outgoing neighbor lists to following ranks.
    reverse : bool
        Whether equal crossing counts should be swapped when the current local
        crossing count is positive.

    Returns
    -------
    None
        ``ranks`` is updated in place.
    """
    order_by_rank = _rank_order_maps(ranks)
    while True:
        delta = 0
        for rank_index, rank_nodes in enumerate(ranks):
            for order in range(len(rank_nodes) - 1):
                left = rank_nodes[order]
                right = rank_nodes[order + 1]
                before = 0
                after = 0
                if rank_index > 0:
                    parent_order = order_by_rank[rank_index - 1]
                    before += _in_cross(left, right, incoming, parent_order)
                    after += _in_cross(right, left, incoming, parent_order)
                if rank_index < len(ranks) - 1:
                    child_order = order_by_rank[rank_index + 1]
                    before += _out_cross(left, right, outgoing, child_order)
                    after += _out_cross(right, left, outgoing, child_order)
                if after < before or (before > 0 and reverse and after == before):
                    rank_nodes[order], rank_nodes[order + 1] = right, left
                    order_by_rank[rank_index][left] = order + 1
                    order_by_rank[rank_index][right] = order
                    delta += before - after
        if delta < 1:
            break


def _in_cross(
    left: int,
    right: int,
    incoming: Dict[int, List[Tuple[int, int]]],
    parent_order: Dict[int, int],
) -> int:
    """Count crossings from incoming edges if ``left`` precedes ``right``.

    Parameters
    ----------
    left : int
        Left node in the current rank.
    right : int
        Right node in the current rank.
    incoming : dict[int, list[tuple[int, int]]]
        Incoming neighbor lists.
    parent_order : dict[int, int]
        Current in-rank order lookup for the parent rank.

    Returns
    -------
    int
        Local crossing count.
    """
    crossings = 0
    for right_parent, right_penalty in incoming[right]:
        right_order = parent_order[right_parent]
        for left_parent, left_penalty in incoming[left]:
            if parent_order[left_parent] > right_order:
                crossings += left_penalty * right_penalty
    return crossings


def _out_cross(
    left: int,
    right: int,
    outgoing: Dict[int, List[Tuple[int, int]]],
    child_order: Dict[int, int],
) -> int:
    """Count crossings from outgoing edges if ``left`` precedes ``right``.

    Parameters
    ----------
    left : int
        Left node in the current rank.
    right : int
        Right node in the current rank.
    outgoing : dict[int, list[tuple[int, int]]]
        Outgoing neighbor lists.
    child_order : dict[int, int]
        Current in-rank order lookup for the child rank.

    Returns
    -------
    int
        Local crossing count.
    """
    crossings = 0
    for right_child, right_penalty in outgoing[right]:
        right_order = child_order[right_child]
        for left_child, left_penalty in outgoing[left]:
            if child_order[left_child] > right_order:
                crossings += left_penalty * right_penalty
    return crossings


def _count_crossings(
    ranks: Sequence[Sequence[int]],
    edges: Sequence[_MincrossEdge],
    node_to_rank: Dict[int, int],
) -> int:
    """Count adjacent-rank edge crossings in the current order.

    Parameters
    ----------
    ranks : sequence of sequence of int
        Ordered ranks.
    edges : sequence of _MincrossEdge
        Adjacent-rank edges.
    node_to_rank : dict[int, int]
        Mapping from node id to rank index.

    Returns
    -------
    int
        Total weighted crossing count across adjacent rank pairs.
    """
    order_map = _node_order_map(ranks)
    by_rank_pair: Dict[int, List[Tuple[int, int, int]]] = {
        rank_index: [] for rank_index in range(max(len(ranks) - 1, 0))
    }
    for edge in edges:
        tail_rank = node_to_rank[edge.tail]
        head_rank = node_to_rank[edge.head]
        upper = edge.tail if tail_rank < head_rank else edge.head
        lower = edge.head if tail_rank < head_rank else edge.tail
        by_rank_pair[min(tail_rank, head_rank)].append((upper, lower, edge.penalty))

    crossings = 0
    for rank_index, edge_list in by_rank_pair.items():
        ordered_edges = sorted(
            ((order_map[upper], order_map[lower], penalty) for upper, lower, penalty in edge_list),
            key=lambda item: item[0],
        )
        # Fenwick indices are lower-rank node orders, not edge positions.
        # Sparse wide ranks can have a terminal node order larger than the
        # number of edges between the rank pair.
        lower_rank_width = len(ranks[rank_index + 1]) if rank_index + 1 < len(ranks) else 0
        fenwick = [0] * (lower_rank_width + 2)
        total_penalty = 0
        index = 0
        while index < len(ordered_edges):
            upper_order = ordered_edges[index][0]
            group_end = index
            while group_end < len(ordered_edges) and ordered_edges[group_end][0] == upper_order:
                group_end += 1

            for _, lower_order, penalty in ordered_edges[index:group_end]:
                crossings += penalty * (total_penalty - _fenwick_sum(fenwick, lower_order + 1))

            for _, lower_order, penalty in ordered_edges[index:group_end]:
                _fenwick_add(fenwick, lower_order + 1, penalty)
                total_penalty += penalty
            index = group_end
    return crossings


def _fenwick_add(tree: List[int], index: int, value: int) -> None:
    """Add ``value`` at one-based ``index`` in a Fenwick tree.

    Parameters
    ----------
    tree : list[int]
        Mutable Fenwick tree.
    index : int
        One-based update index.
    value : int
        Value to add.

    Returns
    -------
    None
        ``tree`` is updated in place.
    """
    while index < len(tree):
        tree[index] += value
        index += index & -index


def _fenwick_sum(tree: Sequence[int], index: int) -> int:
    """Return prefix sum through one-based ``index`` in a Fenwick tree.

    Parameters
    ----------
    tree : sequence[int]
        Fenwick tree.
    index : int
        One-based inclusive query index.

    Returns
    -------
    int
        Prefix sum up to ``index``.
    """
    total = 0
    while index > 0:
        total += tree[index]
        index -= index & -index
    return total


__all__ = ["graphviz_mincross"]
