"""Graphviz dot crossing-minimization fidelity helpers."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

import torch

_CONVERGENCE_RATIO = 0.995
_MC_SCALE = 256
_MIN_QUIT = 8


def graphviz_mincross(
    ranks: Sequence[Sequence[int]],
    edges: Union[torch.Tensor, Sequence[Tuple[int, int]]],
    iterations: int = 24,
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

    Returns
    -------
    list of list of int
        Ordered ranks after median sweeps, best-order restoration, and final
        non-reverse transposition.
    """
    ordered_ranks = [list(rank) for rank in ranks]
    if not ordered_ranks:
        return []

    node_to_rank = _node_rank_map(ordered_ranks)
    adjacent_edges = _normalize_adjacent_edges(edges=edges, node_to_rank=node_to_rank)
    if not adjacent_edges:
        return ordered_ranks

    incoming, outgoing = _build_rank_adjacency(
        ranks=ordered_ranks,
        edges=adjacent_edges,
        node_to_rank=node_to_rank,
    )
    current_cross = _count_crossings(
        ranks=ordered_ranks,
        edges=adjacent_edges,
        node_to_rank=node_to_rank,
    )
    best_cross = current_cross
    best_ranks = [list(rank) for rank in ordered_ranks]
    trying = 0

    for iteration in range(max(int(iterations), 0)):
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
        if current_cross <= best_cross:
            best_ranks = [list(rank) for rank in ordered_ranks]
            if current_cross < _CONVERGENCE_RATIO * float(best_cross):
                trying = 0
            best_cross = current_cross

    if current_cross > best_cross:
        ordered_ranks = [list(rank) for rank in best_ranks]

    if best_cross > 0:
        _transpose(ranks=ordered_ranks, incoming=incoming, outgoing=outgoing, reverse=False)

    return ordered_ranks


def _normalize_adjacent_edges(
    edges: Union[torch.Tensor, Sequence[Tuple[int, int]]],
    node_to_rank: Dict[int, int],
) -> List[Tuple[int, int]]:
    """Return adjacent-rank edges in tail-to-head orientation.

    Parameters
    ----------
    edges : torch.Tensor or sequence of tuple[int, int]
        Edge list supplied to :func:`graphviz_mincross`.
    node_to_rank : dict[int, int]
        Mapping from node id to rank index.

    Returns
    -------
    list of tuple[int, int]
        Edges whose endpoints are present in ``ranks`` and differ by exactly
        one rank.
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

    adjacent_edges: List[Tuple[int, int]] = []
    for tail, head in edge_pairs:
        tail_rank = node_to_rank.get(tail)
        head_rank = node_to_rank.get(head)
        if tail_rank is None or head_rank is None:
            continue
        if abs(head_rank - tail_rank) == 1:
            adjacent_edges.append((tail, head))
    return adjacent_edges


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


def _build_rank_adjacency(
    ranks: Sequence[Sequence[int]],
    edges: Sequence[Tuple[int, int]],
    node_to_rank: Dict[int, int],
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """Build Graphviz-style in/out adjacency lists for mincross.

    Parameters
    ----------
    ranks : sequence of sequence of int
        Ordered node ids grouped by rank.
    edges : sequence of tuple[int, int]
        Adjacent-rank edges.
    node_to_rank : dict[int, int]
        Mapping from node id to rank index.

    Returns
    -------
    tuple of dict[int, list[int]]
        ``(incoming, outgoing)`` neighbor lists keyed by node id.
    """
    incoming: Dict[int, List[int]] = {node: [] for rank in ranks for node in rank}
    outgoing: Dict[int, List[int]] = {node: [] for rank in ranks for node in rank}
    for tail, head in edges:
        tail_rank = node_to_rank[tail]
        head_rank = node_to_rank[head]
        if tail_rank < head_rank:
            outgoing[tail].append(head)
            incoming[head].append(tail)
        else:
            outgoing[head].append(tail)
            incoming[tail].append(head)
    return incoming, outgoing


def _mincross_step(
    ranks: List[List[int]],
    incoming: Dict[int, List[int]],
    outgoing: Dict[int, List[int]],
    iteration: int,
) -> None:
    """Run one Graphviz median pass followed by transposition.

    Parameters
    ----------
    ranks : list of list of int
        Mutable ordered ranks.
    incoming : dict[int, list[int]]
        Incoming neighbor lists from the immediately preceding rank.
    outgoing : dict[int, list[int]]
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
        mvals = _median_values(
            rank_nodes=ranks[rank_index],
            neighbors_by_node=reference,
            order_map=_node_order_map(ranks),
        )
        _reorder_rank(rank_nodes=ranks[rank_index], mvals=mvals, reverse=reverse)

    _transpose(ranks=ranks, incoming=incoming, outgoing=outgoing, reverse=not reverse)


def _median_values(
    rank_nodes: Sequence[int],
    neighbors_by_node: Dict[int, List[int]],
    order_map: Dict[int, int],
) -> Dict[int, float]:
    """Compute Graphviz ``ND_mval`` values for one rank.

    Parameters
    ----------
    rank_nodes : sequence of int
        Nodes in the rank currently being reordered.
    neighbors_by_node : dict[int, list[int]]
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
        values = [_MC_SCALE * order_map[neighbor] for neighbor in neighbors_by_node[node]]
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


def _reorder_rank(rank_nodes: List[int], mvals: Dict[int, float], reverse: bool) -> None:
    """Reorder one rank using Graphviz's pair-exchange median rule.

    Parameters
    ----------
    rank_nodes : list[int]
        Mutable rank contents.
    mvals : dict[int, float]
        Median values for nodes in ``rank_nodes``.
    reverse : bool
        Whether equal median values should be reversed during this pass.

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

        if not reverse:
            end -= 1


def _transpose(
    ranks: List[List[int]],
    incoming: Dict[int, List[int]],
    outgoing: Dict[int, List[int]],
    reverse: bool,
) -> None:
    """Run Graphviz's adjacent transposition refinement to convergence.

    Parameters
    ----------
    ranks : list of list of int
        Mutable ordered ranks.
    incoming : dict[int, list[int]]
        Incoming neighbor lists from preceding ranks.
    outgoing : dict[int, list[int]]
        Outgoing neighbor lists to following ranks.
    reverse : bool
        Whether equal crossing counts should be swapped when the current local
        crossing count is positive.

    Returns
    -------
    None
        ``ranks`` is updated in place.
    """
    while True:
        delta = 0
        for rank_index, rank_nodes in enumerate(ranks):
            for order in range(len(rank_nodes) - 1):
                left = rank_nodes[order]
                right = rank_nodes[order + 1]
                before = 0
                after = 0
                if rank_index > 0:
                    before += _in_cross(left, right, incoming, ranks)
                    after += _in_cross(right, left, incoming, ranks)
                if rank_index < len(ranks) - 1:
                    before += _out_cross(left, right, outgoing, ranks)
                    after += _out_cross(right, left, outgoing, ranks)
                if after < before or (before > 0 and reverse and after == before):
                    rank_nodes[order], rank_nodes[order + 1] = right, left
                    delta += before - after
        if delta < 1:
            break


def _in_cross(
    left: int,
    right: int,
    incoming: Dict[int, List[int]],
    ranks: Sequence[Sequence[int]],
) -> int:
    """Count crossings from incoming edges if ``left`` precedes ``right``.

    Parameters
    ----------
    left : int
        Left node in the current rank.
    right : int
        Right node in the current rank.
    incoming : dict[int, list[int]]
        Incoming neighbor lists.
    ranks : sequence of sequence of int
        Current rank ordering.

    Returns
    -------
    int
        Local crossing count.
    """
    order_map = _node_order_map(ranks)
    crossings = 0
    for right_parent in incoming[right]:
        right_order = order_map[right_parent]
        for left_parent in incoming[left]:
            if order_map[left_parent] > right_order:
                crossings += 1
    return crossings


def _out_cross(
    left: int,
    right: int,
    outgoing: Dict[int, List[int]],
    ranks: Sequence[Sequence[int]],
) -> int:
    """Count crossings from outgoing edges if ``left`` precedes ``right``.

    Parameters
    ----------
    left : int
        Left node in the current rank.
    right : int
        Right node in the current rank.
    outgoing : dict[int, list[int]]
        Outgoing neighbor lists.
    ranks : sequence of sequence of int
        Current rank ordering.

    Returns
    -------
    int
        Local crossing count.
    """
    order_map = _node_order_map(ranks)
    crossings = 0
    for right_child in outgoing[right]:
        right_order = order_map[right_child]
        for left_child in outgoing[left]:
            if order_map[left_child] > right_order:
                crossings += 1
    return crossings


def _count_crossings(
    ranks: Sequence[Sequence[int]],
    edges: Sequence[Tuple[int, int]],
    node_to_rank: Dict[int, int],
) -> int:
    """Count adjacent-rank edge crossings in the current order.

    Parameters
    ----------
    ranks : sequence of sequence of int
        Ordered ranks.
    edges : sequence of tuple[int, int]
        Adjacent-rank edges.
    node_to_rank : dict[int, int]
        Mapping from node id to rank index.

    Returns
    -------
    int
        Total crossing count across adjacent rank pairs.
    """
    order_map = _node_order_map(ranks)
    by_rank_pair: Dict[int, List[Tuple[int, int]]] = {
        rank_index: [] for rank_index in range(max(len(ranks) - 1, 0))
    }
    for tail, head in edges:
        tail_rank = node_to_rank[tail]
        head_rank = node_to_rank[head]
        upper = tail if tail_rank < head_rank else head
        lower = head if tail_rank < head_rank else tail
        by_rank_pair[min(tail_rank, head_rank)].append((upper, lower))

    crossings = 0
    for edge_list in by_rank_pair.values():
        for first_index, (upper_a, lower_a) in enumerate(edge_list):
            upper_order_a = order_map[upper_a]
            lower_order_a = order_map[lower_a]
            for upper_b, lower_b in edge_list[first_index + 1 :]:
                if upper_order_a == order_map[upper_b] or lower_order_a == order_map[lower_b]:
                    continue
                upper_delta = upper_order_a - order_map[upper_b]
                lower_delta = lower_order_a - order_map[lower_b]
                if upper_delta * lower_delta < 0:
                    crossings += 1
    return crossings


__all__ = ["graphviz_mincross"]
