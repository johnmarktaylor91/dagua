"""Chrobak-Payne planar layout pipeline without runtime delegation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import ClassVar, DefaultDict, Iterable, Iterator, Optional, Sequence, Tuple

import numpy as np
import torch

from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory

_PLANAR_EMBEDDING_KEY = "planar_embedding"
_PLANAR_RAW_POS_KEY = "planar_raw_pos"


class PlanarityError(RuntimeError):
    """Raised when the Chrobak-Payne pipeline receives a non-planar graph."""


@dataclass
class _Interval:
    """Return-edge interval used by the Left-Right planarity test.

    Parameters
    ----------
    low : tuple[int, int] | None, optional
        Lowest return edge in the interval.
    high : tuple[int, int] | None, optional
        Highest return edge in the interval.
    """

    low: Optional[tuple[int, int]] = None
    high: Optional[tuple[int, int]] = None

    def empty(self) -> bool:
        """Return whether the interval has no return edges.

        Returns
        -------
        bool
            ``True`` when both bounds are absent.
        """
        return self.low is None and self.high is None

    def copy(self) -> "_Interval":
        """Return a shallow interval copy.

        Returns
        -------
        _Interval
            Copied interval with the same edge bounds.
        """
        return _Interval(self.low, self.high)

    def conflicting(self, edge: tuple[int, int], planarity_state: "_LRPlanarity") -> bool:
        """Return whether this interval conflicts with an edge.

        Parameters
        ----------
        edge : tuple[int, int]
            Directed edge being integrated.
        planarity_state : _LRPlanarity
            Active planarity state containing lowpoint metadata.

        Returns
        -------
        bool
            ``True`` when the interval lies above ``edge``'s lowpoint.
        """
        return (
            not self.empty()
            and self.high is not None
            and planarity_state.lowpt[self.high] > planarity_state.lowpt[edge]
        )


@dataclass
class _ConflictPair:
    """Two interval groups that must lie on opposite sides.

    Parameters
    ----------
    left : _Interval | None, optional
        Left interval group.
    right : _Interval | None, optional
        Right interval group.
    """

    left: _Interval
    right: _Interval

    def __init__(
        self,
        left: Optional[_Interval] = None,
        right: Optional[_Interval] = None,
    ) -> None:
        """Initialize interval groups.

        Parameters
        ----------
        left : _Interval | None, optional
            Left interval group. ``None`` creates an empty interval.
        right : _Interval | None, optional
            Right interval group. ``None`` creates an empty interval.
        """
        self.left = _Interval() if left is None else left
        self.right = _Interval() if right is None else right

    def swap(self) -> None:
        """Swap left and right interval groups.

        Returns
        -------
        None
            The conflict pair is mutated in place.
        """
        self.left, self.right = self.right, self.left

    def lowest(self, planarity_state: "_LRPlanarity") -> int:
        """Return the lowest lowpoint represented by this pair.

        Parameters
        ----------
        planarity_state : _LRPlanarity
            Active planarity state containing lowpoint metadata.

        Returns
        -------
        int
            Minimum lowpoint height in the pair.
        """
        if self.left.empty():
            if self.right.low is None:
                raise RuntimeError("Empty conflict pair has no lowpoint.")
            return planarity_state.lowpt[self.right.low]
        if self.right.empty():
            if self.left.low is None:
                raise RuntimeError("Empty conflict pair has no lowpoint.")
            return planarity_state.lowpt[self.left.low]
        if self.left.low is None or self.right.low is None:
            raise RuntimeError("Malformed conflict pair has no lowpoint.")
        return min(planarity_state.lowpt[self.left.low], planarity_state.lowpt[self.right.low])


def _top_of_stack(stack: Sequence[_ConflictPair]) -> Optional[_ConflictPair]:
    """Return the current top conflict pair.

    Parameters
    ----------
    stack : sequence[_ConflictPair]
        Conflict-pair stack.

    Returns
    -------
    _ConflictPair | None
        Last pair on the stack, or ``None`` when empty.
    """
    if not stack:
        return None
    return stack[-1]


class _Graph:
    """Insertion-ordered undirected graph for integer node ids."""

    def __init__(self, num_nodes: int, edges: Iterable[tuple[int, int]] = ()) -> None:
        """Build the graph.

        Parameters
        ----------
        num_nodes : int
            Number of nodes in the graph.
        edges : iterable[tuple[int, int]], optional
            Undirected edges to add in input order.
        """
        self._adj: dict[int, dict[int, None]] = {node: {} for node in range(num_nodes)}
        for source, target in edges:
            self.add_edge(source, target)

    def __iter__(self) -> Iterator[int]:
        """Iterate nodes in insertion order.

        Returns
        -------
        Iterator[int]
            Iterator over integer node ids.
        """
        return iter(self._adj)

    def __getitem__(self, node: int) -> dict[int, None]:
        """Return adjacency mapping for a node.

        Parameters
        ----------
        node : int
            Node id.

        Returns
        -------
        dict[int, None]
            Neighbor mapping in insertion order.
        """
        return self._adj[node]

    def order(self) -> int:
        """Return the node count.

        Returns
        -------
        int
            Number of nodes.
        """
        return len(self._adj)

    def size(self) -> int:
        """Return the undirected edge count.

        Returns
        -------
        int
            Number of undirected edges.
        """
        return sum(len(neighbors) for neighbors in self._adj.values()) // 2

    def add_edge(self, source: int, target: int) -> None:
        """Add one undirected edge.

        Parameters
        ----------
        source : int
            Source endpoint.
        target : int
            Target endpoint.

        Returns
        -------
        None
            The graph is mutated in place.
        """
        if source not in self._adj or target not in self._adj:
            raise ValueError("edge endpoint is outside the node range.")
        self._adj[source][target] = None
        self._adj[target][source] = None

    def edges(self) -> Iterator[tuple[int, int]]:
        """Iterate undirected edges once in adjacency insertion order.

        Returns
        -------
        Iterator[tuple[int, int]]
            Edge iterator.
        """
        seen: set[tuple[int, int]] = set()
        for source, neighbors in self._adj.items():
            for target in neighbors:
                key = (source, target) if source <= target else (target, source)
                if key in seen:
                    continue
                seen.add(key)
                yield source, target


class _DiGraph:
    """Insertion-ordered directed graph used by the planarity state."""

    def __init__(self, nodes: Iterable[int]) -> None:
        """Initialize an empty directed graph.

        Parameters
        ----------
        nodes : iterable[int]
            Nodes to add in iteration order.
        """
        self._succ: dict[int, dict[int, None]] = {node: {} for node in nodes}

    def __iter__(self) -> Iterator[int]:
        """Iterate nodes in insertion order.

        Returns
        -------
        Iterator[int]
            Node iterator.
        """
        return iter(self._succ)

    def __getitem__(self, node: int) -> dict[int, None]:
        """Return successor mapping for a node.

        Parameters
        ----------
        node : int
            Node id.

        Returns
        -------
        dict[int, None]
            Successor mapping.
        """
        return self._succ[node]

    def add_edge(self, source: int, target: int) -> None:
        """Add one directed edge.

        Parameters
        ----------
        source : int
            Source endpoint.
        target : int
            Target endpoint.

        Returns
        -------
        None
            The graph is mutated in place.
        """
        self._succ[source][target] = None

    def has_edge(self, source: int, target: int) -> bool:
        """Return whether a directed edge exists.

        Parameters
        ----------
        source : int
            Source endpoint.
        target : int
            Target endpoint.

        Returns
        -------
        bool
            ``True`` when ``source`` has ``target`` as a successor.
        """
        return target in self._succ[source]

    def edges(self) -> Iterator[tuple[int, int]]:
        """Iterate directed edges in insertion order.

        Returns
        -------
        Iterator[tuple[int, int]]
            Directed edge iterator.
        """
        for source, neighbors in self._succ.items():
            for target in neighbors:
                yield source, target


class PlanarEmbedding:
    """Combinatorial planar embedding with clockwise neighbor order."""

    def __init__(self, incoming: Optional["PlanarEmbedding"] = None) -> None:
        """Initialize an embedding, optionally copying another one.

        Parameters
        ----------
        incoming : PlanarEmbedding | None, optional
            Existing embedding to copy.
        """
        self._succ: dict[int, dict[int, dict[str, int]]] = {}
        if incoming is not None:
            self.add_nodes_from(incoming)
            for source in incoming:
                for target, data in incoming._succ[source].items():
                    self._succ[source][target] = dict(data)

    def __iter__(self) -> Iterator[int]:
        """Iterate nodes in insertion order.

        Returns
        -------
        Iterator[int]
            Node iterator.
        """
        return iter(self._succ)

    def __len__(self) -> int:
        """Return node count.

        Returns
        -------
        int
            Number of embedded nodes.
        """
        return len(self._succ)

    def __getitem__(self, node: int) -> dict[int, dict[str, int]]:
        """Return directed half-edge data for a node.

        Parameters
        ----------
        node : int
            Node id.

        Returns
        -------
        dict[int, dict[str, int]]
            Neighbor-to-orientation mapping.
        """
        return self._succ[node]

    def add_nodes_from(self, nodes: Iterable[int]) -> None:
        """Add nodes in iteration order.

        Parameters
        ----------
        nodes : iterable[int]
            Node ids.

        Returns
        -------
        None
            Missing nodes are appended to the embedding.
        """
        for node in nodes:
            self._succ.setdefault(node, {})

    def nodes(self) -> list[int]:
        """Return embedded nodes in insertion order.

        Returns
        -------
        list[int]
            Node list.
        """
        return list(self._succ)

    def has_edge(self, source: int, target: int) -> bool:
        """Return whether a half-edge exists.

        Parameters
        ----------
        source : int
            Source endpoint.
        target : int
            Target endpoint.

        Returns
        -------
        bool
            ``True`` when the half-edge exists.
        """
        return source in self._succ and target in self._succ[source]

    def neighbors_cw_order(self, node: int) -> Iterator[int]:
        """Iterate neighbors in clockwise order.

        Parameters
        ----------
        node : int
            Embedded node.

        Returns
        -------
        Iterator[int]
            Clockwise neighbor iterator.
        """
        succs = self._succ[node]
        if not succs:
            return
        start_node = next(reversed(succs))
        yield start_node
        current_node = succs[start_node]["cw"]
        while start_node != current_node:
            yield current_node
            current_node = succs[current_node]["cw"]

    def add_half_edge(
        self,
        start_node: int,
        end_node: int,
        *,
        cw: Optional[int] = None,
        ccw: Optional[int] = None,
    ) -> None:
        """Add one directed half-edge at a referenced cyclic position.

        Parameters
        ----------
        start_node : int
            Source node of the half-edge.
        end_node : int
            Target node of the half-edge.
        cw : int | None, optional
            Clockwise reference neighbor.
        ccw : int | None, optional
            Counterclockwise reference neighbor.

        Returns
        -------
        None
            The embedding is mutated in place.
        """
        self._succ.setdefault(start_node, {})
        self._succ.setdefault(end_node, self._succ.get(end_node, {}))
        succs = self._succ[start_node]
        if succs:
            leftmost_nbr = next(reversed(succs))
            if cw is not None:
                if ccw is not None or cw not in succs:
                    raise RuntimeError("Invalid half-edge reference.")
                ref_ccw = succs[cw]["ccw"]
                succs[end_node] = {"cw": cw, "ccw": ref_ccw}
                succs[ref_ccw]["cw"] = end_node
                succs[cw]["ccw"] = end_node
                move_leftmost_nbr_to_end = cw != leftmost_nbr
            elif ccw is not None:
                if ccw not in succs:
                    raise RuntimeError("Invalid half-edge reference.")
                ref_cw = succs[ccw]["cw"]
                succs[end_node] = {"cw": ref_cw, "ccw": ccw}
                succs[ref_cw]["ccw"] = end_node
                succs[ccw]["cw"] = end_node
                move_leftmost_nbr_to_end = True
            else:
                raise RuntimeError("Reference required for non-first half-edge.")
            if move_leftmost_nbr_to_end:
                succs[leftmost_nbr] = succs.pop(leftmost_nbr)
        else:
            if cw is not None or ccw is not None:
                raise RuntimeError("Invalid reference for first half-edge.")
            succs[end_node] = {"ccw": end_node, "cw": end_node}

    def add_half_edge_first(self, start_node: int, end_node: int) -> None:
        """Add a half-edge and make it the leftmost neighbor.

        Parameters
        ----------
        start_node : int
            Source node.
        end_node : int
            Target node.

        Returns
        -------
        None
            The embedding is mutated in place.
        """
        succs = self._succ.get(start_node)
        leftmost_nbr = next(reversed(succs)) if succs else None
        self.add_half_edge(start_node, end_node, cw=leftmost_nbr)

    def connect_components(self, source: int, target: int) -> None:
        """Connect two components with reciprocal half-edges.

        Parameters
        ----------
        source : int
            Node in the first component.
        target : int
            Node in the second component.

        Returns
        -------
        None
            The embedding is mutated in place.
        """
        source_ref = next(reversed(self._succ[source])) if self._succ.get(source) else None
        self.add_half_edge(source, target, cw=source_ref)
        target_ref = next(reversed(self._succ[target])) if self._succ.get(target) else None
        self.add_half_edge(target, source, cw=target_ref)

    def next_face_half_edge(self, source: int, target: int) -> tuple[int, int]:
        """Return the next half-edge along the same face.

        Parameters
        ----------
        source : int
            Current source endpoint.
        target : int
            Current target endpoint.

        Returns
        -------
        tuple[int, int]
            Next directed half-edge.
        """
        return target, self._succ[target][source]["ccw"]

    def get_data(self) -> dict[int, list[int]]:
        """Return clockwise neighbor lists for each node.

        Returns
        -------
        dict[int, list[int]]
            Mapping from node id to cyclic neighbor order.
        """
        return {node: list(self.neighbors_cw_order(node)) for node in self}


class _LRPlanarity:
    """State machine for the Left-Right planarity algorithm."""

    def __init__(self, graph: _Graph) -> None:
        """Initialize planarity state.

        Parameters
        ----------
        graph : _Graph
            Input undirected graph.
        """
        self.graph: Optional[_Graph] = _Graph(graph.order())
        for source, target in graph.edges():
            if source != target and self.graph is not None:
                self.graph.add_edge(source, target)
        self.roots: list[int] = []
        self.height: DefaultDict[int, Optional[int]] = defaultdict(lambda: None)
        self.lowpt: dict[tuple[int, int], int] = {}
        self.lowpt2: dict[tuple[int, int], int] = {}
        self.nesting_depth: dict[tuple[int, int], int] = {}
        self.parent_edge: DefaultDict[int, Optional[tuple[int, int]]] = defaultdict(lambda: None)
        self.directed_graph = _DiGraph(graph)
        self.adjs: dict[int, list[int]] = {}
        self.ordered_adjs: dict[int, list[int]] = {}
        self.ref: DefaultDict[tuple[int, int], Optional[tuple[int, int]]] = defaultdict(
            lambda: None
        )
        self.side: DefaultDict[tuple[int, int], int] = defaultdict(lambda: 1)
        self.stack: list[_ConflictPair] = []
        self.stack_bottom: dict[tuple[int, int], Optional[_ConflictPair]] = {}
        self.lowpt_edge: dict[Optional[tuple[int, int]], tuple[int, int]] = {}
        self.left_ref: dict[int, int] = {}
        self.right_ref: dict[int, int] = {}
        self.embedding = PlanarEmbedding()

    def lr_planarity(self) -> Optional[PlanarEmbedding]:
        """Run the Left-Right planarity test.

        Returns
        -------
        PlanarEmbedding | None
            Combinatorial embedding when planar, otherwise ``None``.
        """
        if self.graph is None:
            raise RuntimeError("Planarity graph has already been consumed.")
        if self.graph.order() > 2 and self.graph.size() > 3 * self.graph.order() - 6:
            return None

        for node in self.graph:
            self.adjs[node] = list(self.graph[node])

        for node in self.graph:
            if self.height[node] is None:
                self.height[node] = 0
                self.roots.append(node)
                self.dfs_orientation(node)

        for node in self.directed_graph:
            self.ordered_adjs[node] = sorted(
                self.directed_graph[node],
                key=lambda nbr: self.nesting_depth[(node, nbr)],
            )
        for node in self.roots:
            if not self.dfs_testing(node):
                return None

        for edge in list(self.directed_graph.edges()):
            self.nesting_depth[edge] = self.sign(edge) * self.nesting_depth[edge]

        self.embedding.add_nodes_from(self.directed_graph)
        for node in self.directed_graph:
            self.ordered_adjs[node] = sorted(
                self.directed_graph[node],
                key=lambda nbr: self.nesting_depth[(node, nbr)],
            )
            previous_node = None
            for nbr in self.ordered_adjs[node]:
                self.embedding.add_half_edge(node, nbr, ccw=previous_node)
                previous_node = nbr

        for node in self.roots:
            self.dfs_embedding(node)

        return self.embedding

    def dfs_orientation(self, node: int) -> None:
        """Orient edges by DFS and compute lowpoints.

        Parameters
        ----------
        node : int
            DFS root.

        Returns
        -------
        None
            Orientation metadata is written to the state.
        """
        dfs_stack = [node]
        ind: DefaultDict[int, int] = defaultdict(lambda: 0)
        skip_init: DefaultDict[tuple[int, int], bool] = defaultdict(lambda: False)

        while dfs_stack:
            current = dfs_stack.pop()
            parent = self.parent_edge[current]

            for nbr in self.adjs[current][ind[current] :]:
                edge = (current, nbr)
                if not skip_init[edge]:
                    if self.directed_graph.has_edge(current, nbr) or self.directed_graph.has_edge(
                        nbr, current
                    ):
                        ind[current] += 1
                        continue

                    self.directed_graph.add_edge(current, nbr)
                    current_height = self.height[current]
                    if current_height is None:
                        raise RuntimeError("DFS height missing during orientation.")
                    self.lowpt[edge] = current_height
                    self.lowpt2[edge] = current_height
                    if self.height[nbr] is None:
                        self.parent_edge[nbr] = edge
                        self.height[nbr] = current_height + 1
                        dfs_stack.append(current)
                        dfs_stack.append(nbr)
                        skip_init[edge] = True
                        break
                    self.lowpt[edge] = self.height[nbr] or 0

                self.nesting_depth[edge] = 2 * self.lowpt[edge]
                current_height = self.height[current]
                if current_height is not None and self.lowpt2[edge] < current_height:
                    self.nesting_depth[edge] += 1

                if parent is not None:
                    if self.lowpt[edge] < self.lowpt[parent]:
                        self.lowpt2[parent] = min(self.lowpt[parent], self.lowpt2[edge])
                        self.lowpt[parent] = self.lowpt[edge]
                    elif self.lowpt[edge] > self.lowpt[parent]:
                        self.lowpt2[parent] = min(self.lowpt2[parent], self.lowpt[edge])
                    else:
                        self.lowpt2[parent] = min(self.lowpt2[parent], self.lowpt2[edge])
                ind[current] += 1

    def dfs_testing(self, node: int) -> bool:
        """Test whether the DFS orientation admits a planar LR partition.

        Parameters
        ----------
        node : int
            DFS root.

        Returns
        -------
        bool
            ``True`` when no conflict proves non-planarity.
        """
        dfs_stack = [node]
        ind: DefaultDict[int, int] = defaultdict(lambda: 0)
        skip_init: DefaultDict[tuple[int, int], bool] = defaultdict(lambda: False)

        while dfs_stack:
            current = dfs_stack.pop()
            parent = self.parent_edge[current]
            skip_final = False

            for nbr in self.ordered_adjs[current][ind[current] :]:
                edge = (current, nbr)
                if not skip_init[edge]:
                    self.stack_bottom[edge] = _top_of_stack(self.stack)
                    if edge == self.parent_edge[nbr]:
                        dfs_stack.append(current)
                        dfs_stack.append(nbr)
                        skip_init[edge] = True
                        skip_final = True
                        break
                    self.lowpt_edge[edge] = edge
                    self.stack.append(_ConflictPair(right=_Interval(edge, edge)))

                current_height = self.height[current]
                if current_height is not None and self.lowpt[edge] < current_height:
                    if nbr == self.ordered_adjs[current][0]:
                        self.lowpt_edge[parent] = self.lowpt_edge[edge]
                    elif parent is not None and not self.add_constraints(edge, parent):
                        return False
                ind[current] += 1

            if not skip_final and parent is not None:
                self.remove_back_edges(parent)
        return True

    def add_constraints(self, edge_i: tuple[int, int], parent: tuple[int, int]) -> bool:
        """Merge return-edge constraints for one DFS edge.

        Parameters
        ----------
        edge_i : tuple[int, int]
            Edge whose constraints are being integrated.
        parent : tuple[int, int]
            Parent DFS edge.

        Returns
        -------
        bool
            ``False`` when the constraints prove non-planarity.
        """
        pair = _ConflictPair()
        while True:
            conflict = self.stack.pop()
            if not conflict.left.empty():
                conflict.swap()
            if not conflict.left.empty():
                return False
            if conflict.right.low is None:
                raise RuntimeError("Malformed right interval.")
            if self.lowpt[conflict.right.low] > self.lowpt[parent]:
                if pair.right.empty():
                    pair.right = conflict.right.copy()
                else:
                    if pair.right.low is None:
                        raise RuntimeError("Malformed merged interval.")
                    self.ref[pair.right.low] = conflict.right.high
                pair.right.low = conflict.right.low
            else:
                self.ref[conflict.right.low] = self.lowpt_edge[parent]
            if _top_of_stack(self.stack) == self.stack_bottom[edge_i]:
                break

        while self.stack and (
            self.stack[-1].left.conflicting(edge_i, self)
            or self.stack[-1].right.conflicting(edge_i, self)
        ):
            conflict = self.stack.pop()
            if conflict.right.conflicting(edge_i, self):
                conflict.swap()
            if conflict.right.conflicting(edge_i, self):
                return False
            if pair.right.low is None:
                raise RuntimeError("Malformed right interval before merge.")
            self.ref[pair.right.low] = conflict.right.high
            if conflict.right.low is not None:
                pair.right.low = conflict.right.low

            if pair.left.empty():
                pair.left = conflict.left.copy()
            else:
                if pair.left.low is None:
                    raise RuntimeError("Malformed left interval before merge.")
                self.ref[pair.left.low] = conflict.left.high
            pair.left.low = conflict.left.low

        if not (pair.left.empty() and pair.right.empty()):
            self.stack.append(pair)
        return True

    def remove_back_edges(self, edge: tuple[int, int]) -> None:
        """Remove return edges ending at the parent endpoint.

        Parameters
        ----------
        edge : tuple[int, int]
            Parent DFS edge.

        Returns
        -------
        None
            Conflict stack and references are updated in place.
        """
        parent_node = edge[0]
        parent_height = self.height[parent_node]
        if parent_height is None:
            raise RuntimeError("Parent height missing during back-edge removal.")

        while self.stack and self.stack[-1].lowest(self) == parent_height:
            pair = self.stack.pop()
            if pair.left.low is not None:
                self.side[pair.left.low] = -1

        if self.stack:
            pair = self.stack.pop()
            while pair.left.high is not None and pair.left.high[1] == parent_node:
                pair.left.high = self.ref[pair.left.high]
            if pair.left.high is None and pair.left.low is not None:
                self.ref[pair.left.low] = pair.right.low
                self.side[pair.left.low] = -1
                pair.left.low = None

            while pair.right.high is not None and pair.right.high[1] == parent_node:
                pair.right.high = self.ref[pair.right.high]
            if pair.right.high is None and pair.right.low is not None:
                self.ref[pair.right.low] = pair.left.low
                self.side[pair.right.low] = -1
                pair.right.low = None
            self.stack.append(pair)

        if self.lowpt[edge] < parent_height:
            top = _top_of_stack(self.stack)
            if top is None:
                raise RuntimeError("Missing conflict pair for return edge.")
            left_high = top.left.high
            right_high = top.right.high
            if left_high is not None and (
                right_high is None or self.lowpt[left_high] > self.lowpt[right_high]
            ):
                self.ref[edge] = left_high
            else:
                self.ref[edge] = right_high

    def dfs_embedding(self, node: int) -> None:
        """Complete reciprocal half-edge placement from LR sides.

        Parameters
        ----------
        node : int
            DFS root.

        Returns
        -------
        None
            ``self.embedding`` is completed in place.
        """
        dfs_stack = [node]
        ind: DefaultDict[int, int] = defaultdict(lambda: 0)

        while dfs_stack:
            current = dfs_stack.pop()
            for nbr in self.ordered_adjs[current][ind[current] :]:
                ind[current] += 1
                edge = (current, nbr)
                if edge == self.parent_edge[nbr]:
                    self.embedding.add_half_edge_first(nbr, current)
                    self.left_ref[current] = nbr
                    self.right_ref[current] = nbr
                    dfs_stack.append(current)
                    dfs_stack.append(nbr)
                    break
                if self.side[edge] == 1:
                    self.embedding.add_half_edge(nbr, current, ccw=self.right_ref[nbr])
                else:
                    self.embedding.add_half_edge(nbr, current, cw=self.left_ref[nbr])
                    self.left_ref[nbr] = current

    def sign(self, edge: tuple[int, int]) -> int:
        """Resolve a relative side assignment to an absolute side.

        Parameters
        ----------
        edge : tuple[int, int]
            Directed edge whose side is requested.

        Returns
        -------
        int
            Side multiplier, either ``1`` or ``-1``.
        """
        dfs_stack = [edge]
        old_ref: DefaultDict[tuple[int, int], Optional[tuple[int, int]]] = defaultdict(lambda: None)
        while dfs_stack:
            current = dfs_stack.pop()
            if self.ref[current] is not None:
                dfs_stack.append(current)
                dfs_stack.append(self.ref[current])  # type: ignore[arg-type]
                old_ref[current] = self.ref[current]
                self.ref[current] = None
            else:
                ref_edge = old_ref[current]
                if ref_edge is not None:
                    self.side[current] *= self.side[ref_edge]
        return self.side[edge]


def _edge_index_to_graph(edge_index: torch.Tensor, num_nodes: int) -> _Graph:
    """Build an insertion-ordered undirected graph from Dagua edges.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    _Graph
        Undirected simple graph with self-loops ignored by planarity later.
    """
    edges: list[tuple[int, int]] = []
    if edge_index.numel() > 0:
        cpu_edges = edge_index.detach().to(device="cpu", dtype=torch.long)
        for edge_pos in range(cpu_edges.shape[1]):
            edges.append((int(cpu_edges[0, edge_pos].item()), int(cpu_edges[1, edge_pos].item())))
    return _Graph(num_nodes, edges)


def check_planarity(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> tuple[bool, Optional[PlanarEmbedding]]:
    """Check planarity and return a combinatorial embedding.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    tuple[bool, PlanarEmbedding | None]
        Planarity verdict and embedding when planar.
    """
    graph = _edge_index_to_graph(edge_index, num_nodes)
    embedding = _LRPlanarity(graph).lr_planarity()
    return embedding is not None, embedding


def _connected_components(embedding: PlanarEmbedding) -> list[set[int]]:
    """Return connected components of an embedding.

    Parameters
    ----------
    embedding : PlanarEmbedding
        Embedding to inspect.

    Returns
    -------
    list[set[int]]
        Components in first-seen node order.
    """
    seen: set[int] = set()
    components: list[set[int]] = []
    for start in embedding:
        if start in seen:
            continue
        component: set[int] = set()
        stack = [start]
        seen.add(start)
        while stack:
            node = stack.pop()
            component.add(node)
            for nbr in embedding[node]:
                if nbr not in seen:
                    seen.add(nbr)
                    stack.append(nbr)
        components.append(component)
    return components


def _make_bi_connected(
    embedding: PlanarEmbedding,
    starting_node: int,
    outgoing_node: int,
    edges_counted: set[tuple[int, int]],
) -> list[int]:
    """Triangulate a face enough to make the boundary 2-connected.

    Parameters
    ----------
    embedding : PlanarEmbedding
        Embedding being modified.
    starting_node : int
        First node on the face.
    outgoing_node : int
        Second node on the face.
    edges_counted : set[tuple[int, int]]
        Half-edges already assigned to visited faces.

    Returns
    -------
    list[int]
        Face boundary nodes, or an empty list for an already-counted face.
    """
    if (starting_node, outgoing_node) in edges_counted:
        return []
    edges_counted.add((starting_node, outgoing_node))

    v1 = starting_node
    v2 = outgoing_node
    face_list = [starting_node]
    face_set = {starting_node}
    _, v3 = embedding.next_face_half_edge(v1, v2)

    while v2 != starting_node or v3 != outgoing_node:
        if v1 == v2:
            raise PlanarityError("Invalid half-edge")
        if v2 in face_set:
            embedding.add_half_edge(v1, v3, ccw=v2)
            embedding.add_half_edge(v3, v1, cw=v2)
            edges_counted.add((v2, v3))
            edges_counted.add((v3, v1))
            v2 = v1
        else:
            face_set.add(v2)
            face_list.append(v2)

        v1 = v2
        v2, v3 = embedding.next_face_half_edge(v2, v3)
        edges_counted.add((v1, v2))
    return face_list


def _triangulate_face(embedding: PlanarEmbedding, v1: int, v2: int) -> None:
    """Triangulate one face in place.

    Parameters
    ----------
    embedding : PlanarEmbedding
        Embedding being modified.
    v1 : int
        First endpoint of a half-edge on the face.
    v2 : int
        Second endpoint of a half-edge on the face.

    Returns
    -------
    None
        Missing diagonals are inserted into the embedding.
    """
    _, v3 = embedding.next_face_half_edge(v1, v2)
    _, v4 = embedding.next_face_half_edge(v2, v3)
    if v1 in (v2, v3):
        return
    while v1 != v4:
        if embedding.has_edge(v1, v3):
            v1, v2, v3 = v2, v3, v4
        else:
            embedding.add_half_edge(v1, v3, ccw=v2)
            embedding.add_half_edge(v3, v1, cw=v2)
            v1, v2, v3 = v1, v3, v4
        _, v4 = embedding.next_face_half_edge(v2, v3)


def triangulate_embedding(
    embedding: PlanarEmbedding,
    fully_triangulate: bool = True,
) -> tuple[PlanarEmbedding, list[int]]:
    """Triangulate an embedding and return the chosen outer face.

    Parameters
    ----------
    embedding : PlanarEmbedding
        Input combinatorial embedding.
    fully_triangulate : bool, default=True
        Whether to triangulate the selected outer face too.

    Returns
    -------
    tuple[PlanarEmbedding, list[int]]
        New triangulated embedding and outer-face node list.
    """
    if len(embedding) <= 1:
        return embedding, embedding.nodes()

    embedding = PlanarEmbedding(embedding)
    component_nodes = [next(iter(component)) for component in _connected_components(embedding)]
    for idx in range(len(component_nodes) - 1):
        embedding.connect_components(component_nodes[idx], component_nodes[idx + 1])

    outer_face: list[int] = []
    face_list: list[list[int]] = []
    edges_visited: set[tuple[int, int]] = set()
    for node in embedding.nodes():
        for nbr in embedding.neighbors_cw_order(node):
            new_face = _make_bi_connected(embedding, node, nbr, edges_visited)
            if new_face:
                face_list.append(new_face)
                if len(new_face) > len(outer_face):
                    outer_face = new_face

    for face in face_list:
        if face is not outer_face or fully_triangulate:
            _triangulate_face(embedding, face[0], face[1])

    if fully_triangulate:
        v1 = outer_face[0]
        v2 = outer_face[1]
        v3 = embedding[v2][v1]["ccw"]
        outer_face = [v1, v2, v3]

    return embedding, outer_face


def get_canonical_ordering(
    embedding: PlanarEmbedding,
    outer_face: Sequence[int],
) -> list[tuple[int, list[int]]]:
    """Return a canonical node ordering for a triangulated embedding.

    Parameters
    ----------
    embedding : PlanarEmbedding
        Triangulated planar embedding.
    outer_face : sequence[int]
        Nodes on the selected outer face.

    Returns
    -------
    list[tuple[int, list[int]]]
        Canonical order entries ``(node, contour_neighbors)``.
    """
    v1 = outer_face[0]
    v2 = outer_face[1]
    chords: DefaultDict[int, int] = defaultdict(int)
    marked_nodes: set[int] = set()
    ready_to_pick = set(outer_face)

    outer_face_ccw_nbr: dict[int, int] = {}
    prev_nbr = v2
    for idx in range(2, len(outer_face)):
        outer_face_ccw_nbr[prev_nbr] = outer_face[idx]
        prev_nbr = outer_face[idx]
    outer_face_ccw_nbr[prev_nbr] = v1

    outer_face_cw_nbr: dict[int, int] = {}
    prev_nbr = v1
    for idx in range(len(outer_face) - 1, 0, -1):
        outer_face_cw_nbr[prev_nbr] = outer_face[idx]
        prev_nbr = outer_face[idx]

    def is_outer_face_nbr(source: int, target: int) -> bool:
        """Return whether two nodes are adjacent on the current outer face.

        Parameters
        ----------
        source : int
            Candidate source node.
        target : int
            Candidate target node.

        Returns
        -------
        bool
            ``True`` when ``target`` is a clockwise or counterclockwise neighbor.
        """
        if source not in outer_face_ccw_nbr:
            return outer_face_cw_nbr[source] == target
        if source not in outer_face_cw_nbr:
            return outer_face_ccw_nbr[source] == target
        return outer_face_ccw_nbr[source] == target or outer_face_cw_nbr[source] == target

    def is_on_outer_face(node: int) -> bool:
        """Return whether a node is still on the current outer face.

        Parameters
        ----------
        node : int
            Candidate node.

        Returns
        -------
        bool
            ``True`` when the node is unmarked and on the contour.
        """
        return node not in marked_nodes and (node in outer_face_ccw_nbr or node == v1)

    for node in outer_face:
        for nbr in embedding.neighbors_cw_order(node):
            if is_on_outer_face(nbr) and not is_outer_face_nbr(node, nbr):
                chords[node] += 1
                ready_to_pick.discard(node)

    canonical_ordering: list[tuple[int, list[int]]] = [(0, [])] * len(embedding.nodes())
    canonical_ordering[0] = (v1, [])
    canonical_ordering[1] = (v2, [])
    ready_to_pick.discard(v1)
    ready_to_pick.discard(v2)

    for k in range(len(embedding.nodes()) - 1, 1, -1):
        node = ready_to_pick.pop()
        marked_nodes.add(node)

        wp = None
        wq = None
        nbr_iterator = iter(embedding.neighbors_cw_order(node))
        while True:
            nbr = next(nbr_iterator)
            if nbr in marked_nodes:
                continue
            if is_on_outer_face(nbr):
                if nbr == v1:
                    wp = v1
                elif nbr == v2:
                    wq = v2
                elif outer_face_cw_nbr[nbr] == node:
                    wp = nbr
                else:
                    wq = nbr
            if wp is not None and wq is not None:
                break

        wp_wq = [wp]
        nbr = wp
        while nbr != wq:
            if nbr is None:
                raise RuntimeError("Canonical contour endpoint missing.")
            next_nbr = embedding[node][nbr]["ccw"]
            wp_wq.append(next_nbr)
            outer_face_cw_nbr[nbr] = next_nbr
            outer_face_ccw_nbr[next_nbr] = nbr
            nbr = next_nbr

        if len(wp_wq) == 2:
            for endpoint in (wp, wq):
                if endpoint is None:
                    raise RuntimeError("Canonical contour endpoint missing.")
                chords[endpoint] -= 1
                if chords[endpoint] == 0:
                    ready_to_pick.add(endpoint)
        else:
            new_face_nodes = set(wp_wq[1:-1])
            for contour_node in new_face_nodes:
                ready_to_pick.add(contour_node)
                for nbr in embedding.neighbors_cw_order(contour_node):
                    if is_on_outer_face(nbr) and not is_outer_face_nbr(contour_node, nbr):
                        chords[contour_node] += 1
                        ready_to_pick.discard(contour_node)
                        if nbr not in new_face_nodes:
                            chords[nbr] += 1
                            ready_to_pick.discard(nbr)
        canonical_ordering[k] = (node, wp_wq)  # type: ignore[list-item]

    return canonical_ordering


def _set_position(
    parent: int,
    tree: dict[int, Optional[int]],
    remaining_nodes: list[int],
    delta_x: dict[int, int],
    y_coordinate: dict[int, int],
    pos: dict[int, tuple[int, int]],
) -> None:
    """Set the absolute position of a child in the relative placement tree.

    Parameters
    ----------
    parent : int
        Parent node with an already-known absolute position.
    tree : dict[int, int | None]
        Child mapping.
    remaining_nodes : list[int]
        Stack of children still needing traversal.
    delta_x : dict[int, int]
        Relative x offsets.
    y_coordinate : dict[int, int]
        Absolute y coordinates.
    pos : dict[int, tuple[int, int]]
        Position output mapping.

    Returns
    -------
    None
        ``pos`` and ``remaining_nodes`` are mutated in place.
    """
    child = tree[parent]
    parent_node_x = pos[parent][0]
    if child is not None:
        child_x = parent_node_x + delta_x[child]
        pos[child] = (child_x, y_coordinate[child])
        remaining_nodes.append(child)


def combinatorial_embedding_to_pos(
    embedding: PlanarEmbedding,
    fully_triangulate: bool = False,
) -> dict[int, tuple[int, int]]:
    """Assign integer grid positions from a combinatorial embedding.

    Parameters
    ----------
    embedding : PlanarEmbedding
        Planar combinatorial embedding.
    fully_triangulate : bool, default=False
        Whether to triangulate the chosen outer face.

    Returns
    -------
    dict[int, tuple[int, int]]
        Raw integer coordinates keyed by node id.
    """
    if len(embedding.nodes()) < 4:
        default_positions = [(0, 0), (2, 0), (1, 1)]
        return {node: default_positions[idx] for idx, node in enumerate(embedding.nodes())}

    embedding, outer_face = triangulate_embedding(embedding, fully_triangulate)
    left_t_child: dict[int, Optional[int]] = {}
    right_t_child: dict[int, Optional[int]] = {}
    delta_x: dict[int, int] = {}
    y_coordinate: dict[int, int] = {}
    node_list = get_canonical_ordering(embedding, outer_face)

    v1 = node_list[0][0]
    v2 = node_list[1][0]
    v3 = node_list[2][0]
    delta_x[v1] = 0
    y_coordinate[v1] = 0
    right_t_child[v1] = v3
    left_t_child[v1] = None
    delta_x[v2] = 1
    y_coordinate[v2] = 0
    right_t_child[v2] = None
    left_t_child[v2] = None
    delta_x[v3] = 1
    y_coordinate[v3] = 1
    right_t_child[v3] = v2
    left_t_child[v3] = None

    for k in range(3, len(node_list)):
        vk, contour_nbrs = node_list[k]
        wp = contour_nbrs[0]
        wp1 = contour_nbrs[1]
        wq = contour_nbrs[-1]
        wq1 = contour_nbrs[-2]
        adds_mult_tri = len(contour_nbrs) > 2

        delta_x[wp1] += 1
        delta_x[wq] += 1

        delta_x_wp_wq = sum(delta_x[node] for node in contour_nbrs[1:])
        delta_x[vk] = (-y_coordinate[wp] + delta_x_wp_wq + y_coordinate[wq]) // 2
        y_coordinate[vk] = (y_coordinate[wp] + delta_x_wp_wq + y_coordinate[wq]) // 2
        delta_x[wq] = delta_x_wp_wq - delta_x[vk]
        if adds_mult_tri:
            delta_x[wp1] -= delta_x[vk]

        right_t_child[wp] = vk
        right_t_child[vk] = wq
        if adds_mult_tri:
            left_t_child[vk] = wp1
            right_t_child[wq1] = None
        else:
            left_t_child[vk] = None

    pos: dict[int, tuple[int, int]] = {v1: (0, y_coordinate[v1])}
    remaining_nodes = [v1]
    while remaining_nodes:
        parent_node = remaining_nodes.pop()
        _set_position(parent_node, left_t_child, remaining_nodes, delta_x, y_coordinate, pos)
        _set_position(parent_node, right_t_child, remaining_nodes, delta_x, y_coordinate, pos)
    return pos


def rescale_layout(pos: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Rescale positions to NetworkX-compatible centered extents.

    Parameters
    ----------
    pos : numpy.ndarray
        Position array with shape ``[N, 2]``.
    scale : float, default=1.0
        Target maximum absolute coordinate.

    Returns
    -------
    numpy.ndarray
        Rescaled positions, mutated and returned for convenience.
    """
    pos -= pos.mean(axis=0)
    lim = np.abs(pos).max()
    if lim > 0:
        pos *= scale / lim
    return pos


class PlanarityCheck(Op):
    """Compute and cache the combinatorial planar embedding."""

    name: ClassVar[str] = "planarity_check"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Store the embedding in ``state.extras``.

        Parameters
        ----------
        problem : LayoutProblem
            Layout inputs.
        state : SolveState
            Mutable pipeline state.
        ctx : RuntimeContext
            Runtime context, unused.

        Returns
        -------
        SolveState
            State containing the planar embedding.
        """
        del ctx
        is_planar, embedding = check_planarity(problem.edge_index, problem.num_nodes)
        if not is_planar or embedding is None:
            raise PlanarityError("G is not planar.")
        state.extras[_PLANAR_EMBEDDING_KEY] = embedding
        return state


class ChrobakPayneShiftPlacement(Op):
    """Place nodes using the Chrobak-Payne shift method."""

    name: ClassVar[str] = "chrobak_payne_shift_placement"
    category: ClassVar[OpCategory] = OpCategory.INIT
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute raw integer-grid coordinates.

        Parameters
        ----------
        problem : LayoutProblem
            Layout inputs, unused after embedding construction.
        state : SolveState
            Mutable pipeline state with a cached embedding.
        ctx : RuntimeContext
            Runtime context, unused.

        Returns
        -------
        SolveState
            State containing raw coordinate mapping.
        """
        del problem, ctx
        embedding = state.extras.get(_PLANAR_EMBEDDING_KEY)
        if not isinstance(embedding, PlanarEmbedding):
            raise RuntimeError("Planar embedding is missing from pipeline state.")
        state.extras[_PLANAR_RAW_POS_KEY] = combinatorial_embedding_to_pos(embedding)
        return state


class NetworkXPlanarRescale(Op):
    """Convert raw planar coordinates to the public scaled tensor output."""

    name: ClassVar[str] = "planar_rescale"
    category: ClassVar[OpCategory] = OpCategory.POSTPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("pos",)

    def __init__(self, scale: float = 1.0) -> None:
        """Store output scale.

        Parameters
        ----------
        scale : float, default=1.0
            Target maximum absolute coordinate.
        """
        self.scale = scale

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Populate ``state.pos`` with scaled coordinates.

        Parameters
        ----------
        problem : LayoutProblem
            Layout inputs containing node count and output device.
        state : SolveState
            Mutable pipeline state with raw positions.
        ctx : RuntimeContext
            Runtime context, unused.

        Returns
        -------
        SolveState
            State with ``pos`` set to shape ``[N, 2]``.
        """
        del ctx
        raw_pos = state.extras.get(_PLANAR_RAW_POS_KEY)
        if not isinstance(raw_pos, dict):
            raise RuntimeError("Raw planar positions are missing from pipeline state.")
        node_list = list(state.extras[_PLANAR_EMBEDDING_KEY])
        if not node_list:
            state.pos = torch.empty(
                (problem.num_nodes, 2),
                dtype=torch.float64,
                device=problem.edge_index.device,
            )
            return state
        pos_array = np.vstack([raw_pos[node] for node in node_list]).astype(np.float64)
        pos_array = rescale_layout(pos_array, scale=self.scale)
        output = np.zeros((problem.num_nodes, 2), dtype=np.float64)
        for row, node in enumerate(node_list):
            output[node] = pos_array[row]
        state.pos = torch.from_numpy(output).to(device=problem.edge_index.device)
        return state


def build_planar_pipeline(scale: float = 1.0) -> Pipeline:
    """Build the Chrobak-Payne planar layout pipeline.

    Parameters
    ----------
    scale : float, default=1.0
        Target maximum absolute coordinate.

    Returns
    -------
    Pipeline
        Three-stage planar pipeline.
    """
    return Pipeline(
        [PlanarityCheck(), ChrobakPayneShiftPlacement(), NetworkXPlanarRescale(scale=scale)],
        name="planar_pipeline",
    )


def layout_planar_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: Optional[int] = 42,
    edge_weights: Optional[torch.Tensor] = None,
    scale: float = 1.0,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run the deterministic Chrobak-Payne planar layout pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor | None, optional
        Optional node-size tensor with shape ``[N, 2]``.
    seed : int | None, default=42
        Accepted for API consistency; planar layout is deterministic.
    edge_weights : torch.Tensor | None, optional
        Accepted for API consistency; planar layout ignores weights.
    scale : float, default=1.0
        Target maximum absolute coordinate.
    fidelity_dtype : torch.dtype | None, optional
        Optional output dtype override for fidelity checks.

    Returns
    -------
    torch.Tensor
        Coordinate tensor with shape ``[N, 2]``.
    """
    del seed, edge_weights
    problem = LayoutProblem(edge_index=edge_index, num_nodes=num_nodes, node_sizes=node_sizes)
    state = build_planar_pipeline(scale=scale).apply(
        problem,
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device="cpu")),
    )
    if state.pos is None:
        raise RuntimeError("Planar pipeline did not produce positions.")
    if fidelity_dtype is not None:
        return state.pos.to(dtype=fidelity_dtype)
    return state.pos


__all__ = [
    "ChrobakPayneShiftPlacement",
    "NetworkXPlanarRescale",
    "PlanarEmbedding",
    "PlanarityCheck",
    "PlanarityError",
    "build_planar_pipeline",
    "check_planarity",
    "combinatorial_embedding_to_pos",
    "get_canonical_ordering",
    "layout_planar_pipeline",
    "rescale_layout",
    "triangulate_embedding",
]
