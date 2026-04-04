"""Deterministic tidy tree layout via a Reingold-Tilford style traversal."""

from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import torch

from dagua.layout._archive.classic._graph_distances import (
    build_undirected_adjacency as _shared_build_undirected_adjacency,
)


def _layout_device(
    edge_index: torch.Tensor,
    node_sizes: Optional[torch.Tensor],
) -> torch.device:
    """Choose the device used for the returned layout tensor.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``.

    Returns
    -------
    torch.device
        Device for the final output tensor.
    """
    if edge_index.numel() > 0:
        return edge_index.device
    if node_sizes is not None:
        return node_sizes.device
    return torch.device("cpu")


def _node_spacing(
    node_sizes: Optional[torch.Tensor],
    axis: int,
    default: float,
) -> float:
    """Estimate spacing along one axis from node sizes.

    Parameters
    ----------
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``.
    axis : int
        Axis index inside ``node_sizes``.
    default : float
        Fallback spacing when node sizes are unavailable.

    Returns
    -------
    float
        Spacing multiplier for the requested axis.
    """
    if node_sizes is None or node_sizes.numel() == 0:
        return default
    max_size = float(node_sizes.to(dtype=torch.float32, device="cpu")[:, axis].max().item())
    return max(max_size * 1.5, default)


def _root_candidates(edge_index: torch.Tensor, num_nodes: int) -> list[int]:
    """Rank root candidates with zero-indegree nodes first.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    list[int]
        Node indices in deterministic root preference order.
    """
    indegree = [0] * num_nodes
    if edge_index.numel() > 0:
        targets = edge_index.detach().to(device="cpu", dtype=torch.long)[1].tolist()
        for target in targets:
            indegree[target] += 1
    return sorted(
        range(num_nodes),
        key=lambda node_idx: (indegree[node_idx] != 0, indegree[node_idx], node_idx),
    )


def _bfs_forest(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> tuple[list[int], list[list[int]], list[int]]:
    """Extract a deterministic BFS forest from an arbitrary graph.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    tuple[list[int], list[list[int]], list[int]]
        Forest roots, child lists per node, and node depths.
    """
    adjacency = _shared_build_undirected_adjacency(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weights=None,
    )
    children: list[list[int]] = [[] for _ in range(num_nodes)]
    depths = [0] * num_nodes
    visited = [False] * num_nodes
    roots: list[int] = []

    for root in _root_candidates(edge_index=edge_index, num_nodes=num_nodes):
        if visited[root]:
            continue
        roots.append(root)
        visited[root] = True
        queue: deque[int] = deque([root])
        while queue:
            node = queue.popleft()
            for neighbor, _ in adjacency[node]:
                if visited[neighbor]:
                    continue
                visited[neighbor] = True
                depths[neighbor] = depths[node] + 1
                children[node].append(neighbor)
                queue.append(neighbor)

    return roots, children, depths


@dataclass
class _WalkerNode:
    """Store Walker/Buchheim layout state for one rooted tree node.

    Parameters
    ----------
    node_idx : int
        Original graph node index.
    depth : int
        Depth inside the rooted BFS forest.
    number : int
        One-based sibling order within the parent child list.
    parent : _WalkerNode, optional
        Parent Walker node for the rooted tree.
    children : list[_WalkerNode]
        Child Walker nodes in deterministic left-to-right order.
    prelim : float, default=0.0
        Preliminary x coordinate from the first walk.
    mod : float, default=0.0
        Modifier accumulated during apportioning.
    shift : float, default=0.0
        Deferred shift contribution applied in ``_execute_shifts``.
    change : float, default=0.0
        Deferred change term used to propagate sibling adjustments.
    thread : _WalkerNode, optional
        Temporary contour thread linking disjoint subtree boundaries.
    ancestor : _WalkerNode, optional
        Current ancestor representative used for non-sibling conflict resolution.
    _leftmost_sibling : _WalkerNode, optional
        Cached leftmost sibling pointer used during apportioning.
    """

    node_idx: int
    depth: int
    number: int
    parent: Optional["_WalkerNode"] = None
    children: list["_WalkerNode"] = field(default_factory=list)
    prelim: float = 0.0
    mod: float = 0.0
    shift: float = 0.0
    change: float = 0.0
    thread: Optional["_WalkerNode"] = None
    ancestor: Optional["_WalkerNode"] = None
    _leftmost_sibling: Optional["_WalkerNode"] = None

    def __post_init__(self) -> None:
        """Initialize the default ancestor to the node itself."""
        self.ancestor = self

    def left(self) -> Optional["_WalkerNode"]:
        """Return the next node on the left contour.

        Returns
        -------
        _WalkerNode, optional
            Leftmost child when available, otherwise the thread pointer.
        """
        if self.thread is not None:
            return self.thread
        if self.children:
            return self.children[0]
        return None

    def right(self) -> Optional["_WalkerNode"]:
        """Return the next node on the right contour.

        Returns
        -------
        _WalkerNode, optional
            Rightmost child when available, otherwise the thread pointer.
        """
        if self.thread is not None:
            return self.thread
        if self.children:
            return self.children[-1]
        return None

    def left_brother(self) -> Optional["_WalkerNode"]:
        """Return the immediate sibling on the left.

        Returns
        -------
        _WalkerNode, optional
            Left sibling or ``None`` when this node is first in its sibling group.
        """
        if self.parent is None:
            return None
        previous_sibling: Optional[_WalkerNode] = None
        for sibling in self.parent.children:
            if sibling is self:
                return previous_sibling
            previous_sibling = sibling
        return None

    def leftmost_sibling(self) -> Optional["_WalkerNode"]:
        """Return the leftmost sibling in the current sibling group.

        Returns
        -------
        _WalkerNode, optional
            Leftmost sibling when this node has siblings, otherwise ``None``.
        """
        if self._leftmost_sibling is None and self.parent is not None:
            first_child = self.parent.children[0]
            if first_child is not self:
                self._leftmost_sibling = first_child
        return self._leftmost_sibling


def _build_walker_tree(
    root_idx: int,
    children: list[list[int]],
    depth: int,
    number: int,
    subtree_nodes: list[int],
    parent: Optional[_WalkerNode] = None,
) -> _WalkerNode:
    """Create Walker bookkeeping nodes for one rooted subtree.

    Parameters
    ----------
    root_idx : int
        Root node index in the BFS forest.
    children : list[list[int]]
        BFS-forest child lists.
    depth : int
        Depth assigned to ``root_idx``.
    number : int
        One-based sibling order for ``root_idx``.
    subtree_nodes : list[int]
        Output list collecting all graph node ids inside the subtree.
    parent : _WalkerNode, optional
        Parent Walker node, if any.

    Returns
    -------
    _WalkerNode
        Root Walker state for the rooted subtree.
    """
    subtree_nodes.append(root_idx)
    walker_node = _WalkerNode(
        node_idx=root_idx,
        depth=depth,
        number=number,
        parent=parent,
    )
    for child_number, child_idx in enumerate(children[root_idx], start=1):
        walker_node.children.append(
            _build_walker_tree(
                root_idx=child_idx,
                children=children,
                depth=depth + 1,
                number=child_number,
                subtree_nodes=subtree_nodes,
                parent=walker_node,
            )
        )
    return walker_node


def _walker_ancestor(
    left_inner: _WalkerNode,
    node: _WalkerNode,
    default_ancestor: _WalkerNode,
) -> _WalkerNode:
    """Resolve the ancestor used to shift conflicting non-sibling subtrees.

    Parameters
    ----------
    left_inner : _WalkerNode
        Right contour node from the left subtree during apportioning.
    node : _WalkerNode
        Current subtree root being apportioned.
    default_ancestor : _WalkerNode
        Fallback ancestor when ``left_inner`` no longer points into the sibling set.

    Returns
    -------
    _WalkerNode
        Ancestor node that should receive the propagated subtree shift.
    """
    if node.parent is None:
        return default_ancestor
    candidate = left_inner.ancestor or default_ancestor
    if candidate in node.parent.children:
        return candidate
    return default_ancestor


def _move_subtree(left_subtree: _WalkerNode, right_subtree: _WalkerNode, shift: float) -> None:
    """Move a subtree rightward and defer the shared shift to siblings.

    Parameters
    ----------
    left_subtree : _WalkerNode
        Ancestor on the left side receiving the balancing change term.
    right_subtree : _WalkerNode
        Conflicting subtree that must move right.
    shift : float
        Horizontal displacement measured in Walker spacing units.
    """
    subtree_count = right_subtree.number - left_subtree.number
    if subtree_count <= 0:
        return
    shift_per_subtree = shift / float(subtree_count)
    right_subtree.change -= shift_per_subtree
    right_subtree.shift += shift
    left_subtree.change += shift_per_subtree
    right_subtree.prelim += shift
    right_subtree.mod += shift


def _execute_shifts(node: _WalkerNode) -> None:
    """Apply deferred sibling shifts from right to left.

    Parameters
    ----------
    node : _WalkerNode
        Parent node whose child adjustments are ready to be committed.
    """
    shift = 0.0
    change = 0.0
    for child in reversed(node.children):
        child.prelim += shift
        child.mod += shift
        change += child.change
        shift += child.shift + change


def _apportion(node: _WalkerNode, default_ancestor: _WalkerNode, distance: float) -> _WalkerNode:
    """Resolve subtree contour conflicts for one sibling group.

    Parameters
    ----------
    node : _WalkerNode
        Current subtree root being inserted into its sibling group.
    default_ancestor : _WalkerNode
        Ancestor representative carried across previous siblings.
    distance : float
        Minimum spacing between adjacent sibling subtrees.

    Returns
    -------
    _WalkerNode
        Updated default ancestor for later siblings.
    """
    left_brother = node.left_brother()
    if left_brother is None:
        return default_ancestor

    inner_right = node
    outer_right = node
    inner_left = left_brother
    outer_left = node.leftmost_sibling()
    if outer_left is None:
        return default_ancestor

    sum_inner_right = node.mod
    sum_outer_right = node.mod
    sum_inner_left = inner_left.mod
    sum_outer_left = outer_left.mod

    while inner_left.right() is not None and inner_right.left() is not None:
        next_inner_left = inner_left.right()
        next_inner_right = inner_right.left()
        next_outer_left = outer_left.left()
        next_outer_right = outer_right.right()
        if (
            next_inner_left is None
            or next_inner_right is None
            or next_outer_left is None
            or next_outer_right is None
        ):
            break

        inner_left = next_inner_left
        inner_right = next_inner_right
        outer_left = next_outer_left
        outer_right = next_outer_right
        outer_right.ancestor = node

        shift = (inner_left.prelim + sum_inner_left) - (inner_right.prelim + sum_inner_right)
        shift += distance
        if shift > 0.0:
            ancestor = _walker_ancestor(inner_left, node, default_ancestor)
            _move_subtree(ancestor, node, shift)
            sum_inner_right += shift
            sum_outer_right += shift

        sum_inner_left += inner_left.mod
        sum_inner_right += inner_right.mod
        sum_outer_left += outer_left.mod
        sum_outer_right += outer_right.mod

    if inner_left.right() is not None and outer_right.right() is None:
        outer_right.thread = inner_left.right()
        outer_right.mod += sum_inner_left - sum_outer_right
    else:
        if inner_right.left() is not None and outer_left.left() is None:
            outer_left.thread = inner_right.left()
            outer_left.mod += sum_inner_right - sum_outer_left
        default_ancestor = node

    return default_ancestor


def _first_walk(node: _WalkerNode, distance: float) -> None:
    """Assign preliminary coordinates and contour threads bottom-up.

    Parameters
    ----------
    node : _WalkerNode
        Root Walker node for the current subtree.
    distance : float
        Minimum spacing between neighboring sibling subtrees.
    """
    if not node.children:
        left_brother = node.left_brother()
        node.prelim = left_brother.prelim + distance if left_brother is not None else 0.0
        return

    default_ancestor = node.children[0]
    for child in node.children:
        _first_walk(child, distance=distance)
        default_ancestor = _apportion(child, default_ancestor, distance=distance)

    _execute_shifts(node)
    midpoint = 0.5 * (node.children[0].prelim + node.children[-1].prelim)
    left_brother = node.left_brother()
    if left_brother is None:
        node.prelim = midpoint
        return

    node.prelim = left_brother.prelim + distance
    node.mod = node.prelim - midpoint


def _second_walk(
    node: _WalkerNode,
    modifier: float,
    coordinates: dict[int, float],
) -> tuple[float, float]:
    """Accumulate modifiers top-down to produce final x coordinates.

    Parameters
    ----------
    node : _WalkerNode
        Root Walker node for the current subtree.
    modifier : float
        Sum of ancestor modifiers reaching ``node``.
    coordinates : dict[int, float]
        Output mapping from graph node id to final x coordinate in Walker units.

    Returns
    -------
    tuple[float, float]
        Inclusive ``(min_x, max_x)`` span for the subtree.
    """
    x_coordinate = node.prelim + modifier
    coordinates[node.node_idx] = x_coordinate
    min_x = x_coordinate
    max_x = x_coordinate
    for child in node.children:
        child_min, child_max = _second_walk(
            child,
            modifier=modifier + node.mod,
            coordinates=coordinates,
        )
        min_x = min(min_x, child_min)
        max_x = max(max_x, child_max)
    return min_x, max_x


def _assign_preliminary_x(
    root_idx: int,
    children: list[list[int]],
    depths: list[int],
    preliminary_x: list[float],
    component_offset: float,
    component_gap: float,
) -> float:
    """Assign Walker x coordinates for one rooted component.

    Parameters
    ----------
    root_idx : int
        Root node of the component subtree.
    children : list[list[int]]
        BFS-forest child lists.
    depths : list[int]
        BFS depth for each node.
    preliminary_x : list[float]
        Mutable x-coordinate buffer for all nodes.
    component_offset : float
        Starting x offset for the current connected component.
    component_gap : float
        Extra spacing inserted between laid-out components.

    Returns
    -------
    float
        Starting offset for the next component in Walker spacing units.
    """
    subtree_nodes: list[int] = []
    root = _build_walker_tree(
        root_idx=root_idx,
        children=children,
        depth=depths[root_idx],
        number=1,
        subtree_nodes=subtree_nodes,
    )
    _first_walk(root, distance=1.0)

    coordinates: dict[int, float] = {}
    min_x, max_x = _second_walk(root, modifier=0.0, coordinates=coordinates)
    normalization_shift = component_offset - min_x
    for node_idx in subtree_nodes:
        preliminary_x[node_idx] = coordinates[node_idx] + normalization_shift

    return component_offset + (max_x - min_x) + 1.0 + component_gap


def layout_reingold_tilford(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: int = 42,
    horizontal: bool = False,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Lay out a graph as a tidy tree or BFS forest.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]`` used to choose stable
        layer and sibling spacing.
    seed : int, default=42
        Accepted for interface compatibility. This layout is deterministic.
    horizontal : bool, default=False
        If ``True``, rotate the final layout so depth grows along the x-axis.
    edge_weights : torch.Tensor, optional
        Accepted for interface compatibility and validated when provided.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If ``num_nodes`` is negative or ``edge_weights`` has the wrong shape.
    """
    _ = seed

    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
            )

    device = _layout_device(edge_index=edge_index, node_sizes=node_sizes)
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)
    sys.setrecursionlimit(max(sys.getrecursionlimit(), num_nodes * 2))

    roots, children, depths = _bfs_forest(edge_index=edge_index, num_nodes=num_nodes)
    sibling_spacing = _node_spacing(node_sizes=node_sizes, axis=0, default=1.0)
    layer_spacing = _node_spacing(node_sizes=node_sizes, axis=1, default=1.5)
    component_gap = sibling_spacing * 2.0

    preliminary_x = [0.0] * num_nodes
    next_component_x = 0.0
    for root in roots:
        next_component_x = _assign_preliminary_x(
            root_idx=root,
            children=children,
            depths=depths,
            preliminary_x=preliminary_x,
            component_offset=next_component_x,
            component_gap=component_gap,
        )

    positions = torch.zeros((num_nodes, 2), dtype=torch.float32)
    for node_idx in range(num_nodes):
        positions[node_idx, 0] = float(preliminary_x[node_idx]) * sibling_spacing
        positions[node_idx, 1] = float(depths[node_idx]) * layer_spacing

    positions -= positions.mean(dim=0, keepdim=True)
    if horizontal:
        positions = positions[:, [1, 0]]
    return positions.to(device=device)
