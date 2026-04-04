"""Coordinate assignment operations for layered and tree layouts."""

from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

from dagua.layout._archive.classic.sugiyama import _brandes_koepf_x_positions, _resolve_node_sizes
from dagua.layout.ops.base import Op
from dagua.layout.ops.graph_utils import (
    build_undirected_adjacency as _build_undirected_adjacency,
)
from dagua.layout.ops.graph_utils import (
    layout_device as _layout_device,
)
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op


def _target_device(problem: LayoutProblem, state: SolveState) -> torch.device:
    """Resolve the device used for persisted position tensors.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs.
    state : SolveState
        Mutable solve state.

    Returns
    -------
    torch.device
        Device for the updated ``state.pos`` tensor.
    """
    if state.pos is not None:
        return state.pos.device
    if problem.node_sizes is not None:
        return problem.node_sizes.device
    return problem.edge_index.device


def _validate_layers(layers: Optional[torch.Tensor], num_nodes: int) -> torch.Tensor:
    """Return validated CPU layer assignments.

    Parameters
    ----------
    layers : torch.Tensor | None
        Candidate layer tensor with shape ``[N]``.
    num_nodes : int
        Expected node count.

    Returns
    -------
    torch.Tensor
        CPU long tensor with shape ``[N]``.

    Raises
    ------
    ValueError
        If ``layers`` is missing or has an invalid shape.
    """
    if layers is None:
        raise ValueError("state.layers must be populated before coordinate ops run")
    if layers.ndim != 1 or layers.shape[0] != num_nodes:
        raise ValueError(f"state.layers must have shape [{num_nodes}]")
    return layers.detach().to(device="cpu", dtype=torch.long)


def _validate_ordering(ordering: Optional[torch.Tensor], num_nodes: int) -> torch.Tensor:
    """Return validated CPU ordering indices.

    Parameters
    ----------
    ordering : torch.Tensor | None
        Candidate ordering tensor with shape ``[N]``.
    num_nodes : int
        Expected node count.

    Returns
    -------
    torch.Tensor
        CPU long tensor with shape ``[N]``.

    Raises
    ------
    ValueError
        If ``ordering`` is missing or has an invalid shape.
    """
    if ordering is None:
        raise ValueError("state.ordering must be populated before this op runs")
    if ordering.ndim != 1 or ordering.shape[0] != num_nodes:
        raise ValueError(f"state.ordering must have shape [{num_nodes}]")
    return ordering.detach().to(device="cpu", dtype=torch.long)


def _validate_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Return a validated CPU edge list.

    Parameters
    ----------
    edge_index : torch.Tensor
        Input edge tensor expected to have shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    torch.Tensor
        CPU long tensor with shape ``[2, E]``.

    Raises
    ------
    ValueError
        If the edge tensor shape or node references are invalid.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("problem.edge_index must have shape [2, E]")
    edge_index_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    if edge_index_cpu.numel() == 0:
        return edge_index_cpu
    if int(edge_index_cpu.min().item()) < 0 or int(edge_index_cpu.max().item()) >= num_nodes:
        raise ValueError("problem.edge_index references a node outside the valid range")
    return edge_index_cpu


def _ordered_layers_from_state(layers: torch.Tensor, ordering: torch.Tensor) -> List[List[int]]:
    """Build ordered layer lists from per-node layer and ordering tensors.

    Parameters
    ----------
    layers : torch.Tensor
        CPU layer assignments with shape ``[N]``.
    ordering : torch.Tensor
        CPU in-layer ordering values with shape ``[N]``.

    Returns
    -------
    list[list[int]]
        Node ids grouped by layer and sorted left-to-right.
    """
    if layers.numel() == 0:
        return []

    num_layers = int(layers.max().item()) + 1
    ordered_layers: List[List[int]] = [[] for _ in range(num_layers)]
    for node in range(layers.shape[0]):
        ordered_layers[int(layers[node].item())].append(node)

    for layer_nodes in ordered_layers:
        layer_nodes.sort(key=lambda node_idx: (int(ordering[node_idx].item()), node_idx))
    return ordered_layers


def _layered_neighbors_from_edges(
    edge_index: torch.Tensor,
    layers: torch.Tensor,
    num_nodes: int,
) -> Tuple[List[List[int]], List[List[int]]]:
    """Orient edges according to the layer assignment.

    Parameters
    ----------
    edge_index : torch.Tensor
        CPU edge tensor with shape ``[2, E]``.
    layers : torch.Tensor
        CPU layer assignments with shape ``[N]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    tuple
        ``(parents, children)`` indexed by node id.
    """
    parents: List[List[int]] = [[] for _ in range(num_nodes)]
    children: List[List[int]] = [[] for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return parents, children

    for source, target in edge_index.t().tolist():
        source_layer = int(layers[source].item())
        target_layer = int(layers[target].item())
        if source_layer == target_layer:
            continue
        if source_layer < target_layer:
            if source not in parents[target]:
                parents[target].append(source)
            if target not in children[source]:
                children[source].append(target)
            continue
        if target not in parents[source]:
            parents[source].append(target)
        if source not in children[target]:
            children[target].append(source)
    return parents, children


def _node_spacing(
    node_sizes: Optional[torch.Tensor],
    axis: int,
    default: float,
) -> float:
    """Estimate sibling and layer spacing from optional node sizes.

    Parameters
    ----------
    node_sizes : torch.Tensor | None
        Optional node-size tensor with shape ``[N, 2]``.
    axis : int
        Axis index to read from each node size.
    default : float
        Fallback spacing value when node sizes are missing or empty.

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
    """Order BFS roots by indegree and index for deterministic output.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    list[int]
        BFS roots sorted by ``(indegree != 0, indegree, node_id)``.
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
    """Build a deterministic BFS forest from a possibly directed graph.

    Parameters
    ----------
    edge_index : torch.Tensor
        CPU edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    tuple[list[int], list[list[int]], list[int]]
        Forest roots, child lists, and BFS depth per node.
    """
    adjacency = _build_undirected_adjacency(
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
    """Walker bookkeeping node for Buchheim-style tree coordinate assignment."""

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
        """Initialize the default ancestor to the current node."""
        self.ancestor = self

    def left(self) -> Optional["_WalkerNode"]:
        """Return the first leftward contour node."""
        if self.thread is not None:
            return self.thread
        if self.children:
            return self.children[0]
        return None

    def right(self) -> Optional["_WalkerNode"]:
        """Return the first rightward contour node."""
        if self.thread is not None:
            return self.thread
        if self.children:
            return self.children[-1]
        return None

    def left_brother(self) -> Optional["_WalkerNode"]:
        """Return the immediate left sibling, if any."""
        if self.parent is None:
            return None
        previous_sibling: Optional[_WalkerNode] = None
        for sibling in self.parent.children:
            if sibling is self:
                return previous_sibling
            previous_sibling = sibling
        return None

    def leftmost_sibling(self) -> Optional["_WalkerNode"]:
        """Return cached leftmost sibling in the current sibling set."""
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
    """Build tree state recursively for one BFS-rooted component."""
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
    """Resolve the representative ancestor for non-sibling conflict adjustment."""
    if node.parent is None:
        return default_ancestor
    candidate = left_inner.ancestor or default_ancestor
    if candidate in node.parent.children:
        return candidate
    return default_ancestor


def _move_subtree(left_subtree: _WalkerNode, right_subtree: _WalkerNode, shift: float) -> None:
    """Apply the deferred shift across a subtree interval."""
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
    """Apply accumulated sibling shifts from right to left."""
    shift = 0.0
    change = 0.0
    for child in reversed(node.children):
        child.prelim += shift
        child.mod += shift
        change += child.change
        shift += child.shift + change


def _apportion(node: _WalkerNode, default_ancestor: _WalkerNode, distance: float) -> _WalkerNode:
    """Resolve contour overlaps between the current node and previous siblings."""
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
    """Assign preliminary x-values in a post-order pass."""
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
    """Propagate accumulated modifiers top-down to absolute x coordinates."""
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
    """Generate absolute x positions for one BFS-rooted component."""
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


@dataclass(frozen=True)
class ReingoldTilfordTreeConfig:
    """Configuration for :class:`ReingoldTilfordTree`.

    Parameters
    ----------
    sibling_sep : float | None, default=None
        Horizontal sibling spacing in tree units. ``None`` resolves from
        ``problem.node_sizes[:, 0]`` with a ``1.5x`` multiplier and a ``1.0``
        minimum floor.
    layer_sep : float | None, default=None
        Vertical layer spacing in tree units. ``None`` resolves from
        ``problem.node_sizes[:, 1]`` with a ``1.5x`` multiplier and a ``1.5``
        minimum floor.
    component_gap : float | None, default=None
        Extra spacing between disconnected components.
    horizontal : bool, default=False
        Rotate the output so depth maps to x when ``True``.
    """

    sibling_sep: Optional[float] = None
    layer_sep: Optional[float] = None
    component_gap: Optional[float] = None
    horizontal: bool = False


@dataclass(frozen=True)
class BrandesKopf4PassConfig:
    """Configuration for :class:`BrandesKopf4Pass`.

    Parameters
    ----------
    node_sep : float, default=1.0
        Horizontal gap between neighboring nodes.
    rank_sep : float, default=1.0
        Vertical gap between consecutive layers.
    """

    node_sep: float = 1.0
    rank_sep: float = 1.0


@dataclass(frozen=True)
class BucheimWalkerTreeConfig:
    """Configuration for :class:`BucheimWalkerTree`.

    Parameters
    ----------
    sibling_sep : float, default=1.0
        Horizontal spacing multiplier between neighboring siblings.
    layer_sep : float, default=1.5
        Vertical spacing between consecutive tree depths.
    component_gap : float, default=2.0
        Horizontal gap between disconnected tree components.
    """

    sibling_sep: float = 1.0
    layer_sep: float = 1.5
    component_gap: float = 2.0


@register_op
class BrandesKopf4Pass(Op):
    """Assign x coordinates via four-pass Brandes-Kopf compaction."""

    name = "brandes_kopf_4pass"
    category = OpCategory.COORDINATE
    reads = ("layers", "ordering")
    writes = ("pos",)
    requires = ("layers", "ordering")

    def __init__(self, config: Optional[BrandesKopf4PassConfig] = None) -> None:
        """Store the coordinate-assignment configuration.

        Parameters
        ----------
        config : BrandesKopf4PassConfig | None, optional
            Optional op configuration.
        """
        self.config = config or BrandesKopf4PassConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Convert a layered ordering into concrete node coordinates.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with ``layers`` and ``ordering`` populated.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this deterministic op.

        Returns
        -------
        SolveState
            Updated state with ``pos`` populated.
        """
        del ctx

        layers_cpu = _validate_layers(state.layers, problem.num_nodes)
        ordering_cpu = _validate_ordering(state.ordering, problem.num_nodes)
        edge_index_cpu = _validate_edge_index(problem.edge_index, problem.num_nodes)
        ordered_layers = _ordered_layers_from_state(layers_cpu, ordering_cpu)
        parents, children = _layered_neighbors_from_edges(
            edge_index=edge_index_cpu,
            layers=layers_cpu,
            num_nodes=problem.num_nodes,
        )
        node_sizes_cpu = _resolve_node_sizes(problem.node_sizes, problem.num_nodes)

        positions = torch.zeros((problem.num_nodes, 2), dtype=torch.float32)
        if problem.num_nodes > 0:
            x_coordinates = _brandes_koepf_x_positions(
                layers=ordered_layers,
                parents=parents,
                children=children,
                node_sizes=node_sizes_cpu,
                num_nodes=problem.num_nodes,
                num_original_nodes=problem.num_nodes,
                node_sep=self.config.node_sep,
            )
            positions[:, 0] = torch.tensor(x_coordinates, dtype=torch.float32)
            positions[:, 1] = layers_cpu.to(dtype=torch.float32) * self.config.rank_sep

        state.pos = positions.to(device=_target_device(problem, state))
        return state


@register_op
class BucheimWalkerTree(Op):
    """Lay out the graph as a tidy BFS forest using Buchheim's algorithm."""

    name = "bucheim_walker_tree"
    category = OpCategory.COORDINATE
    writes = ("pos",)

    def __init__(self, config: Optional[BucheimWalkerTreeConfig] = None) -> None:
        """Store the tree-layout configuration.

        Parameters
        ----------
        config : BucheimWalkerTreeConfig | None, optional
            Optional op configuration.
        """
        self.config = config or BucheimWalkerTreeConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Assign tidy tree coordinates directly from the graph topology.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this deterministic op.

        Returns
        -------
        SolveState
            Updated state with ``pos`` populated.
        """
        del ctx

        edge_index_cpu = _validate_edge_index(problem.edge_index, problem.num_nodes)
        target_device = _target_device(problem, state)
        if problem.num_nodes == 0:
            state.pos = torch.empty((0, 2), dtype=torch.float32, device=target_device)
            return state

        sys.setrecursionlimit(max(sys.getrecursionlimit(), problem.num_nodes * 2))
        roots, children, depths = _bfs_forest(
            edge_index=edge_index_cpu,
            num_nodes=problem.num_nodes,
        )

        preliminary_x = [0.0] * problem.num_nodes
        next_component_offset = 0.0
        for root in roots:
            next_component_offset = _assign_preliminary_x(
                root_idx=root,
                children=children,
                depths=depths,
                preliminary_x=preliminary_x,
                component_offset=next_component_offset,
                component_gap=self.config.component_gap,
            )

        positions = torch.zeros((problem.num_nodes, 2), dtype=torch.float32)
        for node_idx in range(problem.num_nodes):
            positions[node_idx, 0] = float(preliminary_x[node_idx]) * self.config.sibling_sep
            positions[node_idx, 1] = float(depths[node_idx]) * self.config.layer_sep

        positions -= positions.mean(dim=0, keepdim=True)
        state.pos = positions.to(device=target_device)
        return state


@register_op
class ReingoldTilfordTree(Op):
    """Assign Reingold-Tilford coordinates with optional size-aware spacing."""

    name = "reingold_tilford_tree"
    category = OpCategory.COORDINATE
    writes = ("pos",)

    def __init__(self, config: Optional[ReingoldTilfordTreeConfig] = None) -> None:
        """Store the tree configuration.

        Parameters
        ----------
        config : ReingoldTilfordTreeConfig | None, optional
            Optional op configuration.
        """
        self.config = config or ReingoldTilfordTreeConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute exact Reingold-Tilford coordinates for one problem.

        Parameters
        ----------
        problem : LayoutProblem
            Input graph and optional node size hints.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context (unused).

        Returns
        -------
        SolveState
            Updated state with ``state.pos`` set.
        """
        del ctx

        output_device = _layout_device(edge_index=problem.edge_index, node_sizes=problem.node_sizes)
        edge_index_cpu = _validate_edge_index(problem.edge_index, problem.num_nodes)

        if problem.num_nodes == 0:
            state.pos = torch.empty((0, 2), dtype=torch.float32, device=output_device)
            return state

        sibling_sep = (
            _node_spacing(problem.node_sizes, axis=0, default=1.0)
            if self.config.sibling_sep is None
            else self.config.sibling_sep
        )
        layer_sep = (
            _node_spacing(problem.node_sizes, axis=1, default=1.5)
            if self.config.layer_sep is None
            else self.config.layer_sep
        )
        component_gap = (
            sibling_sep * 2.0 if self.config.component_gap is None else self.config.component_gap
        )

        sys.setrecursionlimit(max(sys.getrecursionlimit(), problem.num_nodes * 2))
        roots, children, depths = _bfs_forest(
            edge_index=edge_index_cpu,
            num_nodes=problem.num_nodes,
        )

        preliminary_x = [0.0] * problem.num_nodes
        next_component_offset = 0.0
        for root in roots:
            next_component_offset = _assign_preliminary_x(
                root_idx=root,
                children=children,
                depths=depths,
                preliminary_x=preliminary_x,
                component_offset=next_component_offset,
                component_gap=component_gap,
            )

        positions = torch.zeros((problem.num_nodes, 2), dtype=torch.float32)
        for node_idx in range(problem.num_nodes):
            positions[node_idx, 0] = float(preliminary_x[node_idx]) * sibling_sep
            positions[node_idx, 1] = float(depths[node_idx]) * layer_sep

        positions -= positions.mean(dim=0, keepdim=True)
        if self.config.horizontal:
            positions = positions[:, [1, 0]]
        state.pos = positions.to(device=output_device, dtype=torch.float32)
        return state
