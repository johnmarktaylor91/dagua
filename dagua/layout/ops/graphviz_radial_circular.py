"""Graphviz-inspired deterministic radial and circular layout operations."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import DefaultDict, Dict, List, Optional, Sequence, Set, Tuple

import torch

from dagua.layout.ops.base import Op
from dagua.layout.ops.graph_utils import bfs_distances, build_undirected_adjacency, layout_device
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_DEFAULT_RANKSEP_POINTS = 72.0
_DEFAULT_NODESEP_POINTS = 18.0
_TWO_PI = 2.0 * math.pi


def _graphviz_twopi_leaf_steps(edge_index: torch.Tensor, num_nodes: int) -> List[int]:
    """Compute Graphviz twopi's minimum steps from each node to any leaf.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    list[int]
        Minimum undirected edge count from each node to a leaf, using
        ``num_nodes * num_nodes`` as Graphviz's unreached sentinel.
    """
    adjacency = build_undirected_adjacency(edge_index, num_nodes)
    sentinel = num_nodes * num_nodes
    steps = [sentinel] * num_nodes
    queue: deque[int] = deque()
    for node, neighbors in enumerate(adjacency):
        distinct_neighbors = {neighbor for neighbor, _ in neighbors if neighbor != node}
        if len(distinct_neighbors) <= 1:
            steps[node] = 0
            queue.append(node)

    while queue:
        node = queue.popleft()
        next_steps = steps[node] + 1
        for neighbor, _ in adjacency[node]:
            if next_steps < steps[neighbor]:
                steps[neighbor] = next_steps
                queue.append(neighbor)
    return steps


def _edge_pairs(edge_index: torch.Tensor) -> List[Tuple[int, int]]:
    """Return edge pairs in CPU input order.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.

    Returns
    -------
    list[tuple[int, int]]
        Source-target pairs in original tensor order.
    """
    if edge_index.numel() == 0:
        return []
    edge_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    return [(int(source), int(target)) for source, target in edge_cpu.t().tolist()]


def choose_twopi_root(edge_index: torch.Tensor, num_nodes: int, root: Optional[int] = None) -> int:
    """Choose the deterministic twopi center node.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    root : int, optional
        Explicit root index. When supplied, it is clamped to the valid node
        range.

    Returns
    -------
    int
        Root node used for radial rings.
    """
    if num_nodes <= 0:
        return 0
    if root is not None:
        return min(max(int(root), 0), num_nodes - 1)

    steps_to_leaf = _graphviz_twopi_leaf_steps(edge_index, num_nodes)
    best_node = 0
    best_steps = -1
    for node, steps in enumerate(steps_to_leaf):
        if steps > best_steps:
            best_steps = steps
            best_node = node
    return best_node


def twopi_ring_levels(
    edge_index: torch.Tensor,
    num_nodes: int,
    root: Optional[int] = None,
) -> List[int]:
    """Assign nodes to twopi BFS rings.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    root : int, optional
        Explicit radial root.

    Returns
    -------
    list[int]
        Ring index per node; disconnected nodes continue after the reachable
        rings in deterministic component order.
    """
    if num_nodes <= 0:
        return []
    adjacency = build_undirected_adjacency(edge_index, num_nodes)
    root_index = choose_twopi_root(edge_index, num_nodes, root)
    distances = bfs_distances(adjacency, root_index)
    levels = [int(distance) for distance in distances.tolist()]
    next_level = max((level for level in levels if level >= 0), default=0) + 1
    for node, level in enumerate(levels):
        if level >= 0:
            continue
        component_distances = bfs_distances(adjacency, node)
        component_nodes = [
            idx for idx, distance in enumerate(component_distances.tolist()) if int(distance) >= 0
        ]
        for idx in component_nodes:
            if levels[idx] < 0:
                levels[idx] = next_level + int(component_distances[idx])
        next_level = max(levels) + 1
    return levels


def _bfs_tree_children(edge_index: torch.Tensor, num_nodes: int, root: int) -> List[List[int]]:
    """Build Graphviz-order BFS tree children for twopi wedge assignment.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    root : int
        Root node.

    Returns
    -------
    list[list[int]]
        Child nodes by parent.
    """
    ordered_neighbors: List[List[int]] = [[] for _ in range(num_nodes)]
    seen_neighbors: List[Set[int]] = [set() for _ in range(num_nodes)]
    for source, target in _edge_pairs(edge_index):
        if target not in seen_neighbors[source]:
            ordered_neighbors[source].append(target)
            seen_neighbors[source].add(target)
        if source not in seen_neighbors[target]:
            ordered_neighbors[target].append(source)
            seen_neighbors[target].add(source)

    children: List[List[int]] = [[] for _ in range(num_nodes)]
    visited = [False] * num_nodes
    visited[root] = True
    queue: deque[int] = deque([root])
    while queue:
        node = queue.popleft()
        for neighbor in ordered_neighbors[node]:
            if visited[neighbor]:
                continue
            visited[neighbor] = True
            children[node].append(neighbor)
            queue.append(neighbor)
    for node in range(num_nodes):
        if visited[node]:
            continue
        visited[node] = True
        queue.append(node)
        while queue:
            parent = queue.popleft()
            for neighbor in ordered_neighbors[parent]:
                if visited[neighbor]:
                    continue
                visited[neighbor] = True
                children[parent].append(neighbor)
                queue.append(neighbor)
    return children


def _subtree_leaf_counts(children: Sequence[Sequence[int]], root: int) -> List[int]:
    """Count terminal leaves below each radial-tree node.

    Parameters
    ----------
    children : sequence[sequence[int]]
        Tree child lists.
    root : int
        Root node.

    Returns
    -------
    list[int]
        Leaf-count weight per node.
    """
    counts = [0] * len(children)

    def visit(node: int) -> int:
        """Recursively count subtree leaves.

        Parameters
        ----------
        node : int
            Node to visit.

        Returns
        -------
        int
            Leaf count for ``node``.
        """
        if not children[node]:
            counts[node] = 1
            return 1
        counts[node] = sum(visit(child) for child in children[node])
        return counts[node]

    visit(root)
    for node in range(len(children)):
        if counts[node] == 0:
            visit(node)
    return counts


def twopi_positions(
    edge_index: torch.Tensor,
    num_nodes: int,
    ranksep: float = _DEFAULT_RANKSEP_POINTS,
    root: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Compute deterministic radial positions.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    ranksep : float, default=72.0
        Radial spacing between BFS rings in points.
    root : int, optional
        Explicit root node index.

    Returns
    -------
    tuple[torch.Tensor, dict[str, object]]
        Position tensor with shape ``[N, 2]`` and stage metadata.
    """
    device = layout_device(edge_index)
    if num_nodes <= 0:
        return torch.zeros((0, 2), dtype=torch.float64, device=device), {"root": 0, "levels": []}
    root_index = choose_twopi_root(edge_index, num_nodes, root)
    levels = twopi_ring_levels(edge_index, num_nodes, root_index)
    children = _bfs_tree_children(edge_index, num_nodes, root_index)
    leaf_counts = _subtree_leaf_counts(children, root_index)
    angles = [0.0] * num_nodes

    def assign(node: int, start_angle: float, width: float) -> None:
        """Assign angular wedges recursively.

        Parameters
        ----------
        node : int
            Node receiving the wedge.
        start_angle : float
            Start angle in radians.
        width : float
            Angular width in radians.

        Returns
        -------
        None
            The function mutates ``angles``.
        """
        angles[node] = start_angle + width / 2.0
        if not children[node]:
            return
        cursor = start_angle
        total = float(sum(leaf_counts[child] for child in children[node]))
        for child in children[node]:
            child_width = width * float(leaf_counts[child]) / total if total > 0.0 else 0.0
            assign(child, cursor, child_width)
            cursor += child_width

    assign(root_index, 0.0, _TWO_PI)
    for node in range(num_nodes):
        if node == root_index or angles[node] != 0.0:
            continue
        angles[node] = _TWO_PI * node / max(num_nodes, 1)

    positions = torch.zeros((num_nodes, 2), dtype=torch.float64, device=device)
    for node, level in enumerate(levels):
        radius = float(level) * float(ranksep)
        positions[node, 0] = radius * math.cos(angles[node])
        positions[node, 1] = radius * math.sin(angles[node])
    return positions, {"root": root_index, "levels": levels, "leaf_counts": leaf_counts}


def biconnected_components(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
    """Find undirected biconnected components with Tarjan's algorithm.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    list[list[int]]
        Biconnected component node memberships in discovery order.
    """
    adjacency: List[List[int]] = [[] for _ in range(num_nodes)]
    seen: List[Set[int]] = [set() for _ in range(num_nodes)]
    for source, target in _edge_pairs(edge_index):
        if source == target:
            continue
        if target not in seen[source]:
            adjacency[source].append(target)
            seen[source].add(target)
        if source not in seen[target]:
            adjacency[target].append(source)
            seen[target].add(source)

    discovery = [-1] * num_nodes
    low = [0] * num_nodes
    edge_stack: List[Tuple[int, int]] = []
    components: List[List[int]] = []
    time = 0

    def visit(node: int, parent: int) -> None:
        """Run one Tarjan DFS visit.

        Parameters
        ----------
        node : int
            Node being visited.
        parent : int
            DFS parent, or ``-1`` for a root.

        Returns
        -------
        None
            The function mutates discovery state and ``components``.
        """
        nonlocal time
        discovery[node] = time
        low[node] = time
        time += 1
        for neighbor in adjacency[node]:
            edge = (min(node, neighbor), max(node, neighbor))
            if discovery[neighbor] < 0:
                edge_stack.append(edge)
                visit(neighbor, node)
                low[node] = min(low[node], low[neighbor])
                if low[neighbor] >= discovery[node]:
                    members: Set[int] = set()
                    while edge_stack:
                        stacked = edge_stack.pop()
                        members.update(stacked)
                        if stacked == edge:
                            break
                    components.append(sorted(members))
            elif neighbor != parent and discovery[neighbor] < discovery[node]:
                edge_stack.append(edge)
                low[node] = min(low[node], discovery[neighbor])

    for node in range(num_nodes):
        if discovery[node] >= 0:
            continue
        if not adjacency[node]:
            components.append([node])
            discovery[node] = time
            low[node] = time
            time += 1
            continue
        visit(node, -1)
        if edge_stack:
            members = set()
            while edge_stack:
                members.update(edge_stack.pop())
            components.append(sorted(members))
    return components


def circo_positions(
    edge_index: torch.Tensor,
    num_nodes: int,
    nodesep: float = _DEFAULT_NODESEP_POINTS,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Compute deterministic circular positions from biconnected blocks.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    nodesep : float, default=18.0
        Approximate chord spacing between adjacent nodes.

    Returns
    -------
    tuple[torch.Tensor, dict[str, object]]
        Position tensor with shape ``[N, 2]`` and block metadata.
    """
    device = layout_device(edge_index)
    if num_nodes <= 0:
        return torch.zeros((0, 2), dtype=torch.float64, device=device), {"blocks": []}
    blocks = biconnected_components(edge_index, num_nodes)
    node_blocks: DefaultDict[int, List[int]] = defaultdict(list)
    for block_index, block in enumerate(blocks):
        for node in block:
            node_blocks[node].append(block_index)

    block_gap = max(float(nodesep) * 4.0, 72.0)
    positions = torch.zeros((num_nodes, 2), dtype=torch.float64, device=device)
    placed: Set[int] = set()
    for block_index, block in enumerate(blocks):
        ordered = sorted(block)
        count = len(ordered)
        if count == 1:
            radius = 0.0
        else:
            radius = max(float(nodesep) * count / _TWO_PI, float(nodesep))
        center_x = float(block_index) * block_gap
        center_y = 0.0
        for offset, node in enumerate(ordered):
            if node in placed and len(node_blocks[node]) > 1:
                continue
            angle = _TWO_PI * offset / max(count, 1)
            positions[node, 0] = center_x + radius * math.cos(angle)
            positions[node, 1] = center_y + radius * math.sin(angle)
            placed.add(node)

    for node in range(num_nodes):
        if node in placed:
            continue
        positions[node, 0] = float(len(blocks) + node) * block_gap
        placed.add(node)
    return positions, {"blocks": blocks}


@register_op
class TwopiAssignRadialCoordinates(Op):
    """Assign radial Graphviz twopi-style coordinates."""

    name = "twopi_assign_radial_coordinates"
    category = OpCategory.COORDINATE
    reads = ("edge_index",)
    writes = ("pos", "extras")

    def __init__(
        self,
        ranksep: float = _DEFAULT_RANKSEP_POINTS,
        root: Optional[int] = None,
    ) -> None:
        """Store radial layout settings.

        Parameters
        ----------
        ranksep : float, default=72.0
            Distance between BFS rings in points.
        root : int, optional
            Explicit root index.

        Returns
        -------
        None
            Settings are stored on the op instance.
        """
        self.ranksep = float(ranksep)
        self.root = root

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute and store radial coordinates.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph topology.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context; accepted for the common op interface.

        Returns
        -------
        SolveState
            State with ``pos`` and twopi metadata populated.
        """
        del ctx
        state.pos, metadata = twopi_positions(
            problem.edge_index,
            problem.num_nodes,
            ranksep=self.ranksep,
            root=self.root,
        )
        state.extras["twopi"] = metadata
        return state


@register_op
class CircoAssignCircularCoordinates(Op):
    """Assign block-aware Graphviz circo-style coordinates."""

    name = "circo_assign_circular_coordinates"
    category = OpCategory.COORDINATE
    reads = ("edge_index",)
    writes = ("pos", "extras")

    def __init__(self, nodesep: float = _DEFAULT_NODESEP_POINTS) -> None:
        """Store circular layout settings.

        Parameters
        ----------
        nodesep : float, default=18.0
            Approximate separation between adjacent block nodes.

        Returns
        -------
        None
            Settings are stored on the op instance.
        """
        self.nodesep = float(nodesep)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute and store circular block coordinates.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph topology.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context; accepted for the common op interface.

        Returns
        -------
        SolveState
            State with ``pos`` and circo metadata populated.
        """
        del ctx
        state.pos, metadata = circo_positions(
            problem.edge_index,
            problem.num_nodes,
            nodesep=self.nodesep,
        )
        state.extras["circo"] = metadata
        return state


__all__ = [
    "CircoAssignCircularCoordinates",
    "TwopiAssignRadialCoordinates",
    "biconnected_components",
    "choose_twopi_root",
    "circo_positions",
    "twopi_positions",
    "twopi_ring_levels",
]
