"""Graphviz-inspired deterministic radial and circular layout operations."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch

from dagua.layout.ops.base import Op
from dagua.layout.ops.graph_utils import bfs_distances, build_undirected_adjacency, layout_device
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_DEFAULT_RANKSEP_POINTS = 72.0
_DEFAULT_NODESEP_POINTS = 18.0
_TWO_PI = 2.0 * math.pi


@dataclass
class _CircoBlock:
    """Owned Graphviz circo block in the block-cutpoint tree.

    Parameters
    ----------
    nodes : list[int]
        Nodes owned by this block. Unlike standard biconnected components,
        Graphviz assigns each articulation point to one block and attaches
        child blocks through ``child``/``parent`` links.
    child : int or None, optional
        Node in this block nearest to the parent block.
    children : list[_CircoBlock], optional
        Child blocks attached below this block.
    """

    nodes: List[int]
    child: Optional[int] = None
    children: List["_CircoBlock"] = field(default_factory=list)
    ordered: List[int] = field(default_factory=list)
    radius: float = 0.0
    rad0: float = 0.0
    parent_pos: float = -1.0
    coalesced: bool = False


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


def _simple_adjacency(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
    """Build strict undirected adjacency in Graphviz edge traversal order.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    list[list[int]]
        Neighbor lists with duplicate edges and self-loops removed.
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
    return adjacency


def _circo_connected_components(adjacency: Sequence[Sequence[int]]) -> List[List[int]]:
    """Return connected components in Graphviz node traversal order.

    Parameters
    ----------
    adjacency : sequence[sequence[int]]
        Strict undirected adjacency.

    Returns
    -------
    list[list[int]]
        Components as node lists.
    """
    visited = [False] * len(adjacency)
    components: List[List[int]] = []
    for start in range(len(adjacency)):
        if visited[start]:
            continue
        visited[start] = True
        queue: deque[int] = deque([start])
        component: List[int] = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in adjacency[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        components.append(component)
    return components


def _graphviz_owned_block_tree(
    adjacency: Sequence[Sequence[int]],
    component: Sequence[int],
) -> Optional[_CircoBlock]:
    """Construct Graphviz's owned block-cutpoint tree for one component.

    Parameters
    ----------
    adjacency : sequence[sequence[int]]
        Strict undirected adjacency for the whole graph.
    component : sequence[int]
        Nodes in the connected component.

    Returns
    -------
    _CircoBlock or None
        Root owned block. ``None`` is returned for an empty component.
    """
    if not component:
        return None
    component_set = set(component)
    value = [0] * len(adjacency)
    low = [0] * len(adjacency)
    parent: List[Optional[int]] = [None] * len(adjacency)
    owner: List[Optional[_CircoBlock]] = [None] * len(adjacency)
    stack: List[Tuple[int, int]] = []
    blocks: List[_CircoBlock] = []
    order_count = 1
    root_node = component[0]

    def make_block() -> _CircoBlock:
        """Create and remember one owned block.

        Returns
        -------
        _CircoBlock
            Empty block appended later by the DFS.
        """
        return _CircoBlock(nodes=[])

    def add_node(block: _CircoBlock, node: int) -> None:
        """Assign a node to a block once.

        Parameters
        ----------
        block : _CircoBlock
            Block receiving the node.
        node : int
            Node to assign.

        Returns
        -------
        None
            The function mutates ownership state.
        """
        if owner[node] is None:
            owner[node] = block
            block.nodes.append(node)

    def visit(node: int, is_root: bool) -> None:
        """Run Graphviz's block-tree DFS.

        Parameters
        ----------
        node : int
            Current DFS node.
        is_root : bool
            Whether ``node`` is the chosen component root.

        Returns
        -------
        None
            Discovery values, owned blocks, and parent links are mutated.
        """
        nonlocal order_count
        value[node] = order_count
        low[node] = order_count
        order_count += 1
        for neighbor in adjacency[node]:
            if neighbor not in component_set:
                continue
            edge = (node, neighbor)
            if value[neighbor] == 0:
                parent[neighbor] = node
                stack.append(edge)
                visit(neighbor, False)
                low[node] = min(low[node], low[neighbor])
                if low[neighbor] >= value[node]:
                    block: Optional[_CircoBlock] = None
                    while stack:
                        source, target = stack.pop()
                        endpoint = target if parent[target] == source else source
                        if owner[endpoint] is None:
                            if block is None:
                                block = make_block()
                            add_node(block, endpoint)
                        if (source, target) == edge:
                            break
                    if block is not None:
                        if owner[node] is None and len(block.nodes) > 1:
                            add_node(block, node)
                        if is_root and owner[node] is block:
                            blocks.insert(0, block)
                        else:
                            blocks.append(block)
            elif parent[node] != neighbor:
                low[node] = min(low[node], value[neighbor])

    visit(root_node, True)
    if owner[root_node] is None:
        root_block = make_block()
        add_node(root_block, root_node)
        blocks.insert(0, root_block)
    if not blocks:
        return None
    root = blocks[0]
    for block in blocks[1:]:
        child = min(block.nodes, key=lambda node: value[node])
        parent_node = parent[child]
        if parent_node is None or owner[parent_node] is None:
            continue
        block.child = child
        owner[parent_node].children.append(block)
    return root


def _block_induced_adjacency(
    adjacency: Sequence[Sequence[int]],
    nodes: Sequence[int],
) -> Dict[int, List[int]]:
    """Return the induced adjacency for one owned block.

    Parameters
    ----------
    adjacency : sequence[sequence[int]]
        Full strict undirected adjacency.
    nodes : sequence[int]
        Nodes owned by the block.

    Returns
    -------
    dict[int, list[int]]
        Block-local neighbor lists.
    """
    node_set = set(nodes)
    return {
        node: [neighbor for neighbor in adjacency[node] if neighbor in node_set] for node in nodes
    }


def _longest_tree_path(tree: Dict[int, List[int]], nodes: Sequence[int]) -> List[int]:
    """Find Graphviz's DFS-tree diameter path for block ordering.

    Parameters
    ----------
    tree : dict[int, list[int]]
        Spanning tree adjacency.
    nodes : sequence[int]
        Block nodes.

    Returns
    -------
    list[int]
        Ordered longest path nodes.
    """
    if len(nodes) <= 1:
        return list(nodes)
    parent: Dict[int, Optional[int]] = {nodes[0]: None}
    stack = [nodes[0]]
    for node in stack:
        for neighbor in tree[node]:
            if neighbor not in parent:
                parent[neighbor] = node
                stack.append(neighbor)
    leaves = [node for node in nodes if len(tree[node]) <= 1]
    if not leaves:
        return list(nodes)
    best_pair = (leaves[0], leaves[0])
    best_distance = -1
    for source in leaves:
        queue: deque[int] = deque([source])
        dist = {source: 0}
        pred: Dict[int, Optional[int]] = {source: None}
        while queue:
            node = queue.popleft()
            for neighbor in tree[node]:
                if neighbor not in dist:
                    dist[neighbor] = dist[node] + 1
                    pred[neighbor] = node
                    queue.append(neighbor)
        for target in leaves:
            if dist.get(target, -1) > best_distance:
                best_distance = dist[target]
                best_pair = (source, target)
    source, target = best_pair
    queue = deque([source])
    pred = {source: None}
    while queue:
        node = queue.popleft()
        if node == target:
            break
        for neighbor in tree[node]:
            if neighbor not in pred:
                pred[neighbor] = node
                queue.append(neighbor)
    path: List[int] = []
    node: Optional[int] = target
    while node is not None:
        path.append(node)
        node = pred[node]
    path.reverse()
    return path


def _circo_block_order(
    adjacency: Sequence[Sequence[int]],
    block: _CircoBlock,
) -> List[int]:
    """Order one block using Graphviz's path-plus-residual heuristic.

    Parameters
    ----------
    adjacency : sequence[sequence[int]]
        Strict undirected adjacency.
    block : _CircoBlock
        Block to order.

    Returns
    -------
    list[int]
        Circular node order for the block.
    """
    nodes = list(block.nodes)
    if len(nodes) <= 2:
        return nodes
    induced = _block_induced_adjacency(adjacency, nodes)
    visited: Set[int] = set()
    tree: Dict[int, List[int]] = {node: [] for node in nodes}

    def dfs(node: int) -> None:
        """Build the DFS spanning tree in neighbor order.

        Parameters
        ----------
        node : int
            Current node.

        Returns
        -------
        None
            The tree adjacency is mutated.
        """
        visited.add(node)
        for neighbor in induced[node]:
            if neighbor not in visited:
                tree[node].append(neighbor)
                tree[neighbor].append(node)
                dfs(neighbor)

    for node in nodes:
        if node not in visited:
            dfs(node)
    ordered = _longest_tree_path(tree, nodes)
    ordered_set = set(ordered)
    for node in nodes:
        if node in ordered_set:
            continue
        neighbors = [neighbor for neighbor in induced[node] if neighbor in ordered_set]
        placed = False
        if len(neighbors) >= 2:
            for index, candidate in enumerate(ordered):
                nxt = ordered[(index + 1) % len(ordered)]
                if candidate in neighbors and nxt in neighbors:
                    ordered.insert(index + 1, node)
                    placed = True
                    break
        if not placed and neighbors:
            anchor = next(candidate for candidate in ordered if candidate in neighbors)
            ordered.insert(ordered.index(anchor) + 1, node)
            placed = True
        if not placed:
            ordered.append(node)
        ordered_set.add(node)
    for index, node in enumerate(ordered):
        if any(child.child == node for child in block.children):
            return ordered[index:] + ordered[:index]
    return ordered


def _is_connected_path(edge_index: torch.Tensor, num_nodes: int) -> bool:
    """Return whether the undirected simple topology is one connected path.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    bool
        ``True`` when all nodes form one non-branching path.
    """
    if num_nodes <= 1:
        return num_nodes == 1
    adjacency: List[Set[int]] = [set() for _ in range(num_nodes)]
    for source, target in _edge_pairs(edge_index):
        if source == target:
            continue
        adjacency[source].add(target)
        adjacency[target].add(source)
    edge_count = sum(len(neighbors) for neighbors in adjacency) // 2
    if edge_count != num_nodes - 1:
        return False
    degree_counts = sorted(len(neighbors) for neighbors in adjacency)
    if degree_counts[:2] != [1, 1] or any(degree != 2 for degree in degree_counts[2:]):
        return False
    distances = bfs_distances(
        [[(neighbor, 1.0) for neighbor in sorted(neighbors)] for neighbors in adjacency],
        0,
    )
    return bool((distances >= 0).all())


def _connected_path_positions(
    edge_index: torch.Tensor,
    num_nodes: int,
    nodesep: float,
    device: torch.device,
) -> torch.Tensor:
    """Place one connected path as Graphviz circo's coalesced edge-block chain.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    nodesep : float
        Horizontal spacing scale in points.
    device : torch.device
        Output tensor device.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    adjacency: List[List[int]] = [[] for _ in range(num_nodes)]
    for source, target in _edge_pairs(edge_index):
        if source == target:
            continue
        if target not in adjacency[source]:
            adjacency[source].append(target)
        if source not in adjacency[target]:
            adjacency[target].append(source)
    start = next(node for node, neighbors in enumerate(adjacency) if len(neighbors) <= 1)
    ordered: List[int] = []
    previous = -1
    current = start
    while current >= 0:
        ordered.append(current)
        next_nodes = [neighbor for neighbor in adjacency[current] if neighbor != previous]
        if not next_nodes:
            break
        previous, current = current, next_nodes[0]

    positions = torch.zeros((num_nodes, 2), dtype=torch.float64, device=device)
    spacing = max(float(nodesep), 1.0)
    for offset, node in enumerate(ordered):
        positions[node, 0] = float(offset) * spacing
    return positions


def _cycle_block_order(edge_index: torch.Tensor, block: Sequence[int]) -> Optional[List[int]]:
    """Return Graphviz-style traversal order for one simple cycle block.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    block : sequence[int]
        Candidate biconnected block node ids.

    Returns
    -------
    list[int] or None
        DFS cycle order when the induced block is a simple cycle, otherwise
        ``None``.
    """
    block_set = set(block)
    adjacency: Dict[int, List[int]] = {node: [] for node in block}
    for source, target in _edge_pairs(edge_index):
        if source == target or source not in block_set or target not in block_set:
            continue
        if target not in adjacency[source]:
            adjacency[source].append(target)
        if source not in adjacency[target]:
            adjacency[target].append(source)
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        return None

    start = min(block)
    ordered: List[int] = []
    visited: Set[int] = set()

    def visit(node: int) -> None:
        """Visit cycle nodes in input-neighbor order.

        Parameters
        ----------
        node : int
            Current cycle node.

        Returns
        -------
        None
            The function mutates ``ordered`` and ``visited``.
        """
        visited.add(node)
        ordered.append(node)
        for neighbor in adjacency[node]:
            if neighbor not in visited:
                visit(neighbor)

    visit(start)
    if len(ordered) != len(block):
        return None
    return ordered


def _rotate_translate(
    points: Dict[int, Tuple[float, float]],
    nodes: Sequence[int],
    x_offset: float,
    y_offset: float,
    rotation: float,
) -> None:
    """Apply rotation followed by translation to selected nodes.

    Parameters
    ----------
    points : dict[int, tuple[float, float]]
        Mutable point map.
    nodes : sequence[int]
        Nodes to transform.
    x_offset : float
        Translation in x.
    y_offset : float
        Translation in y.
    rotation : float
        Rotation angle in radians.

    Returns
    -------
    None
        The point map is updated in place.
    """
    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)
    for node in nodes:
        x_value, y_value = points[node]
        rotated_x = x_value * cos_r - y_value * sin_r
        rotated_y = x_value * sin_r + y_value * cos_r
        points[node] = (rotated_x + x_offset, rotated_y + y_offset)


def _subtree_nodes(block: _CircoBlock) -> List[int]:
    """Return nodes owned by a block subtree.

    Parameters
    ----------
    block : _CircoBlock
        Root block.

    Returns
    -------
    list[int]
        Nodes owned by ``block`` and all descendants.
    """
    nodes = list(block.nodes)
    for child in block.children:
        nodes.extend(_subtree_nodes(child))
    return nodes


def _circo_child_rotation(
    block: _CircoBlock,
    points: Dict[int, Tuple[float, float]],
    x_offset: float,
    y_offset: float,
    theta: float,
) -> float:
    """Compute the Graphviz-style rotation for a child block.

    Parameters
    ----------
    block : _CircoBlock
        Child block being attached.
    points : dict[int, tuple[float, float]]
        Current local point map.
    x_offset : float
        Child center x relative to parent.
    y_offset : float
        Child center y relative to parent.
    theta : float
        Incident angle from parent to child.

    Returns
    -------
    float
        Rotation angle in radians.
    """
    if block.parent_pos >= 0.0:
        rotation = theta + math.pi - block.parent_pos
        return rotation + _TWO_PI if rotation < 0.0 else rotation
    if len(block.ordered) == 2:
        return theta - math.pi / 2.0
    if block.child is None:
        return 0.0
    neighbor = block.child
    neighbor_x, neighbor_y = points[neighbor]
    best_node = neighbor
    best_dist = (neighbor_x + x_offset) ** 2 + (neighbor_y + y_offset) ** 2
    for node in block.nodes:
        if node == neighbor:
            continue
        node_x, node_y = points[node]
        dist = (node_x + x_offset) ** 2 + (node_y + y_offset) ** 2
        if dist < best_dist:
            best_node = node
            best_dist = dist
    if best_node == neighbor:
        return 0.0
    phi = math.atan2(neighbor_y, neighbor_x)
    rotation = theta + math.pi - phi
    return rotation - _TWO_PI if rotation > _TWO_PI else rotation


def _layout_circo_block_tree(
    block: _CircoBlock,
    adjacency: Sequence[Sequence[int]],
    nodesep: float,
    points: Dict[int, Tuple[float, float]],
) -> None:
    """Lay out one block tree using Graphviz circpos formulas.

    Parameters
    ----------
    block : _CircoBlock
        Block subtree root.
    adjacency : sequence[sequence[int]]
        Strict undirected adjacency.
    nodesep : float
        Minimum separation scale.
    points : dict[int, tuple[float, float]]
        Mutable local point map.

    Returns
    -------
    None
        ``points`` and block radius metadata are updated in place.
    """
    for child in block.children:
        _layout_circo_block_tree(child, adjacency, nodesep, points)

    block.ordered = _circo_block_order(adjacency, block)
    count = len(block.ordered)
    if count == 0:
        block.radius = 0.0
        block.rad0 = 0.0
        return
    if count == 1:
        radius = 0.0
    else:
        radius = max(float(nodesep) * count / _TWO_PI, float(nodesep))
    for offset, node in enumerate(block.ordered):
        angle = _TWO_PI * offset / max(count, 1)
        points[node] = (radius * math.cos(angle), radius * math.sin(angle))
    block.radius = radius if count > 1 else float(nodesep) / 2.0
    block.rad0 = block.radius
    block.parent_pos = -1.0

    if not block.children:
        return

    child_count_by_parent: Dict[int, int] = defaultdict(int)
    for child in block.children:
        if child.child is not None:
            parent_candidates = [
                neighbor for neighbor in adjacency[child.child] if neighbor in set(block.nodes)
            ]
            if parent_candidates:
                child_count_by_parent[parent_candidates[0]] += 1

    subtree_radius = block.radius
    for parent_node in block.ordered:
        attached = [
            child
            for child in block.children
            if child.child is not None
            and any(neighbor == parent_node for neighbor in adjacency[child.child])
        ]
        if not attached:
            continue
        parent_index = block.ordered.index(parent_node)
        parent_theta = parent_index * (_TWO_PI / max(count, 1))
        max_child_radius = max(child.radius for child in attached)
        diameter = sum(2.0 * child.radius + float(nodesep) for child in attached)
        child_radius = block.radius + float(nodesep) + max_child_radius
        if count == 1:
            child_radius = max(child_radius, diameter / _TWO_PI)
            child_angle = 0.0
        elif len(attached) == 1:
            child_angle = parent_theta
        else:
            child_angle = parent_theta - diameter / (2.0 * child_radius)
        min_angle = float(nodesep) / max(child_radius, 1.0)
        for child in attached:
            incident = child.radius / max(child_radius, 1.0)
            if count == 1:
                if child_angle != 0.0:
                    child_angle = math.pi if len(attached) == 2 else child_angle + incident
            elif len(attached) > 1:
                child_angle += incident + min_angle / 2.0
            delta_x = child_radius * math.cos(child_angle)
            delta_y = child_radius * math.sin(child_angle)
            rotation = _circo_child_rotation(child, points, delta_x, delta_y, child_angle)
            _rotate_translate(points, _subtree_nodes(child), delta_x, delta_y, rotation)
            if count == 1:
                child_angle += incident + min_angle
            elif len(attached) > 1:
                child_angle += incident + min_angle / 2.0
            subtree_radius = max(subtree_radius, child_radius + child.radius)
    if len(block.children) == 1:
        child = block.children[0]
        shift = -(child.radius + float(nodesep) / 2.0)
        _rotate_translate(points, _subtree_nodes(block), shift, 0.0, 0.0)
        block.radius += float(nodesep) / 2.0 + child.radius
        block.coalesced = True
    else:
        block.radius = subtree_radius


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
    if _is_connected_path(edge_index, num_nodes):
        blocks = biconnected_components(edge_index, num_nodes)
        return _connected_path_positions(edge_index, num_nodes, nodesep, device), {"blocks": blocks}
    if (cycle_order := _cycle_block_order(edge_index, list(range(num_nodes)))) is not None:
        positions = torch.zeros((num_nodes, 2), dtype=torch.float64, device=device)
        for offset, node in enumerate(cycle_order):
            angle = _TWO_PI * offset / max(num_nodes, 1)
            positions[node, 0] = float(nodesep) * math.cos(angle)
            positions[node, 1] = float(nodesep) * math.sin(angle)
        return positions, {"blocks": [cycle_order], "block_order": [cycle_order]}

    blocks = biconnected_components(edge_index, num_nodes)
    adjacency = _simple_adjacency(edge_index, num_nodes)
    components = _circo_connected_components(adjacency)
    block_gap = max(float(nodesep) * 8.0, 144.0)
    positions = torch.zeros((num_nodes, 2), dtype=torch.float64, device=device)
    block_orders: List[List[int]] = []
    component_offset = 0.0
    for component in components:
        root = _graphviz_owned_block_tree(adjacency, component)
        if root is None:
            continue
        points: Dict[int, Tuple[float, float]] = {}
        _layout_circo_block_tree(root, adjacency, nodesep, points)
        xs = [point[0] for point in points.values()]
        min_x = min(xs, default=0.0)
        max_x = max(xs, default=0.0)
        for node in component:
            x_value, y_value = points.get(node, (0.0, 0.0))
            positions[node, 0] = x_value - min_x + component_offset
            positions[node, 1] = y_value
        component_offset += (max_x - min_x) + block_gap

        stack = [root]
        while stack:
            block = stack.pop()
            block_orders.append(list(block.ordered))
            stack.extend(reversed(block.children))
    return positions, {"blocks": blocks, "block_order": block_orders}


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
