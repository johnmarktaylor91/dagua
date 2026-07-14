"""OGDF-style BalloonLayout pipeline without runtime delegation."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple

import torch

from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory

_OGDF_DEFAULT_NODE_SIZE = 20.0
_OGDF_MIN_NODE_RADIUS = 0.007
_OGDF_ESTIMATE_FACTOR = 1.2
_BALLOON_TREE_KEY = "balloon_tree"
_BALLOON_RADII_KEY = "balloon_radii"
_BALLOON_ANGLES_KEY = "balloon_angles"


@dataclass(frozen=True)
class _BalloonTree:
    """Rooted BFS tree metadata used by OGDF BalloonLayout.

    Parameters
    ----------
    root : int
        Selected center root after the OGDF leaf-pruning pass.
    parent : list[int | None]
        Parent for each node in the rooted tree.
    children : list[list[int]]
        Children for each node in OGDF list order.
    """

    root: int
    parent: list[Optional[int]]
    children: list[list[int]]


@dataclass(frozen=True)
class _BalloonRadii:
    """Per-node radius arrays from OGDF's SNS model.

    Parameters
    ----------
    inner : list[float]
        Inner placement radius for each node.
    outer : list[float]
        Outer subtree radius for each node.
    estimate : list[float]
        Sum of child outer radii used for proportional angle assignment.
    """

    inner: list[float]
    outer: list[float]
    estimate: list[float]


def _edge_adjacency(edge_index: torch.Tensor, num_nodes: int) -> list[list[int]]:
    """Return undirected adjacency in input-edge insertion order.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    list[list[int]]
        Neighbor lists for each node.
    """
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return adjacency
    edges = edge_index.to(device="cpu", dtype=torch.long)
    for source, target in zip(edges[0].tolist(), edges[1].tolist()):
        source_idx = int(source)
        target_idx = int(target)
        if source_idx == target_idx:
            continue
        adjacency[source_idx].append(target_idx)
        adjacency[target_idx].append(source_idx)
    return adjacency


def _node_radius(node_sizes: Optional[torch.Tensor], node: int) -> float:
    """Return OGDF's half-diagonal node radius.

    Parameters
    ----------
    node_sizes : torch.Tensor | None
        Optional node sizes with shape ``[N, 2]``.
    node : int
        Node index.

    Returns
    -------
    float
        Positive node radius used by BalloonLayout.
    """
    if node_sizes is None:
        width = _OGDF_DEFAULT_NODE_SIZE
        height = _OGDF_DEFAULT_NODE_SIZE
    else:
        sizes = node_sizes.to(device="cpu", dtype=torch.float64)
        width = float(sizes[node, 0].item())
        height = float(sizes[node, 1].item())
    return max(_OGDF_MIN_NODE_RADIUS, 0.5 * math.sqrt((width * width) + (height * height)))


def _bfs_tree(edge_index: torch.Tensor, num_nodes: int) -> _BalloonTree:
    """Build OGDF's first-node BFS tree and re-root it at the tree center.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    _BalloonTree
        Re-rooted BFS tree metadata.
    """
    if num_nodes == 0:
        return _BalloonTree(root=0, parent=[], children=[])

    adjacency = _edge_adjacency(edge_index, num_nodes)
    parent: list[Optional[int]] = [None] * num_nodes
    children: list[list[int]] = [[] for _ in range(num_nodes)]
    marked = [False] * num_nodes
    queue: deque[int] = deque([0])
    marked[0] = True
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if marked[neighbor]:
                continue
            parent[neighbor] = node
            children[node].append(neighbor)
            marked[neighbor] = True
            queue.append(neighbor)

    degree = [
        len(children[node]) + (1 if parent[node] is not None else 0) for node in range(num_nodes)
    ]
    leaves: deque[int] = deque()
    if num_nodes == 1:
        leaves.append(0)
    else:
        for node, node_degree in enumerate(degree):
            if node_degree == 1:
                leaves.append(node)

    root = 0
    while leaves:
        root = leaves.popleft()
        parent_node = parent[root]
        if parent_node is not None:
            degree[parent_node] -= 1
            if degree[parent_node] == 1:
                leaves.append(parent_node)
        for child in children[root]:
            degree[child] -= 1
            if degree[child] == 1:
                leaves.append(child)

    node = root
    previous: Optional[int] = None
    while node is not None:
        old_parent = parent[node]
        parent[node] = previous
        if previous is not None:
            children[previous].append(node)
        if old_parent is not None:
            children[old_parent].remove(node)
        previous = node
        node = old_parent

    return _BalloonTree(root=root, parent=parent, children=children)


def _compute_radii(
    tree: _BalloonTree,
    node_sizes: Optional[torch.Tensor],
    num_nodes: int,
) -> _BalloonRadii:
    """Compute OGDF BalloonLayout inner and outer radii.

    Parameters
    ----------
    tree : _BalloonTree
        Re-rooted BFS tree metadata.
    node_sizes : torch.Tensor | None
        Optional node sizes with shape ``[N, 2]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    _BalloonRadii
        Radius arrays matching OGDF's fixed child-order path.
    """
    sizes = [_node_radius(node_sizes, node) for node in range(num_nodes)]
    inner = [0.0] * num_nodes
    outer = [0.0] * num_nodes
    estimate = [0.0] * num_nodes
    max_child_radius = [0.0] * num_nodes
    remaining = [len(child_list) for child_list in tree.children]
    leaves: deque[int] = deque()
    level: deque[int] = deque()

    if num_nodes > 1:
        for node, child_count in enumerate(remaining):
            if child_count == 0:
                leaves.append(node)
                outer[node] = sizes[node]
        while leaves:
            node = leaves.popleft()
            parent = tree.parent[node]
            if parent is not None:
                child_outer = outer[node]
                estimate[parent] += child_outer
                max_child_radius[parent] = max(max_child_radius[parent], child_outer)
                remaining[parent] -= 1
                if remaining[parent] == 0:
                    level.append(parent)
            inner[node] = outer[node]

        while level:
            node = level.popleft()
            child_count = len(tree.children[node])
            if child_count == 1:
                inner[node] = max(2.0 * sizes[node], 1.1 * max_child_radius[node])
            else:
                circumference_estimate = (
                    max_child_radius[node] / max(child_count, 4)
                    + _OGDF_ESTIMATE_FACTOR * 2.0 * estimate[node]
                ) / (2.0 * math.pi)
                inner[node] = max(
                    max(circumference_estimate, 2.0 * sizes[node]),
                    1.1 * max_child_radius[node],
                )

            if child_count == 1:
                node_outer = max(inner[node], max_child_radius[node])
            else:
                node_outer = inner[node] + max_child_radius[node]

            parent = tree.parent[node]
            if parent is not None:
                estimate[parent] += node_outer
                max_child_radius[parent] = max(max_child_radius[parent], node_outer)
                remaining[parent] -= 1
                if remaining[parent] == 0:
                    level.append(parent)
            outer[node] = node_outer

    return _BalloonRadii(inner=inner, outer=outer, estimate=estimate)


def _compute_angle_extents(tree: _BalloonTree, radii: _BalloonRadii, num_nodes: int) -> list[float]:
    """Compute OGDF's per-child angular extents.

    Parameters
    ----------
    tree : _BalloonTree
        Re-rooted BFS tree metadata.
    radii : _BalloonRadii
        Radius arrays from the SNS model.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    list[float]
        Angular values that are later converted to absolute directions.
    """
    angles = [0.0] * num_nodes
    queue: deque[int] = deque([tree.root])
    while queue:
        parent = queue.popleft()
        child_count = len(tree.children[parent])
        if child_count == 0:
            continue
        if child_count == 1:
            child = tree.children[parent][0]
            angles[child] = math.pi
            queue.append(child)
            continue

        estimate = radii.estimate[parent]
        parent_estimate = estimate
        full_angle = 2.0 * math.pi
        for child in tree.children[parent]:
            if estimate > 0.0 and radii.outer[child] / estimate > 0.501:
                parent_estimate -= radii.outer[child]
                full_angle = math.pi
                break

        for child in tree.children[parent]:
            queue.append(child)
            if estimate > 0.0 and radii.outer[child] / estimate > 0.501:
                angles[child] = math.pi
            elif parent_estimate > 0.0:
                angles[child] = full_angle * radii.outer[child] / parent_estimate
    return angles


def _compute_positions(
    tree: _BalloonTree,
    radii: _BalloonRadii,
    angles: list[float],
    num_nodes: int,
) -> torch.Tensor:
    """Compute final OGDF BalloonLayout coordinates.

    Parameters
    ----------
    tree : _BalloonTree
        Re-rooted BFS tree metadata.
    radii : _BalloonRadii
        Radius arrays from the SNS model.
    angles : list[float]
        Mutable angular extents, converted in place to directions.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    torch.Tensor
        Coordinate tensor with shape ``[N, 2]`` and dtype ``torch.float64``.
    """
    pos = torch.zeros((num_nodes, 2), dtype=torch.float64)
    if num_nodes == 0:
        return pos

    queue: deque[int] = deque([tree.root])
    while queue:
        parent = queue.popleft()
        children = tree.children[parent]
        if not children:
            continue
        parent_x = float(pos[parent, 0].item())
        parent_y = float(pos[parent, 1].item())
        if len(children) == 1:
            child = children[0]
            queue.append(child)
            direction = angles[parent]
            angles[child] = direction
            pos[child, 0] = parent_x + math.cos(direction) * radii.inner[parent]
            pos[child, 1] = parent_y + math.sin(direction) * radii.inner[parent]
            continue

        angle_sum = math.fmod(angles[parent] - math.pi + angles[children[0]] / 2.0, 2.0 * math.pi)
        for idx, child in enumerate(children):
            queue.append(child)
            next_child = children[(idx + 1) % len(children)]
            pos[child, 0] = parent_x + math.cos(angle_sum) * radii.inner[parent]
            pos[child, 1] = parent_y + math.sin(angle_sum) * radii.inner[parent]
            extent = angles[child]
            angles[child] = angle_sum
            angle_sum = math.fmod(angle_sum + (extent + angles[next_child]) / 2.0, 2.0 * math.pi)
    return pos


class BalloonTreeBuild(Op):
    """Build OGDF's rooted BFS tree for BalloonLayout."""

    name: ClassVar[str] = "balloon_tree_build"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(self, problem: LayoutProblem, state: SolveState, ctx: RuntimeContext) -> SolveState:
        """Store re-rooted BFS tree metadata.

        Parameters
        ----------
        problem : LayoutProblem
            Layout inputs containing graph topology.
        state : SolveState
            Mutable pipeline state.
        ctx : RuntimeContext
            Runtime context, unused.

        Returns
        -------
        SolveState
            State containing ``balloon_tree`` metadata.
        """
        del ctx
        state.extras[_BALLOON_TREE_KEY] = _bfs_tree(problem.edge_index, problem.num_nodes)
        return state


class BalloonRadiiCompute(Op):
    """Compute OGDF BalloonLayout subtree radii."""

    name: ClassVar[str] = "balloon_radii_compute"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(self, problem: LayoutProblem, state: SolveState, ctx: RuntimeContext) -> SolveState:
        """Store inner and outer radius arrays.

        Parameters
        ----------
        problem : LayoutProblem
            Layout inputs containing node sizes.
        state : SolveState
            Mutable pipeline state with a ``balloon_tree`` entry.
        ctx : RuntimeContext
            Runtime context, unused.

        Returns
        -------
        SolveState
            State containing ``balloon_radii`` metadata.
        """
        del ctx
        tree = state.extras.get(_BALLOON_TREE_KEY)
        if not isinstance(tree, _BalloonTree):
            raise RuntimeError("Balloon tree metadata is missing.")
        state.extras[_BALLOON_RADII_KEY] = _compute_radii(
            tree,
            problem.node_sizes,
            problem.num_nodes,
        )
        return state


class BalloonAngleCompute(Op):
    """Compute OGDF BalloonLayout angle extents."""

    name: ClassVar[str] = "balloon_angle_compute"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(self, problem: LayoutProblem, state: SolveState, ctx: RuntimeContext) -> SolveState:
        """Store angle extents for each node.

        Parameters
        ----------
        problem : LayoutProblem
            Layout inputs containing node count.
        state : SolveState
            Mutable pipeline state with tree and radius metadata.
        ctx : RuntimeContext
            Runtime context, unused.

        Returns
        -------
        SolveState
            State containing ``balloon_angles`` metadata.
        """
        del ctx
        tree = state.extras.get(_BALLOON_TREE_KEY)
        radii = state.extras.get(_BALLOON_RADII_KEY)
        if not isinstance(tree, _BalloonTree) or not isinstance(radii, _BalloonRadii):
            raise RuntimeError("Balloon tree or radius metadata is missing.")
        state.extras[_BALLOON_ANGLES_KEY] = _compute_angle_extents(tree, radii, problem.num_nodes)
        return state


class BalloonCoordinatePlacement(Op):
    """Place nodes with OGDF BalloonLayout coordinates."""

    name: ClassVar[str] = "balloon_coordinate_placement"
    category: ClassVar[OpCategory] = OpCategory.INIT
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(self, problem: LayoutProblem, state: SolveState, ctx: RuntimeContext) -> SolveState:
        """Populate ``state.pos`` with final coordinates.

        Parameters
        ----------
        problem : LayoutProblem
            Layout inputs containing output device.
        state : SolveState
            Mutable pipeline state with tree, radius, and angle metadata.
        ctx : RuntimeContext
            Runtime context, unused.

        Returns
        -------
        SolveState
            State with ``pos`` set to shape ``[N, 2]``.
        """
        del ctx
        tree = state.extras.get(_BALLOON_TREE_KEY)
        radii = state.extras.get(_BALLOON_RADII_KEY)
        angles = state.extras.get(_BALLOON_ANGLES_KEY)
        if not isinstance(tree, _BalloonTree) or not isinstance(radii, _BalloonRadii):
            raise RuntimeError("Balloon metadata is missing.")
        if not isinstance(angles, list):
            raise RuntimeError("Balloon angle metadata is missing.")
        state.pos = _compute_positions(tree, radii, list(angles), problem.num_nodes).to(
            device=problem.edge_index.device
        )
        return state


def build_balloon_pipeline() -> Pipeline:
    """Build the OGDF-style BalloonLayout pipeline.

    Returns
    -------
    Pipeline
        Four-stage deterministic BalloonLayout pipeline.
    """
    return Pipeline(
        [
            BalloonTreeBuild(),
            BalloonRadiiCompute(),
            BalloonAngleCompute(),
            BalloonCoordinatePlacement(),
        ],
        name="balloon_pipeline",
    )


def layout_balloon_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: Optional[int] = 42,
    edge_weights: Optional[torch.Tensor] = None,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run OGDF-style BalloonLayout without calling the OGDF runner.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor | None, optional
        Optional node sizes with shape ``[N, 2]``.
    seed : int | None, default=42
        Accepted for API consistency; BalloonLayout is deterministic.
    edge_weights : torch.Tensor | None, optional
        Accepted for API consistency; BalloonLayout ignores weights.
    fidelity_dtype : torch.dtype | None, optional
        Optional output dtype override.

    Returns
    -------
    torch.Tensor
        Coordinate tensor with shape ``[N, 2]``.
    """
    del seed, edge_weights
    problem = LayoutProblem(edge_index=edge_index, num_nodes=num_nodes, node_sizes=node_sizes)
    state = build_balloon_pipeline().apply(
        problem,
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device="cpu")),
    )
    if state.pos is None:
        raise RuntimeError("Balloon pipeline did not produce positions.")
    if fidelity_dtype is not None:
        return state.pos.to(dtype=fidelity_dtype)
    return state.pos


__all__ = [
    "BalloonAngleCompute",
    "BalloonCoordinatePlacement",
    "BalloonRadiiCompute",
    "BalloonTreeBuild",
    "build_balloon_pipeline",
    "layout_balloon_pipeline",
]
