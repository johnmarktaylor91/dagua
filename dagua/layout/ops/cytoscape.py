"""Cytoscape-family layout operations.

The operations in this module are native Python/PyTorch ports of the small
deterministic parts of Cytoscape's layout family plus a legacy CoSE-compatible
spring step. Reference adapters live under ``dagua.eval.competitors`` and are
not imported here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Optional

import torch

from dagua.layout.ops.base import Op
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_MIN_DISTANCE = 1.0e-9
_AVSDF_DEFAULT_NODE_SEPARATION = 60.0
_COSE_DEFAULT_NODE_WIDTH = 1.0
_COSE_DEFAULT_NODE_HEIGHT = 1.0


def _node_sizes(problem: LayoutProblem, device: torch.device) -> torch.Tensor:
    """Return node sizes with Cytoscape-compatible fallbacks.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable graph inputs.
    device : torch.device
        Target tensor device.

    Returns
    -------
    torch.Tensor
        Node sizes with shape ``[N, 2]``.
    """
    if problem.node_sizes is None:
        return torch.ones((problem.num_nodes, 2), dtype=torch.float64, device=device)
    sizes = problem.node_sizes.to(device=device, dtype=torch.float64)
    if sizes.ndim == 1:
        sizes = sizes[:, None].repeat(1, 2)
    if sizes.shape[1] == 1:
        sizes = sizes.repeat(1, 2)
    return sizes


def _adjacency(edge_index: torch.Tensor, num_nodes: int) -> list[set[int]]:
    """Build an undirected simple adjacency list.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    list[set[int]]
        Undirected neighbor sets.
    """
    neighbors = [set() for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return neighbors
    edges = edge_index.detach().cpu().long()
    for edge_pos in range(edges.shape[1]):
        source = int(edges[0, edge_pos].item())
        target = int(edges[1, edge_pos].item())
        if (
            source == target
            or source < 0
            or target < 0
            or source >= num_nodes
            or target >= num_nodes
        ):
            continue
        neighbors[source].add(target)
        neighbors[target].add(source)
    return neighbors


def _unique_edges(edge_index: torch.Tensor, num_nodes: int) -> list[tuple[int, int]]:
    """Return unique undirected non-self edges in encounter order.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    list[tuple[int, int]]
        Unique edge pairs.
    """
    seen: set[tuple[int, int]] = set()
    result: list[tuple[int, int]] = []
    if edge_index.numel() == 0:
        return result
    edges = edge_index.detach().cpu().long()
    for edge_pos in range(edges.shape[1]):
        source = int(edges[0, edge_pos].item())
        target = int(edges[1, edge_pos].item())
        if (
            source == target
            or source < 0
            or target < 0
            or source >= num_nodes
            or target >= num_nodes
        ):
            continue
        key = (source, target) if source < target else (target, source)
        if key not in seen:
            seen.add(key)
            result.append((source, target))
    return result


def _avsdf_order(neighbors: list[set[int]]) -> list[int]:
    """Compute the AVSDF adjacent-vertex-smallest-degree-first order.

    Parameters
    ----------
    neighbors : list[set[int]]
        Undirected neighbor sets.

    Returns
    -------
    list[int]
        Circular node order.
    """
    ordered = [False] * len(neighbors)
    stack: list[int] = []
    order: list[int] = []

    def smallest_unordered() -> Optional[int]:
        """Return the first unordered node with minimum degree.

        Returns
        -------
        int | None
            Node index or ``None`` when all nodes are ordered.
        """
        best_node: Optional[int] = None
        best_degree = math.inf
        for node_index, node_neighbors in enumerate(neighbors):
            degree = len(node_neighbors)
            if not ordered[node_index] and degree < best_degree:
                best_node = node_index
                best_degree = degree
        return best_node

    while len(order) < len(neighbors):
        node = None
        while stack and node is None:
            candidate = stack.pop()
            if not ordered[candidate]:
                node = candidate
        if node is None:
            node = smallest_unordered()
        if node is None:
            break
        ordered[node] = True
        order.append(node)
        candidates = [neighbor for neighbor in neighbors[node] if not ordered[neighbor]]
        candidates.sort(key=lambda item: (len(neighbors[item]), item))
        for neighbor in reversed(candidates):
            if not ordered[neighbor]:
                stack.append(neighbor)
    return order


def _circ_dist(index_by_node: list[int], source: int, target: int, size: int) -> int:
    """Return Cytoscape AVSDF clockwise circular index distance.

    Parameters
    ----------
    index_by_node : list[int]
        Node-to-index map.
    source : int
        Source node.
    target : int
        Target node.
    size : int
        Number of nodes on the circle.

    Returns
    -------
    int
        Clockwise distance in slots.
    """
    diff = index_by_node[target] - index_by_node[source]
    if diff < 0:
        diff += size
    return diff


def _edges_cross(
    index_by_node: list[int],
    first_edge: tuple[int, int],
    second_edge: tuple[int, int],
    size: int,
) -> bool:
    """Return whether two AVSDF circle chords cross.

    Parameters
    ----------
    index_by_node : list[int]
        Node-to-index map.
    first_edge : tuple[int, int]
        First edge.
    second_edge : tuple[int, int]
        Second edge.
    size : int
        Number of circle nodes.

    Returns
    -------
    bool
        ``True`` when the two chords cross under AVSDF's directed test.
    """
    source, target = first_edge
    other_source, other_target = second_edge
    if len({source, target, other_source, other_target}) < 4:
        return False
    other_source_dist = _circ_dist(index_by_node, source, other_source, size)
    other_target_dist = _circ_dist(index_by_node, source, other_target, size)
    this_target_dist = _circ_dist(index_by_node, source, target, size)
    return (
        min(other_source_dist, other_target_dist) < this_target_dist
        and this_target_dist < max(other_source_dist, other_target_dist)
        and other_source_dist != 0
        and other_target_dist != 0
    )


def _node_crossings(
    node: int,
    index_by_node: list[int],
    incident_edges: list[list[tuple[int, int]]],
    all_edges: list[tuple[int, int]],
) -> int:
    """Calculate AVSDF crossing count for one node's incident edges.

    Parameters
    ----------
    node : int
        Node whose incident crossings are counted.
    index_by_node : list[int]
        Node-to-index map.
    incident_edges : list[list[tuple[int, int]]]
        Edges incident to each node.
    all_edges : list[tuple[int, int]]
        All unique layout edges.

    Returns
    -------
    int
        Total crossing count for incident edges.
    """
    total = 0
    size = len(index_by_node)
    node_edges = incident_edges[node]
    for edge in node_edges:
        for other in all_edges:
            if other in node_edges:
                continue
            total += int(_edges_cross(index_by_node, edge, other, size))
    return total


def _avsdf_postprocess(order: list[int], edges: list[tuple[int, int]]) -> list[int]:
    """Run AVSDF's local crossing-reduction postprocess.

    Parameters
    ----------
    order : list[int]
        Initial circular order.
    edges : list[tuple[int, int]]
        Unique undirected edges.

    Returns
    -------
    list[int]
        Locally improved circular order.
    """
    size = len(order)
    if size < 4 or not edges:
        return order
    incident_edges: list[list[tuple[int, int]]] = [[] for _ in range(size)]
    for source, target in edges:
        incident_edges[source].append((source, target))
        incident_edges[target].append((source, target))

    index_by_node = [0] * size
    for order_index, node in enumerate(order):
        index_by_node[node] = order_index
    process_nodes = list(order)
    process_nodes.sort(
        key=lambda node: _node_crossings(node, index_by_node, incident_edges, edges),
        reverse=True,
    )

    for node in process_nodes:
        current = _node_crossings(node, index_by_node, incident_edges, edges)
        for neighbor_edge in incident_edges[node]:
            neighbor = neighbor_edge[1] if neighbor_edge[0] == node else neighbor_edge[0]
            old_index = index_by_node[node]
            new_index = (index_by_node[neighbor] + 1) % size
            if old_index == new_index:
                continue
            trial = index_by_node.copy()
            trial[node] = new_index
            shifted_old_index = old_index + size if old_index < new_index else old_index
            shift_index = new_index
            while shift_index < shifted_old_index:
                shifted_node = order[shift_index % size]
                trial[shifted_node] = (trial[shifted_node] + 1) % size
                shift_index += 1
            updated = _node_crossings(node, trial, incident_edges, edges)
            if updated < current:
                index_by_node = trial
                order = [0] * size
                for item, item_index in enumerate(index_by_node):
                    order[item_index] = item
                current = updated
    return order


@register_op
@dataclass(frozen=True)
class AVSDFLayoutOp(Op):
    """Place nodes with Cytoscape AVSDF circular ordering."""

    name: ClassVar[str] = "avsdf_layout"
    category: ClassVar[OpCategory] = OpCategory.COORDINATE
    reads: ClassVar[tuple[str, ...]] = ()
    writes: ClassVar[tuple[str, ...]] = ("pos", "extras")
    node_separation: float = _AVSDF_DEFAULT_NODE_SEPARATION
    postprocess: bool = True

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute AVSDF coordinates.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context with target device.

        Returns
        -------
        SolveState
            State with ``pos`` shaped ``[N, 2]`` and AVSDF order metadata.
        """
        device = torch.device(ctx.plan.device)
        if problem.num_nodes == 0:
            state.pos = torch.empty((0, 2), dtype=torch.float32, device=device)
            state.extras["avsdf_order"] = []
            return state
        sizes = _node_sizes(problem, device=torch.device("cpu"))
        diagonals = torch.linalg.vector_norm(sizes, dim=1).tolist()
        neighbors = _adjacency(problem.edge_index, problem.num_nodes)
        order = _avsdf_order(neighbors)
        edges = _unique_edges(problem.edge_index, problem.num_nodes)
        if self.postprocess:
            order = _avsdf_postprocess(order, edges)

        perimeter = float(sum(diagonals) + problem.num_nodes * self.node_separation)
        radius = perimeter / (2.0 * math.pi) if perimeter > 0.0 else 0.0
        center = 2.0 * radius
        pos = torch.zeros((problem.num_nodes, 2), dtype=torch.float64)
        previous_angle = 0.0
        for order_index, node in enumerate(order):
            if order_index == 0:
                angle = 0.0
            else:
                previous_node = order[order_index - 1]
                angle = previous_angle + (
                    2.0
                    * math.pi
                    * (
                        diagonals[node] / 2.0
                        + self.node_separation
                        + diagonals[previous_node] / 2.0
                    )
                    / max(perimeter, _MIN_DISTANCE)
                )
            pos[node, 0] = center + radius * math.cos(angle)
            pos[node, 1] = center + radius * math.sin(angle)
            previous_angle = angle

        state.pos = pos.to(device=device, dtype=torch.float32)
        state.extras["avsdf_order"] = order
        state.extras["avsdf_radius"] = radius
        return state


@register_op
@dataclass(frozen=True)
class CytoscapeCircleClusters(Op):
    """Arrange cluster members on separate circles for CiSE-style output."""

    name: ClassVar[str] = "cytoscape_circle_clusters"
    category: ClassVar[OpCategory] = OpCategory.COORDINATE
    reads: ClassVar[tuple[str, ...]] = ()
    writes: ClassVar[tuple[str, ...]] = ("pos", "extras")
    node_separation: float = _AVSDF_DEFAULT_NODE_SEPARATION

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Place cluster members on cluster-local circles.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs, optionally with ``clusters``.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context with target device.

        Returns
        -------
        SolveState
            State with circular cluster coordinates.
        """
        device = torch.device(ctx.plan.device)
        if not problem.clusters:
            return AVSDFLayoutOp(node_separation=self.node_separation).apply(problem, state, ctx)
        groups: list[list[int]] = []
        assigned: set[int] = set()
        for members in problem.clusters.values():
            if isinstance(members, dict):
                continue
            group = sorted(
                int(member) for member in members if 0 <= int(member) < problem.num_nodes
            )
            if group:
                groups.append(group)
                assigned.update(group)
        for node in range(problem.num_nodes):
            if node not in assigned:
                groups.append([node])
        outer_radius = max(self.node_separation, len(groups) * self.node_separation / math.pi)
        pos = torch.zeros((problem.num_nodes, 2), dtype=torch.float64)
        cluster_meta: list[dict[str, float]] = []
        for group_index, group in enumerate(groups):
            center_angle = 2.0 * math.pi * group_index / max(len(groups), 1)
            center_x = outer_radius * math.cos(center_angle)
            center_y = outer_radius * math.sin(center_angle)
            radius = max(self.node_separation, len(group) * self.node_separation / (2.0 * math.pi))
            for local_index, node in enumerate(group):
                angle = 2.0 * math.pi * local_index / max(len(group), 1)
                pos[node, 0] = center_x + radius * math.cos(angle)
                pos[node, 1] = center_y + radius * math.sin(angle)
            cluster_meta.append({"x": center_x, "y": center_y, "r": radius})
        state.pos = pos.to(device=device, dtype=torch.float32)
        state.extras["cise_cluster_circles"] = cluster_meta
        return state


@register_op
@dataclass(frozen=True)
class CytoscapeCoSEStep(Op):
    """Apply one legacy Cytoscape CoSE spring-embedder step."""

    name: ClassVar[str] = "cytoscape_cose_step"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[tuple[str, ...]] = ("pos",)
    writes: ClassVar[tuple[str, ...]] = ("pos", "forces")
    ideal_edge_length: float = 32.0
    node_repulsion: float = 2048.0
    edge_elasticity: float = 32.0
    gravity: float = 1.0
    temperature: float = 1000.0

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Advance positions by one CoSE force step.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state containing ``pos``.
        ctx : RuntimeContext
            Runtime context. Unused except for API consistency.

        Returns
        -------
        SolveState
            State with updated positions.
        """
        del ctx
        if state.pos is None:
            raise ValueError("CytoscapeCoSEStep requires initialized positions.")
        pos = state.pos
        device = pos.device
        dtype = pos.dtype
        sizes = _node_sizes(problem, device=device).to(dtype=dtype)
        forces = torch.zeros_like(pos)
        for source in range(problem.num_nodes):
            for target in range(source + 1, problem.num_nodes):
                delta = pos[target] - pos[source]
                distance_sq = torch.clamp(torch.dot(delta, delta), min=_MIN_DISTANCE)
                distance = torch.sqrt(distance_sq)
                force = (2.0 * self.node_repulsion) / distance_sq
                vector = force * delta / distance
                forces[source] -= vector
                forces[target] += vector
        for source, target in _unique_edges(problem.edge_index, problem.num_nodes):
            delta = pos[target] - pos[source]
            distance = torch.clamp(torch.linalg.vector_norm(delta), min=_MIN_DISTANCE)
            clipped_distance = torch.clamp(
                distance - 0.5 * torch.linalg.vector_norm(sizes[source] + sizes[target]),
                min=_MIN_DISTANCE,
            )
            force = ((self.ideal_edge_length - clipped_distance) ** 2) / self.edge_elasticity
            vector = force * delta / distance
            forces[source] += vector
            forces[target] -= vector
        if self.gravity > 0.0 and problem.num_nodes > 0:
            center = pos.mean(dim=0)
            gravity_delta = center[None, :] - pos
            gravity_dist = torch.clamp(
                torch.linalg.vector_norm(gravity_delta, dim=1),
                min=_MIN_DISTANCE,
            )
            forces += self.gravity * gravity_delta / gravity_dist[:, None]
        magnitude = torch.clamp(torch.linalg.vector_norm(forces, dim=1), min=_MIN_DISTANCE)
        scale = torch.clamp(torch.full_like(magnitude, self.temperature) / magnitude, max=1.0)
        state.pos = pos + forces * scale[:, None]
        state.forces = forces
        return state


@register_op
@dataclass(frozen=True)
class CytoscapeInitialPlacement(Op):
    """Initialize Cytoscape spring layouts deterministically."""

    name: ClassVar[str] = "cytoscape_initial_placement"
    category: ClassVar[OpCategory] = OpCategory.INIT
    reads: ClassVar[tuple[str, ...]] = ("pos",)
    writes: ClassVar[tuple[str, ...]] = ("pos",)
    randomize: bool = False
    extent: float = 1000.0

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Populate initial positions.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context with target device.

        Returns
        -------
        SolveState
            State with initialized positions.
        """
        device = torch.device(ctx.plan.device)
        if state.pos is not None and not self.randomize:
            state.pos = state.pos.to(device=device, dtype=torch.float32)
            return state
        if problem.num_nodes == 0:
            state.pos = torch.empty((0, 2), dtype=torch.float32, device=device)
            return state
        angles = torch.arange(problem.num_nodes, dtype=torch.float64)
        angles = 2.0 * math.pi * angles / max(problem.num_nodes, 1)
        radius = self.extent / (2.0 * math.pi)
        pos = torch.stack((radius * torch.cos(angles), radius * torch.sin(angles)), dim=1)
        if self.randomize:
            generator = torch.Generator(device="cpu").manual_seed(int(problem.seed))
            pos = torch.rand((problem.num_nodes, 2), generator=generator, dtype=torch.float64)
            pos = pos * self.extent
        state.pos = pos.to(device=device, dtype=torch.float32)
        return state


@register_op
@dataclass(frozen=True)
class CytoscapeFinalize(Op):
    """Center Cytoscape-family output."""

    name: ClassVar[str] = "cytoscape_finalize"
    category: ClassVar[OpCategory] = OpCategory.POSTPROCESS
    reads: ClassVar[tuple[str, ...]] = ("pos",)
    writes: ClassVar[tuple[str, ...]] = ("pos",)
    center: bool = True

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Center positions around the origin.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs. Unused.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context. Unused.

        Returns
        -------
        SolveState
            State with centered positions when requested.
        """
        del problem, ctx
        if self.center and state.pos is not None and state.pos.numel() > 0:
            state.pos = state.pos - state.pos.mean(dim=0, keepdim=True)
        return state
