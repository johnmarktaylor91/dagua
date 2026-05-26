"""Composable Large Graph Layout (LGL) operations.

These ops decompose the classic LGL implementation into reusable pipeline
steps so that ``dagua.layout.ops.pipelines.lgl`` can build its workflow from
registered operations only.
"""

from __future__ import annotations

import math
import random
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import ClassVar, List, Optional, Tuple

import torch

from dagua.layout.ops._igraph_rng import IgraphPCG32
from dagua.layout.ops.base import Op
from dagua.layout.ops.graph_utils import layout_device
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_LGL_MIN_DISTANCE = 1.0e-12
_LGL_REPULSION_MIN_DISTANCE = 1.0e-5
_LGL_BUCKET_NEIGHBOR_OFFSETS = ((0, 0), (1, 0), (0, 1), (1, 1))
_LGL_DISCONNECTED_WARNING = "LGL layout does not support disconnected graphs yet."
_LGL_BENCHMARK_OUTPUT_SCALE = 50.0
RandomLike = random.Random | IgraphPCG32


@dataclass(frozen=True)
class LGLPrepareStateConfig:
    """Configuration for :class:`LGLPrepareState`.

    Parameters
    ----------
    maxiter : int, default=150
        Maximum number of refinement iterations for each growth shell.
    maxdelta : float, optional
        Initial temperature bound for refinement.
    area : float, optional
        Drawing area; defaults to ``num_nodes ** 2``.
    coolexp : float, default=1.5
        Temperature cooling exponent.
    repulserad : float, optional
        Repulsion radius used by the LGL spring cutoff rule.
    cellsize : float, optional
        Sparse grid cell size for repulsion.
    root : int, optional
        Optional root vertex used by BFS shell growth.
    use_edge_weights : bool, default=False
        Whether edge weights scale attractive forces. Igraph LGL ignores
        weights, so the default preserves fidelity with that reference.
    fidelity_mode : bool, default=False
        When ``True``, match the python-igraph benchmark adapter's seeded
        Python RNG stream.
    """

    maxiter: int = 150
    maxdelta: Optional[float] = None
    area: Optional[float] = None
    coolexp: float = 1.5
    repulserad: Optional[float] = None
    cellsize: Optional[float] = None
    root: Optional[int] = None
    use_edge_weights: bool = False
    fidelity_mode: bool = False


@dataclass(frozen=True)
class LGLLayeredRefinementConfig:
    """Configuration for :class:`LGLLayeredRefinement`.

    Parameters
    ----------
    convergence_epsilon : float, default=1e-5
        Early-stop threshold for the maximum node movement inside one shell's
        local refinement loop.
    igraph_positive_maxchange : bool, default=True
        Match igraph LGL's historical convergence rule, which only considers
        positive movement components when updating ``maxchange``.
    fidelity_mode : bool, default=False
        When ``True``, match the python-igraph benchmark adapter's seeded
        Python RNG stream.
    """

    convergence_epsilon: float = 1.0e-5
    igraph_positive_maxchange: bool = True
    fidelity_mode: bool = False


def _build_lgl_bfs_layers(
    num_nodes: int,
    root_node: int,
    adjacency: List[List[int]],
) -> Tuple[List[List[int]], List[int], List[int]]:
    """Build BFS shells used by LGL shell growth.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    root_node : int
        Root vertex for breadth-first traversal.
    adjacency : List[List[int]]
        Undirected adjacency list with one sorted neighbor list per node.

    Returns
    -------
    Tuple[List[List[int]], List[int], List[int]]
        Per-depth node layers, BFS parent list, and node-distance list. Nodes
        unreachable from ``root_node`` keep parent and distance values of ``-1``.
    """
    layers: list[list[int]] = []
    parents: list[int] = [-1] * num_nodes
    distance: list[int] = [-1] * num_nodes
    bfs_queue: deque[int] = deque([root_node])
    parents[root_node] = root_node
    distance[root_node] = 0
    while bfs_queue:
        node = bfs_queue.popleft()
        depth = distance[node]
        while len(layers) <= depth:
            layers.append([])
        layers[depth].append(node)
        for neighbor in adjacency[node]:
            if distance[neighbor] != -1:
                continue
            parents[neighbor] = node
            distance[neighbor] = depth + 1
            bfs_queue.append(neighbor)

    return layers, parents, distance


def _build_igraph_bfs_simple(
    num_nodes: int,
    root_node: int,
    adjacency: List[List[int]],
) -> tuple[list[int], list[int], list[int]]:
    """Build igraph ``igraph_bfs_simple`` order, layer bounds, and parents.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    root_node : int
        Root vertex for breadth-first traversal.
    adjacency : List[List[int]]
        Undirected adjacency list. Neighbor order must already match igraph's
        ascending vertex order for ``IGRAPH_ALL`` traversals.

    Returns
    -------
    tuple[list[int], list[int], list[int]]
        BFS order vector, layer-start vector with a final sentinel, and parent
        vector where the root is ``-1`` and unreachable vertices are ``-2``.
    """
    order: list[int] = [root_node]
    layer_bounds: list[int] = [0]
    parents: list[int] = [-2] * num_nodes
    parents[root_node] = -1
    added = [False] * num_nodes
    added[root_node] = True
    bfs_queue: deque[tuple[int, int]] = deque([(root_node, 0)])
    visited_count = 1
    last_layer = -1

    while bfs_queue:
        node, distance = bfs_queue.popleft()
        for neighbor in adjacency[node]:
            if added[neighbor]:
                continue
            added[neighbor] = True
            parents[neighbor] = node
            bfs_queue.append((neighbor, distance + 1))
            if last_layer != distance + 1:
                layer_bounds.append(visited_count)
            order.append(neighbor)
            visited_count += 1
            last_layer = distance + 1

    layer_bounds.append(visited_count)
    return order, layer_bounds, parents


def _lgl_updated_maxchange(
    maxchange: float,
    movement: torch.Tensor,
    *,
    igraph_positive_only: bool,
) -> float:
    """Update shell-refinement convergence tracking.

    Parameters
    ----------
    maxchange : float
        Current maximum movement component.
    movement : torch.Tensor
        Movement vector with shape ``[2]`` for one node.
    igraph_positive_only : bool
        If true, preserve igraph LGL's positive-component convergence quirk.

    Returns
    -------
    float
        Updated maximum component value for convergence testing.
    """
    x_movement = float(movement[0].item())
    y_movement = float(movement[1].item())
    if igraph_positive_only:
        return max(maxchange, x_movement, y_movement)
    return max(maxchange, abs(x_movement), abs(y_movement))


def _lgl_grid_steps(radius: float, cellsize: float) -> int:
    """Compute igraph's bounded two-dimensional grid size.

    Parameters
    ----------
    radius : float
        Positive half-width of the LGL grid bounds.
    cellsize : float
        Positive grid cell width.

    Returns
    -------
    int
        Number of cells along one axis, with a minimum of one cell.
    """
    safe_cellsize = max(float(cellsize), _LGL_MIN_DISTANCE)
    return max(int(math.ceil((2.0 * float(radius)) / safe_cellsize)), 1)


def _lgl_clamped_grid_axis(
    value: float,
    lower_bound: float,
    upper_bound: float,
    cellsize: float,
    steps: int,
) -> int:
    """Map one coordinate axis into igraph's bounded grid.

    Parameters
    ----------
    value : float
        Coordinate value to place.
    lower_bound : float
        Inclusive lower grid bound.
    upper_bound : float
        Inclusive upper grid bound.
    cellsize : float
        Positive grid cell width.
    steps : int
        Number of cells along the axis.

    Returns
    -------
    int
        Clamped integer cell index.
    """
    max_index = max(int(steps) - 1, 0)
    if value <= lower_bound:
        return 0
    if value >= upper_bound:
        return max_index
    safe_cellsize = max(float(cellsize), _LGL_MIN_DISTANCE)
    return min(int(math.floor((value - lower_bound) / safe_cellsize)), max_index)


def _lgl_clamped_grid_cell(
    x_value: float,
    y_value: float,
    radius: float,
    cellsize: float,
    steps: int,
) -> tuple[int, int]:
    """Map a point into igraph's bounded LGL grid cell.

    Parameters
    ----------
    x_value : float
        X coordinate of the point.
    y_value : float
        Y coordinate of the point.
    radius : float
        Positive half-width of the LGL grid bounds.
    cellsize : float
        Positive grid cell width.
    steps : int
        Number of cells along each axis.

    Returns
    -------
    tuple[int, int]
        Clamped integer grid cell ``(x, y)``.
    """
    lower_bound = -float(radius)
    upper_bound = float(radius)
    return (
        _lgl_clamped_grid_axis(x_value, lower_bound, upper_bound, cellsize, steps),
        _lgl_clamped_grid_axis(y_value, lower_bound, upper_bound, cellsize, steps),
    )


@dataclass
class _IgraphLGLGrid:
    """Minimal port of igraph's ``igraph_2dgrid_t`` for LGL fidelity mode.

    Parameters
    ----------
    positions : torch.Tensor
        Mutable coordinate tensor with shape ``[N, 2]`` and dtype ``float64``.
    radius : float
        Positive half-width used for bounded grid clamping.
    cellsize : float
        Positive cell width along each axis.
    """

    positions: torch.Tensor
    radius: float
    cellsize: float

    def __post_init__(self) -> None:
        """Initialize igraph-compatible linked-list cell storage."""
        self.minx = -float(self.radius)
        self.maxx = float(self.radius)
        self.miny = -float(self.radius)
        self.maxy = float(self.radius)
        self.stepsx = _lgl_grid_steps(radius=self.radius, cellsize=self.cellsize)
        self.stepsy = self.stepsx
        self.startidx: list[list[int]] = [
            [0 for _ in range(self.stepsy)] for _ in range(self.stepsx)
        ]
        point_count = int(self.positions.shape[0])
        self.next: list[int] = [0] * point_count
        self.prev: list[int] = [0] * point_count
        self.massx = 0.0
        self.massy = 0.0
        self.vertices = 0

    def _which(self, x_coord: float, y_coord: float) -> tuple[int, int]:
        """Map coordinates to bounded igraph grid cell indices.

        Parameters
        ----------
        x_coord : float
            X coordinate.
        y_coord : float
            Y coordinate.

        Returns
        -------
        tuple[int, int]
            Clamped ``(x, y)`` grid cell indices.
        """
        return (
            _lgl_clamped_grid_axis(x_coord, self.minx, self.maxx, self.cellsize, self.stepsx),
            _lgl_clamped_grid_axis(y_coord, self.miny, self.maxy, self.cellsize, self.stepsy),
        )

    def add(self, node: int, x_coord: float, y_coord: float) -> None:
        """Add one vertex to the grid at an explicit coordinate.

        Parameters
        ----------
        node : int
            Vertex index to add.
        x_coord : float
            X coordinate assigned to the vertex.
        y_coord : float
            Y coordinate assigned to the vertex.

        Returns
        -------
        None
            The grid links, mass, and ``positions`` tensor are updated in place.
        """
        self.positions[node, 0] = x_coord
        self.positions[node, 1] = y_coord
        cell_x, cell_y = self._which(x_coord, y_coord)
        first = self.startidx[cell_x][cell_y]
        self.prev[node] = 0
        self.next[node] = first
        if first != 0:
            self.prev[first - 1] = node + 1
        self.startidx[cell_x][cell_y] = node + 1
        self.massx += x_coord
        self.massy += y_coord
        self.vertices += 1

    def move(self, node: int, x_delta: float, y_delta: float) -> None:
        """Move a vertex using igraph's linked-cell update semantics.

        Parameters
        ----------
        node : int
            Vertex index to move.
        x_delta : float
            X displacement.
        y_delta : float
            Y displacement.

        Returns
        -------
        None
            The grid links, mass, and ``positions`` tensor are updated in place.
        """
        old_x = float(self.positions[node, 0].item())
        old_y = float(self.positions[node, 1].item())
        new_x = old_x + x_delta
        new_y = old_y + y_delta
        old_cell_x, old_cell_y = self._which(old_x, old_y)
        new_cell_x, new_cell_y = self._which(new_x, new_y)

        if old_cell_x != new_cell_x or old_cell_y != new_cell_y:
            previous_node = self.prev[node]
            next_node = self.next[node]
            if previous_node != 0:
                self.next[previous_node - 1] = next_node
            else:
                self.startidx[old_cell_x][old_cell_y] = next_node
            if next_node != 0:
                self.prev[next_node - 1] = previous_node

            first = self.startidx[new_cell_x][new_cell_y]
            self.prev[node] = 0
            self.next[node] = first
            if first != 0:
                self.prev[first - 1] = node + 1
            self.startidx[new_cell_x][new_cell_y] = node + 1

        self.massx += -old_x + new_x
        self.massy += -old_y + new_y
        self.positions[node, 0] = new_x
        self.positions[node, 1] = new_y

    def center(self) -> tuple[float, float]:
        """Return the current grid center of mass.

        Returns
        -------
        tuple[float, float]
            Mean ``(x, y)`` coordinate of vertices added or moved through the
            grid. This intentionally follows igraph's mutable mass counters.
        """
        return self.massx / float(self.vertices), self.massy / float(self.vertices)

    def in_grid(self, node: int) -> bool:
        """Return igraph's historical grid-membership predicate.

        Parameters
        ----------
        node : int
            Vertex index to test.

        Returns
        -------
        bool
            Whether igraph would report the vertex in the grid. LGL relies on
            the fact that zero-initialized ``next`` entries satisfy this check.
        """
        return self.next[node] != -1

    def iter_neighbor_pairs(self) -> list[tuple[int, int]]:
        """Enumerate nearby vertex pairs in igraph grid iterator order.

        Returns
        -------
        list[tuple[int, int]]
            Ordered zero-based vertex pairs produced by
            ``igraph_2dgrid_next`` and ``igraph_2dgrid_next_nei``.
        """
        pairs: list[tuple[int, int]] = []
        x_cell = 0
        y_cell = 0
        vertex = self.startidx[0][0]
        while vertex == 0 and (x_cell < self.stepsx - 1 or y_cell < self.stepsy - 1):
            x_cell += 1
            if x_cell == self.stepsx:
                x_cell = 0
                y_cell += 1
            vertex = self.startidx[x_cell][y_cell]

        while vertex != 0:
            current = vertex
            neighbor_cells: list[tuple[int, int]] = []
            if x_cell != self.stepsx - 1:
                neighbor_cells.append((x_cell + 1, y_cell))
            if y_cell != self.stepsy - 1:
                neighbor_cells.append((x_cell, y_cell + 1))
            if len(neighbor_cells) == 2:
                neighbor_cells.append((x_cell + 1, y_cell + 1))
            neighbor_cells.append((x_cell, y_cell))

            neighbor = self.next[current - 1]
            next_cell_index = len(neighbor_cells) - 1
            while next_cell_index > 0 and neighbor == 0:
                next_cell_index -= 1
                neighbor_x, neighbor_y = neighbor_cells[next_cell_index]
                neighbor = self.startidx[neighbor_x][neighbor_y]
            while neighbor != 0:
                pairs.append((current - 1, neighbor - 1))
                neighbor = self.next[neighbor - 1]
                while next_cell_index > 0 and neighbor == 0:
                    next_cell_index -= 1
                    neighbor_x, neighbor_y = neighbor_cells[next_cell_index]
                    neighbor = self.startidx[neighbor_x][neighbor_y]

            vertex = self.next[vertex - 1]
            while (x_cell < self.stepsx - 1 or y_cell < self.stepsy - 1) and vertex == 0:
                x_cell += 1
                if x_cell == self.stepsx:
                    x_cell = 0
                    y_cell += 1
                vertex = self.startidx[x_cell][y_cell]

        return pairs


def _normalize_lgl_pair(x_value: float, y_value: float) -> tuple[float, float]:
    """Normalize a two-dimensional vector like igraph's LGL helper.

    Parameters
    ----------
    x_value : float
        X component.
    y_value : float
        Y component.

    Returns
    -------
    tuple[float, float]
        Unit vector components, or the original zero vector when its length is
        exactly zero.
    """
    length = math.sqrt(x_value * x_value + y_value * y_value)
    if length != 0.0:
        return x_value / length, y_value / length
    return x_value, y_value


def _lgl_python_igraph_root_draw(rng: random.Random, num_nodes: int) -> int:
    """Draw a random LGL root like python-igraph's external RNG bridge.

    Parameters
    ----------
    rng : random.Random
        Seeded Python RNG installed by the benchmark adapter.
    num_nodes : int
        Number of vertices in the graph.

    Returns
    -------
    int
        Root vertex index in ``[0, num_nodes)``.
    """
    return (rng.getrandbits(32) * num_nodes) >> 32


def _build_lgl_incident_edges(
    num_nodes: int,
    edges: list[tuple[int, int]],
) -> list[list[int]]:
    """Build igraph-like incident edge lists for ``IGRAPH_ALL`` traversal.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    edges : list[tuple[int, int]]
        Directed edge list in edge-id order.

    Returns
    -------
    list[list[int]]
        Incident edge IDs for each vertex, ordered by adjacent vertex then edge
        ID to match igraph on simple graphs and keep duplicate handling stable.
    """
    incident: list[list[tuple[int, int]]] = [[] for _ in range(num_nodes)]
    for edge_id, (source, target) in enumerate(edges):
        incident[source].append((target, edge_id))
        if target != source:
            incident[target].append((source, edge_id))
    return [[edge_id for _, edge_id in sorted(items)] for items in incident]


def _run_igraph_lgl_refinement(
    positions: torch.Tensor,
    *,
    num_nodes: int,
    seed: int,
    root_node: int,
    root_was_random: bool,
    adjacency: List[List[int]],
    directed_edges: list[tuple[int, int]],
    maxiter: int,
    maxdelta: float,
    coolexp: float,
    frk: float,
    repulserad: float,
    cellsize: float,
    radius: float,
) -> torch.Tensor:
    """Run the igraph LGL C algorithm in Python for fidelity mode.

    Parameters
    ----------
    positions : torch.Tensor
        Initial position tensor with shape ``[N, 2]`` and dtype ``float64``.
    num_nodes : int
        Number of graph nodes.
    seed : int
        Benchmark seed used by python-igraph's installed Python RNG.
    root_node : int
        BFS root vertex.
    root_was_random : bool
        Whether root selection consumed one RNG draw before random layout.
    adjacency : List[List[int]]
        Undirected adjacency list for BFS traversal.
    directed_edges : list[tuple[int, int]]
        Edge list in igraph edge-id order.
    maxiter : int
        Maximum cooling iterations per layer.
    maxdelta : float
        Initial maximum per-iteration movement.
    coolexp : float
        Cooling exponent.
    frk : float
        Fruchterman-Reingold spacing constant.
    repulserad : float
        Repulsion cutoff radius.
    cellsize : float
        Grid cell size and local repulsion cutoff.
    radius : float
        Half-width of igraph's bounded grid.

    Returns
    -------
    torch.Tensor
        The updated ``positions`` tensor.
    """
    order, layer_bounds, parents = _build_igraph_bfs_simple(num_nodes, root_node, adjacency)
    if min(parents) <= -2:
        warnings.warn(_LGL_DISCONNECTED_WARNING, UserWarning, stacklevel=2)

    grid = _IgraphLGLGrid(positions=positions, radius=radius, cellsize=cellsize)
    grid.add(root_node, 0.0, 0.0)
    no_of_layers = len(layer_bounds) - 1
    harmonic = sum(1.0 / float(layer_index) for layer_index in range(1, no_of_layers))
    if harmonic == 0.0:
        return positions
    shell_constant = radius / harmonic
    incident_edges = _build_lgl_incident_edges(num_nodes, directed_edges)
    active_edge_ids: list[int] = []
    rng = random.Random(seed)
    if root_was_random:
        _lgl_python_igraph_root_draw(rng, num_nodes)
    for _ in range(2 * num_nodes):
        rng.uniform(-1.0, 1.0)

    epsilon = 10.0e-6
    for active_layer in range(1, no_of_layers):
        child_index = layer_bounds[active_layer]
        for order_index in range(layer_bounds[active_layer - 1], layer_bounds[active_layer]):
            vertex = order[order_index]
            parent = parents[vertex]
            if parent < 0:
                if parent == -1:
                    positions[vertex, 0] = 0.0
                    positions[vertex, 1] = 0.0
                continue

            mass_x, mass_y = grid.center()
            mass_x, mass_y = _normalize_lgl_pair(mass_x, mass_y)
            parent_x = float(positions[vertex, 0].item()) - float(positions[parent, 0].item())
            parent_y = float(positions[vertex, 1].item()) - float(positions[parent, 1].item())
            parent_x, parent_y = _normalize_lgl_pair(parent_x, parent_y)
            sphere_x = mass_x + parent_x + float(positions[vertex, 0].item())
            sphere_y = mass_y + parent_y + float(positions[vertex, 1].item())

            while (
                child_index < layer_bounds[active_layer + 1]
                and parents[order[child_index]] == vertex
            ):
                if active_layer == 1:
                    phi = 2.0 * math.pi / float(layer_bounds[2] - 1) * float(child_index - 1)
                    radial_x = math.cos(phi)
                    radial_y = math.sin(phi)
                else:
                    radial_x = rng.uniform(-1.0, 1.0)
                    radial_y = rng.uniform(-1.0, 1.0)
                radial_x, radial_y = _normalize_lgl_pair(radial_x, radial_y)
                radial_x = radial_x / float(active_layer) * shell_constant
                radial_y = radial_y / float(active_layer) * shell_constant
                grid.add(order[child_index], sphere_x + radial_x, sphere_y + radial_y)
                child_index += 1

        for order_index in range(layer_bounds[active_layer], layer_bounds[active_layer + 1]):
            vertex = order[order_index]
            for edge_id in incident_edges[vertex]:
                source, target = directed_edges[edge_id]
                if (source != vertex and grid.in_grid(source)) or (
                    target != vertex and grid.in_grid(target)
                ):
                    active_edge_ids.append(edge_id)

        iteration = 0
        maxchange = epsilon + 1.0
        while iteration < maxiter and maxchange > epsilon:
            temperature = maxdelta * (((maxiter - iteration) / float(maxiter)) ** coolexp)
            force_x = [0.0] * num_nodes
            force_y = [0.0] * num_nodes
            maxchange = 0.0

            for edge_id in active_edge_ids:
                source, target = directed_edges[edge_id]
                x_delta = float(positions[source, 0].item()) - float(positions[target, 0].item())
                y_delta = float(positions[source, 1].item()) - float(positions[target, 1].item())
                distance = math.sqrt(x_delta * x_delta + y_delta * y_delta)
                if distance != 0.0:
                    x_delta /= distance
                    y_delta /= distance
                force = distance * distance / frk
                force_x[source] -= x_delta * force
                force_x[target] += x_delta * force
                force_y[source] -= y_delta * force
                force_y[target] += y_delta * force

            for vertex, neighbor in grid.iter_neighbor_pairs():
                x_delta = float(positions[vertex, 0].item()) - float(positions[neighbor, 0].item())
                y_delta = float(positions[vertex, 1].item()) - float(positions[neighbor, 1].item())
                distance = math.sqrt(x_delta * x_delta + y_delta * y_delta)
                if distance < cellsize:
                    if distance == 0.0:
                        distance = epsilon
                    x_delta /= distance
                    y_delta /= distance
                    force = frk * frk * (1.0 / distance - distance * distance / repulserad)
                    force_x[vertex] += x_delta * force
                    force_x[neighbor] -= x_delta * force
                    force_y[vertex] += y_delta * force
                    force_y[neighbor] -= y_delta * force

            for order_index in range(layer_bounds[active_layer + 1]):
                vertex = order[order_index]
                move_x = force_x[vertex]
                move_y = force_y[vertex]
                distance = math.sqrt(move_x * move_x + move_y * move_y)
                if distance > temperature:
                    scale = temperature / distance
                    move_x *= scale
                    move_y *= scale
                grid.move(vertex, move_x, move_y)
                if move_x > maxchange:
                    maxchange = move_x
                if move_y > maxchange:
                    maxchange = move_y
            iteration += 1

    return positions


@register_op
@dataclass(frozen=True)
class LGLPrepareState(Op):
    """Populate LGL metadata and graph structures required by later ops."""

    config: LGLPrepareStateConfig = field(default_factory=LGLPrepareStateConfig)

    name: ClassVar[str] = "lgl_prepare_state"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Resolve defaults and build LGL graph-structure caches.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable problem inputs.
        state : SolveState
            Mutable working state receiving LGL metadata.
        ctx : RuntimeContext
            Runtime context (unused).

        Returns
        -------
        SolveState
            State with populated ``state.extras`` fields:
            ``lgl_maxiter``, ``lgl_area``, ``lgl_adjacency``,
            ``lgl_spring_edges``, ``lgl_root``, and related scalar settings.
        """
        del ctx

        num_nodes = problem.num_nodes
        resolved_maxdelta = (
            float(num_nodes) if self.config.maxdelta is None else self.config.maxdelta
        )
        resolved_area = (
            float(num_nodes * num_nodes) if self.config.area is None else self.config.area
        )
        resolved_repulserad = (
            resolved_area * float(max(num_nodes, 1))
            if self.config.repulserad is None
            else self.config.repulserad
        )
        resolved_cellsize = (
            resolved_area**0.25 if self.config.cellsize is None else self.config.cellsize
        )

        adjacency_sets = [set() for _ in range(num_nodes)]
        spring_edges: list[tuple[int, int]] = []
        spring_edge_weights: Optional[list[float]] = None
        if problem.edge_index.numel() > 0:
            edge_index_cpu = problem.edge_index.to(device="cpu", dtype=torch.long)
            source_nodes = edge_index_cpu[0].tolist()
            target_nodes = edge_index_cpu[1].tolist()
            weight_list: Optional[list[float]] = None
            if self.config.use_edge_weights and problem.edge_weights is not None:
                spring_edge_weights = []
                weight_list = (
                    problem.edge_weights.detach().to(device="cpu", dtype=torch.float64).tolist()
                )

            for edge_idx, (source, target) in enumerate(zip(source_nodes, target_nodes)):
                if source == target:
                    continue
                lower = min(source, target)
                upper = max(source, target)
                adjacency_sets[lower].add(upper)
                adjacency_sets[upper].add(lower)
                # Keep edge multiplicity to match classic LGL behavior.
                spring_edges.append((lower, upper))
                if weight_list is not None:
                    spring_edge_weights.append(float(weight_list[edge_idx]))

        adjacency = [sorted(neighbors) for neighbors in adjacency_sets]

        rng: RandomLike = random.Random(problem.seed)
        if self.config.root is None:
            root_node = (
                _lgl_python_igraph_root_draw(rng, num_nodes)
                if self.config.fidelity_mode
                else rng.randrange(num_nodes)
            )
        else:
            root_node = self.config.root
        if root_node < 0 or root_node >= num_nodes:
            raise ValueError("root must lie in [0, num_nodes).")
        frk = math.sqrt(resolved_area / float(max(num_nodes, 1)))

        state.extras["lgl_maxiter"] = self.config.maxiter
        state.extras["lgl_maxdelta"] = resolved_maxdelta
        state.extras["lgl_area"] = resolved_area
        state.extras["lgl_coolexp"] = self.config.coolexp
        state.extras["lgl_repulserad"] = resolved_repulserad
        state.extras["lgl_cellsize"] = resolved_cellsize
        state.extras["lgl_root"] = root_node
        state.extras["lgl_frk"] = frk
        state.extras["lgl_adjacency"] = adjacency
        state.extras["lgl_spring_edges"] = spring_edges
        state.extras["lgl_directed_edges"] = (
            [
                (int(source), int(target))
                for source, target in zip(
                    problem.edge_index.to(device="cpu", dtype=torch.long)[0].tolist(),
                    problem.edge_index.to(device="cpu", dtype=torch.long)[1].tolist(),
                )
                if int(source) != int(target)
            ]
            if problem.edge_index.numel() > 0
            else []
        )
        state.extras["lgl_spring_edge_weights"] = spring_edge_weights
        return state


@dataclass(frozen=True)
class LGLInitializePositionsConfig:
    """Configuration for :class:`LGLInitializePositions`.

    Parameters
    ----------
    fidelity_mode : bool, default=False
        When ``True``, match the python-igraph benchmark adapter's seeded
        Python RNG stream.
    """

    fidelity_mode: bool = False


@register_op
@dataclass(frozen=True)
class LGLInitializePositions(Op):
    """Initialize layout positions with the classic LGL random seed behavior."""

    config: LGLInitializePositionsConfig = field(default_factory=LGLInitializePositionsConfig)

    name: ClassVar[str] = "lgl_initialize_positions"
    category: ClassVar[OpCategory] = OpCategory.INIT
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Seed ``state.pos`` as a random uniform square cloud.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable problem inputs.
        state : SolveState
            Mutable state storing layout coordinates.
        ctx : RuntimeContext
            Runtime context (unused).

        Returns
        -------
        SolveState
            Updated state with initial ``state.pos`` and root fixed at origin.
        """
        del ctx

        area = float(state.extras["lgl_area"])
        radius = math.sqrt(area / math.pi)
        rng: RandomLike = random.Random(problem.seed)
        if bool(state.extras.get("lgl_root_was_random", True)):
            if self.config.fidelity_mode:
                _lgl_python_igraph_root_draw(rng, problem.num_nodes)
            else:
                rng.randrange(problem.num_nodes)
        positions = torch.empty((problem.num_nodes, 2), dtype=torch.float64)
        for axis in range(2):
            for node in range(problem.num_nodes):
                positions[node, axis] = rng.uniform(-1.0, 1.0) * radius
        positions[int(state.extras["lgl_root"])] = 0.0
        state.pos = positions
        return state


@register_op
@dataclass(frozen=True)
class LGLLayeredRefinement(Op):
    """Run BFS shell growth and local FR-style relaxation on each layer."""

    config: LGLLayeredRefinementConfig = field(default_factory=LGLLayeredRefinementConfig)

    name: ClassVar[str] = "lgl_shell_growth_solve"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("pos", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Place shells in BFS order and refine with active-node FR updates.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable problem inputs.
        state : SolveState
            Mutable state with initialization and LGL metadata.
        ctx : RuntimeContext
            Runtime context (unused).

        Returns
        -------
        SolveState
            Updated positions after all shell-placement iterations.
        """
        del ctx

        if state.pos is None:
            raise ValueError("LGLLayeredRefinement requires state.pos to be set.")

        num_nodes = problem.num_nodes
        positions = state.pos
        root_node = int(state.extras["lgl_root"])
        adjacency = state.extras["lgl_adjacency"]
        spring_edges = state.extras["lgl_spring_edges"]
        spring_edge_weights = state.extras["lgl_spring_edge_weights"]
        maxiter = int(state.extras["lgl_maxiter"])
        maxdelta = float(state.extras["lgl_maxdelta"])
        coolexp = float(state.extras["lgl_coolexp"])
        frk = float(state.extras["lgl_frk"])
        repulserad = float(state.extras["lgl_repulserad"])
        cellsize = float(state.extras["lgl_cellsize"])
        area = float(state.extras["lgl_area"])
        radius = math.sqrt(area / math.pi)

        if self.config.fidelity_mode:
            state.pos = _run_igraph_lgl_refinement(
                positions=positions,
                num_nodes=num_nodes,
                seed=problem.seed,
                root_node=root_node,
                root_was_random=bool(state.extras.get("lgl_root_was_random", True)),
                adjacency=adjacency,
                directed_edges=state.extras["lgl_directed_edges"],
                maxiter=maxiter,
                maxdelta=maxdelta,
                coolexp=coolexp,
                frk=frk,
                repulserad=repulserad,
                cellsize=cellsize,
                radius=radius,
            )
            return state

        layers, parents, distance = _build_lgl_bfs_layers(num_nodes, root_node, adjacency)

        if not layers:
            state.pos = positions
            return state
        if any(node_distance < 0 for node_distance in distance):
            warnings.warn(_LGL_DISCONNECTED_WARNING, UserWarning, stacklevel=2)

        placed = torch.zeros(num_nodes, dtype=torch.bool)
        placed[root_node] = True

        active_edges: list[tuple[int, int]] = []
        active_edge_weights: Optional[list[float]]
        if spring_edge_weights is None:
            active_edge_weights = None
        else:
            active_edge_weights = []

        edge_active = [False] * len(spring_edges)
        incident_edge_indices: List[List[int]] = [[] for _ in range(num_nodes)]
        for edge_idx, (source, target) in enumerate(spring_edges):
            incident_edge_indices[source].append(edge_idx)
            incident_edge_indices[target].append(edge_idx)

        num_terms = max(len(layers) - 1, 1)
        shell_scale = radius / (
            sum(1.0 / float(index) for index in range(1, num_terms + 1)) if num_terms > 0 else 1.0
        )

        rng: RandomLike = random.Random(problem.seed)
        if bool(state.extras.get("lgl_root_was_random", True)):
            rng.randrange(num_nodes)
        for _ in range(2 * num_nodes):
            rng.uniform(-1.0, 1.0)

        for layer_index in range(len(layers) - 1):
            current_layer = layers[layer_index]
            next_layer = layers[layer_index + 1]
            if not next_layer:
                continue

            active_node_tensor = torch.nonzero(placed, as_tuple=False).view(-1)
            if active_node_tensor.numel() > 0:
                center_of_mass = positions[active_node_tensor].mean(dim=0)
                center_norm = float(torch.linalg.norm(center_of_mass).item())
                center_direction = (
                    center_of_mass / center_norm
                    if center_norm > _LGL_MIN_DISTANCE
                    else torch.zeros_like(center_of_mass)
                )
            else:
                center_direction = torch.zeros(2, dtype=torch.float64)

            next_depth = layer_index + 1

            for parent in current_layer:
                # Respect the BFS tree here so each shell inherits a stable
                # radial anchor before the FR refinement redistributes it.
                children = [node for node in next_layer if parents[node] == parent]
                if not children:
                    continue

                if parent == root_node:
                    parent_direction = torch.zeros(2, dtype=torch.float64)
                else:
                    parent_of_parent = parents[parent]
                    parent_delta = positions[parent] - positions[parent_of_parent]
                    parent_norm = float(torch.linalg.norm(parent_delta).item())
                    parent_direction = (
                        parent_delta / parent_norm
                        if parent_norm > _LGL_MIN_DISTANCE
                        else torch.zeros_like(parent_delta)
                    )

                anchor = positions[parent] + center_direction + parent_direction
                for child in children:
                    direction = torch.tensor(
                        [rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)],
                        dtype=torch.float64,
                    )
                    direction_norm = float(torch.linalg.norm(direction).item())
                    if direction_norm > _LGL_MIN_DISTANCE:
                        direction = direction / direction_norm
                    else:
                        direction = torch.tensor([1.0, 0.0], dtype=torch.float64)

                    offset = direction * (shell_scale / float(max(next_depth, 1)))
                    positions[child] = anchor + offset
                    placed[child] = True

                    for edge_idx in incident_edge_indices[child]:
                        if edge_active[edge_idx]:
                            continue
                        source, target = spring_edges[edge_idx]
                        # igraph_2dgrid_in() is effectively true for all
                        # vertices, so LGL activates incident springs even
                        # when the opposite endpoint is in a later shell.
                        edge_active[edge_idx] = True
                        active_edges.append((source, target))
                        if active_edge_weights is not None:
                            active_edge_weights.append(float(spring_edge_weights[edge_idx]))

            refinement_nodes = torch.nonzero(placed, as_tuple=False).view(-1).tolist()
            if not refinement_nodes or maxiter == 0:
                continue

            node_count = positions.shape[0]
            weight_tensor = (
                torch.tensor(active_edge_weights, dtype=torch.float64)
                if active_edge_weights is not None
                else None
            )

            for iteration in range(maxiter):
                temperature = maxdelta * (((maxiter - iteration) / float(maxiter)) ** coolexp)
                forces = torch.zeros((node_count, 2), dtype=torch.float64)
                maxchange = 0.0

                if active_edges:
                    source = torch.tensor([edge[0] for edge in active_edges], dtype=torch.long)
                    target = torch.tensor([edge[1] for edge in active_edges], dtype=torch.long)
                    delta = positions[source] - positions[target]
                    distance_matrix = torch.linalg.norm(delta, dim=1)
                    mask = distance_matrix > _LGL_MIN_DISTANCE
                    if bool(mask.any().item()):
                        masked_distance = distance_matrix[mask]
                        direction = delta[mask] / masked_distance.unsqueeze(1)
                        magnitude = masked_distance.square() / max(frk, _LGL_MIN_DISTANCE)
                        if weight_tensor is not None:
                            magnitude = magnitude * weight_tensor[mask]
                        contribution = direction * magnitude.unsqueeze(1)
                        forces.index_add_(0, source[mask], -contribution)
                        forces.index_add_(0, target[mask], contribution)

                buckets: dict[tuple[int, int], list[int]] = {}
                grid_steps = _lgl_grid_steps(radius=radius, cellsize=cellsize)
                safe_cell_size = max(cellsize, _LGL_MIN_DISTANCE)
                for node in refinement_nodes:
                    x_value = float(positions[node, 0].item())
                    y_value = float(positions[node, 1].item())
                    if self.config.fidelity_mode:
                        # igraph uses a finite 2D grid and clamps out-of-bounds
                        # coordinates into boundary cells before pair enumeration.
                        key = _lgl_clamped_grid_cell(
                            x_value=x_value,
                            y_value=y_value,
                            radius=radius,
                            cellsize=cellsize,
                            steps=grid_steps,
                        )
                    else:
                        key = (
                            int(math.floor(x_value / safe_cell_size)),
                            int(math.floor(y_value / safe_cell_size)),
                        )
                    buckets.setdefault(key, []).append(node)

                sorted_cells = sorted(buckets)
                for cell in sorted_cells:
                    nodes_here = buckets[cell]
                    for offset_x, offset_y in _LGL_BUCKET_NEIGHBOR_OFFSETS:
                        neighbor_cell = (cell[0] + offset_x, cell[1] + offset_y)
                        if neighbor_cell not in buckets:
                            continue

                        nodes_there = buckets[neighbor_cell]
                        if neighbor_cell == cell:
                            for left_index in range(len(nodes_here)):
                                for right_index in range(left_index + 1, len(nodes_here)):
                                    left = nodes_here[left_index]
                                    right = nodes_here[right_index]
                                    delta = positions[left] - positions[right]
                                    distance_value = float(torch.linalg.norm(delta).item())
                                    if distance_value >= cellsize:
                                        continue
                                    safe_distance = max(
                                        distance_value,
                                        _LGL_REPULSION_MIN_DISTANCE,
                                    )
                                    direction = delta / safe_distance
                                    magnitude = (frk * frk) * (
                                        (1.0 / safe_distance)
                                        - ((safe_distance * safe_distance) / repulserad)
                                    )
                                    contribution = direction * magnitude
                                    forces[left] += contribution
                                    forces[right] -= contribution
                        else:
                            for left in nodes_here:
                                for right in nodes_there:
                                    delta = positions[left] - positions[right]
                                    distance_value = float(torch.linalg.norm(delta).item())
                                    if distance_value >= cellsize:
                                        continue
                                    safe_distance = max(
                                        distance_value,
                                        _LGL_REPULSION_MIN_DISTANCE,
                                    )
                                    direction = delta / safe_distance
                                    magnitude = (frk * frk) * (
                                        (1.0 / safe_distance)
                                        - ((safe_distance * safe_distance) / repulserad)
                                    )
                                    contribution = direction * magnitude
                                    forces[left] += contribution
                                    forces[right] -= contribution

                for node in refinement_nodes:
                    movement = forces[node]
                    magnitude = float(torch.linalg.norm(movement).item())
                    if magnitude > temperature and magnitude > _LGL_MIN_DISTANCE:
                        movement = movement * (temperature / magnitude)
                    positions[node] += movement
                    maxchange = _lgl_updated_maxchange(
                        maxchange,
                        movement,
                        igraph_positive_only=self.config.igraph_positive_maxchange,
                    )

                if maxchange < self.config.convergence_epsilon:
                    break

        state.pos = positions
        return state


@register_op
@dataclass(frozen=True)
class LGLFinalizePositions(Op):
    """Cast positions to the classic LGL output dtype and output device."""

    output_dtype: torch.dtype = torch.float32
    output_scale: float = 1.0

    name: ClassVar[str] = "lgl_finalize_positions"
    category: ClassVar[OpCategory] = OpCategory.POSTPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Move, scale, and cast final layout coordinates.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable problem inputs used to choose the output device.
        state : SolveState
            Mutable state containing final LGL positions.
        ctx : RuntimeContext
            Runtime context (unused).

        Returns
        -------
        SolveState
            State with positions scaled and moved to the public output device.
        """

        del ctx

        if state.pos is None:
            raise ValueError("LGLFinalizePositions requires state.pos to be set.")

        output_device = layout_device(
            edge_index=problem.edge_index,
            node_sizes=problem.node_sizes,
        )
        state.pos = (state.pos * float(self.output_scale)).to(
            dtype=self.output_dtype,
            device=output_device,
        )
        return state


__all__ = [
    "LGLPrepareState",
    "LGLPrepareStateConfig",
    "LGLInitializePositions",
    "LGLLayeredRefinement",
    "LGLLayeredRefinementConfig",
    "LGLFinalizePositions",
    "_LGL_BUCKET_NEIGHBOR_OFFSETS",
    "_LGL_REPULSION_MIN_DISTANCE",
    "_build_lgl_bfs_layers",
    "_lgl_clamped_grid_cell",
    "_lgl_grid_steps",
    "_lgl_updated_maxchange",
]
