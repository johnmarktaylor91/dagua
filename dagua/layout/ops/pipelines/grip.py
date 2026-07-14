"""GRIP multilevel layout pipeline without runtime delegation."""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_DEFAULT_COARSEST_SIZE = 4
_DEFAULT_NEIGHBOR_FACTOR = 1.0
_DEFAULT_MIN_NEIGHBORS = 3
_DEFAULT_OUTPUT_SCALE = 50.0
_REFERENCE_EDGE_LENGTH = 32
_EPSILON = 1.0e-9
_U64_MASK = 0xFFFFFFFFFFFFFFFF


@dataclass(frozen=True)
class GripConfig:
    """Configuration for the clean-room GRIP pipeline.

    Parameters
    ----------
    rounds : int, default=12
        Number of local refinement rounds per filtration level.
    coarsest_size : int, default=4
        Target size for the coarsest level. The headless GRIP reference uses
        four initial vertices by default.
    neighbor_factor : float, default=1.0
        Multiplier for the paper's ``avg_degree * N / |V_i|`` neighborhood
        schedule.
    min_neighbors : int, default=3
        Lower bound for local neighborhoods and intelligent placement anchors.
    output_scale : float, default=50.0
        Final display scale after centering and unit-extent normalization.
    fidelity_dtype : torch.dtype, default=torch.float32
        Floating dtype used for deterministic internal calculations.
    """

    rounds: int = 12
    coarsest_size: int = _DEFAULT_COARSEST_SIZE
    neighbor_factor: float = _DEFAULT_NEIGHBOR_FACTOR
    min_neighbors: int = _DEFAULT_MIN_NEIGHBORS
    output_scale: float = _DEFAULT_OUTPUT_SCALE
    fidelity_dtype: torch.dtype = torch.float32


class _GripFastRand:
    """Port of GRIP's small linear congruential generator."""

    def __init__(self, seed: int) -> None:
        """Initialize the generator.

        Parameters
        ----------
        seed : int
            Unsigned seed. The reference resets this stream to zero when the
            MISF engine starts, so layout placement uses ``0`` for fidelity.
        """
        self._state = int(seed) & _U64_MASK

    def next(self) -> int:
        """Return the next GRIP ``fast_Rand`` value.

        Returns
        -------
        int
            Unsigned 64-bit generator state.
        """
        self._state = (1664525 * self._state + 1013904223) & _U64_MASK
        return self._state


def _reference_random_point(num_nodes: int, rng: _GripFastRand, dtype: torch.dtype) -> torch.Tensor:
    """Return one GRIP reference initial point.

    Parameters
    ----------
    num_nodes : int
        Component node count used to derive the reference ``diam`` value.
    rng : _GripFastRand
        Reference-compatible LCG.
    dtype : torch.dtype
        Output dtype.

    Returns
    -------
    torch.Tensor
        Integer-valued point with shape ``[2]``.
    """
    diameter = int(math.sqrt(float(max(num_nodes, 0))))
    box_size = int(_REFERENCE_EDGE_LENGTH * diameter * 0.5)
    box_span = 2 * box_size + 1
    x_coord = int(rng.next() % box_span) - box_size
    y_coord = int(rng.next() % box_span) - box_size
    # The C call evaluates z/w random arguments even when ``dim == 2``.
    rng.next()
    rng.next()
    return torch.tensor([x_coord, y_coord], dtype=dtype)


def _validate_grip_inputs(edge_index: torch.Tensor, num_nodes: int, config: GripConfig) -> None:
    """Validate public GRIP pipeline inputs.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    config : GripConfig
        Resolved pipeline configuration.

    Returns
    -------
    None
        Raises on invalid inputs.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")
    if edge_index.numel() > 0:
        min_index = int(edge_index.min().item())
        max_index = int(edge_index.max().item())
        if min_index < 0 or max_index >= num_nodes:
            raise ValueError("edge_index contains a node index outside [0, num_nodes).")
    if config.rounds < 0:
        raise ValueError("rounds must be non-negative.")
    if config.coarsest_size < 1:
        raise ValueError("coarsest_size must be positive.")
    if config.neighbor_factor <= 0.0:
        raise ValueError("neighbor_factor must be positive.")
    if config.min_neighbors < 1:
        raise ValueError("min_neighbors must be positive.")
    if config.output_scale <= 0.0:
        raise ValueError("output_scale must be positive.")


def _build_undirected_adjacency(edge_index: torch.Tensor, num_nodes: int) -> list[list[int]]:
    """Build sorted undirected adjacency lists.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    list[list[int]]
        Sorted neighbor lists for every node.
    """
    neighbors: list[set[int]] = [set() for _ in range(num_nodes)]
    for source, target in edge_index.detach().to(device="cpu", dtype=torch.long).t().tolist():
        source_index = int(source)
        target_index = int(target)
        if source_index == target_index:
            continue
        neighbors[source_index].add(target_index)
        neighbors[target_index].add(source_index)
    return [sorted(node_neighbors) for node_neighbors in neighbors]


def _build_ordered_undirected_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> list[list[int]]:
    """Build undirected adjacency lists preserving edge-file insertion order.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    list[list[int]]
        Neighbor lists in the order used by the GRIP headless reference.
    """
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    for source, target in edge_index.detach().to(device="cpu", dtype=torch.long).t().tolist():
        source_index = int(source)
        target_index = int(target)
        if source_index == target_index:
            continue
        adjacency[source_index].append(target_index)
        adjacency[target_index].append(source_index)
    return adjacency


def _nodes_within_radius(adjacency: list[list[int]], start: int, radius: int) -> set[int]:
    """Return nodes within a graph-distance radius.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency lists.
    start : int
        BFS start node.
    radius : int
        Inclusive maximum graph distance.

    Returns
    -------
    set[int]
        Nodes reached within ``radius`` hops, including ``start``.
    """
    reached = {int(start)}
    queue: deque[tuple[int, int]] = deque([(int(start), 0)])
    while queue:
        node, distance = queue.popleft()
        if distance >= radius:
            continue
        for neighbor in adjacency[node]:
            if neighbor in reached:
                continue
            reached.add(neighbor)
            queue.append((neighbor, distance + 1))
    return reached


def _shortest_distances_from(
    adjacency: list[list[int]],
    start: int,
    allowed: Optional[set[int]] = None,
) -> dict[int, int]:
    """Compute unweighted BFS distances from one node.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency lists.
    start : int
        BFS start node.
    allowed : set[int], optional
        Optional set restricting traversed and returned vertices.

    Returns
    -------
    dict[int, int]
        Mapping from reached node to hop distance.
    """
    if allowed is not None and start not in allowed:
        return {}
    distances = {int(start): 0}
    queue: deque[int] = deque([int(start)])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if allowed is not None and neighbor not in allowed:
                continue
            if neighbor in distances:
                continue
            distances[neighbor] = distances[node] + 1
            queue.append(neighbor)
    return distances


def _graph_distance(adjacency: list[list[int]], source: int, target: int) -> float:
    """Return graph distance between two nodes.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency lists.
    source : int
        Source node.
    target : int
        Target node.

    Returns
    -------
    float
        Shortest-path distance, or ``1.0`` for disconnected anchor fallback.
    """
    if source == target:
        return 0.0
    distances = _shortest_distances_from(adjacency=adjacency, start=source)
    return float(distances.get(target, 1))


def _greedy_mis_next_level(
    adjacency: list[list[int]],
    candidates: Sequence[int],
    radius: int,
    rng: random.Random,
) -> list[int]:
    """Construct one GRIP maximal independent set filtration step.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency lists.
    candidates : sequence[int]
        Current level ``V_i``.
    radius : int
        Exclusion radius ``2**i`` from the paper.
    rng : random.Random
        Seeded Python RNG used for the random draw order.

    Returns
    -------
    list[int]
        Next level ``V_{i+1}``, sorted for stable downstream processing.
    """
    remaining = set(int(node) for node in candidates)
    selected: list[int] = []
    while remaining:
        ordered = sorted(remaining)
        chosen = ordered[rng.randrange(len(ordered))]
        selected.append(chosen)
        remaining.difference_update(
            node for node in _nodes_within_radius(adjacency, chosen, radius) if node in remaining
        )
    return sorted(selected)


def _c_bfs_processed(adjacency: list[list[int]], root: int, depth_limit: int) -> list[int]:
    """Return the C reference BFS color array for one root.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency lists in input order.
    root : int
        BFS root.
    depth_limit : int
        C ``depthLim`` value.

    Returns
    -------
    list[int]
        Color array where processed vertices are marked ``1``.
    """
    color = [0 for _ in adjacency]
    queue: deque[tuple[int, int]] = deque()
    current_depth = 1
    vertex = int(root)
    while current_depth <= depth_limit + 1:
        color[vertex] = 1
        for adjacent in adjacency[vertex]:
            if color[adjacent] == 0:
                color[adjacent] = -1
                queue.append((adjacent, current_depth))
        if not queue:
            break
        vertex, queued_depth = queue.popleft()
        current_depth = queued_depth + 1
    return color


def _c_order_by_degree(adjacency: list[list[int]]) -> list[int]:
    """Order vertices like GRIP's ``order_by_deg`` helper.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency lists in input order.

    Returns
    -------
    list[int]
        Vertices ordered by the reference degree bucket layout.
    """
    num_nodes = len(adjacency)
    if num_nodes == 0:
        return []
    degrees = [len(neighbors) for neighbors in adjacency]
    max_degree = max(degrees)
    num_degree = [0 for _ in range(max_degree + 1)]
    processed = [0 for _ in range(max_degree + 1)]
    for degree in degrees:
        num_degree[degree] += 1
    offset = [0 for _ in range(max_degree + 1)]
    if max_degree >= 1:
        offset[1] = 0
    for degree in range(2, max_degree + 1):
        offset[degree] = offset[degree - 1] + num_degree[degree - 1]
    offset[0] = offset[max_degree] + num_degree[max_degree]
    ordered = [0 for _ in range(num_nodes)]
    for vertex, degree in enumerate(degrees):
        position = offset[degree] + processed[degree]
        ordered[position] = vertex
        processed[degree] += 1
    return ordered


def _build_reference_mis_filtration(
    adjacency: list[list[int]],
    coarsest_size: int,
) -> list[list[int]]:
    """Build GRIP's C-style MIS filtration prefixes.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency lists in input order.
    coarsest_size : int
        Number of initial vertices retained at the top level.

    Returns
    -------
    list[list[int]]
        Prefix levels from finest to coarsest.
    """
    num_nodes = len(adjacency)
    if num_nodes == 0:
        return [[]]
    max_levels = int(math.log2(float(num_nodes))) + 2
    misf = _c_order_by_degree(adjacency)
    levels: list[list[int]] = [misf[:num_nodes]]
    previous_size = num_nodes
    for level in range(1, max_levels):
        marked = [-1 for _ in range(previous_size + 1)]
        kept = 0
        depth_limit = 2**level
        for index in range(previous_size):
            if marked[index + 1] != -1:
                continue
            processed = _c_bfs_processed(adjacency, misf[index], depth_limit)
            marked[index + 1] = 1
            kept += 1
            for later in range(index + 1, previous_size):
                if processed[misf[later]] == 1:
                    marked[later + 1] = 0
        write_index = 0
        for marked_index in range(1, previous_size + 1):
            if marked[marked_index] == 1:
                source_index = marked_index - 1
                misf[write_index], misf[source_index] = misf[source_index], misf[write_index]
                write_index += 1
        effective_size = kept
        if effective_size < coarsest_size + 1:
            effective_size = min(coarsest_size, num_nodes)
            levels.append(misf[:effective_size])
            break
        levels.append(misf[:effective_size])
        previous_size = effective_size
    if len(levels[-1]) > coarsest_size:
        levels[-1] = levels[-1][:coarsest_size]
    return levels


def build_mis_filtration(
    edge_index: torch.Tensor,
    num_nodes: int,
    seed: int = 42,
    coarsest_size: int = _DEFAULT_COARSEST_SIZE,
) -> list[list[int]]:
    """Build the GRIP maximal independent set filtration.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    seed : int, default=42
        Seed controlling random candidate draws.
    coarsest_size : int, default=3
        Desired upper bound for the coarsest level when the graph topology
        permits further coarsening.

    Returns
    -------
    list[list[int]]
        Filtration ``[V_0, V_1, ..., V_k]`` with nested node-index levels.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if coarsest_size < 1:
        raise ValueError("coarsest_size must be positive.")
    adjacency = _build_undirected_adjacency(edge_index=edge_index, num_nodes=num_nodes)
    levels: list[list[int]] = [list(range(num_nodes))]
    rng = random.Random(seed)
    level_index = 0
    while len(levels[-1]) > coarsest_size:
        radius = 2**level_index
        next_level = _greedy_mis_next_level(
            adjacency=adjacency,
            candidates=levels[-1],
            radius=radius,
            rng=rng,
        )
        if not next_level or len(next_level) >= len(levels[-1]):
            break
        levels.append(next_level)
        level_index += 1
    return levels


def _circle_intersections(
    center_a: torch.Tensor,
    radius_a: float,
    center_b: torch.Tensor,
    radius_b: float,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    """Solve the two-circle placement subproblem.

    Parameters
    ----------
    center_a : torch.Tensor
        First anchor position with shape ``[2]``.
    radius_a : float
        Desired distance to the first anchor.
    center_b : torch.Tensor
        Second anchor position with shape ``[2]``.
    radius_b : float
        Desired distance to the second anchor.
    dtype : torch.dtype
        Output tensor dtype.

    Returns
    -------
    list[torch.Tensor]
        Zero, one, or two candidate positions with shape ``[2]``.
    """
    delta = center_b - center_a
    distance = float(torch.linalg.norm(delta).item())
    if distance <= _EPSILON:
        return []
    unit = delta / distance
    along = (radius_a * radius_a - radius_b * radius_b + distance * distance) / (2.0 * distance)
    height_sq = radius_a * radius_a - along * along
    midpoint = center_a + along * unit
    if height_sq < -_EPSILON:
        return []
    if abs(height_sq) <= _EPSILON:
        return [midpoint.to(dtype=dtype)]
    height = math.sqrt(max(height_sq, 0.0))
    perpendicular = torch.tensor([-unit[1].item(), unit[0].item()], dtype=dtype)
    return [
        (midpoint + height * perpendicular).to(dtype=dtype),
        (midpoint - height * perpendicular).to(dtype=dtype),
    ]


def _least_squares_place(
    anchors: list[tuple[int, float]],
    positions: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Place a vertex by linearized trilateration fallback.

    Parameters
    ----------
    anchors : list[tuple[int, float]]
        Anchor node indices and desired graph distances.
    positions : torch.Tensor
        Current position tensor with shape ``[N, 2]``.
    dtype : torch.dtype
        Output dtype.

    Returns
    -------
    torch.Tensor
        Estimated position with shape ``[2]``.
    """
    if not anchors:
        return torch.zeros(2, dtype=dtype)
    if len(anchors) == 1:
        anchor, radius = anchors[0]
        return positions[anchor].to(dtype=dtype) + torch.tensor([radius, 0.0], dtype=dtype)
    first_node, first_radius = anchors[0]
    first_pos = positions[first_node].to(dtype=dtype)
    rows: list[list[float]] = []
    rhs: list[float] = []
    for node, radius in anchors[1:]:
        anchor_pos = positions[node].to(dtype=dtype)
        rows.append((2.0 * (anchor_pos - first_pos)).tolist())
        rhs.append(
            float(
                first_radius * first_radius
                - radius * radius
                - torch.dot(first_pos, first_pos).item()
                + torch.dot(anchor_pos, anchor_pos).item()
            )
        )
    matrix = torch.tensor(rows, dtype=dtype)
    target = torch.tensor(rhs, dtype=dtype)
    solution = torch.linalg.lstsq(matrix, target).solution
    return solution.to(dtype=dtype)


def intelligent_initial_position(
    vertex: int,
    placed: Sequence[int],
    adjacency: list[list[int]],
    positions: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Place one new vertex from its already placed graph-nearest anchors.

    Parameters
    ----------
    vertex : int
        Node to place.
    placed : sequence[int]
        Already placed node indices.
    adjacency : list[list[int]]
        Undirected adjacency lists.
    positions : torch.Tensor
        Existing position tensor with shape ``[N, 2]``.
    dtype : torch.dtype, default=torch.float32
        Output dtype.

    Returns
    -------
    torch.Tensor
        Initial position for ``vertex`` with shape ``[2]``.
    """
    placed_set = set(int(node) for node in placed)
    distances = _shortest_distances_from(adjacency=adjacency, start=int(vertex))
    anchors = sorted(
        (
            (node, float(distances[node]))
            for node in placed_set
            if node in distances and distances[node] > 0
        ),
        key=lambda item: (item[1], item[0]),
    )[:3]
    if len(anchors) < 2:
        return _least_squares_place(anchors=anchors, positions=positions, dtype=dtype)

    candidates: list[torch.Tensor] = []
    for left_index in range(len(anchors)):
        for right_index in range(left_index + 1, len(anchors)):
            left_node, left_radius = anchors[left_index]
            right_node, right_radius = anchors[right_index]
            candidates.extend(
                _circle_intersections(
                    center_a=positions[left_node].to(dtype=dtype),
                    radius_a=left_radius,
                    center_b=positions[right_node].to(dtype=dtype),
                    radius_b=right_radius,
                    dtype=dtype,
                )
            )
    if len(candidates) >= 3:
        best_triplet = min(
            (
                (a, b, c)
                for a in range(len(candidates))
                for b in range(a + 1, len(candidates))
                for c in range(b + 1, len(candidates))
            ),
            key=lambda triplet: float(
                torch.linalg.norm(candidates[triplet[0]] - candidates[triplet[1]]).item()
                + torch.linalg.norm(candidates[triplet[0]] - candidates[triplet[2]]).item()
                + torch.linalg.norm(candidates[triplet[1]] - candidates[triplet[2]]).item()
            ),
        )
        return torch.stack([candidates[index] for index in best_triplet]).mean(dim=0)
    if candidates:
        return torch.stack(candidates).mean(dim=0)
    return _least_squares_place(anchors=anchors, positions=positions, dtype=dtype)


def _initialize_coarsest(
    coarsest: Sequence[int],
    adjacency: list[list[int]],
    positions: torch.Tensor,
    dtype: torch.dtype,
) -> list[int]:
    """Initialize the coarsest GRIP level.

    Parameters
    ----------
    coarsest : sequence[int]
        Coarsest filtration nodes.
    adjacency : list[list[int]]
        Undirected adjacency lists.
    positions : torch.Tensor
        Mutable position tensor with shape ``[N, 2]``.
    dtype : torch.dtype
        Internal floating dtype.

    Returns
    -------
    list[int]
        Placed nodes in deterministic placement order.
    """
    ordered = list(coarsest)
    if not ordered:
        return []
    rng = _GripFastRand(0)
    for vertex in ordered:
        positions[vertex] = _reference_random_point(
            num_nodes=positions.shape[0],
            rng=rng,
            dtype=dtype,
        )
    barycenter = torch.zeros(2, dtype=dtype)
    for vertex in ordered:
        barycenter += positions[vertex]
    barycenter = torch.trunc(barycenter / 3.0)
    for vertex in ordered:
        positions[vertex] -= barycenter
    return ordered


def _reference_barycenter_position(
    vertex: int,
    placed: Sequence[int],
    adjacency: list[list[int]],
    positions: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Place a vertex at the integer barycenter of three nearest placed nodes.

    Parameters
    ----------
    vertex : int
        Node to place.
    placed : sequence[int]
        Already placed node indices.
    adjacency : list[list[int]]
        Undirected adjacency lists.
    positions : torch.Tensor
        Existing position tensor with shape ``[N, 2]``.
    dtype : torch.dtype
        Output dtype.

    Returns
    -------
    torch.Tensor
        Initial position with shape ``[2]``.
    """
    placed_set = set(int(node) for node in placed)
    distances = _shortest_distances_from(adjacency=adjacency, start=int(vertex))
    anchors = sorted(
        (distances[node], node) for node in placed_set if node in distances and node != vertex
    )[:3]
    if not anchors:
        return torch.zeros(2, dtype=dtype)
    total = torch.zeros(2, dtype=dtype)
    for _distance, node in anchors:
        total += positions[node].to(dtype=dtype)
    divisor = max(len(anchors), 1)
    return torch.trunc(total / float(divisor)).to(dtype=dtype)


def _average_degree(adjacency: list[list[int]]) -> float:
    """Compute the undirected average degree.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency lists.

    Returns
    -------
    float
        Mean number of neighbors per node.
    """
    if not adjacency:
        return 0.0
    return float(sum(len(neighbors) for neighbors in adjacency)) / float(len(adjacency))


def _scheduled_neighbor_count(
    num_nodes: int,
    level_size: int,
    avg_degree: float,
    config: GripConfig,
) -> int:
    """Evaluate GRIP's local neighborhood schedule.

    Parameters
    ----------
    num_nodes : int
        Full graph node count.
    level_size : int
        Current filtration level size.
    avg_degree : float
        Undirected average degree of the graph.
    config : GripConfig
        Pipeline configuration.

    Returns
    -------
    int
        Local neighborhood cap for this level.
    """
    if level_size <= 1:
        return 0
    scheduled = math.ceil(config.neighbor_factor * max(avg_degree, 1.0) * num_nodes / level_size)
    return max(config.min_neighbors, min(level_size - 1, int(scheduled)))


def _nearest_level_neighbors(
    vertex: int,
    level_nodes: set[int],
    adjacency: list[list[int]],
    count: int,
) -> list[int]:
    """Find graph-nearest local refinement neighbors in a level.

    Parameters
    ----------
    vertex : int
        Query node.
    level_nodes : set[int]
        Nodes in the current filtration level.
    adjacency : list[list[int]]
        Undirected adjacency lists.
    count : int
        Maximum number of neighbors to return.

    Returns
    -------
    list[int]
        Neighbor node indices sorted by graph distance then index.
    """
    if count <= 0:
        return []
    distances = _shortest_distances_from(adjacency=adjacency, start=vertex)
    candidates = [
        (distances[node], node) for node in level_nodes if node != vertex and node in distances
    ]
    candidates.sort()
    if len(candidates) < count:
        missing = sorted(node for node in level_nodes if node != vertex and node not in distances)
        candidates.extend((10**9, node) for node in missing)
    return [node for _distance, node in candidates[:count]]


def _local_fr_refine(
    positions: torch.Tensor,
    level: Sequence[int],
    adjacency: list[list[int]],
    rounds: int,
    neighbor_count: int,
    use_fr: bool = True,
) -> None:
    """Refine one filtration level with GRIP-shaped local forces.

    Parameters
    ----------
    positions : torch.Tensor
        Mutable position tensor with shape ``[N, 2]``.
    level : sequence[int]
        Current filtration level nodes.
    adjacency : list[list[int]]
        Undirected adjacency lists.
    rounds : int
        Number of local refinement rounds.
    neighbor_count : int
        Number of graph-nearest nodes used for repulsion per vertex.
    use_fr : bool, default=True
        Whether to use the reference's local FR force. ``False`` uses its
        KK-style refinement for coarser levels.

    Returns
    -------
    None
        Updates ``positions`` in place.
    """
    if rounds <= 0 or len(level) <= 1:
        return
    level_nodes = set(int(node) for node in level)
    local_neighbors: dict[int, list[tuple[int, int]]] = {}
    for node in level:
        distances = _shortest_distances_from(adjacency=adjacency, start=int(node))
        candidates = [
            (distances[other], other)
            for other in level_nodes
            if other != node and other in distances
        ]
        candidates.sort()
        local_neighbors[int(node)] = candidates[:neighbor_count]
    dtype = positions.dtype
    heat = torch.full((positions.shape[0],), 5.0, dtype=dtype)
    for _round_index in range(rounds):
        displacement = torch.zeros_like(positions)
        for node in level:
            node_pos = positions[node]
            if use_fr:
                for neighbor in adjacency[node]:
                    delta = positions[neighbor] - node_pos
                    norm2 = torch.clamp(
                        torch.dot(delta, delta),
                        min=torch.tensor(0.01, dtype=dtype),
                    )
                    displacement[node] += delta * (
                        norm2 / float(_REFERENCE_EDGE_LENGTH * _REFERENCE_EDGE_LENGTH)
                    )
                for neighbor, _graph_distance_value in local_neighbors[node]:
                    delta = node_pos - positions[neighbor]
                    norm2 = torch.clamp(
                        torch.dot(delta, delta),
                        min=torch.tensor(0.01, dtype=dtype),
                    )
                    displacement[node] += delta * (
                        0.05 * float(_REFERENCE_EDGE_LENGTH * _REFERENCE_EDGE_LENGTH) / norm2
                    )
            else:
                for neighbor, graph_distance_value in local_neighbors[node]:
                    if graph_distance_value <= 0:
                        continue
                    delta = positions[neighbor] - node_pos
                    norm2 = torch.dot(delta, delta)
                    ideal2 = float(
                        graph_distance_value
                        * graph_distance_value
                        * _REFERENCE_EDGE_LENGTH
                        * _REFERENCE_EDGE_LENGTH
                    )
                    displacement[node] += delta * (norm2 / ideal2 - 1.0)
        for node in level:
            length = torch.linalg.norm(displacement[node])
            if float(length.item()) <= _EPSILON:
                continue
            unit_force = displacement[node] * (float(_REFERENCE_EDGE_LENGTH) / float(length.item()))
            unit_length = torch.linalg.norm(unit_force)
            if float(unit_length.item()) <= _EPSILON:
                continue
            step = unit_force * (heat[node] / unit_length)
            positions[node] += step.to(dtype=dtype)


def _finalize_positions(positions: torch.Tensor, output_scale: float) -> torch.Tensor:
    """Center and scale final GRIP coordinates.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    output_scale : float
        Positive display scale.

    Returns
    -------
    torch.Tensor
        Finalized position tensor with shape ``[N, 2]``.
    """
    if positions.numel() == 0:
        return positions
    centered = positions - positions.mean(dim=0, keepdim=True)
    extent = torch.max(torch.abs(centered))
    if float(extent.item()) <= _EPSILON:
        return centered
    return centered / extent * float(output_scale)


@register_op
class GripBuildFiltration(Op):
    """Build and store the GRIP MIS filtration."""

    name = "grip_build_mis_filtration"
    category = OpCategory.COARSEN
    reads = ("edge_index", "N")
    writes = ("extras",)

    def __init__(self, config: GripConfig) -> None:
        """Initialize the filtration op.

        Parameters
        ----------
        config : GripConfig
            Pipeline configuration.
        """
        self.config = config

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build the filtration for the current layout problem.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context; unused by this CPU topology step.

        Returns
        -------
        SolveState
            State with GRIP filtration metadata in ``extras``.
        """
        del ctx
        adjacency = _build_ordered_undirected_adjacency(
            edge_index=problem.edge_index,
            num_nodes=problem.num_nodes,
        )
        levels = _build_reference_mis_filtration(
            adjacency=adjacency,
            coarsest_size=self.config.coarsest_size,
        )
        state.extras["grip_adjacency"] = adjacency
        state.extras["grip_levels"] = levels
        return state


@register_op
class GripIntelligentPlacement(Op):
    """Place levels from coarsest to finest using GRIP initialization."""

    name = "grip_intelligent_placement"
    category = OpCategory.INIT
    reads = ("extras",)
    writes = ("pos", "extras")

    def __init__(self, config: GripConfig) -> None:
        """Initialize the placement op.

        Parameters
        ----------
        config : GripConfig
            Pipeline configuration.
        """
        self.config = config

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Place all vertices by traversing the MIS filtration.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context used for output device selection.

        Returns
        -------
        SolveState
            State with initialized positions and placement order metadata.
        """
        device = torch.device(ctx.plan.device)
        adjacency = state.extras["grip_adjacency"]
        levels = state.extras["grip_levels"]
        positions = torch.zeros((problem.num_nodes, 2), dtype=self.config.fidelity_dtype)
        placed = _initialize_coarsest(
            coarsest=levels[-1] if levels else [],
            adjacency=adjacency,
            positions=positions,
            dtype=self.config.fidelity_dtype,
        )
        placed_set = set(placed)
        placement_order = list(placed)
        for level_index in range(len(levels) - 2, -1, -1):
            finer = levels[level_index]
            coarser = set(levels[level_index + 1])
            new_vertices = [node for node in finer if node not in coarser]
            for vertex in new_vertices:
                positions[vertex] = _reference_barycenter_position(
                    vertex=vertex,
                    placed=sorted(placed_set),
                    adjacency=adjacency,
                    positions=positions,
                    dtype=self.config.fidelity_dtype,
                )
                placed_set.add(vertex)
                placement_order.append(vertex)
        state.pos = positions.to(device=device)
        state.extras["grip_placement_order"] = placement_order
        return state


@register_op
class GripLocalRefinement(Op):
    """Run per-level neighborhood-restricted FR refinement."""

    name = "grip_local_fr_refinement"
    category = OpCategory.FORCE
    reads = ("pos", "extras")
    writes = ("pos",)

    def __init__(self, config: GripConfig) -> None:
        """Initialize the local refinement op.

        Parameters
        ----------
        config : GripConfig
            Pipeline configuration.
        """
        self.config = config

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Refine each filtration level from coarse to fine.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state with initialized positions.
        ctx : RuntimeContext
            Runtime context used for output device restoration.

        Returns
        -------
        SolveState
            State with refined and finalized positions.

        Raises
        ------
        RuntimeError
            If placement did not populate ``state.pos``.
        """
        if state.pos is None:
            raise RuntimeError("GRIP refinement requires initialized positions.")
        adjacency = state.extras["grip_adjacency"]
        levels = state.extras["grip_levels"]
        positions = state.pos.detach().to(device="cpu", dtype=self.config.fidelity_dtype).clone()
        avg_degree = _average_degree(adjacency)
        for level_index in range(len(levels) - 1, -1, -1):
            level = levels[level_index]
            neighbor_count = _scheduled_neighbor_count(
                num_nodes=problem.num_nodes,
                level_size=len(level),
                avg_degree=avg_degree,
                config=self.config,
            )
            _local_fr_refine(
                positions=positions,
                level=level,
                adjacency=adjacency,
                rounds=self.config.rounds,
                neighbor_count=neighbor_count,
                use_fr=level_index < 1,
            )
        state.pos = _finalize_positions(
            positions=positions,
            output_scale=self.config.output_scale,
        ).to(device=torch.device(ctx.plan.device), dtype=self.config.fidelity_dtype)
        return state


def build_grip_pipeline(config: Optional[GripConfig] = None) -> Pipeline:
    """Build the GRIP operation pipeline.

    Parameters
    ----------
    config : GripConfig, optional
        Pipeline configuration. Defaults to :class:`GripConfig`.

    Returns
    -------
    Pipeline
        Composable GRIP pipeline with MIS filtration, intelligent placement,
        and local FR refinement stages.
    """
    resolved = GripConfig() if config is None else config
    return Pipeline(
        [
            GripBuildFiltration(config=resolved),
            GripIntelligentPlacement(config=resolved),
            GripLocalRefinement(config=resolved),
        ],
        name="grip_pipeline",
    )


def layout_grip_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 12,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    coarsest_size: int = _DEFAULT_COARSEST_SIZE,
    neighbor_factor: float = _DEFAULT_NEIGHBOR_FACTOR,
    min_neighbors: int = _DEFAULT_MIN_NEIGHBORS,
    output_scale: float = _DEFAULT_OUTPUT_SCALE,
    fidelity_dtype: torch.dtype = torch.float32,
    **kwargs: object,
) -> torch.Tensor:
    """Run the GRIP multilevel layout pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``; used only to resolve
        the output device when ``edge_index`` is empty.
    steps : int, default=12
        Number of local refinement rounds per filtration level.
    seed : int, default=42
        Seed controlling MIS candidate draw order.
    edge_weights : torch.Tensor, optional
        Accepted for dispatch compatibility. The paper formulation used here
        is unweighted, so weights are intentionally ignored.
    coarsest_size : int, default=3
        Target coarsest-level size.
    neighbor_factor : float, default=1.0
        Multiplier for the local-neighborhood schedule.
    min_neighbors : int, default=3
        Lower bound for local neighborhoods.
    output_scale : float, default=50.0
        Final display scale after normalization.
    fidelity_dtype : torch.dtype, default=torch.float32
        Internal and output dtype.
    **kwargs : object
        Extra dispatch keywords accepted for compatibility.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If graph or configuration inputs are invalid.
    RuntimeError
        If the composed pipeline does not populate positions.
    """
    del edge_weights, kwargs
    config = GripConfig(
        rounds=steps,
        coarsest_size=coarsest_size,
        neighbor_factor=neighbor_factor,
        min_neighbors=min_neighbors,
        output_scale=output_scale,
        fidelity_dtype=fidelity_dtype,
    )
    _validate_grip_inputs(edge_index=edge_index, num_nodes=num_nodes, config=config)
    output_device = (
        edge_index.device
        if edge_index.numel() > 0
        else node_sizes.device
        if node_sizes is not None
        else torch.device("cpu")
    )
    problem = LayoutProblem(
        edge_index=edge_index.to(device="cpu", dtype=torch.long),
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        seed=seed,
    )
    final_state = build_grip_pipeline(config=config).apply(
        problem,
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device=str(output_device))),
    )
    if final_state.pos is None:
        raise RuntimeError("GRIP pipeline did not produce final positions.")
    return final_state.pos.to(device=output_device, dtype=fidelity_dtype)


__all__ = [
    "GripBuildFiltration",
    "GripConfig",
    "GripIntelligentPlacement",
    "GripLocalRefinement",
    "build_grip_pipeline",
    "build_mis_filtration",
    "intelligent_initial_position",
    "layout_grip_pipeline",
]
