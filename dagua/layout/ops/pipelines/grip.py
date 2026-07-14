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
_REFERENCE_INITIAL_HEAT = _REFERENCE_EDGE_LENGTH // 6
_REFERENCE_FR_LEVELS = 1
_REFERENCE_LOCAL_PLACEMENT_ROUNDS = 3


@dataclass(frozen=True)
class _GripFiltrationState:
    """C-style GRIP filtration state.

    Parameters
    ----------
    order : list[int]
        Shared MISF order array used as prefixes for every level.
    sizes : list[int]
        Prefix size for each filtration level, finest to coarsest.
    vert_depth : list[int]
        Lowest level index at which each vertex appears.
    diameter : int
        Reference global ``diam`` value left by filtration construction.
    """

    order: list[int]
    sizes: list[int]
    vert_depth: list[int]
    diameter: int


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
    # GCC evaluates the four ``construct_Point`` arguments right-to-left in
    # the built reference, so x/y receive the fourth and third draws.
    _w_coord = int(rng.next() % box_span) - box_size
    _z_coord = int(rng.next() % box_span) - box_size
    y_coord = int(rng.next() % box_span) - box_size
    x_coord = int(rng.next() % box_span) - box_size
    return torch.tensor([x_coord, y_coord], dtype=dtype)


def _reference_random_point_values(diameter: int, rng: _GripFastRand) -> list[int]:
    """Return one integer GRIP random point.

    Parameters
    ----------
    diameter : int
        Reference ``diam`` global used to size the random box.
    rng : _GripFastRand
        Reference-compatible LCG.

    Returns
    -------
    list[int]
        Two integer coordinates. The z/w draws are still consumed to match C
        argument evaluation for ``construct_Point(..., dim=2)``.
    """
    box_size = int(_REFERENCE_EDGE_LENGTH * int(diameter) * 0.5)
    box_span = 2 * box_size + 1
    # The reference binary is built by GCC, which evaluates function arguments
    # right-to-left here: w, z, y, then x.
    _w_coord = int(rng.next() % box_span) - box_size
    _z_coord = int(rng.next() % box_span) - box_size
    y_coord = int(rng.next() % box_span) - box_size
    x_coord = int(rng.next() % box_span) - box_size
    return [x_coord, y_coord]


def _c_trunc(value: float) -> int:
    """Cast a floating-point value like C casts to ``int``.

    Parameters
    ----------
    value : float
        Floating-point value to convert.

    Returns
    -------
    int
        Value truncated toward zero.
    """
    return int(value)


def _c_round_l(value: float) -> int:
    """Round a non-negative value like GRIP's ``ROUND_L`` macro.

    Parameters
    ----------
    value : float
        Non-negative floating-point value.

    Returns
    -------
    int
        Rounded integer norm.
    """
    return int(value + 0.5) if value > 0.0 else -int(0.5 - value)


def _point_add(left: list[int], right: list[int]) -> list[int]:
    """Add two 2D integer points.

    Parameters
    ----------
    left : list[int]
        Left-hand point with shape ``[2]``.
    right : list[int]
        Right-hand point with shape ``[2]``.

    Returns
    -------
    list[int]
        Coordinate-wise sum.
    """
    return [left[0] + right[0], left[1] + right[1]]


def _point_sub(left: list[int], right: list[int]) -> list[int]:
    """Subtract two 2D integer points.

    Parameters
    ----------
    left : list[int]
        Left-hand point with shape ``[2]``.
    right : list[int]
        Right-hand point with shape ``[2]``.

    Returns
    -------
    list[int]
        Coordinate-wise difference.
    """
    return [left[0] - right[0], left[1] - right[1]]


def _point_div(point: list[int], divisor: int) -> list[int]:
    """Divide a 2D integer point using C integer truncation.

    Parameters
    ----------
    point : list[int]
        Point with shape ``[2]``.
    divisor : int
        Non-zero integer divisor.

    Returns
    -------
    list[int]
        Truncated coordinate-wise quotient.
    """
    return [_c_trunc(point[0] / divisor), _c_trunc(point[1] / divisor)]


def _point_scale(point: list[int], scalar: float) -> list[int]:
    """Scale a 2D integer point like ``fpoint_mult_eq``.

    Parameters
    ----------
    point : list[int]
        Point with shape ``[2]``.
    scalar : float
        Floating-point scale factor.

    Returns
    -------
    list[int]
        Scaled point with C-style truncation.
    """
    return [_c_trunc(point[0] * scalar), _c_trunc(point[1] * scalar)]


def _point_norm2(point: list[int]) -> int:
    """Return the integer squared norm of a 2D point.

    Parameters
    ----------
    point : list[int]
        Point with shape ``[2]``.

    Returns
    -------
    int
        Squared Euclidean norm.
    """
    return point[0] * point[0] + point[1] * point[1]


def _point_norm(point: list[int]) -> int:
    """Return GRIP's rounded integer norm for a 2D point.

    Parameters
    ----------
    point : list[int]
        Point with shape ``[2]``.

    Returns
    -------
    int
        Rounded Euclidean norm.
    """
    return _c_round_l(math.sqrt(float(_point_norm2(point))))


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
    color, _final_depth = _c_bfs_processed_with_depth(
        adjacency=adjacency,
        root=root,
        depth_limit=depth_limit,
    )
    return color


def _c_bfs_processed_with_depth(
    adjacency: list[list[int]],
    root: int,
    depth_limit: int,
) -> tuple[list[int], int]:
    """Return C BFS colors and the final ``currDepth`` value.

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
    tuple[list[int], int]
        Color array and the final depth used to update the global diameter.
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
    return color, current_depth


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
    state = _build_reference_mis_state(adjacency=adjacency, coarsest_size=coarsest_size)
    return [state.order[:size] for size in state.sizes]


def _build_reference_mis_state(
    adjacency: list[list[int]],
    coarsest_size: int,
) -> _GripFiltrationState:
    """Build GRIP's C-style MIS filtration state.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency lists in input order.
    coarsest_size : int
        Number of initial vertices retained at the top level.

    Returns
    -------
    _GripFiltrationState
        Shared MISF order, prefix sizes, vertex depths, and final diameter.
    """
    num_nodes = len(adjacency)
    if num_nodes == 0:
        return _GripFiltrationState(order=[], sizes=[0], vert_depth=[], diameter=0)
    max_levels = int(math.log2(float(num_nodes))) + 2
    misf = _c_order_by_degree(adjacency)
    vert_depth = [0 for _ in range(num_nodes)]
    sizes: list[int] = [num_nodes]
    previous_size = num_nodes
    diameter = int(math.sqrt(float(num_nodes)))
    for level in range(1, max_levels):
        marked = [-1 for _ in range(previous_size + 1)]
        kept = 0
        depth_limit = 2**level
        diameter = 0
        for index in range(previous_size):
            if marked[index + 1] != -1:
                continue
            processed, final_depth = _c_bfs_processed_with_depth(
                adjacency=adjacency,
                root=misf[index],
                depth_limit=depth_limit,
            )
            diameter = max(diameter, final_depth)
            marked[index + 1] = 1
            kept += 1
            for later in range(index + 1, previous_size):
                if processed[misf[later]] == 1:
                    marked[later + 1] = 0
        write_index = 0
        for marked_index in range(1, previous_size + 1):
            if marked[marked_index] == 1:
                source_index = marked_index - 1
                vert_depth[misf[source_index]] = level
                misf[write_index], misf[source_index] = misf[source_index], misf[write_index]
                write_index += 1
        effective_size = kept
        if effective_size < coarsest_size + 1:
            effective_size = min(coarsest_size, num_nodes)
            for index in range(kept, effective_size):
                vert_depth[misf[index]] = level
            sizes.append(effective_size)
            break
        sizes.append(effective_size)
        previous_size = effective_size
    if sizes[-1] > coarsest_size:
        for index in range(coarsest_size, sizes[-1]):
            vert_depth[misf[index]] = len(sizes) - 2
        sizes[-1] = coarsest_size
    return _GripFiltrationState(
        order=misf,
        sizes=sizes,
        vert_depth=vert_depth,
        diameter=diameter,
    )


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
        Accepted for API compatibility. The built headless reference uses the
        deterministic BFS filtration for this mode, so the value is ignored.
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
    del seed
    adjacency = _build_ordered_undirected_adjacency(edge_index=edge_index, num_nodes=num_nodes)
    return _build_reference_mis_filtration(adjacency=adjacency, coarsest_size=coarsest_size)


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


def _reference_neighbor_counts(
    adjacency: list[list[int]],
    sizes: Sequence[int],
    coarsest_size: int,
) -> list[int]:
    """Compute GRIP's ``nbr[]`` array for local beautification.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency lists in input order.
    sizes : sequence[int]
        MISF prefix sizes for every level.
    coarsest_size : int
        Number of initial vertices used by GRIP.

    Returns
    -------
    list[int]
        Neighbor-pair counts per filtration level.
    """
    if not sizes:
        return []
    num_nodes = len(adjacency)
    if num_nodes == 0:
        return [0 for _size in sizes]
    avg_degree_sum = float(sum(len(neighbors) for neighbors in adjacency))
    max_complexity = int(avg_degree_sum)
    initial_complexity = 10000
    if max_complexity < initial_complexity:
        max_complexity = initial_complexity
    small_level = 0
    for index, size in enumerate(sizes):
        if float(size) * float(size) - float(initial_complexity) <= 0.0:
            small_level = index
            break
    counts: list[int] = []
    for index, size in enumerate(sizes):
        if index >= small_level:
            count = max(size - 1, coarsest_size - 1)
        else:
            scheduled = _reference_sched(
                float(index),
                maximum=0.0,
                maximum_value=2.0,
                minimum=10000.0,
                minimum_value=1.0,
            )
            count = min(int(scheduled * max_complexity / max(size, 1)), max(size - 1, 0))
        counts.append(max(int(count), 0))
    counts[0] = min(2 * counts[0], max(num_nodes - 1, 0))
    return counts


def _reference_sched(
    value: float,
    maximum: float,
    maximum_value: float,
    minimum: float,
    minimum_value: float,
) -> float:
    """Evaluate GRIP's linear schedule helper.

    Parameters
    ----------
    value : float
        Schedule input.
    maximum : float
        Upper anchor in the original C parameter naming.
    maximum_value : float
        Value returned at or below ``maximum``.
    minimum : float
        Lower anchor in the original C parameter naming.
    minimum_value : float
        Value returned at or above ``minimum``.

    Returns
    -------
    float
        Scheduled value.
    """
    if value <= maximum:
        return maximum_value
    if maximum <= value <= minimum:
        return ((minimum_value - maximum_value) / (minimum - maximum)) * (
            value - maximum
        ) + maximum_value
    return minimum_value


def _reference_nbr_bfs(
    adjacency: list[list[int]],
    root: int,
    neighbor_counts: Sequence[int],
    vert_depth: Sequence[int],
    neighbor_cache: dict[int, list[list[tuple[int, int]]]],
) -> list[tuple[int, int]]:
    """Populate C-style local-neighbor arrays for one root.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency lists in input order.
    root : int
        Vertex whose neighbor arrays are being created.
    neighbor_counts : sequence[int]
        GRIP ``nbr[]`` values per level.
    vert_depth : sequence[int]
        Lowest level index at which each vertex appears.
    neighbor_cache : dict[int, list[list[tuple[int, int]]]]
        Mutable cache receiving ``nbrs[root][level]`` entries.

    Returns
    -------
    list[tuple[int, int]]
        Up to three closest higher-depth vertices as ``(vertex, distance)``.
    """
    root_depth = int(vert_depth[root])
    layers: list[list[tuple[int, int]]] = [[] for _level in range(root_depth + 1)]
    neighbor_cache[root] = layers
    color = [0 for _node in adjacency]
    color[root] = 1
    queue: deque[tuple[int, int]] = deque()
    current_depth = 1
    vertex = int(root)
    bottom_layer = 0
    close_vertices: list[tuple[int, int]] = []
    while True:
        color[vertex] = 1
        for adjacent in adjacency[vertex]:
            if color[adjacent] != 0:
                continue
            color[adjacent] = -1
            queue.append((adjacent, current_depth))
            upper_layer = min(int(vert_depth[adjacent]), root_depth)
            for layer in range(bottom_layer, upper_layer + 1):
                if len(layers[layer]) < int(neighbor_counts[layer]):
                    layers[layer].append((adjacent, current_depth))
                else:
                    bottom_layer = layer + 1
            if len(close_vertices) < 3 and int(vert_depth[adjacent]) > root_depth:
                close_vertices.append((adjacent, current_depth))
        if not queue:
            break
        vertex, queued_depth = queue.popleft()
        current_depth = queued_depth + 1
        if len(close_vertices) >= 3 and bottom_layer > root_depth:
            break
    return close_vertices


def _kk_spring_local(
    vertex: int,
    close_vertices: Sequence[tuple[int, int]],
    positions: Sequence[list[int]],
) -> tuple[list[int], int]:
    """Compute GRIP's short KK displacement for one new vertex.

    Parameters
    ----------
    vertex : int
        Vertex being adjusted.
    close_vertices : sequence[tuple[int, int]]
        Three closest anchors as ``(vertex, graph_distance)``.
    positions : sequence[list[int]]
        Integer positions with shape ``[N, 2]``.

    Returns
    -------
    tuple[list[int], int]
        Integer displacement and rounded norm.
    """
    displacement = [0, 0]
    for neighbor, graph_distance_value in close_vertices:
        vector = _point_sub(positions[neighbor], positions[vertex])
        norm2 = _point_norm2(vector)
        factor = (
            float(norm2)
            / float(
                graph_distance_value
                * graph_distance_value
                * _REFERENCE_EDGE_LENGTH
                * _REFERENCE_EDGE_LENGTH
            )
            - 1.0
        )
        displacement = _point_add(displacement, _point_scale(vector, factor))
    return displacement, _point_norm(displacement)


def _kk_spring(
    vertex: int,
    neighbors: Sequence[tuple[int, int]],
    positions: Sequence[list[int]],
) -> tuple[list[int], int]:
    """Compute GRIP's level KK displacement for one vertex.

    Parameters
    ----------
    vertex : int
        Vertex being refined.
    neighbors : sequence[tuple[int, int]]
        Local neighbor entries as ``(vertex, graph_distance)``.
    positions : sequence[list[int]]
        Integer positions with shape ``[N, 2]``.

    Returns
    -------
    tuple[list[int], int]
        Edge-normalized integer displacement and rounded norm.
    """
    displacement = [0, 0]
    for neighbor, graph_distance_value in neighbors:
        if graph_distance_value == 0:
            continue
        vector = _point_sub(positions[neighbor], positions[vertex])
        norm2 = _point_norm2(vector)
        factor = (
            float(norm2)
            / float(
                graph_distance_value
                * graph_distance_value
                * _REFERENCE_EDGE_LENGTH
                * _REFERENCE_EDGE_LENGTH
            )
            - 1.0
        )
        displacement = _point_add(displacement, _point_scale(vector, factor))
    norm_value = math.sqrt(float(_point_norm2(displacement)))
    disp_norm = _c_round_l(norm_value)
    if disp_norm:
        displacement = _point_scale(displacement, float(_REFERENCE_EDGE_LENGTH) / norm_value)
        disp_norm = _point_norm(displacement)
    return displacement, disp_norm


def _fr_spring(
    vertex: int,
    neighbors: Sequence[tuple[int, int]],
    adjacency: list[list[int]],
    positions: Sequence[list[int]],
) -> tuple[list[int], int]:
    """Compute GRIP's localized Fruchterman-Reingold displacement.

    Parameters
    ----------
    vertex : int
        Vertex being refined.
    neighbors : sequence[tuple[int, int]]
        Repulsive local-neighbor entries as ``(vertex, graph_distance)``.
    adjacency : list[list[int]]
        Undirected adjacency lists in input order.
    positions : sequence[list[int]]
        Integer positions with shape ``[N, 2]``.

    Returns
    -------
    tuple[list[int], int]
        Edge-normalized integer displacement and rounded norm.
    """
    displacement = [0, 0]
    edge2 = float(_REFERENCE_EDGE_LENGTH * _REFERENCE_EDGE_LENGTH)
    for adjacent in adjacency[vertex]:
        vector = _point_sub(positions[adjacent], positions[vertex])
        norm2 = float(_point_norm2(vector))
        displacement = _point_add(displacement, _point_scale(vector, norm2 / edge2))
    repulsive_edge2 = 0.05 * edge2
    for neighbor, _graph_distance_value in neighbors:
        vector = _point_sub(positions[vertex], positions[neighbor])
        norm2 = float(_point_norm2(vector))
        if norm2 == 0.0:
            vector = [1, 1]
            norm2 = 0.01
        displacement = _point_add(displacement, _point_scale(vector, repulsive_edge2 / norm2))
    norm_value = math.sqrt(float(_point_norm2(displacement)))
    disp_norm = _c_round_l(norm_value)
    if disp_norm:
        displacement = _point_scale(displacement, float(_REFERENCE_EDGE_LENGTH) / norm_value)
        disp_norm = _point_norm(displacement)
    return displacement, disp_norm


def _update_local_temperature(
    vertex: int,
    displacement: Sequence[int],
    disp_norm: int,
    old_displacement: Sequence[list[int]],
    old_disp_norm: Sequence[int],
    heat: list[int],
    old_cos: list[float],
) -> None:
    """Update one vertex's GRIP local temperature in place.

    Parameters
    ----------
    vertex : int
        Vertex whose temperature is updated.
    displacement : sequence[int]
        Current displacement with shape ``[2]``.
    disp_norm : int
        Rounded norm of ``displacement``.
    old_displacement : sequence[list[int]]
        Previous displacements with shape ``[N, 2]``.
    old_disp_norm : sequence[int]
        Previous rounded displacement norms.
    heat : list[int]
        Mutable integer temperature array.
    old_cos : list[float]
        Mutable previous-cosine array.

    Returns
    -------
    None
        ``heat`` and ``old_cos`` are updated in place.
    """
    norm_old = int(old_disp_norm[vertex])
    norm_new = int(disp_norm)
    if norm_old == 0 or norm_new == 0:
        return
    scalar_product = (
        displacement[0] * old_displacement[vertex][0]
        + displacement[1] * old_displacement[vertex][1]
    )
    cosine = float(scalar_product) / float(norm_old * norm_new)
    r_value = 0.15
    s_value = 3.0
    temp = int(heat[vertex])
    if old_cos[vertex] * cosine > 0.0:
        temp += _c_trunc(temp * s_value * cosine * r_value)
    else:
        temp += _c_trunc(temp * cosine * r_value)
    old_cos[vertex] = cosine
    heat[vertex] = temp


def _scale_displacement_for_move(displacement: list[int], disp_norm: int, heat: int) -> list[int]:
    """Apply GRIP heat and norm division to a displacement.

    Parameters
    ----------
    displacement : list[int]
        Displacement with shape ``[2]``.
    disp_norm : int
        Rounded displacement norm.
    heat : int
        Vertex temperature.

    Returns
    -------
    list[int]
        Move vector with shape ``[2]``.
    """
    move = [displacement[0] * heat, displacement[1] * heat]
    if disp_norm:
        move = _point_div(move, disp_norm)
    return move


def _run_reference_grip_component(
    adjacency: list[list[int]],
    config: GripConfig,
) -> list[list[int]]:
    """Run the reference-shaped GRIP engine for one connected component.

    Parameters
    ----------
    adjacency : list[list[int]]
        Component adjacency lists in input order with local node ids.
    config : GripConfig
        Pipeline configuration.

    Returns
    -------
    list[list[int]]
        Raw integer positions with shape ``[N, 2]``.
    """
    num_nodes = len(adjacency)
    if num_nodes == 0:
        return []
    state = _build_reference_mis_state(
        adjacency=adjacency,
        coarsest_size=min(config.coarsest_size, num_nodes),
    )
    positions = [[0, 0] for _node in range(num_nodes)]
    displacement = [[0, 0] for _node in range(num_nodes)]
    old_displacement = [[0, 0] for _node in range(num_nodes)]
    disp_norm = [0 for _node in range(num_nodes)]
    old_disp_norm = [1 for _node in range(num_nodes)]
    heat = [_REFERENCE_INITIAL_HEAT for _node in range(num_nodes)]
    old_cos = [1.0 for _node in range(num_nodes)]
    neighbor_counts = _reference_neighbor_counts(
        adjacency=adjacency,
        sizes=state.sizes,
        coarsest_size=min(config.coarsest_size, num_nodes),
    )
    neighbor_cache: dict[int, list[list[tuple[int, int]]]] = {}
    rng = _GripFastRand(0)
    rounds = int(config.rounds)
    coarsest_vertices = state.order[: state.sizes[-1]]
    if num_nodes >= 5 and len({len(neighbors) for neighbors in adjacency}) > 1:
        # The C reference allocates oldDispNorm without initialization. On the
        # built headless binary, irregular small components consistently leave
        # the max coarsest vertex at zero and the other coarsest slots nonzero;
        # this controls the first cosine update and is observable in fidelity.
        old_disp_norm[max(coarsest_vertices)] = 0
    for level_index in range(len(state.sizes) - 1, -1, -1):
        level_size = int(state.sizes[level_index])
        if level_index == len(state.sizes) - 1:
            barycenter = [0, 0]
            for order_index in range(level_size):
                vertex = state.order[order_index]
                positions[vertex] = _reference_random_point_values(
                    diameter=state.diameter,
                    rng=rng,
                )
                _reference_nbr_bfs(
                    adjacency=adjacency,
                    root=vertex,
                    neighbor_counts=neighbor_counts,
                    vert_depth=state.vert_depth,
                    neighbor_cache=neighbor_cache,
                )
                barycenter = _point_add(barycenter, positions[vertex])
            barycenter = _point_div(barycenter, 3)
            for order_index in range(level_size):
                vertex = state.order[order_index]
                positions[vertex] = _point_sub(positions[vertex], barycenter)
        if level_index < len(state.sizes) - 1:
            previous_size = int(state.sizes[level_index + 1])
            for order_index in range(previous_size):
                heat[state.order[order_index]] = _REFERENCE_INITIAL_HEAT
            for order_index in range(previous_size, level_size):
                vertex = state.order[order_index]
                close_vertices = _reference_nbr_bfs(
                    adjacency=adjacency,
                    root=vertex,
                    neighbor_counts=neighbor_counts,
                    vert_depth=state.vert_depth,
                    neighbor_cache=neighbor_cache,
                )
                anchors = close_vertices[:3]
                if anchors:
                    total = [0, 0]
                    old_total = [0, 0]
                    for anchor, _distance in anchors:
                        total = _point_add(total, positions[anchor])
                        old_total = _point_add(old_total, old_displacement[anchor])
                    positions[vertex] = _point_div(total, 3)
                    old_displacement[vertex] = _point_div(old_total, 3)
                    old_disp_norm[vertex] = _point_norm(old_displacement[vertex])
                for _round_index in range(_REFERENCE_LOCAL_PLACEMENT_ROUNDS):
                    current_disp, current_norm = _kk_spring_local(
                        vertex=vertex,
                        close_vertices=anchors,
                        positions=positions,
                    )
                    _update_local_temperature(
                        vertex=vertex,
                        displacement=current_disp,
                        disp_norm=current_norm,
                        old_displacement=old_displacement,
                        old_disp_norm=old_disp_norm,
                        heat=heat,
                        old_cos=old_cos,
                    )
                    old_displacement[vertex] = list(current_disp)
                    old_disp_norm[vertex] = current_norm
                    move = _scale_displacement_for_move(
                        displacement=current_disp,
                        disp_norm=current_norm,
                        heat=heat[vertex],
                    )
                    positions[vertex] = _point_add(positions[vertex], move)
        if rounds <= 0:
            continue
        for _round_index in range(rounds):
            moves: list[tuple[int, list[int]]] = []
            for order_index in range(level_size):
                vertex = state.order[order_index]
                neighbors = neighbor_cache[vertex][level_index]
                if level_index < _REFERENCE_FR_LEVELS:
                    current_disp, current_norm = _fr_spring(
                        vertex=vertex,
                        neighbors=neighbors,
                        adjacency=adjacency,
                        positions=positions,
                    )
                else:
                    current_disp, current_norm = _kk_spring(
                        vertex=vertex,
                        neighbors=neighbors,
                        positions=positions,
                    )
                _update_local_temperature(
                    vertex=vertex,
                    displacement=current_disp,
                    disp_norm=current_norm,
                    old_displacement=old_displacement,
                    old_disp_norm=old_disp_norm,
                    heat=heat,
                    old_cos=old_cos,
                )
                old_displacement[vertex] = list(current_disp)
                old_disp_norm[vertex] = current_norm
                displacement[vertex] = _scale_displacement_for_move(
                    displacement=current_disp,
                    disp_norm=current_norm,
                    heat=heat[vertex],
                )
                disp_norm[vertex] = current_norm
                moves.append((vertex, list(displacement[vertex])))
            for vertex, move in moves:
                positions[vertex] = _point_add(positions[vertex], move)
        if level_size == num_nodes:
            break
    return positions


def _connected_components_from_adjacency(adjacency: list[list[int]]) -> list[list[int]]:
    """Return sorted undirected connected components.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency lists.

    Returns
    -------
    list[list[int]]
        Components in ascending first-node order, with sorted node ids.
    """
    components: list[list[int]] = []
    seen: set[int] = set()
    for start in range(len(adjacency)):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in sorted(adjacency[node], reverse=True):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def _component_ordered_adjacency(
    edge_index: torch.Tensor,
    nodes: Sequence[int],
) -> list[list[int]]:
    """Build ordered local adjacency for one component.

    Parameters
    ----------
    edge_index : torch.Tensor
        Full graph edge tensor with shape ``[2, E]``.
    nodes : sequence[int]
        Sorted global component node ids.

    Returns
    -------
    list[list[int]]
        Local adjacency preserving full edge insertion order.
    """
    local_index = {int(node): index for index, node in enumerate(nodes)}
    local_edges: list[tuple[int, int]] = []
    for source, target in edge_index.detach().to(device="cpu", dtype=torch.long).t().tolist():
        source_index = int(source)
        target_index = int(target)
        if source_index == target_index:
            continue
        if source_index in local_index and target_index in local_index:
            local_edges.append((local_index[source_index], local_index[target_index]))
    if not local_edges:
        return [[] for _node in nodes]
    local_edge_index = torch.tensor(local_edges, dtype=torch.long).t()
    return _build_ordered_undirected_adjacency(
        edge_index=local_edge_index,
        num_nodes=len(nodes),
    )


def _run_reference_grip_graph(
    edge_index: torch.Tensor,
    adjacency: list[list[int]],
    config: GripConfig,
) -> torch.Tensor:
    """Run reference-shaped GRIP with reference component composition.

    Parameters
    ----------
    edge_index : torch.Tensor
        Full graph edge tensor with shape ``[2, E]``.
    adjacency : list[list[int]]
        Full graph ordered adjacency.
    config : GripConfig
        Pipeline configuration.

    Returns
    -------
    torch.Tensor
        Raw component-composed positions with shape ``[N, 2]``.
    """
    positions = torch.zeros((len(adjacency), 2), dtype=config.fidelity_dtype)
    offset = 0.0
    for nodes in _connected_components_from_adjacency(adjacency):
        if len(nodes) == 1:
            positions[nodes[0], 0] = offset
            offset += float(_REFERENCE_EDGE_LENGTH)
            continue
        component_adjacency = _component_ordered_adjacency(edge_index=edge_index, nodes=nodes)
        component_positions = _run_reference_grip_component(
            adjacency=component_adjacency,
            config=config,
        )
        for local_node, graph_node in enumerate(nodes):
            positions[graph_node] = torch.tensor(
                component_positions[local_node],
                dtype=config.fidelity_dtype,
            )
        min_x = float(torch.min(positions[nodes, 0]).item())
        max_x = float(torch.max(positions[nodes, 0]).item())
        positions[nodes, 0] += offset - min_x
        offset += (max_x - min_x) + float(_REFERENCE_EDGE_LENGTH)
    return positions


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
        filtration = _build_reference_mis_state(
            adjacency=adjacency,
            coarsest_size=self.config.coarsest_size,
        )
        state.extras["grip_adjacency"] = adjacency
        state.extras["grip_filtration"] = filtration
        state.extras["grip_levels"] = [filtration.order[:size] for size in filtration.sizes]
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
        positions = torch.zeros((problem.num_nodes, 2), dtype=self.config.fidelity_dtype)
        state.pos = positions.to(device=device)
        state.extras["grip_placement_order"] = list(range(problem.num_nodes))
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
        positions = _run_reference_grip_graph(
            edge_index=problem.edge_index,
            adjacency=adjacency,
            config=self.config,
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
