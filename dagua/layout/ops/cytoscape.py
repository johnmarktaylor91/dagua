"""Cytoscape-family layout operations.

The operations in this module are native Python/PyTorch ports of the small
deterministic parts of Cytoscape's layout family plus a legacy CoSE-compatible
spring step. Reference adapters are intentionally not imported here.
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
_CISE_DEFAULT_NODE_SEPARATION = 12.5
_CISE_CLUSTER_MARGIN = 15.0
_CISE_IDEAL_INTER_CLUSTER_EDGE_LENGTH = 70.0
_CISE_DEFAULT_NODE_DIMENSION = 30.0
_CISE_ROTATION_EPSILON = 1.0e-9
_CISE_DEFAULT_EDGE_LENGTH = 50.0
_CISE_DEFAULT_SPRING_STRENGTH = 0.675
_CISE_DEFAULT_REPULSION_STRENGTH = 4500.0
_CISE_MIN_REPULSION_DISTANCE = 5.0
_CISE_MAX_NODE_DISPLACEMENT = 300.0
_CISE_CONVERGENCE_CHECK_PERIOD = 100
_CISE_MAX_ROTATION_ANGLE = math.pi / 36.0
_CISE_REVERSE_PERIOD = 25
_CISE_SWAP_IDLE_DURATION = 45
_CISE_SWAP_PREPARATION_DURATION = 5
_CISE_SWAP_PERIOD = _CISE_SWAP_IDLE_DURATION + _CISE_SWAP_PREPARATION_DURATION
_CISE_SWAP_HISTORY_CLEARANCE_PERIOD = 6 * _CISE_SWAP_PERIOD
_CISE_MIN_DISPLACEMENT_FOR_SWAP = 6.0
_CISE_CLUSTER_ENLARGEMENT_CHECK_PERIOD = 50
_CISE_DEFAULT_INNER_EDGE_LENGTH = _CISE_DEFAULT_EDGE_LENGTH / 3.0
_COSE_DEFAULT_NODE_WIDTH = 1.0
_COSE_DEFAULT_NODE_HEIGHT = 1.0
_COSE_DEFAULT_RENDERED_NODE_CENTER = 15.0
_CYTOSCAPE_LCG_MULTIPLIER = 1664525
_CYTOSCAPE_LCG_INCREMENT = 1013904223
_CYTOSCAPE_LCG_MODULUS = 4294967296


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


def _cytoscape_random(state: SolveState, seed: int) -> float:
    """Return the next value from the verifier's seeded Cytoscape RNG.

    Parameters
    ----------
    state : SolveState
        Mutable solve state used to persist the JavaScript LCG state.
    seed : int
        Initial seed supplied to the Cytoscape reference adapter.

    Returns
    -------
    float
        Pseudorandom value in ``[0, 1)`` matching the Node reference adapter.
    """
    raw_state = state.extras.get("cytoscape_random_state")
    if raw_state is None:
        raw_state = int(seed) & 0xFFFFFFFF
        if raw_state == 0:
            raw_state = 1
    next_state = (
        _CYTOSCAPE_LCG_MULTIPLIER * int(raw_state) + _CYTOSCAPE_LCG_INCREMENT
    ) & 0xFFFFFFFF
    state.extras["cytoscape_random_state"] = next_state
    return next_state / _CYTOSCAPE_LCG_MODULUS


def _best_cise_inter_cluster_rotation(
    group: list[int],
    group_index: int,
    group_index_by_node: dict[int, int],
    centers: list[tuple[float, float]],
    local_pos: torch.Tensor,
    edge_index: torch.Tensor,
) -> float:
    """Return the CiSE rotation angle induced by inter-cluster edges.

    Parameters
    ----------
    group : list[int]
        Global node ids in one CiSE circle.
    group_index : int
        Index of ``group`` within the cluster list.
    group_index_by_node : dict[int, int]
        Mapping from global node id to cluster-circle index.
    centers : list[tuple[float, float]]
        Circle-center coordinates before local member offsets are applied.
    local_pos : torch.Tensor
        Current node coordinates with shape ``[N, 2]`` before cluster centers
        are added.
    edge_index : torch.Tensor
        Global edge tensor with shape ``[2, E]``.

    Returns
    -------
    float
        Rotation angle in radians. Positive angles are counter-clockwise in
        the native coordinate system.
    """
    if len(group) < 2 or edge_index.numel() == 0:
        return 0.0
    center_x, center_y = centers[group_index]
    cos_sum = 0.0
    sin_sum = 0.0
    edges = edge_index.detach().cpu().long()
    for edge_pos in range(edges.shape[1]):
        source = int(edges[0, edge_pos].item())
        target = int(edges[1, edge_pos].item())
        source_group = group_index_by_node.get(source)
        target_group = group_index_by_node.get(target)
        if source_group is None or target_group is None or source_group == target_group:
            continue
        if source_group == group_index:
            local_node = source
            other_group = target_group
        elif target_group == group_index:
            local_node = target
            other_group = source_group
        else:
            continue
        local_x = float(local_pos[local_node, 0])
        local_y = float(local_pos[local_node, 1])
        local_norm = math.hypot(local_x, local_y)
        target_x = centers[other_group][0] - center_x
        target_y = centers[other_group][1] - center_y
        target_norm = math.hypot(target_x, target_y)
        if local_norm <= _CISE_ROTATION_EPSILON or target_norm <= _CISE_ROTATION_EPSILON:
            continue
        local_x /= local_norm
        local_y /= local_norm
        target_x /= target_norm
        target_y /= target_norm
        # This is the closed-form 2D orthogonal Procrustes rotation for unit
        # vectors.  It mirrors CiSE's force-driven rotation using all
        # inter-cluster edge endpoints, not a single arbitrary member.
        cos_sum += local_x * target_x + local_y * target_y
        sin_sum += local_x * target_y - local_y * target_x
    if abs(cos_sum) <= _CISE_ROTATION_EPSILON and abs(sin_sum) <= _CISE_ROTATION_EPSILON:
        return 0.0
    return math.atan2(sin_sum, cos_sum)


@dataclass
class _CiSECircleState:
    """Mutable state for one CiSE on-circle cluster.

    Parameters
    ----------
    members : list[int]
        Global node ids in current circle order.
    center : list[float]
        Parent-circle center as ``[x, y]``.
    radius : float
        Circle radius in layout units.
    angles : dict[int, float]
        Current member angle by global node id, in radians.
    additional_separation : float
        Extra node separation introduced by enlargement checks.
    may_be_reversed : bool
        Whether Step 3 may still reverse this circle.
    displacement_for_swap : dict[int, float]
        Step 4 accumulated tangential displacement by node id.
    can_swap_next : dict[int, bool]
        Whether a node may swap with its next circle neighbor.
    can_swap_prev : dict[int, bool]
        Whether a node may swap with its previous circle neighbor.
    """

    members: list[int]
    center: list[float]
    radius: float
    angles: dict[int, float]
    additional_separation: float
    may_be_reversed: bool
    displacement_for_swap: dict[int, float]
    can_swap_next: dict[int, bool]
    can_swap_prev: dict[int, bool]


@dataclass(frozen=True)
class _CiSESwapPair:
    """Candidate adjacent swap from CiSE Step 4.

    Parameters
    ----------
    first : int
        First node in current circle order.
    second : int
        Next node in current circle order.
    discrepancy : float
        Difference between accumulated tangential displacements.
    same_direction : bool
        Whether the two nodes pulled in the same circular direction.
    """

    first: int
    second: int
    discrepancy: float
    same_direction: bool


def _cise_angle_of_vector(start_x: float, start_y: float, end_x: float, end_y: float) -> float:
    """Return Cytoscape's positive ``[0, 2*pi)`` vector angle.

    Parameters
    ----------
    start_x : float
        Vector start x-coordinate.
    start_y : float
        Vector start y-coordinate.
    end_x : float
        Vector end x-coordinate.
    end_y : float
        Vector end y-coordinate.

    Returns
    -------
    float
        Angle in radians.
    """
    angle = math.atan2(end_y - start_y, end_x - start_x)
    if angle < 0.0:
        angle += 2.0 * math.pi
    return angle


def _cise_half_diagonal() -> float:
    """Return half of the default Cytoscape CiSE node diagonal.

    Returns
    -------
    float
        Half diagonal for a 30x30 reference node.
    """
    return math.sqrt(2.0 * _CISE_DEFAULT_NODE_DIMENSION * _CISE_DEFAULT_NODE_DIMENSION) / 2.0


def _cise_recompute_circle_geometry(circle: _CiSECircleState) -> None:
    """Recompute circle radius after a cluster enlargement.

    Parameters
    ----------
    circle : _CiSECircleState
        Mutable circle state.

    Returns
    -------
    None
        Updates ``circle.radius`` in place.
    """
    perimeter = len(circle.members) * (
        2.0 * _cise_half_diagonal() + _CISE_DEFAULT_NODE_SEPARATION + circle.additional_separation
    )
    circle.radius = perimeter / (2.0 * math.pi) if perimeter > 0.0 else 0.0


def _cise_recalculate_node_positions(circle: _CiSECircleState, pos: torch.Tensor) -> None:
    """Recalculate member angles and positions after order/size changes.

    Parameters
    ----------
    circle : _CiSECircleState
        Mutable circle state.
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    None
        Updates ``circle.angles`` and ``pos`` in place.
    """
    separation = _CISE_DEFAULT_NODE_SEPARATION + circle.additional_separation
    half_diagonal = _cise_half_diagonal()
    for order_index, node in enumerate(circle.members):
        if order_index == 0:
            angle = 0.0
        else:
            previous = circle.members[order_index - 1]
            angle = circle.angles[previous] + (2.0 * half_diagonal + separation) / max(
                circle.radius,
                _MIN_DISTANCE,
            )
        circle.angles[node] = angle % (2.0 * math.pi)
        pos[node, 0] = circle.center[0] + circle.radius * math.cos(circle.angles[node])
        pos[node, 1] = circle.center[1] + circle.radius * math.sin(circle.angles[node])


def _cise_circle_from_meta(
    group: list[int],
    meta: dict[str, object],
    pos: torch.Tensor,
) -> _CiSECircleState:
    """Build mutable CiSE circle state from Step 1 placement metadata.

    Parameters
    ----------
    group : list[int]
        Global node ids in AVSDF order.
    meta : dict[str, object]
        Circle metadata emitted by ``CytoscapeCircleClusters``.
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    _CiSECircleState
        Mutable state used by Steps 3-5.
    """
    center_x = float(meta["x"])
    center_y = float(meta["y"])
    radius = float(meta["r"])
    angles: dict[int, float] = {}
    for node in group:
        angles[node] = _cise_angle_of_vector(
            center_x,
            center_y,
            float(pos[node, 0]),
            float(pos[node, 1]),
        )
    return _CiSECircleState(
        members=list(group),
        center=[center_x, center_y],
        radius=radius,
        angles=angles,
        additional_separation=0.0,
        may_be_reversed=True,
        displacement_for_swap={node: 0.0 for node in group},
        can_swap_next={node: True for node in group},
        can_swap_prev={node: True for node in group},
    )


def _cise_member_group_maps(
    circles: list[_CiSECircleState],
) -> tuple[dict[int, int], dict[int, int]]:
    """Return node-to-circle and node-to-order-index maps.

    Parameters
    ----------
    circles : list[_CiSECircleState]
        Current circle states.

    Returns
    -------
    tuple[dict[int, int], dict[int, int]]
        ``(group_by_node, order_by_node)`` maps.
    """
    group_by_node: dict[int, int] = {}
    order_by_node: dict[int, int] = {}
    for group_index, circle in enumerate(circles):
        for order_index, node in enumerate(circle.members):
            group_by_node[node] = group_index
            order_by_node[node] = order_index
    return group_by_node, order_by_node


def _cise_next_member(circle: _CiSECircleState, node: int) -> int:
    """Return the next node in a circle.

    Parameters
    ----------
    circle : _CiSECircleState
        Circle state.
    node : int
        Node id.

    Returns
    -------
    int
        Next node id.
    """
    index = circle.members.index(node)
    return circle.members[(index + 1) % len(circle.members)]


def _cise_prev_member(circle: _CiSECircleState, node: int) -> int:
    """Return the previous node in a circle.

    Parameters
    ----------
    circle : _CiSECircleState
        Circle state.
    node : int
        Node id.

    Returns
    -------
    int
        Previous node id.
    """
    index = circle.members.index(node)
    return circle.members[(index - 1) % len(circle.members)]


def _cise_segment_intersects(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    """Return whether two line segments intersect.

    Parameters
    ----------
    a : tuple[float, float]
        First segment start.
    b : tuple[float, float]
        First segment end.
    c : tuple[float, float]
        Second segment start.
    d : tuple[float, float]
        Second segment end.

    Returns
    -------
    bool
        ``True`` when segments cross or touch.
    """

    def _orientation(
        p: tuple[float, float],
        q: tuple[float, float],
        r: tuple[float, float],
    ) -> float:
        """Return signed orientation for three points.

        Parameters
        ----------
        p : tuple[float, float]
            First point.
        q : tuple[float, float]
            Second point.
        r : tuple[float, float]
            Third point.

        Returns
        -------
        float
            Signed orientation value.
        """
        return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])

    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    return (o1 > 0.0) != (o2 > 0.0) and (o3 > 0.0) != (o4 > 0.0)


def _cise_edges(edge_index: torch.Tensor, num_nodes: int) -> list[tuple[int, int]]:
    """Return valid CiSE edges in encounter order.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    list[tuple[int, int]]
        Directed encounter-order edge pairs, with invalid/self edges removed.
    """
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
        result.append((source, target))
    return result


def _cise_char_for_index(index: int) -> str:
    """Return CiSE's index character code.

    Parameters
    ----------
    index : int
        Circle order index.

    Returns
    -------
    str
        Lowercase for 0-25, uppercase for 26-51, and ``?`` afterward.
    """
    if index < 26:
        return chr(97 + index)
    if index < 52:
        return chr(65 + index)
    return "?"


def _cise_alignment_score(first: list[str], second: list[str]) -> float:
    """Return Needleman-Wunsch score with Cytoscape CiSE weights.

    Parameters
    ----------
    first : list[str]
        First sequence.
    second : list[str]
        Second sequence.

    Returns
    -------
    float
        Global alignment score.
    """
    rows = len(first) + 1
    cols = len(second) + 1
    gap = -2.0
    matrix = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for row in range(1, rows):
        matrix[row][0] = matrix[row - 1][0] + gap
    for col in range(1, cols):
        matrix[0][col] = matrix[0][col - 1] + gap
    for row in range(1, rows):
        for col in range(1, cols):
            match = 20.0 if first[row - 1] == second[col - 1] else -1.0
            matrix[row][col] = max(
                matrix[row - 1][col - 1] + match,
                matrix[row - 1][col] + gap,
                matrix[row][col - 1] + gap,
            )
    return matrix[-1][-1]


def _cise_decompose_force(
    circle: _CiSECircleState,
    node: int,
    force_x: float,
    force_y: float,
    pos: torch.Tensor,
) -> tuple[float, float, float]:
    """Decompose an on-circle force into rotation and translation.

    Parameters
    ----------
    circle : _CiSECircleState
        Owner circle.
    node : int
        Node id.
    force_x : float
        X force component.
    force_y : float
        Y force component.
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    tuple[float, float, float]
        ``(rotation_amount, displacement_x, displacement_y)``.
    """
    if force_x == 0.0 and force_y == 0.0:
        return 0.0, 0.0, 0.0
    center_x, center_y = circle.center
    node_x = float(pos[node, 0])
    node_y = float(pos[node, 1])
    center_angle = _cise_angle_of_vector(center_x, center_y, node_x, node_y)
    force_angle = _cise_angle_of_vector(0.0, 0.0, force_x, force_y)
    reverse_angle = center_angle + math.pi
    if math.pi <= reverse_angle < 2.0 * math.pi:
        clockwise = center_angle <= force_angle < reverse_angle
    else:
        reverse_angle -= 2.0 * math.pi
        clockwise = not (reverse_angle <= force_angle < center_angle)
    rotation = abs(math.sin(abs(center_angle - force_angle)) * math.hypot(force_x, force_y))
    if not clockwise:
        rotation = -rotation
    return rotation, force_x, force_y


def _cise_calc_forces(
    circles: list[_CiSECircleState],
    edges: list[tuple[int, int]],
    group_by_node: dict[int, int],
    pos: torch.Tensor,
    cooling_factor: float,
    gravity: float,
    gravity_range: float,
    polish: bool,
) -> tuple[list[list[float]], list[float], dict[int, float]]:
    """Compute CiSE Step 3-5 parent translations and rotations.

    Parameters
    ----------
    circles : list[_CiSECircleState]
        Current circles.
    edges : list[tuple[int, int]]
        Valid graph edges.
    group_by_node : dict[int, int]
        Node-to-circle map.
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    cooling_factor : float
        Current reference cooling factor.
    gravity : float
        Root graph gravity strength.
    gravity_range : float
        Root graph gravity range multiplier.
    polish : bool
        Whether Step 5 ideal edge lengths are loosened.

    Returns
    -------
    tuple[list[list[float]], list[float], dict[int, float]]
        Parent force vectors, parent rotation amounts, and per-node rotation
        components used by Step 4 swap preparation.
    """
    parent_forces = [[0.0, 0.0] for _ in circles]
    parent_rotations = [0.0 for _ in circles]
    node_swap_rotation: dict[int, float] = {}
    ideal_length = _CISE_DEFAULT_EDGE_LENGTH * 1.4 * (1.5 if polish else 1.0)
    node_forces = {node: [0.0, 0.0] for circle in circles for node in circle.members}
    for source, target in edges:
        source_group = group_by_node.get(source)
        target_group = group_by_node.get(target)
        if source_group is None or target_group is None or source_group == target_group:
            continue
        delta_x = float(pos[target, 0] - pos[source, 0])
        delta_y = float(pos[target, 1] - pos[source, 1])
        length = math.hypot(delta_x, delta_y)
        if length == 0.0:
            continue
        spring_force = _CISE_DEFAULT_SPRING_STRENGTH * (length - ideal_length)
        force_x = spring_force * delta_x / length
        force_y = spring_force * delta_y / length
        node_forces[source][0] += force_x
        node_forces[source][1] += force_y
        node_forces[target][0] -= force_x
        node_forces[target][1] -= force_y
    for left_index, left in enumerate(circles):
        for right_index in range(left_index + 1, len(circles)):
            right = circles[right_index]
            delta_x = right.center[0] - left.center[0]
            delta_y = right.center[1] - left.center[1]
            if abs(delta_x) < _CISE_MIN_REPULSION_DISTANCE:
                delta_x = math.copysign(_CISE_MIN_REPULSION_DISTANCE, delta_x)
            if abs(delta_y) < _CISE_MIN_REPULSION_DISTANCE:
                delta_y = math.copysign(_CISE_MIN_REPULSION_DISTANCE, delta_y)
            distance_sq = delta_x * delta_x + delta_y * delta_y
            distance = math.sqrt(distance_sq)
            force = (
                _CISE_DEFAULT_REPULSION_STRENGTH
                * len(left.members)
                * len(right.members)
                / distance_sq
            )
            force_x = force * delta_x / distance
            force_y = force * delta_y / distance
            parent_forces[left_index][0] -= cooling_factor * force_x
            parent_forces[left_index][1] -= cooling_factor * force_y
            parent_forces[right_index][0] += cooling_factor * force_x
            parent_forces[right_index][1] += cooling_factor * force_y
    if len(circles) > 1:
        root_center_x = sum(circle.center[0] for circle in circles) / len(circles)
        root_center_y = sum(circle.center[1] for circle in circles) / len(circles)
        estimated_size = max(
            max(abs(circle.center[0] - root_center_x), abs(circle.center[1] - root_center_y))
            + 2.0 * circle.radius
            + _CISE_DEFAULT_NODE_DIMENSION
            for circle in circles
        )
        for circle_index, circle in enumerate(circles):
            distance_x = circle.center[0] - root_center_x
            distance_y = circle.center[1] - root_center_y
            abs_x = abs(distance_x) + circle.radius + _CISE_DEFAULT_NODE_DIMENSION / 2.0
            abs_y = abs(distance_y) + circle.radius + _CISE_DEFAULT_NODE_DIMENSION / 2.0
            if abs_x > estimated_size * gravity_range or abs_y > estimated_size * gravity_range:
                parent_forces[circle_index][0] += cooling_factor * -gravity * distance_x
                parent_forces[circle_index][1] += cooling_factor * -gravity * distance_y
    for node, (force_x, force_y) in node_forces.items():
        circle_index = group_by_node[node]
        rotation, displacement_x, displacement_y = _cise_decompose_force(
            circles[circle_index],
            node,
            cooling_factor * force_x,
            cooling_factor * force_y,
            pos,
        )
        node_swap_rotation[node] = rotation
        parent_forces[circle_index][0] += displacement_x
        parent_forces[circle_index][1] += displacement_y
        parent_rotations[circle_index] += rotation
    return parent_forces, parent_rotations, node_swap_rotation


def _cise_apply_motion(
    circles: list[_CiSECircleState],
    pos: torch.Tensor,
    parent_forces: list[list[float]],
    parent_rotations: list[float],
    cooling_factor: float,
) -> None:
    """Apply CiSE parent translation and rigid-circle rotation.

    Parameters
    ----------
    circles : list[_CiSECircleState]
        Current circles.
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    parent_forces : list[list[float]]
        Parent force vectors.
    parent_rotations : list[float]
        Parent rotation amounts.
    cooling_factor : float
        Current reference cooling factor.

    Returns
    -------
    None
        Updates ``circles`` and ``pos`` in place.
    """
    for circle_index, circle in enumerate(circles):
        count = max(len(circle.members), 1)
        dx = max(
            -_CISE_MAX_NODE_DISPLACEMENT,
            min(_CISE_MAX_NODE_DISPLACEMENT, parent_forces[circle_index][0]),
        )
        dy = max(
            -_CISE_MAX_NODE_DISPLACEMENT,
            min(_CISE_MAX_NODE_DISPLACEMENT, parent_forces[circle_index][1]),
        )
        dx = dx * cooling_factor / count
        dy = dy * cooling_factor / count
        circle.center[0] += dx
        circle.center[1] += dy
        for node in circle.members:
            pos[node, 0] += dx
            pos[node, 1] += dy
        rotation_amount = parent_rotations[circle_index]
        if rotation_amount != 0.0 and circle.radius > _MIN_DISTANCE:
            theta = rotation_amount / count / circle.radius
            theta = max(-_CISE_MAX_ROTATION_ANGLE, min(_CISE_MAX_ROTATION_ANGLE, theta))
            for node in circle.members:
                circle.angles[node] = (circle.angles[node] + theta) % (2.0 * math.pi)
                pos[node, 0] = circle.center[0] + circle.radius * math.cos(circle.angles[node])
                pos[node, 1] = circle.center[1] + circle.radius * math.sin(circle.angles[node])


def _cise_swap_nodes(circle: _CiSECircleState, first: int, second: int, pos: torch.Tensor) -> None:
    """Swap adjacent circle nodes using CiSE's angle update rule.

    Parameters
    ----------
    circle : _CiSECircleState
        Circle state.
    first : int
        First node id.
    second : int
        Second node id.
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    None
        Updates order, angles, and positions in place.
    """
    first_index = circle.members.index(first)
    second_index = circle.members.index(second)
    if first_index > second_index:
        first, second = second, first
        first_index, second_index = second_index, first_index
    if (first_index - 1) % len(circle.members) == second_index:
        first, second = second, first
        first_index, second_index = second_index, first_index
    circle.members[first_index], circle.members[second_index] = (
        circle.members[second_index],
        circle.members[first_index],
    )
    separation = _CISE_DEFAULT_NODE_SEPARATION + circle.additional_separation
    half_diagonal = _cise_half_diagonal()
    previous = circle.members[(first_index - 1) % len(circle.members)]
    circle.angles[second] = (
        circle.angles[previous]
        + (2.0 * half_diagonal + separation) / max(circle.radius, _MIN_DISTANCE)
    ) % (2.0 * math.pi)
    circle.angles[first] = (
        circle.angles[second]
        + (2.0 * half_diagonal + separation) / max(circle.radius, _MIN_DISTANCE)
    ) % (2.0 * math.pi)
    for node in (first, second):
        pos[node, 0] = circle.center[0] + circle.radius * math.cos(circle.angles[node])
        pos[node, 1] = circle.center[1] + circle.radius * math.sin(circle.angles[node])


def _cise_reverse_if_better(
    circle_index: int,
    circles: list[_CiSECircleState],
    edges: list[tuple[int, int]],
    group_by_node: dict[int, int],
    pos: torch.Tensor,
) -> bool:
    """Reverse one circle when the reference alignment heuristic improves.

    Parameters
    ----------
    circle_index : int
        Circle index.
    circles : list[_CiSECircleState]
        Current circles.
    edges : list[tuple[int, int]]
        Valid graph edges.
    group_by_node : dict[int, int]
        Node-to-circle map.
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    bool
        ``True`` when a reversal was performed.
    """
    circle = circles[circle_index]
    inter_edges: list[tuple[float, int, int]] = []
    degree_by_node = {node: 0 for node in circle.members}
    for source, target in edges:
        source_group = group_by_node.get(source)
        target_group = group_by_node.get(target)
        if source_group == target_group or (
            source_group != circle_index and target_group != circle_index
        ):
            continue
        this_end = source if source_group == circle_index else target
        other_end = target if this_end == source else source
        angle = _cise_angle_of_vector(
            circle.center[0],
            circle.center[1],
            float(pos[other_end, 0]),
            float(pos[other_end, 1]),
        )
        inter_edges.append((angle, this_end, other_end))
        degree_by_node[this_end] += 1
    if not circle.may_be_reversed or len(inter_edges) < 2 or len(circle.members) > 52:
        circle.may_be_reversed = False
        return False
    inter_edges.sort(key=lambda item: (item[0], circle.members.index(item[1])))
    repeated_count = len(circle.members) + sum(
        max(0, degree - 1) for degree in degree_by_node.values()
    )
    current_base: list[str] = []
    reversed_base: list[str] = [""] * repeated_count
    index = -1
    for order_index, node in enumerate(circle.members):
        degree = max(1, degree_by_node[node])
        char = _cise_char_for_index(order_index)
        for _ in range(degree):
            index += 1
            current_base.append(char)
            reversed_base[repeated_count - 1 - index] = char
    current = current_base + current_base
    reversed_sequence = reversed_base + reversed_base
    neighbor = [
        _cise_char_for_index(circle.members.index(this_end)) for _, this_end, _ in inter_edges
    ]
    if _cise_alignment_score(reversed_sequence, neighbor) > _cise_alignment_score(
        current,
        neighbor,
    ):
        circle.members = [circle.members[0], *reversed(circle.members[1:])]
        _cise_recalculate_node_positions(circle, pos)
        circle.may_be_reversed = False
        return True
    return False


def _cise_intersections_for_pair(
    first: int,
    second: int,
    edges: list[tuple[int, int]],
    group_by_node: dict[int, int],
    pos: torch.Tensor,
) -> int:
    """Count inter-cluster edge intersections for two on-circle nodes.

    Parameters
    ----------
    first : int
        First node id.
    second : int
        Second node id.
    edges : list[tuple[int, int]]
        Valid graph edges.
    group_by_node : dict[int, int]
        Node-to-circle map.
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    int
        Intersection count.
    """
    first_edges: list[tuple[int, int]] = []
    second_edges: list[tuple[int, int]] = []
    for source, target in edges:
        if group_by_node.get(source) == group_by_node.get(target):
            continue
        if source == first or target == first:
            first_edges.append((source, target))
        if source == second or target == second:
            second_edges.append((source, target))
    count = 0
    for first_edge in first_edges:
        first_other = first_edge[1] if first_edge[0] == first else first_edge[0]
        first_segment = (
            (float(pos[first, 0]), float(pos[first, 1])),
            (float(pos[first_other, 0]), float(pos[first_other, 1])),
        )
        for second_edge in second_edges:
            second_other = second_edge[1] if second_edge[0] == second else second_edge[0]
            if first_other == second_other:
                continue
            second_segment = (
                (float(pos[second, 0]), float(pos[second, 1])),
                (float(pos[second_other, 0]), float(pos[second_other, 1])),
            )
            if _cise_segment_intersects(
                first_segment[0],
                first_segment[1],
                second_segment[0],
                second_segment[1],
            ):
                count += 1
    return count


def _cise_update_swapping_conditions(
    circles: list[_CiSECircleState],
    edges: list[tuple[int, int]],
    group_by_node: dict[int, int],
) -> None:
    """Update Step 4 adjacent-swap crossing guards.

    Parameters
    ----------
    circles : list[_CiSECircleState]
        Current circles.
    edges : list[tuple[int, int]]
        Valid graph edges.
    group_by_node : dict[int, int]
        Node-to-circle map.

    Returns
    -------
    None
        Updates ``can_swap_next`` and ``can_swap_prev`` in place.
    """
    for circle_index, circle in enumerate(circles):
        intra_edges = [
            (source, target)
            for source, target in edges
            if group_by_node.get(source) == circle_index
            and group_by_node.get(target) == circle_index
        ]

        def _crossings(order: dict[int, int], node: int) -> int:
            """Return intra-cluster crossings incident with one node.

            Parameters
            ----------
            order : dict[int, int]
                Candidate order map.
            node : int
                Node id.

            Returns
            -------
            int
                Crossing count.
            """
            incident = [edge for edge in intra_edges if node in edge]
            others = [edge for edge in intra_edges if node not in edge]
            crossings = 0
            for edge_a in incident:
                a1, a2 = sorted((order[edge_a[0]], order[edge_a[1]]))
                for edge_b in others:
                    b1, b2 = sorted((order[edge_b[0]], order[edge_b[1]]))
                    if (a1 < b1 < a2) != (a1 < b2 < a2):
                        crossings += 1
            return crossings

        order = {node: index for index, node in enumerate(circle.members)}
        for node in circle.members:
            current = _crossings(order, node)
            next_node = _cise_next_member(circle, node)
            next_order = dict(order)
            next_order[node], next_order[next_node] = next_order[next_node], next_order[node]
            circle.can_swap_next[node] = _crossings(next_order, node) <= current
            prev_node = _cise_prev_member(circle, node)
            prev_order = dict(order)
            prev_order[node], prev_order[prev_node] = prev_order[prev_node], prev_order[node]
            circle.can_swap_prev[node] = _crossings(prev_order, node) <= current


def _cise_perform_swap_phase(
    circles: list[_CiSECircleState],
    edges: list[tuple[int, int]],
    group_by_node: dict[int, int],
    pos: torch.Tensor,
    swapped_history: list[_CiSESwapPair],
) -> list[_CiSESwapPair]:
    """Run CiSE Step 4 perform-swap phase.

    Parameters
    ----------
    circles : list[_CiSECircleState]
        Current circles.
    edges : list[tuple[int, int]]
        Valid graph edges.
    group_by_node : dict[int, int]
        Node-to-circle map.
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    swapped_history : list[_CiSESwapPair]
        Pairs blocked or swapped in the last phase.

    Returns
    -------
    list[_CiSESwapPair]
        New swap history.
    """
    history_keys = {tuple(sorted((pair.first, pair.second))) for pair in swapped_history}
    new_history: list[_CiSESwapPair] = []
    for circle in circles:
        swapped_nodes: set[int] = set()
        non_safe: list[_CiSESwapPair] = []
        safe: list[_CiSESwapPair] = []
        for first in list(circle.members):
            second = _cise_next_member(circle, first)
            if not circle.can_swap_next.get(first, True) or not circle.can_swap_prev.get(
                second,
                True,
            ):
                continue
            first_disp = circle.displacement_for_swap.get(first, 0.0)
            second_disp = circle.displacement_for_swap.get(second, 0.0)
            discrepancy = first_disp - second_disp
            if discrepancy < 0.0:
                continue
            same_direction = (first_disp > 0.0 and second_disp > 0.0) or (
                first_disp < 0.0 and second_disp < 0.0
            )
            pair = _CiSESwapPair(first, second, discrepancy, same_direction)
            if first_disp == 0.0 or second_disp == 0.0:
                safe.append(pair)
            else:
                non_safe.append(pair)
        non_safe.sort(key=lambda pair: pair.discrepancy)
        while non_safe:
            pair = non_safe.pop()
            key = tuple(sorted((pair.first, pair.second)))
            if key in history_keys:
                new_history.append(pair)
                continue
            before = _cise_intersections_for_pair(
                pair.first,
                pair.second,
                edges,
                group_by_node,
                pos,
            )
            _cise_swap_nodes(circle, pair.first, pair.second, pos)
            after = _cise_intersections_for_pair(pair.first, pair.second, edges, group_by_node, pos)
            rollback = after > before or (
                after == before
                and (pair.same_direction or pair.discrepancy < _CISE_MIN_DISPLACEMENT_FOR_SWAP)
            )
            if rollback:
                _cise_swap_nodes(circle, pair.first, pair.second, pos)
                continue
            swapped_nodes.update((pair.first, pair.second))
            new_history.append(pair)
            break
        for pair in safe:
            key = tuple(sorted((pair.first, pair.second)))
            if (
                pair.same_direction
                or pair.discrepancy < _CISE_MIN_DISPLACEMENT_FOR_SWAP
                or pair.first in swapped_nodes
                or pair.second in swapped_nodes
            ):
                continue
            if key not in history_keys:
                _cise_swap_nodes(circle, pair.first, pair.second, pos)
                swapped_nodes.update((pair.first, pair.second))
            new_history.append(pair)
        for node in circle.members:
            circle.displacement_for_swap[node] = 0.0
    _cise_update_swapping_conditions(circles, edges, group_by_node)
    return new_history


def _cise_enlargement_check(circles: list[_CiSECircleState], pos: torch.Tensor) -> None:
    """Run Step 5 cluster enlargement bookkeeping.

    Parameters
    ----------
    circles : list[_CiSECircleState]
        Current circles.
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    None
        Updates enlarged circles in place.
    """
    for circle in circles:
        if circle.additional_separation > 0.0:
            _cise_recompute_circle_geometry(circle)
            _cise_recalculate_node_positions(circle, pos)


def _clipping_point(
    node_center: torch.Tensor,
    node_size: torch.Tensor,
    direction_x: torch.Tensor,
    direction_y: torch.Tensor,
) -> torch.Tensor:
    """Return Cytoscape core CoSE's rectangle clipping point.

    Parameters
    ----------
    node_center : torch.Tensor
        Node center coordinate with shape ``[2]``.
    node_size : torch.Tensor
        Node size with shape ``[2]`` as ``[width, height]``.
    direction_x : torch.Tensor
        X component of the direction vector.
    direction_y : torch.Tensor
        Y component of the direction vector.

    Returns
    -------
    torch.Tensor
        Clipping point with shape ``[2]``.
    """
    x_coord = node_center[0]
    y_coord = node_center[1]
    width = torch.clamp(node_size[0], min=_COSE_DEFAULT_NODE_WIDTH)
    height = torch.clamp(node_size[1], min=_COSE_DEFAULT_NODE_HEIGHT)

    if float(direction_x.item()) == 0.0 and float(direction_y.item()) > 0.0:
        return torch.stack((x_coord, y_coord + height / 2.0))
    if float(direction_x.item()) == 0.0 and float(direction_y.item()) < 0.0:
        # Cytoscape's core CoSE source returns ``Y + H / 2`` for this case.
        return torch.stack((x_coord, y_coord + height / 2.0))

    direction_slope = direction_y / direction_x
    node_slope = height / width
    if (
        float(direction_x.item()) > 0.0
        and float(direction_slope.item()) >= float((-node_slope).item())
        and float(direction_slope.item()) <= float(node_slope.item())
    ):
        return torch.stack(
            (x_coord + width / 2.0, y_coord + width * direction_y / (2.0 * direction_x))
        )
    if (
        float(direction_x.item()) < 0.0
        and float(direction_slope.item()) >= float((-node_slope).item())
        and float(direction_slope.item()) <= float(node_slope.item())
    ):
        return torch.stack(
            (x_coord - width / 2.0, y_coord - width * direction_y / (2.0 * direction_x))
        )
    if float(direction_y.item()) > 0.0 and (
        float(direction_slope.item()) <= float((-node_slope).item())
        or float(direction_slope.item()) >= float(node_slope.item())
    ):
        return torch.stack(
            (x_coord + height * direction_x / (2.0 * direction_y), y_coord + height / 2.0)
        )
    if float(direction_y.item()) < 0.0 and (
        float(direction_slope.item()) <= float((-node_slope).item())
        or float(direction_slope.item()) >= float(node_slope.item())
    ):
        return torch.stack(
            (x_coord - height * direction_x / (2.0 * direction_y), y_coord - height / 2.0)
        )

    return torch.stack((x_coord, y_coord))


def _cose_bounds(
    state: SolveState,
    pos: torch.Tensor,
    sizes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return cached Cytoscape core CoSE node bounds.

    Parameters
    ----------
    state : SolveState
        Mutable solve state carrying bounds between force iterations.
    pos : torch.Tensor
        Node center coordinates with shape ``[N, 2]``.
    sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ``min_x``, ``max_x``, ``min_y``, and ``max_y`` vectors.
    """
    bounds = state.extras.get("cose_bounds")
    if bounds is not None:
        return bounds
    min_x = pos[:, 0] - sizes[:, 0] / 2.0
    max_x = pos[:, 0] + sizes[:, 0] / 2.0
    min_y = pos[:, 1] - sizes[:, 1] / 2.0
    max_y = pos[:, 1] + sizes[:, 1] / 2.0
    cached = (min_x, max_x, min_y, max_y)
    state.extras["cose_bounds"] = cached
    return cached


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
    node_separation: float = _CISE_DEFAULT_NODE_SEPARATION

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
        for cluster_id in sorted(problem.clusters):
            members = problem.clusters[cluster_id]
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
        group_index_by_node = {
            node: group_index for group_index, group in enumerate(groups) for node in group
        }
        sizes = torch.full(
            (problem.num_nodes, 2),
            _CISE_DEFAULT_NODE_DIMENSION,
            dtype=torch.float64,
        )
        pos = torch.zeros((problem.num_nodes, 2), dtype=torch.float64)
        cluster_meta: list[dict[str, float]] = []
        half_extents: list[float] = []
        for group_index, group in enumerate(groups):
            local_index_by_node = {node: index for index, node in enumerate(group)}
            local_edges: list[tuple[int, int]] = []
            edges = problem.edge_index.detach().cpu().long()
            for edge_pos in range(edges.shape[1]):
                source = int(edges[0, edge_pos].item())
                target = int(edges[1, edge_pos].item())
                if source in local_index_by_node and target in local_index_by_node:
                    local_edges.append((local_index_by_node[source], local_index_by_node[target]))
            if local_edges:
                local_edge_index = torch.tensor(local_edges, dtype=torch.long).t().contiguous()
            else:
                local_edge_index = torch.empty((2, 0), dtype=torch.long)
            local_problem = LayoutProblem(
                edge_index=local_edge_index,
                num_nodes=len(group),
                node_sizes=sizes[torch.tensor(group, dtype=torch.long)],
            )
            local_state = AVSDFLayoutOp(node_separation=self.node_separation).apply(
                local_problem,
                SolveState(),
                ctx,
            )
            if local_state.pos is None:
                raise RuntimeError("CiSE cluster AVSDF placement did not produce positions.")
            local_pos = local_state.pos.to(dtype=torch.float64)
            local_pos = local_pos - local_pos.mean(dim=0, keepdim=True)
            radius = float(torch.linalg.vector_norm(local_pos, dim=1).max().item())
            max_dimension = float(sizes[torch.tensor(group, dtype=torch.long)].max().item())
            half_extents.append(radius + _CISE_CLUSTER_MARGIN + max_dimension / 2.0)
            for local_index, node in enumerate(group):
                pos[node] = local_pos[local_index]
            cluster_meta.append({"x": 0.0, "y": 0.0, "r": radius, "members": list(group)})

        if len(groups) == 2:
            center_distance = (
                half_extents[0] + half_extents[1] + _CISE_IDEAL_INTER_CLUSTER_EDGE_LENGTH
            )
            centers = [(-center_distance / 2.0, 0.0), (center_distance / 2.0, 0.0)]
        else:
            spacing = max(
                _CISE_IDEAL_INTER_CLUSTER_EDGE_LENGTH,
                max(half_extents, default=self.node_separation) * 2.0,
            )
            outer_radius = max(
                spacing,
                len(groups) * spacing / (2.0 * math.pi),
            )
            centers = [
                (
                    outer_radius * math.cos(2.0 * math.pi * index / max(len(groups), 1)),
                    outer_radius * math.sin(2.0 * math.pi * index / max(len(groups), 1)),
                )
                for index in range(len(groups))
            ]

        neighbor_center_counts = [0] * len(groups)
        edges = problem.edge_index.detach().cpu().long()
        for edge_pos in range(edges.shape[1]):
            source = int(edges[0, edge_pos].item())
            target = int(edges[1, edge_pos].item())
            source_group = group_index_by_node.get(source)
            target_group = group_index_by_node.get(target)
            if source_group is None or target_group is None or source_group == target_group:
                continue
            neighbor_center_counts[source_group] += 1
            neighbor_center_counts[target_group] += 1

        for group_index, group in enumerate(groups):
            center_x, center_y = centers[group_index]
            if len(group) > 1 and neighbor_center_counts[group_index] > 0:
                rotation = _best_cise_inter_cluster_rotation(
                    group=group,
                    group_index=group_index,
                    group_index_by_node=group_index_by_node,
                    centers=centers,
                    local_pos=pos,
                    edge_index=problem.edge_index,
                )
                cos_rotation = math.cos(rotation)
                sin_rotation = math.sin(rotation)
                for node in group:
                    local_x = float(pos[node, 0])
                    local_y = float(pos[node, 1])
                    pos[node, 0] = local_x * cos_rotation - local_y * sin_rotation
                    pos[node, 1] = local_x * sin_rotation + local_y * cos_rotation
            for node in group:
                pos[node, 0] += center_x
                pos[node, 1] += center_y
            cluster_meta[group_index]["x"] = center_x
            cluster_meta[group_index]["y"] = center_y
        state.pos = pos.to(device=device, dtype=torch.float32)
        state.extras["cise_cluster_circles"] = cluster_meta
        state.extras["cise_cluster_groups"] = groups
        return state


@register_op
@dataclass(frozen=True)
class CytoscapeCiSERelax(Op):
    """Run Cytoscape CiSE rigid-circle relaxation phases.

    This ports the self-contained CiSE Steps 3-5 from
    ``cytoscape-cise/src/CiSE/CiSELayout.js``: rigid-circle spring
    relaxation with ``CircularForce`` rotation, Step 4 adjacent swaps,
    Step 3 reversal checks, and Step 5 polishing/enlargement periods.
    """

    name: ClassVar[str] = "cytoscape_cise_relax"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[tuple[str, ...]] = ("pos", "extras")
    writes: ClassVar[tuple[str, ...]] = ("pos", "extras")
    steps: int = 2500
    gravity: float = 0.25
    gravity_range: float = 3.8

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run CiSE Steps 3-5 from existing circle placement.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs with flat cluster groups.
        state : SolveState
            Mutable solve state containing ``pos`` and circle metadata.
        ctx : RuntimeContext
            Runtime context. Unused except for API consistency.

        Returns
        -------
        SolveState
            State with relaxed CiSE positions and final circle metadata.
        """
        del ctx
        if state.pos is None:
            raise ValueError("CytoscapeCiSERelax requires initialized positions.")
        if not problem.clusters or self.steps <= 0:
            return state
        raw_meta = state.extras.get("cise_cluster_circles")
        raw_groups = state.extras.get("cise_cluster_groups")
        if not isinstance(raw_meta, list) or not isinstance(raw_groups, list):
            return state
        pos = state.pos.to(dtype=torch.float64).clone()
        circles = [
            _cise_circle_from_meta(list(group), dict(meta), pos)
            for group, meta in zip(raw_groups, raw_meta)
            if len(group) > 0
        ]
        if len(circles) <= 1:
            state.pos = pos.to(dtype=state.pos.dtype, device=state.pos.device)
            return state
        edges = _cise_edges(problem.edge_index, problem.num_nodes)
        group_by_node, _order_by_node = _cise_member_group_maps(circles)
        _cise_update_swapping_conditions(circles, edges, group_by_node)
        step_budget = max(1, int(self.steps))
        phase_lengths = (
            max(1, step_budget // 3),
            max(1, step_budget // 3),
            max(1, step_budget - 2 * (step_budget // 3)),
        )
        swapped_history: list[_CiSESwapPair] = []
        for phase_index, phase_steps in enumerate(phase_lengths):
            initial_cooling = 0.5 if phase_index == 2 else 0.4
            for iteration in range(1, phase_steps + 1):
                if iteration % _CISE_CONVERGENCE_CHECK_PERIOD == 0:
                    cooling_factor = initial_cooling * ((phase_steps - iteration) / phase_steps)
                else:
                    cooling_factor = initial_cooling
                if phase_index == 0 and iteration % _CISE_REVERSE_PERIOD == 0:
                    for circle_index in range(len(circles)):
                        if _cise_reverse_if_better(
                            circle_index,
                            circles,
                            edges,
                            group_by_node,
                            pos,
                        ):
                            break
                    group_by_node, _order_by_node = _cise_member_group_maps(circles)
                perform_swap = False
                prepare_swap = False
                if phase_index == 1:
                    if iteration % _CISE_SWAP_HISTORY_CLEARANCE_PERIOD == 0:
                        swapped_history = []
                    iteration_in_period = iteration % _CISE_SWAP_PERIOD
                    prepare_swap = iteration_in_period >= _CISE_SWAP_IDLE_DURATION
                    perform_swap = iteration_in_period == 0
                elif phase_index == 2 and iteration % _CISE_CLUSTER_ENLARGEMENT_CHECK_PERIOD == 0:
                    _cise_enlargement_check(circles, pos)
                parent_forces, parent_rotations, node_swap_rotation = _cise_calc_forces(
                    circles=circles,
                    edges=edges,
                    group_by_node=group_by_node,
                    pos=pos,
                    cooling_factor=cooling_factor,
                    gravity=float(self.gravity),
                    gravity_range=float(self.gravity_range),
                    polish=phase_index == 2,
                )
                if prepare_swap:
                    for circle in circles:
                        for node in circle.members:
                            circle.displacement_for_swap[node] = node_swap_rotation.get(node, 0.0)
                if perform_swap:
                    swapped_history = _cise_perform_swap_phase(
                        circles,
                        edges,
                        group_by_node,
                        pos,
                        swapped_history,
                    )
                    group_by_node, _order_by_node = _cise_member_group_maps(circles)
                else:
                    _cise_apply_motion(
                        circles,
                        pos,
                        parent_forces,
                        parent_rotations,
                        cooling_factor,
                    )
        state.pos = pos.to(dtype=state.pos.dtype, device=state.pos.device)
        state.extras["cise_cluster_circles"] = [
            {
                "x": circle.center[0],
                "y": circle.center[1],
                "r": circle.radius,
                "members": list(circle.members),
            }
            for circle in circles
        ]
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
    node_overlap: float = 4.0
    client_width: float = 1.0
    client_height: float = 1.0

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
        min_x, max_x, min_y, max_y = _cose_bounds(state, pos, sizes)
        offsets = torch.zeros_like(pos)
        for source in range(problem.num_nodes):
            for target in range(source + 1, problem.num_nodes):
                delta = pos[target] - pos[source]
                if float(delta[0].item()) == 0.0 and float(delta[1].item()) == 0.0:
                    random_x = -1.0 + 2.0 * _cytoscape_random(state, problem.seed)
                    random_y = -1.0 + 2.0 * _cytoscape_random(state, problem.seed)
                    delta = torch.tensor([random_x, random_y], dtype=dtype, device=device)
                if float(delta[0].item()) > 0.0:
                    overlap_x = max_x[source] - min_x[target]
                else:
                    overlap_x = max_x[target] - min_x[source]
                if float(delta[1].item()) > 0.0:
                    overlap_y = max_y[source] - min_y[target]
                else:
                    overlap_y = max_y[target] - min_y[source]
                if float(overlap_x.item()) >= 0.0 and float(overlap_y.item()) >= 0.0:
                    overlap = torch.sqrt(overlap_x * overlap_x + overlap_y * overlap_y)
                    force = self.node_overlap * overlap
                    distance = torch.clamp(torch.linalg.vector_norm(delta), min=_MIN_DISTANCE)
                    vector = force * delta / distance
                else:
                    point_source = _clipping_point(pos[source], sizes[source], delta[0], delta[1])
                    point_target = _clipping_point(pos[target], sizes[target], -delta[0], -delta[1])
                    clipped_delta = point_target - point_source
                    distance_sq = torch.clamp(
                        torch.dot(clipped_delta, clipped_delta),
                        min=_MIN_DISTANCE,
                    )
                    distance = torch.sqrt(distance_sq)
                    force = (2.0 * self.node_repulsion) / distance_sq
                    vector = force * clipped_delta / distance
                offsets[source] -= vector
                offsets[target] += vector
        for source, target in _unique_edges(problem.edge_index, problem.num_nodes):
            delta = pos[target] - pos[source]
            if float(delta[0].item()) == 0.0 and float(delta[1].item()) == 0.0:
                continue
            point_source = _clipping_point(pos[source], sizes[source], delta[0], delta[1])
            point_target = _clipping_point(pos[target], sizes[target], -delta[0], -delta[1])
            clipped_delta = point_target - point_source
            distance = torch.linalg.vector_norm(clipped_delta)
            if float(distance.item()) != 0.0:
                force = ((self.ideal_edge_length - distance) ** 2) / self.edge_elasticity
                vector = force * clipped_delta / distance
            else:
                vector = torch.zeros(2, dtype=dtype, device=device)
            offsets[source] += vector
            offsets[target] -= vector
        if self.gravity > 0.0 and problem.num_nodes > 0:
            center = torch.tensor(
                [self.client_height / 2.0, self.client_width / 2.0],
                dtype=dtype,
                device=device,
            )
            gravity_delta = center[None, :] - pos
            gravity_dist = torch.clamp(
                torch.linalg.vector_norm(gravity_delta, dim=1),
                min=_MIN_DISTANCE,
            )
            offsets += self.gravity * gravity_delta / gravity_dist[:, None]
        magnitude = torch.linalg.vector_norm(offsets, dim=1)
        scale = torch.ones_like(magnitude)
        mask = magnitude > self.temperature
        scale[mask] = self.temperature / magnitude[mask]
        state.pos = pos + offsets * scale[:, None]
        state.extras["cose_bounds"] = (
            state.pos[:, 0] - sizes[:, 0],
            state.pos[:, 0] + sizes[:, 0],
            state.pos[:, 1] - sizes[:, 1],
            state.pos[:, 1] + sizes[:, 1],
        )
        state.forces = offsets
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
    client_width: float = 1.0
    client_height: float = 1.0

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
            sizes = _node_sizes(problem, device=device).to(dtype=state.pos.dtype)
            state.extras["cose_bounds"] = (
                state.pos[:, 0] - sizes[:, 0] / 2.0,
                state.pos[:, 0] + sizes[:, 0] / 2.0,
                state.pos[:, 1] - sizes[:, 1] / 2.0,
                state.pos[:, 1] + sizes[:, 1] / 2.0,
            )
            return state
        if problem.num_nodes == 0:
            state.pos = torch.empty((0, 2), dtype=torch.float32, device=device)
            return state
        pos = torch.full(
            (problem.num_nodes, 2),
            _COSE_DEFAULT_RENDERED_NODE_CENTER,
            dtype=torch.float64,
        )
        if self.randomize:
            pos = torch.empty((problem.num_nodes, 2), dtype=torch.float64)
            for node_index in range(problem.num_nodes):
                pos[node_index, 0] = _cytoscape_random(state, problem.seed) * self.client_width
                pos[node_index, 1] = _cytoscape_random(state, problem.seed) * self.client_height
        state.pos = pos.to(device=device, dtype=torch.float32)
        sizes = _node_sizes(problem, device=device).to(dtype=state.pos.dtype)
        state.extras["cose_bounds"] = (
            state.pos[:, 0] - sizes[:, 0] / 2.0,
            state.pos[:, 0] + sizes[:, 0] / 2.0,
            state.pos[:, 1] - sizes[:, 1] / 2.0,
            state.pos[:, 1] + sizes[:, 1] / 2.0,
        )
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
