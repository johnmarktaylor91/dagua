"""Declared-axis direction, rank, tree, flow, port, and temporal facets."""

# Contract identity docstrings intentionally keep each title and digest on one line.
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import math
from collections import defaultdict, deque
from typing import DefaultDict, Dict, List, Optional, Tuple, Union

import torch

from dagua.eval.ruler_v4._util import (
    declared_axis,
    global_blend,
    mean_result,
    pava,
    resolved_ranks,
    resolved_routes,
)
from dagua.eval.ruler_v4.scene import (
    BoxGeometry,
    FacetResult,
    PortDeclaration,
    ResultState,
    Scene,
    TemporalScene,
    invalid_result,
    na_result,
    value_result,
)


def _feedback_mask(scene: Scene) -> Tuple[bool, ...]:
    """Return declared feedback or the frozen deterministic DFS fallback.

    Parameters
    ----------
    scene : Scene
        Validated directed scene.

    Returns
    -------
    tuple[bool, ...]
        Per-edge feedback bits.
    """

    if scene.graph.feedback is not None:
        return scene.graph.feedback
    outgoing: DefaultDict[int, List[Tuple[int, int]]] = defaultdict(list)
    for edge_index, (source, target) in enumerate(scene.graph.edges):
        outgoing[source].append((target, edge_index))
    state = [0] * scene.node_count
    feedback = [False] * scene.edge_count
    ordered = {node: sorted(edges) for node, edges in outgoing.items()}
    for root in range(scene.node_count):
        if state[root] != 0:
            continue
        state[root] = 1
        stack: List[Tuple[int, int]] = [(root, 0)]
        while stack:
            node, incidence_index = stack[-1]
            incidences = ordered.get(node, [])
            if incidence_index >= len(incidences):
                state[node] = 2
                stack.pop()
                continue
            target, edge_index = incidences[incidence_index]
            stack[-1] = (node, incidence_index + 1)
            if state[target] == 1:
                feedback[edge_index] = True
            elif state[target] == 0:
                state[target] = 1
                stack.append((target, 0))
    return tuple(feedback)


def U31(scene: Scene) -> FacetResult:
    """Direction consistency. Frozen SHA-256: e8c67a87acbfb2a4882c39f54ff50e6c34238cabb5324fad2ab48b0fc5f99ef2."""

    axis = declared_axis(scene)
    if axis is None or not scene.graph.directed or scene.edge_count == 0:
        return na_result("DIRECTION_OR_AXIS_ABSENT")
    feedback = _feedback_mask(scene)
    ranks = resolved_ranks(scene)
    routes = {route.edge_index: route for route in resolved_routes(scene)}
    burdens: List[float] = []
    forward_burdens: List[float] = []
    feedback_burdens: List[float] = []
    for edge_index, (source, target) in enumerate(scene.graph.edges):
        if ranks is not None and ranks[source] == ranks[target]:
            continue
        points = routes[edge_index].points
        delta = points[-1] - points[0]
        length = float(torch.linalg.vector_norm(delta).item())
        if length == 0.0:
            nonzero = points[1:] - points[:-1]
            lengths = torch.linalg.vector_norm(nonzero, dim=1)
            candidates = torch.nonzero(lengths > 0.0, as_tuple=False).flatten()
            if candidates.numel() == 0:
                return invalid_result("all_zero_required_route", {"edge_index": edge_index})
            delta = nonzero[int(candidates[0])]
            length = float(torch.linalg.vector_norm(delta).item())
        direction_sign = -1.0 if feedback[edge_index] else 1.0
        cosine = direction_sign * float(torch.dot(delta / length, axis).item())
        argument = (math.cos(math.radians(20.0)) - cosine) / 0.01
        loss = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, argument))))
        burdens.append(loss)
        (feedback_burdens if feedback[edge_index] else forward_burdens).append(loss)
    if not burdens:
        return na_result("NO_UNEQUAL_RANK_EDGE")
    defect = global_blend(burdens)
    return value_result(
        defect,
        {"U31.headline": defect},
        {
            "feedback_edge_count": sum(feedback),
            "feedback_source": "declared" if scene.graph.feedback is not None else "U31-DFS-FB-1",
            "forward_mean": sum(forward_burdens) / len(forward_burdens)
            if forward_burdens
            else None,
            "feedback_mean": sum(feedback_burdens) / len(feedback_burdens)
            if feedback_burdens
            else None,
        },
    )


def U32(scene: Scene) -> FacetResult:
    """Rank/layer clarity. Frozen SHA-256: e1fdb3ffef119b8151d07ba8673778c88bb7bc565d596248220d2e828fe6eba4."""

    axis = declared_axis(scene)
    if axis is None:
        return na_result("RANK_AXIS_ABSENT")
    rank_values = scene.graph.ranks
    if rank_values is None:
        rank_values = resolved_ranks(scene)
    if rank_values is None:
        return na_result("NO_CANONICAL_DERIVED_RANK")
    ranks = torch.tensor(rank_values, dtype=torch.long)
    projection = scene.positions @ axis
    unique = torch.unique(ranks, sorted=True)
    if unique.numel() < 2:
        return na_result("RANK_AXIS_ABSENT")
    medians = torch.stack([_midpoint_median(projection[ranks == rank]) for rank in unique])
    deviations = torch.stack(
        [
            _midpoint_median(torch.abs(projection[ranks == rank] - medians[index]))
            for index, rank in enumerate(unique)
        ]
    )
    counts = torch.tensor(
        [int((ranks == rank).sum().item()) for rank in unique], dtype=torch.float64
    )
    projected_half_extents = torch.tensor(
        [float(torch.dot(torch.abs(axis), box.half_extents).item()) for box in scene.node_boxes],
        dtype=torch.float64,
    )
    scales = []
    for rank in unique:
        scales.append(
            max(
                scene.intrinsic_unit,
                2.0 * float(torch.max(projected_half_extents[ranks == rank]).item()),
            )
        )
    pitches = []
    for index in range(unique.numel() - 1):
        left_extent = float(torch.max(projected_half_extents[ranks == unique[index]]).item())
        right_extent = float(torch.max(projected_half_extents[ranks == unique[index + 1]]).item())
        pitches.append(max(scene.intrinsic_unit, left_extent + right_extent))
    offsets = [0.0]
    for pitch in pitches:
        offsets.append(offsets[-1] + pitch)
    offset_tensor = torch.tensor(offsets, dtype=torch.float64)
    fitted = pava(medians - offset_tensor, counts) + offset_tensor
    iso_objects = []
    for index, count in enumerate(counts.to(torch.long)):
        burden = 1.0 - math.exp(-((float(medians[index] - fitted[index]) / scales[index]) ** 2))
        iso_objects.extend([burden] * int(count))
    overlaps = []
    overlap_weights = []
    for index, pitch in enumerate(pitches):
        numerator = (
            float(medians[index] + 2.9652 * deviations[index])
            - float(medians[index + 1] - 2.9652 * deviations[index + 1])
            + 0.5 * pitch
        )
        argument = numerator / (0.05 * pitch)
        overlaps.append(1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, argument)))))
        overlap_weights.append(float(counts[index] + counts[index + 1]))
    crisp_objects = []
    for index, count in enumerate(counts.to(torch.long)):
        local = 1.0 - math.exp(-((float(deviations[index]) / scales[index]) ** 2))
        left_resolution = 1.0 - overlaps[index - 1] if index > 0 else 1.0
        right_resolution = 1.0 - overlaps[index] if index < len(overlaps) else 1.0
        resolution = left_resolution * right_resolution
        burden = 1.0 - (1.0 - local) * resolution
        crisp_objects.extend([burden] * int(count))
    values = {
        "U32.L_iso": global_blend(iso_objects),
        "U32.L_crisp": global_blend(crisp_objects),
        "U32.L_overlap": global_blend(overlaps, overlap_weights),
    }
    return mean_result(
        "U32",
        values,
        {
            "rank_count": unique.numel(),
            "medians": tuple(float(value) for value in medians),
            "mads": tuple(float(value) for value in deviations),
            "pitches": tuple(pitches),
            "fitted_centers": tuple(float(value) for value in fitted),
        },
    )


def _midpoint_median(values: torch.Tensor) -> torch.Tensor:
    """Return the conventional midpoint median of one vector.

    Parameters
    ----------
    values : torch.Tensor
        Nonempty one-dimensional float64 values.

    Returns
    -------
    torch.Tensor
        Scalar midpoint median.
    """

    ordered = torch.sort(values).values
    count = ordered.numel()
    if count % 2:
        return ordered[count // 2]
    return (ordered[count // 2 - 1] + ordered[count // 2]) / 2.0


def U33(scene: Scene) -> FacetResult:
    """Tree quality bundle. Frozen SHA-256: b9ea391f1645e98497c216a76ba3c1b0d69901cdf5d54c42872376a7ea4e7feb."""

    parents = scene.graph.tree_parents
    depths = scene.graph.tree_depths
    mode = scene.graph.tree_layout
    if parents is None or depths is None or mode is None:
        return invalid_result("TREE_SEMANTICS_ABSENT")
    if mode not in {"layered", "radial"}:
        return invalid_result("invalid_tree_layout")
    if len(parents) != scene.node_count or len(depths) != scene.node_count:
        return invalid_result("malformed_tree_semantics")
    children: DefaultDict[int, List[int]] = defaultdict(list)
    roots = []
    for node, parent in enumerate(parents):
        if parent is None:
            roots.append(node)
        elif parent < 0 or parent >= scene.node_count or depths[node] != depths[parent] + 1:
            return invalid_result("malformed_tree_semantics")
        else:
            children[parent].append(node)
    if not roots or all(not child_nodes for child_nodes in children.values()):
        return invalid_result("TREE_SEMANTICS_ABSENT")
    if mode == "layered":
        if scene.graph.flow_axis is None:
            return invalid_result("tree_depth_axis_absent")
        return _u33_layered(scene, children)
    return _u33_radial(scene, children, roots, depths)


def _u33_layered(scene: Scene, children: DefaultDict[int, List[int]]) -> FacetResult:
    """Evaluate the four declared layered-tree subterms.

    Parameters
    ----------
    scene : Scene
        Validated declared-tree scene.
    children : defaultdict[int, list[int]]
        Canonical parent-to-child mapping.

    Returns
    -------
    FacetResult
        Frozen weighted layered-tree result.
    """

    axis = torch.tensor(scene.graph.flow_axis, dtype=torch.float64)
    # The declared page direction is top-to-bottom; its clockwise perpendicular
    # therefore makes declared sibling order increase left-to-right.
    cross = torch.tensor([axis[1], -axis[0]], dtype=torch.float64)
    subtree_cache: Dict[int, List[int]] = {}

    def subtree(node: int) -> List[int]:
        """Return canonical descendants including one root node.

        Parameters
        ----------
        node : int
            Subtree root node.

        Returns
        -------
        list[int]
            Root and descendants in canonical order.
        """

        if node not in subtree_cache:
            members = [node]
            for child in sorted(children[node]):
                members.extend(subtree(child))
            subtree_cache[node] = members
        return subtree_cache[node]

    separation_losses: List[float] = []
    separation_weights: List[float] = []
    centering_losses: List[float] = []
    depth_losses: List[float] = []
    order_losses: List[float] = []
    for parent, child_nodes in sorted(children.items()):
        if not child_nodes:
            continue
        child_nodes = sorted(child_nodes)
        for left_index, left in enumerate(child_nodes):
            left_subtree = subtree(left)
            for right in child_nodes[left_index + 1 :]:
                right_subtree = subtree(right)
                clearance = _convex_hull_signed_clearance(
                    scene.positions[left_subtree], scene.positions[right_subtree]
                )
                argument = -clearance / (0.05 * scene.intrinsic_unit)
                separation_losses.append(_stable_sigmoid(argument))
                separation_weights.append(float(min(len(left_subtree), len(right_subtree))))
        child_positions = scene.positions[child_nodes]
        if len(child_nodes) == 1:
            centering_losses.append(0.0)
        else:
            centroid = torch.mean(child_positions, dim=0)
            span = math.sqrt(
                float(torch.mean(torch.sum((child_positions - centroid) ** 2, dim=1)).item())
            )
            offset = abs(float(torch.dot(scene.positions[parent] - centroid, cross).item()))
            centering_losses.append(0.0 if span == 0.0 else 1.0 - math.exp(-((offset / span) ** 2)))
    for node, parent in enumerate(scene.graph.tree_parents or ()):
        if parent is None:
            continue
        delta = scene.positions[node] - scene.positions[parent]
        length = float(torch.linalg.vector_norm(delta).item())
        cosine = float(torch.dot(delta / length, axis).item()) if length > 0.0 else 0.0
        depth_losses.append(_stable_sigmoid((math.cos(math.radians(70.0)) - cosine) / 0.03))
    for parent, declared_order in sorted(scene.graph.ordered_children.items()):
        for left, right in zip(declared_order[:-1], declared_order[1:]):
            delta = scene.positions[right] - scene.positions[left]
            length = float(torch.linalg.vector_norm(delta).item())
            cosine = float(torch.dot(delta / length, cross).item()) if length > 0.0 else 0.0
            order_losses.append(_stable_sigmoid((math.cos(math.radians(70.0)) - cosine) / 0.03))
    values = {
        "U33.layered.2": global_blend(centering_losses),
        "U33.layered.3": global_blend(depth_losses),
    }
    if separation_losses:
        values["U33.layered.1"] = global_blend(separation_losses, separation_weights)
    if scene.graph.ordered_children:
        if order_losses:
            values["U33.layered.4"] = global_blend(order_losses)
    dropped = []
    if not separation_losses:
        dropped.append("U33.layered.1:no_sibling_subtree_pairs")
    if scene.graph.ordered_children and not order_losses:
        dropped.append("U33.layered.4:no_consecutive_declared_child_pairs")
    return mean_result(
        "U33",
        values,
        {"tree_layout": "layered", "dropped_subterms": tuple(dropped)},
    )


def _u33_radial(
    scene: Scene,
    children: DefaultDict[int, List[int]],
    roots: List[int],
    depths: Tuple[int, ...],
) -> FacetResult:
    """Evaluate radius monotonicity and sector allocation for radial trees.

    Parameters
    ----------
    scene : Scene
        Validated declared-tree scene.
    children : defaultdict[int, list[int]]
        Parent-to-child mapping.
    roots : list[int]
        Declared forest roots.
    depths : tuple[int, ...]
        Declared depth per node.

    Returns
    -------
    FacetResult
        Frozen weighted radial-tree result.
    """

    radial_losses: List[float] = []
    sector_losses: List[float] = []
    sector_weights: List[float] = []
    for root in roots:
        members = _tree_members(children, root)
        root_depth = depths[root]
        local_depths = sorted(set(depths[node] - root_depth for node in members))
        radii = torch.tensor(
            [
                float(
                    torch.linalg.vector_norm(scene.positions[node] - scene.positions[root]).item()
                )
                for node in members
            ],
            dtype=torch.float64,
        )
        depth_tensor = torch.tensor([depths[node] - root_depth for node in members])
        medians = torch.stack(
            [_midpoint_median(radii[depth_tensor == depth]) for depth in local_depths]
        )
        counts = torch.tensor(
            [int((depth_tensor == depth).sum().item()) for depth in local_depths],
            dtype=torch.float64,
        )
        offsets = torch.arange(len(local_depths), dtype=torch.float64) * scene.intrinsic_unit
        fitted = pava(medians - offsets, counts) + offsets
        for index, count in enumerate(counts.to(torch.long)):
            burden = 1.0 - math.exp(
                -((float(medians[index] - fitted[index]) / scene.intrinsic_unit) ** 2)
            )
            radial_losses.extend([burden] * int(count))
        root_children = sorted(children[root])
        if len(root_children) >= 2:
            child_members = [_tree_members(children, child) for child in root_children]
            masses = [len(items) for items in child_members]
            total_mass = sum(masses)
            for items, mass in zip(child_members, masses):
                angles = sorted(
                    math.atan2(
                        float((scene.positions[node] - scene.positions[root])[1]),
                        float((scene.positions[node] - scene.positions[root])[0]),
                    )
                    % (2.0 * math.pi)
                    for node in items
                )
                if len(angles) == 1:
                    fraction = 0.0
                else:
                    gaps = [
                        (angles[(index + 1) % len(angles)] - angles[index]) % (2.0 * math.pi)
                        for index in range(len(angles))
                    ]
                    fraction = (2.0 * math.pi - max(gaps)) / (2.0 * math.pi)
                target = mass / total_mass
                sector_losses.append(
                    1.0
                    - math.exp(-(((fraction - target) / (target + 1.0 / len(root_children))) ** 2))
                )
                sector_weights.append(float(mass))
    values = {"U33.radial.1": global_blend(radial_losses)}
    if sector_losses:
        values["U33.radial.2"] = global_blend(sector_losses, sector_weights)
    dropped = () if sector_losses else ("U33.radial.2:no_multi_child_root",)
    return mean_result(
        "U33",
        values,
        {"tree_layout": "radial", "dropped_subterms": dropped},
    )


def _tree_depths(scene: Scene) -> Optional[List[int]]:
    """Derive canonical root depths for a declared tree.

    Parameters
    ----------
    scene : Scene
        Directed rooted scene.

    Returns
    -------
    list[int] or None
        Depth per node, or None when the declaration is disconnected.
    """

    children: DefaultDict[int, List[int]] = defaultdict(list)
    for source, target in scene.graph.edges:
        children[source].append(target)
    depths = [-1] * scene.node_count
    queue = deque((root, 0) for root in scene.graph.roots)
    while queue:
        node, depth = queue.popleft()
        if depths[node] >= 0 and depths[node] <= depth:
            continue
        depths[node] = depth
        queue.extend((child, depth + 1) for child in children[node])
    return depths if all(depth >= 0 for depth in depths) else None


def _stable_sigmoid(argument: float) -> float:
    """Evaluate a numerically stable scalar logistic function.

    Parameters
    ----------
    argument : float
        Unbounded logistic coordinate.

    Returns
    -------
    float
        Value in ``(0, 1)``.
    """

    clipped = max(-60.0, min(60.0, argument))
    return 1.0 / (1.0 + math.exp(-clipped))


def _tree_members(children: DefaultDict[int, List[int]], root: int) -> List[int]:
    """Return one tree's canonical node population.

    Parameters
    ----------
    children : defaultdict[int, list[int]]
        Parent-to-child mapping.
    root : int
        Root node index.

    Returns
    -------
    list[int]
        Root and descendants in canonical depth-first order.
    """

    result = []
    stack = [root]
    while stack:
        node = stack.pop()
        result.append(node)
        stack.extend(reversed(sorted(children[node])))
    return result


def _convex_hull(points: torch.Tensor) -> List[torch.Tensor]:
    """Construct a deterministic two-dimensional monotone-chain hull.

    Parameters
    ----------
    points : torch.Tensor
        Point coordinates with shape ``[N, 2]``.

    Returns
    -------
    list[torch.Tensor]
        Counterclockwise hull vertices without a repeated endpoint.
    """

    ordered = sorted(
        (point for point in points), key=lambda point: (float(point[0]), float(point[1]))
    )
    unique = []
    for point in ordered:
        if not unique or not torch.equal(point, unique[-1]):
            unique.append(point)
    if len(unique) <= 2:
        return unique

    def cross(origin: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> float:
        """Return the signed turn of three hull candidates.

        Parameters
        ----------
        origin, left, right : torch.Tensor
            Candidate points with shape ``[2]``.

        Returns
        -------
        float
            Scalar two-dimensional cross product.
        """

        first = left - origin
        second = right - origin
        return float(first[0] * second[1] - first[1] * second[0])

    lower: List[torch.Tensor] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: List[torch.Tensor] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _point_segment_distance(point: torch.Tensor, start: torch.Tensor, end: torch.Tensor) -> float:
    """Return Euclidean point-to-segment distance.

    Parameters
    ----------
    point, start, end : torch.Tensor
        Coordinates with shape ``[2]``.

    Returns
    -------
    float
        Nonnegative distance.
    """

    direction = end - start
    denominator = float(torch.dot(direction, direction).item())
    if denominator == 0.0:
        return float(torch.linalg.vector_norm(point - start).item())
    parameter = float(torch.dot(point - start, direction).item()) / denominator
    parameter = min(1.0, max(0.0, parameter))
    return float(torch.linalg.vector_norm(point - (start + parameter * direction)).item())


def _convex_hull_signed_clearance(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return signed clearance between two convex node-center hulls.

    Parameters
    ----------
    left, right : torch.Tensor
        Subtree node centers with shapes ``[L, 2]`` and ``[R, 2]``.

    Returns
    -------
    float
        Positive Euclidean separation or negative SAT penetration depth.
    """

    hull_left = _convex_hull(left)
    hull_right = _convex_hull(right)
    if len(hull_left) == 1 and len(hull_right) == 1:
        return float(torch.linalg.vector_norm(hull_left[0] - hull_right[0]).item())
    axes = []
    for hull in (hull_left, hull_right):
        if len(hull) < 2:
            continue
        edge_count = len(hull) if len(hull) > 2 else 1
        for index in range(edge_count):
            direction = hull[(index + 1) % len(hull)] - hull[index]
            length = float(torch.linalg.vector_norm(direction).item())
            if length > 0.0:
                axes.append(
                    torch.tensor([-float(direction[1]), float(direction[0])], dtype=torch.float64)
                    / length
                )
    minimum_overlap = float("inf")
    separated = False
    for axis in axes:
        left_projection = torch.tensor([float(torch.dot(point, axis)) for point in hull_left])
        right_projection = torch.tensor([float(torch.dot(point, axis)) for point in hull_right])
        overlap = min(float(torch.max(left_projection)), float(torch.max(right_projection))) - max(
            float(torch.min(left_projection)), float(torch.min(right_projection))
        )
        if overlap < 0.0:
            separated = True
        minimum_overlap = min(minimum_overlap, overlap)
    if not separated and minimum_overlap != float("inf"):
        return -max(0.0, minimum_overlap)
    left_segments = _hull_segments(hull_left)
    right_segments = _hull_segments(hull_right)
    distances = []
    for start_left, end_left in left_segments:
        for start_right, end_right in right_segments:
            distances.extend(
                (
                    _point_segment_distance(start_left, start_right, end_right),
                    _point_segment_distance(end_left, start_right, end_right),
                    _point_segment_distance(start_right, start_left, end_left),
                    _point_segment_distance(end_right, start_left, end_left),
                )
            )
    return min(distances) if distances else 0.0


def _hull_segments(hull: List[torch.Tensor]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Return closed boundary segments for a point, line, or polygon hull.

    Parameters
    ----------
    hull : list[torch.Tensor]
        Convex hull vertices.

    Returns
    -------
    list[tuple[torch.Tensor, torch.Tensor]]
        Boundary segments; a singleton is represented as a zero-length segment.
    """

    if len(hull) == 1:
        return [(hull[0], hull[0])]
    if len(hull) == 2:
        return [(hull[0], hull[1])]
    return [(hull[index], hull[(index + 1) % len(hull)]) for index in range(len(hull))]


def U34(scene: Scene) -> FacetResult:
    """Flow path traceability. Frozen SHA-256: 30aeff9b4e59ee355dbad183fb26686c084c426eb1430762b8e6f273e2fdb2cc."""

    axis = declared_axis(scene)
    if axis is None or not scene.graph.directed:
        return na_result("FLOW_SEMANTICS_ABSENT")
    paths = _canonical_source_sink_paths(scene)
    if not paths:
        return na_result("NO_SOURCE_SINK_PATH")
    route_by_edge = {route.edge_index: route for route in resolved_routes(scene)}
    degrees = [0] * scene.node_count
    for source, target in scene.graph.edges:
        degrees[source] += 1
        degrees[target] += 1
    back_losses: List[float] = []
    mono_losses: List[float] = []
    continuity_losses: List[float] = []
    path_losses: List[float] = []
    path_weights: List[float] = []
    for path_nodes, path_edges, path_weight in paths:
        deltas = []
        junction_vectors: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for edge_position, edge_index in enumerate(path_edges):
            route = route_by_edge[edge_index]
            local_deltas = route.points[1:] - route.points[:-1]
            local_deltas = local_deltas[torch.linalg.vector_norm(local_deltas, dim=1) > 0.0]
            if local_deltas.numel() == 0:
                return invalid_result("all_zero_required_route", {"edge_index": edge_index})
            deltas.extend(local_deltas)
            if edge_position > 0:
                junction = path_nodes[edge_position]
                incoming_route = route_by_edge[path_edges[edge_position - 1]]
                incoming_deltas = incoming_route.points[1:] - incoming_route.points[:-1]
                incoming_nonzero = incoming_deltas[
                    torch.linalg.vector_norm(incoming_deltas, dim=1) > 0.0
                ]
                junction_vectors[junction] = (-incoming_nonzero[-1], local_deltas[0])
        delta_tensor = torch.stack(deltas)
        lengths = torch.linalg.vector_norm(delta_tensor, dim=1)
        mean_length = float(torch.mean(lengths).item())
        tau = 0.01 * mean_length
        signed = delta_tensor @ axis
        progress = sum(_soft_positive(float(value), tau) for value in signed)
        backtrack = sum(_soft_positive(-float(value), tau) for value in signed)
        back_loss = backtrack / (progress + backtrack)
        unit_directions = delta_tensor / lengths[:, None]
        mono_loss = sum(
            _stable_sigmoid(-float(torch.dot(direction, axis).item()) / 0.03)
            for direction in unit_directions
        ) / len(unit_directions)
        junction_losses = []
        for junction, (incoming, outgoing) in junction_vectors.items():
            if degrees[junction] == 2:
                continue
            incoming = incoming / torch.linalg.vector_norm(incoming)
            outgoing = outgoing / torch.linalg.vector_norm(outgoing)
            junction_losses.append((1.0 + float(torch.dot(incoming, outgoing).item())) / 2.0)
        continuity_loss = sum(junction_losses) / len(junction_losses) if junction_losses else 0.0
        back_losses.append(back_loss)
        mono_losses.append(mono_loss)
        continuity_losses.append(continuity_loss)
        path_losses.append(0.40 * back_loss + 0.35 * mono_loss + 0.25 * continuity_loss)
        path_weights.append(path_weight)
    values = {
        "U34.L_back": global_blend(back_losses, path_weights),
        "U34.L_mono": global_blend(mono_losses, path_weights),
        "U34.L_cont": global_blend(continuity_losses, path_weights),
    }
    return value_result(
        global_blend(path_losses, path_weights),
        values,
        {"path_count": len(paths)},
    )


def _soft_positive(value: float, temperature: float) -> float:
    """Evaluate the scale-homogeneous soft positive-part function.

    Parameters
    ----------
    value : float
        Signed segment progress.
    temperature : float
        Positive path-relative shoulder width.

    Returns
    -------
    float
        ``temperature * log1p(exp(value/temperature))``.
    """

    if temperature <= 0.0:
        return max(0.0, value)
    coordinate = value / temperature
    if coordinate > 40.0:
        return value
    if coordinate < -40.0:
        return temperature * math.exp(coordinate)
    return temperature * math.log1p(math.exp(coordinate))


def _canonical_source_sink_paths(
    scene: Scene,
) -> List[Tuple[List[int], List[int], float]]:
    """Enumerate canonical shortest directed source-to-sink paths.

    Parameters
    ----------
    scene : Scene
        Validated directed graph scene.

    Returns
    -------
    list[tuple[list[int], list[int], float]]
        Node and edge indices plus inverse-inclusion mass for each sampled
        reachable source-sink pair.
    """

    incoming = [0] * scene.node_count
    outgoing: DefaultDict[int, List[Tuple[int, int]]] = defaultdict(list)
    for edge_index, (source, target) in enumerate(scene.graph.edges):
        incoming[target] += 1
        outgoing[source].append((target, edge_index))
    sources = (
        sorted(scene.graph.roots)
        if scene.graph.roots
        else [node for node, degree in enumerate(incoming) if degree == 0]
    )
    sinks = {node for node in range(scene.node_count) if not outgoing[node]}
    source_population = len(sources)
    if source_population > 64:
        sources = sorted(
            sources,
            key=lambda source: hashlib.sha256(
                f"{scene.profile_hash}:U34-source:{source}".encode()
            ).digest(),
        )[:64]
    source_probability = min(1.0, 64.0 / source_population) if source_population else 1.0
    result: List[Tuple[List[int], List[int], float]] = []
    for source in sources:
        predecessor: Dict[int, Tuple[int, int]] = {}
        predecessor_key: Dict[int, bytes] = {}
        distance = {source: 0}
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for target, edge_index in sorted(outgoing[node]):
                candidate = distance[node] + 1
                tie_key = hashlib.sha256(
                    f"{scene.profile_hash}:{source}:{target}:{edge_index}".encode()
                ).digest()
                if target not in distance or (
                    candidate == distance[target] and tie_key < predecessor_key[target]
                ):
                    distance[target] = candidate
                    predecessor[target] = (node, edge_index)
                    predecessor_key[target] = tie_key
                    if target not in queue:
                        queue.append(target)
        reachable = sorted(sinks & set(distance) - {source})
        selected: List[Tuple[int, float]] = []
        if len(reachable) <= 64:
            selected = [(sink, 1.0) for sink in reachable]
        else:
            bands = ((1, 2), (3, 4), (5, 8), (9, math.inf))
            for lower, upper in bands:
                members = [sink for sink in reachable if lower <= distance[sink] <= upper]
                chosen = sorted(
                    members,
                    key=lambda sink: hashlib.sha256(
                        f"{scene.profile_hash}:U34-sink:{source}:{sink}".encode()
                    ).digest(),
                )[:16]
                probability = min(1.0, 16.0 / len(members)) if members else 1.0
                selected.extend((sink, probability) for sink in chosen)
        for sink, sink_probability in selected:
            if sink == source:
                continue
            nodes = [sink]
            edges = []
            cursor = sink
            while cursor != source:
                parent, edge_index = predecessor[cursor]
                nodes.append(parent)
                edges.append(edge_index)
                cursor = parent
            weight = 1.0 / (source_probability * sink_probability)
            result.append((list(reversed(nodes)), list(reversed(edges)), weight))
    return result


def _port_direction(name: Optional[str]) -> Optional[torch.Tensor]:
    """Map a declared cardinal port name to an outward vector.

    Parameters
    ----------
    name : str or None
        Declared port token.

    Returns
    -------
    torch.Tensor or None
        Unit cardinal direction.
    """

    values = {
        "north": torch.tensor([0.0, 1.0], dtype=torch.float64),
        "south": torch.tensor([0.0, -1.0], dtype=torch.float64),
        "east": torch.tensor([1.0, 0.0], dtype=torch.float64),
        "west": torch.tensor([-1.0, 0.0], dtype=torch.float64),
    }
    return values.get(name or "")


def U39(scene: Scene) -> FacetResult:
    """Port compliance. Frozen SHA-256: 5de7639e452aa58f5d54c5d3530bda54f74def1ab2e09e8baaa57aa4cd8f3d90."""

    if not scene.graph.ports:
        return na_result("PORTS_ABSENT")
    route_by_edge = {route.edge_index: route for route in scene.routes}
    rows: Dict[str, List[float]] = {key: [] for key in ("U39.1", "U39.2", "U39.3", "U39.4")}
    endpoints: Dict[Tuple[int, str], List[Tuple[PortDeclaration, torch.Tensor, torch.Tensor]]] = {}
    for edge_index, ports in sorted(scene.graph.ports.items()):
        route = route_by_edge.get(edge_index)
        if route is None:
            return invalid_result("missing_required_port_route")
        for endpoint, declaration in enumerate(ports):
            if declaration is None:
                continue
            terminal = route.points[0] if endpoint == 0 else route.points[-1]
            tangent = _terminal_tangent(route.points, endpoint)
            if tangent is None:
                return invalid_result("all_zero_required_route")
            anchor = _port_anchor(scene.node_boxes[declaration.node_id], declaration)
            half_diagonal = float(
                torch.linalg.vector_norm(scene.node_boxes[declaration.node_id].half_extents)
            )
            displacement = float(torch.linalg.vector_norm(terminal - anchor)) / half_diagonal
            rows["U39.1"].append(1.0 - math.exp(-((displacement / 0.05) ** 2)))
            expected = torch.tensor(declaration.expected_approach, dtype=torch.float64)
            cosine = float(torch.dot(tangent, expected))
            rows["U39.2"].append(_stable_sigmoid((math.cos(math.radians(25.0)) - cosine) / 0.02))
            sample = _sample_from_terminal(route.points, endpoint, 0.5 * scene.intrinsic_unit)
            endpoints.setdefault((declaration.node_id, declaration.side), []).append(
                (declaration, terminal, sample)
            )
    for (node, side), records in endpoints.items():
        ordered = sorted(records, key=lambda item: (item[0].order, item[0].port_id))
        side_length = (
            2.0 * float(scene.node_boxes[node].half_extents[0])
            if side in {"N", "S"}
            else 2.0 * float(scene.node_boxes[node].half_extents[1])
        )
        for left, right in zip(ordered, ordered[1:]):
            if left[0].side_coordinate == right[0].side_coordinate:
                continue
            left_coordinate = _terminal_side_coordinate(left[1], scene.node_boxes[node], side)
            right_coordinate = _terminal_side_coordinate(right[1], scene.node_boxes[node], side)
            margin = (right_coordinate - left_coordinate) / side_length
            rows["U39.3"].append(_stable_sigmoid(-margin / 0.02))
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                separation = (
                    float(torch.linalg.vector_norm(left[2] - right[2])) / scene.intrinsic_unit
                )
                rows["U39.4"].append(_stable_sigmoid((0.20 - separation) / 0.03))
    values = {key: global_blend(items) for key, items in rows.items() if items}
    return mean_result(
        "U39", values, {"population_counts": {key: len(value) for key, value in rows.items()}}
    )


def _port_anchor(box: BoxGeometry, declaration: PortDeclaration) -> torch.Tensor:
    """Derive a declared port anchor on one node box side.

    Parameters
    ----------
    box : BoxGeometry
        Owning derived node box.
    declaration : PortDeclaration
        Immutable side coordinate.

    Returns
    -------
    torch.Tensor
        Anchor point with shape ``[2]``.
    """

    coordinate = declaration.side_coordinate
    if declaration.side == "N":
        return box.center + torch.tensor(
            [(2.0 * coordinate - 1.0) * float(box.half_extents[0]), float(box.half_extents[1])],
            dtype=torch.float64,
        )
    if declaration.side == "S":
        return box.center + torch.tensor(
            [(2.0 * coordinate - 1.0) * float(box.half_extents[0]), -float(box.half_extents[1])],
            dtype=torch.float64,
        )
    if declaration.side == "E":
        return box.center + torch.tensor(
            [float(box.half_extents[0]), (2.0 * coordinate - 1.0) * float(box.half_extents[1])],
            dtype=torch.float64,
        )
    return box.center + torch.tensor(
        [-float(box.half_extents[0]), (2.0 * coordinate - 1.0) * float(box.half_extents[1])],
        dtype=torch.float64,
    )


def _terminal_tangent(points: torch.Tensor, endpoint: int) -> Optional[torch.Tensor]:
    """Return the first nonzero route tangent pointing away from a terminal.

    Parameters
    ----------
    points : torch.Tensor
        Route vertices with shape ``[P, 2]``.
    endpoint : int
        Zero for source or one for target.

    Returns
    -------
    torch.Tensor or None
        Unit tangent, or ``None`` for an all-zero route.
    """

    ordered = points if endpoint == 0 else torch.flip(points, dims=(0,))
    for point in ordered[1:]:
        delta = point - ordered[0]
        length = float(torch.linalg.vector_norm(delta))
        if length > 0.0:
            return delta / length
    return None


def _sample_from_terminal(points: torch.Tensor, endpoint: int, distance: float) -> torch.Tensor:
    """Sample a route at a fixed arc distance from one terminal.

    Parameters
    ----------
    points : torch.Tensor
        Route vertices with shape ``[P, 2]``.
    endpoint : int
        Zero for source or one for target.
    distance : float
        Nonnegative arc distance.

    Returns
    -------
    torch.Tensor
        Sampled point, using the opposite endpoint when the route is shorter.
    """

    ordered = points if endpoint == 0 else torch.flip(points, dims=(0,))
    remaining = distance
    for start, end in zip(ordered[:-1], ordered[1:]):
        length = float(torch.linalg.vector_norm(end - start))
        if length >= remaining and length > 0.0:
            return start + (remaining / length) * (end - start)
        remaining -= length
    return ordered[-1]


def _terminal_side_coordinate(point: torch.Tensor, box: BoxGeometry, side: str) -> float:
    """Return the signed node-local coordinate of a route terminal.

    Parameters
    ----------
    point : torch.Tensor
        Route terminal with shape ``[2]``.
    box : BoxGeometry
        Owning node box.
    side : str
        Cardinal side token.

    Returns
    -------
    float
        Coordinate increasing with the declared side-coordinate convention.
    """

    axis = 0 if side in {"N", "S"} else 1
    return float(point[axis] - box.center[axis])


def U40(scene: Union[Scene, TemporalScene]) -> FacetResult:
    """Temporal / mental-map continuity. Frozen SHA-256: b10f18a4380790db1be284956dc2094f3bdcc72334be500b7c08b5e9cc43ce8c."""

    if isinstance(scene, Scene):
        return na_result("TEMPORAL_PROFILE_ABSENT")
    transition_values = []
    displacement_values = []
    churn_values = []
    identity_values = []
    transforms = []
    for before, after, declaration in zip(scene.frames, scene.frames[1:], scene.transitions):
        before_by_id = {
            identifier: index for index, identifier in enumerate(before.graph.temporal_ids or ())
        }
        after_by_id = {
            identifier: index for index, identifier in enumerate(after.graph.temporal_ids or ())
        }
        common = sorted(set(before_by_id) & set(after_by_id))
        if len(common) < 2:
            continue
        previous = before.positions[[before_by_id[identifier] for identifier in common]]
        current = after.positions[[after_by_id[identifier] for identifier in common]]
        if float(torch.max(torch.cdist(previous, previous))) == 0.0:
            continue
        aligned, transform = _rigid_align(current, previous)
        transforms.append(transform)
        unit = before.intrinsic_unit
        losses = []
        for index, identifier in enumerate(common):
            displacement = float(torch.linalg.vector_norm(aligned[index] - previous[index])) / unit
            target = declaration.expected_displacements[identifier]
            losses.append(1.0 - math.exp(-(((displacement - target) / 0.20) ** 2)))
        displacement = global_blend(losses)
        churn_losses = _temporal_angular_churn(before, after, common, aligned, previous)
        churn = global_blend(churn_losses)
        entering_or_exiting = [
            identifier
            for identifier, state in declaration.states.items()
            if state in {"enter", "exit"}
        ]
        identity = 0.0
        applicable = [(0.55, displacement), (0.25, churn)]
        if entering_or_exiting:
            identity_losses = _temporal_identity_losses(
                before, after, entering_or_exiting, before_by_id, after_by_id, common
            )
            identity = global_blend(identity_losses)
            applicable.append((0.20, identity))
        mass = sum(weight for weight, _ in applicable)
        transition_values.append(
            (
                declaration.elapsed_time,
                sum(weight * value for weight, value in applicable) / mass,
            )
        )
        displacement_values.append(displacement)
        churn_values.append(churn)
        identity_values.append(identity)
    if not transition_values:
        return na_result("INSUFFICIENT_COMMON_NODES")
    elapsed = sum(weight for weight, _ in transition_values)
    defect = sum(weight * value for weight, value in transition_values) / elapsed
    return FacetResult(
        ResultState.VALUE,
        None,
        None,
        {
            "U40.1": sum(displacement_values) / len(displacement_values),
            "U40.2": sum(churn_values) / len(churn_values),
            "U40.3": sum(identity_values) / len(identity_values),
        },
        {
            "transition_count": len(transition_values),
            "temporal_headline": defect,
            "transforms": tuple(transforms),
        },
        defect,
    )


def _rigid_align(
    current: torch.Tensor, previous: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Rigidly align current points to previous points without scale or reflection.

    Parameters
    ----------
    current, previous : torch.Tensor
        Corresponding point arrays with shape ``[N, 2]``.

    Returns
    -------
    tuple[torch.Tensor, dict[str, object]]
        Aligned current points and published transform diagnostics.
    """

    current_center = torch.mean(current, dim=0)
    previous_center = torch.mean(previous, dim=0)
    centered_current = current - current_center
    centered_previous = previous - previous_center
    left, singular, right = torch.linalg.svd(centered_current.T @ centered_previous)
    rotation = (
        torch.eye(2, dtype=torch.float64)
        if singular.numel() == 2 and float(singular[0]) == float(singular[1])
        else left @ right
    )
    if float(torch.linalg.det(rotation)) < 0.0:
        left = left.clone()
        left[:, -1] *= -1.0
        rotation = left @ right
    aligned = centered_current @ rotation + previous_center
    return aligned, {
        "rotation": rotation.tolist(),
        "current_center": current_center.tolist(),
        "previous_center": previous_center.tolist(),
        "singular_values": singular.tolist(),
    }


def _temporal_angular_churn(
    before: Scene,
    after: Scene,
    common: List[str],
    aligned: torch.Tensor,
    previous: torch.Tensor,
) -> List[float]:
    """Measure retained-neighbor angular-order changes for common temporal ids.

    Parameters
    ----------
    before, after : Scene
        Consecutive validated frames.
    common : list[str]
        Canonically ordered common temporal ids.
    aligned, previous : torch.Tensor
        Aligned current and previous common-node positions ``[C, 2]``.

    Returns
    -------
    list[float]
        Per-node unavoidable-churn-adjusted normalized Kendall losses.
    """
    before_ids = before.graph.temporal_ids or ()
    after_ids = after.graph.temporal_ids or ()
    before_neighbors: Dict[str, set[str]] = {identifier: set() for identifier in before_ids}
    after_neighbors: Dict[str, set[str]] = {identifier: set() for identifier in after_ids}
    for source, target in before.graph.edges:
        before_neighbors[before_ids[source]].add(before_ids[target])
        before_neighbors[before_ids[target]].add(before_ids[source])
    for source, target in after.graph.edges:
        after_neighbors[after_ids[source]].add(after_ids[target])
        after_neighbors[after_ids[target]].add(after_ids[source])
    common_set = set(common)
    common_index = {identifier: index for index, identifier in enumerate(common)}
    losses = []
    for identifier in common:
        old_set = before_neighbors[identifier] & common_set
        new_set = after_neighbors[identifier] & common_set
        union = old_set | new_set
        input_churn = 1.0 - len(old_set & new_set) / len(union) if union else 0.0
        retained = sorted(old_set & new_set)
        if len(retained) < 2:
            losses.append(0.0)
            continue
        center = common_index[identifier]
        old_order = sorted(
            retained,
            key=lambda neighbor: (
                math.atan2(
                    float(previous[common_index[neighbor]][1] - previous[center][1]),
                    float(previous[common_index[neighbor]][0] - previous[center][0]),
                ),
                neighbor,
            ),
        )
        new_order = sorted(
            retained,
            key=lambda neighbor: (
                math.atan2(
                    float(aligned[common_index[neighbor]][1] - aligned[center][1]),
                    float(aligned[common_index[neighbor]][0] - aligned[center][0]),
                ),
                neighbor,
            ),
        )
        new_rank = {neighbor: rank for rank, neighbor in enumerate(new_order)}
        inversions = sum(
            new_rank[left] > new_rank[right]
            for left_index, left in enumerate(old_order)
            for right in old_order[left_index + 1 :]
        )
        pair_count = len(retained) * (len(retained) - 1) // 2
        losses.append((1.0 - input_churn) * inversions / pair_count)
    return losses


def _temporal_identity_losses(
    before: Scene,
    after: Scene,
    identifiers: List[str],
    before_by_id: Dict[str, int],
    after_by_id: Dict[str, int],
    common: List[str],
) -> List[float]:
    """Measure enter/exit distance to the nearest visible common-node anchor.

    Parameters
    ----------
    before, after : Scene
        Consecutive validated frames.
    identifiers : list[str]
        Declared entering or exiting temporal ids.
    before_by_id, after_by_id : dict[str, int]
        Temporal-id to frame-local index mappings.
    common : list[str]
        Common temporal ids eligible as anchors.

    Returns
    -------
    list[float]
        Smooth too-near/too-far loss per entering or exiting identity.
    """

    losses = []
    for identifier in identifiers:
        frame = after if identifier in after_by_id else before
        mapping = after_by_id if identifier in after_by_id else before_by_id
        node = mapping[identifier]
        adjacency_by_node: List[List[int]] = [[] for _ in range(frame.node_count)]
        for source, target in frame.graph.edges:
            adjacency_by_node[source].append(target)
            adjacency_by_node[target].append(source)
        distances = {node: 0}
        queue = deque([node])
        while queue:
            current = queue.popleft()
            for target in sorted(adjacency_by_node[current]):
                if target not in distances:
                    distances[target] = distances[current] + 1
                    queue.append(target)
        anchor_id = min(
            common,
            key=lambda item: (distances.get(mapping[item], math.inf), item),
        )
        distance = float(
            torch.linalg.vector_norm(frame.positions[node] - frame.positions[mapping[anchor_id]])
        )
        normalized = distance / frame.intrinsic_unit
        near = 1.0 / (1.0 + math.exp(max(-60.0, min(60.0, (normalized - 0.10) / 0.03))))
        far = 1.0 / (1.0 + math.exp(max(-60.0, min(60.0, (2.0 - normalized) / 0.20))))
        losses.append(near + far - near * far)
    return losses
