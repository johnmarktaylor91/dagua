"""Declared-axis direction, rank, tree, flow, port, and temporal facets."""

# Contract identity docstrings intentionally keep each title and digest on one line.
# ruff: noqa: E501

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import DefaultDict, Dict, List, Optional, Tuple, Union

import torch

from dagua.eval.ruler_v4._util import bounded, declared_axis, mean_result
from dagua.eval.ruler_v4.scene import (
    FacetResult,
    Scene,
    TemporalScene,
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

    def visit(node: int) -> None:
        """Mark DFS back edges from one canonical node."""

        state[node] = 1
        for target, edge_index in sorted(outgoing[node]):
            if state[target] == 1:
                feedback[edge_index] = True
            elif state[target] == 0:
                visit(target)
        state[node] = 2

    for node in range(scene.node_count):
        if state[node] == 0:
            visit(node)
    return tuple(feedback)


def U31(scene: Scene) -> FacetResult:
    """Direction consistency. Frozen SHA-256: e8c67a87acbfb2a4882c39f54ff50e6c34238cabb5324fad2ab48b0fc5f99ef2."""

    axis = declared_axis(scene)
    if (
        axis is None
        or scene.graph.flow_axis is None
        or not scene.graph.directed
        or scene.edge_count == 0
    ):
        return na_result("no_declared_direction")
    feedback = _feedback_mask(scene)
    burdens = []
    for edge_index, (source, target) in enumerate(scene.graph.edges):
        if feedback[edge_index]:
            continue
        delta = scene.positions[target] - scene.positions[source]
        length = float(torch.linalg.vector_norm(delta))
        if length == 0.0:
            burdens.append(1.0)
        else:
            progress = float(torch.dot(delta, axis)) / length
            burdens.append(max(0.0, -progress))
    if not burdens:
        return na_result("no_forward_edges")
    defect = sum(burdens) / len(burdens)
    return value_result(
        defect,
        {"U31.headline": defect},
        {
            "feedback_edge_count": sum(feedback),
            "feedback_source": "declared" if scene.graph.feedback is not None else "U31-DFS-FB-1",
        },
    )


def U32(scene: Scene) -> FacetResult:
    """Rank/layer clarity. Frozen SHA-256: e1fdb3ffef119b8151d07ba8673778c88bb7bc565d596248220d2e828fe6eba4."""

    rank_values = scene.graph.ranks
    if rank_values is None:
        rank_values = _derived_ranks(scene)
    if rank_values is None:
        return na_result("NO_CANONICAL_DERIVED_RANK")
    axis = declared_axis(scene)
    if axis is None:
        return na_result("RANK_AXIS_ABSENT")
    ranks = torch.tensor(rank_values, dtype=torch.long)
    projection = scene.positions @ axis
    unique = torch.unique(ranks, sorted=True)
    centers = torch.stack([torch.mean(projection[ranks == rank]) for rank in unique])
    isotonic = 0.0
    if centers.numel() > 1:
        differences = centers[1:] - centers[:-1]
        isotonic = float(torch.mean(torch.clamp(-differences / scene.intrinsic_unit, min=0.0)))
    crisp = []
    for rank in unique:
        values = projection[ranks == rank]
        crisp.append(bounded(float(torch.std(values, unbiased=False)) / scene.intrinsic_unit))
    overlap = []
    for index in range(unique.numel() - 1):
        left = projection[ranks == unique[index]]
        right = projection[ranks == unique[index + 1]]
        overlap.append(
            math.exp(-abs(float(torch.mean(right) - torch.mean(left))) / scene.intrinsic_unit)
        )
    values = {
        "U32.L_iso": bounded(isotonic),
        "U32.L_crisp": sum(crisp) / len(crisp),
        "U32.L_overlap": sum(overlap) / len(overlap) if overlap else 0.0,
    }
    return mean_result("U32", values, {"rank_count": unique.numel()})


def _derived_ranks(scene: Scene) -> Optional[Tuple[int, ...]]:
    """Derive longest-path source ranks for a directed acyclic graph.

    Parameters
    ----------
    scene : Scene
        Directed graph scene without declared ranks.

    Returns
    -------
    tuple[int, ...] or None
        Canonical ranks, or ``None`` when the graph is cyclic or undirected.
    """

    if not scene.graph.directed:
        return None
    incoming = [0] * scene.node_count
    outgoing: DefaultDict[int, List[int]] = defaultdict(list)
    for source, target in scene.graph.edges:
        incoming[target] += 1
        outgoing[source].append(target)
    queue = deque(index for index, count in enumerate(incoming) if count == 0)
    ranks = [0] * scene.node_count
    visited = 0
    while queue:
        source = queue.popleft()
        visited += 1
        for target in sorted(outgoing[source]):
            ranks[target] = max(ranks[target], ranks[source] + 1)
            incoming[target] -= 1
            if incoming[target] == 0:
                queue.append(target)
    return tuple(ranks) if visited == scene.node_count else None


def U33(scene: Scene) -> FacetResult:
    """Tree quality bundle. Frozen SHA-256: b9ea391f1645e98497c216a76ba3c1b0d69901cdf5d54c42872376a7ea4e7feb."""

    if (
        not scene.graph.directed
        or not scene.graph.roots
        or scene.edge_count != scene.node_count - len(scene.graph.roots)
    ):
        return na_result("not_declared_tree")
    ranks = _tree_depths(scene)
    if ranks is None:
        return na_result("not_declared_tree")
    axis = declared_axis(scene)
    assert axis is not None
    projection = scene.positions @ axis
    cross = scene.positions @ torch.tensor([1.0, 0.0], dtype=torch.float64)
    parent_progress = []
    sibling_spread = []
    edge_variation = []
    child_centering = []
    children: DefaultDict[int, List[int]] = defaultdict(list)
    lengths = []
    for source, target in scene.graph.edges:
        children[source].append(target)
        parent_progress.append(
            max(0.0, float(projection[source] - projection[target]) / scene.intrinsic_unit)
        )
        lengths.append(
            float(torch.linalg.vector_norm(scene.positions[target] - scene.positions[source]))
        )
    for parent, child_nodes in children.items():
        child_values = cross[child_nodes]
        sibling_spread.append(
            bounded(float(torch.std(child_values, unbiased=False)) / scene.intrinsic_unit)
        )
        child_centering.append(
            bounded(abs(float(torch.mean(child_values) - cross[parent])) / scene.intrinsic_unit)
        )
    if lengths:
        length_tensor = torch.tensor(lengths)
        edge_variation.append(
            bounded(
                float(
                    torch.std(length_tensor, unbiased=False)
                    / torch.clamp(torch.mean(length_tensor), min=1e-12)
                )
            )
        )
    radial_error = []
    angle_error = []
    center = torch.mean(scene.positions[list(scene.graph.roots)], dim=0)
    for node, depth in enumerate(ranks):
        radius = (
            float(torch.linalg.vector_norm(scene.positions[node] - center)) / scene.intrinsic_unit
        )
        radial_error.append(bounded(abs(radius - depth)))
        if depth > 0:
            vector = scene.positions[node] - center
            angle_error.append(0.0 if float(torch.linalg.vector_norm(vector)) > 0.0 else 1.0)
    values = {
        "U33.layered.1": bounded(sum(parent_progress) / max(1, len(parent_progress))),
        "U33.layered.2": sum(sibling_spread) / max(1, len(sibling_spread)),
        "U33.layered.3": sum(edge_variation) / max(1, len(edge_variation)),
        "U33.layered.4": sum(child_centering) / max(1, len(child_centering)),
        "U33.radial.1": sum(radial_error) / len(radial_error),
        "U33.radial.2": sum(angle_error) / max(1, len(angle_error)),
    }
    return mean_result("U33", values)


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


def U34(scene: Scene) -> FacetResult:
    """Flow path traceability. Frozen SHA-256: 30aeff9b4e59ee355dbad183fb26686c084c426eb1430762b8e6f273e2fdb2cc."""

    axis = declared_axis(scene)
    if axis is None or not scene.graph.directed or not scene.routes:
        return na_result("no_declared_flow_paths")
    feedback = _feedback_mask(scene)
    back = []
    monotonic = []
    continuity = []
    for route in scene.routes:
        if feedback[route.edge_index]:
            continue
        deltas = route.points[1:] - route.points[:-1]
        progress = deltas @ axis
        back.append(
            float(
                torch.sum(torch.clamp(-progress, min=0.0))
                / torch.clamp(torch.sum(torch.abs(progress)), min=1e-12)
            )
        )
        monotonic.append(float(torch.mean((progress < 0.0).to(torch.float64))))
        if deltas.shape[0] > 1:
            directions = deltas / torch.clamp(
                torch.linalg.vector_norm(deltas, dim=1, keepdim=True), min=1e-12
            )
            continuity.append(
                float(torch.mean(1.0 - torch.sum(directions[1:] * directions[:-1], dim=1)) / 2.0)
            )
        else:
            continuity.append(0.0)
    if not back:
        return na_result("no_forward_flow_paths")
    values = {
        "U34.L_back": sum(back) / len(back),
        "U34.L_mono": sum(monotonic) / len(monotonic),
        "U34.L_cont": sum(continuity) / len(continuity),
    }
    return mean_result("U34", values)


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
        return na_result("no_declared_ports")
    route_by_edge = {route.edge_index: route for route in scene.routes}
    rows: Dict[str, List[float]] = {key: [] for key in ("U39.1", "U39.2", "U39.3", "U39.4")}
    for edge_index, ports in scene.graph.ports.items():
        route = route_by_edge.get(edge_index)
        if route is None:
            continue
        source_direction = _port_direction(ports[0])
        target_direction = _port_direction(ports[1])
        if source_direction is not None:
            actual = route.points[1] - route.points[0]
            actual = actual / torch.clamp(torch.linalg.vector_norm(actual), min=1e-12)
            rows["U39.1"].append((1.0 - float(torch.dot(actual, source_direction))) / 2.0)
            rows["U39.3"].append(
                bounded(
                    float(
                        torch.linalg.vector_norm(
                            route.points[0] - scene.positions[scene.graph.edges[edge_index][0]]
                        )
                    )
                    / scene.intrinsic_unit
                )
            )
        if target_direction is not None:
            actual = route.points[-2] - route.points[-1]
            actual = actual / torch.clamp(torch.linalg.vector_norm(actual), min=1e-12)
            rows["U39.2"].append((1.0 - float(torch.dot(actual, target_direction))) / 2.0)
            rows["U39.4"].append(
                bounded(
                    float(
                        torch.linalg.vector_norm(
                            route.points[-1] - scene.positions[scene.graph.edges[edge_index][1]]
                        )
                    )
                    / scene.intrinsic_unit
                )
            )
    if not any(rows.values()):
        return na_result("missing_port_routes")
    values = {key: sum(items) / len(items) if items else 0.0 for key, items in rows.items()}
    return mean_result("U39", values)


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
        displacement = sum(losses) / len(losses)
        churn = _temporal_angular_churn(before, after, common, aligned, previous)
        entering_or_exiting = [
            identifier
            for identifier, state in declaration.states.items()
            if state in {"enter", "exit"}
        ]
        identity = 0.0
        applicable = [(0.55, displacement), (0.25, churn)]
        if entering_or_exiting:
            identity = _temporal_identity_loss(
                before, after, entering_or_exiting, before_by_id, after_by_id, common
            )
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
    return value_result(
        defect,
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
    rotation = left @ right
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
) -> float:
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
    float
        Mean normalized inversion distance over scoreable common nodes.
    """

    if torch.allclose(aligned, previous, rtol=0.0, atol=1e-12):
        return 0.0
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
            continue
        center = common_index[identifier]
        inversions = 0
        pair_count = 0
        for left_index, left_id in enumerate(retained):
            for right_id in retained[left_index + 1 :]:
                left = common_index[left_id]
                right = common_index[right_id]
                old_left = previous[left] - previous[center]
                old_right = previous[right] - previous[center]
                new_left = aligned[left] - aligned[center]
                new_right = aligned[right] - aligned[center]
                old_cross = float(old_left[0] * old_right[1] - old_left[1] * old_right[0])
                new_cross = float(new_left[0] * new_right[1] - new_left[1] * new_right[0])
                inversions += int(old_cross * new_cross < 0.0)
                pair_count += 1
        losses.append((1.0 - input_churn) * inversions / pair_count)
    return sum(losses) / len(losses) if losses else 0.0


def _temporal_identity_loss(
    before: Scene,
    after: Scene,
    identifiers: List[str],
    before_by_id: Dict[str, int],
    after_by_id: Dict[str, int],
    common: List[str],
) -> float:
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
    float
        Mean smooth too-near/too-far identity loss.
    """

    losses = []
    for identifier in identifiers:
        frame = after if identifier in after_by_id else before
        mapping = after_by_id if identifier in after_by_id else before_by_id
        point = frame.positions[mapping[identifier]]
        anchors = frame.positions[[mapping[item] for item in common]]
        distance = float(torch.min(torch.linalg.vector_norm(anchors - point, dim=1)))
        normalized = distance / frame.intrinsic_unit
        near = 1.0 / (1.0 + math.exp(max(-60.0, min(60.0, (normalized - 0.10) / 0.03))))
        far = 1.0 / (1.0 + math.exp(max(-60.0, min(60.0, (2.0 - normalized) / 0.20))))
        losses.append(near + far - near * far)
    return sum(losses) / len(losses) if losses else 0.0
