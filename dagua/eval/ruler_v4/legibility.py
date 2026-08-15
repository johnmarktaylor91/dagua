"""Primitive clearance, label, resolution, scale, and frame-economy facets."""

# Contract identity docstrings intentionally keep each title and digest on one line.
# ruff: noqa: E501

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch

from dagua.eval.ruler_v4._util import (
    ALPHA_GRID,
    aabb_pair,
    adjacency,
    components,
    global_blend,
    mean_result,
    proper_intersection,
    resolved_routes,
    selected_alpha_grid_offset,
    smoothstep,
    snap_unit,
    soft_pos,
)
from dagua.eval.ruler_v4.frames import (
    declared_content_area,
    overflow_defect,
    point_hull_area,
    robust_core_mask,
    robust_core_positions,
    robust_frame,
)
from dagua.eval.ruler_v4.scene import BoxGeometry, FacetResult, Scene, na_result, value_result


def U17(scene: Scene, alpha_grid_index: Optional[int]) -> FacetResult:
    """Node-node occlusion / clearance. Frozen SHA-256: c4b70fd5614c1c66e1e96cb62e41fbe87803319f5b50f96b4bf8bbc2cc74a322.

    Parameters
    ----------
    scene : Scene
        Validated canonical scene.
    alpha_grid_index : int or None
        Required shared-grid state. A manifest row index selects a value;
        ``None`` publishes the preregistered unselected envelope.

    Returns
    -------
    FacetResult
        Selected scalar value or typed unselected-grid envelope.
    """

    selected_offset = selected_alpha_grid_offset(alpha_grid_index)
    if scene.node_count < 2:
        return na_result("insufficient_node_population")
    node_masses = (
        list(scene.graph.node_masses)
        if scene.graph.node_masses is not None
        else [1.0] * scene.node_count
    )
    graph_adjacency = adjacency(scene)
    diagonals = [
        float(2.0 * torch.linalg.vector_norm(box.half_extents).item()) for box in scene.node_boxes
    ]
    budgets = []
    for node, neighbors in enumerate(graph_adjacency):
        degree = len(neighbors)
        maximum_neighbor = max((diagonals[item] for item in neighbors), default=0.0)
        radius = 1.5 * (diagonals[node] / 2.0 + maximum_neighbor / 2.0)
        demand = sum(diagonals[item] for item in neighbors)
        raw_budget = (2.0 * math.pi * radius - demand) / max(degree, 1)
        if degree == 0:
            raw_budget = 1.5 * diagonals[node] / 2.0
        budgets.append(max(raw_budget, 0.125 * scene.intrinsic_unit))
    grid = tuple((alpha_clear, alpha_high) for _, alpha_clear, alpha_high in ALPHA_GRID)
    per_grid_nodes: List[List[float]] = [[0.0] * scene.node_count for _ in grid]
    survival: List[List[float]] = [[1.0] * scene.node_count for _ in grid]
    overlap_count = 0
    for left in range(scene.node_count):
        for right in range(left + 1, scene.node_count):
            signed, overlap = aabb_pair(scene.node_boxes[left], scene.node_boxes[right])
            overlap_count += int(overlap > 0.0)
            pair_budget = min(budgets[left], budgets[right])
            floor = 0.25 * scene.intrinsic_unit
            logistic_argument = (pair_budget - floor) / (0.5 * floor)
            floor_blend = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, logistic_argument))))
            overlap_area = overlap * min(
                float(4.0 * torch.prod(scene.node_boxes[left].half_extents).item()),
                float(4.0 * torch.prod(scene.node_boxes[right].half_extents).item()),
            )
            absolute = 1.0 - math.exp(
                -overlap_area / (0.05 * scene.intrinsic_unit * scene.intrinsic_unit)
            )
            if signed <= 0.0:
                excess = 1.0
            else:
                excess = 1.0 - float(
                    smoothstep(torch.tensor(signed / pair_budget, dtype=torch.float64))
                )
            for grid_index, (alpha_clear, alpha_high) in enumerate(grid):
                alpha = alpha_clear * (1.0 - floor_blend) + (alpha_clear * floor_blend * alpha_high)
                pair_defect = snap_unit(alpha * absolute + (1.0 - alpha) * excess)
                if overlap_area > 0.0:
                    left_area = float(4.0 * torch.prod(scene.node_boxes[left].half_extents))
                    left_effective = pair_defect * (0.5 + 0.5 * min(1.0, overlap_area / left_area))
                    # Frozen default order is ascending canonical index: the
                    # higher-index node overdraws its lower-index partner.
                    right_effective = pair_defect * 0.5
                else:
                    left_effective = pair_defect
                    right_effective = pair_defect
                survival[grid_index][left] *= 1.0 - left_effective
                survival[grid_index][right] *= 1.0 - right_effective
    envelope_values = []
    for grid_index in range(len(grid)):
        for node in range(scene.node_count):
            per_grid_nodes[grid_index][node] = 1.0 - survival[grid_index][node]
        envelope_values.append(global_blend(per_grid_nodes[grid_index], node_masses))
    lower = min(envelope_values)
    upper = max(envelope_values)
    raw = {
        "grid_envelope": (lower, upper),
        "grid_values": tuple(envelope_values),
        "pair_count": scene.node_count * (scene.node_count - 1) // 2,
        "overlap_count": overlap_count,
        "upper_subterms": {"U17.1": upper},
    }
    if selected_offset is None:
        return na_result("alpha_grid_unselected", raw)
    selected_name = ALPHA_GRID[selected_offset][0]
    selected_value = envelope_values[selected_offset]
    raw.update({"alpha_grid_index": selected_offset + 1, "alpha_grid_name": selected_name})
    return value_result(selected_value, {"U17.1": selected_value}, raw)


def U18(scene: Scene, alpha_grid_index: Optional[int]) -> FacetResult:
    """Label legibility (node labels). Frozen SHA-256: 68e32c4e9edfac024c8d2e651538d5e41e0781134960f75ca06b131cb40b346d.

    Parameters
    ----------
    scene : Scene
        Validated canonical scene.
    alpha_grid_index : int or None
        Required shared-grid state. A manifest row index selects a value;
        ``None`` publishes the preregistered unselected envelope.

    Returns
    -------
    FacetResult
        Selected scalar value or typed unselected-grid envelope.
    """

    selected_offset = selected_alpha_grid_offset(alpha_grid_index)
    labels = scene.node_label_boxes
    if not labels:
        return na_result("no_declared_node_labels")
    graph_adjacency = adjacency(scene)
    node_diagonals = [
        float(2.0 * torch.linalg.vector_norm(box.half_extents)) for box in scene.node_boxes
    ]
    label_diagonals = {
        box.owner: float(2.0 * torch.linalg.vector_norm(box.half_extents)) for box in labels
    }
    budgets: List[float] = []
    for label in labels:
        owner = label.owner
        neighbors = graph_adjacency[owner]
        demand = sum(node_diagonals[node] for node in neighbors) + sum(
            label_diagonals[node] for node in neighbors if node in label_diagonals
        )
        maximum_effective = max(
            (max(node_diagonals[node], label_diagonals.get(node, 0.0)) for node in neighbors),
            default=0.0,
        )
        radius = 1.5 * (label_diagonals[owner] / 2.0 + maximum_effective / 2.0)
        raw_budget = (2.0 * math.pi * radius - demand) / max(len(neighbors), 1)
        if not neighbors:
            raw_budget = 1.5 * label_diagonals[owner] / 2.0
        budgets.append(max(raw_budget, 0.125 * scene.intrinsic_unit))
    grid = tuple((alpha_clear, alpha_high) for _, alpha_clear, alpha_high in ALPHA_GRID)
    owner_to_index = {label.owner: index for index, label in enumerate(labels)}
    survival = [[[1.0, 1.0, 1.0] for _ in labels] for _ in grid]
    pair_counts = [0, 0, 0]

    def add_pair(
        label_index: int,
        class_index: int,
        signed_clearance: float,
        overlap_area: float,
        pair_budget: float,
        occluded_fraction: Optional[float] = None,
    ) -> None:
        """Accumulate one pair debt into every shared alpha-grid row.

        Parameters
        ----------
        label_index : int
            Owning label-object index.
        class_index : int
            Pair-set index: label-label, label-node, or label-edge.
        signed_clearance : float
            Positive separation or nonpositive contact/penetration.
        overlap_area : float
            Exact or ribbon-coverage overlap area in scene units squared.
        pair_budget : float
            Positive input-only clearance target.
        occluded_fraction : float or None
            Visible-order fraction in ``[0, 1]`` for an intersecting pair. ``None``
            denotes a disjoint clearance pair, which retains full burden.
        """

        floor = 0.25 * scene.intrinsic_unit
        argument = (pair_budget - floor) / (0.5 * floor)
        floor_blend = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, argument))))
        absolute = 1.0 - math.exp(
            -overlap_area / (0.05 * scene.intrinsic_unit * scene.intrinsic_unit)
        )
        excess = (
            1.0
            if signed_clearance <= 0.0
            else 1.0
            - float(smoothstep(torch.tensor(signed_clearance / pair_budget, dtype=torch.float64)))
        )
        for grid_index, (alpha_clear, alpha_high) in enumerate(grid):
            alpha = alpha_clear * (1.0 - floor_blend) + (alpha_clear * floor_blend * alpha_high)
            defect = snap_unit(alpha * absolute + (1.0 - alpha) * excess)
            if occluded_fraction is not None:
                defect *= 0.5 + 0.5 * occluded_fraction
            survival[grid_index][label_index][class_index] *= 1.0 - defect
        pair_counts[class_index] += 1

    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1 :]:
            right_index = owner_to_index[right.owner]
            signed, overlap = aabb_pair(left, right)
            overlap_area = overlap * min(
                float(4.0 * torch.prod(left.half_extents)),
                float(4.0 * torch.prod(right.half_extents)),
            )
            budget = min(budgets[left_index], budgets[right_index])
            left_fraction = min(1.0, overlap_area / float(4.0 * torch.prod(left.half_extents)))
            add_pair(
                left_index,
                0,
                signed,
                overlap_area,
                budget,
                left_fraction if overlap_area > 0.0 else None,
            )
            add_pair(
                right_index,
                0,
                signed,
                overlap_area,
                budget,
                0.0 if overlap_area > 0.0 else None,
            )
        for node in scene.node_boxes:
            if node.owner == left.owner:
                continue
            signed, overlap = aabb_pair(left, node)
            overlap_area = overlap * min(
                float(4.0 * torch.prod(left.half_extents)),
                float(4.0 * torch.prod(node.half_extents)),
            )
            add_pair(
                left_index,
                1,
                signed,
                overlap_area,
                budgets[left_index],
                0.0 if overlap_area > 0.0 else None,
            )
        for route in resolved_routes(scene):
            if left.owner in scene.graph.edges[route.edge_index]:
                continue
            centerline_clearance = min(
                _box_segment_clearance(left, start, end)
                for start, end in zip(route.points[:-1], route.points[1:])
            )
            stroke_width = (
                scene.style.edge_stroke_widths[route.edge_index]
                if scene.style.edge_stroke_widths
                else scene.style.route_stroke_width * scene.style.coordinate_scale
            )
            signed = centerline_clearance - stroke_width / 2.0
            overlap_area = _route_box_overlap_area(route.points, left, stroke_width)
            edge_budget = min(budgets[left_index], stroke_width / 2.0 + 0.25 * scene.intrinsic_unit)
            add_pair(
                left_index,
                2,
                signed,
                overlap_area,
                edge_budget,
                0.0 if overlap_area > 0.0 else None,
            )
    label_masses = (
        [float(scene.graph.node_masses[label.owner]) for label in labels]
        if scene.graph.node_masses is not None
        else [1.0] * len(labels)
    )
    envelope_values: List[float] = []
    subterms_by_grid: List[dict[str, float]] = []
    for row in survival:
        class_node_defects = [
            [1.0 - row[label_index][class_index] for label_index in range(len(labels))]
            for class_index in range(3)
        ]
        values: dict[str, float] = {}
        for class_index, key in enumerate(("U18.ll", "U18.ln", "U18.le")):
            if pair_counts[class_index] > 0:
                values[key] = global_blend(class_node_defects[class_index], label_masses)
        label_defects = []
        active_weights = [
            weight if pair_counts[index] > 0 else 0.0
            for index, weight in enumerate((0.40, 0.40, 0.20))
        ]
        for label_index in range(len(labels)):
            label_survival = 1.0
            for class_index, weight in enumerate(active_weights):
                if weight > 0.0:
                    label_survival *= row[label_index][class_index] ** weight
            label_defects.append(1.0 - label_survival)
        envelope_values.append(global_blend(label_defects, label_masses))
        subterms_by_grid.append(values)
    upper_index = max(range(len(envelope_values)), key=envelope_values.__getitem__)
    raw = {
        "grid_envelope": (min(envelope_values), max(envelope_values)),
        "grid_values": tuple(envelope_values),
        "upper_subterms": subterms_by_grid[upper_index],
        "label_count": len(labels),
        "pair_counts": tuple(pair_counts),
    }
    if selected_offset is None:
        return na_result("alpha_grid_unselected", raw)
    raw.update(
        {
            "alpha_grid_index": selected_offset + 1,
            "alpha_grid_name": ALPHA_GRID[selected_offset][0],
        }
    )
    return value_result(envelope_values[selected_offset], subterms_by_grid[selected_offset], raw)


def _route_box_overlap_area(points: torch.Tensor, box: BoxGeometry, stroke_width: float) -> float:
    """Return ribbon-centerline coverage area inside an axis-aligned box.

    Parameters
    ----------
    points : torch.Tensor
        Polyline vertices with shape ``[P, 2]``.
    box : BoxGeometry
        Label obstacle box.
    stroke_width : float
        Positive route ribbon width.

    Returns
    -------
    float
        Covered centerline length times width, capped by box area.
    """

    lower = box.center - box.half_extents
    upper = box.center + box.half_extents
    inside_length = 0.0
    for start, end in zip(points[:-1], points[1:]):
        direction = end - start
        entry = 0.0
        exit_ = 1.0
        intersects = True
        for axis in range(2):
            delta = float(direction[axis])
            if delta == 0.0:
                if float(start[axis]) < float(lower[axis]) or float(start[axis]) > float(
                    upper[axis]
                ):
                    intersects = False
                    break
                continue
            first = (float(lower[axis]) - float(start[axis])) / delta
            second = (float(upper[axis]) - float(start[axis])) / delta
            entry = max(entry, min(first, second))
            exit_ = min(exit_, max(first, second))
            if entry > exit_:
                intersects = False
                break
        if intersects:
            inside_length += (exit_ - entry) * float(torch.linalg.vector_norm(direction))
    return min(
        inside_length * stroke_width,
        float(4.0 * torch.prod(box.half_extents)),
    )


def U19(scene: Scene) -> FacetResult:
    """Text legibility feasibility. Frozen SHA-256: b7166c90068923a4edb1bcd4dad3371ebb7dd9134d7e8f657afa9b2ae2a7e2da."""

    physical = scene.style.physical_output
    if physical is None:
        return na_result("no_declared_physical_size")
    if not scene.node_label_boxes and not scene.edge_label_boxes:
        return na_result("no_declared_labels")
    frame = robust_frame(scene.positions, scene.intrinsic_unit)
    content_width = 2.0 * float(frame.half_extents[0])
    content_height = 2.0 * float(frame.half_extents[1])
    physical_scale = min(
        physical["output_width"] / content_width,
        physical["output_height"] / content_height,
    )
    ratio = physical["h_font"] * physical_scale / physical["h_floor"]
    label_defect = 1.0 - float(smoothstep(torch.tensor(ratio, dtype=torch.float64)))
    defects = [label_defect] * (len(scene.node_label_boxes) + len(scene.edge_label_boxes))
    defect = global_blend(defects)
    return value_result(
        defect,
        {"U19.headline": defect},
        {
            "m_phys": physical_scale,
            "r_l": ratio,
            "slack": math.log(physical_scale / (physical["h_floor"] / physical["h_font"])),
            "label_count": len(defects),
        },
    )


def U20a(scene: Scene) -> FacetResult:
    """Resolution-limit degeneracy. Frozen SHA-256: 65e179ba53966fecc9fb77c4c219e05086bee5dac536540ab46bab785313b530."""

    if scene.node_count < 2:
        return na_result("insufficient_node_population")
    node_masses = (
        list(scene.graph.node_masses)
        if scene.graph.node_masses is not None
        else [1.0] * scene.node_count
    )
    feature_floor = scene.style.minimum_feature_separation * scene.intrinsic_unit
    coincidence_survival = [1.0] * scene.node_count
    node_opportunities = scene.node_count * (scene.node_count - 1) // 2
    node_feature_survival = {("node", node): 1.0 for node in range(scene.node_count)}
    for left in range(scene.node_count):
        for right in range(left + 1, scene.node_count):
            signed, _ = aabb_pair(scene.node_boxes[left], scene.node_boxes[right])
            clearance = max(0.0, signed)
            loss = 1.0 - float(
                smoothstep(torch.tensor(clearance / feature_floor, dtype=torch.float64))
            )
            coincidence_survival[left] *= 1.0 - loss
            coincidence_survival[right] *= 1.0 - loss
            _accumulate_object_loss(
                node_feature_survival,
                (("node", left), ("node", right)),
                loss,
                node_opportunities,
            )
    coincidence = global_blend([1.0 - value for value in coincidence_survival], node_masses)
    frame = robust_frame(scene.positions, scene.intrinsic_unit)
    core_mask = robust_core_mask(scene.positions)
    core = scene.positions[core_mask]
    if core.shape[0] < 2:
        rank_collapse = 1.0
    else:
        # A zero raw cloud is total collapse: the degenerate limit of exact
        # collinearity, which section 6 maps to the catastrophic end (q = 0).
        ratio = _isotropy_quotient(core - frame.center, degenerate=0.0)
        if scene.graph.ranks is not None and scene.graph.flow_axis is not None:
            # E2 is exemption-only: the residual frame may excuse collinearity
            # that the declared rank axis explains, but never lowers the raw
            # quotient of a healthy drawing -- otherwise declaring ranks would
            # exempt cross-axis line collapse, the E3-banned channel. A zero
            # RESIDUAL is exempt only when the ranks explain nonzero axis
            # variance; a fully coincident core stays catastrophic (golden 2).
            residual, explained_spread = _rank_residual_core(scene, core, core_mask)
            exempt = 1.0 if explained_spread > 0.0 else 0.0
            ratio = max(ratio, _isotropy_quotient(residual, degenerate=exempt))
        rank_collapse = 1.0 - float(smoothstep(torch.tensor(ratio / 0.05, dtype=torch.float64)))
    routes = resolved_routes(scene)
    node_route_opportunities = scene.node_count * scene.edge_count
    node_route_survival = {
        **{("node", node): 1.0 for node in range(scene.node_count)},
        **{("route", edge): 1.0 for edge in range(scene.edge_count)},
    }
    for box in scene.node_boxes:
        for route in routes:
            if box.owner in scene.graph.edges[route.edge_index]:
                continue
            clearance = min(
                _box_segment_clearance(box, start, end)
                for start, end in zip(route.points[:-1], route.points[1:])
            )
            width = (
                scene.style.edge_stroke_widths[route.edge_index]
                if scene.style.edge_stroke_widths
                else scene.style.route_stroke_width * scene.style.coordinate_scale
            )
            clearance = max(0.0, clearance - width / 2.0)
            loss = 1.0 - float(
                smoothstep(torch.tensor(clearance / feature_floor, dtype=torch.float64))
            )
            _accumulate_object_loss(
                node_route_survival,
                (("node", box.owner), ("route", route.edge_index)),
                loss,
                node_route_opportunities,
            )
    route_route_opportunities = scene.edge_count * (scene.edge_count - 1) // 2
    route_route_survival = {("route", edge): 1.0 for edge in range(scene.edge_count)}
    for left, route_left in enumerate(routes):
        for route_right in routes[left + 1 :]:
            clearances = [
                _segment_segment_distance(start_left, end_left, start_right, end_right)
                for left_index, (start_left, end_left) in enumerate(
                    zip(route_left.points[:-1], route_left.points[1:])
                )
                for right_index, (start_right, end_right) in enumerate(
                    zip(route_right.points[:-1], route_right.points[1:])
                )
                if not _segments_are_adjacent_at_declared_endpoint(
                    scene,
                    route_left.edge_index,
                    left_index,
                    route_left.points.shape[0] - 1,
                    route_right.edge_index,
                    right_index,
                    route_right.points.shape[0] - 1,
                )
            ]
            if not clearances:
                continue
            clearance = min(clearances)
            left_width = (
                scene.style.edge_stroke_widths[route_left.edge_index]
                if scene.style.edge_stroke_widths
                else scene.style.route_stroke_width * scene.style.coordinate_scale
            )
            right_width = (
                scene.style.edge_stroke_widths[route_right.edge_index]
                if scene.style.edge_stroke_widths
                else scene.style.route_stroke_width * scene.style.coordinate_scale
            )
            clearance = max(0.0, clearance - (left_width + right_width) / 2.0)
            loss = 1.0 - float(
                smoothstep(torch.tensor(clearance / feature_floor, dtype=torch.float64))
            )
            _accumulate_object_loss(
                route_route_survival,
                (("route", route_left.edge_index), ("route", route_right.edge_index)),
                loss,
                route_route_opportunities,
            )
    class_defects = [
        _object_feature_blend(node_feature_survival, node_masses),
        _object_feature_blend(node_route_survival, node_masses),
        _object_feature_blend(route_route_survival, node_masses),
    ]
    feature_survival = 1.0
    for class_defect, exponent in zip(class_defects, (0.5, 0.3, 0.2)):
        feature_survival *= (1.0 - class_defect) ** exponent
    feature_separation = 1.0 - feature_survival
    values = {
        "U20a.i": coincidence,
        "U20a.ii": rank_collapse,
        "U20a.iii": feature_separation,
    }
    return mean_result("U20a", values)


def _isotropy_quotient(centered: torch.Tensor, degenerate: float) -> float:
    """Singular-value quotient of one centered retained-position cloud.

    Parameters
    ----------
    centered : torch.Tensor
        Centered positions with shape ``[K, 2]``.
    degenerate : float
        Quotient assigned to a zero cloud, whose ratio is undefined: the
        collapse reading (``0.0``) for the raw frame, the fully-explained
        exemption (``1.0``) for a rank residual with explained variance.

    Returns
    -------
    float
        ``sigma_2 / sigma_1`` in ``[0, 1]``, or ``degenerate`` for a zero cloud.
    """

    singular = torch.linalg.svdvals(centered)
    if float(singular[0]) <= 0.0:
        return degenerate
    return float(singular[1] / singular[0])


def _rank_residual_core(
    scene: Scene,
    core: torch.Tensor,
    core_mask: torch.Tensor,
) -> Tuple[torch.Tensor, float]:
    """Remove rank-explained declared-axis variance from retained positions.

    Parameters
    ----------
    scene : Scene
        Validated scene with declared ranks and a declared flow axis.
    core : torch.Tensor
        Retained positions with shape ``[K, 2]``.
    core_mask : torch.Tensor
        Retained-node mask with shape ``[N]``.

    Returns
    -------
    tuple[torch.Tensor, float]
        Centered rank-residual positions with shape ``[K, 2]`` and the
        centered spread of the rank-explained axis component (zero exactly
        when the declared ranks explain no axis variance at all).
    """

    axis = torch.tensor(scene.graph.flow_axis, dtype=torch.float64)
    cross = torch.tensor([-axis[1], axis[0]], dtype=torch.float64)
    ranks = torch.tensor(scene.graph.ranks, dtype=torch.long)[core_mask]
    axis_projection = core @ axis
    explained = torch.empty_like(axis_projection)
    for rank in torch.unique(ranks, sorted=True):
        members = ranks == rank
        explained[members] = torch.mean(axis_projection[members])
    residual_axis = axis_projection - explained
    cross_projection = core @ cross
    residual = residual_axis[:, None] * axis + cross_projection[:, None] * cross
    explained_spread = float(torch.linalg.vector_norm(explained - torch.mean(explained)))
    return residual - torch.mean(residual, dim=0), explained_spread


def _accumulate_object_loss(
    survival: Dict[Tuple[str, int], float],
    owners: Tuple[Tuple[str, int], Tuple[str, int]],
    loss: float,
    opportunity_count: int,
) -> None:
    """Accumulate one analytic-opportunity feature loss onto its owners.

    Parameters
    ----------
    survival : dict[tuple[str, int], float]
        Mutable per-feature-owner survival products.
    owners : tuple[tuple[str, int], tuple[str, int]]
        Two node/route owners of the admitted feature pair.
    loss : float
        Pair loss in ``[0, 1]``.
    opportunity_count : int
        Input-only analytic class opportunity count.
    """

    if opportunity_count <= 0:
        return
    factor = 0.0 if loss >= 1.0 else math.exp(math.log1p(-loss) / opportunity_count)
    for owner in owners:
        survival[owner] *= factor


def _object_feature_blend(
    survival: Dict[Tuple[str, int], float], node_masses: List[float]
) -> float:
    """Blend class losses after collapsing pairs onto feature-owning objects.

    Parameters
    ----------
    survival : dict[tuple[str, int], float]
        Per-node and per-route survival products.
    node_masses : list[float]
        Input-owned node masses; routes have unit mass.

    Returns
    -------
    float
        Global 3.7 blend over feature-owning objects.
    """

    if not survival:
        return 0.0
    owners = sorted(survival)
    defects = [1.0 - survival[owner] for owner in owners]
    masses = [node_masses[index] if kind == "node" else 1.0 for kind, index in owners]
    return global_blend(defects, masses)


def _segments_are_adjacent_at_declared_endpoint(
    scene: Scene,
    left_edge: int,
    left_segment: int,
    left_segment_count: int,
    right_edge: int,
    right_segment: int,
    right_segment_count: int,
) -> bool:
    """Return whether two route segments meet at one shared graph endpoint.

    Parameters
    ----------
    scene : Scene
        Validated routed scene.
    left_edge, right_edge : int
        Declared edge indices.
    left_segment, right_segment : int
        Zero-based flattened segment indices.
    left_segment_count, right_segment_count : int
        Number of segments in each route.

    Returns
    -------
    bool
        True only for the two terminal segments adjacent at a shared endpoint.
    """

    left = scene.graph.edges[left_edge]
    right = scene.graph.edges[right_edge]
    for endpoint in set(left) & set(right):
        left_touches = (endpoint == left[0] and left_segment == 0) or (
            endpoint == left[1] and left_segment == left_segment_count - 1
        )
        right_touches = (endpoint == right[0] and right_segment == 0) or (
            endpoint == right[1] and right_segment == right_segment_count - 1
        )
        if left_touches and right_touches:
            return True
    return False


def _point_segment_distance(point: torch.Tensor, start: torch.Tensor, end: torch.Tensor) -> float:
    """Return exact Euclidean distance from a point to a line segment.

    Parameters
    ----------
    point, start, end : torch.Tensor
        Two-dimensional coordinates.

    Returns
    -------
    float
        Nonnegative point-to-segment distance.
    """

    direction = end - start
    denominator = float(torch.dot(direction, direction).item())
    if denominator == 0.0:
        return float(torch.linalg.vector_norm(point - start).item())
    parameter = float(torch.dot(point - start, direction).item()) / denominator
    parameter = min(1.0, max(0.0, parameter))
    return float(torch.linalg.vector_norm(point - (start + parameter * direction)).item())


def _segment_segment_distance(
    start_a: torch.Tensor,
    end_a: torch.Tensor,
    start_b: torch.Tensor,
    end_b: torch.Tensor,
) -> float:
    """Return the Euclidean distance between two closed line segments.

    Parameters
    ----------
    start_a, end_a, start_b, end_b : torch.Tensor
        Segment endpoints with shape ``[2]``.

    Returns
    -------
    float
        Zero for intersecting segments, otherwise the nearest endpoint distance.
    """

    if proper_intersection(start_a, end_a, start_b, end_b):
        return 0.0
    return min(
        _point_segment_distance(start_a, start_b, end_b),
        _point_segment_distance(end_a, start_b, end_b),
        _point_segment_distance(start_b, start_a, end_a),
        _point_segment_distance(end_b, start_a, end_a),
    )


def _box_segment_clearance(box: BoxGeometry, start: torch.Tensor, end: torch.Tensor) -> float:
    """Return centerline clearance between an axis-aligned box and segment.

    Parameters
    ----------
    box : BoxGeometry
        Axis-aligned obstacle box.
    start, end : torch.Tensor
        Segment endpoints with shape ``[2]``.

    Returns
    -------
    float
        Zero for contact/intersection, otherwise Euclidean clearance.
    """

    center = box.center
    half_extents = box.half_extents
    direction = end - start
    lower = 0.0
    upper = 1.0
    for axis in range(2):
        low = float(center[axis] - half_extents[axis])
        high = float(center[axis] + half_extents[axis])
        delta = float(direction[axis])
        origin = float(start[axis])
        if delta == 0.0:
            if origin < low or origin > high:
                break
            continue
        first = (low - origin) / delta
        second = (high - origin) / delta
        lower = max(lower, min(first, second))
        upper = min(upper, max(first, second))
        if lower > upper:
            break
    else:
        if lower <= upper:
            return 0.0
    corners = [
        center + torch.tensor([sx, sy], dtype=torch.float64) * half_extents
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
    ]
    endpoint_clearances = []
    for point in (start, end):
        excess = torch.abs(point - center) - half_extents
        endpoint_clearances.append(
            float(torch.linalg.vector_norm(torch.clamp(excess, min=0.0)).item())
        )
    return min(
        *endpoint_clearances,
        *(_point_segment_distance(corner, start, end) for corner in corners),
    )


def U20b(scene: Scene) -> FacetResult:
    """Scale-legibility plateau. Frozen SHA-256: 8037cbb7308319db2434687b3b149b5ee6797a53f78eda9e81727ebe39531947."""

    if scene.profile.scale_normalized:
        return na_result("scale_normalized_profile")
    if scene.edge_count == 0:
        return na_result("no_edges")
    edges = torch.tensor(scene.graph.edges, dtype=torch.long)
    lengths = torch.linalg.vector_norm(
        scene.positions[edges[:, 0]] - scene.positions[edges[:, 1]], dim=1
    )
    normalized = lengths / scene.intrinsic_unit
    median_length = float(torch.median(normalized).item())
    if median_length == 0.0:
        defect = 1.0
    else:
        coordinate = math.log2(median_length)
        short_argument = (math.log2(1.5) - coordinate) / 0.35
        long_argument = (coordinate - math.log2(8.0)) / 0.35
        short = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, short_argument))))
        long = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, long_argument))))
        defect = short + long
    return value_result(defect, {"U20b.headline": defect}, {"median_edge_length_u": median_length})


def U21(scene: Scene) -> FacetResult:
    """Frame economy / anti-sprawl. Frozen SHA-256: c662b51acf5fa3863ad367627168b280cdc6eb23194f7577203d9b0bd899d4de."""

    if scene.node_count < 2:
        return na_result("insufficient_node_population")
    frame = robust_frame(scene.positions, scene.intrinsic_unit)
    primitive_area = sum(float(4.0 * torch.prod(box.half_extents)) for box in scene.node_boxes)
    component_count = len(components(scene))
    area_reference = primitive_area / 0.10 * (1.0 + 0.5 * (component_count - 1))
    sparse_raw = soft_pos(math.log(frame.area / (4.0 * area_reference)))
    sparse = sparse_raw / (1.0 + sparse_raw)
    mass_out, anchor, overflow = overflow_defect(scene, frame)
    content_area = declared_content_area(scene)
    hull_trim = point_hull_area(robust_core_positions(scene.positions))
    hull_full = point_hull_area(scene.positions)
    defect = 1.0 - (1.0 - sparse) * (1.0 - overflow)
    return value_result(
        defect,
        {"U21.d_sparse_n": sparse, "U21.d_overflow": overflow},
        {
            "frame_area": frame.area,
            "frame_regime": frame.regime,
            "trim_count": frame.trim_count,
            "Fr": (
                float(frame.center[0]),
                float(frame.center[1]),
                float(2.0 * frame.half_extents[0]),
                float(2.0 * frame.half_extents[1]),
            ),
            "D_core": float(torch.linalg.vector_norm(2.0 * frame.half_extents)),
            "A_hull_trim": hull_trim,
            "A_hull_full": hull_full,
            "iso": 1.0 - hull_trim / hull_full if hull_full > 0.0 else 0.0,
            "A_content": content_area,
            "phi_ink": content_area / frame.area,
            "R": frame.area / content_area if content_area > 0.0 else float("inf"),
            "mass_out": mass_out,
            "o_min": anchor,
            "overflow_anchor": anchor,
            "degenerate_frame_coincident": all(frame.floor_bound),
        },
    )
