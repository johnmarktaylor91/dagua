"""Declared edge-weight distance, order, and visual-encoding facets."""

# Contract identity docstrings intentionally keep each title and digest on one line.
# ruff: noqa: E501

from __future__ import annotations

import math
from collections import defaultdict
from typing import DefaultDict, List

import torch

from dagua.eval.ruler_v4._util import global_blend, graph_distances, primary_isotonic_fit, snap_unit
from dagua.eval.ruler_v4.scene import FacetResult, Scene, na_result, value_result
from dagua.eval.ruler_v4.structure import _distance_strata, _stress_from_fit

_DISTANCE_SEMANTICS = frozenset({"distance_cost", "connection_strength"})
_LOCAL_ORDER_SEMANTICS = frozenset({"distance_cost", "connection_strength"})


def U35(scene: Scene) -> FacetResult:
    """Weighted distance fidelity. Frozen SHA-256: 7644f1fef1bcede526f923da3bc70e24005a0c28296369300eec9cc9d4743810."""

    if scene.graph.edge_weights is None:
        return na_result("WEIGHTS_ABSENT")
    if scene.graph.weight_semantics not in _DISTANCE_SEMANTICS:
        return na_result("WEIGHT_SEMANTICS_NOT_DISTANCE")
    declared = torch.tensor(scene.graph.edge_weights, dtype=torch.float64)
    if scene.graph.weight_semantics == "connection_strength":
        costs = torch.median(declared) / declared
    else:
        costs = declared
    weighted_graph = replace_edge_weights(scene, tuple(float(value) for value in costs))
    if not any(len(members) >= 3 for members in _component_members(scene)):
        return na_result("NO_THREE_NODE_WEIGHTED_COMPONENT")
    unweighted_strata = _distance_strata(scene)
    weighted_strata = _distance_strata(
        weighted_graph, graph_distances(weighted_graph, weighted=True)
    )
    logs = torch.log(costs)
    median = torch.median(logs)
    coefficient = (
        1.4826 * float(torch.median(torch.abs(logs - median))) / (abs(float(median)) + 1.0)
    )
    alpha = coefficient / (coefficient + 0.10)
    combined_stress = []
    stratum_weights = []
    for component_index in sorted({item[0] for item in unweighted_strata}):
        unweighted_component = [item for item in unweighted_strata if item[0] == component_index]
        weighted_component = [item for item in weighted_strata if item[0] == component_index]
        unweighted_order = torch.cat([item[2] for item in unweighted_component])
        weighted_order = torch.cat([item[2] for item in weighted_component])
        layout = torch.cat([item[3] for item in unweighted_component])
        fit_unweighted = primary_isotonic_fit(unweighted_order, layout)
        fit_weighted = primary_isotonic_fit(weighted_order, layout)
        cursor = 0
        for unweighted_item, weighted_item in zip(unweighted_component, weighted_component):
            count = unweighted_item[2].numel()
            stress_one = _stress_from_fit(
                unweighted_item[2],
                unweighted_item[3],
                fit_unweighted[cursor : cursor + count],
            )
            stress_weighted = _stress_from_fit(
                weighted_item[2],
                weighted_item[3],
                fit_weighted[cursor : cursor + count],
            )
            combined_stress.append((1.0 - alpha) * stress_one + alpha * stress_weighted)
            stratum_weights.append(float(count))
            cursor += count
    defect = global_blend(combined_stress, stratum_weights)
    return value_result(
        defect,
        {"U35.headline": defect},
        {"pair_count": int(sum(stratum_weights)), "alpha": alpha},
    )


def _component_members(scene: Scene) -> List[List[int]]:
    """Return connected components without introducing another public dependency.

    Parameters
    ----------
    scene : Scene
        Validated graph scene.

    Returns
    -------
    list[list[int]]
        Canonically ordered component memberships.
    """

    from dagua.eval.ruler_v4._util import components

    return components(scene)


def U36(scene: Scene) -> FacetResult:
    """Local weight monotonicity. Frozen SHA-256: 574b04b859567036912ca0815385311a2524dca0cb89ff0bda27acbb4c0ed703."""

    if scene.graph.edge_weights is None:
        return na_result("WEIGHTS_ABSENT")
    if scene.graph.weight_semantics not in _LOCAL_ORDER_SEMANTICS:
        return na_result("WEIGHT_SEMANTICS_NOT_LOCAL_ORDER")
    incident: DefaultDict[int, List[int]] = defaultdict(list)
    for edge_index, (source, target) in enumerate(scene.graph.edges):
        incident[source].append(edge_index)
        incident[target].append(edge_index)
    edges = torch.tensor(scene.graph.edges, dtype=torch.long)
    lengths = torch.linalg.vector_norm(
        scene.positions[edges[:, 0]] - scene.positions[edges[:, 1]], dim=1
    )
    strengths = torch.tensor(scene.graph.edge_weights, dtype=torch.float64)
    if scene.graph.weight_semantics == "distance_cost":
        strengths = 1.0 / strengths
    node_defects = []
    node_weights = []
    comparison_count = 0
    for edge_indices in incident.values():
        burdens = []
        for left_index, left in enumerate(edge_indices):
            for right in edge_indices[left_index + 1 :]:
                if strengths[left] == strengths[right]:
                    continue
                strong, weak = (
                    (left, right) if strengths[left] > strengths[right] else (right, left)
                )
                denominator = float(lengths[strong] + lengths[weak])
                margin = (
                    float(lengths[strong] - lengths[weak]) / denominator
                    if denominator > 0.0
                    else 0.0
                )
                scaled = max(-60.0, min(60.0, margin / 0.03))
                burdens.append(1.0 / (1.0 + math.exp(-scaled)))
        if burdens:
            node_defects.append(snap_unit(sum(burdens) / len(burdens)))
            node_weights.append(float(len(burdens)))
            comparison_count += len(burdens)
    if not node_defects:
        return na_result("NO_LOCAL_WEIGHT_ORDER")
    defect = global_blend(node_defects, node_weights)
    return value_result(
        defect,
        {"U36.headline": defect},
        {"comparison_count": comparison_count, "node_count": len(node_defects)},
    )


def U37(scene: Scene) -> FacetResult:
    """Thickness-only weight encoding. Frozen SHA-256: ad46f1330b123bd7941b47e9e49cd00b59803c0705b2d2432bda4da81ece53da."""

    if scene.graph.edge_weights is None or scene.graph.weight_visual_channel != "stroke_thickness":
        return na_result("THICKNESS_ENCODING_NOT_DECLARED")
    weights = torch.tensor(scene.graph.edge_weights, dtype=torch.float64)
    knots = torch.tensor(scene.graph.weight_encoding_knots, dtype=torch.float64)
    widths = torch.tensor(scene.style.edge_stroke_widths, dtype=torch.float64)
    target_widths = _log_linear_targets(weights, knots)
    log_error = torch.log(widths / target_widths)
    edge_losses = 1.0 - torch.exp(-((log_error / 0.10) ** 2))
    order = torch.argsort(weights, stable=True)
    order_losses = []
    for left, right in zip(order[:-1].tolist(), order[1:].tolist()):
        if weights[left] == weights[right]:
            continue
        argument = float((torch.log(widths[left]) - torch.log(widths[right])) / 0.02)
        argument = max(-60.0, min(60.0, argument))
        order_losses.append(1.0 / (1.0 + math.exp(-argument)))
    per_edge = snap_unit(float(torch.mean(edge_losses)))
    if order_losses:
        order_loss = snap_unit(sum(order_losses) / len(order_losses))
        defect = snap_unit(0.75 * per_edge + 0.25 * order_loss)
    else:
        order_loss = 0.0
        defect = per_edge
    return value_result(
        defect,
        {"U37.ell_e": per_edge, "U37.ell_ord": order_loss},
        {
            "target_widths": tuple(float(value) for value in target_widths),
            "derived_widths": tuple(float(value) for value in widths),
            "concordance_count": len(order_losses),
        },
    )


def _log_linear_targets(weights: torch.Tensor, knots: torch.Tensor) -> torch.Tensor:
    """Evaluate a positive piecewise-linear map in log-weight/log-width space.

    Parameters
    ----------
    weights : torch.Tensor
        Positive declared edge weights with shape ``[E]``.
    knots : torch.Tensor
        Positive monotone ``(weight, width)`` knots with shape ``[K, 2]``.

    Returns
    -------
    torch.Tensor
        Target widths with shape ``[E]``.
    """

    if knots.shape[0] == 1:
        return torch.full_like(weights, float(knots[0, 1]))
    log_weights = torch.log(weights)
    log_knot_weights = torch.log(knots[:, 0])
    log_knot_widths = torch.log(knots[:, 1])
    right = torch.searchsorted(log_knot_weights, log_weights, right=True)
    right = torch.clamp(right, min=1, max=knots.shape[0] - 1)
    left = right - 1
    fraction = (log_weights - log_knot_weights[left]) / (
        log_knot_weights[right] - log_knot_weights[left]
    )
    fraction = torch.clamp(fraction, 0.0, 1.0)
    return torch.exp(
        log_knot_widths[left] + fraction * (log_knot_widths[right] - log_knot_widths[left])
    )


def replace_edge_weights(scene: Scene, weights: tuple[float, ...]) -> Scene:
    """Return a scene with normalized graph-side distance costs.

    Parameters
    ----------
    scene : Scene
        Validated weighted scene.
    weights : tuple[float, ...]
        Positive normalized costs.

    Returns
    -------
    Scene
        Shallow immutable copy with replaced GraphSemantics weights.
    """

    from dataclasses import replace

    return replace(scene, graph=replace(scene.graph, edge_weights=weights))
