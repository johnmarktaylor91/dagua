"""Declared edge-weight distance, order, and visual-encoding facets."""

# Contract identity docstrings intentionally keep each title and digest on one line.
# ruff: noqa: E501

from __future__ import annotations

import math
from collections import defaultdict
from typing import DefaultDict, List

import torch

from dagua.eval.ruler_v4._util import graph_distances, isotonic_stress, pair_values
from dagua.eval.ruler_v4.scene import FacetResult, Scene, na_result, value_result

_DISTANCE_SEMANTICS = frozenset({"distance_cost", "connection_strength"})
_LOCAL_ORDER_SEMANTICS = frozenset({"distance_cost", "connection_strength"})


def U35(scene: Scene) -> FacetResult:
    """Weighted distance fidelity. Frozen SHA-256: 7644f1fef1bcede526f923da3bc70e24005a0c28296369300eec9cc9d4743810."""

    if scene.graph.edge_weights is None:
        return na_result("NO_DECLARED_EDGE_WEIGHTS")
    if scene.graph.weight_semantics not in _DISTANCE_SEMANTICS:
        return na_result("WEIGHT_SEMANTICS_NOT_DISTANCE")
    declared = torch.tensor(scene.graph.edge_weights, dtype=torch.float64)
    if scene.graph.weight_semantics == "connection_strength":
        costs = torch.median(declared) / declared
    else:
        costs = declared
    weighted_graph = replace_edge_weights(scene, tuple(float(value) for value in costs))
    weighted_order, layout = pair_values(
        weighted_graph, graph_distances(weighted_graph, weighted=True)
    )
    unweighted_order, _ = pair_values(scene, graph_distances(scene))
    if weighted_order.numel() < 3:
        return na_result("TOO_FEW_WEIGHTED_PAIRS")
    stress_one = isotonic_stress(unweighted_order, layout)
    stress_weighted = isotonic_stress(weighted_order, layout)
    logs = torch.log(costs)
    median = torch.median(logs)
    coefficient = (
        1.4826 * float(torch.median(torch.abs(logs - median))) / (abs(float(median)) + 1.0)
    )
    alpha = coefficient / (coefficient + 0.10)
    defect = (1.0 - alpha) * stress_one + alpha * stress_weighted
    return value_result(
        defect,
        {"U35.headline": defect},
        {"pair_count": weighted_order.numel(), "alpha": alpha},
    )


def U36(scene: Scene) -> FacetResult:
    """Local weight monotonicity. Frozen SHA-256: 574b04b859567036912ca0815385311a2524dca0cb89ff0bda27acbb4c0ed703."""

    if scene.graph.edge_weights is None:
        return na_result("NO_DECLARED_EDGE_WEIGHTS")
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
    burdens = []
    for edge_indices in incident.values():
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
    if not burdens:
        return na_result("NO_LOCAL_WEIGHT_COMPARISONS")
    defect = sum(burdens) / len(burdens)
    return value_result(defect, {"U36.headline": defect}, {"comparison_count": len(burdens)})


def U37(scene: Scene) -> FacetResult:
    """Thickness-only weight encoding. Frozen SHA-256: ad46f1330b123bd7941b47e9e49cd00b59803c0705b2d2432bda4da81ece53da."""

    if scene.graph.edge_weights is None:
        return na_result("NO_DECLARED_EDGE_WEIGHTS")
    return na_result("NO_DECLARED_PER_EDGE_STROKE_WIDTHS")


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
