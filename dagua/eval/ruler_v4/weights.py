"""Declared edge-weight distance, order, and visual-encoding facets."""

# Contract identity docstrings intentionally keep each title and digest on one line.
# ruff: noqa: E501

from __future__ import annotations

import math
from collections import defaultdict
from typing import DefaultDict, List

import torch

from dagua.eval.ruler_v4._util import correlation_defect, graph_distances, midranks, pair_values
from dagua.eval.ruler_v4.scene import FacetResult, Scene, na_result, value_result

_DISTANCE_SEMANTICS = frozenset({"distance_cost", "length_target"})
_LOCAL_ORDER_SEMANTICS = frozenset({"distance_cost", "strength", "capacity", "importance"})


def U35(scene: Scene) -> FacetResult:
    """Weighted distance fidelity. Frozen SHA-256: 7644f1fef1bcede526f923da3bc70e24005a0c28296369300eec9cc9d4743810."""

    if scene.graph.edge_weights is None:
        return na_result("NO_DECLARED_EDGE_WEIGHTS")
    if scene.graph.weight_semantics not in _DISTANCE_SEMANTICS:
        return na_result("WEIGHT_SEMANTICS_NOT_DISTANCE")
    graph_order, layout = pair_values(scene, graph_distances(scene, weighted=True))
    if graph_order.numel() < 3:
        return na_result("TOO_FEW_WEIGHTED_PAIRS")
    defect = correlation_defect(midranks(graph_order), midranks(layout))
    return value_result(defect, {"U35.headline": defect}, {"pair_count": graph_order.numel()})


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
    weights = torch.tensor(scene.graph.edge_weights, dtype=torch.float64)
    burdens = []
    strength_like = scene.graph.weight_semantics in {"strength", "capacity", "importance"}
    for edge_indices in incident.values():
        for left_index, left in enumerate(edge_indices):
            for right in edge_indices[left_index + 1 :]:
                declared_delta = float(weights[left] - weights[right])
                if declared_delta == 0.0:
                    continue
                layout_delta = float(lengths[left] - lengths[right])
                expected_sign = -1.0 if strength_like else 1.0
                signed = expected_sign * declared_delta * layout_delta
                burdens.append(1.0 / (1.0 + math.exp(max(-60.0, min(60.0, signed)))))
    if not burdens:
        return na_result("NO_LOCAL_WEIGHT_COMPARISONS")
    defect = sum(burdens) / len(burdens)
    return value_result(defect, {"U36.headline": defect}, {"comparison_count": len(burdens)})


def U37(scene: Scene) -> FacetResult:
    """Thickness-only weight encoding. Frozen SHA-256: ad46f1330b123bd7941b47e9e49cd00b59803c0705b2d2432bda4da81ece53da."""

    if scene.graph.edge_weights is None:
        return na_result("NO_DECLARED_EDGE_WEIGHTS")
    return na_result("NO_DECLARED_PER_EDGE_STROKE_WIDTHS")
