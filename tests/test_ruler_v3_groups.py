"""Regression tests for V3 conditional ruler groups."""

from __future__ import annotations

import math
import signal
from typing import Any, Dict, List, Mapping, Optional, Tuple, cast

import pytest
import torch

from dagua.eval.drawing import routes_to_curves
from dagua.eval.ruler_v3 import renormalized_score, score_core_v3
from dagua.eval.ruler_v3_groups import CANONICAL_NODE_HEIGHT_REF, evaluate_conditional_groups
from dagua.metrics import (
    cluster_edge_intrusion_score,
    cluster_exclusion_score,
    cluster_label_occlusion_score,
    cluster_nesting_fidelity_score,
    cluster_sibling_overlap_score,
    composite_drawing,
)


def _sizes(count: int) -> torch.Tensor:
    """Return compact node boxes for ruler probes.

    Parameters
    ----------
    count : int
        Number of nodes.

    Returns
    -------
    torch.Tensor
        Node sizes with shape ``[N, 2]``.
    """
    return torch.full((count, 2), 0.2, dtype=torch.float64)


def _dag_probe() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
    """Create a declared top-bottom DAG flow probe.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]
        Positions, edge index, node sizes, and declared metadata.
    """
    pos = torch.tensor(
        [[0.0, 0.0], [0.0, 2.0], [0.0, 4.0], [0.0, 6.0]],
        dtype=torch.float64,
    )
    edges = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    return (
        pos,
        edges,
        _sizes(4),
        {
            "declared_hierarchical": True,
            "flow_direction": "TB",
            "topological_depth": [0, 1, 2, 3],
        },
    )


def _community_probe() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
    """Create a planted two-community probe.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]
        Positions, edge index, node sizes, and declared metadata.
    """
    pos = torch.tensor(
        [
            [-3.0, -0.2],
            [-3.1, 0.1],
            [-2.8, 0.0],
            [3.0, -0.1],
            [3.2, 0.2],
            [2.9, 0.1],
        ],
        dtype=torch.float64,
    )
    edges = torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]], dtype=torch.long)
    return pos, edges, _sizes(6), {"planted_partition": [0, 0, 0, 1, 1, 1]}


def _cluster_probe(
    scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
    """Create a declared two-cluster probe.

    Parameters
    ----------
    scale : float, optional
        Position-only scale multiplier.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]
        Positions, edge index, node sizes, and declared cluster metadata.
    """
    pos = scale * torch.tensor(
        [
            [-2.0, -0.2],
            [-2.2, 0.2],
            [-1.8, 0.0],
            [2.0, -0.2],
            [2.2, 0.2],
            [1.8, 0.0],
        ],
        dtype=torch.float64,
    )
    edges = torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]], dtype=torch.long)
    meta: Dict[str, object] = {
        "clusters": {"left": [0, 1, 2], "right": [3, 4, 5]},
        "cluster_labels": {"left": "Left", "right": "Right"},
    }
    return pos, edges, _sizes(6), meta


def _tree_probe(
    *,
    radial: bool = False,
    deformation: int = 0,
    scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
    """Create a declared rooted-tree probe.

    Parameters
    ----------
    radial : bool, optional
        Whether to return a radial drawing and convention.
    deformation : int, optional
        Progressive deformation level.
    scale : float, optional
        Position-only scale multiplier.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]
        Positions, edge index, node sizes, and declared tree metadata.
    """
    edges = torch.tensor([[0, 0, 0, 3, 3], [1, 2, 3, 4, 5]], dtype=torch.long)
    if radial:
        coords = [
            (0.0, 0.0),
            (1.5, 1.2),
            (0.0, 2.0),
            (-1.5, 1.2),
            (-1.8, 2.2),
            (-2.4, 1.8),
        ]
        if deformation == 1:
            coords[1] = (-0.4, 1.8)
            coords[4] = (1.0, 1.6)
        elif deformation >= 2:
            coords = [(0.0, 0.0), (1.0, 0.1), (0.9, 0.0), (0.8, -0.1), (0.7, 0.0), (0.6, 0.1)]
        convention = "radial"
    else:
        coords = [
            (0.0, 0.0),
            (-2.0, 1.0),
            (0.0, 1.0),
            (2.0, 1.0),
            (1.5, 2.0),
            (2.5, 2.0),
        ]
        if deformation == 1:
            coords[0] = (0.8, 0.0)
            coords[2] = (2.8, 1.0)
            coords[4] = (2.7, 1.3)
        elif deformation >= 2:
            coords = [(3.0, 2.0), (2.0, 1.0), (1.0, 0.5), (0.0, 0.0), (-1.0, 0.0), (-2.0, 0.0)]
        convention = "layered"
    meta: Dict[str, object] = {
        "declared_tree": True,
        "root": 0,
        "tree_convention": convention,
    }
    return scale * torch.tensor(coords, dtype=torch.float64), edges, _sizes(6), meta


def _weighted_star(lengths: List[float]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a weighted star drawing with declared edge lengths.

    Parameters
    ----------
    lengths : List[float]
        Leaf distances from the center.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Positions, edge index, and node sizes.
    """
    coords = [(0.0, 0.0)]
    for index, length in enumerate(lengths):
        coords.append((float(length), float(index) * 0.05))
    edges = torch.tensor(
        [[0 for _index in lengths], list(range(1, len(lengths) + 1))],
        dtype=torch.long,
    )
    return torch.tensor(coords, dtype=torch.float64), edges, _sizes(len(lengths) + 1)


def _temporal_probe(
    *,
    current_shift: float = 0.2,
    previous_quality: float = 0.98,
    graph_change: float = 0.042,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
    """Create a synthetic declared temporal probe.

    Parameters
    ----------
    current_shift : float, optional
        Non-similarity deformation applied to the current frame.
    previous_quality : float, optional
        Declared static V3-core score for the previous frame.
    graph_change : float, optional
        Declared ground-truth graph-change magnitude.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]
        Positions, edge index, node sizes, and temporal metadata.
    """
    previous = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]],
        dtype=torch.float64,
    )
    current = previous.clone()
    current[3, 0] += current_shift
    edges = torch.tensor([[0, 1, 2], [1, 3, 3]], dtype=torch.long)
    meta: Dict[str, object] = {
        "node_ids": ["a", "b", "c", "d"],
        "previous": {
            "positions": previous.tolist(),
            "node_ids": ["a", "b", "c", "d"],
            "quality": previous_quality,
            "best_static_v3_core": 1.0,
            "graph_change": graph_change,
        },
    }
    return current, edges, _sizes(4), meta


def _ported_probe(
    *,
    wrong_side: bool = False,
    routed: bool = False,
    deformation: int = 0,
    scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
    """Create a synthetic ported/routed probe.

    Parameters
    ----------
    wrong_side : bool, optional
        Whether to declare source ports on the opposite side.
    routed : bool, optional
        Whether to include declared route paths.
    deformation : int, optional
        Progressive route/port deformation level.
    scale : float, optional
        Position-only scale multiplier.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]
        Positions, edge index, node sizes, and port metadata.
    """
    pos = scale * torch.tensor(
        [[0.0, 0.0], [4.0, -1.0], [4.0, 1.0], [8.0, 0.0]],
        dtype=torch.float64,
    )
    edges = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]], dtype=torch.long)
    source_side = "W" if wrong_side else "E"
    ports: List[Dict[str, object]] = [
        {"edge": 0, "endpoint": "source", "side": source_side, "order": 0},
        {"edge": 1, "endpoint": "source", "side": source_side, "order": 1},
        {"edge": 0, "endpoint": "target", "side": "W"},
        {"edge": 1, "endpoint": "target", "side": "W"},
        {"edge": 2, "endpoint": "source", "side": "E"},
        {"edge": 3, "endpoint": "source", "side": "E"},
        {"edge": 2, "endpoint": "target", "side": "W", "order": 0},
        {"edge": 3, "endpoint": "target", "side": "W", "order": 1},
    ]
    meta: Dict[str, object] = {"ports": ports, "flow_direction": "LR"}
    if routed:
        meta["routing_declared"] = True
        route_paths: List[List[Tuple[float, float]]] = [
            [(0.0, 0.0), (2.0, -1.0), (4.0, -1.0)],
            [(0.0, 0.0), (2.0, 1.0), (4.0, 1.0)],
            [(4.0, -1.0), (6.0, -1.0), (8.0, 0.0)],
            [(4.0, 1.0), (6.0, 1.0), (8.0, 0.0)],
        ]
        if deformation == 1:
            route_paths[1] = [(0.0, 0.0), (2.0, -0.8), (4.0, 1.0)]
            route_paths[3] = [(4.0, 1.0), (5.0, 0.0), (7.0, 0.0), (8.0, 0.0)]
        elif deformation >= 2:
            route_paths = [
                [(0.0, 0.0), (-1.0, 0.0), (4.0, -1.0)],
                [(0.0, 0.0), (-1.0, 0.0), (4.0, 1.0)],
                [(4.0, -1.0), (4.0, 2.0), (8.0, 0.0)],
                [(4.0, 1.0), (4.0, -2.0), (8.0, 0.0)],
            ]
        meta["route_paths"] = [
            [(scale * x_value, scale * y_value) for x_value, y_value in route]
            for route in route_paths
        ]
        meta["routed_labels"] = ["a", "b", "c", "d"]
        meta["label_positions"] = [
            (scale * 2.0, scale * -1.0),
            (scale * 2.0, scale * 1.0),
            (scale * 6.0, scale * -1.0),
            (scale * 6.0, scale * 1.0),
        ]
    return pos, edges, _sizes(4), meta


def _facet_score(result: object, code: str) -> float:
    """Return a finite facet score from a V3 score result.

    Parameters
    ----------
    result : object
        Score result exposing a ``facets`` mapping.
    code : str
        Facet code.

    Returns
    -------
    float
        Facet score as a float.
    """
    score = result.facets[code].score  # type: ignore[attr-defined]
    assert score is not None
    return float(score)


G2_SLOT_A_FACETS = (
    "G2_cluster_exclusion",
    "G2_cluster_sibling_overlap",
    "G2_cluster_nesting_fidelity",
    "G2_cluster_edge_intrusion",
    "G2_cluster_label_occlusion",
)


def _canonical_unit_sizes(count: int) -> torch.Tensor:
    """Return default-corpus node boxes with the frozen canonical height.

    Parameters
    ----------
    count : int
        Number of nodes.

    Returns
    -------
    torch.Tensor
        Node sizes with shape ``[N, 2]`` and median height ``34.0``.
    """
    return torch.tensor([[44.0, CANONICAL_NODE_HEIGHT_REF]] * count, dtype=torch.float64)


def _continuous_exact_or_tiny_relative(base: float, scaled: float) -> None:
    """Assert continuous GG-5a equality with the documented fallback tolerance.

    Parameters
    ----------
    base : float
        Baseline facet value.
    scaled : float
        Scaled facet value.

    Returns
    -------
    None
    """
    if scaled == base:
        return
    assert abs(scaled - base) <= 1e-12 * max(1.0, abs(base))


def test_applicability_gates_are_input_only() -> None:
    """Assert G1, G3, and G6 gates fire only on declared metadata.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    dag_pos, dag_edges, dag_sizes, dag_meta = _dag_probe()
    community_pos, community_edges, community_sizes, community_meta = _community_probe()
    weighted_pos, weighted_edges, weighted_sizes = _weighted_star([1.0, 2.0, 4.0, 8.0])

    plain = evaluate_conditional_groups(dag_pos, dag_edges, dag_sizes, {})
    assert not plain["G1"].applicable
    assert not plain["G3"].applicable
    assert not plain["G6"].applicable
    assert not plain["G5"].applicable
    assert not plain["G7"].applicable

    g1 = evaluate_conditional_groups(dag_pos, dag_edges, dag_sizes, dag_meta)
    assert g1["G1"].applicable
    assert not g1["G3"].applicable
    assert not g1["G6"].applicable

    cluster_but_not_planted = {"clusters": [0, 0, 0, 1, 1, 1]}
    g3_absent = evaluate_conditional_groups(
        community_pos,
        community_edges,
        community_sizes,
        cluster_but_not_planted,
    )
    assert not g3_absent["G3"].applicable
    assert evaluate_conditional_groups(
        community_pos,
        community_edges,
        community_sizes,
        community_meta,
    )["G3"].applicable

    weighted_meta: Dict[str, object] = {
        "edge_weights": [1.0, 2.0, 4.0, 8.0],
        "weight_mode": "distance",
    }
    assert evaluate_conditional_groups(
        weighted_pos,
        weighted_edges,
        weighted_sizes,
        weighted_meta,
    )["G6"].applicable
    assert not evaluate_conditional_groups(
        weighted_pos,
        weighted_edges,
        weighted_sizes,
        {**weighted_meta, "weight_mode": "thickness-only"},
    )["G6"].applicable
    assert not evaluate_conditional_groups(
        weighted_pos,
        weighted_edges,
        weighted_sizes,
        {"edge_weights": [2.0, 2.0, 2.0, 2.0], "weight_mode": "distance"},
    )["G6"].applicable

    temporal_pos, temporal_edges, temporal_sizes, temporal_meta = _temporal_probe()
    assert evaluate_conditional_groups(
        temporal_pos,
        temporal_edges,
        temporal_sizes,
        temporal_meta,
    )["G5"].applicable
    assert not evaluate_conditional_groups(
        temporal_pos,
        temporal_edges,
        temporal_sizes,
        {"previous_positions": temporal_pos.tolist()},
    )["G5"].applicable

    port_pos, port_edges, port_sizes, port_meta = _ported_probe()
    assert evaluate_conditional_groups(port_pos, port_edges, port_sizes, port_meta)["G7"].applicable
    assert not evaluate_conditional_groups(port_pos, port_edges, port_sizes, {})["G7"].applicable

    cluster_pos, cluster_edges, cluster_sizes, cluster_meta = _cluster_probe()
    cluster_groups = evaluate_conditional_groups(
        cluster_pos,
        cluster_edges,
        cluster_sizes,
        cluster_meta,
    )
    assert cluster_groups["G2"].applicable
    assert not cluster_groups["G4"].applicable

    tree_pos, tree_edges, tree_sizes, tree_meta = _tree_probe()
    tree_groups = evaluate_conditional_groups(tree_pos, tree_edges, tree_sizes, tree_meta)
    assert tree_groups["G4"].applicable
    assert not tree_groups["G2"].applicable
    assert any(code.startswith("G4_layered_") for code in tree_groups["G4"].facets)

    radial_pos, radial_edges, radial_sizes, radial_meta = _tree_probe(radial=True)
    radial_groups = evaluate_conditional_groups(radial_pos, radial_edges, radial_sizes, radial_meta)
    assert radial_groups["G4"].applicable
    assert any(code.startswith("G4_radial_") for code in radial_groups["G4"].facets)


def test_doc4_no_impute_and_g1_weighted_renormalization() -> None:
    """Assert absent groups do not zero-fill and G1 joins by tier weights.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    pos, edges, sizes, meta = _dag_probe()
    core = score_core_v3(pos, edges, sizes)
    no_groups = score_core_v3(pos, edges, sizes, graph_meta={})
    assert no_groups.scores["tiered"] == pytest.approx(core.scores["tiered"])
    assert no_groups.coverage["applicable_groups"] == 0

    with_g1 = score_core_v3(pos, edges, sizes, graph_meta=meta)
    assert with_g1.coverage["applicable_groups"] == 1
    assert "G1_directed_flow" in with_g1.facets
    assert "G1_depth_order" in with_g1.facets
    g1_slots = with_g1.metadata["conditional_groups"]["G1"].metadata["tier_slots"]
    assert tuple(slot["facet_name"] for slot in g1_slots) == ("G1_directed_flow",)
    assert with_g1.facets["G1_depth_order"].applicable is False
    assert with_g1.facets["G1_depth_order"].effective_weight == 0.0
    assert with_g1.facets["G1_depth_order"].metadata["diagnostic_only"] is True
    values = {code: facet.score for code, facet in with_g1.facets.items()}
    weights = {code: facet.effective_weight for code, facet in with_g1.facets.items()}
    assert with_g1.scores["tiered"] == pytest.approx(renormalized_score(values, weights))
    directed_only_weights = {
        code: weight for code, weight in weights.items() if code != "G1_depth_order"
    }
    assert with_g1.scores["tiered"] == pytest.approx(
        renormalized_score(values, directed_only_weights)
    )


def test_doc4_no_impute_for_g2_and_g4_rows() -> None:
    """Assert G2 and G4 groups join only when applicable.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    tree_pos, tree_edges, tree_sizes, tree_meta = _tree_probe()
    tree_result = score_core_v3(tree_pos, tree_edges, tree_sizes, graph_meta=tree_meta)
    assert tree_result.coverage["applicable_groups"] == 1
    assert any(code.startswith("G4_") for code in tree_result.facets)
    assert not any(code.startswith("G2_") for code in tree_result.facets)
    values = {code: facet.score for code, facet in tree_result.facets.items()}
    weights = {code: facet.effective_weight for code, facet in tree_result.facets.items()}
    assert tree_result.scores["tiered"] == pytest.approx(renormalized_score(values, weights))

    cluster_pos, cluster_edges, cluster_sizes, cluster_meta = _cluster_probe()
    cluster_result = score_core_v3(
        cluster_pos,
        cluster_edges,
        cluster_sizes,
        graph_meta=cluster_meta,
    )
    assert cluster_result.coverage["applicable_groups"] == 1
    assert any(code.startswith("G2_") for code in cluster_result.facets)
    assert not any(code.startswith("G4_") for code in cluster_result.facets)
    values = {code: facet.score for code, facet in cluster_result.facets.items()}
    weights = {code: facet.effective_weight for code, facet in cluster_result.facets.items()}
    assert cluster_result.scores["tiered"] == pytest.approx(renormalized_score(values, weights))


def test_doc4_no_impute_for_plain_graph_g5_g7_absent() -> None:
    """Assert plain rows do not impute temporal or port group facets.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    pos, edges, sizes, _meta = _dag_probe()
    result = score_core_v3(pos, edges, sizes, graph_meta={})
    assert result.coverage["applicable_groups"] == 0
    assert not any(code.startswith("G5_") for code in result.facets)
    assert not any(code.startswith("G7_") for code in result.facets)
    values = {code: facet.score for code, facet in result.facets.items()}
    weights = {code: facet.effective_weight for code, facet in result.facets.items()}
    assert result.scores["tiered"] == pytest.approx(renormalized_score(values, weights))


@pytest.mark.parametrize("alpha", [0.1, 10.0, 50.0])
def test_g3_and_g6_position_scale_invariance(alpha: float) -> None:
    """Assert G3 and G6 geometry facets are position-scale invariant.

    Parameters
    ----------
    alpha : float
        Position-only scale multiplier.

    Returns
    -------
    None
    """
    c_pos, c_edges, c_sizes, c_meta = _community_probe()
    base_g3 = score_core_v3(c_pos, c_edges, c_sizes, graph_meta=c_meta).facets["G3_community_ari"]
    scaled_g3 = score_core_v3(
        alpha * c_pos,
        c_edges,
        c_sizes,
        graph_meta=c_meta,
    ).facets["G3_community_ari"]
    assert scaled_g3.score == pytest.approx(base_g3.score, abs=1e-6)

    w_pos, w_edges, w_sizes = _weighted_star([1.0, 2.0, 4.0, 8.0])
    w_meta: Dict[str, object] = {
        "edge_weights": [1.0, 2.0, 4.0, 8.0],
        "weight_mode": "distance",
    }
    base = score_core_v3(w_pos, w_edges, w_sizes, graph_meta=w_meta)
    scaled = score_core_v3(alpha * w_pos, w_edges, w_sizes, graph_meta=w_meta)
    for code in ("G6_weighted_ksm", "G6_local_weight_monotonicity"):
        assert scaled.facets[code].score == pytest.approx(base.facets[code].score, abs=1e-6)


@pytest.mark.parametrize("alpha", [0.1, 10.0, 50.0])
def test_g2_g4_position_scale_invariance_and_size_anchoring(alpha: float) -> None:
    """Assert G4 geometry and G2 CQ-HAC are invariant while AABB terms move.

    Parameters
    ----------
    alpha : float
        Position-only scale multiplier.

    Returns
    -------
    None
    """
    tree_pos, tree_edges, tree_sizes, tree_meta = _tree_probe()
    base_tree = score_core_v3(tree_pos, tree_edges, tree_sizes, graph_meta=tree_meta)
    scaled_tree = score_core_v3(alpha * tree_pos, tree_edges, tree_sizes, graph_meta=tree_meta)
    for code, facet in base_tree.facets.items():
        if code.startswith("G4_") and code != "G4_layered_contour_separation":
            assert scaled_tree.facets[code].score == pytest.approx(facet.score, abs=1e-6)
    assert "axis_anchored" in str(
        base_tree.facets["G4_layered_parent_centering"].metadata["invariance"]
    )

    cluster_pos, cluster_edges, cluster_sizes, cluster_meta = _cluster_probe()
    base_cluster = score_core_v3(
        cluster_pos,
        cluster_edges,
        cluster_sizes,
        graph_meta=cluster_meta,
    )
    scaled_cluster = score_core_v3(
        alpha * cluster_pos,
        cluster_edges,
        cluster_sizes,
        graph_meta=cluster_meta,
    )
    assert scaled_cluster.facets["G2_cluster_hac_ari"].score == pytest.approx(
        base_cluster.facets["G2_cluster_hac_ari"].score,
        abs=1e-6,
    )
    assert "position_scale_sensitive" in str(
        base_cluster.facets["G2_cluster_compactness_log_band"].metadata["invariance"]
    )

    anchored_pos = torch.tensor(
        [[0.0, 0.0], [0.2, 0.0], [0.4, 0.0], [0.6, 0.0], [0.8, 0.0], [1.0, 0.0]],
        dtype=torch.float64,
    )
    anchored_edges = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
    anchored_sizes = _sizes(6)
    anchored_meta: Dict[str, object] = {
        "clusters": {"a": [0, 1, 2], "b": [3, 4, 5]},
        "cluster_labels": {"a": "A", "b": "B"},
    }
    base_anchored = score_core_v3(
        anchored_pos,
        anchored_edges,
        anchored_sizes,
        graph_meta=anchored_meta,
    )
    scaled_anchored = score_core_v3(
        alpha * anchored_pos,
        anchored_edges,
        anchored_sizes,
        graph_meta=anchored_meta,
    )
    moved_slot_a = any(
        abs(
            float(scaled_anchored.facets[code].score or 0.0)
            - float(base_anchored.facets[code].score or 0.0)
        )
        > 1e-6
        for code in (
            "G2_cluster_exclusion",
            "G2_cluster_sibling_overlap",
            "G2_cluster_edge_intrusion",
            "G2_cluster_label_occlusion",
        )
    )
    assert moved_slot_a

    compactness_pos = torch.tensor(
        [[-5.0, 0.0], [0.0, 0.0], [5.0, 0.0], [15.0, 0.0], [20.0, 0.0], [25.0, 0.0]],
        dtype=torch.float64,
    )
    base_compactness = score_core_v3(
        compactness_pos,
        anchored_edges,
        anchored_sizes,
        graph_meta=anchored_meta,
    )
    scaled_compactness = score_core_v3(
        alpha * compactness_pos,
        anchored_edges,
        anchored_sizes,
        graph_meta=anchored_meta,
    )
    compactness_moved = (
        abs(
            float(scaled_compactness.facets["G2_cluster_compactness_log_band"].score or 0.0)
            - float(base_compactness.facets["G2_cluster_compactness_log_band"].score or 0.0)
        )
        > 1e-6
    )
    assert compactness_moved


def test_g2_slot_a_unit_invariance_gg5a_alpha_battery() -> None:
    """Assert G2 Slot A is exact under joint unit scaling.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    pos, edges, sizes, meta = _cluster_probe()
    baseline = score_core_v3(pos, edges, sizes, graph_meta=meta)
    alphas = (0.02, 0.1, 1.0, 10.0, 50.0)
    for alpha in alphas:
        scaled = score_core_v3(alpha * pos, edges, alpha * sizes, graph_meta=meta)
        for code in G2_SLOT_A_FACETS:
            _continuous_exact_or_tiny_relative(
                _facet_score(baseline, code),
                _facet_score(scaled, code),
            )
        assert _facet_score(scaled, "G2_cluster_sibling_overlap") == 1.0


def test_g2_slot_a_two_engine_unit_fairness() -> None:
    """Assert identical cluster drawings at different units receive identical Slot A scores.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    pos, edges, sizes, meta = _cluster_probe()
    unit_1 = score_core_v3(pos, edges, sizes, graph_meta=meta)
    unit_250 = score_core_v3(250.0 * pos, edges, 250.0 * sizes, graph_meta=meta)
    for code in G2_SLOT_A_FACETS:
        assert unit_250.facets[code].score == unit_1.facets[code].score


def test_g2_cluster_exclusion_is_diagnostic_after_sec_2_3_audit() -> None:
    """Assert the redundant G2 exclusion facet is reported but unweighted.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    pos, edges, sizes, meta = _cluster_probe()
    result = score_core_v3(pos, edges, sizes, graph_meta=meta)
    exclusion = result.facets["G2_cluster_exclusion"]

    assert exclusion.score is not None
    assert exclusion.applicable is False
    assert exclusion.effective_weight == 0.0
    assert exclusion.metadata["diagnostic_only"] is True
    assert exclusion.metadata["audit_demotion"] == "sec_2_3_redundant_with_C4_clean_units_field"
    assert result.facets["G2_cluster_sibling_overlap"].effective_weight == pytest.approx(0.8)
    assert result.facets["G2_cluster_nesting_fidelity"].effective_weight == pytest.approx(0.4)
    assert result.facets["G2_cluster_edge_intrusion"].effective_weight == pytest.approx(0.4)
    assert result.facets["G2_cluster_label_occlusion"].effective_weight == pytest.approx(0.4)


def test_g2_slot_a_degenerate_intrinsic_ruler_guard() -> None:
    """Assert zero node heights pin ``s`` to 1 and raise the existing row flag.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    pos, edges, sizes, meta = _cluster_probe()
    zero_sizes = torch.zeros_like(sizes)
    groups = evaluate_conditional_groups(pos, edges, zero_sizes, meta)
    assert groups["G2"].metadata["flags"] == ("DEGENERATE_SCALE",)
    for code in G2_SLOT_A_FACETS:
        facet = groups["G2"].facets[code]
        assert facet.score is not None
        assert facet.metadata["intrinsic_ruler_scale"] == 1.0
        assert facet.metadata["intrinsic_ruler_degenerate"] is True


def test_g2_slot_a_projection_off_matches_frozen_metric_calls() -> None:
    """Assert canonical-unit Slot A wrappers are bit-identical to frozen metric calls.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    rows = [
        _cluster_probe(),
        (
            torch.tensor(
                [[-5.0, 0.0], [0.0, 0.0], [5.0, 0.0], [15.0, 0.0], [20.0, 0.0], [25.0, 0.0]],
                dtype=torch.float64,
            ),
            torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long),
            _sizes(6),
            {"clusters": {"a": [0, 1, 2], "b": [3, 4, 5]}, "cluster_labels": {"a": "A", "b": "B"}},
        ),
    ]
    for pos, edges, _sizes_in, meta in rows:
        sizes = _canonical_unit_sizes(int(pos.shape[0]))
        result = score_core_v3(pos, edges, sizes, graph_meta=meta)
        clusters = cast(Mapping[str, List[int]], meta["clusters"])
        parents: Mapping[str, Optional[str]] = {}
        labels = cast(Mapping[str, str], meta["cluster_labels"])
        direct = {
            "G2_cluster_exclusion": cluster_exclusion_score(pos, sizes, clusters, parents, labels)[
                "cluster_exclusion_score"
            ],
            "G2_cluster_sibling_overlap": cluster_sibling_overlap_score(
                pos, sizes, clusters, parents, labels
            )["cluster_sibling_overlap_score"],
            "G2_cluster_nesting_fidelity": cluster_nesting_fidelity_score(
                pos, sizes, clusters, parents, labels
            )["cluster_nesting_fidelity_score"],
            "G2_cluster_edge_intrusion": cluster_edge_intrusion_score(
                pos, edges, sizes, clusters, parents, labels
            )["cluster_edge_intrusion_score"],
            "G2_cluster_label_occlusion": cluster_label_occlusion_score(
                pos, sizes, clusters, parents, labels
            )["cluster_label_occlusion_score"],
        }
        for code, score in direct.items():
            assert result.facets[code].metadata["intrinsic_ruler_scale"] == 1.0
            assert result.facets[code].score == score


def test_g1_axis_anchored_declared_transform() -> None:
    """Assert G1 responds to the declared axis instead of rotation invariance.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    pos, edges, sizes, meta = _dag_probe()
    tb = score_core_v3(pos, edges, sizes, graph_meta=meta).facets["G1_directed_flow"]
    lr = score_core_v3(
        pos,
        edges,
        sizes,
        graph_meta={**meta, "flow_direction": "LR"},
    ).facets["G1_directed_flow"]
    assert tb.score is not None
    assert lr.score is not None
    assert tb.score > lr.score
    assert "axis_anchored" in str(tb.metadata["invariance"])


def test_g5_temporal_band_false_stability_and_scale_invariance() -> None:
    """Assert G5 band decay, false-stability penalty, and scale invariance.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    pos, edges, sizes, meta = _temporal_probe(current_shift=0.2, graph_change=0.042)
    within = score_core_v3(pos, edges, sizes, graph_meta=meta)
    within_facet = within.facets["G5_temporal_stability"]
    assert within_facet.score is not None
    assert within_facet.score > 0.5
    assert "STABILITY_BAND" not in within.flags

    below_meta = dict(meta)
    below_previous = cast(Mapping[str, object], meta["previous"])
    below_meta["previous"] = {**dict(below_previous), "quality": 0.2}
    below = score_core_v3(pos, edges, sizes, graph_meta=below_meta)
    below_score = below.facets["G5_temporal_stability"].score
    assert below_score is not None
    assert 0.0 < below_score < within_facet.score
    assert "STABILITY_BAND" in below.flags

    frozen_pos, frozen_edges, frozen_sizes, frozen_meta = _temporal_probe(
        current_shift=0.0,
        graph_change=0.102,
    )
    proportional_pos, proportional_edges, proportional_sizes, proportional_meta = _temporal_probe(
        current_shift=0.5,
        graph_change=0.102,
    )
    frozen = score_core_v3(
        frozen_pos,
        frozen_edges,
        frozen_sizes,
        graph_meta=frozen_meta,
    ).facets["G5_temporal_stability"]
    proportional = score_core_v3(
        proportional_pos,
        proportional_edges,
        proportional_sizes,
        graph_meta=proportional_meta,
    ).facets["G5_temporal_stability"]
    assert frozen.score is not None
    assert proportional.score is not None
    assert frozen.score < proportional.score

    alpha = 10.0
    scaled_meta = dict(meta)
    previous = dict(cast(Mapping[str, object], meta["previous"]))
    previous["positions"] = (alpha * torch.as_tensor(previous["positions"])).tolist()
    scaled_meta["previous"] = previous
    scaled = score_core_v3(alpha * pos, edges, sizes, graph_meta=scaled_meta)
    assert scaled.facets["G5_temporal_stability"].score == pytest.approx(
        within_facet.score,
        abs=1e-6,
    )


def test_g7_port_compliance_cap_routed_curves_and_scale_invariance() -> None:
    """Assert G7 hard compliance, cap, routed bundle, and scale invariance.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    pos, edges, sizes, meta = _ported_probe()
    result = score_core_v3(pos, edges, sizes, graph_meta=meta)
    compliance = result.facets["G7_port_hard_compliance"]
    assert compliance.score == pytest.approx(1.0)
    assert "PORT_VIOLATION" not in result.flags

    wrong_pos, wrong_edges, wrong_sizes, wrong_meta = _ported_probe(wrong_side=True)
    wrong = score_core_v3(wrong_pos, wrong_edges, wrong_sizes, graph_meta=wrong_meta)
    wrong_compliance = wrong.facets["G7_port_hard_compliance"]
    assert wrong_compliance.score is not None
    assert wrong_compliance.score < 1.0
    assert wrong_compliance.score <= 0.5
    assert wrong.metadata["conditional_groups"]["G7"].metadata["group_cap"] == pytest.approx(0.5)
    assert "PORT_VIOLATION" in wrong.flags

    routed_pos, routed_edges, routed_sizes, routed_meta = _ported_probe(routed=True)
    routed = score_core_v3(routed_pos, routed_edges, routed_sizes, graph_meta=routed_meta)
    assert "G7_routed_curve_quality" in routed.facets
    assert "G7_routed_bend_terminal_economy" in routed.facets
    routed_meta_record = routed.facets["G7_routed_curve_quality"].metadata
    assert routed_meta_record["routed_crossing_rate"] >= 0.0
    assert "composite_drawing_reuse" in routed_meta_record

    alpha = 20.0
    scaled_pos, scaled_edges, scaled_sizes, scaled_meta = _ported_probe(scale=alpha)
    scaled = score_core_v3(scaled_pos, scaled_edges, scaled_sizes, graph_meta=scaled_meta)
    assert scaled.facets["G7_port_hard_compliance"].score == pytest.approx(
        compliance.score,
        abs=1e-6,
    )


def test_g7_routed_quality_unit_invariance_gg5a_alpha_battery() -> None:
    """Assert G7 routed quality and hard compliance are exact under joint unit scaling.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    pos, edges, sizes, meta = _ported_probe(routed=True)
    baseline = score_core_v3(pos, edges, sizes, graph_meta=meta)
    for alpha in (0.02, 0.1, 1.0, 10.0, 50.0):
        scaled_pos, _scaled_edges, _scaled_sizes, scaled_meta = _ported_probe(
            routed=True,
            scale=alpha,
        )
        scaled = score_core_v3(scaled_pos, edges, alpha * sizes, graph_meta=scaled_meta)
        _continuous_exact_or_tiny_relative(
            _facet_score(baseline, "G7_port_hard_compliance"),
            _facet_score(scaled, "G7_port_hard_compliance"),
        )
        _continuous_exact_or_tiny_relative(
            _facet_score(baseline, "G7_routed_curve_quality"),
            _facet_score(scaled, "G7_routed_curve_quality"),
        )
        _continuous_exact_or_tiny_relative(
            _facet_score(baseline, "G7_routed_bend_terminal_economy"),
            _facet_score(scaled, "G7_routed_bend_terminal_economy"),
        )


def test_g7_routed_quality_projection_off_reuses_frozen_composite() -> None:
    """Assert canonical-unit G7 routed wrapper passes unchanged inputs to V2 composite.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    pos, edges, _sizes_in, meta = _ported_probe(routed=True)
    sizes = _canonical_unit_sizes(int(pos.shape[0]))
    result = score_core_v3(pos, edges, sizes, graph_meta=meta)
    routes = cast(List[List[Tuple[float, float]]], meta["route_paths"])
    curves = routes_to_curves(routes, pos, edges)
    assert curves is not None
    direct = composite_drawing(
        pos,
        edges,
        sizes,
        curves,
        label_positions=cast(List[Optional[Tuple[float, float]]], meta["label_positions"]),
        edge_labels=cast(List[Any], meta["routed_labels"]),
        seed=0,
    )
    reuse = result.facets["G7_routed_curve_quality"].metadata["composite_drawing_reuse"]
    assert result.facets["G7_routed_curve_quality"].metadata["intrinsic_ruler_scale"] == 1.0
    for key in (
        "drawing_term_crossing",
        "drawing_term_edge_node",
        "drawing_term_label_node",
        "drawing_term_label_label",
        "drawing_term_bend",
    ):
        assert reuse[key] == direct[key]


def test_g5_requires_declared_graph_change_and_ignores_best_static_metadata() -> None:
    """Probe G5 no-default ground truth and frozen band provenance.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    pos, edges, sizes, meta = _temporal_probe(current_shift=0.0, graph_change=0.5)
    previous = dict(cast(Mapping[str, object], meta["previous"]))
    previous.pop("graph_change", None)
    no_change_meta = {**meta, "previous": previous}
    assert not evaluate_conditional_groups(pos, edges, sizes, no_change_meta)["G5"].applicable

    thrash = pos.clone()
    thrash[0, 0] += 10.0
    thrash[3, 1] -= 5.0
    assert not evaluate_conditional_groups(thrash, edges, sizes, no_change_meta)["G5"].applicable

    frozen = score_core_v3(pos, edges, sizes, graph_meta=meta)
    assert _facet_score(frozen, "G5_temporal_stability") < 1e-6

    with_best_one = score_core_v3(pos, edges, sizes, graph_meta=meta)
    varied_previous = {
        **dict(cast(Mapping[str, object], meta["previous"])),
        "best_static_v3_core": 0.5,
    }
    with_best_half = score_core_v3(
        pos,
        edges,
        sizes,
        graph_meta={**meta, "previous": varied_previous},
    )
    assert _facet_score(with_best_half, "G5_temporal_stability") == pytest.approx(
        _facet_score(with_best_one, "G5_temporal_stability"),
        abs=1e-12,
    )


def test_g4_declared_cycle_through_root_returns_malformed() -> None:
    """Probe the G4 cycle-through-root hang guard.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    def timeout_handler(_signum: int, _frame: Any) -> None:
        """Raise when the malformed tree probe exceeds the alarm.

        Parameters
        ----------
        _signum : int
            Signal number.
        _frame : Any
            Current frame.

        Returns
        -------
        None
        """
        raise TimeoutError("G4 malformed-tree probe exceeded 10 seconds")

    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)
    try:
        pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float64)
        edges = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        meta: Dict[str, object] = {"declared_tree": True, "root": 0, "tree_convention": "layered"}
        result = evaluate_conditional_groups(pos, edges, _sizes(3), meta)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
    assert not result["G4"].applicable
    assert result["G4"].applicability_reason == "inapplicable:malformed_declared_tree"


def test_equal_view_weights_conditional_groups_by_slots() -> None:
    """Probe DOC-6 equal-view group slot aggregation.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    pos, edges, sizes, meta = _cluster_probe()
    row_meta = {
        **meta,
        "declared_hierarchical": True,
        "flow_direction": "TB",
        "topological_depth": [0, 1, 1, 2, 2, 3],
        "planted_partition": [0, 0, 0, 1, 1, 1],
    }
    result = score_core_v3(pos, edges, sizes, graph_meta=row_meta)
    from dagua.eval.ruler_v3 import _equal_view_weights

    weights = _equal_view_weights(result.facets)
    g2_share = sum(weight for code, weight in weights.items() if code.startswith("G2_")) / sum(
        weights.values()
    )
    assert g2_share < 0.2
    assert g2_share == pytest.approx(2.0 / 15.0)


def test_g4_radial_facets_pass_perfect_and_crater_on_sector_squeeze() -> None:
    """Probe the radial allocation, overlap, and circular-order fixes.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    edges = torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.long)
    radius = 3.0
    angles = [0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0]
    perfect = torch.tensor(
        [[0.0, 0.0], *[(radius * math.cos(a), radius * math.sin(a)) for a in angles]],
        dtype=torch.float64,
    )
    meta: Dict[str, object] = {"declared_tree": True, "root": 0, "tree_convention": "radial"}
    result = score_core_v3(perfect, edges, _sizes(4), graph_meta=meta)
    for code in (
        "G4_radial_angular_allocation",
        "G4_radial_angular_overlap",
        "G4_radial_circular_order",
    ):
        assert _facet_score(result, code) == pytest.approx(1.0, abs=1e-6)

    squeezed = torch.tensor(
        [[0.0, 0.0], [3.0, -0.05], [3.0, 0.0], [3.0, 0.05]],
        dtype=torch.float64,
    )
    squeezed_result = score_core_v3(squeezed, edges, _sizes(4), graph_meta=meta)
    assert _facet_score(squeezed_result, "G4_radial_angular_overlap") < 0.5


def test_g7_route_only_has_no_compliance_and_routed_facets_are_row_level() -> None:
    """Probe G7 route-only compliance and routed facet-set parity.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    pos = torch.tensor([[0.0, 0.0], [4.0, 0.0]], dtype=torch.float64)
    edges = torch.tensor([[0], [1]], dtype=torch.long)
    route_meta: Dict[str, object] = {
        "routing_declared": True,
        "route_paths": [[(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)]],
    }
    result = score_core_v3(pos, edges, _sizes(2), graph_meta=route_meta)
    assert "G7_port_hard_compliance" not in result.facets
    assert "G7_routed_curve_quality" in result.facets

    port_meta: Dict[str, object] = {
        "ports": [{"edge": 0, "endpoint": "source", "side": "E"}],
        "routing_declared": True,
    }
    with_routes = score_core_v3(pos, edges, _sizes(2), graph_meta={**port_meta, **route_meta})
    without_routes = score_core_v3(pos, edges, _sizes(2), graph_meta=port_meta)
    assert {code for code in with_routes.facets if code.startswith("G7_")} == {
        code for code in without_routes.facets if code.startswith("G7_")
    }


def test_g7_port_order_with_port_sides_fills_node_before_lookup() -> None:
    """Probe port_sides plus port_order declarations without geometry.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    pos = torch.tensor([[0.0, 0.0], [4.0, -1.0], [4.0, 1.0]], dtype=torch.float64)
    edges = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
    meta: Dict[str, object] = {
        "port_sides": {0: {"source": "E"}, 1: {"source": "E"}},
        "port_order": {(0, "E"): 0},
    }
    result = score_core_v3(pos, edges, _sizes(3), graph_meta=meta)
    compliance = result.facets["G7_port_hard_compliance"]
    assert compliance.metadata["order_checks"] == 0
    assert compliance.metadata["compliance_checks"] == 2


def test_g7_reversed_port_order_detected_with_resolvable_positions() -> None:
    """Probe port order violation when explicit port positions are declared.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    pos = torch.tensor([[0.0, 0.0], [4.0, -1.0], [4.0, 1.0]], dtype=torch.float64)
    edges = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
    meta: Dict[str, object] = {
        "ports": [
            {
                "edge": 0,
                "endpoint": "source",
                "side": "E",
                "order": 1,
                "position": (0.1, -1.0),
            },
            {
                "edge": 1,
                "endpoint": "source",
                "side": "E",
                "order": 0,
                "position": (0.1, 1.0),
            },
        ],
    }
    result = score_core_v3(pos, edges, _sizes(3), graph_meta=meta)
    compliance = result.facets["G7_port_hard_compliance"]
    assert compliance.metadata["order_checks"] == 2
    assert compliance.metadata["order_satisfied"] == 0
    assert compliance.score is not None and compliance.score < 1.0


def test_g4_layered_contour_separation_uses_node_sizes() -> None:
    """Probe size-aware sibling subtree contour overlap.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    pos = torch.tensor([[0.0, 0.0], [-0.1, 1.0], [0.1, 1.0]], dtype=torch.float64)
    edges = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
    sizes = torch.full((3, 2), 10.0, dtype=torch.float64)
    meta: Dict[str, object] = {"declared_tree": True, "root": 0, "tree_convention": "layered"}
    result = score_core_v3(pos, edges, sizes, graph_meta=meta)
    assert _facet_score(result, "G4_layered_contour_separation") < 1.0


def test_g1_back_edge_mask_excluded_from_acyclic_fraction() -> None:
    """Probe G1 frac_acyclic with declared feedback edges removed.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    pos = torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
    edges = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    meta: Dict[str, object] = {
        "declared_hierarchical": True,
        "flow_direction": "TB",
        "back_edge_mask": [False, False, True],
    }
    result = score_core_v3(pos, edges, _sizes(3), graph_meta=meta)
    facet = result.facets["G1_directed_flow"]
    assert facet.metadata["frac_acyclic"] == pytest.approx(1.0)
    assert facet.effective_weight > 0.0


def test_g7_exit_stub_does_not_satisfy_wrong_route_direction() -> None:
    """Probe side compliance past a short cosmetic exit stub.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    pos = torch.tensor([[0.0, 0.0], [4.0, -1.0]], dtype=torch.float64)
    edges = torch.tensor([[0], [1]], dtype=torch.long)
    meta: Dict[str, object] = {
        "ports": [{"edge": 0, "endpoint": "source", "side": "E"}],
        "routing_declared": True,
        "route_paths": [[(0.0, 0.0), (0.05, 0.0), (-6.0, 0.0), (4.0, -1.0)]],
    }
    result = score_core_v3(pos, edges, _canonical_unit_sizes(2), graph_meta=meta)
    compliance = result.facets["G7_port_hard_compliance"]
    assert compliance.metadata["side_violation_count"] == 1
    assert compliance.score is not None and compliance.score < 1.0


def test_g7_exit_stub_detected_at_realistic_units() -> None:
    """Probe exit-stub side violations across realistic joint unit scales.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    base_pos = torch.tensor([[0.0, 0.0], [4.0, -1.0]], dtype=torch.float64)
    edges = torch.tensor([[0], [1]], dtype=torch.long)
    base_route = [(0.0, 0.0), (0.05, 0.0), (-6.0, 0.0), (4.0, -1.0)]
    for alpha in (1.0, 34.0, 250.0):
        meta: Dict[str, object] = {
            "ports": [{"edge": 0, "endpoint": "source", "side": "E"}],
            "routing_declared": True,
            "route_paths": [
                [(alpha * x_value, alpha * y_value) for x_value, y_value in base_route]
            ],
        }
        result = score_core_v3(
            alpha * base_pos,
            edges,
            alpha * _canonical_unit_sizes(2),
            graph_meta=meta,
        )
        compliance = result.facets["G7_port_hard_compliance"]
        assert compliance.metadata["side_violation_count"] == 1
        assert compliance.score is not None and compliance.score < 1.0


def test_g4_single_child_parent_centering_unit_invariance_gg5a() -> None:
    """Assert single-child parent centering is exact under joint unit scaling.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    base_pos = torch.tensor([[0.0, 0.0], [0.5, 1.0]], dtype=torch.float64)
    edges = torch.tensor([[0], [1]], dtype=torch.long)
    meta: Dict[str, object] = {"declared_tree": True, "root": 0, "tree_convention": "layered"}
    baseline = score_core_v3(base_pos, edges, _canonical_unit_sizes(2), graph_meta=meta)
    for alpha in (0.02, 0.1, 1.0, 10.0, 50.0):
        scaled = score_core_v3(
            alpha * base_pos,
            edges,
            alpha * _canonical_unit_sizes(2),
            graph_meta=meta,
        )
        _continuous_exact_or_tiny_relative(
            _facet_score(baseline, "G4_layered_parent_centering"),
            _facet_score(scaled, "G4_layered_parent_centering"),
        )


def test_g4_sibling_order_preserves_edge_insertion_order() -> None:
    """Probe sibling order against declared edge order rather than node id.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    pos = torch.tensor([[0.0, 0.0], [1.0, 1.0], [-1.0, 1.0]], dtype=torch.float64)
    edges = torch.tensor([[0, 0], [2, 1]], dtype=torch.long)
    meta: Dict[str, object] = {"declared_tree": True, "root": 0, "tree_convention": "layered"}
    result = score_core_v3(pos, edges, _sizes(3), graph_meta=meta)
    assert _facet_score(result, "G4_layered_sibling_order") == pytest.approx(1.0)


def test_deformation_monotonicity_smoke() -> None:
    """Assert G1, G3, and G6 decay under progressive deformations.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    pos, edges, sizes, meta = _dag_probe()
    g1_positions = [
        pos,
        torch.tensor([[0.0, 0.0], [0.0, 2.0], [0.0, 1.0], [0.0, 3.0]], dtype=torch.float64),
        torch.tensor([[0.0, 3.0], [0.0, 2.0], [0.0, 1.0], [0.0, 0.0]], dtype=torch.float64),
    ]
    g1_scores = [
        _facet_sum(score_core_v3(step, edges, sizes, graph_meta=meta).facets, ("G1_directed_flow",))
        for step in g1_positions
    ]
    _assert_nonincreasing(g1_scores)

    c_pos, c_edges, c_sizes, c_meta = _community_probe()
    g3_positions = [
        c_pos,
        torch.tensor(
            [
                [-3.0, -0.2],
                [-3.1, 0.1],
                [2.8, 0.0],
                [3.0, -0.1],
                [3.2, 0.2],
                [-2.9, 0.1],
            ],
            dtype=torch.float64,
        ),
        torch.tensor(
            [
                [-3.0, -0.2],
                [3.1, 0.1],
                [-2.8, 0.0],
                [3.0, -0.1],
                [-3.2, 0.2],
                [2.9, 0.1],
            ],
            dtype=torch.float64,
        ),
    ]
    g3_scores = [
        _facet_sum(
            score_core_v3(step, c_edges, c_sizes, graph_meta=c_meta).facets,
            ("G3_community_ari",),
        )
        for step in g3_positions
    ]
    _assert_nonincreasing(g3_scores)

    w_edges_meta: Dict[str, object] = {
        "edge_weights": [1.0, 2.0, 4.0, 8.0],
        "weight_mode": "distance",
    }
    g6_scores = []
    for lengths in ([1.0, 2.0, 4.0, 8.0], [1.0, 2.0, 8.0, 4.0], [8.0, 4.0, 2.0, 1.0]):
        w_pos, w_edges, w_sizes = _weighted_star(list(lengths))
        result = score_core_v3(w_pos, w_edges, w_sizes, graph_meta=w_edges_meta)
        g6_scores.append(
            _facet_sum(
                result.facets,
                ("G6_weighted_ksm", "G6_local_weight_monotonicity"),
            )
        )
    _assert_nonincreasing(g6_scores)

    g2_scores = []
    for step in (
        _cluster_probe()[0],
        torch.tensor(
            [[-2.0, -0.2], [-2.2, 0.2], [1.9, 0.0], [2.0, -0.2], [2.2, 0.2], [-1.9, 0.0]],
            dtype=torch.float64,
        ),
        torch.tensor(
            [[-2.0, -0.2], [2.2, 0.2], [-1.8, 0.0], [2.0, -0.2], [-2.2, 0.2], [1.8, 0.0]],
            dtype=torch.float64,
        ),
    ):
        _base_pos, c_edges, c_sizes, cluster_meta = _cluster_probe()
        result = score_core_v3(step, c_edges, c_sizes, graph_meta=cluster_meta)
        g2_scores.append(
            _facet_sum(
                result.facets,
                (
                    "G2_cluster_exclusion",
                    "G2_cluster_sibling_overlap",
                    "G2_cluster_nesting_fidelity",
                    "G2_cluster_edge_intrusion",
                    "G2_cluster_hac_ari",
                    "G2_cluster_compactness_log_band",
                ),
            )
        )
    _assert_nonincreasing(g2_scores)

    g4_layered_scores = []
    for deformation in (0, 1, 2):
        tree_pos, tree_edges, tree_sizes, tree_meta = _tree_probe(deformation=deformation)
        result = score_core_v3(tree_pos, tree_edges, tree_sizes, graph_meta=tree_meta)
        g4_layered_scores.append(
            _facet_sum(
                result.facets,
                tuple(code for code in result.facets if code.startswith("G4_")),
            )
        )
    _assert_nonincreasing(g4_layered_scores)

    g4_radial_scores = []
    for deformation in (0, 1, 2):
        tree_pos, tree_edges, tree_sizes, tree_meta = _tree_probe(
            radial=True,
            deformation=deformation,
        )
        result = score_core_v3(tree_pos, tree_edges, tree_sizes, graph_meta=tree_meta)
        g4_radial_scores.append(
            _facet_sum(
                result.facets,
                tuple(code for code in result.facets if code.startswith("G4_")),
            )
        )
    _assert_nonincreasing(g4_radial_scores)

    g5_scores = []
    for shift, quality, change in ((0.2, 0.98, 0.042), (0.45, 0.90, 0.02), (0.0, 0.70, 0.40)):
        t_pos, t_edges, t_sizes, t_meta = _temporal_probe(
            current_shift=shift,
            previous_quality=quality,
            graph_change=change,
        )
        result = score_core_v3(t_pos, t_edges, t_sizes, graph_meta=t_meta)
        g5_scores.append(_facet_sum(result.facets, ("G5_temporal_stability",)))
    _assert_nonincreasing(g5_scores)

    g7_scores = []
    for deformation in (0, 1, 2):
        p_pos, p_edges, p_sizes, p_meta = _ported_probe(
            wrong_side=deformation >= 2,
            routed=True,
            deformation=deformation,
        )
        result = score_core_v3(p_pos, p_edges, p_sizes, graph_meta=p_meta)
        g7_scores.append(
            _facet_sum(
                result.facets,
                (
                    "G7_port_hard_compliance",
                    "G7_routed_curve_quality",
                    "G7_routed_bend_terminal_economy",
                ),
            )
        )
    _assert_nonincreasing(g7_scores)


def test_g2_compactness_log_ratio_band_discriminates_sprawl() -> None:
    """Assert the rebuilt compactness band separates 20x and 100x inflation.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    _base_pos, edges, sizes, meta = _cluster_probe()
    twenty_pos = torch.tensor(
        [[-10.0, 0.0], [0.0, 0.0], [10.0, 0.0], [30.0, 0.0], [40.0, 0.0], [50.0, 0.0]],
        dtype=torch.float64,
    )
    hundred_pos = torch.tensor(
        [[-50.0, 0.0], [0.0, 0.0], [50.0, 0.0], [150.0, 0.0], [200.0, 0.0], [250.0, 0.0]],
        dtype=torch.float64,
    )
    twenty = score_core_v3(twenty_pos, edges, sizes, graph_meta=meta)
    hundred = score_core_v3(hundred_pos, edges, sizes, graph_meta=meta)
    twenty_score = twenty.facets["G2_cluster_compactness_log_band"].score
    hundred_score = hundred.facets["G2_cluster_compactness_log_band"].score
    assert twenty_score is not None
    assert hundred_score is not None
    assert twenty_score > hundred_score + 0.05


def test_print_g2_g4_g5_g7_publication_records(capsys: pytest.CaptureFixture[str]) -> None:
    """Print representative facet scores and applicability records.

    Parameters
    ----------
    capsys : pytest.CaptureFixture[str]
        Pytest capture fixture.

    Returns
    -------
    None
    """
    cluster_pos, cluster_edges, cluster_sizes, cluster_meta = _cluster_probe()
    tree_pos, tree_edges, tree_sizes, tree_meta = _tree_probe()
    temporal_pos, temporal_edges, temporal_sizes, temporal_meta = _temporal_probe()
    port_pos, port_edges, port_sizes, port_meta = _ported_probe(routed=True)
    cluster = score_core_v3(cluster_pos, cluster_edges, cluster_sizes, graph_meta=cluster_meta)
    tree = score_core_v3(tree_pos, tree_edges, tree_sizes, graph_meta=tree_meta)
    temporal = score_core_v3(
        temporal_pos,
        temporal_edges,
        temporal_sizes,
        graph_meta=temporal_meta,
    )
    ported = score_core_v3(port_pos, port_edges, port_sizes, graph_meta=port_meta)
    print("G2 clustered row")
    for code, facet in cluster.facets.items():
        if code.startswith("G2_"):
            print(code, facet.score, facet.applicability_reason)
    print("G4 tree row")
    for code, facet in tree.facets.items():
        if code.startswith("G4_"):
            print(code, facet.score, facet.applicability_reason)
    print("G5 temporal row")
    for code, facet in temporal.facets.items():
        if code.startswith("G5_"):
            print(code, facet.score, facet.applicability_reason, facet.metadata["flags"])
    print("G7 ported row")
    for code, facet in ported.facets.items():
        if code.startswith("G7_"):
            print(code, facet.score, facet.applicability_reason, facet.metadata["flags"])
    captured = capsys.readouterr()
    assert "G2 clustered row" in captured.out
    assert "G4 tree row" in captured.out
    assert "G5 temporal row" in captured.out
    assert "G7 ported row" in captured.out


def _facet_sum(facets: Mapping[str, object], codes: Tuple[str, ...]) -> float:
    """Return the sum of selected facet scores for monotonicity smoke tests.

    Parameters
    ----------
    facets : Mapping[str, object]
        Facet records keyed by code.
    codes : Tuple[str, ...]
        Facets to sum.

    Returns
    -------
    float
        Sum of finite selected scores.
    """
    total = 0.0
    for code in codes:
        score = facets[code].score  # type: ignore[attr-defined]
        total += 0.0 if score is None else float(score)
    return total


def _assert_nonincreasing(values: List[float]) -> None:
    """Assert a sequence is monotonically nonincreasing.

    Parameters
    ----------
    values : List[float]
        Values to check.

    Returns
    -------
    None
    """
    assert values[0] > values[-1]
    for before, after in zip(values, values[1:]):
        assert before + 1e-9 >= after
