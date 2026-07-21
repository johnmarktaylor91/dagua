"""Regression tests for V3 conditional ruler groups."""

from __future__ import annotations

from typing import Dict, List, Mapping, Tuple

import pytest
import torch

from dagua.eval.ruler_v3 import renormalized_score, score_core_v3
from dagua.eval.ruler_v3_groups import evaluate_conditional_groups


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
    values = {code: facet.score for code, facet in with_g1.facets.items()}
    weights = {code: facet.effective_weight for code, facet in with_g1.facets.items()}
    assert with_g1.scores["tiered"] == pytest.approx(renormalized_score(values, weights))


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
        if code.startswith("G4_"):
            assert scaled_tree.facets[code].score == pytest.approx(facet.score, abs=1e-6)

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
        [[-5.0, 0.0], [0.0, 0.0], [5.0, 0.0], [15.0, 0.0], [20.0, 0.0], [25.0, 0.0]],
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
    compactness_moved = (
        abs(
            float(scaled_anchored.facets["G2_cluster_compactness_log_band"].score or 0.0)
            - float(base_anchored.facets["G2_cluster_compactness_log_band"].score or 0.0)
        )
        > 1e-6
    )
    assert moved_slot_a
    assert compactness_moved


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


def test_print_g2_g4_publication_records(capsys: pytest.CaptureFixture[str]) -> None:
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
    cluster = score_core_v3(cluster_pos, cluster_edges, cluster_sizes, graph_meta=cluster_meta)
    tree = score_core_v3(tree_pos, tree_edges, tree_sizes, graph_meta=tree_meta)
    print("G2 clustered row")
    for code, facet in cluster.facets.items():
        if code.startswith("G2_"):
            print(code, facet.score, facet.applicability_reason)
    print("G4 tree row")
    for code, facet in tree.facets.items():
        if code.startswith("G4_"):
            print(code, facet.score, facet.applicability_reason)
    captured = capsys.readouterr()
    assert "G2 clustered row" in captured.out
    assert "G4 tree row" in captured.out


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
