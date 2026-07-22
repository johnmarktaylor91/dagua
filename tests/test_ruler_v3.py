"""Regression tests for the scoring-only V3 ruler core."""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import pytest
import torch

from dagua.eval.ruler_v3 import (
    CORE_TIERS,
    FAMILY_MARGIN_ALLOWANCES,
    FAMILY_SOFTMIN_TAU,
    PURE_GEOMETRY_FACETS,
    SEVERE_G6_BREACH_FLAG,
    SEVERE_G6_FACETS,
    TIER1_TRADEOFF_FLAG,
    WHITESPACE_RATIO_HI,
    RulerV3Facet,
    RulerV3Result,
    angle_weighted_crossing_score,
    crossing_weight_multiplier,
    material_hold_ineligible,
    referee_eligibility_key,
    renormalized_score,
    score_core_v3,
    severe_g6_breach,
    severe_g6_breach_depth,
    severe_g6_floor_breach,
    sol_declared_weight_subcontract,
    tier1_measurement_weight,
    tier1_tradeoff_flags,
    with_tier1_tradeoff_flag,
)
from dagua.eval.ruler_v3_groups import G6_SEVERE_FLOOR


def _single_crossing_probe(
    angle_degrees: float, *, crossed: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a two-edge probe with a controlled crossing angle.

    Parameters
    ----------
    angle_degrees : float
        Desired crossing angle in degrees.
    crossed : bool
        Whether the two edges should cross.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Positions with shape ``[4, 2]`` and edge index with shape ``[2, 2]``.
    """
    theta = math.radians(angle_degrees)
    if crossed:
        pos = torch.tensor(
            [
                (-1.0, 0.0),
                (1.0, 0.0),
                (-math.cos(theta), -math.sin(theta)),
                (math.cos(theta), math.sin(theta)),
            ],
            dtype=torch.float64,
        )
    else:
        pos = torch.tensor(
            [
                (-1.0, 0.0),
                (1.0, 0.0),
                (-math.cos(theta), 1.0 + math.sin(theta)),
                (math.cos(theta), 1.0 + 2.0 * math.sin(theta)),
            ],
            dtype=torch.float64,
        )
    edges = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    return pos, edges


def _matching_layout_from_permutation(
    permutation: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a bipartite matching layout whose crossings equal inversions.

    Parameters
    ----------
    permutation : Sequence[int]
        Right-side vertical order for each left-side matched edge.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Positions with shape ``[2N, 2]`` and matching edge index with shape
        ``[2, N]``.
    """
    count = len(permutation)
    left = [(0.0, float(index)) for index in range(count)]
    right = [(1.0, float(permutation[index])) for index in range(count)]
    pos = torch.tensor([*left, *right], dtype=torch.float64)
    edges = torch.tensor(
        [[index, count + index] for index in range(count)],
        dtype=torch.long,
    ).t()
    return pos, edges.contiguous()


def _probe_layout() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a compact graph that exercises all V3 CORE facets.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Positions ``[N, 2]``, edge index ``[2, E]``, and node sizes ``[N, 2]``.
    """
    pos = torch.tensor(
        [
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 4.0],
            [0.0, 4.0],
            [8.0, 4.0],
            [12.0, 4.0],
        ],
        dtype=torch.float64,
    )
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 0, 1, 3, 4],
            [1, 2, 3, 0, 2, 3, 4, 5],
        ],
        dtype=torch.long,
    )
    node_sizes = torch.full((6, 2), 0.5, dtype=torch.float64)
    return pos, edge_index, node_sizes


def _chain_layout(
    node_count: int,
    gap: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a straight chain probe.

    Parameters
    ----------
    node_count : int
        Number of chain nodes.
    gap : float, optional
        Horizontal center spacing.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Positions ``[N, 2]``, edge index ``[2, E]``, and node sizes ``[N, 2]``.
    """
    pos = torch.stack(
        (
            torch.arange(node_count, dtype=torch.float64) * gap,
            torch.zeros(node_count, dtype=torch.float64),
        ),
        dim=1,
    )
    edge_index = torch.tensor(
        [
            list(range(node_count - 1)),
            list(range(1, node_count)),
        ],
        dtype=torch.long,
    )
    node_sizes = torch.ones((node_count, 2), dtype=torch.float64)
    return pos, edge_index, node_sizes


def _grid_layout(side: int, gap: float = 3.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a square grid probe.

    Parameters
    ----------
    side : int
        Number of nodes per grid side.
    gap : float, optional
        Center spacing.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Positions ``[N, 2]``, edge index ``[2, E]``, and node sizes ``[N, 2]``.
    """
    coords = [(float(x) * gap, float(y) * gap) for y in range(side) for x in range(side)]
    edges = []
    for y in range(side):
        for x in range(side):
            node = y * side + x
            if x + 1 < side:
                edges.append((node, node + 1))
                edges.append((node + 1, node))
            if y + 1 < side:
                edges.append((node, node + side))
                edges.append((node + side, node))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return (
        torch.tensor(coords, dtype=torch.float64),
        edge_index,
        torch.ones((side * side, 2), dtype=torch.float64),
    )


def _synthetic_tradeoff_result(
    *,
    tiered: float,
    c1_score: float,
) -> RulerV3Result:
    """Create a minimal RulerV3Result for W7 comparison tests.

    Parameters
    ----------
    tiered : float
        Synthetic tiered composite score.
    c1_score : float
        Synthetic T1 facet score.

    Returns
    -------
    RulerV3Result
        Minimal result with one T1 facet and one T2 facet.
    """
    facets = {
        "C1": RulerV3Facet(
            code="C1",
            name="ksm_stress",
            tier=1,
            score=c1_score,
            base_weight=4.0,
            effective_weight=4.0,
            applicable=True,
            applicability_reason="synthetic",
            metadata={},
        ),
        "C4": RulerV3Facet(
            code="C4",
            name="node_occlusion",
            tier=2,
            score=1.0,
            base_weight=2.0,
            effective_weight=2.0,
            applicable=True,
            applicability_reason="synthetic",
            metadata={},
        ),
    }
    return RulerV3Result(
        facets=facets,
        scores={"tiered": tiered, "equal": tiered, "tier1_only": 100.0 * c1_score},
        flags=tuple(),
        applicability={code: facet.applicable for code, facet in facets.items()},
        coverage={
            "applicable_facets": 2,
            "total_facets": 2,
            "tier1_applicable_facets": 1,
            "applicable_groups": 0,
        },
        metadata={},
    )


def _g6_result(
    *,
    weighted_ksm: float | None = None,
    local_weight_monotonicity: float | None = None,
    tiered: float = 50.0,
) -> RulerV3Result:
    """Create a synthetic V3 result with optional severe-G6 facets.

    Parameters
    ----------
    weighted_ksm : float | None, optional
        Synthetic ``G6_weighted_ksm`` score.
    local_weight_monotonicity : float | None, optional
        Synthetic ``G6_local_weight_monotonicity`` score.
    tiered : float, default=50.0
        Synthetic composite score.

    Returns
    -------
    RulerV3Result
        Result with only the requested applicable G6 facets.
    """
    values = {
        "G6_weighted_ksm": weighted_ksm,
        "G6_local_weight_monotonicity": local_weight_monotonicity,
    }
    facets = {
        code: RulerV3Facet(
            code=code,
            name=code,
            tier=1 if code == "G6_weighted_ksm" else 3,
            score=float(score),
            base_weight=4.0 if code == "G6_weighted_ksm" else 1.0,
            effective_weight=4.0 if code == "G6_weighted_ksm" else 1.0,
            applicable=True,
            applicability_reason="synthetic",
            metadata={},
        )
        for code, score in values.items()
        if score is not None
    }
    return RulerV3Result(
        facets=facets,
        scores={"tiered": tiered, "equal": tiered, "tier1_only": tiered},
        flags=tuple(),
        applicability={code: True for code in facets},
        coverage={
            "applicable_facets": len(facets),
            "total_facets": len(facets),
            "tier1_applicable_facets": sum(1 for facet in facets.values() if facet.tier == 1),
            "applicable_groups": 1 if facets else 0,
        },
        metadata={},
    )


def _directed_grid_layout(
    side: int,
    gap: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a square grid probe with one-way right/down edges.

    Parameters
    ----------
    side : int
        Number of nodes per grid side.
    gap : float, optional
        Center spacing.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Positions ``[N, 2]``, edge index ``[2, E]``, and node sizes ``[N, 2]``.
    """
    coords = [(float(x) * gap, float(y) * gap) for y in range(side) for x in range(side)]
    edges = []
    for y in range(side):
        for x in range(side):
            node = y * side + x
            if x + 1 < side:
                edges.append((node, node + 1))
            if y + 1 < side:
                edges.append((node, node + side))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return (
        torch.tensor(coords, dtype=torch.float64),
        edge_index,
        torch.ones((side * side, 2), dtype=torch.float64),
    )


def _one_row_star_layout(leaves: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a directed hub-and-one-leaf-row star probe.

    Parameters
    ----------
    leaves : int
        Number of leaves in the single row.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Positions ``[N, 2]``, edge index ``[2, E]``, and node sizes ``[N, 2]``.
    """
    leaf_x = torch.arange(leaves, dtype=torch.float64) * 3.0
    leaf_x -= float(leaf_x.mean().item())
    coords = [(0.0, 0.0)] + [(float(x), 3.0) for x in leaf_x.tolist()]
    edge_index = torch.tensor(
        [[0 for _ in range(leaves)], list(range(1, leaves + 1))],
        dtype=torch.long,
    )
    return (
        torch.tensor(coords, dtype=torch.float64),
        edge_index,
        torch.ones((leaves + 1, 2), dtype=torch.float64),
    )


def _binary_tree_layout(
    depth: int,
    gap: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a tidy layered binary tree probe.

    Parameters
    ----------
    depth : int
        Maximum tree depth below the root.
    gap : float, optional
        Horizontal and vertical layer spacing.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Positions ``[N, 2]``, edge index ``[2, E]``, and node sizes ``[N, 2]``.
    """
    coords = []
    edges = []
    for layer in range(depth + 1):
        count = 2**layer
        start = count - 1
        offset = (count - 1) * gap / 2.0
        for index in range(count):
            coords.append((index * gap - offset, layer * gap))
            node = start + index
            left = 2 * node + 1
            right = 2 * node + 2
            if layer < depth:
                edges.extend([(node, left), (node, right)])
    node_count = len(coords)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return (
        torch.tensor(coords, dtype=torch.float64),
        edge_index,
        torch.ones((node_count, 2), dtype=torch.float64),
    )


def _radial_star_layout(leaves: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a minimum-circumference radial star probe.

    Parameters
    ----------
    leaves : int
        Number of star leaves around the hub.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Positions ``[N, 2]``, edge index ``[2, E]``, and node sizes ``[N, 2]``.
    """
    radius = leaves / (2.0 * math.pi)
    coords = [(0.0, 0.0)]
    for index in range(leaves):
        theta = 2.0 * math.pi * index / leaves
        coords.append((radius * math.cos(theta), radius * math.sin(theta)))
    edge_index = torch.tensor(
        [[0 for _ in range(leaves)], list(range(1, leaves + 1))],
        dtype=torch.long,
    )
    return (
        torch.tensor(coords, dtype=torch.float64),
        edge_index,
        torch.ones((leaves + 1, 2), dtype=torch.float64),
    )


def _layered_dag_layout(
    layers: int,
    width: int,
    gap: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a layered rank DAG probe.

    Parameters
    ----------
    layers : int
        Number of ranks.
    width : int
        Nodes per rank.
    gap : float, optional
        Rank and within-rank spacing.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Positions ``[N, 2]``, edge index ``[2, E]``, and node sizes ``[N, 2]``.
    """
    coords = []
    edges = []
    for layer in range(layers):
        offset = (width - 1) * gap / 2.0
        for column in range(width):
            coords.append((column * gap - offset, layer * gap))
            if layer + 1 < layers:
                edges.append((layer * width + column, (layer + 1) * width + column))
                if column + 1 < width:
                    edges.append((layer * width + column, (layer + 1) * width + column + 1))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return (
        torch.tensor(coords, dtype=torch.float64),
        edge_index,
        torch.ones((layers * width, 2), dtype=torch.float64),
    )


def _disconnected_collinear_layout() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a disconnected C1 probe with one collinear component.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Positions ``[N, 2]``, edge index ``[2, E]``, and node sizes ``[N, 2]``.
    """
    pos = torch.tensor(
        [
            [0.0, 0.0],
            [4.0, 0.0],
            [0.0, 4.0],
            [8.0, 0.0],
            [11.0, 0.0],
            [14.0, 0.0],
            [17.0, 0.0],
            [20.0, 0.0],
            [23.0, 0.0],
        ],
        dtype=torch.float64,
    )
    edge_index = torch.tensor(
        [
            [0, 1, 2, 0, 3, 4, 5, 6, 7],
            [1, 2, 0, 2, 4, 5, 6, 7, 8],
        ],
        dtype=torch.long,
    )
    node_sizes = torch.ones((9, 2), dtype=torch.float64)
    return pos, edge_index, node_sizes


def _jittered_rotated_disconnected_chain_layout() -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Create the C1 alpha-regime flip probe with a tiny nonzero chain hull.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Positions ``[N, 2]``, edge index ``[2, E]``, and node sizes ``[N, 2]``.
    """
    pos, edge_index, node_sizes = _disconnected_collinear_layout()
    pos = pos.clone()
    pos[6, 1] = 1e-14
    theta = math.radians(17.0)
    rotation = torch.tensor(
        [
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta), math.cos(theta)],
        ],
        dtype=torch.float64,
    )
    return pos @ rotation.T, edge_index, node_sizes


def test_w7_tier1_tradeoff_flag_is_score_neutral() -> None:
    """W7 should flag aggregate-held T1 losses without changing composites.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    baseline = _synthetic_tradeoff_result(tiered=90.0, c1_score=0.95)
    candidate = _synthetic_tradeoff_result(tiered=91.0, c1_score=0.89)

    assert tier1_tradeoff_flags(baseline, candidate) == (TIER1_TRADEOFF_FLAG,)
    flagged = with_tier1_tradeoff_flag(baseline, candidate)

    assert flagged.scores == candidate.scores
    assert flagged.facets == candidate.facets
    assert flagged.flags == (TIER1_TRADEOFF_FLAG,)


def test_unit_invariance_for_all_core_facets() -> None:
    """Joint scaling of positions and node sizes must preserve every facet."""
    pos, edge_index, node_sizes = _probe_layout()
    baseline = score_core_v3(pos, edge_index, node_sizes)

    for alpha in (0.02, 0.1, 1.0, 10.0, 50.0):
        scaled = score_core_v3(alpha * pos, edge_index, alpha * node_sizes)
        for code, facet in baseline.facets.items():
            assert scaled.facets[code].score == pytest.approx(facet.score, abs=1e-9)
        assert scaled.scores["tiered"] == pytest.approx(baseline.scores["tiered"], abs=1e-9)


@pytest.mark.parametrize("angle", (15.0, 45.0, 75.0, 90.0))
def test_c2_angle_weighted_crossing_is_strictly_negative_per_crossing(angle: float) -> None:
    """Assert adding one crossing lowers C2' at every tested angle.

    Parameters
    ----------
    angle : float
        Crossing angle in degrees.

    Returns
    -------
    None
    """
    clear_pos, clear_edges = _single_crossing_probe(angle, crossed=False)
    crossed_pos, crossed_edges = _single_crossing_probe(angle, crossed=True)
    clear = angle_weighted_crossing_score(clear_pos, clear_edges)
    crossed = angle_weighted_crossing_score(crossed_pos, crossed_edges)
    assert crossed["crossing_angle_weight_mean"] >= 0.5
    assert crossed["crossing_angle_weight_mean"] <= 1.5
    assert crossed["edge_crossing_score"] < clear["edge_crossing_score"]


def test_c2_sol_counterexample_is_count_monotone() -> None:
    """Assert two perpendicular crossings cannot outrank one shallow crossing.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    edges = torch.tensor([[0, 2, 4], [1, 3, 5]], dtype=torch.long)
    one_shallow = torch.tensor(
        [
            [0.8353381566711482, -1.0993437673106923],
            [1.3500512486651401, -0.7385221705893695],
            [-1.6376227294973686, 0.8472125848570669],
            [-0.3236051039641243, 1.0049758493947827],
            [-1.7353746109554686, 0.8531447547286918],
            [0.23303430345011902, 1.044852294675133],
        ],
        dtype=torch.float64,
    )
    two_perpendicular = torch.tensor(
        [
            [0.2910327299230801, -0.8581254109036337],
            [0.15620328357839547, 0.8183790749837612],
            [1.017022075537967, 0.37468995055029247],
            [-0.2584253542881093, 0.26551331656227545],
            [0.3173615316656587, -0.7338447788947672],
            [0.17999323883164786, 0.7824427852386611],
        ],
        dtype=torch.float64,
    )

    shallow = angle_weighted_crossing_score(one_shallow, edges)
    perpendicular = angle_weighted_crossing_score(two_perpendicular, edges)

    assert shallow["crossing_count"] == 1
    assert perpendicular["crossing_count"] == 2
    assert perpendicular["edge_crossing_score"] < shallow["edge_crossing_score"]


def test_c2_retains_angle_readability_with_equal_crossing_count() -> None:
    """Assert angle quality remains a within-count C2 refinement.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    shallow_pos, edges = _single_crossing_probe(15.0, crossed=True)
    perpendicular_pos, _ = _single_crossing_probe(90.0, crossed=True)
    shallow = angle_weighted_crossing_score(shallow_pos, edges)
    perpendicular = angle_weighted_crossing_score(perpendicular_pos, edges)

    assert shallow["crossing_count"] == perpendicular["crossing_count"] == 1
    assert perpendicular["edge_crossing_score"] > shallow["edge_crossing_score"]


def test_c2_count_monotonicity_over_random_same_graph_layouts() -> None:
    """Assert higher crossing count strictly lowers C2 across same-graph layouts.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    generator = torch.Generator().manual_seed(7741)
    edges = torch.tensor([[0, 2, 4, 6], [1, 3, 5, 7]], dtype=torch.long)
    measurements: list[Dict[str, float]] = []
    for _index in range(160):
        pos = torch.randn((8, 2), generator=generator, dtype=torch.float64)
        result = angle_weighted_crossing_score(pos, edges)
        measurements.append(result)

    comparable_pairs = 0
    for left in measurements:
        for right in measurements:
            if right["crossing_count"] > left["crossing_count"]:
                comparable_pairs += 1
                assert right["edge_crossing_score"] < left["edge_crossing_score"]
    assert comparable_pairs > 0


def test_c2_has_no_neutral_floor_at_high_crossing_count() -> None:
    """Assert high crossing counts remain strictly ordered above zero.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    five_pos, edges = _matching_layout_from_permutation((3, 2, 0, 1))
    six_pos, _ = _matching_layout_from_permutation((3, 2, 1, 0))
    five = angle_weighted_crossing_score(five_pos, edges)
    six = angle_weighted_crossing_score(six_pos, edges)

    assert five["crossing_count"] == 5
    assert six["crossing_count"] == 6
    assert six["edge_crossing_score"] > 0.0
    assert six["edge_crossing_score"] < five["edge_crossing_score"]


def test_changed_core_facets_keep_unit_scale_invariance() -> None:
    """Assert Phase-A core facet changes are exactly unit-scale invariant.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    pos, edges, sizes = _probe_layout()
    labels = torch.full_like(sizes, 0.25)
    offsets = torch.full_like(sizes, 0.1)
    base = score_core_v3(pos, edges, sizes, label_sizes=labels, label_offsets=offsets)
    scaled = score_core_v3(
        13.0 * pos,
        edges,
        13.0 * sizes,
        label_sizes=13.0 * labels,
        label_offsets=13.0 * offsets,
    )
    for code in ("C2", "C4", "C6", "C8"):
        assert scaled.facets[code].score == pytest.approx(base.facets[code].score, abs=1e-12)


def test_c8_tree_row_demotion_is_materially_unchanged() -> None:
    """Assert C8 is emitted diagnostically without moving a clean tree row.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    pos, edges, sizes = _chain_layout(6, gap=3.0)
    result = score_core_v3(pos, edges, sizes)
    weights = {
        code: (2.0 if code == "C8" else facet.effective_weight)
        for code, facet in result.facets.items()
        if facet.applicable
    }
    restored = renormalized_score(
        {code: facet.score for code, facet in result.facets.items()}, weights
    )
    assert result.facets["C8"].effective_weight == 0.0
    assert result.facets["C8"].score == pytest.approx(1.0)
    assert abs(restored - result.scores["tiered"]) <= 0.25


def test_position_only_scale_invariance_for_pure_geometry_facets() -> None:
    """Pure-geometry facets must ignore position-only scale changes."""
    pos, edge_index, node_sizes = _probe_layout()
    baseline = score_core_v3(pos, edge_index, node_sizes)

    for alpha in (0.02, 0.1, 1.0, 10.0, 50.0):
        scaled = score_core_v3(alpha * pos, edge_index, node_sizes)
        for code in PURE_GEOMETRY_FACETS:
            assert scaled.facets[code].score == pytest.approx(
                baseline.facets[code].score,
                abs=1e-9,
            )


def test_c1_disconnected_collinear_position_scale_invariance() -> None:
    """C1 must not drift when a disconnected component has zero hull area."""
    pos, edge_index, node_sizes = _disconnected_collinear_layout()
    scores = []

    for alpha in (0.02, 0.1, 1.0, 10.0, 50.0):
        result = score_core_v3(alpha * pos, edge_index, node_sizes)
        assert result.facets["C1"].score is not None
        scores.append(result.facets["C1"].score)

    assert max(scores) - min(scores) < 1e-6


def test_c1_jittered_rotated_chain_uses_scale_invariant_degeneracy_gate() -> None:
    """C1 should not drift when tiny nonzero hulls are scaled with positions."""
    pos, edge_index, node_sizes = _jittered_rotated_disconnected_chain_layout()
    scores = []

    for alpha in (0.1, 50.0):
        result = score_core_v3(alpha * pos, edge_index, node_sizes)
        assert result.facets["C1"].score is not None
        scores.append(float(result.facets["C1"].score))

    assert max(scores) - min(scores) < 1e-6


def test_c5_whitespace_band_flags_crater_and_stays_flat_on_legitimate_chain() -> None:
    """The C5 band should tolerate normal chain extent but penalize crater sprawl."""
    pos, edge_index, node_sizes = _chain_layout(10)

    legitimate = score_core_v3(pos, edge_index, node_sizes)
    crater = score_core_v3(1000.0 * pos, edge_index, node_sizes)

    assert legitimate.facets["C5"].score == pytest.approx(1.0)
    assert crater.facets["C5"].score is not None
    assert crater.facets["C5"].score < 0.01
    assert crater.facets["C5"].metadata["whitespace_ratio"] > WHITESPACE_RATIO_HI
    assert "SPRAWL" in crater.flags


def test_c5_directed_tidy_grid_uses_content_floor_for_crowding_side() -> None:
    """A tidy directed grid should not be crowding-penalized by the DAG floor."""
    pos, edge_index, node_sizes = _directed_grid_layout(10, gap=3.0)

    result = score_core_v3(pos, edge_index, node_sizes)
    metadata = result.facets["C5"].metadata

    assert result.facets["C5"].score == pytest.approx(1.0)
    assert "SPRAWL" not in result.flags
    assert metadata["whitespace_crowding_score"] == pytest.approx(1.0)
    assert metadata["whitespace_sprawl_side_score"] == pytest.approx(1.0)
    assert metadata["whitespace_ratio"] == pytest.approx(metadata["whitespace_sprawl_ratio"])


def test_c5_directed_one_row_star_uses_content_floor_for_crowding_side() -> None:
    """A directed one-row star should stay on the C5 plateau."""
    pos, edge_index, node_sizes = _one_row_star_layout(1000)

    result = score_core_v3(pos, edge_index, node_sizes)

    assert result.facets["C5"].score is not None
    assert result.facets["C5"].score >= 0.99
    assert "SPRAWL" not in result.flags


def test_c5_legitimate_sprawl_probe_set_stays_on_plateau() -> None:
    """Structure floors should not false-positive on legitimate broad layouts."""
    probes = {
        "deep_tree": _binary_tree_layout(depth=8),
        "long_chain": _chain_layout(200),
        "radial_star": _radial_star_layout(leaves=1000),
        "layered_ranks": _layered_dag_layout(layers=12, width=30),
    }

    table: Dict[str, float] = {}
    for name, (pos, edge_index, node_sizes) in probes.items():
        result = score_core_v3(pos, edge_index, node_sizes)
        score = result.facets["C5"].score
        assert score == pytest.approx(1.0), name
        assert "SPRAWL" not in result.flags, name
        table[name] = float(result.facets["C5"].metadata["whitespace_ratio"])

    assert set(table) == {"deep_tree", "long_chain", "radial_star", "layered_ranks"}


def test_c5_crater_threshold_locks_still_fire() -> None:
    """The 10x and 50x area craters should remain hard SPRAWL failures."""
    pos, edge_index, node_sizes = _grid_layout(10)

    for alpha in (10.0, 50.0):
        crater = score_core_v3(alpha * pos, edge_index, node_sizes)
        score = crater.facets["C5"].score
        assert score is not None
        assert score < 0.01
        assert "SPRAWL" in crater.flags


def test_c5_bidirectional_grid_x10_x50_locks_still_fire() -> None:
    """Bidirectional grid x10 and x50 crater locks should remain hard failures."""
    pos, edge_index, node_sizes = _grid_layout(10)

    locks: Dict[float, float] = {}
    for alpha in (10.0, 50.0):
        crater = score_core_v3(alpha * pos, edge_index, node_sizes)
        score = crater.facets["C5"].score
        assert score is not None
        assert score < 0.01
        assert "SPRAWL" in crater.flags
        locks[alpha] = float(crater.facets["C5"].metadata["whitespace_ratio"])

    assert locks[50.0] > locks[10.0]


def test_edgeless_margins_are_inapplicable_not_imputed() -> None:
    """C1, C3, and C10 should publish None on edgeless graphs."""
    pos = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]], dtype=torch.float64)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    node_sizes = torch.ones((3, 2), dtype=torch.float64)

    result = score_core_v3(pos, edge_index, node_sizes)

    for code in ("C1", "C3", "C10"):
        assert result.facets[code].score is None
        assert result.facets[code].applicable is False


def test_c4_occlusion_includes_label_extents() -> None:
    """C4 should detect overlap between declared label-inclusive extents."""
    pos = torch.tensor([[0.0, 0.0], [4.0, 0.0]], dtype=torch.float64)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    node_sizes = torch.ones((2, 2), dtype=torch.float64)
    label_sizes = torch.tensor([[4.0, 1.0], [4.0, 1.0]], dtype=torch.float64)
    label_offsets = torch.tensor([[1.5, 0.0], [-1.5, 0.0]], dtype=torch.float64)

    node_only = score_core_v3(pos, edge_index, node_sizes)
    label_inclusive = score_core_v3(
        pos,
        edge_index,
        node_sizes,
        label_sizes=label_sizes,
        label_offsets=label_offsets,
    )

    assert node_only.facets["C4"].score == pytest.approx(1.0)
    assert label_inclusive.facets["C4"].score is not None
    assert label_inclusive.facets["C4"].score < 1.0
    assert label_inclusive.facets["C4"].metadata["label_inclusive"] is True


def test_applicability_renormalization_excludes_missing_facets() -> None:
    """Missing facets should leave the denominator instead of receiving neutral fill."""
    score = renormalized_score(
        {"present": 0.5, "missing": None},
        {"present": 2.0, "missing": 100.0},
    )

    assert score == pytest.approx(50.0)


def test_triple_view_composite_and_tier_weights_are_published() -> None:
    """V3 should publish capped, audit-linear, equal, and Tier-1-only views."""
    pos, edge_index, node_sizes = _probe_layout()
    result = score_core_v3(pos, edge_index, node_sizes)

    assert set(result.scores) == {"tiered", "tiered_linear", "equal", "tier1_only"}
    assert result.facets["C1"].base_weight == pytest.approx(4.0)
    assert result.facets["C5"].base_weight == pytest.approx(2.0)
    assert result.facets["C9"].base_weight == pytest.approx(1.0)
    assert result.facets["C2"].base_weight == pytest.approx(6.0)
    assert result.facets["C2"].effective_weight == pytest.approx(
        6.0 * crossing_weight_multiplier(pos.shape[0])
    )
    assert result.facets["C6"].effective_weight == pytest.approx(0.0)
    assert result.facets["C8"].effective_weight == pytest.approx(0.0)
    assert sum(1 for tier in CORE_TIERS.values() if tier == 1) == 3
    assert result.coverage["applicable_facets"] == 10
    assert 0.0 <= result.scores["equal"] <= 100.0
    assert 0.0 <= result.scores["tiered"] <= 100.0
    assert 0.0 <= result.scores["tiered_linear"] <= 100.0
    assert 0.0 <= result.scores["tier1_only"] <= 100.0


def test_severe_g6_contract_exports_single_floor_and_pair_form() -> None:
    """Shared G6 helpers use the frozen group floor for both absolute and pair forms."""
    assert SEVERE_G6_FACETS == ("G6_weighted_ksm", "G6_local_weight_monotonicity")
    assert G6_SEVERE_FLOOR == pytest.approx(0.55)
    baseline = _g6_result(weighted_ksm=0.70, local_weight_monotonicity=0.95)
    dropped = _g6_result(weighted_ksm=0.50, local_weight_monotonicity=0.95)
    already_low = _g6_result(weighted_ksm=0.54, local_weight_monotonicity=0.95)
    small_drop = _g6_result(weighted_ksm=0.56, local_weight_monotonicity=0.95)

    assert severe_g6_breach(dropped)
    assert severe_g6_breach_depth(dropped) == pytest.approx(0.05)
    assert severe_g6_floor_breach(baseline, dropped)
    assert not severe_g6_floor_breach(already_low, dropped)
    assert not severe_g6_floor_breach(baseline, small_drop)


def test_referee_eligibility_key_hard_ineligible_and_least_breach_fallback() -> None:
    """Native selection prefix makes severe G6 breach hard-ineligible before score."""
    compliant_low_score = _g6_result(weighted_ksm=0.55, tiered=1.0)
    breaching_high_score = _g6_result(weighted_ksm=0.10, tiered=100.0)
    shallow_breach = _g6_result(weighted_ksm=0.54, tiered=10.0)
    deep_breach = _g6_result(weighted_ksm=0.30, tiered=99.0)
    inapplicable = _g6_result(tiered=0.0)

    assert (referee_eligibility_key(compliant_low_score), compliant_low_score.scores["tiered"]) > (
        referee_eligibility_key(breaching_high_score),
        breaching_high_score.scores["tiered"],
    )
    assert (referee_eligibility_key(shallow_breach), shallow_breach.scores["tiered"]) > (
        referee_eligibility_key(deep_breach),
        deep_breach.scores["tiered"],
    )
    assert referee_eligibility_key(inapplicable) == (1, -0.0)


def test_referee_eligibility_key_all_material_hold_pool_uses_least_breach() -> None:
    """All-ineligible pair pools fall back to least breach without restoring eligibility."""
    baseline = _synthetic_tradeoff_result(tiered=100.0, c1_score=1.0)
    shallow_hold = _synthetic_tradeoff_result(tiered=99.0, c1_score=0.94)
    deep_hold = _synthetic_tradeoff_result(tiered=99.0, c1_score=0.80)

    shallow_key = referee_eligibility_key(
        shallow_hold,
        baseline=baseline,
        shape_distance=0.40,
        aggregate_delta_fraction=0.01,
        two_layout_buyback=1.1,
    )
    deep_key = referee_eligibility_key(
        deep_hold,
        baseline=baseline,
        shape_distance=0.40,
        aggregate_delta_fraction=0.01,
        two_layout_buyback=1.1,
    )

    assert material_hold_ineligible(
        baseline,
        shallow_hold,
        shape_distance=0.40,
        aggregate_delta_fraction=0.01,
        two_layout_buyback=1.1,
    )
    assert shallow_key[0] == 0
    assert deep_key[0] == 0
    assert shallow_key > deep_key


def test_pair_material_hold_never_outranks_compliant_candidate() -> None:
    """Pair-aware ineligibility remains a prefix before score tie-breaking."""
    baseline = _synthetic_tradeoff_result(tiered=100.0, c1_score=1.0)
    compliant_low_score = _synthetic_tradeoff_result(tiered=1.0, c1_score=0.99)
    material_hold_high_score = _synthetic_tradeoff_result(tiered=99.0, c1_score=0.90)

    compliant_keyed = (
        referee_eligibility_key(
            compliant_low_score,
            baseline=baseline,
            shape_distance=0.40,
            aggregate_delta_fraction=0.01,
            two_layout_buyback=1.1,
        ),
        compliant_low_score.scores["tiered"],
    )
    hold_keyed = (
        referee_eligibility_key(
            material_hold_high_score,
            baseline=baseline,
            shape_distance=0.40,
            aggregate_delta_fraction=0.01,
            two_layout_buyback=1.1,
        ),
        material_hold_high_score.scores["tiered"],
    )

    assert compliant_keyed > hold_keyed


def test_tier1_measurement_weight_removes_g6_ramp_only_from_instrument() -> None:
    """G6 severe-ramp pricing stays audit-visible while Tier-1 measurement de-ramps."""
    facet = RulerV3Facet(
        code="G6_weighted_ksm",
        name="weighted_shortest_path_isotonic_ksm",
        tier=1,
        score=0.575,
        base_weight=4.0,
        effective_weight=6.0,
        applicable=True,
        applicability_reason="synthetic",
        metadata={"severe_weight_ramp": 1.5},
    )
    ordinary = RulerV3Facet(
        code="C1",
        name="ksm_stress",
        tier=1,
        score=0.9,
        base_weight=4.0,
        effective_weight=4.0,
        applicable=True,
        applicability_reason="synthetic",
        metadata={},
    )

    assert tier1_measurement_weight(facet) == pytest.approx(4.0)
    assert tier1_measurement_weight(ordinary) == pytest.approx(4.0)


def test_softmin_cap_publishes_linear_audit_view_and_family_metadata() -> None:
    """The tiered headline is the family cap and the linear score stays auditable."""
    pos, edge_index, node_sizes = _probe_layout()
    result = score_core_v3(
        pos,
        edge_index,
        node_sizes,
        graph_meta={"ruler_family": "clustered"},
    )
    allowance = FAMILY_MARGIN_ALLOWANCES["clustered"]
    expected = min(result.scores["tiered_linear"], result.scores["tier1_only"] + allowance)

    assert result.metadata["softmin_family"] == "clustered"
    assert result.metadata["softmin_tau"] == pytest.approx(FAMILY_SOFTMIN_TAU)
    assert result.scores["tiered"] <= result.scores["tiered_linear"]
    assert result.scores["tiered"] <= result.scores["tier1_only"] + allowance
    assert result.scores["tiered"] == pytest.approx(expected, abs=1.0)


def test_sol_declared_weight_subcontract_is_named_not_standalone_floor() -> None:
    """The G6 corridor guard identifies sub-severe baseline-relative damage."""
    baseline = _g6_result(weighted_ksm=0.70)
    corridor = _g6_result(weighted_ksm=0.58)
    tiny_drop = _g6_result(weighted_ksm=0.58)

    assert sol_declared_weight_subcontract(baseline, corridor)
    assert not sol_declared_weight_subcontract(_g6_result(weighted_ksm=0.61), tiny_drop)


def test_severe_g6_flag_plumbing_is_score_neutral() -> None:
    """Absolute severe-G6 breach publishes a row flag without changing composites."""
    pos = torch.tensor(
        [[0.0, 0.0], [100.0, 0.0], [101.0, 0.0]],
        dtype=torch.float64,
    )
    edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    sizes = torch.ones((3, 2), dtype=torch.float64)
    meta = {"edge_weights": [1.0, 100.0], "weight_mode": "distance"}

    result = score_core_v3(pos, edges, sizes, graph_meta=meta)

    assert severe_g6_breach(result)
    assert SEVERE_G6_BREACH_FLAG in result.flags
    assert result.scores == {
        "tiered": result.scores["tiered"],
        "tiered_linear": result.scores["tiered_linear"],
        "equal": result.scores["equal"],
        "tier1_only": result.scores["tier1_only"],
    }


def test_severe_g6_flag_absent_on_compliant_weighted_row() -> None:
    """Compliant declared-weight rows do not carry the severe-G6 flag."""
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [101.0, 0.0]], dtype=torch.float64)
    edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    sizes = torch.ones((3, 2), dtype=torch.float64)
    meta = {"edge_weights": [1.0, 100.0], "weight_mode": "distance"}

    result = score_core_v3(pos, edges, sizes, graph_meta=meta)

    assert not severe_g6_breach(result)
    assert SEVERE_G6_BREACH_FLAG not in result.flags
