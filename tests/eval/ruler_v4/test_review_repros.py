"""Live repros from the P4 adversarial review rounds, banked as regressions.

Every test here is a repro that a review lane (P4REVIEW / P4REVERIFY /
P4REVERIFY2 / P4REVERIFY3, Fable and Opus) ran live against a shipped
defect. They are permanent: each one pins the contract-lawful outcome the
fix round established, so no later pass can silently re-open the channel.
"""

from __future__ import annotations

from typing import Optional, Tuple

import pytest
import torch

from dagua.eval.ruler_v4.ingestion import ingest
from dagua.eval.ruler_v4.legibility import U20a
from dagua.eval.ruler_v4.scene import (
    DrawingScene,
    GraphSemantics,
    ObservationProfile,
    ResultState,
    Route,
    Scene,
    StyleContract,
    ValidScene,
)


def _node_scene(
    positions: torch.Tensor,
    ranks: Optional[Tuple[int, ...]] = None,
    flow_axis: Optional[Tuple[float, float]] = None,
    edges: Tuple[Tuple[int, int], ...] = (),
) -> Scene:
    """Ingest one edge-light repro scene.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    ranks : tuple[int, ...] or None
        Declared ranks, if any.
    flow_axis : tuple[float, float] or None
        Declared flow axis, if any.
    edges : tuple[tuple[int, int], ...]
        Canonical graph edges.

    Returns
    -------
    Scene
        Validated static scene.
    """

    graph = GraphSemantics(
        tuple(f"n{index}" for index in range(positions.shape[0])),
        edges,
        ranks=ranks,
        flow_axis=flow_axis,
    )
    routes = tuple(
        Route(index, torch.stack((positions[source], positions[target])))
        for index, (source, target) in enumerate(edges)
    )
    result = ingest(
        graph,
        DrawingScene(positions, routes),
        StyleContract(),
        ObservationProfile(visible_channels=frozenset({"nodes", "routes"})),
    )
    assert isinstance(result, ValidScene)
    return result.scene


# --- P4REVERIFY3_FABLE blocker 1: declaring ranks must not exempt a line ---
# collapse perpendicular to the declared axis (the E3-banned game channel).


def test_u20a_rank_declaration_cannot_exempt_cross_axis_line_collapse() -> None:
    """A horizontal line with vertical-axis ranks stays catastrophic."""

    positions = torch.tensor([[4.0 * index, 0.0] for index in range(6)], dtype=torch.float64)
    declared = U20a(_node_scene(positions, ranks=tuple(range(6)), flow_axis=(0.0, 1.0)))
    undeclared = U20a(_node_scene(positions))
    assert declared.subterms["U20a.ii"] == pytest.approx(1.0, abs=0.0)
    # Mutating the declared rank block cannot improve the composite (golden 3).
    assert declared.value == undeclared.value == pytest.approx(1.0, abs=0.0)


# --- P4REVERIFY3_FABLE blocker 2: the residual frame must not score every ---
# jittered multi-node-per-rank layered drawing a vacuous worst.


@pytest.mark.parametrize("eps", (0.0, 1e-6, 1e-3, 0.1, 0.5))
def test_u20a_jittered_layered_grid_is_not_rank_collapsed(eps: float) -> None:
    """E2 is exemption-only: a healthy raw quotient survives declared ranks."""

    rows = []
    for rank in range(3):
        rows.append([0.0, 8.0 * rank + eps])
        rows.append([6.0, 8.0 * rank - eps])
    positions = torch.tensor(rows, dtype=torch.float64)
    facet = U20a(_node_scene(positions, ranks=(0, 0, 1, 1, 2, 2), flow_axis=(0.0, 1.0)))
    assert facet.subterms["U20a.ii"] == pytest.approx(0.0, abs=0.0)


def test_u20a_exact_declared_axis_column_scores_zero() -> None:
    """The E2 exemption itself: a fully rank-explained column is expected."""

    positions = torch.tensor([[0.0, 4.0 * rank] for rank in range(6)], dtype=torch.float64)
    facet = U20a(_node_scene(positions, ranks=tuple(range(6)), flow_axis=(0.0, 1.0)))
    assert facet.subterms["U20a.ii"] == pytest.approx(0.0, abs=0.0)
    assert facet.value == pytest.approx(0.0, abs=0.0)


# --- P4REVERIFY3_FABLE blocker 3: a declared cluster drawn as separated ---
# lumps must ingest and be scored (and punished), never abort the scene.


def test_scattered_cluster_scene_ingests_and_every_facet_evaluates() -> None:
    """A two-lump cluster whose core center misses the region stays scorable."""

    from dagua.eval.ruler_v4 import FACET_FUNCTIONS, evaluate_facet

    points = [[float(index % 5), float(index // 5)] for index in range(15)]
    points += [[100.0 + float(index % 5), float(index // 5)] for index in range(15)]
    positions = torch.tensor(points, dtype=torch.float64)
    graph = GraphSemantics(
        tuple(f"n{index}" for index in range(30)),
        (),
        clusters={"lump": tuple(range(30))},
    )
    result = ingest(
        graph,
        DrawingScene(positions),
        StyleContract(),
        ObservationProfile(visible_channels=frozenset({"nodes", "routes", "cluster_labels"})),
    )
    assert isinstance(result, ValidScene)
    for facet_id in sorted(FACET_FUNCTIONS):
        evaluate_facet(facet_id, result.scene)
    # The load-bearing claim is ingest + a typed result on the fallback path.
    # Scattering is NOT punished by U29/U30 (measured: the label lands in
    # empty space and sub-term (iii) sees no occluder; DISCREPANCIES.md 19).
    scored = evaluate_facet("U30", result.scene, alpha_grid_index=1)
    assert scored.state is ResultState.VALUE
    assert scored.value is not None


# --- P4REVERIFY3_FABLE blocker 4: the global blend must stay total on ---
# saturated defect populations (float dust pushes an exact-1.0 trim above 1).


def test_u20a_scores_saturated_populations_instead_of_crashing() -> None:
    """Golden 2's catastrophic endpoint scores ~1; it must never raise."""

    coincident = U20a(_node_scene(torch.zeros((6, 2), dtype=torch.float64)))
    assert coincident.state is ResultState.VALUE
    assert coincident.value == pytest.approx(1.0, abs=1e-6)
    nano = U20a(
        _node_scene(
            torch.tensor(
                [[0.0, 0.0], [1e-9, 0.0], [1e-9, 1e-9], [0.0, 1e-9]],
                dtype=torch.float64,
            ),
            edges=((0, 1), (1, 2), (2, 3)),
        )
    )
    assert nano.state is ResultState.VALUE
    assert nano.value == pytest.approx(1.0, abs=1e-6)


def test_u11_scores_plain_declared_rank_dag() -> None:
    """A load-bearing corpus family must evaluate, not die in the blend."""

    from dagua.eval.ruler_v4.edges import U11

    points = [[4.0 * column, 6.0 * rank] for rank in range(3) for column in range(3)]
    edges = tuple(
        (rank * 3 + column, (rank + 1) * 3 + child)
        for rank in range(2)
        for column in range(3)
        for child in range(3)
        if (column + child) % 2 == 0
    )
    scene = _node_scene(
        torch.tensor(points, dtype=torch.float64),
        ranks=tuple(rank for rank in range(3) for _ in range(3)),
        flow_axis=(0.0, 1.0),
        edges=edges,
    )
    result = U11(scene)
    assert result.state is ResultState.VALUE
    assert result.subterms


# --- P4REVERIFY3_OPUS blocker 3: the quintic smoothstep overshoots 1.0 by ---
# one ULP just below its upper knot, driving 1 - smoothstep(...) defect terms
# negative and crashing the blend on derived-placement cluster scenes.


def test_smoothstep_never_exceeds_one() -> None:
    """The helper's promised [0, 1] range holds at the overshoot input."""

    from dagua.eval.ruler_v4._util import smoothstep

    overshoot_input = torch.nextafter(
        torch.tensor(1.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64)
    )
    assert float(smoothstep(overshoot_input)) <= 1.0
    assert float(smoothstep(torch.tensor(1.0, dtype=torch.float64))) == 1.0


# --- P4REVERIFY2..4 recurring minor (fourth round naming it): no test ---
# asserted U17 golden 11 / U27 golden 12 -- the CC-20 property the whole A6
# input-only repair exists to guarantee. The clearance budgets a_v (U17 sec
# 6b) and a_c (U27 sec 6b) are pure functions of the quantities pinned here.


def test_input_only_quantities_are_bit_identical_across_candidate_drawings() -> None:
    """u, extents, masses, and applicability never depend on positions.

    Across candidate drawings of one (graph, StyleContract), every input to
    the a_v / a_c budget constructions -- the intrinsic unit, the derived
    node and label box half-extents, the node masses, and each facet's
    applicability -- must be bit-identical (U17 golden 11, U27 golden 12).
    """

    from dagua.eval.ruler_v4 import evaluate_facet

    graph = GraphSemantics(
        tuple(f"n{index}" for index in range(9)),
        tuple((index, index + 1) for index in range(8)) + ((0, 4), (2, 6)),
        node_masses=tuple(1.0 + 0.25 * index for index in range(9)),
        clusters={"left": (0, 1, 2, 3), "right": (5, 6, 7, 8)},
    )
    generator = torch.Generator().manual_seed(11)
    candidates = [
        30.0 * torch.rand((9, 2), generator=generator, dtype=torch.float64) for _ in range(4)
    ]
    candidates.append(
        torch.tensor([[4.0 * index, 0.5 * index**2] for index in range(9)], dtype=torch.float64)
    )
    candidates.append(1e3 * candidates[0])
    fingerprints = []
    for positions in candidates:
        routes = tuple(
            Route(index, torch.stack((positions[source], positions[target])))
            for index, (source, target) in enumerate(graph.edges)
        )
        result = ingest(
            graph,
            DrawingScene(positions, routes),
            StyleContract(),
            ObservationProfile(visible_channels=frozenset({"nodes", "routes", "cluster_labels"})),
        )
        assert isinstance(result, ValidScene)
        scene = result.scene
        u17 = evaluate_facet("U17", scene, alpha_grid_index=1)
        u27 = evaluate_facet("U27", scene, alpha_grid_index=1)
        fingerprints.append(
            (
                scene.intrinsic_unit,
                tuple(tuple(box.half_extents.tolist()) for box in scene.node_boxes),
                tuple(tuple(box.half_extents.tolist()) for box in scene.node_label_boxes),
                tuple(scene.graph.node_masses),
                u17.state,
                u17.raw["pair_count"],
                u27.state,
            )
        )
    assert all(item == fingerprints[0] for item in fingerprints[1:])


def test_snap_unit_absorbs_dust_and_passes_real_violations() -> None:
    """The producer-side snap clamps <= 1e-12 excess and nothing more."""

    from dagua.eval.ruler_v4._util import snap_unit

    assert snap_unit(-8.008566e-17) == 0.0
    assert snap_unit(1.0 + 2.220446049250313e-16) == 1.0
    assert snap_unit(0.0) == 0.0
    assert snap_unit(1.0) == 1.0
    assert snap_unit(0.5) == 0.5
    # Beyond dust is a real range violation: it must reach the guard intact.
    assert snap_unit(-1e-9) == -1e-9
    assert snap_unit(1.0 + 1e-9) == 1.0 + 1e-9


def test_u30_derived_placement_scores_on_every_alpha_row() -> None:
    """Labels landing exactly on pad_target must score, not crash the blend."""

    from dagua.eval.ruler_v4 import evaluate_facet

    generator = torch.Generator().manual_seed(2)
    lumps = []
    for cluster in range(3):
        base = torch.tensor([25.0 * cluster, 0.0], dtype=torch.float64)
        lumps.append(base + 4.0 * torch.rand((6, 2), generator=generator, dtype=torch.float64))
    positions = torch.cat(lumps)
    graph = GraphSemantics(
        tuple(f"n{index}" for index in range(18)),
        (),
        clusters={
            f"c{cluster}": tuple(range(6 * cluster, 6 * cluster + 6)) for cluster in range(3)
        },
    )
    result = ingest(
        graph,
        DrawingScene(positions),
        StyleContract(),
        ObservationProfile(visible_channels=frozenset({"nodes", "routes", "cluster_labels"})),
    )
    assert isinstance(result, ValidScene)
    for row in range(1, 13):
        facet = evaluate_facet("U30", result.scene, alpha_grid_index=row)
        assert facet.state is ResultState.VALUE
    # Coverage restored for the pre-selection envelope branch.
    unselected = evaluate_facet("U30", result.scene)
    assert unselected.state is ResultState.NA
    assert unselected.reason == "alpha_grid_unselected"


# --- P4REVERIFY3_OPUS blocker 4: a fully coincident retained core is total ---
# collapse (q = 0 -> D^ii = 1, golden 2), and declaring ranks cannot excuse it.


def test_u20a_coincident_core_is_catastrophic_even_with_ranks() -> None:
    """The degenerate zero-cloud fallback reads collapse, not isotropy."""

    coincident = torch.zeros((6, 2), dtype=torch.float64)
    plain = U20a(_node_scene(coincident))
    declared = U20a(_node_scene(coincident, ranks=tuple(range(6)), flow_axis=(0.0, 1.0)))
    assert plain.subterms["U20a.ii"] == pytest.approx(1.0, abs=0.0)
    # Golden 3's mutation clause: the rank block cannot improve the composite.
    assert declared.subterms["U20a.ii"] == pytest.approx(1.0, abs=0.0)
    assert plain.value == declared.value == pytest.approx(1.0, abs=0.0)


# --- P4REVERIFY4_OPUS major 2: the restored unknown-class INVALID was ---
# gated on `ranks is None`, so a ranks-declaring graph with an unparseable
# class string silently kept the layer-profile target. Section 13 case (c)
# conditions INVALID on the class string alone.


def test_u22_unknown_declared_class_is_invalid_even_with_ranks() -> None:
    """Declaring ranks does not buy back a silent unknown-class fallback."""

    from dagua.eval.ruler_v4.structure import U22

    positions = torch.tensor(
        [[0.0, 8.0 * rank] for rank in range(3) for _ in range(2)], dtype=torch.float64
    )
    positions[1::2, 0] = 6.0
    graph = GraphSemantics(
        tuple(f"n{index}" for index in range(6)),
        (),
        ranks=(0, 0, 1, 1, 2, 2),
        declared_graph_class="hypercube",
    )
    result = ingest(
        graph,
        DrawingScene(positions),
        StyleContract(),
        ObservationProfile(visible_channels=frozenset({"nodes", "routes"})),
    )
    assert isinstance(result, ValidScene)
    facet = U22(result.scene)
    assert facet.state is ResultState.INVALID
    assert facet.reason == "unknown_declared_class"


# --- P4REVERIFY3_OPUS major 1: a ranks-only graph keeps its input-only ---
# layer-profile target even though no direction exists to orient the frame.


def _terminal_pair_scene(fork_degrees: float) -> Scene:
    """Ingest two straight routes leaving one node ``fork_degrees`` apart.

    Parameters
    ----------
    fork_degrees : float
        Oriented angle between the two departing tangents.

    Returns
    -------
    Scene
        Validated static scene whose only U11.v pair is the fork.
    """

    import math

    angle = math.radians(fork_degrees)
    positions = torch.tensor(
        [[0.0, 0.0], [10.0, 0.0], [10.0 * math.cos(angle), 10.0 * math.sin(angle)]],
        dtype=torch.float64,
    )
    return _node_scene(positions, edges=((0, 1), (0, 2)))


# --- P4REVERIFY4_OPUS blocker 1: the U11.v angle factor was mirrored at 90 ---
# degrees (acute unoriented angle on tangents documented to point AWAY from
# the node), so a straight through-path -- the most distinguishable terminal
# pair there is -- read Delta_theta = 0 and scored conf = 1.0.


def test_u11_through_path_is_not_merge_identity() -> None:
    """Anti-parallel terminal tangents are maximally distinguishable."""

    from dagua.eval.ruler_v4.edges import U11

    positions = torch.tensor([[0.0, 0.0], [4.0, 0.0], [8.0, 0.0]], dtype=torch.float64)
    through = U11(_node_scene(positions, edges=((0, 1), (1, 2))))
    assert through.subterms["U11.v"] == pytest.approx(0.0, abs=1e-12)


def test_u11_angle_factor_is_monotone_on_the_oriented_range() -> None:
    """conf decreases from the merge-identity limit out to the through-path.

    The pre-fix curve was correct on [0, 90] and reflected on [90, 180]:
    a 179-degree fork scored 0.9955 (near-merge) and a 90-degree fork
    2.3e-16. The contract's merge-identity limit is COINCIDENT tangents
    (U11 sec 5 (v)), so conf must be strictly decreasing in the oriented
    angle.
    """

    from dagua.eval.ruler_v4.edges import U11

    values = [
        U11(_terminal_pair_scene(degrees)).subterms["U11.v"]
        for degrees in (0.5, 45.0, 90.0, 135.0, 179.0)
    ]
    assert values[0] > 0.99
    assert all(left > right for left, right in zip(values, values[1:]))
    assert values[-1] < 1e-12


def test_u11_grid_of_paths_scores_interior_nodes_clean() -> None:
    """A plain grid of straight 4-node paths carries no terminal confusion."""

    from dagua.eval.ruler_v4.edges import U11

    points = [[4.0 * column, 6.0 * row] for row in range(4) for column in range(3)]
    edges = tuple(
        (row * 3 + column, (row + 1) * 3 + column) for row in range(3) for column in range(3)
    )
    facet = U11(_node_scene(torch.tensor(points, dtype=torch.float64), edges=edges))
    assert facet.subterms["U11.v"] == pytest.approx(0.0, abs=1e-12)


# --- P4REVERIFY4_OPUS blocker 3: U38's Jensen-Shannon divergence is ---
# analytically >= 0, but the signed log sum returns ~-8e-17 when the area
# and mass shares agree to within dust, and value_result's unclamped guard
# raised an untyped ValueError. Opus's edgeless-column sweep raised at
# N in {8, 9, 23} of the 39 sizes 2..40; this port of the fixture raises
# at N = 31 at c2a40b0c (same defect, same guard, box sizes differ).


@pytest.mark.parametrize("size", (8, 9, 23, 31))
def test_u38_edgeless_column_scores_instead_of_crashing(size: int) -> None:
    """Every single-node component column returns a typed U38 result."""

    from dagua.eval.ruler_v4.packing import U38

    positions = torch.tensor([[0.0, 3.0 * index] for index in range(size)], dtype=torch.float64)
    graph = GraphSemantics(tuple(f"n{index}" for index in range(size)), ())
    result = ingest(
        graph,
        DrawingScene(positions),
        StyleContract(),
        ObservationProfile(visible_channels=frozenset({"nodes", "routes"})),
    )
    assert isinstance(result, ValidScene)
    facet = U38(result.scene)
    assert facet.state is ResultState.VALUE
    assert facet.subterms["U38.L_prop"] >= 0.0


# --- P4REVERIFY4_OPUS blocker 2: the declared-axis branch compared ---
# depth/breadth against the breadth/depth target, rewarding exactly the
# drawing the contract calls wrong. Fixture is U22.md sec 6's own worked
# example: "a 12-layer DAG whose widest layer holds 40 nodes SHOULD draw
# wide" (A_target = 40/12).


def _layered_scene(layer_gap: float, node_gap: float) -> Scene:
    """Ingest 12 layers (widest 40 nodes) drawn with the given spacings.

    Parameters
    ----------
    layer_gap : float
        Distance between consecutive layers along the declared flow axis.
    node_gap : float
        Distance between neighbours within a layer (the breadth axis).

    Returns
    -------
    Scene
        Validated static scene with declared ranks and flow axis.
    """

    points = []
    ranks = []
    for layer in range(12):
        width = 40 if layer == 0 else 2
        for column in range(width):
            points.append([node_gap * column, layer_gap * layer])
            ranks.append(layer)
    return _node_scene(
        torch.tensor(points, dtype=torch.float64),
        ranks=tuple(ranks),
        flow_axis=(0.0, 1.0),
    )


def test_u22_contract_worked_example_rewards_the_wide_drawing() -> None:
    """The 40-in-12 DAG scores 0 drawn wide and is priced drawn tall."""

    from dagua.eval.ruler_v4.structure import U22

    wide = U22(_layered_scene(layer_gap=4.0, node_gap=4.0))
    tall = U22(_layered_scene(layer_gap=4.0, node_gap=0.3))
    assert wide.raw["measurement"] == "declared_axis"
    assert wide.raw["target"] == pytest.approx(40.0 / 12.0, abs=1e-12)
    assert wide.value == pytest.approx(0.0, abs=0.0)
    assert tall.value is not None and tall.value > 0.25


def test_u22_ranks_only_graph_keeps_layer_profile_target() -> None:
    """U22's target comes from declared ranks alone; the frame stays honest."""

    from dagua.eval.ruler_v4.structure import U22

    positions = torch.tensor(
        [
            [0.0, 0.0],
            [-3.0, 4.0],
            [3.0, 4.0],
            [-3.0, 8.0],
            [3.0, 8.0],
            [-3.0, 12.0],
            [3.0, 12.0],
            [0.0, 16.0],
        ],
        dtype=torch.float64,
    )
    facet = U22(_node_scene(positions, ranks=(0, 1, 1, 2, 2, 3, 3, 4)))
    assert facet.state is ResultState.VALUE
    # Contract quantity (U22.md sec 6): "A_target = max_layer_width /
    # n_layers" = 2/5 = 0.4. The rotation-scan A_obs of the direction-free
    # frame is >= 1 by construction, so DISCREPANCIES.md entry 26 folds the
    # orientation-less target onto the same side of unity: 1 / 0.4 = 2.5.
    profile = 2.0 / 5.0
    assert facet.raw["target"] == pytest.approx(1.0 / profile, abs=0.0)
    assert facet.raw["measurement"] == "frozen_direction_set"


# --- P4REVERIFY5_OPUS blocker: the r4 BLOCKER-2 fix flipped `observed` for ---
# the whole declared-axis branch, but section 6's class table is stated in the
# direction-free elongation convention, so the path/chain kappa was read
# backwards: a declared path drawn correctly ALONG its declared flow axis
# scored 0.791 while the same path drawn ACROSS it scored 0.011.


def _declared_class_scene(
    positions: torch.Tensor,
    declared_graph_class: str,
    flow_axis: Optional[Tuple[float, float]] = (0.0, 1.0),
    **metadata: object,
) -> Scene:
    """Ingest an edgeless scene that declares a graph class.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    declared_graph_class : str
        Frozen exemption-table class string.
    flow_axis : tuple[float, float] or None
        Declared flow axis, if any.
    **metadata : object
        Extra GraphSemantics fields (``tree_depths``, ``lattice_dimensions``).

    Returns
    -------
    Scene
        Validated static scene.
    """

    graph = GraphSemantics(
        tuple(f"n{index}" for index in range(positions.shape[0])),
        (),
        flow_axis=flow_axis,
        declared_graph_class=declared_graph_class,
        **metadata,
    )
    result = ingest(
        graph,
        DrawingScene(positions),
        StyleContract(),
        ObservationProfile(visible_channels=frozenset({"nodes", "routes"})),
    )
    assert isinstance(result, ValidScene)
    return result.scene


def test_u22_declared_path_rewards_drawing_along_its_declared_axis() -> None:
    """The P4REVERIFY5 live pair: along-axis 0.011-ish, across-axis 0.791-ish.

    kappa_class = 8 (U22.md sec 6 table) is an elongation magnitude; a
    path elongates along the declared flow axis, so in the signed
    breadth/depth frame the target is 1/8 (DISCREPANCIES.md entry 30).
    The pinned values are the reviewer's measured pair, which the
    r4 BLOCKER-2 fix had exactly swapped.
    """

    from dagua.eval.ruler_v4.structure import U22

    along = torch.tensor([[0.0, 3.0 * index] for index in range(10)], dtype=torch.float64)
    across = torch.tensor([[3.0 * index, 0.0] for index in range(10)], dtype=torch.float64)
    correct = U22(_declared_class_scene(along, "path"))
    wrong = U22(_declared_class_scene(across, "path"))
    assert correct.raw["target"] == pytest.approx(1.0 / 8.0, abs=0.0)
    assert correct.value == pytest.approx(0.010723, abs=1e-6)
    assert wrong.value == pytest.approx(0.791270, abs=1e-6)
    # The regression's signature was this exact pair, swapped.
    assert correct.value < 0.02 < 0.75 < wrong.value


# --- P2REVIEW_OPUS blocker 1: the free-form reporting-group string was ---
# score-visible under p > 1 (relabelling alone moved l_total 0.4583 -> 0.3606
# on identical rows). V4_SPEC_r4 3.3: groups are a reporting rollup with no
# weight semantics of their own.


def _relabel_probe_table(groups: Tuple[str, ...]):
    """Build the reviewer's four-row probe table under one group labelling.

    Parameters
    ----------
    groups : tuple[str, ...]
        One reporting-group label per row.

    Returns
    -------
    WeightTable
        Four unit-mass rows.
    """

    from dagua.eval.ruler_v4.weights import SubtermWeight, WeightTable

    return WeightTable(
        entries=tuple(
            SubtermWeight(f"F{index}.x", f"F{index}", group, 1.0)
            for index, group in enumerate(groups)
        ),
        d_power=0,
    )


def test_group_relabelling_is_score_inert_under_the_shipped_p_mean() -> None:
    """The Opus P2 probe: (0.9, 0.1, 0.1, 0.1) at p=2 under three labellings."""

    from dagua.eval.ruler_v4.composition import CompositionFamily, CompositionProfile, compose
    from dagua.eval.ruler_v4.scene import value_result

    results = {
        f"F{index}": value_result(value, {f"F{index}.x": value})
        for index, value in enumerate((0.9, 0.1, 0.1, 0.1))
    }
    profile = CompositionProfile(CompositionFamily.P_MEAN, power=2.0)
    labellings = (
        ("A", "B", "B", "B"),
        ("A", "A", "B", "B"),
        ("A", "B", "C", "D"),
        ("A", "A", "A", "A"),
    )
    totals = {
        compose(results, _relabel_probe_table(labels), profile).l_total for labels in labellings
    }
    assert len(totals) == 1


def test_p_mean_sees_a_catastrophic_row_inside_a_populated_group() -> None:
    """The Opus P2 within-group probe: (1,0,...) != (0.111...,) at p > 1."""

    from dagua.eval.ruler_v4.composition import CompositionFamily, CompositionProfile, compose
    from dagua.eval.ruler_v4.scene import value_result
    from dagua.eval.ruler_v4.weights import SubtermWeight, WeightTable

    table = WeightTable(
        entries=tuple(SubtermWeight(f"A.{index}", "A", "one_group", 1.0) for index in range(9)),
        d_power=0,
    )
    concentrated = {
        "A": value_result(1.0, {f"A.{index}": 1.0 if index == 0 else 0.0 for index in range(9)})
    }
    diffuse = {"A": value_result(1.0 / 9.0, {f"A.{index}": 1.0 / 9.0 for index in range(9)})}
    profile = CompositionProfile(CompositionFamily.P_MEAN, power=2.0)
    concentrated_result = compose(concentrated, table, profile)
    diffuse_result = compose(diffuse, table, profile)
    assert concentrated_result.l_mean == pytest.approx(diffuse_result.l_mean)
    assert concentrated_result.l_total > diffuse_result.l_total
