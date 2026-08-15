"""Worked examples for edge-geometry and routed-edge facet contracts."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping, Optional, Tuple

import pytest
import torch

from dagua.eval.ruler_v4._tracing import trace_subterms
from dagua.eval.ruler_v4.composition import CompositionFamily, CompositionProfile
from dagua.eval.ruler_v4.contracts import CONTRACTS
from dagua.eval.ruler_v4.edges import (
    U07,
    U08,
    U10,
    U11,
    U12,
    U13,
    U15,
    U16,
    _parallel_route_integral,
)
from dagua.eval.ruler_v4.headline import HeadlineProfile
from dagua.eval.ruler_v4.ingestion import ingest
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
from dagua.eval.ruler_v4.score import ScoringProfiles
from dagua.eval.ruler_v4.surrogate.traced import (
    TracedSoftScore,
    build_traced_scene,
    score_scene_soft,
)
from dagua.eval.ruler_v4.weight_table import (
    GATE_DIAGNOSTIC_FACETS,
    REQUIRED_PRIOR_FLOOR_FACETS,
    ParameterProvenance,
    SubtermWeight,
    WeightTable,
)


def _scene(
    positions: torch.Tensor,
    edges: Tuple[Tuple[int, int], ...],
    routes: Optional[Tuple[Route, ...]] = None,
    graph_options: Optional[Mapping[str, Any]] = None,
    edge_label_positions: Optional[torch.Tensor] = None,
) -> Scene:
    """Ingest one routed worked-example scene.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edges : tuple[tuple[int, int], ...]
        Canonical graph edges.
    routes : tuple[Route, ...] or None
        Explicit routes, or straight routes when omitted.
    graph_options : mapping[str, Any] or None
        Optional GraphSemantics field overrides.
    edge_label_positions : torch.Tensor or None
        Optional label centers with shape ``[E, 2]``.

    Returns
    -------
    Scene
        Validated routed scene.
    """

    options = dict(graph_options or {})
    graph = GraphSemantics(
        tuple(f"n{index}" for index in range(positions.shape[0])), edges, **options
    )
    route_values = routes or tuple(
        Route(index, torch.stack((positions[source], positions[target])))
        for index, (source, target) in enumerate(edges)
    )
    profile = ObservationProfile(visible_channels=frozenset({"nodes", "routes", "edge_labels"}))
    result = ingest(
        graph,
        DrawingScene(
            positions,
            route_values,
            edge_label_positions=edge_label_positions,
        ),
        StyleContract(),
        profile,
    )
    assert isinstance(result, ValidScene)
    return result.scene


def test_u07_crossing_free_routes_have_exact_zero_defect() -> None:
    """U07's crossing-free golden returns exact zeros rather than NA."""

    positions = torch.tensor([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    result = U07(_scene(positions, ((0, 1), (1, 2))))
    assert result.state is ResultState.VALUE
    assert result.value == pytest.approx(0.0, abs=0.0)
    assert result.raw["crossing_count"] == 0


def test_u07_remote_perpendicular_crossing_pins_guarded_normalization() -> None:
    """U07 G02 pins one zero-severity event and the guarded opportunity count."""

    positions = torch.tensor(
        [[-10.0, 0.0], [10.0, 0.0], [0.0, -10.0], [0.0, 10.0]],
        dtype=torch.float64,
    )
    result = U07(_scene(positions, ((0, 1), (2, 3))), gamma=3.0, lambda_T=1.0)
    assert result.state is ResultState.VALUE
    assert result.raw["crossing_count"] == 1
    assert result.raw["events"][0]["severity"] == pytest.approx(0.0, abs=1e-30)
    assert result.raw["opportunity_guarded"] == 2
    assert result.raw["base_raw"] == pytest.approx(1.0, abs=0.0)
    assert result.raw["tail_raw"] == pytest.approx(0.0, abs=0.0)
    assert result.value == pytest.approx(2.0 / 3.0, abs=1e-15)


@pytest.mark.parametrize(
    ("gamma", "lambda_T"),
    ((0.0, 0.5), (3.1, 0.5), (1.0, -0.1), (1.0, 1.1)),
)
def test_u07_fitted_parameter_ranges_are_enforced(gamma: float, lambda_T: float) -> None:
    """Reject U07 fitted values outside the contract's declared ranges.

    Parameters
    ----------
    gamma, lambda_T : float
        Invalid fitted-parameter pair supplied by pytest.
    """

    scene = _scene(torch.tensor([[0.0, 0.0], [2.0, 0.0]]), ((0, 1),))
    with pytest.raises(ValueError):
        U07(scene, gamma=gamma, lambda_T=lambda_T)


def test_u08_equal_six_spoke_star_has_exact_zero_defect() -> None:
    """U08's equal 60-degree spoke fixture has exact fair angular resolution."""

    angles = torch.arange(6, dtype=torch.float64) * (2.0 * math.pi / 6.0)
    leaves = 5.0 * torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    positions = torch.cat((torch.zeros((1, 2), dtype=torch.float64), leaves), dim=0)
    result = U08(_scene(positions, tuple((0, index) for index in range(1, 7))))
    assert result.state is ResultState.VALUE
    # U08 section 3 freezes exact zero for the uniform angular fixture.
    assert result.value == pytest.approx(0.0, abs=0.0)


def test_u10_clean_route_has_exact_zero_clearance_burden() -> None:
    """U10's clean route beyond the compact clearance band contributes zero."""

    positions = torch.tensor([[0.0, 0.0], [4.0, 0.0], [2.0, 5.0]])
    result = U10(_scene(positions, ((0, 1),)))
    assert result.state is ResultState.VALUE
    assert result.value == pytest.approx(0.0, abs=0.0)


def test_u11_straight_route_hits_all_five_anchored_zeros() -> None:
    """U11's straight zero-bend route has zero on every routed-quality row."""

    positions = torch.tensor([[0.0, 0.0], [4.0, 0.0]])
    result = U11(_scene(positions, ((0, 1),)))
    assert result.state is ResultState.VALUE
    assert set(result.subterms.values()) == {0.0}
    assert result.subterms["U11.i"] == pytest.approx(0.0, abs=0.0)


def test_u11_terminal_disk_clears_coincident_nonterminal_box() -> None:
    """U11 section 7.2 removes the terminal 16-gon from every obstacle."""

    positions = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 0.0]])
    result = U11(_scene(positions, ((0, 1),)))
    assert result.state is ResultState.VALUE
    assert result.raw["edges"][0]["baseline_length"] == pytest.approx(10.0, abs=0.0)
    assert result.raw["edges"][0]["baseline_turn"] == pytest.approx(0.0, abs=0.0)


def test_u11_tangentless_terminal_pair_is_not_best_case() -> None:
    """U11 treats a zero-arc terminal as the coincidence limit."""

    positions = torch.tensor([[0.0, 0.0], [0.0, 0.0], [4.0, 0.0], [0.0, 4.0]])
    edges = ((0, 1), (0, 2), (0, 3))
    routes = tuple(
        Route(index, torch.stack((positions[source], positions[target])))
        for index, (source, target) in enumerate(edges)
    )
    result = U11(_scene(positions, edges, routes))
    # U11 section 5(v): at coincidence both tangent-less pairs read the full
    # gap factor (1.0 each) and the tangent pair is ~0, so d5 = 2/3 by hand.
    assert result.subterms["U11.v"] == pytest.approx(2.0 / 3.0, abs=1e-9)


def test_u11_distant_tangentless_terminal_pair_earns_its_separation() -> None:
    """The closed form's gap factor stays live when a tangent is undefined."""

    positions = torch.tensor([[0.0, 0.0], [0.2, 0.0], [8.0, 0.0], [0.0, 8.0]])
    edges = ((0, 1), (0, 2), (0, 3))
    routes = (
        Route(0, torch.tensor([[0.6, 0.0], [0.6, 0.0]])),
        Route(1, torch.tensor([[0.0, 0.0], [8.0, 0.0]])),
        Route(2, torch.tensor([[0.0, 0.0], [0.0, 8.0]])),
    )
    result = U11(_scene(positions, edges, routes))
    # U11 golden G10's falling branch: separated anchors decay; a flat 1.0 for
    # tangent-less pairs would score this 2/3 like the coincident fixture.
    assert result.subterms["U11.v"] < 0.15


def test_u12_collinear_subdivided_path_has_zero_continuity_defect() -> None:
    """U12 is invariant to collinear route subdivision and returns exact zero."""

    positions = torch.stack((torch.arange(7, dtype=torch.float64), torch.zeros(7)), dim=1)
    edges = tuple((index, index + 1) for index in range(6))
    result = U12(_scene(positions, edges))
    assert result.state is ResultState.VALUE
    assert result.value == pytest.approx(0.0, abs=0.0)


def test_u13_well_separated_parallel_routes_hit_compact_zero() -> None:
    """U13's support-edge golden is exactly zero beyond 3.5 intrinsic units."""

    positions = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 7.0], [10.0, 7.0]])
    result = U13(_scene(positions, ((0, 1), (2, 3))))
    assert result.state is ResultState.VALUE
    assert result.value == pytest.approx(0.0, abs=0.0)


def test_u13_ten_unit_parallel_pair_matches_hand_integral() -> None:
    """U13 golden 2 pins the exact 0.5u-separation polynomial integral."""

    left = torch.tensor([[0.0, 0.0], [10.0, 0.0]], dtype=torch.float64)
    right = torch.tensor([[0.0, 0.5], [10.0, 0.5]], dtype=torch.float64)
    expected = 10.0 * (35.0 / 36.0) ** 2
    assert _parallel_route_integral(left, right, 1.0) == pytest.approx(expected, abs=1e-14)


def test_u13_nearest_extent_is_invariant_to_other_route_subdivision() -> None:
    """A bend-vertex subdivision cannot double-count U13's shared extent."""

    left = torch.tensor([[0.0, 0.0], [10.0, 0.0]], dtype=torch.float64)
    unsplit = torch.tensor([[0.0, 0.5], [10.0, 0.5]], dtype=torch.float64)
    split = torch.tensor([[0.0, 0.5], [5.0, 0.5], [10.0, 0.5]], dtype=torch.float64)
    expected = _parallel_route_integral(left, unsplit, 1.0)
    assert _parallel_route_integral(left, split, 1.0) == pytest.approx(expected, abs=1e-14)


def test_u15_separated_parallel_arcs_have_zero_merge_defect() -> None:
    """U15's doubled-edge golden is zero when relative separation exceeds five percent."""

    positions = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
    routes = (
        Route(0, torch.tensor([[0.0, 0.0], [2.0, 1.0], [8.0, 1.0], [10.0, 0.0]])),
        Route(1, torch.tensor([[0.0, 0.0], [2.0, -1.0], [8.0, -1.0], [10.0, 0.0]])),
    )
    result = U15(_scene(positions, ((0, 1), (0, 1)), routes))
    assert result.state is ResultState.VALUE
    assert result.subterms["U15.i"] == pytest.approx(0.0, abs=0.0)


def _complete_table() -> WeightTable:
    """Build the acceptance battery's complete explicit weight table.

    Returns
    -------
    WeightTable
        One entry per scored subterm; gate diagnostics carry weight zero.
    """

    entries = tuple(
        SubtermWeight(
            subterm_id=subterm_id,
            facet_id=facet_id,
            group=facet_id[:3],
            weight=0.0 if facet_id in GATE_DIAGNOSTIC_FACETS else 1.0,
            prior_driven=facet_id in REQUIRED_PRIOR_FLOOR_FACETS,
            diagnostic=facet_id in GATE_DIAGNOSTIC_FACETS,
            provenance_class=(
                None
                if facet_id in GATE_DIAGNOSTIC_FACETS
                else "preregistered_prior"
                if facet_id in REQUIRED_PRIOR_FLOOR_FACETS
                else "contract_frozen"
            ),
        )
        for facet_id, contract in CONTRACTS.items()
        for subterm_id in contract.scored_subterms
    )
    return WeightTable(
        entries=entries,
        d_power=20,
        prior_floors={facet_id: 1.0 for facet_id in REQUIRED_PRIOR_FLOOR_FACETS},
    )


def _soft_profiles() -> ScoringProfiles:
    """Build the traced-probe scoring profiles.

    Returns
    -------
    ScoringProfiles
        Frozen composition/headline profiles used by the surrogate probe.
    """

    return ScoringProfiles(
        composition=CompositionProfile(CompositionFamily.P_MEAN, power=2.0),
        headline=HeadlineProfile(index_span=100.0, loss_scale=1.0, version="probe"),
        measurement_version="probe-measurement",
        policy_version="probe-policy",
        alpha_grid_index=1,
        parameter_provenance={
            "composition.power": ParameterProvenance("preregistered_prior"),
            "headline.index_span": ParameterProvenance("contract_frozen"),
            "headline.loss_scale": ParameterProvenance("preregistered_prior"),
            "alpha_grid_index": ParameterProvenance("preregistered_prior"),
        },
    )


def _semantic_scene(
    positions: Optional[torch.Tensor] = None,
    route_overrides: Optional[Mapping[int, torch.Tensor]] = None,
) -> Scene:
    """Build the traced-baseline probe's semantic fixture scene.

    Parameters
    ----------
    positions : torch.Tensor or None
        Optional ``[8, 2]`` position override.
    route_overrides : mapping[int, torch.Tensor] or None
        Optional explicit route points per edge index (non-chord producer
        routes; input-owned constants on the traced path).

    Returns
    -------
    Scene
        The eight-node directed layered fixture with straight chord routes.
    """

    if positions is None:
        positions = torch.tensor(
            [
                [0.0, 0.0],
                [2.0, -2.0],
                [2.0, 2.0],
                [4.0, -2.0],
                [4.0, 2.0],
                [6.0, -2.0],
                [6.0, 2.0],
                [8.0, 0.0],
            ],
            dtype=torch.float64,
        )
    overrides = dict(route_overrides or {})
    edges: Tuple[Tuple[int, int], ...] = (
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    )
    graph = GraphSemantics(
        node_ids=tuple(f"n{index}" for index in range(8)),
        edges=edges,
        directed=True,
        node_labels=tuple(f"n{index}" for index in range(8)),
        edge_labels=tuple(None for _ in edges),
        clusters={"left": (0, 1, 2, 3), "right": (4, 5, 6, 7), "parent": tuple(range(8))},
        cluster_parents={"left": "parent", "right": "parent"},
        ranks=(0, 1, 1, 2, 2, 3, 3, 4),
        roots=(0,),
        feedback=tuple(False for _ in edges),
        edge_weights=tuple(float(index + 1) for index in range(len(edges))),
        weight_semantics="distance_cost",
        required_primitives=frozenset({"nodes", "routes"}),
    )
    graph = replace(
        graph,
        tree_parents=(None, 0, 0, 1, 2, 3, 4, 5),
        tree_depths=(0, 1, 1, 2, 2, 3, 3, 4),
        tree_layout="layered",
        flow_axis=(1.0, 0.0),
        ordered_children={0: (1, 2)},
    )
    routes = tuple(
        Route(
            index,
            overrides[index]
            if index in overrides
            else torch.stack((positions[source], positions[target])),
        )
        for index, (source, target) in enumerate(edges)
    )
    drawing = DrawingScene(positions, routes, ("nodes", "routes", "node_labels"))
    result = ingest(
        graph,
        drawing,
        StyleContract(),
        ObservationProfile(visible_channels=frozenset({"nodes", "routes", "node_labels"})),
    )
    assert isinstance(result, ValidScene)
    return result.scene


def _position_gradient_norm(traced: TracedSoftScore, subterm_id: str) -> Optional[float]:
    """Measure one traced subterm's position-gradient norm.

    Parameters
    ----------
    traced : TracedSoftScore
        Traced scoring result.
    subterm_id : str
        Manifest subterm id present in the trace buffer.

    Returns
    -------
    float or None
        Euclidean gradient norm, or None when no graph reaches the leaf.
    """

    tensor = traced.traced_subterms[subterm_id]
    if not tensor.requires_grad:
        return None
    (gradient,) = torch.autograd.grad(
        tensor, traced.positions, retain_graph=True, allow_unused=True
    )
    if gradient is None:
        return None
    return float(torch.linalg.vector_norm(gradient))


def test_traced_semantic_scene_edge_rows_are_live() -> None:
    """The probe fixture's position-responsive edge rows carry live gradients.

    On this straight-route fixture U11.v, U12.headline, and U13.i genuinely
    respond to positions (verified by finite differences); U11.i and U11.iv sit
    at anchored zeros whose exact local gradient is zero; U7 has an empty event
    population and U11.iii/U10 sit on exact anchored zeros, so those rows are
    honest constants (contract zero-event / zero-onset arms).
    """

    scene = _semantic_scene()
    traced = score_scene_soft(scene, _complete_table(), _soft_profiles())
    for live_row in ("U11.v", "U12.headline", "U13.i"):
        assert live_row in traced.bound_subterms
        norm = _position_gradient_norm(traced, live_row)
        assert norm is not None and math.isfinite(norm) and norm > 0.0
    for flat_row in ("U11.i", "U11.iv"):
        # Graph reaches the position leaf; the anchored-zero gradient is exact 0.
        assert _position_gradient_norm(traced, flat_row) == pytest.approx(0.0, abs=0.0)
    # Empty severe-event population: U7 rows are exact zeros with no geometry
    # in the value at all (U07 contract sec 7, zero-events arm).
    assert U07(scene).raw["crossing_count"] == 0
    assert "U7.base" in traced.constant_subterms
    assert "U7.tail" in traced.constant_subterms
    # Exact facet values are unchanged from a direct un-traced evaluation.
    exact = {
        **U11(scene).subterms,
        **U12(scene).subterms,
        **U13(scene).subterms,
    }
    for subterm_id, value in exact.items():
        if subterm_id in traced.traced_subterms:
            traced_value = float(traced.traced_subterms[subterm_id].detach())
            assert traced_value == pytest.approx(value, abs=1e-12)
    # The whole surrogate differentiates end to end without NaN.
    (total_gradient,) = torch.autograd.grad(
        traced.soft.l_total, traced.positions, retain_graph=True
    )
    assert bool(torch.isfinite(total_gradient).all())


def test_traced_crossing_scene_u07_base_is_live() -> None:
    """A transversal crossing makes U7.base a live function of positions.

    Swapping the y-coordinates of layer-2 nodes crosses the (1,3)/(2,4) and
    (3,5)/(4,6) chord pairs while keeping the fixture's graph semantics.
    """

    positions = torch.tensor(
        [
            [0.0, 0.0],
            [2.0, -2.0],
            [2.0, 2.0],
            [4.0, 2.0],
            [4.0, -2.0],
            [6.0, -2.0],
            [6.0, 2.0],
            [8.0, 0.0],
        ],
        dtype=torch.float64,
    )
    scene = _semantic_scene(positions)
    exact = U07(scene)
    assert exact.raw["crossing_count"] >= 1
    traced = score_scene_soft(scene, _complete_table(), _soft_profiles())
    assert "U7.base" in traced.bound_subterms
    norm = _position_gradient_norm(traced, "U7.base")
    assert norm is not None and math.isfinite(norm) and norm > 0.0
    traced_value = float(traced.traced_subterms["U7.base"].detach())
    assert traced_value == pytest.approx(exact.subterms["U7.base"], abs=1e-12)
    rerun = U07(_semantic_scene(positions.clone()))
    assert rerun.subterms["U7.base"] == pytest.approx(exact.subterms["U7.base"], abs=0.0)


def test_traced_repeat_crossing_scene_u07_tail_is_live() -> None:
    """Repeated shallow crossings put mass in U7.tail and keep it live.

    The wiggling route is a non-chord producer route (an input-owned constant
    channel in this surrogate version); the gradient flows through the OTHER
    edge's live chord endpoints, which is exactly the honest channel statement.
    """

    positions = torch.tensor(
        [
            [0.0, 0.0],
            [2.0, -2.0],
            [2.0, -1.95],
            [4.0, -2.0],
            [4.0, -1.95],
            [6.0, -2.0],
            [6.0, 2.0],
            [8.0, 0.0],
        ],
        dtype=torch.float64,
    )
    # Edge 3 is (2, 4); its shallow wiggle crosses edge (1, 3)'s chord (the
    # y = -2 corridor) four times: repeat burden + shallow angles beat the
    # tail threshold s0 = 0.5 * gamma.
    wiggle = torch.tensor(
        [
            [2.0, -1.95],
            [2.5, -2.05],
            [3.0, -1.95],
            [3.5, -2.05],
            [4.0, -1.95],
        ],
        dtype=torch.float64,
    )
    scene = _semantic_scene(positions, {3: wiggle})
    exact = U07(scene)
    assert exact.raw["crossing_count"] >= 4
    assert exact.raw["tail_raw"] > 0.0
    traced = score_scene_soft(scene, _complete_table(), _soft_profiles())
    for row in ("U7.base", "U7.tail"):
        assert row in traced.bound_subterms
        norm = _position_gradient_norm(traced, row)
        assert norm is not None and math.isfinite(norm) and norm > 0.0
        traced_value = float(traced.traced_subterms[row].detach())
        assert traced_value == pytest.approx(exact.subterms[row], abs=1e-12)


def _direct_traced_subterms(
    scene: Scene, facet: Any
) -> Tuple[Mapping[str, torch.Tensor], torch.Tensor]:
    """Trace one facet directly through the surrogate seam.

    Parameters
    ----------
    scene : Scene
        Validated scene.
    facet : callable
        Facet entry point.

    Returns
    -------
    tuple[mapping[str, torch.Tensor], torch.Tensor]
        Trace buffer and the position leaf.
    """

    positions = scene.positions.detach().clone().requires_grad_(True)
    traced_scene = build_traced_scene(scene, positions)
    with trace_subterms() as buffer:
        facet(traced_scene)
    return buffer, positions


def _leaf_gradient_norm(buffer: Mapping[str, torch.Tensor], key: str, leaf: torch.Tensor) -> float:
    """Return one traced subterm's gradient norm against the position leaf."""

    (gradient,) = torch.autograd.grad(buffer[key], leaf, retain_graph=True, allow_unused=True)
    assert gradient is not None
    return float(torch.linalg.vector_norm(gradient))


def test_traced_direct_u08_u10_u15_u16_rows_are_live_on_worked_scenes() -> None:
    """Rows inactive on the probe fixture carry live gradients on their scenes.

    U08 needs a degree->=3 node, U10 an in-band pair, U15.ii a self-loop near a
    live obstacle, U16.ii a declared edge label; each is exercised on its own
    worked-example fixture through the same traced seam, with the exact float
    value pinned against the direct un-traced evaluation.
    """

    # U08: pinched star (two spokes at a small angle) has a live LSE defect.
    angles = torch.tensor([0.0, 0.15, 2.0, 3.0, 4.0, 5.2], dtype=torch.float64)
    leaves = 5.0 * torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    star_positions = torch.cat((torch.zeros((1, 2), dtype=torch.float64), leaves), dim=0)
    star = _scene(star_positions, tuple((0, index) for index in range(1, 7)))
    buffer, leaf = _direct_traced_subterms(star, U08)
    assert _leaf_gradient_norm(buffer, "U08.headline", leaf) > 0.0
    assert float(buffer["U08.headline"].detach()) == pytest.approx(
        U08(star).subterms["U08.headline"], abs=1e-12
    )

    # U10: a route grazing a third node's clearance band is live.
    near_positions = torch.tensor([[0.0, 0.0], [4.0, 0.0], [2.0, 0.4]], dtype=torch.float64)
    near = _scene(near_positions, ((0, 1),))
    exact_u10 = U10(near)
    assert exact_u10.subterms["U10.headline"] > 0.0
    buffer, leaf = _direct_traced_subterms(near, U10)
    assert _leaf_gradient_norm(buffer, "U10.headline", leaf) > 0.0
    assert float(buffer["U10.headline"].detach()) == pytest.approx(
        exact_u10.subterms["U10.headline"], abs=1e-12
    )

    # U15.ii: the loop route is an input-owned constant, but the clearance
    # integral against ANOTHER node's live box carries position gradient.
    loop_positions = torch.tensor([[0.0, 0.0], [1.2, 0.6]], dtype=torch.float64)
    loop_route = Route(
        0,
        torch.tensor(
            [[0.0, 0.0], [1.0, 0.5], [1.4, 0.0], [1.0, -0.5], [0.0, 0.0]],
            dtype=torch.float64,
        ),
    )
    loop = _scene(loop_positions, ((0, 0),), (loop_route,))
    exact_u15 = U15(loop)
    assert exact_u15.subterms["U15.ii"] > 0.0
    buffer, leaf = _direct_traced_subterms(loop, U15)
    assert _leaf_gradient_norm(buffer, "U15.ii", leaf) > 0.0
    assert float(buffer["U15.ii"].detach()) == pytest.approx(
        exact_u15.subterms["U15.ii"], abs=1e-12
    )

    # U16.ii: the label box is an input-owned constant, but its anchoring
    # distance to the live chord route carries position gradient.
    label_positions = torch.tensor([[0.0, 0.0], [10.0, 0.0]], dtype=torch.float64)
    labeled = _scene(
        label_positions,
        ((0, 1),),
        None,
        {"edge_labels": ("edge",)},
        torch.tensor([[5.0, 4.0]]),
    )
    exact_u16 = U16(labeled)
    assert exact_u16.subterms["U16.ii"] > 0.0
    buffer, leaf = _direct_traced_subterms(labeled, U16)
    assert _leaf_gradient_norm(buffer, "U16.ii", leaf) > 0.0
    assert float(buffer["U16.ii"].detach()) == pytest.approx(
        exact_u16.subterms["U16.ii"], abs=1e-12
    )


def test_u16_clean_edge_label_has_zero_overlap() -> None:
    """U16's clean labeled path has no label-to-obstacle overlap."""

    positions = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
    route = Route(0, torch.tensor([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]]))
    result = U16(
        _scene(
            positions,
            ((0, 1),),
            (route,),
            {"edge_labels": ("edge",)},
            torch.tensor([[5.0, 1.5]]),
        )
    )
    assert result.state is ResultState.VALUE
    assert result.subterms["U16.i"] == pytest.approx(0.0, abs=0.0)
