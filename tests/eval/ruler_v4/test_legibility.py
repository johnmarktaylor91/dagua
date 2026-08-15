"""Worked examples for primitive legibility and frame-economy contracts."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Tuple

import pytest
import torch

from dagua.eval.ruler_v4._tracing import trace_subterms
from dagua.eval.ruler_v4.composition import CompositionFamily, CompositionProfile
from dagua.eval.ruler_v4.contracts import CONTRACTS
from dagua.eval.ruler_v4.headline import HeadlineProfile
from dagua.eval.ruler_v4.ingestion import ingest
from dagua.eval.ruler_v4.legibility import U17, U18, U19, U21, U20a, U20b
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
    edges: Tuple[Tuple[int, int], ...] = (),
    labelled: bool = False,
) -> Scene:
    """Ingest one legibility worked-example scene.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edges : tuple[tuple[int, int], ...]
        Canonical graph edges.
    labelled : bool
        Whether every node has a visible declared label.

    Returns
    -------
    Scene
        Validated static scene.
    """

    labels = tuple(f"n{index}" for index in range(positions.shape[0])) if labelled else ()
    graph = GraphSemantics(
        tuple(f"n{index}" for index in range(positions.shape[0])),
        edges,
        node_labels=labels,
    )
    routes = tuple(
        Route(index, torch.stack((positions[source], positions[target])))
        for index, (source, target) in enumerate(edges)
    )
    channels = {"nodes", "routes"}
    if labelled:
        channels.add("node_labels")
    result = ingest(
        graph,
        DrawingScene(positions, routes),
        StyleContract(),
        ObservationProfile(visible_channels=frozenset(channels)),
    )
    assert isinstance(result, ValidScene)
    return result.scene


def test_u17_generously_separated_nodes_have_exact_zero_defect() -> None:
    """U17's clear pair lies beyond the compact half-unit clearance support."""

    result = U17(_scene(torch.tensor([[0.0, 0.0], [5.0, 0.0]])), None)
    assert result.state is ResultState.NA
    assert result.reason == "alpha_grid_unselected"
    assert result.raw["grid_envelope"] == pytest.approx((0.0, 0.0), abs=0.0)


def test_u17_explicit_grid_row_returns_its_exact_scalar() -> None:
    """An explicit shared-grid row selects a scalar without inventing a default row."""

    scene = _scene(torch.tensor([[0.0, 0.0], [5.0, 0.0]]))
    selected = U17(scene, 1)
    assert selected.state is ResultState.VALUE
    assert selected.raw["alpha_grid_index"] == 1
    assert selected.raw["alpha_grid_name"] == "AC15_AH00"
    assert selected.value == pytest.approx(0.0, abs=0.0)


@pytest.mark.parametrize("alpha_grid_index", (0, 13, True))
def test_u17_rejects_non_grid_parameters(alpha_grid_index: int) -> None:
    """Reject off-list shared-grid parameters.

    Parameters
    ----------
    alpha_grid_index : int
        Invalid row supplied by pytest.
    """

    scene = _scene(torch.tensor([[0.0, 0.0], [5.0, 0.0]]))
    with pytest.raises(ValueError):
        U17(scene, alpha_grid_index)


def test_u18_separated_node_labels_have_zero_overlap_terms() -> None:
    """U18's separated declared labels have no label, node, or route collision."""

    result = U18(_scene(torch.tensor([[0.0, 0.0], [6.0, 0.0]]), labelled=True), None)
    assert result.state is ResultState.NA
    assert result.reason == "alpha_grid_unselected"
    assert result.raw["grid_envelope"] == pytest.approx((0.0, 0.0), abs=0.0)


def test_u19_v4_profile_without_physical_output_is_typed_na() -> None:
    """U19's frozen v4.0 applicability golden is NA without physical size."""

    result = U19(_scene(torch.tensor([[0.0, 0.0], [6.0, 0.0]]), labelled=True))
    assert result.state is ResultState.NA
    assert result.reason == "no_declared_physical_size"


def test_u19_physical_label_fixture_pins_exact_legibility_loss() -> None:
    """U19 pins the declared physical scale and quintic legibility loss."""

    positions = torch.tensor([[0.0, 0.0], [4.0, 0.0]], dtype=torch.float64)
    graph = GraphSemantics(("n0", "n1"), ((0, 1),), node_labels=("a", "b"))
    style = StyleContract(
        physical_output={
            "output_width": 10.0,
            "output_height": 10.0,
            "h_font": 1.0,
            "h_floor": 2.0,
        }
    )
    result = ingest(
        graph,
        DrawingScene(positions),
        style,
        ObservationProfile(visible_channels=frozenset({"nodes", "routes", "node_labels"})),
    )
    assert isinstance(result, ValidScene)
    facet = U19(result.scene)
    assert facet.state is ResultState.VALUE
    # U21's small-N frame has x half-extent 6 here, so the 10-unit viewport
    # gives m_phys=10/12 and r_l=(1)*(10/12)/2=5/12.
    ratio = 5.0 / 12.0
    expected = 1.0 - ratio**3 * (ratio * (6.0 * ratio - 15.0) + 10.0)
    assert 0.0 < facet.value < 1.0
    # U19 contract golden 3: "a label at half the floor scores in (0,1)."
    assert facet.value == pytest.approx(expected, abs=1e-15)


def test_u20a_declared_rank_column_uses_residual_frame() -> None:
    """E2 removes rank-explained axis variance from a perfect layered column."""

    positions = torch.tensor([[0.0, 4.0 * rank] for rank in range(6)], dtype=torch.float64)
    graph = GraphSemantics(
        tuple(f"n{rank}" for rank in range(6)),
        (),
        ranks=tuple(range(6)),
        flow_axis=(0.0, 1.0),
    )
    result = ingest(
        graph,
        DrawingScene(positions),
        StyleContract(),
        ObservationProfile(visible_channels=frozenset({"nodes", "routes"})),
    )
    assert isinstance(result, ValidScene)
    facet = U20a(result.scene)
    # U20a E2: correctly layered declared-axis collinearity is expected.
    assert facet.subterms["U20a.ii"] == pytest.approx(0.0, abs=0.0)
    assert facet.value == pytest.approx(0.0, abs=0.0)


def test_u20a_total_collapse_is_worse_than_two_dimensional_spread() -> None:
    """U20a's collapse fixture strictly worsens resolution-limit degeneracy."""

    spread = _scene(torch.tensor([[-4.0, -4.0], [-4.0, 4.0], [4.0, -4.0], [4.0, 4.0]]))
    collapsed = _scene(torch.zeros((4, 2), dtype=torch.float64))
    spread_result = U20a(spread)
    collapsed_result = U20a(collapsed)
    assert spread_result.value is not None
    assert collapsed_result.value is not None
    assert spread_result.value == pytest.approx(0.0, abs=0.0)
    assert collapsed_result.value == pytest.approx(1.0, abs=0.0)
    assert collapsed_result.value > spread_result.value


def test_u20a_exempts_incident_route_features() -> None:
    """Do not make every routed graph maximally degenerate at its terminals."""

    positions = torch.tensor([[0.0, 0.0], [8.0, 0.0], [8.0, 8.0], [0.0, 8.0]])
    result = U20a(_scene(positions, ((0, 1), (1, 2), (2, 3), (3, 0))))
    assert result.state is ResultState.VALUE
    assert result.subterms["U20a.iii"] < 1.0
    assert result.value < 1.0


def test_u20a_scores_nonadjacent_segments_of_endpoint_sharing_routes() -> None:
    """U20a exempts only terminal-adjacent segments of routes sharing a node."""

    positions = torch.tensor([[0.0, 0.0], [10.0, 10.0], [10.0, 9.0]], dtype=torch.float64)
    graph = GraphSemantics(("n0", "n1", "n2"), ((0, 1), (0, 2)))
    routes = (
        Route(0, torch.tensor([[0.0, 0.0], [0.0, 10.0], [10.0, 10.0]])),
        Route(1, torch.tensor([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [10.0, 9.0]])),
    )
    ingested = ingest(
        graph,
        DrawingScene(positions, routes),
        StyleContract(),
        ObservationProfile(visible_channels=frozenset({"nodes", "routes"})),
    )
    assert isinstance(ingested, ValidScene)
    # U20a section 3 admits non-adjacent segments even when routes share an endpoint.
    assert U20a(ingested.scene).subterms["U20a.iii"] > 0.0


def test_u20b_midscale_edges_lie_on_low_defect_plateau() -> None:
    """U20b's midscale edge fixture lies between short- and long-edge burdens."""

    positions = torch.tensor([[0.0, 0.0], [7.0, 0.0], [14.0, 0.0]])
    result = U20b(_scene(positions, ((0, 1), (1, 2))))
    assert result.state is ResultState.VALUE
    coordinate = math.log2(result.raw["median_edge_length_u"])
    expected = 1.0 / (1.0 + math.exp(-(math.log2(1.5) - coordinate) / 0.35))
    expected += 1.0 / (1.0 + math.exp(-(coordinate - math.log2(8.0)) / 0.35))
    # U20b contract golden 2: "monotone shoulders, flat plateau" under the
    # section-6 formula with anchors 1.5u and 8u and shoulder width 0.35.
    assert result.value == pytest.approx(expected, abs=1e-15)


def test_u21_compact_symmetric_scene_has_no_sparse_or_overflow_debt() -> None:
    """U21's compact symmetric fixture has exact zero frame-economy subterms."""

    positions = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    result = U21(_scene(positions, ((0, 1), (1, 3), (3, 2), (2, 0))))
    assert result.state is ResultState.VALUE
    assert result.value == pytest.approx(0.0, abs=0.0)


def test_u21_route_mass_is_invariant_to_declared_edge_weights() -> None:
    """U21 gives every escaped route unit mass regardless of weight semantics."""

    positions = torch.stack((torch.arange(30, dtype=torch.float64), torch.zeros(30)), dim=1)
    positions[-1] = torch.tensor([1000.0, 1000.0], dtype=torch.float64)
    edges = tuple((index, index + 1) for index in range(29))

    def weighted_scene(weights: Tuple[float, ...]) -> Scene:
        """Ingest one geometry with selected semantic edge weights.

        Parameters
        ----------
        weights : tuple[float, ...]
            Positive declared flow weights.

        Returns
        -------
        Scene
            Validated routed scene.
        """

        graph = GraphSemantics(
            tuple(f"n{index}" for index in range(30)),
            edges,
            edge_weights=weights,
            weight_semantics="flow",
        )
        routes = tuple(
            Route(index, torch.stack((positions[source], positions[target])))
            for index, (source, target) in enumerate(edges)
        )
        ingested = ingest(
            graph,
            DrawingScene(positions, routes),
            StyleContract(),
            ObservationProfile(visible_channels=frozenset({"nodes", "routes"})),
        )
        assert isinstance(ingested, ValidScene)
        return ingested.scene

    unit = U21(weighted_scene(tuple(1.0 for _ in edges)))
    skewed = U21(weighted_scene(tuple([1.0] * 28 + [100.0])))
    # U21 section 4 freezes route-ribbon mass at one per declared edge.
    assert skewed.raw["mass_out"] == pytest.approx(unit.raw["mass_out"], abs=0.0)


def _semantic_scene() -> Scene:
    """Build the traced-baseline probe's semantic fixture scene.

    Returns
    -------
    Scene
        Validated eight-node layered scene with labels, ranks, clusters,
        chord routes, and a declared flow axis.
    """

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
        Route(index, torch.stack((positions[source], positions[target])))
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


def _complete_weight_table() -> WeightTable:
    """Build the probe's complete explicit weight table.

    Returns
    -------
    WeightTable
        Unit weight on every non-diagnostic scored sub-term.
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


def _probe_profiles() -> ScoringProfiles:
    """Build the probe's frozen scoring profiles with grid row 1 selected.

    Returns
    -------
    ScoringProfiles
        P-mean composition with the shared alpha grid row selected.
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


@pytest.fixture(scope="module")
def semantic_traced() -> Tuple[Scene, TracedSoftScore]:
    """Score the semantic probe scene once through the traced surrogate.

    Returns
    -------
    tuple[Scene, TracedSoftScore]
        The validated scene and its traced differentiable scoring.
    """

    scene = _semantic_scene()
    return scene, score_scene_soft(scene, _complete_weight_table(), _probe_profiles())


def _position_gradient_norm(traced: TracedSoftScore, subterm_id: str) -> float:
    """Measure one traced subterm's position-gradient magnitude.

    Parameters
    ----------
    traced : TracedSoftScore
        Traced scoring result.
    subterm_id : str
        Buffer key of the traced subterm.

    Returns
    -------
    float
        Euclidean norm of the subterm's gradient at the position leaf.
    """

    tensor = traced.traced_subterms[subterm_id]
    assert tensor.requires_grad
    (gradient,) = torch.autograd.grad(
        tensor, traced.positions, retain_graph=True, allow_unused=True
    )
    assert gradient is not None
    return float(torch.linalg.vector_norm(gradient))


def test_traced_clearance_rows_are_live_and_match_the_exact_path(
    semantic_traced: Tuple[Scene, TracedSoftScore],
) -> None:
    """U17.1 and U18's active clearance rows carry live position gradients.

    Parameters
    ----------
    semantic_traced : tuple[Scene, TracedSoftScore]
        Module-scoped traced scoring of the semantic probe scene.
    """

    scene, traced = semantic_traced
    exact = {
        "U17.1": U17(scene, 1).subterms["U17.1"],
        **{key: value for key, value in U18(scene, 1).subterms.items()},
    }
    for subterm_id in ("U17.1", "U18.ll", "U18.ln"):
        assert subterm_id in traced.bound_subterms
        assert _position_gradient_norm(traced, subterm_id) > 0.0
        traced_value = float(traced.traced_subterms[subterm_id].detach())
        # Traced forwards may differ from exact by accumulation order only.
        assert traced_value == pytest.approx(exact[subterm_id], rel=1e-12, abs=1e-15)
    assert traced.soft.l_total.requires_grad


def test_traced_u18_edge_row_is_plateau_flat_on_the_semantic_scene(
    semantic_traced: Tuple[Scene, TracedSoftScore],
) -> None:
    """U18.le is honestly flat here: every label-route pair is beyond budget.

    Every label-route clearance exceeds its edge budget, so the quintic
    smoothstep sits at its upper knot where the contract pins H(x >= 1) = 1
    with H'(1) = 0 (U17.md section 6c via U18.md section 6): the traced
    graph reaches the position leaf and the exact gradient is zero.

    Parameters
    ----------
    semantic_traced : tuple[Scene, TracedSoftScore]
        Module-scoped traced scoring of the semantic probe scene.
    """

    scene, traced = semantic_traced
    assert "U18.le" in traced.bound_subterms
    assert traced.traced_subterms["U18.le"].requires_grad
    assert _position_gradient_norm(traced, "U18.le") == 0.0
    assert float(traced.traced_subterms["U18.le"].detach()) == U18(scene, 1).subterms["U18.le"]


def test_traced_u18_edge_row_is_live_when_a_route_crowds_a_foreign_label() -> None:
    """U18.le carries a live gradient once a route enters a label's budget."""

    positions = torch.tensor([[0.0, 0.0], [10.0, 0.0], [5.0, 0.9]], dtype=torch.float64)
    graph = GraphSemantics(
        ("n0", "n1", "n2"),
        ((0, 1),),
        node_labels=("n0", "n1", "n2"),
    )
    routes = (Route(0, torch.stack((positions[0], positions[1]))),)
    result = ingest(
        graph,
        DrawingScene(positions, routes),
        StyleContract(),
        ObservationProfile(visible_channels=frozenset({"nodes", "routes", "node_labels"})),
    )
    assert isinstance(result, ValidScene)
    scene = result.scene
    exact = U18(scene, 1).subterms["U18.le"]
    leaf = scene.positions.detach().clone().requires_grad_(True)
    with trace_subterms() as buffer:
        U18(build_traced_scene(scene, leaf), 1)
    tensor = buffer["U18.le"]
    assert tensor.requires_grad
    (gradient,) = torch.autograd.grad(tensor, leaf, retain_graph=True)
    assert float(torch.linalg.vector_norm(gradient)) > 0.0
    assert float(tensor.detach()) == pytest.approx(exact, rel=1e-12, abs=1e-15)


def test_traced_frame_facet_rows_publish_as_exact_value_constants(
    semantic_traced: Tuple[Scene, TracedSoftScore],
) -> None:
    """U21's rows stay constant-channel: the frame seam detaches positions.

    U21 reads positions only through ``frames.robust_frame`` and
    ``frames.overflow_defect``, which this seam version carries detached
    (``frames.py`` casts to float inside), so both sub-terms are bound as
    exact-value constants — published honestly, never silently zero-graded.

    Parameters
    ----------
    semantic_traced : tuple[Scene, TracedSoftScore]
        Module-scoped traced scoring of the semantic probe scene.
    """

    scene, traced = semantic_traced
    exact = U21(scene)
    assert exact.state is ResultState.VALUE
    for subterm_id in ("U21.d_sparse_n", "U21.d_overflow"):
        assert subterm_id in traced.constant_subterms
        assert subterm_id not in traced.traced_subterms


def test_traced_diagnostic_legibility_rows_report_their_gradient_reality(
    semantic_traced: Tuple[Scene, TracedSoftScore],
) -> None:
    """Weight-0 legibility rows trace honestly: U20b live, U20a at plateau.

    U20b's median-shoulder logistic responds to positions everywhere off
    the exact plateau (U20b.md section 6). U20a's three sub-terms sit at
    their good-end plateaus on this well-separated fixture, where the
    contract pins the derivative to exactly zero (U20a.md section 6,
    "sit-at-plateau is desired; no cliff").

    Parameters
    ----------
    semantic_traced : tuple[Scene, TracedSoftScore]
        Module-scoped traced scoring of the semantic probe scene.
    """

    scene, traced = semantic_traced
    assert _position_gradient_norm(traced, "U20b.headline") > 0.0
    traced_value = float(traced.traced_subterms["U20b.headline"].detach())
    exact_value = U20b(scene).subterms["U20b.headline"]
    assert traced_value == pytest.approx(exact_value, rel=1e-12, abs=1e-15)
    exact_u20a = U20a(scene)
    for subterm_id in ("U20a.i", "U20a.ii", "U20a.iii"):
        assert _position_gradient_norm(traced, subterm_id) == 0.0
        assert float(traced.traced_subterms[subterm_id].detach()) == pytest.approx(
            exact_u20a.subterms[subterm_id], rel=1e-12, abs=1e-15
        )
