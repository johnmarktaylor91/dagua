"""Tests for the pure top-level V4 scorer."""

from __future__ import annotations

from dataclasses import replace

from dagua.eval.ruler_v4.composition import CompositionFamily, CompositionProfile
from dagua.eval.ruler_v4.contracts import CONTRACTS
from dagua.eval.ruler_v4.headline import HeadlineProfile
from dagua.eval.ruler_v4.scene import ResultState, Scene
from dagua.eval.ruler_v4.score import OutputType, ScoringProfiles, score, score_scene
from dagua.eval.ruler_v4.weight_table import (
    GATE_DIAGNOSTIC_FACETS,
    REQUIRED_PRIOR_FLOOR_FACETS,
    ParameterProvenance,
    SubtermWeight,
    WeightTable,
)


def _complete_table() -> WeightTable:
    """Build a complete explicit fixed-weight scorer table.

    Returns
    -------
    WeightTable
        Valid 91-row contract table.
    """

    entries = tuple(
        SubtermWeight(
            subterm_id=subterm_id,
            facet_id=facet_id,
            group=facet_id[0:3],
            weight=0.0 if facet_id in GATE_DIAGNOSTIC_FACETS else 1.0,
            prior_driven=facet_id in REQUIRED_PRIOR_FLOOR_FACETS,
            diagnostic=facet_id in GATE_DIAGNOSTIC_FACETS,
        )
        for facet_id, contract in CONTRACTS.items()
        for subterm_id in contract.scored_subterms
    )
    return WeightTable(
        entries=entries,
        d_power=20,
        prior_floors={facet_id: 1.0 for facet_id in REQUIRED_PRIOR_FLOOR_FACETS},
    )


def _profiles() -> ScoringProfiles:
    """Build explicit deterministic profiles for scorer tests.

    Returns
    -------
    ScoringProfiles
        Fixed family, grid, headline map, and versions.
    """

    return ScoringProfiles(
        composition=CompositionProfile(CompositionFamily.P_MEAN, power=2.0),
        headline=HeadlineProfile(index_span=100.0, loss_scale=1.0, version="headline-test"),
        measurement_version="measurement-test",
        policy_version="policy-test",
        alpha_grid_index=1,
        parameter_provenance={
            "composition.power": ParameterProvenance("preregistered_prior"),
            "headline.index_span": ParameterProvenance("contract_frozen"),
            "headline.loss_scale": ParameterProvenance("preregistered_prior"),
            "alpha_grid_index": ParameterProvenance("preregistered_prior"),
        },
    )


def _scorable_scene(scene: Scene) -> Scene:
    """Add a complete declared-tree block to the broad semantic fixture.

    Parameters
    ----------
    scene : Scene
        Directed semantic fixture.

    Returns
    -------
    Scene
        Immutable copy for which every applicable facet is valid or NA.
    """

    graph = replace(
        scene.graph,
        tree_parents=(None, 0, 0, 1, 2, 3, 4, 5),
        tree_depths=(0, 1, 1, 2, 2, 3, 3, 4),
        tree_layout="layered",
        flow_axis=(1.0, 0.0),
        ordered_children={0: (1, 2)},
    )
    return replace(scene, graph=graph)


def test_score_is_deterministic_and_publishes_all_contracts(semantic_scene: Scene) -> None:
    """One scene produces stable TYPE-R/TYPE-M payloads and all 45 facets."""

    scene = _scorable_scene(semantic_scene)
    first = score(scene, _complete_table(), _profiles())
    second = score_scene(scene, _complete_table(), _profiles())

    assert first == second
    assert first.type_r.output_type is OutputType.TYPE_R
    assert first.type_m.output_type is OutputType.TYPE_M
    assert set(first.type_m.facets) == set(CONTRACTS)
    assert first.measurement_version == "measurement-test"
    assert first.policy_version == "policy-test"
    assert first.type_m.headline.name == "ordinal_within_graph_index"


def test_score_carries_diagnostics_without_headline_mass(semantic_scene: Scene) -> None:
    """Every gate-report DIAG row is published with zero normalized mass."""

    result = score(_scorable_scene(semantic_scene), _complete_table(), _profiles())

    for facet_id in GATE_DIAGNOSTIC_FACETS:
        rows = result.type_m.facets[facet_id].contributions
        assert rows
        assert all(row.diagnostic and row.normalized_weight == 0.0 for row in rows)


def test_score_handles_conditionally_dropped_subterms(semantic_scene: Scene) -> None:
    """Alternative-layout and empty-population rows remain structured absences."""

    result = score(_scorable_scene(semantic_scene), _complete_table(), _profiles())
    dropped = [
        row
        for breakdown in result.type_m.facets.values()
        for row in breakdown.contributions
        if row.value is None and breakdown.result.state is ResultState.VALUE
    ]

    assert dropped
    assert all(row.normalized_weight == 0.0 and row.na_reason for row in dropped)


def test_score_routes_a_temporal_scene_to_u40(semantic_scene: Scene) -> None:
    """The pure entrypoint can publish all 45 rows, including U40 (P2 minor)."""

    import torch

    from dagua.eval.ruler_v4.ingestion import ingest, ingest_temporal
    from dagua.eval.ruler_v4.scene import (
        DrawingScene,
        GraphSemantics,
        ObservationProfile,
        Route,
        StyleContract,
        TemporalTransition,
        ValidScene,
        ValidTemporalScene,
    )

    def frame() -> Scene:
        positions = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]], dtype=torch.float64)
        edges = ((0, 1), (0, 2))
        routes = tuple(
            Route(index, torch.stack((positions[source], positions[target])))
            for index, (source, target) in enumerate(edges)
        )
        ingested = ingest(
            GraphSemantics(("a", "b", "c"), edges, temporal_ids=("a", "b", "c")),
            DrawingScene(positions, routes),
            StyleContract(),
            ObservationProfile(visible_channels=frozenset({"nodes", "routes"})),
        )
        assert isinstance(ingested, ValidScene)
        return ingested.scene

    transition = TemporalTransition(
        {"a": "unchanged", "b": "unchanged", "c": "unchanged"},
        {"a": 0.0, "b": 0.0, "c": 0.0},
    )
    temporal = ingest_temporal((frame(), frame()), (transition,))
    assert isinstance(temporal, ValidTemporalScene)

    scene = _scorable_scene(semantic_scene)
    without = score(scene, _complete_table(), _profiles())
    assert without.type_m.facets["U40"].result.reason == "TEMPORAL_PROFILE_ABSENT"

    routed = score(scene, _complete_table(), _profiles(), temporal_scene=temporal.scene)
    u40 = routed.type_m.facets["U40"].result
    assert u40.state is ResultState.VALUE
    assert u40.temporal_headline == 0.0
    # U40 is DIAG at weight 0: the temporal route cannot move the headline.
    assert routed.type_m.headline.value == without.type_m.headline.value
