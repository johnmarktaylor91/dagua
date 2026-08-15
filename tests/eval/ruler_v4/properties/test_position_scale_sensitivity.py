"""Property family: position-only scale is a visible scene edit."""

from __future__ import annotations

from dagua.eval.ruler_v4.scene import ResultState, Scene
from dagua.eval.ruler_v4.score import ScoringProfiles, score
from dagua.eval.ruler_v4.weights import WeightTable

from .conftest import reingest_scaled


def test_position_only_scale_moves_declared_scale_readers(
    scorable_scene: Scene,
    full_weight_table: WeightTable,
    scoring_profiles: ScoringProfiles,
) -> None:
    """Fixed primitives make a position rescale score-visible without a cliff."""

    expanded = reingest_scaled(scorable_scene, 2.0, position_only=True)
    baseline_score = score(scorable_scene, full_weight_table, scoring_profiles)
    expanded_score = score(expanded, full_weight_table, scoring_profiles)

    changed = []
    for facet_id in ("U17", "U20b", "U21", "U24"):
        baseline = baseline_score.type_m.facets[facet_id].result
        transformed = expanded_score.type_m.facets[facet_id].result
        if (
            baseline.state is ResultState.VALUE
            and transformed.state is ResultState.VALUE
            and baseline.value != transformed.value
        ):
            changed.append(facet_id)
    assert changed
    assert baseline_score.type_m.composition.l_total != expanded_score.type_m.composition.l_total
