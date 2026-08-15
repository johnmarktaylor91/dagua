"""Property family: exact whole-scene scale invariance."""

from __future__ import annotations

from dagua.eval.ruler_v4.scene import Scene
from dagua.eval.ruler_v4.score import OutputType, ScoringProfiles, score
from dagua.eval.ruler_v4.weights import WeightTable

from .conftest import reingest_scaled


def test_scene_scale_reexpression_is_exact(
    scorable_scene: Scene,
    full_weight_table: WeightTable,
    scoring_profiles: ScoringProfiles,
) -> None:
    """Scaling coordinates and dimensional style preserves every facet value."""

    scaled = reingest_scaled(scorable_scene, 2.0, position_only=False)
    baseline_score = score(scorable_scene, full_weight_table, scoring_profiles)
    scaled_score = score(scaled, full_weight_table, scoring_profiles)

    assert baseline_score.type_r.output_type is OutputType.TYPE_R
    assert baseline_score.type_m.output_type is OutputType.TYPE_M
    assert baseline_score.type_m.composition.l_total == scaled_score.type_m.composition.l_total
    assert baseline_score.type_m.headline.value == scaled_score.type_m.headline.value
    for facet_id, baseline in baseline_score.type_m.facets.items():
        transformed = scaled_score.type_m.facets[facet_id]
        assert baseline.result.state is transformed.result.state
        assert baseline.result.value == transformed.result.value
        assert baseline.result.subterms == transformed.result.subterms
