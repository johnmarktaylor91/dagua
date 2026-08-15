"""Property family: K12 crossing addition is strictly monotone."""

from __future__ import annotations

from dagua.eval.ruler_v4.composition import CompositionFamily, CompositionProfile, compose
from dagua.eval.ruler_v4.edges import U07
from dagua.eval.ruler_v4.weights import SubtermWeight, WeightTable

from .conftest import build_crossing_scene


def test_adding_one_crossing_strictly_worsens_u07_and_composite() -> None:
    """An endpoint sweep that adds one crossing raises raw count and loss."""

    crossing_free = U07(build_crossing_scene(top_x=0.0, top_y=-0.001))
    crossed = U07(build_crossing_scene(top_x=0.0, top_y=0.001))
    weights = WeightTable(
        entries=(
            SubtermWeight("U7.base", "U07", "X", 1.0),
            SubtermWeight("U7.tail", "U07", "X", 1.0),
        ),
        d_power=0,
    )
    profile = CompositionProfile(CompositionFamily.P_MEAN, power=2.0)
    baseline = compose({"U07": crossing_free}, weights, profile)
    degraded = compose({"U07": crossed}, weights, profile)

    assert crossed.raw["crossing_count"] == crossing_free.raw["crossing_count"] + 1
    assert crossing_free.value is not None
    assert crossed.value is not None
    assert crossed.value > crossing_free.value
    assert degraded.l_total > baseline.l_total
