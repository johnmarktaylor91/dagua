"""Property family: CC-1 continuity and declared event budgets."""

from __future__ import annotations

from dataclasses import replace

from dagua.eval.ruler_v4.composition import (
    ComparisonVerdict,
    CompositionFamily,
    CompositionProfile,
    NearbyEvent,
    compare_with_event_margin,
    compose,
)
from dagua.eval.ruler_v4.edges import U07
from dagua.eval.ruler_v4.events import evaluate_jump_bound, load_event_registry
from dagua.eval.ruler_v4.weights import SubtermWeight, WeightTable

from .conftest import build_crossing_scene


def _crossing_weights() -> WeightTable:
    """Return explicit U07-only weights for event properties.

    Returns
    -------
    WeightTable
        Two-row fixed table.
    """

    return WeightTable(
        entries=(
            SubtermWeight("U7.base", "U07", "X", 1.0),
            SubtermWeight("U7.tail", "U07", "X", 1.0),
        ),
        d_power=0,
    )


def test_cc1_off_manifold_perturbation_is_lipschitz_bounded() -> None:
    """A small crossing-angle perturbation has no undeclared score jump."""

    first = U07(build_crossing_scene(top_x=0.10, top_y=1.0))
    second = U07(build_crossing_scene(top_x=0.100001, top_y=1.0))

    assert first.value is not None
    assert second.value is not None
    assert abs(first.value - second.value) < 1e-5


def test_cc1_crossing_event_respects_bound_and_limits_verdict() -> None:
    """Crossing the declared manifold closes its bound and blocks a strict win."""

    before = U07(build_crossing_scene(top_x=0.0, top_y=-0.001))
    after = U07(build_crossing_scene(top_x=0.0, top_y=0.001))
    registry = load_event_registry()
    event = next(item for item in registry.entries if item.event_id == "U7.crossing-parity")
    context = {
        "lambda_T": 0.0,
        "gamma": 0.0,
        "r_r": 0.0,
        "r_d": 0.0,
        "Dtilde": 0.0,
        "D0": 1.0,
        "Z_prime": 1.0,
        "x0": 1.0,
    }
    assert before.value is not None
    assert after.value is not None
    assert after.value - before.value <= evaluate_jump_bound(event, context)

    profile = CompositionProfile(CompositionFamily.P_MEAN, power=1.0)
    first_composition = compose({"U07": before}, _crossing_weights(), profile)
    second_composition = compose({"U07": after}, _crossing_weights(), profile)
    comparison = compare_with_event_margin(
        first_composition,
        second_composition,
        (NearbyEvent(event.event_id, context),),
        (),
        registry,
    )

    assert comparison.verdict is ComparisonVerdict.EVENT_MARGIN_LIMITED
    assert comparison.event_margin == evaluate_jump_bound(event, context)


def test_cc1_strict_win_requires_margin_greater_than_bound() -> None:
    """Equality is limited, while a larger margin produces a strict outcome."""

    base = compose(
        {"U07": U07(build_crossing_scene(top_x=0.0, top_y=-0.001))},
        _crossing_weights(),
        CompositionProfile(CompositionFamily.P_MEAN, power=1.0),
    )
    registry = load_event_registry()
    event = NearbyEvent("U41_FACE_SPLIT", {"F0": 100.0})
    event_bound = evaluate_jump_bound(
        next(item for item in registry.entries if item.event_id == event.event_id),
        event.context,
    )
    equal_margin = compare_with_event_margin(
        base,
        replace(base, l_total=base.l_total + event_bound),
        (event,),
        (),
        registry,
    )
    strict_margin = compare_with_event_margin(
        base,
        replace(base, l_total=base.l_total + 2.0 * event_bound),
        (event,),
        (),
        registry,
    )

    assert equal_margin.verdict is ComparisonVerdict.EVENT_MARGIN_LIMITED
    assert strict_margin.verdict is ComparisonVerdict.MARGIN_RULE_FIRST_WINS
