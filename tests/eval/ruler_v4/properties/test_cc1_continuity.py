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
from dagua.eval.ruler_v4.weight_table import SubtermWeight, WeightTable

from .conftest import build_crossing_field_scene, build_crossing_scene


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
    """The measured crossing jump respects its declared sub-clamp bound.

    The scene carries seven spectator edges so the U07 opportunity
    normalizer (Z' = eligible + 1 = 37, x0 = 0.25) puts the registry's
    closed-form bound strictly below its 1.0 clamp; the bound-respect and
    verdict assertions are falsifiable in this regime (the P2 reviews found
    the old zeroed context saturated the bound at exactly 1.0, making both
    assertions vacuous on a [0, 1]-valued facet).
    """

    before = U07(build_crossing_field_scene(top_y=-0.001))
    after = U07(build_crossing_field_scene(top_y=0.001))
    registry = load_event_registry()
    event = next(item for item in registry.entries if item.event_id == "U7.crossing-parity")
    # Honest graph-local inputs: facet defaults gamma=1.0/lambda_T=0.5, the
    # registry's own sibling-repeat coefficient r_r=1, no existing events at
    # the birth point (Dtilde=0), and the scene's opportunity normalizer.
    assert before.raw["eligible_pairs"] == 36
    context = {
        "lambda_T": 0.5,
        "gamma": 1.0,
        "r_r": 1.0,
        "r_d": 1.0,
        "Dtilde": 0.0,
        "D0": 1.0,
        "Z_prime": 37.0,
        "x0": 0.25,
    }
    bound = evaluate_jump_bound(event, context)
    assert 0.0 < bound < 1.0
    assert before.value is not None
    assert after.value is not None
    assert int(before.raw["crossing_count"]) == 0
    assert int(after.raw["crossing_count"]) == 1
    jump = after.value - before.value
    assert jump > 0.0
    assert jump <= bound

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

    assert comparison.decision_margin > 0.0
    assert comparison.verdict is ComparisonVerdict.EVENT_MARGIN_LIMITED
    assert comparison.event_margin == bound


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
