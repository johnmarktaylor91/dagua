"""Tests for V4 loss composition and event-margin verdicts."""

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
from dagua.eval.ruler_v4.events import evaluate_jump_bound, load_event_registry
from dagua.eval.ruler_v4.scene import FacetResult, na_result, value_result
from dagua.eval.ruler_v4.weights import SubtermWeight, WeightTable


def _three_group_weights() -> WeightTable:
    """Return one equal explicit mass in each synthetic group.

    Returns
    -------
    WeightTable
        Fixed three-group table.
    """

    return WeightTable(
        entries=(
            SubtermWeight("S.loss", "S1", "S", 1.0),
            SubtermWeight("X.loss", "X1", "X", 1.0),
            SubtermWeight("L.loss", "L1", "L", 1.0),
        ),
        d_power=0,
    )


def _losses(structure: float, edges: float, legibility: float) -> dict[str, FacetResult]:
    """Build three bounded independent facet results.

    Parameters
    ----------
    structure, edges, legibility : float
        Synthetic group defects in ``[0, 1]``.

    Returns
    -------
    dict[str, FacetResult]
        Mapping accepted by the composition fixture.
    """

    return {
        "S1": value_result(structure, {"S.loss": structure}),
        "X1": value_result(edges, {"X.loss": edges}),
        "L1": value_result(legibility, {"L.loss": legibility}),
    }


def test_na_facets_are_excluded_and_remaining_mass_is_renormalized() -> None:
    """NA contributes neither zero debt nor a free perfect bottleneck group."""

    result = compose(
        {
            "S1": value_result(0.2, {"S.loss": 0.2}),
            "X1": na_result("OBSERVATION_ABSENT"),
            "L1": value_result(0.6, {"L.loss": 0.6}),
        },
        _three_group_weights(),
        CompositionProfile(CompositionFamily.P_MEAN, power=1.0),
    )

    assert result.total_mass == 2.0
    assert result.l_mean == 0.4
    assert tuple(group.group for group in result.groups) == ("L", "S")
    missing = next(row for row in result.subterms if row.subterm_id == "X.loss")
    assert missing.value is None
    assert missing.normalized_weight == 0.0
    assert missing.na_reason == "OBSERVATION_ABSENT"


def test_unavailable_subterm_is_excluded_with_its_published_reason() -> None:
    """Conditional rows inside a VALUE facet redistribute only applicable mass."""

    weights = WeightTable(
        entries=(
            SubtermWeight("A.present", "A", "S", 1.0),
            SubtermWeight("A.absent", "A", "S", 3.0),
        ),
        d_power=0,
    )
    facet = value_result(
        0.25,
        {"A.present": 0.25},
        {"dropped_subterms": ("A.absent:no_population",)},
    )

    result = compose(
        {"A": facet},
        weights,
        CompositionProfile(CompositionFamily.P_MEAN, power=1.0),
    )

    assert result.total_mass == 1.0
    assert result.l_total == 0.25
    missing = next(row for row in result.subterms if row.subterm_id == "A.absent")
    assert missing.na_reason == "no_population"


def test_catastrophic_group_cannot_be_erased_by_perfect_means() -> None:
    """Perfect noncatastrophic groups leave a material bottleneck contribution."""

    profile = CompositionProfile(
        CompositionFamily.MEAN_SOFT_BOTTLENECK,
        bottleneck_mix=0.4,
        bottleneck_temperature=0.05,
        group_allowances={"S": 0.1, "X": 0.1, "L": 0.1},
    )
    catastrophic = compose(_losses(1.0, 0.0, 0.0), _three_group_weights(), profile)
    clean = compose(_losses(0.0, 0.0, 0.0), _three_group_weights(), profile)

    assert catastrophic.l_bottleneck is not None
    assert catastrophic.l_bottleneck > 0.79
    assert catastrophic.l_total > catastrophic.l_mean
    assert catastrophic.l_total - clean.l_total > 0.3


def test_loss_increase_never_improves_either_composition_family() -> None:
    """Coordinatewise larger debt cannot lower p-mean or bottleneck loss."""

    profiles = (
        CompositionProfile(CompositionFamily.P_MEAN, power=3.0),
        CompositionProfile(
            CompositionFamily.MEAN_SOFT_BOTTLENECK,
            bottleneck_mix=0.4,
            bottleneck_temperature=0.05,
            group_allowances={"S": 0.1, "X": 0.1, "L": 0.1},
        ),
    )
    for profile in profiles:
        baseline = compose(_losses(0.2, 0.3, 0.4), _three_group_weights(), profile)
        degraded = compose(_losses(0.2, 0.3, 0.5), _three_group_weights(), profile)
        assert degraded.l_total >= baseline.l_total


def test_strict_win_requires_more_than_sum_of_nearby_manifold_bounds() -> None:
    """Equality is limited and distinct occurrences add to the event budget."""

    base = compose(
        _losses(0.2, 0.2, 0.2),
        _three_group_weights(),
        CompositionProfile(CompositionFamily.P_MEAN, power=1.0),
    )
    registry = load_event_registry()
    event_type = next(event for event in registry.entries if event.event_id == "U41_FACE_SPLIT")
    context = {"F0": 100.0}
    one_bound = evaluate_jump_bound(event_type, context)
    nearby = (
        NearbyEvent("U41_FACE_SPLIT", context, "face-split-1"),
        NearbyEvent("U41_FACE_SPLIT", context, "face-split-2"),
    )

    equal = compare_with_event_margin(
        base,
        replace(base, l_total=base.l_total + 2.0 * one_bound),
        nearby,
        (),
        registry,
    )
    strict = compare_with_event_margin(
        base,
        replace(base, l_total=base.l_total + 2.0 * one_bound + 1e-12),
        nearby,
        (),
        registry,
    )

    assert equal.event_margin == 2.0 * one_bound
    assert equal.verdict is ComparisonVerdict.EVENT_MARGIN_LIMITED
    assert strict.verdict is ComparisonVerdict.FIRST_WINS
    assert strict.nearby_manifold_ids == ("face-split-1", "face-split-2")


def test_same_manifold_near_both_rows_is_counted_once() -> None:
    """The union rule does not double-charge one shared nearby occurrence."""

    base = compose(
        _losses(0.2, 0.2, 0.2),
        _three_group_weights(),
        CompositionProfile(CompositionFamily.P_MEAN, power=1.0),
    )
    registry = load_event_registry()
    nearby = NearbyEvent("U41_FACE_SPLIT", {"F0": 100.0}, "shared-face")
    event_type = next(event for event in registry.entries if event.event_id == nearby.event_id)
    expected = evaluate_jump_bound(event_type, nearby.context)

    comparison = compare_with_event_margin(
        base,
        replace(base, l_total=base.l_total + expected),
        (nearby,),
        (nearby,),
        registry,
    )

    assert comparison.event_margin == expected
    assert comparison.verdict is ComparisonVerdict.EVENT_MARGIN_LIMITED
