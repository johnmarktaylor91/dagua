"""Property family: NA ingestion is monotone over available observations."""

from __future__ import annotations

from dagua.eval.ruler_v4.composition import CompositionFamily, CompositionProfile, compose
from dagua.eval.ruler_v4.scene import na_result, value_result
from dagua.eval.ruler_v4.weights import SubtermWeight, WeightTable


def _observation_weights() -> WeightTable:
    """Return two equal explicit observation masses.

    Returns
    -------
    WeightTable
        Synthetic two-observation table.
    """

    return WeightTable(
        entries=(
            SubtermWeight("A.loss", "A", "S", 1.0),
            SubtermWeight("B.loss", "B", "X", 1.0),
        ),
        d_power=0,
    )


def test_removing_observation_excludes_na_and_preserves_available_sign() -> None:
    """NA renormalization cannot reverse the remaining observation's ordering."""

    profile = CompositionProfile(CompositionFamily.P_MEAN, power=1.0)
    first_full = compose(
        {"A": value_result(0.2, {"A.loss": 0.2}), "B": value_result(0.6, {"B.loss": 0.6})},
        _observation_weights(),
        profile,
    )
    second_full = compose(
        {"A": value_result(0.4, {"A.loss": 0.4}), "B": value_result(0.8, {"B.loss": 0.8})},
        _observation_weights(),
        profile,
    )
    first_missing = compose(
        {"A": value_result(0.2, {"A.loss": 0.2}), "B": na_result("OBSERVATION_REMOVED")},
        _observation_weights(),
        profile,
    )
    second_missing = compose(
        {"A": value_result(0.4, {"A.loss": 0.4}), "B": na_result("OBSERVATION_REMOVED")},
        _observation_weights(),
        profile,
    )

    assert first_full.l_total < second_full.l_total
    assert first_missing.l_total < second_missing.l_total
    assert first_missing.total_mass == 1.0
    available = next(row for row in first_missing.subterms if row.subterm_id == "A.loss")
    missing = next(row for row in first_missing.subterms if row.subterm_id == "B.loss")
    assert available.normalized_weight == 1.0
    assert missing.value is None
    assert missing.normalized_weight == 0.0
    assert missing.na_reason == "OBSERVATION_REMOVED"
