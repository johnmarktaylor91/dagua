"""Property family: inapplicable-NA exclusion is exact and order-preserving.

The P2 reviews found the original family vacuous: at ``p = 1`` with one
surviving row, ``l_total`` was identically the fixture input, so every
assertion reduced to arithmetic on the fixture itself. The family now runs
at ``p = 2`` over three rows with unequal masses, pins the renormalized
composition against an independently hand-computed closed form, and pins
the CC-13 seam separating unobserved absence from inapplicable absence.
"""

from __future__ import annotations

import math

import pytest

from dagua.eval.ruler_v4.composition import CompositionFamily, CompositionProfile, compose
from dagua.eval.ruler_v4.scene import na_result, value_result
from dagua.eval.ruler_v4.weights import SubtermWeight, WeightTable


def _observation_weights() -> WeightTable:
    """Return three unequal explicit observation masses.

    Returns
    -------
    WeightTable
        Synthetic three-observation table.
    """

    return WeightTable(
        entries=(
            SubtermWeight("A.loss", "A", "S", 2.0),
            SubtermWeight("B.loss", "B", "X", 1.0),
            SubtermWeight("C.loss", "C", "L", 1.0),
        ),
        d_power=0,
    )


def _results(a: float, b: float, c: float | None):
    """Build three observation results, dropping C when ``None``.

    Parameters
    ----------
    a, b : float
        Observed defects for the surviving rows.
    c : float or None
        Observed defect for C, or ``None`` for an inapplicable absence.

    Returns
    -------
    dict[str, FacetResult]
        Mapping accepted by composition.
    """

    return {
        "A": value_result(a, {"A.loss": a}),
        "B": value_result(b, {"B.loss": b}),
        "C": value_result(c, {"C.loss": c}) if c is not None else na_result("INPUT_INAPPLICABLE"),
    }


def test_inapplicable_exclusion_matches_the_closed_form_renormalization() -> None:
    """The post-NA composition equals the hand-computed renormalized p-mean."""

    profile = CompositionProfile(CompositionFamily.P_MEAN, power=2.0)
    composed = compose(_results(0.2, 0.6, None), _observation_weights(), profile)

    # Independent closed form: masses 2 and 1 renormalize over 3.
    expected = math.sqrt((2.0 / 3.0) * 0.2**2 + (1.0 / 3.0) * 0.6**2)
    assert composed.l_total == pytest.approx(expected, abs=1e-15)
    assert composed.total_mass == 3.0
    # The composed value is NOT the fixture arithmetic of either input row.
    assert composed.l_total != pytest.approx(0.2)
    assert composed.l_total != pytest.approx(0.6)


def test_na_exclusion_preserves_remaining_observation_ordering() -> None:
    """Raising surviving defects strictly worsens the renormalized loss."""

    profile = CompositionProfile(CompositionFamily.P_MEAN, power=2.0)
    first_full = compose(_results(0.2, 0.6, 0.3), _observation_weights(), profile)
    second_full = compose(_results(0.4, 0.8, 0.3), _observation_weights(), profile)
    first_missing = compose(_results(0.2, 0.6, None), _observation_weights(), profile)
    second_missing = compose(_results(0.4, 0.8, None), _observation_weights(), profile)

    assert first_full.l_total < second_full.l_total
    assert first_missing.l_total < second_missing.l_total

    available = next(row for row in first_missing.subterms if row.subterm_id == "A.loss")
    missing = next(row for row in first_missing.subterms if row.subterm_id == "C.loss")
    assert available.normalized_weight == pytest.approx(2.0 / 3.0)
    assert missing.value is None
    assert missing.normalized_weight == 0.0
    assert missing.na_reason == "INPUT_INAPPLICABLE"


def test_unobserved_absence_is_not_the_inapplicable_branch() -> None:
    """CC-13: the renormalizing branch is closed to unobserved mass."""

    profile = CompositionProfile(CompositionFamily.P_MEAN, power=2.0)
    unobserved = {
        "A": value_result(0.2, {"A.loss": 0.2}),
        "B": value_result(0.6, {"B.loss": 0.6}),
        "C": na_result("UNOBSERVED:TIER_BUDGET_ZERO"),
    }
    with pytest.raises(ValueError, match="CC-13"):
        compose(unobserved, _observation_weights(), profile)
