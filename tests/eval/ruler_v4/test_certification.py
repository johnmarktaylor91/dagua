"""Tests for V4 interval, rank, and gradient certification."""

from __future__ import annotations

from typing import Dict, Tuple

import pytest

from dagua.eval.ruler_v4.certification import (
    CertifiedInterval,
    certify_gradient_sanity,
    certify_rank_fidelity,
    compose_certified_intervals,
    exact_term_intervals,
)
from dagua.eval.ruler_v4.composition import CompositionFamily, CompositionProfile, compose
from dagua.eval.ruler_v4.scene import FacetResult, value_result
from dagua.eval.ruler_v4.weight_table import SubtermWeight, WeightTable


def _inputs() -> Tuple[Dict[str, FacetResult], WeightTable, CompositionProfile]:
    """Build a two-row deterministic interval fixture.

    Returns
    -------
    tuple[dict[str, FacetResult], WeightTable, CompositionProfile]
        Exact facets, weights, and p-mean profile.
    """

    facets = {
        "U01": value_result(0.2, {"U01.headline": 0.2}),
        "U03": value_result(0.8, {"U03.r_1": 0.8}),
    }
    table = WeightTable(
        entries=(
            SubtermWeight("U01.headline", "U01", "G1", 1.0),
            SubtermWeight("U03.r_1", "U03", "G2", 1.0),
        ),
        d_power=20,
    )
    profile = CompositionProfile(CompositionFamily.P_MEAN, power=2.0)
    return facets, table, profile


def test_exact_intervals_reproduce_exact_composition() -> None:
    """Degenerate term intervals propagate to a degenerate exact loss."""

    facets, table, profile = _inputs()
    term_intervals = exact_term_intervals(facets, table, profile)
    certified = compose_certified_intervals(
        facets,
        table,
        profile,
        term_intervals,
        confidence_id="deterministic",
    )
    exact = compose(facets, table, profile)

    assert certified.interval.lo == pytest.approx(exact.l_total, abs=1e-15)
    assert certified.interval.hi == pytest.approx(exact.l_total, abs=1e-15)
    assert certified.unobserved_subterms == ()


def test_unobserved_term_uses_full_feasible_interval() -> None:
    """CC-13 omission widens rather than renormalizes the composite."""

    facets, table, profile = _inputs()
    certified = compose_certified_intervals(
        facets,
        table,
        profile,
        {"U01.headline": CertifiedInterval(0.1, 0.3, "round-1")},
        confidence_id="round-1",
    )

    assert certified.term_intervals["U03.r_1"] == CertifiedInterval(0.0, 1.0, "round-1")
    assert certified.unobserved_subterms == ("U03.r_1",)
    assert certified.interval.lo < certified.interval.hi


def test_rank_fidelity_reports_tau_and_explicit_threshold() -> None:
    """A deterministic smoke ranking clears the frozen 0.85 contract."""

    result = certify_rank_fidelity(
        [0.1, 0.2, 0.4, 0.8],
        [0.11, 0.19, 0.41, 0.79],
        threshold=0.85,
    )

    assert result.tau == 1.0
    assert result.certified
    assert result.concordant_pairs == 6


def test_rank_fidelity_with_zero_comparable_pairs_fails_closed() -> None:
    """A batch of joint ties never certifies (P3REVIEW OPUS5 MAJOR-2)."""

    result = certify_rank_fidelity(
        [0.3, 0.3, 0.3],
        [0.7, 0.7, 0.7],
        threshold=0.85,
    )

    assert result.comparable_pairs == 0
    assert not result.certified
    assert result.tau == 0.0


def test_rank_fidelity_publishes_one_sided_tie_counts() -> None:
    """Pairs tied on exactly one side are published, not silently dropped."""

    result = certify_rank_fidelity(
        [0.1, 0.1, 0.4],
        [0.2, 0.3, 0.9],
        threshold=0.85,
    )

    assert result.true_tie_pairs == 1
    assert result.surrogate_tie_pairs == 0
    assert result.comparable_pairs == 2


@pytest.mark.parametrize("value", [0.2, 0.5, 0.8])
def test_gradient_sanity_agrees_and_improves_exact(value: float) -> None:
    """Three analytic probes agree with finite differences and improve exact."""

    result = certify_gradient_sanity(
        lambda coordinate: coordinate * coordinate,
        lambda coordinate: coordinate.square(),
        value,
        step_size=0.05,
        finite_difference_epsilon=1.0e-6,
        relative_tolerance=1.0e-8,
        absolute_tolerance=1.0e-10,
    )

    assert result.agreement
    assert result.exact_improved
    assert result.moved_value < value
