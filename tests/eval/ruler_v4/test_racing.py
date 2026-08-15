"""Tests for paired certificates and certified-interval racing."""

from __future__ import annotations

from typing import Dict, Tuple

import pytest

from dagua.eval.ruler_v4.certification import CertifiedInterval
from dagua.eval.ruler_v4.composition import CompositionFamily, CompositionProfile, compose
from dagua.eval.ruler_v4.racing import (
    EliminationRule,
    InconsistentCertificateError,
    PairedDifferenceCertificate,
    PairedTermRegion,
    RaceCandidate,
    certify_paired_difference,
    race_candidates,
)
from dagua.eval.ruler_v4.scene import FacetResult, value_result
from dagua.eval.ruler_v4.weight_table import SubtermWeight, WeightTable


def _two_term_inputs() -> Tuple[Dict[str, FacetResult], WeightTable, CompositionProfile]:
    """Build the two-term nonlinear paired-certification fixture.

    Returns
    -------
    tuple[dict[str, FacetResult], WeightTable, CompositionProfile]
        Exact facet state, equal-mass weights, and a p=2 p-mean profile.
    """

    facets = {
        "U01": value_result(0.5, {"U01.headline": 0.5}),
        "U03": value_result(0.0, {"U03.r_1": 0.0}),
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


def test_shared_level_uncertainty_is_charged_under_nonlinear_composition() -> None:
    """A zero paired difference does NOT cancel under a p>1 composition.

    P3REVIEW OPUS5 BLOCKER-3 red fixture: U01 is a shared level on the full
    feasible range with a CRN difference certified exactly zero; U03 carries
    a certified difference of +0.5 at level zero. The exact paired functional
    L(A) - L(B) then varies with the shared level (from ~0.354 at c=0 down to
    ~0.083 at c=1), so a degenerate certified interval is unsound. The fixed
    bound must cover the whole true range.
    """

    facets, table, profile = _two_term_inputs()
    certificate = certify_paired_difference(
        facets,
        table,
        profile,
        {
            "U03.r_1": PairedTermRegion(
                CertifiedInterval(0.0, 0.0, "round-1"),
                CertifiedInterval(0.5, 0.5, "round-1"),
            )
        },
        confidence_id="round-1",
    )

    assert certificate.shared_unobserved_subterms == ("U01.headline",)
    # The shared-level row is charged 2 * Lambda_f * level_radius, never zero.
    assert certificate.error_bound > 0.0
    for shared_level in (0.0, 0.25, 0.5, 0.75, 1.0):
        facets_a = {
            "U01": value_result(0.5, {"U01.headline": shared_level}),
            "U03": value_result(0.0, {"U03.r_1": 0.5}),
        }
        facets_b = {
            "U01": value_result(0.5, {"U01.headline": shared_level}),
            "U03": value_result(0.0, {"U03.r_1": 0.0}),
        }
        true_difference = (
            compose(facets_a, table, profile).l_total - compose(facets_b, table, profile).l_total
        )
        assert certificate.interval.lo <= true_difference <= certificate.interval.hi


def test_degenerate_regions_still_certify_a_degenerate_interval() -> None:
    """Fully observed zero-radius regions keep an exact zero error bound."""

    facets, table, profile = _two_term_inputs()
    certificate = certify_paired_difference(
        facets,
        table,
        profile,
        {
            "U01.headline": PairedTermRegion(
                CertifiedInterval(0.5, 0.5, "round-1"),
                CertifiedInterval(0.0, 0.0, "round-1"),
            ),
            "U03.r_1": PairedTermRegion(
                CertifiedInterval(0.0, 0.0, "round-1"),
                CertifiedInterval(0.5, 0.5, "round-1"),
            ),
        },
        confidence_id="round-1",
    )

    assert certificate.error_bound == 0.0
    assert certificate.shared_unobserved_subterms == ()
    assert certificate.interval.lo == certificate.interval.hi


def test_separated_intervals_choose_without_true_score() -> None:
    """Marginal separation produces no expensive escalation call."""

    calls = []
    candidates = (
        RaceCandidate("best", CertifiedInterval(0.1, 0.2)),
        RaceCandidate("worse", CertifiedInterval(0.5, 0.6)),
    )
    result = race_candidates(
        candidates,
        lambda candidate_id: calls.append(candidate_id) or 0.0,
        incumbent_id="best",
        policy_margin=0.05,
        full_score_max_candidates=0,
        true_score_budget=2,
    )

    assert result.winner_id == "best"
    assert calls == []
    assert result.eliminations[0].rule is EliminationRule.MARGINAL_BOUND


def test_overlapping_intervals_escalate_only_survivors() -> None:
    """Overlapping candidates are true-scored after a distant row is pruned."""

    calls = []
    losses = {"a": 0.3, "b": 0.2, "far": 0.9}
    candidates = (
        RaceCandidate("a", CertifiedInterval(0.1, 0.4)),
        RaceCandidate("b", CertifiedInterval(0.2, 0.5)),
        RaceCandidate("far", CertifiedInterval(0.8, 1.0)),
    )
    result = race_candidates(
        candidates,
        lambda candidate_id: calls.append(candidate_id) or losses[candidate_id],
        incumbent_id="a",
        policy_margin=0.0,
        full_score_max_candidates=0,
        true_score_budget=2,
    )

    assert result.winner_id == "b"
    assert calls == ["a", "b"]
    assert result.escalated_ids == ("a", "b")


def test_paired_primary_eliminates_when_marginals_overlap() -> None:
    """A paired difference can act when two marginal boxes cannot."""

    paired = PairedDifferenceCertificate(
        interval=CertifiedInterval(0.2, 0.4, "round-1"),
        center_difference=0.3,
        error_bound=0.1,
        sensitivity_bounds={},
        shared_unobserved_subterms=(),
    )
    candidates = (
        RaceCandidate("a", CertifiedInterval(0.1, 0.8, "round-1"), {"b": paired}),
        RaceCandidate("b", CertifiedInterval(0.1, 0.8, "round-1")),
    )
    result = race_candidates(
        candidates,
        lambda candidate_id: 0.0,
        incumbent_id="b",
        policy_margin=0.05,
        full_score_max_candidates=0,
        true_score_budget=2,
    )

    assert result.winner_id == "b"
    assert result.escalated_ids == ()
    assert result.eliminations[0].rule is EliminationRule.PAIRED_DIFFERENCE


def test_budget_exhaustion_retains_incumbent() -> None:
    """An unresolved overlap fails closed before partial true scoring."""

    candidates = (
        RaceCandidate("incumbent", CertifiedInterval(0.1, 0.5)),
        RaceCandidate("challenger", CertifiedInterval(0.2, 0.6)),
    )
    result = race_candidates(
        candidates,
        lambda candidate_id: 0.0,
        incumbent_id="incumbent",
        policy_margin=0.0,
        full_score_max_candidates=0,
        true_score_budget=1,
    )

    assert result.winner_id == "incumbent"
    assert result.budget_exhausted
    assert result.escalated_ids == ()


def test_mixed_confidence_allocations_are_refused() -> None:
    """Intervals from different allocations cannot race together."""

    candidates = (
        RaceCandidate("a", CertifiedInterval(0.1, 0.2, "anytime-95")),
        RaceCandidate("b", CertifiedInterval(0.5, 0.6, "deterministic")),
    )
    with pytest.raises(ValueError, match="one confidence allocation"):
        race_candidates(
            candidates,
            lambda candidate_id: 0.0,
            incumbent_id="a",
            policy_margin=0.05,
            full_score_max_candidates=0,
            true_score_budget=2,
        )


def _paired_certificate(lo: float, hi: float) -> PairedDifferenceCertificate:
    """Build one minimal paired certificate for race-topology tests.

    Parameters
    ----------
    lo, hi : float
        Certified paired-difference interval endpoints.

    Returns
    -------
    PairedDifferenceCertificate
        Certificate under the shared test allocation.
    """

    return PairedDifferenceCertificate(
        interval=CertifiedInterval(lo, hi, "round-1"),
        center_difference=0.5 * (lo + hi),
        error_bound=0.5 * (hi - lo),
        sensitivity_bounds={},
        shared_unobserved_subterms=(),
    )


def test_cyclic_certificates_fail_closed_with_typed_error() -> None:
    """An emptied survivor set raises the loud typed version failure."""

    cycle = _paired_certificate(0.2, 0.4)
    candidates = (
        RaceCandidate("a", CertifiedInterval(0.0, 1.0, "round-1"), {"b": cycle}),
        RaceCandidate("b", CertifiedInterval(0.0, 1.0, "round-1"), {"c": cycle}),
        RaceCandidate("c", CertifiedInterval(0.0, 1.0, "round-1"), {"a": cycle}),
    )
    with pytest.raises(InconsistentCertificateError, match="excluded every candidate"):
        race_candidates(
            candidates,
            lambda candidate_id: 0.0,
            incumbent_id="a",
            policy_margin=0.05,
            full_score_max_candidates=0,
            true_score_budget=3,
        )


def test_budget_exhaustion_never_resurrects_an_eliminated_incumbent() -> None:
    """An eliminated incumbent yields a typed inconclusive result, not a win."""

    candidates = (
        RaceCandidate("incumbent", CertifiedInterval(0.9, 1.0)),
        RaceCandidate("a", CertifiedInterval(0.0, 0.1)),
        RaceCandidate("b", CertifiedInterval(0.05, 0.15)),
    )
    result = race_candidates(
        candidates,
        lambda candidate_id: 0.0,
        incumbent_id="incumbent",
        policy_margin=0.0,
        full_score_max_candidates=0,
        true_score_budget=1,
    )

    assert result.winner_id is None
    assert result.budget_exhausted
    assert result.reason == "budget_exhausted_no_selection"
    assert any(record.candidate_id == "incumbent" for record in result.eliminations)


def test_small_contest_full_scores_without_surrogate_pruning() -> None:
    """The explicit small-contest policy bypasses even separated intervals."""

    calls = []
    candidates = (
        RaceCandidate("a", CertifiedInterval(0.0, 0.1)),
        RaceCandidate("b", CertifiedInterval(0.9, 1.0)),
    )
    result = race_candidates(
        candidates,
        lambda candidate_id: calls.append(candidate_id) or {"a": 0.3, "b": 0.2}[candidate_id],
        incumbent_id="a",
        policy_margin=0.0,
        full_score_max_candidates=2,
        true_score_budget=2,
    )

    assert result.winner_id == "b"
    assert calls == ["a", "b"]
    assert result.eliminations == ()
