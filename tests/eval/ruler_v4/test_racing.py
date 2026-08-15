"""Tests for paired certificates and certified-interval racing."""

from __future__ import annotations

from typing import Dict, Tuple

from dagua.eval.ruler_v4.certification import CertifiedInterval
from dagua.eval.ruler_v4.composition import CompositionFamily, CompositionProfile
from dagua.eval.ruler_v4.racing import (
    EliminationRule,
    PairedDifferenceCertificate,
    RaceCandidate,
    certify_paired_difference,
    race_candidates,
)
from dagua.eval.ruler_v4.scene import FacetResult, value_result
from dagua.eval.ruler_v4.weight_table import SubtermWeight, WeightTable


def _paired_inputs() -> Tuple[Dict[str, FacetResult], WeightTable, CompositionProfile]:
    """Build one active-row paired-certification fixture.

    Returns
    -------
    tuple[dict[str, FacetResult], WeightTable, CompositionProfile]
        Exact facet state, weights, and p-mean profile.
    """

    facets = {"U01": value_result(0.5, {"U01.headline": 0.5})}
    table = WeightTable(
        entries=(SubtermWeight("U01.headline", "U01", "G1", 1.0),),
        d_power=20,
    )
    profile = CompositionProfile(CompositionFamily.P_MEAN, power=2.0)
    return facets, table, profile


def test_shared_unobserved_paired_mass_cancels_exactly() -> None:
    """A full-width common level contributes exactly zero paired uncertainty."""

    facets, table, profile = _paired_inputs()
    certificate = certify_paired_difference(
        facets,
        table,
        profile,
        confidence_id="round-1",
    )

    assert certificate.interval == CertifiedInterval(0.0, 0.0, "round-1")
    assert certificate.error_bound == 0.0
    assert certificate.shared_unobserved_subterms == ("U01.headline",)


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
        RaceCandidate("a", CertifiedInterval(0.1, 0.8), {"b": paired}),
        RaceCandidate("b", CertifiedInterval(0.1, 0.8)),
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
