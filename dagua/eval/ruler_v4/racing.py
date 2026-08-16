"""Paired certified-difference bounds and adaptive candidate racing."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from dagua.eval.ruler_v4.certification import CertifiedInterval
from dagua.eval.ruler_v4.composition import (
    CompositionFamily,
    CompositionProfile,
    CompositionResult,
    compose,
)
from dagua.eval.ruler_v4.scene import FacetResult, ResultState
from dagua.eval.ruler_v4.weight_table import WeightTable


@dataclass(frozen=True)
class PairedTermRegion:
    """Certified level+difference region for one paired facet term.

    Parameters
    ----------
    level : CertifiedInterval
        Shared level ``c_f`` for candidate B.
    difference : CertifiedInterval
        CRN-paired difference ``D_f = d_f^A - d_f^B``.
    """

    level: CertifiedInterval
    difference: CertifiedInterval


@dataclass(frozen=True)
class PairedDifferenceCertificate:
    """Certified interval on ``L_total(A) - L_total(B)``.

    Parameters
    ----------
    interval : CertifiedInterval
        Paired exact-functional difference interval.
    center_difference : float
        Exact frozen composition difference at the region midpoint.
    error_bound : float
        Sensitivity-propagated radius around the midpoint.
    sensitivity_bounds : mapping[str, float]
        Published per-argument ``Lambda_f`` bounds.
    shared_unobserved_subterms : tuple[str, ...]
        Missing rows whose paired difference is certified exactly zero.
        A zero difference kills only the direct channel: under the frozen
        nonlinear families the shared level still prices every other row's
        certified difference, so these rows are charged their full level
        oscillation bound ``2 * Lambda_f * level_radius`` rather than
        being treated as an exact cancellation.
    """

    interval: CertifiedInterval
    center_difference: float
    error_bound: float
    sensitivity_bounds: Mapping[str, float]
    shared_unobserved_subterms: Tuple[str, ...]


class EliminationRule(str, Enum):
    """Certified rule responsible for an elimination."""

    PAIRED_DIFFERENCE = "paired_difference"
    MARGINAL_BOUND = "marginal_bound"


class InconsistentCertificateError(ValueError):
    """Certified eliminations emptied the survivor set.

    Sound certificates cannot exclude every candidate, so an empty survivor
    set is a certificate-version failure. Racing fails closed with this loud
    typed artifact (V4_SPEC_r4 6.4) instead of selecting from nothing.
    """


@dataclass(frozen=True)
class RaceCandidate:
    """One candidate and its current exact-loss interval.

    Parameters
    ----------
    candidate_id : str
        Stable candidate identity.
    loss_interval : CertifiedInterval
        Certified interval on frozen exact ``L_total``.
    paired_differences : mapping[str, PairedDifferenceCertificate]
        Certificates keyed by comparison candidate B, each estimating
        ``L_total(self) - L_total(B)``.
    """

    candidate_id: str
    loss_interval: CertifiedInterval
    paired_differences: Mapping[str, PairedDifferenceCertificate] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze paired certificates and validate candidate identities.

        Raises
        ------
        ValueError
            If identities are empty or self-referential.
        """

        if not self.candidate_id:
            raise ValueError("race candidate ids must be nonempty")
        pairs = dict(self.paired_differences)
        if any(not candidate_id or candidate_id == self.candidate_id for candidate_id in pairs):
            raise ValueError("paired certificates require distinct nonempty candidate ids")
        object.__setattr__(self, "paired_differences", MappingProxyType(pairs))


@dataclass(frozen=True)
class EliminationRecord:
    """Record one certified surrogate-tier elimination.

    Parameters
    ----------
    candidate_id : str
        Eliminated candidate.
    against_id : str
        Candidate whose bound certified the elimination.
    rule : EliminationRule
        Primary paired or additional marginal sufficient rule.
    interval : CertifiedInterval
        Difference or candidate loss interval used by the rule.
    """

    candidate_id: str
    against_id: str
    rule: EliminationRule
    interval: CertifiedInterval


@dataclass(frozen=True)
class RaceResult:
    """Publish deterministic racing decisions and escalation evidence.

    Parameters
    ----------
    winner_id : str or None
        Selected candidate, the incumbent on exhausted budget, or ``None``
        when the race is inconclusive because the budget was exhausted and
        the incumbent carries a certified elimination (an eliminated
        incumbent is not an admissible fallback winner).
    eliminations : tuple[EliminationRecord, ...]
        Certified pre-escalation eliminations.
    escalated_ids : tuple[str, ...]
        Candidates evaluated by the true scorer.
    true_losses : mapping[str, float]
        Exact losses obtained during escalation.
    budget_exhausted : bool
        Whether the true-score budget could not cover all survivors.
    reason : str
        Stable outcome reason.
    """

    winner_id: Optional[str]
    eliminations: Tuple[EliminationRecord, ...]
    escalated_ids: Tuple[str, ...]
    true_losses: Mapping[str, float]
    budget_exhausted: bool
    reason: str


def _facets_with_values(
    facet_results: Mapping[str, FacetResult],
    values: Mapping[str, float],
) -> Mapping[str, FacetResult]:
    """Bind scalar subterm coordinates into closed facet result shapes.

    Parameters
    ----------
    facet_results : mapping[str, FacetResult]
        Exact facet states defining applicability.
    values : mapping[str, float]
        Active subterm replacements.

    Returns
    -------
    mapping[str, FacetResult]
        Results consumable by frozen ``compose``.
    """

    bound = {}
    for facet_id, result in facet_results.items():
        if result.state is not ResultState.VALUE:
            bound[facet_id] = result
            continue
        subterms = dict(result.subterms)
        for subterm_id in set(subterms) & set(values):
            subterms[subterm_id] = values[subterm_id]
        bound[facet_id] = FacetResult(
            state=result.state,
            value=result.value,
            reason=result.reason,
            subterms=subterms,
            raw=result.raw,
            temporal_headline=result.temporal_headline,
        )
    return bound


def _active_rows(exact: CompositionResult) -> Mapping[str, Tuple[float, str]]:
    """Return normalized mass and group for active exact rows.

    Parameters
    ----------
    exact : CompositionResult
        Frozen composition result.

    Returns
    -------
    mapping[str, tuple[float, str]]
        Active subterm metadata.
    """

    return {
        row.subterm_id: (row.normalized_weight, row.group)
        for row in exact.subterms
        if row.value is not None and not row.diagnostic and row.normalized_weight > 0.0
    }


def _sensitivity_bounds(
    exact: CompositionResult,
    profile: CompositionProfile,
) -> Mapping[str, float]:
    """Publish global per-term sensitivity bounds for the frozen family.

    Parameters
    ----------
    exact : CompositionResult
        Composition used to recover applicable masses and groups.
    profile : CompositionProfile
        Frozen family whose derivative is bounded.

    Returns
    -------
    mapping[str, float]
        Safe global ``sup |dL/dd_f|`` values over ``[0, 1]``.
    """

    rows = _active_rows(exact)
    if profile.family is CompositionFamily.P_MEAN:
        assert profile.power is not None
        # Weighted p-norm coordinate derivatives are bounded by a_i^(1/p),
        # including the one-sided derivative at the all-zero origin.
        return {
            subterm_id: normalized_weight ** (1.0 / profile.power)
            for subterm_id, (normalized_weight, _) in rows.items()
        }

    assert profile.bottleneck_mix is not None
    group_masses: Dict[str, float] = {}
    for normalized_weight, group in rows.values():
        group_masses[group] = group_masses.get(group, 0.0) + normalized_weight
    return {
        subterm_id: (1.0 - profile.bottleneck_mix) * normalized_weight
        + profile.bottleneck_mix * normalized_weight / group_masses[group]
        for subterm_id, (normalized_weight, group) in rows.items()
    }


def certify_paired_difference(
    facet_results: Mapping[str, FacetResult],
    weight_table: WeightTable,
    profile: CompositionProfile,
    paired_terms: Optional[Mapping[str, PairedTermRegion]] = None,
    *,
    confidence_id: str,
) -> PairedDifferenceCertificate:
    """Certify the primary paired exact-functional difference.

    The midpoint is evaluated through the frozen ``compose`` implementation.
    Uncertainty is propagated with published per-argument sensitivity bounds:
    every active row is charged ``Lambda_f * (2 * level_radius +
    difference_radius)``. A missing term is shared unobserved mass: its level
    is ``[0, 1]`` and its CRN-paired difference is exactly ``[0, 0]``. The
    zero difference removes only the direct channel; under a nonlinear frozen
    family (``P_MEAN`` with ``p > 1``, ``MEAN_SOFT_BOTTLENECK``) the shared
    level still sets the exchange rate ``dL/dd`` at ``c + D`` versus ``c``
    for every other certified difference, so the row keeps its
    ``2 * Lambda_f * level_radius`` charge. This is single-counted level
    oscillation, still strictly tighter than double-counted marginals; an
    exact cancellation claim would be valid only for a linear composition.
    This is a knowing deviation from 6.2a's "so it cancels" sentence,
    docketed with the derivation, the measured zero pruning-power gap it
    produces at pilot scale, and the curvature-bound alternative in
    DISCREPANCIES entry 43.

    Parameters
    ----------
    facet_results : mapping[str, FacetResult]
        Exact states defining shared applicability.
    weight_table : WeightTable
        Explicit term masses.
    profile : CompositionProfile
        Frozen composition family.
    paired_terms : mapping[str, PairedTermRegion] or None
        Observed level+difference regions. Missing active terms cancel.
    confidence_id : str
        Simultaneous/anytime allocation identity.

    Returns
    -------
    PairedDifferenceCertificate
        Certified interval on ``L_total(A) - L_total(B)``.

    Raises
    ------
    ValueError
        If regions are inactive, allocation-mismatched, or leave the feasible
        defect range for either candidate.
    """

    supplied = dict(paired_terms or {})
    exact = compose(facet_results, weight_table, profile)
    active = _active_rows(exact)
    unknown = sorted(set(supplied) - set(active))
    if unknown:
        raise ValueError(f"paired regions supplied for inactive subterms: {unknown}")
    shared_unobserved = tuple(subterm_id for subterm_id in active if subterm_id not in supplied)
    effective = {
        subterm_id: supplied.get(
            subterm_id,
            PairedTermRegion(
                CertifiedInterval(0.0, 1.0, confidence_id),
                CertifiedInterval(0.0, 0.0, confidence_id),
            ),
        )
        for subterm_id in active
    }
    center_b = {}
    center_a = {}
    for subterm_id, region in effective.items():
        if (
            region.level.confidence_id != confidence_id
            or region.difference.confidence_id != confidence_id
        ):
            raise ValueError(f"confidence allocation mismatch for {subterm_id}")
        if region.level.lo < 0.0 or region.level.hi > 1.0:
            raise ValueError(f"paired level outside [0, 1]: {subterm_id}")
        if region.level.lo + region.difference.lo < 0.0:
            raise ValueError(f"paired A lower defect outside [0, 1]: {subterm_id}")
        if region.level.hi + region.difference.hi > 1.0:
            raise ValueError(f"paired A upper defect outside [0, 1]: {subterm_id}")
        level_midpoint = 0.5 * (region.level.lo + region.level.hi)
        difference_midpoint = 0.5 * (region.difference.lo + region.difference.hi)
        center_b[subterm_id] = level_midpoint
        center_a[subterm_id] = level_midpoint + difference_midpoint

    first = compose(_facets_with_values(facet_results, center_a), weight_table, profile)
    second = compose(_facets_with_values(facet_results, center_b), weight_table, profile)
    center_difference = first.l_total - second.l_total
    sensitivities = _sensitivity_bounds(exact, profile)
    error_terms: List[float] = []
    for subterm_id, region in effective.items():
        # No linearity skip for D == [0, 0] rows: a zero certified paired
        # difference cancels the direct channel only. The row's shared level
        # still moves sup |dL/dd_i(c + D) - dL/dd_i(c)| for the nonlinear
        # frozen families, so it is charged the full oscillation bound
        # 2 * Lambda_f * level_radius (the triangle-inequality bound the
        # observed rows already carry).
        level_radius = 0.5 * region.level.width
        difference_radius = 0.5 * region.difference.width
        error_terms.append(sensitivities[subterm_id] * (2.0 * level_radius + difference_radius))
    error_bound = math.fsum(error_terms)
    return PairedDifferenceCertificate(
        interval=CertifiedInterval(
            center_difference - error_bound,
            center_difference + error_bound,
            confidence_id,
        ),
        center_difference=center_difference,
        error_bound=error_bound,
        sensitivity_bounds=MappingProxyType(dict(sensitivities)),
        shared_unobserved_subterms=shared_unobserved,
    )


def _validate_race_inputs(
    candidates: Sequence[RaceCandidate],
    incumbent_id: str,
    policy_margin: float,
    full_score_max_candidates: int,
    true_score_budget: Optional[int],
) -> Mapping[str, RaceCandidate]:
    """Validate candidate identities and explicit policy parameters.

    Parameters
    ----------
    candidates : sequence[RaceCandidate]
        Candidate set.
    incumbent_id : str
        Budget-exhaustion fallback.
    policy_margin : float
        Nonnegative elimination margin.
    full_score_max_candidates : int
        Small-contest full-score threshold.
    true_score_budget : int or None
        Maximum escalation evaluations.

    Returns
    -------
    mapping[str, RaceCandidate]
        Candidate lookup in input order.

    Raises
    ------
    ValueError
        If any policy or identity invariant fails.
    """

    lookup = {candidate.candidate_id: candidate for candidate in candidates}
    if not candidates or len(lookup) != len(candidates):
        raise ValueError("racing requires a nonempty unique candidate set")
    if incumbent_id not in lookup:
        raise ValueError("race incumbent must be in the candidate set")
    if not math.isfinite(policy_margin) or policy_margin < 0.0:
        raise ValueError("race policy margin must be finite and nonnegative")
    if isinstance(full_score_max_candidates, bool) or full_score_max_candidates < 0:
        raise ValueError("small-contest threshold must be a nonnegative integer")
    if true_score_budget is not None and (
        isinstance(true_score_budget, bool) or true_score_budget < 0
    ):
        raise ValueError("true-score budget must be a nonnegative integer or None")
    allocation_ids = set()
    for candidate in candidates:
        unknown_pairs = sorted(set(candidate.paired_differences) - set(lookup))
        if unknown_pairs:
            raise ValueError(f"paired certificates name unknown candidates: {unknown_pairs}")
        allocation_ids.add(candidate.loss_interval.confidence_id)
        for certificate in candidate.paired_differences.values():
            allocation_ids.add(certificate.interval.confidence_id)
    if len(allocation_ids) > 1:
        # 6.2b's simultaneous coverage is claimed across candidates, facets,
        # and rounds; intervals from different allocations cannot race.
        raise ValueError(
            "race requires one confidence allocation across all candidate "
            f"intervals and paired certificates, got: {sorted(allocation_ids)}"
        )
    return lookup


def _certified_eliminations(
    candidates: Sequence[RaceCandidate],
    protected_ids: Sequence[str],
    policy_margin: float,
) -> Tuple[EliminationRecord, ...]:
    """Apply paired-primary then marginal-sufficient elimination rules.

    Parameters
    ----------
    candidates : sequence[RaceCandidate]
        Current tier candidates.
    protected_ids : sequence[str]
        Geometry-diversity candidates carried through surrogate pruning.
    policy_margin : float
        Required exclusion beyond zero.

    Returns
    -------
    tuple[EliminationRecord, ...]
        At most one deterministic record per eliminated candidate.
    """

    protected = set(protected_ids)
    records = []
    for candidate in candidates:
        if candidate.candidate_id in protected:
            continue
        paired_record = None
        for other_id in sorted(candidate.paired_differences):
            certificate = candidate.paired_differences[other_id]
            if certificate.interval.lo > policy_margin:
                paired_record = EliminationRecord(
                    candidate.candidate_id,
                    other_id,
                    EliminationRule.PAIRED_DIFFERENCE,
                    certificate.interval,
                )
                break
        if paired_record is not None:
            records.append(paired_record)
            continue
        better = min(
            (other for other in candidates if other.candidate_id != candidate.candidate_id),
            key=lambda other: (other.loss_interval.hi, other.candidate_id),
            default=None,
        )
        if (
            better is not None
            and candidate.loss_interval.lo > better.loss_interval.hi + policy_margin
        ):
            records.append(
                EliminationRecord(
                    candidate.candidate_id,
                    better.candidate_id,
                    EliminationRule.MARGINAL_BOUND,
                    candidate.loss_interval,
                )
            )
    return tuple(records)


def race_candidates(
    candidates: Sequence[RaceCandidate],
    true_scorer: Callable[[str], float],
    *,
    incumbent_id: str,
    policy_margin: float,
    full_score_max_candidates: int,
    true_score_budget: Optional[int],
    protected_candidate_ids: Sequence[str] = (),
) -> RaceResult:
    """Race candidates on certified intervals and escalate only unresolved sets.

    Small contests are full-scored by explicit policy. Larger contests apply
    paired-primary and marginal-sufficient certificates. If one survivor
    remains, no true score is requested. Multiple survivors necessarily lack
    a certified separation from the best admissible set and are escalated.
    When the true-score budget cannot cover that whole overlap set, selection
    fails closed to the incumbent unless the incumbent itself carries a
    certified elimination, in which case the race returns a typed
    inconclusive result (``winner_id=None``). An empty survivor set raises
    :class:`InconsistentCertificateError`. All candidate intervals and paired
    certificates must share one confidence allocation.

    Parameters
    ----------
    candidates : sequence[RaceCandidate]
        Deterministic candidate set.
    true_scorer : callable
        Exact full-tier loss callback keyed by candidate id.
    incumbent_id : str
        Winner on budget exhaustion and exact ties.
    policy_margin : float
        Explicit fit-time elimination margin.
    full_score_max_candidates : int
        Contests at or below this size bypass surrogate pruning.
    true_score_budget : int or None
        Maximum true-score calls, or ``None`` for no cap.
    protected_candidate_ids : sequence[str]
        Geometry-diversity identities protected from surrogate elimination.

    Returns
    -------
    RaceResult
        Winner, certificates, escalation calls, and reason.
    """

    lookup = _validate_race_inputs(
        candidates,
        incumbent_id,
        policy_margin,
        full_score_max_candidates,
        true_score_budget,
    )
    unknown_protected = sorted(set(protected_candidate_ids) - set(lookup))
    if unknown_protected:
        raise ValueError(f"protected ids are not candidates: {unknown_protected}")

    if len(candidates) <= full_score_max_candidates:
        survivors = list(candidates)
        eliminations: Tuple[EliminationRecord, ...] = ()
        reason_prefix = "small_contest_full_score"
    else:
        eliminations = _certified_eliminations(
            candidates,
            protected_candidate_ids,
            policy_margin,
        )
        eliminated_ids = {record.candidate_id for record in eliminations}
        survivors = [
            candidate for candidate in candidates if candidate.candidate_id not in eliminated_ids
        ]
        reason_prefix = "overlap_escalation"

    if not survivors:
        raise InconsistentCertificateError(
            "certified eliminations excluded every candidate; sound "
            "certificates cannot do this, so the certificate version has "
            "failed: "
            + ", ".join(
                f"{record.candidate_id} by {record.against_id} ({record.rule.value})"
                for record in eliminations
            )
        )

    if len(survivors) == 1:
        return RaceResult(
            winner_id=survivors[0].candidate_id,
            eliminations=eliminations,
            escalated_ids=(),
            true_losses=MappingProxyType({}),
            budget_exhausted=False,
            reason="certified_interval_winner",
        )

    required = len(survivors)
    budget = required if true_score_budget is None else true_score_budget
    if budget < required:
        eliminated_ids = {record.candidate_id for record in eliminations}
        if incumbent_id in eliminated_ids:
            # A certifiably eliminated incumbent is not an admissible
            # fallback winner: the same result would certify its winner as
            # strictly worse than a published row. Fail to a typed
            # inconclusive outcome instead.
            return RaceResult(
                winner_id=None,
                eliminations=eliminations,
                escalated_ids=(),
                true_losses=MappingProxyType({}),
                budget_exhausted=True,
                reason="budget_exhausted_no_selection",
            )
        return RaceResult(
            winner_id=incumbent_id,
            eliminations=eliminations,
            escalated_ids=(),
            true_losses=MappingProxyType({}),
            budget_exhausted=True,
            reason="budget_exhausted_incumbent",
        )

    true_losses = {}
    for candidate in survivors:
        value = float(true_scorer(candidate.candidate_id))
        if not math.isfinite(value):
            raise ValueError(f"true scorer returned non-finite loss for {candidate.candidate_id}")
        true_losses[candidate.candidate_id] = value
    best_value = min(true_losses.values())
    tied = sorted(
        candidate_id for candidate_id, value in true_losses.items() if value == best_value
    )
    winner = incumbent_id if incumbent_id in tied else tied[0]
    return RaceResult(
        winner_id=winner,
        eliminations=eliminations,
        escalated_ids=tuple(candidate.candidate_id for candidate in survivors),
        true_losses=MappingProxyType(true_losses),
        budget_exhausted=False,
        reason=f"{reason_prefix}_true_score",
    )
