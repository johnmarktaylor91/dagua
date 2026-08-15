"""Certified interval, rank-fidelity, and gradient-sanity machinery."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence, Tuple, TypeVar

import torch

from dagua.eval.ruler_v4.composition import CompositionProfile, CompositionResult, compose
from dagua.eval.ruler_v4.scene import FacetResult, ResultState
from dagua.eval.ruler_v4.weight_table import WeightTable

SceneT = TypeVar("SceneT")


@dataclass(frozen=True)
class CertifiedInterval:
    """Closed certified interval on a scalar estimand.

    Parameters
    ----------
    lo : float
        Inclusive lower endpoint.
    hi : float
        Inclusive upper endpoint.
    confidence_id : str
        Identifier for the simultaneous/anytime allocation that certified
        the interval. Deterministic exact bounds use ``"deterministic"``.
    """

    lo: float
    hi: float
    confidence_id: str = "deterministic"

    def __post_init__(self) -> None:
        """Validate finite ordered endpoints and allocation identity.

        Raises
        ------
        ValueError
            If endpoints or the confidence identity are invalid.
        """

        if not math.isfinite(self.lo) or not math.isfinite(self.hi) or self.lo > self.hi:
            raise ValueError("certified interval endpoints must be finite and ordered")
        if not self.confidence_id:
            raise ValueError("certified intervals require a confidence identity")

    @property
    def width(self) -> float:
        """Return interval width.

        Returns
        -------
        float
            ``hi - lo``.
        """

        return self.hi - self.lo

    def overlaps(self, other: "CertifiedInterval", margin: float = 0.0) -> bool:
        """Return whether two intervals overlap within a policy margin.

        Parameters
        ----------
        other : CertifiedInterval
            Comparison interval.
        margin : float
            Nonnegative policy margin added symmetrically to the overlap test.

        Returns
        -------
        bool
            Whether neither interval is separated from the other by more than
            ``margin``.

        Raises
        ------
        ValueError
            If ``margin`` is invalid.
        """

        if not math.isfinite(margin) or margin < 0.0:
            raise ValueError("interval overlap margin must be finite and nonnegative")
        return not (self.hi + margin < other.lo or other.hi + margin < self.lo)


@dataclass(frozen=True)
class CertifiedCompositionInterval:
    """Publish a certified interval propagated through exact composition.

    Parameters
    ----------
    interval : CertifiedInterval
        Certified interval on exact ``L_total``.
    lower_composition : CompositionResult
        Exact composition evaluated at all lower defect endpoints.
    upper_composition : CompositionResult
        Exact composition evaluated at all upper defect endpoints.
    term_intervals : mapping[str, CertifiedInterval]
        Effective active-row intervals, including full feasible intervals.
    unobserved_subterms : tuple[str, ...]
        Active rows defaulted to ``[0, 1]`` under CC-13.
    """

    interval: CertifiedInterval
    lower_composition: CompositionResult
    upper_composition: CompositionResult
    term_intervals: Mapping[str, CertifiedInterval]
    unobserved_subterms: Tuple[str, ...]


@dataclass(frozen=True)
class RankFidelityResult:
    """Return a frozen Kendall rank-fidelity certification result.

    Parameters
    ----------
    tau : float
        Kendall tau-b between surrogate and true losses.
    threshold : float
        Explicit fit-time acceptance threshold.
    certified : bool
        Whether ``tau >= threshold``.
    comparable_pairs : int
        Pair count after joint ties are excluded.
    concordant_pairs : int
        Concordant pair count.
    discordant_pairs : int
        Discordant pair count.
    """

    tau: float
    threshold: float
    certified: bool
    comparable_pairs: int
    concordant_pairs: int
    discordant_pairs: int


@dataclass(frozen=True)
class GradientSanityResult:
    """Publish one analytic/finite-difference gradient probe.

    Parameters
    ----------
    value : float
        Probe coordinate.
    analytic_gradient : float
        Autograd derivative of the soft term.
    finite_difference_gradient : float
        Central finite-difference derivative of the exact term.
    agreement : bool
        Whether derivatives agree under explicit tolerances.
    exact_improved : bool
        Whether a negative-gradient step lowers the exact term.
    moved_value : float
        Coordinate after the test step.
    """

    value: float
    analytic_gradient: float
    finite_difference_gradient: float
    agreement: bool
    exact_improved: bool
    moved_value: float


def _active_subterms(exact: CompositionResult) -> Tuple[str, ...]:
    """Return active positive-mass subterm ids from closed composition.

    Parameters
    ----------
    exact : CompositionResult
        Closed composition defining applicability.

    Returns
    -------
    tuple[str, ...]
        Active ids in table order.
    """

    return tuple(
        row.subterm_id
        for row in exact.subterms
        if row.value is not None and not row.diagnostic and row.normalized_weight > 0.0
    )


def _endpoint_facets(
    facet_results: Mapping[str, FacetResult],
    intervals: Mapping[str, CertifiedInterval],
    *,
    upper: bool,
) -> Mapping[str, FacetResult]:
    """Replace active subterms with one hyperrectangle endpoint.

    Parameters
    ----------
    facet_results : mapping[str, FacetResult]
        Exact facet states and applicability.
    intervals : mapping[str, CertifiedInterval]
        Effective active-row intervals.
    upper : bool
        Select upper endpoints when true, lower endpoints otherwise.

    Returns
    -------
    mapping[str, FacetResult]
        Facet rows suitable for the frozen point ``compose`` function.
    """

    endpoints = {}
    for facet_id, result in facet_results.items():
        if result.state is not ResultState.VALUE:
            endpoints[facet_id] = result
            continue
        subterms = dict(result.subterms)
        for subterm_id in set(subterms) & set(intervals):
            interval = intervals[subterm_id]
            subterms[subterm_id] = interval.hi if upper else interval.lo
        endpoints[facet_id] = FacetResult(
            state=result.state,
            value=result.value,
            reason=result.reason,
            subterms=subterms,
            raw=result.raw,
            temporal_headline=result.temporal_headline,
        )
    return endpoints


def compose_certified_intervals(
    facet_results: Mapping[str, FacetResult],
    weight_table: WeightTable,
    profile: CompositionProfile,
    term_intervals: Optional[Mapping[str, CertifiedInterval]] = None,
    *,
    confidence_id: str,
) -> CertifiedCompositionInterval:
    """Propagate simultaneous facet intervals through exact frozen composition.

    Missing active rows receive their full feasible defect range ``[0, 1]``.
    Because the frozen composition is coordinatewise monotone, evaluating its
    exact implementation at the two hyperrectangle endpoints certifies exact
    ``L_total`` without averaging interval bounds.

    Parameters
    ----------
    facet_results : mapping[str, FacetResult]
        Exact facet states used only for applicability and default values.
    weight_table : WeightTable
        Explicit subterm masses.
    profile : CompositionProfile
        Frozen composition family.
    term_intervals : mapping[str, CertifiedInterval] or None
        Observed or bounded active rows. Omitted active rows are unobserved.
    confidence_id : str
        Simultaneous or anytime-valid allocation identity for the joint box.

    Returns
    -------
    CertifiedCompositionInterval
        Certified exact-loss interval and endpoint evidence.

    Raises
    ------
    ValueError
        If a supplied interval is unknown, out of the feasible defect range,
        or belongs to a different confidence allocation.
    """

    supplied = dict(term_intervals or {})
    exact = compose(facet_results, weight_table, profile)
    active = _active_subterms(exact)
    unknown = sorted(set(supplied) - set(active))
    if unknown:
        raise ValueError(f"intervals supplied for inactive subterms: {unknown}")
    for subterm_id, interval in supplied.items():
        if interval.lo < 0.0 or interval.hi > 1.0:
            raise ValueError(f"facet interval outside [0, 1]: {subterm_id}")
        if interval.confidence_id != confidence_id:
            raise ValueError(f"confidence allocation mismatch for {subterm_id}")
    unobserved = tuple(subterm_id for subterm_id in active if subterm_id not in supplied)
    effective = {
        subterm_id: supplied.get(subterm_id, CertifiedInterval(0.0, 1.0, confidence_id))
        for subterm_id in active
    }
    lower = compose(
        _endpoint_facets(facet_results, effective, upper=False),
        weight_table,
        profile,
    )
    upper = compose(
        _endpoint_facets(facet_results, effective, upper=True),
        weight_table,
        profile,
    )
    return CertifiedCompositionInterval(
        interval=CertifiedInterval(lower.l_total, upper.l_total, confidence_id),
        lower_composition=lower,
        upper_composition=upper,
        term_intervals=effective,
        unobserved_subterms=unobserved,
    )


def exact_term_intervals(
    facet_results: Mapping[str, FacetResult],
    weight_table: WeightTable,
    profile: CompositionProfile,
    *,
    confidence_id: str = "deterministic",
) -> Mapping[str, CertifiedInterval]:
    """Build degenerate certified intervals for every exact active term.

    Parameters
    ----------
    facet_results : mapping[str, FacetResult]
        Exact facet outputs.
    weight_table : WeightTable
        Explicit subterm masses.
    profile : CompositionProfile
        Frozen composition family used to determine applicability.
    confidence_id : str
        Joint allocation identity.

    Returns
    -------
    mapping[str, CertifiedInterval]
        Exact active-row intervals.
    """

    exact = compose(facet_results, weight_table, profile)
    return {
        row.subterm_id: CertifiedInterval(row.value, row.value, confidence_id)
        for row in exact.subterms
        if row.value is not None and not row.diagnostic and row.normalized_weight > 0.0
    }


def certify_rank_fidelity(
    true_losses: Sequence[float],
    surrogate_losses: Sequence[float],
    *,
    threshold: float,
) -> RankFidelityResult:
    """Certify surrogate ranking with deterministic Kendall tau-b.

    Parameters
    ----------
    true_losses : sequence[float]
        Full-ruler pre-map losses.
    surrogate_losses : sequence[float]
        Surrogate pre-map losses in the same scene order.
    threshold : float
        Explicit fit-time tau floor. The r4 contract currently requires at
        least 0.85; callers still pass it so later fitted policy is visible.

    Returns
    -------
    RankFidelityResult
        Tau, counts, and certification verdict.

    Raises
    ------
    ValueError
        If inputs are malformed or contain fewer than two scenes.
    """

    if len(true_losses) != len(surrogate_losses) or len(true_losses) < 2:
        raise ValueError("rank fidelity requires equal batches of at least two scenes")
    if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
        raise ValueError("rank-fidelity threshold must lie in [-1, 1]")
    true_values = tuple(float(value) for value in true_losses)
    soft_values = tuple(float(value) for value in surrogate_losses)
    if any(not math.isfinite(value) for value in (*true_values, *soft_values)):
        raise ValueError("rank-fidelity losses must be finite")

    concordant = 0
    discordant = 0
    true_ties = 0
    soft_ties = 0
    for left in range(len(true_values)):
        for right in range(left + 1, len(true_values)):
            true_sign = (true_values[left] > true_values[right]) - (
                true_values[left] < true_values[right]
            )
            soft_sign = (soft_values[left] > soft_values[right]) - (
                soft_values[left] < soft_values[right]
            )
            if true_sign == 0 and soft_sign == 0:
                continue
            if true_sign == 0:
                true_ties += 1
            elif soft_sign == 0:
                soft_ties += 1
            elif true_sign == soft_sign:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + true_ties) * (concordant + discordant + soft_ties)
    )
    tau = (concordant - discordant) / denominator if denominator > 0.0 else 1.0
    return RankFidelityResult(
        tau=tau,
        threshold=threshold,
        certified=tau >= threshold,
        comparable_pairs=concordant + discordant,
        concordant_pairs=concordant,
        discordant_pairs=discordant,
    )


def certify_scene_batch(
    scenes: Sequence[SceneT],
    true_scorer: Callable[[SceneT], float],
    surrogate_scorer: Callable[[SceneT], float],
    *,
    threshold: float,
) -> RankFidelityResult:
    """Run rank certification on a generated deterministic scene batch.

    Parameters
    ----------
    scenes : sequence[SceneT]
        Generated scenes in deterministic order.
    true_scorer : callable
        Full-ruler loss callback.
    surrogate_scorer : callable
        Compiled surrogate loss callback.
    threshold : float
        Explicit Kendall tau floor.

    Returns
    -------
    RankFidelityResult
        Batch rank-fidelity result.
    """

    true_losses = [float(true_scorer(scene)) for scene in scenes]
    surrogate_losses = [float(surrogate_scorer(scene)) for scene in scenes]
    return certify_rank_fidelity(true_losses, surrogate_losses, threshold=threshold)


def certify_gradient_sanity(
    exact_function: Callable[[float], float],
    soft_function: Callable[[torch.Tensor], torch.Tensor],
    value: float,
    *,
    step_size: float,
    finite_difference_epsilon: float,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> GradientSanityResult:
    """Check analytic agreement and exact improvement for one scalar probe.

    Parameters
    ----------
    exact_function : callable
        Exact scalar facet/composition function.
    soft_function : callable
        Differentiable analytic surrogate of the same scalar coordinate.
    value : float
        Probe coordinate.
    step_size : float
        Positive negative-gradient step multiplier.
    finite_difference_epsilon : float
        Positive central finite-difference radius.
    relative_tolerance : float
        Nonnegative derivative agreement tolerance.
    absolute_tolerance : float
        Nonnegative derivative agreement tolerance.

    Returns
    -------
    GradientSanityResult
        Derivatives and exact-improvement verdict.

    Raises
    ------
    ValueError
        If probe parameters are non-finite or outside their legal domains.
    """

    parameters = (
        value,
        step_size,
        finite_difference_epsilon,
        relative_tolerance,
        absolute_tolerance,
    )
    if any(not math.isfinite(parameter) for parameter in parameters):
        raise ValueError("gradient-sanity parameters must be finite")
    if step_size <= 0.0 or finite_difference_epsilon <= 0.0:
        raise ValueError("gradient-sanity step and epsilon must be positive")
    if relative_tolerance < 0.0 or absolute_tolerance < 0.0:
        raise ValueError("gradient-sanity tolerances must be nonnegative")

    probe = torch.tensor(value, dtype=torch.float64, requires_grad=True)
    soft_value = soft_function(probe)
    if soft_value.numel() != 1:
        raise ValueError("soft gradient probe must return one scalar")
    soft_value.backward()
    if probe.grad is None or not bool(torch.isfinite(probe.grad).item()):
        raise ValueError("soft gradient probe produced no finite gradient")
    analytic = float(probe.grad)
    finite_difference = (
        exact_function(value + finite_difference_epsilon)
        - exact_function(value - finite_difference_epsilon)
    ) / (2.0 * finite_difference_epsilon)
    moved = value - step_size * analytic
    agreement = math.isclose(
        analytic,
        finite_difference,
        rel_tol=relative_tolerance,
        abs_tol=absolute_tolerance,
    )
    return GradientSanityResult(
        value=value,
        analytic_gradient=analytic,
        finite_difference_gradient=finite_difference,
        agreement=agreement,
        exact_improved=exact_function(moved) < exact_function(value),
        moved_value=moved,
    )
