"""Loss-space composition and event-margin comparison for RULER V4."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Dict, List, Mapping, Optional, Tuple

from dagua.eval.ruler_v4.events import EventRegistry, evaluate_jump_bound
from dagua.eval.ruler_v4.scene import FacetResult, ResultState
from dagua.eval.ruler_v4.weights import SubtermWeight, WeightTable


class CompositionFamily(str, Enum):
    """Supported r4 loss-space composition families."""

    P_MEAN = "p_mean"
    MEAN_SOFT_BOTTLENECK = "mean_soft_bottleneck"


class ComparisonVerdict(str, Enum):
    """Event-margin comparison outcomes."""

    FIRST_WINS = "FIRST_WINS"
    SECOND_WINS = "SECOND_WINS"
    TIE = "TIE"
    EVENT_MARGIN_LIMITED = "EVENT_MARGIN_LIMITED"


@dataclass(frozen=True)
class CompositionProfile:
    """Declare every composition-family parameter explicitly.

    Parameters
    ----------
    family : CompositionFamily
        Loss-space family selected under R3-DR.
    power : float or None
        Global p-mean exponent, required for ``P_MEAN`` and constrained to at
        least one.
    bottleneck_mix : float or None
        Positive convex weight beta, required for ``MEAN_SOFT_BOTTLENECK``.
    bottleneck_temperature : float or None
        Positive smooth-maximum temperature tau.
    group_allowances : mapping[str, float]
        Explicit nonnegative allowance for every applicable bottleneck group.
    """

    family: CompositionFamily
    power: Optional[float] = None
    bottleneck_mix: Optional[float] = None
    bottleneck_temperature: Optional[float] = None
    group_allowances: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the chosen family's explicit parameter surface.

        Raises
        ------
        ValueError
            If a required parameter is missing or outside its legal domain.
        """

        allowances = {key: float(value) for key, value in self.group_allowances.items()}
        if any(
            not key or not math.isfinite(value) or value < 0.0 for key, value in allowances.items()
        ):
            raise ValueError("group allowances must be finite, nonnegative, and named")
        if self.family is CompositionFamily.P_MEAN:
            if self.power is None or not math.isfinite(self.power) or self.power < 1.0:
                raise ValueError("p-mean composition requires finite power >= 1")
            if self.bottleneck_mix is not None or self.bottleneck_temperature is not None:
                raise ValueError("p-mean composition does not consume bottleneck parameters")
            if allowances:
                raise ValueError("p-mean composition does not consume group allowances")
        elif self.family is CompositionFamily.MEAN_SOFT_BOTTLENECK:
            if (
                self.bottleneck_mix is None
                or not math.isfinite(self.bottleneck_mix)
                or not 0.0 < self.bottleneck_mix <= 1.0
            ):
                raise ValueError("soft-bottleneck composition requires beta in (0, 1]")
            if (
                self.bottleneck_temperature is None
                or not math.isfinite(self.bottleneck_temperature)
                or self.bottleneck_temperature <= 0.0
            ):
                raise ValueError("soft-bottleneck composition requires positive tau")
            if self.power is not None:
                raise ValueError("soft-bottleneck composition does not consume p")
        else:
            raise ValueError(f"unsupported composition family: {self.family}")
        object.__setattr__(self, "group_allowances", MappingProxyType(allowances))


@dataclass(frozen=True)
class SubtermContribution:
    """Publish one sub-term's exact mean-space attribution.

    Parameters
    ----------
    subterm_id : str
        Frozen scored sub-term id.
    facet_id : str
        Owning facet id.
    group : str
        Reporting group.
    value : float or None
        Facet defect, or ``None`` when the row is unavailable.
    declared_weight : float
        Explicit table mass before NA renormalization.
    normalized_weight : float
        Headline mass after excluding NA and diagnostic terms.
    weighted_loss : float
        Exact contribution to ``L_mean``.
    diagnostic : bool
        Whether the term is carried outside the headline.
    prior_driven : bool
        Whether its mass contributes to PM-1.
    na_reason : str or None
        Machine-readable facet or sub-term absence reason.
    """

    subterm_id: str
    facet_id: str
    group: str
    value: Optional[float]
    declared_weight: float
    normalized_weight: float
    weighted_loss: float
    diagnostic: bool
    prior_driven: bool
    na_reason: Optional[str]


@dataclass(frozen=True)
class GroupContribution:
    """Publish one reporting group's rolled-up loss attribution.

    Under the shipped ``P_MEAN`` family the group label is a pure reporting
    rollup with no weight semantics of its own (V4_SPEC_r4 3.3): relabelling
    groups never changes ``l_total``. Only the optional soft-bottleneck family
    consumes the partition, through its explicit per-group allowances
    (DISCREPANCIES entry 33).

    Parameters
    ----------
    group : str
        Stable reporting-group id.
    loss : float
        Weighted mean defect within the group.
    mass : float
        Applicable positive mass before global normalization.
    normalized_mass : float
        Fraction of total applicable headline mass.
    allowance : float or None
        Soft-bottleneck allowance when that family is active.
    """

    group: str
    loss: float
    mass: float
    normalized_mass: float
    allowance: Optional[float]


@dataclass(frozen=True)
class CompositionResult:
    """Return the exact pre-map loss and attribution table.

    Parameters
    ----------
    family : CompositionFamily
        Family used for the result.
    l_mean : float
        Applicable-mass weighted arithmetic loss.
    l_bottleneck : float or None
        Nonnegative soft-bottleneck debt, when active.
    l_total : float
        Exact loss used by ordering and the headline map.
    total_mass : float
        Applicable non-diagnostic mass.
    groups : tuple[GroupContribution, ...]
        Stable group attribution.
    subterms : tuple[SubtermContribution, ...]
        Stable per-sub-term attribution, including NA and diagnostics.
    """

    family: CompositionFamily
    l_mean: float
    l_bottleneck: Optional[float]
    l_total: float
    total_mass: float
    groups: Tuple[GroupContribution, ...]
    subterms: Tuple[SubtermContribution, ...]


@dataclass(frozen=True)
class NearbyEvent:
    """Identify one declared manifold occurrence near a scored row.

    Parameters
    ----------
    event_id : str
        Frozen event-registry type id.
    context : mapping[str, float]
        Graph-local inputs for a closed-form jump bound.
    manifold_id : str or None
        Stable occurrence id. Occurrences are merged ONLY through an explicit
        shared id: every id-less occurrence enters the CC-1 margin sum as its
        own manifold, so one occurrence near both rows is double-charged
        unless the caller identifies it. Omission is always conservative
        (a larger event budget), never a way to shrink the sum.
    """

    event_id: str
    context: Mapping[str, float] = field(default_factory=dict)
    manifold_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Freeze and validate graph-local bound inputs.

        Raises
        ------
        ValueError
            If an id is empty or an input is not finite.
        """

        values = {key: float(value) for key, value in self.context.items()}
        if not self.event_id or (self.manifold_id is not None and not self.manifold_id):
            raise ValueError("event and manifold ids must be nonempty")
        if any(not key or not math.isfinite(value) for key, value in values.items()):
            raise ValueError("event-bound context must contain finite named values")
        object.__setattr__(self, "context", MappingProxyType(values))


@dataclass(frozen=True)
class ComparisonResult:
    """Publish a pre-map pair result and its event-margin budget.

    Parameters
    ----------
    verdict : ComparisonVerdict
        Strict, tied, or event-margin-limited outcome.
    decision_margin : float
        Absolute difference between exact ``L_total`` values.
    event_margin : float
        Sum of nearby declared jump bounds from either row.
    nearby_event_ids : tuple[str, ...]
        Stable event types used in the margin.
    nearby_manifold_ids : tuple[str, ...]
        Stable occurrence identities used in the margin.
    """

    verdict: ComparisonVerdict
    decision_margin: float
    event_margin: float
    nearby_event_ids: Tuple[str, ...]
    nearby_manifold_ids: Tuple[str, ...]


def _smooth_max(values: Tuple[float, ...], temperature: float) -> float:
    """Compute a stable normalized log-sum-exp smooth maximum.

    Parameters
    ----------
    values : tuple[float, ...]
        Nonempty group debts.
    temperature : float
        Positive soft-maximum temperature.

    Returns
    -------
    float
        Smooth maximum with identity behavior for equal inputs.
    """

    scaled = tuple(value / temperature for value in values)
    maximum = max(scaled)
    mean_exponential = math.fsum(math.exp(value - maximum) for value in scaled) / len(scaled)
    return temperature * (maximum + math.log(mean_exponential))


def _smooth_positive(value: float, temperature: float) -> float:
    """Apply a C1 positive-part onset to the smoothed excess debt.

    Parameters
    ----------
    value : float
        Signed smoothed excess over group allowances.
    temperature : float
        Positive onset scale shared with the smooth maximum.

    Returns
    -------
    float
        Zero for ordinary debt and a C1 increasing burden above it.
    """

    if value <= 0.0:
        return 0.0
    return value * value / (value + temperature)


def _dropped_subterm_reason(result: FacetResult, subterm_id: str) -> Optional[str]:
    """Recover a published reason for one unavailable sub-term.

    Parameters
    ----------
    result : FacetResult
        Owning facet result.
    subterm_id : str
        Missing scored row id.

    Returns
    -------
    str or None
        Published drop reason, facet NA reason, or ``None`` when available.
    """

    if result.state is ResultState.NA:
        return result.reason
    if result.state is not ResultState.VALUE or subterm_id in result.subterms:
        return None
    dropped = result.raw.get("dropped_subterms", ())
    if isinstance(dropped, (tuple, list)):
        prefix = f"{subterm_id}:"
        for item in dropped:
            if isinstance(item, str) and item.startswith(prefix):
                return item[len(prefix) :]
    return "SUBTERM_NOT_APPLICABLE"


def compose(
    facet_results: Mapping[str, FacetResult],
    weight_table: WeightTable,
    profile: CompositionProfile,
) -> CompositionResult:
    """Compose independent facets in loss space with exact NA exclusion.

    The shipped ``P_MEAN`` family aggregates the frozen scored sub-term rows
    directly; ``SubtermWeight.group`` is a reporting rollup with no weight
    semantics (V4_SPEC_r4 3.3). The optional soft-bottleneck family consumes
    the group partition through its explicit allowances; that partition is a
    P5/freeze input (DISCREPANCIES entry 33).

    Parameters
    ----------
    facet_results : mapping[str, FacetResult]
        Independent Phase-1 results keyed by facet id.
    weight_table : WeightTable
        Explicit P5/frozen sub-term masses.
    profile : CompositionProfile
        Explicit across-group family parameters.

    Returns
    -------
    CompositionResult
        Pre-map loss plus group and sub-term attributions.

    Raises
    ------
    ValueError
        If a facet is missing or invalid, or no positive headline mass applies.
    """

    active_entries: List[Tuple[SubtermWeight, float]] = []
    for entry in weight_table.entries:
        if entry.facet_id not in facet_results:
            raise ValueError(f"missing facet result for {entry.facet_id}")
        result = facet_results[entry.facet_id]
        if result.state is ResultState.INVALID:
            raise ValueError(f"invalid facet result for {entry.facet_id}: {result.reason}")
        if (
            result.state is ResultState.VALUE
            and entry.subterm_id in result.subterms
            and not entry.diagnostic
            and entry.weight > 0.0
        ):
            active_entries.append((entry, result.subterms[entry.subterm_id]))
    total_mass = math.fsum(entry.weight for entry, _ in active_entries)
    if total_mass <= 0.0:
        raise ValueError("composition requires applicable positive non-diagnostic mass")

    group_rows: Dict[str, List[Tuple[SubtermWeight, float]]] = {}
    for entry, value in active_entries:
        group_rows.setdefault(entry.group, []).append((entry, value))
    groups: List[GroupContribution] = []
    for group in sorted(group_rows):
        rows = group_rows[group]
        mass = math.fsum(entry.weight for entry, _ in rows)
        loss = math.fsum(entry.weight * value for entry, value in rows) / mass
        allowance = (
            profile.group_allowances.get(group)
            if profile.family is CompositionFamily.MEAN_SOFT_BOTTLENECK
            else None
        )
        if profile.family is CompositionFamily.MEAN_SOFT_BOTTLENECK and allowance is None:
            raise ValueError(f"missing explicit soft-bottleneck allowance for {group}")
        groups.append(GroupContribution(group, loss, mass, mass / total_mass, allowance))

    l_mean = math.fsum(group.loss * group.normalized_mass for group in groups)
    if profile.family is CompositionFamily.P_MEAN:
        assert profile.power is not None
        # The shipped default composes over the frozen scored sub-term rows,
        # the finest partition already frozen in the contract inventory, so no
        # free-form label carries weight semantics (V4_SPEC_r4 3.3) and a
        # catastrophic row inside a populated group stays visible at p > 1.
        l_total = math.fsum(
            (entry.weight / total_mass) * value**profile.power for entry, value in active_entries
        ) ** (1.0 / profile.power)
        l_bottleneck = None
    else:
        assert profile.bottleneck_temperature is not None
        assert profile.bottleneck_mix is not None
        allowed_losses = []
        for group in groups:
            assert group.allowance is not None
            allowed_losses.append(group.loss - group.allowance)
        tail = _smooth_max(tuple(allowed_losses), profile.bottleneck_temperature)
        l_bottleneck = _smooth_positive(tail, profile.bottleneck_temperature)
        l_total = (1.0 - profile.bottleneck_mix) * l_mean + profile.bottleneck_mix * l_bottleneck

    subterms: List[SubtermContribution] = []
    for entry in weight_table.entries:
        result = facet_results[entry.facet_id]
        value = result.subterms.get(entry.subterm_id) if result.state is ResultState.VALUE else None
        normalized = (
            entry.weight / total_mass
            if value is not None and not entry.diagnostic and entry.weight > 0.0
            else 0.0
        )
        subterms.append(
            SubtermContribution(
                subterm_id=entry.subterm_id,
                facet_id=entry.facet_id,
                group=entry.group,
                value=value,
                declared_weight=entry.weight,
                normalized_weight=normalized,
                weighted_loss=normalized * value if value is not None else 0.0,
                diagnostic=entry.diagnostic,
                prior_driven=entry.prior_driven,
                na_reason=_dropped_subterm_reason(result, entry.subterm_id),
            )
        )
    return CompositionResult(
        family=profile.family,
        l_mean=l_mean,
        l_bottleneck=l_bottleneck,
        l_total=l_total,
        total_mass=total_mass,
        groups=tuple(groups),
        subterms=tuple(subterms),
    )


def compare_with_event_margin(
    first: CompositionResult,
    second: CompositionResult,
    first_nearby: Tuple[NearbyEvent, ...],
    second_nearby: Tuple[NearbyEvent, ...],
    registry: EventRegistry,
) -> ComparisonResult:
    """Apply the strict-win margin rule to two exact compositions.

    Parameters
    ----------
    first, second : CompositionResult
        Same-graph exact pre-map losses.
    first_nearby, second_nearby : tuple[NearbyEvent, ...]
        Declared manifold occurrences within epsilon of either scored row.
    registry : EventRegistry
        Frozen generated event registry.

    Returns
    -------
    ComparisonResult
        Strict only when the decision margin is greater than the summed nearby
        jump budget.

    Raises
    ------
    ValueError
        If a nearby event is undeclared or one occurrence changes event type.
    """

    declared = {event.event_id: event for event in registry.entries}
    contexts: Dict[str, List[Tuple[str, Mapping[str, float]]]] = {}
    for index, nearby in enumerate((*first_nearby, *second_nearby)):
        if nearby.event_id not in declared:
            raise ValueError(f"undeclared event manifold: {nearby.event_id}")
        # CC-1 sums the bounds of ALL nearby manifolds. Only an explicit
        # shared manifold_id may merge occurrences into one; an id-less
        # occurrence is always its own summand (collapsing repeats of one
        # event type to a max would fail open on the default argument).
        manifold_id = (
            nearby.manifold_id
            if nearby.manifold_id is not None
            else f"{nearby.event_id}#anonymous-{index}"
        )
        contexts.setdefault(manifold_id, []).append((nearby.event_id, nearby.context))
    bounds = []
    event_ids = []
    for manifold_id in sorted(contexts):
        occurrences = contexts[manifold_id]
        occurrence_event_ids = {event_id for event_id, _ in occurrences}
        if len(occurrence_event_ids) != 1:
            raise ValueError(f"manifold occurrence changes event type: {manifold_id}")
        event_id = next(iter(occurrence_event_ids))
        event = declared[event_id]
        # A manifold near both rows is one occurrence. Differing local contexts
        # use the larger graph-local bound so the union margin is conservative.
        bounds.append(max(evaluate_jump_bound(event, context) for _, context in occurrences))
        event_ids.append(event_id)
    event_margin = math.fsum(bounds)
    decision_margin = abs(first.l_total - second.l_total)
    if decision_margin == 0.0:
        verdict = ComparisonVerdict.TIE
    elif decision_margin <= event_margin:
        verdict = ComparisonVerdict.EVENT_MARGIN_LIMITED
    elif first.l_total < second.l_total:
        verdict = ComparisonVerdict.FIRST_WINS
    else:
        verdict = ComparisonVerdict.SECOND_WINS
    return ComparisonResult(
        verdict=verdict,
        decision_margin=decision_margin,
        event_margin=event_margin,
        nearby_event_ids=tuple(sorted(set(event_ids))),
        nearby_manifold_ids=tuple(sorted(contexts)),
    )
