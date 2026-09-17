"""Held-out, JND-band, weight-path, and era diagnostics for P5."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import DefaultDict, Dict, Mapping, Optional, Sequence, Tuple

import torch

from dagua.eval.ruler_v4.fit.bank import SideSwapAuditRow
from dagua.eval.ruler_v4.fit.objective import FitPair, FittingPlan, PairwiseObjective
from dagua.eval.ruler_v4.fit.optimize import FitResult


@dataclass(frozen=True)
class SideSwapAuditResult:
    """Publish FIT-ORD(b)'s audit-only displayed-order diagnostic.

    Parameters
    ----------
    control_legs, resolved_legs, exact_reversals : int
        Input, successful join, and exact exchanged-order counts.
    order_effect : float
        Mean control-minus-base verdict after aligning panel orientation.
    interval : tuple[float, float]
        Normal 95% interval for the mean measured order effect.
    likelihood_row_count : int
        Constitutional containment proof, always zero.
    """

    control_legs: int
    resolved_legs: int
    exact_reversals: int
    order_effect: float
    interval: Tuple[float, float]
    likelihood_row_count: int = 0

    def __post_init__(self) -> None:
        """Validate finite audit publication and zero likelihood consumption.

        Raises
        ------
        ValueError
            If counts disagree, values are nonfinite, or containment is broken.
        """

        if not 0 <= self.exact_reversals <= self.resolved_legs <= self.control_legs:
            raise ValueError("side-swap audit counts must be ordered subsets")
        if not math.isfinite(self.order_effect) or not all(
            math.isfinite(value) for value in self.interval
        ):
            raise ValueError("side-swap audit effect and interval must be finite")
        if self.interval[0] > self.order_effect or self.interval[1] < self.order_effect:
            raise ValueError("side-swap audit interval must bracket the effect")
        if self.likelihood_row_count != 0:
            raise ValueError("side-swap controls may enter no likelihood term")


def side_swap_audit(
    pairs: Sequence[FitPair], controls: Sequence[SideSwapAuditRow]
) -> SideSwapAuditResult:
    """Join exchanged control legs to base replication groups for audit only.

    Parameters
    ----------
    pairs : sequence[FitPair]
        FIT-ORD train rows. They are read for join context but not refitted.
    controls : sequence[SideSwapAuditRow]
        Exchanged-order control legs emitted by the bank loader's separate
        audit channel.

    Returns
    -------
    SideSwapAuditResult
        Join counts and measured order effect with zero likelihood rows.

    Raises
    ------
    ValueError
        If the audit is empty, a leg cannot resolve, or displayed orders are
        not exact reversals of one constant drawing set.
    """

    audit_rows = tuple(controls)
    if not audit_rows:
        raise ValueError("FIT-ORD(b) side-swap audit requires exchanged control legs")
    by_group: DefaultDict[str, list[FitPair]] = defaultdict(list)
    for pair in pairs:
        by_group[pair.replicate_group_id].append(pair)
    effects = []
    for control in audit_rows:
        members = by_group.get(control.replicate_group_id, [])
        if not members:
            raise ValueError(
                f"side-swap control group does not resolve: {control.replicate_group_id}"
            )
        drawing_sets = {frozenset((pair.blind_id_a, pair.blind_id_b)) for pair in members}
        if drawing_sets != {frozenset((control.blind_id_a, control.blind_id_b))}:
            raise ValueError("side-swap audit group crosses drawing sets")
        base_order = (members[0].blind_id_a, members[0].blind_id_b)
        if (control.blind_id_a, control.blind_id_b) != tuple(reversed(base_order)):
            raise ValueError("side-swap control is not the exact reverse of its base order")
        base_verdicts = [
            pair.graded_verdict
            if (pair.blind_id_a, pair.blind_id_b) == base_order
            else -pair.graded_verdict
            for pair in members
        ]
        aligned_control = -control.graded_verdict
        effects.append(aligned_control - math.fsum(base_verdicts) / len(base_verdicts))
    effect = math.fsum(effects) / len(effects)
    if len(effects) == 1:
        margin = 0.0
    else:
        variance = math.fsum((value - effect) ** 2 for value in effects) / (len(effects) - 1)
        margin = 1.96 * math.sqrt(variance / len(effects))
    return SideSwapAuditResult(
        control_legs=len(audit_rows),
        resolved_legs=len(audit_rows),
        exact_reversals=len(audit_rows),
        order_effect=effect,
        interval=(effect - margin, effect + margin),
    )


@dataclass(frozen=True)
class EvaluationMetrics:
    """Summarize three-way preference prediction quality.

    Parameters
    ----------
    count : int
        Evaluated judgments.
    accuracy : float
        A/tie/B argmax accuracy.
    log_loss : float
        Mean negative log likelihood.
    tie_rate_observed, tie_rate_predicted : float
        Observed and mean predicted tie rates.
    """

    count: int
    accuracy: float
    log_loss: float
    tie_rate_observed: float
    tie_rate_predicted: float


@dataclass(frozen=True)
class OrderedCalibrationBin:
    """Publish observed and predicted ordered responses in one score decile.

    Parameters
    ----------
    decile : int
        One-based score-difference decile.
    count : int
        Rows assigned to the bin.
    difference_min, difference_max : float
        Inclusive score-difference range.
    observed, predicted : tuple[float, ...]
        Seven category frequencies in verdict order ``-3`` through ``+3``.
    """

    decile: int
    count: int
    difference_min: float
    difference_max: float
    observed: Tuple[float, ...]
    predicted: Tuple[float, ...]


@dataclass(frozen=True)
class JNDCellCalibration:
    """Publish calibration for one class/size-band JND cell.

    Parameters
    ----------
    primary_class, size_band : str
        Frozen A15 JND-HET cell.
    count : int
        Evaluated judgments.
    jnd : float
        Cell tie half-band.
    observed_tie_rate, predicted_tie_rate : float
        Empirical and model tie rates.
    absolute_error : float
        Absolute calibration gap.
    """

    primary_class: str
    size_band: str
    count: int
    jnd: float
    observed_tie_rate: float
    predicted_tie_rate: float
    absolute_error: float


@dataclass(frozen=True)
class EraRobustness:
    """Keep evaluation results separated by instrument hash and era.

    Parameters
    ----------
    by_stratum : mapping[tuple[str, str], EvaluationMetrics]
        Exact instrument/era metrics; no likelihood pooling.
    cf1_cf4_accuracy_delta : float or None
        CF@4 minus CF@1 accuracy when both eras exist.
    cf1_cf4_log_loss_delta : float or None
        CF@4 minus CF@1 log loss when both eras exist.
    """

    by_stratum: Mapping[Tuple[str, str], EvaluationMetrics]
    cf1_cf4_accuracy_delta: Optional[float]
    cf1_cf4_log_loss_delta: Optional[float]

    def __post_init__(self) -> None:
        """Freeze stratum metrics.

        Raises
        ------
        ValueError
            If no stratum was evaluated.
        """

        values = dict(self.by_stratum)
        if not values:
            raise ValueError("era robustness requires at least one stratum")
        object.__setattr__(self, "by_stratum", MappingProxyType(values))


def evaluate_objective(
    objective: PairwiseObjective, weights: Mapping[str, float]
) -> EvaluationMetrics:
    """Evaluate held-out three-way accuracy and log loss.

    Parameters
    ----------
    objective : PairwiseObjective
        One explicit profile/instrument/era likelihood stratum.
    weights : mapping[str, float]
        Fitted weights keyed by plan identity.

    Returns
    -------
    EvaluationMetrics
        Held-out accuracy, log loss, and tie calibration summary.
    """

    vector = torch.tensor(
        [weights[name] for name in objective.plan.parameter_names], dtype=objective.dtype
    )
    probabilities = objective.directional_probabilities(vector).detach()
    truth = torch.tensor([pair.outcome + 1 for pair in objective.pairs], dtype=torch.int64)
    predictions = probabilities.argmax(dim=1)
    accuracy = float((predictions == truth).to(torch.float64).mean())
    log_loss = float(-torch.log(probabilities.gather(1, truth.unsqueeze(1))).mean())
    observed_tie = float((truth == 1).to(torch.float64).mean())
    predicted_tie = float(probabilities[:, 1].mean())
    return EvaluationMetrics(
        count=len(objective.pairs),
        accuracy=accuracy,
        log_loss=log_loss,
        tie_rate_observed=observed_tie,
        tie_rate_predicted=predicted_tie,
    )


def ordered_response_calibration(
    objective: PairwiseObjective, weights: Mapping[str, float]
) -> Tuple[OrderedCalibrationBin, ...]:
    """Build the frozen observed-vs-predicted table by decile of ``d_j``.

    Parameters
    ----------
    objective : PairwiseObjective
        One ordered-probit likelihood stratum.
    weights : mapping[str, float]
        Fitted outer weights keyed by plan identity.

    Returns
    -------
    tuple[OrderedCalibrationBin, ...]
        Up to ten nonempty rank-balanced score-difference bins.
    """

    vector = torch.tensor(
        [weights[name] for name in objective.plan.parameter_names], dtype=objective.dtype
    )
    differences = objective.score_differences(vector).detach()
    probabilities = objective.outcome_probabilities(vector).detach()
    order = torch.argsort(differences, stable=True)
    count = len(objective.pairs)
    bins: list[OrderedCalibrationBin] = []
    for decile in range(10):
        start = decile * count // 10
        stop = (decile + 1) * count // 10
        if start == stop:
            continue
        indices = order[start:stop]
        observed_counts = [0] * 7
        for index in indices.tolist():
            observed_counts[objective.pairs[index].graded_verdict + 3] += 1
        member_count = stop - start
        bins.append(
            OrderedCalibrationBin(
                decile=decile + 1,
                count=member_count,
                difference_min=float(differences[indices].min()),
                difference_max=float(differences[indices].max()),
                observed=tuple(value / member_count for value in observed_counts),
                predicted=tuple(float(value) for value in probabilities[indices].mean(dim=0)),
            )
        )
    return tuple(bins)


def facet_weight_paths(
    fit_result: FitResult, plan: FittingPlan
) -> Mapping[str, Mapping[str, Tuple[float, ...]]]:
    """Roll fitted parameter paths up by owning facet.

    Parameters
    ----------
    fit_result : FitResult
        Optimizer result containing every weight path.
    plan : FittingPlan
        Parameter-to-facet ownership declaration.

    Returns
    -------
    mapping[str, mapping[str, tuple[float, ...]]]
        Immutable facet/parameter/path mapping.
    """

    by_facet: DefaultDict[str, Dict[str, Tuple[float, ...]]] = defaultdict(dict)
    for parameter in plan.weights:
        for facet_id in parameter.facet_ids:
            by_facet[facet_id][parameter.name] = fit_result.weight_paths[parameter.name]
    return MappingProxyType(
        {facet_id: MappingProxyType(dict(paths)) for facet_id, paths in sorted(by_facet.items())}
    )


def jnd_band_calibration(
    pairs: Sequence[FitPair],
    plan: FittingPlan,
    weights: Mapping[str, float],
    jnd_by_cell: Mapping[Tuple[str, str], float],
) -> Tuple[JNDCellCalibration, ...]:
    """Measure observed versus predicted tie rates for every JND cell.

    Parameters
    ----------
    pairs : sequence[FitPair]
        Evaluation rows from one instrument/era stratum.
    plan : FittingPlan
        Weight plan.
    weights : mapping[str, float]
        Fitted outer weights.
    jnd_by_cell : mapping[tuple[str, str], float]
        Shipped pooled or class-conditional JND bands.

    Returns
    -------
    tuple[JNDCellCalibration, ...]
        Stable cell-level calibration rows.
    """

    grouped: DefaultDict[Tuple[str, str], list[FitPair]] = defaultdict(list)
    for pair in pairs:
        grouped[(pair.primary_class, pair.size_band)].append(pair)
    rows = []
    for cell, members in sorted(grouped.items()):
        if cell not in jnd_by_cell:
            raise ValueError(f"JND missing for calibration cell: {cell}")
        jnd = float(jnd_by_cell[cell])
        adjusted = tuple(replace(member, jnd=jnd) for member in members)
        objective = PairwiseObjective(adjusted, plan)
        metrics = evaluate_objective(objective, weights)
        rows.append(
            JNDCellCalibration(
                primary_class=cell[0],
                size_band=cell[1],
                count=metrics.count,
                jnd=jnd,
                observed_tie_rate=metrics.tie_rate_observed,
                predicted_tie_rate=metrics.tie_rate_predicted,
                absolute_error=abs(metrics.tie_rate_observed - metrics.tie_rate_predicted),
            )
        )
    return tuple(rows)


def era_robustness(
    pairs: Sequence[FitPair], plan: FittingPlan, weights: Mapping[str, float]
) -> EraRobustness:
    """Evaluate CF-era robustness without crossing likelihood strata.

    Parameters
    ----------
    pairs : sequence[FitPair]
        Evaluation rows spanning one or more eras/instruments.
    plan : FittingPlan
        Weight plan shared for comparison.
    weights : mapping[str, float]
        Frozen fitted weights.

    Returns
    -------
    EraRobustness
        Exact-stratum metrics and CF@4-minus-CF@1 deltas when available.
    """

    grouped: DefaultDict[Tuple[str, str], list[FitPair]] = defaultdict(list)
    for pair in pairs:
        grouped[(pair.instrument_hash, pair.era)].append(pair)
    metrics = {
        stratum: evaluate_objective(PairwiseObjective(members, plan), weights)
        for stratum, members in sorted(grouped.items())
    }
    by_era: DefaultDict[str, list[EvaluationMetrics]] = defaultdict(list)
    for (_, era), result in metrics.items():
        by_era[era].append(result)

    def pooled(attribute: str, era: str) -> Optional[float]:
        """Return count-weighted metrics for one era.

        Parameters
        ----------
        attribute : str
            ``EvaluationMetrics`` numeric attribute.
        era : str
            Judge configuration era.

        Returns
        -------
        float or None
            Count-weighted value, or ``None`` without rows.
        """

        values = by_era.get(era, [])
        total = sum(value.count for value in values)
        if total == 0:
            return None
        return sum(getattr(value, attribute) * value.count for value in values) / total

    accuracy_cf1 = pooled("accuracy", "CF@1")
    accuracy_cf4 = pooled("accuracy", "CF@4")
    loss_cf1 = pooled("log_loss", "CF@1")
    loss_cf4 = pooled("log_loss", "CF@4")
    return EraRobustness(
        by_stratum=metrics,
        cf1_cf4_accuracy_delta=(
            None if accuracy_cf1 is None or accuracy_cf4 is None else accuracy_cf4 - accuracy_cf1
        ),
        cf1_cf4_log_loss_delta=(
            None if loss_cf1 is None or loss_cf4 is None else loss_cf4 - loss_cf1
        ),
    )
