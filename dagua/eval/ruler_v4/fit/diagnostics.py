"""Held-out, JND-band, weight-path, and era diagnostics for P5."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from types import MappingProxyType
from typing import DefaultDict, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from dagua.eval.ruler_v4.fit.objective import FitPair, FittingPlan, PairwiseObjective
from dagua.eval.ruler_v4.fit.optimize import FitResult


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


@dataclass(frozen=True)
class JNDFitConfig:
    """Configure replication-only JND-HET optimization.

    Parameters
    ----------
    seed : int, default=20260811
        Deterministic Torch/NumPy/Python seed.
    steps : int, default=1000
        Adam updates.
    learning_rate : float, default=0.03
        Adam learning rate.
    shrinkage : float, default=0.1
        L2 shrinkage for class and band log-JND effects.
    minimum_cell_count : int, default=1
        Minimum replication rows required to estimate a cell.
    initial_jnd : float, default=0.1
        Positive pooled starting band.
    """

    seed: int = 20260811
    steps: int = 1000
    learning_rate: float = 0.03
    shrinkage: float = 0.1
    minimum_cell_count: int = 1
    initial_jnd: float = 0.1

    def __post_init__(self) -> None:
        """Validate JND optimizer settings.

        Raises
        ------
        ValueError
            If a setting is invalid.
        """

        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("JND seed must be a nonnegative integer")
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps <= 0:
            raise ValueError("JND steps must be a positive integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("JND learning rate must be finite and positive")
        if not math.isfinite(self.shrinkage) or self.shrinkage < 0.0:
            raise ValueError("JND shrinkage must be finite and nonnegative")
        if self.minimum_cell_count <= 0:
            raise ValueError("JND cell minimum must be positive")
        if not math.isfinite(self.initial_jnd) or self.initial_jnd <= 0.0:
            raise ValueError("initial JND must be finite and positive")


@dataclass(frozen=True)
class JNDHeterogeneityFit:
    """Publish the replication-only log-additive JND-HET fit.

    Parameters
    ----------
    mu : float
        Pooled log JND.
    tau_class, tau_band : float
        RMS shrinkage spreads of fitted class and band effects.
    class_effects, band_effects : mapping[str, float]
        Centered log-JND effects.
    cell_jnd : mapping[tuple[str, str], float]
        Estimated supported-cell JNDs.
    cell_counts : mapping[tuple[str, str], int]
        Cross-session replication counts.
    unestimated_cells : tuple[tuple[str, str], ...]
        Cells below the preregistered minimum.
    spread : float
        ``p90(cell JND) / p10(cell JND)``.
    loss_path : tuple[float, ...]
        Deterministic optimization path.
    """

    mu: float
    tau_class: float
    tau_band: float
    class_effects: Mapping[str, float]
    band_effects: Mapping[str, float]
    cell_jnd: Mapping[Tuple[str, str], float]
    cell_counts: Mapping[Tuple[str, str], int]
    unestimated_cells: Tuple[Tuple[str, str], ...]
    spread: float
    loss_path: Tuple[float, ...]

    def __post_init__(self) -> None:
        """Freeze JND fit mappings.

        Raises
        ------
        ValueError
            If no supported cell was estimated.
        """

        cells = dict(self.cell_jnd)
        if not cells:
            raise ValueError("JND-HET fit estimated no supported cells")
        object.__setattr__(self, "class_effects", MappingProxyType(dict(self.class_effects)))
        object.__setattr__(self, "band_effects", MappingProxyType(dict(self.band_effects)))
        object.__setattr__(self, "cell_jnd", MappingProxyType(cells))
        object.__setattr__(self, "cell_counts", MappingProxyType(dict(self.cell_counts)))


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
    probabilities = objective.outcome_probabilities(vector).detach()
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
        adjusted = tuple(FitPair(**{**member.__dict__, "jnd": jnd}) for member in members)
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


def fit_jnd_heterogeneity(
    pairs: Sequence[FitPair],
    plan: FittingPlan,
    weights: Mapping[str, float],
    config: JNDFitConfig,
) -> JNDHeterogeneityFit:
    """Fit ``log JND_(c,b) = mu + u_c + v_b`` on replications only.

    Effects are centered after each deterministic Adam update and shrunk
    toward zero. Cells below ``minimum_cell_count`` are listed and excluded,
    never silently imputed. ``tau_class`` and ``tau_band`` are the fitted
    effects' RMS hierarchical spreads; split-half/bootstrap uncertainty is a
    caller-level freeze-fit obligation because its resampling unit is supplied
    by the frozen campaign manifest.

    Parameters
    ----------
    pairs : sequence[FitPair]
        Cross-session replication rows from one instrument/era stratum.
    plan : FittingPlan
        Fitted outer-weight plan.
    weights : mapping[str, float]
        Frozen/fitted outer weights used to compute score differences.
    config : JNDFitConfig
        Deterministic JND-HET settings.

    Returns
    -------
    JNDHeterogeneityFit
        Pooled/effect/spread fit and supported-cell ledger.

    Raises
    ------
    ValueError
        If non-replication rows enter, no cell is supported, or strata cross.
    """

    rows = tuple(pairs)
    if not rows or any(not pair.is_replication for pair in rows):
        raise ValueError("JND-HET accepts cross-session replication rows only")
    by_base_pair: DefaultDict[str, list[FitPair]] = defaultdict(list)
    for pair in rows:
        by_base_pair[pair.base_pair_id].append(pair)
    for base_pair_id, presentations in by_base_pair.items():
        sessions = {pair.session_id for pair in presentations}
        displayed_orders = {(pair.blind_id_a, pair.blind_id_b) for pair in presentations}
        drawing_sets = {frozenset(order) for order in displayed_orders}
        if len(sessions) < 2:
            raise ValueError(f"JND-HET base pair {base_pair_id} lacks cross-session replication")
        if len(drawing_sets) != 1 or len(displayed_orders) < 2:
            raise ValueError(f"JND-HET base pair {base_pair_id} lacks a displayed side swap")
    counts: DefaultDict[Tuple[str, str], int] = defaultdict(int)
    for pair in rows:
        counts[(pair.primary_class, pair.size_band)] += 1
    supported = {cell for cell, count in counts.items() if count >= config.minimum_cell_count}
    unestimated = tuple(sorted(set(counts) - supported))
    selected = tuple(pair for pair in rows if (pair.primary_class, pair.size_band) in supported)
    if not selected:
        raise ValueError("no JND-HET cell meets the replication minimum")
    objective = PairwiseObjective(selected, plan)
    weight_vector = torch.tensor(
        [weights[name] for name in plan.parameter_names], dtype=objective.dtype
    )
    differences = objective.score_differences(weight_vector).detach()
    classes = sorted({pair.primary_class for pair in selected})
    bands = sorted({pair.size_band for pair in selected})
    class_index = {value: index for index, value in enumerate(classes)}
    band_index = {value: index for index, value in enumerate(bands)}
    class_rows = torch.tensor([class_index[pair.primary_class] for pair in selected])
    band_rows = torch.tensor([band_index[pair.size_band] for pair in selected])
    outcomes = torch.tensor([pair.outcome + 1 for pair in selected], dtype=torch.int64)
    lapse = torch.tensor([pair.lapse_rate for pair in selected], dtype=objective.dtype)
    random.seed(config.seed)
    np.random.seed(config.seed % (2**32))
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    mu = torch.tensor(math.log(config.initial_jnd), dtype=objective.dtype, requires_grad=True)
    class_effects = torch.zeros(len(classes), dtype=objective.dtype, requires_grad=True)
    band_effects = torch.zeros(len(bands), dtype=objective.dtype, requires_grad=True)
    optimizer = torch.optim.Adam((mu, class_effects, band_effects), lr=config.learning_rate)
    loss_path = []
    for _ in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        log_jnd = mu + class_effects[class_rows] + band_effects[band_rows]
        jnd = torch.exp(log_jnd)
        lower = torch.sigmoid(-jnd - differences)
        upper = torch.sigmoid(jnd - differences)
        probabilities = torch.stack((lower, upper - lower, 1.0 - upper), dim=1)
        probabilities = (1.0 - lapse.unsqueeze(1)) * probabilities + lapse.unsqueeze(1) / 3.0
        selected_probability = probabilities.gather(1, outcomes.unsqueeze(1)).squeeze(1)
        nll = -torch.log(torch.clamp(selected_probability, min=1.0e-12)).mean()
        penalty = (
            config.shrinkage
            * (torch.square(class_effects).sum() + torch.square(band_effects).sum())
            / len(selected)
        )
        loss = nll + penalty
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("JND-HET objective became nonfinite")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            class_mean = class_effects.mean()
            band_mean = band_effects.mean()
            class_effects.sub_(class_mean)
            band_effects.sub_(band_mean)
            mu.add_(class_mean + band_mean)
        loss_path.append(float(loss.detach()))
    class_values = {
        value: float(class_effects[index].detach()) for value, index in class_index.items()
    }
    band_values = {
        value: float(band_effects[index].detach()) for value, index in band_index.items()
    }
    mu_value = float(mu.detach())
    cell_jnd = {
        cell: math.exp(mu_value + class_values[cell[0]] + band_values[cell[1]])
        for cell in sorted(supported)
    }
    values = np.asarray(tuple(cell_jnd.values()), dtype=np.float64)
    p10, p90 = np.percentile(values, (10.0, 90.0))
    tau_class = float(np.sqrt(np.mean(np.square(tuple(class_values.values())))))
    tau_band = float(np.sqrt(np.mean(np.square(tuple(band_values.values())))))
    return JNDHeterogeneityFit(
        mu=mu_value,
        tau_class=tau_class,
        tau_band=tau_band,
        class_effects=class_values,
        band_effects=band_values,
        cell_jnd=cell_jnd,
        cell_counts=dict(counts),
        unestimated_cells=unestimated,
        spread=float(p90 / p10),
        loss_path=tuple(loss_path),
    )
