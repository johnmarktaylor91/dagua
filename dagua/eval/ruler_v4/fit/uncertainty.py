"""W-13 hierarchical JND estimation and uncertainty publications."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import DefaultDict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from dagua.eval.ruler_v4.fit.access import H_JND_LEDGER_KEY, AccessLedger
from dagua.eval.ruler_v4.fit.objective import FitPair, FittingPlan, PairwiseObjective

_MINIMUM_CELL_COUNT = 25
_Q_BAND = 67
_BOOTSTRAP_REPLICATES = 2000
_FROZEN_SEED = 20260811
_PROFILE_DROP = 1.920729410347062
_LOG_JND_BOUNDS = (math.log(1.0e-6), math.log(100.0))
_LOG_TAU_BOUNDS = (math.log(1.0e-6), math.log(10.0))
_A15_HALF_SALT = "v4-split-2|within"


@dataclass(frozen=True)
class JNDFitConfig:
    """Provide realized quota and ordering inputs to frozen W-13 constants.

    Parameters
    ----------
    role_hash : str
        Verified frozen A15 partition identity for the one-shot branch event.
    top_composite_pair_counts : mapping[str, int]
        Realized distinct top-composite-band base-pair counts by size band.
    rotation_envelopes : mapping[str, float]
        Positive measured rotation envelopes by primary class.
    minimum_cell_count : int
        Frozen distinct-pair support minimum, always 25.
    q_band : int
        Frozen per-band top-composite quota, always 67.
    bootstrap_replicates : int
        Frozen resample count, always 2,000.
    seed : int
        Frozen deterministic campaign seed.
    """

    role_hash: str
    top_composite_pair_counts: Mapping[str, int]
    rotation_envelopes: Mapping[str, float]
    minimum_cell_count: int = field(default=_MINIMUM_CELL_COUNT, init=False)
    q_band: int = field(default=_Q_BAND, init=False)
    bootstrap_replicates: int = field(default=_BOOTSTRAP_REPLICATES, init=False)
    seed: int = field(default=_FROZEN_SEED, init=False)

    def __post_init__(self) -> None:
        """Freeze realized inputs and validate their finite domains.

        Raises
        ------
        ValueError
            If a count or rotation envelope is invalid.
        """

        counts = dict(self.top_composite_pair_counts)
        envelopes = {key: float(value) for key, value in self.rotation_envelopes.items()}
        if not self.role_hash:
            raise ValueError("JND fit requires the frozen A15 role hash")
        if any(
            not key or isinstance(value, bool) or not isinstance(value, int) or value < 0
            for key, value in counts.items()
        ):
            raise ValueError("top-composite counts must be named nonnegative integers")
        if any(
            not key or not math.isfinite(value) or value <= 0.0 for key, value in envelopes.items()
        ):
            raise ValueError("rotation envelopes must be named, finite, and positive")
        object.__setattr__(self, "top_composite_pair_counts", MappingProxyType(counts))
        object.__setattr__(self, "rotation_envelopes", MappingProxyType(envelopes))


@dataclass(frozen=True)
class VarianceComponentCI:
    """Publish a variance-component interval on fitted and consumed scales.

    Parameters
    ----------
    log_scale : tuple[float, float]
        95% profile-likelihood interval for ``log(tau)``.
    ratio_scale : tuple[float, float]
        Exponentiated interval consumed as multiplicative JND ratios.
    """

    log_scale: Tuple[float, float]
    ratio_scale: Tuple[float, float]


@dataclass(frozen=True)
class KJNDDisclosure:
    """Name one cell exceeding three times the pooled JND.

    Parameters
    ----------
    cell : tuple[str, str]
        Primary-class and size-band identity.
    replication_count : int
        Distinct qualifying base-pair count.
    interval : tuple[float, float]
        Published cell JND interval.
    """

    cell: Tuple[str, str]
    replication_count: int
    interval: Tuple[float, float]


@dataclass(frozen=True)
class SplitHalfStability:
    """Publish graph-disjoint half estimates and frozen components.

    Parameters
    ----------
    half_one, half_two : mapping[str, float]
        Half-sample ``tau_class`` and ``tau_band`` estimates.
    frozen_components : tuple[str, ...]
        Components set to their null prior after the stability gate.
    """

    half_one: Mapping[str, float]
    half_two: Mapping[str, float]
    frozen_components: Tuple[str, ...]

    def __post_init__(self) -> None:
        """Freeze half-estimate mappings."""

        object.__setattr__(self, "half_one", MappingProxyType(dict(self.half_one)))
        object.__setattr__(self, "half_two", MappingProxyType(dict(self.half_two)))


@dataclass(frozen=True)
class BootstrapDropRate:
    """Publish cell-support drop rates for both bootstrap unit systems.

    Parameters
    ----------
    graph_clusters, generator_families : float
        Fraction of originally supported cell slots dropped across replicates.
    """

    graph_clusters: float
    generator_families: float


@dataclass(frozen=True)
class JNDHeterogeneityFit:
    """Publish the complete W-13 estimator, uncertainty, and guard record.

    Parameters
    ----------
    mu : float
        Fitted pooled log JND.
    pooled_jnd_ci : tuple[float, float]
        Graph-cluster bootstrap interval consumed by TEST H-JND's ``R_pool``.
    pooled_jnd_profile_ci : tuple[float, float]
        Profile-likelihood interval transformed to the JND scale.
    tau_class, tau_band : float
        Fitted variance components after mandatory shrink responses.
    tau_class_ci, tau_band_ci : VarianceComponentCI
        Profile-likelihood intervals from the estimated model.
    class_effects, band_effects : mapping[str, float]
        Conditional log-JND effects.
    cell_jnd, cell_jnd_ci : mapping[tuple[str, str], object]
        Supported-cell estimates and 95% profile intervals.
    cell_counts : mapping[tuple[str, str], int]
        Distinct qualifying replicated base-pair counts.
    unestimated_cells : tuple[tuple[str, str], ...]
        Cells below the frozen support minimum.
    spread : float
        ``JND_p90 / JND_p10`` over supported cells.
    spread_ci_graph_clusters, spread_ci_generator_families : tuple[float, float]
        Separate 2,000-resample percentile intervals.
    bootstrap_drop_rate : BootstrapDropRate
        Support loss under each unit system.
    effective_dof : float
        Pre-response C-06 smoother trace reported beside the ledgered +2.
    shrink_actions : tuple[str, ...]
        Components frozen at zero by stability or C-06.
    k_jnd_disclosures : tuple[KJNDDisclosure, ...]
        Cells exceeding three times the pooled JND.
    split_half : SplitHalfStability
        Graph-disjoint half-sample audit.
    uncalibrated_classes : tuple[str, ...]
        Classes failing the rotation-envelope ordering invariant.
    tie_rates_by_class : mapping[str, tuple[float, float]]
        Observed tie rates under pooled and cell bands.
    loss_path : tuple[float, ...]
        Deterministic marginal-likelihood optimization trajectory.
    role_hash : str
        Frozen A15 partition identity binding TEST H-JND.
    replication_row_ids : tuple[str, ...]
        Complete stable replication-line identities bound by the fit.
    """

    mu: float
    pooled_jnd_ci: Tuple[float, float]
    pooled_jnd_profile_ci: Tuple[float, float]
    tau_class: float
    tau_band: float
    tau_class_ci: VarianceComponentCI
    tau_band_ci: VarianceComponentCI
    class_effects: Mapping[str, float]
    band_effects: Mapping[str, float]
    cell_jnd: Mapping[Tuple[str, str], float]
    cell_jnd_ci: Mapping[Tuple[str, str], Tuple[float, float]]
    cell_counts: Mapping[Tuple[str, str], int]
    unestimated_cells: Tuple[Tuple[str, str], ...]
    spread: float
    spread_ci_graph_clusters: Tuple[float, float]
    spread_ci_generator_families: Tuple[float, float]
    bootstrap_drop_rate: BootstrapDropRate
    effective_dof: float
    shrink_actions: Tuple[str, ...]
    k_jnd_disclosures: Tuple[KJNDDisclosure, ...]
    split_half: SplitHalfStability
    uncalibrated_classes: Tuple[str, ...]
    tie_rates_by_class: Mapping[str, Tuple[float, float]]
    loss_path: Tuple[float, ...]
    role_hash: str
    replication_row_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        """Freeze publication mappings and require at least one supported cell.

        Raises
        ------
        ValueError
            If the mandatory supported-cell publication is empty.
        """

        cells = dict(self.cell_jnd)
        if not cells or set(cells) != set(self.cell_jnd_ci):
            raise ValueError("JND-HET requires estimates and intervals for supported cells")
        for name in (
            "class_effects",
            "band_effects",
            "cell_jnd",
            "cell_jnd_ci",
            "cell_counts",
            "tie_rates_by_class",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


@dataclass(frozen=True)
class HJNDBranchResult:
    """Publish the once-only TEST H-JND branch decision.

    Parameters
    ----------
    shipped_band : str
        ``class-conditional`` or ``pooled``.
    spread_lower : float
        Graph-cluster bootstrap lower bound used by the rule.
    pooled_ratio : float
        ``JND_pool_hi / JND_pool_lo`` from the same fit.
    row_set_digest : str
        Content digest reserved by LOOK-LEDGER.
    """

    shipped_band: str
    spread_lower: float
    pooled_ratio: float
    row_set_digest: str


@dataclass(frozen=True)
class _CellObservation:
    """Hold one cell's local Laplace observation on the log-JND scale."""

    estimate: float
    variance: float


@dataclass(frozen=True)
class _MetaFit:
    """Hold one Gaussian marginal-likelihood variance-component fit."""

    mu: float
    tau_class: float
    tau_band: float
    class_effects: Mapping[str, float]
    band_effects: Mapping[str, float]
    cell_logs: Mapping[Tuple[str, str], float]
    loss: float
    loss_path: Tuple[float, ...]
    effective_dof_class: float
    effective_dof_band: float
    parameter_vector: np.ndarray
    active_components: Tuple[str, ...]


def _normal_cdf(values: np.ndarray) -> np.ndarray:
    """Evaluate the standard normal CDF.

    Parameters
    ----------
    values : numpy.ndarray
        Finite standardized values.

    Returns
    -------
    numpy.ndarray
        Elementwise standard normal probabilities.
    """

    from scipy.special import ndtr

    return ndtr(values)


def _ordered_nll(
    log_jnd: float,
    differences: np.ndarray,
    verdicts: np.ndarray,
    lapses: np.ndarray,
    multiplicities: Optional[np.ndarray] = None,
) -> float:
    """Evaluate summed seven-category probit NLL for one log JND.

    Parameters
    ----------
    log_jnd : float
        Candidate log tie half-band.
    differences, verdicts, lapses : numpy.ndarray
        Score differences, graded responses, and lapse rates.
    multiplicities : numpy.ndarray or None
        Optional bootstrap observation weights.

    Returns
    -------
    float
        Summed negative log likelihood.
    """

    jnd = math.exp(log_jnd)
    cutpoints = jnd * np.asarray((-3.0, -2.0, -1.0, 1.0, 2.0, 3.0))
    cdf = _normal_cdf(cutpoints[None, :] - differences[:, None])
    boundaries = np.concatenate(
        (np.zeros((len(differences), 1)), cdf, np.ones((len(differences), 1))), axis=1
    )
    probabilities = np.diff(boundaries, axis=1)
    probabilities = (1.0 - lapses[:, None]) * probabilities + lapses[:, None] / 7.0
    selected = np.clip(probabilities[np.arange(len(verdicts)), verdicts + 3], 1.0e-12, 1.0)
    weights = np.ones_like(selected) if multiplicities is None else multiplicities
    return float(-np.sum(weights * np.log(selected)))


def _fit_cell_observation(
    differences: np.ndarray,
    verdicts: np.ndarray,
    lapses: np.ndarray,
    multiplicities: Optional[np.ndarray] = None,
) -> _CellObservation:
    """Laplace-approximate one cell's ordered likelihood on the log scale.

    Parameters
    ----------
    differences, verdicts, lapses : numpy.ndarray
        Cell-level ordered-probit inputs.
    multiplicities : numpy.ndarray or None
        Optional bootstrap weights.

    Returns
    -------
    _CellObservation
        Local log-JND mode and inverse-curvature variance.
    """

    from scipy.optimize import minimize_scalar

    def objective(value: float) -> float:
        """Evaluate the local scalar objective.

        Parameters
        ----------
        value : float
            Candidate log JND.

        Returns
        -------
        float
            Summed ordered-probit NLL.
        """

        return _ordered_nll(float(value), differences, verdicts, lapses, multiplicities)

    result = minimize_scalar(objective, bounds=_LOG_JND_BOUNDS, method="bounded")
    if not result.success:
        raise RuntimeError(f"cell JND optimization failed: {result.message}")
    estimate = float(result.x)
    step = 1.0e-4
    curvature = objective(estimate + step) - 2.0 * objective(estimate) + objective(estimate - step)
    curvature /= step**2
    variance = 1.0 / max(curvature, 1.0e-8)
    return _CellObservation(estimate=estimate, variance=variance)


def _meta_fit(
    observations: Mapping[Tuple[str, str], _CellObservation],
    eligible_bands: frozenset[str],
    frozen_components: frozenset[str] = frozenset(),
) -> _MetaFit:
    """Fit integrated class/band variance components by marginal likelihood.

    Parameters
    ----------
    observations : mapping[tuple[str, str], _CellObservation]
        Cell Laplace observations.
    eligible_bands : frozenset[str]
        Bands meeting the frozen top-composite quota.
    frozen_components : frozenset[str], optional
        Variance components fixed at their null prior.

    Returns
    -------
    _MetaFit
        Marginal MLE, conditional effects, and smoother trace.
    """

    from scipy.optimize import minimize

    cells = tuple(sorted(observations))
    classes = tuple(sorted({cell[0] for cell in cells}))
    bands = tuple(sorted({cell[1] for cell in cells if cell[1] in eligible_bands}))
    values = np.asarray([observations[cell].estimate for cell in cells])
    variances = np.asarray([observations[cell].variance for cell in cells])
    class_design = np.asarray([[float(cell[0] == value) for value in classes] for cell in cells])
    band_design = np.asarray([[float(cell[1] == value) for value in bands] for cell in cells])
    active = tuple(
        component
        for component, present in (("tau_class", bool(classes)), ("tau_band", bool(bands)))
        if present and component not in frozen_components
    )
    loss_path: list[float] = []

    def unpack(parameters: np.ndarray) -> Tuple[float, float, float]:
        """Decode one marginal parameter vector.

        Parameters
        ----------
        parameters : numpy.ndarray
            ``mu`` followed by active log variance components.

        Returns
        -------
        tuple[float, float, float]
            ``mu``, ``tau_class``, and ``tau_band``.
        """

        cursor = 1
        taus = {"tau_class": 0.0, "tau_band": 0.0}
        for component in active:
            taus[component] = math.exp(float(parameters[cursor]))
            cursor += 1
        return float(parameters[0]), taus["tau_class"], taus["tau_band"]

    def evaluate(parameters: np.ndarray, record: bool = True) -> float:
        """Evaluate the integrated Gaussian negative log likelihood.

        Parameters
        ----------
        parameters : numpy.ndarray
            Marginal parameter vector.
        record : bool, default=True
            Whether to append to the optimization trajectory.

        Returns
        -------
        float
            Negative marginal log likelihood.
        """

        mu, tau_class, tau_band = unpack(parameters)
        covariance = np.diag(variances)
        covariance += tau_class**2 * (class_design @ class_design.T)
        if bands:
            covariance += tau_band**2 * (band_design @ band_design.T)
        sign, log_determinant = np.linalg.slogdet(covariance)
        if sign <= 0:
            return math.inf
        residual = values - mu
        loss = 0.5 * (
            log_determinant
            + float(residual @ np.linalg.solve(covariance, residual))
            + len(values) * math.log(2.0 * math.pi)
        )
        if record:
            loss_path.append(loss)
        return loss

    initial = np.asarray(
        [float(np.average(values, weights=1.0 / variances))] + [math.log(0.1)] * len(active)
    )
    bounds = [(_LOG_JND_BOUNDS[0], _LOG_JND_BOUNDS[1])] + [_LOG_TAU_BOUNDS] * len(active)
    result = minimize(evaluate, initial, method="L-BFGS-B", bounds=bounds)
    if not result.success:
        raise RuntimeError(f"JND marginal-likelihood fit failed: {result.message}")
    mu, tau_class, tau_band = unpack(result.x)
    covariance = np.diag(variances)
    covariance += tau_class**2 * (class_design @ class_design.T)
    if bands:
        covariance += tau_band**2 * (band_design @ band_design.T)
    precision_residual = np.linalg.solve(covariance, values - mu)
    class_effect_values = tau_class**2 * class_design.T @ precision_residual
    band_effect_values = tau_band**2 * band_design.T @ precision_residual
    class_effects = dict(zip(classes, class_effect_values))
    band_effects = dict(zip(bands, band_effect_values))
    cell_logs = {
        cell: mu + class_effects.get(cell[0], 0.0) + band_effects.get(cell[1], 0.0)
        for cell in cells
    }
    inverse_covariance = np.linalg.inv(covariance)
    effective_class = float(
        np.trace(tau_class**2 * class_design.T @ inverse_covariance @ class_design)
    )
    effective_band = float(np.trace(tau_band**2 * band_design.T @ inverse_covariance @ band_design))
    return _MetaFit(
        mu=mu,
        tau_class=tau_class,
        tau_band=tau_band,
        class_effects=MappingProxyType(class_effects),
        band_effects=MappingProxyType(band_effects),
        cell_logs=MappingProxyType(cell_logs),
        loss=float(result.fun),
        loss_path=tuple(loss_path),
        effective_dof_class=effective_class,
        effective_dof_band=effective_band,
        parameter_vector=np.asarray(result.x),
        active_components=active,
    )


def _validate_replication_rows(rows: Tuple[FitPair, ...]) -> Mapping[Tuple[str, str], int]:
    """Validate anti-farming provenance and count distinct qualifying pairs.

    Parameters
    ----------
    rows : tuple[FitPair, ...]
        Candidate train-role replication presentations.

    Returns
    -------
    mapping[tuple[str, str], int]
        Distinct cross-session base-pair counts by cell.

    Raises
    ------
    ValueError
        If a row is non-replication, non-train, cross-stratum, or lacks a true
        cross-session displayed side swap.
    """

    if not rows or any(not pair.is_replication for pair in rows):
        raise ValueError("JND-HET accepts cross-session replication rows only")
    if any(pair.purpose.value != "fit" for pair in rows):
        raise ValueError("JND-HET accepts train-role rows only")
    strata = {(pair.instrument_hash, pair.era, pair.observation_profile) for pair in rows}
    if len(strata) != 1:
        raise ValueError("JND-HET cannot cross instrument, era, or profile strata")
    by_base_pair: DefaultDict[str, list[FitPair]] = defaultdict(list)
    for pair in rows:
        by_base_pair[pair.base_pair_id].append(pair)
    counts: DefaultDict[Tuple[str, str], int] = defaultdict(int)
    for base_pair_id, presentations in by_base_pair.items():
        sessions = {pair.session_id for pair in presentations}
        displayed_orders = {(pair.blind_id_a, pair.blind_id_b) for pair in presentations}
        drawing_sets = {frozenset(order) for order in displayed_orders}
        cells = {(pair.primary_class, pair.size_band) for pair in presentations}
        if len(sessions) < 2:
            raise ValueError(f"JND-HET base pair {base_pair_id} lacks cross-session replication")
        if len(drawing_sets) != 1 or len(displayed_orders) < 2:
            raise ValueError(f"JND-HET base pair {base_pair_id} lacks a displayed side swap")
        if len(cells) != 1:
            raise ValueError(f"JND-HET base pair {base_pair_id} crosses cells")
        counts[next(iter(cells))] += 1
    return MappingProxyType(dict(counts))


def _cell_observations(
    rows: Tuple[FitPair, ...], differences: np.ndarray
) -> Mapping[Tuple[str, str], _CellObservation]:
    """Fit local log-JND observations for all realized cells.

    Parameters
    ----------
    rows : tuple[FitPair, ...]
        Replication presentations.
    differences : numpy.ndarray
        Score differences aligned with ``rows``.

    Returns
    -------
    mapping[tuple[str, str], _CellObservation]
        Local Laplace observations.
    """

    indices: DefaultDict[Tuple[str, str], list[int]] = defaultdict(list)
    for index, pair in enumerate(rows):
        indices[(pair.primary_class, pair.size_band)].append(index)
    return MappingProxyType(
        {
            cell: _fit_cell_observation(
                differences[members],
                np.asarray([rows[index].graded_verdict for index in members], dtype=np.int64),
                np.asarray([rows[index].lapse_rate for index in members]),
            )
            for cell, members in sorted(indices.items())
        }
    )


def _meta_profile_intervals(
    observations: Mapping[Tuple[str, str], _CellObservation],
    eligible_bands: frozenset[str],
    fitted: _MetaFit,
) -> Tuple[Tuple[float, float], VarianceComponentCI, VarianceComponentCI]:
    """Profile ``mu`` and both variance components on the log scale.

    Parameters
    ----------
    observations : mapping[tuple[str, str], _CellObservation]
        Cell Laplace observations.
    eligible_bands : frozenset[str]
        Bands entering the random band component.
    fitted : _MetaFit
        Unconstrained marginal optimum.

    Returns
    -------
    tuple[tuple[float, float], VarianceComponentCI, VarianceComponentCI]
        Pooled log-JND and component intervals.

    Notes
    -----
    The small Gaussian mixed-model block is profiled by fixing one coordinate
    and re-optimizing every remaining marginal parameter. This is deliberately
    separate from a Hessian/Wald interval.
    """

    from scipy.optimize import brentq, minimize

    cells = tuple(sorted(observations))
    classes = tuple(sorted({cell[0] for cell in cells}))
    bands = tuple(sorted({cell[1] for cell in cells if cell[1] in eligible_bands}))
    values = np.asarray([observations[cell].estimate for cell in cells])
    variances = np.asarray([observations[cell].variance for cell in cells])
    class_design = np.asarray([[float(cell[0] == value) for value in classes] for cell in cells])
    band_design = np.asarray([[float(cell[1] == value) for value in bands] for cell in cells])
    names = ("mu",) + fitted.active_components
    bounds = [_LOG_JND_BOUNDS] + [_LOG_TAU_BOUNDS] * len(fitted.active_components)
    optimum = fitted.parameter_vector
    target = fitted.loss + _PROFILE_DROP

    def evaluate(parameters: np.ndarray) -> float:
        """Evaluate the marginal model at one complete parameter vector.

        Parameters
        ----------
        parameters : numpy.ndarray
            ``mu`` plus active log components.

        Returns
        -------
        float
            Negative marginal log likelihood.
        """

        tau = {"tau_class": 0.0, "tau_band": 0.0}
        for position, component in enumerate(fitted.active_components, start=1):
            tau[component] = math.exp(float(parameters[position]))
        covariance = np.diag(variances)
        covariance += tau["tau_class"] ** 2 * (class_design @ class_design.T)
        if bands:
            covariance += tau["tau_band"] ** 2 * (band_design @ band_design.T)
        residual = values - float(parameters[0])
        sign, log_determinant = np.linalg.slogdet(covariance)
        if sign <= 0:
            return math.inf
        return 0.5 * (
            log_determinant
            + float(residual @ np.linalg.solve(covariance, residual))
            + len(values) * math.log(2.0 * math.pi)
        )

    def profile(index: int, fixed: float) -> float:
        """Profile one fixed marginal coordinate.

        Parameters
        ----------
        index : int
            Fixed coordinate.
        fixed : float
            Fixed log-scale value.

        Returns
        -------
        float
            Profile loss minus the likelihood-ratio target.
        """

        free = [position for position in range(len(names)) if position != index]
        if not free:
            return evaluate(np.asarray((fixed,))) - target

        def reduced(values_free: np.ndarray) -> float:
            """Evaluate non-fixed profile coordinates.

            Parameters
            ----------
            values_free : numpy.ndarray
                Free-coordinate candidate.

            Returns
            -------
            float
                Complete marginal NLL.
            """

            candidate = optimum.copy()
            candidate[index] = fixed
            candidate[free] = values_free
            return evaluate(candidate)

        result = minimize(
            reduced,
            optimum[free],
            method="L-BFGS-B",
            bounds=[bounds[position] for position in free],
        )
        if not result.success:
            raise RuntimeError(f"JND profile optimization failed: {result.message}")
        return float(result.fun) - target

    def interval(index: int) -> Tuple[float, float]:
        """Find a two-sided profile interval for one coordinate.

        Parameters
        ----------
        index : int
            Profiled coordinate.

        Returns
        -------
        tuple[float, float]
            Lower and upper log-scale endpoints.
        """

        estimate = float(optimum[index])
        lower_bound, upper_bound = bounds[index]
        lower = (
            lower_bound
            if profile(index, lower_bound) <= 0.0
            else float(brentq(lambda value: profile(index, value), lower_bound, estimate))
        )
        upper = (
            upper_bound
            if profile(index, upper_bound) <= 0.0
            else float(brentq(lambda value: profile(index, value), estimate, upper_bound))
        )
        return lower, upper

    mu_interval = interval(0)
    component_intervals = {}
    for component in ("tau_class", "tau_band"):
        if component not in names:
            log_interval = (_LOG_TAU_BOUNDS[0], _LOG_TAU_BOUNDS[0])
        else:
            log_interval = interval(names.index(component))
        component_intervals[component] = VarianceComponentCI(
            log_scale=log_interval,
            ratio_scale=(math.exp(log_interval[0]), math.exp(log_interval[1])),
        )
    return mu_interval, component_intervals["tau_class"], component_intervals["tau_band"]


def _conditional_cell_interval(
    rows: Tuple[FitPair, ...],
    differences: np.ndarray,
    cell: Tuple[str, str],
    center: float,
    prior_variance: float,
) -> Tuple[float, float]:
    """Profile one supported cell's conditional log-JND likelihood.

    Parameters
    ----------
    rows : tuple[FitPair, ...]
        Complete replication presentations.
    differences : numpy.ndarray
        Score differences aligned with rows.
    cell : tuple[str, str]
        Cell being profiled.
    center : float
        Hierarchical conditional log-JND center.
    prior_variance : float
        Estimated random-effect variance supporting the cell.

    Returns
    -------
    tuple[float, float]
        Profile interval transformed to the JND scale.
    """

    from scipy.optimize import brentq, minimize_scalar

    members = [
        index for index, pair in enumerate(rows) if (pair.primary_class, pair.size_band) == cell
    ]
    verdicts = np.asarray([rows[index].graded_verdict for index in members], dtype=np.int64)
    lapses = np.asarray([rows[index].lapse_rate for index in members])
    cell_differences = differences[members]

    def objective(value: float) -> float:
        """Evaluate the conditional profile objective.

        Parameters
        ----------
        value : float
            Candidate log JND.

        Returns
        -------
        float
            Ordered NLL plus estimated random-effect penalty.
        """

        penalty = 0.0 if prior_variance <= 0.0 else 0.5 * (value - center) ** 2 / prior_variance
        return _ordered_nll(value, cell_differences, verdicts, lapses) + penalty

    optimum = minimize_scalar(objective, bounds=_LOG_JND_BOUNDS, method="bounded")
    target = float(optimum.fun) + _PROFILE_DROP

    def root(value: float) -> float:
        """Evaluate the conditional likelihood-ratio root.

        Parameters
        ----------
        value : float
            Candidate log JND.

        Returns
        -------
        float
            Profile objective minus its 95% target.
        """

        return objective(float(value)) - target

    lower = (
        _LOG_JND_BOUNDS[0]
        if root(_LOG_JND_BOUNDS[0]) <= 0.0
        else float(brentq(root, _LOG_JND_BOUNDS[0], float(optimum.x)))
    )
    upper = (
        _LOG_JND_BOUNDS[1]
        if root(_LOG_JND_BOUNDS[1]) <= 0.0
        else float(brentq(root, float(optimum.x), _LOG_JND_BOUNDS[1]))
    )
    return math.exp(lower), math.exp(upper)


def _spread(values: Sequence[float]) -> float:
    """Compute the W-13 p90/p10 spread.

    Parameters
    ----------
    values : sequence[float]
        Positive supported-cell JND estimates.

    Returns
    -------
    float
        Percentile ratio.
    """

    array = np.asarray(values, dtype=np.float64)
    return float(np.percentile(array, 90) / np.percentile(array, 10))


def _bootstrap_spread(
    rows: Tuple[FitPair, ...],
    differences: np.ndarray,
    unit: str,
    config: JNDFitConfig,
    supported_cells: frozenset[Tuple[str, str]],
    fitted_logs: Mapping[Tuple[str, str], float],
    tau_variance: float,
) -> Tuple[Tuple[float, float], float, Tuple[float, float]]:
    """Bootstrap W-13 spread under one frozen resampling unit.

    Parameters
    ----------
    rows : tuple[FitPair, ...]
        Complete replication presentations.
    differences : numpy.ndarray
        Score differences aligned with rows.
    unit : str
        ``graph_hash`` or ``generator_family``.
    config : JNDFitConfig
        Frozen bootstrap settings.
    supported_cells : frozenset[tuple[str, str]]
        Cells supported in the original fit.
    fitted_logs : mapping[tuple[str, str], float]
        Original hierarchical cell centers.
    tau_variance : float
        Combined fitted random-effect variance.

    Returns
    -------
    tuple[tuple[float, float], float, tuple[float, float]]
        Spread interval, cell-slot drop rate, and pooled-JND interval from the
        same resamples.
    """

    units = tuple(sorted({str(getattr(pair, unit)) for pair in rows}))
    by_unit: DefaultDict[str, list[int]] = defaultdict(list)
    for index, pair in enumerate(rows):
        by_unit[str(getattr(pair, unit))].append(index)
    seed_offset = 0 if unit == "graph_hash" else 1
    rng = np.random.default_rng(config.seed + seed_offset)
    spreads = []
    pooled_jnds = []
    dropped = 0
    total_slots = config.bootstrap_replicates * len(supported_cells)
    for _ in range(config.bootstrap_replicates):
        sampled = rng.choice(units, size=len(units), replace=True)
        multiplicity_by_row = np.zeros(len(rows), dtype=np.float64)
        for sampled_unit in sampled:
            multiplicity_by_row[by_unit[str(sampled_unit)]] += 1.0
        active_rows = np.flatnonzero(multiplicity_by_row > 0.0)
        pooled_observation = _fit_cell_observation(
            differences[active_rows],
            np.asarray([rows[int(index)].graded_verdict for index in active_rows]),
            np.asarray([rows[int(index)].lapse_rate for index in active_rows]),
            multiplicity_by_row[active_rows],
        )
        pooled_jnds.append(math.exp(pooled_observation.estimate))
        cell_values = []
        for cell in sorted(supported_cells):
            members = np.asarray(
                [
                    index
                    for index, pair in enumerate(rows)
                    if (pair.primary_class, pair.size_band) == cell
                    and multiplicity_by_row[index] > 0.0
                ],
                dtype=np.int64,
            )
            base_pair_multiplicity: DefaultDict[str, float] = defaultdict(float)
            for index in members:
                base_pair_multiplicity[rows[int(index)].base_pair_id] = max(
                    base_pair_multiplicity[rows[int(index)].base_pair_id],
                    multiplicity_by_row[int(index)],
                )
            replicated_count = int(sum(base_pair_multiplicity.values()))
            if replicated_count < config.minimum_cell_count:
                dropped += 1
                continue
            observation = _fit_cell_observation(
                differences[members],
                np.asarray([rows[int(index)].graded_verdict for index in members]),
                np.asarray([rows[int(index)].lapse_rate for index in members]),
                multiplicity_by_row[members],
            )
            shrinkage = tau_variance / (tau_variance + observation.variance)
            log_value = fitted_logs[cell] + shrinkage * (observation.estimate - fitted_logs[cell])
            cell_values.append(math.exp(log_value))
        if cell_values:
            spreads.append(_spread(cell_values))
    if not spreads:
        raise ValueError(f"all {unit} bootstrap replicates lost cell support")
    interval = tuple(float(value) for value in np.percentile(spreads, (2.5, 97.5)))
    pooled_interval = tuple(float(value) for value in np.percentile(pooled_jnds, (2.5, 97.5)))
    drop_rate = 0.0 if total_slots == 0 else dropped / total_slots
    return (
        (interval[0], interval[1]),
        drop_rate,
        (
            pooled_interval[0],
            pooled_interval[1],
        ),
    )


def _synthetic_half_key(graph_hash: str) -> int:
    """Assign a synthetic fixture graph to a deterministic salted half.

    Parameters
    ----------
    graph_hash : str
        Frozen graph identity.

    Returns
    -------
    int
        Zero or one from the low bit of the salted SHA-256 digest. This helper
        is fixture-only; ADDENDUM-27 does not freeze a real graph-half mapping.
    """

    digest = hashlib.sha256(f"{_A15_HALF_SALT}|{graph_hash}".encode("utf-8")).digest()
    return digest[-1] & 1


def _split_half_fit(
    rows: Tuple[FitPair, ...],
    differences: np.ndarray,
    eligible_bands: frozenset[str],
) -> Tuple[_MetaFit, _MetaFit]:
    """Fit graph-disjoint A15-salted replication halves.

    Parameters
    ----------
    rows : tuple[FitPair, ...]
        Replication presentations.
    differences : numpy.ndarray
        Score differences aligned with rows.
    eligible_bands : frozenset[str]
        Bands entering the random band component.

    Returns
    -------
    tuple[_MetaFit, _MetaFit]
        Independent half-sample variance-component fits.

    Raises
    ------
    ValueError
        If the graph census cannot populate both halves.
    """

    fits = []
    for half in (0, 1):
        indices = [
            index for index, pair in enumerate(rows) if _synthetic_half_key(pair.graph_hash) == half
        ]
        if not indices:
            raise ValueError("A15-salted graph split leaves an empty JND half")
        half_rows = tuple(rows[index] for index in indices)
        half_differences = differences[indices]
        fits.append(_meta_fit(_cell_observations(half_rows, half_differences), eligible_bands))
    return fits[0], fits[1]


def _tie_rates(
    rows: Tuple[FitPair, ...],
    differences: np.ndarray,
    pooled_jnd: float,
    cell_jnd: Mapping[Tuple[str, str], float],
) -> Mapping[str, Tuple[float, float]]:
    """Compute per-class observed tie rates under pooled and cell bands.

    Parameters
    ----------
    rows : tuple[FitPair, ...]
        Replication presentations.
    differences : numpy.ndarray
        Score differences aligned with rows.
    pooled_jnd : float
        Pooled fitted tie half-band.
    cell_jnd : mapping[tuple[str, str], float]
        Supported class/band JNDs.

    Returns
    -------
    mapping[str, tuple[float, float]]
        Pooled-band and cell-band tie rates by class.
    """

    grouped: DefaultDict[str, list[int]] = defaultdict(list)
    for index, pair in enumerate(rows):
        grouped[pair.primary_class].append(index)
    result = {}
    for primary_class, members in sorted(grouped.items()):
        pooled = np.mean(np.abs(differences[members]) < pooled_jnd)
        cell = np.mean(
            [
                abs(float(differences[index]))
                < cell_jnd.get((rows[index].primary_class, rows[index].size_band), pooled_jnd)
                for index in members
            ]
        )
        result[primary_class] = float(pooled), float(cell)
    return MappingProxyType(result)


def fit_jnd_heterogeneity(
    pairs: Sequence[FitPair],
    plan: FittingPlan,
    weights: Mapping[str, float],
    config: JNDFitConfig,
) -> JNDHeterogeneityFit:
    """Fit and publish the frozen W-13 hierarchical uncertainty procedure.

    Parameters
    ----------
    pairs : sequence[FitPair]
        Train-role cross-session replication presentations from one stratum.
    plan : FittingPlan
        Frozen outer-weight plan.
    weights : mapping[str, float]
        Current profiled outer weights.
    config : JNDFitConfig
        Realized quota and rotation-envelope inputs with frozen constants.

    Returns
    -------
    JNDHeterogeneityFit
        Complete variance-component, interval, bootstrap, stability, and guard
        publication object.

    Raises
    ------
    ValueError
        If provenance, support, quota, or rotation inputs are incomplete.
    NotImplementedError
        If real rows reach the estimator before the graph-to-half assignment
        rule under the named A15 salt is frozen.
    """

    rows = tuple(pairs)
    counts = _validate_replication_rows(rows)
    if any(not pair.synthetic for pair in rows):
        raise NotImplementedError(
            "real W-13 split-half fitting is fail-closed because ADDENDUM-27 names "
            "the A15 salt but does not freeze a graph-to-half assignment rule"
        )
    bands = {pair.size_band for pair in rows}
    classes = {pair.primary_class for pair in rows}
    missing_band_counts = sorted(bands - set(config.top_composite_pair_counts))
    missing_envelopes = sorted(classes - set(config.rotation_envelopes))
    if missing_band_counts:
        raise ValueError(f"top-composite counts missing for bands: {missing_band_counts}")
    if missing_envelopes:
        raise ValueError(f"rotation envelopes missing for classes: {missing_envelopes}")
    supported = frozenset(
        cell for cell, count in counts.items() if count >= config.minimum_cell_count
    )
    if not supported:
        raise ValueError("no JND-HET cell meets the frozen 25-pair minimum")
    eligible_bands = frozenset(
        band for band, count in config.top_composite_pair_counts.items() if count >= config.q_band
    )
    vector = torch.tensor([weights[name] for name in plan.parameter_names], dtype=torch.float64)
    objective = PairwiseObjective(rows, plan)
    differences = objective.score_differences(vector).detach().numpy()
    observations = _cell_observations(rows, differences)
    initial_fit = _meta_fit(observations, eligible_bands)
    mu_ci_log, tau_class_ci, tau_band_ci = _meta_profile_intervals(
        observations, eligible_bands, initial_fit
    )
    half_one, half_two = _split_half_fit(rows, differences, eligible_bands)
    frozen_components = set()
    for component, interval in (
        ("tau_class", tau_class_ci.ratio_scale),
        ("tau_band", tau_band_ci.ratio_scale),
    ):
        difference = abs(getattr(half_one, component) - getattr(half_two, component))
        if difference > interval[1] - interval[0]:
            frozen_components.add(component)
    effective_dof = initial_fit.effective_dof_class + initial_fit.effective_dof_band
    if effective_dof > 2.0:
        if initial_fit.effective_dof_class > 0.0:
            frozen_components.add("tau_class")
        if initial_fit.effective_dof_band > 0.0:
            frozen_components.add("tau_band")
    fitted = (
        initial_fit
        if not frozen_components
        else _meta_fit(observations, eligible_bands, frozenset(frozen_components))
    )
    pooled_jnd = math.exp(fitted.mu)
    cell_jnd = {cell: math.exp(fitted.cell_logs[cell]) for cell in sorted(supported)}
    cell_jnd_ci = {
        cell: _conditional_cell_interval(
            rows,
            differences,
            cell,
            fitted.cell_logs[cell],
            fitted.tau_class**2 + (fitted.tau_band**2 if cell[1] in eligible_bands else 0.0),
        )
        for cell in sorted(supported)
    }
    spread = _spread(tuple(cell_jnd.values()))
    tau_variance = fitted.tau_class**2 + fitted.tau_band**2
    graph_interval, graph_drop, pooled_bootstrap_interval = _bootstrap_spread(
        rows,
        differences,
        "graph_hash",
        config,
        supported,
        fitted.cell_logs,
        tau_variance,
    )
    family_interval, family_drop, _ = _bootstrap_spread(
        rows,
        differences,
        "generator_family",
        config,
        supported,
        fitted.cell_logs,
        tau_variance,
    )
    disclosures = tuple(
        KJNDDisclosure(cell, counts[cell], cell_jnd_ci[cell])
        for cell in sorted(supported)
        if cell_jnd[cell] > 3.0 * pooled_jnd
    )
    uncalibrated = tuple(
        sorted(
            primary_class
            for primary_class in classes
            if min(
                (
                    value
                    for (cell_class, _), value in cell_jnd.items()
                    if cell_class == primary_class
                ),
                default=pooled_jnd,
            )
            <= config.rotation_envelopes[primary_class]
        )
    )
    return JNDHeterogeneityFit(
        mu=fitted.mu,
        pooled_jnd_ci=pooled_bootstrap_interval,
        pooled_jnd_profile_ci=(math.exp(mu_ci_log[0]), math.exp(mu_ci_log[1])),
        tau_class=fitted.tau_class,
        tau_band=fitted.tau_band,
        tau_class_ci=tau_class_ci,
        tau_band_ci=tau_band_ci,
        class_effects=fitted.class_effects,
        band_effects=fitted.band_effects,
        cell_jnd=cell_jnd,
        cell_jnd_ci=cell_jnd_ci,
        cell_counts=counts,
        unestimated_cells=tuple(sorted(set(counts) - supported)),
        spread=spread,
        spread_ci_graph_clusters=graph_interval,
        spread_ci_generator_families=family_interval,
        bootstrap_drop_rate=BootstrapDropRate(graph_drop, family_drop),
        effective_dof=effective_dof,
        shrink_actions=tuple(sorted(frozen_components)),
        k_jnd_disclosures=disclosures,
        split_half=SplitHalfStability(
            half_one={"tau_class": half_one.tau_class, "tau_band": half_one.tau_band},
            half_two={"tau_class": half_two.tau_class, "tau_band": half_two.tau_band},
            frozen_components=tuple(sorted(frozen_components)),
        ),
        uncalibrated_classes=uncalibrated,
        tie_rates_by_class=_tie_rates(rows, differences, pooled_jnd, cell_jnd),
        loss_path=initial_fit.loss_path + (() if fitted is initial_fit else fitted.loss_path),
        role_hash=config.role_hash,
        replication_row_ids=tuple(
            sorted(
                f"{row.base_pair_id}\0{row.session_id}\0{row.blind_id_a}\0{row.blind_id_b}"
                for row in rows
            )
        ),
    )


def evaluate_h_jnd_branch(
    fit: JNDHeterogeneityFit,
) -> HJNDBranchResult:
    """Evaluate and ledger TEST H-JND exactly once at the freeze fit.

    Parameters
    ----------
    fit : JNDHeterogeneityFit
        Completed W-13 publication object.
    Returns
    -------
    HJNDBranchResult
        Frozen two-branch decision and ledger digest.

    Raises
    ------
    AccessBudgetConsumedError
        If the branch event was already evaluated.
    """

    r_pool = fit.pooled_jnd_ci[1] / fit.pooled_jnd_ci[0]
    spread_lower = fit.spread_ci_graph_clusters[0]
    shipped = "class-conditional" if spread_lower > r_pool else "pooled"
    digest = AccessLedger().reserve_once(
        fit.role_hash,
        H_JND_LEDGER_KEY,
        fit.replication_row_ids,
        purpose="test-h-jnd-branch-evaluation",
    )
    return HJNDBranchResult(
        shipped_band=shipped,
        spread_lower=spread_lower,
        pooled_ratio=r_pool,
        row_set_digest=digest,
    )
