"""Guarded orchestration for the synthetic FREEZE-1 profiled fit."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from dagua.eval.ruler_v4.fit.access import AccessLedger
from dagua.eval.ruler_v4.fit.bank import _FROZEN_A15_ROLE_HASH, SplitPurpose
from dagua.eval.ruler_v4.fit.objective import (
    FitPair,
    FittingPlan,
    PairwiseObjective,
    partition_fit_ord_lines,
)
from dagua.eval.ruler_v4.fit.optimize import (
    FitResult,
    OuterWeightStability,
    _information_diagnostics,
    _profile_weight_intervals,
    apply_outer_weight_split_half,
)
from dagua.eval.ruler_v4.fit.uncertainty import (
    HJNDBranchResult,
    JNDFitConfig,
    JNDHeterogeneityFit,
    JNDProfileFit,
    _synthetic_half_key,
    evaluate_h_jnd_branch,
    fit_jnd_heterogeneity,
    profile_jnd_block,
)

_JOINT_TOLERANCE = 1.0e-10
_MAX_FIXED_POINT_ITERATIONS = 25
_SYNTHETIC_LAPSE_BOUNDS = (0.0, 0.25)
_SYNTHETIC_LAPSE_INITIAL = 0.01


class FitStartConditionError(RuntimeError):
    """Signal that a real FREEZE-1 fit has not met its start conditions."""


class FitConvergenceError(RuntimeError):
    """Signal that the profiled fixed point exhausted its iteration budget."""


@dataclass(frozen=True)
class RealFitStartConditions:
    """Declare owner-controlled gates that must precede any real fit.

    Parameters
    ----------
    campaign_complete : bool
        Whether the preregistered MAIN campaign has completed.
    protocol_start_authorized : bool
        Whether the owner has authorized the frozen fit start.
    lapse_prior_frozen : bool
        Whether DISCREPANCIES 56 has been resolved.
    graph_half_assignment_frozen : bool
        Whether DISCREPANCIES 57 has been resolved.
    blind_map_attested : bool
        Whether the orchestration attested blind-map separation.
    """

    campaign_complete: bool
    protocol_start_authorized: bool
    lapse_prior_frozen: bool
    graph_half_assignment_frozen: bool
    blind_map_attested: bool

    @property
    def ready(self) -> bool:
        """Return whether every real-fit start gate is affirmative.

        Returns
        -------
        bool
            True only when all owner-controlled gates are satisfied.
        """

        return all(asdict(self).values())


@dataclass(frozen=True)
class FitDriverConfig:
    """Freeze deterministic orchestration constants.

    Parameters
    ----------
    seed : int
        Frozen deterministic seed.
    joint_tolerance : float
        Frozen convergence tolerance on the joint objective.
    maximum_iterations : int
        Fail-closed fixed-point iteration budget.
    """

    seed: int = field(default=20260811, init=False)
    joint_tolerance: float = field(default=_JOINT_TOLERANCE, init=False)
    maximum_iterations: int = field(default=_MAX_FIXED_POINT_ITERATIONS, init=False)


@dataclass(frozen=True)
class FitDriverIteration:
    """Publish one profiled fixed-point trajectory row.

    Parameters
    ----------
    iteration : int
        One-based outer iteration.
    weights : mapping[str, float]
        Updated outer weights.
    lapse_rate : float
        Updated synthetic lapse MLE.
    mu, tau_class, tau_band : float
        Profiled JND-block values used for this update.
    train_objective, jnd_objective, joint_objective : float
        Summed objective components and their joint value.
    jnd_improvement : float or None
        JND-profile block decrease from the preceding fixed-point row.
    weight_lapse_improvement : float
        Train-objective decrease from this row's weight/lapse block.
    joint_improvement : float or None
        Sum of the two coordinate-block decreases; unavailable on the first row.
    """

    iteration: int
    weights: Mapping[str, float]
    lapse_rate: float
    mu: float
    tau_class: float
    tau_band: float
    train_objective: float
    jnd_objective: float
    joint_objective: float
    jnd_improvement: Optional[float]
    weight_lapse_improvement: float
    joint_improvement: Optional[float]

    def __post_init__(self) -> None:
        """Freeze the trajectory weight mapping."""

        object.__setattr__(self, "weights", MappingProxyType(dict(self.weights)))


@dataclass(frozen=True)
class Freeze1FitResult:
    """Publish the complete guarded FREEZE-1 synthetic run.

    Parameters
    ----------
    weight_fit : FitResult
        Full-data outer-weight fit at the converged JND profile.
    outer_weight_stability : OuterWeightStability
        Graph-disjoint split-half response and shipped weights.
    lapse_rate : float
        Fitted synthetic seven-category lapse rate.
    jnd_fit : JNDHeterogeneityFit
        Final W-13 point and uncertainty publications.
    h_jnd_branch : HJNDBranchResult
        Once-only ledgered H-JND branch decision.
    trajectory : tuple[FitDriverIteration, ...]
        Frozen joint-objective fixed-point trajectory.
    access_budget_before, access_budget_after : mapping[str, int]
        Persistent ledger usage surrounding the run.
    run_dir : pathlib.Path
        Newly created artifact directory.
    """

    weight_fit: FitResult
    outer_weight_stability: OuterWeightStability
    lapse_rate: float
    jnd_fit: JNDHeterogeneityFit
    h_jnd_branch: HJNDBranchResult
    trajectory: Tuple[FitDriverIteration, ...]
    access_budget_before: Mapping[str, int]
    access_budget_after: Mapping[str, int]
    run_dir: Path

    def __post_init__(self) -> None:
        """Freeze driver budget mappings."""

        object.__setattr__(
            self, "access_budget_before", MappingProxyType(dict(self.access_budget_before))
        )
        object.__setattr__(
            self, "access_budget_after", MappingProxyType(dict(self.access_budget_after))
        )


def _atomic_write_json(path: Path, payload: object) -> None:
    """Atomically write one deterministic JSON artifact.

    Parameters
    ----------
    path : pathlib.Path
        Final artifact path inside a new run directory.
    payload : object
        JSON-serializable value.
    """

    temporary = path.with_suffix(f"{path.suffix}.tmp")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, f"{encoded}\n".encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Atomically write deterministic JSON Lines.

    Parameters
    ----------
    path : pathlib.Path
        Final JSONL artifact path.
    rows : sequence[mapping[str, object]]
        Ordered trajectory records.
    """

    temporary = path.with_suffix(f"{path.suffix}.tmp")
    encoded = "".join(
        f"{json.dumps(dict(row), sort_keys=True, separators=(',', ':'), allow_nan=False)}\n"
        for row in rows
    ).encode("utf-8")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _input_digest(pairs: Sequence[FitPair]) -> str:
    """Digest stable train-line identities and responses.

    Parameters
    ----------
    pairs : sequence[FitPair]
        Complete train-role input.

    Returns
    -------
    str
        SHA-256 digest for artifact provenance.
    """

    digest = hashlib.sha256()
    for pair in pairs:
        fields = (
            pair.replicate_group_id,
            pair.base_pair_id,
            pair.session_id,
            pair.blind_id_a,
            pair.blind_id_b,
            str(pair.graded_verdict),
        )
        digest.update("\0".join(fields).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _pairs_with_profile(
    pairs: Sequence[FitPair], profile: JNDProfileFit, lapse_rate: float
) -> Tuple[FitPair, ...]:
    """Apply one profiled JND block and lapse to train rows.

    Parameters
    ----------
    pairs : sequence[FitPair]
        Complete train line.
    profile : JNDProfileFit
        Current JND profile with pooled fallbacks.
    lapse_rate : float
        Current uniform lapse value.

    Returns
    -------
    tuple[FitPair, ...]
        Updated immutable likelihood rows.
    """

    pooled_jnd = math.exp(profile.mu)
    return tuple(
        replace(
            pair,
            jnd=profile.jnd_by_cell.get((pair.primary_class, pair.size_band), pooled_jnd),
            lapse_rate=lapse_rate,
        )
        for pair in pairs
    )


def _fit_weight_lapse_block(
    pairs: Sequence[FitPair],
    plan: FittingPlan,
    config: FitDriverConfig,
    initial_weights: Optional[Mapping[str, float]] = None,
    initial_lapse: Optional[float] = None,
) -> Tuple[FitResult, float, float]:
    """Jointly optimize synthetic outer weights and the uniform lapse.

    Parameters
    ----------
    pairs : sequence[FitPair]
        Train rows at a fixed profiled JND block.
    plan : FittingPlan
        Frozen outer-weight plan.
    config : FitDriverConfig
        Deterministic seed and iteration provenance.
    initial_weights : mapping[str, float] or None
        Prior fixed-point weights, defaulting to frozen literature priors.
    initial_lapse : float or None
        Prior fixed-point lapse, defaulting to the synthetic initializer.

    Returns
    -------
    tuple[FitResult, float, float]
        Weight publication, fitted lapse, and mean regularized train objective.
    """

    from scipy.optimize import minimize

    rows = tuple(pairs)
    names = plan.parameter_names
    starting_weights = (
        {parameter.name: float(parameter.prior) for parameter in plan.weights}
        if initial_weights is None
        else dict(initial_weights)
    )
    if set(starting_weights) != set(names):
        raise ValueError("initial weight identities do not match the fitting plan")
    parameter_bounds = []
    for parameter in plan.weights:
        if parameter.lower is None or parameter.upper is None:
            raise RuntimeError("fitting-plan weight bounds were not normalized")
        parameter_bounds.append((parameter.lower, parameter.upper))
    bounds_by_name = {
        parameter.name: parameter_bounds[index] for index, parameter in enumerate(plan.weights)
    }
    starting_lapse = _SYNTHETIC_LAPSE_INITIAL if initial_lapse is None else initial_lapse
    initial = np.asarray(
        [starting_weights[name] for name in names] + [starting_lapse], dtype=np.float64
    )
    bounds = parameter_bounds + [_SYNTHETIC_LAPSE_BOUNDS]

    def evaluate(candidate: np.ndarray) -> float:
        """Evaluate one joint synthetic weight/lapse candidate.

        Parameters
        ----------
        candidate : numpy.ndarray
            Weight coordinates followed by lapse.

        Returns
        -------
        float
            Mean regularized train objective.
        """

        objective = PairwiseObjective(
            tuple(replace(pair, lapse_rate=float(candidate[-1])) for pair in rows),
            plan,
        )
        vector = torch.tensor(candidate[:-1], dtype=torch.float64)
        return float(objective.loss(vector))

    accepted = [initial.copy()]
    losses = [evaluate(initial)]

    def record(candidate: np.ndarray) -> None:
        """Record one accepted L-BFGS-B iterate.

        Parameters
        ----------
        candidate : numpy.ndarray
            Accepted joint candidate.
        """

        accepted.append(np.asarray(candidate, dtype=np.float64).copy())
        losses.append(evaluate(candidate))

    # SciPy's stubs do not model the legal Powell callback/options combination.
    result = minimize(  # type: ignore[call-overload]
        evaluate,
        initial,
        method="Powell",
        bounds=bounds,
        callback=record,
        options={"ftol": 1.0e-13, "xtol": 1.0e-13, "maxiter": 500},
    )
    if not math.isfinite(float(result.fun)):
        raise FloatingPointError(f"joint weight/lapse optimization failed: {result.message}")
    final = np.asarray(result.x, dtype=np.float64)
    if not np.array_equal(accepted[-1], final):
        accepted.append(final.copy())
        losses.append(evaluate(final))
    weights = {name: float(final[index]) for index, name in enumerate(names)}
    lapse_rate = float(final[-1])
    objective = PairwiseObjective(
        tuple(replace(pair, lapse_rate=lapse_rate) for pair in rows),
        plan,
    )
    at_bounds = {
        parameter.name: (
            "fixed"
            if math.isclose(
                bounds_by_name[parameter.name][0],
                bounds_by_name[parameter.name][1],
                abs_tol=1.0e-12,
            )
            else (
                "lower"
                if math.isclose(
                    weights[parameter.name],
                    bounds_by_name[parameter.name][0],
                    abs_tol=1.0e-12,
                )
                else "upper"
            )
        )
        for parameter in plan.weights
        if math.isclose(weights[parameter.name], bounds_by_name[parameter.name][0], abs_tol=1.0e-12)
        or math.isclose(weights[parameter.name], bounds_by_name[parameter.name][1], abs_tol=1.0e-12)
    }
    intervals = _profile_weight_intervals(objective, weights)
    information_rank, condition_number = _information_diagnostics(objective, weights)
    fit = FitResult(
        weights=weights,
        weight_paths={
            name: tuple(float(candidate[index]) for candidate in accepted)
            for index, name in enumerate(names)
        },
        losses=tuple(losses),
        at_bounds=at_bounds,
        intervals=intervals,
        information_rank=information_rank,
        condition_number=condition_number,
        converged=bool(result.success),
        steps_completed=len(accepted) - 1,
        seed=config.seed,
    )
    return fit, lapse_rate, float(result.fun)


def _require_real_start_conditions(
    conditions: Optional[RealFitStartConditions],
) -> None:
    """Fail closed unless every owner-controlled real-fit gate is satisfied.

    Parameters
    ----------
    conditions : RealFitStartConditions or None
        Explicit real-fit authorization state.

    Raises
    ------
    FitStartConditionError
        If any required start condition is absent.
    NotImplementedError
        After gates pass because real adapters remain deliberately inactive.
    """

    if conditions is None or not conditions.ready:
        raise FitStartConditionError(
            "real FREEZE-1 requires campaign completion and every protocol start condition"
        )
    raise NotImplementedError(
        "real FREEZE-1 activation remains disabled until the frozen lapse prior and "
        "graph-half artifacts replace the synthetic-only implementations"
    )


def run_freeze1_fit(
    pairs: Sequence[FitPair],
    plan: FittingPlan,
    jnd_config: JNDFitConfig,
    run_dir: Path,
    ledger_root: Path,
    config: Optional[FitDriverConfig] = None,
    real_start_conditions: Optional[RealFitStartConditions] = None,
) -> Freeze1FitResult:
    """Run guarded synthetic FREEZE-1 fitting and write complete artifacts.

    Parameters
    ----------
    pairs : sequence[FitPair]
        Complete train-role rows containing realized replication groups.
    plan : FittingPlan
        Frozen outer-weight plan.
    jnd_config : JNDFitConfig
        Frozen W-13 support and guard inputs.
    run_dir : pathlib.Path
        New, non-existing output directory.
    ledger_root : pathlib.Path
        Explicit non-campaign ledger root for this synthetic-only run.
    config : FitDriverConfig or None
        Frozen driver configuration; ``None`` constructs the only valid values.
    real_start_conditions : RealFitStartConditions or None
        Required owner gates for real data. Real activation remains disabled.

    Returns
    -------
    Freeze1FitResult
        Complete fitted publications, branch decision, ledger state, and run path.

    Raises
    ------
    FitStartConditionError
        If real rows arrive before campaign and protocol start gates.
    FitConvergenceError
        If the synthetic profiled fixed point misses the frozen tolerance.
    FileExistsError
        If ``run_dir`` already exists.
    ValueError
        If rows violate train-only or deterministic configuration guards, or
        if a synthetic run targets the campaign ledger root.
    """

    rows = tuple(pairs)
    if not rows:
        raise ValueError("FREEZE-1 driver requires nonempty train rows")
    if any(pair.purpose is not SplitPurpose.FIT for pair in rows):
        raise ValueError("FREEZE-1 driver accepts train-role rows only")
    if any(not pair.synthetic for pair in rows):
        _require_real_start_conditions(real_start_conditions)
    driver_config = FitDriverConfig() if config is None else config
    lines = partition_fit_ord_lines(rows)
    output = Path(run_dir)
    ledger = AccessLedger(ledger_root)
    if ledger.is_campaign_root:
        raise ValueError("synthetic FREEZE-1 cannot target the campaign ledger root")
    output.mkdir(parents=False, exist_ok=False)
    budget_before = ledger.budget_usage(_FROZEN_A15_ROLE_HASH)
    _atomic_write_json(
        output / "manifest.json",
        {
            "input_digest": _input_digest(rows),
            "row_count": len(rows),
            "replication_row_count": len(lines.replication),
            "role_hash": _FROZEN_A15_ROLE_HASH,
            "seed": driver_config.seed,
            "joint_tolerance": driver_config.joint_tolerance,
            "maximum_iterations": driver_config.maximum_iterations,
            "synthetic_only": True,
            "access_budget_before": dict(budget_before),
        },
    )
    _atomic_write_json(output / "status.json", {"state": "RUNNING"})
    try:
        current_weights = {parameter.name: float(parameter.prior) for parameter in plan.weights}
        current_lapse = _SYNTHETIC_LAPSE_INITIAL
        trajectory = []
        weight_fit: Optional[FitResult] = None
        profile: Optional[JNDProfileFit] = None
        for iteration in range(1, driver_config.maximum_iterations + 1):
            profile = profile_jnd_block(
                lines.replication,
                plan,
                current_weights,
                jnd_config,
                previous_profile=profile,
            )
            profiled_train = _pairs_with_profile(lines.train, profile, current_lapse)
            weight_fit, current_lapse, train_mean = _fit_weight_lapse_block(
                profiled_train,
                plan,
                driver_config,
                current_weights,
                current_lapse,
            )
            current_weights = dict(weight_fit.weights)
            train_objective_before = weight_fit.losses[0] * len(profiled_train)
            train_objective = train_mean * len(profiled_train)
            joint_objective = train_objective + profile.marginal_loss
            weight_lapse_improvement = max(train_objective_before - train_objective, 0.0)
            improvement = (
                None
                if profile.block_improvement is None
                else profile.block_improvement + weight_lapse_improvement
            )
            trajectory.append(
                FitDriverIteration(
                    iteration=iteration,
                    weights=current_weights,
                    lapse_rate=current_lapse,
                    mu=profile.mu,
                    tau_class=profile.tau_class,
                    tau_band=profile.tau_band,
                    train_objective=train_objective,
                    jnd_objective=profile.marginal_loss,
                    joint_objective=joint_objective,
                    jnd_improvement=profile.block_improvement,
                    weight_lapse_improvement=weight_lapse_improvement,
                    joint_improvement=improvement,
                )
            )
            if improvement is not None and improvement <= driver_config.joint_tolerance:
                break
        else:
            raise FitConvergenceError(
                "FREEZE-1 profiled fixed point exhausted its deterministic iteration budget"
            )
        if weight_fit is None or profile is None:
            raise RuntimeError("FREEZE-1 driver produced no fixed-point iteration")
        jnd_fit = fit_jnd_heterogeneity(
            lines.replication,
            plan,
            current_weights,
            jnd_config,
        )
        final_train = tuple(
            replace(
                pair,
                jnd=jnd_fit.jnd_by_cell.get(
                    (pair.primary_class, pair.size_band), math.exp(jnd_fit.mu)
                ),
                lapse_rate=current_lapse,
            )
            for pair in lines.train
        )
        halves = tuple(
            tuple(pair for pair in final_train if _synthetic_half_key(pair.graph_hash) == half)
            for half in (0, 1)
        )
        if any(not half for half in halves):
            raise ValueError("synthetic graph split leaves an empty outer-weight half")
        half_one, _, _ = _fit_weight_lapse_block(halves[0], plan, driver_config)
        half_two, _, _ = _fit_weight_lapse_block(halves[1], plan, driver_config)
        outer_stability = apply_outer_weight_split_half(weight_fit, half_one, half_two, plan)
        if jnd_fit.uncalibrated_classes:
            raise ValueError(
                f"rotation-envelope guard blocks classes: {list(jnd_fit.uncalibrated_classes)}"
            )
        branch = evaluate_h_jnd_branch(jnd_fit, ledger=ledger, synthetic_only=True)
        budget_after = ledger.budget_usage(_FROZEN_A15_ROLE_HASH)
        trajectory_rows = [
            {
                "iteration": item.iteration,
                "weights": dict(item.weights),
                "lapse_rate": item.lapse_rate,
                "mu": item.mu,
                "tau_class": item.tau_class,
                "tau_band": item.tau_band,
                "train_objective": item.train_objective,
                "jnd_objective": item.jnd_objective,
                "joint_objective": item.joint_objective,
                "jnd_improvement": item.jnd_improvement,
                "weight_lapse_improvement": item.weight_lapse_improvement,
                "joint_improvement": item.joint_improvement,
            }
            for item in trajectory
        ]
        _atomic_write_jsonl(output / "trajectory.jsonl", trajectory_rows)
        _atomic_write_json(
            output / "result.json",
            {
                "weights": dict(weight_fit.weights),
                "shipped_weights": dict(outer_stability.weights),
                "weight_intervals": {
                    name: list(interval) for name, interval in weight_fit.intervals.items()
                },
                "lapse_rate": current_lapse,
                "jnd": {
                    "mu": jnd_fit.mu,
                    "tau_class": jnd_fit.tau_class,
                    "tau_band": jnd_fit.tau_band,
                    "spread": jnd_fit.spread,
                    "spread_ci_graph_clusters": list(jnd_fit.spread_ci_graph_clusters),
                    "spread_ci_generator_families": list(jnd_fit.spread_ci_generator_families),
                    "unestimated_cells": [list(cell) for cell in jnd_fit.unestimated_cells],
                    "effective_dof": jnd_fit.effective_dof,
                    "c06_component_audit": {
                        name: asdict(audit) for name, audit in jnd_fit.c06_component_audit.items()
                    },
                    "n_jnd": jnd_fit.n_jnd,
                    "variance_boundary_disclosures": [
                        asdict(disclosure) for disclosure in jnd_fit.variance_boundary_disclosures
                    ],
                    "split_half_frozen": list(jnd_fit.split_half.frozen_components),
                    "c06_shrink_actions": list(jnd_fit.c06_shrink_actions),
                    "c06_partial_declaration": jnd_fit.c06_partial_declaration,
                },
                "h_jnd_branch": asdict(branch),
                "access_budget_after": dict(budget_after),
                "ledger_annulments": list(ledger.annulment_lines(_FROZEN_A15_ROLE_HASH)),
                "ledger_defects": list(ledger.ledger_defects(_FROZEN_A15_ROLE_HASH)),
                "iterations": len(trajectory),
                "converged": True,
            },
        )
        _atomic_write_json(output / "status.json", {"state": "COMPLETE"})
        return Freeze1FitResult(
            weight_fit=weight_fit,
            outer_weight_stability=outer_stability,
            lapse_rate=current_lapse,
            jnd_fit=jnd_fit,
            h_jnd_branch=branch,
            trajectory=tuple(trajectory),
            access_budget_before=budget_before,
            access_budget_after=budget_after,
            run_dir=output,
        )
    except Exception as error:
        _atomic_write_json(
            output / "status.json",
            {"state": "FAILED", "error_type": type(error).__name__, "message": str(error)},
        )
        raise
