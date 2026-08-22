"""Deterministic projected optimizer for the P5 fitting objective."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Tuple

import numpy as np
import torch

from dagua.eval.ruler_v4.fit.bank import SplitPurpose
from dagua.eval.ruler_v4.fit.objective import FittingPlan, PairwiseObjective


@dataclass(frozen=True)
class OptimizerConfig:
    """Configure one deterministic projected-Adam fit.

    Parameters
    ----------
    seed : int, default=20260811
        Frozen campaign provenance value. The optimizer draws no randomness and
        does not apply this value to any process-global RNG.
    steps : int, default=1000
        Maximum optimizer updates.
    learning_rate : float, default=0.03
        Adam learning rate.
    tolerance : float, default=1e-9
        Absolute best-loss improvement threshold.
    patience : int, default=100
        Consecutive sub-tolerance steps required for convergence.
    """

    seed: int = 20260811
    steps: int = 1000
    learning_rate: float = 0.03
    tolerance: float = 1.0e-9
    patience: int = 100

    def __post_init__(self) -> None:
        """Validate optimizer hyperparameters.

        Raises
        ------
        ValueError
            If a setting is outside its deterministic finite domain.
        """

        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("optimizer seed must be a nonnegative integer")
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps <= 0:
            raise ValueError("optimizer steps must be a positive integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning rate must be finite and positive")
        if not math.isfinite(self.tolerance) or self.tolerance < 0.0:
            raise ValueError("tolerance must be finite and nonnegative")
        if (
            isinstance(self.patience, bool)
            or not isinstance(self.patience, int)
            or self.patience <= 0
        ):
            raise ValueError("patience must be a positive integer")


@dataclass(frozen=True)
class FitResult:
    """Publish fitted weights and their complete regularization path.

    Parameters
    ----------
    weights : mapping[str, float]
        Final bounded outer weights.
    weight_paths : mapping[str, tuple[float, ...]]
        Initial prior followed by every accepted optimizer update.
    losses : tuple[float, ...]
        Objective at the initial prior and every update.
    at_bounds : mapping[str, str]
        Parameters ending at ``lower``, ``upper``, or a ``fixed`` bound.
    intervals : mapping[str, tuple[float, float]]
        95% profile-likelihood intervals on the fitted weight scale.
    information_rank : int
        Rank of the fitted block's observed information.
    condition_number : float
        Condition number of the observed information.
    converged : bool
        Whether the patience rule fired before the step cap.
    steps_completed : int
        Number of optimizer updates.
    seed : int
        Campaign provenance value copied from the optimizer configuration; no
        RNG is seeded because the fitting loop draws no randomness.
    """

    weights: Mapping[str, float]
    weight_paths: Mapping[str, Tuple[float, ...]]
    losses: Tuple[float, ...]
    at_bounds: Mapping[str, str]
    intervals: Mapping[str, Tuple[float, float]]
    information_rank: int
    condition_number: float
    converged: bool
    steps_completed: int
    seed: int

    def __post_init__(self) -> None:
        """Freeze result mappings.

        Raises
        ------
        ValueError
            If paths do not align with the loss history.
        """

        weights = dict(self.weights)
        paths = {key: tuple(value) for key, value in self.weight_paths.items()}
        at_bounds = dict(self.at_bounds)
        intervals = {key: tuple(value) for key, value in self.intervals.items()}
        if set(weights) != set(paths):
            raise ValueError("weight and path identities must match")
        if any(len(path) != len(self.losses) for path in paths.values()):
            raise ValueError("every weight path must align with the loss path")
        if not set(at_bounds) <= set(weights) or any(
            value not in {"lower", "upper", "fixed"} for value in at_bounds.values()
        ):
            raise ValueError("bound flags must name fitted weights and valid bound sides")
        if set(intervals) != set(weights) or any(
            len(interval) != 2
            or not all(math.isfinite(value) for value in interval)
            or interval[0] > weights[name]
            or interval[1] < weights[name]
            for name, interval in intervals.items()
        ):
            raise ValueError("weight intervals must bracket every finite estimate")
        if self.information_rank < 0 or self.information_rank > len(weights):
            raise ValueError("observed-information rank is outside the fitted dimension")
        if math.isnan(self.condition_number) or self.condition_number < 1.0:
            raise ValueError("observed-information condition number must be at least one")
        object.__setattr__(self, "weights", MappingProxyType(weights))
        object.__setattr__(self, "weight_paths", MappingProxyType(paths))
        object.__setattr__(self, "at_bounds", MappingProxyType(at_bounds))
        object.__setattr__(self, "intervals", MappingProxyType(intervals))


@dataclass(frozen=True)
class OuterWeightStability:
    """Publish the 7.8 graph-disjoint split-half weight gate.

    Parameters
    ----------
    weights : mapping[str, float]
        Shipped weights after failing estimates are frozen at their priors.
    half_one, half_two : mapping[str, float]
        Graph-disjoint half-sample estimates.
    frozen_at_prior : tuple[str, ...]
        Weight identities failing the interval-width comparison.
    """

    weights: Mapping[str, float]
    half_one: Mapping[str, float]
    half_two: Mapping[str, float]
    frozen_at_prior: Tuple[str, ...]

    def __post_init__(self) -> None:
        """Freeze split-half publication mappings."""

        object.__setattr__(self, "weights", MappingProxyType(dict(self.weights)))
        object.__setattr__(self, "half_one", MappingProxyType(dict(self.half_one)))
        object.__setattr__(self, "half_two", MappingProxyType(dict(self.half_two)))


def apply_outer_weight_split_half(
    fitted: FitResult,
    half_one: FitResult,
    half_two: FitResult,
    plan: FittingPlan,
) -> OuterWeightStability:
    """Freeze outer weights whose graph-half estimates differ beyond their CI.

    Parameters
    ----------
    fitted : FitResult
        Full-data fit carrying profile-likelihood intervals.
    half_one, half_two : FitResult
        Graph-disjoint half-sample fits under the frozen A15 assignment.
    plan : FittingPlan
        Frozen weight priors.

    Returns
    -------
    OuterWeightStability
        Shipped values and named stability failures.

    Raises
    ------
    ValueError
        If the three fit identities do not match the plan.
    """

    names = set(plan.parameter_names)
    if any(set(result.weights) != names for result in (fitted, half_one, half_two)):
        raise ValueError("outer split-half fit identities do not match the plan")
    frozen = tuple(
        sorted(
            name
            for name in names
            if abs(half_one.weights[name] - half_two.weights[name])
            > fitted.intervals[name][1] - fitted.intervals[name][0]
        )
    )
    priors = {parameter.name: parameter.prior for parameter in plan.weights}
    shipped = {
        name: priors[name] if name in frozen else fitted.weights[name] for name in sorted(names)
    }
    return OuterWeightStability(
        weights=shipped,
        half_one=half_one.weights,
        half_two=half_two.weights,
        frozen_at_prior=frozen,
    )


def _information_diagnostics(
    objective: PairwiseObjective, weights: Mapping[str, float]
) -> Tuple[int, float]:
    """Compute observed-information rank and condition number.

    Parameters
    ----------
    objective : PairwiseObjective
        Fitted ordered-probit objective.
    weights : mapping[str, float]
        Optimum keyed by parameter identity.

    Returns
    -------
    tuple[int, float]
        Numerical matrix rank and condition number.
    """

    vector = torch.tensor(
        [weights[name] for name in objective.plan.parameter_names],
        dtype=objective.dtype,
        requires_grad=True,
    )
    # Excluding the prior prevents regularization from hiding a likelihood-rank
    # deficiency that this mandatory publication exists to expose.
    information = torch.autograd.functional.hessian(
        lambda candidate: objective.negative_log_likelihood(candidate) * len(objective.pairs),
        vector,
    )
    matrix = information.detach().numpy()
    rank = int(np.linalg.matrix_rank(matrix))
    condition = float(np.linalg.cond(matrix))
    return rank, condition


def _profile_weight_intervals(
    objective: PairwiseObjective, weights: Mapping[str, float]
) -> Mapping[str, Tuple[float, float]]:
    """Compute two-sided 95% intervals by profiling the fitted objective.

    Parameters
    ----------
    objective : PairwiseObjective
        Fitted ordered-probit objective including the frozen weight prior.
    weights : mapping[str, float]
        Optimum keyed by parameter identity.

    Returns
    -------
    mapping[str, tuple[float, float]]
        Profile-likelihood intervals on the weight scale.
    """

    from scipy.optimize import brentq, minimize

    names = objective.plan.parameter_names
    optimum = np.asarray([weights[name] for name in names], dtype=np.float64)
    bounds = [
        (float(parameter.lower), float(parameter.upper)) for parameter in objective.plan.weights
    ]
    target = float(objective.loss(torch.tensor(optimum, dtype=objective.dtype)))
    target += 1.920729410347062 / len(objective.pairs)

    def profile(index: int, fixed_value: float) -> float:
        """Profile every weight except one fixed coordinate.

        Parameters
        ----------
        index : int
            Fixed weight coordinate.
        fixed_value : float
            Candidate value on the fitted scale.

        Returns
        -------
        float
            Profile objective minus the 95% likelihood-ratio target.
        """

        free = [position for position in range(len(names)) if position != index]
        if not free:
            candidate = np.asarray((fixed_value,), dtype=np.float64)
            return float(objective.loss(torch.tensor(candidate, dtype=objective.dtype))) - target

        def evaluate(free_values: np.ndarray) -> float:
            """Evaluate one free-coordinate profile candidate.

            Parameters
            ----------
            free_values : numpy.ndarray
                Candidate values for non-fixed weights.

            Returns
            -------
            float
                Mean regularized negative log likelihood.
            """

            candidate = optimum.copy()
            candidate[index] = fixed_value
            candidate[free] = free_values
            return float(objective.loss(torch.tensor(candidate, dtype=objective.dtype)))

        result = minimize(
            evaluate,
            optimum[free],
            method="L-BFGS-B",
            bounds=[bounds[position] for position in free],
        )
        if not result.success:
            raise RuntimeError(f"weight profile optimization failed: {result.message}")
        return float(result.fun) - target

    intervals = {}
    for index, name in enumerate(names):
        estimate = optimum[index]
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
        intervals[name] = (lower, upper)
    return MappingProxyType(intervals)


def fit_weights(objective: PairwiseObjective, config: OptimizerConfig) -> FitResult:
    """Fit bounded weights with deterministic projected Adam.

    The optimizer starts at the preregistered priors and projects after every
    update. Projection makes the spec's ``[0.25x, 4x]`` range and traceability
    floors hard constraints rather than soft penalties.

    Parameters
    ----------
    objective : PairwiseObjective
        Dof-guarded P5 likelihood.
    config : OptimizerConfig
        Deterministic optimizer settings.

    Returns
    -------
    FitResult
        Final bounded weights and complete paths.

    Raises
    ------
    ValueError
        If a non-FIT row reaches the weight optimizer.
    FloatingPointError
        If the objective or a gradient becomes nonfinite.
    """

    if any(pair.purpose is not SplitPurpose.FIT for pair in objective.pairs):
        raise ValueError("weight fitting accepts A15 FIT rows only")
    plan = objective.plan
    names = plan.parameter_names
    lower = torch.tensor([parameter.lower for parameter in plan.weights], dtype=objective.dtype)
    upper = torch.tensor([parameter.upper for parameter in plan.weights], dtype=objective.dtype)
    weights = torch.tensor(
        [parameter.prior for parameter in plan.weights],
        dtype=objective.dtype,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam((weights,), lr=config.learning_rate)
    initial_loss = objective.loss(weights)
    if not bool(torch.isfinite(initial_loss)):
        raise FloatingPointError("initial fitting objective is nonfinite")
    loss_path = [float(initial_loss.detach())]
    paths = {name: [float(weights[index].detach())] for index, name in enumerate(names)}
    best_loss = loss_path[0]
    stale_steps = 0
    converged = False
    for _ in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = objective.loss(weights)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("fitting objective became nonfinite")
        loss.backward()
        if weights.grad is None or not bool(torch.all(torch.isfinite(weights.grad))):
            raise FloatingPointError("fitting gradient became nonfinite")
        optimizer.step()
        with torch.no_grad():
            weights.clamp_(min=lower, max=upper)
        evaluated = objective.loss(weights)
        if not bool(torch.isfinite(evaluated)):
            raise FloatingPointError("projected fitting objective became nonfinite")
        current_loss = float(evaluated.detach())
        loss_path.append(current_loss)
        for index, name in enumerate(names):
            paths[name].append(float(weights[index].detach()))
        if best_loss - current_loss > config.tolerance:
            best_loss = current_loss
            stale_steps = 0
        else:
            stale_steps += 1
        if stale_steps >= config.patience:
            converged = True
            break
    final_weights = {name: paths[name][-1] for name in names}
    at_bounds = {}
    for parameter in plan.weights:
        value = final_weights[parameter.name]
        at_lower = math.isclose(value, float(parameter.lower), rel_tol=0.0, abs_tol=1.0e-12)
        at_upper = math.isclose(value, float(parameter.upper), rel_tol=0.0, abs_tol=1.0e-12)
        if at_lower or at_upper:
            at_bounds[parameter.name] = (
                "fixed" if at_lower and at_upper else ("lower" if at_lower else "upper")
            )
    intervals = _profile_weight_intervals(objective, final_weights)
    information_rank, condition_number = _information_diagnostics(objective, final_weights)
    return FitResult(
        weights=final_weights,
        weight_paths={name: tuple(path) for name, path in paths.items()},
        losses=tuple(loss_path),
        at_bounds=at_bounds,
        intervals=intervals,
        information_rank=information_rank,
        condition_number=condition_number,
        converged=converged,
        steps_completed=len(loss_path) - 1,
        seed=config.seed,
    )
