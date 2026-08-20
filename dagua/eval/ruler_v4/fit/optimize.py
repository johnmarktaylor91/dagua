"""Deterministic projected optimizer for the P5 fitting objective."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Tuple

import numpy as np
import torch

from dagua.eval.ruler_v4.fit.objective import PairwiseObjective


@dataclass(frozen=True)
class OptimizerConfig:
    """Configure one deterministic projected-Adam fit.

    Parameters
    ----------
    seed : int, default=20260811
        Frozen seed applied to Python, NumPy, and Torch.
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
    converged : bool
        Whether the patience rule fired before the step cap.
    steps_completed : int
        Number of optimizer updates.
    seed : int
        Deterministic seed used by the fit.
    """

    weights: Mapping[str, float]
    weight_paths: Mapping[str, Tuple[float, ...]]
    losses: Tuple[float, ...]
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
        if set(weights) != set(paths):
            raise ValueError("weight and path identities must match")
        if any(len(path) != len(self.losses) for path in paths.values()):
            raise ValueError("every weight path must align with the loss path")
        object.__setattr__(self, "weights", MappingProxyType(weights))
        object.__setattr__(self, "weight_paths", MappingProxyType(paths))


def _seed_everything(seed: int) -> None:
    """Seed every RNG used by the fitting harness.

    Parameters
    ----------
    seed : int
        Nonnegative deterministic seed.
    """

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


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
    FloatingPointError
        If the objective or a gradient becomes nonfinite.
    """

    _seed_everything(config.seed)
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
    return FitResult(
        weights=final_weights,
        weight_paths={name: tuple(path) for name, path in paths.items()},
        losses=tuple(loss_path),
        converged=converged,
        steps_completed=len(loss_path) - 1,
        seed=config.seed,
    )
