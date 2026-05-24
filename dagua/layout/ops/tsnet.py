"""Registered tsNET operations for composable layout pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple

import numpy as np
import torch

from dagua.layout.ops.base import Op
from dagua.layout.ops.graph_utils import (
    all_pairs_shortest_paths,
    build_undirected_adjacency,
    layout_device,
    layout_extent,
    normalize_positions,
)
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_TSNET_PERPLEXITY_KEY = "tsnet_perplexity"
_TSNET_PROBABILITIES_KEY = "tsnet_probabilities"
_TSNET_EARLY_EXAGGERATION_KEY = "tsnet_early_exaggeration"
_TSNET_EARLY_STEPS_KEY = "tsnet_early_exaggeration_steps"
_TSNET_MIN_GAIN_KEY = "tsnet_min_gain"
_TSNET_MIN_DISTANCE_KEY = "tsnet_min_distance"
_TSNET_EARLY_LR_KEY = "tsnet_early_learning_rate"
_TSNET_LATE_LR_KEY = "tsnet_late_learning_rate"
_TSNET_UPDATE_KEY = "tsnet_update"
_TSNET_GAINS_KEY = "tsnet_gains"
_TSNET_BEST_ERROR_KEY = "tsnet_best_error"
_TSNET_BEST_ITER_KEY = "tsnet_best_iter"
_TSNET_GRAD_NORM_KEY = "tsnet_grad_norm"


@dataclass(frozen=True)
class TsnetInitializePositionsConfig:
    """Configuration for :class:`TsnetInitializePositions`.

    Parameters
    ----------
    noise_scale : float, default=1e-4
        Standard deviation of the Gaussian initialization noise.
    fidelity_mode : bool, default=False
        When ``True``, use NumPy ``RandomState`` draws to match sklearn
        ``TSNE(init="random")`` seed semantics.
    """

    noise_scale: float = 1.0e-4
    fidelity_mode: bool = False


@dataclass(frozen=True)
class TsnetPrepareStateConfig:
    """Configuration for :class:`TsnetPrepareState`.

    Parameters
    ----------
    default_perplexity : float, default=30.0
        Perplexity used when ``state.extras["tsnet_perplexity"]`` is not set.
    binary_search_tolerance : float, default=1e-5
        Convergence tolerance for the binary search over beta.
    binary_search_iterations : int, default=100
        Maximum iterations for the per-row binary search.
    lr_divisor : float, default=48.0
        Divisor applied to ``num_nodes`` to compute the initial learning rate.
    lr_floor : float, default=50.0
        Minimum learning rate applied when ``num_nodes / lr_divisor`` is small.
    early_exaggeration : float, default=12.0
        Multiplicative weight applied to ``P`` during the early tsNET phase.
    early_exaggeration_steps : int, default=250
        Number of early-exaggeration updates.
    min_gain : float, default=0.01
        Lower bound for per-parameter adaptive gains.
    min_distance : float, default=1e-12
        Numerical floor used when normalizing and taking logarithms.
    """

    default_perplexity: float = 30.0
    binary_search_tolerance: float = 1.0e-5
    binary_search_iterations: int = 100
    lr_divisor: float = 48.0
    lr_floor: float = 50.0
    early_exaggeration: float = 12.0
    early_exaggeration_steps: int = 250
    min_gain: float = 0.01
    min_distance: float = 1.0e-12


@dataclass(frozen=True)
class TsnetGradientStepConfig:
    """Configuration for :class:`TsnetGradientStep`.

    Parameters
    ----------
    early_momentum : float, default=0.5
        Momentum used during the early exaggeration phase.
    late_momentum : float, default=0.8
        Momentum used after the early exaggeration phase.
    gain_increase : float, default=0.2
        Additive gain bump when the gradient sign flips.
    gain_decrease : float, default=0.8
        Multiplicative gain decay when the gradient sign is consistent.
    gradient_scale : float, default=4.0
        Scale factor applied to the KL objective before backpropagation.
    convergence_check_interval : int, default=50
        Number of iterations between sklearn-style convergence checks.
    n_iter_without_progress : int, default=300
        Maximum iterations without KL improvement before stopping.
    min_grad_norm : float, default=1e-7
        Gradient-norm floor below which optimization stops.
    """

    early_momentum: float = 0.5
    late_momentum: float = 0.8
    gain_increase: float = 0.2
    gain_decrease: float = 0.8
    gradient_scale: float = 4.0
    convergence_check_interval: int = 50
    n_iter_without_progress: int = 300
    min_grad_norm: float = 1.0e-7


@register_op
class TsnetInitializePositions(Op):
    """Initialize tsNET coordinates from deterministic Gaussian noise."""

    config: TsnetInitializePositionsConfig

    name: ClassVar[str] = "tsnet_initialize_positions"
    category: ClassVar[OpCategory] = OpCategory.INIT
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    access_pattern: ClassVar[str] = "global"

    def __init__(self, config: Optional[TsnetInitializePositionsConfig] = None) -> None:
        """Store initialization parameters.

        Parameters
        ----------
        config : TsnetInitializePositionsConfig | None, optional
            Optional initialization configuration.
        """
        self.config = config or TsnetInitializePositionsConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Seed ``state.pos`` from deterministic CPU RNG output.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs containing ``num_nodes`` and ``seed``.
        state : SolveState
            Mutable solve state receiving initialized positions.
        ctx : RuntimeContext
            Execution context (unused).

        Returns
        -------
        SolveState
            State with ``state.pos`` populated and ``requires_grad=True``.
        """
        del ctx
        device = layout_device(problem.edge_index, problem.node_sizes)
        if self.config.fidelity_mode:
            random_state = np.random.RandomState(problem.seed)
            initial = torch.from_numpy(
                (
                    self.config.noise_scale * random_state.standard_normal((problem.num_nodes, 2))
                ).astype(np.float32, copy=False)
            )
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(problem.seed)
            initial = (
                torch.randn(problem.num_nodes, 2, generator=generator, dtype=torch.float32)
                * self.config.noise_scale
            )
        state.pos = initial.to(device=device).clone().requires_grad_(True)
        return state


@register_op
class TsnetPrepareState(Op):
    """Build tsNET distances, affinities, and optimizer constants."""

    config: TsnetPrepareStateConfig

    name: ClassVar[str] = "tsnet_prepare_state"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = (f"extras.{_TSNET_PERPLEXITY_KEY}",)
    writes: ClassVar[Tuple[str, ...]] = (
        "distance_matrix",
        f"extras.{_TSNET_PERPLEXITY_KEY}",
        f"extras.{_TSNET_PROBABILITIES_KEY}",
        f"extras.{_TSNET_EARLY_EXAGGERATION_KEY}",
        f"extras.{_TSNET_EARLY_STEPS_KEY}",
        f"extras.{_TSNET_MIN_GAIN_KEY}",
        f"extras.{_TSNET_MIN_DISTANCE_KEY}",
        f"extras.{_TSNET_EARLY_LR_KEY}",
        f"extras.{_TSNET_LATE_LR_KEY}",
    )
    access_pattern: ClassVar[str] = "global"

    def __init__(self, config: Optional[TsnetPrepareStateConfig] = None) -> None:
        """Store deterministic affinity-preparation parameters.

        Parameters
        ----------
        config : TsnetPrepareStateConfig | None, optional
            Optional preparation configuration.
        """
        self.config = config or TsnetPrepareStateConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build tsNET distance, affinity, and optimization-state tensors.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state receiving the dense distance matrix and
            tsNET-specific optimization extras.
        ctx : RuntimeContext
            Execution context (unused).

        Returns
        -------
        SolveState
            State with tsNET precomputed probabilities and scheduler constants.
        """
        del ctx

        device = layout_device(problem.edge_index, problem.node_sizes)
        perplexity = min(
            float(state.extras.get(_TSNET_PERPLEXITY_KEY, self.config.default_perplexity)),
            float(max(problem.num_nodes - 1, 1)),
        )
        adjacency = build_undirected_adjacency(
            problem.edge_index,
            problem.num_nodes,
            edge_weights=problem.edge_weights,
        )
        state.distance_matrix = all_pairs_shortest_paths(
            adjacency,
            weighted=problem.edge_weights is not None,
        )

        rows = []
        num_nodes = int(state.distance_matrix.shape[0])
        for node in range(num_nodes):
            row = state.distance_matrix[node]
            if num_nodes <= 1:
                rows.append(torch.zeros_like(row))
                continue

            mask = torch.ones(num_nodes, dtype=torch.bool)
            mask[int(torch.argmin(row).item())] = False
            squared = row.square()

            beta = torch.tensor(1.0, dtype=torch.float32)
            beta_min = torch.tensor(float("-inf"), dtype=torch.float32)
            beta_max = torch.tensor(float("inf"), dtype=torch.float32)
            target_entropy = torch.log(torch.tensor(perplexity, dtype=torch.float32))

            probabilities = torch.zeros_like(row)
            for _ in range(self.config.binary_search_iterations):
                # The classic implementation zeros the self-edge before
                # normalization so perplexity only reflects true neighbors.
                weights = torch.exp(-squared * beta) * mask.to(dtype=torch.float32)
                weights_sum = weights.sum().clamp(min=self.config.min_distance)
                probabilities = weights / weights_sum
                entropy = -(
                    probabilities[mask]
                    * probabilities[mask].clamp(min=self.config.min_distance).log()
                ).sum()
                error = entropy - target_entropy
                if torch.abs(error) < self.config.binary_search_tolerance:
                    break
                if error > 0:
                    beta_min = beta
                    beta = beta * 2.0 if torch.isinf(beta_max) else (beta + beta_max) * 0.5
                else:
                    beta_max = beta
                    beta = beta * 0.5 if torch.isinf(beta_min) else (beta + beta_min) * 0.5

            probabilities[int(torch.argmin(row).item())] = 0.0
            rows.append(probabilities)

        conditional = torch.stack(rows, dim=0) if rows else torch.empty((0, 0), device=device)
        symmetrized = (conditional + conditional.transpose(0, 1)) / (2.0 * max(num_nodes, 1))
        probabilities = symmetrized.clamp(min=self.config.min_distance).to(device=device)

        state.extras[_TSNET_PERPLEXITY_KEY] = perplexity
        state.extras[_TSNET_PROBABILITIES_KEY] = probabilities
        state.extras[_TSNET_EARLY_EXAGGERATION_KEY] = self.config.early_exaggeration
        state.extras[_TSNET_EARLY_STEPS_KEY] = self.config.early_exaggeration_steps
        state.extras[_TSNET_MIN_GAIN_KEY] = self.config.min_gain
        state.extras[_TSNET_MIN_DISTANCE_KEY] = self.config.min_distance
        early_lr = max(
            float(max(problem.num_nodes, 1)) / self.config.lr_divisor,
            self.config.lr_floor,
        )
        state.extras[_TSNET_EARLY_LR_KEY] = early_lr
        state.extras[_TSNET_LATE_LR_KEY] = early_lr
        return state


@register_op
class TsnetInitializeOptimizer(Op):
    """Initialize tsNET optimizer buffers."""

    name: ClassVar[str] = "tsnet_initialize_optimizer"
    category: ClassVar[OpCategory] = OpCategory.OPTIMIZE
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = (
        f"extras.{_TSNET_UPDATE_KEY}",
        f"extras.{_TSNET_GAINS_KEY}",
        f"extras.{_TSNET_BEST_ERROR_KEY}",
        f"extras.{_TSNET_BEST_ITER_KEY}",
        f"extras.{_TSNET_GRAD_NORM_KEY}",
    )
    requires: ClassVar[Tuple[str, ...]] = ("pos",)
    access_pattern: ClassVar[str] = "global"

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Create zero-update and unit-gain tensors for each node.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs (unused).
        state : SolveState
            Mutable solve state containing initialized positions.
        ctx : RuntimeContext
            Execution context (unused).

        Returns
        -------
        SolveState
            State with ``tsnet_update`` and ``tsnet_gains`` initialized.

        Raises
        ------
        ValueError
            If ``state.pos`` is not set.
        """
        del problem, ctx
        if state.pos is None:
            raise ValueError("TsnetInitializeOptimizer requires state.pos to be set.")
        state.extras[_TSNET_UPDATE_KEY] = torch.zeros_like(state.pos)
        state.extras[_TSNET_GAINS_KEY] = torch.ones_like(state.pos)
        state.extras[_TSNET_BEST_ERROR_KEY] = float("inf")
        state.extras[_TSNET_BEST_ITER_KEY] = 0
        state.extras[_TSNET_GRAD_NORM_KEY] = float("inf")
        return state


@register_op
class TsnetGradientStep(Op):
    """Apply one classic tsNET momentum-and-gains optimization update."""

    config: TsnetGradientStepConfig

    name: ClassVar[str] = "tsnet_gradient_step"
    category: ClassVar[OpCategory] = OpCategory.OPTIMIZE
    reads: ClassVar[Tuple[str, ...]] = (
        "pos",
        f"extras.{_TSNET_PROBABILITIES_KEY}",
        f"extras.{_TSNET_EARLY_EXAGGERATION_KEY}",
        f"extras.{_TSNET_EARLY_STEPS_KEY}",
        f"extras.{_TSNET_MIN_GAIN_KEY}",
        f"extras.{_TSNET_MIN_DISTANCE_KEY}",
        f"extras.{_TSNET_EARLY_LR_KEY}",
        f"extras.{_TSNET_LATE_LR_KEY}",
        f"extras.{_TSNET_UPDATE_KEY}",
        f"extras.{_TSNET_GAINS_KEY}",
        f"extras.{_TSNET_BEST_ERROR_KEY}",
        f"extras.{_TSNET_BEST_ITER_KEY}",
    )
    writes: ClassVar[Tuple[str, ...]] = (
        "pos",
        "prev_loss",
        "converged",
        f"extras.{_TSNET_UPDATE_KEY}",
        f"extras.{_TSNET_GAINS_KEY}",
        f"extras.{_TSNET_BEST_ERROR_KEY}",
        f"extras.{_TSNET_BEST_ITER_KEY}",
        f"extras.{_TSNET_GRAD_NORM_KEY}",
    )
    requires: ClassVar[Tuple[str, ...]] = (
        "pos",
        f"extras.{_TSNET_PROBABILITIES_KEY}",
        f"extras.{_TSNET_UPDATE_KEY}",
        f"extras.{_TSNET_GAINS_KEY}",
    )
    access_pattern: ClassVar[str] = "global"

    def __init__(self, config: Optional[TsnetGradientStepConfig] = None) -> None:
        """Store gain-and-momentum hyperparameters.

        Parameters
        ----------
        config : TsnetGradientStepConfig | None, optional
            Optional gradient-step configuration.
        """
        self.config = config or TsnetGradientStepConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Evaluate KL loss and apply one momentum-plus-gain update.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs (unused).
        state : SolveState
            Mutable solve state containing positions and optimizer buffers.
        ctx : RuntimeContext
            Execution context (unused).

        Returns
        -------
        SolveState
            State with updated positions, ``tsnet_update``, and ``tsnet_gains``.

        Raises
        ------
        ValueError
            If ``state.pos`` is not set.
        """
        del problem, ctx
        if state.pos is None:
            raise ValueError("TsnetGradientStep requires state.pos to be set.")

        probabilities = state.extras[_TSNET_PROBABILITIES_KEY]
        early_exag = state.extras[_TSNET_EARLY_EXAGGERATION_KEY]
        early_steps = state.extras[_TSNET_EARLY_STEPS_KEY]
        min_gain = state.extras[_TSNET_MIN_GAIN_KEY]
        min_distance = state.extras[_TSNET_MIN_DISTANCE_KEY]
        early_lr = state.extras[_TSNET_EARLY_LR_KEY]
        late_lr = state.extras[_TSNET_LATE_LR_KEY]
        update = state.extras[_TSNET_UPDATE_KEY]
        gains = state.extras[_TSNET_GAINS_KEY]

        step = state.step
        with torch.enable_grad():
            exaggeration = early_exag if step < early_steps else 1.0
            effective_probabilities = probabilities * exaggeration
            delta = state.pos.unsqueeze(1) - state.pos.unsqueeze(0)
            squared_distances = delta.square().sum(dim=2)
            numerators = (1.0 + squared_distances).reciprocal()
            # Self-similarities stay zero so the Student-t normalization matches
            # classic t-SNE and only covers off-diagonal mass.
            diagonal_mask = ~torch.eye(
                state.pos.shape[0],
                dtype=torch.bool,
                device=state.pos.device,
            )
            numerators = numerators * diagonal_mask.to(dtype=numerators.dtype)
            q = numerators / numerators.sum().clamp(min=min_distance)
            loss = (
                effective_probabilities
                * (
                    effective_probabilities.clamp(min=min_distance).log()
                    - q.clamp(min=min_distance).log()
                )
            ).sum()
            (loss * self.config.gradient_scale).backward()

        grad = state.pos.grad.detach().clone()
        momentum = self.config.early_momentum if step < early_steps else self.config.late_momentum
        learning_rate = early_lr if step < early_steps else late_lr

        with torch.no_grad():
            # Adaptive gains increase only where the update and current
            # gradient disagree, matching the classic tsNET optimizer state.
            inc = (update * grad) < 0.0
            dec = ~inc
            gains[inc] += self.config.gain_increase
            gains[dec] *= self.config.gain_decrease
            gains.clamp_(min=min_gain)
            grad = grad * gains
            grad_norm = float(torch.linalg.vector_norm(grad).item())
            update = momentum * update - learning_rate * grad
            state.pos.add_(update)
            state.pos.grad.zero_()

        state.extras[_TSNET_UPDATE_KEY] = update
        state.extras[_TSNET_GAINS_KEY] = gains
        state.extras[_TSNET_GRAD_NORM_KEY] = grad_norm
        state.prev_loss = float(loss.detach().item())
        if step == early_steps:
            state.extras[_TSNET_BEST_ERROR_KEY] = float("inf")
            state.extras[_TSNET_BEST_ITER_KEY] = step
        if (step + 1) % self.config.convergence_check_interval == 0:
            best_error = float(state.extras.get(_TSNET_BEST_ERROR_KEY, float("inf")))
            best_iter = int(state.extras.get(_TSNET_BEST_ITER_KEY, step))
            if state.prev_loss < best_error:
                state.extras[_TSNET_BEST_ERROR_KEY] = state.prev_loss
                state.extras[_TSNET_BEST_ITER_KEY] = step
            elif step - best_iter > self.config.n_iter_without_progress:
                state.converged = True
            if grad_norm <= self.config.min_grad_norm:
                state.converged = True
        return state


@register_op
class TsnetFinalizePositions(Op):
    """Center, normalize, and cast tsNET positions to output dtype/device."""

    name: ClassVar[str] = "tsnet_finalize_positions"
    category: ClassVar[OpCategory] = OpCategory.POSTPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)
    access_pattern: ClassVar[str] = "global"

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Normalize the final positions to the configured output extent.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state containing final positions.
        ctx : RuntimeContext
            Execution context (unused).

        Returns
        -------
        SolveState
            State with ``state.pos`` normalized and moved to target device.

        Raises
        ------
        ValueError
            If ``state.pos`` is missing.
        """
        del ctx
        if state.pos is None:
            raise ValueError("TsnetFinalizePositions requires state.pos to be set.")

        device = layout_device(problem.edge_index, problem.node_sizes)
        extent = layout_extent(problem.num_nodes, problem.node_sizes)
        state.pos = normalize_positions(state.pos.detach(), extent).to(
            dtype=torch.float32,
            device=device,
        )
        return state


__all__ = [
    "TsnetInitializePositionsConfig",
    "TsnetInitializePositions",
    "TsnetPrepareStateConfig",
    "TsnetPrepareState",
    "TsnetInitializeOptimizer",
    "TsnetGradientStepConfig",
    "TsnetGradientStep",
    "TsnetFinalizePositions",
]
