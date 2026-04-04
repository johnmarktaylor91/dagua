"""Registered tsNET operations for composable layout pipelines."""

from __future__ import annotations

from typing import ClassVar, Tuple

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

_MIN_DISTANCE = 1.0e-12
_TSNET_EARLY_EXAGGERATION = 12.0
_TSNET_EARLY_STEPS = 250
_TSNET_MIN_GAIN = 0.01


@register_op
class TsnetInitializePositions(Op):
    """Initialize tsNET positions with small Gaussian noise."""

    name: ClassVar[str] = "tsnet_initialize_positions"
    category: ClassVar[OpCategory] = OpCategory.INIT
    writes: ClassVar[Tuple[str, ...]] = ("pos",)

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
        generator = torch.Generator(device="cpu")
        generator.manual_seed(problem.seed)
        state.pos = (
            (torch.randn(problem.num_nodes, 2, generator=generator, dtype=torch.float32) * 1e-4)
            .to(device=device)
            .clone()
            .requires_grad_(True)
        )
        return state


@register_op
class TsnetPrepareState(Op):
    """Compute all-pairs distances and tsNET affinities plus optimization constants."""

    name: ClassVar[str] = "tsnet_prepare_state"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

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
            Mutable solve state receiving prepared ``extras`` values.
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
            float(state.extras.get("tsnet_perplexity", 30.0)),
            float(max(problem.num_nodes - 1, 1)),
        )
        adjacency = build_undirected_adjacency(
            problem.edge_index,
            problem.num_nodes,
            edge_weights=problem.edge_weights,
        )
        distances = all_pairs_shortest_paths(adjacency, weighted=problem.edge_weights is not None)

        rows = []
        num_nodes = int(distances.shape[0])
        for node in range(num_nodes):
            row = distances[node]
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
            for _ in range(100):
                weights = torch.exp(-squared * beta) * mask.to(dtype=torch.float32)
                weights_sum = weights.sum().clamp(min=_MIN_DISTANCE)
                probabilities = weights / weights_sum
                entropy = -(
                    probabilities[mask] * probabilities[mask].clamp(min=_MIN_DISTANCE).log()
                ).sum()
                error = entropy - target_entropy
                if torch.abs(error) < 1.0e-5:
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
        probabilities = symmetrized.clamp(min=_MIN_DISTANCE).to(device=device)

        state.extras["tsnet_perplexity"] = perplexity
        state.extras["tsnet_probabilities"] = probabilities
        state.extras["tsnet_early_exaggeration"] = _TSNET_EARLY_EXAGGERATION
        state.extras["tsnet_early_exaggeration_steps"] = _TSNET_EARLY_STEPS
        state.extras["tsnet_min_gain"] = _TSNET_MIN_GAIN
        early_lr = max(float(max(problem.num_nodes, 1)) / 48.0, 50.0)
        state.extras["tsnet_early_learning_rate"] = early_lr
        state.extras["tsnet_late_learning_rate"] = early_lr
        return state


@register_op
class TsnetInitializeOptimizer(Op):
    """Initialize tsNET optimizer buffers."""

    name: ClassVar[str] = "tsnet_initialize_optimizer"
    category: ClassVar[OpCategory] = OpCategory.INIT
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

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
        state.extras["tsnet_update"] = torch.zeros_like(state.pos)
        state.extras["tsnet_gains"] = torch.ones_like(state.pos)
        return state


@register_op
class TsnetGradientStep(Op):
    """Compute one tsNET gradient step with gains and momentum."""

    name: ClassVar[str] = "tsnet_gradient_step"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("pos", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("pos", "extras")
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

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

        probabilities = state.extras["tsnet_probabilities"]
        early_exag = state.extras["tsnet_early_exaggeration"]
        early_steps = state.extras["tsnet_early_exaggeration_steps"]
        min_gain = state.extras["tsnet_min_gain"]
        early_lr = state.extras["tsnet_early_learning_rate"]
        late_lr = state.extras["tsnet_late_learning_rate"]
        update = state.extras["tsnet_update"]
        gains = state.extras["tsnet_gains"]

        step = state.step
        exaggeration = early_exag if step < early_steps else 1.0
        effective_probabilities = probabilities * exaggeration
        delta = state.pos.unsqueeze(1) - state.pos.unsqueeze(0)
        squared_distances = delta.square().sum(dim=2)
        numerators = (1.0 + squared_distances).reciprocal()
        diagonal_mask = ~torch.eye(
            state.pos.shape[0],
            dtype=torch.bool,
            device=state.pos.device,
        )
        numerators = numerators * diagonal_mask.to(dtype=numerators.dtype)
        q = numerators / numerators.sum().clamp(min=_MIN_DISTANCE)
        loss = (
            effective_probabilities
            * (
                effective_probabilities.clamp(min=_MIN_DISTANCE).log()
                - q.clamp(min=_MIN_DISTANCE).log()
            )
        ).sum()
        loss.backward()

        grad = state.pos.grad.detach().clone()
        momentum = 0.5 if step < early_steps else 0.8
        learning_rate = early_lr if step < early_steps else late_lr

        with torch.no_grad():
            inc = (update * grad) < 0.0
            dec = ~inc
            gains[inc] += 0.2
            gains[dec] *= 0.8
            gains.clamp_(min=min_gain)
            grad = grad * gains
            update = momentum * update - learning_rate * grad
            state.pos.add_(update)
            state.pos.grad.zero_()

        state.extras["tsnet_update"] = update
        state.extras["tsnet_gains"] = gains
        return state


@register_op
class TsnetFinalizePositions(Op):
    """Center, normalize, and cast tsNET positions to output dtype/device."""

    name: ClassVar[str] = "tsnet_finalize_positions"
    category: ClassVar[OpCategory] = OpCategory.POSTPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

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
    "TsnetInitializePositions",
    "TsnetPrepareState",
    "TsnetInitializeOptimizer",
    "TsnetGradientStep",
    "TsnetFinalizePositions",
]
