"""tsNET expressed as a composable ops pipeline."""

from __future__ import annotations

from typing import ClassVar, Optional, Tuple

import torch

from dagua.layout.ops.base import Op, Pipeline, Repeat  # noqa: E402
from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig  # noqa: E402
from dagua.layout.ops.graph_utils import (
    all_pairs_shortest_paths as _all_pairs_shortest_paths,
)
from dagua.layout.ops.graph_utils import (
    build_undirected_adjacency as _build_undirected_adjacency,
)
from dagua.layout.ops.graph_utils import (
    layout_device as _layout_device,
)
from dagua.layout.ops.graph_utils import (
    layout_extent as _layout_extent,
)
from dagua.layout.ops.graph_utils import (
    normalize_positions as _normalize_positions,
)
from dagua.layout.ops.state import (  # noqa: E402
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)
from dagua.layout.ops.taxonomy import OpCategory  # noqa: E402

_MIN_DISTANCE = 1.0e-12


def _row_probabilities(distances: torch.Tensor, perplexity: float) -> torch.Tensor:
    """Match one row's Gaussian bandwidth to a target perplexity.

    Parameters
    ----------
    distances : torch.Tensor
        Row of graph distances with shape ``[N]``.
    perplexity : float
        Target perplexity.

    Returns
    -------
    torch.Tensor
        Conditional probability row with shape ``[N]``.
    """
    num_nodes = int(distances.shape[0])
    if num_nodes <= 1:
        return torch.zeros_like(distances)

    mask = torch.ones(num_nodes, dtype=torch.bool)
    mask[int(torch.argmin(distances).item())] = False
    squared = distances.square()

    beta = torch.tensor(1.0, dtype=torch.float32)
    beta_min = torch.tensor(float("-inf"), dtype=torch.float32)
    beta_max = torch.tensor(float("inf"), dtype=torch.float32)
    target_entropy = torch.log(torch.tensor(perplexity, dtype=torch.float32))

    probabilities = torch.zeros_like(distances)
    for _ in range(100):
        weights = torch.exp(-squared * beta) * mask.to(dtype=torch.float32)
        weights_sum = weights.sum().clamp(min=_MIN_DISTANCE)
        probabilities = weights / weights_sum
        entropy = -(probabilities[mask] * probabilities[mask].clamp(min=_MIN_DISTANCE).log()).sum()
        error = entropy - target_entropy
        if torch.abs(error) < 1.0e-5:
            break
        if error > 0:
            beta_min = beta
            beta = beta * 2.0 if torch.isinf(beta_max) else (beta + beta_max) * 0.5
        else:
            beta_max = beta
            beta = beta * 0.5 if torch.isinf(beta_min) else (beta + beta_min) * 0.5

    probabilities[int(torch.argmin(distances).item())] = 0.0
    return probabilities


def _high_dimensional_affinities(distance_matrix: torch.Tensor, perplexity: float) -> torch.Tensor:
    """Build the symmetric t-SNE input affinity matrix.

    Parameters
    ----------
    distance_matrix : torch.Tensor
        Shortest-path matrix with shape ``[N, N]``.
    perplexity : float
        Target perplexity.

    Returns
    -------
    torch.Tensor
        Symmetric probability matrix ``P`` with shape ``[N, N]``.
    """
    rows = [
        _row_probabilities(distance_matrix[node], perplexity)
        for node in range(distance_matrix.shape[0])
    ]
    conditional = torch.stack(rows, dim=0)
    symmetrized = (conditional + conditional.transpose(0, 1)) / (
        2.0 * max(distance_matrix.shape[0], 1)
    )
    return symmetrized.clamp(min=_MIN_DISTANCE)


def _tsne_loss(positions: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
    """Evaluate the exact t-SNE KL objective.

    Parameters
    ----------
    positions : torch.Tensor
        Current embedding with shape ``[N, 2]``.
    probabilities : torch.Tensor
        Symmetric input affinity matrix ``P``.

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    delta = positions.unsqueeze(1) - positions.unsqueeze(0)
    squared_distances = delta.square().sum(dim=2)
    numerators = (1.0 + squared_distances).reciprocal()
    diagonal_mask = ~torch.eye(
        positions.shape[0],
        dtype=torch.bool,
        device=positions.device,
    )
    numerators = numerators * diagonal_mask.to(dtype=numerators.dtype)
    q = numerators / numerators.sum().clamp(min=_MIN_DISTANCE)
    return (probabilities * (probabilities.log() - q.clamp(min=_MIN_DISTANCE).log())).sum()


def _gradient_descent_step(
    positions: torch.Tensor,
    grad: torch.Tensor,
    update: torch.Tensor,
    gains: torch.Tensor,
    learning_rate: float,
    momentum: float,
    min_gain: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one exact t-SNE gains-plus-momentum update.

    Parameters
    ----------
    positions : torch.Tensor
        Current embedding with shape ``[N, 2]``.
    grad : torch.Tensor
        Gradient tensor with shape ``[N, 2]``.
    update : torch.Tensor
        Velocity tensor with shape ``[N, 2]``.
    gains : torch.Tensor
        Per-parameter gain tensor with shape ``[N, 2]``.
    learning_rate : float
        Step size for the current optimization phase.
    momentum : float
        Momentum coefficient for the current optimization phase.
    min_gain : float
        Floor applied to every gain entry.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Updated ``(positions, update, gains)`` tensors.
    """
    inc = (update * grad) < 0.0
    dec = ~inc
    gains[inc] += 0.2
    gains[dec] *= 0.8
    gains.clamp_(min=min_gain)
    grad = grad * gains

    update = momentum * update - learning_rate * grad
    positions.add_(update)
    return positions, update, gains


class _InitializeTsnetPositions(Op):
    """Initialize positions exactly like classic tsNET.

    Uses ``torch.randn`` with a seeded generator scaled by 1e-4,
    matching sklearn TSNE(init="random").
    """

    name: ClassVar[str] = "tsnet_initialize_positions"
    category: ClassVar[OpCategory] = OpCategory.INIT
    writes: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Seed ``state.pos`` from the torch generator random initialization.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs containing ``num_nodes`` and ``seed``.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with ``state.pos`` populated as a ``float32`` tensor with
            ``requires_grad=True``.
        """
        del ctx

        device = _layout_device(problem.edge_index, problem.node_sizes)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(problem.seed)
        initial = (
            torch.randn(problem.num_nodes, 2, generator=generator, dtype=torch.float32) * 1e-4
        ).to(device)
        state.pos = initial.clone().requires_grad_(True)
        return state


class _PrepareTsnetState(Op):
    """Compute graph distances and high-dimensional affinities."""

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
        """Build distances, affinities, and optimizer state for tsNET.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state receiving tsNET extras.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with tsNET affinities and optimizer tensors stored in
            ``extras``.
        """
        del ctx

        device = _layout_device(problem.edge_index, problem.node_sizes)
        adjacency = _build_undirected_adjacency(
            problem.edge_index, problem.num_nodes, edge_weights=problem.edge_weights
        )
        distances = _all_pairs_shortest_paths(adjacency, weighted=problem.edge_weights is not None)
        user_perplexity = state.extras.get("tsnet_perplexity", 30.0)
        perplexity = min(float(user_perplexity), float(max(problem.num_nodes - 1, 1)))
        probabilities = _high_dimensional_affinities(distances, perplexity).to(device)

        state.extras["tsnet_probabilities"] = probabilities
        state.extras["tsnet_early_exaggeration"] = 12.0
        state.extras["tsnet_early_exaggeration_steps"] = 250
        state.extras["tsnet_min_gain"] = 0.01
        early_lr = max(float(max(problem.num_nodes, 1)) / 48.0, 50.0)
        state.extras["tsnet_early_learning_rate"] = early_lr
        state.extras["tsnet_late_learning_rate"] = early_lr
        # Velocity and gains initialized after pos is set
        return state


class _InitializeTsnetOptimizer(Op):
    """Initialize the velocity and gains tensors from current positions."""

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
        """Create velocity and gains buffers matching ``state.pos``.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state containing initialized ``pos``.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with velocity and gains tensors in ``extras``.

        Raises
        ------
        ValueError
            If ``state.pos`` is missing.
        """
        del problem, ctx

        if state.pos is None:
            raise ValueError("_InitializeTsnetOptimizer requires state.pos to be set.")

        state.extras["tsnet_update"] = torch.zeros_like(state.pos)
        state.extras["tsnet_gains"] = torch.ones_like(state.pos)
        return state


class _TsnetGradientStep(Op):
    """Compute loss, backprop, and apply the gains-plus-momentum update."""

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
        """Execute one tsNET optimization step with exaggeration schedule.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state containing positions and optimizer extras.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with updated positions, velocity, and gains.

        Raises
        ------
        ValueError
            If ``state.pos`` is missing.
        """
        del problem, ctx

        if state.pos is None:
            raise ValueError("_TsnetGradientStep requires state.pos to be set.")

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
        loss = _tsne_loss(state.pos, probabilities * exaggeration)
        loss.backward()

        grad = state.pos.grad.detach().clone()
        momentum = 0.5 if step < early_steps else 0.8
        learning_rate = early_lr if step < early_steps else late_lr

        with torch.no_grad():
            state.pos, update, gains = _gradient_descent_step(
                positions=state.pos,
                grad=grad,
                update=update,
                gains=gains,
                learning_rate=learning_rate,
                momentum=momentum,
                min_gain=min_gain,
            )
            state.pos.grad.zero_()

        state.extras["tsnet_update"] = update
        state.extras["tsnet_gains"] = gains
        return state


class _FinalizeTsnetPositions(Op):
    """Apply classic tsNET's final normalization and cast."""

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
        """Center, normalize, and cast positions like classic tsNET.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state containing final positions.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with final output positions on the classic return device.

        Raises
        ------
        ValueError
            If ``state.pos`` is missing.
        """
        del ctx

        if state.pos is None:
            raise ValueError("_FinalizeTsnetPositions requires state.pos to be set.")

        device = _layout_device(problem.edge_index, problem.node_sizes)
        extent = _layout_extent(problem.num_nodes, problem.node_sizes)
        state.pos = _normalize_positions(state.pos.detach(), extent).to(
            dtype=torch.float32, device=device
        )
        return state


def build_tsnet_pipeline(steps: int = 1000) -> Pipeline:
    """Build a tsNET pipeline that is bit-identical to classic ``layout_tsnet``.

    Parameters
    ----------
    steps : int, default=1000
        Number of optimization updates.

    Returns
    -------
    Pipeline
        Pipeline that reproduces classic tsNET's initialization, affinity
        computation, gains-plus-momentum optimization, and postprocessing.

    Raises
    ------
    ValueError
        If ``steps`` is negative.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative.")

    return Pipeline(
        [
            FixedSteps(FixedStepsConfig(n=steps)),
            _InitializeTsnetPositions(),
            _PrepareTsnetState(),
            _InitializeTsnetOptimizer(),
            Repeat(
                n=steps,
                ops=[
                    _TsnetGradientStep(),
                ],
            ),
            _FinalizeTsnetPositions(),
        ],
        name="tsnet_pipeline",
    )


def layout_tsnet_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    perplexity: float = 30,
    steps: int = 1000,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the tsNET pipeline as a drop-in replacement for classic ``layout_tsnet``.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor used for final scaling.
    perplexity : float, default=30
        Target t-SNE perplexity. Currently only the default value of 30
        preserves bit-identity with classic; non-default values require
        extending ``_PrepareTsnetState``.
    steps : int, default=1000
        Number of optimization updates.
    seed : int, default=42
        Random seed for the torch generator initialization.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Final position tensor with the same dtype, device, and values as
        classic ``layout_tsnet``.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``steps``, ``perplexity``, or ``edge_weights``
        are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if perplexity <= 0:
        raise ValueError("perplexity must be positive.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
            )

    device = _layout_device(edge_index, node_sizes)

    # Handle trivial cases exactly like classic.
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32, device=device)

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    state.extras["tsnet_perplexity"] = perplexity
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_tsnet_pipeline(steps=steps).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("tsNET pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_tsnet_pipeline", "layout_tsnet_pipeline"]
