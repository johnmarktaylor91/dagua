"""NeuLay two-phase graph layout expressed as a composable ops pipeline."""

from __future__ import annotations

import math
from typing import ClassVar, Optional, Tuple

import numpy as np
import torch
from torch import nn

from dagua.layout.classic.neulay import (
    _GCN_REL_TOL,
    _GNN_LR,
    _LINEAR_REL_TOL,
    _PAIR_QUERY_RADIUS_FACTOR,
    _PAIR_REFRESH_INTERVAL,
    _PATIENCE,
    _clean_edge_index,
    _elastic_loss,
    _initial_positions,
    _kdtree_repulsion_loss,
    _kdtree_repulsion_pairs,
    _layout_device,
    _relative_window_difference,
    _ResGCN,
    _set_seed,
    _validate_inputs,
)
from dagua.layout.ops.base import Op, Pipeline, Repeat
from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory

# ---------------------------------------------------------------------------
# Pipeline-local ops
# ---------------------------------------------------------------------------


class _SeedRNG(Op):
    """Seed PyTorch and NumPy RNGs exactly like classic NeuLay."""

    name: ClassVar[str] = "neulay_seed_rng"
    category: ClassVar[OpCategory] = OpCategory.INIT
    writes: ClassVar[Tuple[str, ...]] = ()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Set the global RNG state to match classic NeuLay seeding.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs containing ``seed``.
        state : SolveState
            Mutable solve state. Unchanged by this op.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            Unchanged state (RNG is global side-effect).
        """
        del ctx
        _set_seed(problem.seed)
        return state


class _PrepareNeuLayState(Op):
    """Clean edges and resolve NeuLay hyperparameters into extras."""

    name: ClassVar[str] = "neulay_prepare_state"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def __init__(
        self,
        *,
        dim: int = 2,
        lr: float = 0.01,
        radius: float = 0.4,
        magnitude: Optional[float] = None,
        use_gcn: bool = True,
        gcn_steps: int = 2_000,
        total_steps: int = 20_000,
    ) -> None:
        """Store NeuLay configuration.

        Parameters
        ----------
        dim : int
            Embedding dimensionality.
        lr : float
            RMSprop learning rate for the direct phase.
        radius : float
            Gaussian repulsion radius.
        magnitude : float or None
            Gaussian repulsion magnitude. ``None`` triggers the adaptive
            formula ``100 * N^(1/3) * radius``.
        use_gcn : bool
            Whether to run the GCN reparameterization phase.
        gcn_steps : int
            Number of GCN optimization steps.
        total_steps : int
            Total optimization budget across both phases.
        """
        self._dim = dim
        self._lr = lr
        self._radius = radius
        self._magnitude = magnitude
        self._use_gcn = use_gcn
        self._gcn_steps = gcn_steps
        self._total_steps = total_steps

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Clean edges and store resolved hyperparameters in extras.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state receiving NeuLay extras.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with NeuLay configuration in ``extras``.
        """
        del ctx

        device = _layout_device(problem.edge_index, problem.node_sizes)
        cleaned = _clean_edge_index(edge_index=problem.edge_index, device=device)

        magnitude = self._magnitude
        if magnitude is None:
            magnitude = 100.0 * (problem.num_nodes ** (1.0 / 3.0)) * self._radius

        linear_steps = (
            max(self._total_steps - self._gcn_steps, 0) if self._use_gcn else self._total_steps
        )

        state.extras["neulay_cleaned_edge_index"] = cleaned
        state.extras["neulay_device"] = device
        state.extras["neulay_dim"] = self._dim
        state.extras["neulay_lr"] = self._lr
        state.extras["neulay_radius"] = self._radius
        state.extras["neulay_magnitude"] = magnitude
        state.extras["neulay_use_gcn"] = self._use_gcn
        state.extras["neulay_gcn_steps"] = self._gcn_steps
        state.extras["neulay_linear_steps"] = linear_steps
        state.extras["neulay_query_radius"] = _PAIR_QUERY_RADIUS_FACTOR * self._radius
        return state


class _GCNPhase(Op):
    """Run the optional NeuLay GCN reparameterization phase.

    This op encapsulates the entire GCN training loop to match the classic
    implementation exactly. It creates the ``_ResGCN`` model, trains it with
    RMSprop, and writes the resulting coordinates to ``state.pos``.
    """

    name: ClassVar[str] = "neulay_gcn_phase"
    category: ClassVar[OpCategory] = OpCategory.INIT
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Train the GCN and produce coarse coordinates.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with ``pos`` set to the GCN output coordinates.
        """
        del ctx

        cleaned = state.extras["neulay_cleaned_edge_index"]
        device = state.extras["neulay_device"]
        dim = state.extras["neulay_dim"]
        radius = state.extras["neulay_radius"]
        magnitude = state.extras["neulay_magnitude"]
        gcn_steps = state.extras["neulay_gcn_steps"]
        use_gcn = state.extras["neulay_use_gcn"]
        query_radius = state.extras["neulay_query_radius"]

        if use_gcn and gcn_steps > 0:
            model = _ResGCN(
                num_nodes=problem.num_nodes,
                dim=dim,
                device=device,
                edge_index=cleaned,
            )
            optimizer = torch.optim.RMSprop(model.parameters(), lr=_GNN_LR)
            loss_window = [0.0] * _PATIENCE
            pairs = np.empty((0, 2), dtype=np.int64)

            for step in range(gcn_steps):
                optimizer.zero_grad(set_to_none=True)
                output = model()
                if step % _PAIR_REFRESH_INTERVAL == 0:
                    pairs = _kdtree_repulsion_pairs(output, query_radius)
                loss = _elastic_loss(output, cleaned) + _kdtree_repulsion_loss(
                    output,
                    pairs=pairs,
                    radius=radius,
                    magnitude=magnitude,
                )
                loss.backward()
                optimizer.step()
                loss_window.append(float(loss.detach().item()))
                loss_window.pop(0)
                if _relative_window_difference(loss_window) < (
                    _GCN_REL_TOL * math.sqrt(float(problem.num_nodes))
                ):
                    break

            with torch.no_grad():
                state.pos = model().detach()
        else:
            state.pos = _initial_positions(
                num_nodes=problem.num_nodes,
                dim=dim,
                device=device,
            )

        return state


class _InitDirectPhaseOptimizer(Op):
    """Initialize the RMSprop optimizer and loss window for the direct phase."""

    name: ClassVar[str] = "neulay_init_direct_optimizer"
    category: ClassVar[OpCategory] = OpCategory.INIT
    reads: ClassVar[Tuple[str, ...]] = ("pos", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("optimizer", "extras")
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Create the RMSprop optimizer over a fresh Parameter clone.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state containing ``pos`` from the GCN phase.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with ``optimizer`` and direct-phase bookkeeping in extras.

        Raises
        ------
        ValueError
            If ``state.pos`` is missing.
        """
        del problem, ctx

        if state.pos is None:
            raise ValueError("_InitDirectPhaseOptimizer requires state.pos.")

        lr = state.extras["neulay_lr"]
        pos_param = nn.Parameter(state.pos.clone())
        state.pos = pos_param
        state.optimizer = torch.optim.RMSprop([pos_param], lr=lr)
        state.extras["neulay_loss_window"] = [0.0] * _PATIENCE
        state.extras["neulay_pairs"] = np.empty((0, 2), dtype=np.int64)
        return state


class _DirectPhaseStep(Op):
    """Execute one step of the NeuLay direct refinement phase."""

    name: ClassVar[str] = "neulay_direct_step"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("pos", "optimizer", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("pos", "extras")
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute loss, backprop, and apply one RMSprop step.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state with positions and optimizer.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with updated positions and loss window.

        Raises
        ------
        ValueError
            If ``state.pos`` or ``state.optimizer`` is missing.
        """
        del problem, ctx

        if state.pos is None:
            raise ValueError("_DirectPhaseStep requires state.pos.")
        if state.optimizer is None:
            raise ValueError("_DirectPhaseStep requires state.optimizer.")

        cleaned = state.extras["neulay_cleaned_edge_index"]
        radius = state.extras["neulay_radius"]
        magnitude = state.extras["neulay_magnitude"]
        query_radius = state.extras["neulay_query_radius"]
        loss_window = state.extras["neulay_loss_window"]
        pairs = state.extras["neulay_pairs"]

        state.optimizer.zero_grad(set_to_none=True)
        if state.step % _PAIR_REFRESH_INTERVAL == 0:
            pairs = _kdtree_repulsion_pairs(state.pos, query_radius)
            state.extras["neulay_pairs"] = pairs

        loss = _elastic_loss(state.pos, cleaned) + _kdtree_repulsion_loss(
            state.pos,
            pairs=pairs,
            radius=radius,
            magnitude=magnitude,
        )
        loss.backward()
        state.optimizer.step()

        loss_window.append(float(loss.detach().item()))
        loss_window.pop(0)
        state.extras["neulay_loss_window"] = loss_window
        return state


class _DirectPhaseConvergenceCheck(Op):
    """Check the NeuLay sliding-window convergence criterion."""

    name: ClassVar[str] = "neulay_direct_convergence"
    category: ClassVar[OpCategory] = OpCategory.CONVERGE
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Set ``state.converged`` when the loss window stabilizes.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with loss window in extras.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with ``converged`` potentially set to ``True``.
        """
        del ctx

        loss_window = state.extras["neulay_loss_window"]
        threshold = _LINEAR_REL_TOL * math.sqrt(float(problem.num_nodes))
        if _relative_window_difference(loss_window) < threshold:
            state.converged = True
        return state


class _FinalizeNeuLayPositions(Op):
    """Detach the final positions from the autograd graph."""

    name: ClassVar[str] = "neulay_finalize_positions"
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
        """Detach positions to match classic ``_optimize_positions`` return.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state containing final positions.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with detached positions.

        Raises
        ------
        ValueError
            If ``state.pos`` is missing.
        """
        del problem, ctx

        if state.pos is None:
            raise ValueError("_FinalizeNeuLayPositions requires state.pos.")
        state.pos = state.pos.detach()
        return state


# ---------------------------------------------------------------------------
# Public pipeline constructors
# ---------------------------------------------------------------------------


def build_neulay_pipeline(
    steps: int = 20_000,
    gcn_steps: int = 2_000,
    use_gcn: bool = True,
    dim: int = 2,
    lr: float = 0.01,
    radius: float = 0.4,
    magnitude: Optional[float] = None,
) -> Pipeline:
    """Build a NeuLay pipeline that is bit-identical to classic ``layout_neulay``.

    Parameters
    ----------
    steps : int, default=20000
        Total optimization budget across both phases.
    gcn_steps : int, default=2000
        Number of GCN reparameterization steps.
    use_gcn : bool, default=True
        Whether to run the optional GCN phase.
    dim : int, default=2
        Output embedding dimensionality.
    lr : float, default=0.01
        RMSprop learning rate for the direct phase.
    radius : float, default=0.4
        Gaussian repulsion radius.
    magnitude : float or None, default=None
        Gaussian repulsion magnitude. ``None`` triggers the adaptive formula.

    Returns
    -------
    Pipeline
        Pipeline that reproduces classic NeuLay's GCN phase, direct
        RMSprop refinement, KD-tree repulsion, and convergence check.

    Raises
    ------
    ValueError
        If ``steps`` or ``gcn_steps`` is negative.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if gcn_steps < 0:
        raise ValueError("gcn_steps must be non-negative.")

    linear_steps = max(steps - gcn_steps, 0) if use_gcn else steps

    return Pipeline(
        [
            FixedSteps(FixedStepsConfig(n=steps)),
            _SeedRNG(),
            _PrepareNeuLayState(
                dim=dim,
                lr=lr,
                radius=radius,
                magnitude=magnitude,
                use_gcn=use_gcn,
                gcn_steps=gcn_steps,
                total_steps=steps,
            ),
            _GCNPhase(),
            _InitDirectPhaseOptimizer(),
            Repeat(
                n=linear_steps,
                ops=[
                    _DirectPhaseStep(),
                    _DirectPhaseConvergenceCheck(),
                ],
            ),
            _FinalizeNeuLayPositions(),
        ],
        name="neulay_pipeline",
    )


def layout_neulay_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: int = 42,
    steps: int = 20_000,
    gcn_steps: int = 2_000,
    use_gcn: bool = True,
    dim: int = 2,
    lr: float = 0.01,
    radius: float = 0.4,
    magnitude: Optional[float] = None,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the NeuLay pipeline as a drop-in replacement for classic ``layout_neulay``.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor | None, default=None
        Optional node-size tensor (unused, kept for API compatibility).
    seed : int, default=42
        Random seed for initialization and optimizer trajectory.
    steps : int, default=20000
        Total optimization budget across both phases.
    gcn_steps : int, default=2000
        Number of GCN reparameterization steps.
    use_gcn : bool, default=True
        Whether to run the optional GCN phase.
    dim : int, default=2
        Output embedding dimensionality.
    lr : float, default=0.01
        RMSprop learning rate for the direct phase.
    radius : float, default=0.4
        Gaussian repulsion radius.
    magnitude : float or None, default=None
        Gaussian repulsion magnitude. ``None`` triggers the adaptive formula.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``. Accepted for interface
        consistency.

    Returns
    -------
    torch.Tensor
        Coordinate tensor with shape ``[N, dim]``, bit-identical to classic
        ``layout_neulay``.

    Raises
    ------
    ValueError
        If inputs are invalid.
    RuntimeError
        If the pipeline fails to produce positions.
    """
    _validate_inputs(
        edge_index=edge_index,
        num_nodes=num_nodes,
        steps=steps,
        gcn_steps=gcn_steps,
        dim=dim,
        lr=lr,
        radius=radius,
        magnitude=magnitude,
        edge_weights=edge_weights,
    )
    device = _layout_device(edge_index=edge_index, node_sizes=node_sizes)

    # Handle trivial cases exactly like classic.
    if num_nodes == 0:
        return torch.empty((0, dim), dtype=torch.float32, device=device)
    if num_nodes == 1:
        return torch.zeros((1, dim), dtype=torch.float32, device=device)

    # When linear_steps is 0, classic returns the GCN output directly.
    linear_steps = max(steps - gcn_steps, 0) if use_gcn else steps
    if use_gcn and linear_steps <= 0:
        # GCN-only path: seed, clean, run GCN, return.
        _set_seed(seed)
        cleaned = _clean_edge_index(edge_index=edge_index, device=device)
        if magnitude is None:
            magnitude = 100.0 * (num_nodes ** (1.0 / 3.0)) * radius
        from dagua.layout.classic.neulay import _optimize_gcn_phase

        return _optimize_gcn_phase(
            edge_index=cleaned,
            num_nodes=num_nodes,
            dim=dim,
            device=device,
            steps=gcn_steps,
            radius=radius,
            magnitude=magnitude,
        )

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_neulay_pipeline(
        steps=steps,
        gcn_steps=gcn_steps,
        use_gcn=use_gcn,
        dim=dim,
        lr=lr,
        radius=radius,
        magnitude=magnitude,
    ).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("NeuLay pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_neulay_pipeline", "layout_neulay_pipeline"]
