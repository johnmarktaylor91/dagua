"""GraphOpt expressed as a composable ops pipeline."""

from __future__ import annotations

from typing import ClassVar, Optional, Tuple

import torch

from dagua.layout.classic.graphopt import (
    _COULOMBS_CONSTANT,
    _MAX_REPULSION_DISTANCE,
    _MIN_DISTANCE,
    _initialize_positions,
    _layout_device,
    _spring_edges,
    _validate_inputs,
)
from dagua.layout.ops.base import Op, Pipeline, Repeat
from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory

_SPRING_EDGES_KEY = "graphopt_spring_edges"
_SPRING_WEIGHTS_KEY = "graphopt_spring_weights"
_PAIR_SOURCE_KEY = "graphopt_pair_source"
_PAIR_TARGET_KEY = "graphopt_pair_target"
_MAX_REPULSION_DISTANCE_SQ_KEY = "graphopt_max_repulsion_distance_sq"


class _InitializeGraphOptPositions(Op):
    """Initialize positions exactly like classic GraphOpt."""

    name: ClassVar[str] = "graphopt_initialize_positions"
    category: ClassVar[OpCategory] = OpCategory.INIT
    writes: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Seed ``state.pos`` from Python's ``random.Random`` initialization.

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
            State with ``state.pos`` populated as a CPU ``float64`` tensor.
        """
        del ctx

        if problem.num_nodes == 0:
            state.pos = torch.empty((0, 2), dtype=torch.float64)
            return state

        state.pos = _initialize_positions(num_nodes=problem.num_nodes, seed=problem.seed)
        return state


class _PrepareGraphOptState(Op):
    """Populate cached pair indices and spring metadata for GraphOpt."""

    name: ClassVar[str] = "graphopt_prepare_state"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Store the exact auxiliary tensors used by classic GraphOpt.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state receiving cached tensors.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with GraphOpt-specific tensors stored in ``extras``.
        """
        del ctx

        spring_edges, spring_weights = _spring_edges(
            edge_index=problem.edge_index,
            edge_weights=problem.edge_weights,
        )
        pair_source, pair_target = torch.triu_indices(
            problem.num_nodes,
            problem.num_nodes,
            offset=1,
        )
        state.extras[_SPRING_EDGES_KEY] = spring_edges
        state.extras[_SPRING_WEIGHTS_KEY] = spring_weights
        state.extras[_PAIR_SOURCE_KEY] = pair_source
        state.extras[_PAIR_TARGET_KEY] = pair_target
        state.extras[_MAX_REPULSION_DISTANCE_SQ_KEY] = (
            _MAX_REPULSION_DISTANCE * _MAX_REPULSION_DISTANCE
        )
        return state


class _GraphOptIteration(Op):
    """Execute one exact GraphOpt force-and-move iteration."""

    name: ClassVar[str] = "graphopt_iteration"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("pos", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def __init__(
        self,
        *,
        node_charge: float,
        node_mass: float,
        spring_length: float,
        spring_constant: float,
        max_sa_movement: float,
    ) -> None:
        """Store the immutable GraphOpt force parameters.

        Parameters
        ----------
        node_charge : float
            Charge used by the Coulomb repulsion term.
        node_mass : float
            Shared mass used to convert force into movement.
        spring_length : float
            Rest length of edge springs.
        spring_constant : float
            Hooke-law spring constant.
        max_sa_movement : float
            Maximum per-axis movement allowed in one iteration.

        Returns
        -------
        None
            This constructor stores configuration only.
        """
        self.node_charge = float(node_charge)
        self.node_mass = float(node_mass)
        self.spring_length = float(spring_length)
        self.spring_constant = float(spring_constant)
        self.max_sa_movement = float(max_sa_movement)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Apply GraphOpt's exact synchronous repulsion and spring update.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state containing the current positions.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with ``state.pos`` advanced by one GraphOpt iteration.

        Raises
        ------
        ValueError
            If ``state.pos`` is missing.
        """
        del ctx

        if state.pos is None:
            raise ValueError("_GraphOptIteration requires state.pos to be set.")

        positions = state.pos
        num_nodes = problem.num_nodes
        forces = torch.zeros((num_nodes, 2), dtype=torch.float64)
        spring_edges = state.extras[_SPRING_EDGES_KEY]
        spring_weights = state.extras[_SPRING_WEIGHTS_KEY]
        pair_source = state.extras[_PAIR_SOURCE_KEY]
        pair_target = state.extras[_PAIR_TARGET_KEY]
        max_repulsion_distance_sq = state.extras[_MAX_REPULSION_DISTANCE_SQ_KEY]

        if self.node_charge != 0.0 and num_nodes > 1:
            delta = positions[pair_source] - positions[pair_target]
            distance_sq = delta.square().sum(dim=1)
            mask = (distance_sq > _MIN_DISTANCE) & (distance_sq < max_repulsion_distance_sq)
            if bool(mask.any().item()):
                pair_delta = delta[mask]
                pair_distance_sq = distance_sq[mask]
                pair_distance = torch.sqrt(pair_distance_sq)
                direction = pair_delta / pair_distance.unsqueeze(1)
                magnitude = _COULOMBS_CONSTANT * (self.node_charge * self.node_charge)
                magnitude = magnitude / pair_distance_sq
                contribution = direction * magnitude.unsqueeze(1)
                forces.index_add_(0, pair_source[mask], contribution)
                forces.index_add_(0, pair_target[mask], -contribution)

        if spring_edges.numel() > 0:
            source = spring_edges[0]
            target = spring_edges[1]
            delta = positions[source] - positions[target]
            distance = torch.linalg.norm(delta, dim=1)
            mask = distance > _MIN_DISTANCE
            if bool(mask.any().item()):
                masked_distance = distance[mask]
                direction = delta[mask] / masked_distance.unsqueeze(1)
                stretch = (masked_distance - self.spring_length).abs()
                magnitude = 0.5 * self.spring_constant * stretch
                if spring_weights is not None:
                    magnitude = magnitude * spring_weights[mask].to(dtype=magnitude.dtype)
                source_sign = torch.where(
                    masked_distance > self.spring_length,
                    torch.full_like(masked_distance, -1.0),
                    torch.full_like(masked_distance, 1.0),
                )
                contribution = direction * (magnitude * source_sign).unsqueeze(1)
                forces.index_add_(0, source[mask], contribution)
                forces.index_add_(0, target[mask], -contribution)

        movement = torch.clamp(
            forces / self.node_mass,
            min=-self.max_sa_movement,
            max=self.max_sa_movement,
        )
        state.pos = positions + movement
        return state


class _FinalizeGraphOptPositions(Op):
    """Cast GraphOpt positions to the classic output dtype and device."""

    name: ClassVar[str] = "graphopt_finalize_positions"
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
        """Move final positions onto the classic return device.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state containing final ``float64`` positions.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with final output positions.

        Raises
        ------
        ValueError
            If ``state.pos`` is missing.
        """
        del ctx

        if state.pos is None:
            raise ValueError("_FinalizeGraphOptPositions requires state.pos to be set.")

        output_device = _layout_device(
            edge_index=problem.edge_index,
            node_sizes=problem.node_sizes,
        )
        if problem.num_nodes == 0:
            state.pos = torch.empty((0, 2), dtype=torch.float32, device=output_device)
            return state

        state.pos = state.pos.to(dtype=torch.float32, device=output_device)
        return state


def build_graphopt_pipeline(
    niter: int = 500,
    node_charge: float = 0.001,
    node_mass: float = 30.0,
    spring_length: float = 0.0,
    spring_constant: float = 1.0,
    max_sa_movement: float = 5.0,
) -> Pipeline:
    """Build a GraphOpt pipeline that is bit-identical to ``layout_graphopt``.

    Parameters
    ----------
    niter : int, default=500
        Number of GraphOpt iterations.
    node_charge : float, default=0.001
        Charge used by the Coulomb repulsion term.
    node_mass : float, default=30.0
        Shared mass used to convert force into motion.
    spring_length : float, default=0.0
        Rest length of the spring attraction term.
    spring_constant : float, default=1.0
        Hooke-law spring constant.
    max_sa_movement : float, default=5.0
        Maximum per-axis movement allowed in one iteration.

    Returns
    -------
    Pipeline
        Pipeline that reproduces classic GraphOpt exactly.

    Raises
    ------
    ValueError
        If ``niter``, ``node_mass``, or ``max_sa_movement`` are invalid.
    """
    if niter < 0:
        raise ValueError("niter must be non-negative.")
    if node_mass <= 0.0:
        raise ValueError("node_mass must be positive.")
    if max_sa_movement < 0.0:
        raise ValueError("max_sa_movement must be non-negative.")

    return Pipeline(
        [
            FixedSteps(FixedStepsConfig(n=niter)),
            _InitializeGraphOptPositions(),
            _PrepareGraphOptState(),
            Repeat(
                n=niter,
                ops=[
                    _GraphOptIteration(
                        node_charge=node_charge,
                        node_mass=node_mass,
                        spring_length=spring_length,
                        spring_constant=spring_constant,
                        max_sa_movement=max_sa_movement,
                    ),
                ],
            ),
            _FinalizeGraphOptPositions(),
        ],
        name="graphopt_pipeline",
    )


def layout_graphopt_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: int = 42,
    niter: int = 500,
    node_charge: float = 0.001,
    node_mass: float = 30.0,
    spring_length: float = 0.0,
    spring_constant: float = 1.0,
    max_sa_movement: float = 5.0,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the GraphOpt pipeline as a drop-in replacement for classic GraphOpt.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor used only to resolve the output device.
    seed : int, default=42
        Random seed for the Python ``random.Random`` initialization.
    niter : int, default=500
        Number of GraphOpt iterations.
    node_charge : float, default=0.001
        Charge used by the Coulomb repulsion term.
    node_mass : float, default=30.0
        Shared mass used to convert force into motion.
    spring_length : float, default=0.0
        Rest length of the spring attraction term.
    spring_constant : float, default=1.0
        Hooke-law spring constant.
    max_sa_movement : float, default=5.0
        Maximum per-axis movement allowed in one iteration.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Final position tensor with the same dtype, device, and values as
        classic ``layout_graphopt``.

    Raises
    ------
    ValueError
        If the public GraphOpt inputs are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    _validate_inputs(
        edge_index=edge_index,
        num_nodes=num_nodes,
        niter=niter,
        node_mass=node_mass,
        max_sa_movement=max_sa_movement,
        edge_weights=edge_weights,
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
    final_state = build_graphopt_pipeline(
        niter=niter,
        node_charge=node_charge,
        node_mass=node_mass,
        spring_length=spring_length,
        spring_constant=spring_constant,
        max_sa_movement=max_sa_movement,
    ).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("GraphOpt pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_graphopt_pipeline", "layout_graphopt_pipeline"]
