"""GraphOpt force-directed layout pipeline."""

from __future__ import annotations

from typing import Optional

import torch

from dagua.layout.ops.base import Pipeline, Repeat
from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig
from dagua.layout.ops.force import (
    GraphOptIteration,
    GraphOptIterationConfig,
    GraphOptPrepareState,
    ZeroForces,
)
from dagua.layout.ops.init import GraphOptInitializePositions, ValidateGraphOptInputs
from dagua.layout.ops.postprocess import GraphOptFinalizePositions
from dagua.layout.ops.state import (
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)


def build_graphopt_pipeline(
    niter: int = 500,
    node_charge: float = 0.001,
    node_mass: float = 30.0,
    spring_length: float = 0.0,
    spring_constant: float = 1.0,
    max_sa_movement: float = 5.0,
) -> Pipeline:
    """Build a GraphOpt force-directed layout pipeline.

    Parameters
    ----------
    niter : int, default=500
        Number of GraphOpt iterations.
    node_charge : float, default=0.001
        Coulomb charge used by repulsion.
    node_mass : float, default=30.0
        Shared mass used to convert force to movement.
    spring_length : float, default=0.0
        Rest length of the spring attraction term.
    spring_constant : float, default=1.0
        Spring constant for the attraction term.
    max_sa_movement : float, default=5.0
        Maximum per-axis movement allowed in one step.

    Returns
    -------
    Pipeline
        Pipeline implementing the GraphOpt algorithm. The pipeline produces
        final node coordinates by validating inputs, initializing positions,
        preparing force state, clearing force accumulators, applying repeated
        spring-and-repulsion iterations, and finalizing the layout.

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

    iteration = GraphOptIteration(
        config=GraphOptIterationConfig(
            node_charge=node_charge,
            node_mass=node_mass,
            spring_length=spring_length,
            spring_constant=spring_constant,
            max_sa_movement=max_sa_movement,
        )
    )

    return Pipeline(
        [
            FixedSteps(FixedStepsConfig(n=niter)),
            ValidateGraphOptInputs(),
            GraphOptInitializePositions(),
            GraphOptPrepareState(),
            Repeat(
                n=niter,
                ops=[
                    ZeroForces(),
                    iteration,
                ],
            ),
            GraphOptFinalizePositions(),
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
    """Run the GraphOpt force-directed layout pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]`` used only to resolve
        the output device.
    seed : int, default=42
        Random seed for ``random.Random`` initialization.
    niter : int, default=500
        Number of GraphOpt iterations.
    node_charge : float, default=0.001
        Coulomb charge used by repulsion.
    node_mass : float, default=30.0
        Shared mass used to convert force into movement.
    spring_length : float, default=0.0
        Rest length of spring forces.
    spring_constant : float, default=1.0
        Spring constant for GraphOpt springs.
    max_sa_movement : float, default=5.0
        Maximum per-axis movement allowed in one iteration.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If public GraphOpt inputs are invalid.
    RuntimeError
        If the pipeline does not produce final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if niter < 0:
        raise ValueError("niter must be non-negative.")
    if node_mass <= 0.0:
        raise ValueError("node_mass must be positive.")
    if max_sa_movement < 0.0:
        raise ValueError("max_sa_movement must be non-negative.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
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
