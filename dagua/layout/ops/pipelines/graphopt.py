"""GraphOpt force-directed layout pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional, Tuple, cast

import torch

from dagua.layout.ops.base import Op, Pipeline, Repeat
from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig
from dagua.layout.ops.force import (
    GraphOptIteration,
    GraphOptIterationConfig,
    GraphOptPrepareState,
    GraphOptPrepareStateConfig,
    ZeroForces,
)
from dagua.layout.ops.init import (
    GRAPHOPT_INITIAL_POS_KEY,
    GraphOptInitializePositions,
    GraphOptInitializePositionsConfig,
    ValidateGraphOptInputs,
)
from dagua.layout.ops.pipelines import resolve_fidelity_dtype
from dagua.layout.ops.postprocess import GraphOptFinalizePositions
from dagua.layout.ops.state import (
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)

_GRAPHOPT_COULOMBS_CONSTANT = 8_987_500_000.0
_GRAPHOPT_MAX_REPULSION_DISTANCE = 500.0
_GRAPHOPT_SPRING_EDGES_KEY = "graphopt_spring_edges"


def _igraph_axis_forces(
    positions: list[list[float]],
    directed_force: float,
    distance: float,
    other_node: int,
    this_node: int,
) -> tuple[float, float]:
    """Resolve igraph GraphOpt's per-axis force components.

    Parameters
    ----------
    positions : list[list[float]]
        Current coordinates with shape ``[N, 2]`` represented as Python
        doubles.
    directed_force : float
        Scalar force magnitude in igraph's signed convention.
    distance : float
        Euclidean distance between ``this_node`` and ``other_node``.
    other_node : int
        Node applying the force.
    this_node : int
        Node receiving the force.

    Returns
    -------
    tuple[float, float]
        X and Y force components to add to ``this_node``.
    """
    y_distance = positions[other_node][1] - positions[this_node][1]
    if y_distance < 0.0:
        y_distance = -y_distance
    y_force = -1.0 * ((directed_force * y_distance) / distance)

    x_distance = positions[other_node][0] - positions[this_node][0]
    if x_distance < 0.0:
        x_distance = -x_distance
    x_force = -1.0 * ((directed_force * x_distance) / distance)

    if positions[other_node][0] < positions[this_node][0]:
        x_force = x_force * -1.0
    if positions[other_node][1] < positions[this_node][1]:
        y_force = y_force * -1.0
    return x_force, y_force


@dataclass(frozen=True)
class _GraphOptScalarIterationConfig:
    """Configuration for the igraph-order scalar GraphOpt iteration.

    Parameters
    ----------
    node_charge : float, default=0.001
        Coulomb repulsion charge term.
    node_mass : float, default=30.0
        Shared mass in the explicit displacement step.
    spring_length : float, default=0.0
        Rest length used by explicit springs.
    spring_constant : float, default=1.0
        Spring constant used by explicit edge forces.
    max_sa_movement : float, default=5.0
        Absolute displacement clamp per axis.
    """

    node_charge: float = 0.001
    node_mass: float = 30.0
    spring_length: float = 0.0
    spring_constant: float = 1.0
    max_sa_movement: float = 5.0


@dataclass(frozen=True)
class _GraphOptScalarIteration(Op):
    """Run one GraphOpt step using igraph C's scalar accumulation order.

    Notes
    -----
    This op is intentionally used only by ``fidelity_mode``. The tensor op is
    faster, but igraph's high-gain parameter variants amplify ULP-level drift
    from vectorized reductions over hundreds of explicit Euler steps.
    """

    config: _GraphOptScalarIterationConfig = field(default_factory=_GraphOptScalarIterationConfig)

    name: ClassVar[str] = "graphopt_scalar_iteration"
    reads: ClassVar[Tuple[str, ...]] = ("pos", f"extras.{_GRAPHOPT_SPRING_EDGES_KEY}")
    writes: ClassVar[Tuple[str, ...]] = ("pos", "forces")
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Update positions by one igraph-order scalar GraphOpt iteration.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with current positions and prepared edge cache.
        ctx : RuntimeContext
            Execution context. The scalar fidelity path is CPU-only.

        Returns
        -------
        SolveState
            State with updated ``pos`` and the most recent force buffer.
        """
        del problem, ctx

        if state.pos is None:
            raise ValueError("GraphOpt scalar iteration requires state.pos to be set.")

        spring_edges = state.extras.get(_GRAPHOPT_SPRING_EDGES_KEY)
        if not isinstance(spring_edges, torch.Tensor):
            raise ValueError(
                "GraphOpt scalar iteration requires state.extras['graphopt_spring_edges']."
            )

        positions_tensor = state.pos.detach().to(device="cpu", dtype=torch.float64).contiguous()
        positions = cast(list[list[float]], positions_tensor.tolist())
        node_count = len(positions)
        pending_x = [0.0] * node_count
        pending_y = [0.0] * node_count

        if self.config.node_charge != 0.0:
            for this_node in range(node_count):
                for other_node in range(this_node + 1, node_count):
                    distance = self._distance_between(
                        positions=positions,
                        first_node=this_node,
                        second_node=other_node,
                    )
                    if distance != 0.0 and distance < _GRAPHOPT_MAX_REPULSION_DISTANCE:
                        self._apply_electrical_force(
                            positions=positions,
                            pending_x=pending_x,
                            pending_y=pending_y,
                            other_node=other_node,
                            this_node=this_node,
                            distance=distance,
                        )

        edge_source = spring_edges[0].tolist()
        edge_target = spring_edges[1].tolist()
        for this_node, other_node in zip(edge_source, edge_target):
            self._apply_spring_force(
                positions=positions,
                pending_x=pending_x,
                pending_y=pending_y,
                other_node=int(other_node),
                this_node=int(this_node),
            )

        for this_node in range(node_count):
            x_movement = pending_x[this_node] / float(self.config.node_mass)
            if x_movement > self.config.max_sa_movement:
                x_movement = self.config.max_sa_movement
            elif x_movement < -self.config.max_sa_movement:
                x_movement = -self.config.max_sa_movement

            y_movement = pending_y[this_node] / float(self.config.node_mass)
            if y_movement > self.config.max_sa_movement:
                y_movement = self.config.max_sa_movement
            elif y_movement < -self.config.max_sa_movement:
                y_movement = -self.config.max_sa_movement

            positions[this_node][0] += x_movement
            positions[this_node][1] += y_movement

        state.forces = torch.tensor(
            [[pending_x[index], pending_y[index]] for index in range(node_count)],
            dtype=torch.float64,
        )
        state.pos = torch.tensor(positions, dtype=torch.float64)
        return state

    def _distance_between(
        self,
        positions: list[list[float]],
        first_node: int,
        second_node: int,
    ) -> float:
        """Compute igraph GraphOpt's Euclidean node distance.

        Parameters
        ----------
        positions : list[list[float]]
            Current coordinates with shape ``[N, 2]``.
        first_node : int
            First node index.
        second_node : int
            Second node index.

        Returns
        -------
        float
            Euclidean distance between the two nodes.
        """
        diff_x = positions[first_node][0] - positions[second_node][0]
        diff_y = positions[first_node][1] - positions[second_node][1]
        return math.sqrt(diff_x * diff_x + diff_y * diff_y)

    def _apply_electrical_force(
        self,
        positions: list[list[float]],
        pending_x: list[float],
        pending_y: list[float],
        other_node: int,
        this_node: int,
        distance: float,
    ) -> None:
        """Apply igraph GraphOpt's Coulomb force to pending force buffers.

        Parameters
        ----------
        positions : list[list[float]]
            Current coordinates with shape ``[N, 2]``.
        pending_x : list[float]
            Mutable pending x-force vector with shape ``[N]``.
        pending_y : list[float]
            Mutable pending y-force vector with shape ``[N]``.
        other_node : int
            Node applying the force.
        this_node : int
            Node receiving the force.
        distance : float
            Euclidean distance between the two nodes.

        Returns
        -------
        None
            Force buffers are updated in place.
        """
        directed_force = _GRAPHOPT_COULOMBS_CONSTANT * (
            (float(self.config.node_charge) * float(self.config.node_charge))
            / (distance * distance)
        )
        x_force, y_force = _igraph_axis_forces(
            positions=positions,
            directed_force=directed_force,
            distance=distance,
            other_node=other_node,
            this_node=this_node,
        )
        pending_x[this_node] += x_force
        pending_y[this_node] += y_force
        pending_x[other_node] -= x_force
        pending_y[other_node] -= y_force

    def _apply_spring_force(
        self,
        positions: list[list[float]],
        pending_x: list[float],
        pending_y: list[float],
        other_node: int,
        this_node: int,
    ) -> None:
        """Apply igraph GraphOpt's Hooke spring force to pending buffers.

        Parameters
        ----------
        positions : list[list[float]]
            Current coordinates with shape ``[N, 2]``.
        pending_x : list[float]
            Mutable pending x-force vector with shape ``[N]``.
        pending_y : list[float]
            Mutable pending y-force vector with shape ``[N]``.
        other_node : int
            Edge target node applying the spring force.
        this_node : int
            Edge source node receiving the spring force.

        Returns
        -------
        None
            Force buffers are updated in place.
        """
        distance = self._distance_between(
            positions=positions,
            first_node=other_node,
            second_node=this_node,
        )
        if distance == 0.0:
            return

        displacement = distance - float(self.config.spring_length)
        if displacement < 0.0:
            displacement = -displacement
        directed_force = -1.0 * float(self.config.spring_constant) * displacement

        if distance == float(self.config.spring_length):
            x_force = 0.0
            y_force = 0.0
        else:
            x_force, y_force = _igraph_axis_forces(
                positions=positions,
                directed_force=directed_force,
                distance=distance,
                other_node=other_node,
                this_node=this_node,
            )
            if distance < float(self.config.spring_length):
                x_force = -1.0 * x_force
                y_force = -1.0 * y_force
            x_force = 0.5 * x_force
            y_force = 0.5 * y_force

        pending_x[this_node] += x_force
        pending_y[this_node] += y_force
        pending_x[other_node] -= x_force
        pending_y[other_node] -= y_force


def build_graphopt_pipeline(
    niter: int = 500,
    node_charge: float = 0.001,
    node_mass: float = 30.0,
    spring_length: float = 0.0,
    spring_constant: float = 1.0,
    max_sa_movement: float = 5.0,
    fidelity_mode: bool = False,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> Pipeline:
    """Build a GraphOpt force-directed layout pipeline.

    Reference fidelity
    ------------------
    Targets: igraph 1.0.0 GraphOpt / igraph's Fruchterman-Reingold-derived
        GraphOpt force model.
    Fidelity mode: ``fidelity_mode=True`` uses NumPy ``[-1, 1]`` seeded
        initial positions, ignores edge weights, and applies igraph-style
        near-zero force skip predicates.
    Verified at: round_41 same-seed smoke mean Procrustes RMSD below 1e-8
        against python-igraph 1.0.0 on path, star, clustered, and grid
        topologies. All-seed-pair live-comparison summaries remain non-zero
        because they intentionally include different seed pairings.
    Known divergences:
        - Raw coordinate frames can differ by rotation/reflection/translation
          from python-igraph's returned Layout object; the fidelity metric is
          Procrustes-aligned RMSD.
        - Dagua retains optional weighted behavior outside fidelity mode.
        - High-gain parameter regimes can amplify machine-epsilon force drift
          after many explicit steps. Round 64 scale probes found this for
          ``node_mass=10`` and ``spring_constant=2`` on dense real graphs even
          though the same cases match python-igraph through the early steps.

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
    fidelity_mode : bool, default=False
        Match the igraph benchmark path by using NumPy ``[-1, 1]`` seeded
        initial positions when no explicit matrix is supplied and by ignoring
        GraphOpt edge weights.

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

    resolved_dtype = resolve_fidelity_dtype(fidelity_mode, fidelity_dtype)
    iteration_config = GraphOptIterationConfig(
        node_charge=node_charge,
        node_mass=node_mass,
        spring_length=spring_length,
        spring_constant=spring_constant,
        max_sa_movement=max_sa_movement,
    )
    scalar_iteration_config = _GraphOptScalarIterationConfig(
        node_charge=node_charge,
        node_mass=node_mass,
        spring_length=spring_length,
        spring_constant=spring_constant,
        max_sa_movement=max_sa_movement,
    )
    iteration: Op
    if fidelity_mode:
        iteration = _GraphOptScalarIteration(config=scalar_iteration_config)
    else:
        iteration = GraphOptIteration(config=iteration_config)

    return Pipeline(
        [
            FixedSteps(FixedStepsConfig(n=niter)),
            ValidateGraphOptInputs(),
            GraphOptInitializePositions(
                GraphOptInitializePositionsConfig(fidelity_mode=fidelity_mode)
            ),
            GraphOptPrepareState(GraphOptPrepareStateConfig(fidelity_mode=fidelity_mode)),
            Repeat(
                n=niter,
                ops=[
                    ZeroForces(),
                    iteration,
                ],
            ),
            GraphOptFinalizePositions(
                output_dtype=(
                    resolved_dtype
                    if fidelity_mode and fidelity_dtype is not None
                    else torch.float32
                )
            ),
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
    initial_pos: Optional[Any] = None,
    fidelity_mode: bool = False,
    fidelity_dtype: Optional[torch.dtype] = None,
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
    initial_pos : Any, optional
        Optional initial coordinate matrix with shape ``[N, 2]``. Tensor,
        NumPy array, and nested-sequence inputs are accepted and converted to
        float64 by ``GraphOptInitializePositions``.
    fidelity_mode : bool, default=False
        Match igraph benchmark semantics by using the supplied seed matrix, or
        the adapter-compatible NumPy seed matrix fallback, and by ignoring
        GraphOpt edge weights.

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
    if initial_pos is not None:
        state.extras[GRAPHOPT_INITIAL_POS_KEY] = initial_pos
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_graphopt_pipeline(
        niter=niter,
        node_charge=node_charge,
        node_mass=node_mass,
        spring_length=spring_length,
        spring_constant=spring_constant,
        max_sa_movement=max_sa_movement,
        fidelity_mode=fidelity_mode,
        fidelity_dtype=fidelity_dtype,
    ).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("GraphOpt pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_graphopt_pipeline", "layout_graphopt_pipeline"]
