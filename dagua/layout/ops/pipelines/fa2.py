"""ForceAtlas2 expressed as a composable ops pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple

import torch

from dagua.layout.classic.fa2 import (
    _adjust_speed_and_apply_forces,
    _attraction_force,
    _barnes_hut_repulsion,
    _compute_degree,
    _gravity_force,
    _repulsion_force,
    _unique_undirected_edges_with_weights,
    _validate_inputs,
)
from dagua.layout.ops.base import Op, Pipeline, Repeat
from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory

# ---------------------------------------------------------------------------
# FA2 configuration dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FA2Config:
    """Configuration for the ForceAtlas2 pipeline.

    Attributes
    ----------
    steps : int
        Number of FA2 iterations.
    gravity : float
        Gravity coefficient.
    scaling_ratio : float
        Repulsion scaling coefficient.
    linlog : bool
        Whether to use log-attraction.
    strong_gravity : bool
        Whether to use strong-gravity mode.
    outbound_attraction_distribution : bool
        Whether to normalize attraction by source mass.
    dissuade_hubs : bool
        Whether to further penalize hub attraction.
    edge_weight_influence : float
        Exponent applied to edge weights.
    barnes_hut : bool
        Whether to use Barnes-Hut approximation for repulsion.
    barnes_hut_theta : float
        Acceptance threshold for Barnes-Hut.
    jitter_tolerance : float
        Jitter tolerance for adaptive speed control.
    """

    steps: int = 100
    gravity: float = 1.0
    scaling_ratio: float = 2.0
    linlog: bool = False
    strong_gravity: bool = False
    outbound_attraction_distribution: bool = True
    dissuade_hubs: bool = False
    edge_weight_influence: float = 1.0
    barnes_hut: bool = False
    barnes_hut_theta: float = 1.2
    jitter_tolerance: float = 1.0


# ---------------------------------------------------------------------------
# Pipeline-local ops
# ---------------------------------------------------------------------------


class _InitializeFA2Positions(Op):
    """Initialize positions exactly like classic FA2.

    Uses Python's ``random.random()`` seeded identically to the reference
    ``fa2_modified`` implementation.
    """

    name: ClassVar[str] = "fa2_initialize_positions"
    category: ClassVar[OpCategory] = OpCategory.INIT
    writes: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Seed ``state.pos`` from Python random matching fa2_modified init.

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
            State with ``state.pos`` populated as a ``float32`` tensor.
        """
        del ctx

        import random as _random

        device = problem.edge_index.device
        _random.seed(problem.seed)
        state.pos = torch.tensor(
            [[_random.random(), _random.random()] for _ in range(problem.num_nodes)],
            dtype=torch.float32,
            device=device,
        )
        return state


class _PrepareFA2State(Op):
    """Populate FA2-specific cached state required by force steps.

    Computes undirected edges, masses, outbound attraction compensation,
    and initializes adaptive speed control scalars.
    """

    name: ClassVar[str] = "fa2_prepare_state"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    _config: FA2Config

    def __init__(self, config: FA2Config) -> None:
        """Store the FA2 configuration.

        Parameters
        ----------
        config : FA2Config
            FA2 parameters for this pipeline instance.
        """
        self._config = config

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build undirected edges, masses, and adaptive speed state.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state receiving FA2 extras.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with FA2 adjacency, masses, and speed control in extras.
        """
        del ctx

        undirected_edges, undirected_weights = _unique_undirected_edges_with_weights(
            edge_index=problem.edge_index,
            edge_weights=problem.edge_weights,
        )
        masses = _compute_degree(undirected_edges, num_nodes=problem.num_nodes) + 1.0
        outbound_att_compensation = (
            float(masses.mean().item()) if self._config.outbound_attraction_distribution else 1.0
        )

        state.extras["fa2_undirected_edges"] = undirected_edges
        state.extras["fa2_undirected_weights"] = undirected_weights
        state.extras["fa2_masses"] = masses
        state.extras["fa2_outbound_att_compensation"] = outbound_att_compensation
        state.extras["fa2_speed"] = 1.0
        state.extras["fa2_speed_efficiency"] = 1.0
        state.extras["fa2_old_force"] = None  # set after first iteration
        return state


class _FA2ForceStep(Op):
    """Compute one iteration of FA2 forces and apply adaptive speed update.

    Accumulates repulsion + gravity + attraction, then calls the classic
    adaptive speed controller to update positions, speed, and speed
    efficiency.
    """

    name: ClassVar[str] = "fa2_force_step"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("pos", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("pos", "extras")
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    _config: FA2Config

    def __init__(self, config: FA2Config) -> None:
        """Store the FA2 configuration.

        Parameters
        ----------
        config : FA2Config
            FA2 parameters for this pipeline instance.
        """
        self._config = config

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute forces and apply adaptive speed update.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with positions and FA2 extras.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with updated positions and speed control scalars.

        Raises
        ------
        ValueError
            If ``state.pos`` is missing.
        """
        del ctx

        if state.pos is None:
            raise ValueError("_FA2ForceStep requires state.pos to be set.")

        pos = state.pos
        undirected_edges = state.extras["fa2_undirected_edges"]
        undirected_weights = state.extras["fa2_undirected_weights"]
        masses = state.extras["fa2_masses"]
        outbound_att_compensation = state.extras["fa2_outbound_att_compensation"]
        speed = state.extras["fa2_speed"]
        speed_efficiency = state.extras["fa2_speed_efficiency"]
        old_force_or_none = state.extras["fa2_old_force"]

        if old_force_or_none is None:
            old_force = torch.zeros_like(pos)
        else:
            old_force = old_force_or_none

        # Accumulate forces
        force = torch.zeros_like(pos)
        if self._config.barnes_hut:
            force = force + _barnes_hut_repulsion(
                pos=pos,
                mass=masses,
                scaling_ratio=self._config.scaling_ratio,
                theta=self._config.barnes_hut_theta,
            )
        else:
            force = force + _repulsion_force(
                pos=pos,
                mass=masses,
                scaling_ratio=self._config.scaling_ratio,
            )
        force = force + _gravity_force(
            pos=pos,
            mass=masses,
            gravity=self._config.gravity,
            strong_gravity=self._config.strong_gravity,
            scaling_ratio=self._config.scaling_ratio,
        )
        force = force + _attraction_force(
            pos=pos,
            edge_index=undirected_edges,
            mass=masses,
            outbound_att_compensation=outbound_att_compensation,
            outbound_attraction_distribution=self._config.outbound_attraction_distribution,
            linlog=self._config.linlog,
            edge_weights=undirected_weights,
            dissuade_hubs=self._config.dissuade_hubs,
            edge_weight_influence=self._config.edge_weight_influence,
        )

        # Adaptive speed and position update
        new_pos, new_speed, new_speed_efficiency = _adjust_speed_and_apply_forces(
            pos=pos,
            force=force,
            old_force=old_force,
            mass=masses,
            speed=speed,
            speed_efficiency=speed_efficiency,
            jitter_tolerance=self._config.jitter_tolerance,
        )

        state.pos = new_pos
        state.extras["fa2_speed"] = new_speed
        state.extras["fa2_speed_efficiency"] = new_speed_efficiency
        state.extras["fa2_old_force"] = force
        state.step += 1
        return state


# ---------------------------------------------------------------------------
# Pipeline builder and convenience entry point
# ---------------------------------------------------------------------------


def build_fa2_pipeline(config: Optional[FA2Config] = None) -> Pipeline:
    """Build an FA2 pipeline that is bit-identical to classic ``layout_fa2``.

    Parameters
    ----------
    config : FA2Config, optional
        FA2 configuration. Uses defaults when not provided.

    Returns
    -------
    Pipeline
        Pipeline that reproduces classic FA2's initialization, force
        computation, adaptive speed control, and all mode flags.

    Raises
    ------
    ValueError
        If ``config.steps`` is negative.
    """
    if config is None:
        config = FA2Config()
    if config.steps < 0:
        raise ValueError("steps must be non-negative.")

    return Pipeline(
        [
            FixedSteps(FixedStepsConfig(n=config.steps)),
            _InitializeFA2Positions(),
            _PrepareFA2State(config=config),
            Repeat(
                n=config.steps,
                ops=[
                    _FA2ForceStep(config=config),
                ],
            ),
        ],
        name="fa2_pipeline",
    )


def layout_fa2_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 100,
    seed: int = 42,
    gravity: float = 1.0,
    scaling_ratio: float = 2.0,
    linlog: bool = False,
    strong_gravity: bool = False,
    outbound_attraction_distribution: bool = True,
    edge_weights: Optional[torch.Tensor] = None,
    dissuade_hubs: bool = False,
    edge_weight_influence: float = 1.0,
    barnes_hut: bool = False,
    barnes_hut_theta: float = 1.2,
) -> torch.Tensor:
    """Run the FA2 pipeline as a drop-in replacement for classic ``layout_fa2``.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor. Unused, kept for API compatibility.
    steps : int, default=100
        Number of FA2 iterations.
    seed : int, default=42
        Random seed for the Python-random initialization.
    gravity : float, default=1.0
        Gravity coefficient.
    scaling_ratio : float, default=2.0
        Repulsion scaling coefficient.
    linlog : bool, default=False
        Whether to use log-attraction.
    strong_gravity : bool, default=False
        Whether to use strong-gravity mode.
    outbound_attraction_distribution : bool, default=True
        Whether to normalize attraction by source mass.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.
    dissuade_hubs : bool, default=False
        Whether to further penalize hub attraction.
    edge_weight_influence : float, default=1.0
        Exponent applied to edge weights.
    barnes_hut : bool, default=False
        Whether to use Barnes-Hut approximation for repulsion.
    barnes_hut_theta : float, default=1.2
        Acceptance threshold for Barnes-Hut.

    Returns
    -------
    torch.Tensor
        Final position tensor with the same dtype, device, and values as
        classic ``layout_fa2``.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``steps``, or ``edge_weights`` are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    del node_sizes

    # Validate using the same function as classic FA2
    _validate_inputs(
        edge_index=edge_index,
        num_nodes=num_nodes,
        steps=steps,
        trace_every=0,
        edge_weights=edge_weights,
        barnes_hut_theta=barnes_hut_theta,
    )

    # Handle degenerate cases identically to classic
    device = edge_index.device
    if num_nodes == 0:
        return torch.zeros((0, 2), dtype=torch.float32, device=device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32, device=device)

    config = FA2Config(
        steps=steps,
        gravity=gravity,
        scaling_ratio=scaling_ratio,
        linlog=linlog,
        strong_gravity=strong_gravity,
        outbound_attraction_distribution=outbound_attraction_distribution,
        dissuade_hubs=dissuade_hubs,
        edge_weight_influence=edge_weight_influence,
        barnes_hut=barnes_hut,
        barnes_hut_theta=barnes_hut_theta,
    )

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_fa2_pipeline(config=config).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("FA2 pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["FA2Config", "build_fa2_pipeline", "layout_fa2_pipeline"]
