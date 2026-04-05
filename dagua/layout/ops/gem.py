"""Registered operations implementing the classic GEM primitives.

These ops are intentionally faithful to ``layout.classic.gem`` and provide a
composable, registered building block surface for pipeline definitions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar, Tuple

import torch

from dagua.layout.ops.base import Op, Repeat
from dagua.layout.ops.graph_utils import (
    build_undirected_adjacency,
    layout_device,
    layout_extent,
    normalize_positions,
)
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op


@dataclass(frozen=True)
class GEMPhysicsConfig:
    """Numerical constants shared across GEM solve phases.

    Parameters
    ----------
    min_distance : float, default=1e-9
        Lower bound used to avoid zero-distance divisions.
    degree_divisor : float, default=2.5
        OGDF degree-weight divisor applied before the unit offset.
    degree_offset : float, default=1.0
        Additive offset applied after degree scaling.
    base_desired_length : float, default=20.0
        Constant edge-length bias added after node-size-derived diagonals.
    initial_temperature : float, default=12.0
        Starting local temperature per node.
    minimal_temperature : float, default=0.005
        Early-stop threshold used by both solver variants.
    gravitational_constant : float, default=1/16
        Pull strength toward the weighted barycenter.
    maximal_disturbance : float, default=0.0
        Legacy disturbance amplitude. Non-zero values remain unsupported.
    attraction_formula : int, default=1
        OGDF attraction formula selector.
    rotation_sensitivity : float, default=0.01
        Increment applied when repeated motion indicates rotation.
    oscillation_sensitivity : float, default=0.3
        Temperature scaling applied when motion oscillates.
    rotation_sine_threshold : float, default=sin(pi/2 + pi/6)
        Rotation detector threshold from classic GEM.
    oscillation_cosine_threshold : float, default=cos(pi/4)
        Oscillation detector threshold from classic GEM.
    """

    min_distance: float = 1.0e-9
    degree_divisor: float = 2.5
    degree_offset: float = 1.0
    base_desired_length: float = 20.0
    initial_temperature: float = 12.0
    minimal_temperature: float = 0.005
    gravitational_constant: float = 1.0 / 16.0
    maximal_disturbance: float = 0.0
    attraction_formula: int = 1
    rotation_sensitivity: float = 0.01
    oscillation_sensitivity: float = 0.3
    rotation_sine_threshold: float = math.sin((math.pi / 2.0) + (math.pi / 6.0))
    oscillation_cosine_threshold: float = math.cos(math.pi / 4.0)


@dataclass(frozen=True)
class GEMPrepareStateConfig:
    """Configuration for :class:`GEMPrepareState`.

    Parameters
    ----------
    sequential_node_limit : int, default=5000
        Graph-size cutoff for the exact sequential branch.
    max_rounds : int, default=30000
        OGDF-compatible cap on node updates, regardless of requested steps.
    """

    sequential_node_limit: int = 5_000
    max_rounds: int = 30_000


@dataclass(frozen=True)
class GEMBatchedConfig:
    """Configuration for GEM's vectorized large-graph fallback.

    Parameters
    ----------
    full_repulsion_limit : int, default=2000
        Largest graph size that still uses exact all-pairs repulsion.
    sampled_repulsion_neighbors : int, default=96
        Neighbor sample count for approximate repulsion above the exact limit.
    """

    full_repulsion_limit: int = 2_000
    sampled_repulsion_neighbors: int = 96


@dataclass(frozen=True)
class GEMFinalizeConfig:
    """Configuration for :class:`GEMFinalizePositions`.

    Parameters
    ----------
    default_extent : float, default=1.0
        Fallback normalization extent when preprocessing metadata is absent.
    """

    default_extent: float = 1.0


_GEM_PHYSICS_CONFIG = GEMPhysicsConfig()

_GEM_BATCHED_CACHE_READY_KEY = "gem_batched_cache_ready"
_GEM_BATCHED_DEGREE_WEIGHTS_KEY = "gem_batched_degree_weights"
_GEM_BATCHED_DESIRED_LENGTHS_KEY = "gem_batched_desired_lengths"
_GEM_BATCHED_TEMPERATURES_KEY = "gem_batched_temperatures"
_GEM_BATCHED_PREVIOUS_IMPULSE_KEY = "gem_batched_previous_impulse"
_GEM_BATCHED_SKEW_GAUGE_KEY = "gem_batched_skew_gauge"
_GEM_BATCHED_SAMPLED_DISTANCE_KEY = "gem_batched_sampled_ideal_distance"
_GEM_BATCHED_EDGE_SRC_KEY = "gem_batched_edge_src"
_GEM_BATCHED_EDGE_DST_KEY = "gem_batched_edge_dst"
_GEM_BATCHED_EDGE_WEIGHTS_KEY = "gem_batched_edge_weights"
_GEM_BATCHED_IMPULSE_KEY = "gem_batched_impulse"
_GEM_BATCHED_MOVEMENT_KEY = "gem_batched_movement"
_GEM_BATCHED_EARLY_STOP_KEY = "gem_batched_early_stop"
_GEM_BATCHED_STEP_INDEX_KEY = "gem_batched_step_index"


def _compute_degree_weights(
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    config: GEMPhysicsConfig = _GEM_PHYSICS_CONFIG,
) -> torch.Tensor:
    """Compute OGDF-style degree weights for GEM.

    Parameters
    ----------
    edge_index : torch.Tensor
        Directed edge list with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    device : torch.device
        Device for the returned tensor.
    dtype : torch.dtype, default ``torch.float32``
        Degree weight dtype.
    config : GEMPhysicsConfig, default=GEMPhysicsConfig()
        GEM constants controlling degree scaling.

    Returns
    -------
    torch.Tensor
        Degree-derived weights with shape ``[N]``.
    """
    degree_weights = torch.zeros((num_nodes,), dtype=dtype, device=device)
    if edge_index.numel() > 0:
        src = edge_index[0].to(device=device, dtype=torch.long)
        dst = edge_index[1].to(device=device, dtype=torch.long)
        ones = torch.ones_like(src, dtype=dtype)
        degree_weights.index_add_(0, src, ones)
        degree_weights.index_add_(0, dst, ones)
    return degree_weights / config.degree_divisor + config.degree_offset


def _compute_node_desired_lengths(
    problem: LayoutProblem,
    num_nodes: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    config: GEMPhysicsConfig = _GEM_PHYSICS_CONFIG,
) -> torch.Tensor:
    """Recover per-node target diagonal lengths from node-size annotations.

    Parameters
    ----------
    problem : LayoutProblem
        Layout inputs that may include node-size metadata.
    num_nodes : int
        Number of nodes.
    device : torch.device
        Device for the returned tensor.
    dtype : torch.dtype, default ``torch.float32``
        Desired-length dtype.
    config : GEMPhysicsConfig, default=GEMPhysicsConfig()
        GEM constants controlling desired-length bias.

    Returns
    -------
    torch.Tensor
        Per-node desired lengths with shape ``[N]``.
    """
    node_sizes_cpu = (
        torch.zeros((num_nodes,), dtype=torch.float32, device="cpu")
        if problem.node_sizes is None or problem.node_sizes.numel() == 0
        else problem.node_sizes.to(dtype=torch.float32, device="cpu")
    )
    if (
        node_sizes_cpu.ndim == 2
        and node_sizes_cpu.shape[0] == num_nodes
        and node_sizes_cpu.shape[1] >= 2
    ):
        node_widths = node_sizes_cpu[:, 0]
        node_heights = node_sizes_cpu[:, 1]
        node_desired_lengths = torch.sqrt(node_widths.square() + node_heights.square())
    elif node_sizes_cpu.ndim == 1 and node_sizes_cpu.shape[0] == num_nodes:
        node_desired_lengths = torch.sqrt(2.0 * node_sizes_cpu.square())
    else:
        node_desired_lengths = torch.zeros((num_nodes,), dtype=dtype, device="cpu")
    node_desired_lengths = (
        node_desired_lengths.to(device=device, dtype=dtype) + config.base_desired_length
    )
    return node_desired_lengths


def _initialize_batched_gem_cache(
    problem: LayoutProblem,
    state: SolveState,
    physics_config: GEMPhysicsConfig = _GEM_PHYSICS_CONFIG,
) -> None:
    """Create reusable batched GEM buffers in ``state.extras``.

    The decomposed batched solver mutates these buffers every step, so this
    setup step must run once before the per-iteration ``Repeat`` loop.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable problem inputs.
    state : SolveState
        Mutable solve state.
    physics_config : GEMPhysicsConfig, default=GEMPhysicsConfig()
        GEM constants reused by the batched solver.

    Returns
    -------
    None
        The buffers are stored on ``state.extras``.
    """
    if state.pos is None:
        raise ValueError("GEM batched cache initialization requires state.pos to be set.")
    if _GEM_BATCHED_CACHE_READY_KEY in state.extras:
        return

    positions = state.pos
    device = positions.device
    num_nodes = problem.num_nodes

    degree_weights = _compute_degree_weights(
        edge_index=problem.edge_index,
        num_nodes=num_nodes,
        device=device,
        dtype=torch.float32,
        config=physics_config,
    )
    node_desired_lengths = _compute_node_desired_lengths(
        problem=problem,
        num_nodes=num_nodes,
        device=device,
        dtype=torch.float32,
        config=physics_config,
    )
    temperatures = torch.full(
        (num_nodes,),
        fill_value=physics_config.initial_temperature,
        dtype=torch.float32,
        device=device,
    )
    previous_impulse = torch.zeros_like(positions)
    skew_gauge = torch.zeros((num_nodes,), dtype=torch.float32, device=device)

    extent = float(state.extras["gem_extent"])
    sampled_ideal_distance = max(
        extent / max(float(max(num_nodes, 1)) ** 0.5, 1.0),
        physics_config.min_distance,
    )

    edge_src = torch.empty((0,), dtype=torch.long, device=device)
    edge_dst = torch.empty((0,), dtype=torch.long, device=device)
    edge_weights: torch.Tensor | None = None
    if problem.edge_index.numel() > 0:
        edge_src = problem.edge_index[0].to(device=device, dtype=torch.long)
        edge_dst = problem.edge_index[1].to(device=device, dtype=torch.long)
    if problem.edge_weights is not None:
        edge_weights = problem.edge_weights.to(device=device, dtype=torch.float32)

    state.extras[_GEM_BATCHED_DEGREE_WEIGHTS_KEY] = degree_weights
    state.extras[_GEM_BATCHED_DESIRED_LENGTHS_KEY] = node_desired_lengths
    state.extras[_GEM_BATCHED_TEMPERATURES_KEY] = temperatures
    state.extras[_GEM_BATCHED_PREVIOUS_IMPULSE_KEY] = previous_impulse
    state.extras[_GEM_BATCHED_SKEW_GAUGE_KEY] = skew_gauge
    state.extras[_GEM_BATCHED_SAMPLED_DISTANCE_KEY] = sampled_ideal_distance
    state.extras[_GEM_BATCHED_EDGE_SRC_KEY] = edge_src
    state.extras[_GEM_BATCHED_EDGE_DST_KEY] = edge_dst
    state.extras[_GEM_BATCHED_EDGE_WEIGHTS_KEY] = edge_weights
    state.extras[_GEM_BATCHED_STEP_INDEX_KEY] = 0
    state.extras[_GEM_BATCHED_EARLY_STOP_KEY] = False
    state.extras[_GEM_BATCHED_CACHE_READY_KEY] = True


@register_op
@dataclass(frozen=True)
class InitializeGEMPositions(Op):
    """Seed the initial GEM coordinates exactly like classic GEM.

    The op preserves the original empty-graph and singleton shortcuts before
    falling back to the deterministic CPU Gaussian initializer used by OGDF's
    GEM implementation.
    """

    name: ClassVar[str] = "gem_initialize_positions"
    category: ClassVar[OpCategory] = OpCategory.INIT
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = ("pos", "converged")
    requires: ClassVar[Tuple[str, ...]] = ()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Seed deterministic normal positions and handle trivial graphs.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs including seed and size.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context carrying device hints.

        Returns
        -------
        SolveState
            State with ``state.pos`` populated and ``state.converged`` for
            empty or singleton graphs.
        """
        del ctx

        state.converged = False
        output_device = layout_device(problem.edge_index, problem.node_sizes)

        if problem.num_nodes == 0:
            state.pos = torch.empty((0, 2), dtype=torch.float32, device=output_device)
            state.converged = True
            return state

        if problem.num_nodes == 1:
            state.pos = torch.zeros((1, 2), dtype=torch.float32, device=output_device)
            state.converged = True
            return state

        generator = torch.Generator(device="cpu")
        generator.manual_seed(problem.seed)
        state.pos = torch.randn(
            (problem.num_nodes, 2),
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        ).to(device=output_device)
        return state


@register_op
@dataclass(frozen=True)
class GEMPrepareState(Op):
    """Prepare branch-selection and normalization metadata for GEM.

    This op records the classic GEM extent, iteration cap, output device, and
    whether the solve should take the exact sequential path or the batched
    fallback used for larger graphs.
    """

    config: GEMPrepareStateConfig = field(default_factory=GEMPrepareStateConfig)

    name: ClassVar[str] = "gem_prepare_state"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("converged", "total_steps")
    writes: ClassVar[Tuple[str, ...]] = ("extras",)
    requires: ClassVar[Tuple[str, ...]] = ()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Populate GEM metadata in ``state.extras``.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context.

        Returns
        -------
        SolveState
            State with GEM geometry metadata.
        """
        del ctx

        if state.converged:
            return state

        extent = layout_extent(problem.num_nodes, problem.node_sizes)
        capped_iters = min(int(state.total_steps), self.config.max_rounds)
        state.extras["gem_extent"] = extent
        state.extras["gem_capped_iters"] = capped_iters
        state.extras["gem_device"] = layout_device(problem.edge_index, problem.node_sizes)
        state.extras["gem_is_sequential"] = problem.num_nodes <= self.config.sequential_node_limit

        if not state.extras["gem_is_sequential"]:
            _initialize_batched_gem_cache(problem, state)
        return state


@register_op
@dataclass(frozen=True)
class GEMSequentialStep(Op):
    """Run one exact sequential Gauss-Seidel GEM sweep.

    Notes
    -----
    This is intentionally **not** split into per-iteration sub-ops because the
    next node update depends on immediately-updated positions and an in-loop
    weighted barycenter update.
    """

    config: GEMPhysicsConfig = field(default_factory=GEMPhysicsConfig)

    name: ClassVar[str] = "gem_sequential_step"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = (
        "pos",
        "local_temperatures",
        "extras",
        "converged",
    )
    writes: ClassVar[Tuple[str, ...]] = ("pos", "local_temperatures")
    requires: ClassVar[Tuple[str, ...]] = ("pos", "extras")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run the exact sequential solver path used for ``N <= 5000``.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context (unused).

        Returns
        -------
        SolveState
            Updated state containing post-solve CPU ``float32`` positions.
        """
        del ctx

        if state.converged or not state.extras.get("gem_is_sequential", False):
            return state

        num_nodes = problem.num_nodes
        capped_iters = int(state.extras.get("gem_capped_iters", 0))

        positions = state.pos.to(dtype=torch.float64, device="cpu")

        degree_weights = _compute_degree_weights(
            edge_index=problem.edge_index,
            num_nodes=num_nodes,
            device=torch.device("cpu"),
            dtype=torch.float32,
            config=self.config,
        )
        node_desired_lengths = _compute_node_desired_lengths(
            problem=problem,
            num_nodes=num_nodes,
            device=torch.device("cpu"),
            dtype=torch.float64,
            config=self.config,
        )

        adjacency = build_undirected_adjacency(
            problem.edge_index,
            num_nodes,
            edge_weights=problem.edge_weights,
        )

        local_temperatures = state.local_temperatures
        if local_temperatures is None:
            local_temperatures = torch.full(
                (num_nodes,),
                self.config.initial_temperature,
                dtype=torch.float64,
                device="cpu",
            )
        else:
            local_temperatures = local_temperatures.to(dtype=torch.float64, device="cpu")
            if int(local_temperatures.numel()) != num_nodes:
                local_temperatures = torch.full(
                    (num_nodes,),
                    self.config.initial_temperature,
                    dtype=torch.float64,
                    device="cpu",
                )
        previous_impulse = torch.zeros((num_nodes, 2), dtype=torch.float64)
        skew_gauge = torch.zeros((num_nodes,), dtype=torch.float64)
        barycenter = (positions * degree_weights.unsqueeze(1)).sum(dim=0)
        global_temperature = self.config.initial_temperature
        generator = torch.Generator(device="cpu")
        generator.manual_seed(problem.seed)
        permutation: list[int] = []

        if self.config.maximal_disturbance != 0.0:
            raise NotImplementedError("Non-zero GEM disturbance is intentionally unsupported.")

        rounds_remaining = capped_iters
        while global_temperature > self.config.minimal_temperature and rounds_remaining > 0:
            if not permutation:
                # GEM consumes one deterministic random permutation at a time and
                # keeps reusing the same mutable position tensor within the round.
                permutation = torch.randperm(num_nodes, generator=generator).tolist()
            node_index = int(permutation.pop())

            x_coord = float(positions[node_index, 0].item())
            y_coord = float(positions[node_index, 1].item())
            desired_length = float(node_desired_lengths[node_index].item())
            desired_square = desired_length * desired_length

            impulse_x = (
                float(barycenter[0].item()) / max(num_nodes, 1) - x_coord
            ) * self.config.gravitational_constant
            impulse_y = (
                float(barycenter[1].item()) / max(num_nodes, 1) - y_coord
            ) * self.config.gravitational_constant

            for other_index in range(num_nodes):
                if other_index == node_index:
                    continue
                delta_x = x_coord - float(positions[other_index, 0].item())
                delta_y = y_coord - float(positions[other_index, 1].item())
                distance = math.hypot(delta_x, delta_y)
                if distance > 0.0:
                    distance_square = distance * distance
                    impulse_x += delta_x * desired_square / distance_square
                    impulse_y += delta_y * desired_square / distance_square

            node_weight = float(degree_weights[node_index].item())
            for neighbor_index, edge_weight in adjacency[node_index]:
                delta_x = x_coord - float(positions[neighbor_index, 0].item())
                delta_y = y_coord - float(positions[neighbor_index, 1].item())
                distance = math.hypot(delta_x, delta_y)
                if self.config.attraction_formula == 1:
                    if distance > 0.0:
                        impulse_x -= (
                            edge_weight * delta_x * distance / (desired_length * node_weight)
                        )
                        impulse_y -= (
                            edge_weight * delta_y * distance / (desired_length * node_weight)
                        )
                else:
                    distance_square = distance * distance
                    if distance_square > 0.0:
                        impulse_x -= (
                            edge_weight * delta_x * distance_square / (desired_square * node_weight)
                        )
                        impulse_y -= (
                            edge_weight * delta_y * distance_square / (desired_square * node_weight)
                        )

            raw_x = impulse_x
            raw_y = impulse_y
            impulse_length = math.hypot(raw_x, raw_y)

            if impulse_length > 0.0:
                local_temperature = float(local_temperatures[node_index].item())
                move_x = raw_x * local_temperature / impulse_length
                move_y = raw_y * local_temperature / impulse_length

                positions[node_index, 0] += move_x
                positions[node_index, 1] += move_y

                barycenter[0] += node_weight * move_x
                barycenter[1] += node_weight * move_y

                old_x = float(previous_impulse[node_index, 0].item())
                old_y = float(previous_impulse[node_index, 1].item())
                product = math.hypot(move_x, move_y) * math.hypot(old_x, old_y)
                if product > 0.0:
                    global_temperature -= local_temperature / max(num_nodes, 1)

                    sin_beta = (move_x * old_x - move_y * old_y) / product
                    cos_beta = (move_x * old_x + move_y * old_y) / product

                    skew_value = float(skew_gauge[node_index].item())
                    if sin_beta > self.config.rotation_sine_threshold:
                        skew_value += self.config.rotation_sensitivity

                    if abs(cos_beta) > self.config.oscillation_cosine_threshold:
                        local_temperature *= 1.0 + (cos_beta * self.config.oscillation_sensitivity)

                    # The skew gauge damps nodes that keep turning sharply so the
                    # exact sequential path cools in the same order as classic GEM.
                    local_temperature *= 1.0 - abs(skew_value)
                    if local_temperature >= self.config.initial_temperature:
                        local_temperature = self.config.initial_temperature

                    skew_gauge[node_index] = skew_value
                    local_temperatures[node_index] = local_temperature
                    global_temperature += local_temperature / max(num_nodes, 1)

                previous_impulse[node_index, 0] = move_x
                previous_impulse[node_index, 1] = move_y

            rounds_remaining -= 1

        state.pos = positions.to(dtype=torch.float32)
        state.local_temperatures = local_temperatures
        return state


class GEMSequentialSolve(GEMSequentialStep):
    """Backward-compatible wrapper for the sequential GEM monolithic step."""

    name: ClassVar[str] = "gem_sequential_solve"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = (
        "pos",
        "extras",
    )
    writes: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run the exact sequential path while preserving legacy op naming.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context.

        Returns
        -------
        SolveState
            Updated state containing the sequential solution.
        """
        return super().apply(problem, state, ctx)


@register_op
@dataclass(frozen=True)
class GEMComputeImpulse(Op):
    """Assemble one batched GEM impulse field.

    The op mirrors the sequential solver's gravity, repulsion, and attraction
    terms, but evaluates them in vectorized form so larger graphs can use the
    approximate batched fallback.
    """

    physics_config: GEMPhysicsConfig = field(default_factory=GEMPhysicsConfig)
    batched_config: GEMBatchedConfig = field(default_factory=GEMBatchedConfig)

    name: ClassVar[str] = "gem_compute_impulse"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("pos", "extras", "converged")
    writes: ClassVar[Tuple[str, ...]] = ("extras",)
    requires: ClassVar[Tuple[str, ...]] = ("pos", "extras")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute batched repulsion + attraction + gravity for one iteration.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state holding cached batched buffers.
        ctx : RuntimeContext
            Runtime context (unused).

        Returns
        -------
        SolveState
            State with ``state.extras["gem_batched_impulse"]`` populated.
        """
        del ctx

        if state.pos is None or state.converged:
            return state
        if not state.extras.get(_GEM_BATCHED_CACHE_READY_KEY, False):
            _initialize_batched_gem_cache(
                problem,
                state,
                physics_config=self.physics_config,
            )

        if state.extras.get(_GEM_BATCHED_EARLY_STOP_KEY, False):
            state.extras[_GEM_BATCHED_IMPULSE_KEY] = torch.zeros_like(state.pos)
            return state

        positions = state.pos
        num_nodes = problem.num_nodes
        degree_weights = state.extras[_GEM_BATCHED_DEGREE_WEIGHTS_KEY]
        node_desired_lengths = state.extras[_GEM_BATCHED_DESIRED_LENGTHS_KEY]

        if num_nodes > self.batched_config.full_repulsion_limit:
            sample_size = min(num_nodes, self.batched_config.sampled_repulsion_neighbors)
            step_index = int(state.extras.get(_GEM_BATCHED_STEP_INDEX_KEY, 0))
            generator = torch.Generator(device="cpu")
            generator.manual_seed(problem.seed + step_index + 1)
            sampled = torch.randint(
                0,
                num_nodes,
                (num_nodes, sample_size),
                generator=generator,
                dtype=torch.long,
            ).to(positions.device)
            neighbors = positions[sampled]
            delta = positions.unsqueeze(1) - neighbors
            # Sampled repulsion intentionally follows the existing seeded draw
            # schedule so the batched fallback remains reproducible.
            distances = torch.linalg.norm(delta, dim=2).clamp(min=self.physics_config.min_distance)
            ideal_distance = state.extras[_GEM_BATCHED_SAMPLED_DISTANCE_KEY]
            force = (ideal_distance * ideal_distance) / distances
            repulsive = (delta / distances.unsqueeze(2) * force.unsqueeze(2)).sum(dim=1)
            state.extras[_GEM_BATCHED_STEP_INDEX_KEY] = step_index + 1
        else:
            delta = positions.unsqueeze(1) - positions.unsqueeze(0)
            distance_square = delta.square().sum(dim=2)
            mask = distance_square > self.physics_config.min_distance
            safe_distance_square = torch.where(
                mask,
                distance_square,
                torch.ones_like(distance_square),
            )
            desired_square = node_desired_lengths.square().unsqueeze(1).unsqueeze(2)
            repulsive = delta * desired_square / safe_distance_square.unsqueeze(2)
            repulsive = (repulsive * mask.unsqueeze(2)).sum(dim=1)

        edge_src = state.extras[_GEM_BATCHED_EDGE_SRC_KEY]
        if edge_src.numel() == 0:
            attractive = torch.zeros_like(positions)
        else:
            attractive = torch.zeros_like(positions)
            edge_dst = state.extras[_GEM_BATCHED_EDGE_DST_KEY]
            edge_weights = state.extras.get(_GEM_BATCHED_EDGE_WEIGHTS_KEY)
            source_force = torch.zeros_like(positions)
            source_delta = positions[edge_src] - positions[edge_dst]
            source_distance = torch.linalg.norm(source_delta, dim=1)
            source_weights = degree_weights[edge_src].clamp(min=1.0)
            target_weights = degree_weights[edge_dst].clamp(min=1.0)
            source_desired = node_desired_lengths[edge_src].clamp(
                min=self.physics_config.min_distance
            )
            target_desired = node_desired_lengths[edge_dst].clamp(
                min=self.physics_config.min_distance
            )

            source_force = -source_delta * (
                source_distance / (source_desired * source_weights)
            ).unsqueeze(1)
            target_force = source_delta * (
                source_distance / (target_desired * target_weights)
            ).unsqueeze(1)

            if edge_weights is not None:
                weighted = edge_weights.unsqueeze(1)
                source_force = source_force * weighted
                target_force = target_force * weighted

            attractive.index_add_(0, edge_src, source_force)
            attractive.index_add_(0, edge_dst, target_force)

        barycenter = (positions * degree_weights.unsqueeze(1)).sum(dim=0)
        gravity = (
            barycenter / max(num_nodes, 1) - positions
        ) * self.physics_config.gravitational_constant

        impulse = repulsive + attractive + gravity
        state.extras[_GEM_BATCHED_IMPULSE_KEY] = impulse
        return state


@register_op
@dataclass(frozen=True)
class GEMUpdateTemperatures(Op):
    """Adapt batched node temperatures and derive a movement field.

    The temperature update intentionally mirrors the sequential path's rotation
    and oscillation heuristics so the large-graph fallback cools with the same
    qualitative behavior.
    """

    config: GEMPhysicsConfig = field(default_factory=GEMPhysicsConfig)

    name: ClassVar[str] = "gem_update_temperatures"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("extras", "converged")
    writes: ClassVar[Tuple[str, ...]] = ("extras",)
    requires: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute scaled movement and adapt per-node temperatures.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state containing cached GEM batched buffers.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with updated temperatures and buffered node movement.
        """
        del problem, ctx

        if state.converged:
            return state

        impulse = state.extras[_GEM_BATCHED_IMPULSE_KEY]
        temperatures = state.extras[_GEM_BATCHED_TEMPERATURES_KEY]
        previous_impulse = state.extras[_GEM_BATCHED_PREVIOUS_IMPULSE_KEY]
        skew_gauge = state.extras[_GEM_BATCHED_SKEW_GAUGE_KEY]

        norm = torch.linalg.norm(impulse, dim=1, keepdim=True)
        safe_norm = norm.clamp(min=self.config.min_distance)
        movement = torch.where(
            norm > 0,
            impulse * (temperatures.unsqueeze(1) / safe_norm),
            torch.zeros_like(impulse),
        )

        current_norm = torch.linalg.norm(movement, dim=1)
        previous_norm = torch.linalg.norm(previous_impulse, dim=1)
        product = current_norm * previous_norm
        valid = product > self.config.min_distance
        if bool(valid.any()):
            safe_product = product.clamp(min=self.config.min_distance)
            sin_beta = (
                movement[:, 0] * previous_impulse[:, 0] - movement[:, 1] * previous_impulse[:, 1]
            ) / safe_product
            cos_beta = (
                movement[:, 0] * previous_impulse[:, 0] + movement[:, 1] * previous_impulse[:, 1]
            ) / safe_product

            rotation_mask = valid & (sin_beta > self.config.rotation_sine_threshold)
            skew_gauge = torch.where(
                rotation_mask,
                skew_gauge + self.config.rotation_sensitivity,
                skew_gauge,
            )
            oscillation_mask = valid & (cos_beta.abs() > self.config.oscillation_cosine_threshold)
            oscillation_scale = 1.0 + cos_beta * self.config.oscillation_sensitivity
            temperatures = torch.where(
                oscillation_mask,
                temperatures * oscillation_scale,
                temperatures,
            )
            temperatures = temperatures * (1.0 - skew_gauge.abs())
            temperatures = torch.minimum(
                temperatures,
                torch.full_like(temperatures, self.config.initial_temperature),
            )

        if float(temperatures.mean().item()) < self.config.minimal_temperature:
            state.extras[_GEM_BATCHED_EARLY_STOP_KEY] = True

        state.extras[_GEM_BATCHED_TEMPERATURES_KEY] = temperatures
        state.extras[_GEM_BATCHED_SKEW_GAUGE_KEY] = skew_gauge
        state.extras[_GEM_BATCHED_MOVEMENT_KEY] = movement
        state.extras[_GEM_BATCHED_PREVIOUS_IMPULSE_KEY] = movement
        return state


@register_op
@dataclass(frozen=True)
class GEMApplyDisplacement(Op):
    """Apply the buffered batched GEM movement field to positions."""

    name: ClassVar[str] = "gem_apply_displacement"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("pos", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("pos", "extras")
    requires: ClassVar[Tuple[str, ...]] = ("pos", "extras")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Move nodes by the buffered batched movement vector.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state containing buffered GEM movement.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with positions displaced by the current batched update.
        """
        del problem, ctx

        if state.pos is None:
            return state

        movement = state.extras.get(_GEM_BATCHED_MOVEMENT_KEY)
        if movement is None:
            return state

        state.pos = state.pos + movement
        state.extras[_GEM_BATCHED_PREVIOUS_IMPULSE_KEY] = movement
        return state


@register_op
@dataclass(frozen=True)
class GEMConvergenceCheck(Op):
    """Optional batched convergence marker based on mean temperature."""

    config: GEMPhysicsConfig = field(default_factory=GEMPhysicsConfig)

    name: ClassVar[str] = "gem_convergence_check"
    category: ClassVar[OpCategory] = OpCategory.CONVERGE
    reads: ClassVar[Tuple[str, ...]] = ("extras", "converged")
    writes: ClassVar[Tuple[str, ...]] = ("converged",)
    requires: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Raise ``state.converged`` when batched temperatures fall below threshold.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state containing cached temperatures.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with an updated ``converged`` flag.
        """
        del problem, ctx

        if state.converged:
            return state

        temperatures = state.extras.get(_GEM_BATCHED_TEMPERATURES_KEY)
        if temperatures is not None:
            state.converged = bool(
                float(temperatures.mean().item()) < self.config.minimal_temperature
            )
        return state


@register_op
@dataclass(frozen=True)
class GEMBatchedSolve(Op):
    """Run the vectorized GEM fallback for graphs above the sequential cutoff."""

    physics_config: GEMPhysicsConfig = field(default_factory=GEMPhysicsConfig)
    batched_config: GEMBatchedConfig = field(default_factory=GEMBatchedConfig)

    name: ClassVar[str] = "gem_batched_solve"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = (
        "pos",
        "extras",
        "converged",
        "step",
    )
    writes: ClassVar[Tuple[str, ...]] = ("pos", "extras")
    requires: ClassVar[Tuple[str, ...]] = ("pos", "extras")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run the batched GEM fallback used when ``N > 5000``.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context (unused).

        Returns
        -------
        SolveState
            Updated state with batched GEM positions.
        """
        if state.converged or state.extras.get("gem_is_sequential", True):
            return state

        capped_iters = int(state.extras.get("gem_capped_iters", 0))
        if capped_iters <= 0:
            return state

        _initialize_batched_gem_cache(
            problem,
            state,
            physics_config=self.physics_config,
        )
        state.extras[_GEM_BATCHED_EARLY_STOP_KEY] = False
        state.extras[_GEM_BATCHED_STEP_INDEX_KEY] = 0

        saved_step = state.step
        state = Repeat(
            n=capped_iters,
            ops=[
                GEMComputeImpulse(
                    physics_config=self.physics_config,
                    batched_config=self.batched_config,
                ),
                GEMUpdateTemperatures(config=self.physics_config),
                GEMApplyDisplacement(),
            ],
        ).apply(problem, state, ctx)
        state.step = saved_step
        return state


@register_op
@dataclass(frozen=True)
class GEMFinalizePositions(Op):
    """Normalize GEM coordinates and move them to the resolved output device."""

    config: GEMFinalizeConfig = field(default_factory=GEMFinalizeConfig)

    name: ClassVar[str] = "gem_finalize_positions"
    category: ClassVar[OpCategory] = OpCategory.POSTPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("pos", "extras", "converged")
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Apply final normalization and cast to final dtype/device.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs (used only for device fallback).
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context (unused).

        Returns
        -------
        SolveState
            State with final float32 coordinates.

        Raises
        ------
        ValueError
            If ``state.pos`` is missing.
        """
        del problem, ctx

        if state.pos is None:
            raise ValueError("GEMFinalizePositions requires state.pos to be set.")

        if state.converged:
            return state

        device = state.extras.get("gem_device")
        if not isinstance(device, torch.device):
            device = layout_device(torch.empty((2, 0), dtype=torch.long), None)

        extent = float(state.extras.get("gem_extent", self.config.default_extent))
        state.pos = normalize_positions(state.pos, extent).to(dtype=torch.float32, device=device)
        return state
