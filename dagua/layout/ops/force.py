"""Primitive force-directed layout operations.

These ops implement the non-differentiable force accumulation pattern used by
classic force-directed algorithms:

``ZeroForces -> [force ops accumulate into state.forces] -> ApplyDisplacement``

Some algorithms use specialized apply/update steps instead of the generic
displacement clamp. Those are represented here as dedicated ops such as
``AdaptiveSpeedApply`` and ``GEMNodeTick``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional, Tuple, cast

import numpy as np
import torch

from dagua.layout.classic.fa2 import (
    _adjust_speed_and_apply_forces as _fa2_adjust_speed_and_apply_forces,
)
from dagua.layout.classic.fa2 import (
    _attraction_force as _fa2_attraction_force,
)
from dagua.layout.classic.fa2 import (
    _barnes_hut_force_for_node as _fa2_barnes_hut_force_for_node,
)
from dagua.layout.classic.fa2 import (
    _BarnesHutNode as _FA2BarnesHutNode,
)
from dagua.layout.classic.fa2 import (
    _compute_degree as _fa2_compute_degree,
)
from dagua.layout.classic.fa2 import (
    _gravity_force as _fa2_gravity_force,
)
from dagua.layout.classic.fa2 import (
    _unique_undirected_edges_with_weights as _fa2_unique_undirected_edges_with_weights,
)
from dagua.layout.classic.gem import (
    _INITIAL_TEMPERATURE as _GEM_INITIAL_TEMPERATURE,
)
from dagua.layout.classic.gem import (
    _MIN_DISTANCE as _GEM_MIN_DISTANCE,
)
from dagua.layout.classic.gem import (
    _OSCILLATION_COSINE_THRESHOLD as _GEM_OSCILLATION_COSINE_THRESHOLD,
)
from dagua.layout.classic.gem import (
    _OSCILLATION_SENSITIVITY as _GEM_OSCILLATION_SENSITIVITY,
)
from dagua.layout.classic.gem import (
    _ROTATION_SENSITIVITY as _GEM_ROTATION_SENSITIVITY,
)
from dagua.layout.classic.gem import (
    _ROTATION_SINE_THRESHOLD as _GEM_ROTATION_SINE_THRESHOLD,
)
from dagua.layout.classic.sfdp import (
    _barnes_hut_force_for_index as _sfdp_barnes_hut_force_for_index,
)
from dagua.layout.classic.sfdp import (
    _QuadTreeNode as _SFDPQuadTreeNode,
)
from dagua.layout.classic.stress_majorization import _smacof_update as _stress_maj_smacof_update
from dagua.layout.ops.base import Op
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_FR_MIN_DISTANCE = 0.01
_GRAPHOPT_COULOMBS_CONSTANT = 8_987_500_000.0
_GRAPHOPT_MIN_DISTANCE = 1.0e-12
_GRAPHOPT_MAX_REPULSION_DISTANCE = 500.0
_LGL_MIN_DISTANCE = 1.0e-12
_DENSITY_EPSILON_SCALE = 1.0


def _require_positions(state: SolveState) -> torch.Tensor:
    """Return the current position tensor or raise a descriptive error.

    Parameters
    ----------
    state : SolveState
        Mutable solve state.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If ``state.pos`` is missing.
    """
    if state.pos is None:
        raise ValueError("Force ops require state.pos to be populated.")
    return state.pos


def _require_forces(state: SolveState) -> torch.Tensor:
    """Return the force accumulation buffer or raise a descriptive error.

    Parameters
    ----------
    state : SolveState
        Mutable solve state.

    Returns
    -------
    torch.Tensor
        Force buffer with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If ``state.forces`` is missing.
    """
    if state.forces is None:
        raise ValueError("Force accumulation requires state.forces. Run ZeroForces first.")
    return state.forces


def _resolve_force_area(problem: LayoutProblem, state: SolveState) -> float:
    """Resolve the drawing area used by FR-style spacing constants.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs.
    state : SolveState
        Mutable solve state.

    Returns
    -------
    float
        Positive drawing area.

    Notes
    -----
    The classic FR implementation in this repository uses a unit square.
    Callers can override that by placing ``force_area`` in ``state.extras``.
    """
    area = state.extras.get("force_area", 1.0)
    if not isinstance(area, (int, float)):
        raise ValueError("state.extras['force_area'] must be a real number.")
    return max(float(area), 1.0e-12)


def _resolve_area_k(problem: LayoutProblem, state: SolveState) -> float:
    """Resolve the FR-style spacing constant ``k = sqrt(area / N)``.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs.
    state : SolveState
        Mutable solve state.

    Returns
    -------
    float
        Positive spacing constant.
    """
    area = _resolve_force_area(problem=problem, state=state)
    return math.sqrt(area / float(max(problem.num_nodes, 1)))


def _edge_weights(problem: LayoutProblem, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return per-edge weights aligned with ``problem.edge_index``.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs.
    device : torch.device
        Target device for the result.
    dtype : torch.dtype
        Target dtype for the result.

    Returns
    -------
    torch.Tensor
        Weight vector with shape ``[E]``.
    """
    edge_count = int(problem.edge_index.shape[1])
    if problem.edge_weights is None:
        return torch.ones((edge_count,), device=device, dtype=dtype)
    return problem.edge_weights.to(device=device, dtype=dtype)


def _undirected_degree(
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute unique-undirected degree counts.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    device : torch.device
        Target device for the result.

    Returns
    -------
    torch.Tensor
        Degree tensor with shape ``[N]``.
    """
    degree = _fa2_compute_degree(edge_index=edge_index, num_nodes=num_nodes)
    return degree.to(device=device, dtype=torch.float32)


def _fa2_mass(problem: LayoutProblem, state: SolveState, device: torch.device) -> torch.Tensor:
    """Resolve the ForceAtlas2 mass vector.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs.
    state : SolveState
        Mutable solve state.
    device : torch.device
        Target device for the result.

    Returns
    -------
    torch.Tensor
        Mass tensor with shape ``[N]``.
    """
    if "fa2_mass" in state.extras:
        return cast(torch.Tensor, state.extras["fa2_mass"]).to(device=device, dtype=torch.float32)
    if state.degree is not None:
        return state.degree.to(device=device, dtype=torch.float32) + 1.0
    return _undirected_degree(problem.edge_index, problem.num_nodes, device=device) + 1.0


def _gem_degree_weights(
    problem: LayoutProblem,
    state: SolveState,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Resolve OGDF GEM's degree-derived node weights.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs.
    state : SolveState
        Mutable solve state.
    device : torch.device
        Target device for the result.
    dtype : torch.dtype
        Target dtype for the result.

    Returns
    -------
    torch.Tensor
        Degree-based weights with shape ``[N]``.
    """
    if "gem_degree_weights" in state.extras:
        return cast(torch.Tensor, state.extras["gem_degree_weights"]).to(device=device, dtype=dtype)
    if state.degree is not None:
        return state.degree.to(device=device, dtype=dtype) / 2.5 + 1.0

    degrees = torch.zeros((problem.num_nodes,), dtype=dtype, device=device)
    if problem.edge_index.numel() == 0:
        return degrees + 1.0

    src = problem.edge_index[0].to(device=device, dtype=torch.long)
    dst = problem.edge_index[1].to(device=device, dtype=torch.long)
    ones = torch.ones_like(src, dtype=dtype)
    degrees.index_add_(0, src, ones)
    degrees.index_add_(0, dst, ones)
    return degrees / 2.5 + 1.0


def _spring_lengths_by_node(
    problem: LayoutProblem,
    state: SolveState,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Resolve per-node desired spring lengths for GEM-style attraction.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs.
    state : SolveState
        Mutable solve state.
    device : torch.device
        Target device for the result.
    dtype : torch.dtype
        Target dtype for the result.

    Returns
    -------
    torch.Tensor
        Desired lengths with shape ``[N]``.
    """
    if state.spring_lengths is None:
        return torch.full((problem.num_nodes,), 20.0, device=device, dtype=dtype)

    spring_lengths = state.spring_lengths.to(device=device, dtype=dtype)
    if spring_lengths.ndim != 1:
        raise ValueError("state.spring_lengths must be one-dimensional.")
    if spring_lengths.shape[0] != problem.num_nodes:
        raise ValueError("GEM-style desired lengths require state.spring_lengths with shape [N].")
    return spring_lengths


def _pair_force_delta(pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build pairwise displacement vectors and distances.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Pairwise deltas with shape ``[N, N, 2]`` and distances with shape
        ``[N, N]``.
    """
    delta = pos.unsqueeze(1) - pos.unsqueeze(0)
    distance = torch.linalg.vector_norm(delta, dim=2)
    return delta, distance


def _fa2_barnes_hut_force(
    quadtree: _FA2BarnesHutNode,
    pos: torch.Tensor,
    mass: torch.Tensor,
    theta: float,
) -> torch.Tensor:
    """Evaluate FA2 Barnes-Hut repulsion using a prebuilt quadtree.

    Parameters
    ----------
    quadtree : _BarnesHutNode
        Root Barnes-Hut region.
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    mass : torch.Tensor
        Node mass tensor with shape ``[N]``.
    theta : float
        Barnes-Hut opening threshold.

    Returns
    -------
    torch.Tensor
        Force tensor with shape ``[N, 2]``.
    """
    pos_np = pos.detach().cpu().numpy()
    mass_np = mass.detach().cpu().numpy()
    force_np = np.zeros((pos.shape[0], 2), dtype=np.float64)

    for node_index in range(pos.shape[0]):
        fx, fy = _fa2_barnes_hut_force_for_node(
            node=quadtree,
            pos_np=pos_np,
            mass_np=mass_np,
            index=node_index,
            scaling_ratio=1.0,
            theta=theta,
        )
        force_np[node_index, 0] = fx
        force_np[node_index, 1] = fy

    return torch.from_numpy(force_np).to(device=pos.device, dtype=pos.dtype)


def _sfdp_barnes_hut_force(
    quadtree: _SFDPQuadTreeNode,
    pos: torch.Tensor,
    theta: float,
) -> torch.Tensor:
    """Evaluate SFDP Barnes-Hut repulsion using a prebuilt quadtree.

    Parameters
    ----------
    quadtree : _QuadTreeNode
        Root quadtree node.
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    theta : float
        Barnes-Hut opening threshold.

    Returns
    -------
    torch.Tensor
        Force tensor with shape ``[N, 2]``.
    """
    force = torch.zeros_like(pos)
    pos_cpu = pos.to(dtype=torch.float32, device="cpu")
    for node_index in range(pos.shape[0]):
        node_force = _sfdp_barnes_hut_force_for_index(
            node=quadtree,
            positions=pos_cpu,
            index=node_index,
            theta=theta,
            repulsive_scale=1.0,
            repulsive_exponent=-1.0,
        )
        force[node_index] = node_force.to(device=pos.device, dtype=pos.dtype)
    return force


def _density_grid_gradient(
    density_grid: Any,
    node_index: int,
    pos: torch.Tensor,
) -> torch.Tensor:
    """Estimate a density-grid force via central differences.

    Parameters
    ----------
    density_grid : Any
        Grid object exposing ``coarse_density(position)`` and optionally
        ``fine_density(node, position, positions)``.
    node_index : int
        Node index being evaluated.
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Negative gradient estimate with shape ``[2]``.
    """
    step_size = float(getattr(density_grid, "cell_width", 1.0)) * _DENSITY_EPSILON_SCALE
    basis = torch.eye(2, dtype=pos.dtype, device=pos.device) * step_size
    center = pos[node_index]

    def _energy(candidate: torch.Tensor) -> float:
        """Evaluate the density proxy energy at one candidate position.

        Parameters
        ----------
        candidate : torch.Tensor
            Candidate coordinate with shape ``[2]``.

        Returns
        -------
        float
            Scalar energy.
        """
        if hasattr(density_grid, "fine_density"):
            return float(
                density_grid.fine_density(
                    node=node_index,
                    position=candidate,
                    positions=pos,
                )
            )
        return float(density_grid.coarse_density(position=candidate))

    grad_x = (_energy(center + basis[0]) - _energy(center - basis[0])) / (2.0 * step_size)
    grad_y = (_energy(center + basis[1]) - _energy(center - basis[1])) / (2.0 * step_size)
    return torch.tensor([-grad_x, -grad_y], device=pos.device, dtype=pos.dtype)


def _resolve_gem_node_index(
    problem: LayoutProblem,
    state: SolveState,
    ctx: RuntimeContext,
) -> int:
    """Choose the next node for a sequential GEM tick.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs.
    state : SolveState
        Mutable solve state.
    ctx : RuntimeContext
        Execution infrastructure.

    Returns
    -------
    int
        Node index selected for this tick.
    """
    if "gem_node_index" in state.extras:
        return int(state.extras["gem_node_index"])

    permutation = cast(list[int], state.extras.get("gem_permutation", []))
    if not permutation:
        generator = ctx.generator
        if generator is None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(problem.seed + int(state.step))
        permutation = torch.randperm(problem.num_nodes, generator=generator).tolist()
        state.extras["gem_permutation"] = permutation
    return cast(list[int], state.extras["gem_permutation"]).pop()


def _gem_apply_node_update(
    node_index: int,
    pos: torch.Tensor,
    impulse: torch.Tensor,
    previous_impulse: torch.Tensor,
    local_temperatures: torch.Tensor,
    skew_gauge: torch.Tensor,
    degree_weights: torch.Tensor,
    barycenter: torch.Tensor,
    global_temperature: float,
) -> float:
    """Apply OGDF GEM's sequential node update rule.

    Parameters
    ----------
    node_index : int
        Node to update.
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    impulse : torch.Tensor
        Current node force vector with shape ``[2]``.
    previous_impulse : torch.Tensor
        Previous saved impulses with shape ``[N, 2]``.
    local_temperatures : torch.Tensor
        Per-node temperatures with shape ``[N]``.
    skew_gauge : torch.Tensor
        Per-node skew gauges with shape ``[N]``.
    degree_weights : torch.Tensor
        Degree-based GEM weights with shape ``[N]``.
    barycenter : torch.Tensor
        Weighted barycenter sum with shape ``[2]``.
    global_temperature : float
        Current global GEM temperature.

    Returns
    -------
    float
        Updated global temperature.
    """
    num_nodes = int(pos.shape[0])
    raw_x = float(impulse[0].item())
    raw_y = float(impulse[1].item())
    impulse_length = math.hypot(raw_x, raw_y)
    if impulse_length <= 0.0:
        return global_temperature

    local_temperature = float(local_temperatures[node_index].item())
    move_x = raw_x * local_temperature / impulse_length
    move_y = raw_y * local_temperature / impulse_length

    pos[node_index, 0] += move_x
    pos[node_index, 1] += move_y

    node_weight = float(degree_weights[node_index].item())
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
        if sin_beta > _GEM_ROTATION_SINE_THRESHOLD:
            skew_value += _GEM_ROTATION_SENSITIVITY

        if abs(cos_beta) > _GEM_OSCILLATION_COSINE_THRESHOLD:
            local_temperature *= 1.0 + (cos_beta * _GEM_OSCILLATION_SENSITIVITY)

        local_temperature *= 1.0 - abs(skew_value)
        if local_temperature >= _GEM_INITIAL_TEMPERATURE:
            local_temperature = _GEM_INITIAL_TEMPERATURE

        skew_gauge[node_index] = skew_value
        local_temperatures[node_index] = local_temperature
        global_temperature += local_temperature / max(num_nodes, 1)

    previous_impulse[node_index, 0] = move_x
    previous_impulse[node_index, 1] = move_y
    return global_temperature


@register_op
class ZeroForces(Op):
    """Allocate or reset the force accumulation buffer."""

    name: ClassVar[str] = "zero_forces"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    writes: ClassVar[Tuple[str, ...]] = ("forces",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Allocate or zero the ``state.forces`` buffer.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with a zeroed force buffer.
        """
        del ctx

        if state.pos is not None:
            if state.forces is None or state.forces.shape != state.pos.shape:
                state.forces = torch.zeros_like(state.pos)
            else:
                state.forces.zero_()
            return state

        if state.forces is not None:
            state.forces.zero_()
            return state

        state.forces = torch.zeros((problem.num_nodes, 2), dtype=torch.float32)
        return state


@dataclass(frozen=True)
class InverseDistanceRepulsionConfig:
    """Configuration for ``InverseDistanceRepulsion``.

    Parameters
    ----------
    k_formula : str, default="area"
        Spacing rule. ``"area"`` uses the FR constant ``sqrt(area / N)`` with
        a unit-square default area and an optional ``state.extras['force_area']``
        override.
    """

    k_formula: str = "area"


@register_op
@dataclass(frozen=True)
class InverseDistanceRepulsion(Op):
    """Accumulate FR-style inverse-distance repulsion into ``state.forces``."""

    config: InverseDistanceRepulsionConfig = field(default_factory=InverseDistanceRepulsionConfig)

    name: ClassVar[str] = "inverse_distance_repulsion"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ("forces",)
    requires: ClassVar[Tuple[str, ...]] = ("pos", "forces")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Accumulate the exact FR/GEM-style ``k^2 / d`` repulsion.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with repulsive forces added to ``state.forces``.
        """
        del ctx

        pos = _require_positions(state)
        forces = _require_forces(state)
        if pos.shape[0] <= 1:
            return state
        if self.config.k_formula != "area":
            raise ValueError("InverseDistanceRepulsion only supports k_formula='area'.")

        optimal_distance = _resolve_area_k(problem=problem, state=state)
        delta, distance = _pair_force_delta(pos)
        distance = distance.clamp(min=_FR_MIN_DISTANCE)
        contribution = delta * (
            (optimal_distance * optimal_distance) / distance.square()
        ).unsqueeze(2)
        diagonal = torch.eye(pos.shape[0], dtype=torch.bool, device=pos.device)
        contribution = contribution.masked_fill(diagonal.unsqueeze(2), 0.0)
        state.forces = forces + contribution.sum(dim=1)
        return state


@register_op
class FRCombinedForce(Op):
    """Compute the exact dense FR force update in one einsum."""

    name: ClassVar[str] = "fr_combined_force"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("pos", "extras.fr_adjacency")
    writes: ClassVar[Tuple[str, ...]] = ("forces",)
    requires: ClassVar[Tuple[str, ...]] = ("pos", "extras.fr_adjacency")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Store the classic FR dense displacement in ``state.forces``.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state containing positions and FR adjacency.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with the combined FR displacement stored in ``state.forces``.

        Raises
        ------
        ValueError
            If ``state.extras['fr_adjacency']`` is missing or malformed.
        """
        del ctx

        pos = _require_positions(state)
        adjacency = state.extras.get("fr_adjacency")
        if not isinstance(adjacency, torch.Tensor):
            raise ValueError("FRCombinedForce requires state.extras['fr_adjacency'].")
        if tuple(adjacency.shape) != (problem.num_nodes, problem.num_nodes):
            raise ValueError(
                "state.extras['fr_adjacency'] must have shape "
                f"({problem.num_nodes}, {problem.num_nodes})."
            )

        optimal_distance = _resolve_area_k(problem=problem, state=state)
        delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        distance = torch.linalg.norm(delta, dim=-1)
        distance = torch.clamp(distance, min=_FR_MIN_DISTANCE)
        adjacency = adjacency.to(device=pos.device, dtype=pos.dtype)
        displacement = torch.einsum(
            "ijk,ij->ik",
            delta,
            (optimal_distance * optimal_distance / distance.square())
            - (adjacency * distance / optimal_distance),
        )
        state.forces = displacement
        return state


@dataclass(frozen=True)
class InverseSquareRepulsionConfig:
    """Configuration for ``InverseSquareRepulsion``.

    Parameters
    ----------
    charge : float, default=0.001
        Shared node charge.
    cutoff : float, default=500.0
        Maximum interaction distance.
    """

    charge: float = 0.001
    cutoff: float = 500.0


@register_op
@dataclass(frozen=True)
class InverseSquareRepulsion(Op):
    """Accumulate GraphOpt's inverse-square Coulomb repulsion."""

    config: InverseSquareRepulsionConfig = field(default_factory=InverseSquareRepulsionConfig)

    name: ClassVar[str] = "inverse_square_repulsion"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ("forces",)
    requires: ClassVar[Tuple[str, ...]] = ("pos", "forces")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Accumulate GraphOpt's exact inverse-square repulsive force.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with repulsive forces added to ``state.forces``.
        """
        del problem, ctx

        pos = _require_positions(state)
        forces = _require_forces(state)
        if pos.shape[0] <= 1 or self.config.charge == 0.0:
            return state

        pair_source, pair_target = torch.triu_indices(pos.shape[0], pos.shape[0], offset=1)
        delta = pos[pair_source] - pos[pair_target]
        distance_sq = delta.square().sum(dim=1)
        max_repulsion_distance_sq = self.config.cutoff * self.config.cutoff
        mask = (distance_sq > _GRAPHOPT_MIN_DISTANCE) & (distance_sq < max_repulsion_distance_sq)
        if not bool(mask.any().item()):
            return state

        pair_delta = delta[mask]
        pair_distance_sq = distance_sq[mask]
        pair_distance = torch.sqrt(pair_distance_sq)
        direction = pair_delta / pair_distance.unsqueeze(1)
        magnitude = (
            _GRAPHOPT_COULOMBS_CONSTANT
            * (self.config.charge * self.config.charge)
            / pair_distance_sq
        )
        contribution = direction * magnitude.unsqueeze(1)
        updated = forces.clone()
        updated.index_add_(0, pair_source[mask], contribution)
        updated.index_add_(0, pair_target[mask], -contribution)
        state.forces = updated
        return state


@dataclass(frozen=True)
class InversePowerRepulsionConfig:
    """Configuration for ``InversePowerRepulsion``.

    Parameters
    ----------
    exponent : float, default=-1.0
        Power-law exponent ``p`` in the SFDP denominator ``d^(2 - p)``.
    """

    exponent: float = -1.0


@register_op
@dataclass(frozen=True)
class InversePowerRepulsion(Op):
    """Accumulate SFDP's inverse-power repulsion."""

    config: InversePowerRepulsionConfig = field(default_factory=InversePowerRepulsionConfig)

    name: ClassVar[str] = "inverse_power_repulsion"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ("forces",)
    requires: ClassVar[Tuple[str, ...]] = ("pos", "forces")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Accumulate exact SFDP inverse-power repulsion.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with repulsive forces added to ``state.forces``.
        """
        del ctx

        pos = _require_positions(state)
        forces = _require_forces(state)
        if pos.shape[0] <= 1:
            return state

        ideal_length = float(state.extras.get("ideal_length", _resolve_area_k(problem, state)))
        repulsive_scale = max(ideal_length, 1.0e-9) ** (1.0 - self.config.exponent)
        delta = pos[:, None, :] - pos[None, :, :]
        distance_sq = torch.sum(delta * delta, dim=-1).clamp_min(1.0e-9)
        distance = torch.sqrt(distance_sq)
        diagonal = torch.eye(pos.shape[0], dtype=torch.bool, device=pos.device)
        distance = distance.masked_fill(diagonal, float("inf"))
        denominator = distance.pow(2.0 - self.config.exponent).unsqueeze(-1)
        pairwise_force = repulsive_scale * delta / denominator
        pairwise_force = pairwise_force.masked_fill(diagonal.unsqueeze(-1), 0.0)
        state.forces = forces + pairwise_force.sum(dim=1)
        return state


@dataclass(frozen=True)
class UniformSpringAttractionConfig:
    """Configuration for ``UniformSpringAttraction``.

    Parameters
    ----------
    k_formula : str, default="area"
        ``"area"`` uses the FR attraction law ``d^2 / k``.
        ``"explicit"`` uses GraphOpt's explicit spring length and constant.
    spring_length : float, default=0.0
        Explicit GraphOpt rest length used when ``k_formula="explicit"``.
    spring_constant : float, default=1.0
        Explicit GraphOpt spring constant used when ``k_formula="explicit"``.
    """

    k_formula: str = "area"
    spring_length: float = 0.0
    spring_constant: float = 1.0


@register_op
@dataclass(frozen=True)
class UniformSpringAttraction(Op):
    """Accumulate a uniform spring attraction term on edges."""

    config: UniformSpringAttractionConfig = field(default_factory=UniformSpringAttractionConfig)

    name: ClassVar[str] = "uniform_spring_attraction"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ("forces",)
    requires: ClassVar[Tuple[str, ...]] = ("pos", "forces")
    access_pattern: ClassVar[str] = "edge"

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Accumulate FR or GraphOpt spring forces on graph edges.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with spring forces added to ``state.forces``.
        """
        del ctx

        pos = _require_positions(state)
        forces = _require_forces(state)
        if problem.edge_index.numel() == 0:
            return state

        src = problem.edge_index[0].to(device=pos.device, dtype=torch.long)
        dst = problem.edge_index[1].to(device=pos.device, dtype=torch.long)
        delta = pos[src] - pos[dst]
        edge_weights = _edge_weights(problem=problem, device=pos.device, dtype=pos.dtype)
        updated = forces.clone()

        if self.config.k_formula == "area":
            optimal_distance = _resolve_area_k(problem=problem, state=state)
            contribution = -delta * (
                edge_weights * torch.linalg.vector_norm(delta, dim=1) / optimal_distance
            ).unsqueeze(1)
        elif self.config.k_formula == "explicit":
            distance = torch.linalg.vector_norm(delta, dim=1)
            mask = distance > _GRAPHOPT_MIN_DISTANCE
            if not bool(mask.any().item()):
                return state
            masked_distance = distance[mask]
            direction = delta[mask] / masked_distance.unsqueeze(1)
            stretch = (masked_distance - self.config.spring_length).abs()
            magnitude = 0.5 * self.config.spring_constant * stretch
            magnitude = magnitude * edge_weights[mask]
            source_sign = torch.where(
                masked_distance > self.config.spring_length,
                torch.full_like(masked_distance, -1.0),
                torch.full_like(masked_distance, 1.0),
            )
            contribution = torch.zeros_like(delta)
            contribution[mask] = direction * (magnitude * source_sign).unsqueeze(1)
        else:
            raise ValueError("k_formula must be either 'area' or 'explicit'.")

        updated.index_add_(0, src, contribution)
        updated.index_add_(0, dst, -contribution)
        state.forces = updated
        return state


@register_op
@dataclass(frozen=True)
class DesiredLengthSpringAttraction(Op):
    """Accumulate spring attraction using lengths from ``state.spring_lengths``."""

    name: ClassVar[str] = "desired_length_spring_attraction"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("pos", "spring_lengths")
    writes: ClassVar[Tuple[str, ...]] = ("forces",)
    requires: ClassVar[Tuple[str, ...]] = ("pos", "forces", "spring_lengths")
    access_pattern: ClassVar[str] = "edge"

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Accumulate GEM-style or per-edge desired-length spring forces.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with spring forces added to ``state.forces``.
        """
        del ctx

        pos = _require_positions(state)
        forces = _require_forces(state)
        if state.spring_lengths is None:
            raise ValueError("DesiredLengthSpringAttraction requires state.spring_lengths.")
        if problem.edge_index.numel() == 0:
            return state

        spring_lengths = state.spring_lengths.to(device=pos.device, dtype=pos.dtype)
        if spring_lengths.ndim != 1:
            raise ValueError("state.spring_lengths must be one-dimensional.")

        updated = forces.clone()
        src = problem.edge_index[0].to(device=pos.device, dtype=torch.long)
        dst = problem.edge_index[1].to(device=pos.device, dtype=torch.long)
        delta = pos[src] - pos[dst]
        distances = torch.linalg.vector_norm(delta, dim=1)
        edge_weights = _edge_weights(problem=problem, device=pos.device, dtype=pos.dtype)

        if spring_lengths.shape[0] == problem.num_nodes:
            degree_weights = _gem_degree_weights(
                problem=problem,
                state=state,
                device=pos.device,
                dtype=pos.dtype,
            )
            source_weights = degree_weights[src].clamp(min=1.0)
            target_weights = degree_weights[dst].clamp(min=1.0)
            source_desired = spring_lengths[src].clamp(min=_GEM_MIN_DISTANCE)
            target_desired = spring_lengths[dst].clamp(min=_GEM_MIN_DISTANCE)
            source_force = -delta * (distances / (source_desired * source_weights)).unsqueeze(1)
            target_force = delta * (distances / (target_desired * target_weights)).unsqueeze(1)
            source_force = source_force * edge_weights.unsqueeze(1)
            target_force = target_force * edge_weights.unsqueeze(1)
            updated.index_add_(0, src, source_force)
            updated.index_add_(0, dst, target_force)
        elif spring_lengths.shape[0] == problem.edge_index.shape[1]:
            desired = spring_lengths.clamp(min=_GRAPHOPT_MIN_DISTANCE)
            contribution = -delta * (distances / desired).unsqueeze(1) * edge_weights.unsqueeze(1)
            updated.index_add_(0, src, contribution)
            updated.index_add_(0, dst, -contribution)
        else:
            raise ValueError(
                "state.spring_lengths must have shape [N] for GEM-style lengths "
                "or [E] for per-edge desired lengths."
            )

        state.forces = updated
        return state


@dataclass(frozen=True)
class FA2DegreeCompensatedAttractionConfig:
    """Configuration for ``FA2DegreeCompensatedAttraction``.

    Parameters
    ----------
    linlog : bool, default=False
        Use the linlog attraction law.
    dissuade_hubs : bool, default=False
        Divide by source-node mass a second time.
    outbound_compensation : bool, default=True
        Apply FA2's mean-mass compensation.
    """

    linlog: bool = False
    dissuade_hubs: bool = False
    outbound_compensation: bool = True


@register_op
@dataclass(frozen=True)
class FA2DegreeCompensatedAttraction(Op):
    """Accumulate ForceAtlas2's degree-compensated attraction."""

    config: FA2DegreeCompensatedAttractionConfig = field(
        default_factory=FA2DegreeCompensatedAttractionConfig
    )

    name: ClassVar[str] = "fa2_degree_compensated_attraction"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ("forces",)
    requires: ClassVar[Tuple[str, ...]] = ("pos", "forces")
    access_pattern: ClassVar[str] = "edge"

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Accumulate the exact FA2 attraction term.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with attractive forces added to ``state.forces``.
        """
        del ctx

        pos = _require_positions(state)
        forces = _require_forces(state)
        mass = _fa2_mass(problem=problem, state=state, device=pos.device).to(dtype=pos.dtype)
        undirected_edges, undirected_weights = _fa2_unique_undirected_edges_with_weights(
            edge_index=problem.edge_index,
            edge_weights=problem.edge_weights,
        )
        if undirected_weights is not None:
            undirected_weights = undirected_weights.to(device=pos.device, dtype=pos.dtype)
        outbound_att_compensation = (
            float(mass.mean().item()) if self.config.outbound_compensation else 1.0
        )
        contribution = _fa2_attraction_force(
            pos=pos,
            edge_index=undirected_edges.to(device=pos.device),
            mass=mass,
            outbound_att_compensation=outbound_att_compensation,
            outbound_attraction_distribution=self.config.outbound_compensation,
            linlog=self.config.linlog,
            edge_weights=undirected_weights,
            dissuade_hubs=self.config.dissuade_hubs,
            edge_weight_influence=1.0,
        )
        state.forces = forces + contribution
        return state


@dataclass(frozen=True)
class GravityToOriginConfig:
    """Configuration for ``GravityToOrigin``.

    Parameters
    ----------
    strength : float, default=1.0
        Gravity coefficient.
    strong_mode : bool, default=False
        Use ForceAtlas2's strong-gravity mode.
    """

    strength: float = 1.0
    strong_mode: bool = False


@register_op
@dataclass(frozen=True)
class GravityToOrigin(Op):
    """Accumulate ForceAtlas2 gravity toward the origin."""

    config: GravityToOriginConfig = field(default_factory=GravityToOriginConfig)

    name: ClassVar[str] = "gravity_to_origin"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ("forces",)
    requires: ClassVar[Tuple[str, ...]] = ("pos", "forces")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Accumulate ForceAtlas2 gravity.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with gravity forces added to ``state.forces``.
        """
        del ctx

        pos = _require_positions(state)
        forces = _require_forces(state)
        mass = _fa2_mass(problem=problem, state=state, device=pos.device).to(dtype=pos.dtype)
        contribution = _fa2_gravity_force(
            pos=pos,
            mass=mass,
            gravity=self.config.strength,
            strong_gravity=self.config.strong_mode,
            scaling_ratio=1.0,
        )
        state.forces = forces + contribution
        return state


@dataclass(frozen=True)
class GravityToBarycenterConfig:
    """Configuration for ``GravityToBarycenter``.

    Parameters
    ----------
    constant : float, default=1/16
        GEM gravitational constant.
    """

    constant: float = 1.0 / 16.0


@register_op
@dataclass(frozen=True)
class GravityToBarycenter(Op):
    """Accumulate GEM gravity toward the weighted barycenter."""

    config: GravityToBarycenterConfig = field(default_factory=GravityToBarycenterConfig)

    name: ClassVar[str] = "gravity_to_barycenter"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ("forces",)
    requires: ClassVar[Tuple[str, ...]] = ("pos", "forces")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Accumulate GEM's weighted barycenter gravity.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with gravity forces added to ``state.forces``.
        """
        del ctx

        pos = _require_positions(state)
        forces = _require_forces(state)
        degree_weights = _gem_degree_weights(
            problem=problem,
            state=state,
            device=pos.device,
            dtype=pos.dtype,
        )
        barycenter = (pos * degree_weights.unsqueeze(1)).sum(dim=0, keepdim=True)
        barycenter = barycenter / max(int(pos.shape[0]), 1)
        contribution = (barycenter - pos) * self.config.constant
        state.forces = forces + contribution
        return state


@dataclass(frozen=True)
class BarnesHutForceConfig:
    """Configuration for ``BarnesHutForce``.

    Parameters
    ----------
    theta : float, default=1.2
        Barnes-Hut opening threshold.
    """

    theta: float = 1.2


@register_op
@dataclass(frozen=True)
class BarnesHutForce(Op):
    """Accumulate Barnes-Hut approximate repulsion from ``state.extras``."""

    config: BarnesHutForceConfig = field(default_factory=BarnesHutForceConfig)

    name: ClassVar[str] = "barnes_hut_force"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("pos", "extras.quadtree")
    writes: ClassVar[Tuple[str, ...]] = ("forces",)
    requires: ClassVar[Tuple[str, ...]] = ("pos", "forces")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Accumulate Barnes-Hut forces from a prebuilt quadtree.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with approximate repulsive forces added to ``state.forces``.
        """
        del ctx

        pos = _require_positions(state)
        forces = _require_forces(state)
        if "quadtree" not in state.extras:
            raise ValueError("BarnesHutForce requires state.extras['quadtree'].")

        quadtree = state.extras["quadtree"]
        if isinstance(quadtree, _FA2BarnesHutNode):
            mass = _fa2_mass(problem=problem, state=state, device=pos.device).to(dtype=pos.dtype)
            contribution = _fa2_barnes_hut_force(
                quadtree=quadtree,
                pos=pos,
                mass=mass,
                theta=self.config.theta,
            )
        elif isinstance(quadtree, _SFDPQuadTreeNode):
            contribution = _sfdp_barnes_hut_force(
                quadtree=quadtree,
                pos=pos,
                theta=self.config.theta,
            )
        else:
            raise ValueError("Unsupported quadtree type for BarnesHutForce.")

        state.forces = forces + contribution
        return state


@dataclass(frozen=True)
class DensityGridForceConfig:
    """Configuration for ``DensityGridForce``.

    Parameters
    ----------
    grid_size : int, default=1000
        Density grid width.
    view_size : float, default=4000.0
        Density grid view box size.
    radius : int, default=10
        Tent-kernel radius in grid cells.
    """

    grid_size: int = 1000
    view_size: float = 4000.0
    radius: int = 10


@register_op
@dataclass(frozen=True)
class DensityGridForce(Op):
    """Accumulate a force from a density-grid energy proxy."""

    config: DensityGridForceConfig = field(default_factory=DensityGridForceConfig)

    name: ClassVar[str] = "density_grid_force"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("pos", "extras.density_grid")
    writes: ClassVar[Tuple[str, ...]] = ("forces",)
    requires: ClassVar[Tuple[str, ...]] = ("pos", "forces")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Estimate density-grid forces by differentiating the proxy energy.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with density-grid forces added to ``state.forces``.
        """
        del problem, ctx

        pos = _require_positions(state)
        forces = _require_forces(state)
        if "density_grid" not in state.extras:
            raise ValueError("DensityGridForce requires state.extras['density_grid'].")

        density_grid = state.extras["density_grid"]
        updated = forces.clone()
        for node_index in range(pos.shape[0]):
            updated[node_index] += _density_grid_gradient(
                density_grid=density_grid,
                node_index=node_index,
                pos=pos,
            )
        state.forces = updated
        return state


@dataclass(frozen=True)
class CellGridForceConfig:
    """Configuration for ``CellGridForce``.

    Parameters
    ----------
    cell_size : float or None, default=None
        Sparse-grid cell size. ``None`` follows LGL's ``area**0.25`` default.
    repulse_rad : float or None, default=None
        Repulsion radius. ``None`` follows LGL's ``area * N`` default.
    """

    cell_size: Optional[float] = None
    repulse_rad: Optional[float] = None


@register_op
@dataclass(frozen=True)
class CellGridForce(Op):
    """Accumulate LGL's sparse-cell local repulsion."""

    config: CellGridForceConfig = field(default_factory=CellGridForceConfig)

    name: ClassVar[str] = "cell_grid_force"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ("forces",)
    requires: ClassVar[Tuple[str, ...]] = ("pos", "forces")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Accumulate LGL's local cell-grid repulsion on all nodes.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with repulsive forces added to ``state.forces``.

        Notes
        -----
        When ``cell_size`` is not configured, this op follows LGL's default
        relation ``cell_size = area**0.25`` with ``area = N^2``.
        """
        del ctx

        pos = _require_positions(state)
        forces = _require_forces(state)
        if pos.shape[0] <= 1:
            return state

        num_nodes = max(problem.num_nodes, 1)
        area = float(num_nodes * num_nodes)
        cell_size = area**0.25 if self.config.cell_size is None else self.config.cell_size
        repulse_rad = (
            area * float(num_nodes) if self.config.repulse_rad is None else self.config.repulse_rad
        )
        if cell_size is None or cell_size <= 0.0:
            raise ValueError("cell_size must be positive.")
        if repulse_rad <= 0.0:
            raise ValueError("repulse_rad must be positive.")

        frk = math.sqrt(area / float(num_nodes))
        buckets: dict[Tuple[int, int], list[int]] = {}
        safe_cell_size = max(float(cell_size), _LGL_MIN_DISTANCE)
        for node in range(pos.shape[0]):
            x_value = float(pos[node, 0].item())
            y_value = float(pos[node, 1].item())
            key = (
                int(math.floor(x_value / safe_cell_size)),
                int(math.floor(y_value / safe_cell_size)),
            )
            buckets.setdefault(key, []).append(node)

        updated = forces.clone()
        sorted_cells = sorted(buckets)
        for cell in sorted_cells:
            nodes_here = buckets[cell]
            for offset_y in (-1, 0, 1):
                for offset_x in (-1, 0, 1):
                    neighbor_cell = (cell[0] + offset_x, cell[1] + offset_y)
                    if neighbor_cell not in buckets or neighbor_cell < cell:
                        continue
                    nodes_there = buckets[neighbor_cell]
                    if neighbor_cell == cell:
                        pair_iter = [
                            (nodes_here[left_index], nodes_here[right_index])
                            for left_index in range(len(nodes_here))
                            for right_index in range(left_index + 1, len(nodes_here))
                        ]
                    else:
                        pair_iter = [(left, right) for left in nodes_here for right in nodes_there]

                    for left, right in pair_iter:
                        delta = pos[left] - pos[right]
                        distance = float(torch.linalg.vector_norm(delta).item())
                        if distance >= safe_cell_size:
                            continue
                        safe_distance = max(distance, _LGL_MIN_DISTANCE)
                        direction = delta / safe_distance
                        magnitude = (frk * frk) * (
                            (1.0 / safe_distance) - ((safe_distance * safe_distance) / repulse_rad)
                        )
                        contribution = direction * magnitude
                        updated[left] += contribution
                        updated[right] -= contribution

        state.forces = updated
        return state


@register_op
class ApplyDisplacement(Op):
    """Normalize the accumulated force and clamp motion by temperature."""

    name: ClassVar[str] = "apply_displacement"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("forces", "temperature")
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos", "forces", "temperature")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Apply FR-style temperature-clamped displacement.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with updated positions.
        """
        del problem, ctx

        pos = _require_positions(state)
        forces = _require_forces(state)
        if state.temperature is None:
            raise ValueError("ApplyDisplacement requires state.temperature.")

        length = torch.linalg.vector_norm(forces, dim=1).clamp(min=_FR_MIN_DISTANCE)
        delta_pos = forces * (float(state.temperature) / length).unsqueeze(1)
        state.pos = pos + delta_pos
        return state


@dataclass(frozen=True)
class AdaptiveSpeedApplyConfig:
    """Configuration for ``AdaptiveSpeedApply``.

    Parameters
    ----------
    jitter_tolerance : float, default=1.0
        ForceAtlas2 jitter-tolerance hyperparameter.
    """

    jitter_tolerance: float = 1.0


@register_op
@dataclass(frozen=True)
class AdaptiveSpeedApply(Op):
    """Apply ForceAtlas2's adaptive global-speed update."""

    config: AdaptiveSpeedApplyConfig = field(default_factory=AdaptiveSpeedApplyConfig)

    name: ClassVar[str] = "adaptive_speed_apply"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("forces", "old_forces")
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos", "forces", "old_forces")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Apply FA2's speed controller using the accumulated forces.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with updated positions.
        """
        del ctx

        pos = _require_positions(state)
        forces = _require_forces(state)
        if state.old_forces is None:
            raise ValueError("AdaptiveSpeedApply requires state.old_forces.")

        mass = _fa2_mass(problem=problem, state=state, device=pos.device).to(dtype=pos.dtype)
        speed = float(state.extras.get("fa2_speed", 1.0))
        speed_efficiency = float(state.extras.get("fa2_speed_efficiency", 1.0))
        updated_pos, new_speed, new_speed_efficiency = _fa2_adjust_speed_and_apply_forces(
            pos=pos,
            force=forces,
            old_force=state.old_forces.to(device=pos.device, dtype=pos.dtype),
            mass=mass,
            speed=speed,
            speed_efficiency=speed_efficiency,
            jitter_tolerance=self.config.jitter_tolerance,
        )
        state.pos = updated_pos
        state.extras["fa2_speed"] = new_speed
        state.extras["fa2_speed_efficiency"] = new_speed_efficiency
        return state


@register_op
class GEMNodeTick(Op):
    """Apply one sequential GEM node update from the current force field."""

    name: ClassVar[str] = "gem_node_tick"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("forces", "pos")
    writes: ClassVar[Tuple[str, ...]] = ("pos", "extras")
    requires: ClassVar[Tuple[str, ...]] = ("pos", "forces")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Apply one OGDF-style sequential GEM tick.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with one node moved and GEM extras updated.
        """
        pos = _require_positions(state)
        forces = _require_forces(state)

        node_index = _resolve_gem_node_index(problem=problem, state=state, ctx=ctx)
        degree_weights = _gem_degree_weights(
            problem=problem,
            state=state,
            device=pos.device,
            dtype=pos.dtype,
        )
        previous_impulse = cast(
            torch.Tensor,
            state.extras.get(
                "gem_previous_impulses",
                torch.zeros((problem.num_nodes, 2), device=pos.device, dtype=pos.dtype),
            ),
        ).to(device=pos.device, dtype=pos.dtype)
        local_temperatures = cast(
            torch.Tensor,
            state.extras.get(
                "gem_local_temperatures",
                torch.full(
                    (problem.num_nodes,),
                    _GEM_INITIAL_TEMPERATURE,
                    device=pos.device,
                    dtype=pos.dtype,
                ),
            ),
        ).to(device=pos.device, dtype=pos.dtype)
        skew_gauge = cast(
            torch.Tensor,
            state.extras.get(
                "gem_skew_gauge",
                torch.zeros((problem.num_nodes,), device=pos.device, dtype=pos.dtype),
            ),
        ).to(device=pos.device, dtype=pos.dtype)
        barycenter = cast(
            torch.Tensor,
            state.extras.get(
                "gem_barycenter",
                (pos * degree_weights.unsqueeze(1)).sum(dim=0),
            ),
        ).to(device=pos.device, dtype=pos.dtype)
        global_temperature = float(
            state.extras.get("gem_global_temperature", _GEM_INITIAL_TEMPERATURE)
        )

        global_temperature = _gem_apply_node_update(
            node_index=node_index,
            pos=pos,
            impulse=forces[node_index],
            previous_impulse=previous_impulse,
            local_temperatures=local_temperatures,
            skew_gauge=skew_gauge,
            degree_weights=degree_weights,
            barycenter=barycenter,
            global_temperature=global_temperature,
        )

        state.pos = pos
        state.extras["gem_previous_impulses"] = previous_impulse
        state.extras["gem_local_temperatures"] = local_temperatures
        state.extras["gem_skew_gauge"] = skew_gauge
        state.extras["gem_barycenter"] = barycenter
        state.extras["gem_global_temperature"] = global_temperature
        state.extras["gem_last_node_index"] = node_index
        return state


@dataclass(frozen=True)
class StressSGDPairUpdateConfig:
    """Configuration for ``StressSGDPairUpdate``.

    Parameters
    ----------
    clamp_mu : float, default=1.0
        Upper bound for the pair step coefficient ``mu``.
    """

    clamp_mu: float = 1.0


@register_op
@dataclass(frozen=True)
class StressSGDPairUpdate(Op):
    """Apply one sequential Stress-SGD pair update."""

    config: StressSGDPairUpdateConfig = field(default_factory=StressSGDPairUpdateConfig)

    name: ClassVar[str] = "stress_sgd_pair_update"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("pos", "distance_matrix")
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Apply one exact sequential pair update from the Stress-SGD kernel.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with updated positions.
        """
        del problem, ctx

        pos = _require_positions(state)
        if "stress_sgd_pair" not in state.extras:
            raise ValueError("StressSGDPairUpdate requires state.extras['stress_sgd_pair'].")
        if "stress_sgd_eta" not in state.extras:
            raise ValueError("StressSGDPairUpdate requires state.extras['stress_sgd_eta'].")

        source_index, target_index = cast(tuple[int, int], tuple(state.extras["stress_sgd_pair"]))
        eta = float(state.extras["stress_sgd_eta"])
        if state.distance_matrix is not None:
            target_distance = float(state.distance_matrix[source_index, target_index].item())
        elif "stress_sgd_target_distance" in state.extras:
            target_distance = float(state.extras["stress_sgd_target_distance"])
        else:
            raise ValueError(
                "StressSGDPairUpdate requires state.distance_matrix or "
                "state.extras['stress_sgd_target_distance']."
            )

        if "stress_sgd_weight" in state.extras:
            weight = float(state.extras["stress_sgd_weight"])
        else:
            weight = 1.0 / max(target_distance * target_distance, 1.0e-12)

        mu = min(eta * weight, self.config.clamp_mu)
        dx = float(pos[source_index, 0].item() - pos[target_index, 0].item())
        dy = float(pos[source_index, 1].item() - pos[target_index, 1].item())
        magnitude = math.hypot(dx, dy)
        if magnitude <= 0.0:
            return state

        ratio = mu * (magnitude - target_distance) / (2.0 * magnitude)
        pos[source_index, 0] -= ratio * dx
        pos[target_index, 0] += ratio * dx
        pos[source_index, 1] -= ratio * dy
        pos[target_index, 1] += ratio * dy
        state.pos = pos
        return state


@register_op
class StressMajNodeSweep(Op):
    """Apply one dense SMACOF majorization sweep."""

    name: ClassVar[str] = "stress_majorization_node_sweep"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("pos", "distance_matrix")
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos", "distance_matrix")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Apply one SMACOF majorization update using ``state.distance_matrix``.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with updated positions.
        """
        del problem, ctx

        pos = _require_positions(state)
        if state.distance_matrix is None:
            raise ValueError("StressMajNodeSweep requires state.distance_matrix.")

        target_distances = (
            state.distance_matrix.detach().cpu().numpy().astype(np.float64, copy=True)
        )
        with np.errstate(divide="ignore"):
            weights = np.where(target_distances > 0.0, 1.0 / np.square(target_distances), 0.0)
        np.fill_diagonal(weights, 0.0)

        if "stress_maj_laplacian_pinv" in state.extras:
            laplacian_pinv = np.asarray(state.extras["stress_maj_laplacian_pinv"], dtype=np.float64)
        else:
            laplacian = -weights
            np.fill_diagonal(laplacian, weights.sum(axis=1))
            laplacian_pinv = np.linalg.pinv(laplacian)

        updated = _stress_maj_smacof_update(
            positions=pos.detach().cpu().numpy().astype(np.float64, copy=True),
            target_distances=target_distances,
            weights=weights,
            laplacian_pinv=laplacian_pinv,
        )
        state.pos = torch.from_numpy(updated).to(device=pos.device, dtype=pos.dtype)
        return state
