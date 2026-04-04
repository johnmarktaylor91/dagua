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

from dagua.layout.ops.base import Op
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_FR_MIN_DISTANCE = 0.01
_GRAPHOPT_COULOMBS_CONSTANT = 8_987_500_000.0
_GRAPHOPT_MIN_DISTANCE = 1.0e-12
_GRAPHOPT_MAX_REPULSION_DISTANCE = 500.0
_LGL_MIN_DISTANCE = 1.0e-12
_DENSITY_EPSILON_SCALE = 1.0

_GRAPHOPT_PAIR_SOURCE_KEY = "graphopt_pair_source"
_GRAPHOPT_PAIR_TARGET_KEY = "graphopt_pair_target"
_GRAPHOPT_SPRING_EDGES_KEY = "graphopt_spring_edges"
_GRAPHOPT_SPRING_WEIGHTS_KEY = "graphopt_spring_weights"
_GRAPHOPT_MAX_REPULSION_DISTANCE_SQ_KEY = "graphopt_max_repulsion_distance_sq"
_GEM_INITIAL_TEMPERATURE = 12.0
_GEM_MIN_DISTANCE = 1.0e-9
_GEM_OSCILLATION_COSINE_THRESHOLD = math.cos(0.5 * (math.pi / 2.0))
_GEM_OSCILLATION_SENSITIVITY = 0.3
_GEM_ROTATION_SENSITIVITY = 0.01
_GEM_ROTATION_SINE_THRESHOLD = math.sin((math.pi / 2.0) + (math.pi / 3.0 / 2.0))
_EPSILON = 1.0e-9
_MIN_DISTANCE = 1.0e-9


@dataclass
class _QuadTreeNode:
    """Barnes-Hut quadtree node used for large-graph repulsion.

    Parameters
    ----------
    center : torch.Tensor
        Cell center with shape ``[2]``.
    half_width : float
        Half-width of the square cell.
    indices : list[int]
        Point indices stored under this node.
    level : int
        Depth within the quadtree.
    mass : float, default=0.0
        Aggregate number of points represented by the cell.
    center_of_mass : torch.Tensor, optional
        Mean position of points represented by the cell.
    children : list[_QuadTreeNode], optional
        Child quadrants. Empty means the node is a leaf.
    """

    center: torch.Tensor
    half_width: float
    indices: list[int]
    level: int
    mass: float = 0.0
    center_of_mass: Optional[torch.Tensor] = None
    children: list["_QuadTreeNode"] = field(default_factory=list)


_SFDPQuadTreeNode = _QuadTreeNode


def _barnes_hut_force_for_index(
    node: _QuadTreeNode,
    positions: torch.Tensor,
    index: int,
    theta: float,
    repulsive_scale: float,
    repulsive_exponent: float,
) -> torch.Tensor:
    """Evaluate the Barnes-Hut repulsive force on one node.

    Parameters
    ----------
    node : _QuadTreeNode
        Current quadtree node.
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    index : int
        Query node index.
    theta : float
        Barnes-Hut opening angle threshold.
    repulsive_scale : float
        Global repulsion multiplier.
    repulsive_exponent : float
        SFDP repulsion exponent ``p``.

    Returns
    -------
    torch.Tensor
        Repulsive force vector with shape ``[2]``.
    """
    if node.mass <= 0.0 or node.center_of_mass is None:
        return torch.zeros(2, dtype=torch.float32)

    query = positions[index]
    if len(node.indices) == 1 and node.indices[0] == index:
        return torch.zeros(2, dtype=torch.float32)

    if node.children:
        delta = query - node.center_of_mass
        distance = float(torch.linalg.vector_norm(delta).item())
        width = node.half_width * 2.0
        if index not in node.indices and distance > _EPSILON and (width / distance) < theta:
            denominator = max(distance, _EPSILON) ** (2.0 - repulsive_exponent)
            return repulsive_scale * node.mass * delta / denominator

        force = torch.zeros(2, dtype=torch.float32)
        for child in node.children:
            force = force + _barnes_hut_force_for_index(
                node=child,
                positions=positions,
                index=index,
                theta=theta,
                repulsive_scale=repulsive_scale,
                repulsive_exponent=repulsive_exponent,
            )
        return force

    if len(node.indices) == 0:
        return torch.zeros(2, dtype=torch.float32)

    leaf_indices = [point_index for point_index in node.indices if point_index != index]
    if not leaf_indices:
        return torch.zeros(2, dtype=torch.float32)

    coords = positions[torch.tensor(leaf_indices, dtype=torch.long)]
    delta = query.unsqueeze(0) - coords
    distance = torch.linalg.vector_norm(delta, dim=1).clamp_min(_EPSILON)
    denominator = distance.pow(2.0 - repulsive_exponent).unsqueeze(1)
    return (repulsive_scale * delta / denominator).sum(dim=0)


def _pairwise_distances(positions: np.ndarray) -> np.ndarray:
    """Compute dense Euclidean distances between all node pairs.

    Parameters
    ----------
    positions : numpy.ndarray
        Position array with shape ``[N, 2]``.

    Returns
    -------
    numpy.ndarray
        Pairwise Euclidean distances with shape ``[N, N]``.
    """
    deltas = positions[:, None, :] - positions[None, :, :]
    return np.sqrt(np.sum(deltas * deltas, axis=2))


def _smacof_update(
    positions: np.ndarray,
    target_distances: np.ndarray,
    weights: np.ndarray,
    laplacian_pinv: np.ndarray,
) -> np.ndarray:
    """Apply one SMACOF majorization step.

    Parameters
    ----------
    positions : numpy.ndarray
        Current positions with shape ``[N, 2]``.
    target_distances : numpy.ndarray
        Desired graph distances with shape ``[N, N]``.
    weights : numpy.ndarray
        SMACOF weight matrix with shape ``[N, N]``.
    laplacian_pinv : np.ndarray
        Pseudoinverse of the weighted Laplacian with shape ``[N, N]``.

    Returns
    -------
    numpy.ndarray
        Updated centered positions with shape ``[N, 2]``.
    """
    current_distances = np.maximum(_pairwise_distances(positions), _MIN_DISTANCE)
    ratio = np.zeros_like(target_distances)
    active_mask = weights > 0.0
    ratio[active_mask] = target_distances[active_mask] / current_distances[active_mask]

    b_matrix = -weights * ratio
    np.fill_diagonal(b_matrix, 0.0)
    np.fill_diagonal(b_matrix, -b_matrix.sum(axis=1))

    updated = laplacian_pinv @ (b_matrix @ positions)
    return updated - updated.mean(axis=0, keepdims=True)


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
    Callers can override that by setting ``state.force_area``.
    """
    area = 1.0 if state.force_area is None else state.force_area
    if not isinstance(area, (int, float)):
        raise ValueError("state.force_area must be a real number.")
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
    degree = torch.zeros((num_nodes,), dtype=torch.float32, device=device)
    if edge_index.numel() == 0:
        return degree

    source = edge_index[0].to(device=device, dtype=torch.long)
    target = edge_index[1].to(device=device, dtype=torch.long)
    non_self = source != target
    if not bool(non_self.any().item()):
        return degree

    lower = torch.minimum(source[non_self], target[non_self])
    upper = torch.maximum(source[non_self], target[non_self])
    pairs = torch.stack([lower, upper], dim=1)
    unique_pairs = torch.unique(pairs, dim=0)
    ones = torch.ones(unique_pairs.shape[0], dtype=torch.float32, device=device)
    degree.scatter_add_(0, unique_pairs[:, 0], ones)
    degree.scatter_add_(0, unique_pairs[:, 1], ones)
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


_sfdp_barnes_hut_force_for_index = _barnes_hut_force_for_index
_stress_maj_smacof_update = _smacof_update


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
        a unit-square default area and an optional ``state.force_area`` override.
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
    reads: ClassVar[Tuple[str, ...]] = ("pos", "dense_adjacency")
    writes: ClassVar[Tuple[str, ...]] = ("forces",)
    requires: ClassVar[Tuple[str, ...]] = ("pos", "dense_adjacency")

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
            If ``state.dense_adjacency`` is missing or malformed.
        """
        del ctx

        pos = _require_positions(state)
        adjacency = state.dense_adjacency
        if not isinstance(adjacency, torch.Tensor):
            raise ValueError("FRCombinedForce requires state.dense_adjacency.")
        if tuple(adjacency.shape) != (problem.num_nodes, problem.num_nodes):
            raise ValueError(
                f"state.dense_adjacency must have shape ({problem.num_nodes}, {problem.num_nodes})."
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


@dataclass(frozen=True)
class GraphOptApplyConfig:
    """Configuration for :class:`GraphOptApply`.

    Parameters
    ----------
    node_mass : float
        Shared mass used to convert force into displacement.
    max_sa_movement : float
        Maximum absolute displacement per axis per iteration.
    """

    node_mass: float = 30.0
    max_sa_movement: float = 5.0


@register_op
@dataclass(frozen=True)
class GraphOptApplyDisplacement(Op):
    """Apply GraphOpt-style clamp-to-axis-step displacement."""

    config: GraphOptApplyConfig = field(default_factory=GraphOptApplyConfig)

    name: ClassVar[str] = "graphopt_apply_displacement"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("pos", "forces")
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos", "forces")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Update positions with a clamped force/mass step.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context.

        Returns
        -------
        SolveState
            State with updated positions.
        """
        del problem, ctx

        if self.config.node_mass <= 0.0:
            raise ValueError("GraphOptApplyDisplacement node_mass must be positive.")

        positions = _require_positions(state)
        forces = _require_forces(state)
        movement = torch.clamp(
            forces / float(self.config.node_mass),
            min=-float(self.config.max_sa_movement),
            max=float(self.config.max_sa_movement),
        )
        state.pos = positions + movement
        return state


@dataclass(frozen=True)
class GraphOptPrepareStateConfig:
    """Configuration for :class:`GraphOptPrepareState`.

    Parameters
    ----------
    spring_max_distance : float, default=500.0
        Maximum pairwise distance for GraphOpt coulomb repulsion.
    """

    spring_max_distance: float = 500.0


@register_op
@dataclass(frozen=True)
class GraphOptPrepareState(Op):
    """Prepare and cache GraphOpt-specific data in ``state.extras``.

    Notes
    -----
    This op filters self-loops from the spring edge set (preserving duplicate
    and reciprocal edges), stores edge-level spring weights, and precomputes the
    all-pairs upper-triangle index pairs for repulsion.
    """

    config: GraphOptPrepareStateConfig = field(default_factory=GraphOptPrepareStateConfig)

    name: ClassVar[str] = "graphopt_prepare_state"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    writes: ClassVar[Tuple[str, ...]] = ("extras",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Populate cached pair indices and spring metadata for one GraphOpt solve.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure (unused).

        Returns
        -------
        SolveState
            State with GraphOpt-specific tensors populated in ``extras``.
        """
        del ctx

        if problem.edge_index.numel() == 0:
            state.extras[_GRAPHOPT_SPRING_EDGES_KEY] = torch.empty((2, 0), dtype=torch.long)
            state.extras[_GRAPHOPT_SPRING_WEIGHTS_KEY] = None
        else:
            edges = problem.edge_index.to(device="cpu", dtype=torch.long)
            non_self = edges[0] != edges[1]
            if not bool(non_self.any().item()):
                state.extras[_GRAPHOPT_SPRING_EDGES_KEY] = torch.empty((2, 0), dtype=torch.long)
                state.extras[_GRAPHOPT_SPRING_WEIGHTS_KEY] = None
            else:
                filtered_edges = edges[:, non_self].contiguous()
                state.extras[_GRAPHOPT_SPRING_EDGES_KEY] = filtered_edges

                if problem.edge_weights is not None:
                    spring_weights = (
                        problem.edge_weights.detach()
                        .to(device="cpu", dtype=torch.float64)[non_self]
                        .contiguous()
                    )
                    state.extras[_GRAPHOPT_SPRING_WEIGHTS_KEY] = spring_weights
                else:
                    state.extras[_GRAPHOPT_SPRING_WEIGHTS_KEY] = None

        pair_source, pair_target = torch.triu_indices(
            problem.num_nodes,
            problem.num_nodes,
            offset=1,
        )
        state.extras[_GRAPHOPT_PAIR_SOURCE_KEY] = pair_source
        state.extras[_GRAPHOPT_PAIR_TARGET_KEY] = pair_target
        max_distance_sq = float(self.config.spring_max_distance) * float(
            self.config.spring_max_distance
        )
        state.extras[_GRAPHOPT_MAX_REPULSION_DISTANCE_SQ_KEY] = max_distance_sq
        return state


@dataclass(frozen=True)
class GraphOptIterationConfig:
    """Configuration for :class:`GraphOptIteration`.

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


@register_op
@dataclass(frozen=True)
class GraphOptIteration(Op):
    """Execute one exact GraphOpt force-and-move step.

    Notes
    -----
    This op uses the state cached by :class:`GraphOptPrepareState` so repeated
    iterations avoid rebuilding repulsion and spring indices each step.
    """

    config: GraphOptIterationConfig = field(default_factory=GraphOptIterationConfig)

    name: ClassVar[str] = "graphopt_iteration"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ("pos", "forces")
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Update positions by one GraphOpt force application.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with GraphOpt cache keys populated.
        ctx : RuntimeContext
            Execution context (unused).

        Returns
        -------
        SolveState
            State with updated positions and latest forces.
        """
        del problem, ctx

        if state.pos is None:
            raise ValueError("GraphOptIteration requires state.pos to be set.")

        spring_edges = state.extras.get(_GRAPHOPT_SPRING_EDGES_KEY)
        spring_weights = state.extras.get(_GRAPHOPT_SPRING_WEIGHTS_KEY)
        if not isinstance(spring_edges, torch.Tensor):
            raise ValueError(
                "GraphOptIteration requires state.extras['graphopt_spring_edges'] to be set."
            )

        pair_source = state.extras.get(_GRAPHOPT_PAIR_SOURCE_KEY)
        pair_target = state.extras.get(_GRAPHOPT_PAIR_TARGET_KEY)
        if (
            not isinstance(pair_source, torch.Tensor)
            or not isinstance(pair_target, torch.Tensor)
            or pair_source.shape != pair_target.shape
        ):
            raise ValueError(
                "GraphOptIteration requires repulsion index buffers "
                "state.extras['graphopt_pair_source/target'] to be set."
            )
        max_repulsion_distance_sq = state.extras.get(_GRAPHOPT_MAX_REPULSION_DISTANCE_SQ_KEY)
        if not isinstance(max_repulsion_distance_sq, float):
            max_repulsion_distance_sq = float(
                _GRAPHOPT_MAX_REPULSION_DISTANCE * _GRAPHOPT_MAX_REPULSION_DISTANCE
            )

        positions = state.pos
        forces = torch.zeros_like(positions)
        num_nodes = int(positions.shape[0])

        if self.config.node_charge != 0.0 and num_nodes > 1:
            delta = positions[pair_source] - positions[pair_target]
            distance_sq = delta.square().sum(dim=1)
            mask = (distance_sq > _GRAPHOPT_MIN_DISTANCE) & (
                distance_sq < torch.as_tensor(max_repulsion_distance_sq, device=distance_sq.device)
            )
            if bool(mask.any().item()):
                pair_delta = delta[mask]
                pair_distance_sq = distance_sq[mask]
                pair_distance = torch.sqrt(pair_distance_sq)
                direction = pair_delta / pair_distance.unsqueeze(1)
                magnitude = (
                    _GRAPHOPT_COULOMBS_CONSTANT
                    * (float(self.config.node_charge) * float(self.config.node_charge))
                    / pair_distance_sq
                )
                contribution = direction * magnitude.unsqueeze(1)
                forces.index_add_(0, pair_source[mask], contribution)
                forces.index_add_(0, pair_target[mask], -contribution)

        if spring_edges.numel() > 0:
            source = spring_edges[0]
            target = spring_edges[1]
            delta = positions[source] - positions[target]
            distance = torch.linalg.vector_norm(delta, dim=1)
            mask = distance > _GRAPHOPT_MIN_DISTANCE
            if bool(mask.any().item()):
                masked_distance = distance[mask]
                direction = delta[mask] / masked_distance.unsqueeze(1)
                stretch = (masked_distance - float(self.config.spring_length)).abs()
                magnitude = 0.5 * float(self.config.spring_constant) * stretch
                if spring_weights is not None:
                    if not isinstance(spring_weights, torch.Tensor):
                        raise ValueError(
                            "GraphOptIteration expects state.extras['graphopt_spring_weights'] "
                            "to be a torch.Tensor when present."
                        )
                    magnitude = magnitude * spring_weights[mask].to(dtype=magnitude.dtype)
                source_sign = torch.where(
                    masked_distance > float(self.config.spring_length),
                    torch.full_like(masked_distance, -1.0),
                    torch.full_like(masked_distance, 1.0),
                )
                contribution = direction * (magnitude * source_sign).unsqueeze(1)
                forces.index_add_(0, source[mask], contribution)
                forces.index_add_(0, target[mask], -contribution)

        movement = torch.clamp(
            forces / float(self.config.node_mass),
            min=-float(self.config.max_sa_movement),
            max=float(self.config.max_sa_movement),
        )
        state.forces = forces
        state.pos = positions + movement
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
class FA2ForceStepConfig:
    """Configuration for ``FA2ForceStep``.

    Parameters
    ----------
    gravity : float, default=1.0
        Gravity coefficient.
    scaling_ratio : float, default=2.0
        Repulsion scaling coefficient.
    linlog : bool, default=False
        Whether to use logarithmic attraction.
    strong_gravity : bool, default=False
        Whether to use ForceAtlas2's strong-gravity mode.
    outbound_attraction_distribution : bool, default=True
        Whether to divide attraction by source-node mass.
    dissuade_hubs : bool, default=False
        Whether to divide attraction by source-node mass a second time.
    edge_weight_influence : float, default=1.0
        Exponent applied to edge weights before attraction.
    barnes_hut : bool, default=False
        Whether to approximate repulsion with Barnes-Hut.
    barnes_hut_theta : float, default=1.2
        Barnes-Hut opening threshold.
    jitter_tolerance : float, default=1.0
        Adaptive speed-controller jitter tolerance.
    """

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


@register_op
@dataclass(frozen=True)
class FA2ForceStep(Op):
    """Apply one full ForceAtlas2 iteration."""

    config: FA2ForceStepConfig = field(default_factory=FA2ForceStepConfig)

    name: ClassVar[str] = "fa2_force_step"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("pos", "old_forces", "degree", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("pos", "forces", "old_forces", "extras")
    requires: ClassVar[Tuple[str, ...]] = ("pos", "old_forces")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute FA2 forces and apply the adaptive speed controller.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with initialized FA2 caches.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this deterministic op.

        Returns
        -------
        SolveState
            State with updated positions, forces, and adaptive-speed caches.
        """
        del problem, ctx

        pos = _require_positions(state)
        if state.old_forces is None:
            raise ValueError("FA2ForceStep requires state.old_forces.")
        if "fa2_undirected_edges" not in state.extras:
            raise ValueError("FA2ForceStep requires state.extras['fa2_undirected_edges'].")
        if "fa2_mass" not in state.extras:
            raise ValueError("FA2ForceStep requires state.extras['fa2_mass'].")
        if "fa2_outbound_att_compensation" not in state.extras:
            raise ValueError("FA2ForceStep requires state.extras['fa2_outbound_att_compensation'].")

        undirected_edges = cast(
            torch.Tensor,
            state.extras["fa2_undirected_edges"],
        ).to(device=pos.device, dtype=torch.long)
        raw_weights = state.extras.get("fa2_undirected_weights")
        undirected_weights = (
            None
            if raw_weights is None
            else cast(torch.Tensor, raw_weights).to(device=pos.device, dtype=pos.dtype)
        )
        mass = cast(torch.Tensor, state.extras["fa2_mass"]).to(device=pos.device, dtype=pos.dtype)
        outbound_att_compensation = float(state.extras["fa2_outbound_att_compensation"])
        speed = float(state.extras.get("fa2_speed", 1.0))
        speed_efficiency = float(state.extras.get("fa2_speed_efficiency", 1.0))
        old_force = state.old_forces.to(device=pos.device, dtype=pos.dtype)

        if pos.shape[0] == 0:
            state.forces = torch.zeros_like(pos)
            state.old_forces = state.forces.detach().clone()
            return state

        force = torch.zeros_like(pos)
        if pos.shape[0] > 1:
            if self.config.barnes_hut:
                pos_np = pos.detach().cpu().numpy()
                mass_np = mass.detach().cpu().numpy()
                force_np = np.zeros((pos.shape[0], 2), dtype=np.float64)

                @dataclass(slots=True)
                class BarnesHutNode:
                    """Internal FA2 Barnes-Hut cell used within one iteration."""

                    mass_center_x: float
                    mass_center_y: float
                    mass_value: float
                    size: float
                    children: Optional[list["BarnesHutNode"]]
                    indices: Optional[np.ndarray]

                def build_tree(indices: np.ndarray) -> Optional[BarnesHutNode]:
                    """Build one FA2 Barnes-Hut node for the given particle subset."""
                    if indices.size == 0:
                        return None
                    if indices.size == 1:
                        return BarnesHutNode(
                            mass_center_x=0.0,
                            mass_center_y=0.0,
                            mass_value=0.0,
                            size=0.0,
                            children=None,
                            indices=indices,
                        )

                    cell_mass = float(mass_np[indices].sum())
                    if cell_mass > 0.0:
                        center_x = float((pos_np[indices, 0] * mass_np[indices]).sum() / cell_mass)
                        center_y = float((pos_np[indices, 1] * mass_np[indices]).sum() / cell_mass)
                    else:
                        center_x = float(pos_np[indices, 0].mean())
                        center_y = float(pos_np[indices, 1].mean())

                    x_coord = pos_np[indices, 0]
                    y_coord = pos_np[indices, 1]
                    distance = np.sqrt(((x_coord - center_x) ** 2) + ((y_coord - center_y) ** 2))
                    size = float(2.0 * distance.max())
                    quadrant_masks = (
                        (x_coord < center_x) & (y_coord >= center_y),
                        (x_coord < center_x) & (y_coord < center_y),
                        (x_coord >= center_x) & (y_coord >= center_y),
                        (x_coord >= center_x) & (y_coord < center_y),
                    )

                    children: list[BarnesHutNode] = []
                    for mask in quadrant_masks:
                        child_indices = indices[mask]
                        if child_indices.size == 0:
                            continue
                        if child_indices.size < indices.size:
                            child = build_tree(child_indices)
                            if child is not None:
                                children.append(child)
                            continue
                        for child_index in child_indices:
                            children.append(
                                BarnesHutNode(
                                    mass_center_x=0.0,
                                    mass_center_y=0.0,
                                    mass_value=0.0,
                                    size=0.0,
                                    children=None,
                                    indices=np.asarray([child_index], dtype=np.int64),
                                )
                            )

                    return BarnesHutNode(
                        mass_center_x=center_x,
                        mass_center_y=center_y,
                        mass_value=cell_mass,
                        size=size,
                        children=children,
                        indices=None,
                    )

                def accumulate_from_leaf(node: BarnesHutNode, index: int) -> tuple[float, float]:
                    """Compute the exact force from one FA2 Barnes-Hut leaf."""
                    if node.indices is None:
                        return 0.0, 0.0
                    dx = pos_np[index, 0] - pos_np[node.indices, 0]
                    dy = pos_np[index, 1] - pos_np[node.indices, 1]
                    dist_sq = (dx * dx) + (dy * dy)
                    valid = (node.indices != index) & (dist_sq > 0.0)
                    if not np.any(valid):
                        return 0.0, 0.0
                    factor = (
                        self.config.scaling_ratio
                        * mass_np[index]
                        * mass_np[node.indices[valid]]
                        / dist_sq[valid]
                    )
                    return float(np.sum(factor * dx[valid])), float(np.sum(factor * dy[valid]))

                root = build_tree(np.arange(pos.shape[0], dtype=np.int64))
                if root is not None:
                    for node_index in range(pos.shape[0]):
                        pending = [root]
                        fx = 0.0
                        fy = 0.0
                        while pending:
                            node = pending.pop()
                            if node.children is None:
                                leaf_fx, leaf_fy = accumulate_from_leaf(node, node_index)
                                fx += leaf_fx
                                fy += leaf_fy
                                continue

                            dx = float(pos_np[node_index, 0] - node.mass_center_x)
                            dy = float(pos_np[node_index, 1] - node.mass_center_y)
                            dist_sq = (dx * dx) + (dy * dy)
                            if dist_sq > 0.0 and (node.size * node.size / dist_sq) < (
                                self.config.barnes_hut_theta * self.config.barnes_hut_theta
                            ):
                                dist = math.sqrt(dist_sq)
                                if dist < 1.0e-12:
                                    continue
                                factor = (
                                    self.config.scaling_ratio
                                    * mass_np[node_index]
                                    * node.mass_value
                                    / dist_sq
                                )
                                fx += factor * dx
                                fy += factor * dy
                                continue

                            pending.extend(reversed(node.children))
                        force_np[node_index, 0] = fx
                        force_np[node_index, 1] = fy
                repulsion = torch.from_numpy(force_np).to(device=pos.device, dtype=pos.dtype)
            else:
                delta = pos.unsqueeze(1) - pos.unsqueeze(0)
                distance = torch.cdist(pos, pos, p=2.0)
                distance_sq = distance.square()
                factor = torch.zeros_like(distance_sq)
                valid = distance_sq > 0
                mass_product = mass.unsqueeze(1) * mass.unsqueeze(0)
                factor[valid] = self.config.scaling_ratio * mass_product[valid] / distance_sq[valid]
                repulsion = (delta * factor.unsqueeze(2)).sum(dim=1)
            force = force + repulsion

        if self.config.strong_gravity:
            gravity_factor = torch.zeros_like(mass)
            valid = (pos[:, 0] != 0) & (pos[:, 1] != 0)
            gravity_factor[valid] = self.config.scaling_ratio * mass[valid] * self.config.gravity
            gravity = -pos * gravity_factor.unsqueeze(1)
        else:
            distance = torch.linalg.vector_norm(pos, dim=1)
            gravity_factor = torch.zeros_like(distance)
            valid = distance > 0
            gravity_factor[valid] = mass[valid] * self.config.gravity / distance[valid]
            gravity = -pos * gravity_factor.unsqueeze(1)
        force = force + gravity

        if undirected_edges.numel() > 0:
            source = undirected_edges[0]
            target = undirected_edges[1]
            delta = pos.index_select(0, source) - pos.index_select(0, target)
            if self.config.linlog:
                distance = torch.linalg.vector_norm(delta, dim=1, keepdim=True).clamp(min=1e-6)
                factor = (
                    -float(outbound_att_compensation) * torch.log1p(distance) / distance
                ).squeeze(1)
            else:
                factor = torch.full(
                    (undirected_edges.shape[1],),
                    fill_value=-float(outbound_att_compensation),
                    dtype=pos.dtype,
                    device=pos.device,
                )

            if self.config.outbound_attraction_distribution:
                factor = factor / mass.index_select(0, source)
            if self.config.dissuade_hubs:
                factor = factor / mass.index_select(0, source)
            if undirected_weights is not None:
                transformed_weights = undirected_weights
                if self.config.edge_weight_influence == 0.0:
                    transformed_weights = torch.ones_like(transformed_weights)
                elif self.config.edge_weight_influence != 1.0:
                    transformed_weights = transformed_weights.pow(self.config.edge_weight_influence)
                factor = factor * transformed_weights

            attraction_force = torch.zeros_like(pos)
            attraction = delta * factor.unsqueeze(1)
            attraction_force.scatter_add_(
                0,
                source.unsqueeze(1).expand_as(attraction),
                attraction,
            )
            attraction_force.scatter_add_(
                0,
                target.unsqueeze(1).expand_as(attraction),
                -attraction,
            )
            force = force + attraction_force

        swinging = mass * torch.linalg.vector_norm(old_force - force, dim=1)
        effective_traction = 0.5 * mass * torch.linalg.vector_norm(old_force + force, dim=1)
        total_swinging = float(swinging.sum().item())
        total_effective_traction = float(effective_traction.sum().item())
        estimated_optimal_jt = 0.05 * math.sqrt(float(pos.shape[0]))
        min_jt = math.sqrt(estimated_optimal_jt)
        max_jt = 10.0
        jt = self.config.jitter_tolerance * max(
            min_jt,
            min(
                max_jt,
                estimated_optimal_jt
                * total_effective_traction
                / float(pos.shape[0] * pos.shape[0]),
            ),
        )
        min_speed_efficiency = 0.05
        if total_effective_traction > 0.0 and total_swinging / total_effective_traction > 2.0:
            if speed_efficiency > min_speed_efficiency:
                speed_efficiency *= 0.5
            jt = max(jt, self.config.jitter_tolerance)
        if total_swinging == 0.0:
            target_speed = float("inf")
        else:
            target_speed = jt * speed_efficiency * total_effective_traction / total_swinging
        if total_swinging > jt * total_effective_traction:
            if speed_efficiency > min_speed_efficiency:
                speed_efficiency *= 0.7
        elif speed < 1000.0:
            speed_efficiency *= 1.3
        speed = speed + min(target_speed - speed, 0.5 * speed)
        factor = speed / (1.0 + torch.sqrt(speed * swinging))

        state.forces = force
        state.pos = pos + (force * factor.unsqueeze(1))
        state.old_forces = force.detach().clone()
        state.extras["fa2_speed"] = speed
        state.extras["fa2_speed_efficiency"] = speed_efficiency
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
        raw_edges = state.extras.get("fa2_undirected_edges")
        undirected_edges = (
            cast(torch.Tensor, raw_edges).to(device=pos.device, dtype=torch.long)
            if isinstance(raw_edges, torch.Tensor)
            else None
        )
        raw_weights = state.extras.get("fa2_undirected_weights")
        undirected_weights = (
            cast(torch.Tensor, raw_weights).to(device=pos.device, dtype=pos.dtype)
            if isinstance(raw_weights, torch.Tensor)
            else None
        )
        if undirected_edges is None:
            if problem.edge_index.numel() == 0:
                undirected_edges = torch.empty((2, 0), dtype=torch.long, device=pos.device)
            else:
                source = problem.edge_index[0].to(device=pos.device, dtype=torch.long)
                target = problem.edge_index[1].to(device=pos.device, dtype=torch.long)
                non_self = source != target
                if bool(non_self.any().item()):
                    lower = torch.minimum(source[non_self], target[non_self])
                    upper = torch.maximum(source[non_self], target[non_self])
                    pairs = torch.stack([lower, upper], dim=1)
                    unique_pairs, inverse = torch.unique(pairs, dim=0, return_inverse=True)
                    undirected_edges = unique_pairs.transpose(0, 1).contiguous()
                    if problem.edge_weights is not None:
                        weights = problem.edge_weights[non_self].to(
                            device=pos.device,
                            dtype=pos.dtype,
                        )
                        undirected_weights = torch.zeros(
                            unique_pairs.shape[0],
                            dtype=pos.dtype,
                            device=pos.device,
                        )
                        undirected_weights.scatter_add_(0, inverse, weights)
                else:
                    undirected_edges = torch.empty((2, 0), dtype=torch.long, device=pos.device)
        outbound_att_compensation = (
            float(mass.mean().item()) if self.config.outbound_compensation else 1.0
        )
        contribution = torch.zeros_like(pos)
        if undirected_edges.numel() > 0:
            source = undirected_edges[0]
            target = undirected_edges[1]
            delta = pos.index_select(0, source) - pos.index_select(0, target)
            if self.config.linlog:
                distance = torch.linalg.vector_norm(delta, dim=1, keepdim=True).clamp(min=1.0e-6)
                factor = (
                    -float(outbound_att_compensation) * torch.log1p(distance) / distance
                ).squeeze(1)
            else:
                factor = torch.full(
                    (undirected_edges.shape[1],),
                    fill_value=-float(outbound_att_compensation),
                    dtype=pos.dtype,
                    device=pos.device,
                )
            if self.config.outbound_compensation:
                factor = factor / mass.index_select(0, source)
            if self.config.dissuade_hubs:
                factor = factor / mass.index_select(0, source)
            if undirected_weights is not None:
                factor = factor * undirected_weights
            attraction = delta * factor.unsqueeze(1)
            contribution.scatter_add_(
                0,
                source.unsqueeze(1).expand_as(attraction),
                attraction,
            )
            contribution.scatter_add_(
                0,
                target.unsqueeze(1).expand_as(attraction),
                -attraction,
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
        if self.config.strong_mode:
            factor = torch.zeros_like(mass)
            valid = (pos[:, 0] != 0) & (pos[:, 1] != 0)
            factor[valid] = mass[valid] * float(self.config.strength)
            contribution = -pos * factor.unsqueeze(1)
        else:
            distance = torch.linalg.vector_norm(pos, dim=1)
            factor = torch.zeros_like(distance)
            valid = distance > 0
            factor[valid] = mass[valid] * float(self.config.strength) / distance[valid]
            contribution = -pos * factor.unsqueeze(1)
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
        quadtree = state.quadtree
        if quadtree is None:
            raise ValueError("BarnesHutForce requires state.quadtree.")

        if isinstance(quadtree, _SFDPQuadTreeNode):
            contribution = _sfdp_barnes_hut_force(
                quadtree=quadtree,
                pos=pos,
                theta=self.config.theta,
            )
        elif all(
            hasattr(quadtree, attr)
            for attr in ("mass_center_x", "mass_center_y", "size", "children", "indices")
        ) and (hasattr(quadtree, "mass") or hasattr(quadtree, "mass_value")):
            mass = _fa2_mass(problem=problem, state=state, device=pos.device).to(dtype=pos.dtype)
            pos_np = pos.detach().cpu().numpy()
            mass_np = mass.detach().cpu().numpy()
            force_np = np.zeros((pos.shape[0], 2), dtype=np.float64)

            def accumulate_for_node(node: Any, index: int) -> tuple[float, float]:
                """Recursively accumulate one FA2 Barnes-Hut force contribution."""
                if node is None:
                    return 0.0, 0.0

                indices = getattr(node, "indices", None)
                children = getattr(node, "children", None)
                if children is None:
                    if indices is None:
                        return 0.0, 0.0
                    dx = pos_np[index, 0] - pos_np[indices, 0]
                    dy = pos_np[index, 1] - pos_np[indices, 1]
                    dist_sq = (dx * dx) + (dy * dy)
                    valid = (indices != index) & (dist_sq > 0.0)
                    if not np.any(valid):
                        return 0.0, 0.0
                    factor = mass_np[index] * mass_np[indices[valid]] / dist_sq[valid]
                    return float(np.sum(factor * dx[valid])), float(np.sum(factor * dy[valid]))

                dx = float(pos_np[index, 0] - getattr(node, "mass_center_x"))
                dy = float(pos_np[index, 1] - getattr(node, "mass_center_y"))
                dist_sq = (dx * dx) + (dy * dy)
                if dist_sq > 0.0 and (float(getattr(node, "size")) ** 2 / dist_sq) < (
                    self.config.theta * self.config.theta
                ):
                    dist = math.sqrt(dist_sq)
                    if dist < 1.0e-12:
                        return 0.0, 0.0
                    node_mass = float(getattr(node, "mass", getattr(node, "mass_value", 0.0)))
                    factor = mass_np[index] * node_mass / dist_sq
                    return factor * dx, factor * dy

                fx = 0.0
                fy = 0.0
                for child in children:
                    child_fx, child_fy = accumulate_for_node(child, index)
                    fx += child_fx
                    fy += child_fy
                return fx, fy

            for node_index in range(pos.shape[0]):
                force_np[node_index, 0], force_np[node_index, 1] = accumulate_for_node(
                    quadtree,
                    node_index,
                )
            contribution = torch.from_numpy(force_np).to(device=pos.device, dtype=pos.dtype)
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
        old_force = state.old_forces.to(device=pos.device, dtype=pos.dtype)

        if pos.shape[0] == 0:
            state.pos = pos
            state.old_forces = forces.detach().clone()
            return state

        swinging = mass * torch.linalg.vector_norm(old_force - forces, dim=1)
        effective_traction = 0.5 * mass * torch.linalg.vector_norm(old_force + forces, dim=1)
        total_swinging = float(swinging.sum().item())
        total_effective_traction = float(effective_traction.sum().item())
        estimated_optimal_jt = 0.05 * math.sqrt(float(pos.shape[0]))
        min_jt = math.sqrt(estimated_optimal_jt)
        max_jt = 10.0
        jt = self.config.jitter_tolerance * max(
            min_jt,
            min(
                max_jt,
                estimated_optimal_jt
                * total_effective_traction
                / float(pos.shape[0] * pos.shape[0]),
            ),
        )
        min_speed_efficiency = 0.05
        if total_effective_traction > 0.0 and total_swinging / total_effective_traction > 2.0:
            if speed_efficiency > min_speed_efficiency:
                speed_efficiency *= 0.5
            jt = max(jt, self.config.jitter_tolerance)
        if total_swinging == 0.0:
            target_speed = float("inf")
        else:
            target_speed = jt * speed_efficiency * total_effective_traction / total_swinging
        if total_swinging > jt * total_effective_traction:
            if speed_efficiency > min_speed_efficiency:
                speed_efficiency *= 0.7
        elif speed < 1000.0:
            speed_efficiency *= 1.3

        speed = speed + min(target_speed - speed, 0.5 * speed)
        factor = speed / (1.0 + torch.sqrt(speed * swinging))
        state.pos = pos + (forces * factor.unsqueeze(1))
        state.old_forces = forces.detach().clone()
        state.extras["fa2_speed"] = speed
        state.extras["fa2_speed_efficiency"] = speed_efficiency
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
        local_temperatures = state.local_temperatures
        if local_temperatures is None or local_temperatures.shape != (problem.num_nodes,):
            local_temperatures = torch.full(
                (problem.num_nodes,),
                _GEM_INITIAL_TEMPERATURE,
                device=pos.device,
                dtype=pos.dtype,
            )
        else:
            local_temperatures = local_temperatures.to(device=pos.device, dtype=pos.dtype)
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
        state.local_temperatures = local_temperatures
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
