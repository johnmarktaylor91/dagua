"""ForceAtlas2 expressed as a composable ops pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple

import numpy as np
import torch

from dagua.layout.ops.base import Op, Pipeline, Repeat  # noqa: E402
from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig  # noqa: E402
from dagua.layout.ops.state import (  # noqa: E402
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)
from dagua.layout.ops.taxonomy import OpCategory  # noqa: E402


@dataclass(slots=True)
class _BarnesHutNode:
    """Barnes-Hut region matching ``fa2_modified.fa2util.Region``.

    Parameters
    ----------
    mass_center_x : float
        X coordinate of the region mass center.
    mass_center_y : float
        Y coordinate of the region mass center.
    mass : float
        Total mass contained in the node.
    size : float
        Region diameter, defined as twice the farthest node distance from the
        mass center.
    children : list[_BarnesHutNode] or None
        Child regions in the reference traversal order.
    indices : np.ndarray or None
        Particle indices stored in a leaf region.
    """

    mass_center_x: float
    mass_center_y: float
    mass: float
    size: float
    children: Optional[list["_BarnesHutNode"]]
    indices: Optional[np.ndarray]


def _validate_inputs(
    edge_index: torch.Tensor,
    num_nodes: int,
    steps: int,
    trace_every: int,
    edge_weights: Optional[torch.Tensor],
    barnes_hut_theta: float,
) -> None:
    """Validate public layout arguments.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list tensor with shape ``[2, E]``.
    num_nodes : int
        Declared node count.
    steps : int
        Number of optimization steps.
    trace_every : int
        Snapshot cadence.
    edge_weights : torch.Tensor, optional
        Optional per-edge weights with shape ``[E]``.
    barnes_hut_theta : float
        Barnes-Hut acceptance threshold.

    Returns
    -------
    None
        Raises ``ValueError`` for invalid inputs.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if trace_every < 0:
        raise ValueError("trace_every must be non-negative")
    if barnes_hut_theta <= 0.0:
        raise ValueError("barnes_hut_theta must be positive")
    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if edge_index.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError("edge_index must use an integer dtype")
    if edge_weights is not None:
        if edge_weights.dim() != 1:
            raise ValueError("edge_weights must be a one-dimensional tensor")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError("edge_weights length must match edge_index column count")
    if edge_index.numel() == 0:
        return

    min_index = int(edge_index.min().item())
    max_index = int(edge_index.max().item())
    if min_index < 0:
        raise ValueError("edge_index cannot contain negative node indices")
    if max_index >= num_nodes:
        raise ValueError("edge_index contains node indices outside num_nodes")


def _unique_undirected_edges(edge_index: torch.Tensor) -> torch.Tensor:
    """Collapse a directed edge list into unique undirected edges.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.

    Returns
    -------
    torch.Tensor
        Unique undirected edge list with shape ``[2, E_unique]``.
    """
    undirected_edges, _ = _unique_undirected_edges_with_weights(
        edge_index=edge_index,
        edge_weights=None,
    )
    return undirected_edges


def _unique_undirected_edges_with_weights(
    edge_index: torch.Tensor,
    edge_weights: Optional[torch.Tensor],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Collapse directed edges into unique undirected pairs, summing weights.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    edge_weights : torch.Tensor, optional
        Per-edge weights with shape ``[E]``.

    Returns
    -------
    tuple[torch.Tensor, Optional[torch.Tensor]]
        Unique undirected edge list with shape ``[2, E_unique]``` and the
        summed undirected weights with shape ``[E_unique]`` when provided.
    """
    if edge_index.numel() == 0:
        empty = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
        return empty, None

    source = edge_index[0].to(dtype=torch.long)
    target = edge_index[1].to(dtype=torch.long)
    non_self = source != target
    if not bool(non_self.any().item()):
        empty = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
        return empty, None

    source = source[non_self]
    target = target[non_self]
    lower = torch.minimum(source, target)
    upper = torch.maximum(source, target)
    pairs = torch.stack([lower, upper], dim=1)
    unique_pairs, inverse = torch.unique(pairs, dim=0, return_inverse=True)
    undirected = unique_pairs.transpose(0, 1).contiguous()

    if edge_weights is not None:
        weights = edge_weights[non_self].to(dtype=torch.float32)
        summed_weights = torch.zeros(
            unique_pairs.shape[0],
            dtype=torch.float32,
            device=edge_index.device,
        )
        summed_weights.scatter_add_(0, inverse, weights)
        return undirected, summed_weights
    return undirected, None


def _compute_degree(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Compute deduplicated undirected degree counts.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    torch.Tensor
        Degree tensor with shape ``[N]``.
    """
    degree = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device)
    if edge_index.numel() == 0:
        return degree

    undirected_edges = _unique_undirected_edges(edge_index)
    if undirected_edges.numel() == 0:
        return degree

    ones = torch.ones(undirected_edges.shape[1], dtype=torch.float32, device=edge_index.device)
    degree.scatter_add_(0, undirected_edges[0], ones)
    degree.scatter_add_(0, undirected_edges[1], ones)
    return degree


def _repulsion_force(pos: torch.Tensor, mass: torch.Tensor, scaling_ratio: float) -> torch.Tensor:
    """Compute exact all-pairs ForceAtlas2 repulsion.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    mass : torch.Tensor
        Node masses with shape ``[N]``.
    scaling_ratio : float
        Repulsion coefficient.

    Returns
    -------
    torch.Tensor
        Repulsive displacements with shape ``[N, 2]``.
    """
    if pos.shape[0] <= 1:
        return torch.zeros_like(pos)

    delta = pos.unsqueeze(1) - pos.unsqueeze(0)
    distance = torch.cdist(pos, pos, p=2.0)
    distance_sq = distance.square()
    factor = torch.zeros_like(distance_sq)
    valid = distance_sq > 0
    mass_product = mass.unsqueeze(1) * mass.unsqueeze(0)
    factor[valid] = scaling_ratio * mass_product[valid] / distance_sq[valid]
    return (delta * factor.unsqueeze(2)).sum(dim=1)


def _gravity_force(
    pos: torch.Tensor,
    mass: torch.Tensor,
    gravity: float,
    strong_gravity: bool,
    scaling_ratio: float,
) -> torch.Tensor:
    """Compute ForceAtlas2 gravity toward the origin.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    mass : torch.Tensor
        Node masses with shape ``[N]``.
    gravity : float
        Gravity coefficient.
    strong_gravity : bool
        Whether to use strong-gravity mode.
    scaling_ratio : float
        Reference coefficient used only in strong-gravity mode.

    Returns
    -------
    torch.Tensor
        Gravity displacement with shape ``[N, 2]``.
    """
    if strong_gravity:
        factor = torch.zeros_like(mass)
        # The reference strong-gravity helper skips nodes that lie exactly on
        # either axis because it checks ``xDist != 0 and yDist != 0``.
        valid = (pos[:, 0] != 0) & (pos[:, 1] != 0)
        factor[valid] = scaling_ratio * mass[valid] * gravity
        return -pos * factor.unsqueeze(1)

    distance = torch.linalg.vector_norm(pos, dim=1)
    factor = torch.zeros_like(distance)
    valid = distance > 0
    factor[valid] = mass[valid] * gravity / distance[valid]
    return -pos * factor.unsqueeze(1)


def _attraction_force(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    mass: torch.Tensor,
    outbound_att_compensation: float,
    outbound_attraction_distribution: bool,
    linlog: bool = False,
    edge_weights: Optional[torch.Tensor] = None,
    dissuade_hubs: bool = False,
    edge_weight_influence: float = 1.0,
) -> torch.Tensor:
    """Compute ForceAtlas2 attraction over unique undirected edges.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Unique undirected edge list with shape ``[2, E]``.
    mass : torch.Tensor
        Node masses with shape ``[N]``.
    outbound_att_compensation : float
        Mean-mass compensation used when outbound attraction distribution is on.
    outbound_attraction_distribution : bool
        Whether to divide attraction by the source-node mass.
    linlog : bool, default=False
        Whether to use ``log(1 + distance)`` attraction.
    edge_weights : torch.Tensor, optional
        Per-edge weights with shape ``[E]``.
    dissuade_hubs : bool, default=False
        Whether to divide attraction by source-node mass a second time.
    edge_weight_influence : float, default=1.0
        Exponent applied to edge weights before attraction.

    Returns
    -------
    torch.Tensor
        Attraction displacement with shape ``[N, 2]``.
    """
    force = torch.zeros_like(pos)
    if edge_index.numel() == 0:
        return force

    source = edge_index[0]
    target = edge_index[1]
    delta = pos.index_select(0, source) - pos.index_select(0, target)
    if linlog:
        distance = torch.linalg.vector_norm(delta, dim=1, keepdim=True).clamp(min=1e-6)
        factor = -(float(outbound_att_compensation) * torch.log1p(distance) / distance).squeeze(1)
    else:
        factor = torch.full(
            (edge_index.shape[1],),
            fill_value=-float(outbound_att_compensation),
            dtype=pos.dtype,
            device=pos.device,
        )

    if outbound_attraction_distribution:
        factor = factor / mass.index_select(0, source)
    if dissuade_hubs:
        factor = factor / mass.index_select(0, source)
    if edge_weights is not None:
        transformed_weights = edge_weights.to(dtype=pos.dtype, device=pos.device)
        if edge_weight_influence == 0.0:
            transformed_weights = torch.ones_like(transformed_weights)
        elif edge_weight_influence != 1.0:
            transformed_weights = transformed_weights.pow(edge_weight_influence)
        factor = factor * transformed_weights

    attraction = delta * factor.unsqueeze(1)
    index = source.unsqueeze(1).expand_as(attraction)
    force.scatter_add_(0, index, attraction)
    index = target.unsqueeze(1).expand_as(attraction)
    force.scatter_add_(0, index, -attraction)
    return force


def _build_barnes_hut_tree(
    pos_np: np.ndarray,
    mass_np: np.ndarray,
    indices: np.ndarray,
) -> Optional[_BarnesHutNode]:
    """Build a Barnes-Hut region tree matching ``fa2_modified``.

    Parameters
    ----------
    pos_np : np.ndarray
        Node positions with shape ``[N, 2]``.
    mass_np : np.ndarray
        Node masses with shape ``[N]``.
    indices : np.ndarray
        Particle indices stored in the current region.

    Returns
    -------
    _BarnesHutNode or None
        Region for the current subset, or ``None`` when the subset is empty.
    """
    if indices.size == 0:
        return None

    if indices.size == 1:
        return _BarnesHutNode(
            mass_center_x=0.0,
            mass_center_y=0.0,
            mass=0.0,
            size=0.0,
            children=None,
            indices=indices,
        )

    cell_mass = float(mass_np[indices].sum())
    if cell_mass > 0.0:
        mass_center_x = float((pos_np[indices, 0] * mass_np[indices]).sum() / cell_mass)
        mass_center_y = float((pos_np[indices, 1] * mass_np[indices]).sum() / cell_mass)
    else:
        mass_center_x = float(pos_np[indices, 0].mean())
        mass_center_y = float(pos_np[indices, 1].mean())

    x_coord = pos_np[indices, 0]
    y_coord = pos_np[indices, 1]
    distance = np.sqrt(((x_coord - mass_center_x) ** 2) + ((y_coord - mass_center_y) ** 2))
    size = float(2.0 * distance.max())

    quadrant_masks = (
        (x_coord < mass_center_x) & (y_coord >= mass_center_y),
        (x_coord < mass_center_x) & (y_coord < mass_center_y),
        (x_coord >= mass_center_x) & (y_coord >= mass_center_y),
        (x_coord >= mass_center_x) & (y_coord < mass_center_y),
    )

    children: list[_BarnesHutNode] = []
    for mask in quadrant_masks:
        child_indices = indices[mask]
        if child_indices.size == 0:
            continue
        if child_indices.size < indices.size:
            child = _build_barnes_hut_tree(
                pos_np=pos_np,
                mass_np=mass_np,
                indices=child_indices,
            )
            if child is not None:
                children.append(child)
            continue

        # When all nodes fall into one quadrant, the reference breaks the
        # region into singleton leaves instead of recursing forever.
        for child_index in child_indices:
            children.append(
                _BarnesHutNode(
                    mass_center_x=0.0,
                    mass_center_y=0.0,
                    mass=0.0,
                    size=0.0,
                    children=None,
                    indices=np.asarray([child_index], dtype=np.int64),
                )
            )

    return _BarnesHutNode(
        mass_center_x=mass_center_x,
        mass_center_y=mass_center_y,
        mass=cell_mass,
        size=size,
        children=children,
        indices=None,
    )


def _barnes_hut_force_for_leaf(
    pos_np: np.ndarray,
    mass_np: np.ndarray,
    leaf: _BarnesHutNode,
    index: int,
    scaling_ratio: float,
) -> tuple[float, float]:
    """Compute exact repulsion between one node and a leaf cell.

    Parameters
    ----------
    pos_np : np.ndarray
        Node positions with shape ``[N, 2]``.
    mass_np : np.ndarray
        Node masses with shape ``[N]``.
    leaf : _BarnesHutNode
        Leaf quadtree node.
    index : int
        Index of the node receiving force.
    scaling_ratio : float
        Repulsion coefficient.

    Returns
    -------
    tuple[float, float]
        X and y force contributions from the leaf.
    """
    assert leaf.indices is not None

    dx = pos_np[index, 0] - pos_np[leaf.indices, 0]
    dy = pos_np[index, 1] - pos_np[leaf.indices, 1]
    dist_sq = (dx * dx) + (dy * dy)
    valid = (leaf.indices != index) & (dist_sq > 0.0)
    if not np.any(valid):
        return 0.0, 0.0

    factor = scaling_ratio * mass_np[index] * mass_np[leaf.indices[valid]] / dist_sq[valid]
    return float(np.sum(factor * dx[valid])), float(np.sum(factor * dy[valid]))


def _barnes_hut_force_for_node(
    node: Optional[_BarnesHutNode],
    pos_np: np.ndarray,
    mass_np: np.ndarray,
    index: int,
    scaling_ratio: float,
    theta: float,
) -> tuple[float, float]:
    """Recursively accumulate Barnes-Hut repulsion for one node.

    Parameters
    ----------
    node : _BarnesHutNode or None
        Current quadtree node.
    pos_np : np.ndarray
        Node positions with shape ``[N, 2]``.
    mass_np : np.ndarray
        Node masses with shape ``[N]``.
    index : int
        Index of the node receiving force.
    scaling_ratio : float
        Repulsion coefficient.
    theta : float
        Barnes-Hut acceptance threshold.

    Returns
    -------
    tuple[float, float]
        X and y force contributions from the subtree.
    """
    if node is None:
        return 0.0, 0.0

    if node.children is None:
        return _barnes_hut_force_for_leaf(
            pos_np=pos_np,
            mass_np=mass_np,
            leaf=node,
            index=index,
            scaling_ratio=scaling_ratio,
        )

    dx = float(pos_np[index, 0] - node.mass_center_x)
    dy = float(pos_np[index, 1] - node.mass_center_y)
    dist_sq = (dx * dx) + (dy * dy)

    if dist_sq > 0.0 and (node.size * node.size / dist_sq) < (theta * theta):
        dist = math.sqrt(dist_sq)
        if dist < 1e-12:
            return 0.0, 0.0
        factor = scaling_ratio * mass_np[index] * node.mass / dist_sq
        return factor * dx, factor * dy

    fx = 0.0
    fy = 0.0
    for child in node.children:
        child_fx, child_fy = _barnes_hut_force_for_node(
            node=child,
            pos_np=pos_np,
            mass_np=mass_np,
            index=index,
            scaling_ratio=scaling_ratio,
            theta=theta,
        )
        fx += child_fx
        fy += child_fy
    return fx, fy


def _barnes_hut_repulsion(
    pos: torch.Tensor,
    mass: torch.Tensor,
    scaling_ratio: float,
    theta: float,
) -> torch.Tensor:
    """Approximate ForceAtlas2 repulsion using a Barnes-Hut quadtree.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    mass : torch.Tensor
        Node masses with shape ``[N]``.
    scaling_ratio : float
        Repulsion coefficient.
    theta : float
        Barnes-Hut acceptance threshold.

    Returns
    -------
    torch.Tensor
        Approximate repulsive displacements with shape ``[N, 2]``.
    """
    num_nodes = pos.shape[0]
    if num_nodes <= 1:
        return torch.zeros_like(pos)

    pos_np = pos.detach().cpu().numpy()
    mass_np = mass.detach().cpu().numpy()
    force_np = np.zeros((num_nodes, 2), dtype=np.float64)

    root = _build_barnes_hut_tree(
        pos_np=pos_np,
        mass_np=mass_np,
        indices=np.arange(num_nodes, dtype=np.int64),
    )

    for node_index in range(num_nodes):
        fx, fy = _barnes_hut_force_for_node(
            node=root,
            pos_np=pos_np,
            mass_np=mass_np,
            index=node_index,
            scaling_ratio=scaling_ratio,
            theta=theta,
        )
        force_np[node_index, 0] = fx
        force_np[node_index, 1] = fy

    return torch.from_numpy(force_np).to(dtype=pos.dtype, device=pos.device)


def _adjust_speed_and_apply_forces(
    pos: torch.Tensor,
    force: torch.Tensor,
    old_force: torch.Tensor,
    mass: torch.Tensor,
    speed: float,
    speed_efficiency: float,
    jitter_tolerance: float,
) -> tuple[torch.Tensor, float, float]:
    """Translate ``adjustSpeedAndApplyForces`` into vectorized PyTorch.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    force : torch.Tensor
        Current node displacements with shape ``[N, 2]``.
    old_force : torch.Tensor
        Previous node displacements with shape ``[N, 2]``.
    mass : torch.Tensor
        Node masses with shape ``[N]``.
    speed : float
        Current global speed.
    speed_efficiency : float
        Current speed-efficiency coefficient.
    jitter_tolerance : float
        Reference jitter-tolerance hyperparameter.

    Returns
    -------
    tuple[torch.Tensor, float, float]
        Updated ``(positions, speed, speed_efficiency)``.
    """
    swinging = mass * torch.linalg.vector_norm(old_force - force, dim=1)
    effective_traction = 0.5 * mass * torch.linalg.vector_norm(old_force + force, dim=1)

    total_swinging = float(swinging.sum().item())
    total_effective_traction = float(effective_traction.sum().item())

    estimated_optimal_jt = 0.05 * math.sqrt(float(pos.shape[0]))
    min_jt = math.sqrt(estimated_optimal_jt)
    max_jt = 10.0
    jt = jitter_tolerance * max(
        min_jt,
        min(
            max_jt,
            estimated_optimal_jt * total_effective_traction / float(pos.shape[0] * pos.shape[0]),
        ),
    )

    min_speed_efficiency = 0.05
    if total_effective_traction > 0.0 and total_swinging / total_effective_traction > 2.0:
        if speed_efficiency > min_speed_efficiency:
            speed_efficiency *= 0.5
        jt = max(jt, jitter_tolerance)

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
    return pos + (force * factor.unsqueeze(1)), speed, speed_efficiency


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
