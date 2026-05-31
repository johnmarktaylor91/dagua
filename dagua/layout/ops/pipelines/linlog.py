"""LinLog energy-model layout pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch

from dagua.layout.ops.anneal import LRDecay
from dagua.layout.ops.base import Conditional, LossGroup, Pipeline, Repeat
from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig
from dagua.layout.ops.init import LinLogInitializePositions
from dagua.layout.ops.loss_classic import LinLogLoss, LinLogLossConfig
from dagua.layout.ops.optimize import (
    LinLogCreateOptimizer,
    OptimizerStep,
    OptimizerZeroGrad,
)
from dagua.layout.ops.postprocess import LinLogFinalizePositions
from dagua.layout.ops.state import (
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)

_MIN_DISTANCE = 1.0e-9
_DEFAULT_STEPS = 300
_DEFAULT_ATTR_EXPONENT = 1.0
_DEFAULT_REPU_EXPONENT = 0.0
_DEFAULT_GRAVITY = 0.0
_DEFAULT_BARNES_HUT_THRESHOLD = 2_000
_DEFAULT_BARNES_HUT_THETA = 0.5
_QUADTREE_LEAF_SIZE = 8
_QUADTREE_MAX_DEPTH = 32


@dataclass(frozen=True)
class _LinLogFidelityConfig:
    """Resolved parameters for the in-pipeline LinLog fidelity solver.

    Parameters
    ----------
    steps : int
        Number of optimization iterations.
    attr_exponent : float
        Attraction exponent in Noack's generalized LinLog model.
    repu_exponent : float
        Repulsion exponent in Noack's generalized LinLog model.
    grav_factor : float
        Optional gravity factor toward the weighted barycenter.
    barnes_hut_threshold : int
        Node count above which Barnes-Hut repulsion is used.
    barnes_hut_theta : float
        Barnes-Hut opening threshold.
    """

    steps: int
    attr_exponent: float
    repu_exponent: float
    grav_factor: float
    barnes_hut_threshold: int
    barnes_hut_theta: float


@dataclass
class _QuadTreeNode:
    """One node in the Barnes-Hut quadtree.

    Parameters
    ----------
    indices : torch.Tensor
        Node indices contained in the region with shape ``[M]``.
    center : torch.Tensor
        Region center with shape ``[2]``.
    half_width : float
        Half-width of the square region.
    mass : float
        Sum of node repulsion weights in the region.
    centroid : torch.Tensor
        Weighted centroid with shape ``[2]``.
    children : list[_QuadTreeNode]
        Child regions in deterministic quadrant order.
    """

    indices: torch.Tensor
    center: torch.Tensor
    half_width: float
    mass: float
    centroid: torch.Tensor
    children: list[_QuadTreeNode]


def _resolve_fidelity_config(variant_params: Optional[Mapping[str, Any]]) -> _LinLogFidelityConfig:
    """Resolve public LinLog parameters for the fidelity solver.

    Parameters
    ----------
    variant_params : Mapping[str, Any] or None
        Optional parameter mapping. Dagua short names and Noack-style names
        are both accepted for parity with benchmark variants.

    Returns
    -------
    _LinLogFidelityConfig
        Validated fidelity configuration.

    Raises
    ------
    ValueError
        If a numeric parameter is outside the supported range.
    """
    params = {} if variant_params is None else dict(variant_params)
    steps = int(params.get("steps", params.get("iterations", _DEFAULT_STEPS)))
    attr_exponent = float(params.get("attrExponent", params.get("a", _DEFAULT_ATTR_EXPONENT)))
    repu_exponent = float(params.get("repuExponent", params.get("r", _DEFAULT_REPU_EXPONENT)))
    grav_factor = float(params.get("gravFactor", _DEFAULT_GRAVITY))
    barnes_hut_threshold = int(params.get("barnesHutThreshold", _DEFAULT_BARNES_HUT_THRESHOLD))
    barnes_hut_theta = float(params.get("barnesHutTheta", _DEFAULT_BARNES_HUT_THETA))

    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if attr_exponent <= repu_exponent:
        raise ValueError("attrExponent must be greater than repuExponent.")
    if grav_factor < 0.0:
        raise ValueError("gravFactor must be non-negative.")
    if barnes_hut_threshold < 1:
        raise ValueError("barnesHutThreshold must be positive.")
    if barnes_hut_theta <= 0.0:
        raise ValueError("barnesHutTheta must be positive.")

    return _LinLogFidelityConfig(
        steps=steps,
        attr_exponent=attr_exponent,
        repu_exponent=repu_exponent,
        grav_factor=grav_factor,
        barnes_hut_threshold=barnes_hut_threshold,
        barnes_hut_theta=barnes_hut_theta,
    )


def _initial_fidelity_positions(num_nodes: int, seed: int) -> torch.Tensor:
    """Create deterministic Noack-style initial coordinates.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    seed : int
        CPU RNG seed.

    Returns
    -------
    torch.Tensor
        Initial positions with shape ``[N, 2]`` and dtype ``float64``.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.rand((num_nodes, 2), generator=generator, dtype=torch.float64)


def _prepare_fidelity_edges(
    edge_index: torch.Tensor,
    edge_weights: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return non-loop edge endpoints and non-negative attraction weights.

    Parameters
    ----------
    edge_index : torch.Tensor
        Input edge tensor with shape ``[2, E]``.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        CPU edge tensor with shape ``[2, E']`` and weights with shape ``[E']``.

    Raises
    ------
    ValueError
        If edge weights are negative or have the wrong length.
    """
    prepared_edge_index = edge_index.to(device="cpu", dtype=torch.long)
    if prepared_edge_index.numel() == 0:
        return prepared_edge_index.reshape(2, 0), torch.empty((0,), dtype=torch.float64)

    edge_count = int(prepared_edge_index.shape[1])
    if edge_weights is None:
        weights = torch.ones((edge_count,), dtype=torch.float64)
    else:
        if edge_weights.ndim != 1 or int(edge_weights.shape[0]) != edge_count:
            raise ValueError("edge_weights must have shape [E].")
        weights = edge_weights.to(device="cpu", dtype=torch.float64)
        if bool((weights < 0.0).any()):
            raise ValueError("edge_weights must be non-negative.")

    non_loop = prepared_edge_index[0] != prepared_edge_index[1]
    return prepared_edge_index[:, non_loop], weights[non_loop]


def _node_fidelity_repulsion_weights(
    num_nodes: int,
    edge_index: torch.Tensor,
    edge_weights: torch.Tensor,
) -> torch.Tensor:
    """Compute edge-repulsion node weights from incident edge weights.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    edge_index : torch.Tensor
        Non-loop edge tensor with shape ``[2, E]``.
    edge_weights : torch.Tensor
        Edge attraction weights with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Node repulsion weights with shape ``[N]``.
    """
    weights = torch.zeros((num_nodes,), dtype=torch.float64)
    if edge_index.numel() == 0:
        weights.fill_(1.0)
        return weights
    weights.scatter_add_(0, edge_index[0], edge_weights)
    weights.scatter_add_(0, edge_index[1], edge_weights)
    return weights


def _fidelity_energy_factors(
    repulsion_weights: torch.Tensor,
    edge_weights: torch.Tensor,
    attr_exponent: float,
    repu_exponent: float,
    grav_factor: float,
) -> tuple[float, float]:
    """Compute Noack repulsion and gravity scaling factors.

    Parameters
    ----------
    repulsion_weights : torch.Tensor
        Node repulsion weights with shape ``[N]``.
    edge_weights : torch.Tensor
        Edge attraction weights with shape ``[E]``.
    attr_exponent : float
        Current attraction exponent.
    repu_exponent : float
        Current repulsion exponent.
    grav_factor : float
        User-facing gravity factor.

    Returns
    -------
    tuple[float, float]
        Repulsion factor and scaled gravity factor.
    """
    repu_sum = float(repulsion_weights.sum().item())
    attr_sum = float(edge_weights.sum().item()) * 2.0
    if repu_sum <= 0.0 or attr_sum <= 0.0:
        return 1.0, grav_factor

    density = attr_sum / (repu_sum * repu_sum)
    exponent_gap = attr_exponent - repu_exponent
    repu_factor = density * math.pow(repu_sum, 0.5 * exponent_gap)
    scaled_gravity = density * repu_sum * math.pow(grav_factor, exponent_gap)
    return repu_factor, scaled_gravity


def _scheduled_fidelity_exponents(
    step: int,
    config: _LinLogFidelityConfig,
) -> tuple[float, float]:
    """Return the attraction and repulsion exponents for one iteration.

    Parameters
    ----------
    step : int
        One-based optimization step.
    config : _LinLogFidelityConfig
        Resolved fidelity configuration.

    Returns
    -------
    tuple[float, float]
        Attraction and repulsion exponents.
    """
    attr_exponent = config.attr_exponent
    repu_exponent = config.repu_exponent
    if config.steps < 50 or config.repu_exponent >= 1.0:
        return attr_exponent, repu_exponent

    warmup_gap = 1.0 - config.repu_exponent
    if step <= 0.6 * config.steps:
        scale = 1.0
    elif step <= 0.9 * config.steps:
        scale = (0.9 - (float(step) / float(config.steps))) / 0.3
    else:
        scale = 0.0
    return attr_exponent + 1.1 * warmup_gap * scale, repu_exponent + 0.9 * warmup_gap * scale


def _weighted_fidelity_barycenter(positions: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Compute the weighted barycenter of current positions.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    weights : torch.Tensor
        Non-negative node weights with shape ``[N]``.

    Returns
    -------
    torch.Tensor
        Barycenter coordinate with shape ``[2]``.
    """
    total_weight = weights.sum().clamp(min=_MIN_DISTANCE)
    return (positions * weights[:, None]).sum(dim=0) / total_weight


def _build_fidelity_quadtree(
    positions: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    center: torch.Tensor,
    half_width: float,
    depth: int = 0,
) -> _QuadTreeNode:
    """Build a Barnes-Hut quadtree for weighted node positions.

    Parameters
    ----------
    positions : torch.Tensor
        Current positions with shape ``[N, 2]``.
    weights : torch.Tensor
        Node repulsion weights with shape ``[N]``.
    indices : torch.Tensor
        Node indices assigned to this region with shape ``[M]``.
    center : torch.Tensor
        Region center with shape ``[2]``.
    half_width : float
        Region half-width.
    depth : int, default=0
        Current recursive depth.

    Returns
    -------
    _QuadTreeNode
        Root of the recursively built subtree.
    """
    node_weights = weights[indices]
    mass = float(node_weights.sum().item())
    if mass > 0.0:
        centroid = (positions[indices] * node_weights[:, None]).sum(dim=0) / mass
    else:
        centroid = positions[indices].mean(dim=0)

    children: list[_QuadTreeNode] = []
    if int(indices.numel()) > _QUADTREE_LEAF_SIZE and depth < _QUADTREE_MAX_DEPTH:
        child_half_width = half_width / 2.0
        child_offsets = (
            (-child_half_width, -child_half_width),
            (child_half_width, -child_half_width),
            (-child_half_width, child_half_width),
            (child_half_width, child_half_width),
        )
        local_positions = positions[indices]
        right = local_positions[:, 0] >= center[0]
        top = local_positions[:, 1] >= center[1]
        child_masks = (~right & ~top, right & ~top, ~right & top, right & top)
        for (offset_x, offset_y), in_child in zip(child_offsets, child_masks):
            child_center = center + torch.tensor((offset_x, offset_y), dtype=positions.dtype)
            child_indices = indices[in_child]
            child_count = int(child_indices.numel())
            if child_count == 0 or child_count == int(indices.numel()):
                continue
            children.append(
                _build_fidelity_quadtree(
                    positions=positions,
                    weights=weights,
                    indices=child_indices,
                    center=child_center,
                    half_width=child_half_width,
                    depth=depth + 1,
                )
            )

    return _QuadTreeNode(
        indices=indices,
        center=center,
        half_width=half_width,
        mass=mass,
        centroid=centroid,
        children=children,
    )


def _fidelity_quadtree_root(positions: torch.Tensor, weights: torch.Tensor) -> _QuadTreeNode:
    """Create a quadtree root covering all current positions.

    Parameters
    ----------
    positions : torch.Tensor
        Current positions with shape ``[N, 2]``.
    weights : torch.Tensor
        Node repulsion weights with shape ``[N]``.

    Returns
    -------
    _QuadTreeNode
        Root node for Barnes-Hut traversal.
    """
    min_pos = positions.min(dim=0).values
    max_pos = positions.max(dim=0).values
    center = (min_pos + max_pos) / 2.0
    half_width = float((max_pos - min_pos).max().item()) / 2.0 + _MIN_DISTANCE
    indices = torch.arange(int(positions.shape[0]), dtype=torch.long)
    return _build_fidelity_quadtree(
        positions=positions,
        weights=weights,
        indices=indices,
        center=center,
        half_width=half_width,
    )


def _add_exact_fidelity_repulsion(
    positions: torch.Tensor,
    weights: torch.Tensor,
    repu_factor: float,
    repu_exponent: float,
    forces: torch.Tensor,
    curvature: torch.Tensor,
) -> None:
    """Accumulate exact all-pairs repulsion forces in-place.

    Parameters
    ----------
    positions : torch.Tensor
        Current positions with shape ``[N, 2]``.
    weights : torch.Tensor
        Node repulsion weights with shape ``[N]``.
    repu_factor : float
        Repulsion energy scaling factor.
    repu_exponent : float
        Current repulsion exponent.
    forces : torch.Tensor
        Force accumulator with shape ``[N, 2]``.
    curvature : torch.Tensor
        Per-node curvature accumulator with shape ``[N]``.

    Returns
    -------
    None
        The force and curvature tensors are updated in-place.
    """
    num_nodes = int(positions.shape[0])
    if num_nodes <= 1:
        return

    src, dst = torch.triu_indices(num_nodes, num_nodes, offset=1)
    delta = positions[src] - positions[dst]
    distances = torch.linalg.norm(delta, dim=1).clamp(min=_MIN_DISTANCE)
    pair_weights = repu_factor * weights[src] * weights[dst]
    active = pair_weights > 0.0
    if not bool(active.any()):
        return

    src = src[active]
    dst = dst[active]
    delta = delta[active]
    distances = distances[active]
    pair_weights = pair_weights[active]
    multipliers = pair_weights * distances.pow(repu_exponent - 2.0)
    pair_forces = delta * multipliers[:, None]
    forces.index_add_(0, src, pair_forces)
    forces.index_add_(0, dst, -pair_forces)
    curve = multipliers * abs(repu_exponent - 1.0)
    curvature.index_add_(0, src, curve)
    curvature.index_add_(0, dst, curve)


def _accumulate_fidelity_barnes_hut_node(
    node_index: int,
    node: _QuadTreeNode,
    positions: torch.Tensor,
    weights: torch.Tensor,
    repu_factor: float,
    repu_exponent: float,
    theta: float,
) -> tuple[torch.Tensor, float]:
    """Accumulate Barnes-Hut repulsion for one query node.

    Parameters
    ----------
    node_index : int
        Query node index.
    node : _QuadTreeNode
        Current quadtree node.
    positions : torch.Tensor
        Current positions with shape ``[N, 2]``.
    weights : torch.Tensor
        Node repulsion weights with shape ``[N]``.
    repu_factor : float
        Repulsion energy scaling factor.
    repu_exponent : float
        Current repulsion exponent.
    theta : float
        Barnes-Hut opening threshold.

    Returns
    -------
    tuple[torch.Tensor, float]
        Force vector with shape ``[2]`` and curvature contribution.
    """
    if node.mass <= 0.0 or (not node.children and int(node.indices.numel()) == 1):
        if int(node.indices.numel()) == 1 and int(node.indices[0].item()) == node_index:
            return torch.zeros((2,), dtype=positions.dtype), 0.0

    delta = positions[node_index] - node.centroid
    distance = float(torch.linalg.norm(delta).clamp(min=_MIN_DISTANCE).item())
    width = 2.0 * node.half_width
    if not node.children or width / distance < theta:
        mass = node.mass
        if int(node.indices.numel()) <= _QUADTREE_LEAF_SIZE and node_index in {
            int(index.item()) for index in node.indices
        }:
            mass -= float(weights[node_index].item())
        if mass <= 0.0:
            return torch.zeros((2,), dtype=positions.dtype), 0.0
        multiplier = repu_factor * float(weights[node_index].item()) * mass
        multiplier *= math.pow(distance, repu_exponent - 2.0)
        return delta * multiplier, multiplier * abs(repu_exponent - 1.0)

    total_force = torch.zeros((2,), dtype=positions.dtype)
    total_curve = 0.0
    for child in node.children:
        child_force, child_curve = _accumulate_fidelity_barnes_hut_node(
            node_index=node_index,
            node=child,
            positions=positions,
            weights=weights,
            repu_factor=repu_factor,
            repu_exponent=repu_exponent,
            theta=theta,
        )
        total_force += child_force
        total_curve += child_curve
    return total_force, total_curve


def _add_barnes_hut_fidelity_repulsion(
    positions: torch.Tensor,
    weights: torch.Tensor,
    repu_factor: float,
    repu_exponent: float,
    theta: float,
    forces: torch.Tensor,
    curvature: torch.Tensor,
) -> None:
    """Accumulate Barnes-Hut approximate repulsion forces in-place.

    Parameters
    ----------
    positions : torch.Tensor
        Current positions with shape ``[N, 2]``.
    weights : torch.Tensor
        Node repulsion weights with shape ``[N]``.
    repu_factor : float
        Repulsion energy scaling factor.
    repu_exponent : float
        Current repulsion exponent.
    theta : float
        Barnes-Hut opening threshold.
    forces : torch.Tensor
        Force accumulator with shape ``[N, 2]``.
    curvature : torch.Tensor
        Per-node curvature accumulator with shape ``[N]``.

    Returns
    -------
    None
        The force and curvature tensors are updated in-place.
    """
    tree = _fidelity_quadtree_root(positions, weights)
    for node_index in range(int(positions.shape[0])):
        node_force, node_curve = _accumulate_fidelity_barnes_hut_node(
            node_index=node_index,
            node=tree,
            positions=positions,
            weights=weights,
            repu_factor=repu_factor,
            repu_exponent=repu_exponent,
            theta=theta,
        )
        forces[node_index] += node_force
        curvature[node_index] += node_curve


def _add_fidelity_attraction(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weights: torch.Tensor,
    attr_exponent: float,
    forces: torch.Tensor,
    curvature: torch.Tensor,
) -> None:
    """Accumulate edge attraction forces in-place.

    Parameters
    ----------
    positions : torch.Tensor
        Current positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Non-loop edge index tensor with shape ``[2, E]``.
    edge_weights : torch.Tensor
        Edge attraction weights with shape ``[E]``.
    attr_exponent : float
        Current attraction exponent.
    forces : torch.Tensor
        Force accumulator with shape ``[N, 2]``.
    curvature : torch.Tensor
        Per-node curvature accumulator with shape ``[N]``.

    Returns
    -------
    None
        The force and curvature tensors are updated in-place.
    """
    if edge_index.numel() == 0:
        return
    src = edge_index[0]
    dst = edge_index[1]
    delta = positions[dst] - positions[src]
    distances = torch.linalg.norm(delta, dim=1).clamp(min=_MIN_DISTANCE)
    multipliers = edge_weights * distances.pow(attr_exponent - 2.0)
    edge_forces = delta * multipliers[:, None]
    forces.index_add_(0, src, edge_forces)
    forces.index_add_(0, dst, -edge_forces)
    curve = multipliers * abs(attr_exponent - 1.0)
    curvature.index_add_(0, src, curve)
    curvature.index_add_(0, dst, curve)


def _add_fidelity_gravity(
    positions: torch.Tensor,
    weights: torch.Tensor,
    barycenter: torch.Tensor,
    scaled_gravity: float,
    attr_exponent: float,
    forces: torch.Tensor,
    curvature: torch.Tensor,
) -> None:
    """Accumulate optional barycenter gravity in-place.

    Parameters
    ----------
    positions : torch.Tensor
        Current positions with shape ``[N, 2]``.
    weights : torch.Tensor
        Node repulsion weights with shape ``[N]``.
    barycenter : torch.Tensor
        Weighted barycenter with shape ``[2]``.
    scaled_gravity : float
        Gravity energy scaling factor.
    attr_exponent : float
        Current attraction exponent.
    forces : torch.Tensor
        Force accumulator with shape ``[N, 2]``.
    curvature : torch.Tensor
        Per-node curvature accumulator with shape ``[N]``.

    Returns
    -------
    None
        The force and curvature tensors are updated in-place.
    """
    if scaled_gravity == 0.0:
        return
    delta = barycenter[None, :] - positions
    distances = torch.linalg.norm(delta, dim=1).clamp(min=_MIN_DISTANCE)
    multipliers = scaled_gravity * weights * distances.pow(attr_exponent - 2.0)
    forces += delta * multipliers[:, None]
    curvature += multipliers * abs(attr_exponent - 1.0)


def _average_fidelity_distances(positions: torch.Tensor) -> torch.Tensor:
    """Compute average distance from each node to all other nodes.

    Parameters
    ----------
    positions : torch.Tensor
        Current positions with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Average distances with shape ``[N]``.
    """
    num_nodes = int(positions.shape[0])
    if num_nodes <= 1:
        return torch.zeros((num_nodes,), dtype=positions.dtype)
    distances = torch.cdist(positions, positions).sum(dim=1)
    return distances / float(num_nodes - 1)


def _normalize_fidelity_positions(positions: torch.Tensor) -> torch.Tensor:
    """Center and scale final coordinates for stable comparisons.

    Parameters
    ----------
    positions : torch.Tensor
        Final positions with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Centered positions with max absolute extent near one.
    """
    if int(positions.shape[0]) <= 1:
        return torch.zeros_like(positions, dtype=torch.float32)
    centered = positions - positions.mean(dim=0, keepdim=True)
    scale = centered.abs().max().clamp(min=1.0)
    return (centered / scale).to(dtype=torch.float32)


def _layout_linlog_fidelity(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor],
    config: _LinLogFidelityConfig,
    seed: int,
) -> torch.Tensor:
    """Run Dagua's in-pipeline Noack LinLog fidelity solver.

    Parameters
    ----------
    edge_index : torch.Tensor
        Input edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.
    config : _LinLogFidelityConfig
        Resolved LinLog fidelity configuration.
    seed : int
        Random seed for initial coordinates.

    Returns
    -------
    torch.Tensor
        Final positions with shape ``[N, 2]`` and dtype ``float32``.
    """
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32)

    prepared_edge_index, prepared_edge_weights = _prepare_fidelity_edges(edge_index, edge_weights)
    repulsion_weights = _node_fidelity_repulsion_weights(
        num_nodes,
        prepared_edge_index,
        prepared_edge_weights,
    )
    positions = _initial_fidelity_positions(num_nodes, seed)

    for step in range(1, config.steps + 1):
        attr_exponent, repu_exponent = _scheduled_fidelity_exponents(step, config)
        repu_factor, scaled_gravity = _fidelity_energy_factors(
            repulsion_weights=repulsion_weights,
            edge_weights=prepared_edge_weights,
            attr_exponent=attr_exponent,
            repu_exponent=repu_exponent,
            grav_factor=config.grav_factor,
        )
        forces = torch.zeros_like(positions)
        curvature = torch.zeros((num_nodes,), dtype=positions.dtype)
        barycenter = _weighted_fidelity_barycenter(positions, repulsion_weights)

        # The fidelity path intentionally keeps scalar tree traversal and
        # accumulator ordering local to this pipeline. Registered op splitting
        # would make the arithmetic order easier to change accidentally, while
        # this path exists specifically to match the paper-spec solver step by
        # step without delegating to benchmark reference code.
        if num_nodes > config.barnes_hut_threshold:
            _add_barnes_hut_fidelity_repulsion(
                positions=positions,
                weights=repulsion_weights,
                repu_factor=repu_factor,
                repu_exponent=repu_exponent,
                theta=config.barnes_hut_theta,
                forces=forces,
                curvature=curvature,
            )
        else:
            _add_exact_fidelity_repulsion(
                positions=positions,
                weights=repulsion_weights,
                repu_factor=repu_factor,
                repu_exponent=repu_exponent,
                forces=forces,
                curvature=curvature,
            )

        _add_fidelity_attraction(
            positions=positions,
            edge_index=prepared_edge_index,
            edge_weights=prepared_edge_weights,
            attr_exponent=attr_exponent,
            forces=forces,
            curvature=curvature,
        )
        _add_fidelity_gravity(
            positions=positions,
            weights=repulsion_weights,
            barycenter=barycenter,
            scaled_gravity=scaled_gravity,
            attr_exponent=attr_exponent,
            forces=forces,
            curvature=curvature,
        )

        steps_tensor = forces / curvature.clamp(min=_MIN_DISTANCE)[:, None]
        max_step = _average_fidelity_distances(positions).clamp(min=0.01)
        lengths = torch.linalg.norm(steps_tensor, dim=1).clamp(min=_MIN_DISTANCE)
        steps_tensor *= torch.minimum(torch.ones_like(lengths), max_step / lengths)[:, None]
        cooling = 1.0 - 0.95 * (float(step - 1) / float(max(config.steps, 1)))
        positions += steps_tensor * cooling

    return _normalize_fidelity_positions(positions)


def build_linlog_pipeline(
    steps: int = 300,
    a: float = 1.0,
    r: float = 0.0,
) -> Pipeline:
    """Build a LinLog energy-model pipeline.

    Parameters
    ----------
    steps : int, default=300
        Number of Adam updates.
    a : float, default=1.0
        Attraction exponent in the LinLog objective.
    r : float, default=0.0
        Repulsion exponent in the LinLog objective.

    Returns
    -------
    Pipeline
        Pipeline implementing the LinLog algorithm. The pipeline produces
        final node coordinates by initializing positions, creating an Adam
        optimizer, iteratively evaluating the LinLog loss with the requested
        exponents, stepping the optimizer with learning-rate decay, and
        finalizing the layout.

    Raises
    ------
    ValueError
        If ``steps``, ``a``, or ``r`` are invalid.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if a < 0.0:
        raise ValueError("a must be non-negative.")
    if r < 0.0:
        raise ValueError("r must be non-negative.")

    objective = LinLogLoss(
        config=LinLogLossConfig(exponent_a=a, exponent_r=r),
    )

    optimize_pipeline = Pipeline(
        [
            LinLogCreateOptimizer(),
            Repeat(
                n=steps,
                ops=[
                    OptimizerZeroGrad(),
                    LossGroup([objective]),
                    OptimizerStep(),
                    LRDecay(),
                ],
            ),
        ],
        name="linlog_optimize",
    )

    return Pipeline(
        [
            FixedSteps(FixedStepsConfig(n=steps)),
            LinLogInitializePositions(),
            Conditional(
                predicate=lambda problem, state, ctx: problem.num_nodes > 1,
                op=optimize_pipeline,
            ),
            LinLogFinalizePositions(),
        ],
        name="linlog_pipeline",
    )


def layout_linlog_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 300,
    seed: int = 42,
    a: float = 1.0,
    r: float = 0.0,
    edge_weights: Optional[torch.Tensor] = None,
    fidelity_mode: bool = True,
    fidelity_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Run the LinLog pipeline as a drop-in replacement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]`` used only to scale the
        final layout.
    steps : int, default=300
        Number of Adam updates.
    seed : int, default=42
        Random seed for initialization and repulsion sampling.
    a : float, default=1.0
        Attraction exponent.
    r : float, default=0.0
        Repulsion exponent.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.
    fidelity_mode : bool, default=True
        Use Dagua's in-pipeline Noack displacement kernel. Set to ``False`` to
        keep the historical composable Adam-energy pipeline behavior.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If the public LinLog inputs are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if a < 0.0:
        raise ValueError("a must be non-negative.")
    if r < 0.0:
        raise ValueError("r must be non-negative.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
            )

    if fidelity_mode:
        config = _resolve_fidelity_config({"steps": steps, "a": a, "r": r})
        output_device = edge_index.device
        if edge_index.numel() == 0 and node_sizes is not None:
            output_device = node_sizes.device
        return _layout_linlog_fidelity(
            edge_index=edge_index,
            num_nodes=num_nodes,
            edge_weights=edge_weights,
            config=config,
            seed=seed,
        ).to(device=output_device)

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_linlog_pipeline(steps=steps, a=a, r=r).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("LinLog pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_linlog_pipeline", "layout_linlog_pipeline"]
