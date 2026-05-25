"""Reference LinLog competitor adapter.

This module is a clean Python implementation of Andreas Noack's generalized
LinLog energy model for fidelity comparison. It intentionally does not call
Dagua's own LinLog pipeline, so the benchmark harness has an independent
reference side for ``classic_linlog_*`` variants.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Optional

import torch

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, register

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph

_MIN_DISTANCE = 1.0e-9
_DEFAULT_STEPS = 300
_DEFAULT_SEED = 42
_DEFAULT_ATTR_EXPONENT = 1.0
_DEFAULT_REPU_EXPONENT = 0.0
_DEFAULT_GRAVITY = 0.0
_DEFAULT_BARNES_HUT_THRESHOLD = 2_000
_DEFAULT_BARNES_HUT_THETA = 0.5
_QUADTREE_LEAF_SIZE = 8
_QUADTREE_MAX_DEPTH = 32


@dataclass(frozen=True)
class _LinLogConfig:
    """Resolved LinLog parameters.

    Parameters
    ----------
    steps : int
        Number of optimization iterations.
    attr_exponent : float
        Exponent of the attraction energy. ``1.0`` is LinLog attraction.
    repu_exponent : float
        Exponent of the repulsion energy. ``0.0`` means logarithmic repulsion.
    grav_factor : float
        Optional gravity toward the weighted barycenter.
    barnes_hut_threshold : int
        Node count above which repulsion uses the Barnes-Hut approximation.
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
    """One node in a two-dimensional Barnes-Hut quadtree.

    Parameters
    ----------
    indices : torch.Tensor
        Node indices contained in this region with shape ``[M]``.
    center : torch.Tensor
        Region center with shape ``[2]``.
    half_width : float
        Half of the square region width.
    mass : float
        Sum of repulsion weights in this region.
    centroid : torch.Tensor
        Weighted coordinate centroid with shape ``[2]``.
    children : list[_QuadTreeNode]
        Non-empty child regions.
    """

    indices: torch.Tensor
    center: torch.Tensor
    half_width: float
    mass: float
    centroid: torch.Tensor
    children: list[_QuadTreeNode]


def _resolve_config(variant_params: Optional[Mapping[str, Any]]) -> _LinLogConfig:
    """Resolve variant parameters into a typed LinLog configuration.

    Parameters
    ----------
    variant_params : Mapping[str, Any] | None
        Optional benchmark variant parameters. Both Noack-style names
        (``attrExponent``, ``repuExponent``) and Dagua's short names
        (``a``, ``r``) are accepted.

    Returns
    -------
    _LinLogConfig
        Validated LinLog configuration.

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

    return _LinLogConfig(
        steps=steps,
        attr_exponent=attr_exponent,
        repu_exponent=repu_exponent,
        grav_factor=grav_factor,
        barnes_hut_threshold=barnes_hut_threshold,
        barnes_hut_theta=barnes_hut_theta,
    )


def _initial_positions(num_nodes: int, seed: int) -> torch.Tensor:
    """Create Noack-style random initial positions in ``[0, 1)^2``.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    seed : int
        Random seed for deterministic benchmark runs.

    Returns
    -------
    torch.Tensor
        Initial positions with shape ``[N, 2]`` and dtype ``float64``.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.rand((num_nodes, 2), generator=generator, dtype=torch.float64)


def _prepare_edges(graph: DaguaGraph) -> tuple[torch.Tensor, torch.Tensor]:
    """Return non-loop edge endpoints and non-negative attraction weights.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to lay out.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Edge index tensor with shape ``[2, E]`` and edge weights with shape
        ``[E]``, both on CPU.

    Raises
    ------
    ValueError
        If provided edge weights are negative or have the wrong length.
    """
    edge_index = graph.edge_index.to(device="cpu", dtype=torch.long)
    if edge_index.numel() == 0:
        return edge_index.reshape(2, 0), torch.empty((0,), dtype=torch.float64)

    edge_count = int(edge_index.shape[1])
    if graph.edge_weights is None:
        weights = torch.ones((edge_count,), dtype=torch.float64)
    else:
        if graph.edge_weights.ndim != 1 or int(graph.edge_weights.shape[0]) != edge_count:
            raise ValueError("edge_weights must have shape [E].")
        weights = graph.edge_weights.to(device="cpu", dtype=torch.float64)
        if bool((weights < 0.0).any()):
            raise ValueError("edge_weights must be non-negative.")

    non_loop = edge_index[0] != edge_index[1]
    return edge_index[:, non_loop], weights[non_loop]


def _node_repulsion_weights(
    num_nodes: int,
    edge_index: torch.Tensor,
    edge_weights: torch.Tensor,
) -> torch.Tensor:
    """Compute Noack edge-repulsion node weights from incident edge weights.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    edge_index : torch.Tensor
        Non-loop edge index tensor with shape ``[2, E]``.
    edge_weights : torch.Tensor
        Edge attraction weights with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Repulsion weights with shape ``[N]``.
    """
    weights = torch.zeros((num_nodes,), dtype=torch.float64)
    if edge_index.numel() == 0:
        weights.fill_(1.0)
        return weights
    weights.scatter_add_(0, edge_index[0], edge_weights)
    weights.scatter_add_(0, edge_index[1], edge_weights)
    return weights


def _energy_factors(
    repulsion_weights: torch.Tensor,
    edge_weights: torch.Tensor,
    attr_exponent: float,
    repu_exponent: float,
    grav_factor: float,
) -> tuple[float, float]:
    """Compute Noack's repulsion and gravity scaling factors.

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


def _scheduled_exponents(step: int, config: _LinLogConfig) -> tuple[float, float]:
    """Return the exponents used for one iteration.

    Parameters
    ----------
    step : int
        One-based optimization iteration.
    config : _LinLogConfig
        Final user-requested LinLog parameters.

    Returns
    -------
    tuple[float, float]
        Attraction and repulsion exponents for this iteration.
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


def _weighted_barycenter(positions: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Compute the weighted barycenter of the current positions.

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


def _build_quadtree(
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
        Current tree depth.

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
                _build_quadtree(
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


def _quadtree_root(positions: torch.Tensor, weights: torch.Tensor) -> _QuadTreeNode:
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
    return _build_quadtree(
        positions=positions,
        weights=weights,
        indices=indices,
        center=center,
        half_width=half_width,
    )


def _add_exact_repulsion(
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
        Updates ``forces`` and ``curvature`` in-place.
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


def _accumulate_barnes_hut_node(
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
        child_force, child_curve = _accumulate_barnes_hut_node(
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


def _add_barnes_hut_repulsion(
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
        Updates ``forces`` and ``curvature`` in-place.
    """
    tree = _quadtree_root(positions, weights)
    for node_index in range(int(positions.shape[0])):
        node_force, node_curve = _accumulate_barnes_hut_node(
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


def _add_attraction(
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
        Updates ``forces`` and ``curvature`` in-place.
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


def _add_gravity(
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
        Updates ``forces`` and ``curvature`` in-place.
    """
    if scaled_gravity == 0.0:
        return
    delta = barycenter[None, :] - positions
    distances = torch.linalg.norm(delta, dim=1).clamp(min=_MIN_DISTANCE)
    multipliers = scaled_gravity * weights * distances.pow(attr_exponent - 2.0)
    forces += delta * multipliers[:, None]
    curvature += multipliers * abs(attr_exponent - 1.0)


def _average_distances(positions: torch.Tensor) -> torch.Tensor:
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


def _normalize_positions(positions: torch.Tensor) -> torch.Tensor:
    """Center and scale the final coordinates for stable comparisons.

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


def _layout_linlog_reference(
    graph: DaguaGraph,
    config: _LinLogConfig,
    seed: int,
) -> torch.Tensor:
    """Run the clean Python LinLog reference solver.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to lay out.
    config : _LinLogConfig
        Resolved LinLog parameters.
    seed : int
        Random seed used for initial coordinates.

    Returns
    -------
    torch.Tensor
        Final positions with shape ``[N, 2]``.
    """
    num_nodes = graph.num_nodes
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32)

    edge_index, edge_weights = _prepare_edges(graph)
    repulsion_weights = _node_repulsion_weights(num_nodes, edge_index, edge_weights)
    positions = _initial_positions(num_nodes, seed)

    for step in range(1, config.steps + 1):
        attr_exponent, repu_exponent = _scheduled_exponents(step, config)
        repu_factor, scaled_gravity = _energy_factors(
            repulsion_weights=repulsion_weights,
            edge_weights=edge_weights,
            attr_exponent=attr_exponent,
            repu_exponent=repu_exponent,
            grav_factor=config.grav_factor,
        )
        forces = torch.zeros_like(positions)
        curvature = torch.zeros((num_nodes,), dtype=positions.dtype)
        barycenter = _weighted_barycenter(positions, repulsion_weights)

        if num_nodes > config.barnes_hut_threshold:
            _add_barnes_hut_repulsion(
                positions=positions,
                weights=repulsion_weights,
                repu_factor=repu_factor,
                repu_exponent=repu_exponent,
                theta=config.barnes_hut_theta,
                forces=forces,
                curvature=curvature,
            )
        else:
            _add_exact_repulsion(
                positions=positions,
                weights=repulsion_weights,
                repu_factor=repu_factor,
                repu_exponent=repu_exponent,
                forces=forces,
                curvature=curvature,
            )

        _add_attraction(
            positions=positions,
            edge_index=edge_index,
            edge_weights=edge_weights,
            attr_exponent=attr_exponent,
            forces=forces,
            curvature=curvature,
        )
        _add_gravity(
            positions=positions,
            weights=repulsion_weights,
            barycenter=barycenter,
            scaled_gravity=scaled_gravity,
            attr_exponent=attr_exponent,
            forces=forces,
            curvature=curvature,
        )

        steps = forces / curvature.clamp(min=_MIN_DISTANCE)[:, None]
        max_step = _average_distances(positions).clamp(min=0.01)
        lengths = torch.linalg.norm(steps, dim=1).clamp(min=_MIN_DISTANCE)
        steps *= torch.minimum(torch.ones_like(lengths), max_step / lengths)[:, None]
        cooling = 1.0 - 0.95 * (float(step - 1) / float(max(config.steps, 1)))
        positions += steps * cooling

    return _normalize_positions(positions)


@register
class LinLogReference(CompetitorBase):
    """Competitor adapter for a paper-spec LinLog reference implementation."""

    name = "linlog"
    max_nodes = 50_000
    variant_param_names = frozenset(
        {
            "a",
            "attrExponent",
            "barnesHutTheta",
            "barnesHutThreshold",
            "gravFactor",
            "iterations",
            "r",
            "repuExponent",
            "steps",
        }
    )

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the default LinLog reference layout.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility timeout.
        seed : int | None, default=None
            Random seed for initialization. ``None`` uses ``42``.

        Returns
        -------
        CompetitorResult
            Layout result and runtime metadata.
        """
        return self.layout_with_variant(graph, timeout=timeout, seed=seed, variant_params=None)

    def layout_with_variant(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
        variant_params: Optional[Mapping[str, Any]] = None,
    ) -> CompetitorResult:
        """Run the LinLog reference layout with variant parameters.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility timeout.
        seed : int | None, default=None
            Random seed for initialization. ``None`` uses ``42``.
        variant_params : Mapping[str, Any] | None, default=None
            Optional Noack-style or Dagua-style parameter overrides.

        Returns
        -------
        CompetitorResult
            Layout result and runtime metadata.
        """
        del timeout
        start = time.perf_counter()
        try:
            config = _resolve_config(variant_params)
            layout_seed = _DEFAULT_SEED if seed is None else seed
            pos = _layout_linlog_reference(graph=graph, config=config, seed=layout_seed)
            elapsed = time.perf_counter() - start
            return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error=str(exc),
            )
