"""ForceAtlas2 force-directed layout pipeline."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from dagua.layout.ops.base import Op, Pipeline, Repeat
from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig
from dagua.layout.ops.force import FA2ForceStep, FA2ForceStepConfig
from dagua.layout.ops.init import (
    FA2InitializePositions,
    FA2InitializePositionsConfig,
    ValidateFA2Inputs,
    ValidateFA2InputsConfig,
)
from dagua.layout.ops.pipelines import resolve_fidelity_dtype
from dagua.layout.ops.preprocess import FA2PrepareState, FA2PrepareStateConfig
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def _reference_exact_edge_arrays(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor],
) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Build reference ordered edge, weight, and mass arrays.

    Parameters
    ----------
    edge_index : torch.Tensor
        Input edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.

    Returns
    -------
    tuple[np.ndarray, Optional[np.ndarray], np.ndarray]
        Undirected edge pairs, optional edge weights, and FA2 node masses.
    """
    collapsed: dict[tuple[int, int], float] = {}
    edges_np = edge_index.detach().cpu().numpy()
    weights_np = None if edge_weights is None else edge_weights.detach().cpu().numpy()
    for edge_offset in range(edges_np.shape[1]):
        source = int(edges_np[0, edge_offset])
        target = int(edges_np[1, edge_offset])
        if source == target:
            continue
        key = (min(source, target), max(source, target))
        collapsed[key] = 1.0 if weights_np is None else float(weights_np[edge_offset])

    ordered_pairs = sorted(collapsed)
    edge_pairs = np.asarray(ordered_pairs, dtype=np.int64).reshape((-1, 2))
    weights = None
    if edge_weights is not None:
        weights = np.asarray([collapsed[pair] for pair in ordered_pairs], dtype=np.float64)
    degree = np.zeros(num_nodes, dtype=np.float64)
    for source, target in ordered_pairs:
        degree[source] += 1.0
        degree[target] += 1.0
    return edge_pairs, weights, degree + 1.0


def _layout_fa2_reference_exact(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    steps: int,
    seed: int,
    gravity: float,
    scaling_ratio: float,
    linlog: bool,
    strong_gravity: bool,
    outbound_attraction_distribution: bool,
    edge_weights: Optional[torch.Tensor],
    dissuade_hubs: bool,
    edge_weight_influence: float,
    fidelity_dtype: torch.dtype,
) -> torch.Tensor:
    """Run the live ``fa2`` exact-loop kernel for fidelity mode.

    Parameters
    ----------
    edge_index : torch.Tensor
        Input edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    steps : int
        Number of ForceAtlas2 iterations.
    seed : int
        Python ``random.Random`` seed.
    gravity : float
        Gravity coefficient.
    scaling_ratio : float
        Repulsion scaling coefficient.
    linlog : bool
        Whether to use logarithmic attraction.
    strong_gravity : bool
        Whether to use strong gravity.
    outbound_attraction_distribution : bool
        Whether attraction is divided by source mass.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.
    dissuade_hubs : bool
        Whether to divide attraction by source mass without outbound
        compensation.
    edge_weight_influence : float
        Edge-weight exponent.

    Returns
    -------
    torch.Tensor
        Final reference-order coordinates with shape ``[N, 2]``.
    """
    if num_nodes == 0:
        return torch.zeros((0, 2), dtype=fidelity_dtype, device=edge_index.device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=fidelity_dtype, device=edge_index.device)

    rng = random.Random(seed)
    pos = np.asarray([[rng.random(), rng.random()] for _ in range(num_nodes)], dtype=np.float64)
    edges, weights, mass = _reference_exact_edge_arrays(edge_index, num_nodes, edge_weights)
    outbound_compensation = float(np.mean(mass)) if outbound_attraction_distribution else 1.0
    old_force = np.zeros_like(pos)
    force = np.zeros_like(pos)
    speed = 1.0
    speed_efficiency = 1.0

    for _ in range(steps):
        old_force[:, :] = force
        force[:, :] = 0.0
        for node_index in range(num_nodes):
            for other_index in range(node_index):
                x_dist = float(pos[node_index, 0] - pos[other_index, 0])
                y_dist = float(pos[node_index, 1] - pos[other_index, 1])
                distance_sq = (x_dist * x_dist) + (y_dist * y_dist)
                if distance_sq > 0.0:
                    factor = scaling_ratio * mass[node_index] * mass[other_index] / distance_sq
                    force[node_index, 0] += x_dist * factor
                    force[node_index, 1] += y_dist * factor
                    force[other_index, 0] -= x_dist * factor
                    force[other_index, 1] -= y_dist * factor

        for node_index in range(num_nodes):
            x_coord = float(pos[node_index, 0])
            y_coord = float(pos[node_index, 1])
            if strong_gravity:
                if x_coord != 0.0 or y_coord != 0.0:
                    factor = scaling_ratio * mass[node_index] * gravity
                    force[node_index, 0] -= x_coord * factor
                    force[node_index, 1] -= y_coord * factor
            else:
                distance = math.sqrt((x_coord * x_coord) + (y_coord * y_coord))
                if distance > 0.0:
                    factor = mass[node_index] * gravity / distance
                    force[node_index, 0] -= x_coord * factor
                    force[node_index, 1] -= y_coord * factor

        for edge_offset in range(edges.shape[0]):
            source = int(edges[edge_offset, 0])
            target = int(edges[edge_offset, 1])
            x_dist = float(pos[source, 0] - pos[target, 0])
            y_dist = float(pos[source, 1] - pos[target, 1])
            edge_factor = 1.0
            if weights is not None:
                weight = float(weights[edge_offset])
                if edge_weight_influence == 1.0:
                    edge_factor = weight
                elif edge_weight_influence != 0.0:
                    edge_factor = weight ** float(edge_weight_influence)
            if linlog:
                distance = math.sqrt((x_dist * x_dist) + (y_dist * y_dist))
                if distance <= 0.0:
                    continue
                factor = -outbound_compensation * edge_factor * math.log(1.0 + distance) / distance
            else:
                factor = -outbound_compensation * edge_factor
            if outbound_attraction_distribution:
                factor /= mass[source]
            if dissuade_hubs and not outbound_attraction_distribution:
                factor /= mass[source]
            force[source, 0] += x_dist * factor
            force[source, 1] += y_dist * factor
            force[target, 0] -= x_dist * factor
            force[target, 1] -= y_dist * factor

        total_swinging = 0.0
        total_effective_traction = 0.0
        for node_index in range(num_nodes):
            diff_x = float(old_force[node_index, 0] - force[node_index, 0])
            diff_y = float(old_force[node_index, 1] - force[node_index, 1])
            sum_x = float(old_force[node_index, 0] + force[node_index, 0])
            sum_y = float(old_force[node_index, 1] + force[node_index, 1])
            total_swinging += mass[node_index] * math.sqrt((diff_x * diff_x) + (diff_y * diff_y))
            total_effective_traction += (
                0.5 * mass[node_index] * math.sqrt((sum_x * sum_x) + (sum_y * sum_y))
            )

        estimated_jitter = 0.05 * math.sqrt(num_nodes)
        min_jitter = math.sqrt(estimated_jitter)
        jitter = min_jitter
        if total_effective_traction > 0.0:
            jitter = max(
                min_jitter,
                min(10.0, estimated_jitter * total_effective_traction / (num_nodes * num_nodes)),
            )
        if total_effective_traction > 0.0 and total_swinging / total_effective_traction > 2.0:
            if speed_efficiency > 0.05:
                speed_efficiency *= 0.5
            jitter = max(jitter, 1.0)
        target_speed = (
            float("inf")
            if total_swinging == 0.0
            else jitter * speed_efficiency * total_effective_traction / total_swinging
        )
        if total_swinging > jitter * total_effective_traction:
            if speed_efficiency > 0.05:
                speed_efficiency *= 0.7
        elif speed < 1000.0:
            speed_efficiency *= 1.3
        speed = speed + min(target_speed - speed, 0.5 * speed)

        for node_index in range(num_nodes):
            diff_x = float(old_force[node_index, 0] - force[node_index, 0])
            diff_y = float(old_force[node_index, 1] - force[node_index, 1])
            swinging = mass[node_index] * math.sqrt((diff_x * diff_x) + (diff_y * diff_y))
            factor = speed / (1.0 + math.sqrt(speed * swinging))
            pos[node_index, 0] += force[node_index, 0] * factor
            pos[node_index, 1] += force[node_index, 1] * factor

    return torch.from_numpy(pos).to(device=edge_index.device, dtype=fidelity_dtype)


@dataclass
class _FA2ReferenceNode:
    """Mutable 2D node mirroring ``fa2util.Node2D`` storage.

    Parameters
    ----------
    index : int
        Reference-order node index.
    mass : float
        ForceAtlas2 mass, equal to one plus undirected degree.
    x : float
        Current x-coordinate.
    y : float
        Current y-coordinate.
    dx : float
        Current x-force accumulator.
    dy : float
        Current y-force accumulator.
    old_dx : float
        Previous iteration x-force accumulator.
    old_dy : float
        Previous iteration y-force accumulator.
    size : float
        Node radius for anti-collision mode. Fidelity Barnes-Hut currently
        uses the reference default of no anti-collision, so this remains zero.
    """

    index: int
    mass: float
    x: float
    y: float
    dx: float = 0.0
    dy: float = 0.0
    old_dx: float = 0.0
    old_dy: float = 0.0
    size: float = 0.0


@dataclass(frozen=True)
class _FA2ReferenceEdge:
    """Reference-order ForceAtlas2 edge.

    Parameters
    ----------
    node1 : int
        Source endpoint index used by ``fa2util.apply_attraction``.
    node2 : int
        Target endpoint index used by ``fa2util.apply_attraction``.
    weight : float
        Edge weight after duplicate handling, before edge-weight influence.
    """

    node1: int
    node2: int
    weight: float


class _FA2ReferenceRegion:
    """Pure Python port of ``fa2util.Region`` for 2D fidelity mode."""

    def __init__(self, nodes: list[_FA2ReferenceNode]) -> None:
        """Initialize a Barnes-Hut region and compute its geometry.

        Parameters
        ----------
        nodes : list[_FA2ReferenceNode]
            Node objects contained in this region, in reference insertion order.

        Returns
        -------
        None
            Initializes the region in place.
        """
        self.mass = 0.0
        self.mass_center = [0.0, 0.0]
        self.size = 0.0
        self.nodes = nodes
        self.subregions: list[_FA2ReferenceRegion] = []
        self.update_mass_and_geometry()

    def update_mass_and_geometry(self) -> None:
        """Match ``Region.updateMassAndGeometry`` arithmetic order.

        Returns
        -------
        None
            Updates ``mass``, ``mass_center``, and ``size`` in place.
        """
        if len(self.nodes) == 1:
            node = self.nodes[0]
            self.mass = node.mass
            self.mass_center = [node.x, node.y]
            self.size = 0.0
            return

        if len(self.nodes) == 0:
            return

        self.mass = 0.0
        mass_sum = [0.0, 0.0]
        for node in self.nodes:
            self.mass += node.mass
            mass_sum[0] = mass_sum[0] + node.x * node.mass
            mass_sum[1] = mass_sum[1] + node.y * node.mass

        if self.mass > 0.0:
            self.mass_center[0] = mass_sum[0] / self.mass
            self.mass_center[1] = mass_sum[1] / self.mass

        self.size = 0.0
        for node in self.nodes:
            x_diff = node.x - self.mass_center[0]
            y_diff = node.y - self.mass_center[1]
            distance = math.sqrt((x_diff * x_diff) + (y_diff * y_diff))
            if 2.0 * distance > self.size:
                self.size = 2.0 * distance

    def build_subregions(self) -> None:
        """Partition nodes into Cython ``Region`` buckets and recurse.

        Returns
        -------
        None
            Populates ``subregions`` in bucket order ``0, 1, 2, 3``.
        """
        if len(self.nodes) <= 1:
            return

        buckets: list[list[_FA2ReferenceNode]] = [[] for _ in range(4)]
        for node in self.nodes:
            bucket_idx = 0
            if node.x >= self.mass_center[0]:
                bucket_idx |= 1
            if node.y >= self.mass_center[1]:
                bucket_idx |= 2
            buckets[bucket_idx].append(node)

        for bucket_nodes in buckets:
            if len(bucket_nodes) == 0:
                continue
            if len(bucket_nodes) < len(self.nodes):
                self.subregions.append(_FA2ReferenceRegion(bucket_nodes))
                continue
            for node in bucket_nodes:
                self.subregions.append(_FA2ReferenceRegion([node]))

        for subregion in self.subregions:
            subregion.build_subregions()

    def apply_force(self, node: _FA2ReferenceNode, theta: float, coefficient: float) -> None:
        """Apply this region's repulsion to one node in reference order.

        Parameters
        ----------
        node : _FA2ReferenceNode
            Target node whose ``dx`` and ``dy`` accumulators are updated.
        theta : float
            Barnes-Hut opening threshold.
        coefficient : float
            ForceAtlas2 repulsion scaling ratio.

        Returns
        -------
        None
            Updates ``node`` force accumulators in place.
        """
        if len(self.nodes) == 0:
            return
        if len(self.nodes) < 2:
            if self.nodes[0] is not node:
                self.apply_region_repulsion(node=node, coefficient=coefficient)
            return

        x_diff = node.x - self.mass_center[0]
        y_diff = node.y - self.mass_center[1]
        distance = math.sqrt((x_diff * x_diff) + (y_diff * y_diff))
        if distance * theta > self.size:
            self.apply_region_repulsion(node=node, coefficient=coefficient)
            return

        for subregion in self.subregions:
            subregion.apply_force(node=node, theta=theta, coefficient=coefficient)

    def apply_region_repulsion(self, node: _FA2ReferenceNode, coefficient: float) -> None:
        """Apply ``linRepulsion_region_2d`` to one node.

        Parameters
        ----------
        node : _FA2ReferenceNode
            Target node whose force accumulator is updated.
        coefficient : float
            ForceAtlas2 repulsion scaling ratio.

        Returns
        -------
        None
            Updates ``node.dx`` and ``node.dy`` in place.
        """
        x_dist = node.x - self.mass_center[0]
        y_dist = node.y - self.mass_center[1]
        distance2 = (x_dist * x_dist) + (y_dist * y_dist)
        if distance2 > 0.0:
            factor = coefficient * node.mass * self.mass / distance2
            node.dx += x_dist * factor
            node.dy += y_dist * factor

    def apply_force_on_nodes(
        self,
        nodes: list[_FA2ReferenceNode],
        theta: float,
        coefficient: float,
    ) -> None:
        """Apply Barnes-Hut repulsion to nodes in list order.

        Parameters
        ----------
        nodes : list[_FA2ReferenceNode]
            Target nodes in reference insertion order.
        theta : float
            Barnes-Hut opening threshold.
        coefficient : float
            ForceAtlas2 repulsion scaling ratio.

        Returns
        -------
        None
            Updates each node's force accumulator in place.
        """
        for node in nodes:
            self.apply_force(node=node, theta=theta, coefficient=coefficient)


def _reference_barnes_hut_nodes_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor],
    seed: int,
) -> tuple[list[_FA2ReferenceNode], list[_FA2ReferenceEdge]]:
    """Build reference-order mutable nodes and edges for FA2 fidelity mode.

    Parameters
    ----------
    edge_index : torch.Tensor
        Input edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.
    seed : int
        Python ``random.Random`` seed used by the reference package.

    Returns
    -------
    tuple[list[_FA2ReferenceNode], list[_FA2ReferenceEdge]]
        Mutable nodes and sorted undirected edges.
    """
    edge_pairs, weights, mass = _reference_exact_edge_arrays(edge_index, num_nodes, edge_weights)
    rng = random.Random(seed)
    nodes = [
        _FA2ReferenceNode(
            index=node_index,
            mass=float(mass[node_index]),
            x=float(rng.random()),
            y=float(rng.random()),
        )
        for node_index in range(num_nodes)
    ]
    if weights is None:
        edges = [
            _FA2ReferenceEdge(int(edge_pairs[offset, 0]), int(edge_pairs[offset, 1]), 1.0)
            for offset in range(edge_pairs.shape[0])
        ]
    else:
        edges = [
            _FA2ReferenceEdge(
                int(edge_pairs[offset, 0]),
                int(edge_pairs[offset, 1]),
                float(weights[offset]),
            )
            for offset in range(edge_pairs.shape[0])
        ]
    return nodes, edges


def _apply_reference_gravity(
    nodes: list[_FA2ReferenceNode],
    gravity: float,
    scaling_ratio: float,
    strong_gravity: bool,
) -> None:
    """Apply Cython FA2 gravity to reference nodes.

    Parameters
    ----------
    nodes : list[_FA2ReferenceNode]
        Mutable nodes in reference order.
    gravity : float
        Gravity coefficient.
    scaling_ratio : float
        Strong-gravity scaling coefficient.
    strong_gravity : bool
        Whether to use strong-gravity mode.

    Returns
    -------
    None
        Updates node force accumulators in place.
    """
    for node in nodes:
        if strong_gravity:
            if node.x != 0.0 or node.y != 0.0:
                factor = scaling_ratio * node.mass * gravity
                node.dx -= node.x * factor
                node.dy -= node.y * factor
            continue

        distance = math.sqrt((node.x * node.x) + (node.y * node.y))
        if distance > 0.0:
            factor = node.mass * gravity / distance
            node.dx -= node.x * factor
            node.dy -= node.y * factor


def _apply_reference_attraction(
    nodes: list[_FA2ReferenceNode],
    edges: list[_FA2ReferenceEdge],
    outbound_attraction_distribution: bool,
    outbound_compensation: float,
    edge_weight_influence: float,
    linlog: bool,
) -> None:
    """Apply Cython FA2 attraction loops to reference nodes.

    Parameters
    ----------
    nodes : list[_FA2ReferenceNode]
        Mutable nodes in reference order.
    edges : list[_FA2ReferenceEdge]
        Sorted undirected edges in reference iteration order.
    outbound_attraction_distribution : bool
        Whether to divide attraction by source mass.
    outbound_compensation : float
        Mean-mass compensation used by ForceAtlas2.
    edge_weight_influence : float
        Edge-weight exponent.
    linlog : bool
        Whether to use logarithmic attraction.

    Returns
    -------
    None
        Updates node force accumulators in place.
    """
    for edge in edges:
        edge_factor = 1.0
        if edge_weight_influence == 1.0:
            edge_factor = edge.weight
        elif edge_weight_influence != 0.0:
            edge_factor = edge.weight**edge_weight_influence

        node1 = nodes[edge.node1]
        node2 = nodes[edge.node2]
        x_dist = node1.x - node2.x
        y_dist = node1.y - node2.y
        if linlog:
            distance = math.sqrt((x_dist * x_dist) + (y_dist * y_dist))
            if distance <= 0.0:
                continue
            log_factor = math.log(1.0 + distance) / distance
            if outbound_attraction_distribution:
                factor = -outbound_compensation * edge_factor * log_factor / node1.mass
            else:
                factor = -outbound_compensation * edge_factor * log_factor
        elif outbound_attraction_distribution:
            factor = -outbound_compensation * edge_factor / node1.mass
        else:
            factor = -outbound_compensation * edge_factor

        node1.dx += x_dist * factor
        node1.dy += y_dist * factor
        node2.dx -= x_dist * factor
        node2.dy -= y_dist * factor


def _adjust_reference_speed_and_apply_forces(
    nodes: list[_FA2ReferenceNode],
    speed: float,
    speed_efficiency: float,
    jitter_tolerance: float,
) -> tuple[float, float]:
    """Apply Cython FA2 adaptive speed update and move nodes.

    Parameters
    ----------
    nodes : list[_FA2ReferenceNode]
        Mutable nodes in reference order.
    speed : float
        Previous adaptive speed value.
    speed_efficiency : float
        Previous speed-efficiency value.
    jitter_tolerance : float
        User jitter-tolerance multiplier.

    Returns
    -------
    tuple[float, float]
        Updated speed and speed-efficiency values.
    """
    total_swinging = 0.0
    total_effective_traction = 0.0
    for node in nodes:
        diff_x = node.old_dx - node.dx
        diff_y = node.old_dy - node.dy
        sum_x = node.old_dx + node.dx
        sum_y = node.old_dy + node.dy
        swinging = math.sqrt((diff_x * diff_x) + (diff_y * diff_y))
        total_swinging += node.mass * swinging
        total_effective_traction += 0.5 * node.mass * math.sqrt((sum_x * sum_x) + (sum_y * sum_y))

    estimated_jitter = 0.05 * math.sqrt(len(nodes))
    min_jitter = math.sqrt(estimated_jitter)
    if len(nodes) > 0 and total_effective_traction > 0.0:
        jitter = jitter_tolerance * max(
            min_jitter,
            min(10.0, estimated_jitter * total_effective_traction / (len(nodes) * len(nodes))),
        )
    else:
        jitter = jitter_tolerance * min_jitter

    min_speed_efficiency = 0.05
    if total_effective_traction > 0.0 and total_swinging / total_effective_traction > 2.0:
        if speed_efficiency > min_speed_efficiency:
            speed_efficiency *= 0.5
        jitter = max(jitter, jitter_tolerance)

    if total_swinging == 0.0:
        target_speed = float("inf")
    else:
        target_speed = jitter * speed_efficiency * total_effective_traction / total_swinging

    if total_swinging > jitter * total_effective_traction:
        if speed_efficiency > min_speed_efficiency:
            speed_efficiency *= 0.7
    elif speed < 1000.0:
        speed_efficiency *= 1.3

    speed = speed + min(target_speed - speed, 0.5 * speed)

    for node in nodes:
        diff_x = node.old_dx - node.dx
        diff_y = node.old_dy - node.dy
        swinging = node.mass * math.sqrt((diff_x * diff_x) + (diff_y * diff_y))
        factor = speed / (1.0 + math.sqrt(speed * swinging))
        node.x += node.dx * factor
        node.y += node.dy * factor

    return speed, speed_efficiency


def _layout_fa2_reference_barnes_hut(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    steps: int,
    seed: int,
    gravity: float,
    scaling_ratio: float,
    linlog: bool,
    strong_gravity: bool,
    outbound_attraction_distribution: bool,
    edge_weights: Optional[torch.Tensor],
    edge_weight_influence: float,
    barnes_hut_theta: float,
    fidelity_dtype: torch.dtype,
) -> torch.Tensor:
    """Run the pure Python port of ``fa2util.Region`` in fidelity mode.

    Parameters
    ----------
    edge_index : torch.Tensor
        Input edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    steps : int
        Number of ForceAtlas2 iterations.
    seed : int
        Python ``random.Random`` seed.
    gravity : float
        Gravity coefficient.
    scaling_ratio : float
        Repulsion scaling coefficient.
    linlog : bool
        Whether to use logarithmic attraction.
    strong_gravity : bool
        Whether to use strong-gravity mode.
    outbound_attraction_distribution : bool
        Whether attraction is divided by source mass.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.
    edge_weight_influence : float
        Edge-weight exponent.
    barnes_hut_theta : float
        Barnes-Hut opening threshold.
    fidelity_dtype : torch.dtype
        Output dtype used by fidelity mode.

    Returns
    -------
    torch.Tensor
        Final reference-order coordinates with shape ``[N, 2]``.
    """
    if num_nodes == 0:
        return torch.zeros((0, 2), dtype=fidelity_dtype, device=edge_index.device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=fidelity_dtype, device=edge_index.device)

    nodes, edges = _reference_barnes_hut_nodes_edges(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weights=edge_weights,
        seed=seed,
    )
    outbound_compensation = (
        sum(node.mass for node in nodes) / float(len(nodes))
        if outbound_attraction_distribution
        else 1.0
    )
    speed = 1.0
    speed_efficiency = 1.0

    for _ in range(steps):
        for node in nodes:
            node.old_dx = node.dx
            node.old_dy = node.dy
            node.dx = 0.0
            node.dy = 0.0

        root_region = _FA2ReferenceRegion(nodes)
        root_region.build_subregions()
        root_region.apply_force_on_nodes(
            nodes=nodes,
            theta=barnes_hut_theta,
            coefficient=scaling_ratio,
        )
        _apply_reference_gravity(
            nodes=nodes,
            gravity=gravity,
            scaling_ratio=scaling_ratio,
            strong_gravity=strong_gravity,
        )
        _apply_reference_attraction(
            nodes=nodes,
            edges=edges,
            outbound_attraction_distribution=outbound_attraction_distribution,
            outbound_compensation=outbound_compensation,
            edge_weight_influence=edge_weight_influence,
            linlog=linlog,
        )
        speed, speed_efficiency = _adjust_reference_speed_and_apply_forces(
            nodes=nodes,
            speed=speed,
            speed_efficiency=speed_efficiency,
            jitter_tolerance=1.0,
        )

    pos = np.asarray([(node.x, node.y) for node in nodes], dtype=np.float64)
    return torch.from_numpy(pos).to(device=edge_index.device, dtype=fidelity_dtype)


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
        Whether to divide attraction by source mass when outbound attraction
        distribution is disabled.
    edge_weight_influence : float
        Exponent applied to edge weights.
    barnes_hut : bool
        Whether to use Barnes-Hut approximation for repulsion.
    barnes_hut_theta : float
        Acceptance threshold for Barnes-Hut.
    jitter_tolerance : float
        Jitter tolerance for adaptive speed control.
    fidelity_mode : bool
        Whether to run FA2 internal tensors in float64 for reference parity.
    fidelity_dtype : torch.dtype
        Internal dtype used when fidelity mode is enabled.
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
    fidelity_mode: bool = False
    fidelity_dtype: Optional[torch.dtype] = None


@dataclass(frozen=True)
class _FA2ReferenceBarnesHutSolve(Op):
    """Single-op wrapper for the fidelity Barnes-Hut reference port.

    Parameters
    ----------
    config : FA2Config
        Resolved ForceAtlas2 configuration.
    dtype : torch.dtype
        Output dtype used by fidelity mode.
    """

    config: FA2Config
    dtype: torch.dtype

    name: str = "fa2_reference_barnes_hut_solve"

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run the pure Python Barnes-Hut fidelity solver.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state to receive the final positions.
        ctx : RuntimeContext
            Execution infrastructure. Only the planned device is honored by
            the returned tensor.

        Returns
        -------
        SolveState
            State with ``pos`` populated by the fidelity Barnes-Hut layout.
        """
        del ctx

        state.pos = _layout_fa2_reference_barnes_hut(
            problem.edge_index,
            problem.num_nodes,
            steps=self.config.steps,
            seed=problem.seed,
            gravity=self.config.gravity,
            scaling_ratio=self.config.scaling_ratio,
            linlog=self.config.linlog,
            strong_gravity=self.config.strong_gravity,
            outbound_attraction_distribution=self.config.outbound_attraction_distribution,
            edge_weights=problem.edge_weights,
            edge_weight_influence=self.config.edge_weight_influence,
            barnes_hut_theta=self.config.barnes_hut_theta,
            fidelity_dtype=self.dtype,
        )
        return state


def build_fa2_pipeline(config: Optional[FA2Config] = None) -> Pipeline:
    """Build a ForceAtlas2 pipeline.

    Reference fidelity
    ------------------
    Targets: ``fa2`` 1.1.2 / Jacomy et al. (2014), "ForceAtlas2, a Continuous
        Graph Layout Algorithm for Handy Network Visualization".
    Fidelity mode: ``FA2Config.fidelity_mode=True`` uses float64 state and
        reference duplicate-edge overwrite semantics instead of Dagua's summed
        edge weights.
    Verified at: final 100-seed report, strong equivalent for most variants;
        median RMSD 0.048 to 0.173, with dissuade-hubs partial at 0.104.
    Known divergences:
        - Dagua keeps explicit tensor-device handling and optional weighted
          behavior outside fidelity mode.

    Parameters
    ----------
    config : FA2Config, optional
        ForceAtlas2 hyperparameters controlling iteration count, gravity,
        attraction and repulsion variants, and Barnes-Hut acceleration. Uses
        defaults when not provided.

    Returns
    -------
    Pipeline
        Pipeline implementing the classical ForceAtlas2 algorithm. The
        pipeline produces final node coordinates by validating inputs,
        initializing positions, preparing graph-dependent state, and applying
        repeated FA2 force steps with adaptive speed control.

    Raises
    ------
    ValueError
        If ``config.steps`` is negative.
    """
    resolved = config or FA2Config()
    if resolved.steps < 0:
        raise ValueError("steps must be non-negative.")
    if resolved.barnes_hut_theta <= 0.0:
        raise ValueError("barnes_hut_theta must be positive")

    dtype = (
        resolve_fidelity_dtype(resolved.fidelity_mode, resolved.fidelity_dtype)
        if resolved.fidelity_mode
        else torch.float32
    )
    if resolved.fidelity_mode and resolved.barnes_hut:
        return Pipeline(
            [_FA2ReferenceBarnesHutSolve(config=resolved, dtype=dtype)],
            name="fa2_reference_barnes_hut_pipeline",
        )

    return Pipeline(
        [
            ValidateFA2Inputs(
                ValidateFA2InputsConfig(
                    steps=resolved.steps,
                    barnes_hut_theta=resolved.barnes_hut_theta,
                )
            ),
            FixedSteps(FixedStepsConfig(n=resolved.steps)),
            FA2InitializePositions(FA2InitializePositionsConfig(dtype=dtype)),
            FA2PrepareState(
                FA2PrepareStateConfig(
                    outbound_attraction_distribution=resolved.outbound_attraction_distribution,
                    dtype=dtype,
                    duplicate_weight_policy="last" if resolved.fidelity_mode else "sum",
                )
            ),
            Repeat(
                n=resolved.steps,
                ops=[
                    FA2ForceStep(
                        FA2ForceStepConfig(
                            gravity=resolved.gravity,
                            scaling_ratio=resolved.scaling_ratio,
                            linlog=resolved.linlog,
                            strong_gravity=resolved.strong_gravity,
                            outbound_attraction_distribution=resolved.outbound_attraction_distribution,
                            dissuade_hubs=resolved.dissuade_hubs,
                            edge_weight_influence=resolved.edge_weight_influence,
                            barnes_hut=resolved.barnes_hut,
                            barnes_hut_theta=resolved.barnes_hut_theta,
                            jitter_tolerance=resolved.jitter_tolerance,
                        )
                    ),
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
    fidelity_mode: bool = False,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run the ForceAtlas2 pipeline as a drop-in replacement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``. Unused and retained
        for API compatibility.
    steps : int, default=100
        Number of ForceAtlas2 iterations.
    seed : int, default=42
        Random seed for the Python-random initialization.
    gravity : float, default=1.0
        Gravity coefficient applied each iteration.
    scaling_ratio : float, default=2.0
        Repulsion scaling coefficient.
    linlog : bool, default=False
        Whether to use the LinLog attraction variant.
    strong_gravity : bool, default=False
        Whether to use strong-gravity mode.
    outbound_attraction_distribution : bool, default=True
        Whether to normalize attraction by source mass.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.
    dissuade_hubs : bool, default=False
        Whether to divide attraction by source mass when outbound attraction
        distribution is disabled.
    edge_weight_influence : float, default=1.0
        Exponent applied to edge weights during attraction.
    barnes_hut : bool, default=False
        Whether to use Barnes-Hut approximation for repulsion.
    barnes_hut_theta : float, default=1.2
        Acceptance threshold for Barnes-Hut aggregation.
    fidelity_mode : bool, default=False
        Run FA2 internal tensors in float64 to better match the live
        ForceAtlas2 reference. The default keeps the historical float32 path.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``steps``, or ``edge_weights`` are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    del node_sizes

    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if barnes_hut_theta <= 0.0:
        raise ValueError("barnes_hut_theta must be positive")

    resolved_dtype = resolve_fidelity_dtype(fidelity_mode, fidelity_dtype)
    if fidelity_mode:
        if barnes_hut:
            return _layout_fa2_reference_barnes_hut(
                edge_index,
                num_nodes,
                steps=steps,
                seed=seed,
                gravity=gravity,
                scaling_ratio=scaling_ratio,
                linlog=linlog,
                strong_gravity=strong_gravity,
                outbound_attraction_distribution=outbound_attraction_distribution,
                edge_weights=edge_weights,
                edge_weight_influence=edge_weight_influence,
                barnes_hut_theta=barnes_hut_theta,
                fidelity_dtype=resolved_dtype,
            )
        return _layout_fa2_reference_exact(
            edge_index,
            num_nodes,
            steps=steps,
            seed=seed,
            gravity=gravity,
            scaling_ratio=scaling_ratio,
            linlog=linlog,
            strong_gravity=strong_gravity,
            outbound_attraction_distribution=outbound_attraction_distribution,
            edge_weights=edge_weights,
            dissuade_hubs=dissuade_hubs,
            edge_weight_influence=edge_weight_influence,
            fidelity_dtype=resolved_dtype,
        )

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
        fidelity_mode=fidelity_mode,
        fidelity_dtype=resolved_dtype,
    )
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device=str(edge_index.device)))
    final_state = build_fa2_pipeline(config=config).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("FA2 pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["FA2Config", "build_fa2_pipeline", "layout_fa2_pipeline"]
