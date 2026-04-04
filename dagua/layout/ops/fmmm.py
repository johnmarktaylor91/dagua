"""FM^3 multilevel force-directed layout expressed as a composable ops pipeline.

FM^3 uses solar-system-style coarsening, a coarse FR initialization, Barnes-Hut
repulsion during refinement, and OGDF-style lambda interpolation for prolongation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import ClassVar, Optional, Tuple

import torch

from dagua.layout.ops.base import Op  # noqa: E402
from dagua.layout.ops.graph_utils import (
    layout_device as _layout_device,
)
from dagua.layout.ops.graph_utils import (
    layout_extent as _layout_extent,
)
from dagua.layout.ops.graph_utils import (
    normalize_positions as _normalize_positions,
)
from dagua.layout.ops.pipelines.fr import layout_fr_pipeline
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_MIN_DISTANCE = 1.0e-3
_COARSE_TARGET = 50
_MAX_TREE_DEPTH = 10
_COOLING_FACTOR = 0.99
_SOLAR_RANDOM_TRIES = 20
_WAGGLE_FACTOR = 0.05
_TYPE_SUN = 1
_TYPE_PLANET = 2
_TYPE_PLANET_WITH_MOONS = 3
_TYPE_MOON = 4


@dataclass
class _QuadCell:
    """Recursive quad-tree cell used for Barnes-Hut repulsion."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    indices: list[int]
    center_of_mass: torch.Tensor = field(
        default_factory=lambda: torch.zeros(2, dtype=torch.float32)
    )
    mass: float = 0.0
    children: list["_QuadCell"] = field(default_factory=list)


@dataclass
class _LevelGraph:
    """One multilevel FM^3 graph with per-edge desired lengths."""

    edge_index: torch.Tensor
    edge_lengths: torch.Tensor
    num_nodes: int
    edge_weights: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.float32)
    )


@dataclass
class _HierarchyStep:
    """Metadata needed to prolong one coarse FM^3 level back to its fine graph."""

    mapping: torch.Tensor
    node_types: list[int]
    dedicated_sun: list[int]
    dedicated_sun_distance: list[float]
    pm_nodes: list[int]
    moon_children: list[list[int]]
    lambda_values: list[list[float]]
    neighbor_suns: list[list[int]]


@dataclass
class _RandomNodeSet:
    """Mutable node set mirroring OGDF's swap-delete random selection structure."""

    nodes: list[int]
    positions: list[int]
    last_selectable_index: int
    star_masses: list[int]

    @classmethod
    def from_star_masses(cls, star_masses: list[int]) -> "_RandomNodeSet":
        """Create a selectable node set for the given star masses.

        Parameters
        ----------
        star_masses : list[int]
            Star masses for nodes ``[0, N)``.

        Returns
        -------
        _RandomNodeSet
            Initialized swap-delete node set.
        """
        num_nodes = len(star_masses)
        return cls(
            nodes=list(range(num_nodes)),
            positions=list(range(num_nodes)),
            last_selectable_index=num_nodes - 1,
            star_masses=star_masses,
        )

    def empty(self) -> bool:
        """Return whether no selectable nodes remain.

        Returns
        -------
        bool
            ``True`` when the selectable prefix is empty.
        """
        return self.last_selectable_index < 0

    def is_deleted(self, node: int) -> bool:
        """Return whether a node has been removed from the selectable prefix.

        Parameters
        ----------
        node : int
            Node index.

        Returns
        -------
        bool
            ``True`` if the node is already deleted.
        """
        return self.positions[node] > self.last_selectable_index

    def delete(self, node: int) -> None:
        """Delete a node from the selectable prefix if it is still active.

        Parameters
        ----------
        node : int
            Node index to delete.
        """
        if self.is_deleted(node):
            return
        del_index = self.positions[node]
        last_node = self.nodes[self.last_selectable_index]
        self.nodes[self.last_selectable_index] = node
        self.nodes[del_index] = last_node
        self.positions[node] = self.last_selectable_index
        self.positions[last_node] = del_index
        self.last_selectable_index -= 1

    def _get_random_node_common(self, rand_index: int, last_index: int) -> tuple[int, int]:
        """Swap a sampled node with the current tail and remove it from a prefix.

        Parameters
        ----------
        rand_index : int
            Sampled index inside the active prefix.
        last_index : int
            Tail index of the prefix currently being sampled.

        Returns
        -------
        tuple[int, int]
            The sampled node and the updated tail index.
        """
        random_node = self.nodes[rand_index]
        last_node = self.nodes[last_index]
        self.nodes[last_index] = random_node
        self.nodes[rand_index] = last_node
        self.positions[random_node] = last_index
        self.positions[last_node] = rand_index
        return random_node, last_index - 1

    def get_random_node_with_highest_star_mass(
        self,
        rng: random.Random,
        random_tries: int,
    ) -> int:
        """Sample several distinct candidates and keep the one with highest star mass.

        Parameters
        ----------
        rng : random.Random
            Pseudorandom generator.
        random_tries : int
            Number of distinct candidates to evaluate.

        Returns
        -------
        int
            Selected node index removed from the selectable prefix.
        """
        if self.empty():
            raise ValueError("cannot select from an empty node set")

        best_index = -1
        best_mass = 0
        last_try_index = self.last_selectable_index
        max_tries = min(random_tries, last_try_index + 1)
        for trial_index in range(1, max_tries + 1):
            sampled_index = rng.randint(0, last_try_index)
            mass = self.star_masses[self.nodes[sampled_index]]
            _, last_try_index = self._get_random_node_common(sampled_index, last_try_index)
            if trial_index == 1 or mass > best_mass:
                best_index = last_try_index + 1
                best_mass = mass

        if best_index < 0:
            raise RuntimeError("failed to select a sun node")

        selected_node, self.last_selectable_index = self._get_random_node_common(
            best_index,
            self.last_selectable_index,
        )
        return selected_node


def _fr_ideal_length(area: float, num_nodes: int) -> float:
    """Compute the FR ideal edge length for the current refinement level.

    Parameters
    ----------
    area : float
        Target drawing area for the current level.
    num_nodes : int
        Number of nodes on the current level.

    Returns
    -------
    float
        Ideal FR length ``k = sqrt(area / N)``.
    """
    return max((area / max(num_nodes, 1)) ** 0.5, _MIN_DISTANCE)


def _unique_edges_with_lengths(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert an edge tensor into unique undirected edges with lengths and weights.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    edge_weights : torch.Tensor, optional
        Optional per-edge weights with shape ``[E]``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Unique undirected edges with shape ``[2, E_u]`` and their desired
        lengths and attraction weights with shape ``[E_u]``.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")

    if edge_weights is not None and edge_weights.shape[0] != edge_index.shape[1]:
        raise ValueError(
            f"edge_weights length {edge_weights.shape[0]} does not match "
            f"edge count {edge_index.shape[1]}"
        )

    seen: dict[tuple[int, int], tuple[float, int, float]] = {}
    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    weights_cpu = (
        torch.ones((edge_index.shape[1],), dtype=torch.float32)
        if edge_weights is None
        else edge_weights.detach().to(device="cpu", dtype=torch.float32)
    )
    for edge_id, (source, target) in enumerate(
        zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist())
    ):
        if source < 0 or source >= num_nodes or target < 0 or target >= num_nodes:
            raise ValueError("edge_index contains a node index outside [0, num_nodes).")
        if source == target:
            continue
        pair = (min(source, target), max(source, target))
        length_sum, count, weight_sum = seen.get(pair, (0.0, 0, 0.0))
        seen[pair] = (
            length_sum + 1.0,
            count + 1,
            weight_sum + float(weights_cpu[edge_id].item()),
        )

    if not seen:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
            torch.empty((0,), dtype=torch.float32),
        )

    ordered_pairs = sorted(seen)
    lengths = [seen[pair][0] / seen[pair][1] for pair in ordered_pairs]
    weights = [seen[pair][2] for pair in ordered_pairs]
    return (
        torch.tensor(ordered_pairs, dtype=torch.long).transpose(0, 1).contiguous(),
        torch.tensor(lengths, dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32),
    )


def _build_weighted_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> list[list[tuple[int, int]]]:
    """Build an undirected adjacency list referencing edge indices.

    Parameters
    ----------
    edge_index : torch.Tensor
        Unique undirected edges with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    list[list[tuple[int, int]]]
        One sorted ``(neighbor, edge_id)`` list per node.
    """
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return adjacency

    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    for edge_id, (source, target) in enumerate(
        zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist())
    ):
        adjacency[source].append((target, edge_id))
        adjacency[target].append((source, edge_id))

    for neighbors in adjacency:
        neighbors.sort(key=lambda item: item[0])
    return adjacency


def _star_masses(
    node_masses: torch.Tensor,
    adjacency: list[list[tuple[int, int]]],
) -> list[int]:
    """Compute OGDF's star mass for each node.

    Parameters
    ----------
    node_masses : torch.Tensor
        Current-level node masses with shape ``[N]``.
    adjacency : list[list[tuple[int, int]]]
        Weighted adjacency indexed by edge id.

    Returns
    -------
    list[int]
        Star mass for each node.
    """
    masses_cpu = node_masses.to(device="cpu", dtype=torch.long).tolist()
    return [
        masses_cpu[node] + sum(masses_cpu[neighbor] for neighbor, _ in neighbors)
        for node, neighbors in enumerate(adjacency)
    ]


def _coarsen_level(
    level_graph: _LevelGraph,
    node_masses: torch.Tensor,
    rng: random.Random,
) -> tuple[_HierarchyStep, _LevelGraph, torch.Tensor]:
    """Collapse one FM^3 level using OGDF-style solar systems.

    Parameters
    ----------
    level_graph : _LevelGraph
        Fine graph for the current level.
    node_masses : torch.Tensor
        Current-level node masses with shape ``[N]``.
    rng : random.Random
        Pseudorandom generator used for sun selection.

    Returns
    -------
    tuple[_HierarchyStep, _LevelGraph, torch.Tensor]
        Prolongation metadata, coarse graph, and coarse node masses.
    """
    num_nodes = level_graph.num_nodes
    adjacency = _build_weighted_adjacency(level_graph.edge_index, num_nodes)
    edge_lengths_cpu = level_graph.edge_lengths.to(device="cpu", dtype=torch.float32)
    edge_weights_cpu = (
        level_graph.edge_weights.to(device="cpu", dtype=torch.float32)
        if level_graph.edge_weights.numel() > 0
        else torch.ones((level_graph.edge_index.shape[1],), dtype=torch.float32)
    )
    selectable_nodes = _RandomNodeSet.from_star_masses(_star_masses(node_masses, adjacency))

    mapping = torch.full((num_nodes,), fill_value=-1, dtype=torch.long)
    node_types = [0 for _ in range(num_nodes)]
    dedicated_sun = [-1 for _ in range(num_nodes)]
    dedicated_sun_distance = [0.0 for _ in range(num_nodes)]
    lambda_values: list[list[float]] = [[] for _ in range(num_nodes)]
    neighbor_suns: list[list[int]] = [[] for _ in range(num_nodes)]
    moon_children: list[list[int]] = [[] for _ in range(num_nodes)]
    sun_to_coarse: dict[int, int] = {}

    while not selectable_nodes.empty():
        sun_node = selectable_nodes.get_random_node_with_highest_star_mass(
            rng,
            _SOLAR_RANDOM_TRIES,
        )
        coarse_node = len(sun_to_coarse)
        sun_to_coarse[sun_node] = coarse_node
        mapping[sun_node] = coarse_node
        node_types[sun_node] = _TYPE_SUN
        dedicated_sun[sun_node] = sun_node
        dedicated_sun_distance[sun_node] = 0.0

        planet_nodes: list[int] = []
        for planet_node, edge_id in adjacency[sun_node]:
            distance_to_sun = float(edge_lengths_cpu[edge_id].item())
            node_types[planet_node] = _TYPE_PLANET
            dedicated_sun[planet_node] = sun_node
            dedicated_sun_distance[planet_node] = distance_to_sun
            mapping[planet_node] = coarse_node
            planet_nodes.append(planet_node)

        for planet_node in planet_nodes:
            selectable_nodes.delete(planet_node)

        for planet_node in planet_nodes:
            for possible_moon, _ in adjacency[planet_node]:
                selectable_nodes.delete(possible_moon)

    for node in range(num_nodes):
        if node_types[node] != 0:
            continue

        nearest_neighbor = -1
        nearest_distance = math.inf
        for neighbor, edge_id in adjacency[node]:
            if node_types[neighbor] not in (_TYPE_PLANET, _TYPE_PLANET_WITH_MOONS):
                continue
            distance = float(edge_lengths_cpu[edge_id].item())
            if distance < nearest_distance:
                nearest_neighbor = neighbor
                nearest_distance = distance

        if nearest_neighbor < 0:
            for neighbor, edge_id in adjacency[node]:
                if dedicated_sun[neighbor] < 0:
                    continue
                distance = float(edge_lengths_cpu[edge_id].item())
                if distance < nearest_distance:
                    nearest_neighbor = neighbor
                    nearest_distance = distance

        if nearest_neighbor < 0:
            coarse_node = len(sun_to_coarse)
            sun_to_coarse[node] = coarse_node
            mapping[node] = coarse_node
            node_types[node] = _TYPE_SUN
            dedicated_sun[node] = node
            continue

        assigned_sun = dedicated_sun[nearest_neighbor]
        mapping[node] = sun_to_coarse[assigned_sun]
        node_types[node] = _TYPE_MOON
        dedicated_sun[node] = assigned_sun
        dedicated_sun_distance[node] = nearest_distance + dedicated_sun_distance[nearest_neighbor]
        node_types[nearest_neighbor] = _TYPE_PLANET_WITH_MOONS
        moon_children[nearest_neighbor].append(node)

    pm_nodes = [node for node in range(num_nodes) if node_types[node] == _TYPE_PLANET_WITH_MOONS]

    coarse_masses = torch.zeros((len(sun_to_coarse),), dtype=torch.long)
    for coarse_node in mapping.tolist():
        coarse_masses[coarse_node] += 1

    pair_sums: dict[tuple[int, int], float] = {}
    pair_counts: dict[tuple[int, int], int] = {}
    pair_weight_sums: dict[tuple[int, int], float] = {}
    edge_index_cpu = level_graph.edge_index.to(device="cpu", dtype=torch.long)
    for edge_id, (source, target) in enumerate(
        zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist())
    ):
        source_sun = dedicated_sun[source]
        target_sun = dedicated_sun[target]
        if source_sun == target_sun:
            continue

        coarse_source = sun_to_coarse[source_sun]
        coarse_target = sun_to_coarse[target_sun]
        pair = (min(coarse_source, coarse_target), max(coarse_source, coarse_target))
        edge_length = float(edge_lengths_cpu[edge_id].item())
        edge_weight = float(edge_weights_cpu[edge_id].item())
        new_length = dedicated_sun_distance[source] + edge_length + dedicated_sun_distance[target]
        pair_sums[pair] = pair_sums.get(pair, 0.0) + new_length
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        pair_weight_sums[pair] = pair_weight_sums.get(pair, 0.0) + edge_weight

        source_lambda = dedicated_sun_distance[source] / max(new_length, _MIN_DISTANCE)
        target_lambda = dedicated_sun_distance[target] / max(new_length, _MIN_DISTANCE)
        lambda_values[source].append(source_lambda)
        lambda_values[target].append(target_lambda)
        neighbor_suns[source].append(target_sun)
        neighbor_suns[target].append(source_sun)

    if pair_sums:
        ordered_pairs = sorted(pair_sums)
        coarse_edge_index = (
            torch.tensor(ordered_pairs, dtype=torch.long).transpose(0, 1).contiguous()
        )
        coarse_edge_lengths = torch.tensor(
            [pair_sums[pair] / pair_counts[pair] for pair in ordered_pairs],
            dtype=torch.float32,
        )
        coarse_edge_weights = torch.tensor(
            [pair_weight_sums[pair] for pair in ordered_pairs],
            dtype=torch.float32,
        )
    else:
        coarse_edge_index = torch.empty((2, 0), dtype=torch.long)
        coarse_edge_lengths = torch.empty((0,), dtype=torch.float32)
        coarse_edge_weights = torch.empty((0,), dtype=torch.float32)

    return (
        _HierarchyStep(
            mapping=mapping,
            node_types=node_types,
            dedicated_sun=dedicated_sun,
            dedicated_sun_distance=dedicated_sun_distance,
            pm_nodes=pm_nodes,
            moon_children=moon_children,
            lambda_values=lambda_values,
            neighbor_suns=neighbor_suns,
        ),
        _LevelGraph(
            edge_index=coarse_edge_index,
            edge_lengths=coarse_edge_lengths,
            num_nodes=len(sun_to_coarse),
            edge_weights=coarse_edge_weights,
        ),
        coarse_masses,
    )


def _build_hierarchy(
    edge_index: torch.Tensor,
    num_nodes: int,
    seed: int,
    edge_weights: Optional[torch.Tensor] = None,
) -> tuple[list[_LevelGraph], list[_HierarchyStep]]:
    """Build the FM^3 hierarchy with OGDF-style coarsening metadata.

    Parameters
    ----------
    edge_index : torch.Tensor
        Original edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    seed : int
        Random seed used for sun selection.
    edge_weights : torch.Tensor, optional
        Optional per-edge attraction weights with shape ``[E]``.

    Returns
    -------
    tuple[list[_LevelGraph], list[_HierarchyStep]]
        Hierarchy levels from fine to coarse and prolongation metadata for each
        fine-to-coarse transition.
    """
    base_edges, base_lengths, base_weights = _unique_edges_with_lengths(
        edge_index,
        num_nodes,
        edge_weights=edge_weights,
    )
    levels = [
        _LevelGraph(
            edge_index=base_edges,
            edge_lengths=base_lengths,
            num_nodes=num_nodes,
            edge_weights=base_weights,
        )
    ]
    steps: list[_HierarchyStep] = []
    current_nodes = num_nodes
    current_masses = torch.ones((num_nodes,), dtype=torch.long)
    bad_edge_counter = 0
    rng = random.Random(seed)

    while current_nodes > _COARSE_TARGET:
        if len(levels) > 1:
            previous_edge_count = int(levels[-2].edge_index.shape[1])
            current_edge_count = int(levels[-1].edge_index.shape[1])
            if current_edge_count > 0.8 * previous_edge_count:
                if bad_edge_counter < 5:
                    bad_edge_counter += 1
                else:
                    break

        step, coarse_level, coarse_masses = _coarsen_level(levels[-1], current_masses, rng)
        if coarse_level.num_nodes >= current_nodes:
            break

        steps.append(step)
        levels.append(coarse_level)
        current_nodes = coarse_level.num_nodes
        current_masses = coarse_masses

    return levels, steps


def _bounding_box(positions: torch.Tensor) -> tuple[float, float, float, float]:
    """Compute an axis-aligned bounding box around node positions.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    tuple[float, float, float, float]
        ``(x_min, x_max, y_min, y_max)`` bounds.
    """
    if positions.shape[0] == 0:
        return (-1.0, 1.0, -1.0, 1.0)

    x_min = float(positions[:, 0].min().item())
    x_max = float(positions[:, 0].max().item())
    y_min = float(positions[:, 1].min().item())
    y_max = float(positions[:, 1].max().item())
    padding = max(x_max - x_min, y_max - y_min, 1.0) * 0.05
    return (x_min - padding, x_max + padding, y_min - padding, y_max + padding)


def _build_quadtree(
    positions: torch.Tensor,
    indices: list[int],
    bounds: tuple[float, float, float, float],
    depth: int,
) -> _QuadCell:
    """Recursively build a quad-tree for Barnes-Hut repulsion.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    indices : list[int]
        Node indices inside the current bounds.
    bounds : tuple[float, float, float, float]
        ``(x_min, x_max, y_min, y_max)`` for the current cell.
    depth : int
        Remaining recursion depth.

    Returns
    -------
    _QuadCell
        Root of the constructed subtree.
    """
    x_min, x_max, y_min, y_max = bounds
    cell = _QuadCell(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, indices=indices)
    if not indices:
        return cell

    points = positions[indices]
    cell.mass = float(len(indices))
    cell.center_of_mass = points.mean(dim=0)
    if len(indices) <= 1 or depth <= 0:
        return cell

    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    quadrants = [
        (x_min, x_mid, y_min, y_mid),
        (x_mid, x_max, y_min, y_mid),
        (x_min, x_mid, y_mid, y_max),
        (x_mid, x_max, y_mid, y_max),
    ]
    buckets: list[list[int]] = [[] for _ in range(4)]
    for index in indices:
        x = float(positions[index, 0].item())
        y = float(positions[index, 1].item())
        quadrant = 0
        if x >= x_mid:
            quadrant += 1
        if y >= y_mid:
            quadrant += 2
        buckets[quadrant].append(index)

    for quadrant_bounds, bucket in zip(quadrants, buckets):
        if bucket:
            cell.children.append(_build_quadtree(positions, bucket, quadrant_bounds, depth - 1))

    return cell


def _repulsion_from_cell(
    node_index: int,
    point: torch.Tensor,
    cell: _QuadCell,
    theta: float,
    ideal_length: float,
) -> torch.Tensor:
    """Approximate the repulsion contribution of one quad-tree cell.

    Parameters
    ----------
    node_index : int
        Node for which the force is being evaluated.
    point : torch.Tensor
        Node position with shape ``[2]``.
    cell : _QuadCell
        Current quad-tree cell.
    theta : float
        Barnes-Hut opening angle threshold.
    ideal_length : float
        FR ideal edge length ``k`` for the current level.

    Returns
    -------
    torch.Tensor
        Repulsion vector contributed by the cell.
    """
    if cell.mass == 0.0:
        return torch.zeros(2, dtype=point.dtype, device=point.device)
    if len(cell.indices) == 1 and cell.indices[0] == node_index:
        return torch.zeros(2, dtype=point.dtype, device=point.device)

    delta = point - cell.center_of_mass.to(device=point.device)
    distance = torch.linalg.norm(delta).clamp(min=_MIN_DISTANCE)
    width = max(cell.x_max - cell.x_min, cell.y_max - cell.y_min)
    if not cell.children or width / float(distance.item()) < theta:
        del ideal_length
        return delta * (cell.mass / distance.square())

    total = torch.zeros(2, dtype=point.dtype, device=point.device)
    for child in cell.children:
        total = total + _repulsion_from_cell(
            node_index,
            point,
            child,
            theta,
            ideal_length,
        )
    return total


def _exact_repulsion(positions: torch.Tensor) -> torch.Tensor:
    """Compute exact all-pairs repulsion matching OGDF's f_rep_u_on_v.

    OGDF formula: f_rep_scalar(d) = 1/d, applied as (v-u)/||v-u||^2.
    No ideal_length in the formula, so the force stays a pure 1/d^2 field.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Repulsive force per node with shape ``[N, 2]``.
    """
    n = positions.shape[0]
    if n <= 1:
        return torch.zeros_like(positions)
    delta = positions.unsqueeze(1) - positions.unsqueeze(0)
    dist = torch.cdist(positions, positions).clamp(min=_MIN_DISTANCE)
    factor = 1.0 / (dist * dist)
    factor.fill_diagonal_(0.0)
    return (delta * factor.unsqueeze(2)).sum(dim=1)


def _barnes_hut_repulsion(
    positions: torch.Tensor,
    theta: float,
    ideal_length: float,
) -> torch.Tensor:
    """Compute repulsive forces using a Barnes-Hut quad-tree.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    theta : float
        Barnes-Hut opening angle threshold.
    ideal_length : float
        FR ideal edge length ``k`` for the current level.

    Returns
    -------
    torch.Tensor
        Repulsive force per node.
    """
    root = _build_quadtree(
        positions,
        list(range(int(positions.shape[0]))),
        _bounding_box(positions),
        _MAX_TREE_DEPTH,
    )
    forces = torch.zeros_like(positions)
    for node_index in range(int(positions.shape[0])):
        forces[node_index] = _repulsion_from_cell(
            node_index,
            positions[node_index],
            root,
            theta,
            ideal_length,
        )
    return forces


def _attractive_force(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    ideal_length: float,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute exact attractive forces along graph edges.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Unique undirected edges with shape ``[2, E]``.
    ideal_length : float
        FR ideal edge length ``k`` for the current level.
    edge_weights : torch.Tensor, optional
        Optional per-edge attraction weights with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Attractive force per node.
    """
    forces = torch.zeros_like(positions)
    if edge_index.numel() == 0:
        return forces

    src = edge_index[0].to(device=positions.device, dtype=torch.long)
    dst = edge_index[1].to(device=positions.device, dtype=torch.long)
    delta = positions[dst] - positions[src]
    distances = torch.linalg.norm(delta, dim=1).clamp(min=_MIN_DISTANCE)
    denominator = max(ideal_length**3, _MIN_DISTANCE)
    edge_force = delta * (distances / denominator).unsqueeze(1)
    if edge_weights is not None:
        edge_force = edge_force * edge_weights.to(
            device=positions.device,
            dtype=positions.dtype,
        ).unsqueeze(1)
    forces.index_add_(0, src, edge_force)
    forces.index_add_(0, dst, -edge_force)
    return forces


def _attractive_force_with_lengths(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    edge_lengths: torch.Tensor,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute OGDF-style attractive forces using per-edge desired lengths.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Unique undirected edges with shape ``[2, E]``.
    edge_lengths : torch.Tensor
        Desired edge lengths with shape ``[E]``.
    edge_weights : torch.Tensor, optional
        Optional per-edge attraction weights with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Attractive force per node with shape ``[N, 2]``.
    """
    forces = torch.zeros_like(positions)
    if edge_index.numel() == 0:
        return forces

    src = edge_index[0].to(device=positions.device, dtype=torch.long)
    dst = edge_index[1].to(device=positions.device, dtype=torch.long)
    desired_lengths = edge_lengths.to(device=positions.device, dtype=positions.dtype).clamp(
        min=_MIN_DISTANCE
    )
    delta = positions[dst] - positions[src]
    distances = torch.linalg.norm(delta, dim=1).clamp(min=_MIN_DISTANCE)
    edge_force = delta * (distances / desired_lengths.pow(3)).unsqueeze(1)
    if edge_weights is not None:
        edge_force = edge_force * edge_weights.to(
            device=positions.device,
            dtype=positions.dtype,
        ).unsqueeze(1)
    forces.index_add_(0, src, edge_force)
    forces.index_add_(0, dst, -edge_force)
    return forces


def _refine_level(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    steps: int,
    theta: float,
    area: float,
    edge_weights: Optional[torch.Tensor] = None,
    cooling_factor: float = _COOLING_FACTOR,
) -> torch.Tensor:
    """Run Barnes-Hut force refinement on one hierarchy level.

    Parameters
    ----------
    positions : torch.Tensor
        Initial coordinates with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Unique undirected edges with shape ``[2, E]``.
    steps : int
        Number of refinement iterations.
    theta : float
        Barnes-Hut opening angle threshold.
    area : float
        Target drawing area for the current level.
    edge_weights : torch.Tensor, optional
        Optional per-edge attraction weights with shape ``[E]``.
    cooling_factor : float, default=0.99
        Multiplicative temperature decay applied after each refinement step.

    Returns
    -------
    torch.Tensor
        Refined positions.
    """
    refined = positions.clone()
    if steps <= 0:
        return refined

    ideal_length = _fr_ideal_length(area, int(refined.shape[0]))
    temperature = ideal_length
    use_exact = int(refined.shape[0]) <= 500
    for _ in range(steps):
        repulsive = (
            _exact_repulsion(refined)
            if use_exact
            else _barnes_hut_repulsion(refined, theta, ideal_length)
        )
        attractive = _attractive_force(
            refined,
            edge_index,
            ideal_length,
            edge_weights=edge_weights,
        )
        displacement = repulsive + attractive
        norm = torch.linalg.norm(displacement, dim=1, keepdim=True).clamp(min=_MIN_DISTANCE)
        limited_step = torch.minimum(norm, torch.full_like(norm, temperature))
        refined = refined + (displacement / norm) * limited_step
        temperature = max(temperature * cooling_factor, _MIN_DISTANCE)
    return refined


def _refine_level_with_edge_lengths(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    edge_lengths: torch.Tensor,
    steps: int,
    theta: float,
    area: float,
    edge_weights: Optional[torch.Tensor] = None,
    cooling_factor: float = _COOLING_FACTOR,
) -> torch.Tensor:
    """Run Barnes-Hut force refinement using individual desired edge lengths.

    Parameters
    ----------
    positions : torch.Tensor
        Initial coordinates with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Unique undirected edges with shape ``[2, E]``.
    edge_lengths : torch.Tensor
        Desired edge lengths with shape ``[E]``.
    steps : int
        Number of refinement iterations.
    theta : float
        Barnes-Hut opening angle threshold.
    area : float
        Target drawing area for the current level.
    edge_weights : torch.Tensor, optional
        Optional per-edge attraction weights with shape ``[E]``.
    cooling_factor : float, default=0.99
        Multiplicative temperature decay applied after each refinement step.

    Returns
    -------
    torch.Tensor
        Refined positions.
    """
    refined = positions.clone()
    if steps <= 0:
        return refined

    ideal_length = _fr_ideal_length(area, int(refined.shape[0]))
    temperature = ideal_length
    use_exact = int(refined.shape[0]) <= 500
    for _ in range(steps):
        repulsive = (
            _exact_repulsion(refined)
            if use_exact
            else _barnes_hut_repulsion(refined, theta, ideal_length)
        )
        attractive = _attractive_force_with_lengths(
            refined,
            edge_index,
            edge_lengths,
            edge_weights=edge_weights,
        )
        displacement = repulsive + attractive
        norm = torch.linalg.norm(displacement, dim=1, keepdim=True).clamp(min=_MIN_DISTANCE)
        limited_step = torch.minimum(norm, torch.full_like(norm, temperature))
        refined = refined + (displacement / norm) * limited_step
        temperature = max(temperature * cooling_factor, _MIN_DISTANCE)
    return refined


def _create_random_position(
    center: torch.Tensor,
    radius: float,
    angle_1: float,
    angle_2: float,
    rng: random.Random,
) -> torch.Tensor:
    """Create a random point on a circular sector around ``center``.

    Parameters
    ----------
    center : torch.Tensor
        Center point with shape ``[2]``.
    radius : float
        Sampling radius.
    angle_1 : float
        Lower angle bound in radians.
    angle_2 : float
        Upper angle bound in radians.
    rng : random.Random
        Pseudorandom generator.

    Returns
    -------
    torch.Tensor
        Sampled point with shape ``[2]``.
    """
    random_angle = angle_1 + (angle_2 - angle_1) * rng.random()
    offset = torch.tensor(
        [math.cos(random_angle) * radius, math.sin(random_angle) * radius],
        dtype=center.dtype,
        device=center.device,
    )
    return center + offset


def _waggled_inbetween_position(
    source: torch.Tensor,
    target: torch.Tensor,
    lambda_value: float,
    rng: random.Random,
) -> torch.Tensor:
    """Place a node between two endpoints with OGDF's 5% waggle.

    Parameters
    ----------
    source : torch.Tensor
        Source point with shape ``[2]``.
    target : torch.Tensor
        Target point with shape ``[2]``.
    lambda_value : float
        Interpolation weight from ``source`` toward ``target``.
    rng : random.Random
        Pseudorandom generator.

    Returns
    -------
    torch.Tensor
        Waggled point with shape ``[2]``.
    """
    inbetween = source + lambda_value * (target - source)
    radius = _WAGGLE_FACTOR * float(torch.linalg.norm(target - source).item())
    return _create_random_position(
        inbetween,
        radius * rng.random(),
        0.0,
        2.0 * math.pi,
        rng,
    )


def _barycenter_position(points: list[torch.Tensor]) -> torch.Tensor:
    """Return the barycenter of a non-empty point list.

    Parameters
    ----------
    points : list[torch.Tensor]
        Point list where each tensor has shape ``[2]``.

    Returns
    -------
    torch.Tensor
        Arithmetic mean with shape ``[2]``.
    """
    return torch.stack(points, dim=0).mean(dim=0)


def _prolong_positions(
    coarse_positions: torch.Tensor,
    step: _HierarchyStep,
    rng: random.Random,
) -> torch.Tensor:
    """Prolong one coarse FM^3 level using OGDF-style lambda interpolation.

    Parameters
    ----------
    coarse_positions : torch.Tensor
        Coarse coordinates with shape ``[N_coarse, 2]``.
    step : _HierarchyStep
        Metadata for the current fine-to-coarse transition.
    rng : random.Random
        Pseudorandom generator used for waggle and fallback placement.

    Returns
    -------
    torch.Tensor
        Fine-level coordinates with shape ``[N_fine, 2]``.
    """
    fine_positions = coarse_positions[step.mapping].clone()

    for node, node_type in enumerate(step.node_types):
        if node_type != _TYPE_SUN:
            continue
        fine_positions[node] = coarse_positions[int(step.mapping[node].item())]

    for node, node_type in enumerate(step.node_types):
        if node_type not in (_TYPE_PLANET, _TYPE_MOON):
            continue

        sun_node = step.dedicated_sun[node]
        sun_position = fine_positions[sun_node]
        dedicated_distance = step.dedicated_sun_distance[node]
        candidates = [
            _waggled_inbetween_position(
                sun_position,
                fine_positions[neighbor_sun],
                lambda_value,
                rng,
            )
            for lambda_value, neighbor_sun in zip(
                step.lambda_values[node],
                step.neighbor_suns[node],
            )
        ]
        if not candidates:
            candidates.append(
                _create_random_position(
                    sun_position,
                    dedicated_distance,
                    0.0,
                    2.0 * math.pi,
                    rng,
                )
            )
        fine_positions[node] = _barycenter_position(candidates)

    for node in step.pm_nodes:
        sun_node = step.dedicated_sun[node]
        sun_position = fine_positions[sun_node]
        sun_distance = step.dedicated_sun_distance[node]
        candidates = [
            _waggled_inbetween_position(
                sun_position,
                fine_positions[moon_node],
                sun_distance / max(step.dedicated_sun_distance[moon_node], _MIN_DISTANCE),
                rng,
            )
            for moon_node in step.moon_children[node]
        ]
        candidates.extend(
            _waggled_inbetween_position(
                sun_position,
                fine_positions[neighbor_sun],
                lambda_value,
                rng,
            )
            for lambda_value, neighbor_sun in zip(
                step.lambda_values[node],
                step.neighbor_suns[node],
            )
        )
        if not candidates:
            candidates.append(
                _create_random_position(
                    sun_position,
                    sun_distance,
                    0.0,
                    2.0 * math.pi,
                    rng,
                )
            )
        fine_positions[node] = _barycenter_position(candidates)

    return fine_positions


@register_op
class _InitializeFMMMState(Op):
    """Build the FM^3 multilevel hierarchy and store it in extras.

    Computes the solar-system coarsening hierarchy, extent, refinement area,
    and level budget, storing everything the downstream ops need.
    """

    name: ClassVar[str] = "fmmm_initialize_state"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    _steps: int

    def __init__(self, steps: int = 100) -> None:
        """Store the total refinement budget.

        Parameters
        ----------
        steps : int, default=100
            Total refinement budget across hierarchy levels.

        Returns
        -------
        None
            Configuration stored for use by ``apply()``.
        """
        self._steps = steps

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build the hierarchy and store FM^3 metadata in ``state.extras``.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state receiving FM^3 extras.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with FM^3 hierarchy data stored in ``extras``.
        """
        del ctx

        extent = _layout_extent(problem.num_nodes, problem.node_sizes)
        refinement_area = (2.0 * extent) ** 2
        levels, hierarchy_steps = _build_hierarchy(
            problem.edge_index,
            problem.num_nodes,
            seed=problem.seed,
            edge_weights=problem.edge_weights,
        )

        state.extras["fmmm_levels"] = levels
        state.extras["fmmm_hierarchy_steps"] = hierarchy_steps
        state.extras["fmmm_extent"] = extent
        state.extras["fmmm_refinement_area"] = refinement_area
        state.extras["fmmm_level_budget"] = max(10, self._steps // max(len(levels), 1))
        state.extras["fmmm_steps"] = self._steps
        return state


@register_op
class _InitializeCoarsestLevel(Op):
    """Initialize the coarsest hierarchy level using classic FR.

    Runs ``layout_fr`` on the coarsest level graph exactly as the classic
    implementation does.
    """

    name: ClassVar[str] = "fmmm_initialize_coarsest"
    category: ClassVar[OpCategory] = OpCategory.INIT
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Seed positions from FR on the coarsest hierarchy level.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs containing ``seed``.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with ``state.pos`` populated from coarse FR layout.
        """
        del ctx

        levels = state.extras["fmmm_levels"]
        steps = state.extras["fmmm_steps"]
        coarsest_level = levels[-1]
        coarsest_nodes = coarsest_level.num_nodes

        coarse_positions = layout_fr_pipeline(
            coarsest_level.edge_index,
            coarsest_nodes,
            node_sizes=None,
            steps=max(50, steps),
            seed=problem.seed,
            edge_weights=coarsest_level.edge_weights,
        )
        state.pos = coarse_positions.to(dtype=torch.float32)
        return state


@register_op
class _RefineCoarsestLevel(Op):
    """Refine the coarsest level if the hierarchy has prolongation steps.

    Uses per-edge desired lengths for attraction forces with Barnes-Hut
    repulsion, exactly matching the classic implementation's coarsest-level
    refinement pass.
    """

    name: ClassVar[str] = "fmmm_refine_coarsest"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("pos", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Refine the coarsest level with edge-length-aware forces.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with coarse positions.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with refined coarsest-level positions.
        """
        del ctx

        hierarchy_steps = state.extras["fmmm_hierarchy_steps"]
        if not hierarchy_steps:
            return state

        levels = state.extras["fmmm_levels"]
        coarsest_level = levels[-1]
        level_budget = state.extras["fmmm_level_budget"]
        refinement_area = state.extras["fmmm_refinement_area"]

        state.pos = _refine_level_with_edge_lengths(
            state.pos,
            coarsest_level.edge_index,
            coarsest_level.edge_lengths,
            level_budget,
            theta=1.0,
            area=refinement_area,
            edge_weights=coarsest_level.edge_weights,
        )
        return state


@register_op
class _UncoarsenLoop(Op):
    """Prolong and refine through the hierarchy from coarse to fine.

    Iterates from the coarsest prolongation step back to the finest level,
    prolonging positions using OGDF-style lambda interpolation and then
    refining each level with Barnes-Hut forces.
    """

    name: ClassVar[str] = "fmmm_uncoarsen_loop"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("pos", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Prolong and refine through the hierarchy levels.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs containing ``seed``.
        state : SolveState
            Mutable solve state with coarsest-level positions.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with fine-level positions after uncoarsening.
        """
        del ctx

        hierarchy_steps = state.extras["fmmm_hierarchy_steps"]
        if not hierarchy_steps:
            return state

        levels = state.extras["fmmm_levels"]
        level_budget = state.extras["fmmm_level_budget"]
        refinement_area = state.extras["fmmm_refinement_area"]
        rng = random.Random(problem.seed)

        positions = state.pos
        for level in range(len(hierarchy_steps) - 1, -1, -1):
            fine_positions = _prolong_positions(positions, hierarchy_steps[level], rng)
            positions = _refine_level_with_edge_lengths(
                fine_positions,
                levels[level].edge_index,
                levels[level].edge_lengths,
                level_budget,
                theta=1.0,
                area=refinement_area,
                edge_weights=levels[level].edge_weights,
            )

        state.pos = positions
        return state


@register_op
class _SingleLevelFallback(Op):
    """Refine the single level when no coarsening hierarchy was built.

    When the graph is already small enough that coarsening produces no
    hierarchy steps, this op runs a standard refinement pass on the
    original level using uniform ideal edge length.
    """

    name: ClassVar[str] = "fmmm_single_level_fallback"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("pos", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Refine with uniform-length forces if no hierarchy exists.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with FR-initialized positions.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with refined positions.
        """
        del ctx

        hierarchy_steps = state.extras["fmmm_hierarchy_steps"]
        if hierarchy_steps:
            return state

        levels = state.extras["fmmm_levels"]
        level_budget = state.extras["fmmm_level_budget"]
        refinement_area = state.extras["fmmm_refinement_area"]

        state.pos = _refine_level(
            state.pos,
            levels[0].edge_index,
            level_budget,
            theta=1.0,
            area=refinement_area,
            edge_weights=levels[0].edge_weights,
        )
        return state


@register_op
class _FinalizeFMMMPositions(Op):
    """Apply classic FM^3's final normalization, centering, and scaling."""

    name: ClassVar[str] = "fmmm_finalize_positions"
    category: ClassVar[OpCategory] = OpCategory.POSTPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("pos", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Center, normalize, and cast positions like classic FM^3.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state containing refined positions.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with final output positions.

        Raises
        ------
        ValueError
            If ``state.pos`` is missing.
        """
        del ctx

        if state.pos is None:
            raise ValueError("_FinalizeFMMMPositions requires state.pos to be set.")

        device = _layout_device(problem.edge_index, problem.node_sizes)
        extent = state.extras["fmmm_extent"]
        state.pos = _normalize_positions(state.pos.to(device), extent).to(
            dtype=torch.float32, device=device
        )
        return state
