"""Composable Large Graph Layout (LGL) operations.

These ops decompose the classic LGL implementation into reusable pipeline
steps so that ``dagua.layout.ops.pipelines.lgl`` can build its workflow from
registered operations only.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import ClassVar, List, Optional, Tuple

import torch

from dagua.layout.ops.base import Op
from dagua.layout.ops.graph_utils import layout_device
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_LGL_MIN_DISTANCE = 1.0e-12
_LGL_CONVERGENCE_EPSILON = 1.0e-5


@dataclass(frozen=True)
class LGLPrepareStateConfig:
    """Configuration for :class:`LGLPrepareState`.

    Parameters
    ----------
    maxiter : int, default=150
        Maximum number of refinement iterations for each growth shell.
    maxdelta : float, optional
        Initial temperature bound for refinement.
    area : float, optional
        Drawing area; defaults to ``num_nodes ** 2``.
    coolexp : float, default=1.5
        Temperature cooling exponent.
    repulserad : float, optional
        Repulsion radius used by the LGL spring cutoff rule.
    cellsize : float, optional
        Sparse grid cell size for repulsion.
    root : int, optional
        Optional root vertex used by BFS shell growth.
    """

    maxiter: int = 150
    maxdelta: Optional[float] = None
    area: Optional[float] = None
    coolexp: float = 1.5
    repulserad: Optional[float] = None
    cellsize: Optional[float] = None
    root: Optional[int] = None


@register_op
@dataclass(frozen=True)
class LGLPrepareState(Op):
    """Populate LGL metadata and graph structures required by later ops."""

    config: LGLPrepareStateConfig = field(default_factory=LGLPrepareStateConfig)

    name: ClassVar[str] = "lgl_prepare_state"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Resolve defaults and build LGL graph-structure caches.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable problem inputs.
        state : SolveState
            Mutable working state receiving LGL metadata.
        ctx : RuntimeContext
            Runtime context (unused).

        Returns
        -------
        SolveState
            State with populated ``state.extras`` fields:
            ``lgl_maxiter``, ``lgl_area``, ``lgl_adjacency``,
            ``lgl_spring_edges``, ``lgl_root``, and related scalar settings.
        """
        del ctx

        num_nodes = problem.num_nodes
        resolved_maxdelta = (
            float(num_nodes) if self.config.maxdelta is None else self.config.maxdelta
        )
        resolved_area = (
            float(num_nodes * num_nodes) if self.config.area is None else self.config.area
        )
        resolved_repulserad = (
            resolved_area * float(max(num_nodes, 1))
            if self.config.repulserad is None
            else self.config.repulserad
        )
        resolved_cellsize = (
            resolved_area**0.25 if self.config.cellsize is None else self.config.cellsize
        )

        adjacency_sets = [set() for _ in range(num_nodes)]
        spring_edges: list[tuple[int, int]] = []
        spring_edge_weights: Optional[list[float]] = None
        if problem.edge_index.numel() > 0:
            edge_index_cpu = problem.edge_index.to(device="cpu", dtype=torch.long)
            source_nodes = edge_index_cpu[0].tolist()
            target_nodes = edge_index_cpu[1].tolist()
            weight_list: Optional[list[float]] = None
            if problem.edge_weights is not None:
                spring_edge_weights = []
                weight_list = (
                    problem.edge_weights.detach().to(device="cpu", dtype=torch.float64).tolist()
                )

            for edge_idx, (source, target) in enumerate(zip(source_nodes, target_nodes)):
                if source == target:
                    continue
                lower = min(source, target)
                upper = max(source, target)
                adjacency_sets[lower].add(upper)
                adjacency_sets[upper].add(lower)
                # Keep edge multiplicity to match classic LGL behavior.
                spring_edges.append((lower, upper))
                if weight_list is not None:
                    spring_edge_weights.append(float(weight_list[edge_idx]))

        adjacency = [sorted(neighbors) for neighbors in adjacency_sets]

        rng = random.Random(problem.seed)
        root_node = rng.randrange(num_nodes) if self.config.root is None else self.config.root
        frk = math.sqrt(resolved_area / float(max(num_nodes, 1)))

        state.extras["lgl_maxiter"] = self.config.maxiter
        state.extras["lgl_maxdelta"] = resolved_maxdelta
        state.extras["lgl_area"] = resolved_area
        state.extras["lgl_coolexp"] = self.config.coolexp
        state.extras["lgl_repulserad"] = resolved_repulserad
        state.extras["lgl_cellsize"] = resolved_cellsize
        state.extras["lgl_root"] = root_node
        state.extras["lgl_frk"] = frk
        state.extras["lgl_adjacency"] = adjacency
        state.extras["lgl_spring_edges"] = spring_edges
        state.extras["lgl_spring_edge_weights"] = spring_edge_weights
        return state


@register_op
class LGLInitializePositions(Op):
    """Initialize layout positions with the classic LGL random seed behavior."""

    name: ClassVar[str] = "lgl_initialize_positions"
    category: ClassVar[OpCategory] = OpCategory.INIT
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Seed ``state.pos`` as a random uniform square cloud.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable problem inputs.
        state : SolveState
            Mutable state storing layout coordinates.
        ctx : RuntimeContext
            Runtime context (unused).

        Returns
        -------
        SolveState
            Updated state with initial ``state.pos`` and root fixed at origin.
        """
        del ctx

        area = float(state.extras["lgl_area"])
        radius = math.sqrt(area / math.pi)
        rng = random.Random(problem.seed)
        positions = torch.tensor(
            [
                [rng.uniform(-radius, radius), rng.uniform(-radius, radius)]
                for _ in range(problem.num_nodes)
            ],
            dtype=torch.float64,
        )
        positions[int(state.extras["lgl_root"])] = 0.0
        state.pos = positions
        return state


@register_op
class LGLLayeredRefinement(Op):
    """Run BFS shell growth and local FR-style relaxation on each layer."""

    name: ClassVar[str] = "lgl_shell_growth_solve"
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
        """Place shells in BFS order and refine with active-node FR updates.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable problem inputs.
        state : SolveState
            Mutable state with initialization and LGL metadata.
        ctx : RuntimeContext
            Runtime context (unused).

        Returns
        -------
        SolveState
            Updated positions after all shell-placement iterations.
        """
        del ctx

        if state.pos is None:
            raise ValueError("LGLLayeredRefinement requires state.pos to be set.")

        num_nodes = problem.num_nodes
        positions = state.pos
        root_node = int(state.extras["lgl_root"])
        adjacency = state.extras["lgl_adjacency"]
        spring_edges = state.extras["lgl_spring_edges"]
        spring_edge_weights = state.extras["lgl_spring_edge_weights"]
        maxiter = int(state.extras["lgl_maxiter"])
        maxdelta = float(state.extras["lgl_maxdelta"])
        coolexp = float(state.extras["lgl_coolexp"])
        frk = float(state.extras["lgl_frk"])
        repulserad = float(state.extras["lgl_repulserad"])
        cellsize = float(state.extras["lgl_cellsize"])
        area = float(state.extras["lgl_area"])
        radius = math.sqrt(area / math.pi)

        layers: list[list[int]] = []
        parents: list[int] = [-1] * num_nodes
        distance: list[int] = [-1] * num_nodes
        bfs_queue: deque[int] = deque([root_node])
        parents[root_node] = root_node
        distance[root_node] = 0
        while bfs_queue:
            node = bfs_queue.popleft()
            depth = distance[node]
            while len(layers) <= depth:
                layers.append([])
            layers[depth].append(node)
            for neighbor in adjacency[node]:
                if distance[neighbor] != -1:
                    continue
                parents[neighbor] = node
                distance[neighbor] = depth + 1
                bfs_queue.append(neighbor)

        if not layers:
            state.pos = positions
            return state

        placed = torch.zeros(num_nodes, dtype=torch.bool)
        placed[root_node] = True

        active_edges: list[tuple[int, int]] = []
        active_edge_weights: Optional[list[float]]
        if spring_edge_weights is None:
            active_edge_weights = None
        else:
            active_edge_weights = []

        edge_active = [False] * len(spring_edges)
        incident_edge_indices: List[List[int]] = [[] for _ in range(num_nodes)]
        for edge_idx, (source, target) in enumerate(spring_edges):
            incident_edge_indices[source].append(edge_idx)
            incident_edge_indices[target].append(edge_idx)

        num_terms = max(len(layers) - 1, 1)
        shell_scale = radius / (
            sum(1.0 / float(index) for index in range(1, num_terms + 1)) if num_terms > 0 else 1.0
        )

        rng = random.Random(problem.seed)
        if bool(state.extras.get("lgl_root_was_random", True)):
            rng.randrange(num_nodes)

        for layer_index in range(len(layers) - 1):
            current_layer = layers[layer_index]
            next_layer = layers[layer_index + 1]
            if not next_layer:
                continue

            active_node_tensor = torch.nonzero(placed, as_tuple=False).view(-1)
            if active_node_tensor.numel() > 0:
                center_of_mass = positions[active_node_tensor].mean(dim=0)
                center_norm = float(torch.linalg.norm(center_of_mass).item())
                center_direction = (
                    center_of_mass / center_norm
                    if center_norm > _LGL_MIN_DISTANCE
                    else torch.zeros_like(center_of_mass)
                )
            else:
                center_direction = torch.zeros(2, dtype=torch.float64)

            next_depth = layer_index + 1
            total_layer_children = len(next_layer)
            layer_child_index = 0

            for parent in current_layer:
                children = [node for node in next_layer if parents[node] == parent]
                if not children:
                    continue

                if parent == root_node:
                    parent_direction = torch.zeros(2, dtype=torch.float64)
                else:
                    parent_of_parent = parents[parent]
                    parent_delta = positions[parent] - positions[parent_of_parent]
                    parent_norm = float(torch.linalg.norm(parent_delta).item())
                    parent_direction = (
                        parent_delta / parent_norm
                        if parent_norm > _LGL_MIN_DISTANCE
                        else torch.zeros_like(parent_delta)
                    )

                anchor = positions[parent] + center_direction + parent_direction
                for child in children:
                    if next_depth == 1:
                        angle = (2.0 * math.pi * layer_child_index) / float(
                            max(total_layer_children, 1)
                        )
                        direction = torch.tensor(
                            [math.cos(angle), math.sin(angle)],
                            dtype=torch.float64,
                        )
                    else:
                        direction = torch.tensor(
                            [rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)],
                            dtype=torch.float64,
                        )
                        direction_norm = float(torch.linalg.norm(direction).item())
                        if direction_norm > _LGL_MIN_DISTANCE:
                            direction = direction / direction_norm
                        else:
                            direction = torch.tensor([1.0, 0.0], dtype=torch.float64)

                    offset = direction * (shell_scale / float(max(next_depth, 1)))
                    positions[child] = anchor + offset
                    placed[child] = True
                    layer_child_index += 1

                    for edge_idx in incident_edge_indices[child]:
                        if edge_active[edge_idx]:
                            continue
                        source, target = spring_edges[edge_idx]
                        other = target if source == child else source
                        if not bool(placed[other].item()):
                            continue
                        edge_active[edge_idx] = True
                        active_edges.append((source, target))
                        if active_edge_weights is not None:
                            active_edge_weights.append(float(spring_edge_weights[edge_idx]))

            refinement_nodes = torch.nonzero(placed, as_tuple=False).view(-1).tolist()
            if not refinement_nodes or maxiter == 0:
                continue

            node_count = positions.shape[0]
            weight_tensor = (
                torch.tensor(active_edge_weights, dtype=torch.float64)
                if active_edge_weights is not None
                else None
            )

            for iteration in range(maxiter):
                temperature = maxdelta * (((maxiter - iteration) / float(maxiter)) ** coolexp)
                forces = torch.zeros((node_count, 2), dtype=torch.float64)
                maxchange = 0.0

                if active_edges:
                    source = torch.tensor([edge[0] for edge in active_edges], dtype=torch.long)
                    target = torch.tensor([edge[1] for edge in active_edges], dtype=torch.long)
                    delta = positions[source] - positions[target]
                    distance_matrix = torch.linalg.norm(delta, dim=1)
                    mask = distance_matrix > _LGL_MIN_DISTANCE
                    if bool(mask.any().item()):
                        masked_distance = distance_matrix[mask]
                        direction = delta[mask] / masked_distance.unsqueeze(1)
                        magnitude = masked_distance.square() / max(frk, _LGL_MIN_DISTANCE)
                        if weight_tensor is not None:
                            magnitude = magnitude * weight_tensor[mask]
                        contribution = direction * magnitude.unsqueeze(1)
                        forces.index_add_(0, source[mask], -contribution)
                        forces.index_add_(0, target[mask], contribution)

                buckets: dict[tuple[int, int], list[int]] = {}
                safe_cell_size = max(cellsize, _LGL_MIN_DISTANCE)
                for node in refinement_nodes:
                    x_value = float(positions[node, 0].item())
                    y_value = float(positions[node, 1].item())
                    key = (
                        int(math.floor(x_value / safe_cell_size)),
                        int(math.floor(y_value / safe_cell_size)),
                    )
                    buckets.setdefault(key, []).append(node)

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
                                for left_index in range(len(nodes_here)):
                                    for right_index in range(left_index + 1, len(nodes_here)):
                                        left = nodes_here[left_index]
                                        right = nodes_here[right_index]
                                        delta = positions[left] - positions[right]
                                        distance_value = float(torch.linalg.norm(delta).item())
                                        if distance_value >= cellsize:
                                            continue
                                        safe_distance = max(distance_value, _LGL_MIN_DISTANCE)
                                        direction = delta / safe_distance
                                        magnitude = (frk * frk) * (
                                            (1.0 / safe_distance)
                                            - ((safe_distance * safe_distance) / repulserad)
                                        )
                                        contribution = direction * magnitude
                                        forces[left] += contribution
                                        forces[right] -= contribution
                            else:
                                for left in nodes_here:
                                    for right in nodes_there:
                                        delta = positions[left] - positions[right]
                                        distance_value = float(torch.linalg.norm(delta).item())
                                        if distance_value >= cellsize:
                                            continue
                                        safe_distance = max(distance_value, _LGL_MIN_DISTANCE)
                                        direction = delta / safe_distance
                                        magnitude = (frk * frk) * (
                                            (1.0 / safe_distance)
                                            - ((safe_distance * safe_distance) / repulserad)
                                        )
                                        contribution = direction * magnitude
                                        forces[left] += contribution
                                        forces[right] -= contribution

                for node in refinement_nodes:
                    movement = forces[node]
                    magnitude = float(torch.linalg.norm(movement).item())
                    if magnitude > temperature and magnitude > _LGL_MIN_DISTANCE:
                        movement = movement * (temperature / magnitude)
                    positions[node] += movement
                    maxchange = max(
                        maxchange,
                        abs(float(movement[0].item())),
                        abs(float(movement[1].item())),
                    )

                if maxchange < _LGL_CONVERGENCE_EPSILON:
                    break

        state.pos = positions
        return state


@register_op
class LGLFinalizePositions(Op):
    """Cast positions to the classic LGL output dtype and output device."""

    name: ClassVar[str] = "lgl_finalize_positions"
    category: ClassVar[OpCategory] = OpCategory.POSTPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Move and cast final layout coordinates like the classic implementation."""

        del ctx

        if state.pos is None:
            raise ValueError("LGLFinalizePositions requires state.pos to be set.")

        output_device = layout_device(
            edge_index=problem.edge_index,
            node_sizes=problem.node_sizes,
        )
        state.pos = state.pos.to(dtype=torch.float32, device=output_device)
        return state


__all__ = [
    "LGLPrepareState",
    "LGLPrepareStateConfig",
    "LGLInitializePositions",
    "LGLLayeredRefinement",
    "LGLFinalizePositions",
]
