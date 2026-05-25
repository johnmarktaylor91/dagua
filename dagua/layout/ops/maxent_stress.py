"""Composable maxent-stress operations.

This module contains registered operations for both majorization and gradient
branches of the maxent-stress algorithm.
"""

from __future__ import annotations

import ctypes
import math
from collections import deque
from dataclasses import dataclass, field
from typing import ClassVar, Optional, Tuple

import numpy as np
import torch

from dagua.layout.ops.base import Op
from dagua.layout.ops.graph_utils import (
    _shared_all_pairs_shortest_paths,
    _shared_build_undirected_adjacency,
    layout_device,
    layout_extent,
    normalize_positions,
)
from dagua.layout.ops.graph_utils import (
    _shared_bfs_distances as bfs_distances,
)
from dagua.layout.ops.graph_utils import (
    _shared_dijkstra_distances as dijkstra_distances,
)
from dagua.layout.ops.pipelines.pivot_mds import (
    layout_pivot_mds_pipeline as layout_pivot_mds,
)
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_MIN_DISTANCE: float = 1.0e-3
_FULL_STRESS_LIMIT: int = 1_000
_PIVOT_COUNT: int = 50
_SAMPLED_REPULSION_NEIGHBORS: int = 96
_OGDF_RUNNER_RANDOM_MODULUS: int = 1_000
_OGDF_RUNNER_RANDOM_SCALE: float = 10.0


def _ogdf_runner_initial_positions(num_nodes: int, seed: int) -> torch.Tensor:
    """Return the OGDF runner-owned stress initial layout.

    Parameters
    ----------
    num_nodes : int
        Number of nodes ``N`` in the graph.
    seed : int
        Seed forwarded to ``std::srand`` before drawing coordinates.

    Returns
    -------
    torch.Tensor
        Initial coordinates with shape ``[N, 2]`` and dtype ``float64``.

    Notes
    -----
    ``scripts/ogdf_runner.cpp`` sets ``StressMinimization::hasInitialLayout``
    and fills each coordinate with ``(std::rand() % 1000) / 10.0``. That
    bypasses OGDF's internal PivotMDS warm start for this fidelity target, so
    matching the runner's libc RNG stream is the dominant bit-exact component.
    """
    libc = ctypes.CDLL(None)
    libc.srand(ctypes.c_uint(seed))
    positions = torch.empty((num_nodes, 2), dtype=torch.float64)
    for node_index in range(num_nodes):
        positions[node_index, 0] = (
            float(libc.rand() % _OGDF_RUNNER_RANDOM_MODULUS) / _OGDF_RUNNER_RANDOM_SCALE
        )
        positions[node_index, 1] = (
            float(libc.rand() % _OGDF_RUNNER_RANDOM_MODULUS) / _OGDF_RUNNER_RANDOM_SCALE
        )
    return positions


def _path_warm_start_positions(problem: LayoutProblem) -> Optional[torch.Tensor]:
    """Return OGDF-style line coordinates for simple-path graphs.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable graph inputs for the maxent-stress run.

    Returns
    -------
    torch.Tensor | None
        Raw ``float32`` coordinates with shape ``[N, 2]`` when the simple
        undirected graph is a path, otherwise ``None``.
    """
    if problem.num_nodes <= 1:
        return torch.zeros((problem.num_nodes, 2), dtype=torch.float32)

    adjacency = _shared_build_undirected_adjacency(
        edge_index=problem.edge_index,
        num_nodes=problem.num_nodes,
        edge_weights=problem.edge_weights,
    )
    degrees = [len(neighbors) for neighbors in adjacency]
    if any(degree > 2 for degree in degrees):
        return None

    edge_count = sum(degrees) // 2
    if edge_count != problem.num_nodes - 1:
        return None

    endpoints = [index for index, degree in enumerate(degrees) if degree <= 1]
    if len(endpoints) != 2:
        return None

    positions = torch.zeros((problem.num_nodes, 2), dtype=torch.float32)
    previous = -1
    current = endpoints[0]
    x_position = 0.0
    visited = set()
    while current not in visited:
        visited.add(current)
        positions[current, 0] = x_position
        next_node = -1
        next_weight = 1.0
        for neighbor, weight in adjacency[current]:
            if neighbor != previous:
                next_node = neighbor
                next_weight = float(weight)
                break
        if next_node < 0:
            break
        previous = current
        current = next_node
        x_position += next_weight

    if len(visited) != problem.num_nodes:
        return None
    return positions


@dataclass(frozen=True)
class MaxentInitializePositionsConfig:
    """Configuration for :class:`MaxentInitializePositions`.

    Parameters
    ----------
    pivot_count : int, default=50
        Maximum number of pivots used by the PivotMDS warm start.
    """

    pivot_count: int = _PIVOT_COUNT


@dataclass(frozen=True)
class MaxentPrepareStateConfig:
    """Configuration for :class:`MaxentPrepareState`.

    Parameters
    ----------
    full_stress_limit : int, default=1000
        Node cutoff below which the gradient branch materializes full stress pairs.
    pivot_count : int, default=50
        Maximum number of pivots retained for large-graph approximations.
    """

    full_stress_limit: int = _FULL_STRESS_LIMIT
    pivot_count: int = _PIVOT_COUNT


@register_op
@dataclass(frozen=True)
class MaxentInitializePositions(Op):
    """Initialize maxent-stress positions with the classic PivotMDS warm start.

    Parameters
    ----------
    for_majorization : bool, default=False
        Whether to initialize in ``float64`` on CPU for majorization.
    """

    for_majorization: bool = False
    config: MaxentInitializePositionsConfig = field(default_factory=MaxentInitializePositionsConfig)
    name: ClassVar[str] = "maxent_initialize_positions"
    category: ClassVar[OpCategory] = OpCategory.INIT
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Seed positions from classic PivotMDS initialization.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state receiving initialized positions.
        ctx : RuntimeContext
            Runtime context (unused).

        Returns
        -------
        SolveState
            State containing ``state.pos``.
        """
        del ctx

        if self.for_majorization:
            positions = _ogdf_runner_initial_positions(problem.num_nodes, problem.seed)
        else:
            positions = layout_pivot_mds(
                edge_index=problem.edge_index,
                num_nodes=problem.num_nodes,
                node_sizes=problem.node_sizes,
                n_pivots=min(self.config.pivot_count, problem.num_nodes),
                seed=problem.seed,
                edge_weights=problem.edge_weights,
            )
        if self.for_majorization:
            state.pos = positions.to(device="cpu", dtype=torch.float64)
        else:
            device = layout_device(edge_index=problem.edge_index, node_sizes=problem.node_sizes)
            state.pos = positions.to(device=device, dtype=torch.float32)
        return state


@register_op
@dataclass(frozen=True)
class MaxentPrepareState(Op):
    """Precompute graph-derived terms for either maxent-stress execution branch.

    Parameters
    ----------
    for_majorization : bool, default=False
        Whether to precompute majorization distance terms.
    use_entropy : bool, default=False
        Whether to precompute full non-edge pairs for entropy.
    """

    for_majorization: bool = False
    use_entropy: bool = False
    config: MaxentPrepareStateConfig = field(default_factory=MaxentPrepareStateConfig)
    name: ClassVar[str] = "maxent_prepare_state"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = ("distance_matrix", "adjacency_weighted", "extras")
    requires: ClassVar[Tuple[str, ...]] = ()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build adjacency and stress approximation state for the configured branch.

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
            State with typed adjacency and, for majorization, typed graph
            distances plus maxent-specific extras.
        """
        del ctx

        weighted = problem.edge_weights is not None
        if problem.edge_weights is None or problem.edge_weights.numel() == 0:
            average_edge_cost = 100.0 if self.for_majorization else 1.0
        else:
            average_edge_cost = float(
                problem.edge_weights.detach().to(device="cpu", dtype=torch.float32).mean().item()
            )

        adjacency = _shared_build_undirected_adjacency(
            edge_index=problem.edge_index,
            num_nodes=problem.num_nodes,
            edge_weights=problem.edge_weights,
        )

        state.adjacency_weighted = adjacency

        if self.for_majorization:
            raw_distances = _shared_all_pairs_shortest_paths(
                adjacency,
                weighted=weighted,
            )
            if raw_distances.size == 0:
                graph_distances = torch.empty((0, 0), dtype=torch.float64)
            else:
                # Disconnected entries are replaced with the same heuristic scale
                # used by the classic implementation so every pair remains finite.
                disconnected_distance = average_edge_cost * math.sqrt(
                    float(max(problem.num_nodes, 1))
                )
                cleaned = raw_distances.astype(np.float64, copy=True)
                if weighted:
                    cleaned[np.isinf(cleaned)] = disconnected_distance
                else:
                    cleaned[cleaned < 0] = math.sqrt(float(max(problem.num_nodes, 1)))
                    cleaned *= average_edge_cost
                graph_distances = torch.tensor(cleaned, dtype=torch.float64)
            weight_matrix = torch.zeros_like(graph_distances)
            off_diagonal = ~torch.eye(
                problem.num_nodes, dtype=torch.bool, device=graph_distances.device
            )
            if graph_distances.numel() > 0:
                weight_matrix[off_diagonal] = graph_distances[off_diagonal].reciprocal().square()
            state.distance_matrix = graph_distances
            state.extras["maxent_weight_matrix"] = weight_matrix
            return state

        if problem.num_nodes <= self.config.full_stress_limit:
            raw_distances = _shared_all_pairs_shortest_paths(adjacency, weighted=weighted)
            if raw_distances.size == 0:
                stress_src = torch.empty((0,), dtype=torch.long)
                stress_dst = torch.empty((0,), dtype=torch.long)
                stress_lengths = torch.empty((0,), dtype=torch.float32)
            else:
                disconnected_distance = average_edge_cost * math.sqrt(
                    float(max(problem.num_nodes, 1))
                )
                cleaned = raw_distances.astype(np.float64, copy=True)
                if weighted:
                    cleaned[np.isinf(cleaned)] = disconnected_distance
                else:
                    cleaned[cleaned < 0] = disconnected_distance
                distances = torch.tensor(cleaned, dtype=torch.float32)
                upper = torch.triu_indices(distances.shape[0], distances.shape[1], offset=1)
                stress_src = upper[0]
                stress_dst = upper[1]
                stress_lengths = distances[upper[0], upper[1]]

            pivot_indices = torch.empty((0,), dtype=torch.long)
            pivot_distances = torch.empty((problem.num_nodes, 0), dtype=torch.float32)
            if self.use_entropy:
                if problem.num_nodes <= 1:
                    full_ne_src = torch.empty((0,), dtype=torch.long)
                    full_ne_dst = torch.empty((0,), dtype=torch.long)
                else:
                    adjacency_mask = torch.zeros(
                        (problem.num_nodes, problem.num_nodes), dtype=torch.bool
                    )
                    for source, neighbors in enumerate(adjacency):
                        if neighbors:
                            neighbor_indices = torch.tensor(
                                [neighbor for neighbor, _ in neighbors],
                                dtype=torch.long,
                            )
                            adjacency_mask[source, neighbor_indices] = True
                    upper = torch.triu_indices(problem.num_nodes, problem.num_nodes, offset=1)
                    mask = ~adjacency_mask[upper[0], upper[1]]
                    full_ne_src = upper[0][mask]
                    full_ne_dst = upper[1][mask]
            else:
                full_ne_src = torch.empty((0,), dtype=torch.long)
                full_ne_dst = torch.empty((0,), dtype=torch.long)
        else:
            if problem.num_nodes == 0:
                stress_src = torch.empty((0,), dtype=torch.long)
                stress_dst = torch.empty((0,), dtype=torch.long)
                stress_lengths = torch.empty((0,), dtype=torch.float32)
                pivot_indices = torch.empty((0,), dtype=torch.long)
                pivot_distances = torch.empty((0, 0), dtype=torch.float32)
            else:
                component_ids = torch.full((problem.num_nodes,), -1, dtype=torch.long)
                component_index = 0
                for start in range(problem.num_nodes):
                    if int(component_ids[start].item()) >= 0:
                        continue
                    frontier: deque[int] = deque([start])
                    component_ids[start] = component_index
                    while frontier:
                        node = frontier.popleft()
                        for neighbor, _ in adjacency[node]:
                            if int(component_ids[neighbor].item()) >= 0:
                                continue
                            component_ids[neighbor] = component_index
                            frontier.append(neighbor)
                    component_index += 1

                max_pivots = min(self.config.pivot_count, problem.num_nodes)
                if problem.num_nodes <= max_pivots:
                    pivots = torch.arange(problem.num_nodes, dtype=torch.long)
                else:
                    pivots = []
                    seen_components: set[int] = set()
                    for node, component in enumerate(component_ids.tolist()):
                        if component in seen_components:
                            continue
                        pivots.append(node)
                        seen_components.add(component)
                        if len(pivots) == max_pivots:
                            pivots = torch.tensor(pivots, dtype=torch.long)
                            break

                    if not isinstance(pivots, torch.Tensor):
                        remaining = max_pivots - len(pivots)
                        if remaining <= 0:
                            pivots = torch.tensor(pivots, dtype=torch.long)
                        else:
                            generator = torch.Generator(device="cpu")
                            generator.manual_seed(problem.seed)
                            pivot_mask = torch.zeros(problem.num_nodes, dtype=torch.bool)
                            pivot_mask[torch.tensor(pivots, dtype=torch.long)] = True
                            candidates = torch.arange(problem.num_nodes, dtype=torch.long)[
                                ~pivot_mask
                            ]
                            permutation = torch.randperm(
                                int(candidates.shape[0]),
                                generator=generator,
                            )
                            extra = candidates[permutation[:remaining]]
                            pivots = torch.tensor(pivots, dtype=torch.long)
                            pivots = torch.cat([pivots, extra], dim=0)

                if int(pivots.numel()) == 0:
                    pivot_indices = pivots
                    pivot_distances = torch.empty((problem.num_nodes, 0), dtype=torch.float32)
                else:
                    # Seed pivots with one node per connected component before
                    # random fill so every component receives at least one anchor.
                    disconnected_distance = average_edge_cost * math.sqrt(
                        float(max(problem.num_nodes, 1))
                    )
                    pivot_rows: list[torch.Tensor] = []
                    for pivot in pivots:
                        pivot_id = int(pivot.item())
                        if weighted:
                            pivot_raw = dijkstra_distances(adjacency, pivot_id)
                            pivot_clean = np.where(
                                np.isinf(pivot_raw), disconnected_distance, pivot_raw
                            )
                        else:
                            pivot_raw = bfs_distances(adjacency, pivot_id).astype(np.float64)
                            pivot_clean = np.where(pivot_raw < 0, disconnected_distance, pivot_raw)
                        pivot_rows.append(torch.tensor(pivot_clean, dtype=torch.float32))
                    pivot_distances = torch.stack(pivot_rows, dim=0).transpose(0, 1)

                    pivot_indices = pivots
            stress_src = torch.empty((0,), dtype=torch.long)
            stress_dst = torch.empty((0,), dtype=torch.long)
            stress_lengths = torch.empty((0,), dtype=torch.float32)
            full_ne_src = torch.empty((0,), dtype=torch.long)
            full_ne_dst = torch.empty((0,), dtype=torch.long)

        state.extras["maxent_stress_src"] = stress_src
        state.extras["maxent_stress_dst"] = stress_dst
        state.extras["maxent_stress_lengths"] = stress_lengths
        state.extras["maxent_pivot_indices"] = pivot_indices
        state.extras["maxent_pivot_distances"] = pivot_distances
        state.extras["maxent_full_ne_src"] = full_ne_src
        state.extras["maxent_full_ne_dst"] = full_ne_dst
        return state


@dataclass(frozen=True)
class MaxentInitializeOptimizerConfig:
    """Configuration for :class:`MaxentInitializeOptimizer`.

    Parameters
    ----------
    lr_cap : float, default=0.04
        Upper bound on the initial Adam learning rate.
    lr_numerator : float, default=0.8
        Numerator for the ``lr_numerator / num_nodes`` learning rate formula.
    lr_final_ratio : float, default=0.1
        Floor multiplier for the final learning rate relative to the initial.
    """

    lr_cap: float = 0.04
    lr_numerator: float = 0.8
    lr_final_ratio: float = 0.1


@dataclass(frozen=True)
class MaxentGradientStepConfig:
    """Configuration for :class:`MaxentGradientStep`.

    Parameters
    ----------
    min_batch_size : int, default=16
        Floor for the random-pair batch size in the gradient term.
    batch_multiplier : int, default=3
        Multiplier applied to remaining-pair count when sizing batches.
    min_distance : float, default=1e-3
        Clamp floor applied to Euclidean distances inside the loss.
    full_stress_limit : int, default=1000
        Node cutoff below which the gradient branch uses all-pairs stress terms.
    sampled_repulsion_neighbors : int, default=96
        Approximate non-edge sample budget multiplier for entropy mode.
    """

    min_batch_size: int = 16
    batch_multiplier: int = 3
    min_distance: float = _MIN_DISTANCE
    full_stress_limit: int = _FULL_STRESS_LIMIT
    sampled_repulsion_neighbors: int = _SAMPLED_REPULSION_NEIGHBORS


@register_op
@dataclass(frozen=True)
class MaxentInitializeOptimizer(Op):
    """Create the Adam optimizer used by the gradient branch."""

    config: MaxentInitializeOptimizerConfig = field(default_factory=MaxentInitializeOptimizerConfig)

    name: ClassVar[str] = "maxent_initialize_optimizer"
    category: ClassVar[OpCategory] = OpCategory.OPTIMIZE
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ("pos", "extras", "optimizer")
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Enable gradients and create the optimizer with classic LR bounds.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state with initialized ``pos``.
        ctx : RuntimeContext
            Runtime context (unused).

        Returns
        -------
        SolveState
            State updated with ``state.optimizer`` and LR trace extras.
        """
        del ctx

        assert state.pos is not None
        # ``requires_grad_(True)`` mutates the shared tensor in place so later
        # ops can keep using ``state.pos`` directly.
        state.pos = state.pos.requires_grad_(True)
        initial_lr = min(
            self.config.lr_cap,
            self.config.lr_numerator / float(max(problem.num_nodes, 1)),
        )
        final_lr = max(
            initial_lr * self.config.lr_final_ratio,
            initial_lr / math.sqrt(float(max(state.total_steps, 1))),
        )
        state.optimizer = torch.optim.Adam([state.pos], lr=initial_lr)
        state.extras["maxent_initial_lr"] = initial_lr
        state.extras["maxent_final_lr"] = final_lr
        return state


@register_op
@dataclass(frozen=True)
class MaxentGradientStep(Op):
    """Run one Adam step of the maxent-stress gradient objective."""

    alpha: float = 1.0
    use_entropy: bool = False
    config: MaxentGradientStepConfig = field(default_factory=MaxentGradientStepConfig)
    name: ClassVar[str] = "maxent_gradient_step"
    category: ClassVar[OpCategory] = OpCategory.OPTIMIZE
    reads: ClassVar[Tuple[str, ...]] = ("pos", "optimizer", "adjacency_weighted", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("pos", "optimizer")
    requires: ClassVar[Tuple[str, ...]] = ("pos", "optimizer", "adjacency_weighted", "extras")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Evaluate one maxent-stress gradient objective and update positions.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with precomputed gradient terms.
        ctx : RuntimeContext
            Runtime context (unused).

        Returns
        -------
        SolveState
            State with updated positions and optimizer learning-rate state.
        """
        del ctx

        assert state.pos is not None
        assert state.optimizer is not None

        positions = state.pos
        optimizer = state.optimizer
        step = state.step

        stress_src = state.extras["maxent_stress_src"]
        stress_dst = state.extras["maxent_stress_dst"]
        stress_lengths = state.extras["maxent_stress_lengths"]
        pivot_indices = state.extras["maxent_pivot_indices"]
        pivot_distances = state.extras["maxent_pivot_distances"]
        adjacency = state.adjacency_weighted
        if adjacency is None:
            raise ValueError("MaxentGradientStep requires state.adjacency_weighted to be set.")

        optimizer.zero_grad(set_to_none=True)
        if int(problem.num_nodes) <= self.config.full_stress_limit:
            full_ne_src = state.extras["maxent_full_ne_src"]
            full_ne_dst = state.extras["maxent_full_ne_dst"]
            if int(stress_src.numel()) > 0:
                src = stress_src.to(device=positions.device)
                dst = stress_dst.to(device=positions.device)
                targets = stress_lengths.to(device=positions.device)
                distances = torch.linalg.norm(positions[src] - positions[dst], dim=1).clamp(
                    min=self.config.min_distance
                )
                weights = targets.reciprocal().square()
                loss = (weights * (distances - targets).square()).sum()
            else:
                loss = torch.tensor(0.0, dtype=positions.dtype, device=positions.device)

            if self.use_entropy:
                if int(full_ne_src.numel()) == 0:
                    entropy = torch.tensor(0.0, dtype=positions.dtype, device=positions.device)
                else:
                    non_edge_distances = torch.linalg.norm(
                        positions[full_ne_src.to(device=positions.device)]
                        - positions[full_ne_dst.to(device=positions.device)],
                        dim=1,
                    ).clamp(min=self.config.min_distance)
                    entropy = -torch.log(non_edge_distances).sum()
                loss = loss + self.alpha * entropy
        else:
            if int(stress_src.numel()) > 0:
                src = stress_src.to(device=positions.device)
                dst = stress_dst.to(device=positions.device)
                targets = stress_lengths.to(device=positions.device)
                distances = torch.linalg.norm(positions[src] - positions[dst], dim=1).clamp(
                    min=self.config.min_distance
                )
                weights = targets.reciprocal().square()
                loss = (weights * (distances - targets).square()).sum()
            elif int(pivot_indices.numel()) > 0:
                pivot_positions = positions[pivot_indices.to(device=positions.device)]
                geometric = torch.cdist(positions, pivot_positions).clamp(
                    min=self.config.min_distance
                )
                targets = pivot_distances.to(device=positions.device)
                reachable = targets > 0
                safe_targets = torch.where(reachable, targets, torch.ones_like(targets))
                weights = torch.where(
                    reachable,
                    safe_targets.reciprocal().square(),
                    torch.zeros_like(targets),
                )
                loss = (weights * (geometric - safe_targets).square()).sum()
            else:
                loss = torch.tensor(0.0, dtype=positions.dtype, device=positions.device)

            if self.use_entropy:
                num_nodes = int(len(adjacency))
                total_pairs = num_nodes * (num_nodes - 1) // 2
                edge_count = sum(len(neighbors) for neighbors in adjacency) // 2
                total_non_edges = max(total_pairs - edge_count, 0)
                if total_non_edges == 0:
                    entropy = torch.tensor(0.0, dtype=positions.dtype, device=positions.device)
                else:
                    adjacency_sets = [
                        {neighbor for neighbor, _ in neighbors} | {node}
                        for node, neighbors in enumerate(adjacency)
                    ]
                    total_non_edges_sample = 0
                    sources: list[int] = []
                    targets: list[int] = []
                    generator = torch.Generator(device="cpu")
                    generator.manual_seed(problem.seed + step + 1)
                    sample_size = min(
                        total_non_edges,
                        max(
                            num_nodes,
                            num_nodes * self.config.sampled_repulsion_neighbors // 2,
                        ),
                    )
                    while total_non_edges_sample < sample_size:
                        remaining = sample_size - total_non_edges_sample
                        batch_size = max(
                            remaining * self.config.batch_multiplier, self.config.min_batch_size
                        )
                        candidate_sources = torch.randint(
                            0,
                            num_nodes,
                            (batch_size,),
                            generator=generator,
                            dtype=torch.long,
                        ).tolist()
                        candidate_targets = torch.randint(
                            0,
                            num_nodes,
                            (batch_size,),
                            generator=generator,
                            dtype=torch.long,
                        ).tolist()
                        for source, target in zip(candidate_sources, candidate_targets):
                            if target not in adjacency_sets[source]:
                                # Oversampling candidate pairs is cheaper than
                                # enumerating all non-edges on large graphs.
                                sources.append(min(source, target))
                                targets.append(max(source, target))
                                total_non_edges_sample += 1
                                if total_non_edges_sample >= sample_size:
                                    break

                    sampled_src = torch.tensor(sources, dtype=torch.long, device=positions.device)
                    sampled_dst = torch.tensor(targets, dtype=torch.long, device=positions.device)
                    if sampled_src.numel() == 0:
                        entropy = torch.tensor(0.0, dtype=positions.dtype, device=positions.device)
                    else:
                        non_edge_distances = torch.linalg.norm(
                            positions[sampled_src] - positions[sampled_dst],
                            dim=1,
                        ).clamp(min=self.config.min_distance)
                        entropy = -torch.log(non_edge_distances).sum() * (
                            float(total_non_edges) / float(max(int(sampled_src.numel()), 1))
                        )
                loss = loss + self.alpha * entropy

        loss.backward()
        optimizer.step()

        initial_lr = state.extras["maxent_initial_lr"]
        final_lr = state.extras["maxent_final_lr"]
        # The classic branch linearly anneals Adam's LR from the initialized
        # value toward the computed floor over the requested step budget.
        fraction = float(step + 1) / float(max(state.total_steps, 1))
        optimizer.param_groups[0]["lr"] = initial_lr + (final_lr - initial_lr) * fraction
        return state


@register_op
@dataclass(frozen=True)
class MaxentMajorizationStep(Op):
    """Run one dense Gauss-Seidel majorization sweep over all nodes."""

    name: ClassVar[str] = "maxent_majorization_step"
    category: ClassVar[OpCategory] = OpCategory.OPTIMIZE
    reads: ClassVar[Tuple[str, ...]] = ("pos", "distance_matrix", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos", "distance_matrix", "extras")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Update every node coordinate via stress majorization.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs (unused).
        state : SolveState
            Mutable solve state with precomputed majorization terms.
        ctx : RuntimeContext
            Runtime context (unused).

        Returns
        -------
        SolveState
            State after one majorization update.
        """
        del problem, ctx

        assert state.pos is not None
        positions = state.pos
        graph_distances = state.distance_matrix
        if graph_distances is None:
            raise ValueError("MaxentMajorizationStep requires state.distance_matrix.")
        weight_matrix = state.extras["maxent_weight_matrix"]

        num_nodes = int(positions.shape[0])
        for node_index in range(num_nodes):
            current_x = float(positions[node_index, 0].item())
            current_y = float(positions[node_index, 1].item())
            new_x = 0.0
            new_y = 0.0
            total_weight = 0.0

            for other_index in range(num_nodes):
                if node_index == other_index:
                    continue
                weight = float(weight_matrix[node_index, other_index].item())
                desired_distance = float(graph_distances[node_index, other_index].item())
                other_x = float(positions[other_index, 0].item())
                other_y = float(positions[other_index, 1].item())
                delta_x = current_x - other_x
                delta_y = current_y - other_y
                euclidean_distance = math.hypot(delta_x, delta_y)

                # Each neighbor contributes a weighted vote along the ray from
                # its current position toward the target graph distance.
                vote_x = other_x
                vote_y = other_y
                if euclidean_distance != 0.0:
                    vote_x += desired_distance * (current_x - vote_x) / euclidean_distance
                    vote_y += desired_distance * (current_y - vote_y) / euclidean_distance
                new_x += weight * vote_x
                new_y += weight * vote_y
                total_weight += weight

            if total_weight != 0.0:
                positions[node_index, 0] = new_x / total_weight
                positions[node_index, 1] = new_y / total_weight

        return state


@register_op
@dataclass(frozen=True)
class MaxentFinalizePositions(Op):
    """Normalize final coordinates and cast them onto the output device."""

    for_majorization: bool = False
    name: ClassVar[str] = "maxent_finalize_positions"
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
        """Normalize coordinates and cast to output dtype/device.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs used for output device/extent.
        state : SolveState
            Mutable solve state with final coordinates.
        ctx : RuntimeContext
            Runtime context (unused).

        Returns
        -------
        SolveState
            State with normalized output positions.
        """
        del ctx

        assert state.pos is not None

        device = layout_device(problem.edge_index, problem.node_sizes)
        extent = layout_extent(problem.num_nodes, problem.node_sizes)
        if self.for_majorization:
            state.pos = normalize_positions(state.pos.to(dtype=torch.float32), extent=extent).to(
                dtype=torch.float32, device=device
            )
        else:
            state.pos = normalize_positions(state.pos.detach(), extent=extent).to(
                dtype=torch.float32, device=device
            )
        return state
