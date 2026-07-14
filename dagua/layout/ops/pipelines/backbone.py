"""Backbone layout pipeline after graphlayouts' simmelian backbone layout."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar, List, Optional, Tuple, Union

import numpy as np
import torch

from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.graph_utils import (
    _shared_all_pairs_shortest_paths,
    _shared_build_undirected_adjacency,
)
from dagua.layout.ops.graph_utils import layout_device as _layout_device
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_DEFAULT_KEEP = 0.2
_DEFAULT_ITERATIONS = 500
_DEFAULT_TOLERANCE = 0.0001
_R_LAYOUT_SEED = 42
_MIN_STRESS_DISTANCE = 1.0e-10
_BACKBONE_EDGES_KEY = "backbone_edges"
_BACKBONE_WEIGHTS_KEY = "backbone_weights"
_BACKBONE_EDGE_MASK_KEY = "backbone_edge_mask"


@dataclass(frozen=True)
class BackboneConfig:
    """Configuration for the backbone layout pipeline.

    Parameters
    ----------
    keep : float, default=0.2
        Fraction of highest reweighted edges to keep before unioning with the
        maximum spanning tree. Values are clipped by edge-count ceiling exactly
        like graphlayouts' ``layout_as_backbone(keep=...)`` filter.
    iterations : int, default=500
        Maximum number of graphlayouts-style stress majorization iterations.
    tolerance : float, default=0.0001
        Relative stress-improvement stopping threshold.
    dtype : torch.dtype, default=torch.float32
        Output tensor dtype.
    """

    keep: float = _DEFAULT_KEEP
    iterations: int = _DEFAULT_ITERATIONS
    tolerance: float = _DEFAULT_TOLERANCE
    dtype: torch.dtype = torch.float32


def _canonical_edges(edge_index: torch.Tensor, num_nodes: int) -> List[Tuple[int, int]]:
    """Return unique undirected non-loop edges in input order.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    list[tuple[int, int]]
        Unique undirected edges with lower node id first.

    Raises
    ------
    ValueError
        If an edge references a node outside ``[0, num_nodes)``.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")

    edges: List[Tuple[int, int]] = []
    seen: set[Tuple[int, int]] = set()
    for edge_pos in range(int(edge_index.shape[1])):
        source = int(edge_index[0, edge_pos].item())
        target = int(edge_index[1, edge_pos].item())
        if source < 0 or target < 0 or source >= num_nodes or target >= num_nodes:
            raise ValueError("edge_index contains a node id outside [0, num_nodes).")
        if source == target:
            continue
        key = (source, target) if source < target else (target, source)
        if key in seen:
            continue
        seen.add(key)
        edges.append(key)
    return edges


def _neighbor_sets(edges: List[Tuple[int, int]], num_nodes: int) -> List[set[int]]:
    """Build undirected neighbor sets.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Unique undirected edge list.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    list[set[int]]
        Neighbor set for each node.
    """
    neighbors = [set() for _ in range(num_nodes)]
    for source, target in edges:
        neighbors[source].add(target)
        neighbors[target].add(source)
    return neighbors


def _edge_embeddedness(
    edges: List[Tuple[int, int]],
    neighbors: List[set[int]],
) -> np.ndarray:
    """Score each edge by non-induced C4 embeddedness.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Unique undirected edge list.
    neighbors : list[set[int]]
        Neighbor sets for each node.

    Returns
    -------
    numpy.ndarray
        Embeddedness scores with shape ``[E]``.

    Notes
    -----
    graphlayouts uses ``oaqc::oaqc(..., non_ind_freq=TRUE)$e_orbits_non_ind[, 11]``.
    In ``oaqc`` this is the per-edge non-induced 4-cycle count: for edge
    ``(u, v)``, count distinct length-3 paths from ``u`` to ``v`` that do not
    reuse either endpoint. Computing that orbit directly keeps the production
    pipeline free of R runtime delegation.
    """
    scores = np.zeros(len(edges), dtype=np.float64)
    for index, (source, target) in enumerate(edges):
        count = 0
        for source_neighbor in neighbors[source]:
            if source_neighbor == target:
                continue
            for target_neighbor in neighbors[target]:
                if target_neighbor == source or target_neighbor == source_neighbor:
                    continue
                if target_neighbor in neighbors[source_neighbor]:
                    count += 1
        scores[index] = float(count)
    return scores


def _initial_simmelian_weights(
    edges: List[Tuple[int, int]],
    embeddedness: np.ndarray,
    num_nodes: int,
) -> np.ndarray:
    """Normalize embeddedness by endpoint embeddedness mass.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Unique undirected edge list.
    embeddedness : numpy.ndarray
        Raw edge embeddedness with shape ``[E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    numpy.ndarray
        Normalized edge weights with shape ``[E]``.
    """
    endpoint_mass = np.zeros(num_nodes, dtype=np.float64)
    for edge_pos, (source, target) in enumerate(edges):
        endpoint_mass[source] += embeddedness[edge_pos]
        endpoint_mass[target] += embeddedness[edge_pos]

    weights = np.zeros(len(edges), dtype=np.float64)
    for edge_pos, (source, target) in enumerate(edges):
        denom = math.sqrt(endpoint_mass[source] * endpoint_mass[target])
        if denom > 0.0 and math.isfinite(denom):
            weights[edge_pos] = embeddedness[edge_pos] / denom
    return weights


def _dense_descending_ranks(values: List[float]) -> List[int]:
    """Return graphlayouts/R-style dense ranks for descending weights.

    Parameters
    ----------
    values : list[float]
        Values to rank.

    Returns
    -------
    list[int]
        Zero-based dense ranks, where larger values receive smaller ranks.
    """
    unique_values = sorted(set(values), reverse=True)
    rank_by_value = {value: rank for rank, value in enumerate(unique_values)}
    return [rank_by_value[value] for value in values]


def _ranked_neighbors(
    edges: List[Tuple[int, int]],
    weights: np.ndarray,
    num_nodes: int,
) -> List[List[Tuple[int, int]]]:
    """Build graphlayouts-compatible ranked neighbor tables.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Unique undirected edge list.
    weights : numpy.ndarray
        Edge weights used for neighbor ranking with shape ``[E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    list[list[tuple[int, int]]]
        Per-node ``(neighbor, dense_rank)`` rows sorted by rank.
    """
    weighted_neighbors: List[List[Tuple[int, float]]] = [[] for _ in range(num_nodes)]
    for edge_pos, (source, target) in enumerate(edges):
        weight = float(weights[edge_pos])
        weighted_neighbors[source].append((target, weight))
        weighted_neighbors[target].append((source, weight))

    ranked: List[List[Tuple[int, int]]] = []
    for rows in weighted_neighbors:
        ranks = _dense_descending_ranks([weight for _, weight in rows])
        node_rows = [(rows[index][0], ranks[index]) for index in range(len(rows))]
        node_rows.sort(key=lambda item: (item[1], item[0]))
        ranked.append(node_rows)
    return ranked


def _prefix_jaccard_weights(
    edges: List[Tuple[int, int]],
    ranked: List[List[Tuple[int, int]]],
) -> np.ndarray:
    """Apply graphlayouts' maximum-prefix Jaccard reweighting.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Unique undirected edge list.
    ranked : list[list[tuple[int, int]]]
        Per-node ranked neighbor tables.

    Returns
    -------
    numpy.ndarray
        Reweighted backbone scores with shape ``[E]``.
    """
    new_weights = np.zeros(len(edges), dtype=np.float64)
    for edge_pos, (source, target) in enumerate(edges):
        source_rows = ranked[source]
        target_rows = ranked[target]
        if not source_rows or not target_rows:
            continue
        max_source_rank = max(rank for _, rank in source_rows)
        max_target_rank = max(rank for _, rank in target_rows)
        max_rank = max(max_source_rank, max_target_rank)
        best = 0.0
        for rank in range(max_rank):
            source_cut = min(rank, max_source_rank)
            target_cut = min(rank, max_target_rank)
            source_nodes = {
                neighbor for neighbor, neighbor_rank in source_rows if neighbor_rank <= source_cut
            }
            target_nodes = {
                neighbor for neighbor, neighbor_rank in target_rows if neighbor_rank <= target_cut
            }
            union_size = len(source_nodes.union(target_nodes))
            if union_size == 0:
                continue
            score = len(source_nodes.intersection(target_nodes)) / float(union_size)
            best = max(best, score)
        new_weights[edge_pos] = best
    return new_weights


def _union_maximum_spanning_tree(
    edges: List[Tuple[int, int]],
    weights: np.ndarray,
    num_nodes: int,
) -> set[Tuple[int, int]]:
    """Return graphlayouts' union maximum spanning tree edge set.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Unique undirected edge list.
    weights : numpy.ndarray
        Edge weights with shape ``[E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    set[tuple[int, int]]
        Edges selected by descending tied-rank Kruskal sweeps.
    """
    parent = list(range(num_nodes))

    def find(node: int) -> int:
        """Find the current disjoint-set root for ``node``.

        Parameters
        ----------
        node : int
            Node id.

        Returns
        -------
        int
            Root representative.
        """
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    chosen: set[Tuple[int, int]] = set()
    unique_weights = sorted(set(float(weight) for weight in weights), reverse=True)
    for weight in unique_weights:
        batch = [
            edge_pos for edge_pos, edge_weight in enumerate(weights) if float(edge_weight) == weight
        ]
        accepted: List[Tuple[int, int]] = []
        for edge_pos in batch:
            source, target = edges[edge_pos]
            if find(source) != find(target):
                accepted.append((source, target))
        for source, target in accepted:
            source_root = find(source)
            target_root = find(target)
            if source_root != target_root:
                parent[target_root] = source_root
            chosen.add((source, target))
    return chosen


def backbone_edge_set(
    edge_index: torch.Tensor,
    num_nodes: int,
    keep: float = _DEFAULT_KEEP,
) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray]:
    """Compute the deterministic backbone edge set.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    keep : float, default=0.2
        Fraction of highest reweighted edges to keep.

    Returns
    -------
    tuple[list[tuple[int, int]], numpy.ndarray, numpy.ndarray]
        Backbone edges, reweighted scores for all unique edges, and a boolean
        mask aligned to the unique input-order edge list.

    Raises
    ------
    ValueError
        If ``keep`` is outside ``(0, 1]``.
    """
    if keep <= 0.0 or keep > 1.0:
        raise ValueError("keep must be in the interval (0, 1].")

    edges = _canonical_edges(edge_index=edge_index, num_nodes=num_nodes)
    if not edges:
        return [], np.empty(0, dtype=np.float64), np.empty(0, dtype=bool)

    neighbors = _neighbor_sets(edges=edges, num_nodes=num_nodes)
    embeddedness = _edge_embeddedness(edges=edges, neighbors=neighbors)
    initial_weights = _initial_simmelian_weights(
        edges=edges,
        embeddedness=embeddedness,
        num_nodes=num_nodes,
    )
    ranked = _ranked_neighbors(edges=edges, weights=initial_weights, num_nodes=num_nodes)
    weights = _prefix_jaccard_weights(edges=edges, ranked=ranked)

    cutoff_index = max(0, min(len(edges) - 1, int(math.ceil(len(edges) * keep)) - 1))
    threshold = float(np.sort(weights)[::-1][cutoff_index])
    filtered = {edge for edge_pos, edge in enumerate(edges) if weights[edge_pos] >= threshold}
    umst_edges = _union_maximum_spanning_tree(edges=edges, weights=weights, num_nodes=num_nodes)
    selected = filtered.union(umst_edges)
    mask = np.array([edge in selected for edge in edges], dtype=bool)
    return [edge for edge in edges if edge in selected], weights, mask


@register_op
@dataclass(frozen=True)
class ComputeBackbone(Op):
    """Compute backbone sparsification and store the selected graph."""

    config: BackboneConfig = field(default_factory=BackboneConfig)

    name: ClassVar[str] = "backbone_compute"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = ("edge_index", "extras")
    requires: ClassVar[Tuple[str, ...]] = ()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute the backbone edge tensor.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context. Unused.

        Returns
        -------
        SolveState
            State with ``state.edge_index`` replaced by the backbone graph and
            backbone diagnostics stored in ``state.extras``.
        """
        del ctx

        edges, weights, mask = backbone_edge_set(
            edge_index=problem.edge_index,
            num_nodes=problem.num_nodes,
            keep=self.config.keep,
        )
        if edges:
            edge_array = torch.tensor(edges, dtype=torch.long, device=problem.edge_index.device).T
        else:
            edge_array = torch.empty((2, 0), dtype=torch.long, device=problem.edge_index.device)
        state.edge_index = edge_array
        state.extras[_BACKBONE_EDGES_KEY] = edges
        state.extras[_BACKBONE_WEIGHTS_KEY] = weights
        state.extras[_BACKBONE_EDGE_MASK_KEY] = mask
        return state


@register_op
@dataclass(frozen=True)
class RunBackboneStress(Op):
    """Run graphlayouts-style stress on the selected backbone graph."""

    config: BackboneConfig = field(default_factory=BackboneConfig)

    name: ClassVar[str] = "backbone_stress"
    category: ClassVar[OpCategory] = OpCategory.OPTIMIZE
    reads: ClassVar[Tuple[str, ...]] = ("edge_index",)
    writes: ClassVar[Tuple[str, ...]] = ("pos", "distance_matrix")
    requires: ClassVar[Tuple[str, ...]] = ("edge_index",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Lay out the backbone with graphlayouts-compatible stress sweeps.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state containing the backbone edge tensor.
        ctx : RuntimeContext
            Runtime context. Unused.

        Returns
        -------
        SolveState
            State with final positions in ``state.pos``.
        """
        del ctx

        edge_index = state.edge_index if state.edge_index is not None else problem.edge_index
        positions = _layout_components_with_stress(
            edge_index=edge_index,
            num_nodes=problem.num_nodes,
            iterations=self.config.iterations,
            tolerance=self.config.tolerance,
            seed=_R_LAYOUT_SEED,
        )
        device = _layout_device(edge_index=problem.edge_index, node_sizes=problem.node_sizes)
        state.pos = torch.from_numpy(positions).to(device=device, dtype=self.config.dtype)
        return state


def _component_labels(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> Tuple[np.ndarray, List[List[int]]]:
    """Compute weak connected components for an undirected graph.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    tuple[numpy.ndarray, list[list[int]]]
        Component id per node and node ids grouped by component.
    """
    adjacency = [[] for _ in range(num_nodes)]
    for edge_pos in range(int(edge_index.shape[1])):
        source = int(edge_index[0, edge_pos].item())
        target = int(edge_index[1, edge_pos].item())
        adjacency[source].append(target)
        adjacency[target].append(source)

    labels = np.full(num_nodes, -1, dtype=np.int64)
    components: List[List[int]] = []
    for node in range(num_nodes):
        if labels[node] >= 0:
            continue
        label = len(components)
        stack = [node]
        labels[node] = label
        nodes: List[int] = []
        while stack:
            current = stack.pop()
            nodes.append(current)
            for neighbor in adjacency[current]:
                if labels[neighbor] < 0:
                    labels[neighbor] = label
                    stack.append(neighbor)
        components.append(sorted(nodes))
    return labels, components


def _layout_components_with_stress(
    edge_index: torch.Tensor,
    num_nodes: int,
    iterations: int,
    tolerance: float,
    seed: int,
) -> np.ndarray:
    """Layout all components and pack them like graphlayouts.

    Parameters
    ----------
    edge_index : torch.Tensor
        Backbone edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    iterations : int
        Maximum stress iterations.
    tolerance : float
        Relative stress improvement threshold.
    seed : int
        Random seed used for MDS jitter.

    Returns
    -------
    numpy.ndarray
        Position matrix with shape ``[N, 2]``.
    """
    if num_nodes == 0:
        return np.empty((0, 2), dtype=np.float64)
    labels, components = _component_labels(edge_index=edge_index, num_nodes=num_nodes)
    layouts: List[np.ndarray] = []
    node_order: List[int] = []
    for component in components:
        sub_edges = _induced_component_edges(
            edge_index=edge_index,
            component=component,
            labels=labels,
        )
        layout = _layout_single_component(
            edge_index=sub_edges,
            num_nodes=len(component),
            iterations=iterations,
            tolerance=tolerance,
            seed=seed,
        )
        layouts.append(_move_to_origin(layout))
        node_order.extend(component)
    if len(layouts) > 1:
        order = sorted(range(len(layouts)), key=lambda index: len(components[index]))
        layouts = _pack_components(layouts=layouts, order=order, bbox=30.0)
    stacked = np.vstack(layouts)
    result = np.zeros((num_nodes, 2), dtype=np.float64)
    for row, node in enumerate(node_order):
        result[node, :] = stacked[row, :]
    return result


def _induced_component_edges(
    edge_index: torch.Tensor,
    component: List[int],
    labels: np.ndarray,
) -> torch.Tensor:
    """Return component-local edges.

    Parameters
    ----------
    edge_index : torch.Tensor
        Global edge tensor with shape ``[2, E]``.
    component : list[int]
        Global node ids in the component.
    labels : numpy.ndarray
        Component id per global node with shape ``[N]``.

    Returns
    -------
    torch.Tensor
        Local edge tensor with shape ``[2, E_c]``.
    """
    node_to_local = {node: index for index, node in enumerate(component)}
    component_label = int(labels[component[0]])
    local_edges: List[Tuple[int, int]] = []
    for edge_pos in range(int(edge_index.shape[1])):
        source = int(edge_index[0, edge_pos].item())
        target = int(edge_index[1, edge_pos].item())
        if int(labels[source]) == component_label and int(labels[target]) == component_label:
            local_edges.append((node_to_local[source], node_to_local[target]))
    if not local_edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(local_edges, dtype=torch.long).T


def _layout_single_component(
    edge_index: torch.Tensor,
    num_nodes: int,
    iterations: int,
    tolerance: float,
    seed: int,
) -> np.ndarray:
    """Layout one connected component with graphlayouts stress.

    Parameters
    ----------
    edge_index : torch.Tensor
        Component-local edge tensor with shape ``[2, E]``.
    num_nodes : int
        Component node count.
    iterations : int
        Maximum stress iterations.
    tolerance : float
        Relative stress improvement threshold.
    seed : int
        Seed used for deterministic MDS jitter.

    Returns
    -------
    numpy.ndarray
        Component positions with shape ``[N, 2]``.
    """
    if num_nodes == 1:
        return np.zeros((1, 2), dtype=np.float64)
    if num_nodes == 2:
        return np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)

    adjacency = _shared_build_undirected_adjacency(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weights=None,
    )
    distances = _shared_all_pairs_shortest_paths(adjacency, weighted=False).astype(np.float64)
    distances[distances < 0.0] = float(num_nodes)
    np.fill_diagonal(distances, 0.0)
    with np.errstate(divide="ignore"):
        weights = np.where(distances > 0.0, 1.0 / np.square(distances), 0.0)
    np.fill_diagonal(weights, 0.0)

    positions = _mds_initial_positions(distances=distances, seed=seed)
    return _stress_majorization(
        positions=positions,
        weights=weights,
        distances=distances,
        iterations=iterations,
        tolerance=tolerance,
    )


def _mds_initial_positions(distances: np.ndarray, seed: int) -> np.ndarray:
    """Return graphlayouts-style MDS initialization plus jitter.

    Parameters
    ----------
    distances : numpy.ndarray
        Dense graph distances with shape ``[N, N]``.
    seed : int
        Random seed for uniform jitter in ``[-0.1, 0.1]``.

    Returns
    -------
    numpy.ndarray
        Initial positions with shape ``[N, 2]``.
    """
    num_nodes = int(distances.shape[0])
    squared = distances * distances
    centering = np.eye(num_nodes, dtype=np.float64) - (
        np.ones((num_nodes, num_nodes), dtype=np.float64) / float(num_nodes)
    )
    gram = -0.5 * centering @ squared @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    coordinates = np.zeros((num_nodes, 2), dtype=np.float64)
    for out_axis, eig_index in enumerate(order[:2]):
        if eigenvalues[eig_index] > 0.0:
            coordinates[:, out_axis] = eigenvectors[:, eig_index] * math.sqrt(
                eigenvalues[eig_index]
            )
    rng = np.random.default_rng(seed)
    coordinates += rng.uniform(-0.1, 0.1, size=(num_nodes, 2))
    return coordinates


def _stress_value(positions: np.ndarray, weights: np.ndarray, distances: np.ndarray) -> float:
    """Compute graphlayouts' pairwise stress objective.

    Parameters
    ----------
    positions : numpy.ndarray
        Position matrix with shape ``[N, 2]``.
    weights : numpy.ndarray
        Stress weights with shape ``[N, N]``.
    distances : numpy.ndarray
        Target distances with shape ``[N, N]``.

    Returns
    -------
    float
        Sum of weighted squared residuals over unordered pairs.
    """
    total = 0.0
    num_nodes = int(positions.shape[0])
    for source in range(num_nodes - 1):
        for target in range(source + 1, num_nodes):
            delta = positions[source] - positions[target]
            distance = math.sqrt(float(delta[0] * delta[0] + delta[1] * delta[1]))
            residual = distance - float(distances[source, target])
            total += float(weights[source, target]) * residual * residual
    return total


def _stress_majorization(
    positions: np.ndarray,
    weights: np.ndarray,
    distances: np.ndarray,
    iterations: int,
    tolerance: float,
) -> np.ndarray:
    """Run graphlayouts' serial stress-majorization update.

    Parameters
    ----------
    positions : numpy.ndarray
        Initial coordinates with shape ``[N, 2]``.
    weights : numpy.ndarray
        Stress weights with shape ``[N, N]``.
    distances : numpy.ndarray
        Target distances with shape ``[N, N]``.
    iterations : int
        Maximum number of iterations.
    tolerance : float
        Relative stress-improvement threshold.

    Returns
    -------
    numpy.ndarray
        Final coordinates with shape ``[N, 2]``.
    """
    current = positions.astype(np.float64, copy=True)
    row_sums = weights.sum(axis=1)
    stress_old = _stress_value(positions=current, weights=weights, distances=distances)
    for _ in range(iterations):
        candidate = np.zeros_like(current)
        for source in range(current.shape[0]):
            acc_x = 0.0
            acc_y = 0.0
            for target in range(current.shape[0]):
                if source == target:
                    continue
                dx = float(current[source, 0] - current[target, 0])
                dy = float(current[source, 1] - current[target, 1])
                squared = dx * dx + dy * dy
                if squared <= _MIN_STRESS_DISTANCE:
                    continue
                inv_distance = 1.0 / math.sqrt(squared)
                weight = float(weights[source, target])
                target_distance = float(distances[source, target])
                acc_x += weight * (current[target, 0] + target_distance * dx * inv_distance)
                acc_y += weight * (current[target, 1] + target_distance * dy * inv_distance)
            denom = float(row_sums[source])
            candidate[source, 0] = acc_x / denom if denom > 0.0 else current[source, 0]
            candidate[source, 1] = acc_y / denom if denom > 0.0 else current[source, 1]

        stress_new = _stress_value(positions=candidate, weights=weights, distances=distances)
        if stress_old <= 0.0:
            current = candidate
            break
        improvement = (stress_old - stress_new) / stress_old
        current = candidate
        if improvement <= tolerance:
            break
        stress_old = stress_new
    return current


def _move_to_origin(layout: np.ndarray) -> np.ndarray:
    """Translate one component so its minimum corner is at the origin.

    Parameters
    ----------
    layout : numpy.ndarray
        Component positions with shape ``[N, 2]``.

    Returns
    -------
    numpy.ndarray
        Translated component positions.
    """
    if layout.size == 0:
        return layout
    moved = layout.copy()
    moved[:, 0] -= float(np.min(moved[:, 0]))
    moved[:, 1] -= float(np.min(moved[:, 1]))
    return moved


def _pack_components(layouts: List[np.ndarray], order: List[int], bbox: float) -> List[np.ndarray]:
    """Pack disconnected component layouts into rows.

    Parameters
    ----------
    layouts : list[numpy.ndarray]
        Component layouts.
    order : list[int]
        Component processing order.
    bbox : float
        Maximum row width before wrapping.

    Returns
    -------
    list[numpy.ndarray]
        Layouts translated in place-compatible copies.
    """
    packed = [layout.copy() for layout in layouts]
    cur_x = 0.0
    cur_y = 0.0
    max_y = 0.0
    for component_index in order:
        layout = packed[component_index]
        if cur_x + float(np.max(layout[:, 0])) > bbox:
            cur_x = 0.0
            cur_y = max_y + 1.0
        layout[:, 0] += cur_x
        layout[:, 1] += cur_y
        cur_x = float(np.max(layout[:, 0])) + 1.0
        max_y = max(max_y, float(np.max(layout[:, 1])))
    return packed


def build_backbone_pipeline(
    config: Optional[BackboneConfig] = None,
) -> Pipeline:
    """Build the backbone sparsify-then-stress pipeline.

    Parameters
    ----------
    config : BackboneConfig, optional
        Pipeline configuration. Defaults to :class:`BackboneConfig`.

    Returns
    -------
    Pipeline
        Pipeline with explicit backbone and stress stages.
    """
    resolved = BackboneConfig() if config is None else config
    return Pipeline(
        [
            ComputeBackbone(config=resolved),
            RunBackboneStress(config=resolved),
        ],
        name="backbone_pipeline",
    )


def layout_backbone_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    keep: float = _DEFAULT_KEEP,
    iterations: int = _DEFAULT_ITERATIONS,
    steps: Optional[int] = None,
    seed: int = 42,
    tolerance: float = _DEFAULT_TOLERANCE,
    dtype: torch.dtype = torch.float32,
    return_backbone: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Tuple[int, int]]]]:
    """Run the backbone layout pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``. Used only for device
        placement parity with other pipelines.
    keep : float, default=0.2
        Fraction of edges retained by the backbone filter.
    iterations : int, default=500
        Maximum stress iterations.
    steps : int, optional
        Alias for ``iterations`` used by public ``LayoutConfig.steps`` dispatch.
    seed : int, default=42
        Accepted for dispatch compatibility. graphlayouts fixes stress seed to
        42 internally, so this value does not alter the layout.
    tolerance : float, default=0.0001
        Relative stress-improvement stopping threshold.
    dtype : torch.dtype, default=torch.float32
        Output dtype.
    return_backbone : bool, default=False
        Whether to return selected backbone edges with positions.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, list[tuple[int, int]]]
        Final positions with shape ``[N, 2]`` and optionally the selected
        backbone edge set.

    Raises
    ------
    ValueError
        If graph dimensions or configuration values are invalid.
    RuntimeError
        If the pipeline does not produce final positions.
    """
    del seed

    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if steps is not None:
        iterations = int(steps)
    if iterations < 0:
        raise ValueError("iterations must be non-negative.")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")

    device = _layout_device(edge_index=edge_index, node_sizes=node_sizes)
    if num_nodes == 0:
        empty = torch.empty((0, 2), dtype=dtype, device=device)
        return (empty, []) if return_backbone else empty

    config = BackboneConfig(keep=keep, iterations=iterations, tolerance=tolerance, dtype=dtype)
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        seed=_R_LAYOUT_SEED,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_backbone_pipeline(config=config).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("Backbone pipeline did not produce final positions.")
    if return_backbone:
        edges = final_state.extras.get(_BACKBONE_EDGES_KEY, [])
        return final_state.pos, list(edges)
    return final_state.pos


__all__ = [
    "BackboneConfig",
    "ComputeBackbone",
    "RunBackboneStress",
    "backbone_edge_set",
    "build_backbone_pipeline",
    "layout_backbone_pipeline",
]
