"""Classic algorithm loss operations.

This module exposes loss ops that wrap the reference implementations in
``dagua.layout.classic`` so the op vocabulary can reproduce classic
objectives without re-deriving their math inside the ops layer.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dagua.layout.ops.base import LossOp, Op
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_DH_MIN_DISTANCE = 1.0e-3
_DH_COLLINEAR_EPSILON = 1.0e-10


def _dh_layout_extent(num_nodes: int, node_sizes: Optional[torch.Tensor]) -> float:
    """Estimate a stable drawing extent.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Optional node sizes.

    Returns
    -------
    float
        Bounding-box half-width.
    """
    if node_sizes is None or node_sizes.numel() == 0:
        return max(float(max(num_nodes, 1)) ** 0.5 * 5.0, 1.0)

    max_size = float(node_sizes.to(dtype=torch.float32, device="cpu").max().item())
    return max(max_size * max(float(max(num_nodes, 1)) ** 0.5, 1.0) * 2.0, 1.0)


def _dh_unique_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor] = None,
) -> tuple[list[tuple[int, int]], torch.Tensor]:
    """Convert an edge tensor into unique undirected edges and weights.

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
    tuple[list[tuple[int, int]], torch.Tensor]
        Unique undirected edges and their aggregated weights with shape
        ``[E_unique]``.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")

    seen: dict[tuple[int, int], float] = {}
    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    if edge_weights is None:
        weights_cpu = torch.ones((edge_index.shape[1],), dtype=torch.float32)
    else:
        weights_cpu = edge_weights.detach().to(device="cpu", dtype=torch.float32)

    for edge_id, (source, target) in enumerate(
        zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist())
    ):
        if source < 0 or source >= num_nodes or target < 0 or target >= num_nodes:
            raise ValueError("edge_index contains a node index outside [0, num_nodes).")
        if source == target:
            continue
        pair = (min(source, target), max(source, target))
        seen[pair] = seen.get(pair, 0.0) + float(weights_cpu[edge_id].item())

    ordered_edges = sorted(seen)
    ordered_weights = torch.tensor([seen[edge] for edge in ordered_edges], dtype=torch.float32)
    return ordered_edges, ordered_weights


def _dh_orientation(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> float:
    """Compute the signed triangle area used by segment intersection tests."""
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _dh_segments_intersect(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor
) -> bool:
    """Test whether two line segments intersect."""
    o1 = _dh_orientation(a, b, c)
    o2 = _dh_orientation(a, b, d)
    o3 = _dh_orientation(c, d, a)
    o4 = _dh_orientation(c, d, b)
    return (
        abs(o1) < _DH_COLLINEAR_EPSILON or abs(o2) < _DH_COLLINEAR_EPSILON or o1 * o2 < 0.0
    ) and (abs(o3) < _DH_COLLINEAR_EPSILON or abs(o4) < _DH_COLLINEAR_EPSILON or o3 * o4 < 0.0)


def _dh_point_segment_distance(
    point: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
) -> torch.Tensor:
    """Compute the distance from a point to a segment."""
    segment = end - start
    denom = segment.dot(segment).clamp(min=_DH_MIN_DISTANCE)
    projection = ((point - start).dot(segment) / denom).clamp(0.0, 1.0)
    nearest = start + projection * segment
    return torch.linalg.norm(point - nearest)


def _dh_scale_denominator(numerator_count: int) -> float:
    """Return a non-zero normalization denominator for one energy term."""
    return float(max(numerator_count, 1))


_dh = SimpleNamespace(
    _MIN_DISTANCE=_DH_MIN_DISTANCE,
    _layout_extent=_dh_layout_extent,
    _unique_edges=_dh_unique_edges,
    _segments_intersect=_dh_segments_intersect,
    _point_segment_distance=_dh_point_segment_distance,
    _scale_denominator=_dh_scale_denominator,
)


_LINLOG_FULL_REPULSION_LIMIT = 2_000
_LINLOG_SAMPLED_REPULSION_NEIGHBORS = 128
_LINLOG_MIN_DISTANCE = 1.0e-3


def _linlog_full_all_pairs(
    num_nodes: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Enumerate all unordered node pairs for exact repulsion."""
    return torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)


def _linlog_sample_all_pairs(
    num_nodes: int,
    device: torch.device,
    step: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Draw deterministic node pairs for sampled repulsion."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + step + 1)
    total_pairs = max(num_nodes * (num_nodes - 1) // 2, 0)
    sample_size = min(
        total_pairs,
        max(num_nodes, num_nodes * _LINLOG_SAMPLED_REPULSION_NEIGHBORS // 2),
    )
    if sample_size == 0:
        empty = torch.empty((0,), dtype=torch.long, device=device)
        return empty, empty, total_pairs
    if sample_size == total_pairs:
        full_src, full_dst = _linlog_full_all_pairs(num_nodes=num_nodes, device=device)
        return full_src, full_dst, total_pairs

    sampled_src: list[torch.Tensor] = []
    sampled_dst: list[torch.Tensor] = []
    collected = 0

    for _ in range(8):
        if collected >= sample_size:
            break

        draw_size = max((sample_size - collected) * 3, num_nodes)
        src = torch.randint(0, num_nodes, (draw_size,), generator=generator, dtype=torch.long)
        dst = torch.randint(0, num_nodes, (draw_size,), generator=generator, dtype=torch.long)
        distinct_mask = src != dst
        if not bool(distinct_mask.any()):
            continue

        ordered_src = torch.minimum(src[distinct_mask], dst[distinct_mask])
        ordered_dst = torch.maximum(src[distinct_mask], dst[distinct_mask])
        if ordered_src.numel() == 0:
            continue

        take_count = min(int(ordered_src.numel()), sample_size - collected)
        sampled_src.append(ordered_src[:take_count])
        sampled_dst.append(ordered_dst[:take_count])
        collected += take_count

    if collected == 0:
        empty = torch.empty((0,), dtype=torch.long, device=device)
        return empty, empty, total_pairs

    return (
        torch.cat(sampled_src).to(device=device),
        torch.cat(sampled_dst).to(device=device),
        total_pairs,
    )


def _linlog_linlog_loss(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    seed: int,
    step: int,
    a: float = 1.0,
    r: float = 0.0,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Evaluate the LinLog objective with all-pairs repulsion."""
    attraction = torch.tensor(0.0, dtype=positions.dtype, device=positions.device)
    if edge_index.numel() > 0:
        src = edge_index[0].to(device=positions.device, dtype=torch.long)
        dst = edge_index[1].to(device=positions.device, dtype=torch.long)
        edge_lengths = torch.linalg.norm(positions[src] - positions[dst], dim=1).clamp(
            min=_LINLOG_MIN_DISTANCE
        )
        if edge_weights is not None:
            weights = edge_weights.to(device=positions.device, dtype=edge_lengths.dtype)
            attraction = (weights * edge_lengths.pow(a)).sum()
        else:
            attraction = edge_lengths.pow(a).sum()

    num_nodes = int(positions.shape[0])
    if num_nodes <= _LINLOG_FULL_REPULSION_LIMIT:
        pair_src, pair_dst = _linlog_full_all_pairs(
            num_nodes=num_nodes,
            device=positions.device,
        )
        if int(pair_src.numel()) == 0:
            repulsion = torch.tensor(0.0, dtype=positions.dtype, device=positions.device)
        else:
            pairwise_distances = torch.linalg.norm(
                positions[pair_src] - positions[pair_dst],
                dim=1,
            ).clamp(min=_LINLOG_MIN_DISTANCE)
            if r == 0.0:
                repulsion = -torch.log(pairwise_distances).sum()
            else:
                repulsion = -pairwise_distances.pow(r).sum()
    else:
        src, dst, total_pairs = _linlog_sample_all_pairs(
            num_nodes=num_nodes,
            device=positions.device,
            step=step,
            seed=seed,
        )
        sampled_lengths = torch.linalg.norm(
            positions[src] - positions[dst],
            dim=1,
        ).clamp(min=_LINLOG_MIN_DISTANCE)
        if int(sampled_lengths.numel()) == 0:
            repulsion = torch.tensor(0.0, dtype=positions.dtype, device=positions.device)
        elif r == 0.0:
            repulsion = -torch.log(sampled_lengths).mean() * float(total_pairs)
        else:
            repulsion = -sampled_lengths.pow(r).mean() * float(total_pairs)

    return attraction + repulsion


_linlog = SimpleNamespace(
    _FULL_REPULSION_LIMIT=_LINLOG_FULL_REPULSION_LIMIT,
    _SAMPLED_REPULSION_NEIGHBORS=_LINLOG_SAMPLED_REPULSION_NEIGHBORS,
    _MIN_DISTANCE=_LINLOG_MIN_DISTANCE,
    _full_all_pairs=_linlog_full_all_pairs,
    _linlog_loss=_linlog_linlog_loss,
)


_MAXENT_MIN_DISTANCE = 1.0e-3
_MAXENT_FULL_STRESS_LIMIT = 1_000
_MAXENT_MAJORIZATION_NODE_LIMIT = 5_000
_MAXENT_PIVOT_COUNT = 50
_MAXENT_SAMPLED_REPULSION_NEIGHBORS = 96


def _maxent_build_undirected_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor] = None,
) -> list[list[tuple[int, float]]]:
    """Build an undirected adjacency list from the edge tensor."""
    adjacency: list[dict[int, float]] = [{} for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return [[] for _ in range(num_nodes)]

    ei_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    sources = ei_cpu[0].tolist()
    targets = ei_cpu[1].tolist()
    if edge_weights is not None:
        weights = edge_weights.detach().cpu().float().tolist()
    else:
        weights = [1.0] * len(sources)

    for src, tgt, weight in zip(sources, targets, weights):
        if tgt not in adjacency[src] or weight < adjacency[src][tgt]:
            adjacency[src][tgt] = float(weight)
        if src not in adjacency[tgt] or weight < adjacency[tgt][src]:
            adjacency[tgt][src] = float(weight)

    return [sorted(neighbors.items()) for neighbors in adjacency]


def _maxent_full_non_edge_pairs(
    adjacency: list[list[tuple[int, float]]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Enumerate all unique non-edge pairs for exact entropy repulsion."""
    num_nodes = len(adjacency)
    if num_nodes <= 1:
        empty = torch.empty((0,), dtype=torch.long)
        return empty, empty

    adjacency_mask = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    for source, neighbors in enumerate(adjacency):
        if neighbors:
            neighbor_indices = torch.tensor(
                [neighbor for neighbor, _ in neighbors], dtype=torch.long
            )
            adjacency_mask[source, neighbor_indices] = True

    upper = torch.triu_indices(num_nodes, num_nodes, offset=1)
    mask = ~adjacency_mask[upper[0], upper[1]]
    return upper[0][mask], upper[1][mask]


def _maxent_stress_term(
    positions: torch.Tensor,
    stress_src: torch.Tensor,
    stress_dst: torch.Tensor,
    stress_lengths: torch.Tensor,
    pivot_indices: torch.Tensor,
    pivot_distances: torch.Tensor,
) -> torch.Tensor:
    """Evaluate either exact or pivot-approximated stress."""
    if stress_src.numel() > 0:
        src = stress_src.to(device=positions.device)
        dst = stress_dst.to(device=positions.device)
        targets = stress_lengths.to(device=positions.device)
        distances = torch.linalg.norm(positions[src] - positions[dst], dim=1).clamp(
            min=_MAXENT_MIN_DISTANCE
        )
        weights = targets.reciprocal().square()
        return (weights * (distances - targets).square()).sum()

    if pivot_indices.numel() == 0:
        return torch.tensor(0.0, dtype=positions.dtype, device=positions.device)

    pivot_positions = positions[pivot_indices.to(device=positions.device)]
    geometric = torch.cdist(positions, pivot_positions).clamp(min=_MAXENT_MIN_DISTANCE)
    targets = pivot_distances.to(device=positions.device)
    reachable = targets > 0
    safe_targets = torch.where(reachable, targets, torch.ones_like(targets))
    weights = torch.where(reachable, safe_targets.reciprocal().square(), torch.zeros_like(targets))
    return (weights * (geometric - safe_targets).square()).sum()


def _maxent_entropy_term(
    positions: torch.Tensor,
    non_edge_src: torch.Tensor,
    non_edge_dst: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Evaluate the non-edge logarithmic entropy term."""
    if non_edge_src.numel() == 0:
        return torch.tensor(0.0, dtype=positions.dtype, device=positions.device)

    nonedge_distances = torch.linalg.norm(
        positions[non_edge_src] - positions[non_edge_dst],
        dim=1,
    ).clamp(min=_MAXENT_MIN_DISTANCE)
    return -torch.log(nonedge_distances).sum() * scale


_maxent = SimpleNamespace(
    _MIN_DISTANCE=_MAXENT_MIN_DISTANCE,
    _FULL_STRESS_LIMIT=_MAXENT_FULL_STRESS_LIMIT,
    _MAJORIZATION_NODE_LIMIT=_MAXENT_MAJORIZATION_NODE_LIMIT,
    _PIVOT_COUNT=_MAXENT_PIVOT_COUNT,
    _SAMPLED_REPULSION_NEIGHBORS=_MAXENT_SAMPLED_REPULSION_NEIGHBORS,
    _build_undirected_adjacency=_maxent_build_undirected_adjacency,
    _full_non_edge_pairs=_maxent_full_non_edge_pairs,
    _stress_term=_maxent_stress_term,
    _entropy_term=_maxent_entropy_term,
)


_TSNET_MIN_DISTANCE = 1.0e-12


def _tsnet_row_probabilities(distances: torch.Tensor, perplexity: float) -> torch.Tensor:
    """Match one row's Gaussian bandwidth to a target perplexity."""
    num_nodes = int(distances.shape[0])
    if num_nodes <= 1:
        return torch.zeros_like(distances)

    mask = torch.ones(num_nodes, dtype=torch.bool)
    mask[int(torch.argmin(distances).item())] = False
    squared = distances.square()

    beta = torch.tensor(1.0, dtype=torch.float32)
    beta_min = torch.tensor(float("-inf"), dtype=torch.float32)
    beta_max = torch.tensor(float("inf"), dtype=torch.float32)
    target_entropy = torch.log(torch.tensor(perplexity, dtype=torch.float32))

    probabilities = torch.zeros_like(distances)
    for _ in range(100):
        weights = torch.exp(-squared * beta) * mask.to(dtype=torch.float32)
        weights_sum = weights.sum().clamp(min=_TSNET_MIN_DISTANCE)
        probabilities = weights / weights_sum
        entropy = -(
            probabilities[mask] * probabilities[mask].clamp(min=_TSNET_MIN_DISTANCE).log()
        ).sum()
        error = entropy - target_entropy
        if torch.abs(error) < 1.0e-5:
            break
        if error > 0:
            beta_min = beta
            beta = beta * 2.0 if torch.isinf(beta_max) else (beta + beta_max) * 0.5
        else:
            beta_max = beta
            beta = beta * 0.5 if torch.isinf(beta_min) else (beta + beta_min) * 0.5

    probabilities[int(torch.argmin(distances).item())] = 0.0
    return probabilities


def _tsnet_high_dimensional_affinities(
    distance_matrix: torch.Tensor, perplexity: float
) -> torch.Tensor:
    """Build the symmetric t-SNE input affinity matrix."""
    rows = [
        _tsnet_row_probabilities(distance_matrix[node], perplexity)
        for node in range(distance_matrix.shape[0])
    ]
    conditional = torch.stack(rows, dim=0)
    symmetrized = (conditional + conditional.transpose(0, 1)) / (
        2.0 * max(distance_matrix.shape[0], 1)
    )
    return symmetrized.clamp(min=_TSNET_MIN_DISTANCE)


def _tsnet_tsne_loss(positions: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
    """Evaluate the exact t-SNE KL objective."""
    delta = positions.unsqueeze(1) - positions.unsqueeze(0)
    squared_distances = delta.square().sum(dim=2)
    numerators = (1.0 + squared_distances).reciprocal()
    diagonal_mask = ~torch.eye(
        positions.shape[0],
        dtype=torch.bool,
        device=positions.device,
    )
    numerators = numerators * diagonal_mask.to(dtype=numerators.dtype)
    q = numerators / numerators.sum().clamp(min=_TSNET_MIN_DISTANCE)
    return (probabilities * (probabilities.log() - q.clamp(min=_TSNET_MIN_DISTANCE).log())).sum()


_tsnet = SimpleNamespace(
    _MIN_DISTANCE=_TSNET_MIN_DISTANCE,
    _high_dimensional_affinities=_tsnet_high_dimensional_affinities,
    _tsne_loss=_tsnet_tsne_loss,
)


_UMAP_EPSILON = 1.0e-9


def _umap_fit_ab(min_dist: float, spread: float) -> tuple[float, float]:
    """Fit UMAP curve parameters from distance span values."""
    xv = np.linspace(0.0, 3.0 * spread, 300)
    yv = np.where(xv < min_dist, 1.0, np.exp(-(xv - min_dist) / spread))

    def _curve_function(x: np.ndarray, a: float, b: float) -> np.ndarray:
        return 1.0 / (1.0 + (a * np.power(x, 2.0 * b)))

    try:
        from scipy import optimize

        params, _ = optimize.curve_fit(
            _curve_function,
            xv,
            yv,
            p0=(1.93, 0.79),
            maxfev=10_000,
        )
        return float(params[0]), float(params[1])
    except (RuntimeError, ValueError):
        return 1.93, 0.79


_umap = SimpleNamespace(
    _EPSILON=_UMAP_EPSILON,
    _fit_ab=_umap_fit_ab,
)


_NEULAY_PAIR_QUERY_RADIUS_FACTOR = 4.0


def _neulay_kdtree_repulsion_pairs(pos: torch.Tensor, query_radius: float) -> np.ndarray:
    """Find nearby node pairs using SciPy's cKDTree."""
    from scipy.spatial import cKDTree

    if pos.shape[0] < 2:
        return np.empty((0, 2), dtype=np.int64)
    tree = cKDTree(pos.detach().cpu().numpy())
    pairs = tree.query_pairs(query_radius, output_type="ndarray")
    if pairs.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    return pairs.astype(np.int64)


def _neulay_kdtree_repulsion_loss(
    pos: torch.Tensor,
    pairs: np.ndarray,
    radius: float,
    magnitude: float,
) -> torch.Tensor:
    """Evaluate Gaussian repulsion over cached cKDTree pairs."""
    if pairs.shape[0] == 0 or magnitude == 0.0:
        return pos.sum() * 0.0
    idx = torch.from_numpy(pairs).to(device=pos.device)
    sq_dist = ((pos[idx[:, 0]] - pos[idx[:, 1]]) ** 2).sum(dim=-1)
    return magnitude * torch.exp(-sq_dist / (4.0 * radius * radius)).sum()


def _neulay_elastic_loss(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Evaluate the NeuLay elastic energy."""
    if edge_index.numel() == 0:
        return pos.sum() * 0.0
    # Match the reference by collapsing directed duplicates into one
    # undirected spring before measuring the elastic energy.
    src, dst = edge_index[0], edge_index[1]
    low = torch.minimum(src, dst)
    high = torch.maximum(src, dst)
    pairs = torch.stack([low, high], dim=0)
    unique_pairs = torch.unique(pairs, dim=1)
    diff = pos[unique_pairs[0]] - pos[unique_pairs[1]]
    return diff.square().sum() * 0.5


_neulay = SimpleNamespace(
    _PAIR_QUERY_RADIUS_FACTOR=_NEULAY_PAIR_QUERY_RADIUS_FACTOR,
    _kdtree_repulsion_pairs=_neulay_kdtree_repulsion_pairs,
    _kdtree_repulsion_loss=_neulay_kdtree_repulsion_loss,
    _elastic_loss=_neulay_elastic_loss,
)


_SGD2_EPS = 1.0e-6
_SGD2_SEGMENT_EPS = 1.0e-9
_SGD2_DEFAULT_IDEAL_EDGE_LENGTH = 1.0
_SGD2_DEFAULT_ASPECT_RATIO_TARGET = 1.0
_SGD2_VERTEX_RESOLUTION_SMOOTHNESS = 0.1
_SGD2_NEIGHBORHOOD_DEPTH_LIMIT = 2
_SGD2_NEIGHBORHOOD_NEG_SAMPLE_RATE = 0.5
_SGD2_NEIGHBORHOOD_K_DIST = 1.5
_SGD2_CROSSING_DETECTOR_TRAIN_STEPS = 2
_SGD2_CROSSING_DETECTOR_LR = 0.01


def _sgd2_graph_distances_bfs_distances(
    adjacency: list[list[tuple[int, float]]],
    source: int,
) -> np.ndarray:
    """Compute unweighted shortest-path distances from one source via BFS."""
    num_nodes = len(adjacency)
    distances = np.full(num_nodes, -1, dtype=np.int32)
    distances[source] = 0
    frontier: deque[int] = deque([source])

    while frontier:
        node = frontier.popleft()
        next_distance = int(distances[node]) + 1
        for neighbor, _ in adjacency[node]:
            if int(distances[neighbor]) != -1:
                continue
            distances[neighbor] = next_distance
            frontier.append(neighbor)

    return distances


def _sgd2_graph_distances_dijkstra_distances(
    adjacency: list[list[tuple[int, float]]],
    source: int,
) -> np.ndarray:
    """Compute weighted shortest-path distances from one source via Dijkstra."""
    num_nodes = len(adjacency)
    distances = np.full(num_nodes, np.inf, dtype=np.float64)
    distances[source] = 0.0
    visited = np.zeros(num_nodes, dtype=np.bool_)
    queue: list[tuple[float, int]] = [(0.0, source)]

    while queue:
        dist, node = heapq.heappop(queue)
        if visited[node]:
            continue
        visited[node] = True
        for neighbor, weight in adjacency[node]:
            new_dist = dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(queue, (new_dist, neighbor))

    return distances


def _sgd2_graph_distances_all_pairs_shortest_paths(
    adjacency: list[list[tuple[int, float]]],
    weighted: bool,
) -> np.ndarray:
    """Compute full all-pairs shortest-path distances."""
    num_nodes = len(adjacency)
    if weighted:
        distances = np.full((num_nodes, num_nodes), np.inf, dtype=np.float64)
        for source in range(num_nodes):
            distances[source] = _sgd2_graph_distances_dijkstra_distances(adjacency, source)
    else:
        distances = np.full((num_nodes, num_nodes), -1, dtype=np.int32)
        for source in range(num_nodes):
            distances[source] = _sgd2_graph_distances_bfs_distances(adjacency, source)
    return distances


def _sgd2_build_undirected_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor] = None,
) -> list[list[tuple[int, float]]]:
    """Build a deterministic undirected adjacency list."""
    adjacency: list[dict[int, float]] = [{} for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return [[] for _ in range(num_nodes)]

    ei_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    sources = ei_cpu[0].tolist()
    targets = ei_cpu[1].tolist()
    if edge_weights is not None:
        weights = edge_weights.detach().cpu().float().tolist()
    else:
        weights = [1.0] * len(sources)

    for src, tgt, weight in zip(sources, targets, weights):
        if tgt not in adjacency[src] or weight < adjacency[src][tgt]:
            adjacency[src][tgt] = float(weight)
        if src not in adjacency[tgt] or weight < adjacency[tgt][src]:
            adjacency[tgt][src] = float(weight)

    return [sorted(neighbors.items()) for neighbors in adjacency]


def _sgd2_build_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor] = None,
) -> list[list[tuple[int, float]]]:
    """Build a deterministic undirected adjacency list."""
    return _sgd2_build_undirected_adjacency(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weights=edge_weights,
    )


def _sgd2_all_pairs_shortest_paths(
    adjacency: list[list[tuple[int, float]]],
    device: torch.device,
    weighted: bool,
) -> torch.Tensor:
    """Compute the full all-pairs distance matrix."""
    distances = _sgd2_graph_distances_all_pairs_shortest_paths(
        adjacency=adjacency,
        weighted=weighted,
    )
    cleaned = distances.astype(np.float64, copy=False)
    if not weighted:
        cleaned = cleaned.copy()
        cleaned[cleaned < 0] = np.inf
    return torch.tensor(cleaned, dtype=torch.float32, device=device)


def _sgd2_build_stress_terms(
    distances: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert the distance matrix into sampled stress pairs and weights."""
    upper = torch.triu_indices(
        distances.shape[0],
        distances.shape[1],
        offset=1,
        device=distances.device,
    )
    upper_distances = distances[upper[0], upper[1]]
    finite_mask = torch.isfinite(upper_distances)
    pairs = upper[:, finite_mask]
    positive_distances = upper_distances[finite_mask]
    weights = 1.0 / (positive_distances.square() + _SGD2_EPS)
    return pairs, positive_distances, weights


def _sgd2_build_incident_edge_pairs(
    adjacency: list[list[tuple[int, float]]],
    device: torch.device,
) -> torch.Tensor:
    """Build sampled incident-edge tuples for angular resolution."""
    pairs: list[tuple[int, int, int, int, int]] = []
    for node, neighbors in enumerate(adjacency):
        degree = len(neighbors)
        if degree < 2:
            continue
        neighbor_ids = [neighbor for neighbor, _ in neighbors]
        for left_index in range(degree - 1):
            for right_index in range(left_index + 1, degree):
                pairs.append(
                    (
                        degree,
                        node,
                        neighbor_ids[left_index],
                        node,
                        neighbor_ids[right_index],
                    )
                )
    if not pairs:
        return torch.empty((5, 0), dtype=torch.long, device=device)
    return torch.tensor(pairs, dtype=torch.long, device=device).transpose(0, 1).contiguous()


def _sgd2_build_non_incident_edge_pairs(edges: torch.Tensor) -> torch.Tensor:
    """Build all non-incident undirected edge pairs."""
    edge_count = edges.shape[1]
    if edge_count < 2:
        return torch.empty((4, 0), dtype=torch.long, device=edges.device)

    pair_indices = torch.triu_indices(edge_count, edge_count, offset=1, device=edges.device)
    left = edges[:, pair_indices[0]]
    right = edges[:, pair_indices[1]]
    non_incident = (
        (left[0] != right[0])
        & (left[0] != right[1])
        & (left[1] != right[0])
        & (left[1] != right[1])
    )
    if not bool(non_incident.any().item()):
        return torch.empty((4, 0), dtype=torch.long, device=edges.device)
    return torch.cat([left[:, non_incident], right[:, non_incident]], dim=0)


def _sgd2_clean_undirected_edges(edge_index: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Collapse the input edge list into unique undirected edges."""
    edges = edge_index.to(device=device, dtype=torch.long)
    if edges.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    src = edges[0]
    dst = edges[1]
    non_self = src != dst
    src = src[non_self]
    dst = dst[non_self]
    if src.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    lower = torch.minimum(src, dst)
    upper = torch.maximum(src, dst)
    unique_pairs = torch.unique(torch.stack([lower, upper], dim=1), dim=0)
    return unique_pairs.transpose(0, 1).contiguous()


def _sgd2_prepare_state(
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device,
    needs_distances: bool,
    needs_incident_edge_pairs: bool,
    needs_non_incident_edge_pairs: bool,
    edge_weights: Optional[torch.Tensor],
) -> _sgd2_PreparedState:
    """Precompute graph state shared by all SGD2 criteria."""
    edges = _sgd2_clean_undirected_edges(edge_index=edge_index, device=device)
    adjacency = _sgd2_build_adjacency(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weights=edge_weights,
    )
    incident_edge_pairs = (
        _sgd2_build_incident_edge_pairs(adjacency=adjacency, device=device)
        if needs_incident_edge_pairs
        else None
    )
    non_incident_edge_pairs = (
        _sgd2_build_non_incident_edge_pairs(edges=edges) if needs_non_incident_edge_pairs else None
    )
    if not needs_distances:
        return _sgd2_PreparedState(
            device=device,
            edges=edges,
            adjacency=adjacency,
            all_pairs_distances=None,
            stress_pairs=None,
            stress_distances=None,
            stress_weights=None,
            incident_edge_pairs=incident_edge_pairs,
            non_incident_edge_pairs=non_incident_edge_pairs,
        )

    distances = _sgd2_all_pairs_shortest_paths(
        adjacency=adjacency,
        device=device,
        weighted=edge_weights is not None,
    )
    stress_pairs, stress_distances, stress_weights = _sgd2_build_stress_terms(distances)
    return _sgd2_PreparedState(
        device=device,
        edges=edges,
        adjacency=adjacency,
        all_pairs_distances=distances,
        stress_pairs=stress_pairs,
        stress_distances=stress_distances,
        stress_weights=stress_weights,
        incident_edge_pairs=incident_edge_pairs,
        non_incident_edge_pairs=non_incident_edge_pairs,
    )


def _sgd2_sample_indices(total: int, batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample indices with replacement from a finite range."""
    if total <= 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    return torch.randint(0, total, (batch_size,), device=device)


def _sgd2_sample_nodes(num_nodes: int, batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample a node mini-batch."""
    return _sgd2_sample_indices(total=num_nodes, batch_size=batch_size, device=device)


def _sgd2_sample_pairs(pairs: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Sample a mini-batch of node pairs."""
    if pairs.numel() == 0:
        return pairs
    indices = _sgd2_sample_indices(total=pairs.shape[1], batch_size=batch_size, device=pairs.device)
    return pairs[:, indices]


def _sgd2_sample_edges(edges: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Sample a mini-batch of edges."""
    return _sgd2_sample_pairs(pairs=edges, batch_size=batch_size)


def _sgd2_sample_edge_pairs(
    edges: torch.Tensor, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample non-incident edge pairs for crossing-related criteria."""
    if edges.shape[1] < 2:
        empty = torch.empty((2, 0), dtype=torch.long, device=edges.device)
        return empty, empty

    edge_count = edges.shape[1]
    left_batches: list[torch.Tensor] = []
    right_batches: list[torch.Tensor] = []
    collected = 0
    while collected < batch_size:
        take = max(batch_size - collected, batch_size)
        left_idx = _sgd2_sample_indices(edge_count, take, edges.device)
        right_idx = _sgd2_sample_indices(edge_count, take, edges.device)
        left = edges[:, left_idx]
        right = edges[:, right_idx]
        non_self = left_idx != right_idx
        non_incident = (
            (left[0] != right[0])
            & (left[0] != right[1])
            & (left[1] != right[0])
            & (left[1] != right[1])
        )
        mask = non_self & non_incident
        if not bool(mask.any().item()):
            if edge_count <= 2:
                break
            continue
        left_batches.append(left[:, mask])
        right_batches.append(right[:, mask])
        collected += int(mask.sum().item())
        if edge_count <= 2:
            break

    if len(left_batches) == 0:
        empty = torch.empty((2, 0), dtype=torch.long, device=edges.device)
        return empty, empty

    left_cat = torch.cat(left_batches, dim=1)[:, :batch_size]
    right_cat = torch.cat(right_batches, dim=1)[:, :batch_size]
    return left_cat, right_cat


class _sgd2_CyclicSampler:
    """Epoch-based mini-batch sampler matching the reference DataLoader."""

    __slots__ = ("_total", "_device", "_perm", "_offset")

    def __init__(self, total: int, device: torch.device) -> None:
        self._total = total
        self._device = device
        self._perm = torch.randperm(total, device=device)
        self._offset = 0

    def sample(self, batch_size: int) -> torch.Tensor:
        """Return the next ``batch_size`` indices, reshuffling on epoch boundary."""
        if self._total <= 0:
            return torch.empty((0,), dtype=torch.long, device=self._device)
        bs = min(batch_size, self._total)
        if self._offset + bs > self._total:
            self._perm = torch.randperm(self._total, device=self._device)
            self._offset = 0
        out = self._perm[self._offset : self._offset + bs]
        self._offset += bs
        return out


def _sgd2_stress_loss(
    pos: torch.Tensor,
    pair_batch: torch.Tensor,
    pair_distances: torch.Tensor,
    pair_weights: torch.Tensor,
) -> torch.Tensor:
    """Evaluate stress over a sampled pair batch."""
    if pair_batch.numel() == 0:
        return pos.sum() * 0.0
    lengths = torch.linalg.norm(pos[pair_batch[0]] - pos[pair_batch[1]], dim=1)
    return (pair_weights * (lengths - pair_distances).square()).mean()


def _sgd2_ideal_edge_length_loss(
    pos: torch.Tensor,
    edge_batch: torch.Tensor,
    target: float,
) -> torch.Tensor:
    """Evaluate the ideal-edge-length criterion."""
    if edge_batch.numel() == 0:
        return pos.sum() * 0.0
    lengths = torch.linalg.norm(pos[edge_batch[0]] - pos[edge_batch[1]], dim=1)
    safe_target = max(target, _SGD2_EPS)
    return (((lengths - safe_target) / safe_target).square()).mean()


def _sgd2_lovasz_grad(labels_sorted: torch.Tensor) -> torch.Tensor:
    """Compute the Lovasz-extension gradient for binary labels."""
    positives = labels_sorted.sum()
    intersection = positives - labels_sorted.cumsum(dim=0)
    union = positives + (1.0 - labels_sorted).cumsum(dim=0)
    jaccard = 1.0 - intersection / union.clamp(min=_SGD2_EPS)
    if labels_sorted.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def _sgd2_lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Evaluate the binary Lovasz hinge loss on flattened inputs."""
    if logits.numel() == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, permutation = torch.sort(errors, descending=True)
    labels_sorted = labels[permutation]
    return torch.dot(F.relu(errors_sorted), _sgd2_lovasz_grad(labels_sorted))


def _sgd2_neighborhood_preservation_loss(
    pos: torch.Tensor,
    anchor_nodes: torch.Tensor,
    adjacency: list[list[tuple[int, float]]],
) -> torch.Tensor:
    """Evaluate the reference BFS-based neighborhood-preservation loss."""
    sampled_nodes = _sgd2_sample_neighborhood_nodes(
        adjacency=adjacency,
        root_nodes=anchor_nodes,
        num_nodes=pos.shape[0],
        device=pos.device,
    )
    if sampled_nodes.numel() == 0:
        return pos.sum() * 0.0

    sampled_pos = pos[sampled_nodes]
    logits = -torch.cdist(sampled_pos, sampled_pos) + _SGD2_NEIGHBORHOOD_K_DIST
    labels = _sgd2_induced_adjacency_target(
        sampled_nodes=sampled_nodes,
        adjacency=adjacency,
        device=pos.device,
        dtype=pos.dtype,
    )
    return _sgd2_lovasz_hinge_flat(logits.reshape(-1), labels.reshape(-1))


def _sgd2_bfs_nodes(
    adjacency: list[list[tuple[int, float]]],
    root: int,
    depth_limit: int,
) -> list[int]:
    """Collect nodes from a bounded BFS rooted at one node."""
    visited = {root}
    queue: list[tuple[int, int]] = [(root, 0)]
    ordered = [root]
    queue_index = 0
    while queue_index < len(queue):
        node, depth = queue[queue_index]
        queue_index += 1
        if depth >= depth_limit:
            continue
        for neighbor, _weight in adjacency[node]:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            ordered.append(neighbor)
            queue.append((neighbor, depth + 1))
    return ordered


def _sgd2_sample_neighborhood_nodes(
    adjacency: list[list[tuple[int, float]]],
    root_nodes: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    """Build the sampled subgraph used by neighborhood preservation."""
    if root_nodes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=device)

    positive_nodes: list[int] = []
    for root in root_nodes.tolist():
        positive_nodes.extend(
            _sgd2_bfs_nodes(
                adjacency=adjacency,
                root=root,
                depth_limit=_SGD2_NEIGHBORHOOD_DEPTH_LIMIT,
            )
        )
    if not positive_nodes:
        return torch.empty((0,), dtype=torch.long, device=device)

    positive = torch.tensor(sorted(set(positive_nodes)), dtype=torch.long, device=device)
    negative_count = int(_SGD2_NEIGHBORHOOD_NEG_SAMPLE_RATE * int(positive.numel()))
    if negative_count <= 0:
        return positive

    negatives = torch.randint(0, num_nodes, (negative_count,), device=device)
    return torch.unique(torch.cat([positive, negatives]), sorted=True)


def _sgd2_induced_adjacency_target(
    sampled_nodes: torch.Tensor,
    adjacency: list[list[tuple[int, float]]],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the induced adjacency-plus-identity target matrix."""
    size = int(sampled_nodes.numel())
    target = torch.eye(size, dtype=dtype, device=device)
    local_index = {int(node): offset for offset, node in enumerate(sampled_nodes.tolist())}
    for row, node in enumerate(sampled_nodes.tolist()):
        for neighbor, _weight in adjacency[node]:
            column = local_index.get(neighbor)
            if column is not None:
                target[row, column] = 1.0
    return target


def _sgd2_cross2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute 2D cross products row-wise."""
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


def _sgd2_edge_pair_positions(
    pos: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Gather flattened coordinates for one batch of edge pairs."""
    if left.numel() == 0 or right.numel() == 0:
        return torch.empty((0, 8), dtype=pos.dtype, device=pos.device)
    pair_indices = torch.stack([left[0], left[1], right[0], right[1]], dim=1)
    return pos[pair_indices].reshape(-1, 8)


def _sgd2_point_on_segment(
    point: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
) -> torch.Tensor:
    """Test whether points lie on closed 2D segments."""
    min_x = torch.minimum(start[:, 0], end[:, 0]) - _SGD2_SEGMENT_EPS
    max_x = torch.maximum(start[:, 0], end[:, 0]) + _SGD2_SEGMENT_EPS
    min_y = torch.minimum(start[:, 1], end[:, 1]) - _SGD2_SEGMENT_EPS
    max_y = torch.maximum(start[:, 1], end[:, 1]) + _SGD2_SEGMENT_EPS
    return (
        (point[:, 0] >= min_x)
        & (point[:, 0] <= max_x)
        & (point[:, 1] >= min_y)
        & (point[:, 1] <= max_y)
    )


def _sgd2_are_edge_pairs_crossed(edge_pair_pos: torch.Tensor) -> torch.Tensor:
    """Compute exact geometric crossing labels for disjoint edge pairs."""
    if edge_pair_pos.numel() == 0:
        return torch.empty((0,), dtype=torch.bool, device=edge_pair_pos.device)

    segments = edge_pair_pos.reshape(-1, 4, 2)
    a = segments[:, 0]
    b = segments[:, 1]
    c = segments[:, 2]
    d = segments[:, 3]

    orient_abc = _sgd2_cross2d(b - a, c - a)
    orient_abd = _sgd2_cross2d(b - a, d - a)
    orient_cda = _sgd2_cross2d(d - c, a - c)
    orient_cdb = _sgd2_cross2d(d - c, b - c)

    proper = (
        ((orient_abc > _SGD2_SEGMENT_EPS) & (orient_abd < -_SGD2_SEGMENT_EPS))
        | ((orient_abc < -_SGD2_SEGMENT_EPS) & (orient_abd > _SGD2_SEGMENT_EPS))
    ) & (
        ((orient_cda > _SGD2_SEGMENT_EPS) & (orient_cdb < -_SGD2_SEGMENT_EPS))
        | ((orient_cda < -_SGD2_SEGMENT_EPS) & (orient_cdb > _SGD2_SEGMENT_EPS))
    )
    collinear = (
        ((orient_abc.abs() <= _SGD2_SEGMENT_EPS) & _sgd2_point_on_segment(c, a, b))
        | ((orient_abd.abs() <= _SGD2_SEGMENT_EPS) & _sgd2_point_on_segment(d, a, b))
        | ((orient_cda.abs() <= _SGD2_SEGMENT_EPS) & _sgd2_point_on_segment(a, c, d))
        | ((orient_cdb.abs() <= _SGD2_SEGMENT_EPS) & _sgd2_point_on_segment(b, c, d))
    )
    return proper | collinear


def _sgd2_crossings_loss(
    pos: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    crossing_state: _sgd2_CrossingLossState,
) -> torch.Tensor:
    """Evaluate the reference neural crossing loss."""
    edge_pair_pos = _sgd2_edge_pair_positions(pos=pos, left=left, right=right)
    if edge_pair_pos.numel() == 0:
        return pos.sum() * 0.0

    labels = _sgd2_are_edge_pairs_crossed(edge_pair_pos.detach()).to(
        device=pos.device,
        dtype=pos.dtype,
    )
    crossing_state.detector.train()
    for _ in range(_SGD2_CROSSING_DETECTOR_TRAIN_STEPS):
        preds = crossing_state.detector(edge_pair_pos.detach()).view(-1)
        train_loss = crossing_state.train_loss(preds, labels)
        crossing_state.optimizer.zero_grad(set_to_none=True)
        train_loss.backward()
        crossing_state.optimizer.step()

    crossing_state.detector.eval()
    preds = crossing_state.detector(edge_pair_pos).view(-1)
    return crossing_state.position_loss(preds, torch.zeros_like(preds))


def _sgd2_crossing_angle_loss(
    pos: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the crossing-angle maximization criterion."""
    edge_pair_pos = _sgd2_edge_pair_positions(pos=pos, left=left, right=right)
    if edge_pair_pos.numel() == 0:
        return pos.sum() * 0.0

    labels = _sgd2_are_edge_pairs_crossed(edge_pair_pos).to(device=pos.device, dtype=pos.dtype)
    left_vec = pos[left[1]] - pos[left[0]]
    right_vec = pos[right[1]] - pos[right[0]]
    similarities = F.cosine_similarity(left_vec, right_vec, dim=1)
    similarity_sq = similarities.square()
    return (labels * similarity_sq / (1.0 - similarity_sq + _SGD2_EPS)).mean()


def _sgd2_aspect_ratio_loss(pos: torch.Tensor, target: float) -> torch.Tensor:
    """Evaluate the aspect-ratio criterion from singular values."""
    if pos.shape[0] <= 1:
        return pos.sum() * 0.0
    centered = pos - pos.mean(dim=0, keepdim=True)
    _, singular_values, _ = torch.linalg.svd(centered, full_matrices=False)
    if singular_values.numel() < 2:
        return pos.sum() * 0.0
    ratio = (singular_values[1] / singular_values[0].clamp(min=_SGD2_EPS)).clamp(
        _SGD2_EPS, 1.0 - _SGD2_EPS
    )
    target_tensor = torch.tensor(
        float(min(max(target, 0.0), 1.0)),
        device=pos.device,
        dtype=pos.dtype,
    )
    return F.binary_cross_entropy(ratio, target_tensor, reduction="sum")


def _sgd2_angular_resolution_loss(
    pos: torch.Tensor,
    pair_batch: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the reference angular-resolution criterion."""
    if pair_batch.numel() == 0:
        return pos.sum() * 0.0

    degrees = pair_batch[0].to(device=pos.device, dtype=pos.dtype)
    a = pair_batch[1]
    b = pair_batch[2]
    c = pair_batch[3]
    d = pair_batch[4]
    similarities = F.cosine_similarity(pos[b] - pos[a], pos[d] - pos[c], dim=1)
    angles = torch.arccos(similarities.clamp(-0.99, 0.99))
    optimal = 2.0 * math.pi / degrees.clamp(min=1.0)
    normalized = F.relu((-angles + optimal) / optimal.clamp(min=_SGD2_EPS))
    return F.binary_cross_entropy(normalized, torch.zeros_like(angles))


def _sgd2_vertex_resolution_loss(
    pos: torch.Tensor,
    pair_batch: torch.Tensor,
    state: _sgd2_VertexResolutionState,
) -> torch.Tensor:
    """Evaluate the adaptive vertex-resolution loss."""
    if pair_batch.numel() == 0:
        return pos.sum() * 0.0
    distances = torch.linalg.norm(pos[pair_batch[0]] - pos[pair_batch[1]], dim=1)
    dmax = distances.max().detach()
    target = 1.0 / math.sqrt(float(max(pos.shape[0], 1)))
    target_dist = torch.as_tensor(target, device=pos.device, dtype=pos.dtype) * dmax
    previous_target = state.prev_target_dist.to(device=pos.device, dtype=pos.dtype)
    weight = state.prev_weight * _SGD2_VERTEX_RESOLUTION_SMOOTHNESS + 1.0
    smoothed_target = (
        torch.maximum(target_dist, previous_target)
        + torch.minimum(target_dist, previous_target) * _SGD2_VERTEX_RESOLUTION_SMOOTHNESS
    ) / weight
    state.prev_target_dist = smoothed_target.detach()
    state.prev_weight = weight
    return F.relu(1.0 - distances / smoothed_target.clamp(min=_SGD2_EPS)).square().mean()


def _sgd2_criterion_loss(
    name: str,
    pos: torch.Tensor,
    state: _sgd2_PreparedState,
    batch_size: int,
    sampler: _sgd2_CyclicSampler | None = None,
    vertex_resolution_state: _sgd2_VertexResolutionState | None = None,
    crossing_state: _sgd2_CrossingLossState | None = None,
) -> torch.Tensor:
    """Evaluate one named criterion on a sampled mini-batch."""
    if name == "stress":
        if (
            state.stress_pairs is None
            or state.stress_distances is None
            or state.stress_weights is None
        ):
            return pos.sum() * 0.0
        if sampler is not None:
            sample_index = sampler.sample(batch_size)
        else:
            sample_index = _sgd2_sample_indices(
                total=state.stress_pairs.shape[1],
                batch_size=batch_size,
                device=state.device,
            )
        return _sgd2_stress_loss(
            pos=pos,
            pair_batch=state.stress_pairs[:, sample_index],
            pair_distances=state.stress_distances[sample_index],
            pair_weights=state.stress_weights[sample_index],
        )
    if name == "ideal_edge_length":
        if sampler is not None:
            idx = sampler.sample(batch_size)
            edge_batch = state.edges[:, idx]
        else:
            edge_batch = _sgd2_sample_edges(state.edges, batch_size=batch_size)
        return _sgd2_ideal_edge_length_loss(
            pos=pos,
            edge_batch=edge_batch,
            target=_SGD2_DEFAULT_IDEAL_EDGE_LENGTH,
        )
    if name == "neighborhood_preservation":
        if sampler is not None:
            anchor_nodes = sampler.sample(batch_size)
        else:
            anchor_nodes = _sgd2_sample_nodes(
                num_nodes=pos.shape[0],
                batch_size=batch_size,
                device=state.device,
            )
        return _sgd2_neighborhood_preservation_loss(
            pos=pos,
            anchor_nodes=anchor_nodes,
            adjacency=state.adjacency,
        )
    if name == "crossings":
        if crossing_state is None:
            raise ValueError("crossing_state is required for the crossings criterion.")
        if state.non_incident_edge_pairs is None:
            return pos.sum() * 0.0
        if sampler is not None:
            idx = sampler.sample(batch_size)
            pair_batch = state.non_incident_edge_pairs[:, idx]
        else:
            pair_batch = _sgd2_sample_pairs(state.non_incident_edge_pairs, batch_size=batch_size)
        left = pair_batch[:2]
        right = pair_batch[2:]
        return _sgd2_crossings_loss(
            pos=pos,
            left=left,
            right=right,
            crossing_state=crossing_state,
        )
    if name == "crossing_angle_maximization":
        if state.non_incident_edge_pairs is None:
            left, right = _sgd2_sample_edge_pairs(edges=state.edges, batch_size=batch_size)
        elif sampler is not None:
            idx = sampler.sample(batch_size)
            pair_batch = state.non_incident_edge_pairs[:, idx]
            left = pair_batch[:2]
            right = pair_batch[2:]
        else:
            pair_batch = _sgd2_sample_pairs(state.non_incident_edge_pairs, batch_size=batch_size)
            left = pair_batch[:2]
            right = pair_batch[2:]
        return _sgd2_crossing_angle_loss(pos=pos, left=left, right=right)
    if name == "aspect_ratio":
        if sampler is not None:
            idx = sampler.sample(batch_size)
            sampled_pos = pos[idx]
        else:
            sampled_pos = pos
        return _sgd2_aspect_ratio_loss(pos=sampled_pos, target=_SGD2_DEFAULT_ASPECT_RATIO_TARGET)
    if name == "angular_resolution":
        if state.incident_edge_pairs is None:
            return pos.sum() * 0.0
        if sampler is not None:
            idx = sampler.sample(batch_size)
            pair_batch = state.incident_edge_pairs[:, idx]
        else:
            pair_batch = _sgd2_sample_pairs(state.incident_edge_pairs, batch_size=batch_size)
        return _sgd2_angular_resolution_loss(pos=pos, pair_batch=pair_batch)
    if name == "vertex_resolution":
        if state.stress_pairs is None:
            return pos.sum() * 0.0
        if vertex_resolution_state is None:
            raise ValueError("vertex_resolution_state is required for vertex resolution.")
        if sampler is not None:
            idx = sampler.sample(batch_size)
            pair_batch = state.stress_pairs[:, idx]
        else:
            pair_batch = _sgd2_sample_pairs(state.stress_pairs, batch_size=batch_size)
        return _sgd2_vertex_resolution_loss(
            pos=pos,
            pair_batch=pair_batch,
            state=vertex_resolution_state,
        )
    raise ValueError(f"Unknown (SGD)^2 criterion: {name}")


@dataclass(frozen=True)
class _sgd2_PreparedState:
    """Precomputed graph data needed by the multicriteria objective."""

    device: torch.device
    edges: torch.Tensor
    adjacency: list[list[tuple[int, float]]]
    all_pairs_distances: Optional[torch.Tensor]
    stress_pairs: Optional[torch.Tensor]
    stress_distances: Optional[torch.Tensor]
    stress_weights: Optional[torch.Tensor]
    incident_edge_pairs: Optional[torch.Tensor]
    non_incident_edge_pairs: Optional[torch.Tensor]


@dataclass
class _sgd2_VertexResolutionState:
    """State carried across iterations for vertex-resolution."""

    prev_target_dist: torch.Tensor
    prev_weight: float


@dataclass
class _sgd2_CrossingLossState:
    """Persistent neural detector state for the crossing criterion."""

    detector: nn.Module
    optimizer: torch.optim.Optimizer
    train_loss: nn.Module
    position_loss: nn.Module


class _sgd2_CrossingDetector(nn.Module):
    """Feed-forward crossing detector matching the reference architecture."""

    def __init__(self) -> None:
        """Initialize the detector layers."""
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(8, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Linear(512, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the detector on flattened edge-pair coordinates."""
        return self.main(x)


_sgd2 = SimpleNamespace(
    _EPS=_SGD2_EPS,
    _SEGMENT_EPS=_SGD2_SEGMENT_EPS,
    _DEFAULT_IDEAL_EDGE_LENGTH=_SGD2_DEFAULT_IDEAL_EDGE_LENGTH,
    _DEFAULT_ASPECT_RATIO_TARGET=_SGD2_DEFAULT_ASPECT_RATIO_TARGET,
    _VERTEX_RESOLUTION_SMOOTHNESS=_SGD2_VERTEX_RESOLUTION_SMOOTHNESS,
    _NEIGHBORHOOD_DEPTH_LIMIT=_SGD2_NEIGHBORHOOD_DEPTH_LIMIT,
    _NEIGHBORHOOD_NEG_SAMPLE_RATE=_SGD2_NEIGHBORHOOD_NEG_SAMPLE_RATE,
    _NEIGHBORHOOD_K_DIST=_SGD2_NEIGHBORHOOD_K_DIST,
    _CROSSING_DETECTOR_TRAIN_STEPS=_SGD2_CROSSING_DETECTOR_TRAIN_STEPS,
    _CROSSING_DETECTOR_LR=_SGD2_CROSSING_DETECTOR_LR,
    _PreparedState=_sgd2_PreparedState,
    _VertexResolutionState=_sgd2_VertexResolutionState,
    _CrossingLossState=_sgd2_CrossingLossState,
    _CrossingDetector=_sgd2_CrossingDetector,
    _build_adjacency=_sgd2_build_adjacency,
    _all_pairs_shortest_paths=_sgd2_all_pairs_shortest_paths,
    _prepare_state=_sgd2_prepare_state,
    _sample_pairs=_sgd2_sample_pairs,
    _CyclicSampler=_sgd2_CyclicSampler,
    _edge_pair_positions=_sgd2_edge_pair_positions,
    _are_edge_pairs_crossed=_sgd2_are_edge_pairs_crossed,
    _criterion_loss=_sgd2_criterion_loss,
)


_UMAP_DEFAULT_MIN_DIST = 0.1
_UMAP_DEFAULT_SPREAD = 1.0
_TSNE_DEFAULT_PERPLEXITY = 30.0
_SGD2_SAMPLER_KEY = "sgd2_samplers"
_SGD2_PREPARED_STATE_KEY = "sgd2_prepared_state"
_SGD2_CROSSING_STATE_KEY = "sgd2_crossing_state"
_SGD2_VERTEX_RESOLUTION_STATE_KEY = "sgd2_vertex_resolution_state"
_SGD2_BATCH_SIZE_KEY = "sgd2_batch_size"
_SGD2_ACTIVE_CRITERION_KEY = "sgd2_active_criterion"
_KD_TREE_PAIR_KEY = "neulay_kdtree_pairs"


@dataclass(frozen=True)
class ExactPairStressLossConfig:
    """Configuration for :class:`ExactPairStressLoss`.

    Parameters
    ----------
    weight_fn : str, default="inverse_sq"
        Stress weight transform. Supported values are ``"inverse_sq"``,
        ``"inverse"``, and ``"uniform"``.
    """

    weight_fn: str = "inverse_sq"


@dataclass(frozen=True)
class KLDivergenceLossConfig:
    """Configuration for :class:`KLDivergenceLoss`.

    Parameters
    ----------
    exaggeration : float, default=12.0
        Early-exaggeration multiplier applied before ``exaggeration_steps``.
    exaggeration_steps : int, default=250
        Number of leading steps that use early exaggeration.
    """

    exaggeration: float = 12.0
    exaggeration_steps: int = 250


@dataclass(frozen=True)
class UMAPCrossEntropyLossConfig:
    """Configuration for :class:`UMAPCrossEntropyLoss`.

    Parameters
    ----------
    neg_rate : int, default=5
        Negative samples per positive sample.
    repulsion_strength : float, default=1.0
        UMAP negative-sample repulsion coefficient ``gamma``.
    """

    neg_rate: int = 5
    repulsion_strength: float = 1.0


@dataclass(frozen=True)
class LinLogAttractionLossConfig:
    """Configuration for :class:`LinLogAttractionLoss`.

    Parameters
    ----------
    exponent_a : float, default=1.0
        Attraction exponent ``a`` from the classic LinLog objective.
    """

    exponent_a: float = 1.0


@dataclass(frozen=True)
class LinLogRepulsionLossConfig:
    """Configuration for :class:`LinLogRepulsionLoss`.

    Parameters
    ----------
    exponent_r : float, default=0.0
        Repulsion exponent ``r`` from the classic LinLog objective.
    """

    exponent_r: float = 0.0


@dataclass(frozen=True)
class LinLogLossConfig:
    """Configuration for :class:`LinLogLoss`.

    Parameters
    ----------
    exponent_a : float, default=1.0
        Attraction exponent ``a`` in ``|p_i - p_j|^a``.
    exponent_r : float, default=0.0
        Repulsion exponent ``r`` in ``-|p_i - p_j|^r``.
    """

    exponent_a: float = 1.0
    exponent_r: float = 0.0


@dataclass(frozen=True)
class EntropyLossConfig:
    """Configuration for :class:`EntropyLoss`.

    Parameters
    ----------
    alpha : float, default=1.0
        Entropy-loss scaling applied to the non-edge term.
    """

    alpha: float = 1.0


@dataclass(frozen=True)
class DavidsonHarelEnergyLossConfig:
    """Configuration for :class:`DavidsonHarelEnergyLoss`.

    Parameters
    ----------
    w_distribution : float, default=1.0
        Weight for node-distribution energy.
    w_border : float, default=0.1
        Weight for border repulsion.
    w_edge_length : float, default=0.2
        Weight for edge-length energy.
    w_crossing : float, default=2.0
        Weight for edge crossing count.
    w_node_edge : float, default=0.5
        Weight for node-edge proximity.
    """

    w_distribution: float = 1.0
    w_border: float = 0.1
    w_edge_length: float = 0.2
    w_crossing: float = 2.0
    w_node_edge: float = 0.5


@dataclass(frozen=True)
class KDTreeRepulsionLossConfig:
    """Configuration for :class:`KDTreeRepulsionLoss`.

    Parameters
    ----------
    radius : float, default=0.4
        Gaussian repulsion radius.
    magnitude : float or str or None, default="auto"
        Repulsion magnitude. ``"auto"`` and ``None`` both resolve to the
        NeuLay adaptive formula ``100 * N^(1/3) * radius``.
    """

    radius: float = 0.4
    magnitude: Union[float, str, None] = "auto"


@dataclass(frozen=True)
class SGD2CriterionLossConfig:
    """Configuration for :class:`SGD2CriterionLoss`.

    Parameters
    ----------
    criterion : str, default="stress"
        One criterion name from the reference multicriteria optimizer.
    batch_size : int, default=16
        Mini-batch size for the sampled criterion evaluation.
    """

    criterion: str = "stress"
    batch_size: int = 16


@dataclass(frozen=True)
class SGD2CrossingDetectorStepConfig:
    """Configuration for :class:`SGD2CrossingDetectorStep`.

    Parameters
    ----------
    inner_steps : int, default=2
        Number of detector training steps before evaluating position loss.
    detector_lr : float, default=0.01
        Adam learning rate for the crossing detector.
    """

    inner_steps: int = 2
    detector_lr: float = 0.01


@dataclass(frozen=True)
class CyclicSamplerConfig:
    """Configuration for :class:`CyclicSampler`.

    Parameters
    ----------
    pool_size : int, default=0
        Explicit sampler pool size. ``0`` means infer the pool size from the
        active SGD2 criterion and prepared state.
    """

    pool_size: int = 0


def _require_positions(state: SolveState) -> torch.Tensor:
    """Return the position tensor or raise a descriptive error.

    Parameters
    ----------
    state : SolveState
        Mutable solve state.

    Returns
    -------
    torch.Tensor
        Current positions with shape ``[N, 2]``.
    """
    if state.pos is None:
        raise ValueError("This op requires `state.pos` to be initialized.")
    return state.pos


def _problem_device(problem: LayoutProblem, state: SolveState) -> torch.device:
    """Resolve the compute device for helper tensors.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout problem.
    state : SolveState
        Mutable solve state.

    Returns
    -------
    torch.device
        Preferred compute device.
    """
    if state.pos is not None:
        return state.pos.device
    if problem.edge_index.numel() > 0:
        return problem.edge_index.device
    if problem.node_sizes is not None:
        return problem.node_sizes.device
    return torch.device("cpu")


def _resolve_distance_matrix(problem: LayoutProblem, state: SolveState) -> torch.Tensor:
    """Return the all-pairs distance matrix, computing it when absent.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout problem.
    state : SolveState
        Mutable solve state.

    Returns
    -------
    torch.Tensor
        Distance matrix with shape ``[N, N]``.
    """
    if state.distance_matrix is not None:
        return state.distance_matrix

    device = _problem_device(problem, state)
    adjacency = _sgd2._build_adjacency(
        edge_index=problem.edge_index,
        num_nodes=problem.num_nodes,
        edge_weights=problem.edge_weights,
    )
    return _sgd2._all_pairs_shortest_paths(
        adjacency=adjacency,
        device=device,
        weighted=problem.edge_weights is not None,
    )


def _resolve_tsne_probabilities(problem: LayoutProblem, state: SolveState) -> torch.Tensor:
    """Resolve the symmetric t-SNE probability matrix.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout problem.
    state : SolveState
        Mutable solve state.

    Returns
    -------
    torch.Tensor
        Symmetric affinity matrix ``P`` with shape ``[N, N]``.
    """
    if state.affinity_matrix is not None:
        return state.affinity_matrix
    if "tsne_probabilities" in state.extras:
        return state.extras["tsne_probabilities"]

    perplexity = float(state.extras.get("tsne_perplexity", _TSNE_DEFAULT_PERPLEXITY))
    distances = _resolve_distance_matrix(problem, state)
    probabilities = _tsnet._high_dimensional_affinities(
        distances.to(device="cpu", dtype=torch.float32),
        min(perplexity, float(max(problem.num_nodes - 1, 1))),
    )
    return probabilities.to(device=_problem_device(problem, state))


def _resolve_umap_graph(
    problem: LayoutProblem,
    state: SolveState,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    """Resolve the positive graph and curve parameters for UMAP.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout problem.
    state : SolveState
        Mutable solve state.
    device : torch.device
        Device for returned tensors.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]
        Positive edge heads, positive edge tails, positive weights, and the
        fitted UMAP ``a`` and ``b`` parameters.
    """
    if {"umap_head", "umap_tail", "umap_weight"} <= state.extras.keys():
        head = state.extras["umap_head"].to(device=device, dtype=torch.long)
        tail = state.extras["umap_tail"].to(device=device, dtype=torch.long)
        weight = state.extras["umap_weight"].to(device=device, dtype=torch.float32)
    else:
        unique_edges, unique_weights = _dh._unique_edges(
            problem.edge_index,
            problem.num_nodes,
            edge_weights=problem.edge_weights,
        )
        if unique_edges:
            head = torch.tensor(
                [source for source, _target in unique_edges],
                dtype=torch.long,
                device=device,
            )
            tail = torch.tensor(
                [target for _source, target in unique_edges],
                dtype=torch.long,
                device=device,
            )
            weight = unique_weights.to(device=device, dtype=torch.float32)
        else:
            head = torch.empty((0,), dtype=torch.long, device=device)
            tail = torch.empty((0,), dtype=torch.long, device=device)
            weight = torch.empty((0,), dtype=torch.float32, device=device)

    min_dist = float(state.extras.get("umap_min_dist", _UMAP_DEFAULT_MIN_DIST))
    spread = float(state.extras.get("umap_spread", _UMAP_DEFAULT_SPREAD))
    fit_a, fit_b = _umap._fit_ab(min_dist=min_dist, spread=spread)
    a = float(state.extras.get("umap_a", fit_a))
    b = float(state.extras.get("umap_b", fit_b))
    return head, tail, weight, a, b


def _resolve_exact_stress_pairs(
    problem: LayoutProblem,
    state: SolveState,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build exact stress distances and upper-triangle node pairs.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout problem.
    state : SolveState
        Mutable solve state.
    device : torch.device
        Device for returned tensors.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Pair indices with shape ``[2, P]`` and graph distances with shape
        ``[P]``.
    """
    distances = _resolve_distance_matrix(problem, state).to(device=device, dtype=torch.float32)
    upper = torch.triu_indices(distances.shape[0], distances.shape[1], offset=1, device=device)
    pair_distances = distances[upper[0], upper[1]]
    mask = torch.isfinite(pair_distances) & (pair_distances > 0)
    return upper[:, mask], pair_distances[mask]


def _stress_weights(targets: torch.Tensor, weight_fn: str) -> torch.Tensor:
    """Compute stress weights from graph distances.

    Parameters
    ----------
    targets : torch.Tensor
        Positive graph distances with shape ``[P]``.
    weight_fn : str
        Weight transform name.

    Returns
    -------
    torch.Tensor
        Stress weights with shape ``[P]``.
    """
    if weight_fn == "inverse_sq":
        return targets.reciprocal().square()
    if weight_fn == "inverse":
        return targets.reciprocal()
    if weight_fn == "uniform":
        return torch.ones_like(targets)
    raise ValueError(f"Unsupported stress weight_fn: {weight_fn!r}")


def _edge_weight_vector(
    problem: LayoutProblem,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Resolve per-edge weights or ones for the input edge list.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout problem.
    device : torch.device
        Device for the result.
    dtype : torch.dtype
        Floating dtype for the result.

    Returns
    -------
    torch.Tensor
        Per-edge weights with shape ``[E]``.
    """
    if problem.edge_weights is None:
        return torch.ones((problem.edge_index.shape[1],), dtype=dtype, device=device)
    return problem.edge_weights.to(device=device, dtype=dtype)


def _resolve_non_edge_pairs(
    problem: LayoutProblem,
    state: SolveState,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resolve exact non-edge pairs for maxent-style entropy.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout problem.
    state : SolveState
        Mutable solve state.
    device : torch.device
        Device for returned tensors.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Non-edge source and destination indices.
    """
    if {"maxent_non_edge_src", "maxent_non_edge_dst"} <= state.extras.keys():
        return (
            state.extras["maxent_non_edge_src"].to(device=device, dtype=torch.long),
            state.extras["maxent_non_edge_dst"].to(device=device, dtype=torch.long),
        )

    adjacency = _maxent._build_undirected_adjacency(
        problem.edge_index,
        problem.num_nodes,
        edge_weights=problem.edge_weights,
    )
    src, dst = _maxent._full_non_edge_pairs(adjacency)
    return src.to(device=device), dst.to(device=device)


def _resolve_kdtree_pairs(pos: torch.Tensor, state: SolveState, radius: float) -> Any:
    """Resolve or refresh cached NeuLay cKDTree pair queries.

    Parameters
    ----------
    pos : torch.Tensor
        Current positions with shape ``[N, 2]``.
    state : SolveState
        Mutable solve state.
    radius : float
        NeuLay Gaussian radius.

    Returns
    -------
    Any
        NumPy array of nearby node pairs.
    """
    query_radius = _neulay._PAIR_QUERY_RADIUS_FACTOR * radius
    cached_pairs = state.extras.get(_KD_TREE_PAIR_KEY)
    cached_radius = state.extras.get("neulay_kdtree_query_radius")
    if cached_pairs is not None and cached_radius == query_radius:
        return cached_pairs

    pairs = _neulay._kdtree_repulsion_pairs(pos=pos, query_radius=query_radius)
    state.extras[_KD_TREE_PAIR_KEY] = pairs
    state.extras["neulay_kdtree_query_radius"] = query_radius
    return pairs


def _resolve_kdtree_magnitude(
    num_nodes: int,
    radius: float,
    magnitude: Union[float, str, None],
) -> float:
    """Resolve the NeuLay repulsion magnitude.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    radius : float
        NeuLay Gaussian radius.
    magnitude : float or str or None
        User-configured magnitude.

    Returns
    -------
    float
        Effective repulsion magnitude.
    """
    if magnitude in {None, "auto"}:
        return 100.0 * float(max(num_nodes, 1)) ** (1.0 / 3.0) * radius
    if isinstance(magnitude, str):
        raise ValueError(f"Unsupported KD-tree repulsion magnitude: {magnitude!r}")
    return float(magnitude)


def _resolve_sgd2_state(
    problem: LayoutProblem,
    state: SolveState,
    device: torch.device,
) -> _sgd2._PreparedState:
    """Resolve the precomputed shared state for SGD2 criteria.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout problem.
    state : SolveState
        Mutable solve state.
    device : torch.device
        Device for the returned prepared state.

    Returns
    -------
    _sgd2._PreparedState
        Shared criterion-precompute state.
    """
    prepared = state.extras.get(_SGD2_PREPARED_STATE_KEY)
    if prepared is not None:
        return prepared

    active_criterion = str(state.extras.get(_SGD2_ACTIVE_CRITERION_KEY, "stress"))
    needs_distances = active_criterion in {"stress", "vertex_resolution"}
    needs_incident = active_criterion == "angular_resolution"
    needs_non_incident = active_criterion in {"crossings", "crossing_angle_maximization"}
    prepared = _sgd2._prepare_state(
        edge_index=problem.edge_index,
        num_nodes=problem.num_nodes,
        device=device,
        needs_distances=needs_distances,
        needs_incident_edge_pairs=needs_incident,
        needs_non_incident_edge_pairs=needs_non_incident,
        edge_weights=problem.edge_weights,
    )
    state.extras[_SGD2_PREPARED_STATE_KEY] = prepared
    return prepared


def _resolve_sgd2_sampler_store(state: SolveState) -> Dict[str, _sgd2._CyclicSampler]:
    """Return the mutable SGD2 sampler dictionary from ``state.extras``.

    Parameters
    ----------
    state : SolveState
        Mutable solve state.

    Returns
    -------
    dict[str, _sgd2._CyclicSampler]
        Sampler mapping stored in ``state.extras``.
    """
    samplers = state.extras.get(_SGD2_SAMPLER_KEY)
    if samplers is None:
        samplers = {}
        state.extras[_SGD2_SAMPLER_KEY] = samplers
    return samplers


def _infer_sgd2_pool_size(prepared: _sgd2._PreparedState, criterion: str, num_nodes: int) -> int:
    """Infer a cyclic-sampler pool size for one SGD2 criterion.

    Parameters
    ----------
    prepared : _sgd2._PreparedState
        Shared criterion-precompute state.
    criterion : str
        Criterion name.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    int
        Effective pool size.
    """
    if criterion in {"stress", "vertex_resolution"} and prepared.stress_pairs is not None:
        return int(prepared.stress_pairs.shape[1])
    if criterion == "ideal_edge_length":
        return int(prepared.edges.shape[1])
    if criterion in {"neighborhood_preservation", "aspect_ratio"}:
        return num_nodes
    if criterion == "angular_resolution" and prepared.incident_edge_pairs is not None:
        return int(prepared.incident_edge_pairs.shape[1])
    if criterion in {"crossings", "crossing_angle_maximization"}:
        if prepared.non_incident_edge_pairs is None:
            return 0
        return int(prepared.non_incident_edge_pairs.shape[1])
    return 0


def _resolve_sgd2_sampler(
    problem: LayoutProblem,
    state: SolveState,
    criterion: str,
    pool_size: int,
    device: torch.device,
) -> Optional[_sgd2._CyclicSampler]:
    """Resolve or lazily create the cyclic sampler for one criterion.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout problem.
    state : SolveState
        Mutable solve state.
    criterion : str
        Criterion name.
    pool_size : int
        Explicit pool size. ``0`` means infer from prepared state.
    device : torch.device
        Sampler device.

    Returns
    -------
    _sgd2._CyclicSampler or None
        Criterion sampler, or ``None`` when the criterion has no pool.
    """
    samplers = _resolve_sgd2_sampler_store(state)
    if criterion in samplers:
        return samplers[criterion]

    prepared = _resolve_sgd2_state(problem, state, device)
    total = (
        pool_size
        if pool_size > 0
        else _infer_sgd2_pool_size(prepared, criterion, problem.num_nodes)
    )
    if total <= 0:
        return None
    sampler = _sgd2._CyclicSampler(total, device)
    samplers[criterion] = sampler
    return sampler


def _resolve_sgd2_vertex_resolution_state(
    state: SolveState,
    device: torch.device,
) -> Optional[_sgd2._VertexResolutionState]:
    """Resolve the persistent vertex-resolution smoothing state.

    Parameters
    ----------
    state : SolveState
        Mutable solve state.
    device : torch.device
        Device for new tensors.

    Returns
    -------
    _sgd2._VertexResolutionState or None
        Smoothing state used by the vertex-resolution criterion.
    """
    resolved = state.extras.get(_SGD2_VERTEX_RESOLUTION_STATE_KEY)
    if resolved is not None:
        return resolved

    resolved = _sgd2._VertexResolutionState(
        prev_target_dist=torch.tensor(1.0, dtype=torch.float32, device=device),
        prev_weight=0.0,
    )
    state.extras[_SGD2_VERTEX_RESOLUTION_STATE_KEY] = resolved
    return resolved


def _resolve_sgd2_crossing_state(
    state: SolveState,
    device: torch.device,
    inner_steps: int,
    detector_lr: float,
) -> _sgd2._CrossingLossState:
    """Resolve the persistent SGD2 crossing-detector state.

    Parameters
    ----------
    state : SolveState
        Mutable solve state.
    device : torch.device
        Device for the detector.
    inner_steps : int
        Detector update count used by the helper op.
    detector_lr : float
        Detector optimizer learning rate.

    Returns
    -------
    _sgd2._CrossingLossState
        Crossing detector state.
    """
    resolved = state.extras.get(_SGD2_CROSSING_STATE_KEY)
    if resolved is not None:
        resolved.inner_steps = inner_steps
        return resolved

    detector = _sgd2._CrossingDetector().to(device=device)
    resolved = _sgd2._CrossingLossState(
        detector=detector,
        optimizer=torch.optim.Adam(detector.parameters(), lr=detector_lr),
        train_loss=torch.nn.BCELoss(),
        position_loss=torch.nn.BCELoss(reduction="sum"),
    )
    # The dataclass is mutable, so storing the helper-op setting on the object
    # keeps the public config separate from the reference dataclass shape.
    resolved.inner_steps = inner_steps  # type: ignore[attr-defined]
    state.extras[_SGD2_CROSSING_STATE_KEY] = resolved
    return resolved


def _crossings_loss_with_override_steps(
    pos: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    crossing_state: _sgd2._CrossingLossState,
    inner_steps: int,
) -> torch.Tensor:
    """Evaluate SGD2's crossing loss with configurable detector steps.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    left : torch.Tensor
        Left edge batch with shape ``[2, B]``.
    right : torch.Tensor
        Right edge batch with shape ``[2, B]``.
    crossing_state : _sgd2._CrossingLossState
        Persistent detector state.
    inner_steps : int
        Detector updates to run before evaluating the position loss.

    Returns
    -------
    torch.Tensor
        Scalar crossing loss.
    """
    edge_pair_pos = _sgd2._edge_pair_positions(pos=pos, left=left, right=right)
    if edge_pair_pos.numel() == 0:
        return pos.sum() * 0.0

    labels = _sgd2._are_edge_pairs_crossed(edge_pair_pos.detach()).to(
        device=pos.device,
        dtype=pos.dtype,
    )
    crossing_state.detector.train()
    for _ in range(inner_steps):
        preds = crossing_state.detector(edge_pair_pos.detach()).view(-1)
        train_loss = crossing_state.train_loss(preds, labels)
        crossing_state.optimizer.zero_grad(set_to_none=True)
        train_loss.backward()
        crossing_state.optimizer.step()

    crossing_state.detector.eval()
    preds = crossing_state.detector(edge_pair_pos).view(-1)
    return crossing_state.position_loss(preds, torch.zeros_like(preds))


@register_op
class ExactPairStressLoss(LossOp):
    """Exact graph-stress loss over all finite node pairs."""

    name = "exact_pair_stress_loss"
    category = OpCategory.LOSS
    reads = ("pos", "distance_matrix")
    requires = ("pos",)
    weight_key = "stress"

    def __init__(self, config: Optional[ExactPairStressLossConfig] = None) -> None:
        """Store the exact-stress configuration.

        Parameters
        ----------
        config : ExactPairStressLossConfig, optional
            Weighting configuration.

        Returns
        -------
        None
            The op stores the supplied config.
        """
        self.config = config or ExactPairStressLossConfig()

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Evaluate exact weighted graph stress.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout problem.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar stress loss.
        """
        del ctx
        pos = _require_positions(state)
        pairs, targets = _resolve_exact_stress_pairs(problem, state, pos.device)
        if pairs.numel() == 0:
            return pos.sum() * 0.0
        lengths = torch.linalg.norm(pos[pairs[0]] - pos[pairs[1]], dim=1)
        weights = _stress_weights(targets, self.config.weight_fn)
        return (weights * (targets - lengths).square()).sum()


@register_op
class PivotApproxStressLoss(LossOp):
    """Pivot-approximated maxent-stress objective."""

    name = "pivot_approx_stress_loss"
    category = OpCategory.LOSS
    reads = ("pos", "pivot_indices", "pivot_distances")
    requires = ("pos", "pivot_distances")
    weight_key = "stress"

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Evaluate the pivot-approximated stress term.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout problem.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar stress loss.
        """
        del problem, ctx
        pos = _require_positions(state)
        pivot_indices = (
            torch.empty((0,), dtype=torch.long, device=pos.device)
            if state.pivot_indices is None
            else state.pivot_indices.to(device=pos.device, dtype=torch.long)
        )
        if state.pivot_distances is None:
            raise ValueError("PivotApproxStressLoss requires `state.pivot_distances`.")
        pivot_distances = state.pivot_distances.to(device=pos.device, dtype=pos.dtype)
        empty_long = torch.empty((0,), dtype=torch.long, device=pos.device)
        empty_float = torch.empty((0,), dtype=pos.dtype, device=pos.device)
        return _maxent._stress_term(
            positions=pos,
            stress_src=empty_long,
            stress_dst=empty_long,
            stress_lengths=empty_float,
            pivot_indices=pivot_indices,
            pivot_distances=pivot_distances,
        )


@register_op
class KLDivergenceLoss(LossOp):
    """Exact t-SNE KL divergence with early exaggeration."""

    name = "kl_divergence_loss"
    category = OpCategory.LOSS
    reads = ("pos", "affinity_matrix", "distance_matrix")
    requires = ("pos",)
    weight_key = "kl"

    def __init__(self, config: Optional[KLDivergenceLossConfig] = None) -> None:
        """Store the t-SNE KL configuration.

        Parameters
        ----------
        config : KLDivergenceLossConfig, optional
            Early-exaggeration settings.

        Returns
        -------
        None
            The op stores the supplied config.
        """
        self.config = config or KLDivergenceLossConfig()

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Evaluate the t-SNE KL divergence.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout problem.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar KL divergence.
        """
        del ctx
        pos = _require_positions(state)
        probabilities = _resolve_tsne_probabilities(problem, state).to(
            device=pos.device,
            dtype=pos.dtype,
        )
        exaggeration = (
            self.config.exaggeration if state.step < self.config.exaggeration_steps else 1.0
        )
        return _tsnet._tsne_loss(pos, probabilities * exaggeration)


@register_op
class UMAPCrossEntropyLoss(LossOp):
    """UMAP cross-entropy loss with negative sampling."""

    name = "umap_cross_entropy_loss"
    category = OpCategory.LOSS
    reads = ("pos", "extras.umap_head", "extras.umap_tail", "extras.umap_weight")
    requires = ("pos",)
    weight_key = "umap_ce"
    access_pattern = "sampled"

    def __init__(self, config: Optional[UMAPCrossEntropyLossConfig] = None) -> None:
        """Store the UMAP loss configuration.

        Parameters
        ----------
        config : UMAPCrossEntropyLossConfig, optional
            Negative-sampling configuration.

        Returns
        -------
        None
            The op stores the supplied config.
        """
        self.config = config or UMAPCrossEntropyLossConfig()

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Evaluate the sampled UMAP cross-entropy objective.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout problem.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar UMAP loss.
        """
        pos = _require_positions(state)
        head, tail, weight, a, b = _resolve_umap_graph(problem, state, pos.device)
        if head.numel() == 0:
            return pos.sum() * 0.0

        generator = ctx.generator
        if generator is None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(problem.seed + state.step + 1)

        positive_diff = pos[head] - pos[tail]
        positive_distance_sq = positive_diff.square().sum(dim=1)
        positive_prob = (1.0 + (a * positive_distance_sq.pow(b))).reciprocal()
        positive_loss = -(weight * positive_prob.clamp(min=_umap._EPSILON).log()).sum()

        if self.config.neg_rate <= 0 or self.config.repulsion_strength == 0.0:
            return positive_loss

        negatives = torch.randint(
            0,
            problem.num_nodes,
            (head.shape[0], self.config.neg_rate),
            generator=generator,
            dtype=torch.long,
        ).to(device=pos.device)
        source = head.unsqueeze(1).expand_as(negatives)
        negative_diff = pos[source] - pos[negatives]
        negative_distance_sq = negative_diff.square().sum(dim=2)
        negative_prob = (1.0 + (a * negative_distance_sq.pow(b))).reciprocal()
        negative_loss = (
            -self.config.repulsion_strength
            * torch.log(1.0 - negative_prob.clamp(max=1.0 - _umap._EPSILON)).sum()
        )
        return positive_loss + negative_loss


@register_op
class LinLogAttractionLoss(LossOp):
    """LinLog edge-attraction term from the classic objective."""

    name = "linlog_attraction_loss"
    category = OpCategory.LOSS
    reads = ("pos",)
    requires = ("pos",)
    weight_key = "linlog_attract"

    def __init__(self, config: Optional[LinLogAttractionLossConfig] = None) -> None:
        """Store the LinLog attraction configuration.

        Parameters
        ----------
        config : LinLogAttractionLossConfig, optional
            Attraction exponent configuration.

        Returns
        -------
        None
            The op stores the supplied config.
        """
        self.config = config or LinLogAttractionLossConfig()

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Evaluate the attraction-only LinLog term.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout problem.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar attraction loss.
        """
        del ctx
        pos = _require_positions(state)
        if problem.edge_index.numel() == 0:
            return pos.sum() * 0.0
        src = problem.edge_index[0].to(device=pos.device, dtype=torch.long)
        dst = problem.edge_index[1].to(device=pos.device, dtype=torch.long)
        edge_lengths = torch.linalg.norm(pos[src] - pos[dst], dim=1).clamp(
            min=_linlog._MIN_DISTANCE
        )
        weights = _edge_weight_vector(problem, pos.device, edge_lengths.dtype)
        return (weights * edge_lengths.pow(self.config.exponent_a)).sum()


@register_op
class LinLogRepulsionLoss(LossOp):
    """LinLog all-pairs repulsion term from the classic objective."""

    name = "linlog_repulsion_loss"
    category = OpCategory.LOSS
    reads = ("pos",)
    requires = ("pos",)
    weight_key = "linlog_repel"

    def __init__(self, config: Optional[LinLogRepulsionLossConfig] = None) -> None:
        """Store the LinLog repulsion configuration.

        Parameters
        ----------
        config : LinLogRepulsionLossConfig, optional
            Repulsion exponent configuration.

        Returns
        -------
        None
            The op stores the supplied config.
        """
        self.config = config or LinLogRepulsionLossConfig()

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Evaluate the repulsion-only LinLog term.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout problem.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar repulsion loss.
        """
        del problem, ctx
        pos = _require_positions(state)
        num_nodes = int(pos.shape[0])
        pair_src, pair_dst = _linlog._full_all_pairs(num_nodes=num_nodes, device=pos.device)
        if pair_src.numel() == 0:
            return pos.sum() * 0.0
        distances = torch.linalg.norm(pos[pair_src] - pos[pair_dst], dim=1).clamp(
            min=_linlog._MIN_DISTANCE
        )
        if self.config.exponent_r == 0.0:
            return -torch.log(distances).sum()
        return -distances.pow(self.config.exponent_r).sum()


@register_op
class LinLogLoss(LossOp):
    """Evaluate the full classic LinLog objective (attraction + repulsion)."""

    name = "linlog_loss"
    category = OpCategory.LOSS
    reads = ("pos", "step")
    requires = ("pos",)

    def __init__(self, config: Optional[LinLogLossConfig] = None) -> None:
        """Store full objective exponents for the LinLog criterion.

        Parameters
        ----------
        config : LinLogLossConfig, optional
            Attraction and repulsion exponents.

        Returns
        -------
        None
            The op stores only its resolved configuration.
        """
        self.config = config or LinLogLossConfig()

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Evaluate the full objective via the archived helper.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout problem definition.
        state : SolveState
            Mutable state containing current positions.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar classic LinLog energy value.
        """
        del ctx
        positions = _require_positions(state)
        return _linlog._linlog_loss(
            positions=positions,
            edge_index=problem.edge_index,
            seed=problem.seed,
            step=state.step,
            a=self.config.exponent_a,
            r=self.config.exponent_r,
            edge_weights=problem.edge_weights,
        )


@register_op
class EntropyLoss(LossOp):
    """Maxent-stress non-edge entropy regularizer."""

    name = "entropy_loss"
    category = OpCategory.LOSS
    reads = ("pos", "extras.maxent_non_edge_src", "extras.maxent_non_edge_dst")
    requires = ("pos",)
    weight_key = "entropy"

    def __init__(self, config: Optional[EntropyLossConfig] = None) -> None:
        """Store the entropy-loss configuration.

        Parameters
        ----------
        config : EntropyLossConfig, optional
            Entropy scaling configuration.

        Returns
        -------
        None
            The op stores the supplied config.
        """
        self.config = config or EntropyLossConfig()

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Evaluate the exact non-edge entropy term.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout problem.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar entropy loss.
        """
        del ctx
        pos = _require_positions(state)
        src, dst = _resolve_non_edge_pairs(problem, state, pos.device)
        return self.config.alpha * _maxent._entropy_term(pos, src, dst, scale=1.0)


@register_op
class DavidsonHarelEnergyLoss(LossOp):
    """Five-term Davidson-Harel simulated-annealing energy."""

    name = "davidson_harel_energy_loss"
    category = OpCategory.LOSS
    reads = ("pos",)
    requires = ("pos",)
    weight_key = "davidson_harel"

    def __init__(self, config: Optional[DavidsonHarelEnergyLossConfig] = None) -> None:
        """Store the Davidson-Harel energy weights.

        Parameters
        ----------
        config : DavidsonHarelEnergyLossConfig, optional
            Energy-term weights.

        Returns
        -------
        None
            The op stores the supplied config.
        """
        self.config = config or DavidsonHarelEnergyLossConfig()

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Evaluate the Davidson-Harel energy scalar.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout problem.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar energy value.
        """
        del ctx
        pos = _require_positions(state)
        extent = _dh._layout_extent(problem.num_nodes, problem.node_sizes)
        edges, unique_edge_weights = _dh._unique_edges(
            problem.edge_index,
            problem.num_nodes,
            edge_weights=problem.edge_weights,
        )
        num_nodes = int(pos.shape[0])
        distribution = torch.tensor(0.0, dtype=pos.dtype, device=pos.device)
        if num_nodes > 1:
            src, dst = torch.triu_indices(num_nodes, num_nodes, offset=1, device=pos.device)
            squared_distances = (
                (pos[src] - pos[dst]).square().sum(dim=1).clamp(min=_dh._MIN_DISTANCE)
            )
            distribution = squared_distances.reciprocal().sum()

        border_distances = torch.stack(
            [
                pos[:, 0] + extent,
                extent - pos[:, 0],
                pos[:, 1] + extent,
                extent - pos[:, 1],
            ],
            dim=1,
        ).clamp(min=_dh._MIN_DISTANCE)
        border = border_distances.reciprocal().square().sum()

        edge_length = torch.tensor(0.0, dtype=pos.dtype, device=pos.device)
        if edges:
            edge_weight_tensor = unique_edge_weights.to(device=pos.device, dtype=pos.dtype)
            edge_lengths = [
                torch.linalg.norm(pos[source] - pos[target]).square() * edge_weight_tensor[index]
                for index, (source, target) in enumerate(edges)
            ]
            edge_length = torch.stack(edge_lengths).sum()

        crossings = 0.0
        for index, (a, b) in enumerate(edges):
            for c, d in edges[index + 1 :]:
                if len({a, b, c, d}) < 4:
                    continue
                if _dh._segments_intersect(pos[a], pos[b], pos[c], pos[d]):
                    crossings += 1.0
        crossing_energy = torch.tensor(crossings, dtype=pos.dtype, device=pos.device)

        penalties = []
        for node in range(num_nodes):
            for source, target in edges:
                if node in (source, target):
                    continue
                distance = _dh._point_segment_distance(pos[node], pos[source], pos[target])
                penalties.append(distance.clamp(min=_dh._MIN_DISTANCE).reciprocal().square())
        node_edge = (
            torch.stack(penalties).sum()
            if penalties
            else torch.tensor(0.0, dtype=pos.dtype, device=pos.device)
        )

        edge_count = len(edges)
        distribution_scale = _dh._scale_denominator(num_nodes * max(num_nodes - 1, 1) // 2)
        border_scale = _dh._scale_denominator(num_nodes)
        edge_length_scale = _dh._scale_denominator(edge_count)
        crossing_scale = _dh._scale_denominator(edge_count * edge_count)
        node_edge_scale = _dh._scale_denominator(num_nodes * edge_count)
        return (
            self.config.w_distribution * (distribution / distribution_scale)
            + self.config.w_border * (border / border_scale)
            + self.config.w_edge_length * (edge_length / edge_length_scale)
            + self.config.w_crossing * (crossing_energy / crossing_scale)
            + self.config.w_node_edge * (node_edge / node_edge_scale)
        )

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Evaluate the non-differentiable energy without calling backward.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout problem.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with ``prev_loss`` updated.
        """
        loss = self.evaluate(problem, state, ctx)
        state.prev_loss = float(loss.detach().item())
        return state


@register_op
class ElasticLoss(LossOp):
    """NeuLay elastic edge-attraction loss."""

    name = "elastic_loss"
    category = OpCategory.LOSS
    reads = ("pos",)
    requires = ("pos",)
    weight_key = "elastic"

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Evaluate NeuLay's elastic loss.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout problem.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar elastic loss.
        """
        del ctx
        pos = _require_positions(state)
        edge_index = problem.edge_index.to(device=pos.device, dtype=torch.long)
        return _neulay._elastic_loss(pos=pos, edge_index=edge_index)


@register_op
class KDTreeRepulsionLoss(LossOp):
    """NeuLay Gaussian repulsion over cached KD-tree pairs."""

    name = "kdtree_repulsion_loss"
    category = OpCategory.LOSS
    reads = ("pos", "extras.neulay_kdtree_pairs")
    writes = ("extras.neulay_kdtree_pairs",)
    requires = ("pos",)
    weight_key = "kdtree_repel"
    access_pattern = "sampled"

    def __init__(self, config: Optional[KDTreeRepulsionLossConfig] = None) -> None:
        """Store the KD-tree repulsion configuration.

        Parameters
        ----------
        config : KDTreeRepulsionLossConfig, optional
            Repulsion radius and magnitude configuration.

        Returns
        -------
        None
            The op stores the supplied config.
        """
        self.config = config or KDTreeRepulsionLossConfig()

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Evaluate NeuLay's Gaussian KD-tree repulsion term.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout problem.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar repulsion loss.
        """
        del ctx
        pos = _require_positions(state)
        pairs = _resolve_kdtree_pairs(pos=pos, state=state, radius=self.config.radius)
        magnitude = _resolve_kdtree_magnitude(
            num_nodes=problem.num_nodes,
            radius=self.config.radius,
            magnitude=self.config.magnitude,
        )
        return _neulay._kdtree_repulsion_loss(
            pos=pos,
            pairs=pairs,
            radius=self.config.radius,
            magnitude=magnitude,
        )


@register_op
class SGD2CriterionLoss(LossOp):
    """One sampled criterion from the classic (SGD)^2 optimizer."""

    name = "sgd2_criterion_loss"
    category = OpCategory.LOSS
    reads = ("pos", "extras.sgd2_prepared_state", "extras.sgd2_samplers")
    writes = ("extras.sgd2_prepared_state", "extras.sgd2_samplers")
    requires = ("pos",)
    weight_key = "sgd2_criterion"
    access_pattern = "sampled"

    def __init__(self, config: Optional[SGD2CriterionLossConfig] = None) -> None:
        """Store the SGD2 criterion configuration.

        Parameters
        ----------
        config : SGD2CriterionLossConfig, optional
            Criterion name and batch size.

        Returns
        -------
        None
            The op stores the supplied config.
        """
        self.config = config or SGD2CriterionLossConfig()

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Evaluate one (SGD)^2 criterion on a mini-batch.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout problem.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar criterion loss.
        """
        del ctx
        pos = _require_positions(state)
        state.extras[_SGD2_ACTIVE_CRITERION_KEY] = self.config.criterion
        state.extras[_SGD2_BATCH_SIZE_KEY] = self.config.batch_size
        prepared = _resolve_sgd2_state(problem, state, pos.device)
        sampler = _resolve_sgd2_sampler(
            problem=problem,
            state=state,
            criterion=self.config.criterion,
            pool_size=0,
            device=pos.device,
        )
        vertex_resolution_state = None
        if self.config.criterion == "vertex_resolution":
            vertex_resolution_state = _resolve_sgd2_vertex_resolution_state(state, pos.device)
        crossing_state = None
        if self.config.criterion == "crossings":
            crossing_state = _resolve_sgd2_crossing_state(
                state=state,
                device=pos.device,
                inner_steps=_sgd2._CROSSING_DETECTOR_TRAIN_STEPS,
                detector_lr=_sgd2._CROSSING_DETECTOR_LR,
            )
        return _sgd2._criterion_loss(
            name=self.config.criterion,
            pos=pos,
            state=prepared,
            batch_size=self.config.batch_size,
            sampler=sampler,
            vertex_resolution_state=vertex_resolution_state,
            crossing_state=crossing_state,
        )


@register_op
class SGD2CrossingDetectorStep(Op):
    """Train the SGD2 crossing detector and backpropagate crossing loss."""

    name = "sgd2_crossing_detector_step"
    category = OpCategory.LOSS
    reads = ("pos", "extras.sgd2_prepared_state", "extras.sgd2_samplers")
    writes = ("prev_loss", "extras.sgd2_crossing_state", "extras.sgd2_prepared_state")
    requires = ("pos",)
    access_pattern = "sampled"

    def __init__(self, config: Optional[SGD2CrossingDetectorStepConfig] = None) -> None:
        """Store the crossing-detector configuration.

        Parameters
        ----------
        config : SGD2CrossingDetectorStepConfig, optional
            Detector training-step configuration.

        Returns
        -------
        None
            The op stores the supplied config.
        """
        self.config = config or SGD2CrossingDetectorStepConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run one crossing-detector training/evaluation step.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout problem.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with ``prev_loss`` updated after backpropagation.
        """
        del ctx
        pos = _require_positions(state)
        state.extras[_SGD2_ACTIVE_CRITERION_KEY] = "crossings"
        prepared = _resolve_sgd2_state(problem, state, pos.device)
        crossing_state = _resolve_sgd2_crossing_state(
            state=state,
            device=pos.device,
            inner_steps=self.config.inner_steps,
            detector_lr=self.config.detector_lr,
        )
        sampler = _resolve_sgd2_sampler(
            problem=problem,
            state=state,
            criterion="crossings",
            pool_size=0,
            device=pos.device,
        )
        batch_size = int(state.extras.get(_SGD2_BATCH_SIZE_KEY, 0))
        if batch_size <= 0:
            batch_size = _infer_sgd2_pool_size(prepared, "crossings", problem.num_nodes)
        if prepared.non_incident_edge_pairs is None or batch_size <= 0:
            loss = pos.sum() * 0.0
        else:
            if sampler is not None:
                sample_index = sampler.sample(batch_size)
                pair_batch = prepared.non_incident_edge_pairs[:, sample_index]
            else:
                pair_batch = _sgd2._sample_pairs(
                    prepared.non_incident_edge_pairs,
                    batch_size=batch_size,
                )
            loss = _crossings_loss_with_override_steps(
                pos=pos,
                left=pair_batch[:2],
                right=pair_batch[2:],
                crossing_state=crossing_state,
                inner_steps=self.config.inner_steps,
            )
        loss.backward()
        state.prev_loss = float(loss.detach().item())
        state.extras["sgd2_crossing_loss"] = loss.detach()
        return state


@register_op
class CyclicSampler(Op):
    """Create or refresh an SGD2 cyclic sampler in ``state.extras``."""

    name = "cyclic_sampler"
    category = OpCategory.UTILITY
    reads = ("extras.sgd2_active_criterion", "extras.sgd2_prepared_state")
    writes = ("extras.sgd2_samplers",)

    def __init__(self, config: Optional[CyclicSamplerConfig] = None) -> None:
        """Store the sampler configuration.

        Parameters
        ----------
        config : CyclicSamplerConfig, optional
            Explicit or inferred pool-size configuration.

        Returns
        -------
        None
            The op stores the supplied config.
        """
        self.config = config or CyclicSamplerConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Create or replace the active SGD2 cyclic sampler.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout problem.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with the sampler stored in ``extras``.
        """
        del ctx
        criterion = str(state.extras.get(_SGD2_ACTIVE_CRITERION_KEY, "stress"))
        device = _problem_device(problem, state)
        prepared = _resolve_sgd2_state(problem, state, device)
        total = self.config.pool_size
        if total <= 0:
            total = _infer_sgd2_pool_size(prepared, criterion, problem.num_nodes)
        samplers = _resolve_sgd2_sampler_store(state)
        if total > 0:
            samplers[criterion] = _sgd2._CyclicSampler(total, device)
        return state
