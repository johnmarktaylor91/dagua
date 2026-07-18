"""Layout quality metrics suite for dagua.

Three-tier system designed for computational feasibility at 50M+ nodes:
- Tier 1: O(N) or O(|E|), always compute
- Tier 2: Sampled, O(n_samples * BFS_cost), compute when feasible
- Tier 3: DAG/hierarchy-specific, O(N) given metadata

API:
- quick(pos, edge_index, ...) -> dict   — Tier 1 plus sampled ruler presets for N>2000
- full(pos, edge_index, ...) -> dict    — All tiers including sampled metrics
- compare(pos_a, pos_b) -> dict         — Procrustes comparison of two layouts
- composite(metrics) -> float           — Single scalar 0-100

All metric functions are also importable individually.
"""

from __future__ import annotations

import math
import time as _time
from collections import deque
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch

from dagua.layout.ops.cluster_geometry import ClusterGeometryProfile, build_cluster_geometry_profile

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().cpu() if t.device.type != "cpu" else t.detach()


def _deterministic_sample_indices(num_items: int, max_samples: int) -> np.ndarray:
    """Select evenly spaced indices without randomness.

    Parameters
    ----------
    num_items : int
        Total number of items available for sampling.
    max_samples : int
        Maximum number of indices to return.

    Returns
    -------
    numpy.ndarray
        Sorted ``int64`` indices with length ``<= max_samples``.
    """
    if num_items <= 0 or max_samples <= 0:
        return np.empty(0, dtype=np.int64)
    if num_items <= max_samples:
        return np.arange(num_items, dtype=np.int64)
    return (np.arange(max_samples, dtype=np.int64) * num_items) // max_samples


def _pairwise_distance_correlation(pos_a: torch.Tensor, pos_b: torch.Tensor) -> float:
    """Compute rank correlation between sampled pairwise distances.

    Parameters
    ----------
    pos_a : torch.Tensor
        First position tensor with shape ``[N, 2]``.
    pos_b : torch.Tensor
        Second position tensor with shape ``[N, 2]``.

    Returns
    -------
    float
        Spearman correlation between pairwise distance vectors.
    """
    from scipy.stats import spearmanr

    pairwise_a = torch.pdist(pos_a.to(dtype=torch.float32)).numpy()
    pairwise_b = torch.pdist(pos_b.to(dtype=torch.float32)).numpy()

    if pairwise_a.size == 0:
        return 1.0

    rho = spearmanr(pairwise_a, pairwise_b).statistic
    if rho is None or not np.isfinite(rho):
        return 1.0 if np.allclose(pairwise_a, pairwise_b) else 0.0
    return float(rho)


def _sampled_knn_preservation(pos_a: torch.Tensor, pos_b: torch.Tensor, k: int) -> float:
    """Estimate how well k-nearest-neighbor sets are preserved.

    Parameters
    ----------
    pos_a : torch.Tensor
        First position tensor with shape ``[N, 2]``.
    pos_b : torch.Tensor
        Second position tensor with shape ``[N, 2]``.
    k : int
        Number of nearest neighbors to compare per sampled query node.

    Returns
    -------
    float
        Mean Jaccard similarity across sampled query nodes.
    """
    num_nodes = int(pos_a.shape[0])
    if num_nodes <= 1 or k <= 0:
        return 1.0

    query_indices_np = _deterministic_sample_indices(num_nodes, min(num_nodes, 1000))
    query_indices = torch.from_numpy(query_indices_np).to(dtype=torch.long)
    effective_k = min(k, num_nodes - 1)
    if effective_k <= 0:
        return 1.0

    query_pos_a = pos_a[query_indices].to(dtype=torch.float32)
    query_pos_b = pos_b[query_indices].to(dtype=torch.float32)
    all_pos_a = pos_a.to(dtype=torch.float32)
    all_pos_b = pos_b.to(dtype=torch.float32)

    distances_a = torch.cdist(query_pos_a, all_pos_a)
    distances_b = torch.cdist(query_pos_b, all_pos_b)

    row_indices = torch.arange(query_indices.shape[0], dtype=torch.long)
    distances_a[row_indices, query_indices] = float("inf")
    distances_b[row_indices, query_indices] = float("inf")

    knn_a = torch.topk(distances_a, k=effective_k, largest=False).indices.numpy()
    knn_b = torch.topk(distances_b, k=effective_k, largest=False).indices.numpy()

    jaccard_scores: List[float] = []
    for neighbors_a, neighbors_b in zip(knn_a, knn_b):
        neighbor_set_a = set(int(index) for index in neighbors_a.tolist())
        neighbor_set_b = set(int(index) for index in neighbors_b.tolist())
        union = neighbor_set_a | neighbor_set_b
        if not union:
            jaccard_scores.append(1.0)
            continue
        intersection = neighbor_set_a & neighbor_set_b
        jaccard_scores.append(len(intersection) / len(union))

    return float(np.mean(jaccard_scores)) if jaccard_scores else 1.0


def segments_intersect(
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    p4: torch.Tensor,
) -> torch.Tensor:
    """Vectorized segment intersection test.  All inputs ``[N, 2]``.

    Parameters
    ----------
    p1 : torch.Tensor
        First endpoints for the first segment set with shape ``[N, 2]``.
    p2 : torch.Tensor
        Second endpoints for the first segment set with shape ``[N, 2]``.
    p3 : torch.Tensor
        First endpoints for the second segment set with shape ``[N, 2]``.
    p4 : torch.Tensor
        Second endpoints for the second segment set with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Boolean tensor with shape ``[N]``. Counts both standard transversal
        crossings and *collinear-overlap* crossings (two parallel segments on
        the same infinite line whose 1D projections overlap, e.g. two vertical
        edges that share part of a column). Excludes endpoint-touch cases
        (segments that share only a vertex) via a small ``EPS`` boundary on the
        parametric coordinates.

    Notes
    -----
    Collinear overlap is treated as a crossing because a layout that collapses
    every node to one column produces edges that all live on the same vertical
    line; without this case, a degenerate collapsed layout receives free
    crossing points.
    """
    eps = 1e-6
    p1_np = _ensure_cpu(p1).numpy()
    p2_np = _ensure_cpu(p2).numpy()
    p3_np = _ensure_cpu(p3).numpy()
    p4_np = _ensure_cpu(p4).numpy()
    d1 = p2_np - p1_np
    d2 = p4_np - p3_np
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    parallel = np.abs(cross) < 1e-10

    # Sign-preserving division: clamp(min=1e-10) would flip the sign of
    # negative-cross orientations, silently turning valid intersections
    # into false negatives. Replace only the near-zero (parallel) cases
    # with 1 to avoid div-by-zero.
    safe_cross = np.where(parallel, np.ones_like(cross), cross)

    d3 = p3_np - p1_np
    t = (d3[:, 0] * d2[:, 1] - d3[:, 1] * d2[:, 0]) / safe_cross
    u = (d3[:, 0] * d1[:, 1] - d3[:, 1] * d1[:, 0]) / safe_cross

    # Standard transversal: strict-interior intersection on both segments.
    proper = (~parallel) & (t > eps) & (t < 1 - eps) & (u > eps) & (u < 1 - eps)

    # Collinear-overlap: parallel + p3 lies on the line through p1-p2 +
    # the projections of (p3, p4) onto p1-p2's line direction overlap
    # the open interval (EPS, 1 - EPS).
    collinear = parallel & (np.abs(d1[:, 0] * d3[:, 1] - d1[:, 1] * d3[:, 0]) < 1e-8)
    collinear_overlap = np.zeros_like(parallel)
    if bool(collinear.any()):
        d1_norm_sq = d1[:, 0] ** 2 + d1[:, 1] ** 2
        d1_norm_sq_safe = np.where(
            d1_norm_sq < 1e-10,
            np.ones_like(d1_norm_sq),
            d1_norm_sq,
        )
        t3 = (d3[:, 0] * d1[:, 0] + d3[:, 1] * d1[:, 1]) / d1_norm_sq_safe
        d4 = p4_np - p1_np
        t4 = (d4[:, 0] * d1[:, 0] + d4[:, 1] * d1[:, 1]) / d1_norm_sq_safe
        a = np.minimum(t3, t4)
        b = np.maximum(t3, t4)
        collinear_overlap = collinear & (a < 1 - eps) & (b > eps)

    return torch.from_numpy(proper | collinear_overlap)


class _CrossingGeometry(NamedTuple):
    """Batched geometry and sampling metadata for one edge-pair set."""

    p1: torch.Tensor
    p2: torch.Tensor
    p3: torch.Tensor
    p4: torch.Tensor
    crossing: torch.Tensor
    total_pairs: int
    candidate_pairs: int
    pair_space_exhausted: bool


def _build_csr(edge_index: torch.Tensor, num_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build an undirected CSR adjacency from an edge index.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        CSR row offsets with shape ``[N + 1]`` and target indices with shape
        ``[2E]``.
    """
    src, tgt = edge_index[0], edge_index[1]
    # Make undirected for BFS
    all_src = torch.cat([src, tgt])
    all_tgt = torch.cat([tgt, src])

    order = all_src.argsort()
    csr_tgt = all_tgt[order].numpy()

    degree = torch.zeros(num_nodes, dtype=torch.long)
    degree.scatter_add_(0, all_src, torch.ones(all_src.shape[0], dtype=torch.long))
    offsets = torch.zeros(num_nodes + 1, dtype=torch.long)
    offsets[1:] = degree.cumsum(0)
    return offsets.numpy(), csr_tgt


def _all_pairs_unweighted(
    csr_offsets: np.ndarray,
    csr_targets: np.ndarray,
    num_nodes: int,
    max_dist: int = 20,
) -> np.ndarray:
    """Compute all-pairs unweighted shortest paths with scipy.

    Parameters
    ----------
    csr_offsets : numpy.ndarray
        CSR row offsets with shape ``[N + 1]``.
    csr_targets : numpy.ndarray
        CSR column indices with shape ``[2E]``.
    num_nodes : int
        Number of graph nodes.
    max_dist : int, optional
        Maximum distance to preserve. Larger distances are reported as
        unreachable to match ``_bfs_distances`` truncation.

    Returns
    -------
    numpy.ndarray
        Integer distance matrix with shape ``[N, N]``. Unreachable pairs and
        distances greater than ``max_dist`` are encoded as ``-1``.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    adjacency = csr_matrix(
        (
            np.ones(csr_targets.shape[0], dtype=np.float32),
            csr_targets,
            csr_offsets,
        ),
        shape=(num_nodes, num_nodes),
    )
    distances = shortest_path(
        adjacency,
        method="D",
        directed=False,
        unweighted=True,
    )
    finite = np.isfinite(distances)
    distance_int = np.full(distances.shape, -1, dtype=np.int64)
    distance_int[finite] = distances[finite].astype(np.int64)
    distance_int[distance_int > max_dist] = -1
    return distance_int


def _cached_all_pairs_unweighted(
    edge_index: torch.Tensor,
    num_nodes: int,
    max_dist: int,
    all_pairs_dist: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return cached or freshly computed unweighted all-pairs distances.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    max_dist : int
        Maximum distance to preserve in a newly computed matrix.
    all_pairs_dist : Optional[numpy.ndarray], optional
        Precomputed distance matrix with shape ``[N, N]``.

    Returns
    -------
    numpy.ndarray
        Distance matrix with shape ``[N, N]`` using ``-1`` for unreachable
        pairs and pairs beyond ``max_dist`` when the matrix was computed here.
    """
    if all_pairs_dist is not None:
        return all_pairs_dist
    offsets, targets = _build_csr(_ensure_cpu(edge_index), num_nodes)
    return _all_pairs_unweighted(offsets, targets, num_nodes, max_dist=max_dist)


def _bfs_distances(
    csr_offsets: np.ndarray,
    csr_targets: np.ndarray,
    source: int,
    max_dist: int = 20,
) -> np.ndarray:
    """Compute truncated BFS distances from one source.

    Parameters
    ----------
    csr_offsets : numpy.ndarray
        CSR row offsets with shape ``[N + 1]``.
    csr_targets : numpy.ndarray
        CSR column indices with shape ``[2E]``.
    source : int
        Source node index.
    max_dist : int, optional
        Maximum distance to explore.

    Returns
    -------
    numpy.ndarray
        Distance vector with shape ``[N]``. Unreachable nodes and distances
        greater than ``max_dist`` are encoded as ``-1``.
    """
    N = len(csr_offsets) - 1
    dist = np.full(N, -1, dtype=np.int64)
    dist[source] = 0
    queue = deque([source])

    while queue:
        node = queue.popleft()
        d = dist[node]
        if d >= max_dist:
            continue
        for j in range(csr_offsets[node], csr_offsets[node + 1]):
            child = int(csr_targets[j])
            if dist[child] == -1:
                dist[child] = d + 1
                queue.append(child)
    return dist


# ---------------------------------------------------------------------------
# Tier 1: Always compute — O(N) or O(|E|)
# ---------------------------------------------------------------------------


def edge_length_cv(pos: torch.Tensor, edge_index: torch.Tensor) -> Dict[str, float]:
    """Edge length coefficient of variation + distribution stats.

    Complexity: O(|E|).
    Target: CV < 0.5 good, < 0.3 excellent, > 1.0 problem.
    """
    if edge_index.numel() == 0:
        return {
            "edge_length_cv": 0.0,
            "edge_length_mean": 0.0,
            "edge_length_std": 0.0,
            "edge_length_min": 0.0,
            "edge_length_max": 0.0,
            "edge_length_p01": 0.0,
            "edge_length_p99": 0.0,
        }

    src, tgt = edge_index[0], edge_index[1]
    lengths = torch.norm(pos[src] - pos[tgt], dim=1)
    mean_len = lengths.mean()
    std_len = lengths.std() if lengths.shape[0] > 1 else torch.tensor(0.0)
    cv = (std_len / mean_len).item() if mean_len > 1e-8 else 0.0

    return {
        "edge_length_cv": cv,
        "edge_length_mean": mean_len.item(),
        "edge_length_std": std_len.item(),
        "edge_length_min": lengths.min().item(),
        "edge_length_max": lengths.max().item(),
        "edge_length_p01": lengths.quantile(0.01).item() if len(lengths) > 1 else lengths[0].item(),
        "edge_length_p99": lengths.quantile(0.99).item() if len(lengths) > 1 else lengths[0].item(),
    }


def dag_consistency(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    direction: str = "TB",
    back_edge_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Fraction of edges pointing in the correct DAG direction + violation details.

    Complexity: O(|E|).
    Target: 1.0 (exact).

    Args:
        back_edge_mask: if provided, back edges are excluded from consistency
            checks and reported separately as back_edge_count / back_edge_fraction.
    """
    if edge_index.numel() == 0:
        return {
            "dag_consistency": 1.0,
            "dag_num_violations": 0,
            "dag_mean_violation_mag": 0.0,
            "dag_max_violation_mag": 0.0,
            "back_edge_count": 0,
            "back_edge_fraction": 0.0,
        }

    total_edges = edge_index.shape[1]

    # Filter to forward edges only when back_edge_mask is provided
    if back_edge_mask is not None and back_edge_mask.any():
        forward_mask = ~back_edge_mask
        forward_ei = edge_index[:, forward_mask]
        n_back = back_edge_mask.sum().item()
    else:
        forward_ei = edge_index
        n_back = 0

    result: Dict[str, float] = {}

    if forward_ei.numel() == 0:
        result.update(
            {
                "dag_consistency": 1.0,
                "dag_num_violations": 0,
                "dag_mean_violation_mag": 0.0,
                "dag_max_violation_mag": 0.0,
            }
        )
    else:
        src, tgt = forward_ei[0], forward_ei[1]
        # Self-loops are trivially dag-consistent (no direction to violate).
        # Ties (y_tgt == y_src) come from barycenter ordering and should not
        # penalize -- only strictly reversed edges count as violations.
        self_loops = src == tgt
        if direction == "TB":
            y_src, y_tgt = pos[src, 1], pos[tgt, 1]
            correct = (y_tgt >= y_src) | self_loops
            violation_mag = torch.clamp(y_src[~correct] - y_tgt[~correct], min=0)
        elif direction == "BT":
            y_src, y_tgt = pos[src, 1], pos[tgt, 1]
            correct = (y_tgt <= y_src) | self_loops
            violation_mag = torch.clamp(y_tgt[~correct] - y_src[~correct], min=0)
        elif direction == "LR":
            x_src, x_tgt = pos[src, 0], pos[tgt, 0]
            correct = (x_tgt >= x_src) | self_loops
            violation_mag = torch.clamp(x_src[~correct] - x_tgt[~correct], min=0)
        elif direction == "RL":
            x_src, x_tgt = pos[src, 0], pos[tgt, 0]
            correct = (x_tgt <= x_src) | self_loops
            violation_mag = torch.clamp(x_tgt[~correct] - x_src[~correct], min=0)
        else:
            y_src, y_tgt = pos[src, 1], pos[tgt, 1]
            correct = (y_tgt >= y_src) | self_loops
            violation_mag = torch.clamp(y_src[~correct] - y_tgt[~correct], min=0)

        n_violations = (~correct).sum().item()
        result.update(
            {
                "dag_consistency": correct.float().mean().item(),
                "dag_num_violations": int(n_violations),
                "dag_mean_violation_mag": violation_mag.mean().item() if n_violations > 0 else 0.0,
                "dag_max_violation_mag": violation_mag.max().item() if n_violations > 0 else 0.0,
            }
        )

    result["back_edge_count"] = int(n_back)
    result["back_edge_fraction"] = n_back / total_edges if total_edges > 0 else 0.0
    return result


def depth_position_correlation(
    pos: torch.Tensor,
    topo_depth: torch.Tensor,
    direction: str = "TB",
) -> Dict[str, Optional[float]]:
    """Correlate topological depth with the declared flow-axis projection.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    topo_depth : torch.Tensor
        Pre-layout topological ranks with shape ``[N]``.
    direction : str, optional
        Declared flow direction, one of ``TB``, ``BT``, ``LR``, or ``RL``.

    Returns
    -------
    Dict[str, Optional[float]]
        Spearman correlation and p-value. Both are ``None`` when either
        input is constant, making depth order inapplicable.
    """
    from scipy.stats import spearmanr

    depth_np = (
        topo_depth.cpu().numpy() if isinstance(topo_depth, torch.Tensor) else np.array(topo_depth)
    )
    axis, sign = (
        (0, 1.0)
        if direction == "LR"
        else (0, -1.0)
        if direction == "RL"
        else ((1, -1.0) if direction == "BT" else (1, 1.0))
    )
    projection_np = (sign * pos[:, axis]).cpu().numpy()
    if np.unique(depth_np).size < 2 or np.unique(projection_np).size < 2:
        return {"depth_spearman_rho": None, "depth_spearman_pval": None}
    rho, pval = spearmanr(depth_np, projection_np)
    if not math.isfinite(float(rho)):
        return {"depth_spearman_rho": None, "depth_spearman_pval": None}
    return {"depth_spearman_rho": float(rho), "depth_spearman_pval": float(pval)}


def count_overlaps_detailed(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    *,
    seed: Optional[int] = None,
) -> Dict[str, int]:
    """Count overlapping node bounding box pairs using spatial hashing.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node size tensor with shape ``[N, 2]`` containing width and height.
    seed : Optional[int], optional
        Seed for the capped per-cell subsampling path. ``None`` preserves the
        existing stochastic behavior by using the global RNG state.

    Returns
    -------
    Dict[str, int]
        Overlap statistics containing ``overlap_count``.

    Notes
    -----
    Complexity is ``O(N)`` expected with proper cell sizing. The stochastic
    path only applies on large graphs when a crowded hash cell is capped to
    avoid quadratic work.
    """
    n = pos.shape[0]
    if n <= 1:
        return {"overlap_count": 0}

    half_w = node_sizes[:, 0] / 2
    half_h = node_sizes[:, 1] / 2

    # For small graphs, use exact pairwise
    if n <= 2000:
        dx = (pos[:, 0].unsqueeze(0) - pos[:, 0].unsqueeze(1)).abs()
        dy = (pos[:, 1].unsqueeze(0) - pos[:, 1].unsqueeze(1)).abs()
        min_dx = half_w.unsqueeze(0) + half_w.unsqueeze(1)
        min_dy = half_h.unsqueeze(0) + half_h.unsqueeze(1)
        overlapping = (dx < min_dx) & (dy < min_dy)
        overlapping.fill_diagonal_(False)
        return {"overlap_count": int(overlapping.triu(diagonal=1).sum().item())}

    # Spatial hash: O(N) expected
    cell_size = max(node_sizes[:, 0].max().item(), node_sizes[:, 1].max().item()) + 1.0
    cx = torch.floor(pos[:, 0] / cell_size).long()
    cy = torch.floor(pos[:, 1] / cell_size).long()
    cx_min, cy_min = cx.min(), cy.min()
    cx_rel = cx - cx_min
    cy_rel = cy - cy_min
    cy_range = max(int(cy_rel.max().item()) + 1, 1)
    cell_hash = cx_rel * cy_range + cy_rel

    sorted_idx = cell_hash.argsort()
    sorted_hash = cell_hash[sorted_idx]

    changes = torch.where(sorted_hash[1:] != sorted_hash[:-1])[0] + 1
    starts = torch.cat([torch.zeros(1, dtype=torch.long), changes])
    ends = torch.cat([changes, torch.tensor([n], dtype=torch.long)])
    cell_sizes_arr = ends - starts

    # Only check cells with 2+ nodes (cap per-cell to avoid O(N²) degenerate case)
    multi_mask = cell_sizes_arr >= 2
    multi_starts = starts[multi_mask]
    multi_ends = ends[multi_mask]

    overlap_count = 0
    n_cells = multi_starts.shape[0]
    gen = None if seed is None else torch.Generator(device="cpu").manual_seed(int(seed))

    # Vectorized check per cell, capped at 200 nodes per cell
    for i in range(min(int(n_cells), 100000)):
        s = int(multi_starts[i].item())
        e = int(multi_ends[i].item())
        cell_nodes = sorted_idx[s:e]
        m = cell_nodes.shape[0]
        if m > 200:
            cell_nodes = cell_nodes[torch.randperm(m, generator=gen)[:200]]
            m = 200

        p = pos[cell_nodes]
        hw_c = half_w[cell_nodes]
        hh_c = half_h[cell_nodes]

        dx = (p[:, 0].unsqueeze(1) - p[:, 0].unsqueeze(0)).abs()
        dy = (p[:, 1].unsqueeze(1) - p[:, 1].unsqueeze(0)).abs()
        min_dx = hw_c.unsqueeze(1) + hw_c.unsqueeze(0)
        min_dy = hh_c.unsqueeze(1) + hh_c.unsqueeze(0)
        overlapping = (dx < min_dx) & (dy < min_dy)
        overlapping.fill_diagonal_(False)
        overlap_count += int(overlapping.triu(diagonal=1).sum().item())

    return {"overlap_count": overlap_count}


def aspect_ratio(pos: torch.Tensor, node_sizes: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """Bounding box width / height.

    Complexity: O(N).
    """
    if pos.shape[0] == 0:
        return {"aspect_ratio": 1.0, "bbox_width": 0.0, "bbox_height": 0.0}

    if node_sizes is not None:
        x_min = (pos[:, 0] - node_sizes[:, 0] / 2).min().item()
        x_max = (pos[:, 0] + node_sizes[:, 0] / 2).max().item()
        y_min = (pos[:, 1] - node_sizes[:, 1] / 2).min().item()
        y_max = (pos[:, 1] + node_sizes[:, 1] / 2).max().item()
    else:
        x_min, x_max = pos[:, 0].min().item(), pos[:, 0].max().item()
        y_min, y_max = pos[:, 1].min().item(), pos[:, 1].max().item()

    w = x_max - x_min
    h = y_max - y_min
    return {
        "aspect_ratio": w / max(h, 1e-6),
        "bbox_width": w,
        "bbox_height": h,
    }


def edge_direction_straightness(
    pos: torch.Tensor, edge_index: torch.Tensor, direction: str = "TB"
) -> Dict[str, float]:
    """Mean angular deviation from primary axis (degrees).

    For straight-line edges (no bezier).  For TB/BT: deviation from vertical.
    For LR/RL: deviation from horizontal.

    Complexity: O(|E|).
    Target: mean < 15° is good.
    """
    if edge_index.numel() == 0:
        return {"edge_straightness_mean_deg": 0.0, "edge_straightness_below_15": 1.0}

    src, tgt = edge_index[0], edge_index[1]
    dx = (pos[tgt, 0] - pos[src, 0]).abs()
    dy = (pos[tgt, 1] - pos[src, 1]).abs()

    # Clamp only the denominator for each layout direction so a zero-length
    # edge deterministically reports 0 deg regardless of TB vs LR. Previously
    # LR clamped both dx and dy, making coincident edges report 45 deg.
    if direction in ("LR", "RL"):
        dx = dx.clamp(min=1e-6)
        angles = torch.atan2(dy, dx) * 180 / torch.pi
    else:
        dy = dy.clamp(min=1e-6)
        angles = torch.atan2(dx, dy) * 180 / torch.pi

    return {
        "edge_straightness_mean_deg": angles.mean().item(),
        "edge_straightness_below_15": (angles < 15).float().mean().item(),
    }


# ---------------------------------------------------------------------------
# Tier 2: Sampled metrics
# ---------------------------------------------------------------------------

_COMMON_WEIGHTS: Dict[str, float] = {
    "ksm_score": 25.0,
    "edge_crossing_score": 20.0,
    "node_occlusion_score": 13.0,
    "neighborhood_preservation_score": 12.0,
    "edge_length_deviation_score": 7.0,
    "gabriel_score": 5.0,
    "crossing_angle_score": 5.0,
    "angular_resolution_score": 4.0,
    "path_continuity_score": 4.0,
    "cluster_silhouette_score": 5.0,
}
_DIRECTED_WEIGHTS: Dict[str, float] = {"directed_flow_score": 16.0, "depth_order_score": 9.0}
_CLUSTER_WEIGHTS: Dict[str, float] = {
    "cluster_exclusion_score": 2.0,
    "cluster_sibling_overlap_score": 2.0,
    "cluster_nesting_fidelity_score": 1.0,
    "cluster_edge_intrusion_score": 1.0,
    "cluster_label_occlusion_score": 0.5,
    "cluster_compactness_score": 0.5,
}
_CROSSING_ANGLE_KNEE_RADIANS = math.radians(70.0)
_CLUSTER_PAIR_SAMPLE_CAP = 20_000


def _pav_fitted_values(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit a non-decreasing sequence to observations using weighted PAV.

    Parameters
    ----------
    x : numpy.ndarray
        One-dimensional predictor values.
    y : numpy.ndarray
        One-dimensional response values aligned with ``x``.

    Returns
    -------
    numpy.ndarray
        Isotonic fitted response in the original observation order.
    """
    order = np.argsort(x, kind="stable")
    sorted_x = x[order]
    sorted_y = y[order]
    unique_x, inverse = np.unique(sorted_x, return_inverse=True)
    del unique_x
    counts = np.bincount(inverse).astype(np.float64)
    sums = np.bincount(inverse, weights=sorted_y)
    means = sums / counts

    block_values: List[float] = []
    block_weights: List[float] = []
    block_starts: List[int] = []
    block_ends: List[int] = []
    for index, (value, weight) in enumerate(zip(means, counts)):
        block_values.append(float(value))
        block_weights.append(float(weight))
        block_starts.append(index)
        block_ends.append(index + 1)
        while len(block_values) >= 2 and block_values[-2] > block_values[-1]:
            merged_weight = block_weights[-2] + block_weights[-1]
            merged_value = (
                block_values[-2] * block_weights[-2] + block_values[-1] * block_weights[-1]
            ) / merged_weight
            block_values[-2:] = [merged_value]
            block_weights[-2:] = [merged_weight]
            block_ends[-2:] = [block_ends[-1]]
            block_starts.pop()

    fitted_groups = np.empty(len(means), dtype=np.float64)
    for value, start, end in zip(block_values, block_starts, block_ends):
        fitted_groups[start:end] = value
    fitted_sorted = fitted_groups[inverse]
    fitted = np.empty_like(fitted_sorted)
    fitted[order] = fitted_sorted
    return fitted


def _stratified_graph_pairs(
    edge_index: torch.Tensor,
    num_nodes: int,
    n_sources: int,
    n_targets: int,
    all_pairs_dist: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select deterministic within-component pairs stratified by graph distance.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    n_sources : int
        Maximum number of deterministic BFS source rows.
    n_targets : int
        Maximum number of targets retained per source.
    all_pairs_dist : Optional[numpy.ndarray], optional
        Precomputed unweighted shortest paths with shape ``[N, N]``.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Source indices, target indices, and positive graph distances.
    """
    if num_nodes < 2 or edge_index.numel() == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty
    offsets, targets = _build_csr(_ensure_cpu(edge_index), num_nodes)
    unseen = set(range(num_nodes))
    components: List[np.ndarray] = []
    while unseen:
        root = min(unseen)
        stack = [root]
        component_nodes: List[int] = []
        unseen.remove(root)
        while stack:
            node = stack.pop()
            component_nodes.append(node)
            for offset in range(offsets[node], offsets[node + 1]):
                neighbor = int(targets[offset])
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(np.asarray(sorted(component_nodes), dtype=np.int64))

    # At least one row per nontrivial component prevents a small component
    # from disappearing under a global evenly-spaced sample.
    nontrivial = [component for component in components if len(component) > 1]
    source_budget = max(n_sources, len(nontrivial))
    source_parts: List[np.ndarray] = []
    remaining = source_budget
    remaining_nodes = sum(len(component) for component in nontrivial)
    for component_index, component in enumerate(nontrivial):
        components_left = len(nontrivial) - component_index
        proportional = round(remaining * len(component) / max(remaining_nodes, 1))
        allocation = max(1, min(len(component), proportional, remaining - components_left + 1))
        selected = _deterministic_sample_indices(len(component), allocation)
        source_parts.append(component[selected])
        remaining -= allocation
        remaining_nodes -= len(component)
    sources = np.concatenate(source_parts) if source_parts else np.empty(0, dtype=np.int64)
    pair_sources: List[int] = []
    pair_targets: List[int] = []
    pair_distances: List[int] = []
    for source in sources:
        distances = (
            _bfs_distances(offsets, targets, int(source), max_dist=num_nodes)
            if all_pairs_dist is None
            else all_pairs_dist[int(source)]
        )
        reachable = np.flatnonzero(distances > 0)
        if reachable.size == 0:
            continue
        strata = np.unique(distances[reachable])
        target_budget = max(n_targets, len(strata))
        base = target_budget // len(strata)
        remainder = target_budget % len(strata)
        for stratum_index, distance in enumerate(strata):
            candidates = reachable[distances[reachable] == distance]
            budget = base + (1 if stratum_index < remainder else 0)
            if target_budget >= reachable.size:
                selected = candidates
            else:
                selected = candidates[_deterministic_sample_indices(len(candidates), budget)]
            pair_sources.extend([int(source)] * len(selected))
            pair_targets.extend(int(target) for target in selected)
            pair_distances.extend([int(distance)] * len(selected))
    return (
        np.asarray(pair_sources, dtype=np.int64),
        np.asarray(pair_targets, dtype=np.int64),
        np.asarray(pair_distances, dtype=np.int64),
    )


def isotonic_stress(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: Optional[int] = None,
    n_sources: int = 200,
    n_targets: int = 1000,
    all_pairs_dist: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute aspect-preserving isotonic Kruskal stress on raw coordinates.

    Parameters
    ----------
    pos : torch.Tensor
        Raw node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : Optional[int], optional
        Node count. Defaults to the first dimension of ``pos``.
    n_sources : int, optional
        Deterministic BFS source-row budget.
    n_targets : int, optional
        Per-source target budget, stratified over every observed distance.
    all_pairs_dist : Optional[numpy.ndarray], optional
        Precomputed unweighted shortest paths with shape ``[N, N]``.

    Returns
    -------
    Dict[str, float]
        ``ksm_score`` in ``[0, 1]`` and deterministic sample counts.
    """
    positions = _ensure_cpu(pos).to(dtype=torch.float64).numpy()
    n = int(pos.shape[0]) if num_nodes is None else int(num_nodes)
    sources, targets, graph_distances = _stratified_graph_pairs(
        edge_index, n, n_sources, n_targets, all_pairs_dist=all_pairs_dist
    )
    if sources.size == 0:
        return {"ksm_score": 0.0, "ksm_se": 0.0, "ksm_n_pairs": 0, "ksm_n_sources": 0}
    geometric = np.linalg.norm(positions[sources] - positions[targets], axis=1)
    denominator = float(np.dot(geometric, geometric))
    if denominator <= 1e-24:
        return {
            "ksm_score": 0.0,
            "ksm_se": 0.0,
            "ksm_n_pairs": int(sources.size),
            "ksm_n_sources": int(np.unique(sources).size),
        }
    fitted = _pav_fitted_values(graph_distances.astype(np.float64), geometric)
    residual = geometric - fitted
    squared_residual = residual**2
    squared_geometric = geometric**2
    mean_residual = float(squared_residual.mean())
    mean_geometric = float(squared_geometric.mean())
    stress = math.sqrt(mean_residual / mean_geometric)
    if len(residual) > 1 and stress > 1e-12:
        covariance = np.cov(squared_residual, squared_geometric, ddof=1)
        gradient = np.asarray(
            [
                -0.5 / (stress * mean_geometric),
                0.5 * stress / mean_geometric,
            ]
        )
        variance = float(gradient @ covariance @ gradient) / len(residual)
        standard_error = math.sqrt(max(0.0, variance))
    else:
        standard_error = 0.0
    return {
        "ksm_score": max(0.0, min(1.0, 1.0 - stress)),
        "ksm_se": standard_error,
        "ksm_n_pairs": int(sources.size),
        "ksm_n_sources": int(np.unique(sources).size),
    }


def sampled_stress(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    n_sources: int = 200,
    n_targets: int = 1000,
    max_dist: int = 20,
    all_pairs_dist: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Estimate graph-theoretic stress from deterministic source samples.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes in the graph.
    n_sources : int, optional
        Maximum number of BFS source nodes to sample.
    n_targets : int, optional
        Maximum number of reachable targets to evaluate per source.
    max_dist : int, optional
        Maximum graph distance to explore during BFS.
    all_pairs_dist : Optional[numpy.ndarray], optional
        Precomputed unweighted shortest-path matrix with shape ``[N, N]``.
        When provided, this avoids recomputing graph distances inside repeated
        metric evaluations.

    Returns
    -------
    Dict[str, float]
        Sampled stress value and sampling metadata.

    Notes
    -----
    Positions are normalized to the unit square before computing Euclidean
    distances so stress values are comparable across different output scales.
    Sampling is deterministic to keep comparisons reproducible.
    """
    N = num_nodes

    ei = _ensure_cpu(edge_index)
    if ei.numel() == 0 or N < 2:
        return {
            "sampled_stress": 0.0,
            "stress_n_pairs": 0,
            "stress_n_sources": 0,
            "stress_n_targets": 0,
        }

    pos_cpu = _ensure_cpu(pos).to(dtype=torch.float32)
    mins = pos_cpu.min(dim=0).values
    maxs = pos_cpu.max(dim=0).values
    span = (maxs - mins).clamp(min=1e-6)
    pos_norm = (pos_cpu - mins) / span
    pos_np = pos_norm.numpy()

    if all_pairs_dist is None:
        all_pairs_dist = _cached_all_pairs_unweighted(ei, N, max_dist=max_dist)
    sources = _deterministic_sample_indices(N, min(n_sources, N))

    total_stress = 0.0
    count = 0

    for src_node in sources:
        dist = all_pairs_dist[int(src_node)]

        reachable = np.where(dist > 0)[0]
        if len(reachable) == 0:
            continue

        target_indices = _deterministic_sample_indices(
            len(reachable), min(n_targets, len(reachable))
        )
        targets = reachable[target_indices]
        d_graph = dist[targets].astype(np.float64)
        d_euclidean = np.linalg.norm(pos_np[targets] - pos_np[src_node], axis=1)

        w = 1.0 / (d_graph**2)
        stress = (w * (d_graph - d_euclidean) ** 2).sum()
        total_stress += stress
        count += len(targets)

    return {
        "sampled_stress": total_stress / max(count, 1),
        "stress_n_pairs": count,
        "stress_n_sources": len(sources),
        "stress_n_targets": n_targets,
    }


def sampled_crossing_rate(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    n_samples: int = 1_000_000,
    *,
    seed: Optional[int] = None,
    _geometry: Optional[_CrossingGeometry] = None,
) -> Dict[str, float]:
    """Estimate crossing rate via random edge-pair sampling.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    n_samples : int, optional
        Maximum number of edge pairs to sample.
    seed : Optional[int], optional
        Seed for edge-pair sampling. ``None`` preserves the existing stochastic
        behavior by using the global RNG state.
    _geometry : Optional[_CrossingGeometry], optional
        Precomputed internal geometry used to share one frozen pair set in
        :func:`full`.

    Returns
    -------
    Dict[str, float]
        Estimated crossing statistics for the sampled edge pairs. The
        ``crossing_estimated_total`` and ``crossing_se`` fields are in crossing
        count units over eligible non-adjacent edge pairs, while
        ``crossing_rate`` is conditional on eligible sampled pairs.

    Notes
    -----
    Complexity is ``O(n_samples)``.
    """
    if _geometry is not None:
        return _crossing_rate_from_geometry(_geometry)
    if edge_index.numel() == 0 or edge_index.shape[1] < 2:
        return {
            "crossing_rate": 0.0,
            "crossing_se": 0.0,
            "crossing_estimated_total": 0,
            "crossing_n_samples": 0,
            "crossing_eligible_pairs": 0.0,
        }

    E = edge_index.shape[1]
    src, tgt = edge_index[0], edge_index[1]
    total_pairs = E * (E - 1) // 2
    gen = None if seed is None else torch.Generator(device="cpu").manual_seed(int(seed))

    # When the request would exhaust the pair space, enumerate all pairs
    # exactly. Otherwise sample without replacement so the same pair is
    # never double-counted (under the prior implementation, sampling was
    # with replacement, which biased the estimator on small graphs and
    # gave the polish picker a noisy oracle).
    if total_pairs <= n_samples:
        # Enumerate (i, j) with i < j over edge ids.
        ii, jj = torch.triu_indices(E, E, offset=1)
        idx1 = ii
        idx2 = jj
        pair_space_exhausted = True
    else:
        # Without-replacement sample of unordered pairs (i, j), i < j.
        # Encode pair as p = i * E + j; sample without replacement from the
        # set of valid p values via randperm of total_pairs and decode.
        # For very large E*(E-1)/2 the randperm allocation can be costly;
        # fall back to random pair ids with deduplication.
        if total_pairs <= 10_000_000:
            perm = torch.randperm(total_pairs, generator=gen)[:n_samples]
            # Decode triangular indices.
            ii, jj = torch.triu_indices(E, E, offset=1)
            idx1 = ii[perm]
            idx2 = jj[perm]
            pair_space_exhausted = False
        else:
            idx1 = torch.randint(0, E, (n_samples,), generator=gen)
            idx2 = torch.randint(0, E, (n_samples,), generator=gen)
            # Force i < j to canonicalize and let the same-edge filter drop
            # equal pairs. With-replacement bias remains for huge E only.
            lo = torch.minimum(idx1, idx2)
            hi = torch.maximum(idx1, idx2)
            idx1, idx2 = lo, hi
            pair_space_exhausted = False

    # Exclude pairs sharing a node
    e1s, e1t = src[idx1], tgt[idx1]
    e2s, e2t = src[idx2], tgt[idx2]
    shares_node = (e1s == e2s) | (e1s == e2t) | (e1t == e2s) | (e1t == e2t)
    same_edge = idx1 == idx2
    valid = ~shares_node & ~same_edge

    if valid.sum() == 0:
        return {
            "crossing_rate": 0.0,
            "crossing_se": 0.0,
            "crossing_estimated_total": 0,
            "crossing_n_samples": 0,
            "crossing_eligible_pairs": 0.0,
        }

    p1 = pos[e1s[valid]]
    p2 = pos[e1t[valid]]
    p3 = pos[e2s[valid]]
    p4 = pos[e2t[valid]]

    crossings = segments_intersect(p1, p2, p3, p4)
    n_valid = int(valid.sum().item())
    rate = crossings.float().mean().item()
    if pair_space_exhausted:
        eligible_pairs = float(n_valid)
        estimated_total = int(crossings.sum().item())
        se = 0.0
    else:
        eligible_pairs = float(total_pairs) * (float(n_valid) / float(valid.numel()))
        estimated_total = int(rate * eligible_pairs)
        se = eligible_pairs * (rate * (1 - rate) / n_valid) ** 0.5 if n_valid > 0 else 0.0

    return {
        "crossing_rate": rate,
        "crossing_se": se,
        "crossing_estimated_total": estimated_total,
        "crossing_n_samples": n_valid,
        "crossing_eligible_pairs": eligible_pairs,
    }


def _crossing_rate_from_geometry(geometry: _CrossingGeometry) -> Dict[str, float]:
    """Compute frozen crossing statistics from precomputed batched geometry.

    Parameters
    ----------
    geometry : _CrossingGeometry
        Eligible pair geometry and candidate-pair sampling metadata.

    Returns
    -------
    Dict[str, float]
        Crossing-rate fields matching :func:`sampled_crossing_rate`.
    """
    crossings = geometry.crossing
    n_valid = int(crossings.numel())
    if n_valid == 0:
        return {
            "crossing_rate": 0.0,
            "crossing_se": 0.0,
            "crossing_estimated_total": 0,
            "crossing_n_samples": 0,
            "crossing_eligible_pairs": 0.0,
        }
    rate = crossings.float().mean().item()
    if geometry.pair_space_exhausted:
        eligible_pairs = float(n_valid)
        estimated_total = int(crossings.sum().item())
        se = 0.0
    else:
        eligible_pairs = float(geometry.total_pairs) * (
            float(n_valid) / float(geometry.candidate_pairs)
        )
        estimated_total = int(rate * eligible_pairs)
        se = eligible_pairs * (rate * (1 - rate) / n_valid) ** 0.5
    return {
        "crossing_rate": rate,
        "crossing_se": se,
        "crossing_estimated_total": estimated_total,
        "crossing_n_samples": n_valid,
        "crossing_eligible_pairs": eligible_pairs,
    }


def _legacy_neighborhood_preservation(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    n_samples: int = 5000,
    k: int = 10,
    all_pairs_dist: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Fraction of k-nearest graph neighbors that are also k-nearest Euclidean neighbors.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes in the graph.
    n_samples : int, optional
        Maximum number of sampled source nodes.
    k : int, optional
        Number of nearest graph and Euclidean neighbors to compare.
    all_pairs_dist : Optional[numpy.ndarray], optional
        Precomputed unweighted shortest-path matrix with shape ``[N, N]``.
        When provided, this avoids recomputing graph distances inside repeated
        metric evaluations.

    Returns
    -------
    Dict[str, float]
        Mean, median, and standard deviation of sampled neighborhood
        preservation scores.

    Notes
    -----
    Target: > 0.5 good, > 0.7 excellent.
    """
    pos_np = pos.cpu().numpy()
    ei = _ensure_cpu(edge_index)
    N = num_nodes

    if ei.numel() == 0 or N < k + 1:
        return {"neighborhood_mean": 0.0, "neighborhood_median": 0.0, "neighborhood_std": 0.0}

    if all_pairs_dist is None:
        all_pairs_dist = _cached_all_pairs_unweighted(ei, N, max_dist=k)
    samples = torch.randperm(N)[: min(n_samples, N)].numpy()

    scores = []
    for node in samples:
        # k nearest graph neighbors via BFS
        dist = all_pairs_dist[int(node)]
        reachable = np.where((dist > 0) & (dist <= k))[0]
        if len(reachable) < k:
            graph_neighbors = set(reachable.tolist())
        else:
            # Take the k closest by graph distance
            order = np.argsort(dist[reachable])[:k]
            graph_neighbors = set(reachable[order].tolist())

        if len(graph_neighbors) == 0:
            continue

        # k nearest Euclidean neighbors
        dists_eucl = np.linalg.norm(pos_np - pos_np[node], axis=1)
        dists_eucl[node] = np.inf
        eucl_neighbors = set(np.argpartition(dists_eucl, k)[:k].tolist())

        overlap = len(graph_neighbors & eucl_neighbors)
        scores.append(overlap / min(k, len(graph_neighbors)))

    if not scores:
        return {"neighborhood_mean": 0.0, "neighborhood_median": 0.0, "neighborhood_std": 0.0}

    scores_arr = np.array(scores)
    return {
        "neighborhood_mean": float(scores_arr.mean()),
        "neighborhood_median": float(np.median(scores_arr)),
        "neighborhood_std": float(scores_arr.std()),
    }


def neighborhood_preservation(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: Optional[int] = None,
    n_samples: int = 5000,
    k: Optional[int] = None,
    all_pairs_dist: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Measure adjacency versus layout-neighborhood Jaccard preservation.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : Optional[int], optional
        Node count. Defaults to ``pos.shape[0]``.
    n_samples : int, optional
        Maximum number of frozen, evenly spaced query rows.
    k : Optional[int], optional
        Layout-neighborhood size. Defaults to ``max(1, floor(2m/n))``.
    all_pairs_dist : Optional[numpy.ndarray], optional
        Unused compatibility argument retained for callers of the old metric.

    Returns
    -------
    Dict[str, float]
        Pooled fractional-tie Jaccard score and sampling metadata.
    """
    del all_pairs_dist
    positions = _ensure_cpu(pos).to(dtype=torch.float64)
    edges = _ensure_cpu(edge_index).to(dtype=torch.long)
    n = int(positions.shape[0]) if num_nodes is None else int(num_nodes)
    m = int(edges.shape[1]) if edges.numel() else 0
    effective_k = max(1, math.floor(2 * m / n)) if k is None and n > 0 else int(k or 1)
    effective_k = min(effective_k, max(0, n - 1))
    if n < 2 or effective_k == 0:
        return {
            "neighborhood_preservation_score": 0.0,
            "neighborhood_mean": 0.0,
            "neighborhood_se": 0.0,
            "neighborhood_n_rows": 0,
            "neighborhood_k": effective_k,
        }

    adjacency: List[set[int]] = [set() for _ in range(n)]
    for source, target in edges.t().tolist() if edges.numel() else []:
        if source != target:
            adjacency[int(source)].add(int(target))
            adjacency[int(target)].add(int(source))

    rows_np = _deterministic_sample_indices(n, min(n, n_samples))
    rows = torch.from_numpy(rows_np).to(dtype=torch.long)
    distances = torch.cdist(positions[rows], positions)
    distances[torch.arange(len(rows)), rows] = float("inf")
    intersection_total = 0.0
    union_total = 0.0
    row_scores: List[float] = []
    for row_offset, node in enumerate(rows_np.tolist()):
        row = distances[row_offset]
        boundary = float(torch.kthvalue(row, effective_k).values.item())
        strictly_closer = row < boundary
        boundary_tensor = torch.tensor(boundary, dtype=row.dtype)
        tied = torch.isclose(row, boundary_tensor, rtol=1e-12, atol=1e-12)
        remaining = effective_k - int(strictly_closer.sum().item())
        tie_count = int(tied.sum().item())
        tie_weight = remaining / tie_count if tie_count else 0.0
        adjacent_indices = sorted(adjacency[node])
        if adjacent_indices:
            adjacent_distances = row[adjacent_indices]
            adjacent_ties = torch.isclose(
                adjacent_distances, boundary_tensor, rtol=1e-12, atol=1e-12
            )
            intersection = float((adjacent_distances < boundary).sum().item())
            intersection += tie_weight * float(adjacent_ties.sum().item())
        else:
            intersection = 0.0
        # Both memberships have known mass (degree and k), so pooled Jaccard
        # needs no dense per-row membership allocation.
        union = len(adjacent_indices) + effective_k - intersection
        intersection_total += intersection
        union_total += union
        row_scores.append(intersection / union if union > 0.0 else 0.0)

    score = intersection_total / union_total if union_total > 0.0 else 0.0
    return {
        "neighborhood_preservation_score": max(0.0, min(1.0, score)),
        "neighborhood_mean": float(np.mean(row_scores)) if row_scores else 0.0,
        "neighborhood_se": (
            float(np.std(row_scores, ddof=1) / math.sqrt(len(row_scores)))
            if len(row_scores) > 1
            else 0.0
        ),
        "neighborhood_n_rows": int(len(rows_np)),
        "neighborhood_k": effective_k,
    }


def angular_resolution(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    n_samples: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """Minimum angle between incident edges at each node (sampled).

    Complexity: O(n_samples * avg_degree * log(avg_degree)).
    Target: mean > 20° decent, < 10% below 10°.

    The ``seed`` parameter makes node sampling deterministic; previously the
    global RNG was used, which injected run-to-run variance into benchmark
    scores and masked small improvements from iteration to iteration.
    """
    if edge_index.numel() == 0:
        return {
            "angular_res_mean_deg": 360.0,
            "angular_res_median_deg": 360.0,
            "angular_res_below_10deg": 0.0,
        }

    src, tgt = edge_index[0], edge_index[1]
    N = pos.shape[0]
    PI = torch.pi

    # Build undirected edge angles per node
    all_src = torch.cat([src, tgt])
    all_tgt = torch.cat([tgt, src])
    dx = pos[all_tgt, 0] - pos[all_src, 0]
    dy = pos[all_tgt, 1] - pos[all_src, 1]
    angles = torch.atan2(dy, dx)  # [-pi, pi]

    # Count degree per node
    degree = torch.zeros(N, dtype=torch.long)
    degree.scatter_add_(0, all_src, torch.ones(all_src.shape[0], dtype=torch.long))

    # Only sample nodes with degree >= 2
    candidates = torch.where(degree >= 2)[0]
    if candidates.numel() == 0:
        return {
            "angular_res_mean_deg": 360.0,
            "angular_res_median_deg": 360.0,
            "angular_res_below_10deg": 0.0,
        }

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    sample = candidates[
        torch.randperm(candidates.numel(), generator=gen)[: min(n_samples, candidates.numel())]
    ]

    min_angles = []
    for node in sample:
        mask = all_src == node
        node_angles = angles[mask].sort().values
        if node_angles.numel() < 2:
            continue
        diffs = node_angles[1:] - node_angles[:-1]
        wrap = 2 * PI - (node_angles[-1] - node_angles[0])
        all_diffs = torch.cat([diffs, wrap.unsqueeze(0)])
        min_angles.append(all_diffs.min().item())

    if not min_angles:
        return {
            "angular_res_mean_deg": 360.0,
            "angular_res_median_deg": 360.0,
            "angular_res_below_10deg": 0.0,
        }

    min_angles_t = torch.tensor(min_angles)
    deg = min_angles_t * 180 / PI

    return {
        "angular_res_mean_deg": deg.mean().item(),
        "angular_res_median_deg": deg.median().item(),
        "angular_res_below_10deg": (deg < 10).float().mean().item(),
    }


def node_occlusion_score(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    *,
    seed: Optional[int] = 0,
) -> Dict[str, float]:
    """Score node bbox occlusion with the frozen saturating transform.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node bbox sizes with shape ``[N, 2]``.
    seed : Optional[int], optional
        Frozen seed for the large-graph overlap counter.

    Returns
    -------
    Dict[str, float]
        Overlap count and ``node_occlusion_score`` in ``[0, 1]``.
    """
    n = int(pos.shape[0])
    overlaps = count_overlaps_detailed(_ensure_cpu(pos), _ensure_cpu(node_sizes), seed=seed)[
        "overlap_count"
    ]
    score = 0.0 if n == 0 else 1.0 / (1.0 + 2.0 * overlaps / n)
    return {"overlap_count": overlaps, "node_occlusion_score": score}


def _degree_corrected_crossing_max(edge_index: torch.Tensor, num_nodes: int) -> int:
    """Return the degree-corrected maximum number of eligible edge pairs.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    int
        ``C(m,2) - sum_v C(deg_v,2)``, clamped at zero.
    """
    edges = _ensure_cpu(edge_index).to(dtype=torch.long)
    m = int(edges.shape[1]) if edges.numel() else 0
    degree = torch.zeros(num_nodes, dtype=torch.long)
    if edges.numel():
        degree.scatter_add_(0, edges[0], torch.ones(m, dtype=torch.long))
        degree.scatter_add_(0, edges[1], torch.ones(m, dtype=torch.long))
    adjacent_pairs = int((degree * (degree - 1) // 2).sum().item())
    return max(0, m * (m - 1) // 2 - adjacent_pairs)


def edge_crossing_score(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    n_samples: int = 1_000_000,
    *,
    seed: Optional[int] = 0,
    _geometry: Optional[_CrossingGeometry] = None,
) -> Dict[str, float]:
    """Score straight-edge crossings with square-root range restoration.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    n_samples : int, optional
        Frozen edge-pair sampling budget.
    seed : Optional[int], optional
        Sampling seed.
    _geometry : Optional[_CrossingGeometry], optional
        Precomputed internal geometry used by :func:`full`.

    Returns
    -------
    Dict[str, float]
        Crossing count estimate, degree-corrected maximum, and score.
    """
    c_max = _degree_corrected_crossing_max(edge_index, int(pos.shape[0]))
    sampled = sampled_crossing_rate(
        _ensure_cpu(pos),
        _ensure_cpu(edge_index),
        n_samples=n_samples,
        seed=seed,
        _geometry=_geometry,
    )
    crossings = int(sampled["crossing_estimated_total"])
    score = 1.0 if c_max <= 0 else 1.0 - math.sqrt(min(1.0, crossings / c_max))
    if c_max > 0 and 0 < crossings < c_max:
        score_se = float(sampled["crossing_se"]) / (2.0 * math.sqrt(crossings * c_max))
    else:
        score_se = 0.0
    return {
        **sampled,
        "crossing_count": crossings,
        "crossing_c_max": c_max,
        "edge_crossing_score": max(0.0, min(1.0, score)),
        "edge_crossing_score_se": score_se,
    }


def _crossing_pair_geometry(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    n_samples: Optional[int] = None,
    seed: Optional[int] = 0,
) -> _CrossingGeometry:
    """Enumerate eligible edge-pair endpoints and their intersection mask.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    n_samples : Optional[int], optional
        Maximum number of edge pairs. ``None`` enumerates the full pair space.
    seed : int, optional
        Frozen seed used when pair sampling is required.

    Returns
    -------
    _CrossingGeometry
        Four endpoint tensors, crossing mask, and sampling metadata.
    """
    positions = _ensure_cpu(pos)
    edges = _ensure_cpu(edge_index).to(dtype=torch.long)
    m = int(edges.shape[1]) if edges.numel() else 0
    if m < 2:
        empty = torch.empty((0, 2), dtype=positions.dtype)
        return _CrossingGeometry(
            empty, empty, empty, empty, torch.empty(0, dtype=torch.bool), 0, 0, True
        )
    total_pairs = m * (m - 1) // 2
    if n_samples is None or total_pairs <= n_samples:
        first, second = torch.triu_indices(m, m, offset=1)
    elif total_pairs <= 10_000_000:
        generator = None if seed is None else torch.Generator(device="cpu").manual_seed(int(seed))
        selected = torch.randperm(total_pairs, generator=generator)[:n_samples]
        first, second = _triu_pair_indices_from_linear(selected, m)
    else:
        generator = None if seed is None else torch.Generator(device="cpu").manual_seed(int(seed))
        first = torch.randint(0, m, (n_samples,), generator=generator)
        second = torch.randint(0, m, (n_samples,), generator=generator)
        first, second = torch.minimum(first, second), torch.maximum(first, second)
    a, b = edges[0, first], edges[1, first]
    c, d = edges[0, second], edges[1, second]
    eligible = (a != c) & (a != d) & (b != c) & (b != d)
    p1, p2 = positions[a[eligible]], positions[b[eligible]]
    p3, p4 = positions[c[eligible]], positions[d[eligible]]
    return _CrossingGeometry(
        p1,
        p2,
        p3,
        p4,
        segments_intersect(p1, p2, p3, p4),
        total_pairs,
        int(first.numel()),
        n_samples is None or total_pairs <= n_samples,
    )


def _triu_pair_indices_from_linear(
    linear_indices: torch.Tensor,
    matrix_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map upper-triangle offsets to edge-pair indices.

    Parameters
    ----------
    linear_indices : torch.Tensor
        Zero-based offsets into ``torch.triu_indices(matrix_size, matrix_size,
        offset=1)`` order with shape ``[K]``.
    matrix_size : int
        Number of rows/columns in the conceptual square matrix.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Row and column index tensors with shape ``[K]``.
    """
    if linear_indices.numel() == 0:
        empty = torch.empty_like(linear_indices, dtype=torch.long)
        return empty, empty
    k = linear_indices.to(dtype=torch.float64)
    width = float(2 * matrix_size - 1)
    rows = torch.floor((width - torch.sqrt(width * width - 8.0 * k)) * 0.5).to(torch.long)
    before = rows * (2 * matrix_size - rows - 1) // 2
    # Floating-point floor can land one row high at exact triangular
    # boundaries; correct in integer space without materializing the triangle.
    too_high = before > linear_indices
    rows = torch.where(too_high, rows - 1, rows)
    before = rows * (2 * matrix_size - rows - 1) // 2
    cols = rows + 1 + (linear_indices - before)
    return rows.to(dtype=torch.long), cols.to(dtype=torch.long)


def crossing_angle_score(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    n_samples: int = 1_000_000,
    *,
    seed: int = 0,
    _geometry: Optional[_CrossingGeometry] = None,
) -> Dict[str, float]:
    """Score acute crossing angles with the frozen 70-degree plateau.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    n_samples : int, optional
        Deterministic edge-pair budget. Small pair spaces remain exact.
    seed : int, optional
        Frozen sampling seed.
    _geometry : Optional[_CrossingGeometry], optional
        Precomputed internal geometry used by :func:`full`.

    Returns
    -------
    Dict[str, float]
        Mean plateaued crossing-angle score and counted crossings.
    """
    geometry = _geometry or _crossing_pair_geometry(pos, edge_index, n_samples=n_samples, seed=seed)
    p1, p2, p3, p4, crossing = geometry[:5]
    if not bool(crossing.any().item()):
        return {
            "crossing_angle_score": 1.0,
            "crossing_angle_count": 0,
            "crossing_angle_n_pairs": int(crossing.numel()),
        }
    first = p2[crossing] - p1[crossing]
    second = p4[crossing] - p3[crossing]
    norms = torch.linalg.vector_norm(first, dim=1) * torch.linalg.vector_norm(second, dim=1)
    cosine = torch.where(
        norms > 1e-12,
        (first * second).sum(dim=1).abs() / norms.clamp(min=1e-12),
        torch.ones_like(norms),
    ).clamp(0.0, 1.0)
    angles = torch.acos(cosine)
    score = torch.clamp(angles / _CROSSING_ANGLE_KNEE_RADIANS, max=1.0).mean().item()
    return {
        "crossing_angle_score": float(score),
        "crossing_angle_count": int(crossing.sum()),
        "crossing_angle_n_pairs": int(crossing.numel()),
    }


def gabriel_score(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    n_samples: Optional[int] = None,
    node_samples: Optional[int] = None,
) -> Dict[str, float]:
    """Score node intrusions into non-incident edge Gabriel disks.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    n_samples : Optional[int], optional
        Maximum deterministic edge sample. ``None`` evaluates every edge.
    node_samples : Optional[int], optional
        Maximum deterministic node sample per selected edge. ``None`` checks
        every nonincident node.

    Returns
    -------
    Dict[str, float]
        Gabriel intrusion count, maximum, and square-root score.
    """
    positions = _ensure_cpu(pos).to(dtype=torch.float64)
    edges = _ensure_cpu(edge_index).to(dtype=torch.long)
    n = int(positions.shape[0])
    m = int(edges.shape[1]) if edges.numel() else 0
    g_max = m * max(0, n - 2)
    if g_max <= 0:
        return {"gabriel_score": 1.0, "gabriel_intrusions": 0, "gabriel_g_max": g_max}
    sample_count = m if n_samples is None else min(m, n_samples)
    sampled_edges = edges[:, _deterministic_sample_indices(m, sample_count)]
    sampled_intrusions = 0
    node_ids = torch.arange(n)
    if node_samples is not None:
        node_ids = node_ids[_deterministic_sample_indices(n, min(n, node_samples))]
    # Bound the temporary [edges, nodes, 2] tensor while replacing the Python
    # edge loop. Float64 operation order is unchanged from the scalar version.
    edge_batch_size = max(1, min(sample_count, 1_000_000 // max(1, int(node_ids.numel()))))
    sampled_positions = positions[node_ids]
    for start in range(0, sample_count, edge_batch_size):
        edge_batch = sampled_edges[:, start : start + edge_batch_size].t()
        sources = edge_batch[:, 0]
        targets = edge_batch[:, 1]
        midpoints = (positions[sources] + positions[targets]) / 2.0
        radii_sq = ((positions[sources] - positions[targets]) ** 2).sum(dim=1) / 4.0
        nonincident = (node_ids.unsqueeze(0) != sources.unsqueeze(1)) & (
            node_ids.unsqueeze(0) != targets.unsqueeze(1)
        )
        distances_sq = ((sampled_positions.unsqueeze(0) - midpoints.unsqueeze(1)) ** 2).sum(dim=2)
        observed = ((distances_sq < radii_sq.unsqueeze(1) - 1e-12) & nonincident).sum(dim=1)
        nonincident_counts = nonincident.sum(dim=1)
        for observed_count, nonincident_count in zip(
            observed.tolist(), nonincident_counts.tolist()
        ):
            sampled_intrusions += round(
                int(observed_count) * max(0, n - 2) / max(1, int(nonincident_count))
            )
    intrusions = round(sampled_intrusions * m / sample_count) if sample_count else 0
    score = 1.0 - math.sqrt(min(1.0, intrusions / g_max))
    return {
        "gabriel_score": max(0.0, min(1.0, score)),
        "gabriel_intrusions": intrusions,
        "gabriel_g_max": g_max,
        "gabriel_n_edges": sample_count,
        "gabriel_n_nodes_per_edge": int(node_ids.numel()),
    }


def gabriel_ksm_correlation(records: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Compute the engine-blind Gate-E correlation diagnostic.

    Parameters
    ----------
    records : Sequence[Dict[str, float]]
        Label-stripped corpus records containing ``gabriel_score`` and
        ``ksm_score``.

    Returns
    -------
    Dict[str, float]
        Pearson correlation, usable record count, and tripwire indicator.
    """
    pairs = [
        (float(record["gabriel_score"]), float(record["ksm_score"]))
        for record in records
        if "gabriel_score" in record and "ksm_score" in record
    ]
    if len(pairs) < 2:
        return {"gabriel_ksm_correlation": float("nan"), "gate_e_tripwire": 0.0, "n": len(pairs)}
    values = np.asarray(pairs, dtype=np.float64)
    correlation = float(np.corrcoef(values[:, 0], values[:, 1])[0, 1])
    return {
        "gabriel_ksm_correlation": correlation,
        "gate_e_tripwire": float(np.isfinite(correlation) and correlation > 0.9),
        "n": len(pairs),
    }


def path_continuity_score(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    n_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Score straight-through continuity at undirected degree-two vertices.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    n_samples : Optional[int], optional
        Maximum deterministic degree-two vertex sample.

    Returns
    -------
    Dict[str, Any]
        Score in ``[0, 1]`` or ``None`` when no degree-two vertex applies.
    """
    positions = _ensure_cpu(pos).to(dtype=torch.float64)
    edges = _ensure_cpu(edge_index).to(dtype=torch.long)
    neighbors: List[set[int]] = [set() for _ in range(int(positions.shape[0]))]
    for source, target in edges.t().tolist() if edges.numel() else []:
        if source != target:
            neighbors[source].add(target)
            neighbors[target].add(source)
    candidates = [vertex for vertex, adjacent in enumerate(neighbors) if len(adjacent) == 2]
    if n_samples is not None:
        sampled = _deterministic_sample_indices(len(candidates), min(len(candidates), n_samples))
        candidates = [candidates[index] for index in sampled]
    scores: List[float] = []
    for vertex in candidates:
        adjacent = neighbors[vertex]
        first_index, second_index = sorted(adjacent)
        first = positions[first_index] - positions[vertex]
        second = positions[second_index] - positions[vertex]
        norm_product = float(torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second))
        if norm_product <= 1e-12:
            scores.append(0.0)
            continue
        cosine = float(torch.dot(first, second) / norm_product)
        scores.append(math.acos(max(-1.0, min(1.0, cosine))) / math.pi)
    return {
        "path_continuity_score": float(np.mean(scores)) if scores else None,
        "path_continuity_n_vertices": len(scores),
    }


def edge_length_deviation_score(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    declared_targets: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Score reciprocal mean absolute edge-length deviation.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    declared_targets : Optional[torch.Tensor], optional
        Positive target lengths with shape ``[E]``. A least-squares common
        scale is fitted before measuring deviation.

    Returns
    -------
    Dict[str, float]
        Reciprocal deviation score, mean deviation, and fitted scale.
    """
    positions = _ensure_cpu(pos).to(dtype=torch.float64)
    edges = _ensure_cpu(edge_index).to(dtype=torch.long)
    if edges.numel() == 0:
        return {
            "edge_length_deviation_score": 0.0,
            "edge_length_relative_deviation": 0.0,
            "edge_length_target_scale": 1.0,
        }
    lengths = torch.linalg.vector_norm(positions[edges[0]] - positions[edges[1]], dim=1)
    mean_length = float(lengths.mean().item())
    if mean_length <= 1e-12:
        return {
            "edge_length_deviation_score": 0.0,
            "edge_length_relative_deviation": float("inf"),
            "edge_length_target_scale": 0.0,
        }
    if declared_targets is None:
        reference = torch.full_like(lengths, mean_length)
        alpha = 1.0
    else:
        targets = _ensure_cpu(declared_targets).to(dtype=torch.float64)
        if targets.shape != lengths.shape or bool((targets <= 0).any().item()):
            raise ValueError("declared_targets must be positive with shape [E]")
        denominator = float(torch.dot(targets, targets).item())
        alpha = float(torch.dot(lengths, targets).item()) / denominator
        reference = alpha * targets
    reference_mean = float(reference.mean().item())
    relative = float(torch.abs(lengths - reference).mean().item()) / max(reference_mean, 1e-12)
    return {
        "edge_length_deviation_score": 1.0 / (1.0 + relative),
        "edge_length_relative_deviation": relative,
        "edge_length_target_scale": alpha,
    }


def angular_resolution_score(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    n_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Score minimum incident angles relative to each vertex's ideal angle.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    n_samples : Optional[int], optional
        Maximum deterministic sample of vertices with degree at least two.

    Returns
    -------
    Dict[str, Any]
        Frozen angular-resolution score or ``None`` when no vertex applies.
    """
    positions = _ensure_cpu(pos).to(dtype=torch.float64)
    edges = _ensure_cpu(edge_index).to(dtype=torch.long)
    neighbors: List[set[int]] = [set() for _ in range(int(positions.shape[0]))]
    for source, target in edges.t().tolist() if edges.numel() else []:
        if source != target:
            neighbors[source].add(target)
            neighbors[target].add(source)
    candidates = [vertex for vertex, adjacent in enumerate(neighbors) if len(adjacent) >= 2]
    if n_samples is not None:
        sampled = _deterministic_sample_indices(len(candidates), min(len(candidates), n_samples))
        candidates = [candidates[index] for index in sampled]
    penalties: List[float] = []
    for vertex in candidates:
        adjacent = neighbors[vertex]
        degree = len(adjacent)
        vectors = positions[sorted(adjacent)] - positions[vertex]
        if bool((torch.linalg.vector_norm(vectors, dim=1) <= 1e-12).any().item()):
            penalties.append(1.0)
            continue
        angles = torch.sort(torch.atan2(vectors[:, 1], vectors[:, 0])).values
        gaps = torch.cat(
            (angles[1:] - angles[:-1], (2 * torch.pi - angles[-1] + angles[0]).view(1))
        )
        minimum = float(gaps.min().item())
        ideal = 2.0 * math.pi / degree
        penalties.append((ideal - min(minimum, ideal)) / ideal)
    return {
        "angular_resolution_score": 1.0 - float(np.mean(penalties)) if penalties else None,
        "angular_resolution_n_vertices": len(penalties),
    }


def cluster_silhouette_score(pos: torch.Tensor, cluster_ids: torch.Tensor) -> Dict[str, Any]:
    """Score silhouette separation using declared ground-truth labels only.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    cluster_ids : torch.Tensor
        Ground-truth cluster labels with shape ``[N]``.

    Returns
    -------
    Dict[str, Any]
        Shifted silhouette score or ``None`` when fewer than two labels apply.
    """
    positions = _ensure_cpu(pos).to(dtype=torch.float64)
    labels = _ensure_cpu(cluster_ids).reshape(-1)
    unique = torch.unique(labels)
    if len(labels) != len(positions):
        raise ValueError("cluster_ids must have shape [N]")
    if unique.numel() < 2 or len(labels) < 3:
        return {"cluster_silhouette_score": None, "cluster_silhouette": None}
    if float(torch.pdist(positions).pow(2).sum().item()) <= 1e-24:
        return {"cluster_silhouette_score": 0.0, "cluster_silhouette": -1.0}
    distances = torch.cdist(positions, positions)
    silhouettes: List[float] = []
    for index in range(len(labels)):
        same = labels == labels[index]
        same[index] = False
        if not bool(same.any().item()):
            silhouettes.append(0.0)
            continue
        a = float(distances[index, same].mean().item())
        b = min(
            float(distances[index, labels == other].mean().item())
            for other in unique
            if bool((other != labels[index]).item())
        )
        scale = max(a, b)
        silhouettes.append((b - a) / scale if scale > 1e-12 else -1.0)
    silhouette = float(np.mean(silhouettes))
    return {
        "cluster_silhouette_score": max(0.0, min(1.0, (silhouette + 1.0) / 2.0)),
        "cluster_silhouette": silhouette,
    }


def _bbox_area(bounds: Tuple[float, float, float, float]) -> float:
    """Return non-negative area for an axis-aligned bounding box.

    Parameters
    ----------
    bounds : tuple[float, float, float, float]
        Bounds as ``(x_min, y_min, x_max, y_max)``.

    Returns
    -------
    float
        Non-negative box area.
    """
    return max(0.0, bounds[2] - bounds[0]) * max(0.0, bounds[3] - bounds[1])


def _bbox_intersection_area(
    left: Tuple[float, float, float, float],
    right: Tuple[float, float, float, float],
    *,
    tolerance: float = 0.0,
) -> float:
    """Return overlap area for two AABBs after optional touch tolerance.

    Parameters
    ----------
    left : tuple[float, float, float, float]
        First bounds as ``(x_min, y_min, x_max, y_max)``.
    right : tuple[float, float, float, float]
        Second bounds as ``(x_min, y_min, x_max, y_max)``.
    tolerance : float, optional
        Amount of one-dimensional contact ignored on each axis.

    Returns
    -------
    float
        Positive intersection area beyond the tolerance.
    """
    width = min(left[2], right[2]) - max(left[0], right[0]) - float(tolerance)
    height = min(left[3], right[3]) - max(left[1], right[1]) - float(tolerance)
    return max(0.0, width) * max(0.0, height)


def _bbox_contains(
    outer: Tuple[float, float, float, float],
    inner: Tuple[float, float, float, float],
    *,
    tolerance: float = 1e-9,
) -> bool:
    """Return whether ``outer`` encloses ``inner`` within tolerance.

    Parameters
    ----------
    outer : tuple[float, float, float, float]
        Candidate enclosing bounds.
    inner : tuple[float, float, float, float]
        Candidate enclosed bounds.
    tolerance : float, optional
        Coordinate tolerance for exact-touch numerical noise.

    Returns
    -------
    bool
        ``True`` when every inner side lies inside the outer box.
    """
    return (
        outer[0] <= inner[0] + tolerance
        and outer[1] <= inner[1] + tolerance
        and outer[2] >= inner[2] - tolerance
        and outer[3] >= inner[3] - tolerance
    )


def _segment_aabb_clip_length(
    start: np.ndarray,
    end: np.ndarray,
    bounds: Tuple[float, float, float, float],
) -> float:
    """Return segment length inside an AABB interior via Liang-Barsky clipping.

    Parameters
    ----------
    start : numpy.ndarray
        Segment start point with shape ``[2]``.
    end : numpy.ndarray
        Segment end point with shape ``[2]``.
    bounds : tuple[float, float, float, float]
        Box bounds as ``(x_min, y_min, x_max, y_max)``.

    Returns
    -------
    float
        Length of the segment portion inside the box. Boundary-only contact has
        zero length.
    """
    delta = end - start
    t_min = 0.0
    t_max = 1.0
    for axis, lower, upper in ((0, bounds[0], bounds[2]), (1, bounds[1], bounds[3])):
        if abs(float(delta[axis])) <= 1e-12:
            if start[axis] <= lower or start[axis] >= upper:
                return 0.0
            continue
        inv_delta = 1.0 / float(delta[axis])
        first = (lower - float(start[axis])) * inv_delta
        second = (upper - float(start[axis])) * inv_delta
        enter = min(first, second)
        leave = max(first, second)
        t_min = max(t_min, enter)
        t_max = min(t_max, leave)
        if t_min >= t_max:
            return 0.0
    length = float(np.linalg.norm(delta))
    if length <= 1e-12:
        return 0.0
    return max(0.0, t_max - t_min) * length


def _cluster_profile_or_none(
    pos: torch.Tensor,
    node_sizes: Optional[torch.Tensor],
    clusters: Optional[Mapping[str, Sequence[int]]],
    cluster_parents: Optional[Mapping[str, Optional[str]]],
    cluster_labels: Optional[Mapping[str, str]],
    profile: Optional[ClusterGeometryProfile],
) -> Optional[ClusterGeometryProfile]:
    """Return an existing or newly derived cluster geometry profile.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    node_sizes : Optional[torch.Tensor]
        Node sizes with shape ``[N, 2]``.
    clusters : Optional[Mapping[str, Sequence[int]]]
        Cluster membership metadata.
    cluster_parents : Optional[Mapping[str, Optional[str]]]
        Nested-cluster parent metadata.
    cluster_labels : Optional[Mapping[str, str]]
        Explicit cluster labels.
    profile : Optional[ClusterGeometryProfile]
        Precomputed geometry profile.

    Returns
    -------
    Optional[ClusterGeometryProfile]
        Cluster geometry profile, or ``None`` when metadata is absent.
    """
    if profile is not None:
        return profile
    if node_sizes is None or not clusters:
        return None
    return build_cluster_geometry_profile(
        pos,
        node_sizes,
        cluster_labels,
        clusters,
        cluster_parents,
    )


def _cluster_descendant_names(profile: ClusterGeometryProfile, name: str) -> frozenset[str]:
    """Return cluster names in ``name``'s strict child subtree.

    Parameters
    ----------
    profile : ClusterGeometryProfile
        Shared cluster geometry and hierarchy profile.
    name : str
        Cluster whose descendant clusters should be returned.

    Returns
    -------
    frozenset[str]
        Strict descendant cluster names, excluding ``name`` itself.
    """
    descendants: set[str] = set()
    stack = list(profile.tree.children_per_cluster[name])
    while stack:
        child_name = stack.pop()
        descendants.add(child_name)
        stack.extend(profile.tree.children_per_cluster[child_name])
    return frozenset(descendants)


def cluster_exclusion_score(
    pos: torch.Tensor,
    node_sizes: Optional[torch.Tensor] = None,
    clusters: Optional[Mapping[str, Sequence[int]]] = None,
    cluster_parents: Optional[Mapping[str, Optional[str]]] = None,
    cluster_labels: Optional[Mapping[str, str]] = None,
    *,
    profile: Optional[ClusterGeometryProfile] = None,
) -> Dict[str, Any]:
    """Score whether foreign nodes intrude into derived cluster boxes.

    Formula: for each cluster ``K``, sum foreign node-AABB intruded area and
    divide by ``K``'s box area; the score is ``1 - mean(clamped_cluster_ratio)``.
    Descendant nodes are excluded because membership containment is true by
    construction. Node labels are not first-class in this substrate, so this
    metric intentionally scores node AABBs only.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    node_sizes : Optional[torch.Tensor], optional
        Node sizes with shape ``[N, 2]``.
    clusters : Optional[Mapping[str, Sequence[int]]], optional
        Cluster membership metadata.
    cluster_parents : Optional[Mapping[str, Optional[str]]], optional
        Nested-cluster parent metadata.
    cluster_labels : Optional[Mapping[str, str]], optional
        Explicit cluster labels.
    profile : Optional[ClusterGeometryProfile], optional
        Precomputed read-only cluster profile.

    Returns
    -------
    Dict[str, Any]
        ``cluster_exclusion_score`` and diagnostic intrusion count, or
        ``None`` when no cluster metadata applies.
    """
    cluster_profile = _cluster_profile_or_none(
        pos, node_sizes, clusters, cluster_parents, cluster_labels, profile
    )
    if cluster_profile is None:
        return {"cluster_exclusion_score": None, "cluster_exclusion_intrusions": 0}
    penalties: List[float] = []
    intrusions = 0
    for name in cluster_profile.cluster_names:
        box = cluster_profile.boxes[name]
        cluster_area = max(_bbox_area(box.bounds), 1e-12)
        intruded_area = 0.0
        for node_index, node_bounds in enumerate(cluster_profile.node_bounds):
            if node_index in box.descendants:
                continue
            overlap = _bbox_intersection_area(box.bounds, node_bounds)
            if overlap > 0.0:
                intrusions += 1
                intruded_area += overlap
        penalties.append(min(1.0, intruded_area / cluster_area))
    score = 1.0 - float(np.mean(penalties)) if penalties else 1.0
    return {"cluster_exclusion_score": max(0.0, score), "cluster_exclusion_intrusions": intrusions}


def cluster_sibling_overlap_score(
    pos: torch.Tensor,
    node_sizes: Optional[torch.Tensor] = None,
    clusters: Optional[Mapping[str, Sequence[int]]] = None,
    cluster_parents: Optional[Mapping[str, Optional[str]]] = None,
    cluster_labels: Optional[Mapping[str, str]] = None,
    *,
    profile: Optional[ClusterGeometryProfile] = None,
    pair_cap: int = _CLUSTER_PAIR_SAMPLE_CAP,
) -> Dict[str, Any]:
    """Score overlap between same-parent derived cluster boxes.

    Formula: for each same-parent pair ``(A, B)``, compute
    ``area(A ∩ B) / min(area(A), area(B))`` after a small touch tolerance; the
    score is ``1 - mean(ratio)``. If pair count exceeds ``pair_cap``, evenly
    spaced deterministic pair indices are sampled.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    node_sizes : Optional[torch.Tensor], optional
        Node sizes with shape ``[N, 2]``.
    clusters : Optional[Mapping[str, Sequence[int]]], optional
        Cluster membership metadata.
    cluster_parents : Optional[Mapping[str, Optional[str]]], optional
        Nested-cluster parent metadata.
    cluster_labels : Optional[Mapping[str, str]], optional
        Explicit cluster labels.
    profile : Optional[ClusterGeometryProfile], optional
        Precomputed read-only cluster profile.
    pair_cap : int, optional
        Maximum deterministic sibling pairs to score.

    Returns
    -------
    Dict[str, Any]
        ``cluster_sibling_overlap_score`` and sampled pair count, or ``None``
        when no cluster metadata applies.
    """
    cluster_profile = _cluster_profile_or_none(
        pos, node_sizes, clusters, cluster_parents, cluster_labels, profile
    )
    if cluster_profile is None:
        return {"cluster_sibling_overlap_score": None, "cluster_sibling_overlap_pairs": 0}
    sampled = _deterministic_sample_indices(len(cluster_profile.sibling_pairs), pair_cap)
    penalties: List[float] = []
    for pair_index in sampled:
        left_name, right_name = cluster_profile.sibling_pairs[int(pair_index)]
        left = cluster_profile.boxes[left_name].bounds
        right = cluster_profile.boxes[right_name].bounds
        overlap = _bbox_intersection_area(left, right, tolerance=1e-9)
        denom = max(1e-12, min(_bbox_area(left), _bbox_area(right)))
        penalties.append(min(1.0, overlap / denom))
    score = 1.0 - float(np.mean(penalties)) if penalties else 1.0
    return {
        "cluster_sibling_overlap_score": max(0.0, score),
        "cluster_sibling_overlap_pairs": len(sampled),
    }


def cluster_nesting_fidelity_score(
    pos: torch.Tensor,
    node_sizes: Optional[torch.Tensor] = None,
    clusters: Optional[Mapping[str, Sequence[int]]] = None,
    cluster_parents: Optional[Mapping[str, Optional[str]]] = None,
    cluster_labels: Optional[Mapping[str, str]] = None,
    *,
    profile: Optional[ClusterGeometryProfile] = None,
) -> Dict[str, Any]:
    """Score whether visual enclosure matches declared nested parents.

    Formula: for every cluster box, find the smallest-area non-descendant other
    cluster box that encloses it. A child scores 1 when that minimal enclosing
    box is its declared parent; a root scores 1 when no other box encloses it.
    The metric is the mean of those binary fidelities.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    node_sizes : Optional[torch.Tensor], optional
        Node sizes with shape ``[N, 2]``.
    clusters : Optional[Mapping[str, Sequence[int]]], optional
        Cluster membership metadata.
    cluster_parents : Optional[Mapping[str, Optional[str]]], optional
        Nested-cluster parent metadata.
    cluster_labels : Optional[Mapping[str, str]], optional
        Explicit cluster labels.
    profile : Optional[ClusterGeometryProfile], optional
        Precomputed read-only cluster profile.

    Returns
    -------
    Dict[str, Any]
        ``cluster_nesting_fidelity_score`` and violation count, or ``None``
        when no cluster metadata applies.
    """
    cluster_profile = _cluster_profile_or_none(
        pos, node_sizes, clusters, cluster_parents, cluster_labels, profile
    )
    if cluster_profile is None:
        return {"cluster_nesting_fidelity_score": None, "cluster_nesting_violations": 0}
    matches: List[float] = []
    violations = 0
    for name in cluster_profile.cluster_names:
        bounds = cluster_profile.boxes[name].bounds
        own_descendant_clusters = _cluster_descendant_names(cluster_profile, name)
        enclosing = [
            other_name
            for other_name in cluster_profile.cluster_names
            if other_name != name
            and other_name not in own_descendant_clusters
            and _bbox_contains(cluster_profile.boxes[other_name].bounds, bounds)
        ]
        expected = cluster_profile.tree.parents[name]
        minimal = None
        if enclosing:
            minimal_area = min(
                _bbox_area(cluster_profile.boxes[other_name].bounds) for other_name in enclosing
            )
            minimal_candidates = [
                other_name
                for other_name in enclosing
                if math.isclose(
                    _bbox_area(cluster_profile.boxes[other_name].bounds),
                    minimal_area,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-9,
                )
            ]
            minimal = expected if expected in minimal_candidates else minimal_candidates[0]
        match = minimal == expected
        matches.append(1.0 if match else 0.0)
        if not match:
            violations += 1
    return {
        "cluster_nesting_fidelity_score": float(np.mean(matches)) if matches else 1.0,
        "cluster_nesting_violations": violations,
    }


def cluster_edge_intrusion_score(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: Optional[torch.Tensor] = None,
    clusters: Optional[Mapping[str, Sequence[int]]] = None,
    cluster_parents: Optional[Mapping[str, Optional[str]]] = None,
    cluster_labels: Optional[Mapping[str, str]] = None,
    *,
    profile: Optional[ClusterGeometryProfile] = None,
) -> Dict[str, Any]:
    """Score foreign edge segments that pass through cluster interiors.

    Formula: for each cluster ``K``, clip every foreign edge segment against
    ``K``'s padded box, sum clipped interior length, and divide by the cluster
    diagonal. The score is ``1 - mean(clamped_cluster_ratio)``. Incident edges
    are ignored because they are allowed to enter through the boundary.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : Optional[torch.Tensor], optional
        Node sizes with shape ``[N, 2]``.
    clusters : Optional[Mapping[str, Sequence[int]]], optional
        Cluster membership metadata.
    cluster_parents : Optional[Mapping[str, Optional[str]]], optional
        Nested-cluster parent metadata.
    cluster_labels : Optional[Mapping[str, str]], optional
        Explicit cluster labels.
    profile : Optional[ClusterGeometryProfile], optional
        Precomputed read-only cluster profile.

    Returns
    -------
    Dict[str, Any]
        ``cluster_edge_intrusion_score`` and crossing count, or ``None`` when
        no cluster metadata applies.
    """
    cluster_profile = _cluster_profile_or_none(
        pos, node_sizes, clusters, cluster_parents, cluster_labels, profile
    )
    if cluster_profile is None:
        return {"cluster_edge_intrusion_score": None, "cluster_edge_intrusions": 0}
    positions = _ensure_cpu(pos).to(dtype=torch.float64).numpy()
    edges = _ensure_cpu(edge_index).to(dtype=torch.long)
    penalties: List[float] = []
    intrusions = 0
    if edges.numel() == 0:
        return {"cluster_edge_intrusion_score": 1.0, "cluster_edge_intrusions": 0}
    for name in cluster_profile.cluster_names:
        box = cluster_profile.boxes[name]
        bounds = box.bounds
        diagonal = math.hypot(bounds[2] - bounds[0], bounds[3] - bounds[1])
        if diagonal <= 1e-12:
            continue
        clipped_length = 0.0
        for source, target in edges.t().tolist():
            if int(source) in box.descendants or int(target) in box.descendants:
                continue
            length = _segment_aabb_clip_length(
                positions[int(source)],
                positions[int(target)],
                bounds,
            )
            if length > 1e-9:
                intrusions += 1
                clipped_length += length
        penalties.append(min(1.0, clipped_length / diagonal))
    score = 1.0 - float(np.mean(penalties)) if penalties else 1.0
    return {"cluster_edge_intrusion_score": max(0.0, score), "cluster_edge_intrusions": intrusions}


def cluster_label_occlusion_score(
    pos: torch.Tensor,
    node_sizes: Optional[torch.Tensor] = None,
    clusters: Optional[Mapping[str, Sequence[int]]] = None,
    cluster_parents: Optional[Mapping[str, Optional[str]]] = None,
    cluster_labels: Optional[Mapping[str, str]] = None,
    *,
    profile: Optional[ClusterGeometryProfile] = None,
) -> Dict[str, Any]:
    """Score cluster label bands against visible node and cluster-label AABBs.

    Formula: for every explicit cluster-label band ``L``, compute normalized
    overlap against node AABBs and every other explicit cluster-label band as
    ``area(L ∩ B) / min(area(L), area(B))``. The score is ``1 - mean(ratio)``.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    node_sizes : Optional[torch.Tensor], optional
        Node sizes with shape ``[N, 2]``.
    clusters : Optional[Mapping[str, Sequence[int]]], optional
        Cluster membership metadata.
    cluster_parents : Optional[Mapping[str, Optional[str]]], optional
        Nested-cluster parent metadata.
    cluster_labels : Optional[Mapping[str, str]], optional
        Explicit cluster labels.
    profile : Optional[ClusterGeometryProfile], optional
        Precomputed read-only cluster profile.

    Returns
    -------
    Dict[str, Any]
        ``cluster_label_occlusion_score`` and occlusion count, or ``None`` when
        no cluster labels apply.
    """
    cluster_profile = _cluster_profile_or_none(
        pos, node_sizes, clusters, cluster_parents, cluster_labels, profile
    )
    if cluster_profile is None:
        return {"cluster_label_occlusion_score": None, "cluster_label_occlusions": 0}
    label_boxes = [
        (name, cluster_profile.boxes[name].label_bounds)
        for name in cluster_profile.cluster_names
        if cluster_profile.boxes[name].label_bounds is not None
    ]
    if not label_boxes:
        return {"cluster_label_occlusion_score": None, "cluster_label_occlusions": 0}
    penalties: List[float] = []
    occlusions = 0
    for name, label_bounds_optional in label_boxes:
        if label_bounds_optional is None:
            continue
        label_bounds = label_bounds_optional
        label_area = max(_bbox_area(label_bounds), 1e-12)
        for node_index, node_bounds in enumerate(cluster_profile.node_bounds):
            if node_index in cluster_profile.boxes[name].descendants:
                continue
            overlap = _bbox_intersection_area(label_bounds, node_bounds)
            if overlap > 0.0:
                occlusions += 1
            denominator = max(1e-12, min(label_area, _bbox_area(node_bounds)))
            penalties.append(min(1.0, overlap / denominator))
        for other_name, other_bounds_optional in label_boxes:
            if other_name <= name or other_bounds_optional is None:
                continue
            overlap = _bbox_intersection_area(label_bounds, other_bounds_optional)
            if overlap > 0.0:
                occlusions += 1
            denom = max(1e-12, min(label_area, _bbox_area(other_bounds_optional)))
            penalties.append(min(1.0, overlap / denom))
    score = 1.0 - float(np.mean(penalties)) if penalties else 1.0
    return {
        "cluster_label_occlusion_score": max(0.0, score),
        "cluster_label_occlusions": occlusions,
    }


def cluster_compactness_score(
    pos: torch.Tensor,
    node_sizes: Optional[torch.Tensor] = None,
    clusters: Optional[Mapping[str, Sequence[int]]] = None,
    cluster_parents: Optional[Mapping[str, Optional[str]]] = None,
    cluster_labels: Optional[Mapping[str, str]] = None,
    *,
    profile: Optional[ClusterGeometryProfile] = None,
) -> Dict[str, Any]:
    """Score member spread with a saturating no-collapse reward band.

    Formula: for each leaf cluster, compute the raw member AABB area divided by
    the sum of member node areas. Ratios up to ``4.0`` receive score ``1``;
    above that band, score decays as ``4.0 / ratio``. Non-leaf clusters are
    skipped so nested padding and child separation are not scored as member
    spread, and packing tighter than the band receives no extra reward.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    node_sizes : Optional[torch.Tensor], optional
        Node sizes with shape ``[N, 2]``.
    clusters : Optional[Mapping[str, Sequence[int]]], optional
        Cluster membership metadata.
    cluster_parents : Optional[Mapping[str, Optional[str]]], optional
        Nested-cluster parent metadata.
    cluster_labels : Optional[Mapping[str, str]], optional
        Explicit cluster labels.
    profile : Optional[ClusterGeometryProfile], optional
        Precomputed read-only cluster profile.

    Returns
    -------
    Dict[str, Any]
        ``cluster_compactness_score`` and mean spread ratio, or ``None`` when
        no cluster metadata applies.
    """
    cluster_profile = _cluster_profile_or_none(
        pos, node_sizes, clusters, cluster_parents, cluster_labels, profile
    )
    if cluster_profile is None:
        return {"cluster_compactness_score": None, "cluster_compactness_mean_ratio": None}
    scores: List[float] = []
    ratios: List[float] = []
    for name in cluster_profile.cluster_names:
        if cluster_profile.tree.children_per_cluster[name]:
            continue
        box = cluster_profile.boxes[name]
        member_area = sum(
            _bbox_area(cluster_profile.node_bounds[index]) for index in box.descendants
        )
        if member_area <= 1e-12:
            continue
        ratio = _bbox_area(box.raw_leaf_bounds) / member_area
        ratios.append(ratio)
        scores.append(1.0 if ratio <= 4.0 else max(0.0, min(1.0, 4.0 / ratio)))
    return {
        "cluster_compactness_score": float(np.mean(scores)) if scores else None,
        "cluster_compactness_mean_ratio": float(np.mean(ratios)) if ratios else None,
    }


def cluster_quality_metrics(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: Optional[torch.Tensor] = None,
    clusters: Optional[Mapping[str, Sequence[int]]] = None,
    cluster_parents: Optional[Mapping[str, Optional[str]]] = None,
    cluster_labels: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Compute all six cluster-quality metrics from one shared profile.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : Optional[torch.Tensor], optional
        Node sizes with shape ``[N, 2]``.
    clusters : Optional[Mapping[str, Sequence[int]]], optional
        Cluster membership metadata.
    cluster_parents : Optional[Mapping[str, Optional[str]]], optional
        Nested-cluster parent metadata.
    cluster_labels : Optional[Mapping[str, str]], optional
        Explicit cluster labels.

    Returns
    -------
    Dict[str, Any]
        Flat metric dictionary containing the six cluster-quality scores.
    """
    profile = _cluster_profile_or_none(
        pos,
        node_sizes,
        clusters,
        cluster_parents,
        cluster_labels,
        None,
    )
    result: Dict[str, Any] = {}
    result.update(
        cluster_exclusion_score(
            pos,
            node_sizes,
            clusters,
            cluster_parents,
            cluster_labels,
            profile=profile,
        )
    )
    result.update(
        cluster_sibling_overlap_score(
            pos, node_sizes, clusters, cluster_parents, cluster_labels, profile=profile
        )
    )
    result.update(
        cluster_nesting_fidelity_score(
            pos, node_sizes, clusters, cluster_parents, cluster_labels, profile=profile
        )
    )
    result.update(
        cluster_edge_intrusion_score(
            pos, edge_index, node_sizes, clusters, cluster_parents, cluster_labels, profile=profile
        )
    )
    result.update(
        cluster_label_occlusion_score(
            pos,
            node_sizes,
            clusters,
            cluster_parents,
            cluster_labels,
            profile=profile,
        )
    )
    result.update(
        cluster_compactness_score(
            pos,
            node_sizes,
            clusters,
            cluster_parents,
            cluster_labels,
            profile=profile,
        )
    )
    return result


def directed_flow_score(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    direction: str = "TB",
    back_edge_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Score signed directed flow along a declared layout axis.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Directed edge tensor with shape ``[2, E]``.
    direction : str, optional
        Declared flow direction: ``TB``, ``BT``, ``LR``, or ``RL``.
    back_edge_mask : Optional[torch.Tensor], optional
        Pre-declared feedback edges excluded from the score.

    Returns
    -------
    Dict[str, float]
        Signed tolerance-band directed-flow score.
    """
    positions = _ensure_cpu(pos).to(dtype=torch.float64)
    edges = _ensure_cpu(edge_index).to(dtype=torch.long)
    if back_edge_mask is not None:
        edges = edges[:, ~_ensure_cpu(back_edge_mask).to(dtype=torch.bool)]
    if edges.numel() == 0:
        return {"directed_flow_score": 0.0}
    vectors = positions[edges[1]] - positions[edges[0]]
    norms = torch.linalg.vector_norm(vectors, dim=1)
    axis = {
        "TB": torch.tensor([0.0, 1.0], dtype=torch.float64),
        "BT": torch.tensor([0.0, -1.0], dtype=torch.float64),
        "LR": torch.tensor([1.0, 0.0], dtype=torch.float64),
        "RL": torch.tensor([-1.0, 0.0], dtype=torch.float64),
    }.get(direction, torch.tensor([0.0, 1.0], dtype=torch.float64))
    projection = torch.where(
        norms > 1e-12, (vectors @ axis) / norms.clamp(min=1e-12), -torch.ones_like(norms)
    )
    tolerance = math.sin(math.radians(15.0))
    score = torch.clamp((projection + tolerance) / (2.0 * tolerance), 0.0, 1.0).mean()
    return {"directed_flow_score": float(score.item())}


# ---------------------------------------------------------------------------
# Tier 3: DAG/hierarchy-specific
# ---------------------------------------------------------------------------


def cluster_separation(pos: torch.Tensor, cluster_ids: torch.Tensor) -> Dict[str, float]:
    """Ratio of inter-cluster distance to intra-cluster spread.

    Complexity: O(N + C²) where C = number of clusters.
    Target: mean ratio > 3.0, < 5% overlapping.
    """
    unique_clusters = cluster_ids.unique()
    n_clusters = unique_clusters.numel()

    if n_clusters <= 1:
        return {
            "cluster_mean_sep_ratio": 0.0,
            "cluster_min_sep_ratio": 0.0,
            "cluster_frac_overlapping": 0.0,
        }

    centroids = {}
    spreads = {}
    for c in unique_clusters:
        mask = cluster_ids == c
        cp = pos[mask]
        centroids[c.item()] = cp.mean(dim=0)
        spreads[c.item()] = cp.std(dim=0).norm().item() if cp.shape[0] > 1 else 1.0

    # Sample cluster pairs if too many
    cluster_list = list(centroids.keys())
    if n_clusters > 200:
        # A local generator keeps referee sampling deterministic without
        # perturbing NumPy's process-global RNG state.
        n_pairs = min(20000, n_clusters * (n_clusters - 1) // 2)
        rng = np.random.default_rng(42)
        i_idx = rng.integers(0, n_clusters, n_pairs)
        j_idx = rng.integers(0, n_clusters, n_pairs)
        pairs = [(cluster_list[i], cluster_list[j]) for i, j in zip(i_idx, j_idx) if i < j]
    else:
        pairs = [
            (cluster_list[i], cluster_list[j])
            for i in range(n_clusters)
            for j in range(i + 1, n_clusters)
        ]

    separations = []
    for ci, cj in pairs:
        dist = torch.norm(centroids[ci] - centroids[cj]).item()
        avg_spread = (spreads[ci] + spreads[cj]) / 2
        if avg_spread > 1e-6:
            separations.append(dist / avg_spread)

    if not separations:
        return {
            "cluster_mean_sep_ratio": 0.0,
            "cluster_min_sep_ratio": 0.0,
            "cluster_frac_overlapping": 0.0,
        }

    sep = torch.tensor(separations)
    return {
        "cluster_mean_sep_ratio": sep.mean().item(),
        "cluster_min_sep_ratio": sep.min().item(),
        "cluster_frac_overlapping": (sep < 2.0).float().mean().item(),
    }


def layer_uniformity(pos: torch.Tensor, topo_depth) -> Dict[str, float]:
    """Uniformity of vertical spacing between adjacent topological layers.

    Complexity: O(N).
    Target: layer_spacing_cv < 0.3.
    """
    if isinstance(topo_depth, torch.Tensor):
        depth_t = topo_depth
    else:
        depth_t = torch.tensor(topo_depth, dtype=torch.long)

    unique_depths = depth_t.unique().sort().values
    if unique_depths.numel() < 2:
        return {
            "layer_spacing_cv": 0.0,
            "layer_spacing_mean": 0.0,
            "layer_spacing_min": 0.0,
            "layer_spacing_max": 0.0,
            "n_layers": 1,
        }

    medians = []
    for d in unique_depths:
        mask = depth_t == d
        medians.append(pos[mask, 1].median().item())

    medians_t = torch.tensor(medians)
    spacings = (medians_t[1:] - medians_t[:-1]).abs()

    if spacings.numel() < 2 or spacings.mean() < 1e-8:
        return {
            "layer_spacing_cv": 0.0,
            "layer_spacing_mean": spacings.mean().item(),
            "layer_spacing_min": spacings.min().item(),
            "layer_spacing_max": spacings.max().item(),
            "n_layers": int(unique_depths.numel()),
        }

    return {
        "layer_spacing_cv": (spacings.std() / spacings.mean()).item(),
        "layer_spacing_mean": spacings.mean().item(),
        "layer_spacing_min": spacings.min().item(),
        "layer_spacing_max": spacings.max().item(),
        "n_layers": int(unique_depths.numel()),
    }


def edge_node_crossing_count(
    curves,
    positions: torch.Tensor,
    node_sizes: torch.Tensor,
    edge_index: torch.Tensor,
    T: int = 10,
) -> Dict[str, Any]:
    """Count edge-node bbox crossings by sampling T points per bezier.

    Tier 2 metric. Returns count and rate.
    """
    from dagua.edges import evaluate_bezier

    if not curves or positions.shape[0] == 0:
        return {"edge_node_crossings": 0, "edge_node_crossing_rate": 0.0}

    pos = _ensure_cpu(positions)
    sizes = _ensure_cpu(node_sizes)
    ei = _ensure_cpu(edge_index)
    N = pos.shape[0]
    E = len(curves)

    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    src_list = ei[0].tolist() if ei.numel() > 0 else []
    tgt_list = ei[1].tolist() if ei.numel() > 0 else []

    crossings = 0
    for e_idx, curve in enumerate(curves):
        src_node = src_list[e_idx] if e_idx < len(src_list) else -1
        tgt_node = tgt_list[e_idx] if e_idx < len(tgt_list) else -1

        for t_step in range(1, T):
            t = t_step / T
            px, py = evaluate_bezier(curve, t)
            for n_idx in range(N):
                if n_idx == src_node or n_idx == tgt_node:
                    continue
                nx, ny = pos[n_idx, 0].item(), pos[n_idx, 1].item()
                hw, hh = half_w[n_idx].item(), half_h[n_idx].item()
                if abs(px - nx) < hw and abs(py - ny) < hh:
                    crossings += 1
                    break  # count at most one crossing per sample point

    rate = crossings / max(E * (T - 1), 1)
    return {"edge_node_crossings": crossings, "edge_node_crossing_rate": rate}


def label_overlap_count(
    label_positions: List[Optional[Tuple[float, float]]],
    edge_labels: list,
    positions: torch.Tensor,
    node_sizes: torch.Tensor,
    label_font_size: float = 7.0,
) -> Dict[str, int]:
    """Count overlaps between edge labels and node bboxes, and between labels.

    Tier 1 metric. Uses bbox approximation for label sizes.
    """
    from dagua.utils import measure_text_fallback

    pos = _ensure_cpu(positions)
    sizes = _ensure_cpu(node_sizes)
    N = pos.shape[0]

    # Build label bboxes
    label_bboxes = []
    for i, lp in enumerate(label_positions):
        if lp is None or i >= len(edge_labels) or not edge_labels[i]:
            continue
        lw, lh = measure_text_fallback(edge_labels[i], label_font_size)
        lw += 4.0
        lh += 2.0
        label_bboxes.append((lp[0] - lw / 2, lp[1] - lh / 2, lp[0] + lw / 2, lp[1] + lh / 2))

    # Node bboxes
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2

    # Label-node overlaps
    label_node_overlaps = 0
    for lb in label_bboxes:
        for n_idx in range(N):
            nx, ny = pos[n_idx, 0].item(), pos[n_idx, 1].item()
            hw, hh = half_w[n_idx].item(), half_h[n_idx].item()
            nb = (nx - hw, ny - hh, nx + hw, ny + hh)
            if lb[0] < nb[2] and lb[2] > nb[0] and lb[1] < nb[3] and lb[3] > nb[1]:
                label_node_overlaps += 1

    # Label-label overlaps
    label_overlaps = 0
    for i in range(len(label_bboxes)):
        for j in range(i + 1, len(label_bboxes)):
            a, b = label_bboxes[i], label_bboxes[j]
            if a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]:
                label_overlaps += 1

    return {"label_overlaps": label_overlaps, "label_node_overlaps": label_node_overlaps}


def edge_curvature_consistency(curves) -> Dict[str, float]:
    """Compute curvature statistics across all edges.

    κ at 5 sample points per edge.
    Tier 1 metric.
    """
    from dagua.edges import bezier_tangent

    if not curves:
        return {"edge_curvature_cv": 0.0, "edge_curvature_mean": 0.0}

    mean_curvatures = []
    for curve in curves:
        kappas = []
        for t_step in range(5):
            t = (t_step + 0.5) / 5
            # First derivative
            dx1, dy1 = bezier_tangent(curve, t)
            # Second derivative (finite difference approximation)
            dt = 0.01
            dx2a, dy2a = bezier_tangent(curve, min(t + dt, 1.0))
            dx2b, dy2b = bezier_tangent(curve, max(t - dt, 0.0))
            ddx = (dx2a - dx2b) / (2 * dt)
            ddy = (dy2a - dy2b) / (2 * dt)

            cross = abs(dx1 * ddy - dy1 * ddx)
            norm = (dx1**2 + dy1**2) ** 1.5
            if norm > 1e-8:
                kappas.append(cross / norm)

        if kappas:
            mean_curvatures.append(sum(kappas) / len(kappas))

    if not mean_curvatures:
        return {"edge_curvature_cv": 0.0, "edge_curvature_mean": 0.0}

    mean_k = sum(mean_curvatures) / len(mean_curvatures)
    if len(mean_curvatures) > 1 and mean_k > 1e-8:
        var_k = sum((k - mean_k) ** 2 for k in mean_curvatures) / (len(mean_curvatures) - 1)
        cv = var_k**0.5 / mean_k
    else:
        cv = 0.0

    return {"edge_curvature_cv": cv, "edge_curvature_mean": mean_k}


def port_angular_resolution(
    curves: Sequence[Any],
    edge_index: torch.Tensor,
) -> Dict[str, Any]:
    """Minimum angle between incident edge tangents at actual ports.

    The score uses the same degree-relative normalization as the common
    ruler's angular-resolution term.
    """
    from dagua.edges import bezier_tangent

    if not curves or edge_index.numel() == 0:
        return {"port_angular_res_mean_deg": 360.0, "port_angular_resolution_score": None}

    ei = _ensure_cpu(edge_index)
    src_list = ei[0].tolist()
    tgt_list = ei[1].tolist()

    # Collect tangent vectors per node
    node_tangents: Dict[int, List[Tuple[float, float]]] = {}

    for e_idx, curve in enumerate(curves):
        if e_idx >= len(src_list):
            break
        s, t = src_list[e_idx], tgt_list[e_idx]

        # Source tangent: B'(0)
        sdx, sdy = bezier_tangent(curve, 0.0)
        node_tangents.setdefault(s, []).append((sdx, sdy))

        # Target tangent: B'(1)
        tdx, tdy = bezier_tangent(curve, 1.0)
        node_tangents.setdefault(t, []).append((-tdx, -tdy))

    min_angles = []
    scores = []
    for tangents in node_tangents.values():
        if len(tangents) < 2:
            continue

        if any(math.hypot(dx, dy) <= 1e-12 for dx, dy in tangents):
            min_angles.append(0.0)
            scores.append(0.0)
            continue

        # Compute angles
        angles = []
        for dx, dy in tangents:
            angles.append(math.atan2(dy, dx))
        angles.sort()

        # Min angle between adjacent
        diffs = [angles[i + 1] - angles[i] for i in range(len(angles) - 1)]
        diffs.append(2 * math.pi - (angles[-1] - angles[0]))
        minimum = min(diffs)
        ideal = 2.0 * math.pi / len(tangents)
        min_angles.append(minimum)
        scores.append(1.0 - (ideal - min(minimum, ideal)) / ideal)

    if not min_angles:
        return {"port_angular_res_mean_deg": 360.0, "port_angular_resolution_score": None}

    mean_deg = sum(a * 180 / math.pi for a in min_angles) / len(min_angles)
    return {
        "port_angular_res_mean_deg": mean_deg,
        "port_angular_resolution_score": float(np.mean(scores)),
    }


def within_layer_compactness(pos: torch.Tensor, topo_depth) -> Dict[str, float]:
    """Fraction of each layer's x-range that is occupied by nodes.

    Complexity: O(N).
    """
    if isinstance(topo_depth, torch.Tensor):
        depth_t = topo_depth
    else:
        depth_t = torch.tensor(topo_depth, dtype=torch.long)

    unique_depths = depth_t.unique()
    scores = []

    for d in unique_depths:
        layer_mask = depth_t == d
        layer_nodes = torch.where(layer_mask)[0]
        if layer_nodes.numel() < 2:
            continue

        layer_x = pos[layer_nodes, 0]
        total_width = (layer_x.max() - layer_x.min()).item()
        if total_width < 1e-6:
            scores.append(1.0)
            continue

        n_bins = min(100, layer_nodes.numel())
        hist = torch.histc(layer_x, bins=n_bins)
        occupied = (hist > 0).float().sum() / n_bins
        scores.append(occupied.item())

    if not scores:
        return {"layer_compactness_mean": 1.0, "layer_compactness_min": 1.0}

    scores_t = torch.tensor(scores)
    return {
        "layer_compactness_mean": scores_t.mean().item(),
        "layer_compactness_min": scores_t.min().item(),
    }


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------


# r80-P6 degeneracy guard threshold: below this ratio of mean edge length to
# mean node bounding-box diagonal, a layout is considered "point-collapsed"
# rather than merely compact. See `_is_degenerate_scale` for the exploit this
# closes (S1 HIGH-3: a fully collapsed layout trivially aces edge-length
# uniformity and crossing-rate, scoring HIGHER than a normal random layout).
DEGENERATE_SCALE_RATIO = 0.25


def _is_degenerate_scale(metrics: Dict[str, float]) -> bool:
    """Detect a point-collapsed layout from its metric summary.

    When nodes have collapsed onto (near) the same point, every edge length
    trivially converges to the same tiny value. That vacuously maximizes
    ``edge_length_cv`` (coefficient of variation of an all-equal
    distribution is 0, i.e. full length-uniformity credit) and
    ``crossing_rate`` (zero-length segments never register as crossing
    under the segment-intersection test), even though the layout is a
    fully overlapping unreadable mess. Only the binary overlap term
    correctly zeroes out on its own; this guard closes the other two.

    Degenerate scale is defined as: mean edge length < 0.25 * mean node
    bounding-box diagonal. Both quantities must be present in ``metrics``
    (``edge_length_mean`` from ``edge_length_cv()``, ``node_diag_mean`` from
    ``quick()``/``full()`` when node sizes were supplied) for the guard to
    fire; metrics computed without node sizes, or predating this field,
    conservatively report "not degenerate" (unchanged prior behavior).

    Parameters
    ----------
    metrics : Dict[str, float]
        Metric name to scalar value mapping.

    Returns
    -------
    bool
        ``True`` when the layout is at a degenerate (point-collapsed) scale.
    """
    edge_length_mean = metrics.get("edge_length_mean")
    node_diag_mean = metrics.get("node_diag_mean")
    if edge_length_mean is None or node_diag_mean is None:
        return False
    if float(node_diag_mean) <= 1e-8:
        return False
    return float(edge_length_mean) < DEGENERATE_SCALE_RATIO * float(node_diag_mean)


def _has_declared_hierarchy(metrics: Dict[str, Any]) -> bool:
    """Report whether metric metadata explicitly declares a hierarchy.

    Parameters
    ----------
    metrics : Dict[str, Any]
        Metric values and semantic metadata.

    Returns
    -------
    bool
        True only for explicit DAG or pre-layout rank metadata.
    """
    return bool(
        metrics.get("declared_hierarchical", False)
        or metrics.get("has_dag_metadata", False)
        or metrics.get("has_rank_metadata", False)
    )


def _renormalized_score(metrics: Dict[str, Any], weights: Dict[str, float]) -> float:
    """Combine applicable score terms and renormalize their frozen weights.

    Parameters
    ----------
    metrics : Dict[str, Any]
        Metric values. Missing and ``None`` terms are inapplicable.
    weights : Dict[str, float]
        Frozen term weights.

    Returns
    -------
    float
        Renormalized score on a 0--100 scale.
    """
    applicable = [
        (weight, float(metrics[name]))
        for name, weight in weights.items()
        if name in metrics and metrics[name] is not None and math.isfinite(float(metrics[name]))
    ]
    total_weight = sum(weight for weight, _ in applicable)
    if total_weight <= 0.0:
        return 0.0
    weighted = sum(weight * max(0.0, min(1.0, value)) for weight, value in applicable)
    return 100.0 * weighted / total_weight


def _has_cluster_quality(metrics: Dict[str, Any]) -> bool:
    """Return whether any conditional cluster-quality term is applicable.

    Parameters
    ----------
    metrics : Dict[str, Any]
        Metric dictionary that may contain cluster-quality scores.

    Returns
    -------
    bool
        ``True`` when at least one new cluster-quality score is present and
        finite.
    """
    for name in _CLUSTER_WEIGHTS:
        value = metrics.get(name)
        if value is not None and math.isfinite(float(value)):
            return True
    return False


def composite_undirected(metrics: Dict[str, Any]) -> float:
    """Compute the frozen ten-term common ruler with applicability renormalization.

    Parameters
    ----------
    metrics : Dict[str, Any]
        Honest-ruler metric scores and applicability markers.

    Returns
    -------
    float
        Common composite score on a 0--100 scale.
    """
    scoring_metrics = dict(metrics)
    if _is_degenerate_scale(metrics):
        # Geometry-free maxima (no crossings, no Gabriel intrusions, no
        # crossing angles) are vacuous on a point collapse. Cluster-quality
        # clean scores are likewise free when the whole drawing has collapsed.
        for name in (
            "ksm_score",
            "edge_crossing_score",
            "neighborhood_preservation_score",
            "edge_length_deviation_score",
            "gabriel_score",
            "crossing_angle_score",
            "angular_resolution_score",
            "path_continuity_score",
            "cluster_silhouette_score",
            "cluster_exclusion_score",
            "cluster_sibling_overlap_score",
            "cluster_nesting_fidelity_score",
            "cluster_edge_intrusion_score",
            "cluster_label_occlusion_score",
            "cluster_compactness_score",
        ):
            if name in scoring_metrics and scoring_metrics[name] is not None:
                scoring_metrics[name] = 0.0
    if not _has_cluster_quality(scoring_metrics):
        return _renormalized_score(scoring_metrics, _COMMON_WEIGHTS)
    combined_weights = {**_COMMON_WEIGHTS, **_CLUSTER_WEIGHTS}
    return _renormalized_score(scoring_metrics, combined_weights)


def composite(metrics: Dict[str, Any]) -> float:
    """Compute the hierarchy-gated directed ruler.

    Parameters
    ----------
    metrics : Dict[str, Any]
        Honest-ruler metric scores plus explicit hierarchy metadata.

    Returns
    -------
    float
        Directed score for declared hierarchies, otherwise the common score.
    """
    if not _has_declared_hierarchy(metrics):
        return composite_undirected(metrics)
    common_applicable = {
        name: value
        for name, value in metrics.items()
        if name in _COMMON_WEIGHTS and value is not None
    }
    if _is_degenerate_scale(metrics):
        common_applicable = {name: 0.0 for name in common_applicable}
    directed_applicable = {
        name: value
        for name, value in metrics.items()
        if name in _DIRECTED_WEIGHTS and value is not None
    }
    cluster_applicable = {
        name: value
        for name, value in metrics.items()
        if name in _CLUSTER_WEIGHTS and value is not None
    }
    if _is_degenerate_scale(metrics):
        cluster_applicable = {name: 0.0 for name in cluster_applicable}
    # Directed rows retain the pre-existing 0.75 common-term scaling. The
    # conditional cluster-quality group enters at full weight when applicable.
    combined_weights = {
        **{name: 0.75 * weight for name, weight in _COMMON_WEIGHTS.items()},
        **_DIRECTED_WEIGHTS,
    }
    if cluster_applicable:
        combined_weights.update(_CLUSTER_WEIGHTS)
    return _renormalized_score(
        {**common_applicable, **directed_applicable, **cluster_applicable},
        combined_weights,
    )


def composite_auto(
    metrics: Dict[str, Any], is_semantically_directed: Optional[bool] = None
) -> float:
    """Route only declared hierarchical digraphs to the directed ruler.

    Parameters
    ----------
    metrics : Dict[str, Any]
        Honest-ruler metrics and semantic metadata.
    is_semantically_directed : Optional[bool], optional
        Whether edge direction has domain meaning.

    Returns
    -------
    float
        Directed score only when both semantic direction and hierarchy are declared.
    """
    if is_semantically_directed is True and _has_declared_hierarchy(metrics):
        return composite(metrics)
    return composite_undirected(metrics)


def composite_large(metrics: Dict[str, Any]) -> float:
    """Compatibility wrapper using the same hierarchy-gated ruler at scale.

    Parameters
    ----------
    metrics : Dict[str, Any]
        Sampled honest-ruler metric scores.

    Returns
    -------
    float
        Shared composite score with unavailable terms renormalized away.
    """
    return composite(metrics)


def composite_large_undirected(metrics: Dict[str, Any]) -> float:
    """Compatibility wrapper using the common ruler on sampled metrics.

    Parameters
    ----------
    metrics : Dict[str, Any]
        Sampled honest-ruler metric scores.

    Returns
    -------
    float
        Shared common composite score.
    """
    return composite_undirected(metrics)


def composite_large_auto(
    metrics: Dict[str, Any], is_semantically_directed: Optional[bool] = None
) -> float:
    """Compatibility wrapper for semantic routing at large scale.

    Parameters
    ----------
    metrics : Dict[str, Any]
        Sampled honest-ruler metric scores and metadata.
    is_semantically_directed : Optional[bool], optional
        Whether edge direction has domain meaning.

    Returns
    -------
    float
        Shared automatically routed score.
    """
    return composite_auto(metrics, is_semantically_directed)


def composite_strict(metrics: Dict[str, Any]) -> float:
    """Compute the new ruler while rejecting absent required profile fields.

    Parameters
    ----------
    metrics : Dict[str, Any]
        Complete honest-ruler metric profile. Applicable-empty terms must be
        represented explicitly as ``None``.

    Returns
    -------
    float
        Strict directed or common score.

    Raises
    ------
    ValueError
        If any common term key, or a required directed term, is absent.
    """
    required = set(_COMMON_WEIGHTS)
    if _has_declared_hierarchy(metrics):
        required.update(_DIRECTED_WEIGHTS)
    missing = sorted(required.difference(metrics))
    if missing:
        raise ValueError(f"composite_strict: missing required fields: {missing}")
    return composite(metrics)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def quick(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    topo_depth=None,
    node_sizes: Optional[torch.Tensor] = None,
    direction: str = "TB",
    back_edge_mask: Optional[torch.Tensor] = None,
    *,
    seed: Optional[int] = None,
    declared_hierarchical: bool = False,
    all_pairs_dist: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute the Tier-1 metric bundle.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    topo_depth : Any, optional
        Topological depth metadata. When ``None``, it is derived from the
        graph if edges are present.
    node_sizes : Optional[torch.Tensor], optional
        Node size tensor with shape ``[N, 2]`` used for overlap counting.
    direction : str, optional
        Layout direction for DAG consistency and straightness metrics.
    back_edge_mask : Optional[torch.Tensor], optional
        Boolean mask with shape ``[E]`` identifying back edges that should be
        excluded from DAG consistency.
    seed : Optional[int], optional
        Seed forwarded to stochastic Tier-1 metrics. ``None`` preserves the
        existing stochastic behavior.
    declared_hierarchical : bool, optional
        Explicit declaration that DAG or pre-layout rank semantics apply.
    all_pairs_dist : Optional[numpy.ndarray], optional
        Precomputed unweighted shortest paths with shape ``[N, N]`` for
        large-graph quick metrics that need graph-distance rows.

    Returns
    -------
    Dict[str, Any]
        Flat metric dictionary for the quick metric surface.
    """
    t0 = _time.perf_counter()
    pos = _ensure_cpu(pos)
    ei = _ensure_cpu(edge_index)
    N = pos.shape[0]
    E = ei.shape[1] if ei.numel() > 0 else 0

    bem = None
    if back_edge_mask is not None:
        bem = back_edge_mask.cpu() if back_edge_mask.device.type != "cpu" else back_edge_mask

    result: Dict[str, Any] = {
        "_graph_n_nodes": N,
        "_graph_n_edges": E,
        "declared_hierarchical": declared_hierarchical,
    }

    # Edge length CV
    result.update(edge_length_cv(pos, ei))
    result.update(edge_length_deviation_score(pos, ei))

    # DAG consistency
    result.update(dag_consistency(pos, ei, direction=direction, back_edge_mask=bem))
    result.update(directed_flow_score(pos, ei, direction=direction, back_edge_mask=bem))

    # Depth-position correlation
    if topo_depth is None and ei.numel() > 0:
        from dagua.utils import longest_path_layering

        topo_depth = longest_path_layering(ei, N)

    if topo_depth is not None:
        if not isinstance(topo_depth, torch.Tensor):
            topo_depth = torch.tensor(topo_depth, dtype=torch.long)
        result.update(depth_position_correlation(pos, topo_depth, direction=direction))
        depth_rho = result["depth_spearman_rho"]
        result["depth_order_score"] = (
            None if depth_rho is None else max(0.0, min(1.0, float(depth_rho)))
        )

    # Overlap count
    if node_sizes is not None:
        ns = _ensure_cpu(node_sizes)
        result.update(node_occlusion_score(pos, ns, seed=seed))
        # Mean node bounding-box diagonal (r80-P6): used by the composite
        # degeneracy guard to detect point-collapsed layouts, where edge
        # length has collapsed to near-zero relative to node size. Depends
        # only on node_sizes (label geometry), never on pos, so it is
        # reproducible without re-running any layout.
        diag = torch.sqrt(ns[:, 0] ** 2 + ns[:, 1] ** 2)
        result["node_diag_mean"] = diag.mean().item() if diag.numel() > 0 else 0.0
    else:
        result["node_occlusion_score"] = None

    # Aspect ratio
    ns_arg = _ensure_cpu(node_sizes) if node_sizes is not None else None
    result.update(aspect_ratio(pos, ns_arg))

    # Edge straightness
    result.update(edge_direction_straightness(pos, ei, direction=direction))

    if N > 2_000:
        # Large graphs keep the frozen ten-term identity; only deterministic
        # sample budgets shrink with scale. Terms without applicable metadata
        # remain ``None`` and are renormalized by the shared table.
        source_budget = max(16, min(64, math.ceil(100_000 / N)))
        target_budget = max(64, min(256, math.ceil(500_000 / N)))
        neighborhood_budget = max(128, min(512, math.ceil(1_000_000 / N)))
        pair_budget = max(50_000, min(250_000, 50 * E))
        if N <= 100_000:
            result.update(
                isotonic_stress(
                    pos,
                    ei,
                    N,
                    n_sources=source_budget,
                    n_targets=target_budget,
                    all_pairs_dist=all_pairs_dist,
                )
            )
            result.update(
                neighborhood_preservation(
                    pos,
                    ei,
                    N,
                    n_samples=neighborhood_budget,
                    all_pairs_dist=all_pairs_dist,
                )
            )
            result.update(angular_resolution_score(pos, ei, n_samples=512))
            result.update(path_continuity_score(pos, ei, n_samples=512))
        else:
            # Dense row-to-all-node distance blocks and Python adjacency sets
            # are genuinely infeasible at rare-suite scale, so applicability
            # renormalization drops these terms explicitly.
            result["ksm_score"] = None
            result["neighborhood_preservation_score"] = None
            result["angular_resolution_score"] = None
            result["path_continuity_score"] = None
        result.update(edge_crossing_score(pos, ei, n_samples=pair_budget, seed=0))
        result.update(gabriel_score(pos, ei, n_samples=min(E, 256), node_samples=min(N, 512)))
        result.update(crossing_angle_score(pos, ei, n_samples=pair_budget, seed=0))
        result["cluster_silhouette_score"] = None
        result["_sample_sizes"] = {
            "stress_sources": source_budget,
            "stress_targets": target_budget,
            "crossing_samples": pair_budget,
            "neighborhood_samples": neighborhood_budget,
            "gabriel_edges": min(E, 256),
        }

    result["_compute_time_seconds"] = _time.perf_counter() - t0
    return result


def full(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    topo_depth: Optional[Any] = None,
    node_sizes: Optional[torch.Tensor] = None,
    cluster_ids: Optional[torch.Tensor] = None,
    direction: str = "TB",
    stress_sources: int = 200,
    stress_targets: int = 1000,
    crossing_samples: int = 1_000_000,
    neighborhood_samples: int = 5000,
    curves: Optional[Any] = None,
    label_positions: Optional[Any] = None,
    edge_labels: Optional[Any] = None,
    back_edge_mask: Optional[torch.Tensor] = None,
    declared_hierarchical: bool = False,
    edge_length_targets: Optional[torch.Tensor] = None,
    all_pairs_dist: Optional[np.ndarray] = None,
    clusters: Optional[Mapping[str, Sequence[int]]] = None,
    cluster_parents: Optional[Mapping[str, Optional[str]]] = None,
    cluster_labels: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """All metrics including sampled Tier-2 and DAG-specific Tier-3.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    topo_depth : Optional[Any], optional
        Topological depths with shape ``[N]``. Computed when ``None``.
    node_sizes : Optional[torch.Tensor], optional
        Node sizes with shape ``[N, 2]`` for overlap and cluster metrics.
    cluster_ids : Optional[torch.Tensor], optional
        Cluster assignments with shape ``[N]``.
    direction : str, optional
        Layout direction.
    stress_sources : int, optional
        Number of sampled graph-distance sources for stress.
    stress_targets : int, optional
        Number of sampled reachable targets per stress source.
    crossing_samples : int, optional
        Number of edge pairs to sample for crossings.
    neighborhood_samples : int, optional
        Number of nodes to sample for neighborhood preservation.
    curves : Optional[Any], optional
        Optional BezierCurve list for edge-aware metrics.
    label_positions : Optional[Any], optional
        Optional precomputed label positions.
    edge_labels : Optional[Any], optional
        Optional edge label list for label-overlap metrics.
    back_edge_mask : Optional[torch.Tensor], optional
        Boolean mask with shape ``[E]`` for back edges excluded from
        ``dag_consistency``.
    declared_hierarchical : bool, optional
        Explicit declaration that DAG or pre-layout rank semantics apply.
    edge_length_targets : Optional[torch.Tensor], optional
        Declared target edge lengths with shape ``[E]``.
    all_pairs_dist : Optional[numpy.ndarray], optional
        Precomputed unweighted shortest paths with shape ``[N, N]``.
    clusters : Optional[Mapping[str, Sequence[int]]], optional
        Cluster membership keyed by cluster name for cluster-quality metrics.
    cluster_parents : Optional[Mapping[str, Optional[str]]], optional
        Nested-cluster parent mapping for cluster-quality metrics.
    cluster_labels : Optional[Mapping[str, str]], optional
        Explicit cluster labels for label-band occlusion scoring.

    Returns
    -------
    Dict[str, Any]
        Flat dict of all metrics.
    """
    t0 = _time.perf_counter()
    pos = _ensure_cpu(pos)
    ei = _ensure_cpu(edge_index)
    N = pos.shape[0]
    # Compute topo_depth if needed
    if topo_depth is None and ei.numel() > 0:
        from dagua.utils import longest_path_layering

        topo_depth = longest_path_layering(ei, N)

    if topo_depth is not None and not isinstance(topo_depth, torch.Tensor):
        topo_depth = torch.tensor(topo_depth, dtype=torch.long)

    # Tier 1
    result = quick(
        pos,
        ei,
        topo_depth=topo_depth,
        node_sizes=node_sizes,
        direction=direction,
        back_edge_mask=back_edge_mask,
        declared_hierarchical=declared_hierarchical,
        all_pairs_dist=all_pairs_dist,
    )

    if edge_length_targets is not None:
        result.update(edge_length_deviation_score(pos, ei, edge_length_targets))

    # The ruler samples graph-distance strata without a distance cap and freezes
    # both source rows and edge-pair RNG state.
    result.update(
        isotonic_stress(
            pos,
            ei,
            N,
            n_sources=stress_sources,
            n_targets=stress_targets,
            all_pairs_dist=all_pairs_dist,
        )
    )
    crossing_geometry = _crossing_pair_geometry(pos, ei, n_samples=crossing_samples, seed=0)
    result.update(
        edge_crossing_score(
            pos,
            ei,
            n_samples=crossing_samples,
            seed=0,
            _geometry=crossing_geometry,
        )
    )
    result.update(
        neighborhood_preservation(
            pos,
            ei,
            N,
            n_samples=neighborhood_samples,
            all_pairs_dist=all_pairs_dist,
        )
    )
    result.update(angular_resolution(pos, ei))
    result.update(angular_resolution_score(pos, ei))
    result.update(gabriel_score(pos, ei))
    result.update(
        crossing_angle_score(
            pos,
            ei,
            n_samples=crossing_samples,
            seed=0,
            _geometry=crossing_geometry,
        )
    )
    result.update(path_continuity_score(pos, ei))

    # Edge-aware metrics (when curves are provided)
    if curves is not None:
        result.update(edge_curvature_consistency(curves))
        result.update(port_angular_resolution(curves, ei))
        if node_sizes is not None:
            ns = _ensure_cpu(node_sizes)
            result.update(edge_node_crossing_count(curves, pos, ns, ei))
        if label_positions is not None and edge_labels is not None and node_sizes is not None:
            ns = _ensure_cpu(node_sizes)
            result.update(label_overlap_count(label_positions, edge_labels, pos, ns))

    # Tier 3: DAG-specific
    if cluster_ids is not None:
        cluster_ids = _ensure_cpu(cluster_ids)
        result.update(cluster_separation(pos, cluster_ids))
        result.update(cluster_silhouette_score(pos, cluster_ids))
    else:
        result["cluster_silhouette_score"] = None
        result["cluster_silhouette"] = None
    result.update(
        cluster_quality_metrics(pos, ei, node_sizes, clusters, cluster_parents, cluster_labels)
    )

    if topo_depth is not None:
        result.update(layer_uniformity(pos, topo_depth))
        result.update(within_layer_compactness(pos, topo_depth))

    # Composite score
    result["composite_score"] = composite(result)

    result["_compute_time_seconds"] = _time.perf_counter() - t0
    result["_sample_sizes"] = {
        "stress_sources": stress_sources,
        "stress_targets": stress_targets,
        "crossing_samples": crossing_samples,
        "neighborhood_samples": neighborhood_samples,
    }

    return result


def compare(pos_dagua: torch.Tensor, pos_reference: torch.Tensor) -> Dict[str, float]:
    """Compare two layouts using Procrustes disparity.

    Parameters
    ----------
    pos_dagua : torch.Tensor
        First position tensor with shape ``[N, 2]``.
    pos_reference : torch.Tensor
        Second position tensor with shape ``[N, 2]``.

    Returns
    -------
    Dict[str, float]
        Dictionary containing ``procrustes_disparity``.
    """
    from scipy.spatial import procrustes

    mtx1 = pos_reference.cpu().numpy().astype(np.float64)
    mtx2 = pos_dagua.cpu().numpy().astype(np.float64)

    try:
        _, _, disparity = procrustes(mtx1, mtx2)
        return {"procrustes_disparity": float(disparity)}
    except Exception:
        return {"procrustes_disparity": 1.0}


def layout_similarity(
    pos_a: torch.Tensor,
    pos_b: torch.Tensor,
    edge_index: Optional[torch.Tensor] = None,
    k: int = 5,
) -> Dict[str, float]:
    """Compare two layouts of the same graph up to rigid transforms.

    Parameters
    ----------
    pos_a : torch.Tensor
        First position tensor with shape ``[N, 2]``.
    pos_b : torch.Tensor
        Second position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor, optional
        Edge tensor with shape ``[2, E]``. Accepted for API compatibility with
        topology-aware comparisons.
    k : int, optional
        Number of neighbors used in the k-NN preservation score.

    Returns
    -------
    Dict[str, float]
        Dictionary with Procrustes, pairwise-distance, and k-NN similarity
        metrics.
    """
    del edge_index

    pos_a_cpu = _ensure_cpu(pos_a)
    pos_b_cpu = _ensure_cpu(pos_b)
    if pos_a_cpu.shape != pos_b_cpu.shape:
        raise ValueError(
            "layout_similarity requires tensors with matching shapes; "
            f"got {tuple(pos_a_cpu.shape)} and {tuple(pos_b_cpu.shape)}."
        )
    if pos_a_cpu.ndim != 2 or pos_a_cpu.shape[1] != 2:
        raise ValueError(
            "layout_similarity expects position tensors with shape [N, 2]; "
            f"got {tuple(pos_a_cpu.shape)}."
        )
    if pos_a_cpu.shape[0] <= 1:
        return {
            "procrustes_disparity": 0.0,
            "procrustes_similarity": 1.0,
            "pairwise_distance_correlation": 1.0,
            "knn_preservation": 1.0,
        }

    sampled_indices = torch.from_numpy(
        _deterministic_sample_indices(int(pos_a_cpu.shape[0]), min(int(pos_a_cpu.shape[0]), 1000))
    ).to(dtype=torch.long)
    sampled_pos_a = pos_a_cpu[sampled_indices]
    sampled_pos_b = pos_b_cpu[sampled_indices]

    procrustes_result = compare(sampled_pos_a, sampled_pos_b)
    disparity = procrustes_result.get("procrustes_disparity", 1.0)
    similarity = max(0.0, min(1.0, 1.0 - disparity))

    result = dict(procrustes_result)
    result["procrustes_similarity"] = similarity
    result["pairwise_distance_correlation"] = _pairwise_distance_correlation(
        sampled_pos_a, sampled_pos_b
    )
    result["knn_preservation"] = _sampled_knn_preservation(pos_a_cpu, pos_b_cpu, k=k)
    return result


def evaluate(
    graph: "DaguaGraph",
    positions: torch.Tensor,
    tier: str = "auto",
) -> Dict[str, float]:
    """Compute layout quality metrics for a graph and position tensor.

    Parameters
    ----------
    graph : DaguaGraph
        Graph providing topology, node sizes, cluster assignments, and layout
        direction metadata.
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    tier : str, optional
        ``"quick"``, ``"full"``, or ``"auto"``. Automatic mode selects the
        full metric set for graphs with fewer than 5000 nodes.

    Returns
    -------
    Dict[str, float]
        Computed metric dictionary.
    """
    edge_index = graph.edge_index
    node_sizes = (
        graph.node_sizes
        if graph.node_sizes is not None
        else torch.full((graph.num_nodes, 2), 20.0, dtype=positions.dtype)
    )
    direction = graph.direction if hasattr(graph, "direction") else "TB"
    cluster_ids = graph.cluster_ids if hasattr(graph, "cluster_ids") else None
    clusters = graph.clusters if getattr(graph, "clusters", None) else None
    cluster_parents = graph.cluster_parents if getattr(graph, "cluster_parents", None) else None
    cluster_labels = graph.cluster_labels if getattr(graph, "cluster_labels", None) else None

    effective_tier = tier
    if effective_tier == "auto":
        effective_tier = "full" if graph.num_nodes < 5000 else "quick"

    if effective_tier == "full":
        result_any = full(
            positions,
            edge_index,
            node_sizes=node_sizes,
            cluster_ids=cluster_ids,
            direction=direction,
            clusters=clusters,
            cluster_parents=cluster_parents,
            cluster_labels=cluster_labels,
        )
    elif effective_tier == "quick":
        result_any = quick(positions, edge_index, node_sizes=node_sizes, direction=direction)
    else:
        raise ValueError(f"Unsupported evaluation tier: {tier!r}")

    result = dict(result_any)
    if "composite_score" not in result:
        result["composite_score"] = composite(result)

    numeric_result: Dict[str, float] = {}
    for key, value in result.items():
        if isinstance(value, (int, float)):
            numeric_result[key] = float(value)
    return numeric_result


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


def compute_all_metrics(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    clusters: Optional[Dict] = None,
    direction: str = "TB",
) -> Dict[str, float]:
    """Legacy API — wraps quick() for backward compatibility.

    Returns the same keys as the old implementation plus new ones.
    """
    result = quick(positions, edge_index, node_sizes=node_sizes, direction=direction)

    pos_cpu = _ensure_cpu(positions)
    ei_cpu = _ensure_cpu(edge_index)

    # Map new keys to old keys for backward compat
    compat = {
        "num_nodes": result.get("_graph_n_nodes", 0),
        "num_edges": result.get("_graph_n_edges", 0),
        "edge_crossings": count_crossings(pos_cpu, ei_cpu),
        "dag_fraction": result.get("dag_consistency", 1.0),
        "node_overlaps": result.get("overlap_count", 0),
        "mean_edge_length": result.get("edge_length_mean", 0.0),
        "edge_length_variance": result.get("edge_length_std", 0.0) ** 2,
        "edge_straightness": result.get("edge_straightness_mean_deg", 0.0),
        "x_alignment": compute_x_alignment(pos_cpu, ei_cpu, direction=direction),
        "total_area": result.get("bbox_width", 0.0) * result.get("bbox_height", 0.0),
        "aspect_ratio": result.get("aspect_ratio", 1.0),
    }
    compat["overall_quality"] = overall_quality(compat)
    compat.update(result)
    return compat


def overall_quality(metrics: Dict[str, float]) -> float:
    """Legacy single scalar summarizing layout quality.  Higher is better."""
    score = 0.0
    score -= 100 * metrics.get("node_overlaps", 0)
    score -= 10 * metrics.get("edge_crossings", 0)
    score += 50 * metrics.get("dag_fraction", 1.0)
    score -= 1 * metrics.get("edge_length_variance", 0)
    score -= 2 * metrics.get("x_alignment", 0)
    n = max(metrics.get("num_nodes", 1), 1)
    score -= 0.5 * metrics.get("total_area", 0) / n
    return score


def graphviz_delta(dagua_metrics: Dict, graphviz_metrics: Dict) -> Dict[str, float]:
    """How much worse/better is Dagua vs Graphviz for each metric."""
    delta = {}
    for key in dagua_metrics:
        if key in graphviz_metrics and isinstance(dagua_metrics[key], (int, float)):
            delta[key] = dagua_metrics[key] - graphviz_metrics[key]
    return delta


# Legacy function-name aliases (old API returned scalars)


def count_crossings(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    seed: Optional[int] = None,
) -> int:
    """Count edge crossings exactly on small graphs and by sampling on large ones.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    seed : Optional[int], optional
        Seed forwarded to the sampled large-graph path. ``None`` preserves the
        existing stochastic behavior.

    Returns
    -------
    int
        Exact crossing count when ``E <= 500`` and sampled estimate otherwise.
    """
    pos = _ensure_cpu(pos)
    ei = _ensure_cpu(edge_index)
    if ei.numel() == 0 or ei.shape[1] < 2:
        return 0
    E = ei.shape[1]
    src, tgt = ei[0], ei[1]

    if E <= 500:
        ii, jj = torch.triu_indices(E, E, offset=1)
        e1s, e1t = src[ii], tgt[ii]
        e2s, e2t = src[jj], tgt[jj]
        shares_node = (e1s == e2s) | (e1s == e2t) | (e1t == e2s) | (e1t == e2t)
        valid = ~shares_node
        if not bool(valid.any().item()):
            return 0
        crossings = segments_intersect(
            pos[e1s[valid]],
            pos[e1t[valid]],
            pos[e2s[valid]],
            pos[e2t[valid]],
        )
        return int(crossings.sum().item())
    else:
        result = sampled_crossing_rate(pos, ei, n_samples=125000, seed=seed)
        return int(result["crossing_estimated_total"])


def _segments_intersect_scalar(a, b, c, d) -> bool:
    """Scalar segment intersection test for legacy count_crossings."""
    if np.allclose(a, c) or np.allclose(a, d) or np.allclose(b, c) or np.allclose(b, d):
        return False

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1, d2 = cross(c, d, a), cross(c, d, b)
    d3, d4 = cross(a, b, c), cross(a, b, d)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def compute_dag_fraction(
    pos: torch.Tensor, edge_index: torch.Tensor, direction: str = "TB"
) -> float:
    """Legacy: fraction of edges in correct DAG direction."""
    return dag_consistency(_ensure_cpu(pos), _ensure_cpu(edge_index), direction)["dag_consistency"]


def compute_edge_straightness(
    pos: torch.Tensor, edge_index: torch.Tensor, direction: str = "TB"
) -> float:
    """Legacy: mean angular deviation from primary axis (degrees)."""
    return edge_direction_straightness(_ensure_cpu(pos), _ensure_cpu(edge_index), direction)[
        "edge_straightness_mean_deg"
    ]


def compute_x_alignment(
    pos: torch.Tensor, edge_index: torch.Tensor, direction: str = "TB"
) -> float:
    """Legacy: mean cross-axis displacement between connected nodes."""
    pos = _ensure_cpu(pos)
    ei = _ensure_cpu(edge_index)
    if ei.numel() == 0:
        return 0.0
    src, tgt = ei[0], ei[1]
    if direction in ("LR", "RL"):
        return (pos[src, 1] - pos[tgt, 1]).abs().mean().item()
    return (pos[src, 0] - pos[tgt, 0]).abs().mean().item()


def count_overlaps(pos: torch.Tensor, node_sizes: torch.Tensor) -> int:
    """Legacy: count overlapping node pairs (returns int)."""
    return count_overlaps_detailed(_ensure_cpu(pos), _ensure_cpu(node_sizes))["overlap_count"]


def compute_mean_edge_length(pos: torch.Tensor, edge_index: torch.Tensor) -> float:
    """Legacy: average Euclidean edge length."""
    return edge_length_cv(_ensure_cpu(pos), _ensure_cpu(edge_index))["edge_length_mean"]


def compute_edge_length_variance(pos: torch.Tensor, edge_index: torch.Tensor) -> float:
    """Legacy: variance of edge lengths."""
    r = edge_length_cv(_ensure_cpu(pos), _ensure_cpu(edge_index))
    return r["edge_length_std"] ** 2


def compute_total_area(pos: torch.Tensor, node_sizes: torch.Tensor) -> float:
    """Legacy: bounding box area."""
    r = aspect_ratio(_ensure_cpu(pos), _ensure_cpu(node_sizes))
    return r["bbox_width"] * r["bbox_height"]


def compute_aspect_ratio(pos: torch.Tensor, node_sizes: torch.Tensor) -> float:
    """Legacy: width / height."""
    return aspect_ratio(_ensure_cpu(pos), _ensure_cpu(node_sizes))["aspect_ratio"]


def compute_min_node_gap(pos: torch.Tensor, node_sizes: torch.Tensor) -> float:
    """Legacy: minimum gap between non-overlapping node bboxes (sampled)."""
    pos = _ensure_cpu(pos)
    ns = _ensure_cpu(node_sizes)
    n = pos.shape[0]
    if n <= 1:
        return float("inf")
    min_gap = float("inf")
    for i in range(min(n, 200)):
        for j in range(i + 1, min(n, 200)):
            dx = abs(pos[i, 0] - pos[j, 0]).item()
            dy = abs(pos[i, 1] - pos[j, 1]).item()
            min_dx = (ns[i, 0] + ns[j, 0]).item() / 2
            min_dy = (ns[i, 1] + ns[j, 1]).item() / 2
            gap_x = dx - min_dx
            gap_y = dy - min_dy
            if gap_x > 0 or gap_y > 0:
                gap = max(gap_x, 0) + max(gap_y, 0)
                min_gap = min(min_gap, gap)
    return min_gap


# ---------------------------------------------------------------------------
# Full-drawing metrics (r80-S6) -- routed-path crossings, bends, and the
# drawing composite. Strictly ADDITIVE: nothing above this banner may change,
# and none of the placement composites reference anything below it.
# ---------------------------------------------------------------------------


def _bezier_is_straight(
    p0: Tuple[float, float],
    cp1: Tuple[float, float],
    cp2: Tuple[float, float],
    p1: Tuple[float, float],
    rel_tol: float = 1e-9,
) -> bool:
    """Report whether a cubic bezier degenerates to its chord segment.

    True when both control points are collinear with the chord AND their
    projections fall inside ``[0, 1]`` along it (an out-of-range collinear
    control point makes the curve overshoot the chord, so it is not the
    plain segment even though it lives on the same line).

    Parameters
    ----------
    p0, cp1, cp2, p1 : tuple[float, float]
        Cubic bezier control polygon.
    rel_tol : float, default=1e-9
        Collinearity tolerance relative to the chord length.

    Returns
    -------
    bool
        ``True`` when sampling the curve is equivalent to the chord.
    """
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    chord_sq = dx * dx + dy * dy
    if chord_sq < 1e-18:
        # Degenerate chord: straight only if the control points coincide too.
        return (
            abs(cp1[0] - p0[0]) < 1e-9
            and abs(cp1[1] - p0[1]) < 1e-9
            and abs(cp2[0] - p0[0]) < 1e-9
            and abs(cp2[1] - p0[1]) < 1e-9
        )
    tol = rel_tol * math.sqrt(chord_sq)
    for cx, cy in (cp1, cp2):
        rx = cx - p0[0]
        ry = cy - p0[1]
        cross = dx * ry - dy * rx
        if abs(cross) > tol * math.sqrt(chord_sq):
            return False
        t = (rx * dx + ry * dy) / chord_sq
        if t < -rel_tol or t > 1.0 + rel_tol:
            return False
    return True


def _curve_polyline(curve: Any, samples_per_edge: int = 8) -> List[Tuple[float, float]]:
    """Convert one routed curve into a polyline for intersection tests.

    Waypoint routes (ortho/taxi/external polylines) use their exact vertices,
    preserving hard bends. Degenerate-straight beziers shortcut to the chord
    ``[p0, p1]`` so results match straight-segment metrics exactly. Genuinely
    curved beziers are sampled uniformly in ``t``.

    Parameters
    ----------
    curve : Any
        ``dagua.edges.BezierCurve``-compatible object.
    samples_per_edge : int, default=8
        Number of sample points for genuinely curved beziers (>= 2).

    Returns
    -------
    List[Tuple[float, float]]
        Polyline vertices in draw order.
    """
    from dagua.edges import evaluate_bezier

    waypoints = getattr(curve, "waypoints", None)
    if waypoints is not None:
        return [(float(x), float(y)) for x, y in waypoints]
    p0, cp1, cp2, p1 = curve.p0, curve.cp1, curve.cp2, curve.p1
    if _bezier_is_straight(p0, cp1, cp2, p1):
        return [(float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))]
    n = max(2, int(samples_per_edge))
    return [evaluate_bezier(curve, k / (n - 1)) for k in range(n)]


def routed_crossing_rate(
    curves: Sequence[Any],
    edge_index: torch.Tensor,
    n_samples: int = 1_000_000,
    *,
    seed: Optional[int] = None,
    samples_per_edge: int = 8,
) -> Dict[str, float]:
    """Estimate edge-crossing rate along the ACTUAL routed paths.

    Same edge-pair sampling discipline as :func:`sampled_crossing_rate`
    (identical seed -> identical sampled pairs), but each edge is represented
    by its routed polyline instead of the straight node-center segment. A
    sampled pair counts as crossing when ANY segment of one polyline
    intersects ANY segment of the other. When every curve degenerates to the
    straight node-center segment, the result is bit-identical to
    ``sampled_crossing_rate`` on the same seed.

    Parameters
    ----------
    curves : Sequence[Any]
        One ``BezierCurve``-compatible route per edge, aligned to
        ``edge_index`` columns.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    n_samples : int, optional
        Maximum number of edge pairs to sample.
    seed : Optional[int], optional
        Seed for edge-pair sampling; ``None`` uses the global RNG state.
    samples_per_edge : int, default=8
        Polyline sample count for genuinely curved beziers. Waypoint routes
        use their exact vertices; straight routes shortcut to 2 points.

    Returns
    -------
    Dict[str, float]
        ``routed_crossing_rate`` plus SE/total/sample-count fields mirroring
        :func:`sampled_crossing_rate`.

    Notes
    -----
    Padding note: polylines are padded to a common length by repeating their
    last vertex; degenerate padded segments cannot produce transversal or
    collinear-overlap intersections, so padding never adds crossings.
    Complexity is ``O(n_samples * samples_per_edge^2)``.
    """
    empty = {
        "routed_crossing_rate": 0.0,
        "routed_crossing_se": 0.0,
        "routed_crossing_estimated_total": 0,
        "routed_crossing_n_samples": 0,
        "routed_crossing_eligible_pairs": 0.0,
    }
    if edge_index.numel() == 0 or edge_index.shape[1] < 2 or not curves:
        return empty

    E = edge_index.shape[1]
    if len(curves) != E:
        raise ValueError(
            f"routed_crossing_rate: got {len(curves)} curves for {E} edges; "
            f"curves must align 1:1 with edge_index columns."
        )
    src, tgt = edge_index[0], edge_index[1]
    total_pairs = E * (E - 1) // 2
    gen = None if seed is None else torch.Generator(device="cpu").manual_seed(int(seed))

    # Pair sampling: replicated verbatim from sampled_crossing_rate so a
    # shared seed selects the SAME eligible pair population. (Deliberate
    # duplication -- refactoring the frozen placement metric is out of scope
    # for the additive drawing-metrics stream.)
    if total_pairs <= n_samples:
        ii, jj = torch.triu_indices(E, E, offset=1)
        idx1 = ii
        idx2 = jj
        pair_space_exhausted = True
    else:
        if total_pairs <= 10_000_000:
            perm = torch.randperm(total_pairs, generator=gen)[:n_samples]
            ii, jj = torch.triu_indices(E, E, offset=1)
            idx1 = ii[perm]
            idx2 = jj[perm]
            pair_space_exhausted = False
        else:
            idx1 = torch.randint(0, E, (n_samples,), generator=gen)
            idx2 = torch.randint(0, E, (n_samples,), generator=gen)
            lo = torch.minimum(idx1, idx2)
            hi = torch.maximum(idx1, idx2)
            idx1, idx2 = lo, hi
            pair_space_exhausted = False

    e1s, e1t = src[idx1], tgt[idx1]
    e2s, e2t = src[idx2], tgt[idx2]
    shares_node = (e1s == e2s) | (e1s == e2t) | (e1t == e2s) | (e1t == e2t)
    same_edge = idx1 == idx2
    valid = ~shares_node & ~same_edge
    if valid.sum() == 0:
        return empty

    # Build padded polyline tensor [E, S, 2]; pad by repeating the last
    # vertex (degenerate segments never register as crossings -- see Notes).
    polylines = [_curve_polyline(curve, samples_per_edge) for curve in curves]
    S = max(len(p) for p in polylines)
    poly = torch.empty((E, S, 2), dtype=torch.float32)
    for e, pts in enumerate(polylines):
        for k in range(S):
            x, y = pts[k] if k < len(pts) else pts[-1]
            poly[e, k, 0] = float(x)
            poly[e, k, 1] = float(y)

    idx1v = idx1[valid]
    idx2v = idx2[valid]
    K = int(idx1v.numel())
    n_seg = S - 1

    crossing_any = torch.zeros(K, dtype=torch.bool)
    # Chunk so each segments_intersect batch stays under ~2M rows.
    rows_per_pair = n_seg * n_seg
    chunk = max(1, 2_000_000 // max(1, rows_per_pair))
    for start in range(0, K, chunk):
        stop = min(start + chunk, K)
        a = poly[idx1v[start:stop]]  # [k, S, 2]
        b = poly[idx2v[start:stop]]
        k = a.shape[0]
        a1 = a[:, :-1, :].unsqueeze(2).expand(k, n_seg, n_seg, 2).reshape(-1, 2)
        a2 = a[:, 1:, :].unsqueeze(2).expand(k, n_seg, n_seg, 2).reshape(-1, 2)
        b1 = b[:, :-1, :].unsqueeze(1).expand(k, n_seg, n_seg, 2).reshape(-1, 2)
        b2 = b[:, 1:, :].unsqueeze(1).expand(k, n_seg, n_seg, 2).reshape(-1, 2)
        hits = segments_intersect(a1, a2, b1, b2).reshape(k, rows_per_pair)
        crossing_any[start:stop] = hits.any(dim=1)

    n_valid = K
    rate = crossing_any.float().mean().item()
    if pair_space_exhausted:
        eligible_pairs = float(n_valid)
        estimated_total = int(crossing_any.sum().item())
        se = 0.0
    else:
        eligible_pairs = float(total_pairs) * (float(n_valid) / float(valid.numel()))
        estimated_total = int(rate * eligible_pairs)
        se = eligible_pairs * (rate * (1 - rate) / n_valid) ** 0.5 if n_valid > 0 else 0.0

    return {
        "routed_crossing_rate": rate,
        "routed_crossing_se": se,
        "routed_crossing_estimated_total": estimated_total,
        "routed_crossing_n_samples": n_valid,
        "routed_crossing_eligible_pairs": eligible_pairs,
    }


def bend_count(
    curves: Sequence[Any],
    angle_threshold_deg: float = 15.0,
    samples_per_edge: int = 16,
) -> Dict[str, float]:
    """Count hard direction changes along routed edges.

    Intended for ortho/taxi (waypoint) routings, where every elbow is a
    real bend; smooth beziers sampled at ``samples_per_edge`` points spread
    their turning over many small inter-segment angles that stay below the
    threshold unless the curve genuinely kinks.

    Parameters
    ----------
    curves : Sequence[Any]
        Routed curves, one per edge.
    angle_threshold_deg : float, default=15.0
        Minimum absolute direction change (degrees) at an interior vertex to
        count as a bend.
    samples_per_edge : int, default=16
        Sample count for non-waypoint curves.

    Returns
    -------
    Dict[str, float]
        ``bend_total`` (int), ``bend_mean_per_edge``, and
        ``bend_edges_with_bends`` (count of edges with >= 1 bend).
    """
    if not curves:
        return {"bend_total": 0, "bend_mean_per_edge": 0.0, "bend_edges_with_bends": 0}

    threshold_rad = math.radians(angle_threshold_deg)
    total = 0
    edges_with_bends = 0
    for curve in curves:
        pts = _curve_polyline(curve, samples_per_edge)
        bends = 0
        for i in range(1, len(pts) - 1):
            v1x = pts[i][0] - pts[i - 1][0]
            v1y = pts[i][1] - pts[i - 1][1]
            v2x = pts[i + 1][0] - pts[i][0]
            v2y = pts[i + 1][1] - pts[i][1]
            if (v1x * v1x + v1y * v1y) < 1e-18 or (v2x * v2x + v2y * v2y) < 1e-18:
                continue
            angle = abs(math.atan2(v1x * v2y - v1y * v2x, v1x * v2x + v1y * v2y))
            if angle > threshold_rad:
                bends += 1
        total += bends
        if bends > 0:
            edges_with_bends += 1

    return {
        "bend_total": total,
        "bend_mean_per_edge": total / len(curves),
        "bend_edges_with_bends": edges_with_bends,
    }


def composite_drawing(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    curves: Sequence[Any],
    label_positions: Optional[List[Optional[Tuple[float, float]]]] = None,
    edge_labels: Optional[list] = None,
    *,
    seed: Optional[int] = 0,
    crossing_samples: int = 100_000,
    samples_per_edge: int = 8,
) -> Dict[str, Any]:
    """Score the FULL DRAWING (routed edges + labels) on a 0-100 scale.

    This is a NEW composite, parallel to (and never touching) the placement
    composites. It is computable from ``(positions, sizes, curves,
    label_positions)`` alone -- no live layout run -- and is deterministic
    for a fixed ``seed``.

    Shared common-ruler terms retain their frozen weights and formulas: KSM
    25, routed EC 20, saturating NO 13, and routed-port AR 4. Drawing-only
    terms divide the remaining 38 points in their established proportions.
    Inapplicable label terms are dropped and all remaining weights are
    renormalized; they never receive neutral half-credit.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor ``[2, E]``.
    node_sizes : torch.Tensor
        Node bbox sizes ``[N, 2]``.
    curves : Sequence[Any]
        Routed curves aligned to ``edge_index`` columns.
    label_positions : list[tuple[float, float] | None] | None, optional
        Edge-label anchor positions (None entries = unlabeled edge).
    edge_labels : list | None, optional
        Edge label strings aligned with ``label_positions``.
    seed : int | None, default=0
        Seed for the routed crossing sampler. The default is a FIXED seed so
        the composite is deterministic out of the box.
    crossing_samples : int, default=100_000
        Edge-pair sample budget for the crossing term.
    samples_per_edge : int, default=8
        Polyline resolution for curved beziers.

    Returns
    -------
    Dict[str, Any]
        ``composite_drawing`` (0-100) plus every component value and term.
    """
    pos = _ensure_cpu(pos)
    ei = _ensure_cpu(edge_index)
    sizes = _ensure_cpu(node_sizes)

    rc = routed_crossing_rate(
        curves, ei, crossing_samples, seed=seed, samples_per_edge=samples_per_edge
    )
    enc = edge_node_crossing_count(curves, pos, sizes, ei)
    par = port_angular_resolution(curves, ei)
    ecc = edge_curvature_consistency(curves)
    bends = bend_count(curves)
    overlaps = count_overlaps(pos, sizes)
    ksm = isotonic_stress(pos, ei)

    has_labels = bool(
        label_positions is not None
        and edge_labels is not None
        and any(lbl for lbl in edge_labels)
        and any(lp is not None for lp in label_positions)
    )
    if has_labels:
        assert label_positions is not None
        assert edge_labels is not None
        lo = label_overlap_count(label_positions, edge_labels, pos, sizes)
        label_overlaps = int(lo["label_overlaps"])
        label_node_overlaps = int(lo["label_node_overlaps"])
        term_label_node = 1.0 / (1.0 + label_node_overlaps)
        term_label_label = 1.0 / (1.0 + label_overlaps)
    else:
        label_overlaps = 0
        label_node_overlaps = 0
        term_label_node = None
        term_label_label = None

    c_max = _degree_corrected_crossing_max(ei, int(pos.shape[0]))
    crossing_count = int(rc["routed_crossing_estimated_total"])
    term_crossing = 1.0 if c_max <= 0 else 1.0 - math.sqrt(min(1.0, crossing_count / c_max))
    term_edge_node = max(0.0, 1.0 - float(enc["edge_node_crossing_rate"]) * 10.0)
    term_port = par["port_angular_resolution_score"]
    term_overlap = 1.0 / (1.0 + 2.0 * overlaps / max(int(pos.shape[0]), 1))
    term_stress = ksm["ksm_score"]
    term_curvature = max(0.0, 1.0 - min(1.0, ecc["edge_curvature_cv"]))
    term_bend = max(0.0, 1.0 - bends["bend_mean_per_edge"] / 4.0)

    drawing_terms = {
        "stress": (25.0, term_stress),
        "crossing": (20.0, term_crossing),
        "overlap": (13.0, term_overlap),
        "port": (4.0, term_port),
        "edge_node": (95.0 / 6.0, term_edge_node),
        "label_node": (95.0 / 16.0, term_label_node),
        "label_label": (95.0 / 16.0, term_label_label),
        "curvature": (19.0 / 3.0, term_curvature),
        "bend": (95.0 / 24.0, term_bend),
    }
    applicable_weight = sum(weight for weight, value in drawing_terms.values() if value is not None)
    score = (
        100.0
        * sum(
            weight * float(value) for weight, value in drawing_terms.values() if value is not None
        )
        / applicable_weight
    )

    return {
        "composite_drawing": score,
        "drawing_crossing_rate": rc["routed_crossing_rate"],
        "drawing_crossing_estimated_total": rc["routed_crossing_estimated_total"],
        "drawing_crossing_c_max": c_max,
        "drawing_edge_node_crossings": enc["edge_node_crossings"],
        "drawing_edge_node_crossing_rate": enc["edge_node_crossing_rate"],
        "drawing_port_angular_deg": par["port_angular_res_mean_deg"],
        "drawing_curvature_cv": ecc["edge_curvature_cv"],
        "drawing_bend_mean_per_edge": bends["bend_mean_per_edge"],
        "drawing_bend_total": bends["bend_total"],
        "drawing_label_overlaps": label_overlaps,
        "drawing_label_node_overlaps": label_node_overlaps,
        "drawing_node_overlaps": overlaps,
        "drawing_has_labels": has_labels,
        "drawing_term_stress": term_stress,
        "drawing_term_crossing": term_crossing,
        "drawing_term_edge_node": term_edge_node,
        "drawing_term_label_node": term_label_node,
        "drawing_term_label_label": term_label_label,
        "drawing_term_port": term_port,
        "drawing_term_overlap": term_overlap,
        "drawing_term_curvature": term_curvature,
        "drawing_term_bend": term_bend,
    }
