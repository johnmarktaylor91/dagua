"""Layout quality metrics suite for dagua.

Three-tier system designed for computational feasibility at 50M+ nodes:
- Tier 1: O(N) or O(|E|), always compute
- Tier 2: Sampled, O(n_samples * BFS_cost), compute when feasible
- Tier 3: DAG/hierarchy-specific, O(N) given metadata

API:
- quick(pos, edge_index, ...) -> dict   — Tier-1 only, runs in seconds at any scale
- full(pos, edge_index, ...) -> dict    — All tiers including sampled metrics
- compare(pos_a, pos_b) -> dict         — Procrustes comparison of two layouts
- composite(metrics) -> float           — Single scalar 0-100

All metric functions are also importable individually.
"""

from __future__ import annotations

import math
import time as _time
from collections import deque
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

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
    EPS = 1e-6
    d1 = p2 - p1  # [N, 2]
    d2 = p4 - p3
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]  # [N]

    parallel = cross.abs() < 1e-10

    # Sign-preserving division: clamp(min=1e-10) would flip the sign of
    # negative-cross orientations, silently turning valid intersections
    # into false negatives. Replace only the near-zero (parallel) cases
    # with 1 to avoid div-by-zero.
    safe_cross = torch.where(parallel, torch.ones_like(cross), cross)

    d3 = p3 - p1
    t = (d3[:, 0] * d2[:, 1] - d3[:, 1] * d2[:, 0]) / safe_cross
    u = (d3[:, 0] * d1[:, 1] - d3[:, 1] * d1[:, 0]) / safe_cross

    # Standard transversal: strict-interior intersection on both segments.
    proper = (~parallel) & (t > EPS) & (t < 1 - EPS) & (u > EPS) & (u < 1 - EPS)

    # Collinear-overlap: parallel + p3 lies on the line through p1-p2 +
    # the projections of (p3, p4) onto p1-p2's line direction overlap
    # the open interval (EPS, 1 - EPS).
    collinear = parallel & ((d1[:, 0] * d3[:, 1] - d1[:, 1] * d3[:, 0]).abs() < 1e-8)
    collinear_overlap = torch.zeros_like(parallel)
    if bool(collinear.any().item()):
        d1_norm_sq = d1[:, 0].pow(2) + d1[:, 1].pow(2)
        d1_norm_sq_safe = torch.where(
            d1_norm_sq < 1e-10,
            torch.ones_like(d1_norm_sq),
            d1_norm_sq,
        )
        t3 = (d3[:, 0] * d1[:, 0] + d3[:, 1] * d1[:, 1]) / d1_norm_sq_safe
        d4 = p4 - p1
        t4 = (d4[:, 0] * d1[:, 0] + d4[:, 1] * d1[:, 1]) / d1_norm_sq_safe
        a = torch.minimum(t3, t4)
        b = torch.maximum(t3, t4)
        collinear_overlap = collinear & (a < 1 - EPS) & (b > EPS)

    return proper | collinear_overlap


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


def depth_position_correlation(pos: torch.Tensor, topo_depth: torch.Tensor) -> Dict[str, float]:
    """Spearman rank correlation between topological depth and y-coordinate.

    Complexity: O(N log N) for the sort.
    Target: > 0.95 good, > 0.99 excellent.
    """
    from scipy.stats import spearmanr

    depth_np = (
        topo_depth.cpu().numpy() if isinstance(topo_depth, torch.Tensor) else np.array(topo_depth)
    )
    y_np = pos[:, 1].cpu().numpy()
    rho, pval = spearmanr(depth_np, y_np)
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
        csr_off, csr_tgt = _build_csr(ei, N)
        all_pairs_dist = _all_pairs_unweighted(csr_off, csr_tgt, N, max_dist=max_dist)
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


def neighborhood_preservation(
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
        csr_off, csr_tgt = _build_csr(ei, N)
        all_pairs_dist = _all_pairs_unweighted(csr_off, csr_tgt, N, max_dist=k)
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
        # Random sample of pairs
        n_pairs = min(20000, n_clusters * (n_clusters - 1) // 2)
        i_idx = np.random.randint(0, n_clusters, n_pairs)
        j_idx = np.random.randint(0, n_clusters, n_pairs)
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
    curves,
    edge_index: torch.Tensor,
) -> Dict[str, float]:
    """Minimum angle between incident edge tangents at actual ports.

    Tier 2 metric.
    """
    from dagua.edges import bezier_tangent

    if not curves or edge_index.numel() == 0:
        return {"port_angular_res_mean_deg": 360.0}

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
        node_tangents.setdefault(t, []).append((tdx, tdy))

    min_angles = []
    for node, tangents in node_tangents.items():
        if len(tangents) < 2:
            continue

        # Compute angles
        angles = []
        for dx, dy in tangents:
            angles.append(math.atan2(dy, dx))
        angles.sort()

        # Min angle between adjacent
        diffs = [angles[i + 1] - angles[i] for i in range(len(angles) - 1)]
        diffs.append(2 * math.pi - (angles[-1] - angles[0]))
        min_angles.append(min(diffs))

    if not min_angles:
        return {"port_angular_res_mean_deg": 360.0}

    mean_deg = sum(a * 180 / math.pi for a in min_angles) / len(min_angles)
    return {"port_angular_res_mean_deg": mean_deg}


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


def composite(metrics: Dict[str, float]) -> float:
    """Compute the directed 100-point composite quality score.

    Parameters
    ----------
    metrics : Dict[str, float]
        Metric name to scalar value mapping from ``full()`` or equivalent.

    Returns
    -------
    float
        Weighted composite score where higher is better.

    Notes
    -----
    Weights sum to 100:
    - DAG consistency: 22
    - Edge length uniformity (1 - CV): 18
    - Depth correlation: 13
    - No overlaps (binary): 8
    - Edge straightness: 9
    - Crossings (inverted): 9
    - Stress (inverted sampled_stress): 10
    - Angular resolution: 5
    - Cluster separation: 6
    """
    score = 0.0

    # DAG consistency (22) - most critical for directed graph readability.
    score += 22 * metrics.get("dag_consistency", 0.0)

    # Edge length uniformity (18) - invert CV, cap at 1.0.
    score += 18 * max(0.0, 1.0 - metrics.get("edge_length_cv", 1.0))

    # Depth correlation (13)
    score += 13 * max(0.0, metrics.get("depth_spearman_rho", 0.0))

    # No overlaps (8) - binary.
    score += 8 * (1.0 if metrics.get("overlap_count", 1) == 0 else 0.0)

    # Edge straightness (9) - lower deviation = better.
    straight_deg = metrics.get("edge_straightness_mean_deg", 45.0)
    score += 9 * max(0.0, 1.0 - straight_deg / 45.0)

    # Crossing density (9) - lower is better.
    crossing_score = max(0.0, 1.0 - metrics.get("crossing_rate", 0.5) * 10)
    score += 9 * crossing_score

    # Sampled stress (10) - lower graph-theoretic stress is better.
    stress_score = max(0.0, 1.0 - metrics.get("sampled_stress", 1.0))
    score += 10 * stress_score

    # Angular resolution (5)
    angle_score = min(1.0, metrics.get("angular_res_mean_deg", 20.0) / 40.0)
    score += 5 * angle_score

    # Cluster separation (6)
    if "cluster_mean_sep_ratio" in metrics:
        sep_score = min(1.0, metrics["cluster_mean_sep_ratio"] / 5.0)
        score += 6 * sep_score
    else:
        score += 6 * 0.5  # neutral if no clusters

    return score


def composite_undirected(metrics: Dict[str, float]) -> float:
    """Composite for undirected graphs: drops direction-sensitive metrics.

    Dropped (50 pts total): dag_consistency (25), depth_spearman (15),
    edge_straightness (10). Remaining 50 pts are rescaled to 100.

    Retained metrics and rescaled weights (sum=100):
      edge_length_cv:        20 -> 40
      overlap_count:         10 -> 20
      crossing_rate:         10 -> 20
      angular_resolution:     5 -> 10
      cluster_separation:     5 -> 10

    Same rule per metric as `composite()` (lower-better terms inverted,
    same clamping). See composite() for the per-metric formulas and clamp
    ranges.

    Parameters
    ----------
    metrics : Dict[str, float]
        Metric name to scalar value mapping from ``full()`` or equivalent.

    Returns
    -------
    float
        Weighted undirected composite score where higher is better.
    """
    score = 0.0

    # Edge length uniformity (40) - invert CV, cap at 1.0
    score += 40 * max(0.0, 1.0 - metrics.get("edge_length_cv", 1.0))

    # No overlaps (20) - binary
    score += 20 * (1.0 if metrics.get("overlap_count", 1) == 0 else 0.0)

    # Crossing density (20) - lower is better
    crossing_score = max(0.0, 1.0 - metrics.get("crossing_rate", 0.5) * 10)
    score += 20 * crossing_score

    # Angular resolution (10)
    angle_score = min(1.0, metrics.get("angular_res_mean_deg", 20.0) / 40.0)
    score += 10 * angle_score

    # Cluster separation (10)
    if "cluster_mean_sep_ratio" in metrics:
        sep_score = min(1.0, metrics["cluster_mean_sep_ratio"] / 5.0)
        score += 10 * sep_score
    elif "cluster_sep" in metrics:
        sep_score = min(1.0, metrics["cluster_sep"])
        score += 10 * sep_score
    else:
        score += 10 * 0.5  # neutral if no clusters

    return score


def composite_auto(
    metrics: Dict[str, float], is_semantically_directed: Optional[bool] = None
) -> float:
    """Pick composite or composite_undirected based on direction flag.

    When ``is_semantically_directed`` is True (or None, conservative
    default), returns ``composite(metrics)``. When False, returns
    ``composite_undirected(metrics)``.

    Parameters
    ----------
    metrics : Dict[str, float]
        Metric name to scalar value mapping from ``full()`` or equivalent.
    is_semantically_directed : Optional[bool], optional
        Whether the graph has a meaningful direction. ``None`` is treated as
        directed to preserve the existing conservative behavior.

    Returns
    -------
    float
        Directed or undirected composite score.
    """
    if is_semantically_directed is None or is_semantically_directed:
        return composite(metrics)
    return composite_undirected(metrics)


# Quick-mode fields (available from `quick()`). Used by composite_large to
# avoid silent defaults for fields that only exist in full().
_QUICK_AVAILABLE_FIELDS = frozenset(
    {
        "dag_consistency",
        "edge_length_cv",
        "depth_spearman_rho",
        "overlap_count",
        "edge_straightness_mean_deg",
    }
)

# Fields that composite() expects but quick() does not provide. Required for
# composite_small (the full profile). Missing any of these at N<=2000 is a bug.
_FULL_ONLY_FIELDS = frozenset(
    {
        "crossing_rate",
        "sampled_stress",
        "angular_res_mean_deg",
    }
)


def composite_large(metrics: Dict[str, float]) -> float:
    """Composite score for graphs where only ``quick()`` metrics are available.

    At N>2000 the benchmark switches to ``quick()``, which does not compute
    crossing_rate, angular_res_mean_deg, cluster_mean_sep_ratio, or
    edge_node_crossing_rate. ``composite()`` would silently default those
    fields and produce a score that is NOT comparable to small-graph scores.

    ``composite_large`` uses ONLY quick-available fields and renormalizes to
    0-100. It IS the correct score to use for N>2000 iteration/held-out
    comparisons. It is NOT comparable to ``composite()`` scores computed on
    full metrics at smaller N; see the "profile split" protocol in
    ``.project-context/plans/native_placement_algo/04_evaluation_rubric.md``.

    Weights (sum = 100):
    - DAG consistency: 30
    - Edge length uniformity (1 - CV): 25
    - Depth correlation: 20
    - No overlaps (binary): 15
    - Edge straightness: 10
    """
    missing = [f for f in _QUICK_AVAILABLE_FIELDS if f not in metrics]
    if missing:
        raise ValueError(
            f"composite_large: missing required quick-mode fields: {missing}. "
            f"Did you call quick() before scoring? If the graph is empty or "
            f"trivial, the caller must still populate the expected fields."
        )

    score = 0.0
    score += 30 * metrics["dag_consistency"]
    score += 25 * max(0.0, 1.0 - metrics["edge_length_cv"])
    score += 20 * max(0.0, metrics["depth_spearman_rho"])
    score += 15 * (1.0 if metrics["overlap_count"] == 0 else 0.0)
    score += 10 * max(0.0, 1.0 - metrics["edge_straightness_mean_deg"] / 45.0)
    return score


def composite_strict(metrics: Dict[str, float]) -> float:
    """Strict variant of ``composite()`` that refuses silent defaults.

    Raises ValueError if any required full-profile field is absent.
    Use at N<=2000 where ``full()`` has run; use ``composite_large`` at N>2000.
    """
    missing = [f for f in _FULL_ONLY_FIELDS | _QUICK_AVAILABLE_FIELDS if f not in metrics]
    if missing:
        raise ValueError(
            f"composite_strict: missing required full-profile fields: {missing}. "
            f"At N>2000 use composite_large() instead."
        )
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
    }

    # Edge length CV
    result.update(edge_length_cv(pos, ei))

    # DAG consistency
    result.update(dag_consistency(pos, ei, direction=direction, back_edge_mask=bem))

    # Depth-position correlation
    if topo_depth is None and ei.numel() > 0:
        from dagua.utils import longest_path_layering

        topo_depth = longest_path_layering(ei, N)

    if topo_depth is not None:
        if not isinstance(topo_depth, torch.Tensor):
            topo_depth = torch.tensor(topo_depth, dtype=torch.long)
        result.update(depth_position_correlation(pos, topo_depth))

    # Overlap count
    if node_sizes is not None:
        ns = _ensure_cpu(node_sizes)
        result.update(count_overlaps_detailed(pos, ns, seed=seed))

    # Aspect ratio
    ns_arg = _ensure_cpu(node_sizes) if node_sizes is not None else None
    result.update(aspect_ratio(pos, ns_arg))

    # Edge straightness
    result.update(edge_direction_straightness(pos, ei, direction=direction))

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
    )

    # Tier 2: Sampled metrics. Stress and neighborhood preservation both need
    # graph distances, so full() computes them once per metric evaluation.
    all_pairs_dist = None
    if ei.numel() > 0 and N >= 2:
        csr_off, csr_tgt = _build_csr(ei, N)
        all_pairs_dist = _all_pairs_unweighted(csr_off, csr_tgt, N, max_dist=20)

    result.update(
        sampled_stress(
            pos,
            ei,
            N,
            n_sources=stress_sources,
            n_targets=stress_targets,
            all_pairs_dist=all_pairs_dist,
        )
    )
    # Pass an explicit seed so composite scoring is deterministic
    # for the same positions. measurement found per-call composite
    # spreads up to 6.9 (std 2.6) on small graphs because sampled_crossing_rate
    # was reading from the global torch RNG. Polish pickers had to torch.manual_seed
    # before each candidate; this makes the metric self-deterministic.
    result.update(sampled_crossing_rate(pos, ei, n_samples=crossing_samples, seed=0))
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

    Weights (sum = 100) and rationale:
    - Routed crossings: 30. Edge-edge crossings measured on the paths the
      reader actually sees are the dominant legibility defect in a finished
      drawing (Purchase 1997 ranks crossings first among aesthetics).
    - Edge-node crossings: 20. Edges plowing through unrelated node boxes
      destroy node readability and imply false adjacency.
    - Label overlaps: 15 (7.5 label-vs-node + 7.5 label-vs-label). Occluded
      text is unreadable text; neutral half-credit when the graph has no
      edge labels (same convention as composite()'s cluster term).
    - Port angular resolution: 12. Edges leaving a node at indistinguishable
      angles cannot be traced; measured at the true routed tangents.
    - Node overlap sanity: 10. Binary, mirroring composite(): overlapping
      nodes mean the drawing failed upstream of routing.
    - Curvature consistency: 8. Uniform curvature reads as a deliberate
      style; wildly mixed straight/hooked edges read as noise.
    - Bend economy: 5. Ortho/taxi elbows are fine in moderation; >= 4 bends
      per edge on average makes paths untraceable.

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

    has_labels = bool(
        label_positions is not None
        and edge_labels is not None
        and any(lbl for lbl in edge_labels)
        and any(lp is not None for lp in label_positions)
    )
    if has_labels:
        lo = label_overlap_count(label_positions, edge_labels, pos, sizes)
        label_overlaps = int(lo["label_overlaps"])
        label_node_overlaps = int(lo["label_node_overlaps"])
        term_label_node = 1.0 / (1.0 + label_node_overlaps)
        term_label_label = 1.0 / (1.0 + label_overlaps)
    else:
        label_overlaps = 0
        label_node_overlaps = 0
        # Neutral half-credit when the graph has no edge labels, mirroring
        # composite()'s treatment of the cluster term on cluster-free graphs.
        term_label_node = 0.5
        term_label_label = 0.5

    # Same clamp conventions as the placement composites where a direct
    # analog exists (crossing x10 clamp, angular /40 clamp, binary overlap).
    term_crossing = max(0.0, 1.0 - rc["routed_crossing_rate"] * 10.0)
    term_edge_node = max(0.0, 1.0 - float(enc["edge_node_crossing_rate"]) * 10.0)
    term_port = min(1.0, par["port_angular_res_mean_deg"] / 40.0)
    term_overlap = 1.0 if overlaps == 0 else 0.0
    term_curvature = max(0.0, 1.0 - min(1.0, ecc["edge_curvature_cv"]))
    term_bend = max(0.0, 1.0 - bends["bend_mean_per_edge"] / 4.0)

    score = 0.0
    score += 30.0 * term_crossing
    score += 20.0 * term_edge_node
    score += 7.5 * term_label_node
    score += 7.5 * term_label_label
    score += 12.0 * term_port
    score += 10.0 * term_overlap
    score += 8.0 * term_curvature
    score += 5.0 * term_bend

    return {
        "composite_drawing": score,
        "drawing_crossing_rate": rc["routed_crossing_rate"],
        "drawing_crossing_estimated_total": rc["routed_crossing_estimated_total"],
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
        "drawing_term_crossing": term_crossing,
        "drawing_term_edge_node": term_edge_node,
        "drawing_term_label_node": term_label_node,
        "drawing_term_label_label": term_label_label,
        "drawing_term_port": term_port,
        "drawing_term_overlap": term_overlap,
        "drawing_term_curvature": term_curvature,
        "drawing_term_bend": term_bend,
    }
