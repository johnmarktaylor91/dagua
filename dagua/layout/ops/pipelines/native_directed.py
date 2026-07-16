"""Honest directed-table portfolio for declared-hierarchical DAGs."""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import ClassVar, Dict, Optional, Tuple

import numpy as np
import torch

from dagua.config import LayoutConfig
from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

MAX_DIRECTED_CONTEST_NODES = 2000
DIRECTED_FULL_REFEREE_TOP_K = 6
DIRECTED_FULL_SCORE_MIN_REMAINING_S = 5.0
DIRECTED_LARGE_NODE_THRESHOLD = 250
DIRECTED_LARGE_GRID_MIN_REMAINING_S = 240.0
DIRECTED_GRID_DUMMY_LIMIT = 10_000
DIRECTED_GRID_WIDTH_LIMIT = 80
DIRECTED_SUGIYAMA_SIMPLEX_PRIOR_S = 60.0
DIRECTED_FORCE_PRIOR_S = 90.0
DIRECTED_PREDICTED_COST_MULTIPLIER = 2.0
FORCE_SKIP_RATIO_THRESHOLD = 0.3
DIRECTED_NARROW_SEED_NODE_CAP = 128
DIRECTED_NARROW_SEED_EDGE_CAP = 512
DIRECTED_MRTREE_EDGE_NODE_RATIO_MAX = 3.0
DIRECTED_MRTREE_MAX_RANK_WIDTH = 6
DIRECTED_STRESS_BLEND_WEIGHTS = (0.2, 0.4)
SCALED_SUGIYAMA_RANK_SEP = 72.0
SCALED_SUGIYAMA_NODE_SEP = 18.0
SUGIYAMA_FIDELITY_MODES = ("graphviz_dot", "graphviz", "igraph")
SUGIYAMA_RANK_SEP_GRID = (36.0, 72.0, 108.0)
SUGIYAMA_NODE_SEP_GRID = (18.0, 36.0, 54.0)
IGRAPH_OUTPUT_SCALE = 50.0
_LOGGER = logging.getLogger(__name__)


def _score_directed_candidate(
    pos: torch.Tensor,
    problem: LayoutProblem,
    cluster_ids: Optional[torch.Tensor],
    all_pairs_dist: Optional[np.ndarray] = None,
) -> float:
    """Score a finalist with the frozen honest directed composite.

    Parameters
    ----------
    pos : torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    problem : LayoutProblem
        Directed acyclic layout problem.
    cluster_ids : torch.Tensor, optional
        Per-node cluster ids with shape ``[N]``.
    all_pairs_dist : Optional[numpy.ndarray], optional
        Cached unweighted shortest paths with shape ``[N, N]``.

    Returns
    -------
    float
        Higher-is-better directed composite score.
    """
    from dagua.metrics import composite_auto, full

    numeric = full(
        pos.detach().to(device="cpu", dtype=torch.float32),
        problem.edge_index.detach().to(device="cpu"),
        node_sizes=(
            None
            if problem.node_sizes is None
            else problem.node_sizes.detach().to(device="cpu", dtype=torch.float32)
        ),
        cluster_ids=cluster_ids,
        direction=problem.direction,
        all_pairs_dist=all_pairs_dist,
    )
    numeric["declared_hierarchical"] = True
    return float(composite_auto(numeric, is_semantically_directed=True))


def _score_directed_candidate_cached(
    pos: torch.Tensor,
    problem: LayoutProblem,
    cluster_ids: Optional[torch.Tensor],
    all_pairs_dist: Optional[np.ndarray],
) -> float:
    """Score a directed candidate while preserving old monkeypatch arity.

    Parameters
    ----------
    pos : torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    problem : LayoutProblem
        Directed acyclic layout problem.
    cluster_ids : torch.Tensor, optional
        Per-node cluster ids with shape ``[N]``.
    all_pairs_dist : Optional[numpy.ndarray]
        Cached unweighted shortest paths with shape ``[N, N]``.

    Returns
    -------
    float
        Higher-is-better directed composite score.
    """
    try:
        return _score_directed_candidate(
            pos,
            problem,
            cluster_ids,
            all_pairs_dist=all_pairs_dist,
        )
    except TypeError as exc:
        if "all_pairs_dist" not in str(exc):
            raise
        return _score_directed_candidate(pos, problem, cluster_ids)


def _proxy_directed_candidate(
    pos: torch.Tensor,
    problem: LayoutProblem,
    cluster_ids: Optional[torch.Tensor],
    all_pairs_dist: Optional[np.ndarray] = None,
) -> float:
    """Return a cheap directed-table proxy score for challenger shortlisting.

    Parameters
    ----------
    pos : torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    problem : LayoutProblem
        Directed acyclic layout problem.
    cluster_ids : torch.Tensor, optional
        Per-node cluster ids with shape ``[N]``.
    all_pairs_dist : Optional[numpy.ndarray], optional
        Cached unweighted shortest paths with shape ``[N, N]``.

    Returns
    -------
    float
        Higher-is-better proxy composite score.
    """
    from dagua.metrics import cluster_silhouette_score, composite_auto, quick

    cpu_pos = pos.detach().to(device="cpu", dtype=torch.float32)
    cpu_edges = problem.edge_index.detach().to(device="cpu")
    numeric = quick(
        cpu_pos,
        cpu_edges,
        node_sizes=(
            None
            if problem.node_sizes is None
            else problem.node_sizes.detach().to(device="cpu", dtype=torch.float32)
        ),
        direction=problem.direction,
        all_pairs_dist=all_pairs_dist,
    )
    if cluster_ids is not None:
        numeric.update(cluster_silhouette_score(cpu_pos, cluster_ids))
    numeric["declared_hierarchical"] = True
    return float(composite_auto(numeric, is_semantically_directed=True))


def _directed_candidate_family(candidate_name: str) -> str:
    """Return the base family for one directed portfolio candidate.

    Parameters
    ----------
    candidate_name : str
        Candidate arm name.

    Returns
    -------
    str
        Family name shared by raw, projected, and convergent cleanup variants.
    """
    if candidate_name.endswith("_raw"):
        return candidate_name[:-4]
    if candidate_name.endswith("_convergent"):
        return candidate_name[: -len("_convergent")]
    return candidate_name


def _directed_layer_work_estimate(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> tuple[int, int]:
    """Estimate dummy expansion and widest rank for directed grid gating.

    Parameters
    ----------
    edge_index : torch.Tensor
        Directed edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    tuple[int, int]
        Estimated dummy-node count for long edges and maximum rank width.
    """
    if num_nodes <= 0 or edge_index.numel() == 0:
        return 0, max(num_nodes, 0)
    edges = [(int(src), int(dst)) for src, dst in edge_index.t().detach().cpu().tolist()]
    outgoing: list[list[int]] = [[] for _ in range(num_nodes)]
    indegree = [0] * num_nodes
    for src, dst in edges:
        if src == dst or src < 0 or dst < 0 or src >= num_nodes or dst >= num_nodes:
            continue
        outgoing[src].append(dst)
        indegree[dst] += 1
    queue = [node for node, degree in enumerate(indegree) if degree == 0]
    ranks = [0] * num_nodes
    cursor = 0
    while cursor < len(queue):
        src = queue[cursor]
        cursor += 1
        for dst in outgoing[src]:
            ranks[dst] = max(ranks[dst], ranks[src] + 1)
            indegree[dst] -= 1
            if indegree[dst] == 0:
                queue.append(dst)
    rank_counts: dict[int, int] = {}
    for rank in ranks:
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    dummy_count = sum(max(ranks[dst] - ranks[src] - 1, 0) for src, dst in edges)
    return int(dummy_count), max(rank_counts.values(), default=0)


def _full_sugiyama_grid_enabled(problem: LayoutProblem, config: LayoutConfig) -> bool:
    """Return whether the expensive Cartesian Sugiyama grid may run.

    Parameters
    ----------
    problem : LayoutProblem
        Directed acyclic layout problem.
    config : LayoutConfig
        Prepared native configuration carrying optional benchmark deadline.

    Returns
    -------
    bool
        ``True`` when graph size and remaining budget allow the full grid.
    """
    n = int(problem.num_nodes)
    if n < DIRECTED_LARGE_NODE_THRESHOLD:
        return True
    dummy_count, max_width = _directed_layer_work_estimate(problem.edge_index, n)
    if dummy_count > DIRECTED_GRID_DUMMY_LIMIT or max_width > DIRECTED_GRID_WIDTH_LIMIT:
        return False
    from dagua.layout.ops.pipelines.native_undirected import _portfolio_has_budget

    return _portfolio_has_budget(config, min_remaining_s=DIRECTED_LARGE_GRID_MIN_REMAINING_S)


def _predicted_arm_budget_available(
    config: LayoutConfig,
    predicted_cost_s: float,
) -> bool:
    """Return whether a predicted-cost arm may start before the deadline.

    Parameters
    ----------
    config : LayoutConfig
        Prepared native configuration carrying optional benchmark deadline.
    predicted_cost_s : float
        Estimated wall-clock seconds for the arm.

    Returns
    -------
    bool
        ``True`` when no deadline is known or remaining budget covers twice
        the predicted cost plus the route's return reserve.
    """
    from dagua.layout.ops.pipelines.native_undirected import (
        ABSOLUTE_DEADLINE_RESERVE_S,
        _portfolio_remaining_s,
    )

    remaining = _portfolio_remaining_s(config)
    if remaining is None:
        return True
    required = (
        DIRECTED_PREDICTED_COST_MULTIPLIER * max(0.0, float(predicted_cost_s))
        + ABSOLUTE_DEADLINE_RESERVE_S
    )
    return float(remaining) > required


def _force_challengers_enabled(edge_index: torch.Tensor, num_nodes: int) -> bool:
    """Return whether skip edges or multiedges justify force challengers.

    Parameters
    ----------
    edge_index : torch.Tensor
        Directed acyclic edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    bool
        Whether the R7 topology gate is open.
    """
    edge_count = int(edge_index.shape[1]) if edge_index.numel() else 0
    if edge_count == 0:
        return False
    edges = [(int(src), int(dst)) for src, dst in edge_index.t().detach().cpu().tolist()]
    if len(set(edges)) < edge_count:
        return True

    outgoing: list[list[int]] = [[] for _ in range(num_nodes)]
    indegree = [0] * num_nodes
    for src, dst in edges:
        if src == dst:
            continue
        outgoing[src].append(dst)
        indegree[dst] += 1
    queue = [node for node, degree in enumerate(indegree) if degree == 0]
    ranks = [0] * num_nodes
    cursor = 0
    while cursor < len(queue):
        src = queue[cursor]
        cursor += 1
        for dst in outgoing[src]:
            ranks[dst] = max(ranks[dst], ranks[src] + 1)
            indegree[dst] -= 1
            if indegree[dst] == 0:
                queue.append(dst)
    long_edges = sum(abs(ranks[dst] - ranks[src]) > 1 for src, dst in edges)
    return float(long_edges) / float(edge_count) > FORCE_SKIP_RATIO_THRESHOLD


def _register_challenger_variants(
    name: str,
    raw_pos: torch.Tensor,
    problem: LayoutProblem,
    config: LayoutConfig,
    positions: Dict[str, torch.Tensor],
    preserve_rank_order: bool = False,
    arm_timings: Optional[Dict[str, Tuple[float, float]]] = None,
    timing_span: Optional[Tuple[float, float]] = None,
) -> None:
    """Register guarded raw and projected variants of one challenger.

    Parameters
    ----------
    name : str
        Candidate family name.
    raw_pos : torch.Tensor
        Unprojected positions with shape ``[N, 2]``.
    problem : LayoutProblem
        Directed layout problem.
    config : LayoutConfig
        Prepared native configuration.
    positions : dict[str, torch.Tensor]
        Finalist registry updated in place.
    preserve_rank_order : bool, default=False
        Whether projected variants must retain the raw candidate's within-rank
        ordering.
    arm_timings : dict[str, tuple[float, float]], optional
        Per-arm timing registry updated when ``timing_span`` is supplied.
    timing_span : tuple[float, float], optional
        ``time.perf_counter()`` start/end span for the candidate family.

    Returns
    -------
    None
        Variants are added to ``positions`` when healthy.
    """
    from dagua.layout.ops.pipelines.native_undirected import (
        _candidate_is_degenerate,
        _project_candidate,
        _repair_flung_isolates,
    )

    node_sep = float(getattr(config, "_dagua_native_node_sep", config.node_sep))
    repaired = _repair_flung_isolates(raw_pos, problem, node_sep)
    projected = _project_candidate(repaired, problem, convergent=False)
    convergent = _project_candidate(repaired, problem, convergent=True)
    if preserve_rank_order:
        projected = _restore_projected_rank_order(repaired, projected)
        convergent = _restore_projected_rank_order(repaired, convergent)
    variants = {
        name + "_raw": repaired,
        name: projected,
        name + "_convergent": convergent,
    }
    for variant_name, candidate in variants.items():
        degenerate, reason = _candidate_is_degenerate(
            candidate,
            problem.node_sizes,
            problem.edge_index,
        )
        if degenerate:
            _LOGGER.info("Rejected directed candidate %s: %s", variant_name, reason)
            continue
        positions[variant_name] = candidate
        if arm_timings is not None and timing_span is not None:
            arm_timings[variant_name] = timing_span


def _restore_projected_rank_order(
    raw_positions: torch.Tensor,
    projected_positions: torch.Tensor,
) -> torch.Tensor:
    """Restore one layered candidate's ordering after overlap projection.

    Parameters
    ----------
    raw_positions : torch.Tensor
        Pre-projection positions with shape ``[N, 2]``.
    projected_positions : torch.Tensor
        Projected positions with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Projected coordinates with each raw rank's x ordering restored while
        retaining the projector's separated x-coordinate multiset.
    """
    out = projected_positions.detach().clone()
    rank_values = sorted({float(value) for value in raw_positions[:, 1].tolist()})
    for rank_y in rank_values:
        nodes = [
            node
            for node in range(int(raw_positions.shape[0]))
            if abs(float(raw_positions[node, 1].item()) - rank_y) <= 1.0e-6
        ]
        if len(nodes) < 2:
            continue
        ordered_nodes = sorted(nodes, key=lambda node: float(raw_positions[node, 0].item()))
        ordered_x = sorted(float(out[node, 0].item()) for node in nodes)
        for node, x_value in zip(ordered_nodes, ordered_x):
            out[node, 0] = x_value
    return out


def _directed_rank_profile(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> tuple[list[int], int, float]:
    """Return longest-path ranks, widest rank, and long-edge ratio.

    Parameters
    ----------
    edge_index : torch.Tensor
        Directed edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    tuple[list[int], int, float]
        Per-node ranks, maximum rank width, and fraction of edges spanning
        more than one rank.
    """
    if num_nodes <= 0:
        return [], 0, 0.0
    from dagua.utils import longest_path_layering

    ranks_raw = longest_path_layering(edge_index.detach().to(device="cpu"), num_nodes)
    rank_values = ranks_raw.tolist() if hasattr(ranks_raw, "tolist") else ranks_raw
    ranks = [int(value) for value in rank_values]
    rank_counts: dict[int, int] = {}
    for rank in ranks:
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    edge_count = int(edge_index.shape[1]) if edge_index.numel() else 0
    if edge_count == 0:
        return ranks, max(rank_counts.values(), default=0), 0.0
    long_edges = 0
    for src, dst in edge_index.t().detach().cpu().tolist():
        if 0 <= int(src) < num_nodes and 0 <= int(dst) < num_nodes:
            long_edges += int(abs(ranks[int(dst)] - ranks[int(src)]) > 1)
    return ranks, max(rank_counts.values(), default=0), float(long_edges) / float(edge_count)


def _normalize_seed_to_point_units(
    positions: torch.Tensor,
    node_sizes: Optional[torch.Tensor],
    node_sep: float,
) -> torch.Tensor:
    """Normalize arbitrary seed coordinates into node-size point units.

    Parameters
    ----------
    positions : torch.Tensor
        Raw candidate coordinates with shape ``[N, 2]``.
    node_sizes : torch.Tensor, optional
        Node boxes with shape ``[N, 2]``.
    node_sep : float
        Fallback node separation in points.

    Returns
    -------
    torch.Tensor
        Centered and uniformly scaled coordinates with shape ``[N, 2]``.
    """
    pos = positions.detach().to(device="cpu", dtype=torch.float32).clone()
    if pos.numel() == 0:
        return pos
    centered = pos - pos.mean(dim=0, keepdim=True)
    span = float((centered.max(dim=0).values - centered.min(dim=0).values).max().item())
    if not np.isfinite(span) or span <= 1.0e-6:
        return torch.zeros_like(centered)
    if node_sizes is not None and node_sizes.numel() > 0:
        size_scale = float(
            torch.linalg.vector_norm(node_sizes.detach().to(dtype=torch.float32), dim=1)
            .mean()
            .item()
        )
    else:
        size_scale = float(node_sep)
    target_span = max(size_scale, float(node_sep), 1.0) * max(1.0, np.sqrt(float(pos.shape[0])))
    return centered * (target_span / span)


def _align_to_incumbent(candidate: torch.Tensor, incumbent: torch.Tensor) -> torch.Tensor:
    """Orthogonally align a seed to the incumbent scale and centroid.

    Parameters
    ----------
    candidate : torch.Tensor
        Candidate coordinates with shape ``[N, 2]``.
    incumbent : torch.Tensor
        Incumbent coordinates with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Candidate transformed into the incumbent coordinate frame.
    """
    cand = candidate.detach().to(device="cpu", dtype=torch.float32)
    ref = incumbent.detach().to(device="cpu", dtype=torch.float32)
    if cand.shape != ref.shape or cand.shape[0] <= 1:
        return cand.clone()
    cand_centered = cand - cand.mean(dim=0, keepdim=True)
    ref_centered = ref - ref.mean(dim=0, keepdim=True)
    try:
        u, _, vh = torch.linalg.svd(cand_centered.t().mm(ref_centered))
        rotation = u.mm(vh)
    except RuntimeError:
        rotation = torch.eye(2, dtype=torch.float32)
    rotated = cand_centered.mm(rotation)
    cand_norm = float(torch.linalg.vector_norm(rotated).item())
    ref_norm = float(torch.linalg.vector_norm(ref_centered).item())
    if cand_norm > 1.0e-6 and ref_norm > 1.0e-6:
        rotated = rotated * (ref_norm / cand_norm)
    return rotated + ref.mean(dim=0, keepdim=True)


def _directed_pivot_mds_candidates(
    problem: LayoutProblem,
    incumbent: torch.Tensor,
    node_sep: float,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Build narrow PivotMDS directed challengers.

    Parameters
    ----------
    problem : LayoutProblem
        Directed layout problem.
    incumbent : torch.Tensor
        Full-scored incumbent positions with shape ``[N, 2]``.
    node_sep : float
        Node separation in points for normalization.
    seed : int
        Deterministic seed.

    Returns
    -------
    dict[str, torch.Tensor]
        Candidate family names mapped to raw positions.
    """
    n = int(problem.num_nodes)
    if n <= 2 or n > DIRECTED_NARROW_SEED_NODE_CAP:
        return {}
    edge_count = int(problem.edge_index.shape[1]) if problem.edge_index.numel() else 0
    if edge_count == 0 or edge_count > DIRECTED_NARROW_SEED_EDGE_CAP:
        return {}
    from dagua.layout.ops.pipelines.nnpnet_reference import (
        canonical_adjacency,
        reference_pmds,
    )

    cpu_edges = problem.edge_index.detach().to(device="cpu", dtype=torch.long).numpy()
    adjacency = canonical_adjacency(cpu_edges, n)
    raw_np = reference_pmds(adjacency, dims=2, pivots=min(64, n), seed=seed)
    raw = torch.from_numpy(raw_np).to(dtype=torch.float32)
    normalized = _normalize_seed_to_point_units(raw, problem.node_sizes, node_sep)
    aligned = _align_to_incumbent(normalized, incumbent)
    rotated = torch.stack([-normalized[:, 1], normalized[:, 0]], dim=1)
    rotated = _align_to_incumbent(rotated, incumbent)
    scaled = _align_to_incumbent(normalized * 0.85, incumbent)
    flow_blend = incumbent.detach().to(device="cpu", dtype=torch.float32) * 0.85 + aligned * 0.15
    return {
        "pivot_mds": aligned,
        "pivot_mds_rot90": rotated,
        "pivot_mds_scale085": scaled,
        "pivot_mds_flow_blend": flow_blend,
    }


def _directed_mrtree_enabled(problem: LayoutProblem) -> bool:
    """Return whether the DAG is a narrow MrTree exception candidate.

    Parameters
    ----------
    problem : LayoutProblem
        Directed layout problem.

    Returns
    -------
    bool
        ``True`` for small tree-like or long-skip DAGs.
    """
    n = int(problem.num_nodes)
    edge_count = int(problem.edge_index.shape[1]) if problem.edge_index.numel() else 0
    if n <= 2 or n > DIRECTED_NARROW_SEED_NODE_CAP or edge_count == 0:
        return False
    _, max_width, long_edge_ratio = _directed_rank_profile(problem.edge_index, n)
    return (
        float(edge_count) <= DIRECTED_MRTREE_EDGE_NODE_RATIO_MAX * float(n)
        and max_width <= DIRECTED_MRTREE_MAX_RANK_WIDTH
        and (
            long_edge_ratio >= 0.12
            or edge_count <= max(n, 2)
            or (max_width <= 3 and edge_count <= 2 * max(n, 2))
        )
    )


def _directed_stress_blend_candidates(
    problem: LayoutProblem,
    incumbent: torch.Tensor,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Build affine-aligned native-stress blend seed challengers.

    Parameters
    ----------
    problem : LayoutProblem
        Directed layout problem.
    incumbent : torch.Tensor
        Full-scored incumbent positions with shape ``[N, 2]``.
    seed : int
        Deterministic seed.

    Returns
    -------
    dict[str, torch.Tensor]
        Stress-blend family names mapped to raw positions.
    """
    n = int(problem.num_nodes)
    edge_count = int(problem.edge_index.shape[1]) if problem.edge_index.numel() else 0
    if n <= 2 or n > DIRECTED_NARROW_SEED_NODE_CAP or edge_count > DIRECTED_NARROW_SEED_EDGE_CAP:
        return {}
    from dagua.layout.ops.pipelines.native_stress import (
        NativeStressConfig,
        layout_native_stress_pipeline,
    )

    stress = layout_native_stress_pipeline(
        edge_index=problem.edge_index.detach().to(device="cpu"),
        num_nodes=n,
        node_sizes=(
            None
            if problem.node_sizes is None
            else problem.node_sizes.detach().to(device="cpu", dtype=torch.float32)
        ),
        edge_weights=(
            None
            if problem.edge_weights is None
            else problem.edge_weights.detach().to(device="cpu", dtype=torch.float32)
        ),
        seed=seed,
        config=NativeStressConfig(steps=40, late_steps=8, seed=seed, target_unit="points"),
    )
    aligned = _align_to_incumbent(stress, incumbent)
    incumbent_cpu = incumbent.detach().to(device="cpu", dtype=torch.float32)
    return {
        f"stress_blend_{weight:g}": incumbent_cpu * (1.0 - weight) + aligned * weight
        for weight in DIRECTED_STRESS_BLEND_WEIGHTS
    }


def _segments_cross(
    first_start: torch.Tensor,
    first_end: torch.Tensor,
    second_start: torch.Tensor,
    second_end: torch.Tensor,
) -> bool:
    """Return whether two line segments cross at their interiors.

    Parameters
    ----------
    first_start : torch.Tensor
        First segment start coordinate with shape ``[2]``.
    first_end : torch.Tensor
        First segment end coordinate with shape ``[2]``.
    second_start : torch.Tensor
        Second segment start coordinate with shape ``[2]``.
    second_end : torch.Tensor
        Second segment end coordinate with shape ``[2]``.

    Returns
    -------
    bool
        ``True`` when the two segments strictly cross.
    """

    def _orient(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> float:
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    o1 = _orient(first_start, first_end, second_start)
    o2 = _orient(first_start, first_end, second_end)
    o3 = _orient(second_start, second_end, first_start)
    o4 = _orient(second_start, second_end, first_end)
    return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)


def _exact_crossing_count(pos: torch.Tensor, edge_index: torch.Tensor) -> int:
    """Count exact non-incident straight-line edge crossings.

    Parameters
    ----------
    pos : torch.Tensor
        Positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.

    Returns
    -------
    int
        Number of crossing non-incident edge pairs.
    """
    edges = [(int(src), int(dst)) for src, dst in edge_index.t().detach().cpu().tolist()]
    crossings = 0
    for left_idx, (src_a, dst_a) in enumerate(edges):
        for src_b, dst_b in edges[left_idx + 1 :]:
            if len({src_a, dst_a, src_b, dst_b}) < 4:
                continue
            crossings += int(_segments_cross(pos[src_a], pos[dst_a], pos[src_b], pos[dst_b]))
    return crossings


def _rank_local_zero_crossing_swap_candidate(
    incumbent: torch.Tensor,
    edge_index: torch.Tensor,
    max_passes: int = 3,
) -> torch.Tensor:
    """Greedily swap within-rank x positions when crossings strictly drop.

    Parameters
    ----------
    incumbent : torch.Tensor
        Incumbent positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Directed edge tensor with shape ``[2, E]``.
    max_passes : int, default=3
        Maximum adjacent-swap sweeps.

    Returns
    -------
    torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    """
    pos = incumbent.detach().to(device="cpu", dtype=torch.float32).clone()
    n = int(pos.shape[0])
    if n <= 2 or n > DIRECTED_NARROW_SEED_NODE_CAP:
        return pos
    ranks, max_width, _ = _directed_rank_profile(edge_index, n)
    if max_width < 2 or max_width > 32:
        return pos
    rank_to_nodes: dict[int, list[int]] = {}
    for node, rank in enumerate(ranks):
        rank_to_nodes.setdefault(rank, []).append(node)
    best_crossings = _exact_crossing_count(pos, edge_index)
    for _pass in range(max(0, int(max_passes))):
        changed = False
        for nodes in rank_to_nodes.values():
            ordered = sorted(nodes, key=lambda node: float(pos[node, 0].item()))
            for left, right in zip(ordered, ordered[1:]):
                candidate = pos.clone()
                candidate[left, 0], candidate[right, 0] = pos[right, 0], pos[left, 0]
                crossings = _exact_crossing_count(candidate, edge_index)
                if crossings < best_crossings:
                    pos = candidate
                    best_crossings = crossings
                    changed = True
        if not changed or best_crossings == 0:
            break
    return pos


def layout_native_directed_portfolio(
    problem: LayoutProblem,
    state: SolveState,
    ctx: RuntimeContext,
    config: LayoutConfig,
) -> torch.Tensor:
    """Run the directed candidate contest and return its monotone winner.

    Parameters
    ----------
    problem : LayoutProblem
        Prepared declared-hierarchical problem.
    state : SolveState
        Incoming solve state.
    ctx : RuntimeContext
        Shared runtime context.
    config : LayoutConfig
        Prepared native configuration.

    Returns
    -------
    torch.Tensor
        Winning positions with shape ``[N, 2]``.
    """
    from dagua.layout.ops.pipelines.dagua_native import _run_native_problem
    from dagua.layout.ops.pipelines.native_undirected import (
        _build_cluster_ids,
        _log_marketplace_telemetry,
        _portfolio_has_budget,
        _portfolio_remaining_s,
        _reraise_worker_timeout,
    )

    started = time.perf_counter()
    incumbent_config = copy.copy(config)
    setattr(incumbent_config, "_dagua_native_suppress_portfolio", True)
    incumbent_state = SolveState(pos=None if state.pos is None else state.pos.detach().clone())
    arm_timings: Dict[str, Tuple[float, float]] = {}
    incumbent_started = time.perf_counter()
    incumbent = _run_native_problem(problem, incumbent_state, ctx, incumbent_config)
    arm_timings["incumbent"] = (incumbent_started, time.perf_counter())
    n = int(problem.num_nodes)

    positions: Dict[str, torch.Tensor] = {"incumbent": incumbent}
    seed = int(problem.seed) if problem.seed is not None else 42
    # Fidelity adapters were validated against CPU reference implementations;
    # several intentionally use NumPy internally. Keep challenger inputs on
    # CPU even when the incumbent native solve used CUDA.
    cpu_edges = problem.edge_index.detach().to(device="cpu")
    cpu_sizes = (
        None
        if problem.node_sizes is None
        else problem.node_sizes.detach().to(device="cpu", dtype=torch.float32)
    )
    cpu_weights = (
        None
        if problem.edge_weights is None
        else problem.edge_weights.detach().to(device="cpu", dtype=torch.float32)
    )
    from dagua.metrics import _all_pairs_unweighted, _build_csr

    offsets, targets = _build_csr(cpu_edges, n)
    all_pairs_dist = _all_pairs_unweighted(offsets, targets, n, max_dist=n)
    cluster_ids = _build_cluster_ids(problem)
    scores: Dict[str, float] = {
        "incumbent": _score_directed_candidate_cached(
            incumbent,
            problem,
            cluster_ids,
            all_pairs_dist,
        )
    }
    if n > MAX_DIRECTED_CONTEST_NODES:
        _LOGGER.info(
            "Directed contest gate=incumbent_only n=%d incumbent_score=%.3f wall_time_s=%.3f",
            n,
            scores["incumbent"],
            time.perf_counter() - started,
        )
        return incumbent
    if not _portfolio_has_budget(config):
        _LOGGER.info(
            "Directed marketplace budget exhausted after incumbent n=%d score=%.3f remaining_s=%s",
            n,
            scores["incumbent"],
            _portfolio_remaining_s(config),
        )
        return incumbent
    if _portfolio_has_budget(config, min_remaining_s=2.0):
        try:
            node_sep = float(getattr(config, "_dagua_native_node_sep", config.node_sep))
            candidate_started = time.perf_counter()
            for name, candidate in _directed_pivot_mds_candidates(
                problem,
                incumbent,
                node_sep,
                seed,
            ).items():
                _register_challenger_variants(
                    name,
                    candidate,
                    problem,
                    config,
                    positions,
                    arm_timings=arm_timings,
                    timing_span=(candidate_started, time.perf_counter()),
                )
        except Exception as exc:  # noqa: BLE001 -- challengers cannot sink the incumbent
            _reraise_worker_timeout(exc)
            _LOGGER.warning("directed PivotMDS challenger failed", exc_info=True)
    if _portfolio_has_budget(config, min_remaining_s=2.0) and _directed_mrtree_enabled(problem):
        try:
            from dagua.layout.ops.pipelines.elk_mrtree import layout_elk_mrtree_pipeline

            candidate_started = time.perf_counter()
            candidate = layout_elk_mrtree_pipeline(
                edge_index=cpu_edges,
                num_nodes=n,
                node_sizes=cpu_sizes,
                seed=seed,
                edge_weights=cpu_weights,
                fidelity_dtype=torch.float32,
            )
            _register_challenger_variants(
                "elk_mrtree",
                candidate,
                problem,
                config,
                positions,
                preserve_rank_order=True,
                arm_timings=arm_timings,
                timing_span=(candidate_started, time.perf_counter()),
            )
        except Exception as exc:  # noqa: BLE001 -- challengers cannot sink the incumbent
            _reraise_worker_timeout(exc)
            _LOGGER.warning("directed ELK MrTree challenger failed", exc_info=True)
    if _portfolio_has_budget(config, min_remaining_s=2.0):
        try:
            candidate_started = time.perf_counter()
            for name, candidate in _directed_stress_blend_candidates(
                problem,
                incumbent,
                seed,
            ).items():
                _register_challenger_variants(
                    name,
                    candidate,
                    problem,
                    config,
                    positions,
                    arm_timings=arm_timings,
                    timing_span=(candidate_started, time.perf_counter()),
                )
        except Exception as exc:  # noqa: BLE001 -- challengers cannot sink the incumbent
            _reraise_worker_timeout(exc)
            _LOGGER.warning("directed stress-blend challenger failed", exc_info=True)
    if _portfolio_has_budget(config, min_remaining_s=2.0):
        edge_count = int(problem.edge_index.shape[1]) if problem.edge_index.numel() else 0
        if n <= DIRECTED_NARROW_SEED_NODE_CAP and edge_count <= DIRECTED_NARROW_SEED_EDGE_CAP:
            try:
                candidate_started = time.perf_counter()
                candidate = _rank_local_zero_crossing_swap_candidate(
                    incumbent,
                    problem.edge_index,
                )
                if not torch.equal(candidate, incumbent.detach().to(device="cpu")):
                    _register_challenger_variants(
                        "rank_local_zero_crossing_swap",
                        candidate,
                        problem,
                        config,
                        positions,
                        preserve_rank_order=True,
                        arm_timings=arm_timings,
                        timing_span=(candidate_started, time.perf_counter()),
                    )
            except Exception as exc:  # noqa: BLE001 -- challengers cannot sink the incumbent
                _reraise_worker_timeout(exc)
                _LOGGER.warning("directed rank-local swap challenger failed", exc_info=True)
    if _portfolio_has_budget(config):
        try:
            from dagua.layout.ops.pipelines.sugiyama import layout_sugiyama_pipeline

            candidate_started = time.perf_counter()
            if not _predicted_arm_budget_available(config, DIRECTED_SUGIYAMA_SIMPLEX_PRIOR_S):
                _LOGGER.info("Skipped directed cluster dot-x: insufficient predicted budget")
            else:
                corrected_cluster_dot_x = layout_sugiyama_pipeline(
                    edge_index=cpu_edges,
                    num_nodes=n,
                    node_sizes=cpu_sizes,
                    rank_sep=SCALED_SUGIYAMA_RANK_SEP,
                    node_sep=SCALED_SUGIYAMA_NODE_SEP,
                    seed=seed,
                    edge_weights=cpu_weights,
                    fidelity_mode="graphviz",
                    clusters=problem.clusters,
                    cluster_parents=problem.cluster_parents,
                    graphviz_apply_cluster_constraints=True,
                    graphviz_corrected_dot_x=True,
                )
                if not isinstance(corrected_cluster_dot_x, torch.Tensor):
                    raise RuntimeError(
                        "corrected cluster dot-x Sugiyama returned non-position output"
                    )
                _register_challenger_variants(
                    "graphviz_dotx_cluster_corrected",
                    corrected_cluster_dot_x,
                    problem,
                    config,
                    positions,
                    preserve_rank_order=True,
                    arm_timings=arm_timings,
                    timing_span=(candidate_started, time.perf_counter()),
                )
                cluster_cost_s = max(0.0, time.perf_counter() - candidate_started)
                run_remaining_sugiyama = True
                if not _predicted_arm_budget_available(config, cluster_cost_s):
                    _LOGGER.info("Skipped directed point-unit dot-x: insufficient predicted budget")
                    sibling_cost_s = cluster_cost_s
                    run_remaining_sugiyama = False
                else:
                    candidate_started = time.perf_counter()
                    point_unit_dot_x = layout_sugiyama_pipeline(
                        edge_index=cpu_edges,
                        num_nodes=n,
                        node_sizes=cpu_sizes,
                        rank_sep=SCALED_SUGIYAMA_RANK_SEP,
                        node_sep=SCALED_SUGIYAMA_NODE_SEP,
                        seed=seed,
                        edge_weights=cpu_weights,
                        fidelity_mode="graphviz",
                        clusters=problem.clusters,
                        cluster_parents=problem.cluster_parents,
                        graphviz_preserve_point_units=True,
                    )
                    if not isinstance(point_unit_dot_x, torch.Tensor):
                        raise RuntimeError("point-unit dot-x Sugiyama returned non-position output")
                    _register_challenger_variants(
                        "graphviz_dotx_point_units",
                        point_unit_dot_x,
                        problem,
                        config,
                        positions,
                        arm_timings=arm_timings,
                        timing_span=(candidate_started, time.perf_counter()),
                    )

                    sibling_cost_s = max(0.0, time.perf_counter() - candidate_started)
                    for mode in ("graphviz_dot", "igraph"):
                        if not _portfolio_has_budget(config) or not _predicted_arm_budget_available(
                            config,
                            sibling_cost_s,
                        ):
                            break
                        candidate_started = time.perf_counter()
                        candidate = layout_sugiyama_pipeline(
                            edge_index=cpu_edges,
                            num_nodes=n,
                            node_sizes=cpu_sizes,
                            seed=seed,
                            edge_weights=cpu_weights,
                            fidelity_mode=mode,
                            clusters=problem.clusters,
                            cluster_parents=problem.cluster_parents,
                        )
                        if not isinstance(candidate, torch.Tensor):
                            raise RuntimeError(f"{mode} Sugiyama returned non-position output")
                        _register_challenger_variants(
                            mode,
                            candidate,
                            problem,
                            config,
                            positions,
                            arm_timings=arm_timings,
                            timing_span=(candidate_started, time.perf_counter()),
                        )
                        sibling_cost_s = max(0.0, time.perf_counter() - candidate_started)
                if run_remaining_sugiyama and _full_sugiyama_grid_enabled(problem, config):
                    # The full spacing grid remains exact for small DAGs. At
                    # n>=250 it runs only when structural expansion and remaining
                    # budget leave enough space for the already-returnable incumbent.
                    for mode in SUGIYAMA_FIDELITY_MODES:
                        for rank_sep in SUGIYAMA_RANK_SEP_GRID:
                            for node_sep in SUGIYAMA_NODE_SEP_GRID:
                                if not _portfolio_has_budget(
                                    config
                                ) or not _predicted_arm_budget_available(config, sibling_cost_s):
                                    break
                                candidate_started = time.perf_counter()
                                candidate = layout_sugiyama_pipeline(
                                    edge_index=cpu_edges,
                                    num_nodes=n,
                                    node_sizes=cpu_sizes,
                                    rank_sep=rank_sep,
                                    node_sep=node_sep,
                                    seed=seed,
                                    edge_weights=cpu_weights,
                                    fidelity_mode=mode,
                                    clusters=problem.clusters,
                                    cluster_parents=problem.cluster_parents,
                                )
                                grid_name = f"{mode}_r{rank_sep:g}_n{node_sep:g}"
                                if not isinstance(candidate, torch.Tensor):
                                    raise RuntimeError(
                                        f"{grid_name} Sugiyama returned non-position output"
                                    )
                                if mode == "igraph":
                                    # Match the reference adapter's fixed conversion from
                                    # igraph coordinate units into renderer point units.
                                    candidate = candidate * IGRAPH_OUTPUT_SCALE
                                _register_challenger_variants(
                                    grid_name,
                                    candidate,
                                    problem,
                                    config,
                                    positions,
                                    arm_timings=arm_timings,
                                    timing_span=(candidate_started, time.perf_counter()),
                                )
                                sibling_cost_s = max(0.0, time.perf_counter() - candidate_started)
        except Exception as exc:  # noqa: BLE001 -- challengers cannot sink the incumbent
            _reraise_worker_timeout(exc)
            _LOGGER.warning("directed Sugiyama challenger failed", exc_info=True)

    force_gate = _force_challengers_enabled(problem.edge_index, n)
    if force_gate:
        try:
            from dagua.layout.ops.pipelines.fcose import layout_fcose_pipeline
            from dagua.layout.ops.pipelines.native_undirected import FCOSE_CONTEST_SEEDS

            force_cost_s = DIRECTED_FORCE_PRIOR_S
            for seed_offset in range(FCOSE_CONTEST_SEEDS):
                if not _portfolio_has_budget(config) or (
                    n >= 120 and not _predicted_arm_budget_available(config, force_cost_s)
                ):
                    break
                candidate_started = time.perf_counter()
                candidate = layout_fcose_pipeline(
                    edge_index=cpu_edges,
                    num_nodes=n,
                    node_sizes=cpu_sizes,
                    seed=42 + seed_offset,
                    edge_weights=cpu_weights,
                )
                _register_challenger_variants(
                    f"fcose_seed{seed_offset}",
                    candidate,
                    problem,
                    config,
                    positions,
                    arm_timings=arm_timings,
                    timing_span=(candidate_started, time.perf_counter()),
                )
                force_cost_s = max(0.0, time.perf_counter() - candidate_started)
        except Exception as exc:  # noqa: BLE001
            _reraise_worker_timeout(exc)
            _LOGGER.warning("directed fCoSE challenger failed", exc_info=True)
        try:
            if not _portfolio_has_budget(config) or (
                n >= 120 and not _predicted_arm_budget_available(config, DIRECTED_FORCE_PRIOR_S)
            ):
                _LOGGER.info("Skipped directed YifanHu: insufficient predicted budget")
            else:
                from dagua.layout.ops.pipelines.yifanhu import layout_yifanhu_pipeline

                candidate_started = time.perf_counter()
                candidate = layout_yifanhu_pipeline(
                    edge_index=cpu_edges,
                    num_nodes=n,
                    node_sizes=cpu_sizes,
                    seed=123,
                    edge_weights=cpu_weights,
                    direction=problem.direction,
                )
                _register_challenger_variants(
                    "yifanhu",
                    candidate,
                    problem,
                    config,
                    positions,
                    arm_timings=arm_timings,
                    timing_span=(candidate_started, time.perf_counter()),
                )
        except Exception as exc:  # noqa: BLE001
            _reraise_worker_timeout(exc)
            _LOGGER.warning("directed YifanHu challenger failed", exc_info=True)

    proxy_scores = {
        name: _proxy_directed_candidate(candidate, problem, cluster_ids, all_pairs_dist)
        for name, candidate in positions.items()
    }
    challenger_names = sorted(
        (name for name in positions if name != "incumbent"),
        key=lambda name: (-proxy_scores[name], name),
    )
    if n < DIRECTED_LARGE_NODE_THRESHOLD:
        finalist_names = ["incumbent", *challenger_names]
    else:
        admitted_families: list[str] = []
        for name in challenger_names:
            family = _directed_candidate_family(name)
            if family in admitted_families:
                continue
            admitted_families.append(family)
            if len(admitted_families) >= DIRECTED_FULL_REFEREE_TOP_K:
                break
        finalist_names = ["incumbent"]
        finalist_names.extend(
            name
            for name in challenger_names
            if _directed_candidate_family(name) in admitted_families
        )
    for name in finalist_names:
        if name == "incumbent" or name in scores:
            continue
        if not _portfolio_has_budget(config, min_remaining_s=DIRECTED_FULL_SCORE_MIN_REMAINING_S):
            _LOGGER.info("Skipped directed full score for %s: insufficient return reserve", name)
            continue
        scores[name] = _score_directed_candidate_cached(
            positions[name],
            problem,
            cluster_ids,
            all_pairs_dist,
        )
    best_name = "incumbent"
    for name, score in scores.items():
        if name != "incumbent" and score > scores[best_name]:
            best_name = name
    _log_marketplace_telemetry(
        route="directed",
        structural_gate="force" if force_gate else "layered",
        positions=positions,
        proxy_scores=proxy_scores,
        full_scores=scores,
        finalist_names=list(scores),
        winner_name=best_name,
        started_at=started,
        arm_timings=arm_timings,
    )
    _LOGGER.info(
        "Directed contest gate=%s candidates=%s winner=%s wall_time_s=%.3f",
        "force" if force_gate else "layered",
        ", ".join(f"{name}:{score:.3f}" for name, score in scores.items()),
        best_name,
        time.perf_counter() - started,
    )
    return positions[best_name].to(device=incumbent.device, dtype=incumbent.dtype)


@dataclass(frozen=True)
class DirectedPortfolioRouteConfig:
    """Configuration wrapper for the directed portfolio route."""

    layout_config: Optional[LayoutConfig] = None


@register_op
@dataclass(frozen=True)
class DirectedPortfolioRoute(Op):
    """Run the declared-hierarchical candidate contest."""

    config: DirectedPortfolioRouteConfig = field(default_factory=DirectedPortfolioRouteConfig)
    name: ClassVar[str] = "directed_portfolio_route"
    category: ClassVar[OpCategory] = OpCategory.CONTROL
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run the contest and update the solve state.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime infrastructure.

        Returns
        -------
        SolveState
            State containing the winning positions.
        """
        state.pos = layout_native_directed_portfolio(
            problem,
            state,
            ctx,
            self.config.layout_config or LayoutConfig(),
        )
        return state


def build_native_directed_portfolio_pipeline(config: LayoutConfig) -> Pipeline:
    """Build the directed portfolio as a registered one-op pipeline.

    Parameters
    ----------
    config : LayoutConfig
        Prepared native configuration.

    Returns
    -------
    Pipeline
        Directed portfolio pipeline.
    """
    return Pipeline(
        [DirectedPortfolioRoute(DirectedPortfolioRouteConfig(layout_config=config))],
        name="native_directed_portfolio",
    )


__all__ = [
    "MAX_DIRECTED_CONTEST_NODES",
    "DirectedPortfolioRoute",
    "DirectedPortfolioRouteConfig",
    "build_native_directed_portfolio_pipeline",
    "layout_native_directed_portfolio",
]
