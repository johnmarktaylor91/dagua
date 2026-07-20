"""Honest directed-table portfolio for declared-hierarchical DAGs."""

from __future__ import annotations

import copy
import logging
import math
import time
from dataclasses import dataclass, field
from itertools import permutations
from typing import TYPE_CHECKING, ClassVar, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from dagua.config import LayoutConfig
from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.pipelines.native_budget import admit_native_work
from dagua.layout.ops.pipelines.native_cost_model import NativeWorkCost, estimate_native_work_cost
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

if TYPE_CHECKING:
    from dagua.layout.ops.pipelines.native_finisher import W5HonestAxes

MAX_DIRECTED_CONTEST_NODES = 2000
DIRECTED_FULL_REFEREE_TOP_K = 6
CLUSTER_EXTENDED_SCORE_KEYS = (
    "cluster_exclusion_score",
    "cluster_sibling_overlap_score",
    "cluster_nesting_fidelity_score",
    "cluster_edge_intrusion_score",
    "cluster_label_occlusion_score",
    "cluster_compactness_score",
)
CLUSTER_DUAL_ACCEPTANCE_MARGIN = 0.05
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
DIRECTED_ORDERING_PORTFOLIO_SMALL_NODE_CAP = 64
DIRECTED_ORDERING_MEDIUM_NODE_CAP = 500
DIRECTED_ORDERING_EXHAUSTIVE_WIDTH_CAP = 8
DIRECTED_ORDERING_EXHAUSTIVE_PERM_CAP = 50_000
DIRECTED_ORDERING_EXHAUSTIVE_PER_RANK_PERM_CAP = 720
DIRECTED_ORDERING_DEADLINE_CHECK_INTERVAL = 16
DIRECTED_ORDERING_SMALL_WALL_TIME_CAP_S = 1.5
DIRECTED_ORDERING_MEDIUM_WALL_TIME_CAP_S = 2.5
DIRECTED_ORDERING_MEDIUM_EDGE_PAIR_CAP = 250_000
DIRECTED_ORDERING_TRIAL_PAIR_CAP = 5_000_000
DIRECTED_ORDERING_Y_TOLERANCE = 1.0e-4
DIRECTED_ORDERING_NUDGE_CROSSING_CAP = 64
DIRECTED_ORDERING_NUDGE_TRIAL_CAP = 256
DIRECTED_ORDERING_PAIR_BUDGET_CHECK_INTERVAL = 2_000
DIRECTED_ORDERING_W5_NODE_CAP = 32
DIRECTED_RECOMBINANT_MIN_NODES = 80
DIRECTED_RECOMBINANT_MAX_NODES = 600
DIRECTED_RECOMBINANT_MAX_CANDIDATES = 6
DIRECTED_RECOMBINANT_PRIOR_S = 15.0
DIRECTED_MRTREE_EDGE_NODE_RATIO_MAX = 3.0
DIRECTED_MRTREE_MAX_RANK_WIDTH = 6
DIRECTED_STRESS_BLEND_WEIGHTS = (0.2, 0.4)
SCALED_SUGIYAMA_RANK_SEP = 72.0
SCALED_SUGIYAMA_NODE_SEP = 18.0
EXACT_CROSSING_COUNT_VECTOR_PAIR_CAP = 5_000_000
SUGIYAMA_FIDELITY_MODES = ("graphviz_dot", "graphviz", "igraph")
SUGIYAMA_RANK_SEP_GRID = (36.0, 72.0, 108.0)
SUGIYAMA_NODE_SEP_GRID = (18.0, 36.0, 54.0)
IGRAPH_OUTPUT_SCALE = 50.0
_LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dagua.layout.ops.pipelines.native_finisher import W5ScorePair


@dataclass(frozen=True)
class _RecombinantLayeredSpec:
    """One bounded layered-stage cross for the directed portfolio."""

    name: str
    layering: str
    ordering: str
    xcoord: str


@dataclass(frozen=True)
class _DirectedClusterScoreTelemetry:
    """Old and extended directed composites from one full-ruler metric pass.

    Attributes
    ----------
    extended_score : float
        Directed composite including the six R8 cluster-quality terms.
    old_score : float
        Directed composite after removing only those six terms.
    metrics : dict[str, float]
        Numeric metric payload used for scoring.
    """

    extended_score: float
    old_score: float
    metrics: Dict[str, float]


def _old_cluster_ruler_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    """Return metrics with only the six R8 cluster-quality terms removed.

    Parameters
    ----------
    metrics : dict[str, float]
        Numeric metrics from ``dagua.metrics.full``.

    Returns
    -------
    dict[str, float]
        Copy of ``metrics`` with the extended cluster-quality keys omitted.
    """
    return {key: value for key, value in metrics.items() if key not in CLUSTER_EXTENDED_SCORE_KEYS}


def _directed_cluster_candidate_is_dual_admissible(
    candidate: _DirectedClusterScoreTelemetry,
    incumbent: _DirectedClusterScoreTelemetry,
) -> bool:
    """Return whether a clustered directed challenger is admissible.

    Parameters
    ----------
    candidate : _DirectedClusterScoreTelemetry
        Candidate extended and old-ruler scores.
    incumbent : _DirectedClusterScoreTelemetry
        Incumbent extended and old-ruler scores.

    Returns
    -------
    bool
        ``True`` iff extended improves by the honest margin and old-ruler
        score does not decrease.
    """
    return (
        candidate.extended_score > incumbent.extended_score + CLUSTER_DUAL_ACCEPTANCE_MARGIN
        and candidate.old_score >= incumbent.old_score
    )


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
    return _score_directed_candidate_referee_payload(
        pos,
        problem,
        cluster_ids,
        all_pairs_dist,
    )[0]


def _score_directed_candidate_referee_payload(
    pos: torch.Tensor,
    problem: LayoutProblem,
    cluster_ids: Optional[torch.Tensor],
    all_pairs_dist: Optional[np.ndarray] = None,
) -> tuple[float, Optional[_DirectedClusterScoreTelemetry]]:
    """Score a directed finalist and return clustered-ruler telemetry.

    Parameters
    ----------
    pos : torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    problem : LayoutProblem
        Directed acyclic layout problem with optional cluster and label
        metadata.
    cluster_ids : torch.Tensor, optional
        Per-node cluster ids with shape ``[N]``.
    all_pairs_dist : Optional[numpy.ndarray], optional
        Cached unweighted shortest paths with shape ``[N, N]``.

    Returns
    -------
    tuple[float, _DirectedClusterScoreTelemetry | None]
        Extended directed score and optional old/extended telemetry for
        clustered rows.
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
        label_positions=problem.label_positions,
        edge_labels=problem.edge_labels,
        all_pairs_dist=all_pairs_dist,
        clusters=problem.clusters,
        cluster_parents=problem.cluster_parents,
        cluster_labels=problem.cluster_labels,
    )
    numeric["declared_hierarchical"] = True
    numeric_float = {
        key: float(value) for key, value in numeric.items() if isinstance(value, (int, float))
    }
    score = float(composite_auto(numeric_float, is_semantically_directed=True))
    old_score = float(
        composite_auto(_old_cluster_ruler_metrics(numeric_float), is_semantically_directed=True)
    )
    telemetry = None
    if problem.clusters:
        telemetry = _DirectedClusterScoreTelemetry(
            extended_score=score,
            old_score=old_score,
            metrics=numeric_float,
        )
    return score, telemetry


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


def _score_directed_candidate_pair(
    pos: torch.Tensor,
    problem: LayoutProblem,
    cluster_ids: Optional[torch.Tensor],
    all_pairs_dist: Optional[np.ndarray] = None,
) -> "W5ScorePair":
    """Score a directed candidate under both frozen rulers.

    Parameters
    ----------
    pos : torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    problem : LayoutProblem
        Directed acyclic layout problem.
    cluster_ids : torch.Tensor, optional
        Per-node cluster ids with shape ``[N]``.
    all_pairs_dist : numpy.ndarray, optional
        Cached unweighted shortest paths with shape ``[N, N]``.

    Returns
    -------
    W5ScorePair
        Directed and undirected frozen-ruler composites from the same metric
        pass.
    """
    return _score_directed_candidate_payload(pos, problem, cluster_ids, all_pairs_dist)[0]


def _score_directed_candidate_payload(
    pos: torch.Tensor,
    problem: LayoutProblem,
    cluster_ids: Optional[torch.Tensor],
    all_pairs_dist: Optional[np.ndarray] = None,
) -> tuple["W5ScorePair", "W5HonestAxes"]:
    """Score a directed candidate and expose honest W5 route axes.

    Parameters
    ----------
    pos : torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    problem : LayoutProblem
        Directed acyclic layout problem.
    cluster_ids : torch.Tensor, optional
        Per-node cluster ids with shape ``[N]``.
    all_pairs_dist : numpy.ndarray, optional
        Cached unweighted shortest paths with shape ``[N, N]``.

    Returns
    -------
    tuple[W5ScorePair, W5HonestAxes]
        Directed/undirected composites and honest W5 routing axes from the
        same metric pass.
    """
    from dagua.layout.ops.pipelines.native_finisher import W5ScorePair, w5_honest_axes_from_metrics
    from dagua.metrics import composite, composite_undirected, full

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
        label_positions=problem.label_positions,
        edge_labels=problem.edge_labels,
        all_pairs_dist=all_pairs_dist,
        clusters=problem.clusters,
        cluster_parents=problem.cluster_parents,
        cluster_labels=problem.cluster_labels,
    )
    numeric["declared_hierarchical"] = True
    return (
        W5ScorePair(
            directed=float(composite(numeric)),
            undirected=float(composite_undirected(numeric)),
        ),
        w5_honest_axes_from_metrics(numeric),
    )


def _directed_ordering_candidate_dual_dominates(
    candidate: torch.Tensor,
    incumbent_pair: "W5ScorePair",
    problem: LayoutProblem,
    cluster_ids: Optional[torch.Tensor],
    all_pairs_dist: Optional[np.ndarray],
) -> tuple[bool, "W5ScorePair"]:
    """Return whether an ordering candidate may enter the winner contest.

    Parameters
    ----------
    candidate : torch.Tensor
        Ordering candidate positions with shape ``[N, 2]``.
    incumbent_pair : W5ScorePair
        Incumbent directed and undirected scores.
    problem : LayoutProblem
        Directed acyclic layout problem.
    cluster_ids : torch.Tensor, optional
        Per-node cluster ids with shape ``[N]``.
    all_pairs_dist : numpy.ndarray, optional
        Cached unweighted shortest paths with shape ``[N, N]``.

    Returns
    -------
    tuple[bool, W5ScorePair]
        Whether the candidate dominates under both rulers and the candidate
        score pair.
    """
    from dagua.layout.ops.pipelines.native_finisher import w5_dominates

    candidate_pair = _score_directed_candidate_pair(
        candidate,
        problem,
        cluster_ids,
        all_pairs_dist,
    )
    return w5_dominates(candidate_pair, incumbent_pair), candidate_pair


def _select_directed_winner(
    scores: Dict[str, float],
    telemetry: Dict[str, _DirectedClusterScoreTelemetry],
) -> str:
    """Select a directed winner while preserving current clustered outputs.

    Parameters
    ----------
    scores : dict[str, float]
        Extended full-referee score per finalist.
    telemetry : dict[str, _DirectedClusterScoreTelemetry]
        Clustered old/extended telemetry per finalist. When present, existing
        candidate families are ranked by the old ruler to keep 4A bit-stable;
        new cluster families can use
        ``_directed_cluster_candidate_is_dual_admissible`` before entering
        this contest.

    Returns
    -------
    str
        Winner name. Non-cluster contests retain the existing argmax path.
    """
    best_name = "incumbent"
    incumbent_telemetry = telemetry.get("incumbent")
    for name, score in scores.items():
        if name == "incumbent":
            continue
        if incumbent_telemetry is not None:
            candidate_telemetry = telemetry.get(name)
            best_telemetry = telemetry.get(best_name)
            if candidate_telemetry is None or best_telemetry is None:
                continue
            if candidate_telemetry.old_score > best_telemetry.old_score:
                best_name = name
        elif score > scores[best_name]:
            best_name = name
    return best_name


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
        _portfolio_available_work_s,
    )

    remaining = _portfolio_available_work_s(config, reserve_s=ABSOLUTE_DEADLINE_RESERVE_S)
    if remaining is None:
        return True
    required = DIRECTED_PREDICTED_COST_MULTIPLIER * max(0.0, float(predicted_cost_s))
    return float(remaining) > required


def _native_device_class(config: Optional[LayoutConfig]) -> str:
    """Return the native cost-model device class for a directed config.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared native configuration carrying a device string.

    Returns
    -------
    str
        ``"cuda"`` for CUDA devices, otherwise ``"cpu"``.
    """
    device = str(getattr(config, "device", "cpu")) if config is not None else "cpu"
    return "cuda" if device.startswith("cuda") else "cpu"


def _prediction_cpu_elapsed_s(started_process_time_s: float) -> float:
    """Return per-process CPU seconds elapsed for arm-cost prediction.

    Parameters
    ----------
    started_process_time_s : float
        ``time.process_time()`` reading captured before a candidate arm.

    Returns
    -------
    float
        Non-negative process CPU seconds elapsed since the captured start.
    """
    return max(0.0, time.process_time() - float(started_process_time_s))


def _directed_opaque_arm_cost(
    problem: LayoutProblem,
    config: Optional[LayoutConfig],
    prior_s: float,
) -> NativeWorkCost:
    """Return a deterministic placeholder cost for a directed challenger arm.

    Parameters
    ----------
    problem : LayoutProblem
        Directed layout problem being solved.
    config : LayoutConfig, optional
        Prepared native configuration carrying a device string.
    prior_s : float
        Structural prior seconds for this directed arm family.

    Returns
    -------
    NativeWorkCost
        Modeled work package used for ledger admission.
    """
    edge_count = int(problem.edge_index.shape[1]) if problem.edge_index.ndim >= 2 else 0
    device_class = _native_device_class(config)
    return NativeWorkCost(
        family="opaque",
        generation_dwu=max(0.0, float(prior_s)),
        reserved_score_dwu=0.0,
        metadata={
            "num_nodes": int(problem.num_nodes),
            "num_edges": edge_count,
            "device_class": device_class,
            "prior_s": max(0.0, float(prior_s)),
        },
        device_class=device_class,
    )


def _directed_recombinant_layered_enabled(problem: LayoutProblem) -> bool:
    """Return whether bounded recombinant layered candidates may be built.

    The gate is deliberately structural and narrow: item 3 targets semantic
    DAGs with scale-free/dependency/citation-like layered-axis gaps. Off-class
    graphs do not even construct the candidate family, preserving byte-level
    behavior outside the intended directed rows.

    Parameters
    ----------
    problem : LayoutProblem
        Directed layout problem with prepared topology classification.

    Returns
    -------
    bool
        ``True`` only for the bounded directed layered-gap class.
    """
    n = int(problem.num_nodes)
    edge_count = int(problem.edge_index.shape[1]) if problem.edge_index.numel() else 0
    if n < DIRECTED_RECOMBINANT_MIN_NODES or n > DIRECTED_RECOMBINANT_MAX_NODES or edge_count == 0:
        return False
    structure = problem.structure
    if structure is None:
        return False
    if not bool(getattr(structure, "is_directed_acyclic", getattr(structure, "is_acyclic", True))):
        return False
    if getattr(structure, "is_semantically_directed", True) is False:
        return False
    tags = set(getattr(structure, "topology_tags", ()))
    if tags.intersection({"planar_dag", "lattice_like", "wide_layered", "bipartite_dag"}):
        return False
    effective_layers = int(
        getattr(structure, "num_layers_effective", getattr(structure, "num_layers", 0))
    )
    if effective_layers < 10:
        return False
    # The current target rows are shallow-diameter, hub-skewed DAGs
    # (dependency/citation/power-law). This rejects broad random DAGs and
    # undirected-origin grids without naming benchmark rows.
    return (
        float(getattr(structure, "edge_to_node_ratio", 0.0)) >= 1.8
        and float(getattr(structure, "hub_edge_fraction", 0.0)) >= 0.22
        and 0 < int(getattr(structure, "diameter_estimate", 0)) <= 8
    )


def _recombinant_layered_specs() -> tuple[_RecombinantLayeredSpec, ...]:
    """Return the curated bounded IDEA-2 layered-stage crosses.

    Returns
    -------
    tuple[_RecombinantLayeredSpec, ...]
        Six complete layering/order/x-coordinate combinations. This is a
        curated grid, not a Cartesian product.
    """
    return (
        _RecombinantLayeredSpec(
            name="recomb_lp_bary_bk",
            layering="longest_path",
            ordering="barycenter_transpose",
            xcoord="brandes_koepf",
        ),
        _RecombinantLayeredSpec(
            name="recomb_lp_median_lp",
            layering="longest_path",
            ordering="median",
            xcoord="dot_lp",
        ),
        _RecombinantLayeredSpec(
            name="recomb_ns_bary_bk",
            layering="network_simplex_tightened",
            ordering="barycenter_transpose",
            xcoord="brandes_koepf",
        ),
        _RecombinantLayeredSpec(
            name="recomb_ns_median_lp",
            layering="network_simplex_tightened",
            ordering="median",
            xcoord="dot_lp",
        ),
        _RecombinantLayeredSpec(
            name="recomb_native_discrete_bk",
            layering="native_current",
            ordering="discrete",
            xcoord="brandes_koepf",
        ),
        _RecombinantLayeredSpec(
            name="recomb_lp_discrete_lp",
            layering="longest_path",
            ordering="discrete",
            xcoord="dot_lp",
        ),
    )


def _dense_rank_values(rank_values: Sequence[int]) -> list[int]:
    """Return rank values remapped to dense non-negative layer ids.

    Parameters
    ----------
    rank_values : sequence[int]
        Per-node rank values.

    Returns
    -------
    list[int]
        Dense rank id for each node.
    """
    if not rank_values:
        return []
    unique = {int(value) for value in rank_values}
    lookup = {value: index for index, value in enumerate(sorted(unique))}
    return [lookup[int(value)] for value in rank_values]


def _native_current_rank_values(incumbent: torch.Tensor, problem: LayoutProblem) -> list[int]:
    """Infer rank values from the incumbent's drawn y-layers.

    Parameters
    ----------
    incumbent : torch.Tensor
        Incumbent positions with shape ``[N, 2]``.
    problem : LayoutProblem
        Directed layout problem.

    Returns
    -------
    list[int]
        Dense per-node rank values.
    """
    rank_to_nodes = _rank_to_nodes_from_incumbent_y(
        incumbent,
        problem.edge_index,
        problem.num_nodes,
    )
    ranks = [0] * int(problem.num_nodes)
    for rank, nodes in rank_to_nodes.items():
        for node in nodes:
            if 0 <= int(node) < int(problem.num_nodes):
                ranks[int(node)] = int(rank)
    return _dense_rank_values(ranks)


def _recombinant_rank_values(
    spec: _RecombinantLayeredSpec,
    problem: LayoutProblem,
    incumbent: torch.Tensor,
) -> Optional[list[int]]:
    """Return per-node ranks for one recombinant layering stage.

    Parameters
    ----------
    spec : _RecombinantLayeredSpec
        Candidate specification.
    problem : LayoutProblem
        Directed layout problem.
    incumbent : torch.Tensor
        Incumbent positions with shape ``[N, 2]``.

    Returns
    -------
    list[int] or None
        Dense rank values, or ``None`` when the existing layerer cannot solve
        the graph.
    """
    if spec.layering == "native_current":
        return _native_current_rank_values(incumbent, problem)
    if spec.layering == "longest_path":
        ranks, _max_width, _long_edge_ratio = _directed_rank_profile(
            problem.edge_index,
            int(problem.num_nodes),
        )
        return _dense_rank_values(ranks)
    if spec.layering == "network_simplex_tightened":
        try:
            from dagua.layout.ops.elk import _network_simplex_layers

            edges = [
                (int(src), int(dst))
                for src, dst in problem.edge_index.detach().to(device="cpu").t().tolist()
                if int(src) != int(dst)
            ]
            return _dense_rank_values(_network_simplex_layers(int(problem.num_nodes), edges))
        except Exception:
            return None
    return None


def _recombinant_adjacency_lists(edge_index: torch.Tensor, num_nodes: int) -> list[list[int]]:
    """Return undirected adjacency lists for registered ordering ops.

    Parameters
    ----------
    edge_index : torch.Tensor
        Directed edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    list[list[int]]
        Neighbor ids keyed by node, including both directions so barycenter
        sweeps can see parents and children from the same adjacency cache.
    """
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    for src, dst in edge_index.detach().to(device="cpu", dtype=torch.long).t().tolist():
        src_i = int(src)
        dst_i = int(dst)
        if src_i == dst_i or not (0 <= src_i < num_nodes and 0 <= dst_i < num_nodes):
            continue
        adjacency[src_i].append(dst_i)
        adjacency[dst_i].append(src_i)
    return adjacency


def _initial_recombinant_positions(
    rank_values: Sequence[int],
    node_sizes: Optional[torch.Tensor],
    rank_sep: float,
    node_sep: float,
) -> torch.Tensor:
    """Build deterministic slots for a layered candidate before x assignment.

    Parameters
    ----------
    rank_values : sequence[int]
        Per-node rank values.
    node_sizes : torch.Tensor, optional
        Node sizes with shape ``[N, 2]``.
    rank_sep : float
        Vertical center-to-center rank spacing.
    node_sep : float
        Horizontal gap added to the median node width.

    Returns
    -------
    torch.Tensor
        Initial positions with shape ``[N, 2]``.
    """
    ranks = [int(value) for value in rank_values]
    n = len(ranks)
    if node_sizes is not None and node_sizes.numel() > 0:
        widths = node_sizes.detach().to(device="cpu", dtype=torch.float32)[:, 0]
        slot = float(widths.median().item()) + float(node_sep)
    else:
        slot = max(float(node_sep), 1.0)
    rank_to_nodes: dict[int, list[int]] = {}
    for node, rank in enumerate(ranks):
        rank_to_nodes.setdefault(rank, []).append(node)
    out = torch.zeros((n, 2), dtype=torch.float32)
    for rank, nodes in rank_to_nodes.items():
        center = 0.5 * float(len(nodes) - 1)
        for order, node in enumerate(nodes):
            out[node, 0] = (float(order) - center) * slot
            out[node, 1] = float(rank) * float(rank_sep)
    if n > 0:
        out = out - out.mean(dim=0, keepdim=True)
    return out


def _ordered_layers_from_ordering(
    rank_values: Sequence[int],
    ordering: Optional[torch.Tensor],
) -> list[list[int]]:
    """Return ordered node layers from rank values and optional ordering.

    Parameters
    ----------
    rank_values : sequence[int]
        Per-node rank values.
    ordering : torch.Tensor, optional
        Per-node ordering values with shape ``[N]``.

    Returns
    -------
    list[list[int]]
        Nodes grouped by rank and sorted left-to-right.
    """
    rank_to_nodes: dict[int, list[int]] = {}
    for node, rank in enumerate(rank_values):
        rank_to_nodes.setdefault(int(rank), []).append(node)
    if ordering is None:
        return [rank_to_nodes[rank] for rank in sorted(rank_to_nodes)]
    order_cpu = ordering.detach().to(device="cpu", dtype=torch.long)
    layers: list[list[int]] = []
    for rank in sorted(rank_to_nodes):
        nodes = rank_to_nodes[rank]
        layers.append(sorted(nodes, key=lambda node: (int(order_cpu[node].item()), node)))
    return layers


def _apply_recombinant_ordering(
    spec: _RecombinantLayeredSpec,
    problem: LayoutProblem,
    rank_values: Sequence[int],
    initial_pos: torch.Tensor,
    config: LayoutConfig,
) -> tuple[list[list[int]], torch.Tensor]:
    """Run one existing ordering stage for a recombinant candidate.

    Parameters
    ----------
    spec : _RecombinantLayeredSpec
        Candidate specification.
    problem : LayoutProblem
        Directed layout problem.
    rank_values : sequence[int]
        Dense per-node rank values.
    initial_pos : torch.Tensor
        Slot-based positions with shape ``[N, 2]``.
    config : LayoutConfig
        Prepared native configuration carrying budget metadata.

    Returns
    -------
    tuple[list[list[int]], torch.Tensor]
        Ordered layers and positions after ordering.
    """
    layers = torch.tensor(rank_values, dtype=torch.long)
    state = SolveState(
        pos=initial_pos.detach().clone(),
        layers=layers,
        adjacency=_recombinant_adjacency_lists(problem.edge_index, int(problem.num_nodes)),
    )
    ctx = RuntimeContext()
    if spec.ordering == "barycenter_transpose":
        from dagua.layout.ops.ordering import (
            BarycenterSweep,
            BarycenterSweepConfig,
            TransposeHeuristic,
            TransposeHeuristicConfig,
        )

        state = BarycenterSweep(BarycenterSweepConfig(passes=12, direction="both")).apply(
            problem,
            state,
            ctx,
        )
        state = TransposeHeuristic(TransposeHeuristicConfig(passes=4)).apply(problem, state, ctx)
    elif spec.ordering == "median":
        from dagua.layout.ops.ordering import MedianSweep, MedianSweepConfig

        state = MedianSweep(MedianSweepConfig(passes=12)).apply(problem, state, ctx)
    elif spec.ordering == "discrete":
        ordered_pos = _rank_local_zero_crossing_swap_candidate(
            initial_pos,
            problem.edge_index,
            max_passes=_ordering_portfolio_max_passes(int(problem.num_nodes)),
            config=config,
        )
        state.pos = ordered_pos
        state.ordering = None
    if spec.ordering == "discrete":
        rank_to_nodes = _rank_to_nodes_from_incumbent_y(
            state.pos,
            problem.edge_index,
            problem.num_nodes,
        )
        ordered_layers = [
            sorted(nodes, key=lambda node: float(state.pos[node, 0].item()))
            for _rank, nodes in sorted(rank_to_nodes.items())
        ]
    else:
        ordered_layers = _ordered_layers_from_ordering(rank_values, state.ordering)
    return ordered_layers, state.pos.detach().to(device="cpu", dtype=torch.float32)


def _assign_recombinant_x_coordinates(
    spec: _RecombinantLayeredSpec,
    ordered_layers: Sequence[Sequence[int]],
    problem: LayoutProblem,
    node_sep: float,
) -> Optional[torch.Tensor]:
    """Assign x coordinates for one recombinant candidate.

    Parameters
    ----------
    spec : _RecombinantLayeredSpec
        Candidate specification.
    ordered_layers : sequence[sequence[int]]
        Ordered node layers.
    problem : LayoutProblem
        Directed layout problem.
    node_sep : float
        Horizontal node separation.

    Returns
    -------
    torch.Tensor or None
        X coordinates with shape ``[N]``, or ``None`` if the existing x solver
        cannot run.
    """
    n = int(problem.num_nodes)
    sizes = (
        problem.node_sizes.detach().to(device="cpu", dtype=torch.float32)
        if problem.node_sizes is not None
        else torch.full((n, 2), float(node_sep), dtype=torch.float32)
    )
    if spec.xcoord == "dot_lp":
        try:
            from dagua.layout.ops.pipelines.dagua_native import (
                _graphviz_dot_x_position_network_simplex,
            )

            return _graphviz_dot_x_position_network_simplex(
                rank_ordering=ordered_layers,
                node_widths=sizes[:, 0],
                edge_index=problem.edge_index.detach().to(device="cpu"),
                node_sep=node_sep,
                edge_weights=(
                    None
                    if problem.edge_weights is None
                    else problem.edge_weights.detach().to(device="cpu")
                ),
                center=True,
            ).to(dtype=torch.float32)
        except Exception:
            return None
    if spec.xcoord == "brandes_koepf":
        try:
            from dagua.layout.ops.brandes_koepf import brandes_koepf_x_assignment

            order_index = {
                int(node): order for layer in ordered_layers for order, node in enumerate(layer)
            }
            rank_index = {
                int(node): rank for rank, layer in enumerate(ordered_layers) for node in layer
            }
            predecessors: dict[int, list[int]] = {node: [] for node in range(n)}
            successors: dict[int, list[int]] = {node: [] for node in range(n)}
            candidate_edges = problem.edge_index.detach().to(device="cpu", dtype=torch.long)
            for src, dst in candidate_edges.t().tolist():
                src_i = int(src)
                dst_i = int(dst)
                if src_i == dst_i or not (0 <= src_i < n and 0 <= dst_i < n):
                    continue
                if abs(rank_index.get(dst_i, 0) - rank_index.get(src_i, 0)) != 1:
                    continue
                successors[src_i].append(dst_i)
                predecessors[dst_i].append(src_i)
            for node in range(n):
                predecessors[node].sort(key=lambda item: (order_index.get(item, 0), item))
                successors[node].sort(key=lambda item: (order_index.get(item, 0), item))
            widths = {node: float(sizes[node, 0].item()) for node in range(n)}
            x_map = brandes_koepf_x_assignment(
                layering=ordered_layers,
                predecessors=predecessors,
                successors=successors,
                widths=widths,
                dummy_nodes=set(),
                node_sep=node_sep,
            )
            x_values = torch.tensor([float(x_map.get(node, 0.0)) for node in range(n)])
            return (x_values - x_values.mean()).to(dtype=torch.float32)
        except Exception:
            return None
    return None


def _build_recombinant_layered_candidate(
    spec: _RecombinantLayeredSpec,
    problem: LayoutProblem,
    incumbent: torch.Tensor,
    config: LayoutConfig,
) -> Optional[torch.Tensor]:
    """Build one complete recombinant layered candidate.

    Parameters
    ----------
    spec : _RecombinantLayeredSpec
        Candidate specification.
    problem : LayoutProblem
        Directed layout problem.
    incumbent : torch.Tensor
        Incumbent positions with shape ``[N, 2]``.
    config : LayoutConfig
        Prepared native configuration.

    Returns
    -------
    torch.Tensor or None
        Complete candidate positions with shape ``[N, 2]`` when every existing
        stage succeeds, otherwise ``None``.
    """
    rank_values = _recombinant_rank_values(spec, problem, incumbent)
    if rank_values is None or len(rank_values) != int(problem.num_nodes):
        return None
    rank_sep = float(getattr(config, "_dagua_native_rank_sep", config.rank_sep))
    node_sep = float(getattr(config, "_dagua_native_node_sep", config.node_sep))
    initial = _initial_recombinant_positions(rank_values, problem.node_sizes, rank_sep, node_sep)
    ordered_layers, ordered_pos = _apply_recombinant_ordering(
        spec,
        problem,
        rank_values,
        initial,
        config,
    )
    x_values = _assign_recombinant_x_coordinates(spec, ordered_layers, problem, node_sep)
    if x_values is None or int(x_values.numel()) != int(problem.num_nodes):
        return None
    y_values = torch.tensor(rank_values, dtype=torch.float32) * rank_sep
    out = torch.stack([x_values.to(dtype=torch.float32), y_values], dim=1)
    out[:, 1] = out[:, 1] - out[:, 1].mean()
    if not torch.isfinite(out).all():
        return None
    # Preserve the ordering-stage y frame when native-current/discrete creates
    # a useful drawn-rank interpretation; x is still replaced by the selected
    # coordinate stage, making this a complete layered recombination.
    if spec.layering == "native_current" and ordered_pos.shape == out.shape:
        out[:, 1] = ordered_pos[:, 1] - ordered_pos[:, 1].mean()
    return out


def _directed_recombinant_layered_candidates(
    problem: LayoutProblem,
    incumbent: torch.Tensor,
    config: LayoutConfig,
) -> dict[str, torch.Tensor]:
    """Build bounded recombinant layered candidates for target DAG classes.

    Parameters
    ----------
    problem : LayoutProblem
        Directed layout problem.
    incumbent : torch.Tensor
        Incumbent positions with shape ``[N, 2]``.
    config : LayoutConfig
        Prepared native configuration carrying optional deadline metadata.

    Returns
    -------
    dict[str, torch.Tensor]
        Candidate family names mapped to raw complete layouts.
    """
    if not _directed_recombinant_layered_enabled(problem):
        return {}
    candidates: dict[str, torch.Tensor] = {}
    predicted_cost = estimate_native_work_cost(
        problem,
        "opaque",
        {"volume": DIRECTED_RECOMBINANT_PRIOR_S},
        _native_device_class(config),
    )
    predicted_cost_s = predicted_cost.generation_dwu + predicted_cost.reserved_score_dwu
    for spec in _recombinant_layered_specs():
        if len(candidates) >= DIRECTED_RECOMBINANT_MAX_CANDIDATES:
            break
        if not _predicted_arm_budget_available(config, predicted_cost_s) or not admit_native_work(
            config,
            predicted_cost,
            f"optional_directed_recombinant_{spec.name}",
        ):
            break
        process_started = time.process_time()
        candidate = _build_recombinant_layered_candidate(spec, problem, incumbent, config)
        recombinant_cpu_s = _prediction_cpu_elapsed_s(process_started)
        _LOGGER.info(
            "Directed candidate runtime family=recombinant arm=%s cpu_seconds=%.3f",
            spec.name,
            recombinant_cpu_s,
        )
        if candidate is None:
            continue
        candidates[spec.name] = candidate
    return candidates


def _register_recombinant_layered_candidates(
    problem: LayoutProblem,
    incumbent: torch.Tensor,
    config: LayoutConfig,
    positions: Dict[str, torch.Tensor],
    scores: Dict[str, float],
    incumbent_pair: Optional["W5ScorePair"],
    cluster_ids: Optional[torch.Tensor],
    all_pairs_dist: Optional[np.ndarray],
    arm_timings: Dict[str, Tuple[float, float]],
) -> Optional["W5ScorePair"]:
    """Register only dual-ruler-dominating recombinant candidates.

    Parameters
    ----------
    problem : LayoutProblem
        Directed layout problem.
    incumbent : torch.Tensor
        Incumbent positions with shape ``[N, 2]``.
    config : LayoutConfig
        Prepared native configuration.
    positions : dict[str, torch.Tensor]
        Candidate registry updated in place.
    scores : dict[str, float]
        Full directed scores updated for admitted candidates.
    incumbent_pair : W5ScorePair, optional
        Cached incumbent dual-ruler score pair.
    cluster_ids : torch.Tensor, optional
        Per-node cluster ids with shape ``[N]``.
    all_pairs_dist : numpy.ndarray, optional
        Cached shortest-path matrix with shape ``[N, N]``.
    arm_timings : dict[str, tuple[float, float]]
        Per-arm timing registry updated for admitted candidates.

    Returns
    -------
    W5ScorePair or None
        Cached incumbent score pair when computed, otherwise ``None``.
    """
    from dagua.layout.ops.pipelines.native_undirected import _portfolio_has_budget

    if not _portfolio_has_budget(config, min_remaining_s=2.0):
        return incumbent_pair
    raw_candidates = _directed_recombinant_layered_candidates(problem, incumbent, config)
    if not raw_candidates:
        return incumbent_pair
    if incumbent_pair is None:
        incumbent_pair = _score_directed_candidate_pair(
            incumbent,
            problem,
            cluster_ids,
            all_pairs_dist,
        )
    for name, raw_candidate in raw_candidates.items():
        candidate_started = time.perf_counter()
        variants: Dict[str, torch.Tensor] = {}
        _register_challenger_variants(
            name,
            raw_candidate,
            problem,
            config,
            variants,
            preserve_rank_order=True,
        )
        for variant_name, candidate in variants.items():
            # Item 3 is monotone by construction: recombinant variants are
            # admitted only when the same frozen directed and undirected
            # composites both beat the incumbent. Off-class rows never reach
            # candidate construction at all.
            dominates, candidate_pair = _directed_ordering_candidate_dual_dominates(
                candidate,
                incumbent_pair,
                problem,
                cluster_ids,
                all_pairs_dist,
            )
            if not dominates:
                continue
            positions[variant_name] = candidate
            scores[variant_name] = candidate_pair.directed
            arm_timings[variant_name] = (candidate_started, time.perf_counter())
    return incumbent_pair


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


def _rank_to_nodes_from_incumbent_y(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> dict[int, list[int]]:
    """Group nodes by the y-layers that the incumbent actually drew.

    Parameters
    ----------
    positions : torch.Tensor
        Incumbent positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Directed edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    dict[int, list[int]]
        Layer ids mapped to node ids. If the incumbent has no repeated
        y-layers, the longest-path ranks are used as a conservative fallback.
    """
    if num_nodes <= 0:
        return {}
    cpu_pos = positions.detach().to(device="cpu", dtype=torch.float32)
    y_values = [
        (float(cpu_pos[node, 1].item()), node)
        for node in range(min(num_nodes, int(cpu_pos.shape[0])))
    ]
    y_values.sort()
    rank_to_nodes: dict[int, list[int]] = {}
    current_rank = -1
    current_y = 0.0
    tolerance = DIRECTED_ORDERING_Y_TOLERANCE
    for y_value, node in y_values:
        if current_rank < 0 or abs(y_value - current_y) > tolerance:
            current_rank += 1
            current_y = y_value
            rank_to_nodes[current_rank] = []
        rank_to_nodes[current_rank].append(node)
    if max((len(nodes) for nodes in rank_to_nodes.values()), default=0) >= 2:
        return rank_to_nodes

    ranks, _max_width, _long_edge_ratio = _directed_rank_profile(edge_index, num_nodes)
    fallback: dict[int, list[int]] = {}
    for node, rank in enumerate(ranks):
        fallback.setdefault(rank, []).append(node)
    return fallback


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
    cpu_edges = edge_index.detach().to(device="cpu", dtype=torch.long)
    edge_count = int(cpu_edges.shape[1]) if cpu_edges.numel() else 0
    if edge_count < 2:
        return 0
    pair_count = edge_count * (edge_count - 1) // 2
    if pair_count > EXACT_CROSSING_COUNT_VECTOR_PAIR_CAP:
        return _exact_crossing_count_loop(pos, cpu_edges)

    cpu_pos = pos.detach().to(device="cpu", dtype=torch.float32)
    src = cpu_edges[0]
    dst = cpu_edges[1]
    left, right = torch.triu_indices(edge_count, edge_count, offset=1)
    src_a = src[left]
    dst_a = dst[left]
    src_b = src[right]
    dst_b = dst[right]
    non_incident = (src_a != src_b) & (src_a != dst_b) & (dst_a != src_b) & (dst_a != dst_b)
    if not bool(non_incident.any().item()):
        return 0
    src_a = src_a[non_incident]
    dst_a = dst_a[non_incident]
    src_b = src_b[non_incident]
    dst_b = dst_b[non_incident]
    a = cpu_pos[src_a]
    b = cpu_pos[dst_a]
    c = cpu_pos[src_b]
    d = cpu_pos[dst_b]

    def _orient_batch(
        first: torch.Tensor,
        second: torch.Tensor,
        third: torch.Tensor,
    ) -> torch.Tensor:
        """Return batched signed triangle areas for crossing tests.

        Parameters
        ----------
        first : torch.Tensor
            First point batch with shape ``[P, 2]``.
        second : torch.Tensor
            Second point batch with shape ``[P, 2]``.
        third : torch.Tensor
            Third point batch with shape ``[P, 2]``.

        Returns
        -------
        torch.Tensor
            Signed areas with shape ``[P]``.
        """
        return (second[:, 0] - first[:, 0]) * (third[:, 1] - first[:, 1]) - (
            second[:, 1] - first[:, 1]
        ) * (third[:, 0] - first[:, 0])

    o1 = _orient_batch(a, b, c)
    o2 = _orient_batch(a, b, d)
    o3 = _orient_batch(c, d, a)
    o4 = _orient_batch(c, d, b)
    crossings = (o1 * o2 < 0.0) & (o3 * o4 < 0.0)
    return int(crossings.sum().item())


def _exact_crossing_count_loop(pos: torch.Tensor, edge_index: torch.Tensor) -> int:
    """Count crossings with the pre-vectorized loop implementation.

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


def _crossing_edge_pairs(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    max_pairs: int,
    config: Optional[LayoutConfig] = None,
    started_at: Optional[float] = None,
    wall_time_cap_s: Optional[float] = None,
) -> list[tuple[int, int, int, int]]:
    """Return a bounded list of exact crossing edge endpoint ids.

    Parameters
    ----------
    pos : torch.Tensor
        Positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    max_pairs : int
        Maximum number of crossing pairs to return.
    config : LayoutConfig, optional
        Prepared native configuration carrying an optional benchmark deadline.
    started_at : float, optional
        ``time.perf_counter()`` value captured when the ordering arm started.
    wall_time_cap_s : float, optional
        Absolute wall-clock cap for this invocation.

    Returns
    -------
    list[tuple[int, int, int, int]]
        Crossing pairs as ``(src_a, dst_a, src_b, dst_b)`` endpoint ids.
    """
    if max_pairs <= 0:
        return []
    edges = [(int(src), int(dst)) for src, dst in edge_index.t().detach().cpu().tolist()]
    crossings: list[tuple[int, int, int, int]] = []
    pairs_examined = 0
    for left_idx, (src_a, dst_a) in enumerate(edges):
        for src_b, dst_b in edges[left_idx + 1 :]:
            pairs_examined += 1
            if (
                pairs_examined % DIRECTED_ORDERING_PAIR_BUDGET_CHECK_INTERVAL == 0
                and not _ordering_budget_available(config, started_at, wall_time_cap_s)
            ):
                return crossings
            if len({src_a, dst_a, src_b, dst_b}) < 4:
                continue
            if _segments_cross(pos[src_a], pos[dst_a], pos[src_b], pos[dst_b]):
                crossings.append((src_a, dst_a, src_b, dst_b))
                if len(crossings) >= int(max_pairs):
                    return crossings
    return crossings


def _apply_rank_order(
    positions: torch.Tensor,
    nodes: list[int],
    ordered_nodes: list[int],
) -> torch.Tensor:
    """Assign one rank's x-slot multiset to a requested node ordering.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    nodes : list[int]
        Node ids in the rank being reordered.
    ordered_nodes : list[int]
        Desired left-to-right node order for ``nodes``.

    Returns
    -------
    torch.Tensor
        Reordered candidate positions with shape ``[N, 2]``.
    """
    candidate = positions.clone()
    ordered_x = sorted(float(positions[node, 0].item()) for node in nodes)
    for node, x_value in zip(ordered_nodes, ordered_x):
        candidate[node, 0] = x_value
    return candidate


def _rank_order_from_neighbor_stat(
    nodes: list[int],
    pos: torch.Tensor,
    neighbors: dict[int, list[int]],
    use_median: bool,
) -> list[int]:
    """Order rank nodes by neighboring x-coordinate barycenter or median.

    Parameters
    ----------
    nodes : list[int]
        Node ids in one rank.
    pos : torch.Tensor
        Current positions with shape ``[N, 2]``.
    neighbors : dict[int, list[int]]
        Neighbor ids keyed by node id.
    use_median : bool
        Whether to use the median instead of the mean.

    Returns
    -------
    list[int]
        Desired left-to-right node order.
    """
    ordered = sorted(nodes, key=lambda node: float(pos[node, 0].item()))
    keyed: list[tuple[float, int, int]] = []
    for ordinal, node in enumerate(ordered):
        values = [float(pos[neighbor, 0].item()) for neighbor in neighbors.get(node, [])]
        if values:
            values.sort()
            if use_median:
                middle = len(values) // 2
                key = (
                    values[middle]
                    if len(values) % 2
                    else 0.5 * (values[middle - 1] + values[middle])
                )
            else:
                key = float(sum(values)) / float(len(values))
        else:
            key = float(pos[node, 0].item())
        keyed.append((key, ordinal, node))
    return [node for _key, _ordinal, node in sorted(keyed)]


def _ordering_wall_time_cap_s(num_nodes: int) -> float:
    """Return the absolute wall-clock cap for the ordering arm.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    float
        Maximum seconds allowed for the rank-order search.
    """
    if int(num_nodes) <= DIRECTED_NARROW_SEED_NODE_CAP:
        return DIRECTED_ORDERING_SMALL_WALL_TIME_CAP_S
    return DIRECTED_ORDERING_MEDIUM_WALL_TIME_CAP_S


def _ordering_portfolio_max_passes(num_nodes: int) -> int:
    """Return local-search passes for portfolio ordering admission.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    int
        Local-search pass count for the portfolio ordering arm.
    """
    if DIRECTED_ORDERING_PORTFOLIO_SMALL_NODE_CAP < int(num_nodes) <= DIRECTED_NARROW_SEED_NODE_CAP:
        # In the restored 65..128 band, pass-heavy local search spends the
        # small wall cap before the bounded nudge phase can recover dep100.
        return 0
    return 3


def _ordering_deadline_check_interval(edge_count: int) -> int:
    """Return a trial interval that limits deadline blind windows.

    Parameters
    ----------
    edge_count : int
        Number of directed edges.

    Returns
    -------
    int
        Number of trials between budget checks.
    """
    if int(edge_count) >= 1024:
        return 1
    if int(edge_count) >= 512:
        return 2
    if int(edge_count) >= 128:
        return 4
    return DIRECTED_ORDERING_DEADLINE_CHECK_INTERVAL


def _ordering_trial_estimate(rank_to_nodes: dict[int, list[int]], max_passes: int) -> int:
    """Estimate rank-ordering trial count before exact crossing work starts.

    Parameters
    ----------
    rank_to_nodes : dict[int, list[int]]
        Candidate y-layer groups.
    max_passes : int
        Maximum local-search passes.

    Returns
    -------
    int
        Conservative trial estimate for exhaustive and local moves.
    """
    trials = 0
    exhaustive_total = 0
    for nodes in rank_to_nodes.values():
        width = len(nodes)
        if width < 2:
            continue
        if width <= DIRECTED_ORDERING_EXHAUSTIVE_WIDTH_CAP:
            rank_trials = math.factorial(width)
            if (
                rank_trials <= DIRECTED_ORDERING_EXHAUSTIVE_PER_RANK_PERM_CAP
                and exhaustive_total + rank_trials <= DIRECTED_ORDERING_EXHAUSTIVE_PERM_CAP
            ):
                exhaustive_total += rank_trials
                trials += rank_trials
        non_adjacent_swaps = max((width - 1) * (width - 2) // 2, 0)
        reinsertions = width * max(width - 1, 0)
        neighbor_orders = 4
        adjacent_swaps = max(width - 1, 0)
        trials += max(0, int(max_passes)) * (
            non_adjacent_swaps + reinsertions + neighbor_orders + adjacent_swaps
        )
    if trials > 0:
        return trials
    return trials


def _ordering_cost_admissible(
    num_nodes: int,
    edge_count: int,
    rank_to_nodes: dict[int, list[int]],
    max_passes: int,
) -> bool:
    """Return whether the ordering arm's predicted exact-crossing cost is bounded.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    edge_count : int
        Number of directed edges.
    rank_to_nodes : dict[int, list[int]]
        Candidate y-layer groups.
    max_passes : int
        Maximum local-search passes.

    Returns
    -------
    bool
        ``True`` when edge-pair and trial-pair counts are within the hard cap.
    """
    if edge_count < 2:
        return False
    pair_count = edge_count * (edge_count - 1) // 2
    if int(num_nodes) <= DIRECTED_NARROW_SEED_NODE_CAP:
        if edge_count > DIRECTED_NARROW_SEED_EDGE_CAP:
            return False
        return _ordering_trial_estimate(rank_to_nodes, max_passes) > 0
    elif pair_count > DIRECTED_ORDERING_MEDIUM_EDGE_PAIR_CAP:
        return False
    trial_estimate = _ordering_trial_estimate(rank_to_nodes, max_passes)
    return trial_estimate > 0 and trial_estimate * pair_count <= DIRECTED_ORDERING_TRIAL_PAIR_CAP


def _ordering_budget_available(
    config: Optional[LayoutConfig],
    started_at: Optional[float] = None,
    wall_time_cap_s: Optional[float] = None,
) -> bool:
    """Return whether the ordering arm may perform another trial.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared native configuration carrying an optional benchmark deadline.
    started_at : float, optional
        ``time.perf_counter()`` value captured when the ordering arm started.
    wall_time_cap_s : float, optional
        Absolute wall-clock cap for this invocation.

    Returns
    -------
    bool
        ``True`` when the benchmark deadline and local wall-clock cap both have
        room for another trial.
    """
    from dagua.layout.ops.pipelines.native_undirected import _portfolio_has_budget

    if (
        started_at is not None
        and wall_time_cap_s is not None
        and time.perf_counter() - float(started_at) >= float(wall_time_cap_s)
    ):
        return False
    return _portfolio_has_budget(config, min_remaining_s=DIRECTED_FULL_SCORE_MIN_REMAINING_S)


def _try_rank_order(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    nodes: list[int],
    ordered_nodes: list[int],
    best_crossings: int,
) -> tuple[torch.Tensor, int, bool]:
    """Evaluate one rank ordering and accept it only on fewer crossings.

    Parameters
    ----------
    pos : torch.Tensor
        Current positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    nodes : list[int]
        Rank node ids.
    ordered_nodes : list[int]
        Candidate left-to-right order for ``nodes``.
    best_crossings : int
        Current exact crossing count.

    Returns
    -------
    tuple[torch.Tensor, int, bool]
        Updated positions, crossing count, and whether the trial improved.
    """
    if ordered_nodes == sorted(nodes, key=lambda node: float(pos[node, 0].item())):
        return pos, best_crossings, False
    candidate = _apply_rank_order(pos, nodes, ordered_nodes)
    crossings = _exact_crossing_count(candidate, edge_index)
    if crossings < best_crossings:
        return candidate, crossings, True
    return pos, best_crossings, False


def _try_crossing_endpoint_nudges(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    best_crossings: int,
    config: Optional[LayoutConfig],
    started_at: float,
    wall_time_cap_s: float,
) -> tuple[torch.Tensor, int]:
    """Move crossing endpoints horizontally when slot permutations cannot help.

    Parameters
    ----------
    pos : torch.Tensor
        Current positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    best_crossings : int
        Current exact crossing count.
    config : LayoutConfig, optional
        Prepared native configuration carrying an optional benchmark deadline.
    started_at : float
        ``time.perf_counter()`` value captured when the ordering arm started.
    wall_time_cap_s : float
        Absolute wall-clock cap for this invocation.

    Returns
    -------
    tuple[torch.Tensor, int]
        Updated positions and exact crossing count.
    """
    if best_crossings <= 0:
        return pos, best_crossings
    x_values = pos[:, 0].detach().to(device="cpu", dtype=torch.float32)
    span = float((x_values.max() - x_values.min()).item()) if x_values.numel() else 0.0
    gap = max(1.0, span * 0.05)
    trials = 0
    while trials < DIRECTED_ORDERING_NUDGE_TRIAL_CAP and best_crossings > 0:
        if not _ordering_budget_available(config, started_at, wall_time_cap_s):
            return pos, best_crossings
        crossing_pairs = _crossing_edge_pairs(
            pos,
            edge_index,
            max_pairs=DIRECTED_ORDERING_NUDGE_CROSSING_CAP,
            config=config,
            started_at=started_at,
            wall_time_cap_s=wall_time_cap_s,
        )
        best_candidate: Optional[torch.Tensor] = None
        best_trial_crossings = best_crossings
        best_displacement = math.inf
        for src_a, dst_a, src_b, dst_b in crossing_pairs:
            anchors = sorted(
                {
                    float(pos[src_a, 0].item()),
                    float(pos[dst_a, 0].item()),
                    float(pos[src_b, 0].item()),
                    float(pos[dst_b, 0].item()),
                }
            )
            candidate_x_values = sorted(
                {
                    anchors[0] - gap,
                    anchors[-1] + gap,
                    *[anchor - 2.0 * gap for anchor in anchors],
                    *[anchor - gap for anchor in anchors],
                    *[anchor + gap for anchor in anchors],
                    *[anchor + 2.0 * gap for anchor in anchors],
                }
            )
            for node in (src_a, dst_a, src_b, dst_b):
                original_x = float(pos[node, 0].item())
                for candidate_x in candidate_x_values:
                    if trials >= DIRECTED_ORDERING_NUDGE_TRIAL_CAP:
                        break
                    trials += 1
                    displacement = abs(candidate_x - original_x)
                    if displacement <= 1.0e-6:
                        continue
                    if not _ordering_budget_available(config, started_at, wall_time_cap_s):
                        return pos, best_crossings
                    candidate = pos.clone()
                    candidate[node, 0] = candidate_x
                    crossings = _exact_crossing_count(candidate, edge_index)
                    if crossings < best_trial_crossings or (
                        crossings == best_trial_crossings and displacement < best_displacement
                    ):
                        best_candidate = candidate
                        best_trial_crossings = crossings
                        best_displacement = displacement
                if trials >= DIRECTED_ORDERING_NUDGE_TRIAL_CAP:
                    break
        if best_candidate is None or best_trial_crossings >= best_crossings:
            break
        pos = best_candidate
        best_crossings = best_trial_crossings
    return pos, best_crossings


def _rank_local_zero_crossing_swap_candidate(
    incumbent: torch.Tensor,
    edge_index: torch.Tensor,
    max_passes: int = 3,
    config: Optional[LayoutConfig] = None,
) -> torch.Tensor:
    """Build a discrete within-rank ordering candidate.

    Parameters
    ----------
    incumbent : torch.Tensor
        Incumbent positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Directed edge tensor with shape ``[2, E]``.
    max_passes : int, default=3
        Maximum adjacent-swap sweeps.
    config : LayoutConfig, optional
        Prepared native configuration carrying an optional benchmark deadline.

    Returns
    -------
    torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    """
    started_at = time.perf_counter()
    pos = incumbent.detach().to(device="cpu", dtype=torch.float32).clone()
    n = int(pos.shape[0])
    if n <= 2 or n > DIRECTED_ORDERING_MEDIUM_NODE_CAP:
        return pos
    wall_time_cap_s = _ordering_wall_time_cap_s(n)
    rank_to_nodes = _rank_to_nodes_from_incumbent_y(pos, edge_index, n)
    max_width = max((len(nodes) for nodes in rank_to_nodes.values()), default=0)
    if max_width < 2:
        return pos
    edge_count = int(edge_index.shape[1]) if edge_index.numel() else 0
    if not _ordering_cost_admissible(n, edge_count, rank_to_nodes, max_passes):
        return pos
    best_crossings = _exact_crossing_count(pos, edge_index)
    if best_crossings == 0:
        return pos
    if not _ordering_budget_available(config, started_at, wall_time_cap_s):
        return pos

    exhaustive_trials = 0
    check_interval = _ordering_deadline_check_interval(edge_count)
    if max_width <= DIRECTED_ORDERING_EXHAUSTIVE_WIDTH_CAP:
        for nodes in rank_to_nodes.values():
            width = len(nodes)
            if width < 2:
                continue
            estimated_trials = math.factorial(width)
            if estimated_trials > DIRECTED_ORDERING_EXHAUSTIVE_PER_RANK_PERM_CAP:
                continue
            if exhaustive_trials + estimated_trials > DIRECTED_ORDERING_EXHAUSTIVE_PERM_CAP:
                break
            current_order = sorted(nodes, key=lambda node: float(pos[node, 0].item()))
            rank_best_pos = pos
            rank_best_crossings = best_crossings
            for trial_index, permuted in enumerate(permutations(current_order), start=1):
                if trial_index % check_interval == 0 and not _ordering_budget_available(
                    config, started_at, wall_time_cap_s
                ):
                    break
                candidate = _apply_rank_order(pos, nodes, list(permuted))
                crossings = _exact_crossing_count(candidate, edge_index)
                if crossings < rank_best_crossings:
                    rank_best_pos = candidate
                    rank_best_crossings = crossings
            exhaustive_trials += estimated_trials
            if rank_best_crossings < best_crossings:
                pos = rank_best_pos
                best_crossings = rank_best_crossings
            if best_crossings == 0 or not _ordering_budget_available(
                config,
                started_at,
                wall_time_cap_s,
            ):
                return pos

    if n <= DIRECTED_NARROW_SEED_NODE_CAP:
        for _pass in range(max(0, int(max_passes))):
            if not _ordering_budget_available(config, started_at, wall_time_cap_s):
                break
            changed = False
            for nodes in rank_to_nodes.values():
                if len(nodes) < 2:
                    continue
                ordered = sorted(nodes, key=lambda node: float(pos[node, 0].item()))
                rank_orders: list[list[int]] = []
                for left_index in range(len(ordered) - 1):
                    for right_index in range(left_index + 2, len(ordered)):
                        swapped = list(ordered)
                        swapped[left_index], swapped[right_index] = (
                            swapped[right_index],
                            swapped[left_index],
                        )
                        rank_orders.append(swapped)
                for source_index in range(len(ordered)):
                    for target_index in range(len(ordered)):
                        if source_index == target_index:
                            continue
                        reinserted = list(ordered)
                        node = reinserted.pop(source_index)
                        reinserted.insert(target_index, node)
                        rank_orders.append(reinserted)
                for rank_order in rank_orders:
                    if not _ordering_budget_available(config, started_at, wall_time_cap_s):
                        break
                    pos, best_crossings, improved = _try_rank_order(
                        pos,
                        edge_index,
                        nodes,
                        rank_order,
                        best_crossings,
                    )
                    changed = changed or improved
                if best_crossings == 0:
                    return pos
            if not changed:
                break
        pos, best_crossings = _try_crossing_endpoint_nudges(
            pos,
            edge_index,
            best_crossings,
            config,
            started_at,
            wall_time_cap_s,
        )
        return pos

    incoming: dict[int, list[int]] = {node: [] for node in range(n)}
    outgoing: dict[int, list[int]] = {node: [] for node in range(n)}
    for src, dst in edge_index.t().detach().cpu().tolist():
        src_i = int(src)
        dst_i = int(dst)
        if 0 <= src_i < n and 0 <= dst_i < n:
            outgoing[src_i].append(dst_i)
            incoming[dst_i].append(src_i)
    rank_items = sorted(rank_to_nodes.items())
    for _pass in range(max(0, int(max_passes))):
        if not _ordering_budget_available(config, started_at, wall_time_cap_s):
            break
        changed = False
        for _rank, nodes in rank_items:
            if len(nodes) < 2:
                continue
            for neighbors, use_median in (
                (incoming, False),
                (outgoing, False),
                (incoming, True),
                (outgoing, True),
            ):
                if not _ordering_budget_available(config, started_at, wall_time_cap_s):
                    break
                ordered_nodes = _rank_order_from_neighbor_stat(nodes, pos, neighbors, use_median)
                pos, best_crossings, improved = _try_rank_order(
                    pos,
                    edge_index,
                    nodes,
                    ordered_nodes,
                    best_crossings,
                )
                changed = changed or improved
                if best_crossings == 0:
                    return pos
            ordered = sorted(nodes, key=lambda node: float(pos[node, 0].item()))
            for left_index in range(len(ordered) - 1):
                if not _ordering_budget_available(config, started_at, wall_time_cap_s):
                    break
                swapped = list(ordered)
                swapped[left_index], swapped[left_index + 1] = (
                    swapped[left_index + 1],
                    swapped[left_index],
                )
                pos, best_crossings, improved = _try_rank_order(
                    pos,
                    edge_index,
                    nodes,
                    swapped,
                    best_crossings,
                )
                changed = changed or improved
                if improved:
                    ordered = sorted(nodes, key=lambda node: float(pos[node, 0].item()))
                if best_crossings == 0:
                    return pos
        if not changed:
            break
    pos, best_crossings = _try_crossing_endpoint_nudges(
        pos,
        edge_index,
        best_crossings,
        config,
        started_at,
        wall_time_cap_s,
    )
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
    from dagua.layout.ops.pipelines.dagua_native import (
        _append_terminal_w5_seed,
        _run_native_problem,
    )
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
    for seed_name, seed_pos in list(
        getattr(incumbent_config, "_dagua_native_terminal_w5_seed_bank", [])
    ):
        _append_terminal_w5_seed(config, f"directed_incumbent_{seed_name}", seed_pos)
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
    if problem.clusters:
        incumbent_score, incumbent_score_telemetry = _score_directed_candidate_referee_payload(
            incumbent,
            problem,
            cluster_ids,
            all_pairs_dist,
        )
    else:
        incumbent_score = _score_directed_candidate_cached(
            incumbent,
            problem,
            cluster_ids,
            all_pairs_dist,
        )
        incumbent_score_telemetry = None
    scores: Dict[str, float] = {"incumbent": incumbent_score}
    cluster_score_telemetry: Dict[str, _DirectedClusterScoreTelemetry] = {}
    if incumbent_score_telemetry is not None:
        cluster_score_telemetry["incumbent"] = incumbent_score_telemetry
    ordering_w5_seed: Optional[torch.Tensor] = None
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
    incumbent_pair: Optional["W5ScorePair"] = None
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
        ordering_rank_to_nodes = _rank_to_nodes_from_incumbent_y(
            incumbent.detach().to(device="cpu", dtype=torch.float32),
            problem.edge_index,
            n,
        )
        ordering_portfolio_admissible = n <= DIRECTED_ORDERING_MEDIUM_NODE_CAP
        ordering_max_passes = _ordering_portfolio_max_passes(n)
        if (
            ordering_portfolio_admissible
            and n <= DIRECTED_ORDERING_MEDIUM_NODE_CAP
            and _ordering_cost_admissible(
                n,
                edge_count,
                ordering_rank_to_nodes,
                max_passes=ordering_max_passes,
            )
        ):
            try:
                candidate_started = time.perf_counter()
                incumbent_crossings = _exact_crossing_count(incumbent, problem.edge_index)
                candidate = _rank_local_zero_crossing_swap_candidate(
                    incumbent,
                    problem.edge_index,
                    max_passes=ordering_max_passes,
                    config=config,
                )
                candidate_crossings = _exact_crossing_count(candidate, problem.edge_index)
                if candidate_crossings < incumbent_crossings and not torch.equal(
                    candidate, incumbent.detach().to(device="cpu")
                ):
                    # The discrete ordering arm is intentionally stricter
                    # than legacy directed challengers: it may enter the
                    # winner contest only when it already beats the incumbent
                    # under both frozen rulers, matching the W5 contract.
                    if incumbent_pair is None:
                        incumbent_pair = _score_directed_candidate_pair(
                            incumbent,
                            problem,
                            cluster_ids,
                            all_pairs_dist,
                        )
                    dominates, candidate_pair = _directed_ordering_candidate_dual_dominates(
                        candidate,
                        incumbent_pair,
                        problem,
                        cluster_ids,
                        all_pairs_dist,
                    )
                    if dominates:
                        if n <= DIRECTED_ORDERING_W5_NODE_CAP:
                            ordering_w5_seed = candidate
                        name = "rank_local_zero_crossing_swap"
                        positions[name] = candidate
                        scores[name] = candidate_pair.directed
                        arm_timings[name] = (candidate_started, time.perf_counter())
            except Exception as exc:  # noqa: BLE001 -- challengers cannot sink the incumbent
                _reraise_worker_timeout(exc)
                _LOGGER.warning("directed rank-local swap challenger failed", exc_info=True)
    if _directed_recombinant_layered_enabled(problem) and _predicted_arm_budget_available(
        config,
        DIRECTED_RECOMBINANT_PRIOR_S,
    ):
        try:
            incumbent_pair = _register_recombinant_layered_candidates(
                problem=problem,
                incumbent=incumbent,
                config=config,
                positions=positions,
                scores=scores,
                incumbent_pair=incumbent_pair,
                cluster_ids=cluster_ids,
                all_pairs_dist=all_pairs_dist,
                arm_timings=arm_timings,
            )
        except Exception as exc:  # noqa: BLE001 -- recombinant candidates cannot sink incumbent
            _reraise_worker_timeout(exc)
            _LOGGER.warning("directed recombinant layered challenger failed", exc_info=True)
    if _portfolio_has_budget(config):
        try:
            from dagua.layout.ops.pipelines.sugiyama import layout_sugiyama_pipeline

            sugiyama_cost = _directed_opaque_arm_cost(
                problem,
                config,
                DIRECTED_SUGIYAMA_SIMPLEX_PRIOR_S,
            )
            sugiyama_cost_s = sugiyama_cost.generation_dwu + sugiyama_cost.reserved_score_dwu
            candidate_started = time.perf_counter()
            candidate_started_process = time.process_time()
            if not _predicted_arm_budget_available(
                config, sugiyama_cost_s
            ) or not admit_native_work(
                config,
                sugiyama_cost,
                "optional_directed_sugiyama_cluster_dotx",
            ):
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
                cluster_cpu_s = _prediction_cpu_elapsed_s(candidate_started_process)
                _LOGGER.info(
                    "Directed candidate runtime family=sugiyama arm=cluster_dotx cpu_seconds=%.3f",
                    cluster_cpu_s,
                )
                run_remaining_sugiyama = True
                if not _predicted_arm_budget_available(
                    config, sugiyama_cost_s
                ) or not admit_native_work(
                    config,
                    sugiyama_cost,
                    "optional_directed_sugiyama_point_unit_dotx",
                ):
                    _LOGGER.info("Skipped directed point-unit dot-x: insufficient predicted budget")
                    run_remaining_sugiyama = False
                else:
                    candidate_started = time.perf_counter()
                    candidate_started_process = time.process_time()
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
                    point_cpu_s = _prediction_cpu_elapsed_s(candidate_started_process)
                    _LOGGER.info(
                        "Directed candidate runtime family=sugiyama arm=point_unit_dotx "
                        "cpu_seconds=%.3f",
                        point_cpu_s,
                    )
                    for mode in ("graphviz_dot", "igraph"):
                        if (
                            not _portfolio_has_budget(config)
                            or not _predicted_arm_budget_available(config, sugiyama_cost_s)
                            or not admit_native_work(
                                config,
                                sugiyama_cost,
                                f"optional_directed_sugiyama_{mode}",
                            )
                        ):
                            break
                        candidate_started = time.perf_counter()
                        candidate_started_process = time.process_time()
                        mode_candidate = layout_sugiyama_pipeline(
                            edge_index=cpu_edges,
                            num_nodes=n,
                            node_sizes=cpu_sizes,
                            seed=seed,
                            edge_weights=cpu_weights,
                            fidelity_mode=mode,
                            clusters=problem.clusters,
                            cluster_parents=problem.cluster_parents,
                        )
                        if not isinstance(mode_candidate, torch.Tensor):
                            raise RuntimeError(f"{mode} Sugiyama returned non-position output")
                        _register_challenger_variants(
                            mode,
                            mode_candidate,
                            problem,
                            config,
                            positions,
                            arm_timings=arm_timings,
                            timing_span=(candidate_started, time.perf_counter()),
                        )
                        sibling_cpu_s = _prediction_cpu_elapsed_s(candidate_started_process)
                        _LOGGER.info(
                            "Directed candidate runtime family=sugiyama arm=%s cpu_seconds=%.3f",
                            mode,
                            sibling_cpu_s,
                        )
                if run_remaining_sugiyama and _full_sugiyama_grid_enabled(problem, config):
                    # The full spacing grid remains exact for small DAGs. At
                    # n>=250 it runs only when structural expansion and remaining
                    # budget leave enough space for the already-returnable incumbent.
                    for mode in SUGIYAMA_FIDELITY_MODES:
                        for rank_sep in SUGIYAMA_RANK_SEP_GRID:
                            for node_sep in SUGIYAMA_NODE_SEP_GRID:
                                grid_name = f"{mode}_r{rank_sep:g}_n{node_sep:g}"
                                if (
                                    not _portfolio_has_budget(config)
                                    or not _predicted_arm_budget_available(config, sugiyama_cost_s)
                                    or not admit_native_work(
                                        config,
                                        sugiyama_cost,
                                        f"optional_directed_sugiyama_grid_{grid_name}",
                                    )
                                ):
                                    break
                                candidate_started = time.perf_counter()
                                candidate_started_process = time.process_time()
                                grid_candidate = layout_sugiyama_pipeline(
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
                                if not isinstance(grid_candidate, torch.Tensor):
                                    raise RuntimeError(
                                        f"{grid_name} Sugiyama returned non-position output"
                                    )
                                if mode == "igraph":
                                    # Match the reference adapter's fixed conversion from
                                    # igraph coordinate units into renderer point units.
                                    grid_candidate = grid_candidate * IGRAPH_OUTPUT_SCALE
                                _register_challenger_variants(
                                    grid_name,
                                    grid_candidate,
                                    problem,
                                    config,
                                    positions,
                                    arm_timings=arm_timings,
                                    timing_span=(candidate_started, time.perf_counter()),
                                )
                                grid_cpu_s = _prediction_cpu_elapsed_s(candidate_started_process)
                                _LOGGER.info(
                                    "Directed candidate runtime family=sugiyama arm=%s "
                                    "cpu_seconds=%.3f",
                                    grid_name,
                                    grid_cpu_s,
                                )
        except Exception as exc:  # noqa: BLE001 -- challengers cannot sink the incumbent
            _reraise_worker_timeout(exc)
            _LOGGER.warning("directed Sugiyama challenger failed", exc_info=True)

    force_gate = _force_challengers_enabled(problem.edge_index, n)
    if force_gate:
        try:
            from dagua.layout.ops.pipelines.fcose import layout_fcose_pipeline
            from dagua.layout.ops.pipelines.native_undirected import FCOSE_CONTEST_SEEDS

            force_cost = estimate_native_work_cost(
                problem,
                "fcose",
                {"steps": 250, "samples": None},
                _native_device_class(config),
            )
            force_cost_s = force_cost.generation_dwu + force_cost.reserved_score_dwu
            for seed_offset in range(FCOSE_CONTEST_SEEDS):
                if (
                    not _portfolio_has_budget(config)
                    or (n >= 120 and not _predicted_arm_budget_available(config, force_cost_s))
                    or not admit_native_work(
                        config,
                        force_cost,
                        f"optional_directed_fcose_seed{seed_offset}",
                    )
                ):
                    break
                candidate_started_process = time.process_time()
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
                force_cpu_s = _prediction_cpu_elapsed_s(candidate_started_process)
                _LOGGER.info(
                    "Directed candidate runtime family=fcose seed=%d cpu_seconds=%.3f",
                    seed_offset,
                    force_cpu_s,
                )
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
        if cluster_ids is not None:
            reserved_cluster_name = next(
                (
                    name
                    for name in challenger_names
                    if name not in finalist_names
                    and _directed_candidate_family(name)
                    not in {
                        _directed_candidate_family(finalist)
                        for finalist in finalist_names
                        if finalist != "incumbent"
                    }
                ),
                None,
            )
            if reserved_cluster_name is not None:
                finalist_names.append(reserved_cluster_name)
    for name in finalist_names:
        if name == "incumbent":
            continue
        if not problem.clusters:
            if name in scores:
                continue
            scores[name] = _score_directed_candidate_cached(
                positions[name],
                problem,
                cluster_ids,
                all_pairs_dist,
            )
            continue
        if name in scores and name in cluster_score_telemetry:
            continue
        score, score_telemetry = _score_directed_candidate_referee_payload(
            positions[name],
            problem,
            cluster_ids,
            all_pairs_dist,
        )
        if name not in scores:
            scores[name] = score
        if score_telemetry is not None:
            cluster_score_telemetry[name] = score_telemetry
    best_name = _select_directed_winner(scores, cluster_score_telemetry)
    best_position = positions[best_name]
    if best_name != "incumbent" and _portfolio_has_budget(config, min_remaining_s=2.0):
        edge_count = int(problem.edge_index.shape[1]) if problem.edge_index.numel() else 0
        best_cpu = best_position.detach().to(device="cpu", dtype=torch.float32)
        best_rank_to_nodes = _rank_to_nodes_from_incumbent_y(best_cpu, problem.edge_index, n)
        ordering_portfolio_admissible = n <= DIRECTED_ORDERING_MEDIUM_NODE_CAP
        ordering_max_passes = _ordering_portfolio_max_passes(n)
        if (
            ordering_portfolio_admissible
            and n <= DIRECTED_ORDERING_MEDIUM_NODE_CAP
            and _ordering_cost_admissible(
                n,
                edge_count,
                best_rank_to_nodes,
                max_passes=ordering_max_passes,
            )
        ):
            try:
                candidate_started = time.perf_counter()
                best_crossings = _exact_crossing_count(best_cpu, problem.edge_index)
                candidate = _rank_local_zero_crossing_swap_candidate(
                    best_cpu,
                    problem.edge_index,
                    max_passes=ordering_max_passes,
                    config=config,
                )
                candidate_crossings = _exact_crossing_count(candidate, problem.edge_index)
                if candidate_crossings < best_crossings and not torch.equal(candidate, best_cpu):
                    if best_name == "incumbent":
                        if incumbent_pair is None:
                            incumbent_pair = _score_directed_candidate_pair(
                                incumbent,
                                problem,
                                cluster_ids,
                                all_pairs_dist,
                            )
                        best_pair_for_ordering = incumbent_pair
                    else:
                        best_pair_for_ordering = _score_directed_candidate_pair(
                            best_position,
                            problem,
                            cluster_ids,
                            all_pairs_dist,
                        )
                    dominates, candidate_pair = _directed_ordering_candidate_dual_dominates(
                        candidate,
                        best_pair_for_ordering,
                        problem,
                        cluster_ids,
                        all_pairs_dist,
                    )
                    if dominates:
                        name = f"{best_name}_rank_local_zero_crossing_swap"
                        positions[name] = candidate
                        scores[name] = candidate_pair.directed
                        arm_timings[name] = (candidate_started, time.perf_counter())
                        best_name = name
                        best_position = candidate
            except Exception as exc:  # noqa: BLE001 -- late ordering cannot sink the winner
                _reraise_worker_timeout(exc)
                _LOGGER.warning("directed late rank-local swap challenger failed", exc_info=True)
    if ordering_w5_seed is not None and not bool(getattr(config, "_dagua_native_defer_w5", False)):
        try:
            from dagua.layout.ops.pipelines.native_finisher import (
                W5Seed,
                log_w5_telemetry,
                run_w5_finisher,
                w5_dominates,
            )

            if best_name == "incumbent":
                best_pair, best_axes = _score_directed_candidate_payload(
                    incumbent,
                    problem,
                    cluster_ids,
                    all_pairs_dist,
                )
            else:
                best_pair, best_axes = _score_directed_candidate_payload(
                    best_position,
                    problem,
                    cluster_ids,
                    all_pairs_dist,
                )

            def score_w5_candidate(pos: torch.Tensor) -> "W5ScorePair":
                """Score one W5 checkpoint under both frozen rulers.

                Parameters
                ----------
                pos : torch.Tensor
                    Candidate positions with shape ``[N, 2]``.

                Returns
                -------
                W5ScorePair
                    Directed and undirected frozen-ruler scores.
                """
                return _score_directed_candidate_pair(pos, problem, cluster_ids, all_pairs_dist)

            w5_sizes = (
                cpu_sizes
                if cpu_sizes is not None
                else torch.full((n, 2), float(config.node_sep), dtype=torch.float32)
            )
            w5_result = run_w5_finisher(
                incumbent_pos=best_position,
                incumbent_score_pair=best_pair,
                seeds=[
                    W5Seed("directed_winner", best_position),
                    W5Seed(
                        "directed_ordering",
                        ordering_w5_seed.to(device=best_position.device, dtype=best_position.dtype),
                    ),
                    W5Seed(
                        "directed_incumbent",
                        incumbent.to(device=best_position.device, dtype=best_position.dtype),
                    ),
                ],
                edge_index=cpu_edges,
                node_sizes=w5_sizes,
                score_fn=score_w5_candidate,
                is_semantically_directed=True,
                declared_hierarchical=True,
                direction_is_declared=True,
                config=config,
                incumbent_axes=best_axes,
            )
            log_w5_telemetry(w5_result, config)
            if w5_result.accepted and w5_dominates(w5_result.winner_score_pair, best_pair, 0.05):
                best_name = w5_result.winner_name
                best_position = w5_result.winner_pos
                scores[best_name] = w5_result.winner_score_pair.directed
                positions[best_name] = best_position
        except Exception as exc:  # noqa: BLE001 -- W5 warm starts cannot sink the incumbent
            _reraise_worker_timeout(exc)
            _LOGGER.warning("directed ordering W5 warm start failed", exc_info=True)
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
    _append_terminal_w5_seed(config, "directed_candidate_a", incumbent)
    if ordering_w5_seed is not None:
        _append_terminal_w5_seed(config, "directed_ordering", ordering_w5_seed)
    for seed_rank, seed_name in enumerate(
        sorted(scores, key=lambda name: (-scores[name], name))[:3],
        start=1,
    ):
        _append_terminal_w5_seed(
            config,
            f"directed_top_{seed_rank}_{seed_name}",
            positions[seed_name],
        )
    return best_position.to(device=incumbent.device, dtype=incumbent.dtype)


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
