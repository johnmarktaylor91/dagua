"""Undirected-portfolio native route (r80-S4).

Semantically-undirected graphs (social/community/SBM/small-world/mesh/
scale-free families) are where the layered native default loses most of its
benchmark comparisons to external force engines. This route runs a small
candidate CONTEST instead of betting on one pipeline:

- Candidate A (incumbent): whatever ``_choose_native_pipeline_baseline``
  would have selected if this route did not exist, run through the normal
  native path (including its own polish battery). This guarantees the route
  can never do worse than the pre-portfolio router wherever selection is
  honest.
- Candidate B: dagua's own bit-faithful sfdp reimplementation on the same
  problem tensors, finished with size-aware overlap projection.
- Candidate C: dagua's own bit-faithful neato reimplementation + projection,
  gated by the quality knob (see ``_neato_in_contest``).
- Candidate D (r80-S9, cluster-aware graphs only): the recursive
  ``ClusterAwareDriver`` running an sfdp inner pipeline, so clustered-
  undirected graphs get a candidate that structurally places cluster
  hierarchy levels instead of relying solely on the composite's cluster-
  separation term (see ``_cluster_aware_sfdp_candidate``).
- Candidate E (r80-S9, weighted graphs only): the native-stress core with
  Dijkstra/pivot target distances built from similarity-transformed weights
  (``weight_transform="inverse"``) instead of the default distance
  semantics, for community/social weighted families where a heavy edge
  means "close" (see ``_weighted_similarity_candidate``).
- Candidate F (r81-P1.5): the native-stress core with target distances scaled
  into the point units used by node boxes (see ``_stress_points_candidate``).

All candidates are scored with the SAME honest composite the benchmark
harness uses for undirected rows (``metrics.full`` + ``composite_auto``
with ``is_semantically_directed=False``); argmax wins, ties go to the
incumbent. Challenger candidates additionally pass a degeneracy guard
(collapsed layouts with near-zero edge lengths or a bounding box smaller
than the nodes it must contain are rejected before the contest) so a
pathological challenger score can never launder a broken layout past the
incumbent.

No external layout binaries are invoked anywhere in this module -- the
sfdp/neato pipelines are the fidelity-campaign reimplementations
(``dagua/layout/ops/pipelines/sfdp.py`` / ``neato.py``), pure PyTorch.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Dict, Optional, Tuple

import numpy as np
import torch

from dagua.config import LayoutConfig
from dagua.layout.graph_classify import GraphFamily
from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op
from dagua.layout.projection import project_overlaps

if TYPE_CHECKING:  # pragma: no cover - typing only
    from dagua.layout.aesthetics import AestheticProfile

# Above this size the contest is skipped and the incumbent runs alone.
# Documented cap: the Stage-1 probe only produced candidate data up to 500
# nodes (see .project-context/research/r79_native/P8_PORTFOLIO_PROBE.md);
# probe data for larger graphs would be needed before raising this.
MAX_CONTEST_NODES = 1500
# Shared by the legacy native polish battery; portfolio challenger acceptance
# below is intentionally governed only by deterministic size schedules.
DEFAULT_CANDIDATE_BUDGET_S = 25.0
FULL_REFEREE_TOP_K = 8
MIN_OPTIONAL_ARM_REMAINING_S = 10.0
ABSOLUTE_DEADLINE_RESERVE_S = 5.0
PROCESS_DEADLINE_ATTR = "_dagua_native_process_deadline_s"
MAX_COLLINEAR_WORK = 100_000
MAX_DENSE_STRESS_NODES = 200
MAX_DENSE_STRESS_EDGES = 20_000
LARGE_CONTEST_NODE_THRESHOLD = 250
MID_SIZE_PRISM_NODE_THRESHOLD = 120
MID_SIZE_PRISM_MAX_DEGREE_THRESHOLD = 20
MID_SIZE_PRISM_DEGREE_UNIFORMITY_MAX = 1.0

# Candidate C (neato) participates when the public quality knob resolves to
# at least this value ("high" alias = 0.75)...
NEATO_QUALITY_THRESHOLD = 0.75
# ...OR at balanced quality throughout the measured contest range. The
# iteration schedule below bounds larger SMACOF solves instead of excluding
# the graph families where neato is the reference winner.
NEATO_BALANCED_NODE_CAP = MAX_CONTEST_NODES

# Candidate refinement schedule. The faithful 500-step SFDP solve costs
# 9-20s through 150 nodes on the r81 CPU probe. The measured structural gate
# below uses a bounded small-graph solve; explicit high quality and all other
# topology classes retain their existing schedules.
FULL_REFINEMENT_NODE_CAP = 150
FULL_REFINEMENT_STEPS = 500
BALANCED_SMALL_REFINEMENT_STEPS = 75
BALANCED_LARGE_REFINEMENT_STEPS = 10
HIGH_DEGREE_LARGE_REFINEMENT_STEPS = 20
HIGH_DEGREE_REFINEMENT_THRESHOLD = 20
NEATO_FULL_ITERATIONS = 200
NEATO_BALANCED_SMALL_ITERATIONS = 10
NEATO_MEDIUM_NODE_CAP = 250
NEATO_BALANCED_MEDIUM_ITERATIONS = 40
NEATO_BALANCED_LARGE_ITERATIONS = 4

# R83 Phase 3 common-table challengers use the exact fidelity-campaign
# defaults, independent of the public quality knob. Multiple deterministic
# seeds are bounded substitutes for the reference best-of-many field.
FCOSE_CONTEST_SEEDS = 3
FCOSE_REFERENCE_STEPS = 2500
FCOSE_PRIOR_S = 45.0
TSNET_CONTEST_SEEDS = 3
TSNET_MAX_CONTEST_NODES = 300
TSNET_REFERENCE_STEPS = 500
TSNET_PERPLEXITIES = (30.0, 5.0)
TSNET_PRIOR_S = 90.0
UNDIRECTED_PREDICTED_COST_MULTIPLIER = 2.0
FR_REFERENCE_STEPS = 50
SMALL_WORLD_SEED_NODE_MIN = 100
SMALL_WORLD_SEED_NODE_MAX = 1000
SMALL_WORLD_EDGE_NODE_RATIO_MAX = 4.0
RGG_GEOMETRIC_SEED_NODE_MIN = 100
RGG_GEOMETRIC_SEED_NODE_MAX = 1000
RGG_GEOMETRIC_EDGE_NODE_RATIO_MIN = 4.0

_LOGGER = logging.getLogger(__name__)

# Degeneracy guard thresholds (see _candidate_is_degenerate).
DEGENERACY_MIN_EDGE_TO_DIAGONAL_RATIO = 0.5
DEGENERACY_MIN_BBOX_TO_NODE_AREA_RATIO = 0.5
# Reject challenger layouts that fling ISOLATED (degree-0) nodes far from the
# layout centroid. Scoped to isolated nodes only: the r80 gate sweep proved a
# global max/median radius test also rejects legitimately-dispersed structure
# (multi_component_80 -11.0, er_500 real win -> loss -4.9, scale_free_ba_120
# -1.9). Non-isolated spread is legitimate layout structure and is not judged.
# Threshold 8.0: the pathology class is ORDER-OF-MAGNITUDE fling, not
# peripheral placement. Measured on the r80 store: legitimate isolate
# placements reach 5.4x median at most (er_500 periphery 0.5-4.8x,
# multi_component_80 tiles 2.8-2.9x), while the pathological
# random_bipartite_60 fling starts at 15.1x (measured 15-21x). 8x sits in
# the measured separation gap with margin on both sides.
DEGENERACY_MAX_ISOLATED_SPREAD_RATIO = 8.0


def _marketplace_family(candidate_name: str) -> str:
    """Return a stable family label for a marketplace candidate.

    Parameters
    ----------
    candidate_name : str
        Full candidate arm name.

    Returns
    -------
    str
        Family label used in structured telemetry.
    """
    for marker in ("_seed", "_raw", "_convergent", "_prism"):
        if marker in candidate_name:
            return candidate_name.split(marker, 1)[0]
    return candidate_name


def _portfolio_remaining_s(config: Optional[LayoutConfig]) -> Optional[float]:
    """Return remaining benchmark deadline seconds for optional portfolio work.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared native configuration, possibly carrying the benchmark
        deadline injected by ``DaguaCompetitor``.

    Returns
    -------
    float or None
        Remaining seconds, or ``None`` when no benchmark deadline is known.
    """
    deadline = getattr(config, "_dagua_native_deadline_s", None) if config is not None else None
    if deadline is None:
        return None
    return float(deadline) - time.perf_counter()


def _portfolio_process_remaining_s(config: Optional[LayoutConfig]) -> Optional[float]:
    """Return process-time seconds remaining for optional portfolio gates.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared native configuration, possibly carrying the benchmark
        deadline injected by ``DaguaCompetitor``.

    Returns
    -------
    float or None
        Remaining process CPU seconds, or ``None`` when no benchmark
        deadline is known.
    """
    if config is None:
        return None
    process_deadline = getattr(config, PROCESS_DEADLINE_ATTR, None)
    if process_deadline is None:
        remaining = _portfolio_remaining_s(config)
        if remaining is None:
            return None
        process_deadline = time.process_time() + max(0.0, float(remaining))
        setattr(config, PROCESS_DEADLINE_ATTR, process_deadline)
        return float(remaining)
    return float(process_deadline) - time.process_time()


def _portfolio_has_budget(
    config: Optional[LayoutConfig],
    min_remaining_s: float = MIN_OPTIONAL_ARM_REMAINING_S,
) -> bool:
    """Return whether another optional portfolio arm may start.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared native configuration.
    min_remaining_s : float, default=MIN_OPTIONAL_ARM_REMAINING_S
        Required remaining process-time budget before starting the arm.

    Returns
    -------
    bool
        ``True`` when there is no known deadline or enough remaining budget.
    """
    wall_remaining = _portfolio_remaining_s(config)
    if wall_remaining is not None and wall_remaining <= ABSOLUTE_DEADLINE_RESERVE_S:
        return False
    remaining = _portfolio_process_remaining_s(config)
    required_remaining = max(float(min_remaining_s), ABSOLUTE_DEADLINE_RESERVE_S)
    return remaining is None or remaining > required_remaining


def _portfolio_available_work_s(
    config: Optional[LayoutConfig],
    reserve_s: float = ABSOLUTE_DEADLINE_RESERVE_S,
) -> Optional[float]:
    """Return deadline seconds available before the hard return reserve.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared native configuration.
    reserve_s : float, default=ABSOLUTE_DEADLINE_RESERVE_S
        Seconds reserved for scoring, cleanup, and returning the best finite
        layout to the benchmark worker.

    Returns
    -------
    Optional[float]
        Seconds available for additional work, clamped to zero. ``None`` means
        no benchmark deadline is known.
    """
    wall_remaining = _portfolio_remaining_s(config)
    if wall_remaining is not None and wall_remaining <= float(reserve_s):
        return 0.0
    remaining = _portfolio_process_remaining_s(config)
    if remaining is None:
        return None
    return max(0.0, float(remaining) - float(reserve_s))


def _predicted_undirected_arm_budget_available(
    config: Optional[LayoutConfig],
    predicted_cost_s: float,
) -> bool:
    """Return whether a long optional arm may start before the deadline.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared native configuration carrying optional benchmark deadline.
    predicted_cost_s : float
        Estimated process CPU seconds for the arm.

    Returns
    -------
    bool
        ``True`` when no deadline is known or remaining budget covers the
        predicted arm cost with a conservative multiplier and return reserve.
    """
    available = _portfolio_available_work_s(config)
    if available is None:
        return True
    required = UNDIRECTED_PREDICTED_COST_MULTIPLIER * max(0.0, float(predicted_cost_s))
    return available > required


def _prediction_cpu_elapsed_s(started_process_time_s: float) -> float:
    """Return elapsed per-process CPU seconds for arm-cost prediction.

    Wall-clock time is still the hard deadline clock everywhere in this
    route. This helper is intentionally only for predicting the next
    expensive arm cost, so sibling-worker contention cannot inflate cost
    estimates and flip the admitted arm set under load.

    Parameters
    ----------
    started_process_time_s : float
        ``time.process_time()`` reading captured before the arm started.

    Returns
    -------
    float
        Non-negative per-process CPU seconds elapsed since the start.
    """
    return max(0.0, time.process_time() - float(started_process_time_s))


def _emit_undirected_arm_skip_telemetry(
    *,
    arm: str,
    reason: str,
    config: Optional[LayoutConfig],
    predicted_cost_s: Optional[float],
    remaining_s: Optional[float],
) -> None:
    """Emit structured telemetry for an undirected arm admission skip.

    Parameters
    ----------
    arm : str
        Stable arm label, such as ``"tsnet_perp30_seed0"``.
    reason : str
        Machine-readable skip reason.
    config : LayoutConfig, optional
        Config receiving ``_dagua_native_arm_skip_telemetry`` when available.
    predicted_cost_s : float, optional
        Predicted arm cost in process CPU seconds. ``None`` when unavailable.
    remaining_s : float, optional
        Remaining wall-clock deadline seconds at the skip point.

    Returns
    -------
    None
        Telemetry is logged, printed, optionally stored on ``config``, and
        appended to the configured JSONL telemetry file.
    """
    payload = {
        "event": "native_undirected_arm_skip",
        "arm": arm,
        "reason": reason,
        "predicted_cost_s": None if predicted_cost_s is None else float(predicted_cost_s),
        "remaining_s": None if remaining_s is None else float(remaining_s),
    }
    if config is not None:
        existing = list(getattr(config, "_dagua_native_arm_skip_telemetry", []))
        existing.append(payload)
        setattr(config, "_dagua_native_arm_skip_telemetry", existing)
    telemetry_path = os.environ.get("DAGUA_ARM_TELEMETRY_PATH") or os.environ.get(
        "DAGUA_W5_TELEMETRY_PATH"
    )
    if telemetry_path:
        with open(telemetry_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    print("native_undirected_arm_skip " + json.dumps(payload, sort_keys=True), flush=True)
    _LOGGER.info("Native undirected arm skip telemetry %s", json.dumps(payload, sort_keys=True))


def _record_insufficient_predicted_budget_skip(
    *,
    arm: str,
    config: Optional[LayoutConfig],
    predicted_cost_s: float,
) -> None:
    """Record an expensive-arm skip caused by predicted budget pressure.

    Parameters
    ----------
    arm : str
        Stable skipped arm name.
    config : LayoutConfig, optional
        Prepared native configuration.
    predicted_cost_s : float
        Current process-time cost prediction for the arm.

    Returns
    -------
    None
        Emits structured skip telemetry.
    """
    _emit_undirected_arm_skip_telemetry(
        arm=arm,
        reason="insufficient_predicted_budget",
        config=config,
        predicted_cost_s=predicted_cost_s,
        remaining_s=_portfolio_remaining_s(config),
    )


def _is_worker_timeout_exception(exc: Exception) -> bool:
    """Return whether an exception came from the benchmark worker alarm.

    Parameters
    ----------
    exc : Exception
        Exception raised during optional candidate work.

    Returns
    -------
    bool
        ``True`` for the benchmark timeout exception, including when the
        worker process exposes the class via ``__mp_main__``.
    """
    return type(exc).__name__ == "_WorkerLayoutTimeoutError" or (
        "worker layout timeout exceeded" in str(exc)
    )


def _reraise_worker_timeout(exc: Exception) -> None:
    """Re-raise benchmark worker timeouts instead of treating them as candidates.

    Parameters
    ----------
    exc : Exception
        Exception caught by a broad candidate-failure handler.

    Returns
    -------
    None
        Returns only for non-timeout exceptions.
    """
    if _is_worker_timeout_exception(exc):
        raise exc


def _log_marketplace_telemetry(
    *,
    route: str,
    structural_gate: str,
    positions: Dict[str, torch.Tensor],
    proxy_scores: Dict[str, float],
    full_scores: Dict[str, float],
    finalist_names: list[str],
    winner_name: str,
    started_at: float,
    arm_timings: Optional[Dict[str, Tuple[float, float]]] = None,
    started_process_at: Optional[float] = None,
    arm_process_totals: Optional[Dict[str, float]] = None,
) -> None:
    """Log structured per-arm marketplace telemetry.

    Parameters
    ----------
    route : str
        Portfolio route name, such as ``"undirected"``.
    structural_gate : str
        Structural admission summary for this route.
    positions : dict[str, torch.Tensor]
        Candidate positions keyed by arm name.
    proxy_scores : dict[str, float]
        Cheap first-stage scores keyed by arm name.
    full_scores : dict[str, float]
        Honest full-ruler scores keyed by finalist arm name.
    finalist_names : list[str]
        Arms admitted to full scoring.
    winner_name : str
        Final winning arm name.
    started_at : float
        Route ``time.perf_counter()`` timestamp.
    arm_timings : dict[str, tuple[float, float]], optional
        Per-arm ``time.perf_counter()`` start/end spans. Missing arms fall
        back to the whole route span for backward compatibility.
    started_process_at : float, optional
        Route ``time.process_time()`` timestamp for CPU-time attribution.
    arm_process_totals : dict[str, float], optional
        Per-arm process CPU totals. Missing arms fall back to the whole route
        process span for compatibility with older call sites.

    Returns
    -------
    None
        Telemetry is emitted to the module logger as JSON.
    """
    finalists = set(finalist_names)
    ended_at = time.perf_counter()
    ended_process_at = time.process_time()
    started_process = ended_process_at if started_process_at is None else float(started_process_at)
    ended_wall_time = time.time()
    started_wall_time = ended_wall_time - (ended_at - started_at)
    arms = []
    for name in sorted(positions):
        timing = None if arm_timings is None else arm_timings.get(name)
        if timing is None:
            arm_started_at = started_at
            arm_ended_at = ended_at
        else:
            arm_started_at, arm_ended_at = timing
        arm_started_wall_time = started_wall_time + max(0.0, arm_started_at - started_at)
        arm_ended_wall_time = started_wall_time + max(0.0, arm_ended_at - started_at)
        full_score = full_scores.get(name)
        process_time_s = (
            arm_process_totals.get(name)
            if arm_process_totals is not None and name in arm_process_totals
            else max(0.0, ended_process_at - started_process)
        )
        if name == winner_name:
            status = "winner"
            reason = "highest_full_score"
        elif name in finalists:
            status = "accepted"
            reason = "full_scored"
        else:
            status = "rejected"
            reason = "proxy_filtered"
        arms.append(
            {
                "name": name,
                "family": _marketplace_family(name),
                "structural_gate": structural_gate,
                "start_wall_time_s": arm_started_wall_time,
                "end_wall_time_s": arm_ended_wall_time,
                "wall_time_s": max(0.0, arm_ended_at - arm_started_at),
                "process_time_s": float(process_time_s),
                "raw_score": proxy_scores.get(name),
                "full_score": full_score,
                "status": status,
                "reason": reason,
                "final_winner": winner_name,
            }
        )
    payload = {
        "event": "native_candidate_marketplace",
        "route": route,
        "top_k": FULL_REFEREE_TOP_K,
        "winner": winner_name,
        "process_time_s": max(0.0, ended_process_at - started_process),
        "arms": arms,
    }
    _LOGGER.info("Native marketplace telemetry %s", json.dumps(payload, sort_keys=True))


def _cleanup_variants_for_size(num_nodes: int) -> Tuple[Tuple[str, Optional[bool]], ...]:
    """Return deterministic challenger cleanup variants for a graph size.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the contest problem.

    Returns
    -------
    tuple[tuple[str, bool or None], ...]
        Candidate suffixes and projection modes. ``None`` selects PRISM.
    """
    if num_nodes > LARGE_CONTEST_NODE_THRESHOLD:
        return (("_prism", None),)
    return (("", False), ("_convergent", True), ("_prism", None))


def _use_large_prism_shortlist(problem: LayoutProblem) -> bool:
    """Return whether a problem matches the corpus-backed large shortlist.

    Parameters
    ----------
    problem : LayoutProblem
        Prepared undirected layout problem.

    Returns
    -------
    bool
        Whether SFDP plus PRISM is the only retained large-graph combination.
    """
    n = int(problem.num_nodes)
    if problem.edge_weights is not None or problem.clusters:
        return False
    if problem.edge_index.numel() == 0:
        return False
    degrees = torch.bincount(problem.edge_index.flatten().to(dtype=torch.long), minlength=n)
    max_degree = int(degrees.max().item())
    if n <= LARGE_CONTEST_NODE_THRESHOLD:
        degree_uniformity = float(getattr(problem.structure, "degree_uniformity", float("inf")))
        return (
            n >= MID_SIZE_PRISM_NODE_THRESHOLD
            and max_degree > MID_SIZE_PRISM_MAX_DEGREE_THRESHOLD
            and degree_uniformity <= MID_SIZE_PRISM_DEGREE_UNIFORMITY_MAX
        )
    # Degree-four meshes retain the incumbent-derived geometry arms that win
    # grid_20x20. All measured non-mesh large winners use SFDP plus PRISM.
    return max_degree > 4


def _large_prism_shortlist_candidate(
    problem: LayoutProblem,
    config: LayoutConfig,
) -> Optional[torch.Tensor]:
    """Run the single corpus-winning large candidate combination.

    Parameters
    ----------
    problem : LayoutProblem
        Prepared undirected layout problem.
    config : LayoutConfig
        Prepared layout configuration used for the deterministic step schedule.

    Returns
    -------
    torch.Tensor or None
        Guarded SFDP-plus-PRISM positions, or ``None`` on failure.
    """
    from dagua.layout.ops.pipelines.sfdp import layout_sfdp_pipeline

    n = int(problem.num_nodes)
    seed = int(problem.seed) if problem.seed is not None else 42
    degrees = torch.bincount(problem.edge_index.flatten().to(dtype=torch.long), minlength=n)
    refinement_steps = _candidate_refinement_steps(config, n)
    if (
        refinement_steps == BALANCED_LARGE_REFINEMENT_STEPS
        and int(degrees.max().item()) > HIGH_DEGREE_REFINEMENT_THRESHOLD
    ):
        refinement_steps = HIGH_DEGREE_LARGE_REFINEMENT_STEPS
    raw_pos = layout_sfdp_pipeline(
        edge_index=problem.edge_index,
        num_nodes=n,
        node_sizes=problem.node_sizes,
        steps=refinement_steps,
        seed=seed,
        edge_weights=None,
        fidelity_mode="graphviz",
    )
    node_sep = float(getattr(config, "_dagua_native_node_sep", config.node_sep))
    repaired = _repair_flung_isolates(raw_pos, problem, node_sep)
    projected = _project_candidate_prism(repaired, problem)
    if projected is None:
        return None
    degenerate, _ = _candidate_is_degenerate(
        projected,
        problem.node_sizes,
        problem.edge_index,
    )
    return None if degenerate else projected


@dataclass(frozen=True)
class UndirectedPortfolioRouteConfig:
    """Frozen op-config for the undirected portfolio contest.

    Parameters
    ----------
    layout_config : LayoutConfig, optional
        Prepared native layout configuration used to run the incumbent and
        resolve quality/time budgets. ``None`` falls back to defaults.
    """

    layout_config: Optional[LayoutConfig] = field(default=None)


def _candidate_is_degenerate(
    pos: torch.Tensor,
    node_sizes: Optional[torch.Tensor],
    edge_index: torch.Tensor,
) -> Tuple[bool, str]:
    """Return whether a challenger layout is geometrically collapsed.

    Three symptoms are checked, any one rejects the candidate BEFORE the
    composite contest (composite terms like edge-length uniformity can score
    a geometrically broken layout deceptively well):

    1. Mean edge length below ``DEGENERACY_MIN_EDGE_TO_DIAGONAL_RATIO`` times
       the mean node bounding-box diagonal -- edges shorter than half a node
       mean the drawing cannot visually separate its endpoints.
    2. Layout bounding-box area below
       ``DEGENERACY_MIN_BBOX_TO_NODE_AREA_RATIO`` times the summed node-box
       area -- the canvas is smaller than the nodes it must contain, so
       overlap is unavoidable.
    3. Any ISOLATED (degree-0) node sits further than
       ``DEGENERACY_MAX_ISOLATED_SPREAD_RATIO`` times the median centroid
       distance from the layout centroid -- edge-based composite terms are
       blind to edgeless nodes, so a flung isolate can make the metrics call
       an illegible corner blob a win (random_bipartite_60 pathology).
       Connected-node spread is NOT judged: multi-component tilings and
       ER-periphery layouts legitimately exceed a global max/median radius
       test (r80 gate sweep regressions).

    Parameters
    ----------
    pos : torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    node_sizes : torch.Tensor, optional
        Node bounding boxes with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.

    Returns
    -------
    tuple[bool, str]
        ``(is_degenerate, reason)``. ``reason`` is empty when healthy.
    """
    n = int(pos.shape[0])
    if n <= 1 or node_sizes is None or node_sizes.numel() == 0:
        return False, ""
    # Align helper tensors to the candidate's device: pos may be on CUDA while
    # edge_index / node_sizes came from a CPU graph, in which case indexing
    # pos[edge_index] would raise a device-mismatch error.
    if edge_index is not None and edge_index.numel() > 0:
        edge_index = edge_index.to(pos.device)
    sizes = node_sizes.to(device=pos.device, dtype=pos.dtype)
    if sizes.ndim == 1:
        sizes = sizes.unsqueeze(1).expand(-1, 2)

    mean_diagonal = float(torch.linalg.vector_norm(sizes, dim=1).mean().item())
    if edge_index.numel() > 0 and mean_diagonal > 0.0:
        deltas = pos[edge_index[1]] - pos[edge_index[0]]
        mean_edge_length = float(torch.linalg.vector_norm(deltas, dim=1).mean().item())
        if mean_edge_length < DEGENERACY_MIN_EDGE_TO_DIAGONAL_RATIO * mean_diagonal:
            return True, (
                f"mean edge length {mean_edge_length:.2f} < "
                f"{DEGENERACY_MIN_EDGE_TO_DIAGONAL_RATIO} x mean node diagonal "
                f"{mean_diagonal:.2f}"
            )

    bbox_extent = pos.max(dim=0).values - pos.min(dim=0).values
    # Include node extents so a single-row layout is not falsely zero-area.
    bbox_area = float(
        ((bbox_extent[0] + sizes[:, 0].mean()) * (bbox_extent[1] + sizes[:, 1].mean())).item()
    )
    total_node_area = float((sizes[:, 0] * sizes[:, 1]).sum().item())
    if total_node_area > 0.0 and bbox_area < (
        DEGENERACY_MIN_BBOX_TO_NODE_AREA_RATIO * total_node_area
    ):
        return True, (
            f"bbox area {bbox_area:.1f} < {DEGENERACY_MIN_BBOX_TO_NODE_AREA_RATIO} x "
            f"total node-box area {total_node_area:.1f}"
        )

    # Isolated-node fling check. Judged for degree-0 nodes ONLY: a global
    # max/median test over all nodes also rejected legitimately-dispersed
    # candidates (multi-component tilings, ER periphery) in the r80 gate
    # sweep. Not applicable (ratio 0.0) when there are no isolates, every
    # node is isolated, or the median distance is zero (true collapse is
    # already covered by checks 1-2). Normally pre-empted by the
    # _repair_flung_isolates repair path; kept as a backstop should a
    # repaired (or unrepairable single-component) candidate still fling.
    spread_ratio = _max_isolated_spread_ratio(pos, edge_index)
    if spread_ratio > DEGENERACY_MAX_ISOLATED_SPREAD_RATIO:
        return True, (
            f"isolated-node centroid spread {spread_ratio:.1f}x median > "
            f"{DEGENERACY_MAX_ISOLATED_SPREAD_RATIO}x"
        )
    return False, ""


def _candidate_is_eligible(
    candidate: torch.Tensor,
    input_pos: torch.Tensor,
    node_sizes: Optional[torch.Tensor],
    edge_index: torch.Tensor,
) -> Tuple[bool, str]:
    """Check that a geometry candidate is finite, healthy, and overlap-monotone.

    Parameters
    ----------
    candidate : torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    input_pos : torch.Tensor
        Input positions with shape ``[N, 2]``.
    node_sizes : torch.Tensor, optional
        Node boxes with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edges with shape ``[2, E]``.

    Returns
    -------
    tuple[bool, str]
        Eligibility and an empty reason, or rejection and its reason.
    """
    if not bool(torch.isfinite(candidate).all().item()):
        return False, "non-finite coordinates"
    degenerate, reason = _candidate_is_degenerate(candidate, node_sizes, edge_index)
    if degenerate:
        return False, reason
    if node_sizes is not None and node_sizes.numel() > 0:
        from dagua.metrics import count_overlaps

        sizes = node_sizes.detach().to(device="cpu", dtype=torch.float32)
        before = count_overlaps(input_pos.detach().to(device="cpu"), sizes)
        after = count_overlaps(candidate.detach().to(device="cpu"), sizes)
        if after > before:
            return False, f"overlaps increased {before}->{after}"
    return True, ""


def _max_isolated_spread_ratio(pos: torch.Tensor, edge_index: torch.Tensor) -> float:
    """Return the worst isolated-node centroid-distance / median-distance ratio.

    The exact quantity the isolated-fling guard and the repair trigger
    evaluate. Returns ``0.0`` when the check does not apply: no isolated
    (degree-0) nodes, ALL nodes isolated (no connected core exists to be far
    from), or zero median distance (true collapse is covered by the other
    degeneracy checks).

    Parameters
    ----------
    pos : torch.Tensor
        Positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.

    Returns
    -------
    float
        Max isolated-node centroid distance divided by the median centroid
        distance over all nodes, or ``0.0`` when not applicable.
    """
    n = int(pos.shape[0])
    if n <= 1:
        return 0.0
    isolated_mask = torch.ones(n, dtype=torch.bool)
    if edge_index.numel() > 0:
        isolated_mask[edge_index.reshape(-1).to(dtype=torch.long)] = False
    if not bool(isolated_mask.any()) or bool(isolated_mask.all()):
        return 0.0
    centroid = pos.mean(dim=0, keepdim=True)
    centroid_distances = torch.linalg.vector_norm(pos - centroid, dim=1)
    median_distance = float(torch.median(centroid_distances).item())
    if median_distance <= 0.0:
        return 0.0
    return float(centroid_distances[isolated_mask].max().item()) / median_distance


def _repair_flung_isolates(
    pos: torch.Tensor,
    problem: LayoutProblem,
    node_sep: float,
) -> torch.Tensor:
    """Repair isolated-node fling by re-tiling components; no-op otherwise.

    r80 round 4: packing is a REPAIR, not a default. Unconditional challenger
    packing regressed er_500 (-4.9, honest win lost) and multi_component_80
    (-11.0) whose isolates sat at a legitimate 2.8-4.8x median -- rewriting
    healthy layouts let the composite mildly prefer the original moderate
    spread. A candidate keeps its raw layout byte-identical UNLESS the
    isolated-fling trigger fires (any degree-0 node beyond
    ``DEGENERACY_MAX_ISOLATED_SPREAD_RATIO`` x median centroid distance), in
    which case each weak component keeps its raw internal geometry and the
    components are re-tiled adjacent with the shared
    ``_tile_component_positions`` tiler. Repair-then-rescore: the contest
    referee sees the repaired version.

    Parameters
    ----------
    pos : torch.Tensor
        Raw challenger positions with shape ``[N, 2]``.
    problem : LayoutProblem
        Parent layout problem (edge_index / num_nodes are read).
    node_sep : float
        Node separation passed through to the shared component tiler.

    Returns
    -------
    torch.Tensor
        ``pos`` unchanged when the trigger does not fire, else the repaired
        full-layout positions with shape ``[N, 2]``.
    """
    if _max_isolated_spread_ratio(pos, problem.edge_index) <= (
        DEGENERACY_MAX_ISOLATED_SPREAD_RATIO
    ):
        return pos

    from dagua.layout.ops.coordinate import _weak_components
    from dagua.layout.ops.pipelines._native_shared import _tile_component_positions
    from dagua.layout.ops.postprocess import AspectRatioFit, AspectRatioFitConfig

    components = _weak_components(
        problem.edge_index.detach().to(device="cpu", dtype=torch.long),
        int(problem.num_nodes),
    )
    if len(components) <= 1:
        return pos
    component_results: list[tuple[torch.Tensor, torch.Tensor]] = []
    for component_nodes_list in components:
        component_nodes = torch.tensor(
            component_nodes_list,
            dtype=torch.long,
            device=pos.device,
        )
        component_results.append((component_nodes, pos[component_nodes]))
    tiled_positions = _tile_component_positions(component_results, node_sep=node_sep)
    fit_state = AspectRatioFit(AspectRatioFitConfig()).apply(
        problem,
        SolveState(pos=tiled_positions),
        RuntimeContext(),
    )
    if fit_state.pos is None:
        raise RuntimeError("isolate-fling repair did not produce positions.")
    repaired = fit_state.pos.detach()
    # The guard judges isolate radius against the all-node median radius, so
    # make the repair target that exact geometry after component tiling.
    # Recompute because moving isolates also shifts the all-node centroid.
    isolated_mask = torch.ones(int(problem.num_nodes), dtype=torch.bool, device=repaired.device)
    if problem.edge_index.numel() > 0:
        isolated_mask[
            problem.edge_index.reshape(-1).to(device=repaired.device, dtype=torch.long)
        ] = False
    target_ratio = DEGENERACY_MAX_ISOLATED_SPREAD_RATIO * 0.95
    for _iteration in range(4):
        centroid = repaired.mean(dim=0, keepdim=True)
        distances = torch.linalg.vector_norm(repaired - centroid, dim=1)
        median_distance = float(torch.median(distances).item())
        if median_distance <= 0.0:
            break
        limit = target_ratio * median_distance
        far_mask = isolated_mask & (distances > limit)
        if not bool(far_mask.any()):
            break
        vectors = repaired[far_mask] - centroid
        repaired[far_mask] = centroid + vectors * (limit / distances[far_mask]).unsqueeze(1)
    return repaired


def _build_cluster_ids(problem: LayoutProblem) -> Optional[torch.Tensor]:
    """Reconstruct per-node cluster ids from problem cluster metadata.

    Mirrors ``DaguaGraph.cluster_ids`` (deepest assignment wins, indices
    follow sorted cluster-name order) so the layout-time composite sees the
    same cluster-separation term the benchmark scorer sees. Nested-dict
    cluster values fall back to leaf collection.

    Parameters
    ----------
    problem : LayoutProblem
        Problem carrying optional ``clusters`` metadata.

    Returns
    -------
    torch.Tensor | None
        Cluster ids with shape ``[N]`` or ``None`` when no clusters exist.
    """
    clusters = problem.clusters
    if not clusters or problem.num_nodes == 0:
        return None
    try:
        from dagua.utils import collect_cluster_leaves

        parents = problem.cluster_parents or {}

        def _depth(name: str) -> int:
            depth = 0
            current: Optional[str] = name
            seen = set()
            while current is not None and current not in seen:
                seen.add(current)
                current = parents.get(current)
                depth += 1
            return depth

        ids = torch.full((problem.num_nodes,), -1, dtype=torch.long)
        node_depth = [-1] * problem.num_nodes
        cluster_name_list = sorted(clusters.keys())
        name_to_idx = {name: index for index, name in enumerate(cluster_name_list)}
        for name in cluster_name_list:
            members = clusters[name]
            if isinstance(members, dict):
                members = collect_cluster_leaves(members)
            depth = _depth(name)
            for node_idx in members:
                node_int = int(node_idx)
                if 0 <= node_int < problem.num_nodes and depth > node_depth[node_int]:
                    ids[node_int] = name_to_idx[name]
                    node_depth[node_int] = depth
        return ids
    except Exception:  # noqa: BLE001 -- scoring must not crash the solve
        return None


def _score_undirected_candidate(
    pos: torch.Tensor,
    problem: LayoutProblem,
    cluster_ids: Optional[torch.Tensor],
    aesthetic_profile: Optional["AestheticProfile"] = None,
    all_pairs_dist: Optional[np.ndarray] = None,
) -> float:
    """Score one candidate with the benchmark's honest undirected composite.

    Uses ``metrics.full`` (tier the benchmark uses for graphs under its full
    cutoff -- the contest node cap keeps us in that regime) and
    ``composite_auto(..., is_semantically_directed=False)``. ``full`` is
    self-deterministic for fixed positions (sampled crossing rate seeds its
    own generator), so selection is reproducible.

    r80-S8: when ``aesthetic_profile`` is ``None`` (the default, unset knob)
    this calls ``composite_auto`` exactly as before -- no wrapper, no extra
    float ops, bit-identical to pre-knob behavior. When a profile is
    resolved, every candidate in the contest is scored with
    ``dagua.layout.aesthetics.reweighted_composite`` and that SAME profile
    object (see ``layout_native_undirected_portfolio``), which is required
    for contest fairness.

    Parameters
    ----------
    pos : torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    problem : LayoutProblem
        Problem carrying topology and node sizes.
    cluster_ids : torch.Tensor, optional
        Optional per-node cluster ids for the cluster-separation term.
    aesthetic_profile : AestheticProfile, optional
        Resolved aesthetic-priority profile shared by every candidate in
        the current contest. ``None`` preserves the exact pre-knob scoring
        path.
    all_pairs_dist : Optional[numpy.ndarray], optional
        Cached unweighted shortest paths with shape ``[N, N]``.

    Returns
    -------
    float
        Higher-is-better undirected composite score.
    """
    from dagua.metrics import composite_auto, full

    metrics = full(
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
    numeric = {
        key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))
    }
    if aesthetic_profile is None:
        return float(composite_auto(numeric, is_semantically_directed=False))

    from dagua.layout.aesthetics import reweighted_composite

    return reweighted_composite(numeric, is_directed=False, profile=aesthetic_profile)


def _proxy_undirected_candidate(
    pos: torch.Tensor,
    problem: LayoutProblem,
    cluster_ids: Optional[torch.Tensor],
    all_pairs_dist: Optional[np.ndarray] = None,
) -> float:
    """Return a deterministic cheap proxy for portfolio shortlisting.

    Parameters
    ----------
    pos : torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    problem : LayoutProblem
        Problem carrying topology and node sizes.
    cluster_ids : torch.Tensor, optional
        Optional per-node cluster ids.
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
    return float(composite_auto(numeric, is_semantically_directed=False))


def _score_undirected_candidate_cached(
    pos: torch.Tensor,
    problem: LayoutProblem,
    cluster_ids: Optional[torch.Tensor],
    aesthetic_profile: Optional["AestheticProfile"],
    all_pairs_dist: Optional[np.ndarray],
) -> float:
    """Score an undirected candidate while preserving old monkeypatch arity.

    Parameters
    ----------
    pos : torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    problem : LayoutProblem
        Problem carrying topology and node sizes.
    cluster_ids : torch.Tensor, optional
        Optional per-node cluster ids.
    aesthetic_profile : AestheticProfile, optional
        Shared contest aesthetic profile.
    all_pairs_dist : Optional[numpy.ndarray]
        Cached unweighted shortest paths with shape ``[N, N]``.

    Returns
    -------
    float
        Higher-is-better full composite score.
    """
    try:
        return _score_undirected_candidate(
            pos,
            problem,
            cluster_ids,
            aesthetic_profile,
            all_pairs_dist=all_pairs_dist,
        )
    except TypeError as exc:
        if "all_pairs_dist" not in str(exc):
            raise
        return _score_undirected_candidate(pos, problem, cluster_ids, aesthetic_profile)


# Convergent-cleanup pass budget for challenger candidates. The convergent
# exact projector early-exits at zero overlaps or on measured stagnation,
# so this ceiling is only consumed on hard overlap fields; the contest cap
# (MAX_CONTEST_NODES) bounds the per-pass O(N^2) cost.
CHALLENGER_PROJECTION_ITERATIONS = 200
PRISM_ZERO_MAX_ITERATIONS = 4
PRISM_SCALE_MARGIN = 1.001


def _candidate_refinement_steps(config: Optional[LayoutConfig], num_nodes: int) -> int:
    """Return the quality-scaled force refinement budget.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared public layout configuration.
    num_nodes : int
        Number of nodes in the contest problem.

    Returns
    -------
    int
        SFDP refinement steps for the candidate solve.
    """
    if _resolved_quality(config) >= NEATO_QUALITY_THRESHOLD:
        return FULL_REFINEMENT_STEPS
    if num_nodes <= FULL_REFINEMENT_NODE_CAP:
        return FULL_REFINEMENT_STEPS
    return BALANCED_LARGE_REFINEMENT_STEPS


def _neato_iterations(config: Optional[LayoutConfig], num_nodes: int) -> int:
    """Return the quality-scaled neato SMACOF iteration budget.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared public layout configuration.
    num_nodes : int
        Number of nodes in the contest problem.

    Returns
    -------
    int
        Maximum SMACOF iterations for the candidate solve.
    """
    if _resolved_quality(config) >= NEATO_QUALITY_THRESHOLD:
        return NEATO_FULL_ITERATIONS
    if num_nodes <= FULL_REFINEMENT_NODE_CAP:
        return NEATO_FULL_ITERATIONS
    if num_nodes <= NEATO_MEDIUM_NODE_CAP:
        return NEATO_BALANCED_MEDIUM_ITERATIONS
    return NEATO_BALANCED_LARGE_ITERATIONS


def _overlap_pairs(pos: torch.Tensor, node_sizes: torch.Tensor) -> torch.Tensor:
    """Return upper-triangle pairs whose axis-aligned node boxes overlap.

    Parameters
    ----------
    pos : torch.Tensor
        Positions with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Overlapping node-index pairs with shape ``[K, 2]``.
    """
    deltas = torch.abs(pos.unsqueeze(1) - pos.unsqueeze(0))
    required = (node_sizes.unsqueeze(1) + node_sizes.unsqueeze(0)) * 0.5
    overlaps = (deltas[..., 0] < required[..., 0]) & (deltas[..., 1] < required[..., 1])
    return torch.nonzero(torch.triu(overlaps, diagonal=1), as_tuple=False)


def _scale_past_residual_overlaps(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    overlap_pairs: torch.Tensor,
) -> torch.Tensor:
    """Uniformly scale a layout just past its remaining overlap pairs.

    Parameters
    ----------
    pos : torch.Tensor
        Positions with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.
    overlap_pairs : torch.Tensor
        Residual overlapping pairs with shape ``[K, 2]``.

    Returns
    -------
    torch.Tensor
        Topology-preserving uniformly scaled positions.
    """
    source = overlap_pairs[:, 0]
    target = overlap_pairs[:, 1]
    deltas = torch.abs(pos[target] - pos[source])
    required = (node_sizes[target] + node_sizes[source]) * 0.5
    ratios = required / torch.clamp(deltas, min=torch.finfo(pos.dtype).eps)
    # A pair stops overlapping as soon as either axis clears. Only residual
    # pairs determine the smallest global scale bump, preserving all angles.
    pair_scales = torch.min(ratios, dim=1).values
    scale = float(torch.max(pair_scales).item()) * PRISM_SCALE_MARGIN
    centered = pos - pos.mean(dim=0, keepdim=True)
    return centered * scale + pos.mean(dim=0, keepdim=True)


def _project_candidate_prism(
    pos: torch.Tensor,
    problem: LayoutProblem,
) -> Optional[torch.Tensor]:
    """Apply native PRISM cleanup, failing closed on ineffective or extreme output.

    Parameters
    ----------
    pos : torch.Tensor
        Candidate positions in points with shape ``[N, 2]``.
    problem : LayoutProblem
        Problem carrying topology and node sizes.

    Returns
    -------
    torch.Tensor or None
        Cleaned positions, or ``None`` when bounded cleanup fails.
    """
    if problem.node_sizes is None or problem.node_sizes.numel() == 0:
        return pos
    input_sizes = problem.node_sizes.to(device=pos.device, dtype=pos.dtype)
    initial_overlap_count = int(_overlap_pairs(pos, input_sizes).shape[0])
    input_span = float((pos.max(dim=0).values - pos.min(dim=0).values).max().item())
    from dagua.layout.ops.pipelines.fmmm import _graphviz_fdp_prism_overlap

    points_per_inch = 72.0
    projected = _graphviz_fdp_prism_overlap(
        positions=pos.detach().to(dtype=torch.float64) / points_per_inch,
        edge_index=problem.edge_index.detach().to(device="cpu", dtype=torch.long),
        node_sizes=problem.node_sizes.detach().to(device="cpu", dtype=torch.float64),
    )
    projected = (projected * points_per_inch).to(device=pos.device, dtype=torch.float32)
    sizes = problem.node_sizes.to(device=projected.device, dtype=projected.dtype)
    for _iteration in range(PRISM_ZERO_MAX_ITERATIONS):
        pairs = _overlap_pairs(projected, sizes)
        if pairs.numel() == 0:
            break
        projected = _scale_past_residual_overlaps(projected, sizes, pairs)
    residual_overlap_count = int(_overlap_pairs(projected, sizes).shape[0])
    coordinate_bound = 1.0e6 * max(input_span, 1.0)
    if (
        not bool(torch.isfinite(projected).all().item())
        or residual_overlap_count >= initial_overlap_count > 0
        or float(torch.abs(projected).max().item()) > coordinate_bound
    ):
        return None
    return projected


def _project_candidate(
    pos: torch.Tensor,
    problem: LayoutProblem,
    convergent: bool = False,
) -> torch.Tensor:
    """Apply size-aware overlap projection to one challenger candidate.

    Two cleanup variants exist and NEITHER dominates (r80-S2b petersen_10
    bisect, P7_PROJECTOR_EVIDENCE.md):

    - ``convergent=False``: the legacy projector call the S4 portfolio
      shipped with (default padding/iterations). Its last-write-wins pushes
      stall on dense overlap fields, but its trajectory produced the
      trunk's flagship wins (petersen_10 79.0, weighted_karate_34 69.5,
      weighted_clusters_3x10 68.1 -- all legacy-cleaned neato candidates).
    - ``convergent=True``: the accumulate+damp+deadlock-re-lay projector
      with a generous early-exit ceiling. Provably reaches zero overlaps
      on dense cliques the legacy path stalls on (P3B2 forensics) and
      produced the S2b sweep gains (planar_60 +19.9, regular_4_40 +15.4,
      weighted_clusters sfdp +21.4 over legacy).

    The contest therefore scores BOTH variants as separate candidates --
    never replacing one with the other -- and lets the honest-composite
    referee choose (the S2b regression came from replacing the legacy
    variant instead of adding the convergent one alongside it).

    Parameters
    ----------
    pos : torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    problem : LayoutProblem
        Problem carrying node sizes.
    convergent : bool, default=False
        Select the convergent cleanup variant.

    Returns
    -------
    torch.Tensor
        Overlap-projected positions (new tensor).
    """
    if problem.node_sizes is None or problem.node_sizes.numel() == 0:
        return pos
    projected = pos.detach().clone().to(dtype=torch.float32)
    node_sizes = problem.node_sizes.to(device=projected.device, dtype=projected.dtype)
    if convergent:
        project_overlaps(
            projected,
            node_sizes,
            iterations=CHALLENGER_PROJECTION_ITERATIONS,
            convergent=True,
        )
    else:
        project_overlaps(projected, node_sizes)
    return projected


def _resolved_quality(config: Optional[LayoutConfig]) -> float:
    """Return the normalized [0, 1] quality value from a prepared config.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration.

    Returns
    -------
    float
        Normalized quality; 0.5 (balanced) when unavailable.
    """
    if config is None:
        return 0.5
    try:
        return float(getattr(config, "quality", 0.5))
    except (TypeError, ValueError):
        return 0.5


def _neato_in_contest(config: Optional[LayoutConfig], num_nodes: int) -> bool:
    """Return whether candidate C (neato + projection) joins the contest.

    Two admission paths:

    1. Quality >= high (0.75): neato always joins (up to the contest cap).
    2. Balanced/lower quality with ``num_nodes <= NEATO_BALANCED_NODE_CAP``:
       the Stage-1 probe (P8_PORTFOLIO_PROBE.md) shows every balanced-quality
       contest neato ever wins sits at n <= 80, where its SMACOF loop
       epsilon-exits in <= ~8s; on larger graphs it costs 40-150s and never
       beat the sfdp/incumbent winner in any probe row. The cap keeps the
       neato wins (karate, lattices, grids, petersen, multi-component)
       inside the default wall-time envelope and leaves the slow never-wins
       region to the explicit quality knob.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration.
    num_nodes : int
        Number of nodes in the current problem.

    Returns
    -------
    bool
        ``True`` when candidate C should run.
    """
    if _resolved_quality(config) >= NEATO_QUALITY_THRESHOLD:
        return True
    return num_nodes <= NEATO_BALANCED_NODE_CAP


def _small_world_knn_seed_enabled(problem: LayoutProblem) -> bool:
    """Return whether the local kNN seed should enter the contest.

    Parameters
    ----------
    problem : LayoutProblem
        Undirected portfolio problem.

    Returns
    -------
    bool
        ``True`` for medium sparse cyclic small-world-like topology.
    """
    n = int(problem.num_nodes)
    if n < SMALL_WORLD_SEED_NODE_MIN or n > SMALL_WORLD_SEED_NODE_MAX:
        return False
    edge_count = int(problem.edge_index.shape[1]) if problem.edge_index.numel() else 0
    if edge_count == 0:
        return False
    edge_ratio = float(edge_count) / float(max(n, 1))
    structure = problem.structure
    max_degree = int(getattr(structure, "max_degree", 0)) if structure is not None else 0
    hub_fraction = (
        float(getattr(structure, "hub_edge_fraction", 1.0)) if structure is not None else 1.0
    )
    is_cyclic = not bool(getattr(structure, "is_directed_acyclic", True))
    return (
        is_cyclic
        and max_degree <= 12
        and hub_fraction <= 0.25
        and 1.5 <= edge_ratio <= SMALL_WORLD_EDGE_NODE_RATIO_MAX
    )


def _rgg_geometric_seed_enabled(problem: LayoutProblem) -> bool:
    """Return whether the geometric sparse-stress seed should enter.

    Parameters
    ----------
    problem : LayoutProblem
        Undirected portfolio problem.

    Returns
    -------
    bool
        ``True`` for medium dense spatial/geometric proxy structure.
    """
    n = int(problem.num_nodes)
    if n < RGG_GEOMETRIC_SEED_NODE_MIN or n > RGG_GEOMETRIC_SEED_NODE_MAX:
        return False
    edge_count = int(problem.edge_index.shape[1]) if problem.edge_index.numel() else 0
    edge_ratio = float(edge_count) / float(max(n, 1))
    structure = problem.structure
    if structure is None:
        return False
    diameter = int(getattr(structure, "diameter_estimate", 0))
    communities = int(getattr(structure, "num_communities", 0))
    return (
        edge_ratio >= RGG_GEOMETRIC_EDGE_NODE_RATIO_MIN
        and int(getattr(structure, "max_degree", 0)) <= 80
        and float(getattr(structure, "degree_uniformity", 1.0)) <= 0.45
        and float(getattr(structure, "hub_edge_fraction", 1.0)) <= 0.25
        and diameter > 0
        and float(diameter) <= 0.9 * np.sqrt(float(n))
        and float(getattr(structure, "community_score", 0.0)) >= 0.35
        and 2 <= communities <= max(2, n // 5)
    )


def _unique_undirected_pairs(edge_index: torch.Tensor) -> torch.Tensor:
    """Return sorted unique undirected edge pairs.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.

    Returns
    -------
    torch.Tensor
        Unique undirected pairs with shape ``[2, U]``.
    """
    if edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    pairs = torch.stack(
        [
            torch.minimum(edge_index[0], edge_index[1]),
            torch.maximum(edge_index[0], edge_index[1]),
        ],
        dim=0,
    ).to(device="cpu", dtype=torch.long)
    pairs = pairs[:, pairs[0] != pairs[1]]
    if pairs.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.unique(pairs.t(), dim=0).t().contiguous()


def _small_world_knn_seed_candidate(
    incumbent: torch.Tensor,
    problem: LayoutProblem,
    steps: int = 24,
) -> torch.Tensor:
    """Relax the incumbent with local-neighborhood and edge-CV forces.

    Parameters
    ----------
    incumbent : torch.Tensor
        Incumbent positions with shape ``[N, 2]``.
    problem : LayoutProblem
        Undirected portfolio problem.
    steps : int, default=24
        Deterministic local relaxation iterations.

    Returns
    -------
    torch.Tensor
        Seed positions with shape ``[N, 2]``.
    """
    pos = incumbent.detach().to(device="cpu", dtype=torch.float32).clone()
    pairs = _unique_undirected_pairs(problem.edge_index)
    if pairs.numel() == 0 or pos.shape[0] <= 2:
        return pos
    source = pairs[0]
    target = pairs[1]
    lengths = torch.linalg.vector_norm(pos[target] - pos[source], dim=1)
    target_length = float(torch.median(lengths).clamp_min(1.0).item())
    count = torch.zeros((pos.shape[0], 1), dtype=torch.float32)
    count.scatter_add_(0, source.unsqueeze(1), torch.ones((source.numel(), 1)))
    count.scatter_add_(0, target.unsqueeze(1), torch.ones((target.numel(), 1)))
    count = count.clamp_min(1.0)
    anchor = pos.clone()
    for _iteration in range(max(0, int(steps))):
        delta = pos[target] - pos[source]
        length = torch.linalg.vector_norm(delta, dim=1).clamp_min(1.0e-6)
        correction = 0.035 * ((length - target_length) / length).unsqueeze(1) * delta
        updates = torch.zeros_like(pos)
        updates.scatter_add_(0, source.unsqueeze(1).expand(-1, 2), correction)
        updates.scatter_add_(0, target.unsqueeze(1).expand(-1, 2), -correction)
        neighbor_sum = torch.zeros_like(pos)
        neighbor_sum.scatter_add_(0, source.unsqueeze(1).expand(-1, 2), pos[target])
        neighbor_sum.scatter_add_(0, target.unsqueeze(1).expand(-1, 2), pos[source])
        centroid_pull = (neighbor_sum / count) - pos
        pos = pos + updates + 0.018 * centroid_pull
        pos = anchor + 0.985 * (pos - anchor)
    return pos


def _rgg_geometric_seed_candidate(
    problem: LayoutProblem,
    seed: int,
    node_sep: float,
) -> Optional[torch.Tensor]:
    """Build the sparse/geodesic stress seed for geometric graphs.

    Parameters
    ----------
    problem : LayoutProblem
        Undirected portfolio problem.
    seed : int
        Deterministic seed.
    node_sep : float
        Node separation in points.

    Returns
    -------
    torch.Tensor or None
        Seed positions with shape ``[N, 2]``, or ``None`` when dense geodesic
        work would exceed the guard budget.
    """
    from dagua.layout.ops.pipelines.native_lattice_grid import (
        geodesic_dense_work_is_allowed,
        layout_geodesic_stress_pipeline,
    )

    edge_count = int(problem.edge_index.shape[1]) if problem.edge_index.numel() else 0
    if not geodesic_dense_work_is_allowed(int(problem.num_nodes), edge_count):
        _LOGGER.info("Skipped RGG geometric seed: geodesic dense-work guard")
        return None
    return layout_geodesic_stress_pipeline(
        edge_index=problem.edge_index,
        num_nodes=int(problem.num_nodes),
        node_sizes=problem.node_sizes,
        seed=seed,
        edge_weights=problem.edge_weights,
        node_sep=node_sep,
    )


# r80-S9 Deliverable 2: weighted-similarity Dijkstra-target transform. A
# 3-graph mini-probe (r79_weighted_small_world_120, r79_weighted_community_
# 4x18, real_lesmis_77; see P12_SQUEEZE.md) compared "inverse" (1/w) against
# an ad hoc 1/sqrt(w) transform on the raw and legacy-projected candidate
# tiers (the convergent-projector tier washed out the difference -- 200
# damped passes converge to the same overlap-free arrangement regardless of
# the small-scale stress differences between transforms). "inverse" won 2 of
# 3 graphs and never lost by more than 1.3 points on the graph it lost,
# while both transforms beat the untransformed (today's default) distance
# semantics on every graph. "inverse" is also the transform preprocess.py
# already implements (BuildAdjacencyConfig.weight_transform), so no new
# transform code is needed.
WEIGHTED_SIMILARITY_TRANSFORM = "inverse"


def _cluster_aware_sfdp_candidate(
    problem: LayoutProblem,
    config: LayoutConfig,
    ctx: RuntimeContext,
) -> Optional[torch.Tensor]:
    """Run the recursive cluster-aware driver with an sfdp inner pipeline.

    r80-S9 Deliverable 1, candidate B: clustered-undirected graphs (e.g. the
    ``r79_undirected_sbm_*`` community corpus) reach this contest today
    (the S4-era diagnosis that "the cluster driver preempts routing" no
    longer applies for the ``dagua_native``/default algorithm -- verified
    empirically, see P12_SQUEEZE.md), but candidate A (the incumbent) comes
    from the FLAT native path with cluster-separation LOSS terms only, and
    candidate B (flat sfdp, added below via ``_add_challenger``) also never
    places clusters structurally -- both rely entirely on the scoring
    composite's cluster term to reward containment after the fact. This
    candidate instead PLACES each cluster hierarchy level with dagua's sfdp
    reimplementation via the existing recursive ``ClusterAwareDriver``
    (``dagua/layout/ops/cluster_driver.py`` -- the same machinery
    ``dagua.layout.engine._layout_cluster_aware_pipeline`` uses for the
    algorithms it natively supports; ``"dagua_native"``/``None`` is not one
    of them, which is why clustered-undirected graphs never got this
    candidate before). Returns ``None`` when there are no clusters on this
    problem or the driver cannot be built.

    Parameters
    ----------
    problem : LayoutProblem
        Prepared layout problem, expected to carry cluster metadata.
    config : LayoutConfig
        Prepared native configuration.
    ctx : RuntimeContext
        Shared execution context.

    Returns
    -------
    torch.Tensor or None
        Candidate positions, or ``None`` when clusters are absent or the
        driver could not run.
    """
    if not problem.clusters:
        return None
    from dagua.layout.engine import _build_cluster_inner_pipeline
    from dagua.layout.ops.cluster_driver import ClusterAwareDriver

    inner_pipeline = _build_cluster_inner_pipeline("sfdp", config)
    if inner_pipeline is None:
        return None
    driver = ClusterAwareDriver(
        inner_pipeline=inner_pipeline.ops,
        # No DaguaGraph is available inside this headless contest to merge
        # per-graph cluster_style.padding overrides the way
        # engine._effective_cluster_side_padding does for the top-level
        # cluster driver -- this candidate uses the raw config padding
        # knobs. Documented limitation (P12_SQUEEZE.md): only affects this
        # one candidate's geometry among several scored in the contest.
        side_padding_pt=float(getattr(config, "cluster_side_padding_pt", 8.0)),
        label_band_pt=float(getattr(config, "cluster_label_band_pt", 26.0)),
        external_clearance_pt=float(getattr(config, "cluster_external_clearance_pt", 10.0)),
        cluster_compactness_weight=float(getattr(config, "w_cluster", 1.0)),
    )
    driver_state = driver.apply(problem, SolveState(), ctx)
    return driver_state.pos


def _weighted_similarity_candidate(
    problem: LayoutProblem,
    seed: int,
) -> Optional[torch.Tensor]:
    """Run the native-stress core with weights treated as similarities.

    r80-S9 Deliverable 2: for declared-undirected weighted graphs, the
    default Dijkstra/pivot target-distance costs use edge weights AS
    distances (``weight_transform="none"``) -- but for community/social
    weighted families (this contest only ever runs for declared-undirected
    graphs, exactly the family P3B2_STRESS_FORENSICS.md Ranked Fix 4 is
    about) a heavier weight usually means a STRONGER/closer relationship,
    not a longer one. This candidate reruns the native-stress core with
    ``weight_transform="inverse"`` (``1 / w``, see
    ``WEIGHTED_SIMILARITY_TRANSFORM`` for the mini-probe that picked it)
    so heavy edges pull their endpoints together. Purely additive: it is
    ONE MORE contest candidate, never a change to default weight handling
    (``NativeStressConfig.weight_transform`` defaults to ``"none"``
    everywhere else). Returns ``None`` when the problem carries no edge
    weights.

    Parameters
    ----------
    problem : LayoutProblem
        Prepared layout problem.
    seed : int
        Deterministic seed shared with the rest of the contest.

    Returns
    -------
    torch.Tensor or None
        Candidate positions, or ``None`` when there are no edge weights.
    """
    if problem.edge_weights is None:
        return None
    from dagua.layout.ops.pipelines.native_stress import (
        NativeStressConfig,
        layout_native_stress_pipeline,
    )

    return layout_native_stress_pipeline(
        edge_index=problem.edge_index,
        num_nodes=int(problem.num_nodes),
        node_sizes=problem.node_sizes,
        edge_weights=problem.edge_weights,
        seed=seed,
        config=NativeStressConfig(weight_transform=WEIGHTED_SIMILARITY_TRANSFORM, seed=seed),
    )


def _stress_points_candidate(problem: LayoutProblem, seed: int) -> torch.Tensor:
    """Run native stress with target distances expressed in points.

    Parameters
    ----------
    problem : LayoutProblem
        Prepared undirected layout problem.
    seed : int
        Deterministic seed shared with the rest of the contest.

    Returns
    -------
    torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    """
    from dagua.layout.ops.pipelines.native_stress import (
        NativeStressConfig,
        layout_native_stress_pipeline,
    )

    return layout_native_stress_pipeline(
        edge_index=problem.edge_index,
        num_nodes=int(problem.num_nodes),
        node_sizes=problem.node_sizes,
        edge_weights=problem.edge_weights,
        seed=seed,
        config=NativeStressConfig(target_unit="points", seed=seed),
    )


def _router_v2_large_mini_contest(
    baseline_pos: torch.Tensor,
    problem: LayoutProblem,
    config: LayoutConfig,
    incumbent_pos: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Score router-v2 candidates against the large-graph sfdp+PRISM holder.

    The large-graph fast path (``_use_large_prism_shortlist``) historically
    returned sfdp+PRISM without any contest. Router-v2 keeps that candidate
    as the tie-break holder and lets the structurally-shortlisted geodesic /
    community candidates challenge it under the same honest referee used by
    the full contest. Anything failing (non-finite, degenerate, raising)
    silently leaves the holder in place.

    Parameters
    ----------
    baseline_pos : torch.Tensor
        The guarded sfdp+PRISM positions with shape ``[N, 2]``.
    problem : LayoutProblem
        Prepared undirected layout problem.
    config : LayoutConfig
        Prepared native configuration.
    incumbent_pos : torch.Tensor, optional
        Exact native incumbent positions with shape ``[N, 2]``. When supplied,
        the mini-contest uses the incumbent as the tie-break finalist so W4
        seed challengers remain monotone against the frozen native route.

    Returns
    -------
    torch.Tensor
        Winning positions with shape ``[N, 2]``.
    """
    from dagua.layout.ops.pipelines.dagua_native import _undirected_route_shortlist

    started_at = time.perf_counter()
    started_process_at = time.process_time()
    n = int(problem.num_nodes)
    shortlist = _undirected_route_shortlist(
        problem.structure,
        n,
        has_edge_weights=problem.edge_weights is not None,
    )
    if not shortlist.candidates:
        _LOGGER.info("Undirected contest candidates=sfdp_prism winner=sfdp_prism")
        return baseline_pos

    seed = int(problem.seed) if problem.seed is not None else 42
    node_sep = float(getattr(config, "_dagua_native_node_sep", config.node_sep))
    aesthetic_profile: Optional["AestheticProfile"] = getattr(
        config, "_dagua_native_aesthetic_profile", None
    )
    positions: Dict[str, torch.Tensor] = {"sfdp_prism": baseline_pos}
    if incumbent_pos is not None and bool(torch.isfinite(incumbent_pos).all().item()):
        positions["incumbent"] = incumbent_pos

    def _admit(name: str, raw_pos: Optional[torch.Tensor]) -> None:
        if raw_pos is None or not bool(torch.isfinite(raw_pos).all().item()):
            return
        repaired = _repair_flung_isolates(raw_pos, problem, node_sep)
        degenerate, reason = _candidate_is_degenerate(
            repaired, problem.node_sizes, problem.edge_index
        )
        if degenerate:
            _LOGGER.info("Rejected large mini-contest candidate %s_raw: %s", name, reason)
        else:
            positions[f"{name}_raw"] = repaired
        try:
            projected = _project_candidate_prism(repaired, problem)
        except Exception as exc:  # noqa: BLE001 -- one cleanup variant fails closed
            _reraise_worker_timeout(exc)
            projected = None
        if projected is not None:
            degenerate, reason = _candidate_is_degenerate(
                projected, problem.node_sizes, problem.edge_index
            )
            if not degenerate:
                positions[f"{name}_prism"] = projected

    if "geodesic_stress" in shortlist.candidates and _portfolio_has_budget(config):
        try:
            from dagua.layout.ops.pipelines.native_lattice_grid import (
                geodesic_dense_work_is_allowed,
                layout_geodesic_stress_pipeline,
            )

            if not geodesic_dense_work_is_allowed(n, int(problem.edge_index.shape[1])):
                _LOGGER.info("Skipped large mini-contest geodesic: dense-work guard")
            else:
                _admit(
                    "geodesic_stress",
                    layout_geodesic_stress_pipeline(
                        edge_index=problem.edge_index,
                        num_nodes=n,
                        node_sizes=problem.node_sizes,
                        seed=seed,
                        edge_weights=problem.edge_weights,
                        node_sep=node_sep,
                    ),
                )
                if problem.edge_weights is not None:
                    _admit(
                        "geodesic_stress_unweighted",
                        layout_geodesic_stress_pipeline(
                            edge_index=problem.edge_index,
                            num_nodes=n,
                            node_sizes=problem.node_sizes,
                            seed=seed,
                            edge_weights=None,
                            node_sep=node_sep,
                        ),
                    )
        except Exception as exc:  # noqa: BLE001 -- a failed challenger never sinks the solve
            _reraise_worker_timeout(exc)
            _LOGGER.warning("large mini-contest geodesic challenger failed", exc_info=True)
    if "community_scaffold" in shortlist.candidates and _portfolio_has_budget(config):
        try:
            from dagua.layout.ops.pipelines.native_community import (
                layout_native_community_pipeline,
            )

            _admit(
                "community_scaffold",
                layout_native_community_pipeline(
                    edge_index=problem.edge_index,
                    num_nodes=n,
                    node_sizes=problem.node_sizes,
                    config=config,
                    seed=seed,
                    edge_weights=problem.edge_weights,
                ),
            )
        except Exception as exc:  # noqa: BLE001 -- a failed challenger never sinks the solve
            _reraise_worker_timeout(exc)
            _LOGGER.warning("large mini-contest community challenger failed", exc_info=True)
    if _small_world_knn_seed_enabled(problem) and _portfolio_has_budget(
        config,
        min_remaining_s=2.0,
    ):
        seed_base = incumbent_pos if incumbent_pos is not None else baseline_pos
        try:
            _admit("small_world_knn_seed", _small_world_knn_seed_candidate(seed_base, problem))
        except Exception as exc:  # noqa: BLE001 -- a failed challenger never sinks the solve
            _reraise_worker_timeout(exc)
            _LOGGER.warning("large mini-contest small-world seed failed", exc_info=True)
    if _rgg_geometric_seed_enabled(problem) and _portfolio_has_budget(config):
        try:
            _admit("rgg_geometric_seed", _rgg_geometric_seed_candidate(problem, seed, node_sep))
        except Exception as exc:  # noqa: BLE001 -- a failed challenger never sinks the solve
            _reraise_worker_timeout(exc)
            _LOGGER.warning("large mini-contest RGG geometric seed failed", exc_info=True)

    cluster_ids = _build_cluster_ids(problem)
    from dagua.metrics import _all_pairs_unweighted, _build_csr

    offsets, targets = _build_csr(problem.edge_index.detach().to(device="cpu"), n)
    all_pairs_dist = _all_pairs_unweighted(offsets, targets, n, max_dist=n)
    proxy_scores = {
        name: _proxy_undirected_candidate(pos, problem, cluster_ids, all_pairs_dist)
        for name, pos in positions.items()
    }
    scores = {
        name: _score_undirected_candidate_cached(
            pos,
            problem,
            cluster_ids,
            aesthetic_profile,
            all_pairs_dist,
        )
        for name, pos in positions.items()
    }
    best_name = "incumbent" if "incumbent" in positions else "sfdp_prism"
    for name, score in scores.items():
        if name != best_name and score > scores[best_name]:
            best_name = name
    _log_marketplace_telemetry(
        route="undirected_large_mini",
        structural_gate="large_prism_shortlist",
        positions=positions,
        proxy_scores=proxy_scores,
        full_scores=scores,
        finalist_names=list(scores),
        winner_name=best_name,
        started_at=started_at,
        started_process_at=started_process_at,
    )
    _LOGGER.info(
        "Undirected contest (large mini) candidates=%s winner=%s",
        ", ".join(f"{name}:{score:.3f}" for name, score in scores.items()),
        best_name,
    )
    return positions[best_name]


def layout_native_undirected_portfolio(
    problem: LayoutProblem,
    state: SolveState,
    ctx: RuntimeContext,
    config: LayoutConfig,
) -> torch.Tensor:
    """Run the undirected portfolio contest for one prepared problem.

    Parameters
    ----------
    problem : LayoutProblem
        Prepared layout problem (whole graph or one weak component).
    state : SolveState
        Incoming solve state; candidate runs receive cloned copies.
    ctx : RuntimeContext
        Shared execution context.
    config : LayoutConfig
        Prepared native configuration with ``_dagua_native_*`` metadata.

    Returns
    -------
    torch.Tensor
        Selected positions with shape ``[N, 2]``.
    """
    # Late import avoids a circular import with dagua_native (which imports
    # this module lazily at its two dispatch points).
    from dagua.layout.ops.pipelines.dagua_native import _run_native_problem

    started_at = time.perf_counter()
    started_process_at = time.process_time()

    def _run_incumbent() -> torch.Tensor:
        # Candidate A must be EXACTLY today's default output. Re-enter the
        # router with the portfolio branch suppressed via a private attr --
        # NOT via force_pipeline, because several polish stages (edge
        # equalize best-of-polish, component-tiling polish) are gated on
        # force_pipeline being None and would silently weaken the incumbent.
        incumbent_config = copy.copy(config)
        setattr(incumbent_config, "_dagua_native_suppress_portfolio", True)
        incumbent_state = SolveState(pos=None if state.pos is None else state.pos.detach().clone())
        return _run_native_problem(problem, incumbent_state, ctx, incumbent_config)

    # Contest predicate: the corpus-backed node cap and an explicit caller
    # deadline are deterministic inputs. Within the cap, fixed size-scaled
    # iteration schedules govern challenger work; machine load never changes
    # candidate eligibility.
    n = int(problem.num_nodes)
    if (
        n <= MAX_CONTEST_NODES
        and getattr(config, "time_budget_s", None) is None
        and _use_large_prism_shortlist(problem)
    ):
        try:
            shortlisted_pos = _large_prism_shortlist_candidate(problem, config)
        except Exception as exc:  # noqa: BLE001 -- fall back to the guarded incumbent
            _reraise_worker_timeout(exc)
            _LOGGER.warning("large undirected shortlist failed", exc_info=True)
            shortlisted_pos = None
        if shortlisted_pos is not None:
            # Router-v2 (r2 wave 2): the corpus-backed sfdp+PRISM combination
            # stays the fast-path holder, but where the structural shortlist
            # admits geodesic/community candidates they now compete in a
            # bounded mini-contest instead of being skipped wholesale (the
            # 250 < n <= 1500 non-mesh band was previously unreachable by
            # every new candidate family). The fast-path sfdp+PRISM holder is
            # the incumbent for this branch; pulling in the full native route
            # here re-enters expensive polish work that the branch exists to avoid.
            winner_pos = _router_v2_large_mini_contest(
                shortlisted_pos,
                problem,
                config,
            )
            return _never_nan_winner(
                winner_pos,
                problem,
                float(getattr(config, "_dagua_native_node_sep", config.node_sep)),
                int(problem.seed) if problem.seed is not None else 42,
            )
    incumbent_pos = _run_incumbent()
    if n > MAX_CONTEST_NODES or getattr(config, "time_budget_s", None) is not None:
        return incumbent_pos
    if not _portfolio_has_budget(config):
        _LOGGER.info(
            "Undirected marketplace budget exhausted after incumbent n=%d remaining_s=%s",
            n,
            _portfolio_remaining_s(config),
        )
        return incumbent_pos
    # r80-S8: the aesthetic profile was resolved ONCE in
    # prepare_pipeline_config and stashed on this (already-prepared) config.
    # Reusing that exact object -- rather than re-resolving here -- is what
    # guarantees every candidate in this contest is scored under the
    # identical profile (fairness). `None` when the knob is unset.
    aesthetic_profile: Optional["AestheticProfile"] = getattr(
        config, "_dagua_native_aesthetic_profile", None
    )

    cluster_ids = _build_cluster_ids(problem)
    cluster_count = (
        int(torch.unique(cluster_ids[cluster_ids >= 0]).numel()) if cluster_ids is not None else 0
    )
    degrees = torch.bincount(problem.edge_index.flatten().to(dtype=torch.long), minlength=n)
    max_degree = int(degrees.max().item()) if degrees.numel() else 0
    use_bounded_inner_solvers = _resolved_quality(config) < NEATO_QUALITY_THRESHOLD and (
        (n <= 120 and cluster_count == 4) or (110 <= n <= 150 and max_degree <= 8)
    )
    scores: Dict[str, float] = {}
    positions: Dict[str, torch.Tensor] = {}

    # Candidate A: the incumbent is ALWAYS eligible (degeneracy guard applies
    # to challengers only).
    positions["incumbent"] = incumbent_pos

    # P3 geometry challengers derive from the exact incumbent and bypass
    # projection so their measured transforms reach the honest referee intact.
    from dagua.layout.ops.pipelines.dagua_native import (
        _collinear_dodge,
        _unshear_bimodal_edges,
    )

    geometry_factories = (
        ("collinear_dodge_0.10", lambda: _collinear_dodge(incumbent_pos, problem.edge_index, 0.10)),
        ("collinear_dodge_0.15", lambda: _collinear_dodge(incumbent_pos, problem.edge_index, 0.15)),
        ("unshear", lambda: _unshear_bimodal_edges(incumbent_pos, problem.edge_index)),
    )
    for name, factory in geometry_factories:
        if not _portfolio_has_budget(config, min_remaining_s=2.0):
            break
        if (
            name.startswith("collinear")
            and n * int(problem.edge_index.shape[1]) > MAX_COLLINEAR_WORK
        ):
            continue
        candidate = factory()
        if candidate is None:
            continue
        eligible, reason = _candidate_is_eligible(
            candidate, incumbent_pos, problem.node_sizes, problem.edge_index
        )
        if not eligible:
            _LOGGER.info("Rejected undirected geometry candidate %s: %s", name, reason)
            continue
        positions[name] = candidate

    seed = int(problem.seed) if problem.seed is not None else 42
    challenger_node_sep = float(getattr(config, "_dagua_native_node_sep", config.node_sep))

    raw_finalist_names: list[str] = []

    def _add_challenger(
        name: str,
        raw_pos: torch.Tensor,
        *,
        include_raw: bool = False,
    ) -> None:
        """Repair, project, guard, and score one raw challenger.

        Parameters
        ----------
        name : str
            Stable candidate-family name.
        raw_pos : torch.Tensor
            Unprojected positions with shape ``[N, 2]``.
        include_raw : bool, default=False
            Register a guarded unprojected variant as an honest-ruler
            finalist. Used by fidelity challengers whose benchmark reference
            was scored without overlap projection.

        Returns
        -------
        None
            Candidates are registered in the enclosing contest dictionaries.
        """
        if not bool(torch.isfinite(raw_pos).all().item()):
            _LOGGER.info("Rejected undirected candidate %s: non-finite coordinates", name)
            return
        if include_raw:
            raw_name = f"{name}_raw"
            degenerate, reason = _candidate_is_degenerate(
                raw_pos,
                problem.node_sizes,
                problem.edge_index,
            )
            if degenerate:
                _LOGGER.info("Rejected undirected candidate %s: %s", raw_name, reason)
            else:
                positions[raw_name] = raw_pos
                raw_finalist_names.append(raw_name)

        # Repair, not default (r80 round 4): the candidate keeps its raw
        # layout byte-identical unless the isolated-fling trigger fires, in
        # which case the flung singletons are re-tiled adjacent to the core
        # before projection and the referee scores the repaired version.
        # Applied at this shared entry so every challenger family (sfdp,
        # neato, cluster_sfdp, weighted_similarity) gets the same backstop.
        raw_pos = _repair_flung_isolates(raw_pos, problem, challenger_node_sep)
        if not bool(torch.isfinite(raw_pos).all().item()):
            _LOGGER.info("Rejected repaired undirected candidate %s: non-finite coordinates", name)
            return
        # Below the large threshold all cleanup variants remain additive
        # (r80-S2b). Above it, the corpus profile retains only PRISM, the
        # cleanup that wins the measured large candidate families. The
        # degeneracy guard applies independently to every retained variant.
        for suffix, convergent in _cleanup_variants_for_size(n):
            if not _portfolio_has_budget(config):
                _LOGGER.info(
                    "Skipped undirected candidate %s%s: insufficient remaining budget",
                    name,
                    suffix,
                )
                continue
            try:
                if convergent is None:
                    projected = _project_candidate_prism(raw_pos, problem)
                    if projected is None:
                        _LOGGER.info(
                            "Rejected undirected candidate %s%s: PRISM failed closed", name, suffix
                        )
                        continue
                else:
                    projected = _project_candidate(raw_pos, problem, convergent=convergent)
            except Exception as exc:  # noqa: BLE001 -- one cleanup variant fails closed
                _reraise_worker_timeout(exc)
                _LOGGER.warning(
                    "Rejected undirected candidate %s%s: cleanup failed",
                    name,
                    suffix,
                    exc_info=True,
                )
                continue
            degenerate, reason = _candidate_is_degenerate(
                projected,
                problem.node_sizes,
                problem.edge_index,
            )
            if degenerate:
                _LOGGER.info("Rejected undirected candidate %s%s: %s", name, suffix, reason)
                continue
            positions[name + suffix] = projected

    # Candidate B: our graphviz-fidelity sfdp reimplementation. The contest
    # owns a quality-scaled nonzero budget because LayoutConfig.steps=0 means
    # automatic at the public API, not zero refinement for this challenger.
    if _portfolio_has_budget(config):
        try:
            from dagua.layout.ops.pipelines.sfdp import layout_sfdp_pipeline

            # Raw full-problem solve (round 4): per-component packed solving was
            # tried and regressed healthy multi-component candidates; any
            # isolate fling in this raw output is repaired conditionally inside
            # _add_challenger.
            if problem.edge_weights is None or n <= LARGE_CONTEST_NODE_THRESHOLD:
                sfdp_pos = layout_sfdp_pipeline(
                    edge_index=problem.edge_index,
                    num_nodes=n,
                    node_sizes=problem.node_sizes,
                    steps=(
                        BALANCED_SMALL_REFINEMENT_STEPS
                        if use_bounded_inner_solvers
                        else _candidate_refinement_steps(config, n)
                    ),
                    seed=seed,
                    edge_weights=problem.edge_weights,
                    fidelity_mode="graphviz",
                )
                _add_challenger("sfdp", sfdp_pos)
            if problem.edge_weights is not None and _portfolio_has_budget(config):
                sfdp_unweighted_pos = layout_sfdp_pipeline(
                    edge_index=problem.edge_index,
                    num_nodes=n,
                    node_sizes=problem.node_sizes,
                    steps=(
                        BALANCED_SMALL_REFINEMENT_STEPS
                        if use_bounded_inner_solvers
                        else _candidate_refinement_steps(config, n)
                    ),
                    seed=seed,
                    edge_weights=None,
                    fidelity_mode="graphviz",
                )
                _add_challenger("sfdp_unweighted", sfdp_unweighted_pos)
        except Exception as exc:  # noqa: BLE001 -- a failed challenger never sinks the solve
            _reraise_worker_timeout(exc)
            _LOGGER.warning("SFDP undirected challenger failed", exc_info=True)

    # Candidate C: our neato reimplementation + projection, quality-gated.
    if (
        _portfolio_has_budget(config)
        and _neato_in_contest(config, n)
        and (n <= LARGE_CONTEST_NODE_THRESHOLD or problem.edge_weights is not None)
    ):
        try:
            from dagua.layout.ops.pipelines.neato import layout_neato_pipeline

            # Raw full-problem solve (round 4); isolate fling repaired
            # conditionally inside _add_challenger.
            if n <= LARGE_CONTEST_NODE_THRESHOLD:
                neato_pos = layout_neato_pipeline(
                    edge_index=problem.edge_index,
                    num_nodes=n,
                    node_sizes=problem.node_sizes,
                    seed=seed,
                    edge_weights=problem.edge_weights,
                    maxiter=(
                        NEATO_BALANCED_SMALL_ITERATIONS
                        if use_bounded_inner_solvers
                        else _neato_iterations(config, n)
                    ),
                    fidelity_mode="graphviz",
                    overlap_removal=False,
                )
                _add_challenger("neato", neato_pos)
            if problem.edge_weights is not None:
                neato_unweighted_pos = layout_neato_pipeline(
                    edge_index=problem.edge_index,
                    num_nodes=n,
                    node_sizes=problem.node_sizes,
                    seed=seed,
                    edge_weights=None,
                    maxiter=(
                        NEATO_BALANCED_SMALL_ITERATIONS
                        if use_bounded_inner_solvers
                        else _neato_iterations(config, n)
                    ),
                    fidelity_mode="graphviz",
                    overlap_removal=False,
                )
                _add_challenger("neato_unweighted", neato_unweighted_pos)
        except Exception as exc:  # noqa: BLE001
            _reraise_worker_timeout(exc)
            _LOGGER.warning("neato undirected challenger failed", exc_info=True)

    # Candidate D (r80-S9 Deliverable 1): cluster-aware sfdp driver, only
    # for problems that actually carry cluster metadata. Adds a candidate
    # that structurally places cluster hierarchy levels instead of relying
    # on the composite's cluster-separation term alone (see
    # _cluster_aware_sfdp_candidate). Never replaces the incumbent or the
    # flat sfdp/neato challengers above.
    if (
        problem.clusters
        and n <= LARGE_CONTEST_NODE_THRESHOLD
        and not use_bounded_inner_solvers
        and _portfolio_has_budget(config)
    ):
        try:
            cluster_sfdp_pos = _cluster_aware_sfdp_candidate(problem, config, ctx)
        except Exception as exc:  # noqa: BLE001 -- a failed challenger never sinks the solve
            _reraise_worker_timeout(exc)
            _LOGGER.warning("cluster-SFDP undirected challenger failed", exc_info=True)
            cluster_sfdp_pos = None
        if cluster_sfdp_pos is not None:
            _add_challenger("cluster_sfdp", cluster_sfdp_pos)

    # Candidate E (r80-S9 Deliverable 2): weighted-similarity native-stress
    # core, only for problems that carry edge weights. Adds a candidate
    # whose Dijkstra/pivot target distances treat weights as similarities
    # (see _weighted_similarity_candidate). Never changes default weight
    # handling anywhere else.
    if (
        problem.edge_weights is not None
        and n <= LARGE_CONTEST_NODE_THRESHOLD
        and _portfolio_has_budget(config)
    ):
        try:
            weighted_pos = _weighted_similarity_candidate(problem, seed)
        except Exception as exc:  # noqa: BLE001
            _reraise_worker_timeout(exc)
            _LOGGER.warning("weighted-similarity undirected challenger failed", exc_info=True)
            weighted_pos = None
        if weighted_pos is not None:
            _add_challenger("weighted_similarity", weighted_pos)

    # Candidate F (r81-P1.5): point-unit native stress uses the existing
    # quality-scaled stress schedule. It is additive and contest-scored, so
    # graphs where hop-unit or force candidates are stronger remain unchanged.
    stress_points_pos: Optional[torch.Tensor] = None
    if _portfolio_has_budget(config):
        try:
            edge_count = int(problem.edge_index.shape[1])
            if n > MAX_DENSE_STRESS_NODES or edge_count > MAX_DENSE_STRESS_EDGES:
                _LOGGER.info(
                    "Skipped point-unit stress undirected challenger: dense-work cap n=%d e=%d",
                    n,
                    edge_count,
                )
            else:
                candidate = _stress_points_candidate(problem, seed)
                stress_points_pos = candidate
        except Exception as exc:  # noqa: BLE001 -- a failed challenger never sinks the solve
            _reraise_worker_timeout(exc)
            _LOGGER.warning("point-unit stress undirected challenger failed", exc_info=True)
    if stress_points_pos is not None:
        _add_challenger("stress_points", stress_points_pos)

    # W4 narrow geometry seeds: structurally gated and referee-protected.
    # They only add seed layouts to the existing challenger marketplace; the
    # incumbent remains full-scored and wins every tie.
    if _small_world_knn_seed_enabled(problem) and _portfolio_has_budget(
        config,
        min_remaining_s=2.0,
    ):
        try:
            small_world_seed = _small_world_knn_seed_candidate(incumbent_pos, problem)
            _add_challenger("small_world_knn_seed", small_world_seed)
        except Exception as exc:  # noqa: BLE001 -- a failed seed never sinks the incumbent
            _reraise_worker_timeout(exc)
            _LOGGER.warning("small-world kNN seed challenger failed", exc_info=True)
    if _rgg_geometric_seed_enabled(problem) and _portfolio_has_budget(config):
        try:
            rgg_seed = _rgg_geometric_seed_candidate(problem, seed, challenger_node_sep)
            if rgg_seed is not None:
                _add_challenger("rgg_geometric_seed", rgg_seed)
        except Exception as exc:  # noqa: BLE001 -- a failed seed never sinks the incumbent
            _reraise_worker_timeout(exc)
            _LOGGER.warning("RGG geometric seed challenger failed", exc_info=True)

    # Candidate G (r83-P3.3): local fCoSE at the fidelity campaign's
    # reference defaults. Three adjacent deterministic seeds retain bounded
    # multi-seed coverage without consulting any external adapter.
    fcose_started = time.perf_counter()
    fcose_runs = 0
    if _portfolio_has_budget(config):
        try:
            from dagua.layout.ops.pipelines.fcose import layout_fcose_pipeline

            fcose_cost_s = FCOSE_PRIOR_S
            for seed_offset in range(FCOSE_CONTEST_SEEDS):
                if not _portfolio_has_budget(
                    config
                ) or not _predicted_undirected_arm_budget_available(
                    config,
                    fcose_cost_s,
                ):
                    _record_insufficient_predicted_budget_skip(
                        arm=f"fcose_seed{seed_offset}",
                        config=config,
                        predicted_cost_s=fcose_cost_s,
                    )
                    _LOGGER.info(
                        "Skipped fCoSE seed %d: insufficient predicted budget",
                        seed_offset,
                    )
                    break
                candidate_started_process = time.process_time()
                fcose_pos = layout_fcose_pipeline(
                    edge_index=problem.edge_index,
                    num_nodes=n,
                    node_sizes=problem.node_sizes,
                    steps=FCOSE_REFERENCE_STEPS,
                    seed=seed + seed_offset,
                    edge_weights=problem.edge_weights,
                    quality="default",
                    randomize=True,
                )
                fcose_runs += 1
                _add_challenger(f"fcose_seed{seed_offset}", fcose_pos, include_raw=True)
                fcose_cost_s = _prediction_cpu_elapsed_s(candidate_started_process)
        except Exception as exc:  # noqa: BLE001 -- a failed challenger never sinks the solve
            _reraise_worker_timeout(exc)
            _LOGGER.warning("fCoSE undirected challenger failed", exc_info=True)
    _LOGGER.info(
        "Undirected candidate runtime family=fcose runs=%d seconds=%.3f",
        fcose_runs,
        time.perf_counter() - fcose_started,
    )

    # Candidate H (r83-P3.3): exact local sklearn-compatible tsNET. The
    # quadratic topology-distance/t-SNE work is admitted only through n=300.
    is_mesh = problem.structure is not None and problem.structure.family == GraphFamily.GRID
    if n <= TSNET_MAX_CONTEST_NODES and not is_mesh and _portfolio_has_budget(config):
        tsnet_started = time.perf_counter()
        tsnet_runs = 0
        try:
            from dagua.layout.ops.pipelines.tsnet import layout_tsnet_pipeline

            tsnet_cost_s = TSNET_PRIOR_S
            stop_tsnet = False
            for perplexity in TSNET_PERPLEXITIES:
                for seed_offset in range(TSNET_CONTEST_SEEDS):
                    if not _portfolio_has_budget(
                        config
                    ) or not _predicted_undirected_arm_budget_available(
                        config,
                        tsnet_cost_s,
                    ):
                        _record_insufficient_predicted_budget_skip(
                            arm=f"tsnet_perp{perplexity:g}_seed{seed_offset}",
                            config=config,
                            predicted_cost_s=tsnet_cost_s,
                        )
                        _LOGGER.info(
                            "Skipped tsNET perp=%g seed=%d: insufficient predicted budget",
                            perplexity,
                            seed_offset,
                        )
                        stop_tsnet = True
                        break
                    candidate_started_process = time.process_time()
                    tsnet_pos = layout_tsnet_pipeline(
                        edge_index=problem.edge_index,
                        num_nodes=n,
                        node_sizes=problem.node_sizes,
                        perplexity=perplexity,
                        steps=TSNET_REFERENCE_STEPS,
                        seed=seed + seed_offset,
                        edge_weights=problem.edge_weights,
                        fidelity_mode=True,
                    )
                    tsnet_runs += 1
                    flavor = f"perp{perplexity:g}"
                    _add_challenger(
                        f"tsnet_{flavor}_seed{seed_offset}", tsnet_pos, include_raw=True
                    )
                    tsnet_cost_s = _prediction_cpu_elapsed_s(candidate_started_process)
                if stop_tsnet:
                    break
        except Exception as exc:  # noqa: BLE001 -- a failed challenger never sinks the solve
            _reraise_worker_timeout(exc)
            _LOGGER.warning("tsNET undirected challenger failed", exc_info=True)
        _LOGGER.info(
            "Undirected candidate runtime family=tsnet runs=%d seconds=%.3f",
            tsnet_runs,
            time.perf_counter() - tsnet_started,
        )

    # Candidate I (r83-P3.3): NetworkX-compatible Fruchterman-Reingold is a
    # tiling challenger, so it runs only when weak decomposition is present.
    from dagua.layout.ops.coordinate import _weak_components

    if len(
        _weak_components(problem.edge_index.detach().to(device="cpu"), n)
    ) > 1 and _portfolio_has_budget(config):
        fr_started = time.perf_counter()
        fr_runs = 0
        try:
            from dagua.layout.ops.pipelines.fr import layout_fr_pipeline

            fr_pos = layout_fr_pipeline(
                edge_index=problem.edge_index,
                num_nodes=n,
                node_sizes=problem.node_sizes,
                steps=FR_REFERENCE_STEPS,
                seed=seed,
                edge_weights=problem.edge_weights,
                networkx_compat=True,
            )
            fr_runs = 1
            _add_challenger("fr", fr_pos, include_raw=True)
        except Exception as exc:  # noqa: BLE001 -- a failed challenger never sinks the solve
            _reraise_worker_timeout(exc)
            _LOGGER.warning("FR undirected challenger failed", exc_info=True)
        _LOGGER.info(
            "Undirected candidate runtime family=fr runs=%d seconds=%.3f",
            fr_runs,
            time.perf_counter() - fr_started,
        )

    # Candidates J/K/L (r2 wave 2, router-v2 shortlist): exact-grid
    # certificate, geodesic-MDS stress, and community scaffold, admitted by
    # STRUCTURAL features only (see dagua_native._undirected_route_shortlist;
    # no graph names, no corpus constants). All three are ordinary contest
    # candidates: the honest measured-argmax referee and the incumbent
    # tie-break decide, exactly as for every other challenger family.
    from dagua.layout.ops.pipelines.dagua_native import _undirected_route_shortlist

    shortlist = _undirected_route_shortlist(
        problem.structure,
        n,
        has_edge_weights=problem.edge_weights is not None,
    )
    if shortlist.candidates:
        _LOGGER.info(
            "Router-v2 shortlist classes=%s candidates=%s",
            ",".join(shortlist.classes),
            ",".join(shortlist.candidates),
        )
    if "lattice_cert" in shortlist.candidates and _portfolio_has_budget(
        config, min_remaining_s=2.0
    ):
        try:
            from dagua.layout.ops.pipelines.native_lattice_grid import (
                certificate_grid_positions,
                certify_rect_grid,
            )

            grid_certificate = certify_rect_grid(problem.edge_index, n)
            if grid_certificate is not None:
                cert_pos = certificate_grid_positions(
                    grid_certificate,
                    problem.node_sizes,
                    challenger_node_sep,
                )
                eligible, reason = _candidate_is_eligible(
                    cert_pos, incumbent_pos, problem.node_sizes, problem.edge_index
                )
                if eligible:
                    positions["lattice_cert"] = cert_pos
                else:
                    _LOGGER.info("Rejected undirected candidate lattice_cert: %s", reason)
        except Exception as exc:  # noqa: BLE001 -- a failed challenger never sinks the solve
            _reraise_worker_timeout(exc)
            _LOGGER.warning("lattice certificate candidate failed", exc_info=True)
    if "geodesic_stress" in shortlist.candidates and _portfolio_has_budget(config):
        geodesic_started = time.perf_counter()
        try:
            from dagua.layout.ops.pipelines.native_lattice_grid import (
                geodesic_dense_work_is_allowed,
                layout_geodesic_stress_pipeline,
            )

            if not geodesic_dense_work_is_allowed(n, int(problem.edge_index.shape[1])):
                _LOGGER.info("Skipped geodesic stress challenger: dense-work guard")
            else:
                geodesic_pos = layout_geodesic_stress_pipeline(
                    edge_index=problem.edge_index,
                    num_nodes=n,
                    node_sizes=problem.node_sizes,
                    seed=seed,
                    edge_weights=problem.edge_weights,
                    node_sep=challenger_node_sep,
                )
                _add_challenger("geodesic_stress", geodesic_pos, include_raw=True)
                if problem.edge_weights is not None:
                    # Mirror the sfdp_unweighted/neato_unweighted pattern: the
                    # frozen ruler's stress axes measure HOP distances, so a
                    # hop-geodesic variant competes alongside the weighted one
                    # and the referee picks per graph.
                    geodesic_unweighted_pos = layout_geodesic_stress_pipeline(
                        edge_index=problem.edge_index,
                        num_nodes=n,
                        node_sizes=problem.node_sizes,
                        seed=seed,
                        edge_weights=None,
                        node_sep=challenger_node_sep,
                    )
                    _add_challenger(
                        "geodesic_stress_unweighted",
                        geodesic_unweighted_pos,
                        include_raw=True,
                    )
        except Exception as exc:  # noqa: BLE001 -- a failed challenger never sinks the solve
            _reraise_worker_timeout(exc)
            _LOGGER.warning("geodesic stress undirected challenger failed", exc_info=True)
        _LOGGER.info(
            "Undirected candidate runtime family=geodesic_stress seconds=%.3f",
            time.perf_counter() - geodesic_started,
        )
    if "community_scaffold" in shortlist.candidates and _portfolio_has_budget(config):
        community_started = time.perf_counter()
        try:
            from dagua.layout.ops.pipelines.native_community import (
                layout_native_community_pipeline,
            )

            community_pos = layout_native_community_pipeline(
                edge_index=problem.edge_index,
                num_nodes=n,
                node_sizes=problem.node_sizes,
                config=config,
                seed=seed,
                edge_weights=problem.edge_weights,
            )
            _add_challenger("community_scaffold", community_pos, include_raw=True)
        except Exception as exc:  # noqa: BLE001 -- a failed challenger never sinks the solve
            _reraise_worker_timeout(exc)
            _LOGGER.warning("community scaffold undirected challenger failed", exc_info=True)
        _LOGGER.info(
            "Undirected candidate runtime family=community_scaffold seconds=%.3f",
            time.perf_counter() - community_started,
        )

    # Keep the incumbent plus a deterministic proxy-ranked challenger
    # shortlist. Only these finalists reach the frozen honest ruler.
    from dagua.metrics import _all_pairs_unweighted, _build_csr

    offsets, targets = _build_csr(problem.edge_index.detach().to(device="cpu"), n)
    all_pairs_dist = _all_pairs_unweighted(offsets, targets, n, max_dist=n)
    proxy_scores = {
        name: _proxy_undirected_candidate(pos, problem, cluster_ids, all_pairs_dist)
        for name, pos in positions.items()
    }
    challenger_names = sorted(
        (name for name in positions if name != "incumbent"),
        key=lambda name: (-proxy_scores[name], name),
    )
    full_score_budget = 4 if use_bounded_inner_solvers else len(positions)
    # Fidelity raw variants must reach the same referee that scored their
    # reference counterparts. Preserve proxy budgeting for all other
    # challengers, then append every guarded raw variant deterministically.
    proxy_finalists = challenger_names[: full_score_budget - 1]
    finalist_names = [
        "incumbent",
        *proxy_finalists,
        *(name for name in raw_finalist_names if name not in proxy_finalists),
    ]
    scores = {
        name: _score_undirected_candidate_cached(
            positions[name],
            problem,
            cluster_ids,
            aesthetic_profile,
            all_pairs_dist,
        )
        for name in finalist_names
    }

    # Argmax selection; strict inequality means ties go to the incumbent.
    best_name = "incumbent"
    for name, score in scores.items():
        if name != "incumbent" and score > scores[best_name]:
            best_name = name
    _log_marketplace_telemetry(
        route="undirected",
        structural_gate=(
            "large"
            if n > LARGE_CONTEST_NODE_THRESHOLD
            else "bounded"
            if use_bounded_inner_solvers
            else "default"
        ),
        positions=positions,
        proxy_scores=proxy_scores,
        full_scores=scores,
        finalist_names=finalist_names,
        winner_name=best_name,
        started_at=started_at,
        started_process_at=started_process_at,
    )
    _LOGGER.info(
        "Undirected contest candidates=%s winner=%s",
        ", ".join(f"{name}:{score:.3f}" for name, score in scores.items()),
        best_name,
    )
    return _never_nan_winner(positions[best_name], problem, challenger_node_sep, seed)


def _never_nan_winner(
    winner: torch.Tensor,
    problem: LayoutProblem,
    node_sep: float,
    seed: int,
) -> torch.Tensor:
    """Apply the router-v2 fallback-ladder tail rungs to the contest winner.

    The full four-rung ladder (r2 wave 2):

    1. A challenger that raises or produces non-finite output is dropped
       from the contest (``_add_challenger`` / try-except blocks above).
    2. All challengers dropped means the incumbent runs alone (the argmax
       default above).
    3. THIS rung: a non-finite winner (in practice only possible via the
       incumbent, which the contest never eligibility-checks) is replaced by
       the safe core -- geodesic MDS + clamped stress descent, which cannot
       return non-finite output by construction.
    4. Terminal guard: any residual non-finite coordinate is replaced by a
       deterministic finite circle-plus-jitter layout.

    Finite winners pass through UNCHANGED (bit-identical hot path).

    Parameters
    ----------
    winner : torch.Tensor
        Contest-winning positions with shape ``[N, 2]``.
    problem : LayoutProblem
        Prepared layout problem.
    node_sep : float
        Node separation in points for fallback spacing.
    seed : int
        Deterministic seed shared with the contest.

    Returns
    -------
    torch.Tensor
        Finite positions with shape ``[N, 2]``.
    """
    if bool(torch.isfinite(winner).all().item()):
        return winner
    _LOGGER.warning("Undirected contest winner is non-finite; engaging fallback ladder")
    from dagua.layout.ops.pipelines.native_lattice_grid import (
        _deterministic_fallback_positions,
        _target_edge_length,
        layout_geodesic_stress_pipeline,
    )

    try:  # Rung 3: safe core.
        safe = layout_geodesic_stress_pipeline(
            edge_index=problem.edge_index,
            num_nodes=int(problem.num_nodes),
            node_sizes=problem.node_sizes,
            seed=seed,
            edge_weights=problem.edge_weights,
            node_sep=node_sep,
        )
        if bool(torch.isfinite(safe).all().item()):
            return safe.to(device=winner.device, dtype=winner.dtype)
    except Exception:  # noqa: BLE001 -- the terminal rung below cannot fail
        _LOGGER.warning("fallback-ladder safe core failed", exc_info=True)
    # Rung 4: terminal guard -- always finite, always deterministic.
    spacing = _target_edge_length(problem.node_sizes, node_sep)
    terminal = _deterministic_fallback_positions(int(problem.num_nodes), spacing, seed)
    return terminal.to(device=winner.device, dtype=winner.dtype)


@register_op
@dataclass(frozen=True)
class UndirectedPortfolioRoute(Op):
    """Run the undirected candidate contest and select the honest winner."""

    config: UndirectedPortfolioRouteConfig = field(default_factory=UndirectedPortfolioRouteConfig)

    name: ClassVar[str] = "undirected_portfolio_route"
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
        """Run the portfolio contest and write the winning positions.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with the contest winner's positions.
        """
        layout_config = self.config.layout_config or LayoutConfig()
        state.pos = layout_native_undirected_portfolio(
            problem=problem,
            state=state,
            ctx=ctx,
            config=layout_config,
        )
        return state


def build_native_undirected_portfolio_pipeline(config: LayoutConfig) -> Pipeline:
    """Build the undirected-portfolio route as a one-op pipeline.

    Follows the existing top-level-route precedent: the route is a
    registered op composed into a named pipeline, so pipeline-level callers
    (``build_dagua_pipeline``) and the direct ``_run_native_problem`` branch
    share one implementation.

    Parameters
    ----------
    config : LayoutConfig
        Prepared native configuration.

    Returns
    -------
    Pipeline
        Single-op pipeline running the candidate contest.
    """
    return Pipeline(
        [UndirectedPortfolioRoute(UndirectedPortfolioRouteConfig(layout_config=config))],
        name="native_undirected_portfolio",
    )


__all__ = [
    "MAX_CONTEST_NODES",
    "FCOSE_CONTEST_SEEDS",
    "FR_REFERENCE_STEPS",
    "NEATO_BALANCED_NODE_CAP",
    "NEATO_QUALITY_THRESHOLD",
    "TSNET_CONTEST_SEEDS",
    "TSNET_MAX_CONTEST_NODES",
    "TSNET_PERPLEXITIES",
    "WEIGHTED_SIMILARITY_TRANSFORM",
    "UndirectedPortfolioRoute",
    "UndirectedPortfolioRouteConfig",
    "build_native_undirected_portfolio_pipeline",
    "layout_native_undirected_portfolio",
]
