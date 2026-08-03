"""Anytime native coarsest solver for scale strategies."""

from __future__ import annotations

import copy
import time
from typing import Optional, Tuple

import torch

from dagua.config import LayoutConfig
from dagua.layout.ops.pipelines.dagua_native import layout_dagua_native_pipeline
from dagua.layout.ops.pipelines.native_budget import WALL_DEADLINE_ATTR, install_budget_ledger
from dagua.layout.ops.pipelines.native_v3_referee import score_v3_runtime
from dagua.layout.ops.pipelines.stress_sgd import layout_stress_sgd_pipeline
from dagua.layout.ops.state import LayoutProblem

_STRESS_FALLBACK_STEPS = 30
_STRESS_FALLBACK_MAX_EXACT_NODES = 512
_NATIVE_ATTEMPT_MAX_NODES = 512
_RETURN_RESERVE_S = 1.0


def _finite_position_or_none(pos: torch.Tensor, num_nodes: int) -> Optional[torch.Tensor]:
    """Return a detached finite position tensor, if ``pos`` is valid.

    Parameters
    ----------
    pos : torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    num_nodes : int
        Expected number of rows in the candidate.

    Returns
    -------
    torch.Tensor or None
        Detached ``float32`` positions with shape ``[N, 2]`` when valid;
        otherwise ``None``.
    """
    if pos.ndim != 2 or tuple(pos.shape) != (int(num_nodes), 2):
        return None
    detached = pos.detach().to(dtype=torch.float32)
    if not bool(torch.isfinite(detached).all().item()):
        return None
    return detached


def _score_v3_position(
    problem: LayoutProblem,
    pos: torch.Tensor,
) -> Tuple[Tuple[int, float], float]:
    """Score one position tensor with the runtime frozen V3 referee.

    Parameters
    ----------
    problem : LayoutProblem
        Coarsest layout problem containing topology and node geometry.
    pos : torch.Tensor
        Candidate positions with shape ``[N, 2]``.

    Returns
    -------
    tuple[tuple[int, float], float]
        Severe-G6 eligibility key and higher-is-better V3 tiered score.
    """
    key, score, _facets = score_v3_runtime(
        pos.detach().to(device="cpu", dtype=torch.float32),
        _cpu_problem(problem),
    )
    return key, float(score)


def _candidate_is_better(
    candidate: Tuple[Tuple[int, float], float],
    incumbent: Tuple[Tuple[int, float], float],
) -> bool:
    """Return whether ``candidate`` is a strict frozen-ruler improvement.

    Parameters
    ----------
    candidate : tuple[tuple[int, float], float]
        Candidate severe-G6 key and V3 tiered score.
    incumbent : tuple[tuple[int, float], float]
        Incumbent severe-G6 key and V3 tiered score.

    Returns
    -------
    bool
        ``True`` when the severe-G6 key is lexicographically better, or the
        key ties and the V3 tiered score strictly improves.
    """
    candidate_key, candidate_score = candidate
    incumbent_key, incumbent_score = incumbent
    return candidate_key > incumbent_key or (
        candidate_key == incumbent_key and candidate_score > incumbent_score
    )


def _cpu_problem(problem: LayoutProblem) -> LayoutProblem:
    """Return a CPU copy of the scale coarsest problem.

    Parameters
    ----------
    problem : LayoutProblem
        Source layout problem.

    Returns
    -------
    LayoutProblem
        Problem with tensor fields moved to CPU for native portfolio scoring
        and fallback execution.
    """
    return LayoutProblem(
        edge_index=problem.edge_index.detach().to(device="cpu", dtype=torch.long),
        num_nodes=int(problem.num_nodes),
        node_sizes=(
            None
            if problem.node_sizes is None
            else problem.node_sizes.detach().to(device="cpu", dtype=torch.float32)
        ),
        node_labels=problem.node_labels,
        direction=problem.direction,
        clusters=problem.clusters,
        cluster_parents=problem.cluster_parents,
        cluster_labels=problem.cluster_labels,
        label_positions=problem.label_positions,
        edge_labels=problem.edge_labels,
        node_shapes=problem.node_shapes,
        edge_label_boxes=(
            None
            if problem.edge_label_boxes is None
            else problem.edge_label_boxes.detach().to(device="cpu", dtype=torch.float32)
        ),
        cluster_label_boxes=problem.cluster_label_boxes,
        cluster_tree=problem.cluster_tree,
        structure=problem.structure,
        flex=problem.flex,
        edge_weights=(
            None
            if problem.edge_weights is None
            else problem.edge_weights.detach().to(device="cpu", dtype=torch.float32)
        ),
        seed=int(problem.seed),
    )


def _stress_sgd_fallback(problem: LayoutProblem, seed: int) -> torch.Tensor:
    """Run the deterministic priced Stress-SGD coarsest fallback.

    Parameters
    ----------
    problem : LayoutProblem
        Coarsest layout problem.
    seed : int
        Deterministic seed.

    Returns
    -------
    torch.Tensor
        Finite fallback positions with shape ``[N, 2]``.
    """
    cpu_problem = _cpu_problem(problem)
    result = layout_stress_sgd_pipeline(
        edge_index=cpu_problem.edge_index,
        num_nodes=int(cpu_problem.num_nodes),
        node_sizes=cpu_problem.node_sizes,
        steps=_STRESS_FALLBACK_STEPS,
        seed=int(seed),
        max_exact_nodes=_STRESS_FALLBACK_MAX_EXACT_NODES,
        edge_weights=cpu_problem.edge_weights,
    )
    pos = result[0] if isinstance(result, tuple) else result
    finite = _finite_position_or_none(pos, int(cpu_problem.num_nodes))
    if finite is None:
        raise RuntimeError("stress_sgd fallback did not produce finite coarsest positions.")
    return finite


def _run_budgeted_native(
    problem: LayoutProblem,
    config: LayoutConfig,
    *,
    deadline_s: float,
    seed: int,
    fallback_pos: torch.Tensor,
    fallback_score: Tuple[Tuple[int, float], float],
    native_max_nodes: int = _NATIVE_ATTEMPT_MAX_NODES,
) -> Optional[torch.Tensor]:
    """Run native with scale-only hard-budget metadata installed.

    Parameters
    ----------
    problem : LayoutProblem
        CPU coarsest layout problem.
    config : LayoutConfig
        User configuration copied before mutation.
    deadline_s : float
        Absolute ``time.perf_counter()`` deadline.
    seed : int
        Deterministic seed.
    fallback_pos : torch.Tensor
        Stress-SGD fallback positions with shape ``[N, 2]``.
    fallback_score : tuple[tuple[int, float], float]
        Frozen-ruler score for ``fallback_pos``.
    native_max_nodes : int, default=_NATIVE_ATTEMPT_MAX_NODES
        Scale-private admission cap for native attempts.

    Returns
    -------
    torch.Tensor or None
        Native positions when the budgeted portfolio completes or returns an
        admitted anytime incumbent. ``None`` means the caller should use the
        stress fallback.
    """
    remaining_s = max(0.001, float(deadline_s) - time.perf_counter())
    if int(problem.num_nodes) > int(native_max_nodes) or remaining_s <= _RETURN_RESERVE_S:
        return None

    native_config = copy.copy(config)
    native_config.time_budget_s = None
    native_config.seed = int(seed)
    setattr(native_config, "_dagua_scale_anytime_native", True)
    setattr(native_config, WALL_DEADLINE_ATTR, float(deadline_s))
    setattr(native_config, "_dagua_native_initial_anytime_best", fallback_pos.detach().clone())
    setattr(native_config, "_dagua_native_initial_anytime_score", fallback_score)
    setattr(native_config, "_dagua_native_anytime_score_problem", problem)
    install_budget_ledger(
        native_config,
        remaining_s,
        reserved_tail_dwu=0.0,
        return_reserve_dwu=_RETURN_RESERVE_S,
    )

    try:
        pos = layout_dagua_native_pipeline(
            edge_index=problem.edge_index,
            num_nodes=int(problem.num_nodes),
            node_sizes=(
                problem.node_sizes
                if problem.node_sizes is not None
                else torch.full((int(problem.num_nodes), 2), native_config.node_sep)
            ),
            config=native_config,
            device=str(getattr(native_config, "device", "cpu")),
            clusters=problem.clusters,
            cluster_parents=problem.cluster_parents,
            cluster_labels=problem.cluster_labels,
            label_positions=problem.label_positions,
            edge_labels=problem.edge_labels,
            node_shapes=problem.node_shapes,
            edge_label_boxes=problem.edge_label_boxes,
            cluster_label_boxes=problem.cluster_label_boxes,
            graph_structure=problem.structure,
            seed=int(seed),
            edge_weights=problem.edge_weights,
        )
    except Exception:
        return None
    return _finite_position_or_none(pos, int(problem.num_nodes))


def anytime_native_coarsest(
    problem: LayoutProblem,
    config: LayoutConfig,
    *,
    time_budget_s: float,
    seed: int = 42,
) -> torch.Tensor:
    """Return a priced best-so-far native coarsest layout.

    The certified native default is not entered through this function. Scale
    strategies call this wrapper explicitly with a wall-clock budget; the
    wrapper precomputes a deterministic Stress-SGD fallback, then lets the
    native portfolio run under private hard-deadline metadata. The returned
    tensor is monotone against the fallback under the frozen V3 referee.

    Parameters
    ----------
    problem : LayoutProblem
        Coarsest layout problem with edge tensor shape ``[2, E]`` and optional
        node-size tensor shape ``[N, 2]``.
    config : LayoutConfig
        Layout configuration. It is copied before any scale-only budget
        metadata is attached.
    time_budget_s : float
        Wall-clock budget in seconds.
    seed : int, default=42
        Deterministic seed forwarded to both fallback and native portfolio.

    Returns
    -------
    torch.Tensor
        Finite position tensor with shape ``[N, 2]``.
    """
    if time_budget_s <= 0.0:
        raise ValueError("time_budget_s must be positive.")
    cpu_problem = _cpu_problem(problem)
    started = time.perf_counter()
    deadline_s = started + float(time_budget_s)
    fallback_pos = _stress_sgd_fallback(cpu_problem, int(seed))
    fallback_score = _score_v3_position(cpu_problem, fallback_pos)
    native_max_nodes = int(
        config.algorithm_params.get("scale_native_max_nodes", _NATIVE_ATTEMPT_MAX_NODES)
    )
    native_pos = _run_budgeted_native(
        cpu_problem,
        config,
        deadline_s=deadline_s,
        seed=int(seed),
        fallback_pos=fallback_pos,
        fallback_score=fallback_score,
        native_max_nodes=native_max_nodes,
    )
    if native_pos is None:
        return fallback_pos
    native_score = _score_v3_position(cpu_problem, native_pos)
    if _candidate_is_better(native_score, fallback_score):
        return native_pos
    return fallback_pos
