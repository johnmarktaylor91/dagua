"""Bounded W5 differentiable finisher for native layout candidates."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch

from dagua.config import LayoutConfig
from dagua.layout.ops.pipelines.native_surrogates import (
    angular_resolution_loss,
    barrier_floor_loss,
    crossing_angle_loss,
    depth_order_score_surrogate,
    edge_length_cv_loss,
    gabriel_intrusion_loss,
    overlap_hinge_loss,
    path_continuity_loss,
    signed_flow_score_surrogate,
    soft_crossing_loss,
    soft_knn_neighborhood_loss,
)

_LOGGER = logging.getLogger(__name__)
_ABSOLUTE_DEADLINE_RESERVE_S = 5.0
_MIN_BENCHMARK_REMAINING_S = 30.0
_FINISHER_SCORE_RESERVE_S = 2.0
_DEFAULT_FINISHER_SLICE_S = 4.0
_MIN_FINISHER_ENTRY_S = 1.0
_MAX_W5_SPEND_S = 20.0
_TOTAL_BUDGET_FRACTION = 0.10
_W5_ACCEPT_MARGIN = 0.05
_PREDICTED_COST_SKIP_NODES = 250
_PREDICTED_COST_LATE_ENTRY_REMAINING_S = 90.0
_DISABLE_W5_ENV = "DAGUA_NATIVE_DISABLE_W5"


@dataclass(frozen=True)
class W5ScorePair:
    """Directed and undirected honest composites from one metrics evaluation.

    Parameters
    ----------
    directed : float
        Score from the hierarchy-gated directed composite.
    undirected : float
        Score from the frozen common undirected composite.
    """

    directed: float
    undirected: float


@dataclass(frozen=True)
class W5Seed:
    """Warm-start position for the W5 finisher.

    Parameters
    ----------
    name : str
        Stable seed family label.
    pos : torch.Tensor
        Seed positions with shape ``[N, 2]``.
    """

    name: str
    pos: torch.Tensor


@dataclass(frozen=True)
class W5Checkpoint:
    """Honest-scored W5 checkpoint telemetry.

    Parameters
    ----------
    seed : str
        Seed family label.
    mode : str
        Finisher mode.
    step : int
        Optimization step represented by the checkpoint.
    surrogate_delta : float
        Start loss minus checkpoint loss, so positive means the surrogate improved.
    honest_delta : float
        Directed honest score minus the incumbent/current-winner directed score.
    undirected_honest_delta : float
        Undirected honest score minus the incumbent/current-winner undirected score.
    honest_score_pair : W5ScorePair
        Directed and undirected honest scores for the checkpoint.
    accepted : bool
        Whether this checkpoint cleared the honest W5 accept margin.
    reason : str
        Accept or reject reason.
    """

    seed: str
    mode: str
    step: int
    surrogate_delta: float
    honest_delta: float
    undirected_honest_delta: float
    honest_score_pair: W5ScorePair
    accepted: bool
    reason: str


@dataclass(frozen=True)
class W5Candidate:
    """Accepted W5 candidate.

    Parameters
    ----------
    name : str
        Candidate label.
    pos : torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    score_pair : W5ScorePair
        Directed and undirected honest scores.
    mode : str
        Finisher mode that produced the candidate.
    """

    name: str
    pos: torch.Tensor
    score_pair: W5ScorePair
    mode: str


@dataclass(frozen=True)
class W5FinisherResult:
    """Anytime winner result for one W5 finisher invocation.

    Parameters
    ----------
    winner_pos : torch.Tensor
        Current W5 winner, initialized from the incumbent.
    incumbent_score_pair : W5ScorePair
        Directed and undirected scores for the entry incumbent.
    winner_score_pair : W5ScorePair
        Directed and undirected scores for ``winner_pos``.
    winner_name : str
        Incumbent or accepted checkpoint label.
    deadline_returned : bool
        Whether W5 returned early because deadline or budget was exhausted.
    accepted : tuple[W5Candidate, ...]
        Accepted W5 checkpoints.
    rejected : tuple[W5Checkpoint, ...]
        Rejected W5 checkpoints.
    checkpoints : tuple[W5Checkpoint, ...]
        Honest-scored checkpoints, accepted or rejected.
    mode : str
        Routed mode, or ``"skip"`` when no seed ran.
    steps : int
        Total optimizer steps completed across seeds.
    skipped_reason : str, optional
        Reason the finisher skipped all work.
    slice_s : float, optional
        Work slice granted to this invocation.
    spent_s : float
        Wall-clock seconds spent in this invocation.
    remaining_entry_s : float, optional
        Benchmark seconds remaining at entry.
    remaining_exit_s : float, optional
        Benchmark seconds remaining at return.
    node_count : int
        Number of graph nodes.
    edge_count : int
        Number of graph edges.
    is_semantically_directed : bool
        Routed directedness flag.
    declared_hierarchical : bool
        Routed hierarchy flag.
    direction_is_declared : bool
        Whether directedness came from a user/config declaration.
    """

    winner_pos: torch.Tensor
    incumbent_score_pair: W5ScorePair
    winner_score_pair: W5ScorePair
    winner_name: str
    deadline_returned: bool
    accepted: tuple[W5Candidate, ...]
    rejected: tuple[W5Checkpoint, ...]
    checkpoints: tuple[W5Checkpoint, ...]
    mode: str
    steps: int
    skipped_reason: Optional[str] = None
    slice_s: Optional[float] = None
    spent_s: float = 0.0
    remaining_entry_s: Optional[float] = None
    remaining_exit_s: Optional[float] = None
    node_count: int = 0
    edge_count: int = 0
    is_semantically_directed: bool = False
    declared_hierarchical: bool = False
    direction_is_declared: bool = False


def is_worker_timeout_like_exception(exc: Exception) -> bool:
    """Return whether ``exc`` is the benchmark worker timeout signal.

    Parameters
    ----------
    exc : Exception
        Exception caught by optional polish or W5 code.

    Returns
    -------
    bool
        ``True`` when the worker alarm raised the exception.
    """
    return type(exc).__name__ == "_WorkerLayoutTimeoutError" or (
        "worker layout timeout exceeded" in str(exc)
    )


def w5_dominates(
    candidate: W5ScorePair,
    incumbent: W5ScorePair,
    margin: float = _W5_ACCEPT_MARGIN,
) -> bool:
    """Return whether ``candidate`` beats ``incumbent`` under both rulers.

    Parameters
    ----------
    candidate : W5ScorePair
        Candidate directed and undirected scores.
    incumbent : W5ScorePair
        Current winner directed and undirected scores.
    margin : float, default=0.05
        Required improvement in both score components.

    Returns
    -------
    bool
        ``True`` only when both finite components clear ``margin``.
    """
    return (
        math.isfinite(candidate.directed)
        and math.isfinite(candidate.undirected)
        and candidate.directed > incumbent.directed + margin
        and candidate.undirected > incumbent.undirected + margin
    )


def _remaining_s(config: Optional[LayoutConfig]) -> Optional[float]:
    """Return remaining benchmark seconds when a deadline is known.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration.

    Returns
    -------
    float or None
        Remaining seconds, or ``None`` outside benchmark deadline mode.
    """
    deadline = getattr(config, "_dagua_native_deadline_s", None) if config is not None else None
    if deadline is None:
        return None
    return float(deadline) - time.perf_counter()


def _w5_disabled_by_env() -> bool:
    """Return whether W5 is disabled by the diagnostic environment flag.

    Returns
    -------
    bool
        ``True`` when ``DAGUA_NATIVE_DISABLE_W5`` is set to a truthy value.
    """
    raw = os.environ.get(_DISABLE_W5_ENV, "")
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _w5_spend_cap_s(config: Optional[LayoutConfig], remaining: Optional[float]) -> float:
    """Return the per-layout accumulated W5 wall-clock cap.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration carrying benchmark budget metadata.
    remaining : float, optional
        Remaining benchmark seconds, used as a fallback budget outside normal
        benchmark configuration.

    Returns
    -------
    float
        Maximum total seconds W5 may spend for this layout invocation.
    """
    if config is None or (
        remaining is None and not hasattr(config, "_dagua_native_total_budget_s")
    ):
        return _DEFAULT_FINISHER_SLICE_S
    fallback_budget = _DEFAULT_FINISHER_SLICE_S if remaining is None else remaining
    total_budget = float(getattr(config, "_dagua_native_total_budget_s", fallback_budget))
    return min(_MAX_W5_SPEND_S, _TOTAL_BUDGET_FRACTION * total_budget)


def _w5_spent_s(config: Optional[LayoutConfig], started_perf: Optional[float] = None) -> float:
    """Return accumulated W5 seconds, including this invocation when supplied.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration carrying accumulated W5 runtime.
    started_perf : float, optional
        ``time.perf_counter()`` value for the active invocation.

    Returns
    -------
    float
        Accumulated W5 wall-clock seconds.
    """
    previous = float(getattr(config, "_dagua_native_w5_spent_s", 0.0))
    if started_perf is None:
        return previous
    return previous + max(0.0, time.perf_counter() - started_perf)


def w5_predicted_skip_reason(
    node_count: int,
    edge_count: int,
    config: Optional[LayoutConfig],
) -> Optional[str]:
    """Return a predicted-cost skip reason before W5 does expensive work.

    Parameters
    ----------
    node_count : int
        Number of graph nodes.
    edge_count : int
        Number of graph edges.
    config : LayoutConfig, optional
        Prepared layout configuration carrying benchmark deadline metadata.

    Returns
    -------
    str or None
        Skip reason when W5 should preserve the incumbent without running, or
        ``None`` when W5 may proceed.
    """
    if _w5_disabled_by_env():
        return "disabled_by_env"
    if node_count >= _PREDICTED_COST_SKIP_NODES:
        return "predicted_cost_large_graph"
    remaining = _remaining_s(config)
    if remaining is not None and remaining < _PREDICTED_COST_LATE_ENTRY_REMAINING_S:
        return "predicted_cost_late_entry"
    return None


def _finisher_slice_s(config: Optional[LayoutConfig]) -> Optional[float]:
    """Return the bounded W5 work slice, or ``None`` when it must skip.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration.

    Returns
    -------
    float or None
        Available finisher seconds before the return reserve.
    """
    if _w5_disabled_by_env():
        return None
    remaining = _remaining_s(config)
    if remaining is None:
        return _DEFAULT_FINISHER_SLICE_S
    if remaining < _MIN_BENCHMARK_REMAINING_S:
        return None
    available = remaining - _ABSOLUTE_DEADLINE_RESERVE_S
    if available < _MIN_FINISHER_ENTRY_S + _FINISHER_SCORE_RESERVE_S:
        return None
    spend_cap = _w5_spend_cap_s(config, remaining)
    spent = _w5_spent_s(config)
    remaining_w5_budget = spend_cap - spent
    if remaining_w5_budget < _MIN_FINISHER_ENTRY_S + _FINISHER_SCORE_RESERVE_S:
        return None
    return max(0.0, min(remaining_w5_budget, available))


def make_w5_skip_result(
    *,
    incumbent_pos: torch.Tensor,
    incumbent_score_pair: Optional[W5ScorePair],
    reason: str,
    edge_index: Optional[torch.Tensor] = None,
    config: Optional[LayoutConfig] = None,
    is_semantically_directed: bool = False,
    declared_hierarchical: bool = False,
    direction_is_declared: bool = False,
) -> W5FinisherResult:
    """Build a telemetry-friendly W5 skip result without doing W5 pre-work.

    Parameters
    ----------
    incumbent_pos : torch.Tensor
        Incumbent positions with shape ``[N, 2]``.
    incumbent_score_pair : W5ScorePair, optional
        Precomputed incumbent score pair, when available without extra work.
    reason : str
        Skip reason to report.
    edge_index : torch.Tensor, optional
        Edge tensor with shape ``[2, E]``.
    config : LayoutConfig, optional
        Prepared layout configuration.
    is_semantically_directed : bool, default=False
        Routed directedness flag.
    declared_hierarchical : bool, default=False
        Routed hierarchy flag.
    direction_is_declared : bool, default=False
        Whether directedness was explicitly declared.

    Returns
    -------
    W5FinisherResult
        Incumbent winner result with skip metadata.
    """
    fallback_score = incumbent_score_pair or W5ScorePair(float("nan"), float("nan"))
    return W5FinisherResult(
        winner_pos=incumbent_pos,
        incumbent_score_pair=fallback_score,
        winner_score_pair=fallback_score,
        winner_name="incumbent",
        deadline_returned=reason in {"no_budget", "deadline"},
        accepted=(),
        rejected=(),
        checkpoints=(),
        mode="skip",
        steps=0,
        skipped_reason=reason,
        slice_s=None,
        spent_s=0.0,
        remaining_entry_s=_remaining_s(config),
        remaining_exit_s=_remaining_s(config),
        node_count=int(incumbent_pos.shape[0]),
        edge_count=(
            int(edge_index.shape[1]) if edge_index is not None and edge_index.ndim == 2 else 0
        ),
        is_semantically_directed=is_semantically_directed,
        declared_hierarchical=declared_hierarchical,
        direction_is_declared=direction_is_declared,
    )


def _dedupe_seeds(seeds: Sequence[W5Seed], max_seeds: int = 3) -> list[W5Seed]:
    """Return finite, non-duplicate W5 seeds.

    Parameters
    ----------
    seeds : Sequence[W5Seed]
        Candidate warm starts.
    max_seeds : int, default=3
        Maximum seeds to keep.

    Returns
    -------
    list[W5Seed]
        Deduplicated finite seeds.
    """
    kept: list[W5Seed] = []
    for seed in seeds:
        if len(kept) >= max_seeds or not bool(torch.isfinite(seed.pos).all().item()):
            continue
        if any(
            torch.allclose(seed.pos, existing.pos, atol=1.0e-5, rtol=1.0e-5) for existing in kept
        ):
            continue
        kept.append(seed)
    return kept


def _longest_path_depth(
    edge_index: torch.Tensor,
    node_count: int,
    device: torch.device,
) -> torch.Tensor:
    """Return longest-path depths, falling back to zeros on cyclic inputs.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_count : int
        Number of nodes.
    device : torch.device
        Output device.

    Returns
    -------
    torch.Tensor
        Depth tensor with shape ``[N]``.
    """
    try:
        from dagua.utils import longest_path_layering

        depth = longest_path_layering(edge_index.detach().to(device="cpu"), node_count)
        if not isinstance(depth, torch.Tensor):
            depth = torch.as_tensor(depth, dtype=torch.long)
        return depth.to(device=device, dtype=torch.long)
    except Exception:  # noqa: BLE001 -- cyclic semantic graphs use flow-only barriers
        return torch.zeros(node_count, dtype=torch.long, device=device)


def _route_mode(
    seed_pos: torch.Tensor,
    edge_index: torch.Tensor,
    topo_depth: torch.Tensor,
    *,
    is_semantically_directed: bool,
    declared_hierarchical: bool,
    direction_is_declared: bool,
) -> str:
    """Choose the W5 descent mode from structural/local incumbent signals.

    Parameters
    ----------
    seed_pos : torch.Tensor
        Seed positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    topo_depth : torch.Tensor
        Depth tensor with shape ``[N]``.
    is_semantically_directed : bool
        Whether edge direction has semantic meaning.
    declared_hierarchical : bool
        Whether the honest ruler treats the graph as hierarchical.
    direction_is_declared : bool
        Whether semantic direction came from user/config metadata.

    Returns
    -------
    str
        One of ``"x_only"``, ``"barrier_2d"``, or ``"undirected_2d_sampled"``.
    """
    if not is_semantically_directed or not direction_is_declared:
        return "undirected_2d_sampled"
    if not declared_hierarchical:
        return "barrier_2d"
    flow = float(signed_flow_score_surrogate(seed_pos, edge_index).detach().item())
    depth = float(depth_order_score_surrogate(seed_pos, topo_depth).detach().item())
    node_count = int(seed_pos.shape[0])
    if flow >= 0.95 and depth >= 0.95 and (node_count <= 64 or node_count >= 250):
        return "x_only"
    return "barrier_2d"


def _overlap_count(pos: torch.Tensor, node_sizes: torch.Tensor) -> int:
    """Count exact overlapping boxes for W5 regression rejection.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.

    Returns
    -------
    int
        Number of overlapping unordered node pairs.
    """
    if pos.shape[0] <= 1:
        return 0
    work_pos = pos.detach().to(device="cpu", dtype=torch.float32)
    work_sizes = node_sizes.detach().to(device="cpu", dtype=torch.float32)
    dx = (work_pos[:, None, 0] - work_pos[None, :, 0]).abs()
    dy = (work_pos[:, None, 1] - work_pos[None, :, 1]).abs()
    min_dx = (work_sizes[:, None, 0] + work_sizes[None, :, 0]) * 0.5
    min_dy = (work_sizes[:, None, 1] + work_sizes[None, :, 1]) * 0.5
    overlap = (dx < min_dx) & (dy < min_dy)
    overlap.fill_diagonal_(False)
    return int(overlap.triu(diagonal=1).sum().item())


def _is_degenerate(pos: torch.Tensor, node_sizes: torch.Tensor) -> bool:
    """Return whether positions are non-finite or collapsed.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.

    Returns
    -------
    bool
        ``True`` when the candidate is not safe to score.
    """
    if not bool(torch.isfinite(pos).all().item()):
        return True
    if pos.shape[0] <= 1:
        return False
    extent = pos.detach().amax(dim=0) - pos.detach().amin(dim=0)
    min_extent = float(node_sizes.detach().to(dtype=torch.float32).mean().item()) * 0.1
    return float(extent.max().item()) <= max(1.0e-6, min_extent)


def _surrogate_loss(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    topo_depth: torch.Tensor,
    mode: str,
    floors: dict[str, float],
) -> torch.Tensor:
    """Evaluate the composite-weighted W5 surrogate objective.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    topo_depth : torch.Tensor
        Depth tensor with shape ``[N]``.
    mode : str
        Routed W5 mode.
    floors : dict[str, float]
        Incumbent floor values for barrier terms.

    Returns
    -------
    torch.Tensor
        Scalar loss to minimize.
    """
    crossing = soft_crossing_loss(pos, edge_index)
    flow_score = signed_flow_score_surrogate(pos, edge_index)
    depth_score = depth_order_score_surrogate(pos, topo_depth)
    common_scale = 0.75 if mode in {"x_only", "barrier_2d"} else 1.0
    loss = (
        common_scale * 20.0 * crossing
        + common_scale * 13.0 * overlap_hinge_loss(pos, node_sizes)
        + common_scale * 12.0 * soft_knn_neighborhood_loss(pos, edge_index)
        + common_scale * 7.0 * edge_length_cv_loss(pos, edge_index)
        + common_scale * 5.0 * gabriel_intrusion_loss(pos, edge_index)
        + common_scale * 5.0 * crossing_angle_loss(pos, edge_index)
        + common_scale * 4.0 * angular_resolution_loss(pos, edge_index)
        + common_scale * 4.0 * path_continuity_loss(pos, edge_index)
    )
    if mode in {"x_only", "barrier_2d"}:
        loss = loss + 16.0 * (1.0 - flow_score) + 9.0 * (1.0 - depth_score)
    if mode == "barrier_2d":
        loss = (
            loss
            + 64.0 * barrier_floor_loss(flow_score, floors.get("flow"))
            + 36.0 * barrier_floor_loss(depth_score, floors.get("depth"))
            + 20.0 * torch.relu(crossing - floors.get("crossing_loss", crossing.detach())).square()
        )
    return torch.nan_to_num(loss, nan=1.0e6, posinf=1.0e6, neginf=1.0e6)


def _optimize_seed(
    seed: W5Seed,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    topo_depth: torch.Tensor,
    mode: str,
    deadline: float,
) -> tuple[torch.Tensor, int, float, list[tuple[int, torch.Tensor, float]]]:
    """Run one bounded W5 descent from ``seed``.

    Parameters
    ----------
    seed : W5Seed
        Warm start.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    topo_depth : torch.Tensor
        Depth tensor with shape ``[N]``.
    mode : str
        Routed W5 mode.
    deadline : float
        Absolute ``time.monotonic()`` deadline for optimizer work.

    Returns
    -------
    tuple[torch.Tensor, int, float, list[tuple[int, torch.Tensor, float]]]
        Final positions, completed steps, start loss, and checkpoint
        positions/losses.
    """
    work = seed.pos.detach().clone().to(dtype=torch.float32)
    work.requires_grad_(True)
    start_y = work[:, 1].detach().clone()
    floors = {
        "flow": float(signed_flow_score_surrogate(work, edge_index).detach().item()),
        "depth": float(depth_order_score_surrogate(work, topo_depth).detach().item()),
        "crossing_loss": float(soft_crossing_loss(work, edge_index).detach().item()),
    }
    start_loss_tensor = _surrogate_loss(work, edge_index, node_sizes, topo_depth, mode, floors)
    start_loss = float(start_loss_tensor.detach().item())
    median_size = float(node_sizes.detach().to(dtype=torch.float32).mean().item())
    lr = max(0.01, min(4.0, 0.04 * median_size))
    optimizer = torch.optim.Adam([work], lr=lr)
    node_count = int(work.shape[0])
    desired_steps = 24 if node_count >= 300 else 36
    effective_max_steps = desired_steps
    checkpoints: list[tuple[int, torch.Tensor, float]] = []
    checkpoint_steps = {max(1, desired_steps // 2), desired_steps}
    completed_steps = 0
    for step in range(1, desired_steps + 1):
        if step > effective_max_steps:
            break
        if time.monotonic() >= deadline:
            break
        step_started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss = _surrogate_loss(work, edge_index, node_sizes, topo_depth, mode, floors)
        if not bool(torch.isfinite(loss).all().item()):
            break
        loss.backward()
        optimizer.step()
        if step == 1:
            step_cost = max(1.0e-6, time.monotonic() - step_started)
            remaining_step_budget = max(0.0, deadline - time.monotonic())
            steps_that_fit = 1 + int(remaining_step_budget / step_cost)
            effective_max_steps = max(1, min(desired_steps, steps_that_fit))
            checkpoint_steps = {max(1, effective_max_steps // 2), effective_max_steps}
        if mode == "x_only":
            with torch.no_grad():
                work[:, 1] = start_y
        completed_steps = step
        if step in checkpoint_steps:
            checkpoint_pos = work.detach().clone()
            checkpoint_loss = float(
                _surrogate_loss(checkpoint_pos, edge_index, node_sizes, topo_depth, mode, floors)
                .detach()
                .item()
            )
            checkpoints.append((step, checkpoint_pos, checkpoint_loss))
    return work.detach(), completed_steps, start_loss, checkpoints


def run_w5_finisher(
    *,
    incumbent_pos: torch.Tensor,
    incumbent_score_pair: W5ScorePair,
    seeds: Sequence[W5Seed],
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    score_fn: Callable[[torch.Tensor], W5ScorePair],
    is_semantically_directed: bool,
    declared_hierarchical: bool,
    direction_is_declared: bool = False,
    config: Optional[LayoutConfig] = None,
    accept_margin: float = _W5_ACCEPT_MARGIN,
) -> W5FinisherResult:
    """Run the W5 finisher and return the anytime honest winner.

    Parameters
    ----------
    incumbent_pos : torch.Tensor
        Incumbent positions with shape ``[N, 2]``.
    incumbent_score_pair : W5ScorePair
        Directed and undirected honest scores for ``incumbent_pos``.
    seeds : Sequence[W5Seed]
        Warm starts already generated by the native contest/polish path.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    score_fn : Callable[[torch.Tensor], W5ScorePair]
        Dual-ruler honest scorer.
    is_semantically_directed : bool
        Whether edge direction has semantic meaning.
    declared_hierarchical : bool
        Whether the honest ruler uses directed hierarchy terms.
    direction_is_declared : bool, default=False
        Whether semantic direction came from user/config metadata.
    config : LayoutConfig, optional
        Prepared native configuration carrying optional benchmark deadline.
    accept_margin : float, default=0.05
        Required improvement over the current winner in both score components.

    Returns
    -------
    W5FinisherResult
        Anytime winner plus telemetry for all honest-scored checkpoints.
    """
    started_perf = time.perf_counter()
    remaining_entry = _remaining_s(config)
    slice_s = _finisher_slice_s(config)
    node_count = int(incumbent_pos.shape[0])
    edge_count = int(edge_index.shape[1]) if edge_index.ndim == 2 else 0
    predicted_skip_reason = w5_predicted_skip_reason(node_count, edge_count, config)
    if slice_s is None and predicted_skip_reason != "disabled_by_env":
        predicted_skip_reason = None
    if predicted_skip_reason is not None:
        slice_s = None

    def finish(
        *,
        winner_pos: torch.Tensor,
        winner_score_pair: W5ScorePair,
        winner_name: str,
        deadline_returned: bool,
        accepted: list[W5Candidate],
        rejected: list[W5Checkpoint],
        checkpoints: list[W5Checkpoint],
        mode: str,
        steps: int,
        skipped_reason: Optional[str],
    ) -> W5FinisherResult:
        """Finalize result metadata and update config W5 spend.

        Parameters
        ----------
        winner_pos : torch.Tensor
            Current winner tensor with shape ``[N, 2]``.
        winner_score_pair : W5ScorePair
            Directed and undirected scores for ``winner_pos``.
        winner_name : str
            Current winner label.
        deadline_returned : bool
            Whether the return was forced by a deadline or exhausted budget.
        accepted : list[W5Candidate]
            Accepted checkpoint candidates.
        rejected : list[W5Checkpoint]
            Rejected checkpoint telemetry.
        checkpoints : list[W5Checkpoint]
            All honest-scored checkpoint telemetry.
        mode : str
            Last routed mode or ``"skip"``.
        steps : int
            Optimizer steps completed.
        skipped_reason : str, optional
            Reason no checkpoint was accepted or no work ran.

        Returns
        -------
        W5FinisherResult
            Finalized W5 result.
        """
        if winner_pos is not incumbent_pos and not w5_dominates(
            winner_score_pair,
            incumbent_score_pair,
            float(accept_margin),
        ):
            winner_pos = incumbent_pos
            winner_score_pair = incumbent_score_pair
            winner_name = "incumbent"
            skipped_reason = "clamped_to_incumbent"
        spent_s = max(0.0, time.perf_counter() - started_perf)
        if config is not None:
            previous_spent = float(getattr(config, "_dagua_native_w5_spent_s", 0.0))
            setattr(config, "_dagua_native_w5_spent_s", previous_spent + spent_s)
        return W5FinisherResult(
            winner_pos=winner_pos,
            incumbent_score_pair=incumbent_score_pair,
            winner_score_pair=winner_score_pair,
            winner_name=winner_name,
            deadline_returned=deadline_returned,
            accepted=tuple(accepted),
            rejected=tuple(rejected),
            checkpoints=tuple(checkpoints),
            mode=mode,
            steps=steps,
            skipped_reason=skipped_reason,
            slice_s=slice_s,
            spent_s=spent_s,
            remaining_entry_s=remaining_entry,
            remaining_exit_s=_remaining_s(config),
            node_count=node_count,
            edge_count=edge_count,
            is_semantically_directed=is_semantically_directed,
            declared_hierarchical=declared_hierarchical,
            direction_is_declared=direction_is_declared,
        )

    if slice_s is None:
        return finish(
            winner_pos=incumbent_pos,
            winner_score_pair=incumbent_score_pair,
            winner_name="incumbent",
            deadline_returned=predicted_skip_reason in {None, "predicted_cost_late_entry"},
            accepted=[],
            rejected=[],
            checkpoints=[],
            mode="skip",
            steps=0,
            skipped_reason=predicted_skip_reason or "no_budget",
        )
    kept_seeds = _dedupe_seeds(seeds)
    if not kept_seeds:
        return finish(
            winner_pos=incumbent_pos,
            winner_score_pair=incumbent_score_pair,
            winner_name="incumbent",
            deadline_returned=False,
            accepted=[],
            rejected=[],
            checkpoints=[],
            mode="skip",
            steps=0,
            skipped_reason="no_finite_seed",
        )
    deadline = time.monotonic() + slice_s
    edge_work = edge_index.detach().to(device=kept_seeds[0].pos.device, dtype=torch.long)
    size_work = node_sizes.detach().to(device=kept_seeds[0].pos.device, dtype=torch.float32)
    topo_depth = _longest_path_depth(
        edge_work,
        int(kept_seeds[0].pos.shape[0]),
        kept_seeds[0].pos.device,
    )
    winner_pos = incumbent_pos
    winner_score_pair = incumbent_score_pair
    winner_name = "incumbent"
    accepted: list[W5Candidate] = []
    rejected: list[W5Checkpoint] = []
    checkpoints: list[W5Checkpoint] = []
    steps_total = 0
    routed_mode = "skip"
    incumbent_overlap = _overlap_count(incumbent_pos, size_work)
    deadline_returned = False
    for seed in kept_seeds:
        if _w5_spent_s(config, started_perf) >= _w5_spend_cap_s(config, remaining_entry):
            deadline_returned = True
            break
        if time.monotonic() >= deadline - _FINISHER_SCORE_RESERVE_S:
            deadline_returned = True
            break
        mode = _route_mode(
            seed.pos.detach().to(device=edge_work.device, dtype=torch.float32),
            edge_work,
            topo_depth,
            is_semantically_directed=is_semantically_directed,
            declared_hierarchical=declared_hierarchical,
            direction_is_declared=direction_is_declared,
        )
        routed_mode = mode
        try:
            final_pos, steps, start_loss, scored_points = _optimize_seed(
                seed,
                edge_work,
                size_work,
                topo_depth,
                mode,
                deadline - _FINISHER_SCORE_RESERVE_S,
            )
        except Exception as exc:  # noqa: BLE001 -- W5 is optional candidate generation
            if is_worker_timeout_like_exception(exc):
                raise
            _LOGGER.warning("W5 finisher seed %s failed", seed.name, exc_info=True)
            continue
        steps_total += steps
        if not scored_points or scored_points[-1][0] != steps:
            final_loss = float(
                _surrogate_loss(final_pos, edge_work, size_work, topo_depth, mode, {})
                .detach()
                .item()
            )
            scored_points.append((steps, final_pos, final_loss))
        for step, checkpoint_pos, checkpoint_loss in scored_points[:2]:
            if _w5_spent_s(config, started_perf) >= _w5_spend_cap_s(config, remaining_entry):
                deadline_returned = True
                break
            if time.monotonic() >= deadline:
                deadline_returned = True
                break
            viable = not _is_degenerate(checkpoint_pos, size_work) and (
                _overlap_count(checkpoint_pos, size_work) <= incumbent_overlap
            )
            if not viable:
                continue
            try:
                score_pos = checkpoint_pos.to(device=edge_index.device, dtype=torch.float32)
                honest = score_fn(score_pos)
            except Exception as exc:
                if is_worker_timeout_like_exception(exc):
                    raise
                continue
            if not math.isfinite(honest.directed) or not math.isfinite(honest.undirected):
                continue
            directed_delta = honest.directed - winner_score_pair.directed
            undirected_delta = honest.undirected - winner_score_pair.undirected
            surrogate_delta = start_loss - checkpoint_loss
            is_accepted = w5_dominates(honest, winner_score_pair, float(accept_margin))
            reason = "dominates" if is_accepted else "does_not_dominate_both"
            checkpoint = W5Checkpoint(
                seed=seed.name,
                mode=mode,
                step=int(step),
                surrogate_delta=float(surrogate_delta),
                honest_delta=float(directed_delta),
                undirected_honest_delta=float(undirected_delta),
                honest_score_pair=honest,
                accepted=is_accepted,
                reason=reason,
            )
            checkpoints.append(checkpoint)
            if is_accepted:
                name = f"w5_{mode}_{seed.name}_{step}"
                winner_pos = checkpoint_pos.to(
                    device=incumbent_pos.device,
                    dtype=incumbent_pos.dtype,
                )
                winner_score_pair = honest
                winner_name = name
                accepted_candidate = W5Candidate(
                    name=name,
                    pos=winner_pos,
                    score_pair=honest,
                    mode=mode,
                )
                accepted.append(accepted_candidate)
            else:
                rejected.append(checkpoint)
    skipped = None if accepted else ("no_checkpoint_improved" if checkpoints else "no_checkpoint")
    return finish(
        winner_pos=winner_pos,
        winner_score_pair=winner_score_pair,
        winner_name=winner_name,
        deadline_returned=deadline_returned,
        accepted=accepted,
        rejected=rejected,
        checkpoints=checkpoints,
        mode=routed_mode,
        steps=steps_total,
        skipped_reason=skipped,
    )


def log_w5_telemetry(result: W5FinisherResult, config: Optional[LayoutConfig]) -> None:
    """Emit and attach structured W5 finisher telemetry.

    Parameters
    ----------
    result : W5FinisherResult
        Finisher result to report.
    config : LayoutConfig, optional
        Config receiving ``_dagua_native_w5_telemetry`` when available.

    Returns
    -------
    None
        Telemetry is logged and optionally stored on ``config``.
    """

    def pair_payload(pair: W5ScorePair) -> dict[str, float]:
        """Convert a score pair to a JSON-serializable payload.

        Parameters
        ----------
        pair : W5ScorePair
            Directed and undirected score pair.

        Returns
        -------
        dict[str, float]
            JSON-ready score fields.
        """
        return {"directed": float(pair.directed), "undirected": float(pair.undirected)}

    payload = {
        "event": "native_w5_finisher",
        "node_count": result.node_count,
        "edge_count": result.edge_count,
        "mode": result.mode,
        "steps": result.steps,
        "skipped_reason": result.skipped_reason,
        "deadline_returned": result.deadline_returned,
        "direction_is_declared": result.direction_is_declared,
        "direction_is_inferred": not result.direction_is_declared,
        "is_semantically_directed": result.is_semantically_directed,
        "declared_hierarchical": result.declared_hierarchical,
        "slice_s": result.slice_s,
        "spent_s": result.spent_s,
        "remaining_entry_s": result.remaining_entry_s,
        "remaining_exit_s": result.remaining_exit_s,
        "winner_name": result.winner_name,
        "incumbent_score_pair": pair_payload(result.incumbent_score_pair),
        "winner_score_pair": pair_payload(result.winner_score_pair),
        "accepted": [candidate.name for candidate in result.accepted],
        "rejected": [
            {
                "seed": checkpoint.seed,
                "mode": checkpoint.mode,
                "step": checkpoint.step,
                "reason": checkpoint.reason,
            }
            for checkpoint in result.rejected
        ],
        "checkpoints": [
            {
                "seed": checkpoint.seed,
                "mode": checkpoint.mode,
                "step": checkpoint.step,
                "surrogate_delta": checkpoint.surrogate_delta,
                "directed_honest_delta": checkpoint.honest_delta,
                "undirected_honest_delta": checkpoint.undirected_honest_delta,
                "honest_score_pair": pair_payload(checkpoint.honest_score_pair),
                "accepted": checkpoint.accepted,
                "reason": checkpoint.reason,
            }
            for checkpoint in result.checkpoints
        ],
    }
    if config is not None:
        existing = list(getattr(config, "_dagua_native_w5_telemetry", []))
        existing.append(payload)
        setattr(config, "_dagua_native_w5_telemetry", existing)
    telemetry_path = os.environ.get("DAGUA_W5_TELEMETRY_PATH")
    if telemetry_path:
        with open(telemetry_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    print("native_w5_finisher " + json.dumps(payload, sort_keys=True), flush=True)
    _LOGGER.info("Native W5 finisher telemetry %s", json.dumps(payload, sort_keys=True))


__all__ = [
    "W5Candidate",
    "W5Checkpoint",
    "W5FinisherResult",
    "W5ScorePair",
    "W5Seed",
    "is_worker_timeout_like_exception",
    "log_w5_telemetry",
    "make_w5_skip_result",
    "run_w5_finisher",
    "w5_dominates",
]
