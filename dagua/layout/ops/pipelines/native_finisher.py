"""Bounded W5 differentiable finisher for native layout candidates."""

from __future__ import annotations

import inspect
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import torch

from dagua.config import LayoutConfig
from dagua.layout.ops.pipelines.native_budget import (
    DETERMINISTIC_BUDGET_ATTR,
    PROCESS_DEADLINE_ATTR,
    available_process_work_s,
    remaining_process_s,
    remaining_wall_s,
    wall_reserve_exhausted,
)
from dagua.layout.ops.pipelines.native_shape_geometry import (
    NativeShapeGeometry,
    pairwise_shape_signed_gap,
)
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
from dagua.layout.projection import project_overlaps

_LOGGER = logging.getLogger(__name__)
_ABSOLUTE_DEADLINE_RESERVE_S = 5.0
_MIN_BENCHMARK_REMAINING_S = 30.0
_FINISHER_SCORE_RESERVE_S = 2.0
_DEFAULT_FINISHER_SLICE_S = 4.0
_MIN_FINISHER_ENTRY_S = 1.0
_MAX_W5_SPEND_S = 20.0
_TOTAL_BUDGET_FRACTION = 0.10
_W5_ACCEPT_MARGIN = 0.05
_PREDICTED_COST_LATE_ENTRY_REMAINING_S = 90.0
_PREDICTED_COST_RETURN_RESERVE_S = 2.0
_MEASURED_COST_MAX_SEEDS = 2
_MEASURED_COST_MAX_CHECKPOINTS = 2
_MEASURED_COST_TINY_MAX_CHECKPOINTS = 4
_MEASURED_COST_TINY_REFEREE_S = 0.05
_MEASURED_COST_TINY_MAX_N = 64
_MEASURED_COST_TINY_STEPS = 96
_TINY_ROW_CONTINUATION_CAP_S = 2.0
_TINY_ROW_DETERMINISTIC_STEP_COST_S = 0.04
_TINY_ROW_DETERMINISTIC_REFEREE_COST_S = 0.012
_MEASURED_COST_DEFAULT_REFEREE_S = 0.40
_MEASURED_COST_SURROGATE_STEPS = 4
_W5_STRESS_MAX_SOURCES = 200
_W5_STRESS_MAX_PAIRS = 100_000
_DISABLE_W5_ENV = "DAGUA_NATIVE_DISABLE_W5"
_GRAPH_NAME_ATTR = "_dagua_native_graph_name"
_W5_PROJECTION_ITERATIONS = 20


@dataclass(frozen=True)
class W5PhaseTiming:
    """Per-seed W5 phase timing telemetry.

    Parameters
    ----------
    seed : str
        Seed family label.
    mode : str
        Routed W5 mode for this seed.
    pass_id : int
        Surrogate pass identifier, ``1`` for the existing objective and ``2``
        for the honest-aligned continuation.
    route_s : float
        Wall-clock seconds spent choosing the route.
    optimize_s : float
        Wall-clock seconds spent in surrogate descent.
    viability_s : float
        Wall-clock seconds spent projecting and checking checkpoints.
    score_s : float
        Wall-clock seconds spent in honest scoring for checkpoints.
    """

    seed: str
    mode: str
    pass_id: int
    route_s: float
    optimize_s: float
    viability_s: float
    score_s: float


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
class W5HonestAxes:
    """Honest per-axis W5 routing signals from the frozen metrics ruler.

    Parameters
    ----------
    flow : float, optional
        Honest ``directed_flow_score`` in ``[0, 1]`` when available.
    depth : float, optional
        Honest ``depth_order_score`` in ``[0, 1]`` when available.
    ksm : float, optional
        Honest ``ksm_score`` in ``[0, 1]`` when available.
    edge_length : float, optional
        Honest ``edge_length_deviation_score`` in ``[0, 1]`` when available.
    """

    flow: Optional[float] = None
    depth: Optional[float] = None
    ksm: Optional[float] = None
    edge_length: Optional[float] = None


def w5_honest_axes_from_metrics(numeric: dict[str, Any]) -> W5HonestAxes:
    """Extract route-safe honest axes from a ``full()`` metric dictionary.

    Parameters
    ----------
    numeric : dict[str, Any]
        Metric payload returned by ``dagua.metrics.full``.

    Returns
    -------
    W5HonestAxes
        Finite honest axis scores used by W5 routing and barrier weighting.
    """

    def finite_float(key: str) -> Optional[float]:
        """Return a finite float from ``numeric`` or ``None``.

        Parameters
        ----------
        key : str
            Metric key to read.

        Returns
        -------
        float or None
            Finite metric value when present.
        """
        value = numeric.get(key)
        if value is None:
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    return W5HonestAxes(
        flow=finite_float("directed_flow_score"),
        depth=finite_float("depth_order_score"),
        ksm=finite_float("ksm_score"),
        edge_length=finite_float("edge_length_deviation_score"),
    )


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
class W5CostPlan:
    """Measured W5 work bounds admitted by the shared spend cap.

    Parameters
    ----------
    seeds : int
        Number of finite seeds to run.
    steps : int
        Maximum optimizer steps per seed.
    checkpoints : int
        Maximum honest-scored checkpoints per seed, capped at two.
    measured_step_s : float
        Wall-clock seconds for one post-warmup surrogate step.
    warmup_s : float
        One-time wall-clock warmup charge from the first surrogate step.
    referee_s : float
        Wall-clock seconds for one honest referee score, measured on the
        incumbent before W5 entry.
    budget_s : float
        Wall-clock seconds available under the shared W5 spend cap.
    budget_usable_s : float
        Wall-clock seconds available after the explicit return reserve.
    predicted_s : float
        Conservative wall-clock cost estimate for the admitted plan.
    """

    seeds: int
    steps: int
    checkpoints: int
    measured_step_s: float
    warmup_s: float
    referee_s: float
    budget_s: float
    budget_usable_s: float
    predicted_s: float


@dataclass(frozen=True)
class W5StepMeasurement:
    """Measured W5 surrogate cost components.

    Parameters
    ----------
    step_s : float
        Wall-clock seconds for one post-warmup optimizer step.
    warmup_s : float
        Wall-clock seconds charged once for the first optimizer step.
    """

    step_s: float
    warmup_s: float


@dataclass(frozen=True)
class W5Checkpoint:
    """Honest-scored W5 checkpoint telemetry.

    Parameters
    ----------
    seed : str
        Seed family label.
    mode : str
        Finisher mode.
    pass_id : int
        Surrogate pass identifier, ``1`` for the existing objective and ``2``
        for the honest-aligned continuation.
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
    pass_spend_s : float
        Wall-clock seconds spent in this seed/mode/pass through the checkpoint
        scoring event.
    """

    seed: str
    mode: str
    pass_id: int
    step: int
    surrogate_delta: float
    honest_delta: float
    undirected_honest_delta: float
    honest_score_pair: W5ScorePair
    accepted: bool
    reason: str
    pass_spend_s: float


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
    graph_name : str, optional
        Benchmark graph name when the driver supplied one on the config.
    incumbent_axes : W5HonestAxes, optional
        Honest per-axis incumbent scores used to route W5.
    phase_timings_s : tuple[W5PhaseTiming, ...]
        Per-seed route/optimize/viability/score timing records.
    viability_counts : dict[str, int]
        Counts for viability outcomes, including projection outcomes.
    viability_drop_counts : dict[str, int]
        Counts for pre-score viability drop reasons.
    cost_plan : W5CostPlan, optional
        Measured-cost admission math, when the terminal W5 path used it.
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
    graph_name: Optional[str] = None
    incumbent_axes: Optional[W5HonestAxes] = None
    phase_timings_s: tuple[W5PhaseTiming, ...] = ()
    viability_counts: dict[str, int] = field(default_factory=dict)
    viability_drop_counts: dict[str, int] = field(default_factory=dict)
    cost_plan: Optional[W5CostPlan] = None


@dataclass(frozen=True)
class W5StressSample:
    """Fixed differentiable stress sample for W5 pass 2.

    Parameters
    ----------
    sources : torch.Tensor
        Source node indices with shape ``[P]``.
    targets : torch.Tensor
        Target node indices with shape ``[P]``.
    graph_distances : torch.Tensor
        Positive graph distances with shape ``[P]``.
    """

    sources: torch.Tensor
    targets: torch.Tensor
    graph_distances: torch.Tensor


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
    return remaining_wall_s(config)


def _w5_first_score_epilogue_has_budget(config: Optional[LayoutConfig]) -> bool:
    """Return whether one terminal checkpoint score has deterministic budget.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration carrying benchmark deadline metadata.

    Returns
    -------
    bool
        ``True`` when no benchmark budget exists, or when the deterministic
        process-time budget can fit one referee score with a safety margin.
        Wall-clock only vetoes when the hard return reserve is already gone.
    """
    if wall_reserve_exhausted(config, _ABSOLUTE_DEADLINE_RESERVE_S):
        return False
    referee_s = float(
        getattr(config, "_dagua_native_w5_referee_cost_s", _MEASURED_COST_DEFAULT_REFEREE_S)
    )
    referee_s = max(1.0e-6, referee_s)
    process_remaining = _process_remaining_s(config)
    return process_remaining is None or process_remaining > 2.0 * referee_s + 1.0


def _w5_first_score_epilogue_has_wall_headroom(config: Optional[LayoutConfig]) -> bool:
    """Return whether one terminal checkpoint score may run.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration carrying benchmark deadline metadata.

    Returns
    -------
    bool
        Alias for the deterministic epilogue budget gate, kept for existing
        tests and callers that monkeypatch the historical helper name.
    """
    return _w5_first_score_epilogue_has_budget(config)


def _stack_graph_name() -> Optional[str]:
    """Return a benchmark graph name discovered from active driver frames.

    Returns
    -------
    str or None
        Graph name from ``scripts/run_benchmark.py``/``dagua.eval.benchmark``
        locals when W5 is executing under those drivers, otherwise ``None``.
    """
    frame = inspect.currentframe()
    current = None if frame is None else frame.f_back
    try:
        while current is not None:
            locals_by_name: dict[str, Any] = current.f_locals
            work_item = locals_by_name.get("work_item")
            graph_name = getattr(work_item, "graph_name", None)
            if graph_name is not None:
                return str(graph_name)
            test_graph = locals_by_name.get("test_graph")
            graph_name = getattr(test_graph, "name", None)
            if graph_name is not None:
                return str(graph_name)
            benchmark_graph = locals_by_name.get("bg")
            graph_name = getattr(getattr(benchmark_graph, "test_graph", None), "name", None)
            if graph_name is not None:
                return str(graph_name)
            current = current.f_back
        return None
    finally:
        del frame
        del current


def _graph_name(config: Optional[LayoutConfig]) -> Optional[str]:
    """Return the benchmark graph name when it was attached to ``config``.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration.

    Returns
    -------
    str or None
        Stable graph name, or ``None`` outside benchmark/name-aware callers.
    """
    name = getattr(config, _GRAPH_NAME_ATTR, None) if config is not None else None
    if name is not None:
        return str(name)
    return _stack_graph_name()


def _process_remaining_s(config: Optional[LayoutConfig]) -> Optional[float]:
    """Return process-time seconds remaining for optional W5 admission gates.

    Wall-clock deadline checks remain the hard return guard. This helper
    gives optional work a CPU-time ruler so sibling-worker contention does
    not change late-entry or predicted-cost admission after the process
    deadline is initialized.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration carrying benchmark deadline metadata.

    Returns
    -------
    float or None
        Process CPU seconds remaining, or ``None`` without benchmark budget
        metadata.
    """
    return remaining_process_s(config)


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
    """Return the per-layout accumulated W5 deterministic spend cap.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration carrying benchmark budget metadata.
    remaining : float, optional
        Remaining deterministic benchmark seconds, used as a fallback budget
        outside normal benchmark configuration.

    Returns
    -------
    float
        Maximum total process seconds W5 may spend for this layout invocation.
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


def _w5_process_spent_s(
    config: Optional[LayoutConfig],
    started_process: Optional[float] = None,
) -> float:
    """Return accumulated W5 process seconds, including this invocation.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration carrying accumulated process runtime.
    started_process : float, optional
        ``time.process_time()`` value for the active invocation.

    Returns
    -------
    float
        Accumulated W5 process seconds.
    """
    previous = float(getattr(config, "_dagua_native_w5_process_spent_s", 0.0))
    if started_process is None:
        return previous
    return previous + max(0.0, time.process_time() - started_process)


def _use_tiny_row_deterministic_w5_costs(
    config: Optional[LayoutConfig],
    node_count: int,
) -> bool:
    """Return whether tiny-row W5 admission should use fixed cost units.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration carrying optional benchmark metadata.
    node_count : int
        Number of layout nodes in the W5 seed.

    Returns
    -------
    bool
        ``True`` for tiny benchmark-budgeted rows where measured process-time
        step costs are too small and noisy to be a stable admission unit.
    """
    return (
        config is not None
        and getattr(config, DETERMINISTIC_BUDGET_ATTR, None) is not None
        and getattr(config, PROCESS_DEADLINE_ATTR, None) is not None
        and int(node_count) <= _MEASURED_COST_TINY_MAX_N
    )


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
    del node_count, edge_count
    if _w5_disabled_by_env():
        return "disabled_by_env"
    if bool(getattr(config, "_dagua_native_w5_measured_sizing", False)):
        return None
    remaining = _process_remaining_s(config)
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
    process_remaining = _process_remaining_s(config)
    if process_remaining is None and _remaining_s(config) is None:
        return _DEFAULT_FINISHER_SLICE_S
    if wall_reserve_exhausted(config, _ABSOLUTE_DEADLINE_RESERVE_S):
        return None
    if process_remaining is None:
        return None
    if process_remaining < _MIN_BENCHMARK_REMAINING_S:
        return None
    available = available_process_work_s(config, _ABSOLUTE_DEADLINE_RESERVE_S)
    assert available is not None
    if available < _MIN_FINISHER_ENTRY_S + _FINISHER_SCORE_RESERVE_S:
        return None
    spend_cap = _w5_spend_cap_s(config, process_remaining)
    spent = _w5_process_spent_s(config)
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
        graph_name=_graph_name(config),
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
    honest_axes: Optional[W5HonestAxes] = None,
) -> str:
    """Choose the W5 descent mode from honest incumbent axes.

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
    honest_axes : W5HonestAxes, optional
        Frozen-ruler incumbent axis scores. When absent, W5 falls back to the
        old surrogate route for compatibility with direct unit callers.

    Returns
    -------
    str
        One of ``"x_only"``, ``"barrier_2d"``, or ``"undirected_2d_sampled"``.
    """
    if not is_semantically_directed or not direction_is_declared:
        return "undirected_2d_sampled"
    if not declared_hierarchical:
        return "barrier_2d"
    # The mode decision must use the same honest axes as the accept ruler:
    # a high surrogate flow self-report cannot route a flow-deficient
    # incumbent into x_only, where y-motion is frozen and the flow gap is
    # unreachable. The dominance gate below remains the monotone safety rail.
    if honest_axes is not None:
        flow = 0.0 if honest_axes.flow is None else float(honest_axes.flow)
        depth = 0.0 if honest_axes.depth is None else float(honest_axes.depth)
    else:
        flow = float(signed_flow_score_surrogate(seed_pos, edge_index).detach().item())
        depth = float(depth_order_score_surrogate(seed_pos, topo_depth).detach().item())
    node_count = int(seed_pos.shape[0])
    if flow >= 0.95 and depth >= 0.95 and (node_count <= 64 or node_count >= 250):
        return "x_only"
    return "barrier_2d"


def _mode_ladder(mode: str, *, is_semantically_directed: bool) -> tuple[str, ...]:
    """Return the W5 mode ladder for one seed.

    Parameters
    ----------
    mode : str
        Initially routed mode.
    is_semantically_directed : bool
        Whether edge direction has semantic meaning.

    Returns
    -------
    tuple[str, ...]
        Modes to try in order. The second directed pass is only useful for
        ``x_only`` failures; it gives the same seed y-motion without changing
        the monotone accept gate.
    """
    if is_semantically_directed and mode == "x_only":
        return ("x_only", "barrier_2d")
    return (mode,)


def _overlap_count(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    shape_geometry: Optional[NativeShapeGeometry] = None,
) -> int:
    """Count exact overlapping boxes for W5 regression rejection.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    shape_geometry : NativeShapeGeometry, optional
        Optional non-box shape descriptors.

    Returns
    -------
    int
        Number of overlapping unordered node pairs.
    """
    if pos.shape[0] <= 1:
        return 0
    if shape_geometry is not None:
        gaps = pairwise_shape_signed_gap(
            pos.detach().to(device="cpu", dtype=torch.float32),
            node_sizes.detach().to(device="cpu", dtype=torch.float32),
            shape_geometry.to(device=torch.device("cpu"), dtype=torch.float32),
            max_nodes=int(pos.shape[0]),
        )
        return int((gaps < 0.0).sum().item())
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


def _project_checkpoint_for_viability(
    checkpoint_pos: torch.Tensor,
    node_sizes: torch.Tensor,
    shape_geometry: Optional[NativeShapeGeometry] = None,
) -> torch.Tensor:
    """Return an overlap-projected checkpoint for W5 viability checks.

    Parameters
    ----------
    checkpoint_pos : torch.Tensor
        Candidate checkpoint positions with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    shape_geometry : NativeShapeGeometry, optional
        Optional non-box shape descriptors.

    Returns
    -------
    torch.Tensor
        Projected checkpoint positions with shape ``[N, 2]``.
    """
    projected = checkpoint_pos.detach().clone().to(dtype=torch.float32)
    sizes = node_sizes.detach().to(device=projected.device, dtype=projected.dtype)
    project_overlaps(
        projected,
        sizes,
        iterations=_W5_PROJECTION_ITERATIONS,
        convergent=True,
    )
    if shape_geometry is not None:
        _project_shape_overlaps(projected, sizes, shape_geometry)
    return projected


def _project_shape_overlaps(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    shape_geometry: NativeShapeGeometry,
    *,
    iterations: int = _W5_PROJECTION_ITERATIONS,
) -> None:
    """Resolve true-shape overlaps with deterministic pairwise radial shifts.

    Parameters
    ----------
    pos : torch.Tensor
        Mutable position tensor with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.
    shape_geometry : NativeShapeGeometry
        Per-node shape descriptors.
    iterations : int, default=_W5_PROJECTION_ITERATIONS
        Maximum repair sweeps.

    Returns
    -------
    None
        ``pos`` is modified in place.
    """
    if int(pos.shape[0]) < 2:
        return
    geometry = shape_geometry.to(device=pos.device, dtype=pos.dtype)
    for _ in range(iterations):
        moved = False
        for left in range(int(pos.shape[0]) - 1):
            for right in range(left + 1, int(pos.shape[0])):
                gap = pairwise_shape_signed_gap(
                    pos[[left, right]],
                    node_sizes[[left, right]],
                    NativeShapeGeometry(kind_codes=geometry.kind_codes[[left, right]]),
                    max_nodes=2,
                )
                if gap.numel() == 0 or float(gap[0].item()) >= 0.0:
                    continue
                delta = pos[right] - pos[left]
                distance = torch.linalg.vector_norm(delta).clamp_min(1.0e-6)
                if float(distance.item()) <= 1.0e-5:
                    angle = float(left * 92821 + right * 68917) * 0.0001
                    direction = pos.new_tensor((math.cos(angle), math.sin(angle)))
                else:
                    direction = delta / distance
                shift = direction * ((-gap[0] + 1.0e-4) * 0.5)
                pos[left] -= shift
                pos[right] += shift
                moved = True
        if not moved:
            break


def _closed_over_all_pairs_dist(score_fn: Callable[[torch.Tensor], W5ScorePair]) -> Optional[Any]:
    """Return ``all_pairs_dist`` transitively captured by the honest scorer.

    Parameters
    ----------
    score_fn : Callable[[torch.Tensor], W5ScorePair]
        Honest scoring closure built by the terminal native path.

    Returns
    -------
    object or None
        Existing all-pairs distance matrix from the scorer closure. ``None``
        means pass 2 must omit the stress term rather than compute APSP here.
    """

    def search_closure(fn: object, visited: set[int]) -> Optional[Any]:
        """Search one closure chain for the APSP matrix without invoking code.

        Parameters
        ----------
        fn : object
            Candidate function-valued object whose closure may capture
            ``all_pairs_dist``.
        visited : set[int]
            Object identities already scanned, used to guard closure cycles.

        Returns
        -------
        object or None
            Captured APSP matrix when present.
        """
        fn_id = id(fn)
        if fn_id in visited:
            return None
        visited.add(fn_id)
        code = getattr(fn, "__code__", None)
        closure = getattr(fn, "__closure__", None)
        if code is None or closure is None:
            return None
        function_cells: list[object] = []
        for name, cell in zip(code.co_freevars, closure):
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if name == "all_pairs_dist":
                return value
            if callable(value) and getattr(value, "__closure__", None) is not None:
                function_cells.append(value)
        for value in function_cells:
            nested = search_closure(value, visited)
            if nested is not None:
                return nested
        return None

    return search_closure(score_fn, set())


def _build_w5_stress_sample(
    edge_index: torch.Tensor,
    node_count: int,
    all_pairs_dist: Optional[Any],
    device: torch.device,
) -> Optional[W5StressSample]:
    """Build a fixed capped stress sample from an existing APSP matrix.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_count : int
        Number of graph nodes.
    all_pairs_dist : object, optional
        Precomputed all-pairs graph distances. No sample is built when this is
        absent, preserving the no-new-APSP invariant.
    device : torch.device
        Device for returned index and target tensors.

    Returns
    -------
    W5StressSample or None
        Deterministic detached stress sample, capped at ``_W5_STRESS_MAX_PAIRS``.
    """
    if all_pairs_dist is None or node_count < 2 or edge_index.numel() == 0:
        return None
    try:
        from dagua.metrics import _deterministic_sample_indices, _stratified_graph_pairs

        n_targets = max(1, _W5_STRESS_MAX_PAIRS // _W5_STRESS_MAX_SOURCES)
        sources, targets, graph_distances = _stratified_graph_pairs(
            edge_index.detach().to(device="cpu", dtype=torch.long),
            int(node_count),
            _W5_STRESS_MAX_SOURCES,
            n_targets,
            all_pairs_dist=all_pairs_dist,
        )
        if sources.size == 0:
            return None
        if sources.size > _W5_STRESS_MAX_PAIRS:
            keep = _deterministic_sample_indices(int(sources.size), _W5_STRESS_MAX_PAIRS)
            sources = sources[keep]
            targets = targets[keep]
            graph_distances = graph_distances[keep]
        source_tensor = torch.as_tensor(sources, dtype=torch.long, device=device).detach()
        target_tensor = torch.as_tensor(targets, dtype=torch.long, device=device).detach()
        distance_tensor = torch.as_tensor(
            graph_distances,
            dtype=torch.float32,
            device=device,
        ).detach()
    except Exception:  # noqa: BLE001 -- pass-2 stress is optional candidate guidance
        return None
    if source_tensor.numel() == 0:
        return None
    return W5StressSample(
        sources=source_tensor,
        targets=target_tensor,
        graph_distances=distance_tensor.clamp_min(1.0),
    )


def _increment_count(counts: dict[str, int], key: str) -> None:
    """Increment ``key`` in a telemetry count dictionary.

    Parameters
    ----------
    counts : dict[str, int]
        Mutable telemetry count dictionary.
    key : str
        Count key to increment.

    Returns
    -------
    None
        ``counts`` is updated in place.
    """
    counts[key] = int(counts.get(key, 0)) + 1


def _surrogate_loss(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    topo_depth: torch.Tensor,
    mode: str,
    floors: dict[str, float],
    shape_geometry: Optional[NativeShapeGeometry] = None,
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
    shape_geometry : NativeShapeGeometry, optional
        Optional non-box shape descriptors.

    Returns
    -------
    torch.Tensor
        Scalar loss to minimize.
    """
    crossing = soft_crossing_loss(pos, edge_index)
    flow_score = signed_flow_score_surrogate(pos, edge_index)
    depth_score = depth_order_score_surrogate(pos, topo_depth)
    overlap_loss = overlap_hinge_loss(pos, node_sizes, shape_geometry=shape_geometry)
    knn_loss = soft_knn_neighborhood_loss(pos, edge_index)
    edge_cv_loss = edge_length_cv_loss(pos, edge_index)
    common_scale = 0.75 if mode in {"x_only", "barrier_2d"} else 1.0
    loss = (
        common_scale * 20.0 * crossing
        + common_scale * 13.0 * overlap_loss
        + common_scale * 12.0 * knn_loss
        + common_scale * 7.0 * edge_cv_loss
        + common_scale * 5.0 * gabriel_intrusion_loss(pos, edge_index)
        + common_scale * 5.0 * crossing_angle_loss(pos, edge_index)
        + common_scale * 4.0 * angular_resolution_loss(pos, edge_index)
        + common_scale * 4.0 * path_continuity_loss(pos, edge_index)
    )
    if mode in {"x_only", "barrier_2d"}:
        loss = loss + 16.0 * (1.0 - flow_score) + 9.0 * (1.0 - depth_score)
    if mode == "barrier_2d":
        honest_flow = floors.get("honest_flow")
        flow_headroom = 0.0 if honest_flow is None else max(0.0, 1.0 - float(honest_flow))
        honest_ksm = floors.get("honest_ksm")
        ksm_floor_weight = 24.0 if honest_ksm is None else 24.0 + 24.0 * float(honest_ksm)
        edge_cv_weight = 10.0 + 42.0 * flow_headroom
        loss = (
            loss
            + (24.0 + 96.0 * flow_headroom) * (1.0 - flow_score)
            + 64.0 * barrier_floor_loss(flow_score, floors.get("flow"))
            + 36.0 * barrier_floor_loss(depth_score, floors.get("depth"))
            + 20.0 * torch.relu(crossing - floors.get("crossing_loss", crossing.detach())).square()
            + 20.0
            * torch.relu(overlap_loss - floors.get("overlap_loss", overlap_loss.detach())).square()
            + ksm_floor_weight
            * torch.relu(knn_loss - floors.get("knn_loss", knn_loss.detach())).square()
            + edge_cv_weight * edge_cv_loss
        )
    return torch.nan_to_num(loss, nan=1.0e6, posinf=1.0e6, neginf=1.0e6)


def _stress_gain_loss(pos: torch.Tensor, stress_sample: Optional[W5StressSample]) -> torch.Tensor:
    """Return a scale-fitted sampled stress loss for W5 pass 2.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    stress_sample : W5StressSample, optional
        Fixed source/target graph-distance sample.

    Returns
    -------
    torch.Tensor
        Scalar normalized stress residual. Zero is returned when no fixed
        sample is available.
    """
    if stress_sample is None or stress_sample.sources.numel() == 0:
        return pos.new_zeros(())
    geometric = torch.linalg.vector_norm(
        pos[stress_sample.sources] - pos[stress_sample.targets],
        dim=1,
    )
    targets = stress_sample.graph_distances.to(device=pos.device, dtype=pos.dtype)
    denominator = torch.dot(targets, targets).clamp_min(1.0e-12)
    scale = torch.dot(geometric, targets) / denominator
    reference = scale * targets
    residual = geometric - reference
    normalizer = geometric.detach().square().mean().clamp_min(1.0e-12)
    return residual.square().mean() / normalizer


def _edge_length_l1_deviation_loss(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Return MAD/mean edge-length deviation matching the honest metric shape.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.

    Returns
    -------
    torch.Tensor
        Scalar relative mean absolute deviation over edge lengths.
    """
    if edge_index.numel() == 0:
        return pos.new_zeros(())
    lengths = torch.linalg.vector_norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
    mean_length = lengths.mean().clamp_min(1.0e-12)
    return torch.abs(lengths - mean_length).mean() / mean_length.detach().clamp_min(1.0e-12)


def _aligned_surrogate_loss(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    topo_depth: torch.Tensor,
    mode: str,
    floors: dict[str, float],
    stress_sample: Optional[W5StressSample],
    shape_geometry: Optional[NativeShapeGeometry] = None,
) -> torch.Tensor:
    """Evaluate the pass-2 W5 objective with honest-aligned additive terms.

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
    stress_sample : W5StressSample, optional
        Fixed sampled graph-distance pairs for stress guidance.
    shape_geometry : NativeShapeGeometry, optional
        Optional non-box shape descriptors.

    Returns
    -------
    torch.Tensor
        Scalar loss to minimize during pass 2.
    """
    loss = _surrogate_loss(
        pos,
        edge_index,
        node_sizes,
        topo_depth,
        mode,
        floors,
        shape_geometry,
    )
    honest_ksm = floors.get("honest_ksm")
    ksm_headroom = 0.0 if honest_ksm is None else max(0.0, 1.0 - float(honest_ksm))
    stress_weight = 12.0 + 72.0 * ksm_headroom
    edge_l1_weight = 6.0
    loss = (
        loss
        + stress_weight * _stress_gain_loss(pos, stress_sample)
        + edge_l1_weight * _edge_length_l1_deviation_loss(pos, edge_index)
    )
    return torch.nan_to_num(loss, nan=1.0e6, posinf=1.0e6, neginf=1.0e6)


def _checkpoint_steps(desired_steps: int, max_checkpoints: int) -> set[int]:
    """Return deterministic checkpoint steps for a bounded optimizer pass.

    Parameters
    ----------
    desired_steps : int
        Planned optimizer steps for the pass.
    max_checkpoints : int
        Maximum number of checkpoints to score.

    Returns
    -------
    set[int]
        Pass-local step indices to checkpoint.
    """
    if max_checkpoints <= 0:
        return set()
    if max_checkpoints == 1:
        return {max(1, int(desired_steps))}
    return {
        int(desired_steps)
        if index == max_checkpoints
        else max(1, int(desired_steps) * index // max_checkpoints)
        for index in range(1, max_checkpoints + 1)
    }


def _pass_loss(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    topo_depth: torch.Tensor,
    mode: str,
    floors: dict[str, float],
    pass_id: int,
    stress_sample: Optional[W5StressSample],
    shape_geometry: Optional[NativeShapeGeometry] = None,
) -> torch.Tensor:
    """Evaluate the selected W5 pass loss.

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
    pass_id : int
        Surrogate pass identifier.
    stress_sample : W5StressSample, optional
        Fixed sampled graph-distance pairs for pass 2.
    shape_geometry : NativeShapeGeometry, optional
        Optional non-box shape descriptors.

    Returns
    -------
    torch.Tensor
        Scalar loss for the requested pass.
    """
    if pass_id == 1:
        return _surrogate_loss(
            pos,
            edge_index,
            node_sizes,
            topo_depth,
            mode,
            floors,
            shape_geometry,
        )
    return _aligned_surrogate_loss(
        pos,
        edge_index,
        node_sizes,
        topo_depth,
        mode,
        floors,
        stress_sample,
        shape_geometry,
    )


def _optimize_seed(
    seed: W5Seed,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    topo_depth: torch.Tensor,
    mode: str,
    deadline: float,
    honest_axes: Optional[W5HonestAxes] = None,
    max_steps: Optional[int] = None,
    max_checkpoints: int = _MEASURED_COST_MAX_CHECKPOINTS,
    step_timing_hook: Optional[Callable[[int, float], None]] = None,
    pass_id: int = 1,
    stress_sample: Optional[W5StressSample] = None,
    shape_geometry: Optional[NativeShapeGeometry] = None,
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
    honest_axes : W5HonestAxes, optional
        Honest incumbent axis scores used to weight barrier-mode gains.
    max_steps : int, optional
        Maximum optimizer steps allowed for measured cost sizing.
    max_checkpoints : int, default=2
        Maximum checkpoints to return for honest scoring.
    step_timing_hook : Callable[[int, float], None], optional
        Callback receiving each completed step index and wall-clock step
        duration. Used only by measured admission sizing.
    pass_id : int, default=1
        Surrogate pass identifier. Pass 1 uses the unchanged objective; pass 2
        adds honest-aligned stress and L1 edge-length terms.
    stress_sample : W5StressSample, optional
        Fixed sampled graph-distance pairs for pass 2.
    shape_geometry : NativeShapeGeometry, optional
        Optional non-box shape descriptors.

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
        "overlap_loss": float(
            overlap_hinge_loss(work, node_sizes, shape_geometry=shape_geometry).detach().item()
        ),
        "knn_loss": float(soft_knn_neighborhood_loss(work, edge_index).detach().item()),
        "edge_cv_loss": float(edge_length_cv_loss(work, edge_index).detach().item()),
    }
    if honest_axes is not None:
        if honest_axes.flow is not None:
            floors["honest_flow"] = float(honest_axes.flow)
        if honest_axes.ksm is not None:
            floors["honest_ksm"] = float(honest_axes.ksm)
    start_loss_tensor = _pass_loss(
        work,
        edge_index,
        node_sizes,
        topo_depth,
        mode,
        floors,
        pass_id,
        stress_sample,
        shape_geometry,
    )
    start_loss = float(start_loss_tensor.detach().item())
    median_size = float(node_sizes.detach().to(dtype=torch.float32).mean().item())
    lr = max(0.01, min(4.0, 0.04 * median_size))
    optimizer = torch.optim.Adam([work], lr=lr)
    node_count = int(work.shape[0])
    base_desired_steps = 24 if node_count >= 300 else 36
    desired_steps = int(max_steps) if max_steps is not None else base_desired_steps
    desired_steps = max(1, desired_steps)
    effective_max_steps = desired_steps
    checkpoints: list[tuple[int, torch.Tensor, float]] = []
    checkpoint_steps = _checkpoint_steps(desired_steps, max_checkpoints)
    completed_steps = 0
    for step in range(1, desired_steps + 1):
        if step > effective_max_steps:
            break
        if time.monotonic() >= deadline:
            break
        step_started_perf = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = _pass_loss(
            work,
            edge_index,
            node_sizes,
            topo_depth,
            mode,
            floors,
            pass_id,
            stress_sample,
            shape_geometry,
        )
        if not bool(torch.isfinite(loss).all().item()):
            break
        loss.backward()
        optimizer.step()
        step_wall_s = max(1.0e-6, time.perf_counter() - step_started_perf)
        if step_timing_hook is not None:
            step_timing_hook(step, step_wall_s)
        if step == 1:
            remaining_step_budget = max(0.0, deadline - time.monotonic())
            steps_that_fit = 1 + int(remaining_step_budget / step_wall_s)
            effective_max_steps = max(1, min(desired_steps, steps_that_fit))
            checkpoint_steps = _checkpoint_steps(effective_max_steps, max_checkpoints)
        if mode == "x_only":
            with torch.no_grad():
                work[:, 1] = start_y
        completed_steps = step
        if step in checkpoint_steps:
            checkpoint_pos = work.detach().clone()
            checkpoint_loss_tensor = _pass_loss(
                checkpoint_pos,
                edge_index,
                node_sizes,
                topo_depth,
                mode,
                floors,
                pass_id,
                stress_sample,
                shape_geometry,
            )
            checkpoint_loss = float(checkpoint_loss_tensor.detach().item())
            checkpoints.append((step, checkpoint_pos, checkpoint_loss))
    return work.detach(), completed_steps, start_loss, checkpoints


def _run_optimize_seed_pass(
    seed: W5Seed,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    topo_depth: torch.Tensor,
    mode: str,
    deadline: float,
    honest_axes: Optional[W5HonestAxes],
    *,
    max_steps: Optional[int],
    max_checkpoints: int,
    pass_id: int,
    stress_sample: Optional[W5StressSample],
    shape_geometry: Optional[NativeShapeGeometry] = None,
) -> tuple[torch.Tensor, int, float, list[tuple[int, torch.Tensor, float]]]:
    """Call the active optimizer with optional pass-2 arguments when supported.

    Parameters
    ----------
    seed : W5Seed
        Warm start for this pass.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    topo_depth : torch.Tensor
        Depth tensor with shape ``[N]``.
    mode : str
        Routed W5 mode.
    deadline : float
        Absolute optimizer deadline.
    honest_axes : W5HonestAxes, optional
        Honest incumbent axis scores.
    max_steps : int, optional
        Maximum optimizer steps.
    max_checkpoints : int
        Maximum pass checkpoints.
    pass_id : int
        Surrogate pass identifier.
    stress_sample : W5StressSample, optional
        Fixed sampled graph-distance pairs for pass 2.
    shape_geometry : NativeShapeGeometry, optional
        Optional non-box shape descriptors.

    Returns
    -------
    tuple[torch.Tensor, int, float, list[tuple[int, torch.Tensor, float]]]
        Final positions, completed steps, start loss, and checkpoint
        positions/losses.
    """
    if shape_geometry is None:
        return _optimize_seed(
            seed,
            edge_index,
            node_sizes,
            topo_depth,
            mode,
            deadline,
            honest_axes,
            max_steps=max_steps,
            max_checkpoints=max_checkpoints,
            pass_id=pass_id,
            stress_sample=stress_sample,
        )
    return _optimize_seed(
        seed,
        edge_index,
        node_sizes,
        topo_depth,
        mode,
        deadline,
        honest_axes,
        max_steps=max_steps,
        max_checkpoints=max_checkpoints,
        pass_id=pass_id,
        stress_sample=stress_sample,
        shape_geometry=shape_geometry,
    )


def _measure_one_surrogate_step_s(
    seed: W5Seed,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    topo_depth: torch.Tensor,
    mode: str,
    honest_axes: Optional[W5HonestAxes],
    measurement_budget_s: float,
    shape_geometry: Optional[NativeShapeGeometry] = None,
) -> W5StepMeasurement:
    """Measure wall-clock cost for one steady-state W5 surrogate step.

    Parameters
    ----------
    seed : W5Seed
        Finite warm start used as the measurement surrogate.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    topo_depth : torch.Tensor
        Depth tensor with shape ``[N]``.
    mode : str
        Routed W5 optimization mode.
    honest_axes : W5HonestAxes, optional
        Honest incumbent axes used by barrier weighting.
    measurement_budget_s : float
        Wall-clock seconds available for the surrogate probe.
    shape_geometry : NativeShapeGeometry, optional
        Optional non-box shape descriptors.

    Returns
    -------
    W5StepMeasurement
        Positive wall-clock seconds for steady-state step cost and one-time
        first-step warmup.
    """
    step_times_s: list[float] = []

    def timing_hook(_step: int, duration_s: float) -> None:
        """Record one measured optimizer step duration.

        Parameters
        ----------
        _step : int
            Completed step index, unused by the averaging logic.
        duration_s : float
            Wall-clock duration for the completed step.

        Returns
        -------
        None
            ``step_times_s`` is appended in place.
        """
        step_times_s.append(duration_s)

    if shape_geometry is None:
        _optimize_seed(
            seed,
            edge_index,
            node_sizes,
            topo_depth,
            mode,
            time.monotonic() + max(1.0e-6, measurement_budget_s),
            honest_axes,
            max_steps=_MEASURED_COST_SURROGATE_STEPS,
            max_checkpoints=0,
            step_timing_hook=timing_hook,
        )
    else:
        _optimize_seed(
            seed,
            edge_index,
            node_sizes,
            topo_depth,
            mode,
            time.monotonic() + max(1.0e-6, measurement_budget_s),
            honest_axes,
            max_steps=_MEASURED_COST_SURROGATE_STEPS,
            max_checkpoints=0,
            step_timing_hook=timing_hook,
            shape_geometry=shape_geometry,
        )
    if not step_times_s:
        return W5StepMeasurement(step_s=1.0e-6, warmup_s=0.0)
    warmup_s = max(0.0, step_times_s[0])
    if len(step_times_s) == 1:
        step_s = warmup_s
        warmup_s = 0.0
    else:
        steady_times_s = step_times_s[1:]
        step_s = sum(steady_times_s) / float(len(steady_times_s))
    return W5StepMeasurement(step_s=max(1.0e-6, step_s), warmup_s=warmup_s)


def _measured_cost_plan(
    *,
    seeds: Sequence[W5Seed],
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    topo_depth: torch.Tensor,
    routed_mode: str,
    slice_s: float,
    config: Optional[LayoutConfig],
    started_perf: float,
    started_process: float,
    remaining_entry: Optional[float],
    honest_axes: Optional[W5HonestAxes],
    shape_geometry: Optional[NativeShapeGeometry] = None,
) -> Optional[W5CostPlan]:
    """Return a measured W5 plan that fits the deterministic process cap.

    Parameters
    ----------
    seeds : Sequence[W5Seed]
        Finite deduplicated W5 seeds.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    topo_depth : torch.Tensor
        Depth tensor with shape ``[N]``.
    routed_mode : str
        First routed W5 mode.
    slice_s : float
        Work slice admitted by deadline gates.
    config : LayoutConfig, optional
        Prepared configuration carrying spend and referee measurements.
    started_perf : float
        ``time.perf_counter()`` value captured at W5 entry.
    started_process : float
        ``time.process_time()`` value captured at W5 entry.
    remaining_entry : float, optional
        Benchmark process seconds remaining at W5 entry.
    honest_axes : W5HonestAxes, optional
        Honest incumbent axes used by barrier weighting.
    shape_geometry : NativeShapeGeometry, optional
        Optional non-box shape descriptors.

    Returns
    -------
    W5CostPlan or None
        Admitted plan, or ``None`` when one seed and one checkpoint cannot fit.
    """
    referee_s = float(
        getattr(config, "_dagua_native_w5_referee_cost_s", _MEASURED_COST_DEFAULT_REFEREE_S)
    )
    referee_s = max(1.0e-6, referee_s)
    node_count = int(seeds[0].pos.shape[0])
    use_deterministic_costs = _use_tiny_row_deterministic_w5_costs(config, node_count)
    pre_measure_cap_remaining = _w5_spend_cap_s(config, remaining_entry) - _w5_process_spent_s(
        config,
        started_process,
    )
    measurement_budget_s = max(
        1.0e-6,
        min(float(slice_s), pre_measure_cap_remaining) - _PREDICTED_COST_RETURN_RESERVE_S,
    )
    step_measurement = _measure_one_surrogate_step_s(
        seeds[0],
        edge_index,
        node_sizes,
        topo_depth,
        routed_mode,
        honest_axes,
        measurement_budget_s,
        shape_geometry,
    )
    spend_cap = _w5_spend_cap_s(config, remaining_entry)
    if use_deterministic_costs:
        cap_remaining = spend_cap
    else:
        cap_remaining = spend_cap - _w5_process_spent_s(config, started_process)
    budget_s = max(0.0, min(float(slice_s), cap_remaining))
    # Admission uses process seconds so sibling-worker load cannot change plan
    # size. The wall deadline below remains the hard runaway guard.
    usable_s = budget_s - _PREDICTED_COST_RETURN_RESERVE_S
    step_s = (
        _TINY_ROW_DETERMINISTIC_STEP_COST_S if use_deterministic_costs else step_measurement.step_s
    )
    warmup_s = step_measurement.warmup_s
    if use_deterministic_costs:
        referee_s = _TINY_ROW_DETERMINISTIC_REFEREE_COST_S
    minimum_predicted_s = step_s + referee_s
    if usable_s < minimum_predicted_s:
        if config is not None:
            setattr(
                config,
                "_dagua_native_w5_cost_plan",
                W5CostPlan(
                    seeds=0,
                    steps=0,
                    checkpoints=0,
                    measured_step_s=step_s,
                    warmup_s=warmup_s,
                    referee_s=referee_s,
                    budget_s=budget_s,
                    budget_usable_s=usable_s,
                    predicted_s=minimum_predicted_s,
                ),
            )
        return None
    is_tiny_referee = (
        node_count <= _MEASURED_COST_TINY_MAX_N and referee_s < _MEASURED_COST_TINY_REFEREE_S
    )
    base_steps = 24 if node_count >= 300 else 36
    base_seeds = min(3, len(seeds))
    base_checkpoints = _MEASURED_COST_MAX_CHECKPOINTS
    max_checkpoints = base_checkpoints

    def build_plan(seed_count: int, steps: int, checkpoints: int) -> W5CostPlan:
        """Build a process-denominated plan payload for candidate work bounds.

        Parameters
        ----------
        seed_count : int
            Number of W5 seeds to run.
        steps : int
            Optimizer steps per seed.
        checkpoints : int
            Honest checkpoints per seed.

        Returns
        -------
        W5CostPlan
            Cost plan for post-probe work under the remaining process budget.
        """
        predicted_s = seed_count * (steps * step_s + checkpoints * referee_s)
        return W5CostPlan(
            seeds=seed_count,
            steps=steps,
            checkpoints=checkpoints,
            measured_step_s=step_s,
            warmup_s=warmup_s,
            referee_s=referee_s,
            budget_s=budget_s,
            budget_usable_s=usable_s,
            predicted_s=predicted_s,
        )

    base_plan = build_plan(base_seeds, base_steps, base_checkpoints)
    if base_plan.predicted_s <= usable_s:
        if is_tiny_referee:
            raised_usable_s = min(usable_s, base_plan.predicted_s + _TINY_ROW_CONTINUATION_CAP_S)
            for steps in range(_MEASURED_COST_TINY_STEPS, base_steps - 1, -1):
                for checkpoints in range(
                    _MEASURED_COST_TINY_MAX_CHECKPOINTS,
                    base_checkpoints - 1,
                    -1,
                ):
                    raised_plan = build_plan(base_seeds, steps, checkpoints)
                    if raised_plan.predicted_s <= raised_usable_s:
                        if config is not None:
                            setattr(config, "_dagua_native_w5_cost_plan", raised_plan)
                        return raised_plan
        if config is not None:
            setattr(config, "_dagua_native_w5_cost_plan", base_plan)
        return base_plan

    max_seeds = min(_MEASURED_COST_MAX_SEEDS, base_seeds)
    for steps in range(base_steps, 0, -1):
        for checkpoints in range(max_checkpoints, 0, -1):
            for seed_count in range(max_seeds, 0, -1):
                candidate_plan = build_plan(seed_count, steps, checkpoints)
                if candidate_plan.predicted_s <= usable_s:
                    if config is not None:
                        setattr(config, "_dagua_native_w5_cost_plan", candidate_plan)
                    return candidate_plan
    return None


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
    incumbent_axes: Optional[W5HonestAxes] = None,
    shape_geometry: Optional[NativeShapeGeometry] = None,
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
    incumbent_axes : W5HonestAxes, optional
        Honest incumbent axes from the same ``full()`` metrics pass that
        produced ``incumbent_score_pair``.
    shape_geometry : NativeShapeGeometry, optional
        Optional non-box shape descriptors for overlap loss and viability.

    Returns
    -------
    W5FinisherResult
        Anytime winner plus telemetry for all honest-scored checkpoints.
    """
    started_perf = time.perf_counter()
    started_process = time.process_time()
    remaining_entry = _process_remaining_s(config)
    slice_s = _finisher_slice_s(config)
    node_count = int(incumbent_pos.shape[0])
    edge_count = int(edge_index.shape[1]) if edge_index.ndim == 2 else 0
    predicted_skip_reason = w5_predicted_skip_reason(node_count, edge_count, config)
    cost_plan: Optional[W5CostPlan] = None
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
        phase_timings: list[W5PhaseTiming],
        viability_counts: dict[str, int],
        viability_drop_counts: dict[str, int],
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
        phase_timings : list[W5PhaseTiming]
            Per-seed phase timing records.
        viability_counts : dict[str, int]
            Viability outcome counts.
        viability_drop_counts : dict[str, int]
            Pre-score viability drop counts by reason.
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
            previous_process_spent = float(getattr(config, "_dagua_native_w5_process_spent_s", 0.0))
            process_spent_s = max(0.0, time.process_time() - started_process)
            setattr(
                config,
                "_dagua_native_w5_process_spent_s",
                previous_process_spent + process_spent_s,
            )
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
            graph_name=_graph_name(config),
            incumbent_axes=incumbent_axes,
            phase_timings_s=tuple(phase_timings),
            viability_counts=dict(viability_counts),
            viability_drop_counts=dict(viability_drop_counts),
            cost_plan=cost_plan,
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
            phase_timings=[],
            viability_counts={},
            viability_drop_counts={},
            mode="skip",
            steps=0,
            skipped_reason=predicted_skip_reason or "no_budget",
        )
    use_measured_cost = bool(getattr(config, "_dagua_native_w5_measured_sizing", False))
    kept_seeds = _dedupe_seeds(
        seeds,
        max_seeds=3,
    )
    if not kept_seeds:
        return finish(
            winner_pos=incumbent_pos,
            winner_score_pair=incumbent_score_pair,
            winner_name="incumbent",
            deadline_returned=False,
            accepted=[],
            rejected=[],
            checkpoints=[],
            phase_timings=[],
            viability_counts={},
            viability_drop_counts={},
            mode="skip",
            steps=0,
            skipped_reason="no_finite_seed",
        )
    deadline = time.monotonic() + slice_s
    edge_work = edge_index.detach().to(device=kept_seeds[0].pos.device, dtype=torch.long)
    size_work = node_sizes.detach().to(device=kept_seeds[0].pos.device, dtype=torch.float32)
    shape_work = (
        shape_geometry.to(device=kept_seeds[0].pos.device, dtype=torch.float32)
        if shape_geometry is not None
        else None
    )
    topo_depth = _longest_path_depth(
        edge_work,
        int(kept_seeds[0].pos.shape[0]),
        kept_seeds[0].pos.device,
    )
    stress_sample: Optional[W5StressSample] = None
    stress_sample_ready = False
    max_steps: Optional[int] = None
    max_checkpoints = _MEASURED_COST_MAX_CHECKPOINTS
    if use_measured_cost:
        first_mode = _route_mode(
            kept_seeds[0].pos.detach().to(device=edge_work.device, dtype=torch.float32),
            edge_work,
            topo_depth,
            is_semantically_directed=is_semantically_directed,
            declared_hierarchical=declared_hierarchical,
            direction_is_declared=direction_is_declared,
            honest_axes=incumbent_axes,
        )
        cost_plan = _measured_cost_plan(
            seeds=kept_seeds,
            edge_index=edge_work,
            node_sizes=size_work,
            topo_depth=topo_depth,
            routed_mode=first_mode,
            slice_s=float(slice_s),
            config=config,
            started_perf=started_perf,
            started_process=started_process,
            remaining_entry=remaining_entry,
            honest_axes=incumbent_axes,
            shape_geometry=shape_work,
        )
        if cost_plan is None:
            cost_plan = getattr(config, "_dagua_native_w5_cost_plan", None)
            return finish(
                winner_pos=incumbent_pos,
                winner_score_pair=incumbent_score_pair,
                winner_name="incumbent",
                deadline_returned=True,
                accepted=[],
                rejected=[],
                checkpoints=[],
                phase_timings=[],
                viability_counts={},
                viability_drop_counts={},
                mode="skip",
                steps=0,
                skipped_reason="predicted_cost_measured",
            )
        kept_seeds = kept_seeds[: cost_plan.seeds]
        max_steps = cost_plan.steps
        max_checkpoints = cost_plan.checkpoints
    winner_pos = incumbent_pos
    winner_score_pair = incumbent_score_pair
    winner_name = "incumbent"
    accepted: list[W5Candidate] = []
    rejected: list[W5Checkpoint] = []
    checkpoints: list[W5Checkpoint] = []
    phase_timings: list[W5PhaseTiming] = []
    viability_counts: dict[str, int] = {}
    viability_drop_counts: dict[str, int] = {}
    steps_total = 0
    routed_mode = "skip"
    incumbent_overlap = _overlap_count(incumbent_pos, size_work, shape_work)
    deadline_returned = False
    first_score_epilogue_attempted = False
    for seed in kept_seeds:
        seed_accepted_entry = len(accepted)
        if _w5_process_spent_s(config, started_process) >= _w5_spend_cap_s(
            config,
            remaining_entry,
        ):
            deadline_returned = True
            break
        if time.monotonic() >= deadline - _FINISHER_SCORE_RESERVE_S:
            deadline_returned = True
            break
        route_started = time.perf_counter()
        mode = _route_mode(
            seed.pos.detach().to(device=edge_work.device, dtype=torch.float32),
            edge_work,
            topo_depth,
            is_semantically_directed=is_semantically_directed,
            declared_hierarchical=declared_hierarchical,
            direction_is_declared=direction_is_declared,
            honest_axes=incumbent_axes,
        )
        route_s = max(0.0, time.perf_counter() - route_started)
        for ladder_index, mode in enumerate(
            _mode_ladder(mode, is_semantically_directed=is_semantically_directed)
        ):
            if ladder_index > 0 and _w5_process_spent_s(
                config,
                started_process,
            ) >= _w5_spend_cap_s(config, remaining_entry):
                deadline_returned = True
                break
            if time.monotonic() >= deadline - _FINISHER_SCORE_RESERVE_S:
                deadline_returned = True
                break
            routed_mode = mode
            mode_seed = (
                W5Seed(f"{seed.name}_incumbent", winner_pos)
                if len(accepted) > seed_accepted_entry
                else seed
            )
            pass_seed = mode_seed
            for pass_id in (1, 2):
                if pass_id == 2:
                    if deadline_returned:
                        break
                    if _w5_process_spent_s(config, started_process) >= _w5_spend_cap_s(
                        config,
                        remaining_entry,
                    ):
                        deadline_returned = True
                        break
                    if time.monotonic() >= deadline - _FINISHER_SCORE_RESERVE_S:
                        deadline_returned = True
                        break
                    if not stress_sample_ready:
                        all_pairs_dist = _closed_over_all_pairs_dist(score_fn)
                        stress_sample = _build_w5_stress_sample(
                            edge_work,
                            int(kept_seeds[0].pos.shape[0]),
                            all_pairs_dist,
                            kept_seeds[0].pos.device,
                        )
                        stress_sample_ready = True
                optimize_s = 0.0
                viability_s = 0.0
                score_s = 0.0
                try:
                    optimize_started = time.perf_counter()
                    final_pos, steps, start_loss, scored_points = _run_optimize_seed_pass(
                        pass_seed,
                        edge_work,
                        size_work,
                        topo_depth,
                        mode,
                        deadline - _FINISHER_SCORE_RESERVE_S,
                        incumbent_axes,
                        max_steps=max_steps if use_measured_cost else None,
                        max_checkpoints=max_checkpoints,
                        pass_id=pass_id,
                        stress_sample=stress_sample,
                        shape_geometry=shape_work,
                    )
                    optimize_s = max(0.0, time.perf_counter() - optimize_started)
                except Exception as exc:  # noqa: BLE001 -- W5 is optional candidate generation
                    if is_worker_timeout_like_exception(exc):
                        raise
                    _LOGGER.warning("W5 finisher seed %s failed", seed.name, exc_info=True)
                    phase_timings.append(
                        W5PhaseTiming(
                            seed=seed.name,
                            mode=mode,
                            pass_id=pass_id,
                            route_s=route_s if ladder_index == 0 and pass_id == 1 else 0.0,
                            optimize_s=optimize_s,
                            viability_s=viability_s,
                            score_s=score_s,
                        )
                    )
                    continue
                steps_total += steps
                if not scored_points or scored_points[-1][0] != steps:
                    final_loss = float(
                        _pass_loss(
                            final_pos,
                            edge_work,
                            size_work,
                            topo_depth,
                            mode,
                            {},
                            pass_id,
                            stress_sample,
                            shape_work,
                        )
                        .detach()
                        .item()
                    )
                    scored_points.append((steps, final_pos, final_loss))
                for step, checkpoint_pos, checkpoint_loss in scored_points[:max_checkpoints]:
                    epilogue_scoring = False
                    if _w5_process_spent_s(config, started_process) >= _w5_spend_cap_s(
                        config,
                        remaining_entry,
                    ):
                        deadline_returned = True
                        if (
                            checkpoints
                            or first_score_epilogue_attempted
                            or not scored_points
                            or not _w5_first_score_epilogue_has_wall_headroom(config)
                        ):
                            break
                        step, checkpoint_pos, checkpoint_loss = scored_points[-1]
                        first_score_epilogue_attempted = True
                        epilogue_scoring = True
                    if not epilogue_scoring and time.monotonic() >= deadline:
                        deadline_returned = True
                        if (
                            checkpoints
                            or first_score_epilogue_attempted
                            or not scored_points
                            or not _w5_first_score_epilogue_has_wall_headroom(config)
                        ):
                            break
                        step, checkpoint_pos, checkpoint_loss = scored_points[-1]
                        first_score_epilogue_attempted = True
                        epilogue_scoring = True
                    viability_started = time.perf_counter()
                    checkpoint_overlap = _overlap_count(checkpoint_pos, size_work, shape_work)
                    if checkpoint_overlap > incumbent_overlap:
                        _increment_count(viability_counts, "projected_overlap_candidate")
                        if shape_work is None:
                            checkpoint_pos = _project_checkpoint_for_viability(
                                checkpoint_pos,
                                size_work,
                            )
                        else:
                            checkpoint_pos = _project_checkpoint_for_viability(
                                checkpoint_pos,
                                size_work,
                                shape_work,
                            )
                        projected_overlap = _overlap_count(checkpoint_pos, size_work, shape_work)
                        if projected_overlap <= incumbent_overlap:
                            _increment_count(viability_counts, "projection_resolved_overlap")
                    if _is_degenerate(checkpoint_pos, size_work):
                        _increment_count(viability_counts, "drop_degenerate")
                        _increment_count(viability_drop_counts, "degenerate")
                        viability_s += max(0.0, time.perf_counter() - viability_started)
                        if epilogue_scoring:
                            break
                        continue
                    if _overlap_count(checkpoint_pos, size_work, shape_work) > incumbent_overlap:
                        _increment_count(viability_counts, "drop_overlap_regressed")
                        _increment_count(viability_drop_counts, "overlap_regressed")
                        viability_s += max(0.0, time.perf_counter() - viability_started)
                        if epilogue_scoring:
                            break
                        continue
                    _increment_count(viability_counts, "scored_viable")
                    viability_s += max(0.0, time.perf_counter() - viability_started)
                    try:
                        score_started = time.perf_counter()
                        score_pos = checkpoint_pos.to(device=edge_index.device, dtype=torch.float32)
                        honest = score_fn(score_pos)
                        score_s += max(0.0, time.perf_counter() - score_started)
                    except Exception as exc:
                        score_s += max(0.0, time.perf_counter() - score_started)
                        if is_worker_timeout_like_exception(exc):
                            raise
                        _increment_count(viability_counts, "drop_score_exception")
                        if epilogue_scoring:
                            break
                        continue
                    if not math.isfinite(honest.directed) or not math.isfinite(honest.undirected):
                        _increment_count(viability_counts, "drop_nonfinite_score")
                        if epilogue_scoring:
                            break
                        continue
                    directed_delta = honest.directed - winner_score_pair.directed
                    undirected_delta = honest.undirected - winner_score_pair.undirected
                    surrogate_delta = start_loss - checkpoint_loss
                    is_accepted = w5_dominates(honest, winner_score_pair, float(accept_margin))
                    reason = "dominates" if is_accepted else "does_not_dominate_both"
                    checkpoint = W5Checkpoint(
                        seed=seed.name,
                        mode=mode,
                        pass_id=pass_id,
                        step=int(step),
                        surrogate_delta=float(surrogate_delta),
                        honest_delta=float(directed_delta),
                        undirected_honest_delta=float(undirected_delta),
                        honest_score_pair=honest,
                        accepted=is_accepted,
                        reason=reason,
                        pass_spend_s=float(optimize_s + viability_s + score_s),
                    )
                    checkpoints.append(checkpoint)
                    if is_accepted:
                        name = f"w5_p{pass_id}_{mode}_{seed.name}_{step}"
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
                        incumbent_overlap = min(
                            incumbent_overlap,
                            _overlap_count(winner_pos, size_work, shape_work),
                        )
                    else:
                        rejected.append(checkpoint)
                    if epilogue_scoring:
                        break
                phase_timings.append(
                    W5PhaseTiming(
                        seed=seed.name,
                        mode=mode,
                        pass_id=pass_id,
                        route_s=route_s if ladder_index == 0 and pass_id == 1 else 0.0,
                        optimize_s=optimize_s,
                        viability_s=viability_s,
                        score_s=score_s,
                    )
                )
                pass_seed_pos = winner_pos
                pass_seed = W5Seed(f"{seed.name}_p{pass_id}", pass_seed_pos)
                if deadline_returned:
                    break
            if deadline_returned:
                break
    skipped = None if accepted else ("no_checkpoint_improved" if checkpoints else "no_checkpoint")
    return finish(
        winner_pos=winner_pos,
        winner_score_pair=winner_score_pair,
        winner_name=winner_name,
        deadline_returned=deadline_returned,
        accepted=accepted,
        rejected=rejected,
        checkpoints=checkpoints,
        phase_timings=phase_timings,
        viability_counts=viability_counts,
        viability_drop_counts=viability_drop_counts,
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

    def axes_payload(axes: Optional[W5HonestAxes]) -> Optional[dict[str, Optional[float]]]:
        """Convert honest route axes to a JSON-serializable payload.

        Parameters
        ----------
        axes : W5HonestAxes, optional
            Honest incumbent axes used by the route.

        Returns
        -------
        dict[str, float | None] or None
            JSON-ready axis payload.
        """
        if axes is None:
            return None
        return {
            "flow": axes.flow,
            "depth": axes.depth,
            "ksm": axes.ksm,
            "edge_length": axes.edge_length,
        }

    payload = {
        "event": "native_w5_finisher",
        "graph_name": result.graph_name,
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
        "measured_step_s": (None if result.cost_plan is None else result.cost_plan.measured_step_s),
        "warmup_s": None if result.cost_plan is None else result.cost_plan.warmup_s,
        "referee_s": None if result.cost_plan is None else result.cost_plan.referee_s,
        "budget_usable_s": (None if result.cost_plan is None else result.cost_plan.budget_usable_s),
        "predicted_s": None if result.cost_plan is None else result.cost_plan.predicted_s,
        "plan_seeds": None if result.cost_plan is None else result.cost_plan.seeds,
        "plan_steps": None if result.cost_plan is None else result.cost_plan.steps,
        "plan_checkpoints": None if result.cost_plan is None else result.cost_plan.checkpoints,
        "phase_timings_s": [
            {
                "seed": timing.seed,
                "mode": timing.mode,
                "pass_id": timing.pass_id,
                "route_s": timing.route_s,
                "optimize_s": timing.optimize_s,
                "viability_s": timing.viability_s,
                "score_s": timing.score_s,
            }
            for timing in result.phase_timings_s
        ],
        "viability_counts": result.viability_counts,
        "viability_drop_counts": result.viability_drop_counts,
        "winner_name": result.winner_name,
        "incumbent_score_pair": pair_payload(result.incumbent_score_pair),
        "incumbent_axes": axes_payload(result.incumbent_axes),
        "winner_score_pair": pair_payload(result.winner_score_pair),
        "accepted": [candidate.name for candidate in result.accepted],
        "rejected": [
            {
                "seed": checkpoint.seed,
                "mode": checkpoint.mode,
                "pass_id": checkpoint.pass_id,
                "step": checkpoint.step,
                "reason": checkpoint.reason,
            }
            for checkpoint in result.rejected
        ],
        "checkpoints": [
            {
                "seed": checkpoint.seed,
                "mode": checkpoint.mode,
                "pass_id": checkpoint.pass_id,
                "step": checkpoint.step,
                "surrogate_delta": checkpoint.surrogate_delta,
                "directed_honest_delta": checkpoint.honest_delta,
                "undirected_honest_delta": checkpoint.undirected_honest_delta,
                "honest_score_pair": pair_payload(checkpoint.honest_score_pair),
                "accepted": checkpoint.accepted,
                "reason": checkpoint.reason,
                "pass_spend_s": checkpoint.pass_spend_s,
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
    "W5HonestAxes",
    "W5PhaseTiming",
    "W5ScorePair",
    "W5Seed",
    "is_worker_timeout_like_exception",
    "log_w5_telemetry",
    "make_w5_skip_result",
    "run_w5_finisher",
    "w5_honest_axes_from_metrics",
    "w5_dominates",
]
