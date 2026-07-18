"""Tests for the W5 native finisher wrapper."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional

import pytest
import torch

from dagua.config import LayoutConfig
from dagua.layout.ops.pipelines.native_finisher import (
    W5HonestAxes,
    W5ScorePair,
    W5Seed,
    log_w5_telemetry,
    run_w5_finisher,
    w5_predicted_skip_reason,
)


class _WorkerLayoutTimeoutError(RuntimeError):
    """Local stand-in for the benchmark worker alarm exception."""


def _pair(directed: float, undirected: float) -> W5ScorePair:
    """Build a W5 score pair for tests.

    Parameters
    ----------
    directed : float
        Directed composite score.
    undirected : float
        Undirected composite score.

    Returns
    -------
    W5ScorePair
        Test score pair.
    """
    return W5ScorePair(directed=directed, undirected=undirected)


def _tiny_layout() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a finite four-node layout for W5 tests.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Position, edge-index, and node-size tensors.
    """
    pos = torch.tensor(
        [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]],
        dtype=torch.float32,
    )
    edge_index = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    node_sizes = torch.full((4, 2), 2.0)
    return pos, edge_index, node_sizes


def test_w5_finisher_returns_exact_incumbent_when_candidates_are_worse() -> None:
    """W5 returns the incumbent tensor values when every checkpoint is worse."""
    pos, edge_index, node_sizes = _tiny_layout()

    result = run_w5_finisher(
        incumbent_pos=pos,
        incumbent_score_pair=_pair(10.0, 10.0),
        seeds=[W5Seed("incumbent", pos)],
        edge_index=edge_index,
        node_sizes=node_sizes,
        score_fn=lambda candidate: _pair(9.0, 9.0),
        is_semantically_directed=False,
        declared_hierarchical=False,
    )

    assert torch.equal(result.winner_pos, pos)
    assert result.winner_score_pair == _pair(10.0, 10.0)
    assert result.accepted == ()
    assert all(not checkpoint.accepted for checkpoint in result.checkpoints)


def test_w5_finisher_deadline_returns_exact_incumbent() -> None:
    """Expired benchmark budget returns the incumbent with deadline telemetry."""
    config = LayoutConfig()
    config._dagua_native_deadline_s = time.perf_counter() + 1.0
    config._dagua_native_total_budget_s = 300.0
    pos, edge_index, node_sizes = _tiny_layout()

    result = run_w5_finisher(
        incumbent_pos=pos,
        incumbent_score_pair=_pair(0.0, 0.0),
        seeds=[W5Seed("incumbent", pos)],
        edge_index=edge_index,
        node_sizes=node_sizes,
        score_fn=lambda candidate: _pair(1.0, 1.0),
        is_semantically_directed=False,
        declared_hierarchical=False,
        config=config,
    )

    assert result.skipped_reason == "no_budget"
    assert result.deadline_returned is True
    assert torch.equal(result.winner_pos, pos)


def test_w5_finisher_large_graph_runs_when_measured_budget_fits() -> None:
    """Measured cost sizing admits a large row with the real wall-clock surrogate."""
    pos = torch.stack(
        (
            torch.arange(500, dtype=torch.float32),
            torch.zeros(500, dtype=torch.float32),
        ),
        dim=1,
    )
    edge_index = torch.empty((2, 0), dtype=torch.long)
    node_sizes = torch.full((500, 2), 2.0)
    config = LayoutConfig()
    config._dagua_native_deadline_s = time.perf_counter() + 300.0
    config._dagua_native_total_budget_s = 300.0
    config._dagua_native_w5_referee_cost_s = 0.01
    config._dagua_native_w5_measured_sizing = True
    scored = {"count": 0}

    def score_fn(candidate: torch.Tensor) -> W5ScorePair:
        """Count honest score calls and return a non-dominating score.

        Parameters
        ----------
        candidate : torch.Tensor
            Candidate positions with shape ``[N, 2]``.

        Returns
        -------
        W5ScorePair
            Finite non-dominating score pair.
        """
        del candidate
        scored["count"] += 1
        return _pair(0.0, 0.0)

    result = run_w5_finisher(
        incumbent_pos=pos,
        incumbent_score_pair=_pair(0.0, 0.0),
        seeds=[W5Seed("incumbent", pos)],
        edge_index=edge_index,
        node_sizes=node_sizes,
        score_fn=score_fn,
        is_semantically_directed=False,
        declared_hierarchical=False,
        config=config,
    )

    assert scored["count"] > 0
    assert result.skipped_reason == "no_checkpoint_improved"
    assert result.deadline_returned is False
    assert torch.equal(result.winner_pos, pos)
    assert result.cost_plan is not None
    assert result.cost_plan.steps > 0
    assert result.cost_plan.measured_step_s < 1.0
    assert result.cost_plan.budget_usable_s >= result.cost_plan.predicted_s


def test_measured_cost_plan_uses_wall_surrogate_and_restores_small_row_work() -> None:
    """The real measured plan uses steady-state wall cost and admits base work."""
    import importlib

    native_finisher = importlib.import_module("dagua.layout.ops.pipelines.native_finisher")
    large_pos = torch.stack(
        (
            torch.arange(500, dtype=torch.float32),
            torch.zeros(500, dtype=torch.float32),
        ),
        dim=1,
    )
    large_edges = torch.empty((2, 0), dtype=torch.long)
    large_sizes = torch.full((500, 2), 2.0)
    large_depth = torch.zeros(500, dtype=torch.long)

    measurement = native_finisher._measure_one_surrogate_step_s(
        W5Seed("large", large_pos),
        large_edges,
        large_sizes,
        large_depth,
        "undirected_2d_sampled",
        None,
        18.0,
    )
    assert measurement.step_s < 1.0
    assert measurement.warmup_s >= 0.0

    large_config = LayoutConfig()
    large_config._dagua_native_deadline_s = time.perf_counter() + 300.0
    large_config._dagua_native_total_budget_s = 300.0
    large_config._dagua_native_w5_referee_cost_s = 0.01
    large_plan = native_finisher._measured_cost_plan(
        seeds=[W5Seed("large", large_pos)],
        edge_index=large_edges,
        node_sizes=large_sizes,
        topo_depth=large_depth,
        routed_mode="undirected_2d_sampled",
        slice_s=20.0,
        config=large_config,
        started_perf=time.perf_counter(),
        remaining_entry=300.0,
        honest_axes=None,
    )
    assert large_plan is not None
    assert large_plan.steps > 0
    assert large_plan.measured_step_s < 1.0

    small_pos = torch.stack(
        (
            torch.arange(200, dtype=torch.float32),
            torch.zeros(200, dtype=torch.float32),
        ),
        dim=1,
    )
    small_edges = torch.empty((2, 0), dtype=torch.long)
    small_sizes = torch.full((200, 2), 2.0)
    small_depth = torch.zeros(200, dtype=torch.long)
    small_config = LayoutConfig()
    small_config._dagua_native_deadline_s = time.perf_counter() + 300.0
    small_config._dagua_native_total_budget_s = 300.0
    small_config._dagua_native_w5_referee_cost_s = 0.001
    small_plan = native_finisher._measured_cost_plan(
        seeds=[
            W5Seed("small_a", small_pos),
            W5Seed("small_b", small_pos + 10.0),
            W5Seed("small_c", small_pos + 20.0),
        ],
        edge_index=small_edges,
        node_sizes=small_sizes,
        topo_depth=small_depth,
        routed_mode="undirected_2d_sampled",
        slice_s=20.0,
        config=small_config,
        started_perf=time.perf_counter(),
        remaining_entry=300.0,
        honest_axes=None,
    )
    assert small_plan is not None
    assert small_plan.steps == 36
    assert small_plan.seeds == 3
    assert small_plan.predicted_s == pytest.approx(
        small_plan.seeds
        * (
            small_plan.steps * small_plan.measured_step_s
            + small_plan.checkpoints * small_plan.referee_s
        )
    )


def test_measured_surrogate_probe_averages_three_post_warmup_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Measured admission keeps step one as warmup and averages steps two through four."""
    import importlib

    native_finisher = importlib.import_module("dagua.layout.ops.pipelines.native_finisher")
    pos, edge_index, node_sizes = _tiny_layout()
    topo_depth = torch.zeros(pos.shape[0], dtype=torch.long)
    step_times_s = [0.6, 0.2, 0.3, 0.4]

    def fake_optimize_seed(
        seed: W5Seed,
        edge_work: torch.Tensor,
        size_work: torch.Tensor,
        depth_work: torch.Tensor,
        mode: str,
        deadline: float,
        honest_axes: Optional[W5HonestAxes] = None,
        max_steps: Optional[int] = None,
        max_checkpoints: int = 2,
        step_timing_hook: Optional[Callable[[int, float], None]] = None,
    ) -> tuple[torch.Tensor, int, float, list[tuple[int, torch.Tensor, float]]]:
        """Emit deterministic per-step timings for the probe measurement."""
        del edge_work, size_work, depth_work, mode, deadline, honest_axes, max_checkpoints
        assert max_steps == 4
        assert step_timing_hook is not None
        for step, duration_s in enumerate(step_times_s, start=1):
            step_timing_hook(step, duration_s)
        return seed.pos, len(step_times_s), 0.0, []

    monkeypatch.setattr(native_finisher, "_optimize_seed", fake_optimize_seed)

    measurement = native_finisher._measure_one_surrogate_step_s(
        W5Seed("probe", pos),
        edge_index,
        node_sizes,
        topo_depth,
        "undirected_2d_sampled",
        None,
        10.0,
    )

    assert measurement.warmup_s == pytest.approx(0.6)
    assert measurement.step_s == pytest.approx((0.2 + 0.3 + 0.4) / 3.0)


def test_measured_cost_plan_does_not_double_charge_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plan that fits the post-probe budget is admitted without charging warmup again."""
    import importlib

    native_finisher = importlib.import_module("dagua.layout.ops.pipelines.native_finisher")
    pos, edge_index, node_sizes = _tiny_layout()
    topo_depth = torch.zeros(pos.shape[0], dtype=torch.long)
    config = LayoutConfig()
    config._dagua_native_total_budget_s = 63.0
    config._dagua_native_w5_referee_cost_s = 0.2
    warmup_s = 3.0
    spent_s = iter([0.0, warmup_s])

    def fake_measure(*args: object) -> object:
        """Return a noisy warmup that must stay out of forward prediction."""
        del args
        return native_finisher.W5StepMeasurement(step_s=0.1, warmup_s=warmup_s)

    def fake_w5_spent_s(config_arg: object, started_perf: Optional[float] = None) -> float:
        """Charge the probe warmup only when budget is recomputed after measurement."""
        del config_arg, started_perf
        return next(spent_s)

    monkeypatch.setattr(native_finisher, "_measure_one_surrogate_step_s", fake_measure)
    monkeypatch.setattr(native_finisher, "_w5_spent_s", fake_w5_spent_s)

    plan = native_finisher._measured_cost_plan(
        seeds=[W5Seed("incumbent", pos)],
        edge_index=edge_index,
        node_sizes=node_sizes,
        topo_depth=topo_depth,
        routed_mode="undirected_2d_sampled",
        slice_s=20.0,
        config=config,
        started_perf=0.0,
        remaining_entry=300.0,
        honest_axes=None,
    )

    assert plan is not None
    assert plan.warmup_s == pytest.approx(warmup_s)
    assert plan.predicted_s == pytest.approx(
        plan.seeds * (plan.steps * plan.measured_step_s + plan.checkpoints * plan.referee_s)
    )
    assert plan.predicted_s <= plan.budget_usable_s
    assert plan.predicted_s + plan.warmup_s > plan.budget_usable_s


def test_measured_cost_plan_uses_largest_fitting_non_tier_step_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pressured small row uses the largest fitting integer step count, not a coarse tier."""
    import importlib

    native_finisher = importlib.import_module("dagua.layout.ops.pipelines.native_finisher")
    pos = torch.stack(
        (
            torch.arange(200, dtype=torch.float32),
            torch.zeros(200, dtype=torch.float32),
        ),
        dim=1,
    )
    edge_index = torch.empty((2, 0), dtype=torch.long)
    node_sizes = torch.full((200, 2), 2.0)
    topo_depth = torch.zeros(200, dtype=torch.long)
    config = LayoutConfig()
    config._dagua_native_total_budget_s = 57.5
    config._dagua_native_w5_referee_cost_s = 0.01
    spent_s = iter([0.0, 0.7])

    def fake_measure(*args: object) -> object:
        """Return costs where 30 two-checkpoint steps fit but 31 steps do not."""
        del args
        return native_finisher.W5StepMeasurement(step_s=0.1, warmup_s=0.7)

    def fake_w5_spent_s(config_arg: object, started_perf: Optional[float] = None) -> float:
        """Make the recomputed usable budget exactly exercise the partial step gap."""
        del config_arg, started_perf
        return next(spent_s)

    monkeypatch.setattr(native_finisher, "_measure_one_surrogate_step_s", fake_measure)
    monkeypatch.setattr(native_finisher, "_w5_spent_s", fake_w5_spent_s)

    plan = native_finisher._measured_cost_plan(
        seeds=[W5Seed("small", pos)],
        edge_index=edge_index,
        node_sizes=node_sizes,
        topo_depth=topo_depth,
        routed_mode="undirected_2d_sampled",
        slice_s=20.0,
        config=config,
        started_perf=0.0,
        remaining_entry=300.0,
        honest_axes=None,
    )

    assert plan is not None
    assert plan.steps == 30
    assert plan.steps not in {24, 18, 12, 6, 3, 2, 1}
    assert plan.checkpoints == 2
    assert plan.predicted_s == pytest.approx(3.02)


def test_w5_finisher_builds_stress_sample_after_admitted_pass_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Measured W5 builds pass-2 stress lazily after the protected pass-1 prefix."""
    import importlib

    native_finisher = importlib.import_module("dagua.layout.ops.pipelines.native_finisher")
    pos, edge_index, node_sizes = _tiny_layout()
    sample = native_finisher.W5StressSample(
        sources=torch.tensor([0], dtype=torch.long),
        targets=torch.tensor([1], dtype=torch.long),
        graph_distances=torch.tensor([1.0], dtype=torch.float32),
    )
    config = LayoutConfig()
    config._dagua_native_w5_measured_sizing = True
    events: list[str] = []

    def fake_plan(*args: object, **kwargs: object) -> object:
        """Record measured planning and admit two seeds with one checkpoint."""
        del args, kwargs
        events.append("plan")
        return native_finisher.W5CostPlan(
            seeds=2,
            steps=5,
            checkpoints=1,
            measured_step_s=0.01,
            warmup_s=0.0,
            referee_s=0.01,
            budget_s=10.0,
            budget_usable_s=8.0,
            predicted_s=0.12,
        )

    def fake_closed_over_all_pairs_dist(score_fn: object) -> object:
        """Record the first admitted pass-2 sample dependency lookup."""
        del score_fn
        events.append("closed")
        return object()

    def fake_build_stress_sample(
        edge_work: torch.Tensor,
        node_count: int,
        all_pairs_dist: object,
        device: torch.device,
    ) -> object:
        """Record the lazy sample build and return a fixed sample object."""
        del edge_work, node_count, all_pairs_dist, device
        events.append("stress build")
        return sample

    def fake_optimize_seed(
        seed: W5Seed,
        edge_work: torch.Tensor,
        size_work: torch.Tensor,
        topo_depth: torch.Tensor,
        mode: str,
        deadline: float,
        honest_axes: Optional[W5HonestAxes] = None,
        *,
        max_steps: Optional[int] = None,
        max_checkpoints: int = 2,
        pass_id: int = 1,
        stress_sample: Optional[object] = None,
    ) -> tuple[torch.Tensor, int, float, list[tuple[int, torch.Tensor, float]]]:
        """Record pass order and assert only pass 2 receives the fixed sample."""
        del edge_work, size_work, topo_depth, mode, deadline, honest_axes, max_checkpoints
        assert max_steps == 5
        events.append(f"{seed.name} pass {pass_id}")
        if pass_id == 1 and "stress build" not in events:
            assert stress_sample is None
        if pass_id == 2:
            assert stress_sample is sample
        return seed.pos, 1, 2.0, [(1, seed.pos, 1.0)]

    monkeypatch.setattr(native_finisher, "_measured_cost_plan", fake_plan)
    monkeypatch.setattr(
        native_finisher,
        "_closed_over_all_pairs_dist",
        fake_closed_over_all_pairs_dist,
    )
    monkeypatch.setattr(native_finisher, "_build_w5_stress_sample", fake_build_stress_sample)
    monkeypatch.setattr(native_finisher, "_optimize_seed", fake_optimize_seed)

    result = run_w5_finisher(
        incumbent_pos=pos,
        incumbent_score_pair=_pair(10.0, 10.0),
        seeds=[W5Seed("seed_a", pos), W5Seed("seed_b", pos + 5.0)],
        edge_index=edge_index,
        node_sizes=node_sizes,
        score_fn=lambda candidate: _pair(9.0, 9.0),
        is_semantically_directed=False,
        declared_hierarchical=False,
        config=config,
    )

    assert result.cost_plan is not None
    assert events[:5] == [
        "plan",
        "seed_a pass 1",
        "closed",
        "stress build",
        "seed_a_p1 pass 2",
    ]
    assert "seed_b pass 1" in events
    assert "seed_b_p1 pass 2" in events
    assert events.count("stress build") == 1


def test_w5_finisher_does_not_build_stress_sample_when_pass_two_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deadline or cap return after pass 1 must not run pass-2-only prep."""
    import importlib

    native_finisher = importlib.import_module("dagua.layout.ops.pipelines.native_finisher")
    pos, edge_index, node_sizes = _tiny_layout()
    config = LayoutConfig()
    config._dagua_native_w5_spent_s = 0.0
    config._dagua_native_w5_measured_sizing = True
    events: list[str] = []

    def fake_plan(*args: object, **kwargs: object) -> object:
        """Admit pass 1 so the later pass-2 budget guard owns the denial."""
        del args, kwargs
        events.append("plan")
        return native_finisher.W5CostPlan(
            seeds=1,
            steps=5,
            checkpoints=1,
            measured_step_s=0.01,
            warmup_s=0.0,
            referee_s=0.01,
            budget_s=10.0,
            budget_usable_s=8.0,
            predicted_s=0.06,
        )

    def fake_optimize_seed(
        seed: W5Seed,
        edge_work: torch.Tensor,
        size_work: torch.Tensor,
        topo_depth: torch.Tensor,
        mode: str,
        deadline: float,
        honest_axes: Optional[W5HonestAxes] = None,
        *,
        max_steps: Optional[int] = None,
        max_checkpoints: int = 2,
        pass_id: int = 1,
        stress_sample: Optional[object] = None,
    ) -> tuple[torch.Tensor, int, float, list[tuple[int, torch.Tensor, float]]]:
        """Record that only pass 1 ran before cap exhaustion."""
        del edge_work, size_work, topo_depth, mode, deadline, honest_axes
        del max_steps, max_checkpoints
        assert pass_id == 1
        assert stress_sample is None
        events.append(f"{seed.name} pass {pass_id}")
        return seed.pos, 1, 2.0, [(1, seed.pos, 1.0)]

    def fake_w5_spent_s(config_arg: object, started_perf: Optional[float] = None) -> float:
        """Allow seed entry, then deny the pass-2 admission guard."""
        del config_arg, started_perf
        return 0.0 if events != ["plan", "seed_a pass 1"] else 999.0

    def forbidden_build_stress_sample(*args: object, **kwargs: object) -> None:
        """Fail if denied pass 2 still prepares the stress sample."""
        del args, kwargs
        raise AssertionError("pass-2-only stress sample built before admission")

    monkeypatch.setattr(native_finisher, "_measured_cost_plan", fake_plan)
    monkeypatch.setattr(native_finisher, "_optimize_seed", fake_optimize_seed)
    monkeypatch.setattr(native_finisher, "_w5_spent_s", fake_w5_spent_s)
    monkeypatch.setattr(native_finisher, "_build_w5_stress_sample", forbidden_build_stress_sample)

    result = run_w5_finisher(
        incumbent_pos=pos,
        incumbent_score_pair=_pair(10.0, 10.0),
        seeds=[W5Seed("seed_a", pos)],
        edge_index=edge_index,
        node_sizes=node_sizes,
        score_fn=lambda candidate: _pair(9.0, 9.0),
        is_semantically_directed=False,
        declared_hierarchical=False,
        config=config,
    )

    assert events == ["plan", "seed_a pass 1"]
    assert result.deadline_returned is True
    assert result.steps == 1


def test_measured_cost_plan_keeps_a3c_boundary_terminal_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rgg_500 boundary fixture selects one seed, five steps, and terminal scoring."""
    import importlib

    native_finisher = importlib.import_module("dagua.layout.ops.pipelines.native_finisher")
    pos = torch.stack(
        (
            torch.arange(500, dtype=torch.float32),
            torch.zeros(500, dtype=torch.float32),
        ),
        dim=1,
    )
    edge_index = torch.empty((2, 0), dtype=torch.long)
    node_sizes = torch.full((500, 2), 2.0)
    topo_depth = torch.zeros(500, dtype=torch.long)
    config = LayoutConfig()
    config._dagua_native_total_budget_s = 50.0
    config._dagua_native_w5_referee_cost_s = 0.432
    spent_s = iter([0.0, 0.700])

    def fake_measure(*args: object) -> object:
        """Use recorded non-tiny costs that make five steps the largest fit."""
        del args
        return native_finisher.W5StepMeasurement(step_s=0.100, warmup_s=0.700)

    def fake_w5_spent_s(config_arg: object, started_perf: Optional[float] = None) -> float:
        """Replay recorded pre/post probe spend for the A3c boundary."""
        del config_arg, started_perf
        return next(spent_s)

    monkeypatch.setattr(native_finisher, "_measure_one_surrogate_step_s", fake_measure)
    monkeypatch.setattr(native_finisher, "_w5_spent_s", fake_w5_spent_s)

    plan = native_finisher._measured_cost_plan(
        seeds=[W5Seed("rgg", pos), W5Seed("rgg_b", pos + 1.0)],
        edge_index=edge_index,
        node_sizes=node_sizes,
        topo_depth=topo_depth,
        routed_mode="undirected_2d_sampled",
        slice_s=2.950,
        config=config,
        started_perf=0.0,
        remaining_entry=300.0,
        honest_axes=None,
    )

    assert plan is not None
    assert (plan.seeds, plan.steps, plan.checkpoints) == (1, 5, 1)
    assert plan.referee_s == pytest.approx(0.432)
    assert plan.budget_usable_s == pytest.approx(0.950)
    assert plan.predicted_s == pytest.approx(0.932)
    assert native_finisher._checkpoint_steps(5, 1) == {5}


def test_measured_cost_plan_tiny_fixture_retains_raised_work_and_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tiny rows keep raised 96-step work and pass 2 receives the lazy sample."""
    import importlib

    native_finisher = importlib.import_module("dagua.layout.ops.pipelines.native_finisher")
    pos, edge_index, node_sizes = _tiny_layout()
    topo_depth = torch.zeros(pos.shape[0], dtype=torch.long)
    config = LayoutConfig()
    config._dagua_native_deadline_s = time.perf_counter() + 300.0
    config._dagua_native_total_budget_s = 300.0
    config._dagua_native_w5_referee_cost_s = 0.001
    config._dagua_native_w5_measured_sizing = True
    sample = native_finisher.W5StressSample(
        sources=torch.tensor([0], dtype=torch.long),
        targets=torch.tensor([1], dtype=torch.long),
        graph_distances=torch.tensor([1.0], dtype=torch.float32),
    )
    pass_two_samples: list[object] = []

    def fake_measure(*args: object) -> object:
        """Return a tiny-row step cost that admits the raised continuation fixture."""
        del args
        return native_finisher.W5StepMeasurement(step_s=0.009, warmup_s=0.0)

    def fake_build_stress_sample(
        edge_work: torch.Tensor,
        node_count: int,
        all_pairs_dist: object,
        device: torch.device,
    ) -> object:
        """Return the fixed sample used to verify pass-2 propagation."""
        del edge_work, node_count, all_pairs_dist, device
        return sample

    def fake_optimize_seed(
        seed: W5Seed,
        edge_work: torch.Tensor,
        size_work: torch.Tensor,
        depth_work: torch.Tensor,
        mode: str,
        deadline: float,
        honest_axes: Optional[W5HonestAxes] = None,
        *,
        max_steps: Optional[int] = None,
        max_checkpoints: int = 2,
        pass_id: int = 1,
        stress_sample: Optional[object] = None,
    ) -> tuple[torch.Tensor, int, float, list[tuple[int, torch.Tensor, float]]]:
        """Verify the raised plan reaches both optimizer passes unchanged."""
        del edge_work, size_work, depth_work, mode, deadline, honest_axes
        assert max_steps == 96
        assert max_checkpoints == 4
        if pass_id == 2:
            pass_two_samples.append(stress_sample)
        return seed.pos, 1, 2.0, [(1, seed.pos, 1.0)]

    monkeypatch.setattr(native_finisher, "_measure_one_surrogate_step_s", fake_measure)
    monkeypatch.setattr(native_finisher, "_closed_over_all_pairs_dist", lambda score_fn: object())
    monkeypatch.setattr(native_finisher, "_build_w5_stress_sample", fake_build_stress_sample)
    monkeypatch.setattr(native_finisher, "_optimize_seed", fake_optimize_seed)

    plan = native_finisher._measured_cost_plan(
        seeds=[
            W5Seed("tiny_a", pos),
            W5Seed("tiny_b", pos + 5.0),
            W5Seed("tiny_c", pos + 10.0),
        ],
        edge_index=edge_index,
        node_sizes=node_sizes,
        topo_depth=topo_depth,
        routed_mode="undirected_2d_sampled",
        slice_s=20.0,
        config=config,
        started_perf=time.perf_counter(),
        remaining_entry=300.0,
        honest_axes=None,
    )
    result = run_w5_finisher(
        incumbent_pos=pos,
        incumbent_score_pair=_pair(10.0, 10.0),
        seeds=[
            W5Seed("tiny_a", pos),
            W5Seed("tiny_b", pos + 5.0),
            W5Seed("tiny_c", pos + 10.0),
        ],
        edge_index=edge_index,
        node_sizes=node_sizes,
        score_fn=lambda candidate: _pair(9.0, 9.0),
        is_semantically_directed=False,
        declared_hierarchical=False,
        config=config,
    )

    assert plan is not None
    assert (plan.steps, plan.checkpoints) == (96, 4)
    assert result.cost_plan is not None
    assert (result.cost_plan.steps, result.cost_plan.checkpoints) == (96, 4)
    assert pass_two_samples
    assert all(seen is sample for seen in pass_two_samples)


def test_w5_finisher_measured_cost_skip_returns_exact_incumbent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Measured cost sizing skips when one seed and checkpoint cannot fit."""
    import importlib

    native_finisher = importlib.import_module("dagua.layout.ops.pipelines.native_finisher")
    pos, edge_index, node_sizes = _tiny_layout()
    config = LayoutConfig()
    config._dagua_native_deadline_s = time.perf_counter() + 300.0
    config._dagua_native_total_budget_s = 30.0
    config._dagua_native_w5_referee_cost_s = 0.4
    config._dagua_native_w5_measured_sizing = True

    def forbidden_score_fn(candidate: torch.Tensor) -> W5ScorePair:
        """Fail if a measured-cost skip accidentally scores a candidate.

        Parameters
        ----------
        candidate : torch.Tensor
            Candidate positions with shape ``[N, 2]``.

        Returns
        -------
        W5ScorePair
            Never returned.
        """
        del candidate
        raise AssertionError("measured-cost skip should not call score_fn")

    def slow_measure(*args: object) -> object:
        """Return a surrogate-step cost that exceeds the tiny W5 budget."""
        del args
        return native_finisher.W5StepMeasurement(step_s=0.7, warmup_s=0.0)

    monkeypatch.setattr(native_finisher, "_measure_one_surrogate_step_s", slow_measure)

    result = run_w5_finisher(
        incumbent_pos=pos,
        incumbent_score_pair=_pair(0.0, 0.0),
        seeds=[W5Seed("incumbent", pos)],
        edge_index=edge_index,
        node_sizes=node_sizes,
        score_fn=forbidden_score_fn,
        is_semantically_directed=False,
        declared_hierarchical=False,
        config=config,
    )

    assert result.skipped_reason == "predicted_cost_measured"
    assert result.deadline_returned is True
    assert torch.equal(result.winner_pos, pos)
    assert result.cost_plan is not None
    assert result.cost_plan.steps == 0
    assert result.cost_plan.predicted_s > result.cost_plan.budget_usable_s


def test_w5_finisher_accumulated_cap_returns_exact_incumbent() -> None:
    """Exhausted accumulated W5 spend returns the incumbent without scoring."""
    config = LayoutConfig()
    config._dagua_native_deadline_s = time.perf_counter() + 120.0
    config._dagua_native_total_budget_s = 300.0
    config._dagua_native_w5_spent_s = 19.5
    pos, edge_index, node_sizes = _tiny_layout()

    def forbidden_score_fn(candidate: torch.Tensor) -> W5ScorePair:
        """Fail if exhausted W5 budget accidentally scores a candidate.

        Parameters
        ----------
        candidate : torch.Tensor
            Candidate positions with shape ``[N, 2]``.

        Returns
        -------
        W5ScorePair
            Never returned.
        """
        raise AssertionError("exhausted W5 budget should not call score_fn")

    result = run_w5_finisher(
        incumbent_pos=pos,
        incumbent_score_pair=_pair(0.0, 0.0),
        seeds=[W5Seed("incumbent", pos)],
        edge_index=edge_index,
        node_sizes=node_sizes,
        score_fn=forbidden_score_fn,
        is_semantically_directed=False,
        declared_hierarchical=False,
        config=config,
    )

    assert result.skipped_reason == "no_budget"
    assert result.deadline_returned is True
    assert torch.equal(result.winner_pos, pos)


def test_w5_finisher_reraises_worker_timeout_like_exception() -> None:
    """Worker-timeout-like exceptions are not swallowed by optional catches."""
    pos, edge_index, node_sizes = _tiny_layout()

    def raising_score_fn(candidate: torch.Tensor) -> W5ScorePair:
        """Raise the local worker-timeout sentinel.

        Parameters
        ----------
        candidate : torch.Tensor
            Candidate positions with shape ``[N, 2]``.

        Returns
        -------
        W5ScorePair
            Never returned.
        """
        raise _WorkerLayoutTimeoutError("worker layout timeout exceeded")

    with pytest.raises(_WorkerLayoutTimeoutError):
        run_w5_finisher(
            incumbent_pos=pos,
            incumbent_score_pair=_pair(0.0, 0.0),
            seeds=[W5Seed("incumbent", pos)],
            edge_index=edge_index,
            node_sizes=node_sizes,
            score_fn=raising_score_fn,
            is_semantically_directed=False,
            declared_hierarchical=False,
        )


def test_w5_finisher_rejects_one_sided_composite_win() -> None:
    """The dominance gate rejects wins under only one frozen-ruler branch."""
    pos, edge_index, node_sizes = _tiny_layout()

    result = run_w5_finisher(
        incumbent_pos=pos,
        incumbent_score_pair=_pair(10.0, 10.0),
        seeds=[W5Seed("incumbent", pos)],
        edge_index=edge_index,
        node_sizes=node_sizes,
        score_fn=lambda candidate: _pair(11.0, 10.0),
        is_semantically_directed=True,
        declared_hierarchical=True,
        direction_is_declared=True,
    )

    assert torch.equal(result.winner_pos, pos)
    assert result.accepted == ()
    assert result.rejected
    assert all(checkpoint.reason == "does_not_dominate_both" for checkpoint in result.rejected)


def test_w5_routes_directed_mode_from_honest_flow_not_surrogate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A high surrogate flow self-report cannot force x_only routing."""
    import importlib

    native_finisher = importlib.import_module("dagua.layout.ops.pipelines.native_finisher")
    pos = torch.tensor(
        [[0.0, 0.0], [0.0, 10.0], [0.0, 20.0], [0.0, 30.0]],
        dtype=torch.float32,
    )
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    node_sizes = torch.full((4, 2), 1.0)
    modes: list[str] = []

    def fake_optimize_seed(
        seed: W5Seed,
        edge_work: torch.Tensor,
        size_work: torch.Tensor,
        topo_depth: torch.Tensor,
        mode: str,
        deadline: float,
        honest_axes: Optional[W5HonestAxes] = None,
        *,
        max_steps: Optional[int] = None,
        max_checkpoints: int = 2,
        pass_id: int = 1,
        stress_sample: Optional[object] = None,
    ) -> tuple[torch.Tensor, int, float, list[tuple[int, torch.Tensor, float]]]:
        """Capture the routed mode and emit one non-dominant checkpoint."""
        del (
            edge_work,
            size_work,
            topo_depth,
            deadline,
            honest_axes,
            max_steps,
            max_checkpoints,
            pass_id,
            stress_sample,
        )
        modes.append(mode)
        return seed.pos, 1, 1.0, [(1, seed.pos, 1.0)]

    monkeypatch.setattr(native_finisher, "_optimize_seed", fake_optimize_seed)

    result = run_w5_finisher(
        incumbent_pos=pos,
        incumbent_score_pair=_pair(10.0, 10.0),
        seeds=[W5Seed("residual_like", pos)],
        edge_index=edge_index,
        node_sizes=node_sizes,
        score_fn=lambda candidate: _pair(9.0, 9.0),
        is_semantically_directed=True,
        declared_hierarchical=True,
        direction_is_declared=True,
        incumbent_axes=W5HonestAxes(flow=0.753, depth=1.0, ksm=0.922, edge_length=0.876),
    )

    assert modes == ["barrier_2d", "barrier_2d"]
    assert result.phase_timings_s[0].mode == "barrier_2d"
    assert [timing.pass_id for timing in result.phase_timings_s] == [1, 2]
    assert result.incumbent_axes is not None
    assert result.incumbent_axes.flow == pytest.approx(0.753)


def test_w5_barrier_flow_gain_improves_honest_flow_and_keeps_ksm_floor() -> None:
    """Barrier mode raises honest flow without giving away the KSM floor."""
    import importlib

    from dagua.metrics import directed_flow_score, full

    native_finisher = importlib.import_module("dagua.layout.ops.pipelines.native_finisher")
    pos = torch.tensor(
        [[0.0, 0.0], [8.0, 0.0], [16.0, 0.0], [24.0, 0.0]],
        dtype=torch.float32,
    )
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    node_sizes = torch.full((4, 2), 1.0)
    topo_depth = torch.arange(4, dtype=torch.long)
    start_metrics = full(pos, edge_index, topo_depth=topo_depth, node_sizes=node_sizes)
    final_pos, steps, _start_loss, _checkpoints = native_finisher._optimize_seed(
        W5Seed("flow_deficient", pos),
        edge_index,
        node_sizes,
        topo_depth,
        "barrier_2d",
        time.monotonic() + 2.0,
        W5HonestAxes(flow=0.5, depth=0.5, ksm=float(start_metrics["ksm_score"])),
    )
    final_metrics = full(final_pos, edge_index, topo_depth=topo_depth, node_sizes=node_sizes)
    start_flow = directed_flow_score(pos, edge_index)["directed_flow_score"]
    final_flow = directed_flow_score(final_pos, edge_index)["directed_flow_score"]

    assert steps > 0
    assert final_flow > start_flow
    assert float(final_metrics["ksm_score"]) >= float(start_metrics["ksm_score"]) - 0.05


def test_w5_mode_ladder_runs_barrier_after_x_only_no_accept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The directed ladder tries barrier_2d after x_only produces no accept."""
    import importlib

    native_finisher = importlib.import_module("dagua.layout.ops.pipelines.native_finisher")
    pos, edge_index, node_sizes = _tiny_layout()
    modes: list[str] = []

    def fake_optimize_seed(
        seed: W5Seed,
        edge_work: torch.Tensor,
        size_work: torch.Tensor,
        topo_depth: torch.Tensor,
        mode: str,
        deadline: float,
        honest_axes: Optional[W5HonestAxes] = None,
        *,
        max_steps: Optional[int] = None,
        max_checkpoints: int = 2,
        pass_id: int = 1,
        stress_sample: Optional[object] = None,
    ) -> tuple[torch.Tensor, int, float, list[tuple[int, torch.Tensor, float]]]:
        """Capture each ladder mode and emit one rejected checkpoint."""
        del (
            edge_work,
            size_work,
            topo_depth,
            deadline,
            honest_axes,
            max_steps,
            max_checkpoints,
            pass_id,
            stress_sample,
        )
        modes.append(mode)
        return seed.pos, 1, 2.0, [(1, seed.pos, 1.5)]

    monkeypatch.setattr(native_finisher, "_optimize_seed", fake_optimize_seed)

    result = run_w5_finisher(
        incumbent_pos=pos,
        incumbent_score_pair=_pair(10.0, 10.0),
        seeds=[W5Seed("hierarchical", pos)],
        edge_index=edge_index,
        node_sizes=node_sizes,
        score_fn=lambda candidate: _pair(9.0, 9.0),
        is_semantically_directed=True,
        declared_hierarchical=True,
        direction_is_declared=True,
        incumbent_axes=W5HonestAxes(flow=0.99, depth=0.99, ksm=0.9, edge_length=0.9),
    )

    assert modes == ["x_only", "x_only", "barrier_2d", "barrier_2d"]
    assert [timing.mode for timing in result.phase_timings_s] == modes
    assert [timing.pass_id for timing in result.phase_timings_s] == [1, 2, 1, 2]
    assert result.accepted == ()


def test_w5_projects_overlapping_checkpoint_before_viability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An overlapping checkpoint reaches the scorer after projection repairs it."""
    import importlib

    native_finisher = importlib.import_module("dagua.layout.ops.pipelines.native_finisher")
    pos, edge_index, node_sizes = _tiny_layout()
    overlapping = torch.tensor(
        [[0.0, 0.0], [0.5, 0.0], [4.0, 0.0], [8.0, 0.0]],
        dtype=torch.float32,
    )
    scored_candidates: list[torch.Tensor] = []

    def fake_optimize_seed(
        seed: W5Seed,
        edge_work: torch.Tensor,
        size_work: torch.Tensor,
        topo_depth: torch.Tensor,
        mode: str,
        deadline: float,
        honest_axes: Optional[W5HonestAxes] = None,
        *,
        max_steps: Optional[int] = None,
        max_checkpoints: int = 2,
        pass_id: int = 1,
        stress_sample: Optional[object] = None,
    ) -> tuple[torch.Tensor, int, float, list[tuple[int, torch.Tensor, float]]]:
        """Return a checkpoint that overlaps until W5 applies projection."""
        del (
            seed,
            edge_work,
            size_work,
            topo_depth,
            mode,
            deadline,
            honest_axes,
            max_steps,
            max_checkpoints,
            pass_id,
            stress_sample,
        )
        return overlapping, 1, 2.0, [(1, overlapping, 1.0)]

    def score_fn(candidate: torch.Tensor) -> W5ScorePair:
        """Capture the post-viability candidate and return a dominant score."""
        scored_candidates.append(candidate.detach().cpu())
        return _pair(1.0, 1.0)

    monkeypatch.setattr(native_finisher, "_optimize_seed", fake_optimize_seed)

    result = run_w5_finisher(
        incumbent_pos=pos,
        incumbent_score_pair=_pair(0.0, 0.0),
        seeds=[W5Seed("overlap", pos)],
        edge_index=edge_index,
        node_sizes=node_sizes,
        score_fn=score_fn,
        is_semantically_directed=False,
        declared_hierarchical=False,
    )

    assert scored_candidates
    assert result.accepted
    assert len(scored_candidates) == 2
    assert result.viability_counts["projected_overlap_candidate"] == 2
    assert result.viability_counts["projection_resolved_overlap"] == 2
    assert result.viability_drop_counts == {}


def test_w5_drops_checkpoint_when_projection_still_overlap_regresses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A checkpoint that remains overlap-regressed after projection is not scored."""
    import importlib

    native_finisher = importlib.import_module("dagua.layout.ops.pipelines.native_finisher")
    pos, edge_index, node_sizes = _tiny_layout()
    overlapping = torch.tensor(
        [[0.0, 0.0], [0.5, 0.0], [4.0, 0.0], [8.0, 0.0]],
        dtype=torch.float32,
    )

    def fake_optimize_seed(
        seed: W5Seed,
        edge_work: torch.Tensor,
        size_work: torch.Tensor,
        topo_depth: torch.Tensor,
        mode: str,
        deadline: float,
        honest_axes: Optional[W5HonestAxes] = None,
        *,
        max_steps: Optional[int] = None,
        max_checkpoints: int = 2,
        pass_id: int = 1,
        stress_sample: Optional[object] = None,
    ) -> tuple[torch.Tensor, int, float, list[tuple[int, torch.Tensor, float]]]:
        """Return a checkpoint that stays overlapped under the patched projector."""
        del (
            seed,
            edge_work,
            size_work,
            topo_depth,
            mode,
            deadline,
            honest_axes,
            max_steps,
            max_checkpoints,
            pass_id,
            stress_sample,
        )
        return overlapping, 1, 2.0, [(1, overlapping, 1.0)]

    def no_op_project(checkpoint_pos: torch.Tensor, size_work: torch.Tensor) -> torch.Tensor:
        """Leave the overlapping checkpoint unchanged."""
        del size_work
        return checkpoint_pos

    def forbidden_score_fn(candidate: torch.Tensor) -> W5ScorePair:
        """Fail if an overlap-regressed checkpoint reaches honest scoring."""
        del candidate
        raise AssertionError("overlap-regressed checkpoint should be dropped")

    monkeypatch.setattr(native_finisher, "_optimize_seed", fake_optimize_seed)
    monkeypatch.setattr(native_finisher, "_project_checkpoint_for_viability", no_op_project)

    result = run_w5_finisher(
        incumbent_pos=pos,
        incumbent_score_pair=_pair(0.0, 0.0),
        seeds=[W5Seed("overlap", pos)],
        edge_index=edge_index,
        node_sizes=node_sizes,
        score_fn=forbidden_score_fn,
        is_semantically_directed=False,
        declared_hierarchical=False,
    )

    assert result.checkpoints == ()
    assert result.skipped_reason == "no_checkpoint"
    assert result.viability_counts["drop_overlap_regressed"] == 2
    assert result.viability_drop_counts == {"overlap_regressed": 2}


def test_w5_late_entry_prediction_uses_process_time_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unloaded process-time and wall-clock admission make the same W5 decision."""
    import importlib

    native_finisher = importlib.import_module("dagua.layout.ops.pipelines.native_finisher")
    config = LayoutConfig()
    config._dagua_native_deadline_s = 200.0
    monkeypatch.setattr(native_finisher.time, "perf_counter", lambda: 80.0)
    monkeypatch.setattr(native_finisher.time, "process_time", lambda: 20.0)

    assert w5_predicted_skip_reason(34, 78, config) is None
    assert getattr(config, "_dagua_native_process_deadline_s") == 140.0


def test_measured_terminal_w5_bypasses_late_entry_prediction() -> None:
    """Measured terminal sizing, not the old late-entry gate, owns terminal skips."""
    config = LayoutConfig()
    config._dagua_native_process_deadline_s = time.process_time() + 10.0
    config._dagua_native_w5_measured_sizing = True

    assert w5_predicted_skip_reason(500, 1470, config) is None


def test_w5_first_score_epilogue_scores_terminal_once_with_wall_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The score-loop spend guard scores exactly one terminal checkpoint epilogue."""
    import importlib

    native_finisher = importlib.import_module("dagua.layout.ops.pipelines.native_finisher")
    pos, edge_index, node_sizes = _tiny_layout()
    early_pos = pos + 1.0
    terminal_pos = pos + 2.0
    score_calls: list[torch.Tensor] = []
    spent_calls = {"count": 0}

    def fake_optimize_seed(
        seed: W5Seed,
        edge_work: torch.Tensor,
        size_work: torch.Tensor,
        topo_depth: torch.Tensor,
        mode: str,
        deadline: float,
        honest_axes: Optional[W5HonestAxes] = None,
        *,
        max_steps: Optional[int] = None,
        max_checkpoints: int = 2,
        pass_id: int = 1,
        stress_sample: Optional[object] = None,
    ) -> tuple[torch.Tensor, int, float, list[tuple[int, torch.Tensor, float]]]:
        """Return an early checkpoint plus a better terminal checkpoint."""
        del seed, edge_work, size_work, topo_depth, mode, deadline, honest_axes
        del max_steps, max_checkpoints, pass_id, stress_sample
        return terminal_pos, 7, 10.0, [(3, early_pos, 4.0), (7, terminal_pos, 1.0)]

    def fake_w5_spent_s(config: object, started_perf: object = None) -> float:
        """Trip the spend guard only when checkpoint scoring begins."""
        del config, started_perf
        spent_calls["count"] += 1
        return 0.0 if spent_calls["count"] == 1 else 999.0

    def score_fn(candidate: torch.Tensor) -> W5ScorePair:
        """Record honest score calls and return a dominating score."""
        score_calls.append(candidate.detach().cpu())
        return _pair(2.0, 2.0)

    monkeypatch.setattr(native_finisher, "_optimize_seed", fake_optimize_seed)
    monkeypatch.setattr(native_finisher, "_w5_spent_s", fake_w5_spent_s)

    result = run_w5_finisher(
        incumbent_pos=pos,
        incumbent_score_pair=_pair(0.0, 0.0),
        seeds=[W5Seed("incumbent", pos)],
        edge_index=edge_index,
        node_sizes=node_sizes,
        score_fn=score_fn,
        is_semantically_directed=False,
        declared_hierarchical=False,
    )

    assert len(score_calls) == 1
    assert torch.equal(score_calls[0], terminal_pos)
    assert len(result.checkpoints) == 1
    assert result.checkpoints[0].step == 7
    assert result.checkpoints[0].accepted is True
    assert result.deadline_returned is True


def test_w5_first_score_epilogue_requires_wall_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first-score epilogue does not fire without benchmark wall headroom."""
    import importlib

    native_finisher = importlib.import_module("dagua.layout.ops.pipelines.native_finisher")
    pos, edge_index, node_sizes = _tiny_layout()
    terminal_pos = pos + 2.0
    spent_calls = {"count": 0}

    def fake_optimize_seed(
        seed: W5Seed,
        edge_work: torch.Tensor,
        size_work: torch.Tensor,
        topo_depth: torch.Tensor,
        mode: str,
        deadline: float,
        honest_axes: Optional[W5HonestAxes] = None,
        *,
        max_steps: Optional[int] = None,
        max_checkpoints: int = 2,
        pass_id: int = 1,
        stress_sample: Optional[object] = None,
    ) -> tuple[torch.Tensor, int, float, list[tuple[int, torch.Tensor, float]]]:
        """Return a terminal checkpoint that must not be scored."""
        del seed, edge_work, size_work, topo_depth, mode, deadline, honest_axes
        del max_steps, max_checkpoints, pass_id, stress_sample
        return terminal_pos, 7, 10.0, [(7, terminal_pos, 1.0)]

    def fake_w5_spent_s(config: object, started_perf: object = None) -> float:
        """Trip the spend guard only when checkpoint scoring begins."""
        del config, started_perf
        spent_calls["count"] += 1
        return 0.0 if spent_calls["count"] == 1 else 999.0

    def forbidden_score_fn(candidate: torch.Tensor) -> W5ScorePair:
        """Fail if the no-headroom terminal checkpoint reaches scoring."""
        del candidate
        raise AssertionError("first-score epilogue should not run without wall headroom")

    monkeypatch.setattr(native_finisher, "_optimize_seed", fake_optimize_seed)
    monkeypatch.setattr(native_finisher, "_w5_spent_s", fake_w5_spent_s)
    monkeypatch.setattr(
        native_finisher,
        "_w5_first_score_epilogue_has_wall_headroom",
        lambda config: False,
    )

    result = run_w5_finisher(
        incumbent_pos=pos,
        incumbent_score_pair=_pair(0.0, 0.0),
        seeds=[W5Seed("incumbent", pos)],
        edge_index=edge_index,
        node_sizes=node_sizes,
        score_fn=forbidden_score_fn,
        is_semantically_directed=False,
        declared_hierarchical=False,
    )

    assert result.checkpoints == ()
    assert result.skipped_reason == "no_checkpoint"
    assert result.deadline_returned is True
    assert torch.equal(result.winner_pos, pos)


def test_w5_first_score_epilogue_never_runs_after_checkpoint_scored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A normal first checkpoint makes the epilogue a no-op for green rows."""
    import importlib

    native_finisher = importlib.import_module("dagua.layout.ops.pipelines.native_finisher")
    pos, edge_index, node_sizes = _tiny_layout()
    early_pos = pos + 1.0
    terminal_pos = pos + 2.0
    score_calls: list[torch.Tensor] = []
    spent_calls = {"count": 0}

    def fake_optimize_seed(
        seed: W5Seed,
        edge_work: torch.Tensor,
        size_work: torch.Tensor,
        topo_depth: torch.Tensor,
        mode: str,
        deadline: float,
        honest_axes: Optional[W5HonestAxes] = None,
        *,
        max_steps: Optional[int] = None,
        max_checkpoints: int = 2,
        pass_id: int = 1,
        stress_sample: Optional[object] = None,
    ) -> tuple[torch.Tensor, int, float, list[tuple[int, torch.Tensor, float]]]:
        """Return two checkpoints so the second guard could otherwise epilogue."""
        del seed, edge_work, size_work, topo_depth, mode, deadline, honest_axes
        del max_steps, max_checkpoints, pass_id, stress_sample
        return terminal_pos, 7, 10.0, [(3, early_pos, 4.0), (7, terminal_pos, 1.0)]

    def fake_w5_spent_s(config: object, started_perf: object = None) -> float:
        """Allow one normal checkpoint score, then trip the spend guard."""
        del config, started_perf
        spent_calls["count"] += 1
        return 0.0 if spent_calls["count"] <= 2 else 999.0

    def score_fn(candidate: torch.Tensor) -> W5ScorePair:
        """Record honest score calls and return a non-dominating score."""
        score_calls.append(candidate.detach().cpu())
        return _pair(0.0, 0.0)

    monkeypatch.setattr(native_finisher, "_optimize_seed", fake_optimize_seed)
    monkeypatch.setattr(native_finisher, "_w5_spent_s", fake_w5_spent_s)

    result = run_w5_finisher(
        incumbent_pos=pos,
        incumbent_score_pair=_pair(10.0, 10.0),
        seeds=[W5Seed("incumbent", pos)],
        edge_index=edge_index,
        node_sizes=node_sizes,
        score_fn=score_fn,
        is_semantically_directed=False,
        declared_hierarchical=False,
    )

    assert len(score_calls) == 1
    assert torch.equal(score_calls[0], early_pos)
    assert len(result.checkpoints) == 1
    assert result.checkpoints[0].step == 3
    assert result.deadline_returned is True


def test_w5_finisher_budget_exhaustion_after_worse_checkpoint_returns_incumbent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Budget exhaustion after optimizer steps still returns the exact incumbent."""
    import importlib

    native_finisher = importlib.import_module("dagua.layout.ops.pipelines.native_finisher")

    pos, edge_index, node_sizes = _tiny_layout()
    worse_pos = pos + 1000.0
    spent_calls = {"count": 0}

    def fake_optimize_seed(
        seed: W5Seed,
        edge_work: torch.Tensor,
        size_work: torch.Tensor,
        topo_depth: torch.Tensor,
        mode: str,
        deadline: float,
        honest_axes: Optional[W5HonestAxes] = None,
        *,
        max_steps: Optional[int] = None,
        max_checkpoints: int = 2,
        pass_id: int = 1,
        stress_sample: Optional[object] = None,
    ) -> tuple[torch.Tensor, int, float, list[tuple[int, torch.Tensor, float]]]:
        """Return a worse checkpoint after one completed optimizer step."""
        del (
            seed,
            edge_work,
            size_work,
            topo_depth,
            mode,
            deadline,
            honest_axes,
            max_steps,
            max_checkpoints,
            pass_id,
            stress_sample,
        )
        return worse_pos, 1, 1.0, [(1, worse_pos, 2.0)]

    def fake_w5_spent_s(config: object, started_perf: object = None) -> float:
        """Exhaust the W5 cap only after the seed optimizer has run."""
        del config, started_perf
        spent_calls["count"] += 1
        return 0.0 if spent_calls["count"] == 1 else 999.0

    def forbidden_score_fn(candidate: torch.Tensor) -> W5ScorePair:
        """Fail if a budget-exhausted checkpoint is scored or accepted."""
        del candidate
        raise AssertionError("budget-exhausted checkpoint should not be scored")

    monkeypatch.setattr(native_finisher, "_optimize_seed", fake_optimize_seed)
    monkeypatch.setattr(native_finisher, "_w5_spent_s", fake_w5_spent_s)
    monkeypatch.setattr(
        native_finisher,
        "_w5_first_score_epilogue_has_wall_headroom",
        lambda config: False,
    )

    result = run_w5_finisher(
        incumbent_pos=pos,
        incumbent_score_pair=_pair(10.0, 10.0),
        seeds=[W5Seed("incumbent", pos)],
        edge_index=edge_index,
        node_sizes=node_sizes,
        score_fn=forbidden_score_fn,
        is_semantically_directed=False,
        declared_hierarchical=False,
    )

    assert result.steps == 1
    assert result.deadline_returned is True
    assert torch.equal(result.winner_pos, pos)
    assert result.winner_score_pair == _pair(10.0, 10.0)
    assert result.accepted == ()


def test_w5_finisher_finish_clamps_non_dominant_winner_to_incumbent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The finish clamp preserves the incumbent when a winner stops dominating."""
    import importlib

    native_finisher = importlib.import_module("dagua.layout.ops.pipelines.native_finisher")
    telemetry_path = tmp_path / "w5-clamp.jsonl"
    monkeypatch.setenv("DAGUA_W5_TELEMETRY_PATH", str(telemetry_path))
    pos, edge_index, node_sizes = _tiny_layout()
    candidate_pos = pos + 5.0
    dominance_calls = {"count": 0}

    def fake_optimize_seed(
        seed: W5Seed,
        edge_work: torch.Tensor,
        size_work: torch.Tensor,
        topo_depth: torch.Tensor,
        mode: str,
        deadline: float,
        honest_axes: Optional[W5HonestAxes] = None,
        *,
        max_steps: Optional[int] = None,
        max_checkpoints: int = 2,
        pass_id: int = 1,
        stress_sample: Optional[object] = None,
    ) -> tuple[torch.Tensor, int, float, list[tuple[int, torch.Tensor, float]]]:
        """Return one scoreable checkpoint that is initially accepted."""
        del (
            seed,
            edge_work,
            size_work,
            topo_depth,
            mode,
            deadline,
            honest_axes,
            max_steps,
            max_checkpoints,
            pass_id,
            stress_sample,
        )
        return candidate_pos, 1, 2.0, [(1, candidate_pos, 1.0)]

    def fake_w5_dominates(
        candidate: W5ScorePair,
        incumbent: W5ScorePair,
        margin: float = 0.05,
    ) -> bool:
        """Accept inside the loop, then force the final clamp branch."""
        del candidate, incumbent, margin
        dominance_calls["count"] += 1
        return dominance_calls["count"] == 1

    monkeypatch.setattr(native_finisher, "_optimize_seed", fake_optimize_seed)
    monkeypatch.setattr(native_finisher, "w5_dominates", fake_w5_dominates)

    result = run_w5_finisher(
        incumbent_pos=pos,
        incumbent_score_pair=_pair(10.0, 10.0),
        seeds=[W5Seed("incumbent", pos)],
        edge_index=edge_index,
        node_sizes=node_sizes,
        score_fn=lambda candidate: _pair(10.2, 10.2),
        is_semantically_directed=False,
        declared_hierarchical=False,
    )
    log_w5_telemetry(result, None)

    capsys.readouterr()
    records = [json.loads(line) for line in telemetry_path.read_text().splitlines()]

    assert dominance_calls["count"] == 3
    assert torch.equal(result.winner_pos, pos)
    assert result.winner_score_pair == _pair(10.0, 10.0)
    assert result.winner_name == "incumbent"
    assert result.skipped_reason == "clamped_to_incumbent"
    assert records[-1]["skipped_reason"] == "clamped_to_incumbent"


def test_w5_telemetry_emits_skip_reject_accept_and_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Telemetry is visible for skip, reject, accept, and deadline returns."""
    telemetry_path = tmp_path / "w5.jsonl"
    monkeypatch.setenv("DAGUA_W5_TELEMETRY_PATH", str(telemetry_path))
    pos, edge_index, node_sizes = _tiny_layout()

    skip_result = run_w5_finisher(
        incumbent_pos=pos,
        incumbent_score_pair=_pair(0.0, 0.0),
        seeds=[W5Seed("bad", torch.full_like(pos, float("nan")))],
        edge_index=edge_index,
        node_sizes=node_sizes,
        score_fn=lambda candidate: _pair(1.0, 1.0),
        is_semantically_directed=False,
        declared_hierarchical=False,
    )
    reject_result = run_w5_finisher(
        incumbent_pos=pos,
        incumbent_score_pair=_pair(10.0, 10.0),
        seeds=[W5Seed("incumbent", pos)],
        edge_index=edge_index,
        node_sizes=node_sizes,
        score_fn=lambda candidate: _pair(11.0, 10.0),
        is_semantically_directed=False,
        declared_hierarchical=False,
    )
    accept_result = run_w5_finisher(
        incumbent_pos=pos,
        incumbent_score_pair=_pair(0.0, 0.0),
        seeds=[W5Seed("incumbent", pos)],
        edge_index=edge_index,
        node_sizes=node_sizes,
        score_fn=lambda candidate: _pair(1.0, 1.0),
        is_semantically_directed=False,
        declared_hierarchical=False,
    )
    config = LayoutConfig()
    config._dagua_native_deadline_s = time.perf_counter() + 1.0
    config._dagua_native_graph_name = "unit_graph"
    deadline_result = run_w5_finisher(
        incumbent_pos=pos,
        incumbent_score_pair=_pair(0.0, 0.0),
        seeds=[W5Seed("incumbent", pos)],
        edge_index=edge_index,
        node_sizes=node_sizes,
        score_fn=lambda candidate: _pair(1.0, 1.0),
        is_semantically_directed=False,
        declared_hierarchical=False,
        config=config,
    )

    for result in (skip_result, reject_result, accept_result, deadline_result):
        log_w5_telemetry(result, None)

    stdout = capsys.readouterr().out
    records = [json.loads(line) for line in telemetry_path.read_text().splitlines()]

    assert stdout.count("native_w5_finisher ") == 4
    assert len(records) == 4
    assert any(record["skipped_reason"] == "no_finite_seed" for record in records)
    assert any(record["rejected"] for record in records)
    assert any(record["accepted"] for record in records)
    assert any(record["deadline_returned"] for record in records)
    assert records[-1]["graph_name"] == "unit_graph"
    assert all("phase_timings_s" in record for record in records)
    assert all("viability_drop_counts" in record for record in records)
