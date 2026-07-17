"""Tests for the W5 native finisher wrapper."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import torch

from dagua.config import LayoutConfig
from dagua.layout.ops.pipelines.native_finisher import (
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


def test_w5_finisher_predicted_large_graph_skip_returns_exact_incumbent() -> None:
    """Predicted-cost skip preserves the incumbent without scoring candidates."""
    pos = torch.stack(
        (
            torch.arange(250, dtype=torch.float32),
            torch.zeros(250, dtype=torch.float32),
        ),
        dim=1,
    )
    edge_index = torch.empty((2, 0), dtype=torch.long)
    node_sizes = torch.full((250, 2), 2.0)

    def forbidden_score_fn(candidate: torch.Tensor) -> W5ScorePair:
        """Fail if predicted-cost skip accidentally scores a candidate.

        Parameters
        ----------
        candidate : torch.Tensor
            Candidate positions with shape ``[N, 2]``.

        Returns
        -------
        W5ScorePair
            Never returned.
        """
        raise AssertionError("predicted-cost skip should not call score_fn")

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

    assert result.skipped_reason == "predicted_cost_large_graph"
    assert result.deadline_returned is False
    assert torch.equal(result.winner_pos, pos)


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
    ) -> tuple[torch.Tensor, int, float, list[tuple[int, torch.Tensor, float]]]:
        """Return a checkpoint that overlaps until W5 applies projection."""
        del seed, edge_work, size_work, topo_depth, mode, deadline
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
    assert result.viability_counts["projected_overlap_candidate"] == 1
    assert result.viability_counts["projection_resolved_overlap"] == 1
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
    ) -> tuple[torch.Tensor, int, float, list[tuple[int, torch.Tensor, float]]]:
        """Return a checkpoint that stays overlapped under the patched projector."""
        del seed, edge_work, size_work, topo_depth, mode, deadline
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
    assert result.viability_counts["drop_overlap_regressed"] == 1
    assert result.viability_drop_counts == {"overlap_regressed": 1}


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
    ) -> tuple[torch.Tensor, int, float, list[tuple[int, torch.Tensor, float]]]:
        """Return a worse checkpoint after one completed optimizer step."""
        del seed, edge_work, size_work, topo_depth, mode, deadline
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
    ) -> tuple[torch.Tensor, int, float, list[tuple[int, torch.Tensor, float]]]:
        """Return one scoreable checkpoint that is initially accepted."""
        del seed, edge_work, size_work, topo_depth, mode, deadline
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

    assert dominance_calls["count"] == 2
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
