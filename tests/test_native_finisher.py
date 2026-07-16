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
