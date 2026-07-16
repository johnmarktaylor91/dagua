"""Unit tests for the unified benchmark script helpers."""

from __future__ import annotations

import sys
from concurrent.futures import Future
from pathlib import Path

from pytest import MonkeyPatch

from scripts.run_benchmark import (
    BenchmarkRecord,
    effective_timeout,
    expired_watchdog_futures,
    is_record_complete,
    parse_args,
    position_relative_path,
    refresh_watchdog_start_times,
    seeds_for_engine,
)


def test_seeds_for_engine_respects_stochastic_registry() -> None:
    """Stochastic engines should expand into a reproducible seed range."""
    assert seeds_for_engine("classic_fr", seed_count=3, seed_start=42) == [42, 43, 44]
    assert seeds_for_engine("classic_fr", seed_count=3, seed_start=50) == [50, 51, 52]
    assert seeds_for_engine("dagua", seed_count=3, seed_start=42) == [None]


def test_effective_timeout_keeps_full_budget_for_dagua() -> None:
    """Dagua keeps the requested timeout while other engines stay size-scaled."""
    assert effective_timeout(300.0, 120, "dagua") == 300.0
    assert effective_timeout(300.0, 120, "graphviz_sfdp") == 72.0


def test_parse_args_accepts_seed_start(monkeypatch: MonkeyPatch) -> None:
    """The benchmark CLI should expose a configurable stochastic seed start."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_benchmark.py", "--seeds", "3", "--seed-start", "50"],
    )

    args = parse_args()

    assert args.seeds == 3
    assert args.seed_start == 50


def test_position_relative_path_sanitizes_and_formats_seed_suffixes() -> None:
    """Saved tensor paths should match the documented naming scheme."""
    deterministic_path = position_relative_path("grid 5/5", "dagua", None)
    stochastic_path = position_relative_path("grid 5/5", "classic_fr", 44)

    assert deterministic_path == Path("positions/grid_5_5__dagua.pt")
    assert stochastic_path == Path("positions/grid_5_5__classic_fr__seed44.pt")


def test_is_record_complete_requires_positions_when_enabled(tmp_path: Path) -> None:
    """Resume should rerun successful records when their tensor file is missing."""
    record = BenchmarkRecord(
        graph_name="chain_5",
        engine_name="dagua",
        seed=None,
        status="ok",
        runtime_seconds=0.1,
        error=None,
        positions_file="positions/chain_5__dagua.pt",
        num_nodes=5,
        num_edges=4,
        is_stochastic=False,
        skip_reason=None,
        original_for=[],
        reimpl_of=[],
        git_sha="test-sha",
    )

    assert not is_record_complete(record, output_dir=tmp_path, save_positions=True)

    positions_path = tmp_path / "positions" / "chain_5__dagua.pt"
    positions_path.parent.mkdir(parents=True, exist_ok=True)
    positions_path.write_bytes(b"tensor")

    assert is_record_complete(record, output_dir=tmp_path, save_positions=True)
    assert is_record_complete(record, output_dir=tmp_path, save_positions=False)


def test_is_record_complete_treats_running_records_as_incomplete(tmp_path: Path) -> None:
    """In-flight records should never be skipped by resume."""
    record = BenchmarkRecord(
        graph_name="chain_5",
        engine_name="dagua",
        seed=None,
        status="running",
        runtime_seconds=None,
        error=None,
        positions_file=None,
        num_nodes=5,
        num_edges=4,
        is_stochastic=False,
        skip_reason=None,
        original_for=[],
        reimpl_of=[],
        git_sha="test-sha",
    )

    assert not is_record_complete(record, output_dir=tmp_path, save_positions=True)


def test_watchdog_timers_scope_to_active_worker_slots() -> None:
    """Watchdog expiry should not poison queued rolling-window peers."""
    active_future: Future[list[dict[str, object]]] = Future()
    queued_future: Future[list[dict[str, object]]] = Future()
    inflight = {
        active_future: (),
        queued_future: (),
    }
    started_at: dict[Future[list[dict[str, object]]], float] = {}

    refresh_watchdog_start_times(
        inflight,
        started_at,
        max_active=1,
        now=10.0,
    )

    assert started_at == {active_future: 10.0}
    assert expired_watchdog_futures(
        inflight,
        started_at,
        max_active=1,
        watchdog_timeout=5.0,
        now=16.0,
    ) == [active_future]
    assert queued_future not in started_at
