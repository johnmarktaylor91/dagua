"""Unit tests for the unified benchmark script helpers."""

from __future__ import annotations

import sys
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
from pytest import MonkeyPatch

from scripts.run_benchmark import (
    POSITION_DIRNAME,
    RECOVERED_GIT_SHA_PREFIX,
    BenchmarkRecord,
    effective_timeout,
    expired_watchdog_futures,
    final_exit_code,
    is_record_complete,
    merge_recovered_results,
    non_resume_out_of_scope_keys,
    parse_args,
    position_relative_path,
    recover_results_from_positions,
    refresh_watchdog_start_times,
    seeds_for_engine,
    serial_watchdog_warning,
)


def _record(
    *,
    status: str = "ok",
    runtime_seconds: Optional[float] = 0.1,
    error: Optional[str] = None,
    positions_file: Optional[str] = "positions/chain_5__dagua.pt",
    git_sha: Optional[str] = "executed-sha",
) -> BenchmarkRecord:
    """Build a minimal benchmark record for merge/resume tests.

    Parameters
    ----------
    status : str
        Record status.
    runtime_seconds : float | None
        Measured runtime.
    error : str | None
        Error message.
    positions_file : str | None
        Relative tensor path.
    git_sha : str | None
        Provenance sha.

    Returns
    -------
    BenchmarkRecord
        Synthetic record keyed as ``chain_5::dagua::deterministic``.
    """
    return BenchmarkRecord(
        graph_name="chain_5",
        engine_name="dagua",
        seed=None,
        status=status,
        runtime_seconds=runtime_seconds,
        error=error,
        positions_file=positions_file,
        num_nodes=5,
        num_edges=4,
        is_stochastic=False,
        skip_reason=None,
        original_for=[],
        reimpl_of=[],
        git_sha=git_sha,
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


def test_parse_args_accepts_deterministic_native(monkeypatch: MonkeyPatch) -> None:
    """The benchmark CLI should expose deterministic native measurement mode."""
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "--deterministic-native"])

    args = parse_args()

    assert args.deterministic_native is True


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


def test_recovered_rows_carry_recovered_provenance_marker(tmp_path: Path) -> None:
    """Recovery must never stamp the bare current sha on adopted tensors."""
    test_graph = SimpleNamespace(
        name="chain_3",
        tags={"unit"},
        graph=SimpleNamespace(
            edge_index=torch.tensor([[0, 1], [1, 2]]),
            num_nodes=3,
        ),
    )
    competitor = SimpleNamespace(name="dagua")
    positions_dir = tmp_path / POSITION_DIRNAME
    positions_dir.mkdir()
    torch.save(torch.zeros(3, 2), positions_dir / "chain_3__dagua.pt")

    recovered = recover_results_from_positions(
        output_dir=tmp_path,
        graphs=[test_graph],
        engines=[competitor],
        seed_count=1,
        seed_start=42,
        seed_refs=set(),
        git_sha="abc123",
    )

    assert sorted(recovered) == ["chain_3::dagua::deterministic"]
    record = recovered["chain_3::dagua::deterministic"]
    assert record.status == "ok"
    assert record.git_sha == f"{RECOVERED_GIT_SHA_PREFIX}abc123"
    assert record.git_sha != "abc123"


def test_merge_recovered_results_never_overrides_executed_rows() -> None:
    """Recovered rows only fill missing or crash-interrupted keys."""
    key = "chain_5::dagua::deterministic"
    recovered_ok = _record(runtime_seconds=None, git_sha="recovered:abc123")

    # Executed ok row keeps its provenance and runtime.
    executed_ok = _record(status="ok", runtime_seconds=1.5, git_sha="executed-sha")
    merged = merge_recovered_results({key: executed_ok}, {key: recovered_ok})
    assert merged[key] is executed_ok

    # Error rows must not self-heal into successes via orphan tensors.
    errored = _record(
        status="error",
        runtime_seconds=None,
        error="watchdog: future exceeded timeout",
        positions_file=None,
        git_sha="executed-sha",
    )
    merged = merge_recovered_results({key: errored}, {key: recovered_ok})
    assert merged[key] is errored

    # Crash-mid-flight (running) rows ARE recovered.
    running = _record(status="running", runtime_seconds=None, positions_file=None)
    merged = merge_recovered_results({key: running}, {key: recovered_ok})
    assert merged[key] is recovered_ok

    # Missing keys are adopted.
    merged = merge_recovered_results({}, {key: recovered_ok})
    assert merged[key] is recovered_ok


def test_non_resume_out_of_scope_keys_flags_only_foreign_rows() -> None:
    """The scope guard lists exactly the rows a non-resume run would drop."""
    disk = {
        "chain_5::dagua::deterministic": _record(),
        "chain_5::classic_fr::seed42": _record(),
    }

    assert non_resume_out_of_scope_keys(disk, ["chain_5::dagua::deterministic"]) == [
        "chain_5::classic_fr::seed42"
    ]
    assert (
        non_resume_out_of_scope_keys(
            disk,
            ["chain_5::dagua::deterministic", "chain_5::classic_fr::seed42"],
        )
        == []
    )
    assert non_resume_out_of_scope_keys({}, ["chain_5::dagua::deterministic"]) == []


def test_parse_args_accepts_scope_and_retry_flags(monkeypatch: MonkeyPatch) -> None:
    """The CLI exposes the scope-rewrite override and watchdog retry flags."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_benchmark.py", "--force-scope-rewrite", "--retry-watchdog-errors"],
    )

    args = parse_args()

    assert args.force_scope_rewrite is True
    assert args.retry_watchdog_errors is True


def test_serial_watchdog_warning_fires_only_for_inert_flag() -> None:
    """An explicit --watchdog-timeout under --workers 1 warns loudly."""
    assert serial_watchdog_warning(1, 3600.0) is not None
    assert "NO EFFECT" in str(serial_watchdog_warning(1, 3600.0))
    assert serial_watchdog_warning(4, 3600.0) is None
    assert serial_watchdog_warning(1, None) is None


def test_final_exit_code_fails_on_error_or_timeout_rows() -> None:
    """A regen with failed rows must not exit 0."""
    assert final_exit_code({"ok": 121}, for_counts_ok=True) == 0
    assert final_exit_code({"ok": 120, "error": 1}, for_counts_ok=True) == 1
    assert final_exit_code({"ok": 120, "timeout": 1}, for_counts_ok=True) == 1
    assert final_exit_code({"ok": 121}, for_counts_ok=False) == 1
    assert final_exit_code({"ok": 100, "skipped": 21}, for_counts_ok=True) == 0


def test_is_record_complete_retries_watchdog_errors_only_when_asked(
    tmp_path: Path,
) -> None:
    """Watchdog-error rows stay permanent unless --retry-watchdog-errors."""
    watchdog_error = _record(
        status="error",
        runtime_seconds=None,
        error="watchdog: future exceeded timeout",
        positions_file=None,
    )
    other_error = _record(
        status="error",
        runtime_seconds=None,
        error="ValueError: boom",
        positions_file=None,
    )

    # Default: error rows are complete (never retried) -- historical behavior.
    assert is_record_complete(watchdog_error, output_dir=tmp_path, save_positions=True)
    assert is_record_complete(other_error, output_dir=tmp_path, save_positions=True)

    # Flagged: only watchdog errors become retryable.
    assert not is_record_complete(
        watchdog_error,
        output_dir=tmp_path,
        save_positions=True,
        retry_watchdog_errors=True,
    )
    assert is_record_complete(
        other_error,
        output_dir=tmp_path,
        save_positions=True,
        retry_watchdog_errors=True,
    )


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
