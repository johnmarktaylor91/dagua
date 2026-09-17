"""Unit tests for the benchmark store integrity validator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.validate_benchmark_integrity import (
    validate_missing_store,
    validate_pt_sync,
)


def _row(engine: str, positions_file: str | None, status: str = "ok") -> dict[str, object]:
    """Build a minimal results.json row payload.

    Parameters
    ----------
    engine : str
        Engine name.
    positions_file : str | None
        Relative tensor path, or ``None`` for no-positions rows.
    status : str
        Row status.

    Returns
    -------
    dict[str, object]
        Row payload.
    """
    return {
        "graph_name": "g1",
        "engine_name": engine,
        "seed": None,
        "status": status,
        "positions_file": positions_file,
    }


def _write_store(
    data_dir: Path,
    rows: dict[str, dict[str, object]],
    tensor_names: list[str],
) -> None:
    """Materialize a synthetic .pt-format benchmark store.

    Parameters
    ----------
    data_dir : Path
        Output directory.
    rows : dict[str, dict[str, object]]
        results.json payload.
    tensor_names : list[str]
        Filenames to create under ``positions/`` (existence-only checks).
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "results.json").write_text(json.dumps(rows), encoding="utf-8")
    positions_dir = data_dir / "positions"
    positions_dir.mkdir(exist_ok=True)
    for name in tensor_names:
        (positions_dir / name).write_bytes(b"pt")


def test_validate_pt_sync_flags_missing_tensors_and_orphans(tmp_path: Path) -> None:
    """The .pt store gets the same DESYNC/ORPHAN coverage as positions.h5."""
    rows = {
        "g1::eng_a::deterministic": _row("eng_a", "positions/g1__eng_a.pt"),
        "g1::eng_b::deterministic": _row("eng_b", "positions/g1__eng_b.pt"),
    }
    _write_store(tmp_path, rows, ["g1__eng_a.pt", "zzz__eng_c.pt"])

    errors = validate_pt_sync(tmp_path)

    assert any("DESYNC: eng_b has 1 ok results" in e for e in errors)
    assert any("ORPHAN" in e and "zzz__eng_c.pt" in e for e in errors)
    assert len(errors) == 2


def test_validate_pt_sync_respects_engine_scope(tmp_path: Path) -> None:
    """Engine-scoped validation ignores out-of-scope desyncs."""
    rows = {
        "g1::eng_a::deterministic": _row("eng_a", "positions/g1__eng_a.pt"),
        "g1::eng_b::deterministic": _row("eng_b", "positions/g1__eng_b.pt"),
    }
    _write_store(tmp_path, rows, ["g1__eng_a.pt"])

    assert validate_pt_sync(tmp_path, engines={"eng_a"}) == []
    assert any("eng_b" in e for e in validate_pt_sync(tmp_path, engines={"eng_b"}))


def test_validate_pt_sync_flags_ok_rows_without_positions_reference(tmp_path: Path) -> None:
    """An ok row lacking positions_file in a .pt dir counts as a desync."""
    rows = {"g1::eng_a::deterministic": _row("eng_a", None)}
    _write_store(tmp_path, rows, [])

    errors = validate_pt_sync(tmp_path)

    assert any("DESYNC: eng_a has 1 ok results" in e for e in errors)


def test_validate_missing_store_flags_dangling_references(tmp_path: Path) -> None:
    """Ok rows referencing tensors fail when no store of either format exists."""
    results_path = tmp_path / "results.json"
    results_path.write_text(
        json.dumps({"g1::eng_a::deterministic": _row("eng_a", "positions/g1__eng_a.pt")}),
        encoding="utf-8",
    )

    errors = validate_missing_store(results_path)

    assert len(errors) == 1
    assert "MISSING STORE" in errors[0]


def test_validate_missing_store_allows_no_positions_runs(tmp_path: Path) -> None:
    """--no-positions rows (positions_file absent) are legitimately tensor-less."""
    results_path = tmp_path / "results.json"
    results_path.write_text(
        json.dumps({"g1::eng_a::deterministic": _row("eng_a", None)}),
        encoding="utf-8",
    )

    assert validate_missing_store(results_path) == []


def test_cli_main_fails_on_pt_store_desync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI must actually inspect the .pt store, not exit OK past it."""
    from scripts import validate_benchmark_integrity as vbi

    rows = {"g1::eng_a::deterministic": _row("eng_a", "positions/g1__eng_a.pt")}
    _write_store(tmp_path, rows, [])  # positions/ exists but tensor missing

    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_benchmark_integrity.py", "--data-dir", str(tmp_path)],
    )

    with pytest.raises(SystemExit) as excinfo:
        vbi.main()

    assert excinfo.value.code == 1
