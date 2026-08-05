"""Unit tests for the fidelity output post-flight validator."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

from scripts import validate_fidelity_output as vfo


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    """Write a small CSV fixture.

    Parameters
    ----------
    path : Path
        Destination CSV path.
    fieldnames : list[str]
        Column names.
    rows : list[dict[str, str]]
        Row payloads.
    """
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_warning_only_output(data_dir: Path) -> None:
    """Materialize output that produces warnings but no errors.

    Parameters
    ----------
    data_dir : Path
        Output directory.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        data_dir / "algorithm_summary.csv",
        ["algorithm_family", "verdict", "procrustes_rmsd_mean"],
        [
            {"algorithm_family": "fr", "verdict": "equivalent", "procrustes_rmsd_mean": "0.1"},
            {"algorithm_family": "kk", "verdict": "equivalent", "procrustes_rmsd_mean": "0.2"},
        ],
    )
    _write_csv(
        data_dir / "per_graph_detail.csv",
        ["algorithm_family", "verdict", "procrustes_rmsd_mean"],
        [
            {
                "algorithm_family": "fr",
                "verdict": "insufficient_data",
                "procrustes_rmsd_mean": "0.5",
            },
            {
                "algorithm_family": "fr",
                "verdict": "insufficient_data",
                "procrustes_rmsd_mean": "0.5",
            },
            {
                "algorithm_family": "kk",
                "verdict": "insufficient_data",
                "procrustes_rmsd_mean": "0.5",
            },
            {"algorithm_family": "kk", "verdict": "equivalent", "procrustes_rmsd_mean": "0.5"},
        ],
    )


def test_validate_output_separates_errors_from_warnings(tmp_path: Path) -> None:
    """Severity is carried structurally, never re-parsed from message text."""
    missing_dir = tmp_path / "missing"
    missing_dir.mkdir()
    errors, warnings = vfo.validate_output(missing_dir)
    assert errors == ["algorithm_summary.csv does not exist"]
    assert warnings == []

    warn_dir = tmp_path / "warn_only"
    _write_warning_only_output(warn_dir)
    errors, warnings = vfo.validate_output(warn_dir)
    assert errors == []
    assert len(warnings) == 1
    assert "insufficient_data" in warnings[0]


def test_main_exit_code_comes_from_error_severity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Errors exit 1; warnings alone exit 0 -- regardless of message wording."""
    missing_dir = tmp_path / "missing"
    missing_dir.mkdir()
    monkeypatch.setattr(sys, "argv", ["validate_fidelity_output.py", "--data", str(missing_dir)])
    with pytest.raises(SystemExit) as excinfo:
        vfo.main()
    assert excinfo.value.code == 1

    warn_dir = tmp_path / "warn_only"
    _write_warning_only_output(warn_dir)
    monkeypatch.setattr(sys, "argv", ["validate_fidelity_output.py", "--data", str(warn_dir)])
    with pytest.raises(SystemExit) as excinfo:
        vfo.main()
    assert excinfo.value.code == 0
