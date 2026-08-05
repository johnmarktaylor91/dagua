"""Unit tests for the two-store safe purge tool."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts import safe_purge_variants as spv


def _row(graph: str, engine: str, positions_file: str | None) -> dict[str, object]:
    """Build a minimal results.json row payload.

    Parameters
    ----------
    graph : str
        Graph name.
    engine : str
        Engine name.
    positions_file : str | None
        Relative tensor path.

    Returns
    -------
    dict[str, object]
        Row payload.
    """
    return {
        "graph_name": graph,
        "engine_name": engine,
        "seed": None,
        "status": "ok",
        "positions_file": positions_file,
    }


def _build_pt_store(data_dir: Path) -> None:
    """Materialize a two-engine .pt-format store with one orphan tensor.

    Parameters
    ----------
    data_dir : Path
        Output directory.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = {
        "g1::eng_a::deterministic": _row("g1", "eng_a", "positions/g1__eng_a.pt"),
        "g2::eng_a::seed42": _row("g2", "eng_a", "positions/g2__eng_a__seed42.pt"),
        "g1::eng_b::deterministic": _row("g1", "eng_b", "positions/g1__eng_b.pt"),
    }
    (data_dir / "results.json").write_text(json.dumps(rows), encoding="utf-8")
    positions_dir = data_dir / "positions"
    positions_dir.mkdir()
    for name in (
        "g1__eng_a.pt",
        "g2__eng_a__seed42.pt",
        "g3__eng_a.pt",  # orphan: row already gone, must still be purged
        "g1__eng_b.pt",
    ):
        (positions_dir / name).write_bytes(b"pt")


def test_engine_position_pt_paths_matches_seeded_and_orphan_tensors(
    tmp_path: Path,
) -> None:
    """Filename attribution covers deterministic, seeded, and orphan tensors."""
    _build_pt_store(tmp_path)

    matched = spv.engine_position_pt_paths(tmp_path / "positions", {"eng_a"})

    assert sorted(path.name for path in matched) == [
        "g1__eng_a.pt",
        "g2__eng_a__seed42.pt",
        "g3__eng_a.pt",
    ]


def test_engine_position_pt_paths_does_not_match_for_variant_suffix(
    tmp_path: Path,
) -> None:
    """Purging an engine must not delete __for__ reference-variant tensors."""
    positions_dir = tmp_path / "positions"
    positions_dir.mkdir(parents=True)
    (positions_dir / "g1__classic_sfdp.pt").write_bytes(b"pt")
    (positions_dir / "g1__graphviz_sfdp__for__classic_sfdp.pt").write_bytes(b"pt")
    (positions_dir / "g1__graphviz_sfdp__for__classic_sfdp__seed42.pt").write_bytes(b"pt")

    plain = spv.engine_position_pt_paths(positions_dir, {"classic_sfdp"})
    variant = spv.engine_position_pt_paths(positions_dir, {"graphviz_sfdp__for__classic_sfdp"})

    assert [path.name for path in plain] == ["g1__classic_sfdp.pt"]
    assert sorted(path.name for path in variant) == [
        "g1__graphviz_sfdp__for__classic_sfdp.pt",
        "g1__graphviz_sfdp__for__classic_sfdp__seed42.pt",
    ]


def test_purge_removes_pt_tensors_and_rows_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A confirmed purge on a .pt store leaves no resurrectable orphans."""
    data_dir = tmp_path / "store"
    _build_pt_store(data_dir)
    monkeypatch.setattr(
        sys,
        "argv",
        ["safe_purge_variants.py", "eng_a", "--data-dir", str(data_dir), "--confirm"],
    )

    spv.main()

    remaining_rows = json.loads((data_dir / "results.json").read_text(encoding="utf-8"))
    assert sorted(remaining_rows) == ["g1::eng_b::deterministic"]
    remaining_tensors = sorted(p.name for p in (data_dir / "positions").glob("*.pt"))
    assert remaining_tensors == ["g1__eng_b.pt"]


def test_dry_run_touches_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without --confirm the purge is report-only."""
    data_dir = tmp_path / "store"
    _build_pt_store(data_dir)
    before_rows = (data_dir / "results.json").read_text(encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["safe_purge_variants.py", "eng_a", "--data-dir", str(data_dir)],
    )

    spv.main()

    assert (data_dir / "results.json").read_text(encoding="utf-8") == before_rows
    assert len(list((data_dir / "positions").glob("*.pt"))) == 4


def test_purge_commits_results_json_before_position_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Crash-safety ordering: rows commit first, positions second (WP12-F24).

    A crash between the two commits must leave harmless orphan tensors,
    never ok rows whose positions are already gone.
    """
    h5py = pytest.importorskip("h5py")
    data_dir = tmp_path / "store"
    data_dir.mkdir()
    rows = {
        "g1::eng_a::seed42": _row("g1", "eng_a", None),
        "g1::eng_b::seed42": _row("g1", "eng_b", None),
    }
    (data_dir / "results.json").write_text(json.dumps(rows), encoding="utf-8")
    with h5py.File(data_dir / "positions.h5", "w") as h5f:
        h5f.create_dataset("g1::eng_a::seed42", data=[1.0, 2.0])
        h5f.create_dataset("g1::eng_b::seed42", data=[3.0, 4.0])

    rename_destinations: list[str] = []
    real_rename = spv.os.rename

    def _recording_rename(src: object, dst: object) -> None:
        rename_destinations.append(str(dst))
        real_rename(src, dst)

    monkeypatch.setattr(spv.os, "rename", _recording_rename)
    monkeypatch.setattr(
        sys,
        "argv",
        ["safe_purge_variants.py", "eng_a", "--data-dir", str(data_dir), "--confirm"],
    )

    spv.main()

    assert len(rename_destinations) == 2
    assert rename_destinations[0].endswith("results.json")
    assert rename_destinations[1].endswith("positions.h5")
