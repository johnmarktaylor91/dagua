"""Tests for the r79 baseline harness's provably-fresh sweep guarantees.

Covers the r80-P6 stale-resume fix: row-level git-sha/timestamp provenance
stamping, a loud warning when a resumed sweep reuses cached rows, and
``--fresh`` refusing to proceed if any row survives staging preparation.
Also covers the ``--size-blind-externals`` CLI flag added alongside it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dagua.eval.size_policy import set_size_aware_externals, size_aware_externals
from scripts import r79_baseline


def test_parse_args_fresh_and_resume_are_mutually_exclusive() -> None:
    """``--fresh`` and ``--resume`` together must fail argument parsing.

    Returns
    -------
    None
    """
    with pytest.raises(SystemExit):
        r79_baseline.parse_args(["--fresh", "--resume"])


def test_parse_args_fresh_alone_is_accepted() -> None:
    """``--fresh`` alone parses cleanly and defaults resume to False.

    Returns
    -------
    None
    """
    args = r79_baseline.parse_args(["--fresh"])
    assert args.fresh is True
    assert args.resume is False


def test_stamp_row_adds_git_sha_and_timestamp() -> None:
    """``stamp_row`` attaches provenance without mutating the input row.

    Returns
    -------
    None
    """
    row = {"graph": "g", "engine": "e", "status": "OK"}
    stamped = r79_baseline.stamp_row(row)

    assert "row_git_sha" not in row  # original row is untouched
    assert stamped["row_git_sha"] == r79_baseline.git_sha()
    assert isinstance(stamped["row_git_sha"], str) and stamped["row_git_sha"]
    assert stamped["row_written_at"].endswith("Z")
    # Base fields survive the stamp.
    assert stamped["graph"] == "g"
    assert stamped["engine"] == "e"
    assert stamped["status"] == "OK"


def test_append_row_writes_stamped_rows_to_a_tiny_fake_store(tmp_path: Path) -> None:
    """``append_row`` persists provenance-stamped rows to the JSONL store.

    Parameters
    ----------
    tmp_path : Path
        Temporary fake baseline output directory.

    Returns
    -------
    None
    """
    row_a = {"graph": "ga", "engine": "dagua", "status": "OK"}
    row_b = {"graph": "gb", "engine": "dagua", "status": "SKIP"}

    r79_baseline.append_row(tmp_path, row_a)
    r79_baseline.append_row(tmp_path, row_b)

    rows = r79_baseline.load_jsonl_rows(tmp_path)
    assert len(rows) == 2
    for row in rows:
        assert row["row_git_sha"] == r79_baseline.git_sha()
        assert "row_written_at" in row


def test_warn_resumed_rows_prints_loud_warning_with_count(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A resumed sweep with cached rows prints a loud, counted warning.

    Parameters
    ----------
    capsys : pytest.CaptureFixture[str]
        Captures stdout for the warning banner.

    Returns
    -------
    None
    """
    skipped = {("graph_a", "dagua"), ("graph_b", "graphviz_dot")}

    r79_baseline.warn_resumed_rows(skipped, resume=True)

    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "2 CACHED ROW(S)" in out
    assert "--fresh" in out


def test_warn_resumed_rows_silent_when_not_resuming(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No warning is printed for a non-resumed run or an empty skip set.

    Parameters
    ----------
    capsys : pytest.CaptureFixture[str]
        Captures stdout to assert nothing was printed.

    Returns
    -------
    None
    """
    r79_baseline.warn_resumed_rows(set(), resume=False)
    r79_baseline.warn_resumed_rows({("g", "dagua")}, resume=False)

    assert capsys.readouterr().out == ""


def test_assert_fresh_store_passes_on_empty_staging_store(tmp_path: Path) -> None:
    """An empty staging store satisfies the fresh-run invariant.

    Parameters
    ----------
    tmp_path : Path
        Temporary fake staging directory with no rows.

    Returns
    -------
    None
    """
    r79_baseline.assert_fresh_store(tmp_path)  # must not raise


def test_assert_fresh_store_refuses_cached_rows(tmp_path: Path) -> None:
    """A staging store with a resumable row fails the fresh-run invariant.

    This is the core stale-resume-hole regression test: a leftover OK row
    (with its position file present, matching ``complete_keys``'s
    requirement) must be refused outright rather than silently reused.

    Parameters
    ----------
    tmp_path : Path
        Temporary fake staging directory.

    Returns
    -------
    None
    """
    positions_dir = tmp_path / "positions"
    positions_dir.mkdir(parents=True)
    position_relpath = "positions/stale__dagua.pt"
    (tmp_path / position_relpath).write_bytes(b"not a real tensor, just needs to exist")
    stale_row = {
        "graph": "stale_graph",
        "engine": "dagua",
        "status": "OK",
        "positions_path": position_relpath,
    }
    r79_baseline.append_row(tmp_path, stale_row)

    with pytest.raises(RuntimeError, match="remain resumable"):
        r79_baseline.assert_fresh_store(tmp_path)


def test_parse_args_size_blind_externals_defaults_false() -> None:
    """``--size-blind-externals`` defaults off (size-aware is the default).

    Returns
    -------
    None
    """
    args = r79_baseline.parse_args([])
    assert args.size_blind_externals is False


def test_parse_args_size_blind_externals_flag_sets_true() -> None:
    """``--size-blind-externals`` flips the flag on when passed.

    Returns
    -------
    None
    """
    args = r79_baseline.parse_args(["--size-blind-externals"])
    assert args.size_blind_externals is True


def test_main_wires_size_blind_externals_into_size_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``main()`` propagates ``--size-blind-externals`` to the size policy.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to short-circuit ``main()`` after the size-policy wiring line so
        this test does not need a real corpus or engine run.

    Returns
    -------
    None
    """

    class _StopAfterWiring(Exception):
        pass

    original_parse_args = r79_baseline.parse_args

    def fake_build_corpus():
        raise _StopAfterWiring

    monkeypatch.setattr(
        r79_baseline, "parse_args", lambda: original_parse_args(["--size-blind-externals"])
    )
    monkeypatch.setattr(r79_baseline, "build_corpus", fake_build_corpus)
    try:
        with pytest.raises(_StopAfterWiring):
            r79_baseline.main()
        assert size_aware_externals() is False
    finally:
        set_size_aware_externals(True)
