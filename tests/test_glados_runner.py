"""Tests for the GLaDOS holdout runner and blind subset module.

All tests run on synthetic fixtures under ``tests/fixtures/glados/`` --
invented topologies, never derived from any real corpus (holdout opacity,
R-12). No test touches ``eval_output/stdcorpora`` or any fetched content.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import pytest

import dagua
import scripts.glados_holdout_run as glados
import scripts.glados_subset as subset_mod
from scripts.run_benchmark import build_record_key

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "glados"

# Worktree/editable-install contamination guard: these tests are only
# meaningful when they exercise the tree that contains them.
assert Path(dagua.__file__).resolve().as_posix().startswith(REPO_ROOT.as_posix()), (
    f"dagua imports from {dagua.__file__}, not {REPO_ROOT}; editable-install "
    "contamination -- test results would be meaningless"
)


def _copy_fixture(corpus_dir: Path, corpus: str, filename: str) -> Path:
    """Copy one synthetic fixture into a scratch corpus directory.

    Parameters
    ----------
    corpus_dir : Path
        Scratch corpus root.
    corpus : str
        Corpus subdirectory name.
    filename : str
        Fixture file name.

    Returns
    -------
    Path
        Destination path.
    """
    source = FIXTURES / corpus / filename
    destination = corpus_dir / corpus / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())
    return destination


def _write_engines_file(tmp_path: Path, engines: List[str]) -> Path:
    """Write an --engines-file JSON list.

    Parameters
    ----------
    tmp_path : Path
        Test temp dir.
    engines : List[str]
        Engine names.

    Returns
    -------
    Path
        Engines file path.
    """
    path = tmp_path / "engines.json"
    path.write_text(json.dumps(engines), encoding="utf-8")
    return path


def _run_main(argv: List[str], monkeypatch: pytest.MonkeyPatch) -> int:
    """Run the runner's main() in-process with a clean native env.

    Parameters
    ----------
    argv : List[str]
        CLI arguments.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to guarantee ``DAGUA_NATIVE_DISABLE_W5`` is unset.

    Returns
    -------
    int
        Exit code.
    """
    monkeypatch.delenv("DAGUA_NATIVE_DISABLE_W5", raising=False)
    return glados.main(argv)


# ---------------------------------------------------------------------------
# 1. Blind subset module
# ---------------------------------------------------------------------------


def test_rank_candidates_matches_hand_computed_sha256_ordering() -> None:
    """Hash-rank ordering equals an independently hand-computed ordering.

    Also proves PURITY: the names rank identically whether or not the files
    exist (none of these exist anywhere).
    """
    seed = "glados-holdout-2026-08-05"
    names = ["zeta.graph", "alpha.graph", "mid.graph", "nonexistent-file.graph"]
    expected = sorted(
        (hashlib.sha256(f"{seed}\0rome\0{name}".encode("utf-8")).hexdigest(), name)
        for name in names
    )
    ranked = subset_mod.rank_candidates(seed, [("rome", name) for name in names])
    assert list(ranked) == ["rome"]
    assert [(digest, name) for digest, _, name in ranked["rome"]] == expected


def test_partition_selects_ceil_fraction_per_corpus_disjoint_and_complete() -> None:
    """Selected/sealed partition is disjoint, complete, and ceil-sized."""
    candidates = [
        {"corpus": "rome", "filename": f"r{i}.graph", "relpath": f"rome/r{i}.graph"}
        for i in range(5)
    ] + [
        {"corpus": "north", "filename": f"n{i}.gml", "relpath": f"north/n{i}.gml"} for i in range(3)
    ]
    selected, sealed, per_corpus = subset_mod.partition_candidates("seed-x", 0.45, candidates)
    assert per_corpus["rome"] == {"candidates": 5, "selected": math.ceil(0.45 * 5)}
    assert per_corpus["north"] == {"candidates": 3, "selected": math.ceil(0.45 * 3)}
    selected_keys = {(item["corpus"], item["filename"]) for item in selected}
    sealed_keys = {(item["corpus"], item["filename"]) for item in sealed}
    assert not selected_keys & sealed_keys
    assert selected_keys | sealed_keys == {
        (candidate["corpus"], candidate["filename"]) for candidate in candidates
    }
    assert len(selected) == 3 + 2
    # Sorted by (corpus, rank_hash).
    assert selected == sorted(selected, key=lambda item: (item["corpus"], item["rank_hash"]))


def test_rank_candidates_duplicate_basenames_hard_error() -> None:
    """Duplicate basenames within one corpus abort selection (WP10-F05)."""
    with pytest.raises(ValueError, match="duplicate basenames"):
        subset_mod.rank_candidates("s", [("rome", "a.graph"), ("rome", "a.graph")])


def test_subset_cli_writes_partition_and_refuses_overwrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI writes SUBSET.json + SEALED_REMAINDER.json without opening candidates."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    lines = (
        [f"{corpus_dir.as_posix()}/rome/g{i}.graph" for i in range(4)]
        + [f"{corpus_dir.as_posix()}/north/d{i}.gml" for i in range(2)]
        + [
            f"{corpus_dir.as_posix()}/suitesparse/README.md",  # unsupported ext
            f"{corpus_dir.as_posix()}/mystery/x.graph",  # unknown corpus
            "/elsewhere/rome/y.graph",  # not under corpus dir
        ]
    )
    fetched = corpus_dir / "FETCHED_FILES.txt"
    fetched.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # NONE of the candidate files exist: selection must be pure over names.

    exit_code = subset_mod.main(["--corpus-dir", str(corpus_dir)])
    assert exit_code == 0
    err = capsys.readouterr().err
    assert "unsupported extension" in err
    assert "unknown corpus" in err
    assert "not under corpus dir" in err

    payload = json.loads((corpus_dir / "SUBSET.json").read_text(encoding="utf-8"))
    sealed = json.loads((corpus_dir / "SEALED_REMAINDER.json").read_text(encoding="utf-8"))
    assert payload["seed_string"] == subset_mod.DEFAULT_SEED_STRING
    assert payload["fraction"] == 0.45
    assert payload["per_corpus"]["rome"] == {"candidates": 4, "selected": 2}
    assert payload["per_corpus"]["north"] == {"candidates": 2, "selected": 1}
    assert payload["fetched_list_sha256"] == hashlib.sha256(fetched.read_bytes()).hexdigest()
    assert len(payload["selected"]) == 3
    assert len(sealed["sealed"]) == 3
    for entry in payload["selected"]:
        assert entry["rank_hash"] == subset_mod.rank_hash(
            payload["seed_string"], entry["corpus"], entry["filename"]
        )

    # Refuses overwrite without --force (merge freeze anchor); --force works.
    assert subset_mod.main(["--corpus-dir", str(corpus_dir)]) == 1
    assert subset_mod.main(["--corpus-dir", str(corpus_dir), "--force"]) == 0


def test_subset_cli_duplicate_basenames_abort_before_writing(tmp_path: Path) -> None:
    """Duplicate stems abort with nothing written."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "FETCHED_FILES.txt").write_text(
        f"{corpus_dir.as_posix()}/rome/a.graph\n{corpus_dir.as_posix()}/rome/a.graph\n",
        encoding="utf-8",
    )
    assert subset_mod.main(["--corpus-dir", str(corpus_dir)]) == 1
    assert not (corpus_dir / "SUBSET.json").exists()
    assert not (corpus_dir / "SEALED_REMAINDER.json").exists()


# ---------------------------------------------------------------------------
# 2. Runner unit pins (deterministic seam, native env, directedness policy)
# ---------------------------------------------------------------------------


def test_native_layout_kwargs_pin_deterministic_native_true() -> None:
    """The adapter-call seam passes deterministic_native=True (WP03-F02)."""
    kwargs = glados._native_layout_kwargs(1800.0, 42, True)
    assert kwargs == {"timeout": 1800.0, "seed": 42, "deterministic_native": True}
    assert "deterministic_native" not in glados._native_layout_kwargs(1800.0, 42, False)


def test_preflight_native_env_rejects_disable_w5(monkeypatch: pytest.MonkeyPatch) -> None:
    """A set DAGUA_NATIVE_DISABLE_W5 fails preflight (WP02A-F01)."""
    monkeypatch.setenv("DAGUA_NATIVE_DISABLE_W5", "1")
    with pytest.raises(glados.PreflightError, match="DAGUA_NATIVE_DISABLE_W5"):
        glados._preflight_native_env()
    monkeypatch.delenv("DAGUA_NATIVE_DISABLE_W5")
    glados._preflight_native_env()  # clean env passes


@pytest.mark.parametrize(
    ("corpus", "filename", "expected_directed", "expected_source"),
    [
        ("rome", "ring6.graph", False, "policy"),
        ("suitesparse", "tri4.mtx", False, "policy"),
        ("north", "chain4.gml", True, "policy"),  # metadata absent -> directed
        ("north", "fan5.graphml", True, "policy"),  # edgedefault absent -> directed
        ("north", "spine3.gml", True, "format"),  # explicit directed 1
    ],
)
def test_directedness_policy_table(
    corpus: str, filename: str, expected_directed: bool, expected_source: str
) -> None:
    """The per-corpus directedness policy matches the spec 5.2 table."""
    override, source = glados.directedness_policy(corpus, FIXTURES / corpus / filename)
    assert override is expected_directed
    assert source == expected_source


def test_directedness_policy_misc_uses_loader_default(tmp_path: Path) -> None:
    """Misc (scratch) corpora keep the loader default."""
    override, source = glados.directedness_policy("misc", tmp_path / "x.graph")
    assert override is None
    assert source == "loader_default"


# ---------------------------------------------------------------------------
# 3. Load phase (isolation, sanity gate, dedupe, policy, scale gate)
# ---------------------------------------------------------------------------


def test_load_phase_rome_undirected_even_when_path_contains_dag(tmp_path: Path) -> None:
    """F02 regression: a 'dag' substring in the absolute path never directs rome."""
    corpus_dir = tmp_path / "dagville" / "corpus"
    _copy_fixture(corpus_dir, "rome", "ring6.graph")
    assert "dag" in corpus_dir.resolve().as_posix()

    entries, load_rows, _ = glados.load_phase(corpus_dir, None, 2000, 200_000)

    assert load_rows == []
    assert len(entries) == 1
    assert entries[0].name == "rome/ring6"
    assert entries[0].directed is False
    assert entries[0].directed_source == "policy"


def test_load_phase_north_metadata_absent_is_directed_by_policy(tmp_path: Path) -> None:
    """North .gml/.graphml without direction metadata score directed (policy)."""
    corpus_dir = tmp_path / "corpus"
    _copy_fixture(corpus_dir, "north", "chain4.gml")
    _copy_fixture(corpus_dir, "north", "fan5.graphml")
    _copy_fixture(corpus_dir, "north", "spine3.gml")

    entries, load_rows, _ = glados.load_phase(corpus_dir, None, 2000, 200_000)

    assert load_rows == []
    by_name = {entry.name: entry for entry in entries}
    assert by_name["north/chain4"].directed is True
    assert by_name["north/chain4"].directed_source == "policy"
    assert by_name["north/fan5"].directed is True
    assert by_name["north/fan5"].directed_source == "policy"
    assert by_name["north/spine3"].directed is True
    assert by_name["north/spine3"].directed_source == "format"


def test_load_phase_isolates_corrupt_file_as_load_error(tmp_path: Path) -> None:
    """One corrupt file becomes a LOAD_ERROR row; the run continues (WP10-F01)."""
    corpus_dir = tmp_path / "corpus"
    _copy_fixture(corpus_dir, "rome", "ring6.graph")
    _copy_fixture(corpus_dir, "rome", "broken.gml")

    entries, load_rows, _ = glados.load_phase(corpus_dir, None, 2000, 200_000)

    assert [entry.name for entry in entries] == ["rome/ring6"]
    assert len(load_rows) == 1
    assert load_rows[0]["status"] == "LOAD_ERROR"
    assert load_rows[0]["graph"] == "rome/broken"
    assert load_rows[0]["error"]


def test_load_phase_flags_trivial_parse_as_suspect(tmp_path: Path) -> None:
    """A many-row file collapsing to 1 node is LOAD_SUSPECT, not tallied (WP10-F03)."""
    corpus_dir = tmp_path / "corpus"
    _copy_fixture(corpus_dir, "rome", "collapse1.graph")
    _copy_fixture(corpus_dir, "rome", "ring6.graph")

    entries, load_rows, _ = glados.load_phase(corpus_dir, None, 2000, 200_000)

    assert [entry.name for entry in entries] == ["rome/ring6"]
    suspect = [row for row in load_rows if row["status"] == "LOAD_SUSPECT"]
    assert len(suspect) == 1
    assert suspect[0]["graph"] == "rome/collapse1"
    assert "numeric rows" in suspect[0]["reason"]
    assert suspect[0]["telemetry"]["parse_branch"] == "padded_adjacency"


def test_load_phase_duplicate_stems_abort(tmp_path: Path) -> None:
    """Two formats sharing a stem in one corpus are fatal (WP10-F05)."""
    corpus_dir = tmp_path / "corpus"
    _copy_fixture(corpus_dir, "rome", "ring6.graph")
    (corpus_dir / "rome" / "ring6.gml").write_text(
        'graph [\n  node [ id 0 label "a" ]\n  node [ id 1 label "b" ]\n'
        "  edge [ source 0 target 1 ]\n]\n",
        encoding="utf-8",
    )
    with pytest.raises(glados.LoadPhaseFatal, match="duplicate loaded graph names"):
        glados.load_phase(corpus_dir, None, 2000, 200_000)


def test_load_phase_nonsquare_mtx_skipped_with_reason(tmp_path: Path) -> None:
    """Rectangular matrices are excluded with a nonsquare_mtx reason (WP10-F06)."""
    corpus_dir = tmp_path / "corpus"
    _copy_fixture(corpus_dir, "suitesparse", "rect3x5.mtx")
    _copy_fixture(corpus_dir, "suitesparse", "tri4.mtx")

    entries, load_rows, _ = glados.load_phase(corpus_dir, None, 2000, 200_000)

    assert [entry.name for entry in entries] == ["suitesparse/tri4"]
    assert len(load_rows) == 1
    assert load_rows[0]["status"] == "SKIP"
    assert load_rows[0]["reason"].startswith("nonsquare_mtx:3x5")


def test_load_phase_scale_gate_bounds_and_telemetry(tmp_path: Path) -> None:
    """N/E verification: graphs above --max-edges are excluded with reasons."""
    corpus_dir = tmp_path / "corpus"
    _copy_fixture(corpus_dir, "rome", "ring6.graph")  # 6 nodes / 6 edges

    entries, load_rows, _ = glados.load_phase(corpus_dir, None, 2000, 5)
    assert entries == []
    assert load_rows[0]["reason"] == "scale_gate_edges:6>5"

    entries, load_rows, _ = glados.load_phase(corpus_dir, None, 5, 200_000)
    assert entries == []
    assert load_rows[0]["reason"] == "max_nodes:6"

    entries, _, _ = glados.load_phase(corpus_dir, None, 2000, 200_000)
    telemetry = entries[0].telemetry
    assert telemetry["parse_branch"] == "adjacency_list"
    assert telemetry["raw_numeric_rows"] == 7
    assert telemetry["edges_kept"] == 6
    assert telemetry["self_loops_dropped"] == 0
    assert telemetry["duplicates_dropped"] == 6  # undirected adjacency lists both dirs


def test_load_phase_subset_filter_and_missing_subset_file(tmp_path: Path) -> None:
    """Subset keeps only selected relpaths; missing selected files are LOAD_ERRORs."""
    corpus_dir = tmp_path / "corpus"
    _copy_fixture(corpus_dir, "rome", "ring6.graph")
    _copy_fixture(corpus_dir, "rome", "petal5.graph")

    subset = {"rome/ring6.graph", "rome/gone.graph"}
    entries, load_rows, stats = glados.load_phase(corpus_dir, subset, 2000, 200_000)

    assert [entry.name for entry in entries] == ["rome/ring6"]
    assert stats["sealed_skipped"] == 1  # petal5 stays sealed, never loaded
    missing = [row for row in load_rows if row["status"] == "LOAD_ERROR"]
    assert len(missing) == 1
    assert missing[0]["relpath"] == "rome/gone.graph"
    assert "subset file missing" in missing[0]["error"]


# ---------------------------------------------------------------------------
# 4. Row store: fsynced appends, torn lines, dedupe
# ---------------------------------------------------------------------------


def test_load_rows_tolerant_truncates_torn_final_line(tmp_path: Path) -> None:
    """A torn FINAL line is tolerated + truncated; interior tears are fatal."""
    directory = tmp_path / "staging"
    glados.append_row(directory, {"record_key": "a", "status": "OK"})
    glados.append_row(directory, {"record_key": "b", "status": "OK"})
    with glados.rows_path(directory).open("a", encoding="utf-8") as handle:
        handle.write('{"record_key": "c", "status": "OK", "trunca')

    rows, warnings = glados.load_rows_tolerant(directory)
    assert [row["record_key"] for row in rows] == ["a", "b"]
    assert warnings and "torn final" in warnings[0]
    # The torn line is gone from disk; a re-read is clean.
    rows2, warnings2 = glados.load_rows_tolerant(directory)
    assert [row["record_key"] for row in rows2] == ["a", "b"]
    assert warnings2 == []

    glados.rows_path(directory).write_text(
        '{"broken\n{"record_key": "d", "status": "OK"}\n', encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="malformed interior"):
        glados.load_rows_tolerant(directory)


def test_dedupe_rows_last_occurrence_wins() -> None:
    """Scored re-appends supersede their layout rows."""
    rows = [
        {"record_key": "k", "status": "OK"},
        {"record_key": "k", "status": "OK", "v3_tiered": 88.0},
    ]
    assert glados.dedupe_rows(rows)["k"]["v3_tiered"] == 88.0


# ---------------------------------------------------------------------------
# 5. Guards + refusals (exit codes 3 and 4)
# ---------------------------------------------------------------------------


def test_missing_subset_on_real_corpus_dir_refused_exit_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corpus dir string-matching eval_output/stdcorpora requires --subset."""
    corpus_dir = tmp_path / "eval_output" / "stdcorpora"
    (corpus_dir / "rome").mkdir(parents=True)
    exit_code = _run_main(
        [
            "--corpus-dir",
            str(corpus_dir),
            "--output-dir",
            str(tmp_path / "out"),
        ],
        monkeypatch,
    )
    assert exit_code == 4


def test_rss_abort_publishes_partial_and_exits_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parent RSS ceiling stops the run, publishes partial results, exits 3."""
    corpus_dir = tmp_path / "corpus"
    _copy_fixture(corpus_dir, "rome", "ring6.graph")
    _copy_fixture(corpus_dir, "rome", "petal5.graph")
    output_dir = tmp_path / "out"

    monkeypatch.setattr(glados, "PARENT_GC_TRIM_INTERVAL_ROWS", 1)
    monkeypatch.setattr(glados, "parent_rss_bytes", lambda: glados.PARENT_RSS_ABORT_BYTES)

    exit_code = _run_main(
        [
            "--corpus-dir",
            str(corpus_dir),
            "--output-dir",
            str(output_dir),
            "--engines-file",
            str(_write_engines_file(tmp_path, ["graphviz_dot"])),
            "--workers",
            "1",
        ],
        monkeypatch,
    )

    assert exit_code == 3
    payload = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    assert payload["aborted"] is True
    assert "abort ceiling" in payload["aborted_reason"]
    assert len(payload["rows"]) == 1  # guard fired after the first row


def test_child_memkill_row_recorded_under_selftest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--rss-guard-selftest exercises the child RSS watchdog (WP10-F09)."""
    corpus_dir = tmp_path / "corpus"
    _copy_fixture(corpus_dir, "rome", "ring6.graph")
    output_dir = tmp_path / "out"

    exit_code = _run_main(
        [
            "--corpus-dir",
            str(corpus_dir),
            "--output-dir",
            str(output_dir),
            "--engines-file",
            str(_write_engines_file(tmp_path, ["graphviz_dot"])),
            "--workers",
            "1",
            "--rss-guard-selftest",
        ],
        monkeypatch,
    )

    assert exit_code == 0
    payload = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    rows = payload["rows"]
    assert len(rows) == 1
    assert rows[0]["status"] == "ERROR"
    assert rows[0]["status_detail"] == "memkill"
    assert "exceeded ceiling" in rows[0]["error"]
    report = (output_dir / "GLADOS_RUN_REPORT.md").read_text(encoding="utf-8")
    assert "memkills 1" in report


# ---------------------------------------------------------------------------
# 6. Resume hardening (seed-aware keys, torn line, stale dot temps)
# ---------------------------------------------------------------------------


def test_resume_skips_completed_seed_aware_rows_and_cleans_temps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume: sentinel row untouched, torn line tolerated, dot temps cleaned."""
    corpus_dir = tmp_path / "corpus"
    _copy_fixture(corpus_dir, "rome", "ring6.graph")
    _copy_fixture(corpus_dir, "rome", "petal5.graph")
    output_dir = tmp_path / "out"
    staging = output_dir.with_name(f"{output_dir.name}.tmp")

    sentinel_key = build_record_key("rome/ring6", "graphviz_dot", None)
    sentinel: Dict[str, Any] = {
        "graph": "rome/ring6",
        "corpus": "rome",
        "engine": "graphviz_dot",
        "seed": None,
        "record_key": sentinel_key,
        "status": "OK",
        "runtime_s": -12345.0,  # sentinel: proves the row was not recomputed
        "positions_path": None,
        "v3_tiered": 55.5,  # pre-scored: scorer must skip it too
        "nodes": 6,
        "edges": 6,
        "directed": False,
        "directed_source": "policy",
        "source_path": str(corpus_dir / "rome" / "ring6.graph"),
        "error": None,
    }
    glados.append_row(staging, sentinel)
    with glados.rows_path(staging).open("a", encoding="utf-8") as handle:
        handle.write('{"record_key": "torn-line", "stat')  # crash mid-append
    stale_temp = staging / "positions" / ".rome_x__dagua__deterministic.child.pt"
    stale_temp.parent.mkdir(parents=True, exist_ok=True)
    stale_temp.write_bytes(b"stale")

    exit_code = _run_main(
        [
            "--corpus-dir",
            str(corpus_dir),
            "--output-dir",
            str(output_dir),
            "--engines-file",
            str(_write_engines_file(tmp_path, ["graphviz_dot"])),
            "--workers",
            "1",
            "--score-workers",
            "1",
            "--resume",
        ],
        monkeypatch,
    )

    assert exit_code == 0
    payload = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    rows = {row["record_key"]: row for row in payload["rows"]}
    # Sentinel preserved byte-for-byte semantics: never recomputed or rescored.
    assert rows[sentinel_key]["runtime_s"] == -12345.0
    assert rows[sentinel_key]["v3_tiered"] == 55.5
    # The pending graph ran with a seed-aware deterministic key.
    petal_key = build_record_key("rome/petal5", "graphviz_dot", None)
    assert rows[petal_key]["status"] == "OK"
    assert rows[petal_key]["v3_tiered"] is not None
    # Torn line tolerated + logged; stale dot temp cleaned; validate passed
    # (publish would have raised otherwise).
    assert any("torn final" in warning for warning in payload["warnings"])
    assert not (output_dir / "positions" / stale_temp.name).exists()


# ---------------------------------------------------------------------------
# 7. End-to-end smoke + native determinism (spawn children, V3 scoring)
# ---------------------------------------------------------------------------


def _read_positions_sha(output_dir: Path, relpath: str) -> str:
    """Return the sha256 of one published position tensor.

    Parameters
    ----------
    output_dir : Path
        Published run directory.
    relpath : str
        Row ``positions_path``.

    Returns
    -------
    str
        Hex digest.
    """
    return hashlib.sha256((output_dir / relpath).read_bytes()).hexdigest()


def test_runner_end_to_end_two_fixture_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """E2E: native + one field engine on a 2-fixture scratch corpus.

    Exercises subprocess isolation (spawn children), V3 scoring (ruler=v3 on
    every scored row), tally, report rendering, work-plan printout, and the
    report-context passthrough. Runs main() in-process so the spawn children
    are real but the parent import cost is paid once.
    """
    corpus_dir = tmp_path / "corpus"
    _copy_fixture(corpus_dir, "rome", "ring6.graph")
    _copy_fixture(corpus_dir, "north", "chain4.gml")
    output_dir = tmp_path / "out"
    context_path = tmp_path / "context.json"
    context_path.write_text(
        json.dumps(
            {"item4_strict": 106, "scale_fix_note": "test note", "ruler_ledger_path": "missing"}
        ),
        encoding="utf-8",
    )

    exit_code = _run_main(
        [
            "--corpus-dir",
            str(corpus_dir),
            "--output-dir",
            str(output_dir),
            "--seed",
            "42",
            "--native-deterministic",
            "--native-timeout",
            "120",
            "--engine-timeout",
            "60",
            "--workers",
            "2",
            "--engines-file",
            str(_write_engines_file(tmp_path, ["dagua", "graphviz_dot"])),
            "--report-context",
            str(context_path),
            "--archive-dir",
            str(tmp_path / "archive"),
        ],
        monkeypatch,
    )

    assert exit_code == 0
    payload = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    assert glados.rows_path(output_dir).is_file()

    rows = payload["rows"]
    ok_rows = [row for row in rows if row["status"] == "OK"]
    assert len(ok_rows) == 4  # 2 graphs x (dagua + graphviz_dot)
    for row in ok_rows:
        assert row["v3_tiered"] is not None
        assert row["ruler_version"] == "v3"
        assert (output_dir / row["positions_path"]).is_file()
        assert row["scoring_signature"] == payload["scoring_signature"]
    native_rows = [row for row in ok_rows if row["engine"] == "dagua"]
    assert {row["record_key"] for row in native_rows} == {
        build_record_key("rome/ring6", "dagua", None),
        build_record_key("north/chain4", "dagua", None),
    }
    # Directedness plumbed through to rows (policy table).
    by_graph = {row["graph"]: row for row in native_rows}
    assert by_graph["rome/ring6"]["directed"] is False
    assert by_graph["north/chain4"]["directed"] is True

    tally = payload["tally"]
    assert tally["overall"]["total"] == 2
    assert (
        tally["overall"]["strictly_best"]
        + tally["overall"]["tied"]
        + tally["overall"]["behind"]
        + tally["overall"]["missing"]
        == 2
    )

    report = (output_dir / "GLADOS_RUN_REPORT.md").read_text(encoding="utf-8")
    for heading in (
        "## 1. Provenance",
        "## 2. Overall holdout tally",
        "## 3. Per-corpus breakdown",
        "## 4. Per-engine field-best counts",
        "## 5. Native losses",
        "## 6. Errors, skips, timeouts, memory events",
        "## 7. Ruler-bug ledger (appendix)",
        "## 8. Scale-fix disclosure and adopted item-4 number",
        "## 9. Assumption log",
    ):
        assert heading in report
    assert "item4_strict: 106" in report
    assert "test note" in report

    # A-S2 archive: full tree copy + sha256 manifest.
    manifest = (tmp_path / "archive" / "glados_holdout" / "MANIFEST.sha256").read_text(
        encoding="utf-8"
    )
    assert "results.json" in manifest


def test_native_two_runs_byte_identical_positions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two deterministic native runs produce byte-identical position tensors."""
    corpus_dir = tmp_path / "corpus"
    _copy_fixture(corpus_dir, "rome", "ring6.graph")
    engines_file = _write_engines_file(tmp_path, ["dagua"])

    shas: List[str] = []
    for run in ("a", "b"):
        output_dir = tmp_path / f"out_{run}"
        exit_code = _run_main(
            [
                "--corpus-dir",
                str(corpus_dir),
                "--output-dir",
                str(output_dir),
                "--seed",
                "42",
                "--native-deterministic",
                "--native-timeout",
                "120",
                "--workers",
                "1",
                "--score-workers",
                "1",
                "--engines-file",
                str(engines_file),
            ],
            monkeypatch,
        )
        assert exit_code == 0
        payload = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
        (row,) = [r for r in payload["rows"] if r["engine"] == "dagua"]
        assert row["status"] == "OK"
        shas.append(_read_positions_sha(output_dir, row["positions_path"]))
        # score_position's own tensor hash must agree with the on-disk bytes.
        assert row["position_sha256"] == shas[-1]
    assert shas[0] == shas[1], "deterministic native rows must be byte-identical"


def test_load_error_row_keeps_run_alive_exit_0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corrupt corpus file yields a LOAD_ERROR row and the run still exits 0."""
    corpus_dir = tmp_path / "corpus"
    _copy_fixture(corpus_dir, "rome", "ring6.graph")
    _copy_fixture(corpus_dir, "rome", "broken.gml")
    output_dir = tmp_path / "out"

    exit_code = _run_main(
        [
            "--corpus-dir",
            str(corpus_dir),
            "--output-dir",
            str(output_dir),
            "--engines-file",
            str(_write_engines_file(tmp_path, ["graphviz_dot"])),
            "--workers",
            "1",
            "--score-workers",
            "1",
        ],
        monkeypatch,
    )

    assert exit_code == 0
    payload = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    load_rows = payload["load_rows"]
    assert any(row["status"] == "LOAD_ERROR" and row["graph"] == "rome/broken" for row in load_rows)
    assert any(row["status"] == "OK" for row in payload["rows"])
    report = (output_dir / "GLADOS_RUN_REPORT.md").read_text(encoding="utf-8")
    assert "LOAD_ERROR rome/broken" in report


def test_load_suspect_excluded_from_tally_and_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A trivially-parsed graph is LOAD_SUSPECT: reported, never tallied."""
    corpus_dir = tmp_path / "corpus"
    _copy_fixture(corpus_dir, "rome", "ring6.graph")
    _copy_fixture(corpus_dir, "rome", "collapse1.graph")
    output_dir = tmp_path / "out"

    exit_code = _run_main(
        [
            "--corpus-dir",
            str(corpus_dir),
            "--output-dir",
            str(output_dir),
            "--engines-file",
            str(_write_engines_file(tmp_path, ["graphviz_dot"])),
            "--workers",
            "1",
            "--score-workers",
            "1",
        ],
        monkeypatch,
    )

    assert exit_code == 0
    payload = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    assert payload["tally"]["overall"]["total"] == 1  # collapse1 excluded
    assert all(row["graph"] != "rome/collapse1" for row in payload["rows"])
    suspects = [row for row in payload["load_rows"] if row["status"] == "LOAD_SUSPECT"]
    assert len(suspects) == 1
    report = (output_dir / "GLADOS_RUN_REPORT.md").read_text(encoding="utf-8")
    assert "LOAD_SUSPECT rome/collapse1" in report
    assert any("LOAD_SUSPECT excluded" in item for item in payload["assumptions"])
