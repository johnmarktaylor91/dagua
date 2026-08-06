"""Tests for the GLaDOS holdout runner and blind subset module.

All tests run on synthetic fixtures under ``tests/fixtures/glados/`` --
invented topologies, never derived from any real corpus (holdout opacity,
R-12). No test touches ``eval_output/stdcorpora`` or any fetched content.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch

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


def _current_scoring_signature() -> str:
    """Return the live scoring signature (for resume-sentinel fixtures).

    Returns
    -------
    str
        Current ``native_sprint_score.scoring_signature()``.
    """
    import scripts.native_sprint_score as nss

    return nss.scoring_signature()


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
        Fixture used to guarantee every forbidden env knob is unset.

    Returns
    -------
    int
        Exit code.
    """
    for variable in glados.PREFLIGHT_FORBIDDEN_ENV:
        monkeypatch.delenv(variable, raising=False)
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
    with pytest.raises(ValueError, match="duplicate stems"):
        subset_mod.rank_candidates("s", [("rome", "a.graph"), ("rome", "a.graph")])


def test_rank_candidates_same_stem_different_extension_hard_error() -> None:
    """Same stem across formats collides on the runner's corpus/stem row key.

    Sol WP-25 review HIGH-2: rome/a.graph + rome/a.gml both load as
    ``rome/a`` -- the subset guard must key duplicates on (corpus, stem),
    not (corpus, filename).
    """
    with pytest.raises(ValueError, match="duplicate stems"):
        subset_mod.rank_candidates("s", [("rome", "a.graph"), ("rome", "a.gml")])
    # Same stem in DIFFERENT corpora stays legal.
    ranked = subset_mod.rank_candidates("s", [("rome", "a.graph"), ("north", "a.gml")])
    assert set(ranked) == {"rome", "north"}


def test_subset_cli_stem_collision_aborts_before_writing(tmp_path: Path) -> None:
    """A stem collision in FETCHED_FILES.txt aborts with no SUBSET.json written."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "FETCHED_FILES.txt").write_text(
        f"{corpus_dir.as_posix()}/rome/a.graph\n{corpus_dir.as_posix()}/rome/a.gml\n",
        encoding="utf-8",
    )
    assert subset_mod.main(["--corpus-dir", str(corpus_dir)]) == 1
    assert not (corpus_dir / "SUBSET.json").exists()
    assert not (corpus_dir / "SEALED_REMAINDER.json").exists()


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
# 4b. Process-tree containment (Sol HIGH-1)
# ---------------------------------------------------------------------------


def _tree_kill_probe_worker(pid_file: str, use_setsid: bool) -> None:
    """Spawn-child probe: optionally setsid, spawn a sleeping grandchild.

    Mirrors the row child: ``use_setsid=True`` matches ``_row_layout_worker``
    (killpg path); ``use_setsid=False`` exercises the psutil-snapshot
    fallback for descendants outside the child's process group.

    Parameters
    ----------
    pid_file : str
        Where to publish the grandchild pid.
    use_setsid : bool
        Whether to become a session leader first.

    Returns
    -------
    None
    """
    import subprocess as sp

    if use_setsid:
        try:
            os.setsid()
        except OSError:
            pass
    grandchild = sp.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    Path(pid_file).write_text(str(grandchild.pid), encoding="utf-8")
    time.sleep(300)


def _pid_running(pid: int) -> bool:
    """Return whether a pid is alive and not a reaped-pending zombie.

    Parameters
    ----------
    pid : int
        Process id.

    Returns
    -------
    bool
        ``True`` for a live (non-zombie) process.
    """
    import psutil

    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


@pytest.mark.parametrize("use_setsid", [True, False])
def test_kill_process_tree_kills_sleeping_grandchild(tmp_path: Path, use_setsid: bool) -> None:
    """Memkill/timeout containment kills the WHOLE measured tree (Sol HIGH-1).

    A row child spawns a sleeping grandchild (standing in for an adapter's
    dot/java/node binary); after ``_kill_process_tree`` the grandchild must
    be dead -- via killpg when the child owns its session (the production
    row-child path) and via the psutil descendant snapshot otherwise.
    """
    import multiprocessing as mp

    context = mp.get_context("spawn")
    pid_file = tmp_path / "grandchild.pid"
    process = context.Process(target=_tree_kill_probe_worker, args=(str(pid_file), use_setsid))
    process.start()
    grandchild_pid: int = -1
    try:
        deadline = time.time() + 90.0
        while time.time() < deadline:
            try:
                grandchild_pid = int(pid_file.read_text(encoding="utf-8"))
                break
            except (OSError, ValueError):
                time.sleep(0.2)
        assert grandchild_pid > 0, "probe child never spawned its grandchild"
        assert _pid_running(grandchild_pid)

        glados._kill_process_tree(process)

        assert not process.is_alive()
        deadline = time.time() + 10.0
        while time.time() < deadline and _pid_running(grandchild_pid):
            time.sleep(0.2)
        assert not _pid_running(grandchild_pid), "grandchild survived the tree kill"
    finally:
        if grandchild_pid > 0:
            try:
                os.kill(grandchild_pid, 9)
            except (ProcessLookupError, PermissionError):
                pass
        if process.is_alive():
            process.kill()
            process.join()
        try:
            process.close()
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# 4c. Champion-selection tie stability (Sol MEDIUM-1)
# ---------------------------------------------------------------------------


def _fake_scored_row(
    graph: str, engine: str, seed: Any, v3_tiered: float, corpus: str = "rome"
) -> Dict[str, Any]:
    """Build a minimal scored row for tally-order tests.

    Parameters
    ----------
    graph : str
        Graph name.
    engine : str
        Engine name.
    seed : int | None
        Row seed.
    v3_tiered : float
        Score.
    corpus : str, default="rome"
        Corpus name.

    Returns
    -------
    Dict[str, Any]
        Row dict sufficient for the imported champion selection.
    """
    return {
        "graph": graph,
        "corpus": corpus,
        "engine": engine,
        "seed": seed,
        "record_key": build_record_key(graph, engine, seed),
        "status": "OK",
        "v3_tiered": v3_tiered,
        "positions_path": "positions/fake.pt",
    }


def test_tally_tie_winner_independent_of_completion_order(tmp_path: Path) -> None:
    """Exact-tied field rows produce the same winner in either insertion order.

    Sol WP-25 review MEDIUM-1: field rows complete on concurrent threads and
    the imported selection keeps the first exact tie, so the runner must feed
    a stable (record_key-sorted) order to best_rows_by_graph.
    """
    corpus_dir = tmp_path / "corpus"
    _copy_fixture(corpus_dir, "rome", "ring6.graph")
    entries, _, _ = glados.load_phase(corpus_dir, None, 2000, 200_000)

    native = _fake_scored_row("rome/ring6", "dagua", None, 50.0)
    tied_a = _fake_scored_row("rome/ring6", "aaa_engine", None, 60.0)
    tied_z = _fake_scored_row("rome/ring6", "zzz_engine", None, 60.0)

    outcomes = []
    for rows in ([native, tied_a, tied_z], [native, tied_z, tied_a]):
        tally = glados.compute_tally(entries, rows, tmp_path)
        (detail,) = tally["details"]
        outcomes.append(
            (detail["field_engine"], tuple(sorted(tally["per_engine_field_best"].items())))
        )
    assert outcomes[0] == outcomes[1]
    # Deterministic rule: lexicographically-first record_key wins exact ties.
    assert outcomes[0][0] == "aaa_engine"


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
        # A kept OK row must OWN a present tensor (Sol R3 F2), like every
        # real completed row does.
        "positions_path": "positions/rome__ring6__graphviz_dot.pt",
        "v3_tiered": 55.5,  # pre-scored: scorer must skip it too
        # A scored row without the CURRENT scoring signature is quarantined
        # on resume (dry-well B4-F3 / Sol B3-2); a preserved sentinel must
        # therefore carry it, like every real scored row does.
        "scoring_signature": _current_scoring_signature(),
        "nodes": 6,
        "edges": 6,
        "directed": False,
        "directed_source": "policy",
        "source_path": str(corpus_dir / "rome" / "ring6.graph"),
        "error": None,
    }
    sentinel_tensor = staging / "positions" / "rome__ring6__graphviz_dot.pt"
    sentinel_tensor.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.zeros((6, 2)), sentinel_tensor)
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
            "300",  # generous: 120s native rows flaked under load (2026-08-05 Sol review)
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
                "300",  # generous: load-robust (2026-08-05 Sol-review incident)
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


# ---------------------------------------------------------------------------
# 8. Dry-well Round-1 fix round (B4-F1..F8, Sol B3-2/B4-3/B5-1, B5-F01)
# ---------------------------------------------------------------------------


def _fake_published_run(directory: Path) -> Path:
    """Create a minimal fake published run directory.

    Parameters
    ----------
    directory : Path
        Where to create the run.

    Returns
    -------
    Path
        The run directory.
    """
    (directory / "positions").mkdir(parents=True, exist_ok=True)
    (directory / "results.json").write_text("{}", encoding="utf-8")
    (directory / "positions" / "g__e__deterministic.pt").write_bytes(b"tensor")
    return directory


@pytest.mark.parametrize("variable", sorted(glados.PREFLIGHT_FORBIDDEN_ENV))
def test_preflight_native_env_rejects_every_forbidden_var(
    variable: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B5-F01: every behavior-changing env knob fails preflight when set."""
    for name in glados.PREFLIGHT_FORBIDDEN_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(variable, "1")
    with pytest.raises(glados.PreflightError, match=variable):
        glados._preflight_native_env()
    monkeypatch.delenv(variable)
    glados._preflight_native_env()  # clean env passes


def test_archive_refuses_overlapping_destination(tmp_path: Path) -> None:
    """B4-F1 (CRITICAL): archive must never rmtree/copy onto the published run.

    The destination leaf is hardcoded ``glados_holdout``; with an output dir
    of the same name, ``--archive-dir <parent-of-output>`` used to resolve
    destination == output dir and DELETE the published run.
    """
    out = _fake_published_run(tmp_path / "glados_holdout")

    # destination == output dir (the reproduced one-keystroke disaster).
    warning = glados.archive_run(out, tmp_path)
    assert warning is not None and "REFUSING" in warning
    assert (out / "results.json").is_file()
    assert (out / "positions" / "g__e__deterministic.pt").is_file()

    # destination inside the output dir.
    warning = glados.archive_run(out, out)
    assert warning is not None and "REFUSING" in warning
    assert (out / "results.json").is_file()

    # destination is an ancestor of the output dir.
    nested_out = _fake_published_run(tmp_path / "arch" / "glados_holdout" / "run")
    warning = glados.archive_run(nested_out, tmp_path / "arch")
    assert warning is not None and "REFUSING" in warning
    assert (nested_out / "results.json").is_file()


def test_archive_failed_copy_preserves_previous_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B4-F1: a failed copy never destroys the previous archive, and the
    warning only claims the run is intact after verifying it."""
    out = _fake_published_run(tmp_path / "out")
    archive_dir = tmp_path / "arch"
    previous = archive_dir / "glados_holdout"
    previous.mkdir(parents=True)
    (previous / "MARKER.txt").write_text("previous archive", encoding="utf-8")

    def boom(*args: Any, **kwargs: Any) -> None:
        raise OSError("injected copy failure")

    monkeypatch.setattr(glados.shutil, "copytree", boom)
    warning = glados.archive_run(out, archive_dir)

    assert warning is not None
    assert "published run verified intact" in warning
    assert (previous / "MARKER.txt").is_file(), "previous archive was destroyed"
    assert (out / "results.json").is_file()


def test_archive_success_swaps_and_keeps_manifest(tmp_path: Path) -> None:
    """B4-F1: the temp-then-rename swap replaces a previous archive cleanly."""
    out = _fake_published_run(tmp_path / "out")
    archive_dir = tmp_path / "arch"
    previous = archive_dir / "glados_holdout"
    previous.mkdir(parents=True)
    (previous / "MARKER.txt").write_text("old", encoding="utf-8")

    assert glados.archive_run(out, archive_dir) is None
    destination = archive_dir / "glados_holdout"
    assert (destination / "results.json").is_file()
    assert (destination / "MANIFEST.sha256").is_file()
    assert not (destination / "MARKER.txt").exists()  # old archive replaced
    assert not (archive_dir / ".glados_holdout.prev").exists()


def test_partition_resumed_rows_quarantine_rules() -> None:
    """B4-F3 + Sol B3-2/B4-3/B5-1: the three quarantine rules, unit-level."""
    signature = "current-sig"
    valid_keys = {
        build_record_key("rome/g", "dagua", None),
        build_record_key("rome/g", "graphviz_dot", None),
    }
    kept_layout = {  # layout-only row, in universe: kept, scored fresh later
        "record_key": build_record_key("rome/g", "graphviz_dot", None),
        "engine": "graphviz_dot",
        "status": "OK",
        "positions_path": "positions/rome__g__graphviz_dot.pt",
    }
    kept_scored_native = {  # scored native row, current sig + current seed
        "record_key": build_record_key("rome/g", "dagua", None),
        "engine": "dagua",
        "status": "OK",
        "v3_tiered": 50.0,
        "scoring_signature": signature,
        "native_child_seed": 42,
        "positions_path": "positions/rome__g__dagua.pt",
    }
    stale_sig = {  # scored under an older ruler -> rescore (Sol B3-2)
        "record_key": build_record_key("rome/g", "graphviz_dot", None),
        "engine": "graphviz_dot",
        "status": "OK",
        "v3_tiered": 61.0,
        "scoring_signature": "older-sig",
    }
    ghost_engine = {  # outside current field -> never a champion (Sol B4-3)
        "record_key": build_record_key("rome/g", "zzz_ghost", None),
        "engine": "zzz_ghost",
        "status": "OK",
        "v3_tiered": 99.9,
        "scoring_signature": signature,
    }
    wrong_native_seed = {  # native row from a --seed 41 partial (Sol B5-1)
        "record_key": build_record_key("rome/g", "dagua", None),
        "engine": "dagua",
        "status": "OK",
        "native_child_seed": 41,
    }
    legacy_native = {  # predates native_child_seed: quarantined conservatively
        "record_key": build_record_key("rome/g", "dagua", None),
        "engine": "dagua",
        "status": "OK",
    }

    tensorless = {  # OK row whose tensor vanished (e.g. orphan-swept while
        # out-of-universe in an earlier resume) -> rerun from scratch
        "record_key": build_record_key("rome/g", "graphviz_dot", None),
        "engine": "graphviz_dot",
        "status": "OK",
        "v3_tiered": 58.0,
        "scoring_signature": signature,
        "positions_path": "positions/gone.pt",
    }

    kept, quarantined, counts = glados.partition_resumed_rows(
        [
            kept_layout,
            kept_scored_native,
            stale_sig,
            ghost_engine,
            wrong_native_seed,
            legacy_native,
            tensorless,
        ],
        signature,
        valid_keys,
        42,
        tensor_exists=lambda relpath: relpath != "positions/gone.pt",
    )

    assert kept == [kept_layout, kept_scored_native]
    reasons = [row["quarantine_reason"] for row in quarantined]
    assert reasons == [
        "stale scoring signature",
        "outside current subset/field/seed battery",
        "native row generated under a different --seed",
        "native row generated under a different --seed",
        "positions tensor missing from the row store",
    ]
    assert counts == {"signature": 1, "universe": 1, "native_seed": 2, "tensor_missing": 1}


def test_resume_quarantine_integration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stale resumed rows are quarantined loudly and never select the champion."""
    corpus_dir = tmp_path / "corpus"
    _copy_fixture(corpus_dir, "rome", "ring6.graph")
    output_dir = tmp_path / "out"
    staging = output_dir.with_name(f"{output_dir.name}.tmp")

    stale_sig_row: Dict[str, Any] = {
        "graph": "rome/ring6",
        "corpus": "rome",
        "engine": "graphviz_dot",
        "seed": None,
        "record_key": build_record_key("rome/ring6", "graphviz_dot", None),
        "status": "OK",
        "runtime_s": -1.0,
        "positions_path": None,
        "v3_tiered": 61.0,
        "scoring_signature": "stale-sig-from-before-the-hotfix",
    }
    ghost_row: Dict[str, Any] = {
        "graph": "rome/ring6",
        "corpus": "rome",
        "engine": "zzz_ghost",
        "seed": None,
        "record_key": build_record_key("rome/ring6", "zzz_ghost", None),
        "status": "OK",
        "runtime_s": -1.0,
        "positions_path": None,
        "v3_tiered": 99.9,  # would win field-best if it ever competed
        "scoring_signature": _current_scoring_signature(),
    }
    glados.append_row(staging, stale_sig_row)
    glados.append_row(staging, ghost_row)

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
    quarantined = payload["quarantined_rows"]
    assert {row["quarantine_reason"] for row in quarantined} == {
        "stale scoring signature",
        "outside current subset/field/seed battery",
    }
    assert len(quarantined) == 2
    assert any(
        warning.startswith("QUARANTINE: 2 stale resumed row(s)") for warning in payload["warnings"]
    )
    # The ghost engine never competes: the fresh in-field row is champion.
    rows = {row["record_key"]: row for row in payload["rows"]}
    assert build_record_key("rome/ring6", "zzz_ghost", None) not in rows
    (detail,) = payload["tally"]["details"]
    assert detail["field_engine"] == "graphviz_dot"
    assert detail["field_v3_tiered"] != 99.9
    # The stale-signature row's key was re-run and re-scored fresh.
    fresh = rows[build_record_key("rome/ring6", "graphviz_dot", None)]
    assert fresh["status"] == "OK"
    assert fresh["scoring_signature"] == payload["scoring_signature"]
    assert fresh["runtime_s"] != -1.0
    report = (output_dir / "GLADOS_RUN_REPORT.md").read_text(encoding="utf-8")
    assert "Quarantined stale resume rows: 2" in report


def test_field_worker_records_harness_error_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B4-F4: a worker exception becomes an ERROR row, never a vanished row."""
    corpus_dir = tmp_path / "corpus"
    _copy_fixture(corpus_dir, "rome", "ring6.graph")
    output_dir = tmp_path / "out"

    def boom(self: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise OSError("injected ENOSPC")

    monkeypatch.setattr(glados.RowExecutor, "run_row", boom)
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
    (row,) = payload["rows"]
    assert row["status"] == "ERROR"
    assert row["status_detail"] == "harness:OSError"
    assert "injected ENOSPC" in row["error"]


def test_load_child_message_tolerates_torn_files(tmp_path: Path) -> None:
    """B4-F4: torn/empty child handshake files parse to None, not a crash."""
    torn = tmp_path / "torn.json"
    torn.write_text('{"status": "OK", "runt', encoding="utf-8")
    assert glados._load_child_message(torn) is None
    empty = tmp_path / "empty.json"
    empty.write_bytes(b"")
    assert glados._load_child_message(empty) is None
    missing = tmp_path / "missing.json"
    assert glados._load_child_message(missing) is None
    nondict = tmp_path / "nondict.json"
    nondict.write_text("[1, 2]", encoding="utf-8")
    assert glados._load_child_message(nondict) is None
    good = tmp_path / "good.json"
    good.write_text('{"status": "OK"}', encoding="utf-8")
    assert glados._load_child_message(good) == {"status": "OK"}


def test_system_floor_memkill_during_poll(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """B4-F5: the system floor is enforced DURING flight, not just at dispatch."""
    corpus_dir = tmp_path / "corpus"
    _copy_fixture(corpus_dir, "rome", "ring6.graph")
    output_dir = tmp_path / "out"

    calls: List[int] = []

    def fake_available() -> int:
        calls.append(1)
        # First call = the dispatch-time floor check (plenty); every later
        # call = the in-poll re-check (below the floor).
        return 1024**4 if len(calls) == 1 else 1 * 1024**3

    monkeypatch.setattr(glados, "system_available_bytes", fake_available)
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
    (row,) = payload["rows"]
    assert row["status"] == "ERROR"
    assert row["status_detail"] == "memkill:system-floor"
    assert "fell below" in row["error"]
    report = (output_dir / "GLADOS_RUN_REPORT.md").read_text(encoding="utf-8")
    assert "memkills 1" in report


def test_fresh_run_refuses_existing_data_without_force_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B4-F6: a non-resume run never silently destroys prior run data."""
    corpus_dir = tmp_path / "corpus"
    _copy_fixture(corpus_dir, "rome", "ring6.graph")
    output_dir = tmp_path / "out"
    staging = output_dir.with_name(f"{output_dir.name}.tmp")
    glados.append_row(staging, {"record_key": "precious", "status": "OK"})

    argv = [
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
    ]
    assert _run_main(argv, monkeypatch) == 4
    rows, _ = glados.load_rows_tolerant(staging)
    assert rows and rows[0]["record_key"] == "precious", "staging was destroyed"

    # A published run in the output dir is protected the same way.
    output_dir2 = tmp_path / "out2"
    output_dir2.mkdir()
    (output_dir2 / "results.json").write_text("{}", encoding="utf-8")
    argv2 = list(argv)
    argv2[argv2.index(str(output_dir))] = str(output_dir2)
    assert _run_main(argv2, monkeypatch) == 4

    # --force-fresh explicitly authorizes the destruction.
    assert _run_main([*argv, "--force-fresh"], monkeypatch) == 0
    payload = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    assert all(row["record_key"] != "precious" for row in payload["rows"])


def test_edge_list_node_id_cap_prevents_wedge(tmp_path: Path) -> None:
    """B4-F7: a 2-line poison file must fail fast, not wedge/OOM the parent."""
    from scripts.stdcorpora_loaders import MAX_EDGE_LIST_NODES, load_graph_file

    poison = tmp_path / "corpus" / "rome" / "poison.graph"
    poison.parent.mkdir(parents=True)
    # The EXACT reproduced wedge shape (dry-well B4-F7 measured 121.8s and
    # ~935MB at 50,000 declared ids): must now be REFUSED fast, proving the
    # cap sits below the measured failure, not just below absurdity
    # (Sol round-2 F4).
    poison.write_text("5000 1\n1 50000\n", encoding="utf-8")

    started = time.time()
    with pytest.raises(ValueError, match="refusing pre-allocation"):
        load_graph_file(poison, directed_override=False)
    assert time.time() - started < 5.0, "cap must reject before any allocation"
    assert MAX_EDGE_LIST_NODES == 25_000

    _copy_fixture(tmp_path / "corpus", "rome", "ring6.graph")
    entries, load_rows, _ = glados.load_phase(tmp_path / "corpus", None, 2000, 200_000)
    assert [entry.name for entry in entries] == ["rome/ring6"]
    (error_row,) = [row for row in load_rows if row["status"] == "LOAD_ERROR"]
    assert error_row["graph"] == "rome/poison"
    assert "refusing pre-allocation" in error_row["error"]


def test_mtx_declared_dims_cap_pre_guard(tmp_path: Path) -> None:
    """B4-F7 companion: absurd declared .mtx dims are rejected from the header."""
    corpus_dir = tmp_path / "corpus"
    big = corpus_dir / "suitesparse" / "huge.mtx"
    big.parent.mkdir(parents=True)
    big.write_text(
        # 50k = the measured parent-wedge scale (Sol round-2 F4), not just
        # an absurd declaration.
        "%%MatrixMarket matrix coordinate pattern general\n50000 50000 1\n1 2\n",
        encoding="utf-8",
    )
    _copy_fixture(corpus_dir, "suitesparse", "tri4.mtx")

    started = time.time()
    entries, load_rows, _ = glados.load_phase(corpus_dir, None, 2000, 200_000)
    assert time.time() - started < 5.0, "pre-guard must reject without loading"
    assert [entry.name for entry in entries] == ["suitesparse/tri4"]
    (skip_row,) = load_rows
    assert skip_row["status"] == "SKIP"
    assert skip_row["reason"].startswith("mtx_dims_cap:50000x50000")


# ---------------------------------------------------------------------------
# 9. Dry-well fix-round 3 (Sol round-2 F1-F4: structural publish/archive/resume)
# ---------------------------------------------------------------------------


def test_archive_refuses_prev_and_temp_aliases(tmp_path: Path) -> None:
    """Sol round-2 F1: the hidden swap paths must pass the overlap predicate.

    Sol's probes aliased the published run to ``.glados_holdout.prev`` and
    ``.glados_holdout.tmp-<pid>`` and deleted it through the swap's own
    bookkeeping paths. Both shapes are refusals now.
    """
    archive_dir = tmp_path / "arch"
    archive_dir.mkdir()

    prev_alias = _fake_published_run(archive_dir / ".glados_holdout.prev")
    warning = glados.archive_run(prev_alias, archive_dir)
    assert warning is not None and "REFUSING" in warning
    assert (prev_alias / "results.json").is_file(), "run behind the .prev alias was deleted"

    temp_alias = _fake_published_run(archive_dir / f".glados_holdout.tmp-{os.getpid()}")
    warning = glados.archive_run(temp_alias, archive_dir)
    assert warning is not None and "REFUSING" in warning
    assert (temp_alias / "results.json").is_file(), "run behind the temp alias was deleted"


def test_publish_validates_staging_before_touching_prior_run(tmp_path: Path) -> None:
    """Structural reorder: staging validates FIRST; the prior run is untouched.

    Sol round-2 F3: the old shape swapped, deleted the prior run, and only
    then validated. Now a staging payload with an OK row referencing a
    missing tensor raises BEFORE any rename, with the prior run intact and
    staging preserved.
    """
    output_dir = tmp_path / "out"
    _fake_published_run(output_dir)
    (output_dir / "MARKER.txt").write_text("prior run", encoding="utf-8")
    staging = tmp_path / "out.tmp"
    (staging / "positions").mkdir(parents=True)
    payload = {
        "rows": [
            {
                "record_key": "g::e::deterministic",
                "status": "OK",
                "positions_path": "positions/missing.pt",
            }
        ]
    }

    with pytest.raises(RuntimeError, match="missing position files"):
        glados.publish_results(output_dir, staging, payload)

    assert (output_dir / "MARKER.txt").is_file(), "prior run was touched"
    assert (output_dir / "results.json").is_file()
    assert staging.is_dir(), "staging must be preserved for inspection/resume"
    assert not output_dir.with_name("out.prev").exists()


def test_publish_restores_prior_run_when_post_swap_validation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The prior run is restored if the installed output fails re-validation."""
    output_dir = tmp_path / "out"
    _fake_published_run(output_dir)
    (output_dir / "MARKER.txt").write_text("prior run", encoding="utf-8")
    staging = tmp_path / "out.tmp"
    (staging / "positions").mkdir(parents=True)
    payload: Dict[str, Any] = {"rows": []}

    real_validate = glados.validate_store

    def validate_only_staging(directory: Path, inner_payload: Dict[str, Any]) -> None:
        if directory == output_dir:
            raise RuntimeError("injected post-swap validation failure")
        real_validate(directory, inner_payload)

    monkeypatch.setattr(glados, "validate_store", validate_only_staging)
    with pytest.raises(RuntimeError, match="injected post-swap"):
        glados.publish_results(output_dir, staging, payload)

    assert (output_dir / "MARKER.txt").is_file(), "prior run was not restored"
    failed = output_dir.with_name("out.failed-publish")
    assert failed.is_dir(), "failed candidate must be kept for inspection"
    assert (failed / "results.json").is_file()


def test_publish_failure_exits_1_and_preserves_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() exits 1 on publish validation failure; staging survives."""
    corpus_dir = tmp_path / "corpus"
    _copy_fixture(corpus_dir, "rome", "ring6.graph")
    output_dir = tmp_path / "out"
    staging = output_dir.with_name(f"{output_dir.name}.tmp")

    def always_fail(directory: Path, payload: Dict[str, Any]) -> None:
        raise RuntimeError("injected staging validation failure")

    monkeypatch.setattr(glados, "validate_store", always_fail)
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

    assert exit_code == 1
    assert not (output_dir / "results.json").exists(), "nothing must be published"
    assert glados.rows_path(staging).is_file(), "staging must survive for --resume"


def test_rewrite_rows_replaces_store_atomically(tmp_path: Path) -> None:
    """rewrite_rows leaves exactly the kept rows, readable by the loader."""
    directory = tmp_path / "staging"
    glados.append_row(directory, {"record_key": "a", "status": "OK"})
    glados.append_row(directory, {"record_key": "b", "status": "OK"})
    glados.rewrite_rows(directory, [{"record_key": "b", "status": "OK"}])
    rows, warnings = glados.load_rows_tolerant(directory)
    assert [row["record_key"] for row in rows] == ["b"]
    assert warnings == []


def test_resume_field_removal_then_readd_reruns_from_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sol round-2 F2's reproduced detonation sequence, end to end.

    Run 1 scores graphviz_dot; run 2 resumes WITHOUT that engine (its rows
    quarantine out-of-universe, the store is rewritten, its tensor is
    orphan-swept); run 3 re-adds the engine under the SAME signature. The
    old shape resurrected the run-1 scored row pointing at the swept tensor
    and detonated post-swap validation; now the key vacates and the row
    reruns from scratch.
    """
    corpus_dir = tmp_path / "corpus"
    _copy_fixture(corpus_dir, "rome", "ring6.graph")
    output_dir = tmp_path / "out"
    dot_key = build_record_key("rome/ring6", "graphviz_dot", None)

    def run(engines: List[str], resume: bool) -> Dict[str, Any]:
        argv = [
            "--corpus-dir",
            str(corpus_dir),
            "--output-dir",
            str(output_dir),
            "--engines-file",
            str(_write_engines_file(tmp_path, engines)),
            "--workers",
            "1",
            "--score-workers",
            "1",
            "--seeds",
            "1",
        ]
        if resume:
            argv.append("--resume")
        assert _run_main(argv, monkeypatch) == 0
        return json.loads((output_dir / "results.json").read_text(encoding="utf-8"))

    payload1 = run(["graphviz_dot"], resume=False)
    rows1 = {row["record_key"]: row for row in payload1["rows"]}
    assert rows1[dot_key]["status"] == "OK"
    tensor1_relpath = rows1[dot_key]["positions_path"]
    assert (output_dir / tensor1_relpath).is_file()

    # Run 2: remove the field engine. Its rows quarantine, the store is
    # rewritten without them, and the tensor is orphan-swept.
    payload2 = run(["classic_kk"], resume=True)
    assert all(row["engine"] != "graphviz_dot" for row in payload2["rows"])
    assert any(
        row["engine"] == "graphviz_dot"
        and row["quarantine_reason"] == "outside current subset/field/seed battery"
        for row in payload2["quarantined_rows"]
    )
    store_rows, _ = glados.load_rows_tolerant(output_dir)
    assert all(row.get("record_key") != dot_key for row in store_rows), (
        "quarantined rows must be REMOVED from the primary store"
    )
    assert not (output_dir / tensor1_relpath).is_file()

    # Run 3: re-add the engine under the same signature -> full rerun, green
    # publish (previously: resurrection + post-swap detonation).
    payload3 = run(["graphviz_dot"], resume=True)
    rows3 = {row["record_key"]: row for row in payload3["rows"]}
    fresh = rows3[dot_key]
    assert fresh["status"] == "OK"
    assert fresh["v3_tiered"] is not None
    assert fresh["scoring_signature"] == payload3["scoring_signature"]
    assert (output_dir / fresh["positions_path"]).is_file()


def test_resume_preparation_failure_leaves_canonical_run_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sol round-3 F1: resume must never displace the published run.

    Resume re-opens a published partial run COPY-based; a failure in
    fallible resume preparation (here: the quarantine rewrite) must leave
    the canonical output directory exactly as published.
    """
    corpus_dir = tmp_path / "corpus"
    _copy_fixture(corpus_dir, "rome", "ring6.graph")
    output_dir = tmp_path / "out"
    base_argv = [
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
        "--seeds",
        "1",
    ]
    assert _run_main(base_argv, monkeypatch) == 0
    published = (output_dir / "results.json").read_bytes()

    def boom(*args: Any, **kwargs: Any) -> None:
        raise OSError("injected rewrite failure (Sol R3 F1)")

    monkeypatch.setattr(glados, "rewrite_rows", boom)
    # Force the quarantine path so rewrite_rows is reached: resume with the
    # engine removed from the field.
    resume_argv = [
        "--corpus-dir",
        str(corpus_dir),
        "--output-dir",
        str(output_dir),
        "--engines-file",
        str(_write_engines_file(tmp_path, ["classic_kk"])),
        "--workers",
        "1",
        "--score-workers",
        "1",
        "--seeds",
        "1",
        "--resume",
    ]
    exit_code = _run_main(resume_argv, monkeypatch)
    assert exit_code == 1
    assert (output_dir / "results.json").read_bytes() == published, (
        "canonical published run must be byte-untouched by a failed resume"
    )
    rows, torn = glados.load_rows_tolerant(output_dir)
    assert rows and not torn


def test_partition_quarantines_ok_row_with_null_positions_path() -> None:
    """Sol round-3 F2: an OK row with a null/absent positions_path is as
    unpublishable as one whose tensor was swept."""
    row = {
        "record_key": "rome/ring6|graphviz_dot|0",
        "engine": "graphviz_dot",
        "status": "OK",
        "scoring_signature": "sig",
        "positions_path": None,
    }
    kept, quarantined, counts = glados.partition_resumed_rows(
        [row],
        signature="sig",
        valid_keys={"rome/ring6|graphviz_dot|0"},
        current_seed=42,
        tensor_exists=lambda _p: True,
    )
    assert not kept
    assert quarantined[0]["quarantine_reason"] == ("positions tensor missing from the row store")
    assert counts["tensor_missing"] == 1
