"""Tests for the r79 standard-corpora evaluation harness."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

import scripts.r79_stdcorpora_eval as stdcorpora_eval
from scripts.r79_stdcorpora_eval import (
    append_row,
    load_gml_file,
    load_graph_file,
    load_graphml_file,
    load_jsonl_rows,
    load_mtx_file,
    rows_path,
    staging_dir,
)


def edge_count(path_graph: object) -> int:
    """Return the finalized edge count for a loaded graph wrapper.

    Parameters
    ----------
    path_graph : object
        ``LoadedGraph`` returned by the harness loaders.

    Returns
    -------
    int
        Number of edges in the underlying ``DaguaGraph``.
    """
    graph = getattr(path_graph, "graph")
    return int(graph.edge_index.shape[1])


def test_graph_loader_reads_adjacency_fixture(tmp_path: Path) -> None:
    """Load a tiny Rome-style adjacency-list ``.graph`` fixture.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.

    Returns
    -------
    None
    """
    path = tmp_path / "rome" / "tiny.graph"
    path.parent.mkdir()
    path.write_text("3\n2 3\n1 3\n1 2\n", encoding="utf-8")

    loaded = load_graph_file(path)

    assert loaded.graph.num_nodes == 3
    assert edge_count(loaded) == 3
    assert loaded.corpus == "rome"
    assert loaded.directed is False


def test_gml_loader_reads_networkx_fixture(tmp_path: Path) -> None:
    """Load a tiny directed North-style GML fixture.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.

    Returns
    -------
    None
    """
    path = tmp_path / "north" / "tiny.gml"
    path.parent.mkdir()
    path.write_text(
        "graph [\n"
        "  directed 1\n"
        '  node [ id 0 label "a" ]\n'
        '  node [ id 1 label "b" ]\n'
        '  node [ id 2 label "c" ]\n'
        "  edge [ source 0 target 1 ]\n"
        "  edge [ source 1 target 2 ]\n"
        "]\n",
        encoding="utf-8",
    )

    loaded = load_gml_file(path)

    assert loaded.graph.num_nodes == 3
    assert edge_count(loaded) == 2
    assert loaded.corpus == "north"
    assert loaded.directed is True


def test_graphml_loader_reads_rome_style_fixture(tmp_path: Path) -> None:
    """Load a tiny undirected Rome-style GraphML fixture (Y-Files export shape).

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.

    Returns
    -------
    None
    """
    path = tmp_path / "rome" / "tiny.graphml"
    path.parent.mkdir()
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<graphml>\n"
        '<graph edgedefault="undirected" id="G">\n'
        '<node id="n0"/>\n'
        '<node id="n1"/>\n'
        '<node id="n2"/>\n'
        '<edge id="e0" source="n0" target="n1"/>\n'
        '<edge id="e1" source="n1" target="n2"/>\n'
        "</graph>\n"
        "</graphml>\n",
        encoding="utf-8",
    )

    loaded = load_graphml_file(path)

    assert loaded.graph.num_nodes == 3
    assert edge_count(loaded) == 2
    assert loaded.corpus == "rome"
    assert loaded.directed is False


def test_graphml_loader_infers_north_directed_when_edgedefault_missing(tmp_path: Path) -> None:
    """North GraphML exports omit ``edgedefault``; the loader must still mark them directed.

    Real North/AT&T GraphML exports from graphdrawing.unipg.it omit the ``edgedefault``
    attribute entirely, which makes NetworkX's ``is_directed()`` report ``False``. The
    loader must fall back to the path-based ``north`` heuristic instead of trusting that
    default, or every North DAG would silently score as undirected.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.

    Returns
    -------
    None
    """
    path = tmp_path / "north" / "tiny.graphml"
    path.parent.mkdir()
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<graphml>\n"
        '<graph id="G">\n'
        '<node id="n0"/>\n'
        '<node id="n1"/>\n'
        '<edge id="e0" source="n0" target="n1"/>\n'
        "</graph>\n"
        "</graphml>\n",
        encoding="utf-8",
    )

    loaded = load_graphml_file(path)

    assert loaded.graph.num_nodes == 2
    assert edge_count(loaded) == 1
    assert loaded.corpus == "north"
    assert loaded.directed is True


def test_mtx_loader_reads_coordinate_fixture(tmp_path: Path) -> None:
    """Load a tiny SuiteSparse-style Matrix Market fixture.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.

    Returns
    -------
    None
    """
    path = tmp_path / "suitesparse" / "tiny.mtx"
    path.parent.mkdir()
    path.write_text(
        "%%MatrixMarket matrix coordinate pattern symmetric\n3 3 2\n1 2\n2 3\n",
        encoding="utf-8",
    )

    loaded = load_mtx_file(path)

    assert loaded.graph.num_nodes == 3
    assert edge_count(loaded) == 2
    assert loaded.corpus == "suitesparse"
    assert loaded.directed is False


def test_harness_runs_on_synthetic_mini_corpus(tmp_path: Path) -> None:
    """Run the CLI end-to-end on one fixture per supported format.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.

    Returns
    -------
    None
    """
    corpus_dir = tmp_path / "corpus"
    output_dir = tmp_path / "out"
    (corpus_dir / "rome").mkdir(parents=True)
    (corpus_dir / "north").mkdir()
    (corpus_dir / "suitesparse").mkdir()
    (corpus_dir / "rome" / "tiny.graph").write_text("3\n2\n1 3\n2\n", encoding="utf-8")
    (corpus_dir / "north" / "tiny.gml").write_text(
        "graph [\n"
        "  directed 1\n"
        '  node [ id 0 label "a" ]\n'
        '  node [ id 1 label "b" ]\n'
        "  edge [ source 0 target 1 ]\n"
        "]\n",
        encoding="utf-8",
    )
    (corpus_dir / "suitesparse" / "tiny.mtx").write_text(
        "%%MatrixMarket matrix coordinate pattern general\n2 2 1\n1 2\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/r79_stdcorpora_eval.py",
            "--corpus-dir",
            str(corpus_dir),
            "--output-dir",
            str(output_dir),
            "--engines",
            "dagua",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload: Dict[str, object] = json.loads((output_dir / "results.json").read_text())
    rows = payload["rows"]

    assert "OK" in result.stdout
    assert payload["graph_count"] == 3
    assert len(rows) == 3
    assert all(row["engine"] == "dagua" for row in rows)  # type: ignore[index]
    assert all(row["status"] == "OK" for row in rows)  # type: ignore[index]
    assert (output_dir / "STDCORPORA.md").is_file()
    assert all(
        (output_dir / row["positions_path"]).is_file()
        for row in rows  # type: ignore[index]
    )


def _write_two_graph_corpus(corpus_dir: Path) -> None:
    """Write a tiny two-graph Rome-only corpus fixture.

    Parameters
    ----------
    corpus_dir : Path
        Directory to populate with ``rome/`` fixture files.

    Returns
    -------
    None
    """
    (corpus_dir / "rome").mkdir(parents=True)
    (corpus_dir / "rome" / "alpha.graph").write_text("3\n2 3\n1 3\n1 2\n", encoding="utf-8")
    (corpus_dir / "rome" / "beta.graph").write_text("3\n2\n1 3\n2\n", encoding="utf-8")


def test_jsonl_row_helpers_roundtrip(tmp_path: Path) -> None:
    """``append_row`` and ``load_jsonl_rows`` round-trip completed rows.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.

    Returns
    -------
    None
    """
    output_dir = tmp_path / "staging"
    assert load_jsonl_rows(output_dir) == []

    row_a: Dict[str, Any] = {"graph": "rome/alpha", "engine": "dagua", "status": "OK"}
    row_b: Dict[str, Any] = {"graph": "rome/beta", "engine": "dagua", "status": "OK"}
    append_row(output_dir, row_a)
    append_row(output_dir, row_b)

    loaded = load_jsonl_rows(output_dir)
    assert loaded == [row_a, row_b]
    assert rows_path(output_dir).is_file()
    # Appends are flushed per call, not batched -- the file must contain
    # exactly one JSON object per line for --resume to be able to read a
    # partially written run left behind by a crash.
    lines = rows_path(output_dir).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        json.loads(line)  # each line parses standalone


def test_resume_skips_completed_rows_from_jsonl(tmp_path: Path) -> None:
    """``--resume`` must not recompute rows already present in the staging jsonl.

    Simulates the real OOM-crash recovery path: a prior invocation left a
    ``<output>.tmp/results.rows.jsonl`` with one completed row (here a
    synthetic sentinel row, standing in for a real completed result) and
    was killed before finishing the second graph. Rerunning with
    ``--resume`` must skip the sentinel row untouched and only compute the
    remaining graph.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.

    Returns
    -------
    None
    """
    corpus_dir = tmp_path / "corpus"
    output_dir = tmp_path / "out"
    _write_two_graph_corpus(corpus_dir)

    staging = staging_dir(output_dir)
    sentinel_row: Dict[str, Any] = {
        "graph": "rome/alpha",
        "corpus": "rome",
        "engine": "dagua",
        "status": "OK",
        "runtime_s": 0.0,
        "metrics": {},
        "reported_metrics": {},
        "composite": -12345.0,  # sentinel: proves this row was not recomputed
        "positions_path": None,
        "nodes": 3,
        "edges": 3,
        "directed": False,
        "source_path": str(corpus_dir / "rome" / "alpha.graph"),
        "error": None,
    }
    append_row(staging, sentinel_row)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/r79_stdcorpora_eval.py",
            "--corpus-dir",
            str(corpus_dir),
            "--output-dir",
            str(output_dir),
            "--engines",
            "dagua",
            "--resume",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    # The sentinel graph must never be re-run: no fresh OK/ERROR/SKIP line
    # for it should appear in stdout, only for the still-pending graph.
    assert "rome/alpha" not in result.stdout
    assert "rome/beta" in result.stdout

    payload: Dict[str, Any] = json.loads((output_dir / "results.json").read_text())
    rows: List[Dict[str, Any]] = payload["rows"]  # type: ignore[assignment]
    assert len(rows) == 2
    by_graph = {row["graph"]: row for row in rows}
    assert by_graph["rome/alpha"]["composite"] == -12345.0
    assert by_graph["rome/beta"]["status"] == "OK"
    assert by_graph["rome/beta"]["composite"] != -12345.0


def test_corpus_flag_filters_to_one_corpus(tmp_path: Path) -> None:
    """``--corpus rome`` must exclude north/suitesparse fixtures from the run.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.

    Returns
    -------
    None
    """
    corpus_dir = tmp_path / "corpus"
    output_dir = tmp_path / "out"
    (corpus_dir / "rome").mkdir(parents=True)
    (corpus_dir / "north").mkdir()
    (corpus_dir / "rome" / "tiny.graph").write_text("3\n2\n1 3\n2\n", encoding="utf-8")
    (corpus_dir / "north" / "tiny.gml").write_text(
        "graph [\n"
        "  directed 1\n"
        '  node [ id 0 label "a" ]\n'
        '  node [ id 1 label "b" ]\n'
        "  edge [ source 0 target 1 ]\n"
        "]\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/r79_stdcorpora_eval.py",
            "--corpus-dir",
            str(corpus_dir),
            "--output-dir",
            str(output_dir),
            "--engines",
            "dagua",
            "--corpus",
            "rome",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload: Dict[str, Any] = json.loads((output_dir / "results.json").read_text())
    rows: List[Dict[str, Any]] = payload["rows"]  # type: ignore[assignment]
    assert payload["graph_count"] == 1
    assert len(rows) == 1
    assert all(row["corpus"] == "rome" for row in rows)


def test_rss_abort_guard_publishes_partial_results_and_creates_output_dir_early(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The RSS ceiling must stop the run early, publish partial results, and exit 3.

    Also exercises that ``main()`` creates ``--output-dir`` immediately at
    startup, independent of whether the run finishes or aborts.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture.

    Returns
    -------
    None
    """
    corpus_dir = tmp_path / "corpus"
    output_dir = tmp_path / "out"
    _write_two_graph_corpus(corpus_dir)

    # Force the guard to evaluate on every row and immediately trip the
    # abort ceiling so the test runs fast and deterministically.
    monkeypatch.setattr(stdcorpora_eval, "GC_TRIM_INTERVAL_ROWS", 1)
    monkeypatch.setattr(
        stdcorpora_eval, "current_rss_bytes", lambda: stdcorpora_eval.RSS_ABORT_BYTES
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "r79_stdcorpora_eval.py",
            "--corpus-dir",
            str(corpus_dir),
            "--output-dir",
            str(output_dir),
            "--engines",
            "dagua",
        ],
    )

    exit_code = stdcorpora_eval.main()

    assert exit_code == 3
    assert output_dir.is_dir()
    payload: Dict[str, Any] = json.loads((output_dir / "results.json").read_text())
    assert payload["aborted"] is True
    assert payload["aborted_reason"] is not None
    rows: List[Dict[str, Any]] = payload["rows"]  # type: ignore[assignment]
    # Only the first row should have run before the guard fired.
    assert len(rows) == 1
