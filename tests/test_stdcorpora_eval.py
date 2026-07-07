"""Tests for the r79 standard-corpora evaluation harness."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict

from scripts.r79_stdcorpora_eval import load_gml_file, load_graph_file, load_mtx_file


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
