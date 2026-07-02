"""Regression tests for definitive fidelity benchmark overlay hygiene."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts import definitive_fidelity_analysis as analysis


def _write_results(data_dir: Path, rows: dict[str, dict[str, Any]]) -> None:
    """Write a synthetic benchmark ``results.json`` file.

    Parameters
    ----------
    data_dir : pathlib.Path
        Benchmark root to create.
    rows : dict[str, dict[str, Any]]
        Raw result rows keyed by benchmark record id.

    Returns
    -------
    None
        The file is written in place.
    """
    data_dir.mkdir()
    with (data_dir / "results.json").open("w") as file_obj:
        json.dump(rows, file_obj)


def _row(graph: str, engine: str, seed: int, status: str = "ok") -> dict[str, Any]:
    """Build one minimal benchmark result row.

    Parameters
    ----------
    graph : str
        Graph name.
    engine : str
        Engine name.
    seed : int
        Benchmark seed value.
    status : str, default="ok"
        Benchmark row status.

    Returns
    -------
    dict[str, Any]
        Minimal row payload accepted by the fidelity loader.
    """
    return {
        "graph_name": graph,
        "engine_name": engine,
        "seed": seed,
        "status": status,
        "positions_file": f"positions/{graph}_{engine}_{seed}.pt",
        "runtime_seconds": 0.1,
        "num_nodes": 4,
    }


def _seed_key(graph: str, engine: str, seed: int) -> str:
    """Return a benchmark record key for one seeded row.

    Parameters
    ----------
    graph : str
        Graph name.
    engine : str
        Engine name.
    seed : int
        Benchmark seed value.

    Returns
    -------
    str
        Key of the form ``graph::engine::seedN``.
    """
    return f"{graph}::{engine}::seed{seed}"


def test_overlay_uses_later_dir_only_for_combo_seed_overlap(tmp_path: Path) -> None:
    """Later ok coverage should replace, not union with, earlier combo rows."""
    graph = "parallel_cycles_4x5"
    engine = "classic_classical_mds_igraph_fidelity"
    early = tmp_path / "early"
    late = tmp_path / "late"
    _write_results(
        early,
        {_seed_key(graph, engine, seed): _row(graph, engine, seed) for seed in range(42, 45)},
    )
    _write_results(
        late,
        {_seed_key(graph, engine, seed): _row(graph, engine, seed) for seed in range(100, 103)},
    )

    index = analysis.index_results(analysis.load_results_multi([early, late]))

    rows = index[(graph, engine)]
    assert [row.seed for row in rows] == [100, 101, 102]
    assert {Path(str(row.positions_file)).parts[-3] for row in rows} == {"late"}


def test_overlay_keeps_earlier_dir_for_combo_absent_from_later(tmp_path: Path) -> None:
    """Older benchmark roots should still supply combos missing from newer roots."""
    graph = "path_10"
    engine = "classic_fr"
    early = tmp_path / "early"
    late = tmp_path / "late"
    _write_results(early, {_seed_key(graph, engine, 42): _row(graph, engine, 42)})
    _write_results(late, {_seed_key("other_graph", engine, 100): _row("other_graph", engine, 100)})

    index = analysis.index_results(analysis.load_results_multi([early, late]))

    rows = index[(graph, engine)]
    assert [row.seed for row in rows] == [42]
    assert Path(str(rows[0].positions_file)).parts[-3] == "early"


def test_overlay_ignores_later_dir_with_only_error_rows(tmp_path: Path) -> None:
    """A newer directory only wins a combo when it has at least one ok row."""
    graph = "grid_4x4"
    engine = "classic_umap"
    early = tmp_path / "early"
    late = tmp_path / "late"
    _write_results(early, {_seed_key(graph, engine, 42): _row(graph, engine, 42)})
    _write_results(late, {_seed_key(graph, engine, 100): _row(graph, engine, 100, "error")})

    index = analysis.index_results(analysis.load_results_multi([early, late]))

    rows = index[(graph, engine)]
    assert [row.seed for row in rows] == [42]
    assert rows[0].status == "ok"
    assert Path(str(rows[0].positions_file)).parts[-3] == "early"


def test_overlay_applies_same_rule_to_reference_engines(tmp_path: Path) -> None:
    """Co-benchmarked reference engines should not union seed eras either."""
    graph = "tree_12"
    engine = "igraph_mds__for__classic_classical_mds_igraph_fidelity"
    early = tmp_path / "early"
    late = tmp_path / "late"
    _write_results(
        early,
        {_seed_key(graph, engine, seed): _row(graph, engine, seed) for seed in (42, 43)},
    )
    _write_results(
        late,
        {_seed_key(graph, engine, seed): _row(graph, engine, seed) for seed in (100, 101)},
    )

    index = analysis.index_results(analysis.load_results_multi([early, late]))

    rows = index[(graph, engine)]
    assert [row.seed for row in rows] == [100, 101]
    assert {Path(str(row.positions_file)).parts[-3] for row in rows} == {"late"}
