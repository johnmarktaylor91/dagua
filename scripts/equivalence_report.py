#!/usr/bin/env python3
"""Generate layout equivalence reports from stored benchmark positions."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dagua.eval.equivalence_metrics import (  # noqa: E402
    DEFAULT_MAX_AUTOMORPHISMS,
    EquivalenceMetrics,
    compute_equivalence_metrics,
)
from dagua.eval.graphs import get_test_graphs  # noqa: E402

HOLDOUT_ENGINE_PREFIXES: tuple[str, ...] = (
    "classic_sugiyama_",
    "classic_classical_mds_",
    "classic_pivot_mds_",
    "classic_spectral_random_walk",
)


@dataclass(frozen=True)
class ComparisonPair:
    """Benchmark record pair to compare."""

    graph: str
    engine: str
    reference_engine: str
    seed: Optional[int]
    dagua_key: str
    reference_key: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="Benchmark results.json path")
    parser.add_argument(
        "--positions",
        type=Path,
        required=True,
        help="positions.h5 file or positions/ directory with per-run .pt tensors",
    )
    parser.add_argument(
        "--combos",
        type=Path,
        default=None,
        help="Optional JSON list of {'graph': ..., 'engine': ...} combos to include",
    )
    parser.add_argument(
        "--graphs-from",
        default="test_graphs",
        choices=("test_graphs",),
        help="Graph registry to use for graph structure lookup",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--max-automorphisms",
        type=int,
        default=DEFAULT_MAX_AUTOMORPHISMS,
        help="Maximum automorphisms to enumerate per graph",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional cap for quick exploratory reports",
    )
    return parser.parse_args()


def load_results(results_path: Path) -> dict[str, dict[str, Any]]:
    """Load benchmark results.

    Parameters
    ----------
    results_path : pathlib.Path
        Path to ``results.json``.

    Returns
    -------
    dict[str, dict[str, Any]]
        Benchmark records keyed by result key.
    """
    with results_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {str(key): dict(value) for key, value in raw.items()}


def load_combo_filter(combos_path: Optional[Path]) -> Optional[set[tuple[str, str]]]:
    """Load an optional graph/engine combo filter.

    Parameters
    ----------
    combos_path : pathlib.Path, optional
        JSON path containing a list of ``{"graph": str, "engine": str}``.

    Returns
    -------
    set[tuple[str, str]] | None
        Allowed ``(graph, engine)`` tuples, or ``None`` when no filter is set.
    """
    if combos_path is None:
        return None
    combos = json.loads(combos_path.read_text(encoding="utf-8"))
    allowed: set[tuple[str, str]] = set()
    for combo in combos:
        allowed.add((str(combo["graph"]), str(combo["engine"])))
    return allowed


def build_graph_edge_index() -> dict[str, torch.Tensor]:
    """Build a graph-name to edge-index lookup from the evaluation registry.

    Returns
    -------
    dict[str, torch.Tensor]
        Edge indices keyed by benchmark graph name.
    """
    return {test_graph.name: test_graph.graph.edge_index for test_graph in get_test_graphs()}


def select_comparison_pairs(
    results: dict[str, dict[str, Any]],
    *,
    combo_filter: Optional[set[tuple[str, str]]] = None,
    max_pairs: Optional[int] = None,
) -> list[ComparisonPair]:
    """Select reimplementation/reference position pairs from result metadata.

    Parameters
    ----------
    results : dict[str, dict[str, Any]]
        Benchmark records keyed by result key.
    combo_filter : set[tuple[str, str]], optional
        Optional allowed ``(graph, engine)`` combos.
    max_pairs : int, optional
        Optional maximum number of pairs to return.

    Returns
    -------
    list[ComparisonPair]
        Ordered comparison pairs.
    """
    records_by_engine_graph: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = {}
    for key, record in results.items():
        if record.get("status") != "ok" or not record.get("positions_file"):
            continue
        engine = str(record.get("engine_name", ""))
        graph = str(record.get("graph_name", ""))
        records_by_engine_graph.setdefault((engine, graph), []).append((key, record))

    pairs: list[ComparisonPair] = []
    for dagua_key, dagua_record in results.items():
        if dagua_record.get("status") != "ok" or not dagua_record.get("positions_file"):
            continue
        engine = str(dagua_record.get("engine_name", ""))
        graph = str(dagua_record.get("graph_name", ""))
        if combo_filter is not None and (graph, engine) not in combo_filter:
            continue
        if combo_filter is None and not engine.startswith(HOLDOUT_ENGINE_PREFIXES):
            continue
        references = dagua_record.get("reimpl_of") or []
        if not references:
            continue
        seed = _optional_int(dagua_record.get("seed"))
        for reference_engine in references:
            reference_key = _find_reference_key(
                records_by_engine_graph=records_by_engine_graph,
                reference_engine=str(reference_engine),
                graph=graph,
                seed=seed,
            )
            if reference_key is None:
                continue
            pairs.append(
                ComparisonPair(
                    graph=graph,
                    engine=engine,
                    reference_engine=str(reference_engine),
                    seed=seed,
                    dagua_key=dagua_key,
                    reference_key=reference_key,
                )
            )
            break
        if max_pairs is not None and len(pairs) >= max_pairs:
            break
    return pairs


def load_position(
    *,
    positions_path: Path,
    results_dir: Path,
    results: dict[str, dict[str, Any]],
    key: str,
    h5_file: Optional[h5py.File],
) -> np.ndarray:
    """Load one position matrix from HDF5 or a ``.pt`` artifact.

    Parameters
    ----------
    positions_path : pathlib.Path
        HDF5 file path or positions directory path.
    results_dir : pathlib.Path
        Directory containing ``results.json``.
    results : dict[str, dict[str, Any]]
        Benchmark records keyed by result key.
    key : str
        Result key to load.
    h5_file : h5py.File, optional
        Open HDF5 handle when ``positions_path`` is an HDF5 file.

    Returns
    -------
    numpy.ndarray
        ``float64`` position matrix with shape ``[N, 2]``.
    """
    if h5_file is not None:
        if key not in h5_file:
            raise KeyError(f"Missing HDF5 position key: {key}")
        return np.asarray(h5_file[key][...], dtype=np.float64)
    record = results[key]
    relative_path = Path(str(record["positions_file"]))
    candidates = [
        positions_path / relative_path.name,
        positions_path / relative_path,
        results_dir / relative_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            tensor = torch.load(candidate, map_location="cpu", weights_only=False)
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Position artifact is not a torch.Tensor: {candidate}")
            return tensor.detach().cpu().numpy().astype(np.float64, copy=False)
    raise FileNotFoundError(f"Could not resolve position artifact for key {key!r}.")


def write_reports(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Write Markdown and JSON equivalence reports.

    Parameters
    ----------
    output_dir : pathlib.Path
        Destination directory.
    rows : list[dict[str, Any]]
        Computed report rows.

    Returns
    -------
    None
        Writes files in ``output_dir``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "equivalence.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# Layout Equivalence Report",
        "",
        (
            "| Graph | Engine | Seed | Plain RMSD | Aut RMSD | Aut Group | Stress Delta | "
            "Dist Corr | Gram Eig Diff | Verdict |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {graph} | {engine} | {seed} | {plain:.6g} | {aut:.6g} | {group} | "
            "{stress:.6g} | {corr:.6g} | {eig:.6g} | {verdict} |".format(
                graph=row["graph"],
                engine=row["engine"],
                seed="det" if row["seed"] is None else row["seed"],
                plain=row["plain_procrustes_rmsd"],
                aut=row["aut_procrustes_rmsd"],
                group=row["aut_group_size"],
                stress=row["stress_rel_delta"],
                corr=row["dist_matrix_corr"],
                eig=row["gram_eig_max_absdiff"],
                verdict=row["verdict"],
            )
        )
    (output_dir / "equivalence_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Run the equivalence report command.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args()
    results = load_results(args.results)
    combo_filter = load_combo_filter(args.combos)
    pairs = select_comparison_pairs(results, combo_filter=combo_filter, max_pairs=args.max_pairs)
    graph_edges = build_graph_edge_index()
    h5_file: Optional[h5py.File] = None
    if args.positions.is_file():
        h5_file = h5py.File(args.positions, "r")
    rows: list[dict[str, Any]] = []
    try:
        for pair in pairs:
            if pair.graph not in graph_edges:
                print(f"[skip] graph not found in registry: {pair.graph}", file=sys.stderr)
                continue
            dagua_pos = load_position(
                positions_path=args.positions,
                results_dir=args.results.parent,
                results=results,
                key=pair.dagua_key,
                h5_file=h5_file,
            )
            reference_pos = load_position(
                positions_path=args.positions,
                results_dir=args.results.parent,
                results=results,
                key=pair.reference_key,
                h5_file=h5_file,
            )
            metrics = compute_equivalence_metrics(
                dagua_pos,
                reference_pos,
                graph_edges[pair.graph],
                max_automorphisms=args.max_automorphisms,
            )
            rows.append(_row_from_metrics(pair, metrics))
    finally:
        if h5_file is not None:
            h5_file.close()
    write_reports(args.output, rows)
    print(f"Wrote {len(rows)} rows to {args.output}", file=sys.stderr)
    return 0


def _find_reference_key(
    *,
    records_by_engine_graph: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]],
    reference_engine: str,
    graph: str,
    seed: Optional[int],
) -> Optional[str]:
    """Find the best reference record key for a reimplementation seed.

    Parameters
    ----------
    records_by_engine_graph : dict[tuple[str, str], list[tuple[str, dict[str, Any]]]]
        Successful records grouped by ``(engine, graph)``.
    reference_engine : str
        Reference engine name.
    graph : str
        Benchmark graph name.
    seed : int, optional
        Reimplementation seed.

    Returns
    -------
    str | None
        Matching result key, or ``None`` if no reference record exists.
    """
    candidates = records_by_engine_graph.get((reference_engine, graph), [])
    if not candidates:
        return None
    for key, record in candidates:
        if _optional_int(record.get("seed")) == seed:
            return key
    for key, record in candidates:
        if record.get("seed") is None or not bool(record.get("is_stochastic", True)):
            return key
    return candidates[0][0]


def _row_from_metrics(pair: ComparisonPair, metrics: EquivalenceMetrics) -> dict[str, Any]:
    """Merge pair metadata and metric fields into one report row.

    Parameters
    ----------
    pair : ComparisonPair
        Compared benchmark records.
    metrics : EquivalenceMetrics
        Computed metric result.

    Returns
    -------
    dict[str, Any]
        JSON-ready report row.
    """
    row: dict[str, Any] = {
        "graph": pair.graph,
        "engine": pair.engine,
        "reference_engine": pair.reference_engine,
        "seed": pair.seed,
        "dagua_key": pair.dagua_key,
        "reference_key": pair.reference_key,
    }
    row.update(metrics.to_dict())
    return row


def _optional_int(value: Any) -> Optional[int]:
    """Convert benchmark seed values to ``int`` or ``None``.

    Parameters
    ----------
    value : Any
        Raw seed value.

    Returns
    -------
    int | None
        Integer seed, or ``None`` for deterministic records.
    """
    if value is None:
        return None
    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())
