"""Build the frozen r79 native-layout quality baseline."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

from dagua.eval.competitors import get_competitor
from dagua.eval.competitors.base import CompetitorBase, CompetitorResult
from dagua.eval.graphs import TestGraph, get_test_graphs
from dagua.metrics import composite_auto, evaluate

OUTPUT_DIR = Path("eval_output/r79_baseline")
POSITIONS_DIR = OUTPUT_DIR / "positions"
RESULTS_PATH = OUTPUT_DIR / "results.json"
REPORT_PATH = OUTPUT_DIR / "BASELINE.md"
SEED = 42
TIMEOUT_SECONDS = 120.0
TIE_BAND = 0.5

R79_NEW_GRAPH_NAMES = {
    "r79_weighted_community_4x18",
    "r79_weighted_mesh_10x12",
    "r79_weighted_skew_dag_6x10",
    "r79_weighted_hub_spoke_4x18",
    "r79_weighted_small_world_120",
    "r79_weighted_ladder_40",
    "r79_weighted_bipartite_16x24",
    "r79_nested_clusters_3x2x10",
    "r79_nested_clusters_2x3x12",
    "r79_nested_clusters_4x2x8",
    "r79_undirected_sbm_low_mix_4x25",
    "r79_undirected_sbm_mid_mix_5x20",
    "r79_undirected_sbm_high_mix_3x30",
    "r79_directed_scc_90_3cores",
    "r79_directed_scc_120_2cores",
}

ENGINE_NAMES = [
    "dagua",
    "graphviz_dot",
    "graphviz_sfdp",
    "graphviz_neato",
    "elk_layered",
    "dagre",
    "nx_spring",
    "igraph_kamada_kawai",
    "igraph_sugiyama",
]

EXTERNAL_ENGINE_NAMES = [name for name in ENGINE_NAMES if name != "dagua"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed CLI options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dagua-only",
        action="store_true",
        help="Rerun only Dagua rows against frozen external rows and positions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Baseline output directory.",
    )
    return parser.parse_args()


def git_sha() -> str:
    """Return the current git commit SHA.

    Returns
    -------
    str
        Current commit SHA, or ``unknown`` if git cannot answer.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def safe_component(value: str) -> str:
    """Return a filesystem-safe artifact name component.

    Parameters
    ----------
    value : str
        Raw graph or engine name.

    Returns
    -------
    str
        Sanitized name suitable for a position tensor filename.
    """
    return "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in value)


def position_relpath(graph_name: str, engine_name: str) -> str:
    """Return the relative position-tensor path for a graph and engine.

    Parameters
    ----------
    graph_name : str
        Benchmark graph name.
    engine_name : str
        Benchmark engine name.

    Returns
    -------
    str
        POSIX-style relative path below the output directory.
    """
    filename = f"{safe_component(graph_name)}__{safe_component(engine_name)}.pt"
    return str(Path("positions") / filename)


def json_clean(value: Any) -> Any:
    """Convert metric payload values into JSON-safe primitives.

    Parameters
    ----------
    value : Any
        Raw value from metrics or runtime metadata.

    Returns
    -------
    Any
        JSON-serializable value.
    """
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return json_clean(value.item())
        return [json_clean(item) for item in value.detach().cpu().flatten().tolist()]
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return numeric if math.isfinite(numeric) else None


def load_existing_results(output_dir: Path) -> Dict[str, Any]:
    """Load an existing results payload.

    Parameters
    ----------
    output_dir : Path
        Baseline output directory containing ``results.json``.

    Returns
    -------
    Dict[str, Any]
        Parsed results payload.

    Raises
    ------
    FileNotFoundError
        If the existing results file is absent.
    """
    with (output_dir / "results.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_results(output_dir: Path, payload: Dict[str, Any]) -> None:
    """Write the results payload to disk.

    Parameters
    ----------
    output_dir : Path
        Baseline output directory.
    payload : Dict[str, Any]
        JSON-serializable results payload.

    Returns
    -------
    None
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_corpus() -> List[TestGraph]:
    """Build the benchmark corpus once and filter it to small graphs.

    Returns
    -------
    List[TestGraph]
        All registered test graphs with at most 500 nodes.
    """
    graphs = get_test_graphs(max_nodes=500)
    for test_graph in graphs:
        test_graph.graph.compute_node_sizes()
    return graphs


def graph_population(graph_name: str) -> str:
    """Return the r79 reporting population for a graph.

    Parameters
    ----------
    graph_name : str
        Benchmark graph name.

    Returns
    -------
    str
        ``"extended"`` for r79 additions, otherwise ``"legacy"``.
    """
    return "extended" if graph_name in R79_NEW_GRAPH_NAMES else "legacy"


def engine_availability(engine_names: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Inspect availability of requested competitor adapters.

    Parameters
    ----------
    engine_names : Sequence[str]
        Engine names to check.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Availability metadata keyed by engine name.
    """
    availability: Dict[str, Dict[str, Any]] = {}
    for engine_name in engine_names:
        competitor = get_competitor(engine_name)
        if competitor is None:
            availability[engine_name] = {"available": False, "reason": "adapter not registered"}
            continue
        try:
            available = competitor.available()
        except Exception as exc:  # noqa: BLE001
            availability[engine_name] = {"available": False, "reason": str(exc)}
            continue
        reason = None if available else "adapter unavailable"
        availability[engine_name] = {"available": bool(available), "reason": reason}
    return availability


def is_semantically_directed(test_graph: TestGraph) -> bool:
    """Return whether graph direction should affect the composite score.

    Parameters
    ----------
    test_graph : TestGraph
        Graph metadata and topology.

    Returns
    -------
    bool
        ``False`` only for graphs tagged as undirected.
    """
    return "undirected" not in test_graph.tags


def row_key(row: Dict[str, Any]) -> Tuple[str, str]:
    """Return the graph-engine key for a results row.

    Parameters
    ----------
    row : Dict[str, Any]
        Results row.

    Returns
    -------
    Tuple[str, str]
        Graph name and engine name.
    """
    return str(row["graph"]), str(row["engine"])


def make_skip_row(
    test_graph: TestGraph,
    engine_name: str,
    reason: str,
) -> Dict[str, Any]:
    """Create a SKIP row for one graph-engine pair.

    Parameters
    ----------
    test_graph : TestGraph
        Benchmark graph metadata.
    engine_name : str
        Engine name.
    reason : str
        Skip explanation.

    Returns
    -------
    Dict[str, Any]
        Results row.
    """
    return {
        "graph": test_graph.name,
        "population": graph_population(test_graph.name),
        "engine": engine_name,
        "status": "SKIP",
        "runtime_s": 0.0,
        "metrics": {},
        "composite": None,
        "positions_path": None,
        "nodes": test_graph.graph.num_nodes,
        "edges": int(test_graph.graph.edge_index.shape[1]),
        "reason": reason,
    }


def run_engine(
    test_graph: TestGraph,
    competitor: CompetitorBase,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run one engine on one graph and persist positions for OK rows.

    Parameters
    ----------
    test_graph : TestGraph
        Benchmark graph metadata.
    competitor : CompetitorBase
        Layout competitor adapter.
    output_dir : Path
        Baseline output directory.

    Returns
    -------
    Dict[str, Any]
        Results row.
    """
    graph = test_graph.graph
    try:
        result: CompetitorResult = competitor.layout(graph, timeout=TIMEOUT_SECONDS, seed=SEED)
    except Exception as exc:  # noqa: BLE001
        return {
            "graph": test_graph.name,
            "population": graph_population(test_graph.name),
            "engine": competitor.name,
            "status": "ERROR",
            "runtime_s": 0.0,
            "metrics": {},
            "composite": None,
            "positions_path": None,
            "nodes": graph.num_nodes,
            "edges": int(graph.edge_index.shape[1]),
            "error": f"{type(exc).__name__}: {exc}",
        }

    base_row = {
        "graph": test_graph.name,
        "population": graph_population(test_graph.name),
        "engine": competitor.name,
        "runtime_s": float(result.runtime_seconds),
        "nodes": graph.num_nodes,
        "edges": int(graph.edge_index.shape[1]),
    }
    if result.pos is None:
        return {
            **base_row,
            "status": "ERROR",
            "metrics": {},
            "composite": None,
            "positions_path": None,
            "error": result.error or "adapter returned no positions",
        }

    positions = result.pos.detach().cpu().to(dtype=torch.float32)
    try:
        metrics = evaluate(graph, positions, tier="full")
        composite = composite_auto(metrics, is_semantically_directed(test_graph))
    except Exception as exc:  # noqa: BLE001
        return {
            **base_row,
            "status": "ERROR",
            "metrics": {},
            "composite": None,
            "positions_path": None,
            "error": f"metrics {type(exc).__name__}: {exc}",
        }

    relpath = position_relpath(test_graph.name, competitor.name)
    position_path = output_dir / relpath
    position_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(positions, position_path)
    return {
        **base_row,
        "status": "OK",
        "metrics": json_clean(metrics),
        "composite": float(composite),
        "positions_path": relpath,
        "error": None,
    }


def validate_store(output_dir: Path) -> None:
    """Validate one-to-one consistency between OK rows and position tensors.

    Parameters
    ----------
    output_dir : Path
        Baseline output directory.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If a row or position artifact is missing its counterpart.
    """
    payload = load_existing_results(output_dir)
    rows = payload.get("rows", [])
    ok_paths = {
        str(row.get("positions_path"))
        for row in rows
        if row.get("status") == "OK" and row.get("positions_path")
    }
    missing = sorted(path for path in ok_paths if not (output_dir / path).is_file())
    position_paths = {
        str(path.relative_to(output_dir))
        for path in (output_dir / "positions").glob("*.pt")
        if path.is_file()
    }
    orphaned = sorted(position_paths - ok_paths)
    if missing or orphaned:
        details = []
        if missing:
            details.append(f"missing position files for OK rows: {missing}")
        if orphaned:
            details.append(f"orphaned position files without OK rows: {orphaned}")
        raise RuntimeError("; ".join(details))


def graph_best_external(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Find the best external OK row for each graph.

    Parameters
    ----------
    rows : Iterable[Dict[str, Any]]
        Results rows.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Best external row keyed by graph name.
    """
    best: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if row.get("engine") not in EXTERNAL_ENGINE_NAMES or row.get("status") != "OK":
            continue
        composite = row.get("composite")
        if composite is None:
            continue
        graph_name = str(row["graph"])
        if graph_name not in best or float(composite) > float(best[graph_name]["composite"]):
            best[graph_name] = row
    return best


def summarize_wtl(rows: List[Dict[str, Any]], population: str) -> Tuple[int, int, int]:
    """Summarize Dagua wins, ties, and losses for one population.

    Parameters
    ----------
    rows : List[Dict[str, Any]]
        Results rows.
    population : str
        Reporting population: ``legacy`` or ``extended``.

    Returns
    -------
    Tuple[int, int, int]
        Win, tie, and loss counts.
    """
    rows_for_population = [row for row in rows if row.get("population") == population]
    best_external = graph_best_external(rows_for_population)
    dagua_rows = {
        str(row["graph"]): row
        for row in rows_for_population
        if row.get("engine") == "dagua" and row.get("status") == "OK"
    }
    wins = ties = losses = 0
    for graph_name, dagua_row in dagua_rows.items():
        external_row = best_external.get(graph_name)
        if external_row is None:
            continue
        delta = float(dagua_row["composite"]) - float(external_row["composite"])
        if delta > TIE_BAND:
            wins += 1
        elif delta >= -TIE_BAND:
            ties += 1
        else:
            losses += 1
    return wins, ties, losses


def per_graph_comparison(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build per-graph Dagua-vs-best-external comparison rows.

    Parameters
    ----------
    rows : List[Dict[str, Any]]
        Results rows.

    Returns
    -------
    List[Dict[str, Any]]
        Comparison rows sorted by population then graph name.
    """
    best_external = graph_best_external(rows)
    dagua_rows = {
        str(row["graph"]): row
        for row in rows
        if row.get("engine") == "dagua" and row.get("status") == "OK"
    }
    comparisons: List[Dict[str, Any]] = []
    for graph_name, dagua_row in sorted(dagua_rows.items()):
        external_row = best_external.get(graph_name)
        if external_row is None:
            continue
        delta = float(dagua_row["composite"]) - float(external_row["composite"])
        comparisons.append(
            {
                "graph": graph_name,
                "population": dagua_row["population"],
                "dagua": float(dagua_row["composite"]),
                "best_external": float(external_row["composite"]),
                "delta": delta,
                "winner": "dagua" if delta > TIE_BAND else str(external_row["engine"]),
            }
        )
    return sorted(comparisons, key=lambda item: (str(item["population"]), str(item["graph"])))


def generate_report(output_dir: Path, payload: Dict[str, Any]) -> None:
    """Generate the Markdown baseline report.

    Parameters
    ----------
    output_dir : Path
        Baseline output directory.
    payload : Dict[str, Any]
        Results payload.

    Returns
    -------
    None
    """
    rows = list(payload["rows"])
    comparisons = per_graph_comparison(rows)
    losses = sorted(
        [item for item in comparisons if float(item["delta"]) < -TIE_BAND],
        key=lambda item: float(item["delta"]),
    )
    lines = [
        "# R79 Baseline",
        "",
        f"- Date: {payload['metadata']['date']}",
        f"- Git SHA: {payload['metadata']['git_sha']}",
        f"- Corpus <=500 nodes: {payload['metadata']['graph_count']}",
        f"- Legacy graphs: {payload['metadata']['legacy_count']}",
        f"- R79 extension graphs: {payload['metadata']['extended_count']}",
        f"- Tie band: +/-{TIE_BAND:.1f} composite points",
        "",
        "## Engine Availability",
        "",
        "| Engine | Available | Reason |",
        "| --- | ---: | --- |",
    ]
    for engine_name in ENGINE_NAMES:
        info = payload["metadata"]["engine_availability"].get(engine_name, {})
        lines.append(
            f"| {engine_name} | {str(bool(info.get('available')))} | {info.get('reason') or ''} |"
        )

    lines.extend(
        [
            "",
            "## Scoreboards",
            "",
            "| Population | W | T | L |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for population in ("legacy", "extended"):
        wins, ties, losses_count = summarize_wtl(rows, population)
        lines.append(f"| {population} | {wins} | {ties} | {losses_count} |")

    lines.extend(
        [
            "",
            "## Per-Graph Comparison",
            "",
            "| Population | Graph | Dagua | Best External | Delta | Winning Engine |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for item in comparisons:
        lines.append(
            "| {population} | {graph} | {dagua:.3f} | {best_external:.3f} | "
            "{delta:.3f} | {winner} |".format(**item)
        )

    lines.extend(
        [
            "",
            "## Losses Worst First",
            "",
            "| Population | Graph | Delta | Winning Engine |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for item in losses:
        lines.append("| {population} | {graph} | {delta:.3f} | {winner} |".format(**item))

    if payload["metadata"].get("positions_note"):
        lines.extend(["", "## Position Store", "", str(payload["metadata"]["positions_note"])])

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "BASELINE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_full(output_dir: Path, graphs: List[TestGraph]) -> Dict[str, Any]:
    """Run the complete r79 baseline from scratch.

    Parameters
    ----------
    output_dir : Path
        Baseline output directory.
    graphs : List[TestGraph]
        Corpus graphs.

    Returns
    -------
    Dict[str, Any]
        Results payload.
    """
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "positions").mkdir(parents=True, exist_ok=True)
    availability = engine_availability(ENGINE_NAMES)
    rows: List[Dict[str, Any]] = []
    for test_graph in graphs:
        for engine_name in ENGINE_NAMES:
            competitor = get_competitor(engine_name)
            info = availability[engine_name]
            if competitor is None or not info["available"]:
                rows.append(make_skip_row(test_graph, engine_name, str(info.get("reason"))))
                continue
            print(f"RUN {test_graph.name} {engine_name}", flush=True)
            rows.append(run_engine(test_graph, competitor, output_dir))
    return build_payload(rows, graphs, availability)


def run_dagua_only(output_dir: Path, graphs: List[TestGraph]) -> Dict[str, Any]:
    """Rerun only Dagua rows while preserving frozen external rows.

    Parameters
    ----------
    output_dir : Path
        Baseline output directory.
    graphs : List[TestGraph]
        Corpus graphs.

    Returns
    -------
    Dict[str, Any]
        Updated results payload.
    """
    existing = load_existing_results(output_dir)
    availability = dict(existing["metadata"].get("engine_availability", {}))
    availability["dagua"] = engine_availability(["dagua"])["dagua"]
    external_rows = [row for row in existing["rows"] if row.get("engine") != "dagua"]
    for path in (output_dir / "positions").glob("*__dagua.pt"):
        path.unlink()
    competitor = get_competitor("dagua")
    if competitor is None:
        raise RuntimeError("dagua adapter not registered")
    dagua_rows = []
    for test_graph in graphs:
        print(f"RUN {test_graph.name} dagua", flush=True)
        dagua_rows.append(run_engine(test_graph, competitor, output_dir))
    rows = sorted([*external_rows, *dagua_rows], key=lambda row: (row["graph"], row["engine"]))
    return build_payload(rows, graphs, availability)


def build_payload(
    rows: List[Dict[str, Any]],
    graphs: List[TestGraph],
    availability: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the persisted results payload.

    Parameters
    ----------
    rows : List[Dict[str, Any]]
        Results rows.
    graphs : List[TestGraph]
        Corpus graphs.
    availability : Dict[str, Dict[str, Any]]
        Engine availability metadata.

    Returns
    -------
    Dict[str, Any]
        Results payload ready to serialize.
    """
    legacy_count = sum(1 for graph in graphs if graph_population(graph.name) == "legacy")
    extended_count = sum(1 for graph in graphs if graph_population(graph.name) == "extended")
    return {
        "metadata": {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_sha": git_sha(),
            "seed": SEED,
            "timeout_s": TIMEOUT_SECONDS,
            "graph_count": len(graphs),
            "legacy_count": legacy_count,
            "extended_count": extended_count,
            "new_graph_names": sorted(R79_NEW_GRAPH_NAMES),
            "engine_names": ENGINE_NAMES,
            "engine_availability": availability,
        },
        "rows": rows,
    }


def main() -> int:
    """Run the r79 baseline CLI.

    Returns
    -------
    int
        Process exit status.
    """
    args = parse_args()
    output_dir: Path = args.output_dir
    start = time.perf_counter()
    graphs = build_corpus()
    print(f"Corpus <=500 nodes: {len(graphs)}", flush=True)
    if args.dagua_only:
        payload = run_dagua_only(output_dir, graphs)
    else:
        payload = run_full(output_dir, graphs)
    payload["metadata"]["wall_time_s"] = round(time.perf_counter() - start, 3)
    write_results(output_dir, payload)
    validate_store(output_dir)
    generate_report(output_dir, payload)
    validate_store(output_dir)
    print(f"Wrote {output_dir / 'results.json'}", flush=True)
    print(f"Wrote {output_dir / 'BASELINE.md'}", flush=True)
    print(f"Wall time: {payload['metadata']['wall_time_s']:.3f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
