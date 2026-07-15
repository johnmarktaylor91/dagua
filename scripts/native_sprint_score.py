"""Score native runs vs the full originals field on the frozen r83 honest ruler.

Reads benchmark ``results.json`` stores directly (field baseline + one or more
native runs), scores every stored layout with the frozen r83 honest composite
(``dagua.metrics.composite_auto``), and reports the per-graph best-or-tied
verdict for the native engine against the best field engine on the shared
frozen 108-graph corpus.

The field scoring is cached to ``--field-cache`` so later sprint rounds only
re-score the native side.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from dagua.eval.benchmark import _declares_hierarchy
from dagua.eval.graphs import TestGraph, get_test_graphs, is_semantically_directed
from dagua.eval.size_policy import set_size_aware_externals
from dagua.metrics import composite_auto, evaluate

TIE_BAND = 0.5
_CLASSICAL_ENGINES = (
    "dagre",
    "elk_layered",
    "graphviz_dot",
    "graphviz_neato",
    "graphviz_sfdp",
    "igraph_kamada_kawai",
    "igraph_sugiyama",
    "nx_spring",
)
_CORPUS_RE = re.compile(
    r"^(?P<graph>.+)__(?P<engine>" + "|".join(_CLASSICAL_ENGINES) + r")(?P<variant>.*)\.pt$"
)
_WORKER_GRAPHS: Dict[str, TestGraph] = {}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line options.

    Parameters
    ----------
    argv : Optional[Sequence[str]], optional
        Explicit argument sequence, or ``None`` for ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--field-dir",
        type=Path,
        required=True,
        help="Benchmark output dir of the field baseline (results.json + positions/).",
    )
    parser.add_argument(
        "--native-dir",
        type=Path,
        required=True,
        help="Benchmark output dir of the native run to compare (engine 'dagua').",
    )
    parser.add_argument(
        "--corpus-positions",
        type=Path,
        default=Path("/home/jtaylor/projects/dagua/eval_output/r81_regate2/positions"),
        help="Frozen r81 positions dir that defines the shared 108-graph corpus.",
    )
    parser.add_argument(
        "--field-cache",
        type=Path,
        default=None,
        help="JSON cache of field scores (computed and written when absent).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) // 2))
    return parser.parse_args(argv)


def corpus_names(corpus_positions: Path) -> List[str]:
    """Return the exact frozen 108-graph shared corpus.

    Parameters
    ----------
    corpus_positions : Path
        Directory holding the frozen r81 classical position tensors.

    Returns
    -------
    List[str]
        Sorted shared corpus names.
    """
    names = {
        match.group("graph")
        for path in corpus_positions.glob("*.pt")
        if (match := _CORPUS_RE.match(path.name)) is not None
    }
    if len(names) != 108:
        raise RuntimeError(f"expected the frozen 108-graph corpus, found {len(names)}")
    return sorted(names)


def load_run_tasks(
    run_dir: Path, names: Sequence[str], engine_filter: Optional[str] = None
) -> List[Tuple[str, str, List[str]]]:
    """Collect scoreable (graph, engine, paths) tasks from a benchmark store.

    Parameters
    ----------
    run_dir : Path
        Benchmark output dir containing ``results.json`` and ``positions/``.
    names : Sequence[str]
        Frozen corpus names to keep.
    engine_filter : Optional[str], optional
        Keep only this engine when set.

    Returns
    -------
    List[Tuple[str, str, List[str]]]
        Graph name, engine name, and absolute tensor paths.
    """
    wanted = set(names)
    results = json.loads((run_dir / "results.json").read_text())
    grouped: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for record in results.values():
        if record.get("status") != "ok" or not record.get("positions_file"):
            continue
        graph = str(record["graph_name"])
        engine = str(record["engine_name"])
        if graph not in wanted:
            continue
        if engine_filter is not None and engine != engine_filter:
            continue
        path = run_dir / str(record["positions_file"])
        if path.exists():
            grouped[(graph, engine)].append(str(path))
    return [(graph, engine, sorted(paths)) for (graph, engine), paths in sorted(grouped.items())]


def build_graph_map(names: Sequence[str]) -> Dict[str, TestGraph]:
    """Build current corpus graphs and select the frozen names.

    Parameters
    ----------
    names : Sequence[str]
        Frozen corpus graph names.

    Returns
    -------
    Dict[str, TestGraph]
        Test graphs keyed by name, with measured node sizes.
    """
    wanted = set(names)
    selected = {
        graph.name: graph for graph in get_test_graphs(max_nodes=500) if graph.name in wanted
    }
    missing = sorted(wanted - set(selected))
    if missing:
        raise RuntimeError(f"current corpus cannot reconstruct frozen graph(s): {missing}")
    for test_graph in selected.values():
        test_graph.graph.compute_node_sizes()
    return selected


def init_worker(graphs: Dict[str, TestGraph]) -> None:
    """Initialize process-local graph state.

    Parameters
    ----------
    graphs : Dict[str, TestGraph]
        Reconstructed corpus graphs.

    Returns
    -------
    None
    """
    global _WORKER_GRAPHS
    _WORKER_GRAPHS = graphs
    torch.set_num_threads(1)
    set_size_aware_externals(True)


def score_group(task: Tuple[str, str, List[str]]) -> Dict[str, Any]:
    """Score every stored layout for one graph-engine pair.

    Parameters
    ----------
    task : Tuple[str, str, List[str]]
        Graph name, engine name, and matching tensor paths.

    Returns
    -------
    Dict[str, Any]
        Best honest-composite result for the pair.
    """
    graph_name, engine, paths = task
    test_graph = _WORKER_GRAPHS[graph_name]
    best_score: Optional[float] = None
    best_path: Optional[str] = None
    errors: List[str] = []
    expected_shape = (test_graph.graph.num_nodes, 2)
    for path_string in paths:
        try:
            positions = torch.load(path_string, map_location="cpu", weights_only=True)
            if not isinstance(positions, torch.Tensor) or tuple(positions.shape) != expected_shape:
                raise ValueError(f"shape {getattr(positions, 'shape', None)} != {expected_shape}")
            metrics = evaluate(test_graph.graph, positions.to(dtype=torch.float32), tier="full")
            metrics["declared_hierarchical"] = _declares_hierarchy(test_graph)
            score = float(composite_auto(metrics, is_semantically_directed(test_graph)))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path_string}: {type(exc).__name__}: {exc}")
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_path = path_string
    return {
        "graph": graph_name,
        "engine": engine,
        "best_composite": best_score,
        "best_path": best_path,
        "error_count": len(errors),
        "errors": errors[:5],
    }


def score_tasks(
    tasks: List[Tuple[str, str, List[str]]],
    graphs: Dict[str, TestGraph],
    workers: int,
) -> List[Dict[str, Any]]:
    """Score all tasks with a worker pool.

    Parameters
    ----------
    tasks : List[Tuple[str, str, List[str]]]
        Scoring tasks.
    graphs : Dict[str, TestGraph]
        Reconstructed corpus graphs.
    workers : int
        Process count.

    Returns
    -------
    List[Dict[str, Any]]
        Scored rows.
    """
    if workers <= 1:
        init_worker(graphs)
        return [score_group(task) for task in tasks]
    context = mp.get_context("fork")
    with context.Pool(workers, initializer=init_worker, initargs=(graphs,)) as pool:
        return list(pool.imap_unordered(score_group, tasks, chunksize=4))


def classify(delta: float) -> str:
    """Classify a native score delta using the frozen tie band.

    Parameters
    ----------
    delta : float
        Native composite minus best field composite.

    Returns
    -------
    str
        ``strictly_best``, ``tied``, or ``behind``.
    """
    if delta > TIE_BAND:
        return "strictly_best"
    if delta >= -TIE_BAND:
        return "tied"
    return "behind"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the native-vs-field honest-ruler comparison.

    Parameters
    ----------
    argv : Optional[Sequence[str]], optional
        Explicit argument sequence, or ``None`` for ``sys.argv``.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    names = corpus_names(args.corpus_positions)
    graphs = build_graph_map(names)

    field_rows: Optional[List[Dict[str, Any]]] = None
    if args.field_cache is not None and args.field_cache.exists():
        field_rows = json.loads(args.field_cache.read_text())["rows"]
        print(f"[score] field cache hit: {len(field_rows)} rows", flush=True)
    if field_rows is None:
        field_tasks = load_run_tasks(args.field_dir, names)
        print(f"[score] scoring field: {len(field_tasks)} graph-engine pairs", flush=True)
        field_rows = score_tasks(field_tasks, graphs, args.workers)
        if args.field_cache is not None:
            args.field_cache.parent.mkdir(parents=True, exist_ok=True)
            args.field_cache.write_text(json.dumps({"rows": field_rows}, indent=1))

    native_tasks = load_run_tasks(args.native_dir, names, engine_filter="dagua")
    print(f"[score] scoring native: {len(native_tasks)} graphs", flush=True)
    native_rows = score_tasks(native_tasks, graphs, args.workers)

    native_by_graph = {
        str(row["graph"]): row for row in native_rows if row["best_composite"] is not None
    }
    field_by_graph: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in field_rows:
        if row["best_composite"] is not None:
            field_by_graph[str(row["graph"])].append(row)

    init_worker(graphs)
    per_graph: List[Dict[str, Any]] = []
    tallies = {"strictly_best": 0, "tied": 0, "behind": 0, "missing": 0}
    for graph_name in names:
        native_row = native_by_graph.get(graph_name)
        competitors = field_by_graph.get(graph_name, [])
        if native_row is None or not competitors:
            tallies["missing"] += 1
            per_graph.append(
                {
                    "graph": graph_name,
                    "status": "missing",
                    "native": None if native_row is None else native_row["best_composite"],
                    "field_count": len(competitors),
                }
            )
            continue
        winner = max(competitors, key=lambda row: float(row["best_composite"]))
        delta = float(native_row["best_composite"]) - float(winner["best_composite"])
        status = classify(delta)
        tallies[status] += 1
        test_graph = graphs[graph_name]
        per_graph.append(
            {
                "graph": graph_name,
                "family": ",".join(sorted(test_graph.tags)) if test_graph.tags else "",
                "ruler": ("directed" if is_semantically_directed(test_graph) else "undirected"),
                "native": float(native_row["best_composite"]),
                "field_best": float(winner["best_composite"]),
                "field_best_engine": winner["engine"],
                "delta": delta,
                "status": status,
            }
        )

    best_or_tied = tallies["strictly_best"] + tallies["tied"]
    summary = {
        "corpus_size": len(names),
        "tie_band": TIE_BAND,
        "best_or_tied": best_or_tied,
        "tallies": tallies,
        "native_dir": str(args.native_dir),
        "field_dir": str(args.field_dir),
        "per_graph": per_graph,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=1))
    print(
        f"[score] best-or-tied {best_or_tied}/{len(names)} "
        f"(best {tallies['strictly_best']}, tied {tallies['tied']}, "
        f"behind {tallies['behind']}, missing {tallies['missing']})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
