"""Re-score every stored r83 field layout on the frozen honest ruler."""

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

DEFAULT_POSITION_ROOT = Path("/home/jtaylor/projects/dagua/eval_output")
DEFAULT_REFERENCE_POSITIONS = DEFAULT_POSITION_ROOT / "r81_regate2" / "positions"
DEFAULT_OUTPUT = Path("eval_output/r83_rescore/results.json")
TIE_BAND = 0.5
CLASSICAL_ENGINES = {
    "dagre",
    "elk_layered",
    "graphviz_dot",
    "graphviz_neato",
    "graphviz_sfdp",
    "igraph_kamada_kawai",
    "igraph_sugiyama",
    "nx_spring",
}
REFERENCE_ENGINES = ("dagua", *sorted(CLASSICAL_ENGINES))
MODERN_ENGINES = (
    "classic_sgd2_multi",
    "classic_neulay",
    "classic_umap",
    "classic_tsnet",
    "classic_fa2",
    "classic_drl",
    "classic_fcose",
    "cytoscape_fcose",
    "gephi_yifanhu",
)
ALL_ENGINES = (*REFERENCE_ENGINES, *MODERN_ENGINES)
_POSITION_RE = re.compile(
    r"^(?P<graph>.+)__(?P<engine>" + "|".join(ALL_ENGINES) + r")(?P<variant>.*)\.pt$"
)
_TERM_WEIGHTS = {
    "ksm_score": 25.0,
    "edge_crossing_score": 20.0,
    "node_occlusion_score": 13.0,
    "neighborhood_preservation_score": 12.0,
    "edge_length_deviation_score": 7.0,
    "gabriel_score": 5.0,
    "crossing_angle_score": 5.0,
    "angular_resolution_score": 4.0,
    "path_continuity_score": 4.0,
    "cluster_silhouette_score": 5.0,
    "directed_flow_score": 16.0,
    "depth_order_score": 9.0,
}
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
    parser.add_argument("--position-root", type=Path, default=DEFAULT_POSITION_ROOT)
    parser.add_argument("--reference-positions", type=Path, default=DEFAULT_REFERENCE_POSITIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) // 2))
    return parser.parse_args(argv)


def corpus_names(reference_positions: Path) -> List[str]:
    """Return the exact shared 108-graph corpus from saved position names.

    Parameters
    ----------
    reference_positions : Path
        Directory holding the incumbent and classical saved tensors.

    Returns
    -------
    List[str]
        Sorted shared corpus names.
    """
    names = {
        match.group("graph")
        for path in reference_positions.glob("*.pt")
        if (match := _POSITION_RE.match(path.name)) is not None
        and match.group("engine") in CLASSICAL_ENGINES
    }
    if len(names) != 108:
        raise RuntimeError(f"expected 108 shared r81 graphs, found {len(names)}")
    return sorted(names)


def index_positions(
    root: Path, reference_positions: Path, names: Sequence[str]
) -> Dict[Tuple[str, str], List[str]]:
    """Index all reference and modern position tensors on the corpus.

    Parameters
    ----------
    root : Path
        Evaluation-output root containing benchmark position directories.
    reference_positions : Path
        Directory containing Dagua and classical reference tensors.
    names : Sequence[str]
        Exact graph names in the frozen corpus.

    Returns
    -------
    Dict[Tuple[str, str], List[str]]
        Paths grouped by graph and engine name.
    """
    wanted = set(names)
    indexed: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    candidate_paths = list(reference_positions.glob("*.pt"))
    candidate_paths.extend(root.glob("benchmark_*/positions/*.pt"))
    for path in candidate_paths:
        match = _POSITION_RE.match(path.name)
        if match is None or match.group("graph") not in wanted:
            continue
        engine = match.group("engine")
        is_reference = path.parent == reference_positions
        if (engine in REFERENCE_ENGINES) == is_reference:
            indexed[(match.group("graph"), engine)].append(str(path))
    return {key: sorted(paths) for key, paths in indexed.items()}


def build_graph_map(names: Sequence[str]) -> Dict[str, TestGraph]:
    """Build current corpus graphs and select the frozen r81 names.

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
    """Score every variant and seed for one graph-engine pair.

    Parameters
    ----------
    task : Tuple[str, str, List[str]]
        Graph name, engine name, and all matching tensor paths.

    Returns
    -------
    Dict[str, Any]
        Best current-composite result plus candidate/error counts.
    """
    graph_name, engine, paths = task
    test_graph = _WORKER_GRAPHS[graph_name]
    best_score: Optional[float] = None
    best_path: Optional[str] = None
    best_metrics: Optional[Dict[str, Optional[float]]] = None
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
            best_metrics = {
                name: None if metrics.get(name) is None else float(metrics[name])
                for name in _TERM_WEIGHTS
            }
    return {
        "graph": graph_name,
        "engine": engine,
        "candidate_count": len(paths),
        "scored_count": len(paths) - len(errors),
        "error_count": len(errors),
        "errors": errors[:20],
        "best_composite": best_score,
        "best_path": best_path,
        "best_metrics": best_metrics,
    }


def classify(delta: float) -> str:
    """Classify a Dagua score delta using the frozen tie band.

    Parameters
    ----------
    delta : float
        Dagua composite minus comparison composite.

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


def summarize(
    names: Sequence[str],
    scored_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build uniform honest-ruler comparisons from newly scored rows.

    Parameters
    ----------
    names : Sequence[str]
        Exact shared corpus names.
    scored_rows : Sequence[Dict[str, Any]]
        Best newly scored row for every covered graph-engine pair.

    Returns
    -------
    Dict[str, Any]
        Machine-readable comparison summary.
    """
    by_graph: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_pair: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in scored_rows:
        by_pair[(str(row["graph"]), str(row["engine"]))] = row
        if row["best_composite"] is not None:
            by_graph[str(row["graph"])].append(row)

    per_graph: List[Dict[str, Any]] = []
    for graph_name in names:
        dagua_row = by_pair.get((graph_name, "dagua"))
        if dagua_row is None or dagua_row["best_composite"] is None:
            raise RuntimeError(f"missing scoreable Dagua position for {graph_name}")
        competitors = [row for row in by_graph[graph_name] if row["engine"] != "dagua"]
        if not competitors:
            raise RuntimeError(f"no scoreable competitor position for {graph_name}")
        winner = max(competitors, key=lambda row: float(row["best_composite"]))
        dagua_score = float(dagua_row["best_composite"])
        winner_score = float(winner["best_composite"])
        delta = dagua_score - winner_score
        directed = bool(_declares_hierarchy(_WORKER_GRAPHS[graph_name]))
        per_graph.append(
            {
                "graph": graph_name,
                "ruler": "directed" if directed else "common",
                "dagua": dagua_score,
                "dagua_path": dagua_row["best_path"],
                "dagua_metrics": dagua_row["best_metrics"],
                "competitor_engine": winner["engine"],
                "competitor_best": winner_score,
                "competitor_path": winner["best_path"],
                "competitor_metrics": winner["best_metrics"],
                "delta": delta,
                "status": classify(delta),
            }
        )

    def counts(field: str) -> Dict[str, int]:
        """Count comparison classifications for one result field.

        Parameters
        ----------
        field : str
            Per-graph status field to count.

        Returns
        -------
        Dict[str, int]
            Counts by status plus the combined best-or-tied count.
        """
        result = {
            label: sum(row[field] == label for row in per_graph)
            for label in ("strictly_best", "tied", "behind")
        }
        result["best_or_tied"] = result["strictly_best"] + result["tied"]
        return result

    return {
        "corpus_names": list(names),
        "tie_band": TIE_BAND,
        "full": counts("status"),
        "per_graph": per_graph,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the complete modern-field re-score.

    Parameters
    ----------
    argv : Optional[Sequence[str]], optional
        Explicit CLI arguments, or ``None`` for the process arguments.

    Returns
    -------
    int
        Process exit status.
    """
    args = parse_args(argv)
    set_size_aware_externals(True)
    names = corpus_names(args.reference_positions)
    indexed = index_positions(args.position_root, args.reference_positions, names)
    graphs = build_graph_map(names)
    init_worker(graphs)
    tasks = [(graph, engine, paths) for (graph, engine), paths in sorted(indexed.items())]
    context = mp.get_context("fork")
    with context.Pool(args.workers, initializer=init_worker, initargs=(graphs,)) as pool:
        modern_rows = list(pool.imap_unordered(score_group, tasks, chunksize=1))
    modern_rows.sort(key=lambda row: (str(row["graph"]), str(row["engine"])))
    result = {
        "metadata": {
            "position_root": str(args.position_root),
            "reference_positions": str(args.reference_positions),
            "candidate_count": sum(len(paths) for paths in indexed.values()),
            "workers": args.workers,
            "policy": "evaluate(tier=full) + composite_auto; size-aware externals; overlap=prism",
        },
        "scored_rows": modern_rows,
        "coverage": [
            {
                "graph": graph,
                "engine": engine,
                "candidate_count": len(indexed.get((graph, engine), [])),
                "present": bool(indexed.get((graph, engine))),
            }
            for graph in names
            for engine in ALL_ENGINES
        ],
        "summary": summarize(names, modern_rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
