"""Evaluate one Dagua pipeline against the frozen r79 baseline store."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

import dagua
from dagua.config import LayoutConfig
from dagua.eval.graphs import TestGraph, get_test_graphs, is_semantically_directed
from dagua.metrics import composite_auto, evaluate

sys.path.insert(0, str(Path(__file__).resolve().parent))

from r79_baseline import (  # noqa: E402
    SEED,
    TIE_BAND,
    graph_best_external,
    json_clean,
    position_relpath,
    safe_component,
)

BASELINE_DIR = Path("eval_output/r79_baseline")
EVIDENCE_PATH = Path(".project-context/research/r79_native/P2_EVIDENCE.md")

TARGET_EXACT_NAMES = {
    "small_world_100",
    "small_world_500",
    "parallel_cycles_4x5",
    "triangular_lattice_36",
    "dense_pair_50",
    "real_karate_34",
    "real_lesmis_77",
    "real_football_115",
    "regular_3_30",
}
TARGET_PREFIXES = ("er_", "rgg_", "r79_undirected_sbm_", "r79_weighted_")
TARGET_TAGS = {
    "undirected",
    "cyclic",
    "lattice",
    "mesh",
    "community",
    "small-world",
    "dense",
    "nonhierarchical",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", default="native_stress", help="Dagua algorithm name.")
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=BASELINE_DIR,
        help="Frozen r79 baseline directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Probe output directory. Defaults to eval_output/r79_probe_<algorithm>.",
    )
    parser.add_argument("--graphs", nargs="+", default=None, help="Optional graph-name filter.")
    return parser.parse_args()


def load_baseline_rows(baseline_dir: Path) -> List[Dict[str, Any]]:
    """Load frozen baseline rows.

    Parameters
    ----------
    baseline_dir : Path
        Directory containing ``results.json``.

    Returns
    -------
    list[dict[str, Any]]
        Baseline result rows.
    """
    with (baseline_dir / "results.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return list(payload["rows"])


def target_graphs(graph_filter: Optional[Sequence[str]] = None) -> List[TestGraph]:
    """Return the r79 target-class graph subset.

    Parameters
    ----------
    graph_filter : sequence[str], optional
        Optional exact graph-name filter.

    Returns
    -------
    list[TestGraph]
        Selected corpus graphs with at most 500 nodes.
    """
    requested = set(graph_filter or [])
    selected: List[TestGraph] = []
    for test_graph in get_test_graphs(max_nodes=500):
        name = test_graph.name
        tags = {str(tag).lower() for tag in test_graph.tags}
        is_target = (
            name in TARGET_EXACT_NAMES
            or any(name.startswith(prefix) for prefix in TARGET_PREFIXES)
            or bool(tags & TARGET_TAGS)
        )
        if requested and name not in requested:
            continue
        if is_target:
            test_graph.graph.compute_node_sizes()
            selected.append(test_graph)
    return sorted(selected, key=lambda item: item.name)


def weighted_distance_matrix(test_graph: TestGraph) -> Optional[np.ndarray]:
    """Compute exact weighted all-pairs distances for weighted graphs.

    Parameters
    ----------
    test_graph : TestGraph
        Test graph with optional edge weights.

    Returns
    -------
    numpy.ndarray | None
        Weighted distance matrix with shape ``[N, N]``, or ``None`` when no
        edge weights are attached.
    """
    graph = test_graph.graph
    edge_weights = getattr(graph, "edge_weights", None)
    if edge_weights is None:
        return None
    num_nodes = int(graph.num_nodes)
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(num_nodes)]
    edges = graph.edge_index.detach().to(device="cpu", dtype=torch.long)
    weights = edge_weights.detach().to(device="cpu", dtype=torch.float64)
    for source, target, weight in zip(edges[0].tolist(), edges[1].tolist(), weights.tolist()):
        if int(source) == int(target):
            continue
        cost = float(weight)
        adjacency[int(source)].append((int(target), cost))
        adjacency[int(target)].append((int(source), cost))
    matrix = np.full((num_nodes, num_nodes), np.inf, dtype=np.float64)
    for source in range(num_nodes):
        matrix[source] = _dijkstra(adjacency, source)
    finite = np.isfinite(matrix)
    fill_value = float(matrix[finite].max()) + 1.0 if bool(finite.any()) else 0.0
    matrix[~finite] = fill_value
    np.fill_diagonal(matrix, 0.0)
    return matrix


def _dijkstra(adjacency: list[list[tuple[int, float]]], source: int) -> np.ndarray:
    """Compute one weighted shortest-path row.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Weighted undirected adjacency list.
    source : int
        Source node index.

    Returns
    -------
    numpy.ndarray
        Distance row with shape ``[N]``.
    """
    import heapq

    distances = np.full((len(adjacency),), np.inf, dtype=np.float64)
    distances[source] = 0.0
    heap: list[tuple[float, int]] = [(0.0, source)]
    while heap:
        distance, node = heapq.heappop(heap)
        if distance > distances[node]:
            continue
        for neighbor, weight in adjacency[node]:
            candidate = distance + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                heapq.heappush(heap, (candidate, neighbor))
    return distances


def normalized_weighted_stress(
    pos: torch.Tensor,
    distances: Optional[np.ndarray],
) -> Optional[float]:
    """Compute scale-normalized weighted stress.

    Parameters
    ----------
    pos : torch.Tensor
        Layout coordinates with shape ``[N, 2]``.
    distances : numpy.ndarray | None
        Weighted graph-distance matrix.

    Returns
    -------
    float | None
        Normalized stress, lower is better, or ``None`` for unweighted graphs.
    """
    if distances is None or distances.shape[0] <= 1:
        return None
    positions = pos.detach().to(device="cpu", dtype=torch.float64).numpy()
    deltas = positions[:, None, :] - positions[None, :, :]
    euclidean = np.sqrt(np.sum(deltas * deltas, axis=2))
    upper = np.triu_indices(distances.shape[0], k=1)
    graph_d = distances[upper]
    layout_d = euclidean[upper]
    mask = np.isfinite(graph_d) & (graph_d > 0.0)
    if not bool(mask.any()):
        return None
    graph_d = graph_d[mask]
    layout_d = layout_d[mask]
    weights = 1.0 / np.square(graph_d)
    numerator = float(np.sum(weights * graph_d * layout_d))
    denominator = float(np.sum(weights * layout_d * layout_d))
    scale = numerator / denominator if denominator > 0.0 else 1.0
    residual = graph_d - scale * layout_d
    return float(np.sum(weights * residual * residual) / max(int(mask.sum()), 1))


def run_probe_row(
    test_graph: TestGraph,
    algorithm: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run one Dagua algorithm on one graph.

    Parameters
    ----------
    test_graph : TestGraph
        Corpus graph.
    algorithm : str
        Pipeline algorithm name.
    output_dir : Path
        Probe output directory.

    Returns
    -------
    dict[str, Any]
        Result row with metrics and persisted position path on success.
    """
    graph = test_graph.graph
    start = time.perf_counter()
    try:
        pos = dagua.layout(graph, LayoutConfig(algorithm=algorithm, seed=SEED, steps=0))
    except Exception as exc:  # noqa: BLE001
        return {
            "graph": test_graph.name,
            "engine": algorithm,
            "status": "ERROR",
            "runtime_s": time.perf_counter() - start,
            "error": f"{type(exc).__name__}: {exc}",
        }
    runtime_s = time.perf_counter() - start
    metrics = evaluate(graph, pos, tier="full")
    composite = composite_auto(metrics, is_semantically_directed(test_graph))
    relpath = position_relpath(test_graph.name, algorithm)
    path = output_dir / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pos.detach().cpu().to(dtype=torch.float32), path)
    weighted_stress = normalized_weighted_stress(pos, weighted_distance_matrix(test_graph))
    return {
        "graph": test_graph.name,
        "engine": algorithm,
        "status": "OK",
        "runtime_s": float(runtime_s),
        "nodes": graph.num_nodes,
        "edges": int(graph.edge_index.shape[1]),
        "metrics": json_clean(metrics),
        "composite": float(composite),
        "weighted_stress": weighted_stress,
        "positions_path": relpath,
    }


def row_map(rows: Iterable[Dict[str, Any]], engine: str) -> Dict[str, Dict[str, Any]]:
    """Map OK rows for one engine by graph name.

    Parameters
    ----------
    rows : iterable[dict[str, Any]]
        Result rows.
    engine : str
        Engine name.

    Returns
    -------
    dict[str, dict[str, Any]]
        OK rows keyed by graph name.
    """
    return {
        str(row["graph"]): row
        for row in rows
        if row.get("engine") == engine and row.get("status") == "OK"
    }


def write_results(output_dir: Path, algorithm: str, rows: Sequence[Dict[str, Any]]) -> None:
    """Write probe results JSON.

    Parameters
    ----------
    output_dir : Path
        Output directory.
    algorithm : str
        Evaluated algorithm.
    rows : sequence[dict[str, Any]]
        Probe rows.

    Returns
    -------
    None
        Writes ``results.json``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": {"algorithm": algorithm, "seed": SEED}, "rows": list(rows)}
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def evidence_rows(
    probe_rows: Sequence[Dict[str, Any]],
    baseline_rows: Sequence[Dict[str, Any]],
    baseline_dir: Path,
    graphs: Sequence[TestGraph],
) -> List[Dict[str, Any]]:
    """Build comparison rows for evidence reporting.

    Parameters
    ----------
    probe_rows : sequence[dict[str, Any]]
        Native-stress probe rows.
    baseline_rows : sequence[dict[str, Any]]
        Frozen baseline rows.
    baseline_dir : Path
        Frozen baseline directory containing position tensors.
    graphs : sequence[TestGraph]
        Target graph metadata used for weighted-stress comparison.

    Returns
    -------
    list[dict[str, Any]]
        Per-graph comparison rows.
    """
    default_rows = row_map(baseline_rows, "dagua")
    best_external = graph_best_external(baseline_rows)
    graph_by_name = {test_graph.name: test_graph for test_graph in graphs}
    comparisons: List[Dict[str, Any]] = []
    for row in sorted(probe_rows, key=lambda item: str(item.get("graph"))):
        if row.get("status") != "OK":
            comparisons.append({"graph": row.get("graph"), "status": row.get("status")})
            continue
        graph = str(row["graph"])
        default = default_rows.get(graph)
        external = best_external.get(graph)
        default_weighted = default_weighted_stress(
            test_graph=graph_by_name[graph],
            default_row=default,
            baseline_dir=baseline_dir,
        )
        comparisons.append(
            {
                "graph": graph,
                "native_stress": float(row["composite"]),
                "default": None if default is None else float(default["composite"]),
                "best_external": None if external is None else float(external["composite"]),
                "best_external_engine": None if external is None else str(external["engine"]),
                "weighted_stress": row.get("weighted_stress"),
                "default_weighted_stress": default_weighted,
            }
        )
    return comparisons


def default_weighted_stress(
    test_graph: TestGraph,
    default_row: Optional[Dict[str, Any]],
    baseline_dir: Path,
) -> Optional[float]:
    """Compute weighted stress for the frozen default Dagua position.

    Parameters
    ----------
    test_graph : TestGraph
        Target graph metadata.
    default_row : dict[str, Any] | None
        Frozen default Dagua row.
    baseline_dir : Path
        Baseline directory containing position tensors.

    Returns
    -------
    float | None
        Weighted stress for default Dagua, or ``None`` when unavailable.
    """
    distances = weighted_distance_matrix(test_graph)
    if distances is None or default_row is None:
        return None
    relpath = default_row.get("positions_path")
    if not relpath:
        return None
    path = baseline_dir / str(relpath)
    if not path.is_file():
        return None
    positions = torch.load(path, map_location="cpu")
    return normalized_weighted_stress(positions, distances)


def write_evidence(
    path: Path,
    algorithm: str,
    comparisons: Sequence[Dict[str, Any]],
    baseline_rows: Sequence[Dict[str, Any]],
) -> None:
    """Write Markdown evidence table.

    Parameters
    ----------
    path : Path
        Markdown output path.
    algorithm : str
        Evaluated algorithm name.
    comparisons : sequence[dict[str, Any]]
        Per-graph comparison rows.
    baseline_rows : sequence[dict[str, Any]]
        Frozen baseline rows used to recover default weighted stress.

    Returns
    -------
    None
        Writes ``path``.
    """
    del baseline_rows
    improved = external_beaten = weighted_better = weighted_total = 0
    ok_count = 0
    lines = [
        "# R79 P2 Native Stress Evidence",
        "",
        f"- Algorithm: {algorithm}",
        f"- Tie band vs external: +/-{TIE_BAND:.1f}",
        "",
        "| Graph | Native Stress | Default Dagua | Best External | External Engine | "
        "Result | Weighted Stress |",
        "| --- | ---: | ---: | ---: | --- | --- | ---: |",
    ]
    for item in comparisons:
        if item.get("status") and item.get("status") != "OK":
            lines.append(f"| {item.get('graph')} | | | | | {item.get('status')} | |")
            continue
        ok_count += 1
        native = float(item["native_stress"])
        default = item.get("default")
        external = item.get("best_external")
        if default is not None and native > float(default):
            improved += 1
        if external is not None and native >= float(external) - TIE_BAND:
            external_beaten += 1
        graph = str(item["graph"])
        default_weighted = item.get("default_weighted_stress")
        weighted_value = item.get("weighted_stress")
        weighted_text = ""
        if weighted_value is not None:
            weighted_total += 1
            if default_weighted is not None and float(weighted_value) < float(default_weighted):
                weighted_better += 1
            weighted_text = (
                f"{float(weighted_value):.6f}"
                if default_weighted is None
                else f"{float(weighted_value):.6f} / {float(default_weighted):.6f}"
            )
        result = "improved" if default is not None and native > float(default) else "not-improved"
        lines.append(
            (
                "| {graph} | {native:.3f} | {default} | {external} | {engine} | "
                "{result} | {weighted} |"
            ).format(
                graph=graph,
                native=native,
                default="" if default is None else f"{float(default):.3f}",
                external="" if external is None else f"{float(external):.3f}",
                engine=item.get("best_external_engine") or "",
                result=result,
                weighted=weighted_text,
            )
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"- Default improvement: {improved}/{ok_count}",
            f"- Within tie band of best external: {external_beaten}/{ok_count}",
            f"- Weighted stress better than default: {weighted_better}/{weighted_total}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the pipeline probe and write evidence artifacts.

    Returns
    -------
    None
        Outputs are written to the configured directories.
    """
    args = parse_args()
    algorithm = str(args.algorithm)
    output_dir = args.output_dir or Path(f"eval_output/r79_probe_{safe_component(algorithm)}")
    baseline_rows = load_baseline_rows(args.baseline_dir)
    graphs = target_graphs(args.graphs)
    probe_rows = [
        run_probe_row(test_graph, algorithm=algorithm, output_dir=output_dir)
        for test_graph in graphs
    ]
    write_results(output_dir=output_dir, algorithm=algorithm, rows=probe_rows)
    comparisons = evidence_rows(
        probe_rows=probe_rows,
        baseline_rows=baseline_rows,
        baseline_dir=args.baseline_dir,
        graphs=graphs,
    )
    write_evidence(
        path=EVIDENCE_PATH,
        algorithm=algorithm,
        comparisons=comparisons,
        baseline_rows=baseline_rows,
    )
    print(f"wrote {output_dir / 'results.json'}")
    print(f"wrote {EVIDENCE_PATH}")


if __name__ == "__main__":
    main()
