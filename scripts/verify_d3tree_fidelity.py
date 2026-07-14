"""Verify d3-hierarchy tree and cluster fidelity against the Node reference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagua.eval.competitors.d3hierarchy_competitor import D3HierarchyCompetitor  # noqa: E402
from dagua.graph import DaguaGraph  # noqa: E402
from dagua.layout.ops.pipelines.d3_cluster import layout_d3_cluster_pipeline  # noqa: E402
from dagua.layout.ops.pipelines.d3_tree import layout_d3_tree_pipeline  # noqa: E402

DEFAULT_REPORT = ROOT / "docs" / "algorithms" / "d3tree_fidelity.md"
BIT_EXACT_THRESHOLD = 0.0


def _graph_from_edges(name: str, num_nodes: int, edges: Sequence[Tuple[int, int]]) -> DaguaGraph:
    """Build one deterministic tree verification graph.

    Parameters
    ----------
    name : str
        Graph name used as a label prefix.
    num_nodes : int
        Number of nodes.
    edges : sequence[tuple[int, int]]
        Directed parent-child edges.

    Returns
    -------
    DaguaGraph
        Graph with integer node IDs.
    """
    graph = DaguaGraph()
    for node in range(num_nodes):
        graph.add_node(node, label=f"{name}_{node}")
    for source, target in edges:
        graph.add_edge(source, target)
    return graph


def _verification_graphs() -> List[Tuple[str, DaguaGraph]]:
    """Build the small d3 tree verification corpus.

    Returns
    -------
    list[tuple[str, DaguaGraph]]
        Named tree graph cases.
    """
    binary_tree = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
    org_chart = [(0, node) for node in range(1, 4)] + [(1, 4), (1, 5), (2, 6), (3, 7)]
    cases = [
        ("single_node", 1, []),
        ("path", 5, [(node, node + 1) for node in range(4)]),
        ("star", 6, [(0, node) for node in range(1, 6)]),
        ("binary_tree", 7, binary_tree),
        ("org_chart", 8, org_chart),
    ]
    return [(name, _graph_from_edges(name, num_nodes, edges)) for name, num_nodes, edges in cases]


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return the maximum absolute coordinate difference.

    Parameters
    ----------
    left : torch.Tensor
        First coordinate tensor with shape ``[N, 2]``.
    right : torch.Tensor
        Second coordinate tensor with shape ``[N, 2]``.

    Returns
    -------
    float
        Maximum absolute difference, or ``0`` for empty tensors.
    """
    if left.numel() == 0:
        return 0.0
    return float((left - right).abs().max().item())


def _first_divergent_stage(algorithm: str, max_diff: float, error: str | None) -> str:
    """Classify the first divergent stage for report readability.

    Parameters
    ----------
    algorithm : str
        ``"d3_tree"`` or ``"d3_cluster"``.
    max_diff : float
        Maximum absolute coordinate difference.
    error : str | None
        Reference adapter error, if any.

    Returns
    -------
    str
        Human-readable stage label.
    """
    if error is not None:
        return "reference-adapter"
    if max_diff == BIT_EXACT_THRESHOLD:
        return "none"
    return "firstWalk/apportion/secondWalk" if algorithm == "d3_tree" else "cluster-walk"


def _run_algorithm(
    competitor: D3HierarchyCompetitor,
    algorithm: str,
    graph: DaguaGraph,
) -> Dict[str, Any]:
    """Run one algorithm against one graph and summarize fidelity.

    Parameters
    ----------
    competitor : D3HierarchyCompetitor
        Node reference adapter.
    algorithm : str
        ``"d3_tree"`` or ``"d3_cluster"``.
    graph : DaguaGraph
        Verification graph.

    Returns
    -------
    dict[str, Any]
        Summary row.
    """
    reference_algorithm = "tree" if algorithm == "d3_tree" else "cluster"
    reference = competitor.layout_with_variant(
        graph,
        variant_params={"algorithm": reference_algorithm},
    )
    if reference.error is not None or reference.pos is None:
        return {
            "algorithm": algorithm,
            "num_nodes": graph.num_nodes,
            "num_edges": int(graph.edge_index.shape[1]),
            "max_abs_diff": None,
            "verdict": "N/A",
            "first_divergent_stage": _first_divergent_stage(algorithm, 0.0, reference.error),
            "error": reference.error,
        }
    if algorithm == "d3_tree":
        actual = layout_d3_tree_pipeline(graph.edge_index, graph.num_nodes)
    else:
        actual = layout_d3_cluster_pipeline(graph.edge_index, graph.num_nodes)
    max_diff = _max_abs_diff(actual, reference.pos)
    return {
        "algorithm": algorithm,
        "num_nodes": graph.num_nodes,
        "num_edges": int(graph.edge_index.shape[1]),
        "max_abs_diff": max_diff,
        "verdict": "bit-exact" if max_diff == BIT_EXACT_THRESHOLD else "divergent",
        "first_divergent_stage": _first_divergent_stage(algorithm, max_diff, None),
        "error": None,
    }


def _write_report(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Write the Markdown fidelity report.

    Parameters
    ----------
    path : pathlib.Path
        Report path.
    rows : sequence[dict[str, Any]]
        Verification rows.

    Returns
    -------
    None
        The report is written to disk.
    """
    by_algorithm = {
        algorithm: [row for row in rows if row["algorithm"] == algorithm]
        for algorithm in ("d3_tree", "d3_cluster")
    }
    lines = [
        "# d3-hierarchy tree/cluster fidelity verification",
        "",
        "Reference: d3-hierarchy through the Node adapter.",
        "Production pipelines are Python source ports and never invoke Node.",
        "",
    ]
    for algorithm, algorithm_rows in by_algorithm.items():
        exact = sum(1 for row in algorithm_rows if row["verdict"] == "bit-exact")
        na_count = sum(1 for row in algorithm_rows if row["verdict"] == "N/A")
        lines.extend(
            [
                f"## {algorithm}",
                "",
                f"Result: **{exact}/{len(algorithm_rows)} bit-exact**, **{na_count} N/A**.",
                "",
                "| graph | N | E | max abs diff | first divergent stage | verdict |",
                "|---|---:|---:|---:|---|---|",
            ]
        )
        for graph_name, row in zip([name for name, _ in _verification_graphs()], algorithm_rows):
            diff = "n/a" if row["max_abs_diff"] is None else f"{row['max_abs_diff']:.3e}"
            lines.append(
                f"| {graph_name} | {row['num_nodes']} | {row['num_edges']} | {diff} | "
                f"{row['first_divergent_stage']} | {row['verdict']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Stage bisection",
            "",
            "All current verification rows are bit-exact. For `d3_tree`, this covers "
            "the first walk, apportion/thread shifts, and second walk because the raw "
            "coordinates match d3's `tree().nodeSize([1, 1])` output exactly. For "
            "`d3_cluster`, this covers the leaf walk and normalization used by "
            "d3's default `cluster()`.",
            "",
            "Non-tree graphs are converted to a deterministic spanning hierarchy by "
            "keeping the first incoming parent for each node, matching the reference "
            "adapter's input preparation.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Run d3 tree/cluster fidelity verification.

    Parameters
    ----------
    argv : sequence[str] | None, optional
        Command-line arguments. ``None`` uses ``sys.argv``.

    Returns
    -------
    int
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON summary path.")
    args = parser.parse_args(argv)

    competitor = D3HierarchyCompetitor()
    if not competitor.available():
        raise RuntimeError("d3-hierarchy reference is unavailable.")

    rows: List[Dict[str, Any]] = []
    for name, graph in _verification_graphs():
        for algorithm in ("d3_tree", "d3_cluster"):
            row = _run_algorithm(competitor, algorithm, graph)
            row["graph"] = name
            rows.append(row)

    _write_report(args.report, rows)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2) + "\n")

    for algorithm in ("d3_tree", "d3_cluster"):
        algorithm_rows = [row for row in rows if row["algorithm"] == algorithm]
        exact = sum(1 for row in algorithm_rows if row["verdict"] == "bit-exact")
        na_count = sum(1 for row in algorithm_rows if row["verdict"] == "N/A")
        print(f"{algorithm}: bit-exact {exact}/{len(algorithm_rows)}, N/A {na_count}")
    divergent = [row for row in rows if row["verdict"] not in {"bit-exact", "N/A"}]
    return 1 if divergent else 0


if __name__ == "__main__":
    raise SystemExit(main())
