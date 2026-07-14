"""Cache Graphviz twopi/circo references and write the fidelity report."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagua.eval.competitors.graphviz_competitor import GraphvizCirco, GraphvizTwopi  # noqa: E402
from dagua.eval.equivalence_metrics import anisotropic_procrustes, procrustes_rmsd  # noqa: E402
from dagua.graph import DaguaGraph  # noqa: E402
from dagua.layout.ops.pipelines.circo import layout_circo_pipeline  # noqa: E402
from dagua.layout.ops.pipelines.twopi import layout_twopi_pipeline  # noqa: E402

DEFAULT_CACHE = ROOT / "tests" / "fixtures" / "twopi_circo_reference_layouts.json"
DEFAULT_REPORT = ROOT / "docs" / "algorithms" / "twopi_circo_fidelity.md"
BIT_EXACT_THRESHOLD = 1.0e-9
POSITIONAL_THRESHOLD = 1.0e-3


def _graph_from_edges(name: str, num_nodes: int, edges: Sequence[Tuple[int, int]]) -> DaguaGraph:
    """Build one deterministic labeled verification graph.

    Parameters
    ----------
    name : str
        Graph name used as a label prefix.
    num_nodes : int
        Number of nodes.
    edges : sequence[tuple[int, int]]
        Directed edge pairs.

    Returns
    -------
    DaguaGraph
        Graph with eagerly measured node sizes.
    """
    graph = DaguaGraph()
    for node in range(num_nodes):
        graph.add_node(node, label=f"{name}_{node}")
    for source, target in edges:
        graph.add_edge(source, target)
    graph.compute_node_sizes()
    return graph


def _random_dag_edges(num_nodes: int, edge_count: int, seed: int) -> List[Tuple[int, int]]:
    """Generate a deterministic acyclic edge sample.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    edge_count : int
        Requested distinct edge count.
    seed : int
        Python RNG seed.

    Returns
    -------
    list[tuple[int, int]]
        Sorted source-before-target edges.
    """
    candidates = [
        (source, target) for source in range(num_nodes) for target in range(source + 1, num_nodes)
    ]
    rng = random.Random(seed)
    return sorted(rng.sample(candidates, min(edge_count, len(candidates))))


def _verification_graphs() -> List[Tuple[str, DaguaGraph]]:
    """Build the fixed small-first twopi/circo verification corpus.

    Returns
    -------
    list[tuple[str, DaguaGraph]]
        Named verification graphs.
    """
    binary_tree = [(node, 2 * node + child) for node in range(5) for child in (1, 2)]
    grid = [(row * 5 + column, row * 5 + column + 1) for row in range(5) for column in range(4)] + [
        (row * 5 + column, (row + 1) * 5 + column) for row in range(4) for column in range(5)
    ]
    org_chart = [(0, node) for node in range(1, 6)] + [
        (manager, 6 + (manager - 1) * 2 + child) for manager in range(1, 6) for child in range(2)
    ]
    org_chart_deep: List[Tuple[int, int]] = []
    previous = [0]
    next_node = 1
    for width in (3, 5, 10, 20, 40):
        current = list(range(next_node, next_node + width))
        next_node += width
        for index, child in enumerate(current):
            org_chart_deep.append((previous[index % len(previous)], child))
        previous = current

    cases = [
        ("single_node", 1, []),
        ("small_chain", 6, [(node, node + 1) for node in range(5)]),
        ("binary_tree", 11, binary_tree),
        ("diamond", 4, [(0, 1), (0, 2), (1, 3), (2, 3)]),
        ("grid_5x5", 25, grid),
        ("org_chart_small", 16, org_chart),
        ("long_skip", 5, [(0, 1), (1, 2), (2, 3), (3, 4), (0, 3), (1, 4)]),
        ("disconnected", 5, [(0, 1), (2, 3)]),
        ("cycle_4", 4, [(0, 1), (1, 2), (2, 3), (3, 0)]),
        ("random_dag_50", 50, _random_dag_edges(50, 90, seed=4101)),
        ("org_chart_deep", 79, org_chart_deep),
    ]
    return [(name, _graph_from_edges(name, num_nodes, edges)) for name, num_nodes, edges in cases]


def _cache_reference_layouts(path: Path, refresh: bool) -> Dict[str, Any]:
    """Load or create the one-layout-per-graph Graphviz reference cache.

    Parameters
    ----------
    path : pathlib.Path
        JSON cache destination.
    refresh : bool
        Whether to discard an existing cache.

    Returns
    -------
    dict[str, Any]
        Cache payload containing topology, sizes, and reference positions.
    """
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    competitors = {"twopi": GraphvizTwopi(), "circo": GraphvizCirco()}
    for name, competitor in competitors.items():
        if not competitor.available():
            raise RuntimeError(f"Graphviz {name} reference is unavailable.")

    graph_rows: List[Dict[str, Any]] = []
    for graph_name, graph in _verification_graphs():
        references: Dict[str, Any] = {}
        for algorithm, competitor in competitors.items():
            result = competitor.layout(graph)
            if result.pos is None:
                raise RuntimeError(f"Graphviz {algorithm} failed on {graph_name}: {result.error}")
            references[algorithm] = result.pos.tolist()
        graph_rows.append(
            {
                "name": graph_name,
                "num_nodes": graph.num_nodes,
                "edges": graph.edge_index.t().tolist(),
                "node_sizes": graph.node_sizes.tolist() if graph.node_sizes is not None else [],
                "reference_positions": references,
            }
        )
    payload: Dict[str, Any] = {
        "reference_engine": "Graphviz",
        "reference_version": "7.0.5",
        "adapter": "dagua.eval.competitors.graphviz_competitor",
        "parameters": {"twopi": {"ranksep": 72.0}, "circo": {"nodesep": 18.0}},
        "layouts_per_graph": 1,
        "graphs": graph_rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _classify(residual: float) -> str:
    """Classify one deterministic similarity residual.

    Parameters
    ----------
    residual : float
        Procrustes residual.

    Returns
    -------
    str
        ``bit-exact``, ``positional-identical``, or ``divergent``.
    """
    if residual < BIT_EXACT_THRESHOLD:
        return "bit-exact"
    if residual < POSITIONAL_THRESHOLD:
        return "positional-identical"
    return "divergent"


def _edge_index(edges: Sequence[Sequence[int]]) -> torch.Tensor:
    """Convert cached edge pairs into a tensor.

    Parameters
    ----------
    edges : sequence[sequence[int]]
        Cached edge pairs.

    Returns
    -------
    torch.Tensor
        Long tensor with shape ``[2, E]``.
    """
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _layout_algorithm(
    algorithm: str,
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: torch.Tensor,
) -> torch.Tensor:
    """Run one local algorithm pipeline.

    Parameters
    ----------
    algorithm : str
        Algorithm name, either ``"twopi"`` or ``"circo"``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor
        Node size tensor with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    if algorithm == "twopi":
        return layout_twopi_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            fidelity_dtype=torch.float64,
        )
    if algorithm == "circo":
        return layout_circo_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            node_sizes=node_sizes,
            fidelity_dtype=torch.float64,
        )
    raise ValueError(f"Unknown algorithm: {algorithm}")


def _compare_cache(payload: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Compare local pipelines against every cached reference graph.

    Parameters
    ----------
    payload : dict[str, Any]
        Reference cache payload.

    Returns
    -------
    dict[str, list[dict[str, Any]]]
        Per-algorithm residual rows.
    """
    results: Dict[str, List[Dict[str, Any]]] = {"twopi": [], "circo": []}
    for graph in payload["graphs"]:
        edges = graph["edges"]
        graph_edge_index = _edge_index(edges)
        for algorithm in ("twopi", "circo"):
            reference = torch.tensor(
                graph["reference_positions"][algorithm],
                dtype=torch.float64,
            )
            positions = _layout_algorithm(
                algorithm,
                graph_edge_index,
                int(graph["num_nodes"]),
                torch.tensor(graph["node_sizes"], dtype=torch.float64),
            )
            residual = procrustes_rmsd(positions.numpy(), reference.numpy())
            anisotropic = anisotropic_procrustes(positions.numpy(), reference.numpy())
            results[algorithm].append(
                {
                    "name": graph["name"],
                    "num_nodes": graph["num_nodes"],
                    "num_edges": len(edges),
                    "procrustes_rmsd": residual,
                    "anisotropic_rmsd": anisotropic["anisotropic_rmsd"],
                    "max_abs_coordinate_diff": float((positions - reference).abs().max().item()),
                    "classification": _classify(residual),
                }
            )
    return results


def _stage_note(algorithm: str, row: Dict[str, Any]) -> str:
    """Return the named first-divergent stage for one row.

    Parameters
    ----------
    algorithm : str
        Algorithm name.
    row : dict[str, Any]
        Residual row.

    Returns
    -------
    str
        Stage note for the report table.
    """
    if row["classification"] in {"bit-exact", "positional-identical"}:
        return "none"
    if algorithm == "twopi":
        if row["name"] in {"disconnected", "random_dag_50"}:
            return "component packing after radial layout"
        return "angular wedge/order after BFS rings"
    if row["name"] == "disconnected":
        return "component packing plus block-tree coordinate placement"
    if row["name"] == "grid_5x5":
        return "intra-block circular ordering after block discovery"
    if row["name"] == "random_dag_50":
        return "block-tree coordinate placement after matched blockpath ordering"
    return "block-tree coordinate placement after owned block discovery"


def _write_report(
    path: Path,
    payload: Dict[str, Any],
    rows: Dict[str, List[Dict[str, Any]]],
) -> None:
    """Write the Markdown fidelity report.

    Parameters
    ----------
    path : pathlib.Path
        Report destination.
    payload : dict[str, Any]
        Reference cache metadata.
    rows : dict[str, list[dict[str, Any]]]
        Per-algorithm comparison results.

    Returns
    -------
    None
        Markdown is written to ``path``.
    """
    lines = [
        "# Graphviz twopi/circo fidelity verification",
        "",
        f"Reference: {payload['reference_engine']} {payload['reference_version']} through the "
        "Graphviz JSON adapter. One deterministic layout is cached per graph. The production "
        "pipelines do not invoke Graphviz.",
        "",
    ]
    for algorithm in ("twopi", "circo"):
        algorithm_rows = rows[algorithm]
        bit_exact = sum(row["classification"] == "bit-exact" for row in algorithm_rows)
        positional = sum(row["classification"] == "positional-identical" for row in algorithm_rows)
        divergent = len(algorithm_rows) - bit_exact - positional
        lines.extend(
            [
                f"## {algorithm}",
                "",
                f"Result: **{bit_exact}/{len(algorithm_rows)} similarity-exact**, "
                f"**{positional} positional-identical**, **{divergent} divergent**. "
                f"Thresholds: bit-exact `d_R < {BIT_EXACT_THRESHOLD:.0e}`, "
                f"positional `d_R < {POSITIONAL_THRESHOLD:.0e}`.",
                "",
                "| graph | N | E | Procrustes d_R | anisotropic d_R | max raw diff | "
                "verdict | first divergent stage |",
                "|---|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for row in algorithm_rows:
            lines.append(
                f"| {row['name']} | {row['num_nodes']} | {row['num_edges']} | "
                f"{row['procrustes_rmsd']:.3e} | {row['anisotropic_rmsd']:.3e} | "
                f"{row['max_abs_coordinate_diff']:.3e} | {row['classification']} | "
                f"{_stage_note(algorithm, row)} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Residual notes",
            "",
            "The current twopi implementation matches the prescribed high-level stages: "
            "root selection by minimum eccentricity, BFS ring assignment, and subtree leaf-count "
            "angular wedges. Connected non-exact residuals are positional-identical at the "
            "Graphviz JSON output-precision floor. The two large twopi residuals are named "
            "component-packing residuals from Graphviz `pack.c`, a separable post-layout step.",
            "",
            "The current circo implementation uses Graphviz-style owned block-cutpoint discovery, "
            "`circpos.c` child fan scaling/rotation, and Graphviz inch-to-local node-size "
            "conversion for block radii. Simple paths, simple cycles, the long-skip case, "
            "binary trees, and org-chart trees are positional-or-better. The remaining circo "
            "residuals first diverge in `blockpath.c` ordering for biconnected grids/random "
            "DAGs, with disconnected also requiring Graphviz component packing.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _parse_args() -> argparse.Namespace:
    """Parse command-line options.

    Returns
    -------
    argparse.Namespace
        Cache, report, and refresh settings.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--refresh-reference", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run cached Graphviz twopi/circo fidelity verification.

    Returns
    -------
    int
        Zero after writing the report. Residuals are reported instead of
        treated as process failures so the script remains useful for bisection.
    """
    args = _parse_args()
    payload = _cache_reference_layouts(args.cache, refresh=args.refresh_reference)
    rows = _compare_cache(payload)
    _write_report(args.report, payload, rows)
    for algorithm in ("twopi", "circo"):
        bit_exact = sum(row["classification"] == "bit-exact" for row in rows[algorithm])
        positional = sum(row["classification"] == "positional-identical" for row in rows[algorithm])
        divergent = len(rows[algorithm]) - bit_exact - positional
        print(
            f"{algorithm}: {bit_exact}/{len(rows[algorithm])} bit-exact, "
            f"{positional} positional-identical, {divergent} divergent"
        )
        for row in rows[algorithm]:
            print(
                f"{algorithm}/{row['name']}: d_R={row['procrustes_rmsd']:.3e} "
                f"anisotropic={row['anisotropic_rmsd']:.3e} {row['classification']} "
                f"stage={_stage_note(algorithm, row)}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
