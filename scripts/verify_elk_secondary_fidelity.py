"""Verify ELK secondary pipelines against elkjs references."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagua.eval.competitors.elk_competitor import (  # noqa: E402
    ElkForce,
    ElkMrTree,
    ElkRadial,
    ElkStress,
)
from dagua.eval.equivalence_metrics import anisotropic_procrustes, procrustes_rmsd  # noqa: E402
from dagua.graph import DaguaGraph  # noqa: E402
from dagua.layout.ops.pipelines.elk_force import layout_elk_force_pipeline  # noqa: E402
from dagua.layout.ops.pipelines.elk_mrtree import layout_elk_mrtree_pipeline  # noqa: E402
from dagua.layout.ops.pipelines.elk_radial import layout_elk_radial_pipeline  # noqa: E402
from dagua.layout.ops.pipelines.elk_stress import layout_elk_stress_pipeline  # noqa: E402

DEFAULT_CACHE = ROOT / "tests" / "fixtures" / "elk_secondary_reference_layouts.json"
DEFAULT_REPORT = ROOT / "docs" / "algorithms" / "elk_secondary_fidelity.md"
BIT_EXACT_THRESHOLD = 1.0e-9
POSITIONAL_THRESHOLD = 0.5
DEFAULT_SEED = 1


def _graph_from_edges(name: str, num_nodes: int, edges: Sequence[Tuple[int, int]]) -> DaguaGraph:
    """Build one deterministic verification graph.

    Parameters
    ----------
    name : str
        Graph name used for node labels.
    num_nodes : int
        Number of graph nodes.
    edges : sequence[tuple[int, int]]
        Edge pairs.

    Returns
    -------
    DaguaGraph
        Graph with explicit nodes ``0..N-1``.
    """
    graph = DaguaGraph()
    for node in range(num_nodes):
        graph.add_node(node, label=f"{name}_{node}")
    for source, target in edges:
        graph.add_edge(source, target)
    graph.compute_node_sizes()
    return graph


def _verification_graphs(algorithm: str) -> List[Tuple[str, DaguaGraph]]:
    """Return the small-first ELK secondary verification corpus.

    Parameters
    ----------
    algorithm : str
        Algorithm name; tree-only ELK algorithms receive only tree fixtures.

    Returns
    -------
    list[tuple[str, DaguaGraph]]
        Named graph fixtures.
    """
    tree_cases = [
        ("single", _graph_from_edges("single", 1, [])),
        ("chain5", _graph_from_edges("chain5", 5, [(0, 1), (1, 2), (2, 3), (3, 4)])),
        (
            "binary7",
            _graph_from_edges(
                "binary7",
                7,
                [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)],
            ),
        ),
    ]
    if algorithm in {"elk_mrtree", "elk_radial"}:
        return tree_cases
    return [
        *tree_cases,
        ("diamond", _graph_from_edges("diamond", 4, [(0, 1), (0, 2), (1, 3), (2, 3)])),
        ("cycle4", _graph_from_edges("cycle4", 4, [(0, 1), (1, 2), (2, 3), (3, 0)])),
    ]


def _competitors() -> Dict[str, Any]:
    """Return ELK reference competitors keyed by pipeline name.

    Returns
    -------
    dict[str, Any]
        Instantiated reference adapters.
    """
    return {
        "elk_force": ElkForce(),
        "elk_stress": ElkStress(),
        "elk_mrtree": ElkMrTree(),
        "elk_radial": ElkRadial(),
    }


def _cache_reference_layouts(path: Path, refresh: bool) -> Dict[str, Any]:
    """Load or create cached ELK reference layouts.

    Parameters
    ----------
    path : pathlib.Path
        JSON cache path.
    refresh : bool
        Whether to discard the existing cache.

    Returns
    -------
    dict[str, Any]
        Reference payload.

    Raises
    ------
    RuntimeError
        If elkjs is unavailable or a reference layout fails.
    """
    if path.exists() and not refresh:
        return json.loads(path.read_text())

    competitors = _competitors()
    if not all(competitor.available() for competitor in competitors.values()):
        raise RuntimeError("elkjs is not available to Node.")

    rows: List[Dict[str, Any]] = []
    for algo, competitor in competitors.items():
        for graph_name, graph in _verification_graphs(algo):
            result = competitor.layout(graph, seed=DEFAULT_SEED)
            if result.pos is None:
                raise RuntimeError(f"{algo} reference failed for {graph_name}: {result.error}")
            rows.append(
                {
                    "algorithm": algo,
                    "name": graph_name,
                    "num_nodes": graph.num_nodes,
                    "edges": graph.edge_index.t().tolist() if graph.edge_index.numel() else [],
                    "reference_positions": result.pos.tolist(),
                }
            )
    payload = {"reference": "elkjs", "seed": DEFAULT_SEED, "graphs": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _edge_index(edges: Sequence[Sequence[int]]) -> torch.Tensor:
    """Convert cached edge pairs to edge-index tensor.

    Parameters
    ----------
    edges : sequence[sequence[int]]
        Cached source-target pairs.

    Returns
    -------
    torch.Tensor
        Long edge-index tensor with shape ``[2, E]``.
    """
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _pipeline_positions(algorithm: str, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Run one local ELK secondary pipeline.

    Parameters
    ----------
    algorithm : str
        Pipeline algorithm name.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    node_sizes = torch.full((num_nodes, 2), 120.0, dtype=torch.float64)
    if algorithm == "elk_force":
        return layout_elk_force_pipeline(edge_index, num_nodes, node_sizes, seed=DEFAULT_SEED)
    if algorithm == "elk_stress":
        return layout_elk_stress_pipeline(edge_index, num_nodes, node_sizes)
    if algorithm == "elk_mrtree":
        return layout_elk_mrtree_pipeline(edge_index, num_nodes, node_sizes, roots=[0])
    if algorithm == "elk_radial":
        return layout_elk_radial_pipeline(edge_index, num_nodes, node_sizes, roots=[0])
    raise ValueError(f"Unknown algorithm: {algorithm}")


def _tier(residual: float) -> str:
    """Classify one residual.

    Parameters
    ----------
    residual : float
        Similarity-invariant residual.

    Returns
    -------
    str
        Fidelity tier label.
    """
    if residual < BIT_EXACT_THRESHOLD:
        return "bit/similarity-exact"
    if residual < POSITIONAL_THRESHOLD:
        return "positional"
    return "distributional"


def _evaluate(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Evaluate cached references against local pipelines.

    Parameters
    ----------
    payload : dict[str, Any]
        Cached reference payload.

    Returns
    -------
    list[dict[str, Any]]
        Per-algorithm, per-graph metric rows.
    """
    rows: List[Dict[str, Any]] = []
    for entry in payload["graphs"]:
        algorithm = str(entry["algorithm"])
        edge_index = _edge_index(entry["edges"])
        num_nodes = int(entry["num_nodes"])
        reference = torch.tensor(entry["reference_positions"], dtype=torch.float64)
        actual = _pipeline_positions(algorithm, edge_index, num_nodes)
        residual = procrustes_rmsd(actual, reference)
        anisotropic = anisotropic_procrustes(actual, reference)
        rows.append(
            {
                "algorithm": algorithm,
                "name": entry["name"],
                "num_nodes": num_nodes,
                "num_edges": int(edge_index.shape[1]),
                "d_R": residual,
                "d_A": anisotropic["anisotropic_rmsd"],
                "tier": _tier(residual),
            }
        )
    return rows


def _write_report(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Write the Markdown fidelity report.

    Parameters
    ----------
    path : pathlib.Path
        Output report path.
    rows : sequence[dict[str, Any]]
        Metric rows from :func:`_evaluate`.

    Returns
    -------
    None
        The report file is overwritten.
    """
    lines = [
        "# ELK Secondary Fidelity",
        "",
        "Reference: `elkjs` with flat graphs, seed `1`, and pinned secondary algorithms.",
        "Runtime pipelines do not import `elkjs`, spawn Node, or call the reference adapter.",
        "",
        "| algorithm | graph | N | E | d_R | d_A | tier |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        d_a = row["d_A"]
        d_a_text = "nan" if d_a is None else f"{float(d_a):.3e}"
        lines.append(
            f"| {row['algorithm']} | {row['name']} | {row['num_nodes']} | {row['num_edges']} | "
            f"{float(row['d_R']):.3e} | {d_a_text} | {row['tier']} |"
        )
    lines.extend(
        [
            "",
            "## Residuals",
            "",
            "- `elk_force`: first-divergent stage is initial graph import / node "
            "micro-layout; local port matches the documented Eades/FR displacement "
            "loop but not ELK's pre-layout coordinates.",
            "- `elk_stress`: first-divergent stage is initial graph import; the "
            "majorization update follows the ELK loop, but starts from local "
            "deterministic coordinates.",
            "- `elk_mrtree`: first-divergent stage is ELK treeification / ordering / "
            "compaction; local implementation uses the existing Reingold-Tilford "
            "tidy-tree op.",
            "- `elk_radial`: first-divergent stage is ELK radial treeification / "
            "angular ordering; local implementation is distinct from `radial_tree` "
            "and uses concentric RT depths.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the ELK secondary fidelity verifier.

    Parameters
    ----------
    argv : sequence[str] or None, default=None
        Optional command-line arguments.

    Returns
    -------
    int
        Process exit status.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)

    payload = _cache_reference_layouts(args.cache, args.refresh)
    rows = _evaluate(payload)
    _write_report(args.report, rows)
    for row in rows:
        print(
            f"{row['algorithm']} {row['name']}: d_R={float(row['d_R']):.3e} "
            f"d_A={float(row['d_A']) if row['d_A'] is not None else float('nan'):.3e} "
            f"tier={row['tier']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
