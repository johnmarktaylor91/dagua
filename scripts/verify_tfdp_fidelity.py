"""Verify t-FDP pipeline fidelity against the cloned reference adapter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagua.eval.competitors.tfdp_competitor import TFDPCompetitor  # noqa: E402
from dagua.eval.equivalence_metrics import (  # noqa: E402
    compute_equivalence_metrics,
    procrustes_rmsd,
)
from dagua.graph import DaguaGraph  # noqa: E402
from dagua.layout.ops.pipelines.tfdp import layout_tfdp_pipeline  # noqa: E402

DEFAULT_REPORT = ROOT / "docs" / "algorithms" / "tfdp_fidelity.md"
DEFAULT_SEED = 1
DEFAULT_ITERATIONS = 30
BIT_EXACT_THRESHOLD = 1.0e-6
SIMILAR_THRESHOLD = 0.05


def _graph_from_edges(name: str, num_nodes: int, edges: Sequence[Tuple[int, int]]) -> DaguaGraph:
    """Build one deterministic verification graph.

    Parameters
    ----------
    name : str
        Graph label prefix.
    num_nodes : int
        Number of nodes.
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


def _verification_graphs() -> List[Tuple[str, DaguaGraph]]:
    """Build the small-first t-FDP verification corpus.

    Returns
    -------
    list[tuple[str, DaguaGraph]]
        Named graphs used for fidelity reporting.
    """
    grid = [(row * 3 + col, row * 3 + col + 1) for row in range(3) for col in range(2)] + [
        (row * 3 + col, (row + 1) * 3 + col) for row in range(2) for col in range(3)
    ]
    cases = [
        ("single_node", 1, []),
        ("small_chain", 6, [(node, node + 1) for node in range(5)]),
        ("diamond", 4, [(0, 1), (0, 2), (1, 3), (2, 3)]),
        ("cycle_4", 4, [(0, 1), (1, 2), (2, 3), (3, 0)]),
        ("grid_3x3", 9, grid),
        ("disconnected", 5, [(0, 1), (2, 3)]),
    ]
    return [(name, _graph_from_edges(name, num_nodes, edges)) for name, num_nodes, edges in cases]


def _tier(residual: float) -> str:
    """Classify a Procrustes residual.

    Parameters
    ----------
    residual : float
        Rotation-invariant layout residual.

    Returns
    -------
    str
        Fidelity tier label.
    """
    if residual <= BIT_EXACT_THRESHOLD:
        return "BIT/SIMILARITY_EXACT"
    if residual <= SIMILAR_THRESHOLD:
        return "POSITIONAL"
    return "DISTRIBUTIONAL"


def _format_float(value: float) -> str:
    """Format a finite or non-finite float for reports.

    Parameters
    ----------
    value : float
        Numeric value.

    Returns
    -------
    str
        Compact formatted value.
    """
    if not torch.isfinite(torch.tensor(value)):
        return "nan"
    return f"{value:.6g}"


def _write_report(path: Path, rows: list[dict[str, Any]], blocker: str | None) -> None:
    """Write the t-FDP fidelity markdown report.

    Parameters
    ----------
    path : pathlib.Path
        Destination report path.
    rows : list[dict[str, Any]]
        Per-graph metrics.
    blocker : str or None
        Reference blocker text, when reference execution failed.

    Returns
    -------
    None
        Report is written to disk.
    """
    lines = [
        "# t-FDP fidelity",
        "",
        "Implementation: native torch exact t-force loop with PMDS/random initialization.",
        "",
    ]
    if blocker is not None:
        lines.extend(["## Reference blocker", "", blocker, ""])
    lines.extend(
        [
            "## Exact mode",
            "",
            "| graph | residual | tier | stress delta | neighborhood delta |",
            "| --- | ---: | --- | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {graph} | {residual} | {tier} | {stress} | {neighborhood} |".format(
                graph=row["graph"],
                residual=_format_float(float(row["residual"])),
                tier=row["tier"],
                stress=_format_float(float(row["stress_delta"])),
                neighborhood=_format_float(float(row["neighborhood_delta"])),
            )
        )
    lines.extend(
        [
            "",
            "## FFT mode",
            "",
            "The public `force_mode='fft'` hook is present, but native Dagua currently "
            "falls back to the exact force evaluator. The reference FFT path is a "
            "pyFFTW/Numba interpolation implementation and remains a named gap.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> int:
    """Run t-FDP fidelity verification.

    Returns
    -------
    int
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    competitor = TFDPCompetitor()
    rows: list[dict[str, Any]] = []
    blocker: str | None = None
    for name, graph in _verification_graphs():
        actual = layout_tfdp_pipeline(
            graph.edge_index,
            graph.num_nodes,
            seed=args.seed,
            max_iter=args.iterations,
            force_mode="exact",
        ).to(dtype=torch.float64)
        reference = competitor.layout_with_variant(
            graph,
            timeout=120.0,
            seed=args.seed,
            variant_params={"algo": "Exact", "max_iter": args.iterations},
        )
        if reference.pos is None:
            blocker = f"t-FDP reference failed for {name}: {reference.error}"
            residual = float("nan")
            stress_delta = float("nan")
            neighborhood_delta = float("nan")
            tier = "REFERENCE_BLOCKED"
        else:
            residual = procrustes_rmsd(actual.numpy(), reference.pos.numpy())
            metrics = compute_equivalence_metrics(
                actual,
                reference.pos,
                graph.edge_index,
                engine_name="tfdp",
            )
            stress_delta = metrics.stress_rel_delta
            neighborhood_delta = metrics.neighborhood_preservation_delta
            tier = _tier(residual)
        rows.append(
            {
                "graph": name,
                "residual": residual,
                "tier": tier,
                "stress_delta": stress_delta,
                "neighborhood_delta": neighborhood_delta,
            }
        )
        print(
            "{name}: residual={residual} tier={tier} stress_delta={stress} "
            "neighborhood_delta={neighborhood}".format(
                name=name,
                residual=_format_float(float(residual)),
                tier=tier,
                stress=_format_float(float(stress_delta)),
                neighborhood=_format_float(float(neighborhood_delta)),
            )
        )
    _write_report(args.report, rows, blocker)
    print(f"report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
