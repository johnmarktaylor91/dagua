"""Verify sparse-stress pipeline fidelity against the Java reference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagua.eval.competitors.sparse_stress_competitor import SparseStressCompetitor  # noqa: E402
from dagua.eval.equivalence_metrics import (  # noqa: E402
    compute_equivalence_metrics,
    procrustes_rmsd,
)
from dagua.graph import DaguaGraph  # noqa: E402
from dagua.layout.ops.pipelines.sparse_stress import layout_sparse_stress_pipeline  # noqa: E402

DEFAULT_REPORT = ROOT / "docs" / "algorithms" / "sparse_stress_fidelity.md"
DEFAULT_SEED = 3
DEFAULT_ITERATIONS = 20
DEFAULT_PIVOTS = 4
DEFAULT_MDS_PIVOTS = 4
BIT_EXACT_THRESHOLD = 1.0e-6
POSITIONAL_THRESHOLD = 0.05


def _graph_from_edges(name: str, num_nodes: int, edges: Sequence[Tuple[int, int]]) -> DaguaGraph:
    """Build a deterministic connected verification graph.

    Parameters
    ----------
    name : str
        Graph label prefix.
    num_nodes : int
        Number of nodes.
    edges : sequence[tuple[int, int]]
        Undirected edge pairs.

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
    """Return the small-first sparse-stress fidelity corpus.

    Returns
    -------
    list[tuple[str, DaguaGraph]]
        Named connected graphs for reference comparison.
    """
    grid = [(row * 3 + col, row * 3 + col + 1) for row in range(3) for col in range(2)] + [
        (row * 3 + col, (row + 1) * 3 + col) for row in range(2) for col in range(3)
    ]
    complete_5 = [(source, target) for source in range(5) for target in range(source + 1, 5)]
    wheel_6 = [(0, node) for node in range(1, 6)] + [(node, 1 + (node % 5)) for node in range(1, 6)]
    cases = [
        ("diamond", 4, [(0, 1), (0, 2), (1, 3), (2, 3)]),
        ("complete_5", 5, complete_5),
        ("wheel_6", 6, wheel_6),
        ("grid_3x3", 9, grid),
    ]
    return [(name, _graph_from_edges(name, num_nodes, edges)) for name, num_nodes, edges in cases]


def _tier(residual: float) -> str:
    """Classify a rotation-invariant residual.

    Parameters
    ----------
    residual : float
        Procrustes RMSD.

    Returns
    -------
    str
        Fidelity tier label.
    """
    if residual <= BIT_EXACT_THRESHOLD:
        return "BIT/SIMILARITY_EXACT"
    if residual <= POSITIONAL_THRESHOLD:
        return "POSITIONAL"
    return "DISTRIBUTIONAL"


def _format_float(value: float) -> str:
    """Format a finite or non-finite float.

    Parameters
    ----------
    value : float
        Numeric value.

    Returns
    -------
    str
        Compact display string.
    """
    if not torch.isfinite(torch.tensor(value)):
        return "nan"
    return f"{value:.6g}"


def _write_report(path: Path, rows: list[dict[str, Any]], blocker: str | None) -> None:
    """Write the sparse-stress fidelity report.

    Parameters
    ----------
    path : pathlib.Path
        Destination markdown path.
    rows : list[dict[str, Any]]
        Per-graph metrics.
    blocker : str or None
        Reference blocker text, if any graph failed to run.

    Returns
    -------
    None
        The report is written to disk.
    """
    lines = [
        "# sparse-stress fidelity",
        "",
        "Reference: MarkOrtmann/sparse-stress Java implementation, built manually with `javac`",
        "because Gradle 2.5 cannot parse Java 17",
        "(`Could not determine java version from '17.0.19'`).",
        "",
        "Native implementation: sampler and sparse term aggregation ported from source; PivotMDS",
        "uses NumPy's deterministic symmetric eigensolver on the same centered kernel.",
        "",
        "Named residual stage: `initialization_eigensolver`; pivots/sampler and sparse terms are",
        "matched in isolation for the pinned cases.",
        "",
    ]
    if blocker is not None:
        lines.extend(["## Reference blocker", "", blocker, ""])
    lines.extend(
        [
            "## Results",
            "",
            "| graph | residual | tier | stress delta | neighborhood delta | quality |",
            "| --- | ---: | --- | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {graph} | {residual} | {tier} | {stress} | {neighborhood} | {quality} |".format(
                graph=row["graph"],
                residual=_format_float(float(row["residual"])),
                tier=row["tier"],
                stress=_format_float(float(row["stress_delta"])),
                neighborhood=_format_float(float(row["neighborhood_delta"])),
                quality=row["quality"],
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The production pipeline never calls this adapter or any subprocess.",
            "- Reference input is restricted to connected simple undirected graphs, "
            "matching the reference README.",
            "- Remaining residual is expected to appear first at PivotMDS eigenvector "
            "orientation/order, not in sampler or sparse terms.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> int:
    """Run sparse-stress fidelity verification.

    Returns
    -------
    int
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--pivots", type=int, default=DEFAULT_PIVOTS)
    parser.add_argument("--mds-pivots", type=int, default=DEFAULT_MDS_PIVOTS)
    parser.add_argument("--sampler", choices=["random", "maxmin", "kmeans"], default="kmeans")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    competitor = SparseStressCompetitor()
    rows: list[dict[str, Any]] = []
    blocker: str | None = None
    for name, graph in _verification_graphs():
        actual = layout_sparse_stress_pipeline(
            graph.edge_index,
            graph.num_nodes,
            seed=args.seed,
            steps=args.iterations,
            pivots=args.pivots,
            sampler=args.sampler,
            mds_pivots=args.mds_pivots,
            kmeans_features=min(args.pivots, args.mds_pivots),
            dtype=torch.float64,
        ).to(dtype=torch.float64)
        reference = competitor.layout_with_variant(
            graph,
            timeout=120.0,
            seed=args.seed,
            variant_params={
                "steps": args.iterations,
                "pivots": args.pivots,
                "sampler": args.sampler,
                "mds_pivots": args.mds_pivots,
                "kmeans_features": min(args.pivots, args.mds_pivots),
            },
        )
        if reference.pos is None:
            blocker = f"sparse-stress reference failed for {name}: {reference.error}"
            residual = float("nan")
            stress_delta = float("nan")
            neighborhood_delta = float("nan")
            tier = "REFERENCE_BLOCKED"
            quality = "blocked"
        else:
            residual = procrustes_rmsd(actual.numpy(), reference.pos.numpy())
            metrics = compute_equivalence_metrics(
                actual,
                reference.pos,
                graph.edge_index,
                engine_name="sparse_stress",
            )
            stress_delta = metrics.stress_rel_delta
            neighborhood_delta = metrics.neighborhood_preservation_delta
            tier = _tier(residual)
            quality = metrics.verdict
        rows.append(
            {
                "graph": name,
                "residual": residual,
                "tier": tier,
                "stress_delta": stress_delta,
                "neighborhood_delta": neighborhood_delta,
                "quality": quality,
            }
        )
        print(
            "{name}: residual={residual} tier={tier} stress_delta={stress} "
            "neighborhood_delta={neighborhood} quality={quality}".format(
                name=name,
                residual=_format_float(float(residual)),
                tier=tier,
                stress=_format_float(float(stress_delta)),
                neighborhood=_format_float(float(neighborhood_delta)),
                quality=quality,
            )
        )
    _write_report(args.report, rows, blocker)
    print(f"report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
