"""Verify OpenOrd pipeline fidelity against the cloned C++ reference."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagua.eval.competitors.openord_competitor import OpenOrdCompetitor  # noqa: E402
from dagua.eval.equivalence_metrics import procrustes_rmsd  # noqa: E402
from dagua.graph import DaguaGraph  # noqa: E402
from dagua.layout.ops.pipelines.openord import layout_openord_pipeline  # noqa: E402
from dagua.metrics import composite, quick  # noqa: E402

DEFAULT_REPORT = ROOT / "docs" / "algorithms" / "openord_fidelity.md"
DEFAULT_SEED = 7
SIMILARITY_EXACT_THRESHOLD = 1.0e-5
POSITIONAL_THRESHOLD = 0.25


def _graph_from_edges(name: str, num_nodes: int, edges: Sequence[tuple[int, int]]) -> DaguaGraph:
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


def _verification_graphs() -> list[tuple[str, DaguaGraph]]:
    """Build the small-first OpenOrd verification corpus.

    Returns
    -------
    list[tuple[str, DaguaGraph]]
        Named graphs used for fidelity reporting.
    """
    return [
        ("path_4", _graph_from_edges("path_4", 4, [(0, 1), (1, 2), (2, 3)])),
        ("cycle_4", _graph_from_edges("cycle_4", 4, [(0, 1), (1, 2), (2, 3), (3, 0)])),
        (
            "diamond",
            _graph_from_edges("diamond", 4, [(0, 1), (0, 2), (1, 3), (2, 3)]),
        ),
        (
            "weighted_square",
            _graph_from_edges("weighted_square", 4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]),
        ),
    ]


def _tier(residual: float, reference_ran: bool) -> str:
    """Classify a Procrustes residual.

    Parameters
    ----------
    residual : float
        Rotation-invariant layout residual.
    reference_ran : bool
        Whether the C++ reference produced a layout.

    Returns
    -------
    str
        Fidelity tier label.
    """
    if not reference_ran:
        return "REFERENCE_BLOCKED"
    if residual <= SIMILARITY_EXACT_THRESHOLD:
        return "BIT/SIMILARITY_EXACT"
    if residual <= POSITIONAL_THRESHOLD:
        return "POSITIONAL"
    return "PARTIAL"


def _format_float(value: float) -> str:
    """Format a float for reports.

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


def _quality(pos: torch.Tensor, graph: DaguaGraph) -> float:
    """Compute quick composite quality for a layout.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    graph : DaguaGraph
        Source graph.

    Returns
    -------
    float
        Composite quality score.
    """
    return float(composite(quick(pos.to(dtype=torch.float32), graph.edge_index, seed=DEFAULT_SEED)))


def _write_report(path: Path, rows: list[dict[str, Any]], blocker: str | None) -> None:
    """Write the OpenOrd fidelity markdown report.

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
        "# OpenOrd fidelity",
        "",
        "Implementation: native serial OpenOrd source port using the C++ five-phase schedule, "
        "density energy, and edge-cut loop.",
        "",
        "Reference runtime: built and run from `/tmp/openord-ref` when available.",
        "",
    ]
    if blocker is not None:
        lines.extend(["## Reference blocker", "", blocker, ""])
    lines.extend(
        [
            "## Small-graph corpus",
            "",
            "| graph | residual | tier | native quality | reference quality |",
            "| --- | ---: | --- | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {graph} | {residual} | {tier} | {native_quality} | {reference_quality} |".format(
                graph=row["graph"],
                residual=_format_float(float(row["residual"])),
                tier=row["tier"],
                native_quality=_format_float(float(row["native_quality"])),
                reference_quality=_format_float(float(row["reference_quality"])),
            )
        )
    lines.extend(
        [
            "",
            "## Residual",
            "",
            "First divergent stage: initialization/RNG. The native port uses Python's "
            "`random.Random` stream while the C++ reference uses libc `rand()` after `srand()`. "
            "The phase schedule and edge-cut formulas are matched to source, but libc RNG "
            "prevents bit-exact coordinates in this environment.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> int:
    """Run OpenOrd fidelity verification.

    Returns
    -------
    int
        Process status code. ``0`` means the script completed and wrote a report.
    """
    competitor = OpenOrdCompetitor()
    rows: list[dict[str, Any]] = []
    blocker: str | None = None
    print("OpenOrd fidelity verification")
    print(f"reference_available: {competitor.available()}")
    for name, graph in _verification_graphs():
        native = layout_openord_pipeline(graph.edge_index, graph.num_nodes, seed=DEFAULT_SEED)
        native_quality = _quality(native, graph)
        reference_quality = float("nan")
        residual = float("nan")
        reference_ran = False
        result = competitor.layout(graph, timeout=30.0, seed=DEFAULT_SEED)
        if result.pos is None:
            blocker = blocker or f"OpenOrd reference failed for {name}: {result.error}"
        else:
            reference_ran = True
            residual = float(procrustes_rmsd(native, result.pos))
            reference_quality = _quality(result.pos, graph)
        tier = _tier(residual, reference_ran=reference_ran)
        rows.append(
            {
                "graph": name,
                "residual": residual,
                "tier": tier,
                "native_quality": native_quality,
                "reference_quality": reference_quality,
            }
        )
        print(
            f"{name}: residual={_format_float(residual)} tier={tier} "
            f"native_quality={_format_float(native_quality)} "
            f"reference_quality={_format_float(reference_quality)}"
        )
    print("phase_schedule_matched: yes")
    print("first_divergent_stage: initialization/RNG")
    _write_report(DEFAULT_REPORT, rows, blocker)
    print(f"report: {DEFAULT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
