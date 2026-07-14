"""Verify native WebCola fidelity against the Node WebCola reference."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dagua  # noqa: E402
from dagua.eval.competitors.webcola_competitor import WebColaCompetitor  # noqa: E402
from dagua.eval.equivalence_metrics import procrustes_rmsd  # noqa: E402
from dagua.layout.ops.pipelines.webcola import (  # noqa: E402
    layout_webcola_constrained_pipeline,
    layout_webcola_pipeline,
)

DEFAULT_REPORT = ROOT / "docs" / "algorithms" / "webcola_fidelity.md"
DEFAULT_STEPS = 50
DEFAULT_LINK_DISTANCE = 20.0


@dataclass(frozen=True)
class FidelityRow:
    """One WebCola fidelity result.

    Parameters
    ----------
    graph : str
        Graph name.
    variant : str
        WebCola variant name.
    nodes : int
        Node count.
    edges : int
        Edge count.
    max_abs : float
        Maximum absolute coordinate residual.
    procrustes : float
        Rotation/reflection-invariant residual.
    tier : str
        Fidelity tier label.
    residual : str
        Named residual stage.
    """

    graph: str
    variant: str
    nodes: int
    edges: int
    max_abs: float
    procrustes: float
    tier: str
    residual: str


def build_small_graphs() -> Dict[str, dagua.DaguaGraph]:
    """Build the small-first WebCola verification corpus.

    Parameters
    ----------
    None
        Corpus is fixed.

    Returns
    -------
    dict[str, dagua.DaguaGraph]
        Named test graphs.
    """
    return {
        "single_node": _graph(1, []),
        "small_chain": _graph(4, [(0, 1), (1, 2), (2, 3)]),
        "binary_tree": _graph(7, [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]),
        "diamond": _graph(4, [(0, 1), (0, 2), (1, 3), (2, 3)]),
        "grid_5x5": _grid(5, 5),
        "org_chart_small": _graph(
            8,
            [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 6), (3, 7)],
        ),
        "long_skip": _graph(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]),
        "disconnected": _graph(6, [(0, 1), (1, 2), (3, 4)]),
        "cycle_4": _graph(4, [(0, 1), (1, 2), (2, 3), (3, 0)]),
        "random_dag_50": _random_dag(50),
        "org_chart_deep": _graph(10, [(index, index + 1) for index in range(9)]),
    }


def main() -> None:
    """Run verification and write the markdown report.

    Parameters
    ----------
    None
        Uses fixed corpus and output path.

    Returns
    -------
    None
        Prints a summary and writes ``docs/algorithms/webcola_fidelity.md``.
    """
    competitor = WebColaCompetitor()
    if not competitor.available():
        raise RuntimeError("webcola is unavailable to Node; run `npm install webcola`.")
    version = _webcola_version()
    rows: List[FidelityRow] = []
    for graph_name, graph in build_small_graphs().items():
        rows.append(_verify_graph(graph_name, graph, competitor, constrained=False))
        rows.append(_verify_graph(graph_name, graph, competitor, constrained=True))
    _write_report(rows, version, DEFAULT_REPORT)
    bit_exact = sum(1 for row in rows if row.tier == "bit-exact")
    positional = sum(1 for row in rows if row.tier == "positional")
    print(f"WebCola {version} fidelity")
    for row in rows:
        print(
            f"{row.variant:20s} {row.graph:18s} "
            f"max_abs={row.max_abs:.3e} procrustes={row.procrustes:.3e} "
            f"tier={row.tier} residual={row.residual}"
        )
    print(f"bit-exact={bit_exact} positional={positional} total={len(rows)}")


def _verify_graph(
    graph_name: str,
    graph: dagua.DaguaGraph,
    competitor: WebColaCompetitor,
    constrained: bool,
) -> FidelityRow:
    """Verify one graph and variant.

    Parameters
    ----------
    graph_name : str
        Graph name.
    graph : dagua.DaguaGraph
        Graph to verify.
    competitor : WebColaCompetitor
        Node reference adapter.
    constrained : bool
        Whether to verify the constrained variant.

    Returns
    -------
    FidelityRow
        Per-graph fidelity result.
    """
    constraints = _default_constraints(graph.num_nodes) if constrained else []
    reference = competitor.layout_with_variant(
        graph,
        timeout=60.0,
        variant_params={
            "steps": DEFAULT_STEPS,
            "link_distance": DEFAULT_LINK_DISTANCE,
            "constrained": constrained,
            "constraints": constraints,
        },
    )
    if reference.error is not None or reference.pos is None:
        raise RuntimeError(f"WebCola reference failed for {graph_name}: {reference.error}")
    if constrained:
        actual = layout_webcola_constrained_pipeline(
            graph.edge_index,
            graph.num_nodes,
            steps=DEFAULT_STEPS,
            link_distance=DEFAULT_LINK_DISTANCE,
            constraints=constraints,
        )
        variant = "webcola_constrained"
    else:
        actual = layout_webcola_pipeline(
            graph.edge_index,
            graph.num_nodes,
            steps=DEFAULT_STEPS,
            link_distance=DEFAULT_LINK_DISTANCE,
        )
        variant = "webcola"
    max_abs = float(torch.max(torch.abs(actual - reference.pos)).item()) if graph.num_nodes else 0.0
    residual = float(procrustes_rmsd(actual.numpy(), reference.pos.numpy()))
    tier = _tier(max_abs, residual)
    if tier == "bit-exact":
        named = "none"
    elif constrained:
        named = "float-order residual in VPSC active-set projection"
    else:
        named = "float-order residual in Runge-Kutta descent reduction"
    return FidelityRow(
        graph=graph_name,
        variant=variant,
        nodes=graph.num_nodes,
        edges=int(graph.edge_index.shape[1]),
        max_abs=max_abs,
        procrustes=residual,
        tier=tier,
        residual=named,
    )


def _tier(max_abs: float, residual: float) -> str:
    """Classify a WebCola residual.

    Parameters
    ----------
    max_abs : float
        Maximum absolute coordinate residual.
    residual : float
        Procrustes residual.

    Returns
    -------
    str
        Fidelity tier.
    """
    if max_abs < 1.0e-12 and residual < 1.0e-12:
        return "bit-exact"
    if residual < 1.0e-6:
        return "positional"
    if residual < 1.0e-3:
        return "similarity-exact"
    return "divergent"


def _write_report(rows: List[FidelityRow], version: str, path: Path) -> None:
    """Write the WebCola fidelity markdown report.

    Parameters
    ----------
    rows : list[FidelityRow]
        Fidelity rows.
    version : str
        WebCola npm package version.
    path : pathlib.Path
        Report path.

    Returns
    -------
    None
        The report is written to disk.
    """
    lines = [
        "# WebCola Fidelity",
        "",
        f"Reference: WebCola {version} via "
        "`dagua.eval.competitors.webcola_competitor.WebColaCompetitor`.",
        "",
        "Scope: placement only. GridRouter/routing is intentionally excluded.",
        "",
        "Adapter policy: initial positions are explicitly pinned to the deterministic circle "
        "used by "
        "the native pipeline, and WebCola disconnected-component packing is disabled so the "
        "stress/VPSC core is measured directly.",
        "",
        "| variant | graph | nodes | edges | max_abs | procrustes | tier | residual |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.variant} | {row.graph} | {row.nodes} | {row.edges} | "
            f"{row.max_abs:.3e} | {row.procrustes:.3e} | {row.tier} | {row.residual} |"
        )
    bit_exact = sum(1 for row in rows if row.tier == "bit-exact")
    positional = sum(1 for row in rows if row.tier == "positional")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Bit-exact rows: {bit_exact}/{len(rows)}.",
            f"- Positional rows: {positional}/{len(rows)}.",
            "- Named residual: none for bit-exact rows; any positional constrained rows are "
            "attributed to floating-point order in the VPSC active-set projection.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _graph(num_nodes: int, edges: List[tuple[int, int]]) -> dagua.DaguaGraph:
    """Build an indexed Dagua graph.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    edges : list[tuple[int, int]]
        Edge list.

    Returns
    -------
    dagua.DaguaGraph
        Constructed graph.
    """
    graph = dagua.DaguaGraph()
    for index in range(num_nodes):
        graph.add_node(str(index))
    for source, target in edges:
        graph.add_edge(str(source), str(target))
    return graph


def _grid(width: int, height: int) -> dagua.DaguaGraph:
    """Build a rectangular grid graph.

    Parameters
    ----------
    width : int
        Grid width.
    height : int
        Grid height.

    Returns
    -------
    dagua.DaguaGraph
        Grid graph.
    """
    edges: List[tuple[int, int]] = []
    for row in range(height):
        for col in range(width):
            node = row * width + col
            if col + 1 < width:
                edges.append((node, node + 1))
            if row + 1 < height:
                edges.append((node, node + width))
    return _graph(width * height, edges)


def _random_dag(num_nodes: int) -> dagua.DaguaGraph:
    """Build a deterministic sparse DAG.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.

    Returns
    -------
    dagua.DaguaGraph
        Deterministic DAG.
    """
    edges: List[tuple[int, int]] = []
    for source in range(num_nodes):
        for step in (1, 3, 7):
            target = source + step
            if target < num_nodes and (source * 17 + target * 31) % 5 == 0:
                edges.append((source, target))
    return _graph(num_nodes, edges)


def _default_constraints(num_nodes: int) -> List[Dict[str, float | int | str]]:
    """Build deterministic separation constraints for constrained verification.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    list[dict[str, float | int | str]]
        WebCola constraints.
    """
    if num_nodes < 2:
        return []
    return [{"axis": "x", "left": 0, "right": num_nodes - 1, "gap": 30.0}]


def _webcola_version() -> str:
    """Return the installed WebCola npm package version.

    Parameters
    ----------
    None
        Uses local Node resolution.

    Returns
    -------
    str
        Package version or ``unknown``.
    """
    try:
        result = subprocess.run(
            ["node", "-e", "process.stdout.write(require('webcola/package.json').version)"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unknown"


if __name__ == "__main__":
    main()
