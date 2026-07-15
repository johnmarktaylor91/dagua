"""Verify backbone pipeline fidelity and quality status."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagua.eval.competitors.backbone_competitor import (  # noqa: E402
    BackboneCompetitor,
    _graph_edge_csv,
    _parse_reference_output,
    _reference_script,
)
from dagua.eval.equivalence_metrics import procrustes_rmsd  # noqa: E402
from dagua.graph import DaguaGraph  # noqa: E402
from dagua.layout.ops.pipelines.backbone import (  # noqa: E402
    backbone_edge_set,
    layout_backbone_pipeline,
)
from dagua.metrics import composite, quick  # noqa: E402

BIT_EXACT_THRESHOLD = 1.0e-9
SIMILAR_THRESHOLD = 1.0e-3


@dataclass(frozen=True)
class VerificationGraph:
    """Backbone verification graph.

    Parameters
    ----------
    name : str
        Graph name.
    num_nodes : int
        Number of nodes.
    edges : list[tuple[int, int]]
        Undirected edge list.
    keep : float
        Backbone keep fraction.
    iterations : int
        Stress iterations.
    """

    name: str
    num_nodes: int
    edges: list[tuple[int, int]]
    keep: float
    iterations: int


def _edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    """Build an edge-index tensor.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Edge pairs.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    sources, targets = zip(*edges)
    return torch.tensor([list(sources), list(targets)], dtype=torch.long)


def _graphs() -> list[VerificationGraph]:
    """Return the fixed verification corpus.

    Returns
    -------
    list[VerificationGraph]
        Small graphs covering sparse, dense, and disconnected inputs.
    """
    return [
        VerificationGraph(
            name="path4",
            num_nodes=4,
            edges=[(0, 1), (1, 2), (2, 3)],
            keep=0.5,
            iterations=20,
        ),
        VerificationGraph(
            name="cycle_diagonal",
            num_nodes=4,
            edges=[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)],
            keep=0.4,
            iterations=20,
        ),
        VerificationGraph(
            name="triangle_tail",
            num_nodes=4,
            edges=[(0, 1), (1, 2), (2, 0), (2, 3)],
            keep=0.4,
            iterations=20,
        ),
        VerificationGraph(
            name="two_components",
            num_nodes=6,
            edges=[(0, 1), (1, 2), (3, 4), (4, 5)],
            keep=0.5,
            iterations=20,
        ),
    ]


def _tier(residual: float | None, reference_ran: bool) -> str:
    """Classify one fidelity residual.

    Parameters
    ----------
    residual : float | None
        Procrustes residual, or ``None`` when no reference ran.
    reference_ran : bool
        Whether graphlayouts produced a reference.

    Returns
    -------
    str
        Fidelity tier label.
    """
    if not reference_ran or residual is None:
        return "quality-tier-clean-room"
    if residual <= BIT_EXACT_THRESHOLD:
        return "bit/similarity-exact"
    if residual <= SIMILAR_THRESHOLD:
        return "similarity-exact"
    return "partial"


def _overall_tier(residuals: list[float], reference_ran: bool) -> str:
    """Classify the aggregate verification tier.

    Parameters
    ----------
    residuals : list[float]
        Per-graph Procrustes residuals from successful reference runs.
    reference_ran : bool
        Whether the R reference was available.

    Returns
    -------
    str
        Aggregate fidelity tier.
    """
    if not reference_ran or not residuals:
        return "source-faithful-clean-room"
    worst_residual = max(residuals)
    if worst_residual <= BIT_EXACT_THRESHOLD:
        return "reference-verified-bit/similarity-exact"
    if worst_residual <= SIMILAR_THRESHOLD:
        return "reference-verified-similarity-exact"
    return "reference-verified-partial"


def _reference_layout(
    graph: DaguaGraph,
    keep: float,
    timeout: float = 60.0,
) -> tuple[torch.Tensor, list[int]] | None:
    """Run graphlayouts and return coordinates plus one-based backbone ids.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to lay out.
    keep : float
        Backbone keep fraction.
    timeout : float, default=60.0
        Maximum Rscript runtime.

    Returns
    -------
    tuple[torch.Tensor, list[int]] | None
        Reference positions and backbone edge ids, or ``None`` when unavailable.
    """
    if not BackboneCompetitor().available():
        return None
    with tempfile.TemporaryDirectory(prefix="dagua_backbone_verify_") as tmpdir:
        tmp_path = Path(tmpdir)
        edge_path = tmp_path / "edges.csv"
        script_path = tmp_path / "run_backbone.R"
        edge_path.write_text(_graph_edge_csv(graph), encoding="utf-8")
        script_path.write_text(_reference_script(), encoding="utf-8")
        result = subprocess.run(
            ["Rscript", str(script_path), str(edge_path), str(graph.num_nodes), str(keep), "42"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            return None
        return _parse_reference_output(result.stdout, graph.num_nodes)


def _format_float(value: float | None) -> str:
    """Format optional floats for compact reports.

    Parameters
    ----------
    value : float | None
        Value to format.

    Returns
    -------
    str
        Human-readable value.
    """
    if value is None:
        return "NA"
    return f"{value:.6g}"


def main() -> int:
    """Run backbone verification and print per-graph results.

    Returns
    -------
    int
        Process exit code.
    """
    reference_available = BackboneCompetitor().available()
    print(f"reference_r_package_ran: {'yes' if reference_available else 'no'}")
    print("named_residual: stress_initialization_mds_rng_parity")
    print("graph,residual,tier,quality,backbone_edge_set_matched")

    residuals: list[float] = []
    edge_matches: list[bool] = []
    for spec in _graphs():
        edge_index = _edge_index(spec.edges)
        graph = DaguaGraph.from_edge_list(spec.edges, num_nodes=spec.num_nodes)
        graph.compute_node_sizes()
        actual, native_edges = layout_backbone_pipeline(
            edge_index=edge_index,
            num_nodes=spec.num_nodes,
            keep=spec.keep,
            iterations=spec.iterations,
            return_backbone=True,
        )
        reference = _reference_layout(graph=graph, keep=spec.keep) if reference_available else None
        residual = None
        edge_match = "not-run"
        if reference is not None:
            reference_pos, reference_edge_ids = reference
            residual = procrustes_rmsd(actual.numpy(), reference_pos.numpy())
            canonical_edges, _, _ = backbone_edge_set(edge_index, spec.num_nodes, keep=spec.keep)
            native_one_based = [
                index + 1
                for index, edge in enumerate(spec.edges)
                if tuple(sorted(edge)) in native_edges
            ]
            native_one_based_from_canonical = [
                index + 1
                for index, edge in enumerate(spec.edges)
                if tuple(sorted(edge)) in canonical_edges
            ]
            edge_match = (
                "yes"
                if sorted(reference_edge_ids) == sorted(native_one_based)
                or sorted(reference_edge_ids) == sorted(native_one_based_from_canonical)
                else "no"
            )
            residuals.append(residual)
            edge_matches.append(edge_match == "yes")
        metrics = quick(actual, edge_index, seed=42)
        quality = composite(metrics)
        print(
            f"{spec.name},{_format_float(residual)},{_tier(residual, reference is not None)},"
            f"{quality:.3f},{edge_match}"
        )
    if reference_available:
        print(f"overall_tier: {_overall_tier(residuals, reference_available)}")
        print(f"all_backbone_edge_sets_matched: {'yes' if all(edge_matches) else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
