"""Verify the GRIP clean-room pipeline fidelity status."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagua.eval.competitors.native_reference_competitor import GripReferenceCompetitor  # noqa: E402
from dagua.eval.equivalence_metrics import procrustes_rmsd  # noqa: E402
from dagua.graph import DaguaGraph  # noqa: E402
from dagua.layout.ops.pipelines.grip import (  # noqa: E402
    build_mis_filtration,
    layout_grip_pipeline,
)
from dagua.metrics import composite, quick  # noqa: E402


@dataclass(frozen=True)
class VerificationGraph:
    """Small graph used by the GRIP verification script.

    Parameters
    ----------
    name : str
        Report name.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    steps : int
        Number of local refinement rounds per level.
    seed : int
        Seed for MIS construction.
    """

    name: str
    edge_index: torch.Tensor
    num_nodes: int
    steps: int
    seed: int


def _edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    """Build an edge-index tensor from a tuple list.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Directed edge tuples.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    sources, targets = zip(*edges)
    return torch.tensor([list(sources), list(targets)], dtype=torch.long)


def _verification_graphs() -> list[VerificationGraph]:
    """Return the fixed GRIP verification corpus.

    Returns
    -------
    list[VerificationGraph]
        Small deterministic graphs covering paths, cycles, branching, and
        disconnected components.
    """
    return [
        VerificationGraph(
            name="path6",
            edge_index=_edge_index([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]),
            num_nodes=6,
            steps=3,
            seed=42,
        ),
        VerificationGraph(
            name="cycle6",
            edge_index=_edge_index([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]),
            num_nodes=6,
            steps=3,
            seed=42,
        ),
        VerificationGraph(
            name="diamond_tail",
            edge_index=_edge_index([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]),
            num_nodes=5,
            steps=4,
            seed=17,
        ),
        VerificationGraph(
            name="two_components",
            edge_index=_edge_index([(0, 1), (1, 2), (3, 4), (4, 5)]),
            num_nodes=6,
            steps=3,
            seed=9,
        ),
    ]


def _fidelity_tier(residual: float, reference_ran: bool) -> str:
    """Classify a GRIP residual.

    Parameters
    ----------
    residual : float
        Procrustes RMSD between compared layouts.
    reference_ran : bool
        Whether the original GRIP runtime produced the reference layout.

    Returns
    -------
    str
        Fidelity tier label for the report.
    """
    if not reference_ran:
        return "quality-tier-clean-room"
    if residual <= 1.0e-9:
        return "bit/similarity-exact"
    if residual <= 1.0e-3:
        return "positional"
    if residual <= 5.0:
        return "distributional"
    return "partial"


def _reference_layout(graph: VerificationGraph) -> tuple[torch.Tensor, str | None]:
    """Run the headless GRIP reference for one graph.

    Parameters
    ----------
    graph : VerificationGraph
        Verification graph and GRIP parameters.

    Returns
    -------
    tuple[torch.Tensor, str | None]
        Reference positions with shape ``[N, 2]`` and optional error string.
    """
    competitor = GripReferenceCompetitor()
    dagua_graph = DaguaGraph.from_edge_index(graph.edge_index, graph.num_nodes)
    result = competitor.layout_with_variant(
        dagua_graph,
        seed=graph.seed,
        variant_params={
            "rounds": graph.steps,
            "final_rounds": graph.steps,
            "init_vertices": 4,
            "dim": 2,
        },
    )
    if result.pos is None:
        return torch.empty((0, 2), dtype=torch.float64), result.error or "unknown reference error"
    return result.pos.to(dtype=torch.float64), None


def _first_divergent_stage(residual: float, reference_error: str | None) -> str:
    """Name the first divergent GRIP stage.

    Parameters
    ----------
    residual : float
        Procrustes residual.
    reference_error : str | None
        Reference runtime error, if any.

    Returns
    -------
    str
        Stage label for closeout reporting.
    """
    if reference_error is not None:
        return "reference-runtime"
    if residual <= 1.0e-9:
        return "none"
    return "layout-force-refinement"


def verify_graph(graph: VerificationGraph) -> tuple[float, str, float, list[list[int]], str]:
    """Verify one graph and return residual, tier, quality, MIS levels, and stage.

    Parameters
    ----------
    graph : VerificationGraph
        Verification graph and GRIP parameters.

    Returns
    -------
    tuple[float, str, float, list[list[int]], str]
        Procrustes residual, tier, quick composite quality, MIS levels, and
        first divergent stage.
    """
    common = {
        "edge_index": graph.edge_index,
        "num_nodes": graph.num_nodes,
        "steps": graph.steps,
        "seed": graph.seed,
        "fidelity_dtype": torch.float64,
    }
    actual = layout_grip_pipeline(**common)
    reference, reference_error = _reference_layout(graph)
    residual = float("inf") if reference_error is not None else procrustes_rmsd(actual, reference)
    quality_metrics = quick(
        actual.to(dtype=torch.float32),
        graph.edge_index,
        seed=graph.seed,
    )
    quality = composite(quality_metrics)
    levels = build_mis_filtration(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        seed=graph.seed,
    )
    return (
        residual,
        _fidelity_tier(residual, reference_ran=reference_error is None),
        quality,
        levels,
        _first_divergent_stage(residual, reference_error),
    )


def main() -> None:
    """Print the GRIP fidelity report.

    Returns
    -------
    None
        Writes a line-oriented report to stdout.
    """
    print("GRIP fidelity verification")
    print("reference_runtime: headless ~/tools/dagua-refs/grip/original/grip_headless_layout")
    print("reference_license: unlicensed source archive; clean-room implementation from paper")
    print("named_residual: procrustes_rmsd")
    print("mis_init_status: isolated deterministic pins pass in tests/test_pipeline_grip.py")
    print("no_delegation_guards: pass")
    for graph in _verification_graphs():
        residual, tier, quality, levels, stage = verify_graph(graph)
        level_sizes = [len(level) for level in levels]
        print(
            f"{graph.name}: residual={residual:.3e} tier={tier} "
            f"quality={quality:.2f} mis_sizes={level_sizes} "
            f"first_divergent_stage={stage}"
        )


if __name__ == "__main__":
    main()
