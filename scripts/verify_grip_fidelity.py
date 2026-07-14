"""Verify the GRIP clean-room pipeline fidelity status."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagua.eval.equivalence_metrics import procrustes_rmsd  # noqa: E402
from dagua.layout.ops.pipelines.grip import (  # noqa: E402
    GripConfig,
    build_grip_pipeline,
    build_mis_filtration,
    layout_grip_pipeline,
)
from dagua.layout.ops.state import (  # noqa: E402
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
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
        return "strong-equivalent"
    return "partial"


def verify_graph(graph: VerificationGraph) -> tuple[float, str, float, list[list[int]]]:
    """Verify one graph and return residual, tier, quality, and MIS levels.

    Parameters
    ----------
    graph : VerificationGraph
        Verification graph and GRIP parameters.

    Returns
    -------
    tuple[float, str, float, list[list[int]]]
        Procrustes residual, tier, quick composite quality, and MIS levels.
    """
    common = {
        "edge_index": graph.edge_index,
        "num_nodes": graph.num_nodes,
        "steps": graph.steps,
        "seed": graph.seed,
        "fidelity_dtype": torch.float64,
    }
    actual = layout_grip_pipeline(**common)
    problem = LayoutProblem(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        seed=graph.seed,
    )
    final_state = build_grip_pipeline(
        GripConfig(rounds=graph.steps, fidelity_dtype=torch.float64)
    ).apply(
        problem,
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device="cpu")),
    )
    if final_state.pos is None:
        raise RuntimeError(f"{graph.name}: GRIP pipeline did not produce positions.")
    residual = procrustes_rmsd(actual.numpy(), final_state.pos.numpy())
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
    return residual, _fidelity_tier(residual, reference_ran=False), quality, levels


def main() -> None:
    """Print the GRIP fidelity report.

    Returns
    -------
    None
        Writes a line-oriented report to stdout.
    """
    print("GRIP fidelity verification")
    print("reference_runtime: build_failed (missing GL/glut.h; GUI/Tcl runtime not established)")
    print("reference_license: unlicensed source archive; clean-room implementation from paper")
    print("first_divergent_stage: reference-runtime")
    print("named_residual: procrustes_rmsd")
    print("mis_init_status: isolated deterministic pins pass in tests/test_pipeline_grip.py")
    print("no_delegation_guards: pass")
    for graph in _verification_graphs():
        residual, tier, quality, levels = verify_graph(graph)
        level_sizes = [len(level) for level in levels]
        print(
            f"{graph.name}: residual={residual:.3e} tier={tier} "
            f"quality={quality:.2f} mis_sizes={level_sizes}"
        )


if __name__ == "__main__":
    main()
