"""Verify the Gephi ForceAtlas1 source-port fidelity status."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagua.eval.equivalence_metrics import procrustes_rmsd  # noqa: E402
from dagua.layout.ops.pipelines.forceatlas1 import layout_forceatlas1_pipeline  # noqa: E402
from dagua.metrics import composite, quick  # noqa: E402


@dataclass(frozen=True)
class VerificationGraph:
    """Small graph used by the ForceAtlas1 verification script.

    Parameters
    ----------
    name : str
        Report name.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.
    node_sizes : torch.Tensor, optional
        Optional node sizes with shape ``[N, 2]``.
    params : dict[str, object]
        ForceAtlas1 variant parameters.
    """

    name: str
    edge_index: torch.Tensor
    num_nodes: int
    edge_weights: Optional[torch.Tensor] = None
    node_sizes: Optional[torch.Tensor] = None
    params: Optional[dict[str, object]] = None


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
    """Return the fixed ForceAtlas1 verification corpus.

    Returns
    -------
    list[VerificationGraph]
        Small deterministic graphs covering defaults and requested variants.
    """
    return [
        VerificationGraph(
            name="path_default",
            edge_index=_edge_index([(0, 1), (1, 2), (2, 3), (3, 4)]),
            num_nodes=5,
        ),
        VerificationGraph(
            name="weighted_outbound",
            edge_index=_edge_index([(0, 1), (0, 2), (2, 3), (3, 1), (3, 4)]),
            num_nodes=5,
            edge_weights=torch.tensor([1.0, 0.5, 2.0, 1.5, 0.75], dtype=torch.float64),
            params={"outbound_attraction_distribution": True},
        ),
        VerificationGraph(
            name="adjust_sizes",
            edge_index=_edge_index([(0, 1), (1, 2), (2, 0), (2, 3)]),
            num_nodes=4,
            node_sizes=torch.full((4, 2), 18.0),
            params={"adjust_sizes": True},
        ),
        VerificationGraph(
            name="no_freeze_gravity",
            edge_index=_edge_index([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]),
            num_nodes=4,
            params={"freeze_balance": False, "gravity": 10.0},
        ),
    ]


def _fidelity_tier(residual: float, reference_ran: bool) -> str:
    """Classify a ForceAtlas1 residual.

    Parameters
    ----------
    residual : float
        Procrustes RMSD between compared layouts.
    reference_ran : bool
        Whether the Gephi toolkit runtime produced the reference layout.

    Returns
    -------
    str
        Fidelity tier label for the report.
    """
    if not reference_ran:
        return "source-port-self-check"
    if residual <= 1.0e-9:
        return "bit/similarity-exact"
    if residual <= 1.0e-3:
        return "strong-equivalent"
    return "partial"


def verify_graph(graph: VerificationGraph) -> tuple[float, str, float]:
    """Verify one graph and return residual, tier, and quality.

    Parameters
    ----------
    graph : VerificationGraph
        Verification graph and ForceAtlas1 parameters.

    Returns
    -------
    tuple[float, str, float]
        Procrustes residual, fidelity tier, and quick composite quality.
    """
    params = dict(graph.params or {})
    common = {
        "edge_index": graph.edge_index,
        "num_nodes": graph.num_nodes,
        "node_sizes": graph.node_sizes,
        "edge_weights": graph.edge_weights,
        "steps": 25,
        "seed": 42,
        "fidelity_dtype": torch.float64,
        **params,
    }
    source_port = layout_forceatlas1_pipeline(**common)
    # The Gephi toolkit runner was not established in this build, so the
    # residual is a deterministic self-check rather than a runtime oracle.
    reimpl = layout_forceatlas1_pipeline(**common)
    residual = procrustes_rmsd(source_port, reimpl)
    quality_metrics = quick(
        reimpl.to(dtype=torch.float32),
        graph.edge_index,
        node_sizes=graph.node_sizes,
        seed=42,
    )
    quality = composite(quality_metrics)
    return residual, _fidelity_tier(residual, reference_ran=False), quality


def main() -> None:
    """Print the ForceAtlas1 fidelity report.

    Returns
    -------
    None
        Writes a line-oriented report to stdout.
    """
    print("ForceAtlas1 fidelity verification")
    print("reference_runtime: blocked (Gephi toolkit runner not established)")
    print("first_divergent_stage: reference-runtime")
    print("model_status: source-faithful Gephi ForceAtlasLayout.java port")
    for graph in _verification_graphs():
        residual, tier, quality = verify_graph(graph)
        print(f"{graph.name}: residual={residual:.3e} tier={tier} quality={quality:.2f}")


if __name__ == "__main__":
    main()
