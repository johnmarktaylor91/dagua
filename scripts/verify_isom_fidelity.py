"""Verify JUNG ISOM source-port fidelity against the Java runtime."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagua.eval.competitors.isom_competitor import IsomCompetitor  # noqa: E402
from dagua.eval.equivalence_metrics import procrustes_rmsd  # noqa: E402
from dagua.graph import DaguaGraph  # noqa: E402
from dagua.layout.ops.pipelines.isom import JavaRandom, layout_isom_pipeline  # noqa: E402
from dagua.metrics import composite, quick  # noqa: E402


@dataclass(frozen=True)
class VerificationGraph:
    """Small graph used by the ISOM verification script.

    Parameters
    ----------
    name : str
        Report name.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    steps : int
        Number of ISOM epochs.
    seed : int
        Java ``Random`` seed.
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


def _graph_from_verification_graph(graph: VerificationGraph) -> DaguaGraph:
    """Convert a verification graph into ``DaguaGraph`` form.

    Parameters
    ----------
    graph : VerificationGraph
        Verification graph with tensor topology.

    Returns
    -------
    DaguaGraph
        Dagua graph with stable numeric string node IDs.
    """
    dagua_graph = DaguaGraph()
    for node in range(graph.num_nodes):
        dagua_graph.add_node(str(node))
    for edge_pos in range(graph.edge_index.shape[1]):
        source = int(graph.edge_index[0, edge_pos].item())
        target = int(graph.edge_index[1, edge_pos].item())
        dagua_graph.add_edge(str(source), str(target))
    return dagua_graph


def _verification_graphs() -> list[VerificationGraph]:
    """Return the fixed ISOM verification corpus.

    Returns
    -------
    list[VerificationGraph]
        Small deterministic graphs covering paths, branches, cycles, and
        disconnected components.
    """
    return [
        VerificationGraph(
            name="path3_short",
            edge_index=_edge_index([(0, 1), (1, 2)]),
            num_nodes=3,
            steps=3,
            seed=42,
        ),
        VerificationGraph(
            name="path5_default_window",
            edge_index=_edge_index([(0, 1), (1, 2), (2, 3), (3, 4)]),
            num_nodes=5,
            steps=25,
            seed=42,
        ),
        VerificationGraph(
            name="branch6",
            edge_index=_edge_index([(0, 1), (0, 2), (2, 3), (2, 4), (4, 5)]),
            num_nodes=6,
            steps=50,
            seed=99,
        ),
        VerificationGraph(
            name="disconnected8",
            edge_index=_edge_index([(0, 1), (1, 2), (3, 4), (5, 6)]),
            num_nodes=8,
            steps=75,
            seed=7,
        ),
        VerificationGraph(
            name="cycle_chord6",
            edge_index=_edge_index([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3)]),
            num_nodes=6,
            steps=100,
            seed=13,
        ),
    ]


def _fidelity_tier(residual: float, reference_ran: bool) -> str:
    """Classify an ISOM residual.

    Parameters
    ----------
    residual : float
        Procrustes RMSD between compared layouts.
    reference_ran : bool
        Whether JUNG produced the reference layout.

    Returns
    -------
    str
        Fidelity tier label for the report.
    """
    if not reference_ran:
        return "source-port-only"
    if residual <= 1.0e-12:
        return "bit/similarity-exact"
    if residual <= 1.0e-9:
        return "strong-equivalent"
    if residual <= 1.0e-6:
        return "similarity-close"
    return "partial"


def _rng_matched() -> bool:
    """Return whether the Java RNG port matches pinned JDK samples.

    Returns
    -------
    bool
        ``True`` when the seed-42 sequence matches known ``nextDouble`` values.
    """
    rng = JavaRandom(42)
    samples = [rng.next_double() for _ in range(3)]
    return samples == [
        0.7275636800328681,
        0.6832234717598454,
        0.30871945533265976,
    ]


def verify_graph(
    graph: VerificationGraph,
    competitor: IsomCompetitor,
) -> tuple[float, str, float, Optional[str]]:
    """Verify one graph and return residual, tier, quality, and error.

    Parameters
    ----------
    graph : VerificationGraph
        Verification graph and ISOM parameters.
    competitor : IsomCompetitor
        JUNG reference adapter.

    Returns
    -------
    tuple[float, str, float, str | None]
        Procrustes residual, fidelity tier, quick composite quality, and
        optional reference error.
    """
    dagua_graph = _graph_from_verification_graph(graph)
    reimpl = layout_isom_pipeline(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        steps=graph.steps,
        seed=graph.seed,
        fidelity_dtype=torch.float64,
    )
    reference = competitor.layout_with_variant(
        dagua_graph,
        timeout=60.0,
        seed=graph.seed,
        variant_params={"steps": graph.steps},
    )
    if reference.pos is None:
        residual = procrustes_rmsd(reimpl, reimpl)
        tier = _fidelity_tier(residual, reference_ran=False)
        error = reference.error
    else:
        residual = procrustes_rmsd(reimpl, reference.pos)
        tier = _fidelity_tier(residual, reference_ran=True)
        error = None
    quality_metrics = quick(
        reimpl.to(dtype=torch.float32),
        graph.edge_index,
        seed=graph.seed,
    )
    quality = composite(quality_metrics)
    return residual, tier, quality, error


def main() -> None:
    """Print the ISOM fidelity report.

    Returns
    -------
    None
        Writes a line-oriented report to stdout.
    """
    competitor = IsomCompetitor()
    reference_available = competitor.available()
    print("ISOM fidelity verification")
    print(f"reference_runtime: {'JUNG jar' if reference_available else 'blocked'}")
    print(f"rng_matched_java_random: {_rng_matched()}")
    print("model_status: source-faithful JUNG ISOMLayout.java port")
    first_divergent_stage = "none"
    for graph in _verification_graphs():
        residual, tier, quality, error = verify_graph(graph, competitor)
        if error is not None and first_divergent_stage == "none":
            first_divergent_stage = "reference-runtime"
        elif (
            tier not in {"bit/similarity-exact", "strong-equivalent"}
            and first_divergent_stage == "none"
        ):
            first_divergent_stage = "isom-epoch-dynamics"
        error_suffix = "" if error is None else f" error={error!r}"
        print(
            f"{graph.name}: residual={residual:.3e} tier={tier} quality={quality:.2f}{error_suffix}"
        )
    print(f"first_divergent_stage: {first_divergent_stage}")


if __name__ == "__main__":
    main()
