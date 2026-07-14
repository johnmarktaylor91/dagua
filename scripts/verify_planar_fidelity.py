"""Verify planar pipeline fidelity against NetworkX 3.6.1."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dagua.layout.ops.pipelines.planar import (  # noqa: E402
    PlanarityError,
    check_planarity,
    layout_planar_pipeline,
)


@dataclass(frozen=True)
class FidelityCase:
    """One graph case for planar fidelity verification.

    Parameters
    ----------
    name : str
        Stable graph case name.
    graph : nx.Graph
        NetworkX graph with arbitrary hashable node ids.
    """

    name: str
    graph: nx.Graph


@dataclass(frozen=True)
class FidelityResult:
    """Planar fidelity result for one graph.

    Parameters
    ----------
    name : str
        Graph case name.
    nodes : int
        Node count.
    edges : int
        Edge count.
    status : str
        ``"bit-exact"`` or ``"N/A"``.
    max_abs : float
        Maximum absolute direct coordinate difference.
    embedding_match : bool
        Whether cyclic adjacency matched before placement.
    reason : str
        Empty for successful cases, otherwise N/A reason.
    """

    name: str
    nodes: int
    edges: int
    status: str
    max_abs: float
    embedding_match: bool
    reason: str = ""


def _edge_index_from_graph(graph: nx.Graph) -> torch.Tensor:
    """Return a Dagua edge tensor from a graph.

    Parameters
    ----------
    graph : nx.Graph
        Graph with integer node ids.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    edges = list(graph.edges())
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _reference_tensor(graph: nx.Graph) -> torch.Tensor:
    """Return NetworkX planar-layout positions as a tensor.

    Parameters
    ----------
    graph : nx.Graph
        Graph with integer node ids.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]`` and dtype ``torch.float64``.
    """
    pos = nx.planar_layout(graph)
    if graph.number_of_nodes() == 0:
        return torch.empty((0, 2), dtype=torch.float64)
    array = np.vstack([pos[node] for node in range(graph.number_of_nodes())])
    return torch.from_numpy(array).to(dtype=torch.float64)


def _fidelity_cases() -> list[FidelityCase]:
    """Build the standard small-graph planar fidelity suite.

    Returns
    -------
    list[FidelityCase]
        Named graph cases.
    """
    cases = [
        FidelityCase("empty", nx.empty_graph(0)),
        FidelityCase("single", nx.empty_graph(1)),
        FidelityCase("path_4", nx.path_graph(4)),
        FidelityCase("cycle_6", nx.cycle_graph(6)),
        FidelityCase("k4", nx.complete_graph(4)),
        FidelityCase("grid_3x3", nx.grid_2d_graph(3, 3)),
        FidelityCase("triangular_lattice_2x3", nx.triangular_lattice_graph(2, 3)),
        FidelityCase("disconnected_paths", nx.disjoint_union(nx.path_graph(3), nx.path_graph(2))),
        FidelityCase("k5_non_planar", nx.complete_graph(5)),
        FidelityCase("k3_3_non_planar", nx.complete_bipartite_graph(3, 3)),
    ]
    return cases


def _evaluate_case(case: FidelityCase) -> FidelityResult:
    """Evaluate one planar fidelity case.

    Parameters
    ----------
    case : FidelityCase
        Graph case to evaluate.

    Returns
    -------
    FidelityResult
        Per-case fidelity result.
    """
    graph = nx.convert_node_labels_to_integers(case.graph)
    edge_index = _edge_index_from_graph(graph)
    reference_is_planar, reference_embedding = nx.check_planarity(graph)
    actual_is_planar, actual_embedding = check_planarity(edge_index, graph.number_of_nodes())
    embedding_match = reference_is_planar == actual_is_planar and (
        not reference_is_planar
        or (
            actual_embedding is not None
            and reference_embedding is not None
            and actual_embedding.get_data() == reference_embedding.get_data()
        )
    )

    if not reference_is_planar:
        try:
            layout_planar_pipeline(
                edge_index,
                graph.number_of_nodes(),
                fidelity_dtype=torch.float64,
            )
        except PlanarityError as exc:
            return FidelityResult(
                name=case.name,
                nodes=graph.number_of_nodes(),
                edges=graph.number_of_edges(),
                status="N/A",
                max_abs=0.0,
                embedding_match=embedding_match,
                reason=str(exc),
            )
        return FidelityResult(
            name=case.name,
            nodes=graph.number_of_nodes(),
            edges=graph.number_of_edges(),
            status="failed",
            max_abs=float("inf"),
            embedding_match=embedding_match,
            reason="expected non-planar rejection",
        )

    actual = layout_planar_pipeline(
        edge_index,
        graph.number_of_nodes(),
        fidelity_dtype=torch.float64,
    )
    expected = _reference_tensor(graph)
    max_abs = float((actual.cpu() - expected).abs().max().item()) if actual.numel() else 0.0
    status = "bit-exact" if torch.equal(actual.cpu(), expected) and embedding_match else "residual"
    return FidelityResult(
        name=case.name,
        nodes=graph.number_of_nodes(),
        edges=graph.number_of_edges(),
        status=status,
        max_abs=max_abs,
        embedding_match=embedding_match,
    )


def main() -> None:
    """Run the fidelity suite and print a compact report.

    Returns
    -------
    None
        Results are printed to stdout.
    """
    results = [_evaluate_case(case) for case in _fidelity_cases()]
    bit_exact = sum(result.status == "bit-exact" for result in results)
    not_applicable = sum(result.status == "N/A" for result in results)
    residual = [result for result in results if result.status not in {"bit-exact", "N/A"}]

    print("planar fidelity vs networkx 3.6.1")
    print(f"bit-exact={bit_exact} N/A={not_applicable} residual={len(residual)}")
    print("name,nodes,edges,status,d_R,embedding_match,reason")
    for result in results:
        print(
            f"{result.name},{result.nodes},{result.edges},{result.status},"
            f"{result.max_abs:.17g},{result.embedding_match},{result.reason}"
        )
    if residual:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
