"""Round 41 ForceAtlas2 reference-fidelity smoke harness."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import scipy.sparse as sp
import torch
from fa2 import ForceAtlas2

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dagua.layout.ops.pipelines.fa2 import layout_fa2_pipeline  # noqa: E402
from scripts.algo_fidelity_cross import fidelity_procrustes  # noqa: E402


@dataclass(frozen=True)
class SmokeGraph:
    """Small graph used by the FA2 reference smoke harness.

    Parameters
    ----------
    name : str
        Human-readable graph name.
    num_nodes : int
        Number of nodes in the graph.
    edges : list[tuple[int, int]]
        Undirected graph edges in reference insertion order.
    """

    name: str
    num_nodes: int
    edges: list[tuple[int, int]]


def edge_index_from_edges(edges: Iterable[tuple[int, int]]) -> torch.Tensor:
    """Build a Dagua edge tensor from an edge iterable.

    Parameters
    ----------
    edges : Iterable[tuple[int, int]]
        Edge pairs as ``(source, target)`` tuples.

    Returns
    -------
    torch.Tensor
        Edge index tensor with shape ``[2, E]``.
    """
    edge_list = list(edges)
    if not edge_list:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(list(zip(*edge_list)), dtype=torch.long)


def sparse_matrix_from_edges(num_nodes: int, edges: Iterable[tuple[int, int]]) -> sp.csr_matrix:
    """Build the symmetric SciPy matrix consumed by the FA2 reference.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    edges : Iterable[tuple[int, int]]
        Undirected graph edges.

    Returns
    -------
    scipy.sparse.csr_matrix
        Symmetric adjacency matrix with shape ``[N, N]``.
    """
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for source, target in edges:
        if source == target:
            continue
        rows.extend([source, target])
        cols.extend([target, source])
        data.extend([1.0, 1.0])
    return sp.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))


def smoke_graphs() -> list[SmokeGraph]:
    """Return the four required FA2 smoke topologies.

    Returns
    -------
    list[SmokeGraph]
        Path, star, clustered, and grid topologies.
    """
    clustered_edges: list[tuple[int, int]] = []
    for base in (0, 6):
        for source in range(base, base + 6):
            for target in range(source + 1, base + 6):
                clustered_edges.append((source, target))
    clustered_edges.append((5, 6))

    grid_edges: list[tuple[int, int]] = []
    width = 4
    height = 4
    for row in range(height):
        for col in range(width):
            node = row * width + col
            if col + 1 < width:
                grid_edges.append((node, node + 1))
            if row + 1 < height:
                grid_edges.append((node, node + width))

    return [
        SmokeGraph("path", 12, [(node, node + 1) for node in range(11)]),
        SmokeGraph("star", 12, [(0, node) for node in range(1, 12)]),
        SmokeGraph("clustered", 12, clustered_edges),
        SmokeGraph("grid", 16, grid_edges),
    ]


def reference_layout(
    graph: SmokeGraph,
    seed: int,
    *,
    iterations: int,
    barnes_hut: bool,
) -> torch.Tensor:
    """Run the installed ``fa2`` reference layout.

    Parameters
    ----------
    graph : SmokeGraph
        Graph to lay out.
    seed : int
        Reference RNG seed.
    iterations : int
        Number of ForceAtlas2 iterations.
    barnes_hut : bool
        Whether to enable Barnes-Hut optimization.

    Returns
    -------
    torch.Tensor
        Reference coordinates with shape ``[N, 2]`` and dtype ``float64``.
    """
    engine = ForceAtlas2(
        outboundAttractionDistribution=True,
        edgeWeightInfluence=1.0,
        jitterTolerance=1.0,
        barnesHutOptimize=barnes_hut,
        barnesHutTheta=1.2,
        scalingRatio=2.0,
        strongGravityMode=False,
        gravity=1.0,
        seed=seed,
        verbose=False,
    )
    matrix = sparse_matrix_from_edges(graph.num_nodes, graph.edges).tolil()
    return torch.tensor(engine.forceatlas2(matrix, iterations=iterations), dtype=torch.float64)


def dagua_layout(
    graph: SmokeGraph,
    seed: int,
    *,
    iterations: int,
    barnes_hut: bool,
) -> torch.Tensor:
    """Run Dagua FA2 in reference-fidelity mode.

    Parameters
    ----------
    graph : SmokeGraph
        Graph to lay out.
    seed : int
        Dagua RNG seed.
    iterations : int
        Number of ForceAtlas2 iterations.
    barnes_hut : bool
        Whether to enable Barnes-Hut optimization.

    Returns
    -------
    torch.Tensor
        Dagua coordinates with shape ``[N, 2]`` and dtype ``float64``.
    """
    return (
        layout_fa2_pipeline(
            edge_index_from_edges(graph.edges),
            graph.num_nodes,
            steps=iterations,
            seed=seed,
            outbound_attraction_distribution=True,
            barnes_hut=barnes_hut,
            fidelity_mode=True,
        )
        .detach()
        .cpu()
        .to(dtype=torch.float64)
    )


def run_smoke(output_path: Optional[Path] = None) -> list[dict[str, float | int | str | bool]]:
    """Run the FA2 smoke comparison and optionally write CSV rows.

    Parameters
    ----------
    output_path : pathlib.Path, optional
        CSV path to write. When omitted, rows are only printed.

    Returns
    -------
    list[dict[str, float | int | str | bool]]
        Smoke result rows containing topology, seed, Barnes-Hut flag, RMSD, and
        max absolute coordinate delta.
    """
    rows: list[dict[str, float | int | str | bool]] = []
    for barnes_hut in (False, True):
        for graph in smoke_graphs():
            for seed in (0, 1, 2):
                dagua = dagua_layout(graph, seed, iterations=200, barnes_hut=barnes_hut)
                reference = reference_layout(graph, seed, iterations=200, barnes_hut=barnes_hut)
                rmsd, _ = fidelity_procrustes(dagua, reference)
                rows.append(
                    {
                        "topology": graph.name,
                        "seed": seed,
                        "barnes_hut": barnes_hut,
                        "rmsd": rmsd,
                        "max_abs": float((dagua - reference).abs().max().item()),
                    }
                )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        header = "topology,seed,barnes_hut,rmsd,max_abs\n"
        body = "\n".join(
            f"{row['topology']},{row['seed']},{row['barnes_hut']},"
            f"{row['rmsd']:.17g},{row['max_abs']:.17g}"
            for row in rows
        )
        output_path.write_text(header + body + "\n", encoding="utf-8")
    return rows


def main() -> None:
    """Run the smoke harness from the command line.

    Returns
    -------
    None
        Results are printed and written to ``smoke_after.csv``.
    """
    output_path = Path(__file__).with_name("smoke_after.csv")
    rows = run_smoke(output_path)
    for row in rows:
        print(
            f"{row['topology']:9s} seed={row['seed']} bh={row['barnes_hut']} "
            f"rmsd={row['rmsd']:.12g} max_abs={row['max_abs']:.12g}"
        )
    mean_rmsd = sum(float(row["rmsd"]) for row in rows) / len(rows)
    max_rmsd = max(float(row["rmsd"]) for row in rows)
    print(f"overall mean={mean_rmsd:.12g} max={max_rmsd:.12g}")


if __name__ == "__main__":
    main()
