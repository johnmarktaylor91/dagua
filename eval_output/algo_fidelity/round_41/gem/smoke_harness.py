"""Round 41 GEM fidelity smoke harness against the OGDF runner."""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dagua.layout.ops.pipelines.gem import layout_gem_pipeline  # noqa: E402
from scripts.algo_fidelity_cross import fidelity_procrustes  # noqa: E402


@dataclass(frozen=True)
class SmokeGraph:
    """Small graph used for GEM reference fidelity diagnostics.

    Parameters
    ----------
    name : str
        Stable topology label.
    num_nodes : int
        Number of nodes in the graph.
    edges : Sequence[tuple[int, int]]
        Edge list in runner insertion order.
    """

    name: str
    num_nodes: int
    edges: Sequence[tuple[int, int]]


def smoke_graphs() -> tuple[SmokeGraph, ...]:
    """Build the four required round 41 GEM smoke topologies.

    Returns
    -------
    tuple[SmokeGraph, ...]
        Path, star, clustered, and grid test graphs.
    """
    grid_edges = [(row * 3 + col, row * 3 + col + 1) for row in range(3) for col in range(2)] + [
        (row * 3 + col, (row + 1) * 3 + col) for row in range(2) for col in range(3)
    ]
    return (
        SmokeGraph("path", 6, tuple((index, index + 1) for index in range(5))),
        SmokeGraph("star", 7, tuple((0, index) for index in range(1, 7))),
        SmokeGraph(
            "clustered",
            8,
            (
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 4),
                (3, 4),
            ),
        ),
        SmokeGraph("grid", 9, tuple(grid_edges)),
    )


def edge_tensor(edges: Sequence[tuple[int, int]]) -> torch.Tensor:
    """Convert a Python edge list into Dagua ``edge_index`` format.

    Parameters
    ----------
    edges : Sequence[tuple[int, int]]
        Edge list in source-target order.

    Returns
    -------
    torch.Tensor
        Long tensor with shape ``[2, E]``.
    """
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def run_ogdf(graph: SmokeGraph, seed: int) -> torch.Tensor:
    """Run the standalone OGDF GEM reference adapter.

    Parameters
    ----------
    graph : SmokeGraph
        Smoke graph to lay out.
    seed : int
        Seed forwarded to the OGDF runner.

    Returns
    -------
    torch.Tensor
        OGDF position tensor with shape ``[N, 2]``.
    """
    payload = json.dumps(
        {
            "nodes": graph.num_nodes,
            "edges": [list(edge) for edge in graph.edges],
            "algorithm": "gem",
            "seed": int(seed),
        }
    )
    result = subprocess.run(
        [str(REPO_ROOT / "scripts" / "ogdf_runner")],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    )
    positions = json.loads(result.stdout)["positions"]
    return torch.tensor(positions, dtype=torch.float32)


def run_dagua(graph: SmokeGraph, seed: int) -> torch.Tensor:
    """Run Dagua's GEM pipeline in OGDF fidelity mode.

    Parameters
    ----------
    graph : SmokeGraph
        Smoke graph to lay out.
    seed : int
        Seed forwarded to the Dagua pipeline.

    Returns
    -------
    torch.Tensor
        Dagua position tensor with shape ``[N, 2]``.
    """
    return layout_gem_pipeline(
        edge_index=edge_tensor(graph.edges),
        num_nodes=graph.num_nodes,
        max_iters=30_000,
        seed=int(seed),
        fidelity_mode=True,
    )


def main() -> None:
    """Run the smoke comparison and print a markdown table.

    Returns
    -------
    None
        Results are printed to stdout.
    """
    rows: list[tuple[str, int, float, float]] = []
    for graph in smoke_graphs():
        for seed in (42, 43, 44):
            dagua_pos = run_dagua(graph, seed)
            ogdf_pos = run_ogdf(graph, seed)
            rmsd, _ = fidelity_procrustes(dagua_pos, ogdf_pos)
            direct = float(torch.sqrt(torch.mean((dagua_pos - ogdf_pos).square())).item())
            rows.append((graph.name, seed, rmsd, direct))

    print("| graph | seed | procrustes_rmsd | direct_rmsd |")
    print("|---|---:|---:|---:|")
    for graph_name, seed, rmsd, direct in rows:
        print(f"| {graph_name} | {seed} | {rmsd:.9f} | {direct:.9f} |")
    mean_rmsd = statistics.fmean(row[2] for row in rows)
    print(f"\noverall_mean_procrustes_rmsd={mean_rmsd:.9f}")


if __name__ == "__main__":
    main()
