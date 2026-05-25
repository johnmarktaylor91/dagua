"""Round 41 classical-MDS smoke comparison against OGDF PivotMDS."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dagua.layout.ops.pipelines.classical_mds import layout_classical_mds_pipeline  # noqa: E402
from scripts.algo_fidelity_cross import fidelity_procrustes  # noqa: E402


@dataclass(frozen=True)
class SmokeGraph:
    """One graph topology for the Round 41 smoke harness.

    Parameters
    ----------
    name : str
        Stable topology name.
    num_nodes : int
        Number of graph nodes.
    edges : list[list[int]]
        Edge list shaped like ``[[source, target], ...]``.
    """

    name: str
    num_nodes: int
    edges: list[list[int]]


def _path_graph() -> SmokeGraph:
    """Build the path smoke graph.

    Returns
    -------
    SmokeGraph
        Eight-node path graph.
    """
    return SmokeGraph("path", 8, [[node, node + 1] for node in range(7)])


def _star_graph() -> SmokeGraph:
    """Build the star smoke graph.

    Returns
    -------
    SmokeGraph
        Nine-node star graph.
    """
    return SmokeGraph("star", 9, [[0, node] for node in range(1, 9)])


def _clustered_graph() -> SmokeGraph:
    """Build the clustered smoke graph.

    Returns
    -------
    SmokeGraph
        Two four-clique clusters joined by bridge edges.
    """
    edges: list[list[int]] = []
    for base in (0, 4):
        for source in range(base, base + 4):
            for target in range(source + 1, base + 4):
                edges.append([source, target])
    edges.extend([[3, 4], [2, 5]])
    return SmokeGraph("clustered", 8, edges)


def _grid_graph() -> SmokeGraph:
    """Build the grid smoke graph.

    Returns
    -------
    SmokeGraph
        Three-by-three grid graph.
    """
    edges: list[list[int]] = []
    for row in range(3):
        for col in range(3):
            node = row * 3 + col
            if col + 1 < 3:
                edges.append([node, node + 1])
            if row + 1 < 3:
                edges.append([node, node + 3])
    return SmokeGraph("grid", 9, edges)


def _edge_index(edges: list[list[int]]) -> torch.Tensor:
    """Convert an edge list to Dagua's edge-index tensor.

    Parameters
    ----------
    edges : list[list[int]]
        Edge list shaped like ``[[source, target], ...]``.

    Returns
    -------
    torch.Tensor
        Edge index tensor with shape ``[2, E]``.
    """
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _ogdf_positions(graph: SmokeGraph, seed: int) -> torch.Tensor:
    """Run OGDF PivotMDS with all nodes as pivots.

    Parameters
    ----------
    graph : SmokeGraph
        Smoke graph to lay out.
    seed : int
        Runner seed. PivotMDS's internal SVD seed is fixed, but this keeps the
        harness aligned with multi-seed fidelity infrastructure.

    Returns
    -------
    torch.Tensor
        OGDF position tensor with shape ``[N, 2]``.
    """
    payload = json.dumps(
        {
            "nodes": graph.num_nodes,
            "edges": graph.edges,
            "algorithm": "pivot_mds",
            "seed": seed,
            "numberOfPivots": graph.num_nodes,
        }
    )
    result = subprocess.run(
        [str(REPO_ROOT / "scripts" / "ogdf_runner")],
        input=payload,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "OGDF runner failed")
    return torch.tensor(json.loads(result.stdout)["positions"], dtype=torch.float32)


def _dagua_positions(graph: SmokeGraph, seed: int, *, ogdf_fidelity: bool) -> torch.Tensor:
    """Run Dagua classical MDS for one smoke graph.

    Parameters
    ----------
    graph : SmokeGraph
        Smoke graph to lay out.
    seed : int
        Dagua layout seed.
    ogdf_fidelity : bool
        Whether to use the Round 41 OGDF fidelity path.

    Returns
    -------
    torch.Tensor
        Dagua position tensor with shape ``[N, 2]``.
    """
    return layout_classical_mds_pipeline(
        edge_index=_edge_index(graph.edges),
        num_nodes=graph.num_nodes,
        seed=seed,
        ogdf_fidelity=ogdf_fidelity,
    )


def main() -> int:
    """Run the smoke comparison and print a Markdown table.

    Returns
    -------
    int
        Process exit status.
    """
    graphs = [_path_graph(), _star_graph(), _clustered_graph(), _grid_graph()]
    seeds = [0, 1, 2]
    print("| topology | seed | baseline RMSD | ogdf_fidelity RMSD |")
    print("|---|---:|---:|---:|")
    baseline_values: list[float] = []
    fidelity_values: list[float] = []
    for graph in graphs:
        for seed in seeds:
            reference = _ogdf_positions(graph, seed)
            baseline, _ = fidelity_procrustes(
                _dagua_positions(graph, seed, ogdf_fidelity=False),
                reference,
            )
            fidelity, _ = fidelity_procrustes(
                _dagua_positions(graph, seed, ogdf_fidelity=True),
                reference,
            )
            baseline_values.append(float(baseline))
            fidelity_values.append(float(fidelity))
            print(f"| {graph.name} | {seed} | {baseline:.9f} | {fidelity:.9f} |")
    print(
        f"| overall_mean | - | "
        f"{sum(baseline_values) / len(baseline_values):.9f} | "
        f"{sum(fidelity_values) / len(fidelity_values):.9f} |"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
