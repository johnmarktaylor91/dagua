"""Round 41 Reingold-Tilford smoke fidelity harness."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import igraph as ig
import torch

from dagua.layout.ops.coordinate import ReingoldTilfordTree, ReingoldTilfordTreeConfig
from dagua.layout.ops.pipelines.reingold_tilford import layout_reingold_tilford_pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState

TopologyBuilder = Callable[[int], Tuple[int, List[Tuple[int, int]]]]


def _edge_index(edges: Sequence[Tuple[int, int]]) -> torch.Tensor:
    """Build a Dagua edge tensor from Python edge pairs.

    Parameters
    ----------
    edges : sequence of tuple of int
        Directed graph edges.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    sources, targets = zip(*edges)
    return torch.tensor([sources, targets], dtype=torch.long)


def _path(seed: int) -> Tuple[int, List[Tuple[int, int]]]:
    """Build a path topology with seed-stable edge ordering.

    Parameters
    ----------
    seed : int
        Seed used to vary edge insertion order.

    Returns
    -------
    tuple
        Node count and directed edges.
    """
    edges = [(node, node + 1) for node in range(11)]
    if seed % 2:
        edges = list(reversed(edges))
    return 12, edges


def _star(seed: int) -> Tuple[int, List[Tuple[int, int]]]:
    """Build a star topology with deterministic leaf permutation.

    Parameters
    ----------
    seed : int
        Seed used to rotate leaf order.

    Returns
    -------
    tuple
        Node count and directed edges.
    """
    leaves = list(range(1, 13))
    shift = seed % len(leaves)
    leaves = leaves[shift:] + leaves[:shift]
    return 13, [(0, leaf) for leaf in leaves]


def _clustered(seed: int) -> Tuple[int, List[Tuple[int, int]]]:
    """Build a clustered two-root DAG topology.

    Parameters
    ----------
    seed : int
        Seed used to rotate sibling insertion order.

    Returns
    -------
    tuple
        Node count and directed edges.
    """
    edges = [
        (5, 6),
        (5, 7),
        (0, 1),
        (2, 3),
    ]
    shift = seed % len(edges)
    return 10, edges[shift:] + edges[:shift]


def _grid(seed: int) -> Tuple[int, List[Tuple[int, int]]]:
    """Build a small directed grid topology.

    Parameters
    ----------
    seed : int
        Seed used to choose row-major or column-major edge insertion.

    Returns
    -------
    tuple
        Node count and directed edges.
    """
    width = 4
    height = 3
    horizontal = [
        (row * width + col, row * width + col + 1)
        for row in range(height)
        for col in range(width - 1)
    ]
    vertical = [
        (row * width + col, (row + 1) * width + col)
        for row in range(height - 1)
        for col in range(width)
    ]
    edges = horizontal + vertical if seed % 2 == 0 else vertical + horizontal
    return width * height, edges


def _reference_layout(num_nodes: int, edges: Sequence[Tuple[int, int]]) -> torch.Tensor:
    """Run the python-igraph Reingold-Tilford reference adapter.

    Parameters
    ----------
    num_nodes : int
        Number of graph vertices.
    edges : sequence of tuple of int
        Directed graph edges.

    Returns
    -------
    torch.Tensor
        Scaled reference positions with shape ``[N, 2]``.
    """
    graph = ig.Graph(directed=True)
    graph.add_vertices(num_nodes)
    graph.add_edges(list(edges))
    layout = graph.layout("reingold_tilford", mode="out")
    positions = torch.zeros((num_nodes, 2), dtype=torch.float32)
    for node in range(num_nodes):
        positions[node, 0] = float(layout[node][0]) * 50.0
        positions[node, 1] = float(layout[node][1]) * 50.0
    return positions


def _previous_layout(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Run the pre-R41 coordinate-op fidelity path directly.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph vertices.

    Returns
    -------
    torch.Tensor
        Previous Dagua fidelity positions with shape ``[N, 2]``.
    """
    problem = LayoutProblem(edge_index=edge_index, num_nodes=num_nodes)
    state = SolveState()
    context = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = ReingoldTilfordTree(ReingoldTilfordTreeConfig(fidelity_mode="igraph")).apply(
        problem, state, context
    )
    if final_state.pos is None:
        raise RuntimeError("Previous RT path did not produce positions.")
    return final_state.pos


def _procrustes_rmsd(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    """Compute scale-normalized best-reflection Procrustes RMSD.

    Parameters
    ----------
    candidate : torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    reference : torch.Tensor
        Reference positions with shape ``[N, 2]``.

    Returns
    -------
    float
        Aligned RMSD.
    """
    left = candidate.to(dtype=torch.float64) - candidate.to(dtype=torch.float64).mean(
        dim=0, keepdim=True
    )
    right = reference.to(dtype=torch.float64) - reference.to(dtype=torch.float64).mean(
        dim=0, keepdim=True
    )
    left_norm = left.norm()
    right_norm = right.norm()
    if float(left_norm.item()) > 0.0:
        left = left / left_norm
    if float(right_norm.item()) > 0.0:
        right = right / right_norm
    u_matrix, _, vh_matrix = torch.linalg.svd(left.t() @ right)
    rotation = u_matrix @ vh_matrix
    reflected = left @ rotation
    reflected_rmsd = torch.sqrt(torch.mean(torch.sum((reflected - right).square(), dim=1)))

    determinant = torch.det(u_matrix @ vh_matrix)
    correction = torch.diag(
        torch.tensor([1.0, float(torch.sign(determinant).item())], dtype=left.dtype)
    )
    proper = left @ (u_matrix @ correction @ vh_matrix)
    proper_rmsd = torch.sqrt(torch.mean(torch.sum((proper - right).square(), dim=1)))
    return float(torch.minimum(reflected_rmsd, proper_rmsd).item())


def run_smoke(output_dir: Path) -> List[Dict[str, str]]:
    """Run the round 41 smoke comparison and write a CSV table.

    Parameters
    ----------
    output_dir : Path
        Directory where ``smoke_rmsd.csv`` will be written.

    Returns
    -------
    list of dict
        CSV rows for summary rendering.
    """
    builders: Dict[str, TopologyBuilder] = {
        "path": _path,
        "star": _star,
        "clustered": _clustered,
        "grid": _grid,
    }
    rows: List[Dict[str, str]] = []
    for topology, builder in builders.items():
        for seed in (41, 42, 43):
            num_nodes, edges = builder(seed)
            edge_index = _edge_index(edges)
            reference = _reference_layout(num_nodes=num_nodes, edges=edges)
            previous = _previous_layout(edge_index=edge_index, num_nodes=num_nodes)
            current = layout_reingold_tilford_pipeline(
                edge_index=edge_index,
                num_nodes=num_nodes,
                fidelity_mode="igraph",
            )
            previous_rmsd = _procrustes_rmsd(previous, reference)
            current_rmsd = _procrustes_rmsd(current, reference)
            rows.append(
                {
                    "topology": topology,
                    "seed": str(seed),
                    "nodes": str(num_nodes),
                    "edges": str(len(edges)),
                    "before_rmsd": f"{previous_rmsd:.9f}",
                    "after_rmsd": f"{current_rmsd:.9f}",
                    "dominant_component": "contour_offset_kernel",
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "smoke_rmsd.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    """Run the smoke harness and print before/after aggregate RMSD."""
    output_dir = Path("eval_output/algo_fidelity/round_41/reingold_tilford")
    rows = run_smoke(output_dir=output_dir)
    before = [float(row["before_rmsd"]) for row in rows]
    after = [float(row["after_rmsd"]) for row in rows]
    print(f"before_mean={statistics.mean(before):.9f}")
    print(f"after_mean={statistics.mean(after):.9f}")
    print(f"after_max={max(after):.9f}")
    print(f"wrote={output_dir / 'smoke_rmsd.csv'}")


if __name__ == "__main__":
    main()
