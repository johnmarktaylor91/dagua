"""Round 62 Davidson-Harel pure-port fidelity smoke harness."""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dagua.layout.ops.pipelines.davidson_harel import layout_davidson_harel_pipeline  # noqa: E402

_OUTPUT_DIR = Path("eval_output/algo_fidelity/round_62/davidson_harel")
_ROUNDS = 1
_SEEDS = (0, 1, 2)


def _edge_index_from_edges(edges: list[tuple[int, int]]) -> torch.Tensor:
    """Build a PyG-style edge tensor from an ordered edge list.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Directed graph edges in file/reference order.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    sources, targets = zip(*edges)
    return torch.tensor([sources, targets], dtype=torch.long)


def _path_graph() -> tuple[int, torch.Tensor]:
    """Return the path smoke topology.

    Returns
    -------
    tuple[int, torch.Tensor]
        Node count and edge tensor.
    """
    num_nodes = 8
    return num_nodes, _edge_index_from_edges([(index, index + 1) for index in range(num_nodes - 1)])


def _star_graph() -> tuple[int, torch.Tensor]:
    """Return the star smoke topology.

    Returns
    -------
    tuple[int, torch.Tensor]
        Node count and edge tensor.
    """
    num_nodes = 9
    return num_nodes, _edge_index_from_edges([(0, index) for index in range(1, num_nodes)])


def _clustered_graph() -> tuple[int, torch.Tensor]:
    """Return a two-cluster smoke topology with sparse bridges.

    Returns
    -------
    tuple[int, torch.Tensor]
        Node count and edge tensor.
    """
    left = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    right = [(4, 5), (5, 6), (6, 7), (7, 4), (4, 6)]
    bridges = [(2, 4), (3, 5)]
    return 8, _edge_index_from_edges(left + right + bridges)


def _grid_graph() -> tuple[int, torch.Tensor]:
    """Return a 3-by-3 grid smoke topology.

    Returns
    -------
    tuple[int, torch.Tensor]
        Node count and edge tensor.
    """
    edges: list[tuple[int, int]] = []
    width = 3
    height = 3
    for row in range(height):
        for col in range(width):
            node = row * width + col
            if col + 1 < width:
                edges.append((node, node + 1))
            if row + 1 < height:
                edges.append((node, node + width))
    return width * height, _edge_index_from_edges(edges)


def _reference_positions(edge_index: torch.Tensor, num_nodes: int, seed: int) -> torch.Tensor:
    """Run python-igraph's Davidson-Harel reference adapter.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    seed : int
        Stochastic layout seed.

    Returns
    -------
    torch.Tensor
        Reference coordinates with shape ``[N, 2]`` in Dagua scale.
    """
    import igraph

    graph = igraph.Graph(directed=True)
    graph.add_vertices(num_nodes)
    if edge_index.numel() > 0:
        edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
        graph.add_edges(
            [
                (int(edge_index_cpu[0, edge_id].item()), int(edge_index_cpu[1, edge_id].item()))
                for edge_id in range(edge_index_cpu.shape[1])
            ]
        )

    kwargs = {
        "seed": np.random.RandomState(seed).uniform(-1.0, 1.0, size=(num_nodes, 2)).tolist(),
        "maxiter": _ROUNDS,
    }
    igraph.set_random_number_generator(random.Random(seed))
    try:
        layout = graph.layout("davidson_harel", **kwargs)
    finally:
        igraph.set_random_number_generator(None)
    return torch.tensor(np.asarray(layout, dtype=np.float64) * 50.0, dtype=torch.float64)


def _rmsd(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    """Compute reflection-tolerant Procrustes RMSD for one layout pair.

    Parameters
    ----------
    candidate : torch.Tensor
        Candidate coordinates with shape ``[N, 2]``.
    reference : torch.Tensor
        Reference coordinates with shape ``[N, 2]``.

    Returns
    -------
    float
        Scale-normalized Procrustes RMSD.
    """
    candidate_centered = candidate - candidate.mean(dim=0, keepdim=True)
    reference_centered = reference - reference.mean(dim=0, keepdim=True)
    candidate_norm = float(candidate_centered.norm().item())
    reference_norm = float(reference_centered.norm().item())
    if candidate_norm > 0.0:
        candidate_centered = candidate_centered / candidate_norm
    if reference_norm > 0.0:
        reference_centered = reference_centered / reference_norm

    covariance = candidate_centered.t() @ reference_centered
    left_singular, _, right_singular_t = torch.linalg.svd(covariance)
    det_value = torch.det(left_singular @ right_singular_t)
    correction = torch.diag(
        torch.tensor([1.0, float(torch.sign(det_value).item())], dtype=candidate_centered.dtype)
    )
    rotation = left_singular @ correction @ right_singular_t
    aligned = candidate_centered @ rotation
    rmsd = float(torch.sqrt(((aligned - reference_centered).square()).sum(dim=1).mean()).item())

    reflected_rotation = left_singular @ right_singular_t
    reflected_aligned = candidate_centered @ reflected_rotation
    reflected_rmsd = float(
        torch.sqrt(((reflected_aligned - reference_centered).square()).sum(dim=1).mean()).item()
    )
    return min(rmsd, reflected_rmsd)


def main() -> None:
    """Run all smoke cases and write the CSV result table.

    Returns
    -------
    None
        Results are written to ``smoke_rmsd.csv`` and summarized on stdout.
    """
    topologies: dict[str, Callable[[], tuple[int, torch.Tensor]]] = {
        "path": _path_graph,
        "star": _star_graph,
        "clustered": _clustered_graph,
        "grid": _grid_graph,
    }
    rows: list[dict[str, object]] = []
    for topology, builder in topologies.items():
        num_nodes, edge_index = builder()
        for seed in _SEEDS:
            reference = _reference_positions(edge_index, num_nodes, seed)
            after = layout_davidson_harel_pipeline(
                edge_index=edge_index,
                num_nodes=num_nodes,
                rounds=_ROUNDS,
                seed=seed,
                fidelity_mode=True,
                fidelity_dtype=torch.float64,
            )
            rows.append(
                {
                    "topology": topology,
                    "seed": seed,
                    "rmsd": _rmsd(after, reference),
                    "max_abs": float(torch.max(torch.abs(after - reference)).item()),
                }
            )

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (_OUTPUT_DIR / "smoke_rmsd.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["topology", "seed", "rmsd", "max_abs"])
        writer.writeheader()
        writer.writerows(rows)

    max_rmsd = max(float(row["rmsd"]) for row in rows)
    max_abs = max(float(row["max_abs"]) for row in rows)
    print(f"max_rmsd={max_rmsd:.12g}")
    print(f"max_abs={max_abs:.12g}")


if __name__ == "__main__":
    main()
