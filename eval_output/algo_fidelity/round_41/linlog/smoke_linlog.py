"""Round 41 LinLog fidelity smoke harness."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import torch

from dagua.eval.competitors.linlog_competitor import _layout_linlog_reference, _resolve_config
from dagua.layout.ops.pipelines.linlog import layout_linlog_pipeline


def _edge_index(edges: Iterable[tuple[int, int]]) -> torch.Tensor:
    """Build a ``[2, E]`` edge-index tensor.

    Parameters
    ----------
    edges : Iterable[tuple[int, int]]
        Edge endpoint pairs in graph-file order.

    Returns
    -------
    torch.Tensor
        Edge index tensor with shape ``[2, E]``.
    """
    edge_list = list(edges)
    if not edge_list:
        return torch.empty((2, 0), dtype=torch.long)
    sources, targets = zip(*edge_list)
    return torch.tensor([sources, targets], dtype=torch.long)


def _grid_edges(rows: int, cols: int) -> torch.Tensor:
    """Build a rectangular grid edge-index tensor.

    Parameters
    ----------
    rows : int
        Number of grid rows.
    cols : int
        Number of grid columns.

    Returns
    -------
    torch.Tensor
        Edge index tensor with shape ``[2, E]``.
    """
    edges: list[tuple[int, int]] = []
    for row in range(rows):
        for col in range(cols):
            node = row * cols + col
            if col + 1 < cols:
                edges.append((node, node + 1))
            if row + 1 < rows:
                edges.append((node, node + cols))
    return _edge_index(edges)


def _procrustes_rmsd(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Compute scale-normalized two-dimensional Procrustes RMSD.

    Parameters
    ----------
    actual : torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    expected : torch.Tensor
        Reference positions with shape ``[N, 2]``.

    Returns
    -------
    float
        Root-mean-square deviation after centering, unit-norm scaling, and
        optimal rotation/reflection.
    """
    actual64 = actual.to(torch.float64)
    expected64 = expected.to(torch.float64)
    if actual64.numel() == 0:
        return 0.0

    actual64 = actual64 - actual64.mean(dim=0, keepdim=True)
    expected64 = expected64 - expected64.mean(dim=0, keepdim=True)
    actual_norm = torch.linalg.norm(actual64)
    expected_norm = torch.linalg.norm(expected64)
    if float(actual_norm) == 0.0 or float(expected_norm) == 0.0:
        delta = actual64 - expected64
        return float(torch.sqrt((delta * delta).sum(dim=1).mean()).item())

    actual64 = actual64 / actual_norm
    expected64 = expected64 / expected_norm
    left, _, right_t = torch.linalg.svd(actual64.T @ expected64)
    aligned = actual64 @ (left @ right_t)
    delta = aligned - expected64
    return float(torch.sqrt((delta * delta).sum(dim=1).mean()).item())


def _topologies() -> dict[str, tuple[int, torch.Tensor]]:
    """Return the fixed smoke topologies.

    Returns
    -------
    dict[str, tuple[int, torch.Tensor]]
        Mapping from topology name to node count and edge-index tensor.
    """
    return {
        "path": (8, _edge_index((node, node + 1) for node in range(7))),
        "star": (9, _edge_index((0, node) for node in range(1, 9))),
        "clustered": (
            10,
            _edge_index(
                [
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (3, 4),
                    (0, 2),
                    (1, 3),
                    (5, 6),
                    (6, 7),
                    (7, 8),
                    (8, 9),
                    (5, 7),
                    (6, 8),
                    (4, 5),
                ]
            ),
        ),
        "grid": (9, _grid_edges(3, 3)),
    }


def run_smoke(output_path: Path) -> None:
    """Run before/after LinLog smoke comparison and write CSV results.

    Parameters
    ----------
    output_path : Path
        Destination CSV path.

    Returns
    -------
    None
        The function writes results to ``output_path`` and prints a compact
        table for terminal use.
    """
    config = _resolve_config({"steps": 300, "a": 1.0, "r": 0.0})
    rows: list[dict[str, str]] = []
    for topology, (num_nodes, edge_index) in _topologies().items():
        for seed in (0, 1, 2):
            graph = SimpleNamespace(
                num_nodes=num_nodes,
                edge_index=edge_index,
                edge_weights=None,
                node_sizes=None,
            )
            reference = _layout_linlog_reference(graph=graph, config=config, seed=seed)
            before = layout_linlog_pipeline(
                edge_index=edge_index,
                num_nodes=num_nodes,
                steps=300,
                seed=seed,
                fidelity_mode=False,
            )
            after = layout_linlog_pipeline(
                edge_index=edge_index,
                num_nodes=num_nodes,
                steps=300,
                seed=seed,
                fidelity_mode=True,
            )
            before_rmsd = _procrustes_rmsd(before, reference)
            after_rmsd = _procrustes_rmsd(after, reference)
            rows.append(
                {
                    "topology": topology,
                    "seed": str(seed),
                    "before_rmsd": f"{before_rmsd:.9f}",
                    "after_rmsd": f"{after_rmsd:.9f}",
                    "after_max_abs": f"{float((after - reference).abs().max().item()):.9f}",
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["topology", "seed", "before_rmsd", "after_rmsd", "after_max_abs"],
        )
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['topology']:9s} seed={row['seed']} "
            f"before={row['before_rmsd']} after={row['after_rmsd']} "
            f"max_abs={row['after_max_abs']}"
        )
    mean_before = sum(float(row["before_rmsd"]) for row in rows) / len(rows)
    mean_after = sum(float(row["after_rmsd"]) for row in rows) / len(rows)
    print(f"mean before={mean_before:.9f} after={mean_after:.9f}")


if __name__ == "__main__":
    run_smoke(Path(__file__).with_name("smoke_rmsd.csv"))
