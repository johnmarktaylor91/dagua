#!/usr/bin/env python3
"""Round 41 GraphOpt same-seed igraph fidelity smoke harness."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import igraph as ig
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dagua.layout.ops.pipelines.graphopt import layout_graphopt_pipeline  # noqa: E402

DEFAULT_SEEDS: tuple[int, ...] = (42, 43, 44)
DEFAULT_NITER = 500
PASS_THRESHOLD = 0.005


@dataclass(frozen=True)
class Topology:
    """GraphOpt smoke topology definition."""

    name: str
    num_nodes: int
    edges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class SmokeRow:
    """One GraphOpt smoke comparison row."""

    topology: str
    seed: int
    niter: int
    init_rmsd: float
    one_iter_rmsd: float
    full_rmsd: float
    raw_full_rmsd: float
    diagnosis: str


def _edge_index(edges: Sequence[tuple[int, int]]) -> torch.Tensor:
    """Build an edge-index tensor from ordered edge pairs.

    Parameters
    ----------
    edges : Sequence[tuple[int, int]]
        Ordered graph edges as ``(source, target)`` pairs.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]`` and integer dtype.
    """
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    sources, targets = zip(*edges)
    return torch.tensor([list(sources), list(targets)], dtype=torch.long)


def _seed_matrix(num_nodes: int, seed: int) -> np.ndarray:
    """Return the seed matrix used by the igraph reference adapter.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    seed : int
        RandomState seed.

    Returns
    -------
    np.ndarray
        Initial coordinate matrix with shape ``[N, 2]``.
    """
    return np.random.RandomState(seed).uniform(-1.0, 1.0, size=(num_nodes, 2))


def _procrustes_rmsd(left: torch.Tensor, right: torch.Tensor) -> float:
    """Compute scale-normalized Procrustes RMSD.

    Parameters
    ----------
    left : torch.Tensor
        First coordinate tensor with shape ``[N, 2]``.
    right : torch.Tensor
        Second coordinate tensor with shape ``[N, 2]``.

    Returns
    -------
    float
        RMSD after centering, unit-Frobenius scaling, and best rotation or
        reflection.
    """
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}")
    if left.numel() == 0:
        return 0.0

    left_centered = left.to(dtype=torch.float64) - left.to(dtype=torch.float64).mean(
        dim=0,
        keepdim=True,
    )
    right_centered = right.to(dtype=torch.float64) - right.to(dtype=torch.float64).mean(
        dim=0,
        keepdim=True,
    )
    left_norm = left_centered.norm()
    right_norm = right_centered.norm()
    if float(left_norm.item()) > 0.0:
        left_centered = left_centered / left_norm
    if float(right_norm.item()) > 0.0:
        right_centered = right_centered / right_norm

    covariance = left_centered.t() @ right_centered
    left_singular, _, right_singular_t = torch.linalg.svd(covariance)
    rotation = left_singular @ right_singular_t
    reflected = left_centered @ rotation
    reflected_rmsd = torch.sqrt(((reflected - right_centered).square()).sum(dim=1).mean())

    det_value = torch.det(rotation)
    correction = torch.diag(
        torch.tensor([1.0, float(torch.sign(det_value).item())], dtype=torch.float64),
    )
    proper_rotation = left_singular @ correction @ right_singular_t
    aligned = left_centered @ proper_rotation
    proper_rmsd = torch.sqrt(((aligned - right_centered).square()).sum(dim=1).mean())
    return float(torch.minimum(proper_rmsd, reflected_rmsd).item())


def _raw_rmsd(left: torch.Tensor, right: torch.Tensor) -> float:
    """Compute unaligned coordinate RMSD.

    Parameters
    ----------
    left : torch.Tensor
        First coordinate tensor with shape ``[N, 2]``.
    right : torch.Tensor
        Second coordinate tensor with shape ``[N, 2]``.

    Returns
    -------
    float
        RMSD in the raw output coordinate frame.
    """
    delta = left.to(dtype=torch.float64) - right.to(dtype=torch.float64)
    return float(torch.sqrt(delta.square().sum(dim=1).mean()).item())


def _topologies() -> tuple[Topology, ...]:
    """Return the four required smoke topologies.

    Returns
    -------
    tuple[Topology, ...]
        Path, star, clustered, and grid topologies.
    """
    grid_edges = tuple(
        [(row * 3 + col, row * 3 + col + 1) for row in range(3) for col in range(2)]
        + [(row * 3 + col, (row + 1) * 3 + col) for row in range(2) for col in range(3)]
    )
    return (
        Topology("path", 8, tuple((index, index + 1) for index in range(7))),
        Topology("star", 9, tuple((0, index) for index in range(1, 9))),
        Topology(
            "clustered",
            10,
            ((0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (8, 9), (2, 7), (4, 5)),
        ),
        Topology("grid", 9, grid_edges),
    )


def _diagnose(init_rmsd: float, one_iter_rmsd: float, full_rmsd: float, raw_rmsd: float) -> str:
    """Label the dominant residual component for one run.

    Parameters
    ----------
    init_rmsd : float
        Procrustes RMSD for Dagua's seed-matrix initialization.
    one_iter_rmsd : float
        Procrustes RMSD after one GraphOpt iteration.
    full_rmsd : float
        Procrustes RMSD after the requested full iteration count.
    raw_rmsd : float
        Unaligned raw-coordinate RMSD after the requested full iteration count.

    Returns
    -------
    str
        Short diagnosis suitable for CSV output.
    """
    if full_rmsd < 1.0e-6 and raw_rmsd > 1.0:
        return "output_frame_only"
    if init_rmsd > max(one_iter_rmsd, full_rmsd):
        return "initialization"
    if one_iter_rmsd > full_rmsd:
        return "first_iteration_force_kernel"
    return "accumulated_force_or_order"


def run_smoke(seeds: Sequence[int], niter: int) -> list[SmokeRow]:
    """Run same-seed GraphOpt comparisons against python-igraph.

    Parameters
    ----------
    seeds : Sequence[int]
        Seeds to evaluate for every topology.
    niter : int
        Full GraphOpt iteration count.

    Returns
    -------
    list[SmokeRow]
        Per-topology, per-seed smoke comparison rows.
    """
    rows: list[SmokeRow] = []
    for topology in _topologies():
        edge_index = _edge_index(topology.edges)
        graph = ig.Graph(n=topology.num_nodes, edges=list(topology.edges), directed=True)
        for seed in seeds:
            initial_pos = _seed_matrix(topology.num_nodes, seed)
            initial_tensor = torch.as_tensor(initial_pos, dtype=torch.float64)
            dagua_initial = layout_graphopt_pipeline(
                edge_index=edge_index,
                num_nodes=topology.num_nodes,
                seed=seed,
                niter=0,
                initial_pos=initial_pos,
                fidelity_mode=True,
            )
            init_rmsd = _procrustes_rmsd(dagua_initial, initial_tensor)

            igraph_one = torch.as_tensor(
                graph.layout_graphopt(seed=initial_pos, niter=1),
                dtype=torch.float64,
            )
            dagua_one = layout_graphopt_pipeline(
                edge_index=edge_index,
                num_nodes=topology.num_nodes,
                seed=seed,
                niter=1,
                initial_pos=initial_pos,
                fidelity_mode=True,
            )
            one_iter_rmsd = _procrustes_rmsd(dagua_one, igraph_one)

            igraph_full = torch.as_tensor(
                graph.layout_graphopt(seed=initial_pos, niter=niter),
                dtype=torch.float64,
            )
            dagua_full = layout_graphopt_pipeline(
                edge_index=edge_index,
                num_nodes=topology.num_nodes,
                seed=seed,
                niter=niter,
                initial_pos=initial_pos,
                fidelity_mode=True,
            )
            full_rmsd = _procrustes_rmsd(dagua_full, igraph_full)
            raw_full_rmsd = _raw_rmsd(dagua_full, igraph_full)
            rows.append(
                SmokeRow(
                    topology=topology.name,
                    seed=seed,
                    niter=niter,
                    init_rmsd=init_rmsd,
                    one_iter_rmsd=one_iter_rmsd,
                    full_rmsd=full_rmsd,
                    raw_full_rmsd=raw_full_rmsd,
                    diagnosis=_diagnose(init_rmsd, one_iter_rmsd, full_rmsd, raw_full_rmsd),
                )
            )
    return rows


def write_outputs(rows: Sequence[SmokeRow], output_dir: Path) -> dict[str, object]:
    """Write smoke CSV and JSON summary.

    Parameters
    ----------
    rows : Sequence[SmokeRow]
        Smoke rows to persist.
    output_dir : Path
        Directory where ``smoke_rmsd.csv`` and ``smoke_summary.json`` are written.

    Returns
    -------
    dict[str, object]
        Summary object also written to disk.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "smoke_rmsd.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    full_values = [row.full_rmsd for row in rows if math.isfinite(row.full_rmsd)]
    topology_means = {
        topology: statistics.mean(row.full_rmsd for row in rows if row.topology == topology)
        for topology in sorted({row.topology for row in rows})
    }
    summary: dict[str, object] = {
        "rows": len(rows),
        "threshold": PASS_THRESHOLD,
        "overall_mean_full_rmsd": statistics.mean(full_values),
        "overall_max_full_rmsd": max(full_values),
        "topology_mean_full_rmsd": topology_means,
        "verdict": "pass" if statistics.mean(full_values) < PASS_THRESHOLD else "fail",
    }
    summary_path = output_dir / "smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line smoke harness.

    Parameters
    ----------
    argv : Sequence[str], optional
        Optional argument vector. ``None`` reads from ``sys.argv``.

    Returns
    -------
    int
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--niter", type=int, default=DEFAULT_NITER)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(argv)

    rows = run_smoke(seeds=args.seeds, niter=args.niter)
    summary = write_outputs(rows, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
