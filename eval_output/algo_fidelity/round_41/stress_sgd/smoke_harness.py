"""Round 41 Stress-SGD versus OGDF stress smoke harness."""

from __future__ import annotations

import csv
import json
import statistics
import subprocess
import sys
import types
from pathlib import Path
from typing import Iterable

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = Path("eval_output/algo_fidelity/round_41/stress_sgd")
SEEDS = (11, 42, 97)


def install_dagua_namespace_stubs() -> None:
    """Install namespace stubs to avoid unrelated package-level imports.

    Returns
    -------
    None
        ``sys.modules`` is populated with package shells pointing at the local
        source tree.
    """
    package_paths = {
        "dagua": REPO_ROOT / "dagua",
        "dagua.layout": REPO_ROOT / "dagua" / "layout",
        "dagua.layout.ops": REPO_ROOT / "dagua" / "layout" / "ops",
        "dagua.layout.ops.pipelines": REPO_ROOT / "dagua" / "layout" / "ops" / "pipelines",
    }
    for package_name, package_path in package_paths.items():
        if package_name in sys.modules:
            continue
        module = types.ModuleType(package_name)
        module.__path__ = [str(package_path)]  # type: ignore[attr-defined]
        sys.modules[package_name] = module


install_dagua_namespace_stubs()

from dagua.layout.ops.pipelines.stress_sgd import layout_stress_sgd_pipeline  # noqa: E402


def edge_index_from_edges(edges: Iterable[tuple[int, int]]) -> torch.Tensor:
    """Build an edge-index tensor from edge tuples.

    Parameters
    ----------
    edges : Iterable[tuple[int, int]]
        Graph edges as ``(source, target)`` pairs.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    edge_list = list(edges)
    if not edge_list:
        return torch.empty((2, 0), dtype=torch.long)
    sources, targets = zip(*edge_list)
    return torch.tensor([sources, targets], dtype=torch.long)


def smoke_graphs() -> dict[str, tuple[int, torch.Tensor]]:
    """Return the four required smoke topologies.

    Returns
    -------
    dict[str, tuple[int, torch.Tensor]]
        Mapping from topology name to ``(num_nodes, edge_index)``.
    """
    path_edges = [(index, index + 1) for index in range(7)]
    star_edges = [(0, index) for index in range(1, 9)]
    clustered_edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (8, 9),
        (9, 10),
        (10, 11),
        (11, 8),
        (3, 4),
        (7, 8),
    ]
    grid_edges = []
    width = 4
    height = 3
    for row in range(height):
        for col in range(width):
            node = row * width + col
            if col + 1 < width:
                grid_edges.append((node, node + 1))
            if row + 1 < height:
                grid_edges.append((node, node + width))
    return {
        "path": (8, edge_index_from_edges(path_edges)),
        "star": (9, edge_index_from_edges(star_edges)),
        "clustered": (12, edge_index_from_edges(clustered_edges)),
        "grid": (width * height, edge_index_from_edges(grid_edges)),
    }


def procrustes_rmsd(left: torch.Tensor, right: torch.Tensor) -> float:
    """Compute scale-normalized Procrustes RMSD.

    Parameters
    ----------
    left : torch.Tensor
        First position tensor with shape ``[N, 2]``.
    right : torch.Tensor
        Second position tensor with shape ``[N, 2]``.

    Returns
    -------
    float
        Root-mean-square deviation after centering, scaling, rotation, and
        optional reflection.
    """
    left64 = left.to(dtype=torch.float64)
    right64 = right.to(dtype=torch.float64)
    left_centered = left64 - left64.mean(dim=0, keepdim=True)
    right_centered = right64 - right64.mean(dim=0, keepdim=True)
    left_scale = torch.linalg.norm(left_centered)
    right_scale = torch.linalg.norm(right_centered)
    if float(left_scale) == 0.0 or float(right_scale) == 0.0:
        return float(torch.sqrt(torch.mean((left_centered - right_centered).square())).item())
    left_centered = left_centered / left_scale
    right_centered = right_centered / right_scale
    covariance = left_centered.T @ right_centered
    u_matrix, _, vh_matrix = torch.linalg.svd(covariance)
    rotation = vh_matrix.T @ u_matrix.T
    aligned = left_centered @ rotation
    rmsd = torch.sqrt(torch.mean(torch.sum((aligned - right_centered).square(), dim=1)))
    reflection = torch.diag(torch.tensor([1.0, -1.0], dtype=torch.float64))
    reflected = left_centered @ (vh_matrix.T @ reflection @ u_matrix.T)
    reflected_rmsd = torch.sqrt(torch.mean(torch.sum((reflected - right_centered).square(), dim=1)))
    return float(min(rmsd, reflected_rmsd).item())


def run_ogdf_reference(edge_index: torch.Tensor, num_nodes: int, seed: int) -> torch.Tensor:
    """Run the local OGDF runner for one stress layout.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    seed : int
        Runner seed.

    Returns
    -------
    torch.Tensor
        Reference positions with shape ``[N, 2]``.
    """
    runner = REPO_ROOT / "scripts" / "ogdf_runner"
    edges = [
        [int(edge_index[0, index]), int(edge_index[1, index])]
        for index in range(edge_index.shape[1])
    ]
    payload = json.dumps(
        {
            "nodes": num_nodes,
            "edges": edges,
            "algorithm": "stress",
            "seed": seed,
            "iterations": 200,
        }
    )
    result = subprocess.run(
        [str(runner)],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
        timeout=30.0,
    )
    output = json.loads(result.stdout)
    return torch.tensor(output["positions"], dtype=torch.float32)


def run_smoke() -> list[dict[str, str]]:
    """Run baseline and OGDF-fidelity smoke comparisons.

    Returns
    -------
    list[dict[str, str]]
        CSV-ready result rows.
    """
    rows: list[dict[str, str]] = []
    for topology, (num_nodes, edge_index) in smoke_graphs().items():
        for seed in SEEDS:
            reference = run_ogdf_reference(edge_index=edge_index, num_nodes=num_nodes, seed=seed)
            baseline = layout_stress_sgd_pipeline(
                edge_index=edge_index,
                num_nodes=num_nodes,
                steps=200,
                seed=seed,
                fidelity_mode=True,
            )
            after = layout_stress_sgd_pipeline(
                edge_index=edge_index,
                num_nodes=num_nodes,
                steps=200,
                seed=seed,
                fidelity_mode="ogdf",
            )
            rows.append(
                {
                    "topology": topology,
                    "seed": str(seed),
                    "baseline_rmsd": f"{procrustes_rmsd(baseline, reference):.9f}",
                    "after_rmsd": f"{procrustes_rmsd(after, reference):.9f}",
                }
            )
    return rows


def write_rows(rows: list[dict[str, str]]) -> None:
    """Write smoke rows and print aggregate means.

    Parameters
    ----------
    rows : list[dict[str, str]]
        CSV-ready result rows.

    Returns
    -------
    None
        Results are written to ``round_41_smoke.csv``.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "round_41_smoke.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("topology", "seed", "baseline_rmsd", "after_rmsd"),
        )
        writer.writeheader()
        writer.writerows(rows)
    baseline = [float(row["baseline_rmsd"]) for row in rows]
    after = [float(row["after_rmsd"]) for row in rows]
    print(f"wrote {output_path}")
    print(f"baseline_mean={statistics.fmean(baseline):.9f}")
    print(f"after_mean={statistics.fmean(after):.9f}")
    print(f"after_max={max(after):.9f}")


def main() -> None:
    """Run the round 41 smoke harness."""
    write_rows(run_smoke())


if __name__ == "__main__":
    main()
