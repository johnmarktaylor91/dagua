"""Round 41 smoke harness for maxent-stress reference fidelity."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dagua.eval.competitors.ogdf_competitor import OGDFStress  # noqa: E402
from dagua.graph import DaguaGraph  # noqa: E402
from dagua.layout.ops.pipelines.maxent_stress import layout_maxent_stress_pipeline  # noqa: E402
from scripts.algo_fidelity_cross import fidelity_procrustes  # noqa: E402

OUTPUT_DIR = Path("eval_output/algo_fidelity/round_41/maxent_stress")
SMOKE_SEEDS: tuple[int, ...] = (42, 43, 44)


def _edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    """Build a ``[2, E]`` edge tensor from edge tuples.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Directed edge tuples in graph-file order.

    Returns
    -------
    torch.Tensor
        Edge index tensor with shape ``[2, E]`` and dtype ``torch.long``.
    """
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _path_graph() -> DaguaGraph:
    """Build the path smoke graph.

    Returns
    -------
    DaguaGraph
        Twelve-node path graph.
    """
    return DaguaGraph.from_edge_index(_edge_index([(i, i + 1) for i in range(11)]), num_nodes=12)


def _star_graph() -> DaguaGraph:
    """Build the star smoke graph.

    Returns
    -------
    DaguaGraph
        Thirteen-node hub-and-spoke graph.
    """
    return DaguaGraph.from_edge_index(_edge_index([(0, i) for i in range(1, 13)]), num_nodes=13)


def _clustered_graph() -> DaguaGraph:
    """Build the clustered smoke graph.

    Returns
    -------
    DaguaGraph
        Three dense-ish five-node communities with sparse bridges.
    """
    edges: list[tuple[int, int]] = []
    for base in (0, 5, 10):
        edges.extend((base + i, base + ((i + 1) % 5)) for i in range(5))
        edges.extend((base, base + i) for i in range(1, 5))
    edges.extend([(2, 7), (8, 12)])
    return DaguaGraph.from_edge_index(_edge_index(edges), num_nodes=15)


def _grid_graph() -> DaguaGraph:
    """Build the grid smoke graph.

    Returns
    -------
    DaguaGraph
        Four-by-four grid graph.
    """
    edges: list[tuple[int, int]] = []
    width = 4
    height = 4
    for y_index in range(height):
        for x_index in range(width):
            node = y_index * width + x_index
            if x_index + 1 < width:
                edges.append((node, node + 1))
            if y_index + 1 < height:
                edges.append((node, node + width))
    return DaguaGraph.from_edge_index(_edge_index(edges), num_nodes=width * height)


def _smoke_graphs() -> list[tuple[str, Callable[[], DaguaGraph]]]:
    """Return the fixed Round 41 smoke graph builders.

    Returns
    -------
    list[tuple[str, Callable[[], DaguaGraph]]]
        Topology names paired with graph factories.
    """
    return [
        ("path", _path_graph),
        ("star", _star_graph),
        ("clustered", _clustered_graph),
        ("grid", _grid_graph),
    ]


def run_smoke() -> list[dict[str, str]]:
    """Run maxent-stress against the OGDF stress adapter.

    Returns
    -------
    list[dict[str, str]]
        CSV-ready rows containing topology, seed, and Procrustes RMSD.
    """
    reference = OGDFStress()
    rows: list[dict[str, str]] = []
    for topology, make_graph in _smoke_graphs():
        graph = make_graph()
        for seed in SMOKE_SEEDS:
            reference_result = reference.layout_with_variant(
                graph,
                seed=seed,
                variant_params={"iterations": 200},
            )
            if reference_result.error is not None or reference_result.pos is None:
                raise RuntimeError(
                    f"OGDF stress failed for {topology}/{seed}: {reference_result.error}"
                )
            candidate = layout_maxent_stress_pipeline(
                graph.edge_index,
                graph.num_nodes,
                node_sizes=graph.node_sizes,
                edge_weights=graph.edge_weights,
                seed=seed,
                steps=200,
            )
            rmsd, _ = fidelity_procrustes(
                candidate.detach().to(device="cpu", dtype=torch.float32),
                reference_result.pos.detach().to(device="cpu", dtype=torch.float32),
            )
            rows.append({"topology": topology, "seed": str(seed), "rmsd": f"{rmsd:.12g}"})
    return rows


def main() -> None:
    """Write the smoke CSV and print a compact topology summary."""
    rows = run_smoke()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "smoke_rmsd.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["topology", "seed", "rmsd"])
        writer.writeheader()
        writer.writerows(rows)

    for topology, _ in _smoke_graphs():
        values = [float(row["rmsd"]) for row in rows if row["topology"] == topology]
        print(f"{topology}: mean={sum(values) / len(values):.12g} max={max(values):.12g}")
    all_values = [float(row["rmsd"]) for row in rows]
    print(f"overall: mean={sum(all_values) / len(all_values):.12g} max={max(all_values):.12g}")


if __name__ == "__main__":
    main()
