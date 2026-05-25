"""Round 41 Pivot-MDS OGDF fidelity smoke harness."""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dagua.eval.competitors.ogdf_competitor import OGDFPivotMDS  # noqa: E402
from dagua.graph import DaguaGraph  # noqa: E402
from dagua.layout.ops.pipelines.pivot_mds import layout_pivot_mds_pipeline  # noqa: E402
from scripts.fidelity_analysis import fidelity_procrustes  # noqa: E402

_OUTPUT_DIR = Path("eval_output/algo_fidelity/round_41/pivot_mds")
_N_PIVOTS = 50
_SEEDS = (0, 1, 2)


def _edge_index_from_edges(edges: list[tuple[int, int]]) -> torch.Tensor:
    """Build a PyG-style edge tensor from an ordered edge list.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Directed graph edges in reference insertion order.

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
    """Return a path smoke topology.

    Returns
    -------
    tuple[int, torch.Tensor]
        Node count and edge tensor.
    """
    num_nodes = 8
    return num_nodes, _edge_index_from_edges([(index, index + 1) for index in range(num_nodes - 1)])


def _star_graph() -> tuple[int, torch.Tensor]:
    """Return a star smoke topology.

    Returns
    -------
    tuple[int, torch.Tensor]
        Node count and edge tensor.
    """
    num_nodes = 9
    return num_nodes, _edge_index_from_edges([(0, index) for index in range(1, num_nodes)])


def _clustered_graph() -> tuple[int, torch.Tensor]:
    """Return a clustered smoke topology with two dense blocks and bridges.

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
    """Run the OGDF Pivot-MDS reference adapter.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    seed : int
        Benchmark seed forwarded through the competitor interface.

    Returns
    -------
    torch.Tensor
        Reference coordinates with shape ``[N, 2]``.

    Raises
    ------
    RuntimeError
        If the OGDF adapter fails.
    """
    graph = DaguaGraph.from_edge_index(edge_index, num_nodes)
    result = OGDFPivotMDS().layout_with_variant(
        graph,
        seed=seed,
        variant_params={"n_pivots": _N_PIVOTS},
    )
    if result.pos is None:
        raise RuntimeError(f"OGDF Pivot-MDS failed: {result.error}")
    return result.pos


def _rmsd(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    """Compute scale-normalized Procrustes RMSD for one layout pair.

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
    rmsd, _, _, _ = fidelity_procrustes(candidate, reference)
    return float(rmsd)


def _dagua_positions(
    edge_index: torch.Tensor,
    num_nodes: int,
    use_ogdf_solver: bool,
) -> torch.Tensor:
    """Run Dagua's Pivot-MDS pipeline in old or OGDF eigensolver mode.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    use_ogdf_solver : bool
        Whether to use the Round 41 OGDF eigensolver trigger.

    Returns
    -------
    torch.Tensor
        Dagua coordinates with shape ``[N, 2]``.
    """
    if use_ogdf_solver:
        return layout_pivot_mds_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            n_pivots=_N_PIVOTS,
            first_pivot="first_node",
            compute_dtype=torch.float64,
            distance_scale=100.0,
            ogdf_path_special_case=True,
        )
    return layout_pivot_mds_pipeline(
        edge_index=edge_index,
        num_nodes=num_nodes,
        n_pivots=_N_PIVOTS,
        first_pivot_index=0,
        compute_dtype=torch.float64,
        distance_scale=100.0,
        ogdf_path_special_case=True,
    )


def main() -> None:
    """Run the Pivot-MDS smoke matrix and write ``smoke_rmsd.csv``.

    Returns
    -------
    None
        Results are written to the Round 41 Pivot-MDS output directory.
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
            before = _dagua_positions(edge_index, num_nodes, use_ogdf_solver=False)
            after = _dagua_positions(edge_index, num_nodes, use_ogdf_solver=True)
            rows.append(
                {
                    "topology": topology,
                    "seed": seed,
                    "before_rmsd": _rmsd(before, reference),
                    "after_rmsd": _rmsd(after, reference),
                }
            )

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (_OUTPUT_DIR / "smoke_rmsd.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["topology", "seed", "before_rmsd", "after_rmsd"],
        )
        writer.writeheader()
        writer.writerows(rows)

    before_mean = statistics.fmean(float(row["before_rmsd"]) for row in rows)
    after_mean = statistics.fmean(float(row["after_rmsd"]) for row in rows)
    print(f"mean_before={before_mean:.9f}")
    print(f"mean_after={after_mean:.9f}")
    for row in rows:
        print(
            f"{row['topology']} seed={row['seed']} "
            f"before={float(row['before_rmsd']):.9f} after={float(row['after_rmsd']):.9f}"
        )


if __name__ == "__main__":
    main()
