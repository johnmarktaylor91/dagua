"""Round 41 LGL adapter-fidelity smoke harness."""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import igraph
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dagua.layout.ops.pipelines.lgl import layout_lgl_pipeline  # noqa: E402
from scripts.algo_fidelity_cross import fidelity_procrustes  # noqa: E402

EdgeList = List[Tuple[int, int]]
GraphSpec = Tuple[int, EdgeList]


def _edge_index(edges: EdgeList) -> torch.Tensor:
    """Build a PyTorch edge tensor from ordered edges.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Ordered directed edge list.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _graphs() -> Dict[str, GraphSpec]:
    """Return the four topology smoke graphs.

    Returns
    -------
    dict[str, tuple[int, list[tuple[int, int]]]]
        Path, star, clustered, and grid graph specifications.
    """
    clustered_edges: EdgeList = []
    for base in (0, 5):
        for source in range(base, base + 5):
            for target in range(source + 1, base + 5):
                clustered_edges.append((source, target))
    clustered_edges.extend([(4, 5), (3, 6), (2, 7)])

    grid_edges: EdgeList = []
    for row in range(4):
        for col in range(4):
            node = row * 4 + col
            if col < 3:
                grid_edges.append((node, node + 1))
            if row < 3:
                grid_edges.append((node, node + 4))

    return {
        "path": (12, [(node, node + 1) for node in range(11)]),
        "star": (13, [(0, node) for node in range(1, 13)]),
        "clustered": (10, clustered_edges),
        "grid": (16, grid_edges),
    }


def _igraph_lgl_positions(num_nodes: int, edges: EdgeList, seed: int) -> torch.Tensor:
    """Run the python-igraph LGL reference adapter.

    Parameters
    ----------
    num_nodes : int
        Number of graph vertices.
    edges : list[tuple[int, int]]
        Ordered directed edge list.
    seed : int
        Python RNG seed routed through igraph.

    Returns
    -------
    torch.Tensor
        Reference position tensor with shape ``[N, 2]`` and the benchmark
        adapter's 50x coordinate scale.
    """
    graph = igraph.Graph(directed=True)
    graph.add_vertices(num_nodes)
    graph.add_edges(edges)
    igraph.set_random_number_generator(random.Random(seed))
    try:
        layout = graph.layout_lgl(
            maxiter=150,
            maxdelta=num_nodes,
            area=num_nodes * num_nodes,
            coolexp=1.5,
            repulserad=num_nodes * num_nodes * num_nodes,
            cellsize=math.sqrt(num_nodes),
            root=None,
        )
    finally:
        igraph.set_random_number_generator(None)
    return torch.tensor(
        [[layout[node][0] * 50.0, layout[node][1] * 50.0] for node in range(num_nodes)],
        dtype=torch.float32,
    )


def main() -> None:
    """Print per-topology, per-seed LGL Procrustes RMSD smoke results.

    Returns
    -------
    None
        The function prints a Markdown table and an overall mean.
    """
    all_values: list[float] = []
    print("| topology | seed 42 | seed 43 | seed 44 | mean |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for name, (num_nodes, edges) in _graphs().items():
        values: list[float] = []
        for seed in (42, 43, 44):
            dagua_positions = layout_lgl_pipeline(
                _edge_index(edges),
                num_nodes,
                seed=seed,
                fidelity_mode=True,
            )
            reference_positions = _igraph_lgl_positions(num_nodes, edges, seed)
            rmsd, _ = fidelity_procrustes(dagua_positions, reference_positions)
            values.append(rmsd)
            all_values.append(rmsd)
        mean_value = sum(values) / len(values)
        print(
            f"| {name} | {values[0]:.8f} | {values[1]:.8f} | {values[2]:.8f} | {mean_value:.8f} |"
        )
    print(f"\noverall_mean={sum(all_values) / len(all_values):.8f}")


if __name__ == "__main__":
    main()
