#!/usr/bin/env python3
"""Round 41 DrL smoke fidelity harness."""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

import igraph
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dagua.layout.ops.pipelines.drl import layout_drl_pipeline  # noqa: E402
from scripts.algo_fidelity_cross import fidelity_procrustes  # noqa: E402


@dataclass(frozen=True)
class Topology:
    """Small graph topology used by the DrL smoke harness.

    Parameters
    ----------
    name : str
        Stable topology label.
    num_nodes : int
        Number of graph nodes.
    edges : list[tuple[int, int]]
        Directed edge list in graph-file order.
    """

    name: str
    num_nodes: int
    edges: list[tuple[int, int]]


def _path_edges(num_nodes: int) -> list[tuple[int, int]]:
    """Build a directed path edge list.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    list[tuple[int, int]]
        Directed path edges.
    """
    return [(node, node + 1) for node in range(num_nodes - 1)]


def _star_edges(num_nodes: int) -> list[tuple[int, int]]:
    """Build a directed star edge list.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes including the hub.

    Returns
    -------
    list[tuple[int, int]]
        Directed hub-to-leaf edges.
    """
    return [(0, node) for node in range(1, num_nodes)]


def _clustered_edges() -> list[tuple[int, int]]:
    """Build two dense clusters with a sparse bridge.

    Returns
    -------
    list[tuple[int, int]]
        Directed clustered graph edges.
    """
    edges: list[tuple[int, int]] = []
    for base in (0, 5):
        for source in range(base, base + 5):
            for target in range(source + 1, base + 5):
                edges.append((source, target))
    edges.extend([(2, 7), (3, 8)])
    return edges


def _grid_edges(width: int, height: int) -> list[tuple[int, int]]:
    """Build a directed rectangular grid edge list.

    Parameters
    ----------
    width : int
        Grid width in nodes.
    height : int
        Grid height in nodes.

    Returns
    -------
    list[tuple[int, int]]
        Rightward and downward grid edges.
    """
    edges: list[tuple[int, int]] = []
    for row in range(height):
        for col in range(width):
            node = (row * width) + col
            if col + 1 < width:
                edges.append((node, node + 1))
            if row + 1 < height:
                edges.append((node, node + width))
    return edges


def _edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    """Convert edge tuples to a PyTorch edge index.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Directed edge list.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(list(zip(*edges)), dtype=torch.long)


def _reference_layout(topology: Topology, seed: int) -> torch.Tensor:
    """Run the igraph DrL adapter contract for one topology and seed.

    Parameters
    ----------
    topology : Topology
        Graph topology to lay out.
    seed : int
        Seed used for the adapter seed matrix and Python RNG hook.

    Returns
    -------
    torch.Tensor
        Reference positions with shape ``[N, 2]``.
    """
    graph = igraph.Graph(directed=True)
    graph.add_vertices(topology.num_nodes)
    graph.add_edges(topology.edges)

    igraph.set_random_number_generator(random.Random(seed))
    try:
        seed_matrix = np.random.RandomState(seed).uniform(-1.0, 1.0, size=(topology.num_nodes, 2))
        layout = graph.layout("drl", seed=seed_matrix.tolist(), options="default")
    finally:
        igraph.set_random_number_generator(None)

    positions = torch.empty((topology.num_nodes, 2), dtype=torch.float32)
    for node in range(topology.num_nodes):
        positions[node, 0] = float(layout[node][0]) * 50.0
        positions[node, 1] = float(layout[node][1]) * 50.0
    return positions


def _dagua_layout(topology: Topology, seed: int) -> torch.Tensor:
    """Run Dagua DrL fidelity mode for one topology and seed.

    Parameters
    ----------
    topology : Topology
        Graph topology to lay out.
    seed : int
        Seed forwarded to the DRL pipeline.

    Returns
    -------
    torch.Tensor
        Dagua positions with shape ``[N, 2]``.
    """
    return layout_drl_pipeline(
        _edge_index(topology.edges),
        topology.num_nodes,
        seed=seed,
        fidelity_mode=True,
    )


def main() -> None:
    """Run the round 41 DrL smoke matrix and print RMSD rows.

    Returns
    -------
    None
        Prints topology, seed, and overall Procrustes RMSD values.
    """
    topologies = [
        Topology("path", 12, _path_edges(12)),
        Topology("star", 12, _star_edges(12)),
        Topology("clustered", 10, _clustered_edges()),
        Topology("grid", 16, _grid_edges(4, 4)),
    ]
    seeds = [42, 43, 44]
    rmsd_values: list[float] = []

    for topology in topologies:
        topology_values: list[float] = []
        for seed in seeds:
            rmsd, _ = fidelity_procrustes(
                _dagua_layout(topology, seed),
                _reference_layout(topology, seed),
            )
            topology_values.append(rmsd)
            rmsd_values.append(rmsd)
            print(f"{topology.name}\tseed={seed}\trmsd={rmsd:.9f}")
        mean = sum(topology_values) / len(topology_values)
        print(f"{topology.name}\tmean={mean:.9f}")

    overall = sum(rmsd_values) / len(rmsd_values)
    print(f"overall\tmean={overall:.9f}")


if __name__ == "__main__":
    main()
