"""Deep-structure regression pins for Family B reference pipelines.

These tests pin the GLaDOS-prep round-2 crash fixes: reference ports must
survive legal corpus-range topologies whose DFS/tree depth exceeds Python's
default recursion limit (R2-B2-F01), and the ELK layered ranking must survive
parallel edges in its acyclic active set (R2-B2-F02) -- dagre.js runs its
network simplex on ``simplify(g)``, so the reference machinery never sees
parallel edges.

The 40x40-grid repros for ``dagre``/``elk`` are intentionally NOT pinned at
the pipeline level: their network-simplex exchange loop is minutes-slow on a
1600-node mesh (a pre-existing performance property, profiled in-simplex).
The grid-exposed recursion sites (tight-tree growth and tree preorder walks)
recurse per-node exactly like the 1500-path rows pinned here, which run in
seconds.
"""

from __future__ import annotations

import pytest
import torch

from dagua.layout.ops.elk import _component_network_simplex_layers
from dagua.layout.ops.pipelines import get_pipeline_function


def _path_edge_index(num_nodes: int) -> torch.Tensor:
    """Build a directed path graph.

    Parameters
    ----------
    num_nodes : int
        Number of chain nodes.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, num_nodes - 1]``.
    """
    edges = [(index, index + 1) for index in range(num_nodes - 1)]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _grid_edge_index(side: int) -> torch.Tensor:
    """Build a directed side x side grid graph.

    Parameters
    ----------
    side : int
        Grid side length.

    Returns
    -------
    torch.Tensor
        Edge tensor with right/down mesh edges.
    """
    edges = []
    for row in range(side):
        for col in range(side):
            if col + 1 < side:
                edges.append((row * side + col, row * side + col + 1))
            if row + 1 < side:
                edges.append((row * side + col, (row + 1) * side + col))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


_PATH_PIPELINES = (
    "twopi",
    "circo",
    "d3_tree",
    "d3_tree_radial",
    "dagre",
    "elk",
    "elk_layered_bk",
    "elk_layered_ns",
    "tidy",
)

_GRID_PIPELINES = ("twopi", "circo", "d3_tree", "d3_tree_radial", "tidy")


@pytest.mark.parametrize("pipeline", _PATH_PIPELINES)
def test_pipeline_survives_1500_node_path(pipeline: str) -> None:
    """Lay out a 1500-node directed path without exhausting recursion.

    Parameters
    ----------
    pipeline : str
        Registered pipeline name.

    Returns
    -------
    None
        The layout must complete with finite positions for every node.
    """
    num_nodes = 1500
    edge_index = _path_edge_index(num_nodes)

    positions = get_pipeline_function(pipeline)(edge_index=edge_index, num_nodes=num_nodes)

    assert tuple(positions.shape) == (num_nodes, 2)
    assert bool(torch.isfinite(positions).all())


@pytest.mark.parametrize("pipeline", _GRID_PIPELINES)
def test_pipeline_survives_40x40_grid(pipeline: str) -> None:
    """Lay out a 40x40 grid (deep serpentine DFS) without recursion errors.

    Parameters
    ----------
    pipeline : str
        Registered pipeline name.

    Returns
    -------
    None
        The layout must complete with finite positions for every node.
    """
    side = 40
    edge_index = _grid_edge_index(side)

    positions = get_pipeline_function(pipeline)(edge_index=edge_index, num_nodes=side * side)

    assert tuple(positions.shape) == (side * side, 2)
    assert bool(torch.isfinite(positions).all())


# Minimal deterministic parallel-edge repro for the network-simplex exchange
# invariant (R2-B2-F02): feeding these 19 records (with duplicate pairs
# (4, 6) x3, (2, 8) x2, and (3, 11) x2) RAW into the simplex dies with
# ``min() arg is an empty sequence`` in enterEdge.
_F02_EDGES = [
    (0, 1),
    (1, 2),
    (0, 3),
    (3, 4),
    (3, 5),
    (4, 6),
    (6, 7),
    (2, 8),
    (3, 9),
    (6, 10),
    (8, 11),
    (2, 10),
    (6, 11),
    (4, 6),
    (3, 11),
    (5, 10),
    (2, 8),
    (3, 11),
    (4, 6),
]


def test_elk_network_simplex_layers_tolerate_parallel_edges() -> None:
    """Rank a component whose active edge set carries parallel duplicates.

    Returns
    -------
    None
        Layering must complete and satisfy every edge's layer constraint.
    """
    component = list(range(12))

    layers = _component_network_simplex_layers(component, _F02_EDGES)

    assert set(layers) == set(component)
    assert min(layers.values()) == 0
    for source, target in set(_F02_EDGES):
        assert layers[target] - layers[source] >= 1


def test_elk_pipeline_survives_parallel_and_reversed_edges() -> None:
    """Run the full ELK pipeline on a graph with duplicates and a 2-cycle.

    Cycle breaking reverses one arm of the 2-cycle, manufacturing exactly the
    parallel-edge shape that used to violate the exchange invariant.

    Returns
    -------
    None
        The layout must complete with finite positions.
    """
    edges = list(_F02_EDGES) + [(7, 6), (11, 8)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    positions = get_pipeline_function("elk")(edge_index=edge_index, num_nodes=12)

    assert tuple(positions.shape) == (12, 2)
    assert bool(torch.isfinite(positions).all())
