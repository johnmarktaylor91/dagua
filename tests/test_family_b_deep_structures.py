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


def _ring_edge_index(num_nodes: int) -> torch.Tensor:
    """Build a directed ring graph.

    Parameters
    ----------
    num_nodes : int
        Number of ring nodes.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, num_nodes]``.
    """
    edges = [(index, (index + 1) % num_nodes) for index in range(num_nodes)]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _triangle_chain_edge_index(count: int) -> torch.Tensor:
    """Build a chain of triangles sharing cut vertices.

    Parameters
    ----------
    count : int
        Number of chained triangles; the graph has ``2 * count + 1`` nodes.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, 3 * count]``.
    """
    edges = []
    for i in range(count):
        a, b, c = 2 * i, 2 * i + 1, 2 * i + 2
        edges += [(a, b), (b, c), (a, c)]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def test_circo_survives_1500_node_ring() -> None:
    """Order a 1500-node simple-cycle block without recursion errors.

    Returns
    -------
    None
        The layout must complete with finite positions.
    """
    num_nodes = 1500
    positions = get_pipeline_function("circo")(
        edge_index=_ring_edge_index(num_nodes),
        num_nodes=num_nodes,
    )

    assert tuple(positions.shape) == (num_nodes, 2)
    assert bool(torch.isfinite(positions).all())


def test_circo_survives_999_chained_triangles() -> None:
    """Walk a 999-deep block-cut tree without recursion errors.

    Returns
    -------
    None
        The layout must complete with finite positions for 1999 nodes.
    """
    count = 999
    positions = get_pipeline_function("circo")(
        edge_index=_triangle_chain_edge_index(count),
        num_nodes=2 * count + 1,
    )

    assert tuple(positions.shape) == (2 * count + 1, 2)
    assert bool(torch.isfinite(positions).all())


def test_circo_keeps_finite_output_when_float32_would_overflow() -> None:
    """Return finite float64 coordinates on a deep articulation chain.

    Chains of articulation-linked blocks legitimately grow block radii past
    float32 range (reference Graphviz circo shows the same exponential
    growth); the pipeline must not cast finite float64 internals into
    infinities.

    Returns
    -------
    None
        Every coordinate must stay finite; magnitudes above float32 range
        force a float64 result.
    """
    edges = []
    next_node = 1
    prev = 0
    for i in range(250):
        a, b = next_node, next_node + 1
        edges += [(prev, a), (a, b), (b, prev)]
        next_node += 2
        if i % 2 == 0:
            prev = a
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    positions = get_pipeline_function("circo")(edge_index=edge_index, num_nodes=next_node)

    assert bool(torch.isfinite(positions).all())
    assert positions.dtype == torch.float64
    assert float(positions.abs().max()) > 3.4e38


def test_elk_depth_first_strategy_survives_1500_node_path() -> None:
    """Break cycles with the depth_first strategy on a deep path.

    Returns
    -------
    None
        The public strategy parameter must not exhaust the recursion limit.
    """
    num_nodes = 1500
    positions = get_pipeline_function("elk")(
        edge_index=_path_edge_index(num_nodes),
        num_nodes=num_nodes,
        cycle_breaking_strategy="depth_first",
    )

    assert tuple(positions.shape) == (num_nodes, 2)
    assert bool(torch.isfinite(positions).all())


def _deep_cluster_problem(depth: int):
    """Build a six-node path problem with a deep singleton-cluster chain.

    Parameters
    ----------
    depth : int
        Number of chained cluster levels.

    Returns
    -------
    LayoutProblem
        Problem whose cluster metadata nests ``depth`` levels.
    """
    from dagua.layout.ops.state import LayoutProblem

    clusters = {"c0": [0]}
    parents: dict = {}
    for level in range(1, depth):
        clusters[f"c{level}"] = []
        parents[f"c{level - 1}"] = f"c{level}"
    parents[f"c{depth - 1}"] = None
    return LayoutProblem(
        edge_index=_path_edge_index(6),
        num_nodes=6,
        node_sizes=torch.full((6, 2), 40.0, dtype=torch.float64),
        clusters=clusters,
        cluster_parents=parents,
    )


def test_dagre_compound_walks_survive_1200_cluster_chain() -> None:
    """Run every dagre compound-tree walk on 1200 nested clusters.

    The full pipeline is minutes-slow at this depth for pre-existing
    (non-recursive) reasons, so the previously-crashing walks are pinned at
    the op level: preparation (cluster emission), nesting (tree depths +
    border dummies), postorder interval numbering, and border segments.

    Returns
    -------
    None
        Every compound walk must complete without recursion errors.
    """
    from dagua.layout.ops.dagre import (
        DagreBorderSegments,
        DagreMakeAcyclic,
        DagreNestingGraph,
        DagrePrepareGraph,
        _compound_postorder_numbers,
        _require_graph,
    )
    from dagua.layout.ops.state import ExecutionPlan, RuntimeContext, SolveState

    problem = _deep_cluster_problem(1200)
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))

    state = DagrePrepareGraph().apply(problem, SolveState(), ctx)
    state = DagreMakeAcyclic().apply(problem, state, ctx)
    state = DagreNestingGraph().apply(problem, state, ctx)
    intervals = _compound_postorder_numbers(_require_graph(state))
    state = DagreBorderSegments().apply(problem, state, ctx)

    assert len(intervals) > 1200


def test_dagre_compound_ordering_survives_800_node_path() -> None:
    """Order a deep compound graph without recursion errors.

    Returns
    -------
    None
        The compound successor-first ordering must survive an 800-node path
        carrying cluster metadata.
    """
    num_nodes = 800
    positions = get_pipeline_function("dagre")(
        edge_index=_path_edge_index(num_nodes),
        num_nodes=num_nodes,
        clusters={"c0": [0, 1]},
        cluster_parents={"c0": None},
    )

    assert tuple(positions.shape) == (num_nodes, 2)
    assert bool(torch.isfinite(positions).all())


def test_fdp_fidelity_survives_deep_root_level_component() -> None:
    """Walk a 1100-node root-level component in the fdp derived graph.

    Returns
    -------
    None
        The generalized connected-component DFS must not exhaust the
        recursion limit when a recursion level's direct children form a
        deep component.
    """
    from dagua.layout.ops.pipelines.fmmm import graphviz_fdp_fidelity

    num_nodes = 1102
    edges = [(i, i + 1) for i in range(1099)] + [(1100, 1101)]
    positions = graphviz_fdp_fidelity(
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        num_nodes=num_nodes,
        clusters={"k": [1100, 1101]},
        cluster_parents={"k": None},
        steps=1,
    )

    assert tuple(positions.shape) == (num_nodes, 2)
    assert bool(torch.isfinite(positions).all())
