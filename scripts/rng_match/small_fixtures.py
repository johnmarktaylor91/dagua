"""Small deterministic graph fixtures for RNG-matched fidelity checks."""

from __future__ import annotations

from collections.abc import Callable

from dagua.graph import DaguaGraph

FixtureBuilder = Callable[[], DaguaGraph]


def _graph_from_edges(num_nodes: int, edges: list[tuple[int, int]]) -> DaguaGraph:
    """Build a ``DaguaGraph`` from integer node IDs and directed edges.

    Parameters
    ----------
    num_nodes : int
        Number of nodes to create.
    edges : list[tuple[int, int]]
        Directed edge list using integer node IDs in ``[0, num_nodes)``.

    Returns
    -------
    DaguaGraph
        Graph with stable node insertion order and the requested edges.
    """
    graph = DaguaGraph()
    for node_id in range(num_nodes):
        graph.add_node(node_id)
    for source, target in edges:
        graph.add_edge(source, target)
    return graph


def path8() -> DaguaGraph:
    """Return an eight-node path graph.

    Returns
    -------
    DaguaGraph
        Path fixture with 8 nodes and 7 edges.
    """
    return _graph_from_edges(8, [(idx, idx + 1) for idx in range(7)])


def cycle6() -> DaguaGraph:
    """Return a six-node cycle graph.

    Returns
    -------
    DaguaGraph
        Cycle fixture with 6 nodes and 6 edges.
    """
    return _graph_from_edges(6, [(idx, (idx + 1) % 6) for idx in range(6)])


def star8() -> DaguaGraph:
    """Return an eight-node star graph.

    Returns
    -------
    DaguaGraph
        Star fixture with node 0 connected to seven leaves.
    """
    return _graph_from_edges(8, [(0, node_id) for node_id in range(1, 8)])


def _grid(width: int, height: int) -> DaguaGraph:
    """Return a rectangular grid graph.

    Parameters
    ----------
    width : int
        Number of columns.
    height : int
        Number of rows.

    Returns
    -------
    DaguaGraph
        Directed right/down grid fixture.
    """
    edges: list[tuple[int, int]] = []
    for row in range(height):
        for col in range(width):
            node_id = row * width + col
            if col + 1 < width:
                edges.append((node_id, node_id + 1))
            if row + 1 < height:
                edges.append((node_id, node_id + width))
    return _graph_from_edges(width * height, edges)


def grid3x3() -> DaguaGraph:
    """Return a 3-by-3 grid graph.

    Returns
    -------
    DaguaGraph
        Grid fixture with 9 nodes.
    """
    return _grid(3, 3)


def grid4x4() -> DaguaGraph:
    """Return a 4-by-4 grid graph.

    Returns
    -------
    DaguaGraph
        Grid fixture with 16 nodes.
    """
    return _grid(4, 4)


def complete5() -> DaguaGraph:
    """Return a five-node complete graph.

    Returns
    -------
    DaguaGraph
        Complete fixture with one directed edge for each undirected pair.
    """
    edges = [(source, target) for source in range(5) for target in range(source + 1, 5)]
    return _graph_from_edges(5, edges)


def complete_bipartite_3x3() -> DaguaGraph:
    """Return a complete bipartite 3-by-3 graph.

    Returns
    -------
    DaguaGraph
        Bipartite fixture with 6 nodes and 9 cross-partition edges.
    """
    return _graph_from_edges(6, [(source, target) for source in range(3) for target in range(3, 6)])


def balanced_tree_2x3() -> DaguaGraph:
    """Return a balanced binary tree of depth three.

    Returns
    -------
    DaguaGraph
        Tree fixture with 15 nodes.
    """
    edges = []
    for parent in range(7):
        edges.append((parent, 2 * parent + 1))
        edges.append((parent, 2 * parent + 2))
    return _graph_from_edges(15, edges)


def two_triangles_bridge() -> DaguaGraph:
    """Return two triangles joined by a bridge.

    Returns
    -------
    DaguaGraph
        Six-node fixture with two dense components and one bridge.
    """
    return _graph_from_edges(
        6,
        [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)],
    )


def small_dag_10() -> DaguaGraph:
    """Return a ten-node layered DAG.

    Returns
    -------
    DaguaGraph
        DAG fixture with fan-out, skip, and fan-in edges.
    """
    return _graph_from_edges(
        10,
        [
            (0, 1),
            (0, 2),
            (1, 3),
            (1, 4),
            (2, 4),
            (2, 5),
            (3, 6),
            (4, 6),
            (4, 7),
            (5, 7),
            (6, 8),
            (7, 8),
            (8, 9),
        ],
    )


def small_random_12() -> DaguaGraph:
    """Return a fixed twelve-node pseudo-random graph.

    Returns
    -------
    DaguaGraph
        Deterministic sparse graph with varied degrees.
    """
    return _graph_from_edges(
        12,
        [
            (0, 1),
            (0, 4),
            (1, 2),
            (1, 5),
            (2, 3),
            (2, 8),
            (3, 7),
            (4, 5),
            (4, 9),
            (5, 6),
            (6, 7),
            (6, 10),
            (7, 11),
            (8, 9),
            (9, 10),
            (10, 11),
            (1, 9),
            (3, 10),
        ],
    )


def petersen_10() -> DaguaGraph:
    """Return the Petersen graph.

    Returns
    -------
    DaguaGraph
        Ten-node 3-regular Petersen fixture.
    """
    outer = [(idx, (idx + 1) % 5) for idx in range(5)]
    spokes = [(idx, idx + 5) for idx in range(5)]
    inner = [(5, 7), (7, 9), (9, 6), (6, 8), (8, 5)]
    return _graph_from_edges(10, outer + spokes + inner)


def wheel7() -> DaguaGraph:
    """Return a seven-node wheel graph.

    Returns
    -------
    DaguaGraph
        Wheel fixture with one hub and a six-node rim.
    """
    rim = [(node_id, 1 + (node_id % 6)) for node_id in range(1, 7)]
    spokes = [(0, node_id) for node_id in range(1, 7)]
    return _graph_from_edges(7, rim + spokes)


def ladder5() -> DaguaGraph:
    """Return a five-rung ladder graph.

    Returns
    -------
    DaguaGraph
        Ladder fixture with 10 nodes.
    """
    top = [(idx, idx + 1) for idx in range(4)]
    bottom = [(idx + 5, idx + 6) for idx in range(4)]
    rungs = [(idx, idx + 5) for idx in range(5)]
    return _graph_from_edges(10, top + bottom + rungs)


FIXTURES: dict[str, FixtureBuilder] = {
    "path8": path8,
    "cycle6": cycle6,
    "star8": star8,
    "grid3x3": grid3x3,
    "grid4x4": grid4x4,
    "complete5": complete5,
    "complete_bipartite_3x3": complete_bipartite_3x3,
    "balanced_tree_2x3": balanced_tree_2x3,
    "two_triangles_bridge": two_triangles_bridge,
    "small_dag_10": small_dag_10,
    "small_random_12": small_random_12,
    "petersen_10": petersen_10,
    "wheel7": wheel7,
    "ladder5": ladder5,
}


def small_fixtures() -> dict[str, DaguaGraph]:
    """Build all small RNG-match fixtures.

    Returns
    -------
    dict[str, DaguaGraph]
        Fresh graph instances keyed by fixture name.
    """
    return {name: builder() for name, builder in FIXTURES.items()}
