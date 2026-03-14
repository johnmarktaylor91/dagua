"""Compare classic layout implementations against authoritative references.

FR  -> NetworkX spring_layout (Fruchterman-Reingold)
KK  -> NetworkX kamada_kawai_layout (stress minimization)
Sugiyama -> Graphviz dot (layered DAG layout)

These tests verify our educational reimplementations are faithful to the
original algorithms, not just "some force-directed thing."
"""

from __future__ import annotations

import importlib.util
from typing import Any

import pytest
import torch

from dagua.layout.classic import layout_fr, layout_kk, layout_sugiyama

_NETWORKX_AVAILABLE = importlib.util.find_spec("networkx") is not None
_PYDOT_AVAILABLE = importlib.util.find_spec("pydot") is not None


def _make_small_random_graph(seed: int = 42) -> tuple[torch.Tensor, int, Any]:
    """Build a small undirected random graph in torch and NetworkX formats.

    Parameters
    ----------
    seed : int, default=42
        Seed used for deterministic edge sampling.

    Returns
    -------
    tuple[torch.Tensor, int, Any]
        ``(edge_index, num_nodes, graph)`` for a 20-node undirected graph.
    """
    nx = pytest.importorskip("networkx")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    num_nodes = 20
    src = torch.randint(0, num_nodes, (60,), generator=generator, dtype=torch.long)
    tgt = torch.randint(0, num_nodes, (60,), generator=generator, dtype=torch.long)
    mask = src != tgt
    src = src[mask]
    tgt = tgt[mask]
    edge_index = torch.stack([src, tgt])

    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    for edge_idx in range(edge_index.shape[1]):
        graph.add_edge(
            int(edge_index[0, edge_idx].item()),
            int(edge_index[1, edge_idx].item()),
        )

    return edge_index, num_nodes, graph


def _make_diamond_dag() -> tuple[torch.Tensor, int, Any]:
    """Build the classic diamond DAG in torch and NetworkX formats.

    Returns
    -------
    tuple[torch.Tensor, int, Any]
        ``(edge_index, num_nodes, graph)`` for the 4-node diamond DAG.
    """
    nx = pytest.importorskip("networkx")

    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]], dtype=torch.long)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(4))
    graph.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
    return edge_index, 4, graph


def _make_layered_dag(seed: int = 42) -> tuple[torch.Tensor, int, Any]:
    """Build a three-layer DAG in torch and NetworkX formats.

    Parameters
    ----------
    seed : int, default=42
        Seed used for deterministic inter-layer edge sampling.

    Returns
    -------
    tuple[torch.Tensor, int, Any]
        ``(edge_index, num_nodes, graph)`` for a 12-node layered DAG.
    """
    nx = pytest.importorskip("networkx")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    edges_src: list[int] = []
    edges_tgt: list[int] = []

    for src_layer, tgt_layer in [(0, 4), (4, 8)]:
        for source in range(src_layer, src_layer + 4):
            for target in range(tgt_layer, tgt_layer + 4):
                if float(torch.rand(1, generator=generator).item()) < 0.5:
                    edges_src.append(source)
                    edges_tgt.append(target)

    edges_src.extend([0, 4])
    edges_tgt.extend([4, 8])
    edge_index = torch.tensor([edges_src, edges_tgt], dtype=torch.long)

    graph = nx.DiGraph()
    graph.add_nodes_from(range(12))
    for source, target in zip(edges_src, edges_tgt):
        graph.add_edge(source, target)

    return edge_index, 12, graph


def _make_petersen_graph() -> tuple[torch.Tensor, int, Any]:
    """Build the Petersen graph in torch and NetworkX formats.

    Returns
    -------
    tuple[torch.Tensor, int, Any]
        ``(edge_index, num_nodes, graph)`` for the 10-node Petersen graph.
    """
    nx = pytest.importorskip("networkx")

    graph = nx.petersen_graph()
    edges = list(graph.edges())
    edge_index = torch.tensor(
        [[source for source, _ in edges], [target for _, target in edges]],
        dtype=torch.long,
    )
    return edge_index, 10, graph


def _make_fr_initial_positions(num_nodes: int, seed: int, area: float) -> torch.Tensor:
    """Mirror the FR initializer so both implementations share the same start.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    seed : int
        Seed used for deterministic initialization.
    area : float
        Layout area used by the FR implementation.

    Returns
    -------
    torch.Tensor
        Initial positions with shape ``[N, 2]``.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.rand((num_nodes, 2), generator=generator, dtype=torch.float32) * (area**0.5)


def _normalize_positions(pos: torch.Tensor) -> torch.Tensor:
    """Normalize positions into the unit square.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Normalized positions in ``[0, 1] x [0, 1]``.
    """
    mins = pos.min(dim=0).values
    maxs = pos.max(dim=0).values
    span = (maxs - mins).clamp(min=1e-6)
    return (pos - mins) / span


def _pairwise_distances(pos: torch.Tensor) -> torch.Tensor:
    """Compute the full pairwise Euclidean distance matrix.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Pairwise distance matrix with shape ``[N, N]``.
    """
    return torch.cdist(pos.unsqueeze(0), pos.unsqueeze(0)).squeeze(0)


def _upper_triangle_values(matrix: torch.Tensor) -> torch.Tensor:
    """Extract the unique off-diagonal entries from a square matrix.

    Parameters
    ----------
    matrix : torch.Tensor
        Square matrix with shape ``[N, N]``.

    Returns
    -------
    torch.Tensor
        Upper-triangular values for ``i < j``.
    """
    upper = torch.triu_indices(matrix.shape[0], matrix.shape[1], offset=1)
    return matrix[upper[0], upper[1]]


def _pearson_correlation(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    """Compute the Pearson correlation between two vectors.

    Parameters
    ----------
    lhs : torch.Tensor
        First vector.
    rhs : torch.Tensor
        Second vector.

    Returns
    -------
    float
        Correlation coefficient in ``[-1, 1]``.
    """
    lhs_centered = lhs - lhs.mean()
    rhs_centered = rhs - rhs.mean()
    denominator = torch.sqrt(lhs_centered.square().sum() * rhs_centered.square().sum())
    if float(denominator.item()) == 0.0:
        return 1.0
    return float((lhs_centered * rhs_centered).sum().item() / denominator.item())


def _nearest_neighbor_overlaps(pos_a: torch.Tensor, pos_b: torch.Tensor, k: int) -> list[float]:
    """Measure k-nearest-neighbor overlap for each node across two layouts.

    Parameters
    ----------
    pos_a : torch.Tensor
        First layout positions with shape ``[N, 2]``.
    pos_b : torch.Tensor
        Second layout positions with shape ``[N, 2]``.
    k : int
        Number of nearest neighbors to compare.

    Returns
    -------
    list[float]
        Per-node neighbor-overlap fractions.
    """
    distances_a = _pairwise_distances(pos_a)
    distances_b = _pairwise_distances(pos_b)
    distances_a.fill_diagonal_(float("inf"))
    distances_b.fill_diagonal_(float("inf"))

    overlaps: list[float] = []
    for node in range(pos_a.shape[0]):
        neighbors_a = set(torch.topk(distances_a[node], k=k, largest=False).indices.tolist())
        neighbors_b = set(torch.topk(distances_b[node], k=k, largest=False).indices.tolist())
        overlaps.append(len(neighbors_a & neighbors_b) / float(k))
    return overlaps


def _positions_from_networkx_dict(pos_dict: Any, num_nodes: int) -> torch.Tensor:
    """Convert a NetworkX position dictionary into a dense tensor.

    Parameters
    ----------
    pos_dict : Any
        Mapping from node id to 2D coordinates.
    num_nodes : int
        Number of nodes expected in the output.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    rows = [[float(value) for value in pos_dict[node]] for node in range(num_nodes)]
    return torch.tensor(rows, dtype=torch.float32)


def _normalize_distance_vector(pos: torch.Tensor) -> torch.Tensor:
    """Return max-normalized unique pairwise distances for a layout.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Unique pairwise distances scaled by their maximum value.
    """
    values = _upper_triangle_values(_pairwise_distances(pos))
    return values / values.max().clamp(min=1e-6)


def _sorted_normalized_distance_vector(pos: torch.Tensor) -> torch.Tensor:
    """Return a permutation-robust pairwise-distance signature.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Sorted, max-normalized pairwise distances.
    """
    return torch.sort(_normalize_distance_vector(pos)).values


def _all_pairs_shortest_paths(graph: Any, num_nodes: int) -> dict[int, dict[int, int]]:
    """Compute graph-theoretic all-pairs shortest-path lengths.

    Parameters
    ----------
    graph : Any
        NetworkX graph.
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    dict[int, dict[int, int]]
        Nested dictionary of shortest-path lengths.
    """
    nx = pytest.importorskip("networkx")

    shortest_paths = dict(nx.all_pairs_shortest_path_length(graph))
    return {node: shortest_paths[node] for node in range(num_nodes)}


def _compute_stress(
    pos: torch.Tensor,
    graph_distances: dict[int, dict[int, int]],
    num_nodes: int,
) -> float:
    """Compute Kamada-Kawai stress after fitting the best global scale.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    graph_distances : dict[int, dict[int, int]]
        Graph-theoretic shortest-path distances.
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    float
        Weighted stress value.
    """
    pairwise = _pairwise_distances(pos)
    numerator = 0.0
    denominator = 0.0

    for source in range(num_nodes):
        for target in range(source + 1, num_nodes):
            graph_distance = float(graph_distances[source][target])
            weight = 1.0 / (graph_distance**2)
            numerator += weight * float(pairwise[source, target].item()) * graph_distance
            denominator += weight * graph_distance * graph_distance

    scale = numerator / denominator if denominator > 0.0 else 1.0
    stress = 0.0
    for source in range(num_nodes):
        for target in range(source + 1, num_nodes):
            graph_distance = float(graph_distances[source][target])
            weight = 1.0 / (graph_distance**2)
            difference = float(pairwise[source, target].item()) - (scale * graph_distance)
            stress += weight * difference * difference

    return stress


def _graphviz_dot_positions(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Run Graphviz dot via pydot and parse node coordinates.

    Parameters
    ----------
    edge_index : torch.Tensor
        Directed edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    torch.Tensor
        Parsed Graphviz positions with shape ``[N, 2]``.
    """
    pydot = pytest.importorskip("pydot")

    dot_graph = pydot.Dot(graph_type="digraph", rankdir="TB")
    for node in range(num_nodes):
        dot_graph.add_node(pydot.Node(str(node)))
    for edge_idx in range(edge_index.shape[1]):
        dot_graph.add_edge(
            pydot.Edge(
                str(int(edge_index[0, edge_idx].item())),
                str(int(edge_index[1, edge_idx].item())),
            )
        )

    parsed = pydot.graph_from_dot_data(dot_graph.create_dot().decode())[0]
    pos = torch.zeros((num_nodes, 2), dtype=torch.float32)
    seen_nodes: set[int] = set()
    for node in parsed.get_nodes():
        name = node.get_name().strip('"')
        if name in {"node", "edge", "graph", "\\n", ""}:
            continue
        pos_str = node.get("pos")
        if pos_str is None:
            continue

        x_str, y_str = pos_str.strip('"').split(",")
        node_index = int(name)
        pos[node_index] = torch.tensor([float(x_str), float(y_str)], dtype=torch.float32)
        seen_nodes.add(node_index)

    if len(seen_nodes) != num_nodes:
        raise ValueError("Graphviz dot did not emit positions for every node.")

    return pos


def _infer_layer_indices(
    pos: torch.Tensor, edge_index: torch.Tensor, atol: float = 1e-4
) -> torch.Tensor:
    """Infer discrete layer indices from y coordinates.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Directed edge tensor with shape ``[2, E]``.
    atol : float, default=1e-4
        Tolerance used to group nearly identical y coordinates.

    Returns
    -------
    torch.Tensor
        Layer index for each node.
    """
    if pos.shape[0] == 0:
        return torch.empty(0, dtype=torch.long)

    y_values = pos[:, 1]
    if edge_index.numel() == 0:
        return torch.zeros(pos.shape[0], dtype=torch.long)

    edge_deltas = y_values[edge_index[1]] - y_values[edge_index[0]]
    reverse = float(edge_deltas.mean().item()) < 0.0
    ordered_y = sorted((float(value) for value in y_values.tolist()), reverse=reverse)

    centers: list[float] = []
    for value in ordered_y:
        if not centers or abs(value - centers[-1]) > atol:
            centers.append(value)

    layer_indices: list[int] = []
    for value in y_values.tolist():
        assigned_layer = next(
            layer_idx
            for layer_idx, center in enumerate(centers)
            if abs(float(value) - center) <= atol
        )
        layer_indices.append(assigned_layer)

    return torch.tensor(layer_indices, dtype=torch.long)


def _has_no_overlaps_within_layers(
    pos: torch.Tensor, layers: torch.Tensor, atol: float = 1e-6
) -> bool:
    """Check that nodes on the same layer have distinct x coordinates.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    layers : torch.Tensor
        Layer index for each node.
    atol : float, default=1e-6
        Minimum allowable x separation.

    Returns
    -------
    bool
        ``True`` when no same-layer node pair overlaps horizontally.
    """
    if pos.shape[0] <= 1:
        return True

    for layer in torch.unique(layers):
        node_indices = torch.nonzero(layers == layer, as_tuple=False).flatten()
        if node_indices.numel() <= 1:
            continue

        x_coords = torch.sort(pos[node_indices, 0]).values
        if bool(torch.any((x_coords[1:] - x_coords[:-1]) <= atol).item()):
            return False

    return True


def _count_crossings(edge_index: torch.Tensor, pos: torch.Tensor) -> int:
    """Count straight-line edge crossings using segment intersection tests.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    int
        Number of non-endpoint edge crossings.
    """
    crossings = 0
    num_edges = edge_index.shape[1]
    for first_edge in range(num_edges):
        for second_edge in range(first_edge + 1, num_edges):
            u1 = int(edge_index[0, first_edge].item())
            v1 = int(edge_index[1, first_edge].item())
            u2 = int(edge_index[0, second_edge].item())
            v2 = int(edge_index[1, second_edge].item())
            if len({u1, v1, u2, v2}) < 4:
                continue
            if _segments_intersect(pos[u1], pos[v1], pos[u2], pos[v2]):
                crossings += 1
    return crossings


def _segments_intersect(
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    p4: torch.Tensor,
) -> bool:
    """Check whether two closed line segments intersect strictly.

    Parameters
    ----------
    p1 : torch.Tensor
        First endpoint of the first segment.
    p2 : torch.Tensor
        Second endpoint of the first segment.
    p3 : torch.Tensor
        First endpoint of the second segment.
    p4 : torch.Tensor
        Second endpoint of the second segment.

    Returns
    -------
    bool
        ``True`` when the segments cross at a non-endpoint interior point.
    """

    def cross(origin: torch.Tensor, point_a: torch.Tensor, point_b: torch.Tensor) -> float:
        """Compute the 2D cross product of OA and OB.

        Parameters
        ----------
        origin : torch.Tensor
            Shared origin point.
        point_a : torch.Tensor
            First point.
        point_b : torch.Tensor
            Second point.

        Returns
        -------
        float
            Signed cross-product value.
        """
        return float(
            ((point_a[0] - origin[0]) * (point_b[1] - origin[1]))
            - ((point_a[1] - origin[1]) * (point_b[0] - origin[0]))
        )

    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    return ((d1 > 0.0 and d2 < 0.0) or (d1 < 0.0 and d2 > 0.0)) and (
        (d3 > 0.0 and d4 < 0.0) or (d3 < 0.0 and d4 > 0.0)
    )


@pytest.mark.skipif(not _NETWORKX_AVAILABLE, reason="networkx is not installed")
def test_fr_matches_networkx_spring_layout() -> None:
    """Compare FR against NetworkX using shared initialization and distance structure.

    Returns
    -------
    None
        The assertions validate pairwise-distance similarity and local neighborhoods.
    """
    nx = pytest.importorskip("networkx")

    edge_index, num_nodes, graph = _make_small_random_graph()
    area = float(num_nodes * 100)
    steps = 800
    ideal_length = (area / num_nodes) ** 0.5

    init_pos = _make_fr_initial_positions(num_nodes=num_nodes, seed=42, area=area)
    init_dict = {
        node: (float(init_pos[node, 0].item()), float(init_pos[node, 1].item()))
        for node in range(num_nodes)
    }

    nx_pos = nx.spring_layout(
        graph,
        pos=init_dict,
        iterations=steps,
        seed=42,
        k=ideal_length,
        scale=None,
    )
    nx_tensor = _positions_from_networkx_dict(nx_pos, num_nodes=num_nodes)
    our_tensor = layout_fr(
        edge_index=edge_index, num_nodes=num_nodes, steps=steps, seed=42, area=area
    )

    nx_signature = _sorted_normalized_distance_vector(nx_tensor)
    our_signature = _sorted_normalized_distance_vector(our_tensor)
    torch.testing.assert_close(our_signature, nx_signature, atol=0.05, rtol=0.0)

    labeled_corr = _pearson_correlation(
        _normalize_distance_vector(nx_tensor),
        _normalize_distance_vector(our_tensor),
    )
    assert labeled_corr > 0.85

    overlaps = _nearest_neighbor_overlaps(nx_tensor, our_tensor, k=3)
    assert sum(overlaps) / len(overlaps) >= 0.5
    assert sum(overlap >= 0.5 for overlap in overlaps) / len(overlaps) >= 0.6


@pytest.mark.skipif(not _NETWORKX_AVAILABLE, reason="networkx is not installed")
def test_kk_stress_matches_networkx() -> None:
    """Compare KK against NetworkX via stress and permutation-robust distances.

    Returns
    -------
    None
        The assertions validate comparable stress minimization quality.
    """
    nx = pytest.importorskip("networkx")

    edge_index, num_nodes, graph = _make_petersen_graph()
    nx_tensor = _positions_from_networkx_dict(nx.kamada_kawai_layout(graph), num_nodes=num_nodes)
    our_tensor = layout_kk(edge_index=edge_index, num_nodes=num_nodes, steps=500, seed=42)

    graph_distances = _all_pairs_shortest_paths(graph, num_nodes=num_nodes)
    nx_stress = _compute_stress(nx_tensor, graph_distances=graph_distances, num_nodes=num_nodes)
    our_stress = _compute_stress(our_tensor, graph_distances=graph_distances, num_nodes=num_nodes)

    assert our_stress <= (2.0 * nx_stress)

    nx_signature = _sorted_normalized_distance_vector(nx_tensor)
    our_signature = _sorted_normalized_distance_vector(our_tensor)
    corr = _pearson_correlation(nx_signature, our_signature)
    assert corr > 0.95


@pytest.mark.skipif(
    not (_NETWORKX_AVAILABLE and _PYDOT_AVAILABLE),
    reason="networkx and pydot are required",
)
def test_sugiyama_matches_graphviz_dot() -> None:
    """Compare Sugiyama against Graphviz dot using layered-structure invariants.

    Returns
    -------
    None
        The assertions validate layer structure, crossings, and overlap behavior.
    """
    layered_edge_index, layered_num_nodes, _ = _make_layered_dag()
    gv_pos = _graphviz_dot_positions(layered_edge_index, num_nodes=layered_num_nodes)
    our_pos = layout_sugiyama(edge_index=layered_edge_index, num_nodes=layered_num_nodes)

    gv_layers = _infer_layer_indices(gv_pos, layered_edge_index)
    our_layers = _infer_layer_indices(our_pos, layered_edge_index)

    for source, target in zip(layered_edge_index[0].tolist(), layered_edge_index[1].tolist()):
        assert int(gv_layers[source].item()) < int(gv_layers[target].item())
        assert int(our_layers[source].item()) < int(our_layers[target].item())

    assert int(torch.unique(gv_layers).numel()) == int(torch.unique(our_layers).numel())
    assert _has_no_overlaps_within_layers(gv_pos, gv_layers)
    assert _has_no_overlaps_within_layers(our_pos, our_layers)

    gv_crossings = _count_crossings(layered_edge_index, gv_pos)
    our_crossings = _count_crossings(layered_edge_index, our_pos)
    if gv_crossings == 0:
        assert our_crossings == 0
    else:
        assert our_crossings <= (2 * gv_crossings)

    diamond_edge_index, diamond_num_nodes, _ = _make_diamond_dag()
    diamond_gv_pos = _graphviz_dot_positions(diamond_edge_index, num_nodes=diamond_num_nodes)
    diamond_our_pos = layout_sugiyama(edge_index=diamond_edge_index, num_nodes=diamond_num_nodes)

    diamond_gv_layers = _infer_layer_indices(diamond_gv_pos, diamond_edge_index)
    diamond_our_layers = _infer_layer_indices(diamond_our_pos, diamond_edge_index)

    assert int(torch.unique(diamond_gv_layers).numel()) == 3
    assert int(torch.unique(diamond_our_layers).numel()) == 3
    assert int(diamond_gv_layers[0].item()) < int(diamond_gv_layers[1].item())
    assert int(diamond_gv_layers[1].item()) == int(diamond_gv_layers[2].item())
    assert int(diamond_gv_layers[3].item()) > int(diamond_gv_layers[1].item())
    assert int(diamond_our_layers[0].item()) < int(diamond_our_layers[1].item())
    assert int(diamond_our_layers[1].item()) == int(diamond_our_layers[2].item())
    assert int(diamond_our_layers[3].item()) > int(diamond_our_layers[1].item())
