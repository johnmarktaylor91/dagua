"""Compare classic layout implementations against authoritative references.

FR  -> NetworkX spring_layout (Fruchterman-Reingold)
KK  -> NetworkX kamada_kawai_layout (stress minimization)
Sugiyama -> Graphviz dot (layered DAG layout)

These tests verify our educational reimplementations are faithful to the
original algorithms, not just "some force-directed thing."
"""

from __future__ import annotations

import importlib
import importlib.util
import math
from typing import Any

import pytest
import torch

from dagua.layout.classic import layout_fr, layout_kk, layout_stress_sgd, layout_sugiyama
from dagua.layout.classic.davidson_harel import _energy as _davidson_harel_energy
from dagua.layout.classic.fa2 import _adjust_speed_and_apply_forces as _fa2_adjust_speed
from dagua.layout.classic.gem import _attractive_force as _gem_attractive_force
from dagua.layout.classic.gem import _repulsive_force_full as _gem_repulsive_force_full
from dagua.layout.classic.gem import _rotate_impulse as _gem_rotate_impulse
from dagua.layout.classic.gem import _update_temperatures as _gem_update_temperatures
from dagua.layout.classic.linlog import _linlog_loss
from dagua.layout.classic.linlog import _sample_all_pairs as _linlog_sample_all_pairs
from dagua.layout.classic.maxent_stress import _build_undirected_adjacency as _maxent_adjacency
from dagua.layout.classic.maxent_stress import _full_non_edge_pairs as _maxent_full_non_edge_pairs
from dagua.layout.classic.maxent_stress import _full_stress_terms as _maxent_full_stress_terms
from dagua.layout.classic.maxent_stress import _majorization_iteration, _maxent_stress_loss
from dagua.layout.classic.stress_sgd import _learning_rate as _stress_sgd_learning_rate

_NETWORKX_AVAILABLE = importlib.util.find_spec("networkx") is not None
_PYDOT_AVAILABLE = importlib.util.find_spec("pydot") is not None
davidson_harel_module = importlib.import_module("dagua.layout.classic.davidson_harel")


def _path_edge_index(num_nodes: int) -> torch.Tensor:
    """Build a directed path edge index.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the path.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    if num_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long)

    source = torch.arange(0, num_nodes - 1, dtype=torch.long)
    target = source + 1
    return torch.stack([source, target], dim=0)


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


def test_fr_recenters_positions_around_origin() -> None:
    """FR should rescale the final layout around the origin."""
    edge_index = torch.empty((2, 0), dtype=torch.long)
    positions = layout_fr(edge_index=edge_index, num_nodes=12, steps=10, seed=7, area=1.0)

    assert torch.linalg.norm(positions.mean(dim=0)) < 1.0e-4
    assert float(positions.min().item()) < 0.0
    assert float(positions.max().item()) > 0.0


def test_kk_solver_variants_return_centered_layouts() -> None:
    """KK solver modes should all return centered 2D layouts."""
    edge_index, num_nodes, _ = _make_petersen_graph()

    auto_positions = layout_kk(edge_index=edge_index, num_nodes=num_nodes, steps=60, seed=42)
    newton_positions = layout_kk(
        edge_index=edge_index,
        num_nodes=num_nodes,
        steps=60,
        seed=42,
        solver="newton",
    )
    adam_positions = layout_kk(
        edge_index=edge_index,
        num_nodes=num_nodes,
        steps=60,
        seed=42,
        solver="adam",
    )

    for positions in (auto_positions, newton_positions, adam_positions):
        assert positions.shape == (num_nodes, 2)
        assert torch.linalg.norm(positions.mean(dim=0)) < 1.0e-4


def test_fa2_adjust_speed_matches_reference_math() -> None:
    """FA2 should match the reference speed adaptation and movement factors."""
    pos = torch.tensor([[0.0, 0.0], [1.0, -1.0]], dtype=torch.float32)
    old_force = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    force = torch.tensor([[0.0, 1.0], [2.0, 0.0]], dtype=torch.float32)
    mass = torch.tensor([2.0, 3.0], dtype=torch.float32)

    updated_pos, speed, speed_efficiency = _fa2_adjust_speed(
        pos=pos,
        force=force,
        old_force=old_force,
        mass=mass,
        speed=1.0,
        speed_efficiency=1.0,
        jitter_tolerance=1.0,
    )

    total_swinging = 5.0 * math.sqrt(2.0)
    total_effective_traction = 2.5 * math.sqrt(2.0)
    estimated_optimal_jt = 0.05 * math.sqrt(2.0)
    min_jt = math.sqrt(estimated_optimal_jt)
    jt = max(min_jt, estimated_optimal_jt * total_effective_traction / 4.0)
    target_speed = jt * total_effective_traction / total_swinging
    expected_speed = 1.0 + min(target_speed - 1.0, 0.5)
    expected_factor = expected_speed / (
        1.0
        + torch.sqrt(expected_speed * (mass * torch.linalg.vector_norm(old_force - force, dim=1)))
    )
    expected_pos = pos + (force * expected_factor.unsqueeze(1))

    torch.testing.assert_close(updated_pos, expected_pos)
    assert speed == pytest.approx(expected_speed)
    assert speed_efficiency == pytest.approx(0.7)


def test_gem_repulsion_matches_fr_force_law() -> None:
    """GEM repulsion should use the FR-style ``k^2 / dist`` magnitude."""
    positions = torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
    force = _gem_repulsive_force_full(positions=positions, ideal_distance=3.0)

    expected = torch.tensor([[-4.5, 0.0], [4.5, 0.0]], dtype=torch.float32)
    torch.testing.assert_close(force, expected)


def test_gem_attraction_divides_by_ideal_distance() -> None:
    """GEM attraction should include OGDF's ``1 / (k * weight(v))`` scale."""
    positions = torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)

    force = _gem_attractive_force(positions=positions, edge_index=edge_index, ideal_distance=4.0)

    expected = torch.tensor([[1.0 / 1.4, 0.0], [-(1.0 / 1.4), 0.0]], dtype=torch.float32)
    torch.testing.assert_close(force, expected)


def test_gem_rotation_and_temperature_updates_match_reference_rules() -> None:
    """GEM should use OGDF's oscillation and skew-gauge temperature updates."""
    impulse = torch.tensor([[3.0, 4.0], [0.0, 2.0]], dtype=torch.float32)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(3)
    rotated = _gem_rotate_impulse(impulse, generator=generator, device=torch.device("cpu"))

    torch.testing.assert_close(rotated.norm(dim=1), impulse.norm(dim=1))

    temperatures = torch.ones((3,), dtype=torch.float32)
    current_impulse = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    previous_impulse = torch.tensor([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
    skew_gauge = torch.zeros((3,), dtype=torch.float32)
    updated, updated_skew = _gem_update_temperatures(
        temperatures=temperatures,
        current_impulse=current_impulse,
        previous_impulse=previous_impulse,
        skew_gauge=skew_gauge,
        initial_temperature=2.0,
    )

    expected = torch.tensor([1.287, 0.7, 1.0], dtype=torch.float32)
    torch.testing.assert_close(updated, expected)
    torch.testing.assert_close(updated_skew, torch.tensor([0.01, 0.0, 0.0], dtype=torch.float32))


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
    """Compare FR against the exact NetworkX spring-layout reference.

    Returns
    -------
    None
        The assertions validate a low Procrustes disparity.
    """
    nx = pytest.importorskip("networkx")
    spatial = pytest.importorskip("scipy.spatial")

    edge_index, num_nodes, _ = _make_small_random_graph()
    directed_graph = nx.DiGraph()
    directed_graph.add_nodes_from(range(num_nodes))
    directed_graph.add_edges_from(edge_index.transpose(0, 1).tolist())

    nx_pos = nx.spring_layout(directed_graph, iterations=50, seed=42, scale=1)
    nx_tensor = _positions_from_networkx_dict(nx_pos, num_nodes=num_nodes)
    our_tensor = layout_fr(edge_index=edge_index, num_nodes=num_nodes, steps=50, seed=42)
    our_tensor = _normalize_positions(our_tensor)

    _, _, disparity = spatial.procrustes(nx_tensor.numpy(), our_tensor.numpy())
    assert disparity < 0.01


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


def test_stress_sgd_auto_matches_full_epoch_for_small_graphs() -> None:
    """Stress-SGD should default to a full epoch on small graphs."""
    edge_index = _path_edge_index(6)

    auto_positions = layout_stress_sgd(
        edge_index=edge_index,
        num_nodes=6,
        steps=80,
        seed=17,
        sample_size="auto",
    )
    full_epoch_positions = layout_stress_sgd(
        edge_index=edge_index,
        num_nodes=6,
        steps=80,
        seed=17,
        sample_size=6,
    )

    torch.testing.assert_close(auto_positions, full_epoch_positions)


def test_stress_sgd_learning_rate_matches_paper_schedule() -> None:
    """Stress-SGD should match the exponential ``s_gd2`` learning-rate schedule."""
    assert _stress_sgd_learning_rate(0, 5, 1.0, 4.0) == pytest.approx(16.0)
    assert _stress_sgd_learning_rate(2, 5, 1.0, 4.0) == pytest.approx(0.4)
    assert _stress_sgd_learning_rate(4, 5, 1.0, 4.0) == pytest.approx(0.01)


def test_linlog_matches_noack_sum_energy() -> None:
    """LinLog should repel all unordered node pairs in the Noack energy model."""
    positions = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]], dtype=torch.float32)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    distances = torch.pdist(positions, p=2)
    expected = distances[[0, 2]].sum() - torch.log(distances).sum()

    actual = _linlog_loss(positions, edge_index, seed=0, step=0)

    torch.testing.assert_close(actual, expected)


def test_linlog_supports_generalized_exponents_without_gravity() -> None:
    """Generalized LinLog exponents should repel all pairwise distances."""
    positions = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]], dtype=torch.float32)
    offset = torch.tensor([11.0, -7.0], dtype=torch.float32)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    distances = torch.pdist(positions, p=2)
    expected = distances[[0, 2]].pow(2.0).sum() - distances.pow(1.5).sum()

    actual = _linlog_loss(positions, edge_index, seed=0, step=0, a=2.0, r=1.5)
    translated = _linlog_loss(positions + offset, edge_index, seed=0, step=0, a=2.0, r=1.5)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(translated, actual)


def test_linlog_sampled_repulsion_draws_from_all_pairs() -> None:
    """The sampled LinLog repulsion path should include edge pairs as well."""
    edge_index = _path_edge_index(6)

    src, dst, total_pairs = _linlog_sample_all_pairs(
        num_nodes=6,
        device=torch.device("cpu"),
        step=0,
        seed=0,
    )

    edge_pairs = {
        (min(source, target), max(source, target))
        for source, target in zip(edge_index[0].tolist(), edge_index[1].tolist())
    }
    sampled_pairs = {(int(source.item()), int(target.item())) for source, target in zip(src, dst)}

    assert total_pairs == 15
    assert src.numel() == 15
    assert all(int(source.item()) < int(target.item()) for source, target in zip(src, dst))
    assert edge_pairs <= sampled_pairs


def test_davidson_harel_segments_intersect_treats_tiny_orientations_as_collinear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tiny orientation magnitudes should be treated as collinear intersections."""
    orientations = iter((1.0e-12, 1.0, -1.0, -1.0e-12))

    def fake_orientation(
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
    ) -> float:
        """Return a scripted sequence of near-collinear orientation values."""
        _ = (a, b, c)
        return next(orientations)

    monkeypatch.setattr(davidson_harel_module, "_orientation", fake_orientation)

    origin = torch.zeros(2, dtype=torch.float32)

    assert davidson_harel_module._segments_intersect(origin, origin, origin, origin)


def test_maxent_stress_includes_longer_shortest_paths() -> None:
    """Maxent-stress should include stress pairs beyond two hops on small graphs."""
    edge_index = _path_edge_index(4)
    adjacency = _maxent_adjacency(edge_index, num_nodes=4)

    stress_src, stress_dst, stress_lengths = _maxent_full_stress_terms(adjacency)
    stress_map = {
        (int(source.item()), int(target.item())): float(length.item())
        for source, target, length in zip(stress_src, stress_dst, stress_lengths)
    }

    assert stress_map[(0, 3)] == pytest.approx(3.0)


def test_maxent_stress_replaces_disconnected_pairs_with_sqrt_n_distance() -> None:
    """Disconnected stress pairs should use OGDF's ``avgEdgeCost * sqrt(N)`` fill."""
    edge_index = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    adjacency = _maxent_adjacency(edge_index, num_nodes=4)

    stress_src, stress_dst, stress_lengths = _maxent_full_stress_terms(adjacency)
    stress_map = {
        (int(source.item()), int(target.item())): float(length.item())
        for source, target, length in zip(stress_src, stress_dst, stress_lengths)
    }

    assert stress_map[(0, 2)] == pytest.approx(2.0)
    assert stress_map[(1, 3)] == pytest.approx(2.0)


def test_maxent_stress_matches_exact_small_graph_objective() -> None:
    """Maxent-stress should use full-path stress and exact non-edge log repulsion."""
    positions = torch.tensor([[0.0, 0.0], [2.0, 0.0], [5.0, 0.0]], dtype=torch.float32)
    edge_index = _path_edge_index(3)
    adjacency = _maxent_adjacency(edge_index, num_nodes=3)
    stress_src, stress_dst, stress_lengths = _maxent_full_stress_terms(adjacency)
    non_edge_src, non_edge_dst = _maxent_full_non_edge_pairs(adjacency)

    actual = _maxent_stress_loss(
        positions=positions,
        stress_src=stress_src,
        stress_dst=stress_dst,
        stress_lengths=stress_lengths,
        pivot_indices=torch.empty((0,), dtype=torch.long),
        pivot_distances=torch.empty((3, 0), dtype=torch.float32),
        non_edge_src=non_edge_src,
        non_edge_dst=non_edge_dst,
        alpha=0.5,
        use_entropy=True,
    )

    expected_stress = (2.0 - 1.0) ** 2 + (3.0 - 1.0) ** 2 + 0.25 * ((5.0 - 2.0) ** 2)
    expected = expected_stress - (0.5 * math.log(5.0))

    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float32))

    translated = _maxent_stress_loss(
        positions=positions + torch.tensor([13.0, -4.0], dtype=torch.float32),
        stress_src=stress_src,
        stress_dst=stress_dst,
        stress_lengths=stress_lengths,
        pivot_indices=torch.empty((0,), dtype=torch.long),
        pivot_distances=torch.empty((3, 0), dtype=torch.float32),
        non_edge_src=non_edge_src,
        non_edge_dst=non_edge_dst,
        alpha=0.5,
        use_entropy=True,
    )
    torch.testing.assert_close(translated, actual)


def test_maxent_stress_majorization_uses_gauss_seidel_votes() -> None:
    """Stress majorization should update nodes sequentially within one sweep."""
    positions = torch.tensor([[0.0, 0.0], [2.0, 0.0], [5.0, 0.0]], dtype=torch.float64)
    graph_distances = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )
    weight_matrix = torch.tensor(
        [
            [0.0, 1.0, 0.25],
            [1.0, 0.0, 1.0],
            [0.25, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )

    _majorization_iteration(positions, graph_distances, weight_matrix)

    expected = torch.tensor([[1.4, 0.0], [3.2, 0.0], [4.04, 0.0]], dtype=torch.float64)
    torch.testing.assert_close(positions, expected)


def test_maxent_stress_pure_mode_drops_entropy_term() -> None:
    """The pure-stress mode should ignore the non-edge entropy repulsion term."""
    positions = torch.tensor([[0.0, 0.0], [2.0, 0.0], [5.0, 0.0]], dtype=torch.float32)
    edge_index = _path_edge_index(3)
    adjacency = _maxent_adjacency(edge_index, num_nodes=3)
    stress_src, stress_dst, stress_lengths = _maxent_full_stress_terms(adjacency)
    non_edge_src, non_edge_dst = _maxent_full_non_edge_pairs(adjacency)

    actual = _maxent_stress_loss(
        positions=positions,
        stress_src=stress_src,
        stress_dst=stress_dst,
        stress_lengths=stress_lengths,
        pivot_indices=torch.empty((0,), dtype=torch.long),
        pivot_distances=torch.empty((3, 0), dtype=torch.float32),
        non_edge_src=non_edge_src,
        non_edge_dst=non_edge_dst,
        alpha=5.0,
        use_entropy=False,
    )

    expected = (2.0 - 1.0) ** 2 + (3.0 - 1.0) ** 2 + 0.25 * ((5.0 - 2.0) ** 2)
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float32))


@pytest.mark.smoke
def test_fmmm_force_model_matches_fr_coefficients() -> None:
    """FM^3 refinement should match OGDF's default force coefficients."""
    from dagua.layout.classic.fmmm import _attractive_force as fmmm_attractive_force
    from dagua.layout.classic.fmmm import _barnes_hut_repulsion as fmmm_barnes_hut_repulsion

    positions = torch.tensor([[0.0, 0.0], [3.0, 0.0]], dtype=torch.float32)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)

    repulsive = fmmm_barnes_hut_repulsion(positions, theta=0.0, ideal_length=2.0)
    attractive = fmmm_attractive_force(positions, edge_index, ideal_length=2.0)

    torch.testing.assert_close(
        repulsive,
        torch.tensor([[-1.0 / 3.0, 0.0], [1.0 / 3.0, 0.0]], dtype=torch.float32),
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    torch.testing.assert_close(
        attractive,
        torch.tensor(
            [
                [math.log2(3.0 / 2.0) * 9.0 / 8.0, 0.0],
                [-(math.log2(3.0 / 2.0) * 9.0 / 8.0), 0.0],
            ],
            dtype=torch.float32,
        ),
        atol=1.0e-5,
        rtol=1.0e-5,
    )


def test_davidson_harel_uses_one_move_per_node_per_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Davidson-Harel should use one candidate move per node in each round."""
    davidson_harel_module = importlib.import_module("dagua.layout.classic.davidson_harel")

    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    energy_calls = 0

    def fake_energy(
        positions: torch.Tensor,
        edges: list[tuple[int, int]],
        extent: float,
    ) -> torch.Tensor:
        """Count energy evaluations without changing the acceptance logic."""
        del edges, extent
        nonlocal energy_calls
        energy_calls += 1
        return torch.tensor(0.0, dtype=positions.dtype, device=positions.device)

    monkeypatch.setattr(davidson_harel_module, "_energy", fake_energy)

    positions = davidson_harel_module.layout_davidson_harel(
        edge_index=edge_index,
        num_nodes=3,
        rounds=2,
        seed=0,
    )

    assert positions.shape == (3, 2)
    assert energy_calls == 1 + (3 * 2)
    assert davidson_harel_module._COOLING_FACTOR == pytest.approx(0.75)


def test_davidson_harel_energy_uses_summed_terms_and_all_borders() -> None:
    """Davidson-Harel should use sum-based energies with four-border repulsion."""
    positions = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]],
        dtype=torch.float32,
    )
    edges = [(0, 1)]
    extent = 3.0

    distribution = (1.0 / 4.0) + (1.0 / 2.0) + (1.0 / 2.0)
    border = (
        4.0 / 9.0
        + (1.0 / 25.0 + 1.0 + 1.0 / 9.0 + 1.0 / 9.0)
        + (1.0 / 16.0 + 1.0 / 4.0 + 1.0 / 16.0 + 1.0 / 4.0)
    )
    edge_length = 4.0
    node_edge = 1.0
    expected = (
        distribution / 3.0 + 0.1 * (border / 3.0) + 0.2 * edge_length + 0.5 * (node_edge / 3.0)
    )

    actual = float(_davidson_harel_energy(positions, edges, extent).item())

    assert actual == pytest.approx(expected)
