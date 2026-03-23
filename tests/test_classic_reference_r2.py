"""Reference comparisons for the second batch of classic layout algorithms."""

from __future__ import annotations

import importlib.util
import json
import random
import subprocess
import warnings
from collections import deque
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable

import numpy as np
import pytest
import torch

from dagua.layout.classic.davidson_harel import layout_davidson_harel
from dagua.layout.classic.fmmm import layout_fmmm
from dagua.layout.classic.fr import layout_fr
from dagua.layout.classic.gem import layout_gem
from dagua.layout.classic.kk import layout_kk
from dagua.layout.classic.linlog import _linlog_loss, layout_linlog
from dagua.layout.classic.maxent_stress import layout_maxent_stress
from dagua.layout.classic.pivot_mds import layout_pivot_mds
from dagua.layout.classic.spectral import layout_spectral
from dagua.layout.classic.stress_sgd import layout_stress_sgd
from dagua.layout.classic.sugiyama import layout_sugiyama
from dagua.layout.classic.tsnet import layout_tsnet
from dagua.metrics import count_crossings, edge_length_cv
from tests.test_classic_reference import (
    _make_diamond_dag,
    _make_layered_dag,
    _make_petersen_graph,
    _normalize_positions,
    _pairwise_distances,
    _pearson_correlation,
    _upper_triangle_values,
)

_NETWORKX_AVAILABLE = importlib.util.find_spec("networkx") is not None
_SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None
_SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None
_OGDF_RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "ogdf_runner"

try:
    import igraph as _igraph_module
except ImportError:
    _igraph_module = None

_IGRAPH_AVAILABLE = _igraph_module is not None
if _IGRAPH_AVAILABLE:
    _IGRAPH_GEM_AVAILABLE = hasattr(_igraph_module.Graph, "layout_gem")
    if not _IGRAPH_GEM_AVAILABLE:
        try:
            _igraph_module.Graph(n=2, edges=[(0, 1)], directed=False).layout("gem")
            _IGRAPH_GEM_AVAILABLE = True
        except Exception:
            _IGRAPH_GEM_AVAILABLE = False
else:
    _IGRAPH_GEM_AVAILABLE = False


def _make_karate_edge_index() -> tuple[torch.Tensor, int, Any]:
    """Build an unweighted Karate Club graph in torch and NetworkX formats.

    Returns
    -------
    tuple[torch.Tensor, int, Any]
        ``(edge_index, num_nodes, graph)`` for an unweighted Karate Club graph.
    """
    nx = pytest.importorskip("networkx")
    base_graph = nx.karate_club_graph()
    graph = nx.Graph()
    graph.add_nodes_from(base_graph.nodes())
    graph.add_edges_from(base_graph.edges())
    num_nodes = graph.number_of_nodes()
    edges = list(graph.edges())
    sources = [source for source, _ in edges] + [target for _, target in edges]
    targets = [target for _, target in edges] + [source for source, _ in edges]
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    return edge_index, num_nodes, graph


def _normalize_for_procrustes(positions: np.ndarray) -> np.ndarray:
    """Center and scale positions before Procrustes comparison.

    Parameters
    ----------
    positions : numpy.ndarray
        Position array with shape ``[N, 2]``.

    Returns
    -------
    numpy.ndarray
        Centered array with max absolute coordinate at most ``1``.
    """
    centered = positions - positions.mean(axis=0, keepdims=True)
    scale = float(np.abs(centered).max())
    if scale > 1.0e-6:
        centered = centered / scale
    return centered


def _procrustes_disparity(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Compute SciPy Procrustes disparity between two layouts.

    Parameters
    ----------
    reference : numpy.ndarray
        Reference coordinates with shape ``[N, 2]``.
    candidate : numpy.ndarray
        Candidate coordinates with shape ``[N, 2]``.

    Returns
    -------
    float
        Procrustes disparity.
    """
    spatial = pytest.importorskip("scipy.spatial")
    _, _, disparity = spatial.procrustes(reference, candidate)
    return float(disparity)


def _ogdf_layout_reference(
    num_nodes: int,
    edges: list[list[int]],
    algorithm: str,
) -> np.ndarray:
    """Run the standalone OGDF helper and return its coordinates.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    edges : list[list[int]]
        Edge list shaped like ``[[source, target], ...]``.
    algorithm : str
        OGDF algorithm selector understood by ``scripts/ogdf_runner``.

    Returns
    -------
    numpy.ndarray
        Position array with shape ``[N, 2]``.
    """
    if not _OGDF_RUNNER.exists():
        pytest.skip("OGDF runner is not available in this checkout")

    payload = json.dumps({"nodes": num_nodes, "edges": edges, "algorithm": algorithm})
    result = subprocess.run(
        [str(_OGDF_RUNNER)],
        input=payload,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(result.stderr.strip() or f"OGDF {algorithm} runner failed")

    output = json.loads(result.stdout)
    return np.asarray(output["positions"], dtype=np.float64)


def _make_connected_random_graph(seed: int = 42) -> tuple[torch.Tensor, int]:
    """Build a small connected undirected random graph.

    Parameters
    ----------
    seed : int, default=42
        Seed used for deterministic edge sampling.

    Returns
    -------
    tuple[torch.Tensor, int]
        ``(edge_index, num_nodes)`` for a 20-node connected graph.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    num_nodes = 20
    edges = {(node, node + 1) for node in range(num_nodes - 1)}

    while len(edges) < 40:
        source = int(torch.randint(0, num_nodes, (1,), generator=generator).item())
        target = int(torch.randint(0, num_nodes, (1,), generator=generator).item())
        if source == target:
            continue
        edges.add(tuple(sorted((source, target))))

    edge_index = torch.tensor(sorted(edges), dtype=torch.long).transpose(0, 1).contiguous()
    return edge_index, num_nodes


def _make_grid_graph() -> tuple[torch.Tensor, int]:
    """Build a 4x4 grid graph.

    Returns
    -------
    tuple[torch.Tensor, int]
        ``(edge_index, num_nodes)`` for the grid graph.
    """
    edges: list[tuple[int, int]] = []
    width = 4
    height = 4

    for row in range(height):
        for col in range(width):
            node = row * width + col
            if col + 1 < width:
                edges.append((node, node + 1))
            if row + 1 < height:
                edges.append((node, node + width))

    edge_index = torch.tensor(edges, dtype=torch.long).transpose(0, 1).contiguous()
    return edge_index, width * height


def _make_small_dh_graph() -> tuple[torch.Tensor, int]:
    """Build a very small connected graph for Davidson-Harel comparisons.

    Returns
    -------
    tuple[torch.Tensor, int]
        ``(edge_index, num_nodes)`` for a 6-node graph.
    """
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 0, 0, 1],
            [1, 2, 3, 4, 5, 5, 3, 4],
        ],
        dtype=torch.long,
    )
    return edge_index, 6


def _edge_list(edge_index: torch.Tensor) -> list[tuple[int, int]]:
    """Convert an edge tensor into a Python edge list.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.

    Returns
    -------
    list[tuple[int, int]]
        Edge tuples.
    """
    return [tuple(edge) for edge in edge_index.transpose(0, 1).tolist()]


def _make_networkx_graph(edge_index: torch.Tensor, num_nodes: int) -> Any:
    """Build a NetworkX graph from an edge tensor.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    Any
        ``networkx.Graph`` instance.
    """
    nx = pytest.importorskip("networkx")
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(_edge_list(edge_index))
    return graph


def _make_igraph_graph(edge_index: torch.Tensor, num_nodes: int) -> Any:
    """Build an igraph graph from an edge tensor.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    Any
        ``igraph.Graph`` instance.
    """
    ig = pytest.importorskip("igraph")
    return ig.Graph(n=num_nodes, edges=_edge_list(edge_index), directed=False)


def _make_igraph_digraph(edge_index: torch.Tensor, num_nodes: int) -> Any:
    """Build a directed igraph graph from an edge tensor.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    Any
        ``igraph.Graph`` instance with ``directed=True``.
    """
    ig = pytest.importorskip("igraph")
    return ig.Graph(n=num_nodes, edges=_edge_list(edge_index), directed=True)


def _make_sparse_adjacency(edge_index: torch.Tensor, num_nodes: int) -> Any:
    """Build a symmetric SciPy sparse adjacency matrix.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    Any
        ``scipy.sparse.csr_matrix`` adjacency matrix.
    """
    scipy_sparse = pytest.importorskip("scipy.sparse")

    rows: list[int] = []
    cols: list[int] = []
    for source, target in _edge_list(edge_index):
        rows.extend((source, target))
        cols.extend((target, source))

    data = np.ones(len(rows), dtype=np.float64)
    return scipy_sparse.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))


def _all_pairs_shortest_path_lengths(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Compute all-pairs unweighted shortest-path distances.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    torch.Tensor
        Distance matrix with shape ``[N, N]``.
    """
    adjacency = [set() for _ in range(num_nodes)]
    for source, target in _edge_list(edge_index):
        adjacency[source].add(target)
        adjacency[target].add(source)

    distances: list[list[float]] = []
    diameter = 0
    for source in range(num_nodes):
        row = [-1] * num_nodes
        row[source] = 0
        frontier: deque[int] = deque([source])

        while frontier:
            node = frontier.popleft()
            next_distance = row[node] + 1
            for neighbor in adjacency[node]:
                if row[neighbor] >= 0:
                    continue
                row[neighbor] = next_distance
                diameter = max(diameter, next_distance)
                frontier.append(neighbor)

        distances.append(row)

    fill_value = float(diameter + 1 if num_nodes > 1 else 0.0)
    return torch.tensor(
        [
            [fill_value if distance < 0 else float(distance) for distance in row]
            for row in distances
        ],
        dtype=torch.float32,
    )


def _exact_stress(pos: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> float:
    """Compute a scale-normalized exact stress score for a small graph.

    Parameters
    ----------
    pos : torch.Tensor
        Layout positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    float
        Mean weighted stress over unique node pairs.
    """
    normalized = _normalize_positions(pos)
    graph_distances = _all_pairs_shortest_path_lengths(edge_index, num_nodes)
    euclidean_distances = _pairwise_distances(normalized)
    upper = torch.triu_indices(num_nodes, num_nodes, offset=1)
    graph_values = graph_distances[upper[0], upper[1]]
    euclidean_values = euclidean_distances[upper[0], upper[1]]
    weights = 1.0 / graph_values.square().clamp(min=1.0)
    return float((weights * (graph_values - euclidean_values).square()).mean().item())


def _min_pairwise_distance(pos: torch.Tensor) -> float:
    """Compute the minimum off-diagonal pairwise distance.

    Parameters
    ----------
    pos : torch.Tensor
        Layout positions with shape ``[N, 2]``.

    Returns
    -------
    float
        Minimum pairwise distance after unit-square normalization.
    """
    distances = _pairwise_distances(_normalize_positions(pos))
    values = _upper_triangle_values(distances)
    return float(values.min().item()) if values.numel() > 0 else 0.0


def _layout_metric_summary(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> dict[str, float]:
    """Compute a compact quality summary for a layout.

    Parameters
    ----------
    pos : torch.Tensor
        Layout positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    dict[str, float]
        Summary metrics for stress, crossings, and edge-length variation.
    """
    normalized = _normalize_positions(pos)
    return {
        "stress": _exact_stress(normalized, edge_index, num_nodes),
        "crossing_count": float(count_crossings(normalized, edge_index)),
        "edge_length_cv": float(edge_length_cv(normalized, edge_index)["edge_length_cv"]),
    }


def _distribution_mean_and_std(values: list[float]) -> tuple[float, float]:
    """Compute population mean and standard deviation.

    Parameters
    ----------
    values : list[float]
        Sample values.

    Returns
    -------
    tuple[float, float]
        Mean and population standard deviation.
    """
    return mean(values), pstdev(values) if len(values) > 1 else 0.0


def _assert_metric_distribution_overlap(
    ours: list[dict[str, float]],
    theirs: list[dict[str, float]],
    metric_name: str,
    tolerance_multiplier: float = 1.0,
) -> None:
    """Assert that two stochastic metric distributions overlap.

    Parameters
    ----------
    ours : list[dict[str, float]]
        Metrics from our implementation across seeds.
    theirs : list[dict[str, float]]
        Metrics from the reference implementation across seeds.
    metric_name : str
        Metric key to compare.
    tolerance_multiplier : float, default=1.0
        Additional multiplier applied to the overlap tolerance for
        high-variance stochastic metrics.
    """
    ours_values = [metrics[metric_name] for metrics in ours]
    theirs_values = [metrics[metric_name] for metrics in theirs]
    ours_mean, ours_std = _distribution_mean_and_std(ours_values)
    theirs_mean, theirs_std = _distribution_mean_and_std(theirs_values)
    tolerance = 2.0 * tolerance_multiplier * max(ours_std, theirs_std, 1.0e-6)
    assert abs(ours_mean - theirs_mean) < tolerance


def _run_igraph_layout(graph: Any, layout_name: str, seed: int) -> torch.Tensor:
    """Run an igraph layout with deterministic seeding.

    Parameters
    ----------
    graph : Any
        ``igraph.Graph`` instance.
    layout_name : str
        Layout name or method suffix.
    seed : int
        Seed passed to igraph's RNG bridge.

    Returns
    -------
    torch.Tensor
        Layout positions with shape ``[N, 2]``.
    """
    ig = pytest.importorskip("igraph")
    ig.set_random_number_generator(random.Random(seed))

    method_name = f"layout_{layout_name}"
    if hasattr(graph, method_name):
        layout = getattr(graph, method_name)()
    else:
        layout = graph.layout(layout_name)
    array = np.asarray(list(layout), dtype=np.float32)
    return torch.from_numpy(array)


def _run_sklearn_mds(distance_matrix: Any, seed: int) -> torch.Tensor:
    """Run sklearn metric MDS on a precomputed distance matrix.

    Parameters
    ----------
    distance_matrix : Any
        Square graph distance matrix.
    seed : int
        Random seed for the sklearn estimator.

    Returns
    -------
    torch.Tensor
        Layout positions with shape ``[N, 2]``.
    """
    sklearn_manifold = pytest.importorskip("sklearn.manifold")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        estimator = sklearn_manifold.MDS(
            n_components=2,
            metric_mds=True,
            init="random",
            random_state=seed,
            n_init=4,
            max_iter=300,
            eps=1.0e-6,
            dissimilarity="precomputed",
            normalized_stress="auto",
        )
        positions = estimator.fit_transform(distance_matrix)
    return torch.tensor(positions, dtype=torch.float32)


def _run_sklearn_tsne(distance_matrix: Any, seed: int, perplexity: float) -> torch.Tensor:
    """Run sklearn t-SNE on a precomputed distance matrix.

    Parameters
    ----------
    distance_matrix : Any
        Square graph distance matrix.
    seed : int
        Random seed for the sklearn estimator.
    perplexity : float
        Target perplexity.

    Returns
    -------
    torch.Tensor
        Layout positions with shape ``[N, 2]``.
    """
    sklearn_manifold = pytest.importorskip("sklearn.manifold")
    distance_matrix_copy = np.array(distance_matrix, copy=True)
    estimator = sklearn_manifold.TSNE(
        n_components=2,
        metric="precomputed",
        init="random",
        perplexity=perplexity,
        random_state=seed,
        method="exact",
        max_iter=500,
    )
    positions = estimator.fit_transform(distance_matrix_copy)
    return torch.tensor(positions, dtype=torch.float32)


def _within_between_distance_stats(
    pos: torch.Tensor,
    graph_distances: torch.Tensor,
    within_radius: float,
    between_radius: float,
) -> tuple[float, float, float]:
    """Summarize embedded distances for near and far graph pairs.

    Parameters
    ----------
    pos : torch.Tensor
        Layout positions with shape ``[N, 2]``.
    graph_distances : torch.Tensor
        All-pairs graph distance matrix with shape ``[N, N]``.
    within_radius : float
        Maximum graph distance counted as a near pair.
    between_radius : float
        Minimum graph distance counted as a far pair.

    Returns
    -------
    tuple[float, float, float]
        Mean near-pair distance, mean far-pair distance, and the far/near
        separation ratio.
    """
    normalized = _normalize_positions(pos)
    euclidean = _pairwise_distances(normalized)
    upper = torch.triu_indices(pos.shape[0], pos.shape[0], offset=1)
    graph_values = graph_distances[upper[0], upper[1]]
    euclidean_values = euclidean[upper[0], upper[1]]
    within = euclidean_values[graph_values <= within_radius]
    between = euclidean_values[graph_values >= between_radius]
    within_mean = float(within.mean().item())
    between_mean = float(between.mean().item())
    return within_mean, between_mean, between_mean / within_mean


def _make_two_cliques_with_bridge() -> tuple[torch.Tensor, int, torch.Tensor]:
    """Build two six-node cliques connected by a single bridge edge.

    Returns
    -------
    tuple[torch.Tensor, int, torch.Tensor]
        ``(edge_index, num_nodes, community_labels)``.
    """
    clique_size = 6
    num_nodes = clique_size * 2
    edges: list[tuple[int, int]] = []

    for offset in (0, clique_size):
        for source in range(offset, offset + clique_size):
            for target in range(source + 1, offset + clique_size):
                edges.append((source, target))

    edges.append((clique_size - 1, clique_size))
    edge_index = torch.tensor(edges, dtype=torch.long).transpose(0, 1).contiguous()
    labels = torch.tensor([0] * clique_size + [1] * clique_size, dtype=torch.long)
    return edge_index, num_nodes, labels


def _layout_linlog_all_pairs_baseline(
    edge_index: torch.Tensor,
    num_nodes: int,
    steps: int,
    seed: int,
) -> torch.Tensor:
    """Optimize the pre-fix LinLog objective with all-pairs repulsion.

    Parameters
    ----------
    edge_index : torch.Tensor
        Undirected edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    steps : int
        Number of Adam updates.
    seed : int
        Random seed for deterministic initialization.

    Returns
    -------
    torch.Tensor
        Normalized layout positions with shape ``[N, 2]``.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    positions = torch.randn((num_nodes, 2), generator=generator, dtype=torch.float32)
    positions = positions.requires_grad_(True)
    learning_rate = min(0.05, 0.8 / float(max(num_nodes, 1)))
    optimizer = torch.optim.Adam([positions], lr=learning_rate)

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        src = edge_index[0].to(dtype=torch.long)
        dst = edge_index[1].to(dtype=torch.long)
        attraction = torch.linalg.norm(positions[src] - positions[dst], dim=1).clamp(min=1.0e-3)
        pair_src, pair_dst = torch.triu_indices(num_nodes, num_nodes, offset=1)
        pair_lengths = torch.linalg.norm(
            positions[pair_src] - positions[pair_dst],
            dim=1,
        ).clamp(min=1.0e-3)
        loss = attraction.sum() - torch.log(pair_lengths).sum()
        loss.backward()
        optimizer.step()

    return _normalize_positions(positions.detach())


def _community_separation_ratio(pos: torch.Tensor, labels: torch.Tensor) -> float:
    """Measure inter-community separation relative to within-community spread.

    Parameters
    ----------
    pos : torch.Tensor
        Layout positions with shape ``[N, 2]``.
    labels : torch.Tensor
        Community labels with shape ``[N]``.

    Returns
    -------
    float
        Centroid separation divided by the combined mean radial spread.
    """
    community_zero = pos[labels == 0]
    community_one = pos[labels == 1]
    center_zero = community_zero.mean(dim=0)
    center_one = community_one.mean(dim=0)
    between = torch.linalg.norm(center_zero - center_one)
    within = torch.linalg.norm(community_zero - center_zero, dim=1).mean()
    within += torch.linalg.norm(community_one - center_one, dim=1).mean()
    return float((between / within.clamp(min=1.0e-6)).item())


def _graph_knn_preservation(
    pos: torch.Tensor,
    graph_distances: torch.Tensor,
    k: int,
) -> float:
    """Measure how well Euclidean k-nearest neighbors match graph neighbors.

    Parameters
    ----------
    pos : torch.Tensor
        Layout positions with shape ``[N, 2]``.
    graph_distances : torch.Tensor
        All-pairs graph distance matrix.
    k : int
        Neighbor count to compare.

    Returns
    -------
    float
        Mean node-wise overlap fraction.
    """
    normalized = _normalize_positions(pos)
    euclidean = _pairwise_distances(normalized)
    euclidean.fill_diagonal_(float("inf"))
    graph = graph_distances.clone()
    graph.fill_diagonal_(float("inf"))

    overlaps: list[float] = []
    for node in range(normalized.shape[0]):
        graph_neighbors = set(torch.topk(graph[node], k=k, largest=False).indices.tolist())
        euclidean_neighbors = set(torch.topk(euclidean[node], k=k, largest=False).indices.tolist())
        overlaps.append(len(graph_neighbors & euclidean_neighbors) / float(k))

    return mean(overlaps)


@pytest.mark.skipif(not _NETWORKX_AVAILABLE, reason="networkx not installed")
@pytest.mark.parametrize(
    ("graph_factory", "graph_name"),
    [
        (_make_diamond_dag, "diamond"),
        (_make_connected_random_graph, "connected_random"),
    ],
)
def test_spectral_vs_networkx(
    graph_factory: Callable[[], tuple[torch.Tensor, int] | tuple[torch.Tensor, int, Any]],
    graph_name: str,
) -> None:
    """Spectral layouts should match NetworkX up to rigid-motion invariants."""
    nx = pytest.importorskip("networkx")
    graph_parts = graph_factory()
    edge_index, num_nodes = graph_parts[:2]

    graph = _make_networkx_graph(edge_index, num_nodes)
    reference_mapping = nx.spectral_layout(graph, dim=2)
    reference_positions = torch.tensor(
        np.asarray([reference_mapping[node] for node in range(num_nodes)], dtype=np.float32)
    )
    our_positions = layout_spectral(edge_index, num_nodes, seed=42)

    reference_distances = _upper_triangle_values(
        _pairwise_distances(_normalize_positions(reference_positions))
    )
    our_distances = _upper_triangle_values(_pairwise_distances(_normalize_positions(our_positions)))
    correlation = _pearson_correlation(reference_distances, our_distances)
    assert correlation > 0.85, graph_name
    assert torch.linalg.norm(our_positions.mean(dim=0)) < 1.0e-4
    assert float(our_positions.abs().max().item()) <= 1.0 + 1.0e-5


@pytest.mark.skipif(not (_NETWORKX_AVAILABLE and _SCIPY_AVAILABLE), reason="networkx/scipy missing")
@pytest.mark.parametrize(
    ("name", "nx_fn", "our_fn", "kwargs"),
    [
        (
            "FR",
            lambda graph: pytest.importorskip("networkx").spring_layout(
                graph,
                seed=42,
                iterations=50,
                scale=1,
            ),
            layout_fr,
            {"seed": 42},
        ),
        (
            "KK",
            lambda graph: pytest.importorskip("networkx").kamada_kawai_layout(graph, scale=1),
            layout_kk,
            {"seed": 42},
        ),
        (
            "Spectral",
            lambda graph: pytest.importorskip("networkx").spectral_layout(graph, scale=1),
            layout_spectral,
            {"seed": 42},
        ),
    ],
)
def test_classic_layouts_match_networkx_procrustes(
    name: str,
    nx_fn: Callable[[Any], Any],
    our_fn: Callable[..., torch.Tensor],
    kwargs: dict[str, int],
) -> None:
    """FR, KK, and Spectral should match NetworkX on Karate Club."""
    edge_index, num_nodes, graph = _make_karate_edge_index()

    reference_mapping = nx_fn(graph)
    reference_positions = np.asarray(
        [reference_mapping[node] for node in range(num_nodes)],
        dtype=np.float64,
    )
    our_positions = our_fn(edge_index, num_nodes, **kwargs).cpu().numpy().astype(np.float64)

    disparity = _procrustes_disparity(
        reference_positions,
        _normalize_for_procrustes(our_positions),
    )
    assert disparity < 0.01, f"{name} disparity {disparity:.6f} exceeded 0.01"


@pytest.mark.skipif(
    not (_SCIPY_AVAILABLE and _SKLEARN_AVAILABLE),
    reason="scipy/sklearn not installed",
)
@pytest.mark.parametrize(
    ("graph_factory", "graph_name"),
    [
        (_make_layered_dag, "layered_dag"),
        (_make_grid_graph, "grid"),
    ],
)
def test_pivot_mds_vs_sklearn(
    graph_factory: Callable[[], tuple[torch.Tensor, int] | tuple[torch.Tensor, int, Any]],
    graph_name: str,
) -> None:
    """Pivot MDS should preserve pairwise structure like metric MDS."""
    scipy_csgraph = pytest.importorskip("scipy.sparse.csgraph")
    graph_parts = graph_factory()
    edge_index, num_nodes = graph_parts[:2]

    adjacency = _make_sparse_adjacency(edge_index, num_nodes)
    distance_matrix = scipy_csgraph.shortest_path(adjacency, directed=False, unweighted=True)
    reference_positions = _run_sklearn_mds(distance_matrix, seed=42)
    our_positions = layout_pivot_mds(edge_index, num_nodes, n_pivots=min(8, num_nodes), seed=42)

    reference_distances = _upper_triangle_values(
        _pairwise_distances(_normalize_positions(reference_positions))
    )
    our_distances = _upper_triangle_values(_pairwise_distances(_normalize_positions(our_positions)))
    correlation = _pearson_correlation(reference_distances, our_distances)
    assert correlation > 0.8, graph_name


@pytest.mark.skipif(not _IGRAPH_AVAILABLE, reason="igraph not installed")
def test_gem_vs_igraph() -> None:
    """GEM should match igraph's GEM distribution when that reference exists."""
    if not _IGRAPH_GEM_AVAILABLE:
        pytest.skip("igraph GEM reference layout is not available in this environment")

    edge_index, num_nodes = _make_connected_random_graph()
    graph = _make_igraph_graph(edge_index, num_nodes)
    seeds = [0, 1, 2, 3, 4]

    our_metrics = [
        _layout_metric_summary(
            layout_gem(edge_index, num_nodes, max_iters=300, seed=seed),
            edge_index,
            num_nodes,
        )
        for seed in seeds
    ]
    reference_metrics = [
        _layout_metric_summary(
            _run_igraph_layout(graph, "gem", seed),
            edge_index,
            num_nodes,
        )
        for seed in seeds
    ]

    _assert_metric_distribution_overlap(our_metrics, reference_metrics, "stress")
    _assert_metric_distribution_overlap(our_metrics, reference_metrics, "crossing_count")
    _assert_metric_distribution_overlap(our_metrics, reference_metrics, "edge_length_cv")


@pytest.mark.skipif(not _IGRAPH_AVAILABLE, reason="igraph not installed")
def test_davidson_harel_vs_igraph() -> None:
    """Davidson-Harel should have a similar stochastic quality profile to igraph."""
    edge_index, num_nodes = _make_small_dh_graph()
    graph = _make_igraph_graph(edge_index, num_nodes)
    seeds = [0, 1, 2, 3, 4]

    our_metrics = [
        _layout_metric_summary(
            layout_davidson_harel(edge_index, num_nodes, rounds=60, seed=seed),
            edge_index,
            num_nodes,
        )
        for seed in seeds
    ]
    reference_metrics = [
        _layout_metric_summary(
            _run_igraph_layout(graph, "davidson_harel", seed),
            edge_index,
            num_nodes,
        )
        for seed in seeds
    ]

    _assert_metric_distribution_overlap(our_metrics, reference_metrics, "stress")
    _assert_metric_distribution_overlap(our_metrics, reference_metrics, "crossing_count")
    _assert_metric_distribution_overlap(
        our_metrics,
        reference_metrics,
        "edge_length_cv",
        tolerance_multiplier=2.0,
    )


@pytest.mark.parametrize(
    ("graph_factory", "graph_name"),
    [
        (_make_connected_random_graph, "connected_random"),
        (_make_grid_graph, "grid"),
    ],
)
def test_linlog_reduces_energy(
    graph_factory: Callable[[], tuple[torch.Tensor, int]],
    graph_name: str,
) -> None:
    """LinLog should reduce its own energy relative to the seeded initial layout."""
    edge_index, num_nodes = graph_factory()
    seed = 42
    checkpoints = [0, 25, 100, 250]
    energies: list[float] = []

    for steps in checkpoints:
        pos = layout_linlog(edge_index, num_nodes, steps=steps, seed=seed)
        energies.append(float(_linlog_loss(pos, edge_index, seed, 0).item()))

    assert min(energies[1:]) < energies[0] * 0.8, graph_name
    assert energies[-1] < energies[0], graph_name


def test_linlog_non_edge_repulsion_separates_bridge_connected_communities() -> None:
    """Non-edge repulsion should separate weakly bridged cliques more strongly."""
    edge_index, num_nodes, labels = _make_two_cliques_with_bridge()
    seed = 0

    updated_positions = layout_linlog(edge_index, num_nodes, steps=120, seed=seed)
    baseline_positions = _layout_linlog_all_pairs_baseline(edge_index, num_nodes, 120, seed)

    updated_ratio = _community_separation_ratio(updated_positions, labels)
    baseline_ratio = _community_separation_ratio(baseline_positions, labels)

    assert updated_ratio > baseline_ratio * 5.0


@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="sklearn not installed")
@pytest.mark.parametrize(
    ("graph_factory", "graph_name"),
    [
        (_make_connected_random_graph, "connected_random"),
        (_make_grid_graph, "grid"),
    ],
)
def test_tsnet_vs_sklearn(
    graph_factory: Callable[[], tuple[torch.Tensor, int]],
    graph_name: str,
) -> None:
    """tsNET should preserve graph neighborhoods comparably to sklearn t-SNE."""
    scipy_csgraph = pytest.importorskip("scipy.sparse.csgraph")
    edge_index, num_nodes = graph_factory()
    adjacency = _make_sparse_adjacency(edge_index, num_nodes)
    distance_matrix_np = scipy_csgraph.shortest_path(adjacency, directed=False, unweighted=True)
    graph_distances = torch.tensor(distance_matrix_np, dtype=torch.float32)
    perplexity = min(5.0, float(max(num_nodes - 1, 1)))

    reference_positions = _run_sklearn_tsne(distance_matrix_np, seed=42, perplexity=perplexity)
    our_positions = layout_tsnet(edge_index, num_nodes, perplexity=perplexity, steps=500, seed=42)

    reference_score = _graph_knn_preservation(reference_positions, graph_distances, k=4)
    our_score = _graph_knn_preservation(our_positions, graph_distances, k=4)

    assert our_score >= reference_score - 0.15, graph_name
    assert our_score > 0.55, graph_name


@pytest.mark.skipif(
    not (_SCIPY_AVAILABLE and _SKLEARN_AVAILABLE),
    reason="scipy/sklearn not installed",
)
def test_tsnet_matches_sklearn_within_between_distance_distributions() -> None:
    """tsNET should match sklearn's near/far distance statistics across seeds."""
    scipy_csgraph = pytest.importorskip("scipy.sparse.csgraph")
    edge_index, num_nodes = _make_connected_random_graph()
    adjacency = _make_sparse_adjacency(edge_index, num_nodes)
    graph_distances = torch.tensor(
        scipy_csgraph.shortest_path(adjacency, directed=False, unweighted=True),
        dtype=torch.float32,
    )
    perplexity = min(5.0, float(max(num_nodes - 1, 1)))

    reference_stats: list[tuple[float, float, float]] = []
    our_stats: list[tuple[float, float, float]] = []
    for seed in range(10):
        reference_positions = _run_sklearn_tsne(
            graph_distances.numpy(),
            seed=seed,
            perplexity=perplexity,
        )
        our_positions = layout_tsnet(
            edge_index,
            num_nodes,
            perplexity=perplexity,
            steps=500,
            seed=seed,
        )
        reference_stats.append(
            _within_between_distance_stats(
                reference_positions,
                graph_distances,
                within_radius=2.0,
                between_radius=3.0,
            )
        )
        our_stats.append(
            _within_between_distance_stats(
                our_positions,
                graph_distances,
                within_radius=2.0,
                between_radius=3.0,
            )
        )

    for index, tolerance in enumerate((0.05, 0.08)):
        reference_mean = mean(stats[index] for stats in reference_stats)
        our_mean = mean(stats[index] for stats in our_stats)
        assert abs(reference_mean - our_mean) < tolerance


@pytest.mark.skipif(not _IGRAPH_AVAILABLE, reason="igraph not installed")
@pytest.mark.parametrize(
    ("graph_factory", "graph_name"),
    [
        (_make_diamond_dag, "diamond"),
        (_make_layered_dag, "layered"),
        (
            lambda: (
                torch.tensor(
                    [[0, 0, 0, 0, 1, 2, 3, 4], [1, 2, 3, 4, 5, 5, 5, 5]],
                    dtype=torch.long,
                ),
                6,
                None,
            ),
            "fan",
        ),
    ],
)
def test_sugiyama_vs_igraph_procrustes(
    graph_factory: Callable[[], tuple[torch.Tensor, int] | tuple[torch.Tensor, int, Any]],
    graph_name: str,
) -> None:
    """Sugiyama should stay within a small Procrustes gap to igraph."""
    graph_parts = graph_factory()
    edge_index, num_nodes = graph_parts[:2]
    graph = _make_igraph_digraph(edge_index, num_nodes)

    reference_positions = np.asarray(list(graph.layout_sugiyama()), dtype=np.float64)
    our_positions = layout_sugiyama(edge_index, num_nodes).cpu().numpy().astype(np.float64)

    disparity = _procrustes_disparity(
        _normalize_for_procrustes(reference_positions),
        _normalize_for_procrustes(our_positions),
    )
    assert disparity < 0.05, f"{graph_name} disparity {disparity:.6f} exceeded 0.05"


@pytest.mark.parametrize(
    ("graph_factory", "graph_name"),
    [
        (_make_connected_random_graph, "connected_random"),
        (_make_grid_graph, "grid"),
        (lambda: _make_petersen_graph()[:2], "petersen"),
    ],
)
def test_maxent_stress_no_collapse(
    graph_factory: Callable[[], tuple[torch.Tensor, int]],
    graph_name: str,
) -> None:
    """Pure stress mode should stay non-collapsed and competitive with Stress-SGD."""
    edge_index, num_nodes = graph_factory()

    maxent_positions = layout_maxent_stress(
        edge_index,
        num_nodes,
        steps=180,
        seed=42,
        use_entropy=False,
    )
    stress_sgd_positions = layout_stress_sgd(edge_index, num_nodes, steps=220, seed=42)

    maxent_stress = _exact_stress(maxent_positions, edge_index, num_nodes)
    stress_sgd_stress = _exact_stress(stress_sgd_positions, edge_index, num_nodes)
    maxent_edge_cv = float(
        edge_length_cv(_normalize_positions(maxent_positions), edge_index)["edge_length_cv"]
    )

    assert _min_pairwise_distance(maxent_positions) > 1.0e-4, graph_name
    assert maxent_stress <= stress_sgd_stress * 1.35, graph_name
    assert maxent_edge_cv < 0.6, graph_name


@pytest.mark.skipif(not _SCIPY_AVAILABLE, reason="scipy not installed")
@pytest.mark.parametrize(
    ("algorithm", "layout_fn", "kwargs", "threshold"),
    [
        ("gem", layout_gem, {"max_iters": 300, "seed": 42}, 0.35),
        ("fmmm", layout_fmmm, {"steps": 120, "seed": 42}, 0.15),
        (
            "stress",
            layout_maxent_stress,
            {"steps": 220, "seed": 42, "use_entropy": False},
            0.20,
        ),
    ],
)
def test_classic_layouts_track_ogdf_runner_on_task_graph(
    algorithm: str,
    layout_fn: Callable[..., torch.Tensor],
    kwargs: dict[str, Any],
    threshold: float,
) -> None:
    """The patched classic layouts should stay close to OGDF on the task graph."""
    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [8, 9],
        [0, 5],
        [2, 7],
    ]
    edge_index = torch.tensor(edges, dtype=torch.long).transpose(0, 1).contiguous()

    reference_positions = _ogdf_layout_reference(10, edges, algorithm)
    our_positions = layout_fn(edge_index, 10, **kwargs).cpu().numpy().astype(np.float64)

    disparity = _procrustes_disparity(reference_positions, our_positions)
    assert disparity <= threshold, f"{algorithm} disparity {disparity:.6f} exceeded {threshold:.2f}"


def test_fmmm_coarsening_preserves_and_averages_parallel_edge_lengths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FM^3 coarsening should preserve OGDF-style coarse edge lengths."""
    fmmm_module = importlib.import_module("dagua.layout.classic.fmmm")

    chosen_suns = iter([0, 3])

    def _pick_forced_sun(
        self: Any,
        rng: random.Random,
        random_tries: int,
    ) -> int:
        """Select predetermined suns to make the coarsening structure deterministic."""
        del rng, random_tries
        sun = next(chosen_suns)
        self.delete(sun)
        return sun

    monkeypatch.setattr(
        fmmm_module._RandomNodeSet,
        "get_random_node_with_highest_star_mass",
        _pick_forced_sun,
    )

    edge_index = (
        torch.tensor(
            [[0, 1], [1, 2], [3, 4], [4, 5], [1, 4], [2, 3]],
            dtype=torch.long,
        )
        .transpose(0, 1)
        .contiguous()
    )
    level_graph = fmmm_module._LevelGraph(
        edge_index=edge_index,
        edge_lengths=torch.ones((edge_index.shape[1],), dtype=torch.float32),
        num_nodes=6,
    )

    step, coarse_level, coarse_masses = fmmm_module._coarsen_level(
        level_graph,
        torch.ones((6,), dtype=torch.long),
        random.Random(42),
    )

    assert coarse_level.num_nodes == 2
    torch.testing.assert_close(coarse_level.edge_lengths, torch.tensor([3.0], dtype=torch.float32))
    assert int(coarse_masses.sum().item()) == 6
    assert step.lambda_values[1]
    assert step.lambda_values[2]


def test_fmmm_prolongation_uses_lambda_interpolation_with_waggle() -> None:
    """FM^3 prolongation should stay within OGDF's 5% waggle radius."""
    fmmm_module = importlib.import_module("dagua.layout.classic.fmmm")

    coarse_positions = torch.tensor([[0.0, 0.0], [10.0, 0.0]], dtype=torch.float32)
    step = fmmm_module._HierarchyStep(
        mapping=torch.tensor([0, 0, 1], dtype=torch.long),
        node_types=[fmmm_module._TYPE_SUN, fmmm_module._TYPE_PLANET, fmmm_module._TYPE_SUN],
        dedicated_sun=[0, 0, 2],
        dedicated_sun_distance=[0.0, 2.5, 0.0],
        pm_nodes=[],
        moon_children=[[], [], []],
        lambda_values=[[], [0.25], []],
        neighbor_suns=[[], [2], []],
    )

    fine_positions = fmmm_module._prolong_positions(coarse_positions, step, random.Random(7))
    interpolated = torch.tensor([2.5, 0.0], dtype=torch.float32)

    torch.testing.assert_close(fine_positions[0], coarse_positions[0])
    torch.testing.assert_close(fine_positions[2], coarse_positions[1])
    assert torch.linalg.norm(fine_positions[1] - interpolated) <= 0.5 + 1.0e-6


@pytest.mark.parametrize(
    ("graph_factory", "graph_name"),
    [
        (_make_connected_random_graph, "connected_random"),
        (_make_grid_graph, "grid"),
        (lambda: _make_petersen_graph()[:2], "petersen"),
    ],
)
def test_fmmm_produces_reasonable_layout(
    graph_factory: Callable[[], tuple[torch.Tensor, int]],
    graph_name: str,
) -> None:
    """FM^3 should remain competitive with FR and KK on small reference graphs."""
    edge_index, num_nodes = graph_factory()

    fmmm_positions = layout_fmmm(edge_index, num_nodes, steps=80, seed=42)
    fr_positions = layout_fr(edge_index, num_nodes, steps=150, seed=42)
    kk_positions = layout_kk(edge_index, num_nodes, steps=200, seed=42)

    fmmm_normalized = _normalize_positions(fmmm_positions)
    fr_normalized = _normalize_positions(fr_positions)
    kk_normalized = _normalize_positions(kk_positions)

    fmmm_stress = _exact_stress(fmmm_normalized, edge_index, num_nodes)
    fr_stress = _exact_stress(fr_normalized, edge_index, num_nodes)
    kk_stress = _exact_stress(kk_normalized, edge_index, num_nodes)

    fmmm_crossings = count_crossings(fmmm_normalized, edge_index)
    fr_crossings = count_crossings(fr_normalized, edge_index)

    fmmm_edge_cv = float(edge_length_cv(fmmm_normalized, edge_index)["edge_length_cv"])
    baseline_edge_cv = max(
        float(edge_length_cv(fr_normalized, edge_index)["edge_length_cv"]),
        float(edge_length_cv(kk_normalized, edge_index)["edge_length_cv"]),
    )

    assert _min_pairwise_distance(fmmm_positions) > 1.0e-4, graph_name
    # Exact NetworkX KK is a stronger small-graph baseline than the previous
    # approximate path, so keep this as a competitiveness check rather than
    # requiring FM^3 to nearly match a direct KK solve.
    assert fmmm_stress <= min(fr_stress, kk_stress) * 1.3, graph_name
    assert fmmm_crossings <= max(2 * fr_crossings, 2), graph_name
    assert fmmm_edge_cv <= baseline_edge_cv + 0.1, graph_name


def test_layout_gem_edge_weights_change_the_layout() -> None:
    """GEM should respond to weighted attraction changes."""
    edge_index = torch.tensor([[0, 1, 2, 1], [1, 2, 3, 4]], dtype=torch.long)
    uniform_weights = torch.ones(edge_index.shape[1], dtype=torch.float32)
    weighted = uniform_weights.clone()
    weighted[1] = 8.0

    uniform_pos = layout_gem(edge_index, 5, max_iters=120, seed=7, edge_weights=uniform_weights)
    weighted_pos = layout_gem(edge_index, 5, max_iters=120, seed=7, edge_weights=weighted)

    assert not torch.allclose(uniform_pos, weighted_pos)


def test_layout_gem_rejects_mismatched_edge_weights() -> None:
    """GEM should reject edge-weight tensors that do not match ``E``."""
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

    with pytest.raises(ValueError, match="edge_weights length"):
        layout_gem(edge_index, 4, edge_weights=torch.ones(2, dtype=torch.float32))


def test_layout_davidson_harel_edge_weights_change_the_layout() -> None:
    """Davidson-Harel should weight the edge-length energy term."""
    edge_index = torch.tensor([[0, 1, 2, 1], [1, 2, 3, 4]], dtype=torch.long)
    uniform_weights = torch.ones(edge_index.shape[1], dtype=torch.float32)
    weighted = uniform_weights.clone()
    weighted[1] = 8.0

    uniform_pos = layout_davidson_harel(
        edge_index,
        5,
        rounds=40,
        seed=13,
        edge_weights=uniform_weights,
    )
    weighted_pos = layout_davidson_harel(
        edge_index,
        5,
        rounds=40,
        seed=13,
        edge_weights=weighted,
    )

    assert not torch.allclose(uniform_pos, weighted_pos)


def test_layout_davidson_harel_rejects_mismatched_edge_weights() -> None:
    """Davidson-Harel should reject edge-weight tensors that do not match ``E``."""
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

    with pytest.raises(ValueError, match="edge_weights length"):
        layout_davidson_harel(edge_index, 4, edge_weights=torch.ones(2, dtype=torch.float32))


def test_layout_fmmm_edge_weights_change_the_layout() -> None:
    """FM^3 should scale its spring forces when weights are supplied."""
    edge_index = torch.tensor([[0, 1, 2, 1], [1, 2, 3, 4]], dtype=torch.long)
    uniform_weights = torch.ones(edge_index.shape[1], dtype=torch.float32)
    weighted = uniform_weights.clone()
    weighted[1] = 8.0

    uniform_pos = layout_fmmm(edge_index, 5, steps=80, seed=5, edge_weights=uniform_weights)
    weighted_pos = layout_fmmm(edge_index, 5, steps=80, seed=5, edge_weights=weighted)

    assert not torch.allclose(uniform_pos, weighted_pos)


def test_layout_fmmm_rejects_mismatched_edge_weights() -> None:
    """FM^3 should reject edge-weight tensors that do not match ``E``."""
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

    with pytest.raises(ValueError, match="edge_weights length"):
        layout_fmmm(edge_index, 4, edge_weights=torch.ones(2, dtype=torch.float32))
