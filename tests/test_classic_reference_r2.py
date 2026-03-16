"""Reference comparisons for the second batch of classic layout algorithms."""

from __future__ import annotations

import importlib.util
import random
import warnings
from collections import deque
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
    estimator = sklearn_manifold.TSNE(
        n_components=2,
        metric="precomputed",
        init="random",
        perplexity=perplexity,
        random_state=seed,
        method="exact",
        max_iter=500,
    )
    positions = estimator.fit_transform(distance_matrix)
    return torch.tensor(positions, dtype=torch.float32)


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
    our_positions = layout_tsnet(edge_index, num_nodes, perplexity=perplexity, steps=250, seed=42)

    reference_score = _graph_knn_preservation(reference_positions, graph_distances, k=4)
    our_score = _graph_knn_preservation(our_positions, graph_distances, k=4)

    assert our_score >= reference_score - 0.15, graph_name
    assert our_score > 0.55, graph_name


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
    """Maxent-stress should stay non-collapsed and competitive with Stress-SGD."""
    edge_index, num_nodes = graph_factory()

    maxent_positions = layout_maxent_stress(edge_index, num_nodes, steps=180, seed=42)
    stress_sgd_positions = layout_stress_sgd(edge_index, num_nodes, steps=220, seed=42)

    maxent_stress = _exact_stress(maxent_positions, edge_index, num_nodes)
    stress_sgd_stress = _exact_stress(stress_sgd_positions, edge_index, num_nodes)
    maxent_edge_cv = float(
        edge_length_cv(_normalize_positions(maxent_positions), edge_index)["edge_length_cv"]
    )

    assert _min_pairwise_distance(maxent_positions) > 1.0e-4, graph_name
    assert maxent_stress <= stress_sgd_stress * 1.25, graph_name
    assert maxent_edge_cv < 0.6, graph_name


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
    assert fmmm_stress <= min(fr_stress, kk_stress) * 1.25, graph_name
    assert fmmm_crossings <= max(2 * fr_crossings, 2), graph_name
    assert fmmm_edge_cv <= baseline_edge_cv + 0.1, graph_name
