"""Pipeline pins and sklearn-fidelity checks for graph geodesic t-SNE."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from dagua.config import LayoutConfig
from dagua.eval.competitors.tsne_competitor import TSNEGraph, _distance_matrix
from dagua.eval.equivalence_metrics import procrustes_rmsd
from dagua.graph import DaguaGraph
from dagua.layout.engine import layout
from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.tsne_graph import (
    _graph_geodesic_distances,
    _joint_probabilities,
    layout_tsne_graph_pipeline,
    layout_tsne_pipeline,
)


def _graph_from_edges(num_nodes: int, edges: list[tuple[int, int]]) -> DaguaGraph:
    """Build a deterministic graph for t-SNE checks.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    edges : list[tuple[int, int]]
        Edge pairs.

    Returns
    -------
    DaguaGraph
        Graph with nodes ``0..N-1``.
    """
    graph = DaguaGraph.from_edge_list(edges, num_nodes=num_nodes)
    graph.compute_node_sizes()
    return graph


def test_tsne_pipeline_is_registered_separately_from_tsnet() -> None:
    """Register graph t-SNE without disturbing tsNET.

    Returns
    -------
    None
        Registry lookups must resolve the new t-SNE entries and keep tsNET on
        its existing module.
    """
    assert PIPELINE_REGISTRY["tsne"] == (
        "dagua.layout.ops.pipelines.tsne_graph",
        "layout_tsne_pipeline",
    )
    assert PIPELINE_REGISTRY["tsne_graph"] == (
        "dagua.layout.ops.pipelines.tsne_graph",
        "layout_tsne_graph_pipeline",
    )
    assert PIPELINE_REGISTRY["tsnet"] == (
        "dagua.layout.ops.pipelines.tsnet",
        "layout_tsnet_pipeline",
    )
    assert get_pipeline_function("TSNE") is layout_tsne_pipeline
    assert get_pipeline_function("tsne_graph") is layout_tsne_graph_pipeline


def test_tsne_geodesic_distances_match_competitor_adapter() -> None:
    """Mirror the reference adapter's APSP and disconnected-pair fill.

    Returns
    -------
    None
        Dense graph-geodesic distances must match the sklearn competitor's
        private adapter helper.
    """
    graph = _graph_from_edges(6, [(0, 1), (1, 2), (3, 4)])
    actual = _graph_geodesic_distances(graph.edge_index, graph.num_nodes, graph.edge_weights)
    expected = _distance_matrix(graph)
    np.testing.assert_array_equal(actual, expected)


def test_tsne_joint_probability_matrix_matches_sklearn_exact() -> None:
    """Pin the exact-method P-matrix before optimization.

    Returns
    -------
    None
        Local perplexity search and symmetrization must match sklearn's exact
        private helper for a precomputed distance matrix.
    """
    from sklearn.manifold._t_sne import _joint_probabilities as sklearn_joint_probabilities

    graph = _graph_from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    distances = _graph_geodesic_distances(graph.edge_index, graph.num_nodes, None)
    squared_distances = distances.copy()
    squared_distances **= 2

    actual = _joint_probabilities(squared_distances, desired_perplexity=3.0)
    expected = sklearn_joint_probabilities(squared_distances, desired_perplexity=3.0, verbose=0)

    np.testing.assert_array_equal(actual, expected)


def test_tsne_pipeline_matches_sklearn_exact_reference() -> None:
    """Compare the pipeline against sklearn's exact graph-distance t-SNE.

    Returns
    -------
    None
        The small deterministic exact-method embedding should match sklearn
        bit-for-bit for the pinned reference configuration.
    """
    graph = _graph_from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    reference = TSNEGraph().layout_with_variant(
        graph,
        seed=7,
        variant_params={"perplexity": 3.0, "max_iter": 250, "learning_rate": "auto"},
    )
    assert reference.error is None
    assert reference.pos is not None

    actual = layout_tsne_graph_pipeline(
        graph.edge_index,
        graph.num_nodes,
        perplexity=3.0,
        max_iter=250,
        seed=7,
    )

    torch.testing.assert_close(actual, reference.pos, rtol=0.0, atol=0.0)
    assert procrustes_rmsd(actual.numpy(), reference.pos.numpy()) < 1.0e-12


def test_layout_config_algorithm_tsne_works() -> None:
    """Exercise public engine dispatch for graph t-SNE.

    Returns
    -------
    None
        ``LayoutConfig(algorithm="tsne")`` must return finite ``[N, 2]``
        positions and accept sklearn-style variant parameters.
    """
    graph = _graph_from_edges(4, [(0, 1), (1, 2), (2, 3)])
    positions = layout(
        graph,
        LayoutConfig(
            algorithm="tsne",
            seed=7,
            algorithm_params={"perplexity": 2.0, "max_iter": 250},
        ),
    )

    assert positions.shape == (4, 2)
    assert torch.isfinite(positions).all()


def test_tsne_production_pipeline_has_no_runtime_delegation() -> None:
    """Guard the production pipeline against sklearn/reference delegation.

    Returns
    -------
    None
        Production source must not import the sklearn estimator or competitor.
    """
    source_path = (
        Path(__file__).parents[1] / "dagua" / "layout" / "ops" / "pipelines" / ("tsne_graph.py")
    )
    source = source_path.read_text()
    assert "from sklearn.manifold import TSNE" not in source
    assert "sklearn.manifold.TSNE" not in source
    assert "TSNEGraph" not in source
