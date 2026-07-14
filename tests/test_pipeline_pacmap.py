"""Pipeline pins and PaCMAP-fidelity checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import dagua
from dagua.config import LayoutConfig
from dagua.eval.equivalence_metrics import procrustes_rmsd
from dagua.graph import DaguaGraph
from dagua.layout.engine import layout
from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.pacmap import (
    _decide_num_pairs,
    _fit_pacmap,
    _generate_pairs,
    _graph_geodesic_distances,
    layout_pacmap_pipeline,
)


def _graph_from_edges(num_nodes: int, edges: list[tuple[int, int]]) -> DaguaGraph:
    """Build a deterministic test graph.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
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


def test_pacmap_pipeline_is_registered() -> None:
    """Register the PaCMAP graph-distance pipeline.

    Returns
    -------
    None
        Registry lookup must resolve the public PaCMAP entrypoint.
    """
    assert PIPELINE_REGISTRY["pacmap"] == (
        "dagua.layout.ops.pipelines.pacmap",
        "layout_pacmap_pipeline",
    )
    assert get_pipeline_function("PaCMAP") is layout_pacmap_pipeline


def test_pacmap_pair_counts_follow_reference_small_sample_rules() -> None:
    """Pin PaCMAP's small-sample count reorganization.

    Returns
    -------
    None
        Pair counts must match the reference ``decide_num_pairs`` arithmetic.
    """
    assert _decide_num_pairs(8, 10, 0.5, 2.0) == (2, 1, 4)
    assert _decide_num_pairs(20, 10, 0.5, 2.0) == (5, 2, 11)


def test_pacmap_pair_sampling_matches_reference_rng() -> None:
    """Verify deterministic mid-near and further-pair RNG in isolation.

    Returns
    -------
    None
        Native pair sampling must match PaCMAP's global reseeding behavior for
        deterministic runs.
    """
    import pacmap.pacmap as reference

    features = np.arange(80, dtype=np.float32).reshape(8, 10) / 80.0
    n_neighbors, n_mn, n_fp = _decide_num_pairs(8, 10, 0.5, 2.0)
    actual_neighbors, actual_mn, actual_fp = _generate_pairs(features, n_neighbors, n_mn, n_fp, 7)

    reference._RANDOM_STATE = 7
    expected_mn = reference.sample_MN_pair_deterministic(features, n_mn, 7, 0)
    expected_fp = reference.sample_FP_pair_deterministic(
        features,
        actual_neighbors,
        n_neighbors,
        n_fp,
        7,
    )

    np.testing.assert_array_equal(actual_mn, expected_mn)
    np.testing.assert_array_equal(actual_fp, expected_fp)


def test_pacmap_optimizer_matches_reference_core() -> None:
    """Compare native optimization against the PaCMAP core function.

    Returns
    -------
    None
        With identical graph-distance features and sampled pairs, the final
        embedding should match the package's core optimizer within float32
        arithmetic noise.
    """
    import pacmap.pacmap as reference

    graph = _graph_from_edges(8, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (2, 6), (6, 7)])
    distances = _graph_geodesic_distances(graph.edge_index, graph.num_nodes, None)
    actual, pairs = _fit_pacmap(
        distances,
        init="pca",
        n_neighbors=10,
        mn_ratio=0.5,
        fp_ratio=2.0,
        lr=1.0,
        num_iters=(2, 2, 2),
        seed=7,
    )

    native_processed = distances.copy().astype(np.float32)
    native_processed -= np.min(native_processed)
    native_processed /= np.max(native_processed)
    native_processed -= np.mean(native_processed, axis=0)
    reference._RANDOM_STATE = 7
    expected, _, _, _, _ = reference.pacmap(
        native_processed,
        2,
        pairs[0],
        pairs[1],
        pairs[2],
        1.0,
        (2, 2, 2),
        "pca",
        False,
        False,
        [],
        False,
        reference.PCA(n_components=2, random_state=7).fit(native_processed),
    )

    np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-5)
    assert procrustes_rmsd(actual, expected) < 3.0e-6


def test_layout_config_algorithm_pacmap_works() -> None:
    """Exercise public engine dispatch for ``algorithm='pacmap'``.

    Returns
    -------
    None
        Dispatch must return finite ``[N, 2]`` coordinates.
    """
    graph = dagua.DaguaGraph.from_edge_list(
        [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f"), ("c", "g"), ("g", "h")]
    )
    positions = layout(
        graph,
        LayoutConfig(
            algorithm="pacmap",
            seed=7,
            algorithm_params={"num_iters": (1, 1, 1), "n_neighbors": 10},
        ),
    )

    assert positions.shape == (8, 2)
    assert torch.isfinite(positions).all()


def test_pacmap_production_pipeline_has_no_runtime_delegation() -> None:
    """Guard the production pipeline against PaCMAP-package delegation.

    Returns
    -------
    None
        Production source must not import or construct the reference package.
    """
    source_path = Path(__file__).parents[1] / "dagua" / "layout" / "ops" / "pipelines" / "pacmap.py"
    source = source_path.read_text()
    assert "from pacmap" not in source
    assert "import pacmap" not in source
    assert "PaCMAP(" not in source
