"""Pipeline pins and reference checks for sparse stress."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest
import torch

import dagua
from dagua.config import LayoutConfig
from dagua.eval.competitors.sparse_stress_competitor import SparseStressCompetitor
from dagua.graph import DaguaGraph
from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.sparse_stress import (
    SparseStressConfig,
    _build_sparse_stress_graph,
    _sample_pivots,
    build_sparse_stress_pipeline,
    layout_sparse_stress_pipeline,
)
from dagua.layout.ops.taxonomy import get_op_class


def _graph_from_edges(num_nodes: int, edges: list[tuple[int, int]]) -> DaguaGraph:
    """Build a deterministic test graph.

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


def test_sparse_stress_pipeline_and_ops_are_registered() -> None:
    """Register public sparse-stress algorithm and stage ops.

    Returns
    -------
    None
        Registry lookups must resolve sparse-stress entries.
    """
    assert PIPELINE_REGISTRY["sparse_stress"] == (
        "dagua.layout.ops.pipelines.sparse_stress",
        "layout_sparse_stress_pipeline",
    )
    assert get_pipeline_function("SPARSE_STRESS") is layout_sparse_stress_pipeline
    assert get_op_class("sparse_stress_prepare_graph").__name__ == "PrepareSparseStressGraph"
    assert get_op_class("sparse_stress_initialize").__name__ == "InitializeSparseStressPositions"
    assert get_op_class("sparse_stress_terms").__name__ == "BuildSparseStressTerms"
    assert get_op_class("sparse_stress_majorization").__name__ == "RunSparseStressMajorization"


def test_sparse_stress_pipeline_has_stage_composition() -> None:
    """Pin sparse-stress as an explicit op composition.

    Returns
    -------
    None
        The top-level stage names must remain visible.
    """
    pipeline = build_sparse_stress_pipeline(SparseStressConfig(pivots=2, steps=2, sampler="random"))
    assert [operation.name for operation in pipeline.ops] == [
        "sparse_stress_prepare_graph",
        "sparse_stress_initialize",
        "sparse_stress_terms",
        "sparse_stress_majorization",
    ]


@pytest.mark.parametrize(
    ("sampler", "expected"),
    [
        ("random", [7, 8, 5, 3]),
        ("maxmin", [6, 0, 3, 9]),
        ("kmeans", [1, 8, 5, 3]),
    ],
)
def test_sparse_stress_sampler_pins(sampler: str, expected: list[int]) -> None:
    """Sampler outputs should stay pinned for a fixed seed.

    Parameters
    ----------
    sampler : str
        Sampler name.
    expected : list[int]
        Expected pivot ids before the reference sort step.

    Returns
    -------
    None
        Sampled pivots must match the pinned Java-RNG stream.
    """
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8, 9]])
    graph = _build_sparse_stress_graph(edge_index, 10, None, weighted=False)
    config = SparseStressConfig(
        pivots=4,
        sampler=sampler,
        seed=7,
        mds_pivots=3,
        kmeans_features=3,
    )

    assert _sample_pivots(graph, config) == expected


def test_sparse_stress_is_seed_deterministic() -> None:
    """Sparse stress should return identical positions for identical seeds.

    Returns
    -------
    None
        Repeated runs must match exactly.
    """
    graph = _graph_from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    first = layout_sparse_stress_pipeline(
        graph.edge_index,
        graph.num_nodes,
        seed=11,
        steps=8,
        pivots=3,
        sampler="random",
        mds_pivots=3,
    )
    second = layout_sparse_stress_pipeline(
        graph.edge_index,
        graph.num_nodes,
        seed=11,
        steps=8,
        pivots=3,
        sampler="random",
        mds_pivots=3,
    )
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert first.shape == (5, 2)


def test_sparse_stress_public_dispatch_returns_positions() -> None:
    """``dagua.layout`` should dispatch to the sparse-stress pipeline.

    Returns
    -------
    None
        LayoutConfig dispatch must return finite coordinates.
    """
    graph = _graph_from_edges(4, [(0, 1), (1, 2), (2, 3)])
    positions = dagua.layout(
        graph,
        LayoutConfig(
            algorithm="sparse_stress",
            steps=5,
            seed=7,
            algorithm_params={"pivots": 2, "sampler": "random", "mds_pivots": 2},
        ),
    )
    assert positions.shape == (4, 2)
    assert torch.isfinite(positions).all()


def test_sparse_stress_single_step_regression_pin() -> None:
    """Pin one small graph against current native sparse-stress output.

    Returns
    -------
    None
        Coordinates should remain bit-stable for this deterministic case.
    """
    graph = _graph_from_edges(4, [(0, 1), (1, 2), (2, 3)])
    actual = layout_sparse_stress_pipeline(
        graph.edge_index,
        graph.num_nodes,
        seed=7,
        steps=3,
        pivots=2,
        sampler="random",
        mds_pivots=2,
    )
    expected = torch.tensor(
        [
            [1.5000313520, -0.0003049528],
            [0.5000322461, 0.0002784448],
            [-0.4999640286, 0.0001230769],
            [-1.4999648333, -0.0000127525],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(actual, expected, atol=1.0e-6, rtol=1.0e-6)


def test_sparse_stress_reference_adapter_runs_when_available() -> None:
    """Run a small Java reference smoke if the built jar exists.

    Returns
    -------
    None
        The adapter should produce finite positions when available.
    """
    competitor = SparseStressCompetitor()
    if not competitor.available():
        pytest.skip("sparse-stress reference jar is not built")
    graph = _graph_from_edges(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
    result = competitor.layout_with_variant(
        graph,
        timeout=30.0,
        seed=7,
        variant_params={"pivots": 2, "steps": 3, "sampler": "random", "mds_pivots": 4},
    )
    assert result.error is None
    assert result.pos is not None
    assert result.pos.shape == (4, 2)
    assert torch.isfinite(result.pos).all()


def test_sparse_stress_production_pipeline_has_no_runtime_delegation() -> None:
    """Guard the production pipeline against reference delegation.

    Returns
    -------
    None
        Production source must not contain subprocess or reference hooks.
    """
    source_path = (
        Path(__file__).parents[1] / "dagua" / "layout" / "ops" / "pipelines" / "sparse_stress.py"
    )
    source = source_path.read_text()
    assert "subprocess" not in source
    assert "SparseStressCompetitor" not in source
    assert "/tmp/sparse-stress-ref" not in source
    assert "java -jar" not in source


def test_sparse_stress_competitor_is_not_imported_by_pipeline() -> None:
    """Pin pipeline source against eval-adapter imports.

    Returns
    -------
    None
        The native pipeline must not import the competitor adapter.
    """
    sparse_stress = importlib.import_module("dagua.layout.ops.pipelines.sparse_stress")

    source = inspect.getsource(sparse_stress)
    assert "competitors" not in source
