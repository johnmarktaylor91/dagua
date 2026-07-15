"""Tests for the LargeVis pipeline."""

from __future__ import annotations

import inspect
from pathlib import Path

import torch

import dagua
from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.layout.ops.largevis import LargeVisGraph, symmetrize_largevis_similarity
from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.largevis import (
    build_largevis_pipeline,
    layout_largevis_pipeline,
)
from dagua.layout.ops.taxonomy import get_op_class


def _edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    """Build an edge tensor from pairs.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Edge pairs.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def test_largevis_pipeline_and_ops_are_registered() -> None:
    """LargeVis should be discoverable through pipeline and op registries.

    Returns
    -------
    None
        Registry lookups must resolve.
    """
    assert PIPELINE_REGISTRY["largevis"] == (
        "dagua.layout.ops.pipelines.largevis",
        "layout_largevis_pipeline",
    )
    assert get_pipeline_function("LargeVis") is layout_largevis_pipeline
    assert get_op_class("largevis_build_similarity").__name__ == "LargeVisBuildSimilarity"
    assert get_op_class("largevis_optimize_embedding").__name__ == "LargeVisOptimizeEmbedding"


def test_largevis_pipeline_has_shared_stage_composition() -> None:
    """LargeVis should expose its shared operation stages.

    Returns
    -------
    None
        Stage names must remain stable.
    """
    pipeline = build_largevis_pipeline(samples=3, seed=5)
    assert [operation.name for operation in pipeline.ops] == [
        "largevis_build_similarity",
        "largevis_optimize_embedding",
    ]


def test_largevis_similarity_symmetrizes_missing_reverse_edges() -> None:
    """LargeVis similarity should add reverse edges and average weights.

    Returns
    -------
    None
        The output graph should contain reciprocal directed edges.
    """
    graph = LargeVisGraph(
        source=torch.tensor([0], dtype=torch.long).numpy(),
        target=torch.tensor([1], dtype=torch.long).numpy(),
        weight=torch.tensor([1.0], dtype=torch.float32).numpy(),
        num_nodes=2,
    )
    similarity = symmetrize_largevis_similarity(graph, perplexity=2.0)

    assert similarity.source.tolist() == [0, 1]
    assert similarity.target.tolist() == [1, 0]
    assert torch.allclose(torch.from_numpy(similarity.weight), torch.tensor([0.5, 0.5]))


def test_largevis_is_seed_deterministic() -> None:
    """LargeVis should return identical coordinates for identical seeds.

    Returns
    -------
    None
        Repeated calls must match exactly.
    """
    edges = _edge_index([(0, 1), (1, 2), (2, 3), (3, 0)])
    first = layout_largevis_pipeline(edges, 4, samples=25, seed=11)
    second = layout_largevis_pipeline(edges, 4, samples=25, seed=11)

    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert first.shape == (4, 2)


def test_largevis_public_dispatch_returns_positions() -> None:
    """``dagua.layout`` should dispatch algorithm='largevis'.

    Returns
    -------
    None
        Dispatch must return finite coordinates.
    """
    graph = DaguaGraph.from_edge_list([(0, 1), (1, 2), (2, 3)])
    positions = dagua.layout(
        graph,
        LayoutConfig(
            algorithm="largevis",
            seed=7,
            algorithm_params={"samples": 20, "n_neighbors": 3},
        ),
    )

    assert positions.shape == (4, 2)
    assert torch.isfinite(positions).all()


def test_largevis_pipeline_has_no_runtime_delegation() -> None:
    """Production LargeVis code must not call reference binaries.

    Returns
    -------
    None
        Source should not contain subprocess/reference hooks.
    """
    root = Path(__file__).parents[1]
    source = "\n".join(
        [
            (root / "dagua/layout/ops/largevis.py").read_text(),
            (root / "dagua/layout/ops/pipelines/largevis.py").read_text(),
            inspect.getsource(layout_largevis_pipeline),
        ]
    )

    assert "subprocess" not in source
    assert "/tmp/LargeVis" not in source
    assert "LargeVis_run.py" not in source
