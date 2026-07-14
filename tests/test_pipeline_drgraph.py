"""Tests for the DRGraph pipeline."""

from __future__ import annotations

import inspect
from pathlib import Path

import torch

import dagua
from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.drgraph import build_drgraph_pipeline, layout_drgraph_pipeline
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


def test_drgraph_pipeline_and_ops_are_registered() -> None:
    """DRGraph should be discoverable through pipeline and op registries.

    Returns
    -------
    None
        Registry lookups must resolve.
    """
    assert PIPELINE_REGISTRY["drgraph"] == (
        "dagua.layout.ops.pipelines.drgraph",
        "layout_drgraph_pipeline",
    )
    assert get_pipeline_function("DRGraph") is layout_drgraph_pipeline
    assert get_op_class("drgraph_build_similarity").__name__ == "DRGraphBuildSimilarity"
    assert get_op_class("largevis_optimize_embedding").__name__ == "LargeVisOptimizeEmbedding"


def test_drgraph_pipeline_reuses_largevis_optimizer_stage() -> None:
    """DRGraph should reuse the shared LargeVis negative-sampling optimizer.

    Returns
    -------
    None
        Stage names must show shared optimizer reuse.
    """
    pipeline = build_drgraph_pipeline(samples=3, seed=5)
    assert [operation.name for operation in pipeline.ops] == [
        "drgraph_build_similarity",
        "largevis_optimize_embedding",
    ]


def test_drgraph_is_seed_deterministic() -> None:
    """DRGraph should return identical coordinates for identical seeds.

    Returns
    -------
    None
        Repeated calls must match exactly.
    """
    edges = _edge_index([(0, 1), (1, 2), (2, 3), (3, 0)])
    first = layout_drgraph_pipeline(edges, 4, samples=25, seed=11)
    second = layout_drgraph_pipeline(edges, 4, samples=25, seed=11)

    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert first.shape == (4, 2)


def test_drgraph_ab_mode_changes_layout() -> None:
    """DRGraph A/B mode should select a different force curve.

    Returns
    -------
    None
        A/B mode should produce finite but different coordinates.
    """
    edges = _edge_index([(0, 1), (1, 2), (2, 3), (3, 0)])
    fallback = layout_drgraph_pipeline(edges, 4, samples=40, seed=13)
    ab_mode = layout_drgraph_pipeline(edges, 4, samples=40, seed=13, a=2.0, b=1.0)

    assert torch.isfinite(ab_mode).all()
    assert not torch.equal(fallback, ab_mode)


def test_drgraph_public_dispatch_returns_positions() -> None:
    """``dagua.layout`` should dispatch algorithm='drgraph'.

    Returns
    -------
    None
        Dispatch must return finite coordinates.
    """
    graph = DaguaGraph.from_edge_list([(0, 1), (1, 2), (2, 3)])
    positions = dagua.layout(
        graph,
        LayoutConfig(
            algorithm="drgraph",
            seed=7,
            algorithm_params={"samples": 20, "a": 2.0, "b": 1.0},
        ),
    )

    assert positions.shape == (4, 2)
    assert torch.isfinite(positions).all()


def test_drgraph_pipeline_has_no_runtime_delegation() -> None:
    """Production DRGraph code must not call reference binaries.

    Returns
    -------
    None
        Source should not contain subprocess/reference hooks.
    """
    root = Path(__file__).parents[1]
    source = "\n".join(
        [
            (root / "dagua/layout/ops/largevis.py").read_text(),
            (root / "dagua/layout/ops/pipelines/drgraph.py").read_text(),
            inspect.getsource(layout_drgraph_pipeline),
        ]
    )

    assert "subprocess" not in source
    assert "/tmp/DRGraph" not in source
    assert "./Vis" not in source
