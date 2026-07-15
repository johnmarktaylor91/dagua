"""Regression tests for the fCoSE pipeline and competitor registration."""

from __future__ import annotations

import torch

from dagua.eval.competitors import get_competitor
from dagua.graph import DaguaGraph
from dagua.layout.ops.pipelines import get_pipeline_function
from dagua.layout.ops.pipelines.fcose import layout_fcose_pipeline


def test_fcose_pipeline_produces_finite_layout() -> None:
    """The fCoSE pipeline should return finite coordinates for a small graph."""
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]],
        dtype=torch.long,
    )
    pos = layout_fcose_pipeline(edge_index=edge_index, num_nodes=4, steps=25, seed=7)

    assert pos.shape == (4, 2)
    assert torch.isfinite(pos).all()
    assert float(pos.std().item()) > 0.0


def test_fcose_randomized_initial_placement_honors_seed() -> None:
    """Randomized spectral placement should be repeatable per seed and vary across seeds."""
    edge_index = torch.tensor(
        [[0, 0, 1, 2, 2, 3, 4, 5], [1, 2, 3, 3, 4, 5, 6, 7]],
        dtype=torch.long,
    )

    first = layout_fcose_pipeline(
        edge_index=edge_index,
        num_nodes=8,
        quality="draft",
        seed=100,
    )
    repeated = layout_fcose_pipeline(
        edge_index=edge_index,
        num_nodes=8,
        quality="draft",
        seed=100,
    )
    second = layout_fcose_pipeline(
        edge_index=edge_index,
        num_nodes=8,
        quality="draft",
        seed=101,
    )

    assert torch.equal(first, repeated)
    assert not torch.equal(first, second)


def test_fcose_pipeline_registry_entry() -> None:
    """The dynamic pipeline registry should resolve ``fcose``."""
    pipeline_function = get_pipeline_function("fcose")

    assert pipeline_function is layout_fcose_pipeline


def test_classic_fcose_competitor_produces_layout() -> None:
    """The benchmark competitor should run the fCoSE reimplementation."""
    graph = DaguaGraph()
    graph.add_node("a")
    graph.add_node("b")
    graph.add_node("c")
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")
    competitor = get_competitor("classic_fcose")
    assert competitor is not None

    result = competitor.layout_with_variant(graph, seed=11, variant_params={"steps": 20})

    assert result.error is None
    assert result.pos is not None
    assert result.pos.shape == (3, 2)
    assert torch.isfinite(result.pos).all()
