"""Pipeline pins and guards for the backbone layout."""

from __future__ import annotations

import importlib
import inspect

import pytest
import torch

import dagua
from dagua.config import LayoutConfig
from dagua.eval.competitors.backbone_competitor import BackboneCompetitor
from dagua.graph import DaguaGraph
from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.backbone import (
    BackboneConfig,
    backbone_edge_set,
    build_backbone_pipeline,
    layout_backbone_pipeline,
)
from dagua.layout.ops.taxonomy import get_op_class


def _edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    """Build an edge-index tensor.

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
    sources, targets = zip(*edges)
    return torch.tensor([list(sources), list(targets)], dtype=torch.long)


def _graph_from_edges(num_nodes: int, edges: list[tuple[int, int]]) -> DaguaGraph:
    """Build a deterministic graph.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    edges : list[tuple[int, int]]
        Edge pairs.

    Returns
    -------
    DaguaGraph
        Graph with computed node sizes.
    """
    graph = DaguaGraph.from_edge_list(edges, num_nodes=num_nodes)
    graph.compute_node_sizes()
    return graph


def test_backbone_pipeline_and_ops_are_registered() -> None:
    """Register the public backbone algorithm and stage ops.

    Returns
    -------
    None
        Registry lookups must resolve backbone entries.
    """
    assert PIPELINE_REGISTRY["backbone"] == (
        "dagua.layout.ops.pipelines.backbone",
        "layout_backbone_pipeline",
    )
    assert get_pipeline_function("BACKBONE") is layout_backbone_pipeline
    assert get_op_class("backbone_compute").__name__ == "ComputeBackbone"
    assert get_op_class("backbone_stress").__name__ == "RunBackboneStress"


def test_backbone_pipeline_has_stage_composition() -> None:
    """Pin backbone as an explicit two-stage pipeline.

    Returns
    -------
    None
        Stage names should remain visible for bisection.
    """
    pipeline = build_backbone_pipeline(BackboneConfig(keep=0.4, iterations=2))
    assert [operation.name for operation in pipeline.ops] == [
        "backbone_compute",
        "backbone_stress",
    ]


def test_backbone_edge_set_regression_pin() -> None:
    """Backbone sparsification should keep the pinned edge subset.

    Returns
    -------
    None
        The deterministic edge scoring and UMST union must match the pin.
    """
    edge_index = _edge_index([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    edges, weights, mask = backbone_edge_set(edge_index, num_nodes=4, keep=0.4)

    assert edges == [(0, 1), (1, 2), (2, 3), (0, 3)]
    assert weights.tolist() == pytest.approx([0.5, 0.5, 0.5, 0.5, 0.0])
    assert mask.tolist() == [True, True, True, True, False]


def test_backbone_layout_regression_pin() -> None:
    """Pin one small graph against current native backbone output.

    Returns
    -------
    None
        Coordinates should remain stable for deterministic inputs.
    """
    edge_index = _edge_index([(0, 1), (1, 2), (2, 0), (2, 3)])
    actual = layout_backbone_pipeline(edge_index, num_nodes=4, keep=0.4, iterations=5)
    expected = torch.tensor(
        [
            [1.8158175945, 0.2443312109],
            [0.7215125561, 1.7231062651],
            [0.8334040046, 0.6640670896],
            [0.0000000000, 0.0000000000],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(actual, expected, atol=1.0e-6, rtol=1.0e-6)


def test_backbone_is_seed_deterministic() -> None:
    """Backbone should return identical positions for identical inputs.

    Returns
    -------
    None
        Repeated runs must match exactly.
    """
    edge_index = _edge_index([(0, 1), (1, 2), (2, 3)])
    first = layout_backbone_pipeline(edge_index, num_nodes=4, keep=0.5, iterations=8, seed=1)
    second = layout_backbone_pipeline(edge_index, num_nodes=4, keep=0.5, iterations=8, seed=999)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()


def test_backbone_public_dispatch_returns_positions() -> None:
    """``dagua.layout`` should dispatch to the backbone pipeline.

    Returns
    -------
    None
        LayoutConfig dispatch must return finite coordinates.
    """
    graph = _graph_from_edges(4, [(0, 1), (1, 2), (2, 3)])
    positions = dagua.layout(
        graph,
        LayoutConfig(
            algorithm="backbone",
            steps=5,
            seed=7,
            algorithm_params={"keep": 0.5},
        ),
    )
    assert positions.shape == (4, 2)
    assert torch.isfinite(positions).all()


def test_backbone_pipeline_never_delegates_to_rscript() -> None:
    """Production backbone code must not call R at runtime.

    Returns
    -------
    None
        The pipeline source must remain free of subprocess/Rscript delegation.
    """
    backbone_module = importlib.import_module("dagua.layout.ops.pipelines.backbone")
    source = inspect.getsource(backbone_module)
    assert "Rscript" not in source
    assert "subprocess" not in source


def test_backbone_reference_adapter_runs_when_available() -> None:
    """Run a small graphlayouts reference smoke if R packages are installed.

    Returns
    -------
    None
        The optional adapter should produce finite positions when available.
    """
    competitor = BackboneCompetitor()
    if not competitor.available():
        pytest.skip("R graphlayouts/oaqc reference packages are not installed")
    graph = _graph_from_edges(4, [(0, 1), (1, 2), (2, 0), (2, 3)])
    result = competitor.layout_with_variant(
        graph,
        timeout=30.0,
        seed=42,
        variant_params={"keep": 0.4},
    )
    assert result.error is None
    assert result.pos is not None
    assert result.pos.shape == (4, 2)
    assert torch.isfinite(result.pos).all()
