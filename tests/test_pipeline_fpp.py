"""Fidelity tests for the OGDF-style FPP planar pipeline."""

from __future__ import annotations

import inspect

import pytest
import torch

from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.layout import layout
from dagua.layout.ops.pipelines import get_pipeline_function
from dagua.layout.ops.pipelines.fpp import layout_fpp_pipeline
from dagua.layout.ops.pipelines.planar import PlanarityError


def _edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    """Return a Dagua edge tensor.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Edge list.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


@pytest.mark.parametrize(
    ("num_nodes", "edges", "expected"),
    [
        (
            3,
            [(0, 1), (1, 2)],
            [[0.0, 40.0], [80.0, 40.0], [40.0, 0.0]],
        ),
        (
            4,
            [(0, 1), (1, 2), (2, 3)],
            [[0.0, 80.0], [160.0, 80.0], [80.0, 0.0], [80.0, 40.0]],
        ),
        (
            4,
            [(i, j) for i in range(4) for j in range(i + 1, 4)],
            [[0.0, 80.0], [160.0, 80.0], [80.0, 40.0], [80.0, 0.0]],
        ),
    ],
)
def test_layout_fpp_pipeline_matches_ogdf_cached_planar_cases(
    num_nodes: int,
    edges: list[tuple[int, int]],
    expected: list[list[float]],
) -> None:
    """The FPP pipeline should match cached OGDF runner coordinates."""
    actual = layout_fpp_pipeline(_edge_index(edges), num_nodes, fidelity_dtype=torch.float64)

    assert torch.equal(actual.cpu(), torch.tensor(expected, dtype=torch.float64))


def test_layout_fpp_pipeline_rejects_non_planar_before_reference_call() -> None:
    """Non-planar graphs should be N/A through the Python planarity gate."""
    edges = [(i, j) for i in range(5) for j in range(i + 1, 5)]

    with pytest.raises(PlanarityError):
        layout_fpp_pipeline(_edge_index(edges), 5)


def test_fpp_algorithm_is_registered() -> None:
    """The fpp key should resolve to the local pipeline function."""
    assert get_pipeline_function("fpp") is layout_fpp_pipeline


def test_layout_config_algorithm_fpp_dispatches() -> None:
    """LayoutConfig should dispatch algorithm='fpp' through the registry."""
    graph = DaguaGraph.from_edge_list([(0, 1), (1, 2)])

    pos = layout(graph, LayoutConfig(algorithm="fpp", fidelity_dtype=torch.float64))

    assert torch.equal(pos.cpu(), layout_fpp_pipeline(graph.edge_index, 3).to(dtype=torch.float32))


def test_fpp_pipeline_does_not_delegate_to_runner() -> None:
    """Pipeline source should not reference the OGDF runner or subprocess."""
    source = inspect.getsource(layout_fpp_pipeline)

    assert "ogdf_runner" not in source
    assert "subprocess" not in source
