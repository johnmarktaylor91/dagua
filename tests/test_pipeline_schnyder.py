"""Fidelity tests for the OGDF-style Schnyder planar pipeline."""

from __future__ import annotations

import inspect

import pytest
import torch

from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.layout import layout
from dagua.layout.ops.pipelines import get_pipeline_function
from dagua.layout.ops.pipelines.planar import PlanarityError
from dagua.layout.ops.pipelines.schnyder import layout_schnyder_pipeline


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
            4,
            [(0, 1), (1, 2), (2, 3)],
            [[80.0, 40.0], [40.0, 80.0], [40.0, 40.0], [0.0, 0.0]],
        ),
        (
            4,
            [(0, 1), (1, 2), (2, 3), (3, 0)],
            [[80.0, 40.0], [40.0, 80.0], [40.0, 40.0], [0.0, 0.0]],
        ),
        (
            4,
            [(i, j) for i in range(4) for j in range(i + 1, 4)],
            [[80.0, 40.0], [40.0, 80.0], [40.0, 40.0], [0.0, 0.0]],
        ),
    ],
)
def test_layout_schnyder_pipeline_matches_ogdf_cached_planar_cases(
    num_nodes: int,
    edges: list[tuple[int, int]],
    expected: list[list[float]],
) -> None:
    """The Schnyder pipeline should match cached OGDF runner coordinates."""
    actual = layout_schnyder_pipeline(_edge_index(edges), num_nodes, fidelity_dtype=torch.float64)

    assert torch.equal(actual.cpu(), torch.tensor(expected, dtype=torch.float64))


def test_layout_schnyder_pipeline_rejects_non_planar_before_reference_call() -> None:
    """Non-planar graphs should be N/A through the Python planarity gate."""
    left = range(3)
    right = range(3, 6)
    edges = [(source, target) for source in left for target in right]

    with pytest.raises(PlanarityError):
        layout_schnyder_pipeline(_edge_index(edges), 6)


def test_schnyder_algorithm_is_registered() -> None:
    """The schnyder key should resolve to the local pipeline function."""
    assert get_pipeline_function("schnyder") is layout_schnyder_pipeline


def test_layout_config_algorithm_schnyder_dispatches() -> None:
    """LayoutConfig should dispatch algorithm='schnyder' through the registry."""
    graph = DaguaGraph.from_edge_list([(0, 1), (1, 2), (2, 3)])

    pos = layout(graph, LayoutConfig(algorithm="schnyder", fidelity_dtype=torch.float64))

    expected = layout_schnyder_pipeline(graph.edge_index, 4).to(dtype=torch.float32)
    assert torch.equal(pos.cpu(), expected)


def test_schnyder_pipeline_does_not_delegate_to_runner() -> None:
    """Pipeline source should not reference the OGDF runner or subprocess."""
    source = inspect.getsource(layout_schnyder_pipeline)

    assert "ogdf_runner" not in source
    assert "subprocess" not in source
