"""Fidelity tests for the OGDF-style BalloonLayout pipeline."""

from __future__ import annotations

import inspect

import pytest
import torch

from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.layout import layout
from dagua.layout.ops.pipelines import get_pipeline_function
from dagua.layout.ops.pipelines.balloon import layout_balloon_pipeline


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
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


@pytest.mark.parametrize(
    ("num_nodes", "edges", "expected"),
    [
        (
            3,
            [(0, 1), (1, 2)],
            [
                [1.7319121124709867e-15, 28.284271247461902],
                [0.0, 0.0],
                [1.7319121124709867e-15, -28.284271247461902],
            ],
        ),
        (
            4,
            [(0, 1), (1, 2), (2, 3)],
            [
                [3.6370154361890722e-15, 59.396969619669996],
                [1.9051033237180856e-15, 31.112698372208094],
                [0.0, 0.0],
                [1.9051033237180856e-15, -31.112698372208094],
            ],
        ),
        (
            4,
            [(0, 1), (1, 2), (2, 3), (3, 0)],
            [
                [0.0, 0.0],
                [1.9051033237180856e-15, -31.112698372208094],
                [3.6370154361890722e-15, -59.396969619669996],
                [1.9051033237180856e-15, 31.112698372208094],
            ],
        ),
    ],
)
def test_layout_balloon_pipeline_matches_ogdf_small_trees_and_cycle(
    num_nodes: int,
    edges: list[tuple[int, int]],
    expected: list[list[float]],
) -> None:
    """The Balloon pipeline should match cached OGDF runner coordinates."""
    actual = layout_balloon_pipeline(_edge_index(edges), num_nodes, fidelity_dtype=torch.float64)

    assert torch.allclose(actual.cpu(), torch.tensor(expected, dtype=torch.float64), atol=1e-12)


def test_balloon_algorithm_is_registered() -> None:
    """The balloon key should resolve to the local pipeline function."""
    assert get_pipeline_function("balloon") is layout_balloon_pipeline


def test_layout_config_algorithm_balloon_dispatches() -> None:
    """LayoutConfig should dispatch algorithm='balloon' through the registry."""
    graph = DaguaGraph.from_edge_list([(0, 1), (1, 2)])

    pos = layout(graph, LayoutConfig(algorithm="balloon", fidelity_dtype=torch.float64))

    assert pos.shape == (3, 2)
    expected = layout_balloon_pipeline(graph.edge_index, 3, node_sizes=graph.node_sizes).to(
        dtype=torch.float32
    )
    assert torch.equal(pos.cpu(), expected)


def test_balloon_pipeline_does_not_delegate_to_runner() -> None:
    """Pipeline source should not reference the OGDF runner or subprocess."""
    source = inspect.getsource(layout_balloon_pipeline)

    assert "ogdf_runner" not in source
    assert "subprocess" not in source
