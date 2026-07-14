"""Fidelity tests for the OGDF-style Bertault pipeline."""

from __future__ import annotations

import inspect

import pytest
import torch

from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.layout import layout
from dagua.layout.ops.pipelines import get_pipeline_function
from dagua.layout.ops.pipelines.bertault import layout_bertault_pipeline


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
                [-3511.6092627744565, -416.60514115617576],
                [-3488.9437407001105, -432.6322251943325],
                [-3426.564345334339, -476.77093424317013],
            ],
        ),
        (
            4,
            [(0, 1), (1, 2), (2, 3)],
            [
                [99.27553198825484, 161.40910423585186],
                [96.20129354140481, 105.88313979887081],
                [92.37777109719656, 36.82372290995234],
                [90.15523893078345, -3.319232597929606],
            ],
        ),
        (
            4,
            [(0, 1), (1, 2), (2, 3), (3, 0)],
            [
                [31.743557476783444, 176.68488513353714],
                [149.18391323900443, 90.69041213391853],
                [75.88122433948044, -34.92206422542782],
                [-44.76856892093936, 48.84445009787429],
            ],
        ),
        (
            4,
            [(i, j) for i in range(4) for j in range(i + 1, 4)],
            [
                [20.203957803753383, 182.2754547947671],
                [159.55196468605448, 157.30205534069063],
                [138.3793716535179, -99.51329067872854],
                [-56.373629762585416, -63.13299906149087],
            ],
        ),
    ],
)
def test_layout_bertault_pipeline_matches_ogdf_cached_small_cases(
    num_nodes: int,
    edges: list[tuple[int, int]],
    expected: list[list[float]],
) -> None:
    """The Bertault pipeline should match cached OGDF coordinates numerically."""
    actual = layout_bertault_pipeline(
        _edge_index(edges),
        num_nodes,
        seed=1,
        fidelity_dtype=torch.float64,
    )

    assert torch.allclose(
        actual.cpu(),
        torch.tensor(expected, dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-6,
    )


def test_bertault_algorithm_is_registered() -> None:
    """The bertault key should resolve to the local pipeline function."""
    assert get_pipeline_function("bertault") is layout_bertault_pipeline


def test_layout_config_algorithm_bertault_dispatches() -> None:
    """LayoutConfig should dispatch algorithm='bertault' through the registry."""
    graph = DaguaGraph.from_edge_list([(0, 1), (1, 2), (2, 3)])

    pos = layout(graph, LayoutConfig(algorithm="bertault", seed=1, fidelity_dtype=torch.float64))

    expected = layout_bertault_pipeline(
        graph.edge_index,
        4,
        seed=1,
        node_sizes=graph.node_sizes,
    ).to(dtype=torch.float32)
    assert torch.equal(pos.cpu(), expected)


def test_bertault_pipeline_does_not_delegate_to_runner() -> None:
    """Pipeline source should not reference the OGDF runner or subprocess."""
    source = inspect.getsource(layout_bertault_pipeline)

    assert "ogdf_runner" not in source
    assert "subprocess" not in source
