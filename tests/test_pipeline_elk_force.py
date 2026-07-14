"""Tests for the ELK Force pipeline."""

from __future__ import annotations

from pathlib import Path

import torch

from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.layout import layout
from dagua.layout.ops.pipelines import get_pipeline_function
from dagua.layout.ops.pipelines.elk_force import layout_elk_force_pipeline


def _path_graph() -> tuple[torch.Tensor, torch.Tensor]:
    """Build a small path graph tensor fixture.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Edge-index tensor with shape ``[2, E]`` and node sizes ``[N, 2]``.
    """
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    node_sizes = torch.full((4, 2), 20.0, dtype=torch.float64)
    return edge_index, node_sizes


def test_elk_force_pipeline_is_registered() -> None:
    """ELK Force should resolve through the shared pipeline registry."""
    assert get_pipeline_function("elk_force") is layout_elk_force_pipeline


def test_elk_force_is_deterministic_for_seed() -> None:
    """Seeded ELK Force should return repeatable finite positions."""
    edge_index, node_sizes = _path_graph()
    first = layout_elk_force_pipeline(edge_index, 4, node_sizes, seed=1, iterations=4)
    second = layout_elk_force_pipeline(edge_index, 4, node_sizes, seed=1, iterations=4)
    assert first.shape == (4, 2)
    assert torch.isfinite(first).all()
    assert torch.allclose(first, second)


def test_elk_force_dispatches_from_layout_config() -> None:
    """The public layout engine should dispatch ``algorithm='elk_force'``."""
    graph = DaguaGraph()
    for node in range(4):
        graph.add_node(node)
    for source in range(3):
        graph.add_edge(source, source + 1)
    pos = layout(graph, LayoutConfig(algorithm="elk_force", algorithm_params={"iterations": 2}))
    assert pos.shape == (4, 2)


def test_elk_force_pipeline_does_not_delegate_to_elkjs() -> None:
    """Runtime pipeline source must not import or execute elkjs/Node."""
    source = Path("dagua/layout/ops/pipelines/elk_force.py").read_text()
    assert "elkjs" not in source
    assert "subprocess" not in source
