"""Regression tests for Graphviz-dot x-position fidelity helpers."""

from __future__ import annotations

import pytest
import torch

from dagua.config import LayoutConfig
from dagua.layout.ops.pipelines.dagua_native import (
    _graphviz_dot_x_position_network_simplex,
    layout_dagua_native_pipeline,
)


def test_dot_x_position_network_simplex_matches_complete_bipartite_golden() -> None:
    """Match Graphviz ``dot -Tplain`` x coordinates for a two-rank graph.

    The golden vector was captured from Graphviz with:
    ``graph[nodesep=0.25, ranksep=0.5]`` and fixed 1-inch box nodes for
    ``a -> c; a -> d; b -> c; b -> d``. Coordinates are converted to points
    and centered, matching the stable comparison frame used by Dagua metrics.
    """
    edge_index = torch.tensor([[0, 0, 1, 1], [2, 3, 2, 3]], dtype=torch.long)
    node_widths = torch.full((4,), 72.0, dtype=torch.float64)

    x_coords = _graphviz_dot_x_position_network_simplex(
        rank_ordering=[[0, 1], [2, 3]],
        node_widths=node_widths,
        edge_index=edge_index,
    )

    expected = torch.tensor([-45.0, 45.0, -45.0, 45.0], dtype=torch.float64)
    torch.testing.assert_close(x_coords, expected, rtol=0.0, atol=1.0e-9)


def test_dot_x_position_network_simplex_rounds_unequal_width_lr_constraints() -> None:
    """Match Graphviz rounded left-to-right spacing for unequal same-rank boxes."""
    node_widths = torch.tensor([72.0, 36.0, 108.0], dtype=torch.float64)
    edge_index = torch.zeros((2, 0), dtype=torch.long)

    x_coords = _graphviz_dot_x_position_network_simplex(
        rank_ordering=[[0, 1, 2]],
        node_widths=node_widths,
        edge_index=edge_index,
    )

    expected = torch.tensor([-78.0, -6.0, 84.0], dtype=torch.float64)
    torch.testing.assert_close(x_coords, expected, rtol=0.0, atol=1.0e-9)


def test_dot_x_position_network_simplex_rejects_incomplete_rank_ordering() -> None:
    """Invalid rank-order input should fail before solving the auxiliary LP."""
    node_widths = torch.ones(3, dtype=torch.float64)
    edge_index = torch.zeros((2, 0), dtype=torch.long)

    with pytest.raises(ValueError, match="every node"):
        _graphviz_dot_x_position_network_simplex(
            rank_ordering=[[0, 1]],
            node_widths=node_widths,
            edge_index=edge_index,
        )


def test_dagua_native_dot_position_fidelity_uses_x_position_component() -> None:
    """The narrow fidelity selector should expose the x-position port end to end."""
    edge_index = torch.tensor([[0, 0, 1, 1], [2, 3, 2, 3]], dtype=torch.long)
    node_sizes = torch.full((4, 2), 72.0, dtype=torch.float32)

    positions = layout_dagua_native_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        node_sizes=node_sizes,
        config=LayoutConfig(seed=42, algorithm="dagua_native"),
        fidelity_mode="graphviz_dot_position",
    )

    expected_x = torch.tensor([-45.0, 45.0, -45.0, 45.0], dtype=torch.float32)
    torch.testing.assert_close(positions[:, 0].cpu(), expected_x, rtol=0.0, atol=1.0e-6)
