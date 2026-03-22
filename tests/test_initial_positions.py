"""Tests for initial position forwarding in FR and KK."""

from __future__ import annotations

import pytest
import torch

from dagua.layout.classic.fr import layout_fr
from dagua.layout.classic.kk import layout_kk


def _chain_edges(num_nodes: int) -> torch.Tensor:
    """Build a directed chain edge tensor.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the chain graph.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, max(num_nodes - 1, 0)]``.
    """
    sources = list(range(num_nodes - 1))
    targets = list(range(1, num_nodes))
    return torch.tensor([sources, targets], dtype=torch.long)


class TestFRInitialPositions:
    """Regression tests for FR initial position forwarding."""

    def test_custom_init_accepted(self) -> None:
        """FR accepts custom initial positions."""
        edge_index = _chain_edges(5)
        initial_positions = torch.rand(5, 2, dtype=torch.float64)

        positions = layout_fr(edge_index, 5, steps=10, pos=initial_positions)

        assert positions.shape == (5, 2)

    def test_custom_init_differs_from_default(self) -> None:
        """Custom init produces a different result than the default init."""
        edge_index = _chain_edges(10)

        default_positions = layout_fr(edge_index, 10, steps=30, seed=42)
        initial_positions = torch.ones(10, 2, dtype=torch.float64) * 0.5
        custom_positions = layout_fr(edge_index, 10, steps=30, seed=42, pos=initial_positions)

        assert not torch.allclose(default_positions, custom_positions, atol=0.1)

    def test_zero_steps_returns_rescaled_input(self) -> None:
        """Zero FR steps still return a finite rescaled layout."""
        edge_index = _chain_edges(5)
        initial_positions = torch.rand(5, 2, dtype=torch.float64)

        positions = layout_fr(edge_index, 5, steps=0, pos=initial_positions)

        assert positions.shape == (5, 2)
        assert torch.isfinite(positions).all()

    def test_wrong_shape_raises(self) -> None:
        """Wrong shape pos raises ``ValueError``."""
        edge_index = _chain_edges(5)

        with pytest.raises(ValueError, match="pos must have shape"):
            layout_fr(edge_index, 5, steps=1, pos=torch.rand(4, 2))

    def test_none_pos_uses_default(self) -> None:
        """``pos=None`` gives the same result as omitting the parameter."""
        edge_index = _chain_edges(5)

        first_positions = layout_fr(edge_index, 5, steps=10, seed=42)
        second_positions = layout_fr(edge_index, 5, steps=10, seed=42, pos=None)

        assert torch.allclose(first_positions, second_positions)


class TestKKInitialPositions:
    """Regression tests for KK initial position forwarding."""

    def test_custom_init_accepted(self) -> None:
        """KK accepts custom initial positions."""
        edge_index = _chain_edges(5)
        initial_positions = torch.rand(5, 2)

        positions = layout_kk(edge_index, 5, steps=10, pos=initial_positions)

        assert positions.shape == (5, 2)

    def test_custom_init_differs_from_default(self) -> None:
        """Custom init produces a different result than the default init."""
        edge_index = _chain_edges(10)

        default_positions = layout_kk(edge_index, 10, steps=50)
        initial_positions = torch.rand(10, 2)
        custom_positions = layout_kk(edge_index, 10, steps=50, pos=initial_positions)

        assert not torch.allclose(default_positions, custom_positions, atol=0.1)

    def test_wrong_shape_raises(self) -> None:
        """Wrong shape pos raises ``ValueError``."""
        edge_index = _chain_edges(5)

        with pytest.raises(ValueError, match="pos must have shape"):
            layout_kk(edge_index, 5, steps=1, pos=torch.rand(4, 2))

    def test_none_pos_uses_default(self) -> None:
        """``pos=None`` gives the same result as omitting the parameter."""
        edge_index = _chain_edges(5)

        first_positions = layout_kk(edge_index, 5, steps=50)
        second_positions = layout_kk(edge_index, 5, steps=50, pos=None)

        assert torch.allclose(first_positions, second_positions)

    def test_trace_with_custom_init(self) -> None:
        """Tracing works when KK starts from custom positions."""
        edge_index = _chain_edges(5)
        initial_positions = torch.rand(5, 2)

        positions, traces = layout_kk(
            edge_index,
            5,
            steps=50,
            trace_every=10,
            pos=initial_positions,
        )

        assert positions.shape == (5, 2)
        assert len(traces) > 0
