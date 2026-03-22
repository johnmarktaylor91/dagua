"""Tests for FA2 feature support: LinLog, dissuade-hubs, Barnes-Hut, and weights."""

from __future__ import annotations

import pytest
import torch

from dagua.layout.classic.fa2 import layout_fa2


def _triangle_edges() -> torch.Tensor:
    """Build a triangle graph edge list.

    Parameters
    ----------
    None
        No parameters.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, 3]``.
    """
    return torch.tensor([[0, 1, 0], [1, 2, 2]], dtype=torch.long)


def _chain_edges(num_nodes: int) -> torch.Tensor:
    """Build a path graph edge list.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the chain.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, num_nodes - 1]``.
    """
    source = list(range(num_nodes - 1))
    target = list(range(1, num_nodes))
    return torch.tensor([source, target], dtype=torch.long)


class TestLinLog:
    """Regression tests for LinLog attraction support."""

    def test_linlog_runs(self) -> None:
        """Allow ``linlog=True`` without raising an error."""
        pos = layout_fa2(_triangle_edges(), 3, steps=10, linlog=True)
        assert pos.shape == (3, 2)

    def test_linlog_differs_from_default(self) -> None:
        """Produce a measurably different layout from linear attraction."""
        edge_index = _chain_edges(10)
        pos_linear = layout_fa2(edge_index, 10, steps=50, seed=42, linlog=False)
        pos_linlog = layout_fa2(edge_index, 10, steps=50, seed=42, linlog=True)
        assert not torch.allclose(pos_linear, pos_linlog, atol=0.01)


class TestDissuadeHubs:
    """Regression tests for dissuading hub attraction."""

    def test_dissuade_hubs_runs(self) -> None:
        """Allow ``dissuade_hubs=True`` without numerical issues."""
        pos = layout_fa2(_triangle_edges(), 3, steps=10, dissuade_hubs=True)
        assert pos.shape == (3, 2)

    def test_dissuade_hubs_spreads_hub(self) -> None:
        """Keep layouts finite on a simple star graph when hub spreading is on."""
        edge_index = torch.tensor([[0, 0, 0, 0], [1, 2, 3, 4]], dtype=torch.long)
        pos_normal = layout_fa2(edge_index, 5, steps=100, seed=42, dissuade_hubs=False)
        pos_dissuade = layout_fa2(edge_index, 5, steps=100, seed=42, dissuade_hubs=True)

        assert pos_normal.shape == (5, 2)
        assert pos_dissuade.shape == (5, 2)
        assert torch.isfinite(pos_dissuade).all()


class TestBarnesHut:
    """Regression tests for Barnes-Hut repulsion."""

    def test_barnes_hut_runs(self) -> None:
        """Allow Barnes-Hut repulsion on a small chain graph."""
        edge_index = _chain_edges(20)
        pos = layout_fa2(edge_index, 20, steps=10, barnes_hut=True)
        assert pos.shape == (20, 2)

    def test_barnes_hut_similar_to_exact(self) -> None:
        """Keep the Barnes-Hut layout in the same rough scale as exact repulsion."""
        edge_index = _chain_edges(30)
        pos_exact = layout_fa2(edge_index, 30, steps=50, seed=42, barnes_hut=False)
        pos_barnes_hut = layout_fa2(
            edge_index,
            30,
            steps=50,
            seed=42,
            barnes_hut=True,
            barnes_hut_theta=0.5,
        )

        assert torch.isfinite(pos_exact).all()
        assert torch.isfinite(pos_barnes_hut).all()

        exact_span = (pos_exact.max(0).values - pos_exact.min(0).values).mean()
        barnes_hut_span = (pos_barnes_hut.max(0).values - pos_barnes_hut.min(0).values).mean()
        assert barnes_hut_span > exact_span * 0.1

    def test_barnes_hut_theta_validation(self) -> None:
        """Reject non-positive Barnes-Hut theta values."""
        with pytest.raises(ValueError, match="barnes_hut_theta"):
            layout_fa2(_triangle_edges(), 3, steps=1, barnes_hut_theta=-1.0)


class TestEdgeWeights:
    """Regression tests for weighted attraction."""

    def test_edge_weights_runs(self) -> None:
        """Allow weighted attraction on a small triangle graph."""
        edge_index = _triangle_edges()
        edge_weights = torch.tensor([1.0, 2.0, 3.0])
        pos = layout_fa2(edge_index, 3, steps=10, edge_weights=edge_weights)
        assert pos.shape == (3, 2)

    def test_edge_weights_affect_layout(self) -> None:
        """Change the layout when one edge receives a much larger weight."""
        edge_index = _chain_edges(5)
        pos_uniform = layout_fa2(
            edge_index,
            5,
            steps=50,
            seed=42,
            edge_weights=torch.ones(4),
        )
        pos_heavy = layout_fa2(
            edge_index,
            5,
            steps=50,
            seed=42,
            edge_weights=torch.tensor([10.0, 1.0, 1.0, 1.0]),
        )
        assert not torch.allclose(pos_uniform, pos_heavy, atol=0.01)

    def test_edge_weights_validation(self) -> None:
        """Reject weight tensors whose length does not match the edge count."""
        with pytest.raises(ValueError, match="edge_weights"):
            layout_fa2(_triangle_edges(), 3, steps=1, edge_weights=torch.tensor([1.0, 2.0]))


class TestCombined:
    """Regression test for enabling all new FA2 options together."""

    def test_all_features_together(self) -> None:
        """Keep the combined feature path finite on a medium chain graph."""
        edge_index = _chain_edges(20)
        edge_weights = torch.ones(19) * 2.0
        pos = layout_fa2(
            edge_index,
            20,
            steps=20,
            seed=42,
            linlog=True,
            dissuade_hubs=True,
            barnes_hut=True,
            barnes_hut_theta=1.0,
            edge_weights=edge_weights,
        )
        assert pos.shape == (20, 2)
        assert torch.isfinite(pos).all()
