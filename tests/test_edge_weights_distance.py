"""Tests for edge weight support in distance-based classic layouts."""

from __future__ import annotations

import pytest
import torch

from dagua.layout.classic.kk import layout_kk
from dagua.layout.classic.maxent_stress import layout_maxent_stress
from dagua.layout.classic.pivot_mds import layout_pivot_mds
from dagua.layout.classic.sgd2_multi import layout_sgd2_multi
from dagua.layout.classic.stress_sgd import layout_stress_sgd
from dagua.layout.classic.tsnet import layout_tsnet


def _chain_edges(num_nodes: int) -> torch.Tensor:
    """Build a directed chain edge list.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the chain.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    source = torch.arange(0, num_nodes - 1, dtype=torch.long)
    target = source + 1
    return torch.stack([source, target], dim=0)


def _unwrap_positions(
    result: torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]],
) -> torch.Tensor:
    """Extract the position tensor from a layout result.

    Parameters
    ----------
    result : torch.Tensor or tuple[torch.Tensor, list[torch.Tensor]]
        Layout output.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    if isinstance(result, tuple):
        return result[0]
    return result


@pytest.mark.parametrize(
    ("layout_fn", "kwargs"),
    [
        (layout_kk, {"steps": 50}),
        (layout_stress_sgd, {"steps": 10}),
        (layout_maxent_stress, {"steps": 10}),
        (layout_pivot_mds, {}),
        (layout_tsnet, {"steps": 10, "perplexity": 2.0}),
        (layout_sgd2_multi, {"steps": 10}),
    ],
)
class TestDistanceAlgoWeights:
    """Regression coverage for weighted graph-distance layouts."""

    def test_no_weights_backward_compat(self, layout_fn: object, kwargs: dict[str, object]) -> None:
        """Passing ``edge_weights=None`` should match the legacy code path."""
        edge_index = _chain_edges(10)
        positions_a = _unwrap_positions(layout_fn(edge_index, 10, seed=42, **kwargs))
        positions_b = _unwrap_positions(
            layout_fn(edge_index, 10, seed=42, edge_weights=None, **kwargs)
        )
        assert torch.allclose(positions_a, positions_b, atol=1.0e-4)

    def test_weights_run(self, layout_fn: object, kwargs: dict[str, object]) -> None:
        """Weighted runs should produce finite 2D coordinates."""
        edge_index = _chain_edges(10)
        edge_weights = torch.full((9,), 2.0, dtype=torch.float32)
        positions = _unwrap_positions(
            layout_fn(edge_index, 10, seed=42, edge_weights=edge_weights, **kwargs)
        )
        assert positions.shape == (10, 2)
        assert torch.isfinite(positions).all()

    def test_weights_affect_layout(self, layout_fn: object, kwargs: dict[str, object]) -> None:
        """Changing edge weights should change the weighted layout target."""
        edge_index = _chain_edges(10)
        uniform_weights = torch.ones(9, dtype=torch.float32)
        heavy_weights = torch.tensor([10.0] + [1.0] * 8, dtype=torch.float32)
        positions_uniform = _unwrap_positions(
            layout_fn(edge_index, 10, seed=42, edge_weights=uniform_weights, **kwargs)
        )
        positions_heavy = _unwrap_positions(
            layout_fn(edge_index, 10, seed=42, edge_weights=heavy_weights, **kwargs)
        )
        assert not torch.allclose(positions_uniform, positions_heavy, atol=0.1)
