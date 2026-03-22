"""Tests for edge weight support in force-based classic layouts."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from dagua.layout.classic.fr import layout_fr
from dagua.layout.classic.graphopt import layout_graphopt
from dagua.layout.classic.lgl import layout_lgl
from dagua.layout.classic.linlog import layout_linlog
from dagua.layout.classic.spectral import layout_spectral


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
    source = list(range(num_nodes - 1))
    target = list(range(1, num_nodes))
    return torch.tensor([source, target], dtype=torch.long)


def _weighted_layout_call(
    layout_fn: Callable[..., torch.Tensor],
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Invoke a layout with an intentionally invalid edge-weight vector.

    Parameters
    ----------
    layout_fn : Callable[..., torch.Tensor]
        Layout function under test.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    torch.Tensor
        Layout result when validation does not fail.
    """
    kwargs: dict[str, object] = {"edge_weights": torch.ones(edge_index.shape[1] - 1)}
    if layout_fn is layout_fr:
        kwargs["steps"] = 1
    if layout_fn is layout_graphopt:
        kwargs["niter"] = 1
    if layout_fn is layout_lgl:
        kwargs["maxiter"] = 1
        kwargs["root"] = 0
    if layout_fn is layout_linlog:
        kwargs["steps"] = 1
    return layout_fn(edge_index, num_nodes, **kwargs)


@pytest.mark.parametrize(
    ("layout_fn", "num_nodes"),
    [
        (layout_fr, 5),
        (layout_graphopt, 5),
        (layout_lgl, 5),
        (layout_linlog, 5),
        (layout_spectral, 5),
    ],
)
def test_edge_weight_length_validation(
    layout_fn: Callable[..., torch.Tensor],
    num_nodes: int,
) -> None:
    """All weighted layouts reject edge-weight vectors with the wrong length."""
    edge_index = _chain_edges(num_nodes)

    with pytest.raises(ValueError, match="edge_weights"):
        _weighted_layout_call(layout_fn, edge_index, num_nodes)


class TestFRWeights:
    """Coverage for weighted FR layouts."""

    def test_no_weights_backward_compat(self) -> None:
        """Passing ``edge_weights=None`` preserves prior FR output."""
        edge_index = _chain_edges(5)

        first = layout_fr(edge_index, 5, steps=10, seed=42)
        second = layout_fr(edge_index, 5, steps=10, seed=42, edge_weights=None)

        torch.testing.assert_close(first, second)

    def test_weights_affect_layout(self) -> None:
        """Heavier FR edges should change the final layout."""
        edge_index = _chain_edges(5)

        uniform = layout_fr(edge_index, 5, steps=30, seed=42, edge_weights=torch.ones(4))
        heavy = layout_fr(
            edge_index,
            5,
            steps=30,
            seed=42,
            edge_weights=torch.tensor([10.0, 1.0, 1.0, 1.0]),
        )

        assert not torch.allclose(uniform, heavy, atol=0.01)


class TestGraphOptWeights:
    """Coverage for weighted GraphOpt layouts."""

    def test_no_weights_backward_compat(self) -> None:
        """Passing ``edge_weights=None`` preserves prior GraphOpt output."""
        edge_index = _chain_edges(5)

        first = layout_graphopt(edge_index, 5, niter=10, seed=42)
        second = layout_graphopt(edge_index, 5, niter=10, seed=42, edge_weights=None)

        torch.testing.assert_close(first, second)

    def test_weights_run(self) -> None:
        """Weighted GraphOpt should produce finite coordinates."""
        edge_index = _chain_edges(5)

        positions = layout_graphopt(
            edge_index,
            5,
            niter=10,
            seed=42,
            edge_weights=torch.ones(4) * 2.0,
        )

        assert positions.shape == (5, 2)
        assert torch.isfinite(positions).all()


class TestLGLWeights:
    """Coverage for weighted LGL layouts."""

    def test_no_weights_backward_compat(self) -> None:
        """Passing ``edge_weights=None`` preserves prior LGL output."""
        edge_index = _chain_edges(10)

        first = layout_lgl(edge_index, 10, seed=42, maxiter=5, root=0)
        second = layout_lgl(edge_index, 10, seed=42, maxiter=5, root=0, edge_weights=None)

        torch.testing.assert_close(first, second)

    def test_weights_run(self) -> None:
        """Weighted LGL should produce finite coordinates."""
        edge_index = _chain_edges(10)

        positions = layout_lgl(
            edge_index,
            10,
            seed=42,
            maxiter=5,
            root=0,
            edge_weights=torch.ones(9) * 3.0,
        )

        assert positions.shape == (10, 2)
        assert torch.isfinite(positions).all()


class TestLinLogWeights:
    """Coverage for weighted LinLog layouts."""

    def test_no_weights_backward_compat(self) -> None:
        """Passing ``edge_weights=None`` preserves prior LinLog output."""
        edge_index = _chain_edges(5)

        first = layout_linlog(edge_index, 5, steps=10, seed=42)
        second = layout_linlog(edge_index, 5, steps=10, seed=42, edge_weights=None)

        torch.testing.assert_close(first, second)

    def test_weights_affect_layout(self) -> None:
        """Heavier LinLog edges should change the optimized layout."""
        edge_index = _chain_edges(5)

        uniform = layout_linlog(edge_index, 5, steps=30, seed=42, edge_weights=torch.ones(4))
        heavy = layout_linlog(
            edge_index,
            5,
            steps=30,
            seed=42,
            edge_weights=torch.tensor([10.0, 1.0, 1.0, 1.0]),
        )

        assert not torch.allclose(uniform, heavy, atol=0.01)


class TestSpectralWeights:
    """Coverage for weighted spectral layouts."""

    def test_no_weights_backward_compat(self) -> None:
        """Passing ``edge_weights=None`` preserves prior spectral output."""
        edge_index = _chain_edges(5)

        first = layout_spectral(edge_index, 5)
        second = layout_spectral(edge_index, 5, edge_weights=None)

        torch.testing.assert_close(first, second)

    def test_weights_run(self) -> None:
        """Weighted spectral layout should produce finite coordinates."""
        edge_index = _chain_edges(5)

        positions = layout_spectral(edge_index, 5, edge_weights=torch.ones(4) * 2.0)

        assert positions.shape == (5, 2)
        assert torch.isfinite(positions).all()

    def test_weights_affect_eigendecomposition(self) -> None:
        """Weighted adjacency should change the spectral embedding."""
        edge_index = _chain_edges(5)

        uniform = layout_spectral(edge_index, 5, edge_weights=torch.ones(4))
        heavy = layout_spectral(
            edge_index,
            5,
            edge_weights=torch.tensor([10.0, 1.0, 1.0, 1.0]),
        )

        assert not torch.allclose(uniform, heavy, atol=0.01)
