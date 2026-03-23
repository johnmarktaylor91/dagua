"""Tests for the NeuLay classic layout."""

from __future__ import annotations

import torch

from dagua.layout.classic.neulay import layout_neulay


def _cycle_edge_index(num_nodes: int) -> torch.Tensor:
    """Build a directed cycle edge list.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the cycle.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    source = torch.arange(0, num_nodes, dtype=torch.long)
    target = (source + 1) % num_nodes
    return torch.stack([source, target], dim=0)


def test_layout_neulay_returns_expected_shape() -> None:
    """NeuLay should return one 2D coordinate per node."""
    edge_index = _cycle_edge_index(5)

    positions = layout_neulay(
        edge_index=edge_index,
        num_nodes=5,
        seed=42,
        steps=96,
        gcn_steps=0,
        use_gcn=False,
    )

    assert positions.shape == (5, 2)
    assert torch.isfinite(positions).all()


def test_layout_neulay_is_deterministic_without_gcn() -> None:
    """The FDL-only NeuLay path should be deterministic for a fixed seed."""
    edge_index = _cycle_edge_index(6)

    positions_a = layout_neulay(
        edge_index=edge_index,
        num_nodes=6,
        seed=7,
        steps=128,
        gcn_steps=0,
        use_gcn=False,
    )
    positions_b = layout_neulay(
        edge_index=edge_index,
        num_nodes=6,
        seed=7,
        steps=128,
        gcn_steps=0,
        use_gcn=False,
    )

    assert torch.allclose(positions_a, positions_b)


def test_layout_neulay_avoids_collapse_on_path_graph() -> None:
    """NeuLay should spread nodes rather than collapsing them to one point."""
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

    positions = layout_neulay(
        edge_index=edge_index,
        num_nodes=5,
        seed=19,
        steps=192,
        gcn_steps=0,
        use_gcn=False,
    )

    pairwise = torch.cdist(positions, positions)
    assert float(pairwise.max().item()) > 0.1


def test_layout_neulay_accepts_edge_weights_without_gcn_support() -> None:
    """NeuLay should accept edge weights even though the current port ignores them."""
    edge_index = _cycle_edge_index(5)

    positions = layout_neulay(
        edge_index=edge_index,
        num_nodes=5,
        seed=11,
        steps=64,
        gcn_steps=0,
        use_gcn=False,
        edge_weights=torch.ones(edge_index.shape[1], dtype=torch.float32),
    )

    assert positions.shape == (5, 2)
    assert torch.isfinite(positions).all()


def test_layout_neulay_rejects_mismatched_edge_weights() -> None:
    """NeuLay should reject edge-weight tensors that do not match ``E``."""
    edge_index = _cycle_edge_index(4)

    try:
        layout_neulay(
            edge_index=edge_index,
            num_nodes=4,
            steps=16,
            gcn_steps=0,
            use_gcn=False,
            edge_weights=torch.ones(3, dtype=torch.float32),
        )
    except ValueError as exc:
        assert "edge_weights length" in str(exc)
    else:
        raise AssertionError("layout_neulay accepted mismatched edge_weights")
