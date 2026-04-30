"""Regression tests for Reingold-Tilford igraph fidelity mode helpers."""

from __future__ import annotations

import pytest
import torch

from dagua.layout.ops.pipelines.reingold_tilford import layout_reingold_tilford_pipeline


def test_rt_igraph_fidelity_uses_unit_spacing() -> None:
    """Verify igraph fidelity mode ignores node-size-derived default spacing."""
    edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
    compact_sizes = torch.ones((3, 2), dtype=torch.float32)
    wide_sizes = torch.tensor([[100.0, 5.0], [1.0, 1.0], [1.0, 1.0]], dtype=torch.float32)

    compact_pos = layout_reingold_tilford_pipeline(
        edge_index,
        3,
        node_sizes=compact_sizes,
        fidelity_mode="igraph",
    )
    wide_pos = layout_reingold_tilford_pipeline(
        edge_index,
        3,
        node_sizes=wide_sizes,
        fidelity_mode="igraph",
    )

    torch.testing.assert_close(compact_pos, wide_pos)


def test_rt_igraph_fidelity_uses_out_traversal() -> None:
    """Verify the igraph default OUT traversal does not follow incoming edges."""
    edge_index = torch.tensor([[0, 2], [1, 1]], dtype=torch.long)

    fidelity_pos = layout_reingold_tilford_pipeline(edge_index, 3, fidelity_mode="igraph")

    torch.testing.assert_close(fidelity_pos[0, 1], fidelity_pos[2, 1])
    assert float(fidelity_pos[1, 1].item()) > float(fidelity_pos[0, 1].item())


def test_rt_igraph_fidelity_dedupes_duplicate_edges_for_roots() -> None:
    """Verify duplicate edges do not change fidelity root ranking."""
    simple_edges = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
    duplicate_edges = torch.tensor([[0, 0, 0], [1, 1, 2]], dtype=torch.long)

    simple_pos = layout_reingold_tilford_pipeline(simple_edges, 3, fidelity_mode="igraph")
    duplicate_pos = layout_reingold_tilford_pipeline(duplicate_edges, 3, fidelity_mode="igraph")

    torch.testing.assert_close(simple_pos, duplicate_pos)


def test_rt_igraph_fidelity_returns_igraph_scaled_uncentered_units() -> None:
    """Verify igraph mode mirrors the adapter scale and origin policy."""
    edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)

    fidelity_pos = layout_reingold_tilford_pipeline(edge_index, 3, fidelity_mode="igraph")

    torch.testing.assert_close(fidelity_pos[0], torch.tensor([25.0, 0.0]))
    torch.testing.assert_close(fidelity_pos[1], torch.tensor([0.0, 50.0]))
    torch.testing.assert_close(fidelity_pos[2], torch.tensor([50.0, 50.0]))


def test_rt_igraph_fidelity_accepts_explicit_roots_and_rootlevel() -> None:
    """Verify explicit roots and rootlevel control directed traversal fixtures."""
    edge_index = torch.tensor([[0, 1], [2, 2]], dtype=torch.long)

    rooted_pos = layout_reingold_tilford_pipeline(
        edge_index,
        3,
        fidelity_mode="igraph",
        roots=[1],
        rootlevel=[2],
    )

    torch.testing.assert_close(rooted_pos[1, 1], torch.tensor(100.0))
    torch.testing.assert_close(rooted_pos[2, 1], torch.tensor(150.0))


def test_rt_igraph_fidelity_rejects_unknown_mode() -> None:
    """Verify invalid fidelity mode values fail explicitly."""
    edge_index = torch.empty((2, 0), dtype=torch.long)

    with pytest.raises(ValueError, match="fidelity_mode"):
        layout_reingold_tilford_pipeline(edge_index, 1, fidelity_mode="reference")
