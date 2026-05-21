"""Regression tests for classic NeuLay and tsNET autograd setup."""

from __future__ import annotations

import torch

from dagua.layout.ops.pipelines.neulay import layout_neulay_pipeline
from dagua.layout.ops.pipelines.tsnet import layout_tsnet_pipeline


def _small_edge_index() -> torch.Tensor:
    """Return a connected undirected test graph.

    Returns
    -------
    torch.Tensor
        Edge-index tensor with shape ``[2, E]``.
    """
    return torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 2, 1, 3],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 2, 0, 3, 1],
        ],
        dtype=torch.long,
    )


def test_neulay_pipeline_backward_survives_outer_no_grad() -> None:
    """Confirm NeuLay GCN and direct losses restore grad mode for backward."""
    edge_index = _small_edge_index()

    with torch.no_grad():
        positions = layout_neulay_pipeline(
            edge_index=edge_index,
            num_nodes=6,
            seed=42,
            steps=2,
            gcn_steps=1,
            fdl_steps=1,
            use_gcn=True,
            lr=0.1,
            radius=0.4,
        )

    assert positions.shape == (6, 2)
    assert positions.dtype == torch.float32
    assert torch.isfinite(positions).all()


def test_tsnet_pipeline_backward_survives_outer_no_grad() -> None:
    """Confirm tsNET KL loss restores grad mode for backward."""
    edge_index = _small_edge_index()

    with torch.no_grad():
        positions = layout_tsnet_pipeline(
            edge_index=edge_index,
            num_nodes=6,
            perplexity=3.0,
            steps=1,
            seed=42,
        )

    assert positions.shape == (6, 2)
    assert positions.dtype == torch.float32
    assert torch.isfinite(positions).all()
