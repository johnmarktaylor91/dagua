"""Regression tests for the Cytoscape CoSE pipeline."""

from __future__ import annotations

import torch

from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.cose import layout_cose_pipeline


def test_cose_pipeline_is_registered() -> None:
    """The dynamic registry should resolve ``cose``.

    Returns
    -------
    None
        Registry lookup must return the public entrypoint.
    """
    assert PIPELINE_REGISTRY["cose"] == (
        "dagua.layout.ops.pipelines.cose",
        "layout_cose_pipeline",
    )
    assert get_pipeline_function("cose") is layout_cose_pipeline


def test_cose_pipeline_produces_finite_layout() -> None:
    """CoSE should produce finite coordinates for a small graph.

    Returns
    -------
    None
        Output coordinates must be finite and non-collapsed.
    """
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]], dtype=torch.long)
    node_sizes = torch.tensor([[40.0, 20.0]] * 4)
    pos = layout_cose_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        node_sizes=node_sizes,
        steps=3,
        randomize=False,
    )

    assert pos.shape == (4, 2)
    assert torch.isfinite(pos).all()
    assert float(pos.std().item()) > 0.0


def test_cose_randomize_honors_seed() -> None:
    """Randomized CoSE initial placement should be repeatable per seed.

    Returns
    -------
    None
        Same seed must match and different seed should change coordinates.
    """
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    first = layout_cose_pipeline(edge_index, 4, steps=0, seed=7, randomize=True)
    repeated = layout_cose_pipeline(edge_index, 4, steps=0, seed=7, randomize=True)
    second = layout_cose_pipeline(edge_index, 4, steps=0, seed=8, randomize=True)

    assert torch.equal(first, repeated)
    assert not torch.equal(first, second)
