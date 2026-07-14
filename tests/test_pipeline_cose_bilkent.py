"""Regression tests for the Cytoscape CoSE-Bilkent pipeline."""

from __future__ import annotations

import torch

from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.cose_bilkent import layout_cose_bilkent_pipeline


def test_cose_bilkent_pipeline_is_registered() -> None:
    """The dynamic registry should resolve ``cose_bilkent``.

    Returns
    -------
    None
        Registry lookup must return the public entrypoint.
    """
    assert PIPELINE_REGISTRY["cose_bilkent"] == (
        "dagua.layout.ops.pipelines.cose_bilkent",
        "layout_cose_bilkent_pipeline",
    )
    assert get_pipeline_function("cose_bilkent") is layout_cose_bilkent_pipeline


def test_cose_bilkent_quality_tiers_run() -> None:
    """CoSE-Bilkent draft/default/proof variants should execute.

    Returns
    -------
    None
        Every tier should return finite positions.
    """
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    clusters = {"outer": [0, 1, 2, 3]}
    for quality in ("draft", "default", "proof"):
        pos = layout_cose_bilkent_pipeline(
            edge_index=edge_index,
            num_nodes=4,
            steps=2,
            seed=3,
            quality=quality,
            clusters=clusters,
        )
        assert pos.shape == (4, 2)
        assert torch.isfinite(pos).all()
