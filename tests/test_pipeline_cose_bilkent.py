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


def test_cose_bilkent_compound_groups_separate_centroids() -> None:
    """Compound members should be laid out as grouped CoSE-Bilkent packs.

    Returns
    -------
    None
        Cluster centroids should be separated by the compound parent stage.
    """
    edge_index = torch.tensor(
        [[0, 1, 3, 4, 2, 5], [1, 2, 4, 5, 3, 0]],
        dtype=torch.long,
    )
    clusters = {"a": [0, 1, 2], "b": [3, 4, 5]}
    pos = layout_cose_bilkent_pipeline(
        edge_index=edge_index,
        num_nodes=6,
        steps=5,
        seed=7,
        clusters=clusters,
    )

    first_center = pos[torch.tensor(clusters["a"])].mean(dim=0)
    second_center = pos[torch.tensor(clusters["b"])].mean(dim=0)
    assert torch.linalg.vector_norm(first_center - second_center) > 100.0
