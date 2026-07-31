"""Regression tests for the Cytoscape CiSE pipeline."""

from __future__ import annotations

import torch

from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.cise import layout_cise_pipeline


def test_cise_pipeline_is_registered() -> None:
    """The dynamic registry should resolve ``cise``.

    Returns
    -------
    None
        Registry lookup must return the public entrypoint.
    """
    assert PIPELINE_REGISTRY["cise"] == (
        "dagua.layout.ops.pipelines.cise",
        "layout_cise_pipeline",
    )
    assert get_pipeline_function("cise") is layout_cise_pipeline


def test_cise_places_cluster_members_on_circles() -> None:
    """CiSE should keep cluster members on circular rings.

    Returns
    -------
    None
        Members in each cluster should have equal distance from their local
        centroid for this symmetric fixture.
    """
    edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 4]], dtype=torch.long)
    clusters = {"left": [0, 1, 2, 3], "right": [4, 5, 6, 7]}
    pos = layout_cise_pipeline(edge_index=edge_index, num_nodes=8, clusters=clusters)

    assert pos.shape == (8, 2)
    assert torch.isfinite(pos).all()
    for members in clusters.values():
        member_pos = pos[torch.tensor(members)]
        center = member_pos.mean(dim=0, keepdim=True)
        radii = torch.linalg.vector_norm(member_pos - center, dim=1)
        assert torch.allclose(radii, radii.mean().expand_as(radii), atol=1.0e-4)


def test_cise_uses_cytoscape_default_circle_geometry() -> None:
    """CiSE cluster radii should follow Cytoscape's default node dimensions.

    Returns
    -------
    None
        The radius should be computed from 30x30 Cytoscape layout dimensions,
        not from Dagua's text-aware node-size tensor.
    """
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    clusters = {"cluster": [0, 1, 2]}
    node_sizes = torch.tensor([[80.0, 60.0], [80.0, 60.0], [80.0, 60.0]])
    pos = layout_cise_pipeline(
        edge_index=edge_index,
        num_nodes=3,
        node_sizes=node_sizes,
        clusters=clusters,
    )

    center = pos.mean(dim=0, keepdim=True)
    radii = torch.linalg.vector_norm(pos - center, dim=1)
    expected = (3.0 * (30.0**2 + 30.0**2) ** 0.5 + 3.0 * 12.5) / (2.0 * torch.pi)
    assert torch.allclose(radii, torch.full_like(radii, float(expected)), atol=1.0e-4)


def test_cise_relaxation_is_deterministic() -> None:
    """CiSE Steps 3-5 should be byte-identical for identical inputs.

    Returns
    -------
    None
        Repeated runs with the same seed/options must match exactly.
    """
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 0, 1, 2], [1, 2, 0, 4, 5, 3, 3, 4, 5]],
        dtype=torch.long,
    )
    clusters = {"left": [0, 1, 2], "right": [3, 4, 5]}

    first = layout_cise_pipeline(
        edge_index=edge_index,
        num_nodes=6,
        clusters=clusters,
        steps=75,
        seed=11,
    )
    repeated = layout_cise_pipeline(
        edge_index=edge_index,
        num_nodes=6,
        clusters=clusters,
        steps=75,
        seed=11,
    )

    assert torch.equal(first, repeated)


def test_cise_steps_zero_preserves_static_circle_placement() -> None:
    """Zero-step CiSE should preserve the Step 1/2 static placement.

    Returns
    -------
    None
        The compatibility path should still expose the pre-relaxation layout.
    """
    edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 3, 4]], dtype=torch.long)
    clusters = {"left": [0, 1, 2], "right": [3, 4, 5]}

    static = layout_cise_pipeline(edge_index=edge_index, num_nodes=6, clusters=clusters, steps=0)
    relaxed = layout_cise_pipeline(edge_index=edge_index, num_nodes=6, clusters=clusters, steps=60)

    assert static.shape == relaxed.shape == (6, 2)
    assert not torch.equal(static, relaxed)


def test_cise_relaxation_keeps_members_on_rigid_circles() -> None:
    """Relaxed CiSE clusters should remain rigid circular bodies.

    Returns
    -------
    None
        Within each cluster, all members keep the reference ring radius.
    """
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 0, 1, 2, 6], [1, 2, 0, 4, 5, 3, 3, 4, 5, 0]],
        dtype=torch.long,
    )
    clusters = {"left": [0, 1, 2], "right": [3, 4, 5], "single": [6]}
    pos = layout_cise_pipeline(edge_index=edge_index, num_nodes=7, clusters=clusters, steps=90)

    for members in (clusters["left"], clusters["right"]):
        member_pos = pos[torch.tensor(members)]
        center = member_pos.mean(dim=0, keepdim=True)
        radii = torch.linalg.vector_norm(member_pos - center, dim=1)
        assert torch.allclose(radii, radii.mean().expand_as(radii), atol=1.0e-4)
