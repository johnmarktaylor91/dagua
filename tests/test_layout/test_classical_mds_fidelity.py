"""Regression tests for opt-in classical-MDS igraph fidelity behavior."""

from __future__ import annotations

import torch

from dagua.eval.competitors.igraph_competitor import IgraphMDS
from dagua.eval.variants import get_variant
from dagua.layout.ops.embed import _classical_mds_embedding
from dagua.layout.ops.pipelines.classical_mds import layout_classical_mds_pipeline


def _single_edge_index() -> torch.Tensor:
    """Build a two-node edge tensor.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, 1]``.
    """
    return torch.tensor([[0], [1]], dtype=torch.long)


def _path_edge_index(num_nodes: int) -> torch.Tensor:
    """Build a directed path edge tensor.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the path graph.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    return torch.tensor(
        [[index for index in range(num_nodes - 1)], [index + 1 for index in range(num_nodes - 1)]],
        dtype=torch.long,
    )


def test_igraph_fidelity_uses_two_node_special_case_and_scale() -> None:
    """igraph fidelity should match the reference two-node raw layout."""
    edge_index = _single_edge_index()

    positions = layout_classical_mds_pipeline(
        edge_index=edge_index,
        num_nodes=2,
        igraph_fidelity=True,
    )
    default_positions = layout_classical_mds_pipeline(edge_index=edge_index, num_nodes=2)

    torch.testing.assert_close(
        positions,
        torch.tensor([[0.0, 0.0], [50.0, 50.0]], dtype=torch.float32),
    )
    assert not torch.equal(default_positions, positions)


def test_igraph_fidelity_reverses_raw_embedding_dimensions() -> None:
    """igraph writes the largest selected eigenpair into the last dimension."""
    distances = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )

    default_positions = _classical_mds_embedding(distances)
    fidelity_positions = _classical_mds_embedding(distances, igraph_fidelity=True)

    assert float(default_positions[:, 0].abs().max().item()) > 0.0
    torch.testing.assert_close(default_positions[:, 1], torch.zeros(3))
    torch.testing.assert_close(fidelity_positions[:, 0], torch.zeros(3))
    torch.testing.assert_close(
        fidelity_positions[:, 1].abs(),
        default_positions[:, 0].abs(),
    )


def test_igraph_fidelity_ignores_edge_weights_like_igraph_mds() -> None:
    """igraph fidelity should not feed edge weights into shortest paths."""
    edge_index = _path_edge_index(3)
    edge_weights = torch.tensor([1.0, 8.0], dtype=torch.float32)

    weighted = layout_classical_mds_pipeline(
        edge_index=edge_index,
        num_nodes=3,
        edge_weights=edge_weights,
        igraph_fidelity=True,
    )
    unweighted = layout_classical_mds_pipeline(
        edge_index=edge_index,
        num_nodes=3,
        igraph_fidelity=True,
    )
    default_weighted = layout_classical_mds_pipeline(
        edge_index=edge_index,
        num_nodes=3,
        edge_weights=edge_weights,
    )

    torch.testing.assert_close(weighted, unweighted)
    assert not torch.equal(default_weighted, weighted)


def test_igraph_mds_metadata_marks_rng_usage() -> None:
    """igraph MDS should be seeded because disconnected component merge is random."""
    assert IgraphMDS.uses_igraph_rng is True


def test_igraph_fidelity_variant_is_registered_against_igraph_mds() -> None:
    """The opt-in fidelity path should be benchmark-addressable as a variant."""
    variant = get_variant("classic_classical_mds_igraph_fidelity")
    default_variant = get_variant("classic_classical_mds_default")

    assert variant is not None
    assert variant.reimpl_params == {"igraph_fidelity": True}
    assert variant.original_engine == "igraph_mds"
    assert variant.is_stochastic is True
    assert default_variant is not None
    assert default_variant.is_stochastic is True


def test_ogdf_fidelity_uses_path_special_case() -> None:
    """OGDF fidelity should return PivotMDS's raw path layout."""
    edge_index = _path_edge_index(4)

    positions = layout_classical_mds_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        ogdf_fidelity=True,
    )

    torch.testing.assert_close(
        positions,
        torch.tensor([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0], [300.0, 0.0]]),
    )


def test_classical_mds_fidelity_modes_are_mutually_exclusive() -> None:
    """Reference-specific fidelity modes should not be combined."""
    edge_index = _single_edge_index()

    try:
        layout_classical_mds_pipeline(
            edge_index=edge_index,
            num_nodes=2,
            igraph_fidelity=True,
            ogdf_fidelity=True,
        )
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("Expected mutually exclusive fidelity modes to fail.")
