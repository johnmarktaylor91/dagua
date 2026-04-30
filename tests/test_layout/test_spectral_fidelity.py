"""Regression tests for opt-in spectral NetworkX fidelity mode."""

from __future__ import annotations

import numpy as np
import torch

from dagua.eval.variants import get_variant
from dagua.layout.ops.embed import _select_embedding_columns
from dagua.layout.ops.pipelines.spectral import layout_spectral_pipeline


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
        Number of graph nodes.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    return torch.tensor(
        [[index for index in range(num_nodes - 1)], [index + 1 for index in range(num_nodes - 1)]],
        dtype=torch.long,
    )


def test_networkx_fidelity_collapses_two_node_graph_to_center() -> None:
    """NetworkX fidelity should match the reference two-node special case."""
    edge_index = _single_edge_index()

    positions = layout_spectral_pipeline(
        edge_index=edge_index,
        num_nodes=2,
        networkx_fidelity=True,
    )
    default_positions = layout_spectral_pipeline(edge_index=edge_index, num_nodes=2)

    assert torch.equal(positions, torch.zeros((2, 2), dtype=torch.float32))
    assert not torch.equal(default_positions, positions)


def test_networkx_fidelity_forces_unnormalized_laplacian() -> None:
    """NetworkX fidelity should override Dagua's symmetric default Laplacian."""
    edge_index = _path_edge_index(5)

    fidelity_positions = layout_spectral_pipeline(
        edge_index=edge_index,
        num_nodes=5,
        normalization="symmetric",
        networkx_fidelity=True,
    )
    unnormalized_positions = layout_spectral_pipeline(
        edge_index=edge_index,
        num_nodes=5,
        normalization="unnormalized",
        networkx_fidelity=True,
    )

    assert torch.equal(fidelity_positions, unnormalized_positions)


def test_networkx_fidelity_selects_sorted_slice_after_first_eigenvector() -> None:
    """NetworkX fidelity should keep extra zero modes after skipping index zero."""
    eigenvalues = np.array([0.0, 0.0, 2.0, 3.0], dtype=np.float64)
    eigenvectors = np.eye(4, dtype=np.float64)

    selected = _select_embedding_columns(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        dim=2,
        skip_first=True,
    )
    robust_selected = _select_embedding_columns(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        dim=2,
    )

    np.testing.assert_array_equal(selected, eigenvectors[:, [1, 2]])
    np.testing.assert_array_equal(robust_selected, eigenvectors[:, [2, 3]])


def test_networkx_fidelity_variant_is_registered_against_nx_spectral() -> None:
    """The opt-in fidelity path should be benchmark-addressable as a variant."""
    variant = get_variant("classic_spectral_nx_fidelity")

    assert variant is not None
    assert variant.reimpl_params == {"networkx_fidelity": True}
    assert variant.original_engine == "nx_spectral"
