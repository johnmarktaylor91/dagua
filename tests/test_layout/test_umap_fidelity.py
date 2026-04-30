"""Regression tests for UMAP graph-fidelity behavior."""

from __future__ import annotations

import torch

from dagua.layout.ops.umap import _knn_from_distances


def test_knn_from_distances_counts_self_neighbor() -> None:
    """Verify precomputed UMAP kNN semantics count the self neighbor."""
    distances = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    indices, knn_distances = _knn_from_distances(distances=distances, n_neighbors=2)

    expected_indices = torch.tensor([[0, 1], [1, 0], [2, 1]], dtype=torch.long)
    expected_distances = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    assert torch.equal(indices, expected_indices)
    assert torch.equal(knn_distances, expected_distances)


def test_knn_from_distances_uses_stable_index_tie_order() -> None:
    """Verify tied graph distances retain ascending node-index order."""
    distances = torch.tensor(
        [
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    indices, _ = _knn_from_distances(distances=distances, n_neighbors=4)

    expected_indices = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 0, 2, 3],
            [2, 0, 1, 3],
            [3, 0, 1, 2],
        ],
        dtype=torch.long,
    )
    assert torch.equal(indices, expected_indices)
