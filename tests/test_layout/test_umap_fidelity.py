"""Regression tests for UMAP graph-fidelity behavior."""

from __future__ import annotations

import torch

from dagua.layout.ops.umap import (
    _build_undirected_adjacency,
    _knn_from_distances,
    _optimize_embedding,
)


def test_build_undirected_adjacency_uses_weights_as_distances() -> None:
    """Verify weighted UMAP graph distances match the reference adapter."""
    edge_index = torch.tensor([[0, 0], [1, 1]], dtype=torch.long)
    edge_weights = torch.tensor([2.0, 3.0], dtype=torch.float32)

    adjacency = _build_undirected_adjacency(
        edge_index=edge_index,
        num_nodes=2,
        edge_weights=edge_weights,
    )

    assert adjacency == [[(1, 5.0)], [(0, 5.0)]]


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


def test_optimize_embedding_waits_until_first_sample_interval() -> None:
    """Verify UMAP SGD does not perform reference-forbidden epoch-zero updates."""
    embedding = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)

    optimized = _optimize_embedding(
        embedding=embedding.clone(),
        head=torch.tensor([0], dtype=torch.long),
        tail=torch.tensor([1], dtype=torch.long),
        epochs_per_sample=torch.tensor([1.0], dtype=torch.float32),
        n_epochs=1,
        learning_rate=1.0,
        negative_sample_rate=0,
        gamma=1.0,
        a=1.0,
        b=1.0,
        seed=42,
    )

    assert torch.equal(optimized, embedding)
