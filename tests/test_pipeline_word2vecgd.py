"""Pipeline pins for the Word2VecGD graph layout."""

from __future__ import annotations

from pathlib import Path

import torch

import dagua
from dagua.config import LayoutConfig
from dagua.layout.engine import layout
from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.word2vecgd import (
    _adjacency_lists,
    _cosine_distance_targets,
    _pca_initial_positions,
    cosine_stress_sgd,
    generate_random_walks,
    layout_word2vecgd_pipeline,
    train_skipgram_embeddings,
)


def _edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    """Build a ``[2, E]`` edge tensor.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Source-target edge pairs.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _normalized_cosine_stress(positions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute the normalized stress objective used by Word2VecGD.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    targets : torch.Tensor
        Cosine-distance target matrix with shape ``[N, N]``.

    Returns
    -------
    float
        Scalar normalized stress.
    """
    num_nodes = positions.shape[0]
    mask = ~torch.eye(num_nodes, dtype=torch.bool)
    distances = torch.cdist(positions, positions)
    ratio = distances[mask] / torch.clamp(targets[mask], min=1.0e-4)
    alpha = torch.sum(ratio) / torch.clamp(torch.sum(ratio * ratio), min=1.0e-8)
    residual = (distances[mask] * alpha - targets[mask]) / torch.clamp(
        targets[mask],
        min=1.0e-4,
    )
    return float(0.5 * torch.sum(residual * residual))


def test_word2vecgd_pipeline_is_registered() -> None:
    """Register the Word2VecGD pipeline.

    Returns
    -------
    None
        Registry lookup must resolve the public Word2VecGD entrypoint.
    """
    assert PIPELINE_REGISTRY["word2vecgd"] == (
        "dagua.layout.ops.pipelines.word2vecgd",
        "layout_word2vecgd_pipeline",
    )
    assert get_pipeline_function("Word2VecGD") is layout_word2vecgd_pipeline


def test_word2vecgd_walk_sampling_matches_graphv_nn_rng_order() -> None:
    """Pin graphv_nn's Python-random walk sampling order.

    Returns
    -------
    None
        Walks must follow node-major iteration and ``random.choice`` neighbor
        selection.
    """
    adjacency = [[1, 2], [0, 2], [0, 1, 3], [2]]
    walks = generate_random_walks(adjacency, num_walks=2, walk_length=4, seed=7)

    assert walks == [
        [0, 2, 0, 2, 3],
        [0, 1, 0, 1, 2],
        [1, 0, 1, 0, 1],
        [1, 2, 1, 0, 1],
        [2, 0, 2, 0, 1],
        [2, 0, 1, 2, 0],
        [3, 2, 0, 1, 2],
        [3, 2, 0, 1, 2],
    ]


def test_word2vecgd_embeddings_are_seed_deterministic() -> None:
    """Verify deterministic skip-gram training.

    Returns
    -------
    None
        Identical seeds must reproduce the same embeddings, while a distinct
        seed changes the trained table.
    """
    edge_index = _edge_index([(0, 1), (1, 2), (2, 3), (3, 4)])
    adjacency = _adjacency_lists(edge_index, 5)
    walks = generate_random_walks(adjacency, num_walks=2, walk_length=4, seed=11)

    first = train_skipgram_embeddings(walks, 5, embedding_dim=4, epochs=2, seed=11)
    second = train_skipgram_embeddings(walks, 5, embedding_dim=4, epochs=2, seed=11)
    third = train_skipgram_embeddings(walks, 5, embedding_dim=4, epochs=2, seed=12)

    torch.testing.assert_close(torch.tensor(first), torch.tensor(second), rtol=0.0, atol=0.0)
    assert not torch.allclose(torch.tensor(first), torch.tensor(third))


def test_word2vecgd_cosine_stress_optimization_improves_objective() -> None:
    """Check cosine-stress SGD reduces its own objective.

    Returns
    -------
    None
        Optimized positions should improve over PCA initialization for a small
        deterministic embedding table.
    """
    edge_index = _edge_index([(0, 1), (1, 2), (2, 3), (1, 4), (4, 5)])
    adjacency = _adjacency_lists(edge_index, 6)
    walks = generate_random_walks(adjacency, num_walks=2, walk_length=4, seed=5)
    embeddings = train_skipgram_embeddings(walks, 6, embedding_dim=4, epochs=2, seed=5)
    targets = _cosine_distance_targets(embeddings)
    initial = _pca_initial_positions(embeddings, seed=5)
    optimized = cosine_stress_sgd(embeddings, steps=20, learning_rate=0.03, seed=5)

    assert _normalized_cosine_stress(optimized, targets) < _normalized_cosine_stress(
        initial,
        targets,
    )


def test_layout_config_algorithm_word2vecgd_works() -> None:
    """Exercise public engine dispatch for ``algorithm='word2vecgd'``.

    Returns
    -------
    None
        Dispatch must return finite ``[N, 2]`` coordinates.
    """
    graph = dagua.DaguaGraph.from_edge_list(
        [("a", "b"), ("b", "c"), ("c", "d"), ("b", "e"), ("e", "f")]
    )
    positions = layout(
        graph,
        LayoutConfig(
            algorithm="word2vecgd",
            seed=5,
            algorithm_params={
                "num_walks": 2,
                "walk_length": 4,
                "epochs": 2,
                "embedding_dim": 4,
                "steps": 5,
            },
        ),
    )

    assert positions.shape == (6, 2)
    assert torch.isfinite(positions).all()


def test_word2vecgd_production_pipeline_has_no_runtime_delegation() -> None:
    """Guard the production pipeline against graphv_nn/gensim delegation.

    Returns
    -------
    None
        Production source must not import the reference script or gensim.
    """
    source_path = (
        Path(__file__).parents[1] / "dagua" / "layout" / "ops" / "pipelines" / "word2vecgd.py"
    )
    source = source_path.read_text()
    assert "import graphv_nn" not in source
    assert "from graphv_nn" not in source
    assert "import gensim" not in source
    assert "from gensim" not in source
