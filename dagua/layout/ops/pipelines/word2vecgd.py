"""Word2VecGD graph layout using random-walk skip-gram embeddings."""

from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np
import torch

from dagua.layout.ops.pipelines import resolve_fidelity_dtype

_EPSILON = 1.0e-8


def _adjacency_lists(edge_index: torch.Tensor, num_nodes: int) -> list[list[int]]:
    """Build deterministic undirected adjacency lists.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    list[list[int]]
        Sorted neighbor lists indexed by node ID.
    """
    neighbors: list[set[int]] = [set() for _ in range(num_nodes)]
    edge_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    for source, target in edge_cpu.t().tolist():
        if 0 <= int(source) < num_nodes and 0 <= int(target) < num_nodes and source != target:
            neighbors[int(source)].add(int(target))
            neighbors[int(target)].add(int(source))
    return [sorted(values) for values in neighbors]


def generate_random_walks(
    adjacency: list[list[int]],
    num_walks: int = 10,
    walk_length: int = 10,
    seed: Optional[int] = 42,
) -> list[list[int]]:
    """Generate graphv_nn-style random walks.

    Parameters
    ----------
    adjacency : list[list[int]]
        Neighbor lists indexed by node ID.
    num_walks : int, default=10
        Number of walks started at each node.
    walk_length : int, default=10
        Number of random transitions attempted per walk.
    seed : int, optional
        Python ``random`` seed. ``None`` uses the module-level RNG.

    Returns
    -------
    list[list[int]]
        Random walks as node-ID sequences. This mirrors ``graphv_nn`` by
        iterating nodes in order and using ``random.choice`` at each step.
    """
    rng = random if seed is None else random.Random(seed)
    walks: list[list[int]] = []
    for node in range(len(adjacency)):
        for _ in range(num_walks):
            walk = [node]
            current = node
            for _ in range(walk_length):
                neighbors = adjacency[current]
                if not neighbors:
                    break
                current = int(rng.choice(neighbors))
                walk.append(current)
            walks.append(walk)
    return walks


def _sigmoid(value: float) -> float:
    """Return a numerically stable scalar sigmoid.

    Parameters
    ----------
    value : float
        Input value.

    Returns
    -------
    float
        Sigmoid value in ``[0, 1]``.
    """
    if value >= 0.0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _skipgram_pairs(walks: list[list[int]], window: int) -> list[tuple[int, int]]:
    """Extract skip-gram center-context pairs from walks.

    Parameters
    ----------
    walks : list[list[int]]
        Random-walk corpus.
    window : int
        Maximum context radius.

    Returns
    -------
    list[tuple[int, int]]
        Ordered center-context training pairs.
    """
    pairs: list[tuple[int, int]] = []
    for walk in walks:
        for center_index, center in enumerate(walk):
            start = max(0, center_index - window)
            stop = min(len(walk), center_index + window + 1)
            for context_index in range(start, stop):
                if context_index != center_index:
                    pairs.append((int(center), int(walk[context_index])))
    return pairs


def train_skipgram_embeddings(
    walks: list[list[int]],
    num_nodes: int,
    embedding_dim: int = 8,
    window: int = 5,
    epochs: int = 10,
    negative_samples: int = 5,
    learning_rate: float = 0.025,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """Train deterministic word2vec-style node embeddings.

    Parameters
    ----------
    walks : list[list[int]]
        Random-walk corpus.
    num_nodes : int
        Number of graph nodes.
    embedding_dim : int, default=8
        Embedding dimension.
    window : int, default=5
        Skip-gram context radius.
    epochs : int, default=10
        Number of corpus passes.
    negative_samples : int, default=5
        Number of negative samples per positive pair.
    learning_rate : float, default=0.025
        Initial SGD learning rate.
    seed : int, optional
        NumPy seed for initialization, shuffling, and negative sampling.

    Returns
    -------
    numpy.ndarray
        Trained input embeddings with shape ``[N, embedding_dim]``.
    """
    rng = np.random.RandomState(seed if seed is not None else None)
    input_vectors = rng.uniform(
        -0.5 / embedding_dim,
        0.5 / embedding_dim,
        (num_nodes, embedding_dim),
    )
    output_vectors = np.zeros((num_nodes, embedding_dim), dtype=np.float64)
    pairs = _skipgram_pairs(walks, window)
    if not pairs:
        return input_vectors.astype(np.float32)

    counts = np.ones(num_nodes, dtype=np.float64)
    for walk in walks:
        for node in walk:
            counts[int(node)] += 1.0
    negative_probabilities = counts**0.75
    negative_probabilities /= np.sum(negative_probabilities)

    total_updates = max(len(pairs) * epochs, 1)
    update_index = 0
    for _ in range(epochs):
        order = rng.permutation(len(pairs))
        for pair_index in order:
            center, context = pairs[int(pair_index)]
            progress = update_index / total_updates
            alpha = max(learning_rate * (1.0 - progress), learning_rate * 0.0001)
            _update_skipgram_pair(input_vectors, output_vectors, center, context, 1, alpha)
            negatives = rng.choice(
                num_nodes,
                size=negative_samples,
                replace=True,
                p=negative_probabilities,
            )
            for negative in negatives:
                _update_skipgram_pair(
                    input_vectors,
                    output_vectors,
                    center,
                    int(negative),
                    0,
                    alpha,
                )
            update_index += 1
    return input_vectors.astype(np.float32)


def _update_skipgram_pair(
    input_vectors: np.ndarray,
    output_vectors: np.ndarray,
    center: int,
    context: int,
    label: int,
    learning_rate: float,
) -> None:
    """Apply one negative-sampling skip-gram update.

    Parameters
    ----------
    input_vectors : numpy.ndarray
        Input embedding table with shape ``[N, D]``.
    output_vectors : numpy.ndarray
        Output embedding table with shape ``[N, D]``.
    center : int
        Center token index.
    context : int
        Context or negative token index.
    label : int
        ``1`` for positive examples and ``0`` for negative examples.
    learning_rate : float
        SGD learning rate for this update.

    Returns
    -------
    None
        Embedding tables are updated in place.
    """
    input_copy = input_vectors[center].copy()
    score = _sigmoid(float(np.dot(input_copy, output_vectors[context])))
    gradient = (float(label) - score) * learning_rate
    input_vectors[center] += gradient * output_vectors[context]
    output_vectors[context] += gradient * input_copy


def _cosine_distance_targets(embeddings: np.ndarray) -> torch.Tensor:
    """Convert embeddings into positive cosine-distance targets.

    Parameters
    ----------
    embeddings : numpy.ndarray
        Node embeddings with shape ``[N, D]``.

    Returns
    -------
    torch.Tensor
        Cosine distance matrix with shape ``[N, N]``.
    """
    emb = torch.tensor(embeddings, dtype=torch.float64)
    emb = emb / torch.clamp(torch.linalg.norm(emb, dim=1, keepdim=True), min=_EPSILON)
    similarity = torch.clamp(emb @ emb.t(), min=-1.0, max=1.0)
    targets = torch.clamp(1.0 - similarity, min=1.0e-4)
    targets.fill_diagonal_(0.0)
    return targets


def _pca_initial_positions(embeddings: np.ndarray, seed: Optional[int]) -> torch.Tensor:
    """Initialize two-dimensional positions from embedding PCA.

    Parameters
    ----------
    embeddings : numpy.ndarray
        Node embeddings with shape ``[N, D]``.
    seed : int, optional
        Fallback random seed used when the embedding rank is degenerate.

    Returns
    -------
    torch.Tensor
        Initial positions with shape ``[N, 2]``.
    """
    centered = embeddings.astype(np.float64) - np.mean(embeddings, axis=0, keepdims=True)
    if np.linalg.norm(centered) < _EPSILON:
        rng = np.random.RandomState(seed if seed is not None else None)
        return torch.tensor(
            rng.normal(scale=1.0e-4, size=(embeddings.shape[0], 2)),
            dtype=torch.float64,
        )
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T
    coords = centered @ basis
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(coords.shape[0], dtype=np.float64)])
    return torch.tensor(coords[:, :2], dtype=torch.float64)


def cosine_stress_sgd(
    embeddings: np.ndarray,
    steps: int = 200,
    learning_rate: float = 0.05,
    seed: Optional[int] = 42,
) -> torch.Tensor:
    """Place node embeddings by normalized cosine-stress optimization.

    Parameters
    ----------
    embeddings : numpy.ndarray
        Node embeddings with shape ``[N, D]``.
    steps : int, default=200
        Adam optimization steps.
    learning_rate : float, default=0.05
        Adam learning rate.
    seed : int, optional
        Seed for degenerate initialization fallback.

    Returns
    -------
    torch.Tensor
        Optimized positions with shape ``[N, 2]`` and dtype ``float64``.
    """
    num_nodes = int(embeddings.shape[0])
    if num_nodes <= 1:
        return torch.zeros((num_nodes, 2), dtype=torch.float64)
    targets = _cosine_distance_targets(embeddings)
    positions = _pca_initial_positions(embeddings, seed).requires_grad_(True)
    optimizer = torch.optim.Adam([positions], lr=learning_rate)
    mask = ~torch.eye(num_nodes, dtype=torch.bool)
    for _ in range(max(int(steps), 0)):
        optimizer.zero_grad()
        distances = torch.cdist(positions, positions)
        ratio = distances[mask] / torch.clamp(targets[mask], min=1.0e-4)
        alpha = torch.sum(ratio) / torch.clamp(torch.sum(ratio * ratio), min=_EPSILON)
        residual = (distances[mask] * alpha - targets[mask]) / torch.clamp(
            targets[mask],
            min=1.0e-4,
        )
        loss = 0.5 * torch.sum(residual * residual)
        loss.backward()
        optimizer.step()
    return positions.detach()


def layout_word2vecgd_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    embedding_dim: int = 8,
    num_walks: int = 10,
    walk_length: int = 10,
    window: int = 5,
    epochs: int = 10,
    negative_samples: int = 5,
    embedding_lr: float = 0.025,
    layout_lr: float = 0.05,
    steps: int = 200,
    seed: Optional[int] = 42,
    edge_weights: Optional[torch.Tensor] = None,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run Word2VecGD random-walk embedding and cosine-stress placement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node sizes with shape ``[N, 2]``. Used only for output device.
    embedding_dim : int, default=8
        Skip-gram embedding dimension.
    num_walks : int, default=10
        Number of random walks per node.
    walk_length : int, default=10
        Number of transitions per random walk.
    window : int, default=5
        Skip-gram context radius.
    epochs : int, default=10
        Skip-gram corpus passes.
    negative_samples : int, default=5
        Negative samples per positive pair.
    embedding_lr : float, default=0.025
        Skip-gram learning rate.
    layout_lr : float, default=0.05
        Cosine-stress Adam learning rate.
    steps : int, default=200
        Cosine-stress optimization steps.
    seed : int, optional
        Random seed for walks, embedding training, and fallback initialization.
    edge_weights : torch.Tensor, optional
        Accepted for layout API compatibility; Word2VecGD uses unweighted walks.
    fidelity_dtype : torch.dtype, optional
        Output dtype override.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.
    """
    del edge_weights
    if num_nodes <= 1:
        dtype = resolve_fidelity_dtype(True, fidelity_dtype)
        return torch.zeros((num_nodes, 2), dtype=dtype, device=edge_index.device)
    adjacency = _adjacency_lists(edge_index, num_nodes)
    walks = generate_random_walks(
        adjacency,
        num_walks=num_walks,
        walk_length=walk_length,
        seed=seed,
    )
    embeddings = train_skipgram_embeddings(
        walks,
        num_nodes,
        embedding_dim=embedding_dim,
        window=window,
        epochs=epochs,
        negative_samples=negative_samples,
        learning_rate=embedding_lr,
        seed=seed,
    )
    positions = cosine_stress_sgd(
        embeddings,
        steps=steps,
        learning_rate=layout_lr,
        seed=seed,
    )
    device = node_sizes.device if node_sizes is not None else edge_index.device
    return positions.to(dtype=resolve_fidelity_dtype(True, fidelity_dtype), device=device)


__all__ = [
    "cosine_stress_sgd",
    "generate_random_walks",
    "layout_word2vecgd_pipeline",
    "train_skipgram_embeddings",
]
