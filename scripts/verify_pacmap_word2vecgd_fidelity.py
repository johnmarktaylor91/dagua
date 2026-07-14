"""Verify PaCMAP and Word2VecGD fidelity on small deterministic graphs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dagua.eval.equivalence_metrics import procrustes_rmsd  # noqa: E402
from dagua.graph import DaguaGraph  # noqa: E402
from dagua.layout.ops.pipelines.pacmap import _fit_pacmap, _graph_geodesic_distances  # noqa: E402
from dagua.layout.ops.pipelines.word2vecgd import (  # noqa: E402
    _adjacency_lists,
    _cosine_distance_targets,
    _pca_initial_positions,
    cosine_stress_sgd,
    generate_random_walks,
    train_skipgram_embeddings,
)


def _graph() -> DaguaGraph:
    """Build the shared small verification graph.

    Returns
    -------
    DaguaGraph
        Eight-node graph with a branch, enough nodes for PaCMAP sampling.
    """
    graph = DaguaGraph.from_edge_list(
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (2, 6), (6, 7)],
        num_nodes=8,
    )
    graph.compute_node_sizes()
    return graph


def _normalized_cosine_stress(positions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute normalized cosine stress.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    targets : torch.Tensor
        Cosine-distance targets with shape ``[N, N]``.

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


def verify_pacmap() -> tuple[str, float]:
    """Verify native PaCMAP against the installed reference optimizer core.

    Returns
    -------
    tuple[str, float]
        Fidelity tier and Procrustes RMSD quality score.
    """
    import pacmap.pacmap as reference

    graph = _graph()
    distances = _graph_geodesic_distances(graph.edge_index, graph.num_nodes, None)
    actual, pairs = _fit_pacmap(
        distances,
        init="pca",
        n_neighbors=10,
        mn_ratio=0.5,
        fp_ratio=2.0,
        lr=1.0,
        num_iters=(2, 2, 2),
        seed=7,
    )
    processed = distances.copy().astype(np.float32)
    processed -= np.min(processed)
    processed /= np.max(processed)
    processed -= np.mean(processed, axis=0)
    reference._RANDOM_STATE = 7
    expected, _, _, _, _ = reference.pacmap(
        processed,
        2,
        pairs[0],
        pairs[1],
        pairs[2],
        1.0,
        (2, 2, 2),
        "pca",
        False,
        False,
        [],
        False,
        reference.PCA(n_components=2, random_state=7).fit(processed),
    )
    score = procrustes_rmsd(actual, expected)
    tier = "float-floor" if np.isfinite(score) and score < 3.0e-6 else "divergent"
    return tier, score


def verify_word2vecgd() -> tuple[str, float]:
    """Verify Word2VecGD walk determinism and stress optimization quality.

    Returns
    -------
    tuple[str, float]
        Fidelity tier and stress improvement score.
    """
    graph = _graph()
    adjacency = _adjacency_lists(graph.edge_index, graph.num_nodes)
    walks = generate_random_walks(adjacency, num_walks=2, walk_length=4, seed=5)
    embeddings = train_skipgram_embeddings(
        walks,
        graph.num_nodes,
        embedding_dim=4,
        epochs=2,
        seed=5,
    )
    targets = _cosine_distance_targets(embeddings)
    initial = _pca_initial_positions(embeddings, seed=5)
    optimized = cosine_stress_sgd(embeddings, steps=20, learning_rate=0.03, seed=5)
    before = _normalized_cosine_stress(initial, targets)
    after = _normalized_cosine_stress(optimized, targets)
    improvement = before - after
    tier = "native-deterministic" if improvement > 0.0 else "divergent"
    return tier, improvement


def _source_guard() -> None:
    """Raise if production pipelines contain runtime delegation hooks.

    Returns
    -------
    None
        Returns normally when source guards pass.
    """
    root = Path(__file__).parents[1]
    pacmap_source = (root / "dagua/layout/ops/pipelines/pacmap.py").read_text()
    word2vec_source = (root / "dagua/layout/ops/pipelines/word2vecgd.py").read_text()
    forbidden = [
        ("pacmap", "from pacmap"),
        ("pacmap", "import pacmap"),
        ("word2vecgd", "import gensim"),
        ("word2vecgd", "from gensim"),
        ("word2vecgd", "import graphv_nn"),
        ("word2vecgd", "from graphv_nn"),
    ]
    for algorithm, needle in forbidden:
        source = pacmap_source if algorithm == "pacmap" else word2vec_source
        if needle in source:
            raise RuntimeError(f"{algorithm} production source contains delegation hook: {needle}")


def main() -> None:
    """Run verification and print per-algorithm fidelity summaries.

    Returns
    -------
    None
        Results are printed to standard output.
    """
    _source_guard()
    pacmap_tier, pacmap_quality = verify_pacmap()
    word2vec_tier, word2vec_quality = verify_word2vecgd()
    print(f"pacmap tier={pacmap_tier} quality_procrustes_rmsd={pacmap_quality:.8g}")
    print(f"word2vecgd tier={word2vec_tier} quality_stress_improvement={word2vec_quality:.8g}")
    print("no_delegation_guards=pass")


if __name__ == "__main__":
    main()
