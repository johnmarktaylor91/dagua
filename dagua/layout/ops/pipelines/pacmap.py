"""PaCMAP graph layout on shortest-path distance features."""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
import torch

from dagua.layout.ops.pipelines import resolve_fidelity_dtype
from dagua.layout.ops.pipelines.tsne_graph import _graph_geodesic_distances

_PACMAP_W_MN_INIT = 1000.0
_PACMAP_BETA1 = 0.9
_PACMAP_BETA2 = 0.999
_PACMAP_EPS = 1.0e-7


def _preprocess_features(
    features: np.ndarray,
    seed: Optional[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize PaCMAP features and build the PCA initialization basis.

    Parameters
    ----------
    features : numpy.ndarray
        Input feature matrix with shape ``[N, D]``.
    seed : int, optional
        Seed forwarded to sklearn PCA, matching the PaCMAP reference.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Normalized ``float32`` feature matrix and PCA coordinates with shape
        ``[N, 2]``.
    """
    from sklearn.decomposition import PCA

    x = np.asarray(features, dtype=np.float32).copy()
    xmin = float(np.min(x))
    x -= xmin
    xmax = float(np.max(x))
    if xmax != 0.0:
        x /= xmax
    xmean = np.mean(x, axis=0)
    x -= xmean
    pca = PCA(n_components=2, random_state=seed if seed is not None else 0)
    pca_coordinates = pca.fit_transform(x).astype(np.float32)
    return x, pca_coordinates


def _decide_num_pairs(
    num_nodes: int,
    n_neighbors: Optional[int],
    mn_ratio: float,
    fp_ratio: float,
) -> tuple[int, int, int]:
    """Resolve PaCMAP neighbor, mid-near, and further-pair counts.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    n_neighbors : int, optional
        Requested nearest-neighbor pair count per node.
    mn_ratio : float
        Ratio of mid-near pairs to nearest-neighbor pairs.
    fp_ratio : float
        Ratio of further pairs to nearest-neighbor pairs.

    Returns
    -------
    tuple[int, int, int]
        Counts ``(n_neighbors, n_MN, n_FP)`` after PaCMAP small-sample
        clamping and reorganization.

    Raises
    ------
    ValueError
        Raised when the graph is too small for PaCMAP pair sampling.
    """
    resolved_neighbors = 10 if n_neighbors is None and num_nodes <= 10_000 else n_neighbors
    if resolved_neighbors is None:
        resolved_neighbors = int(round(10 + 15 * (math.log10(num_nodes) - 4)))
    resolved_mn = int(round(resolved_neighbors * mn_ratio))
    resolved_fp = int(round(resolved_neighbors * fp_ratio))

    resolved_neighbors = min(resolved_neighbors, num_nodes - 1)
    resolved_fp = min(resolved_fp, num_nodes - 1 - resolved_neighbors)
    resolved_mn = min(resolved_mn, num_nodes - 1)
    if resolved_neighbors + resolved_mn + resolved_fp >= num_nodes:
        denominator = 1.0 + mn_ratio + fp_ratio
        resolved_neighbors = int(num_nodes / denominator)
        resolved_mn = int(num_nodes / denominator * mn_ratio)
        resolved_fp = int(num_nodes / denominator * fp_ratio)

    if resolved_neighbors < 1 or resolved_fp < 1:
        raise ValueError("PaCMAP requires at least one nearest-neighbor and further pair.")
    return resolved_neighbors, resolved_mn, resolved_fp


def _exact_neighbor_distances(
    features: np.ndarray,
    n_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact PaCMAP candidate neighbors from feature distances.

    Parameters
    ----------
    features : numpy.ndarray
        Normalized feature matrix with shape ``[N, D]``.
    n_neighbors : int
        Number of extra neighbor candidates requested by PaCMAP.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Neighbor indices and Euclidean distances, both shaped ``[N, K]``.
    """
    diff = features[:, None, :] - features[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=2, dtype=np.float32), dtype=np.float32)
    order = np.argsort(distances, axis=1, kind="stable")[:, 1 : n_neighbors + 1]
    row = np.arange(features.shape[0])[:, None]
    return order.astype(np.int32), distances[row, order].astype(np.float32)


def _scale_distances(
    knn_distances: np.ndarray,
    neighbors: np.ndarray,
) -> np.ndarray:
    """Scale candidate distances with PaCMAP's local sigma rule.

    Parameters
    ----------
    knn_distances : numpy.ndarray
        Candidate neighbor distances with shape ``[N, K]``.
    neighbors : numpy.ndarray
        Candidate neighbor indices with shape ``[N, K]``.

    Returns
    -------
    numpy.ndarray
        Scaled distance matrix with shape ``[N, K]``.
    """
    sigma_slice = knn_distances[:, 3:6]
    sigma = np.maximum(np.mean(sigma_slice, axis=1), 1.0e-10).astype(np.float32)
    return (knn_distances**2 / sigma[:, None] / sigma[neighbors]).astype(np.float32)


def _sample_neighbors_pair(
    scaled_distances: np.ndarray,
    neighbors: np.ndarray,
    n_neighbors: int,
) -> np.ndarray:
    """Sample PaCMAP nearest-neighbor pairs from scaled distances.

    Parameters
    ----------
    scaled_distances : numpy.ndarray
        Scaled candidate distances with shape ``[N, K]``.
    neighbors : numpy.ndarray
        Candidate neighbor indices with shape ``[N, K]``.
    n_neighbors : int
        Number of nearest-neighbor pairs per node.

    Returns
    -------
    numpy.ndarray
        Pair index matrix with shape ``[N * n_neighbors, 2]``.
    """
    num_nodes = int(neighbors.shape[0])
    pairs = np.empty((num_nodes * n_neighbors, 2), dtype=np.int32)
    for node in range(num_nodes):
        scaled_sort = np.argsort(scaled_distances[node], kind="quicksort")
        for pair_offset in range(n_neighbors):
            index = node * n_neighbors + pair_offset
            pairs[index, 0] = node
            pairs[index, 1] = neighbors[node, scaled_sort[pair_offset]]
    return pairs


def _legacy_sample_fp(
    n_samples: int,
    maximum: int,
    reject_ind: np.ndarray,
    self_ind: int,
) -> np.ndarray:
    """Sample PaCMAP further candidates using NumPy's legacy global RNG.

    Parameters
    ----------
    n_samples : int
        Number of distinct samples to draw.
    maximum : int
        Exclusive upper bound for sampled node IDs.
    reject_ind : numpy.ndarray
        Node IDs disallowed for this draw.
    self_ind : int
        Current node ID, also disallowed.

    Returns
    -------
    numpy.ndarray
        Sampled node IDs with shape ``[n_samples]``.
    """
    result = np.empty(n_samples, dtype=np.int32)
    reject = {int(value) for value in np.asarray(reject_ind, dtype=np.int32).tolist()}
    for sample_index in range(n_samples):
        while True:
            candidate = int(np.random.randint(maximum))
            if candidate == self_ind:
                continue
            if candidate in result[:sample_index]:
                continue
            if candidate in reject:
                continue
            result[sample_index] = candidate
            break
    return result


def _squared_feature_distance(features: np.ndarray, first: int, second: int) -> float:
    """Return squared Euclidean distance between two feature rows.

    Parameters
    ----------
    features : numpy.ndarray
        Feature matrix with shape ``[N, D]``.
    first : int
        First row index.
    second : int
        Second row index.

    Returns
    -------
    float
        Squared Euclidean distance in feature space.
    """
    delta = features[first] - features[second]
    return float(np.sum(delta * delta, dtype=np.float32))


def _sample_mid_near_pairs(
    features: np.ndarray,
    n_mn: int,
    seed: Optional[int],
) -> np.ndarray:
    """Sample deterministic PaCMAP mid-near pairs.

    Parameters
    ----------
    features : numpy.ndarray
        Normalized feature matrix with shape ``[N, D]``.
    n_mn : int
        Number of mid-near pairs per node.
    seed : int, optional
        PaCMAP random state. ``None`` preserves global RNG behavior.

    Returns
    -------
    numpy.ndarray
        Pair index matrix with shape ``[N * n_mn, 2]``.
    """
    num_nodes = int(features.shape[0])
    pairs = np.empty((num_nodes * n_mn, 2), dtype=np.int32)
    for node in range(num_nodes):
        for offset in range(n_mn):
            if seed is not None:
                np.random.seed(int(seed) + node * n_mn + offset)
            start = node * n_mn
            sampled = _legacy_sample_fp(6, num_nodes, pairs[start : start + offset, 1], node)
            sampled_distances = [
                _squared_feature_distance(features, node, int(candidate)) for candidate in sampled
            ]
            distances = np.array(
                sampled_distances,
                dtype=np.float32,
            )
            min_index = int(np.argmin(distances))
            sampled = np.delete(sampled, [min_index])
            distances = np.delete(distances, [min_index])
            picked = int(sampled[int(np.argmin(distances))])
            pairs[start + offset, 0] = node
            pairs[start + offset, 1] = picked
    return pairs


def _sample_further_pairs(
    num_nodes: int,
    pair_neighbors: np.ndarray,
    n_neighbors: int,
    n_fp: int,
    seed: Optional[int],
) -> np.ndarray:
    """Sample deterministic PaCMAP further pairs.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    pair_neighbors : numpy.ndarray
        Nearest-neighbor pair matrix with shape ``[N * n_neighbors, 2]``.
    n_neighbors : int
        Number of nearest-neighbor pairs per node.
    n_fp : int
        Number of further pairs per node.
    seed : int, optional
        PaCMAP random state. ``None`` preserves global RNG behavior.

    Returns
    -------
    numpy.ndarray
        Pair index matrix with shape ``[N * n_fp, 2]``.
    """
    pairs = np.empty((num_nodes * n_fp, 2), dtype=np.int32)
    for node in range(num_nodes):
        if seed is not None:
            np.random.seed(int(seed) + node)
        reject = pair_neighbors[node * n_neighbors : (node + 1) * n_neighbors, 1]
        sampled = _legacy_sample_fp(n_fp, num_nodes, reject, node)
        for offset in range(n_fp):
            pairs[node * n_fp + offset, 0] = node
            pairs[node * n_fp + offset, 1] = sampled[offset]
    return pairs


def _generate_pairs(
    features: np.ndarray,
    n_neighbors: int,
    n_mn: int,
    n_fp: int,
    seed: Optional[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate all PaCMAP pair classes.

    Parameters
    ----------
    features : numpy.ndarray
        Normalized feature matrix with shape ``[N, D]``.
    n_neighbors : int
        Nearest-neighbor pair count per node.
    n_mn : int
        Mid-near pair count per node.
    n_fp : int
        Further pair count per node.
    seed : int, optional
        PaCMAP random state.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Nearest-neighbor, mid-near, and further pairs.
    """
    n_neighbors_extra = min(n_neighbors + 50, features.shape[0] - 1)
    neighbors, knn_distances = _exact_neighbor_distances(features, n_neighbors_extra)
    scaled_distances = _scale_distances(knn_distances, neighbors)
    pair_neighbors = _sample_neighbors_pair(scaled_distances, neighbors, n_neighbors)
    pair_mn = _sample_mid_near_pairs(features, n_mn, seed)
    pair_fp = _sample_further_pairs(features.shape[0], pair_neighbors, n_neighbors, n_fp, seed)
    return pair_neighbors, pair_mn, pair_fp


def _find_weight(
    iteration: int,
    num_iters: tuple[int, int, int],
) -> tuple[float, float, float]:
    """Return PaCMAP's three-phase pair weights.

    Parameters
    ----------
    iteration : int
        Zero-based optimizer iteration.
    num_iters : tuple[int, int, int]
        Iteration counts for PaCMAP phases one, two, and three.

    Returns
    -------
    tuple[float, float, float]
        Weights for mid-near, nearest-neighbor, and further pairs.
    """
    phase_1_iters, phase_2_iters, _ = num_iters
    if iteration < phase_1_iters:
        progress = iteration / phase_1_iters
        return (1.0 - progress) * _PACMAP_W_MN_INIT + progress * 3.0, 2.0, 1.0
    if iteration < phase_1_iters + phase_2_iters:
        return 3.0, 3.0, 1.0
    return 0.0, 1.0, 1.0


def _pacmap_gradient(
    embedding: np.ndarray,
    pair_neighbors: np.ndarray,
    pair_mn: np.ndarray,
    pair_fp: np.ndarray,
    w_neighbors: float,
    w_mn: float,
    w_fp: float,
) -> np.ndarray:
    """Compute PaCMAP's pairwise gradient.

    Parameters
    ----------
    embedding : numpy.ndarray
        Current embedding with shape ``[N, 2]``.
    pair_neighbors : numpy.ndarray
        Nearest-neighbor pairs with shape ``[P, 2]``.
    pair_mn : numpy.ndarray
        Mid-near pairs with shape ``[P, 2]``.
    pair_fp : numpy.ndarray
        Further pairs with shape ``[P, 2]``.
    w_neighbors : float
        Nearest-neighbor attraction weight.
    w_mn : float
        Mid-near attraction weight.
    w_fp : float
        Further-pair repulsion weight.

    Returns
    -------
    numpy.ndarray
        Gradient matrix with shape ``[N, 2]``.
    """
    grad = np.zeros_like(embedding, dtype=np.float32)
    for pairs, numerator, denominator, sign in (
        (pair_neighbors, 20.0 * w_neighbors, 10.0, 1.0),
        (pair_mn, 20000.0 * w_mn, 10000.0, 1.0),
        (pair_fp, 2.0 * w_fp, 1.0, -1.0),
    ):
        for first, second in pairs:
            delta = embedding[int(first)] - embedding[int(second)]
            d_ij = np.float32(1.0 + np.sum(delta * delta, dtype=np.float32))
            weight = np.float32(numerator / (denominator + float(d_ij)) ** 2)
            grad[int(first)] += np.float32(sign) * weight * delta
            grad[int(second)] -= np.float32(sign) * weight * delta
    return grad


def _fit_pacmap(
    features: np.ndarray,
    init: Union[str, np.ndarray, None],
    n_neighbors: Optional[int],
    mn_ratio: float,
    fp_ratio: float,
    lr: float,
    num_iters: tuple[int, int, int],
    seed: Optional[int],
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Fit native PaCMAP on precomputed graph-distance features.

    Parameters
    ----------
    features : numpy.ndarray
        Feature matrix with shape ``[N, D]``.
    init : str or numpy.ndarray, optional
        PaCMAP initialization mode, ``"pca"`` or ``"random"``.
    n_neighbors : int, optional
        Nearest-neighbor pair count per node.
    mn_ratio : float
        Ratio of mid-near pairs to nearest-neighbor pairs.
    fp_ratio : float
        Ratio of further pairs to nearest-neighbor pairs.
    lr : float
        Adam learning rate.
    num_iters : tuple[int, int, int]
        Iteration counts for the three PaCMAP phases.
    seed : int, optional
        PaCMAP random state.

    Returns
    -------
    tuple[numpy.ndarray, tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]]
        Final embedding and sampled pair matrices.
    """
    processed, pca_coordinates = _preprocess_features(features, seed)
    num_nodes = int(processed.shape[0])
    resolved_neighbors, n_mn, n_fp = _decide_num_pairs(num_nodes, n_neighbors, mn_ratio, fp_ratio)
    pair_neighbors, pair_mn, pair_fp = _generate_pairs(
        processed,
        resolved_neighbors,
        n_mn,
        n_fp,
        seed,
    )

    if isinstance(init, np.ndarray):
        from sklearn import preprocessing

        scaler = preprocessing.StandardScaler().fit(init.astype(np.float32))
        embedding = (scaler.transform(init.astype(np.float32)) * 0.0001).astype(np.float32)
    elif init is None or init == "pca":
        embedding = (0.01 * pca_coordinates).astype(np.float32)
    elif init == "random":
        if seed is not None:
            np.random.seed(int(seed))
        embedding = (np.random.normal(size=(num_nodes, 2)).astype(np.float32) * 0.0001).astype(
            np.float32
        )
    else:
        raise ValueError("init must be None, 'pca', 'random', or a numpy.ndarray.")

    m = np.zeros_like(embedding, dtype=np.float32)
    v = np.zeros_like(embedding, dtype=np.float32)
    for iteration in range(sum(num_iters)):
        w_mn, w_neighbors, w_fp = _find_weight(iteration, num_iters)
        grad = _pacmap_gradient(
            embedding,
            pair_neighbors,
            pair_mn,
            pair_fp,
            w_neighbors,
            w_mn,
            w_fp,
        )
        lr_t = (
            lr
            * math.sqrt(1.0 - _PACMAP_BETA2 ** (iteration + 1))
            / (1.0 - _PACMAP_BETA1 ** (iteration + 1))
        )
        m += (1.0 - _PACMAP_BETA1) * (grad - m)
        v += (1.0 - _PACMAP_BETA2) * (grad * grad - v)
        embedding -= np.float32(lr_t) * m / (np.sqrt(v) + _PACMAP_EPS)
    return embedding.astype(np.float32), (pair_neighbors, pair_mn, pair_fp)


def layout_pacmap_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    n_neighbors: Optional[int] = 10,
    MN_ratio: float = 0.5,
    FP_ratio: float = 2.0,
    lr: float = 1.0,
    num_iters: tuple[int, int, int] = (100, 100, 250),
    init: Union[str, np.ndarray, None] = "pca",
    seed: Optional[int] = 42,
    edge_weights: Optional[torch.Tensor] = None,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run PaCMAP on graph geodesic distance features.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node sizes with shape ``[N, 2]``. Used only to preserve
        output device conventions.
    n_neighbors : int, optional
        PaCMAP nearest-neighbor pair count per node.
    MN_ratio : float, default=0.5
        Ratio of mid-near pairs to nearest-neighbor pairs.
    FP_ratio : float, default=2.0
        Ratio of further pairs to nearest-neighbor pairs.
    lr : float, default=1.0
        Adam learning rate.
    num_iters : tuple[int, int, int], default=(100, 100, 250)
        Iteration counts for the three PaCMAP phases.
    init : {"pca", "random"} or numpy.ndarray, optional
        Initial embedding mode.
    seed : int, optional
        Random seed.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.
    fidelity_dtype : torch.dtype, optional
        Output dtype override.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.
    """
    if num_nodes == 0:
        return torch.zeros((0, 2), dtype=resolve_fidelity_dtype(True, fidelity_dtype))
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=resolve_fidelity_dtype(True, fidelity_dtype))
    distances = _graph_geodesic_distances(edge_index, num_nodes, edge_weights)
    embedding, _ = _fit_pacmap(
        distances,
        init=init,
        n_neighbors=n_neighbors,
        mn_ratio=MN_ratio,
        fp_ratio=FP_ratio,
        lr=lr,
        num_iters=tuple(int(value) for value in num_iters),
        seed=seed,
    )
    device = node_sizes.device if node_sizes is not None else edge_index.device
    return torch.tensor(
        embedding,
        dtype=resolve_fidelity_dtype(True, fidelity_dtype),
        device=device,
    )


__all__ = [
    "_decide_num_pairs",
    "_fit_pacmap",
    "_generate_pairs",
    "_graph_geodesic_distances",
    "layout_pacmap_pipeline",
]
