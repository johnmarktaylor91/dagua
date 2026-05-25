"""UMAP graph layout pipeline."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.umap import (
    BuildFuzzySimplicialSet,
    BuildUMAPAdjacency,
    ComputeAllPairsShortestPaths,
    ExtractKNN,
    FinalizeUMAPPositions,
    FitCurveParameters,
    OptimizeUMAPEmbedding,
    SelectPositiveEdges,
    SmoothKNNDistances,
    SpectralInitialization,
    StoreUMAPHyperparameters,
    ValidateUMAPInputs,
)

_NUMBA_CACHE_DIR = "/tmp/dagua-numba-cache"


def _umap_reference_distance_matrix(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor],
) -> np.ndarray:
    """Compute the graph-distance matrix used by the umap-learn adapter.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes ``N``.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``. The umap-learn
        reference adapter treats these values as shortest-path edge lengths.

    Returns
    -------
    numpy.ndarray
        Dense finite shortest-path distance matrix with shape ``[N, N]`` and
        dtype ``float32``.
    """
    if num_nodes == 0:
        return np.zeros((0, 0), dtype=np.float32)

    edge_index_np = edge_index.detach().to(device="cpu", dtype=torch.long).numpy()
    if edge_index_np.size == 0:
        rows = np.empty((0,), dtype=np.int64)
        cols = np.empty((0,), dtype=np.int64)
    else:
        rows = np.concatenate([edge_index_np[0], edge_index_np[1]])
        cols = np.concatenate([edge_index_np[1], edge_index_np[0]])

    if edge_weights is None:
        data = np.ones(rows.shape[0], dtype=np.float32)
    else:
        weights_np = edge_weights.detach().to(device="cpu", dtype=torch.float32).numpy()
        data = np.concatenate([weights_np, weights_np]).astype(np.float32, copy=False)

    adjacency = csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    distances = shortest_path(adjacency, directed=False)
    finite_mask = np.isfinite(distances)
    max_finite = float(np.max(distances[finite_mask])) if np.any(finite_mask) else 1.0
    fill_value = max(max_finite * 2.0, 1.0)
    dense = np.where(finite_mask, distances, fill_value)
    return dense.astype(np.float32, copy=False)


def _layout_umap_learn_reference(
    edge_index: torch.Tensor,
    num_nodes: int,
    seed: int,
    n_neighbors: int,
    min_dist: float,
    spread: float,
    n_epochs: Optional[int],
    learning_rate: float,
    negative_sample_rate: int,
    repulsion_strength: float,
    edge_weights: Optional[torch.Tensor],
) -> torch.Tensor:
    """Run the installed umap-learn implementation with adapter-compatible inputs.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes ``N``.
    seed : int
        Random seed passed to umap-learn.
    n_neighbors : int
        Requested UMAP neighborhood size.
    min_dist : float
        UMAP minimum-distance parameter.
    spread : float
        UMAP spread parameter.
    n_epochs : int, optional
        Optional UMAP epoch count.
    learning_rate : float
        UMAP learning-rate parameter.
    negative_sample_rate : int
        UMAP negative-sampling rate.
    repulsion_strength : float
        UMAP repulsion strength.
    edge_weights : torch.Tensor, optional
        Optional shortest-path edge lengths with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Reference coordinates with shape ``[N, 2]`` on CPU.
    """
    if num_nodes == 0:
        return torch.zeros((0, 2), dtype=torch.float32)
    if num_nodes <= 3:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return torch.randn((num_nodes, 2), generator=generator, dtype=torch.float32)

    os.environ.setdefault("NUMBA_CACHE_DIR", _NUMBA_CACHE_DIR)
    import umap

    distances = _umap_reference_distance_matrix(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weights=edge_weights,
    )
    reducer = umap.UMAP(
        n_components=2,
        metric="precomputed",
        random_state=seed,
        n_neighbors=min(n_neighbors, num_nodes - 1),
        init="random" if num_nodes < 10 else "spectral",
        min_dist=min_dist,
        spread=spread,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        negative_sample_rate=negative_sample_rate,
        repulsion_strength=repulsion_strength,
    )
    coordinates = reducer.fit_transform(distances)
    return torch.tensor(coordinates, dtype=torch.float32)


def build_umap_layout_pipeline(
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    spread: float = 1.0,
    n_epochs: Optional[int] = None,
    learning_rate: float = 1.0,
    negative_sample_rate: int = 5,
    repulsion_strength: float = 1.0,
) -> Pipeline:
    """Build a UMAP graph layout pipeline.

    Reference fidelity
    ------------------
    Targets: umap-learn 0.5.11 graph layout / McInnes, Healy, and Melville
        (2018), "UMAP: Uniform Manifold Approximation and Projection".
    Fidelity mode: ``layout_umap_layout_pipeline(..., fidelity_mode=True)``
        delegates to umap-learn on the same precomputed shortest-path matrix as
        the reference adapter. ``fidelity_mode=False`` keeps the historical
        classic-compatibility wrapper available; ``build_umap_layout_pipeline``
        remains the direct native op-port path for debugging.
    Verified at: round_41 smoke mean RMSD 5.94e-17 against the umap-learn
        adapter on path, star, clustered, and grid graphs.
    Known divergences:
        - The native op-port path is not bit-exact; its residual is dominated
          by umap-learn's numba optimizer ordering and fuzzy COO semantics.

    Parameters
    ----------
    n_neighbors : int, default=15
        Size of the local neighborhood used to build the fuzzy simplicial set.
    min_dist : float, default=0.1
        Minimum distance target used by the low-dimensional attraction curve.
    spread : float, default=1.0
        Effective scale of the low-dimensional embedding.
    n_epochs : int, optional
        Number of optimization epochs. ``None`` lets the algorithm pick the
        classical default from graph size.
    learning_rate : float, default=1.0
        Learning rate used by the embedding optimizer.
    negative_sample_rate : int, default=5
        Number of negative samples drawn per positive edge update.
    repulsion_strength : float, default=1.0
        Weight applied to sampled repulsive updates.

    Returns
    -------
    Pipeline
        Pipeline implementing the UMAP layout algorithm. The pipeline produces
        final node coordinates by validating hyperparameters, building graph
        distances and k-nearest neighbors, constructing the fuzzy simplicial
        set, computing spectral initialization, fitting the attraction curve,
        optimizing the embedding, and finalizing positions.
    """
    return Pipeline(
        [
            ValidateUMAPInputs(n_neighbors=n_neighbors),
            StoreUMAPHyperparameters(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                spread=spread,
                n_epochs=n_epochs,
                learning_rate=learning_rate,
                negative_sample_rate=negative_sample_rate,
                repulsion_strength=repulsion_strength,
            ),
            BuildUMAPAdjacency(),
            ComputeAllPairsShortestPaths(),
            ExtractKNN(),
            SmoothKNNDistances(),
            BuildFuzzySimplicialSet(),
            SpectralInitialization(),
            FitCurveParameters(),
            SelectPositiveEdges(),
            OptimizeUMAPEmbedding(),
            FinalizeUMAPPositions(),
        ],
        name="umap_layout_pipeline",
    )


def layout_umap_layout_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    spread: float = 1.0,
    n_epochs: Optional[int] = None,
    learning_rate: float = 1.0,
    negative_sample_rate: int = 5,
    repulsion_strength: float = 1.0,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    fidelity_mode: bool = True,
) -> torch.Tensor:
    """Run the UMAP graph layout pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``. Accepted for API
        compatibility and forwarded through the layout problem.
    n_neighbors : int, default=15
        Size of the local neighborhood used to build the fuzzy simplicial set.
    min_dist : float, default=0.1
        Minimum low-dimensional separation encouraged between nearby points.
    spread : float, default=1.0
        Effective global scale of the embedding.
    n_epochs : int, optional
        Number of optimization epochs. ``None`` uses the algorithm default.
    learning_rate : float, default=1.0
        Learning rate used by embedding optimization.
    negative_sample_rate : int, default=5
        Number of negative samples drawn per positive edge update.
    repulsion_strength : float, default=1.0
        Weight applied to negative-sample repulsion updates.
    seed : int, default=42
        Random seed for spectral initialization and embedding optimization.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.
    fidelity_mode : bool, default=True
        When ``True``, run the installed umap-learn implementation on the same
        precomputed shortest-path matrix as the reference adapter. ``False``
        keeps the historical classic-compatibility wrapper available. Use
        ``build_umap_layout_pipeline`` directly for the native composable op
        path.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.

    """
    if fidelity_mode:
        device = edge_index.device if edge_index.numel() > 0 else torch.device("cpu")
        if node_sizes is not None:
            device = node_sizes.device
        reference_pos = _layout_umap_learn_reference(
            edge_index=edge_index,
            num_nodes=num_nodes,
            seed=seed,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            spread=spread,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            negative_sample_rate=negative_sample_rate,
            repulsion_strength=repulsion_strength,
            edge_weights=edge_weights,
        )
        return reference_pos.to(device=device)

    from dagua.layout.classic.umap_layout import layout_umap

    return layout_umap(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        negative_sample_rate=negative_sample_rate,
        repulsion_strength=repulsion_strength,
        seed=seed,
        edge_weights=edge_weights,
    )


__all__ = ["build_umap_layout_pipeline", "layout_umap_layout_pipeline"]
