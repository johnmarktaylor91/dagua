"""UMAP graph layout pipeline."""

from __future__ import annotations

from typing import Optional

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.pipelines import resolve_fidelity_dtype
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
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
    Fidelity mode: ``layout_umap_layout_pipeline(..., fidelity_mode=True)`` now
        runs this native port directly. It constructs precomputed
        shortest-path neighborhoods, the fuzzy simplicial set, spectral/random
        initialization, and the sampled Euclidean SGD loop without importing
        umap-learn.
    Verified at: round_62 smoke; see
        ``eval_output/algo_fidelity/round_62/umap_layout/SUMMARY.md``.
    Known divergences:
        - Tiny graphs with ``N <= 3`` use dagua's historical explicit
          fallbacks because umap-learn's spectral path is not defined there.

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
        UMAP default from graph size.
    learning_rate : float, default=1.0
        Learning rate used by the embedding optimizer.
    negative_sample_rate : int, default=5
        Number of negative samples drawn per positive edge update.
    repulsion_strength : float, default=1.0
        Weight applied to sampled repulsive updates.

    Returns
    -------
    Pipeline
        Pipeline implementing the UMAP layout algorithm.
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
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run the native UMAP graph layout pipeline.

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
        Optional edge-length tensor with shape ``[E]``.
    fidelity_mode : bool, default=True
        Compatibility flag retained for the public pipeline API. Both ``True``
        and ``False`` run the native port; the flag only influences the default
        output dtype through ``fidelity_dtype``.
    fidelity_dtype : torch.dtype, optional
        Optional output dtype. Defaults to ``torch.float64`` when
        ``fidelity_mode`` is truthy and ``torch.float32`` otherwise.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.
    """
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    pipeline = build_umap_layout_pipeline(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        negative_sample_rate=negative_sample_rate,
        repulsion_strength=repulsion_strength,
    )
    final_state = pipeline.apply(
        problem=problem,
        state=SolveState(),
        ctx=RuntimeContext(plan=ExecutionPlan(device=str(edge_index.device))),
    )
    if final_state.pos is None:
        raise RuntimeError("UMAP layout pipeline finished without positions.")

    return final_state.pos.to(dtype=resolve_fidelity_dtype(fidelity_mode, fidelity_dtype))


__all__ = ["build_umap_layout_pipeline", "layout_umap_layout_pipeline"]
