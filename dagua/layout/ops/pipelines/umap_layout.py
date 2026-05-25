"""UMAP graph layout pipeline."""

from __future__ import annotations

from typing import Optional

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.state import (
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)
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
    Fidelity mode: no public flag; this pipeline incorporates reference-style
        smooth-kNN search, spectral initialization scaling, tiny-graph random
        policy, per-source Tausworthe negative sampling, and SciPy default
        curve-fit initialization from the fidelity rounds.
    Verified at: round_33 bounded subset median RMSD 0.195290 after leftovers;
        final 100-seed report marks UMAP variants partial match at median RMSD
        0.120 to 0.174.
    Known divergences:
        - Dagua still emits one undirected optimizer row per pair, while
          umap-learn optimizes both COO orientations after fuzzy union.
        - Tiny benchmark graphs exercise the random-init path more than the
          spectral changes.

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

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.

    Raises
    ------
    RuntimeError
        If the pipeline does not populate final positions.
    """
    if num_nodes <= 3:
        # The classic adapter already carries the tiny-graph normalization
        # behavior; keep this compatibility branch out of the general op path.
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

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_umap_layout_pipeline(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        negative_sample_rate=negative_sample_rate,
        repulsion_strength=repulsion_strength,
    ).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("UMAP pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_umap_layout_pipeline", "layout_umap_layout_pipeline"]
