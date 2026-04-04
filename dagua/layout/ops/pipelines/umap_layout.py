"""UMAP graph layout expressed as a composable ops pipeline."""

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
    """Build a UMAP pipeline that is bit-identical to ``layout_umap``."""
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
    """Run the UMAP pipeline as a drop-in replacement for classic ``layout_umap``."""
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
