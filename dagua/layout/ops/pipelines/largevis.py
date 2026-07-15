"""LargeVis graph layout pipeline."""

from __future__ import annotations

from typing import Optional

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.largevis import (
    LargeVisBuildSimilarity,
    LargeVisConfig,
    LargeVisOptimizeEmbedding,
)
from dagua.layout.ops.pipelines import resolve_fidelity_dtype
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def build_largevis_pipeline(
    n_neighbors: int = 150,
    samples: Optional[int] = None,
    alpha: float = 1.0,
    negative_samples: int = 5,
    gamma: float = 7.0,
    perplexity: float = 50.0,
    seed: int = 314159265,
    dtype: torch.dtype = torch.float32,
) -> Pipeline:
    """Build the LargeVis layout pipeline.

    Parameters
    ----------
    n_neighbors : int, default=150
        Number of geodesic neighbors used to build the sparse similarity graph.
    samples : int or None, default=None
        Number of positive-edge SGD samples. ``None`` uses a bounded native
        equivalent of the C++ million-sample heuristic.
    alpha : float, default=1.0
        Initial learning rate.
    negative_samples : int, default=5
        Number of negative samples per positive edge.
    gamma : float, default=7.0
        Negative-sample repulsion weight.
    perplexity : float, default=50.0
        Target perplexity for row-wise Gaussian similarity calibration.
    seed : int, default=314159265
        Deterministic seed. The C++ reference hard-codes this GSL seed.
    dtype : torch.dtype, default=torch.float32
        Internal and output coordinate dtype.

    Returns
    -------
    Pipeline
        Composable LargeVis layout pipeline.
    """
    config = LargeVisConfig(
        n_neighbors=n_neighbors,
        samples=samples,
        alpha=alpha,
        negative_samples=negative_samples,
        gamma=gamma,
        perplexity=perplexity,
        seed=seed,
        dtype=dtype,
    )
    return Pipeline(
        [
            LargeVisBuildSimilarity(n_neighbors=n_neighbors, perplexity=perplexity),
            LargeVisOptimizeEmbedding(config=config),
        ],
        name="largevis_pipeline",
    )


def layout_largevis_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    n_neighbors: int = 150,
    samples: Optional[int] = None,
    alpha: float = 1.0,
    negative_samples: int = 5,
    gamma: float = 7.0,
    perplexity: float = 50.0,
    seed: int = 314159265,
    edge_weights: Optional[torch.Tensor] = None,
    fidelity_mode: bool = True,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run the native LargeVis graph layout pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N``.
    node_sizes : torch.Tensor, optional
        Node sizes with shape ``[N, 2]``. Accepted for API compatibility.
    n_neighbors : int, default=150
        Number of geodesic neighbors used to build the sparse similarity graph.
    samples : int or None, default=None
        Number of positive-edge SGD samples.
    alpha : float, default=1.0
        Initial learning rate.
    negative_samples : int, default=5
        Number of negative samples per positive edge.
    gamma : float, default=7.0
        Negative-sample repulsion weight.
    perplexity : float, default=50.0
        Target perplexity for row-wise Gaussian similarity calibration.
    seed : int, default=314159265
        Deterministic seed for initialization and sampling.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``. Retained for dispatch
        compatibility; LargeVis graph mode uses unweighted graph geodesics.
    fidelity_mode : bool, default=True
        Compatibility flag controlling default output dtype.
    fidelity_dtype : torch.dtype, optional
        Optional output dtype.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    dtype = resolve_fidelity_dtype(fidelity_mode, fidelity_dtype)
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    pipeline = build_largevis_pipeline(
        n_neighbors=n_neighbors,
        samples=samples,
        alpha=alpha,
        negative_samples=negative_samples,
        gamma=gamma,
        perplexity=perplexity,
        seed=seed,
        dtype=dtype,
    )
    state = pipeline.apply(
        problem=problem,
        state=SolveState(),
        ctx=RuntimeContext(plan=ExecutionPlan(device=str(edge_index.device))),
    )
    if state.pos is None:
        raise RuntimeError("LargeVis layout pipeline finished without positions.")
    return state.pos.to(dtype=dtype)


__all__ = ["build_largevis_pipeline", "layout_largevis_pipeline"]
