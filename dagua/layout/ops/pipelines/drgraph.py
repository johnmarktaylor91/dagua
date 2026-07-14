"""DRGraph graph layout pipeline."""

from __future__ import annotations

from typing import Optional

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.largevis import (
    DRGraphBuildSimilarity,
    DRGraphConfig,
    LargeVisOptimizeEmbedding,
)
from dagua.layout.ops.pipelines import resolve_fidelity_dtype
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def build_drgraph_pipeline(
    samples: Optional[int] = None,
    alpha: float = 1.0,
    negative_samples: int = 5,
    gamma: float = 0.01,
    a: float = -1.0,
    b: float = -1.0,
    multilevel: bool = True,
    seed: int = 314159265,
    dtype: torch.dtype = torch.float32,
) -> Pipeline:
    """Build the DRGraph layout pipeline.

    Parameters
    ----------
    samples : int or None, default=None
        Number of positive-edge SGD samples. ``None`` uses a bounded native
        equivalent of DRGraph's node-count sample ratio.
    alpha : float, default=1.0
        Initial learning rate.
    negative_samples : int, default=5
        Number of negative samples per positive edge.
    gamma : float, default=0.01
        Negative-sample repulsion weight.
    a : float, default=-1.0
        DRGraph curve parameter A. Values ``<= 0`` use the source fallback.
    b : float, default=-1.0
        DRGraph curve parameter B.
    multilevel : bool, default=True
        Retained for API fidelity. This native port runs deterministic
        single-level optimization.
    seed : int, default=314159265
        Deterministic seed. The C++ reference hard-codes this GSL seed.
    dtype : torch.dtype, default=torch.float32
        Internal and output coordinate dtype.

    Returns
    -------
    Pipeline
        Composable DRGraph layout pipeline.
    """
    config = DRGraphConfig(
        samples=samples,
        alpha=alpha,
        negative_samples=negative_samples,
        gamma=gamma,
        seed=seed,
        dtype=dtype,
        a=a,
        b=b,
        multilevel=multilevel,
    )
    curve = (a, b) if a > 0.0 and b > 0.0 else None
    return Pipeline(
        [
            DRGraphBuildSimilarity(),
            LargeVisOptimizeEmbedding(config=config, drgraph_ab=curve),
        ],
        name="drgraph_pipeline",
    )


def layout_drgraph_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    samples: Optional[int] = None,
    alpha: float = 1.0,
    negative_samples: int = 5,
    gamma: float = 0.01,
    a: float = -1.0,
    b: float = -1.0,
    multilevel: bool = True,
    seed: int = 314159265,
    edge_weights: Optional[torch.Tensor] = None,
    fidelity_mode: bool = True,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run the native DRGraph graph layout pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N``.
    node_sizes : torch.Tensor, optional
        Node sizes with shape ``[N, 2]``. Accepted for API compatibility.
    samples : int or None, default=None
        Number of positive-edge SGD samples.
    alpha : float, default=1.0
        Initial learning rate.
    negative_samples : int, default=5
        Number of negative samples per positive edge.
    gamma : float, default=0.01
        Negative-sample repulsion weight.
    a : float, default=-1.0
        DRGraph curve parameter A.
    b : float, default=-1.0
        DRGraph curve parameter B.
    multilevel : bool, default=True
        Retained for API fidelity.
    seed : int, default=314159265
        Deterministic seed for initialization and sampling.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``. Retained for dispatch
        compatibility; graph mode uses topology distances.
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
    pipeline = build_drgraph_pipeline(
        samples=samples,
        alpha=alpha,
        negative_samples=negative_samples,
        gamma=gamma,
        a=a,
        b=b,
        multilevel=multilevel,
        seed=seed,
        dtype=dtype,
    )
    state = pipeline.apply(
        problem=problem,
        state=SolveState(),
        ctx=RuntimeContext(plan=ExecutionPlan(device=str(edge_index.device))),
    )
    if state.pos is None:
        raise RuntimeError("DRGraph layout pipeline finished without positions.")
    return state.pos.to(dtype=dtype)


__all__ = ["build_drgraph_pipeline", "layout_drgraph_pipeline"]
