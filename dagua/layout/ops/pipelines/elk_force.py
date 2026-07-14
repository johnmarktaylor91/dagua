"""ELK Force layout pipeline."""

from __future__ import annotations

from typing import Optional

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.elk_secondary import ElkForceConfig, layout_elk_force


def build_elk_force_pipeline() -> Pipeline:
    """Build the ELK Force marker pipeline.

    Returns
    -------
    Pipeline
        Empty marker pipeline; the public entrypoint composes the local ELK
        Force ops directly.
    """
    return Pipeline([], name="elk_force_pipeline")


def layout_elk_force_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: int = 1,
    iterations: int = 300,
    model: str = "eades",
    spacing: float = 80.0,
    repulsion: float = 5.0,
    temperature: float = 1.0,
    edge_weights: Optional[torch.Tensor] = None,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run the ELK Force pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Node sizes with shape ``[N, 2]``.
    seed : int, default=1
        ELK random seed for coincident-node jitter.
    iterations : int, default=300
        Number of model iterations.
    model : str, default="eades"
        Force model: ``"eades"`` or ``"fruchterman_reingold"``.
    spacing : float, default=80.0
        ELK node-node spacing.
    repulsion : float, default=5.0
        Eades repulsion factor.
    temperature : float, default=1.0
        FR temperature.
    edge_weights : torch.Tensor, optional
        Accepted for engine compatibility; ELK Force ignores weights.
    fidelity_dtype : torch.dtype, optional
        Optional output dtype. Defaults to ``torch.float64``.

    Returns
    -------
    torch.Tensor
        Node coordinates with shape ``[N, 2]``.
    """
    del edge_weights
    dtype = torch.float64 if fidelity_dtype is None else fidelity_dtype
    config = ElkForceConfig(
        iterations=iterations,
        model=model,
        spacing=spacing,
        repulsion=repulsion,
        temperature=temperature,
        seed=seed,
    )
    return layout_elk_force(edge_index, num_nodes, node_sizes, config=config, dtype=dtype)


__all__ = ["build_elk_force_pipeline", "layout_elk_force_pipeline"]
