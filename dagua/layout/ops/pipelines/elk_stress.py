"""ELK Stress layout pipeline."""

from __future__ import annotations

from typing import Optional

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.elk_secondary import layout_elk_stress


def build_elk_stress_pipeline() -> Pipeline:
    """Build the ELK Stress marker pipeline.

    Returns
    -------
    Pipeline
        Empty marker pipeline; the public entrypoint composes the local stress
        majorization op directly.
    """
    return Pipeline([], name="elk_stress_pipeline")


def layout_elk_stress_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: int = 1,
    desired_edge_length: float = 100.0,
    epsilon: float = 1.0e-4,
    iteration_limit: int = 200,
    edge_weights: Optional[torch.Tensor] = None,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run the ELK Stress pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Node sizes with shape ``[N, 2]``.
    seed : int, default=1
        Accepted for API consistency; ELK Stress is deterministic.
    desired_edge_length : float, default=100.0
        Desired edge length.
    epsilon : float, default=1e-4
        Relative stress-improvement threshold.
    iteration_limit : int, default=200
        Maximum majorization iterations.
    edge_weights : torch.Tensor, optional
        Accepted for engine compatibility; this ELK mode uses option lengths.
    fidelity_dtype : torch.dtype, optional
        Optional output dtype. Defaults to ``torch.float64``.

    Returns
    -------
    torch.Tensor
        Node coordinates with shape ``[N, 2]``.
    """
    del seed, edge_weights
    dtype = torch.float64 if fidelity_dtype is None else fidelity_dtype
    return layout_elk_stress(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        desired_edge_length=desired_edge_length,
        epsilon=epsilon,
        iteration_limit=iteration_limit,
        dtype=dtype,
    )


__all__ = ["build_elk_stress_pipeline", "layout_elk_stress_pipeline"]
