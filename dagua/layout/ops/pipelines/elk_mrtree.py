"""ELK MrTree layout pipeline."""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.elk_secondary import layout_elk_mrtree


def build_elk_mrtree_pipeline() -> Pipeline:
    """Build the ELK MrTree marker pipeline.

    Returns
    -------
    Pipeline
        Empty marker pipeline; the public entrypoint composes the local tidy
        tree op directly.
    """
    return Pipeline([], name="elk_mrtree_pipeline")


def layout_elk_mrtree_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: int = 1,
    roots: Optional[Sequence[int]] = None,
    edge_weights: Optional[torch.Tensor] = None,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run the ELK MrTree pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Node sizes with shape ``[N, 2]``.
    seed : int, default=1
        Accepted for API consistency; MrTree is deterministic.
    roots : sequence[int] or None, default=None
        Optional explicit root order.
    edge_weights : torch.Tensor, optional
        Accepted for engine compatibility; MrTree ignores weights.
    fidelity_dtype : torch.dtype, optional
        Optional output dtype. Defaults to ``torch.float64``.

    Returns
    -------
    torch.Tensor
        Node coordinates with shape ``[N, 2]``.
    """
    del seed, edge_weights
    dtype = torch.float64 if fidelity_dtype is None else fidelity_dtype
    return layout_elk_mrtree(edge_index, num_nodes, node_sizes, roots=roots, dtype=dtype)


__all__ = ["build_elk_mrtree_pipeline", "layout_elk_mrtree_pipeline"]
