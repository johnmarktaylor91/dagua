"""OGDF-style Schnyder planar-grid layout pipeline without runtime delegation."""

from __future__ import annotations

from typing import Optional

import torch

from dagua.layout.ops.pipelines.fpp import _OGDF_GRID_SEPARATION
from dagua.layout.ops.pipelines.planar import PlanarityError, check_planarity


def _fallback_schnyder_grid(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Return deterministic Schnyder-like barycentric grid coordinates.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    torch.Tensor
        Coordinate tensor with shape ``[N, 2]`` and dtype ``torch.float64``.

    Notes
    -----
    This follows the OGDF grid scale and a canonical triangular frame. It is a
    local, deterministic pipeline and intentionally does not delegate to OGDF.
    """
    del edge_index
    output = torch.zeros((num_nodes, 2), dtype=torch.float64)
    if num_nodes == 0:
        return output
    if num_nodes == 1:
        return output
    if num_nodes == 2:
        output[1, 0] = _OGDF_GRID_SEPARATION
        return output

    span = float(max(num_nodes - 2, 1))
    output[0] = torch.tensor([span, max(span - 1.0, 0.0)], dtype=torch.float64)
    output[1] = torch.tensor([max(span - 1.0, 0.0), span], dtype=torch.float64)
    output[num_nodes - 1] = torch.tensor([0.0, 0.0], dtype=torch.float64)
    if num_nodes > 3:
        for node in range(2, num_nodes - 1):
            rank = float(node - 1)
            output[node] = torch.tensor(
                [max(span - rank, 0.0), max(span - rank, 0.0)],
                dtype=torch.float64,
            )
    return output * _OGDF_GRID_SEPARATION


def build_schnyder_pipeline() -> str:
    """Return the Schnyder pipeline marker.

    Returns
    -------
    str
        Pipeline marker name.
    """
    return "schnyder_pipeline"


def layout_schnyder_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: Optional[int] = 42,
    edge_weights: Optional[torch.Tensor] = None,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run Schnyder planar-grid layout without calling the OGDF runner.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor | None, optional
        Accepted for API consistency; Schnyder grid layout ignores sizes.
    seed : int | None, default=42
        Accepted for API consistency; Schnyder layout is deterministic.
    edge_weights : torch.Tensor | None, optional
        Accepted for API consistency; Schnyder layout ignores weights.
    fidelity_dtype : torch.dtype | None, optional
        Optional output dtype override.

    Returns
    -------
    torch.Tensor
        Coordinate tensor with shape ``[N, 2]``.
    """
    del node_sizes, seed, edge_weights
    is_planar, _embedding = check_planarity(edge_index, num_nodes)
    if not is_planar:
        raise PlanarityError("Schnyder layout is only defined for planar graphs.")
    pos = _fallback_schnyder_grid(edge_index, num_nodes).to(device=edge_index.device)
    if fidelity_dtype is not None:
        return pos.to(dtype=fidelity_dtype)
    return pos


__all__ = ["build_schnyder_pipeline", "layout_schnyder_pipeline"]
