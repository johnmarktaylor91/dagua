"""igraph-compatible circular Reingold-Tilford radial tree pipeline."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch

from dagua.layout.ops._reingold_tilford import layout_igraph_reingold_tilford
from dagua.layout.ops.base import Pipeline


def radial_tree_from_rt_units(rt_positions: torch.Tensor) -> torch.Tensor:
    """Apply igraph's circular RT polar transform to RT unit coordinates.

    Parameters
    ----------
    rt_positions : torch.Tensor
        Reingold-Tilford unit coordinates with shape ``[N, 2]`` where ``x`` is
        tidy-tree order and ``y`` is tree depth.

    Returns
    -------
    torch.Tensor
        Circular radial coordinates with shape ``[N, 2]``.
    """
    if rt_positions.ndim != 2 or rt_positions.shape[1] != 2:
        raise ValueError("rt_positions must have shape [N, 2].")
    num_nodes = int(rt_positions.shape[0])
    if num_nodes == 0:
        return rt_positions.clone()

    rt64 = rt_positions.to(dtype=torch.float64, device="cpu")
    x_values = rt64[:, 0]
    min_x = float(torch.min(x_values).item())
    max_x = float(torch.max(x_values).item())
    ratio = 2.0 * math.pi * (float(num_nodes) - 1.0) / float(num_nodes)
    if max_x > min_x:
        ratio /= max_x - min_x

    phi = (x_values - min_x) * ratio
    radius = rt64[:, 1]
    output = torch.empty_like(rt64)
    output[:, 0] = radius * torch.cos(phi)
    output[:, 1] = radius * torch.sin(phi)
    return output


def build_radial_tree_pipeline() -> Pipeline:
    """Build the radial-tree pipeline placeholder.

    Returns
    -------
    Pipeline
        Empty marker pipeline; the public entrypoint composes the local
        igraph-compatible RT port with the circular polar transform.
    """
    return Pipeline([], name="radial_tree_pipeline")


def layout_radial_tree_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: int = 42,
    traversal_mode: str = "out",
    roots: Optional[Sequence[int]] = None,
    rootlevel: Optional[Sequence[int]] = None,
    output_scale: float = 50.0,
    edge_weights: Optional[torch.Tensor] = None,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run igraph-compatible circular Reingold-Tilford radial tree layout.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Accepted for engine compatibility; igraph RT is size-blind.
    seed : int, default=42
        Accepted for interface compatibility. Radial tree is deterministic.
    traversal_mode : str, default="out"
        Edge traversal mode: ``"out"``, ``"in"``, or ``"all"``.
    roots : sequence of int | None, default=None
        Optional explicit root vertices.
    rootlevel : sequence of int | None, default=None
        Optional depth per explicit root.
    output_scale : float, default=50.0
        Uniform scale applied after igraph's unit circular transform.
    edge_weights : torch.Tensor, optional
        Accepted for interface compatibility; igraph RT ignores weights.
    fidelity_dtype : torch.dtype, optional
        Optional output dtype. Defaults to ``torch.float64`` for fidelity.

    Returns
    -------
    torch.Tensor
        Final radial-tree coordinates with shape ``[N, 2]``.
    """
    del node_sizes, seed, edge_weights
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if output_scale <= 0.0:
        raise ValueError("output_scale must be positive.")
    dtype = torch.float64 if fidelity_dtype is None else fidelity_dtype
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=dtype, device=edge_index.device)

    rt_units = layout_igraph_reingold_tilford(
        edge_index=edge_index,
        num_nodes=num_nodes,
        traversal_mode=traversal_mode,
        roots=roots,
        rootlevel=rootlevel,
        center_output=False,
        output_scale=1.0,
    )
    radial_units = radial_tree_from_rt_units(rt_units)
    return (radial_units * float(output_scale)).to(dtype=dtype, device=edge_index.device)


__all__ = [
    "build_radial_tree_pipeline",
    "layout_radial_tree_pipeline",
    "radial_tree_from_rt_units",
]
