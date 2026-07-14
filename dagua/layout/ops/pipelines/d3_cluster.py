"""d3-hierarchy cluster dendrogram pipeline."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.d3tree import D3ClusterLayout
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState

if TYPE_CHECKING:
    from dagua.config import LayoutConfig


def build_d3_cluster_pipeline(
    *,
    dx: float = 1.0,
    dy: float = 1.0,
    node_size: bool = False,
    radial: bool = False,
) -> Pipeline:
    """Build the d3-hierarchy cluster pipeline.

    Parameters
    ----------
    dx : float, default=1.0
        d3 horizontal size or node-size scale.
    dy : float, default=1.0
        d3 vertical size or node-size scale.
    node_size : bool, default=False
        Whether to use ``cluster.nodeSize([dx, dy])`` semantics.
    radial : bool, default=False
        Whether to apply the radial polar transform.

    Returns
    -------
    Pipeline
        Single-op d3 cluster pipeline.
    """
    return Pipeline(
        [D3ClusterLayout(dx=dx, dy=dy, node_size=node_size, radial=radial)],
        name="d3_cluster_pipeline",
    )


def layout_d3_cluster_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: Optional[int] = 42,
    edge_weights: Optional[torch.Tensor] = None,
    dx: float = 1.0,
    dy: float = 1.0,
    node_size: bool = False,
    size: Optional[Tuple[float, float]] = None,
    radial: bool = False,
    fidelity_dtype: Optional[torch.dtype] = None,
    config: Optional["LayoutConfig"] = None,
) -> torch.Tensor:
    """Run the deterministic d3-hierarchy cluster source port.

    Parameters
    ----------
    edge_index : torch.Tensor
        Directed parent-child edges with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N``.
    node_sizes : torch.Tensor | None, optional
        Accepted for engine compatibility; d3 cluster ignores node boxes.
    seed : int | None, default=42
        Accepted for dispatch compatibility; d3 cluster is deterministic.
    edge_weights : torch.Tensor | None, optional
        Accepted for API compatibility; unused.
    dx : float, default=1.0
        d3 horizontal size or node-size scale.
    dy : float, default=1.0
        d3 vertical size or node-size scale.
    node_size : bool, default=False
        Whether to use ``cluster.nodeSize([dx, dy])`` semantics.
    size : tuple[float, float] | None, optional
        Alias for d3 ``cluster.size([x, y])``; overrides ``dx``, ``dy``, and
        sets ``node_size=False``.
    radial : bool, default=False
        Whether to apply d3's radial polar transform.
    fidelity_dtype : torch.dtype | None, optional
        Accepted for engine compatibility. Output is always ``float64``.
    config : LayoutConfig | None, optional
        Accepted for engine compatibility.

    Returns
    -------
    torch.Tensor
        Coordinates with shape ``[N, 2]``.
    """
    del node_sizes, seed, edge_weights, fidelity_dtype, config
    resolved_dx = float(dx)
    resolved_dy = float(dy)
    resolved_node_size = bool(node_size)
    if size is not None:
        resolved_dx = float(size[0])
        resolved_dy = float(size[1])
        resolved_node_size = False
    problem = LayoutProblem(edge_index=edge_index, num_nodes=num_nodes)
    state = SolveState()
    context = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    pipeline = build_d3_cluster_pipeline(
        dx=resolved_dx,
        dy=resolved_dy,
        node_size=resolved_node_size,
        radial=radial,
    )
    final_state = pipeline.apply(problem, state, context)
    if final_state.pos is None:
        raise RuntimeError("d3 cluster pipeline did not produce final positions.")
    return final_state.pos


def layout_d3_cluster_radial_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    **kwargs: object,
) -> torch.Tensor:
    """Run d3 cluster with d3's radial transform.

    Parameters
    ----------
    edge_index : torch.Tensor
        Directed parent-child edges with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor | None, optional
        Accepted for API compatibility; unused.
    **kwargs : object
        Additional d3 cluster pipeline parameters.

    Returns
    -------
    torch.Tensor
        Radial Cartesian coordinates with shape ``[N, 2]``.
    """
    kwargs.setdefault("size", (2.0 * math.pi, 1.0))
    kwargs["radial"] = True
    return layout_d3_cluster_pipeline(edge_index, num_nodes, node_sizes, **kwargs)


__all__ = [
    "build_d3_cluster_pipeline",
    "layout_d3_cluster_pipeline",
    "layout_d3_cluster_radial_pipeline",
]
