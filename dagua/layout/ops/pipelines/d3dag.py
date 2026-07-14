"""d3-dag Sugiyama-compatible composable pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.d3dag import (
    D3DagCoordinate,
    D3DagDecross,
    D3DagLayering,
    D3DagPrepare,
    D3DagSugify,
)
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState

if TYPE_CHECKING:
    from dagua.config import LayoutConfig

_D3DAG_DEFAULT_GAP = 1.0


def build_d3dag_pipeline(
    layering: str = "simplex",
    decross: str = "twoLayer",
    coord: str = "simplex",
    x_gap: float = _D3DAG_DEFAULT_GAP,
    y_gap: float = _D3DAG_DEFAULT_GAP,
    decross_passes: int = 24,
) -> Pipeline:
    """Build the d3-dag 1.2.2 Sugiyama pipeline.

    Reference fidelity
    ------------------
    Targets: ``erikbrinkman/d3-dag`` 1.2.2 ``sugiyama``. The default
        composition is ``layeringSimplex`` + ``decrossTwoLayer`` +
        ``coordSimplex`` with ``gap([1, 1])``.
    Fidelity mode: deterministic Python source port. The production pipeline
        never delegates to Node or the competitor adapter.

    Parameters
    ----------
    layering : str, default="simplex"
        Layering operator: ``"simplex"`` or ``"longestPath"``.
    decross : str, default="twoLayer"
        Decross operator: ``"twoLayer"``, ``"opt"``, or ``"dfs"``.
    coord : str, default="simplex"
        Coordinate operator: ``"simplex"`` or ``"greedy"``.
    x_gap : float, default=1.0
        Horizontal d3-dag gap.
    y_gap : float, default=1.0
        Vertical d3-dag gap.
    decross_passes : int, default=24
        Maximum two-layer decross sweeps.

    Returns
    -------
    Pipeline
        Five-stage composable d3-dag Sugiyama pipeline.
    """
    return Pipeline(
        [
            D3DagPrepare(x_gap=x_gap, y_gap=y_gap),
            D3DagLayering(method=layering),
            D3DagSugify(),
            D3DagDecross(method=decross, passes=decross_passes),
            D3DagCoordinate(method=coord),
        ],
        name="d3dag_pipeline",
    )


def layout_d3dag_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: Optional[int] = 42,
    edge_weights: Optional[torch.Tensor] = None,
    layering: str = "simplex",
    decross: str = "twoLayer",
    coord: str = "simplex",
    x_gap: float = _D3DAG_DEFAULT_GAP,
    y_gap: float = _D3DAG_DEFAULT_GAP,
    gap: Optional[float] = None,
    decross_passes: int = 24,
    fidelity_dtype: Optional[torch.dtype] = None,
    config: Optional["LayoutConfig"] = None,
) -> torch.Tensor:
    """Run the deterministic d3-dag Sugiyama pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Directed acyclic graph edges with shape ``[2, E]``.
    num_nodes : int
        Number of original nodes ``N``.
    node_sizes : torch.Tensor | None, optional
        Node box sizes with shape ``[N, 2]``. Missing sizes use d3-dag's
        default ``[1, 1]``.
    seed : int | None, default=42
        Accepted for dispatch compatibility; d3-dag Sugiyama is deterministic.
    edge_weights : torch.Tensor | None, optional
        Accepted for pipeline API compatibility. d3-dag default Sugiyama does
        not use edge weights.
    layering : str, default="simplex"
        ``"simplex"`` or ``"longestPath"``.
    decross : str, default="twoLayer"
        ``"twoLayer"``, ``"opt"``, or ``"dfs"``.
    coord : str, default="simplex"
        ``"simplex"`` or ``"greedy"``.
    x_gap : float, default=1.0
        Horizontal d3-dag gap.
    y_gap : float, default=1.0
        Vertical d3-dag gap.
    gap : float | None, optional
        Scalar alias overriding both ``x_gap`` and ``y_gap``.
    decross_passes : int, default=24
        Maximum two-layer decross sweeps.
    fidelity_dtype : torch.dtype | None, optional
        Accepted for engine compatibility. The source port computes in Python
        double precision and returns ``float64``.
    config : LayoutConfig | None, optional
        Accepted for engine dispatch compatibility.

    Returns
    -------
    torch.Tensor
        Node positions with shape ``[N, 2]``.

    Raises
    ------
    RuntimeError
        If the composed stages do not produce final positions.
    """
    del seed, edge_weights, fidelity_dtype, config
    resolved_x_gap = float(x_gap if gap is None else gap)
    resolved_y_gap = float(y_gap if gap is None else gap)
    problem = LayoutProblem(edge_index=edge_index, num_nodes=num_nodes, node_sizes=node_sizes)
    state = SolveState()
    context = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    pipeline = build_d3dag_pipeline(
        layering=layering,
        decross=decross,
        coord=coord,
        x_gap=resolved_x_gap,
        y_gap=resolved_y_gap,
        decross_passes=decross_passes,
    )
    final_state = pipeline.apply(problem, state, context)
    if final_state.pos is None:
        raise RuntimeError("d3-dag pipeline did not produce final positions.")
    return final_state.pos


def layout_d3dag_simplex_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    **kwargs: object,
) -> torch.Tensor:
    """Run the default simplex/two-layer/simplex d3-dag variant.

    Parameters
    ----------
    edge_index : torch.Tensor
        Directed edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of original nodes.
    node_sizes : torch.Tensor | None, optional
        Node sizes with shape ``[N, 2]``.
    **kwargs : object
        Additional pipeline keyword arguments.

    Returns
    -------
    torch.Tensor
        Node positions with shape ``[N, 2]``.
    """
    return layout_d3dag_pipeline(edge_index, num_nodes, node_sizes, **kwargs)


def layout_d3dag_longestpath_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    **kwargs: object,
) -> torch.Tensor:
    """Run d3-dag with longest-path layering.

    Parameters
    ----------
    edge_index : torch.Tensor
        Directed edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of original nodes.
    node_sizes : torch.Tensor | None, optional
        Node sizes with shape ``[N, 2]``.
    **kwargs : object
        Additional pipeline keyword arguments.

    Returns
    -------
    torch.Tensor
        Node positions with shape ``[N, 2]``.
    """
    kwargs["layering"] = "longestPath"
    return layout_d3dag_pipeline(edge_index, num_nodes, node_sizes, **kwargs)


def layout_d3dag_opt_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    **kwargs: object,
) -> torch.Tensor:
    """Run d3-dag with exact small-graph crossing minimization.

    Parameters
    ----------
    edge_index : torch.Tensor
        Directed edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of original nodes.
    node_sizes : torch.Tensor | None, optional
        Node sizes with shape ``[N, 2]``.
    **kwargs : object
        Additional pipeline keyword arguments.

    Returns
    -------
    torch.Tensor
        Node positions with shape ``[N, 2]``.
    """
    kwargs["decross"] = "opt"
    return layout_d3dag_pipeline(edge_index, num_nodes, node_sizes, **kwargs)


def layout_d3dag_greedy_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    **kwargs: object,
) -> torch.Tensor:
    """Run d3-dag with greedy coordinate assignment.

    Parameters
    ----------
    edge_index : torch.Tensor
        Directed edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of original nodes.
    node_sizes : torch.Tensor | None, optional
        Node sizes with shape ``[N, 2]``.
    **kwargs : object
        Additional pipeline keyword arguments.

    Returns
    -------
    torch.Tensor
        Node positions with shape ``[N, 2]``.
    """
    kwargs["coord"] = "greedy"
    return layout_d3dag_pipeline(edge_index, num_nodes, node_sizes, **kwargs)


__all__ = [
    "build_d3dag_pipeline",
    "layout_d3dag_greedy_pipeline",
    "layout_d3dag_longestpath_pipeline",
    "layout_d3dag_opt_pipeline",
    "layout_d3dag_pipeline",
    "layout_d3dag_simplex_pipeline",
]
