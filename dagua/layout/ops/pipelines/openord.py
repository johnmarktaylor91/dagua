"""OpenOrd layout pipeline."""

from __future__ import annotations

from typing import Optional

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.graph_utils import layout_device
from dagua.layout.ops.openord import (
    OpenOrdFinalizePositions,
    OpenOrdInitializePositions,
    OpenOrdOptions,
    OpenOrdPhaseSolve,
    OpenOrdPrepareState,
    OpenOrdPrepareStateConfig,
)
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def build_openord_pipeline(
    options: OpenOrdOptions = "default",
    edge_cut: Optional[float] = None,
    multilevel: Optional[bool] = None,
) -> Pipeline:
    """Build the composable OpenOrd operation pipeline.

    Parameters
    ----------
    options : str or Mapping[str, object] or OpenOrdOptionObject, default="default"
        OpenOrd preset name or per-phase override provider.
    edge_cut : float, optional
        Edge-cutting ratio in ``[0, 1]``. ``None`` uses the preset default.
    multilevel : bool, optional
        Whether to force or disable the recursive OpenOrd coarsen/refine path.

    Returns
    -------
    Pipeline
        Pipeline implementing serial OpenOrd initialization, five-phase solve,
        and output finalization.
    """
    return Pipeline(
        [
            OpenOrdPrepareState(
                config=OpenOrdPrepareStateConfig(
                    options=options,
                    edge_cut=edge_cut,
                    multilevel=multilevel,
                )
            ),
            OpenOrdInitializePositions(),
            OpenOrdPhaseSolve(),
            OpenOrdFinalizePositions(),
        ],
        name="openord_pipeline",
    )


def layout_openord_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: int = 0,
    edge_weights: Optional[torch.Tensor] = None,
    options: OpenOrdOptions = "default",
    edge_cut: Optional[float] = None,
    multilevel: Optional[bool] = None,
    steps: int = 0,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run the OpenOrd layout pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``. Unused compatibility
        placeholder.
    seed : int, default=0
        Random seed matching OpenOrd's ``-s`` option default.
    edge_weights : torch.Tensor, optional
        Optional positive edge-weight vector with shape ``[E]``.
    options : str or Mapping[str, object] or OpenOrdOptionObject, default="default"
        Preset name or mapping/object of per-phase overrides.
    edge_cut : float, optional
        Edge-cutting ratio in ``[0, 1]``. ``None`` uses the preset default.
    multilevel : bool, optional
        Whether to force or disable the recursive coarsen/refine path.
    steps : int, default=0
        Accepted for LayoutConfig dispatch compatibility. OpenOrd uses phase
        iteration counts from ``options`` rather than a single global step.
    fidelity_dtype : torch.dtype, optional
        Accepted for common pipeline signature compatibility. OpenOrd emulates
        C++ ``float`` state internally and returns ``float32``.

    Returns
    -------
    torch.Tensor
        Final layout positions with shape ``[N, 2]`` and dtype ``float32``.

    Raises
    ------
    ValueError
        If graph inputs or edge weights are invalid.
    RuntimeError
        If the pipeline does not populate final positions.
    """
    del steps, fidelity_dtype
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")
    if edge_weights is not None and edge_weights.shape[0] != edge_index.shape[1]:
        raise ValueError(
            f"edge_weights length {edge_weights.shape[0]} does not match "
            f"edge count {edge_index.shape[1]}"
        )
    if edge_index.numel() > 0:
        edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
        if int(edge_index_cpu.min().item()) < 0:
            raise ValueError("edge_index cannot contain negative node indices.")
        if int(edge_index_cpu.max().item()) >= num_nodes:
            raise ValueError("edge_index contains node indices outside [0, num_nodes).")
    if edge_weights is not None and bool(torch.any(edge_weights <= 0.0).item()):
        raise ValueError("edge_weights must be strictly positive.")

    if num_nodes == 0:
        device = layout_device(edge_index=edge_index, node_sizes=node_sizes)
        return torch.empty((0, 2), dtype=torch.float32, device=device)

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = build_openord_pipeline(
        options=options,
        edge_cut=edge_cut,
        multilevel=multilevel,
    ).apply(
        problem,
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device="cpu")),
    )
    if state.pos is None:
        raise RuntimeError("OpenOrd pipeline did not produce final positions.")
    return state.pos


def layout_openord_refine_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: int = 0,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the OpenOrd refine preset.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``.
    seed : int, default=0
        Deterministic random seed.
    edge_weights : torch.Tensor, optional
        Optional positive edge weights.

    Returns
    -------
    torch.Tensor
        Final positions with shape ``[N, 2]``.
    """
    return layout_openord_pipeline(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        seed=seed,
        edge_weights=edge_weights,
        options="refine",
    )


def layout_openord_final_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: int = 0,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the OpenOrd final preset.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``.
    seed : int, default=0
        Deterministic random seed.
    edge_weights : torch.Tensor, optional
        Optional positive edge weights.

    Returns
    -------
    torch.Tensor
        Final positions with shape ``[N, 2]``.
    """
    return layout_openord_pipeline(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        seed=seed,
        edge_weights=edge_weights,
        options="final",
    )


__all__ = [
    "build_openord_pipeline",
    "layout_openord_final_pipeline",
    "layout_openord_pipeline",
    "layout_openord_refine_pipeline",
]
