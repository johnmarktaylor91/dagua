"""Distributed Recursive Layout (DrL) expressed as a composable ops pipeline."""

from __future__ import annotations

from typing import Optional

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.drl import (
    DRLFinalizePositions,
    DRLInitializePositions,
    DrLOptions,
    DRLPhaseSolve,
    DRLPrepareState,
    DRLPrepareStateConfig,
)
from dagua.layout.ops.graph_utils import layout_device
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def build_drl_pipeline(options: DrLOptions = "default") -> Pipeline:
    """Build a DRL pipeline that matches ``layout_drl`` output.

    Parameters
    ----------
    options : str or Mapping[str, object] or OptionObject, default="default"
        DrL preset name or per-phase override container.

    Returns
    -------
    Pipeline
        Composable pipeline of registered DRL operations.
    """
    return Pipeline(
        [
            DRLPrepareState(config=DRLPrepareStateConfig(options=options)),
            DRLInitializePositions(),
            DRLPhaseSolve(),
            DRLFinalizePositions(),
        ],
        name="drl_pipeline",
    )


def layout_drl_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    options: DrLOptions = "default",
) -> torch.Tensor:
    """Run the registered DRL pipeline as a drop-in replacement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    num_nodes : int
        Total number of graph nodes.
    node_sizes : torch.Tensor, optional
        Unused compatibility placeholder.
    seed : int, default=42
        Random seed for initial layout and random perturbations.
    edge_weights : torch.Tensor, optional
        Optional positive edge-weight vector with shape ``[E]``.
    options : str or Mapping[str, object] or OptionObject, default="default"
        Preset name or mapping/object of per-phase overrides.

    Returns
    -------
    torch.Tensor
        Final layout positions with shape ``[N, 2]`` and dtype ``float32``.
    """
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
        min_index = int(edge_index_cpu.min().item())
        max_index = int(edge_index_cpu.max().item())
        if min_index < 0:
            raise ValueError("edge_index cannot contain negative node indices.")
        if max_index >= num_nodes:
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
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_drl_pipeline(options=options).apply(problem, state, ctx)

    if final_state.pos is None:
        raise RuntimeError("DRL pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_drl_pipeline", "layout_drl_pipeline"]
