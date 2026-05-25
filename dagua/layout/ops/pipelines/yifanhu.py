"""Yifan Hu multilevel force-directed layout pipeline."""

from __future__ import annotations

from typing import Optional

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.graph_utils import layout_device as _layout_device
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.yifanhu import (
    BuildYifanHuGraph,
    BuildYifanHuHierarchy,
    InitYifanHuCoarsestPositions,
    YifanHuFinalizePositions,
    YifanHuFinalTuning,
    YifanHuProlongateAndRefineLevels,
    YifanHuRefineCoarsestLevel,
    final_tuning_steps,
)

_DEFAULT_THETA = 1.2
_DEFAULT_REPULSIVE_EXPONENT = 0.0


def build_yifanhu_pipeline(
    steps: int = 500,
    theta: float = _DEFAULT_THETA,
    repulsive_exponent: float = _DEFAULT_REPULSIVE_EXPONENT,
) -> Pipeline:
    """Build a native YifanHu multilevel force-directed pipeline.

    Reference fidelity
    ------------------
    Targets: Gephi Yifan Hu-style multilevel layout / Hu (2005), "Efficient,
        High-Quality Force-Directed Graph Drawing".
    Fidelity mode: no reference mode; Round 33 found no importable Python
        YifanHu reference, so this is a Dagua-only native implementation.
    Verified at: round_33 smoke evaluation on five bounded graphs; no paired
        RMSD/TOST result was available.
    Known divergences:
        - No Python reference was available for paired fidelity comparison.
        - Uses Dagua's existing multilevel and Barnes-Hut force skeleton with
          YifanHu-style defaults.

    Parameters
    ----------
    steps : int, default=500
        Maximum number of force-directed iterations per hierarchy level.
    theta : float, default=1.2
        Barnes-Hut opening angle threshold.
    repulsive_exponent : float, default=0.0
        Repulsive force exponent. The default gives inverse-distance Hu-style
        repulsion through the shared spring-electrical force primitive.

    Returns
    -------
    Pipeline
        Pipeline that coarsens with heavy-edge matching, embeds the coarsest
        graph, prolongates through finer levels, performs a final tuning pass,
        and normalizes the output coordinates.

    Raises
    ------
    ValueError
        If ``steps`` is negative.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative.")

    return Pipeline(
        [
            BuildYifanHuGraph(),
            BuildYifanHuHierarchy(),
            InitYifanHuCoarsestPositions(),
            YifanHuRefineCoarsestLevel(
                steps=steps,
                theta=theta,
                repulsive_exponent=repulsive_exponent,
            ),
            YifanHuProlongateAndRefineLevels(
                steps=steps,
                theta=theta,
                repulsive_exponent=repulsive_exponent,
            ),
            YifanHuFinalTuning(
                steps=final_tuning_steps(steps),
                theta=theta,
                repulsive_exponent=repulsive_exponent,
            ),
            YifanHuFinalizePositions(),
        ],
        name="yifanhu_pipeline",
    )


def layout_yifanhu_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 500,
    seed: int = 123,
    theta: float = _DEFAULT_THETA,
    repulsive_exponent: float = _DEFAULT_REPULSIVE_EXPONENT,
    edge_weights: Optional[torch.Tensor] = None,
    direction: str = "TB",
) -> torch.Tensor:
    """Run the native YifanHu layout pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]`` used for output extent.
    steps : int, default=500
        Maximum number of force-directed iterations per hierarchy level.
    seed : int, default=123
        Random seed for coarsening order, coarsest initialization, and
        prolongation jitter.
    theta : float, default=1.2
        Barnes-Hut opening angle threshold.
    repulsive_exponent : float, default=0.0
        Repulsive force exponent.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.
    direction : str, default="TB"
        Requested layout flow direction: ``TB``, ``BT``, ``LR``, or ``RL``.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If inputs are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")
    if edge_index.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError("edge_index must use an integer dtype.")
    if edge_weights is not None and edge_weights.shape[0] != edge_index.shape[1]:
        raise ValueError(
            f"edge_weights length {edge_weights.shape[0]} does not match "
            f"edge count {edge_index.shape[1]}"
        )
    if edge_index.numel() != 0:
        edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
        if int(edge_index_cpu.min().item()) < 0:
            raise ValueError("edge_index cannot contain negative node indices.")
        if int(edge_index_cpu.max().item()) >= num_nodes:
            raise ValueError("edge_index contains node indices outside num_nodes.")

    device = _layout_device(edge_index=edge_index, node_sizes=node_sizes)
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32, device=device)

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
        direction=direction,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_yifanhu_pipeline(
        steps=steps,
        theta=theta,
        repulsive_exponent=repulsive_exponent,
    ).apply(problem, state, ctx)

    if final_state.pos is None:
        raise RuntimeError("YifanHu pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_yifanhu_pipeline", "layout_yifanhu_pipeline"]
