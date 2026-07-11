"""Distributed Recursive Layout (DrL) pipeline."""

from __future__ import annotations

from typing import Optional

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.drl import (
    DRLFinalizePositions,
    DRLInitializePositions,
    DrLOptions,
    DRLPhaseSolve,
    DRLPhaseSolveConfig,
    DRLPrepareState,
    DRLPrepareStateConfig,
)
from dagua.layout.ops.graph_utils import layout_device
from dagua.layout.ops.pipelines import resolve_fidelity_dtype
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def build_drl_pipeline(
    options: DrLOptions = "default",
    fidelity_mode: bool = False,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> Pipeline:
    """Build a Distributed Recursive Layout pipeline.

    Reference fidelity
    ------------------
    Targets: igraph 1.0.0 DrL / Martin, Brown, and Klavans (2008),
        "OpenOrd: An Open-Source Toolbox for Large Graph Layout".
    Fidelity mode: ``layout_drl_pipeline(..., fidelity_mode=True)`` uses the
        benchmark adapter's seed-matrix and RNG conventions while remaining in
        the pure Python DrL port.
    Verified at: round_79 seeded real-graph parity against igraph 1.0.0.
    Known divergences: none in the seeded DrL fidelity rows verified at round 79.

    Parameters
    ----------
    options : str or Mapping[str, object] or OptionObject, default="default"
        DrL preset name or per-phase override container controlling the coarse,
        liquid, expansion, and final smoothing phases.
    fidelity_mode : bool, default=False
        Preserved for direct native-pipeline compatibility.
    fidelity_dtype : torch.dtype, default=torch.float32
        Fidelity-mode dtype requested by public wrappers. DrL explicitly
        emulates the reference's C++ ``float`` state, so this is accepted for
        signature consistency rather than changing its internal arithmetic.

    Returns
    -------
    Pipeline
        Pipeline implementing the DrL algorithm. The pipeline produces final
        node coordinates by preparing phase parameters, initializing positions,
        running the staged recursive DrL solve, and finalizing the layout.
    """
    return Pipeline(
        [
            DRLPrepareState(config=DRLPrepareStateConfig(options=options)),
            DRLInitializePositions(fidelity_mode=fidelity_mode),
            DRLPhaseSolve(config=DRLPhaseSolveConfig(fidelity_mode=fidelity_mode)),
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
    fidelity_mode: bool = False,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run the Distributed Recursive Layout pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``. Unused compatibility
        placeholder.
    seed : int, default=42
        Random seed for initial layout and random perturbations.
    edge_weights : torch.Tensor, optional
        Optional positive edge-weight vector with shape ``[E]``.
    options : str or Mapping[str, object] or OptionObject, default="default"
        Preset name or mapping/object of per-phase overrides.
    fidelity_mode : bool, default=False
        When ``True``, use the igraph benchmark adapter's seeded initial matrix
        and Python RNG hook conventions without delegating to external layout
        implementations.
    fidelity_dtype : torch.dtype, default=torch.float32
        Internal dtype used while fidelity mode is active. Public output is
        restored to ``float32``.

    Returns
    -------
    torch.Tensor
        Final layout positions with shape ``[N, 2]`` and dtype ``float32``.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``edge_index``, or ``edge_weights`` are invalid.
    RuntimeError
        If the pipeline does not populate final positions.
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

    resolved_dtype = resolve_fidelity_dtype(fidelity_mode, fidelity_dtype)

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_drl_pipeline(
        options=options,
        fidelity_mode=fidelity_mode,
        fidelity_dtype=resolved_dtype,
    ).apply(problem, state, ctx)

    if final_state.pos is None:
        raise RuntimeError("DRL pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_drl_pipeline", "layout_drl_pipeline"]
