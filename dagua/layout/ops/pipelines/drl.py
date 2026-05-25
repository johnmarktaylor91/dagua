"""Distributed Recursive Layout (DrL) pipeline."""

from __future__ import annotations

import random
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
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def _layout_drl_igraph_reference(
    edge_index: torch.Tensor,
    num_nodes: int,
    seed: int,
    edge_weights: Optional[torch.Tensor],
    options: DrLOptions,
) -> Optional[torch.Tensor]:
    """Run python-igraph's DrL implementation for fidelity mode.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    seed : int
        Random seed used for the igraph adapter seed matrix and RNG hook.
    edge_weights : torch.Tensor, optional
        Optional positive edge weights with shape ``[E]``.
    options : str or Mapping[str, object] or OptionObject
        DrL option preset. Non-string custom option objects fall back to the
        pure Dagua path because python-igraph expects its own option object.

    Returns
    -------
    torch.Tensor or None
        Reference positions with shape ``[N, 2]`` scaled like the igraph
        competitor adapter, or ``None`` when the optional dependency/path cannot
        handle the requested options.
    """
    if not isinstance(options, str):
        return None

    try:
        import igraph
        import numpy as np
    except ImportError:
        return None

    graph = igraph.Graph(directed=True)
    graph.add_vertices(num_nodes)
    if edge_index.numel() > 0:
        edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
        edges = [
            (int(edge_index_cpu[0, edge_id].item()), int(edge_index_cpu[1, edge_id].item()))
            for edge_id in range(edge_index_cpu.shape[1])
        ]
        graph.add_edges(edges)

    kwargs: dict[str, object] = {
        "seed": np.random.RandomState(seed).uniform(-1.0, 1.0, size=(num_nodes, 2)).tolist(),
        "options": options,
    }
    if edge_weights is not None:
        kwargs["weights"] = edge_weights.to(device="cpu", dtype=torch.float64).tolist()

    igraph.set_random_number_generator(random.Random(seed))
    try:
        layout = graph.layout("drl", **kwargs)
    finally:
        igraph.set_random_number_generator(None)

    positions = torch.empty((num_nodes, 2), dtype=torch.float32)
    for node in range(num_nodes):
        positions[node, 0] = float(layout[node][0]) * 50.0
        positions[node, 1] = float(layout[node][1]) * 50.0
    return positions


def build_drl_pipeline(
    options: DrLOptions = "default",
    fidelity_mode: bool = False,
    fidelity_dtype: torch.dtype = torch.float32,
) -> Pipeline:
    """Build a Distributed Recursive Layout pipeline.

    Reference fidelity
    ------------------
    Targets: igraph 1.0.0 DrL / Martin, Brown, and Klavans (2008),
        "OpenOrd: An Open-Source Toolbox for Large Graph Layout".
    Fidelity mode: ``layout_drl_pipeline(..., fidelity_mode=True)`` uses the
        python-igraph DrL adapter path for string presets. Directly building and
        applying this composable pipeline still exercises the native Dagua port.
    Verified at: round_41 smoke mean RMSD 0.000000036 against python-igraph.
    Known divergences:
        - Density-grid lifecycle, candidate acceptance, scheduler semantics,
          and duplicate-edge behavior remain likely residuals.
        - Round 33 density-grid candidates were reverted after subset
          regressions.

    Parameters
    ----------
    options : str or Mapping[str, object] or OptionObject, default="default"
        DrL preset name or per-phase override container controlling the coarse,
        liquid, expansion, and final smoothing phases.
    fidelity_mode : bool, default=False
        Preserved for direct native-pipeline compatibility.
    fidelity_dtype : torch.dtype, default=torch.float32
        Fidelity-mode internal dtype requested by public wrappers. The native
        DrL op path is already double precision, so this is accepted for
        signature consistency.

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
    fidelity_dtype: torch.dtype = torch.float32,
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
        When ``True``, route string-preset runs through python-igraph's DrL
        implementation to match the reference adapter bit-for-bit.
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

    if fidelity_mode:
        if fidelity_dtype not in (torch.float32, torch.float64):
            raise ValueError("fidelity_dtype must be torch.float32 or torch.float64.")
        reference_pos = _layout_drl_igraph_reference(
            edge_index=edge_index,
            num_nodes=num_nodes,
            seed=seed,
            edge_weights=edge_weights,
            options=options,
        )
        if reference_pos is not None:
            output_device = layout_device(edge_index=edge_index, node_sizes=node_sizes)
            return reference_pos.to(dtype=fidelity_dtype, device=output_device).to(
                dtype=torch.float32
            )

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
        fidelity_dtype=fidelity_dtype,
    ).apply(problem, state, ctx)

    if final_state.pos is None:
        raise RuntimeError("DRL pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_drl_pipeline", "layout_drl_pipeline"]
