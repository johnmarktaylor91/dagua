"""NeuLay two-phase graph layout pipeline."""

from __future__ import annotations

import torch

from dagua.layout.ops.base import Pipeline, Repeat
from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig
from dagua.layout.ops.neulay import (
    NeuLayDirectConvergenceCheck,
    NeuLayDirectStep,
    NeuLayFinalizePositions,
    NeuLayPrepareDirectOptimizer,
    NeuLayPrepareState,
    NeuLayPrepareStateConfig,
    NeuLayRunGCNPhase,
    NeuLaySeedRNG,
)
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def build_neulay_pipeline(
    steps: int = 20_000,
    gcn_steps: int = 2_000,
    use_gcn: bool = True,
    dim: int = 2,
    lr: float = 0.01,
    radius: float = 0.4,
    magnitude: float | None = None,
) -> Pipeline:
    """Build a NeuLay two-phase graph layout pipeline.

    Parameters
    ----------
    steps : int, default=20000
        Total optimization budget across both phases.
    gcn_steps : int, default=2000
        Number of GCN reparameterization steps.
    use_gcn : bool, default=True
        Whether to run the optional GCN phase.
    dim : int, default=2
        Output embedding dimensionality.
    lr : float, default=0.01
        RMSprop learning rate for the direct phase.
    radius : float, default=0.4
        Gaussian repulsion radius.
    magnitude : float | None, default=None
        Gaussian repulsion magnitude. ``None`` triggers the adaptive formula.

    Returns
    -------
    Pipeline
        Pipeline implementing the NeuLay algorithm. The pipeline produces
        final node coordinates by seeding the RNG, preparing the GCN and
        direct-optimization state, optionally running the GCN phase, refining
        positions with RMSprop and KD-tree repulsion, checking convergence, and
        finalizing the embedding.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if gcn_steps < 0:
        raise ValueError("gcn_steps must be non-negative.")

    linear_steps = max(steps - gcn_steps, 0) if use_gcn else steps

    return Pipeline(
        [
            FixedSteps(FixedStepsConfig(n=steps)),
            NeuLaySeedRNG(),
            NeuLayPrepareState(
                NeuLayPrepareStateConfig(
                    dim=dim,
                    lr=lr,
                    radius=radius,
                    magnitude=magnitude,
                    use_gcn=use_gcn,
                    gcn_steps=gcn_steps,
                    total_steps=steps,
                )
            ),
            NeuLayRunGCNPhase(),
            NeuLayPrepareDirectOptimizer(),
            Repeat(
                n=linear_steps,
                ops=[
                    NeuLayDirectStep(),
                    NeuLayDirectConvergenceCheck(),
                ],
            ),
            NeuLayFinalizePositions(),
        ],
        name="neulay_pipeline",
    )


def layout_neulay_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: torch.Tensor | None = None,
    seed: int = 42,
    steps: int = 20_000,
    gcn_steps: int = 2_000,
    use_gcn: bool = True,
    dim: int = 2,
    lr: float = 0.01,
    radius: float = 0.4,
    magnitude: float | None = None,
    edge_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the NeuLay two-phase graph layout pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor | None, default=None
        Optional node-size tensor used to resolve output device.
    seed : int, default=42
        Random seed used for optimization.
    steps : int, default=20_000
        Total optimization budget across both phases.
    gcn_steps : int, default=2_000
        Number of GCN reparameterization steps.
    use_gcn : bool, default=True
        Whether to run the optional GCN phase.
    dim : int, default=2
        Output dimensionality.
    lr : float, default=0.01
        RMSprop learning rate for the direct phase.
    radius : float, default=0.4
        Gaussian repulsion radius.
    magnitude : float | None, default=None
        Gaussian repulsion magnitude. ``None`` uses adaptive formula.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``. Accepted for interface
        consistency.

    Returns
    -------
    torch.Tensor
        Final coordinate tensor with shape ``[N, dim]``.

    Raises
    ------
    ValueError
        If inputs are invalid.
    RuntimeError
        If the pipeline does not produce final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if gcn_steps < 0:
        raise ValueError("gcn_steps must be non-negative.")
    if dim <= 0:
        raise ValueError("dim must be positive.")
    if lr <= 0.0:
        raise ValueError("lr must be positive.")
    if radius <= 0.0:
        raise ValueError("radius must be positive.")
    if magnitude is not None and magnitude < 0.0:
        raise ValueError("magnitude must be non-negative.")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} does not match "
                f"edge count {edge_index.shape[1]}"
            )
    if edge_index.numel() != 0:
        if edge_index.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ValueError("edge_index must use an integer dtype.")
        min_index = int(edge_index.min().item())
        max_index = int(edge_index.max().item())
        if min_index < 0 or max_index >= num_nodes:
            raise ValueError("edge_index contains node indices outside [0, num_nodes).")

    output_device = (
        edge_index.device
        if edge_index.numel() > 0
        else node_sizes.device
        if node_sizes is not None
        else torch.device("cpu")
    )
    if num_nodes == 0:
        return torch.empty((0, dim), dtype=torch.float32, device=output_device)
    if num_nodes == 1:
        return torch.zeros((1, dim), dtype=torch.float32, device=output_device)

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device=str(output_device)))
    final_state = build_neulay_pipeline(
        steps=steps,
        gcn_steps=gcn_steps,
        use_gcn=use_gcn,
        dim=dim,
        lr=lr,
        radius=radius,
        magnitude=magnitude,
    ).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("NeuLay pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_neulay_pipeline", "layout_neulay_pipeline"]
