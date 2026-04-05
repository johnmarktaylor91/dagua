"""tsNET layout pipeline."""

from __future__ import annotations

from typing import Optional

import torch

from dagua.layout.ops.base import Pipeline, Repeat  # noqa: E402
from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig  # noqa: E402
from dagua.layout.ops.graph_utils import layout_device
from dagua.layout.ops.state import (  # noqa: E402
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)
from dagua.layout.ops.tsnet import (  # noqa: E402
    TsnetFinalizePositions,
    TsnetGradientStep,
    TsnetInitializeOptimizer,
    TsnetInitializePositions,
    TsnetPrepareState,
)


def build_tsnet_pipeline(steps: int = 1000) -> Pipeline:
    """Build a tsNET layout pipeline.

    Parameters
    ----------
    steps : int, default=1000
        Number of optimization updates.

    Returns
    -------
    Pipeline
        Pipeline implementing the tsNET algorithm. The pipeline produces final
        node coordinates by initializing positions, preparing t-SNE-style
        affinities, creating the optimizer state, applying repeated
        gains-and-momentum gradient steps, and finalizing the embedding.

    Raises
    ------
    ValueError
        If ``steps`` is negative.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative.")

    return Pipeline(
        [
            FixedSteps(FixedStepsConfig(n=steps)),
            TsnetInitializePositions(),
            TsnetPrepareState(),
            TsnetInitializeOptimizer(),
            Repeat(
                n=steps,
                ops=[
                    TsnetGradientStep(),
                ],
            ),
            TsnetFinalizePositions(),
        ],
        name="tsnet_pipeline",
    )


def layout_tsnet_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    perplexity: float = 30,
    steps: int = 1000,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the tsNET pipeline as a drop-in replacement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]`` used for final
        scaling.
    perplexity : float, default=30
        Target t-SNE perplexity. Currently only the default value of 30
        preserves bit-identity with classic; non-default values require
        extending ``TsnetPrepareState``.
    steps : int, default=1000
        Number of optimization updates.
    seed : int, default=42
        Random seed for the torch generator initialization.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``steps``, ``perplexity``, or ``edge_weights``
        are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if perplexity <= 0:
        raise ValueError("perplexity must be positive.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
            )

    device = layout_device(edge_index, node_sizes)

    # Handle trivial cases exactly like classic.
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
    )
    state = SolveState()
    state.extras["tsnet_perplexity"] = perplexity
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_tsnet_pipeline(steps=steps).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("tsNET pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_tsnet_pipeline", "layout_tsnet_pipeline"]
