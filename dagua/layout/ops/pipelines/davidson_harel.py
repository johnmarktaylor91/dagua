"""Davidson-Harel simulated-annealing layout pipeline."""

from __future__ import annotations

from typing import Optional

import torch

from dagua.layout.ops.base import Pipeline, Repeat  # noqa: E402
from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig  # noqa: E402
from dagua.layout.ops.davidson_harel import (
    DHAnnealingRound,
    DHCool,
    FinalizeDHPositions,
    InitializeDHPositions,
    PrepareDHState,
)
from dagua.layout.ops.state import (  # noqa: E402
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)


def build_davidson_harel_pipeline(rounds: int = 100) -> Pipeline:
    """Build a Davidson-Harel simulated-annealing pipeline.

    Parameters
    ----------
    rounds : int, default=100
        Number of annealing rounds to execute.

    Returns
    -------
    Pipeline
        Pipeline implementing the Davidson-Harel algorithm. The pipeline
        produces final node coordinates by initializing positions, preparing
        annealing state, proposing and accepting moves across repeated rounds,
        cooling the temperature, and finalizing the layout.

    Raises
    ------
    ValueError
        If ``rounds`` is negative.
    """
    if rounds < 0:
        raise ValueError("rounds must be non-negative.")

    return Pipeline(
        [
            FixedSteps(FixedStepsConfig(n=rounds)),
            InitializeDHPositions(),
            PrepareDHState(),
            Repeat(
                n=rounds,
                ops=[
                    DHAnnealingRound(),
                    DHCool(),
                ],
            ),
            FinalizeDHPositions(),
        ],
        name="davidson_harel_pipeline",
    )


def layout_davidson_harel_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    rounds: int = 100,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the Davidson-Harel pipeline as a drop-in replacement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]`` used only for output
        device and extent selection.
    rounds : int, default=100
        Number of annealing rounds.
    seed : int, default=42
        RNG seed for initialization and move proposals.
    edge_weights : torch.Tensor, optional
        Optional edge-weight vector with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``rounds``, or ``edge_weights`` are invalid.
    RuntimeError
        If the pipeline does not populate final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if rounds < 0:
        raise ValueError("rounds must be non-negative.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge_count {edge_index.shape[1]}"
            )

    device = (
        edge_index.device
        if edge_index.numel() > 0
        else node_sizes.device
        if node_sizes is not None
        else torch.device("cpu")
    )
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
    ctx = RuntimeContext(plan=ExecutionPlan(device=str(device)))
    final_state = build_davidson_harel_pipeline(rounds=rounds).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("Davidson-Harel pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_davidson_harel_pipeline", "layout_davidson_harel_pipeline"]
