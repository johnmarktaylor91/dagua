"""MaxEnt-Stress expressed as a composable ops pipeline."""

from __future__ import annotations

from typing import Optional

import torch

from dagua.layout.ops.base import Pipeline, Repeat
from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig
from dagua.layout.ops.graph_utils import layout_device
from dagua.layout.ops.maxent_stress import (
    MaxentFinalizePositions,
    MaxentGradientStep,
    MaxentInitializeOptimizer,
    MaxentInitializePositions,
    MaxentMajorizationStep,
    MaxentPrepareState,
)
from dagua.layout.ops.state import (  # noqa: E402
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)

_MAJORIZATION_NODE_LIMIT = 5_000


def build_maxent_stress_majorization_pipeline(steps: int = 200) -> Pipeline:
    """Build a pipeline for the majorization branch of maxent-stress.

    Parameters
    ----------
    steps : int, default=200
        Number of majorization iterations.

    Returns
    -------
    Pipeline
        Pipeline that reproduces the classical majorization branch.
    """
    return Pipeline(
        [
            FixedSteps(FixedStepsConfig(n=steps)),
            MaxentInitializePositions(for_majorization=True),
            MaxentPrepareState(for_majorization=True),
            Repeat(
                n=steps,
                ops=[MaxentMajorizationStep()],
            ),
            MaxentFinalizePositions(for_majorization=True),
        ],
        name="maxent_stress_majorization_pipeline",
    )


def build_maxent_stress_gradient_pipeline(
    steps: int = 200,
    alpha: float = 1.0,
    use_entropy: bool = False,
) -> Pipeline:
    """Build a pipeline for the gradient branch of maxent-stress.

    Parameters
    ----------
    steps : int, default=200
        Number of Adam updates.
    alpha : float, default=1.0
        Entropy repulsion weight.
    use_entropy : bool, default=False
        Whether to include entropy repulsion term.

    Returns
    -------
    Pipeline
        Pipeline that reproduces the classical gradient branch.
    """
    return Pipeline(
        [
            FixedSteps(FixedStepsConfig(n=steps)),
            MaxentInitializePositions(for_majorization=False),
            MaxentPrepareState(for_majorization=False, use_entropy=use_entropy),
            MaxentInitializeOptimizer(),
            Repeat(
                n=steps,
                ops=[MaxentGradientStep(alpha=alpha, use_entropy=use_entropy)],
            ),
            MaxentFinalizePositions(for_majorization=False),
        ],
        name="maxent_stress_gradient_pipeline",
    )


def build_maxent_stress_pipeline(
    steps: int = 200,
    alpha: float = 1.0,
    use_entropy: bool = False,
    use_majorization: bool = True,
    num_nodes: int = 0,
) -> Pipeline:
    """Build the classical maxent-stress dispatch pipeline.

    Parameters
    ----------
    steps : int, default=200
        Number of optimization iterations.
    alpha : float, default=1.0
        Entropy repulsion weight.
    use_entropy : bool, default=False
        Whether to include entropy repulsion term.
    use_majorization : bool, default=True
        Whether to prefer majorization for eligible graphs.
    num_nodes : int, default=0
        Number of nodes used for branch dispatch.

    Returns
    -------
    Pipeline
        Majorization pipeline when eligible; otherwise gradient pipeline.
    """
    if (
        use_majorization
        and not use_entropy
        and num_nodes <= _MAJORIZATION_NODE_LIMIT
        and steps == 200
    ):
        return build_maxent_stress_majorization_pipeline(steps=steps)
    return build_maxent_stress_gradient_pipeline(
        steps=steps,
        alpha=alpha,
        use_entropy=use_entropy,
    )


def layout_maxent_stress_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 200,
    alpha: float = 1.0,
    seed: int = 42,
    use_entropy: bool = False,
    use_majorization: bool = True,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the maxent-stress pipeline as a drop-in replacement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor used to select output device.
    steps : int, default=200
        Number of optimization iterations.
    alpha : float, default=1.0
        Entropy repulsion weight.
    seed : int, default=42
        Random seed.
    use_entropy : bool, default=False
        Whether to include entropy repulsion term.
    use_majorization : bool, default=True
        Prefer majorization for eligible graphs.
    edge_weights : torch.Tensor, optional
        Optional per-edge weights with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Final node positions with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If inputs are invalid.
    RuntimeError
        If pipeline execution does not produce final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if alpha < 0:
        raise ValueError("alpha must be non-negative.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
            )

    device = layout_device(edge_index=edge_index, node_sizes=node_sizes)
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32, device=device)

    pipeline = build_maxent_stress_pipeline(
        steps=steps,
        alpha=alpha,
        use_entropy=use_entropy,
        use_majorization=use_majorization,
        num_nodes=num_nodes,
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
    final_state = pipeline.apply(problem, state, ctx)

    if final_state.pos is None:
        raise RuntimeError("MaxEnt-stress pipeline did not produce final positions.")
    return final_state.pos


__all__ = [
    "build_maxent_stress_gradient_pipeline",
    "build_maxent_stress_majorization_pipeline",
    "build_maxent_stress_pipeline",
    "layout_maxent_stress_pipeline",
]
