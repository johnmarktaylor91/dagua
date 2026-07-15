"""Cytoscape core CoSE layout pipeline."""

from __future__ import annotations

from typing import Optional

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.cytoscape import (
    CytoscapeCoSEStep,
    CytoscapeFinalize,
    CytoscapeInitialPlacement,
)
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def build_cose_pipeline(
    steps: int = 1000,
    randomize: bool = False,
    node_repulsion: float = 2048.0,
    ideal_edge_length: float = 32.0,
    edge_elasticity: float = 32.0,
    gravity: float = 1.0,
    initial_temp: float = 1000.0,
    cooling_factor: float = 0.99,
    min_temp: float = 1.0,
) -> Pipeline:
    """Build the Cytoscape core CoSE pipeline.

    Parameters
    ----------
    steps : int, default=1000
        Maximum number of force iterations.
    randomize : bool, default=False
        Whether to randomize initial positions.
    node_repulsion : float, default=2048.0
        Cytoscape CoSE node repulsion multiplier.
    ideal_edge_length : float, default=32.0
        Desired edge length.
    edge_elasticity : float, default=32.0
        Edge-force divisor.
    gravity : float, default=1.0
        Gravity force.
    initial_temp : float, default=1000.0
        Initial displacement cap.
    cooling_factor : float, default=0.99
        Temperature multiplier per iteration.
    min_temp : float, default=1.0
        Stop threshold. Used to compute the effective repeat count.

    Returns
    -------
    Pipeline
        Composable CoSE pipeline.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    temperatures = []
    temperature = initial_temp
    for _ in range(steps):
        if temperature < min_temp:
            break
        temperatures.append(temperature)
        temperature *= cooling_factor
    ops = [CytoscapeInitialPlacement(randomize=randomize)]
    ops.extend(
        CytoscapeCoSEStep(
            ideal_edge_length=ideal_edge_length,
            node_repulsion=node_repulsion,
            edge_elasticity=edge_elasticity,
            gravity=gravity,
            temperature=temperature,
        )
        for temperature in temperatures
    )
    ops.append(CytoscapeFinalize())
    return Pipeline(ops, name="cose_pipeline")


def layout_cose_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 1000,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    quality: str = "default",
    randomize: bool = False,
    nodeRepulsion: float = 2048.0,
    idealEdgeLength: float = 32.0,
    edgeElasticity: float = 32.0,
    gravity: float = 1.0,
    initialTemp: float = 1000.0,
    coolingFactor: float = 0.99,
    minTemp: float = 1.0,
    pos: Optional[torch.Tensor] = None,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run the Cytoscape core CoSE pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor | None, optional
        Node-size tensor with shape ``[N, 2]``.
    steps : int, default=1000
        Maximum number of force iterations.
    seed : int, default=42
        Seed for randomized initial placement.
    edge_weights : torch.Tensor | None, optional
        Accepted for API consistency; CoSE ignores weights.
    quality : str, default="default"
        Variant label accepted for family consistency.
    randomize : bool, default=False
        Whether to randomize initial positions.
    nodeRepulsion : float, default=2048.0
        Cytoscape CoSE node repulsion multiplier.
    idealEdgeLength : float, default=32.0
        Desired edge length.
    edgeElasticity : float, default=32.0
        Edge-force divisor.
    gravity : float, default=1.0
        Gravity force.
    initialTemp : float, default=1000.0
        Initial displacement cap.
    coolingFactor : float, default=0.99
        Temperature multiplier per iteration.
    minTemp : float, default=1.0
        Stop threshold.
    pos : torch.Tensor | None, optional
        Optional warm-start positions with shape ``[N, 2]``.
    fidelity_dtype : torch.dtype | None, optional
        Optional output dtype override.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    del edge_weights, quality
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        seed=seed,
    )
    state = SolveState(pos=pos)
    state = build_cose_pipeline(
        steps=steps,
        randomize=randomize,
        node_repulsion=nodeRepulsion,
        ideal_edge_length=idealEdgeLength,
        edge_elasticity=edgeElasticity,
        gravity=gravity,
        initial_temp=initialTemp,
        cooling_factor=coolingFactor,
        min_temp=minTemp,
    ).apply(problem, state, RuntimeContext(plan=ExecutionPlan(device="cpu")))
    if state.pos is None:
        raise RuntimeError("CoSE pipeline did not produce positions.")
    if fidelity_dtype is not None:
        return state.pos.to(dtype=fidelity_dtype)
    return state.pos


__all__ = ["build_cose_pipeline", "layout_cose_pipeline"]
