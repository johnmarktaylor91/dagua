"""d3-force layout pipeline."""

from __future__ import annotations

from typing import Optional

import torch

from dagua.layout.ops.base import Pipeline, Repeat
from dagua.layout.ops.d3force import (
    D3ForceCenter,
    D3ForceConfig,
    D3ForceInitialize,
    D3ForceIntegrate,
    D3ForceLink,
    D3ForceManyBody,
    D3ForceUpdateAlpha,
)
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def build_d3force_pipeline(
    config: Optional[D3ForceConfig] = None,
    dtype: torch.dtype = torch.float64,
) -> Pipeline:
    """Build the composable d3-force operation pipeline.

    Parameters
    ----------
    config : D3ForceConfig, optional
        d3-force parameter bundle. ``None`` uses d3 defaults.
    dtype : torch.dtype, default=torch.float64
        Internal coordinate dtype.

    Returns
    -------
    Pipeline
        Pipeline composed from explicit d3-force stages.
    """
    resolved = D3ForceConfig() if config is None else config
    tick_ops = [
        D3ForceUpdateAlpha(),
        D3ForceLink(resolved),
        D3ForceManyBody(resolved),
        D3ForceCenter(enabled=resolved.center),
        D3ForceIntegrate(resolved),
    ]
    return Pipeline(
        [
            D3ForceInitialize(resolved, dtype=dtype),
            Repeat(max(0, int(resolved.ticks)), tick_ops),
        ],
        name="d3force",
    )


def layout_d3force_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: Optional[int] = 1,
    steps: int = 0,
    ticks: Optional[int] = None,
    many_body_strength: float = -30.0,
    link_distance: float = 30.0,
    link_iterations: int = 1,
    velocity_decay: float = 0.4,
    theta: float = 0.9,
    center: bool = True,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run d3-force with default link, many-body, and center forces.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Accepted for pipeline signature consistency; d3-force ignores sizes.
    seed : int, optional
        LCG seed. ``1`` matches d3-force's built-in source.
    steps : int, default=0
        LayoutConfig-forwarded step count. Used only when ``ticks`` is
        ``None`` and positive.
    ticks : int, optional
        Explicit number of simulation ticks. Defaults to d3's 300-tick
        alpha-min horizon.
    many_body_strength : float, default=-30.0
        Constant many-body strength.
    link_distance : float, default=30.0
        Constant link distance.
    link_iterations : int, default=1
        Link force iterations per tick.
    velocity_decay : float, default=0.4
        Public d3 ``simulation.velocityDecay`` value. The integrator uses
        ``1 - velocity_decay``.
    theta : float, default=0.9
        Barnes-Hut theta option, retained for variant compatibility.
    center : bool, default=True
        Whether to apply d3 ``forceCenter(0, 0)``.
    fidelity_dtype : torch.dtype, optional
        Internal coordinate dtype. ``None`` uses ``torch.float64``.

    Returns
    -------
    torch.Tensor
        Final positions with shape ``[N, 2]``.
    """
    del node_sizes
    dtype = torch.float64 if fidelity_dtype is None else fidelity_dtype
    resolved_ticks = int(ticks if ticks is not None else (steps if steps > 0 else 300))
    config = D3ForceConfig(
        ticks=resolved_ticks,
        seed=1 if seed is None else int(seed),
        many_body_strength=float(many_body_strength),
        link_distance=float(link_distance),
        link_iterations=int(link_iterations),
        velocity_decay_factor=1.0 - float(velocity_decay),
        theta=float(theta),
        center=bool(center),
    )
    problem = LayoutProblem(edge_index=edge_index, num_nodes=num_nodes, seed=config.seed)
    state = build_d3force_pipeline(config=config, dtype=dtype).apply(
        problem,
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device="cpu")),
    )
    if state.pos is None:
        return torch.zeros((0, 2), dtype=dtype)
    return state.pos.to(dtype=dtype)


def layout_d3force_default_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: Optional[int] = 1,
    steps: int = 0,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run the named default d3-force variant.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Ignored by d3-force.
    seed : int, optional
        LCG seed.
    steps : int, default=0
        Optional tick count from LayoutConfig.
    fidelity_dtype : torch.dtype, optional
        Internal coordinate dtype.

    Returns
    -------
    torch.Tensor
        Final positions with shape ``[N, 2]``.
    """
    return layout_d3force_pipeline(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        seed=seed,
        steps=steps,
        fidelity_dtype=fidelity_dtype,
    )


def layout_d3force_strong_repulsion_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: Optional[int] = 1,
    steps: int = 0,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run a d3-force variant with stronger many-body repulsion.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Ignored by d3-force.
    seed : int, optional
        LCG seed.
    steps : int, default=0
        Optional tick count from LayoutConfig.
    fidelity_dtype : torch.dtype, optional
        Internal coordinate dtype.

    Returns
    -------
    torch.Tensor
        Final positions with shape ``[N, 2]``.
    """
    return layout_d3force_pipeline(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        seed=seed,
        steps=steps,
        many_body_strength=-80.0,
        fidelity_dtype=fidelity_dtype,
    )
