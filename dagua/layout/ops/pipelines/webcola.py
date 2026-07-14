"""WebCola constraint/stress placement pipeline."""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from dagua.config import LayoutConfig
from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.state import (
    ExecutionPlan,
    FlexConstraints,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)
from dagua.layout.ops.webcola import (
    BuildWebColaFlexConstraints,
    ColaDescentConfig,
    InitializeWebColaPositions,
    RunWebColaDescent,
    WebColaConstraint,
)
from dagua.layout.resolve import build_flex_constraints

_DEFAULT_STEPS = 50
_DEFAULT_LINK_DISTANCE = 20.0


def build_webcola_pipeline(
    *,
    steps: int = _DEFAULT_STEPS,
    link_distance: float = _DEFAULT_LINK_DISTANCE,
    constrained: bool = False,
    p_stress: bool = False,
    threshold: float = 0.01,
) -> Pipeline:
    """Build the composable WebCola pipeline.

    Parameters
    ----------
    steps : int, default=50
        Number of WebCola Runge-Kutta iterations.
    link_distance : float, default=20.0
        Constant WebCola link distance.
    constrained : bool, default=False
        Whether to build and apply VPSC constraints.
    p_stress : bool, default=False
        Whether to enable WebCola's p-stress matrix for non-edge pairs.
    threshold : float, default=0.01
        WebCola convergence threshold.

    Returns
    -------
    Pipeline
        Pipeline composed from reusable WebCola ops.
    """
    if steps < 0:
        raise ValueError("steps must be nonnegative.")
    ops = [InitializeWebColaPositions(link_distance=link_distance)]
    if constrained:
        ops.append(BuildWebColaFlexConstraints())
    ops.append(
        RunWebColaDescent(
            ColaDescentConfig(
                steps=int(steps),
                link_distance=float(link_distance),
                constrained=bool(constrained),
                p_stress=bool(p_stress),
                threshold=float(threshold),
            )
        )
    )
    return Pipeline(ops, name="webcola_constrained" if constrained else "webcola")


def layout_webcola_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = _DEFAULT_STEPS,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    link_distance: float = _DEFAULT_LINK_DISTANCE,
    threshold: float = 0.01,
    init_pos: Optional[torch.Tensor] = None,
    constraints: Optional[Sequence[WebColaConstraint]] = None,
    config: Optional[LayoutConfig] = None,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run the unconstrained WebCola stress layout.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Node sizes with shape ``[N, 2]``. Accepted for dispatch symmetry.
    steps : int, default=50
        Number of WebCola Runge-Kutta iterations.
    seed : int, default=42
        Deterministic seed stored on ``LayoutProblem`` for API symmetry.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.
    link_distance : float, default=20.0
        Constant WebCola link distance.
    threshold : float, default=0.01
        WebCola convergence threshold.
    init_pos : torch.Tensor, optional
        Optional initial positions with shape ``[N, 2]``.
    constraints : sequence[dict[str, Any]], optional
        Ignored by the unconstrained variant; accepted to keep variant calls
        uniform.
    config : LayoutConfig, optional
        Public layout config. Accepted for dispatch symmetry.
    fidelity_dtype : torch.dtype, optional
        Output dtype. ``None`` preserves float64 internally.

    Returns
    -------
    torch.Tensor
        Final positions with shape ``[N, 2]``.
    """
    del constraints
    return _layout_webcola(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        steps=steps,
        seed=seed,
        edge_weights=edge_weights,
        link_distance=link_distance,
        threshold=threshold,
        init_pos=init_pos,
        constrained=False,
        p_stress=False,
        config=config,
        fidelity_dtype=fidelity_dtype,
    )


def layout_webcola_constrained_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = _DEFAULT_STEPS,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    link_distance: float = _DEFAULT_LINK_DISTANCE,
    threshold: float = 0.01,
    init_pos: Optional[torch.Tensor] = None,
    constraints: Optional[Sequence[WebColaConstraint]] = None,
    config: Optional[LayoutConfig] = None,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run the constrained WebCola stress/VPSC layout.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Node sizes with shape ``[N, 2]``.
    steps : int, default=50
        Number of constrained WebCola iterations.
    seed : int, default=42
        Deterministic seed stored on ``LayoutProblem``.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.
    link_distance : float, default=20.0
        Constant WebCola link distance.
    threshold : float, default=0.01
        WebCola convergence threshold.
    init_pos : torch.Tensor, optional
        Optional initial positions with shape ``[N, 2]``.
    constraints : sequence[dict[str, Any]], optional
        Explicit WebCola constraints.
    config : LayoutConfig, optional
        Public layout config containing optional Flex constraints.
    fidelity_dtype : torch.dtype, optional
        Output dtype. ``None`` preserves float64 internally.

    Returns
    -------
    torch.Tensor
        Final positions with shape ``[N, 2]``.
    """
    return _layout_webcola(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        steps=steps,
        seed=seed,
        edge_weights=edge_weights,
        link_distance=link_distance,
        threshold=threshold,
        init_pos=init_pos,
        constraints=constraints,
        constrained=True,
        p_stress=False,
        config=config,
        fidelity_dtype=fidelity_dtype,
    )


def _layout_webcola(
    *,
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor],
    steps: int,
    seed: int,
    edge_weights: Optional[torch.Tensor],
    link_distance: float,
    threshold: float,
    init_pos: Optional[torch.Tensor],
    constrained: bool,
    p_stress: bool,
    config: Optional[LayoutConfig],
    fidelity_dtype: Optional[torch.dtype],
    constraints: Optional[Sequence[WebColaConstraint]] = None,
) -> torch.Tensor:
    """Shared WebCola pipeline runner.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Node sizes with shape ``[N, 2]``.
    steps : int
        Number of descent iterations.
    seed : int
        Problem seed.
    edge_weights : torch.Tensor, optional
        Edge weights with shape ``[E]``.
    link_distance : float
        Constant link distance.
    threshold : float
        Convergence threshold.
    init_pos : torch.Tensor, optional
        Initial positions with shape ``[N, 2]``.
    constrained : bool
        Whether to apply constraints.
    p_stress : bool
        Whether to apply p-stress.
    config : LayoutConfig, optional
        Optional public config.
    fidelity_dtype : torch.dtype, optional
        Output dtype.
    constraints : sequence[dict[str, Any]], optional
        Explicit WebCola constraints.

    Returns
    -------
    torch.Tensor
        Final positions with shape ``[N, 2]``.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be nonnegative.")
    flex = _resolve_pipeline_flex(config, num_nodes, edge_index.device)
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        flex=flex,
        edge_weights=edge_weights,
        seed=int(seed),
    )
    state = SolveState(
        total_steps=int(steps), extras={"webcola_constraints": list(constraints or [])}
    )
    if init_pos is not None:
        state.pos = init_pos.detach().to(device=edge_index.device, dtype=torch.float64).clone()
        if tuple(state.pos.shape) != (num_nodes, 2):
            raise ValueError("init_pos must have shape [num_nodes, 2].")
        ops = []
        if constrained:
            ops.append(BuildWebColaFlexConstraints())
        ops.append(
            RunWebColaDescent(
                ColaDescentConfig(
                    steps=int(steps),
                    link_distance=float(link_distance),
                    constrained=bool(constrained),
                    p_stress=bool(p_stress),
                    threshold=float(threshold),
                )
            )
        )
        pipeline = Pipeline(ops, name="webcola_warm_start")
    else:
        pipeline = build_webcola_pipeline(
            steps=steps,
            link_distance=link_distance,
            constrained=constrained,
            p_stress=p_stress,
            threshold=threshold,
        )
    result_state = pipeline.apply(
        problem,
        state,
        RuntimeContext(plan=ExecutionPlan(device=str(edge_index.device))),
    )
    if result_state.pos is None:
        raise RuntimeError("WebCola pipeline did not produce positions.")
    if fidelity_dtype is None:
        return result_state.pos
    return result_state.pos.to(dtype=fidelity_dtype)


def _resolve_pipeline_flex(
    config: Optional[LayoutConfig],
    num_nodes: int,
    device: torch.device,
) -> Optional[FlexConstraints]:
    """Resolve public config Flex into pipeline constraints.

    Parameters
    ----------
    config : LayoutConfig, optional
        Public layout config.
    num_nodes : int
        Number of graph nodes.
    device : torch.device
        Target tensor device.

    Returns
    -------
    FlexConstraints or None
        Resolved constraints, when available.
    """
    if config is None or config.flex is None:
        return None
    return build_flex_constraints(config, num_nodes=num_nodes, device=device)
