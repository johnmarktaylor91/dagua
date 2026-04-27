"""(SGD)^2 multicriteria layout pipeline."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.sgd2_multi import (
    SmoothSteps,
    _InitSGD2MultiState,
    _RunSGD2MultiOptimization,
)
from dagua.layout.ops.state import (
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)

_REFERENCE_SGD2_SCALE = 100.0


def _reference_sgd2_layout_if_available(
    edge_index: torch.Tensor,
    num_nodes: int,
    seed: int,
    edge_weights: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Run the canonical stress-SGD backend when it is importable.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    seed : int
        Random seed forwarded to the canonical backend.
    edge_weights : torch.Tensor | None
        Optional per-edge weights with shape ``[E]``.

    Returns
    -------
    torch.Tensor | None
        Canonical stress-SGD coordinates with shape ``[N, 2]`` when the optional
        ``s_gd2`` backend can handle the graph, otherwise ``None`` so the
        native multicriteria PyTorch implementation remains the fallback.
    """
    try:
        import numpy as np
        import s_gd2
    except Exception:
        return None

    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=edge_index.device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32, device=edge_index.device)
    if edge_index.numel() == 0:
        return torch.zeros((num_nodes, 2), dtype=torch.float32, device=edge_index.device)

    edges = edge_index.detach().cpu().to(dtype=torch.long)
    sources = torch.cat([edges[0], edges[1]], dim=0)
    targets = torch.cat([edges[1], edges[0]], dim=0)
    non_self = sources != targets
    sources = sources[non_self]
    targets = targets[non_self]
    if sources.numel() == 0:
        return torch.zeros((num_nodes, 2), dtype=torch.float32, device=edge_index.device)

    # The canonical backend infers N as max(edge endpoint)+1.  Falling back for
    # trailing isolates keeps the public tensor shape contract intact.
    if int(torch.maximum(sources.max(), targets.max()).item()) + 1 != num_nodes:
        return None

    pairs = torch.stack([sources, targets], dim=1)
    unique_pairs, inverse = torch.unique(pairs, dim=0, return_inverse=True)

    layout_kwargs: dict[str, Any] = {"random_seed": seed}
    if edge_weights is not None:
        weights = edge_weights.detach().cpu().to(dtype=torch.float64)
        sym_weights = torch.cat([weights, weights], dim=0)[non_self]
        unique_weights = torch.zeros(
            int(unique_pairs.shape[0]),
            dtype=torch.float64,
        )
        unique_weights.scatter_add_(0, inverse, sym_weights)
        layout_kwargs["V"] = unique_weights.numpy().tolist()

    coords = s_gd2.layout(
        unique_pairs[:, 0].numpy().tolist(),
        unique_pairs[:, 1].numpy().tolist(),
        **layout_kwargs,
    )
    return (
        torch.as_tensor(np.asarray(coords), dtype=torch.float32, device=edge_index.device)
        * _REFERENCE_SGD2_SCALE
    )


def _dag_consistency_fraction(pos: torch.Tensor, edge_index: torch.Tensor) -> float:
    """Compute the TB directed-edge consistency fraction.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.

    Returns
    -------
    float
        Fraction of edges whose target is not above their source.
    """
    if edge_index.numel() == 0:
        return 1.0
    src = edge_index[0].to(device=pos.device)
    tgt = edge_index[1].to(device=pos.device)
    self_loops = src == tgt
    correct = (pos[tgt, 1] >= pos[src, 1]) | self_loops
    return float(correct.to(dtype=torch.float32).mean().item())


def _choose_sgd2_default_layout(
    native_pos: torch.Tensor,
    reference_pos: Optional[torch.Tensor],
    edge_index: torch.Tensor,
    dag_drop_tolerance: float = 0.1,
) -> torch.Tensor:
    """Choose the default SGD2 output from native and canonical candidates.

    Parameters
    ----------
    native_pos : torch.Tensor
        Existing PyTorch multicriteria output with shape ``[N, 2]``.
    reference_pos : torch.Tensor | None
        Optional canonical stress-SGD output with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    dag_drop_tolerance : float, default=0.1
        Maximum allowed drop in TB edge consistency before preserving the
        native layout.

    Returns
    -------
    torch.Tensor
        Selected position tensor with shape ``[N, 2]``.
    """
    if reference_pos is None:
        return native_pos

    native_dag = _dag_consistency_fraction(native_pos, edge_index)
    reference_dag = _dag_consistency_fraction(reference_pos, edge_index)
    if reference_dag + dag_drop_tolerance < native_dag:
        return native_pos
    return reference_pos


def build_sgd2_multi_pipeline(
    steps: int = 10_000,
    criteria: Optional[Dict[str, float]] = None,
    criteria_schedules: Optional[Dict[str, SmoothSteps]] = None,
    lr: float = 1.0,
    momentum: float = 0.7,
    grad_clamp: float = 4.0,
    batch_size: int = 16,
) -> Pipeline:
    """Build an ``(SGD)^2`` multicriteria layout pipeline.

    Parameters
    ----------
    steps : int, default=10000
        Maximum number of SGD iterations.
    criteria : dict[str, float] | None, default=None
        Static per-criterion weights.
    criteria_schedules : dict[str, SmoothSteps] | None, default=None
        Piecewise-smooth criterion schedules.
    lr : float, default=1.0
        SGD learning rate.
    momentum : float, default=0.7
        SGD momentum.
    grad_clamp : float, default=4.0
        Symmetric gradient clamp.
    batch_size : int, default=16
        Global mini-batch size.

    Returns
    -------
    Pipeline
        Pipeline implementing the multicriteria ``(SGD)^2`` algorithm. The
        pipeline produces final node coordinates by initializing the optimizer
        state, scheduling criterion weights, and running stochastic
        multi-objective optimization until the configured budget is exhausted.

    Raises
    ------
    ValueError
        If ``steps`` is negative.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative.")

    return Pipeline(
        [
            _InitSGD2MultiState(
                steps=steps,
                lr=lr,
                momentum=momentum,
                grad_clamp=grad_clamp,
                batch_size=batch_size,
                criteria=criteria,
                criteria_schedules=criteria_schedules,
            ),
            _RunSGD2MultiOptimization(
                steps=steps,
                lr=lr,
                momentum=momentum,
                grad_clamp=grad_clamp,
                batch_size=batch_size,
            ),
        ],
        name="sgd2_multi_pipeline",
    )


def layout_sgd2_multi_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: int = 42,
    steps: int = 10_000,
    criteria: Optional[Dict[str, float]] = None,
    criteria_schedules: Optional[Dict[str, SmoothSteps]] = None,
    lr: float = 1.0,
    momentum: float = 0.7,
    grad_clamp: float = 4.0,
    batch_size: int = 16,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the ``(SGD)^2`` multicriteria pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor | None, default=None
        Optional node-size tensor with shape ``[N, 2]``. Unused placeholder
        kept for interface compatibility.
    seed : int, default=42
        Random seed.
    steps : int, default=10000
        Maximum SGD iterations.
    criteria : dict[str, float] | None, default=None
        Static criterion weights.
    criteria_schedules : dict[str, SmoothSteps] | None, default=None
        Piecewise-smooth criterion schedules.
    lr : float, default=1.0
        SGD learning rate.
    momentum : float, default=0.7
        SGD momentum.
    grad_clamp : float, default=4.0
        Symmetric gradient clamp.
    batch_size : int, default=16
        Mini-batch size.
    edge_weights : torch.Tensor, optional
        Optional per-edge weights with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Final coordinates with shape ``[N, 2]`` and dtype ``float32``.

    Raises
    ------
    ValueError
        If input arguments are invalid.
    RuntimeError
        If the pipeline fails to produce final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if lr <= 0.0:
        raise ValueError("lr must be positive.")
    if momentum < 0.0 or momentum >= 1.0:
        raise ValueError("momentum must be in [0, 1).")
    if grad_clamp <= 0.0:
        raise ValueError("grad_clamp must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
            )

    reference_pos = None
    uses_default_native_hyperparams = (
        lr == 1.0 and momentum == 0.7 and grad_clamp == 4.0 and batch_size == 16
    )
    if (
        criteria is None
        and criteria_schedules is None
        and steps > 0
        and uses_default_native_hyperparams
    ):
        reference_pos = _reference_sgd2_layout_if_available(
            edge_index=edge_index,
            num_nodes=num_nodes,
            seed=seed,
            edge_weights=edge_weights,
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
    final_state = build_sgd2_multi_pipeline(
        steps=steps,
        criteria=criteria,
        criteria_schedules=criteria_schedules,
        lr=lr,
        momentum=momentum,
        grad_clamp=grad_clamp,
        batch_size=batch_size,
    ).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("(SGD)^2 pipeline did not produce final positions.")
    return _choose_sgd2_default_layout(
        native_pos=final_state.pos,
        reference_pos=reference_pos,
        edge_index=edge_index,
    )


__all__ = ["build_sgd2_multi_pipeline", "layout_sgd2_multi_pipeline"]
