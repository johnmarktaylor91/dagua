"""NetworkX ARF-layout pipeline."""

from __future__ import annotations

from typing import Optional

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.networkx_simple import NetworkXSimpleLayout
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def build_arf_pipeline(
    scaling: float = 1.0,
    a: float = 1.1,
    etol: float = 1.0e-6,
    dt: float = 1.0e-3,
    max_iter: int = 1000,
    seed: Optional[int] = 42,
) -> Pipeline:
    """Build the NetworkX ARF-layout pipeline.

    Parameters
    ----------
    scaling : float, default=1.0
        Radius scale from NetworkX equation 10.
    a : float, default=1.1
        Spring strength for connected node pairs.
    etol : float, default=1e-6
        Sum-gradient convergence tolerance.
    dt : float, default=1e-3
        Integration timestep.
    max_iter : int, default=1000
        Maximum iteration guard.
    seed : int | None, default=42
        NumPy RandomState seed for the random initializer.

    Returns
    -------
    Pipeline
        Single-stage composable coordinate pipeline.
    """
    return Pipeline(
        [
            NetworkXSimpleLayout(
                "arf",
                {
                    "scaling": scaling,
                    "a": a,
                    "etol": etol,
                    "dt": dt,
                    "max_iter": max_iter,
                    "seed": seed,
                },
            )
        ],
        name="arf_pipeline",
    )


def layout_arf_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: Optional[int] = 42,
    edge_weights: Optional[torch.Tensor] = None,
    scaling: float = 1.0,
    a: float = 1.1,
    etol: float = 1.0e-6,
    dt: float = 1.0e-3,
    max_iter: int = 1000,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run the deterministic seeded NetworkX ARF-layout source port.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor | None, optional
        Optional node-size tensor with shape ``[N, 2]``.
    seed : int | None, default=42
        NumPy RandomState seed for the random initializer.
    edge_weights : torch.Tensor | None, optional
        Accepted for API consistency; ARF ignores weights.
    scaling : float, default=1.0
        Radius scale from NetworkX equation 10.
    a : float, default=1.1
        Spring strength for connected node pairs.
    etol : float, default=1e-6
        Sum-gradient convergence tolerance.
    dt : float, default=1e-3
        Integration timestep.
    max_iter : int, default=1000
        Maximum iteration guard.
    fidelity_dtype : torch.dtype | None, optional
        Output dtype for direct fidelity checks.

    Returns
    -------
    torch.Tensor
        Coordinate tensor with shape ``[N, 2]``.
    """
    del edge_weights
    problem = LayoutProblem(edge_index=edge_index, num_nodes=num_nodes, node_sizes=node_sizes)
    state = build_arf_pipeline(
        scaling=scaling,
        a=a,
        etol=etol,
        dt=dt,
        max_iter=max_iter,
        seed=seed,
    ).apply(problem, SolveState(), RuntimeContext(plan=ExecutionPlan(device="cpu")))
    if state.pos is None:
        raise RuntimeError("ARF pipeline did not produce positions.")
    return state.pos.to(dtype=fidelity_dtype) if fidelity_dtype is not None else state.pos


__all__ = ["build_arf_pipeline", "layout_arf_pipeline"]
