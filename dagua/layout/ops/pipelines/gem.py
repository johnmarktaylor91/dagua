"""GEM graph-embedder layout pipeline."""

from __future__ import annotations

from typing import Optional, Union

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig
from dagua.layout.ops.gem import (
    GEMBatchedSolve,
    GEMFinalizePositions,
    GEMPrepareState,
    GEMSequentialSolve,
    InitializeGEMPositions,
)
from dagua.layout.ops.pipelines import resolve_fidelity_dtype
from dagua.layout.ops.state import (  # noqa: E402
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)

GEMFidelityMode = Optional[Union[bool, str]]


def build_gem_pipeline(
    max_iters: int = 500,
    fidelity_mode: GEMFidelityMode = False,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> Pipeline:
    """Build a GEM graph-embedder pipeline.

    Reference fidelity
    ------------------
    Targets: OGDF GEMLayout / Frick, Ludwig, and Mehldau (1995), "A Fast
        Adaptive Layout Algorithm for Undirected Graphs".
    Fidelity mode: ``fidelity_mode=True`` or ``"ogdf"`` enables the
        OGDF-compatible seed bridge, update order, node geometry, component
        solving, packing, and final coordinate contract for sequential-size
        graphs.
    Verified at: round_43 smoke mean RMSD 0.000410 after the sequential
        fidelity branch was moved to scalar double state.
    Known divergences:
        - Larger graphs still use Dagua's historical batched approximation.
        - Star seed 43 keeps a topology-sensitive residual: a raw impulse
          delta first appears at update 45, crosses `1e-6` in coordinates at
          update 402, and crosses `1e-3` at update 624.

    Parameters
    ----------
    max_iters : int, default=500
        Maximum number of OGDF node updates.
    fidelity_mode : bool or str or None, default=False
        When ``"ogdf"`` or ``True``, use OGDF runner-compatible semantics for
        sequential-size graphs.

    Returns
    -------
    Pipeline
        Pipeline implementing the GEM algorithm. The pipeline produces final
        node coordinates by initializing node positions, preparing local state,
        performing the classic sequential and batched update phases, and
        finalizing the coordinates.

    Raises
    ------
    ValueError
        If ``max_iters`` is negative.
    """
    if max_iters < 0:
        raise ValueError("max_iters must be non-negative.")

    resolved_dtype = resolve_fidelity_dtype(fidelity_mode, fidelity_dtype)
    return Pipeline(
        [
            FixedSteps(FixedStepsConfig(n=max_iters)),
            InitializeGEMPositions(fidelity_mode=fidelity_mode, fidelity_dtype=resolved_dtype),
            GEMPrepareState(fidelity_mode=fidelity_mode),
            GEMSequentialSolve(fidelity_dtype=resolved_dtype),
            GEMBatchedSolve(),
            GEMFinalizePositions(fidelity_dtype=resolved_dtype if fidelity_mode else torch.float32),
        ],
        name="gem_pipeline",
    )


def layout_gem_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    max_iters: int = 500,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    fidelity_mode: GEMFidelityMode = False,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run the GEM pipeline as a drop-in replacement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]`` used only to resolve
        the output device.
    max_iters : int, default=500
        Maximum number of OGDF node updates.
    seed : int, default=42
        Random seed for initialization and permutations.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.
    fidelity_mode : bool or str or None, default=False
        When ``"ogdf"`` or ``True``, use OGDF runner-compatible semantics for
        sequential-size graphs.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``max_iters``, or ``edge_weights`` are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if max_iters < 0:
        raise ValueError("max_iters must be non-negative.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
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
    final_state = build_gem_pipeline(
        max_iters=max_iters,
        fidelity_mode=fidelity_mode,
        fidelity_dtype=fidelity_dtype,
    ).apply(
        problem,
        state,
        ctx,
    )
    if final_state.pos is None:
        raise RuntimeError("GEM pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_gem_pipeline", "layout_gem_pipeline"]
