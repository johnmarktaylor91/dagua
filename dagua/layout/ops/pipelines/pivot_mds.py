"""Pivot MDS expressed as a composable ops pipeline."""

from __future__ import annotations

from typing import ClassVar, Optional, Tuple

import torch

from dagua.layout.classic.pivot_mds import (
    _build_undirected_adjacency,
    _layout_device,
    _layout_extent,
    _normalize_positions,
    _pivot_mds_coordinates,
    _select_pivots,
)
from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory


class _PreparePivotMDSState(Op):
    """Populate pivot-MDS adjacency and pivot-distance caches."""

    name: ClassVar[str] = "pivot_mds_prepare_state"
    category: ClassVar[OpCategory] = OpCategory.DISTANCE
    writes: ClassVar[Tuple[str, ...]] = ("pos", "pivot_indices", "pivot_distances")

    def __init__(self, n_pivots: int) -> None:
        """Store the pivot-selection budget.

        Parameters
        ----------
        n_pivots : int
            Maximum number of pivots to select.

        Returns
        -------
        None
            This constructor stores configuration only.
        """
        self.n_pivots = int(n_pivots)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Select pivots and cache their graph distances.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this deterministic op.

        Returns
        -------
        SolveState
            State with pivot-MDS caches or trivial positions for graphs with
            fewer than two nodes.

        Raises
        ------
        ValueError
            If ``n_pivots`` is not positive.
        """
        del ctx

        if self.n_pivots <= 0:
            raise ValueError("n_pivots must be positive.")

        if problem.num_nodes == 0:
            state.pos = torch.empty((0, 2), dtype=torch.float32)
            state.pivot_indices = torch.empty((0,), dtype=torch.long)
            state.pivot_distances = torch.empty((0, 0), dtype=torch.float32)
            return state
        if problem.num_nodes == 1:
            state.pos = torch.zeros((1, 2), dtype=torch.float32)
            state.pivot_indices = torch.tensor([0], dtype=torch.long)
            state.pivot_distances = torch.zeros((1, 1), dtype=torch.float32)
            return state

        adjacency = _build_undirected_adjacency(
            edge_index=problem.edge_index,
            num_nodes=problem.num_nodes,
            edge_weights=problem.edge_weights,
        )
        pivot_indices, pivot_distances = _select_pivots(
            adjacency=adjacency,
            n_pivots=min(self.n_pivots, problem.num_nodes),
            seed=problem.seed,
            weighted=problem.edge_weights is not None,
        )
        state.pivot_indices = pivot_indices
        state.pivot_distances = pivot_distances
        return state


class _EmbedPivotMDS(Op):
    """Recover the raw 2D embedding from pivot distances."""

    name: ClassVar[str] = "pivot_mds_embed"
    category: ClassVar[OpCategory] = OpCategory.EMBED
    reads: ClassVar[Tuple[str, ...]] = ("pos", "pivot_distances")
    writes: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run the classic SVD embedding step.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state containing pivot distances.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this deterministic op.

        Returns
        -------
        SolveState
            State with raw pivot-MDS coordinates stored in ``state.pos``.

        Raises
        ------
        RuntimeError
            If the pivot-distance matrix is missing.
        """
        del problem, ctx

        if state.pos is not None:
            return state
        if state.pivot_distances is None:
            raise RuntimeError("_EmbedPivotMDS requires state.pivot_distances to be set.")

        state.pos = _pivot_mds_coordinates(state.pivot_distances)
        return state


class _FinalizePivotMDSPositions(Op):
    """Apply the classic Pivot-MDS centering and extent normalization."""

    name: ClassVar[str] = "pivot_mds_finalize_positions"
    category: ClassVar[OpCategory] = OpCategory.POSTPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Normalize the embedding like classic ``layout_pivot_mds``.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state containing raw coordinates.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with final output positions.

        Raises
        ------
        ValueError
            If ``state.pos`` is missing.
        """
        del ctx

        if state.pos is None:
            raise ValueError("_FinalizePivotMDSPositions requires state.pos to be set.")

        output_device = _layout_device(
            edge_index=problem.edge_index,
            node_sizes=problem.node_sizes,
        )
        extent = _layout_extent(num_nodes=problem.num_nodes, node_sizes=problem.node_sizes)
        normalized = _normalize_positions(state.pos.to(device=output_device), extent=extent)
        state.pos = normalized.to(dtype=torch.float32, device=output_device)
        return state


def build_pivot_mds_pipeline(n_pivots: int = 50) -> Pipeline:
    """Build a Pivot-MDS pipeline that matches classic ``layout_pivot_mds``.

    Parameters
    ----------
    n_pivots : int, default=50
        Maximum number of pivots to select.

    Returns
    -------
    Pipeline
        Pipeline that reproduces classic Pivot MDS exactly.

    Raises
    ------
    ValueError
        If ``n_pivots`` is not positive.
    """
    if n_pivots <= 0:
        raise ValueError("n_pivots must be positive.")

    return Pipeline(
        [
            _PreparePivotMDSState(n_pivots=n_pivots),
            _EmbedPivotMDS(),
            _FinalizePivotMDSPositions(),
        ],
        name="pivot_mds_pipeline",
    )


def layout_pivot_mds_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    n_pivots: int = 50,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the Pivot-MDS pipeline as a drop-in replacement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor used to scale the final drawing extent.
    n_pivots : int, default=50
        Maximum number of pivots to select.
    seed : int, default=42
        Random seed for the first pivot.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Final position tensor with the same dtype, device, and values as
        classic ``layout_pivot_mds``.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``n_pivots``, or ``edge_weights`` are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if n_pivots <= 0:
        raise ValueError("n_pivots must be positive.")
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
    final_state = build_pivot_mds_pipeline(n_pivots=n_pivots).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("Pivot-MDS pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_pivot_mds_pipeline", "layout_pivot_mds_pipeline"]
