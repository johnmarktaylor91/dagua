"""Reingold-Tilford tree layout expressed as a composable ops pipeline."""

from __future__ import annotations

import sys
from typing import ClassVar, Optional, Tuple

import torch

from dagua.layout.classic.reingold_tilford import (
    _assign_preliminary_x,
    _bfs_forest,
    _layout_device,
    _node_spacing,
)
from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory


class _PrepareTreeState(Op):
    """Extract the BFS forest and compute spacing from the graph topology."""

    name: ClassVar[str] = "rt_prepare_tree_state"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build the BFS forest and cache spacing parameters.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs containing edge topology and node sizes.
        state : SolveState
            Mutable solve state receiving cached tree metadata.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this deterministic op.

        Returns
        -------
        SolveState
            State with BFS forest and spacing cached in ``extras``.
        """
        del ctx

        if problem.num_nodes == 0:
            state.pos = torch.empty((0, 2), dtype=torch.float32)
            return state

        sys.setrecursionlimit(max(sys.getrecursionlimit(), problem.num_nodes * 2))

        roots, children, depths = _bfs_forest(
            edge_index=problem.edge_index,
            num_nodes=problem.num_nodes,
        )
        sibling_spacing = _node_spacing(node_sizes=problem.node_sizes, axis=0, default=1.0)
        layer_spacing = _node_spacing(node_sizes=problem.node_sizes, axis=1, default=1.5)
        component_gap = sibling_spacing * 2.0

        state.extras["rt_roots"] = roots
        state.extras["rt_children"] = children
        state.extras["rt_depths"] = depths
        state.extras["rt_sibling_spacing"] = sibling_spacing
        state.extras["rt_layer_spacing"] = layer_spacing
        state.extras["rt_component_gap"] = component_gap
        return state


class _AssignTreePositions(Op):
    """Run the Walker/Buchheim first and second walks to assign coordinates."""

    name: ClassVar[str] = "rt_assign_tree_positions"
    category: ClassVar[OpCategory] = OpCategory.EMBED
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("pos",)

    def __init__(self, horizontal: bool = False) -> None:
        """Store the orientation flag.

        Parameters
        ----------
        horizontal : bool, default=False
            If ``True``, rotate the final layout so depth grows along x.

        Returns
        -------
        None
            This constructor stores configuration only.
        """
        self.horizontal = horizontal

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Assign Walker x and depth-based y positions for every node.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with cached tree metadata.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this deterministic op.

        Returns
        -------
        SolveState
            State with ``pos`` populated as a float32 tensor.

        Raises
        ------
        RuntimeError
            If the BFS forest metadata is missing from ``state.extras``.
        """
        del ctx

        # Degenerate case already handled by _PrepareTreeState.
        if state.pos is not None:
            return state

        roots = state.extras.get("rt_roots")
        children = state.extras.get("rt_children")
        depths = state.extras.get("rt_depths")
        sibling_spacing = state.extras.get("rt_sibling_spacing")
        layer_spacing = state.extras.get("rt_layer_spacing")
        component_gap = state.extras.get("rt_component_gap")

        if roots is None or children is None or depths is None:
            raise RuntimeError(
                "_AssignTreePositions requires rt_roots, rt_children, rt_depths in state.extras."
            )

        preliminary_x = [0.0] * problem.num_nodes
        next_component_x = 0.0
        for root in roots:
            next_component_x = _assign_preliminary_x(
                root_idx=root,
                children=children,
                depths=depths,
                preliminary_x=preliminary_x,
                component_offset=next_component_x,
                component_gap=component_gap,
            )

        positions = torch.zeros((problem.num_nodes, 2), dtype=torch.float32)
        for node_idx in range(problem.num_nodes):
            positions[node_idx, 0] = float(preliminary_x[node_idx]) * sibling_spacing
            positions[node_idx, 1] = float(depths[node_idx]) * layer_spacing

        positions -= positions.mean(dim=0, keepdim=True)
        if self.horizontal:
            positions = positions[:, [1, 0]]

        state.pos = positions
        return state


class _FinalizeTreePositions(Op):
    """Move positions to the output device matching classic behavior."""

    name: ClassVar[str] = "rt_finalize_positions"
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
        """Cast positions to the classic output device.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state containing final positions.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with positions on the correct output device.

        Raises
        ------
        ValueError
            If ``state.pos`` is missing.
        """
        del ctx

        if state.pos is None:
            raise ValueError("_FinalizeTreePositions requires state.pos to be set.")

        output_device = _layout_device(
            edge_index=problem.edge_index,
            node_sizes=problem.node_sizes,
        )
        state.pos = state.pos.to(device=output_device)
        return state


def build_reingold_tilford_pipeline(horizontal: bool = False) -> Pipeline:
    """Build a pipeline that is bit-identical to classic ``layout_reingold_tilford``.

    Parameters
    ----------
    horizontal : bool, default=False
        If ``True``, rotate the final layout so depth grows along the x-axis.

    Returns
    -------
    Pipeline
        Pipeline that reproduces classic Reingold-Tilford layout exactly:
        BFS forest extraction, Walker/Buchheim coordinate assignment, and
        device finalization.
    """
    return Pipeline(
        [
            _PrepareTreeState(),
            _AssignTreePositions(horizontal=horizontal),
            _FinalizeTreePositions(),
        ],
        name="reingold_tilford_pipeline",
    )


def layout_reingold_tilford_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: int = 42,
    horizontal: bool = False,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the Reingold-Tilford pipeline as a drop-in replacement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]`` used to choose
        stable layer and sibling spacing.
    seed : int, default=42
        Accepted for interface compatibility. This layout is deterministic.
    horizontal : bool, default=False
        If ``True``, rotate the final layout so depth grows along the x-axis.
    edge_weights : torch.Tensor, optional
        Accepted for interface compatibility and validated when provided.

    Returns
    -------
    torch.Tensor
        Final position tensor with the same dtype, device, and values as
        classic ``layout_reingold_tilford``.

    Raises
    ------
    ValueError
        If ``num_nodes`` is negative or ``edge_weights`` has the wrong shape.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    _ = seed

    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
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
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_reingold_tilford_pipeline(horizontal=horizontal).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("Reingold-Tilford pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_reingold_tilford_pipeline", "layout_reingold_tilford_pipeline"]
