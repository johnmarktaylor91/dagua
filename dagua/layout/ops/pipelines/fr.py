"""Fruchterman-Reingold expressed as a composable ops pipeline."""

from __future__ import annotations

from math import sqrt
from typing import ClassVar, Optional, Tuple

import torch

from dagua.layout.ops.anneal import InitTemperatureFromExtent, LinearCool
from dagua.layout.ops.base import Op, Pipeline, Repeat  # noqa: E402
from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig, FRConvergenceCheck  # noqa: E402
from dagua.layout.ops.force import ApplyDisplacement, FRCombinedForce
from dagua.layout.ops.graph_utils import (
    initialize_numpy_positions as _initialize_positions,
)
from dagua.layout.ops.graph_utils import (
    layout_device as _layout_device,
)
from dagua.layout.ops.graph_utils import (
    rescale_layout as _rescale_layout,
)
from dagua.layout.ops.state import (  # noqa: E402
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)
from dagua.layout.ops.taxonomy import OpCategory  # noqa: E402

_OUTPUT_SCALE_FACTOR = 50.0


def _adjacency_matrix(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build the directed dense adjacency matrix used by NetworkX FR.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Dense adjacency matrix with shape ``[N, N]`` and dtype ``float64``.

    Raises
    ------
    ValueError
        If ``edge_index`` has an invalid shape or contains out-of-range nodes.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")

    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float64)
    if edge_index.numel() == 0:
        return adjacency

    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    sources = edge_index_cpu[0]
    targets = edge_index_cpu[1]
    if (
        torch.any(sources < 0)
        or torch.any(sources >= num_nodes)
        or torch.any(targets < 0)
        or torch.any(targets >= num_nodes)
    ):
        raise ValueError("edge_index contains a node index outside [0, num_nodes).")

    if edge_weights is not None:
        weights = edge_weights.detach().to(device="cpu", dtype=torch.float64)
        adjacency[sources, targets] = weights
    else:
        adjacency[sources, targets] = 1.0
    return adjacency


class _InitializeFRPositions(Op):
    """Initialize positions exactly like classic FR."""

    name: ClassVar[str] = "fr_initialize_positions"
    category: ClassVar[OpCategory] = OpCategory.INIT
    writes: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Seed ``state.pos`` from NumPy's unit-square random initialization.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs containing ``num_nodes`` and ``seed``.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with ``state.pos`` populated as a CPU ``float64`` tensor.
        """
        del ctx

        state.pos = _initialize_positions(num_nodes=problem.num_nodes, seed=problem.seed)
        return state


class _PrepareFRState(Op):
    """Populate FR-specific cached state required by the force step."""

    name: ClassVar[str] = "fr_prepare_state"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build the dense adjacency matrix and unit-area metadata.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state receiving FR extras.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with FR adjacency and unit area stored in ``extras``.
        """
        del ctx

        if problem.num_nodes == 0:
            state.extras["fr_adjacency"] = torch.zeros((0, 0), dtype=torch.float64)
        else:
            state.extras["fr_adjacency"] = _adjacency_matrix(
                edge_index=problem.edge_index,
                num_nodes=problem.num_nodes,
                edge_weights=problem.edge_weights,
            )
        state.extras["force_area"] = 1.0
        return state


class _FinalizeFRPositions(Op):
    """Apply classic FR's final recenter, rescale, and cast."""

    name: ClassVar[str] = "fr_finalize_positions"
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
        """Center, normalize, scale, and cast positions like classic FR.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state containing final ``float64`` positions.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with final output positions on the classic return device.

        Raises
        ------
        ValueError
            If ``state.pos`` is missing.
        """
        del ctx

        if state.pos is None:
            raise ValueError("_FinalizeFRPositions requires state.pos to be set.")

        output_device = _layout_device(
            edge_index=problem.edge_index,
            node_sizes=problem.node_sizes,
        )
        if state.pos.shape[0] == 0:
            state.pos = torch.empty((0, 2), dtype=torch.float32, device=output_device)
            return state

        scaled = _rescale_layout(state.pos, scale=1.0)
        scaled = scaled * (sqrt(float(max(problem.num_nodes, 1))) * _OUTPUT_SCALE_FACTOR)
        state.pos = scaled.to(dtype=torch.float32, device=output_device)
        return state


def build_fr_pipeline(steps: int = 50) -> Pipeline:
    """Build an FR pipeline that is bit-identical to classic ``layout_fr``.

    Parameters
    ----------
    steps : int, default=50
        Maximum number of FR iterations.

    Returns
    -------
    Pipeline
        Pipeline that reproduces classic FR's initialization, dense force
        update, cooling schedule, convergence rule, and postprocessing.

    Raises
    ------
    ValueError
        If ``steps`` is negative.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative.")

    return Pipeline(
        [
            FixedSteps(FixedStepsConfig(n=steps)),
            _InitializeFRPositions(),
            _PrepareFRState(),
            InitTemperatureFromExtent(),
            Repeat(
                n=steps,
                ops=[
                    FRCombinedForce(),
                    ApplyDisplacement(),
                    FRConvergenceCheck(),
                    LinearCool(),
                ],
            ),
            _FinalizeFRPositions(),
        ],
        name="fr_pipeline",
    )


def layout_fr_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 50,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the FR pipeline as a drop-in replacement for classic ``layout_fr``.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor used only to resolve the output device.
    steps : int, default=50
        Maximum number of FR iterations.
    seed : int, default=42
        Random seed for the NumPy unit-square initialization.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Final position tensor with the same dtype, device, and values as
        classic ``layout_fr``.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``steps``, or ``edge_weights`` are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
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
    final_state = build_fr_pipeline(steps=steps).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("FR pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_fr_pipeline", "layout_fr_pipeline"]
