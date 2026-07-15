"""Deterministic Graphviz osage-style packing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional

import torch

from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.graph_utils import layout_device
from dagua.layout.ops.networkx_simple import graphviz_osage_array_positions
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op


@register_op
@dataclass
class OsageArrayPackLayout(Op):
    """Assign Graphviz osage-style array-packed coordinates.

    Parameters
    ----------
    separation : float, default=4.0
        Point gap between packed node boxes.
    dtype : torch.dtype, default=torch.float64
        Output floating-point dtype.
    """

    separation: float = 4.0
    dtype: torch.dtype = torch.float64

    name: ClassVar[str] = "graphviz_osage_array_layout"
    category: ClassVar[OpCategory] = OpCategory.COORDINATE
    reads: ClassVar[tuple[str, ...]] = ("N", "node_sizes")
    writes: ClassVar[tuple[str, ...]] = ("pos", "extras")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Assign final osage-style coordinates to ``state.pos``.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs, including node count ``N`` and optional
            ``node_sizes`` with shape ``[N, 2]``.
        state : SolveState
            Mutable solve state receiving ``pos`` and provenance extras.
        ctx : RuntimeContext
            Runtime context; accepted for composable-op API consistency.

        Returns
        -------
        SolveState
            State with osage-style positions and metadata populated.
        """
        del ctx
        device = layout_device(problem.edge_index, problem.node_sizes)
        state.pos = graphviz_osage_array_positions(
            num_nodes=problem.num_nodes,
            node_sizes=problem.node_sizes,
            dtype=self.dtype,
            device=device,
            separation=self.separation,
        )
        state.extras["osage"] = {
            "packmode": "array",
            "separation": self.separation,
        }
        return state


def build_osage_pipeline(
    scale: float = 1.0,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> Pipeline:
    """Build the deterministic osage-style packing pipeline.

    Parameters
    ----------
    scale : float, default=1.0
        Accepted for API consistency; Graphviz osage uses node-size units.
    fidelity_dtype : torch.dtype | None, optional
        Output dtype for direct fidelity checks.

    Returns
    -------
    Pipeline
        Single-stage composable coordinate pipeline.
    """
    del scale
    return Pipeline(
        [OsageArrayPackLayout(dtype=torch.float64 if fidelity_dtype is None else fidelity_dtype)],
        name="osage_pipeline",
    )


def layout_osage_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: Optional[int] = 42,
    edge_weights: Optional[torch.Tensor] = None,
    scale: float = 1.0,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run deterministic osage-style node packing.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor | None, optional
        Optional node-size tensor with shape ``[N, 2]``.
    seed : int | None, default=42
        Accepted for API consistency; osage layout is deterministic.
    edge_weights : torch.Tensor | None, optional
        Accepted for API consistency; osage layout ignores weights.
    scale : float, default=1.0
        Layout half-width.
    fidelity_dtype : torch.dtype | None, optional
        Output dtype for direct fidelity checks.

    Returns
    -------
    torch.Tensor
        Coordinate tensor with shape ``[N, 2]``.
    """
    del seed, edge_weights
    problem = LayoutProblem(edge_index=edge_index, num_nodes=num_nodes, node_sizes=node_sizes)
    state = build_osage_pipeline(scale=scale, fidelity_dtype=fidelity_dtype).apply(
        problem,
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device="cpu")),
    )
    if state.pos is None:
        raise RuntimeError("Osage pipeline did not produce positions.")
    return state.pos


__all__ = ["OsageArrayPackLayout", "build_osage_pipeline", "layout_osage_pipeline"]
