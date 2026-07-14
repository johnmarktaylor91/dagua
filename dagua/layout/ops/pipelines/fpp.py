"""OGDF-style FPP planar-grid layout pipeline without runtime delegation."""

from __future__ import annotations

from typing import ClassVar, Optional, Tuple

import torch

from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.pipelines.planar import (
    PlanarEmbedding,
    PlanarityCheck,
    PlanarityError,
    check_planarity,
    combinatorial_embedding_to_pos,
)
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory

_FPP_RAW_POS_KEY = "fpp_raw_pos"
_OGDF_GRID_SEPARATION = 40.0


class FPPShiftPlacement(Op):
    """Compute FPP integer-grid positions from a planar embedding."""

    name: ClassVar[str] = "fpp_shift_placement"
    category: ClassVar[OpCategory] = OpCategory.INIT
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(self, problem: LayoutProblem, state: SolveState, ctx: RuntimeContext) -> SolveState:
        """Store raw FPP grid coordinates.

        Parameters
        ----------
        problem : LayoutProblem
            Layout inputs, unused after planarity preprocessing.
        state : SolveState
            Mutable state containing ``planar_embedding``.
        ctx : RuntimeContext
            Runtime context, unused.

        Returns
        -------
        SolveState
            State containing raw integer coordinates keyed by node id.
        """
        del problem, ctx
        embedding = state.extras.get("planar_embedding")
        if not isinstance(embedding, PlanarEmbedding):
            raise RuntimeError("Planar embedding is missing from FPP pipeline state.")
        state.extras[_FPP_RAW_POS_KEY] = combinatorial_embedding_to_pos(
            embedding,
            fully_triangulate=True,
        )
        return state


class OGDFGridMap(Op):
    """Map integer grid coordinates like OGDF GridLayoutModule."""

    name: ClassVar[str] = "ogdf_grid_map"
    category: ClassVar[OpCategory] = OpCategory.POSTPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(self, problem: LayoutProblem, state: SolveState, ctx: RuntimeContext) -> SolveState:
        """Populate ``state.pos`` with OGDF-mapped grid coordinates.

        Parameters
        ----------
        problem : LayoutProblem
            Layout inputs containing node count and output device.
        state : SolveState
            Mutable state containing raw grid coordinates.
        ctx : RuntimeContext
            Runtime context, unused.

        Returns
        -------
        SolveState
            State with ``pos`` set to shape ``[N, 2]``.
        """
        del ctx
        raw_pos = state.extras.get(_FPP_RAW_POS_KEY)
        if not isinstance(raw_pos, dict):
            raise RuntimeError("Raw FPP positions are missing from pipeline state.")
        output = torch.zeros((problem.num_nodes, 2), dtype=torch.float64)
        y_max = max((int(value[1]) for value in raw_pos.values()), default=0)
        for node in range(problem.num_nodes):
            x_coord, y_coord = raw_pos.get(node, (0, 0))
            output[node, 0] = float(x_coord) * _OGDF_GRID_SEPARATION
            output[node, 1] = float(y_max - int(y_coord)) * _OGDF_GRID_SEPARATION
        state.pos = output.to(device=problem.edge_index.device)
        return state


def build_fpp_pipeline() -> Pipeline:
    """Build the deterministic FPP planar-grid pipeline.

    Returns
    -------
    Pipeline
        Planarity, shift-placement, and OGDF grid-map stages.
    """
    return Pipeline([PlanarityCheck(), FPPShiftPlacement(), OGDFGridMap()], name="fpp_pipeline")


def layout_fpp_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: Optional[int] = 42,
    edge_weights: Optional[torch.Tensor] = None,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run FPP planar-grid layout without calling the OGDF runner.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor | None, optional
        Accepted for API consistency; FPP grid layout ignores sizes before
        OGDF's fixed grid mapping.
    seed : int | None, default=42
        Accepted for API consistency; FPP is deterministic.
    edge_weights : torch.Tensor | None, optional
        Accepted for API consistency; FPP ignores weights.
    fidelity_dtype : torch.dtype | None, optional
        Optional output dtype override.

    Returns
    -------
    torch.Tensor
        Coordinate tensor with shape ``[N, 2]``.
    """
    del seed, edge_weights
    is_planar, _embedding = check_planarity(edge_index, num_nodes)
    if not is_planar:
        raise PlanarityError("FPP layout is only defined for planar graphs.")
    problem = LayoutProblem(edge_index=edge_index, num_nodes=num_nodes, node_sizes=node_sizes)
    state = build_fpp_pipeline().apply(
        problem,
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device="cpu")),
    )
    if state.pos is None:
        raise RuntimeError("FPP pipeline did not produce positions.")
    if fidelity_dtype is not None:
        return state.pos.to(dtype=fidelity_dtype)
    return state.pos


__all__ = ["FPPShiftPlacement", "OGDFGridMap", "build_fpp_pipeline", "layout_fpp_pipeline"]
