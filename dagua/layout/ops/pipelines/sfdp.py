"""SFDP (Scalable Force-Directed Placement) expressed as a composable ops pipeline.

Hu's multilevel spring-electrical algorithm:
1. Build undirected weighted graph
2. Coarsen via heavy-edge matching until small enough
3. Random-init the coarsest level
4. Spring-electrical refinement on the coarsest level (adaptive cooling)
5. For each finer level: prolongate, then refine (fixed cooling)
6. Normalize final positions

Every helper function is imported from the classic implementation so
the pipeline produces bit-identical output.
"""

from __future__ import annotations

from typing import ClassVar, Optional, Tuple

import torch

from dagua.layout.classic.sfdp import (
    _DEFAULT_P,
    _DEFAULT_STEP,
    _DEFAULT_THETA,
    _average_edge_length,
    _build_graph,
    _GraphData,
    _heavy_edge_matching,
    _layout_device,
    _layout_extent,
    _normalize_positions,
    _prolongate_positions,
    _random_positions,
    _spring_electrical_layout,
    _validate_inputs,
)
from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory


class _BuildSFDPGraph(Op):
    """Build the undirected weighted graph representation for SFDP."""

    name: ClassVar[str] = "sfdp_build_graph"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build the SFDP graph and store it in extras.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs containing edge topology and weights.
        state : SolveState
            Mutable solve state receiving the graph in ``extras``.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with ``extras["sfdp_base_graph"]`` populated.
        """
        del ctx

        state.extras["sfdp_base_graph"] = _build_graph(
            edge_index=problem.edge_index,
            num_nodes=problem.num_nodes,
            edge_weights=problem.edge_weights,
        )
        return state


class _CoarsenHierarchy(Op):
    """Build the multilevel coarsening hierarchy via heavy-edge matching."""

    name: ClassVar[str] = "sfdp_coarsen_hierarchy"
    category: ClassVar[OpCategory] = OpCategory.COARSEN
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Iteratively coarsen until heavy-edge matching can reduce no further.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs containing ``seed``.
        state : SolveState
            Mutable solve state. Reads ``extras["sfdp_base_graph"]``.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with ``extras["sfdp_graphs"]`` (list of _GraphData from
            finest to coarsest) and ``extras["sfdp_mappings"]`` (list of
            fine_to_coarse tensors) populated. Also stores the generator
            in ``extras["sfdp_generator"]`` for later reuse.
        """
        del ctx

        generator = torch.Generator(device="cpu")
        generator.manual_seed(problem.seed)

        base_graph: _GraphData = state.extras["sfdp_base_graph"]
        graphs: list[_GraphData] = [base_graph]
        mappings: list[torch.Tensor] = []
        current_graph = base_graph

        while True:
            coarsened = _heavy_edge_matching(graph=current_graph, generator=generator)
            if coarsened is None:
                break
            fine_to_coarse, coarse_graph = coarsened
            mappings.append(fine_to_coarse)
            graphs.append(coarse_graph)
            current_graph = coarse_graph

        state.extras["sfdp_graphs"] = graphs
        state.extras["sfdp_mappings"] = mappings
        state.extras["sfdp_generator"] = generator
        return state


class _InitCoarsestPositions(Op):
    """Initialize random positions on the coarsest graph level."""

    name: ClassVar[str] = "sfdp_init_coarsest"
    category: ClassVar[OpCategory] = OpCategory.INIT
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("pos", "extras")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Seed positions and compute initial ideal edge length on the coarsest level.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused directly.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with ``pos`` set to coarsest-level random positions and
            ``extras["sfdp_ideal_length"]`` computed.
        """
        del problem, ctx

        graphs: list[_GraphData] = state.extras["sfdp_graphs"]
        generator: torch.Generator = state.extras["sfdp_generator"]

        coarsest = graphs[-1]
        positions = _random_positions(num_nodes=coarsest.num_nodes, generator=generator)
        ideal_length = _average_edge_length(graph=coarsest, positions=positions)

        state.pos = positions
        state.extras["sfdp_ideal_length"] = ideal_length
        return state


class _RefineCoarsest(Op):
    """Run spring-electrical layout on the coarsest level with adaptive cooling."""

    name: ClassVar[str] = "sfdp_refine_coarsest"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("pos", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("pos",)

    steps: int
    theta: float
    repulsive_exponent: float

    def __init__(
        self,
        steps: int = 500,
        theta: float = _DEFAULT_THETA,
        repulsive_exponent: float = _DEFAULT_P,
    ) -> None:
        """Store refinement parameters.

        Parameters
        ----------
        steps : int
            Maximum iterations for the coarsest level.
        theta : float
            Barnes-Hut opening angle threshold.
        repulsive_exponent : float
            SFDP repulsion exponent ``p``.
        """
        self.steps = steps
        self.theta = theta
        self.repulsive_exponent = repulsive_exponent

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run spring-electrical refinement with adaptive cooling on the coarsest level.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused directly.
        state : SolveState
            Mutable solve state with ``pos`` and SFDP extras.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with ``pos`` refined at the coarsest level.
        """
        del problem, ctx

        graphs: list[_GraphData] = state.extras["sfdp_graphs"]
        coarsest = graphs[-1]

        state.pos = _spring_electrical_layout(
            graph=coarsest,
            positions=state.pos,
            ideal_length=state.extras["sfdp_ideal_length"],
            steps=self.steps,
            theta=self.theta,
            repulsive_exponent=self.repulsive_exponent,
            adaptive_cooling=True,
            step_init=_DEFAULT_STEP,
        )
        return state


class _ProlongateAndRefine(Op):
    """Prolongate from coarser to finer levels and refine each level.

    This op walks the hierarchy from coarsest back to finest, applying
    prolongation (interpolation + smoothing + noise) followed by
    spring-electrical refinement with fixed cooling at each level.
    """

    name: ClassVar[str] = "sfdp_prolongate_and_refine"
    category: ClassVar[OpCategory] = OpCategory.PROLONG
    reads: ClassVar[Tuple[str, ...]] = ("pos", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("pos", "extras")

    steps: int
    theta: float
    repulsive_exponent: float

    def __init__(
        self,
        steps: int = 500,
        theta: float = _DEFAULT_THETA,
        repulsive_exponent: float = _DEFAULT_P,
    ) -> None:
        """Store prolongation and refinement parameters.

        Parameters
        ----------
        steps : int
            Maximum iterations per refinement level.
        theta : float
            Barnes-Hut opening angle threshold.
        repulsive_exponent : float
            SFDP repulsion exponent ``p``.
        """
        self.steps = steps
        self.theta = theta
        self.repulsive_exponent = repulsive_exponent

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Walk the hierarchy from coarsest to finest, prolongating and refining.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused directly.
        state : SolveState
            Mutable solve state with ``pos`` at the coarsest level.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with ``pos`` at the finest (original) level.
        """
        del problem, ctx

        graphs: list[_GraphData] = state.extras["sfdp_graphs"]
        mappings: list[torch.Tensor] = state.extras["sfdp_mappings"]
        generator: torch.Generator = state.extras["sfdp_generator"]
        ideal_length: float = state.extras["sfdp_ideal_length"]

        from dagua.layout.classic.sfdp import _MIN_SPAN, _REFINEMENT_K_DECAY

        positions = state.pos
        for level_index in range(len(mappings) - 1, -1, -1):
            ideal_length = max(ideal_length * _REFINEMENT_K_DECAY, _MIN_SPAN)
            fine_graph = graphs[level_index]
            positions = _prolongate_positions(
                graph=fine_graph,
                fine_to_coarse=mappings[level_index],
                coarse_positions=positions,
                ideal_length=ideal_length,
                generator=generator,
            )
            positions = _spring_electrical_layout(
                graph=fine_graph,
                positions=positions,
                ideal_length=ideal_length,
                steps=self.steps,
                theta=self.theta,
                repulsive_exponent=self.repulsive_exponent,
                adaptive_cooling=False,
                step_init=_DEFAULT_STEP,
            )

        state.pos = positions
        state.extras["sfdp_ideal_length"] = ideal_length
        return state


class _FinalizeSFDPPositions(Op):
    """Normalize and cast final SFDP positions."""

    name: ClassVar[str] = "sfdp_finalize_positions"
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
        """Center, normalize, scale, and cast positions like classic SFDP.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs used for device and extent.
        state : SolveState
            Mutable solve state containing final positions.
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
            raise ValueError("_FinalizeSFDPPositions requires state.pos to be set.")

        device = _layout_device(
            edge_index=problem.edge_index,
            node_sizes=problem.node_sizes,
        )

        extent = _layout_extent(num_nodes=problem.num_nodes, node_sizes=problem.node_sizes)
        normalized = _normalize_positions(positions=state.pos, extent=extent)
        state.pos = normalized.to(dtype=torch.float32, device=device)
        return state


def build_sfdp_pipeline(
    steps: int = 500,
    theta: float = _DEFAULT_THETA,
    repulsive_exponent: float = _DEFAULT_P,
) -> Pipeline:
    """Build an SFDP pipeline that is bit-identical to classic ``layout_sfdp``.

    Parameters
    ----------
    steps : int, default=500
        Maximum number of spring-electrical iterations per level.
    theta : float, default=0.6
        Barnes-Hut opening angle threshold.
    repulsive_exponent : float, default=-1.0
        SFDP repulsion exponent ``p``.

    Returns
    -------
    Pipeline
        Pipeline that reproduces classic SFDP's multilevel coarsening,
        spring-electrical refinement, prolongation, and normalization.

    Raises
    ------
    ValueError
        If ``steps`` is negative.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative.")

    return Pipeline(
        [
            _BuildSFDPGraph(),
            _CoarsenHierarchy(),
            _InitCoarsestPositions(),
            _RefineCoarsest(
                steps=steps,
                theta=theta,
                repulsive_exponent=repulsive_exponent,
            ),
            _ProlongateAndRefine(
                steps=steps,
                theta=theta,
                repulsive_exponent=repulsive_exponent,
            ),
            _FinalizeSFDPPositions(),
        ],
        name="sfdp_pipeline",
    )


def layout_sfdp_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 500,
    seed: int = 123,
    theta: float = _DEFAULT_THETA,
    repulsive_exponent: float = _DEFAULT_P,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the SFDP pipeline as a drop-in replacement for classic ``layout_sfdp``.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor used only for output scaling.
    steps : int, default=500
        Maximum number of spring-electrical iterations per level.
    seed : int, default=123
        Random seed for coarsening order, coarsest initialization, and
        prolongation noise.
    theta : float, default=0.6
        Barnes-Hut opening angle threshold.
    repulsive_exponent : float, default=-1.0
        SFDP repulsion exponent ``p``.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Final position tensor with the same dtype, device, and values as
        classic ``layout_sfdp``.

    Raises
    ------
    ValueError
        If inputs are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    _validate_inputs(
        edge_index=edge_index,
        num_nodes=num_nodes,
        steps=steps,
        edge_weights=edge_weights,
    )

    device = _layout_device(edge_index=edge_index, node_sizes=node_sizes)
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32, device=device)

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_sfdp_pipeline(
        steps=steps,
        theta=theta,
        repulsive_exponent=repulsive_exponent,
    ).apply(problem, state, ctx)

    if final_state.pos is None:
        raise RuntimeError("SFDP pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_sfdp_pipeline", "layout_sfdp_pipeline"]
