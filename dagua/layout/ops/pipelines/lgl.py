"""Large Graph Layout (LGL) expressed as a composable ops pipeline.

The LGL algorithm uses BFS shell growth: vertices are placed one layer at a
time starting from a root, with an FR-style relaxation after each layer.  The
shell placement and incremental edge activation make the per-layer loop
non-decomposable into independent force/apply ops, so the core growth+refine
logic is wrapped in a single op that delegates to classic helpers.

This pipeline delegates to the classic helpers so that output is bit-identical
to ``layout_lgl``.
"""

from __future__ import annotations

import math
import random
from typing import ClassVar, Optional, Tuple

import torch

from dagua.layout.classic.lgl import (
    _MIN_DISTANCE,
    _bfs_layers,
    _build_undirected_graph,
    _harmonic_number,
    _initialize_positions,
    _layout_device,
    _normalize,
    _run_refinement,
    _validate_inputs,
)
from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory


class _InitializeLGLPositions(Op):
    """Initialize positions exactly like classic LGL."""

    name: ClassVar[str] = "lgl_initialize_positions"
    category: ClassVar[OpCategory] = OpCategory.INIT
    writes: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Seed ``state.pos`` from the classic LGL random placement.

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
            State with ``state.pos`` populated as a CPU ``float64`` tensor
            and the root vertex zeroed.
        """
        del ctx

        area = state.extras["lgl_area"]
        radius = math.sqrt(area / math.pi)
        positions = _initialize_positions(
            num_nodes=problem.num_nodes,
            radius=radius,
            seed=problem.seed,
        )
        root_node = state.extras["lgl_root"]
        positions[root_node] = 0.0
        state.pos = positions
        return state


class _PrepareLGLState(Op):
    """Populate LGL-specific cached state required by the shell growth op."""

    name: ClassVar[str] = "lgl_prepare_state"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    _maxiter: int
    _maxdelta: Optional[float]
    _area: Optional[float]
    _coolexp: float
    _repulserad: Optional[float]
    _cellsize: Optional[float]
    _root: Optional[int]

    def __init__(
        self,
        *,
        maxiter: int = 150,
        maxdelta: Optional[float] = None,
        area: Optional[float] = None,
        coolexp: float = 1.5,
        repulserad: Optional[float] = None,
        cellsize: Optional[float] = None,
        root: Optional[int] = None,
    ) -> None:
        """Store LGL configuration parameters.

        Parameters
        ----------
        maxiter : int, default=150
            Maximum refinement iterations per BFS layer.
        maxdelta : float, optional
            Maximum initial movement magnitude.
        area : float, optional
            Layout area.
        coolexp : float, default=1.5
            Power-law cooling exponent.
        repulserad : float, optional
            Modified FR repulsion cutoff.
        cellsize : float, optional
            Sparse-grid cell size.
        root : int, optional
            BFS root vertex.

        Returns
        -------
        None
            Configuration stored for use by ``apply()``.
        """
        self._maxiter = maxiter
        self._maxdelta = maxdelta
        self._area = area
        self._coolexp = coolexp
        self._repulserad = repulserad
        self._cellsize = cellsize
        self._root = root

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Resolve LGL parameters and build the undirected graph structure.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state receiving LGL extras.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with all LGL-specific configuration and graph data
            stored in ``extras``.
        """
        del ctx

        num_nodes = problem.num_nodes
        resolved_maxdelta = float(num_nodes) if self._maxdelta is None else self._maxdelta
        resolved_area = float(num_nodes * num_nodes) if self._area is None else self._area
        resolved_repulserad = (
            resolved_area * float(max(num_nodes, 1))
            if self._repulserad is None
            else self._repulserad
        )
        resolved_cellsize = resolved_area**0.25 if self._cellsize is None else self._cellsize

        adjacency, spring_edges, spring_edge_weights = _build_undirected_graph(
            edge_index=problem.edge_index,
            num_nodes=num_nodes,
            edge_weights=problem.edge_weights,
        )

        rng = random.Random(problem.seed)
        root_node = rng.randrange(num_nodes) if self._root is None else self._root

        frk = math.sqrt(resolved_area / float(max(num_nodes, 1)))

        state.extras["lgl_maxiter"] = self._maxiter
        state.extras["lgl_maxdelta"] = resolved_maxdelta
        state.extras["lgl_area"] = resolved_area
        state.extras["lgl_coolexp"] = self._coolexp
        state.extras["lgl_repulserad"] = resolved_repulserad
        state.extras["lgl_cellsize"] = resolved_cellsize
        state.extras["lgl_root"] = root_node
        state.extras["lgl_frk"] = frk
        state.extras["lgl_adjacency"] = adjacency
        state.extras["lgl_spring_edges"] = spring_edges
        state.extras["lgl_spring_edge_weights"] = spring_edge_weights
        return state


class _LGLShellGrowthSolve(Op):
    """Run the BFS shell growth and per-layer refinement loop.

    This op wraps the full LGL shell growth because the incremental vertex
    placement, edge activation, and per-layer refinement form a tightly
    coupled loop that is not decomposable into separate composable ops.
    Each layer depends on the positions and active set from the previous
    layer.
    """

    name: ClassVar[str] = "lgl_shell_growth_solve"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("pos", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Execute the BFS shell growth with per-layer refinement.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with initialized positions and LGL extras.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with positions updated by the full LGL shell growth
            and refinement loop.
        """
        del ctx

        if state.pos is None:
            raise ValueError("_LGLShellGrowthSolve requires state.pos to be set.")

        num_nodes = problem.num_nodes
        positions = state.pos
        root_node = state.extras["lgl_root"]
        adjacency = state.extras["lgl_adjacency"]
        spring_edges = state.extras["lgl_spring_edges"]
        spring_edge_weights = state.extras["lgl_spring_edge_weights"]
        maxiter = state.extras["lgl_maxiter"]
        maxdelta = state.extras["lgl_maxdelta"]
        coolexp = state.extras["lgl_coolexp"]
        frk = state.extras["lgl_frk"]
        repulserad = state.extras["lgl_repulserad"]
        cellsize = state.extras["lgl_cellsize"]
        area = state.extras["lgl_area"]
        radius = math.sqrt(area / math.pi)

        layers, parents = _bfs_layers(adjacency=adjacency, root=root_node)
        if not layers:
            state.pos = positions
            return state

        placed = torch.zeros(num_nodes, dtype=torch.bool)
        placed[root_node] = True
        active_edges: list[tuple[int, int]] = []
        active_edge_weights: Optional[list[float]] = [] if spring_edge_weights is not None else None
        edge_active = [False] * len(spring_edges)
        incident_edge_indices: list[list[int]] = [[] for _ in range(num_nodes)]
        for edge_idx, (source, target) in enumerate(spring_edges):
            incident_edge_indices[source].append(edge_idx)
            incident_edge_indices[target].append(edge_idx)
        shell_scale = radius / _harmonic_number(max(len(layers) - 1, 1))

        rng = random.Random(problem.seed)
        # Consume the same RNG call as layout_lgl to keep streams aligned.
        if self._root_was_random(state):
            rng.randrange(num_nodes)

        for layer_index in range(len(layers) - 1):
            current_layer = layers[layer_index]
            next_layer = layers[layer_index + 1]
            if not next_layer:
                continue

            active_nodes = torch.nonzero(placed, as_tuple=False).view(-1)
            center_of_mass = (
                positions[active_nodes].mean(dim=0)
                if active_nodes.numel() > 0
                else torch.zeros(2, dtype=torch.float64)
            )
            center_direction = _normalize(center_of_mass)
            next_depth = layer_index + 1
            total_layer_children = len(next_layer)
            layer_child_index = 0

            for parent in current_layer:
                children = [node for node in next_layer if parents[node] == parent]
                if not children:
                    continue

                if parent == root_node:
                    parent_direction = torch.zeros(2, dtype=torch.float64)
                else:
                    parent_of_parent = parents[parent]
                    parent_direction = _normalize(positions[parent] - positions[parent_of_parent])

                anchor = positions[parent] + center_direction + parent_direction
                for child in children:
                    if next_depth == 1:
                        angle = (2.0 * math.pi * layer_child_index) / float(
                            max(total_layer_children, 1)
                        )
                        direction = torch.tensor(
                            [math.cos(angle), math.sin(angle)],
                            dtype=torch.float64,
                        )
                    else:
                        direction = torch.tensor(
                            [rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)],
                            dtype=torch.float64,
                        )
                        direction = _normalize(direction)
                        if float(torch.linalg.norm(direction).item()) <= _MIN_DISTANCE:
                            direction = torch.tensor([1.0, 0.0], dtype=torch.float64)

                    offset = direction * (shell_scale / float(max(next_depth, 1)))
                    positions[child] = anchor + offset
                    placed[child] = True
                    layer_child_index += 1

                    for edge_idx in incident_edge_indices[child]:
                        if edge_active[edge_idx]:
                            continue
                        edge = spring_edges[edge_idx]
                        source, target = edge
                        other = target if source == child else source
                        if not bool(placed[other].item()):
                            continue
                        edge_active[edge_idx] = True
                        active_edges.append(edge)
                        if active_edge_weights is not None:
                            active_edge_weights.append(float(spring_edge_weights[edge_idx]))

            refinement_nodes = torch.nonzero(placed, as_tuple=False).view(-1).tolist()
            _run_refinement(
                positions=positions,
                active_nodes=refinement_nodes,
                active_edges=active_edges,
                active_edge_weights=active_edge_weights,
                maxiter=maxiter,
                maxdelta=maxdelta,
                coolexp=coolexp,
                frk=frk,
                repulserad=repulserad,
                cellsize=cellsize,
            )

        state.pos = positions
        return state

    @staticmethod
    def _root_was_random(state: SolveState) -> bool:
        """Check whether the root was randomly selected.

        Parameters
        ----------
        state : SolveState
            Solve state containing LGL extras.

        Returns
        -------
        bool
            True if the root was randomly selected (no explicit root).
        """
        return state.extras.get("lgl_root_was_random", True)


class _FinalizeLGLPositions(Op):
    """Cast positions to float32 on the output device, matching classic LGL."""

    name: ClassVar[str] = "lgl_finalize_positions"
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
        """Cast and move positions to the output device.

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
            State with final output positions as ``float32`` on the
            classic return device.

        Raises
        ------
        ValueError
            If ``state.pos`` is missing.
        """
        del ctx

        if state.pos is None:
            raise ValueError("_FinalizeLGLPositions requires state.pos to be set.")

        output_device = _layout_device(
            edge_index=problem.edge_index,
            node_sizes=problem.node_sizes,
        )
        state.pos = state.pos.to(dtype=torch.float32, device=output_device)
        return state


def build_lgl_pipeline(
    maxiter: int = 150,
    maxdelta: Optional[float] = None,
    area: Optional[float] = None,
    coolexp: float = 1.5,
    repulserad: Optional[float] = None,
    cellsize: Optional[float] = None,
    root: Optional[int] = None,
) -> Pipeline:
    """Build an LGL pipeline that is bit-identical to classic ``layout_lgl``.

    Parameters
    ----------
    maxiter : int, default=150
        Maximum refinement iterations per BFS layer.
    maxdelta : float, optional
        Maximum initial movement magnitude. Defaults to ``num_nodes``.
    area : float, optional
        Layout area. Defaults to ``num_nodes ** 2``.
    coolexp : float, default=1.5
        Power-law cooling exponent.
    repulserad : float, optional
        Modified FR repulsion cutoff. Defaults to ``area * num_nodes``.
    cellsize : float, optional
        Sparse-grid cell size. Defaults to ``area ** 0.25``.
    root : int, optional
        BFS root. When omitted, a seed-controlled random vertex is chosen.

    Returns
    -------
    Pipeline
        Pipeline that reproduces classic LGL's BFS shell growth,
        per-layer refinement, and final dtype/device cast.

    Raises
    ------
    ValueError
        If ``maxiter`` is negative or ``coolexp`` is not positive.
    """
    if maxiter < 0:
        raise ValueError("maxiter must be non-negative.")
    if coolexp <= 0.0:
        raise ValueError("coolexp must be positive.")

    return Pipeline(
        [
            _PrepareLGLState(
                maxiter=maxiter,
                maxdelta=maxdelta,
                area=area,
                coolexp=coolexp,
                repulserad=repulserad,
                cellsize=cellsize,
                root=root,
            ),
            _InitializeLGLPositions(),
            _LGLShellGrowthSolve(),
            _FinalizeLGLPositions(),
        ],
        name="lgl_pipeline",
    )


def layout_lgl_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: int = 42,
    maxiter: int = 150,
    maxdelta: Optional[float] = None,
    area: Optional[float] = None,
    coolexp: float = 1.5,
    repulserad: Optional[float] = None,
    cellsize: Optional[float] = None,
    root: Optional[int] = None,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the LGL pipeline as a drop-in replacement for classic ``layout_lgl``.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Unused placeholder kept for interface compatibility.
    seed : int, default=42
        Random seed controlling root selection and stochastic shell placement.
    maxiter : int, default=150
        Maximum refinement iterations per BFS layer.
    maxdelta : float, optional
        Maximum initial movement magnitude. Defaults to ``num_nodes``.
    area : float, optional
        Layout area. Defaults to ``num_nodes ** 2``.
    coolexp : float, default=1.5
        Power-law cooling exponent.
    repulserad : float, optional
        Modified FR repulsion cutoff. Defaults to ``area * num_nodes``.
    cellsize : float, optional
        Sparse-grid cell size. Defaults to ``area ** 0.25``.
    root : int, optional
        BFS root. When omitted, a seed-controlled random vertex is chosen.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]`` that scales the spring
        force for each non-self edge occurrence.

    Returns
    -------
    torch.Tensor
        Final position tensor with the same dtype, device, and values as
        classic ``layout_lgl``.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``maxiter``, ``coolexp``, or ``edge_weights``
        are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    _validate_inputs(
        edge_index=edge_index,
        num_nodes=num_nodes,
        maxiter=maxiter,
        coolexp=coolexp,
        edge_weights=edge_weights,
    )

    if num_nodes == 0:
        device = _layout_device(edge_index=edge_index, node_sizes=node_sizes)
        return torch.empty((0, 2), dtype=torch.float32, device=device)

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))

    # Track whether root was explicitly provided so the shell growth op
    # can reproduce the same RNG stream as the classic implementation.
    state.extras["lgl_root_was_random"] = root is None

    final_state = build_lgl_pipeline(
        maxiter=maxiter,
        maxdelta=maxdelta,
        area=area,
        coolexp=coolexp,
        repulserad=repulserad,
        cellsize=cellsize,
        root=root,
    ).apply(problem, state, ctx)

    if final_state.pos is None:
        raise RuntimeError("LGL pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_lgl_pipeline", "layout_lgl_pipeline"]
