"""Distributed Recursive Layout (DrL) expressed as a composable ops pipeline.

DrL is a 6-phase density-grid layout algorithm with sequential node updates,
phase-dependent attraction exponents, greedy local search, and late-stage edge
cutting.  The per-node energy evaluation, density grid bookkeeping, and
stochastic perturbation form a tightly coupled loop that is not decomposable
into independent force/apply ops, so the core solver is wrapped in a single op
that delegates to classic helpers.

This pipeline delegates to the classic helpers so that output is bit-identical
to ``layout_drl``.
"""

from __future__ import annotations

import random
from typing import ClassVar, Mapping, Optional, Tuple, Union

import torch

from dagua.layout.classic.drl import (
    _CUT_BASE,
    _GRID_RADIUS,
    _GRID_SIZE,
    _VIEW_SIZE,
    _build_undirected_adjacency,
    _compute_energy,
    _DensityGrid,
    _DrlParameters,
    _initialize_positions,
    _layout_device,
    _maybe_cut_long_edge,
    _OptionObject,
    _resolve_drl_parameters,
    _validate_inputs,
    _weighted_centroid,
)
from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory


class _InitializeDRLPositions(Op):
    """Initialize positions exactly like classic DrL."""

    name: ClassVar[str] = "drl_initialize_positions"
    category: ClassVar[OpCategory] = OpCategory.INIT
    writes: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Seed ``state.pos`` from Python's ``random.Random`` unit-square initialization.

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


class _PrepareDRLState(Op):
    """Populate DrL-specific cached state required by the solver op."""

    name: ClassVar[str] = "drl_prepare_state"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    _options: Union[str, Mapping[str, object], _OptionObject]

    def __init__(
        self,
        options: Union[str, Mapping[str, object], _OptionObject] = "default",
    ) -> None:
        """Store DrL configuration options.

        Parameters
        ----------
        options : str or Mapping[str, object] or _OptionObject, default="default"
            igraph DrL preset name or per-phase override container.

        Returns
        -------
        None
            Configuration stored for use by ``apply()``.
        """
        self._options = options

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Resolve DrL parameters and build the undirected adjacency map.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state receiving DrL extras.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with DrL parameters and adjacency stored in ``extras``.
        """
        del ctx

        params = _resolve_drl_parameters(options=self._options)
        adjacency = _build_undirected_adjacency(
            edge_index=problem.edge_index,
            num_nodes=problem.num_nodes,
            edge_weights=problem.edge_weights,
        )

        state.extras["drl_params"] = params
        state.extras["drl_adjacency"] = adjacency
        return state


class _DRLPhaseSolve(Op):
    """Run the full 6-phase DrL solver loop.

    This op wraps the entire DrL solve because the sequential node updates,
    density grid bookkeeping, stochastic perturbation, and inter-phase
    parameter annealing form a tightly coupled loop that is not decomposable
    into separate composable ops.  Each node update depends on the density
    grid state left by the previous node update within the same iteration.
    """

    name: ClassVar[str] = "drl_phase_solve"
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
        """Execute all 6 DrL phases with density grid and greedy local search.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with initialized positions and DrL extras.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with positions updated by the full DrL solver.

        Raises
        ------
        ValueError
            If ``state.pos`` is missing.
        """
        del ctx

        if state.pos is None:
            raise ValueError("_DRLPhaseSolve requires state.pos to be set.")

        params: _DrlParameters = state.extras["drl_params"]
        adjacency: list[dict[int, float]] = state.extras["drl_adjacency"]
        num_nodes = problem.num_nodes
        positions = state.pos

        density_grid = _DensityGrid(
            grid_size=_GRID_SIZE,
            view_size=_VIEW_SIZE,
            radius=_GRID_RADIUS,
        )
        for node in range(num_nodes):
            density_grid.add_node(node=node, position=positions[node])

        rng = random.Random(problem.seed)
        cut_end = _CUT_BASE * (1.0 - params.edge_cut)
        cut_off_length = 4.0 * cut_end
        cut_rate = 0.0 if cut_end <= 0.0 else (3.0 * cut_end) / 400.0
        min_edges = 20.0

        phase_specs = [
            ("init", params.init),
            ("liquid", params.liquid),
            ("expansion", params.expansion),
            ("cooldown", params.cooldown),
            ("crunch", params.crunch),
            ("simmer", params.simmer),
        ]

        for phase_name, phase in phase_specs:
            temperature = phase.temperature
            attraction = phase.attraction
            damping_mult = phase.damping_mult
            fine_density = phase_name == "simmer"

            for _ in range(phase.iterations):
                if phase_name == "expansion":
                    if attraction > 1.0:
                        attraction = max(1.0, attraction - 0.05)
                    if min_edges > 12.0:
                        min_edges = max(12.0, min_edges - 0.05)
                    if cut_end > 0.0:
                        cut_off_length = max(cut_end, cut_off_length - cut_rate)
                    if damping_mult > 0.1:
                        damping_mult = max(0.1, damping_mult - 0.005)
                elif phase_name == "cooldown":
                    if temperature > 50.0:
                        temperature = max(50.0, temperature - 10.0)
                    if cut_end > 0.0:
                        cut_off_length = max(cut_end, cut_off_length - (2.0 * cut_rate))
                    if min_edges > 1.0:
                        min_edges = max(1.0, min_edges - 0.2)
                elif phase_name == "simmer" and temperature > 50.0:
                    temperature = max(50.0, temperature - 2.0)

                for node in range(num_nodes):
                    density_grid.remove_node(node=node)
                    current = positions[node].clone()
                    current_energy = _compute_energy(
                        node=node,
                        candidate=current,
                        positions=positions,
                        adjacency=adjacency,
                        attraction=attraction,
                        phase_name=phase_name,
                        density_grid=density_grid,
                        fine_density=fine_density,
                    )

                    centroid = _weighted_centroid(
                        node=node,
                        positions=positions,
                        adjacency=adjacency,
                    )
                    analytic = (positions[node] * (1.0 - damping_mult)) + (centroid * damping_mult)

                    if phase_name in {"expansion", "cooldown"} and cut_end > 0.0:
                        _maybe_cut_long_edge(
                            node=node,
                            positions=positions,
                            adjacency=adjacency,
                            min_edges=min_edges,
                            cut_off_length=cut_off_length,
                        )

                    jump_length = 0.01 * temperature
                    random_offset = torch.tensor(
                        [
                            rng.uniform(-0.5, 0.5) * jump_length,
                            rng.uniform(-0.5, 0.5) * jump_length,
                        ],
                        dtype=torch.float64,
                    )
                    perturbed = analytic + random_offset

                    analytic_energy = _compute_energy(
                        node=node,
                        candidate=analytic,
                        positions=positions,
                        adjacency=adjacency,
                        attraction=attraction,
                        phase_name=phase_name,
                        density_grid=density_grid,
                        fine_density=fine_density,
                    )
                    perturbed_energy = _compute_energy(
                        node=node,
                        candidate=perturbed,
                        positions=positions,
                        adjacency=adjacency,
                        attraction=attraction,
                        phase_name=phase_name,
                        density_grid=density_grid,
                        fine_density=fine_density,
                    )

                    best_position = current
                    best_energy = current_energy
                    if analytic_energy < best_energy:
                        best_position = analytic
                        best_energy = analytic_energy
                    if perturbed_energy < best_energy:
                        best_position = perturbed

                    positions[node] = best_position
                    density_grid.add_node(node=node, position=positions[node])

        state.pos = positions
        return state


class _FinalizeDRLPositions(Op):
    """Cast positions to float32 on the output device, matching classic DrL."""

    name: ClassVar[str] = "drl_finalize_positions"
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
            raise ValueError("_FinalizeDRLPositions requires state.pos to be set.")

        output_device = _layout_device(
            edge_index=problem.edge_index,
            node_sizes=problem.node_sizes,
        )
        state.pos = state.pos.to(dtype=torch.float32, device=output_device)
        return state


def build_drl_pipeline(
    options: Union[str, Mapping[str, object], _OptionObject] = "default",
) -> Pipeline:
    """Build a DrL pipeline that is bit-identical to classic ``layout_drl``.

    Parameters
    ----------
    options : str or Mapping[str, object] or _OptionObject, default="default"
        igraph DrL preset name or a mapping/object of per-phase overrides.

    Returns
    -------
    Pipeline
        Pipeline that reproduces classic DrL's 6-phase density-grid solver
        with sequential node updates, greedy local search, and edge cutting.
    """
    return Pipeline(
        [
            _PrepareDRLState(options=options),
            _InitializeDRLPositions(),
            _DRLPhaseSolve(),
            _FinalizeDRLPositions(),
        ],
        name="drl_pipeline",
    )


def layout_drl_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    options: Union[str, Mapping[str, object], _OptionObject] = "default",
) -> torch.Tensor:
    """Run the DrL pipeline as a drop-in replacement for classic ``layout_drl``.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Unused placeholder kept for interface compatibility.
    seed : int, default=42
        Random seed for the initial placement and stochastic node updates.
    edge_weights : torch.Tensor, optional
        Optional positive edge weights with shape ``[E]``.
    options : str or Mapping[str, object] or _OptionObject, default="default"
        igraph DrL preset name or a mapping/object of per-phase overrides using
        underscore-separated field names such as ``liquid_iterations``.

    Returns
    -------
    torch.Tensor
        Final position tensor with the same dtype, device, and values as
        classic ``layout_drl``.

    Raises
    ------
    ValueError
        If ``num_nodes`` or ``edge_weights`` are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    _validate_inputs(
        edge_index=edge_index,
        num_nodes=num_nodes,
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

    final_state = build_drl_pipeline(options=options).apply(problem, state, ctx)

    if final_state.pos is None:
        raise RuntimeError("DrL pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_drl_pipeline", "layout_drl_pipeline"]
