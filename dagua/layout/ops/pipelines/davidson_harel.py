"""Davidson-Harel simulated annealing expressed as a composable ops pipeline."""

from __future__ import annotations

from typing import ClassVar, List, Optional, Tuple

import torch

from dagua.layout.classic.davidson_harel import (
    _COOLING_FACTOR,
    _MIN_DISTANCE,
    _energy,
    _initialize_positions,
    _layout_device,
    _layout_extent,
    _unique_edges,
)
from dagua.layout.ops.base import Op, Pipeline, Repeat
from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory


class _InitializeDHPositions(Op):
    """Initialize positions exactly like classic Davidson-Harel."""

    name: ClassVar[str] = "dh_initialize_positions"
    category: ClassVar[OpCategory] = OpCategory.INIT
    writes: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Seed ``state.pos`` from torch-based random initialization.

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
            State with ``state.pos`` populated.
        """
        del ctx

        device = _layout_device(problem.edge_index, problem.node_sizes)
        extent = _layout_extent(problem.num_nodes, problem.node_sizes)
        state.pos = _initialize_positions(problem.num_nodes, extent, device, problem.seed)
        state.extras["dh_extent"] = extent
        state.extras["dh_device"] = device
        return state


class _PrepareDHState(Op):
    """Populate Davidson-Harel-specific cached state."""

    name: ClassVar[str] = "dh_prepare_state"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build unique edges, compute initial energy, and set temperature.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with positions already initialized.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with DH edges, weights, energy, temperature, and RNG
            stored in ``extras``.
        """
        del ctx

        assert state.pos is not None

        edges, unique_edge_weights = _unique_edges(
            problem.edge_index,
            problem.num_nodes,
            edge_weights=problem.edge_weights,
        )
        state.extras["dh_edges"] = edges
        state.extras["dh_unique_edge_weights"] = unique_edge_weights

        if problem.edge_weights is None:
            current_energy = _energy(state.pos, edges, state.extras["dh_extent"])
        else:
            current_energy = _energy(
                state.pos, edges, state.extras["dh_extent"], unique_edge_weights
            )
        state.extras["dh_current_energy"] = current_energy

        initial_temperature = max(0.1 * float(current_energy.item()), _MIN_DISTANCE)
        state.extras["dh_initial_temperature"] = initial_temperature
        state.temperature = initial_temperature

        generator = torch.Generator(device="cpu")
        generator.manual_seed(problem.seed)
        state.extras["dh_generator"] = generator

        return state


class _DHAnnealingRound(Op):
    """Execute one round of Davidson-Harel simulated annealing moves.

    Each round proposes ``num_nodes`` moves using the same Metropolis
    accept/reject logic as the classic implementation.
    """

    name: ClassVar[str] = "dh_annealing_round"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("pos", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("pos", "extras")
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Propose and accept/reject node moves for one SA round.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with positions, edges, energy, and RNG.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with updated positions and energy after one round.
        """
        del ctx

        assert state.pos is not None

        positions = state.pos
        edges: List[Tuple[int, int]] = state.extras["dh_edges"]
        unique_edge_weights: torch.Tensor = state.extras["dh_unique_edge_weights"]
        extent: float = state.extras["dh_extent"]
        current_energy: torch.Tensor = state.extras["dh_current_energy"]
        initial_temperature: float = state.extras["dh_initial_temperature"]
        temperature: float = state.temperature  # type: ignore[assignment]
        generator: torch.Generator = state.extras["dh_generator"]
        device: torch.device = state.extras["dh_device"]
        has_edge_weights = problem.edge_weights is not None

        moves_per_round = problem.num_nodes

        for _ in range(moves_per_round):
            node = int(torch.randint(0, problem.num_nodes, (1,), generator=generator).item())
            move_scale = 0.25 * extent * (temperature / max(initial_temperature, _MIN_DISTANCE))
            delta = ((torch.rand((2,), generator=generator) * 2.0) - 1.0).to(device) * move_scale
            candidate = positions.clone()
            candidate[node] = (candidate[node] + delta).clamp(min=-extent, max=extent)
            if not has_edge_weights:
                candidate_energy = _energy(candidate, edges, extent)
            else:
                candidate_energy = _energy(candidate, edges, extent, unique_edge_weights)
            delta_energy = candidate_energy - current_energy
            if delta_energy <= 0:
                positions = candidate
                current_energy = candidate_energy
                continue

            acceptance = torch.exp(-delta_energy / max(temperature, _MIN_DISTANCE)).clamp(max=1.0)
            threshold = float(torch.rand((1,), generator=generator).item())
            if threshold < float(acceptance.item()):
                positions = candidate
                current_energy = candidate_energy

        state.pos = positions
        state.extras["dh_current_energy"] = current_energy
        return state


class _DHCool(Op):
    """Apply geometric cooling to the DH temperature."""

    name: ClassVar[str] = "dh_cool"
    category: ClassVar[OpCategory] = OpCategory.ANNEAL
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = ()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Multiply temperature by the cooling factor.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with cooled temperature.
        """
        del problem, ctx

        assert state.temperature is not None
        state.temperature = state.temperature * _COOLING_FACTOR
        return state


class _FinalizeDHPositions(Op):
    """Apply classic Davidson-Harel final centering and scaling."""

    name: ClassVar[str] = "dh_finalize_positions"
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
        """Center, normalize, scale, and cast positions like classic DH.

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
            State with final output positions.

        Raises
        ------
        ValueError
            If ``state.pos`` is missing.
        """
        del ctx

        if state.pos is None:
            raise ValueError("_FinalizeDHPositions requires state.pos to be set.")

        device: torch.device = state.extras["dh_device"]
        extent: float = state.extras["dh_extent"]

        centered = state.pos - state.pos.mean(dim=0, keepdim=True)
        span = centered.abs().max().clamp(min=1.0)
        state.pos = (centered * (extent / span)).to(dtype=torch.float32, device=device)
        return state


def build_davidson_harel_pipeline(rounds: int = 100) -> Pipeline:
    """Build a DH pipeline that is bit-identical to classic ``layout_davidson_harel``.

    Parameters
    ----------
    rounds : int, default=100
        Number of annealing rounds.

    Returns
    -------
    Pipeline
        Pipeline that reproduces classic Davidson-Harel initialization,
        energy evaluation, simulated annealing loop, and postprocessing.

    Raises
    ------
    ValueError
        If ``rounds`` is negative.
    """
    if rounds < 0:
        raise ValueError("rounds must be non-negative.")

    return Pipeline(
        [
            FixedSteps(FixedStepsConfig(n=rounds)),
            _InitializeDHPositions(),
            _PrepareDHState(),
            Repeat(
                n=rounds,
                ops=[
                    _DHAnnealingRound(),
                    _DHCool(),
                ],
            ),
            _FinalizeDHPositions(),
        ],
        name="davidson_harel_pipeline",
    )


def layout_davidson_harel_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    rounds: int = 100,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the DH pipeline as a drop-in replacement for classic ``layout_davidson_harel``.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor used for drawing scale and output device.
    rounds : int, default=100
        Number of annealing rounds.
    seed : int, default=42
        Random seed for initialization and SA moves.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Final position tensor with the same dtype, device, and values as
        classic ``layout_davidson_harel``.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``rounds``, or ``edge_weights`` are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if rounds < 0:
        raise ValueError("rounds must be non-negative.")
    if edge_weights is not None:
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
            )

    # Handle trivial cases identically to classic implementation
    device = _layout_device(edge_index, node_sizes)
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
    final_state = build_davidson_harel_pipeline(rounds=rounds).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("Davidson-Harel pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_davidson_harel_pipeline", "layout_davidson_harel_pipeline"]
