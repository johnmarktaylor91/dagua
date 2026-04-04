"""Distributed Recursive Layout (DrL) expressed as a composable ops pipeline.

DrL is a 6-phase density-grid layout algorithm with sequential node updates,
phase-dependent attraction exponents, greedy local search, and late-stage edge
cutting. The per-node energy evaluation, density grid bookkeeping, and
stochastic perturbation form a tightly coupled loop that is not decomposable
into independent force/apply ops, so the core solver is wrapped in a single op.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import ClassVar, Mapping, Optional, Protocol, Tuple, Union, cast

import torch

from dagua.layout.ops.base import Op, Pipeline  # noqa: E402
from dagua.layout.ops.graph_utils import (
    build_undirected_adjacency as _shared_build_undirected_adjacency,
)
from dagua.layout.ops.graph_utils import (
    layout_device as _layout_device,
)
from dagua.layout.ops.state import (  # noqa: E402
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)
from dagua.layout.ops.taxonomy import OpCategory  # noqa: E402

_GRID_SIZE = 1000
_VIEW_SIZE = 4000.0
_GRID_RADIUS = 10
_MIN_DISTANCE = 1.0e-12
_FINE_REPULSION_SCALE = 1.0e-4
_CUT_BASE = 40_000.0


class _OptionObject(Protocol):
    """Structural protocol for attribute-based DrL option lookup."""

    def __getattr__(self, name: str) -> object:
        """Return an option value by attribute name."""


@dataclass(frozen=True)
class _PhaseParameters:
    """Parameter bundle for one DrL phase."""

    iterations: int
    temperature: float
    attraction: float
    damping_mult: float


@dataclass(frozen=True)
class _DrlParameters:
    """Resolved DrL parameter set for all phases."""

    edge_cut: float
    init: _PhaseParameters
    liquid: _PhaseParameters
    expansion: _PhaseParameters
    cooldown: _PhaseParameters
    crunch: _PhaseParameters
    simmer: _PhaseParameters


_DRL_PRESETS: dict[str, _DrlParameters] = {
    "default": _DrlParameters(
        edge_cut=32.0 / 40.0,
        init=_PhaseParameters(0, 2000.0, 10.0, 1.0),
        liquid=_PhaseParameters(200, 2000.0, 10.0, 1.0),
        expansion=_PhaseParameters(200, 2000.0, 2.0, 1.0),
        cooldown=_PhaseParameters(200, 2000.0, 1.0, 0.1),
        crunch=_PhaseParameters(50, 250.0, 1.0, 0.25),
        simmer=_PhaseParameters(100, 250.0, 0.5, 0.0),
    ),
    "coarsen": _DrlParameters(
        edge_cut=32.0 / 40.0,
        init=_PhaseParameters(0, 2000.0, 10.0, 1.0),
        liquid=_PhaseParameters(200, 2000.0, 2.0, 1.0),
        expansion=_PhaseParameters(200, 2000.0, 10.0, 1.0),
        cooldown=_PhaseParameters(200, 2000.0, 1.0, 0.1),
        crunch=_PhaseParameters(50, 250.0, 1.0, 0.25),
        simmer=_PhaseParameters(100, 250.0, 0.5, 0.0),
    ),
    "coarsest": _DrlParameters(
        edge_cut=32.0 / 40.0,
        init=_PhaseParameters(0, 2000.0, 10.0, 1.0),
        liquid=_PhaseParameters(200, 2000.0, 2.0, 1.0),
        expansion=_PhaseParameters(200, 2000.0, 10.0, 1.0),
        cooldown=_PhaseParameters(200, 2000.0, 1.0, 0.1),
        crunch=_PhaseParameters(200, 250.0, 1.0, 0.25),
        simmer=_PhaseParameters(100, 250.0, 0.5, 0.0),
    ),
    "refine": _DrlParameters(
        edge_cut=32.0 / 40.0,
        init=_PhaseParameters(0, 50.0, 0.5, 1.0),
        liquid=_PhaseParameters(0, 2000.0, 2.0, 1.0),
        expansion=_PhaseParameters(50, 500.0, 0.1, 0.25),
        cooldown=_PhaseParameters(50, 250.0, 1.0, 0.1),
        crunch=_PhaseParameters(50, 250.0, 1.0, 0.25),
        simmer=_PhaseParameters(0, 250.0, 0.5, 0.0),
    ),
    "final": _DrlParameters(
        edge_cut=32.0 / 40.0,
        init=_PhaseParameters(0, 50.0, 0.5, 0.0),
        liquid=_PhaseParameters(0, 2000.0, 2.0, 1.0),
        expansion=_PhaseParameters(50, 2000.0, 2.0, 1.0),
        cooldown=_PhaseParameters(50, 200.0, 1.0, 0.1),
        crunch=_PhaseParameters(50, 250.0, 1.0, 0.25),
        simmer=_PhaseParameters(25, 250.0, 0.5, 0.0),
    ),
}


def _validate_inputs(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor],
) -> None:
    """Validate the public DrL arguments.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.

    Returns
    -------
    None
        Raises ``ValueError`` when an input is invalid.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")
    if edge_weights is not None and edge_weights.shape[0] != edge_index.shape[1]:
        raise ValueError(
            f"edge_weights length {edge_weights.shape[0]} does not match "
            f"edge count {edge_index.shape[1]}"
        )

    if edge_index.numel() == 0:
        return

    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    min_index = int(edge_index_cpu.min().item())
    max_index = int(edge_index_cpu.max().item())
    if min_index < 0:
        raise ValueError("edge_index cannot contain negative node indices.")
    if max_index >= num_nodes:
        raise ValueError("edge_index contains node indices outside [0, num_nodes).")
    if edge_weights is not None and bool(torch.any(edge_weights <= 0.0).item()):
        raise ValueError("edge_weights must be strictly positive.")


def _lookup_option(
    options: Union[str, Mapping[str, object], _OptionObject],
    name: str,
) -> Optional[object]:
    """Read one DrL option from a mapping or an object.

    Parameters
    ----------
    options : str or Mapping[str, object] or _OptionObject
        Raw ``options`` argument passed to ``layout_drl``.
    name : str
        Option name using underscore separators.

    Returns
    -------
    object or None
        The provided override, or ``None`` if the option is absent.
    """
    if isinstance(options, str):
        return None
    if isinstance(options, Mapping):
        return options.get(name)
    return getattr(options, name, None)


def _resolve_drl_parameters(
    options: Union[str, Mapping[str, object], _OptionObject],
) -> _DrlParameters:
    """Resolve DrL options against the igraph preset table.

    Parameters
    ----------
    options : str or Mapping[str, object] or _OptionObject
        DrL option preset name or override container.

    Returns
    -------
    _DrlParameters
        Fully populated parameter bundle.
    """
    if isinstance(options, str):
        try:
            return _DRL_PRESETS[options]
        except KeyError as exc:
            available = ", ".join(sorted(_DRL_PRESETS))
            raise ValueError(
                f"unknown DrL preset {options!r}; expected one of {available}."
            ) from exc

    default = _DRL_PRESETS["default"]
    values: dict[str, float] = {
        "edge_cut": default.edge_cut,
        "init_iterations": float(default.init.iterations),
        "init_temperature": default.init.temperature,
        "init_attraction": default.init.attraction,
        "init_damping_mult": default.init.damping_mult,
        "liquid_iterations": float(default.liquid.iterations),
        "liquid_temperature": default.liquid.temperature,
        "liquid_attraction": default.liquid.attraction,
        "liquid_damping_mult": default.liquid.damping_mult,
        "expansion_iterations": float(default.expansion.iterations),
        "expansion_temperature": default.expansion.temperature,
        "expansion_attraction": default.expansion.attraction,
        "expansion_damping_mult": default.expansion.damping_mult,
        "cooldown_iterations": float(default.cooldown.iterations),
        "cooldown_temperature": default.cooldown.temperature,
        "cooldown_attraction": default.cooldown.attraction,
        "cooldown_damping_mult": default.cooldown.damping_mult,
        "crunch_iterations": float(default.crunch.iterations),
        "crunch_temperature": default.crunch.temperature,
        "crunch_attraction": default.crunch.attraction,
        "crunch_damping_mult": default.crunch.damping_mult,
        "simmer_iterations": float(default.simmer.iterations),
        "simmer_temperature": default.simmer.temperature,
        "simmer_attraction": default.simmer.attraction,
        "simmer_damping_mult": default.simmer.damping_mult,
    }
    for key in tuple(values):
        override = _lookup_option(options=options, name=key)
        if override is not None:
            values[key] = float(cast(float, override))

    return _DrlParameters(
        edge_cut=values["edge_cut"],
        init=_PhaseParameters(
            int(values["init_iterations"]),
            values["init_temperature"],
            values["init_attraction"],
            values["init_damping_mult"],
        ),
        liquid=_PhaseParameters(
            int(values["liquid_iterations"]),
            values["liquid_temperature"],
            values["liquid_attraction"],
            values["liquid_damping_mult"],
        ),
        expansion=_PhaseParameters(
            int(values["expansion_iterations"]),
            values["expansion_temperature"],
            values["expansion_attraction"],
            values["expansion_damping_mult"],
        ),
        cooldown=_PhaseParameters(
            int(values["cooldown_iterations"]),
            values["cooldown_temperature"],
            values["cooldown_attraction"],
            values["cooldown_damping_mult"],
        ),
        crunch=_PhaseParameters(
            int(values["crunch_iterations"]),
            values["crunch_temperature"],
            values["crunch_attraction"],
            values["crunch_damping_mult"],
        ),
        simmer=_PhaseParameters(
            int(values["simmer_iterations"]),
            values["simmer_temperature"],
            values["simmer_attraction"],
            values["simmer_damping_mult"],
        ),
    )


def _build_undirected_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor],
) -> list[dict[int, float]]:
    """Build a symmetric weighted adjacency map.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.

    Returns
    -------
    list[dict[int, float]]
        One weighted neighbor dictionary per node.
    """
    adjacency = _shared_build_undirected_adjacency(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weights=edge_weights,
    )
    return [dict(neighbors) for neighbors in adjacency]


def _initialize_positions(num_nodes: int, seed: int) -> torch.Tensor:
    """Create the default DrL starting positions.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    seed : int
        Random seed for the initial layout.

    Returns
    -------
    torch.Tensor
        Initial positions with shape ``[N, 2]`` and dtype ``float64``.
    """
    rng = random.Random(seed)
    data = [[rng.random(), rng.random()] for _ in range(num_nodes)]
    return torch.tensor(data, dtype=torch.float64)


class _DensityGrid:
    """Density proxy used by the DrL energy function."""

    def __init__(self, grid_size: int, view_size: float, radius: int) -> None:
        """Initialize the density grid and its tent kernel.

        Parameters
        ----------
        grid_size : int
            Number of cells per axis.
        view_size : float
            Width of the square viewing window.
        radius : int
            Tent-kernel radius measured in cells.
        """
        self.grid_size = grid_size
        self.view_size = view_size
        self.radius = radius
        self.cell_width = view_size / float(grid_size)
        self.origin = -0.5 * view_size
        self.density = torch.zeros((grid_size, grid_size), dtype=torch.float64)
        self.node_cells: dict[int, tuple[int, int]] = {}
        self.buckets: dict[tuple[int, int], set[int]] = {}

        axis = torch.arange(-radius, radius + 1, dtype=torch.float64)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        distance = torch.sqrt(xx.square() + yy.square())
        self.kernel = torch.clamp(1.0 - (distance / float(radius)), min=0.0)

    def _cell_index(self, position: torch.Tensor) -> tuple[int, int]:
        """Convert a coordinate to a clamped integer grid cell.

        Parameters
        ----------
        position : torch.Tensor
            Position tensor with shape ``[2]``.

        Returns
        -------
        tuple[int, int]
            ``(cell_x, cell_y)`` indices in ``[0, grid_size)``.
        """
        x_value = float(position[0].item())
        y_value = float(position[1].item())
        cell_x = int(math.floor((x_value - self.origin) / self.cell_width))
        cell_y = int(math.floor((y_value - self.origin) / self.cell_width))
        return (
            max(0, min(self.grid_size - 1, cell_x)),
            max(0, min(self.grid_size - 1, cell_y)),
        )

    def _apply_kernel(self, cell_x: int, cell_y: int, sign: float) -> None:
        """Add or subtract one tent kernel at the given cell location.

        Parameters
        ----------
        cell_x : int
            Grid x-index.
        cell_y : int
            Grid y-index.
        sign : float
            ``+1`` to add density, ``-1`` to remove it.
        """
        x_start = max(0, cell_x - self.radius)
        x_end = min(self.grid_size, cell_x + self.radius + 1)
        y_start = max(0, cell_y - self.radius)
        y_end = min(self.grid_size, cell_y + self.radius + 1)

        kernel_x_start = x_start - (cell_x - self.radius)
        kernel_x_end = kernel_x_start + (x_end - x_start)
        kernel_y_start = y_start - (cell_y - self.radius)
        kernel_y_end = kernel_y_start + (y_end - y_start)

        self.density[y_start:y_end, x_start:x_end] += (
            sign * self.kernel[kernel_y_start:kernel_y_end, kernel_x_start:kernel_x_end]
        )

    def add_node(self, node: int, position: torch.Tensor) -> None:
        """Insert a node into the coarse grid and fine buckets.

        Parameters
        ----------
        node : int
            Node index.
        position : torch.Tensor
            Position tensor with shape ``[2]``.
        """
        cell = self._cell_index(position)
        self.node_cells[node] = cell
        self._apply_kernel(cell[0], cell[1], sign=1.0)
        self.buckets.setdefault(cell, set()).add(node)

    def remove_node(self, node: int) -> None:
        """Remove a node from the coarse grid and fine buckets.

        Parameters
        ----------
        node : int
            Node index to remove.
        """
        cell = self.node_cells.pop(node, None)
        if cell is None:
            return
        self._apply_kernel(cell[0], cell[1], sign=-1.0)
        bucket = self.buckets.get(cell)
        if bucket is None:
            return
        bucket.discard(node)
        if not bucket:
            del self.buckets[cell]

    def coarse_density(self, position: torch.Tensor) -> float:
        """Return the coarse density penalty at one position.

        Parameters
        ----------
        position : torch.Tensor
            Candidate position with shape ``[2]``.

        Returns
        -------
        float
            Squared coarse density value.
        """
        cell_x, cell_y = self._cell_index(position)
        value = float(self.density[cell_y, cell_x].item())
        return value * value

    def fine_density(self, node: int, position: torch.Tensor, positions: torch.Tensor) -> float:
        """Return the exact simmer-stage local repulsion penalty.

        Parameters
        ----------
        node : int
            Node being evaluated.
        position : torch.Tensor
            Candidate position with shape ``[2]``.
        positions : torch.Tensor
            Current position matrix with shape ``[N, 2]``.

        Returns
        -------
        float
            Exact local repulsion energy.
        """
        cell_x, cell_y = self._cell_index(position)
        density = 0.0
        for offset_y in (-1, 0, 1):
            for offset_x in (-1, 0, 1):
                neighbor_cell = (cell_x + offset_x, cell_y + offset_y)
                bucket = self.buckets.get(neighbor_cell)
                if not bucket:
                    continue
                for other in bucket:
                    if other == node:
                        continue
                    delta = position - positions[other]
                    distance_sq = float(delta.dot(delta).item()) + _MIN_DISTANCE
                    density += _FINE_REPULSION_SCALE / distance_sq
        return density


def _stage_power(phase_name: str) -> int:
    """Return the distance-squared exponent for one DrL phase.

    Parameters
    ----------
    phase_name : str
        One of the DrL phase names.

    Returns
    -------
    int
        Exponent applied to squared distances.
    """
    if phase_name in {"init", "liquid"}:
        return 4
    if phase_name == "expansion":
        return 2
    return 1


def _compute_energy(
    node: int,
    candidate: torch.Tensor,
    positions: torch.Tensor,
    adjacency: list[dict[int, float]],
    attraction: float,
    phase_name: str,
    density_grid: _DensityGrid,
    fine_density: bool,
) -> float:
    """Evaluate the DrL objective for one node at one candidate position.

    Parameters
    ----------
    node : int
        Node index being updated.
    candidate : torch.Tensor
        Candidate position with shape ``[2]``.
    positions : torch.Tensor
        Current position matrix with shape ``[N, 2]``.
    adjacency : list[dict[int, float]]
        Symmetric weighted adjacency map.
    attraction : float
        Attraction coefficient for the current phase.
    phase_name : str
        Name of the current phase.
    density_grid : _DensityGrid
        Coarse and fine density proxy.
    fine_density : bool
        Whether to use simmer-stage exact local repulsion.

    Returns
    -------
    float
        Scalar energy value.
    """
    energy = 0.0
    attraction_factor = float(attraction**4) * 0.02
    power = _stage_power(phase_name=phase_name)

    for neighbor, weight in adjacency[node].items():
        delta = candidate - positions[neighbor]
        distance_sq = float(delta.dot(delta).item())
        if distance_sq <= 0.0:
            continue
        energy += weight * attraction_factor * (distance_sq**power)

    if fine_density:
        energy += density_grid.fine_density(node=node, position=candidate, positions=positions)
    else:
        energy += density_grid.coarse_density(position=candidate)
    return energy


def _weighted_centroid(
    node: int,
    positions: torch.Tensor,
    adjacency: list[dict[int, float]],
) -> torch.Tensor:
    """Compute the weighted neighbor centroid for one node.

    Parameters
    ----------
    node : int
        Node index being updated.
    positions : torch.Tensor
        Position matrix with shape ``[N, 2]``.
    adjacency : list[dict[int, float]]
        Symmetric weighted adjacency map.

    Returns
    -------
    torch.Tensor
        Weighted centroid with shape ``[2]``.
    """
    neighbors = adjacency[node]
    if not neighbors:
        return positions[node].clone()

    total_weight = sum(neighbors.values())
    if total_weight <= 0.0:
        return positions[node].clone()

    centroid = torch.zeros(2, dtype=torch.float64)
    for neighbor, weight in neighbors.items():
        centroid += positions[neighbor] * weight
    return centroid / total_weight


def _maybe_cut_long_edge(
    node: int,
    positions: torch.Tensor,
    adjacency: list[dict[int, float]],
    min_edges: float,
    cut_off_length: float,
) -> None:
    """Prune one high-stress edge during the cutting phases.

    Parameters
    ----------
    node : int
        Node index being considered.
    positions : torch.Tensor
        Position matrix with shape ``[N, 2]``.
    adjacency : list[dict[int, float]]
        Symmetric weighted adjacency map, mutated in place when an edge is cut.
    min_edges : float
        Minimum degree threshold required before cutting is attempted.
    cut_off_length : float
        Current stress threshold for removing one edge.
    """
    neighbors = adjacency[node]
    if float(len(neighbors)) < min_edges or not neighbors:
        return

    centroid = _weighted_centroid(node=node, positions=positions, adjacency=adjacency)
    worst_neighbor = -1
    worst_score = -1.0

    for neighbor in neighbors:
        degree_factor = math.sqrt(float(max(len(adjacency[neighbor]), 1)))
        delta = positions[neighbor] - centroid
        score = float(delta.dot(delta).item()) * degree_factor
        if score > worst_score:
            worst_score = score
            worst_neighbor = neighbor

    if worst_neighbor >= 0 and worst_score > cut_off_length:
        adjacency[node].pop(worst_neighbor, None)
        adjacency[worst_neighbor].pop(node, None)


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
