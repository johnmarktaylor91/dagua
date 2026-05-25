"""Registered operations implementing the classic DrL algorithm.

The implementation keeps DrL's procedural state updates and local search
behavior inside composable ops so fidelity fixes can target the igraph reference
without changing unrelated layout machinery.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import ClassVar, Mapping, Optional, Protocol, Tuple, Union, cast

import torch

from dagua.layout.ops._igraph_rng import IgraphPCG32, make_igraph_default_rng
from dagua.layout.ops.base import Op
from dagua.layout.ops.graph_utils import layout_device
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op


@dataclass(frozen=True)
class DRLDensityGridConfig:
    """Configuration for DrL's coarse density proxy.

    Parameters
    ----------
    grid_size : int, default=1000
        Number of cells per axis in the coarse density grid.
    view_size : float, default=4000.0
        Side length of the square density view.
    radius : int, default=10
        Tent-kernel radius measured in grid cells.
    """

    grid_size: int = 1000
    view_size: float = 4000.0
    radius: int = 10


@dataclass(frozen=True)
class DRLEnergyConfig:
    """Configuration for DrL node-energy evaluation.

    Parameters
    ----------
    min_distance : float, default=1e-12
        Small additive guard used by fine-density repulsion.
    fine_repulsion_scale : float, default=1e-4
        Scale factor for exact local repulsion during the simmer phase.
    attraction_factor_scale : float, default=0.02
        Multiplier applied after raising the attraction coefficient to the
        classic fourth power.
    jump_temperature_scale : float, default=0.01
        Fraction of the current temperature used for random candidate jumps.
    """

    min_distance: float = 1.0e-12
    fine_repulsion_scale: float = 1.0e-4
    attraction_factor_scale: float = 0.02
    jump_temperature_scale: float = 0.01


@dataclass(frozen=True)
class DRLPhaseDynamicsConfig:
    """Configuration for intra-phase DrL schedule updates.

    Parameters
    ----------
    expansion_attraction_floor : float, default=1.0
        Minimum attraction allowed during the expansion phase.
    expansion_attraction_delta : float, default=0.05
        Per-iteration attraction decrement during expansion.
    expansion_min_edges_floor : float, default=12.0
        Lower bound for the expansion-phase edge-cut threshold.
    expansion_min_edges_delta : float, default=0.05
        Per-iteration decrement for the expansion edge-cut threshold.
    expansion_damping_floor : float, default=0.1
        Lower bound for the expansion damping multiplier.
    expansion_damping_delta : float, default=0.005
        Per-iteration decrement for the expansion damping multiplier.
    cooldown_temperature_floor : float, default=50.0
        Minimum cooldown temperature.
    cooldown_temperature_delta : float, default=10.0
        Per-iteration temperature decrement during cooldown.
    cooldown_cut_rate_multiplier : float, default=2.0
        Extra multiplier applied to cut-rate cooling during cooldown.
    cooldown_min_edges_floor : float, default=1.0
        Lower bound for the cooldown edge-cut threshold.
    cooldown_min_edges_delta : float, default=0.2
        Per-iteration decrement for the cooldown edge-cut threshold.
    simmer_temperature_floor : float, default=50.0
        Minimum simmer temperature.
    simmer_temperature_delta : float, default=2.0
        Per-iteration temperature decrement during simmer.
    """

    expansion_attraction_floor: float = 1.0
    expansion_attraction_delta: float = 0.05
    expansion_min_edges_floor: float = 12.0
    expansion_min_edges_delta: float = 0.05
    expansion_damping_floor: float = 0.1
    expansion_damping_delta: float = 0.005
    cooldown_temperature_floor: float = 50.0
    cooldown_temperature_delta: float = 10.0
    cooldown_cut_rate_multiplier: float = 2.0
    cooldown_min_edges_floor: float = 1.0
    cooldown_min_edges_delta: float = 0.2
    simmer_temperature_floor: float = 50.0
    simmer_temperature_delta: float = 2.0


class OptionObject(Protocol):
    """Attribute-style DrL option container.

    Parameters
    ----------
    name : str
        Arbitrary option key using ``underscore`` separators.
    """

    def __getattr__(self, name: str) -> object:
        """Return an option value by attribute name.

        Parameters
        ----------
        name : str
            Attribute name to resolve.

        Returns
        -------
        object
            Option value.
        """


DrLOptions = Union[str, Mapping[str, object], OptionObject]
RandomLike = Union[random.Random, IgraphPCG32]


@dataclass(frozen=True)
class DRLPrepareStateConfig:
    """Configuration for :class:`DRLPrepareState`.

    Parameters
    ----------
    options : str or Mapping[str, object] or OptionObject, default="default"
        Preset name or override provider.
    """

    options: DrLOptions = "default"


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
        init=_PhaseParameters(0, 50.0, 0.5, 0.0),
        liquid=_PhaseParameters(0, 2000.0, 2.0, 1.0),
        expansion=_PhaseParameters(50, 500.0, 0.1, 0.25),
        cooldown=_PhaseParameters(50, 200.0, 1.0, 0.1),
        crunch=_PhaseParameters(50, 250.0, 1.0, 0.25),
        simmer=_PhaseParameters(0, 250.0, 0.5, 0.0),
    ),
    "final": _DrlParameters(
        edge_cut=32.0 / 40.0,
        init=_PhaseParameters(0, 50.0, 0.5, 0.0),
        liquid=_PhaseParameters(0, 2000.0, 2.0, 1.0),
        expansion=_PhaseParameters(50, 50.0, 0.1, 0.25),
        cooldown=_PhaseParameters(50, 200.0, 1.0, 0.1),
        crunch=_PhaseParameters(50, 250.0, 1.0, 0.25),
        simmer=_PhaseParameters(25, 250.0, 0.5, 0.0),
    ),
}


def _lookup_option(options: DrLOptions, name: str) -> Optional[object]:
    """Read one DrL option from a mapping or attribute object.

    Parameters
    ----------
    options : str or Mapping[str, object] or OptionObject
        Raw options passed to ``layout_drl``.
    name : str
        Option key.

    Returns
    -------
    object or None
        Returned option value if present.
    """
    if isinstance(options, str):
        return None
    if isinstance(options, Mapping):
        return options.get(name)
    return getattr(options, name, None)


def _resolve_drl_parameters(options: DrLOptions) -> _DrlParameters:
    """Resolve DRL options against preset defaults.

    Parameters
    ----------
    options : str or Mapping[str, object] or OptionObject
        Preset name or override container.

    Returns
    -------
    _DrlParameters
        Resolved and type-coerced parameter bundle.

    Raises
    ------
    ValueError
        If a preset name is unknown.
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
    """Build symmetric weighted adjacency as dictionaries.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    edge_weights : torch.Tensor, optional
        Optional weight tensor with shape ``[E]``.

    Returns
    -------
    list[dict[int, float]]
        One weighted neighbor dictionary per node.
    """
    adjacency: list[dict[int, float]] = [dict() for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return adjacency

    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    if edge_weights is None:
        weights_cpu = torch.ones(edge_index.shape[1], dtype=torch.float64)
    else:
        weights_cpu = edge_weights.to(device="cpu", dtype=torch.float64)

    sources = edge_index_cpu[0].tolist()
    targets = edge_index_cpu[1].tolist()
    for edge_id, (source, target) in enumerate(zip(sources, targets)):
        if source == target:
            continue
        weight = float(weights_cpu[edge_id].item())
        # igraph stores neighbors in a map and assigns duplicate keys, so the
        # last parallel edge weight wins instead of summing multiedges.
        adjacency[source][target] = weight
        adjacency[target][source] = weight
    return adjacency


def _initialize_positions(num_nodes: int, seed: int, fidelity_mode: bool = False) -> torch.Tensor:
    """Create default DRL initialization coordinates.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    seed : int
        Seed value for deterministic ``random.Random`` draws.
    fidelity_mode : bool, default=False
        When ``True``, use igraph's compiled default RNG stream.

    Returns
    -------
    torch.Tensor
        Initial positions with shape ``[N, 2]`` and dtype ``float64``.

    """
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float64)

    rng: RandomLike = make_igraph_default_rng(seed) if fidelity_mode else random.Random(seed)
    data = [[rng.random(), rng.random()] for _ in range(num_nodes)]
    return torch.tensor(data, dtype=torch.float64)


class _DensityGrid:
    """Density proxy used by DrL's coarse/fine repulsion term."""

    def __init__(self, config: DRLDensityGridConfig) -> None:
        """Initialize the coarse density grid.

        Parameters
        ----------
        config : DRLDensityGridConfig
            Grid resolution, view size, and kernel radius.
        """
        self.grid_size = config.grid_size
        self.view_size = config.view_size
        self.radius = config.radius
        self.cell_width = config.view_size / float(config.grid_size)
        self.origin = -0.5 * config.view_size
        self.density = torch.zeros((config.grid_size, config.grid_size), dtype=torch.float64)
        self.node_cells: dict[int, tuple[int, int]] = {}
        self.buckets: dict[tuple[int, int], set[int]] = {}

        axis = torch.arange(-config.radius, config.radius + 1, dtype=torch.float64)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        distance = torch.sqrt(xx.square() + yy.square())
        self.kernel = torch.clamp(1.0 - (distance / float(config.radius)), min=0.0)

    def _cell_index(self, position: torch.Tensor) -> tuple[int, int]:
        """Convert a point to a clamped cell index.

        Parameters
        ----------
        position : torch.Tensor
            Candidate point ``[2]``.

        Returns
        -------
        tuple[int, int]
            Cell coordinates in ``[0, grid_size)``.
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
        """Apply one tent kernel into the coarse density grid.

        Parameters
        ----------
        cell_x : int
            Center x-cell.
        cell_y : int
            Center y-cell.
        sign : float
            ``+1.0`` to add density, ``-1.0`` to remove it.
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
        """Insert or update one node in the density grid."""
        cell = self._cell_index(position)
        self.node_cells[node] = cell
        self._apply_kernel(cell[0], cell[1], sign=1.0)
        self.buckets.setdefault(cell, set()).add(node)

    def remove_node(self, node: int) -> None:
        """Remove one node from density grid and local neighbor buckets."""
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
        """Return coarse density penalty for one coordinate."""
        cell_x, cell_y = self._cell_index(position)
        value = float(self.density[cell_y, cell_x].item())
        return value * value

    def fine_density(
        self,
        node: int,
        position: torch.Tensor,
        positions: torch.Tensor,
        config: DRLEnergyConfig,
    ) -> float:
        """Return the exact local repulsion term for the simmer stage.

        Parameters
        ----------
        node : int
            Query node index.
        position : torch.Tensor
            Candidate coordinate with shape ``[2]``.
        positions : torch.Tensor
            Shared position tensor with shape ``[N, 2]``.
        config : DRLEnergyConfig
            Energy constants controlling repulsion scaling.

        Returns
        -------
        float
            Local fine-density penalty for the candidate coordinate.
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
                    distance_sq = float(delta.dot(delta).item()) + config.min_distance
                    density += config.fine_repulsion_scale / distance_sq
        return density


def _stage_power(phase_name: str) -> int:
    """Return attraction exponent for a phase name."""
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
    config: DRLEnergyConfig,
) -> float:
    """Evaluate one-node DrL energy for a candidate coordinate.

    Parameters
    ----------
    node : int
        Node index under evaluation.
    candidate : torch.Tensor
        Candidate coordinate with shape ``[2]``.
    positions : torch.Tensor
        Shared position tensor with shape ``[N, 2]``.
    adjacency : list[dict[int, float]]
        Weighted undirected adjacency used for attraction terms.
    attraction : float
        Current phase attraction coefficient.
    phase_name : str
        Active phase name controlling the distance exponent.
    density_grid : _DensityGrid
        Shared density proxy for repulsion.
    fine_density : bool
        Whether to use exact local density instead of the coarse grid.
    config : DRLEnergyConfig
        Energy constants controlling repulsion and jump scaling.

    Returns
    -------
    float
        Scalar energy for the candidate coordinate.
    """
    energy = 0.0
    attraction_factor = float(attraction**4) * config.attraction_factor_scale
    power = _stage_power(phase_name=phase_name)

    for neighbor, weight in adjacency[node].items():
        delta = candidate - positions[neighbor]
        distance_sq = float(delta.dot(delta).item())
        if distance_sq <= 0.0:
            continue
        energy += weight * attraction_factor * (distance_sq**power)

    if fine_density:
        energy += density_grid.fine_density(
            node=node,
            position=candidate,
            positions=positions,
            config=config,
        )
    else:
        energy += density_grid.coarse_density(position=candidate)
    return energy


def _weighted_centroid(
    node: int,
    positions: torch.Tensor,
    adjacency: list[dict[int, float]],
) -> torch.Tensor:
    """Compute weighted neighbor centroid for one node."""
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
    """Prune at most one long, high-stress outgoing edge for one node.

    Parameters
    ----------
    node : int
        Current node whose neighbor map may be cut.
    positions : torch.Tensor
        Current coordinates with shape ``[N, 2]``.
    adjacency : list[dict[int, float]]
        Mutable weighted adjacency. igraph cuts only ``adjacency[node]``, so the
        reverse neighbor map intentionally remains intact.
    min_edges : float
        Minimum current-node degree required before cutting is attempted.
    cut_off_length : float
        Score threshold above which one neighbor entry is removed.

    Returns
    -------
    None
        The current node's adjacency may be mutated in place.
    """
    neighbors = adjacency[node]
    if float(len(neighbors)) < min_edges or not neighbors:
        return

    centroid = _weighted_centroid(node=node, positions=positions, adjacency=adjacency)
    worst_neighbor = -1
    worst_score = -1.0
    degree_factor = math.sqrt(float(len(neighbors)))
    for neighbor in neighbors:
        delta = positions[neighbor] - centroid
        score = float(delta.dot(delta).item()) * degree_factor
        if score > worst_score:
            worst_score = score
            worst_neighbor = neighbor

    if worst_neighbor >= 0 and worst_score > cut_off_length:
        adjacency[node].pop(worst_neighbor, None)


class DRLNodeUpdate:
    """Update one node for a single DRL phase.

    Parameters
    ----------
    phase_name : str
        Name of the active DRL phase.
    fine_density : bool
        ``True`` for the final simmer phase, otherwise ``False``.
    """

    _phase_name: str
    _fine_density: bool
    _energy_config: DRLEnergyConfig

    def __init__(
        self,
        phase_name: str,
        fine_density: bool,
        energy_config: DRLEnergyConfig,
    ) -> None:
        """Store per-node phase context for repeated updates.

        Parameters
        ----------
        phase_name : str
            Active phase label.
        fine_density : bool
            ``True`` when the phase uses fine-grained density.
        energy_config : DRLEnergyConfig
            Constants used in energy evaluation and random jump sizing.
        """
        self._phase_name = phase_name
        self._fine_density = fine_density
        self._energy_config = energy_config

    def apply(
        self,
        node: int,
        positions: torch.Tensor,
        adjacency: list[dict[int, float]],
        rng: RandomLike,
        attraction: float,
        temperature: float,
        damping_mult: float,
        min_edges: float,
        cut_end: float,
        cut_off_length: float,
        density_grid: _DensityGrid,
    ) -> None:
        """Apply the DRL best-candidate update for a single node in-place.

        Parameters
        ----------
        node : int
            Node index currently updated.
        positions : torch.Tensor
            Shared node coordinate tensor with shape ``[N, 2]``.
        adjacency : list[dict[int, float]]
            Mutable adjacency used for attraction and edge-cut updates.
        rng : random.Random or IgraphPCG32
            Deterministic RNG used for node perturbation.
        attraction : float
            Current phase attraction weight.
        temperature : float
            Current phase temperature.
        damping_mult : float
            Current phase damping factor.
        min_edges : float
            Current minimum-edge threshold for candidate edge pruning.
        cut_end : float
            Final cut threshold scaled from graph-level edge-cut ratio.
        cut_off_length : float
            Current edge-cut score threshold.
        density_grid : _DensityGrid
            Shared density field proxy.

        Notes
        -----
        This method intentionally mutates ``positions``, ``adjacency``, and
        ``density_grid`` in place, matching the in-place behavior of classic DRL.
        """
        density_grid.remove_node(node=node)
        current = positions[node].clone()
        current_energy = _compute_energy(
            node=node,
            candidate=current,
            positions=positions,
            adjacency=adjacency,
            attraction=attraction,
            phase_name=self._phase_name,
            density_grid=density_grid,
            fine_density=self._fine_density,
            config=self._energy_config,
        )

        centroid = _weighted_centroid(
            node=node,
            positions=positions,
            adjacency=adjacency,
        )
        analytic = (positions[node] * (1.0 - damping_mult)) + (centroid * damping_mult)

        if self._phase_name in {"expansion", "cooldown"} and cut_end > 0.0:
            _maybe_cut_long_edge(
                node=node,
                positions=positions,
                adjacency=adjacency,
                min_edges=min_edges,
                cut_off_length=cut_off_length,
            )

        # DrL tests two candidates per node: the damped analytic point and a
        # small random perturbation scaled by the current phase temperature.
        jump_length = self._energy_config.jump_temperature_scale * temperature
        random_offset = torch.tensor(
            [
                (0.5 - rng.random()) * jump_length,
                (0.5 - rng.random()) * jump_length,
            ],
            dtype=torch.float64,
        )
        perturbed = analytic + random_offset

        perturbed_energy = _compute_energy(
            node=node,
            candidate=perturbed,
            positions=positions,
            adjacency=adjacency,
            attraction=attraction,
            phase_name=self._phase_name,
            density_grid=density_grid,
            fine_density=self._fine_density,
            config=self._energy_config,
        )

        # igraph compares old-position energy to perturbed-position energy, but
        # writes the analytic position when the old energy wins.
        if current_energy < perturbed_energy:
            positions[node] = analytic
        else:
            positions[node] = perturbed
        density_grid.add_node(node=node, position=positions[node])


class DRLPhaseStep:
    """Run one DRL phase, one node update at a time.

    Parameters
    ----------
    phase_name : str
        Name of the phase to execute.
    phase : _PhaseParameters
        Phase parameter bundle controlling schedule and attraction settings.
    """

    _phase_name: str
    _phase: _PhaseParameters
    _energy_config: DRLEnergyConfig
    _phase_dynamics_config: DRLPhaseDynamicsConfig

    def __init__(
        self,
        phase_name: str,
        phase: _PhaseParameters,
        energy_config: DRLEnergyConfig,
        phase_dynamics_config: DRLPhaseDynamicsConfig,
    ) -> None:
        """Store phase label and baseline parameters.

        Parameters
        ----------
        phase_name : str
            Active phase label.
        phase : _PhaseParameters
            Baseline phase parameters.
        energy_config : DRLEnergyConfig
            Energy-function constants used by node updates.
        phase_dynamics_config : DRLPhaseDynamicsConfig
            Per-phase decrement floors and step sizes.
        """
        self._phase_name = phase_name
        self._phase = phase
        self._energy_config = energy_config
        self._phase_dynamics_config = phase_dynamics_config

    def _update_count(self) -> int:
        """Return the number of node-update sweeps for this phase.

        Returns
        -------
        int
            Sweep count including igraph's pre-control init, boundary, and final
            updates.
        """
        if self._phase_name == "init":
            return 1
        if self._phase_name == "liquid":
            return int(self._phase.iterations)
        if self._phase_name == "simmer":
            return int(self._phase.iterations) + 2
        return int(self._phase.iterations) + 1

    def _advance_after_update(
        self,
        update_index: int,
        attraction: float,
        temperature: float,
        damping_mult: float,
        min_edges: float,
        cut_end: float,
        cut_rate: float,
        cut_off_length: float,
    ) -> Tuple[float, float, float, float, float]:
        """Advance igraph's automatic scheduler after one node-update sweep.

        Parameters
        ----------
        update_index : int
            Zero-based sweep index within this Dagua phase.
        attraction : float
            Current attraction value.
        temperature : float
            Current temperature value.
        damping_mult : float
            Current damping multiplier.
        min_edges : float
            Current edge-cut degree threshold.
        cut_end : float
            Final cut threshold.
        cut_rate : float
            Per-iteration cut cooling rate.
        cut_off_length : float
            Current cut-off length.

        Returns
        -------
        tuple[float, float, float, float, float]
            Updated ``(attraction, temperature, damping_mult, min_edges,
            cut_off_length)`` values.
        """
        if update_index >= int(self._phase.iterations):
            if self._phase_name == "cooldown":
                cut_off_length = cut_end
                min_edges = self._phase_dynamics_config.cooldown_min_edges_floor
            elif self._phase_name == "crunch":
                min_edges = 99.0
            return attraction, temperature, damping_mult, min_edges, cut_off_length

        if self._phase_name == "expansion":
            if attraction > self._phase_dynamics_config.expansion_attraction_floor:
                attraction = max(
                    self._phase_dynamics_config.expansion_attraction_floor,
                    attraction - self._phase_dynamics_config.expansion_attraction_delta,
                )
            if min_edges > self._phase_dynamics_config.expansion_min_edges_floor:
                min_edges = max(
                    self._phase_dynamics_config.expansion_min_edges_floor,
                    min_edges - self._phase_dynamics_config.expansion_min_edges_delta,
                )
            if cut_end > 0.0:
                cut_off_length = max(cut_end, cut_off_length - cut_rate)
            if damping_mult > self._phase_dynamics_config.expansion_damping_floor:
                damping_mult = max(
                    self._phase_dynamics_config.expansion_damping_floor,
                    damping_mult - self._phase_dynamics_config.expansion_damping_delta,
                )
        elif self._phase_name == "cooldown":
            if temperature > self._phase_dynamics_config.cooldown_temperature_floor:
                temperature = max(
                    self._phase_dynamics_config.cooldown_temperature_floor,
                    temperature - self._phase_dynamics_config.cooldown_temperature_delta,
                )
            if cut_end > 0.0:
                cut_off_length = max(
                    cut_end,
                    cut_off_length
                    - (self._phase_dynamics_config.cooldown_cut_rate_multiplier * cut_rate),
                )
            if min_edges > self._phase_dynamics_config.cooldown_min_edges_floor:
                min_edges = max(
                    self._phase_dynamics_config.cooldown_min_edges_floor,
                    min_edges - self._phase_dynamics_config.cooldown_min_edges_delta,
                )
        elif (
            self._phase_name == "simmer"
            and temperature > self._phase_dynamics_config.simmer_temperature_floor
        ):
            temperature = max(
                self._phase_dynamics_config.simmer_temperature_floor,
                temperature - self._phase_dynamics_config.simmer_temperature_delta,
            )
        return attraction, temperature, damping_mult, min_edges, cut_off_length

    def apply(
        self,
        positions: torch.Tensor,
        adjacency: list[dict[int, float]],
        rng: RandomLike,
        density_grid: _DensityGrid,
        cut_end: float,
        cut_rate: float,
        min_edges: float,
        cut_off_length: float,
    ) -> Tuple[float, float, float]:
        """Execute one DRL phase and return updated edge-cut state.

        Parameters
        ----------
        positions : torch.Tensor
            Shared coordinates updated in place, shape ``[N, 2]``.
        adjacency : list[dict[int, float]]
            Mutable adjacency structure.
        rng : random.Random
            Deterministic RNG used by node perturbations.
        density_grid : _DensityGrid
            Shared density field proxy.
        cut_end : float
            Scaled cut threshold boundary used across all phases.
        cut_rate : float
            Per-iteration cooling rate for cut-off length.
        min_edges : float
            Current minimum-edge threshold.
        cut_off_length : float
            Current cut-off length at phase entry.

        Returns
        -------
        tuple[float, float, float]
            Updated ``(temperature, min_edges, cut_off_length)`` after this phase.
        """
        attraction = float(self._phase.attraction)
        damping_mult = float(self._phase.damping_mult)
        fine_density = self._phase_name == "simmer"
        node_update = DRLNodeUpdate(
            phase_name=self._phase_name,
            fine_density=fine_density,
            energy_config=self._energy_config,
        )
        num_nodes = len(positions)

        temperature = float(self._phase.temperature)
        for update_index in range(self._update_count()):
            for node in range(num_nodes):
                node_update.apply(
                    node=node,
                    positions=positions,
                    adjacency=adjacency,
                    rng=rng,
                    attraction=attraction,
                    temperature=temperature,
                    damping_mult=damping_mult,
                    min_edges=min_edges,
                    cut_end=cut_end,
                    cut_off_length=cut_off_length,
                    density_grid=density_grid,
                )

            (
                attraction,
                temperature,
                damping_mult,
                min_edges,
                cut_off_length,
            ) = self._advance_after_update(
                update_index=update_index,
                attraction=attraction,
                temperature=temperature,
                damping_mult=damping_mult,
                min_edges=min_edges,
                cut_end=cut_end,
                cut_rate=cut_rate,
                cut_off_length=cut_off_length,
            )

        return temperature, min_edges, cut_off_length


@register_op
@dataclass(frozen=True)
class DRLPrepareState(Op):
    """Resolve DRL parameters and build the mutable adjacency state.

    The resulting extras entries are shared across all later DrL phases, so the
    adjacency remains mutable and can reflect the exact edge-cut behavior of the
    classic implementation.
    """

    name: ClassVar[str] = "drl_prepare_state"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[tuple[str, ...]] = ()
    writes: ClassVar[tuple[str, ...]] = ("extras",)
    requires: ClassVar[tuple[str, ...]] = ()
    config: DRLPrepareStateConfig = field(default_factory=DRLPrepareStateConfig)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Resolve options and build DRL adjacency in ``state.extras``.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph layout inputs.
        state : SolveState
            Mutable solve state receiving DRL extras.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with resolved DRL parameters and undirected adjacency.
        """
        del ctx

        params = _resolve_drl_parameters(options=self.config.options)
        adjacency = _build_undirected_adjacency(
            edge_index=problem.edge_index,
            num_nodes=problem.num_nodes,
            edge_weights=problem.edge_weights,
        )

        state.extras["drl_params"] = params
        state.extras["drl_adjacency"] = adjacency
        return state


@register_op
@dataclass(frozen=True)
class DRLInitializePositions(Op):
    """Seed the deterministic random starting layout used by classic DrL."""

    fidelity_mode: bool = False

    name: ClassVar[str] = "drl_initialize_positions"
    category: ClassVar[OpCategory] = OpCategory.INIT
    reads: ClassVar[tuple[str, ...]] = ()
    writes: ClassVar[tuple[str, ...]] = ("pos",)
    requires: ClassVar[tuple[str, ...]] = ()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Seed deterministic RNG-based starting coordinates.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs containing the seed and node count.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with initial ``float64`` DrL coordinates.
        """
        del ctx

        state.pos = _initialize_positions(
            num_nodes=problem.num_nodes,
            seed=problem.seed,
            fidelity_mode=self.fidelity_mode,
        )
        return state


@dataclass(frozen=True)
class DRLPhaseSolveConfig:
    """Configuration for :class:`DRLPhaseSolve`.

    Parameters
    ----------
    initial_min_edges : float, default=20.0
        Starting minimum-edge threshold for candidate edge pruning.
    cut_off_multiplier : float, default=4.0
        Multiplier applied to ``cut_end`` to initialize ``cut_off_length``.
    cut_rate_numerator : float, default=3.0
        Numerator multiplied by ``cut_end`` in the cut-rate formula.
    cut_rate_divisor : float, default=400.0
        Divisor for computing the per-iteration cut-rate cooling.
    cut_base : float, default=40000.0
        Base value multiplied by ``1 - edge_cut`` to derive ``cut_end``.
    density_grid : DRLDensityGridConfig, optional
        Density-grid resolution and view parameters.
    fidelity_mode : bool, default=False
        When ``True``, use igraph's compiled default RNG stream for random
        node perturbations.
    energy : DRLEnergyConfig, optional
        Energy-function constants used during node updates.
    phase_dynamics : DRLPhaseDynamicsConfig, optional
        Per-phase decrement floors and step sizes.
    """

    initial_min_edges: float = 20.0
    cut_off_multiplier: float = 4.0
    cut_rate_numerator: float = 3.0
    cut_rate_divisor: float = 400.0
    cut_base: float = 40_000.0
    density_grid: DRLDensityGridConfig = field(default_factory=DRLDensityGridConfig)
    fidelity_mode: bool = False
    energy: DRLEnergyConfig = field(default_factory=DRLEnergyConfig)
    phase_dynamics: DRLPhaseDynamicsConfig = field(default_factory=DRLPhaseDynamicsConfig)


@register_op
@dataclass(frozen=True)
class DRLPhaseSolve(Op):
    """Run all DRL phases and all node-level energy updates.

    The six phases intentionally run inside one coordinating op because DRL keeps
    a single mutable density grid and prunable adjacency across all phases.  A
    node update removes it from the density grid, samples candidates, then re-adds
    it. This state is consumed immediately by the next node in the same phase, and
    removed edges persist into later phases for exact edge-cut behavior. Any
    decomposition that snapshots these structures between node updates or phases
    would alter the optimization trajectory and break bit-identical output.
    """

    name: ClassVar[str] = "drl_phase_solve"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[tuple[str, ...]] = ("pos", "extras")
    writes: ClassVar[tuple[str, ...]] = ("pos", "extras")
    requires: ClassVar[tuple[str, ...]] = ("pos", "extras")
    config: DRLPhaseSolveConfig = field(default_factory=DRLPhaseSolveConfig)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Execute the full six-phase sequential DRL update loop.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs including node count and seed.
        state : SolveState
            Mutable solve state containing initialized positions and DRL extras.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with final DrL positions after all sequential phases.
        """
        del ctx

        if state.pos is None:
            raise ValueError("DRLPhaseSolve requires state.pos to be set.")

        params: _DrlParameters = state.extras["drl_params"]
        adjacency: list[dict[int, float]] = state.extras["drl_adjacency"]
        num_nodes = problem.num_nodes
        positions = state.pos

        density_grid = _DensityGrid(config=self.config.density_grid)
        for node in range(num_nodes):
            density_grid.add_node(node=node, position=positions[node])

        rng: RandomLike = (
            make_igraph_default_rng(problem.seed)
            if self.config.fidelity_mode
            else random.Random(problem.seed)
        )
        cut_end = self.config.cut_base * (1.0 - params.edge_cut)
        cut_off_length = self.config.cut_off_multiplier * cut_end
        cut_rate = (
            0.0
            if cut_end <= 0.0
            else (self.config.cut_rate_numerator * cut_end) / self.config.cut_rate_divisor
        )
        min_edges = self.config.initial_min_edges

        phase_specs = [
            ("init", params.init),
            ("liquid", params.liquid),
            ("expansion", params.expansion),
            ("cooldown", params.cooldown),
            ("crunch", params.crunch),
            ("simmer", params.simmer),
        ]

        for phase_name, phase in phase_specs:
            # Each phase mutates the shared adjacency and density state in place,
            # which is why the full phase loop remains inside one op.
            phase_step = DRLPhaseStep(
                phase_name=phase_name,
                phase=phase,
                energy_config=self.config.energy,
                phase_dynamics_config=self.config.phase_dynamics,
            )
            _, min_edges, cut_off_length = phase_step.apply(
                positions=positions,
                adjacency=adjacency,
                rng=rng,
                density_grid=density_grid,
                cut_end=cut_end,
                cut_rate=cut_rate,
                min_edges=min_edges,
                cut_off_length=cut_off_length,
            )

        state.pos = positions
        return state


@register_op
@dataclass(frozen=True)
class DRLFinalizePositions(Op):
    """Cast final DrL coordinates to the classic output dtype and device."""

    name: ClassVar[str] = "drl_finalize_positions"
    category: ClassVar[OpCategory] = OpCategory.POSTPROCESS
    reads: ClassVar[tuple[str, ...]] = ("pos",)
    writes: ClassVar[tuple[str, ...]] = ("pos",)
    requires: ClassVar[tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Match classic DrL output dtype and device.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs used to resolve the output device.
        state : SolveState
            Mutable solve state containing the final DrL coordinates.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with final ``float32`` coordinates on the target device.
        """
        del ctx

        if state.pos is None:
            raise ValueError("DRLFinalizePositions requires state.pos to be set.")

        output_device = layout_device(
            edge_index=problem.edge_index,
            node_sizes=problem.node_sizes,
        )
        state.pos = state.pos.to(dtype=torch.float32, device=output_device)
        return state
