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

import numpy as np
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

_IGRAPH_OUTPUT_SCALE = 50.0
_DENSITY_BOUNDARY_CELLS = 10
_DENSITY_EDGE_PENALTY = 10_000.0
_FINE_DENSITY_EPSILON = 1.0e-50


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
        weight = float(np.float32(weights_cpu[edge_id].item()))
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


def _initialize_adapter_seed_positions(num_nodes: int, seed: int) -> torch.Tensor:
    """Create the seeded matrix used by the igraph benchmark adapter.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    seed : int
        Integer seed forwarded by benchmark runners.

    Returns
    -------
    torch.Tensor
        Initial positions with shape ``[N, 2]`` and dtype ``float64``.
    """
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float64)
    positions = np.random.RandomState(seed).uniform(-1.0, 1.0, size=(num_nodes, 2))
    return torch.from_numpy(positions.astype(np.float64, copy=False))


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
        self.half_view = 0.5 * config.view_size
        self.view_to_grid = float(config.grid_size) / config.view_size
        self.density = np.zeros((config.grid_size, config.grid_size), dtype=np.float32)
        self.node_cells: dict[int, tuple[int, int]] = {}
        self.node_sub_positions: dict[int, tuple[float, float]] = {}
        self.buckets: dict[tuple[int, int], list[tuple[int, float, float]]] = {}

        diameter = (config.radius * 2) + 1
        self.kernel = np.zeros((diameter, diameter), dtype=np.float32)
        for row, offset_y in enumerate(range(-config.radius, config.radius + 1)):
            for col, offset_x in enumerate(range(-config.radius, config.radius + 1)):
                falloff_y = np.float32(
                    (config.radius - abs(float(np.float32(offset_y)))) / config.radius
                )
                falloff_x = np.float32(
                    (config.radius - abs(float(np.float32(offset_x)))) / config.radius
                )
                self.kernel[row, col] = np.float32(falloff_y * falloff_x)

    def _cell_index(self, position: torch.Tensor) -> tuple[int, int]:
        """Convert a point to a clamped cell index.

        Parameters
        ----------
        position : torch.Tensor
            Candidate point ``[2]``.

        Returns
        -------
        tuple[int, int]
            Unclamped cell coordinates.
        """
        x_value = float(position[0].item())
        y_value = float(position[1].item())
        return self._cell_index_xy(x_value=x_value, y_value=y_value)

    def _cell_index_xy(self, x_value: float, y_value: float) -> tuple[int, int]:
        """Convert scalar coordinates using igraph's bucket formula.

        Parameters
        ----------
        x_value : float
            X coordinate in DrL layout units.
        y_value : float
            Y coordinate in DrL layout units.

        Returns
        -------
        tuple[int, int]
            Unclamped integer cell coordinates.
        """
        cell_x = int((x_value + self.half_view + 0.5) * self.view_to_grid)
        cell_y = int((y_value + self.half_view + 0.5) * self.view_to_grid)
        return cell_x, cell_y

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
        x_start = cell_x - self.radius
        y_start = cell_y - self.radius
        diameter = self.radius * 2
        if (
            x_start >= self.grid_size
            or x_start < 0
            or y_start >= self.grid_size
            or y_start < 0
            or x_start + diameter >= self.grid_size
            or y_start + diameter >= self.grid_size
        ):
            raise RuntimeError("Exceeded density grid in DrL.")

        self.density[
            y_start : y_start + diameter + 1,
            x_start : x_start + diameter + 1,
        ] += np.float32(sign) * self.kernel

    def add_node(
        self,
        node: int,
        position: torch.Tensor,
        fine_density: bool = False,
    ) -> None:
        """Insert one node into the active density structure.

        Parameters
        ----------
        node : int
            Node index.
        position : torch.Tensor
            Coordinate tensor with shape ``[2]``.
        fine_density : bool, default=False
            Whether to update fine-density buckets instead of the coarse grid.
        """
        x_value = float(np.float32(position[0].item()))
        y_value = float(np.float32(position[1].item()))
        cell = self._cell_index_xy(x_value=x_value, y_value=y_value)
        self.node_cells[node] = cell
        self.node_sub_positions[node] = (x_value, y_value)
        if fine_density:
            self.buckets.setdefault(cell, []).append((node, x_value, y_value))
        else:
            self._apply_kernel(cell[0], cell[1], sign=1.0)

    def remove_node(
        self,
        node: int,
        fine_density: bool = False,
        first_add: bool = False,
        fine_first_add: bool = False,
    ) -> None:
        """Remove one node according to igraph's lifecycle flags.

        Parameters
        ----------
        node : int
            Node index.
        fine_density : bool, default=False
            Whether the solver is in the fine-density phase.
        first_add : bool, default=False
            Whether the coarse grid has not yet received its first sweep.
        fine_first_add : bool, default=False
            Whether the fine buckets have not yet received their first sweep.
        """
        if fine_density and not fine_first_add:
            self._fine_subtract(node=node)
        elif not first_add:
            self._coarse_subtract(node=node)

    def _coarse_subtract(self, node: int) -> None:
        """Subtract the node's last coarse-grid footprint.

        Parameters
        ----------
        node : int
            Node index to subtract.
        """
        sub_position = self.node_sub_positions.get(node)
        if sub_position is None:
            return
        cell = self._cell_index_xy(x_value=sub_position[0], y_value=sub_position[1])
        self._apply_kernel(cell[0], cell[1], sign=-1.0)

    def _fine_subtract(self, node: int) -> None:
        """Pop one node copy from its last fine-density bucket.

        Parameters
        ----------
        node : int
            Node index whose stored bucket position is removed.
        """
        sub_position = self.node_sub_positions.get(node)
        if sub_position is None:
            return
        cell = self._cell_index_xy(x_value=sub_position[0], y_value=sub_position[1])
        bucket = self.buckets.get(cell)
        if not bucket:
            return
        bucket.pop(0)
        if not bucket:
            del self.buckets[cell]

    def coarse_density(self, position: torch.Tensor) -> float:
        """Return coarse density penalty for one coordinate."""
        cell_x, cell_y = self._cell_index(position)
        if (
            cell_x > self.grid_size - _DENSITY_BOUNDARY_CELLS
            or cell_x < _DENSITY_BOUNDARY_CELLS
            or cell_y > self.grid_size - _DENSITY_BOUNDARY_CELLS
            or cell_y < _DENSITY_BOUNDARY_CELLS
        ):
            return _DENSITY_EDGE_PENALTY
        value = float(np.float32(self.density[cell_y, cell_x]))
        return _as_float32(value * value)

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
        if (
            cell_x > self.grid_size - _DENSITY_BOUNDARY_CELLS
            or cell_x < _DENSITY_BOUNDARY_CELLS
            or cell_y > self.grid_size - _DENSITY_BOUNDARY_CELLS
            or cell_y < _DENSITY_BOUNDARY_CELLS
        ):
            return _DENSITY_EDGE_PENALTY
        density = 0.0
        for offset_y in (-1, 0, 1):
            for offset_x in (-1, 0, 1):
                neighbor_cell = (cell_x + offset_x, cell_y + offset_y)
                bucket = self.buckets.get(neighbor_cell)
                if not bucket:
                    continue
                for other, other_x, other_y in bucket:
                    del other
                    x_dist = _as_float32(float(np.float32(position[0].item())) - other_x)
                    y_dist = _as_float32(float(np.float32(position[1].item())) - other_y)
                    distance_sq = _as_float32(
                        _as_float32(x_dist * x_dist) + _as_float32(y_dist * y_dist)
                    )
                    density = _as_float32(
                        density
                        + (config.fine_repulsion_scale / (distance_sq + _FINE_DENSITY_EPSILON))
                    )
        return _as_float32(density)


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


def _as_float32(value: float) -> float:
    """Round a scalar through C++ ``float`` precision.

    Parameters
    ----------
    value : float
        Input scalar.

    Returns
    -------
    float
        Python float carrying the nearest ``float32`` value.
    """
    return float(np.float32(value))


def _tensor_from_xy(x_value: float, y_value: float) -> torch.Tensor:
    """Build a two-coordinate tensor for density-grid calls.

    Parameters
    ----------
    x_value : float
        X coordinate.
    y_value : float
        Y coordinate.

    Returns
    -------
    torch.Tensor
        Coordinate tensor with shape ``[2]`` and dtype ``float64``.
    """
    return torch.tensor([x_value, y_value], dtype=torch.float64)


def _positions_to_nodes(positions: torch.Tensor) -> list[list[float]]:
    """Convert an initialized tensor into mutable float32 coordinates.

    Parameters
    ----------
    positions : torch.Tensor
        Initial position tensor with shape ``[N, 2]``.

    Returns
    -------
    list[list[float]]
        Mutable ``[[x, y], ...]`` coordinates rounded to C++ ``float`` values.
    """
    positions_cpu = positions.to(device="cpu", dtype=torch.float64)
    return [
        [
            _as_float32(float(positions_cpu[node, 0].item())),
            _as_float32(float(positions_cpu[node, 1].item())),
        ]
        for node in range(positions_cpu.shape[0])
    ]


def _nodes_to_tensor(nodes: list[list[float]]) -> torch.Tensor:
    """Convert mutable runtime nodes back to a tensor.

    Parameters
    ----------
    nodes : list[list[float]]
        Mutable ``[[x, y], ...]`` runtime coordinates.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]`` and dtype ``float64``.
    """
    if not nodes:
        return torch.empty((0, 2), dtype=torch.float64)
    return torch.tensor(nodes, dtype=torch.float64)


def _runtime_energy(
    node: int,
    nodes: list[list[float]],
    adjacency: list[dict[int, float]],
    attraction: float,
    stage: int,
    fine_density: bool,
    density_grid: _DensityGrid,
) -> float:
    """Compute igraph DrL node energy for the current coordinate.

    Parameters
    ----------
    node : int
        Node index to evaluate.
    nodes : list[list[float]]
        Mutable runtime coordinates.
    adjacency : list[dict[int, float]]
        Weighted undirected adjacency.
    attraction : float
        Current attraction parameter.
    stage : int
        Current igraph stage number.
    fine_density : bool
        Whether fine-density buckets are active.
    density_grid : _DensityGrid
        Density grid matching igraph bucket semantics.

    Returns
    -------
    float
        Scalar energy rounded through float32 accumulation.
    """
    attraction_factor = _as_float32(
        _as_float32(_as_float32(_as_float32(attraction * attraction) * attraction) * attraction)
        * 2.0e-2
    )
    node_energy = _as_float32(0.0)
    node_x, node_y = nodes[node]

    for neighbor in sorted(adjacency[node]):
        weight = _as_float32(adjacency[node][neighbor])
        x_dis = _as_float32(node_x - nodes[neighbor][0])
        y_dis = _as_float32(node_y - nodes[neighbor][1])
        energy_distance = _as_float32(_as_float32(x_dis * x_dis) + _as_float32(y_dis * y_dis))
        if stage < 2:
            energy_distance = _as_float32(energy_distance * energy_distance)
        if stage == 0:
            energy_distance = _as_float32(energy_distance * energy_distance)
        node_energy = _as_float32(
            node_energy + _as_float32(_as_float32(weight * attraction_factor) * energy_distance)
        )

    position = _tensor_from_xy(x_value=node_x, y_value=node_y)
    if fine_density:
        density = density_grid.fine_density(
            node=node,
            position=position,
            positions=_nodes_to_tensor(nodes),
            config=DRLEnergyConfig(),
        )
    else:
        density = density_grid.coarse_density(position=position)
    return _as_float32(node_energy + _as_float32(density))


def _runtime_solve_analytic(
    node: int,
    nodes: list[list[float]],
    adjacency: list[dict[int, float]],
    damping_mult: float,
    min_edges: float,
    cut_end: float,
    cut_off_length: float,
) -> tuple[float, float]:
    """Compute igraph's analytic centroid candidate and prune one edge.

    Parameters
    ----------
    node : int
        Node index to update.
    nodes : list[list[float]]
        Mutable runtime coordinates.
    adjacency : list[dict[int, float]]
        Mutable weighted adjacency.
    damping_mult : float
        Current damping multiplier.
    min_edges : float
        Current minimum degree threshold for pruning.
    cut_end : float
        Unfloored edge-cut parameter used by igraph's no-cut guard.
    cut_off_length : float
        Current edge-cut length threshold.

    Returns
    -------
    tuple[float, float]
        Analytic candidate coordinate.
    """
    total_weight = _as_float32(0.0)
    x_sum = _as_float32(0.0)
    y_sum = _as_float32(0.0)

    for neighbor in sorted(adjacency[node]):
        weight = _as_float32(adjacency[node][neighbor])
        total_weight = _as_float32(total_weight + weight)
        x_sum = _as_float32(x_sum + _as_float32(weight * nodes[neighbor][0]))
        y_sum = _as_float32(y_sum + _as_float32(weight * nodes[neighbor][1]))

    if total_weight > 0.0:
        x_cen = _as_float32(x_sum / total_weight)
        y_cen = _as_float32(y_sum / total_weight)
        damping = _as_float32(1.0 - damping_mult)
        centroid_scale = 1.0 - damping
        pos_x = _as_float32(_as_float32(damping * nodes[node][0]) + (centroid_scale * x_cen))
        pos_y = _as_float32(_as_float32(damping * nodes[node][1]) + (centroid_scale * y_cen))
    else:
        x_cen = _as_float32(0.0)
        y_cen = _as_float32(0.0)
        pos_x = nodes[node][0]
        pos_y = nodes[node][1]

    if min_edges == 99.0 or cut_end >= 39_500.0:
        return pos_x, pos_y

    num_connections = _as_float32(math.sqrt(float(len(adjacency[node]))))
    max_length = _as_float32(0.0)
    max_neighbor: Optional[int] = None
    for neighbor in sorted(adjacency[node]):
        if len(adjacency[node]) < min_edges:
            continue
        x_dis = _as_float32(x_cen - nodes[neighbor][0])
        y_dis = _as_float32(y_cen - nodes[neighbor][1])
        distance = _as_float32(_as_float32(x_dis * x_dis) + _as_float32(y_dis * y_dis))
        distance = _as_float32(distance * num_connections)
        if distance > max_length:
            max_length = distance
            max_neighbor = neighbor

    if max_neighbor is not None and max_length > cut_off_length:
        adjacency[node].pop(max_neighbor, None)
    return pos_x, pos_y


def _runtime_update_node(
    node: int,
    nodes: list[list[float]],
    adjacency: list[dict[int, float]],
    rng: random.Random,
    density_grid: _DensityGrid,
    stage: int,
    temperature: float,
    attraction: float,
    damping_mult: float,
    min_edges: float,
    cut_end: float,
    cut_off_length: float,
    first_add: bool,
    fine_first_add: bool,
    fine_density: bool,
) -> tuple[float, float, float]:
    """Compute one node's next coordinate using igraph's candidate rule.

    Parameters
    ----------
    node : int
        Node index to update.
    nodes : list[list[float]]
        Mutable runtime coordinates.
    adjacency : list[dict[int, float]]
        Mutable weighted adjacency.
    rng : random.Random
        Python RNG installed by the igraph benchmark adapter.
    density_grid : _DensityGrid
        Coarse and fine density storage.
    stage : int
        Current igraph stage number.
    temperature : float
        Current temperature.
    attraction : float
        Current attraction.
    damping_mult : float
        Current damping multiplier.
    min_edges : float
        Current minimum degree threshold for pruning.
    cut_end : float
        Unfloored edge-cut parameter.
    cut_off_length : float
        Current edge-cut length threshold.
    first_add : bool
        Whether the coarse grid is still empty.
    fine_first_add : bool
        Whether the fine buckets are still empty.
    fine_density : bool
        Whether fine-density buckets are active.

    Returns
    -------
    tuple[float, float, float]
        New x-coordinate, new y-coordinate, and accepted energy.
    """
    old_x, old_y = nodes[node]
    density_grid.remove_node(
        node=node,
        fine_density=fine_density,
        first_add=first_add,
        fine_first_add=fine_first_add,
    )
    old_energy = _runtime_energy(
        node=node,
        nodes=nodes,
        adjacency=adjacency,
        attraction=attraction,
        stage=stage,
        fine_density=fine_density,
        density_grid=density_grid,
    )

    analytic_x, analytic_y = _runtime_solve_analytic(
        node=node,
        nodes=nodes,
        adjacency=adjacency,
        damping_mult=damping_mult,
        min_edges=min_edges,
        cut_end=cut_end,
        cut_off_length=cut_off_length,
    )
    nodes[node][0] = analytic_x
    nodes[node][1] = analytic_y

    jump_length = _as_float32(0.010 * temperature)
    random_x = _as_float32(analytic_x + ((0.5 - rng.random()) * jump_length))
    random_y = _as_float32(analytic_y + ((0.5 - rng.random()) * jump_length))
    nodes[node][0] = random_x
    nodes[node][1] = random_y
    random_energy = _runtime_energy(
        node=node,
        nodes=nodes,
        adjacency=adjacency,
        attraction=attraction,
        stage=stage,
        fine_density=fine_density,
        density_grid=density_grid,
    )

    nodes[node][0] = old_x
    nodes[node][1] = old_y
    if not fine_density and not first_add:
        density_grid.add_node(
            node=node,
            position=_tensor_from_xy(x_value=old_x, y_value=old_y),
            fine_density=fine_density,
        )
    elif not fine_first_add:
        density_grid.add_node(
            node=node,
            position=_tensor_from_xy(x_value=old_x, y_value=old_y),
            fine_density=fine_density,
        )

    if old_energy < random_energy:
        return analytic_x, analytic_y, old_energy
    return random_x, random_y, random_energy


def _runtime_update_density(
    node: int,
    nodes: list[list[float]],
    new_x: float,
    new_y: float,
    density_grid: _DensityGrid,
    first_add: bool,
    fine_first_add: bool,
    fine_density: bool,
) -> None:
    """Apply igraph's old-position subtraction and new-position insertion.

    Parameters
    ----------
    node : int
        Node index to update.
    nodes : list[list[float]]
        Mutable runtime coordinates.
    new_x : float
        Accepted x-coordinate.
    new_y : float
        Accepted y-coordinate.
    density_grid : _DensityGrid
        Coarse and fine density storage.
    first_add : bool
        Whether the coarse grid is still empty.
    fine_first_add : bool
        Whether the fine buckets are still empty.
    fine_density : bool
        Whether fine-density buckets are active.
    """
    density_grid.remove_node(
        node=node,
        fine_density=fine_density,
        first_add=first_add,
        fine_first_add=fine_first_add,
    )
    nodes[node][0] = new_x
    nodes[node][1] = new_y
    density_grid.add_node(
        node=node,
        position=_tensor_from_xy(x_value=new_x, y_value=new_y),
        fine_density=fine_density,
    )


def _runtime_update_nodes(
    nodes: list[list[float]],
    energies: list[float],
    adjacency: list[dict[int, float]],
    rng: random.Random,
    density_grid: _DensityGrid,
    stage: int,
    temperature: float,
    attraction: float,
    damping_mult: float,
    min_edges: float,
    cut_end: float,
    cut_off_length: float,
    first_add: bool,
    fine_first_add: bool,
    fine_density: bool,
) -> tuple[bool, bool]:
    """Run one full igraph node-update sweep.

    Parameters
    ----------
    nodes : list[list[float]]
        Mutable runtime coordinates.
    energies : list[float]
        Per-node accepted energies.
    adjacency : list[dict[int, float]]
        Mutable weighted adjacency.
    rng : random.Random
        Python RNG used for random jumps.
    density_grid : _DensityGrid
        Coarse and fine density storage.
    stage : int
        Current igraph stage number.
    temperature : float
        Current temperature.
    attraction : float
        Current attraction.
    damping_mult : float
        Current damping multiplier.
    min_edges : float
        Current minimum degree threshold for pruning.
    cut_end : float
        Unfloored edge-cut parameter.
    cut_off_length : float
        Current edge-cut length threshold.
    first_add : bool
        Whether the coarse grid is still empty.
    fine_first_add : bool
        Whether the fine buckets are still empty.
    fine_density : bool
        Whether fine-density buckets are active.

    Returns
    -------
    tuple[bool, bool]
        Updated ``first_add`` and ``fine_first_add`` flags.
    """
    for node in range(len(nodes)):
        new_x, new_y, energy = _runtime_update_node(
            node=node,
            nodes=nodes,
            adjacency=adjacency,
            rng=rng,
            density_grid=density_grid,
            stage=stage,
            temperature=temperature,
            attraction=attraction,
            damping_mult=damping_mult,
            min_edges=min_edges,
            cut_end=cut_end,
            cut_off_length=cut_off_length,
            first_add=first_add,
            fine_first_add=fine_first_add,
            fine_density=fine_density,
        )
        energies[node] = energy
        _runtime_update_density(
            node=node,
            nodes=nodes,
            new_x=new_x,
            new_y=new_y,
            density_grid=density_grid,
            first_add=first_add,
            fine_first_add=fine_first_add,
            fine_density=fine_density,
        )

    first_add = False
    if fine_density:
        fine_first_add = False
    return first_add, fine_first_add


def _run_reference_drl(
    initial_positions: torch.Tensor,
    adjacency: list[dict[int, float]],
    params: _DrlParameters,
    seed: int,
    density_config: DRLDensityGridConfig,
) -> torch.Tensor:
    """Run igraph's single-process DrL state machine without delegation.

    Parameters
    ----------
    initial_positions : torch.Tensor
        Initial coordinates with shape ``[N, 2]``.
    adjacency : list[dict[int, float]]
        Mutable weighted adjacency.
    params : _DrlParameters
        Resolved DrL phase parameters.
    seed : int
        Seed for igraph's Python RNG hook used by the benchmark adapter.
    density_config : DRLDensityGridConfig
        Density-grid constants.

    Returns
    -------
    torch.Tensor
        Final unscaled positions with shape ``[N, 2]`` and dtype ``float64``.
    """
    nodes = _positions_to_nodes(initial_positions)
    energies = [_as_float32(0.0) for _ in nodes]
    density_grid = _DensityGrid(config=density_config)
    rng = random.Random(seed)

    stage = 0
    iterations = int(params.init.iterations)
    temperature = _as_float32(params.init.temperature)
    attraction = _as_float32(params.init.attraction)
    damping_mult = _as_float32(params.init.damping_mult)
    min_edges = _as_float32(20.0)
    first_add = True
    fine_first_add = True
    fine_density = False

    edge_cut = _as_float32(params.edge_cut)
    cut_end = _as_float32(40_000.0 * (1.0 - edge_cut))
    cut_length_end = cut_end
    if cut_length_end <= 1.0:
        cut_length_end = _as_float32(1.0)
    cut_length_start = _as_float32(4.0 * cut_length_end)
    cut_off_length = cut_length_start
    cut_rate = _as_float32((cut_length_start - cut_length_end) / 400.0)

    while True:
        first_add, fine_first_add = _runtime_update_nodes(
            nodes=nodes,
            energies=energies,
            adjacency=adjacency,
            rng=rng,
            density_grid=density_grid,
            stage=stage,
            temperature=temperature,
            attraction=attraction,
            damping_mult=damping_mult,
            min_edges=min_edges,
            cut_end=cut_end,
            cut_off_length=cut_off_length,
            first_add=first_add,
            fine_first_add=fine_first_add,
            fine_density=fine_density,
        )
        if stage == 6:
            break

        if stage == 0:
            if iterations < int(params.liquid.iterations):
                temperature = _as_float32(params.liquid.temperature)
                attraction = _as_float32(params.liquid.attraction)
                damping_mult = _as_float32(params.liquid.damping_mult)
                iterations += 1
            else:
                temperature = _as_float32(params.expansion.temperature)
                attraction = _as_float32(params.expansion.attraction)
                damping_mult = _as_float32(params.expansion.damping_mult)
                iterations = 0
                stage = 1

        if stage == 1:
            if iterations < int(params.expansion.iterations):
                if attraction > 1.0:
                    attraction = _as_float32(attraction - np.float32(0.05))
                if min_edges > 12.0:
                    min_edges = _as_float32(min_edges - np.float32(0.05))
                cut_off_length = _as_float32(cut_off_length - cut_rate)
                if damping_mult > 0.1:
                    damping_mult = _as_float32(damping_mult - np.float32(0.005))
                iterations += 1
            else:
                min_edges = _as_float32(12.0)
                damping_mult = _as_float32(params.cooldown.damping_mult)
                stage = 2
                attraction = _as_float32(params.cooldown.attraction)
                temperature = _as_float32(params.cooldown.temperature)
                iterations = 0
        elif stage == 2:
            if iterations < int(params.cooldown.iterations):
                if temperature > 50.0:
                    temperature = _as_float32(temperature - 10.0)
                if cut_off_length > cut_length_end:
                    cut_off_length = _as_float32(
                        cut_off_length - _as_float32(cut_rate * np.float32(2.0))
                    )
                if min_edges > 1.0:
                    min_edges = _as_float32(min_edges - np.float32(0.2))
                iterations += 1
            else:
                cut_off_length = cut_length_end
                temperature = _as_float32(params.crunch.temperature)
                damping_mult = _as_float32(params.crunch.damping_mult)
                min_edges = _as_float32(1.0)
                stage = 3
                iterations = 0
                attraction = _as_float32(params.crunch.attraction)
        elif stage == 3:
            if iterations < int(params.crunch.iterations):
                iterations += 1
            else:
                iterations = 0
                temperature = _as_float32(params.simmer.temperature)
                attraction = _as_float32(params.simmer.attraction)
                damping_mult = _as_float32(params.simmer.damping_mult)
                min_edges = _as_float32(99.0)
                fine_density = True
                stage = 5
        elif stage == 5:
            if iterations < int(params.simmer.iterations):
                if temperature > 50.0:
                    temperature = _as_float32(temperature - 2.0)
                iterations += 1
            else:
                stage = 6

    return _nodes_to_tensor(nodes)


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

        if self.fidelity_mode:
            state.pos = _initialize_adapter_seed_positions(
                num_nodes=problem.num_nodes,
                seed=problem.seed,
            )
        else:
            state.pos = _initialize_positions(
                num_nodes=problem.num_nodes,
                seed=problem.seed,
                fidelity_mode=False,
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
        del num_nodes

        state.pos = _run_reference_drl(
            initial_positions=state.pos,
            adjacency=adjacency,
            params=params,
            seed=problem.seed,
            density_config=self.config.density_grid,
        )
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
        state.pos = (state.pos * _IGRAPH_OUTPUT_SCALE).to(dtype=torch.float32, device=output_device)
        return state
