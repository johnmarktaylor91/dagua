"""Davidson-Harel ops for the composable pipeline implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar, List, Optional, Tuple

import torch

from dagua.layout.ops.base import Op
from dagua.layout.ops.graph_utils import layout_device as _layout_device
from dagua.layout.ops.graph_utils import layout_extent as _layout_extent
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_MIN_DISTANCE = 1.0e-3
_NODE_DIST_WEIGHT = 1.0
_BORDER_WEIGHT = 0.0
_EDGE_LENGTH_WEIGHT = 0.0001
_CROSSING_WEIGHT = 1.0
_NODE_EDGE_WEIGHT = 0.2
_COLLINEAR_EPSILON = 1.0e-10
_MOVE_TRIES = 30
_MOVE_DIRECTIONS = tuple(
    (math.cos(2.0 * math.pi * index / _MOVE_TRIES), math.sin(2.0 * math.pi * index / _MOVE_TRIES))
    for index in range(_MOVE_TRIES)
)

_DH_EDGES_KEY = "dh_edges"
_DH_UNIQUE_EDGE_WEIGHTS_KEY = "dh_unique_edge_weights"
_DH_CURRENT_ENERGY_KEY = "dh_current_energy"
_DH_INITIAL_TEMPERATURE_KEY = "dh_initial_temperature"
_DH_GENERATOR_KEY = "dh_generator"
_DH_DEVICE_KEY = "dh_device"
_DH_EXTENT_KEY = "dh_extent"


def _unique_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor] = None,
) -> tuple[list[tuple[int, int]], torch.Tensor]:
    """Return unique undirected edges with aggregated weights."""
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")

    seen: dict[tuple[int, int], float] = {}
    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    if edge_weights is None:
        weights_cpu = torch.ones((edge_index.shape[1],), dtype=torch.float32)
    else:
        weights_cpu = edge_weights.detach().to(device="cpu", dtype=torch.float32)

    for edge_id, (source, target) in enumerate(
        zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist())
    ):
        if source < 0 or source >= num_nodes or target < 0 or target >= num_nodes:
            raise ValueError("edge_index contains a node index outside [0, num_nodes).")
        if source == target:
            continue
        pair = (min(source, target), max(source, target))
        seen[pair] = seen.get(pair, 0.0) + float(weights_cpu[edge_id].item())

    ordered_edges = sorted(seen)
    ordered_weights = torch.tensor(
        [seen[edge] for edge in ordered_edges],
        dtype=torch.float32,
    )
    return ordered_edges, ordered_weights


def _initialize_positions(
    num_nodes: int,
    extent: float,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Seed deterministic positions in ``[-extent, extent]``."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return ((torch.rand((num_nodes, 2), generator=generator) * 2.0) - 1.0).to(device) * extent


def _orientation(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> float:
    """Return signed area for three points."""
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _segments_intersect(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
) -> bool:
    """Return ``True`` when two segments intersect in XY plane."""
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    return (abs(o1) < _COLLINEAR_EPSILON or abs(o2) < _COLLINEAR_EPSILON or o1 * o2 < 0.0) and (
        abs(o3) < _COLLINEAR_EPSILON or abs(o4) < _COLLINEAR_EPSILON or o3 * o4 < 0.0
    )


def _point_segment_distance(
    point: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
) -> torch.Tensor:
    """Compute distance from a point to a segment endpoint pair."""
    segment = end - start
    denom = segment.dot(segment).clamp(min=_MIN_DISTANCE)
    projection = ((point - start).dot(segment) / denom).clamp(0.0, 1.0)
    nearest = start + projection * segment
    return torch.linalg.norm(point - nearest)


def _energy(
    positions: torch.Tensor,
    edges: List[Tuple[int, int]],
    extent: float,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Evaluate the Davidson-Harel objective for one candidate layout.

    Parameters
    ----------
    positions : torch.Tensor
        Candidate node coordinates with shape ``[N, 2]``.
    edges : list of tuple of int
        Undirected edge endpoints used by the layout energy.
    extent : float
        Half-width and half-height of the square layout bounding box.
    edge_weights : torch.Tensor, optional
        Aggregated edge weights with shape ``[E]``. When omitted, each edge has
        unit weight.

    Returns
    -------
    torch.Tensor
        Scalar unnormalized Davidson-Harel energy using igraph's default
        weights.
    """
    num_nodes = int(positions.shape[0])
    distribution = torch.tensor(0.0, dtype=positions.dtype, device=positions.device)
    if num_nodes > 1:
        src, dst = torch.triu_indices(num_nodes, num_nodes, offset=1, device=positions.device)
        squared_distances = (
            (positions[src] - positions[dst]).square().sum(dim=1).clamp(min=_MIN_DISTANCE)
        )
        distribution = squared_distances.reciprocal().sum()

    border_distances = torch.stack(
        [
            positions[:, 0] + extent,
            extent - positions[:, 0],
            positions[:, 1] + extent,
            extent - positions[:, 1],
        ],
        dim=1,
    ).clamp(min=_MIN_DISTANCE)
    border = border_distances.reciprocal().square().sum()

    edge_length = torch.tensor(0.0, dtype=positions.dtype, device=positions.device)
    if edges:
        edge_weight_tensor = (
            torch.ones((len(edges),), dtype=positions.dtype, device=positions.device)
            if edge_weights is None
            else edge_weights.to(device=positions.device, dtype=positions.dtype)
        )
        edge_lengths = [
            torch.linalg.norm(positions[source] - positions[target]).square()
            * edge_weight_tensor[index]
            for index, (source, target) in enumerate(edges)
        ]
        edge_length = torch.stack(edge_lengths).sum()

    crossings = 0.0
    for index, (a, b) in enumerate(edges):
        for c, d in edges[index + 1 :]:
            if len({a, b, c, d}) < 4:
                continue
            if _segments_intersect(positions[a], positions[b], positions[c], positions[d]):
                crossings += 1.0
    crossing_energy = torch.tensor(crossings, dtype=positions.dtype, device=positions.device)

    node_edge = torch.tensor(0.0, dtype=positions.dtype, device=positions.device)
    penalties: list[torch.Tensor] = []
    for node in range(num_nodes):
        for source, target in edges:
            if node in (source, target):
                continue
            distance = _point_segment_distance(
                positions[node], positions[source], positions[target]
            )
            penalties.append(distance.clamp(min=_MIN_DISTANCE).reciprocal().square())
    if penalties:
        node_edge = torch.stack(penalties).sum()

    return (
        _NODE_DIST_WEIGHT * distribution
        + _BORDER_WEIGHT * border
        + _EDGE_LENGTH_WEIGHT * edge_length
        + _CROSSING_WEIGHT * crossing_energy
        + _NODE_EDGE_WEIGHT * node_edge
    )


@dataclass(frozen=True)
class PrepareDHStateConfig:
    """Configuration for :class:`PrepareDHState`.

    Parameters
    ----------
    initial_temperature_scale : float, default=0.1
        Fraction of the starting energy used as the initial annealing
        temperature before the minimum-distance floor is applied.
        Deprecated for Davidson-Harel parity; igraph uses the move radius as
        the simulated-annealing denominator.
    """

    initial_temperature_scale: float = 0.1


@dataclass(frozen=True)
class DHAnnealingRoundConfig:
    """Configuration for :class:`DHAnnealingRound`.

    Parameters
    ----------
    move_scale_fraction : float, default=0.25
        Fraction of the layout extent used for the maximum proposal radius at
        the starting temperature.
        Deprecated for Davidson-Harel parity; igraph initializes the move
        radius to the full half-width of the layout square.
    """

    move_scale_fraction: float = 0.25


@dataclass(frozen=True)
class DHCoolConfig:
    """Configuration for :class:`DHCool`.

    Parameters
    ----------
    cooling_factor : float, default=0.75
        Geometric decay factor applied to ``state.temperature`` after each
        annealing round.
    """

    cooling_factor: float = 0.75


@register_op
@dataclass(frozen=True)
class InitializeDHPositions(Op):
    """Initialize random coordinates and cached layout extent metadata."""

    name: ClassVar[str] = "dh_initialize_positions"
    category: ClassVar[OpCategory] = OpCategory.INIT
    writes: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Seed ``state.pos`` and cache extent/device for DH rounds."""
        del ctx
        extent = _layout_extent(problem.num_nodes, problem.node_sizes)
        device = _layout_device(problem.edge_index, problem.node_sizes)
        state.pos = _initialize_positions(problem.num_nodes, extent, device, problem.seed)
        state.extras[_DH_EXTENT_KEY] = extent
        state.extras[_DH_DEVICE_KEY] = device
        return state


@register_op
@dataclass(frozen=True)
class PrepareDHState(Op):
    """Build cached, undirected edge state and the initial annealing energy."""

    config: PrepareDHStateConfig = field(default_factory=PrepareDHStateConfig)

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
        """Precompute edge cache, generator, and initial energy estimate."""
        del ctx

        assert state.pos is not None

        edges, unique_edge_weights = _unique_edges(
            problem.edge_index,
            problem.num_nodes,
            edge_weights=problem.edge_weights,
        )
        state.extras[_DH_EDGES_KEY] = edges
        state.extras[_DH_UNIQUE_EDGE_WEIGHTS_KEY] = unique_edge_weights

        if problem.edge_weights is None:
            current_energy = _energy(state.pos, edges, state.extras[_DH_EXTENT_KEY])
        else:
            current_energy = _energy(
                state.pos,
                edges,
                state.extras[_DH_EXTENT_KEY],
                unique_edge_weights,
            )
        state.extras[_DH_CURRENT_ENERGY_KEY] = current_energy

        # igraph's Davidson-Harel implementation uses ``move_radius`` both as
        # proposal radius and as the denominator in exp(-dE / move_radius).
        # The initial radius is width / 2, which is the same half-width stored
        # in ``_DH_EXTENT_KEY`` for size-free layouts.
        initial_temperature = max(float(state.extras[_DH_EXTENT_KEY]), _MIN_DISTANCE)
        state.extras[_DH_INITIAL_TEMPERATURE_KEY] = initial_temperature
        state.temperature = initial_temperature

        generator = torch.Generator(device="cpu")
        generator.manual_seed(problem.seed)
        state.extras[_DH_GENERATOR_KEY] = generator

        return state


@register_op
@dataclass(frozen=True)
class DHAnnealingRound(Op):
    """Apply one Davidson-Harel annealing round."""

    config: DHAnnealingRoundConfig = field(default_factory=DHAnnealingRoundConfig)

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
        """Propose and accept/reject circular moves for each node."""
        del ctx

        assert state.pos is not None

        positions = state.pos
        edges: List[Tuple[int, int]] = state.extras[_DH_EDGES_KEY]
        unique_edge_weights: torch.Tensor = state.extras[_DH_UNIQUE_EDGE_WEIGHTS_KEY]
        extent: float = state.extras[_DH_EXTENT_KEY]
        current_energy: torch.Tensor = state.extras[_DH_CURRENT_ENERGY_KEY]
        temperature: float = state.temperature
        generator: torch.Generator = state.extras[_DH_GENERATOR_KEY]
        device: torch.device = state.extras[_DH_DEVICE_KEY]

        node_order = torch.randperm(problem.num_nodes, generator=generator)

        for node in node_order.tolist():
            move_scale = min(max(temperature, _MIN_DISTANCE), extent)
            direction_order = torch.randperm(_MOVE_TRIES, generator=generator)
            for direction_id in direction_order.tolist():
                delta = (
                    torch.tensor(
                        _MOVE_DIRECTIONS[direction_id],
                        dtype=positions.dtype,
                        device=device,
                    )
                    * move_scale
                )
                candidate = positions.clone()
                candidate[node] = (candidate[node] + delta).clamp(min=-extent, max=extent)
                if problem.edge_weights is None:
                    candidate_energy = _energy(candidate, edges, extent)
                else:
                    candidate_energy = _energy(candidate, edges, extent, unique_edge_weights)

                delta_energy = candidate_energy - current_energy
                if delta_energy <= 0:
                    positions = candidate
                    current_energy = candidate_energy
                    continue

                # Match simulated annealing semantics: uphill moves remain possible
                # while the move radius is positive, but become exponentially rarer.
                acceptance = torch.exp(-delta_energy / max(temperature, _MIN_DISTANCE)).clamp(
                    max=1.0
                )
                threshold = float(torch.rand((1,), generator=generator).item())
                if threshold < float(acceptance.item()):
                    positions = candidate
                    current_energy = candidate_energy

        state.pos = positions
        state.extras[_DH_CURRENT_ENERGY_KEY] = current_energy
        return state


@register_op
@dataclass(frozen=True)
class DHCool(Op):
    """Exponential schedule used by Davidson-Harel."""

    config: DHCoolConfig = field(default_factory=DHCoolConfig)

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
        """Apply geometric annealing to ``state.temperature``."""
        del problem, ctx
        assert state.temperature is not None
        state.temperature = state.temperature * self.config.cooling_factor
        return state


@register_op
@dataclass(frozen=True)
class FinalizeDHPositions(Op):
    """Center and scale final DH coordinates to deterministic extent."""

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
        """Shift to centroid and scale to the configured extent."""
        del problem, ctx

        if state.pos is None:
            raise ValueError("FinalizeDHPositions requires state.pos to be set.")

        device = state.extras[_DH_DEVICE_KEY]
        extent: float = state.extras[_DH_EXTENT_KEY]

        # Normalize around the centroid first so the final scale is independent
        # of where the last accepted annealing move left the cloud.
        centered = state.pos - state.pos.mean(dim=0, keepdim=True)
        span = centered.abs().max().clamp(min=1.0)
        state.pos = (centered * (extent / span)).to(dtype=torch.float32, device=device)
        return state
