"""Davidson-Harel ops for the composable pipeline implementation."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from dagua.layout.ops.base import Op
from dagua.layout.ops.graph_utils import layout_device as _layout_device
from dagua.layout.ops.graph_utils import layout_extent as _layout_extent
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_MIN_DISTANCE = 1.0e-3
_BORDER_WEIGHT = 0.1
_EDGE_LENGTH_WEIGHT = 0.2
_CROSSING_WEIGHT = 2.0
_NODE_EDGE_WEIGHT = 0.5
_COOLING_FACTOR = 0.75
_COLLINEAR_EPSILON = 1.0e-10

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


def _scale_denominator(numerator_count: int) -> float:
    """Return safe integer scaling denominator."""
    return float(max(numerator_count, 1))


def _energy(
    positions: torch.Tensor,
    edges: List[Tuple[int, int]],
    extent: float,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Evaluate the Davidson-Harel objective for one candidate layout."""
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

    edge_count = len(edges)
    distribution_scale = _scale_denominator(num_nodes * max(num_nodes - 1, 1) // 2)
    border_scale = _scale_denominator(num_nodes)
    edge_length_scale = _scale_denominator(edge_count)
    crossing_scale = _scale_denominator(edge_count * edge_count)
    node_edge_scale = _scale_denominator(num_nodes * edge_count)

    return (
        distribution / distribution_scale
        + _BORDER_WEIGHT * (border / border_scale)
        + _EDGE_LENGTH_WEIGHT * (edge_length / edge_length_scale)
        + _CROSSING_WEIGHT * (crossing_energy / crossing_scale)
        + _NODE_EDGE_WEIGHT * (node_edge / node_edge_scale)
    )


@register_op
class InitializeDHPositions(Op):
    """Initialize random coordinates and cached layout extent metadata."""

    name = "dh_initialize_positions"
    category = OpCategory.INIT
    writes = ("pos",)

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
class PrepareDHState(Op):
    """Build cached, undirected edge state and starting energy."""

    name = "dh_prepare_state"
    category = OpCategory.PREPROCESS
    reads = ("pos",)
    writes = ("extras",)

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

        initial_temperature = max(0.1 * float(current_energy.item()), _MIN_DISTANCE)
        state.extras[_DH_INITIAL_TEMPERATURE_KEY] = initial_temperature
        state.temperature = initial_temperature

        generator = torch.Generator(device="cpu")
        generator.manual_seed(problem.seed)
        state.extras[_DH_GENERATOR_KEY] = generator

        return state


@register_op
class DHAnnealingRound(Op):
    """Apply one Davidson-Harel annealing round."""

    name = "dh_annealing_round"
    category = OpCategory.FORCE
    reads = ("pos", "extras")
    writes = ("pos", "extras")
    requires = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Propose and accept/reject one move per node."""
        del ctx

        assert state.pos is not None

        positions = state.pos
        edges: List[Tuple[int, int]] = state.extras[_DH_EDGES_KEY]
        unique_edge_weights: torch.Tensor = state.extras[_DH_UNIQUE_EDGE_WEIGHTS_KEY]
        extent: float = state.extras[_DH_EXTENT_KEY]
        current_energy: torch.Tensor = state.extras[_DH_CURRENT_ENERGY_KEY]
        initial_temperature: float = state.extras[_DH_INITIAL_TEMPERATURE_KEY]
        temperature: float = state.temperature
        generator: torch.Generator = state.extras[_DH_GENERATOR_KEY]
        device: torch.device = state.extras[_DH_DEVICE_KEY]

        moves_per_round = problem.num_nodes

        for _ in range(moves_per_round):
            node = int(torch.randint(0, problem.num_nodes, (1,), generator=generator).item())
            move_scale = 0.25 * extent * (temperature / max(initial_temperature, _MIN_DISTANCE))
            delta = ((torch.rand((2,), generator=generator) * 2.0) - 1.0).to(device) * move_scale
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

            acceptance = torch.exp(-delta_energy / max(temperature, _MIN_DISTANCE)).clamp(max=1.0)
            threshold = float(torch.rand((1,), generator=generator).item())
            if threshold < float(acceptance.item()):
                positions = candidate
                current_energy = candidate_energy

        state.pos = positions
        state.extras[_DH_CURRENT_ENERGY_KEY] = current_energy
        return state


@register_op
class DHCool(Op):
    """Exponential schedule used by Davidson-Harel."""

    name = "dh_cool"
    category = OpCategory.ANNEAL
    reads = ()
    writes = ()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Apply geometric annealing to ``state.temperature``."""
        del problem, ctx
        assert state.temperature is not None
        state.temperature = state.temperature * _COOLING_FACTOR
        return state


@register_op
class FinalizeDHPositions(Op):
    """Center and scale final DH coordinates to deterministic extent."""

    name = "dh_finalize_positions"
    category = OpCategory.POSTPROCESS
    reads = ("pos",)
    writes = ("pos",)
    requires = ("pos",)

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

        centered = state.pos - state.pos.mean(dim=0, keepdim=True)
        span = centered.abs().max().clamp(min=1.0)
        state.pos = (centered * (extent / span)).to(dtype=torch.float32, device=device)
        return state
