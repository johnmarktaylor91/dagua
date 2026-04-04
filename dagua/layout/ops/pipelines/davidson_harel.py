"""Davidson-Harel simulated annealing expressed as a composable ops pipeline."""

from __future__ import annotations

from typing import ClassVar, List, Optional, Tuple

import torch

from dagua.layout.ops.graph_utils import (
    layout_device as _layout_device,
)
from dagua.layout.ops.graph_utils import (
    layout_extent as _layout_extent,
)

# ---------------------------------------------------------------------------
# Algorithm-specific constants and functions copied from
# dagua/layout/classic/davidson_harel.py (bit-identical)
# ---------------------------------------------------------------------------

_MIN_DISTANCE = 1.0e-3
_BORDER_WEIGHT = 0.1
_EDGE_LENGTH_WEIGHT = 0.2
_CROSSING_WEIGHT = 2.0
_NODE_EDGE_WEIGHT = 0.5
_COOLING_FACTOR = 0.75
_COLLINEAR_EPSILON = 1.0e-10


def _unique_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor] = None,
) -> tuple[list[tuple[int, int]], torch.Tensor]:
    """Convert an edge tensor into unique undirected edges and weights.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    edge_weights : torch.Tensor, optional
        Optional per-edge weights with shape ``[E]``.

    Returns
    -------
    tuple[list[tuple[int, int]], torch.Tensor]
        Unique undirected edges and their aggregated weights with shape
        ``[E_unique]``. Parallel or mirrored edges are summed so the collapsed
        undirected energy term preserves total attraction strength.
    """
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
    """Create deterministic random coordinates inside the drawing box.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    extent : float
        Half-width of the drawing box.
    device : torch.device
        Device for the result.
    seed : int
        Random seed.

    Returns
    -------
    torch.Tensor
        Initial coordinates with shape ``[N, 2]``.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return ((torch.rand((num_nodes, 2), generator=generator) * 2.0) - 1.0).to(device) * extent


def _orientation(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> float:
    """Compute the signed triangle area used by segment intersection tests.

    Parameters
    ----------
    a : torch.Tensor
        First point with shape ``[2]``.
    b : torch.Tensor
        Second point with shape ``[2]``.
    c : torch.Tensor
        Third point with shape ``[2]``.

    Returns
    -------
    float
        Signed cross product value.
    """
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _segments_intersect(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> bool:
    """Test whether two line segments intersect.

    Parameters
    ----------
    a : torch.Tensor
        First endpoint of the first segment.
    b : torch.Tensor
        Second endpoint of the first segment.
    c : torch.Tensor
        First endpoint of the second segment.
    d : torch.Tensor
        Second endpoint of the second segment.

    Returns
    -------
    bool
        ``True`` if the segments intersect.
    """
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    return (abs(o1) < _COLLINEAR_EPSILON or abs(o2) < _COLLINEAR_EPSILON or o1 * o2 < 0.0) and (
        abs(o3) < _COLLINEAR_EPSILON or abs(o4) < _COLLINEAR_EPSILON or o3 * o4 < 0.0
    )


def _point_segment_distance(
    point: torch.Tensor, start: torch.Tensor, end: torch.Tensor
) -> torch.Tensor:
    """Compute the Euclidean distance from a point to a segment.

    Parameters
    ----------
    point : torch.Tensor
        Point with shape ``[2]``.
    start : torch.Tensor
        Segment start point.
    end : torch.Tensor
        Segment end point.

    Returns
    -------
    torch.Tensor
        Distance scalar.
    """
    segment = end - start
    denom = segment.dot(segment).clamp(min=_MIN_DISTANCE)
    projection = ((point - start).dot(segment) / denom).clamp(0.0, 1.0)
    nearest = start + projection * segment
    return torch.linalg.norm(point - nearest)


def _scale_denominator(numerator_count: int) -> float:
    """Return a non-zero normalization denominator for one energy term.

    Parameters
    ----------
    numerator_count : int
        Expected scale factor for the corresponding summed energy term.

    Returns
    -------
    float
        Positive normalization denominator.
    """
    return float(max(numerator_count, 1))


def _energy(
    positions: torch.Tensor,
    edges: list[tuple[int, int]],
    extent: float,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Evaluate the Davidson-Harel layout energy.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edges : list[tuple[int, int]]
        Unique undirected edges.
    extent : float
        Half-width of the drawing box.
    edge_weights : torch.Tensor, optional
        Optional edge weights aligned with ``edges`` and shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Scalar energy value.

    Notes
    -----
    The paper defines the individual energy terms as sums. This implementation
    keeps that formulation, then normalizes each term by its natural graph-size
    scale so the fixed weights remain comparable across different graph sizes.
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


from dagua.layout.ops.base import Op, Pipeline, Repeat  # noqa: E402
from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig  # noqa: E402
from dagua.layout.ops.state import (  # noqa: E402
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)
from dagua.layout.ops.taxonomy import OpCategory  # noqa: E402


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
