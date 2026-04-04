"""GEM (Graph Embedder) expressed as a composable ops pipeline.

The GEM algorithm uses a Gauss-Seidel per-node update schedule with per-node
temperatures, oscillation/rotation detection, and weighted gravity.  For small
graphs (N <= 5000) the sequential update order is essential for correctness --
each node sees already-updated positions from earlier nodes in the same round.
For larger graphs a batched fallback with vectorized forces is used.

This pipeline delegates to the classic helpers so that output is bit-identical
to ``layout_gem``.
"""

from __future__ import annotations

import math
from typing import ClassVar, Optional, Tuple

import torch

from dagua.layout.ops.base import Op, Pipeline  # noqa: E402
from dagua.layout.ops.graph_utils import (
    build_undirected_adjacency,
    layout_device,
    layout_extent,
    normalize_positions,
)
from dagua.layout.ops.state import (  # noqa: E402
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)
from dagua.layout.ops.taxonomy import OpCategory  # noqa: E402

_MIN_DISTANCE = 1.0e-9
_SEQUENTIAL_NODE_LIMIT = 5_000
_FULL_REPULSION_LIMIT = 2_000
_SAMPLED_REPULSION_NEIGHBORS = 96
_NUMBER_OF_ROUNDS = 30_000
_MINIMAL_TEMPERATURE = 0.005
_INITIAL_TEMPERATURE = 12.0
_GRAVITATIONAL_CONSTANT = 1.0 / 16.0
_DESIRED_LENGTH = 20.0
_MAXIMAL_DISTURBANCE = 0.0
_ROTATION_ANGLE = math.pi / 3.0
_OSCILLATION_ANGLE = math.pi / 2.0
_ROTATION_SENSITIVITY = 0.01
_OSCILLATION_SENSITIVITY = 0.3
_ATTRACTION_FORMULA = 1
_ROTATION_SINE_THRESHOLD = math.sin((math.pi / 2.0) + (_ROTATION_ANGLE / 2.0))
_OSCILLATION_COSINE_THRESHOLD = math.cos(_OSCILLATION_ANGLE / 2.0)


def _initialize_positions(num_nodes: int, device: torch.device, seed: int) -> torch.Tensor:
    """Create deterministic random coordinates.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    device : torch.device
        Device for the result.
    seed : int
        Random seed.

    Returns
    -------
    torch.Tensor
        Initial positions with shape ``[N, 2]``.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn((num_nodes, 2), generator=generator, dtype=torch.float32).to(device)


def _ideal_distance(num_nodes: int, extent: float) -> float:
    """Estimate a fallback spacing constant for sampled repulsion.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    extent : float
        Target half-width of the final drawing box.

    Returns
    -------
    float
        Ideal pairwise spacing constant.
    """
    return max(extent / max(float(max(num_nodes, 1)) ** 0.5, 1.0), _MIN_DISTANCE)


def _degree_weights(
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute OGDF's degree-based node weights.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    device : torch.device
        Device for the returned tensor.

    Returns
    -------
    torch.Tensor
        Degree-derived weights with shape ``[N]``.
    """
    degrees = torch.zeros((num_nodes,), dtype=torch.float32, device=device)
    if edge_index.numel() > 0:
        src = edge_index[0].to(device=device, dtype=torch.long)
        dst = edge_index[1].to(device=device, dtype=torch.long)
        degrees.index_add_(0, src, torch.ones_like(src, dtype=torch.float32))
        degrees.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
    return degrees / 2.5 + 1.0


def _node_diagonals(num_nodes: int, node_sizes: Optional[torch.Tensor]) -> torch.Tensor:
    """Compute per-node diagonal lengths used by OGDF GEM.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node sizes. When the tensor has shape ``[N, 2]`` it is treated
        as ``(width, height)``; a 1D tensor is treated as square side lengths.

    Returns
    -------
    torch.Tensor
        Node diagonal lengths with shape ``[N]``.
    """
    if node_sizes is None or node_sizes.numel() == 0:
        return torch.zeros((num_nodes,), dtype=torch.float32)

    sizes_cpu = node_sizes.to(device="cpu", dtype=torch.float32)
    if sizes_cpu.ndim == 2 and sizes_cpu.shape[0] == num_nodes and sizes_cpu.shape[1] >= 2:
        widths = sizes_cpu[:, 0]
        heights = sizes_cpu[:, 1]
        return torch.sqrt(widths.square() + heights.square())
    if sizes_cpu.ndim == 1 and sizes_cpu.shape[0] == num_nodes:
        return torch.sqrt(2.0 * sizes_cpu.square())
    return torch.zeros((num_nodes,), dtype=torch.float32)


def _repulsive_force_batch(
    positions: torch.Tensor,
    node_desired_lengths: torch.Tensor,
) -> torch.Tensor:
    """Compute batched OGDF GEM repulsion with per-source desired lengths.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    node_desired_lengths : torch.Tensor
        Per-node desired lengths with shape ``[N]``.

    Returns
    -------
    torch.Tensor
        Repulsive force per node with shape ``[N, 2]``.
    """
    delta = positions.unsqueeze(1) - positions.unsqueeze(0)
    distance_square = delta.square().sum(dim=2)
    mask = distance_square > _MIN_DISTANCE
    safe_distance_square = torch.where(mask, distance_square, torch.ones_like(distance_square))
    desired_square = node_desired_lengths.square().unsqueeze(1).unsqueeze(2)
    force = delta * desired_square / safe_distance_square.unsqueeze(2)
    return (force * mask.unsqueeze(2)).sum(dim=1)


def _repulsive_force_sampled(
    positions: torch.Tensor,
    seed: int,
    step: int,
    ideal_distance: float,
) -> torch.Tensor:
    """Approximate repulsion by sampled random neighbors.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    seed : int
        Base random seed.
    step : int
        Iteration index.
    ideal_distance : float
        Scalar desired length used for the coarse fallback.

    Returns
    -------
    torch.Tensor
        Approximate repulsive force per node.
    """
    num_nodes = int(positions.shape[0])
    sample_size = min(num_nodes, _SAMPLED_REPULSION_NEIGHBORS)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + step + 1)
    sampled = torch.randint(
        0,
        num_nodes,
        (num_nodes, sample_size),
        generator=generator,
        dtype=torch.long,
    ).to(positions.device)
    neighbors = positions[sampled]
    delta = positions.unsqueeze(1) - neighbors
    distances = torch.linalg.norm(delta, dim=2).clamp(min=_MIN_DISTANCE)
    force = (ideal_distance * ideal_distance) / distances
    return (delta / distances.unsqueeze(2) * force.unsqueeze(2)).sum(dim=1)


def _attractive_force_batch(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    node_desired_lengths: torch.Tensor,
    degree_weights: torch.Tensor,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute batched OGDF GEM attraction with per-node desired lengths.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    node_desired_lengths : torch.Tensor
        Per-node desired lengths with shape ``[N]``.
    degree_weights : torch.Tensor
        Degree-based OGDF node weights with shape ``[N]``.
    edge_weights : torch.Tensor, optional
        Optional per-edge attraction weights with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Attractive force per node with shape ``[N, 2]``.
    """
    forces = torch.zeros_like(positions)
    if edge_index.numel() == 0:
        return forces

    src = edge_index[0].to(device=positions.device, dtype=torch.long)
    dst = edge_index[1].to(device=positions.device, dtype=torch.long)
    delta = positions[src] - positions[dst]
    distances = torch.linalg.norm(delta, dim=1)
    source_weights = degree_weights[src].clamp(min=1.0)
    target_weights = degree_weights[dst].clamp(min=1.0)
    source_desired = node_desired_lengths[src].clamp(min=_MIN_DISTANCE)
    target_desired = node_desired_lengths[dst].clamp(min=_MIN_DISTANCE)

    if _ATTRACTION_FORMULA == 1:
        source_force = -delta * (distances / (source_desired * source_weights)).unsqueeze(1)
        target_force = delta * (distances / (target_desired * target_weights)).unsqueeze(1)
    else:
        distance_square = distances.square()
        source_force = -delta * (
            distance_square / (source_desired.square() * source_weights)
        ).unsqueeze(1)
        target_force = delta * (
            distance_square / (target_desired.square() * target_weights)
        ).unsqueeze(1)

    if edge_weights is not None:
        edge_weight_tensor = edge_weights.to(
            device=positions.device,
            dtype=positions.dtype,
        ).unsqueeze(1)
        source_force = source_force * edge_weight_tensor
        target_force = target_force * edge_weight_tensor

    forces.index_add_(0, src, source_force)
    forces.index_add_(0, dst, target_force)
    return forces


def _gravity_force(
    positions: torch.Tensor,
    degree_weights: torch.Tensor,
) -> torch.Tensor:
    """Compute OGDF's weighted gravity toward the global barycenter.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    degree_weights : torch.Tensor
        Degree-based OGDF node weights with shape ``[N]``.

    Returns
    -------
    torch.Tensor
        Gravity impulse per node with shape ``[N, 2]``.

    Notes
    -----
    OGDF stores the weighted coordinate sum and divides by ``N`` when using it
    during the impulse computation. This is intentionally not normalized by the
    total weight sum.
    """
    barycenter = (positions * degree_weights.unsqueeze(1)).sum(dim=0, keepdim=True)
    barycenter = barycenter / max(int(positions.shape[0]), 1)
    return (barycenter - positions) * _GRAVITATIONAL_CONSTANT


def _update_temperatures(
    temperatures: torch.Tensor,
    current_impulse: torch.Tensor,
    previous_impulse: torch.Tensor,
    skew_gauge: torch.Tensor,
    initial_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adjust per-node temperatures with OGDF's oscillation and skew rules.

    Parameters
    ----------
    temperatures : torch.Tensor
        Per-node temperature tensor with shape ``[N]``.
    current_impulse : torch.Tensor
        Current scaled node movements with shape ``[N, 2]``.
    previous_impulse : torch.Tensor
        Previous scaled node movements with shape ``[N, 2]``.
    skew_gauge : torch.Tensor
        Per-node rotation accumulator with shape ``[N]``.
    initial_temperature : float
        Maximum allowed per-node temperature.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Updated temperatures and skew-gauge values, both with shape ``[N]``.
    """
    current_norm = torch.linalg.norm(current_impulse, dim=1)
    previous_norm = torch.linalg.norm(previous_impulse, dim=1)
    product = current_norm * previous_norm
    valid = product > _MIN_DISTANCE

    if not bool(valid.any()):
        return temperatures, skew_gauge

    safe_product = product.clamp(min=_MIN_DISTANCE)
    sin_beta = (
        current_impulse[:, 0] * previous_impulse[:, 0]
        - current_impulse[:, 1] * previous_impulse[:, 1]
    ) / safe_product
    cos_beta = (
        current_impulse[:, 0] * previous_impulse[:, 0]
        + current_impulse[:, 1] * previous_impulse[:, 1]
    ) / safe_product

    rotation_mask = valid & (sin_beta > _ROTATION_SINE_THRESHOLD)
    skew_gauge = torch.where(rotation_mask, skew_gauge + _ROTATION_SENSITIVITY, skew_gauge)

    oscillation_mask = valid & (cos_beta.abs() > _OSCILLATION_COSINE_THRESHOLD)
    oscillation_scale = 1.0 + cos_beta * _OSCILLATION_SENSITIVITY
    temperatures = torch.where(oscillation_mask, temperatures * oscillation_scale, temperatures)
    temperatures = temperatures * (1.0 - skew_gauge.abs())
    temperatures = torch.minimum(temperatures, torch.full_like(temperatures, initial_temperature))
    return temperatures, skew_gauge


def _compute_impulse_sequential(
    node_index: int,
    positions: torch.Tensor,
    barycenter: torch.Tensor,
    adjacency: list[list[tuple[int, float]]],
    node_desired_lengths: torch.Tensor,
    degree_weights: torch.Tensor,
) -> torch.Tensor:
    """Compute one OGDF GEM impulse with immediate-position semantics.

    Parameters
    ----------
    node_index : int
        Node index being updated.
    positions : torch.Tensor
        Current position tensor with shape ``[N, 2]`` on CPU.
    barycenter : torch.Tensor
        Weighted coordinate sum with shape ``[2]``.
    adjacency : list[list[tuple[int, float]]]
        Undirected weighted adjacency list.
    node_desired_lengths : torch.Tensor
        Per-node desired lengths with shape ``[N]``.
    degree_weights : torch.Tensor
        OGDF node weights with shape ``[N]``.

    Returns
    -------
    torch.Tensor
        Raw impulse vector with shape ``[2]``.
    """
    num_nodes = int(positions.shape[0])
    x_coord = float(positions[node_index, 0].item())
    y_coord = float(positions[node_index, 1].item())
    desired_length = float(node_desired_lengths[node_index].item())
    desired_square = desired_length * desired_length

    impulse_x = (
        float(barycenter[0].item()) / max(num_nodes, 1) - x_coord
    ) * _GRAVITATIONAL_CONSTANT
    impulse_y = (
        float(barycenter[1].item()) / max(num_nodes, 1) - y_coord
    ) * _GRAVITATIONAL_CONSTANT

    if _MAXIMAL_DISTURBANCE > 0.0:
        raise NotImplementedError("Non-zero GEM disturbance is intentionally unsupported.")

    for other_index in range(num_nodes):
        if other_index == node_index:
            continue
        delta_x = x_coord - float(positions[other_index, 0].item())
        delta_y = y_coord - float(positions[other_index, 1].item())
        distance = math.hypot(delta_x, delta_y)
        if distance > 0.0:
            distance_square = distance * distance
            impulse_x += delta_x * desired_square / distance_square
            impulse_y += delta_y * desired_square / distance_square

    node_weight = float(degree_weights[node_index].item())
    for neighbor_index, edge_weight in adjacency[node_index]:
        delta_x = x_coord - float(positions[neighbor_index, 0].item())
        delta_y = y_coord - float(positions[neighbor_index, 1].item())
        distance = math.hypot(delta_x, delta_y)
        if _ATTRACTION_FORMULA == 1:
            impulse_x -= edge_weight * delta_x * distance / (desired_length * node_weight)
            impulse_y -= edge_weight * delta_y * distance / (desired_length * node_weight)
        else:
            distance_square = distance * distance
            impulse_x -= edge_weight * delta_x * distance_square / (desired_square * node_weight)
            impulse_y -= edge_weight * delta_y * distance_square / (desired_square * node_weight)

    return torch.tensor([impulse_x, impulse_y], dtype=positions.dtype)


def _update_node_sequential(
    node_index: int,
    positions: torch.Tensor,
    impulse: torch.Tensor,
    previous_impulse: torch.Tensor,
    local_temperatures: torch.Tensor,
    skew_gauge: torch.Tensor,
    degree_weights: torch.Tensor,
    barycenter: torch.Tensor,
    global_temperature: float,
) -> float:
    """Apply one OGDF GEM node update in-place.

    Parameters
    ----------
    node_index : int
        Node index being updated.
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]`` on CPU.
    impulse : torch.Tensor
        Raw impulse vector for the node with shape ``[2]``.
    previous_impulse : torch.Tensor
        Previous saved node impulses with shape ``[N, 2]``.
    local_temperatures : torch.Tensor
        Local node temperatures with shape ``[N]``.
    skew_gauge : torch.Tensor
        Per-node skew-gauge values with shape ``[N]``.
    degree_weights : torch.Tensor
        OGDF node weights with shape ``[N]``.
    barycenter : torch.Tensor
        Weighted coordinate sum with shape ``[2]``.
    global_temperature : float
        Current OGDF global temperature.

    Returns
    -------
    float
        Updated global temperature.
    """
    num_nodes = int(positions.shape[0])
    raw_x = float(impulse[0].item())
    raw_y = float(impulse[1].item())
    impulse_length = math.hypot(raw_x, raw_y)
    if impulse_length <= 0.0:
        return global_temperature

    local_temperature = float(local_temperatures[node_index].item())
    move_x = raw_x * local_temperature / impulse_length
    move_y = raw_y * local_temperature / impulse_length

    positions[node_index, 0] += move_x
    positions[node_index, 1] += move_y

    node_weight = float(degree_weights[node_index].item())
    barycenter[0] += node_weight * move_x
    barycenter[1] += node_weight * move_y

    old_x = float(previous_impulse[node_index, 0].item())
    old_y = float(previous_impulse[node_index, 1].item())
    product = math.hypot(move_x, move_y) * math.hypot(old_x, old_y)
    if product > 0.0:
        global_temperature -= local_temperature / max(num_nodes, 1)

        sin_beta = (move_x * old_x - move_y * old_y) / product
        cos_beta = (move_x * old_x + move_y * old_y) / product

        skew_value = float(skew_gauge[node_index].item())
        if sin_beta > _ROTATION_SINE_THRESHOLD:
            skew_value += _ROTATION_SENSITIVITY

        if abs(cos_beta) > _OSCILLATION_COSINE_THRESHOLD:
            local_temperature *= 1.0 + (cos_beta * _OSCILLATION_SENSITIVITY)

        local_temperature *= 1.0 - abs(skew_value)
        if local_temperature >= _INITIAL_TEMPERATURE:
            local_temperature = _INITIAL_TEMPERATURE

        skew_gauge[node_index] = skew_value
        local_temperatures[node_index] = local_temperature
        global_temperature += local_temperature / max(num_nodes, 1)

    previous_impulse[node_index, 0] = move_x
    previous_impulse[node_index, 1] = move_y
    return global_temperature


class _InitializeGEMPositions(Op):
    """Initialize positions exactly like classic GEM."""

    name: ClassVar[str] = "gem_initialize_positions"
    category: ClassVar[OpCategory] = OpCategory.INIT
    writes: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Seed ``state.pos`` from torch randn, matching classic GEM init.

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
            State with ``state.pos`` populated as a CPU ``float32`` tensor,
            or early-exit positions for trivial graphs.
        """
        del ctx

        device = layout_device(problem.edge_index, problem.node_sizes)
        if problem.num_nodes == 0:
            state.pos = torch.empty((0, 2), dtype=torch.float32, device=device)
            state.converged = True
            return state
        if problem.num_nodes == 1:
            state.pos = torch.zeros((1, 2), dtype=torch.float32, device=device)
            state.converged = True
            return state

        state.pos = _initialize_positions(problem.num_nodes, device, problem.seed)
        return state


class _PrepareGEMState(Op):
    """Populate GEM-specific cached state required by the solve step."""

    name: ClassVar[str] = "gem_prepare_state"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build GEM-specific auxiliary data structures.

        Stores degree weights, node desired lengths, adjacency lists,
        initial temperatures, extent, and capped iteration count into
        ``state.extras`` for downstream ops.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state receiving GEM extras.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with GEM auxiliary data stored in ``extras``.
        """
        del ctx

        if state.converged:
            return state

        device = layout_device(problem.edge_index, problem.node_sizes)
        extent = layout_extent(problem.num_nodes, problem.node_sizes)
        capped_iters = min(state.total_steps, _NUMBER_OF_ROUNDS)

        state.extras["gem_extent"] = extent
        state.extras["gem_capped_iters"] = capped_iters
        state.extras["gem_device"] = device
        state.extras["gem_is_sequential"] = problem.num_nodes <= _SEQUENTIAL_NODE_LIMIT
        return state


class _GEMSequentialSolve(Op):
    """Run the OGDF-like sequential Gauss-Seidel GEM loop.

    This op wraps the full sequential iteration because the Gauss-Seidel
    update order is not decomposable into separate force/apply ops -- each
    node sees already-updated positions of previously processed nodes.
    """

    name: ClassVar[str] = "gem_sequential_solve"
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
        """Execute the sequential GEM loop, identical to classic.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with initialized positions.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with positions updated by the sequential GEM solver.
        """
        del ctx

        if state.converged or not state.extras.get("gem_is_sequential", False):
            return state

        num_nodes = problem.num_nodes
        capped_iters = state.extras["gem_capped_iters"]

        positions = state.pos.to(device="cpu", dtype=torch.float64)
        degree_wts = _degree_weights(
            problem.edge_index,
            num_nodes,
            torch.device("cpu"),
        ).to(dtype=torch.float64)
        node_desired_lengths = (
            _node_diagonals(num_nodes, problem.node_sizes).to(dtype=torch.float64) + _DESIRED_LENGTH
        )
        adjacency = build_undirected_adjacency(
            problem.edge_index, num_nodes, edge_weights=problem.edge_weights
        )
        local_temperatures = torch.full((num_nodes,), _INITIAL_TEMPERATURE, dtype=torch.float64)
        previous_impulse = torch.zeros((num_nodes, 2), dtype=torch.float64)
        skew_gauge = torch.zeros((num_nodes,), dtype=torch.float64)
        barycenter = (positions * degree_wts.unsqueeze(1)).sum(dim=0)
        global_temperature = _INITIAL_TEMPERATURE

        generator = torch.Generator(device="cpu")
        generator.manual_seed(problem.seed)
        permutation: list[int] = []
        rounds_remaining = capped_iters

        while global_temperature > _MINIMAL_TEMPERATURE and rounds_remaining > 0:
            if not permutation:
                permutation = torch.randperm(num_nodes, generator=generator).tolist()
            node_index = permutation.pop()
            impulse = _compute_impulse_sequential(
                node_index=node_index,
                positions=positions,
                barycenter=barycenter,
                adjacency=adjacency,
                node_desired_lengths=node_desired_lengths,
                degree_weights=degree_wts,
            )
            global_temperature = _update_node_sequential(
                node_index=node_index,
                positions=positions,
                impulse=impulse,
                previous_impulse=previous_impulse,
                local_temperatures=local_temperatures,
                skew_gauge=skew_gauge,
                degree_weights=degree_wts,
                barycenter=barycenter,
                global_temperature=global_temperature,
            )
            rounds_remaining -= 1

        state.pos = positions.to(dtype=torch.float32)
        return state


class _GEMBatchedSolve(Op):
    """Run the vectorized batched GEM loop for large graphs.

    For graphs with N > 5000 the batched fallback uses vectorized
    repulsion, attraction, and gravity forces with per-node temperature
    adaptation identical to the classic batched path.
    """

    name: ClassVar[str] = "gem_batched_solve"
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
        """Execute the batched GEM loop, identical to classic.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with initialized positions.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with positions updated by the batched GEM solver.
        """
        del ctx

        if state.converged or state.extras.get("gem_is_sequential", True):
            return state

        num_nodes = problem.num_nodes
        device = state.extras["gem_device"]
        extent = state.extras["gem_extent"]
        capped_iters = state.extras["gem_capped_iters"]

        positions = state.pos
        degree_wts = _degree_weights(problem.edge_index, num_nodes, device)
        node_desired_lengths = (
            _node_diagonals(num_nodes, problem.node_sizes) + _DESIRED_LENGTH
        ).to(device=device)
        temperatures = torch.full(
            (num_nodes,),
            fill_value=_INITIAL_TEMPERATURE,
            dtype=torch.float32,
            device=device,
        )
        previous_impulse = torch.zeros_like(positions)
        skew_gauge = torch.zeros((num_nodes,), dtype=torch.float32, device=device)
        sampled_ideal_distance = _ideal_distance(num_nodes, extent)

        for step in range(capped_iters):
            if float(temperatures.mean().item()) < _MINIMAL_TEMPERATURE:
                break

            if num_nodes > _FULL_REPULSION_LIMIT:
                repulsive = _repulsive_force_sampled(
                    positions, problem.seed, step, sampled_ideal_distance
                )
            else:
                repulsive = _repulsive_force_batch(positions, node_desired_lengths)
            attractive = _attractive_force_batch(
                positions=positions,
                edge_index=problem.edge_index,
                node_desired_lengths=node_desired_lengths,
                degree_weights=degree_wts,
                edge_weights=problem.edge_weights,
            )
            gravity = _gravity_force(positions, degree_wts)
            impulse = repulsive + attractive + gravity
            norm = torch.linalg.norm(impulse, dim=1, keepdim=True)
            safe_norm = norm.clamp(min=_MIN_DISTANCE)
            movement = torch.where(
                norm > 0,
                impulse * (temperatures.unsqueeze(1) / safe_norm),
                torch.zeros_like(impulse),
            )

            temperatures, skew_gauge = _update_temperatures(
                temperatures=temperatures,
                current_impulse=movement,
                previous_impulse=previous_impulse,
                skew_gauge=skew_gauge,
                initial_temperature=_INITIAL_TEMPERATURE,
            )

            positions = positions + movement
            previous_impulse = movement

        state.pos = positions
        return state


class _FinalizeGEMPositions(Op):
    """Apply classic GEM's final normalize, rescale, and cast."""

    name: ClassVar[str] = "gem_finalize_positions"
    category: ClassVar[OpCategory] = OpCategory.POSTPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("pos", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Center, normalize, scale, and cast positions like classic GEM.

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
            State with final output positions on the classic return device.

        Raises
        ------
        ValueError
            If ``state.pos`` is missing.
        """
        del ctx

        if state.pos is None:
            raise ValueError("_FinalizeGEMPositions requires state.pos to be set.")

        if state.converged:
            return state

        device = state.extras["gem_device"]
        extent = state.extras["gem_extent"]

        state.pos = normalize_positions(state.pos, extent).to(
            dtype=torch.float32,
            device=device,
        )
        return state


def build_gem_pipeline(max_iters: int = 500) -> Pipeline:
    """Build a GEM pipeline that is bit-identical to classic ``layout_gem``.

    Parameters
    ----------
    max_iters : int, default=500
        Maximum number of OGDF node updates.

    Returns
    -------
    Pipeline
        Pipeline that reproduces classic GEM's initialization, sequential
        or batched solve, and postprocessing.

    Raises
    ------
    ValueError
        If ``max_iters`` is negative.
    """
    if max_iters < 0:
        raise ValueError("max_iters must be non-negative.")

    from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig

    return Pipeline(
        [
            FixedSteps(FixedStepsConfig(n=max_iters)),
            _InitializeGEMPositions(),
            _PrepareGEMState(),
            _GEMSequentialSolve(),
            _GEMBatchedSolve(),
            _FinalizeGEMPositions(),
        ],
        name="gem_pipeline",
    )


def layout_gem_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    max_iters: int = 500,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the GEM pipeline as a drop-in replacement for classic ``layout_gem``.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor used for scaling.
    max_iters : int, default=500
        Maximum number of OGDF node updates.
    seed : int, default=42
        Random seed for initialization and node permutations.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Final position tensor with the same dtype, device, and values as
        classic ``layout_gem``.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``max_iters``, or ``edge_weights`` are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if max_iters < 0:
        raise ValueError("max_iters must be non-negative.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
            )

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_gem_pipeline(max_iters=max_iters).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("GEM pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_gem_pipeline", "layout_gem_pipeline"]
