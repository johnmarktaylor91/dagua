"""Registered operations implementing the classic GEM primitives.

These ops are intentionally faithful to ``layout.classic.gem`` and provide a
composable, registered building block surface for pipeline definitions.
"""

from __future__ import annotations

import math
from typing import ClassVar, Tuple

import torch

from dagua.layout.ops.base import Op
from dagua.layout.ops.graph_utils import (
    build_undirected_adjacency,
    layout_device,
    layout_extent,
    normalize_positions,
)
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

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


@register_op
class InitializeGEMPositions(Op):
    """Initialize positions for GEM with the classic random generator seed."""

    name: ClassVar[str] = "gem_initialize_positions"
    category: ClassVar[OpCategory] = OpCategory.INIT
    writes: ClassVar[Tuple[str, ...]] = ("pos", "converged")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Seed deterministic normal positions and handle trivial graphs.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs including seed and size.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context carrying device hints.

        Returns
        -------
        SolveState
            State with ``state.pos`` populated and ``state.converged`` for
            empty or singleton graphs.
        """
        del ctx

        state.converged = False
        output_device = layout_device(problem.edge_index, problem.node_sizes)

        if problem.num_nodes == 0:
            state.pos = torch.empty((0, 2), dtype=torch.float32, device=output_device)
            state.converged = True
            return state

        if problem.num_nodes == 1:
            state.pos = torch.zeros((1, 2), dtype=torch.float32, device=output_device)
            state.converged = True
            return state

        generator = torch.Generator(device="cpu")
        generator.manual_seed(problem.seed)
        state.pos = torch.randn(
            (problem.num_nodes, 2),
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        ).to(device=output_device)
        return state


@register_op
class GEMPrepareState(Op):
    """Prepare cached metadata consumed by GEM sequential and batched branches."""

    name: ClassVar[str] = "gem_prepare_state"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Populate GEM metadata in ``state.extras``.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context.

        Returns
        -------
        SolveState
            State with GEM geometry metadata.
        """
        del ctx

        if state.converged:
            return state

        extent = layout_extent(problem.num_nodes, problem.node_sizes)
        capped_iters = min(int(state.total_steps), _NUMBER_OF_ROUNDS)
        state.extras["gem_extent"] = extent
        state.extras["gem_capped_iters"] = capped_iters
        state.extras["gem_device"] = layout_device(problem.edge_index, problem.node_sizes)
        state.extras["gem_is_sequential"] = problem.num_nodes <= _SEQUENTIAL_NODE_LIMIT
        return state


@register_op
class GEMSequentialSolve(Op):
    """Run the OGDF-like sequential Gauss-Seidel GEM loop on CPU."""

    name: ClassVar[str] = "gem_sequential_solve"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = (
        "pos",
        "extras",
    )
    writes: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run the exact sequential solver path used for ``N <= 5000``.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context (unused).

        Returns
        -------
        SolveState
            Updated state containing post-solve CPU ``float32`` positions.
        """
        del ctx

        if state.converged or not state.extras.get("gem_is_sequential", False):
            return state

        num_nodes = problem.num_nodes
        capped_iters = int(state.extras.get("gem_capped_iters", 0))

        positions = state.pos.to(dtype=torch.float64, device="cpu")

        degree_weights = torch.zeros((num_nodes,), dtype=torch.float32, device="cpu")
        if problem.edge_index.numel() > 0:
            src = problem.edge_index[0].to(device="cpu", dtype=torch.long)
            dst = problem.edge_index[1].to(device="cpu", dtype=torch.long)
            ones = torch.ones_like(src, dtype=torch.float32)
            degree_weights.index_add_(0, src, ones)
            degree_weights.index_add_(0, dst, ones)
        degree_weights = degree_weights / 2.5 + 1.0

        node_sizes_cpu = (
            torch.zeros((num_nodes,), dtype=torch.float32)
            if problem.node_sizes is None or problem.node_sizes.numel() == 0
            else problem.node_sizes.to(dtype=torch.float32, device="cpu")
        )
        if (
            node_sizes_cpu.ndim == 2
            and node_sizes_cpu.shape[0] == num_nodes
            and node_sizes_cpu.shape[1] >= 2
        ):
            node_widths = node_sizes_cpu[:, 0]
            node_heights = node_sizes_cpu[:, 1]
            node_desired_lengths = torch.sqrt(node_widths.square() + node_heights.square())
        elif node_sizes_cpu.ndim == 1 and node_sizes_cpu.shape[0] == num_nodes:
            node_desired_lengths = torch.sqrt(2.0 * node_sizes_cpu.square())
        else:
            node_desired_lengths = torch.zeros((num_nodes,), dtype=torch.float32)
        node_desired_lengths = node_desired_lengths.to(dtype=torch.float64) + _DESIRED_LENGTH

        adjacency = build_undirected_adjacency(
            problem.edge_index,
            num_nodes,
            edge_weights=problem.edge_weights,
        )

        local_temperatures = torch.full((num_nodes,), _INITIAL_TEMPERATURE, dtype=torch.float64)
        previous_impulse = torch.zeros((num_nodes, 2), dtype=torch.float64)
        skew_gauge = torch.zeros((num_nodes,), dtype=torch.float64)
        barycenter = (positions * degree_weights.unsqueeze(1)).sum(dim=0)
        global_temperature = _INITIAL_TEMPERATURE
        generator = torch.Generator(device="cpu")
        generator.manual_seed(problem.seed)
        permutation: list[int] = []

        if not (_MAXIMAL_DISTURBANCE == 0.0):
            raise NotImplementedError("Non-zero GEM disturbance is intentionally unsupported.")

        rounds_remaining = capped_iters
        while global_temperature > _MINIMAL_TEMPERATURE and rounds_remaining > 0:
            if not permutation:
                permutation = torch.randperm(num_nodes, generator=generator).tolist()
            node_index = int(permutation.pop())

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
                    if distance > 0.0:
                        impulse_x -= (
                            edge_weight * delta_x * distance / (desired_length * node_weight)
                        )
                        impulse_y -= (
                            edge_weight * delta_y * distance / (desired_length * node_weight)
                        )
                else:
                    distance_square = distance * distance
                    if distance_square > 0.0:
                        impulse_x -= (
                            edge_weight * delta_x * distance_square / (desired_square * node_weight)
                        )
                        impulse_y -= (
                            edge_weight * delta_y * distance_square / (desired_square * node_weight)
                        )

            raw_x = impulse_x
            raw_y = impulse_y
            impulse_length = math.hypot(raw_x, raw_y)

            if impulse_length > 0.0:
                local_temperature = float(local_temperatures[node_index].item())
                move_x = raw_x * local_temperature / impulse_length
                move_y = raw_y * local_temperature / impulse_length

                positions[node_index, 0] += move_x
                positions[node_index, 1] += move_y

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

            rounds_remaining -= 1

        state.pos = positions.to(dtype=torch.float32)
        return state


@register_op
class GEMBatchedSolve(Op):
    """Run the vectorized batched GEM solver for larger graphs."""

    name: ClassVar[str] = "gem_batched_solve"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = (
        "pos",
        "extras",
    )
    writes: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run the exact batched GEM fallback used when ``N > 5000``.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context (unused).

        Returns
        -------
        SolveState
            Updated state with batched GEM positions.
        """
        del ctx

        if state.converged or state.extras.get("gem_is_sequential", True):
            return state

        num_nodes = problem.num_nodes
        extent = float(state.extras["gem_extent"])
        capped_iters = int(state.extras.get("gem_capped_iters", 0))
        positions = state.pos

        degree_weights = torch.zeros((num_nodes,), dtype=torch.float32, device=positions.device)
        if problem.edge_index.numel() > 0:
            src = problem.edge_index[0].to(device=positions.device, dtype=torch.long)
            dst = problem.edge_index[1].to(device=positions.device, dtype=torch.long)
            ones = torch.ones_like(src, dtype=torch.float32)
            degree_weights.index_add_(0, src, ones)
            degree_weights.index_add_(0, dst, ones)
        degree_weights = degree_weights / 2.5 + 1.0

        node_sizes = (
            torch.zeros((num_nodes,), dtype=torch.float32, device=positions.device)
            if problem.node_sizes is None or problem.node_sizes.numel() == 0
            else problem.node_sizes.to(dtype=torch.float32, device=positions.device)
        )
        if node_sizes.ndim == 2 and node_sizes.shape[0] == num_nodes and node_sizes.shape[1] >= 2:
            node_desired_lengths = torch.sqrt(node_sizes[:, 0].square() + node_sizes[:, 1].square())
        elif node_sizes.ndim == 1 and node_sizes.shape[0] == num_nodes:
            node_desired_lengths = torch.sqrt(2.0 * node_sizes.square())
        else:
            node_desired_lengths = torch.zeros(
                (num_nodes,), dtype=torch.float32, device=positions.device
            )
        node_desired_lengths = node_desired_lengths + _DESIRED_LENGTH

        temperatures = torch.full(
            (num_nodes,),
            fill_value=_INITIAL_TEMPERATURE,
            dtype=torch.float32,
            device=positions.device,
        )
        previous_impulse = torch.zeros_like(positions)
        skew_gauge = torch.zeros((num_nodes,), dtype=torch.float32, device=positions.device)
        sampled_ideal_distance = max(
            extent / max(float(max(num_nodes, 1)) ** 0.5, 1.0),
            _MIN_DISTANCE,
        )

        edges = problem.edge_index
        if edges.numel() == 0:
            edges = torch.empty((2, 0), dtype=torch.long, device=positions.device)

        for step in range(capped_iters):
            if float(temperatures.mean().item()) < _MINIMAL_TEMPERATURE:
                break

            if num_nodes > _FULL_REPULSION_LIMIT:
                sample_size = min(num_nodes, _SAMPLED_REPULSION_NEIGHBORS)
                generator = torch.Generator(device="cpu")
                generator.manual_seed(problem.seed + step + 1)
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
                force = (sampled_ideal_distance * sampled_ideal_distance) / distances
                repulsive = (delta / distances.unsqueeze(2) * force.unsqueeze(2)).sum(dim=1)
            else:
                delta = positions.unsqueeze(1) - positions.unsqueeze(0)
                distance_square = delta.square().sum(dim=2)
                mask = distance_square > _MIN_DISTANCE
                safe_distance_square = torch.where(
                    mask, distance_square, torch.ones_like(distance_square)
                )
                desired_square = node_desired_lengths.square().unsqueeze(1).unsqueeze(2)
                repulsive = delta * desired_square / safe_distance_square.unsqueeze(2)
                repulsive = (repulsive * mask.unsqueeze(2)).sum(dim=1)

            if edges.numel() == 0:
                attractive = torch.zeros_like(positions)
            else:
                attractive = torch.zeros_like(positions)
                src = edges[0].to(device=positions.device, dtype=torch.long)
                dst = edges[1].to(device=positions.device, dtype=torch.long)
                delta = positions[src] - positions[dst]
                distances = torch.linalg.norm(delta, dim=1)
                source_weights = degree_weights[src].clamp(min=1.0)
                target_weights = degree_weights[dst].clamp(min=1.0)
                source_desired = node_desired_lengths[src].clamp(min=_MIN_DISTANCE)
                target_desired = node_desired_lengths[dst].clamp(min=_MIN_DISTANCE)

                source_force = -delta * (distances / (source_desired * source_weights)).unsqueeze(1)
                target_force = delta * (distances / (target_desired * target_weights)).unsqueeze(1)
                if problem.edge_weights is not None:
                    edge_weight_tensor = problem.edge_weights.to(
                        device=positions.device,
                        dtype=positions.dtype,
                    ).unsqueeze(1)
                    source_force = source_force * edge_weight_tensor
                    target_force = target_force * edge_weight_tensor

                attractive.index_add_(0, src, source_force)
                attractive.index_add_(0, dst, target_force)

            barycenter = (positions * degree_weights.unsqueeze(1)).sum(dim=0, keepdim=True)
            barycenter = barycenter / max(num_nodes, 1)
            gravity = (barycenter - positions) * _GRAVITATIONAL_CONSTANT

            impulse = repulsive + attractive + gravity
            norm = torch.linalg.norm(impulse, dim=1, keepdim=True)
            safe_norm = norm.clamp(min=_MIN_DISTANCE)
            movement = torch.where(
                norm > 0,
                impulse * (temperatures.unsqueeze(1) / safe_norm),
                torch.zeros_like(impulse),
            )

            current_norm = torch.linalg.norm(movement, dim=1)
            previous_norm = torch.linalg.norm(previous_impulse, dim=1)
            product = current_norm * previous_norm
            valid = product > _MIN_DISTANCE
            if bool(valid.any()):
                safe_product = product.clamp(min=_MIN_DISTANCE)
                sin_beta = (
                    movement[:, 0] * previous_impulse[:, 0]
                    - movement[:, 1] * previous_impulse[:, 1]
                ) / safe_product
                cos_beta = (
                    movement[:, 0] * previous_impulse[:, 0]
                    + movement[:, 1] * previous_impulse[:, 1]
                ) / safe_product
                rotation_mask = valid & (sin_beta > _ROTATION_SINE_THRESHOLD)
                skew_gauge = torch.where(
                    rotation_mask,
                    skew_gauge + _ROTATION_SENSITIVITY,
                    skew_gauge,
                )
                oscillation_mask = valid & (cos_beta.abs() > _OSCILLATION_COSINE_THRESHOLD)
                oscillation_scale = 1.0 + cos_beta * _OSCILLATION_SENSITIVITY
                temperatures = torch.where(
                    oscillation_mask,
                    temperatures * oscillation_scale,
                    temperatures,
                )
                temperatures = temperatures * (1.0 - skew_gauge.abs())
                temperatures = torch.minimum(
                    temperatures,
                    torch.full_like(temperatures, _INITIAL_TEMPERATURE),
                )

            positions = positions + movement
            previous_impulse = movement

        state.pos = positions
        return state


@register_op
class GEMFinalizePositions(Op):
    """Normalize and cast GEM results to the final output device."""

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
        """Apply final normalization and cast to final dtype/device.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs (used only for device fallback).
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context (unused).

        Returns
        -------
        SolveState
            State with final float32 coordinates.

        Raises
        ------
        ValueError
            If ``state.pos`` is missing.
        """
        del problem, ctx

        if state.pos is None:
            raise ValueError("GEMFinalizePositions requires state.pos to be set.")

        if state.converged:
            return state

        device = state.extras.get("gem_device")
        if not isinstance(device, torch.device):
            device = layout_device(torch.empty((2, 0), dtype=torch.long), None)

        extent = float(state.extras.get("gem_extent", 1.0))
        state.pos = normalize_positions(state.pos, extent).to(dtype=torch.float32, device=device)
        return state
