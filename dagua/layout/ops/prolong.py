"""Multilevel prolongation operations for composable layout pipelines."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch

from dagua.layout._archive.classic.fmmm import (
    _TYPE_MOON,
    _TYPE_PLANET,
    _TYPE_SUN,
)
from dagua.layout.ops.base import Op
from dagua.layout.ops.coarsen import SolarHierarchyStep
from dagua.layout.ops.state import HierarchyLevel, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_SOLAR_STEPS_KEY = "solar_system_steps"
_MIN_DISTANCE = 1.0e-3


@dataclass(frozen=True)
class DirectMappingConfig:
    """Configuration for :class:`DirectMapping`.

    Parameters
    ----------
    jitter_scale : float, default=5.0
        Standard deviation of the Gaussian offset added after coarse-position
        copying.
    """

    jitter_scale: float = 5.0


@dataclass(frozen=True)
class LambdaInterpolationConfig:
    """Configuration for :class:`LambdaInterpolation`.

    Parameters
    ----------
    waggle_factor : float, default=0.05
        Fraction of the source-target distance used as the maximum random
        waggle radius for interpolation candidates.
    """

    waggle_factor: float = 0.05


@dataclass(frozen=True)
class NeighborSmoothingConfig:
    """Configuration for :class:`NeighborSmoothing`.

    Parameters
    ----------
    blend_factor : float, default=0.5
        Weight kept on the current position; ``1 - blend_factor`` is applied to
        the neighbor mean.
    """

    blend_factor: float = 0.5


def _validated_positions(state: SolveState) -> torch.Tensor:
    """Return the current position tensor after shape validation.

    Parameters
    ----------
    state : SolveState
        Mutable solve state.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If ``state.pos`` is absent or malformed.
    """
    if state.pos is None:
        raise ValueError("state.pos must be populated before this op runs")
    if state.pos.ndim != 2 or state.pos.shape[1] != 2:
        raise ValueError("state.pos must have shape [N, 2]")
    return state.pos


def _active_hierarchy_level(state: SolveState) -> Tuple[int, HierarchyLevel]:
    """Return the hierarchy transition matching the current coarse positions.

    Parameters
    ----------
    state : SolveState
        Mutable solve state carrying ``hierarchy`` and current coarse positions.

    Returns
    -------
    tuple[int, HierarchyLevel]
        Matching hierarchy index and level descriptor.

    Raises
    ------
    ValueError
        If the hierarchy is missing or no level matches the current position
        count.
    """
    positions = _validated_positions(state)
    if not state.hierarchy:
        raise ValueError("state.hierarchy must be populated before this op runs")

    coarse_node_count = int(positions.shape[0])
    for level_index in range(len(state.hierarchy) - 1, -1, -1):
        level = state.hierarchy[level_index]
        if level.num_nodes == coarse_node_count:
            return level_index, level
    raise ValueError(f"state.hierarchy does not contain a level with num_nodes={coarse_node_count}")


def _validated_mapping(level: HierarchyLevel, coarse_node_count: int) -> torch.Tensor:
    """Return the fine-to-coarse mapping for the active prolongation level.

    Parameters
    ----------
    level : HierarchyLevel
        Active hierarchy transition.
    coarse_node_count : int
        Number of nodes in the current coarse position tensor.

    Returns
    -------
    torch.Tensor
        CPU ``long`` tensor with shape ``[N_fine]``.

    Raises
    ------
    ValueError
        If the mapping is missing or references a coarse node out of range.
    """
    if level.fine_to_coarse is None:
        raise ValueError("active hierarchy level is missing fine_to_coarse")

    fine_to_coarse = level.fine_to_coarse.detach().to(device="cpu", dtype=torch.long)
    if fine_to_coarse.ndim != 1 or fine_to_coarse.shape[0] != level.num_fine:
        raise ValueError(f"fine_to_coarse must have shape [{level.num_fine}]")
    if fine_to_coarse.numel() > 0 and int(fine_to_coarse.max().item()) >= coarse_node_count:
        raise ValueError("fine_to_coarse references a coarse node outside state.pos")
    if fine_to_coarse.numel() > 0 and int(fine_to_coarse.min().item()) < 0:
        raise ValueError("fine_to_coarse cannot contain negative coarse indices")
    return fine_to_coarse


def _torch_generator(
    problem: LayoutProblem,
    ctx: RuntimeContext,
    seed_offset: int,
) -> torch.Generator:
    """Resolve the torch RNG used by Gaussian jitter prolongation.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs containing the fallback seed.
    ctx : RuntimeContext
        Execution infrastructure, optionally carrying a shared generator.
    seed_offset : int
        Deterministic offset applied when creating a local fallback generator.

    Returns
    -------
    torch.Generator
        CPU generator used for ``torch.randn`` sampling.
    """
    if ctx.generator is not None:
        return ctx.generator
    generator = torch.Generator(device="cpu")
    generator.manual_seed(problem.seed + seed_offset)
    return generator


def _python_rng(problem: LayoutProblem, seed_offset: int) -> random.Random:
    """Create the Python RNG used by FM^3 lambda interpolation.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs containing the fallback seed.
    seed_offset : int
        Deterministic level offset.

    Returns
    -------
    random.Random
        Private pseudorandom generator for waggle placement.
    """
    return random.Random(problem.seed + seed_offset)


def _solar_steps(state: SolveState) -> Sequence[SolarHierarchyStep]:
    """Return cached solar-system prolongation metadata.

    Parameters
    ----------
    state : SolveState
        Mutable solve state.

    Returns
    -------
    sequence[SolarHierarchyStep]
        One prolongation metadata record per hierarchy transition.

    Raises
    ------
    ValueError
        If the metadata is missing or malformed.
    """
    steps = state.extras.get(_SOLAR_STEPS_KEY)
    if not isinstance(steps, list):
        raise ValueError("solar-system prolongation metadata is missing from state.extras")
    if not all(isinstance(step, SolarHierarchyStep) for step in steps):
        raise ValueError("solar-system prolongation metadata has an unexpected shape")
    return steps


def _create_random_position(
    center: torch.Tensor,
    radius: float,
    angle_1: float,
    angle_2: float,
    rng: random.Random,
) -> torch.Tensor:
    """Create a random point on a circular sector around ``center``.

    Parameters
    ----------
    center : torch.Tensor
        Center point with shape ``[2]``.
    radius : float
        Sampling radius.
    angle_1 : float
        Lower angle bound in radians.
    angle_2 : float
        Upper angle bound in radians.
    rng : random.Random
        Python RNG used for angle sampling.

    Returns
    -------
    torch.Tensor
        Sampled point with shape ``[2]``.
    """
    random_angle = angle_1 + (angle_2 - angle_1) * rng.random()
    offset = torch.tensor(
        [math.cos(random_angle) * radius, math.sin(random_angle) * radius],
        dtype=center.dtype,
        device=center.device,
    )
    return center + offset


def _waggled_inbetween_position(
    source: torch.Tensor,
    target: torch.Tensor,
    lambda_value: float,
    rng: random.Random,
    waggle_factor: float,
) -> torch.Tensor:
    """Place a node between two endpoints with configurable random waggle.

    Parameters
    ----------
    source : torch.Tensor
        Source point with shape ``[2]``.
    target : torch.Tensor
        Target point with shape ``[2]``.
    lambda_value : float
        Interpolation weight from ``source`` toward ``target``.
    rng : random.Random
        Python RNG used for waggle placement.
    waggle_factor : float
        Maximum waggle radius as a fraction of the source-target distance.

    Returns
    -------
    torch.Tensor
        Waggled point with shape ``[2]``.
    """
    inbetween = source + lambda_value * (target - source)
    radius = waggle_factor * float(torch.linalg.norm(target - source).item())
    return _create_random_position(
        inbetween,
        radius * rng.random(),
        0.0,
        2.0 * math.pi,
        rng,
    )


def _barycenter_position(points: List[torch.Tensor]) -> torch.Tensor:
    """Return the arithmetic mean of a non-empty point list.

    Parameters
    ----------
    points : list[torch.Tensor]
        Point list where each tensor has shape ``[2]``.

    Returns
    -------
    torch.Tensor
        Barycenter with shape ``[2]``.
    """
    return torch.stack(points, dim=0).mean(dim=0)


def _neighbor_indices(neighbors: Sequence[Union[int, Tuple[int, float]]]) -> List[int]:
    """Normalize a neighbor bucket into plain node indices.

    Parameters
    ----------
    neighbors : sequence[int | tuple[int, float]]
        Neighbor records from an adjacency list.

    Returns
    -------
    list[int]
        Neighbor indices in their original order.
    """
    indices: List[int] = []
    for neighbor in neighbors:
        if isinstance(neighbor, tuple):
            indices.append(int(neighbor[0]))
        else:
            indices.append(int(neighbor))
    return indices


@register_op
class DirectMapping(Op):
    """Copy coarse positions to fine nodes and add Gaussian jitter.

    Notes
    -----
    Randomness uses a CPU ``torch.Generator``. Each call consumes one
    ``torch.randn([N_fine, 2])`` sample when ``jitter_scale`` is non-zero.
    """

    name = "direct_mapping"
    category = OpCategory.PROLONG
    reads = ("hierarchy", "pos")
    writes = ("pos",)
    requires = ("hierarchy", "pos")

    def __init__(self, config: Optional[DirectMappingConfig] = None) -> None:
        """Store the direct-mapping configuration.

        Parameters
        ----------
        config : DirectMappingConfig, optional
            Prolongation jitter configuration.

        Returns
        -------
        None
            The operation stores the resolved configuration.
        """
        self.config = config or DirectMappingConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Prolong the active hierarchy level by copying coarse coordinates.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state carrying the current coarse coordinates.
        ctx : RuntimeContext
            Execution infrastructure, optionally carrying a shared torch RNG.

        Returns
        -------
        SolveState
            Updated state with fine-level positions.
        """
        coarse_pos = _validated_positions(state)
        level_index, level = _active_hierarchy_level(state)
        fine_to_coarse = _validated_mapping(level, coarse_pos.shape[0]).to(device=coarse_pos.device)

        fine_pos = coarse_pos[fine_to_coarse].clone()
        if self.config.jitter_scale != 0.0 and level.num_fine > 0:
            generator = _torch_generator(problem, ctx, seed_offset=level_index)
            noise = torch.randn(
                (level.num_fine, 2),
                generator=generator,
                dtype=coarse_pos.dtype,
                device="cpu",
            ).to(device=coarse_pos.device)
            fine_pos = fine_pos + noise * self.config.jitter_scale

        state.pos = fine_pos
        return state


@register_op
class LambdaInterpolation(Op):
    """Prolong FM^3 solar-system hierarchies via lambda interpolation.

    Notes
    -----
    Randomness uses a private ``random.Random`` instance seeded from
    ``problem.seed + level_index``. Candidate placement consumes repeated
    ``random()`` calls in fine-node order.
    """

    name = "lambda_interpolation"
    category = OpCategory.PROLONG
    reads = ("hierarchy", "pos")
    writes = ("pos",)
    requires = ("hierarchy", "pos")

    def __init__(self, config: Optional[LambdaInterpolationConfig] = None) -> None:
        """Store the lambda-interpolation configuration.

        Parameters
        ----------
        config : LambdaInterpolationConfig, optional
            Candidate waggle configuration.

        Returns
        -------
        None
            The operation stores the resolved configuration.
        """
        self.config = config or LambdaInterpolationConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Prolong the active FM^3 level using sun/planet/moon metadata.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state carrying the current coarse coordinates and
            cached FM^3 metadata in ``state.extras``.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            Updated state with fine-level positions.
        """
        del ctx

        coarse_pos = _validated_positions(state)
        level_index, level = _active_hierarchy_level(state)
        fine_to_coarse = _validated_mapping(level, coarse_pos.shape[0])
        steps = _solar_steps(state)
        if level_index >= len(steps):
            raise ValueError("solar-system prolongation metadata is shorter than state.hierarchy")
        step = steps[level_index]
        if not torch.equal(step.mapping.to(device="cpu", dtype=torch.long), fine_to_coarse):
            raise ValueError("solar-system prolongation metadata does not match active hierarchy")

        fine_positions = coarse_pos[fine_to_coarse.to(device=coarse_pos.device)].clone()
        rng = _python_rng(problem, seed_offset=level_index)

        for node, node_type in enumerate(step.node_types):
            if node_type != _TYPE_SUN:
                continue
            fine_positions[node] = coarse_pos[int(step.mapping[node].item())]

        for node, node_type in enumerate(step.node_types):
            if node_type not in (_TYPE_PLANET, _TYPE_MOON):
                continue

            sun_node = step.dedicated_sun[node]
            sun_position = fine_positions[sun_node]
            dedicated_distance = step.dedicated_sun_distance[node]
            candidates = [
                _waggled_inbetween_position(
                    sun_position,
                    fine_positions[neighbor_sun],
                    lambda_value,
                    rng,
                    self.config.waggle_factor,
                )
                for lambda_value, neighbor_sun in zip(
                    step.lambda_values[node],
                    step.neighbor_suns[node],
                )
            ]
            if not candidates:
                candidates.append(
                    _create_random_position(
                        sun_position,
                        dedicated_distance,
                        0.0,
                        2.0 * math.pi,
                        rng,
                    )
                )
            fine_positions[node] = _barycenter_position(candidates)

        for node in step.pm_nodes:
            sun_node = step.dedicated_sun[node]
            sun_position = fine_positions[sun_node]
            sun_distance = step.dedicated_sun_distance[node]
            candidates = [
                _waggled_inbetween_position(
                    sun_position,
                    fine_positions[moon_node],
                    sun_distance / max(step.dedicated_sun_distance[moon_node], _MIN_DISTANCE),
                    rng,
                    self.config.waggle_factor,
                )
                for moon_node in step.moon_children[node]
            ]
            candidates.extend(
                _waggled_inbetween_position(
                    sun_position,
                    fine_positions[neighbor_sun],
                    lambda_value,
                    rng,
                    self.config.waggle_factor,
                )
                for lambda_value, neighbor_sun in zip(
                    step.lambda_values[node],
                    step.neighbor_suns[node],
                )
            )
            if not candidates:
                candidates.append(
                    _create_random_position(
                        sun_position,
                        sun_distance,
                        0.0,
                        2.0 * math.pi,
                        rng,
                    )
                )
            fine_positions[node] = _barycenter_position(candidates)

        state.pos = fine_positions
        return state


@register_op
class NeighborSmoothing(Op):
    """Blend each node toward the mean position of its neighbors."""

    name = "neighbor_smoothing"
    category = OpCategory.PROLONG
    reads = ("pos", "adjacency")
    writes = ("pos",)
    requires = ("pos", "adjacency")

    def __init__(self, config: Optional[NeighborSmoothingConfig] = None) -> None:
        """Store the neighbor-smoothing configuration.

        Parameters
        ----------
        config : NeighborSmoothingConfig, optional
            Blending configuration.

        Returns
        -------
        None
            The operation stores the resolved configuration.
        """
        self.config = config or NeighborSmoothingConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Blend positions toward neighbor means using the active adjacency.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state with ``pos`` and list-based ``adjacency``.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            Updated state with smoothed positions.
        """
        del problem
        del ctx

        if not 0.0 <= self.config.blend_factor <= 1.0:
            raise ValueError("blend_factor must be within [0.0, 1.0]")
        if state.adjacency is None or not isinstance(state.adjacency, list):
            raise ValueError("state.adjacency must be a list-based adjacency before this op runs")

        positions = _validated_positions(state)
        if len(state.adjacency) != positions.shape[0]:
            raise ValueError("state.adjacency length must match the number of positioned nodes")

        smoothed = positions.clone()
        for node, neighbors in enumerate(state.adjacency):
            neighbor_ids = _neighbor_indices(neighbors)
            if not neighbor_ids:
                continue
            neighbor_index = torch.tensor(neighbor_ids, dtype=torch.long, device=positions.device)
            neighbor_mean = positions[neighbor_index].mean(dim=0)
            smoothed[node] = (self.config.blend_factor * positions[node]) + (
                (1.0 - self.config.blend_factor) * neighbor_mean
            )

        state.pos = smoothed
        return state
