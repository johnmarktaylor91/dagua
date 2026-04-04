"""Multilevel coarsening operations for composable layout pipelines."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from importlib import import_module
from typing import List, Optional, Sequence, Tuple

import torch

from dagua.layout._archive.classic.fmmm import (
    _TYPE_MOON,
    _TYPE_PLANET,
    _TYPE_PLANET_WITH_MOONS,
    _TYPE_SUN,
    _build_weighted_adjacency,
    _LevelGraph,
    _RandomNodeSet,
    _star_masses,
    _unique_edges_with_lengths,
)
from dagua.layout._archive.classic.sfdp import _build_graph as _build_sfdp_graph
from dagua.layout._archive.classic.sfdp import _heavy_edge_matching as _sfdp_heavy_edge_matching
from dagua.layout.ops.base import Op
from dagua.layout.ops.state import HierarchyLevel, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_multilevel = import_module("dagua.layout.multilevel")
_SOLAR_STEPS_KEY = "solar_system_steps"
_MIN_DISTANCE = 1.0e-3


@dataclass(frozen=True)
class SolarSystemCoarsenConfig:
    """Configuration for :class:`SolarSystemCoarsen`.

    Parameters
    ----------
    random_tries : int, default=20
        Number of distinct sun-node candidates sampled before keeping the
        highest star-mass node.
    target : int, default=50
        Stop coarsening when the current graph has at most this many nodes.
    """

    random_tries: int = 20
    target: int = 50


@dataclass(frozen=True)
class LayerAwareCoarsenConfig:
    """Configuration for :class:`LayerAwareCoarsen`.

    Parameters
    ----------
    hub_threshold_percentile : float, default=90.0
        Percentile used to identify within-layer hubs that should remain
        singleton anchors during grouping.
    min_nodes : int, default=2000
        Stop coarsening once the graph reaches at most this many nodes.
    max_levels : int, default=20
        Maximum number of hierarchy transitions to build.
    """

    hub_threshold_percentile: float = 90.0
    min_nodes: int = 2000
    max_levels: int = 20


@dataclass(frozen=True)
class StreamingCoarsenConfig:
    """Configuration for :class:`StreamingCoarsen`.

    Parameters
    ----------
    chunk_size : int, default=100_000_000
        Node or edge threshold above which the multilevel builder switches to
        the streaming coarsening path.
    """

    chunk_size: int = 100_000_000


@dataclass(frozen=True)
class SolarHierarchyStep:
    """FM^3 prolongation metadata for one fine-to-coarse transition.

    Parameters
    ----------
    mapping : torch.Tensor
        Fine-to-coarse node map with shape ``[N_fine]``.
    node_types : list[int]
        Per-node type labels using the FM^3 sun/planet/moon encoding.
    dedicated_sun : list[int]
        Sun node assigned to each fine node.
    dedicated_sun_distance : list[float]
        Distance from each fine node to its dedicated sun.
    pm_nodes : list[int]
        Planet nodes that own one or more moon children.
    moon_children : list[list[int]]
        Moon child lists keyed by fine node index.
    lambda_values : list[list[float]]
        Per-node interpolation weights toward neighboring suns.
    neighbor_suns : list[list[int]]
        Per-node neighboring sun IDs aligned with ``lambda_values``.
    """

    mapping: torch.Tensor
    node_types: List[int]
    dedicated_sun: List[int]
    dedicated_sun_distance: List[float]
    pm_nodes: List[int]
    moon_children: List[List[int]]
    lambda_values: List[List[float]]
    neighbor_suns: List[List[int]]


def _validated_edge_index(problem: LayoutProblem) -> torch.Tensor:
    """Return a validated CPU edge tensor.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs.

    Returns
    -------
    torch.Tensor
        CPU ``long`` tensor with shape ``[2, E]``.

    Raises
    ------
    ValueError
        If the edge tensor shape or node references are invalid.
    """
    edge_index = problem.edge_index
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("problem.edge_index must have shape [2, E]")

    edge_index_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    if edge_index_cpu.numel() == 0:
        return edge_index_cpu

    if int(edge_index_cpu.min().item()) < 0:
        raise ValueError("problem.edge_index cannot contain negative node indices")
    if int(edge_index_cpu.max().item()) >= problem.num_nodes:
        raise ValueError("problem.edge_index references a node outside problem.num_nodes")
    return edge_index_cpu.contiguous()


def _validated_edge_weights(problem: LayoutProblem) -> Optional[torch.Tensor]:
    """Return validated CPU edge weights when provided.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs.

    Returns
    -------
    torch.Tensor | None
        CPU float tensor with shape ``[E]`` when weights exist.

    Raises
    ------
    ValueError
        If ``problem.edge_weights`` does not match the edge count.
    """
    if problem.edge_weights is None:
        return None

    edge_weights = problem.edge_weights.detach().to(device="cpu", dtype=torch.float32)
    if edge_weights.ndim != 1 or edge_weights.shape[0] != problem.edge_index.shape[1]:
        raise ValueError("problem.edge_weights must have shape [E]")
    return edge_weights


def _resolved_node_sizes(problem: LayoutProblem) -> torch.Tensor:
    """Return CPU node sizes normalized to shape ``[N, 2]``.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs.

    Returns
    -------
    torch.Tensor
        CPU node-size tensor with shape ``[N, 2]``.
    """
    return _multilevel._ensure_node_sizes_2d(problem.node_sizes, problem.num_nodes).to(device="cpu")


def _torch_generator(problem: LayoutProblem, ctx: RuntimeContext) -> torch.Generator:
    """Resolve the torch RNG used by SFDP-style coarsening.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs containing the fallback seed.
    ctx : RuntimeContext
        Execution infrastructure, optionally carrying a shared generator.

    Returns
    -------
    torch.Generator
        CPU generator whose call sequence matches repeated SFDP
        ``torch.randperm`` sampling.
    """
    if ctx.generator is not None:
        return ctx.generator
    generator = torch.Generator(device="cpu")
    generator.manual_seed(problem.seed)
    return generator


def _python_rng(problem: LayoutProblem) -> random.Random:
    """Create the Python RNG used by FM^3-style coarsening.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs containing the seed.

    Returns
    -------
    random.Random
        Private pseudorandom generator seeded from ``problem.seed``.
    """
    return random.Random(problem.seed)


def _aggregate_node_sizes(
    fine_node_sizes: torch.Tensor,
    fine_to_coarse: torch.Tensor,
    num_coarse_nodes: int,
) -> torch.Tensor:
    """Aggregate fine node sizes into coarse-node bounding boxes.

    Parameters
    ----------
    fine_node_sizes : torch.Tensor
        Fine node sizes with shape ``[N_fine, 2]``.
    fine_to_coarse : torch.Tensor
        Fine-to-coarse mapping with shape ``[N_fine]``.
    num_coarse_nodes : int
        Number of coarse nodes.

    Returns
    -------
    torch.Tensor
        Coarse node sizes with shape ``[N_coarse, 2]``.
    """
    coarse_sizes = torch.zeros(
        (num_coarse_nodes, 2),
        dtype=fine_node_sizes.dtype,
        device="cpu",
    )
    expanded_index = fine_to_coarse.to(device="cpu", dtype=torch.long).unsqueeze(1).expand(-1, 2)
    coarse_sizes.scatter_reduce_(0, expanded_index, fine_node_sizes, reduce="amax")
    return coarse_sizes


def _convert_multilevel_levels(levels: Sequence[_multilevel.CoarseLevel]) -> List[HierarchyLevel]:
    """Convert multilevel hierarchy levels into the ops-state schema.

    Parameters
    ----------
    levels : sequence[multilevel.CoarseLevel]
        Levels returned by :mod:`dagua.layout.multilevel`.

    Returns
    -------
    list[HierarchyLevel]
        Converted hierarchy levels from finest to coarsest.
    """
    converted: List[HierarchyLevel] = []
    for level in levels:
        converted.append(
            HierarchyLevel(
                num_nodes=level.num_nodes,
                num_fine=level.num_fine,
                edge_index=None if level.edge_index is None else level.edge_index.detach().cpu(),
                node_sizes=None if level.node_sizes is None else level.node_sizes.detach().cpu(),
                fine_to_coarse=(
                    None if level.fine_to_coarse is None else level.fine_to_coarse.detach().cpu()
                ),
                fine_layer_assignments=(
                    None
                    if level.fine_layer_assignments is None
                    else level.fine_layer_assignments.detach().cpu()
                ),
                coarse_layer_assignments=(
                    None
                    if level.coarse_layer_assignments is None
                    else level.coarse_layer_assignments.detach().cpu()
                ),
            )
        )
    return converted


def _coarsen_solar_level(
    level_graph: _LevelGraph,
    node_masses: torch.Tensor,
    rng: random.Random,
    random_tries: int,
) -> Tuple[SolarHierarchyStep, _LevelGraph, torch.Tensor]:
    """Collapse one FM^3 level using configurable solar-system selection.

    Parameters
    ----------
    level_graph : _LevelGraph
        Fine graph for the current hierarchy level.
    node_masses : torch.Tensor
        Current-level node masses with shape ``[N]``.
    rng : random.Random
        Private Python RNG matching the classic FM^3 backend.
    random_tries : int
        Number of candidate suns sampled before taking the highest star mass.

    Returns
    -------
    tuple[SolarHierarchyStep, _LevelGraph, torch.Tensor]
        Prolongation metadata, the coarsened graph, and coarse node masses.
    """
    num_nodes = level_graph.num_nodes
    adjacency = _build_weighted_adjacency(level_graph.edge_index, num_nodes)
    edge_lengths_cpu = level_graph.edge_lengths.to(device="cpu", dtype=torch.float32)
    edge_weights_cpu = (
        level_graph.edge_weights.to(device="cpu", dtype=torch.float32)
        if level_graph.edge_weights.numel() > 0
        else torch.ones((level_graph.edge_index.shape[1],), dtype=torch.float32)
    )
    selectable_nodes = _RandomNodeSet.from_star_masses(_star_masses(node_masses, adjacency))

    mapping = torch.full((num_nodes,), fill_value=-1, dtype=torch.long)
    node_types = [0 for _ in range(num_nodes)]
    dedicated_sun = [-1 for _ in range(num_nodes)]
    dedicated_sun_distance = [0.0 for _ in range(num_nodes)]
    lambda_values: List[List[float]] = [[] for _ in range(num_nodes)]
    neighbor_suns: List[List[int]] = [[] for _ in range(num_nodes)]
    moon_children: List[List[int]] = [[] for _ in range(num_nodes)]
    sun_to_coarse: dict[int, int] = {}

    while not selectable_nodes.empty():
        sun_node = selectable_nodes.get_random_node_with_highest_star_mass(rng, random_tries)
        coarse_node = len(sun_to_coarse)
        sun_to_coarse[sun_node] = coarse_node
        mapping[sun_node] = coarse_node
        node_types[sun_node] = _TYPE_SUN
        dedicated_sun[sun_node] = sun_node
        dedicated_sun_distance[sun_node] = 0.0

        planet_nodes: List[int] = []
        for planet_node, edge_id in adjacency[sun_node]:
            distance_to_sun = float(edge_lengths_cpu[edge_id].item())
            node_types[planet_node] = _TYPE_PLANET
            dedicated_sun[planet_node] = sun_node
            dedicated_sun_distance[planet_node] = distance_to_sun
            mapping[planet_node] = coarse_node
            planet_nodes.append(planet_node)

        for planet_node in planet_nodes:
            selectable_nodes.delete(planet_node)

        for planet_node in planet_nodes:
            for possible_moon, _edge_id in adjacency[planet_node]:
                selectable_nodes.delete(possible_moon)

    for node in range(num_nodes):
        if node_types[node] != 0:
            continue

        nearest_neighbor = -1
        nearest_distance = math.inf
        for neighbor, edge_id in adjacency[node]:
            if node_types[neighbor] not in (_TYPE_PLANET, _TYPE_PLANET_WITH_MOONS):
                continue
            distance = float(edge_lengths_cpu[edge_id].item())
            if distance < nearest_distance:
                nearest_neighbor = neighbor
                nearest_distance = distance

        if nearest_neighbor < 0:
            for neighbor, edge_id in adjacency[node]:
                if dedicated_sun[neighbor] < 0:
                    continue
                distance = float(edge_lengths_cpu[edge_id].item())
                if distance < nearest_distance:
                    nearest_neighbor = neighbor
                    nearest_distance = distance

        if nearest_neighbor < 0:
            coarse_node = len(sun_to_coarse)
            sun_to_coarse[node] = coarse_node
            mapping[node] = coarse_node
            node_types[node] = _TYPE_SUN
            dedicated_sun[node] = node
            continue

        assigned_sun = dedicated_sun[nearest_neighbor]
        mapping[node] = sun_to_coarse[assigned_sun]
        node_types[node] = _TYPE_MOON
        dedicated_sun[node] = assigned_sun
        dedicated_sun_distance[node] = nearest_distance + dedicated_sun_distance[nearest_neighbor]
        node_types[nearest_neighbor] = _TYPE_PLANET_WITH_MOONS
        moon_children[nearest_neighbor].append(node)

    pm_nodes = [node for node in range(num_nodes) if node_types[node] == _TYPE_PLANET_WITH_MOONS]

    coarse_masses = torch.zeros((len(sun_to_coarse),), dtype=torch.long)
    for coarse_node in mapping.tolist():
        coarse_masses[coarse_node] += 1

    pair_sums: dict[Tuple[int, int], float] = {}
    pair_counts: dict[Tuple[int, int], int] = {}
    pair_weight_sums: dict[Tuple[int, int], float] = {}
    edge_index_cpu = level_graph.edge_index.to(device="cpu", dtype=torch.long)
    for edge_id, (source, target) in enumerate(
        zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist())
    ):
        source_sun = dedicated_sun[source]
        target_sun = dedicated_sun[target]
        if source_sun == target_sun:
            continue

        coarse_source = sun_to_coarse[source_sun]
        coarse_target = sun_to_coarse[target_sun]
        pair = (min(coarse_source, coarse_target), max(coarse_source, coarse_target))
        edge_length = float(edge_lengths_cpu[edge_id].item())
        edge_weight = float(edge_weights_cpu[edge_id].item())
        new_length = dedicated_sun_distance[source] + edge_length + dedicated_sun_distance[target]
        pair_sums[pair] = pair_sums.get(pair, 0.0) + new_length
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        pair_weight_sums[pair] = pair_weight_sums.get(pair, 0.0) + edge_weight

        source_lambda = dedicated_sun_distance[source] / max(new_length, _MIN_DISTANCE)
        target_lambda = dedicated_sun_distance[target] / max(new_length, _MIN_DISTANCE)
        lambda_values[source].append(source_lambda)
        lambda_values[target].append(target_lambda)
        neighbor_suns[source].append(target_sun)
        neighbor_suns[target].append(source_sun)

    if pair_sums:
        ordered_pairs = sorted(pair_sums)
        coarse_edge_index = (
            torch.tensor(ordered_pairs, dtype=torch.long).transpose(0, 1).contiguous()
        )
        coarse_edge_lengths = torch.tensor(
            [pair_sums[pair] / pair_counts[pair] for pair in ordered_pairs],
            dtype=torch.float32,
        )
        coarse_edge_weights = torch.tensor(
            [pair_weight_sums[pair] for pair in ordered_pairs],
            dtype=torch.float32,
        )
    else:
        coarse_edge_index = torch.empty((2, 0), dtype=torch.long)
        coarse_edge_lengths = torch.empty((0,), dtype=torch.float32)
        coarse_edge_weights = torch.empty((0,), dtype=torch.float32)

    return (
        SolarHierarchyStep(
            mapping=mapping,
            node_types=node_types,
            dedicated_sun=dedicated_sun,
            dedicated_sun_distance=dedicated_sun_distance,
            pm_nodes=pm_nodes,
            moon_children=moon_children,
            lambda_values=lambda_values,
            neighbor_suns=neighbor_suns,
        ),
        _LevelGraph(
            edge_index=coarse_edge_index,
            edge_lengths=coarse_edge_lengths,
            num_nodes=len(sun_to_coarse),
            edge_weights=coarse_edge_weights,
        ),
        coarse_masses,
    )


def _build_solar_hierarchy(
    problem: LayoutProblem,
    config: SolarSystemCoarsenConfig,
) -> Tuple[List[HierarchyLevel], List[SolarHierarchyStep]]:
    """Build an FM^3-style hierarchy and matching prolongation metadata.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs.
    config : SolarSystemCoarsenConfig
        Solar-system coarsening parameters.

    Returns
    -------
    tuple[list[HierarchyLevel], list[SolarHierarchyStep]]
        Hierarchy levels and per-level prolongation metadata, both ordered from
        finest transition to coarsest transition.
    """
    edge_index = _validated_edge_index(problem)
    edge_weights = _validated_edge_weights(problem)
    current_sizes = _resolved_node_sizes(problem)
    base_edges, base_lengths, base_weights = _unique_edges_with_lengths(
        edge_index=edge_index,
        num_nodes=problem.num_nodes,
        edge_weights=edge_weights,
    )

    levels = [
        _LevelGraph(
            edge_index=base_edges,
            edge_lengths=base_lengths,
            num_nodes=problem.num_nodes,
            edge_weights=base_weights,
        )
    ]
    hierarchy: List[HierarchyLevel] = []
    prolong_steps: List[SolarHierarchyStep] = []
    current_masses = torch.ones((problem.num_nodes,), dtype=torch.long)
    current_nodes = problem.num_nodes
    bad_edge_counter = 0
    rng = _python_rng(problem)

    while current_nodes > config.target:
        if len(levels) > 1:
            previous_edge_count = int(levels[-2].edge_index.shape[1])
            current_edge_count = int(levels[-1].edge_index.shape[1])
            if current_edge_count > 0.8 * previous_edge_count:
                if bad_edge_counter < 5:
                    bad_edge_counter += 1
                else:
                    break

        step, coarse_level, coarse_masses = _coarsen_solar_level(
            levels[-1],
            current_masses,
            rng,
            config.random_tries,
        )
        if coarse_level.num_nodes >= current_nodes:
            break

        coarse_sizes = _aggregate_node_sizes(current_sizes, step.mapping, coarse_level.num_nodes)
        hierarchy.append(
            HierarchyLevel(
                num_nodes=coarse_level.num_nodes,
                num_fine=current_nodes,
                edge_index=coarse_level.edge_index.clone(),
                node_sizes=coarse_sizes.clone(),
                fine_to_coarse=step.mapping.clone(),
            )
        )
        prolong_steps.append(step)
        levels.append(coarse_level)
        current_nodes = coarse_level.num_nodes
        current_masses = coarse_masses
        current_sizes = coarse_sizes

    return hierarchy, prolong_steps


def _build_layered_hierarchy(
    problem: LayoutProblem,
    state: SolveState,
    min_nodes: int,
    max_levels: int,
    hub_threshold_percentile: Optional[float] = None,
    streaming_threshold: Optional[int] = None,
) -> List[HierarchyLevel]:
    """Build a hierarchy through the shared multilevel coarsener.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs.
    state : SolveState
        Mutable solve state carrying optional precomputed layers.
    min_nodes : int
        Stop threshold passed to :func:`dagua.layout.multilevel.build_hierarchy`.
    max_levels : int
        Maximum hierarchy depth.
    hub_threshold_percentile : float | None, optional
        Temporary override for the multilevel hub threshold percentile.
    streaming_threshold : int | None, optional
        Temporary override for the multilevel streaming threshold.

    Returns
    -------
    list[HierarchyLevel]
        Built hierarchy levels from finest to coarsest.
    """
    edge_index = _validated_edge_index(problem)
    node_sizes = _resolved_node_sizes(problem)
    initial_layers = (
        None if state.layers is None else state.layers.detach().to(device="cpu", dtype=torch.long)
    )

    original_hub_percentile = _multilevel._HUB_PERCENTILE
    original_streaming_threshold = _multilevel._STREAMING_THRESHOLD
    try:
        if hub_threshold_percentile is not None:
            _multilevel._HUB_PERCENTILE = float(hub_threshold_percentile)
        if streaming_threshold is not None:
            _multilevel._STREAMING_THRESHOLD = int(streaming_threshold)
        levels = _multilevel.build_hierarchy(
            edge_index=edge_index,
            num_nodes=problem.num_nodes,
            node_sizes=node_sizes,
            min_nodes=min_nodes,
            max_levels=max_levels,
            device="cpu",
            initial_layer_assignments=initial_layers,
            offload_to_disk=False,
        )
    finally:
        _multilevel._HUB_PERCENTILE = original_hub_percentile
        _multilevel._STREAMING_THRESHOLD = original_streaming_threshold

    return _convert_multilevel_levels(levels)


@register_op
class HeavyEdgeMatching(Op):
    """Build an SFDP-style hierarchy using random-order heavy-edge matching.

    Notes
    -----
    Randomness uses a CPU ``torch.Generator``. Each coarsening level consumes
    one ``torch.randperm(num_nodes)`` call through the classic SFDP matcher.
    """

    name = "heavy_edge_matching"
    category = OpCategory.COARSEN
    writes = ("hierarchy",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build a full SFDP hierarchy and store it on ``state.hierarchy``.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure carrying the optional torch RNG.

        Returns
        -------
        SolveState
            Updated state with a finest-to-coarsest SFDP hierarchy.
        """
        edge_index = _validated_edge_index(problem)
        edge_weights = _validated_edge_weights(problem)
        graph = _build_sfdp_graph(
            edge_index=edge_index,
            num_nodes=problem.num_nodes,
            edge_weights=edge_weights,
        )
        generator = _torch_generator(problem, ctx)
        current_sizes = _resolved_node_sizes(problem)

        hierarchy: List[HierarchyLevel] = []
        while True:
            coarsened = _sfdp_heavy_edge_matching(graph=graph, generator=generator)
            if coarsened is None:
                break

            fine_to_coarse, coarse_graph = coarsened
            coarse_sizes = _aggregate_node_sizes(
                current_sizes,
                fine_to_coarse,
                coarse_graph.num_nodes,
            )
            hierarchy.append(
                HierarchyLevel(
                    num_nodes=coarse_graph.num_nodes,
                    num_fine=graph.num_nodes,
                    edge_index=coarse_graph.edge_index.clone(),
                    node_sizes=coarse_sizes.clone(),
                    fine_to_coarse=fine_to_coarse.clone(),
                )
            )
            graph = coarse_graph
            current_sizes = coarse_sizes

        state.hierarchy = hierarchy
        state.extras.pop(_SOLAR_STEPS_KEY, None)
        return state


@register_op
class SolarSystemCoarsen(Op):
    """Build an FM^3-style hierarchy plus lambda-prolongation metadata.

    Notes
    -----
    Randomness uses a private ``random.Random`` instance seeded from
    ``problem.seed``. Sun selection consumes repeated ``randint`` calls through
    the FM^3 swap-delete node set.
    """

    name = "solar_system_coarsen"
    category = OpCategory.COARSEN
    writes = ("hierarchy",)

    def __init__(self, config: Optional[SolarSystemCoarsenConfig] = None) -> None:
        """Store the solar-system coarsening configuration.

        Parameters
        ----------
        config : SolarSystemCoarsenConfig, optional
            Solar-system coarsening parameters.

        Returns
        -------
        None
            The operation stores the resolved configuration.
        """
        self.config = config or SolarSystemCoarsenConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build an FM^3 hierarchy and cache the prolongation metadata.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            Updated state with ``hierarchy`` and solar-system metadata.
        """
        del ctx

        hierarchy, prolong_steps = _build_solar_hierarchy(problem, self.config)
        state.hierarchy = hierarchy
        state.extras[_SOLAR_STEPS_KEY] = prolong_steps
        return state


@register_op
class LayerAwareCoarsen(Op):
    """Build a layered hierarchy via the native multilevel coarsener."""

    name = "layer_aware_coarsen"
    category = OpCategory.COARSEN
    reads = ("layers",)
    writes = ("hierarchy",)

    def __init__(self, config: Optional[LayerAwareCoarsenConfig] = None) -> None:
        """Store the layer-aware coarsening configuration.

        Parameters
        ----------
        config : LayerAwareCoarsenConfig, optional
            Layer-aware hierarchy parameters.

        Returns
        -------
        None
            The operation stores the resolved configuration.
        """
        self.config = config or LayerAwareCoarsenConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build a full layer-aware hierarchy using the multilevel builder.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state, optionally with precomputed ``layers``.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            Updated state with ``hierarchy`` populated.
        """
        del ctx

        state.hierarchy = _build_layered_hierarchy(
            problem=problem,
            state=state,
            min_nodes=self.config.min_nodes,
            max_levels=self.config.max_levels,
            hub_threshold_percentile=self.config.hub_threshold_percentile,
        )
        state.extras.pop(_SOLAR_STEPS_KEY, None)
        return state


@register_op
class StreamingCoarsen(Op):
    """Build a hierarchy using the multilevel streaming threshold override."""

    name = "streaming_coarsen"
    category = OpCategory.COARSEN
    reads = ("layers",)
    writes = ("hierarchy",)

    def __init__(self, config: Optional[StreamingCoarsenConfig] = None) -> None:
        """Store the streaming coarsening configuration.

        Parameters
        ----------
        config : StreamingCoarsenConfig, optional
            Streaming threshold configuration.

        Returns
        -------
        None
            The operation stores the resolved configuration.
        """
        self.config = config or StreamingCoarsenConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build a hierarchy while forcing the configured streaming threshold.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state, optionally with precomputed ``layers``.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            Updated state with ``hierarchy`` populated.
        """
        del ctx

        state.hierarchy = _build_layered_hierarchy(
            problem=problem,
            state=state,
            min_nodes=2000,
            max_levels=20,
            streaming_threshold=self.config.chunk_size,
        )
        state.extras.pop(_SOLAR_STEPS_KEY, None)
        return state
