"""SFDP multilevel force-directed layout pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Optional, Tuple, Union

import torch

from dagua.layout.ops.base import Op, Pipeline, Repeat
from dagua.layout.ops.graph_utils import (
    layout_device as _layout_device,
)
from dagua.layout.ops.quadtree import GraphvizQuadTree, graphviz_supernode_repulsive_force
from dagua.layout.ops.sfdp import (
    _BASE_GRAPH_KEY,
    _GENERATOR_KEY,
    _GRAPH_KEY,
    _MAPPING_KEY,
    _SFDP_ALGORITHM_CONFIG,
    _SFDP_CURRENT_STEP_KEY,
    _SFDP_FORCE_NORM_KEY,
    _SFDP_PREVIOUS_FORCE_NORM_KEY,
    BuildSFDPGraph,
    BuildSFDPHierarchy,
    GraphData,
    GraphvizRandom,
    InitSFDPCoarsestPositions,
    SFDPAdaptiveCool,
    SFDPAdaptiveCoolConfig,
    SFDPFinalizePositions,
    SFDPHierarchyConfig,
    SFDPProlongateAndRefineLevels,
    SFDPRefineCoarsestLevel,
    _build_graph,
    _prolongate_positions,
)
from dagua.layout.ops.state import (
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)
from dagua.layout.ops.taxonomy import OpCategory

_DEFAULT_THETA = 0.6
_DEFAULT_P = -1.0
_GRAPHVIZ_MAX_CLUSTER_SIZE = 4
_GRAPHVIZ_FIDELITY_MODE = "graphviz"
_GRAPHVIZ_MIN_DISTANCE = 1.0e-15
SFDPFidelityMode = Optional[Union[bool, str]]


def _decompose_graphviz_supervariables(graph: GraphData) -> list[list[int]]:
    """Group nodes with Graphviz's sparse-matrix supervariable refinement.

    Parameters
    ----------
    graph : GraphData
        Symmetric SFDP graph whose adjacency lists represent sparse matrix rows.

    Returns
    -------
    list[list[int]]
        Supervariable groups in Graphviz creation order. Each inner list contains
        fine node IDs that have the same sparse-matrix column pattern.
    """
    num_nodes = graph.num_nodes
    if num_nodes == 0:
        return []

    super_ids = [0 for _ in range(num_nodes)]
    super_sizes = [0 for _ in range(num_nodes + 1)]
    mask = [-1 for _ in range(num_nodes)]
    newmap = [0 for _ in range(num_nodes)]
    super_sizes[0] = num_nodes
    next_super_id = 1

    for row_index, neighbors in enumerate(graph.adjacency):
        neighbor_ids = [neighbor for neighbor, _ in neighbors]
        for neighbor in neighbor_ids:
            super_sizes[super_ids[neighbor]] -= 1

        for neighbor in neighbor_ids:
            old_super = super_ids[neighbor]
            if mask[old_super] < row_index:
                mask[old_super] = row_index
                if super_sizes[old_super] == 0:
                    super_sizes[old_super] = 1
                    newmap[old_super] = old_super
                else:
                    new_super = next_super_id
                    next_super_id += 1
                    newmap[old_super] = new_super
                    super_sizes[new_super] = 1
                    super_ids[neighbor] = new_super
            else:
                mapped_super = newmap[old_super]
                super_ids[neighbor] = mapped_super
                super_sizes[mapped_super] += 1

    groups: list[list[int]] = [[] for _ in range(next_super_id)]
    for node, super_id in enumerate(super_ids):
        groups[super_id].append(node)
    return groups


def _graphviz_sfdp_cluster_nodes(
    graph: GraphData,
    generator: GraphvizRandom,
    config: SFDPHierarchyConfig,
) -> Optional[torch.Tensor]:
    """Build Graphviz SFDP fine-to-coarse clusters for one internal pass.

    Parameters
    ----------
    graph : GraphData
        Fine graph for the current internal coarsening pass.
    generator : GraphvizRandom
        Graphviz-compatible RNG stream used for the unmatched-node permutation.
    config : SFDPHierarchyConfig
        Graphviz-compatible coarsening stop thresholds.

    Returns
    -------
    torch.Tensor | None
        Fine-to-coarse assignment with shape ``[N]`` when this pass produces a
        valid Graphviz internal coarsening, otherwise ``None``.
    """
    num_nodes = graph.num_nodes
    if num_nodes < config.min_coarse_size:
        return None

    matched = [False for _ in range(num_nodes)]
    clusters: list[list[int]] = []

    for super_group in _decompose_graphviz_supervariables(graph):
        if len(super_group) <= 1:
            continue
        for start in range(0, len(super_group), _GRAPHVIZ_MAX_CLUSTER_SIZE):
            chunk = super_group[start : start + _GRAPHVIZ_MAX_CLUSTER_SIZE]
            for node in chunk:
                matched[node] = True
            clusters.append(chunk)

    for node in generator.permutation(num_nodes):
        if matched[node]:
            continue

        partner = -1
        partner_weight = -1.0
        for neighbor, weight in graph.adjacency[node]:
            if matched[neighbor]:
                continue
            if weight > partner_weight:
                partner = neighbor
                partner_weight = weight

        if partner >= 0:
            matched[node] = True
            matched[partner] = True
            clusters.append([node, partner])

    for node, is_matched in enumerate(matched):
        if not is_matched:
            clusters.append([node])

    coarse_num_nodes = len(clusters)
    if coarse_num_nodes == num_nodes or coarse_num_nodes < config.min_coarse_size:
        return None

    fine_to_coarse = torch.empty((num_nodes,), dtype=torch.long)
    for coarse_node, cluster in enumerate(clusters):
        for fine_node in cluster:
            fine_to_coarse[fine_node] = coarse_node
    return fine_to_coarse


def _build_graphviz_matrix_coarse_graph(
    graph: GraphData,
    fine_to_coarse: torch.Tensor,
    coarse_num_nodes: int,
) -> GraphData:
    """Aggregate a coarse graph through Graphviz's ``R * A * P`` semantics.

    Parameters
    ----------
    graph : GraphData
        Fine graph represented as unique undirected weighted edges.
    fine_to_coarse : torch.Tensor
        Fine-to-coarse assignment with shape ``[N_fine]``.
    coarse_num_nodes : int
        Number of coarse clusters.

    Returns
    -------
    GraphData
        Coarse graph with diagonal entries removed after matrix aggregation.
    """
    coarse_edges: dict[tuple[int, int], float] = {}
    for edge_id in range(graph.edge_index.shape[1]):
        source = int(graph.edge_index[0, edge_id].item())
        target = int(graph.edge_index[1, edge_id].item())
        coarse_source = int(fine_to_coarse[source].item())
        coarse_target = int(fine_to_coarse[target].item())
        if coarse_source == coarse_target:
            continue
        lower = min(coarse_source, coarse_target)
        upper = max(coarse_source, coarse_target)
        coarse_edges[(lower, upper)] = coarse_edges.get((lower, upper), 0.0) + float(
            graph.edge_weight[edge_id].item()
        )

    if coarse_edges:
        ordered_edges = sorted(coarse_edges.items())
        edge_pairs = [edge for edge, _ in ordered_edges]
        edge_weights = [weight for _, weight in ordered_edges]
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).transpose(0, 1).contiguous()
        weight_tensor = torch.tensor(edge_weights, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        weight_tensor = torch.empty((0,), dtype=torch.float32)

    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(coarse_num_nodes)]
    for edge_id in range(edge_index.shape[1]):
        source = int(edge_index[0, edge_id].item())
        target = int(edge_index[1, edge_id].item())
        weight = float(weight_tensor[edge_id].item())
        adjacency[source].append((target, weight))
        adjacency[target].append((source, weight))

    return GraphData(
        num_nodes=coarse_num_nodes,
        edge_index=edge_index,
        edge_weight=weight_tensor,
        adjacency=adjacency,
        graphviz_average=True,
    )


def _graphviz_sfdp_coarsen(
    graph: GraphData,
    generator: GraphvizRandom,
    config: SFDPHierarchyConfig,
) -> Optional[tuple[torch.Tensor, GraphData]]:
    """Coarsen one Graphviz SFDP multilevel edge using matrix aggregation.

    Parameters
    ----------
    graph : GraphData
        Fine graph at the current hierarchy level.
    generator : GraphvizRandom
        Graphviz-compatible RNG stream used by unmatched-node heavy-edge passes.
    config : SFDPHierarchyConfig
        Coarsening thresholds. ``min_coarsen_reduction`` controls the wrapper
        loop that forces sufficient reduction across internal passes.

    Returns
    -------
    tuple[torch.Tensor, GraphData] | None
        Composed fine-to-coarse mapping and coarse graph, or ``None`` if the
        Graphviz internal pass cannot produce a usable coarse graph.
    """
    original_num_nodes = graph.num_nodes
    current_graph = graph
    composed_mapping: Optional[torch.Tensor] = None
    last_result: Optional[tuple[torch.Tensor, GraphData]] = None

    while True:
        internal_mapping = _graphviz_sfdp_cluster_nodes(
            graph=current_graph,
            generator=generator,
            config=config,
        )
        if internal_mapping is None:
            return last_result

        coarse_num_nodes = int(internal_mapping.max().item()) + 1
        coarse_graph = _build_graphviz_matrix_coarse_graph(
            graph=current_graph,
            fine_to_coarse=internal_mapping,
            coarse_num_nodes=coarse_num_nodes,
        )
        composed_mapping = (
            internal_mapping
            if composed_mapping is None
            else internal_mapping[composed_mapping].to(dtype=torch.long)
        )
        last_result = (composed_mapping.clone(), coarse_graph)

        if coarse_num_nodes <= config.min_coarsen_reduction * float(original_num_nodes):
            return last_result

        current_graph = coarse_graph


@dataclass(frozen=True)
class BuildGraphvizSFDPMatrixHierarchy(Op):
    """Build Graphviz-compatible SFDP hierarchy using matrix coarsening."""

    config: SFDPHierarchyConfig = field(default_factory=SFDPHierarchyConfig)

    name: ClassVar[str] = "sfdp_graphviz_matrix_coarsen_hierarchy"
    category: ClassVar[OpCategory] = OpCategory.COARSEN
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)
    requires: ClassVar[Tuple[str, ...]] = (f"extras.{_BASE_GRAPH_KEY}",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Iteratively build the fidelity-mode Graphviz SFDP hierarchy.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs containing the deterministic seed.
        state : SolveState
            Mutable solve state. Reads the base SFDP graph from ``extras``.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with SFDP graph levels, mappings, and generator populated.
        """
        del ctx
        permutation_generator = GraphvizRandom(seed=1)

        base_graph = _build_graph(
            edge_index=problem.edge_index,
            num_nodes=problem.num_nodes,
            edge_weights=problem.edge_weights,
            graphviz_order=True,
        )
        state.extras[_BASE_GRAPH_KEY] = base_graph
        graphs: list[GraphData] = [base_graph]
        mappings: list[torch.Tensor] = []
        current_graph = base_graph

        while True:
            coarsened = _graphviz_sfdp_coarsen(
                graph=current_graph,
                generator=permutation_generator,
                config=self.config,
            )
            if coarsened is None:
                break

            fine_to_coarse, coarse_graph = coarsened
            mappings.append(fine_to_coarse)
            graphs.append(coarse_graph)
            current_graph = coarse_graph

        state.extras[_GRAPH_KEY] = graphs
        state.extras[_MAPPING_KEY] = mappings
        # Graphviz coarsening consumes the process-default rand stream before
        # spring_electrical.c resets srand(ctrl->random_seed) for random_start.
        state.extras[_GENERATOR_KEY] = GraphvizRandom(seed=problem.seed)
        return state


def _is_graphviz_fidelity_mode(fidelity_mode: SFDPFidelityMode) -> bool:
    """Return whether the SFDP Graphviz fidelity path is enabled.

    Parameters
    ----------
    fidelity_mode : bool or str or None
        Fidelity selector. ``True`` is kept as a backwards-compatible alias for
        the Graphviz path already used by the matrix-coarsening slice;
        ``"graphviz"`` is the explicit public selector for this sprint.

    Returns
    -------
    bool
        ``True`` when Graphviz fidelity behavior should be used.

    Raises
    ------
    ValueError
        If a string selector other than ``"graphviz"`` is supplied.
    """
    if fidelity_mode is None or fidelity_mode is False:
        return False
    if fidelity_mode is True or fidelity_mode == _GRAPHVIZ_FIDELITY_MODE:
        return True
    raise ValueError("fidelity_mode must be False, True, None, or 'graphviz'.")


def _sfdp_force_scales(ideal_length: float, repulsive_exponent: float) -> tuple[float, float]:
    """Compute Graphviz SFDP attractive and repulsive scale constants.

    Parameters
    ----------
    ideal_length : float
        Current ideal edge length ``K``.
    repulsive_exponent : float
        SFDP repulsive exponent ``p``.

    Returns
    -------
    tuple[float, float]
        ``(CRK, KP)`` scale constants from ``spring_electrical.c``.
    """
    bounded_ideal_length = max(ideal_length, _SFDP_ALGORITHM_CONFIG.min_span)
    attractive_scale = (
        _SFDP_ALGORITHM_CONFIG.force_scaling ** ((2.0 - repulsive_exponent) / 3.0)
    ) / bounded_ideal_length
    repulsive_scale = bounded_ideal_length ** (1.0 - repulsive_exponent)
    return attractive_scale, repulsive_scale


def _graphviz_power_law_repulsive_exponent(
    graph: GraphData,
    repulsive_exponent: float,
) -> float:
    """Return Graphviz's auto-selected SFDP repulsive exponent.

    Parameters
    ----------
    graph : GraphData
        Graph whose degree distribution is tested with Graphviz's
        ``power_law_graph`` heuristic.
    repulsive_exponent : float
        Requested repulsive exponent. The Graphviz default arrives through the
        Dagua fidelity selector as ``-1.0``.

    Returns
    -------
    float
        ``-1.8`` for Graphviz power-law graphs under the default selector,
        otherwise the requested exponent.
    """
    if repulsive_exponent != _SFDP_ALGORITHM_CONFIG.default_repulsive_exponent:
        return repulsive_exponent

    degree_counts = [0 for _ in range(graph.num_nodes + 1)]
    max_count = 0
    for neighbors in graph.adjacency:
        degree = len(neighbors)
        degree_counts[degree] += 1
        max_count = max(max_count, degree_counts[degree])

    if degree_counts[1] > 0.8 * max_count and degree_counts[1] > 0.3 * graph.num_nodes:
        return -1.8
    return repulsive_exponent


@dataclass(frozen=True)
class _SFDPGraphvizSequentialStep(Op):
    """Apply one Graphviz-style sequential spring-electrical iteration.

    Parameters
    ----------
    graph : GraphData
        Current SFDP graph level.
    attractive_scale : float
        Graphviz ``CRK`` attractive-force coefficient.
    repulsive_scale : float
        Graphviz ``KP`` repulsive-force coefficient.
    theta : float, default=0.6
        Barnes-Hut opening angle threshold used by Graphviz's quadtree path.
    repulsive_exponent : float, default=-1.0
        SFDP repulsion exponent ``p``.

    Notes
    -----
    This ports the non-fast ``spring_electrical_embedding`` update order: each
    vertex force is computed from the latest coordinate array and immediately
    written back before the next vertex is evaluated.
    """

    graph: GraphData
    attractive_scale: float
    repulsive_scale: float
    theta: float = _DEFAULT_THETA
    repulsive_exponent: float = _SFDP_ALGORITHM_CONFIG.default_repulsive_exponent

    name: ClassVar[str] = "sfdp_graphviz_sequential_step"
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
        """Move vertices one at a time using Graphviz's SFDP force order.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable state containing positions with shape ``[N, 2]`` and the
            current SFDP step size in ``extras``.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with updated positions and current total force norm.

        Raises
        ------
        ValueError
            If positions are missing.
        """
        del problem, ctx
        if state.pos is None:
            raise ValueError("_SFDPGraphvizSequentialStep requires state.pos to be set.")

        positions = state.pos.to(dtype=torch.float64, device="cpu").clone()
        current_step = float(
            state.extras.get(_SFDP_CURRENT_STEP_KEY, _SFDP_ALGORITHM_CONFIG.default_step)
        )
        force_norm = 0.0
        quadtree = (
            GraphvizQuadTree.from_points(
                coordinates=positions,
                max_level=_SFDP_ALGORITHM_CONFIG.max_quadtree_depth,
            )
            if self.graph.num_nodes >= _SFDP_ALGORITHM_CONFIG.barnes_hut_threshold
            else None
        )

        for node in range(self.graph.num_nodes):
            force = torch.zeros((2,), dtype=torch.float64)

            for neighbor, _weight in self.graph.adjacency[node]:
                if neighbor == node:
                    continue
                delta = positions[node] - positions[neighbor]
                distance = torch.linalg.vector_norm(delta)
                force = force - (self.attractive_scale * delta * distance)

            if quadtree is not None:
                force = force + graphviz_supernode_repulsive_force(
                    tree=quadtree,
                    positions=positions,
                    node_index=node,
                    theta=self.theta,
                    repulsive_scale=self.repulsive_scale,
                    repulsive_exponent=self.repulsive_exponent,
                ).to(dtype=torch.float64, device="cpu")
            else:
                for other in range(self.graph.num_nodes):
                    if other == node:
                        continue
                    delta = positions[node] - positions[other]
                    distance = max(
                        float(torch.linalg.vector_norm(delta).item()),
                        _GRAPHVIZ_MIN_DISTANCE,
                    )
                    denominator = distance ** (1.0 - self.repulsive_exponent)
                    force = force + (self.repulsive_scale * delta / denominator)

            node_force_norm = float(torch.linalg.vector_norm(force).item())
            force_norm += node_force_norm
            if node_force_norm > 0.0:
                positions[node] = positions[node] + (current_step * force / node_force_norm)

        state.pos = positions
        state.extras[_SFDP_FORCE_NORM_KEY] = force_norm
        return state


def _apply_graphviz_sequential_refinement(
    problem: LayoutProblem,
    state: SolveState,
    ctx: RuntimeContext,
    *,
    graph: GraphData,
    steps: int,
    ideal_length: float,
    theta: float,
    repulsive_exponent: float,
    adaptive_cooling: bool,
) -> SolveState:
    """Run Graphviz-style sequential refinement for one SFDP level.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs forwarded to inner ops.
    state : SolveState
        Mutable solve state containing current positions.
    ctx : RuntimeContext
        Execution infrastructure forwarded to inner ops.
    graph : GraphData
        Current SFDP graph level.
    steps : int
        Maximum spring-electrical iterations for this level.
    ideal_length : float
        Current ideal edge length ``K``.
    theta : float
        Barnes-Hut opening angle threshold.
    repulsive_exponent : float
        SFDP repulsive exponent ``p``.
    adaptive_cooling : bool
        Whether to use Graphviz's force-norm adaptive cooling.

    Returns
    -------
    SolveState
        State with refined positions.
    """
    if state.pos is None:
        raise ValueError("Graphviz SFDP refinement requires state.pos to be set.")
    if steps == 0 or state.pos.shape[0] <= 1:
        return state

    attractive_scale, repulsive_scale = _sfdp_force_scales(
        ideal_length=ideal_length,
        repulsive_exponent=repulsive_exponent,
    )
    state.extras[_SFDP_CURRENT_STEP_KEY] = _SFDP_ALGORITHM_CONFIG.default_step
    state.extras[_SFDP_PREVIOUS_FORCE_NORM_KEY] = float("inf")
    state = Repeat(
        n=steps,
        ops=[
            _SFDPGraphvizSequentialStep(
                graph=graph,
                attractive_scale=attractive_scale,
                repulsive_scale=repulsive_scale,
                theta=theta,
                repulsive_exponent=repulsive_exponent,
            ),
            SFDPAdaptiveCool(config=SFDPAdaptiveCoolConfig(adaptive_cooling=adaptive_cooling)),
        ],
    ).apply(problem, state, ctx)
    state.converged = False
    return state


@dataclass(frozen=True)
class _SFDPGraphvizRefineCoarsestLevel(Op):
    """Refine the coarsest level with Graphviz sequential node updates.

    Parameters
    ----------
    steps : int, default=500
        Maximum number of spring-electrical iterations.
    theta : float, default=0.6
        Barnes-Hut opening angle threshold.
    repulsive_exponent : float, default=-1.0
        SFDP repulsion exponent ``p``.
    """

    steps: int = 500
    theta: float = _DEFAULT_THETA
    repulsive_exponent: float = _SFDP_ALGORITHM_CONFIG.default_repulsive_exponent

    name: ClassVar[str] = "sfdp_graphviz_refine_coarsest"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("pos", "ideal_length", "extras", "converged")
    writes: ClassVar[Tuple[str, ...]] = ("pos", "extras", "converged")
    requires: ClassVar[Tuple[str, ...]] = ("pos", "ideal_length", f"extras.{_GRAPH_KEY}")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run sequential refinement on the coarsest hierarchy graph.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state containing coarsest positions and ideal length.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with coarsest positions refined.

        Raises
        ------
        ValueError
            If ``ideal_length`` is missing.
        """
        if state.ideal_length is None:
            raise ValueError("_SFDPGraphvizRefineCoarsestLevel requires state.ideal_length.")
        graphs: list[GraphData] = state.extras[_GRAPH_KEY]
        effective_repulsive_exponent = _graphviz_power_law_repulsive_exponent(
            graph=graphs[0],
            repulsive_exponent=self.repulsive_exponent,
        )
        return _apply_graphviz_sequential_refinement(
            problem=problem,
            state=state,
            ctx=ctx,
            graph=graphs[-1],
            steps=self.steps,
            ideal_length=float(state.ideal_length),
            theta=self.theta,
            repulsive_exponent=effective_repulsive_exponent,
            adaptive_cooling=True,
        )


@dataclass(frozen=True)
class _SFDPGraphvizProlongateAndRefineLevels(Op):
    """Prolongate levels and refine each with Graphviz sequential updates.

    Parameters
    ----------
    steps : int, default=500
        Maximum number of spring-electrical iterations per level.
    theta : float, default=0.6
        Barnes-Hut opening angle threshold.
    repulsive_exponent : float, default=-1.0
        SFDP repulsion exponent ``p``.
    """

    steps: int = 500
    theta: float = _DEFAULT_THETA
    repulsive_exponent: float = _SFDP_ALGORITHM_CONFIG.default_repulsive_exponent

    name: ClassVar[str] = "sfdp_graphviz_prolongate_and_refine"
    category: ClassVar[OpCategory] = OpCategory.PROLONG
    reads: ClassVar[Tuple[str, ...]] = ("pos", "ideal_length", "extras", "converged")
    writes: ClassVar[Tuple[str, ...]] = ("pos", "ideal_length", "extras", "converged")
    requires: ClassVar[Tuple[str, ...]] = (
        "pos",
        "ideal_length",
        f"extras.{_GRAPH_KEY}",
        f"extras.{_MAPPING_KEY}",
        f"extras.{_GENERATOR_KEY}",
    )

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Walk from coarse to fine levels with sequential refinement.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state containing coarsest positions.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with positions on the finest graph level.

        Raises
        ------
        ValueError
            If ``ideal_length`` is missing.
        """
        graphs: list[GraphData] = state.extras[_GRAPH_KEY]
        mappings: list[torch.Tensor] = state.extras[_MAPPING_KEY]
        generator: GraphvizRandom = state.extras[_GENERATOR_KEY]
        if state.ideal_length is None:
            raise ValueError("_SFDPGraphvizProlongateAndRefineLevels requires state.ideal_length.")
        effective_repulsive_exponent = _graphviz_power_law_repulsive_exponent(
            graph=graphs[0],
            repulsive_exponent=self.repulsive_exponent,
        )

        positions = state.pos
        ideal_length = float(state.ideal_length)
        for level_index in range(len(mappings) - 1, -1, -1):
            ideal_length = max(
                ideal_length * _SFDP_ALGORITHM_CONFIG.refinement_k_decay,
                _SFDP_ALGORITHM_CONFIG.min_span,
            )
            fine_graph = graphs[level_index]
            positions = _prolongate_positions(
                graph=fine_graph,
                fine_to_coarse=mappings[level_index],
                coarse_positions=positions,
                ideal_length=ideal_length,
                generator=generator,
            )
            state.pos = positions
            state = _apply_graphviz_sequential_refinement(
                problem=problem,
                state=state,
                ctx=ctx,
                graph=fine_graph,
                steps=self.steps,
                ideal_length=ideal_length,
                theta=self.theta,
                repulsive_exponent=effective_repulsive_exponent,
                adaptive_cooling=False,
            )
            positions = state.pos

        state.pos = positions
        state.ideal_length = ideal_length
        return state


def build_sfdp_pipeline(
    steps: int = 500,
    theta: float = _DEFAULT_THETA,
    repulsive_exponent: float = _DEFAULT_P,
    fidelity_mode: SFDPFidelityMode = False,
    fidelity_dtype: torch.dtype = torch.float32,
) -> Pipeline:
    """Build an SFDP multilevel force-directed pipeline.

    Reference fidelity
    ------------------
    Targets: Graphviz 7.0.5 sfdp / Hu (2005), "Efficient, High-Quality
        Force-Directed Graph Drawing".
    Fidelity mode: ``fidelity_mode=True`` or ``"graphviz"`` enables the
        Graphviz matrix hierarchy and sequential spring-electrical node update
        path.
    Verified at: final 100-seed report, strong equivalent; median RMSD 0.079
        to 0.100. Round 33 force-law alignment improved the bounded subset to
        median RMSD 0.004724.
    Known divergences:
        - Remaining residual is dominated by ``parallel_multiedge_bundle``.
        - Output normalization remains Dagua-specific.
        - Random initialization, unmatched-node permutation, and prolongation
          jitter use the Graphviz ``srand``/``rand`` stream under fidelity mode.

    Parameters
    ----------
    steps : int, default=500
        Maximum number of spring-electrical iterations per level.
    theta : float, default=0.6
        Barnes-Hut opening angle threshold.
    repulsive_exponent : float, default=-1.0
        SFDP repulsion exponent ``p``.
    fidelity_mode : bool or str or None, default=False
        ``True`` or ``"graphviz"`` enables Graphviz-compatible matrix
        coarsening and sequential in-place updates.

    Returns
    -------
    Pipeline
        Pipeline implementing the classical SFDP algorithm. The pipeline
        produces final node coordinates by building the multilevel graph,
        initializing the coarsest level, refining with spring-electrical
        updates, prolongating through finer levels, and normalizing the final
        positions.

    Raises
    ------
    ValueError
        If ``steps`` is negative.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative.")

    graphviz_fidelity = _is_graphviz_fidelity_mode(fidelity_mode=fidelity_mode)
    hierarchy_op: Op
    refine_op: Op
    prolongate_op: Op
    if graphviz_fidelity:
        hierarchy_op = BuildGraphvizSFDPMatrixHierarchy()
        refine_op = _SFDPGraphvizRefineCoarsestLevel(
            steps=steps,
            theta=theta,
            repulsive_exponent=repulsive_exponent,
        )
        prolongate_op = _SFDPGraphvizProlongateAndRefineLevels(
            steps=steps,
            theta=theta,
            repulsive_exponent=repulsive_exponent,
        )
    else:
        hierarchy_op = BuildSFDPHierarchy()
        refine_op = SFDPRefineCoarsestLevel(
            steps=steps,
            theta=theta,
            repulsive_exponent=repulsive_exponent,
        )
        prolongate_op = SFDPProlongateAndRefineLevels(
            steps=steps,
            theta=theta,
            repulsive_exponent=repulsive_exponent,
        )

    return Pipeline(
        [
            BuildSFDPGraph(),
            hierarchy_op,
            InitSFDPCoarsestPositions(),
            refine_op,
            prolongate_op,
            SFDPFinalizePositions(),
        ],
        name="sfdp_pipeline",
    )


def layout_sfdp_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 500,
    seed: int = 123,
    theta: float = _DEFAULT_THETA,
    repulsive_exponent: float = _DEFAULT_P,
    edge_weights: Optional[torch.Tensor] = None,
    direction: str = "TB",
    fidelity_mode: SFDPFidelityMode = False,
    fidelity_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Run the SFDP pipeline as a drop-in replacement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]`` used only for output
        scaling.
    steps : int, default=500
        Maximum number of spring-electrical iterations per level.
    seed : int, default=123
        Random seed for coarsening order, coarsest initialization, and
        prolongation noise.
    theta : float, default=0.6
        Barnes-Hut opening angle threshold.
    repulsive_exponent : float, default=-1.0
        SFDP repulsion exponent ``p``.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.
    direction : str, default="TB"
        Requested layout flow direction: ``TB``, ``BT``, ``LR``, or ``RL``.
    fidelity_mode : bool or str or None, default=False
        ``True`` or ``"graphviz"`` enables Graphviz-compatible matrix
        coarsening and sequential in-place updates. Existing behavior remains
        unchanged when this flag is ``False``.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If inputs are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")
    if edge_index.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError("edge_index must use an integer dtype.")
    if edge_weights is not None and edge_weights.shape[0] != edge_index.shape[1]:
        raise ValueError(
            f"edge_weights length {edge_weights.shape[0]} does not match "
            f"edge count {edge_index.shape[1]}"
        )
    _is_graphviz_fidelity_mode(fidelity_mode=fidelity_mode)
    if edge_index.numel() != 0:
        edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
        if int(edge_index_cpu.min().item()) < 0:
            raise ValueError("edge_index cannot contain negative node indices.")
        if int(edge_index_cpu.max().item()) >= num_nodes:
            raise ValueError("edge_index contains node indices outside num_nodes.")

    device = _layout_device(edge_index=edge_index, node_sizes=node_sizes)
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
        direction=direction,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_sfdp_pipeline(
        steps=steps,
        theta=theta,
        repulsive_exponent=repulsive_exponent,
        fidelity_mode=fidelity_mode,
        fidelity_dtype=fidelity_dtype,
    ).apply(problem, state, ctx)

    if final_state.pos is None:
        raise RuntimeError("SFDP pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_sfdp_pipeline", "layout_sfdp_pipeline"]
