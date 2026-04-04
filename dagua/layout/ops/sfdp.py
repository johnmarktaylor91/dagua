"""Composable SFDP operations.

This module contains the algorithm-specific operations and helper routines for
``layout_sfdp_pipeline``. All SFDP internals live here so the pipeline module can
compose only registered operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, List, Optional, Tuple

import torch

from dagua.layout.ops.base import Op, Repeat
from dagua.layout.ops.graph_utils import layout_device as _layout_device
from dagua.layout.ops.graph_utils import layout_extent as _layout_extent
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_EPSILON = 1.0e-9
_MIN_SPAN = 1.0e-6
_FORCE_SCALING = 0.2
_DEFAULT_P = -1.0
_DEFAULT_THETA = 0.6
_DEFAULT_TOLERANCE = 1.0e-3
_DEFAULT_STEP = 0.1
_MAX_QUADTREE_DEPTH = 10
_MIN_COARSE_SIZE = 4
_MIN_COARSEN_REDUCTION = 0.75
_BARNES_HUT_THRESHOLD = 10_000
_PROLONGATION_NOISE_SCALE = 1.0e-3
_PROLONGATION_SMOOTHING = 0.5
_REFINEMENT_K_DECAY = 0.75

_GRAPH_KEY = "sfdp_graphs"
_MAPPING_KEY = "sfdp_mappings"
_GENERATOR_KEY = "sfdp_generator"
_BASE_GRAPH_KEY = "sfdp_base_graph"
_IDEAL_LENGTH_KEY = "sfdp_ideal_length"
_SFDP_CURRENT_STEP_KEY = "sfdp_current_step"
_SFDP_PREVIOUS_FORCE_NORM_KEY = "sfdp_previous_force_norm"
_SFDP_FORCE_NORM_KEY = "sfdp_force_norm"


@dataclass
class GraphData:
    """Undirected weighted graph representation for SFDP.

    Parameters
    ----------
    num_nodes : int
        Number of nodes at the current multilevel stage.
    edge_index : torch.Tensor
        Unique undirected edges with shape ``[2, E]`` on CPU.
    edge_weight : torch.Tensor
        Non-negative edge weights with shape ``[E]`` on CPU.
    adjacency : list[list[tuple[int, float]]]
        Per-node neighbor lists storing ``(neighbor, weight)`` tuples.
    """

    num_nodes: int
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    adjacency: List[List[Tuple[int, float]]] = field(default_factory=list)


@dataclass
class QuadTreeNode:
    """Barnes-Hut quadtree node used for large-graph repulsion.

    Parameters
    ----------
    center : torch.Tensor
        Cell center with shape ``[2]``.
    half_width : float
        Half-width of the square cell.
    indices : list[int]
        Point indices stored under this node.
    level : int
        Depth within the quadtree.
    mass : float, default=0.0
        Aggregate number of points represented by the cell.
    center_of_mass : torch.Tensor, optional
        Mean position of points represented by the cell.
    children : list[QuadTreeNode], optional
        Child quadrants. Empty means the node is a leaf.
    """

    center: torch.Tensor
    half_width: float
    indices: List[int]
    level: int
    mass: float = 0.0
    center_of_mass: Optional[torch.Tensor] = None
    children: List["QuadTreeNode"] = field(default_factory=list)


def _normalize_positions(positions: torch.Tensor, extent: float) -> torch.Tensor:
    """Center and scale coordinates into a deterministic bounding box.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    extent : float
        Target half-width.

    Returns
    -------
    torch.Tensor
        Centered and scaled coordinates.
    """
    if positions.shape[0] <= 1:
        return torch.zeros_like(positions)

    centered = positions - positions.mean(dim=0, keepdim=True)
    span = float(centered.abs().max().item())
    if span < _MIN_SPAN:
        centered = centered.clone()
        centered[:, 0] = torch.linspace(
            -1.0,
            1.0,
            steps=positions.shape[0],
            device=positions.device,
        )
        span = float(centered.abs().max().item())
    return centered * (extent / max(span, _MIN_SPAN))


def _build_graph(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor] = None,
) -> GraphData:
    """Build an undirected weighted graph from a directed edge list.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.

    Returns
    -------
    GraphData
        CPU resident weighted graph representation.
    """
    adjacency: list[dict[int, float]] = [dict() for _ in range(num_nodes)]
    if edge_index.numel() > 0:
        edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
        weights_cpu = (
            torch.ones((edge_index.shape[1],), dtype=torch.float32)
            if edge_weights is None
            else edge_weights.detach().to(device="cpu", dtype=torch.float32)
        )
        for edge_id, (source, target) in enumerate(
            zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist())
        ):
            if source == target:
                continue
            lower = min(source, target)
            upper = max(source, target)
            adjacency[lower][upper] = adjacency[lower].get(upper, 0.0) + float(
                weights_cpu[edge_id].item()
            )

    edge_pairs: List[Tuple[int, int]] = []
    edge_weight_values: List[float] = []
    adjacency_lists: List[List[Tuple[int, float]]] = [[] for _ in range(num_nodes)]
    for source in range(num_nodes):
        for target, weight in sorted(adjacency[source].items()):
            edge_pairs.append((source, target))
            edge_weight_values.append(weight)
            adjacency_lists[source].append((target, weight))
            adjacency_lists[target].append((source, weight))

    if edge_pairs:
        edge_tensor = torch.tensor(edge_pairs, dtype=torch.long).transpose(0, 1).contiguous()
        weight_tensor = torch.tensor(edge_weight_values, dtype=torch.float32)
    else:
        edge_tensor = torch.empty((2, 0), dtype=torch.long)
        weight_tensor = torch.empty((0,), dtype=torch.float32)

    return GraphData(
        num_nodes=num_nodes,
        edge_index=edge_tensor,
        edge_weight=weight_tensor,
        adjacency=adjacency_lists,
    )


def _build_graph_from_mapping(
    graph: GraphData,
    fine_to_coarse: torch.Tensor,
    coarse_num_nodes: int,
) -> GraphData:
    """Aggregate a coarse graph from a fine-to-coarse assignment.

    Parameters
    ----------
    graph : GraphData
        Fine-level graph.
    fine_to_coarse : torch.Tensor
        Mapping from fine nodes to coarse nodes with shape ``[N_fine]``.
    coarse_num_nodes : int
        Number of coarse nodes.

    Returns
    -------
    GraphData
        Aggregated coarse graph.
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
        coarse_index = torch.tensor(list(coarse_edges.keys()), dtype=torch.long).transpose(0, 1)
        coarse_weight = torch.tensor(list(coarse_edges.values()), dtype=torch.float32)
    else:
        coarse_index = torch.empty((2, 0), dtype=torch.long)
        coarse_weight = torch.empty((0,), dtype=torch.float32)

    adjacency_lists: list[list[tuple[int, float]]] = [[] for _ in range(coarse_num_nodes)]
    for edge_id in range(coarse_index.shape[1]):
        source = int(coarse_index[0, edge_id].item())
        target = int(coarse_index[1, edge_id].item())
        weight = float(coarse_weight[edge_id].item())
        adjacency_lists[source].append((target, weight))
        adjacency_lists[target].append((source, weight))

    return GraphData(
        num_nodes=coarse_num_nodes,
        edge_index=coarse_index.contiguous(),
        edge_weight=coarse_weight,
        adjacency=adjacency_lists,
    )


def _heavy_edge_matching(
    graph: GraphData,
    generator: torch.Generator,
) -> Optional[tuple[torch.Tensor, GraphData]]:
    """Coarsen one level with random-order heavy-edge matching.

    Parameters
    ----------
    graph : GraphData
        Fine-level graph.
    generator : torch.Generator
        Deterministic CPU random generator.

    Returns
    -------
    tuple[torch.Tensor, GraphData] | None
        ``(fine_to_coarse, coarse_graph)`` when coarsening is worthwhile,
        otherwise ``None``.
    """
    num_nodes = graph.num_nodes
    if num_nodes < _MIN_COARSE_SIZE:
        return None

    order = torch.randperm(num_nodes, generator=generator).tolist()
    matched = torch.zeros(num_nodes, dtype=torch.bool)
    fine_to_coarse = torch.full((num_nodes,), -1, dtype=torch.long)
    coarse_node = 0

    for node in order:
        if bool(matched[node].item()):
            continue
        matched[node] = True
        partner = -1
        partner_weight = -1.0
        for neighbor, weight in graph.adjacency[node]:
            if bool(matched[neighbor].item()):
                continue
            if weight > partner_weight:
                partner = neighbor
                partner_weight = weight

        fine_to_coarse[node] = coarse_node
        if partner >= 0:
            matched[partner] = True
            fine_to_coarse[partner] = coarse_node
        coarse_node += 1

    coarse_num_nodes = coarse_node
    if coarse_num_nodes == graph.num_nodes:
        return None
    if coarse_num_nodes < _MIN_COARSE_SIZE:
        return None
    if coarse_num_nodes > int(_MIN_COARSEN_REDUCTION * float(graph.num_nodes)):
        return None

    coarse_graph = _build_graph_from_mapping(
        graph=graph,
        fine_to_coarse=fine_to_coarse,
        coarse_num_nodes=coarse_num_nodes,
    )
    return fine_to_coarse, coarse_graph


def _random_positions(num_nodes: int, generator: torch.Generator) -> torch.Tensor:
    """Initialize positions uniformly in the unit square.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    generator : torch.Generator
        Deterministic CPU random generator.

    Returns
    -------
    torch.Tensor
        Initial positions with shape ``[N, 2]``.
    """
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32)
    return torch.rand((num_nodes, 2), generator=generator, dtype=torch.float32)


def _average_edge_length(graph: GraphData, positions: torch.Tensor) -> float:
    """Compute the average current edge length for a graph.

    Parameters
    ----------
    graph : GraphData
        Graph whose edges define the length average.
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    float
        Average Euclidean edge length, or ``1.0`` when the graph is edgeless.
    """
    if graph.edge_index.numel() == 0:
        return 1.0

    delta = positions[graph.edge_index[0]] - positions[graph.edge_index[1]]
    lengths = torch.linalg.vector_norm(delta, dim=1)
    mean_length = float(lengths.mean().item())
    return max(mean_length, _MIN_SPAN)


def _spring_forces(
    graph: GraphData,
    positions: torch.Tensor,
    attractive_scale: float,
) -> torch.Tensor:
    """Compute linear spring forces along weighted graph edges.

    Parameters
    ----------
    graph : GraphData
        Graph whose edges contribute attractive forces.
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    attractive_scale : float
        Spring coefficient already incorporating the SFDP ``C`` and ``K`` terms.

    Returns
    -------
    torch.Tensor
        Attractive force tensor with shape ``[N, 2]``.
    """
    force = torch.zeros_like(positions)
    if graph.edge_index.numel() == 0:
        return force

    source = graph.edge_index[0]
    target = graph.edge_index[1]
    delta = positions[source] - positions[target]
    weight = graph.edge_weight.unsqueeze(1)
    edge_force = attractive_scale * weight * delta
    force.index_add_(0, source, -edge_force)
    force.index_add_(0, target, edge_force)
    return force


def _exact_repulsive_forces(
    positions: torch.Tensor,
    repulsive_scale: float,
    repulsive_exponent: float,
) -> torch.Tensor:
    """Compute exact all-pairs inverse-power repulsion.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    repulsive_scale : float
        Global repulsion multiplier.
    repulsive_exponent : float
        SFDP repulsion exponent ``p``.

    Returns
    -------
    torch.Tensor
        Repulsive force tensor with shape ``[N, 2]``.
    """
    if positions.shape[0] <= 1:
        return torch.zeros_like(positions)

    delta = positions[:, None, :] - positions[None, :, :]
    distance_sq = torch.sum(delta * delta, dim=-1).clamp_min(_EPSILON)
    distance = torch.sqrt(distance_sq)
    diagonal = torch.eye(positions.shape[0], dtype=torch.bool)
    distance = distance.masked_fill(diagonal, float("inf"))
    denominator = distance.pow(2.0 - repulsive_exponent).unsqueeze(-1)
    pairwise_force = repulsive_scale * delta / denominator
    pairwise_force = pairwise_force.masked_fill(diagonal.unsqueeze(-1), 0.0)
    return pairwise_force.sum(dim=1)


def _build_quadtree(positions: torch.Tensor) -> Optional[QuadTreeNode]:
    """Construct a Barnes-Hut quadtree for 2D repulsion.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    QuadTreeNode | None
        Root quadtree node, or ``None`` for empty inputs.
    """
    num_nodes = positions.shape[0]
    if num_nodes == 0:
        return None

    minimum = torch.amin(positions, dim=0)
    maximum = torch.amax(positions, dim=0)
    span = float(torch.max(maximum - minimum).item())
    half_width = max(span * 0.5 + _MIN_SPAN, _MIN_SPAN)
    center = (minimum + maximum) * 0.5
    indices = list(range(num_nodes))
    return _build_quadtree_node(
        positions=positions,
        center=center,
        half_width=half_width,
        indices=indices,
        level=0,
    )


def _build_quadtree_node(
    positions: torch.Tensor,
    center: torch.Tensor,
    half_width: float,
    indices: List[int],
    level: int,
) -> QuadTreeNode:
    """Recursively build one Barnes-Hut quadtree node.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    center : torch.Tensor
        Cell center with shape ``[2]``.
    half_width : float
        Half-width of the square cell.
    indices : list[int]
        Point indices assigned to the cell.
    level : int
        Current quadtree depth.

    Returns
    -------
    QuadTreeNode
        Fully populated quadtree node.
    """
    node = QuadTreeNode(center=center, half_width=half_width, indices=indices, level=level)
    if not indices:
        node.mass = 0.0
        node.center_of_mass = center.clone()
        return node

    index_tensor = torch.tensor(indices, dtype=torch.long)
    coords = positions[index_tensor]
    node.mass = float(len(indices))
    node.center_of_mass = coords.mean(dim=0)

    if len(indices) <= 1 or level >= _MAX_QUADTREE_DEPTH:
        return node

    child_half_width = half_width * 0.5
    quadrants: List[List[int]] = [[], [], [], []]
    for point_index in indices:
        point = positions[point_index]
        horizontal = 1 if float(point[0].item()) >= float(center[0].item()) else 0
        vertical = 1 if float(point[1].item()) >= float(center[1].item()) else 0
        quadrants[vertical * 2 + horizontal].append(point_index)

    offsets = [
        torch.tensor([-child_half_width, -child_half_width], dtype=torch.float32),
        torch.tensor([child_half_width, -child_half_width], dtype=torch.float32),
        torch.tensor([-child_half_width, child_half_width], dtype=torch.float32),
        torch.tensor([child_half_width, child_half_width], dtype=torch.float32),
    ]
    for quadrant_index, quadrant in enumerate(quadrants):
        if not quadrant:
            continue
        child_center = center + offsets[quadrant_index]
        child = _build_quadtree_node(
            positions=positions,
            center=child_center,
            half_width=child_half_width,
            indices=quadrant,
            level=level + 1,
        )
        node.children.append(child)

    return node


def _barnes_hut_force_for_index(
    node: QuadTreeNode,
    positions: torch.Tensor,
    index: int,
    theta: float,
    repulsive_scale: float,
    repulsive_exponent: float,
) -> torch.Tensor:
    """Evaluate the Barnes-Hut repulsive force on one node.

    Parameters
    ----------
    node : QuadTreeNode
        Current quadtree node.
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    index : int
        Query node index.
    theta : float
        Barnes-Hut opening angle threshold.
    repulsive_scale : float
        Global repulsion multiplier.
    repulsive_exponent : float
        SFDP repulsion exponent ``p``.

    Returns
    -------
    torch.Tensor
        Repulsive force vector with shape ``[2]``.
    """
    if node.mass <= 0.0 or node.center_of_mass is None:
        return torch.zeros(2, dtype=torch.float32)

    query = positions[index]
    if len(node.indices) == 1 and node.indices[0] == index:
        return torch.zeros(2, dtype=torch.float32)

    if node.children:
        delta = query - node.center_of_mass
        distance = float(torch.linalg.vector_norm(delta).item())
        width = node.half_width * 2.0
        if index not in node.indices and distance > _EPSILON and (width / distance) < theta:
            denominator = max(distance, _EPSILON) ** (2.0 - repulsive_exponent)
            return repulsive_scale * node.mass * delta / denominator

        force = torch.zeros(2, dtype=torch.float32)
        for child in node.children:
            force = force + _barnes_hut_force_for_index(
                node=child,
                positions=positions,
                index=index,
                theta=theta,
                repulsive_scale=repulsive_scale,
                repulsive_exponent=repulsive_exponent,
            )
        return force

    if len(node.indices) == 0:
        return torch.zeros(2, dtype=torch.float32)

    leaf_indices = [point_index for point_index in node.indices if point_index != index]
    if not leaf_indices:
        return torch.zeros(2, dtype=torch.float32)

    coords = positions[torch.tensor(leaf_indices, dtype=torch.long)]
    delta = query.unsqueeze(0) - coords
    distance = torch.linalg.vector_norm(delta, dim=1).clamp_min(_EPSILON)
    denominator = distance.pow(2.0 - repulsive_exponent).unsqueeze(1)
    return (repulsive_scale * delta / denominator).sum(dim=0)


def _barnes_hut_repulsive_forces(
    positions: torch.Tensor,
    repulsive_scale: float,
    repulsive_exponent: float,
    theta: float,
) -> torch.Tensor:
    """Compute Barnes-Hut approximate repulsion for large graphs.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    repulsive_scale : float
        Global repulsion multiplier.
    repulsive_exponent : float
        SFDP repulsion exponent ``p``.
    theta : float
        Barnes-Hut opening angle threshold.

    Returns
    -------
    torch.Tensor
        Repulsive force tensor with shape ``[N, 2]``.
    """
    root = _build_quadtree(positions=positions)
    if root is None:
        return torch.zeros_like(positions)

    force = torch.zeros_like(positions)
    for index in range(positions.shape[0]):
        force[index] = _barnes_hut_force_for_index(
            node=root,
            positions=positions,
            index=index,
            theta=theta,
            repulsive_scale=repulsive_scale,
            repulsive_exponent=repulsive_exponent,
        )
    return force


def _repulsive_forces(
    positions: torch.Tensor,
    repulsive_scale: float,
    repulsive_exponent: float,
    theta: float,
) -> torch.Tensor:
    """Choose exact or Barnes-Hut repulsion based on graph size.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    repulsive_scale : float
        Global repulsion multiplier.
    repulsive_exponent : float
        SFDP repulsion exponent ``p``.
    theta : float
        Barnes-Hut opening angle threshold.

    Returns
    -------
    torch.Tensor
        Repulsive force tensor with shape ``[N, 2]``.
    """
    if positions.shape[0] < _BARNES_HUT_THRESHOLD:
        return _exact_repulsive_forces(
            positions=positions,
            repulsive_scale=repulsive_scale,
            repulsive_exponent=repulsive_exponent,
        )
    return _barnes_hut_repulsive_forces(
        positions=positions,
        repulsive_scale=repulsive_scale,
        repulsive_exponent=repulsive_exponent,
        theta=theta,
    )


def _update_step(step: float, force_norm: float, previous_force_norm: float) -> float:
    """Apply the Graphviz-style adaptive cooling rule.

    Parameters
    ----------
    step : float
        Current step size.
    force_norm : float
        Current total force norm.
    previous_force_norm : float
        Previous iteration total force norm.

    Returns
    -------
    float
        Updated step size.
    """
    if force_norm >= previous_force_norm:
        return step * 0.90
    if force_norm > 0.95 * previous_force_norm:
        return step
    return step * 1.1


@register_op
@dataclass(frozen=True)
class SFDPSpringElectricalStep(Op):
    """Apply one force-displacement iteration for SFDP.

    Parameters
    ----------
    graph : GraphData
        Graph whose adjacency and weights define spring interactions.
    attractive_scale : float
        Precomputed attractive force coefficient for this level.
    repulsive_scale : float
        Precomputed repulsive force coefficient for this level.
    theta : float, default=0.6
        Barnes-Hut opening angle threshold.
    repulsive_exponent : float, default=-1.0
        SFDP repulsion exponent.
    """

    graph: GraphData
    attractive_scale: float
    repulsive_scale: float
    theta: float = _DEFAULT_THETA
    repulsive_exponent: float = _DEFAULT_P

    name: ClassVar[str] = "sfdp_spring_electrical_step"
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
        """Compute one iteration and move nodes with the current step size.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state containing the current positions and current step
            metadata.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with updated ``pos`` and latest force norm in extras.
        """
        del ctx
        if state.pos is None:
            raise ValueError("SFDPSpringElectricalStep requires state.pos to be set.")

        current_step = float(state.extras.get(_SFDP_CURRENT_STEP_KEY, _DEFAULT_STEP))

        attractive = _spring_forces(
            graph=self.graph,
            positions=state.pos,
            attractive_scale=self.attractive_scale,
        )
        repulsive = _repulsive_forces(
            positions=state.pos,
            repulsive_scale=self.repulsive_scale,
            repulsive_exponent=self.repulsive_exponent,
            theta=self.theta,
        )
        total_force = attractive + repulsive
        node_force_norm = torch.linalg.vector_norm(total_force, dim=1, keepdim=True)
        direction = torch.where(
            node_force_norm > _EPSILON,
            total_force / node_force_norm.clamp_min(_EPSILON),
            torch.zeros_like(total_force),
        )
        state.pos = state.pos + (current_step * direction)
        state.pos = state.pos - state.pos.mean(dim=0, keepdim=True)
        state.extras[_SFDP_FORCE_NORM_KEY] = float(torch.linalg.vector_norm(total_force).item())
        return state


@register_op
@dataclass(frozen=True)
class SFDPAdaptiveCool(Op):
    """Apply the SFDP adaptive step-size update.

    Parameters
    ----------
    adaptive_cooling : bool, default=True
        Whether to apply graphviz-style step-size adaptation.
    """

    adaptive_cooling: bool = True

    name: ClassVar[str] = "sfdp_adaptive_cool"
    category: ClassVar[OpCategory] = OpCategory.ANNEAL
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Adapt movement size for the next iteration and handle convergence.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state with force norms in ``extras``.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with updated step size metadata.
        """
        del problem
        if not self.adaptive_cooling:
            return state

        force_norm = float(state.extras.get(_SFDP_FORCE_NORM_KEY, 0.0))
        previous_force_norm = float(state.extras.get(_SFDP_PREVIOUS_FORCE_NORM_KEY, float("inf")))
        current_step = float(state.extras.get(_SFDP_CURRENT_STEP_KEY, _DEFAULT_STEP))

        if previous_force_norm < float("inf"):
            current_step = _update_step(
                step=current_step,
                force_norm=force_norm,
                previous_force_norm=previous_force_norm,
            )

        state.extras[_SFDP_CURRENT_STEP_KEY] = current_step
        state.extras[_SFDP_PREVIOUS_FORCE_NORM_KEY] = force_norm
        if current_step < _DEFAULT_TOLERANCE:
            state.converged = True
        return state


def _prolongate_positions(
    graph: GraphData,
    fine_to_coarse: torch.Tensor,
    coarse_positions: torch.Tensor,
    ideal_length: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Interpolate, smooth, and perturb positions for a finer graph level.

    Parameters
    ----------
    graph : GraphData
        Fine-level graph.
    fine_to_coarse : torch.Tensor
        Mapping from fine nodes to coarse nodes with shape ``[N_fine]``.
    coarse_positions : torch.Tensor
        Coarse positions with shape ``[N_coarse, 2]``.
    ideal_length : float
        Current ideal edge length ``K``.
    generator : torch.Generator
        Deterministic CPU random generator.

    Returns
    -------
    torch.Tensor
        Interpolated fine positions with shape ``[N_fine, 2]``.
    """
    positions = coarse_positions[fine_to_coarse].clone()
    if graph.num_nodes == 0:
        return positions

    smoothed = positions.clone()
    for node in range(graph.num_nodes):
        neighbors = graph.adjacency[node]
        if not neighbors:
            continue
        neighbor_indices = torch.tensor([neighbor for neighbor, _ in neighbors], dtype=torch.long)
        neighbor_mean = positions[neighbor_indices].mean(dim=0)
        smoothed[node] = (_PROLONGATION_SMOOTHING * positions[node]) + (
            (1.0 - _PROLONGATION_SMOOTHING) * neighbor_mean
        )

    groups: dict[int, List[int]] = {}
    for fine_index, coarse_index in enumerate(fine_to_coarse.tolist()):
        groups.setdefault(coarse_index, []).append(fine_index)

    noise_scale = ideal_length * _PROLONGATION_NOISE_SCALE
    for group in groups.values():
        for fine_index in group[1:]:
            noise = (torch.rand((2,), generator=generator, dtype=torch.float32) - 0.5) * noise_scale
            smoothed[fine_index] = smoothed[fine_index] + noise

    return smoothed


@register_op
@dataclass(frozen=True)
class BuildSFDPGraph(Op):
    """Build the undirected weighted graph representation for SFDP."""

    name: ClassVar[str] = "sfdp_build_graph"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build the SFDP graph and store it in extras.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs containing edge topology and weights.
        state : SolveState
            Mutable solve state receiving the graph in ``extras``.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with ``extras['sfdp_base_graph']`` populated.
        """
        del ctx
        state.extras[_BASE_GRAPH_KEY] = _build_graph(
            edge_index=problem.edge_index,
            num_nodes=problem.num_nodes,
            edge_weights=problem.edge_weights,
        )
        return state


@register_op
@dataclass(frozen=True)
class BuildSFDPHierarchy(Op):
    """Build the multilevel SFDP hierarchy via heavy-edge matching."""

    name: ClassVar[str] = "sfdp_coarsen_hierarchy"
    category: ClassVar[OpCategory] = OpCategory.COARSEN
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Iteratively coarsen until heavy-edge matching reduces no further.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs containing ``seed``.
        state : SolveState
            Mutable solve state. Reads ``extras['sfdp_base_graph']``.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with ``extras['sfdp_graphs']``, ``extras['sfdp_mappings']``
            and ``extras['sfdp_generator']`` populated.
        """
        del ctx
        generator = torch.Generator(device="cpu")
        generator.manual_seed(problem.seed)

        base_graph: GraphData = state.extras[_BASE_GRAPH_KEY]
        graphs: list[GraphData] = [base_graph]
        mappings: list[torch.Tensor] = []
        current_graph = base_graph

        while True:
            coarsened = _heavy_edge_matching(graph=current_graph, generator=generator)
            if coarsened is None:
                break
            fine_to_coarse, coarse_graph = coarsened
            mappings.append(fine_to_coarse)
            graphs.append(coarse_graph)
            current_graph = coarse_graph

        state.extras[_GRAPH_KEY] = graphs
        state.extras[_MAPPING_KEY] = mappings
        state.extras[_GENERATOR_KEY] = generator
        return state


@register_op
@dataclass(frozen=True)
class InitSFDPCoarsestPositions(Op):
    """Initialize random positions at the coarsest SFDP level."""

    name: ClassVar[str] = "sfdp_init_coarsest"
    category: ClassVar[OpCategory] = OpCategory.INIT
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("pos", "extras")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Seed positions and compute initial ideal edge length on the coarsest level.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused directly.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with ``pos`` set to coarsest-level random positions and
            ``extras['sfdp_ideal_length']`` computed.
        """
        del ctx
        graphs: List[GraphData] = state.extras[_GRAPH_KEY]
        generator: torch.Generator = state.extras[_GENERATOR_KEY]

        coarsest = graphs[-1]
        positions = _random_positions(num_nodes=coarsest.num_nodes, generator=generator)
        ideal_length = _average_edge_length(graph=coarsest, positions=positions)

        state.pos = positions
        state.extras[_IDEAL_LENGTH_KEY] = ideal_length
        return state


@register_op
@dataclass(frozen=True)
class SFDPRefineCoarsestLevel(Op):
    """Run spring-electrical layout on the coarsest level with adaptive cooling."""

    name: ClassVar[str] = "sfdp_refine_coarsest"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("pos", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("pos", "extras")

    steps: int = 500
    theta: float = _DEFAULT_THETA
    repulsive_exponent: float = _DEFAULT_P

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Refine the coarsest level and update state positions.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused directly.
        state : SolveState
            Mutable solve state with ``pos`` and SFDP extras.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with refined ``pos`` at the coarsest level.
        """
        graphs: List[GraphData] = state.extras[_GRAPH_KEY]
        coarsest = graphs[-1]

        if state.pos is None:
            raise ValueError("SFDPRefineCoarsestLevel requires state.pos to be set.")
        if self.steps == 0 or state.pos.shape[0] <= 1:
            return state

        ideal_length = float(state.extras[_IDEAL_LENGTH_KEY])
        attractive_scale = (_FORCE_SCALING ** ((2.0 - self.repulsive_exponent) / 3.0)) / max(
            ideal_length,
            _MIN_SPAN,
        )
        repulsive_scale = max(ideal_length, _MIN_SPAN) ** (1.0 - self.repulsive_exponent)
        state.extras[_SFDP_CURRENT_STEP_KEY] = _DEFAULT_STEP
        state.extras[_SFDP_PREVIOUS_FORCE_NORM_KEY] = float("inf")

        state = Repeat(
            n=self.steps,
            ops=[
                SFDPSpringElectricalStep(
                    graph=coarsest,
                    attractive_scale=attractive_scale,
                    repulsive_scale=repulsive_scale,
                    theta=self.theta,
                    repulsive_exponent=self.repulsive_exponent,
                ),
                SFDPAdaptiveCool(),
            ],
        ).apply(problem, state, ctx)
        state.converged = False
        return state


@register_op
@dataclass(frozen=True)
class SFDPProlongateAndRefineLevels(Op):
    """Prolongate from coarse to fine levels and refine after each step.

    Notes
    -----
    The hierarchy traversal remains explicit because each level changes the graph,
    fine-to-coarse mapping, and current target edge length before refinement.
    """

    name: ClassVar[str] = "sfdp_prolongate_and_refine"
    category: ClassVar[OpCategory] = OpCategory.PROLONG
    reads: ClassVar[Tuple[str, ...]] = ("pos", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("pos", "extras")

    steps: int = 500
    theta: float = _DEFAULT_THETA
    repulsive_exponent: float = _DEFAULT_P

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Walk the hierarchy from coarsest to finest, prolongate and refine.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused directly.
        state : SolveState
            Mutable solve state with ``pos`` at the coarsest level.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with ``pos`` at the finest (original) level.
        """
        graphs: List[GraphData] = state.extras[_GRAPH_KEY]
        mappings: list[torch.Tensor] = state.extras[_MAPPING_KEY]
        generator: torch.Generator = state.extras[_GENERATOR_KEY]
        ideal_length: float = state.extras[_IDEAL_LENGTH_KEY]

        positions = state.pos
        for level_index in range(len(mappings) - 1, -1, -1):
            ideal_length = max(ideal_length * _REFINEMENT_K_DECAY, _MIN_SPAN)
            fine_graph = graphs[level_index]
            positions = _prolongate_positions(
                graph=fine_graph,
                fine_to_coarse=mappings[level_index],
                coarse_positions=positions,
                ideal_length=ideal_length,
                generator=generator,
            )
            if positions.shape[0] > 1 and self.steps > 0:
                state.pos = positions
                state.extras[_SFDP_CURRENT_STEP_KEY] = _DEFAULT_STEP
                state.extras[_SFDP_PREVIOUS_FORCE_NORM_KEY] = float("inf")
                attractive_scale = (
                    _FORCE_SCALING ** ((2.0 - self.repulsive_exponent) / 3.0)
                ) / max(
                    ideal_length,
                    _MIN_SPAN,
                )
                repulsive_scale = max(ideal_length, _MIN_SPAN) ** (1.0 - self.repulsive_exponent)
                state = Repeat(
                    n=self.steps,
                    ops=[
                        SFDPSpringElectricalStep(
                            graph=fine_graph,
                            attractive_scale=attractive_scale,
                            repulsive_scale=repulsive_scale,
                            theta=self.theta,
                            repulsive_exponent=self.repulsive_exponent,
                        ),
                        SFDPAdaptiveCool(adaptive_cooling=False),
                    ],
                ).apply(problem, state, ctx)
                state.converged = False
                positions = state.pos

        state.pos = positions
        state.extras[_IDEAL_LENGTH_KEY] = ideal_length
        return state


@register_op
@dataclass(frozen=True)
class SFDPFinalizePositions(Op):
    """Normalize and cast final SFDP positions."""

    name: ClassVar[str] = "sfdp_finalize_positions"
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
        """Center, normalize, scale, and cast positions like classic SFDP.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs used for device and extent.
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
            raise ValueError("SFDPFinalizePositions requires state.pos to be set.")

        device = _layout_device(
            edge_index=problem.edge_index,
            node_sizes=problem.node_sizes,
        )

        extent = _layout_extent(num_nodes=problem.num_nodes, node_sizes=problem.node_sizes)
        normalized = _normalize_positions(positions=state.pos, extent=extent)
        state.pos = normalized.to(dtype=torch.float32, device=device)
        return state


__all__ = [
    "GraphData",
    "QuadTreeNode",
    "BuildSFDPGraph",
    "BuildSFDPHierarchy",
    "InitSFDPCoarsestPositions",
    "SFDPSpringElectricalStep",
    "SFDPAdaptiveCool",
    "SFDPRefineCoarsestLevel",
    "SFDPProlongateAndRefineLevels",
    "SFDPFinalizePositions",
]
