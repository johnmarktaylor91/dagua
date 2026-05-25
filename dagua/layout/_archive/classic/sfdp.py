"""Hu-style spring-electrical multilevel layout used by Graphviz SFDP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

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
_BARNES_HUT_THRESHOLD = 45
_PROLONGATION_NOISE_SCALE = 1.0e-3
_PROLONGATION_SMOOTHING = 0.5
_REFINEMENT_K_DECAY = 0.75


@dataclass
class _GraphData:
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
    adjacency: list[list[tuple[int, float]]] = field(default_factory=list)


@dataclass
class _QuadTreeNode:
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
    children : list[_QuadTreeNode], optional
        Child quadrants. Empty means the node is a leaf.
    """

    center: torch.Tensor
    half_width: float
    indices: list[int]
    level: int
    mass: float = 0.0
    center_of_mass: Optional[torch.Tensor] = None
    children: list["_QuadTreeNode"] = field(default_factory=list)


def _layout_device(
    edge_index: torch.Tensor,
    node_sizes: Optional[torch.Tensor],
) -> torch.device:
    """Choose the device used for the returned tensor.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor.

    Returns
    -------
    torch.device
        Output device for the final layout tensor.
    """
    if edge_index.numel() > 0:
        return edge_index.device
    if node_sizes is not None:
        return node_sizes.device
    return torch.device("cpu")


def _layout_extent(num_nodes: int, node_sizes: Optional[torch.Tensor]) -> float:
    """Estimate a stable output extent from graph size and node sizes.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor.

    Returns
    -------
    float
        Target half-width after normalization.
    """
    if node_sizes is None or node_sizes.numel() == 0:
        return max(float(max(num_nodes, 1)) ** 0.5 * 5.0, 1.0)

    max_size = float(node_sizes.to(dtype=torch.float32, device="cpu").max().item())
    return max(max_size * max(float(max(num_nodes, 1)) ** 0.5, 1.0) * 2.0, 1.0)


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


def _principal_component_rotate(positions: torch.Tensor) -> torch.Tensor:
    """Rotate coordinates onto Graphviz SFDP's principal-component frame.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Centered positions rotated so the major principal axis is horizontal.
    """
    if positions.shape[0] <= 1:
        return torch.zeros_like(positions)

    centered = positions - positions.mean(dim=0, keepdim=True)
    covariance = centered.transpose(0, 1) @ centered
    cross = covariance[0, 1]

    if abs(float(cross.item())) <= _EPSILON:
        axis = torch.tensor([0.0, 1.0], dtype=positions.dtype, device=positions.device)
    else:
        xx = covariance[0, 0]
        yy = covariance[1, 1]
        discriminant = torch.sqrt((xx * xx) + (4.0 * cross * cross) - (2.0 * xx * yy) + (yy * yy))
        axis_x = -(-xx + yy - discriminant) / (2.0 * cross)
        axis = torch.stack([axis_x, torch.ones((), dtype=positions.dtype, device=positions.device)])
        axis = axis / torch.linalg.vector_norm(axis).clamp_min(_EPSILON)

    x_rotated = centered[:, 0] * axis[0] + centered[:, 1] * axis[1]
    y_rotated = -centered[:, 0] * axis[1] + centered[:, 1] * axis[0]
    return torch.stack([x_rotated, y_rotated], dim=1)


def _reflect_flow_axis(positions: torch.Tensor, direction: str) -> torch.Tensor:
    """Reflect positions along the axis used by the requested layout direction.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    direction : str
        Layout direction: ``TB``, ``BT``, ``LR``, or ``RL``.

    Returns
    -------
    torch.Tensor
        Reflected position tensor with shape ``[N, 2]``.
    """
    reflected = positions.clone()
    if direction in {"LR", "RL"}:
        reflected[:, 0] = -reflected[:, 0]
    else:
        reflected[:, 1] = -reflected[:, 1]
    return reflected


def _edge_flow_score(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    direction: str,
) -> float:
    """Score how strongly edges advance in the requested flow direction.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Directed edge tensor with shape ``[2, E]``.
    direction : str
        Layout direction: ``TB``, ``BT``, ``LR``, or ``RL``.

    Returns
    -------
    float
        Mean signed edge advance along the requested flow axis. Larger is better.
    """
    if edge_index.numel() == 0 or positions.shape[0] <= 1:
        return 0.0

    edges = edge_index.to(device=positions.device, dtype=torch.long)
    source = edges[0]
    target = edges[1]
    delta = positions[target] - positions[source]
    if direction in {"LR", "RL"}:
        advance = delta[:, 0]
    else:
        advance = delta[:, 1]
    if direction in {"BT", "RL"}:
        advance = -advance
    return float(advance.mean().item())


def _orient_positions_to_flow(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    direction: str,
) -> torch.Tensor:
    """Canonicalize SFDP's rotation-invariant output for directed graph flow.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Directed edge tensor with shape ``[2, E]``.
    direction : str
        Layout direction: ``TB``, ``BT``, ``LR``, or ``RL``.

    Returns
    -------
    torch.Tensor
        Oriented position tensor with shape ``[N, 2]``.
    """
    if positions.shape[0] <= 1 or edge_index.numel() == 0:
        return positions

    normalized_direction = direction if direction in {"TB", "BT", "LR", "RL"} else "TB"
    rotated = _principal_component_rotate(positions)
    candidates = [
        positions,
        _reflect_flow_axis(positions, normalized_direction),
        rotated,
        _reflect_flow_axis(rotated, normalized_direction),
    ]
    return max(
        candidates,
        key=lambda candidate: _edge_flow_score(
            positions=candidate,
            edge_index=edge_index,
            direction=normalized_direction,
        ),
    )


def _validate_inputs(
    edge_index: torch.Tensor,
    num_nodes: int,
    steps: int,
    edge_weights: Optional[torch.Tensor],
) -> None:
    """Validate public SFDP arguments.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    steps : int
        Maximum number of force iterations per level.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.

    Returns
    -------
    None
        Raises ``ValueError`` when an input is invalid.
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
    if edge_index.numel() == 0:
        return
    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    if int(edge_index_cpu.min().item()) < 0:
        raise ValueError("edge_index cannot contain negative node indices.")
    if int(edge_index_cpu.max().item()) >= num_nodes:
        raise ValueError("edge_index contains node indices outside num_nodes.")


def _build_graph(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor] = None,
) -> _GraphData:
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
    _GraphData
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

    edge_pairs: list[tuple[int, int]] = []
    edge_weight_values: list[float] = []
    adjacency_lists: list[list[tuple[int, float]]] = [[] for _ in range(num_nodes)]
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

    return _GraphData(
        num_nodes=num_nodes,
        edge_index=edge_tensor,
        edge_weight=weight_tensor,
        adjacency=adjacency_lists,
    )


def _build_graph_from_mapping(
    graph: _GraphData,
    fine_to_coarse: torch.Tensor,
    coarse_num_nodes: int,
) -> _GraphData:
    """Aggregate a coarse graph from a fine-to-coarse assignment.

    Parameters
    ----------
    graph : _GraphData
        Fine-level graph.
    fine_to_coarse : torch.Tensor
        Mapping from fine nodes to coarse nodes with shape ``[N_fine]``.
    coarse_num_nodes : int
        Number of coarse nodes.

    Returns
    -------
    _GraphData
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

    return _GraphData(
        num_nodes=coarse_num_nodes,
        edge_index=coarse_index.contiguous(),
        edge_weight=coarse_weight,
        adjacency=adjacency_lists,
    )


def _heavy_edge_matching(
    graph: _GraphData,
    generator: torch.Generator,
) -> Optional[tuple[torch.Tensor, _GraphData]]:
    """Coarsen one level with random-order heavy-edge matching.

    Parameters
    ----------
    graph : _GraphData
        Fine-level graph.
    generator : torch.Generator
        Deterministic CPU random generator.

    Returns
    -------
    tuple[torch.Tensor, _GraphData] or None
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
    if coarse_num_nodes == num_nodes:
        return None
    if coarse_num_nodes < _MIN_COARSE_SIZE:
        return None
    if coarse_num_nodes > int(_MIN_COARSEN_REDUCTION * float(num_nodes)):
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


def _average_edge_length(graph: _GraphData, positions: torch.Tensor) -> float:
    """Compute the average current edge length for a graph.

    Parameters
    ----------
    graph : _GraphData
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
    graph: _GraphData,
    positions: torch.Tensor,
    attractive_scale: float,
) -> torch.Tensor:
    """Compute linear spring forces along weighted graph edges.

    Parameters
    ----------
    graph : _GraphData
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
    distance = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
    # Graphviz spring_electrical.c scales attraction by the current edge
    # distance before normalizing the total per-node force vector.
    edge_force = attractive_scale * weight * delta * distance
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
    denominator = distance.pow(1.0 - repulsive_exponent).unsqueeze(-1)
    pairwise_force = repulsive_scale * delta / denominator
    pairwise_force = pairwise_force.masked_fill(diagonal.unsqueeze(-1), 0.0)
    return pairwise_force.sum(dim=1)


def _build_quadtree(positions: torch.Tensor) -> Optional[_QuadTreeNode]:
    """Construct a Barnes-Hut quadtree for 2D repulsion.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    _QuadTreeNode or None
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
    indices: list[int],
    level: int,
) -> _QuadTreeNode:
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
    _QuadTreeNode
        Fully populated quadtree node.
    """
    node = _QuadTreeNode(center=center, half_width=half_width, indices=indices, level=level)
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
    quadrants: list[list[int]] = [[], [], [], []]
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
    node: _QuadTreeNode,
    positions: torch.Tensor,
    index: int,
    theta: float,
    repulsive_scale: float,
    repulsive_exponent: float,
) -> torch.Tensor:
    """Evaluate the Barnes-Hut repulsive force on one node.

    Parameters
    ----------
    node : _QuadTreeNode
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
            denominator = max(distance, _EPSILON) ** (1.0 - repulsive_exponent)
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
    denominator = distance.pow(1.0 - repulsive_exponent).unsqueeze(1)
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


def _spring_electrical_layout(
    graph: _GraphData,
    positions: torch.Tensor,
    ideal_length: float,
    steps: int,
    theta: float,
    repulsive_exponent: float,
    adaptive_cooling: bool,
    step_init: float,
) -> torch.Tensor:
    """Run Hu's spring-electrical fixed-step refinement on one graph level.

    Parameters
    ----------
    graph : _GraphData
        Current graph level.
    positions : torch.Tensor
        Initial positions with shape ``[N, 2]``.
    ideal_length : float
        Ideal edge length ``K``.
    steps : int
        Maximum number of iterations.
    theta : float
        Barnes-Hut opening angle threshold.
    repulsive_exponent : float
        SFDP repulsion exponent ``p``.
    adaptive_cooling : bool
        Whether to adapt the step size from force progress. When disabled,
        Graphviz still applies fixed ``0.90`` cooling.
    step_init : float
        Initial fixed movement length per iteration.

    Returns
    -------
    torch.Tensor
        Refined positions with shape ``[N, 2]``.
    """
    if positions.shape[0] <= 1 or steps == 0:
        return positions

    attractive_scale = (_FORCE_SCALING ** ((2.0 - repulsive_exponent) / 3.0)) / max(
        ideal_length,
        _MIN_SPAN,
    )
    repulsive_scale = max(ideal_length, _MIN_SPAN) ** (1.0 - repulsive_exponent)
    current_step = step_init
    previous_force_norm = float("inf")

    for _ in range(steps):
        if current_step < _DEFAULT_TOLERANCE:
            break

        attractive = _spring_forces(
            graph=graph,
            positions=positions,
            attractive_scale=attractive_scale,
        )
        repulsive = _repulsive_forces(
            positions=positions,
            repulsive_scale=repulsive_scale,
            repulsive_exponent=repulsive_exponent,
            theta=theta,
        )
        total_force = attractive + repulsive
        node_force_norm = torch.linalg.vector_norm(total_force, dim=1, keepdim=True)
        direction = torch.where(
            node_force_norm > _EPSILON,
            total_force / node_force_norm.clamp_min(_EPSILON),
            torch.zeros_like(total_force),
        )
        positions = positions + (current_step * direction)

        force_norm = float(node_force_norm.sum().item())
        if not adaptive_cooling:
            current_step *= 0.90
        elif previous_force_norm < float("inf"):
            current_step = _update_step(
                step=current_step,
                force_norm=force_norm,
                previous_force_norm=previous_force_norm,
            )
        previous_force_norm = force_norm

    return positions


def _prolongate_positions(
    graph: _GraphData,
    fine_to_coarse: torch.Tensor,
    coarse_positions: torch.Tensor,
    ideal_length: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Interpolate, smooth, and perturb positions for a finer graph level.

    Parameters
    ----------
    graph : _GraphData
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

    groups: dict[int, list[int]] = {}
    for fine_index, coarse_index in enumerate(fine_to_coarse.tolist()):
        groups.setdefault(coarse_index, []).append(fine_index)

    noise_scale = ideal_length * _PROLONGATION_NOISE_SCALE
    for group in groups.values():
        for fine_index in group[1:]:
            noise = (torch.rand((2,), generator=generator, dtype=torch.float32) - 0.5) * noise_scale
            smoothed[fine_index] = smoothed[fine_index] + noise

    return smoothed


def layout_sfdp(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 500,
    seed: int = 123,
    theta: float = _DEFAULT_THETA,
    repulsive_exponent: float = _DEFAULT_P,
    edge_weights: Optional[torch.Tensor] = None,
    direction: str = "TB",
) -> torch.Tensor:
    """Lay out a graph with Hu's multilevel spring-electrical SFDP algorithm.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor used only for output scaling.
    steps : int, default=500
        Maximum number of spring-electrical iterations per level.
    seed : int, default=123
        Random seed for coarsening order, coarsest initialization, and
        prolongation noise.
    theta : float, default=0.6
        Barnes-Hut opening-angle threshold.
    repulsive_exponent : float, default=-1.0
        Exponent ``p`` controlling the inverse-power repulsive term.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]`` forwarded into SFDP's
        weighted spring graph construction.
    direction : str, default="TB"
        Requested layout flow direction: ``TB``, ``BT``, ``LR``, or ``RL``.

    Returns
    -------
    torch.Tensor
        Final positions with shape ``[N, 2]``.
    """
    _validate_inputs(
        edge_index=edge_index,
        num_nodes=num_nodes,
        steps=steps,
        edge_weights=edge_weights,
    )

    device = _layout_device(edge_index=edge_index, node_sizes=node_sizes)
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32, device=device)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    base_graph = _build_graph(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weights=edge_weights,
    )
    graphs: list[_GraphData] = [base_graph]
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

    positions = _random_positions(num_nodes=graphs[-1].num_nodes, generator=generator)
    ideal_length = _average_edge_length(graph=graphs[-1], positions=positions)
    positions = _spring_electrical_layout(
        graph=graphs[-1],
        positions=positions,
        ideal_length=ideal_length,
        steps=steps,
        theta=theta,
        repulsive_exponent=repulsive_exponent,
        adaptive_cooling=True,
        step_init=_DEFAULT_STEP,
    )

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
        positions = _spring_electrical_layout(
            graph=fine_graph,
            positions=positions,
            ideal_length=ideal_length,
            steps=steps,
            theta=theta,
            repulsive_exponent=repulsive_exponent,
            adaptive_cooling=False,
            step_init=_DEFAULT_STEP,
        )

    oriented = _orient_positions_to_flow(
        positions=positions,
        edge_index=edge_index,
        direction=direction,
    )
    extent = _layout_extent(num_nodes=num_nodes, node_sizes=node_sizes)
    normalized = _normalize_positions(positions=oriented, extent=extent)
    return normalized.to(dtype=torch.float32, device=device)
