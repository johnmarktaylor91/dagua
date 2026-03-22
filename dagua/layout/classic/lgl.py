"""igraph LGL translated into a conservative pure-PyTorch solver.

The port preserves the layered BFS growth strategy from igraph's Large Graph
Layout: place vertices one shell at a time, then run an FR-style relaxation on
all vertices placed so far.
"""

from __future__ import annotations

import math
import random
from collections import deque
from typing import Optional

import torch

_MIN_DISTANCE = 1.0e-12
_CONVERGENCE_EPSILON = 1.0e-5


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


def _validate_inputs(
    edge_index: torch.Tensor,
    num_nodes: int,
    maxiter: int,
    coolexp: float,
    edge_weights: Optional[torch.Tensor],
) -> None:
    """Validate the public LGL arguments.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    maxiter : int
        Maximum refinement iterations per layer.
    coolexp : float
        Cooling exponent.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.

    Returns
    -------
    None
        Raises ``ValueError`` when an input is invalid.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if maxiter < 0:
        raise ValueError("maxiter must be non-negative.")
    if coolexp <= 0.0:
        raise ValueError("coolexp must be positive.")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
            )

    if edge_index.numel() == 0:
        return

    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    min_index = int(edge_index_cpu.min().item())
    max_index = int(edge_index_cpu.max().item())
    if min_index < 0:
        raise ValueError("edge_index cannot contain negative node indices.")
    if max_index >= num_nodes:
        raise ValueError("edge_index contains node indices outside [0, num_nodes).")


def _build_undirected_graph(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor] = None,
) -> tuple[list[list[int]], list[tuple[int, int]], Optional[list[float]]]:
    """Build the undirected adjacency list and spring list.

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
    tuple[list[list[int]], list[tuple[int, int]], Optional[list[float]]]
        Symmetric adjacency list for BFS growth and a spring list that keeps
        every non-self edge occurrence, plus matching spring weights when
        provided.
    """
    adjacency_sets = [set() for _ in range(num_nodes)]
    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    if edge_index.numel() == 0:
        return [[] for _ in range(num_nodes)], [], None

    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    sources = edge_index_cpu[0].tolist()
    targets = edge_index_cpu[1].tolist()
    weight_list = (
        edge_weights.detach().to(device="cpu", dtype=torch.float64).tolist()
        if edge_weights is not None
        else None
    )
    for edge_idx, (source, target) in enumerate(zip(sources, targets)):
        if source == target:
            continue
        lower = min(source, target)
        upper = max(source, target)
        adjacency_sets[lower].add(upper)
        adjacency_sets[upper].add(lower)
        # LGL's force model reacts to parallel and reciprocal edges, so we keep
        # multiplicity in the spring list even though BFS only needs unique
        # undirected neighbors.
        edges.append((lower, upper))
        if weight_list is not None:
            weights.append(float(weight_list[edge_idx]))

    adjacency = [sorted(neighbors) for neighbors in adjacency_sets]
    return adjacency, edges, weights if weight_list is not None else None


def _initialize_positions(num_nodes: int, radius: float, seed: int) -> torch.Tensor:
    """Create the random LGL starting positions.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    radius : float
        Characteristic radius of the drawing box.
    seed : int
        Random seed for the initial placement.

    Returns
    -------
    torch.Tensor
        Initial positions with shape ``[N, 2]`` and dtype ``float64``.
    """
    rng = random.Random(seed)
    data = [[rng.uniform(-radius, radius), rng.uniform(-radius, radius)] for _ in range(num_nodes)]
    return torch.tensor(data, dtype=torch.float64)


def _normalize(vector: torch.Tensor) -> torch.Tensor:
    """Return a safely normalized 2D vector.

    Parameters
    ----------
    vector : torch.Tensor
        Vector with shape ``[2]``.

    Returns
    -------
    torch.Tensor
        Unit vector, or zeros when the input magnitude is tiny.
    """
    norm = float(torch.linalg.norm(vector).item())
    if norm <= _MIN_DISTANCE:
        return torch.zeros_like(vector)
    return vector / norm


def _bfs_layers(
    adjacency: list[list[int]],
    root: int,
) -> tuple[list[list[int]], list[int]]:
    """Run one BFS and group vertices by distance from the root.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency list.
    root : int
        BFS root vertex.

    Returns
    -------
    tuple[list[list[int]], list[int]]
        Layer list and one parent index per node. Unreached nodes keep parent
        ``-1`` and are omitted from the layer list.
    """
    num_nodes = len(adjacency)
    parents = [-1] * num_nodes
    distance = [-1] * num_nodes
    layers: list[list[int]] = []
    queue: deque[int] = deque([root])
    parents[root] = root
    distance[root] = 0

    while queue:
        node = queue.popleft()
        depth = distance[node]
        while len(layers) <= depth:
            layers.append([])
        layers[depth].append(node)
        for neighbor in adjacency[node]:
            if distance[neighbor] != -1:
                continue
            parents[neighbor] = node
            distance[neighbor] = depth + 1
            queue.append(neighbor)
    return layers, parents


def _harmonic_number(num_terms: int) -> float:
    """Compute the harmonic number used by LGL shell placement.

    Parameters
    ----------
    num_terms : int
        Number of terms to include.

    Returns
    -------
    float
        Harmonic number, clamped away from zero.
    """
    if num_terms <= 0:
        return 1.0
    return sum(1.0 / float(index) for index in range(1, num_terms + 1))


def _build_cell_buckets(
    active_nodes: list[int],
    positions: torch.Tensor,
    cellsize: float,
) -> dict[tuple[int, int], list[int]]:
    """Bucket active vertices into a sparse integer grid.

    Parameters
    ----------
    active_nodes : list[int]
        Nodes currently participating in refinement.
    positions : torch.Tensor
        Position matrix with shape ``[N, 2]``.
    cellsize : float
        Repulsion cell size.

    Returns
    -------
    dict[tuple[int, int], list[int]]
        Sparse cell-to-node mapping.
    """
    buckets: dict[tuple[int, int], list[int]] = {}
    safe_cellsize = max(cellsize, _MIN_DISTANCE)
    for node in active_nodes:
        x_value = float(positions[node, 0].item())
        y_value = float(positions[node, 1].item())
        key = (
            int(math.floor(x_value / safe_cellsize)),
            int(math.floor(y_value / safe_cellsize)),
        )
        buckets.setdefault(key, []).append(node)
    return buckets


def _run_refinement(
    positions: torch.Tensor,
    active_nodes: list[int],
    active_edges: list[tuple[int, int]],
    active_edge_weights: Optional[list[float]],
    maxiter: int,
    maxdelta: float,
    coolexp: float,
    frk: float,
    repulserad: float,
    cellsize: float,
) -> None:
    """Run one LGL FR-style relaxation phase in place.

    Parameters
    ----------
    positions : torch.Tensor
        Position matrix with shape ``[N, 2]``.
    active_nodes : list[int]
        Nodes placed so far.
    active_edges : list[tuple[int, int]]
        Edges whose endpoints are both active.
    active_edge_weights : list[float], optional
        Edge weights for ``active_edges`` in the same order.
    maxiter : int
        Maximum iterations for this phase.
    maxdelta : float
        Maximum movement magnitude at the start of the cooling schedule.
    coolexp : float
        Cooling exponent.
    frk : float
        Fruchterman-Reingold spacing constant.
    repulserad : float
        Natural cutoff scale for repulsion.
    cellsize : float
        Sparse-grid cell size.

    Returns
    -------
    None
        ``positions`` is updated in place.
    """
    if not active_nodes or maxiter == 0:
        return

    node_count = positions.shape[0]
    weight_tensor = (
        torch.tensor(active_edge_weights, dtype=torch.float64)
        if active_edge_weights is not None
        else None
    )
    for iteration in range(maxiter):
        temperature = maxdelta * (((maxiter - iteration) / float(maxiter)) ** coolexp)
        forces = torch.zeros((node_count, 2), dtype=torch.float64)
        maxchange = 0.0

        if active_edges:
            source = torch.tensor([edge[0] for edge in active_edges], dtype=torch.long)
            target = torch.tensor([edge[1] for edge in active_edges], dtype=torch.long)
            delta = positions[source] - positions[target]
            distance = torch.linalg.norm(delta, dim=1)
            mask = distance > _MIN_DISTANCE
            if bool(mask.any().item()):
                masked_distance = distance[mask]
                direction = delta[mask] / masked_distance.unsqueeze(1)
                magnitude = masked_distance.square() / max(frk, _MIN_DISTANCE)
                if weight_tensor is not None:
                    magnitude = magnitude * weight_tensor[mask]
                contribution = direction * magnitude.unsqueeze(1)
                forces.index_add_(0, source[mask], -contribution)
                forces.index_add_(0, target[mask], contribution)

        buckets = _build_cell_buckets(
            active_nodes=active_nodes,
            positions=positions,
            cellsize=cellsize,
        )
        sorted_cells = sorted(buckets)
        for cell in sorted_cells:
            nodes_here = buckets[cell]
            for offset_y in (-1, 0, 1):
                for offset_x in (-1, 0, 1):
                    neighbor_cell = (cell[0] + offset_x, cell[1] + offset_y)
                    if neighbor_cell not in buckets or neighbor_cell < cell:
                        continue
                    nodes_there = buckets[neighbor_cell]
                    if neighbor_cell == cell:
                        for left_index in range(len(nodes_here)):
                            for right_index in range(left_index + 1, len(nodes_here)):
                                left = nodes_here[left_index]
                                right = nodes_here[right_index]
                                delta = positions[left] - positions[right]
                                distance = float(torch.linalg.norm(delta).item())
                                if distance >= cellsize:
                                    continue
                                safe_distance = max(distance, _MIN_DISTANCE)
                                direction = delta / safe_distance
                                magnitude = (frk * frk) * (
                                    (1.0 / safe_distance)
                                    - ((safe_distance * safe_distance) / repulserad)
                                )
                                contribution = direction * magnitude
                                forces[left] += contribution
                                forces[right] -= contribution
                    else:
                        for left in nodes_here:
                            for right in nodes_there:
                                delta = positions[left] - positions[right]
                                distance = float(torch.linalg.norm(delta).item())
                                if distance >= cellsize:
                                    continue
                                safe_distance = max(distance, _MIN_DISTANCE)
                                direction = delta / safe_distance
                                magnitude = (frk * frk) * (
                                    (1.0 / safe_distance)
                                    - ((safe_distance * safe_distance) / repulserad)
                                )
                                contribution = direction * magnitude
                                forces[left] += contribution
                                forces[right] -= contribution

        for node in active_nodes:
            movement = forces[node]
            magnitude = float(torch.linalg.norm(movement).item())
            if magnitude > temperature and magnitude > _MIN_DISTANCE:
                movement = movement * (temperature / magnitude)
            positions[node] += movement
            maxchange = max(
                maxchange,
                abs(float(movement[0].item())),
                abs(float(movement[1].item())),
            )

        if maxchange < _CONVERGENCE_EPSILON:
            break


def layout_lgl(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: int = 42,
    maxiter: int = 150,
    maxdelta: Optional[float] = None,
    area: Optional[float] = None,
    coolexp: float = 1.5,
    repulserad: Optional[float] = None,
    cellsize: Optional[float] = None,
    root: Optional[int] = None,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Lay out a graph with the igraph Large Graph Layout algorithm.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Unused placeholder kept for interface compatibility.
    seed : int, default=42
        Random seed controlling root selection and stochastic shell placement.
    maxiter : int, default=150
        Maximum refinement iterations per BFS layer.
    maxdelta : float, optional
        Maximum initial movement magnitude. Defaults to ``num_nodes``.
    area : float, optional
        Layout area. Defaults to ``num_nodes ** 2``.
    coolexp : float, default=1.5
        Power-law cooling exponent.
    repulserad : float, optional
        Modified FR repulsion cutoff. Defaults to ``area * num_nodes``.
    cellsize : float, optional
        Sparse-grid cell size. Defaults to ``area ** 0.25``.
    root : int, optional
        BFS root. When omitted, a seed-controlled random vertex is chosen.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]`` that scales the spring
        force for each non-self edge occurrence.

    Returns
    -------
    torch.Tensor
        Final positions with shape ``[N, 2]`` and dtype ``float32``.
    """
    _validate_inputs(
        edge_index=edge_index,
        num_nodes=num_nodes,
        maxiter=maxiter,
        coolexp=coolexp,
        edge_weights=edge_weights,
    )
    device = _layout_device(edge_index=edge_index, node_sizes=node_sizes)
    del node_sizes
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)

    resolved_maxdelta = float(num_nodes) if maxdelta is None else maxdelta
    resolved_area = float(num_nodes * num_nodes) if area is None else area
    resolved_repulserad = (
        resolved_area * float(max(num_nodes, 1)) if repulserad is None else repulserad
    )
    resolved_cellsize = resolved_area**0.25 if cellsize is None else cellsize
    if resolved_maxdelta < 0.0 or resolved_area <= 0.0 or resolved_repulserad <= 0.0:
        raise ValueError("maxdelta must be non-negative, and area/repulserad must be positive.")
    if resolved_cellsize <= 0.0:
        raise ValueError("cellsize must be positive.")

    adjacency, spring_edges, spring_edge_weights = _build_undirected_graph(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weights=edge_weights,
    )
    rng = random.Random(seed)
    root_node = rng.randrange(num_nodes) if root is None else root
    if root_node < 0 or root_node >= num_nodes:
        raise ValueError("root must lie in [0, num_nodes).")

    frk = math.sqrt(resolved_area / float(max(num_nodes, 1)))
    radius = math.sqrt(resolved_area / math.pi)
    positions = _initialize_positions(num_nodes=num_nodes, radius=radius, seed=seed)
    positions[root_node] = 0.0

    layers, parents = _bfs_layers(adjacency=adjacency, root=root_node)
    if not layers:
        return positions.to(dtype=torch.float32, device=device)

    placed = torch.zeros(num_nodes, dtype=torch.bool)
    placed[root_node] = True
    active_edges: list[tuple[int, int]] = []
    active_edge_weights: Optional[list[float]] = [] if spring_edge_weights is not None else None
    edge_active = [False] * len(spring_edges)
    incident_edge_indices: list[list[int]] = [[] for _ in range(num_nodes)]
    for edge_idx, (source, target) in enumerate(spring_edges):
        incident_edge_indices[source].append(edge_idx)
        incident_edge_indices[target].append(edge_idx)
    shell_scale = radius / _harmonic_number(max(len(layers) - 1, 1))

    for layer_index in range(len(layers) - 1):
        current_layer = layers[layer_index]
        next_layer = layers[layer_index + 1]
        if not next_layer:
            continue

        active_nodes = torch.nonzero(placed, as_tuple=False).view(-1)
        center_of_mass = (
            positions[active_nodes].mean(dim=0)
            if active_nodes.numel() > 0
            else torch.zeros(2, dtype=torch.float64)
        )
        center_direction = _normalize(center_of_mass)
        next_depth = layer_index + 1
        total_layer_children = len(next_layer)
        layer_child_index = 0

        for parent in current_layer:
            children = [node for node in next_layer if parents[node] == parent]
            if not children:
                continue

            if parent == root_node:
                parent_direction = torch.zeros(2, dtype=torch.float64)
            else:
                parent_of_parent = parents[parent]
                parent_direction = _normalize(positions[parent] - positions[parent_of_parent])

            anchor = positions[parent] + center_direction + parent_direction
            for child in children:
                if next_depth == 1:
                    angle = (2.0 * math.pi * layer_child_index) / float(
                        max(total_layer_children, 1)
                    )
                    direction = torch.tensor(
                        [math.cos(angle), math.sin(angle)],
                        dtype=torch.float64,
                    )
                else:
                    direction = torch.tensor(
                        [rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)],
                        dtype=torch.float64,
                    )
                    direction = _normalize(direction)
                    if float(torch.linalg.norm(direction).item()) <= _MIN_DISTANCE:
                        direction = torch.tensor([1.0, 0.0], dtype=torch.float64)

                offset = direction * (shell_scale / float(max(next_depth, 1)))
                positions[child] = anchor + offset
                placed[child] = True
                layer_child_index += 1

                for edge_idx in incident_edge_indices[child]:
                    if edge_active[edge_idx]:
                        continue
                    edge = spring_edges[edge_idx]
                    source, target = edge
                    other = target if source == child else source
                    if not bool(placed[other].item()):
                        continue
                    edge_active[edge_idx] = True
                    active_edges.append(edge)
                    if active_edge_weights is not None:
                        active_edge_weights.append(float(spring_edge_weights[edge_idx]))

        refinement_nodes = torch.nonzero(placed, as_tuple=False).view(-1).tolist()
        _run_refinement(
            positions=positions,
            active_nodes=refinement_nodes,
            active_edges=active_edges,
            active_edge_weights=active_edge_weights,
            maxiter=maxiter,
            maxdelta=resolved_maxdelta,
            coolexp=coolexp,
            frk=frk,
            repulserad=resolved_repulserad,
            cellsize=resolved_cellsize,
        )

    # Disconnected vertices are left at their random initialization, matching
    # igraph's documented limitation for LGL on disconnected graphs.
    return positions.to(dtype=torch.float32, device=device)
