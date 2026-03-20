"""FM^3-style multilevel force-directed layout.

This module implements a conservative multilevel layout routine inspired by
FM^3: solar-system-style coarsening, a coarse FR initialization, and Barnes-Hut
repulsion during refinement at each uncoarsening level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from dagua.layout.classic.fr import layout_fr

_MIN_DISTANCE = 1.0e-3
_COARSE_TARGET = 50
_MAX_TREE_DEPTH = 10
_COOLING_FACTOR = 0.99


@dataclass
class _QuadCell:
    """Recursive quad-tree cell used for Barnes-Hut repulsion."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    indices: list[int]
    center_of_mass: torch.Tensor = field(
        default_factory=lambda: torch.zeros(2, dtype=torch.float32)
    )
    mass: float = 0.0
    children: list["_QuadCell"] = field(default_factory=list)


def _layout_device(
    edge_index: torch.Tensor,
    node_sizes: Optional[torch.Tensor],
) -> torch.device:
    """Choose the output device for the layout.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor.

    Returns
    -------
    torch.device
        Output device.
    """
    if edge_index.numel() > 0:
        return edge_index.device
    if node_sizes is not None:
        return node_sizes.device
    return torch.device("cpu")


def _layout_extent(num_nodes: int, node_sizes: Optional[torch.Tensor]) -> float:
    """Estimate a stable drawing extent.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Optional node sizes.

    Returns
    -------
    float
        Target half-width.
    """
    if node_sizes is None or node_sizes.numel() == 0:
        return max(float(max(num_nodes, 1)) ** 0.5 * 5.0, 1.0)

    max_size = float(node_sizes.to(dtype=torch.float32, device="cpu").max().item())
    return max(max_size * max(float(max(num_nodes, 1)) ** 0.5, 1.0) * 2.0, 1.0)


def _normalize_positions(positions: torch.Tensor, extent: float) -> torch.Tensor:
    """Center and scale coordinates into a stable drawing box.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    extent : float
        Target half-width.

    Returns
    -------
    torch.Tensor
        Normalized coordinates.
    """
    if positions.shape[0] <= 1:
        return torch.zeros_like(positions)

    centered = positions - positions.mean(dim=0, keepdim=True)
    span = centered.abs().max().clamp(min=1.0)
    return centered * (extent / span)


def _fr_ideal_length(area: float, num_nodes: int) -> float:
    """Compute the FR ideal edge length for the current refinement level.

    Parameters
    ----------
    area : float
        Target drawing area for the current level.
    num_nodes : int
        Number of nodes on the current level.

    Returns
    -------
    float
        Ideal FR length ``k = sqrt(area / N)``.
    """
    return max((area / max(num_nodes, 1)) ** 0.5, _MIN_DISTANCE)


def _unique_edges(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Convert an edge tensor into unique undirected edges.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    torch.Tensor
        Unique undirected edge tensor with shape ``[2, E_u]``.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")

    seen: set[tuple[int, int]] = set()
    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    for source, target in zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist()):
        if source < 0 or source >= num_nodes or target < 0 or target >= num_nodes:
            raise ValueError("edge_index contains a node index outside [0, num_nodes).")
        if source == target:
            continue
        seen.add((min(source, target), max(source, target)))

    if not seen:
        return torch.empty((2, 0), dtype=torch.long)

    ordered = sorted(seen)
    return torch.tensor(ordered, dtype=torch.long).transpose(0, 1).contiguous()


def _build_undirected_adjacency(edge_index: torch.Tensor, num_nodes: int) -> list[list[int]]:
    """Build an undirected adjacency list from unique edges.

    Parameters
    ----------
    edge_index : torch.Tensor
        Unique undirected edges with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    list[list[int]]
        One sorted neighbor list per node.
    """
    adjacency_sets = [set() for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return [[] for _ in range(num_nodes)]

    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    for source, target in zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist()):
        adjacency_sets[source].add(target)
        adjacency_sets[target].add(source)

    return [sorted(neighbors) for neighbors in adjacency_sets]


def _solar_system_mapping(adjacency: list[list[int]]) -> torch.Tensor:
    """Cluster nodes into sun-planet-moon groups for coarsening.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency list.

    Returns
    -------
    torch.Tensor
        Mapping from fine nodes to coarse node ids.
    """
    num_nodes = len(adjacency)
    degrees = [len(neighbors) for neighbors in adjacency]
    order = sorted(range(num_nodes), key=lambda node: (-degrees[node], node))

    mapping = torch.full((num_nodes,), fill_value=-1, dtype=torch.long)
    cluster_id = 0
    for sun in order:
        if mapping[sun] >= 0:
            continue
        mapping[sun] = cluster_id
        direct_members = [sun]
        for planet in adjacency[sun]:
            if mapping[planet] < 0:
                mapping[planet] = cluster_id
                direct_members.append(planet)
        for planet in direct_members[1:]:
            for moon in adjacency[planet]:
                if mapping[moon] < 0:
                    mapping[moon] = cluster_id
        cluster_id += 1

    for node in range(num_nodes):
        if mapping[node] < 0:
            mapping[node] = cluster_id
            cluster_id += 1

    return mapping


def _coarsen_edges(edge_index: torch.Tensor, mapping: torch.Tensor) -> torch.Tensor:
    """Project fine edges onto the coarse graph.

    Parameters
    ----------
    edge_index : torch.Tensor
        Fine-level unique undirected edges.
    mapping : torch.Tensor
        Fine-to-coarse node mapping.

    Returns
    -------
    torch.Tensor
        Unique undirected coarse edges.

    Notes
    -----
    OGDF preserves distance-weighted coarse edges during galaxy coarsening.
    This reimplementation still collapses them by deduplicating endpoint
    pairs, which remains an intentional simplification called out by the task.
    """
    coarse_edges: set[tuple[int, int]] = set()
    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    mapping_cpu = mapping.to(device="cpu", dtype=torch.long)
    for source, target in zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist()):
        coarse_source = int(mapping_cpu[source].item())
        coarse_target = int(mapping_cpu[target].item())
        if coarse_source == coarse_target:
            continue
        coarse_edges.add((min(coarse_source, coarse_target), max(coarse_source, coarse_target)))

    if not coarse_edges:
        return torch.empty((2, 0), dtype=torch.long)

    ordered = sorted(coarse_edges)
    return torch.tensor(ordered, dtype=torch.long).transpose(0, 1).contiguous()


def _hierarchy(
    edge_index: torch.Tensor, num_nodes: int
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Build the multilevel graph hierarchy.

    Parameters
    ----------
    edge_index : torch.Tensor
        Original edge tensor.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    tuple[list[torch.Tensor], list[torch.Tensor]]
        Edge tensors per level, and fine-to-coarse mappings between levels.
    """
    edges_per_level = [_unique_edges(edge_index, num_nodes)]
    mappings: list[torch.Tensor] = []
    current_nodes = num_nodes

    while current_nodes > _COARSE_TARGET:
        adjacency = _build_undirected_adjacency(edges_per_level[-1], current_nodes)
        mapping = _solar_system_mapping(adjacency)
        coarse_nodes = int(mapping.max().item()) + 1 if mapping.numel() > 0 else 0
        if coarse_nodes >= current_nodes:
            break
        mappings.append(mapping)
        coarse_edges = _coarsen_edges(edges_per_level[-1], mapping)
        edges_per_level.append(coarse_edges)
        current_nodes = coarse_nodes

    return edges_per_level, mappings


def _bounding_box(positions: torch.Tensor) -> tuple[float, float, float, float]:
    """Compute an axis-aligned bounding box around node positions.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    tuple[float, float, float, float]
        ``(x_min, x_max, y_min, y_max)`` bounds.
    """
    if positions.shape[0] == 0:
        return (-1.0, 1.0, -1.0, 1.0)

    x_min = float(positions[:, 0].min().item())
    x_max = float(positions[:, 0].max().item())
    y_min = float(positions[:, 1].min().item())
    y_max = float(positions[:, 1].max().item())
    padding = max(x_max - x_min, y_max - y_min, 1.0) * 0.05
    return (x_min - padding, x_max + padding, y_min - padding, y_max + padding)


def _build_quadtree(
    positions: torch.Tensor,
    indices: list[int],
    bounds: tuple[float, float, float, float],
    depth: int,
) -> _QuadCell:
    """Recursively build a quad-tree for Barnes-Hut repulsion.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    indices : list[int]
        Node indices inside the current bounds.
    bounds : tuple[float, float, float, float]
        ``(x_min, x_max, y_min, y_max)`` for the current cell.
    depth : int
        Remaining recursion depth.

    Returns
    -------
    _QuadCell
        Root of the constructed subtree.
    """
    x_min, x_max, y_min, y_max = bounds
    cell = _QuadCell(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, indices=indices)
    if not indices:
        return cell

    points = positions[indices]
    cell.mass = float(len(indices))
    cell.center_of_mass = points.mean(dim=0)
    if len(indices) <= 1 or depth <= 0:
        return cell

    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    quadrants = [
        (x_min, x_mid, y_min, y_mid),
        (x_mid, x_max, y_min, y_mid),
        (x_min, x_mid, y_mid, y_max),
        (x_mid, x_max, y_mid, y_max),
    ]
    buckets = [[] for _ in range(4)]
    for index in indices:
        x = float(positions[index, 0].item())
        y = float(positions[index, 1].item())
        quadrant = 0
        if x >= x_mid:
            quadrant += 1
        if y >= y_mid:
            quadrant += 2
        buckets[quadrant].append(index)

    for quadrant_bounds, bucket in zip(quadrants, buckets):
        if bucket:
            cell.children.append(_build_quadtree(positions, bucket, quadrant_bounds, depth - 1))

    return cell


def _repulsion_from_cell(
    node_index: int,
    point: torch.Tensor,
    cell: _QuadCell,
    theta: float,
    ideal_length: float,
) -> torch.Tensor:
    """Approximate the repulsion contribution of one quad-tree cell.

    Parameters
    ----------
    node_index : int
        Node for which the force is being evaluated.
    point : torch.Tensor
        Node position with shape ``[2]``.
    cell : _QuadCell
        Current quad-tree cell.
    theta : float
        Barnes-Hut opening angle threshold.
    ideal_length : float
        FR ideal edge length ``k`` for the current level.

    Returns
    -------
    torch.Tensor
        Repulsion vector contributed by the cell.
    """
    if cell.mass == 0.0:
        return torch.zeros(2, dtype=point.dtype, device=point.device)
    if len(cell.indices) == 1 and cell.indices[0] == node_index:
        return torch.zeros(2, dtype=point.dtype, device=point.device)

    delta = point - cell.center_of_mass.to(device=point.device)
    distance = torch.linalg.norm(delta).clamp(min=_MIN_DISTANCE)
    width = max(cell.x_max - cell.x_min, cell.y_max - cell.y_min)
    if not cell.children or width / float(distance.item()) < theta:
        del ideal_length
        return delta * (cell.mass / distance.square())

    total = torch.zeros(2, dtype=point.dtype, device=point.device)
    for child in cell.children:
        total = total + _repulsion_from_cell(
            node_index,
            point,
            child,
            theta,
            ideal_length,
        )
    return total


def _exact_repulsion(positions: torch.Tensor) -> torch.Tensor:
    """Compute exact all-pairs repulsion matching OGDF's f_rep_u_on_v.

    OGDF formula: f_rep_scalar(d) = 1/d, applied as (v-u)/||v-u||^2.
    No ideal_length in the formula — pure 1/d^2 directional force.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Repulsive force per node with shape ``[N, 2]``.
    """
    n = positions.shape[0]
    if n <= 1:
        return torch.zeros_like(positions)
    delta = positions.unsqueeze(1) - positions.unsqueeze(0)  # [N, N, 2]
    dist = torch.cdist(positions, positions).clamp(min=_MIN_DISTANCE)  # [N, N]
    # OGDF: scalar = 1/d / d = 1/d^2, applied to (v-u) direction
    factor = 1.0 / (dist * dist)  # [N, N]
    factor.fill_diagonal_(0.0)
    return (delta * factor.unsqueeze(2)).sum(dim=1)


def _barnes_hut_repulsion(
    positions: torch.Tensor,
    theta: float,
    ideal_length: float,
) -> torch.Tensor:
    """Compute repulsive forces using a Barnes-Hut quad-tree.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    theta : float
        Barnes-Hut opening angle threshold.
    ideal_length : float
        FR ideal edge length ``k`` for the current level.

    Returns
    -------
    torch.Tensor
        Repulsive force per node.
    """
    root = _build_quadtree(
        positions,
        list(range(int(positions.shape[0]))),
        _bounding_box(positions),
        _MAX_TREE_DEPTH,
    )
    forces = torch.zeros_like(positions)
    for node_index in range(int(positions.shape[0])):
        forces[node_index] = _repulsion_from_cell(
            node_index,
            positions[node_index],
            root,
            theta,
            ideal_length,
        )
    return forces


def _attractive_force(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    ideal_length: float,
) -> torch.Tensor:
    """Compute exact attractive forces along graph edges.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Unique undirected edges with shape ``[2, E]``.
    ideal_length : float
        FR ideal edge length ``k`` for the current level.

    Returns
    -------
    torch.Tensor
        Attractive force per node.
    """
    forces = torch.zeros_like(positions)
    if edge_index.numel() == 0:
        return forces

    src = edge_index[0].to(device=positions.device, dtype=torch.long)
    dst = edge_index[1].to(device=positions.device, dtype=torch.long)
    delta = positions[dst] - positions[src]
    distances = torch.linalg.norm(delta, dim=1).clamp(min=_MIN_DISTANCE)
    denominator = max(ideal_length**3, _MIN_DISTANCE)
    edge_force = delta * (distances / denominator).unsqueeze(1)
    forces.index_add_(0, src, edge_force)
    forces.index_add_(0, dst, -edge_force)
    return forces


def _refine_level(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    steps: int,
    theta: float,
    area: float,
    cooling_factor: float = _COOLING_FACTOR,
) -> torch.Tensor:
    """Run Barnes-Hut force refinement on one hierarchy level.

    Parameters
    ----------
    positions : torch.Tensor
        Initial coordinates with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Unique undirected edges with shape ``[2, E]``.
    steps : int
        Number of refinement iterations.
    theta : float
        Barnes-Hut opening angle threshold.
    area : float
        Target drawing area for the current level.
    cooling_factor : float, default=0.9
        Multiplicative temperature decay applied after each refinement step.

    Returns
    -------
    torch.Tensor
        Refined positions.
    """
    refined = positions.clone()
    if steps <= 0:
        return refined

    ideal_length = _fr_ideal_length(area, int(refined.shape[0]))
    temperature = ideal_length
    use_exact = int(refined.shape[0]) <= 500  # match OGDF's exact path for small graphs
    for _ in range(steps):
        repulsive = (
            _exact_repulsion(refined)
            if use_exact
            else _barnes_hut_repulsion(refined, theta, ideal_length)
        )
        attractive = _attractive_force(refined, edge_index, ideal_length)
        displacement = repulsive + attractive
        norm = torch.linalg.norm(displacement, dim=1, keepdim=True).clamp(min=_MIN_DISTANCE)
        limited_step = torch.minimum(norm, torch.full_like(norm, temperature))
        refined = refined + (displacement / norm) * limited_step
        temperature = max(temperature * cooling_factor, _MIN_DISTANCE)
    return refined


def layout_fmmm(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 100,
    seed: int = 42,
) -> torch.Tensor:
    """Lay out a graph with an FM^3-style multilevel force-directed scheme.

    Reference
    ---------
    Hachul and Junger, "Drawing Large Graphs with a Potential-Field-Based
    Multilevel Algorithm" (2004).

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Optional node sizes used only for final scaling.
    steps : int, default=100
        Total refinement budget across hierarchy levels.
    seed : int, default=42
        Random seed for coarse initialization and uncoarsening jitter.

    Returns
    -------
    torch.Tensor
        Node positions with shape ``[N, 2]``.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")

    device = _layout_device(edge_index, node_sizes)
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32, device=device)

    extent = _layout_extent(num_nodes, node_sizes)
    refinement_area = (2.0 * extent) ** 2
    edges_per_level, mappings = _hierarchy(edge_index, num_nodes)
    coarsest_edges = edges_per_level[-1]
    coarsest_nodes = num_nodes if not mappings else int(mappings[-1].max().item()) + 1
    positions = layout_fr(
        coarsest_edges,
        coarsest_nodes,
        node_sizes=None,
        steps=max(50, steps),
        seed=seed,
    ).to(dtype=torch.float32)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    level_budget = max(10, steps // max(len(edges_per_level), 1))

    for level in range(len(mappings) - 1, -1, -1):
        mapping = mappings[level]
        fine_nodes = int(mapping.shape[0])
        fine_positions = positions[mapping].clone()
        jitter_scale = max(extent / max(fine_nodes, 1) ** 0.5, 0.05) * 0.1
        jitter = (
            torch.randn((fine_nodes, 2), generator=generator, dtype=torch.float32) * jitter_scale
        )
        fine_positions = fine_positions + jitter
        positions = _refine_level(
            fine_positions,
            edges_per_level[level],
            level_budget,
            theta=1.0,
            area=refinement_area,
        )

    if not mappings:
        positions = _refine_level(
            positions,
            edges_per_level[0],
            level_budget,
            theta=1.0,
            area=refinement_area,
        )

    return _normalize_positions(positions.to(device), extent).to(dtype=torch.float32, device=device)
