"""OGDF-style Bertault force layout pipeline without runtime delegation."""

from __future__ import annotations

import ctypes
import math
from typing import Optional

import numpy as np
import torch

_OGDF_RAND_BUCKETS = 1000
_OGDF_RAND_SCALE = 10.0
_SECTION_COUNT = 8
_DEFAULT_EDGE_LENGTH = 50.0
_EDGE_FORCE_LIMIT_FACTOR = 4.0
_ZONE_RADIUS_DIVISOR = 3.0


def _edge_list(edge_index: torch.Tensor) -> list[tuple[int, int]]:
    """Return an input-order edge list.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.

    Returns
    -------
    list[tuple[int, int]]
        Edge endpoints in the same order as the input tensor.
    """
    if edge_index.numel() == 0:
        return []
    edge_index_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    return [
        (int(source), int(target))
        for source, target in zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist())
        if int(source) != int(target)
    ]


def _adjacency(edges: list[tuple[int, int]], num_nodes: int) -> list[list[int]]:
    """Build OGDF-style undirected adjacency in insertion order.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Edge list.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    list[list[int]]
        Adjacent node indices for each node.
    """
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    for source, target in edges:
        adjacency[source].append(target)
        adjacency[target].append(source)
    return adjacency


def _ogdf_initial_positions(num_nodes: int, seed: int) -> np.ndarray:
    """Generate the standalone runner's seeded GraphAttributes positions.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    seed : int
        Seed forwarded to ``std::srand`` in ``scripts/ogdf_runner.cpp``.

    Returns
    -------
    np.ndarray
        Initial coordinates with shape ``[N, 2]`` and dtype ``float64``.
    """
    libc = ctypes.CDLL("libc.so.6")
    libc.srand(ctypes.c_uint(seed))
    positions = np.empty((num_nodes, 2), dtype=np.float64)
    for node in range(num_nodes):
        positions[node, 0] = float(libc.rand() % _OGDF_RAND_BUCKETS) / _OGDF_RAND_SCALE
        positions[node, 1] = float(libc.rand() % _OGDF_RAND_BUCKETS) / _OGDF_RAND_SCALE
    return positions


def _required_length(positions: np.ndarray, edges: list[tuple[int, int]]) -> float:
    """Return OGDF Bertault's default desired edge length.

    Parameters
    ----------
    positions : np.ndarray
        Current coordinates with shape ``[N, 2]``.
    edges : list[tuple[int, int]]
        Edge list.

    Returns
    -------
    float
        Average Euclidean edge length, or ``0.0`` for edgeless graphs.
    """
    total = 0.0
    for source, target in edges:
        x_diff = positions[source, 0] - positions[target, 0]
        y_diff = positions[source, 1] - positions[target, 1]
        total += math.sqrt((x_diff * x_diff) + (y_diff * y_diff))
    return total / float(len(edges) if edges else 1)


def _section_for_vector(x_diff: float, y_diff: float) -> int:
    """Return OGDF's octant section number for a vector.

    Parameters
    ----------
    x_diff : float
        Vector x component.
    y_diff : float
        Vector y component.

    Returns
    -------
    int
        Section in ``1..8``.
    """
    if x_diff >= 0.0:
        if y_diff >= 0.0:
            return 1 if x_diff >= y_diff else 2
        return 8 if x_diff >= -y_diff else 7
    if y_diff >= 0.0:
        return 4 if -x_diff >= y_diff else 3
    return 5 if -x_diff >= -y_diff else 6


def _wrapped_section(section: int) -> int:
    """Wrap an integer section into OGDF's ``1..8`` range.

    Parameters
    ----------
    section : int
        Possibly out-of-range section.

    Returns
    -------
    int
        Wrapped section in ``1..8``.
    """
    wrapped = 1 + ((section - 1) % _SECTION_COUNT)
    if wrapped <= 0:
        wrapped += _SECTION_COUNT
    return wrapped


def _projection(positions: np.ndarray, node: int, edge: tuple[int, int]) -> tuple[float, float]:
    """Compute OGDF's slope-based projection of a node onto an edge line.

    Parameters
    ----------
    positions : np.ndarray
        Current coordinates with shape ``[N, 2]``.
    node : int
        Node to project.
    edge : tuple[int, int]
        Edge endpoints.

    Returns
    -------
    tuple[float, float]
        Projection coordinates.
    """
    source, target = edge
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.float64(positions[source, 1] - positions[target, 1]) / np.float64(
            positions[source, 0] - positions[target, 0]
        )
        perpendicular = np.float64(-1.0) / slope
        intercept = np.float64(positions[source, 1]) - (slope * np.float64(positions[source, 0]))
        perpendicular_intercept = np.float64(positions[node, 1]) - (
            perpendicular * np.float64(positions[node, 0])
        )
        x_coord = (perpendicular_intercept - intercept) / (slope - perpendicular)
        y_coord = (slope * x_coord) + intercept
    return float(x_coord), float(y_coord)


def _projection_is_on_edge(
    positions: np.ndarray,
    projection: tuple[float, float],
    edge: tuple[int, int],
) -> bool:
    """Return whether a projection falls inside an edge bounding box.

    Parameters
    ----------
    positions : np.ndarray
        Current coordinates with shape ``[N, 2]``.
    projection : tuple[float, float]
        Projection coordinates.
    edge : tuple[int, int]
        Edge endpoints.

    Returns
    -------
    bool
        ``True`` when both projected coordinates are between the endpoints.
    """
    source, target = edge
    x_coord, y_coord = projection
    x_good = (x_coord <= positions[source, 0] and x_coord >= positions[target, 0]) or (
        x_coord >= positions[source, 0] and x_coord <= positions[target, 0]
    )
    y_good = (y_coord <= positions[source, 1] and y_coord >= positions[target, 1]) or (
        y_coord >= positions[source, 1] and y_coord <= positions[target, 1]
    )
    return bool(x_good and y_good)


def _apply_node_forces(
    positions: np.ndarray,
    node: int,
    adjacency: list[list[int]],
    req_length: float,
    forces: np.ndarray,
) -> None:
    """Accumulate OGDF node-node repulsive and attractive forces.

    Parameters
    ----------
    positions : np.ndarray
        Current coordinates with shape ``[N, 2]``.
    node : int
        Node receiving forces.
    adjacency : list[list[int]]
        Undirected adjacency list.
    req_length : float
        Desired edge length.
    forces : np.ndarray
        Mutable force array with shape ``[N, 2]``.

    Returns
    -------
    None
        ``forces`` is updated in place.
    """
    for other in range(positions.shape[0]):
        if other == node:
            continue
        x_diff = positions[node, 0] - positions[other, 0]
        y_diff = positions[node, 1] - positions[other, 1]
        distance = math.sqrt((x_diff * x_diff) + (y_diff * y_diff))
        if distance == 0.0:
            continue
        scale = (req_length / distance) * (req_length / distance)
        forces[node, 0] += scale * x_diff
        forces[node, 1] += scale * y_diff

    for other in adjacency[node]:
        x_diff = positions[node, 0] - positions[other, 0]
        y_diff = positions[node, 1] - positions[other, 1]
        distance = math.sqrt((x_diff * x_diff) + (y_diff * y_diff))
        if req_length == 0.0:
            continue
        forces[node, 0] += -(distance / req_length) * x_diff
        forces[node, 1] += -(distance / req_length) * y_diff


def _apply_edge_force(
    positions: np.ndarray,
    node: int,
    edge: tuple[int, int],
    projection: tuple[float, float],
    req_length: float,
    forces: np.ndarray,
) -> None:
    """Accumulate OGDF node-edge repulsive force.

    Parameters
    ----------
    positions : np.ndarray
        Current coordinates with shape ``[N, 2]``.
    node : int
        Node repelled from the edge.
    edge : tuple[int, int]
        Edge endpoints receiving the opposite force.
    projection : tuple[float, float]
        Projection of ``node`` onto the edge line.
    req_length : float
        Desired edge length.
    forces : np.ndarray
        Mutable force array with shape ``[N, 2]``.

    Returns
    -------
    None
        ``forces`` is updated in place.
    """
    x_diff = positions[node, 0] - projection[0]
    y_diff = positions[node, 1] - projection[1]
    distance = math.sqrt((x_diff * x_diff) + (y_diff * y_diff))
    limit = _EDGE_FORCE_LIMIT_FACTOR * req_length
    if distance <= limit and distance > 0.0:
        fx = ((limit - distance) * (limit - distance) * x_diff) / distance
        fy = ((limit - distance) * (limit - distance) * y_diff) / distance
        forces[node, 0] += fx
        forces[node, 1] += fy
        source, target = edge
        forces[source, 0] -= fx
        forces[source, 1] -= fy
        forces[target, 0] -= fx
        forces[target, 1] -= fy


def _update_on_edge_sections(
    positions: np.ndarray,
    node: int,
    edge: tuple[int, int],
    projection: tuple[float, float],
    sections: np.ndarray,
) -> None:
    """Update OGDF section radii for an in-segment projection.

    Parameters
    ----------
    positions : np.ndarray
        Current coordinates with shape ``[N, 2]``.
    node : int
        Node being compared to the edge.
    edge : tuple[int, int]
        Edge endpoints.
    projection : tuple[float, float]
        Projection of ``node`` onto the edge segment.
    sections : np.ndarray
        Mutable radius table with shape ``[N, 9]``.

    Returns
    -------
    None
        ``sections`` is updated in place.
    """
    source, target = edge
    x_diff = projection[0] - positions[node, 0]
    y_diff = projection[1] - positions[node, 1]
    section = _section_for_vector(x_diff, y_diff)
    max_radius = math.sqrt((x_diff * x_diff) + (y_diff * y_diff)) / _ZONE_RADIUS_DIVISOR
    for raw_section in range(section - 2, section + 3):
        sections[node, _wrapped_section(raw_section)] = min(
            sections[node, _wrapped_section(raw_section)],
            max_radius,
        )
    for raw_section in range(section + 2, section + 7):
        wrapped = _wrapped_section(raw_section)
        sections[source, wrapped] = min(sections[source, wrapped], max_radius)
        sections[target, wrapped] = min(sections[target, wrapped], max_radius)


def _update_outside_edge_sections(
    positions: np.ndarray,
    node: int,
    edge: tuple[int, int],
    sections: np.ndarray,
) -> None:
    """Update OGDF section radii for an outside-segment projection.

    Parameters
    ----------
    positions : np.ndarray
        Current coordinates with shape ``[N, 2]``.
    node : int
        Node being compared to the edge.
    edge : tuple[int, int]
        Edge endpoints.
    sections : np.ndarray
        Mutable radius table with shape ``[N, 9]``.

    Returns
    -------
    None
        ``sections`` is updated in place.
    """
    source, target = edge
    dist_source = math.sqrt(
        ((positions[node, 0] - positions[source, 0]) ** 2)
        + ((positions[node, 1] - positions[source, 1]) ** 2)
    )
    dist_target = math.sqrt(
        ((positions[node, 0] - positions[target, 0]) ** 2)
        + ((positions[node, 1] - positions[target, 1]) ** 2)
    )
    for section in range(1, _SECTION_COUNT + 1):
        sections[node, section] = min(sections[node, section], min(dist_source, dist_target) / 3.0)
        sections[source, section] = min(sections[source, section], dist_source / 3.0)
        sections[target, section] = min(sections[target, section], dist_target / 3.0)


def _move_nodes(positions: np.ndarray, forces: np.ndarray, sections: np.ndarray) -> None:
    """Apply OGDF serial node movement with section radius caps.

    Parameters
    ----------
    positions : np.ndarray
        Mutable coordinates with shape ``[N, 2]``.
    forces : np.ndarray
        Force array with shape ``[N, 2]``.
    sections : np.ndarray
        Radius table with shape ``[N, 9]``.

    Returns
    -------
    None
        ``positions`` is updated in place.
    """
    for node in range(positions.shape[0]):
        fx = forces[node, 0]
        fy = forces[node, 1]
        section = _section_for_vector(float(fx), float(fy))
        magnitude = math.sqrt((fx * fx) + (fy * fy))
        if magnitude > 0.0 and sections[node, section] < magnitude:
            fx = (fx / magnitude) * sections[node, section]
            fy = (fy / magnitude) * sections[node, section]
        positions[node, 0] += fx
        positions[node, 1] += fy


def _run_bertault(
    edge_index: torch.Tensor,
    num_nodes: int,
    seed: int,
    iterations: Optional[int],
    required_length: Optional[float],
) -> np.ndarray:
    """Run the OGDF Bertault default force loop.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    seed : int
        Runner-compatible initial-position seed.
    iterations : int | None
        Explicit iteration count, or ``None`` for OGDF's ``10 * N`` default.
    required_length : float | None
        Explicit desired edge length, or ``None`` for the initial average.

    Returns
    -------
    np.ndarray
        Final coordinates with shape ``[N, 2]``.
    """
    positions = _ogdf_initial_positions(num_nodes, seed)
    if num_nodes == 0:
        return positions
    edges = _edge_list(edge_index)
    adjacency = _adjacency(edges, num_nodes)
    iter_no = (num_nodes * 10) if iterations is None or iterations <= 0 else int(iterations)
    req_length = (
        _required_length(positions, edges)
        if required_length is None or required_length <= 0.0
        else float(required_length)
    )
    if req_length == 0.0:
        req_length = _DEFAULT_EDGE_LENGTH

    for _iteration in range(iter_no):
        forces = np.zeros((num_nodes, 2), dtype=np.float64)
        sections = np.full((num_nodes, 9), np.finfo(np.float64).max, dtype=np.float64)
        for node in range(num_nodes):
            _apply_node_forces(positions, node, adjacency, req_length, forces)
            for edge in edges:
                source, target = edge
                if source == node or target == node:
                    continue
                projection = _projection(positions, node, edge)
                if _projection_is_on_edge(positions, projection, edge):
                    _apply_edge_force(positions, node, edge, projection, req_length, forces)
                    _update_on_edge_sections(positions, node, edge, projection, sections)
                else:
                    _update_outside_edge_sections(positions, node, edge, sections)
        _move_nodes(positions, forces, sections)
    return positions


def build_bertault_pipeline() -> str:
    """Return the Bertault pipeline marker.

    Returns
    -------
    str
        Pipeline marker name.
    """
    return "bertault_pipeline"


def layout_bertault_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: Optional[int] = 42,
    edge_weights: Optional[torch.Tensor] = None,
    fidelity_dtype: Optional[torch.dtype] = None,
    iterations: Optional[int] = None,
    steps: Optional[int] = None,
    required_length: Optional[float] = None,
) -> torch.Tensor:
    """Run OGDF-style Bertault layout without calling the OGDF runner.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor | None, optional
        Accepted for API consistency; OGDF's default Bertault path ignores
        node geometry after the runner-owned initialization.
    seed : int | None, default=42
        Seed for runner-compatible initial coordinates.
    edge_weights : torch.Tensor | None, optional
        Accepted for API consistency; Bertault ignores weights.
    fidelity_dtype : torch.dtype | None, optional
        Optional output dtype override.
    iterations : int | None, optional
        Explicit Bertault iteration count. ``None`` uses ``10 * num_nodes``.
    steps : int | None, optional
        Alias for ``iterations`` used by generic layout dispatch.
    required_length : float | None, optional
        Explicit desired edge length. ``None`` uses the initial average edge
        length, matching OGDF.

    Returns
    -------
    torch.Tensor
        Coordinate tensor with shape ``[N, 2]``.
    """
    del node_sizes, edge_weights
    resolved_seed = 42 if seed is None else int(seed)
    positions = _run_bertault(
        edge_index=edge_index,
        num_nodes=num_nodes,
        seed=resolved_seed,
        iterations=iterations if iterations is not None else steps,
        required_length=required_length,
    )
    dtype = torch.float64 if fidelity_dtype is None else fidelity_dtype
    return torch.tensor(positions, dtype=dtype, device=edge_index.device)


__all__ = [
    "build_bertault_pipeline",
    "layout_bertault_pipeline",
]
