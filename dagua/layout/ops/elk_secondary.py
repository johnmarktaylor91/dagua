"""Local ports for the ELK secondary layout algorithms.

The implementations in this module are deliberately self-contained and never
delegate to ``elkjs``.  They follow the algorithm families used by ELK's Force,
Stress, MrTree, and Radial providers closely enough for local execution and
fidelity measurement; the verification script records the remaining residuals
against the JavaScript reference.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from dagua.layout.ops._reingold_tilford import layout_igraph_reingold_tilford

_DEFAULT_NODE_WIDTH = 120.0
_DEFAULT_NODE_HEIGHT = 40.0
_ELK_PADDING = 50.0
_ELK_FORCE_SPACING = 80.0
_ELK_FORCE_REPULSION = 5.0
_ELK_FORCE_TEMPERATURE = 0.001
_ELK_FORCE_DISP_BOUND_FACTOR = 16.0
_ELK_STRESS_EDGE_LENGTH = 100.0
_ELK_STRESS_EPSILON = 1.0e-3
_ELK_STRESS_ITERATION_LIMIT = 2_147_483_647
_ELK_TREE_LAYER_SPACING = 100.0
_ELK_TREE_NODE_SPACING = 40.0
_JAVA_RANDOM_MULTIPLIER = 0x5DEECE66D
_JAVA_RANDOM_ADDEND = 0xB
_JAVA_RANDOM_MASK = (1 << 48) - 1


@dataclass
class ElkForceConfig:
    """Configuration for the local ELK Force model.

    Parameters
    ----------
    iterations : int
        Number of displacement iterations.
    model : str
        Force model name, either ``"eades"`` or ``"fruchterman_reingold"``.
    spacing : float
        ELK node-node spacing used as the Eades spring length and FR scale.
    repulsion : float
        Eades repulsion factor.
    temperature : float
        FR temperature.
    seed : int
        Java-style random seed used only for coincident particles.
    """

    iterations: int = 300
    model: str = "fruchterman_reingold"
    spacing: float = _ELK_FORCE_SPACING
    repulsion: float = _ELK_FORCE_REPULSION
    temperature: float = _ELK_FORCE_TEMPERATURE
    seed: int = 1


class JavaRandom:
    """Minimal Java ``Random`` port for ELK-compatible jitter.

    Parameters
    ----------
    seed : int
        Seed value passed to Java ``Random(long)``.
    """

    def __init__(self, seed: int) -> None:
        self.seed = (int(seed) ^ _JAVA_RANDOM_MULTIPLIER) & _JAVA_RANDOM_MASK

    def next_bits(self, bits: int) -> int:
        """Return the next Java random bit field.

        Parameters
        ----------
        bits : int
            Number of high bits to return.

        Returns
        -------
        int
            Non-negative integer containing ``bits`` random bits.
        """
        self.seed = (self.seed * _JAVA_RANDOM_MULTIPLIER + _JAVA_RANDOM_ADDEND) & _JAVA_RANDOM_MASK
        return self.seed >> (48 - bits)

    def next_double(self) -> float:
        """Return the next Java ``double`` in ``[0, 1)``.

        Returns
        -------
        float
            Pseudorandom floating-point value.
        """
        return ((self.next_bits(26) << 27) + self.next_bits(27)) / float(1 << 53)


def _node_sizes_or_default(
    node_sizes: Optional[torch.Tensor],
    num_nodes: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return node sizes, filling ELK's benchmark default when absent.

    Parameters
    ----------
    node_sizes : torch.Tensor or None
        Optional size tensor with shape ``[N, 2]``.
    num_nodes : int
        Number of nodes.
    dtype : torch.dtype
        Output dtype.

    Returns
    -------
    torch.Tensor
        CPU tensor with shape ``[N, 2]``.
    """
    if node_sizes is None:
        sizes = torch.empty((num_nodes, 2), dtype=dtype)
        sizes[:, 0] = _DEFAULT_NODE_WIDTH
        sizes[:, 1] = _DEFAULT_NODE_HEIGHT
        return sizes
    return node_sizes.detach().to(device="cpu", dtype=dtype)


def _edge_pairs(edge_index: torch.Tensor) -> List[Tuple[int, int]]:
    """Convert an edge-index tensor to Python edge pairs.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.

    Returns
    -------
    list[tuple[int, int]]
        Source-target pairs in model order.
    """
    if edge_index.numel() == 0:
        return []
    return [(int(source), int(target)) for source, target in edge_index.t().detach().cpu().tolist()]


def _initial_model_positions(num_nodes: int, node_sizes: torch.Tensor) -> torch.Tensor:
    """Create deterministic non-overlapping initial top-left positions.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Center positions with shape ``[N, 2]``.
    """
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=node_sizes.dtype)
    columns = max(1, int(math.ceil(math.sqrt(float(num_nodes)))))
    max_width = float(torch.max(node_sizes[:, 0]).item()) if num_nodes else _DEFAULT_NODE_WIDTH
    max_height = float(torch.max(node_sizes[:, 1]).item()) if num_nodes else _DEFAULT_NODE_HEIGHT
    x_step = max_width + _ELK_TREE_NODE_SPACING
    y_step = max_height + _ELK_TREE_NODE_SPACING
    pos = torch.empty((num_nodes, 2), dtype=node_sizes.dtype)
    for node in range(num_nodes):
        row, col = divmod(node, columns)
        pos[node, 0] = col * x_step + node_sizes[node, 0] / 2.0
        pos[node, 1] = row * y_step + node_sizes[node, 1] / 2.0
    return pos


def _initial_random_positions(num_nodes: int, rng: JavaRandom, dtype: torch.dtype) -> torch.Tensor:
    """Create ELK Force's non-interactive random initial center positions.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    rng : JavaRandom
        Java-compatible random generator seeded from ``elk.randomSeed``.
    dtype : torch.dtype
        Output dtype.

    Returns
    -------
    torch.Tensor
        Center positions with shape ``[N, 2]``.
    """
    pos = torch.empty((num_nodes, 2), dtype=dtype)
    pos_scale = float(num_nodes)
    for node in range(num_nodes):
        pos[node, 0] = rng.next_double() * pos_scale
        pos[node, 1] = rng.next_double() * pos_scale
    return pos


def _center_to_elk_topleft(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    padding: float = 0.0,
) -> torch.Tensor:
    """Convert internal center coordinates to ELK top-left coordinates.

    Parameters
    ----------
    pos : torch.Tensor
        Center coordinates with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.
    padding : float, default=0.0
        Root graph padding added by ELK's force graph importer.

    Returns
    -------
    torch.Tensor
        Top-left coordinates with shape ``[N, 2]`` and minimum at the origin.
    """
    if pos.numel() == 0:
        return pos.clone()
    top_left = pos - node_sizes / 2.0
    mins = torch.min(top_left, dim=0).values
    return top_left + (float(padding) - mins)


def _connection_counts(num_nodes: int, edges: Sequence[Tuple[int, int]]) -> torch.Tensor:
    """Build ELK Force's directed-pair connection-count matrix.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    edges : sequence[tuple[int, int]]
        Edge pairs.

    Returns
    -------
    torch.Tensor
        Float64 matrix with shape ``[N, N]``.
    """
    counts = torch.zeros((num_nodes, num_nodes), dtype=torch.float64)
    for source, target in edges:
        if source == target:
            continue
        counts[source, target] += 1.0
        counts[target, source] += 1.0
    return counts


def layout_elk_force(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    config: Optional[ElkForceConfig] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Run a local ELK Force-style Eades/FR solver.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Node sizes with shape ``[N, 2]``.
    config : ElkForceConfig, optional
        Force model configuration.
    dtype : torch.dtype, default=torch.float64
        Internal and output dtype.

    Returns
    -------
    torch.Tensor
        ELK top-left node coordinates with shape ``[N, 2]``.
    """
    cfg = config or ElkForceConfig()
    sizes = _node_sizes_or_default(node_sizes, num_nodes, dtype)
    rng = JavaRandom(cfg.seed)
    pos = _initial_random_positions(num_nodes, rng, dtype)
    if num_nodes <= 1:
        return _center_to_elk_topleft(pos, sizes, _ELK_PADDING).to(
            device=edge_index.device,
            dtype=dtype,
        )

    connections = _connection_counts(num_nodes, _edge_pairs(edge_index))
    model = cfg.model.lower()
    iterations = max(0, int(cfg.iterations))
    temperature = float(cfg.temperature)
    threshold = temperature / float(iterations) if iterations > 0 else 0.0
    displacement_bound = max(
        float(num_nodes) * _ELK_FORCE_DISP_BOUND_FACTOR + float(edge_index.shape[1]),
        _ELK_FORCE_DISP_BOUND_FACTOR * _ELK_FORCE_DISP_BOUND_FACTOR,
    )
    total_width = float(torch.sum(sizes[:, 0]).item())
    total_height = float(torch.sum(sizes[:, 1]).item())
    area = max(total_width * total_height, 1.0)
    k = math.sqrt(area / (2.0 * float(num_nodes))) * float(cfg.spacing) * 0.01
    size_lengths = torch.linalg.norm(sizes, dim=1)

    step = 0
    while step < iterations and (model != "fruchterman_reingold" or temperature > 0.0):
        if model == "fruchterman_reingold":
            temperature -= threshold
        displacement = torch.zeros_like(pos)
        for forcee in range(num_nodes):
            for forcer in range(num_nodes):
                if forcer == forcee:
                    continue
                while pos[forcer, 0] == pos[forcee, 0] and pos[forcer, 1] == pos[forcee, 1]:
                    pos[forcer, 0] += rng.next_double() - 0.5
                    pos[forcer, 1] += rng.next_double() - 0.5
                    pos[forcee, 0] += rng.next_double() - 0.5
                    pos[forcee, 1] += rng.next_double() - 0.5
                vector = pos[forcee] - pos[forcer]
                length = float(torch.linalg.norm(vector).item())
                radius_sum = float(size_lengths[forcer] / 2 + size_lengths[forcee] / 2)
                clearance = max(0.0, length - radius_sum)
                connection = float(connections[forcer, forcee].item())
                if model == "fruchterman_reingold":
                    force = (k * k / clearance) if clearance > 0.0 else (k * k * 100.0)
                    if connection > 0.0:
                        force -= (clearance * clearance / k) * connection
                    force *= temperature
                else:
                    if connection > 0.0:
                        attractive = (
                            math.log(clearance / float(cfg.spacing)) if clearance > 0 else -100.0
                        )
                        force = -attractive * connection
                    else:
                        force = (
                            float(cfg.repulsion) / (clearance * clearance)
                            if clearance > 0.0
                            else float(cfg.repulsion) * 100.0
                        )
                displacement[forcee] += vector * (force / length)
        pos += torch.clamp(displacement, min=-displacement_bound, max=displacement_bound)
        step += 1

    return _center_to_elk_topleft(pos, sizes, _ELK_PADDING).to(
        device=edge_index.device,
        dtype=dtype,
    )


def _undirected_weighted_distances(
    num_nodes: int,
    edges: Sequence[Tuple[int, int]],
    edge_length: float,
) -> torch.Tensor:
    """Compute ELK Stress Dijkstra distances.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    edges : sequence[tuple[int, int]]
        Edge pairs.
    edge_length : float
        Desired length for each edge.

    Returns
    -------
    torch.Tensor
        Distance matrix with shape ``[N, N]``.
    """
    adjacency: List[List[Tuple[int, float]]] = [[] for _ in range(num_nodes)]
    for source, target in edges:
        if source == target:
            continue
        adjacency[source].append((target, edge_length))
        adjacency[target].append((source, edge_length))
    distances = torch.full((num_nodes, num_nodes), float("inf"), dtype=torch.float64)
    for source in range(num_nodes):
        distances[source, source] = 0.0
        heap: List[Tuple[float, int]] = [(0.0, source)]
        while heap:
            current_distance, node = heapq.heappop(heap)
            if current_distance != float(distances[source, node].item()):
                continue
            for neighbor, weight in adjacency[node]:
                candidate = current_distance + weight
                if candidate < float(distances[source, neighbor].item()):
                    distances[source, neighbor] = candidate
                    heapq.heappush(heap, (candidate, neighbor))
    return distances


def _stress_value(pos: torch.Tensor, distances: torch.Tensor, weights: torch.Tensor) -> float:
    """Compute ELK Stress objective value.

    Parameters
    ----------
    pos : torch.Tensor
        Center coordinates with shape ``[N, 2]``.
    distances : torch.Tensor
        Graph-theoretic distances with shape ``[N, N]``.
    weights : torch.Tensor
        Stress weights with shape ``[N, N]``.

    Returns
    -------
    float
        Weighted stress over upper-triangle pairs.
    """
    stress = 0.0
    num_nodes = int(pos.shape[0])
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if not math.isfinite(float(distances[i, j].item())):
                continue
            euclidean = float(torch.linalg.norm(pos[i] - pos[j]).item())
            delta = euclidean - float(distances[i, j].item())
            stress += float(weights[i, j].item()) * delta * delta
    return stress


def layout_elk_stress(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    desired_edge_length: float = _ELK_STRESS_EDGE_LENGTH,
    epsilon: float = _ELK_STRESS_EPSILON,
    iteration_limit: int = _ELK_STRESS_ITERATION_LIMIT,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Run ELK Stress-style majorization.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Node sizes with shape ``[N, 2]``.
    desired_edge_length : float, default=100.0
        Desired length assigned to every edge.
    epsilon : float, default=1e-4
        Relative stress improvement stopping threshold.
    iteration_limit : int, default=200
        Maximum majorization iterations.
    dtype : torch.dtype, default=torch.float64
        Internal and output dtype.

    Returns
    -------
    torch.Tensor
        ELK top-left node coordinates with shape ``[N, 2]``.
    """
    sizes = _node_sizes_or_default(node_sizes, num_nodes, dtype)
    # ELK Stress performs a non-interactive ELK Force layout first, then
    # imports those node locations as the stress-majorization warm start.
    force_top_left = layout_elk_force(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=sizes,
        config=ElkForceConfig(),
        dtype=dtype,
    ).to(device="cpu", dtype=dtype)
    pos = force_top_left + sizes / 2.0
    if num_nodes <= 1:
        return _center_to_elk_topleft(pos, sizes, _ELK_PADDING).to(
            device=edge_index.device,
            dtype=dtype,
        )

    distances = _undirected_weighted_distances(
        num_nodes,
        _edge_pairs(edge_index),
        desired_edge_length,
    )
    finite = torch.isfinite(distances) & (distances > 0)
    weights = torch.zeros_like(distances)
    weights[finite] = 1.0 / (distances[finite] * distances[finite])

    previous = _stress_value(pos, distances, weights)
    current = float("inf")
    count = 0
    while True:
        if count > 0:
            previous = current
        for node in range(num_nodes):
            weight_sum = float(torch.sum(weights[node]).item())
            if weight_sum <= 0.0:
                continue
            x_disp = 0.0
            y_disp = 0.0
            for other in range(num_nodes):
                if other == node or not math.isfinite(float(distances[node, other].item())):
                    continue
                wij = float(weights[node, other].item())
                euclidean = float(torch.linalg.norm(pos[node] - pos[other]).item())
                if euclidean > 0.0:
                    graph_distance = float(distances[node, other].item())
                    x_disp += wij * (
                        float(pos[other, 0].item())
                        + graph_distance * float(pos[node, 0] - pos[other, 0]) / euclidean
                    )
                    y_disp += wij * (
                        float(pos[other, 1].item())
                        + graph_distance * float(pos[node, 1] - pos[other, 1]) / euclidean
                    )
            pos[node, 0] = x_disp / weight_sum
            pos[node, 1] = y_disp / weight_sum
        current = _stress_value(pos, distances, weights)
        done = (
            previous == 0.0 or (previous - current) / previous < epsilon or count >= iteration_limit
        )
        count += 1
        if done:
            break
    return _center_to_elk_topleft(pos, sizes, _ELK_PADDING).to(
        device=edge_index.device,
        dtype=dtype,
    )


def layout_elk_mrtree(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    roots: Optional[Sequence[int]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Run a Walker/Reingold-Tilford tidy-tree layout for ELK MrTree.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Node sizes with shape ``[N, 2]``.
    roots : sequence[int] or None, default=None
        Optional explicit root order.
    dtype : torch.dtype, default=torch.float64
        Output dtype.

    Returns
    -------
    torch.Tensor
        Top-left tree coordinates with shape ``[N, 2]``.
    """
    sizes = _node_sizes_or_default(node_sizes, num_nodes, dtype)
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=dtype, device=edge_index.device)
    rt = layout_igraph_reingold_tilford(
        edge_index=edge_index,
        num_nodes=num_nodes,
        traversal_mode="out",
        roots=roots,
        center_output=False,
        output_scale=1.0,
    ).to(dtype=dtype)
    centers = torch.empty_like(rt)
    centers[:, 0] = rt[:, 0] * (_DEFAULT_NODE_WIDTH + _ELK_TREE_NODE_SPACING) + sizes[:, 0] / 2.0
    centers[:, 1] = rt[:, 1] * (_DEFAULT_NODE_HEIGHT + _ELK_TREE_LAYER_SPACING) + sizes[:, 1] / 2.0
    return _center_to_elk_topleft(centers, sizes).to(device=edge_index.device, dtype=dtype)


def layout_elk_radial(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    roots: Optional[Sequence[int]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Run a concentric radial-tree layout for ELK Radial.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Node sizes with shape ``[N, 2]``.
    roots : sequence[int] or None, default=None
        Optional explicit root order.
    dtype : torch.dtype, default=torch.float64
        Output dtype.

    Returns
    -------
    torch.Tensor
        Top-left radial coordinates with shape ``[N, 2]``.
    """
    sizes = _node_sizes_or_default(node_sizes, num_nodes, dtype)
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=dtype, device=edge_index.device)
    rt = layout_igraph_reingold_tilford(
        edge_index=edge_index,
        num_nodes=num_nodes,
        traversal_mode="out",
        roots=roots,
        center_output=False,
        output_scale=1.0,
    ).to(dtype=dtype)
    order = rt[:, 0]
    min_order = float(torch.min(order).item())
    max_order = float(torch.max(order).item())
    if max_order > min_order:
        angle = (order - min_order) * (2.0 * math.pi / (max_order - min_order + 1.0))
    else:
        angle = torch.zeros(num_nodes, dtype=dtype)
    radius = rt[:, 1] * (_DEFAULT_NODE_HEIGHT + _ELK_TREE_LAYER_SPACING)
    centers = torch.empty_like(rt)
    centers[:, 0] = radius * torch.cos(angle) + sizes[:, 0] / 2.0
    centers[:, 1] = radius * torch.sin(angle) + sizes[:, 1] / 2.0
    return _center_to_elk_topleft(centers, sizes).to(device=edge_index.device, dtype=dtype)


__all__ = [
    "ElkForceConfig",
    "JavaRandom",
    "layout_elk_force",
    "layout_elk_mrtree",
    "layout_elk_radial",
    "layout_elk_stress",
]
