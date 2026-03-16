"""Kamada-Kawai spring layout based on graph-theoretic distances.

Every pair of nodes is connected by a spring whose ideal length is proportional
to its shortest-path distance. For smaller graphs this module uses the original
per-node Newton-Raphson solver from Kamada and Kawai; for very large graphs it
keeps the existing Adam-based stress minimizer as a practical fallback.

Reference: Kamada & Kawai, "An Algorithm for Drawing General Undirected Graphs"
(1989), Information Processing Letters.
"""

from __future__ import annotations

from collections import deque
from typing import Optional, Union

import torch

_PIVOT_THRESHOLD = 5000
_PIVOT_COUNT = 200
_NEWTON_DISTANCE_EPSILON = 1.0e-6
_NEWTON_GLOBAL_EPSILON = 1.0e-4
_NEWTON_LOCAL_EPSILON = 1.0e-4
_NEWTON_HESSIAN_REGULARIZATION = 1.0e-6


def _build_undirected_adjacency(edge_index: torch.Tensor, num_nodes: int) -> list[list[int]]:
    """Build an undirected adjacency list from a ``[2, E]`` edge tensor.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    list[list[int]]
        Adjacency list with one neighbor list per node.

    Raises
    ------
    ValueError
        If ``edge_index`` has an invalid shape or references an out-of-range node.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")

    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return adjacency

    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    sources = edge_index_cpu[0].tolist()
    targets = edge_index_cpu[1].tolist()

    for source, target in zip(sources, targets):
        if source < 0 or source >= num_nodes or target < 0 or target >= num_nodes:
            raise ValueError("edge_index contains a node index outside [0, num_nodes).")
        adjacency[source].append(target)
        if source != target:
            adjacency[target].append(source)

    return adjacency


def _bfs_distances(adjacency: list[list[int]], start: int) -> list[int]:
    """Compute unweighted shortest-path distances from one source node.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency list.
    start : int
        Source node index.

    Returns
    -------
    list[int]
        Distances from ``start`` to every node, with ``-1`` for unreachable nodes.
    """
    num_nodes = len(adjacency)
    distances = [-1] * num_nodes
    distances[start] = 0
    frontier: deque[int] = deque([start])

    while frontier:
        node = frontier.popleft()
        next_distance = distances[node] + 1
        for neighbor in adjacency[node]:
            if distances[neighbor] == -1:
                distances[neighbor] = next_distance
                frontier.append(neighbor)

    return distances


def _fill_unreachable_distances(distances: torch.Tensor, diameter: int) -> torch.Tensor:
    """Replace unreachable entries with ``diameter + 1``.

    Parameters
    ----------
    distances : torch.Tensor
        Distance tensor containing ``-1`` for unreachable entries.
    diameter : int
        Maximum finite shortest-path distance observed so far.

    Returns
    -------
    torch.Tensor
        Distance tensor with all entries finite.
    """
    fill_value = float(diameter + 1 if distances.shape[0] > 1 else 0)
    return torch.where(distances >= 0, distances, torch.full_like(distances, fill_value))


def _compute_all_pairs_shortest_paths(adjacency: list[list[int]]) -> torch.Tensor:
    """Compute the full all-pairs shortest-path matrix with BFS.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency list.

    Returns
    -------
    torch.Tensor
        Shortest-path distances with shape ``[N, N]`` and dtype ``float32``.
    """
    num_nodes = len(adjacency)
    rows: list[list[int]] = []
    diameter = 0

    for node in range(num_nodes):
        distances = _bfs_distances(adjacency, node)
        finite_distances = [distance for distance in distances if distance >= 0]
        if finite_distances:
            diameter = max(diameter, max(finite_distances))
        rows.append(distances)

    distance_matrix = torch.tensor(rows, dtype=torch.float32)
    return _fill_unreachable_distances(distance_matrix, diameter)


def _sample_pivot_distances(
    adjacency: list[list[int]],
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate all-pairs distances with BFS from sampled pivot nodes.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency list.
    seed : int
        Random seed for deterministic pivot sampling.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Pivot indices with shape ``[P]`` and pivot distances with shape ``[N, P]``.
    """
    num_nodes = len(adjacency)
    pivot_count = min(_PIVOT_COUNT, num_nodes)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    pivot_indices = torch.randperm(num_nodes, generator=generator)[:pivot_count]

    rows: list[list[int]] = []
    diameter = 0
    for pivot in pivot_indices.tolist():
        distances = _bfs_distances(adjacency, pivot)
        finite_distances = [distance for distance in distances if distance >= 0]
        if finite_distances:
            diameter = max(diameter, max(finite_distances))
        rows.append(distances)

    pivot_distances = torch.tensor(rows, dtype=torch.float32).transpose(0, 1).contiguous()
    return pivot_indices.to(dtype=torch.long), _fill_unreachable_distances(
        pivot_distances, diameter
    )


def _ideal_edge_length(num_nodes: int) -> float:
    """Compute the base target length for graph-theoretic distance one.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    float
        Ideal length multiplier ``L0``.
    """
    area = float(max(num_nodes, 1))
    return float((area / max(num_nodes, 1)) ** 0.5)


def _target_lengths_and_strengths(distances: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert graph distances into KK target lengths and spring strengths.

    Parameters
    ----------
    distances : torch.Tensor
        Shortest-path distances with shape ``[N, N]`` or ``[N, P]``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Target lengths ``L`` and spring strengths ``K``.
    """
    lengths = distances * _ideal_edge_length(int(distances.shape[0]))
    safe_distances = torch.where(distances > 0, distances, torch.ones_like(distances))
    strengths = torch.where(
        distances > 0,
        1.0 / safe_distances.square(),
        torch.zeros_like(distances),
    )
    return lengths, strengths


def _initialize_positions(num_nodes: int, seed: int) -> torch.Tensor:
    """Create deterministic random initial positions.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    seed : int
        Random seed for the initial placement.

    Returns
    -------
    torch.Tensor
        Initial positions with shape ``[N, 2]``.
    """
    if num_nodes <= 1:
        return torch.zeros((num_nodes, 2), dtype=torch.float32)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn((num_nodes, 2), generator=generator, dtype=torch.float32)


def _resolve_solver(num_nodes: int, solver: str) -> str:
    """Resolve the public solver mode into a concrete implementation name.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    solver : str
        Requested solver mode.

    Returns
    -------
    str
        Concrete solver name, either ``"newton"`` or ``"adam"``.

    Raises
    ------
    ValueError
        If ``solver`` is not one of ``"auto"``, ``"newton"``, or ``"adam"``.
    """
    if solver not in {"auto", "newton", "adam"}:
        raise ValueError("solver must be one of 'auto', 'newton', or 'adam'.")
    if solver == "auto":
        return "newton" if num_nodes <= _PIVOT_THRESHOLD else "adam"
    return solver


def _node_energy_gradient_and_hessian(
    positions: torch.Tensor,
    target_lengths: torch.Tensor,
    spring_strengths: torch.Tensor,
    node_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the KK gradient and Hessian for one node.

    Parameters
    ----------
    positions : torch.Tensor
        Current node positions with shape ``[N, 2]``.
    target_lengths : torch.Tensor
        Ideal spring lengths with shape ``[N, N]``.
    spring_strengths : torch.Tensor
        Spring strengths with shape ``[N, N]``.
    node_index : int
        Node whose Newton system should be evaluated.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Gradient vector ``[2]`` and Hessian matrix ``[2, 2]``.
    """
    delta = positions[node_index] - positions
    distances = torch.linalg.norm(delta, dim=1).clamp(min=_NEWTON_DISTANCE_EPSILON)
    mask = torch.ones(positions.shape[0], dtype=torch.bool, device=positions.device)
    mask[node_index] = False

    dx = delta[mask, 0]
    dy = delta[mask, 1]
    dist = distances[mask]
    lengths = target_lengths[node_index, mask]
    strengths = spring_strengths[node_index, mask]

    factor = 1.0 - lengths / dist
    gradient = torch.stack(
        [
            torch.sum(strengths * dx * factor),
            torch.sum(strengths * dy * factor),
        ]
    )

    inv_dist_cubed = dist.pow(-3)
    hessian_xx = torch.sum(strengths * (1.0 - lengths * dy.square() * inv_dist_cubed))
    hessian_yy = torch.sum(strengths * (1.0 - lengths * dx.square() * inv_dist_cubed))
    hessian_xy = torch.sum(strengths * lengths * dx * dy * inv_dist_cubed)
    hessian = torch.stack(
        [
            torch.stack([hessian_xx, hessian_xy]),
            torch.stack([hessian_xy, hessian_yy]),
        ]
    )
    return gradient, hessian


def _all_node_deltas(
    positions: torch.Tensor,
    target_lengths: torch.Tensor,
    spring_strengths: torch.Tensor,
) -> torch.Tensor:
    """Compute the KK stationarity measure for every node.

    Parameters
    ----------
    positions : torch.Tensor
        Current node positions with shape ``[N, 2]``.
    target_lengths : torch.Tensor
        Ideal spring lengths with shape ``[N, N]``.
    spring_strengths : torch.Tensor
        Spring strengths with shape ``[N, N]``.

    Returns
    -------
    torch.Tensor
        Per-node delta tensor with shape ``[N]``.
    """
    deltas = torch.empty((positions.shape[0],), dtype=positions.dtype, device=positions.device)
    for node_index in range(positions.shape[0]):
        gradient, _ = _node_energy_gradient_and_hessian(
            positions=positions,
            target_lengths=target_lengths,
            spring_strengths=spring_strengths,
            node_index=node_index,
        )
        deltas[node_index] = torch.linalg.norm(gradient)
    return deltas


def _solve_newton_displacement(gradient: torch.Tensor, hessian: torch.Tensor) -> torch.Tensor:
    """Solve the 2x2 Newton system for one KK node update.

    Parameters
    ----------
    gradient : torch.Tensor
        KK gradient vector with shape ``[2]``.
    hessian : torch.Tensor
        KK Hessian matrix with shape ``[2, 2]``.

    Returns
    -------
    torch.Tensor
        Position displacement vector with shape ``[2]``.
    """
    regularized_hessian = hessian + (
        _NEWTON_HESSIAN_REGULARIZATION * torch.eye(2, dtype=hessian.dtype, device=hessian.device)
    )
    rhs = -gradient.unsqueeze(1)
    try:
        return torch.linalg.solve(regularized_hessian, rhs).squeeze(1)
    except RuntimeError:
        return torch.zeros((2,), dtype=gradient.dtype, device=gradient.device)


def _layout_kk_newton(
    target_lengths: torch.Tensor,
    spring_strengths: torch.Tensor,
    num_nodes: int,
    steps: int,
    seed: int,
    trace_every: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run the original KK per-node Newton-Raphson solver.

    Parameters
    ----------
    target_lengths : torch.Tensor
        Ideal spring lengths with shape ``[N, N]``.
    spring_strengths : torch.Tensor
        Spring strengths with shape ``[N, N]``.
    num_nodes : int
        Number of nodes.
    steps : int
        Maximum number of outer KK iterations.
    seed : int
        Random seed for the initial placement.
    trace_every : int
        Snapshot cadence in outer iterations.

    Returns
    -------
    tuple[torch.Tensor, list[torch.Tensor]]
        Final positions and optional trace snapshots.
    """
    positions = _initialize_positions(num_nodes=num_nodes, seed=seed).to(dtype=torch.float64)
    lengths = target_lengths.to(dtype=torch.float64)
    strengths = spring_strengths.to(dtype=torch.float64)
    traces: list[torch.Tensor] = []
    if steps == 0:
        return positions.to(dtype=torch.float32), traces

    inner_limit = max(8, min(steps, 32))
    for step_index in range(steps):
        deltas = _all_node_deltas(
            positions=positions,
            target_lengths=lengths,
            spring_strengths=strengths,
        )
        max_delta, node_tensor = torch.max(deltas, dim=0)
        if float(max_delta.item()) < _NEWTON_GLOBAL_EPSILON:
            break

        node_index = int(node_tensor.item())
        for _ in range(inner_limit):
            gradient, hessian = _node_energy_gradient_and_hessian(
                positions=positions,
                target_lengths=lengths,
                spring_strengths=strengths,
                node_index=node_index,
            )
            if float(torch.linalg.norm(gradient).item()) < _NEWTON_LOCAL_EPSILON:
                break
            positions[node_index] = positions[node_index] + _solve_newton_displacement(
                gradient=gradient,
                hessian=hessian,
            )

        if trace_every > 0 and (step_index + 1) % trace_every == 0:
            traces.append(positions.to(dtype=torch.float32).clone())

    return positions.to(dtype=torch.float32), traces


def _layout_kk_adam(
    adjacency: list[list[int]],
    num_nodes: int,
    steps: int,
    seed: int,
    trace_every: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run the practical Adam-based KK stress fallback.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency list.
    num_nodes : int
        Number of nodes.
    steps : int
        Number of optimizer iterations.
    seed : int
        Random seed for the initial placement and pivot sampling.
    trace_every : int
        Snapshot cadence in optimizer iterations.

    Returns
    -------
    tuple[torch.Tensor, list[torch.Tensor]]
        Final positions and optional trace snapshots.
    """
    use_pivots = num_nodes > _PIVOT_THRESHOLD
    if use_pivots:
        pivot_indices, pivot_distances = _sample_pivot_distances(adjacency=adjacency, seed=seed)
        target_lengths, spring_strengths = _target_lengths_and_strengths(pivot_distances)
    else:
        pivot_indices = torch.empty(0, dtype=torch.long)
        distance_matrix = _compute_all_pairs_shortest_paths(adjacency)
        target_lengths, spring_strengths = _target_lengths_and_strengths(distance_matrix)

    positions = torch.nn.Parameter(_initialize_positions(num_nodes=num_nodes, seed=seed))
    optimizer = torch.optim.Adam([positions], lr=1.0)
    traces: list[torch.Tensor] = []

    for step_index in range(steps):
        optimizer.zero_grad()
        if use_pivots:
            pivot_positions = positions[pivot_indices]
            pairwise_distances = torch.cdist(positions, pivot_positions)
            stress = (spring_strengths * (pairwise_distances - target_lengths).square()).sum()
        else:
            pairwise_distances = torch.cdist(positions, positions)
            weighted_error = spring_strengths * (pairwise_distances - target_lengths).square()
            stress = torch.triu(weighted_error, diagonal=1).sum()
        stress.backward()
        optimizer.step()

        if trace_every > 0 and (step_index + 1) % trace_every == 0:
            traces.append(positions.detach().clone())

    return positions.detach(), traces


def layout_kk(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 500,
    seed: int = 42,
    trace_every: int = 0,
    solver: str = "auto",
) -> Union[torch.Tensor, tuple[torch.Tensor, list[torch.Tensor]]]:
    """Run Kamada-Kawai layout.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list, shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Unused, accepted for interface compatibility.
    steps : int
        Maximum number of optimization iterations.
    seed : int
        Random seed for initial placement.
    trace_every : int
        If > 0, record snapshots every N iterations.
    solver : {"auto", "newton", "adam"}, default="auto"
        Solver selection. ``"newton"`` uses the original per-node KK solver,
        ``"adam"`` uses the existing full-stress fallback, and ``"auto"``
        selects Newton for graphs with at most 5000 nodes.

    Returns
    -------
    torch.Tensor or tuple
        Final positions ``[N, 2]``, or ``(positions, traces)`` if tracing.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``steps``, ``trace_every``, or ``solver`` are invalid.
    """
    del node_sizes

    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if trace_every < 0:
        raise ValueError("trace_every must be non-negative.")

    if num_nodes == 0:
        empty_positions = torch.empty((0, 2), dtype=torch.float32)
        return (empty_positions, []) if trace_every > 0 else empty_positions

    adjacency = _build_undirected_adjacency(edge_index=edge_index, num_nodes=num_nodes)
    resolved_solver = _resolve_solver(num_nodes=num_nodes, solver=solver)

    if resolved_solver == "newton":
        distance_matrix = _compute_all_pairs_shortest_paths(adjacency)
        target_lengths, spring_strengths = _target_lengths_and_strengths(distance_matrix)
        final_positions, traces = _layout_kk_newton(
            target_lengths=target_lengths,
            spring_strengths=spring_strengths,
            num_nodes=num_nodes,
            steps=steps,
            seed=seed,
            trace_every=trace_every,
        )
    else:
        final_positions, traces = _layout_kk_adam(
            adjacency=adjacency,
            num_nodes=num_nodes,
            steps=steps,
            seed=seed,
            trace_every=trace_every,
        )

    return (final_positions, traces) if trace_every > 0 else final_positions
