"""Stochastic Gradient Descent stress minimization.

Modern scalable alternative to Kamada-Kawai. Instead of computing stress over
all N² pairs, samples random node pairs and updates positions via SGD on the
stress of each sampled pair. Scales to large graphs.

Reference: Zheng, Pawar & Goodman, "Graph Drawing by Stochastic Gradient
Descent" (2018), IEEE TVCG.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional, Union

import torch

_MAX_PIVOTS = 200
_MIN_DISTANCE = 0.01
_UNREACHED = -1


def _build_undirected_adjacency(edge_index: torch.Tensor, num_nodes: int) -> list[list[int]]:
    """Build a deterministic undirected adjacency list.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list, shape ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    list[list[int]]
        Undirected adjacency list with sorted neighbors.
    """
    adjacency_sets = [set() for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return [[] for _ in range(num_nodes)]

    edge_index_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    sources = edge_index_cpu[0].tolist()
    targets = edge_index_cpu[1].tolist()

    for source, target in zip(sources, targets):
        if source == target:
            continue
        adjacency_sets[source].add(target)
        adjacency_sets[target].add(source)

    return [sorted(neighbors) for neighbors in adjacency_sets]


def _bfs_distances(adjacency: list[list[int]], source: int) -> torch.Tensor:
    """Compute full shortest-path distances from one source.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency list.
    source : int
        BFS root node index.

    Returns
    -------
    torch.Tensor
        Integer hop distances, shape ``[N]`` with ``-1`` for unreachable nodes.
    """
    num_nodes = len(adjacency)
    distances = torch.full((num_nodes,), _UNREACHED, dtype=torch.long)
    distances[source] = 0
    frontier: deque[int] = deque([source])

    while frontier:
        node = frontier.popleft()
        next_distance = int(distances[node].item()) + 1
        for neighbor in adjacency[node]:
            if int(distances[neighbor].item()) != _UNREACHED:
                continue
            distances[neighbor] = next_distance
            frontier.append(neighbor)

    return distances


def _connected_components(adjacency: list[list[int]]) -> torch.Tensor:
    """Label connected components in the undirected graph.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency list.

    Returns
    -------
    torch.Tensor
        Component labels, shape ``[N]``.
    """
    num_nodes = len(adjacency)
    component_ids = torch.full((num_nodes,), _UNREACHED, dtype=torch.long)
    component_index = 0

    for start in range(num_nodes):
        if int(component_ids[start].item()) != _UNREACHED:
            continue

        frontier: deque[int] = deque([start])
        component_ids[start] = component_index

        while frontier:
            node = frontier.popleft()
            for neighbor in adjacency[node]:
                if int(component_ids[neighbor].item()) != _UNREACHED:
                    continue
                component_ids[neighbor] = component_index
                frontier.append(neighbor)

        component_index += 1

    return component_ids


def _choose_pivots(
    component_ids: torch.Tensor,
    max_pivots: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Choose pivot nodes for approximate shortest-path queries.

    Parameters
    ----------
    component_ids : torch.Tensor
        Connected-component labels, shape ``[N]``.
    max_pivots : int
        Maximum number of pivots to use.
    generator : torch.Generator
        Random generator for deterministic sampling.

    Returns
    -------
    torch.Tensor
        Selected pivot indices, shape ``[P]``.
    """
    num_nodes = int(component_ids.shape[0])
    if num_nodes == 0:
        return torch.empty(0, dtype=torch.long)

    if num_nodes <= max_pivots:
        return torch.arange(num_nodes, dtype=torch.long)

    pivots: list[int] = []
    seen_components: set[int] = set()
    for node, component in enumerate(component_ids.tolist()):
        if component in seen_components:
            continue
        pivots.append(node)
        seen_components.add(component)
        if len(pivots) == max_pivots:
            return torch.tensor(pivots, dtype=torch.long)

    remaining = max_pivots - len(pivots)
    if remaining <= 0:
        return torch.tensor(pivots, dtype=torch.long)

    pivot_mask = torch.zeros(num_nodes, dtype=torch.bool)
    pivot_mask[torch.tensor(pivots, dtype=torch.long)] = True
    candidates = torch.arange(num_nodes, dtype=torch.long)[~pivot_mask]
    permutation = torch.randperm(int(candidates.shape[0]), generator=generator)
    extra = candidates[permutation[:remaining]]
    if extra.numel() == 0:
        return torch.tensor(pivots, dtype=torch.long)
    return torch.cat([torch.tensor(pivots, dtype=torch.long), extra], dim=0)


def _compute_pivot_distances(
    adjacency: list[list[int]],
    pivots: torch.Tensor,
) -> torch.Tensor:
    """Run BFS from each pivot node.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency list.
    pivots : torch.Tensor
        Pivot node indices, shape ``[P]``.

    Returns
    -------
    torch.Tensor
        Pivot-to-node distances, shape ``[P, N]``.
    """
    num_pivots = int(pivots.shape[0])
    num_nodes = len(adjacency)
    if num_pivots == 0:
        return torch.empty((0, num_nodes), dtype=torch.long)

    distances = torch.empty((num_pivots, num_nodes), dtype=torch.long)
    for pivot_index, pivot in enumerate(pivots.tolist()):
        distances[pivot_index] = _bfs_distances(adjacency, pivot)
    return distances


def _approx_distance(
    i_indices: torch.Tensor,
    j_indices: torch.Tensor,
    pivot_dist: torch.Tensor,
    component_ids: torch.Tensor,
    disconnected_distance: float,
) -> torch.Tensor:
    """Approximate graph distances for sampled node pairs.

    Parameters
    ----------
    i_indices : torch.Tensor
        Source node indices, shape ``[S]``.
    j_indices : torch.Tensor
        Target node indices, shape ``[S]``.
    pivot_dist : torch.Tensor
        Pivot-to-node distances, shape ``[P, N]``.
    component_ids : torch.Tensor
        Connected-component labels, shape ``[N]``.
    disconnected_distance : float
        Target distance used for pairs in different components.

    Returns
    -------
    torch.Tensor
        Approximate graph distances, shape ``[S]``.
    """
    if pivot_dist.numel() == 0:
        return torch.full(
            (i_indices.shape[0],),
            disconnected_distance,
            dtype=torch.float32,
            device=i_indices.device,
        )

    pivot_i = pivot_dist.index_select(1, i_indices)
    pivot_j = pivot_dist.index_select(1, j_indices)

    reachable = (pivot_i >= 0) & (pivot_j >= 0)
    lower_bounds = (pivot_i - pivot_j).abs().to(dtype=torch.float32)
    upper_bounds = (pivot_i + pivot_j).to(dtype=torch.float32)

    lower_bounds = torch.where(reachable, lower_bounds, torch.zeros_like(lower_bounds))
    upper_bounds = torch.where(
        reachable,
        upper_bounds,
        torch.full_like(upper_bounds, float("inf")),
    )

    best_lower = lower_bounds.max(dim=0).values
    best_upper = upper_bounds.min(dim=0).values
    midpoint = (best_lower + best_upper) * 0.5

    same_component = component_ids.index_select(0, i_indices) == component_ids.index_select(
        0, j_indices
    )
    fallback_same_component = torch.ones_like(midpoint)
    fallback_disconnected = torch.full_like(midpoint, disconnected_distance)
    fallback = torch.where(same_component, fallback_same_component, fallback_disconnected)

    approx = torch.where(torch.isfinite(best_upper), midpoint, fallback)
    return approx.clamp(min=1.0)


def _trace_snapshot(traces: list[torch.Tensor], pos: torch.Tensor) -> None:
    """Append a detached position snapshot to the trace list.

    Parameters
    ----------
    traces : list[torch.Tensor]
        Mutable trace buffer.
    pos : torch.Tensor
        Current positions, shape ``[N, 2]``.

    Returns
    -------
    None
        This function mutates ``traces`` in place.
    """
    traces.append(pos.detach().clone())


def layout_stress_sgd(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 500,
    seed: int = 42,
    sample_size: int = 200,
    trace_every: int = 0,
) -> Union[torch.Tensor, tuple[torch.Tensor, list[torch.Tensor]]]:
    """Run stochastic stress minimization layout.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list, shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Unused, accepted for interface compatibility.
    steps : int
        Number of SGD epochs.
    seed : int
        Random seed.
    sample_size : int
        Number of node pairs to sample per step.
    trace_every : int
        If > 0, record snapshots every N steps.

    Returns
    -------
    torch.Tensor or tuple
        Final positions ``[N, 2]``, or ``(positions, traces)`` if tracing.
    """
    del node_sizes

    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if trace_every < 0:
        raise ValueError("trace_every must be non-negative")

    device = edge_index.device
    positions = torch.zeros((num_nodes, 2), dtype=torch.float32, device=device)
    traces: list[torch.Tensor] = []
    if num_nodes == 0:
        return (positions, traces) if trace_every > 0 else positions

    generator_device = device.type if device.type != "mps" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)

    adjacency = _build_undirected_adjacency(edge_index, num_nodes)
    component_ids_cpu = _connected_components(adjacency)
    pivots_cpu = _choose_pivots(component_ids_cpu, min(num_nodes, _MAX_PIVOTS), generator)
    pivot_dist_cpu = _compute_pivot_distances(adjacency, pivots_cpu)

    finite_distances = pivot_dist_cpu[pivot_dist_cpu >= 0]
    graph_diameter = (
        max(int(finite_distances.max().item()), 1) if int(finite_distances.numel()) > 0 else 1
    )
    disconnected_distance = float(max(graph_diameter * 3, 2))

    pivot_dist = pivot_dist_cpu.to(device=device)
    component_ids = component_ids_cpu.to(device=device)

    positions = torch.randn(
        (num_nodes, 2), generator=generator, device=device, dtype=torch.float32
    ) * math.sqrt(float(max(num_nodes, 1)))
    positions -= positions.mean(dim=0, keepdim=True)

    if trace_every > 0 and steps == 0:
        _trace_snapshot(traces, positions)

    for step_index in range(steps):
        progress = step_index / max(steps - 1, 1)
        eta = math.exp(progress * math.log(1.0 / float(graph_diameter**2)))

        i_indices = torch.randint(
            0,
            num_nodes,
            (sample_size,),
            generator=generator,
            device=device,
        )
        j_indices = torch.randint(
            0,
            num_nodes,
            (sample_size,),
            generator=generator,
            device=device,
        )
        valid_pairs = i_indices != j_indices
        if not bool(valid_pairs.any()):
            if trace_every > 0 and (step_index + 1) % trace_every == 0:
                _trace_snapshot(traces, positions)
            continue

        i_indices = i_indices[valid_pairs]
        j_indices = j_indices[valid_pairs]
        target_distances = _approx_distance(
            i_indices=i_indices,
            j_indices=j_indices,
            pivot_dist=pivot_dist,
            component_ids=component_ids,
            disconnected_distance=disconnected_distance,
        )

        for pair_index in range(int(i_indices.shape[0])):
            source_index = int(i_indices[pair_index].item())
            target_index = int(j_indices[pair_index].item())
            target_distance = target_distances[pair_index]

            delta = positions[source_index] - positions[target_index]
            current_distance = delta.norm().clamp(min=_MIN_DISTANCE)
            weight = 1.0 / target_distance.square().clamp(min=_MIN_DISTANCE)
            magnitude = eta * weight * (1.0 - target_distance / current_distance)
            displacement = magnitude * delta * 0.5

            positions[source_index] -= displacement
            positions[target_index] += displacement

        positions -= positions.mean(dim=0, keepdim=True)

        if trace_every > 0 and (step_index + 1) % trace_every == 0:
            _trace_snapshot(traces, positions)

    return (positions, traces) if trace_every > 0 else positions
