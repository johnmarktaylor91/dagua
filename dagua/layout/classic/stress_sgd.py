"""Stochastic Gradient Descent stress minimization.

This implementation follows the public ``s_gd2`` defaults for connected,
unweighted graphs and keeps the repo's sampling fallback for larger graphs as a
documented acceleration.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional, Union

import torch

_MAX_PIVOTS = 200
_AUTO_FULL_EPOCH_THRESHOLD = 1_000
_AUTO_SAMPLE_THRESHOLD = 1_000
_EXACT_DISTANCE_THRESHOLD = 10_000
_MIN_DISTANCE = 0.01
_UNREACHED = -1
_DEFAULT_EPS = 0.01


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


def _is_connected(adjacency: list[list[int]]) -> bool:
    """Report whether the undirected graph is connected.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency list.

    Returns
    -------
    bool
        ``True`` when every node is reachable from node zero.
    """
    if len(adjacency) <= 1:
        return True
    return bool((_bfs_distances(adjacency, 0) >= 0).all().item())


def _all_pairs_shortest_paths(adjacency: list[list[int]]) -> torch.Tensor:
    """Compute exact all-pairs shortest paths with repeated BFS.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency list.

    Returns
    -------
    torch.Tensor
        Distance matrix with shape ``[N, N]``.
    """
    rows = [_bfs_distances(adjacency, node) for node in range(len(adjacency))]
    return torch.stack(rows, dim=0).to(dtype=torch.float32)


def _choose_pivots(num_nodes: int, max_pivots: int, generator: torch.Generator) -> torch.Tensor:
    """Choose pivot nodes for approximate shortest-path queries.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    max_pivots : int
        Maximum number of pivots to use.
    generator : torch.Generator
        Random generator for deterministic sampling.

    Returns
    -------
    torch.Tensor
        Selected pivot indices, shape ``[P]``.
    """
    if num_nodes <= max_pivots:
        return torch.arange(num_nodes, dtype=torch.long)

    order = torch.randperm(num_nodes, generator=generator)
    return order[:max_pivots]


def _compute_pivot_distances(adjacency: list[list[int]], pivots: torch.Tensor) -> torch.Tensor:
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
        return torch.empty((0, num_nodes), dtype=torch.float32)

    distances = torch.empty((num_pivots, num_nodes), dtype=torch.float32)
    for pivot_index, pivot in enumerate(pivots.tolist()):
        distances[pivot_index] = _bfs_distances(adjacency, pivot).to(dtype=torch.float32)
    return distances


def _approx_distance(
    source_index: int,
    target_index: int,
    pivot_dist: torch.Tensor,
) -> float:
    """Approximate graph distance for one node pair.

    Parameters
    ----------
    source_index : int
        Source node index.
    target_index : int
        Target node index.
    pivot_dist : torch.Tensor
        Pivot-to-node distances, shape ``[P, N]``.

    Returns
    -------
    float
        Approximate shortest-path distance.
    """
    if pivot_dist.numel() == 0:
        return 1.0

    pivot_i = pivot_dist[:, source_index]
    pivot_j = pivot_dist[:, target_index]
    lower = torch.abs(pivot_i - pivot_j)
    upper = pivot_i + pivot_j
    best_lower = float(lower.max().item())
    best_upper = float(upper.min().item())
    if math.isfinite(best_upper):
        return max((best_lower + best_upper) * 0.5, 1.0)
    return max(best_lower, 1.0)


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


def _resolve_sample_size(num_nodes: int, sample_size: Union[int, str]) -> int:
    """Resolve the effective epoch sampling budget.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    sample_size : int or str
        User-provided sampling budget. ``"auto"`` uses a full epoch for small
        graphs and a fixed-size sample otherwise.

    Returns
    -------
    int
        Effective per-epoch pair budget.

    Raises
    ------
    ValueError
        If ``sample_size`` is invalid.
    """
    if isinstance(sample_size, str):
        if sample_size != "auto":
            raise ValueError("sample_size must be a positive integer or 'auto'.")
        if num_nodes <= _AUTO_FULL_EPOCH_THRESHOLD:
            return num_nodes
        return _AUTO_SAMPLE_THRESHOLD

    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    return sample_size


def _schedule_bounds(distance_data: torch.Tensor) -> tuple[float, float]:
    """Compute the distance bounds used by the ``s_gd2`` schedule.

    Parameters
    ----------
    distance_data : torch.Tensor
        Exact or approximate graph distances.

    Returns
    -------
    tuple[float, float]
        Minimum and maximum positive graph distances.
    """
    positive_distances = distance_data[distance_data > 0]
    if int(positive_distances.numel()) == 0:
        return 1.0, 1.0

    d_min = float(positive_distances.min().item())
    d_max = float(positive_distances.max().item())
    return max(d_min, 1.0), max(d_max, d_min, 1.0)


def _learning_rate(
    step_index: int,
    steps: int,
    d_min: float,
    d_max: float,
    eps: float = _DEFAULT_EPS,
) -> float:
    """Evaluate the exponential ``s_gd2`` learning-rate schedule.

    Parameters
    ----------
    step_index : int
        Zero-based optimization step.
    steps : int
        Total number of optimization steps.
    d_min : float
        Minimum positive graph distance.
    d_max : float
        Maximum positive graph distance.
    eps : float, default=0.01
        Smallest relative step-size scale.

    Returns
    -------
    float
        Step size ``eta_t``.
    """
    eta_max = 1.0 / max(d_min * d_min, _MIN_DISTANCE)
    eta_min = eps / max(d_max * d_max, _MIN_DISTANCE)
    if steps <= 1:
        return eta_max
    lambd = math.log(eta_max / eta_min) / float(steps - 1)
    return eta_max * math.exp(-lambd * float(step_index))


def _shuffled_full_pairs(
    num_nodes: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return all unordered node pairs in shuffled order.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    generator : torch.Generator
        Random generator for deterministic shuffling.
    device : torch.device
        Device for the returned tensors.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Source and target node indices with matching shape ``[S]``.
    """
    pair_indices = torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)
    order = torch.randperm(pair_indices.shape[1], generator=generator, device=device)
    return pair_indices[0, order], pair_indices[1, order]


def _sample_pairs(
    num_nodes: int,
    sample_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of distinct node pairs for one SGD epoch.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    sample_size : int
        Per-epoch pair budget.
    generator : torch.Generator
        Random generator for deterministic sampling.
    device : torch.device
        Device for the returned tensors.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Source and target node indices with matching shape ``[S]``.
    """
    if num_nodes <= 1:
        empty = torch.empty((0,), dtype=torch.long, device=device)
        return empty, empty

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
    return i_indices[valid_pairs], j_indices[valid_pairs]


def _pair_distance(
    source_index: int,
    target_index: int,
    exact_distances: Optional[torch.Tensor],
    pivot_dist: Optional[torch.Tensor],
) -> float:
    """Lookup the target graph distance for one pair.

    Parameters
    ----------
    source_index : int
        Source node index.
    target_index : int
        Target node index.
    exact_distances : torch.Tensor, optional
        Exact distance matrix with shape ``[N, N]``.
    pivot_dist : torch.Tensor, optional
        Pivot-to-node distances with shape ``[P, N]``.

    Returns
    -------
    float
        Target distance.
    """
    if exact_distances is not None:
        return float(exact_distances[source_index, target_index].item())
    if pivot_dist is None:
        return 1.0
    return _approx_distance(
        source_index=source_index,
        target_index=target_index,
        pivot_dist=pivot_dist,
    )


def _apply_pair_update(
    positions: torch.Tensor,
    source_index: int,
    target_index: int,
    target_distance: float,
    eta: float,
) -> None:
    """Apply one symmetric Stress-SGD pair update in place.

    Parameters
    ----------
    positions : torch.Tensor
        Current positions, shape ``[N, 2]``.
    source_index : int
        Source node index.
    target_index : int
        Target node index.
    target_distance : float
        Desired graph distance for the pair.
    eta : float
        Current learning rate.

    Returns
    -------
    None
        Positions are updated in place.
    """
    weight = 1.0 / max(target_distance * target_distance, _MIN_DISTANCE)
    mu = min(weight * eta, 1.0)
    delta = positions[target_index] - positions[source_index]
    current_distance = float(torch.linalg.norm(delta).item())
    current_distance = max(current_distance, _MIN_DISTANCE)
    ratio = (current_distance - target_distance) / (2.0 * current_distance)
    movement = mu * ratio * delta
    positions[source_index] += movement
    positions[target_index] -= movement


def layout_stress_sgd(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 30,
    seed: int = 42,
    sample_size: Union[int, str] = "auto",
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
    steps : int, default=30
        Number of SGD epochs.
    seed : int, default=42
        Random seed.
    sample_size : int or str, default="auto"
        Per-epoch pair budget. ``"auto"`` performs a full shuffled epoch for
        graphs with at most ``1000`` nodes and samples ``1000`` pairs
        otherwise. Explicit values greater than or equal to ``num_nodes`` also
        request a full epoch, preserving the previous public API.
    trace_every : int, default=0
        If greater than zero, record snapshots every ``trace_every`` steps.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, list[torch.Tensor]]
        Final positions ``[N, 2]``, or ``(positions, traces)`` if tracing.

    Raises
    ------
    ValueError
        If the graph is disconnected.
    """
    del node_sizes

    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if trace_every < 0:
        raise ValueError("trace_every must be non-negative")

    device = edge_index.device
    positions = torch.zeros((num_nodes, 2), dtype=torch.float32, device=device)
    traces: list[torch.Tensor] = []
    if num_nodes == 0:
        return (positions, traces) if trace_every > 0 else positions
    if num_nodes == 1:
        return (positions, traces) if trace_every > 0 else positions

    generator_device = device.type if device.type != "mps" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)

    adjacency = _build_undirected_adjacency(edge_index, num_nodes)
    if not _is_connected(adjacency):
        raise ValueError("Stress-SGD requires a connected graph.")

    effective_sample_size = _resolve_sample_size(num_nodes, sample_size)
    use_full_epoch = effective_sample_size >= num_nodes and num_nodes <= _AUTO_FULL_EPOCH_THRESHOLD

    exact_distances: Optional[torch.Tensor] = None
    pivot_dist: Optional[torch.Tensor] = None
    if num_nodes <= _EXACT_DISTANCE_THRESHOLD:
        exact_distances_cpu = _all_pairs_shortest_paths(adjacency)
        exact_distances = exact_distances_cpu.to(device=device)
        d_min, d_max = _schedule_bounds(exact_distances_cpu)
    else:
        pivots_cpu = _choose_pivots(num_nodes, min(num_nodes, _MAX_PIVOTS), generator)
        pivot_dist_cpu = _compute_pivot_distances(adjacency, pivots_cpu)
        pivot_dist = pivot_dist_cpu.to(device=device)
        d_min, d_max = _schedule_bounds(pivot_dist_cpu)

    positions = torch.rand((num_nodes, 2), generator=generator, device=device, dtype=torch.float32)

    if trace_every > 0 and steps == 0:
        _trace_snapshot(traces, positions)

    for step_index in range(steps):
        eta = _learning_rate(
            step_index=step_index,
            steps=steps,
            d_min=d_min,
            d_max=d_max,
        )
        if use_full_epoch:
            i_indices, j_indices = _shuffled_full_pairs(
                num_nodes=num_nodes,
                generator=generator,
                device=device,
            )
        else:
            i_indices, j_indices = _sample_pairs(
                num_nodes=num_nodes,
                sample_size=effective_sample_size,
                generator=generator,
                device=device,
            )

        for pair_index in range(int(i_indices.shape[0])):
            source_index = int(i_indices[pair_index].item())
            target_index = int(j_indices[pair_index].item())
            target_distance = _pair_distance(
                source_index=source_index,
                target_index=target_index,
                exact_distances=exact_distances,
                pivot_dist=pivot_dist,
            )
            _apply_pair_update(
                positions=positions,
                source_index=source_index,
                target_index=target_index,
                target_distance=target_distance,
                eta=eta,
            )

        if trace_every > 0 and (step_index + 1) % trace_every == 0:
            _trace_snapshot(traces, positions)

    return (positions, traces) if trace_every > 0 else positions
