"""Maxent-stress graph layout.

The implementation combines a stress term based on graph-theoretic distances
with the logarithmic entropy repulsion from Gansner et al. For small graphs it
uses full BFS shortest paths; for larger graphs it falls back to a KK-style
pivot approximation for the stress term.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

import torch

from dagua.layout.classic.pivot_mds import layout_pivot_mds

_MIN_DISTANCE = 1.0e-3
_FULL_STRESS_LIMIT = 1_000
_MAJORIZATION_NODE_LIMIT = 5_000
_PIVOT_COUNT = 50
_SAMPLED_REPULSION_NEIGHBORS = 96


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


def _initialize_positions(num_nodes: int, device: torch.device, seed: int) -> torch.Tensor:
    """Create deterministic random initial coordinates.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    device : torch.device
        Device for the result.
    seed : int
        Random seed.

    Returns
    -------
    torch.Tensor
        Initial positions with shape ``[N, 2]``.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn((num_nodes, 2), generator=generator, dtype=torch.float32).to(device)


def _normalize_positions(positions: torch.Tensor, extent: float) -> torch.Tensor:
    """Center and scale coordinates into a bounded drawing box.

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


def _build_undirected_adjacency(edge_index: torch.Tensor, num_nodes: int) -> list[list[int]]:
    """Build an undirected adjacency list from the edge tensor.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    list[list[int]]
        One neighbor list per node.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")

    adjacency_sets = [set() for _ in range(num_nodes)]
    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    for source, target in zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist()):
        if source < 0 or source >= num_nodes or target < 0 or target >= num_nodes:
            raise ValueError("edge_index contains a node index outside [0, num_nodes).")
        if source == target:
            continue
        adjacency_sets[source].add(target)
        adjacency_sets[target].add(source)

    return [sorted(neighbors) for neighbors in adjacency_sets]


def _connected_components(adjacency: list[list[int]]) -> torch.Tensor:
    """Label connected components in the undirected graph.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency list.

    Returns
    -------
    torch.Tensor
        Component labels with shape ``[N]``.
    """
    num_nodes = len(adjacency)
    component_ids = torch.full((num_nodes,), -1, dtype=torch.long)
    component_index = 0

    for start in range(num_nodes):
        if int(component_ids[start].item()) >= 0:
            continue

        frontier: deque[int] = deque([start])
        component_ids[start] = component_index
        while frontier:
            node = frontier.popleft()
            for neighbor in adjacency[node]:
                if int(component_ids[neighbor].item()) >= 0:
                    continue
                component_ids[neighbor] = component_index
                frontier.append(neighbor)

        component_index += 1

    return component_ids


def _bfs_distances(adjacency: list[list[int]], source: int) -> torch.Tensor:
    """Compute full shortest-path distances from one source node.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency list.
    source : int
        BFS root node index.

    Returns
    -------
    torch.Tensor
        Integer distances with shape ``[N]`` and ``-1`` for unreachable nodes.
    """
    num_nodes = len(adjacency)
    distances = torch.full((num_nodes,), -1, dtype=torch.long)
    distances[source] = 0
    frontier: deque[int] = deque([source])

    while frontier:
        node = frontier.popleft()
        next_distance = int(distances[node].item()) + 1
        for neighbor in adjacency[node]:
            if int(distances[neighbor].item()) >= 0:
                continue
            distances[neighbor] = next_distance
            frontier.append(neighbor)

    return distances


def _cross_component_distance(num_nodes: int, average_edge_cost: float = 1.0) -> float:
    """Compute OGDF's disconnected-pair fallback distance.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    average_edge_cost : float, default=1.0
        Average edge cost used by the shortest-path objective.

    Returns
    -------
    float
        Replacement distance for unreachable node pairs.
    """
    return average_edge_cost * math.sqrt(float(max(num_nodes, 1)))


def _all_pairs_shortest_paths(adjacency: list[list[int]]) -> torch.Tensor:
    """Compute the full BFS shortest-path matrix for a small graph.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency list.

    Returns
    -------
    torch.Tensor
        Float shortest-path matrix with shape ``[N, N]``.
    """
    num_nodes = len(adjacency)
    rows = [_bfs_distances(adjacency, node) for node in range(num_nodes)]
    if not rows:
        return torch.empty((0, 0), dtype=torch.float32)

    distances = torch.stack(rows, dim=0).to(dtype=torch.float32)
    unreachable = distances < 0
    if bool(unreachable.any()):
        distances = distances.clone()
        distances[unreachable] = _cross_component_distance(num_nodes)
    return distances


def _full_stress_terms(
    adjacency: list[list[int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build exact stress pairs from the all-pairs shortest-path matrix.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency list.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Unique reachable node pairs and their graph distances.
    """
    distances = _all_pairs_shortest_paths(adjacency)
    if distances.numel() == 0:
        empty = torch.empty((0,), dtype=torch.long)
        return empty, empty, torch.empty((0,), dtype=torch.float32)

    upper = torch.triu_indices(distances.shape[0], distances.shape[1], offset=1)
    pair_distances = distances[upper[0], upper[1]]
    return (
        upper[0],
        upper[1],
        pair_distances.to(dtype=torch.float32),
    )


def _choose_pivots(
    component_ids: torch.Tensor,
    max_pivots: int,
    seed: int,
) -> torch.Tensor:
    """Choose pivot nodes while covering every connected component.

    Parameters
    ----------
    component_ids : torch.Tensor
        Component labels with shape ``[N]``.
    max_pivots : int
        Maximum number of pivots to select.
    seed : int
        Random seed for the remaining pivot slots.

    Returns
    -------
    torch.Tensor
        Pivot indices with shape ``[P]``.
    """
    num_nodes = int(component_ids.shape[0])
    if num_nodes == 0:
        return torch.empty((0,), dtype=torch.long)
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

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    pivot_mask = torch.zeros(num_nodes, dtype=torch.bool)
    pivot_mask[torch.tensor(pivots, dtype=torch.long)] = True
    candidates = torch.arange(num_nodes, dtype=torch.long)[~pivot_mask]
    permutation = torch.randperm(int(candidates.shape[0]), generator=generator)
    extra = candidates[permutation[:remaining]]
    return torch.cat([torch.tensor(pivots, dtype=torch.long), extra], dim=0)


def _pivot_stress_terms(
    adjacency: list[list[int]],
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a KK-style pivot approximation for large-graph stress.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency list.
    seed : int
        Random seed for pivot selection.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Pivot indices with shape ``[P]`` and node-to-pivot distances with shape
        ``[N, P]``. Unreachable entries remain ``-1``.
    """
    component_ids = _connected_components(adjacency)
    pivots = _choose_pivots(component_ids, min(_PIVOT_COUNT, len(adjacency)), seed)
    if int(pivots.numel()) == 0:
        return pivots, torch.empty((len(adjacency), 0), dtype=torch.float32)

    rows = [_bfs_distances(adjacency, int(pivot.item())) for pivot in pivots]
    distances = torch.stack(rows, dim=0).transpose(0, 1).to(dtype=torch.float32)
    unreachable = distances < 0
    if bool(unreachable.any()):
        distances = distances.clone()
        distances[unreachable] = _cross_component_distance(len(adjacency))
    return pivots, distances


def _full_non_edge_pairs(adjacency: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Enumerate all unique non-edge pairs for exact entropy repulsion.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency list.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Unique non-edge pairs ``(i, j)`` with ``i < j``.
    """
    num_nodes = len(adjacency)
    if num_nodes <= 1:
        empty = torch.empty((0,), dtype=torch.long)
        return empty, empty

    adjacency_mask = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    for source, neighbors in enumerate(adjacency):
        if neighbors:
            adjacency_mask[source, torch.tensor(neighbors, dtype=torch.long)] = True

    upper = torch.triu_indices(num_nodes, num_nodes, offset=1)
    mask = ~adjacency_mask[upper[0], upper[1]]
    return upper[0][mask], upper[1][mask]


def _sample_non_edges(
    adjacency: list[list[int]],
    step: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Sample random non-edge pairs for entropy repulsion.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency list.
    step : int
        Optimization step.
    seed : int
        Base random seed.
    device : torch.device
        Device for the returned tensors.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, int]
        Sampled source indices, sampled target indices, and the total number of
        unique non-edge pairs in the graph.
    """
    num_nodes = len(adjacency)
    total_pairs = num_nodes * (num_nodes - 1) // 2
    edge_count = sum(len(neighbors) for neighbors in adjacency) // 2
    total_non_edges = max(total_pairs - edge_count, 0)
    if total_non_edges == 0:
        empty = torch.empty((0,), dtype=torch.long, device=device)
        return empty, empty, 0

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + step + 1)
    sample_size = min(
        total_non_edges,
        max(num_nodes, num_nodes * _SAMPLED_REPULSION_NEIGHBORS // 2),
    )

    adjacency_sets = [set(neighbors) | {node} for node, neighbors in enumerate(adjacency)]
    sources: list[int] = []
    targets: list[int] = []
    while len(sources) < sample_size:
        remaining = sample_size - len(sources)
        batch_size = max(remaining * 3, 16)
        candidate_sources = torch.randint(
            0,
            num_nodes,
            (batch_size,),
            generator=generator,
            dtype=torch.long,
        ).tolist()
        candidate_targets = torch.randint(
            0,
            num_nodes,
            (batch_size,),
            generator=generator,
            dtype=torch.long,
        ).tolist()
        for source, target in zip(candidate_sources, candidate_targets):
            if target not in adjacency_sets[source]:
                sources.append(min(source, target))
                targets.append(max(source, target))
                if len(sources) >= sample_size:
                    break

    return (
        torch.tensor(sources, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
        total_non_edges,
    )


def _stress_term(
    positions: torch.Tensor,
    stress_src: torch.Tensor,
    stress_dst: torch.Tensor,
    stress_lengths: torch.Tensor,
    pivot_indices: torch.Tensor,
    pivot_distances: torch.Tensor,
) -> torch.Tensor:
    """Evaluate either exact or pivot-approximated stress.

    Parameters
    ----------
    positions : torch.Tensor
        Current coordinates with shape ``[N, 2]``.
    stress_src : torch.Tensor
        Exact stress-pair source indices.
    stress_dst : torch.Tensor
        Exact stress-pair target indices.
    stress_lengths : torch.Tensor
        Graph distances for exact stress pairs.
    pivot_indices : torch.Tensor
        Pivot node indices with shape ``[P]``.
    pivot_distances : torch.Tensor
        Node-to-pivot graph distances with shape ``[N, P]``.

    Returns
    -------
    torch.Tensor
        Scalar stress term.
    """
    if stress_src.numel() > 0:
        src = stress_src.to(device=positions.device)
        dst = stress_dst.to(device=positions.device)
        targets = stress_lengths.to(device=positions.device)
        distances = torch.linalg.norm(positions[src] - positions[dst], dim=1).clamp(
            min=_MIN_DISTANCE
        )
        weights = targets.reciprocal().square()
        return (weights * (distances - targets).square()).sum()

    if pivot_indices.numel() == 0:
        return torch.tensor(0.0, dtype=positions.dtype, device=positions.device)

    pivot_positions = positions[pivot_indices.to(device=positions.device)]
    geometric = torch.cdist(positions, pivot_positions).clamp(min=_MIN_DISTANCE)
    targets = pivot_distances.to(device=positions.device)
    reachable = targets > 0
    safe_targets = torch.where(reachable, targets, torch.ones_like(targets))
    weights = torch.where(reachable, safe_targets.reciprocal().square(), torch.zeros_like(targets))
    return (weights * (geometric - safe_targets).square()).sum()


def _entropy_term(
    positions: torch.Tensor,
    non_edge_src: torch.Tensor,
    non_edge_dst: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Evaluate the non-edge logarithmic entropy term.

    Parameters
    ----------
    positions : torch.Tensor
        Current coordinates with shape ``[N, 2]``.
    non_edge_src : torch.Tensor
        Non-edge source indices.
    non_edge_dst : torch.Tensor
        Non-edge target indices.
    scale : float
        Scaling factor for Monte Carlo estimates. Use ``1.0`` for exact sums.

    Returns
    -------
    torch.Tensor
        Scalar entropy term ``-sum(log ||p_i - p_j||)``.
    """
    if non_edge_src.numel() == 0:
        return torch.tensor(0.0, dtype=positions.dtype, device=positions.device)

    nonedge_distances = torch.linalg.norm(
        positions[non_edge_src] - positions[non_edge_dst],
        dim=1,
    ).clamp(min=_MIN_DISTANCE)
    return -torch.log(nonedge_distances).sum() * scale


def _maxent_stress_loss(
    positions: torch.Tensor,
    stress_src: torch.Tensor,
    stress_dst: torch.Tensor,
    stress_lengths: torch.Tensor,
    pivot_indices: torch.Tensor,
    pivot_distances: torch.Tensor,
    non_edge_src: torch.Tensor,
    non_edge_dst: torch.Tensor,
    alpha: float,
    use_entropy: bool = True,
) -> torch.Tensor:
    """Evaluate the prepared maxent-stress objective.

    Parameters
    ----------
    positions : torch.Tensor
        Current coordinates with shape ``[N, 2]``.
    stress_src : torch.Tensor
        Exact stress-pair source indices.
    stress_dst : torch.Tensor
        Exact stress-pair target indices.
    stress_lengths : torch.Tensor
        Exact graph distances for the stress pairs.
    pivot_indices : torch.Tensor
        Pivot node indices for large-graph approximation.
    pivot_distances : torch.Tensor
        Node-to-pivot graph distances with shape ``[N, P]``.
    non_edge_src : torch.Tensor
        Non-edge source indices.
    non_edge_dst : torch.Tensor
        Non-edge target indices.
    alpha : float
        Entropy repulsion weight.
    use_entropy : bool, default=True
        Whether to include the logarithmic entropy repulsion term.

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    loss = _stress_term(
        positions,
        stress_src,
        stress_dst,
        stress_lengths,
        pivot_indices,
        pivot_distances,
    )
    if use_entropy:
        loss = loss + alpha * _entropy_term(positions, non_edge_src, non_edge_dst, scale=1.0)
    return loss


def _majorization_iteration(
    positions: torch.Tensor,
    graph_distances: torch.Tensor,
    weight_matrix: torch.Tensor,
) -> None:
    """Run one in-place Gauss-Seidel stress-majorization iteration.

    Parameters
    ----------
    positions : torch.Tensor
        Current coordinates with shape ``[N, 2]`` on CPU.
    graph_distances : torch.Tensor
        All-pairs graph distances with shape ``[N, N]``.
    weight_matrix : torch.Tensor
        Stress weights ``1 / d_ij^2`` with shape ``[N, N]``.

    Returns
    -------
    None
        The position tensor is updated in place.
    """
    num_nodes = int(positions.shape[0])
    for node_index in range(num_nodes):
        current_x = float(positions[node_index, 0].item())
        current_y = float(positions[node_index, 1].item())
        new_x = 0.0
        new_y = 0.0
        total_weight = 0.0

        for other_index in range(num_nodes):
            if node_index == other_index:
                continue

            weight = float(weight_matrix[node_index, other_index].item())
            desired_distance = float(graph_distances[node_index, other_index].item())
            other_x = float(positions[other_index, 0].item())
            other_y = float(positions[other_index, 1].item())
            delta_x = current_x - other_x
            delta_y = current_y - other_y
            euclidean_distance = math.hypot(delta_x, delta_y)

            vote_x = other_x
            vote_y = other_y
            if euclidean_distance != 0.0:
                vote_x += desired_distance * (current_x - vote_x) / euclidean_distance
                vote_y += desired_distance * (current_y - vote_y) / euclidean_distance

            new_x += weight * vote_x
            new_y += weight * vote_y
            total_weight += weight

        if total_weight != 0.0:
            positions[node_index, 0] = new_x / total_weight
            positions[node_index, 1] = new_y / total_weight


def _layout_stress_majorization(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor],
    steps: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Run OGDF-style stress majorization for small graphs.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node sizes used only for final scaling.
    steps : int
        Number of majorization iterations.
    seed : int
        Seed for the PivotMDS initialization.
    device : torch.device
        Target device for the returned positions.

    Returns
    -------
    torch.Tensor
        Normalized positions with shape ``[N, 2]``.
    """
    adjacency = _build_undirected_adjacency(edge_index, num_nodes)
    graph_distances = _all_pairs_shortest_paths(adjacency).to(dtype=torch.float64)
    weight_matrix = torch.zeros_like(graph_distances)
    off_diagonal_mask = ~torch.eye(num_nodes, dtype=torch.bool)
    weight_matrix[off_diagonal_mask] = graph_distances[off_diagonal_mask].reciprocal().square()

    positions = layout_pivot_mds(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        n_pivots=min(_PIVOT_COUNT, num_nodes),
        seed=seed,
    ).to(device="cpu", dtype=torch.float64)

    for _ in range(steps):
        _majorization_iteration(
            positions=positions,
            graph_distances=graph_distances,
            weight_matrix=weight_matrix,
        )

    extent = _layout_extent(num_nodes, node_sizes)
    return _normalize_positions(positions.to(dtype=torch.float32), extent).to(
        dtype=torch.float32,
        device=device,
    )


def _layout_maxent_stress_gradient(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor],
    steps: int,
    alpha: float,
    seed: int,
    use_entropy: bool,
    device: torch.device,
) -> torch.Tensor:
    """Run the existing gradient-based maxent-stress fallback.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node sizes used only for final scaling.
    steps : int
        Number of Adam updates.
    alpha : float
        Entropy repulsion weight.
    seed : int
        Random seed.
    use_entropy : bool
        Whether to include the logarithmic entropy term.
    device : torch.device
        Target output device.

    Returns
    -------
    torch.Tensor
        Normalized positions with shape ``[N, 2]``.
    """
    adjacency = _build_undirected_adjacency(edge_index, num_nodes)
    if num_nodes <= _FULL_STRESS_LIMIT:
        stress_src, stress_dst, stress_lengths = _full_stress_terms(adjacency)
        pivot_indices = torch.empty((0,), dtype=torch.long)
        pivot_distances = torch.empty((num_nodes, 0), dtype=torch.float32)
        if use_entropy:
            full_non_edge_src, full_non_edge_dst = _full_non_edge_pairs(adjacency)
        else:
            full_non_edge_src = torch.empty((0,), dtype=torch.long)
            full_non_edge_dst = torch.empty((0,), dtype=torch.long)
    else:
        stress_src = torch.empty((0,), dtype=torch.long)
        stress_dst = torch.empty((0,), dtype=torch.long)
        stress_lengths = torch.empty((0,), dtype=torch.float32)
        pivot_indices, pivot_distances = _pivot_stress_terms(adjacency, seed)
        full_non_edge_src = torch.empty((0,), dtype=torch.long)
        full_non_edge_dst = torch.empty((0,), dtype=torch.long)

    positions = layout_pivot_mds(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        n_pivots=min(_PIVOT_COUNT, num_nodes),
        seed=seed,
    ).to(device=device, dtype=torch.float32)
    positions = positions.requires_grad_(True)
    initial_lr = min(0.04, 0.8 / float(max(num_nodes, 1)))
    final_lr = max(initial_lr * 0.1, initial_lr / math.sqrt(float(max(steps, 1))))
    optimizer = torch.optim.Adam([positions], lr=initial_lr)

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        if num_nodes <= _FULL_STRESS_LIMIT:
            loss = _maxent_stress_loss(
                positions,
                stress_src,
                stress_dst,
                stress_lengths,
                pivot_indices,
                pivot_distances,
                full_non_edge_src.to(device=positions.device),
                full_non_edge_dst.to(device=positions.device),
                alpha,
                use_entropy=use_entropy,
            )
        else:
            loss = _stress_term(
                positions,
                stress_src,
                stress_dst,
                stress_lengths,
                pivot_indices,
                pivot_distances,
            )
            if use_entropy:
                sampled_src, sampled_dst, total_non_edges = _sample_non_edges(
                    adjacency,
                    step,
                    seed,
                    positions.device,
                )
                sample_count = max(int(sampled_src.numel()), 1)
                loss = loss + alpha * _entropy_term(
                    positions,
                    sampled_src,
                    sampled_dst,
                    scale=(
                        float(total_non_edges) / float(sample_count) if total_non_edges > 0 else 1.0
                    ),
                )
        loss.backward()
        optimizer.step()
        fraction = float(step + 1) / float(max(steps, 1))
        optimizer.param_groups[0]["lr"] = initial_lr + (final_lr - initial_lr) * fraction

    extent = _layout_extent(num_nodes, node_sizes)
    return _normalize_positions(positions.detach(), extent).to(dtype=torch.float32, device=device)


def layout_maxent_stress(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 200,
    alpha: float = 1.0,
    seed: int = 42,
    use_entropy: bool = False,
    use_majorization: bool = True,
) -> torch.Tensor:
    """Lay out a graph with OGDF-style stress minimization and optional entropy.

    Reference
    ---------
    Gansner, Hu, and North, "A Maxent-Stress Model for Graph Layout" (2013).

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Optional node sizes used only for final scaling.
    steps : int, default=200
        Number of optimization iterations.
    alpha : float, default=1.0
        Entropy repulsion weight.
    seed : int, default=42
        Random seed.
    use_entropy : bool, default=False
        Include the logarithmic entropy repulsion term. ``False`` matches
        OGDF's pure stress objective more closely.
    use_majorization : bool, default=True
        Use OGDF's sequential stress-majorization update for small pure-stress
        problems. The implementation falls back to the gradient path when
        entropy repulsion is enabled, the graph is too large, or the caller
        requests a non-default iteration count.

    Returns
    -------
    torch.Tensor
        Node positions with shape ``[N, 2]``.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if alpha < 0:
        raise ValueError("alpha must be non-negative.")

    device = _layout_device(edge_index, node_sizes)
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32, device=device)

    if (
        use_majorization
        and not use_entropy
        and num_nodes <= _MAJORIZATION_NODE_LIMIT
        and steps == 200
    ):
        return _layout_stress_majorization(
            edge_index=edge_index,
            num_nodes=num_nodes,
            node_sizes=node_sizes,
            steps=steps,
            seed=seed,
            device=device,
        )

    return _layout_maxent_stress_gradient(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        steps=steps,
        alpha=alpha,
        seed=seed,
        use_entropy=use_entropy,
        device=device,
    )
