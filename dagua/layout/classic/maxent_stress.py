"""Maxent-stress graph layout.

The implementation combines sparse stress terms over short graph distances with
sampled logarithmic repulsion over non-edge pairs, optimized with Adam.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import torch

_MIN_DISTANCE = 1.0e-3
_TWO_HOP_LIMIT = 2_000
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


def _stress_pairs(
    adjacency: list[list[int]],
    include_two_hop: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect sparse stress pairs and their graph distances.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency list.
    include_two_hop : bool
        Whether to include pairs at graph distance two.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Source indices, target indices, and target graph distances.
    """
    sources: list[int] = []
    targets: list[int] = []
    lengths: list[float] = []
    seen: set[tuple[int, int]] = set()

    for source, neighbors in enumerate(adjacency):
        for target in neighbors:
            key = (min(source, target), max(source, target))
            if key not in seen:
                seen.add(key)
                sources.append(key[0])
                targets.append(key[1])
                lengths.append(1.0)

        if not include_two_hop:
            continue

        queue: deque[tuple[int, int]] = deque([(source, 0)])
        visited = {source}
        while queue:
            node, depth = queue.popleft()
            if depth >= 2:
                continue
            for neighbor in adjacency[node]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                next_depth = depth + 1
                if next_depth == 2:
                    key = (min(source, neighbor), max(source, neighbor))
                    if key[0] != key[1] and key not in seen:
                        seen.add(key)
                        sources.append(key[0])
                        targets.append(key[1])
                        lengths.append(2.0)
                queue.append((neighbor, next_depth))

    if not sources:
        empty = torch.empty((0,), dtype=torch.long)
        return empty, empty, torch.empty((0,), dtype=torch.float32)

    return (
        torch.tensor(sources, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.float32),
    )


def _sample_non_edges(
    adjacency: list[list[int]],
    step: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    tuple[torch.Tensor, torch.Tensor]
        Source and target index tensors of equal length.
    """
    num_nodes = len(adjacency)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + step + 1)
    sample_size = max(1, min(num_nodes, _SAMPLED_REPULSION_NEIGHBORS))

    adjacency_sets = [set(neighbors) | {node} for node, neighbors in enumerate(adjacency)]
    sources: list[int] = []
    targets: list[int] = []

    for source in range(num_nodes):
        candidates = torch.randint(
            0,
            num_nodes,
            (sample_size * 3,),
            generator=generator,
            dtype=torch.long,
        ).tolist()
        accepted = 0
        for target in candidates:
            if target not in adjacency_sets[source]:
                sources.append(source)
                targets.append(target)
                accepted += 1
                if accepted >= sample_size:
                    break

    return torch.tensor(sources, dtype=torch.long, device=device), torch.tensor(
        targets,
        dtype=torch.long,
        device=device,
    )


def _maxent_stress_loss(
    positions: torch.Tensor,
    stress_src: torch.Tensor,
    stress_dst: torch.Tensor,
    stress_lengths: torch.Tensor,
    adjacency: list[list[int]],
    alpha: float,
    seed: int,
    step: int,
) -> torch.Tensor:
    """Evaluate the sampled maxent-stress objective.

    Parameters
    ----------
    positions : torch.Tensor
        Current coordinates with shape ``[N, 2]``.
    stress_src : torch.Tensor
        Stress-pair source indices.
    stress_dst : torch.Tensor
        Stress-pair target indices.
    stress_lengths : torch.Tensor
        Graph distances for stress pairs.
    adjacency : list[list[int]]
        Undirected adjacency list.
    alpha : float
        Repulsion weight.
    seed : int
        Base random seed.
    step : int
        Optimization step.

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    stress = torch.tensor(0.0, dtype=positions.dtype, device=positions.device)
    if stress_src.numel() > 0:
        src = stress_src.to(device=positions.device)
        dst = stress_dst.to(device=positions.device)
        targets = stress_lengths.to(device=positions.device)
        distances = torch.linalg.norm(positions[src] - positions[dst], dim=1).clamp(
            min=_MIN_DISTANCE
        )
        weights = targets.reciprocal().square()
        stress = (weights * (distances - targets).square()).mean()

    repulsion = torch.tensor(0.0, dtype=positions.dtype, device=positions.device)
    sampled_src, sampled_dst = _sample_non_edges(adjacency, step, seed, positions.device)
    if sampled_src.numel() > 0:
        nonedge_distances = torch.linalg.norm(
            positions[sampled_src] - positions[sampled_dst],
            dim=1,
        ).clamp(min=_MIN_DISTANCE)
        repulsion = -torch.log(nonedge_distances).mean()

    gravity = 0.01 * positions.square().mean()
    return stress + alpha * repulsion + gravity


def layout_maxent_stress(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 200,
    alpha: float = 1.0,
    seed: int = 42,
) -> torch.Tensor:
    """Lay out a graph with sparse stress plus entropy repulsion.

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
        Number of Adam updates.
    alpha : float, default=1.0
        Repulsion weight.
    seed : int, default=42
        Random seed.

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

    adjacency = _build_undirected_adjacency(edge_index, num_nodes)
    stress_src, stress_dst, stress_lengths = _stress_pairs(
        adjacency,
        include_two_hop=num_nodes <= _TWO_HOP_LIMIT,
    )

    positions = _initialize_positions(num_nodes, device, seed).requires_grad_(True)
    optimizer = torch.optim.Adam([positions], lr=0.08)

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = _maxent_stress_loss(
            positions,
            stress_src,
            stress_dst,
            stress_lengths,
            adjacency,
            alpha,
            seed,
            step,
        )
        loss.backward()
        optimizer.step()
        optimizer.param_groups[0]["lr"] = (
            0.08 * (1.0 - float(step + 1) / float(max(steps, 1))) + 0.01
        )

    extent = _layout_extent(num_nodes, node_sizes)
    return _normalize_positions(positions.detach(), extent).to(dtype=torch.float32, device=device)
