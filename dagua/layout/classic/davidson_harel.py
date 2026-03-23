"""Davidson-Harel simulated annealing layout.

This module implements a small simulated-annealing graph drawing routine with
five energy terms: node distribution, border repulsion, edge lengths, edge
crossings, and node-edge proximity.
"""

from __future__ import annotations

from typing import Optional

import torch

_MIN_DISTANCE = 1.0e-3
_BORDER_WEIGHT = 0.1
_EDGE_LENGTH_WEIGHT = 0.2
_CROSSING_WEIGHT = 2.0
_NODE_EDGE_WEIGHT = 0.5
_COOLING_FACTOR = 0.75


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
        Bounding-box half-width.
    """
    if node_sizes is None or node_sizes.numel() == 0:
        return max(float(max(num_nodes, 1)) ** 0.5 * 5.0, 1.0)

    max_size = float(node_sizes.to(dtype=torch.float32, device="cpu").max().item())
    return max(max_size * max(float(max(num_nodes, 1)) ** 0.5, 1.0) * 2.0, 1.0)


def _unique_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor] = None,
) -> tuple[list[tuple[int, int]], torch.Tensor]:
    """Convert an edge tensor into unique undirected edges and weights.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    edge_weights : torch.Tensor, optional
        Optional per-edge weights with shape ``[E]``.

    Returns
    -------
    tuple[list[tuple[int, int]], torch.Tensor]
        Unique undirected edges and their aggregated weights with shape
        ``[E_unique]``. Parallel or mirrored edges are summed so the collapsed
        undirected energy term preserves total attraction strength.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")

    seen: dict[tuple[int, int], float] = {}
    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    if edge_weights is None:
        weights_cpu = torch.ones((edge_index.shape[1],), dtype=torch.float32)
    else:
        weights_cpu = edge_weights.detach().to(device="cpu", dtype=torch.float32)

    for edge_id, (source, target) in enumerate(
        zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist())
    ):
        if source < 0 or source >= num_nodes or target < 0 or target >= num_nodes:
            raise ValueError("edge_index contains a node index outside [0, num_nodes).")
        if source == target:
            continue
        pair = (min(source, target), max(source, target))
        seen[pair] = seen.get(pair, 0.0) + float(weights_cpu[edge_id].item())

    ordered_edges = sorted(seen)
    ordered_weights = torch.tensor(
        [seen[edge] for edge in ordered_edges],
        dtype=torch.float32,
    )
    return ordered_edges, ordered_weights


def _initialize_positions(
    num_nodes: int,
    extent: float,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Create deterministic random coordinates inside the drawing box.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    extent : float
        Half-width of the drawing box.
    device : torch.device
        Device for the result.
    seed : int
        Random seed.

    Returns
    -------
    torch.Tensor
        Initial coordinates with shape ``[N, 2]``.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return ((torch.rand((num_nodes, 2), generator=generator) * 2.0) - 1.0).to(device) * extent


def _orientation(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> float:
    """Compute the signed triangle area used by segment intersection tests.

    Parameters
    ----------
    a : torch.Tensor
        First point with shape ``[2]``.
    b : torch.Tensor
        Second point with shape ``[2]``.
    c : torch.Tensor
        Third point with shape ``[2]``.

    Returns
    -------
    float
        Signed cross product value.
    """
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _segments_intersect(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> bool:
    """Test whether two line segments intersect.

    Parameters
    ----------
    a : torch.Tensor
        First endpoint of the first segment.
    b : torch.Tensor
        Second endpoint of the first segment.
    c : torch.Tensor
        First endpoint of the second segment.
    d : torch.Tensor
        Second endpoint of the second segment.

    Returns
    -------
    bool
        ``True`` if the segments intersect.
    """
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    return (o1 == 0.0 or o2 == 0.0 or o1 * o2 < 0.0) and (o3 == 0.0 or o4 == 0.0 or o3 * o4 < 0.0)


def _point_segment_distance(
    point: torch.Tensor, start: torch.Tensor, end: torch.Tensor
) -> torch.Tensor:
    """Compute the Euclidean distance from a point to a segment.

    Parameters
    ----------
    point : torch.Tensor
        Point with shape ``[2]``.
    start : torch.Tensor
        Segment start point.
    end : torch.Tensor
        Segment end point.

    Returns
    -------
    torch.Tensor
        Distance scalar.
    """
    segment = end - start
    denom = segment.dot(segment).clamp(min=_MIN_DISTANCE)
    projection = ((point - start).dot(segment) / denom).clamp(0.0, 1.0)
    nearest = start + projection * segment
    return torch.linalg.norm(point - nearest)


def _scale_denominator(numerator_count: int) -> float:
    """Return a non-zero normalization denominator for one energy term.

    Parameters
    ----------
    numerator_count : int
        Expected scale factor for the corresponding summed energy term.

    Returns
    -------
    float
        Positive normalization denominator.
    """
    return float(max(numerator_count, 1))


def _energy(
    positions: torch.Tensor,
    edges: list[tuple[int, int]],
    extent: float,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Evaluate the Davidson-Harel layout energy.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edges : list[tuple[int, int]]
        Unique undirected edges.
    extent : float
        Half-width of the drawing box.
    edge_weights : torch.Tensor, optional
        Optional edge weights aligned with ``edges`` and shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Scalar energy value.

    Notes
    -----
    The paper defines the individual energy terms as sums. This implementation
    keeps that formulation, then normalizes each term by its natural graph-size
    scale so the fixed weights remain comparable across different graph sizes.
    """
    num_nodes = int(positions.shape[0])
    distribution = torch.tensor(0.0, dtype=positions.dtype, device=positions.device)
    if num_nodes > 1:
        src, dst = torch.triu_indices(num_nodes, num_nodes, offset=1, device=positions.device)
        squared_distances = (
            (positions[src] - positions[dst]).square().sum(dim=1).clamp(min=_MIN_DISTANCE)
        )
        distribution = squared_distances.reciprocal().sum()

    border_distances = torch.stack(
        [
            positions[:, 0] + extent,
            extent - positions[:, 0],
            positions[:, 1] + extent,
            extent - positions[:, 1],
        ],
        dim=1,
    ).clamp(min=_MIN_DISTANCE)
    border = border_distances.reciprocal().square().sum()

    edge_length = torch.tensor(0.0, dtype=positions.dtype, device=positions.device)
    if edges:
        edge_weight_tensor = (
            torch.ones((len(edges),), dtype=positions.dtype, device=positions.device)
            if edge_weights is None
            else edge_weights.to(device=positions.device, dtype=positions.dtype)
        )
        edge_lengths = [
            torch.linalg.norm(positions[source] - positions[target]).square()
            * edge_weight_tensor[index]
            for index, (source, target) in enumerate(edges)
        ]
        edge_length = torch.stack(edge_lengths).sum()

    crossings = 0.0
    for index, (a, b) in enumerate(edges):
        for c, d in edges[index + 1 :]:
            if len({a, b, c, d}) < 4:
                continue
            if _segments_intersect(positions[a], positions[b], positions[c], positions[d]):
                crossings += 1.0
    crossing_energy = torch.tensor(crossings, dtype=positions.dtype, device=positions.device)

    node_edge = torch.tensor(0.0, dtype=positions.dtype, device=positions.device)
    penalties: list[torch.Tensor] = []
    for node in range(num_nodes):
        for source, target in edges:
            if node in (source, target):
                continue
            distance = _point_segment_distance(
                positions[node], positions[source], positions[target]
            )
            penalties.append(distance.clamp(min=_MIN_DISTANCE).reciprocal().square())
    if penalties:
        node_edge = torch.stack(penalties).sum()

    edge_count = len(edges)
    distribution_scale = _scale_denominator(num_nodes * max(num_nodes - 1, 1) // 2)
    border_scale = _scale_denominator(num_nodes)
    edge_length_scale = _scale_denominator(edge_count)
    crossing_scale = _scale_denominator(edge_count * edge_count)
    node_edge_scale = _scale_denominator(num_nodes * edge_count)

    return (
        distribution / distribution_scale
        + _BORDER_WEIGHT * (border / border_scale)
        + _EDGE_LENGTH_WEIGHT * (edge_length / edge_length_scale)
        + _CROSSING_WEIGHT * (crossing_energy / crossing_scale)
        + _NODE_EDGE_WEIGHT * (node_edge / node_edge_scale)
    )


def layout_davidson_harel(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    rounds: int = 100,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Lay out a graph with Davidson-Harel simulated annealing.

    Reference
    ---------
    Davidson and Harel, "Drawing Graphs Nicely Using Simulated Annealing" (1996).

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Optional node sizes used only for drawing scale.
    rounds : int, default=100
        Number of annealing rounds.
    seed : int, default=42
        Random seed.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``. The edge-length energy term
        is scaled by these weights after collapsing mirrored edges into a
        unique undirected edge set.

    Returns
    -------
    torch.Tensor
        Node positions with shape ``[N, 2]``.

    Notes
    -----
    This routine follows igraph's more aggressive annealing schedule by using
    one move attempt per node per round and a ``0.75`` cooling factor. The
    annealing temperature is initialized from the starting energy so the
    acceptance schedule tracks the sum-based objective scale.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if rounds < 0:
        raise ValueError("rounds must be non-negative.")
    if edge_weights is not None and edge_weights.shape[0] != edge_index.shape[1]:
        raise ValueError(
            f"edge_weights length {edge_weights.shape[0]} does not match "
            f"edge count {edge_index.shape[1]}"
        )

    device = _layout_device(edge_index, node_sizes)
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32, device=device)

    extent = _layout_extent(num_nodes, node_sizes)
    positions = _initialize_positions(num_nodes, extent, device, seed)
    edges, unique_edge_weights = _unique_edges(edge_index, num_nodes, edge_weights=edge_weights)
    if edge_weights is None:
        current_energy = _energy(positions, edges, extent)
    else:
        current_energy = _energy(positions, edges, extent, unique_edge_weights)
    initial_temperature = max(0.1 * float(current_energy.item()), _MIN_DISTANCE)
    temperature = initial_temperature

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    moves_per_round = num_nodes

    for _ in range(rounds):
        for _ in range(moves_per_round):
            node = int(torch.randint(0, num_nodes, (1,), generator=generator).item())
            move_scale = 0.25 * extent * (temperature / max(initial_temperature, _MIN_DISTANCE))
            delta = ((torch.rand((2,), generator=generator) * 2.0) - 1.0).to(device) * (move_scale)
            candidate = positions.clone()
            candidate[node] = (candidate[node] + delta).clamp(min=-extent, max=extent)
            if edge_weights is None:
                candidate_energy = _energy(candidate, edges, extent)
            else:
                candidate_energy = _energy(candidate, edges, extent, unique_edge_weights)
            delta_energy = candidate_energy - current_energy
            if delta_energy <= 0:
                positions = candidate
                current_energy = candidate_energy
                continue

            acceptance = torch.exp(-delta_energy / max(temperature, _MIN_DISTANCE)).clamp(max=1.0)
            threshold = float(torch.rand((1,), generator=generator).item())
            if threshold < float(acceptance.item()):
                positions = candidate
                current_energy = candidate_energy

        temperature *= _COOLING_FACTOR

    centered = positions - positions.mean(dim=0, keepdim=True)
    span = centered.abs().max().clamp(min=1.0)
    return (centered * (extent / span)).to(dtype=torch.float32, device=device)
