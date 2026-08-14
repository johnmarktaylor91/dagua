"""Torch-only numerical and topology helpers shared by V4 facet families."""

from __future__ import annotations

import math
from collections import deque
from typing import List, Mapping, Optional, Set, Tuple

import torch

from dagua.eval.ruler_v4.scene import BoxGeometry, FacetResult, Scene, na_result, value_result


def smoothstep(value: torch.Tensor) -> torch.Tensor:
    """Evaluate the extended quintic smoothstep.

    Parameters
    ----------
    value : torch.Tensor
        Arbitrary float64 arguments.

    Returns
    -------
    torch.Tensor
        Values in ``[0, 1]`` with zero endpoint derivatives.
    """

    clipped = torch.clamp(value, 0.0, 1.0)
    return clipped**3 * (clipped * (6.0 * clipped - 15.0) + 10.0)


def soft_pos(value: float, constant: float = 0.5) -> float:
    """Evaluate the contracts' C1 one-sided positive map.

    Parameters
    ----------
    value : float
        Signed excess.
    constant : float
        Positive knee constant.

    Returns
    -------
    float
        Zero for nonpositive inputs and ``x^2/(x+c)`` otherwise.
    """

    if value <= 0.0:
        return 0.0
    return value * value / (value + constant)


def bounded(value: float) -> float:
    """Map a nonnegative unbounded burden into ``[0, 1)``.

    Parameters
    ----------
    value : float
        Nonnegative burden.

    Returns
    -------
    float
        Saturating cap-free defect.
    """

    nonnegative = max(0.0, value)
    return nonnegative / (1.0 + nonnegative)


def mean_result(
    facet_id: str, values: Mapping[str, float], raw: Optional[Mapping[str, object]] = None
) -> FacetResult:
    """Build a facet value as the fixed-ratio mean of scored rows.

    Parameters
    ----------
    facet_id : str
        Contract id used only for validation context.
    values : mapping[str, float]
        Scored sub-term values.
    raw : mapping[str, object] or None
        Published raw statistics.

    Returns
    -------
    FacetResult
        Bounded fixed-ratio result.
    """

    if not values:
        return na_result(f"{facet_id.lower()}_no_objects", raw)
    result = sum(values.values()) / len(values)
    return value_result(result, values, raw)


def adjacency(scene: Scene) -> List[Set[int]]:
    """Build simple undirected adjacency, ignoring self-loops.

    Parameters
    ----------
    scene : Scene
        Validated graph scene.

    Returns
    -------
    list[set[int]]
        Neighbor sets in canonical node order.
    """

    result = [set() for _ in range(scene.node_count)]
    for source, target in scene.graph.edges:
        if source == target:
            continue
        result[source].add(target)
        result[target].add(source)
    return result


def components(scene: Scene) -> List[List[int]]:
    """Return simple-support connected components.

    Parameters
    ----------
    scene : Scene
        Validated graph scene.

    Returns
    -------
    list[list[int]]
        Canonically ordered components.
    """

    graph = adjacency(scene)
    unseen = set(range(scene.node_count))
    result = []
    while unseen:
        root = min(unseen)
        queue = deque([root])
        unseen.remove(root)
        component = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in sorted(graph[node]):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        result.append(component)
    return result


def graph_distances(scene: Scene, weighted: bool = False) -> torch.Tensor:
    """Compute exhaustive same-support path distances.

    Parameters
    ----------
    scene : Scene
        Validated graph scene.
    weighted : bool
        Use declared positive edge weights as distance costs.

    Returns
    -------
    torch.Tensor
        Float64 distance matrix ``[N, N]`` with infinity across components.
    """

    count = scene.node_count
    result = torch.full((count, count), float("inf"), dtype=torch.float64)
    result.fill_diagonal_(0.0)
    weights = scene.graph.edge_weights if weighted else None
    for index, (source, target) in enumerate(scene.graph.edges):
        if source == target:
            continue
        cost = float(weights[index]) if weights is not None else 1.0
        result[source, target] = min(float(result[source, target]), cost)
        result[target, source] = min(float(result[target, source]), cost)
    # Dense Floyd-Warshall is exact and deterministic for the phase-1 fixtures. The
    # later approximation tier can replace this without changing facet contracts.
    for pivot in range(count):
        result = torch.minimum(result, result[:, pivot, None] + result[pivot, None, :])
    return result


def pair_values(scene: Scene, order: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract finite unordered graph-order and layout-distance pairs.

    Parameters
    ----------
    scene : Scene
        Validated graph scene.
    order : torch.Tensor
        Square graph-side order or distance matrix.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Graph-side and Euclidean pair vectors.
    """

    row, col = torch.triu_indices(scene.node_count, scene.node_count, offset=1)
    mask = torch.isfinite(order[row, col])
    layout = torch.linalg.vector_norm(scene.positions[row] - scene.positions[col], dim=1)
    return order[row[mask], col[mask]], layout[mask]


def pava(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Compute weighted nondecreasing isotonic regression.

    Parameters
    ----------
    values : torch.Tensor
        Block values in required order.
    weights : torch.Tensor
        Positive block weights.

    Returns
    -------
    torch.Tensor
        Fitted value per input block.
    """

    means: List[float] = []
    masses: List[float] = []
    starts: List[int] = []
    ends: List[int] = []
    for index, (value, weight) in enumerate(zip(values.tolist(), weights.tolist())):
        means.append(float(value))
        masses.append(float(weight))
        starts.append(index)
        ends.append(index + 1)
        while len(means) >= 2 and means[-2] > means[-1]:
            mass = masses[-2] + masses[-1]
            mean = (means[-2] * masses[-2] + means[-1] * masses[-1]) / mass
            means[-2:] = [mean]
            masses[-2:] = [mass]
            ends[-2:] = [ends[-1]]
            starts.pop()
    fitted = torch.empty_like(values)
    for mean, start, end in zip(means, starts, ends):
        fitted[start:end] = mean
    return fitted


def isotonic_stress(order: torch.Tensor, layout: torch.Tensor) -> float:
    """Compute Kruskal stress-1 with primary-tie PAVA.

    Parameters
    ----------
    order : torch.Tensor
        Graph-side order coordinates.
    layout : torch.Tensor
        Euclidean distances.

    Returns
    -------
    float
        Self-normalized stress in ``[0, 1]``.
    """

    if order.numel() == 0:
        return 0.0
    levels, inverse = torch.unique(order, sorted=True, return_inverse=True)
    sums = torch.zeros_like(levels, dtype=torch.float64)
    counts = torch.zeros_like(levels, dtype=torch.float64)
    sums.scatter_add_(0, inverse, layout)
    counts.scatter_add_(0, inverse, torch.ones_like(layout))
    fitted_levels = pava(sums / counts, counts)
    fitted = fitted_levels[inverse]
    denominator = float(torch.sum(layout * layout).item())
    if denominator == 0.0:
        return 1.0 if levels.numel() > 1 else 0.0
    return min(1.0, math.sqrt(float(torch.sum((layout - fitted) ** 2).item()) / denominator))


def midranks(values: torch.Tensor) -> torch.Tensor:
    """Assign deterministic average ranks with ties.

    Parameters
    ----------
    values : torch.Tensor
        One-dimensional values.

    Returns
    -------
    torch.Tensor
        Zero-based midranks.
    """

    unique, inverse, counts = torch.unique(
        values, sorted=True, return_inverse=True, return_counts=True
    )
    del unique
    cumulative = torch.cumsum(counts, dim=0)
    starts = cumulative - counts
    rank_by_level = (starts.to(torch.float64) + cumulative.to(torch.float64) - 1.0) / 2.0
    return rank_by_level[inverse]


def correlation_defect(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return one minus Pearson correlation, mapped to ``[0, 1]``.

    Parameters
    ----------
    left, right : torch.Tensor
        Equal-length observations.

    Returns
    -------
    float
        ``(1-r)/2`` or zero for identical constants.
    """

    if left.numel() < 2:
        return 0.0
    x = left - torch.mean(left)
    y = right - torch.mean(right)
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denominator) == 0.0:
        return 0.0 if torch.allclose(left, right) else 1.0
    correlation = float(torch.dot(x, y) / denominator)
    if correlation >= 1.0 - 1e-15:
        return 0.0
    if correlation <= -1.0 + 1e-15:
        return 1.0
    return min(1.0, max(0.0, (1.0 - correlation) / 2.0))


def aabb_pair(box_a: BoxGeometry, box_b: BoxGeometry) -> Tuple[float, float]:
    """Return signed clearance and overlap fraction for two axis-aligned boxes.

    Parameters
    ----------
    box_a, box_b : BoxGeometry
        Derived primitive boxes.

    Returns
    -------
    tuple[float, float]
        Euclidean signed clearance and intersection over smaller area.
    """

    delta = torch.abs(box_a.center - box_b.center) - (box_a.half_extents + box_b.half_extents)
    outside = torch.linalg.vector_norm(torch.clamp(delta, min=0.0))
    inside = min(max(float(delta[0]), float(delta[1])), 0.0)
    signed = float(outside) + inside
    overlap_extent = torch.clamp(-delta, min=0.0)
    intersection = float(torch.prod(overlap_extent).item())
    area_a = float(4.0 * torch.prod(box_a.half_extents).item())
    area_b = float(4.0 * torch.prod(box_b.half_extents).item())
    fraction = intersection / min(area_a, area_b) if min(area_a, area_b) > 0.0 else 0.0
    return signed, min(1.0, fraction)


def route_segments(scene: Scene) -> List[Tuple[int, int, torch.Tensor, torch.Tensor]]:
    """Flatten routes into indexed line segments.

    Parameters
    ----------
    scene : Scene
        Validated route scene.

    Returns
    -------
    list[tuple[int, int, torch.Tensor, torch.Tensor]]
        Route index, segment index, and endpoints.
    """

    result = []
    for route_index, route in enumerate(scene.routes):
        for segment_index in range(route.points.shape[0] - 1):
            result.append(
                (
                    route_index,
                    segment_index,
                    route.points[segment_index],
                    route.points[segment_index + 1],
                )
            )
    return result


def proper_intersection(
    start_a: torch.Tensor,
    end_a: torch.Tensor,
    start_b: torch.Tensor,
    end_b: torch.Tensor,
) -> bool:
    """Test exact proper intersection of two line-segment interiors.

    Parameters
    ----------
    start_a, end_a, start_b, end_b : torch.Tensor
        Segment endpoints with shape ``[2]``.

    Returns
    -------
    bool
        True only for a transversal interior crossing.
    """

    def cross(left: torch.Tensor, right: torch.Tensor) -> float:
        """Return the scalar 2D cross product."""

        return float(left[0] * right[1] - left[1] * right[0])

    direction_a = end_a - start_a
    direction_b = end_b - start_b
    denominator = cross(direction_a, direction_b)
    if denominator == 0.0:
        return False
    offset = start_b - start_a
    parameter_a = cross(offset, direction_b) / denominator
    parameter_b = cross(offset, direction_a) / denominator
    return 0.0 < parameter_a < 1.0 and 0.0 < parameter_b < 1.0


def route_lengths(scene: Scene) -> torch.Tensor:
    """Return total arc length per declared route.

    Parameters
    ----------
    scene : Scene
        Validated route scene.

    Returns
    -------
    torch.Tensor
        Float64 route lengths.
    """

    return torch.tensor(
        [
            float(torch.sum(torch.linalg.vector_norm(route.points[1:] - route.points[:-1], dim=1)))
            for route in scene.routes
        ],
        dtype=torch.float64,
    )


def node_degrees(scene: Scene) -> torch.Tensor:
    """Return simple-support node degrees.

    Parameters
    ----------
    scene : Scene
        Validated graph scene.

    Returns
    -------
    torch.Tensor
        Float64 degree vector.
    """

    return torch.tensor([len(neighbors) for neighbors in adjacency(scene)], dtype=torch.float64)


def declared_axis(scene: Scene) -> Optional[torch.Tensor]:
    """Return a canonical flow axis when direction or ranks declare one.

    Parameters
    ----------
    scene : Scene
        Validated graph scene.

    Returns
    -------
    torch.Tensor or None
        Unit axis selected without drawing-side optimization.
    """

    if scene.graph.flow_axis is not None:
        return torch.tensor(scene.graph.flow_axis, dtype=torch.float64)
    if scene.graph.ranks is not None:
        return torch.tensor([0.0, 1.0], dtype=torch.float64)
    return None
