"""Torch-only numerical and topology helpers shared by V4 facet families."""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import torch

from dagua.eval.ruler_v4.scene import (
    BoxGeometry,
    FacetResult,
    Route,
    Scene,
    na_result,
    value_result,
)

_BLEND_WEIGHTS = (0.65, 0.25, 0.10)
_TRIM_FRACTION = 0.05
_CVAR_TAIL_FRACTION = 0.10
_SMOOTH_MAX_TEMPERATURE = 0.05

ALPHA_GRID: Tuple[Tuple[str, float, float], ...] = (
    ("AC15_AH00", 0.15, 0.00),
    ("AC15_AH20", 0.15, 0.20),
    ("AC15_AH50", 0.15, 0.50),
    ("AC15_AH100", 0.15, 1.00),
    ("AC30_AH00", 0.30, 0.00),
    ("AC30_AH20", 0.30, 0.20),
    ("AC30_AH50", 0.30, 0.50),
    ("AC30_AH100", 0.30, 1.00),
    ("AC50_AH00", 0.50, 0.00),
    ("AC50_AH20", 0.50, 0.20),
    ("AC50_AH50", 0.50, 0.50),
    ("AC50_AH100", 0.50, 1.00),
)

_FACET_ROW_WEIGHTS: Dict[str, Dict[str, float]] = {
    "U01b": {"U01b.local": 0.5, "U01b.long": 0.5},
    "U03": {"U03.r_1": 0.5, "U03.r_2": 0.3, "U03.r_4": 0.2},
    "U04a": {"U04a.2u": 0.5, "U04a.8u": 0.5},
    "U04b": {"U04b.part_1": 0.5, "U04b.part_2": 0.5},
    "U07": {"U7.base": 1.0, "U7.tail": 1.0},
    "U11": {
        "U11.i": 0.30,
        "U11.ii": 0.15,
        "U11.iii": 0.20,
        "U11.iv": 0.15,
        "U11.v": 0.20,
    },
    "U13": {"U13.i": 0.7, "U13.ii": 0.3},
    "U15": {"U15.i": 0.5, "U15.ii": 0.5},
    "U16": {"U16.i": 0.6, "U16.ii": 0.4},
    "U18": {"U18.ll": 0.4, "U18.ln": 0.4, "U18.le": 0.2},
    "U20a": {"U20a.i": 0.45, "U20a.ii": 0.25, "U20a.iii": 0.30},
    "U26": {"U26.i": 0.50, "U26.ii": 0.25, "U26.iii": 0.25},
    "U27": {"U27.i": 0.50, "U27.ii": 0.20, "U27.iii": 0.30},
    "U28": {"U28.i": 0.45, "U28.ii": 0.35, "U28.iii": 0.20},
    "U30": {"U30.i": 0.35, "U30.ii": 0.25, "U30.iii": 0.40},
    "U32": {"U32.L_iso": 0.40, "U32.L_crisp": 0.30, "U32.L_overlap": 0.30},
    "U33.layered": {
        "U33.layered.1": 5.0 / 16.0,
        "U33.layered.2": 1.0 / 4.0,
        "U33.layered.3": 5.0 / 16.0,
        "U33.layered.4": 1.0 / 8.0,
    },
    "U33.radial": {"U33.radial.1": 0.55, "U33.radial.2": 0.45},
    "U34": {"U34.L_back": 0.40, "U34.L_mono": 0.35, "U34.L_cont": 0.25},
    "U37": {"U37.ell_e": 0.75, "U37.ell_ord": 0.25},
    "U38": {"U38.L_clear": 0.55, "U38.L_pack": 0.30, "U38.L_prop": 0.15},
    "U39": {"U39.1": 0.30, "U39.2": 0.25, "U39.3": 0.20, "U39.4": 0.25},
    "U40": {"U40.1": 0.55, "U40.2": 0.25, "U40.3": 0.20},
    "U41": {"U41.L_conv": 0.60, "U41.L_area": 0.40},
    "U42": {"U42.i": 0.40, "U42.ii": 0.35, "U42.iv": 0.25},
}

_NOISY_OR_FACETS = frozenset({"U18", "U20a", "U26", "U27", "U28", "U30", "U42"})


def selected_alpha_grid_offset(alpha_grid_index: Optional[int]) -> Optional[int]:
    """Validate and convert one shared U17 grid selection.

    Parameters
    ----------
    alpha_grid_index : int or None
        Manifest row index in ``[1, 12]``. ``None`` denotes the contract's
        preregistered unselected state and requires publishing an envelope.

    Returns
    -------
    int or None
        Zero-based row offset, or ``None`` for the unselected state.

    Raises
    ------
    ValueError
        If the supplied index is not a row of the frozen grid.
    """

    if alpha_grid_index is None:
        return None
    if isinstance(alpha_grid_index, bool) or not isinstance(alpha_grid_index, int):
        raise ValueError("alpha_grid_index must be an integer in [1, 12] or None")
    if not 1 <= alpha_grid_index <= len(ALPHA_GRID):
        raise ValueError("alpha_grid_index must be an integer in [1, 12] or None")
    return alpha_grid_index - 1


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
    # The quintic can overshoot 1.0 by one ULP just below the upper knot; the
    # promised [0, 1] range is load-bearing (defect terms of the form
    # 1 - smoothstep(...) must never go negative under the blend domain guard).
    return torch.clamp(clipped**3 * (clipped * (6.0 * clipped - 15.0) + 10.0), 0.0, 1.0)


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


_UNIT_DUST = 1e-12


def snap_unit(value: float) -> float:
    """Snap float dust off an analytically-``[0, 1]`` quantity.

    Producers whose closed form is bounded to the unit interval can still
    leave it by accumulated rounding: signed log sums (Jensen-Shannon
    divergences), renormalized convex combinations, and equal/weighted
    means of in-range populations. That dust is clamped here, at the
    producer. Excess beyond ``_UNIT_DUST`` is a real range violation and
    is returned unchanged so the downstream ``[0, 1]`` guards
    (``value_result``, the blend domain checks) still raise on it.

    Parameters
    ----------
    value : float
        Producer output whose analytic range is ``[0, 1]``.

    Returns
    -------
    float
        The value with sub-dust excess clamped into ``[0, 1]``.
    """

    if -_UNIT_DUST <= value < 0.0:
        return 0.0
    if 1.0 < value <= 1.0 + _UNIT_DUST:
        return 1.0
    return value


def mean_result(
    facet_id: str,
    values: Mapping[str, float],
    raw: Optional[Mapping[str, object]] = None,
    *,
    renormalize_missing: bool = True,
) -> FacetResult:
    """Build a facet value with its frozen row-composition operator.

    Parameters
    ----------
    facet_id : str
        Contract id used only for validation context.
    values : mapping[str, float]
        Scored sub-term values.
    raw : mapping[str, object] or None
        Published raw statistics.
    renormalize_missing : bool
        Whether to redistribute absent row mass over the applicable rows. Set
        false only where a facet contract explicitly freezes absent mass.

    Returns
    -------
    FacetResult
        Bounded fixed-ratio result.
    """

    if not values:
        return na_result(f"{facet_id.lower()}_no_objects", raw)
    weights = _FACET_ROW_WEIGHTS.get(facet_id)
    if weights is None and facet_id == "U33":
        has_radial_rows = any(key.startswith("U33.radial") for key in values)
        mode = "U33.radial" if has_radial_rows else "U33.layered"
        weights = _FACET_ROW_WEIGHTS[mode]
    if weights is None:
        weights = {key: 1.0 for key in values}
    applicable = {key: weight for key, weight in weights.items() if key in values}
    mass = sum(applicable.values())
    if mass <= 0.0:
        return na_result(f"{facet_id.lower()}_no_objects", raw)
    effective = (
        {key: weight / mass for key, weight in applicable.items()}
        if renormalize_missing
        else applicable
    )
    if facet_id in _NOISY_OR_FACETS:
        survival = 1.0
        for key, weight in effective.items():
            survival *= (1.0 - float(values[key])) ** weight
        result = 1.0 - survival
    else:
        # The renormalized row weights are a convex combination only up to
        # rounding; on saturated rows the sum can carry one ULP of dust.
        result = snap_unit(sum(effective[key] * float(values[key]) for key in effective))
    return value_result(result, values, raw)


def resolved_ranks(scene: Scene) -> Optional[Tuple[int, ...]]:
    """Return declared ranks or deterministic longest-path DAG ranks.

    Parameters
    ----------
    scene : Scene
        Canonical graph scene.

    Returns
    -------
    tuple[int, ...] or None
        Declared ranks when present, derived ranks for a directed acyclic graph,
        or ``None`` for an undirected or cyclic graph without declarations.
    """

    if scene.graph.ranks is not None:
        return scene.graph.ranks
    if not scene.graph.directed:
        return None
    incoming = [0] * scene.node_count
    outgoing: List[List[int]] = [[] for _ in range(scene.node_count)]
    for source, target in scene.graph.edges:
        incoming[target] += 1
        outgoing[source].append(target)
    queue = deque(index for index, count in enumerate(incoming) if count == 0)
    ranks = [0] * scene.node_count
    visited = 0
    while queue:
        source = queue.popleft()
        visited += 1
        for target in sorted(outgoing[source]):
            ranks[target] = max(ranks[target], ranks[source] + 1)
            incoming[target] -= 1
            if incoming[target] == 0:
                queue.append(target)
    return tuple(ranks) if visited == scene.node_count else None


def _weighted_interval_mean(
    values: Sequence[float], weights: Sequence[float], lower: float, upper: float
) -> float:
    """Average a weighted empirical distribution over a mass interval.

    Parameters
    ----------
    values : sequence[float]
        Defects sorted in nondecreasing order.
    weights : sequence[float]
        Corresponding positive population weights normalized to total mass one.
    lower, upper : float
        Half-open cumulative-mass interval in ``[0, 1]``.

    Returns
    -------
    float
        Exact fractional-boundary weighted mean over the requested interval.
    """

    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("weighted interval must satisfy 0 <= lower < upper <= 1")
    total = 0.0
    cursor = 0.0
    for value, weight in zip(values, weights):
        next_cursor = cursor + weight
        overlap = max(0.0, min(next_cursor, upper) - max(cursor, lower))
        total += overlap * value
        cursor = next_cursor
        if cursor >= upper:
            break
    return total / (upper - lower)


def global_blend(
    defects: Iterable[float], population_weights: Optional[Iterable[float]] = None
) -> float:
    """Evaluate the ecosystem-normative U11 section 17 mean-plus-tail blend.

    Parameters
    ----------
    defects : iterable[float]
        Per-object defects in ``[0, 1]`` after any contract-defined fade transform.
    population_weights : iterable[float] or None
        Positive input-owned masses. Equal mass is used when omitted.

    Returns
    -------
    float
        ``0.65*TrimmedMean_5% + 0.25*CVaR_0.10 + 0.10*SmoothMax_0.05``.

    Raises
    ------
    ValueError
        If the population is empty, a defect is out of range, or weights are invalid.
    """

    return blend_with_weights(defects, population_weights, _BLEND_WEIGHTS)


def blend_with_weights(
    defects: Iterable[float],
    population_weights: Optional[Iterable[float]],
    blend_weights: Tuple[float, float, float],
    *,
    robust_mean: Optional[float] = None,
) -> float:
    """Evaluate the global blend with one contract-declared component vector.

    Parameters
    ----------
    defects : iterable[float]
        Per-object defects in ``[0, 1]``.
    population_weights : iterable[float] or None
        Positive input-owned object masses, or equal mass.
    blend_weights : tuple[float, float, float]
        Nonnegative ``(trimmed_mean, CVaR, smooth_max)`` coefficients summing to one.
    robust_mean : float or None
        Contract-declared replacement for the usual trimmed-mean component.

    Returns
    -------
    float
        Bounded weighted component blend.

    Raises
    ------
    ValueError
        If the population or any coefficient is malformed.
    """

    invalid_weights = any(weight < 0.0 or not math.isfinite(weight) for weight in blend_weights)
    weights_do_not_sum_to_one = not math.isclose(
        sum(blend_weights), 1.0, rel_tol=0.0, abs_tol=1e-12
    )
    if invalid_weights or weights_do_not_sum_to_one:
        raise ValueError("blend component weights must be nonnegative and sum to one")
    values = [float(value) for value in defects]
    if not values:
        raise ValueError("global blend requires a nonempty object population")
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
        raise ValueError("global blend defects must be finite and lie in [0, 1]")
    if population_weights is None:
        masses = [1.0] * len(values)
    else:
        masses = [float(weight) for weight in population_weights]
    if len(masses) != len(values):
        raise ValueError("global blend weights must match the defect population")
    if any(not math.isfinite(weight) or weight <= 0.0 for weight in masses):
        raise ValueError("global blend weights must be finite and positive")
    ordered = sorted(zip(values, masses), key=lambda item: item[0])
    total_mass = sum(weight for _, weight in ordered)
    sorted_values = [value for value, _ in ordered]
    normalized = [weight / total_mass for _, weight in ordered]
    if robust_mean is None:
        # The interval mean of an in-[0, 1] population can escape the interval
        # only by float summation dust (six exact-1.0 defects trim to
        # 1.0000000000000002); clamp rather than reject the port's own output.
        trimmed = min(
            1.0,
            max(
                0.0,
                _weighted_interval_mean(
                    sorted_values, normalized, _TRIM_FRACTION, 1.0 - _TRIM_FRACTION
                ),
            ),
        )
    else:
        trimmed = float(robust_mean)
        if not math.isfinite(trimmed) or not 0.0 <= trimmed <= 1.0:
            raise ValueError("robust-mean override must be finite and lie in [0, 1]")
    cvar = _weighted_interval_mean(sorted_values, normalized, 1.0 - _CVAR_TAIL_FRACTION, 1.0)
    maximum = sorted_values[-1]
    exponential_mean = sum(
        weight * math.exp((value - maximum) / _SMOOTH_MAX_TEMPERATURE)
        for value, weight in zip(sorted_values, normalized)
    )
    smooth_max = maximum + _SMOOTH_MAX_TEMPERATURE * math.log(exponential_mean)
    mean_weight, cvar_weight, maximum_weight = blend_weights
    blend = mean_weight * trimmed + cvar_weight * cvar + maximum_weight * smooth_max
    return min(1.0, max(0.0, blend))


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
    fitted = primary_isotonic_fit(order, layout)
    levels = torch.unique(order, sorted=True)
    denominator = float(torch.sum(layout * layout).item())
    if denominator == 0.0:
        return 1.0 if levels.numel() > 1 else 0.0
    return min(1.0, math.sqrt(float(torch.sum((layout - fitted) ** 2).item()) / denominator))


def primary_isotonic_fit(order: torch.Tensor, layout: torch.Tensor) -> torch.Tensor:
    """Fit nondecreasing disparities with Kruskal primary tie handling.

    Parameters
    ----------
    order : torch.Tensor
        Graph-side coordinates with shape ``[P]``.
    layout : torch.Tensor
        Layout distances with shape ``[P]``.

    Returns
    -------
    torch.Tensor
        Fitted disparities in the original pair order.
    """

    if order.shape != layout.shape or order.ndim != 1:
        raise ValueError("primary isotonic inputs must be equal one-dimensional arrays")
    if order.numel() == 0:
        return torch.empty_like(layout)
    ordered_indices: List[int] = []
    for level in torch.unique(order, sorted=True):
        indices = torch.nonzero(order == level, as_tuple=False).flatten()
        local = indices[torch.argsort(layout[indices], stable=True)]
        ordered_indices.extend(int(index) for index in local)
    permutation = torch.tensor(ordered_indices, dtype=torch.long, device=layout.device)
    ordered_layout = layout[permutation]
    fitted_ordered = pava(ordered_layout, torch.ones_like(ordered_layout))
    fitted = torch.empty_like(layout)
    fitted[permutation] = fitted_ordered
    return fitted


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
    for route in resolved_routes(scene):
        for segment_index in range(route.points.shape[0] - 1):
            result.append(
                (
                    route.edge_index,
                    segment_index,
                    route.points[segment_index],
                    route.points[segment_index + 1],
                )
            )
    return result


def resolved_routes(scene: Scene) -> Tuple[Route, ...]:
    """Return one render-truth route per declared edge with chord fallbacks.

    Parameters
    ----------
    scene : Scene
        Validated graph scene.

    Returns
    -------
    tuple[Route, ...]
        Routes in declared edge order. An absent record is the straight segment
        between endpoint positions, as required by the K17 identity.
    """

    explicit = {route.edge_index: route for route in scene.routes}
    routes: List[Route] = []
    for edge_index, (source, target) in enumerate(scene.graph.edges):
        route = explicit.get(edge_index)
        if route is None:
            route = Route(
                edge_index=edge_index,
                points=torch.stack((scene.positions[source], scene.positions[target])),
            )
        routes.append(route)
    return tuple(routes)


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
            for route in resolved_routes(scene)
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
    return None
