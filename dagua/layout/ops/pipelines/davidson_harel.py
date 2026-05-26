"""Davidson-Harel simulated-annealing layout pipeline."""

from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np
import torch

from dagua.layout.ops.base import Pipeline, Repeat  # noqa: E402
from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig  # noqa: E402
from dagua.layout.ops.davidson_harel import (
    DHAnnealingRound,
    DHAnnealingRoundConfig,
    DHCool,
    FinalizeDHPositions,
    InitializeDHPositions,
    PrepareDHState,
)
from dagua.layout.ops.pipelines import resolve_fidelity_dtype
from dagua.layout.ops.state import (  # noqa: E402
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)

_MOVE_TRIES = 30
_FINE_TUNING_FACTOR = 0.01


def _uint32_bounded(rng: random.Random, range_value: int) -> int:
    """Generate igraph's bounded 32-bit integer.

    Parameters
    ----------
    rng : random.Random
        Python RNG backing the python-igraph external RNG bridge.
    range_value : int
        Exclusive upper bound for the generated integer.

    Returns
    -------
    int
        Uniform integer in ``[0, range_value)``.
    """
    threshold = ((1 << 32) - range_value) % range_value
    while True:
        value = rng.getrandbits(32)
        product = value * range_value
        low_word = product & 0xFFFFFFFF
        if low_word >= threshold:
            return product >> 32


def _igraph_integer(rng: random.Random, low: int, high: int) -> int:
    """Return an igraph-style random integer in ``[low, high]``.

    Parameters
    ----------
    rng : random.Random
        Python RNG backing the python-igraph external RNG bridge.
    low : int
        Inclusive lower bound.
    high : int
        Inclusive upper bound.

    Returns
    -------
    int
        Random integer generated with igraph's C bounded-integer path.
    """
    if high <= low:
        return low
    return low + _uint32_bounded(rng, high - low + 1)


def _shuffle_igraph(values: list[int], rng: random.Random) -> None:
    """Shuffle values in place with igraph's Fisher-Yates loop.

    Parameters
    ----------
    values : list[int]
        Mutable vector to shuffle.
    rng : random.Random
        Python RNG backing the python-igraph external RNG bridge.

    Returns
    -------
    None
        The input list is modified in place.
    """
    size = len(values)
    while size > 1:
        swap_index = _igraph_integer(rng, 0, size - 1)
        size -= 1
        values[size], values[swap_index] = values[swap_index], values[size]


def _segments_intersect_igraph(
    p0_x: float,
    p0_y: float,
    p1_x: float,
    p1_y: float,
    p2_x: float,
    p2_y: float,
    p3_x: float,
    p3_y: float,
) -> bool:
    """Return igraph's C segment-intersection predicate.

    Parameters
    ----------
    p0_x, p0_y, p1_x, p1_y : float
        First segment endpoints.
    p2_x, p2_y, p3_x, p3_y : float
        Second segment endpoints.

    Returns
    -------
    bool
        ``True`` when the two closed segments intersect.
    """
    s1_x = p1_x - p0_x
    s1_y = p1_y - p0_y
    s2_x = p3_x - p2_x
    s2_y = p3_y - p2_y
    s1 = (-s1_y * (p0_x - p2_x)) + (s1_x * (p0_y - p2_y))
    s2 = (-s2_x * s1_y) + (s1_x * s2_y)
    if s2 == 0.0:
        return False
    t1 = (s2_x * (p0_y - p2_y)) - (s2_y * (p0_x - p2_x))
    t2 = (-s2_x * s1_y) + (s1_x * s2_y)
    s = s1 / s2
    t = t1 / t2
    return s >= 0.0 and s <= 1.0 and t >= 0.0 and t <= 1.0


def _point_segment_dist2_igraph(
    v_x: float,
    v_y: float,
    u1_x: float,
    u1_y: float,
    u2_x: float,
    u2_y: float,
) -> float:
    """Return igraph's squared point-to-segment distance.

    Parameters
    ----------
    v_x, v_y : float
        Query point.
    u1_x, u1_y : float
        First segment endpoint.
    u2_x, u2_y : float
        Second segment endpoint.

    Returns
    -------
    float
        Squared distance from the point to the segment.
    """
    dx = u2_x - u1_x
    dy = u2_y - u1_y
    length2 = (dx * dx) + (dy * dy)
    if length2 == 0.0:
        return ((v_x - u1_x) * (v_x - u1_x)) + ((v_y - u1_y) * (v_y - u1_y))
    t = (((v_x - u1_x) * dx) + ((v_y - u1_y) * dy)) / length2
    if t < 0.0:
        return ((v_x - u1_x) * (v_x - u1_x)) + ((v_y - u1_y) * (v_y - u1_y))
    if t > 1.0:
        return ((v_x - u2_x) * (v_x - u2_x)) + ((v_y - u2_y) * (v_y - u2_y))
    p_x = u1_x + (t * dx)
    p_y = u1_y + (t * dy)
    return ((v_x - p_x) * (v_x - p_x)) + ((v_y - p_y) * (v_y - p_y))


def _reciprocal_or_inf(value: float) -> float:
    """Return C-like floating reciprocal semantics.

    Parameters
    ----------
    value : float
        Denominator value.

    Returns
    -------
    float
        Reciprocal value, or positive infinity for zero.
    """
    if value == 0.0:
        return math.inf
    return 1.0 / value


def _edge_list(edge_index: torch.Tensor, num_nodes: int) -> list[tuple[int, int]]:
    """Return edge endpoints in igraph edge-id order.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge endpoint tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    list[tuple[int, int]]
        Directed edge endpoints in input order.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")
    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    edges: list[tuple[int, int]] = []
    for edge_id in range(edge_index_cpu.shape[1]):
        source = int(edge_index_cpu[0, edge_id].item())
        target = int(edge_index_cpu[1, edge_id].item())
        if source < 0 or source >= num_nodes or target < 0 or target >= num_nodes:
            raise ValueError("edge_index contains a node index outside [0, num_nodes).")
        edges.append((source, target))
    return edges


def _neighbors_and_incidents(
    edges: list[tuple[int, int]],
    num_nodes: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """Build all-mode neighbor and incident edge lists without loops.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Directed edge endpoints in igraph edge-id order.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    tuple[list[list[int]], list[list[int]]]
        Per-node neighbor ids and incident edge ids.
    """
    neighbors: list[list[int]] = [[] for _ in range(num_nodes)]
    keyed_incidents: list[list[tuple[int, int]]] = [[] for _ in range(num_nodes)]
    for edge_id, (source, target) in enumerate(edges):
        if source == target:
            continue
        neighbors[source].append(target)
        neighbors[target].append(source)
        keyed_incidents[source].append((target, edge_id))
        keyed_incidents[target].append((source, edge_id))
    for node_neighbors in neighbors:
        node_neighbors.sort()
    incidents = [[edge_id for _, edge_id in sorted(items)] for items in keyed_incidents]
    return neighbors, incidents


def _resolve_fineiter(num_nodes: int, fineiter: Optional[int]) -> int:
    """Resolve python-igraph's negative fine-iteration default.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    fineiter : int, optional
        Explicit fine-tuning iteration count.

    Returns
    -------
    int
        Fine-tuning iteration count.
    """
    if fineiter is not None:
        return fineiter
    return min(int(math.log2(num_nodes)) if num_nodes > 1 else 0, 10)


def _resolve_weights(num_nodes: int, edge_count: int) -> tuple[float, float, float, float, float]:
    """Resolve python-igraph's default Davidson-Harel weights.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    edge_count : int
        Number of directed graph edges.

    Returns
    -------
    tuple[float, float, float, float, float]
        Node-distance, border, edge-length, crossing, and node-edge weights.
    """
    density = edge_count / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0.0
    return (1.0, 0.0, density / 10.0, 1.0 - math.sqrt(density), 0.2 - (0.2 * density))


def _align_positions_igraph(
    positions: list[list[float]],
    edges: list[tuple[int, int]],
) -> list[list[float]]:
    """Apply igraph_layout_align for 2D Davidson-Harel output.

    Parameters
    ----------
    positions : list[list[float]]
        Mutable coordinate rows with shape ``[N, 2]``.
    edges : list[tuple[int, int]]
        Directed edge endpoints in igraph edge-id order.

    Returns
    -------
    list[list[float]]
        Centered and axis-aligned coordinates.
    """
    if not positions:
        return positions

    layout = np.asarray(positions, dtype=np.float64)
    layout = layout - layout.mean(axis=0, keepdims=True)
    matrix = np.zeros((2, 2), dtype=np.float64)
    correction = np.zeros((2, 2), dtype=np.float64)
    correction_saved = False
    norm2_sum = 0.0
    norm2_sum_correction = 0.0

    for source, target in edges:
        if source == target:
            continue
        edge_vec = layout[source] - layout[target]
        term = np.outer(edge_vec, edge_vec)
        matrix += term
        norm2_sum += float(term[0, 0] + term[1, 1])
        if not correction_saved and norm2_sum > 0.0:
            correction_saved = True
            norm2_sum_correction = norm2_sum
            correction = matrix.copy()

    if norm2_sum == 0.0:
        for node in range(layout.shape[0]):
            term = np.outer(layout[node], layout[node])
            matrix += term
            norm2_sum += float(term[0, 0] + term[1, 1])
            if not correction_saved and norm2_sum > 0.0:
                correction_saved = True
                norm2_sum_correction = norm2_sum
                correction = matrix.copy()

    if norm2_sum == 0.0:
        return layout.tolist()

    retried = False
    while True:
        tensor = matrix.copy()
        tensor *= 1.0 / norm2_sum
        tensor[0, 0] -= 0.5
        tensor[1, 1] -= 0.5
        from scipy import linalg

        eigenvalues, rotation = linalg.eigh(tensor, driver="evr")
        matrix_norm = float(np.max(np.abs(eigenvalues)))
        if matrix_norm > 1.0e-3 or retried:
            break
        matrix -= correction
        norm2_sum -= norm2_sum_correction
        retried = True

    temp_layout = layout @ rotation
    extents = np.max(temp_layout, axis=0) - np.min(temp_layout, axis=0)
    permutation = np.argsort(-extents, kind="stable")
    return temp_layout[:, permutation].tolist()


def _pure_igraph_davidson_harel_positions(
    edge_index: torch.Tensor,
    num_nodes: int,
    seed: int,
    rounds: int,
    fineiter: Optional[int],
    device: torch.device,
    fidelity_dtype: torch.dtype,
) -> torch.Tensor:
    """Run the pure-Python igraph Davidson-Harel port.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    seed : int
        Fidelity seed used for both the igraph RNG and initial coordinate
        matrix.
    rounds : int
        Number of annealing iterations passed as igraph ``maxiter``.
    fineiter : int, optional
        Number of fine-tuning iterations passed to igraph. ``None`` preserves
        python-igraph's graph-size-dependent default.
    device : torch.device
        Device for the returned tensor.
    fidelity_dtype : torch.dtype
        Floating dtype for the returned tensor.

    Returns
    -------
    torch.Tensor
        igraph-compatible coordinates with shape ``[N, 2]``, scaled exactly
        like the benchmark igraph adapter.
    """
    resolved_fineiter = _resolve_fineiter(num_nodes, fineiter)
    if resolved_fineiter < 0:
        raise ValueError("fineiter must be non-negative.")
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=fidelity_dtype, device=device)

    width = math.sqrt(num_nodes) * 10.0
    half_width = width / 2.0
    move_radius = half_width
    initial = np.random.RandomState(seed).uniform(-1.0, 1.0, size=(num_nodes, 2))
    positions = [[float(initial[node, 0]), float(initial[node, 1])] for node in range(num_nodes)]
    min_x = min(position[0] for position in positions)
    max_x = max(position[0] for position in positions)
    min_y = min(position[1] for position in positions)
    max_y = max(position[1] for position in positions)
    edges = _edge_list(edge_index, num_nodes)
    neighbors, incident_edges = _neighbors_and_incidents(edges, num_nodes)
    w_node_dist, w_border, w_edge_lengths, w_crossings, w_node_edge = _resolve_weights(
        num_nodes, len(edges)
    )
    del w_border

    move_x = [math.cos((2.0 * math.pi / _MOVE_TRIES) * index) for index in range(_MOVE_TRIES)]
    move_y = [math.sin((2.0 * math.pi / _MOVE_TRIES) * index) for index in range(_MOVE_TRIES)]
    permutation = list(range(num_nodes))
    try_order = list(range(_MOVE_TRIES))
    rng = random.Random(seed)

    for round_id in range(rounds + resolved_fineiter):
        _shuffle_igraph(permutation, rng)
        fine_tuning = round_id >= rounds
        if fine_tuning:
            fine_x = _FINE_TUNING_FACTOR * (max_x - min_x)
            fine_y = _FINE_TUNING_FACTOR * (max_y - min_y)
            move_radius = fine_x if fine_x < fine_y else fine_y

        for node in permutation:
            _shuffle_igraph(try_order, rng)
            for try_id in try_order:
                old_x = positions[node][0]
                old_y = positions[node][1]
                new_x = old_x + (move_radius * move_x[try_id])
                new_y = old_y + (move_radius * move_y[try_id])
                if new_x < -half_width:
                    new_x = -half_width - 1.0e-6
                if new_x > half_width:
                    new_x = half_width - 1.0e-6
                if new_y < -half_width:
                    new_y = -half_width - 1.0e-6
                if new_y > half_width:
                    new_y = half_width - 1.0e-6

                diff_energy = 0.0
                for other in range(num_nodes):
                    if other == node:
                        continue
                    old_dx = old_x - positions[other][0]
                    old_dy = old_y - positions[other][1]
                    new_dx = new_x - positions[other][0]
                    new_dy = new_y - positions[other][1]
                    old_dist2 = (old_dx * old_dx) + (old_dy * old_dy)
                    new_dist2 = (new_dx * new_dx) + (new_dy * new_dy)
                    diff_energy += w_node_dist * (
                        _reciprocal_or_inf(new_dist2) - _reciprocal_or_inf(old_dist2)
                    )

                if w_edge_lengths != 0.0:
                    for other in neighbors[node]:
                        old_dx = old_x - positions[other][0]
                        old_dy = old_y - positions[other][1]
                        new_dx = new_x - positions[other][0]
                        new_dy = new_y - positions[other][1]
                        old_dist2 = (old_dx * old_dx) + (old_dy * old_dy)
                        new_dist2 = (new_dx * new_dx) + (new_dy * new_dy)
                        diff_energy += w_edge_lengths * (new_dist2 - old_dist2)

                if w_crossings != 0.0:
                    crossing_delta = 0
                    for other in neighbors[node]:
                        other_x = positions[other][0]
                        other_y = positions[other][1]
                        for source, target in edges:
                            if (
                                source == node
                                or target == node
                                or source == other
                                or target == other
                            ):
                                continue
                            source_x = positions[source][0]
                            source_y = positions[source][1]
                            target_x = positions[target][0]
                            target_y = positions[target][1]
                            crossing_delta -= _segments_intersect_igraph(
                                old_x,
                                old_y,
                                other_x,
                                other_y,
                                source_x,
                                source_y,
                                target_x,
                                target_y,
                            )
                            crossing_delta += _segments_intersect_igraph(
                                new_x,
                                new_y,
                                other_x,
                                other_y,
                                source_x,
                                source_y,
                                target_x,
                                target_y,
                            )
                    diff_energy += w_crossings * crossing_delta

                if w_node_edge != 0.0 and fine_tuning:
                    for source, target in edges:
                        if source == node or target == node:
                            continue
                        source_x = positions[source][0]
                        source_y = positions[source][1]
                        target_x = positions[target][0]
                        target_y = positions[target][1]
                        old_dist = _point_segment_dist2_igraph(
                            old_x, old_y, source_x, source_y, target_x, target_y
                        )
                        new_dist = _point_segment_dist2_igraph(
                            new_x, new_y, source_x, source_y, target_x, target_y
                        )
                        diff_energy += w_node_edge * (
                            _reciprocal_or_inf(new_dist) - _reciprocal_or_inf(old_dist)
                        )

                    for edge_id in incident_edges[node]:
                        source, target = edges[edge_id]
                        other = target if source == node else source
                        other_x = positions[other][0]
                        other_y = positions[other][1]
                        for third in range(num_nodes):
                            if third == node or third == other:
                                continue
                            third_x = positions[third][0]
                            third_y = positions[third][1]
                            old_dist = _point_segment_dist2_igraph(
                                third_x, third_y, old_x, old_y, other_x, other_y
                            )
                            new_dist = _point_segment_dist2_igraph(
                                third_x, third_y, new_x, new_y, other_x, other_y
                            )
                            diff_energy += w_node_edge * (
                                _reciprocal_or_inf(new_dist) - _reciprocal_or_inf(old_dist)
                            )

                if diff_energy < 0.0 or (
                    not fine_tuning and rng.random() < math.exp(-diff_energy / move_radius)
                ):
                    positions[node][0] = new_x
                    positions[node][1] = new_y
                    if new_x < min_x:
                        min_x = new_x
                    elif new_x > max_x:
                        max_x = new_x
                    if new_y < min_y:
                        min_y = new_y
                    elif new_y > max_y:
                        max_y = new_y

        move_radius *= 0.75

    aligned = _align_positions_igraph(positions, edges)
    return torch.tensor(
        [[x * 50.0, y * 50.0] for x, y in aligned],
        dtype=fidelity_dtype,
        device=device,
    )


def build_davidson_harel_pipeline(
    rounds: int = 100,
    fineiter: int = 10,
    skip_finalization: bool = True,
) -> Pipeline:
    """Build a Davidson-Harel simulated-annealing pipeline.

    Reference fidelity
    ------------------
    Targets: igraph 1.0.0 Davidson-Harel / Davidson and Harel (1996), "Drawing
        Graphs Nicely Using Simulated Annealing".
    Fidelity mode: no explicit flag; ``skip_finalization=True`` preserves the
        igraph-style final coordinate contract used by benchmark variants.
    Verified at: final 100-seed report, partial match; median RMSD 0.168 for
        100 rounds and 0.194 for 50 rounds. The 200-round variant had
        insufficient data.
    Known divergences:
        - Several final-report failures are from skipped or errored
          reimplementation rows on bounded graphs.
        - Seed forwarding is handled by the shared OGDF/igraph adapter path,
          not this builder.

    Parameters
    ----------
    rounds : int, default=100
        Number of annealing rounds to execute.
    fineiter : int, default=10
        Number of igraph-style fine-tuning rounds to execute after annealing.
    skip_finalization : bool, default=True
        Whether to skip Dagua's legacy final centering/scaling pass. igraph
        fidelity mode leaves the last accepted coordinates unchanged.

    Returns
    -------
    Pipeline
        Pipeline implementing the Davidson-Harel algorithm. The pipeline
        produces final node coordinates by initializing positions, preparing
        annealing state, proposing and accepting moves across repeated rounds,
        cooling the temperature, and finalizing the layout.

    Raises
    ------
    ValueError
        If ``rounds`` or ``fineiter`` is negative.
    """
    if rounds < 0:
        raise ValueError("rounds must be non-negative.")
    if fineiter < 0:
        raise ValueError("fineiter must be non-negative.")

    ops = [
        FixedSteps(FixedStepsConfig(n=rounds + fineiter)),
        InitializeDHPositions(),
        PrepareDHState(),
        Repeat(
            n=rounds,
            ops=[
                DHAnnealingRound(),
                DHCool(),
            ],
        ),
        Repeat(
            n=fineiter,
            ops=[
                DHAnnealingRound(DHAnnealingRoundConfig(fine_tuning=True)),
            ],
        ),
    ]
    if not skip_finalization:
        ops.append(FinalizeDHPositions())

    return Pipeline(
        ops,
        name="davidson_harel_pipeline",
    )


def layout_davidson_harel_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    rounds: int = 100,
    fineiter: Optional[int] = None,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    skip_finalization: bool = True,
    fidelity_mode: bool = True,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run the Davidson-Harel pipeline as a drop-in replacement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]`` used only for output
        device and extent selection.
    rounds : int, default=100
        Number of annealing rounds.
    fineiter : int, optional
        Number of igraph-style fine-tuning rounds. ``None`` preserves
        python-igraph's default in fidelity mode and uses 10 in the local
        composable fallback.
    seed : int, default=42
        RNG seed for initialization and move proposals.
    edge_weights : torch.Tensor, optional
        Optional edge-weight vector with shape ``[E]``.
    skip_finalization : bool, default=True
        Whether to skip Dagua's legacy final centering/scaling pass.
    fidelity_mode : bool, default=True
        Whether to use the pure-Python igraph fidelity port. When ``False``,
        the local composable pipeline is used.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``rounds``, ``fineiter``, or ``edge_weights`` are
        invalid.
    RuntimeError
        If the pipeline does not populate final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if rounds < 0:
        raise ValueError("rounds must be non-negative.")
    if fineiter is not None and fineiter < 0:
        raise ValueError("fineiter must be non-negative.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge_count {edge_index.shape[1]}"
            )

    device = (
        edge_index.device
        if edge_index.numel() > 0
        else node_sizes.device
        if node_sizes is not None
        else torch.device("cpu")
    )
    if num_nodes == 0:
        resolved_dtype = resolve_fidelity_dtype(fidelity_mode, fidelity_dtype)
        return torch.empty((0, 2), dtype=resolved_dtype, device=device)
    if num_nodes == 1:
        resolved_dtype = resolve_fidelity_dtype(fidelity_mode, fidelity_dtype)
        return torch.zeros((1, 2), dtype=resolved_dtype, device=device)

    resolved_dtype = resolve_fidelity_dtype(fidelity_mode, fidelity_dtype)
    if fidelity_mode and edge_weights is None and skip_finalization:
        return _pure_igraph_davidson_harel_positions(
            edge_index=edge_index,
            num_nodes=num_nodes,
            seed=seed,
            rounds=rounds,
            fineiter=fineiter,
            device=device,
            fidelity_dtype=resolved_dtype,
        )

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device=str(device)))
    final_state = build_davidson_harel_pipeline(
        rounds=rounds,
        fineiter=10 if fineiter is None else fineiter,
        skip_finalization=skip_finalization,
    ).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("Davidson-Harel pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_davidson_harel_pipeline", "layout_davidson_harel_pipeline"]
