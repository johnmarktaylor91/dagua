"""Kamada-Kawai spring-embedding layout pipeline."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig
from dagua.layout.ops.distance import KamadaKawaiAllPairsShortestPaths
from dagua.layout.ops.graph_utils import layout_device as _layout_device
from dagua.layout.ops.init import KamadaKawaiInitializePositions
from dagua.layout.ops.optimize import LBFGSStep, LBFGSStepConfig
from dagua.layout.ops.postprocess import (
    KamadaKawaiFinalizePositions,
    KamadaKawaiFinalizePositionsConfig,
)
from dagua.layout.ops.state import (
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)

_DIRECTIONAL_ORIENTATION_MIN_GAIN = 0.05
_DIRECTION_TOP_TO_BOTTOM = "TB"
_DIRECTION_BOTTOM_TO_TOP = "BT"
_DIRECTION_LEFT_TO_RIGHT = "LR"
_DIRECTION_RIGHT_TO_LEFT = "RL"
_IGRAPH_KK_EPS = 1.0e-13
_IGRAPH_KK_CIRCLE_SCALE = 0.36
_IGRAPH_OUTPUT_SCALE = 50.0
_IGRAPH_DEFAULT_MAXITER_FACTOR = 50


def _igraph_seed_matrix(seed: Optional[int], num_nodes: int) -> np.ndarray:
    """Generate python-igraph's adapter-level seeded KK start matrix.

    Parameters
    ----------
    seed : int or None
        Benchmark seed forwarded by the fidelity adapter.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    numpy.ndarray
        Initial coordinate matrix with shape ``[N, 2]``.
    """
    if seed is None:
        theta = np.linspace(0.0, 1.0, num=num_nodes + 1, dtype=np.float64)[:-1]
        positions = np.column_stack((np.cos(theta * (2.0 * np.pi)), np.sin(theta * (2.0 * np.pi))))
        return positions * (_IGRAPH_KK_CIRCLE_SCALE * np.sqrt(float(num_nodes)))
    rng = np.random.RandomState(seed)
    return rng.uniform(-1.0, 1.0, size=(num_nodes, 2)).astype(np.float64, copy=False)


def _igraph_all_pairs_distances(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor],
) -> np.ndarray:
    """Compute igraph-compatible undirected KK shortest-path distances.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    edge_weights : torch.Tensor, optional
        Positive edge lengths with shape ``[E]``.

    Returns
    -------
    numpy.ndarray
        Dense float64 distance matrix with shape ``[N, N]``. Unreachable
        pairs are filled with the largest finite graph distance, matching
        igraph's 2D KK implementation.

    Raises
    ------
    ValueError
        If a supplied edge weight is non-positive.
    """
    distances = np.full((num_nodes, num_nodes), np.inf, dtype=np.float64)
    np.fill_diagonal(distances, 0.0)
    if edge_index.numel() > 0:
        edges = edge_index.detach().to(device="cpu", dtype=torch.long).numpy()
        weights: np.ndarray
        if edge_weights is None:
            weights = np.ones(edges.shape[1], dtype=np.float64)
        else:
            weights = edge_weights.detach().to(device="cpu", dtype=torch.float64).numpy()
        for edge_offset in range(edges.shape[1]):
            source = int(edges[0, edge_offset])
            target = int(edges[1, edge_offset])
            weight = float(weights[edge_offset])
            if weight <= 0.0:
                raise ValueError("edge_weights must be positive for igraph KK fidelity mode.")
            if 0 <= source < num_nodes and 0 <= target < num_nodes and source != target:
                if weight < distances[source, target]:
                    distances[source, target] = weight
                    distances[target, source] = weight

    for pivot in range(num_nodes):
        distances = np.minimum(distances, distances[:, [pivot]] + distances[[pivot], :])

    finite = distances[np.isfinite(distances)]
    max_distance = float(finite.max()) if finite.size > 0 else 0.0
    distances[distances > max_distance] = max_distance
    return distances


def _run_igraph_kamada_kawai(
    edge_index: torch.Tensor,
    num_nodes: int,
    steps: Optional[int],
    seed: Optional[int],
    pos: Optional[torch.Tensor],
    edge_weights: Optional[torch.Tensor],
) -> np.ndarray:
    """Run the igraph 2D Kamada-Kawai max-delta Newton update.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    steps : int or None
        Maximum number of outer iterations. ``None`` selects python-igraph's
        default scale of ``50 * N`` iterations.
    seed : int or None
        Seed used to build python-igraph's starting matrix.
    pos : torch.Tensor, optional
        Explicit initial coordinates with shape ``[N, 2]``.
    edge_weights : torch.Tensor, optional
        Positive edge lengths with shape ``[E]``.

    Returns
    -------
    numpy.ndarray
        Solved coordinates with shape ``[N, 2]`` before adapter output scale.
    """
    if pos is None:
        result = _igraph_seed_matrix(seed=seed, num_nodes=num_nodes)
    else:
        result = pos.detach().to(device="cpu", dtype=torch.float64).numpy().copy()
    if num_nodes <= 1:
        return result

    dij = _igraph_all_pairs_distances(edge_index, num_nodes, edge_weights)
    max_dij = float(np.max(dij))
    if max_dij <= 0.0:
        return result

    spring_length_scale = np.sqrt(float(num_nodes)) / max_dij
    kij = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    lij = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for j in range(num_nodes):
        for i in range(num_nodes):
            if i != j:
                distance = float(dij[i, j])
                kij[i, j] = float(num_nodes) / (distance * distance)
                lij[i, j] = spring_length_scale * distance

    d1 = np.zeros(num_nodes, dtype=np.float64)
    d2 = np.zeros(num_nodes, dtype=np.float64)
    for i in range(num_nodes):
        for m in range(num_nodes):
            if i == m:
                continue
            dx = result[m, 0] - result[i, 0]
            dy = result[m, 1] - result[i, 1]
            distance = float(np.sqrt(dx * dx + dy * dy))
            d1[m] += kij[m, i] * (dx - lij[m, i] * dx / distance)
            d2[m] += kij[m, i] * (dy - lij[m, i] * dy / distance)

    maxiter = _IGRAPH_DEFAULT_MAXITER_FACTOR * num_nodes if steps is None else int(steps)
    for _ in range(maxiter):
        selected = 0
        max_delta = -1.0
        for i in range(num_nodes):
            delta = float(d1[i] * d1[i] + d2[i] * d2[i])
            if delta > max_delta:
                selected = i
                max_delta = delta
        if max_delta < 0.0:
            break

        old_x = float(result[selected, 0])
        old_y = float(result[selected, 1])
        a_value = 0.0
        b_value = 0.0
        c_value = 0.0
        for i in range(num_nodes):
            if i == selected:
                continue
            dx = old_x - float(result[i, 0])
            dy = old_y - float(result[i, 1])
            distance_sq = dx * dx + dy * dy
            distance = float(np.sqrt(distance_sq))
            denominator = distance * distance_sq
            a_value += kij[selected, i] * (1.0 - lij[selected, i] * dy * dy / denominator)
            b_value += kij[selected, i] * lij[selected, i] * dx * dy / denominator
            c_value += kij[selected, i] * (1.0 - lij[selected, i] * dx * dx / denominator)

        my_d1 = float(d1[selected])
        my_d2 = float(d2[selected])
        if my_d1 * my_d1 + my_d2 * my_d2 < _IGRAPH_KK_EPS * _IGRAPH_KK_EPS:
            delta_x = 0.0
            delta_y = 0.0
        else:
            determinant = c_value * a_value - b_value * b_value
            delta_y = (b_value * my_d1 - a_value * my_d2) / determinant
            delta_x = (b_value * my_d2 - c_value * my_d1) / determinant

        new_x = old_x + delta_x
        new_y = old_y + delta_y
        d1[selected] = 0.0
        d2[selected] = 0.0
        for i in range(num_nodes):
            if i == selected:
                continue
            old_dx = old_x - float(result[i, 0])
            old_dy = old_y - float(result[i, 1])
            old_distance = float(np.sqrt(old_dx * old_dx + old_dy * old_dy))
            new_dx = new_x - float(result[i, 0])
            new_dy = new_y - float(result[i, 1])
            new_distance = float(np.sqrt(new_dx * new_dx + new_dy * new_dy))

            d1[i] -= kij[selected, i] * (-old_dx + lij[selected, i] * old_dx / old_distance)
            d2[i] -= kij[selected, i] * (-old_dy + lij[selected, i] * old_dy / old_distance)
            d1[i] += kij[selected, i] * (-new_dx + lij[selected, i] * new_dx / new_distance)
            d2[i] += kij[selected, i] * (-new_dy + lij[selected, i] * new_dy / new_distance)
            d1[selected] += kij[selected, i] * (new_dx - lij[selected, i] * new_dx / new_distance)
            d2[selected] += kij[selected, i] * (new_dy - lij[selected, i] * new_dy / new_distance)

        result[selected, 0] = new_x
        result[selected, 1] = new_y

    return result


def _directional_edge_fraction(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    direction: str,
) -> float:
    """Return the fraction of edges aligned with the requested direction.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    direction : str
        Layout direction, one of ``"TB"``, ``"BT"``, ``"LR"``, or ``"RL"``.

    Returns
    -------
    float
        Fraction of edges whose target is not behind the source along the
        requested layout axis. Self-loops count as aligned.
    """
    if edge_index.numel() == 0:
        return 1.0

    device_edge_index = edge_index.to(device=positions.device)
    source = device_edge_index[0]
    target = device_edge_index[1]
    self_loops = source == target
    if direction == _DIRECTION_LEFT_TO_RIGHT:
        aligned = (positions[target, 0] >= positions[source, 0]) | self_loops
    elif direction == _DIRECTION_RIGHT_TO_LEFT:
        aligned = (positions[target, 0] <= positions[source, 0]) | self_loops
    elif direction == _DIRECTION_BOTTOM_TO_TOP:
        aligned = (positions[target, 1] <= positions[source, 1]) | self_loops
    else:
        aligned = (positions[target, 1] >= positions[source, 1]) | self_loops
    return float(aligned.to(dtype=torch.float32).mean().item())


def _orient_positions_to_direction(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    direction: str,
) -> torch.Tensor:
    """Flip a KK embedding when doing so materially improves edge direction.

    Parameters
    ----------
    positions : torch.Tensor
        Solved position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    direction : str
        Layout direction, one of ``"TB"``, ``"BT"``, ``"LR"``, or ``"RL"``.

    Returns
    -------
    torch.Tensor
        Original or axis-flipped positions with shape ``[N, 2]``.
    """
    if positions.ndim != 2 or positions.shape[1] < 2 or edge_index.numel() == 0:
        return positions
    if direction not in {
        _DIRECTION_TOP_TO_BOTTOM,
        _DIRECTION_BOTTOM_TO_TOP,
        _DIRECTION_LEFT_TO_RIGHT,
        _DIRECTION_RIGHT_TO_LEFT,
    }:
        return positions

    base_fraction = _directional_edge_fraction(
        positions=positions,
        edge_index=edge_index,
        direction=direction,
    )
    flipped = positions.clone()
    if direction in {_DIRECTION_LEFT_TO_RIGHT, _DIRECTION_RIGHT_TO_LEFT}:
        flipped[:, 0] = -flipped[:, 0]
    else:
        flipped[:, 1] = -flipped[:, 1]
    flipped_fraction = _directional_edge_fraction(
        positions=flipped,
        edge_index=edge_index,
        direction=direction,
    )
    if flipped_fraction >= base_fraction + _DIRECTIONAL_ORIENTATION_MIN_GAIN:
        return flipped
    return positions


def build_kk_pipeline(
    steps: Optional[int] = None,
    trace_every: int = 0,
    preserve_float64: bool = False,
) -> Pipeline:
    """Build a Kamada-Kawai spring-embedding pipeline.

    Reference fidelity
    ------------------
    Targets: NetworkX 3.6.1 ``kamada_kawai_layout`` / Kamada and Kawai
        (1989), "An Algorithm for Drawing General Undirected Graphs".
    Fidelity mode: no dedicated flag; ``preserve_float64=True`` keeps audit
        output in double precision.
    Verified at: final 100-seed report, strong equivalent; median RMSD 0.000
        across 100, 300, and 1000 iteration variants.
    Known divergences:
        - Wrapper-level post-flip may orient directed graphs top-to-bottom for
          Dagua ergonomics.
        - The pipeline is tensor-native after shortest-path preparation rather
          than a direct NetworkX call.

    Parameters
    ----------
    steps : int, optional
        Maximum L-BFGS-B iterations. ``None`` or ``0`` leaves ``maxiter``
        unset to match classic KK.
    trace_every : int, default=0
        Snapshot cadence for optimizer traces.
    preserve_float64 : bool, default=False
        When ``True``, keep final coordinates in float64 for fidelity audits.

    Returns
    -------
    Pipeline
        Pipeline implementing the classical Kamada-Kawai algorithm. The
        pipeline produces final node coordinates by computing all-pairs
        shortest paths, initializing positions, minimizing the spring-energy
        objective with L-BFGS, and finalizing the layout.

    Raises
    ------
    ValueError
        If ``steps`` or ``trace_every`` are invalid.
    """
    if steps is not None and steps < 0:
        raise ValueError("steps must be non-negative.")
    if trace_every < 0:
        raise ValueError("trace_every must be non-negative.")

    optimize_config = LBFGSStepConfig(
        maxiter=steps,
        trace_every=trace_every,
        trace_key="kk_traces" if trace_every > 0 else None,
    )
    final_dtype = torch.float64 if preserve_float64 else torch.float32
    return Pipeline(
        [
            FixedSteps(FixedStepsConfig(n=0 if steps is None else steps)),
            KamadaKawaiAllPairsShortestPaths(),
            KamadaKawaiInitializePositions(),
            LBFGSStep(config=optimize_config),
            KamadaKawaiFinalizePositions(
                config=KamadaKawaiFinalizePositionsConfig(output_dtype=final_dtype)
            ),
        ],
        name="kk_pipeline",
    )


def layout_kk_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: Optional[int] = None,
    seed: int = 42,
    trace_every: int = 0,
    solver: str = "auto",
    pos: Optional[torch.Tensor] = None,
    edge_weights: Optional[torch.Tensor] = None,
    direction: str = "TB",
    orient_to_direction: bool = False,
    preserve_float64: bool = False,
    fidelity_mode: bool | str = False,
    fidelity_dtype: torch.dtype = torch.float32,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    """Run the Kamada-Kawai pipeline as a drop-in replacement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``. Unused and accepted
        for interface compatibility.
    steps : int, optional
        Maximum L-BFGS-B iterations. ``None`` or ``0`` leaves ``maxiter``
        unset to match classic KK.
    seed : int, default=42
        Accepted for interface compatibility. The 2D classic KK path uses
        deterministic circular initialization and does not consume a seed.
    trace_every : int, default=0
        Snapshot cadence for optimizer traces.
    solver : {"auto", "newton", "adam"}, default="auto"
        Retained for interface compatibility. All accepted values resolve to
        the classic SciPy L-BFGS-B path.
    pos : torch.Tensor, optional
        Initial positions with shape ``[N, 2]``. When provided, overrides the
        default circular initialization.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.
    direction : {"TB", "BT", "LR", "RL"}, default="TB"
        Requested graph reading direction used only when
        ``orient_to_direction`` is enabled.
    orient_to_direction : bool, default=False
        Whether to choose the axis orientation that materially improves edge
        direction consistency. The default is disabled so direct pipeline calls
        remain bit-exact with the archived NetworkX-style KK port.
    preserve_float64 : bool, default=False
        When ``True``, preserve float64 final coordinates for sub-percent
        fidelity audits. The default keeps historical float32 outputs.
    fidelity_mode : bool or str, default=False
        When ``True`` or ``"igraph"``, run the igraph-compatible KK update
        path for reference audits. The default retains the historical
        NetworkX-style L-BFGS implementation.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, list[torch.Tensor]]
        Final positions with shape ``[N, 2]``. When ``trace_every > 0``,
        periodic optimizer snapshots are returned alongside the final layout.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``steps``, ``trace_every``, ``solver``, or ``pos``
        are invalid.
    RuntimeError
        If the pipeline does not produce final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if steps is not None and steps < 0:
        raise ValueError("steps must be non-negative.")
    if trace_every < 0:
        raise ValueError("trace_every must be non-negative.")
    if solver not in {"auto", "newton", "adam"}:
        raise ValueError("solver must be one of 'auto', 'newton', or 'adam'.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
            )
        if bool(torch.any(edge_weights < 0).item()):
            raise ValueError("edge_weights must be nonnegative for KK shortest paths.")
    if pos is not None and pos.shape != (num_nodes, 2):
        raise ValueError(f"pos must have shape ({num_nodes}, 2), got {tuple(pos.shape)}")
    if fidelity_mode not in {False, True, "igraph"}:
        raise ValueError("fidelity_mode must be False, True, or 'igraph'.")

    output_device = _layout_device(edge_index=edge_index, node_sizes=node_sizes)
    output_dtype = torch.float64 if preserve_float64 else torch.float32
    if num_nodes == 0:
        empty = torch.empty((0, 2), dtype=output_dtype, device=output_device)
        return (empty, []) if trace_every > 0 else empty
    if num_nodes == 1:
        single = torch.zeros((1, 2), dtype=output_dtype, device=output_device)
        return (single, []) if trace_every > 0 else single
    if fidelity_mode in {True, "igraph"}:
        coordinates = _run_igraph_kamada_kawai(
            edge_index=edge_index,
            num_nodes=num_nodes,
            steps=steps,
            seed=seed,
            pos=pos,
            edge_weights=edge_weights,
        )
        final = torch.from_numpy(coordinates * _IGRAPH_OUTPUT_SCALE).to(
            device=output_device,
            dtype=output_dtype,
        )
        if orient_to_direction:
            final = _orient_positions_to_direction(
                positions=final,
                edge_index=edge_index,
                direction=direction,
            )
        return (final, []) if trace_every > 0 else final

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        direction=direction,
    )
    state = SolveState()
    if pos is not None:
        state.extras["kk_initial_pos"] = pos
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_kk_pipeline(
        steps=steps,
        trace_every=trace_every,
        preserve_float64=preserve_float64,
    ).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("KK pipeline did not produce final positions.")
    if orient_to_direction:
        final_state.pos = _orient_positions_to_direction(
            positions=final_state.pos,
            edge_index=edge_index,
            direction=direction,
        )

    if trace_every > 0:
        traces = final_state.extras.get("kk_traces", [])
        return final_state.pos, traces
    return final_state.pos


__all__ = ["build_kk_pipeline", "layout_kk_pipeline"]
