"""Stress-SGD layout pipeline."""

from __future__ import annotations

import ctypes
import math
from collections import deque
from typing import Optional, Union

import numpy as np
import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.pipelines import resolve_fidelity_dtype
from dagua.layout.ops.preprocess import BuildAdjacency, BuildAdjacencyConfig
from dagua.layout.ops.state import (  # noqa: E402
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)
from dagua.layout.ops.stress_sgd import (  # noqa: E402
    InitializeStressSGDState,
    PrepareStressSGDTerms,
    RunStressSGDApproximateSchedule,
    RunStressSGDExactSchedule,
)

_DEFAULT_EPS = 0.01
_DEFAULT_MAX_EXACT_NODES = 10_000
_OGDF_DEFAULT_ITERATIONS = 200
_OGDF_EDGE_COST = 100.0
_OGDF_RAND_BUCKETS = 1000
_OGDF_RAND_SCALE = 10.0


def _ogdf_adjacency(edge_index: torch.Tensor, num_nodes: int) -> list[list[int]]:
    """Build an OGDF-style undirected adjacency list.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    list[list[int]]
        Undirected adjacency list in input edge order.
    """
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return adjacency
    edge_index_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    for source, target in zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist()):
        if source == target:
            continue
        adjacency[int(source)].append(int(target))
        adjacency[int(target)].append(int(source))
    return adjacency


def _ogdf_initial_positions(num_nodes: int, seed: int) -> np.ndarray:
    """Generate the standalone OGDF runner's seeded initial coordinates.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    seed : int
        Seed forwarded to ``std::srand`` by ``scripts/ogdf_runner.cpp``.

    Returns
    -------
    np.ndarray
        Initial coordinates with shape ``[N, 2]`` and dtype ``float64``.
    """
    libc = ctypes.CDLL("libc.so.6")
    libc.srand(ctypes.c_uint(seed))
    positions = np.empty((num_nodes, 2), dtype=np.float64)
    for node_index in range(num_nodes):
        positions[node_index, 0] = float(libc.rand() % _OGDF_RAND_BUCKETS) / _OGDF_RAND_SCALE
        positions[node_index, 1] = float(libc.rand() % _OGDF_RAND_BUCKETS) / _OGDF_RAND_SCALE
    return positions


def _ogdf_bfs_hops(adjacency: list[list[int]], source: int) -> list[int]:
    """Compute unweighted BFS hop counts.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency list.
    source : int
        Source node index.

    Returns
    -------
    list[int]
        Hop counts with ``-1`` for unreachable nodes.
    """
    distances = [-1] * len(adjacency)
    distances[source] = 0
    queue: deque[int] = deque([source])
    while queue:
        current = queue.popleft()
        next_distance = distances[current] + 1
        for neighbor in adjacency[current]:
            if distances[neighbor] != -1:
                continue
            distances[neighbor] = next_distance
            queue.append(neighbor)
    return distances


def _ogdf_distance_matrix(adjacency: list[list[int]]) -> np.ndarray:
    """Build OGDF ``StressMinimization`` graph-distance matrix.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency list.

    Returns
    -------
    np.ndarray
        Dense distance matrix with shape ``[N, N]``. Each graph hop costs 100,
        and unreachable entries use ``100 * sqrt(N)``.
    """
    num_nodes = len(adjacency)
    distances = np.full((num_nodes, num_nodes), math.inf, dtype=np.float64)
    for source in range(num_nodes):
        for target, hop_count in enumerate(_ogdf_bfs_hops(adjacency, source)):
            if hop_count >= 0:
                distances[source, target] = float(hop_count) * _OGDF_EDGE_COST
    if num_nodes > 1:
        distances[~np.isfinite(distances)] = _OGDF_EDGE_COST * math.sqrt(float(num_nodes))
    np.fill_diagonal(distances, 0.0)
    return distances


def _ogdf_serial_sweep(positions: np.ndarray, distances: np.ndarray, weights: np.ndarray) -> None:
    """Apply one OGDF serial in-place stress-minimization sweep.

    Parameters
    ----------
    positions : np.ndarray
        Mutable coordinates with shape ``[N, 2]``.
    distances : np.ndarray
        Dense graph-distance matrix with shape ``[N, N]``.
    weights : np.ndarray
        Dense inverse-square weights with shape ``[N, N]``.

    Returns
    -------
    None
        ``positions`` is updated in place.
    """
    for source in range(int(positions.shape[0])):
        new_x = 0.0
        new_y = 0.0
        current_x = float(positions[source, 0])
        current_y = float(positions[source, 1])
        total_weight = 0.0
        for target in range(int(positions.shape[0])):
            if source == target:
                continue
            target_x = float(positions[target, 0])
            target_y = float(positions[target, 1])
            x_diff = current_x - target_x
            y_diff = current_y - target_y
            euclidean_dist = math.sqrt(x_diff * x_diff + y_diff * y_diff)
            weight = float(weights[source, target])
            vote_x = target_x
            vote_y = target_y
            if euclidean_dist != 0.0:
                desired_distance = float(distances[source, target])
                vote_x += desired_distance * (current_x - vote_x) / euclidean_dist
                vote_y += desired_distance * (current_y - vote_y) / euclidean_dist
            new_x += weight * vote_x
            new_y += weight * vote_y
            total_weight += weight
        if total_weight != 0.0:
            positions[source, 0] = new_x / total_weight
            positions[source, 1] = new_y / total_weight


def _layout_ogdf_stress(
    edge_index: torch.Tensor,
    num_nodes: int,
    seed: int,
    steps: int,
    init_pos: Optional[torch.Tensor],
    fidelity_dtype: torch.dtype,
) -> torch.Tensor:
    """Run the OGDF ``StressMinimization`` reference path.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    seed : int
        Runner seed.
    steps : int
        Number of serial stress sweeps.
    init_pos : torch.Tensor | None
        Optional warm-start positions with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Final coordinates with shape ``[N, 2]``.
    """
    if num_nodes <= 1:
        return torch.zeros((num_nodes, 2), dtype=fidelity_dtype, device=edge_index.device)
    positions = (
        _ogdf_initial_positions(num_nodes=num_nodes, seed=seed)
        if init_pos is None
        else init_pos.detach().cpu().numpy().astype(np.float64, copy=True)
    )
    distances = _ogdf_distance_matrix(_ogdf_adjacency(edge_index=edge_index, num_nodes=num_nodes))
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(distances > 0.0, 1.0 / (distances * distances), 0.0)
    np.fill_diagonal(weights, 0.0)
    iterations = steps if steps > 0 else _OGDF_DEFAULT_ITERATIONS
    for _ in range(iterations):
        _ogdf_serial_sweep(positions=positions, distances=distances, weights=weights)
    return torch.from_numpy(positions).to(device=edge_index.device, dtype=fidelity_dtype)


def build_stress_sgd_pipeline(
    steps: int = 30,
    eps: float = _DEFAULT_EPS,
    max_exact_nodes: int = _DEFAULT_MAX_EXACT_NODES,
    sample_size: Union[int, str] = "auto",
    trace_every: int = 0,
    fidelity_mode: bool = False,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> Pipeline:
    """Build a Stress-SGD layout pipeline.

    Reference fidelity
    ------------------
    Targets: ``s_gd2`` 1.8.1 stress SGD / Zheng, Pawar, and Goodman (2018),
        "Graph Drawing by Stochastic Gradient Descent".
    Fidelity mode: ``fidelity_mode=True`` enables reference preprocessing,
        disconnected-distance policy, float64 exact terms, and native term
        traversal order for ``classic_stress_sgd`` comparisons.
    Verified at: final 100-seed report, weak equivalent; median RMSD 0.035 to
        0.049 across epsilon and step-count variants.
    Known divergences:
        - Approximate large-graph mode still uses Dagua's sample-budget policy.
        - The fidelity contract is for stress-only SGD, not the unavailable
          historical multi-criteria reference.

    Parameters
    ----------
    steps : int
        Number of optimization epochs.
    eps : float
        Final schedule shrinkage factor.
    max_exact_nodes : int
        Node cutoff for choosing exact versus approximate terms.
    sample_size : int | str
        Sample budget for approximate mode.
    trace_every : int
        Optional snapshot interval.
    fidelity_mode : bool, default=False
        Enable reference-parity preprocessing, exact term precision, and native
        term traversal order for ``classic_stress_sgd`` versus ``s_gd2``
        comparisons.

    Returns
    -------
    Pipeline
        Pipeline implementing the Stress-SGD algorithm. The pipeline produces
        final node coordinates by building weighted adjacency, initializing the
        SGD state, preparing exact or approximate stress terms, and running the
        corresponding annealed optimization schedule with optional traces.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if sample_size != "auto" and (not isinstance(sample_size, int) or sample_size <= 0):
        raise ValueError("sample_size must be a positive integer or 'auto'.")

    return Pipeline(
        [
            BuildAdjacency(
                BuildAdjacencyConfig(
                    weighted=True,
                    dedup="sum" if fidelity_mode else "min",
                    format="list",
                    directed=False,
                )
            ),
            InitializeStressSGDState(
                trace_every=trace_every,
                reference_disconnected_policy=fidelity_mode,
            ),
            PrepareStressSGDTerms(
                max_exact_nodes=max_exact_nodes,
                exact_float64_terms=fidelity_mode,
                reference_term_order=fidelity_mode,
            ),
            RunStressSGDExactSchedule(steps=steps, eps=eps, trace_every=trace_every),
            RunStressSGDApproximateSchedule(
                steps=steps,
                eps=eps,
                sample_size=sample_size,
                trace_every=trace_every,
            ),
        ],
        name="stress_sgd_pipeline",
    )


def layout_stress_sgd_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    init_pos: Optional[torch.Tensor] = None,
    steps: int = 30,
    seed: int = 42,
    sample_size: Union[int, str] = "auto",
    trace_every: int = 0,
    eps: float = _DEFAULT_EPS,
    max_exact_nodes: int = _DEFAULT_MAX_EXACT_NODES,
    edge_weights: Optional[torch.Tensor] = None,
    fidelity_mode: bool = False,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> Union[torch.Tensor, "tuple[torch.Tensor, list[torch.Tensor]]"]:
    """Run the Stress-SGD pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor | None
        Optional node-size tensor with shape ``[N, 2]``.
    init_pos : torch.Tensor | None
        Optional initial coordinates with shape ``[N, 2]``. When provided,
        exact Stress-SGD starts from these coordinates instead of drawing from
        NumPy.
    steps : int
        Number of optimization epochs.
    seed : int
        RNG seed.
    sample_size : int | str
        Approximate-mode sample size.
    trace_every : int
        Optional trace interval.
    eps : float
        Final schedule shrinkage factor.
    max_exact_nodes : int
        Exact-path cutoff.
    edge_weights : torch.Tensor | None
        Optional edge-weight tensor with shape ``[E]``.
    fidelity_mode : bool, default=False
        Enable reference-parity edge preprocessing, disconnected-graph policy,
        exact ``float64`` term storage, and native term traversal order for
        ``s_gd2`` fidelity runs.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, list[torch.Tensor]]
        Final coordinates with shape ``[N, 2]``. When ``trace_every > 0``,
        periodic trace snapshots are returned alongside the final layout.

    Raises
    ------
    ValueError
        If the public Stress-SGD inputs are invalid.
    RuntimeError
        If the pipeline does not populate final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if trace_every < 0:
        raise ValueError("trace_every must be non-negative")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if max_exact_nodes < 0:
        raise ValueError("max_exact_nodes must be non-negative")

    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must be shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
            )
    if init_pos is not None:
        if init_pos.ndim != 2 or init_pos.shape != (num_nodes, 2):
            raise ValueError("init_pos must be shape [N, 2].")
    resolved_dtype = resolve_fidelity_dtype(fidelity_mode, fidelity_dtype)
    if fidelity_mode == "ogdf":
        positions = _layout_ogdf_stress(
            edge_index=edge_index,
            num_nodes=num_nodes,
            seed=seed,
            steps=steps,
            init_pos=init_pos,
            fidelity_dtype=resolved_dtype,
        )
        if trace_every > 0:
            return positions, []
        return positions

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    output_device = edge_index.device
    prepared_init_pos = (
        init_pos.to(device=output_device, dtype=resolved_dtype) if init_pos is not None else None
    )
    state = SolveState(pos=prepared_init_pos)
    ctx = RuntimeContext(plan=ExecutionPlan(device=str(output_device)))
    final_state = build_stress_sgd_pipeline(
        steps=steps,
        eps=eps,
        max_exact_nodes=max_exact_nodes,
        sample_size=sample_size,
        trace_every=trace_every,
        fidelity_mode=fidelity_mode,
        fidelity_dtype=resolved_dtype,
    ).apply(problem, state, ctx)

    if final_state.pos is None:
        raise RuntimeError("Stress-SGD pipeline did not produce final positions.")

    traces = final_state.extras.get("stress_sgd_traces", [])
    if trace_every > 0:
        return final_state.pos, traces
    return final_state.pos


__all__ = [
    "build_stress_sgd_pipeline",
    "layout_stress_sgd_pipeline",
]
