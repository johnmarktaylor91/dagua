"""Fruchterman-Reingold force-directed layout pipeline."""

from __future__ import annotations

import math
import random
from typing import Optional, Sequence, Union

import numpy as np
import torch

from dagua.layout.ops.anneal import InitTemperatureFromExtent, LinearCool
from dagua.layout.ops.base import Conditional, Pipeline, Repeat  # noqa: E402
from dagua.layout.ops.converge import (
    FixedSteps,
    FixedStepsConfig,
    FRConvergenceCheck,
)  # noqa: E402
from dagua.layout.ops.force import ApplyDisplacement, ApplyDisplacementConfig, FRCombinedForce
from dagua.layout.ops.init import RandomUniformInit, RandomUniformInitConfig
from dagua.layout.ops.postprocess import FRFinalizePositions, FRFinalizePositionsConfig
from dagua.layout.ops.preprocess import FRPrepareAdjacency, FRPrepareAdjacencyConfig
from dagua.layout.ops.state import (  # noqa: E402
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)

_LEGACY_CLASSIC_FR_STEPS = 200
_CANONICAL_NX_SPRING_STEPS = 50
_IGRAPH_FR_DEFAULT_STEPS = 500
_IGRAPH_ADAPTER_SCALE = 50.0
_IGRAPH_JITTER = 1.0e-9
_FR_DAG_DROP_TOLERANCE = 0.1
_FR_SCORE_DROP_TOLERANCE = 1.0e-6


def _igraph_layout_align_positions(positions: np.ndarray, edges: list[tuple[int, int]]) -> None:
    """Center and rotate positions using igraph's layout-alignment rule.

    Parameters
    ----------
    positions : numpy.ndarray
        Mutable position matrix with shape ``[N, 2]`` and dtype ``float64``.
    edges : list[tuple[int, int]]
        Edge list in igraph edge order.

    Returns
    -------
    None
        The ``positions`` array is modified in-place.
    """
    num_nodes = int(positions.shape[0])
    if num_nodes == 0:
        return

    center_x = 0.0
    center_y = 0.0
    for node in range(num_nodes):
        center_x += float(positions[node, 0])
        center_y += float(positions[node, 1])
    center_x /= float(num_nodes)
    center_y /= float(num_nodes)
    for node in range(num_nodes):
        positions[node, 0] -= center_x
        positions[node, 1] -= center_y

    matrix = np.zeros((2, 2), dtype=np.float64)
    correction = np.zeros((2, 2), dtype=np.float64)
    norm2_sum = 0.0
    correction_norm2_sum = 0.0
    correction_saved = False

    for source, target in edges:
        if source == target:
            continue

        edge_vec = (
            float(positions[source, 0] - positions[target, 0]),
            float(positions[source, 1] - positions[target, 1]),
        )
        for row in range(2):
            for col in range(2):
                term = edge_vec[row] * edge_vec[col]
                matrix[row, col] += term
                if row == col:
                    norm2_sum += term

        if not correction_saved and norm2_sum > 0.0:
            correction_saved = True
            correction_norm2_sum = norm2_sum
            correction[:, :] = matrix

    if norm2_sum == 0.0:
        for node in range(num_nodes):
            vertex_vec = (float(positions[node, 0]), float(positions[node, 1]))
            for row in range(2):
                for col in range(2):
                    term = vertex_vec[row] * vertex_vec[col]
                    matrix[row, col] += term
                    if row == col:
                        norm2_sum += term

            if not correction_saved and norm2_sum > 0.0:
                correction_saved = True
                correction_norm2_sum = norm2_sum
                correction[:, :] = matrix

    if norm2_sum == 0.0:
        return

    retried = False
    while True:
        tensor = matrix.copy()
        tensor *= 1.0 / norm2_sum
        tensor[0, 0] -= 0.5
        tensor[1, 1] -= 0.5

        eigenvalues, rotation = np.linalg.eigh(tensor)
        matrix_norm = max(abs(float(eigenvalues[0])), abs(float(eigenvalues[1])))
        if matrix_norm > 1.0e-3 or retried:
            break

        matrix -= correction
        norm2_sum -= correction_norm2_sum
        if norm2_sum == 0.0:
            return
        retried = True

    temp_layout = positions @ rotation
    extent_x = float(np.max(temp_layout[:, 0]) - np.min(temp_layout[:, 0]))
    extent_y = float(np.max(temp_layout[:, 1]) - np.min(temp_layout[:, 1]))
    if extent_x >= extent_y:
        positions[:, :] = temp_layout
    else:
        positions[:, 0] = temp_layout[:, 1]
        positions[:, 1] = temp_layout[:, 0]


def _dag_consistency_fraction(pos: torch.Tensor, edge_index: torch.Tensor) -> float:
    """Compute the TB directed-edge consistency fraction.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.

    Returns
    -------
    float
        Fraction of edges whose target is not above their source.
    """
    if edge_index.numel() == 0:
        return 1.0
    source = edge_index[0].to(device=pos.device)
    target = edge_index[1].to(device=pos.device)
    self_loops = source == target
    correct = (pos[target, 1] >= pos[source, 1]) | self_loops
    return float(correct.to(dtype=torch.float32).mean().item())


def _quick_directed_composite_score(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: Optional[torch.Tensor],
) -> float:
    """Compute the cheap directed composite used by the FR default selector.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]`` for overlap scoring.

    Returns
    -------
    float
        Directed composite score from Tier-1 metrics only.
    """
    from dagua.metrics import composite, quick

    return float(composite(quick(pos, edge_index, node_sizes=node_sizes, seed=0)))


def _choose_fr_default_layout(
    legacy_pos: torch.Tensor,
    canonical_pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: Optional[torch.Tensor],
    dag_drop_tolerance: float = _FR_DAG_DROP_TOLERANCE,
    score_drop_tolerance: float = _FR_SCORE_DROP_TOLERANCE,
) -> torch.Tensor:
    """Choose between legacy 200-step FR and canonical NetworkX-style FR.

    Parameters
    ----------
    legacy_pos : torch.Tensor
        Existing dagua ``classic_fr`` default output with shape ``[N, 2]``.
    canonical_pos : torch.Tensor
        NetworkX-compatible 50-step FR output with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]`` for overlap scoring.
    dag_drop_tolerance : float, default=0.1
        Maximum allowed drop in TB edge consistency before preserving the
        legacy layout.
    score_drop_tolerance : float, default=1.0e-6
        Maximum allowed Tier-1 composite drop before preserving the legacy
        layout.

    Returns
    -------
    torch.Tensor
        Selected position tensor with shape ``[N, 2]``.
    """
    legacy_dag = _dag_consistency_fraction(legacy_pos, edge_index)
    canonical_dag = _dag_consistency_fraction(canonical_pos, edge_index)
    if canonical_dag + dag_drop_tolerance < legacy_dag:
        return legacy_pos

    legacy_score = _quick_directed_composite_score(legacy_pos, edge_index, node_sizes)
    canonical_score = _quick_directed_composite_score(canonical_pos, edge_index, node_sizes)
    if canonical_score + score_drop_tolerance < legacy_score:
        return legacy_pos
    return canonical_pos


def _normalize_fixed_indices(
    fixed: Optional[Union[Sequence[int], torch.Tensor]],
    num_nodes: int,
) -> tuple[int, ...]:
    """Validate and normalize fixed-node indices.

    Parameters
    ----------
    fixed : sequence of int or torch.Tensor, optional
        Node indices whose FR displacement should be zeroed.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    tuple[int, ...]
        Sorted unique fixed-node indices.

    Raises
    ------
    ValueError
        If any fixed index is outside ``[0, num_nodes)``.
    """
    if fixed is None:
        return ()
    if isinstance(fixed, torch.Tensor):
        raw_indices = fixed.detach().to(device="cpu", dtype=torch.long).flatten().tolist()
    else:
        raw_indices = [int(index) for index in fixed]
    normalized = tuple(sorted(set(int(index) for index in raw_indices)))
    if any(index < 0 or index >= num_nodes for index in normalized):
        raise ValueError("fixed contains a node index outside [0, num_nodes).")
    return normalized


def _is_igraph_fidelity_mode(fidelity_mode: Optional[Union[bool, str]]) -> bool:
    """Return whether the igraph C-reference FR loop is requested.

    Parameters
    ----------
    fidelity_mode : bool or str, optional
        Fidelity selector. ``True`` and ``"igraph"`` enable igraph fidelity;
        ``False`` and ``None`` preserve the existing NetworkX-compatible path.

    Returns
    -------
    bool
        Whether to use the igraph-compatible solver.

    Raises
    ------
    ValueError
        If ``fidelity_mode`` is an unsupported string or value.
    """
    if fidelity_mode in (None, False):
        return False
    if fidelity_mode is True:
        return True
    if isinstance(fidelity_mode, str) and fidelity_mode == "igraph":
        return True
    raise ValueError("FR fidelity_mode must be None, False, True, or 'igraph'.")


def _weakly_connected(edge_index: torch.Tensor, num_nodes: int) -> bool:
    """Evaluate igraph's weak-connectivity branch condition.

    Parameters
    ----------
    edge_index : torch.Tensor
        Directed edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    bool
        ``True`` when all nodes are in one weak component.
    """
    if num_nodes <= 1:
        return True
    if edge_index.numel() == 0:
        return False

    neighbors: list[list[int]] = [[] for _ in range(num_nodes)]
    for source, target in edge_index.detach().to(device="cpu", dtype=torch.long).t().tolist():
        source_index = int(source)
        target_index = int(target)
        if source_index == target_index:
            continue
        neighbors[source_index].append(target_index)
        neighbors[target_index].append(source_index)

    seen = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for neighbor in neighbors[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return len(seen) == num_nodes


def _igraph_seed_positions(num_nodes: int, seed: int) -> np.ndarray:
    """Create the python-igraph adapter's seeded initial FR matrix.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    seed : int
        Integer seed supplied to the benchmark adapter.

    Returns
    -------
    numpy.ndarray
        Initial positions with shape ``[N, 2]`` sampled from ``[-1, 1]``.
    """
    rng = np.random.RandomState(seed)
    return np.asarray(rng.uniform(-1.0, 1.0, size=(num_nodes, 2)), dtype=np.float64)


def _igraph_fr_reference_positions(
    edge_index: torch.Tensor,
    num_nodes: int,
    steps: int,
    seed: int,
    edge_weights: Optional[torch.Tensor] = None,
    pos: Optional[torch.Tensor] = None,
    start_temp: Optional[float] = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Run the igraph 2D Fruchterman-Reingold C-reference loop.

    Parameters
    ----------
    edge_index : torch.Tensor
        Directed edge tensor with shape ``[2, E]``. Edge order is preserved.
    num_nodes : int
        Number of graph nodes.
    steps : int
        Number of igraph FR iterations.
    seed : int
        Seed for initial positions and displacement jitter.
    edge_weights : torch.Tensor, optional
        Optional positive edge weights with shape ``[E]``.
    pos : torch.Tensor, optional
        Optional initial positions with shape ``[N, 2]``.
    start_temp : float, optional
        Initial temperature. ``None`` uses python-igraph's
        ``sqrt(num_nodes) / 10`` default.
    output_dtype : torch.dtype, default=torch.float32
        Floating dtype for the returned tensor.

    Returns
    -------
    torch.Tensor
        Adapter-scaled positions with shape ``[N, 2]``.
    """
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=output_dtype)
    if pos is None:
        positions = _igraph_seed_positions(num_nodes=num_nodes, seed=seed)
    else:
        positions = pos.detach().to(device="cpu", dtype=torch.float64).numpy().copy()

    edges = [
        (int(source), int(target))
        for source, target in edge_index.detach().to(device="cpu", dtype=torch.long).t().tolist()
    ]
    for source, target in edges:
        source_index = int(source)
        target_index = int(target)
        if (
            source_index < 0
            or source_index >= num_nodes
            or target_index < 0
            or target_index >= num_nodes
        ):
            raise ValueError("edge_index contains a node index outside [0, num_nodes).")

    if edge_weights is None:
        weights = [1.0] * len(edges)
    else:
        weight_tensor = edge_weights.detach().to(device="cpu", dtype=torch.float64)
        weights = [float(weight) for weight in weight_tensor.tolist()]
        if any(weight <= 0.0 for weight in weights):
            raise ValueError("Weights must be positive for igraph FR fidelity mode.")

    if steps == 0:
        _igraph_layout_align_positions(positions=positions, edges=edges)
        return torch.from_numpy(positions * _IGRAPH_ADAPTER_SCALE).to(dtype=output_dtype)

    temp = math.sqrt(float(num_nodes)) / 10.0 if start_temp is None else float(start_temp)
    difftemp = temp / float(steps)
    connected = _weakly_connected(edge_index=edge_index, num_nodes=num_nodes)
    component_constant = float(num_nodes) * math.sqrt(float(num_nodes))
    jitter_rng = random.Random(seed)
    dispx = np.zeros(num_nodes, dtype=np.float64)
    dispy = np.zeros(num_nodes, dtype=np.float64)

    for _ in range(steps):
        dispx.fill(0.0)
        dispy.fill(0.0)

        for source in range(num_nodes):
            for target in range(source + 1, num_nodes):
                dx = positions[source, 0] - positions[target, 0]
                dy = positions[source, 1] - positions[target, 1]
                dlen = dx * dx + dy * dy
                while dlen == 0.0:
                    dx = jitter_rng.uniform(-_IGRAPH_JITTER, _IGRAPH_JITTER)
                    dy = jitter_rng.uniform(-_IGRAPH_JITTER, _IGRAPH_JITTER)
                    dlen = dx * dx + dy * dy
                if connected:
                    dispx[source] += dx / dlen
                    dispy[source] += dy / dlen
                    dispx[target] -= dx / dlen
                    dispy[target] -= dy / dlen
                else:
                    rdlen = math.sqrt(dlen)
                    contribution_x = (
                        dx * (component_constant - dlen * rdlen) / (dlen * component_constant)
                    )
                    contribution_y = (
                        dy * (component_constant - dlen * rdlen) / (dlen * component_constant)
                    )
                    dispx[source] += contribution_x
                    dispy[source] += contribution_y
                    dispx[target] -= contribution_x
                    dispy[target] -= contribution_y

        for (source, target), weight in zip(edges, weights):
            source_index = int(source)
            target_index = int(target)
            dx = positions[source_index, 0] - positions[target_index, 0]
            dy = positions[source_index, 1] - positions[target_index, 1]
            dlen = math.sqrt(dx * dx + dy * dy) * float(weight)
            dispx[source_index] -= dx * dlen
            dispy[source_index] -= dy * dlen
            dispx[target_index] += dx * dlen
            dispy[target_index] += dy * dlen

        for node in range(num_nodes):
            dx = dispx[node] + jitter_rng.uniform(-_IGRAPH_JITTER, _IGRAPH_JITTER)
            dy = dispy[node] + jitter_rng.uniform(-_IGRAPH_JITTER, _IGRAPH_JITTER)
            displen = math.sqrt(dx * dx + dy * dy)
            if displen > temp:
                dx *= temp / displen
                dy *= temp / displen
            if displen > 0.0:
                positions[node, 0] += dx
                positions[node, 1] += dy

        temp -= difftemp

    _igraph_layout_align_positions(positions=positions, edges=edges)
    return torch.from_numpy(positions * _IGRAPH_ADAPTER_SCALE).to(dtype=output_dtype)


def _networkx_adjacency_matrix(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor],
) -> np.ndarray:
    """Build NetworkX ``DiGraph`` adjacency data with summed duplicate edges.

    Parameters
    ----------
    edge_index : torch.Tensor
        Directed edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.

    Returns
    -------
    numpy.ndarray
        Dense float64 adjacency matrix with shape ``[N, N]`` matching the
        NetworkX competitor adapter's simple ``DiGraph`` duplicate handling.
    """
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    edge_array = edge_index.detach().to(device="cpu", dtype=torch.long).numpy()
    weight_array = None if edge_weights is None else edge_weights.detach().cpu().numpy()
    for edge_offset in range(edge_array.shape[1]):
        source = int(edge_array[0, edge_offset])
        target = int(edge_array[1, edge_offset])
        if source < 0 or source >= num_nodes or target < 0 or target >= num_nodes:
            raise ValueError("edge_index contains a node index outside [0, num_nodes).")
        weight = 1.0 if weight_array is None else float(weight_array[edge_offset])
        adjacency[source, target] += weight
    return adjacency


def _networkx_fr_reference_positions(
    edge_index: torch.Tensor,
    num_nodes: int,
    steps: int,
    seed: int,
    edge_weights: Optional[torch.Tensor] = None,
    pos: Optional[torch.Tensor] = None,
    k: Optional[float] = None,
    fixed: tuple[int, ...] = (),
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Run NetworkX 3.6.1's dense Fruchterman-Reingold loop locally.

    Parameters
    ----------
    edge_index : torch.Tensor
        Directed edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    steps : int
        NetworkX ``iterations`` value.
    seed : int
        Seed forwarded to NetworkX's ``np_random_state`` wrapper.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.
    pos : torch.Tensor, optional
        Optional initial positions with shape ``[N, 2]``.
    k : float, optional
        Optimal node distance. ``None`` uses ``sqrt(1 / N)``.
    fixed : tuple[int, ...], default=()
        Node indices whose displacement is zeroed each iteration.
    output_dtype : torch.dtype, default=torch.float32
        Floating dtype for the returned tensor.

    Returns
    -------
    torch.Tensor
        NetworkX-adapter-scaled positions with shape ``[N, 2]``.
    """
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=output_dtype)

    adjacency = _networkx_adjacency_matrix(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weights=edge_weights,
    )
    if pos is None:
        rng = np.random.RandomState(seed)
        positions = np.asarray(rng.rand(num_nodes, 2), dtype=adjacency.dtype)
    else:
        positions = (
            pos.detach()
            .to(device="cpu", dtype=torch.float64)
            .numpy()
            .astype(
                adjacency.dtype,
                copy=True,
            )
        )

    optimal_distance = math.sqrt(1.0 / num_nodes) if k is None else float(k)
    temperature = (
        max(
            max(positions.T[0]) - min(positions.T[0]),
            max(positions.T[1]) - min(positions.T[1]),
        )
        * 0.1
    )
    cooling_delta = temperature / (steps + 1)
    fixed_array = np.asarray(fixed, dtype=np.int64) if fixed else None

    for _ in range(steps):
        delta = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        distance = np.linalg.norm(delta, axis=-1)
        np.clip(distance, 0.01, None, out=distance)
        displacement = np.einsum(
            "ijk,ij->ik",
            delta,
            (optimal_distance * optimal_distance / distance**2)
            - (adjacency * distance / optimal_distance),
        )
        length = np.linalg.norm(displacement, axis=-1)
        np.clip(length, a_min=0.01, a_max=None, out=length)
        delta_pos = np.einsum("ij,i->ij", displacement, temperature / length)
        if fixed_array is not None:
            delta_pos[fixed_array] = 0.0
        positions += delta_pos
        temperature -= cooling_delta
        if (np.linalg.norm(delta_pos) / num_nodes) < 1.0e-4:
            break

    if fixed_array is None:
        positions = positions - positions.mean(axis=0)
        lim = np.abs(positions).max()
        if lim > 0.0:
            positions = positions * (1.0 / lim)

    output_scale = 1.0 if fixed_array is not None else 500.0
    return torch.from_numpy(positions * output_scale).to(dtype=output_dtype)


def build_fr_pipeline(
    steps: int = 50,
    networkx_compat: bool = False,
    k: Optional[float] = None,
    fixed_indices: Optional[Sequence[int]] = None,
) -> Pipeline:
    """Build a Fruchterman-Reingold force-directed layout pipeline.

    Reference fidelity
    ------------------
    Targets: NetworkX 3.6.1 ``spring_layout`` / Fruchterman and Reingold
        (1991), "Graph Drawing by Force-directed Placement".
    Fidelity mode: ``networkx_compat=True`` switches final scaling to the
        NetworkX adapter contract; fixed nodes additionally suppress rescaling.
    Verified at: final 100-seed report, strong equivalent; median RMSD 0.088
        to 0.136 across step-count variants.
    Known divergences:
        - Default dagua display scaling remains larger than NetworkX output.
        - Directed graph selection still happens in the wrapper, not this
          builder.

    Parameters
    ----------
    steps : int, default=50
        Maximum number of cooling iterations to run.
    networkx_compat : bool, default=False
        If ``True``, use NetworkX adapter-scale finalization instead of
        dagua's legacy ``50 * sqrt(N)`` display scale.
    k : float, optional
        Explicit NetworkX-style optimal node spacing.
    fixed_indices : sequence of int, optional
        Node indices whose displacement should be zeroed. When provided, final
        centering/scaling is skipped to match NetworkX fixed-node semantics.

    Returns
    -------
    Pipeline
        Pipeline implementing the classical Fruchterman-Reingold algorithm.
        The pipeline produces final node coordinates by sampling unit-square
        initial positions, building adjacency data, setting an initial
        temperature from the current extent, iterating attraction and
        repulsion force updates with displacement and linear cooling, then
        finalizing the coordinates.

    Raises
    ------
    ValueError
        If ``steps`` is negative.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative.")

    return Pipeline(
        [
            FixedSteps(FixedStepsConfig(n=steps)),
            Conditional(
                predicate=lambda problem, state, ctx: state.pos is None,
                op=RandomUniformInit(
                    RandomUniformInitConfig(
                        scale="none",
                        rng_backend="numpy",
                    ),
                ),
            ),
            FRPrepareAdjacency(FRPrepareAdjacencyConfig(k=k)),
            InitTemperatureFromExtent(),
            Repeat(
                n=steps,
                ops=[
                    FRCombinedForce(),
                    ApplyDisplacement(
                        ApplyDisplacementConfig(
                            fixed_indices=tuple(fixed_indices or ()),
                        ),
                    ),
                    FRConvergenceCheck(),
                    LinearCool(),
                ],
            ),
            FRFinalizePositions(
                FRFinalizePositionsConfig(
                    output_scale_factor=500.0 if networkx_compat else 50.0,
                    scale_by_sqrt_num_nodes=not networkx_compat,
                    skip_rescale=bool(fixed_indices),
                ),
            ),
        ],
        name="fr_pipeline",
    )


def layout_fr_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 50,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    pos: Optional[torch.Tensor] = None,
    networkx_compat: bool = False,
    k: Optional[float] = None,
    fixed: Optional[Union[Sequence[int], torch.Tensor]] = None,
    fidelity_mode: Optional[Union[bool, str]] = None,
    fidelity_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Run the Fruchterman-Reingold pipeline as a drop-in replacement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]`` used only to resolve
        the output device.
    steps : int, default=50
        Maximum number of cooling iterations to run.
    seed : int, default=42
        Random seed for the NumPy-backed unit-square initialization.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.
    pos : torch.Tensor, optional
        Initial positions with shape ``[N, 2]``. When provided, the pipeline
        starts from these coordinates instead of sampling a random
        initialization.
    networkx_compat : bool, default=False
        If ``True``, use NetworkX-compatible adapter-scale finalization. This
        preserves the force loop while avoiding dagua's legacy display scale.
    k : float, optional
        Explicit NetworkX-style optimal node spacing. ``None`` preserves
        ``sqrt(1 / num_nodes)``.
    fixed : sequence of int or torch.Tensor, optional
        Node indices to hold fixed during displacement. A full ``pos`` tensor
        must also be provided, matching NetworkX's fixed-node requirement.
    fidelity_mode : bool or str, optional
        ``True`` or ``"igraph"`` uses the igraph C-reference force loop. The
        unchanged NetworkX default of ``steps=50`` maps to python-igraph's
        default ``niter=500`` in this mode.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``steps``, ``edge_weights``, or ``pos`` are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
            )
    if pos is not None and pos.shape != (num_nodes, 2):
        raise ValueError(f"pos must have shape ({num_nodes}, 2), got {tuple(pos.shape)}")
    if k is not None and k <= 0.0:
        raise ValueError("k must be positive when provided.")
    fixed_indices = _normalize_fixed_indices(fixed=fixed, num_nodes=num_nodes)
    if fixed_indices and pos is None:
        raise ValueError("fixed nodes require a full pos tensor.")
    if _is_igraph_fidelity_mode(fidelity_mode=fidelity_mode):
        if fixed_indices:
            raise ValueError("fixed nodes are not supported in igraph FR fidelity mode.")
        if k is not None:
            raise ValueError("k is not supported in igraph FR fidelity mode.")
        output_device = (
            edge_index.device
            if edge_index.numel() > 0
            else node_sizes.device
            if node_sizes is not None
            else torch.device("cpu")
        )
        igraph_steps = _IGRAPH_FR_DEFAULT_STEPS if steps == _CANONICAL_NX_SPRING_STEPS else steps
        return _igraph_fr_reference_positions(
            edge_index=edge_index,
            num_nodes=num_nodes,
            steps=igraph_steps,
            seed=seed,
            edge_weights=edge_weights,
            pos=pos,
            output_dtype=fidelity_dtype,
        ).to(device=output_device)

    if networkx_compat:
        output_device = (
            edge_index.device
            if edge_index.numel() > 0
            else node_sizes.device
            if node_sizes is not None
            else torch.device("cpu")
        )
        return _networkx_fr_reference_positions(
            edge_index=edge_index,
            num_nodes=num_nodes,
            steps=steps,
            seed=seed,
            edge_weights=edge_weights,
            pos=pos,
            k=k,
            fixed=fixed_indices,
            output_dtype=fidelity_dtype,
        ).to(device=output_device)

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    if pos is not None:
        state.pos = pos.detach().clone().to(dtype=torch.float64)
    output_device = (
        edge_index.device
        if edge_index.numel() > 0
        else node_sizes.device
        if node_sizes is not None
        else torch.device("cpu")
    )
    if pos is not None and steps == 0:
        return state.pos.to(device=output_device, dtype=torch.float32)
    ctx = RuntimeContext(plan=ExecutionPlan(device=str(output_device)))
    final_state = build_fr_pipeline(
        steps=steps,
        networkx_compat=networkx_compat,
        k=k,
        fixed_indices=fixed_indices,
    ).apply(
        problem,
        state,
        ctx,
    )
    if final_state.pos is None:
        raise RuntimeError("FR pipeline did not produce final positions.")
    return final_state.pos


def layout_fr_default_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = _LEGACY_CLASSIC_FR_STEPS,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    pos: Optional[torch.Tensor] = None,
    networkx_compat: bool = False,
    k: Optional[float] = None,
    fixed: Optional[Union[Sequence[int], torch.Tensor]] = None,
    fidelity_mode: Optional[Union[bool, str]] = None,
    fidelity_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Run the benchmark default FR layout with canonical-fidelity selection.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]`` for selector scoring.
    steps : int, default=200
        Requested FR iteration count. Non-default values run exactly as
        requested and bypass the selector.
    seed : int, default=42
        Random seed for both default candidates.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.
    pos : torch.Tensor, optional
        Initial positions with shape ``[N, 2]``. Warm starts run exactly as
        requested and bypass the selector.
    networkx_compat : bool, default=False
        If ``True``, forwarded to :func:`layout_fr_pipeline` for exact
        NetworkX adapter-style output scaling.
    k : float, optional
        Explicit NetworkX-style optimal node spacing.
    fixed : sequence of int or torch.Tensor, optional
        Node indices to hold fixed during displacement.
    fidelity_mode : bool or str, optional
        Forwarded to :func:`layout_fr_pipeline`.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.
    """
    if (
        steps != _LEGACY_CLASSIC_FR_STEPS
        or pos is not None
        or k is not None
        or fixed is not None
        or _is_igraph_fidelity_mode(fidelity_mode=fidelity_mode)
    ):
        return layout_fr_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            node_sizes=node_sizes,
            steps=steps,
            seed=seed,
            edge_weights=edge_weights,
            pos=pos,
            networkx_compat=networkx_compat,
            k=k,
            fixed=fixed,
            fidelity_mode=fidelity_mode,
            fidelity_dtype=fidelity_dtype,
        )

    legacy_pos = layout_fr_pipeline(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        steps=_LEGACY_CLASSIC_FR_STEPS,
        seed=seed,
        edge_weights=edge_weights,
        networkx_compat=networkx_compat,
        k=k,
        fixed=fixed,
    )
    canonical_pos = layout_fr_pipeline(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        steps=_CANONICAL_NX_SPRING_STEPS,
        seed=seed,
        edge_weights=edge_weights,
        networkx_compat=networkx_compat,
        k=k,
        fixed=fixed,
    )
    return _choose_fr_default_layout(
        legacy_pos=legacy_pos,
        canonical_pos=canonical_pos,
        edge_index=edge_index,
        node_sizes=node_sizes,
    )


__all__ = [
    "build_fr_pipeline",
    "layout_fr_default_pipeline",
    "layout_fr_pipeline",
]
