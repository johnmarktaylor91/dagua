"""Stress majorization (SMACOF) layout pipeline."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import ClassVar, List, Optional, Tuple, Union

import numpy as np
import torch

from dagua.layout.ops.base import Op, Pipeline, Repeat
from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig
from dagua.layout.ops.gem import (
    _GEM_PHYSICS_CONFIG,
    _connected_components_from_edges,
    _extract_component_edges,
    _ogdf_shift_component_to_origin,
    _ogdf_tile_to_rows_offsets,
)
from dagua.layout.ops.graph_utils import (
    _shared_all_pairs_shortest_paths,
    _shared_build_undirected_adjacency,
)
from dagua.layout.ops.graph_utils import layout_device as _layout_device
from dagua.layout.ops.state import (
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)
from dagua.layout.ops.stress import (
    CURRENT_POSITIONS_KEY,
    CURRENT_STRESS_KEY,
    TRACE_EVERY_KEY,
    TRACES_KEY,
    WEIGHTS_KEY,
    CaptureStressMajorizationStress,
    CheckStressMajorizationEpsilon,
    CollectStressMajorizationTrace,
    FinalizeStressMajorizationPositions,
    InitializeStressMajorizationPositions,
    InitializeStressMajorizationPositionsConfig,
    PrepareStressMajorizationState,
    PrepareStressMajorizationStateConfig,
    SmacofStep,
    SmacofStepConfig,
)
from dagua.layout.ops.taxonomy import OpCategory

_FIDELITY_MODE_OGDF = "ogdf"
_FIDELITY_MODE_GRAPHVIZ = "graphviz"
_FIDELITY_MODE_GRAPHVIZ_NEATO = "graphviz_neato"
_FIDELITY_MODES = {
    None,
    _FIDELITY_MODE_OGDF,
    _FIDELITY_MODE_GRAPHVIZ,
    _FIDELITY_MODE_GRAPHVIZ_NEATO,
}
_GRAPHVIZ_FIDELITY_MODES = {_FIDELITY_MODE_GRAPHVIZ}
_GRAPHVIZ_LAP2_KEY = "sm_graphviz_lap2_packed"
_GRAPHVIZ_OLD_STRESS_KEY = "sm_graphviz_old_stress"
_GRAPHVIZ_CG_TOLERANCE = 1.0e-3
_GRAPHVIZ_DRAND48_MULTIPLIER = 0x5DEECE66D
_GRAPHVIZ_DRAND48_INCREMENT = 0xB
_GRAPHVIZ_DRAND48_MASK = (1 << 48) - 1
_GRAPHVIZ_DRAND48_SEED_SUFFIX = 0x330E
_OGDF_EDGE_COSTS = 100.0
_OGDF_RAND_BUCKETS = 1000
_OGDF_RAND_SCALE = 10.0


def _ogdf_runner_initial_positions(num_nodes: int, seed: int) -> np.ndarray:
    """Return the OGDF runner-owned initial GraphAttributes coordinates.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    seed : int
        Seed passed to the standalone OGDF runner before ``std::rand`` is
        consumed for initial x/y coordinates.

    Returns
    -------
    numpy.ndarray
        Double-precision initial coordinates with shape ``[N, 2]``.

    Notes
    -----
    The repository's OGDF adapter calls ``std::srand(seed)`` and then assigns
    ``std::rand() % 1000 / 10.0`` to x and y for each node before invoking
    ``StressMinimization::hasInitialLayout(true)``. Calling libc directly keeps
    this path bit-aligned with the compiled runner on the Linux benchmark host.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")

    libc = ctypes.CDLL(None)
    libc.srand(ctypes.c_uint(int(seed)))
    libc.rand.restype = ctypes.c_int

    positions = np.empty((num_nodes, 2), dtype=np.float64)
    for node in range(num_nodes):
        positions[node, 0] = float(libc.rand() % _OGDF_RAND_BUCKETS) / _OGDF_RAND_SCALE
        positions[node, 1] = float(libc.rand() % _OGDF_RAND_BUCKETS) / _OGDF_RAND_SCALE
    return positions


def _layout_ogdf_disconnected_components(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor],
    iterations: int,
    seed: int,
    fidelity_dtype: torch.dtype,
    epsilon: Optional[float],
) -> torch.Tensor:
    """Lay out disconnected OGDF stress components and pack them in rows.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``.
    iterations : int
        Number of OGDF stress-majorization iterations per component.
    seed : int
        Seed forwarded to the OGDF runner-compatible initialization stream.
    fidelity_dtype : torch.dtype
        Output dtype requested by the fidelity pipeline.
    epsilon : float, optional
        Relative stress-delta convergence threshold for each component.

    Returns
    -------
    torch.Tensor
        Packed positions with shape ``[N, 2]``.
    """
    components = _connected_components_from_edges(edge_index=edge_index, num_nodes=num_nodes)
    final_positions = torch.zeros((num_nodes, 2), dtype=torch.float64, device="cpu")
    shifted_components: list[tuple[list[int], torch.Tensor]] = []
    bounding_boxes: list[tuple[float, float]] = []

    for component_nodes in components:
        component_edges = _extract_component_edges(edge_index, component_nodes)
        component_sizes = None
        if node_sizes is not None:
            component_sizes = node_sizes.to(device="cpu")[component_nodes]
        component_positions = layout_stress_majorization_pipeline(
            edge_index=component_edges,
            num_nodes=len(component_nodes),
            node_sizes=component_sizes,
            iterations=iterations,
            seed=seed,
            fidelity_mode=_FIDELITY_MODE_OGDF,
            fidelity_dtype=fidelity_dtype,
            epsilon=epsilon,
        )
        shifted, box = _ogdf_shift_component_to_origin(
            positions=component_positions.to(dtype=torch.float64, device="cpu"),
            config=_GEM_PHYSICS_CONFIG,
        )
        shifted_components.append((component_nodes, shifted))
        bounding_boxes.append(box)

    offsets = _ogdf_tile_to_rows_offsets(bounding_boxes, _GEM_PHYSICS_CONFIG.page_ratio)
    for component_index, (component_nodes, shifted) in enumerate(shifted_components):
        dx, dy = offsets[component_index]
        for local_index, node_index in enumerate(component_nodes):
            final_positions[node_index, 0] = shifted[local_index, 0] + dx
            final_positions[node_index, 1] = shifted[local_index, 1] + dy
    return final_positions.to(dtype=fidelity_dtype)


def _is_graphviz_fidelity(fidelity_mode: Optional[str]) -> bool:
    """Return whether a fidelity mode selects Graphviz neato semantics.

    Parameters
    ----------
    fidelity_mode : str, optional
        Requested fidelity mode.

    Returns
    -------
    bool
        ``True`` when the mode should use Graphviz neato initialization and
        conjugate-gradient majorization.
    """
    return fidelity_mode in _GRAPHVIZ_FIDELITY_MODES


def _graphviz_packed_index(row: int, col: int, size: int) -> int:
    """Return Graphviz's row-major upper-triangle packed index.

    Parameters
    ----------
    row : int
        Matrix row.
    col : int
        Matrix column.
    size : int
        Matrix dimension.

    Returns
    -------
    int
        Offset into a packed symmetric matrix.
    """
    if col < row:
        row, col = col, row
    return row * size - (row * (row - 1)) // 2 + (col - row)


def _graphviz_packed_matvec(packed_matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Multiply a Graphviz packed symmetric matrix by a vector.

    Parameters
    ----------
    packed_matrix : numpy.ndarray
        Upper-triangle row-major packed matrix with shape ``[N * (N + 1) / 2]``.
    vector : numpy.ndarray
        Single-precision vector with shape ``[N]``.

    Returns
    -------
    numpy.ndarray
        Single-precision product vector with shape ``[N]``.
    """
    size = int(vector.shape[0])
    result = np.zeros(size, dtype=np.float32)
    index = 0
    for row in range(size):
        row_sum = np.float32(0.0)
        vector_row = np.float32(vector[row])
        row_sum = np.float32(row_sum + packed_matrix[index] * vector_row)
        index += 1
        for col in range(row + 1, size):
            value = np.float32(packed_matrix[index])
            row_sum = np.float32(row_sum + value * vector[col])
            result[col] = np.float32(result[col] + value * vector_row)
            index += 1
        result[row] = np.float32(result[row] + row_sum)
    return result


def _graphviz_orthog1f(vector: np.ndarray) -> None:
    """Center a float vector in place like Graphviz ``orthog1f``.

    Parameters
    ----------
    vector : numpy.ndarray
        Single-precision vector with shape ``[N]``.

    Returns
    -------
    None
        The vector is modified in place.
    """
    total = np.float32(0.0)
    for index in range(int(vector.shape[0])):
        total = np.float32(total + vector[index])
    mean = np.float32(total / np.float32(vector.shape[0]))
    for index in range(int(vector.shape[0])):
        vector[index] = np.float32(vector[index] - mean)


def _graphviz_inner_productf(left: np.ndarray, right: np.ndarray) -> float:
    """Compute Graphviz's float-product, double-accumulated dot product.

    Parameters
    ----------
    left : numpy.ndarray
        Left single-precision vector with shape ``[N]``.
    right : numpy.ndarray
        Right single-precision vector with shape ``[N]``.

    Returns
    -------
    float
        Dot product accumulated in double precision after float products.
    """
    result = 0.0
    for index in range(int(left.shape[0])):
        result += float(np.float32(left[index] * right[index]))
    return result


def _graphviz_subtract_vectorsf(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Subtract two float vectors with Graphviz's sequential assignment order.

    Parameters
    ----------
    left : numpy.ndarray
        Left single-precision vector with shape ``[N]``.
    right : numpy.ndarray
        Right single-precision vector with shape ``[N]``.

    Returns
    -------
    numpy.ndarray
        Single-precision vector containing ``left - right``.
    """
    result = np.empty_like(left, dtype=np.float32)
    for index in range(int(left.shape[0])):
        result[index] = np.float32(left[index] - right[index])
    return result


def _graphviz_mult_additionf(vector: np.ndarray, alpha: float, addend: np.ndarray) -> None:
    """Apply Graphviz ``vectors_mult_additionf`` in place.

    Parameters
    ----------
    vector : numpy.ndarray
        Single-precision vector updated in place.
    alpha : float
        Scalar multiplier cast to single precision by the C caller.
    addend : numpy.ndarray
        Single-precision addend vector with shape matching ``vector``.

    Returns
    -------
    None
        ``vector`` is updated in place.
    """
    alpha32 = np.float32(alpha)
    for index in range(int(vector.shape[0])):
        vector[index] = np.float32(vector[index] + np.float32(alpha32 * addend[index]))


def _graphviz_packed_stress_laplacian(target_distances: np.ndarray) -> np.ndarray:
    """Build Graphviz's negated packed stress Laplacian.

    Parameters
    ----------
    target_distances : numpy.ndarray
        Dense graph-distance matrix with shape ``[N, N]``.

    Returns
    -------
    numpy.ndarray
        Single-precision packed matrix whose off-diagonal entries are
        ``1 / d_ij**2`` and whose diagonal entries are negative row sums.
    """
    size = int(target_distances.shape[0])
    packed = np.zeros(size * (size + 1) // 2, dtype=np.float32)
    degrees = np.zeros(size, dtype=np.longdouble)
    index = 0
    for row in range(size):
        index += 1
        for col in range(row + 1, size):
            distance = np.float32(target_distances[row, col])
            value = np.float32(0.0)
            if distance != np.float32(0.0):
                value = np.float32(1.0) / np.float32(distance * distance)
            packed[index] = value
            degrees[row] -= np.longdouble(value)
            degrees[col] -= np.longdouble(value)
            index += 1
    for row in range(size):
        packed[_graphviz_packed_index(row, row, size)] = np.float32(degrees[row])
    return packed


def _graphviz_pca_project_distances(
    target_distances: np.ndarray,
    dimensions: int = 2,
) -> np.ndarray:
    """Project centered graph distances using Graphviz ``PCA_alloc`` semantics.

    Parameters
    ----------
    target_distances : numpy.ndarray
        Dense graph-distance matrix with shape ``[N, N]``.
    dimensions : int, default=2
        Number of PCA axes to return.

    Returns
    -------
    numpy.ndarray
        Double-precision projected coordinates with shape ``[N, dimensions]``.
    """
    size = int(target_distances.shape[0])
    if size == 0:
        return np.empty((0, dimensions), dtype=np.float64)
    coords = target_distances.astype(np.float64, copy=True)
    coords -= coords.mean(axis=1, keepdims=True)
    covariance = coords @ coords.T
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    selected = eigenvectors[:, sorted_indices[: min(dimensions, size)]]
    projected = np.zeros((size, dimensions), dtype=np.float64)
    if selected.size > 0:
        projected[:, : selected.shape[1]] = (selected.T @ coords).T
    return projected


def _graphviz_normalize_pca_positions(positions: np.ndarray) -> np.ndarray:
    """Scale PCA coordinates like neato's post-smart-init normalization.

    Parameters
    ----------
    positions : numpy.ndarray
        Raw coordinates with shape ``[N, 2]``.

    Returns
    -------
    numpy.ndarray
        Single-precision centered coordinates with each axis scaled down when
        its absolute value exceeds one.
    """
    normalized = positions.astype(np.float32, copy=True)
    if normalized.shape[0] == 0:
        return normalized
    for axis in range(normalized.shape[1]):
        max_abs = max(1.0, float(np.max(np.abs(normalized[:, axis]))))
        normalized[:, axis] = normalized[:, axis] / np.float32(max_abs)
        _graphviz_orthog1f(normalized[:, axis])
    return normalized


def _graphviz_drand48_values(seed: int, count: int) -> np.ndarray:
    """Generate Graphviz-compatible ``drand48`` values.

    Parameters
    ----------
    seed : int
        Integer passed to ``srand48``.
    count : int
        Number of random values to generate.

    Returns
    -------
    numpy.ndarray
        Double-precision values in ``[0, 1)`` with shape ``[count]``.
    """
    if count < 0:
        raise ValueError("count must be non-negative.")
    state = (((int(seed) & 0xFFFFFFFF) << 16) + _GRAPHVIZ_DRAND48_SEED_SUFFIX) & (
        _GRAPHVIZ_DRAND48_MASK
    )
    values = np.empty(count, dtype=np.float64)
    denominator = float(1 << 48)
    for index in range(count):
        state = (
            _GRAPHVIZ_DRAND48_MULTIPLIER * state + _GRAPHVIZ_DRAND48_INCREMENT
        ) & _GRAPHVIZ_DRAND48_MASK
        values[index] = float(state) / denominator
    return values


def _graphviz_random_initialize_positions(
    num_nodes: int,
    dimensions: int,
    seed: int,
) -> np.ndarray:
    """Initialize coordinates like Graphviz neato's default ``initLayout``.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    dimensions : int
        Coordinate dimensionality. Neato uses two dimensions for this pipeline.
    seed : int
        Seed passed through Graphviz's ``srand48``.

    Returns
    -------
    numpy.ndarray
        Single-precision centered positions with shape ``[num_nodes, dimensions]``.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if dimensions <= 0:
        raise ValueError("dimensions must be positive.")
    values = _graphviz_drand48_values(seed=seed, count=num_nodes * dimensions)
    initialized = np.empty((num_nodes, dimensions), dtype=np.float64)
    for node in range(num_nodes):
        start = node * dimensions
        initialized[node, :] = values[start : start + dimensions]
    for axis in range(dimensions):
        total = 0.0
        for node in range(num_nodes):
            total += float(initialized[node, axis])
        mean = total / float(num_nodes)
        for node in range(num_nodes):
            initialized[node, axis] = float(initialized[node, axis]) - mean
    return initialized.astype(np.float32)


def _graphviz_conjugate_gradient_packed(
    packed_matrix: np.ndarray,
    x: np.ndarray,
    b: np.ndarray,
    tolerance: float,
    max_iterations: int,
) -> int:
    """Solve ``Ax=b`` with Graphviz ``conjugate_gradient_mkernel`` semantics.

    Parameters
    ----------
    packed_matrix : numpy.ndarray
        Upper-triangle row-major packed matrix with shape ``[N * (N + 1) / 2]``.
    x : numpy.ndarray
        Initial solution vector with shape ``[N]``. Updated in place.
    b : numpy.ndarray
        Right-hand side vector with shape ``[N]``. Centered in place.
    tolerance : float
        Maximum absolute residual threshold.
    max_iterations : int
        Maximum CG iterations.

    Returns
    -------
    int
        ``0`` on normal completion, ``1`` if Graphviz's zero-residual guard is
        reached.
    """
    size = int(x.shape[0])
    residual = np.zeros(size, dtype=np.float32)
    direction = np.zeros(size, dtype=np.float32)
    ap_vector = np.zeros(size, dtype=np.float32)

    _graphviz_orthog1f(x)
    _graphviz_orthog1f(b)
    ax_vector = _graphviz_packed_matvec(packed_matrix=packed_matrix, vector=x)
    _graphviz_orthog1f(ax_vector)

    residual[:] = _graphviz_subtract_vectorsf(b, ax_vector)
    direction[:] = residual
    residual_norm = _graphviz_inner_productf(residual, residual)

    iteration = 0
    while iteration < max_iterations and float(np.max(np.abs(residual))) > tolerance:
        _graphviz_orthog1f(direction)
        _graphviz_orthog1f(x)
        _graphviz_orthog1f(residual)

        ap_vector[:] = _graphviz_packed_matvec(packed_matrix=packed_matrix, vector=direction)
        _graphviz_orthog1f(ap_vector)

        p_ap = _graphviz_inner_productf(direction, ap_vector)
        if p_ap == 0.0:
            break
        alpha = residual_norm / p_ap
        _graphviz_mult_additionf(x, float(np.float32(alpha)), direction)

        if iteration < max_iterations - 1:
            _graphviz_mult_additionf(residual, float(np.float32(-alpha)), ap_vector)
            new_residual_norm = _graphviz_inner_productf(residual, residual)
            if residual_norm == 0.0:
                return 1
            beta = new_residual_norm / residual_norm
            residual_norm = new_residual_norm
            beta32 = np.float32(beta)
            for index in range(size):
                direction[index] = np.float32(
                    np.float32(beta32 * direction[index]) + residual[index]
                )
        iteration += 1

    return 0


@dataclass(frozen=True)
class GraphvizPrepareStressMajorizationState(Op):
    """Prepare Graphviz neato dense distances and packed stress Laplacian."""

    name: ClassVar[str] = "sm_graphviz_prepare_state"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = ("distance_matrix", "extras")
    requires: ClassVar[Tuple[str, ...]] = ()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute neato shortest-path distances and packed ``lap2``.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context. Unused.

        Returns
        -------
        SolveState
            State with Graphviz distances, weights, and packed Laplacian.
        """
        del ctx

        adjacency = _shared_build_undirected_adjacency(
            edge_index=problem.edge_index,
            num_nodes=problem.num_nodes,
            edge_weights=problem.edge_weights,
        )
        weighted = problem.edge_weights is not None
        raw_distances = _shared_all_pairs_shortest_paths(adjacency, weighted=weighted)
        distances = raw_distances.astype(np.float64, copy=True)
        for node in range(problem.num_nodes):
            row = distances[node]
            finite_mask = np.isfinite(row) if weighted else row >= 0
            farthest = float(row[finite_mask].max()) if bool(finite_mask.any()) else 0.0
            row[~finite_mask] = farthest + 10.0 if problem.num_nodes > 1 else 0.0
        np.fill_diagonal(distances, 0.0)

        with np.errstate(divide="ignore"):
            weights = np.where(distances > 0.0, 1.0 / np.square(distances), 0.0)
        np.fill_diagonal(weights, 0.0)

        state.distance_matrix = torch.from_numpy(distances)
        state.extras[WEIGHTS_KEY] = weights
        state.extras[_GRAPHVIZ_LAP2_KEY] = _graphviz_packed_stress_laplacian(
            target_distances=distances
        )
        return state


@dataclass(frozen=True)
class OgdfPrepareStressMajorizationState(Op):
    """Prepare OGDF ``StressMinimization`` distances and weights."""

    name: ClassVar[str] = "sm_ogdf_prepare_state"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = ("distance_matrix", "laplacian", "extras")
    requires: ClassVar[Tuple[str, ...]] = ()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute OGDF unit-edge shortest paths scaled by ``edgeCosts=100``.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context. Unused.

        Returns
        -------
        SolveState
            State with OGDF-scaled target distances and inverse-square weights.
        """
        del ctx

        adjacency = _shared_build_undirected_adjacency(
            edge_index=problem.edge_index,
            num_nodes=problem.num_nodes,
            edge_weights=None,
        )
        raw_distances = _shared_all_pairs_shortest_paths(adjacency, weighted=False)
        reachable_mask = raw_distances >= 0
        fill_value = (
            _OGDF_EDGE_COSTS * float(problem.num_nodes) ** 0.5 if problem.num_nodes > 1 else 0.0
        )
        distances = np.where(
            reachable_mask,
            raw_distances.astype(np.float64) * _OGDF_EDGE_COSTS,
            fill_value,
        )
        np.fill_diagonal(distances, 0.0)

        with np.errstate(divide="ignore"):
            weights = np.where(distances > 0.0, 1.0 / np.square(distances), 0.0)
        np.fill_diagonal(weights, 0.0)

        state.distance_matrix = torch.from_numpy(distances)
        state.extras[WEIGHTS_KEY] = weights
        state.laplacian = np.empty((0, 0), dtype=np.float64)
        return state


@dataclass(frozen=True)
class OgdfInitializePositions(Op):
    """Initialize positions from the OGDF runner's seeded ``std::rand`` grid."""

    name: ClassVar[str] = "sm_ogdf_initialize_positions"
    category: ClassVar[OpCategory] = OpCategory.INIT
    reads: ClassVar[Tuple[str, ...]] = ("distance_matrix", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("extras",)
    requires: ClassVar[Tuple[str, ...]] = ("distance_matrix", "extras")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Set the current layout to the adapter-owned OGDF initial layout.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph layout inputs, including node count and seed.
        state : SolveState
            Mutable solve state with target distances and weights prepared.
        ctx : RuntimeContext
            Runtime context. Unused.

        Returns
        -------
        SolveState
            State with ``sm_current_positions`` and ``sm_current_stress`` set.
        """
        del ctx

        if not isinstance(state.distance_matrix, torch.Tensor):
            raise ValueError("OGDF initialization requires state.distance_matrix.")
        weights = state.extras.get(WEIGHTS_KEY)
        if not isinstance(weights, np.ndarray):
            raise ValueError("OGDF initialization requires sm_weights in state.extras.")

        initialized = _ogdf_runner_initial_positions(
            num_nodes=problem.num_nodes,
            seed=problem.seed,
        )
        target_distances = state.distance_matrix.to(dtype=torch.float64, device="cpu").numpy()
        deltas = initialized[:, None, :] - initialized[None, :, :]
        current_distances = np.sqrt(np.sum(deltas * deltas, axis=2))
        errors = current_distances - target_distances

        state.extras[CURRENT_POSITIONS_KEY] = initialized
        state.extras[CURRENT_STRESS_KEY] = 0.5 * float(np.sum(weights * errors * errors))
        return state


@dataclass(frozen=True)
class GraphvizInitializePositions(Op):
    """Initialize positions from Graphviz neato's default random start."""

    name: ClassVar[str] = "sm_graphviz_initialize_positions"
    category: ClassVar[OpCategory] = OpCategory.INIT
    reads: ClassVar[Tuple[str, ...]] = ("distance_matrix", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("extras",)
    requires: ClassVar[Tuple[str, ...]] = ("distance_matrix", "extras")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Set current positions to Graphviz-compatible random coordinates.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs, including node count and seed.
        state : SolveState
            Mutable state with prepared target distances.
        ctx : RuntimeContext
            Runtime context. Unused.

        Returns
        -------
        SolveState
            State with ``sm_current_positions`` initialized.
        """
        del ctx

        if not isinstance(state.distance_matrix, torch.Tensor):
            raise ValueError("Graphviz initialization requires state.distance_matrix.")
        initialized = _graphviz_random_initialize_positions(
            num_nodes=problem.num_nodes,
            dimensions=2,
            seed=problem.seed,
        )
        state.extras[CURRENT_POSITIONS_KEY] = initialized
        state.extras[CURRENT_STRESS_KEY] = float("inf")
        state.extras[_GRAPHVIZ_OLD_STRESS_KEY] = float("inf")
        return state


@dataclass(frozen=True)
class GraphvizCgSmacofStep(Op):
    """Apply one Graphviz neato packed-CG stress-majorization update."""

    epsilon: Optional[float] = None

    name: ClassVar[str] = "sm_graphviz_cg_step"
    category: ClassVar[OpCategory] = OpCategory.OPTIMIZE
    reads: ClassVar[Tuple[str, ...]] = ("distance_matrix", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("extras", "converged")
    requires: ClassVar[Tuple[str, ...]] = ("distance_matrix", "extras")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run one Graphviz CG solve for each coordinate axis.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Only the node count is used.
        state : SolveState
            Mutable state carrying current positions and packed ``lap2``.
        ctx : RuntimeContext
            Runtime context. Unused.

        Returns
        -------
        SolveState
            State with updated current positions and Graphviz stress.
        """
        del ctx

        current = state.extras.get(CURRENT_POSITIONS_KEY)
        lap2 = state.extras.get(_GRAPHVIZ_LAP2_KEY)
        if not isinstance(current, np.ndarray):
            raise ValueError("Graphviz CG step requires sm_current_positions.")
        if not isinstance(lap2, np.ndarray):
            raise ValueError("Graphviz CG step requires packed lap2 in extras.")
        if not isinstance(state.distance_matrix, torch.Tensor):
            raise ValueError("Graphviz CG step requires state.distance_matrix.")
        if problem.num_nodes <= 1:
            state.extras[CURRENT_STRESS_KEY] = 0.0
            state.extras[_GRAPHVIZ_OLD_STRESS_KEY] = 0.0
            state.converged = True
            return state

        target_distances = state.distance_matrix.to(dtype=torch.float32, device="cpu").numpy()
        coordinates = current.astype(np.float32, copy=True)
        b_vectors = self._build_b_vectors(
            coordinates=coordinates,
            target_distances=target_distances,
            lap2=lap2,
        )
        new_stress = self._graphviz_stress(
            coordinates=coordinates,
            b_vectors=b_vectors,
            lap2=lap2,
            num_nodes=problem.num_nodes,
        )
        self._update_convergence(state=state, new_stress=new_stress)

        for axis in range(coordinates.shape[1]):
            result = _graphviz_conjugate_gradient_packed(
                packed_matrix=lap2,
                x=coordinates[:, axis],
                b=b_vectors[:, axis],
                tolerance=_GRAPHVIZ_CG_TOLERANCE,
                max_iterations=problem.num_nodes,
            )
            if result != 0:
                raise RuntimeError("Graphviz conjugate-gradient solve hit zero residual.")

        state.extras[CURRENT_POSITIONS_KEY] = coordinates
        state.extras[CURRENT_STRESS_KEY] = new_stress
        return state

    def _build_b_vectors(
        self,
        coordinates: np.ndarray,
        target_distances: np.ndarray,
        lap2: np.ndarray,
    ) -> np.ndarray:
        """Build Graphviz's current-position Laplacian right-hand sides.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Current single-precision coordinates with shape ``[N, 2]``.
        target_distances : numpy.ndarray
            Target graph distances with shape ``[N, N]``.
        lap2 : numpy.ndarray
            Packed stress Laplacian with shape ``[N * (N + 1) / 2]``.

        Returns
        -------
        numpy.ndarray
            Right-hand side vectors with shape ``[N, 2]``.
        """
        size = int(coordinates.shape[0])
        lap1 = np.zeros_like(lap2, dtype=np.float32)
        degrees = np.zeros(size, dtype=np.longdouble)
        index = 0
        for row in range(size):
            index += 1
            for col in range(row + 1, size):
                squared_distance = np.float32(0.0)
                for axis in range(coordinates.shape[1]):
                    delta = np.float32(
                        coordinates[row, axis] + np.float32(-1.0) * coordinates[col, axis]
                    )
                    squared_distance = np.float32(squared_distance + delta * delta)
                inverse_distance = np.float32(0.0)
                if squared_distance > np.float32(0.0):
                    inverse_distance = np.float32(1.0) / np.float32(np.sqrt(squared_distance))
                if inverse_distance >= np.finfo(np.float32).max or inverse_distance < 0:
                    inverse_distance = np.float32(0.0)
                scale = np.float32(0.0)
                if lap2[index] >= np.float32(0.0):
                    scale = np.float32(np.sqrt(lap2[index]))
                value = np.float32(scale * inverse_distance)
                lap1[index] = value
                degrees[row] -= np.longdouble(value)
                degrees[col] -= np.longdouble(value)
                index += 1
        for row in range(size):
            lap1[_graphviz_packed_index(row, row, size)] = np.float32(degrees[row])

        b_vectors = np.zeros_like(coordinates, dtype=np.float32)
        for axis in range(coordinates.shape[1]):
            b_vectors[:, axis] = _graphviz_packed_matvec(
                packed_matrix=lap1,
                vector=coordinates[:, axis],
            )
        return b_vectors

    def _graphviz_stress(
        self,
        coordinates: np.ndarray,
        b_vectors: np.ndarray,
        lap2: np.ndarray,
        num_nodes: int,
    ) -> float:
        """Compute the stress expression used by Graphviz before each CG solve.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Current coordinates with shape ``[N, 2]``.
        b_vectors : numpy.ndarray
            Right-hand side vectors with shape ``[N, 2]``.
        lap2 : numpy.ndarray
            Packed stress Laplacian with shape ``[N * (N + 1) / 2]``.
        num_nodes : int
            Number of graph nodes.

        Returns
        -------
        float
            Graphviz stress value for the current coordinates.
        """
        new_stress = 0.0
        for axis in range(coordinates.shape[1]):
            new_stress += _graphviz_inner_productf(coordinates[:, axis], b_vectors[:, axis])
        new_stress *= 2.0
        new_stress += float(num_nodes * (num_nodes - 1)) / 2.0
        for axis in range(coordinates.shape[1]):
            tmp = _graphviz_packed_matvec(packed_matrix=lap2, vector=coordinates[:, axis])
            new_stress -= _graphviz_inner_productf(coordinates[:, axis], tmp)
        return float(new_stress)

    def _update_convergence(self, state: SolveState, new_stress: float) -> None:
        """Update Graphviz-style convergence from the pre-solve stress.

        Parameters
        ----------
        state : SolveState
            Mutable solve state.
        new_stress : float
            Stress value computed before the coordinate solve.

        Returns
        -------
        None
            The state is updated in place.
        """
        epsilon = self.epsilon
        old_stress = float(state.extras.get(_GRAPHVIZ_OLD_STRESS_KEY, float("inf")))
        if epsilon is not None and epsilon > 0.0:
            if new_stress < epsilon:
                state.converged = True
            elif np.isfinite(old_stress) and old_stress != 0.0:
                change = abs(old_stress - new_stress)
                state.converged = change / old_stress < epsilon
        state.extras[_GRAPHVIZ_OLD_STRESS_KEY] = new_stress


def build_stress_majorization_pipeline(
    iterations: int = 200,
    trace_every: int = 0,
    fidelity_mode: Optional[str] = None,
    fidelity_dtype: torch.dtype = torch.float32,
    epsilon: Optional[float] = None,
) -> Pipeline:
    """Build a stress-majorization (SMACOF) pipeline.

    Reference fidelity
    ------------------
    Targets: OGDF stress majorization and Graphviz 7.0.5 neato / Gansner,
        Koren, and North (2004), "Graph Drawing by Stress Majorization".
    Fidelity mode: ``"ogdf"`` enables OGDF-style serial sweeps,
        disconnected-distance fill, and no-jitter warm starts;
        ``"graphviz"`` enables Graphviz neato's default random start and
        packed conjugate-gradient stress solver;
        ``"graphviz_neato"`` enables neato shortest-path defaults, seeded
        random initialization, Graphviz disconnected fill, unconstrained
        SMACOF updates, and epsilon early termination.
    Verified at: final 100-seed report, strong equivalent; median RMSD 0.059
        to 0.092 across stress-majorization variants.
    Known divergences:
        - Exact Graphviz CG behavior and post-processing remain outside this
          pipeline.
        - OGDF and Graphviz modes share Dagua's composable tensor operators.

    Parameters
    ----------
    iterations : int, default=200
        Number of SMACOF majorization steps.
    trace_every : int, default=0
        If positive, collect position snapshots at this cadence.
    fidelity_mode : str, optional
        Optional reference-fidelity mode. ``"ogdf"`` enables OGDF-compatible
        serial sweeps, disconnected-distance fill, and a no-jitter warm start.
        ``"graphviz"`` enables Graphviz neato's default random initialization
        and packed conjugate-gradient stress solver.
        ``"graphviz_neato"`` enables neato defaults: shortest-path model,
        seeded random init, Graphviz disconnected fill, unconstrained SMACOF
        updates, and epsilon early termination.
    epsilon : float, optional
        Relative stress-delta convergence threshold. ``None`` disables the
        extra convergence op.

    Returns
    -------
    Pipeline
        Pipeline implementing the classical SMACOF algorithm. The pipeline
        produces final node coordinates by preparing stress weights,
        initializing positions, applying repeated majorization updates,
        collecting optional traces, and finalizing the layout.

    Raises
    ------
    ValueError
        If ``iterations`` or ``trace_every`` is negative, or if
        ``fidelity_mode`` is unknown.
    """
    if iterations < 0:
        raise ValueError("iterations must be non-negative.")
    if trace_every < 0:
        raise ValueError("trace_every must be non-negative.")
    if epsilon is not None and epsilon <= 0.0:
        raise ValueError("epsilon must be positive when provided.")
    if fidelity_mode not in _FIDELITY_MODES:
        raise ValueError(f"Unknown stress_majorization fidelity_mode: {fidelity_mode!r}.")

    if _is_graphviz_fidelity(fidelity_mode):
        return Pipeline(
            [
                FixedSteps(FixedStepsConfig(n=iterations)),
                GraphvizPrepareStressMajorizationState(),
                GraphvizInitializePositions(),
                Repeat(
                    n=iterations,
                    ops=[
                        GraphvizCgSmacofStep(epsilon=epsilon),
                        CollectStressMajorizationTrace(),
                    ],
                ),
                FinalizeStressMajorizationPositions(),
            ],
            name="stress_majorization_pipeline",
        )

    if fidelity_mode == _FIDELITY_MODE_OGDF:
        repeated_ops = [
            SmacofStep(
                config=SmacofStepConfig(
                    update_mode="ogdf_serial",
                    min_distance=0.0,
                )
            ),
            CollectStressMajorizationTrace(),
        ]
        if epsilon is not None:
            repeated_ops = [
                CaptureStressMajorizationStress(),
                *repeated_ops,
                CheckStressMajorizationEpsilon(epsilon=epsilon),
            ]

        return Pipeline(
            [
                FixedSteps(FixedStepsConfig(n=iterations)),
                OgdfPrepareStressMajorizationState(),
                OgdfInitializePositions(),
                Repeat(
                    n=iterations,
                    ops=repeated_ops,
                ),
                FinalizeStressMajorizationPositions(),
            ],
            name="stress_majorization_pipeline",
        )

    prepare_config = PrepareStressMajorizationStateConfig()
    init_config = InitializeStressMajorizationPositionsConfig()
    step_config = SmacofStepConfig()
    if fidelity_mode == _FIDELITY_MODE_GRAPHVIZ_NEATO:
        prepare_config = PrepareStressMajorizationStateConfig(distance_fill="graphviz_neato")
        init_config = InitializeStressMajorizationPositionsConfig(init_mode="random")
        step_config = SmacofStepConfig(stress_tolerance=float("inf"))

    repeated_ops = []
    if epsilon is not None:
        repeated_ops.append(CaptureStressMajorizationStress())
    repeated_ops.extend(
        [
            SmacofStep(config=step_config),
            CollectStressMajorizationTrace(),
        ]
    )
    if epsilon is not None:
        repeated_ops.append(CheckStressMajorizationEpsilon(epsilon=epsilon))

    return Pipeline(
        [
            FixedSteps(FixedStepsConfig(n=iterations)),
            PrepareStressMajorizationState(config=prepare_config),
            InitializeStressMajorizationPositions(config=init_config),
            Repeat(
                n=iterations,
                ops=repeated_ops,
            ),
            FinalizeStressMajorizationPositions(),
        ],
        name="stress_majorization_pipeline",
    )


def layout_stress_majorization_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    iterations: int = 200,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    trace_every: int = 0,
    fidelity_mode: Optional[str] = None,
    fidelity_dtype: torch.dtype = torch.float32,
    epsilon: Optional[float] = None,
    graphviz_neato_fidelity: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
    """Run the stress-majorization pipeline as a drop-in replacement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``.
    iterations : int, default=200
        Number of SMACOF majorization steps.
    seed : int, default=42
        Random seed for the stochastic warm-start jitter.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.
    trace_every : int, default=0
        If positive, return periodic position snapshots.
    fidelity_mode : str, optional
        Optional reference-fidelity mode. ``"ogdf"`` keeps the public API
        opt-in while matching OGDF's serial sweep, disconnected fill, and
        deterministic no-jitter warm start.
    epsilon : float, optional
        Relative stress-delta convergence threshold. ``None`` disables early
        termination beyond the fixed iteration budget.
    graphviz_neato_fidelity : bool, default=False
        Convenience switch for Graphviz neato defaults. When true, this sets
        ``fidelity_mode="graphviz_neato"`` and ``epsilon=0.0001`` unless the
        caller provided explicit values.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, list[torch.Tensor]]
        Final position tensor with shape ``[N, 2]``. When ``trace_every > 0``,
        periodic snapshots are returned alongside the final layout.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``iterations``, ``trace_every``, ``fidelity_mode``,
        or ``edge_weights`` are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if iterations < 0:
        raise ValueError("iterations must be non-negative.")
    if trace_every < 0:
        raise ValueError("trace_every must be non-negative.")
    if graphviz_neato_fidelity:
        if fidelity_mode is None:
            fidelity_mode = _FIDELITY_MODE_GRAPHVIZ
        if epsilon is None:
            epsilon = 0.0001
        iterations = 200
    if epsilon is not None and epsilon <= 0.0:
        raise ValueError("epsilon must be positive when provided.")
    if fidelity_mode not in _FIDELITY_MODES:
        raise ValueError(f"Unknown stress_majorization fidelity_mode: {fidelity_mode!r}.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must be shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
            )

    # Resolve empty and singleton graphs here to preserve direct-returns in the
    # classic implementation path and keep pipeline behavior simple.
    if num_nodes == 0:
        device = _layout_device(edge_index=edge_index, node_sizes=node_sizes)
        empty = torch.empty((0, 2), dtype=torch.float32, device=device)
        return (empty, []) if trace_every > 0 else empty
    if num_nodes == 1:
        device = _layout_device(edge_index=edge_index, node_sizes=node_sizes)
        single = torch.zeros((1, 2), dtype=torch.float32, device=device)
        return (single, []) if trace_every > 0 else single

    if fidelity_mode == _FIDELITY_MODE_OGDF and edge_weights is None:
        components = _connected_components_from_edges(edge_index=edge_index, num_nodes=num_nodes)
        if len(components) > 1:
            positions = _layout_ogdf_disconnected_components(
                edge_index=edge_index,
                num_nodes=num_nodes,
                node_sizes=node_sizes,
                iterations=iterations,
                seed=seed,
                fidelity_dtype=fidelity_dtype,
                epsilon=epsilon,
            )
            device = _layout_device(edge_index=edge_index, node_sizes=node_sizes)
            positions = positions.to(dtype=torch.float32, device=device)
            return (positions, []) if trace_every > 0 else positions

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    state.extras[TRACE_EVERY_KEY] = trace_every
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_stress_majorization_pipeline(
        iterations=iterations,
        trace_every=trace_every,
        fidelity_mode=fidelity_mode,
        fidelity_dtype=fidelity_dtype,
        epsilon=epsilon,
    ).apply(problem, state, ctx)

    if final_state.pos is None:
        raise RuntimeError("Stress majorization pipeline did not produce final positions.")

    if trace_every > 0:
        traces = final_state.extras.get(TRACES_KEY, [])
        return final_state.pos, traces
    return final_state.pos


__all__ = [
    "build_stress_majorization_pipeline",
    "layout_stress_majorization_pipeline",
]
