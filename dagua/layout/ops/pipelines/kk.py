"""Kamada-Kawai expressed as a composable ops pipeline."""

from __future__ import annotations

from typing import Any, ClassVar, List, Optional, Tuple, Union

import numpy as np
import torch

from dagua.layout.ops.graph_utils import (
    bfs_distances,
    build_directed_adjacency,
    dijkstra_distances,
)
from dagua.layout.ops.graph_utils import (
    layout_device as _layout_device,
)

UNREACHABLE_DISTANCE = 1.0e6
DISTANCE_EPSILON = 1.0e-3
CENTERING_WEIGHT = 1.0e-3


def _shortest_path_distance_matrix(
    adjacency: list[list[tuple[int, float]]],
    weighted: bool,
) -> np.ndarray:
    """Compute directed all-pairs shortest-path distances.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Directed adjacency list.
    weighted : bool
        Whether to compute weighted shortest paths with Dijkstra.

    Returns
    -------
    numpy.ndarray
        Distance matrix with shape ``[N, N]`` and ``1e6`` for unreachable pairs.
    """
    num_nodes = len(adjacency)
    distances = np.full((num_nodes, num_nodes), UNREACHABLE_DISTANCE, dtype=np.float64)
    for source in range(num_nodes):
        if weighted:
            source_distances = dijkstra_distances(adjacency, source)
            source_distances[np.isinf(source_distances)] = UNREACHABLE_DISTANCE
            distances[source] = source_distances
            continue

        source_distances = bfs_distances(adjacency, source).astype(np.float64)
        source_distances[source_distances < 0] = UNREACHABLE_DISTANCE
        distances[source] = source_distances
    return distances


def _rescale_layout(positions: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Center and scale coordinates like ``networkx.rescale_layout``.

    Parameters
    ----------
    positions : numpy.ndarray
        Position array with shape ``[N, 2]``.
    scale : float, default=1.0
        Target half-width after rescaling.

    Returns
    -------
    numpy.ndarray
        Rescaled coordinates.
    """
    positions = positions.copy()
    positions -= positions.mean(axis=0)
    limit = np.abs(positions).max()
    if limit > 0:
        positions *= scale / limit
    return positions


def _circular_layout(num_nodes: int) -> np.ndarray:
    """Create the exact 2D circular initialization used by NetworkX.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.

    Returns
    -------
    numpy.ndarray
        Circular coordinates with shape ``[N, 2]``.
    """
    if num_nodes == 0:
        return np.empty((0, 2), dtype=np.float64)
    if num_nodes == 1:
        return np.zeros((1, 2), dtype=np.float64)

    theta = np.linspace(0, 1, num_nodes + 1)[:-1] * (2.0 * np.pi)
    theta = theta.astype(np.float32)
    positions = np.column_stack((np.cos(theta), np.sin(theta)))
    return _rescale_layout(positions.astype(np.float64, copy=False), scale=1.0)


def _kamada_kawai_costfn(
    pos_vec: np.ndarray,
    np_module: Any,
    invdist: np.ndarray,
    meanweight: float,
    dim: int,
) -> tuple[float, np.ndarray]:
    """Compute the exact NetworkX Kamada-Kawai objective and gradient.

    Parameters
    ----------
    pos_vec : numpy.ndarray
        Flattened position vector with shape ``[N * dim]``.
    np_module : Any
        NumPy module passed through to mirror the NetworkX helper signature.
    invdist : numpy.ndarray
        Inverse preferred-distance matrix with shape ``[N, N]``.
    meanweight : float
        Weight of the origin-centering penalty.
    dim : int
        Layout dimension.

    Returns
    -------
    tuple[float, numpy.ndarray]
        Objective value and flattened analytic gradient.
    """
    num_nodes = invdist.shape[0]
    pos_arr = pos_vec.reshape((num_nodes, dim))

    delta = pos_arr[:, np_module.newaxis, :] - pos_arr[np_module.newaxis, :, :]
    node_separation = np_module.linalg.norm(delta, axis=-1)
    direction = np_module.einsum(
        "ijk,ij->ijk",
        delta,
        1.0 / (node_separation + np_module.eye(num_nodes) * DISTANCE_EPSILON),
    )

    offset = (node_separation * invdist) - 1.0
    offset[np_module.diag_indices(num_nodes)] = 0.0

    cost = 0.5 * float(np_module.sum(offset**2))
    gradient = np_module.einsum(
        "ij,ij,ijk->ik",
        invdist,
        offset,
        direction,
    ) - np_module.einsum(
        "ij,ij,ijk->jk",
        invdist,
        offset,
        direction,
    )

    sum_positions = np_module.sum(pos_arr, axis=0)
    cost += 0.5 * meanweight * float(np_module.sum(sum_positions**2))
    gradient += meanweight * sum_positions
    return cost, gradient.ravel()


def _solve_kamada_kawai(
    distance_matrix: np.ndarray,
    initial_positions: np.ndarray,
    steps: Optional[int],
    trace_every: int,
) -> tuple[np.ndarray, list[torch.Tensor]]:
    """Run the exact SciPy L-BFGS-B solver used by NetworkX KK.

    Parameters
    ----------
    distance_matrix : numpy.ndarray
        Preferred graph distances with shape ``[N, N]``.
    initial_positions : numpy.ndarray
        Initial coordinates with shape ``[N, 2]``.
    steps : int, optional
        Maximum optimization iterations. ``None`` or ``0`` mirrors the
        uncapped NetworkX behavior by leaving ``maxiter`` unset.
    trace_every : int
        Callback snapshot cadence.

    Returns
    -------
    tuple[numpy.ndarray, list[torch.Tensor]]
        Final coordinates and optional trace snapshots.
    """
    try:
        import scipy as sp
    except ImportError as error:
        raise ImportError("layout_kk requires scipy to match NetworkX exactly.") from error

    inverse_distances = 1.0 / (
        distance_matrix + np.eye(distance_matrix.shape[0], dtype=np.float64) * DISTANCE_EPSILON
    )
    traces: list[torch.Tensor] = []
    iteration = 0

    def _callback(pos_vec: np.ndarray) -> None:
        """Collect optimizer traces without affecting the solve."""
        nonlocal iteration

        iteration += 1
        if trace_every > 0 and iteration % trace_every == 0:
            snapshot = pos_vec.reshape((-1, 2)).copy()
            traces.append(torch.from_numpy(snapshot).to(dtype=torch.float32))

    minimize_kwargs: dict[str, Any] = {
        "method": "L-BFGS-B",
        "args": (np, inverse_distances, CENTERING_WEIGHT, 2),
        "jac": True,
        "callback": _callback,
    }
    if steps not in {None, 0}:
        minimize_kwargs["options"] = {"maxiter": steps}

    optresult = sp.optimize.minimize(
        _kamada_kawai_costfn,
        initial_positions.ravel(),
        **minimize_kwargs,
    )
    return optresult.x.reshape((-1, 2)), traces


from dagua.layout.ops.base import Op, Pipeline  # noqa: E402
from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig  # noqa: E402
from dagua.layout.ops.state import (  # noqa: E402
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)
from dagua.layout.ops.taxonomy import OpCategory  # noqa: E402

_INITIAL_POS_KEY = "kk_initial_pos"
_DISTANCE_MATRIX_KEY = "kk_distance_matrix"
_SOLVED_POSITIONS_KEY = "kk_solved_positions"
_TRACE_KEY = "kk_traces"
_SHORT_CIRCUIT_KEY = "kk_short_circuit"


class _PrepareKKState(Op):
    """Build the exact classic KK distance matrix and initialization."""

    name: ClassVar[str] = "kk_prepare_state"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("pos", "extras")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Populate shortest-path targets and initial positions for KK.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State primed for the one-shot SciPy solve.
        """
        del ctx

        if problem.num_nodes == 0:
            state.extras[_SHORT_CIRCUIT_KEY] = True
            state.extras[_TRACE_KEY] = []
            state.pos = torch.empty((0, 2), dtype=torch.float32)
            return state
        if problem.num_nodes == 1:
            state.extras[_SHORT_CIRCUIT_KEY] = True
            state.extras[_TRACE_KEY] = []
            state.pos = torch.zeros((1, 2), dtype=torch.float32)
            return state

        adjacency = build_directed_adjacency(
            edge_index=problem.edge_index,
            num_nodes=problem.num_nodes,
            edge_weights=problem.edge_weights,
        )
        distance_matrix = _shortest_path_distance_matrix(
            adjacency,
            weighted=problem.edge_weights is not None,
        )
        initial_pos = state.extras.get(_INITIAL_POS_KEY)
        if isinstance(initial_pos, torch.Tensor):
            initial_positions = initial_pos.detach().cpu().numpy().astype(np.float64)
        else:
            initial_positions = _circular_layout(num_nodes=problem.num_nodes)

        state.extras[_SHORT_CIRCUIT_KEY] = False
        state.extras[_DISTANCE_MATRIX_KEY] = distance_matrix
        state.extras[_INITIAL_POS_KEY] = initial_positions
        return state


class _SolveKK(Op):
    """Run the exact SciPy L-BFGS-B Kamada-Kawai solve."""

    name: ClassVar[str] = "kk_solve"
    category: ClassVar[OpCategory] = OpCategory.OPTIMIZE
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("pos", "extras")

    def __init__(self, *, steps: Optional[int], trace_every: int) -> None:
        """Store the immutable solve controls.

        Parameters
        ----------
        steps : int, optional
            Maximum optimization iterations. ``None`` or ``0`` leaves
            SciPy's ``maxiter`` unset to match classic KK.
        trace_every : int
            Callback snapshot cadence.

        Returns
        -------
        None
            This constructor stores configuration only.
        """
        self.steps = steps
        self.trace_every = int(trace_every)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Execute the one-shot SciPy optimizer and capture optional traces.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with solved positions and optional raw traces stored.
        """
        del problem, ctx

        if bool(state.extras.get(_SHORT_CIRCUIT_KEY, False)):
            return state

        distance_matrix = state.extras[_DISTANCE_MATRIX_KEY]
        initial_positions = state.extras[_INITIAL_POS_KEY]
        solved_positions, traces = _solve_kamada_kawai(
            distance_matrix=distance_matrix,
            initial_positions=initial_positions,
            steps=self.steps,
            trace_every=self.trace_every,
        )
        state.extras[_SOLVED_POSITIONS_KEY] = solved_positions
        state.extras[_TRACE_KEY] = traces
        state.pos = torch.from_numpy(solved_positions)
        return state


class _FinalizeKKPositions(Op):
    """Apply the classic KK rescale and output-device cast."""

    name: ClassVar[str] = "kk_finalize_positions"
    category: ClassVar[OpCategory] = OpCategory.POSTPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("pos", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("pos", "extras")
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Rescale solved positions and move snapshots to the classic device.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state containing solved positions.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with final output positions and trace snapshots.

        Raises
        ------
        ValueError
            If ``state.pos`` is missing.
        """
        del ctx

        if state.pos is None:
            raise ValueError("_FinalizeKKPositions requires state.pos to be set.")

        output_device = _layout_device(
            edge_index=problem.edge_index,
            node_sizes=problem.node_sizes,
        )
        traces = state.extras.get(_TRACE_KEY, [])
        state.extras[_TRACE_KEY] = [trace.to(device=output_device) for trace in traces]

        if bool(state.extras.get(_SHORT_CIRCUIT_KEY, False)):
            state.pos = state.pos.to(dtype=torch.float32, device=output_device)
            return state

        solved_positions = state.extras[_SOLVED_POSITIONS_KEY]
        state.pos = torch.from_numpy(_rescale_layout(solved_positions, scale=1.0)).to(
            dtype=torch.float32,
            device=output_device,
        )
        return state


def build_kk_pipeline(steps: Optional[int] = None, trace_every: int = 0) -> Pipeline:
    """Build a KK pipeline that is bit-identical to classic ``layout_kk``.

    Parameters
    ----------
    steps : int, optional
        Maximum L-BFGS-B iterations. ``None`` or ``0`` leaves ``maxiter``
        unset to match classic KK.
    trace_every : int, default=0
        Callback snapshot cadence.

    Returns
    -------
    Pipeline
        Pipeline that reproduces classic KK exactly.

    Raises
    ------
    ValueError
        If ``steps`` or ``trace_every`` are invalid.
    """
    if steps is not None and steps < 0:
        raise ValueError("steps must be non-negative.")
    if trace_every < 0:
        raise ValueError("trace_every must be non-negative.")

    return Pipeline(
        [
            FixedSteps(FixedStepsConfig(n=0 if steps is None else steps)),
            _PrepareKKState(),
            _SolveKK(steps=steps, trace_every=trace_every),
            _FinalizeKKPositions(),
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
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
    """Run the KK pipeline as a drop-in replacement for classic ``layout_kk``.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Unused, accepted for interface compatibility.
    steps : int, optional
        Maximum L-BFGS-B iterations. ``None`` or ``0`` leaves ``maxiter``
        unset to match classic KK.
    seed : int, default=42
        Accepted for interface compatibility. The 2D classic KK path uses
        deterministic circular initialization and does not consume a seed.
    trace_every : int, default=0
        Callback snapshot cadence.
    solver : {"auto", "newton", "adam"}, default="auto"
        Retained for interface compatibility. All accepted values resolve to
        the classic SciPy L-BFGS-B path.
    pos : torch.Tensor, optional
        Initial positions with shape ``[N, 2]``. When provided, replaces the
        circular initialization.
    edge_weights : torch.Tensor, optional
        Optional per-edge weights with shape ``[E]``.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, list[torch.Tensor]]
        Final positions with shape ``[N, 2]``. When tracing is enabled,
        returns ``(positions, traces)``.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``steps``, ``trace_every``, ``solver``, or ``pos``
        are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    _ = seed

    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if steps is not None and steps < 0:
        raise ValueError("steps must be non-negative.")
    if trace_every < 0:
        raise ValueError("trace_every must be non-negative.")
    if solver not in {"auto", "newton", "adam"}:
        raise ValueError("solver must be one of 'auto', 'newton', or 'adam'.")

    output_device = _layout_device(edge_index=edge_index, node_sizes=node_sizes)
    if num_nodes == 0:
        empty = torch.empty((0, 2), dtype=torch.float32, device=output_device)
        return (empty, []) if trace_every > 0 else empty
    if num_nodes == 1:
        single = torch.zeros((1, 2), dtype=torch.float32, device=output_device)
        return (single, []) if trace_every > 0 else single
    if pos is not None and pos.shape != (num_nodes, 2):
        raise ValueError(f"pos must have shape ({num_nodes}, 2), got {tuple(pos.shape)}")

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
    )
    state = SolveState()
    if pos is not None:
        state.extras[_INITIAL_POS_KEY] = pos
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_kk_pipeline(steps=steps, trace_every=trace_every).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("KK pipeline did not produce final positions.")

    if trace_every > 0:
        traces = final_state.extras.get(_TRACE_KEY, [])
        return final_state.pos, traces
    return final_state.pos


__all__ = ["build_kk_pipeline", "layout_kk_pipeline"]
