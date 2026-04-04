"""Stress majorization (SMACOF) expressed as a composable ops pipeline."""

from __future__ import annotations

from typing import ClassVar, List, Optional, Tuple, Union

import numpy as np
import torch

from dagua.layout.ops.graph_utils import (
    layout_device as _layout_device,
)
from dagua.layout.ops.graph_utils import (
    shortest_path_distances as _shortest_path_distances,
)

# ---------------------------------------------------------------------------
# Algorithm-specific constants and functions copied from
# dagua/layout/classic/stress_majorization.py (bit-identical)
# ---------------------------------------------------------------------------

_SM_MIN_DISTANCE = 1.0e-9
_MIN_SPAN = 1.0e-6


def _layout_classical_mds(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Lay out a graph with classical multidimensional scaling.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]`` used only to pick a
        stable drawing extent.
    seed : int, default=42
        Accepted for interface compatibility. Classical MDS is deterministic
        once the graph distances are fixed.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]`` used when computing
        shortest-path distances.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If ``num_nodes`` is negative or ``edge_weights`` has the wrong shape.
    """
    del seed

    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
            )

    device = _layout_device(edge_index=edge_index, node_sizes=node_sizes)
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)

    distances = _shortest_path_distances(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weights=edge_weights,
    )
    raw_positions = _classical_mds_embedding(distances)
    extent = _layout_extent(num_nodes=num_nodes, node_sizes=node_sizes)
    normalized = _normalize_positions(raw_positions.to(device=device), extent=extent)
    return normalized.to(dtype=torch.float32, device=device)


def _layout_extent(
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
) -> float:
    """Estimate a stable output scale for the embedding.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``.

    Returns
    -------
    float
        Target half-width after normalization.
    """
    if node_sizes is None or node_sizes.numel() == 0:
        return max(float(max(num_nodes, 1)) ** 0.5 * 5.0, 1.0)

    max_size = float(node_sizes.to(dtype=torch.float32, device="cpu").max().item())
    return max(max_size * max(float(max(num_nodes, 1)) ** 0.5, 1.0) * 2.0, 1.0)


def _normalize_positions(positions: torch.Tensor, extent: float) -> torch.Tensor:
    """Center and scale coordinates into a stable bounding box.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    extent : float
        Target half-width after normalization.

    Returns
    -------
    torch.Tensor
        Centered and scaled coordinates.
    """
    if positions.shape[0] <= 1:
        return torch.zeros_like(positions)

    centered = positions - positions.mean(dim=0, keepdim=True)
    span = float(centered.abs().max().item())
    if span < _MIN_SPAN:
        centered = centered.clone()
        centered[:, 0] = torch.linspace(
            -1.0,
            1.0,
            steps=positions.shape[0],
            device=positions.device,
            dtype=positions.dtype,
        )
        span = float(centered.abs().max().item())
    return centered * (extent / max(span, _MIN_SPAN))


def _classical_mds_embedding(distances: np.ndarray) -> torch.Tensor:
    """Recover a rank-2 embedding from pairwise graph distances.

    Parameters
    ----------
    distances : numpy.ndarray
        Dense shortest-path distance matrix with shape ``[N, N]``.

    Returns
    -------
    torch.Tensor
        Raw coordinates with shape ``[N, 2]`` on CPU.
    """
    num_nodes = int(distances.shape[0])
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32)

    squared = distances * distances
    centering = np.eye(num_nodes, dtype=np.float64) - (
        np.ones((num_nodes, num_nodes), dtype=np.float64) / float(num_nodes)
    )
    gram = -0.5 * centering @ squared @ centering

    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    positive_indices = [index for index in sorted_indices if eigenvalues[index] > 0.0][:2]

    coordinates = np.zeros((num_nodes, 2), dtype=np.float64)
    if positive_indices:
        selected_values = np.clip(eigenvalues[positive_indices], a_min=0.0, a_max=None)
        selected_vectors = eigenvectors[:, positive_indices]
        coordinates[:, : len(positive_indices)] = selected_vectors * np.sqrt(selected_values)
    else:
        coordinates[:, 0] = np.linspace(-1.0, 1.0, num_nodes, dtype=np.float64)

    return torch.from_numpy(coordinates).to(dtype=torch.float32)


def _pairwise_distances(positions: np.ndarray) -> np.ndarray:
    """Compute dense Euclidean distances between all node pairs.

    Parameters
    ----------
    positions : numpy.ndarray
        Position array with shape ``[N, 2]``.

    Returns
    -------
    numpy.ndarray
        Pairwise Euclidean distances with shape ``[N, N]``.
    """
    deltas = positions[:, None, :] - positions[None, :, :]
    return np.sqrt(np.sum(deltas * deltas, axis=2))


def _stress_value(
    positions: np.ndarray,
    target_distances: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Compute the weighted stress objective for one embedding.

    Parameters
    ----------
    positions : numpy.ndarray
        Position array with shape ``[N, 2]``.
    target_distances : numpy.ndarray
        Desired graph distances with shape ``[N, N]``.
    weights : numpy.ndarray
        SMACOF weight matrix with shape ``[N, N]``.

    Returns
    -------
    float
        Weighted stress value.
    """
    current_distances = _pairwise_distances(positions)
    errors = current_distances - target_distances
    return 0.5 * float(np.sum(weights * errors * errors))


def _smacof_update(
    positions: np.ndarray,
    target_distances: np.ndarray,
    weights: np.ndarray,
    laplacian_pinv: np.ndarray,
) -> np.ndarray:
    """Apply one SMACOF majorization step.

    Parameters
    ----------
    positions : numpy.ndarray
        Current positions with shape ``[N, 2]``.
    target_distances : numpy.ndarray
        Desired graph distances with shape ``[N, N]``.
    weights : numpy.ndarray
        SMACOF weight matrix with shape ``[N, N]``.
    laplacian_pinv : numpy.ndarray
        Pseudoinverse of the weighted Laplacian with shape ``[N, N]``.

    Returns
    -------
    numpy.ndarray
        Updated centered positions with shape ``[N, 2]``.
    """
    current_distances = np.maximum(_pairwise_distances(positions), _SM_MIN_DISTANCE)
    ratio = np.zeros_like(target_distances)
    active_mask = weights > 0.0
    ratio[active_mask] = target_distances[active_mask] / current_distances[active_mask]

    b_matrix = -weights * ratio
    np.fill_diagonal(b_matrix, 0.0)
    np.fill_diagonal(b_matrix, -b_matrix.sum(axis=1))

    updated = laplacian_pinv @ (b_matrix @ positions)
    return updated - updated.mean(axis=0, keepdims=True)


def _initial_positions(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor],
    seed: int,
    edge_weights: Optional[torch.Tensor],
) -> np.ndarray:
    """Build a stable stochastic warm start for SMACOF.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``.
    seed : int
        Random seed for the warm-start jitter.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.

    Returns
    -------
    numpy.ndarray
        Initial positions with shape ``[N, 2]``.
    """
    baseline = _layout_classical_mds(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        seed=42,
        edge_weights=edge_weights,
    )
    rng = np.random.default_rng(seed)
    jitter = rng.normal(loc=0.0, scale=0.05, size=(num_nodes, 2))
    initialized = baseline.detach().cpu().numpy().astype(np.float64) + jitter
    return initialized - initialized.mean(axis=0, keepdims=True)


from dagua.layout.ops.base import Op, Pipeline, Repeat  # noqa: E402
from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig  # noqa: E402
from dagua.layout.ops.state import (  # noqa: E402
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)
from dagua.layout.ops.taxonomy import OpCategory  # noqa: E402

_MIN_DISTANCE = 1.0e-9

_TARGET_DISTANCES_KEY = "sm_target_distances"
_WEIGHTS_KEY = "sm_weights"
_LAPLACIAN_PINV_KEY = "sm_laplacian_pinv"
_CURRENT_POSITIONS_KEY = "sm_current_positions"
_CURRENT_STRESS_KEY = "sm_current_stress"
_TRACES_KEY = "sm_traces"
_TRACE_EVERY_KEY = "sm_trace_every"
_SHORT_CIRCUIT_KEY = "sm_short_circuit"


class _PrepareStressMajorizationState(Op):
    """Compute target distances, SMACOF weights, and Laplacian pseudoinverse."""

    name: ClassVar[str] = "sm_prepare_state"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build the distance, weight, and Laplacian matrices for SMACOF.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state receiving precomputed matrices.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with SMACOF matrices stored in ``extras``.
        """
        del ctx

        target_distances = _shortest_path_distances(
            edge_index=problem.edge_index,
            num_nodes=problem.num_nodes,
            edge_weights=problem.edge_weights,
        )
        with np.errstate(divide="ignore"):
            weights = np.where(
                target_distances > 0.0,
                1.0 / np.square(target_distances),
                0.0,
            )
        np.fill_diagonal(weights, 0.0)

        laplacian = -weights.copy()
        np.fill_diagonal(laplacian, weights.sum(axis=1))
        laplacian_pinv = np.linalg.pinv(laplacian)

        state.extras[_TARGET_DISTANCES_KEY] = target_distances
        state.extras[_WEIGHTS_KEY] = weights
        state.extras[_LAPLACIAN_PINV_KEY] = laplacian_pinv
        return state


class _InitializeStressMajorizationPositions(Op):
    """Initialize positions exactly like classic stress majorization."""

    name: ClassVar[str] = "sm_initialize_positions"
    category: ClassVar[OpCategory] = OpCategory.INIT
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Seed positions from classical MDS with stochastic jitter.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs containing graph topology and seed.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with initial numpy positions and stress stored in
            ``extras``.
        """
        del ctx

        current = _initial_positions(
            edge_index=problem.edge_index,
            num_nodes=problem.num_nodes,
            node_sizes=problem.node_sizes,
            seed=problem.seed,
            edge_weights=problem.edge_weights,
        )
        current_stress = _stress_value(
            current,
            target_distances=state.extras[_TARGET_DISTANCES_KEY],
            weights=state.extras[_WEIGHTS_KEY],
        )
        state.extras[_CURRENT_POSITIONS_KEY] = current
        state.extras[_CURRENT_STRESS_KEY] = current_stress
        return state


class _SmacofStep(Op):
    """Apply one SMACOF majorization step with monotonicity safeguard."""

    name: ClassVar[str] = "sm_smacof_step"
    category: ClassVar[OpCategory] = OpCategory.OPTIMIZE
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run one SMACOF update with conservative blending fallback.

        This mirrors the classic implementation exactly: attempt a full
        SMACOF update, and if the candidate stress exceeds the current
        stress by more than 1e-8, apply up to 8 rounds of conservative
        blending. If blending fails, keep the current positions.

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
            State with updated positions and stress in ``extras``.
        """
        del problem, ctx

        current = state.extras[_CURRENT_POSITIONS_KEY]
        current_stress = state.extras[_CURRENT_STRESS_KEY]
        target_distances = state.extras[_TARGET_DISTANCES_KEY]
        weights = state.extras[_WEIGHTS_KEY]
        laplacian_pinv = state.extras[_LAPLACIAN_PINV_KEY]

        candidate = _smacof_update(
            positions=current,
            target_distances=target_distances,
            weights=weights,
            laplacian_pinv=laplacian_pinv,
        )
        candidate_stress = _stress_value(
            candidate,
            target_distances=target_distances,
            weights=weights,
        )

        if candidate_stress > current_stress + 1.0e-8:
            blended = candidate
            for _ in range(8):
                blended = 0.5 * (blended + current)
                candidate_stress = _stress_value(
                    blended,
                    target_distances=target_distances,
                    weights=weights,
                )
                if candidate_stress <= current_stress + 1.0e-8:
                    candidate = blended
                    break
            else:
                candidate = current
                candidate_stress = current_stress

        state.extras[_CURRENT_POSITIONS_KEY] = candidate
        state.extras[_CURRENT_STRESS_KEY] = candidate_stress
        return state


class _CollectTrace(Op):
    """Optionally snapshot positions for trace output."""

    name: ClassVar[str] = "sm_collect_trace"
    category: ClassVar[OpCategory] = OpCategory.POSTPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Append a position snapshot if trace cadence is met.

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
            State with trace list updated if appropriate.
        """
        del ctx

        trace_every = state.extras.get(_TRACE_EVERY_KEY, 0)
        if trace_every <= 0:
            return state

        device = _layout_device(
            edge_index=problem.edge_index,
            node_sizes=problem.node_sizes,
        )
        iterations = state.total_steps
        # state.step is incremented by Repeat AFTER the inner pipeline runs,
        # so inside the inner pipeline state.step is the 0-based iteration
        # index of the CURRENT iteration (not yet incremented).
        iteration_idx = state.step
        traces = state.extras.setdefault(_TRACES_KEY, [])

        if (iteration_idx + 1) % trace_every == 0 or iteration_idx + 1 == iterations:
            current = state.extras[_CURRENT_POSITIONS_KEY]
            traces.append(torch.from_numpy(current).to(dtype=torch.float32, device=device))

        return state


class _FinalizeStressMajorizationPositions(Op):
    """Convert numpy result to float32 output tensor on the correct device."""

    name: ClassVar[str] = "sm_finalize_positions"
    category: ClassVar[OpCategory] = OpCategory.POSTPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("pos", "extras")
    requires: ClassVar[Tuple[str, ...]] = ()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Move final numpy positions to the output device as float32.

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
            State with ``pos`` set to the final positions and trace list
            finalized.
        """
        del ctx

        device = _layout_device(
            edge_index=problem.edge_index,
            node_sizes=problem.node_sizes,
        )
        current = state.extras[_CURRENT_POSITIONS_KEY]
        final_positions = torch.from_numpy(current).to(dtype=torch.float32, device=device)
        state.pos = final_positions

        # Finalize traces: ensure the last snapshot matches the final output.
        trace_every = state.extras.get(_TRACE_EVERY_KEY, 0)
        if trace_every > 0:
            traces = state.extras.get(_TRACES_KEY, [])
            if not traces or not torch.allclose(traces[-1], final_positions):
                traces.append(final_positions.clone())
            state.extras[_TRACES_KEY] = traces

        return state


def build_stress_majorization_pipeline(
    iterations: int = 200,
    trace_every: int = 0,
) -> Pipeline:
    """Build a stress majorization pipeline bit-identical to the classic.

    Parameters
    ----------
    iterations : int, default=200
        Number of SMACOF majorization steps.
    trace_every : int, default=0
        If positive, collect position snapshots at this cadence.

    Returns
    -------
    Pipeline
        Pipeline that reproduces classic stress majorization exactly.

    Raises
    ------
    ValueError
        If ``iterations`` or ``trace_every`` is negative.
    """
    if iterations < 0:
        raise ValueError("iterations must be non-negative.")
    if trace_every < 0:
        raise ValueError("trace_every must be non-negative.")

    return Pipeline(
        [
            FixedSteps(FixedStepsConfig(n=iterations)),
            _PrepareStressMajorizationState(),
            _InitializeStressMajorizationPositions(),
            Repeat(
                n=iterations,
                ops=[
                    _SmacofStep(),
                    _CollectTrace(),
                ],
            ),
            _FinalizeStressMajorizationPositions(),
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
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
    """Run the stress majorization pipeline as a drop-in classic replacement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
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

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, list[torch.Tensor]]
        Final position tensor with shape ``[N, 2]``. When ``trace_every > 0``,
        also returns periodic snapshots.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``iterations``, ``trace_every``, or ``edge_weights``
        are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if iterations < 0:
        raise ValueError("iterations must be non-negative.")
    if trace_every < 0:
        raise ValueError("trace_every must be non-negative.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
            )

    device = _layout_device(edge_index=edge_index, node_sizes=node_sizes)
    if num_nodes == 0:
        empty = torch.empty((0, 2), dtype=torch.float32, device=device)
        return (empty, []) if trace_every > 0 else empty
    if num_nodes == 1:
        single = torch.zeros((1, 2), dtype=torch.float32, device=device)
        return (single, []) if trace_every > 0 else single

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    state.extras[_TRACE_EVERY_KEY] = trace_every
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_stress_majorization_pipeline(
        iterations=iterations,
        trace_every=trace_every,
    ).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("Stress majorization pipeline did not produce final positions.")

    if trace_every > 0:
        traces = final_state.extras.get(_TRACES_KEY, [])
        return final_state.pos, traces
    return final_state.pos


__all__ = [
    "build_stress_majorization_pipeline",
    "layout_stress_majorization_pipeline",
]
