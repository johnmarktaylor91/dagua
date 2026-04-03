"""Stress majorization (SMACOF) expressed as a composable ops pipeline."""

from __future__ import annotations

from typing import ClassVar, List, Optional, Tuple, Union

import numpy as np
import torch

from dagua.layout.classic.classical_mds import _shortest_path_distances
from dagua.layout.classic.stress_majorization import (
    _initial_positions,
    _layout_device,
    _smacof_update,
    _stress_value,
)
from dagua.layout.ops.base import Op, Pipeline, Repeat
from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory

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
