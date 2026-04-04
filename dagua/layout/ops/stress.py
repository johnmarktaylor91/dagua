"""Stress-majorization primitive operations.

These ops expose the dense SMACOF implementation used by
stress-majorization as composable, registered layout operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Tuple

import numpy as np
import torch

from dagua.layout.ops.base import Op
from dagua.layout.ops.graph_utils import layout_device as _layout_device
from dagua.layout.ops.graph_utils import shortest_path_distances as _shortest_path_distances
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_SM_MIN_DISTANCE = 1.0e-9
_SM_MIN_SPAN = 1.0e-6


@dataclass(frozen=True)
class InitializeStressMajorizationPositionsConfig:
    """Configuration for :class:`InitializeStressMajorizationPositions`.

    Parameters
    ----------
    jitter_scale : float, default=0.05
        Standard deviation of the Gaussian jitter added to the MDS warm start.
    """

    jitter_scale: float = 0.05


@dataclass(frozen=True)
class SmacofStepConfig:
    """Configuration for :class:`SmacofStep`.

    Parameters
    ----------
    stress_tolerance : float, default=1e-8
        Absolute stress-increase tolerance for accepting a candidate step.
    max_halving_steps : int, default=8
        Maximum bisection halving attempts when the candidate increases stress.
    """

    stress_tolerance: float = 1.0e-8
    max_halving_steps: int = 8


WEIGHTS_KEY = "sm_weights"
CURRENT_POSITIONS_KEY = "sm_current_positions"
CURRENT_STRESS_KEY = "sm_current_stress"
TRACES_KEY = "sm_traces"
TRACE_EVERY_KEY = "sm_trace_every"


@register_op
@dataclass(frozen=True)
class PrepareStressMajorizationState(Op):
    """Prepare distance-derived SMACOF state.

    This op computes all-pairs shortest-path distances, the inverse
    stress weights, and the pseudoinverse of the weighted graph
    Laplacian.
    """

    name: ClassVar[str] = "sm_prepare_state"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = ("extras",)
    requires: ClassVar[Tuple[str, ...]] = ()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute distance matrix, SMACOF weights, and Laplacian inverse.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context (unused).

        Returns
        -------
        SolveState
            State with SMACOF precomputation in ``state.extras``.
        """
        del self, ctx

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

        laplacian = -weights
        np.fill_diagonal(laplacian, weights.sum(axis=1))
        laplacian_pinv = np.linalg.pinv(laplacian)

        state.extras[WEIGHTS_KEY] = weights
        state.distance_matrix = torch.from_numpy(target_distances)
        state.laplacian = laplacian_pinv
        return state


@register_op
@dataclass(frozen=True)
class InitializeStressMajorizationPositions(Op):
    """Seed positions from classical MDS with deterministic jitter."""

    config: InitializeStressMajorizationPositionsConfig = field(
        default_factory=InitializeStressMajorizationPositionsConfig
    )

    name: ClassVar[str] = "sm_initialize_positions"
    category: ClassVar[OpCategory] = OpCategory.INIT
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)
    requires: ClassVar[Tuple[str, ...]] = ("extras",)

    def _layout_extent(
        self,
        num_nodes: int,
        node_sizes: torch.Tensor | None,
    ) -> float:
        """Estimate a stable scale target for classical-MDS warm starts.

        Parameters
        ----------
        num_nodes : int
            Number of nodes.
        node_sizes : torch.Tensor | None
            Optional node-size tensor with shape ``[N, 2]``.

        Returns
        -------
        float
            Half-width scale factor.
        """
        if node_sizes is None or node_sizes.numel() == 0:
            return max(float(max(num_nodes, 1)) ** 0.5 * 5.0, 1.0)

        max_size = float(node_sizes.to(dtype=torch.float32, device="cpu").max().item())
        return max(max_size * max(float(max(num_nodes, 1)) ** 0.5, 1.0) * 2.0, 1.0)

    def _classical_mds_embedding(self, distances: np.ndarray) -> torch.Tensor:
        """Recover a rank-2 embedding from pairwise distances.

        Parameters
        ----------
        distances : numpy.ndarray
            Dense pairwise graph distances with shape ``[N, N]``.

        Returns
        -------
        torch.Tensor
            Rank-2 classical-MDS coordinates on CPU.
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
            selected_values = np.clip(
                eigenvalues[positive_indices],
                a_min=0.0,
                a_max=None,
            )
            selected_vectors = eigenvectors[:, positive_indices]
            coordinates[:, : len(positive_indices)] = selected_vectors * np.sqrt(selected_values)
        else:
            coordinates[:, 0] = np.linspace(-1.0, 1.0, num_nodes, dtype=np.float64)

        return torch.from_numpy(coordinates).to(dtype=torch.float32)

    @staticmethod
    def _normalize_positions(positions: torch.Tensor, extent: float) -> torch.Tensor:
        """Center and scale coordinates into a stable drawing box.

        Parameters
        ----------
        positions : torch.Tensor
            Unnormalized coordinates with shape ``[N, 2]``.
        extent : float
            Target half-width.

        Returns
        -------
        torch.Tensor
            Centered and scaled coordinates.
        """
        if positions.shape[0] <= 1:
            return torch.zeros_like(positions)

        centered = positions - positions.mean(dim=0, keepdim=True)
        span = float(centered.abs().max().item())
        if span < _SM_MIN_SPAN:
            centered = centered.clone()
            centered[:, 0] = torch.linspace(
                -1.0,
                1.0,
                steps=positions.shape[0],
                device=positions.device,
                dtype=positions.dtype,
            )
            span = float(centered.abs().max().item())
        return centered * (extent / max(span, _SM_MIN_SPAN))

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build one stochastic MDS warm start.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with target distances prepared.
        ctx : RuntimeContext
            Runtime context (unused).

        Returns
        -------
        SolveState
            State with initialized ``sm_current_positions`` and stress.
        """
        del ctx

        if not isinstance(state.distance_matrix, torch.Tensor):
            raise ValueError(
                "sm_initialize_positions requires target distances in state.distance_matrix."
            )

        target_distances = state.distance_matrix.to(dtype=torch.float64, device="cpu").numpy()

        raw_positions = self._classical_mds_embedding(distances=target_distances)
        extent = self._layout_extent(
            num_nodes=problem.num_nodes,
            node_sizes=problem.node_sizes,
        )
        device = _layout_device(
            edge_index=problem.edge_index,
            node_sizes=problem.node_sizes,
        )
        baseline = self._normalize_positions(raw_positions.to(device=device), extent=extent)

        rng = np.random.default_rng(problem.seed)
        jitter = rng.normal(loc=0.0, scale=self.config.jitter_scale, size=(problem.num_nodes, 2))
        initialized = baseline.detach().cpu().numpy().astype(np.float64) + jitter
        initialized = initialized - initialized.mean(axis=0, keepdims=True)

        deltas = initialized[:, None, :] - initialized[None, :, :]
        current_distances = np.sqrt(np.sum(deltas * deltas, axis=2))

        weights = state.extras.get(WEIGHTS_KEY)
        if not isinstance(weights, np.ndarray):
            raise ValueError("sm_initialize_positions requires sm_weights in state.extras.")

        errors = current_distances - target_distances
        current_stress = 0.5 * float(np.sum(weights * errors * errors))

        state.extras[CURRENT_POSITIONS_KEY] = initialized
        state.extras[CURRENT_STRESS_KEY] = current_stress
        return state


@register_op
@dataclass(frozen=True)
class SmacofStep(Op):
    """Apply one dense SMACOF majorization update."""

    config: SmacofStepConfig = field(default_factory=SmacofStepConfig)

    name: ClassVar[str] = "sm_smacof_step"
    category: ClassVar[OpCategory] = OpCategory.OPTIMIZE
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)
    requires: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run one safeguarded SMACOF update.

        If the candidate position decreases stress relative to the current
        position, this op applies a conservative halving blend up to eight
        times. If no blend improves stress, the step is rejected.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable state with current positions, target distances, weights,
            and Laplacian pseudoinverse.
        ctx : RuntimeContext
            Runtime context (unused).

        Returns
        -------
        SolveState
            State with updated positions and stress in ``state.extras``.
        """
        del ctx

        current = state.extras[CURRENT_POSITIONS_KEY]
        current_stress = state.extras[CURRENT_STRESS_KEY]
        target_distances = state.distance_matrix
        weights = state.extras[WEIGHTS_KEY]
        laplacian_pinv = state.laplacian

        if not isinstance(current, np.ndarray):
            raise ValueError("sm_smacof_step requires sm_current_positions as numpy array.")
        if not isinstance(current_stress, float):
            current_stress = float(current_stress)
        if not isinstance(target_distances, torch.Tensor):
            raise ValueError(
                "sm_smacof_step requires sm_target_distances in state.distance_matrix."
            )
        if not isinstance(weights, np.ndarray):
            raise ValueError("sm_smacof_step requires sm_weights in state.extras.")
        target_distances_np = target_distances.to(dtype=torch.float64, device="cpu").numpy()
        if isinstance(laplacian_pinv, torch.Tensor):
            laplacian_pinv_np = laplacian_pinv.to(dtype=torch.float64, device="cpu").numpy()
        elif isinstance(laplacian_pinv, np.ndarray):
            laplacian_pinv_np = laplacian_pinv
        else:
            raise ValueError("sm_smacof_step requires sm_laplacian_pinv in state.laplacian.")

        current_distances = np.maximum(
            np.sqrt(
                np.sum(
                    (current[:, None, :] - current[None, :, :])
                    * (current[:, None, :] - current[None, :, :]),
                    axis=2,
                )
            ),
            _SM_MIN_DISTANCE,
        )

        ratio = np.zeros_like(target_distances_np)
        active_mask = weights > 0.0
        ratio[active_mask] = target_distances_np[active_mask] / current_distances[active_mask]

        b_matrix = -weights * ratio
        np.fill_diagonal(b_matrix, 0.0)
        np.fill_diagonal(b_matrix, -b_matrix.sum(axis=1))

        candidate = laplacian_pinv_np @ (b_matrix @ current)
        candidate = candidate - candidate.mean(axis=0, keepdims=True)

        candidate_distances = np.sqrt(
            np.sum(
                (candidate[:, None, :] - candidate[None, :, :])
                * (candidate[:, None, :] - candidate[None, :, :]),
                axis=2,
            )
        )
        candidate_stress = 0.5 * float(
            np.sum(weights * (candidate_distances - target_distances_np) ** 2)
        )

        tolerance = self.config.stress_tolerance
        if candidate_stress > current_stress + tolerance:
            # Bisection halving: blend candidate toward current until stress improves
            blended = candidate
            for _ in range(self.config.max_halving_steps):
                blended = 0.5 * (blended + current)
                blended_distances = np.sqrt(
                    np.sum(
                        (blended[:, None, :] - blended[None, :, :])
                        * (blended[:, None, :] - blended[None, :, :]),
                        axis=2,
                    )
                )
                blended_stress = 0.5 * float(
                    np.sum(weights * (blended_distances - target_distances_np) ** 2)
                )
                if blended_stress <= current_stress + tolerance:
                    candidate = blended
                    candidate_stress = blended_stress
                    break
            else:
                candidate = current
                candidate_stress = current_stress

        state.extras[CURRENT_POSITIONS_KEY] = candidate
        state.extras[CURRENT_STRESS_KEY] = candidate_stress
        return state


@register_op
@dataclass(frozen=True)
class CollectStressMajorizationTrace(Op):
    """Collect periodic position snapshots from SMACOF iterations."""

    name: ClassVar[str] = "sm_collect_trace"
    category: ClassVar[OpCategory] = OpCategory.POSTPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)
    requires: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Append a trace snapshot when the cadence is satisfied.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable state with tracked positions and iteration counters.
        ctx : RuntimeContext
            Runtime context (unused).

        Returns
        -------
        SolveState
            State with traces updated when requested.
        """
        del self, ctx

        trace_every = state.extras.get(TRACE_EVERY_KEY, 0)
        if not isinstance(trace_every, int) or trace_every <= 0:
            return state

        iterations = state.total_steps
        iteration_idx = state.step
        traces = state.extras.setdefault(TRACES_KEY, [])
        if (iteration_idx + 1) % trace_every == 0 or iteration_idx + 1 == iterations:
            current = state.extras[CURRENT_POSITIONS_KEY]
            device = _layout_device(
                edge_index=problem.edge_index,
                node_sizes=problem.node_sizes,
            )
            traces.append(torch.from_numpy(current).to(dtype=torch.float32, device=device))

        return state


@register_op
@dataclass(frozen=True)
class FinalizeStressMajorizationPositions(Op):
    """Materialize final SMACOF coordinates in the solve state."""

    name: ClassVar[str] = "sm_finalize_positions"
    category: ClassVar[OpCategory] = OpCategory.POSTPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("pos", "extras")
    requires: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Write final positions and finalize optional trace list.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context (unused).

        Returns
        -------
        SolveState
            State with ``state.pos`` filled and traces finalized.
        """
        del self, ctx

        current = state.extras[CURRENT_POSITIONS_KEY]
        if not isinstance(current, np.ndarray):
            raise ValueError("sm_finalize_positions requires sm_current_positions in state.extras.")

        device = _layout_device(
            edge_index=problem.edge_index,
            node_sizes=problem.node_sizes,
        )
        final_positions = torch.from_numpy(current).to(dtype=torch.float32, device=device)
        state.pos = final_positions

        trace_every = state.extras.get(TRACE_EVERY_KEY, 0)
        if trace_every > 0:
            traces = state.extras.get(TRACES_KEY, [])
            if not traces or not torch.allclose(traces[-1], final_positions):
                traces.append(final_positions.clone())
            state.extras[TRACES_KEY] = traces

        return state
