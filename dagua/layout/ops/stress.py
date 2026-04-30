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
from dagua.layout.ops.graph_utils import (
    _shared_all_pairs_shortest_paths,
    _shared_build_undirected_adjacency,
)
from dagua.layout.ops.graph_utils import layout_device as _layout_device
from dagua.layout.ops.graph_utils import shortest_path_distances as _shortest_path_distances
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_SM_MIN_DISTANCE = 1.0e-9
_SM_MIN_SPAN = 1.0e-6
_SM_DISTANCE_FILL_CLASSIC = "classic"
_SM_DISTANCE_FILL_OGDF = "ogdf"
_SM_UPDATE_DENSE = "dense"
_SM_UPDATE_OGDF_SERIAL = "ogdf_serial"


@dataclass(frozen=True)
class PrepareStressMajorizationStateConfig:
    """Configuration for :class:`PrepareStressMajorizationState`.

    Parameters
    ----------
    distance_fill : str, default="classic"
        Unreachable-distance fill policy. ``"classic"`` preserves dagua's
        existing ``diameter + 1`` fill, while ``"ogdf"`` uses ``sqrt(N)`` in
        unit-distance scale to match OGDF's ``100 * sqrt(N)`` after global
        scale normalization.
    """

    distance_fill: str = _SM_DISTANCE_FILL_CLASSIC


@dataclass(frozen=True)
class InitializeStressMajorizationPositionsConfig:
    """Configuration for :class:`InitializeStressMajorizationPositions`.

    Parameters
    ----------
    jitter_scale : float, default=0.05
        Standard deviation of the Gaussian jitter added to the MDS warm start.
    base_extent_scale : float, default=5.0
        Base extent multiplier used when no node sizes are available.
    size_extent_multiplier : float, default=2.0
        Extra padding factor applied when node sizes define the layout scale.
    min_extent : float, default=1.0
        Minimum half-width used for degenerate or tiny layouts.
    fallback_line_start : float, default=-1.0
        Start of the deterministic line fallback when MDS has no positive modes.
    fallback_line_stop : float, default=1.0
        End of the deterministic line fallback when MDS has no positive modes.
    min_span : float, default=1e-6
        Minimum span accepted before the normalized embedding is re-expanded.
    """

    jitter_scale: float = 0.05
    base_extent_scale: float = 5.0
    size_extent_multiplier: float = 2.0
    min_extent: float = 1.0
    fallback_line_start: float = -1.0
    fallback_line_stop: float = 1.0
    min_span: float = 1.0e-6


@dataclass(frozen=True)
class SmacofStepConfig:
    """Configuration for :class:`SmacofStep`.

    Parameters
    ----------
    stress_tolerance : float, default=1e-8
        Absolute stress-increase tolerance for accepting a candidate step.
    max_halving_steps : int, default=8
        Maximum bisection halving attempts when the candidate increases stress.
    min_distance : float, default=1e-9
        Minimum Euclidean distance used to avoid division by zero in ``B(X)``.
    update_mode : str, default="dense"
        Majorization update implementation. ``"dense"`` uses dagua's existing
        pseudoinverse SMACOF update, while ``"ogdf_serial"`` uses OGDF's
        in-place serial weighted vote sweep.
    """

    stress_tolerance: float = 1.0e-8
    max_halving_steps: int = 8
    min_distance: float = 1.0e-9
    update_mode: str = _SM_UPDATE_DENSE


WEIGHTS_KEY = "sm_weights"
CURRENT_POSITIONS_KEY = "sm_current_positions"
CURRENT_STRESS_KEY = "sm_current_stress"
TRACES_KEY = "sm_traces"
TRACE_EVERY_KEY = "sm_trace_every"


@register_op
@dataclass(frozen=True)
class PrepareStressMajorizationState(Op):
    """Prepare the dense state required by the SMACOF update.

    This op computes all-pairs shortest-path distances, the inverse
    stress weights, and the pseudoinverse of the weighted graph
    Laplacian. The resulting tensors exactly match the classic dense
    stress-majorization implementation.
    """

    config: PrepareStressMajorizationStateConfig = field(
        default_factory=PrepareStressMajorizationStateConfig
    )

    name: ClassVar[str] = "sm_prepare_state"
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
        del ctx

        target_distances = self._target_distances(problem=problem)
        with np.errstate(divide="ignore"):
            # Classical stress majorization uses inverse-squared target distances
            # and explicitly zeroes the diagonal so self-pairs stay inactive.
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

    def _target_distances(self, problem: LayoutProblem) -> np.ndarray:
        """Compute shortest-path distances using the configured fill policy.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph layout inputs.

        Returns
        -------
        numpy.ndarray
            Dense finite distance matrix with shape ``[N, N]``.

        Raises
        ------
        ValueError
            If the configured fill policy is unknown.
        """
        if self.config.distance_fill == _SM_DISTANCE_FILL_CLASSIC:
            return _shortest_path_distances(
                edge_index=problem.edge_index,
                num_nodes=problem.num_nodes,
                edge_weights=problem.edge_weights,
            )
        if self.config.distance_fill != _SM_DISTANCE_FILL_OGDF:
            raise ValueError(f"Unknown stress distance fill: {self.config.distance_fill!r}.")

        adjacency = _shared_build_undirected_adjacency(
            edge_index=problem.edge_index,
            num_nodes=problem.num_nodes,
            edge_weights=problem.edge_weights,
        )
        weighted = problem.edge_weights is not None
        raw_distances = _shared_all_pairs_shortest_paths(adjacency, weighted=weighted)
        finite_mask = np.isfinite(raw_distances) if weighted else raw_distances >= 0
        fill_value = float(problem.num_nodes) ** 0.5 if problem.num_nodes > 1 else 0.0
        cleaned = np.where(finite_mask, raw_distances, fill_value).astype(np.float64, copy=False)
        np.fill_diagonal(cleaned, 0.0)
        return cleaned


@register_op
@dataclass(frozen=True)
class InitializeStressMajorizationPositions(Op):
    """Build the classical-MDS warm start used by the SMACOF pipeline.

    The op reproduces the classic initialization recipe: recover a rank-2
    embedding from graph distances, normalize it into a stable extent, then add
    seeded Gaussian jitter so disconnected symmetries break deterministically.
    """

    config: InitializeStressMajorizationPositionsConfig = field(
        default_factory=InitializeStressMajorizationPositionsConfig
    )

    name: ClassVar[str] = "sm_initialize_positions"
    category: ClassVar[OpCategory] = OpCategory.INIT
    reads: ClassVar[Tuple[str, ...]] = ("distance_matrix", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("extras",)
    requires: ClassVar[Tuple[str, ...]] = ("distance_matrix", "extras")

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
            return max(
                float(max(num_nodes, 1)) ** 0.5 * self.config.base_extent_scale,
                self.config.min_extent,
            )

        max_size = float(node_sizes.to(dtype=torch.float32, device="cpu").max().item())
        return max(
            max_size
            * max(float(max(num_nodes, 1)) ** 0.5, self.config.min_extent)
            * self.config.size_extent_multiplier,
            self.config.min_extent,
        )

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
            return torch.empty((0, 2), dtype=torch.float64)
        if num_nodes == 1:
            return torch.zeros((1, 2), dtype=torch.float64)

        squared = distances * distances
        centering = np.eye(num_nodes, dtype=np.float64) - (
            np.ones((num_nodes, num_nodes), dtype=np.float64) / float(num_nodes)
        )
        # Double-centering converts squared distances into the Gram matrix used
        # by classical MDS.
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
            # Degenerate spectra fall back to a deterministic line so the later
            # jitter still has a stable scaffold to perturb.
            coordinates[:, 0] = np.linspace(
                self.config.fallback_line_start,
                self.config.fallback_line_stop,
                num_nodes,
                dtype=np.float64,
            )

        return torch.from_numpy(coordinates)

    def _normalize_positions(self, positions: torch.Tensor, extent: float) -> torch.Tensor:
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
        if span < self.config.min_span:
            centered = centered.clone()
            centered[:, 0] = torch.linspace(
                self.config.fallback_line_start,
                self.config.fallback_line_stop,
                steps=positions.shape[0],
                device=positions.device,
                dtype=positions.dtype,
            )
            span = float(centered.abs().max().item())
        return centered * (extent / max(span, self.config.min_span))

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

        # Keep the initial stress cached in extras so each SMACOF step can apply
        # the classical monotonicity safeguard without recomputing prior state.
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
    """Apply one stress-majorization update."""

    config: SmacofStepConfig = field(default_factory=SmacofStepConfig)

    name: ClassVar[str] = "sm_smacof_step"
    category: ClassVar[OpCategory] = OpCategory.OPTIMIZE
    reads: ClassVar[Tuple[str, ...]] = ("distance_matrix", "laplacian", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("extras",)
    requires: ClassVar[Tuple[str, ...]] = ("distance_matrix", "laplacian", "extras")

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
        if self.config.update_mode == _SM_UPDATE_OGDF_SERIAL:
            candidate, candidate_stress = self._ogdf_serial_sweep(
                current=current,
                target_distances=target_distances_np,
                weights=weights,
            )
            state.extras[CURRENT_POSITIONS_KEY] = candidate
            state.extras[CURRENT_STRESS_KEY] = candidate_stress
            return state
        if self.config.update_mode != _SM_UPDATE_DENSE:
            raise ValueError(f"Unknown stress update mode: {self.config.update_mode!r}.")

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
            self.config.min_distance,
        )

        ratio = np.zeros_like(target_distances_np)
        active_mask = weights > 0.0
        ratio[active_mask] = target_distances_np[active_mask] / current_distances[active_mask]

        # ``B(X)`` reweights the Laplacian by the ratio between graph-space and
        # Euclidean distances at the current iterate.
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

    def _ogdf_serial_sweep(
        self,
        current: np.ndarray,
        target_distances: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Run one OGDF-compatible in-place serial vote sweep.

        Parameters
        ----------
        current : numpy.ndarray
            Current coordinates with shape ``[N, 2]``.
        target_distances : numpy.ndarray
            Dense graph-distance matrix with shape ``[N, N]``.
        weights : numpy.ndarray
            Inverse-square stress weights with shape ``[N, N]``.

        Returns
        -------
        tuple[numpy.ndarray, float]
            Updated coordinates and their weighted stress value.
        """
        candidate = current.copy()
        num_nodes = int(candidate.shape[0])
        for node in range(num_nodes):
            total_weight = 0.0
            vote = np.zeros(2, dtype=np.float64)
            current_coord = candidate[node].copy()
            for other in range(num_nodes):
                if other == node:
                    continue
                weight = float(weights[node, other])
                if weight <= 0.0:
                    continue
                other_coord = candidate[other]
                offset = current_coord - other_coord
                euclidean_dist = float(np.linalg.norm(offset))
                vote_coord = other_coord.copy()
                if euclidean_dist > self.config.min_distance:
                    vote_coord += target_distances[node, other] * offset / euclidean_dist
                vote += weight * vote_coord
                total_weight += weight
            if total_weight > 0.0:
                candidate[node] = vote / total_weight

        deltas = candidate[:, None, :] - candidate[None, :, :]
        candidate_distances = np.sqrt(np.sum(deltas * deltas, axis=2))
        candidate_stress = 0.5 * float(
            np.sum(weights * (candidate_distances - target_distances) ** 2)
        )
        return candidate, candidate_stress


@register_op
@dataclass(frozen=True)
class CollectStressMajorizationTrace(Op):
    """Collect periodic snapshots from the SMACOF iteration state."""

    name: ClassVar[str] = "sm_collect_trace"
    category: ClassVar[OpCategory] = OpCategory.POSTPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("extras", "step", "total_steps")
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
            # Trace snapshots are materialized on the output device so the
            # public adapter can return them directly without extra copies.
            traces.append(torch.from_numpy(current).to(dtype=torch.float32, device=device))

        return state


@register_op
@dataclass(frozen=True)
class FinalizeStressMajorizationPositions(Op):
    """Materialize final SMACOF coordinates and close out optional tracing."""

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
            # Ensure the returned trace always includes the final accepted state,
            # even when the cadence does not land on the last iteration.
            if not traces or not torch.allclose(traces[-1], final_positions):
                traces.append(final_positions.clone())
            state.extras[TRACES_KEY] = traces

        return state
