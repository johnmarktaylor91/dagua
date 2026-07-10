"""Native stress-layout support operations.

The ops in this module keep the r79 native-stress pipeline decomposed while
covering behavior that the classic stress pipelines did not need: node-size
aware target inflation, warm-started SMACOF polish, and small state resets
between independently composed optimization phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Tuple, Union

import numpy as np
import torch

from dagua.layout.ops.base import Op
from dagua.layout.ops.graph_utils import (
    _shared_all_pairs_shortest_paths,
    _shared_build_undirected_adjacency,
)
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.stress import (
    CURRENT_POSITIONS_KEY,
    CURRENT_STRESS_KEY,
    WEIGHTS_KEY,
)
from dagua.layout.ops.stress_sgd import (
    _STRESS_SGD_DEVICE_KEY,
    _STRESS_SGD_EXACT_KEY,
    _STRESS_SGD_NUM_NODES_KEY,
    _STRESS_SGD_RNG_KEY,
    _STRESS_SGD_TRACE_KEY,
    _apply_pair_update,
    _approx_distance,
    _initial_positions_from_state,
    _learning_rate,
    _resolve_sample_size,
    _sample_pairs,
    _schedule_bounds,
    _trace_snapshot,
)
from dagua.layout.ops.taxonomy import OpCategory, register_op

_MIN_TARGET_DISTANCE = 1.0e-6


def _node_bounding_radii(node_sizes: torch.Tensor | None) -> np.ndarray:
    """Return half-diagonal node radii on CPU.

    Parameters
    ----------
    node_sizes : torch.Tensor | None
        Optional node-size tensor with shape ``[N, 2]``.

    Returns
    -------
    numpy.ndarray
        Radius vector with shape ``[N]``. Empty when node sizes are absent.
    """
    if node_sizes is None or node_sizes.numel() == 0:
        return np.empty((0,), dtype=np.float64)
    sizes = node_sizes.detach().to(device="cpu", dtype=torch.float64).numpy()
    if sizes.ndim != 2 or sizes.shape[1] != 2:
        raise ValueError("node_sizes must have shape [N, 2].")
    return 0.5 * np.sqrt(np.sum(sizes * sizes, axis=1))


def _adjacent_pair_set(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    """Return unique undirected non-self edge endpoint pairs.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.

    Returns
    -------
    set[tuple[int, int]]
        Canonical ``(min_endpoint, max_endpoint)`` endpoint pairs.
    """
    if edge_index.numel() == 0:
        return set()
    edges = edge_index.detach().to(device="cpu", dtype=torch.long)
    pairs: set[tuple[int, int]] = set()
    for source, target in zip(edges[0].tolist(), edges[1].tolist()):
        left = int(source)
        right = int(target)
        if left == right:
            continue
        pairs.add((min(left, right), max(left, right)))
    return pairs


def _inflate_dense_adjacent_distances(
    distances: np.ndarray,
    radii: np.ndarray,
    edge_pairs: set[tuple[int, int]],
    scale: float,
) -> np.ndarray:
    """Inflate dense target distances for adjacent node pairs.

    Parameters
    ----------
    distances : numpy.ndarray
        Dense target-distance matrix with shape ``[N, N]``.
    radii : numpy.ndarray
        Per-node bounding radii with shape ``[N]``.
    edge_pairs : set[tuple[int, int]]
        Canonical adjacent endpoint pairs.
    scale : float
        Multiplier applied to the summed node radii.

    Returns
    -------
    numpy.ndarray
        Copy of ``distances`` with adjacent off-diagonal entries inflated.
    """
    inflated = distances.astype(np.float64, copy=True)
    for source, target in edge_pairs:
        if source >= inflated.shape[0] or target >= inflated.shape[1]:
            continue
        padding = float(scale) * float(radii[source] + radii[target])
        inflated[source, target] = max(inflated[source, target] + padding, _MIN_TARGET_DISTANCE)
        inflated[target, source] = max(inflated[target, source] + padding, _MIN_TARGET_DISTANCE)
    np.fill_diagonal(inflated, 0.0)
    return inflated


@register_op
@dataclass(frozen=True)
class ResetConvergence(Op):
    """Clear ``SolveState.converged`` before a new optimization phase."""

    name: ClassVar[str] = "reset_convergence"
    category: ClassVar[OpCategory] = OpCategory.CONTROL
    reads: ClassVar[Tuple[str, ...]] = ("converged",)
    writes: ClassVar[Tuple[str, ...]] = ("converged",)
    requires: ClassVar[Tuple[str, ...]] = ()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Reset the convergence flag.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context. Unused.

        Returns
        -------
        SolveState
            State with ``converged`` set to ``False``.
        """
        del problem, ctx
        state.converged = False
        return state


@dataclass(frozen=True)
class InflateStressTargetDistancesConfig:
    """Configuration for :class:`InflateStressTargetDistances`.

    Parameters
    ----------
    enabled : bool, default=True
        Whether to apply node-size aware inflation.
    scale : float, default=1.0
        Multiplier applied to the summed endpoint radii.
    """

    enabled: bool = True
    scale: float = 1.0


@register_op
@dataclass(frozen=True)
class InflateStressTargetDistances(Op):
    """Inflate adjacent stress targets by node bounding radii.

    The operation updates every target representation used by native-stress
    stages when present: Pivot-MDS rows, dense distance matrices, and exact
    Stress-SGD term arrays. Non-adjacent graph distances are left unchanged.
    """

    config: InflateStressTargetDistancesConfig = field(
        default_factory=InflateStressTargetDistancesConfig
    )

    name: ClassVar[str] = "inflate_stress_target_distances"
    category: ClassVar[OpCategory] = OpCategory.DISTANCE
    reads: ClassVar[Tuple[str, ...]] = (
        "pivot_indices",
        "pivot_distances",
        "distance_matrix",
        "extras",
    )
    writes: ClassVar[Tuple[str, ...]] = ("pivot_distances", "distance_matrix", "extras")
    requires: ClassVar[Tuple[str, ...]] = ()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Apply size inflation to available stress target distances.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs with optional ``node_sizes``.
        state : SolveState
            Mutable state carrying distance targets.
        ctx : RuntimeContext
            Runtime context. Unused.

        Returns
        -------
        SolveState
            State with adjacent target distances inflated when enabled.
        """
        del ctx
        if not self.config.enabled:
            return state
        if self.config.scale < 0.0:
            raise ValueError("InflateStressTargetDistances scale must be nonnegative.")

        radii = _node_bounding_radii(problem.node_sizes)
        if radii.size == 0:
            return state
        edge_pairs = _adjacent_pair_set(problem.edge_index)
        if not edge_pairs:
            return state

        self._inflate_pivot_distances(state=state, radii=radii, edge_pairs=edge_pairs)
        self._inflate_dense_matrix(state=state, radii=radii, edge_pairs=edge_pairs)
        self._inflate_exact_terms(state=state, radii=radii, edge_pairs=edge_pairs)
        return state

    def _inflate_pivot_distances(
        self,
        state: SolveState,
        radii: np.ndarray,
        edge_pairs: set[tuple[int, int]],
    ) -> None:
        """Inflate Pivot-MDS distance rows in place.

        Parameters
        ----------
        state : SolveState
            Mutable state with optional ``pivot_indices`` and ``pivot_distances``.
        radii : numpy.ndarray
            Per-node bounding radii with shape ``[N]``.
        edge_pairs : set[tuple[int, int]]
            Canonical adjacent endpoint pairs.
        """
        if state.pivot_indices is None or state.pivot_distances is None:
            return
        distances = state.pivot_distances.detach().clone()
        pivots = state.pivot_indices.detach().to(device="cpu", dtype=torch.long).tolist()
        pivot_row = {int(pivot): row for row, pivot in enumerate(pivots)}
        for source, target in edge_pairs:
            padding = float(self.config.scale) * float(radii[source] + radii[target])
            if source in pivot_row and target < distances.shape[1]:
                distances[pivot_row[source], target] += padding
            if target in pivot_row and source < distances.shape[1]:
                distances[pivot_row[target], source] += padding
        state.pivot_distances = distances.clamp(min=0.0)

    def _inflate_dense_matrix(
        self,
        state: SolveState,
        radii: np.ndarray,
        edge_pairs: set[tuple[int, int]],
    ) -> None:
        """Inflate dense all-pairs target matrices.

        Parameters
        ----------
        state : SolveState
            Mutable state with optional dense ``distance_matrix``.
        radii : numpy.ndarray
            Per-node bounding radii with shape ``[N]``.
        edge_pairs : set[tuple[int, int]]
            Canonical adjacent endpoint pairs.
        """
        if state.distance_matrix is None or state.distance_matrix.ndim != 2:
            return
        if state.distance_matrix.shape[0] != state.distance_matrix.shape[1]:
            return
        inflated = _inflate_dense_adjacent_distances(
            distances=state.distance_matrix.detach().to(device="cpu").numpy(),
            radii=radii,
            edge_pairs=edge_pairs,
            scale=float(self.config.scale),
        )
        state.distance_matrix = torch.as_tensor(
            inflated,
            dtype=state.distance_matrix.dtype,
            device=state.distance_matrix.device,
        )

    def _inflate_exact_terms(
        self,
        state: SolveState,
        radii: np.ndarray,
        edge_pairs: set[tuple[int, int]],
    ) -> None:
        """Inflate exact Stress-SGD term arrays.

        Parameters
        ----------
        state : SolveState
            Mutable state with optional exact Stress-SGD term arrays.
        radii : numpy.ndarray
            Per-node bounding radii with shape ``[N]``.
        edge_pairs : set[tuple[int, int]]
            Canonical adjacent endpoint pairs.
        """
        keys = {"stress_sgd_sources", "stress_sgd_targets", "stress_sgd_distances"}
        if not keys <= state.extras.keys():
            return
        sources = np.asarray(state.extras["stress_sgd_sources"])
        targets = np.asarray(state.extras["stress_sgd_targets"])
        distances = np.asarray(state.extras["stress_sgd_distances"]).copy()
        for index, (source, target) in enumerate(zip(sources.tolist(), targets.tolist())):
            pair = (min(int(source), int(target)), max(int(source), int(target)))
            if pair not in edge_pairs:
                continue
            distances[index] = max(
                float(distances[index])
                + float(self.config.scale) * float(radii[pair[0]] + radii[pair[1]]),
                _MIN_TARGET_DISTANCE,
            )
        state.extras["stress_sgd_distances"] = distances
        state.extras["stress_sgd_weights"] = (1.0 / np.square(distances)).astype(
            distances.dtype,
            copy=False,
        )


@dataclass(frozen=True)
class PrepareWarmStartStressMajorizationConfig:
    """Configuration for :class:`PrepareWarmStartStressMajorization`.

    Parameters
    ----------
    size_aware : bool, default=True
        Whether to inflate adjacent target distances by node radii.
    size_scale : float, default=1.0
        Multiplier applied when ``size_aware`` is enabled.
    """

    size_aware: bool = True
    size_scale: float = 1.0


@register_op
@dataclass(frozen=True)
class PrepareWarmStartStressMajorization(Op):
    """Prepare SMACOF state from the current native-stress coordinates."""

    config: PrepareWarmStartStressMajorizationConfig = field(
        default_factory=PrepareWarmStartStressMajorizationConfig
    )

    name: ClassVar[str] = "prepare_warm_start_stress_majorization"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ("distance_matrix", "laplacian", "extras")
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build dense SMACOF matrices using ``state.pos`` as the warm start.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable state containing current positions with shape ``[N, 2]``.
        ctx : RuntimeContext
            Runtime context. Unused.

        Returns
        -------
        SolveState
            State with SMACOF distances, weights, Laplacian inverse, and
            current-position cache populated.
        """
        del ctx
        if state.pos is None:
            raise ValueError("PrepareWarmStartStressMajorization requires state.pos.")
        if state.pos.shape != (problem.num_nodes, 2):
            raise ValueError("Warm-start SMACOF positions must have shape [N, 2].")

        adjacency = _shared_build_undirected_adjacency(
            edge_index=problem.edge_index,
            num_nodes=problem.num_nodes,
            edge_weights=problem.edge_weights,
        )
        raw_distances = _shared_all_pairs_shortest_paths(
            adjacency,
            weighted=problem.edge_weights is not None,
        ).astype(np.float64, copy=False)
        finite_mask = (
            np.isfinite(raw_distances) if problem.edge_weights is not None else raw_distances >= 0
        )
        max_distance = float(raw_distances[finite_mask].max()) if bool(finite_mask.any()) else 0.0
        fill_value = max_distance + 1.0 if problem.num_nodes > 1 else 0.0
        target_distances = np.where(finite_mask, raw_distances, fill_value).astype(np.float64)
        np.fill_diagonal(target_distances, 0.0)

        if self.config.size_aware:
            target_distances = _inflate_dense_adjacent_distances(
                distances=target_distances,
                radii=_node_bounding_radii(problem.node_sizes),
                edge_pairs=_adjacent_pair_set(problem.edge_index),
                scale=float(self.config.size_scale),
            )

        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.where(target_distances > 0.0, 1.0 / np.square(target_distances), 0.0)
        np.fill_diagonal(weights, 0.0)
        laplacian = -weights
        np.fill_diagonal(laplacian, weights.sum(axis=1))

        current = state.pos.detach().to(device="cpu", dtype=torch.float64).numpy().copy()
        current -= current.mean(axis=0, keepdims=True)
        deltas = current[:, None, :] - current[None, :, :]
        current_distances = np.sqrt(np.sum(deltas * deltas, axis=2))
        current_stress = 0.5 * float(
            np.sum(weights * np.square(current_distances - target_distances))
        )

        state.distance_matrix = torch.from_numpy(target_distances)
        state.laplacian = np.linalg.pinv(laplacian)
        state.extras[WEIGHTS_KEY] = weights
        state.extras[CURRENT_POSITIONS_KEY] = current
        state.extras[CURRENT_STRESS_KEY] = current_stress
        return state


@dataclass(frozen=True)
class RunWarmStartStressSGDApproximateScheduleConfig:
    """Configuration for warm-started approximate Stress-SGD.

    Parameters
    ----------
    steps : int, default=30
        Number of approximate Stress-SGD epochs.
    eps : float, default=0.01
        Final learning-rate shrinkage factor.
    sample_size : int | str, default="auto"
        Per-epoch sampled pair budget.
    trace_every : int, default=0
        Snapshot interval. ``0`` disables trace snapshots.
    auto_full_epoch_threshold : int, default=1000
        Node cutoff below which ``"auto"`` expands to a full epoch.
    auto_sample_threshold : int, default=1000
        Sample count used by ``"auto"`` above the full-epoch cutoff.
    """

    steps: int = 30
    eps: float = 0.01
    sample_size: Union[int, str] = "auto"
    trace_every: int = 0
    auto_full_epoch_threshold: int = 1_000
    auto_sample_threshold: int = 1_000


@register_op
@dataclass(frozen=True)
class RunWarmStartStressSGDApproximateSchedule(Op):
    """Run large-graph approximate Stress-SGD from current coordinates."""

    config: RunWarmStartStressSGDApproximateScheduleConfig = field(
        default_factory=RunWarmStartStressSGDApproximateScheduleConfig
    )

    name: ClassVar[str] = "native_stress_warm_start_approximate_schedule"
    category: ClassVar[OpCategory] = OpCategory.OPTIMIZE
    reads: ClassVar[Tuple[str, ...]] = ("pos", "distance_matrix", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("pos", "extras", "converged")
    requires: ClassVar[Tuple[str, ...]] = ("pos", "distance_matrix", "extras")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Refine Pivot-MDS coordinates with approximate Stress-SGD.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs. Unused after preprocessing.
        state : SolveState
            Mutable solve state carrying current positions with shape ``[N, 2]``,
            pivot distances in ``distance_matrix``, and Stress-SGD metadata.
        ctx : RuntimeContext
            Runtime context. Unused by the sequential CPU-style kernel.

        Returns
        -------
        SolveState
            State with approximate Stress-SGD positions when large-graph mode
            is active.
        """
        del problem, ctx
        if state.converged or state.extras.get(_STRESS_SGD_EXACT_KEY, True):
            return state
        if not isinstance(state.distance_matrix, torch.Tensor):
            raise ValueError("Warm-start approximate Stress-SGD requires pivot distances.")

        config = self.config
        if config.steps < 0:
            raise ValueError("steps must be nonnegative.")
        if config.eps <= 0.0:
            raise ValueError("eps must be positive.")

        num_nodes = int(state.extras[_STRESS_SGD_NUM_NODES_KEY])
        device = state.extras[_STRESS_SGD_DEVICE_KEY]
        rng = state.extras[_STRESS_SGD_RNG_KEY]
        pivot_dist = state.distance_matrix.to(dtype=torch.float64, device="cpu").numpy()
        positions = _initial_positions_from_state(state=state, num_nodes=num_nodes)
        traces: list[torch.Tensor] = []
        d_min, d_max = _schedule_bounds(pivot_dist)
        effective_sample_size = _resolve_sample_size(
            num_nodes=num_nodes,
            sample_size=config.sample_size,
            auto_full_epoch_threshold=config.auto_full_epoch_threshold,
            auto_sample_threshold=config.auto_sample_threshold,
        )
        use_full_epoch = (
            effective_sample_size >= num_nodes and num_nodes <= config.auto_full_epoch_threshold
        )
        full_sources: np.ndarray | None = None
        full_targets: np.ndarray | None = None
        full_order: np.ndarray | None = None
        if use_full_epoch:
            full_sources, full_targets = np.triu_indices(num_nodes, k=1)
            full_order = np.arange(full_sources.shape[0], dtype=np.int64)

        if config.trace_every > 0 and config.steps == 0:
            _trace_snapshot(traces, positions, device)

        for step_index in range(config.steps):
            eta = _learning_rate(
                step_index=step_index,
                steps=config.steps,
                d_min=d_min,
                d_max=d_max,
                eps=config.eps,
            )
            if use_full_epoch:
                if full_sources is None or full_targets is None or full_order is None:
                    raise RuntimeError("Warm-start approximate pair arrays were not initialized.")
                rng.shuffle(full_order)
                for pair_index in full_order:
                    source_index = int(full_sources[pair_index])
                    target_index = int(full_targets[pair_index])
                    self._apply_approximate_update(
                        positions=positions,
                        source_index=source_index,
                        target_index=target_index,
                        pivot_dist=pivot_dist,
                        eta=eta,
                    )
            else:
                sampled_sources, sampled_targets = _sample_pairs(
                    num_nodes=num_nodes,
                    sample_size=effective_sample_size,
                    rng=rng,
                )
                for source_index, target_index in zip(
                    sampled_sources.tolist(),
                    sampled_targets.tolist(),
                ):
                    self._apply_approximate_update(
                        positions=positions,
                        source_index=int(source_index),
                        target_index=int(target_index),
                        pivot_dist=pivot_dist,
                        eta=eta,
                    )
            if config.trace_every > 0 and (step_index + 1) % config.trace_every == 0:
                _trace_snapshot(traces, positions, device)

        state.pos = torch.from_numpy(positions.astype(np.float32, copy=False)).to(device=device)
        state.extras[_STRESS_SGD_TRACE_KEY] = traces
        state.converged = True
        return state

    def _apply_approximate_update(
        self,
        positions: np.ndarray,
        source_index: int,
        target_index: int,
        pivot_dist: np.ndarray,
        eta: float,
    ) -> None:
        """Apply one approximate Stress-SGD pair update.

        Parameters
        ----------
        positions : numpy.ndarray
            Mutable coordinate array with shape ``[N, 2]``.
        source_index : int
            Source endpoint.
        target_index : int
            Target endpoint.
        pivot_dist : numpy.ndarray
            Pivot distance rows with shape ``[P, N]``.
        eta : float
            Current learning-rate schedule value.

        Returns
        -------
        None
            ``positions`` is updated in place.
        """
        target_distance = _approx_distance(
            source_index=source_index,
            target_index=target_index,
            pivot_dist=pivot_dist,
        )
        weight = 1.0 / float(target_distance * target_distance)
        _apply_pair_update(
            positions=positions,
            source_index=source_index,
            target_index=target_index,
            target_distance=target_distance,
            weight=weight,
            eta=eta,
        )


__all__ = [
    "InflateStressTargetDistances",
    "InflateStressTargetDistancesConfig",
    "PrepareWarmStartStressMajorization",
    "PrepareWarmStartStressMajorizationConfig",
    "ResetConvergence",
    "RunWarmStartStressSGDApproximateSchedule",
    "RunWarmStartStressSGDApproximateScheduleConfig",
]
