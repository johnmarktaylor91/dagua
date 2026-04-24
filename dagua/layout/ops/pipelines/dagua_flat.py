"""Direction-agnostic Dagua pipeline for undirected-origin graphs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar, Optional, Tuple, Union

import numpy as np
import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.distance import PivotDistanceQueries, PivotSelection, PivotSelectionConfig
from dagua.layout.ops.embed import PivotMDSComputeCoordinates
from dagua.layout.ops.postprocess import (
    AspectRatioFit,
    AspectRatioFitConfig,
    PivotMDSFinalizePositions,
)
from dagua.layout.ops.preprocess import BuildAdjacency, BuildAdjacencyConfig
from dagua.layout.ops.project import OverlapProjection, OverlapProjectionConfig
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.stress_sgd import (
    _STRESS_SGD_DEVICE_KEY,
    _STRESS_SGD_EXACT_KEY,
    _STRESS_SGD_NUM_NODES_KEY,
    _STRESS_SGD_RNG_KEY,
    _STRESS_SGD_TRACE_KEY,
    InitializeStressSGDState,
    PrepareStressSGDTerms,
    _apply_pair_update,
    _approx_distance,
    _learning_rate,
    _resolve_sample_size,
    _sample_pairs,
    _schedule_bounds,
    _schedule_from_weights,
    _trace_snapshot,
)
from dagua.layout.ops.taxonomy import OpCategory

_DEFAULT_EPS = 0.01
_DEFAULT_MAX_EXACT_NODES = 10_000


def _initial_positions_from_state(state: SolveState, num_nodes: int) -> np.ndarray:
    """Return CPU float64 warm-start coordinates from ``state.pos``.

    Parameters
    ----------
    state : SolveState
        Mutable solve state expected to carry Pivot-MDS positions.
    num_nodes : int
        Number of nodes ``N`` expected by the Stress-SGD refinement.

    Returns
    -------
    np.ndarray
        Warm-start coordinates with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If Pivot-MDS did not provide a correctly shaped position tensor.
    """
    if state.pos is None:
        raise ValueError("dagua_flat Stress-SGD refinement requires Pivot-MDS positions.")
    if state.pos.shape != (num_nodes, 2):
        raise ValueError(
            f"dagua_flat warm start must have shape ({num_nodes}, 2), got {tuple(state.pos.shape)}."
        )
    return state.pos.detach().to(device="cpu", dtype=torch.float64).numpy().copy()


@dataclass(frozen=True)
class _WarmStartStressSGDConfig:
    """Configuration shared by warm-start Stress-SGD schedule ops.

    Parameters
    ----------
    steps : int
        Number of Stress-SGD epochs.
    eps : float
        Final schedule shrinkage factor.
    sample_size : int | str
        Approximate-mode sample budget.
    trace_every : int
        Optional trace interval.
    """

    steps: int = 30
    eps: float = _DEFAULT_EPS
    sample_size: Union[int, str] = "auto"
    trace_every: int = 0


class _WarmStartStressSGDExactSchedule:
    """Run exact Stress-SGD from existing Pivot-MDS coordinates."""

    name: ClassVar[str] = "dagua_flat_stress_sgd_exact"
    category: ClassVar[OpCategory] = OpCategory.OPTIMIZE
    reads: ClassVar[Tuple[str, ...]] = ("pos", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("pos", "extras", "converged")
    requires: ClassVar[Tuple[str, ...]] = ("pos", "extras")

    def __init__(self, config: _WarmStartStressSGDConfig) -> None:
        """Store exact warm-start schedule settings.

        Parameters
        ----------
        config : _WarmStartStressSGDConfig
            Stress-SGD schedule configuration.
        """
        self.config = config

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Refine ``state.pos`` with exact all-pairs Stress-SGD updates.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state containing Pivot-MDS coordinates.
        ctx : RuntimeContext
            Runtime context, unused by this CPU-style sequential kernel.

        Returns
        -------
        SolveState
            State with refined positions when exact mode is active.
        """
        del problem, ctx

        if state.converged or not state.extras.get(_STRESS_SGD_EXACT_KEY):
            return state

        num_nodes = int(state.extras[_STRESS_SGD_NUM_NODES_KEY])
        device = state.extras[_STRESS_SGD_DEVICE_KEY]
        rng = state.extras[_STRESS_SGD_RNG_KEY]
        sources = state.extras["stress_sgd_sources"]
        targets = state.extras["stress_sgd_targets"]
        distances = state.extras["stress_sgd_distances"]
        weights = state.extras["stress_sgd_weights"]

        positions = _initial_positions_from_state(state=state, num_nodes=num_nodes)
        traces: list[torch.Tensor] = []
        schedule = _schedule_from_weights(
            steps=self.config.steps,
            w_min=float(weights.min()),
            w_max=float(weights.max()),
            eps=self.config.eps,
        )
        order = np.arange(sources.shape[0], dtype=np.int64)

        if self.config.trace_every > 0 and self.config.steps == 0:
            _trace_snapshot(traces, positions, device)

        for step_index, eta in enumerate(schedule):
            rng.shuffle(order)
            for term_index in order:
                _apply_pair_update(
                    positions=positions,
                    source_index=int(sources[term_index]),
                    target_index=int(targets[term_index]),
                    target_distance=float(distances[term_index]),
                    weight=float(weights[term_index]),
                    eta=float(eta),
                )
            if self.config.trace_every > 0 and (step_index + 1) % self.config.trace_every == 0:
                _trace_snapshot(traces, positions, device)

        state.pos = torch.from_numpy(positions.astype(np.float32, copy=False)).to(device=device)
        state.extras[_STRESS_SGD_TRACE_KEY] = traces
        state.converged = True
        return state


class _WarmStartStressSGDApproximateSchedule:
    """Run approximate Stress-SGD from existing Pivot-MDS coordinates."""

    name: ClassVar[str] = "dagua_flat_stress_sgd_approximate"
    category: ClassVar[OpCategory] = OpCategory.OPTIMIZE
    reads: ClassVar[Tuple[str, ...]] = ("pos", "distance_matrix", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("pos", "extras", "converged")
    requires: ClassVar[Tuple[str, ...]] = ("pos", "distance_matrix", "extras")

    def __init__(self, config: _WarmStartStressSGDConfig) -> None:
        """Store approximate warm-start schedule settings.

        Parameters
        ----------
        config : _WarmStartStressSGDConfig
            Stress-SGD schedule configuration.
        """
        self.config = config

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Refine ``state.pos`` with pivot-approximate Stress-SGD updates.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state containing Pivot-MDS coordinates.
        ctx : RuntimeContext
            Runtime context, unused by this CPU-style sequential kernel.

        Returns
        -------
        SolveState
            State with refined positions when approximate mode is active.
        """
        del problem, ctx

        if state.converged or state.extras.get(_STRESS_SGD_EXACT_KEY, True):
            return state
        if not isinstance(state.distance_matrix, torch.Tensor):
            raise ValueError("dagua_flat approximate Stress-SGD requires pivot distances.")

        num_nodes = int(state.extras[_STRESS_SGD_NUM_NODES_KEY])
        device = state.extras[_STRESS_SGD_DEVICE_KEY]
        rng = state.extras[_STRESS_SGD_RNG_KEY]
        pivot_dist = state.distance_matrix.to(dtype=torch.float64, device="cpu").numpy()
        positions = _initial_positions_from_state(state=state, num_nodes=num_nodes)
        traces: list[torch.Tensor] = []
        d_min, d_max = _schedule_bounds(pivot_dist)
        effective_sample_size = _resolve_sample_size(
            num_nodes=num_nodes,
            sample_size=self.config.sample_size,
            auto_full_epoch_threshold=1_000,
            auto_sample_threshold=1_000,
        )
        use_full_epoch = effective_sample_size >= num_nodes and num_nodes <= 1_000
        full_sources: Optional[np.ndarray] = None
        full_targets: Optional[np.ndarray] = None
        full_order: Optional[np.ndarray] = None
        if use_full_epoch:
            full_sources, full_targets = np.triu_indices(num_nodes, k=1)
            full_order = np.arange(full_sources.shape[0], dtype=np.int64)

        if self.config.trace_every > 0 and self.config.steps == 0:
            _trace_snapshot(traces, positions, device)

        for step_index in range(self.config.steps):
            eta = _learning_rate(
                step_index=step_index,
                steps=self.config.steps,
                d_min=d_min,
                d_max=d_max,
                eps=self.config.eps,
            )
            if use_full_epoch:
                if full_sources is None or full_targets is None or full_order is None:
                    raise RuntimeError("dagua_flat full-epoch pair arrays were not initialized.")
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
            if self.config.trace_every > 0 and (step_index + 1) % self.config.trace_every == 0:
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
        positions : np.ndarray
            Mutable coordinate array with shape ``[N, 2]``.
        source_index : int
            Source node index.
        target_index : int
            Target node index.
        pivot_dist : np.ndarray
            Pivot-to-node distances with shape ``[P, N]``.
        eta : float
            Current schedule step size.
        """
        target_distance = _approx_distance(
            source_index=source_index,
            target_index=target_index,
            pivot_dist=pivot_dist,
        )
        weight = 1.0 / float(math.pow(target_distance, 2))
        _apply_pair_update(
            positions=positions,
            source_index=source_index,
            target_index=target_index,
            target_distance=target_distance,
            weight=weight,
            eta=eta,
        )


def build_dagua_flat_pipeline(
    steps: int = 30,
    n_pivots: int = 50,
    eps: float = _DEFAULT_EPS,
    max_exact_nodes: int = _DEFAULT_MAX_EXACT_NODES,
    sample_size: Union[int, str] = "auto",
    overlap_padding: float = 2.0,
    overlap_iterations: int = 10,
    target_aspect: Optional[float] = None,
) -> Pipeline:
    """Build the undirected ``dagua_flat`` pipeline.

    Parameters
    ----------
    steps : int, default=30
        Number of warm-start Stress-SGD refinement epochs.
    n_pivots : int, default=50
        Maximum number of Pivot-MDS pivots used for initialization.
    eps : float, default=0.01
        Final Stress-SGD schedule shrinkage factor.
    max_exact_nodes : int, default=10000
        Node cutoff for exact Stress-SGD terms.
    sample_size : int | str, default="auto"
        Approximate-mode sample budget.
    overlap_padding : float, default=2.0
        Padding used by the final overlap projection.
    overlap_iterations : int, default=10
        Number of final overlap projection passes.
    target_aspect : float, optional
        Optional target aspect ratio forwarded to ``AspectRatioFit``.

    Returns
    -------
    Pipeline
        Pipeline composed as Pivot-MDS init, Stress-SGD refinement,
        overlap projection, and aspect-ratio fit.

    Raises
    ------
    ValueError
        If public numeric arguments are invalid.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if n_pivots <= 0:
        raise ValueError("n_pivots must be positive.")
    if eps <= 0.0:
        raise ValueError("eps must be positive.")
    if max_exact_nodes < 0:
        raise ValueError("max_exact_nodes must be non-negative.")
    if sample_size != "auto" and (not isinstance(sample_size, int) or sample_size <= 0):
        raise ValueError("sample_size must be a positive integer or 'auto'.")

    stress_config = _WarmStartStressSGDConfig(steps=steps, eps=eps, sample_size=sample_size)
    return Pipeline(
        [
            BuildAdjacency(BuildAdjacencyConfig(weighted=False, dedup="min", format="list")),
            PivotSelection(PivotSelectionConfig(n_pivots=n_pivots)),
            PivotDistanceQueries(),
            PivotMDSComputeCoordinates(),
            PivotMDSFinalizePositions(),
            BuildAdjacency(
                BuildAdjacencyConfig(weighted=True, dedup="min", format="list", directed=False)
            ),
            InitializeStressSGDState(),
            PrepareStressSGDTerms(max_exact_nodes=max_exact_nodes),
            _WarmStartStressSGDExactSchedule(stress_config),
            _WarmStartStressSGDApproximateSchedule(stress_config),
            OverlapProjection(
                OverlapProjectionConfig(
                    padding=overlap_padding,
                    iterations=overlap_iterations,
                )
            ),
            AspectRatioFit(AspectRatioFitConfig(target_aspect=target_aspect)),
        ],
        name="dagua_flat_pipeline",
    )


def layout_dagua_flat_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 30,
    n_pivots: int = 50,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    eps: float = _DEFAULT_EPS,
    max_exact_nodes: int = _DEFAULT_MAX_EXACT_NODES,
    sample_size: Union[int, str] = "auto",
    overlap_padding: float = 2.0,
    overlap_iterations: int = 10,
    target_aspect: Optional[float] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Run the ``dagua_flat`` pipeline for undirected-origin graphs.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``.
    steps : int, default=30
        Number of warm-start Stress-SGD refinement epochs.
    n_pivots : int, default=50
        Maximum number of Pivot-MDS pivots used for initialization.
    seed : int, default=42
        Random seed consumed by Pivot-MDS and Stress-SGD.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.
    eps : float, default=0.01
        Final Stress-SGD schedule shrinkage factor.
    max_exact_nodes : int, default=10000
        Node cutoff for exact Stress-SGD terms.
    sample_size : int | str, default="auto"
        Approximate-mode sample budget.
    overlap_padding : float, default=2.0
        Padding used by the final overlap projection.
    overlap_iterations : int, default=10
        Number of final overlap projection passes.
    target_aspect : float, optional
        Optional target aspect ratio forwarded to ``AspectRatioFit``.
    **kwargs : Any
        Ignored compatibility keywords from generic dispatchers.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If public inputs are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    del kwargs

    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
            )

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    output_device = (
        edge_index.device
        if edge_index.numel() > 0
        else node_sizes.device
        if node_sizes is not None
        else torch.device("cpu")
    )
    ctx = RuntimeContext(plan=ExecutionPlan(device=str(output_device)))
    final_state = build_dagua_flat_pipeline(
        steps=steps,
        n_pivots=n_pivots,
        eps=eps,
        max_exact_nodes=max_exact_nodes,
        sample_size=sample_size,
        overlap_padding=overlap_padding,
        overlap_iterations=overlap_iterations,
        target_aspect=target_aspect,
    ).apply(problem, SolveState(), ctx)
    if final_state.pos is None:
        raise RuntimeError("dagua_flat pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_dagua_flat_pipeline", "layout_dagua_flat_pipeline"]
