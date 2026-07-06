"""Multilevel scale path for the native stress layout core."""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch

from dagua.config import LayoutConfig
from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.coarsen import AggressiveHybridCoarsen, AggressiveHybridCoarsenConfig
from dagua.layout.ops.distance import PivotDistanceQueries, PivotSelection, PivotSelectionConfig
from dagua.layout.ops.embed import PivotMDSComputeCoordinates
from dagua.layout.ops.native_stress import (
    InflateStressTargetDistances,
    InflateStressTargetDistancesConfig,
    ResetConvergence,
    RunWarmStartStressSGDApproximateSchedule,
    RunWarmStartStressSGDApproximateScheduleConfig,
)
from dagua.layout.ops.pipelines.native_stress import (
    NativeStressConfig,
    _config_from_public,
    _layout_connected_native_stress,
    layout_native_stress_pipeline,
)
from dagua.layout.ops.postprocess import (
    AspectRatioFit,
    AspectRatioFitConfig,
    PivotMDSFinalizePositions,
)
from dagua.layout.ops.preprocess import BuildAdjacency, BuildAdjacencyConfig
from dagua.layout.ops.project import OverlapProjection, OverlapProjectionConfig
from dagua.layout.ops.prolong import (
    DirectMapping,
    DirectMappingConfig,
    NeighborSmoothing,
    NeighborSmoothingConfig,
)
from dagua.layout.ops.spatial_hash import cell_list_candidate_pairs
from dagua.layout.ops.state import (
    ExecutionPlan,
    HierarchyLevel,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)
from dagua.layout.ops.stress_sgd import (
    InitializeStressSGDState,
    PrepareStressSGDTerms,
    RunStressSGDExactSchedule,
)
from dagua.layout.ops.taxonomy import OpCategory, register_op
from dagua.layout.resolve import normalize_node_sizes

_DEFAULT_ML_MIN_NODES = 5_000
_DEFAULT_ML_MIN_EDGES = 50_000
_DEFAULT_COARSEST_NODES = 1_000
_DEFAULT_MAX_LEVELS = 20
_DEFAULT_HASH_REPULSION_NODES = 100_000
_DEFAULT_OVERLAP_MAX_NODES = 5_000
_MEMORY_ABORT_FRACTION = 0.70


@dataclass(frozen=True)
class NativeStressMLConfig:
    """Configuration for ``native_stress_ml``.

    Parameters
    ----------
    ml_min_nodes : int, default=5000
        Node sketch threshold that enables multilevel layout.
    ml_min_edges : int, default=50000
        Edge sketch threshold that enables multilevel layout.
    coarsest_nodes : int, default=1000
        Target maximum node count for the coarsest solve.
    max_levels : int, default=20
        Maximum number of fine-to-coarse transitions used by the V-cycle.
    coarse_steps : int, default=0
        Coarsest native-stress step count. ``0`` uses native-stress auto
        defaults.
    refine_steps : int, default=12
        Short warm-start Stress-SGD refinement budget per prolongation level.
    refine_sample_size : int | str, default="auto"
        Pair sample budget for warm-start approximate refinement.
    repulsion_mode : str, default="auto"
        Large-level repulsion selector: ``"auto"``, ``"spatial_hash"``,
        ``"barnes_hut"``, ``"negative_sampling"``, or ``"none"``.
    hash_repulsion_nodes : int, default=100000
        Node cutoff below which the auto selector uses spatial-hash pairs.
    overlap_max_nodes : int, default=100000
        Maximum final node count for direct overlap projection.
    coarsen_min_shrink_ratio : float, default=0.50
        Fractional shrink threshold below which coarsening escalates from
        heavy-edge matching to hub-star contraction.
    seed : int, default=42
        Deterministic seed.
    """

    ml_min_nodes: int = _DEFAULT_ML_MIN_NODES
    ml_min_edges: int = _DEFAULT_ML_MIN_EDGES
    coarsest_nodes: int = _DEFAULT_COARSEST_NODES
    max_levels: int = _DEFAULT_MAX_LEVELS
    coarse_steps: int = 0
    refine_steps: int = 12
    refine_sample_size: Union[int, str] = "auto"
    repulsion_mode: str = "auto"
    hash_repulsion_nodes: int = _DEFAULT_HASH_REPULSION_NODES
    overlap_max_nodes: int = _DEFAULT_OVERLAP_MAX_NODES
    coarsen_min_shrink_ratio: float = 0.50
    seed: int = 42


def should_use_native_stress_ml(
    num_nodes: int,
    num_edges: int,
    config: Optional[NativeStressMLConfig] = None,
) -> bool:
    """Return whether the sketch gate selects the multilevel path.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the input graph.
    num_edges : int
        Number of edges in the input graph.
    config : NativeStressMLConfig, optional
        Multilevel threshold configuration.

    Returns
    -------
    bool
        ``True`` when either the node or edge threshold is reached.
    """
    resolved = config or NativeStressMLConfig()
    return int(num_nodes) >= resolved.ml_min_nodes or int(num_edges) >= resolved.ml_min_edges


def _available_ram_bytes() -> int:
    """Return currently available system RAM in bytes.

    Returns
    -------
    int
        Available RAM according to :mod:`psutil`.

    Raises
    ------
    RuntimeError
        If ``psutil`` is unavailable.
    """
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError("native_stress_ml requires psutil for memory budget checks.") from exc
    return int(psutil.virtual_memory().available)


def _assert_memory_budget(
    stage: str,
    num_nodes: int,
    num_edges: int,
    bytes_per_node: int,
    bytes_per_edge: int,
    temporary_factor: float = 3.0,
) -> None:
    """Abort before an O(N)/O(E) stage if estimated peak memory is unsafe.

    Parameters
    ----------
    stage : str
        Human-readable stage name for diagnostics.
    num_nodes : int
        Node count touched by the stage.
    num_edges : int
        Edge count touched by the stage.
    bytes_per_node : int
        Estimated persistent bytes per node.
    bytes_per_edge : int
        Estimated persistent bytes per edge.
    temporary_factor : float, default=3.0
        Multiplier for temporaries, dtype promotions, and allocator overhead.

    Returns
    -------
    None
        The function raises on unsafe estimates.
    """
    base = (max(num_nodes, 0) * bytes_per_node) + (max(num_edges, 0) * bytes_per_edge)
    peak = int(float(base) * float(temporary_factor))
    available = _available_ram_bytes()
    budget = int(float(available) * _MEMORY_ABORT_FRACTION)
    if peak > budget:
        raise MemoryError(
            f"{stage} estimated peak {peak / 1e9:.2f} GB exceeds "
            f"{_MEMORY_ABORT_FRACTION:.0%} of available RAM ({budget / 1e9:.2f} GB)."
        )


def _log_stage(stage: str, num_nodes: int, num_edges: int) -> None:
    """Log a potentially long multilevel stage before it starts.

    Parameters
    ----------
    stage : str
        Stage name.
    num_nodes : int
        Node count.
    num_edges : int
        Edge count.

    Returns
    -------
    None
        The message is written to stderr.
    """
    print(
        f"native_stress_ml: {stage} (N={int(num_nodes)}, E={int(num_edges)})",
        file=sys.stderr,
        flush=True,
    )


def _resolve_ml_config(
    config: Optional[Union[NativeStressMLConfig, LayoutConfig]],
    seed: int,
) -> NativeStressMLConfig:
    """Resolve public algorithm parameters into a multilevel config.

    Parameters
    ----------
    config : NativeStressMLConfig or LayoutConfig, optional
        Explicit multilevel config or public layout config.
    seed : int
        Fallback seed.

    Returns
    -------
    NativeStressMLConfig
        Resolved multilevel configuration.
    """
    if isinstance(config, NativeStressMLConfig):
        return config
    params = getattr(config, "algorithm_params", {}) if config is not None else {}
    public_seed = getattr(config, "seed", seed) if config is not None else seed
    resolved = NativeStressMLConfig(
        ml_min_nodes=int(params.get("ml_min_nodes", _DEFAULT_ML_MIN_NODES)),
        ml_min_edges=int(params.get("ml_min_edges", _DEFAULT_ML_MIN_EDGES)),
        coarsest_nodes=int(params.get("coarsest_nodes", _DEFAULT_COARSEST_NODES)),
        max_levels=int(params.get("max_levels", _DEFAULT_MAX_LEVELS)),
        coarse_steps=int(params.get("coarse_steps", 0)),
        refine_steps=int(params.get("refine_steps", 12)),
        refine_sample_size=params.get("refine_sample_size", "auto"),
        repulsion_mode=str(params.get("ml_repulsion_mode", params.get("repulsion_mode", "auto"))),
        hash_repulsion_nodes=int(params.get("hash_repulsion_nodes", _DEFAULT_HASH_REPULSION_NODES)),
        overlap_max_nodes=int(params.get("overlap_max_nodes", _DEFAULT_OVERLAP_MAX_NODES)),
        coarsen_min_shrink_ratio=float(params.get("coarsen_min_shrink_ratio", 0.50)),
        seed=int(public_seed if public_seed is not None else 42),
    )
    if resolved.repulsion_mode not in {
        "auto",
        "spatial_hash",
        "barnes_hut",
        "negative_sampling",
        "none",
    }:
        raise ValueError(
            "repulsion_mode must be 'auto', 'spatial_hash', 'barnes_hut', "
            "'negative_sampling', or 'none'."
        )
    if resolved.refine_steps < 0:
        raise ValueError("refine_steps must be nonnegative.")
    if resolved.max_levels < 0:
        raise ValueError("max_levels must be nonnegative.")
    if resolved.coarsen_min_shrink_ratio < 0.0 or resolved.coarsen_min_shrink_ratio >= 1.0:
        raise ValueError("coarsen_min_shrink_ratio must be in [0, 1).")
    return resolved


def _native_config_for_stage(
    num_nodes: int,
    public_config: Optional[Union[NativeStressMLConfig, LayoutConfig]],
    seed: int,
    target_aspect: Optional[float],
    steps: int,
    overlap_iterations: int,
) -> NativeStressConfig:
    """Build a native-stress config for one multilevel stage.

    Parameters
    ----------
    num_nodes : int
        Active level node count.
    public_config : NativeStressMLConfig or LayoutConfig, optional
        User configuration.
    seed : int
        Deterministic seed.
    target_aspect : float, optional
        Aspect target for the final fit.
    steps : int
        Requested Stress-SGD steps, with ``0`` preserving auto defaults.
    overlap_iterations : int
        Final overlap iterations inside native-stress; refinement passes use
        ``0`` to avoid repeated O(N^2) projection.

    Returns
    -------
    NativeStressConfig
        Resolved native-stress stage config.
    """
    base = _config_from_public(
        num_nodes=num_nodes,
        config=public_config if isinstance(public_config, LayoutConfig) else None,
        seed=seed,
        target_aspect=target_aspect,
    )
    return NativeStressConfig(
        steps=steps if steps > 0 else base.steps,
        late_steps=1,
        n_pivots=base.n_pivots,
        eps=base.eps,
        max_exact_nodes=min(base.max_exact_nodes, 1_000),
        sample_size=base.sample_size,
        size_aware=base.size_aware,
        size_scale=base.size_scale,
        repulsion_mode="none",
        smacof_iters=0,
        smacof_max_nodes=base.smacof_max_nodes,
        overlap_padding=base.overlap_padding,
        overlap_iterations=overlap_iterations,
        target_aspect=target_aspect,
        seed=seed,
    )


def _level_problem(
    original: LayoutProblem,
    levels: list[HierarchyLevel],
    level_index: int,
) -> LayoutProblem:
    """Return the fine graph for one hierarchy transition.

    Parameters
    ----------
    original : LayoutProblem
        Original finest problem.
    levels : list[HierarchyLevel]
        Selected hierarchy levels from finest to coarsest.
    level_index : int
        Index of the transition whose fine graph should be materialized.

    Returns
    -------
    LayoutProblem
        Fine-level problem for refinement after prolongation.
    """
    if level_index == 0:
        return original
    previous = levels[level_index - 1]
    if previous.edge_index is None or previous.node_sizes is None:
        raise ValueError("Selected hierarchy level is missing graph payloads.")
    return LayoutProblem(
        edge_index=previous.edge_index,
        num_nodes=previous.num_nodes,
        node_sizes=previous.node_sizes,
        edge_weights=previous.edge_weights,
        seed=original.seed + level_index,
    )


def _coarsest_problem(original: LayoutProblem, levels: list[HierarchyLevel]) -> LayoutProblem:
    """Return the coarsest problem represented by selected levels.

    Parameters
    ----------
    original : LayoutProblem
        Original finest problem.
    levels : list[HierarchyLevel]
        Selected hierarchy levels from finest to coarsest.

    Returns
    -------
    LayoutProblem
        Coarsest layout problem.
    """
    if not levels:
        return original
    coarsest = levels[-1]
    if coarsest.edge_index is None or coarsest.node_sizes is None:
        raise ValueError("Coarsest hierarchy level is missing graph payloads.")
    return LayoutProblem(
        edge_index=coarsest.edge_index,
        num_nodes=coarsest.num_nodes,
        node_sizes=coarsest.node_sizes,
        edge_weights=coarsest.edge_weights,
        seed=original.seed + len(levels),
    )


def _select_levels(
    hierarchy: list[HierarchyLevel],
    config: NativeStressMLConfig,
) -> list[HierarchyLevel]:
    """Select a bounded finest-to-coarsest prefix from a full hierarchy.

    Parameters
    ----------
    hierarchy : list[HierarchyLevel]
        Full heavy-edge hierarchy.
    config : NativeStressMLConfig
        Coarsest target and level cap.

    Returns
    -------
    list[HierarchyLevel]
        Selected levels.
    """
    selected: list[HierarchyLevel] = []
    for level in hierarchy:
        if len(selected) >= config.max_levels:
            break
        selected.append(level)
        if level.num_nodes <= config.coarsest_nodes:
            break
    return selected


def _build_refine_pipeline(config: NativeStressConfig, ml_config: NativeStressMLConfig) -> Pipeline:
    """Build the short warm-start refinement pipeline.

    Parameters
    ----------
    config : NativeStressConfig
        Native-stress settings for the active level.
    ml_config : NativeStressMLConfig
        Multilevel settings carrying the refinement sample-size override.

    Returns
    -------
    Pipeline
        Warm-start Stress-SGD pipeline that preserves ``state.pos``.
    """
    return Pipeline(
        [
            BuildAdjacency(
                BuildAdjacencyConfig(weighted=True, dedup="min", format="list", directed=False)
            ),
            InitializeStressSGDState(independent_shuffle_rng=True),
            PrepareStressSGDTerms(max_exact_nodes=config.max_exact_nodes),
            InflateStressTargetDistances(
                InflateStressTargetDistancesConfig(
                    enabled=config.size_aware,
                    scale=config.size_scale,
                )
            ),
            RunStressSGDExactSchedule(steps=config.steps, eps=config.eps),
            RunWarmStartStressSGDApproximateSchedule(
                RunWarmStartStressSGDApproximateScheduleConfig(
                    steps=config.steps,
                    eps=config.eps,
                    sample_size=ml_config.refine_sample_size,
                )
            ),
            ResetConvergence(),
        ],
        name="native_stress_ml_refine",
    )


def _build_coarsest_pipeline(
    config: NativeStressConfig,
    ml_config: NativeStressMLConfig,
) -> Pipeline:
    """Build the minimal coarsest-level native-stress pipeline.

    Parameters
    ----------
    config : NativeStressConfig
        Native-stress settings for the coarsest level.
    ml_config : NativeStressMLConfig
        Multilevel settings carrying the sample-size override.

    Returns
    -------
    Pipeline
        PivotMDS initialization plus Stress-SGD, without late crossing polish.
    """
    return Pipeline(
        [
            BuildAdjacency(
                BuildAdjacencyConfig(weighted=True, dedup="min", format="list", directed=False)
            ),
            PivotSelection(PivotSelectionConfig(n_pivots=config.n_pivots)),
            PivotDistanceQueries(),
            InflateStressTargetDistances(
                InflateStressTargetDistancesConfig(
                    enabled=config.size_aware,
                    scale=config.size_scale,
                )
            ),
            PivotMDSComputeCoordinates(),
            PivotMDSFinalizePositions(),
            InitializeStressSGDState(independent_shuffle_rng=True),
            PrepareStressSGDTerms(max_exact_nodes=config.max_exact_nodes),
            InflateStressTargetDistances(
                InflateStressTargetDistancesConfig(
                    enabled=config.size_aware,
                    scale=config.size_scale,
                )
            ),
            RunStressSGDExactSchedule(steps=config.steps, eps=config.eps),
            RunWarmStartStressSGDApproximateSchedule(
                RunWarmStartStressSGDApproximateScheduleConfig(
                    steps=config.steps,
                    eps=config.eps,
                    sample_size=ml_config.refine_sample_size,
                )
            ),
            ResetConvergence(),
        ],
        name="native_stress_ml_coarsest",
    )


@dataclass(frozen=True)
class SampledCoarsestSolveConfig:
    """Configuration for :class:`SampledCoarsestSolve`.

    Parameters
    ----------
    target_nodes : int
        Maximum sampled pivot subset size.
    native_config : NativeStressConfig
        Native-stress configuration used for the sampled exact solve.
    ml_config : NativeStressMLConfig
        Multilevel settings carrying the approximate sample-size override.
    """

    target_nodes: int
    native_config: NativeStressConfig
    ml_config: NativeStressMLConfig


def _sampled_coarsest_indices(problem: LayoutProblem, target_nodes: int) -> torch.Tensor:
    """Select deterministic hub-and-stride pivots for a coarse fallback solve.

    Parameters
    ----------
    problem : LayoutProblem
        Coarsest graph that remains above the exact-solve cap.
    target_nodes : int
        Maximum number of sampled nodes.

    Returns
    -------
    torch.Tensor
        Sorted sampled node indices with shape ``[K]``.
    """
    sample_count = min(max(int(target_nodes), 1), problem.num_nodes)
    if sample_count >= problem.num_nodes:
        return torch.arange(problem.num_nodes, dtype=torch.long)

    edge_index = problem.edge_index.detach().to(device="cpu", dtype=torch.long)
    degrees = torch.zeros((problem.num_nodes,), dtype=torch.long)
    if edge_index.numel() > 0:
        degrees.scatter_add_(0, edge_index[0], torch.ones_like(edge_index[0]))
        degrees.scatter_add_(0, edge_index[1], torch.ones_like(edge_index[1]))

    hub_count = min(max(sample_count // 4, 1), sample_count)
    hub_order = torch.argsort(degrees, descending=True, stable=True)[:hub_count]
    selected = torch.zeros((problem.num_nodes,), dtype=torch.bool)
    selected[hub_order] = True

    if int(selected.sum().item()) < sample_count:
        stride_positions = (
            torch.linspace(
                0,
                problem.num_nodes - 1,
                steps=sample_count * 2,
                dtype=torch.float64,
            )
            .round()
            .to(dtype=torch.long)
        )
        for node in stride_positions.tolist():
            selected[int(node)] = True
            if int(selected.sum().item()) >= sample_count:
                break

    if int(selected.sum().item()) < sample_count:
        remaining = torch.nonzero(~selected, as_tuple=False).squeeze(1)
        need = sample_count - int(selected.sum().item())
        selected[remaining[:need]] = True

    return torch.nonzero(selected, as_tuple=False).squeeze(1)


def _induced_sample_problem(problem: LayoutProblem, sample_indices: torch.Tensor) -> LayoutProblem:
    """Create the induced sampled problem for fallback coarsest solving.

    Parameters
    ----------
    problem : LayoutProblem
        Original oversized coarsest problem.
    sample_indices : torch.Tensor
        Sampled original node ids with shape ``[K]``.

    Returns
    -------
    LayoutProblem
        Induced problem whose node ids are local to ``sample_indices``.
    """
    sample_cpu = sample_indices.to(device="cpu", dtype=torch.long)
    remap = torch.full((problem.num_nodes,), -1, dtype=torch.long)
    remap[sample_cpu] = torch.arange(sample_cpu.shape[0], dtype=torch.long)
    edge_index = problem.edge_index.detach().to(device="cpu", dtype=torch.long)
    if edge_index.numel() == 0:
        sample_edges = torch.empty((2, 0), dtype=torch.long)
        sample_weights = None
    else:
        local_src = remap[edge_index[0]]
        local_tgt = remap[edge_index[1]]
        keep = (local_src >= 0) & (local_tgt >= 0) & (local_src != local_tgt)
        sample_edges = torch.stack([local_src[keep], local_tgt[keep]], dim=0)
        sample_weights = (
            None
            if problem.edge_weights is None
            else problem.edge_weights.detach().to(device="cpu", dtype=torch.float32)[keep]
        )

    sample_sizes = (
        None
        if problem.node_sizes is None
        else problem.node_sizes.detach().to(device="cpu")[sample_cpu].clone()
    )
    return LayoutProblem(
        edge_index=sample_edges,
        num_nodes=int(sample_cpu.shape[0]),
        node_sizes=sample_sizes,
        edge_weights=sample_weights,
        seed=problem.seed,
    )


def _deterministic_grid_positions(
    num_nodes: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return deterministic fallback positions on a square grid.

    Parameters
    ----------
    num_nodes : int
        Number of positions to generate.
    dtype : torch.dtype, default=torch.float32
        Output dtype.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    if num_nodes <= 0:
        return torch.empty((0, 2), dtype=dtype)
    width = int(torch.ceil(torch.sqrt(torch.tensor(float(num_nodes)))).item())
    ids = torch.arange(num_nodes, dtype=torch.long)
    x_pos = torch.remainder(ids, width).to(dtype=dtype)
    y_pos = torch.div(ids, width, rounding_mode="floor").to(dtype=dtype)
    return torch.stack([x_pos, y_pos], dim=1)


def _interpolate_from_sample(
    problem: LayoutProblem,
    sample_indices: torch.Tensor,
    sample_pos: torch.Tensor,
    max_passes: int = 8,
) -> torch.Tensor:
    """Place unsampled coarse nodes from weighted neighbor positions.

    Parameters
    ----------
    problem : LayoutProblem
        Oversized coarsest problem.
    sample_indices : torch.Tensor
        Original node ids solved exactly with shape ``[K]``.
    sample_pos : torch.Tensor
        Solved sampled positions with shape ``[K, 2]``.
    max_passes : int, default=8
        Maximum graph-neighbor propagation passes before grid fallback.

    Returns
    -------
    torch.Tensor
        Interpolated full position tensor with shape ``[N, 2]``.
    """
    positions = torch.zeros((problem.num_nodes, 2), dtype=torch.float32)
    placed = torch.zeros((problem.num_nodes,), dtype=torch.bool)
    sample_cpu = sample_indices.to(device="cpu", dtype=torch.long)
    positions[sample_cpu] = sample_pos.detach().to(device="cpu", dtype=torch.float32)
    placed[sample_cpu] = True

    edge_index = problem.edge_index.detach().to(device="cpu", dtype=torch.long)
    if edge_index.numel() > 0:
        src = edge_index[0]
        tgt = edge_index[1]
        weights = (
            torch.ones((edge_index.shape[1],), dtype=torch.float32)
            if problem.edge_weights is None
            else problem.edge_weights.detach().to(device="cpu", dtype=torch.float32)
        )
        for _pass_index in range(max_passes):
            unplaced_before = int((~placed).sum().item())
            if unplaced_before == 0:
                break
            sums = torch.zeros_like(positions)
            counts = torch.zeros((problem.num_nodes,), dtype=torch.float32)

            src_to_tgt = placed[src] & ~placed[tgt]
            if bool(src_to_tgt.any()):
                weighted = weights[src_to_tgt].unsqueeze(1)
                sums.index_add_(0, tgt[src_to_tgt], positions[src[src_to_tgt]] * weighted)
                counts.index_add_(0, tgt[src_to_tgt], weights[src_to_tgt])

            tgt_to_src = placed[tgt] & ~placed[src]
            if bool(tgt_to_src.any()):
                weighted = weights[tgt_to_src].unsqueeze(1)
                sums.index_add_(0, src[tgt_to_src], positions[tgt[tgt_to_src]] * weighted)
                counts.index_add_(0, src[tgt_to_src], weights[tgt_to_src])

            newly_placed = (~placed) & (counts > 0.0)
            if not bool(newly_placed.any()):
                break
            positions[newly_placed] = sums[newly_placed] / counts[newly_placed].unsqueeze(1)
            placed[newly_placed] = True

    if not bool(placed.all()):
        fallback = _deterministic_grid_positions(problem.num_nodes)
        positions[~placed] = fallback[~placed]

    return positions


@register_op
class SampledCoarsestSolve(Op):
    """Solve a bounded sampled subset and interpolate the remaining nodes."""

    name = "sampled_coarsest_solve"
    category = OpCategory.OPTIMIZE
    writes = ("pos",)

    def __init__(self, config: SampledCoarsestSolveConfig) -> None:
        """Store sampled-solve settings.

        Parameters
        ----------
        config : SampledCoarsestSolveConfig
            Target size and native-stress solver settings.

        Returns
        -------
        None
            The operation stores the frozen config.
        """
        self.config = config

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run sampled fallback and write full coarsest positions.

        Parameters
        ----------
        problem : LayoutProblem
            Oversized coarsest graph.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context reused by the sampled exact solve.

        Returns
        -------
        SolveState
            State with ``pos`` shaped ``[problem.num_nodes, 2]``.
        """
        config = self.config
        sample_indices = _sampled_coarsest_indices(problem, config.target_nodes)
        sample_problem = _induced_sample_problem(problem, sample_indices)
        if sample_problem.edge_index.numel() == 0:
            sample_pos = _deterministic_grid_positions(sample_problem.num_nodes)
        else:
            sample_state = _build_coarsest_pipeline(
                config.native_config,
                config.ml_config,
            ).apply(sample_problem, SolveState(), ctx)
            if sample_state.pos is None:
                raise RuntimeError("sampled coarsest solve did not produce positions.")
            sample_pos = sample_state.pos
        state.pos = _interpolate_from_sample(problem, sample_indices, sample_pos)
        state.extras["sampled_coarsest_nodes"] = int(sample_indices.shape[0])
        return state


def _selected_repulsion_mode(num_nodes: int, config: NativeStressMLConfig) -> str:
    """Resolve the refinement repulsion approximation mode.

    Parameters
    ----------
    num_nodes : int
        Active level node count.
    config : NativeStressMLConfig
        Repulsion selector configuration.

    Returns
    -------
    str
        Concrete repulsion mode.
    """
    if config.repulsion_mode != "auto":
        return config.repulsion_mode
    if num_nodes < config.hash_repulsion_nodes:
        return "spatial_hash"
    return "negative_sampling"


def _apply_local_repulsion(
    state: SolveState,
    problem: LayoutProblem,
    mode: str,
    seed: int,
) -> None:
    """Apply a bounded local repulsion nudge before stress refinement.

    Parameters
    ----------
    state : SolveState
        Current solve state with positions.
    problem : LayoutProblem
        Active level problem.
    mode : str
        Repulsion approximation mode.
    seed : int
        Deterministic seed for sampled negative pairs.

    Returns
    -------
    None
        ``state.pos`` is updated in place when candidate pairs exist.
    """
    if state.pos is None or mode == "none" or problem.num_nodes <= 1:
        return
    positions = state.pos
    if mode == "spatial_hash":
        mean_size = (
            float(problem.node_sizes.detach().to(dtype=torch.float32, device="cpu").mean().item())
            if problem.node_sizes is not None and problem.node_sizes.numel() > 0
            else 1.0
        )
        pairs = cell_list_candidate_pairs(positions, cutoff_radius=max(mean_size * 2.0, 1.0))
    else:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        sample_count = min(max(problem.num_nodes * 4, 1), 200_000)
        left = torch.randint(0, problem.num_nodes, (sample_count,), generator=generator)
        right = torch.randint(0, problem.num_nodes, (sample_count,), generator=generator)
        keep = left != right
        pairs = torch.stack([left[keep], right[keep]], dim=0).to(device=positions.device)
    if pairs.numel() == 0:
        return
    if pairs.shape[1] > 200_000:
        pairs = pairs[:, :200_000]
    left_pos = positions[pairs[0]]
    right_pos = positions[pairs[1]]
    delta = left_pos - right_pos
    dist = torch.linalg.vector_norm(delta, dim=1).clamp(min=1.0e-3)
    strength = (1.0 / dist.clamp(min=1.0)).unsqueeze(1) * 0.01
    update = delta / dist.unsqueeze(1) * strength
    nudged = torch.zeros_like(positions)
    nudged.index_add_(0, pairs[0], update)
    nudged.index_add_(0, pairs[1], -update)
    state.pos = positions + nudged.clamp(min=-0.5, max=0.5)


def _run_multilevel_connected(
    problem: LayoutProblem,
    public_config: Optional[Union[NativeStressMLConfig, LayoutConfig]],
    ml_config: NativeStressMLConfig,
    target_aspect: Optional[float],
) -> torch.Tensor:
    """Run the connected-graph multilevel native-stress V-cycle.

    Parameters
    ----------
    problem : LayoutProblem
        Finest layout problem.
    public_config : NativeStressMLConfig or LayoutConfig, optional
        User-facing configuration.
    ml_config : NativeStressMLConfig
        Resolved multilevel configuration.
    target_aspect : float, optional
        Optional final aspect target.

    Returns
    -------
    torch.Tensor
        Final positions with shape ``[N, 2]``.
    """
    _assert_memory_budget(
        "aggressive hybrid coarsen",
        problem.num_nodes,
        problem.edge_index.shape[1],
        96,
        64,
    )
    _log_stage("aggressive hybrid coarsen", problem.num_nodes, problem.edge_index.shape[1])
    ctx = RuntimeContext(plan=ExecutionPlan(device=str(problem.edge_index.device)))
    hierarchy_state = AggressiveHybridCoarsen(
        AggressiveHybridCoarsenConfig(
            target=ml_config.coarsest_nodes,
            max_levels=ml_config.max_levels,
            min_shrink_ratio=ml_config.coarsen_min_shrink_ratio,
        )
    ).apply(problem, SolveState(), ctx)
    hierarchy = hierarchy_state.hierarchy or []
    selected_levels = _select_levels(hierarchy, ml_config)
    profile_nodes = [problem.num_nodes, *(level.num_nodes for level in selected_levels)]
    print(
        "native_stress_ml: contraction profile "
        + " -> ".join(f"{node_count:,}" for node_count in profile_nodes),
        file=sys.stderr,
        flush=True,
    )
    if not selected_levels:
        native_config = _native_config_for_stage(
            num_nodes=problem.num_nodes,
            public_config=public_config,
            seed=ml_config.seed,
            target_aspect=target_aspect,
            steps=0,
            overlap_iterations=10,
        )
        return _layout_connected_native_stress(problem, SolveState(), ctx, native_config)

    coarse_problem = _coarsest_problem(problem, selected_levels)
    _assert_memory_budget(
        "coarsest native-stress solve",
        coarse_problem.num_nodes,
        coarse_problem.edge_index.shape[1],
        256,
        96,
    )
    _log_stage(
        "coarsest native-stress solve",
        coarse_problem.num_nodes,
        coarse_problem.edge_index.shape[1],
    )
    coarse_config = _native_config_for_stage(
        num_nodes=coarse_problem.num_nodes,
        public_config=public_config,
        seed=ml_config.seed,
        target_aspect=None,
        steps=ml_config.coarse_steps,
        overlap_iterations=0,
    )
    state = SolveState()
    if coarse_problem.num_nodes > ml_config.coarsest_nodes:
        _log_stage(
            "sampled coarsest fallback",
            coarse_problem.num_nodes,
            coarse_problem.edge_index.shape[1],
        )
        state = SampledCoarsestSolve(
            SampledCoarsestSolveConfig(
                target_nodes=ml_config.coarsest_nodes,
                native_config=coarse_config,
                ml_config=ml_config,
            )
        ).apply(coarse_problem, state, ctx)
    else:
        state = _build_coarsest_pipeline(coarse_config, ml_config).apply(coarse_problem, state, ctx)
    if state.pos is None:
        raise RuntimeError("native_stress_ml coarsest solve did not produce positions.")

    for level_index in range(len(selected_levels) - 1, -1, -1):
        level = selected_levels[level_index]
        fine_problem = _level_problem(problem, selected_levels, level_index)
        _assert_memory_budget(
            "prolong and refine",
            fine_problem.num_nodes,
            fine_problem.edge_index.shape[1],
            192,
            80,
        )
        _log_stage("prolong and refine", fine_problem.num_nodes, fine_problem.edge_index.shape[1])
        state.hierarchy = [level]
        state = DirectMapping(DirectMappingConfig(jitter_scale=0.25)).apply(
            fine_problem,
            state,
            ctx,
        )
        state = BuildAdjacency(
            BuildAdjacencyConfig(weighted=True, dedup="min", format="list", directed=False)
        ).apply(fine_problem, state, ctx)
        state = NeighborSmoothing(NeighborSmoothingConfig(blend_factor=0.85)).apply(
            fine_problem,
            state,
            ctx,
        )
        mode = _selected_repulsion_mode(fine_problem.num_nodes, ml_config)
        _apply_local_repulsion(state, fine_problem, mode=mode, seed=ml_config.seed + level_index)
        refine_config = _native_config_for_stage(
            num_nodes=fine_problem.num_nodes,
            public_config=public_config,
            seed=ml_config.seed + level_index,
            target_aspect=None,
            steps=ml_config.refine_steps,
            overlap_iterations=0,
        )
        state.converged = False
        state = _build_refine_pipeline(refine_config, ml_config).apply(fine_problem, state, ctx)

    if state.pos is None:
        raise RuntimeError("native_stress_ml did not produce positions.")
    if problem.num_nodes <= ml_config.overlap_max_nodes:
        _log_stage("final overlap projection", problem.num_nodes, problem.edge_index.shape[1])
        state = OverlapProjection(OverlapProjectionConfig(padding=2.0, iterations=5)).apply(
            problem,
            state,
            ctx,
        )
    else:
        _log_stage(
            "skip final O(N^2) overlap projection",
            problem.num_nodes,
            problem.edge_index.shape[1],
        )
    state = AspectRatioFit(AspectRatioFitConfig(target_aspect=target_aspect)).apply(
        problem,
        state,
        ctx,
    )
    if state.pos is None:
        raise RuntimeError("native_stress_ml aspect fit did not produce positions.")
    return state.pos.detach()


def layout_native_stress_ml_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    config: Optional[Union[NativeStressMLConfig, LayoutConfig]] = None,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    target_aspect: Optional[float] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Run the multilevel native-stress scale path.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``.
    config : NativeStressMLConfig or LayoutConfig, optional
        Multilevel or public layout configuration.
    seed : int, default=42
        Deterministic seed.
    edge_weights : torch.Tensor, optional
        Optional per-edge weights with shape ``[E]``.
    target_aspect : float, optional
        Optional output aspect ratio.
    **kwargs : Any
        Compatibility keywords accepted by generic dispatchers.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.
    """
    del kwargs
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if edge_weights is not None and edge_weights.shape[0] != edge_index.shape[1]:
        raise ValueError("edge_weights length must match edge count.")

    ml_config = _resolve_ml_config(config=config, seed=seed)
    if not should_use_native_stress_ml(
        num_nodes=num_nodes,
        num_edges=int(edge_index.shape[1]),
        config=ml_config,
    ):
        return layout_native_stress_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            node_sizes=node_sizes,
            config=config if isinstance(config, LayoutConfig) else None,
            seed=seed,
            edge_weights=edge_weights,
            target_aspect=target_aspect,
        )

    output_device = edge_index.device if edge_index.numel() > 0 else torch.device("cpu")
    if node_sizes is not None:
        output_device = node_sizes.device
    if node_sizes is None:
        resolved_node_sizes = torch.ones((num_nodes, 2), dtype=torch.float32, device=output_device)
    else:
        resolved_node_sizes = normalize_node_sizes(node_sizes=node_sizes, device=output_device)

    problem = LayoutProblem(
        edge_index=edge_index.to(device=output_device, dtype=torch.long),
        num_nodes=num_nodes,
        node_sizes=resolved_node_sizes,
        edge_weights=None if edge_weights is None else edge_weights.to(device=output_device),
        seed=ml_config.seed,
    )
    copied_config = copy.copy(config) if isinstance(config, LayoutConfig) else config
    return _run_multilevel_connected(
        problem=problem,
        public_config=copied_config,
        ml_config=ml_config,
        target_aspect=target_aspect,
    )


__all__ = [
    "NativeStressMLConfig",
    "SampledCoarsestSolve",
    "SampledCoarsestSolveConfig",
    "layout_native_stress_ml_pipeline",
    "should_use_native_stress_ml",
]
