"""Honest directed-table portfolio for declared-hierarchical DAGs."""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import ClassVar, Dict, Optional, Tuple

import numpy as np
import torch

from dagua.config import LayoutConfig
from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

MAX_DIRECTED_CONTEST_NODES = 2000
FORCE_SKIP_RATIO_THRESHOLD = 0.3
SCALED_SUGIYAMA_RANK_SEP = 72.0
SCALED_SUGIYAMA_NODE_SEP = 18.0
SUGIYAMA_FIDELITY_MODES = ("graphviz_dot", "graphviz", "igraph")
SUGIYAMA_RANK_SEP_GRID = (36.0, 72.0, 108.0)
SUGIYAMA_NODE_SEP_GRID = (18.0, 36.0, 54.0)
IGRAPH_OUTPUT_SCALE = 50.0
_LOGGER = logging.getLogger(__name__)


def _score_directed_candidate(
    pos: torch.Tensor,
    problem: LayoutProblem,
    cluster_ids: Optional[torch.Tensor],
    all_pairs_dist: Optional[np.ndarray] = None,
) -> float:
    """Score a finalist with the frozen honest directed composite.

    Parameters
    ----------
    pos : torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    problem : LayoutProblem
        Directed acyclic layout problem.
    cluster_ids : torch.Tensor, optional
        Per-node cluster ids with shape ``[N]``.
    all_pairs_dist : Optional[numpy.ndarray], optional
        Cached unweighted shortest paths with shape ``[N, N]``.

    Returns
    -------
    float
        Higher-is-better directed composite score.
    """
    from dagua.metrics import composite_auto, full

    numeric = full(
        pos.detach().to(device="cpu", dtype=torch.float32),
        problem.edge_index.detach().to(device="cpu"),
        node_sizes=(
            None
            if problem.node_sizes is None
            else problem.node_sizes.detach().to(device="cpu", dtype=torch.float32)
        ),
        cluster_ids=cluster_ids,
        direction=problem.direction,
        all_pairs_dist=all_pairs_dist,
    )
    numeric["declared_hierarchical"] = True
    return float(composite_auto(numeric, is_semantically_directed=True))


def _score_directed_candidate_cached(
    pos: torch.Tensor,
    problem: LayoutProblem,
    cluster_ids: Optional[torch.Tensor],
    all_pairs_dist: Optional[np.ndarray],
) -> float:
    """Score a directed candidate while preserving old monkeypatch arity.

    Parameters
    ----------
    pos : torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    problem : LayoutProblem
        Directed acyclic layout problem.
    cluster_ids : torch.Tensor, optional
        Per-node cluster ids with shape ``[N]``.
    all_pairs_dist : Optional[numpy.ndarray]
        Cached unweighted shortest paths with shape ``[N, N]``.

    Returns
    -------
    float
        Higher-is-better directed composite score.
    """
    try:
        return _score_directed_candidate(
            pos,
            problem,
            cluster_ids,
            all_pairs_dist=all_pairs_dist,
        )
    except TypeError as exc:
        if "all_pairs_dist" not in str(exc):
            raise
        return _score_directed_candidate(pos, problem, cluster_ids)


def _force_challengers_enabled(edge_index: torch.Tensor, num_nodes: int) -> bool:
    """Return whether skip edges or multiedges justify force challengers.

    Parameters
    ----------
    edge_index : torch.Tensor
        Directed acyclic edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    bool
        Whether the R7 topology gate is open.
    """
    edge_count = int(edge_index.shape[1]) if edge_index.numel() else 0
    if edge_count == 0:
        return False
    edges = [(int(src), int(dst)) for src, dst in edge_index.t().detach().cpu().tolist()]
    if len(set(edges)) < edge_count:
        return True

    outgoing: list[list[int]] = [[] for _ in range(num_nodes)]
    indegree = [0] * num_nodes
    for src, dst in edges:
        if src == dst:
            continue
        outgoing[src].append(dst)
        indegree[dst] += 1
    queue = [node for node, degree in enumerate(indegree) if degree == 0]
    ranks = [0] * num_nodes
    cursor = 0
    while cursor < len(queue):
        src = queue[cursor]
        cursor += 1
        for dst in outgoing[src]:
            ranks[dst] = max(ranks[dst], ranks[src] + 1)
            indegree[dst] -= 1
            if indegree[dst] == 0:
                queue.append(dst)
    long_edges = sum(abs(ranks[dst] - ranks[src]) > 1 for src, dst in edges)
    return float(long_edges) / float(edge_count) > FORCE_SKIP_RATIO_THRESHOLD


def _register_challenger_variants(
    name: str,
    raw_pos: torch.Tensor,
    problem: LayoutProblem,
    config: LayoutConfig,
    positions: Dict[str, torch.Tensor],
    preserve_rank_order: bool = False,
) -> None:
    """Register guarded raw and projected variants of one challenger.

    Parameters
    ----------
    name : str
        Candidate family name.
    raw_pos : torch.Tensor
        Unprojected positions with shape ``[N, 2]``.
    problem : LayoutProblem
        Directed layout problem.
    config : LayoutConfig
        Prepared native configuration.
    positions : dict[str, torch.Tensor]
        Finalist registry updated in place.
    preserve_rank_order : bool, default=False
        Whether projected variants must retain the raw candidate's within-rank
        ordering.

    Returns
    -------
    None
        Variants are added to ``positions`` when healthy.
    """
    from dagua.layout.ops.pipelines.native_undirected import (
        _candidate_is_degenerate,
        _project_candidate,
        _repair_flung_isolates,
    )

    node_sep = float(getattr(config, "_dagua_native_node_sep", config.node_sep))
    repaired = _repair_flung_isolates(raw_pos, problem, node_sep)
    projected = _project_candidate(repaired, problem, convergent=False)
    convergent = _project_candidate(repaired, problem, convergent=True)
    if preserve_rank_order:
        projected = _restore_projected_rank_order(repaired, projected)
        convergent = _restore_projected_rank_order(repaired, convergent)
    variants = {
        name + "_raw": repaired,
        name: projected,
        name + "_convergent": convergent,
    }
    for variant_name, candidate in variants.items():
        degenerate, reason = _candidate_is_degenerate(
            candidate,
            problem.node_sizes,
            problem.edge_index,
        )
        if degenerate:
            _LOGGER.info("Rejected directed candidate %s: %s", variant_name, reason)
            continue
        positions[variant_name] = candidate


def _restore_projected_rank_order(
    raw_positions: torch.Tensor,
    projected_positions: torch.Tensor,
) -> torch.Tensor:
    """Restore one layered candidate's ordering after overlap projection.

    Parameters
    ----------
    raw_positions : torch.Tensor
        Pre-projection positions with shape ``[N, 2]``.
    projected_positions : torch.Tensor
        Projected positions with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Projected coordinates with each raw rank's x ordering restored while
        retaining the projector's separated x-coordinate multiset.
    """
    out = projected_positions.detach().clone()
    rank_values = sorted({float(value) for value in raw_positions[:, 1].tolist()})
    for rank_y in rank_values:
        nodes = [
            node
            for node in range(int(raw_positions.shape[0]))
            if abs(float(raw_positions[node, 1].item()) - rank_y) <= 1.0e-6
        ]
        if len(nodes) < 2:
            continue
        ordered_nodes = sorted(nodes, key=lambda node: float(raw_positions[node, 0].item()))
        ordered_x = sorted(float(out[node, 0].item()) for node in nodes)
        for node, x_value in zip(ordered_nodes, ordered_x):
            out[node, 0] = x_value
    return out


def layout_native_directed_portfolio(
    problem: LayoutProblem,
    state: SolveState,
    ctx: RuntimeContext,
    config: LayoutConfig,
) -> torch.Tensor:
    """Run the directed candidate contest and return its monotone winner.

    Parameters
    ----------
    problem : LayoutProblem
        Prepared declared-hierarchical problem.
    state : SolveState
        Incoming solve state.
    ctx : RuntimeContext
        Shared runtime context.
    config : LayoutConfig
        Prepared native configuration.

    Returns
    -------
    torch.Tensor
        Winning positions with shape ``[N, 2]``.
    """
    from dagua.layout.ops.pipelines.dagua_native import _run_native_problem
    from dagua.layout.ops.pipelines.native_undirected import (
        _build_cluster_ids,
        _log_marketplace_telemetry,
        _portfolio_has_budget,
        _portfolio_remaining_s,
        _reraise_worker_timeout,
    )

    started = time.perf_counter()
    incumbent_config = copy.copy(config)
    setattr(incumbent_config, "_dagua_native_suppress_portfolio", True)
    incumbent_state = SolveState(pos=None if state.pos is None else state.pos.detach().clone())
    incumbent = _run_native_problem(problem, incumbent_state, ctx, incumbent_config)
    n = int(problem.num_nodes)
    if n > MAX_DIRECTED_CONTEST_NODES:
        _LOGGER.info(
            "Directed contest gate=incumbent_only n=%d wall_time_s=%.3f",
            n,
            time.perf_counter() - started,
        )
        return incumbent
    if not _portfolio_has_budget(config):
        _LOGGER.info(
            "Directed marketplace budget exhausted after incumbent n=%d remaining_s=%s",
            n,
            _portfolio_remaining_s(config),
        )
        return incumbent

    positions: Dict[str, torch.Tensor] = {"incumbent": incumbent}
    seed = int(problem.seed) if problem.seed is not None else 42
    # Fidelity adapters were validated against CPU reference implementations;
    # several intentionally use NumPy internally. Keep challenger inputs on
    # CPU even when the incumbent native solve used CUDA.
    cpu_edges = problem.edge_index.detach().to(device="cpu")
    cpu_sizes = (
        None
        if problem.node_sizes is None
        else problem.node_sizes.detach().to(device="cpu", dtype=torch.float32)
    )
    cpu_weights = (
        None
        if problem.edge_weights is None
        else problem.edge_weights.detach().to(device="cpu", dtype=torch.float32)
    )
    if _portfolio_has_budget(config):
        try:
            from dagua.layout.ops.pipelines.sugiyama import layout_sugiyama_pipeline

            corrected_cluster_dot_x = layout_sugiyama_pipeline(
                edge_index=cpu_edges,
                num_nodes=n,
                node_sizes=cpu_sizes,
                rank_sep=SCALED_SUGIYAMA_RANK_SEP,
                node_sep=SCALED_SUGIYAMA_NODE_SEP,
                seed=seed,
                edge_weights=cpu_weights,
                fidelity_mode="graphviz",
                clusters=problem.clusters,
                cluster_parents=problem.cluster_parents,
                graphviz_apply_cluster_constraints=True,
                graphviz_corrected_dot_x=True,
            )
            if not isinstance(corrected_cluster_dot_x, torch.Tensor):
                raise RuntimeError("corrected cluster dot-x Sugiyama returned non-position output")
            _register_challenger_variants(
                "graphviz_dotx_cluster_corrected",
                corrected_cluster_dot_x,
                problem,
                config,
                positions,
                preserve_rank_order=True,
            )
            point_unit_dot_x = layout_sugiyama_pipeline(
                edge_index=cpu_edges,
                num_nodes=n,
                node_sizes=cpu_sizes,
                rank_sep=SCALED_SUGIYAMA_RANK_SEP,
                node_sep=SCALED_SUGIYAMA_NODE_SEP,
                seed=seed,
                edge_weights=cpu_weights,
                fidelity_mode="graphviz",
                clusters=problem.clusters,
                cluster_parents=problem.cluster_parents,
                graphviz_preserve_point_units=True,
            )
            if not isinstance(point_unit_dot_x, torch.Tensor):
                raise RuntimeError("point-unit dot-x Sugiyama returned non-position output")
            _register_challenger_variants(
                "graphviz_dotx_point_units",
                point_unit_dot_x,
                problem,
                config,
                positions,
            )

            for mode in ("graphviz_dot", "igraph"):
                if not _portfolio_has_budget(config):
                    break
                candidate = layout_sugiyama_pipeline(
                    edge_index=cpu_edges,
                    num_nodes=n,
                    node_sizes=cpu_sizes,
                    seed=seed,
                    edge_weights=cpu_weights,
                    fidelity_mode=mode,
                    clusters=problem.clusters,
                    cluster_parents=problem.cluster_parents,
                )
                if not isinstance(candidate, torch.Tensor):
                    raise RuntimeError(f"{mode} Sugiyama returned non-position output")
                _register_challenger_variants(mode, candidate, problem, config, positions)
            # The fixed Cartesian grid is global: every directed graph sees every
            # fidelity mode and spacing point before the honest referee chooses.
            for mode in SUGIYAMA_FIDELITY_MODES:
                for rank_sep in SUGIYAMA_RANK_SEP_GRID:
                    for node_sep in SUGIYAMA_NODE_SEP_GRID:
                        if not _portfolio_has_budget(config):
                            break
                        candidate = layout_sugiyama_pipeline(
                            edge_index=cpu_edges,
                            num_nodes=n,
                            node_sizes=cpu_sizes,
                            rank_sep=rank_sep,
                            node_sep=node_sep,
                            seed=seed,
                            edge_weights=cpu_weights,
                            fidelity_mode=mode,
                            clusters=problem.clusters,
                            cluster_parents=problem.cluster_parents,
                        )
                        grid_name = f"{mode}_r{rank_sep:g}_n{node_sep:g}"
                        if not isinstance(candidate, torch.Tensor):
                            raise RuntimeError(f"{grid_name} Sugiyama returned non-position output")
                        if mode == "igraph":
                            # Match the reference adapter's fixed conversion from
                            # igraph coordinate units into renderer point units.
                            candidate = candidate * IGRAPH_OUTPUT_SCALE
                        _register_challenger_variants(
                            grid_name,
                            candidate,
                            problem,
                            config,
                            positions,
                        )
        except Exception as exc:  # noqa: BLE001 -- challengers cannot sink the incumbent
            _reraise_worker_timeout(exc)
            _LOGGER.warning("directed Sugiyama challenger failed", exc_info=True)

    force_gate = _force_challengers_enabled(problem.edge_index, n)
    if force_gate:
        try:
            from dagua.layout.ops.pipelines.fcose import layout_fcose_pipeline
            from dagua.layout.ops.pipelines.native_undirected import FCOSE_CONTEST_SEEDS

            for seed_offset in range(FCOSE_CONTEST_SEEDS):
                if not _portfolio_has_budget(config):
                    break
                candidate = layout_fcose_pipeline(
                    edge_index=cpu_edges,
                    num_nodes=n,
                    node_sizes=cpu_sizes,
                    seed=42 + seed_offset,
                    edge_weights=cpu_weights,
                )
                _register_challenger_variants(
                    f"fcose_seed{seed_offset}", candidate, problem, config, positions
                )
        except Exception as exc:  # noqa: BLE001
            _reraise_worker_timeout(exc)
            _LOGGER.warning("directed fCoSE challenger failed", exc_info=True)
        try:
            if not _portfolio_has_budget(config):
                raise RuntimeError("directed YifanHu skipped: insufficient remaining budget")
            from dagua.layout.ops.pipelines.yifanhu import layout_yifanhu_pipeline

            candidate = layout_yifanhu_pipeline(
                edge_index=cpu_edges,
                num_nodes=n,
                node_sizes=cpu_sizes,
                seed=123,
                edge_weights=cpu_weights,
                direction=problem.direction,
            )
            _register_challenger_variants("yifanhu", candidate, problem, config, positions)
        except Exception as exc:  # noqa: BLE001
            _reraise_worker_timeout(exc)
            _LOGGER.warning("directed YifanHu challenger failed", exc_info=True)

    from dagua.metrics import _all_pairs_unweighted, _build_csr

    offsets, targets = _build_csr(cpu_edges, n)
    all_pairs_dist = _all_pairs_unweighted(offsets, targets, n, max_dist=n)
    cluster_ids = _build_cluster_ids(problem)
    scores = {
        name: _score_directed_candidate_cached(candidate, problem, cluster_ids, all_pairs_dist)
        for name, candidate in positions.items()
    }
    best_name = "incumbent"
    for name, score in scores.items():
        if name != "incumbent" and score > scores[best_name]:
            best_name = name
    _log_marketplace_telemetry(
        route="directed",
        structural_gate="force" if force_gate else "layered",
        positions=positions,
        proxy_scores=scores,
        full_scores=scores,
        finalist_names=list(scores),
        winner_name=best_name,
        started_at=started,
    )
    _LOGGER.info(
        "Directed contest gate=%s candidates=%s winner=%s wall_time_s=%.3f",
        "force" if force_gate else "layered",
        ", ".join(f"{name}:{score:.3f}" for name, score in scores.items()),
        best_name,
        time.perf_counter() - started,
    )
    return positions[best_name].to(device=incumbent.device, dtype=incumbent.dtype)


@dataclass(frozen=True)
class DirectedPortfolioRouteConfig:
    """Configuration wrapper for the directed portfolio route."""

    layout_config: Optional[LayoutConfig] = None


@register_op
@dataclass(frozen=True)
class DirectedPortfolioRoute(Op):
    """Run the declared-hierarchical candidate contest."""

    config: DirectedPortfolioRouteConfig = field(default_factory=DirectedPortfolioRouteConfig)
    name: ClassVar[str] = "directed_portfolio_route"
    category: ClassVar[OpCategory] = OpCategory.CONTROL
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run the contest and update the solve state.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime infrastructure.

        Returns
        -------
        SolveState
            State containing the winning positions.
        """
        state.pos = layout_native_directed_portfolio(
            problem,
            state,
            ctx,
            self.config.layout_config or LayoutConfig(),
        )
        return state


def build_native_directed_portfolio_pipeline(config: LayoutConfig) -> Pipeline:
    """Build the directed portfolio as a registered one-op pipeline.

    Parameters
    ----------
    config : LayoutConfig
        Prepared native configuration.

    Returns
    -------
    Pipeline
        Directed portfolio pipeline.
    """
    return Pipeline(
        [DirectedPortfolioRoute(DirectedPortfolioRouteConfig(layout_config=config))],
        name="native_directed_portfolio",
    )


__all__ = [
    "MAX_DIRECTED_CONTEST_NODES",
    "DirectedPortfolioRoute",
    "DirectedPortfolioRouteConfig",
    "build_native_directed_portfolio_pipeline",
    "layout_native_directed_portfolio",
]
