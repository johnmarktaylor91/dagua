"""Undirected-portfolio native route (r80-S4).

Semantically-undirected graphs (social/community/SBM/small-world/mesh/
scale-free families) are where the layered native default loses most of its
benchmark comparisons to external force engines. This route runs a small
candidate CONTEST instead of betting on one pipeline:

- Candidate A (incumbent): whatever ``_choose_native_pipeline_baseline``
  would have selected if this route did not exist, run through the normal
  native path (including its own polish battery). This guarantees the route
  can never do worse than the pre-portfolio router wherever selection is
  honest.
- Candidate B: dagua's own bit-faithful sfdp reimplementation on the same
  problem tensors, finished with size-aware overlap projection.
- Candidate C: dagua's own bit-faithful neato reimplementation + projection,
  gated by the quality knob (see ``_neato_in_contest``).

All candidates are scored with the SAME honest composite the benchmark
harness uses for undirected rows (``metrics.full`` + ``composite_auto``
with ``is_semantically_directed=False``); argmax wins, ties go to the
incumbent. Challenger candidates additionally pass a degeneracy guard
(collapsed layouts with near-zero edge lengths or a bounding box smaller
than the nodes it must contain are rejected before the contest) so a
pathological challenger score can never launder a broken layout past the
incumbent.

No external layout binaries are invoked anywhere in this module -- the
sfdp/neato pipelines are the fidelity-campaign reimplementations
(``dagua/layout/ops/pipelines/sfdp.py`` / ``neato.py``), pure PyTorch.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import ClassVar, Dict, Optional, Tuple

import torch

from dagua.config import LayoutConfig
from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op
from dagua.layout.projection import project_overlaps

# Above this size the contest is skipped and the incumbent runs alone.
# Documented cap: the Stage-1 probe only produced candidate data up to 500
# nodes (see .project-context/research/r79_native/P8_PORTFOLIO_PROBE.md);
# probe data for larger graphs would be needed before raising this.
MAX_CONTEST_NODES = 1500

# Candidate C (neato) participates when the public quality knob resolves to
# at least this value ("high" alias = 0.75)...
NEATO_QUALITY_THRESHOLD = 0.75
# ...OR, at balanced quality, when the problem is small enough that neato's
# SMACOF converges within seconds. Probe-derived (see _neato_in_contest and
# P8_PORTFOLIO_PROBE.md): all balanced-quality neato contest wins are at
# n <= 80 (max 8s); at n > 80 it costs 40-150s and never won a probe row.
NEATO_BALANCED_NODE_CAP = 80

# Degeneracy guard thresholds (see _candidate_is_degenerate).
DEGENERACY_MIN_EDGE_TO_DIAGONAL_RATIO = 0.5
DEGENERACY_MIN_BBOX_TO_NODE_AREA_RATIO = 0.5


@dataclass(frozen=True)
class UndirectedPortfolioRouteConfig:
    """Frozen op-config for the undirected portfolio contest.

    Parameters
    ----------
    layout_config : LayoutConfig, optional
        Prepared native layout configuration used to run the incumbent and
        resolve quality/time budgets. ``None`` falls back to defaults.
    """

    layout_config: Optional[LayoutConfig] = field(default=None)


def _candidate_is_degenerate(
    pos: torch.Tensor,
    node_sizes: Optional[torch.Tensor],
    edge_index: torch.Tensor,
) -> Tuple[bool, str]:
    """Return whether a challenger layout is geometrically collapsed.

    Two symptoms are checked, either one rejects the candidate BEFORE the
    composite contest (composite terms like edge-length uniformity can score
    a fully-collapsed layout deceptively well):

    1. Mean edge length below ``DEGENERACY_MIN_EDGE_TO_DIAGONAL_RATIO`` times
       the mean node bounding-box diagonal -- edges shorter than half a node
       mean the drawing cannot visually separate its endpoints.
    2. Layout bounding-box area below
       ``DEGENERACY_MIN_BBOX_TO_NODE_AREA_RATIO`` times the summed node-box
       area -- the canvas is smaller than the nodes it must contain, so
       overlap is unavoidable.

    Parameters
    ----------
    pos : torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    node_sizes : torch.Tensor, optional
        Node bounding boxes with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.

    Returns
    -------
    tuple[bool, str]
        ``(is_degenerate, reason)``. ``reason`` is empty when healthy.
    """
    n = int(pos.shape[0])
    if n <= 1 or node_sizes is None or node_sizes.numel() == 0:
        return False, ""
    sizes = node_sizes.to(dtype=pos.dtype)
    if sizes.ndim == 1:
        sizes = sizes.unsqueeze(1).expand(-1, 2)

    mean_diagonal = float(torch.linalg.vector_norm(sizes, dim=1).mean().item())
    if edge_index.numel() > 0 and mean_diagonal > 0.0:
        deltas = pos[edge_index[1]] - pos[edge_index[0]]
        mean_edge_length = float(torch.linalg.vector_norm(deltas, dim=1).mean().item())
        if mean_edge_length < DEGENERACY_MIN_EDGE_TO_DIAGONAL_RATIO * mean_diagonal:
            return True, (
                f"mean edge length {mean_edge_length:.2f} < "
                f"{DEGENERACY_MIN_EDGE_TO_DIAGONAL_RATIO} x mean node diagonal "
                f"{mean_diagonal:.2f}"
            )

    bbox_extent = pos.max(dim=0).values - pos.min(dim=0).values
    # Include node extents so a single-row layout is not falsely zero-area.
    bbox_area = float(
        ((bbox_extent[0] + sizes[:, 0].mean()) * (bbox_extent[1] + sizes[:, 1].mean())).item()
    )
    total_node_area = float((sizes[:, 0] * sizes[:, 1]).sum().item())
    if total_node_area > 0.0 and bbox_area < (
        DEGENERACY_MIN_BBOX_TO_NODE_AREA_RATIO * total_node_area
    ):
        return True, (
            f"bbox area {bbox_area:.1f} < {DEGENERACY_MIN_BBOX_TO_NODE_AREA_RATIO} x "
            f"total node-box area {total_node_area:.1f}"
        )
    return False, ""


def _build_cluster_ids(problem: LayoutProblem) -> Optional[torch.Tensor]:
    """Reconstruct per-node cluster ids from problem cluster metadata.

    Mirrors ``DaguaGraph.cluster_ids`` (deepest assignment wins, indices
    follow sorted cluster-name order) so the layout-time composite sees the
    same cluster-separation term the benchmark scorer sees. Nested-dict
    cluster values fall back to leaf collection.

    Parameters
    ----------
    problem : LayoutProblem
        Problem carrying optional ``clusters`` metadata.

    Returns
    -------
    torch.Tensor | None
        Cluster ids with shape ``[N]`` or ``None`` when no clusters exist.
    """
    clusters = problem.clusters
    if not clusters or problem.num_nodes == 0:
        return None
    try:
        from dagua.utils import collect_cluster_leaves

        parents = problem.cluster_parents or {}

        def _depth(name: str) -> int:
            depth = 0
            current: Optional[str] = name
            seen = set()
            while current is not None and current not in seen:
                seen.add(current)
                current = parents.get(current)
                depth += 1
            return depth

        ids = torch.full((problem.num_nodes,), -1, dtype=torch.long)
        node_depth = [-1] * problem.num_nodes
        cluster_name_list = sorted(clusters.keys())
        name_to_idx = {name: index for index, name in enumerate(cluster_name_list)}
        for name in cluster_name_list:
            members = clusters[name]
            if isinstance(members, dict):
                members = collect_cluster_leaves(members)
            depth = _depth(name)
            for node_idx in members:
                node_int = int(node_idx)
                if 0 <= node_int < problem.num_nodes and depth > node_depth[node_int]:
                    ids[node_int] = name_to_idx[name]
                    node_depth[node_int] = depth
        return ids
    except Exception:  # noqa: BLE001 -- scoring must not crash the solve
        return None


def _score_undirected_candidate(
    pos: torch.Tensor,
    problem: LayoutProblem,
    cluster_ids: Optional[torch.Tensor],
) -> float:
    """Score one candidate with the benchmark's honest undirected composite.

    Uses ``metrics.full`` (tier the benchmark uses for graphs under its full
    cutoff -- the contest node cap keeps us in that regime) and
    ``composite_auto(..., is_semantically_directed=False)``. ``full`` is
    self-deterministic for fixed positions (sampled crossing rate seeds its
    own generator), so selection is reproducible.

    Parameters
    ----------
    pos : torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    problem : LayoutProblem
        Problem carrying topology and node sizes.
    cluster_ids : torch.Tensor, optional
        Optional per-node cluster ids for the cluster-separation term.

    Returns
    -------
    float
        Higher-is-better undirected composite score.
    """
    from dagua.metrics import composite_auto, full

    metrics = full(
        pos.detach().to(device="cpu", dtype=torch.float32),
        problem.edge_index.detach().to(device="cpu"),
        node_sizes=(
            None
            if problem.node_sizes is None
            else problem.node_sizes.detach().to(device="cpu", dtype=torch.float32)
        ),
        cluster_ids=cluster_ids,
        direction=problem.direction,
    )
    numeric = {
        key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))
    }
    return float(composite_auto(numeric, is_semantically_directed=False))


# Convergent-cleanup pass budget for challenger candidates. The convergent
# exact projector early-exits at zero overlaps or on measured stagnation,
# so this ceiling is only consumed on hard overlap fields; the contest cap
# (MAX_CONTEST_NODES) bounds the per-pass O(N^2) cost.
CHALLENGER_PROJECTION_ITERATIONS = 200


def _project_candidate(
    pos: torch.Tensor,
    problem: LayoutProblem,
) -> torch.Tensor:
    """Apply size-aware CONVERGENT overlap projection to one challenger.

    Uses the projector's existing public entry point with real node boxes,
    with ``convergent=True`` (r80-S2b): the legacy default projector's
    last-write-wins pushes stall on the dense overlap fields that
    sfdp/neato candidates arrive with (P3B2 forensics: 37+ residual
    overlaps on sbm_4x30 after 50 passes), leaving the 20-point no-overlap
    composite term on the table. The convergent projector provably reaches
    zero overlaps there. Trajectory risk is fully referee-protected on
    this path: a candidate whose cleanup damages CV/crossings simply loses
    the honest-composite contest to the incumbent (plus the degeneracy
    guard rejects collapsed layouts outright), so unlike the default path
    -- where r80-S2 showed unguarded convergent projection regressing
    graphs -- a bad outcome here cannot ship.

    Parameters
    ----------
    pos : torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    problem : LayoutProblem
        Problem carrying node sizes.

    Returns
    -------
    torch.Tensor
        Overlap-projected positions (new tensor).
    """
    if problem.node_sizes is None or problem.node_sizes.numel() == 0:
        return pos
    projected = pos.detach().clone().to(dtype=torch.float32)
    node_sizes = problem.node_sizes.to(device=projected.device, dtype=projected.dtype)
    project_overlaps(
        projected,
        node_sizes,
        iterations=CHALLENGER_PROJECTION_ITERATIONS,
        convergent=True,
    )
    return projected


def _resolved_quality(config: Optional[LayoutConfig]) -> float:
    """Return the normalized [0, 1] quality value from a prepared config.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration.

    Returns
    -------
    float
        Normalized quality; 0.5 (balanced) when unavailable.
    """
    if config is None:
        return 0.5
    try:
        return float(getattr(config, "quality", 0.5))
    except (TypeError, ValueError):
        return 0.5


def _neato_in_contest(config: Optional[LayoutConfig], num_nodes: int) -> bool:
    """Return whether candidate C (neato + projection) joins the contest.

    Two admission paths:

    1. Quality >= high (0.75): neato always joins (up to the contest cap).
    2. Balanced/lower quality with ``num_nodes <= NEATO_BALANCED_NODE_CAP``:
       the Stage-1 probe (P8_PORTFOLIO_PROBE.md) shows every balanced-quality
       contest neato ever wins sits at n <= 80, where its SMACOF loop
       epsilon-exits in <= ~8s; on larger graphs it costs 40-150s and never
       beat the sfdp/incumbent winner in any probe row. The cap keeps the
       neato wins (karate, lattices, grids, petersen, multi-component)
       inside the default wall-time envelope and leaves the slow never-wins
       region to the explicit quality knob.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration.
    num_nodes : int
        Number of nodes in the current problem.

    Returns
    -------
    bool
        ``True`` when candidate C should run.
    """
    if _resolved_quality(config) >= NEATO_QUALITY_THRESHOLD:
        return True
    return num_nodes <= NEATO_BALANCED_NODE_CAP


def layout_native_undirected_portfolio(
    problem: LayoutProblem,
    state: SolveState,
    ctx: RuntimeContext,
    config: LayoutConfig,
) -> torch.Tensor:
    """Run the undirected portfolio contest for one prepared problem.

    Parameters
    ----------
    problem : LayoutProblem
        Prepared layout problem (whole graph or one weak component).
    state : SolveState
        Incoming solve state; candidate runs receive cloned copies.
    ctx : RuntimeContext
        Shared execution context.
    config : LayoutConfig
        Prepared native configuration with ``_dagua_native_*`` metadata.

    Returns
    -------
    torch.Tensor
        Selected positions with shape ``[N, 2]``.
    """
    # Late import avoids a circular import with dagua_native (which imports
    # this module lazily at its two dispatch points).
    from dagua.layout.ops.pipelines.dagua_native import _run_native_problem

    def _run_incumbent() -> torch.Tensor:
        # Candidate A must be EXACTLY today's default output. Re-enter the
        # router with the portfolio branch suppressed via a private attr --
        # NOT via force_pipeline, because several polish stages (edge
        # equalize best-of-polish, component-tiling polish) are gated on
        # force_pipeline being None and would silently weaken the incumbent.
        incumbent_config = copy.copy(config)
        setattr(incumbent_config, "_dagua_native_suppress_portfolio", True)
        incumbent_state = SolveState(pos=None if state.pos is None else state.pos.detach().clone())
        return _run_native_problem(problem, incumbent_state, ctx, incumbent_config)

    incumbent_pos = _run_incumbent()

    # Contest predicate: documented caps. Above MAX_CONTEST_NODES the probe
    # has no candidate data; with an explicit wall-clock budget the extra
    # candidate solves would silently blow it.
    n = int(problem.num_nodes)
    if n > MAX_CONTEST_NODES or getattr(config, "time_budget_s", None) is not None:
        return incumbent_pos

    cluster_ids = _build_cluster_ids(problem)
    scores: Dict[str, float] = {}
    positions: Dict[str, torch.Tensor] = {}

    # Candidate A: the incumbent is ALWAYS eligible (degeneracy guard applies
    # to challengers only).
    positions["incumbent"] = incumbent_pos
    scores["incumbent"] = _score_undirected_candidate(incumbent_pos, problem, cluster_ids)

    seed = int(problem.seed) if problem.seed is not None else 42

    def _add_challenger(name: str, raw_pos: torch.Tensor) -> None:
        projected = _project_candidate(raw_pos, problem)
        degenerate, _reason = _candidate_is_degenerate(
            projected,
            problem.node_sizes,
            problem.edge_index,
        )
        if degenerate:
            return
        positions[name] = projected
        scores[name] = _score_undirected_candidate(projected, problem, cluster_ids)

    # Candidate B: our sfdp reimplementation + projection. Weighted graphs
    # pass edge weights through unchanged (the pipeline handles them).
    # steps mirrors the engine's pipeline dispatch (config.steps, default 0):
    # the Stage-1 probe ran sfdp through the public engine path, which
    # forwards config.steps -- 0 skips the per-level sequential refinement
    # and keeps only the multilevel spring-electrical solve. The probe's
    # headroom numbers correspond to THAT candidate (and it is ~100x
    # cheaper than the standalone default of 500 refinement steps).
    try:
        from dagua.layout.ops.pipelines.sfdp import layout_sfdp_pipeline

        sfdp_pos = layout_sfdp_pipeline(
            edge_index=problem.edge_index,
            num_nodes=n,
            node_sizes=problem.node_sizes,
            steps=max(int(getattr(config, "steps", 0) or 0), 0),
            seed=seed,
            edge_weights=problem.edge_weights,
        )
        _add_challenger("sfdp", sfdp_pos)
    except Exception:  # noqa: BLE001 -- a failed challenger never sinks the solve
        pass

    # Candidate C: our neato reimplementation + projection, quality-gated.
    if _neato_in_contest(config, n):
        try:
            from dagua.layout.ops.pipelines.neato import layout_neato_pipeline

            neato_pos = layout_neato_pipeline(
                edge_index=problem.edge_index,
                num_nodes=n,
                node_sizes=problem.node_sizes,
                seed=seed,
                edge_weights=problem.edge_weights,
            )
            _add_challenger("neato", neato_pos)
        except Exception:  # noqa: BLE001
            pass

    # Argmax selection; strict inequality means ties go to the incumbent.
    best_name = "incumbent"
    for name, score in scores.items():
        if name != "incumbent" and score > scores[best_name]:
            best_name = name
    return positions[best_name]


@register_op
@dataclass(frozen=True)
class UndirectedPortfolioRoute(Op):
    """Run the undirected candidate contest and select the honest winner."""

    config: UndirectedPortfolioRouteConfig = field(default_factory=UndirectedPortfolioRouteConfig)

    name: ClassVar[str] = "undirected_portfolio_route"
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
        """Run the portfolio contest and write the winning positions.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with the contest winner's positions.
        """
        layout_config = self.config.layout_config or LayoutConfig()
        state.pos = layout_native_undirected_portfolio(
            problem=problem,
            state=state,
            ctx=ctx,
            config=layout_config,
        )
        return state


def build_native_undirected_portfolio_pipeline(config: LayoutConfig) -> Pipeline:
    """Build the undirected-portfolio route as a one-op pipeline.

    Follows the existing top-level-route precedent: the route is a
    registered op composed into a named pipeline, so pipeline-level callers
    (``build_dagua_pipeline``) and the direct ``_run_native_problem`` branch
    share one implementation.

    Parameters
    ----------
    config : LayoutConfig
        Prepared native configuration.

    Returns
    -------
    Pipeline
        Single-op pipeline running the candidate contest.
    """
    return Pipeline(
        [UndirectedPortfolioRoute(UndirectedPortfolioRouteConfig(layout_config=config))],
        name="native_undirected_portfolio",
    )


__all__ = [
    "MAX_CONTEST_NODES",
    "NEATO_BALANCED_NODE_CAP",
    "NEATO_QUALITY_THRESHOLD",
    "UndirectedPortfolioRoute",
    "UndirectedPortfolioRouteConfig",
    "build_native_undirected_portfolio_pipeline",
    "layout_native_undirected_portfolio",
]
