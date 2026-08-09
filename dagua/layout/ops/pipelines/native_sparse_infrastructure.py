"""Sparse-infrastructure detector, t-FDP challenger, and band mini-contest (W1-A).

Power grids and similar sparse meshed-tree infrastructure graphs (bridge-rich,
long-diameter, hub-free) were structurally uncontested in the native route:

- Rows above ``MAX_CONTEST_NODES`` (1500) skipped the undirected contest
  entirely and returned the bare incumbent unrefereed.
- Rows on the large sfdp+PRISM fast path had no long-range-repulsion family
  in their router-v2 mini-contest shortlist.

This module closes both holes with three pieces, all admitted by STRUCTURAL
features only (no graph names, corpus ids, or score-derived inputs):

1. A sparse-infrastructure detector over ``problem.structure`` router
   features (:func:`sparse_infrastructure_gate`).
2. A t-FDP challenger builder (:func:`tfdp_sparse_positions`) that CALLS the
   in-house ``layout_tfdp_pipeline`` unchanged -- the t-kernel's long-range
   repulsion is exactly what unfolds meshed trees that plain stress/sfdp
   collapse. The reimpl pipeline itself is never modified (fidelity
   invariant); only parameters are supplied here.
3. A bounded band mini-contest for ``1500 < n <= band_max_nodes`` gated rows
   (:func:`sparse_band_mini_contest`), modeled on the router-v2 large
   mini-contest: the incumbent always competes and holds ties, challengers
   are cheap-proxied, and only a small finalist set pays the honest V3
   referee. Admission is cost-modelled through the deterministic DWU ledger
   (``admit_native_work``), never a bare node count and never wall-clock.

Byte-inertness contract: on any row where the gate does not fire, no code in
this module runs and output is byte-identical. When the band contest fires
and the incumbent wins, the exact incumbent tensor is returned unchanged.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, cast

import torch

from dagua.layout.ops.pipelines.native_budget import admit_native_work
from dagua.layout.ops.pipelines.native_cost_model import estimate_native_work_cost
from dagua.layout.ops.state import LayoutProblem

if TYPE_CHECKING:  # pragma: no cover - typing only
    from dagua.config import LayoutConfig
    from dagua.layout.aesthetics import AestheticProfile
    from dagua.layout.graph_classify import GraphStructure

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SparseInfrastructureConfig:
    """Frozen sparse-infrastructure thresholds (RouterV2Config comment style).

    Every threshold is structural with a documented justification; none is a
    graph name, corpus id, or per-graph constant. Initial values follow the
    tri-lab consensus spec; re-fitting happens on training cousins (never on
    the dev set) via the family-stratified fold protocol.
    """

    # Meshed-tree infrastructure (power/road/rail networks) measures
    # E/N ~ 1.1-1.5: mostly tree with sparse cross-bracing. ER/SBM/social
    # graphs at benchmark densities sit well above 2.
    ratio_max: float = 1.65
    # Band-admission escape hatch: the >1500 band was previously uncontested,
    # so a looser density bound is safe there -- the honest referee, not the
    # gate, does the selecting; the gate only bounds runtime.
    band_ratio_max: float = 3.0
    # Infrastructure networks are physically degree-bounded (a bus connects
    # to a handful of lines). Scale-free hubs blow far past this.
    max_degree_max: int = 20
    # Fraction of edges incident to the top-5%-degree nodes. Hub-free
    # infrastructure sits near ~0.1-0.3; scale-free tails concentrate 0.5+.
    hub_edge_fraction_max: float = 0.4
    # Long-range structure requirement: 2D-embedded infrastructure has
    # diameter ~ sqrt(N) (0.65 relaxes RouterV2's lattice-interior 1.2 factor
    # because meshed trees are less regular than grid patches); small-world /
    # community graphs sit at ~ log N and stay closed.
    diameter_sqrt_factor: float = 0.65
    # Below ~200 nodes the normal contest already carries stress-family
    # candidates that cover this class; the t-FDP arm adds nothing but cost.
    min_nodes: int = 200
    # Band lower edge == MAX_CONTEST_NODES: the band is a NEW bounded code
    # path exactly where the full contest has never run.
    band_min_nodes_exclusive: int = 1500
    # Initial conservative band cap: all dev-set target rows are <= 1723;
    # widening past 3000 requires cousin-fitted cost evidence first.
    band_max_nodes: int = 3000
    # Proxy/referee substrate holds a dense [N, N] hop-distance matrix in
    # float32: n^2 * 4B. 40 MB bounds it at the band cap (3000^2*4 = 36 MB).
    apsp_bytes_max: int = 40_000_000
    # t-FDP gamma sweep: gamma shapes the t-kernel tail (1.0 = heavy
    # long-range repulsion, 3.0 = tighter). The reference default is 2.0;
    # one variant either side lets the referee pick per graph.
    tfdp_gammas: tuple[float, ...] = (1.0, 2.0, 3.0)
    # Honest-referee finalists beyond the incumbent (cheap-proxy ranks the
    # rest). 2 matches the spec's "honest-referee top 2-3" economics.
    referee_finalists: int = 2


SPARSE_INFRA = SparseInfrastructureConfig()


def tfdp_iteration_schedule(num_nodes: int) -> int:
    """Return the deterministic t-FDP iteration count for one graph size.

    Keeps modeled per-arm work roughly flat across the band: the reference
    300 iterations up to ~2000 nodes, then a 1/n taper floored at 150 so the
    largest band rows stay bounded. Pure function of node count -- never
    wall-clock.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    int
        Iteration count in ``[150, 300]``.
    """
    if num_nodes <= 0:
        return 300
    return int(min(300, max(150, 600_000 // num_nodes)))


def tfdp_pivot_schedule(num_nodes: int) -> int:
    """Return the deterministic PMDS pivot count for one graph size.

    The reference default (100) already covers the sub-band sizes; above
    ~1600 nodes pivots grow with sqrt(N) so the PMDS skeleton keeps pace
    with the longer graph diameter while bounding the pivots*E BFS cost.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    int
        Pivot count, at least the reference 100 and at most ``num_nodes``.
    """
    if num_nodes <= 0:
        return 100
    return min(num_nodes, max(100, int(round(2.5 * math.sqrt(float(num_nodes))))))


def sparse_infrastructure_gate(
    structure: Optional["GraphStructure"],
    num_nodes: int,
    *,
    has_clusters: bool = False,
    ratio_max: Optional[float] = None,
) -> bool:
    """Return whether the sparse-infrastructure detector fires.

    Conservative by construction: unmeasured features (zero diameter,
    ``None`` structure) keep the gate closed, preserving pre-W1-A behavior.
    All inputs are structural (``problem.structure`` router features,
    measured up to ``ROUTER_FEATURE_MAX_NODES``); no graph names, corpus
    ids, or score-derived values participate.

    Parameters
    ----------
    structure : GraphStructure, optional
        Classified graph topology.
    num_nodes : int
        Number of nodes (``<= 0`` means unknown).
    has_clusters : bool, default=False
        Whether the problem carries cluster metadata (clustered rows are
        owned by the cluster-aware families, never by this arm).
    ratio_max : float, optional
        Density bound override; ``None`` uses the strict shortlist bound,
        the band contest passes ``SPARSE_INFRA.band_ratio_max``.

    Returns
    -------
    bool
        ``True`` when the graph presents as sparse hub-free long-diameter
        infrastructure.
    """
    cfg = SPARSE_INFRA
    limit = cfg.ratio_max if ratio_max is None else float(ratio_max)
    if structure is None or has_clusters or num_nodes < cfg.min_nodes:
        return False
    # The undirected contest route already implies undirected semantics;
    # this defends the gate if it is ever consulted from a broader seam.
    if getattr(structure, "is_semantically_directed", None) is True:
        return False
    if not bool(getattr(structure, "has_dominant_component", False)):
        return False
    diameter = int(getattr(structure, "diameter_estimate", 0))
    if diameter <= 0:
        return False
    return (
        float(getattr(structure, "edge_to_node_ratio", float("inf"))) <= limit
        and int(getattr(structure, "max_degree", num_nodes)) <= cfg.max_degree_max
        and float(getattr(structure, "hub_edge_fraction", 1.0)) <= cfg.hub_edge_fraction_max
        and float(diameter) >= cfg.diameter_sqrt_factor * math.sqrt(float(num_nodes))
    )


def sparse_band_contest_eligible(problem: LayoutProblem) -> bool:
    """Return whether one problem enters the large-sparse band mini-contest.

    Deterministic and input-only: band size bounds, the sparse detector at
    the relaxed band density, and the APSP memory bound. The DWU ledger
    admission (modeled work) happens inside the contest itself.

    Parameters
    ----------
    problem : LayoutProblem
        Prepared undirected layout problem.

    Returns
    -------
    bool
        ``True`` when the band mini-contest may run.
    """
    cfg = SPARSE_INFRA
    n = int(problem.num_nodes)
    if not cfg.band_min_nodes_exclusive < n <= cfg.band_max_nodes:
        return False
    if 4 * n * n > cfg.apsp_bytes_max:
        return False
    return sparse_infrastructure_gate(
        cast("Optional[GraphStructure]", problem.structure),
        n,
        has_clusters=bool(problem.clusters),
        ratio_max=cfg.band_ratio_max,
    )


def tfdp_sparse_positions(
    problem: LayoutProblem,
    *,
    gamma: float,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Run one t-FDP challenger with the sparse-infrastructure schedules.

    CALLS the in-house reimpl pipeline unchanged (fidelity invariant --
    ``pipelines/tfdp.py`` is executed by the tfdp_reimpl field engine and is
    never modified here); only deterministic parameters are supplied.

    Parameters
    ----------
    problem : LayoutProblem
        Prepared undirected layout problem.
    gamma : float
        t-force exponent variant.
    seed : int, optional
        Deterministic seed override; ``None`` uses the problem seed.

    Returns
    -------
    torch.Tensor
        Raw t-FDP positions with shape ``[N, 2]``.
    """
    from dagua.layout.ops.pipelines.tfdp import layout_tfdp_pipeline

    n = int(problem.num_nodes)
    resolved_seed = (
        int(seed) if seed is not None else (int(problem.seed) if problem.seed is not None else 42)
    )
    return layout_tfdp_pipeline(
        edge_index=problem.edge_index,
        num_nodes=n,
        node_sizes=problem.node_sizes,
        seed=resolved_seed,
        init="pmds",
        force_mode="exact",
        gamma=float(gamma),
        max_iter=tfdp_iteration_schedule(n),
        pmds_pivots=tfdp_pivot_schedule(n),
    )


def _band_arm_cost_admitted(
    problem: LayoutProblem,
    config: Optional["LayoutConfig"],
    device_class: str,
    reason: str,
) -> bool:
    """Admit one band challenger arm through the deterministic DWU ledger.

    The arm is priced as spec'd n*e modeled work (stress-family pair term
    with ``sample_pairs = n * e``, one step) plus the standard reserved
    scoring sample volume. This deliberately prices admission on the graph's
    structural size -- never a bare node-count check and never wall-clock.

    Parameters
    ----------
    problem : LayoutProblem
        Prepared undirected layout problem.
    config : LayoutConfig, optional
        Prepared native configuration carrying the optional budget ledger.
    device_class : str
        Frozen cost-table device axis (``"cpu"`` or ``"cuda"``).
    reason : str
        Stable ledger decision reason.

    Returns
    -------
    bool
        ``True`` when the arm was admitted (or no ledger is active).
    """
    n = int(problem.num_nodes)
    num_edges = int(problem.edge_index.shape[1]) if problem.edge_index.numel() else 0
    cost = estimate_native_work_cost(
        problem,
        "stress",
        {"steps": 1, "sample_pairs": n * max(num_edges, 1)},
        device_class,
    )
    return admit_native_work(config, cost, reason)


def sparse_band_mini_contest(
    incumbent_pos: torch.Tensor,
    problem: LayoutProblem,
    config: "LayoutConfig",
) -> torch.Tensor:
    """Referee t-FDP and sfdp+PRISM challengers against the band incumbent.

    Modeled on ``_router_v2_large_mini_contest``: the incumbent (today's
    unrefereed band output) always competes and holds ties; challengers pass
    the shared repair/degeneracy/PRISM plumbing; every candidate is
    cheap-proxied and only the incumbent plus the top proxy finalists pay
    the honest V3 referee. Anything failing leaves the incumbent in place.

    Parameters
    ----------
    incumbent_pos : torch.Tensor
        Exact incumbent positions with shape ``[N, 2]`` (returned unchanged
        when no challenger strictly beats it).
    problem : LayoutProblem
        Prepared undirected layout problem.
    config : LayoutConfig
        Prepared native configuration with ``_dagua_native_*`` metadata.

    Returns
    -------
    torch.Tensor
        Winning positions with shape ``[N, 2]``.
    """
    # Late imports: native_undirected lazily imports this module at its band
    # seam, so the helper imports must stay function-local (no cycle).
    from dagua.layout.ops.pipelines.native_undirected import (
        _admit_v3_referee_score,
        _build_cluster_ids,
        _candidate_is_degenerate,
        _large_prism_shortlist_candidate,
        _log_marketplace_telemetry,
        _native_device_class,
        _never_nan_winner,
        _portfolio_has_budget,
        _project_candidate_prism,
        _proxy_undirected_candidate,
        _repair_flung_isolates,
        _reraise_worker_timeout,
        _score_undirected_candidate_payload,
        _select_undirected_winner,
    )

    started_at = time.perf_counter()
    started_process_at = time.process_time()
    n = int(problem.num_nodes)
    seed = int(problem.seed) if problem.seed is not None else 42
    node_sep = float(getattr(config, "_dagua_native_node_sep", config.node_sep))
    aesthetic_profile: Optional["AestheticProfile"] = getattr(
        config, "_dagua_native_aesthetic_profile", None
    )
    device_class = _native_device_class(config)
    positions: Dict[str, torch.Tensor] = {"incumbent": incumbent_pos}

    def _admit(name: str, raw_pos: Optional[torch.Tensor]) -> None:
        if raw_pos is None or not bool(torch.isfinite(raw_pos).all().item()):
            return
        repaired = _repair_flung_isolates(raw_pos, problem, node_sep)
        degenerate, reason = _candidate_is_degenerate(
            repaired, problem.node_sizes, problem.edge_index
        )
        if degenerate:
            _LOGGER.info("Rejected sparse-band candidate %s_raw: %s", name, reason)
        else:
            positions[f"{name}_raw"] = repaired
        try:
            projected = _project_candidate_prism(repaired, problem)
        except Exception as exc:  # noqa: BLE001 -- one cleanup variant fails closed
            _reraise_worker_timeout(exc)
            projected = None
        if projected is not None:
            degenerate, reason = _candidate_is_degenerate(
                projected, problem.node_sizes, problem.edge_index
            )
            if not degenerate:
                positions[f"{name}_prism"] = projected

    # Shared proxy/referee substrate (dense hop-distance matrix) is modeled
    # work too; a ledger veto here returns the incumbent untouched.
    apsp_cost = estimate_native_work_cost(problem, "apsp", {}, device_class)
    if not admit_native_work(config, apsp_cost, "sparse_band_apsp_substrate"):
        return incumbent_pos

    # Challenger 1: the large fast-path holder family. Band rows never ran
    # it as a refereed candidate before (only as an unrefereed early return
    # below the cap), so it earns a contest seat rather than a walkover.
    if _portfolio_has_budget(config) and _band_arm_cost_admitted(
        problem, config, device_class, "sparse_band_sfdp_prism"
    ):
        try:
            sfdp_pos = _large_prism_shortlist_candidate(problem, config)
            if sfdp_pos is not None and bool(torch.isfinite(sfdp_pos).all().item()):
                degenerate, reason = _candidate_is_degenerate(
                    sfdp_pos, problem.node_sizes, problem.edge_index
                )
                if degenerate:
                    _LOGGER.info("Rejected sparse-band candidate sfdp_prism: %s", reason)
                else:
                    positions["sfdp_prism"] = sfdp_pos
        except Exception as exc:  # noqa: BLE001 -- a failed challenger never sinks the solve
            _reraise_worker_timeout(exc)
            _LOGGER.warning("sparse-band sfdp+PRISM challenger failed", exc_info=True)

    # Challengers 2..4: the t-FDP gamma sweep.
    for gamma in SPARSE_INFRA.tfdp_gammas:
        if not _portfolio_has_budget(config):
            break
        if not _band_arm_cost_admitted(
            problem, config, device_class, f"sparse_band_tfdp_g{gamma:g}"
        ):
            continue
        try:
            _admit(f"tfdp_g{gamma:g}", tfdp_sparse_positions(problem, gamma=gamma))
        except Exception as exc:  # noqa: BLE001 -- a failed challenger never sinks the solve
            _reraise_worker_timeout(exc)
            _LOGGER.warning("sparse-band tfdp challenger failed", exc_info=True)

    if len(positions) <= 1:
        _LOGGER.info("Sparse-band contest n=%d: no admissible challenger, incumbent holds", n)
        return incumbent_pos

    cluster_ids = _build_cluster_ids(problem)
    from dagua.metrics import _all_pairs_unweighted, _build_csr

    offsets, targets = _build_csr(problem.edge_index.detach().to(device="cpu"), n)
    all_pairs_dist = _all_pairs_unweighted(offsets, targets, n, max_dist=n)
    proxy_scores = {
        name: _proxy_undirected_candidate(pos, problem, cluster_ids, all_pairs_dist)
        for name, pos in positions.items()
    }
    # Finalists: incumbent (mandatory floor) + top proxy challengers, with a
    # deterministic name tie-break so equal proxies never reorder by dict
    # iteration accident.
    challengers = sorted(
        (name for name in positions if name != "incumbent"),
        key=lambda name: (-proxy_scores[name], name),
    )
    finalists = ["incumbent", *challengers[: SPARSE_INFRA.referee_finalists]]
    proxy_argmax = challengers[0] if challengers else "incumbent"

    scores: Dict[str, float] = {}
    telemetry: Dict[str, Any] = {}
    for name in finalists:
        if not _admit_v3_referee_score(
            problem,
            config,
            mandatory_floor=name in ("incumbent", proxy_argmax),
        ):
            continue
        score, score_telemetry = _score_undirected_candidate_payload(
            positions[name],
            problem,
            cluster_ids,
            aesthetic_profile,
            all_pairs_dist,
        )
        scores[name] = score
        if score_telemetry is not None:
            telemetry[name] = score_telemetry
    best_name = _select_undirected_winner(scores, telemetry, "incumbent")
    _log_marketplace_telemetry(
        route="undirected_sparse_band",
        structural_gate="sparse_infrastructure",
        positions=positions,
        proxy_scores=proxy_scores,
        full_scores=scores,
        finalist_names=list(scores),
        winner_name=best_name,
        started_at=started_at,
        started_process_at=started_process_at,
        config=config,
    )
    _LOGGER.info(
        "Sparse-infrastructure band contest fired n=%d candidates=%s winner=%s",
        n,
        ", ".join(f"{name}:{score:.3f}" for name, score in scores.items()),
        best_name,
    )
    if best_name == "incumbent":
        # Byte-inertness: a no-win contest returns the exact incumbent tensor.
        return incumbent_pos
    return _never_nan_winner(positions[best_name], problem, node_sep, seed)


__all__ = [
    "SPARSE_INFRA",
    "SparseInfrastructureConfig",
    "sparse_band_contest_eligible",
    "sparse_band_mini_contest",
    "sparse_infrastructure_gate",
    "tfdp_iteration_schedule",
    "tfdp_pivot_schedule",
    "tfdp_sparse_positions",
]
