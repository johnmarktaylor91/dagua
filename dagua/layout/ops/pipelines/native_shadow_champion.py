"""Shadow-champion legacy-track contest support (sprint2 W2-4a, cluster C10 core).

The marketplace contests rank candidates by contest-stage referee score, but
the post-contest pipeline (contest-local tail arms, the W5 finisher, global
scale sweep, SMACOF stress polish, continuous facet polish, small-N anneal)
is a basin-local deterministic hill-climb whose gain depends on the starting
geometry. Contest-stage rank is therefore NOT monotone in final rank: a
new-arm candidate can honestly win the contest from a poor terminal basin and
emit a final drawing worse than what the displaced legacy-family winner would
have produced (rome/grafo1000.14 regression, 90.74 -> 84.82,
DIAG_GRAFO1000.md).

Fix (review W2-4a F1-F3 shape): a new-arm family may win a contest argmax
ONLY when the complete legacy-track shadow package -- one full pipeline
re-run with every new-arm family disabled, plus the final two-track referee
contest -- is reserved all-or-nothing on the solve's entry ledger at that
instant (:func:`try_reserve_shadow_package`). When the reservation is vetoed
the contest falls back to the legacy-family champion (fail-closed to the
legacy track; the weaker new-arm drawing is never emitted and no shadow runs).
When it is granted, the displacement is recorded and the outermost native
pipeline invocation re-runs the ENTIRE pipeline once with every new-arm
family disabled -- funded by exactly the reserved deterministic budget, never
a second copy of the entry plan -- scores both FINAL drawings with the
runtime referee, and emits the higher (ties go to the legacy track). The
legacy track is therefore the byte-identical output of the actual pipeline
with all new arms absent AT THE RESERVED BUDGET: finalist selection,
contest-local tails, seed banks, budget consumption, and the complete
terminal chain all replay from legacy state. Both tracks together fit the
one fixed entry budget by construction. An ordinary shadow failure after a
displacement propagates: the primary is never silently emitted once a
new-arm displacement occurred. The gate fires only when a new-arm family
actually won a contest, so every other row is byte-inert.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from dagua.config import LayoutConfig

_LOGGER = logging.getLogger(__name__)

# Candidate-name prefixes of contest arms that postdate the terminal-anneal
# basin assumption ("new arms"). A contest argmax won by one of these
# families records a displacement, arming the legacy-track shadow re-run.
# Structural, not row-keyed: W1-B contributes the planar-certificate family
# ("planar_"); W1-A contributes the t-FDP family ("tfdp": the normal-contest
# "tfdp"/"tfdp_raw" names, the router-v2 large mini-contest "tfdp_g{gamma}"
# sweep, and the sparse-band "tfdp_*" candidates). New contest arms with the
# same stage-locality exposure register their prefix here.
NEW_ARM_FAMILY_PREFIXES: tuple[str, ...] = ("planar_", "tfdp")

# Shared mutable displacement log installed by the outermost pipeline
# invocation; shallow config copies share the list, so contest-level appends
# reach the orchestrator.
NEW_ARM_DISPLACEMENTS_ATTR = "_dagua_native_new_arm_displacements"
# When true, every new-arm generation site is gate-closed and the pipeline
# reproduces the legacy (pre-new-arm) behavior byte-for-byte.
DISABLE_NEW_ARMS_ATTR = "_dagua_native_disable_new_arms"
# Terminal-contest telemetry written to the caller's config for tests and
# benchmark forensics.
SHADOW_CONTEST_TELEMETRY_ATTR = "_dagua_native_shadow_contest_telemetry"
# Shared mutable reservation state installed by the outermost invocation
# (same aliasing contract as the displacement log): the priced shadow
# package, whether it has been reserved on the entry ledger, and any
# reservation vetoes (each veto is a fail-closed-to-legacy contest event).
SHADOW_RESERVATION_STATE_ATTR = "_dagua_native_shadow_reservation_state"

# The legacy-track shadow re-run's share of the entry ledger's free capacity.
# The shadow is the same anytime pipeline as the primary minus the new arms,
# so the equal split is the canonical two-tracks-in-one-budget partition: the
# primary's optional admissions after the reservation and the whole shadow
# re-run each fit their half, and their sum can never exceed the entry plan.
SHADOW_TRACK_SHARE = 0.5


@dataclass(frozen=True)
class LedgerPlanSnapshot:
    """Entry-state deterministic budget plan of one pipeline invocation.

    Parameters
    ----------
    total_dwu : float
        Total deterministic work-unit budget installed at entry.
    safety : float
        Ledger safety multiplier at entry.
    reserved_tail_dwu : float
        Tail reservation at entry.
    return_reserve_dwu : float
        Modeled return reserve at entry.
    """

    total_dwu: float
    safety: float
    reserved_tail_dwu: float
    return_reserve_dwu: float


@dataclass(frozen=True)
class ShadowPackagePlan:
    """Priced legacy-track shadow package of one pipeline invocation.

    Parameters
    ----------
    shadow_track_dwu : float
        Deterministic budget granted to the shadow re-run's own fresh ledger
        (:data:`SHADOW_TRACK_SHARE` of the entry ledger's free capacity net
        of the final referee cost).
    referee_dwu : float
        Modeled cost of the final two-track referee contest, charged to the
        entry ledger when a displacement occurs.
    package_dwu : float
        Complete reservation (``shadow_track_dwu + referee_dwu``) that must
        fit the entry ledger all-or-nothing before a new arm may win.
    viable : bool
        Whether the priced package is large enough to be worth reserving
        (the shadow track must at least afford its own final scoring);
        a non-viable package fail-closes every new-arm win to legacy.
    """

    shadow_track_dwu: float
    referee_dwu: float
    package_dwu: float
    viable: bool


def is_new_arm_candidate(name: str) -> bool:
    """Return whether a contest candidate belongs to a new-arm family.

    Parameters
    ----------
    name : str
        Full contest candidate name.

    Returns
    -------
    bool
        ``True`` when the candidate's family postdates the terminal-basin
        assumption and its contest win records a displacement.
    """
    return name.startswith(NEW_ARM_FAMILY_PREFIXES)


def new_arms_disabled(config: Optional[LayoutConfig]) -> bool:
    """Return whether every new-arm generation site is gate-closed.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration.

    Returns
    -------
    bool
        ``True`` on the legacy-track shadow re-run (and on explicit
        caller opt-out), where the pipeline must reproduce the
        no-new-arm behavior byte-for-byte.
    """
    return config is not None and bool(getattr(config, DISABLE_NEW_ARMS_ATTR, False))


def record_new_arm_displacement(
    config: Optional[LayoutConfig],
    *,
    route: str,
    winner_name: str,
) -> None:
    """Record that a new-arm family won one contest argmax.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared per-invocation layout configuration. Shallow copies share
        the installed displacement list, so a contest running deep inside
        the solve reaches the outermost orchestrator.
    route : str
        Contest route that selected the new-arm winner (for example
        ``"undirected"``, ``"undirected_sparse_band"``,
        ``"undirected_large_mini"``, or ``"directed"``).
    winner_name : str
        Winning new-arm candidate name.
    """
    if config is None:
        return
    displacements = getattr(config, NEW_ARM_DISPLACEMENTS_ATTR, None)
    if displacements is None:
        displacements = []
        setattr(config, NEW_ARM_DISPLACEMENTS_ATTR, displacements)
    displacements.append({"route": route, "winner_name": winner_name})
    _LOGGER.info(
        "New-arm displacement recorded route=%s winner=%s",
        route,
        winner_name,
    )


def snapshot_ledger_plan(config: Optional[LayoutConfig]) -> Optional[LedgerPlanSnapshot]:
    """Capture the entry-state deterministic budget plan of one invocation.

    Must run BEFORE the primary solve spends from the shared mutable ledger:
    the snapshot fixes the envelope from which the shadow package is priced
    (:func:`price_shadow_package`) and carries the sizing-heuristic total the
    shadow re-run inherits.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration possibly carrying an installed ledger.

    Returns
    -------
    LedgerPlanSnapshot or None
        Entry ledger plan, or ``None`` when no deterministic ledger is
        active (the shadow re-run is then equally unbudgeted).
    """
    from dagua.layout.ops.pipelines.native_budget import LEDGER_ATTR, NativeBudgetLedger

    ledger = getattr(config, LEDGER_ATTR, None) if config is not None else None
    if not isinstance(ledger, NativeBudgetLedger):
        return None
    return LedgerPlanSnapshot(
        total_dwu=float(ledger.total_dwu),
        safety=float(ledger.safety),
        reserved_tail_dwu=float(ledger.reserved_tail_dwu),
        return_reserve_dwu=float(ledger.return_reserve_dwu),
    )


def price_shadow_package(
    config: Optional[LayoutConfig],
    *,
    num_nodes: int,
    num_edges: int,
    has_clusters: bool,
    has_weights: bool,
) -> Optional[ShadowPackagePlan]:
    """Price the complete legacy-track shadow package from entry state.

    Must run BEFORE the primary solve spends from the shared ledger: the
    package is priced against the entry free capacity so the same solve
    always produces the same plan.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration possibly carrying an installed ledger.
    num_nodes : int
        Number of graph nodes (final referee cost model input).
    num_edges : int
        Number of graph edges (final referee cost model input).
    has_clusters : bool
        Whether runtime-visible clusters affect the referee cost.
    has_weights : bool
        Whether runtime-visible edge weights affect the referee cost.

    Returns
    -------
    ShadowPackagePlan or None
        Priced package, or ``None`` when no deterministic ledger is active
        (an unbudgeted row funds an equally unbudgeted shadow, so there is
        nothing to reserve).
    """
    from dagua.layout.ops.pipelines.native_budget import LEDGER_ATTR, NativeBudgetLedger
    from dagua.layout.ops.pipelines.native_cost_model import estimate_v3_referee_cost

    ledger = getattr(config, LEDGER_ATTR, None) if config is not None else None
    if not isinstance(ledger, NativeBudgetLedger):
        return None
    device = str(getattr(config, "device", "cpu")) if config is not None else "cpu"
    referee_cost = estimate_v3_referee_cost(
        int(num_nodes),
        int(num_edges),
        has_clusters=bool(has_clusters),
        has_weights=bool(has_weights),
        device_class="cuda" if device.startswith("cuda") else "cpu",
    )
    referee_dwu = 2.0 * float(referee_cost.reserved_score_dwu)
    free_entry = max(0.0, ledger.capacity_dwu() - ledger.committed_dwu())
    shadow_track_dwu = SHADOW_TRACK_SHARE * max(0.0, free_entry - referee_dwu)
    return ShadowPackagePlan(
        shadow_track_dwu=shadow_track_dwu,
        referee_dwu=referee_dwu,
        package_dwu=shadow_track_dwu + referee_dwu,
        viable=shadow_track_dwu >= referee_dwu and shadow_track_dwu > 0.0,
    )


def install_shadow_reservation_state(
    config: Optional[LayoutConfig],
    plan: Optional[ShadowPackagePlan],
) -> Dict[str, Any]:
    """Install the shared mutable reservation state on one invocation.

    Parameters
    ----------
    config : LayoutConfig, optional
        Outermost-invocation configuration; shallow copies inside the solve
        alias the installed dict, so a contest-level reservation reaches the
        orchestrator.
    plan : ShadowPackagePlan, optional
        Entry-priced shadow package (``None`` on unbudgeted rows).

    Returns
    -------
    dict[str, Any]
        The installed state (``plan`` / ``reserved`` / ``vetoes``).
    """
    state: Dict[str, Any] = {"plan": plan, "reserved": False, "vetoes": []}
    if config is not None:
        setattr(config, SHADOW_RESERVATION_STATE_ATTR, state)
    return state


def try_reserve_shadow_package(
    config: Optional[LayoutConfig],
    *,
    route: str,
    winner_name: str,
) -> bool:
    """Reserve the complete shadow package all-or-nothing on the entry ledger.

    Called at the contest argmax the moment a new-arm candidate would win.
    The reservation is idempotent (one package funds the single whole-solve
    shadow re-run however many contests displace) and all-or-nothing: it
    either fits the live ledger in full or the caller must fall back to the
    legacy-family champion.

    Parameters
    ----------
    config : LayoutConfig, optional
        Contest-level configuration aliasing the orchestrator's reservation
        state and shared ledger.
    route : str
        Contest route requesting the reservation (veto telemetry).
    winner_name : str
        New-arm candidate that would win (veto telemetry).

    Returns
    -------
    bool
        ``True`` when the package is reserved (or the row is unbudgeted, or
        no orchestrator installed reservation state). ``False`` means the
        package was vetoed and the new arm must not win.
    """
    from dagua.layout.ops.pipelines.native_budget import (
        LEDGER_ATTR,
        NativeBudgetLedger,
        reserve_tail,
    )

    state = getattr(config, SHADOW_RESERVATION_STATE_ATTR, None) if config is not None else None
    if not isinstance(state, dict):
        # No orchestrator (direct single-track callers): nothing to reserve
        # and no shadow will run; preserve the pre-existing contest outcome.
        return True
    if bool(state.get("reserved")):
        return True
    plan = state.get("plan")
    if plan is None:
        # Unbudgeted row: the shadow re-run is equally unbudgeted.
        state["reserved"] = True
        return True
    ledger = getattr(config, LEDGER_ATTR, None) if config is not None else None
    veto_reason: Optional[str] = None
    if not isinstance(ledger, NativeBudgetLedger):
        veto_reason = "ledger_missing_at_argmax"
    elif not plan.viable:
        veto_reason = "package_not_viable"
    elif ledger.committed_dwu() + plan.package_dwu > ledger.capacity_dwu():
        veto_reason = "package_does_not_fit"
    if veto_reason is not None:
        state["vetoes"].append({"route": route, "winner_name": winner_name, "reason": veto_reason})
        _LOGGER.info(
            "Shadow package reservation vetoed route=%s winner=%s reason=%s",
            route,
            winner_name,
            veto_reason,
        )
        return False
    reserve_tail(config, plan.package_dwu, "shadow_champion_package")
    state["reserved"] = True
    return True


def resolve_contest_winner(
    config: Optional[LayoutConfig],
    *,
    route: str,
    best_name: str,
    scores: Dict[str, float],
    telemetry: Dict[str, Any],
    select_winner: Callable[[Dict[str, float], Dict[str, Any]], str],
) -> str:
    """Resolve one contest argmax through the shadow-package admission gate.

    A legacy-family winner passes through untouched (byte-inert hot path).
    A new-arm winner is admitted only when :func:`try_reserve_shadow_package`
    commits the complete legacy-track shadow package; on a veto the contest
    fails closed to the legacy-family champion, re-selected by the contest's
    OWN selection function on the legacy-restricted score set (tie semantics
    identical by construction).

    Parameters
    ----------
    config : LayoutConfig, optional
        Contest-level configuration.
    route : str
        Contest route (displacement and veto telemetry).
    best_name : str
        Unrestricted contest argmax.
    scores : dict[str, float]
        Contest score per finalist.
    telemetry : dict[str, Any]
        Contest score telemetry per finalist (selector input).
    select_winner : Callable
        The contest's own ``(scores, telemetry) -> name`` selector.

    Returns
    -------
    str
        The emitted contest winner: ``best_name`` when it is legacy or its
        displacement was admitted, else the legacy-family champion.
    """
    if not is_new_arm_candidate(best_name):
        return best_name
    if try_reserve_shadow_package(config, route=route, winner_name=best_name):
        record_new_arm_displacement(config, route=route, winner_name=best_name)
        return best_name
    legacy_scores = {
        name: score for name, score in scores.items() if not is_new_arm_candidate(name)
    }
    legacy_telemetry = {
        name: value for name, value in telemetry.items() if not is_new_arm_candidate(name)
    }
    fallback = select_winner(legacy_scores, legacy_telemetry)
    _LOGGER.info(
        "Shadow package vetoed: contest fails closed to legacy champion "
        "route=%s displaced_winner=%s legacy_winner=%s",
        route,
        best_name,
        fallback,
    )
    return fallback


def build_legacy_shadow_config(
    config: Optional[LayoutConfig],
    ledger_plan: Optional[LedgerPlanSnapshot],
    package_plan: Optional[ShadowPackagePlan],
) -> LayoutConfig:
    """Build the isolated legacy-track configuration for the shadow re-run.

    The returned config reproduces the caller's entry state with every
    new-arm family disabled. Its FRESH deterministic ledger carries exactly
    the reserved shadow-track budget (never a second copy of the entry plan
    and never the primary run's depleted ledger), so primary and shadow
    together fit the one entry budget. It carries no wall-clock or process
    deadline (the safety track is ledger-only; machine load can never decide
    whether it executes -- review W2-4a F1). Budget-FRACTION sizing
    heuristics (for example the W5 finisher spend cap) read the row's entry
    envelope, not the reservation, so the shadow's terminal chain is sized
    exactly like the legacy run it must reproduce; spend authority stays the
    reserved ledger.

    Parameters
    ----------
    config : LayoutConfig, optional
        The caller's configuration as passed to the outermost invocation.
    ledger_plan : LedgerPlanSnapshot, optional
        Entry budget plan captured by :func:`snapshot_ledger_plan` before
        the primary solve spent anything.
    package_plan : ShadowPackagePlan, optional
        Entry-priced shadow package whose ``shadow_track_dwu`` funds the
        fresh ledger.

    Returns
    -------
    LayoutConfig
        Isolated legacy-track configuration.
    """
    from dagua.layout.ops.pipelines.native_budget import (
        DETERMINISTIC_BUDGET_ATTR,
        PROCESS_DEADLINE_ATTR,
        TOTAL_BUDGET_ATTR,
        WALL_DEADLINE_ATTR,
        install_budget_ledger,
    )

    shadow = copy.copy(config) if config is not None else LayoutConfig()
    setattr(shadow, DISABLE_NEW_ARMS_ATTR, True)
    for stale_attr in (
        WALL_DEADLINE_ATTR,
        PROCESS_DEADLINE_ATTR,
        NEW_ARM_DISPLACEMENTS_ATTR,
        SHADOW_CONTEST_TELEMETRY_ATTR,
        SHADOW_RESERVATION_STATE_ATTR,
        "_dagua_native_terminal_w5_owner",
        "_dagua_native_terminal_w5_done",
    ):
        if hasattr(shadow, stale_attr):
            try:
                delattr(shadow, stale_attr)
            except AttributeError:
                pass
    if ledger_plan is not None and package_plan is not None:
        install_budget_ledger(
            shadow,
            timeout_s=package_plan.shadow_track_dwu,
            safety=ledger_plan.safety,
            reserved_tail_dwu=0.0,
            return_reserve_dwu=0.0,
        )
        # Sizing heuristics keyed to the row's total envelope (W5 spend cap)
        # must match the legacy run being reproduced; the reserved ledger
        # above remains the only spend authority.
        setattr(shadow, TOTAL_BUDGET_ATTR, ledger_plan.total_dwu)
        setattr(shadow, DETERMINISTIC_BUDGET_ATTR, ledger_plan.total_dwu)
    return shadow


def pop_new_arm_displacements(config: Optional[LayoutConfig]) -> list[dict[str, Any]]:
    """Return and clear the displacement log of one pipeline invocation.

    Parameters
    ----------
    config : LayoutConfig, optional
        Configuration carrying the displacement list installed by the
        outermost orchestrator.

    Returns
    -------
    list[dict[str, Any]]
        Recorded displacements (empty when no new arm won any contest).
    """
    if config is None:
        return []
    displacements = getattr(config, NEW_ARM_DISPLACEMENTS_ATTR, None)
    if not isinstance(displacements, list):
        return []
    drained = list(displacements)
    displacements.clear()
    return drained


__all__ = [
    "DISABLE_NEW_ARMS_ATTR",
    "NEW_ARM_DISPLACEMENTS_ATTR",
    "NEW_ARM_FAMILY_PREFIXES",
    "SHADOW_CONTEST_TELEMETRY_ATTR",
    "SHADOW_RESERVATION_STATE_ATTR",
    "SHADOW_TRACK_SHARE",
    "LedgerPlanSnapshot",
    "ShadowPackagePlan",
    "build_legacy_shadow_config",
    "install_shadow_reservation_state",
    "is_new_arm_candidate",
    "new_arms_disabled",
    "pop_new_arm_displacements",
    "price_shadow_package",
    "record_new_arm_displacement",
    "resolve_contest_winner",
    "snapshot_ledger_plan",
    "try_reserve_shadow_package",
]
