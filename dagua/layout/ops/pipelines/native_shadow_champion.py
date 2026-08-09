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

Fix (review W2-4a F1-F3 shape): whenever a NEW-ARM family wins any contest
argmax, the contest records the displacement here; the outermost native
pipeline invocation then re-runs the ENTIRE pipeline once with every new-arm
family disabled -- an isolated legacy track with its own fresh deterministic
DWU ledger and no wall-clock deadline -- scores both FINAL drawings with the
runtime referee, and emits the higher (ties go to the legacy track). The
legacy track is therefore the byte-identical output of the actual pipeline
with all new arms absent: finalist selection, contest-local tails, seed
banks, budget consumption, and the complete terminal chain all replay from
legacy state. The safety track can never be skipped by budget or load: it
does not draw on the primary run's ledger and carries no wall-clock veto.
The gate fires only when a new-arm family actually won a contest, so every
other row is byte-inert.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Optional

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
    the snapshot is the complete fixed budget plan that the legacy-track
    shadow re-run receives as its own fresh ledger (all-or-nothing by
    construction -- the safety track never bids against the primary run's
    remaining budget and can never be vetoed).

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


def build_legacy_shadow_config(
    config: Optional[LayoutConfig],
    ledger_plan: Optional[LedgerPlanSnapshot],
) -> LayoutConfig:
    """Build the isolated legacy-track configuration for the shadow re-run.

    The returned config reproduces the caller's entry state with every
    new-arm family disabled: a FRESH deterministic ledger carrying the entry
    budget plan (never the primary run's depleted ledger), and no wall-clock
    or process deadline (the safety track is ledger-only; machine load can
    never decide whether it executes -- review W2-4a F1).

    Parameters
    ----------
    config : LayoutConfig, optional
        The caller's configuration as passed to the outermost invocation.
    ledger_plan : LedgerPlanSnapshot, optional
        Entry budget plan captured by :func:`snapshot_ledger_plan` before
        the primary solve spent anything.

    Returns
    -------
    LayoutConfig
        Isolated legacy-track configuration.
    """
    from dagua.layout.ops.pipelines.native_budget import (
        PROCESS_DEADLINE_ATTR,
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
        "_dagua_native_terminal_w5_owner",
        "_dagua_native_terminal_w5_done",
    ):
        if hasattr(shadow, stale_attr):
            try:
                delattr(shadow, stale_attr)
            except AttributeError:
                pass
    if ledger_plan is not None:
        install_budget_ledger(
            shadow,
            timeout_s=ledger_plan.total_dwu,
            safety=ledger_plan.safety,
            reserved_tail_dwu=ledger_plan.reserved_tail_dwu,
            return_reserve_dwu=ledger_plan.return_reserve_dwu,
        )
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
    "LedgerPlanSnapshot",
    "build_legacy_shadow_config",
    "is_new_arm_candidate",
    "new_arms_disabled",
    "pop_new_arm_displacements",
    "record_new_arm_displacement",
    "snapshot_ledger_plan",
]
