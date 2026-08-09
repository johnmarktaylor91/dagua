"""Shadow-champion terminal contest support (sprint2 W2-4a, cluster C10 core).

The marketplace contests rank candidates by contest-stage referee score, but
the post-contest terminal chain (W5 finisher, global scale sweep, SMACOF
stress polish, continuous facet polish, small-N anneal) is a basin-local
deterministic hill-climb whose gain depends on the starting geometry.
Contest-stage rank is therefore NOT monotone in final rank: a new-arm
candidate can honestly win the contest from a poor terminal basin and emit a
final drawing worse than what the displaced legacy-family winner would have
produced (rome/grafo1000.14 regression, 90.74 -> 84.82, DIAG_GRAFO1000.md).

Fix: whenever a NEW-ARM family displaces the legacy argmax at contest stage,
the contest stashes the legacy-track champion here; the terminal chain then
carries BOTH champions through the entire deterministic terminal chain,
scores both FINAL drawings with the runtime referee, and emits the higher
(ties go to the legacy/incumbent-family track). The gate fires only when a
new-family displacement actually happened, so every other row is byte-inert,
and the shadow run is admitted through the deterministic DWU ledger
(iteration-based only, never wall-clock).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, TypeVar

import torch

from dagua.config import LayoutConfig

_LOGGER = logging.getLogger(__name__)

# Candidate-name prefixes of contest arms that postdate the terminal-anneal
# basin assumption ("new arms"). A displacement by one of these families arms
# the shadow-champion terminal contest. Structural, not row-keyed: the W1-B
# planar-certificate arm is currently the only family strong enough at
# contest stage to displace legacy winners out of good terminal basins; new
# contest arms with the same stage-locality exposure register their prefix
# here.
NEW_ARM_FAMILY_PREFIXES: tuple[str, ...] = ("planar_",)

_SHADOW_CHAMPION_ATTR = "_dagua_native_shadow_champion"

_TelemetryT = TypeVar("_TelemetryT")


@dataclass(frozen=True)
class ShadowChampion:
    """Legacy-track contest champion carried into the terminal contest.

    Parameters
    ----------
    route : str
        Contest route that stashed the champion (``"undirected"`` or
        ``"directed"``).
    winner_name : str
        New-arm candidate name that won the contest argmax.
    shadow_name : str
        Displaced legacy-track argmax candidate name.
    pos : torch.Tensor
        Legacy champion positions with shape ``[N, 2]``, already carried
        through the contest's own emission transforms.
    """

    route: str
    winner_name: str
    shadow_name: str
    pos: torch.Tensor


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
        assumption and can arm the shadow-champion terminal contest.
    """
    return name.startswith(NEW_ARM_FAMILY_PREFIXES)


def legacy_shadow_name(
    best_name: str,
    scores: Mapping[str, float],
    telemetry: Mapping[str, _TelemetryT],
    select_winner: Callable[[Dict[str, float], Dict[str, _TelemetryT]], str],
) -> Optional[str]:
    """Return the displaced legacy-track argmax when a new arm won the contest.

    Parameters
    ----------
    best_name : str
        Contest argmax winner name.
    scores : Mapping[str, float]
        Full-referee score per finalist.
    telemetry : Mapping[str, _TelemetryT]
        Referee telemetry per finalist, keyed like ``scores``.
    select_winner : Callable[[dict[str, float], dict[str, _TelemetryT]], str]
        The contest's own winner-selection function, re-run on the
        legacy-family restriction so tie semantics stay identical.

    Returns
    -------
    str or None
        Legacy-track champion name when a new-family displacement actually
        happened; ``None`` otherwise (the gate stays closed).
    """
    if not is_new_arm_candidate(best_name):
        return None
    legacy_scores = {
        name: score for name, score in scores.items() if not is_new_arm_candidate(name)
    }
    if not legacy_scores:
        return None
    legacy_telemetry = {
        name: entry for name, entry in telemetry.items() if not is_new_arm_candidate(name)
    }
    shadow_name = select_winner(legacy_scores, legacy_telemetry)
    if shadow_name == best_name or shadow_name not in legacy_scores:
        return None
    return shadow_name


def stash_shadow_champion(config: Optional[LayoutConfig], champion: ShadowChampion) -> None:
    """Record the displaced legacy champion for the terminal contest.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared per-invocation layout configuration shared with the terminal
        chain (same seam as ``_dagua_native_terminal_w5_seed_bank``).
    champion : ShadowChampion
        Legacy-track champion payload.
    """
    if config is None:
        return
    setattr(config, _SHADOW_CHAMPION_ATTR, champion)
    _LOGGER.info(
        "Shadow champion armed route=%s winner=%s shadow=%s",
        champion.route,
        champion.winner_name,
        champion.shadow_name,
    )


def pop_shadow_champion(
    config: Optional[LayoutConfig],
    *,
    expected_nodes: int,
) -> Optional[ShadowChampion]:
    """Consume the stashed legacy champion for one terminal contest.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration possibly carrying a stash.
    expected_nodes : int
        Node count of the tensor entering the terminal chain. A component
        contest can stash a champion whose shape does not match the assembled
        full-graph tensor; such a stash is discarded (byte-inert drop).

    Returns
    -------
    ShadowChampion or None
        The stashed champion when it matches the terminal tensor shape.
    """
    if config is None:
        return None
    champion = getattr(config, _SHADOW_CHAMPION_ATTR, None)
    if champion is None:
        return None
    delattr(config, _SHADOW_CHAMPION_ATTR)
    if not isinstance(champion, ShadowChampion):
        return None
    if int(champion.pos.shape[0]) != int(expected_nodes):
        _LOGGER.info(
            "Shadow champion dropped: component-local stash (%d nodes) does not "
            "match terminal tensor (%d nodes)",
            int(champion.pos.shape[0]),
            int(expected_nodes),
        )
        return None
    if not bool(torch.isfinite(champion.pos).all().item()):
        return None
    return champion


__all__ = [
    "NEW_ARM_FAMILY_PREFIXES",
    "ShadowChampion",
    "is_new_arm_candidate",
    "legacy_shadow_name",
    "pop_shadow_champion",
    "stash_shadow_champion",
]
