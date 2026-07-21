"""Deterministic budget helpers for native benchmark admission gates."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from dagua.config import LayoutConfig
from dagua.layout.ops.pipelines.native_cost_model import NativeWorkCost

WALL_DEADLINE_ATTR = "_dagua_native_deadline_s"
PROCESS_DEADLINE_ATTR = "_dagua_native_process_deadline_s"
TOTAL_BUDGET_ATTR = "_dagua_native_total_budget_s"
DETERMINISTIC_BUDGET_ATTR = "_dagua_native_deterministic_budget_s"
LEDGER_ATTR = "_dagua_native_budget_ledger"
DECISION_LOG_ATTR = "_dagua_native_budget_decision_log"
DEFAULT_LEDGER_SAFETY = 0.90


@dataclass
class NativeBudgetLedger:
    """Deterministic modeled-work ledger attached by benchmark seams.

    Parameters
    ----------
    total_dwu : float
        Total deterministic work-unit budget. DWU is the design alias for the
        modeled wall-second unit in the joint budget design.
    safety : float, default=0.90
        Fraction of ``total_dwu`` available for modeled work after residual
        overhead margin.
    spent_dwu : float, default=0.0
        Generation and mandatory work already charged.
    reserved_tail_dwu : float, default=0.0
        Deterministic tail reservation that every optional admission must
        preserve.
    return_reserve_dwu : float, default=0.0
        Modeled return reserve preserved for benchmark cleanup.
    event_log : list[dict[str, object]]
        Deterministic decision records for replay and debugging.
    """

    total_dwu: float
    safety: float = DEFAULT_LEDGER_SAFETY
    spent_dwu: float = 0.0
    reserved_tail_dwu: float = 0.0
    return_reserve_dwu: float = 0.0
    event_log: list[dict[str, Any]] = field(default_factory=list)

    def capacity_dwu(self) -> float:
        """Return the safety-adjusted modeled-work capacity.

        Returns
        -------
        float
            ``safety * total_dwu`` clamped to zero.
        """
        return max(0.0, float(self.safety) * max(0.0, float(self.total_dwu)))

    def committed_dwu(self) -> float:
        """Return modeled units already committed outside the next candidate.

        Returns
        -------
        float
            Sum of spent work, tail reservation, and return reserve.
        """
        return (
            max(0.0, float(self.spent_dwu))
            + max(0.0, float(self.reserved_tail_dwu))
            + max(0.0, float(self.return_reserve_dwu))
        )


def install_process_budget(config: LayoutConfig, timeout_s: float) -> None:
    """Attach deterministic process-time budget metadata to a layout config.

    Parameters
    ----------
    config : LayoutConfig
        Prepared benchmark layout configuration.
    timeout_s : float
        Total CPU seconds available for deterministic admission decisions.

    Returns
    -------
    None
        The function mutates ``config`` in place.
    """
    install_budget_ledger(config, timeout_s)


def install_budget_ledger(
    config: LayoutConfig,
    timeout_s: float,
    safety: float = DEFAULT_LEDGER_SAFETY,
    reserved_tail_dwu: float = 0.0,
    return_reserve_dwu: float = 0.0,
) -> None:
    """Attach a deterministic modeled-work ledger to a layout config.

    Parameters
    ----------
    config : LayoutConfig
        Prepared benchmark layout configuration.
    timeout_s : float
        Total deterministic work units available; one DWU is one modeled
        wall-second equivalent.
    safety : float, default=0.90
        Safety multiplier capped at ``0.90`` per the joint budget decision.
    reserved_tail_dwu : float, default=0.0
        Initial finalist or mandatory-tail reservation.
    return_reserve_dwu : float, default=0.0
        Initial modeled return reserve.

    Returns
    -------
    None
        The function mutates ``config`` in place and does not install legacy
        process-time metadata.
    """
    total_dwu = max(0.001, float(timeout_s))
    capped_safety = min(DEFAULT_LEDGER_SAFETY, max(0.0, float(safety)))
    ledger = NativeBudgetLedger(
        total_dwu=total_dwu,
        safety=capped_safety,
        reserved_tail_dwu=max(0.0, float(reserved_tail_dwu)),
        return_reserve_dwu=max(0.0, float(return_reserve_dwu)),
    )
    setattr(config, LEDGER_ATTR, ledger)
    setattr(config, TOTAL_BUDGET_ATTR, total_dwu)
    setattr(config, DETERMINISTIC_BUDGET_ATTR, total_dwu)
    setattr(config, DECISION_LOG_ATTR, ledger.event_log)


def _ledger(config: Optional[LayoutConfig]) -> Optional[NativeBudgetLedger]:
    """Return the installed native budget ledger, if any.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration.

    Returns
    -------
    NativeBudgetLedger or None
        Ledger attached by ``install_budget_ledger``.
    """
    candidate = getattr(config, LEDGER_ATTR, None) if config is not None else None
    return candidate if isinstance(candidate, NativeBudgetLedger) else None


def _record_decision(
    config: Optional[LayoutConfig],
    event: str,
    reason: str,
    amount_dwu: float,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Append one deterministic budget decision record.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration carrying a ledger.
    event : str
        Decision class such as ``"admit"``, ``"skip"``, ``"charge"`` or
        ``"veto"``.
    reason : str
        Stable reason string for telemetry and replay.
    amount_dwu : float
        Candidate or charged modeled units.
    metadata : dict[str, object], optional
        Additional JSON-compatible details.

    Returns
    -------
    None
        The function mutates the ledger event log when installed.
    """
    ledger = _ledger(config)
    if ledger is None:
        return
    record: dict[str, Any] = {
        "event": event,
        "reason": str(reason),
        "amount_dwu": max(0.0, float(amount_dwu)),
        "spent_dwu": max(0.0, float(ledger.spent_dwu)),
        "reserved_tail_dwu": max(0.0, float(ledger.reserved_tail_dwu)),
        "return_reserve_dwu": max(0.0, float(ledger.return_reserve_dwu)),
        "remaining_dwu": remaining_dwu(config),
    }
    if metadata:
        record["metadata"] = dict(metadata)
    ledger.event_log.append(record)


def charge(config: Optional[LayoutConfig], cost_dwu: float, reason: str) -> None:
    """Charge mandatory or already-admitted modeled work to the ledger.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration carrying an optional ledger.
    cost_dwu : float
        Deterministic work units to debit.
    reason : str
        Stable charge reason for decision logs.

    Returns
    -------
    None
        The function mutates the ledger when installed. With no ledger, it is a
        no-op so existing native code paths remain unchanged in B1.
    """
    ledger = _ledger(config)
    if ledger is None:
        return
    amount = max(0.0, float(cost_dwu))
    ledger.spent_dwu += amount
    _record_decision(config, "charge", reason, amount)


def reserve_tail(config: Optional[LayoutConfig], cost_dwu: float, reason: str) -> float:
    """Reserve modeled tail work that every later admission must preserve.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration carrying an optional ledger.
    cost_dwu : float
        Deterministic work units to add to the tail reservation.
    reason : str
        Stable reservation reason for decision logs.

    Returns
    -------
    float
        Reserved deterministic work units. ``0.0`` when no ledger is active.
    """
    ledger = _ledger(config)
    if ledger is None:
        return 0.0
    amount = max(0.0, float(cost_dwu))
    ledger.reserved_tail_dwu += amount
    _record_decision(config, "reserve_tail", reason, amount)
    return amount


def release_tail_reservation(
    config: Optional[LayoutConfig],
    reservation_dwu: float,
    reason: str,
) -> None:
    """Release deterministic tail work after a deterministic non-budget exit.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration carrying an optional ledger.
    reservation_dwu : float
        Tail reservation units previously returned by ``reserve_tail``.
    reason : str
        Stable release reason for decision logs.

    Returns
    -------
    None
        The function mutates the ledger when installed.
    """
    ledger = _ledger(config)
    if ledger is None:
        return
    amount = min(max(0.0, float(reservation_dwu)), max(0.0, float(ledger.reserved_tail_dwu)))
    ledger.reserved_tail_dwu -= amount
    _record_decision(config, "release_tail", reason, amount)


def _fits_ledger(ledger: NativeBudgetLedger, cost: NativeWorkCost) -> bool:
    """Return whether a candidate package fits the safety invariant.

    Parameters
    ----------
    ledger : NativeBudgetLedger
        Installed modeled-work ledger.
    cost : NativeWorkCost
        Candidate generation and score-reserve cost.

    Returns
    -------
    bool
        ``True`` when the full package fits inside ``safety * total_dwu``.
    """
    package = (
        max(0.0, float(cost.generation_dwu))
        + max(0.0, float(cost.reserved_score_dwu))
        + max(0.0, float(cost.mandatory_tail_dwu))
    )
    return ledger.committed_dwu() + package <= ledger.capacity_dwu()


def admit_native_work(
    config: Optional[LayoutConfig],
    cost: NativeWorkCost,
    reserve_reason: str,
) -> bool:
    """Admit and charge optional native work if the ledger invariant fits.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration carrying an optional ledger.
    cost : NativeWorkCost
        Candidate generation and reserved scoring package.
    reserve_reason : str
        Stable decision reason recorded for replay.

    Returns
    -------
    bool
        ``True`` when no ledger is active, or when the candidate package was
        admitted and charged. ``False`` records a skip/veto without charging.
    """
    ledger = _ledger(config)
    if ledger is None:
        return True
    amount = max(0.0, float(cost.generation_dwu)) + max(0.0, float(cost.reserved_score_dwu))
    metadata = {"family": cost.family, **cost.metadata}
    if wall_reserve_exhausted(config, ledger.return_reserve_dwu):
        _record_decision(config, "veto", "wall_reserve_exhausted", amount, metadata)
        return False
    if not _fits_ledger(ledger, cost):
        _record_decision(config, "skip", reserve_reason, amount, metadata)
        return False
    ledger.spent_dwu += max(0.0, float(cost.generation_dwu))
    ledger.spent_dwu += max(0.0, float(cost.reserved_score_dwu))
    _record_decision(config, "admit", reserve_reason, amount, metadata)
    return True


def release_reserved_score(config: Optional[LayoutConfig], cost: NativeWorkCost) -> None:
    """Release a reserved score charge after deterministic rejection.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration carrying an optional ledger.
    cost : NativeWorkCost
        Candidate package whose score reserve should be released. Generation
        work is never refunded.

    Returns
    -------
    None
        The function mutates the ledger when installed.
    """
    ledger = _ledger(config)
    if ledger is None:
        return
    amount = min(max(0.0, float(cost.reserved_score_dwu)), max(0.0, float(ledger.spent_dwu)))
    ledger.spent_dwu -= amount
    _record_decision(config, "release_reserved_score", cost.family, amount, cost.metadata)


def remaining_dwu(config: Optional[LayoutConfig]) -> Optional[float]:
    """Return remaining safety-adjusted deterministic work units.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration carrying an optional ledger.

    Returns
    -------
    float or None
        Remaining DWU after spent and reservations, clamped to zero. ``None``
        means no modeled-work ledger is active.
    """
    ledger = _ledger(config)
    if ledger is None:
        return None
    return max(0.0, ledger.capacity_dwu() - ledger.committed_dwu())


def remaining_wall_s(config: Optional[LayoutConfig]) -> Optional[float]:
    """Return wall-clock seconds remaining before the hard benchmark deadline.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration carrying optional benchmark metadata.

    Returns
    -------
    float or None
        Remaining wall seconds, or ``None`` when no wall deadline is known.
    """
    deadline = getattr(config, WALL_DEADLINE_ATTR, None) if config is not None else None
    if deadline is None:
        return None
    return float(deadline) - time.perf_counter()


def remaining_process_s(config: Optional[LayoutConfig]) -> Optional[float]:
    """Return deterministic CPU seconds remaining for admission decisions.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration carrying optional benchmark metadata.

    Returns
    -------
    float or None
        Remaining process CPU seconds, or ``None`` when no deterministic
        budget can be inferred. A legacy wall-only benchmark config initializes
        the process budget from the current wall remainder once, preserving
        existing unloaded behavior while making later checks CPU-time based.
    """
    if config is None:
        return None
    ledger_remaining = remaining_dwu(config)
    if ledger_remaining is not None:
        return ledger_remaining
    process_deadline = getattr(config, PROCESS_DEADLINE_ATTR, None)
    if process_deadline is None:
        remaining = remaining_wall_s(config)
        if remaining is None:
            return None
        process_deadline = time.process_time() + max(0.0, float(remaining))
        setattr(config, PROCESS_DEADLINE_ATTR, process_deadline)
        return float(remaining)
    return float(process_deadline) - time.process_time()


def wall_reserve_exhausted(config: Optional[LayoutConfig], reserve_s: float) -> bool:
    """Return whether the hard wall-clock return reserve is exhausted.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration carrying optional wall deadline metadata.
    reserve_s : float
        Required wall seconds to reserve for cleanup and returning a result.

    Returns
    -------
    bool
        ``True`` only when a wall deadline exists and its reserve is gone.
    """
    remaining = remaining_wall_s(config)
    return remaining is not None and remaining <= float(reserve_s)


def available_process_work_s(
    config: Optional[LayoutConfig],
    reserve_s: float,
) -> Optional[float]:
    """Return process-time work seconds available before a return reserve.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration carrying optional benchmark metadata.
    reserve_s : float
        Deterministic CPU seconds reserved for cleanup and scoring.

    Returns
    -------
    float or None
        Available deterministic work seconds, clamped to zero. ``None`` means
        no benchmark budget is active, so optional admission is unchanged.
    """
    remaining = remaining_dwu(config)
    if remaining is None:
        remaining = remaining_process_s(config)
    if remaining is None:
        return None
    return max(0.0, float(remaining) - float(reserve_s))


def has_process_budget(
    config: Optional[LayoutConfig],
    min_remaining_s: float,
    reserve_s: float,
) -> bool:
    """Return whether deterministic admission has enough process budget.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared layout configuration carrying optional benchmark metadata.
    min_remaining_s : float
        Minimum remaining deterministic seconds before starting optional work.
    reserve_s : float
        Deterministic return reserve that must also remain available.

    Returns
    -------
    bool
        ``True`` when no budget is active, or when the process-time remainder
        exceeds both requested thresholds.
    """
    remaining = remaining_dwu(config)
    if remaining is None:
        remaining = remaining_process_s(config)
    required_remaining = max(float(min_remaining_s), float(reserve_s))
    return remaining is None or remaining > required_remaining
