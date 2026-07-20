"""Deterministic budget helpers for native benchmark admission gates."""

from __future__ import annotations

import time
from typing import Optional

from dagua.config import LayoutConfig

WALL_DEADLINE_ATTR = "_dagua_native_deadline_s"
PROCESS_DEADLINE_ATTR = "_dagua_native_process_deadline_s"
TOTAL_BUDGET_ATTR = "_dagua_native_total_budget_s"
DETERMINISTIC_BUDGET_ATTR = "_dagua_native_deterministic_budget_s"


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
    budget_s = max(0.001, float(timeout_s))
    setattr(config, PROCESS_DEADLINE_ATTR, time.process_time() + budget_s)
    setattr(config, TOTAL_BUDGET_ATTR, budget_s)
    setattr(config, DETERMINISTIC_BUDGET_ATTR, budget_s)


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
    remaining = remaining_process_s(config)
    required_remaining = max(float(min_remaining_s), float(reserve_s))
    return remaining is None or remaining > required_remaining
