"""Tests for deterministic native modeled-cost ledger infrastructure."""

from __future__ import annotations

import pytest

from dagua.config import LayoutConfig
from dagua.layout.ops.pipelines.native_budget import (
    DECISION_LOG_ATTR,
    DETERMINISTIC_BUDGET_ATTR,
    LEDGER_ATTR,
    PROCESS_DEADLINE_ATTR,
    NativeBudgetLedger,
    admit_native_work,
    charge,
    install_budget_ledger,
    release_reserved_score,
    remaining_dwu,
    remaining_process_s,
)
from dagua.layout.ops.pipelines.native_cost_model import NativeWorkCost


def _cost(
    generation_dwu: float,
    reserved_score_dwu: float,
    family: str = "unit",
) -> NativeWorkCost:
    """Build a native work cost for ledger tests.

    Parameters
    ----------
    generation_dwu : float
        Generation units charged on admission.
    reserved_score_dwu : float
        Reserved scoring units charged on admission.
    family : str, default="unit"
        Synthetic family label.

    Returns
    -------
    NativeWorkCost
        Cost package for admission tests.
    """
    return NativeWorkCost(
        family=family,
        generation_dwu=generation_dwu,
        reserved_score_dwu=reserved_score_dwu,
        metadata={"case": family},
    )


def test_install_budget_ledger_no_install_parity() -> None:
    """Ledger install is explicit and does not backfill process deadlines.

    Returns
    -------
    None
        Assertions validate config attrs.
    """
    config = LayoutConfig()

    assert remaining_dwu(config) is None

    install_budget_ledger(config, timeout_s=100.0)

    ledger = getattr(config, LEDGER_ATTR)
    assert isinstance(ledger, NativeBudgetLedger)
    assert ledger.total_dwu == pytest.approx(100.0)
    assert ledger.safety == pytest.approx(0.90)
    assert getattr(config, DETERMINISTIC_BUDGET_ATTR) == pytest.approx(100.0)
    assert getattr(config, DECISION_LOG_ATTR) is ledger.event_log
    assert not hasattr(config, PROCESS_DEADLINE_ATTR)


def test_charge_admit_and_remaining_dwu_safety_invariant() -> None:
    """Admission charges generation plus reserved score under SAFETY capacity.

    Returns
    -------
    None
        Assertions validate ledger arithmetic and skip logging.
    """
    config = LayoutConfig()
    install_budget_ledger(
        config,
        timeout_s=100.0,
        reserved_tail_dwu=10.0,
        return_reserve_dwu=5.0,
    )

    charge(config, 20.0, "mandatory_spine")
    assert remaining_dwu(config) == pytest.approx(55.0)

    assert admit_native_work(config, _cost(30.0, 10.0), "candidate_a")
    ledger = getattr(config, LEDGER_ATTR)
    assert ledger.spent_dwu == pytest.approx(60.0)
    assert remaining_dwu(config) == pytest.approx(15.0)

    assert not admit_native_work(config, _cost(20.0, 1.0, "too_large"), "candidate_b")
    assert ledger.spent_dwu == pytest.approx(60.0)
    assert ledger.event_log[-1]["event"] == "skip"
    assert ledger.event_log[-1]["reason"] == "candidate_b"


def test_release_reserved_score_refunds_only_score_reserve() -> None:
    """Deterministic rejection releases score reserve but never generation.

    Returns
    -------
    None
        Assertions validate refund arithmetic.
    """
    config = LayoutConfig()
    install_budget_ledger(config, timeout_s=100.0)
    cost = _cost(12.0, 8.0, "rejectable")

    assert admit_native_work(config, cost, "candidate")
    release_reserved_score(config, cost)

    ledger = getattr(config, LEDGER_ATTR)
    assert ledger.spent_dwu == pytest.approx(12.0)
    assert remaining_dwu(config) == pytest.approx(78.0)
    assert ledger.event_log[-1]["event"] == "release_reserved_score"


def test_admit_vetoes_when_wall_reserve_is_exhausted() -> None:
    """Ledger admission keeps the wall backstop as a veto only.

    Returns
    -------
    None
        Assertions validate veto logging and no charge.
    """
    config = LayoutConfig()
    setattr(config, "_dagua_native_deadline_s", 0.0)
    install_budget_ledger(config, timeout_s=100.0, return_reserve_dwu=5.0)

    assert not admit_native_work(config, _cost(1.0, 1.0), "candidate")

    ledger = getattr(config, LEDGER_ATTR)
    assert ledger.spent_dwu == pytest.approx(0.0)
    assert ledger.event_log[-1]["event"] == "veto"
    assert ledger.event_log[-1]["reason"] == "wall_reserve_exhausted"


def test_ledger_decisions_do_not_read_physical_meters_without_wall_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ledger arithmetic remains pure when no wall deadline is installed.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to fail physical meter reads.

    Returns
    -------
    None
        Assertions validate that admission did not call time meters.
    """
    from dagua.layout.ops.pipelines import native_budget

    def fail_meter() -> float:
        """Raise if a physical time meter is read."""
        raise AssertionError("physical time meter read")

    config = LayoutConfig()
    install_budget_ledger(config, timeout_s=100.0)
    monkeypatch.setattr(native_budget.time, "perf_counter", fail_meter)
    monkeypatch.setattr(native_budget.time, "process_time", fail_meter)

    assert admit_native_work(config, _cost(1.0, 1.0), "candidate")
    assert remaining_process_s(config) is None
