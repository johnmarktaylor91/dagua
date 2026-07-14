"""Tests for Lane C visual parity ledger generation."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict

from scripts.visual_parity import ledger
from scripts.visual_parity.io import read_ledger

LEDGER_PATH = Path(".project-context/research/sprint_visual_parity_v2/ledger.json")


def _locked_by_id() -> Dict[str, Dict[str, object]]:
    """Return locked rows from the committed ledger keyed by row id.

    Returns
    -------
    Dict[str, Dict[str, object]]
        Locked rows by id.
    """

    data = read_ledger(LEDGER_PATH)
    return {str(row["row_id"]): row for row in ledger.locked_rows(data)}


def test_init_seeds_prior_locks_for_v2_revalidation() -> None:
    """Ledger seed should contain rows, ratchets, warnings, and lock rows.

    Returns
    -------
    None
        The test asserts the committed ledger structure.
    """

    data = read_ledger(LEDGER_PATH)

    assert data["ratchets"]["global_in_tol_floor_pct"] == 85.0
    assert data["rows"]
    assert data["knobs"]
    assert ledger.locked_rows(data)
    assert any("tripwire_status.json missing" in warning for warning in data.get("warnings", []))


def test_generated_lock_regeneration_is_byte_identical() -> None:
    """Generated lock tests should be deterministic.

    Returns
    -------
    None
        The test asserts byte identity with the committed generated file.
    """

    data = read_ledger(LEDGER_PATH)

    assert ledger.render_lock_tests(data) == Path("tests/test_visual_parity_locks.py").read_text(
        encoding="utf-8"
    )


def test_editing_locked_current_value_changes_expectation() -> None:
    """Changing a locked row current value should break generated expectations.

    Returns
    -------
    None
        The test asserts lock rows are sensitive to metric drift.
    """

    data = read_ledger(LEDGER_PATH)
    locked = ledger.locked_rows(data)[0]
    edited = copy.deepcopy(locked)
    edited["metrics"][0]["current"] = 0.5

    assert ledger.lock_expectation(edited) != ledger.lock_expectation(locked)


def test_stalled_p0_p1_rows_are_stop_blockers() -> None:
    """Stalled P0/P1 rows without waiver or residual should block STOP.

    Returns
    -------
    None
        The test asserts F12 blocker detection.
    """

    data = read_ledger(LEDGER_PATH)
    row = copy.deepcopy(data["rows"][0])
    row["priority"] = "P1"
    row["parity_status"] = "stalled"
    row["waiver"] = None
    row["residual_class"] = None
    data["rows"] = [row]

    assert ledger.stalled_stop_blockers(data) == [row]
