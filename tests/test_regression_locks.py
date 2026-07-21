"""Regression-lock harness unit tests: locks must fire on a banked-row drop."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import roundloop_common as rl  # noqa: E402


def _lock(graph: str = "grid_5x5", floor: float = 80.0) -> rl.Lock:
    """Build a synthetic lock.

    Parameters
    ----------
    graph : str, optional
        Lock graph name.
    floor : float, optional
        Lock floor (field best minus tie band).

    Returns
    -------
    rl.Lock
        Synthetic lock with native banked slightly above the floor.
    """
    return rl.Lock(
        graph=graph,
        position_sha256="a" * 64,
        native_extended=floor + 1.0,
        field_best=floor + rl.TIE_BAND,
        field_best_engine="dagre",
        floor=floor,
    )


@pytest.mark.smoke
def test_lock_fires_on_score_drop() -> None:
    """A candidate below the lock floor must FIRE the lock."""
    lock = _lock(floor=80.0)
    result = rl.evaluate_lock(lock, candidate_sha="b" * 64, rescore=lambda: 79.2)
    assert result.status == "fired"
    assert not result.ok
    assert result.new_score == pytest.approx(79.2)
    summary = rl.summarize_lock_results([result])
    assert summary["counts"]["fired"] == 1
    assert summary["fired_graphs"] == ["grid_5x5"]
    assert not summary["ok"]


@pytest.mark.smoke
def test_lock_passes_on_identical_sha_without_rescoring() -> None:
    """Sha identity is the deterministic fast path: no scoring call at all."""
    lock = _lock()

    def _must_not_score() -> float:
        raise AssertionError("rescore must not be called on sha match")

    result = rl.evaluate_lock(lock, candidate_sha="a" * 64, rescore=_must_not_score)
    assert result.status == "pass_sha"
    assert result.ok
    assert result.new_score == pytest.approx(lock.native_extended)


@pytest.mark.smoke
def test_lock_passes_on_rescored_score_above_floor() -> None:
    """A changed position that still clears the floor passes with drift noted."""
    lock = _lock(floor=80.0)
    result = rl.evaluate_lock(lock, candidate_sha="c" * 64, rescore=lambda: 80.4)
    assert result.status == "pass_rescored"
    assert result.ok
    # Above the floor even though below the banked native score: tie band holds.
    assert result.new_score == pytest.approx(80.4)


@pytest.mark.smoke
def test_lock_epsilon_tolerates_float_noise_at_the_floor() -> None:
    """Scores at the floor within LOCK_EPSILON must not fire."""
    lock = _lock(floor=80.0)
    at_floor = rl.evaluate_lock(lock, candidate_sha="d" * 64, rescore=lambda: 80.0)
    assert at_floor.status == "pass_rescored"
    noise = rl.evaluate_lock(
        lock, candidate_sha="d" * 64, rescore=lambda: 80.0 - rl.LOCK_EPSILON / 2
    )
    assert noise.status == "pass_rescored"
    below = rl.evaluate_lock(lock, candidate_sha="d" * 64, rescore=lambda: 80.0 - 1e-6)
    assert below.status == "fired"


@pytest.mark.smoke
def test_lock_missing_candidate_and_failed_rescore() -> None:
    """Missing candidate positions and unscoreable candidates are failures."""
    lock = _lock()
    missing = rl.evaluate_lock(lock, candidate_sha=None, rescore=lambda: 99.0)
    assert missing.status == "missing"
    assert not missing.ok
    assert not rl.summarize_lock_results([missing])["ok"]
    unscoreable = rl.evaluate_lock(lock, candidate_sha="e" * 64, rescore=lambda: None)
    assert unscoreable.status == "fired"


@pytest.mark.smoke
def test_build_locks_only_banks_best_or_tied_rows() -> None:
    """Locks are armed exactly for strictly_best/tied rows with V2 floors."""
    statuses = {"won": "strictly_best", "tied": "tied", "lost": "behind", "gone": "missing"}
    native_rows = {
        name: {"position_sha256": f"{index}" * 64, "extended_composite": 90.0 + index}
        for index, name in enumerate(statuses)
    }
    field_best = {name: {"engine": "elk_layered", "extended_composite": 88.0} for name in statuses}
    locks = rl.build_locks(statuses, native_rows, field_best)
    assert sorted(lock.graph for lock in locks) == ["tied", "won"]
    for lock in locks:
        assert lock.floor == pytest.approx(88.0 - rl.TIE_BAND)
        assert lock.field_best_engine == "elk_layered"


@pytest.mark.smoke
def test_lock_json_roundtrip() -> None:
    """Locks survive JSON serialization unchanged."""
    lock = _lock()
    assert rl.Lock.from_json(lock.to_json()) == lock
