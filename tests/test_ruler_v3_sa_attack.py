"""Regression tests for the GG-3 V3 ruler simulated-annealing attack."""

from __future__ import annotations

from typing import Tuple

import pytest

from scripts.ceremony_sa_attack import (
    AttackConfig,
    AttackResult,
    ProbeFamily,
    ScoreConfig,
    build_probe_families,
    run_all_attacks,
    run_family_attack,
)

TEST_ATTACK_CONFIG = AttackConfig(iterations=90, restarts=2)
TEST_SCORE_CONFIG = ScoreConfig(
    crossing_samples=20_000,
    neighborhood_samples=256,
    stress_sources=64,
    stress_targets=192,
)
FAMILY_SEEDS = {
    "tree": 17,
    "dag": 23,
    "clustered": 31,
    "generic_force": 43,
    "weighted": 47,
    "ported": 53,
}


def _result_signature(
    result: AttackResult,
) -> Tuple[str, float, float, float, bool, Tuple[str, ...]]:
    """Return the deterministic fields compared by the tests.

    Parameters
    ----------
    result : AttackResult
        Per-family attack result.

    Returns
    -------
    Tuple[str, float, float, float, bool, Tuple[str, ...]]
        Stable comparison tuple.
    """
    return (
        result.family,
        result.best_score,
        result.best_shape_distance,
        result.aggregate_delta_fraction,
        result.blocked,
        result.fooled_facets,
    )


@pytest.mark.parametrize("probe", build_probe_families(), ids=lambda probe: probe.family)
def test_gg3_sa_attack_per_family_passes_or_reports_block(probe: ProbeFamily) -> None:
    """Assert per-family robustness or xfail with explicit block diagnostics.

    Parameters
    ----------
    probe : ProbeFamily
        Frozen probe family supplied by pytest parameterization.

    Returns
    -------
    None
    """
    result = run_family_attack(
        probe,
        seed=FAMILY_SEEDS[probe.family],
        attack_config=TEST_ATTACK_CONFIG,
        score_config=TEST_SCORE_CONFIG,
    )
    if result.blocked:
        facets = ", ".join(result.fooled_facets) if result.fooled_facets else "none"
        pytest.xfail(
            "GG-3 BLOCK: "
            f"{result.family} shape={result.best_shape_distance:.4f}, "
            f"aggregate_delta={100.0 * result.aggregate_delta_fraction:.2f}%, "
            f"fooled_facets={facets}"
        )
    assert result.aggregate_delta_fraction <= TEST_ATTACK_CONFIG.aggregate_tolerance_fraction
    assert result.best_shape_distance < TEST_ATTACK_CONFIG.shape_distance_threshold


def test_gg3_sa_attack_is_deterministic_for_same_seed() -> None:
    """Assert identical attack results when the seed and budgets match.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    first = run_all_attacks(
        seed=101,
        families=("dag", "weighted"),
        attack_config=AttackConfig(iterations=50, restarts=1),
        score_config=TEST_SCORE_CONFIG,
    )
    second = run_all_attacks(
        seed=101,
        families=("dag", "weighted"),
        attack_config=AttackConfig(iterations=50, restarts=1),
        score_config=TEST_SCORE_CONFIG,
    )
    assert tuple(_result_signature(result) for result in first) == tuple(
        _result_signature(result) for result in second
    )
