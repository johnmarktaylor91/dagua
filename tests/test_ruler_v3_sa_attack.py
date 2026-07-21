"""Regression tests for the GG-3 V3 ruler simulated-annealing attack."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import pytest
import torch

from scripts.ceremony_sa_attack import (
    AGGREGATE_TOLERANCE_FRACTION,
    GG3_BLOCK_AGGREGATE_DELTA_FRACTION,
    PRIMARY_FAITHFULNESS_DROP_THRESHOLD,
    AttackConfig,
    AttackResult,
    ProbeFamily,
    ScoreConfig,
    build_probe_families,
    run_all_attacks,
    run_diagnostics,
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
) -> Tuple[str, float, float, float, bool, float, bool, bool, Tuple[str, ...]]:
    """Return the deterministic fields compared by the tests.

    Parameters
    ----------
    result : AttackResult
        Per-family attack result.

    Returns
    -------
    Tuple[str, float, float, float, bool, float, bool, bool, Tuple[str, ...]]
        Stable comparison tuple.
    """
    return (
        result.family,
        result.best_score,
        result.best_shape_distance,
        result.aggregate_delta_fraction,
        result.blocked,
        result.primary_faithfulness_drop,
        result.sol_variant_blocked,
        result.tier1_tradeoff,
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
    material_shape = result.best_shape_distance >= TEST_ATTACK_CONFIG.shape_distance_threshold
    material_faith = result.primary_faithfulness_drop >= PRIMARY_FAITHFULNESS_DROP_THRESHOLD
    aggregate_held = result.aggregate_delta_fraction <= GG3_BLOCK_AGGREGATE_DELTA_FRACTION
    tradeoff_band = (
        GG3_BLOCK_AGGREGATE_DELTA_FRACTION
        < result.aggregate_delta_fraction
        < AGGREGATE_TOLERANCE_FRACTION
    )
    assert not (material_shape and material_faith and aggregate_held)
    if material_shape and material_faith and tradeoff_band:
        assert result.tier1_tradeoff


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
    for left, right in zip(first, second):
        assert torch.equal(left.best_positions, right.best_positions)


def test_gg3_diagnostics_writes_requested_artifacts(tmp_path: Path) -> None:
    """Assert diagnostics expose positions and save re-score artifacts.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory supplied by pytest.

    Returns
    -------
    None
    """
    results = run_diagnostics(
        seed=101,
        families=("dag",),
        attack_config=AttackConfig(iterations=4, restarts=1),
        score_config=ScoreConfig(
            crossing_samples=1_000,
            neighborhood_samples=64,
            stress_sources=16,
            stress_targets=64,
        ),
        output_dir=tmp_path,
    )
    assert len(results) == 1
    result = results[0]
    assert result.best_positions.shape == result.best_positions.detach().clone().shape
    for suffix in ("baseline.pt", "morph.pt", "facets.json", "compare.png"):
        assert (tmp_path / f"dag_{suffix}").exists()

    payload = json.loads((tmp_path / "dag_facets.json").read_text(encoding="utf-8"))
    assert set(payload) >= {"baseline", "morph", "decomposition"}
    assert {"tiered", "equal", "tier1_only"} <= set(payload["morph"]["scores"])
    assert any(record["code"] == "C1" for record in payload["morph"]["facets"])
