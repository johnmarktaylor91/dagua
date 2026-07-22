"""Regression tests for the GG-3 V3 ruler simulated-annealing attack."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import pytest
import torch

import scripts.ceremony_sa_attack as sa_attack
from dagua.eval.ruler_v3 import RulerV3Facet, RulerV3Result
from scripts.ceremony_sa_attack import (
    AGGREGATE_TOLERANCE_FRACTION,
    GG3_BLOCK_AGGREGATE_DELTA_FRACTION,
    GG3_VERDICT_BLOCK,
    GG3_VERDICT_PASS_DEGENERATE_ESCAPE,
    GG3_VERDICT_PASS_WITH_T1_TRADEOFF,
    PRIMARY_FAITHFULNESS_DROP_THRESHOLD,
    TWO_LAYOUT_BUYBACK_BAR,
    AttackConfig,
    AttackResult,
    ProbeFamily,
    ScoreConfig,
    _aggregate_delta_fraction,
    _gg3_gate_verdict,
    _objective,
    build_probe_families,
    format_results_table,
    primary_faithfulness_drop,
    probe_by_family,
    procrustes_shape_distance,
    run_all_attacks,
    run_diagnostics,
    run_family_attack,
    two_layout_buyback_decomposition,
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
SAVED_GG3_DIR = Path.home() / ".claude/research/dagua/megasprint/gg3_fresh"
OFFICIAL_GG3_DIAG_DIR = Path("/tmp/sol_gg3_diag")


def _result_signature(
    result: AttackResult,
) -> Tuple[object, ...]:
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
        result.objective_mode,
        result.seed,
        result.best_score,
        result.best_shape_distance,
        result.aggregate_delta_fraction,
        result.blocked,
        result.primary_faithfulness_drop,
        result.sol_variant_blocked,
        result.tier1_tradeoff,
        result.gate_verdict,
        result.tier1_only_drop,
        result.max_blockregion_tier1_only_drop,
        result.blockregion_shape_distance,
        result.blockregion_aggregate_delta_fraction,
        result.blockregion_primary_faithfulness_drop,
        result.blockregion_severe_g6_floor_breach,
        result.buyback_headroom,
        result.two_layout_buyback,
        result.buyback_pass_through,
        result.tiered_drop,
        result.margin_audit_flag,
        result.margin_audit_margin,
        result.margin_audit_allowance,
        result.severe_g6_floor_breach,
        result.degenerate_escape,
        result.fooled_facets,
        result.gaining_facets,
    )


def _result_from_saved_payload(payload: dict) -> RulerV3Result:
    """Create a V3 result from a saved GG-3 facets payload section.

    Parameters
    ----------
    payload : dict
        Saved ``baseline`` or ``morph`` section from ``*_facets.json``.

    Returns
    -------
    RulerV3Result
        Result reconstructed from persisted facet records.
    """
    facets = {
        record["code"]: RulerV3Facet(
            code=record["code"],
            name=record["name"],
            tier=int(record["tier"]),
            score=None if record["score"] is None else float(record["score"]),
            base_weight=float(record["base_weight"]),
            effective_weight=float(record["effective_weight"]),
            applicable=bool(record["applicable"]),
            applicability_reason=str(record["applicability_reason"]),
            metadata=dict(record["metadata"]),
        )
        for record in payload["facets"]
    }
    return RulerV3Result(
        facets=facets,
        scores={str(key): float(value) for key, value in payload["scores"].items()},
        flags=tuple(str(flag) for flag in payload["flags"]),
        applicability={code: facet.applicable for code, facet in facets.items()},
        coverage={str(key): int(value) for key, value in payload["coverage"].items()},
        metadata={},
    )


def _load_saved_case(
    family: str,
) -> Tuple[RulerV3Result, RulerV3Result, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load saved GG-3 facet and geometry artifacts for one family.

    Parameters
    ----------
    family : str
        Saved family name.

    Returns
    -------
    Tuple[RulerV3Result, RulerV3Result, torch.Tensor, torch.Tensor, torch.Tensor]
        Baseline result, morph result, baseline positions, morph positions, and
        edge index.
    """
    facets_payload = json.loads((SAVED_GG3_DIR / f"{family}_facets.json").read_text())
    baseline_pt = torch.load(SAVED_GG3_DIR / f"{family}_baseline.pt", map_location="cpu")
    morph_pt = torch.load(SAVED_GG3_DIR / f"{family}_morph.pt", map_location="cpu")
    return (
        _result_from_saved_payload(facets_payload["baseline"]),
        _result_from_saved_payload(facets_payload["morph"]),
        baseline_pt["positions"],
        morph_pt["positions"],
        baseline_pt["edges"],
    )


def _load_official_diag_results(family: str) -> Tuple[RulerV3Result, RulerV3Result]:
    """Load official-seed GG-3 facet diagnostics for one family.

    Parameters
    ----------
    family : str
        Saved family name.

    Returns
    -------
    Tuple[RulerV3Result, RulerV3Result]
        Baseline and morph result reconstructed from official diagnostics.
    """
    facets_payload = json.loads(
        (OFFICIAL_GG3_DIAG_DIR / family / f"{family}_facets.json").read_text()
    )
    return (
        _result_from_saved_payload(facets_payload["baseline"]),
        _result_from_saved_payload(facets_payload["morph"]),
    )


def _load_official_diag_positions(family: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load official-seed GG-3 baseline and morph positions for one family.

    Parameters
    ----------
    family : str
        Saved family name.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Baseline and morph position tensors with shape ``[N, 2]``.
    """
    baseline_pt = torch.load(
        OFFICIAL_GG3_DIAG_DIR / family / f"{family}_baseline.pt",
        map_location="cpu",
    )
    morph_pt = torch.load(
        OFFICIAL_GG3_DIAG_DIR / family / f"{family}_morph.pt",
        map_location="cpu",
    )
    return baseline_pt["positions"], morph_pt["positions"]


def _synthetic_result(
    *,
    tiered: float,
    tier1_only: float,
    c3: float,
    c1: float = 0.9,
    g6_weighted_ksm: float | None = None,
    g6_local_weight_monotonicity: float | None = None,
) -> RulerV3Result:
    """Create a compact V3 result for GG-3 gate-only tests.

    Parameters
    ----------
    tiered : float
        Synthetic tiered score.
    tier1_only : float
        Synthetic tier1-only score.
    c3 : float
        Synthetic C3 score.
    c1 : float, optional
        Synthetic C1 score.
    g6_weighted_ksm : float | None, optional
        Optional weighted KSM score.
    g6_local_weight_monotonicity : float | None, optional
        Optional local weight monotonicity score.

    Returns
    -------
    RulerV3Result
        Minimal score result with applicable requested facets.
    """
    facet_specs = [
        ("C1", 1, c1, 4.0),
        ("C3", 1, c3, 4.0),
    ]
    if g6_weighted_ksm is not None:
        facet_specs.append(("G6_weighted_ksm", 1, g6_weighted_ksm, 4.0))
    if g6_local_weight_monotonicity is not None:
        facet_specs.append(("G6_local_weight_monotonicity", 3, g6_local_weight_monotonicity, 1.0))
    facets = {
        code: RulerV3Facet(
            code=code,
            name=code,
            tier=tier,
            score=float(score),
            base_weight=weight,
            effective_weight=weight,
            applicable=True,
            applicability_reason="synthetic",
            metadata={},
        )
        for code, tier, score, weight in facet_specs
    }
    return RulerV3Result(
        facets=facets,
        scores={"tiered": tiered, "equal": tiered, "tier1_only": tier1_only},
        flags=tuple(),
        applicability={code: True for code in facets},
        coverage={
            "applicable_facets": len(facets),
            "total_facets": len(facets),
            "tier1_applicable_facets": sum(1 for facet in facets.values() if facet.tier == 1),
            "applicable_groups": 1 if g6_weighted_ksm is not None else 0,
        },
        metadata={},
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
        facets = ", ".join(result.gaining_facets) if result.gaining_facets else "none"
        pytest.xfail(
            "GG-3 BLOCK: "
            f"{result.family} shape={result.best_shape_distance:.4f}, "
            f"aggregate_delta={100.0 * result.aggregate_delta_fraction:.2f}%, "
            f"gaining_facets={facets}"
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
    if material_shape and material_faith and aggregate_held:
        if result.two_layout_buyback >= TWO_LAYOUT_BUYBACK_BAR:
            assert result.gate_verdict == GG3_VERDICT_BLOCK
        else:
            assert result.gate_verdict in {
                GG3_VERDICT_PASS_DEGENERATE_ESCAPE,
                GG3_VERDICT_PASS_WITH_T1_TRADEOFF,
            }
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
    assert payload["attack"]["max_blockregion_tier1_only_drop"] == pytest.approx(
        result.max_blockregion_tier1_only_drop
    )
    assert "max block-region T1 drop" in format_results_table(results)


def test_joint_gg3_gate_saved_generic_force_passes_with_t1_tradeoff() -> None:
    """Assert saved generic-force artifact is flagged but non-blocking.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    baseline, morph, baseline_pos, morph_pos, edges = _load_saved_case("generic_force")
    shape_distance = procrustes_shape_distance(baseline_pos, morph_pos)
    aggregate_delta = _aggregate_delta_fraction(
        float(morph.scores["tiered"]),
        float(baseline.scores["tiered"]),
    )
    verdict = _gg3_gate_verdict(
        shape_distance=shape_distance,
        aggregate_delta_fraction=aggregate_delta,
        faithfulness_drop=primary_faithfulness_drop(baseline, morph),
        baseline_result=baseline,
        candidate_result=morph,
        baseline_pos=baseline_pos,
        edges=edges,
    )
    assert verdict.verdict == GG3_VERDICT_PASS_WITH_T1_TRADEOFF
    assert verdict.tier1_tradeoff
    assert verdict.tier1_only_drop == pytest.approx(4.7791, abs=0.01)
    assert not verdict.degenerate_escape
    assert not verdict.severe_g6_floor_breach


def test_joint_gg3_gate_saved_weighted_passes_as_degenerate_escape() -> None:
    """Assert saved weighted artifact qualifies for the narrow escape waiver.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    baseline, morph, baseline_pos, morph_pos, edges = _load_saved_case("weighted")
    shape_distance = procrustes_shape_distance(baseline_pos, morph_pos)
    aggregate_delta = _aggregate_delta_fraction(
        float(morph.scores["tiered"]),
        float(baseline.scores["tiered"]),
    )
    verdict = _gg3_gate_verdict(
        shape_distance=shape_distance,
        aggregate_delta_fraction=aggregate_delta,
        faithfulness_drop=primary_faithfulness_drop(baseline, morph),
        baseline_result=baseline,
        candidate_result=morph,
        baseline_pos=baseline_pos,
        edges=edges,
    )
    assert verdict.verdict == GG3_VERDICT_PASS_DEGENERATE_ESCAPE
    assert verdict.degenerate_escape
    assert verdict.tier1_only_drop < 0.0
    assert not verdict.severe_g6_floor_breach


def test_joint_gg3_gate_blocks_material_tier1_only_drop() -> None:
    """Assert the pre-registered material tier1 trapdoor blocks.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    baseline = _synthetic_result(tiered=90.0, tier1_only=95.0, c1=0.9, c3=0.95)
    morph = _synthetic_result(tiered=89.0, tier1_only=89.5, c1=0.9, c3=0.87)
    baseline_pos = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=torch.float64,
    )
    edges = torch.tensor([[0, 1, 2], [1, 3, 3]], dtype=torch.int64)
    verdict = _gg3_gate_verdict(
        shape_distance=0.5,
        aggregate_delta_fraction=0.01,
        faithfulness_drop=primary_faithfulness_drop(baseline, morph),
        baseline_result=baseline,
        candidate_result=morph,
        baseline_pos=baseline_pos,
        edges=edges,
    )
    assert verdict.verdict == GG3_VERDICT_BLOCK
    assert verdict.material_tier1_loss
    assert verdict.material_buyback
    assert not verdict.degenerate_escape


def test_joint_gg3_gate_passes_low_buyback_tier1_tradeoff() -> None:
    """Assert material T1 loss needs material two-layout buyback to block.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    baseline = _synthetic_result(tiered=90.0, tier1_only=95.0, c1=0.9, c3=0.95)
    morph = _synthetic_result(tiered=84.8, tier1_only=89.5, c1=0.9, c3=0.87)
    baseline_pos = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=torch.float64,
    )
    edges = torch.tensor([[0, 1, 2], [1, 3, 3]], dtype=torch.int64)
    verdict = _gg3_gate_verdict(
        shape_distance=0.5,
        aggregate_delta_fraction=0.01,
        faithfulness_drop=primary_faithfulness_drop(baseline, morph),
        baseline_result=baseline,
        candidate_result=morph,
        baseline_pos=baseline_pos,
        edges=edges,
    )
    assert verdict.material_tier1_loss
    assert verdict.two_layout_buyback < TWO_LAYOUT_BUYBACK_BAR
    assert verdict.verdict != GG3_VERDICT_BLOCK


def test_joint_gg3_gate_blocks_degenerate_escape_with_g6_floor_breach() -> None:
    """Assert the escape waiver cannot launder severe G6 damage.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    baseline_pos_payload = torch.load(SAVED_GG3_DIR / "weighted_baseline.pt", map_location="cpu")
    baseline = _synthetic_result(
        tiered=80.0,
        tier1_only=84.0,
        c3=0.78,
        c1=0.9,
        g6_weighted_ksm=0.7,
        g6_local_weight_monotonicity=1.0,
    )
    morph = _synthetic_result(
        tiered=80.2,
        tier1_only=83.5,
        c3=0.86,
        c1=0.9,
        g6_weighted_ksm=0.5,
        g6_local_weight_monotonicity=0.9,
    )
    verdict = _gg3_gate_verdict(
        shape_distance=0.9,
        aggregate_delta_fraction=0.0025,
        faithfulness_drop=primary_faithfulness_drop(baseline, morph),
        baseline_result=baseline,
        candidate_result=morph,
        baseline_pos=baseline_pos_payload["positions"],
        edges=baseline_pos_payload["edges"],
    )
    assert verdict.verdict == GG3_VERDICT_BLOCK
    assert verdict.severe_g6_floor_breach
    assert not verdict.degenerate_escape


@pytest.mark.parametrize(
    ("family", "expected"),
    (
        ("dag", 0.60),
        ("clustered", 3.84),
        ("generic_force", 2.97),
        ("weighted", 3.29),
    ),
)
def test_two_layout_buyback_reproduces_saved_gg3_morphs(
    family: str,
    expected: float,
) -> None:
    """Assert saved GG-3 morph buyback matches the pre-registered table.

    Parameters
    ----------
    family : str
        Saved GG-3 morph family.
    expected : float
        Expected buyback from Fable Section 2.

    Returns
    -------
    None
    """
    baseline, morph = _load_official_diag_results(family)
    decomposition = two_layout_buyback_decomposition(baseline, morph)
    assert decomposition["buyback"] == pytest.approx(expected, abs=0.25)


@pytest.mark.parametrize(
    ("family", "minimum_delta"),
    (
        ("dag", 1.5),
        ("clustered", 4.0),
        ("generic_force", 4.0),
        ("weighted", 3.0),
    ),
)
def test_saved_gg3_morphs_rescore_below_baseline_after_phase_a(
    family: str,
    minimum_delta: float,
) -> None:
    """Assert current V3 scoring drops the saved GG-3 morphs as predicted.

    Parameters
    ----------
    family : str
        Saved GG-3 morph family.
    minimum_delta : float
        Minimum baseline-minus-morph tiered drop in score points.

    Returns
    -------
    None
    """
    baseline_pos, morph_pos = _load_official_diag_positions(family)
    probe = probe_by_family(family)
    score_config = ScoreConfig()
    baseline_result = sa_attack._score_probe(probe, baseline_pos, score_config)
    morph_result = sa_attack._score_probe(probe, morph_pos, score_config)
    delta = float(baseline_result.scores["tiered"]) - float(morph_result.scores["tiered"])
    assert delta >= minimum_delta


def test_retargeted_objective_modes_are_wired_and_distinct() -> None:
    """Assert the new objective modes score a fixed candidate differently.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    baseline, morph, baseline_pos, morph_pos, edges = _load_saved_case("weighted")
    shape_distance = procrustes_shape_distance(baseline_pos, morph_pos)
    aggregate_delta = _aggregate_delta_fraction(
        float(morph.scores["tiered"]),
        float(baseline.scores["tiered"]),
    )
    faithfulness_drop = primary_faithfulness_drop(baseline, morph)
    shape_value = _objective(
        shape_distance,
        aggregate_delta,
        faithfulness_drop,
        AttackConfig(objective_mode="shape"),
        baseline_result=baseline,
        candidate_result=morph,
        baseline_pos=baseline_pos,
        edges=edges,
    )
    tier1_value = _objective(
        shape_distance,
        aggregate_delta,
        faithfulness_drop,
        AttackConfig(objective_mode="tier1_loss"),
        baseline_result=baseline,
        candidate_result=morph,
        baseline_pos=baseline_pos,
        edges=edges,
    )
    g6_value = _objective(
        shape_distance,
        aggregate_delta,
        faithfulness_drop,
        AttackConfig(objective_mode="g6_damage"),
        baseline_result=baseline,
        candidate_result=morph,
        baseline_pos=baseline_pos,
        edges=edges,
    )
    assert tier1_value != pytest.approx(shape_value)
    assert g6_value != pytest.approx(shape_value)


def _synthetic_square_probe() -> ProbeFamily:
    """Return a compact square probe for monkeypatched SA candidate tests.

    Parameters
    ----------
    None

    Returns
    -------
    ProbeFamily
        Four-node square probe with non-degenerate baseline geometry.
    """
    return ProbeFamily(
        family="generic_force",
        pos=torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=torch.float64,
        ),
        edges=torch.tensor([[0, 1, 2], [1, 3, 3]], dtype=torch.int64),
        sizes=torch.ones((4, 2), dtype=torch.float64) * 0.1,
        meta={},
    )


def test_tier1_loss_reports_in_band_constrained_best(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assert 5% exploratory candidates cannot suppress the 2% held best.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture.

    Returns
    -------
    None
    """
    probe = _synthetic_square_probe()
    in_band = probe.pos.detach().clone()
    in_band[1, 0] = 2.0
    out_of_band = probe.pos.detach().clone()
    out_of_band[1, 0] = 3.0
    proposals = iter((in_band, out_of_band))

    def fake_propose_positions(
        current: torch.Tensor,
        reference: torch.Tensor,
        rng: object,
        temperature: float,
    ) -> torch.Tensor:
        """Return the next scripted proposal.

        Parameters
        ----------
        current : torch.Tensor
            Current positions with shape ``[N, 2]``.
        reference : torch.Tensor
            Baseline positions with shape ``[N, 2]``.
        rng : object
            Unused deterministic RNG.
        temperature : float
            Unused SA temperature.

        Returns
        -------
        torch.Tensor
            Scripted candidate positions.
        """
        del current, reference, rng, temperature
        return next(proposals)

    def fake_score_probe(
        scored_probe: ProbeFamily,
        positions: torch.Tensor,
        score_config: ScoreConfig,
    ) -> RulerV3Result:
        """Return synthetic scores keyed by the scripted candidate.

        Parameters
        ----------
        scored_probe : ProbeFamily
            Probe being scored.
        positions : torch.Tensor
            Candidate positions with shape ``[N, 2]``.
        score_config : ScoreConfig
            Unused score budget.

        Returns
        -------
        RulerV3Result
            Synthetic V3 result.
        """
        del scored_probe, score_config
        marker = float(positions[1, 0].item())
        if abs(marker - 2.0) < 1.0e-12:
            return _synthetic_result(tiered=98.0, tier1_only=94.0, c1=0.9, c3=0.85)
        if abs(marker - 3.0) < 1.0e-12:
            return _synthetic_result(tiered=97.0, tier1_only=80.0, c1=0.9, c3=0.85)
        return _synthetic_result(tiered=100.0, tier1_only=100.0, c1=0.9, c3=0.95)

    monkeypatch.setattr(sa_attack, "_propose_positions", fake_propose_positions)
    monkeypatch.setattr(sa_attack, "_score_probe", fake_score_probe)
    result = run_family_attack(
        probe,
        seed=1,
        attack_config=AttackConfig(iterations=2, restarts=1, objective_mode="tier1_loss"),
        score_config=ScoreConfig(),
    )
    assert result.aggregate_delta_fraction == pytest.approx(0.02)
    assert result.tier1_only_drop == pytest.approx(6.0)
    assert result.max_blockregion_tier1_only_drop == pytest.approx(6.0)
    assert result.blockregion_aggregate_delta_fraction == pytest.approx(0.02)
    assert result.gate_verdict == GG3_VERDICT_BLOCK
    assert result.blocked


def test_tier1_loss_blockregion_respects_shape_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assert high-drop low-shape candidates are excluded from the deciding max.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture.

    Returns
    -------
    None
    """
    probe = _synthetic_square_probe()
    low_shape = probe.pos.detach().clone()
    low_shape[1, 0] = 1.2

    def fake_propose_positions(
        current: torch.Tensor,
        reference: torch.Tensor,
        rng: object,
        temperature: float,
    ) -> torch.Tensor:
        """Return the scripted low-shape candidate.

        Parameters
        ----------
        current : torch.Tensor
            Current positions with shape ``[N, 2]``.
        reference : torch.Tensor
            Baseline positions with shape ``[N, 2]``.
        rng : object
            Unused deterministic RNG.
        temperature : float
            Unused SA temperature.

        Returns
        -------
        torch.Tensor
            Scripted candidate positions.
        """
        del current, reference, rng, temperature
        return low_shape

    def fake_score_probe(
        scored_probe: ProbeFamily,
        positions: torch.Tensor,
        score_config: ScoreConfig,
    ) -> RulerV3Result:
        """Return a high T1 drop candidate below the shape gate.

        Parameters
        ----------
        scored_probe : ProbeFamily
            Probe being scored.
        positions : torch.Tensor
            Candidate positions with shape ``[N, 2]``.
        score_config : ScoreConfig
            Unused score budget.

        Returns
        -------
        RulerV3Result
            Synthetic V3 result.
        """
        del scored_probe, score_config
        marker = float(positions[1, 0].item())
        if abs(marker - 1.2) < 1.0e-12:
            return _synthetic_result(tiered=99.0, tier1_only=80.0, c1=0.9, c3=0.85)
        return _synthetic_result(tiered=100.0, tier1_only=100.0, c1=0.9, c3=0.95)

    monkeypatch.setattr(sa_attack, "_propose_positions", fake_propose_positions)
    monkeypatch.setattr(sa_attack, "_score_probe", fake_score_probe)
    result = run_family_attack(
        probe,
        seed=1,
        attack_config=AttackConfig(iterations=1, restarts=1, objective_mode="tier1_loss"),
        score_config=ScoreConfig(),
    )
    assert result.best_shape_distance < 0.35
    assert result.tier1_only_drop == pytest.approx(20.0)
    assert result.max_blockregion_tier1_only_drop == pytest.approx(0.0)
    assert result.gate_verdict != GG3_VERDICT_BLOCK
    assert not result.blocked


def test_tier1_loss_bounded_generic_force_retarget_no_longer_blocks() -> None:
    """Assert the fixed generic-force retarget no longer trips the trapdoor.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    probe = next(probe for probe in build_probe_families() if probe.family == "generic_force")
    result = run_family_attack(
        probe,
        seed=43,
        attack_config=AttackConfig(iterations=250, restarts=2, objective_mode="tier1_loss"),
        score_config=ScoreConfig(),
    )
    assert result.two_layout_buyback < TWO_LAYOUT_BUYBACK_BAR
    assert result.gate_verdict != GG3_VERDICT_BLOCK
    assert not result.blocked
